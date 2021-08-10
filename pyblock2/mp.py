
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

from block2 import *
from block2.su2 import *
from pyscf import lib, ao2mo
import numpy as np

class MP(lib.StreamObject):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mp_order=2):
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        self.mol = mf.mol
        self._scf = mf
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.nroots = 1
        self.frozen = frozen
        self.verbose = self.mol.verbose
        self.mp_order = mp_order
        self.stdout = self.mol.stdout

    @property
    def e_tot(self):
        return np.asarray(self.e_corr) + self._scf.e_tot

    def kernel(self):

        # Global
        Random.rand_seed(123456)
        scratch = './nodex'
        n_threads = lib.num_threads()
        fcidump_tol = 1E-13
        memory = lib.param.MAX_MEMORY * 1E6
        init_memory(isize=int(memory * 0.05),
            dsize=int(memory * 0.95), save_dir=scratch)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
            n_threads, n_threads, 1)
        Global.threading.seq_type = SeqTypes.Nothing
        Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
        Global.frame.load_buffering = False
        Global.frame.save_buffering = False
        Global.frame.use_main_stack = False
        Global.frame.minimal_disk_usage = True
        if self.verbose >= 5:
            print(Global.frame)
            print(Global.threading)
        
        # FCIDUMP
        h1e = self.mo_coeff.T @ self._scf.get_hcore() @ self.mo_coeff
        e_core = self.mol.energy_nuc()
        h1e = h1e.flatten()
        g2e = ao2mo.restore(8, ao2mo.kernel(self.mol, self.mo_coeff), self.mol.nao)
        h1e[np.abs(h1e) < fcidump_tol] = 0
        g2e[np.abs(g2e) < fcidump_tol] = 0
        na, nb = self.mol.nelec
        fcidump = FCIDUMP()
        fcidump.initialize_su2(self.mol.nao, na + nb, abs(na - nb), 1, e_core, h1e, g2e)
        error = fcidump.symmetrize(VectorUInt8([0] * self.mol.nao))
        if self.verbose >= 5:
            print('symm error = ', error)
        
        n_orbs = self.mol.nao
        self.n_orbs = self.mol.nao
        
        # Hamiltonian
        n_inactive = len(self.mo_occ[self.mo_occ > 1])
        n_external = len(self.mo_occ[self.mo_occ <= 1])
        assert n_inactive + n_external == n_orbs
        big_left_orig = CSFBigSite(n_inactive, 2, False,
            fcidump, VectorUInt8([0] * n_inactive), max(min(self.verbose - 4, 3), 0))
        big_right_orig = CSFBigSite(n_external, 2, True,
            fcidump, VectorUInt8([0] * n_external), max(min(self.verbose - 4, 3), 0))
        big_left = SimplifiedBigSite(big_left_orig, RuleQC())
        big_right = SimplifiedBigSite(big_right_orig, RuleQC())
        vacuum = SU2(0)
        target = SU2(na + nb, abs(na - nb), 0)
        hamil = HamiltonianQCBigSite(vacuum, n_orbs, VectorUInt8([0] * n_orbs), fcidump,
            big_left, big_right)
        n_sites = hamil.n_sites

        self.hamil = hamil

        # MPS
        info = CASCIMPSInfo(n_sites, vacuum, target, hamil.basis, 1, 0, 1)
        info.set_bond_dimension(1)
        mps = MPS(n_sites, 0, 2)
        mps.initialize(info)
        mps.random_canonicalize()
        mps.tensors[mps.center].normalize()
        mps.save_mutable()
        info.save_mutable()

        # MPO
        mpo = MPOQC(hamil, QCTypes.NC)
        mpo = SimplifiedMPO(mpo, RuleQC(), True)

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(False)
        dmrg = DMRG(me, VectorUBond([2000]), VectorDouble([0]))
        dmrg.davidson_conv_thrds = VectorDouble([1E-12])
        dmrg.cutoff = 0
        dmrg.iprint = max(min(self.verbose - 4, 3), 0)
        ener = dmrg.solve(1, True, 0.0)

        self.converged = True
        self.e_corr = ener - self._scf.e_tot

        lib.logger.note(self, 'E(MP1) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        self.mps = mps

        if self.mp_order == 1:
            return self.e_corr, mps

        # FCIDUMP
        dm1 = self.make_rdm1(mps).copy(order='C')
        fd_dyall = DyallFCIDUMP(fcidump, n_inactive, n_external)
        fd_dyall.initialize_from_1pdm_su2(dm1)
        error = fd_dyall.symmetrize(VectorUInt8([0] * self.mol.nao))
        if self.verbose >= 5:
            print('symm error = ', error)
        
        # Hamiltonian
        big_left = SimplifiedBigSite(big_left_orig, NoTransposeRule(RuleQC()))
        big_right = SimplifiedBigSite(big_right_orig, NoTransposeRule(RuleQC()))
        hamil = HamiltonianQCBigSite(vacuum, n_orbs, VectorUInt8([0] * n_orbs), fcidump,
            big_left, big_right)
        big_left_orig = CSFBigSite(n_inactive, 2, False,
            fd_dyall, VectorUInt8([0] * n_inactive), max(min(self.verbose - 4, 3), 0))
        big_right_orig = CSFBigSite(n_external, 2, True,
            fd_dyall, VectorUInt8([0] * n_external), max(min(self.verbose - 4, 3), 0))
        big_left = SimplifiedBigSite(big_left_orig, RuleQC())
        big_right = SimplifiedBigSite(big_right_orig, RuleQC())
        hm_dyall = HamiltonianQCBigSite(vacuum, n_orbs, VectorUInt8([0] * n_orbs), fd_dyall,
            big_left, big_right)

        # Left MPO
        lmpo = MPOQC(hm_dyall, QCTypes.NC)
        lmpo = SimplifiedMPO(lmpo, RuleQC(), True)
        lmpo.const_e -= self.e_tot
        lmpo = lmpo * -1

        # Right MPO
        rmpo = MPOQC(hamil, QCTypes.NC)
        rmpo = SimplifiedMPO(rmpo, NoTransposeRule(RuleQC()), True)
        rmpo.const_e -= self.e_tot

        # MPS
        mps.dot = 2
        if mps.center == mps.n_sites - 1 and mps.dot == 2:
            mps.center = mps.n_sites - 2
        bra_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
        bra_info.tag = 'BRA'
        bra_info.set_bond_dimension(2000)
        bra = MPS(n_sites, mps.center, 2)
        bra.initialize(bra_info)
        bra.random_canonicalize()
        bra.tensors[bra.center].normalize()
        bra.save_mutable()
        bra_info.save_mutable()
        if bra.center == 0 and bra.dot == 2:
            bra.move_left(hamil.opf.cg, None)
        elif bra.center == bra.n_sites - 2 and bra.dot == 2:
            bra.move_right(hamil.opf.cg, None)
        bra.center = mps.center

        # Linear
        lme = MovingEnvironment(lmpo, bra, bra, "LME")
        lme.init_environments(False)
        lme.delayed_contraction = OpNamesSet.normal_ops()
        lme.cached_contraction = False
        rme = MovingEnvironment(rmpo, bra, mps, "RME")
        rme.init_environments(False)
        linear = Linear(lme, rme, None, VectorUBond([2000]), VectorUBond([2000]), VectorDouble([0]))
        linear.cutoff = 0
        linear.iprint = max(min(self.verbose - 4, 3), 0)
        linear.minres_conv_thrds = VectorDouble([1E-13])

        self.e_corr = linear.solve(1, mps.center == 0, 0)

        lib.logger.note(self, 'E(MP2) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        impo = IdentityMPO(hamil.basis, hamil.basis, hamil.vacuum, hamil.opf)
        impo = SimplifiedMPO(impo, Rule())

        ime = MovingEnvironment(impo, bra, mps, "IME")
        ime.init_environments(False)

        ovl = Expect(ime, 2000, 2000).solve(False)
        ilmpo = 1.0 * impo
        irmpo = (1 - ovl) * impo

        # MPS
        fmps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
        fmps_info.tag = 'FMPS'
        fmps_info.set_bond_dimension(2000)
        fmps = MPS(n_sites, mps.center, 2)
        fmps.initialize(fmps_info)
        fmps.random_canonicalize()
        fmps.tensors[fmps.center].normalize()
        fmps.save_mutable()
        fmps_info.save_mutable()
        if fmps.center == 0 and fmps.dot == 2:
            fmps.move_left(hamil.opf.cg, None)
        elif fmps.center == fmps.n_sites - 2 and fmps.dot == 2:
            fmps.move_right(hamil.opf.cg, None)
        fmps.center = mps.center

        lme = MovingEnvironment(ilmpo, fmps, bra, "LME")
        lme.init_environments(False)
        rme = MovingEnvironment(irmpo, fmps, mps, "RME")
        rme.init_environments(False)
        linear = Linear(None, lme, rme, VectorUBond([2000]), VectorUBond([2000]), VectorDouble([0]))
        linear.eq_type = EquationTypes.FitAddition
        linear.cutoff = 0
        linear.iprint = max(min(self.verbose - 4, 3), 0)
        linear.minres_conv_thrds = VectorDouble([1E-13])

        linear.solve(1, fmps.center == 0, 0)
        
        fmps.load_mutable()
        fmps.tensors[fmps.center].normalize()
        fmps.save_mutable()

        self.mps = fmps

        if self.mp_order == 2:
            return self.e_corr, fmps
    
    def make_rdm1(self, state=None, norb=None, nelec=None):
        '''
        Spin-traced one-particle density matrix in MO basis.
        dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>
        '''

        if state is None:
            state = self.mps
        
        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil, 0)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        # 1PDM
        pme = MovingEnvironment(pmpo, state, state, "1PDM")
        pme.init_environments(False)
        expect = Expect(pme, state.info.bond_dim, state.info.bond_dim)
        expect.iprint = max(min(self.verbose - 4, 3), 0)
        expect.solve(True, state.center == 0)
        dmr = expect.get_1pdm_spatial(self.n_orbs)
        dm = np.array(dmr).copy()
        dmr.deallocate()

        return dm.transpose((1, 0))

    def make_rdm2(self, state=None, norb=None, nelec=None):
        '''
        Spin-traced two-particle density matrix in MO basis
        dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>
        '''

        if state is None:
            state = self.mps
        
        # 2PDM MPO
        pmpo = PDM2MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        # 2PDM
        pme = MovingEnvironment(pmpo, state, state, "2PDM")
        pme.init_environments(False)
        expect = Expect(pme, state.info.bond_dim, state.info.bond_dim)
        expect.iprint = max(min(self.verbose - 4, 3), 0)
        expect.solve(True, state.center == 0)
        dmr = expect.get_2pdm_spatial(self.n_orbs)
        dm = np.array(dmr, copy=True)

        return dm.transpose((0, 3, 1, 2))

def MP2(*args, **kwargs):
    kwargs['mp_order'] = 2
    return MP(*args, **kwargs)

if __name__ == '__main__':

    from pyscf import gto, scf, ci, mp

    mol = gto.M(
        atom='''
            O  0.000000  0.000000  0.000000
            H  0.758602  0.000000  0.504284
            H  0.758602  0.000000 -0.504284
        ''',
        basis='ccpvdz',
        verbose=3, symmetry=False, spin=0)

    mf = scf.RHF(mol).run()
    mymp = MP2(mf).run()
    # dm1 = mymp.make_rdm1()
    # print(np.diag(dm1))
    mymp = mp.MP2(mf).run()
    # dm1x = mymp.make_rdm1()
    # print(np.diag(dm1x))
    # print(np.linalg.norm(dm1 - dm1x))
