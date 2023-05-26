
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
        self.e_corrs = []
        self.nroots = 1
        self.frozen = frozen
        self.verbose = self.mol.verbose
        self.mp_order = mp_order
        self.stdout = self.mol.stdout

    @property
    def e_tot(self):
        return np.asarray(self.e_corr) + self._scf.e_tot
    
    def _get_random_mps(self, hamil, name, ref=False):
        if ref:
            info = CASCIMPSInfo(hamil.n_sites, hamil.vacuum, self.target, hamil.basis, 1, 0, 1)
            info.set_bond_dimension(1)
        else:
            info = MPSInfo(hamil.n_sites, hamil.vacuum, self.target, hamil.basis)
            info.set_bond_dimension(2000)
        info.tag = name
        mps = MPS(hamil.n_sites, 0, 2)
        mps.initialize(info)
        mps.random_canonicalize()
        mps.tensors[0].normalize()
        mps.save_mutable()
        info.save_mutable()
        if mps.center == 0 and mps.dot == 2:
            mps.move_left(hamil.opf.cg, None)
        mps.center = 0
        return mps
    
    def _mps_addition(self, hamil, name, mpoa, mpsa, mpob, mpsb):

        fmps = self._get_random_mps(hamil, name)

        lme = MovingEnvironment(mpoa, fmps, mpsa, "LME")
        lme.init_environments(False)
        rme = MovingEnvironment(mpob, fmps, mpsb, "RME")
        rme.init_environments(False)
        linear = Linear(None, lme, rme, VectorUBond([2000]), VectorUBond([2000]), VectorDouble([0]))
        linear.eq_type = EquationTypes.FitAddition
        linear.cutoff = 0
        linear.iprint = max(min(self.verbose - 4, 3), 0)
        linear.linear_conv_thrds = VectorDouble([1E-20])
        linear.solve(1, True, 0)

        return fmps
    
    def _build_hamiltonian(self, fcidump, ci_order, no_trans=False):

        big_left_orig = CSFBigSite(self.n_inactive, ci_order, False,
            fcidump, VectorUInt8([0] * self.n_inactive), max(min(self.verbose - 4, 3), 0))
        big_right_orig = CSFBigSite(self.n_external, ci_order, True,
            fcidump, VectorUInt8([0] * self.n_external), max(min(self.verbose - 4, 3), 0))
        big_left = SimplifiedBigSite(big_left_orig,
            NoTransposeRule(RuleQC()) if no_trans else RuleQC())
        big_right = SimplifiedBigSite(big_right_orig,
            NoTransposeRule(RuleQC()) if no_trans else RuleQC())
        hamil = HamiltonianQCBigSite(self.vacuum, self.n_orbs,
            VectorUInt8([0] * self.n_orbs), fcidump, big_left, big_right)
        self.n_sites = hamil.n_sites
        return hamil
    
    def _solve_linear(self, lmpo, rmpo, bra, ket):

        lme = MovingEnvironment(lmpo, bra, bra, "LME")
        lme.init_environments(False)
        lme.delayed_contraction = OpNamesSet.normal_ops()
        lme.cached_contraction = False
        rme = MovingEnvironment(rmpo, bra, ket, "RME")
        rme.init_environments(False)
        linear = Linear(lme, rme, None, VectorUBond([2000]), VectorUBond([2000]), VectorDouble([0]))
        linear.cutoff = 0
        linear.iprint = max(min(self.verbose - 4, 3), 0)
        linear.linear_conv_thrds = VectorDouble([1E-20])
        return linear.solve(1, True, 0)
    
    def _expectation(self, mpo, bra, ket):

        ime = MovingEnvironment(mpo, bra, ket, "IME")
        ime.init_environments(False)
        return Expect(ime, 2000, 2000).solve(False)

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
        h1e = h1e.ravel()
        g2e = ao2mo.restore(8, ao2mo.kernel(self.mol, self.mo_coeff), self.mol.nao)
        h1e[np.abs(h1e) < fcidump_tol] = 0
        g2e[np.abs(g2e) < fcidump_tol] = 0
        na, nb = self.mol.nelec
        fcidump = FCIDUMP()
        fcidump.initialize_su2(self.mol.nao, na + nb, abs(na - nb), 1, e_core, h1e, g2e)
        error = fcidump.symmetrize(VectorUInt8([0] * self.mol.nao))
        if self.verbose >= 5:
            print('symm error = ', error)

        self.n_orbs = self.mol.nao
        ci_order = 2

        # Hamiltonian
        self.n_inactive = len(self.mo_occ[self.mo_occ > 1])
        self.n_external = len(self.mo_occ[self.mo_occ <= 1])
        assert self.n_inactive + self.n_external == self.n_orbs

        self.vacuum = SU2(0)
        self.target = SU2(na + nb, abs(na - nb), 0)

        hamil = self._build_hamiltonian(fcidump, ci_order)

        self.hamil = hamil

        ket0 = self._get_random_mps(hamil, "KET0", True)

        # MPO
        mpo = MPOQC(hamil, QCTypes.NC)
        mpo = SimplifiedMPO(mpo, RuleQC(), True)

        # DMRG
        me = MovingEnvironment(mpo, ket0, ket0, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(False)
        dmrg = DMRG(me, VectorUBond([2000]), VectorDouble([0]))
        dmrg.davidson_conv_thrds = VectorDouble([1E-20])
        dmrg.cutoff = 0
        dmrg.iprint = max(min(self.verbose - 4, 3), 0)
        ener = dmrg.solve(1, True, 0.0)

        self.converged = True
        self.e_corr = ener - self._scf.e_tot
        self.e_corrs.append(ener - self._scf.e_tot)

        lib.logger.note(self, 'E(MP1) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        self.mps = ket0

        if self.mp_order == 1:
            return self.e_corr, ket0

        # FCIDUMP
        dm1 = self.make_rdm1(ket0).copy(order='C')
        fd_dyall = DyallFCIDUMP(fcidump, self.n_inactive, self.n_external)
        fd_dyall.initialize_from_1pdm_su2(dm1)
        error = fd_dyall.symmetrize(VectorUInt8([0] * self.mol.nao))
        if self.verbose >= 5:
            print('symm error = ', error)
        
        hm_dyall = self._build_hamiltonian(fd_dyall, ci_order, False)
        hamil = self._build_hamiltonian(fcidump, ci_order, True)

        # Left MPO
        lmpo = MPOQC(hm_dyall, QCTypes.NC)
        lmpo = SimplifiedMPO(lmpo, RuleQC(), True)
        lmpo.const_e -= self.e_tot
        lmpo = lmpo * -1

        # Right MPO
        rmpo = MPOQC(hamil, QCTypes.NC)
        rmpo = SimplifiedMPO(rmpo, NoTransposeRule(RuleQC()), True)
        rmpo.const_e -= self.e_tot

        # Identity MPO
        impo = IdentityMPO(hamil.basis, hamil.basis, hamil.vacuum, hamil.opf)
        impo = SimplifiedMPO(impo, Rule())

        bra = self._get_random_mps(hamil, "BRA")
        self.e_corr += self._solve_linear(lmpo, rmpo, bra, ket0)
        self.e_corrs.append(self.e_corr - self.e_corrs[-1])

        lib.logger.note(self, 'E(MP2) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        dp01 = self._expectation(impo, bra, ket0)
        ket1 = self._mps_addition(hamil, "KET1", impo, bra, (-dp01) * impo, ket0)
        dp11 = self._expectation(impo, ket1, ket1)

        # fac = 1 / np.sqrt(dp11 + 1), 1 / np.sqrt(dp11 + 1)
        fac = np.sqrt(1 - dp11), 1
        # mps1 = fac[0] * ket0 + fac[1] * ket1
        mps1 = self._mps_addition(hamil, "MPS1", fac[0] * impo, ket0, fac[1] * impo, ket1)

        self.mps = mps1

        if self.mp_order == 2:
            return self.e_corr, self.mps

        hex1 = self._expectation(rmpo, ket1, ket1)
        h0ex1 = -self._expectation(lmpo, ket1, ket1)

        self.e_corr += hex1 - h0ex1 - self.e_corrs[0] * dp11
        self.e_corrs.append(self.e_corr - self.e_corrs[-1])

        lib.logger.note(self, 'E(MP3) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        if self.mp_order == 3:
            return self.e_corr, self.mps

        ci_order = 4

        hm_dyall = self._build_hamiltonian(fd_dyall, ci_order, True)
        hamil = self._build_hamiltonian(fcidump, ci_order, True)

        # Left MPO
        lmpo = MPOQC(hm_dyall, QCTypes.NC)
        lmpo = SimplifiedMPO(lmpo, NoTransposeRule(RuleQC()), True)
        lmpo.const_e -= self.e_corrs[0] + self._scf.e_tot
        lmpo = lmpo * -1

        # Right MPO
        rmpo = MPOQC(hamil, QCTypes.NC)
        rmpo = SimplifiedMPO(rmpo, NoTransposeRule(RuleQC()), True)
        rmpo.const_e -= self._scf.e_tot

        # Identity MPO
        impo = IdentityMPO(hamil.basis, hamil.basis, hamil.vacuum, hamil.opf)
        impo = SimplifiedMPO(impo, Rule())

        bra1 = self._mps_addition(hamil, "BRA1", lmpo, ket1, rmpo, ket1)
        dp10 = self._expectation(impo, bra1, ket0)

        bra2 = self._mps_addition(hamil, "BRA2", impo, bra1, (-dp10) * impo, ket0)

        # Left MPO
        lmpo = MPOQC(hm_dyall, QCTypes.NC)
        lmpo = SimplifiedMPO(lmpo, RuleQC(), True)
        lmpo.const_e -= self._scf.e_tot
        lmpo = lmpo * -1

        bra = self._get_random_mps(hamil, "BRA")
        self.e_corr += self._solve_linear(lmpo, impo, bra, bra2) - self.e_corrs[1] * dp11
        self.e_corrs.append(self.e_corr - self.e_corrs[-1])

        lib.logger.note(self, 'E(MP4) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        self.mps = bra

        if self.mp_order == 4:
            return self.e_corr, self.mps
        
        dp02 = self._expectation(impo, bra, ket0)
        ket2 = self._mps_addition(hamil, "KET2", impo, bra, (-dp02) * impo, ket0)
        dp21 = self._expectation(impo, ket2, ket1)
        dp22 = self._expectation(impo, ket2, ket2)
        hex2 = self._expectation(rmpo, ket2, ket2)
        h0ex2 = -self._expectation(lmpo, ket2, ket2)

        self.e_corr += hex2 - h0ex2 - self.e_corrs[0] * dp22 - \
            2 * self.e_corrs[1] * dp21 - self.e_corrs[2] * dp11 
        self.e_corrs.append(self.e_corr - self.e_corrs[-1])

        lib.logger.note(self, 'E(MP5) = %.16g  E_corr = %.16g', self.e_tot, self.e_corr)

        self.mps = ket2

        if self.mp_order == 5:
            return self.e_corr, self.mps
    
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

def MP3(*args, **kwargs):
    kwargs['mp_order'] = 3
    return MP(*args, **kwargs)

def MP4(*args, **kwargs):
    kwargs['mp_order'] = 4
    return MP(*args, **kwargs)

def MP5(*args, **kwargs):
    kwargs['mp_order'] = 5
    return MP(*args, **kwargs)

if __name__ == '__main__':

    from pyscf import gto, scf, ci, mp

    mol = gto.M(
        atom='''
            O  0.000000  0.000000  0.000000
            H  0.758602  0.000000  0.504284
            H  0.758602  0.000000 -0.504284
        ''',
        basis='6-31g',
        verbose=3, symmetry=False, spin=0)

    mf = scf.RHF(mol).set(conv_tol=1E-12).run()
    # print(mf.scf_summary['e1'])
    # print(mf.scf_summary['e2'])
    # print(mf.mo_energy[:5].sum() * 2 + mf.mol.energy_nuc())
    mymp = MP2(mf).run()
    dm1 = mymp.make_rdm1()
    mymp = mp.MP2(mf).run()
    dm1x = mymp.make_rdm1()
    h1e = mymp.mo_coeff.T @ mymp._scf.get_hcore() @ mymp.mo_coeff
    print(np.einsum('pq,pq->', h1e, dm1))
    print(np.einsum('pq,pq->', h1e, dm1x))
    print(np.linalg.norm(dm1 - dm1x))
    quit()
    print(np.diag(dm1))
    print(np.diag(dm1x))
    from pyscf.mp.dfmp2_native import DFMP2
    mymp = DFMP2(mf).run()
    dm1yy = mymp.make_rdm1()
    dm1y = mymp.make_rdm1_unrelaxed()
    dm1z = mymp.make_rdm1_relaxed()
    print(np.linalg.norm(dm1x - dm1y))
    print(np.linalg.norm(dm1x - dm1z))
    print(np.diag(dm1y))
    print(np.diag(dm1z))
    # for xxx in dm1:
    #     print(xxx)
    # quit()
    # print(np.linalg.norm(dm1 - dm1x))
    # print(np.linalg.norm(dm1 - dm1yy))
    # print(np.linalg.norm(dm1 - dm1y))
    # print(np.diag(dm1z))
    # print(np.linalg.norm(dm1 - dm1z))
