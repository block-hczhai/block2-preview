
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

from pyscf import lib, ao2mo, gto, mcscf, tools, symm
import numpy as np

try:
    from pyblock2.dmrgscf import DMRGCI as _DMRGCI
except ImportError:
    from dmrgscf import DMRGCI as _DMRGCI

class MRPT(lib.StreamObject):

    def __init__(self, mc, mp_order=2, ci_order=2, theory='nevpt', spin_adapted=True):
        self._casci = mc
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.e_corr = None
        self.mp_order = mp_order
        self.ci_order = ci_order
        self.dmrg_args = {
            "startM": 250, "maxM": 1800, "schedule": "default",
            "sweep_tol": 1E-8, "cutoff": 1E-16,
            "memory": lib.param.MAX_MEMORY * 1E6
        }
        self.theory = theory
        self.scratch = './nodex'
        self.spin_adapted = spin_adapted

    @property
    def e_tot(self):
        return np.asarray(self.e_corr) + self._casci.e_tot

    def kernel(self, mo_coeff=None, write_fd=None, write_conf=None):

        if mo_coeff is None:
            mo_coeff = self._casci.mo_coeff

        from block2 import Random, Global, Threading, FCIDUMP, VectorUInt8, init_memory, VectorInt
        from block2 import ThreadingTypes, SeqTypes, DoubleFPCodec, QCTypes, OpNamesSet, OpNames
        from block2 import SU2, SZ, VectorUBond, VectorDouble, NoiseTypes, DyallFCIDUMP, FinkFCIDUMP
        if self.spin_adapted:
            SX = SU2
            from block2.su2 import CSFBigSite, SimplifiedBigSite, RuleQC, HamiltonianQCBigSite
            from block2.su2 import CASCIMPSInfo, MPOQC, SimplifiedMPO, IdentityAddedMPO
            from block2.su2 import MovingEnvironment, DMRGBigSite, NoTransposeRule, MPSInfo, MPS
            from block2.su2 import LinearBigSite
        else:
            SX = SZ
            from block2.sz import SCIFockBigSite, SimplifiedBigSite, RuleQC, HamiltonianQCBigSite
            from block2.sz import CASCIMPSInfo, MPOQC, SimplifiedMPO, IdentityAddedMPO
            from block2.sz import MovingEnvironment, DMRGBigSite, NoTransposeRule, MPSInfo, MPS
            from block2.sz import LinearBigSite

        # Global
        Random.rand_seed(123456)
        scratch = self.scratch
        n_threads = lib.num_threads()
        fcidump_tol = 1E-13
        memory = self.dmrg_args["memory"]
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
        
        mc = self._casci
        
        if self.mol.groupname.lower() == 'c1':
            orb_sym = [0] * self.mol.nao
            orb_sym_fd = [1] * self.mol.nao
        else:
            from pyscf import symm, tools
            orb_sym_str = symm.label_orb_symm(self.mol, self.mol.irrep_name, self.mol.symm_orb, mo_coeff)
            orb_sym = [symm.irrep_name2id(self.mol.groupname, i) for i in orb_sym_str]
            orb_sym_fd = [tools.fcidump.ORBSYM_MAP[self.mol.groupname][i] for i in orb_sym]

        # FCIDUMP
        h1e = mo_coeff.T @ self._scf.get_hcore() @ mo_coeff
        e_core = self.mol.energy_nuc()
        h1e = h1e.flatten()
        g2e = ao2mo.restore(8, ao2mo.kernel(self.mol, mo_coeff), self.mol.nao)
        h1e[np.abs(h1e) < fcidump_tol] = 0
        g2e[np.abs(g2e) < fcidump_tol] = 0
        na, nb = self.mol.nelec
        fcidump = FCIDUMP()
        fcidump.initialize_su2(self.mol.nao, na + nb, abs(na - nb), 1, e_core, h1e, g2e)
        fcidump.orb_sym = VectorUInt8(orb_sym_fd)
        error = fcidump.symmetrize(VectorUInt8(orb_sym))
        if self.verbose >= 5:
            print('symm error = ', error)

        n_orbs = self.mol.nao
        self.n_orbs = self.mol.nao

        # Hamiltonian
        n_inactive = mc.ncore
        n_external = self.n_orbs - mc.ncas - mc.ncore
        assert n_inactive + n_external <= n_orbs

        if write_fd is not None:
            fcidump.write(write_fd)
        
        if write_conf is not None:
            with open(write_conf, "w") as conf:
                conf.write("""
                    sym %s
                    orbitals %s

                    schedule default
                    maxM %d
                    maxiter %d

                    %s %d %d %d
                    noreorder
                """ % (
                    self.mol.groupname.lower(),
                    write_fd if write_fd is not None else "FCIDUMP",
                    self.dmrg_args["maxM"],
                    self.dmrg_args["maxiter"],
                    "nevpt2sd" if self.theory == "nevpt" else "mrrept2sd",
                    n_inactive, n_orbs - n_inactive - n_external, n_external
                ))
            self.e_corr = 0.0
            return None

        if self.spin_adapted:
            big_left_orig = CSFBigSite(n_inactive, self.ci_order, False,
                fcidump, VectorUInt8(orb_sym[:n_inactive]), max(min(self.verbose - 4, 3), 0))
            big_right_orig = CSFBigSite(n_external, self.ci_order, True,
                fcidump, VectorUInt8(orb_sym[-n_external:]), max(min(self.verbose - 4, 3), 0))
        else:
            poccl = SCIFockBigSite.ras_space(False, n_inactive, *[self.ci_order] * 3, VectorInt([]))
            poccr = SCIFockBigSite.ras_space(True, n_external, *[self.ci_order] * 3, VectorInt([]))
            big_left_orig = SCIFockBigSite(n_orbs, n_inactive, False, fcidump, VectorUInt8(orb_sym), poccl, self.verbose > 4)
            big_right_orig = SCIFockBigSite(n_orbs, n_external, True, fcidump, VectorUInt8(orb_sym), poccr, self.verbose > 4)
        big_left = SimplifiedBigSite(big_left_orig, RuleQC())
        big_right = SimplifiedBigSite(big_right_orig, RuleQC())
        vacuum = SX(0)
        target = SX(na + nb, abs(na - nb), 0)
        hamil = HamiltonianQCBigSite(vacuum, n_orbs, VectorUInt8(orb_sym), fcidump,
            big_left, big_right)
        n_sites = hamil.n_sites

        self.hamil = hamil

        if self.dmrg_args["schedule"] == "default":
            _DMRGCI.get_schedule(self.dmrg_args)

        bond_dims, dav_thrds, noises = self.dmrg_args["schedule"]
        sweep_tol = self.dmrg_args["sweep_tol"]

        # MPS
        info = CASCIMPSInfo(n_sites, vacuum, target, hamil.basis,
            1 if n_inactive != 0 else 0, mc.ncas, 1 if n_external != 0 else 0)
        info.set_bond_dimension(bond_dims[0])
        mps = MPS(n_sites, 0, 2)
        mps.initialize(info)
        mps.random_canonicalize()
        mps.tensors[mps.center].normalize()
        mps.save_mutable()
        info.save_mutable()
        forward = mps.center == 0

        # MPO
        mpo = MPOQC(hamil, QCTypes.NC)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        mpo = IdentityAddedMPO(mpo)

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(self.verbose >= 5)
        dmrg = DMRGBigSite(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.last_site_svd = True
        dmrg.last_site_1site = True
        dmrg.decomp_last_site = False
        dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
        dmrg.cutoff = self.dmrg_args["cutoff"]
        dmrg.iprint = max(min(self.verbose - 4, 3), 0)
        dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
        ener = dmrg.solve(len(bond_dims), forward, sweep_tol)

        self.converged = True
        self.e_corr = 0

        lib.logger.note(self, 'E(%s1) = %.16g  E_corr = %.16g', self.theory.upper(), self.e_tot, self.e_corr)

        self.mps = mps

        if self.mp_order == 1:
            return self.e_corr, mps

        # FCIDUMP
        if self.theory == 'nevpt':
            dm1 = self.make_rdm1(mps).copy(order='C')
            fd_zero = DyallFCIDUMP(fcidump, n_inactive, n_external)
            fd_zero.initialize_from_1pdm_su2(dm1)
        else:
            fd_zero = FinkFCIDUMP(fcidump, n_inactive, n_external)
        error = fd_zero.symmetrize(VectorUInt8(orb_sym))
        if self.verbose >= 5:
            print('symm error = ', error)
        
        # Hamiltonian
        big_left = SimplifiedBigSite(big_left_orig, NoTransposeRule(RuleQC()))
        big_right = SimplifiedBigSite(big_right_orig, NoTransposeRule(RuleQC()))
        hamil = HamiltonianQCBigSite(vacuum, n_orbs, VectorUInt8(orb_sym), fcidump,
            big_left, big_right)
        if self.spin_adapted:
            big_left_orig = CSFBigSite(n_inactive, self.ci_order, False,
                fd_zero, VectorUInt8(orb_sym[:n_inactive]), max(min(self.verbose - 4, 3), 0))
            big_right_orig = CSFBigSite(n_external, self.ci_order, True,
                fd_zero, VectorUInt8(orb_sym[-n_external:]), max(min(self.verbose - 4, 3), 0))
        else:
            poccl = SCIFockBigSite.ras_space(False, n_inactive, *[self.ci_order] * 3, VectorInt([]))
            poccr = SCIFockBigSite.ras_space(True, n_external, *[self.ci_order] * 3, VectorInt([]))
            big_left_orig = SCIFockBigSite(n_orbs, n_inactive, False, fd_zero, VectorUInt8(orb_sym), poccl, self.verbose > 4)
            big_right_orig = SCIFockBigSite(n_orbs, n_external, True, fd_zero, VectorUInt8(orb_sym), poccr, self.verbose > 4)
        big_left = SimplifiedBigSite(big_left_orig, RuleQC())
        big_right = SimplifiedBigSite(big_right_orig, RuleQC())
        hm_zero = HamiltonianQCBigSite(vacuum, n_orbs, VectorUInt8(orb_sym), fd_zero,
            big_left, big_right)

        # Left MPO
        lmpo = MPOQC(hm_zero, QCTypes.NC)
        lmpo = SimplifiedMPO(lmpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        lmpo.const_e -= self.e_tot
        lmpo = lmpo * -1
        lmpo = IdentityAddedMPO(lmpo)

        # Right MPO
        rmpo = MPOQC(hamil, QCTypes.NC)
        rmpo = SimplifiedMPO(rmpo, NoTransposeRule(RuleQC()),
            True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        rmpo.const_e -= self.e_tot
        rmpo = IdentityAddedMPO(rmpo)

        # MPS
        mps.dot = 2
        if mps.center == mps.n_sites - 1 and mps.dot == 2:
            mps.center = mps.n_sites - 2
        bra_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
        bra_info.tag = 'BRA'
        bra_info.set_bond_dimension(bond_dims[0])
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
        lme.init_environments(self.verbose >= 5)
        lme.delayed_contraction = OpNamesSet.normal_ops()
        lme.cached_contraction = False
        rme = MovingEnvironment(rmpo, bra, mps, "RME")
        rme.init_environments(self.verbose >= 5)
        linear = LinearBigSite(lme, rme, None, VectorUBond(bond_dims),
            VectorUBond([mps.info.bond_dim + 400]), VectorDouble(noises))
        linear.last_site_svd = True
        linear.last_site_1site = True
        linear.decomp_last_site = False
        linear.cutoff = self.dmrg_args["cutoff"]
        linear.iprint = max(min(self.verbose - 4, 3), 0)
        linear.linear_conv_thrds = VectorDouble([x / 50 for x in dav_thrds])

        self.e_corr = linear.solve(len(bond_dims), mps.center == 0, sweep_tol)

        lib.logger.note(self, 'E(%s2) = %.16g  E_corr = %.16g', self.theory.upper(), self.e_tot, self.e_corr)

        self.mps = bra

        if self.mp_order == 2:
            return self.e_corr, bra

    def make_rdm1(self, state=None, norb=None, nelec=None):
        '''
        Spin-traced one-particle density matrix in MO basis.
        dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>
        '''

        from block2 import CG
        if self.spin_adapted:
            from block2.su2 import PDM1MPOQC, RuleQC, Expect, SimplifiedMPO, MovingEnvironment
        else:
            from block2.sz import PDM1MPOQC, RuleQC, Expect, SimplifiedMPO, MovingEnvironment

        if state is None:
            state = self.mps
            state.dot = 1
        
        if state.center == state.n_sites - 1 and state.canonical_form[state.center] == 'K':
            state.move_left(CG(), None)
            state.center = state.n_sites - 2
        
        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil, 0)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        # 1PDM
        pme = MovingEnvironment(pmpo, state, state, "1PDM")
        pme.init_environments(self.verbose >= 5)

        expect = Expect(pme, state.info.bond_dim, state.info.bond_dim)
        expect.iprint = max(min(self.verbose - 4, 3), 0)
        expect.solve(True, state.center == 0)
        dmr = expect.get_1pdm_spatial(self.n_orbs)
        dm = np.array(dmr).copy()
        dmr.deallocate()

        return dm.transpose((1, 0))

def UCNEVPT2(*args, **kwargs):
    kwargs['mp_order'] = 2
    kwargs['theory'] = 'nevpt'
    return MRPT(*args, **kwargs)

def UCMRREPT2(*args, **kwargs):
    kwargs['mp_order'] = 2
    kwargs['theory'] = 'mrrept'
    return MRPT(*args, **kwargs)

if __name__ == '__main__':

    from pyscf import gto, scf, fci

    mol = gto.M(
        atom='''
            O  0.000000  0.000000  0.000000
            H  0.758602  0.000000  0.504284
            H  0.758602  0.000000  -0.504284
        ''',
        basis='6-31g',
        verbose=4, symmetry=False, spin=0)

    # Mean Field
    mf = scf.RHF(mol).run()

    # DMRG CASCI
    myci = mcscf.CASCI(mf, 8, 8)
    myci.fcisolver = _DMRGCI(mf)
    myci.fcisolver.dmrg_args['maxM'] = 1000
    myci.fcisolver.dmrg_args['sweep_tol'] = 1E-16
    myci = myci.run()
    dm1 = myci.make_rdm1()

    # DMRG NEVPT2
    mypt = UCNEVPT2(myci)
    mypt.dmrg_args['maxM'] = 1000
    mypt.dmrg_args['sweep_tol'] = 1E-16
    mypt = mypt.run()

    # DMRG NEVPT2 singles
    mypt = UCNEVPT2(myci, ci_order=1)
    mypt.dmrg_args['maxM'] = 1000
    mypt.dmrg_args['sweep_tol'] = 1E-16
    mypt = mypt.run()
