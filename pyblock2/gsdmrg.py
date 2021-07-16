
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

"""
Ground-State DMRG
using pyscf and block2.

Author: Huanchen Zhai, Dec 1, 2020
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
import time
import numpy as np

# Set spin-adapted or non-spin-adapted here
# SpinLabel = SU2
SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
else:
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect

try:
    if SpinLabel == SU2:
        from block2.su2 import MPICommunicator
    else:
        from block2.sz import MPICommunicator
    MPI = MPICommunicator()
    from mpi4py import MPI as PYMPI
    comm = PYMPI.COMM_WORLD

    def _print(*args, **kwargs):
        if MPI.rank == 0:
            print(*args, **kwargs)
except ImportError:
    MPI = None
    _print = print


class GSDMRGError(Exception):
    pass


class GSDMRG:
    """
    Ground-State DMRG for molecules.
    """

    def __init__(self, scratch='./nodex', memory=1 * 1E9, omp_threads=8, verbose=2,
                 print_statistics=True, mpi=None, dctr=True):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """

        if mpi is not None:
            memory = memory / mpi.size
            if mpi.rank != 0:
                verbose = 0
                print_statistics = False

        Random.rand_seed(0)
        init_memory(isize=int(memory * 0.1),
                    dsize=int(memory * 0.9), save_dir=scratch)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
        Global.threading.seq_type = SeqTypes.Simple
        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.print_statistics = print_statistics
        self.mpi = mpi
        self.dctr = dctr

        if mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC
            else:
                from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
            self.prule = ParallelRuleQC(mpi)
            self.pdmrule = ParallelRuleNPDMQC(mpi)
        else:
            self.prule = None
            self.pdmrule = None

    def init_hamiltonian_fcidump(self, pg, filename):
        """Read integrals from FCIDUMP file.
        pg : point group, pg = 'c1' or 'd2h'
        filename : FCIDUMP filename
        """
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                PointGroup.swap_d2h(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
        assert pg in ["d2h", "c1"]

    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym,
                         e_core, h1e, g2e, tol=1E-13, save_fcidump=None):
        """Initialize integrals using h1e, g2e, etc."""
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        self.fcidump = FCIDUMP()
        if not isinstance(h1e, tuple):
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.flatten()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
            self.fcidump.initialize_su2(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        else:
            assert SpinLabel == SZ
            assert isinstance(h1e, tuple) and len(h1e) == 2
            assert isinstance(g2e, tuple) and len(g2e) == 3
            mh1e_a = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e_b = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e = (mh1e_a, mh1e_b)
            for xmh1e, xh1e in zip(mh1e, h1e):
                k = 0
                for i in range(0, n_sites):
                    for j in range(0, i + 1):
                        assert abs(xh1e[i, j] - xh1e[j, i]) < tol
                        xmh1e[k] = xh1e[i, j]
                        k += 1
                xmh1e[np.abs(xmh1e) < tol] = 0.0
            mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        self.orb_sym = VectorUInt8(map(PointGroup.swap_d2h, orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(n_elec, twos, PointGroup.swap_d2h(isym))
        self.n_sites = n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        if save_fcidump is not None:
            self.fcidump.orb_sym = VectorUInt8(orb_sym)
            self.fcidump.write(save_fcidump)
        assert pg in ["d2h", "c1"]

    @staticmethod
    def fmt_size(i, suffix='B'):
        if i < 1000:
            return "%d %s" % (i, suffix)
        else:
            a = 1024
            for pf in "KMGTPEZY":
                p = 2
                for k in [10, 100, 1000]:
                    if i < k * a:
                        return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                    p -= 1
                a *= 1024
        return "??? " + suffix

    def dmrg(self, bond_dims, noises, dav_thrds, occs=None, bias=1.0, n_steps=30, conv_tol=1E-7, cutoff=1E-14):
        """Ground-State DMRG.

        Args:
            bond_dims : list(int), MPS bond dims for each sweep
            noises : list(double), noise for each sweep
            dav_thrds : list(double), Davidson convergence for each sweep
            occs : list(double) or None
                if occs = None, use FCI init MPS
                if occs = list(double), use occ init MPS
            bias : double, effective when occs is not None
                bias = 0.0: HF occ (e.g. occs => 2 2 2 ... 0 0 0)
                bias = 1.0: no bias (e.g. occs => 1.7 1.8 ... 0.1 0.05)
                bias = +inf: fully random (e.g. occs => 1 1 1 ... 1 1 1)
                
                increase bias if you want an init MPS with larger bond dim

                0 <= occ <= 2
                biased occ = 1 + (occ - 1) ** bias    (if occ > 1)
                           = 1 - (1 - occ) ** bias    (if occ < 1)
            n_steps : int, maximal number of sweeps
            conv_tol : double, energy convergence
            cutoff : cutoff of density matrix eigenvalues (default is the same as StackBlock)
        """

        if self.verbose >= 2:
            print('>>> START GS-DMRG <<<')
        t = time.perf_counter()

        # MPSInfo
        mps_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                           self.target, self.hamil.basis)
        mps_info.tag = 'KET'
        if occs is None:
            if self.verbose >= 2:
                print("Using FCI INIT MPS")
            mps_info.set_bond_dimension(bond_dims[0])
        else:
            if self.verbose >= 2:
                print("Using occupation number INIT MPS")
            mps_info.set_bond_dimension_using_occ(
                bond_dims[0], VectorDouble(occs), bias=bias)
        mps = MPS(self.n_sites, 0, 2)
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        # MPO
        tx = time.perf_counter()
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True)
        self.mpo_orig = mpo

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
            mpo = ParallelMPO(mpo, self.prule)

        if self.verbose >= 3:
            print('MPO time = ', time.perf_counter() - tx)

        if self.print_statistics:
            print('GS MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            print("GS INIT MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info.left_dims]))
            print("GS EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            print("GS EST PEAK MEM = ", GSDMRG.fmt_size(
                mem2), " SCRATCH = ", GSDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        if self.dctr:
            me.delayed_contraction = OpNamesSet.normal_ops()
        tx = time.perf_counter()
        me.init_environments(self.verbose >= 4)
        if self.verbose >= 3:
            print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
        dmrg.noise_type = NoiseTypes.ReducedPerturbative
        dmrg.decomp_type = DecompositionTypes.DensityMatrix
        dmrg.iprint = max(self.verbose - 1, 0)
        dmrg.cutoff = cutoff
        dmrg.solve(n_steps, mps.center == 0, conv_tol)

        self.gs_energy = dmrg.energies[-1][0]
        self.bond_dim = bond_dims[-1]

        mps.save_data()
        mps_info.save_data(self.scratch + "/GS_MPS_INFO")
        mps_info.deallocate()

        if self.print_statistics:
            dmain, dseco, imain, iseco = Global.frame.peak_used_memory
            print("GS PEAK MEM USAGE:",
                  "DMEM = ", GSDMRG.fmt_size(dmain + dseco),
                  "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                  "IMEM = ", GSDMRG.fmt_size(imain + iseco),
                  "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        if self.verbose >= 1:
            print("=== GS Energy = %20.15f" % self.gs_energy)

        if self.verbose >= 2:
            print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        return self.gs_energy

    def expectation(self):
        """Expectation on ground-state MPS (after GS-DMRG)."""
        if self.verbose >= 2:
            print('>>> START expectation <<<')
        t = time.perf_counter()

        if MPI is not None:
            MPI.barrier()

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        # MPO
        tx = time.perf_counter()
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True)

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
            mpo = ParallelMPO(mpo, self.prule)

        if self.verbose >= 3:
            print('MPO time = ', time.perf_counter() - tx)

        if self.print_statistics:
            print('GS EXPECT MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            mps.info.load_mutable()
            _, mem2, disk = mpo.estimate_storage(mps_info, 2)
            print("GS MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps.info.left_dims]))
            print("GS EST PEAK MEM = ", GSDMRG.fmt_size(
                mem2), " SCRATCH = ", GSDMRG.fmt_size(disk))
            mps.info.deallocate_mutable()

        if self.verbose >= 1:
            print("=== GS Energy = %20.15f" % self.gs_energy)

        if self.verbose >= 2:
            print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        pme = MovingEnvironment(mpo, mps, mps, "EX")
        pme.init_environments(False)
        expect = Expect(pme, mps.info.bond_dim + 100, mps.info.bond_dim + 100)
        expect.iprint = max(self.verbose - 1, 0)
        ener = expect.solve(False, mps.center == 0)

        mps.save_data()
        mps_info.deallocate()
        mpo.deallocate()

        return ener

    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, ridx=None):
        """1PDM on ground-state MPS (after GS-DMRG).
        ridx : reordering of orbitals (if no reordering, use ridx = None)"""
        if self.verbose >= 2:
            print('>>> START one-pdm <<<')
        t = time.perf_counter()

        if MPI is not None:
            MPI.barrier()

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        mps.info.load_mutable()
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        if self.mpi is not None:
            pmpo = ParallelMPO(pmpo, self.pdmrule)

        # 1PDM
        pme = MovingEnvironment(pmpo, mps, mps, "1PDM")
        pme.init_environments(False)
        expect = Expect(pme, mps.info.bond_dim, mps.info.bond_dim)
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, mps.center == 0)
        if SpinLabel == SU2:
            dmr = expect.get_1pdm_spatial(self.n_sites)
            dm = np.array(dmr).copy()
        else:
            dmr = expect.get_1pdm(self.n_sites)
            dm = np.array(dmr).copy()
            dm = dm.reshape((self.n_sites, 2,
                             self.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))

        if ridx is not None:
            dm[:, :] = dm[ridx, :][:, ridx]

        mps.save_data()
        mps_info.deallocate()
        dmr.deallocate()
        pmpo.deallocate()

        if self.verbose >= 2:
            print('>>> COMPLETE one-pdm | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        if SpinLabel == SU2:
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        else:
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        release_memory()


if __name__ == "__main__":

    # parameters
    n_threads = 4
    hf_type = "RHF"  # RHF or UHF
    mpg = 'd2h'  # point group: d2h or c1
    scratch = './tmp'

    memory = 1E9  # in bytes
    verbose = 3
    do_ccsd = True # True: use ccsd init MPS; False: use random init MPS
    bias = 10

    # if n_sweeps is larger than len(bond_dims), the last value will be repeated
    bond_dims = [250, 250, 250, 500, 500, 1000]
    # unit : norm(wfn) ** 2 (same as StackBlock)
    noises = [1E-6, 1E-7, 1E-8, 1E-9, 1E-9]
    # unit : norm(wfn) ** 2 (same as StackBlock)
    dav_thrds = [1E-6, 1E-6, 1E-7, 1E-7, 1E-8, 1E-8]
    n_sweeps = 30
    conv_tol = 1E-8

    import os
    if MPI is None or MPI.rank == 0:
        if not os.path.isdir(scratch):
            os.makedirs(scratch)
    if MPI is not None:
        MPI.barrier()
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf, symm, ao2mo, cc

    # H chain
    N = 10
    BOHR = 0.52917721092  # Angstroms
    R = 1.8 * BOHR
    mol = gto.M(atom=[['H', (0, 0, i * R)] for i in range(N)],
                basis='sto6g', verbose=0, symmetry=mpg)
    pg = mol.symmetry.lower()

    if hf_type == "RHF":
        mf = scf.RHF(mol)
    elif hf_type == "UHF":
        assert SpinLabel == SZ
        mf = scf.UHF(mol)

    if MPI is None or MPI.rank == 0:
        ener = mf.kernel()
    else:
        ener = 0
        mf.mo_coeff = None

    if MPI is not None:
        ener = comm.bcast(ener, root=0)
        mf.mo_coeff = comm.bcast(mf.mo_coeff, root=0)

    _print("SCF Energy = %20.15f" % ener)
    _print(("NON-" if SpinLabel == SZ else "") + "SPIN-ADAPTED")

    pg = mol.symmetry.lower()
    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
    else:
        raise GSDMRGError("Point group %d not supported yet!" % pg)

    mo_coeff = mf.mo_coeff

    if hf_type == "RHF":

        n_mo = mo_coeff.shape[1]

        orb_sym_str = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeff)
        orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])

        h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
        g2e = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff), n_mo)
        ecore = mol.energy_nuc()
        na = nb = mol.nelectron // 2

    else:

        mo_coeff_a, mo_coeff_b = mo_coeff[0], mo_coeff[1]
        n_mo = mo_coeff_b.shape[1]

        orb_sym_str_a = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeff_a)
        orb_sym_a = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str_a])

        orb_sym = orb_sym_a

        h1ea = mo_coeff_a.T @ mf.get_hcore() @ mo_coeff_a
        h1eb = mo_coeff_b.T @ mf.get_hcore() @ mo_coeff_b
        g2eaa = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff_a), n_mo)
        g2ebb = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff_b), n_mo)
        g2eab = ao2mo.kernel(
            mol, [mo_coeff_a, mo_coeff_a, mo_coeff_b, mo_coeff_b])
        h1e = (h1ea, h1eb)
        g2e = (g2eaa, g2ebb, g2eab)
        ecore = mol.energy_nuc()
        na, nb = mol.nelec
    
    if do_ccsd:
        if MPI is None or MPI.rank == 0:
            mcc = cc.CCSD(mf)
            mcc.kernel()
            dmmo = mcc.make_rdm1()
            if hf_type == "RHF":
                occs = np.diag(dmmo)
            else:
                occs = np.diag(dmmo[0]) + np.diag(dmmo[1])
        else:
            occs = None
        if MPI is not None:
            occs = comm.bcast(occs, root=0)
        _print('OCCS = ', occs)
    else:
        occs = None

    dmrg = GSDMRG(scratch=scratch, memory=memory,
                  verbose=verbose, omp_threads=n_threads, mpi=MPI)
    dmrg.init_hamiltonian(pg, n_sites=n_mo, n_elec=na + nb, twos=na - nb, isym=1,
                          orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e)
    # calculate ground-state energy
    ener = dmrg.dmrg(bond_dims=bond_dims, noises=noises, dav_thrds=dav_thrds, occs=occs, bias=bias,
                     n_steps=n_sweeps, conv_tol=conv_tol)
    # calculate pdm before changing FCIDUMP
    pdm = dmrg.get_one_pdm()

    # change integrals
    dmrg.init_hamiltonian(pg, n_sites=n_mo, n_elec=na + nb, twos=na - nb, isym=1,
                          orb_sym=orb_sym, e_core=ecore - 5, h1e=h1e, g2e=g2e)
    # calculate expectation of new FCIDUMP on ground-state MPS
    expect = dmrg.expectation()

    _print('GS ENERGY = %20.15f' % ener)
    _print('EXPECTATION = %20.15f' % expect)
    _print('pdm =', pdm)
    assert abs(ener - (-5.424385376325292)) < conv_tol
    assert abs(expect - (-5.424385376325292 - 5)) < conv_tol
    assert abs(pdm[0, 0, 0] - 9.90728423e-01) < np.sqrt(conv_tol)

    del dmrg  # IMPORTANT!!! --> to release stack memory
