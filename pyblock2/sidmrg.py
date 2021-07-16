
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
DMRG with state interaction spin-orbit (SISO) coupling
using pyscf and block2.

Author: Huanchen Zhai, Mar 22, 2021
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
import time
import numpy as np
import copy

# Set spin-adapted or non-spin-adapted here
SpinLabel = SU2
# SpinLabel = SZ

if SpinLabel == SU2:
    from block2 import VectorSU2 as VectorSL
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC, CG
    from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.su2 import MultiMPSInfo, MultiMPS, ParallelMPO, ParallelRuleQC, ParallelRuleNPDMQC
else:
    from block2 import VectorSZ as VectorSL
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC, CG
    from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.sz import MultiMPSInfo, MultiMPS, ParallelMPO, ParallelRuleQC, ParallelRuleNPDMQC

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


class SIDMRGError(Exception):
    pass


class SIDMRG:
    """
    Multi-State DMRG for molecules.
    """

    def __init__(self, scratch='./nodex', memory=1 * 1E9, omp_threads=8, verbose=2,
                 print_statistics=False, mpi=None, dctr=True):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """

        Random.rand_seed(0)
        init_memory(isize=int(memory * 0.1),
                    dsize=int(memory * 0.9), save_dir=scratch)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
        Global.threading.seq_type = SeqTypes.Simple
        Global.frame.load_buffering = False
        Global.frame.save_buffering = False
        Global.frame.use_main_stack = False
        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.print_statistics = print_statistics
        self.mpi = mpi
        self.dctr = dctr

        if self.verbose >= 2:
            print(Global.frame)
            print(Global.threading)

        if mpi is not None:
            self.prule = ParallelRuleQC(mpi)
            self.pdmrule = ParallelRuleNPDMQC(mpi)
        else:
            self.prule = None
            self.pdmrule = None

    def init_hamiltonian_fcidump(self, pg, filename):
        """Read integrals from FCIDUMP file.
        pg : point group, pg = 'c1', 'c2v' or 'd2h'
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
        swap_pg = getattr(PointGroup, "swap_" + pg)
        self.orb_sym = VectorUInt8(map(swap_pg, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                swap_pg(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
        assert pg in ["d2h", "c2v", "c1"]

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
        swap_pg = getattr(PointGroup, "swap_" + pg)
        self.orb_sym = VectorUInt8(map(swap_pg, orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(n_elec, twos, swap_pg(isym))
        self.n_sites = n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        if save_fcidump is not None:
            self.fcidump.orb_sym = VectorUInt8(orb_sym)
            if self.mpi is None or self.mpi.rank == 0:
                self.fcidump.write(save_fcidump)
        assert pg in ["d2h", "c2v", "c1"]

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

    def dmrg(self, target, nroots, bond_dims, noises, dav_thrds, weights=None,
             tag='KET', occs=None, bias=1.0, n_steps=30, conv_tol=1E-7, cutoff=1E-14):
        """State-averaged multi-state DMRG.

        Args:
            target : quantum number of wavefunction
            nroots : number of states to solve
            weights : list(float) weight of each root
            tag : str; MPS tag
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
            print('>>> START State-Averaged DMRG <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        # MPSInfo
        targets = VectorSL([target])
        _print('TARGETS = ', list(targets), flush=True)
        mps_info = MultiMPSInfo(
            self.n_sites, self.hamil.vacuum, targets, self.hamil.basis)
        mps_info.tag = tag
        if occs is None:
            if self.verbose >= 2:
                _print("Using FCI INIT MPS")
            mps_info.set_bond_dimension(bond_dims[0])
        else:
            if self.verbose >= 2:
                _print("Using occupation number INIT MPS")
            mps_info.set_bond_dimension_using_occ(
                bond_dims[0], VectorDouble(occs), bias=bias)
        mps = MultiMPS(self.n_sites, 0, 2, nroots)
        if weights is not None:
            mps.weights = VectorDouble([float(x) for x in weights])
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        # MPO
        tx = time.perf_counter()
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
        self.mpo_orig = mpo

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        if self.verbose >= 3:
            _print('MPO time = ', time.perf_counter() - tx)

        if self.print_statistics:
            _print('MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("INIT MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info.left_dims]))
            _print("EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("EST PEAK MEM = ", SIDMRG.fmt_size(
                mem2), " SCRATCH = ", SIDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        if self.dctr:
            me.delayed_contraction = OpNamesSet.normal_ops()
        tx = time.perf_counter()
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(self.verbose >= 4)
        if self.verbose >= 3:
            _print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
        dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
        dmrg.decomp_type = DecompositionTypes.DensityMatrix
        dmrg.iprint = max(self.verbose - 1, 0)
        dmrg.cutoff = cutoff
        dmrg.solve(n_steps, mps.center == 0, conv_tol)

        self.energies = dmrg.energies[-1]
        self.bond_dim = bond_dims[-1]

        mps.save_data()
        mps_info.save_data(self.scratch + "/MPS_INFO.%s" % tag)
        mps_info.deallocate()

        if self.print_statistics:
            dmain, dseco, imain, iseco = Global.frame.peak_used_memory
            _print("PEAK MEM USAGE:",
                  "DMEM = ", SIDMRG.fmt_size(dmain + dseco),
                  "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                  "IMEM = ", SIDMRG.fmt_size(imain + iseco),
                  "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        if self.verbose >= 1:
            _print(("=== Energy = " + ("%15.8f") * len(self.energies)) %
                  tuple(self.energies))

        if self.verbose >= 2:
            _print('>>> COMPLETE DMRG | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        return self.energies

    def prepare_mps(self, tags, iroots=None):
        """Separate state-averaged MPS for later treatment.

        Args:
            tags : list(str)
                MPS tags to include.
            iroots : list(list(int)) or None
                For each tag, which root to include.
                If None, all roots are included.
        """
        if self.mpi is not None:
            self.mpi.barrier()

        impo = IdentityMPO(self.hamil)
        impo = SimplifiedMPO(impo, NoTransposeRule(RuleQC()),
                             True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        if self.mpi is not None:
            impo = ParallelMPO(impo, self.prule)

        mpss = []
        assert iroots is None or len(iroots) == len(tags)
        for it, tag in enumerate(tags):
            mps_info = MultiMPSInfo(0)
            mps_info.load_data(scratch + "/MPS_INFO.%s" % tag)
            mps = MultiMPS(mps_info)
            mps.load_data()
            if iroots is None:
                troots = range(mps.nroots)
            else:
                troots = iroots[it]
            for ir in troots:
                smps = mps.extract(ir, mps_info.tag + "-%d" % ir)
                if self.mpi is not None:
                    self.mpi.barrier()
                if smps.center != 0:
                    me = MovingEnvironment(impo, smps, smps, "EX")
                    me.delayed_contraction = OpNamesSet.normal_ops()
                    me.cached_contraction = True
                    me.save_partition_info = True
                    me.init_environments(0)
                    expect = Expect(me, smps.info.bond_dim +
                                    100, smps.info.bond_dim + 100)
                    expect.iprint = 0
                    expect.solve(True, False)
                    smps.save_data()
                mpss.append(smps)

        impo.deallocate()
        return mpss

    def energy_expectation(self, mpss):
        """Energy expectation on MPS."""

        if self.mpi is not None:
            self.mpi.barrier()

        # MPO
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        eners = []
        for mps in mpss:
            mps.load_data()
            me = MovingEnvironment(mpo, mps, mps, "EX")
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
            me.save_partition_info = True
            me.init_environments(self.verbose >= 4)
            expect = Expect(me, mps.info.bond_dim + 100,
                            mps.info.bond_dim + 100)
            expect.iprint = max(self.verbose - 1, 0)
            ener = expect.solve(False, mps.center == 0)
            mps.save_data()
            _print("TAG = %5s TARGET = %r EXPT = %15.8f" %
                  (mps.info.tag, mps.info.targets, ener))
            eners.append(ener)

        mpo.deallocate()
        return eners

    def trans_onepdm(self, mpss, soc=True, has_tran=True):
        """
        transition one-particle density matrix
        """
        if self.mpi is not None:
            self.mpi.barrier()

        # MPO
        pmpo = PDM1MPOQC(self.hamil, 1 if soc else 0)
        pmpo = SimplifiedMPO(pmpo,
                             NoTransposeRule(
                                 RuleQC()) if has_tran else RuleQC(),
                             True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        if self.mpi is not None:
            pmpo = ParallelMPO(pmpo, self.pdmrule)

        pdms = np.zeros((len(mpss), len(mpss)), dtype=object)
        for sbra, bmps_orig in enumerate(mpss):
            for sket, kmps_orig in enumerate(mpss):
                bmps = bmps_orig.extract(0, 'TMP-BRA')
                kmps = kmps_orig.extract(0, 'TMP-KET')
                if self.mpi is not None:
                    self.mpi.barrier()
                assert bmps.center == 0 and kmps.center == 0
                bmps.load_data()
                kmps.load_data()
                me = MovingEnvironment(pmpo, bmps, kmps, "EX")
                me.delayed_contraction = OpNamesSet.normal_ops()
                me.cached_contraction = True
                me.save_partition_info = True
                me.init_environments(self.verbose >= 4)
                expect = Expect(me, bmps.info.bond_dim + 100,
                                kmps.info.bond_dim + 100)
                expect.iprint = max(self.verbose - 1, 0)
                expect.solve(True, kmps.center == 0)
                if SpinLabel == SU2:
                    dmr = expect.get_1pdm_spatial(self.n_sites)
                    dm = np.array(dmr).copy()
                    qsbra = bmps.info.targets[0].twos
                    # fix different Wigner–Eckart theorem convention
                    dm *= np.sqrt(qsbra + 1)
                else:
                    dmr = expect.get_1pdm(self.n_sites)
                    dm = np.array(dmr).copy()
                    dm = dm.reshape((self.n_sites, 2,
                                     self.n_sites, 2))
                    dm = np.transpose(dm, (0, 2, 1, 3))
                dmr.deallocate()
                _print("IBRA = %2d (%5s - %10r) IKET = %2d (%5s - %10r) TRACE = %15.8f" % (sbra,
                                                                                          bmps_orig.info.tag, bmps.info.targets[
                                                                                              0], sket, kmps_orig.info.tag,
                                                                                          kmps.info.targets[0], np.diag(dm).sum()))
                pdms[sbra, sket] = dm / np.sqrt(2)

        pmpo.deallocate()
        return pdms

    def onepdm(self, mpss, soc=False, has_tran=False):
        """
        one-particle density matrix
        pdm[i, j] -> <AD_{i,alpha} A_{j,alpha}> + <AD_{i,beta} A_{j,beta}>
        """

        if self.mpi is not None:
            self.mpi.barrier()

        # MPO
        pmpo = PDM1MPOQC(self.hamil, 1 if soc else 0)
        pmpo = SimplifiedMPO(pmpo,
                             NoTransposeRule(
                                 RuleQC()) if has_tran else RuleQC(),
                             True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        if self.mpi is not None:
            pmpo = ParallelMPO(pmpo, self.pdmrule)

        pdms = []
        for mps_orig in mpss:
            mps = mps_orig.extract(0, 'TMP-KET')
            if self.mpi is not None:
                self.mpi.barrier()
            assert mps.center == 0
            mps.load_data()
            me = MovingEnvironment(pmpo, mps, mps, "EX")
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
            me.save_partition_info = True
            me.init_environments(self.verbose >= 4)
            expect = Expect(me, mps.info.bond_dim + 100,
                            mps.info.bond_dim + 100)
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
            dmr.deallocate()
            _print("TAG = %5s TARGET = %r TRACE = %15.8f" % (mps.info.tag,
                                                            mps.info.targets, np.diag(dm).sum()))
            pdms.append(dm)

        pmpo.deallocate()
        return pdms

    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        release_memory()


def get_jk(mol, dm0):
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(
        3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum('yijkl,lk->yij', hso2e, dm0)
    vk = np.einsum('yijkl,jk->yil', hso2e, dm0)
    vk += np.einsum('yijkl,li->ykj', hso2e, dm0)
    return vj, vk


def get_jk_amfi(mol, dm0):
    '''Atomic-mean-field approximation'''
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk

# hso (complex, pure imag) in unit cm-1


def compute_hso_ao(mol, dm0, qed_fac=1, amfi=False):
    from pyscf.data import nist
    alpha2 = nist.ALPHA ** 2
    hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    vj, vk = get_jk_amfi(mol, dm0) if amfi else get_jk(mol, dm0)
    hso2e = vj - vk * 1.5
    hso = qed_fac * (alpha2 / 4) * (hso1e + hso2e)
    return hso * 1j

# separate T^1 to T^1_(-1,0,1)


def spin_proj(cg, pdm, tjo, tjb, tjk):
    nmo = pdm.shape[0]
    ppdm = np.zeros((tjb + 1, tjk + 1, tjo + 1, nmo, nmo))
    for ibra in range(tjb + 1):
        for iket in range(tjk + 1):
            for iop in range(tjo + 1):
                tmb = -tjb + 2 * ibra
                tmk = -tjk + 2 * iket
                tmo = -tjo + 2 * iop
                factor = (-1) ** ((tjb - tmb) // 2) * \
                    cg.wigner_3j(tjb, tjo, tjk, -tmb, tmo, tmk)
                if factor != 0:
                    ppdm[ibra, iket, iop] = pdm * factor
    return ppdm

# from T^1_(-1,0,1) to Tx, Ty, Tz


def xyz_proj(ppdm):
    xpdm = np.zeros(ppdm.shape, dtype=complex)
    xpdm[:, :, 0] = (0.5 + 0j) * (ppdm[:, :, 0] - ppdm[:, :, 2])
    xpdm[:, :, 1] = (0.5j + 0) * (ppdm[:, :, 0] + ppdm[:, :, 2])
    xpdm[:, :, 2] = (np.sqrt(0.5) + 0j) * ppdm[:, :, 1]
    return xpdm


do_h2o = False
do_cu_atom = True

if __name__ == "__main__" and do_h2o:

    # parameters
    n_threads = 4
    hf_type = "RHF"  # RHF or UHF
    mpg = 'c2v'  # point group: d2h or c1
    scratch = './tmp'

    memory = 1E9  # in bytes
    verbose = 1
    do_ccsd = False  # True: use ccsd init MPS; False: use random init MPS
    bias = 10

    # if n_sweeps is larger than len(bond_dims), the last value will be repeated
    bond_dims = [250, 250, 250, 500, 500, 1000]
    # unit : norm(wfn) ** 2 (same as StackBlock)
    noises = [1E-6, 1E-7, 1E-8, 1E-9, 1E-9, 0]
    # unit : norm(wfn) ** 2 (same as StackBlock)
    dav_thrds = [1E-6, 1E-6, 1E-7, 1E-7, 1E-8, 1E-8, 1E-10]
    n_sweeps = 30
    conv_tol = 1E-14

    import os
    if MPI is None or MPI.rank == 0:
        if not os.path.isdir(scratch):
            os.makedirs(scratch)
    if MPI is not None:
        MPI.barrier()
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf, symm, ao2mo, cc
    from pyscf.data import nist

    mol = gto.M(atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284
    """, basis='sto3g', verbose=0, symmetry=mpg)
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

    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c2v':
        fcidump_sym = ['A1', 'B1', 'B2', 'A2']
    elif pg == 'c1':
        fcidump_sym = ["A"]
    else:
        raise SIDMRGError("Point group %d not supported yet!" % pg)

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

    dmrg = SIDMRG(scratch=scratch, memory=memory,
                  verbose=verbose, omp_threads=n_threads, mpi=MPI)
    dmrg.init_hamiltonian(pg, n_sites=n_mo, n_elec=na + nb, twos=na - nb, isym=1,
                          orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e,
                          save_fcidump=scratch + "/FCIDUMP")
    dmrg_opts = dict(bond_dims=bond_dims, noises=noises, dav_thrds=dav_thrds, occs=occs, bias=bias,
                     n_steps=n_sweeps, conv_tol=conv_tol)
    # SpinLabel(nelec, 2S, point group irrep)
    charge = 0
    e1 = dmrg.dmrg(target=SpinLabel(na + nb - charge, 0 +
                                    charge, 0), nroots=4, tag='MPS1', **dmrg_opts)
    e2 = dmrg.dmrg(target=SpinLabel(na + nb - charge, 2 +
                                    charge, 2), nroots=4, tag='MPS2', **dmrg_opts)
    e3 = dmrg.dmrg(target=SpinLabel(na + nb - charge, 2 +
                                    charge, 1), nroots=4, tag='MPS3', **dmrg_opts)
    e4 = dmrg.dmrg(target=SpinLabel(na + nb - charge, 2 +
                                    charge, 3), nroots=4, tag='MPS4', **dmrg_opts)
    eners = np.concatenate([e1, e2, e3, e4])
    mpss = dmrg.prepare_mps(tags=['MPS1', 'MPS2', 'MPS3', 'MPS4'])
    dmmo = dmrg.onepdm(mpss, )
    mo_coeff_inv = np.linalg.inv(mo_coeff)
    dm0ao = mo_coeff_inv.T @ dmmo[0] @ mo_coeff_inv
    hsoao = compute_hso_ao(mol, dm0ao, amfi=True) * 2
    if True:
        v = 2.1281747964476273E-004j * 2
        hsoao[:] = 0
        hsoao[0, 4, 3] = hsoao[1, 2, 4] = hsoao[2, 3, 2] = v
        hsoao[0, 3, 4] = hsoao[1, 4, 2] = hsoao[2, 2, 3] = -v
    hso = np.einsum('rij,ip,jq->rpq', hsoao, mo_coeff, mo_coeff)
    cg = CG(200)
    cg.initialize()
    print('HSO.SHAPE = ', hso.shape)
    pdms = dmrg.trans_onepdm(mpss)
    print('PDMS.SHAPE = ', pdms.shape)
    thrds = 29.0  # cm-1
    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
    n_mstates = sum([mpss[ibra].info.targets[0].twos +
                     1 for ibra in range(len(pdms))])
    hsiso = np.zeros((n_mstates, n_mstates), dtype=complex)
    hdiag = np.zeros((n_mstates, ), dtype=complex)
    qls = []
    imb = 0
    for ibra in range(len(pdms)):
        imk = 0
        tjb = mpss[ibra].info.targets[0].twos
        for iket in range(len(pdms)):
            pdm = pdms[ibra, iket]
            tjk = mpss[iket].info.targets[0].twos
            xpdm = xyz_proj(spin_proj(cg, pdm, 2, tjb, tjk))
            for ibm in range(xpdm.shape[0]):
                for ikm in range(xpdm.shape[1]):
                    somat = np.einsum('rij,rij->', xpdm[ibm, ikm], hso)
                    hsiso[ibm + imb, ikm + imk] = somat
                    somat *= au2cm
                    if abs(somat) > thrds:
                        print(('I1 = %4d (E1 = %15.8f) S1 = %4.1f MS1 = %4.1f '
                               + 'I2 = %4d (E2 = %15.8f) S2 = %4.1f MS2 = %4.1f Re = %9.3f Im = %9.3f')
                              % (ibra, eners[ibra], tjb / 2, -tjb / 2 + ibm, iket, eners[iket], tjk / 2,
                                 -tjk / 2 + ikm, somat.real, somat.imag))
            imk += tjk + 1
        for ibm in range(tjb + 1):
            qls.append((ibra, eners[ibra], tjb / 2, -tjb / 2 + ibm))
        hdiag[imb:imb + tjb + 1] = eners[ibra]
        imb += tjb + 1
    symm_err = np.linalg.norm(np.abs(hsiso - hsiso.T.conj()))
    print('SYMM Error (should be small) = ', symm_err)
    assert symm_err < 1E-10
    hfull = hsiso + np.diag(hdiag)
    heig, hvec = np.linalg.eigh(hfull)
    print('Total energies including SO-coupling:\n')
    for i in range(len(heig)):
        shvec = np.zeros(len(eners))
        imb = 0
        for ibra in range(len(eners)):
            tjb = mpss[ibra].info.targets[0].twos
            shvec[ibra] = np.linalg.norm(hvec[imb:imb + tjb + 1, i]) ** 2
            imb += tjb + 1
        iv = np.argmax(np.abs(shvec))
        print(' State %4d Total energy: %15.8f | largest |coeff|**2 %10.6f from I = %4d E = %15.8f S = %4.1f'
              % (i, heig[i], shvec[iv], iv, eners[iv], mpss[iv].info.targets[0].twos / 2))

    del dmrg  # IMPORTANT!!! --> to release stack memory


if __name__ == "__main__" and do_cu_atom:

    # parameters
    n_threads = 14
    mpg = 'c1'  # point group: d2h or c1
    scratch = './tmp'

    memory = 20E9  # in bytes
    verbose = 3
    bias = 5

    # if n_sweeps is larger than len(bond_dims), the last value will be repeated
    bond_dims = [250, 250, 250, 500, 500, 500, 1000, 1000, 1000]
    # unit : norm(wfn) ** 2 (same as StackBlock)
    noises = [1E-4, 1E-4, 1E-5, 1E-5, 1E-6, 1E-6, 1E-6, 1E-6, 0]
    # unit : norm(wfn) ** 2 (same as StackBlock)
    dav_thrds = [1E-5, 1E-5, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7, 1E-7]
    n_sweeps = 10
    conv_tol = 1E-6

    import os
    if MPI is None or MPI.rank == 0:
        if not os.path.isdir(scratch):
            os.makedirs(scratch)
    if MPI is not None:
        MPI.barrier()
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf, symm, ao2mo, cc, mcscf, tools
    from pyscf.mcscf import casci_symm
    from pyscf.data import nist

    ano = gto.basis.parse("""
    #BASIS SET: (21s,15p,10d,6f,4g) -> [6s,5p,3d,2f]
    Cu    S
        9148883.               0.00011722            -0.00003675             0.00001383            -0.00000271             0.00000346            -0.00001401
        1369956.               0.00033155            -0.00010403             0.00003914            -0.00000766             0.00000979            -0.00003959
        311782.6               0.00088243            -0.00027733             0.00010439            -0.00002047             0.00002626            -0.00010612
        88318.80               0.00217229            -0.00068417             0.00025745            -0.00005030             0.00006392            -0.00025900
        28815.53               0.00529547            -0.00167686             0.00063203            -0.00012442             0.00016087            -0.00064865
        10403.46               0.01296630            -0.00413460             0.00155718            -0.00030269             0.00038023            -0.00154770
        4057.791               0.03188050            -0.01035127             0.00391659            -0.00077628             0.00101716            -0.00408342
        1682.974               0.07576569            -0.02534791             0.00959476            -0.00185240             0.00229152            -0.00940206
        733.7543               0.16244637            -0.05836947             0.02238837            -0.00448270             0.00597497            -0.02389117
        333.2677               0.28435959            -0.11673284             0.04528606            -0.00866024             0.01044695            -0.04359832
        156.4338               0.34173643            -0.18466860             0.07498378            -0.01544726             0.02151583            -0.08588857
        74.69721               0.20955087            -0.14700899             0.06188170            -0.01089864             0.01030128            -0.04734865
        33.32262               0.03544751             0.18483257            -0.09047327             0.01467867            -0.00984324             0.06382592
        16.62237              -0.00243996             0.56773609            -0.39288472             0.09154122            -0.14292630             0.59363541
        8.208260               0.00146990             0.37956092            -0.35664956             0.06330857            -0.05612707             0.41506062
        3.609400              -0.00064379             0.04703755             0.34554793            -0.06093879             0.09135650            -1.70027942
        1.683449               0.00024738            -0.00072783             0.70639197            -0.27148545             0.45831402            -0.29799956
        0.733757              -0.00008118             0.00102976             0.25335911            -0.10138944            -0.19726740             1.70719425
        0.110207               0.00001615            -0.00007020             0.00658749             0.71212180            -1.30310157            -0.65600053
        0.038786              -0.00001233             0.00007090            -0.00044247             0.34613175             0.87975568            -0.31924524
        0.015514               0.00000452            -0.00002454             0.00072338             0.07198709             0.57310889             0.61772456
    Cu    P
    9713.253                   0.00039987            -0.00014879             0.00005053            -0.00015878             0.00012679
    2300.889                   0.00210157            -0.00078262             0.00026719            -0.00085359             0.00068397
        746.7706               0.01001097            -0.00376229             0.00127718            -0.00399919             0.00320291
        284.6806               0.03836039            -0.01457968             0.00499760            -0.01608092             0.01301351
        119.9999               0.11626756            -0.04571093             0.01561376            -0.04924864             0.04021371
        54.07386               0.25899831            -0.10593013             0.03689701            -0.12165679             0.10150109
        25.37321               0.38428226            -0.16836228             0.05796469            -0.17713367             0.13700954
        12.20962               0.29911210            -0.10373213             0.03791966            -0.14448197             0.10628972
        5.757421               0.08000672             0.21083155            -0.11706499             0.64743311            -0.83390265
        2.673402               0.00319440             0.48916443            -0.20912257             0.54797692             0.14412002
        1.186835               0.00147252             0.38468638            -0.05800532            -0.78180321             0.98301239
        0.481593              -0.00023391             0.08529338             0.16430736            -0.52607778            -0.52956418
        0.192637               0.00020761            -0.00161045             0.52885466             0.22437177            -0.75303984
        0.077055              -0.00008963             0.00206530             0.37373016             0.33392453             0.44288700
        0.030822               0.00002783            -0.00064175             0.08399866             0.14446374             0.57511589
    Cu    D
        249.3497               0.00134561            -0.00158359             0.00191116
        74.63837               0.01080983            -0.01281116             0.01772495
        28.37641               0.04733793            -0.05642679             0.07056381
        11.94893               0.13772582            -0.16641393             0.22906101
        5.317646               0.26263833            -0.35106014             0.43938511
        2.364417               0.33470401            -0.23717890            -0.26739908
        1.012386               0.31000597             0.21994641            -0.62590624
        0.406773               0.21316819             0.48841363             0.12211340
        0.147331               0.07865365             0.29537131             0.53683629
        0.058932               0.00603096             0.04295539             0.19750055
    Cu    F
        15.4333                0.03682063            -0.06200336
        6.1172                 0.24591418            -0.52876669
        2.4246                 0.50112688            -0.29078386
        0.9610                 0.35850648             0.52124157
        0.3809                 0.14728062             0.37532205
        0.1510                 0.02564748             0.10659798
    """)

    mol = gto.M(atom='Cu 0 0 0',
                symmetry=mpg,
                basis=ano, spin=1, charge=0, verbose=5)
    mf = scf.newton(scf.RHF(mol).sfx2c1e())
    pg = mol.symmetry.lower()

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

    cas_list = [9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26]
    mc = mcscf.CASSCF(mf, 11, (6, 5))
    mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list, base=0)
    mc.state_average_((0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    mc.fix_spin_(ss=0.75)

    if MPI is None or MPI.rank == 0:
        mc.kernel(mo)
        eners = mc.e_states
        dmao = mc.make_rdm1()
    else:
        mc.mo_coeff = None
        eners = None
        dmao = None

    if MPI is not None:
        mc.mo_coeff = comm.bcast(mc.mo_coeff, root=0)
        eners = comm.bcast(eners, root=0)
        dmao = comm.bcast(dmao, root=0)

    _print("CASSCF Energies =", "".join(["%20.10f" % x for x in eners]))

    # wfn (nroots, multiplicity, irrep)
    states = [(6, 2, 'A')]
    weights = [(0.5, ) + (0.1, ) * 5]
    tags = ['MPS%d' % x for x in range(len(states))]
    # active space (n_orbirals, n_electrons)
    # cas = (45, 19)
    cas = (11, 11)
    nactorb, nelec = cas
    mf.mo_coeff = mc.mo_coeff
    mo_coeff = mf.mo_coeff
    ff = mcscf.CASCI(mf, cas[0], cas[1])
    ncore = ff.ncore
    ff.mo_coeff = casci_symm.label_symmetry_(ff, mf.mo_coeff)
    orbsym = ff.mo_coeff.orbsym
    orbsym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i]
              for i in orbsym[ncore:ncore + nactorb]]
    _print(ff.mo_coeff.orbsym[ncore:ncore + nactorb])
    h1e, e_core = ff.get_h1cas()
    g2e = ao2mo.restore(1, ff.get_h2cas(), nactorb)
    if MPI is not None:
        ff.mo_coeff = comm.bcast(ff.mo_coeff, root=0)
        e_core = comm.bcast(e_core, root=0)
        h1e = comm.bcast(h1e, root=0)
        g2e = comm.bcast(g2e, root=0)

    _print('Integrals = ', h1e.shape, g2e.shape, e_core)

    dmrg = SIDMRG(scratch=scratch, memory=memory,
                  verbose=verbose, omp_threads=n_threads, mpi=MPI)
    dmrg.init_hamiltonian(pg, n_sites=nactorb, n_elec=nelec, twos=mol.spin, isym=1,
                          orb_sym=orbsym, e_core=e_core, h1e=h1e, g2e=g2e,
                          save_fcidump=scratch + "/FCIDUMP")
    dmrg_opts = dict(bond_dims=bond_dims, noises=noises, dav_thrds=dav_thrds,
                     occs=None, bias=bias, n_steps=n_sweeps, conv_tol=conv_tol)

    eners = [None] * len(states)
    for istate, state in enumerate(states):
        nroots = state[0]
        mult = state[1]
        wfnsym = state[2]
        tag = tags[istate]
        ms2 = mult - 1
        na = (nelec + ms2) // 2
        nb = nelec - na
        ff.nelecas = (na, nb)
        isym = mol.irrep_name.index(wfnsym)
        eners[istate] = dmrg.dmrg(target=SpinLabel(na + nb, na - nb, isym),
                                  nroots=nroots, tag=tag, weights=weights[istate], **dmrg_opts)

    eners = np.concatenate(eners)
    mpss = dmrg.prepare_mps(tags=tags)
    mo_coeff_inv = np.linalg.inv(mo_coeff)
    _print('\nGenerating Spin-Orbit Integrals:\n')
    if MPI is None or MPI.rank == 0:
        hsoao = compute_hso_ao(mol, dmao, amfi=True) * 2
        hso = np.einsum('rij,ip,jq->rpq', hsoao,
                        mo_coeff[:, ncore:ncore + nactorb],
                        mo_coeff[:, ncore:ncore + nactorb])
    else:
        hso = None
    if MPI is not None:
        hso = comm.bcast(hso, root=0)
    cg = CG(200)
    cg.initialize()
    _print('HSO.SHAPE = ', hso.shape)
    dmrg.verbose = 1
    pdms = dmrg.trans_onepdm(mpss)
    _print('PDMS.SHAPE = ', pdms.shape)
    thrds = 29.0  # cm-1
    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
    _print("\nComplex SO-Hamiltonian matrix elements over spin components of spin-free eigenstates:")
    _print("(In cm-1. Print threshold:%10.3f cm-1)\n" % thrds)
    n_mstates = sum([mpss[ibra].info.targets[0].twos +
                     1 for ibra in range(len(pdms))])
    hsiso = np.zeros((n_mstates, n_mstates), dtype=complex)
    hdiag = np.zeros((n_mstates, ), dtype=complex)
    qls = []
    imb = 0
    for ibra in range(len(pdms)):
        imk = 0
        tjb = mpss[ibra].info.targets[0].twos
        for iket in range(len(pdms)):
            pdm = pdms[ibra, iket]
            tjk = mpss[iket].info.targets[0].twos
            xpdm = xyz_proj(spin_proj(cg, pdm, 2, tjb, tjk))
            for ibm in range(xpdm.shape[0]):
                for ikm in range(xpdm.shape[1]):
                    somat = np.einsum('rij,rij->', xpdm[ibm, ikm], hso)
                    hsiso[ibm + imb, ikm + imk] = somat
                    somat *= au2cm
                    if abs(somat) > thrds:
                        _print(('I1 = %4d (E1 = %15.8f) S1 = %4.1f MS1 = %4.1f '
                               + 'I2 = %4d (E2 = %15.8f) S2 = %4.1f MS2 = %4.1f Re = %9.3f Im = %9.3f')
                              % (ibra, eners[ibra], tjb / 2, -tjb / 2 + ibm, iket, eners[iket], tjk / 2,
                                 -tjk / 2 + ikm, somat.real, somat.imag))
            imk += tjk + 1
        for ibm in range(tjb + 1):
            qls.append((ibra, eners[ibra], tjb / 2, -tjb / 2 + ibm))
        hdiag[imb:imb + tjb + 1] = eners[ibra]
        imb += tjb + 1
    symm_err = np.linalg.norm(np.abs(hsiso - hsiso.T.conj()))
    _print('SYMM Error (should be small) = ', symm_err)
    assert symm_err < 1E-10
    hfull = hsiso + np.diag(hdiag)
    heig, hvec = np.linalg.eigh(hfull)
    _print('\nTotal energies including SO-coupling:\n')
    for i in range(len(heig)):
        shvec = np.zeros(len(eners))
        imb = 0
        for ibra in range(len(eners)):
            tjb = mpss[ibra].info.targets[0].twos
            shvec[ibra] = np.linalg.norm(hvec[imb:imb + tjb + 1, i]) ** 2
            imb += tjb + 1
        iv = np.argmax(np.abs(shvec))
        _print(' State %4d Total energy: %15.8f | largest |coeff|**2 %10.6f from I = %4d E = %15.8f S = %4.1f'
              % (i, heig[i], shvec[iv], iv, eners[iv], mpss[iv].info.targets[0].twos / 2))

    e0 = np.average(heig[0:2])
    e1 = np.average(heig[2:8])
    e2 = np.average(heig[8:12])

    au2ev = 27.21139
    _print("")
    _print("E 2D(5/2)         = %10.4f eV" % ((e1 - e0) * au2ev))
    _print("E 2D(3/2)         = %10.4f eV" % ((e2 - e0) * au2ev))
    _print("2D(5/2) - 2D(3/2) = %10.4f eV" % ((e2 - e1) * au2ev))

    del dmrg  # IMPORTANT!!! --> to release stack memory
