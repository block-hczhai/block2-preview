
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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
DDMRG++ for Green's Function.
using pyscf and block2.

Author: Huanchen Zhai, Nov 5, 2020
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
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


class GFDMRGError(Exception):
    pass


class GFDMRG:
    """
    DDMRG++ for Green's Function for molecules.
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
        Global.threading.seq_type = SeqTypes.Tasked
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
        """Read integrals from FCIDUMP file."""
        assert self.fcidump is None
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
        assert self.fcidump is None
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

    def dmrg(self, bond_dims, noises, n_steps=30, conv_tol=1E-7, cutoff=1E-14, occs=None, bias=1.0):
        """Ground-State DMRG."""

        if self.verbose >= 2:
            print('>>> START GS-DMRG <<<')
        t = time.perf_counter()

        # MultiMPSInfo
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
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
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
            print("GS EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            print("GS EST PEAK MEM = ", GFDMRG.fmt_size(
                mem2), " SCRATCH = ", GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        if self.dctr:
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
        tx = time.perf_counter()
        me.init_environments(self.verbose >= 4)
        if self.verbose >= 3:
            print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_soft_max_iter = 4000
        dmrg.noise_type = NoiseTypes.ReducedPerturbative
        dmrg.decomp_type = DecompositionTypes.SVD
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
                  "DMEM = ", GFDMRG.fmt_size(dmain + dseco),
                  "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                  "IMEM = ", GFDMRG.fmt_size(imain + iseco),
                  "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        if self.verbose >= 1:
            print("=== GS Energy = %20.15f" % self.gs_energy)

        if self.verbose >= 2:
            print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                  (time.perf_counter() - t))

    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, ridx=None):
        if self.verbose >= 2:
            print('>>> START one-pdm <<<')
        t = time.perf_counter()

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

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

    def greens_function(self, bond_dims, noises, gmres_tol, conv_tol, n_steps,
                        gs_bond_dims, gs_noises, gs_conv_tol, gs_n_steps, idxs,
                        eta, freqs, addition, cutoff=1E-14, diag_only=False, 
                        alpha=True, occs=None, bias=1.0):
        """Green's function."""
        ops = [None] * len(idxs)
        rkets = [None] * len(idxs)
        rmpos = [None] * len(idxs)

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        if addition:
            mpo = -1.0 * self.mpo_orig
            mpo.const_e += self.gs_energy
        else:
            mpo = 1.0 * self.mpo_orig
            mpo.const_e -= self.gs_energy

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        if self.print_statistics:
            print('GF MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            print("GF EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            print("GF EST PEAK MEM = ", GFDMRG.fmt_size(
                mem2), " SCRATCH = ", GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        impo = SimplifiedMPO(IdentityMPO(self.hamil),
                             NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

        if self.mpi is not None:
            impo = ParallelMPO(impo, self.prule)

        def align_mps_center(ket, ref):
            cf = ket.canonical_form
            if ref.center == 0:
                ket.center += 1
                ket.canonical_form = ket.canonical_form[:-1] + 'S'
                while ket.center != 0:
                    ket.move_left(mpo.tf.opf.cg, self.prule)
            else:
                ket.canonical_form = 'K' + ket.canonical_form[1:]
                while ket.center != ket.n_sites - 1:
                    ket.move_right(mpo.tf.opf.cg, self.prule)
                ket.center -= 1
            if self.verbose >= 2:
                print('CF = %s --> %s' % (cf, ket.canonical_form))

        for ii, idx in enumerate(idxs):
            if self.verbose >= 2:
                print('>>> START Compression Site = %4d <<<' % idx)
            t = time.perf_counter()

            if SpinLabel == SZ:
                if addition:
                    ops[ii] = OpElement(OpNames.C, SiteIndex(
                        (idx, ), (0 if alpha else 1, )), SZ(1, 1 if alpha else -1, self.orb_sym[idx]))
                else:
                    ops[ii] = OpElement(OpNames.D, SiteIndex(
                        (idx, ), (0 if alpha else 1, )), SZ(-1, -1 if alpha else 1, self.orb_sym[idx]))
            else:
                if addition:
                    ops[ii] = OpElement(OpNames.C, SiteIndex(
                        (idx, ), ()), SU2(1, 1, self.orb_sym[idx]))
                else:
                    ops[ii] = OpElement(OpNames.D, SiteIndex(
                        (idx, ), ()), SU2(-1, 1, self.orb_sym[idx]))

            rket_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target + ops[ii].q_label, self.hamil.basis)
            rket_info.tag = 'DKET%d' % idx
            rket_info.set_bond_dimension(mps.info.bond_dim)
            if occs is None:
                if self.verbose >= 2:
                    print("Using FCI INIT MPS")
                rket_info.set_bond_dimension(mps.info.bond_dim)
            else:
                if self.verbose >= 2:
                    print("Using occupation number INIT MPS")
                rket_info.set_bond_dimension_using_occ(
                    mps.info.bond_dim, VectorDouble(occs), bias=bias)
            rkets[ii] = MPS(self.n_sites, mps.center, 2)
            rkets[ii].initialize(rket_info)
            rkets[ii].random_canonicalize()

            rkets[ii].save_mutable()
            rkets[ii].deallocate()
            rket_info.save_mutable()
            rket_info.deallocate_mutable()

            rmpos[ii] = SimplifiedMPO(
                SiteMPO(self.hamil, ops[ii]), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

            if self.mpi is not None:
                rmpos[ii] = ParallelMPO(rmpos[ii], self.prule)

            pme = MovingEnvironment(mpo, rkets[ii], rkets[ii], "PERT")
            pme.init_environments(False)
            rme = MovingEnvironment(rmpos[ii], rkets[ii], mps, "RHS")
            rme.init_environments(False)
            if self.dctr:
                pme.delayed_contraction = OpNamesSet.normal_ops()
                rme.delayed_contraction = OpNamesSet.normal_ops()

            cps = Linear(pme, rme, VectorUBond(gs_bond_dims),
                         VectorUBond([mps.info.bond_dim]), VectorDouble(gs_noises))
            cps.noise_type = NoiseTypes.ReducedPerturbative
            cps.decomp_type = DecompositionTypes.SVD
            cps.eq_type = EquationTypes.PerturbativeCompression
            cps.iprint = max(self.verbose - 1, 0)
            cps.cutoff = cutoff
            cps.solve(gs_n_steps, mps.center == 0, gs_conv_tol)

            if self.verbose >= 2:
                print('>>> COMPLETE Compression Site = %4d | Time = %.2f <<<' %
                      (idx, time.perf_counter() - t))

        gf_mat = np.zeros((len(idxs), len(idxs), len(freqs)), dtype=complex)

        for ii, idx in enumerate(idxs):

            if rkets[ii].center != mps.center:
                align_mps_center(rkets[ii], mps)
            lme = MovingEnvironment(mpo, rkets[ii], rkets[ii], "LHS")
            lme.init_environments(False)
            rme = MovingEnvironment(rmpos[ii], rkets[ii], mps, "RHS")
            rme.init_environments(False)
            if self.dctr:
                lme.delayed_contraction = OpNamesSet.normal_ops()
                rme.delayed_contraction = OpNamesSet.normal_ops()

            linear = Linear(lme, rme, VectorUBond(bond_dims),
                            VectorUBond(bond_dims), VectorDouble(noises))
            linear.gf_eta = eta
            linear.minres_conv_thrds = VectorDouble([gmres_tol] * n_steps)
            linear.noise_type = NoiseTypes.ReducedPerturbative
            linear.decomp_type = DecompositionTypes.SVD
            # TZ: Not raising error even if CG is not converged
            linear.minres_soft_max_iter = 4000
            linear.minres_max_iter = 5000
            linear.eq_type = EquationTypes.GreensFunction
            linear.iprint = max(self.verbose - 1, 0)
            linear.cutoff = cutoff

            for iw, w in enumerate(freqs):

                if self.verbose >= 2:
                    print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' %
                          (w, idx, idx))
                t = time.perf_counter()

                linear.tme = None
                linear.noises[0] = noises[0]
                linear.gf_omega = w
                linear.solve(n_steps, mps.center == 0, conv_tol)
                rgf, igf = linear.targets[-1]
                gf_mat[ii, ii, iw] = rgf + 1j * igf

                dmain, dseco, imain, iseco = Global.frame.peak_used_memory

                if self.verbose >= 1:
                    print("=== %3s GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                          ("ADD" if addition else "REM", idx, idx, w, rgf, igf, time.perf_counter() - t))

                if self.verbose >= 2:
                    print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                          (w, idx, idx, time.perf_counter() - t))

                if diag_only:
                    continue

                for jj, idx2 in enumerate(idxs):

                    if jj > ii and rkets[jj].info.target == rkets[ii].info.target:

                        if rkets[jj].center != rkets[ii].center:
                            align_mps_center(rkets[jj], rkets[ii])

                        if self.verbose >= 2:
                            print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' % (
                                w, idx2, idx))
                        t = time.perf_counter()

                        tme = MovingEnvironment(
                            impo, rkets[jj], rkets[ii], "GF")
                        tme.init_environments(False)
                        if self.dctr:
                            tme.delayed_contraction = OpNamesSet.normal_ops()
                        linear.noises[0] = noises[-1]
                        linear.tme = tme
                        linear.solve(1, mps.center != 0, 0)
                        rgf, igf = linear.targets[-1]
                        gf_mat[jj, ii, iw] = rgf + 1j * igf
                        gf_mat[ii, jj, iw] = rgf + 1j * igf

                        if self.verbose >= 1:
                            print("=== %3s GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                  ("ADD" if addition else "REM", idx2, idx, w, rgf, igf, time.perf_counter() - t))

                        if self.verbose >= 2:
                            print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                                  (w, idx2, idx, time.perf_counter() - t))
        mps.save_data()
        mps_info.deallocate()

        if self.print_statistics:
            print("GF PEAK MEM USAGE:",
                  "DMEM = ", GFDMRG.fmt_size(dmain + dseco),
                  "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                  "IMEM = ", GFDMRG.fmt_size(imain + iseco),
                  "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        return gf_mat

    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        release_memory()


def dmrg_mo_gf(mf, freqs, delta, mo_orbs=None, gmres_tol=1E-7, add_rem='+-',
               n_threads=8, memory=1E9, verbose=1, ignore_ecore=True,
               gs_bond_dims=[500], gs_noises=[1E-5, 1E-5, 1E-6, 1E-7, 0], gs_tol=1E-10, gs_n_steps=30,
               gf_bond_dims=[750], gf_noises=[1E-5, 0], gf_tol=1E-8, gf_n_steps=20, scratch='./tmp',
               lowdin=False, diag_only=False, alpha=True, occs=None, bias=1.0, cutoff=1E-14, mpi=None):
    '''
    Calculate the DMRG GF matrix in the MO basis.

    Args:
        mf : scf object
        freqs : np.ndarray of frequencies (real)
        delta : broadening (real)
        mo_orbs : list of indices of molecular orbtials
        gmres_tol : conjugate gradient (min res) conv tol (if too low will be extemely time-consuming)
        add_rem : '+' (addition) or '-' (removal) or '+-' (both)
        n_threads : number of threads (need parallel MKL library)
        memory : stack memory in bytes (default is 1 GB)
        verbose : 0 (quiet) 1 (per omega) 2 (per sweep) 3 (per orbital) 4 (per cg iteration)
        ignore_ecore : if True, set ecore to zero (should not affect GF)
        gs_bond_dims : np.ndarray of integers. Ground-State DMRG MPS bond dims for each sweep
        gs_noises : np.ndarray of float64. Ground-State DMRG noises for each sweep
        gs_tol : float64. Ground-State DMRG energy convergence.
        gs_n_steps : int. Ground-State DMRG max number of sweeps.
        gf_bond_dims : np.ndarray of integers. Green's function MPS bond dims for each sweep
        gf_noises : np.ndarray of float64. Green's function noises for each sweep
        gf_tol : float64. Green's function Im GF (i, i) convergence.
        gf_n_steps : int. Green's function max number of sweeps.
        scratch : scratch folder for temporary files.
        lowdin : if True, will use lowdin orbitals instead of molecular orbitals
        diag_only : if True, only calculate diagonal GF elements.
        alpha : bool. alpha spin or beta spin (not used if SU2)
        occs : list(double) or None
            if occs = None, use FCI init MPS (default)
            if occs = list(double), use occ init MPS
            either
                (A) len(occs) == n_sites (RHF or UHF, 0 <= occ <= 2)
             or (B) len(occs) == n_sites * 2 (UHF, 0 <= occ <= 1)
                    order: 0a, 0b, 1a, 1b, ...
        bias : float64, effective when occs is not None
            bias = 0.0: HF occ (e.g. occs => 2 2 2 ... 0 0 0)
            bias = 1.0: no bias (e.g. occs => 1.7 1.8 ... 0.1 0.05)
            bias = +inf: fully random (e.g. occs => 1 1 1 ... 1 1 1)

            increase bias if you want an init MPS with larger bond dim

            0 <= occ <= 2
            biased occ = 1 + (occ - 1) ** bias    (if occ > 1)
                        = 1 - (1 - occ) ** bias    (if occ < 1)
        cutoff : float64, lowest kept density matrix eigen value (default 1E-14)
            use smaller cutoff (e.g. 1E-20) when you need super accurate ground state energy
            noise smaller than cutoff will have no effect.
        mpi : if not None, MPI is used

    Returns:
        gfmat : np.ndarray of dims (len(mo_orbs), len(mo_orbs), len(freqs)) (complex)
            GF matrix in the MO basis (for selected mo_orbs).
    '''
    from pyscf import lo, symm, ao2mo

    mol = mf.mol

    pg = mol.symmetry.lower()
    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
    else:
        raise GFDMRGError("Point group %d not supported yet!" % pg)

    if lowdin:
        mo_coeff = lo.orth.lowdin(mol.intor('cint1e_ovlp_sph'))
    else:
        mo_coeff = mf.mo_coeff

    is_uhf = isinstance(mo_coeff, tuple) or mo_coeff.ndim == 3

    if not is_uhf:

        n_mo = mo_coeff.shape[1]

        orb_sym_str = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeff)
        orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])

        h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
        g2e = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff), n_mo)
        ecore = mol.energy_nuc()
        if ignore_ecore:
            ecore = 0
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
        if ignore_ecore:
            ecore = 0
        na, nb = mol.nelec

    if mo_orbs is None:
        mo_orbs = range(n_mo)

    dmrg = GFDMRG(scratch=scratch, memory=memory,
                  verbose=verbose, omp_threads=n_threads, mpi=mpi)
    dmrg.init_hamiltonian(pg, n_sites=n_mo, n_elec=na + nb, twos=na - nb, isym=1,
                          orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e)
    dmrg.dmrg(bond_dims=gs_bond_dims, noises=gs_noises,
              n_steps=gs_n_steps, conv_tol=gs_tol, occs=occs, bias=bias, cutoff=cutoff)
    pdm = dmrg.get_one_pdm()
    print('pdm = ', pdm)
    gf = 0
    for addit in [x == '+' for x in add_rem]:
        # only calculate alpha spin
        gf += dmrg.greens_function(gf_bond_dims, gf_noises, gmres_tol, gf_tol, gf_n_steps,
                                   gs_bond_dims, gs_noises, gs_tol, gs_n_steps, idxs=mo_orbs,
                                   eta=delta, freqs=freqs, addition=addit, diag_only=diag_only,
                                   alpha=alpha)

    del dmrg

    return pdm, gf


if __name__ == "__main__":

    # parameters
    n_threads = 28
    hf_type = "RHF"  # RHF or UHF
    mpg = 'c1'  # point group: d2h or c1
    scratch = './tmp'
    lowdin = False
    do_ccsd = True

    import os
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf

    # H chain
    N = 10
    BOHR = 0.52917721092  # Angstroms
    R = 1.8 * BOHR
    mol = gto.M(atom=[['H', (0, 0, i * R)] for i in range(N)],
                basis='sto6g', verbose=0, symmetry=mpg)
    pg = mol.symmetry.lower()

    if hf_type == "RHF":
        mf = scf.RHF(mol)
        ener = mf.kernel()
    elif hf_type == "UHF":
        assert SpinLabel == SZ
        mf = scf.UHF(mol)
        ener = mf.kernel()

    print("SCF Energy = %20.15f" % ener)
    print(("NON-" if SpinLabel == SZ else "") + "SPIN-ADAPTED")

    if do_ccsd:
        from pyscf import cc
        mcc = cc.CCSD(mf)
        mcc.kernel()
        dmmo = mcc.make_rdm1()
        if hf_type == "RHF":
            # RHF: 0 <= occ <= 2, len = n_mo
            occs = np.diag(dmmo)
        else:
            # UHF: 0 <= occ <= 1, order: 0a, 0b, 1a, 1b, ..., len = n_mo * 2
            occs = np.array([i for j in zip(np.diag(dmmo[0]), np.diag(dmmo[1])) for i in j])
        print('OCCS = ', occs)
    else:
        occs = None

    if lowdin:

        eta = 0.005
        freqs = np.arange(-0.8, -0.2, 0.01)
        mo_orbs = [4]
        t = time.perf_counter()
        pdm, gfmat = dmrg_mo_gf(mf, freqs=freqs, delta=eta, mo_orbs=mo_orbs, scratch=scratch, add_rem='-',
                                gf_bond_dims=[150], gf_noises=[1E-3, 5E-4], gf_tol=1E-4,
                                gmres_tol=1E-8, lowdin=True, ignore_ecore=False, n_threads=n_threads)

        print(gfmat)  # alpha only

        # alpha + beta
        pdos = (-2 / np.pi) * gfmat.imag.trace(axis1=0, axis2=1)
        print("PDOS = ", pdos)
        print("TIME = ", time.perf_counter() - t)

        import matplotlib.pyplot as plt

        plt.plot(freqs, pdos, 'o-', markersize=2)
        plt.xlabel('Frequency $\\omega$ (a.u.)')
        plt.ylabel('LDOS (a.u.)')
        plt.savefig('gf-figure.png', dpi=600)

    else:

        eta = 0.005
        freqs = [-0.2]
        t = time.perf_counter()
        pdm, gfmat = dmrg_mo_gf(mf, freqs=freqs, delta=eta, mo_orbs=None, scratch=scratch, add_rem='+-',
                                gs_bond_dims=[500], gs_noises=[1E-7, 1E-8, 1E-10, 0], gs_tol=1E-14, gs_n_steps=30,
                                gf_bond_dims=[500], gf_noises=[1E-7, 1E-8, 1E-10, 0], gf_tol=1E-8,
                                gmres_tol=1E-20, lowdin=False, ignore_ecore=False, alpha=False, verbose=3,
                                n_threads=n_threads, occs=occs, bias=1.0)
        gfmat = np.einsum('ip,pqr,jq->ijr', mf.mo_coeff, gfmat, mf.mo_coeff)
        print(gfmat)
