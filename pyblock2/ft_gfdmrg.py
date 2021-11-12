
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2021 Henrik R. Larsson <larsson@caltech.edu>
#  Copyright (C) 2021 Huanchen Zhai <hczhai@caltech.edu>
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

""" Finite temperature DDMRG++ for Green's Function.

:author: Henrik R. Larsson, Sep 2021
        Based on zero temperature GFDMRG from Huanchen Zhai
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16
from block2 import LinearSolverTypes

import time

# Set spin-adapted or non-spin-adapted here
#SpinLabel = SU2
SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import AncillaMPO, AncillaMPSInfo
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO, IdentityAddedMPO
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.su2 import VectorOpElement, LocalMPO
    try:
        from block2.su2 import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False
    from ft_dmrg import FTDMRG_SU2 as FTDMRG
else:
    from block2.sz import AncillaMPO, AncillaMPSInfo
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO, IdentityAddedMPO
    from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.sz import VectorOpElement, LocalMPO
    try:
        from block2.sz import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False
    from ft_dmrg import FTDMRG_SZ as FTDMRG
import tools; tools.init(SpinLabel)
from tools import saveMPStoDir, loadMPSfromDir
import numpy as np
from typing import List, Union, Tuple


if hasMPI:
    MPI = MPICommunicator()
else:
    class _MPI:
        rank = 0
    MPI = _MPI()



_print = tools.getVerbosePrinter(MPI.rank == 0, flush=True)


class GFDMRG(FTDMRG):
    """
    DDMRG++ for Green's Function for molecules.
    """

    def greens_function(self, mps: MPS, E0: float, omegas: np.ndarray, eta: float,
                        idxs: List[int],
                        bond_dims: List[int], noises: List[float],
                        solver_tol: float, conv_tol: float, n_sweeps: int,
                        cps_bond_dims: List[int], cps_noises: List[float],
                        cps_conv_tol: float, cps_n_sweeps: float,
                        diag_only=False, alpha=True, addition = False,
                        cutoff=1E-14,
                        max_solver_iter=20000,
                        max_solver_iter_off_diag=0,
                        occs=None, bias=1.0, mo_coeff=None,
                        solver_type = LinearSolverTypes.LSQR,
                        use_preconditioner=False,
                        linear_solver_params = (80,-1),
                        callback=lambda i,j,w,gf:None) -> np.ndarray:
        """ Solve for the Green's function.
        GF_ij(omega + i eta) = <psi0| V_i' inv(H - E0 + omega + i eta) V_j |psi0>
        With V_i = a_i or a'_i.
        Note the definition of the sign of frequency omega.
        The GF is solved via the correction MPS C = inv(H - E0 + omega + i eta) V |psi0>

        :param mps: Start state psi0
        :param E0: Hamiltonian shift, typically ground state energy
        :param omegas: Frequencies the GF is computed
        :param eta: Broadening parameter
        :param idxs: GF orbital indices to compute GF_ij, with i,j in idxs
        :param bond_dims: Number of bond dimensions for each sweep for the correction MPS
        :param noises: Noises for each sweep
        :param solver_tol: Convergence tolerance for linear system solver
        :param conv_tol: Sweep convergence tolerance
        :param n_sweeps: Number of sweeps
        :param cps_bond_dims: Number of bond dimensions for each sweep for V |þsi0>
        :param cps_noises: Noises for each sweep for V |þsi0>
        :param cps_conv_tol:  Sweep convergence tolerance for V |þsi0>
        :param cps_n_sweeps: Number of sweeps for obtaining V |psi0>
        :param addition: If true, use -H + E0 instead of H - E0
        :param cutoff: Bond dimension cutoff for sweeps
        :param diag_only: Solve only diagonal of GF: GF_ii
        :param alpha: Creation/annihilation operator refers to alpha spin (otherwise: beta spin)
        :param max_solver_iter: Max. solver iterations for GF
        :param max_solver_iter_off_diag: Max. solver iteration for off diagonal terms (if 0, is set to max_solver_iter)
        :param occs: Optional occupation number vector for V|psi0> initialization
        :param bias: Optional occupation number bias for V|psi0> initialization
        :param mo_coeff: MPO is in MO basis but GF should be computed in AO basis
        :param solver_type: Linear solver type. Supported: GCROT; IDRS; LSQR
                LSQR is most robust
        :param use_preconditioner: Preconditioner for solver. No one or diagonal. No one is most robust.
        :param linear_solver_params: Sizes for either GCROT(M,K) or for IDR(S) (first entry; 2nd will be ignored)
            Will be ignored for LSQR.
        :param callback: Callback function after each GF computation.
                        Called as callback(i,j,w,GF_ij(omega))
        :return: the GF matrix
        """
        ops = [None] * len(idxs)
        rkets = [None] * len(idxs)
        rmpos = [None] * len(idxs)

        if self.mpi is not None:
            self.mpi.barrier()

        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(AncillaMPO(mpo), RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo


        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        if addition:
            mpo = -1.0 * self.mpo_orig
            mpo.const_e += E0
        else:
            mpo = 1.0 * self.mpo_orig
            mpo.const_e -= E0

        #omegas += mpo.const_e
        #mpo.const_e = 0 # hrl: sometimes const_e causes trouble for AncillaMPO
        mpo = IdentityAddedMPO(mpo) # hrl: alternative

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        if self.print_statistics:
            _print('GF MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = AncillaMPSInfo(self.n_physical_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("GF EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("GF EST PEAK MEM = ", GFDMRG.fmt_size(
                mem2), " SCRATCH = ", GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        impo = SimplifiedMPO(AncillaMPO(IdentityMPO(self.hamil)),
                             NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

        if self.mpi is not None:
            impo = ParallelMPO(impo, self.identrule)

        def align_mps_center(ket, ref):
            # center itself may be "K" or "C"
            isOk = ket.center == ref.center
            isOk = isOk and ket.canonical_form[ket.center+1:] == ref.canonical_form[ref.center+1:]
            isOk = isOk and ket.canonical_form[:ket.center] == ref.canonical_form[:ref.center]
            if isOk:
                return
            if self.mpi is not None:
                self.mpi.barrier()
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
                _print('CF = %s --> %s' % (cf, ket.canonical_form))

        ############################################
        # Prepare creation/annihilation operators
        ############################################
        if mo_coeff is None:
            if self.ridx is not None:
                gidxs = self.ridx[np.array(idxs)]
            else:
                gidxs = idxs
        else:
            if self.idx is not None:
                mo_coeff = mo_coeff[:, self.idx]
            gidxs = list(range(self.n_sites))
            ops = [None] * self.n_sites
            _print('idxs = ', idxs, 'gidxs = ', gidxs)

        for ii, idx in enumerate(gidxs):
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

        ############################################
        # Solve V_i |psi0>
        ############################################
        for ii, idx in enumerate(idxs):
            if self.mpi is not None:
                self.mpi.barrier()
            if self.verbose >= 2:
                _print('>>> START Compression Site = %4d <<<' % idx)
            t = time.perf_counter()

            rket_info = AncillaMPSInfo(self.n_physical_sites, self.hamil.vacuum,
                                       self.target + ops[ii].q_label, self.hamil.basis)
            rket_info.tag = 'DKET%d' % idx
            bond_dim = mps.info.bond_dim
            bond_dim = bond_dim if bond_dim != 0 else max([x.n_states_total for x in mps.info.left_dims])
            if occs is None:
                if self.verbose >= 2:
                    _print("Using FCI INIT MPS,bond_dim=", bond_dim)
                rket_info.set_bond_dimension(bond_dim)
            else:
                if self.verbose >= 2:
                    _print("Using occupation number INIT MPS; bond_dim=", bond_dim)
                rket_info.set_bond_dimension_using_occ(
                    bond_dim, VectorDouble(occs), bias=bias)
            rkets[ii] = MPS(self.n_sites, mps.center, 2)
            rkets[ii].initialize(rket_info)
            rkets[ii].random_canonicalize()

            rkets[ii].save_mutable()
            rkets[ii].deallocate()
            rket_info.save_mutable()
            rket_info.deallocate_mutable()

            if mo_coeff is None:
                # the mpo and gf are in the same basis
                # the mpo is SiteMPO
                rmpos[ii] = SimplifiedMPO(
                    AncillaMPO(SiteMPO(self.hamil, ops[ii])), NoTransposeRule(RuleQC()),
                    True, True, OpNamesSet((OpNames.R, OpNames.RD)))
            else:
                # the mpo is in mo basis and gf is in ao basis
                # the mpo is sum of SiteMPO (LocalMPO)
                ao_ops = VectorOpElement([None] * self.n_sites)
                for ix in range(self.n_physical_sites):
                    iix = ix * 2 # not ancilla sites
                    ao_ops[iix] = ops[iix] * mo_coeff[idx, ix]
                rmpos[ii] = SimplifiedMPO(
                    AncillaMPO(LocalMPO(self.hamil, ao_ops)), NoTransposeRule(RuleQC()),
                    True, True, OpNamesSet((OpNames.R, OpNames.RD)))

            if self.mpi is not None:
                rmpos[ii] = ParallelMPO(rmpos[ii], self.siterule)

            if len(cps_noises) == 1 and cps_noises[0] == 0:
                pme = None
            else:
                pme = MovingEnvironment(mpo, rkets[ii], rkets[ii], "PERT")
                pme.init_environments(False)
            rme = MovingEnvironment(rmpos[ii], rkets[ii], mps, "RHS")
            rme.init_environments(False)
            if self.delayed_contraction:
                if pme is not None:
                    pme.delayed_contraction = OpNamesSet.normal_ops()
                rme.delayed_contraction = OpNamesSet.normal_ops()

            # ME
            cps = Linear(pme, rme, VectorUBond(cps_bond_dims),
                         VectorUBond([mps.info.get_max_bond_dimension() + 100]),
                         VectorDouble(cps_noises))
            cps.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
            if pme is not None:
                cps.eq_type = EquationTypes.PerturbativeCompression
            cps.iprint = max(self.verbose - 1, 0)
            cps.cutoff = cutoff
            cps.solve(cps_n_sweeps, mps.center == 0, cps_conv_tol)


            if self.verbose >= 2:
                _print('>>> COMPLETE Compression Site = %4d | Time = %.2f <<<' %
                       (idx, time.perf_counter() - t))

        ############################################
        # Big GF LOOP
        ############################################
        gf_mat = np.zeros((len(idxs), len(idxs), len(omegas)), dtype=complex)

        for ii, idx in enumerate(idxs):

            if rkets[ii].center != mps.center:
                align_mps_center(rkets[ii], mps)
            lme = MovingEnvironment(mpo, rkets[ii], rkets[ii], "LHS")
            lme.init_environments(False)
            rme = MovingEnvironment(rmpos[ii], rkets[ii], mps, "RHS")
            rme.init_environments(False)
            if self.delayed_contraction:
                lme.delayed_contraction = OpNamesSet.normal_ops()
                rme.delayed_contraction = OpNamesSet.normal_ops()

            linear = Linear(lme, rme, VectorUBond(bond_dims),
                            VectorUBond(cps_bond_dims[-1:]), VectorDouble(noises))
            linear.gf_eta = eta
            linear.linear_conv_thrds = VectorDouble([solver_tol] * n_sweeps)
            linear.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
            linear.linear_solver_params = linear_solver_params
            linear.solver_type = solver_type
            linear.linear_use_precondition = use_preconditioner


            # TZ: Not raising error even if CG is not converged
            linear.linear_soft_max_iter = max_solver_iter
            linear.linear_max_iter = max_solver_iter + 1000
            linear.eq_type = EquationTypes.GreensFunction
            linear.iprint = max(self.verbose - 1, 0)
            linear.cutoff = cutoff

            for iw, w in enumerate(omegas):

                if self.verbose >= 2:
                    _print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' %
                           (w, idx, idx))
                t = time.perf_counter()

                linear.tme = None
                linear.linear_soft_max_iter = max_solver_iter
                linear.noises = VectorDouble(noises)
                linear.bra_bond_dims = VectorUBond(bond_dims)
                linear.gf_omega = w
                linear.solve(n_sweeps, mps.center == 0, conv_tol)
                # set noises to 0 for next round
                noises = [0]
                min_site = np.argmin(np.array(linear.sweep_targets)[:, 1])
                if mps.center == 0:
                    min_site = mps.n_sites - 2 - min_site
                _print("GF.IMAG MIN SITE = %4d" % min_site)
                rgf, igf = linear.targets[-1]
                gf_mat[ii, ii, iw] = rgf + 1j * igf
                callback(ii,ii,w,gf_mat[ii,ii,iw])

                dmain, dseco, imain, iseco = Global.frame.peak_used_memory

                if self.verbose >= 1:
                    _print("=== %3s GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                           ("ADD" if addition else "REM", idx, idx, w, rgf, igf, time.perf_counter() - t))

                if self.verbose >= 2:
                    _print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                           (w, idx, idx, time.perf_counter() - t))

                if diag_only:
                    continue

                for jj, idx2 in enumerate(idxs):

                    if jj > ii and rkets[jj].info.target == rkets[ii].info.target:

                        if rkets[jj].center != rkets[ii].center:
                            align_mps_center(rkets[jj], rkets[ii])

                        if self.verbose >= 2:
                            _print('>>>   START  GF OMEGA = %10.5f Site = %4d %4d <<<' % (
                                w, idx2, idx))
                        t = time.perf_counter()

                        tme = MovingEnvironment(
                            impo, rkets[jj], rkets[ii], "GF")
                        tme.init_environments(False)
                        if self.delayed_contraction:
                            tme.delayed_contraction = OpNamesSet.normal_ops()
                        linear.noises = VectorDouble(noises[-1:])
                        linear.bra_bond_dims = VectorUBond(bond_dims[-1:])
                        linear.target_bra_bond_dim = cps_bond_dims[-1]
                        linear.target_ket_bond_dim = cps_bond_dims[-1]
                        linear.tme = tme
                        if max_solver_iter_off_diag == 0:
                            linear.solve(1, mps.center != 0, 0)
                            rgf, igf = linear.targets[-1]
                        else:
                            linear.linear_soft_max_iter = max_solver_iter_off_diag if \
                                max_solver_iter_off_diag != -1 else max_solver_iter
                            linear.solve(1, mps.center == 0, 0)
                            if mps.center == 0:
                                rgf, igf = np.array(linear.sweep_targets)[::-1][min_site]
                            else:
                                rgf, igf = np.array(linear.sweep_targets)[min_site]
                        gf_mat[jj, ii, iw] = rgf + 1j * igf
                        gf_mat[ii, jj, iw] = rgf + 1j * igf
                        callback(ii,jj,w,gf_mat[ii,jj,iw])

                        if self.verbose >= 1:
                            _print("=== %3s GF (%4d%4d | OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                   ("ADD" if addition else "REM", idx2, idx, w, rgf, igf, time.perf_counter() - t))

                        if self.verbose >= 2:
                            _print('>>> COMPLETE GF OMEGA = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                                   (w, idx2, idx, time.perf_counter() - t))
        mps.save_data()
        mps.info.deallocate()

        if self.print_statistics:
            _print("GF PEAK MEM USAGE:",
                   "DMEM = ", GFDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", GFDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        return gf_mat

if __name__ == "__main__":

    # parameters
    n_threads = 2
    point_group = 'c1'
    scratch = '/tmp/block2'
    load_dir = None

    import os
    if MPI.rank == 0:
        tools.mkDir(scratch)
    if hasMPI:
        MPI.barrier()
    os.environ['TMPDIR'] = scratch
    E0 = -2.1379703474141984 #H4
    E0 = -1.568351864513 # H3


    MAX_M = 50
    gf_bond_dims=[MAX_M]
    gf_noises=[1E-3, 5E-4, 0]
    gf_tol=1E-4
    solver_tol=1E-6
    gf_n_steps = 20
    cps_bond_dims=[MAX_M] # Bond dimension of \hat a_i |psi> *and* |psi> (in time evolution)
    cps_noises=[0]
    cps_tol=1E-7
    cps_n_sweeps=30

    beta = 80 # inverse temperature
    dbeta = 0.5 # "time" step
    mu = -0.026282794560 # Chemical potential for initial state preparation

    #################################################
    # Prepare inital state
    #################################################

    dmrg = GFDMRG(scratch=scratch, memory=4e9,
                  verbose=3, omp_threads=n_threads)
    dmrg.init_hamiltonian_fcidump(point_group, "fcidump")
    #mps, mu = dmrg.optimize_mu(dmrg.fcidump.n_elec,mu, beta, dbeta, MAX_M)
    mps = dmrg.prepare_ground_state(mu, beta, dbeta, MAX_M)[0]


    eta = 0.005
    omegas = np.linspace(-1,0,200)
    # Frequencies more far away seem to be much harder to converge.
    # So it is better to start with small omegas and then use that as initial guess.
    omegas = omegas[::-1]
    idxs = [0] # Calc S_ii
    alpha = True # alpha or beta spin
    fOut = open("ft_gfdmrg_freqs.dat","w")
    fOut.write(f"# eta={eta}\n")
    fOut.write("# omega  Re(gf)  Im(gf)\n")
    def callback(i,j,omega,gf):
        print("-----",omega,":",gf)
        fOut.write(f"{omega:16.7f}   {gf.real:16.7f}  {gf.imag:16.7f}\n")
        fOut.flush()
    gf = dmrg.greens_function(mps, E0, omegas, eta, idxs,
                              gf_bond_dims, gf_noises, solver_tol, gf_tol, gf_n_steps,
                              cps_bond_dims, cps_noises, cps_tol, cps_n_sweeps,
                              addition=False, diag_only=False,
                              alpha=alpha, max_solver_iter_off_diag=0,
                              callback=callback)
    fOut.close()
