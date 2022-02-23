
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

""" Finite temperature td-DMRG for Green's Function (time-domain => frequency domain).

:author: Henrik R. Larsson, Sep 2021
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16, TETypes
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
    from block2.su2 import ComplexExpect, MultiMPS, TimeEvolution
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
    from block2.sz import ComplexExpect, MultiMPS, TimeEvolution
    try:
        from block2.sz import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False
    from ft_dmrg import FTDMRG_SZ as FTDMRG
import tools; tools.init(SpinLabel)
from tools import saveMPStoDir, loadMPSfromDir, changeCanonicalForm
import numpy as np
from typing import List, Tuple, Union

if hasMPI:
    MPI = MPICommunicator()
else:
    class _MPI:
        rank = 0
    MPI = _MPI()



_print = tools.getVerbosePrinter(MPI.rank == 0, flush=True)


class RT_GFDMRG(FTDMRG):
    """
    Finite temperature td-DMRG for Green's Function in time domain.
    """

    def greens_function(self, mps: MPS, E0: float,
                        tmax: float,
                        dt: float,
                        idxs: List[int],
                        bond_dim: int,
                        cps_bond_dims: List[int], cps_noises: List[float],
                        cps_conv_tol: float, cps_n_sweeps: float,
                        diag_only=False, alpha=True,
                        cutoff=1E-14,
                        occs=None, bias=1.0, mo_coeff=None,
                        callback=lambda i,j,t,gf:None,
                        n_sub_sweeps=2, n_sub_sweeps_init = 4, exp_tol = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """ Solve for the Green's function in time-domain.
        GF_ij(t) = -i theta(t) <psi0| V_i' exp[i (H-E0) t] V_j |psi0>
        With V_i = a_i or a'_i. theta(t) is the step function (no time-reversal symm)
        Note the definition of the sign of frequency omega.

        :param mps: Start state psi0
        :param E0: Hamiltonian shift, typically ground state energy
        :param tmax: Max. propagation time
        :param dt: time step
        :param idxs: GF orbital indices to compute GF_ij, with i,j in idxs
        :param bond_dim: Max bond dimension during propagation
        :param cps_bond_dims: Number of bond dimensions for each sweep for V |þsi0>
        :param cps_noises: Noises for each sweep for V |þsi0>
        :param cps_conv_tol:  Sweep convergence tolerance for V |þsi0>
        :param cps_n_sweeps: Number of sweeps for obtaining V |psi0>
        :param cutoff: Bond dimension cutoff for sweeps
        :param diag_only: Solve only diagonal of GF: GF_ii
        :param alpha: Creation/annihilation operator refers to alpha spin (otherwise: beta spin)
        :param occs: Optional occupation number vector for V|psi0> initialization
        :param bias: Optional occupation number bias for V|psi0> initialization
        :param mo_coeff: MPO is in MO basis but GF should be computed in AO basis
        :param callback: Callback function after each GF computation.
                        Called as callback(i,j,t,GF_ij(t), GF_ij(2t) * delta_ij)
        :param n_sub_sweeps: Number of sub sweeps for RK4 time evolution
        :param n_sub_sweeps_int:  Number of sub sweeps for RK4 time evolution during initial dt
        :param exp_tol: exp(M) |vec> solver tolerance
        :return: propagation times, GF matrix, diagonal of GF matrix(2*t) [assuming real-valued initial state]
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

        mpo = 1.0 * self.mpo_orig
        mpo.const_e -= E0

        #mpo.const_e = 0 # hrl: sometimes const_e causes trouble for AncillaMPO
        mpo = IdentityAddedMPO(mpo) # hrl: alternative

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        if self.print_statistics:
            _print('RT-GF MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            mps_info2 = AncillaMPSInfo(self.n_physical_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(bond_dim)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("RT-GF EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("RT-GF EST PEAK MEM = ", RT_GFDMRG.fmt_size(
                mem2), " SCRATCH = ", RT_GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

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
                ops[ii] = OpElement(OpNames.D, SiteIndex(
                    (idx, ), (0 if alpha else 1, )), SZ(-1, -1 if alpha else 1, self.orb_sym[idx]))
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
        n_steps = int(tmax/dt + 1)
        ts = np.linspace(0, tmax, n_steps) # times
        dt = ts[1]-ts[0]
        gf_mat = np.zeros((len(idxs), len(idxs), n_steps), dtype=complex)
        gf_mat2t = np.zeros((len(idxs), n_steps), dtype=complex) # diagonal
        ts2t = 2 * ts

        # for autocorrelation
        idMPO = SimplifiedMPO(AncillaMPO(IdentityMPO(self.hamil)), RuleQC(), True, True)
        if self.mpi is not None:
            idMPO = ParallelMPO(idMPO, self.identrule)

        for ii, idx in enumerate(idxs):

            #
            # make MPS complex-valued
            #
            mps = MultiMPS.make_complex(rkets[ii], "mps_t")
            mps_t0 = MultiMPS.make_complex(rkets[ii], "mps_t0")
            if mps.dot != 1: # change to 2dot
                mps.load_data()
                mps_t0.load_data()
                mps.canonical_form = 'M' + mps.canonical_form[1:]
                mps_t0.canonical_form = 'M' + mps_t0.canonical_form[1:]
                mps.dot = 2
                mps_t0.dot = 2
                mps.save_data()
                mps_t0.save_data()

            # MPO for autocorrelation
            idME = MovingEnvironment(idMPO, mps_t0, mps, "acorr")
            acorr = ComplexExpect(idME, bond_dim, bond_dim)

            me = MovingEnvironment(mpo, mps, mps, "TE")
            if self.delayed_contraction:
                me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
            me.init_environments()

            method = TETypes.RK4
            te = TimeEvolution(me, VectorUBond([bond_dim]), method, n_sub_sweeps_init)
            te.cutoff = 0  # for tiny systems, this is important
            te.iprint = 6  # ft.verbose
            te.normalize_mps = False

            for it, tt in enumerate(ts):

                if self.verbose >= 2:
                    _print('>>>   START  TD-GF TIME = %10.5f Site = %4d %4d <<<' %
                           (tt, idx, idx))
                t = time.perf_counter()

                if it != 0: # time zero: no propagation
                    te.solve(1, +1j * dt, mps.center == 0, tol=exp_tol)
                    te.n_sub_sweeps = n_sub_sweeps

                idME.init_environments()
                gf_mat[ii, ii, it] = acorr.solve(False) * -1j
                #
                # double time trick
                # acorr(2t) = <psi(t)*|psi(t)>: here: site of center wavefunction ("state averaged")
                # This only works for the diagonal of the GF matrix, though
                if mps.wfns[0].data.size == 0:
                    loaded = True
                    mps.load_tensor(mps.center)
                vec = mps.wfns[0].data + 1j * mps.wfns[1].data
                gf_mat2t[ii, it] = np.vdot(vec.conj(),vec) * -1j
                if loaded:
                    mps.unload_tensor(mps.center)
                callback(ii,ii,tt,gf_mat[ii,ii,it], gf_mat2t[ii,it])
                #

                dmain, dseco, imain, iseco = Global.frame.peak_used_memory

                if self.verbose >= 1:
                    rgf, igf = gf_mat[ii,ii,it].real, gf_mat[ii,ii,it].imag
                    _print("=== TD-GF (%4d%4d | TIME = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                           (idx, idx, tt, rgf, igf, time.perf_counter() - t))

                if self.verbose >= 2:
                    _print('>>> COMPLETE TD-GF TIME = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                           (tt, idx, idx, time.perf_counter() - t))

                if diag_only:
                    continue

                for jj, idx2 in enumerate(idxs):
                    if jj > ii and rkets[jj].info.target == rkets[ii].info.target:

                        if self.verbose >= 2:
                            _print('>>>   START  TD-GF TIME = %10.5f Site = %4d %4d <<<' % (
                                tt, idx2, idx))
                        t = time.perf_counter()
                        if rkets[jj].center != mps.center:
                            # TODO: don't do this on rkets[jj] directly
                            changeCanonicalForm(rkets[jj], self.identrule)
                        mps_t0j = MultiMPS.make_complex(rkets[jj], "mps_t0j")
                        assert mps_t0j.center == mps.center, f"\nmps[j] form: {mps_t0j.canonical_form} \n" \
                                                                 f" mps[i] form: {mps.canonical_form}"

                        idMEj = MovingEnvironment(idMPO, mps_t0j, mps, "acorr_j")
                        idMEj.init_environments()
                        acorrj = ComplexExpect(idMEj, bond_dim, bond_dim)
                        gf_mat[jj, ii, it] = gf_mat[ii, jj, it] = acorrj.solve(False) * -1j

                        callback(ii,jj,tt,gf_mat[ii,jj,it],0.0)

                        if self.verbose >= 1:
                            rgf, igf = gf_mat[ii, jj, it].real, gf_mat[ii, jj, it].imag
                            _print("=== TD-GF (%4d%4d | TIME = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                                   (idx2, idx, tt, rgf, igf, time.perf_counter() - t))
                        if self.verbose >= 2:
                            _print('>>> COMPLETE GF TIME = %10.5f Site = %4d %4d | Time = %.2f <<<' %
                                   (tt, idx2, idx, time.perf_counter() - t))
        mps.save_data()
        idMPO.deallocate()
        mps.info.deallocate()

        if self.print_statistics:
            _print("TD-GF PEAK MEM USAGE:",
                   "DMEM = ", RT_GFDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", RT_GFDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        return ts, gf_mat, gf_mat2t

    def fourier_transform_gf(self,ts: np.ndarray, gf:np.ndarray, eta:float)\
            -> Tuple[np.ndarray, np.ndarray]:
        """  Fourier transform the GF
         (assuming that the time-date of the gf-tensor is in t the last dimension)

        :param ts: Propagation times
        :param gf: Greens function tensor: i x j x ts
        :param eta: Regularization
        :returns: omegas, gf in frequency domain
        """
        assert np.allclose(np.linspace(0,ts[-1],len(ts)), ts), "assume evenly spaced ts"
        dt = ts[1] - ts[0]
        omegas = np.fft.fftshift(np.fft.fftfreq(len(ts), dt)) * 2 * np.pi
        gf_freq = np.fft.fftshift(np.fft.fft(gf * np.exp(-eta*ts)),axes=-1) * dt
        gf_freq.real *= -1 # make consistent with GF definition
        return omegas, gf_freq


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
    cps_bond_dims=[MAX_M] # Bond dimension of \hat a_i |psi> *and* |psi> (in time evolution)
    cps_noises=[0]
    cps_tol=1E-7
    cps_n_sweeps=30

    beta = 80 # inverse temperature
    dbeta = 0.5 # "time" step
    mu = -0.026282794560 # Chemical potential for initial state preparation

    #################################################
    # Prepare initial state
    #################################################

    dmrg = RT_GFDMRG(scratch=scratch, memory=4e9,
                  verbose=3, omp_threads=n_threads)
    dmrg.init_hamiltonian_fcidump(point_group, "fcidump")
    #mps, mu = dmrg.optimize_mu(dmrg.fcidump.n_elec,mu, beta, dbeta, MAX_M)
    mps = dmrg.prepare_ground_state(mu, beta, dbeta, MAX_M)[0]


    tEnd = np.pi / 0.01
    dt = 0.1 # 1 / max(E)
    idxs = [0] # Calc S_ii
    alpha = True # alpha or beta spin
    fOut = open("test_ft_tddmrg_gf.dat","w")
    fOut.write("# t  Re(gf)  Im(gf)\n")
    fOut2 = open("test_ft_tddmrg_gf_2t.dat","w")
    fOut2.write("# t  Re(gf)  Im(gf)\n")
    def callback(i,j,t,gf, gf2t):
        print("-----",t,":",gf)
        fOut.write(f"{t:16.7f}   {gf.real:16.7f}  {gf.imag:16.7f}\n")
        fOut.flush()
        fOut2.write(f"{2*t:16.7f}   {gf2t.real:16.7f}  {gf2t.imag:16.7f}\n")
        fOut2.flush()
    ts, gf ,gf2t = dmrg.greens_function(mps, E0, tEnd, dt, idxs,
                          MAX_M,
                          cps_bond_dims, cps_noises, cps_tol, cps_n_sweeps,
                          diag_only=False,
                          alpha=alpha, max_solver_iter_off_diag=0,
                          callback=callback)
    eta = 0.005
    omegas, gf_freq = dmrg.fourier_transform_gf(ts, gf, eta)
    gf_freq = gf_freq.ravel()

    fOut.close()
    fOut2.close()
    fOut = open("test_ft_tddmrg_gf_ft.dat","w")
    fOut.write("# omega  Re(gf)  Im(gf)\n")
    for i in range(len(omegas)):
        fOut.write(f"{omegas[i]:16.7f}   {gf_freq[i].real:16.7f}  {gf_freq[i].imag:16.7f}\n")
    fOut.close()
    fOut = open("test_ft_tddmrg_gf_ft2.dat","w")
    omegas, gf_freq = dmrg.fourier_transform_gf(2*ts, gf2t, eta)
    gf_freq = gf_freq.ravel()
    fOut.write("# omega  Re(gf)  Im(gf)\n")
    for i in range(len(omegas)):
        fOut.write(f"{omegas[i]:16.7f}   {gf_freq[i].real:16.7f}  {gf_freq[i].imag:16.7f}\n")
    fOut.close()
