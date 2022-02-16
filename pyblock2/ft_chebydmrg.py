#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2021 Huanchen Zhai <hczhai@caltech.edu>
#  Copyright (C) 2022 Henrik R. Larsson <larsson@caltech.edu>
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

""" Finite temperature chebychev-DMRG for Green's Function

:author: Henrik R. Larsson, Jan 2022
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16
import time, math

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
from tools import saveMPStoDir, loadMPSfromDir, changeCanonicalForm
import numpy as np
from typing import List, Tuple, Union
import scipy.linalg as la

if hasMPI:
    MPI = MPICommunicator()
else:
    class _MPI:
        rank = 0
    MPI = _MPI()



_print = tools.getVerbosePrinter(MPI.rank == 0, flush=True)

def chebyCoeffNumerical(j, polOrder, fct, sigma, tau):
    """ Computes Chebychev coefficient numerically.

    :see: - http://arxiv.org/abs/1704.00512; J. Chem. Theory Comput. 2017, 13, 10, 4684–4698
          - NumRecs, 3rd edition page 234, eq 5.8.7

    :param j: jth Chebychev cofficient
    :param polOrder: Polynomial expansion number
    :param fct: Function to approximate
    :param sigma: 2 / (eMax-eMin)
    :param tau: (eMin + eMax)/ 2
    """
    c = 0
    for k in range(polOrder):
        c += fct(np.cos(np.pi * (k + 1 / 2) / polOrder) / sigma + tau) * np.cos(np.pi * j * (k + 1 / 2) / polOrder)
    c *= 2 / polOrder
    return c

def getMaxNCheby(eMin, eMax, eta, maxInterval=0.98):
    """ Computes estimate of required Chebychev expansion for Inv[H-eMin+1j * eta].

    :returns expansion estimate, estimate of chebychev coefficient at that expansion."""
    scale = 2 * maxInterval / (eMax - eMin)  # 1/a = deltaH
    Hbar = eMin + maxInterval / scale  # (eMax + eMin) / 2
    maxNCheby = math.ceil(1.1 / abs(scale * eta))  # 1.1: slightly larger
    chebC = chebyCoeffNumerical(maxNCheby - 1, maxNCheby, lambda x: (x - eMin + 1j * eta)**-1,
                                scale, Hbar)
    return maxNCheby, chebC

def dampJackson(n,N):
    """Gaussian damping"""
    Np = N+1
    g = (Np-n) * np.cos(np.pi * n / Np) / Np
    g += np.sin(np.pi * n / Np) / (np.tan(np.pi/Np) * Np)
    return g

def dampLorentz(n,N, lamda=4):
    """Lorentzian damping"""
    return np.sinh(lamda * (1-n/N)) / np.sinh(lamda)

def dampNothing(*args,**kwargs):
    return 1

dampToFunction = {"jackson": dampJackson,
                  "lorentz": dampLorentz, "none": dampNothing}

def chebyShift(x, scale, eMin, maxInterval):
    return scale * (x - eMin) - maxInterval

def chebyGreensFunction(freqs:np.ndarray, moments:np.ndarray,
                           eta:float, eMin:float, eMax:float,
                           maxInterval=0.98, damping="Jackson",addition=False):
    """ Get (damped) Green's function, G=<A| inv[ H + freqs - eMin + i * eta] |B>, via Chebychev expansion
    using 3 different methods.

    a) Computes Im(G) *without* eta using Kernel Polynomial Method; Ref B, C, E
    b) Computes G using analytical Chebychev expansion; Ref A
    c) Computes G using numerical Chebychev expansion; Ref I

    :param freqs: Frequencies to compute at
    :param moments: Chebychev moments: <A| Tn| B>,
            where Tn is the Chebychev polynomial of inv[ H + freqs - eMin + i * eta]
    :param eta: Damping. Note that this is somewhat redundant with the additional Chebychev damping.
    :param eMin: min( spectrum(H) ); used for the scaling in the Chebychev expansion
    :param eMax: max( spectrum(H) ); used for the scaling in the Chebychev expansion
    :param maxInterval: Interval scaling ([-1,1] or smaller) used for the Chebychev expansion, <= 1
    :param damping: Damping function of the Chebychev expansion. Options are
        'Jackson' for Gaussian damping, 'Lorentz' for Lorentzian damping or 'none' for no damping. See Ref E.
    :param addition: If true, use -H + E0 instead of H - E0
    :return: a), b), c) as np.ndarray
    """
    damp = dampToFunction[damping.lower()]
    assert eMin < eMax

    scale = 2 * maxInterval / (eMax - eMin)  # 1/a = deltaH
    Hbar = eMin + maxInterval / scale  # (eMax + eMin) / 2
    if addition:
        FAC = +freqs + eMin
    else:
        FAC = -freqs + eMin
    FREQSshift = chebyShift(FAC, scale, eMin, maxInterval)
    FREQSshift[FREQSshift < -maxInterval] = maxInterval  # ATTENTION; important to avoid extrapolation
    FREQSshift[FREQSshift > maxInterval] = maxInterval
    #                                     vv zs needs to be complex, for numerical reasons
    zs = chebyShift(FAC + 1j * min(eta,1e-12), scale, eMin, maxInterval) # Ref I
    zs = np.require(zs, dtype=np.complex256)

    Ncheby = len(moments)
    Ts = [np.ones_like(FREQSshift), FREQSshift.copy()]
    for i in range(2, Ncheby):
        Ts.append(2 * FREQSshift * Ts[-1] - Ts[-2])

    numericalCoeffs = np.empty([Ncheby, len(freqs)], dtype=complex)
    for i in range(Ncheby):
        if addition:
            numericalCoeffs[i, :] = -chebyCoeffNumerical(i, Ncheby, lambda x: (x - freqs - eMin + 1j * eta) ** -1,
                                                        scale, Hbar)
        else:
            numericalCoeffs[i, :] = chebyCoeffNumerical(i, Ncheby, lambda x: (x + freqs - eMin + 1j * eta) ** -1,
                                                    scale, Hbar)
    #                                          vv not important but just in case
    spectrumDelta = np.zeros_like(freqs,dtype=np.float128) # Ref B => not whole Green's function
    spectrumNum = np.zeros_like(freqs,dtype=np.complex256) # Numerical; Ref I
    spectrumGreen = np.zeros_like(freqs,dtype=np.complex256) # Ref A
    for n in range(Ncheby):
        if n == 0:
            fac = 1
            div = .5
        else:
            fac = 2
            div = 1.
        g = damp(n, len(moments)-1)
        spectrumDelta += g * moments[n] * Ts[n]
        spectrumNum += g * div * numericalCoeffs[n,:] * moments[n]
        # Green's function formula is a bit ill-conditioned.
        # This should be improvable
        prac = (1 + np.sqrt(zs**2) * np.sqrt(zs**2 - 1)/ zs**2)**(-n)
        prec = (+zs)**(n+1) * np.sqrt(1-zs**-2)
        idxSml = np.abs(prec) == 0 # yes, tiny things matter as well!
        prec[idxSml] = 1
        sumTerm = g * fac * moments[n] * prac / prec
        spectrumGreen[~idxSml] += sumTerm[~idxSml]
    spectrumDelta *= -scale / np.sqrt(1 - FREQSshift)
    spectrumGreen *= scale
    if addition:
        spectrumGreen.real *= -1
        spectrumNum.imag *= -1
    return spectrumDelta, spectrumGreen, spectrumNum

class FT_Cheb_GFDMRG(FTDMRG):
    """
    Finite temperature Chebychev DMRG for Green's Function

    References:
    A) PHYSICAL REVIEW B 90, 165112 (2014) http://dx.doi.org/10.1103/PhysRevB.90.165112
    B) PHYSICAL REVIEW B 83, 195115 (2011) http://dx.doi.org/10.1103/PhysRevB.83.195115
    C) PHYSICAL REVIEW B 97, 075111 (2018) https://doi.org/10.1103/PhysRevB.97.075111
    D) J. Phys. Chem. Lett. 2021, 12, 9344−9352
    E) The kernel polynomial method, http://dx.doi.org/10.1103/RevModPhys.78.275
    F) The Journal of Chemical Physics 103, 2903 (1995); doi: 10.1063/1.470477
    G) The Journal of Chemical Physics 102, 7390 (1995); doi: 10.1063/1.469051
    H) Chen, Guo, Computer Physics Communications 119 (1999) 19-31
    I) Numerical J. Chem. Theory Comput. 2017, 13, 10, 4684–4698; NumRecs, 3rd edition page 234, eq 5.8.7
    """
    def greens_function(self, mps: MPS,
                        eMin: float,
                        eMax: float,
                        maxNCheby: int,
                        idxs: List[int],
                        bond_dim: int,
                        cps_bond_dims: List[int], cps_noises: List[float],
                        cps_conv_tol: float, cps_n_sweeps: float,
                        cheb_bond_dims: List[int], cheb_noises: List[float],
                        cheb_conv_tol: float, cheb_n_sweeps: float,
                        saveDir: Union[str,None],
                        diag_only=False, alpha=True, addition = False,
                        cutoff=1E-14,
                        chebyMaxInterval = 0.98,
                        occs=None, bias=1.0, mo_coeff=None,
                        callback=lambda i,j,iCheby,gf:None,
                        restart=None,
                        ) -> np.ndarray:
        """ Solve for the Green's function using Chebychev expansion
        GF_ij(t) = -i theta(t) <psi0| V_i' exp[i (H-eMin) t] V_j |psi0>
        With V_i = a_i or a'_i. theta(t) is the step function (no time-reversal symm)
        Note the definition of the sign of frequency omega.

        :param mps: Start state psi0
        :param eMin: Minimal energy of H
        :param eMax: Maximal energy of H
        :param maxNCheby: number of Chebychev polynomials used
        :param idxs: GF orbital indices to compute GF_ij, with i,j in idxs
        :param bond_dim: Max bond dimension
        :param cps_bond_dims: Number of bond dimensions for each sweep for V |þsi0>
        :param cps_noises: Noises for each sweep for V |þsi0>
        :param cps_conv_tol:  Sweep convergence tolerance for V |þsi0>
        :param cps_n_sweeps: Number of sweeps for obtaining V |psi0>
        :param cheb_bond_dims: Number of bond dimensions for each sweep for Cheb. polynomials
        :param cheb_noises: Noises for each sweep for Cheb. polynomials
        :param cheb_conv_tol:  Sweep convergence tolerance for Cheb. polynomials
        :param cheb_n_sweeps: Number of sweeps for obtaining Cheb. polynomials
        :param saveDir: Directory for saving the Chebychev polynomials. If None, nothing will be saved (not recommended)
        :param diag_only: Solve only diagonal of GF: GF_ii
        :param cutoff: Bond dimension cutoff for sweeps
        :param chebyMaxInterval: [-W,W] for scaling, where W <= 1. Use of 1-epsilon recommended, where epsilon is small
        :param alpha: Creation/annihilation operator refers to alpha spin (otherwise: beta spin)
        :param addition: If true, use -H + E0 instead of H - E0
        :param occs: Optional occupation number vector for V|psi0> initialization
        :param bias: Optional occupation number bias for V|psi0> initialization
        :param mo_coeff: MPO is in MO basis but GF should be computed in AO basis
        :param callback: Callback function after each GF computation.
                        Called as callback(i,j,iCheby,GF_ij(iCheby))
        :param restart: if not None, used for increasing the Chebychev expansion.
                        For that, it must be the value of the last used maxNCheby
        :return: GF matrix
        """
        ops = [None] * len(idxs)
        rkets = [None] * len(idxs)
        rmpos = [None] * len(idxs)
        if restart is not None:
            assert isinstance(restart, int)
            assert restart < maxNCheby
            assert restart > 2, "Add code"

        if self.mpi is not None:
            self.mpi.barrier()

        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(AncillaMPO(mpo), RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo

        mpoNonHerm = MPOQC(self.hamil, QCTypes.Conventional)
        mpoNonHerm = SimplifiedMPO(AncillaMPO(mpoNonHerm), NoTransposeRule(RuleQC()), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        # ATTENTION: Always scale the same, despite "addition". H always need to be in range [-1,1] ([-W,W])
        mpo = 1.0 * self.mpo_orig
        mpo.const_e -= eMin
        mpoNonHerm = 1.0 * mpoNonHerm
        mpoNonHerm.const_e -= eMin

        mpo = IdentityAddedMPO(mpo)
        mpoNonHerm = IdentityAddedMPO(mpoNonHerm)

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)
            mpoNonHerm = ParallelMPO(mpoNonHerm, self.prule)

        if self.print_statistics:
            _print('FT-CGF MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            mps_info2 = AncillaMPSInfo(self.n_physical_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(bond_dim)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("FT-CGF EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("FT-CGF EST PEAK MEM = ", FT_Cheb_GFDMRG.fmt_size(
                mem2), " SCRATCH = ", FT_Cheb_GFDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        ############################################
        # Prepare creation/annihilation operators
        ############################################
        getChebTag = self.getChebTag
        getChebDir = lambda idx, iCheb: self.getChebDir(idx, iCheb, saveDir)

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
            if restart is not None:
                assert saveDir is not None
                if self.verbose >= 2:
                    _print('>>> LOAD IN phi Site = %4d <<<' % idx)
                state = loadMPSfromDir(None, getChebDir(ii, 0), self.mpi)
                state.info.tag = 'DKET%d' % idx
                state.info.save_mutable()
                state.save_mutable()
                state.save_data()
                rkets[ii] = state
                continue
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
            cps.target_ket_bond_dim  = cps.target_bra_bond_dim = max(cps_bond_dims)
            cps.solve(cps_n_sweeps, mps.center == 0, cps_conv_tol)



            if self.verbose >= 2:
                _print('>>> COMPLETE Compression Site = %4d | Time = %.2f <<<' %
                       (idx, time.perf_counter() - t))

        ############################################
        # Big GF LOOP
        ############################################
        # Cheby stuff
        scale = 2 * chebyMaxInterval / (eMax - eMin)  # 1/a = deltaH
        #Hbar = eMin + chebyMaxInterval / scale  # (eMax + eMin) / 2
        # chebychev moments
        gf_moments = np.zeros((len(idxs), len(idxs), maxNCheby))

        braket = lambda a,b: self.braket(a,b, idMPONonHerm)
        align_mps_center = lambda a,b: self.align_mps_center(a,b, mpoNonHerm)

        mpoScaled = mpoNonHerm * scale  # mpo already has eMin subtracted
        mpoScaled.const_e -= chebyMaxInterval
        mpoScaled2 = 2 * mpoScaled

        idMPONonHerm = SimplifiedMPO(AncillaMPO(IdentityMPO(self.hamil)),
                              NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        minusIdMPONonHerm = -1 * idMPONonHerm
        if self.mpi is not None:
            idMPONonHerm = ParallelMPO(idMPONonHerm, self.identrule)
            minusIdMPONonHerm = ParallelMPO(minusIdMPONonHerm, self.identrule)

        _print('>>> CHEBY LOOP <<<')


        if saveDir is not None:
            tools.mkDir(saveDir)
        def nonDiagPart(ii, idx, iCheb, phi):
            if not diag_only:
                for jj, idx2 in enumerate(idxs):
                    if jj > ii and rkets[jj].info.target == rkets[ii].info.target:
                        currentTime = time.perf_counter()
                        if self.verbose >= 2:
                            _print(
                                f">>> CHEBY LOOP jdx {idx2:4d} idx {idx:4d} cheb-expansion {iCheb:04d} time={currentTime:10.5f}")

                        align_mps_center(phi, rkets[jj])  # always make sure phi is fine
                        gf_moments[jj, ii, iCheb] = gf_moments[ii, jj, iCheb] = braket(rkets[jj], phi)
                        if self.mpi is None or MPI.rank == 0:
                            callback(jj,ii, iCheb, gf_moments[jj,ii,iCheb])

                        if self.verbose >= 1:
                            _print("=== CH-GF (%4d%4d | iCheb = %04d ) = %20.15f === " %
                                   (idx2, idx, iCheb, gf_moments[ii, jj, iCheb]))
                        if self.verbose >= 2:
                            elapsedTime = time.perf_counter() - currentTime
                            _print(f">>> cheb-expansion jdx {idx2:4d} idx {idx:4d} |"
                                   f" elapsed time = {elapsedTime:10.5f}")

        for ii, idx in enumerate(idxs):
            _print(f'>>> CHEBY LOOP idx {idx:4d}')
            # "copy" ket to phi
            iCheb = 0# chebchev expansion coefficient
            if self.mpi is not None:
                self.mpi.barrier()
            phi = rkets[ii].deep_copy(getChebTag(ii, iCheb))
            if self.mpi is not None:
                self.mpi.barrier()
            # Save first one, even though it is the same as rket, just for consistency
            gf_moments[ii,ii, iCheb] = braket(rkets[ii],phi)
            if self.mpi is None or MPI.rank == 0:
                callback(ii, ii, iCheb, gf_moments[ii, ii, iCheb])
            chebDir = getChebDir(ii, iCheb)
            if saveDir is not None:
                tools.mkDir(chebDir)
            if saveDir is not None:
                saveMPStoDir(phi, chebDir, self.mpi)
            nonDiagPart(ii, idx, iCheb, phi)
            # Second one: ~H phi
            iCheb = 1
            if self.mpi is not None:
                self.mpi.barrier()
            phiOld = phi
            phi = phiOld.deep_copy(getChebTag(ii, iCheb))
            if self.mpi is not None:
                self.mpi.barrier()
            chebDir = getChebDir(ii, iCheb)
            if restart is not None:
                if self.verbose >= 2:
                    _print(f">>> CHEBY LOOP idx {idx:4d} cheb-expansion {iCheb:04d} LOAD IN")
                phi = loadMPSfromDir(None, getChebDir(ii, iCheb), self.mpi)
            else:
                rme = MovingEnvironment(mpoScaled, phi, phiOld, "chebFit")
                if self.delayed_contraction:
                    rme.delayed_contraction = OpNamesSet.normal_ops()
                #vv does not work for multiple MEs in Linear
                #rme.cached_contraction = True
                rme.init_environments()

                if len(cheb_noises) == 1 and cheb_noises[0] == 0:
                    pme = None
                else:
                    pme = MovingEnvironment(mpo, phi, phi, "PERT")
                    if self.delayed_contraction:
                        pme.delayed_contraction = OpNamesSet.normal_ops()
                    pme.init_environments(False)
                cps = Linear(pme, rme, VectorUBond(cheb_bond_dims),
                             VectorUBond([phi.info.get_max_bond_dimension() + 100]),
                             VectorDouble(cheb_noises))
                cps.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
                cps.target_ket_bond_dim = cps.target_bra_bond_dim = max(cheb_bond_dims)
                if pme is not None:
                    cps.eq_type = EquationTypes.PerturbativeCompression
                cps.iprint = max(self.verbose - 1, 0)
                cps.cutoff = cutoff

                currentTime = time.perf_counter()
                _print(f">>> CHEBY LOOP idx {idx:4d} cheb-expansion {iCheb:04d} time={currentTime:10.5f}")
                ex = cps.solve(cheb_n_sweeps, phi.center == 0, cheb_conv_tol)
                elapsedTime = time.perf_counter() - currentTime
                _print(f">>> cheb-expansion {iCheb:04d} scaled energy = {ex:18.12f} |"
                       f" elapsed time = {elapsedTime:10.5f}")
                if self.print_statistics and self.verbose >= 2:
                    self.printStatistics()

            align_mps_center(phi, rkets[ii]) # always make sure phi is fine
            if self.mpi is not None:
                self.mpi.barrier()
            if saveDir is not None:
                saveMPStoDir(phi, chebDir, self.mpi)
            gf_moments[ii,ii, iCheb] = braket(rkets[ii],phi)
            nonDiagPart(ii, idx, iCheb, phi)
            #phi.load_mutable()
            phiOldOld = phiOld
            phiOld = phi

            for iCheb in range(2, maxNCheby):
                currentTime = time.perf_counter()
                _print(f">>> CHEBY LOOP idx {idx:4d} cheb-expansion {iCheb:04d} time={currentTime:10.5f}")
                if self.mpi is not None:
                    self.mpi.barrier()
                phi = phiOld.deep_copy(getChebTag(ii, iCheb))
                if self.mpi is not None:
                    self.mpi.barrier()
                chebDir = getChebDir(ii, iCheb)
                if restart is not None and iCheb < restart:
                    if self.verbose >= 2:
                        _print(f">>> CHEBY LOOP idx {idx:4d} cheb-expansion {iCheb:04d} LOAD IN")
                    phi = loadMPSfromDir(None, getChebDir(ii, iCheb), self.mpi)
                else:
                    align_mps_center(phi, rkets[ii])
                    align_mps_center(phiOld, rkets[ii])
                    align_mps_center(phiOldOld, rkets[ii])
                    if self.mpi is not None:
                        self.mpi.barrier()

                    # phi = 2 * H * PhiOld - phiOldOld
                    rme = MovingEnvironment(mpoScaled2, phi, phiOld, "chebFit2")
                    if self.delayed_contraction:
                        rme.delayed_contraction = OpNamesSet.normal_ops()
                    rme.init_environments()

                    tme = MovingEnvironment(minusIdMPONonHerm, phi, phiOldOld, "chebFit3")
                    tme.init_environments()

                    if len(cheb_noises) == 1 and cheb_noises[0] == 0:
                        pme = None
                    else:
                        pme = MovingEnvironment(mpo, phi, phi, "PERT")
                        if self.delayed_contraction:
                            pme.delayed_contraction = OpNamesSet.normal_ops()
                        pme.init_environments(False)
                    cps = Linear(pme, rme, tme, VectorUBond(cheb_bond_dims),
                                 VectorUBond([phi.info.get_max_bond_dimension() + 100]),
                                 VectorDouble(cheb_noises))
                    cps.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
                    # vv important because phiOldOld is only occurring in tme
                    cps.target_ket_bond_dim = cps.target_bra_bond_dim = max(cheb_bond_dims)
                    cps.eq_type = EquationTypes.FitAddition
                    cps.iprint = max(self.verbose - 1, 0)
                    cps.cutoff = cutoff
                    currentTime = time.perf_counter()

                    _print(f">>> CHEBY LOOP idx {idx:4d} cheb-expansion {iCheb:04d} time={currentTime:10.5f}")
                    ex = cps.solve(cheb_n_sweeps, phi.center == 0, cheb_conv_tol)
                    elapsedTime = time.perf_counter() - currentTime
                    _print(f">>> cheb-expansion {iCheb:04d} scaled energy = {ex:18.12f} |"
                           f" elapsed time = {elapsedTime:10.5f}")
                    if self.print_statistics and self.verbose >= 2:
                        self.printStatistics()

                align_mps_center(phi, rkets[ii])  # always make sure phi is fine
                if saveDir is not None:
                    saveMPStoDir(phi, chebDir, self.mpi)
                gf_moments[ii, ii, iCheb] = braket(rkets[ii], phi)
                nonDiagPart(ii, idx, iCheb, phi)
                phiOldOld.info.deallocate()
                phiOldOld = phiOld
                phiOld = phi

        idMPONonHerm.deallocate()
        minusIdMPONonHerm.deallocate()

        if self.print_statistics:
            self.printStatistics()

        return gf_moments

    def braket(self, a, b, mpo):
        assert a.center == b.center
        if a.dot == 2:
            assert a.center == 0 or a.center == a.n_sites-2, "medium not allowed"
        ime = MovingEnvironment(mpo, a, b, "braket")
        ime.init_environments()
        bd = max( a.info.get_max_bond_dimension(), b.info.get_max_bond_dimension() ) + 100
        res = Expect(ime, bd,bd)
        res.iprint = 0
        #                           vv just finish sweep, i.e., don't do actual propagation!
        resv = res.solve(True, a.center!=0)
        if len(res.expectations[0]) == 0 and resv == 0:# resv is 0 due to bug, but should be resolved now
            assert len(res.expectations[-1]) > 0
            resv = res.expectations[-1][0][1]
        return resv

    def getChebTag(self, idx, iCheb):
        return f"cheb{idx:04d}_{iCheb:04d}"

    def getChebDir(self, idx, iCheb, saveDir):
        if saveDir is None:
            return None
        sDir = saveDir + f"/idx_{idx:04d}/cheb_{iCheb:04d}"
        tools.mkDir(sDir)
        return sDir

    def align_mps_center(self, ket, ref, mpo):
        # center itself may be "K" or "C"
        isOk = ket.center == ref.center
        isOk2 = ket.canonical_form[ket.center+1:] == ref.canonical_form[ref.center+1:]
        if not isOk2 and ket.dot == 2 and ket.center == ket.n_sites - 2:
            #vvv 'LLLC' and 'LLLS' happens but this is OK
            isOk2 = ket.canonical_form[-2] == ref.canonical_form[-2]
        isOk = isOk and isOk2 #ket.canonical_form[ket.center+1:] == ref.canonical_form[ref.center+1:]
        isOk = isOk and ket.canonical_form[:ket.center] == ref.canonical_form[:ref.center]
        if isOk:
            return
        if self.mpi is not None:
            self.mpi.barrier()
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
        return

    def greens_function_via_krylov_space(self,
                                         freqs:np.ndarray,
                                         eMin: float,
                                         eMax: float,
                                         eta:float,
                                         maxNCheby: int,
                                         idxs: List[int],
                                         saveDir: str,
                                         chebyMaxInterval=0.98,
                                         useUnscaledHamiltonian=True,
                                         addition=False,
                                         ):
        """ Make use of Chebychev polynomials computed from `greens_function`
        by building Krylov space and computing the GF in that space

        :param addition: If true, use -H + E0 instead of H - E0
        """
        # Note: I recompute H and S
        #  I could extract them from the polynomials, but
        #  unscaling the scaled Hamiltonian is numerically ill-conditioned.
        assert self.mpo_orig is not None
        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(AncillaMPO(mpo), NoTransposeRule(RuleQC()), True, True,
                                   OpNamesSet((OpNames.R, OpNames.RD)))
        mpo = IdentityAddedMPO(mpo)
        idMPO = SimplifiedMPO(AncillaMPO(IdentityMPO(self.hamil)),
                                     NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)
            idMPO = ParallelMPO(idMPO, self.identrule)

        Nidx = len(idxs)
        Hs = np.empty([Nidx,Nidx, maxNCheby, maxNCheby])
        Ss = np.empty([Nidx,Nidx, maxNCheby, maxNCheby])
        GF = np.empty([Nidx,Nidx, len(freqs)], dtype=complex)
        rkets = [ loadMPSfromDir(None, self.getChebDir(ii, 0, saveDir), self.mpi) for ii in range(Nidx)]
        if useUnscaledHamiltonian:
            scale = 2 * chebyMaxInterval / (eMax - eMin)  # 1/a = deltaH
        if addition:
            freqs = -freqs

        _print(f'>>> GREENS FUNCTION VIA KRYLOV SPACE')
        for ii, idx in enumerate(idxs):
            for jj, jdx in enumerate(idxs):
                if jj >= ii and rkets[jj].info.target == rkets[ii].info.target:
                    currentTime = time.perf_counter()
                    _print(f">>> CHEBY LOOP idx {idx:4d} jdx {jdx:4d} time={currentTime:10.5f}")
                    for iCheb in range(maxNCheby):
                        bra = loadMPSfromDir(None, self.getChebDir(ii, iCheb, saveDir), self.mpi) #\
                            #if iCheb > 0 else rkets[ii]
                        for jCheb in range(iCheb,maxNCheby):
                            ket = loadMPSfromDir(None, self.getChebDir(jj, jCheb, saveDir), self.mpi) #\
                                #if jCheb > 0 else rkets[jj]
                            if self.mpi is not None:
                                self.mpi.barrier()
                            self.align_mps_center(ket,bra, mpo)
                            if not useUnscaledHamiltonian or iCheb == maxNCheby-1 or jCheb == maxNCheby-1:
                                # TODO: add option to use maxNcheby-1 space
                                hv = self.braket(bra, ket, mpo)
                                Hs[ii,jj,iCheb,jCheb] = Hs[jj,ii,iCheb,jCheb] = \
                                    Hs[ii,jj,jCheb,iCheb] = Hs[jj,ii,jCheb,iCheb] = hv
                            sv = self.braket(bra, ket, idMPO)
                            Ss[ii,jj,iCheb,jCheb] = Ss[jj,ii,iCheb,jCheb] = \
                                Ss[ii,jj,jCheb,iCheb] = Ss[jj,ii,jCheb,iCheb] = sv
                    if useUnscaledHamiltonian:
                        # H T0 = T1 ; Tn+1 = 2H Tn - Tn-1
                        # => <i|H|j> = .5 [ <i|j+1> + <i|j-1>
                        for i in range(1,maxNCheby - 1):
                            #              vv H[0,j] is well defined but H3[j,0] is not, due to j-1
                            for j in range(i, maxNCheby - 1):
                                Hs[ii,jj,i, j] = Hs[ii,jj, j, i] = .5 * (Ss[ii,jj,i, j + 1] + Ss[ii,jj,i, j - 1])
                        # H |0> = |1>
                        Hs[ii,jj,:, 0] = Hs[ii,jj,0,:] = Hs[jj,ii,0,:] = Hs[jj,ii,0,:] = Ss[ii,jj, :, 1]
        # do it
        if self.mpi is None or MPI.rank == 0:
            for ii, idx in enumerate(idxs):
                for jj, jdx in enumerate(idxs):
                    if jj >= ii and rkets[jj].info.target == rkets[ii].info.target:
                        H = Hs[ii,jj]
                        S = Ss[ii,jj]
                        s, uvS = la.eigh(-S) # largest first
                        s *= -1
                        s[s < 0] = 0 # numerics
                        s = np.sqrt(s) # matches SVD values; uvS = v.T
                        rank = np.sum(s / np.max(s) > 1e-8)
                        v = uvS[:,:].T.conj()
                        T = v[:rank,:].T.conj() @ np.diag(s[:rank]**-1) @ v[:rank,:]
                        Hc = T.T.conj() @ H @ T
                        _print(ii,jj,"Cheby Krylov space: rank=", rank, "max=", maxNCheby)
                        ket = T.T.conj() @ Ss[ii,ii,:,0] # psi is identical to 1st. Cheby polynomial
                        bra = T.T.conj() @ Ss[jj,jj,:,0]
                        HcVar = np.empty_like(Hc,dtype=complex)
                        for io, omega in enumerate(freqs):
                            HcVar[:,:] = Hc
                            if not useUnscaledHamiltonian:
                                HcVar[np.diag_indices_from(Hc)] += (omega - eMin + 1j * eta)
                            else:
                                # scale
                                ooo = scale * ( (omega - eMin + 1j * eta) - eMin ) - chebyMaxInterval
                                HcVar[np.diag_indices_from(Hc)] += ooo
                                # TODO scale back later?
                            c = la.solve(HcVar, ket,overwrite_a=True)
                            GF[ii,jj,io] = GF[jj,ii,io] = np.vdot(bra,c)


        idMPO.deallocate()
        mpo.deallocate()
        if addition:
            GF.real *= -1
        return GF, Hs, Ss



    def printStatistics(self):
        dmain, dseco, imain, iseco = Global.frame.peak_used_memory
        _print("CH-GF PEAK MEM USAGE:",
               "DMEM = ", FT_Cheb_GFDMRG.fmt_size(dmain + dseco),
               "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
               "IMEM = ", FT_Cheb_GFDMRG.fmt_size(imain + iseco),
               "(%.0f%%)" % (imain * 100 / (imain + iseco)))



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
    eMin = -2.1379703474141984 #H4
    eMax = -eMin
    point_group = 'c1'


    MAX_M = 50
    gs_bond_dims=[MAX_M]
    gs_noises = [1E-3, 5E-4]
    gs_tol = 1E-4
    gf_bond_dims = gs_bond_dims
    gf_noises = [1E-3, 5E-4, 0]
    occs = None
    bias = 1
    cutoff = 1e-14
    gf_n_sweeps = 10
    gs_n_steps = 20
    cps_bond_dims=[MAX_M] # Bond dimension of \hat a_i |psi> *and* |psi> (in time evolution)
    cps_noises = [0]
    cps_tol = 1E-10
    cps_n_sweeps = 30
    solver_tol = 1e-8
    solver_tol = 1e-12

    beta = 80 # inverse temperature
    dbeta = 0.5 # "time" step
    mu = -0.026282794560 # Chemical potential for initial state preparation

    dmrg = FT_Cheb_GFDMRG(scratch=scratch, memory=4e9,
                       verbose=3, omp_threads=n_threads)
    dmrg.init_hamiltonian_fcidump(point_group, "fcidump")
    #mps, mu = dmrg.optimize_mu(dmrg.fcidump.n_elec,mu, beta, dbeta, MAX_M)
    mps = dmrg.prepare_ground_state(mu, beta, dbeta, MAX_M)[0]

    eta = 0.005
    FREQS = np.linspace(-2, 2, 200)
    idxs = [0,1,2]  # Calc S_ii
    alpha = True  # alpha or beta spin
    nCheby = getMaxNCheby(eMin, eMax, eta) # ATTENTION: Much fewer are needed for Krylov-Space method

    saveDir ="chebMPSs"
    addition = False
    gf = dmrg.greens_function(mps, eMin, eMax, nCheby, idxs,
                              50,
                              cps_bond_dims, cps_noises, cps_tol, cps_n_sweeps,
                              gf_bond_dims, gf_noises, solver_tol, gf_n_sweeps,
                              saveDir,
                              mo_coeff = None,
                              addition=addition,
                              diag_only=False,
                              alpha=alpha,
                              #        restart=4,
                              )
    fOut = open("ft_chebdmrg_freqs_block2.dat", "w")
    fOut.write("# idx jdx omega  Re(gfGreen)  Im(gfGreen)  delta Re(gfNum)  Im(gfNum)\n")
    for ii, idx in enumerate(idxs):
        for jj, jdx in enumerate(idxs):
            if jj >= ii:
                spectrumDelta, spectrumGreen, spectrumNum = \
                        chebyGreensFunction(FREQS, gf[ii,jj],
                                eta, eMin,eMax, addition=addition)
                for io, omega in enumerate(FREQS):
                    resA = spectrumGreen[io]
                    resB = spectrumDelta[io] #* np.pi
                    resNum = spectrumNum[io]
                    fOut.write(f"{ii:2d} {jj:2d} {omega:16.7f}   {-resA.real:16.7f}  {resA.imag:16.7f} {resB:16.7f}"
                               f"{resNum.real:16.7f} {resNum.imag:16.7f}\n")
                    fOut.flush()
    fOut.close()

    gf2, Hs, Ss = dmrg.greens_function_via_krylov_space(FREQS,  eMin, eMax, eta, nCheby, idxs, saveDir,
                                                        useUnscaledHamiltonian=True, addition=addition)
    fOut = open("ft_chebdmrg_freqs_block2_krylov.dat", "w")
    fOut.write("# idx jdx omega  Re(gf)  Im(gf)\n")
    for ii, idx in enumerate(idxs):
        for jj, jdx in enumerate(idxs):
            if jj >= ii:
                for io, omega in enumerate(FREQS):
                    resA = gf2[ii,jj,io]
                    fOut.write(f"{ii:2d} {jj:2d} {omega:16.7f}   {resA.real:16.7f}  {resA.imag:16.7f} \n")
                    fOut.flush()
    fOut.close()
