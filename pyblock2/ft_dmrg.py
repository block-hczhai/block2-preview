
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

""" Finite temperature DMRG

:author: Henrik R. Larsson, Sep 2021
        Based on zero temperature GFDMRG from Huanchen Zhai
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16, TETypes
import time

def _init(SpinLabel):
    if SpinLabel == SU2:
        from block2.su2 import AncillaMPO, AncillaMPSInfo
        from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
        from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO, IdentityAddedMPO
        from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
        from block2.su2 import VectorOpElement, LocalMPO
        from block2.su2 import TimeEvolution

        try:
            from block2.su2 import MPICommunicator
            hasMPI = True
        except ImportError:
            hasMPI = False
    else:
        from block2.sz import AncillaMPO, AncillaMPSInfo
        from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
        from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO, IdentityAddedMPO
        from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
        from block2.sz import VectorOpElement, LocalMPO
        from block2.sz import TimeEvolution

        try:
            from block2.sz import MPICommunicator
            hasMPI = True
        except ImportError:
            hasMPI = False
    import tools; tools.init(SpinLabel)
    from tools import saveMPStoDir, loadMPSfromDir
    import numpy as np
    from typing import List

    if hasMPI:
        MPI = MPICommunicator()
    else:
        class _MPI:
            rank = 0
        MPI = _MPI()



    _print = tools.getVerbosePrinter(MPI.rank == 0, flush=True)


    class FTDMRG:
        """
        Finite-temperature DMRG
        """

        def __init__(self, scratch='./nodex', memory=1 * 1E9, omp_threads=8, verbose=2,
                     print_statistics=True, mpi=None, delayed_contraction=True):
            """
            :param scratch: Used scratchdir
            :param memory: Stackmemory in bytes
            :param omp_threads: Number of omp threads
            :param verbose: Detail of print statements: 0 (quiet), 2 (per sweep), 3 (per iteration)
            :param print_statistics: Print memory statistics before/ after each call
            :param mpi: MPI class
            :param delayed_contraction: Use of it in MovingEnvironment (recommended if there are not multiple MEs)
            """
            self.fcidump = None
            self.hamil = None
            self.verbose = verbose
            self.scratch = scratch
            self.mpo_orig = None
            self.print_statistics = print_statistics
            self.mpi = mpi
            self.delayed_contraction = delayed_contraction
            self.idx = None # reorder
            self.ridx = None # inv reorder
            self.swap_pg = None # PointGroup.swap_XX; initialized in init_fcidump*
            self.orb_sym = None
            self.target = None
            self.n_physical_sites = None
            self.n_sites = None

            Random.rand_seed(0)
            isize = int(1e8) # hrl: typically <  200MB
            init_memory(isize=isize, dsize=int(memory), save_dir=scratch)
            Global.threading = Threading(
                ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
            Global.threading.seq_type = SeqTypes.Tasked
            Global.frame.load_buffering = False
            Global.frame.save_buffering = False
            Global.frame.use_main_stack = False
            Global.frame.minimal_disk_usage = True

            if self.verbose >= 2:
                _print(Global.frame)
                _print(Global.threading)

            if mpi is not None:
                if SpinLabel == SU2:
                    from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelRuleSiteQC
                    from block2.su2 import ParallelRuleSiteQC, ParallelRuleIdentity
                else:
                    from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
                    from block2.sz import ParallelRuleSiteQC, ParallelRuleIdentity
                self.prule = ParallelRuleQC(mpi)
                self.pdmrule = ParallelRuleNPDMQC(mpi)
                self.siterule = ParallelRuleSiteQC(mpi)
                self.identrule = ParallelRuleIdentity(mpi)
            else:
                self.prule = None
                self.pdmrule = None
                self.siterule = None
                self.identrule = None

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

        def init_hamiltonian_fcidump(self, pointgroup: str, filename: str, idx=None):
            """Read integrals from FCIDUMP file

            :param pointgroup: Used point group  (c1, c2v,..)
            :param filename: FCIDUMP file
            :param idx: Optional orbital reordering index
            """
            assert self.fcidump is None
            self.fcidump = FCIDUMP()
            self.fcidump.read(filename)
            if idx is not None:
                self.fcidump.reorder(VectorUInt16(idx))
                self.idx = idx
                self.ridx = np.argsort(idx)
            self.swap_pg = getattr(PointGroup, "swap_" + pointgroup)
            self.orb_sym = VectorUInt8(
                map(self.swap_pg, self.fcidump.orb_sym))

            vacuum = SpinLabel(0)
            # hrl: twos should be 0 for thermal state; and n_elec = n_sites (2*n_sites_physical)
            self.target = SpinLabel(2 * self.fcidump.n_sites, 0, self.swap_pg(self.fcidump.isym))
            self.n_physical_sites = self.fcidump.n_sites
            self.n_sites = self.fcidump.n_sites * 2

            self.hamil = HamiltonianQC(
                vacuum, self.n_physical_sites, self.orb_sym, self.fcidump)

        def init_hamiltonian(self, n_elec: int, twos: int, isym: int, orb_sym: List[int],
                             e_core: float, h1e: np.ndarray, g2e: np.ndarray, tol=1E-13, idx=None,
                             save_fcidump=None):
            """ Initialize fcidump based on integrals

            :param pointgroup: Used point group  (c1, c2v,..)
            :param n_elec: Total number of electrons
            :param twos: 2S quantum number
            :param isym: Wavefunction point group symmetry
            :param orb_sym: Orbital symmetry in *molpro* notation (not PySCF XOR)
            :param e_core: Core energy
            :param h1e: One-electron integrals
            :param g2e: Two-electron integrals
            :param tol:  All integral values below tol will be set to zero
            :param idx: Optional orbital reordering index
            :param save_fcidump: Use this file to write an fcidump file (for later usage)
            """
            n_sites = h1e.shape[0]
            assert self.fcidump is None
            self.fcidump = FCIDUMP()
            self.swap_pg = getattr(PointGroup, "swap_" + pointgroup)
            if not isinstance(h1e, tuple):
                mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
                k = 0
                for i in range(0, n_sites):
                    for j in range(0, i + 1):
                        assert abs(h1e[i, j] - h1e[j, i]) < tol
                        mh1e[k] = h1e[i, j]
                        k += 1
                mg2e = g2e.ravel()
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
                mg2e = tuple(xg2e.ravel() for xg2e in g2e)
                for xmg2e in mg2e:
                    xmg2e[np.abs(xmg2e) < tol] = 0.0
                self.fcidump.initialize_sz(
                    n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
            assert not np.any(np.array(orb_sym) == 0), "orb_sym should be in molpro notation"
            self.fcidump.orb_sym = VectorUInt8(orb_sym)
            if idx is not None:
                self.fcidump.reorder(VectorUInt16(idx))
                self.idx = idx
                self.ridx = np.argsort(idx)
            self.orb_sym = VectorUInt8(map(self.swap_pg, self.fcidump.orb_sym))

            vacuum = SpinLabel(0)
            self.target = SpinLabel(2 * self.fcidump.n_sites, 0, self.swap_pg(self.fcidump.isym))
            self.n_physical_sites = self.fcidump.n_sites
            self.n_sites = self.fcidump.n_sites * 2

            self.hamil = HamiltonianQC(vacuum, self.n_sites, self.orb_sym, self.fcidump)

            if save_fcidump is not None:
                if self.mpi is None or self.mpi.rank == 0:
                    self.fcidump.orb_sym = VectorUInt8(orb_sym)
                    self.fcidump.write(save_fcidump)
                if self.mpi is not None:
                    self.mpi.barrier()

        def prepare_ground_state(self, mu: float,
                                 beta: float, dbeta: float,
                                 bond_dim: int,
                                 save_dir=None,
                                 tag="psi_t0", dot=2,
                                 n_sub_sweeps_init=6, n_sub_sweeps=2) -> MPS:
            """ Get the initial ground state by propagating a maximally entangled state until beta
            exp(-beta H) |max_entangled>
            Currently uses RK4 algorithm

            :param mu: Chemical potential added during the propagation (switched off afterward)
            :param beta: Inverse temperature
            :param dbeta: "time" step
            :param bond_dim: Max bond dimension for MPS
            :param save_dir: If not None, save final MPS to this directory
            :param tag: MPS tag
            :param dot: MPS dot (2 or 1)
            :param n_sub_sweeps_init: Initial number of sweeps for RK4 algorithm
            :param n_sub_sweeps:  Number of sweeps for RK4 algorithm
            :return: mps
            """
            assert self.fcidump is not None, "call init_hamiltonian first"
            mps_info = AncillaMPSInfo(self.n_physical_sites, self.hamil.vacuum, self.target, self.hamil.basis)
            mps_info.tag = tag
            mps_info.set_thermal_limit()
            mps_info.save_mutable()
            mps_info.deallocate_mutable()

            assert dot == 2 or dot == 1
            mps = MPS(self.n_sites, self.n_sites - dot, dot)
            mps_info.load_mutable()
            mps.initialize(mps_info)
            mps.fill_thermal_limit()
            mps.canonicalize()

            mps.save_mutable()
            mps.deallocate()
            mps_info.deallocate_mutable()
            mps.save_data()


            # Propagation
            if self.verbose > 0:
                _print("##############################")
                _print("# Initial MPS propagation")
            n_steps = int( ( beta / 2) / dbeta + 1)  # propagate until beta/2 (ancillary system)
            dbeta = (beta / 2) / (n_steps + 1) # make sure we have the correct step
            if self.verbose > 0:
                _print("# # time evolution steps:", n_steps)
                _print("# used beta step:", dbeta)
            # MPO
            self.hamil.mu = mu
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(AncillaMPO(mpo), RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            mpo = IdentityAddedMPO(mpo) # hrl: sometimes const_e causes trouble for AncillaMPO
            # TE
            me = MovingEnvironment(mpo, mps, mps, "ground_state_prep")
            if self.delayed_contraction:
                me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
            me.init_environments(False)
            te = TimeEvolution(me, VectorUBond([bond_dim]), TETypes.RK4, n_sub_sweeps_init)
            te.iprint = self.verbose
            te.solve(n_steps, dbeta, mps.center == 0)
            if n_steps != 1:
                # after the first beta step, use 2 sweeps (or 1 sweep) for each beta step
                te.solve(n_steps-1, dbeta, mps.center == 0, n_sub_sweeps)

            mpo.deallocate()
            self.hamil.mu = 0
            if save_dir is not None:
                saveMPStoDir(mps, save_dir)
            _print("# done")
            _print("##############################")
            return mps

        def __del__(self):
            if self.hamil is not None:
                self.hamil.deallocate()
            if self.fcidump is not None:
                self.fcidump.deallocate()
            if self.mpo_orig is not None:
                self.mpo_orig.deallocate()
            release_memory()

    return FTDMRG

FTDMRG_SZ = _init(SZ)
FTDMRG_SU2 = _init(SU2)
