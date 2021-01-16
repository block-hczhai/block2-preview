#! /usr/bin/env python

import sys
from libdmet_solid.solver.settings import BLOCK2PATH 
sys.path[:0] = [BLOCK2PATH + "/build"]

from block2 import SZ, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, DoubleFPCodec
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, ParallelCommTypes, TruncationTypes
from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, ParallelMPS
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
import numpy as np
import time

try:
    from block2.sz import MPICommunicator
    MPI = MPICommunicator()
    from mpi4py import MPI as PYMPI
    comm = PYMPI.COMM_WORLD

    def _print(*args, **kwargs):
        if MPI.rank == 0:
            print(*args, **kwargs)
except ImportError:
    raise ValueError("MPI import fails.") 
    MPI = None
    _print = print

tx = time.perf_counter()

Random.rand_seed(1234)
# ZHC NOTE all op, MPS ...
# from prefix parser
scratch = '/scratch/global/hczhai/cpr2-01'
# ZHC NOTE only MPS
restart_dir = '/scratch/global/hczhai/cpr2-01-restart'
n_threads = 28
bond_dims = [800] * 5 + [1200] * 5 + [1500] * 5 + [1800] * 5 + [2500] * 5 + [3000] * 5
noises = [1E-4] * 10 + [1E-5] * 5 + [1E-6] * 5 + [1E-7] * 20 + [0]
dav_thrds = [1E-5] * 10 + [1E-6] * 5 + [1E-7] * 5 + [1E-8] * 20 + [0]

if MPI is not None and MPI.rank == 0:
    import os
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    if not os.path.isdir(restart_dir):
        os.mkdir(restart_dir)
    os.environ['TMPDIR'] = scratch
if MPI is not None:
    MPI.barrier()

memory = int(100 * 1E9)
init_memory(isize=int(1E9), dsize=int(memory), save_dir=scratch)
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, n_threads, n_threads, 1)
Global.threading.seq_type = SeqTypes.Tasked
Global.frame.restart_dir = restart_dir
Global.frame.fp_codec = DoubleFPCodec(1E-20, 1024)
Global.frame.load_buffering = False
Global.frame.save_buffering = False
Global.frame.use_main_stack = False
_print(Global.frame)
_print(Global.threading)

if MPI is not None:
    from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
    prule = ParallelRuleQC(MPI)

# occs = np.diag(np.load('rdm1_ccsd_mp2_basis.npy')[0] + np.load('rdm1_ccsd_mp2_basis.npy')[1])
# occs = VectorDouble(np.array(occs))
# _print(["%.4f" % x for x in occs])
# _print(sum(occs))

mps_info = MPSInfo(0)
# ZHC NOTE from the prefix folder
mps_info.load_data('../prepare/mps_info.bin')
mps_info.tag = "KET"
# ZHC NOTE use the 1st iter M
mps_info.set_bond_dimension(500)
mps = MPS(mps_info.n_sites, 0, 2)
mps.initialize(mps_info)
mps.random_canonicalize()

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

mpo = MPO(0)
# ZHC NOTE prefix
mpo.load_data('../prepare/mpo.bin')
mpo.tf = TensorFunctions(OperatorFunctions(CG()))

_print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

if MPI is not None:
    from block2.sz import ParallelMPO
    mpo = ParallelMPO(mpo, prule)

_print("para mpo finished", time.perf_counter() - tx)

mps.save_mutable()
mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()

me = MovingEnvironment(mpo, mps, mps, "DMRG")
me.delayed_contraction = OpNamesSet.normal_ops()
me.cached_contraction = True
me.save_partition_info = True
me.init_environments(True)

_print("env init finished", time.perf_counter() - tx)

dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
dmrg.decomp_type = DecompositionTypes.DensityMatrix
dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
# ZHC NOTE maxiter, sweep_tol from parser
dmrg.solve(36, mps.center == 0, 1e-6)

mps_info.deallocate()
