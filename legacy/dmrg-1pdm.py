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
from block2.sz import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
import numpy as np
import time
import shutil

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
scratch = '/scratch/global/hczhai/cpr2-11'
restart_dir = '/scratch/global/hczhai/cpr2-11-restart'
load_dir = '/scratch/global/hczhai/cpr2-03-restart-sw17'
n_threads = 28
noises = [1E-4] * 5 + [1E-5] * 5 + [1E-6] * 5 + [1E-7] * 20 + [0]

if (MPI is not None and MPI.rank == 0) or MPI is None:
    import os
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    if not os.path.isdir(restart_dir):
        os.mkdir(restart_dir)
    os.environ['TMPDIR'] = scratch
    for k in os.listdir(load_dir):
        if '.KET.' in k:
            shutil.copy(load_dir + "/" + k, scratch + "/" + k)
if MPI is not None:
    MPI.barrier()

memory = int(100 * 1E9)
init_memory(isize=int(1E9), dsize=int(memory), save_dir=scratch)
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, 36, 36, 1)
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
    prule = ParallelRuleNPDMQC(MPI)

mps_info = MPSInfo(0)
mps_info.load_data('../prepare2/mps_info.bin')
mps_info.tag = "KET"
mps_info.load_mutable()
# should be the last M in schedule
mps_info.bond_dim = 4500
mps = MPS(mps_info)
mps.load_data()
mps.load_mutable()

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

mpo = MPO(0)
mpo.load_data('../prepare2/mpo-1pdm.bin')
mpo.tf = TensorFunctions(OperatorFunctions(CG()))

_print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

if MPI is not None:
    from block2.sz import ParallelMPO
    mpo = ParallelMPO(mpo, prule)

_print("para mpo finished", time.perf_counter() - tx)

mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()

me = MovingEnvironment(mpo, mps, mps, "1PDM")
me.delayed_contraction = OpNamesSet.normal_ops()
me.cached_contraction = True
me.save_partition_info = True
me.init_environments(True)

_print("env init finished", time.perf_counter() - tx)

expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
# True do sweep.
expect.solve(True, mps.center == 0)

if (MPI is not None and MPI.rank == 0) or MPI is None:
    dmr = expect.get_1pdm(me.n_sites)
    dm = np.array(dmr).copy()
    dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
    dm = np.transpose(dm, (0, 2, 1, 3))
    dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

    np.save("1pdm-mo-basis-m4500-1.npy", dm)

mps_info.deallocate()
