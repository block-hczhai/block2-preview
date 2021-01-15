
import sys
sys.path[:0] = ["../../../build"]

from block2 import SZ, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, DoubleFPCodec
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, ParallelCommTypes, TruncationTypes
from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, ParallelMPS
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
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
    MPI = None
    _print = print

tx = time.perf_counter()

Random.rand_seed(1234)
scratch = '/scratch/global/hczhai/cpr2-07'
restart_dir = '/scratch/global/hczhai/cpr2-07-restart'
load_dir = '/scratch/global/hczhai/cpr2-03-restart-sw09'
n_threads = 28
bond_dims = [3500] * 5 + [3000] * 4 + [2500] * 4 + [2000] * 4 + [1500] * 4
noises = [0] * 21
dav_thrds = [1E-9] * 21

if MPI is not None and MPI.rank == 0:
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
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, 28, 28, 1)
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

mps_info = MPSInfo(0)
mps_info.load_data('../prepare2/mps_info.bin')
mps_info.tag = "KET"
mps_info.load_mutable()
mps = MPS(mps_info)
mps.load_data()
mps.load_mutable()

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

mpo = MPO(0)
mpo.load_data('../prepare2/mpo.bin')
mpo.tf = TensorFunctions(OperatorFunctions(CG()))

_print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

if MPI is not None:
    from block2.sz import ParallelMPO
    mpo = ParallelMPO(mpo, prule)

_print("para mpo finished", time.perf_counter() - tx)

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
dmrg.solve(21, mps.center == 0, tol=1E-10)

mps_info.deallocate()
