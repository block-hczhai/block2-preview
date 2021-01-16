#! /usr/bin/env python

import sys
from libdmet_solid.solver.settings import BLOCK2PATH 
sys.path[:0] = [BLOCK2PATH + "/build"]

from block2 import SZ, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, DoubleFPCodec
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, OpNames
from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
import numpy as np
import time

from parser import parse, read_integral

np.set_printoptions(3, linewidth=1000, suppress=True)
if len(sys.argv) > 1:
    fin = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "pre":
        pre_run = True
    else:
        pre_run = False
else:
    raise ValueError("usage: pre.py dmrg.conf")

if pre_run:
    MPI = None
    _print = print
else:
    from block2.sz import MPICommunicator
    MPI = MPICommunicator()
    from mpi4py import MPI as PYMPI
    comm = PYMPI.COMM_WORLD

    def _print(*args, **kwargs):
        if MPI.rank == 0:
            print(*args, **kwargs)

tx = time.perf_counter()

Random.rand_seed(1234)

dic = parse(fin)
scratch = dic.get("prefix", "./node0/")
restart_dir = dic.get("prefix_restart", "./node0_restart/")
n_threads = int(dic.get("num_thrds", 28))
bond_dims, dav_thrds, noises = dic["schedule"]
sweep_tol = float(dic.get("sweep_tol", 1e-6))

if MPI is not None and MPI.rank == 0:
    import os
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    if not os.path.isdir(restart_dir):
        os.mkdir(restart_dir)
    os.environ['TMPDIR'] = scratch
if MPI is not None:
    MPI.barrier()

memory = int(int(dic.get("mem", "40").split()[0]) * 1e9)

init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
# ZHC NOTE nglobal_threads, nop_threads, MKL_NUM_THREADS
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
    prule_npdm = ParallelRuleNPDMQC(MPI)

if pre_run:
    nelec = int(dic["nelec"])
    spin = int(dic["spin"])
    fints = dic["orbitals"]
    if fints[-7:] == "FCIDUMP":
        fcidump = FCIDUMP()
        fcidump.read(fints)
        # FIXME
        #fcidump.params["nelec"] = str(nelec)
        #fcidump.params["ms2"] = str(spin)
    else:
        fcidump = read_integral(fints, nelec, spin)

    _print("read integral finished", time.perf_counter() - tx)

    vacuum = SZ(0)
    target = SZ(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))
    n_sites = fcidump.n_sites
    orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
    hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)

    bond_dim = bond_dims[0]
    mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
    mps_info.tag = "KET"
    # we can use gsdmrg.py line 247
    mps_info.set_bond_dimension(bond_dim)
    mps_info.save_data(scratch + '/mps_info.bin')
    mps = MPS(n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()
else:
    mps_info = MPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = "KET"
    mps_info.set_bond_dimension(bond_dims[0])
    mps = MPS(mps_info.n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

if pre_run:
    _print("build mpo", time.perf_counter() - tx)
    mpo = MPOQC(hamil, QCTypes.Conventional)
    _print("simpl mpo", time.perf_counter() - tx)
    mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    _print("simpl mpo finished", time.perf_counter() - tx)

    _print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

    mpo.save_data(scratch + '/mpo.bin')
    # ZHC NOTE 1pdm
    _print("build mpo", time.perf_counter() - tx)
    pmpo = PDM1MPOQC(hamil)
    _print("simpl mpo", time.perf_counter() - tx)
    pmpo = SimplifiedMPO(pmpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    _print("simpl mpo finished", time.perf_counter() - tx)

    _print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in pmpo.right_operator_names]))

    pmpo.save_data(scratch + '/mpo-1pdm.bin')
else:
    mpo = MPO(0)
    mpo.load_data(scratch + '/mpo.bin')
    mpo.tf = TensorFunctions(OperatorFunctions(CG()))

    _print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
    
    pmpo = MPO(0)
    pmpo.load_data(scratch + '/mpo-1pdm.bin')
    pmpo.tf = TensorFunctions(OperatorFunctions(CG()))

    _print('1PDM MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in pmpo.left_operator_names]))
    
    if MPI is not None:
        from block2.sz import ParallelMPO
        mpo = ParallelMPO(mpo, prule)
        pmpo = ParallelMPO(pmpo, prule_npdm)

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
    E_dmrg = dmrg.solve(len(bond_dims), mps.center == 0, sweep_tol)
    
    if (MPI is not None and MPI.rank == 0) or MPI is None:
        np.save(scratch + "E_dmrg.npy", E_dmrg)
    _print("DMRG finished.")
    
    # ZHC FIXME twodot to onedot
    me = MovingEnvironment(pmpo, mps, mps, "1PDM")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.save_partition_info = True
    me.init_environments(True)

    _print("env init finished", time.perf_counter() - tx)

    expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
    # ZHC NOTE True do sweep.
    expect.solve(True, mps.center == 0)

    if (MPI is not None and MPI.rank == 0) or MPI is None:
        dmr = expect.get_1pdm(me.n_sites)
        dm = np.array(dmr).copy()
        dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
        dm = np.transpose(dm, (0, 2, 1, 3))
        dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

        np.save(scratch + "1pdm.npy", dm)

    mps_info.deallocate()
    


