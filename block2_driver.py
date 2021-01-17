#! /usr/bin/env python
"""
block2 wrapper.

Author:
    Huanchen Zhai
    Zhi-Hao Cui
"""

from block2 import SZ, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, DoubleFPCodec
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, OpNames
from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
import numpy as np
import time
import os
import sys

from parser import parse, read_integral

DEBUG = True

if len(sys.argv) > 1:
    fin = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "pre":
        pre_run = True
    else:
        pre_run = False
    if len(sys.argv) > 2 and sys.argv[2] == "run":
        no_pre_run = True
    else:
        no_pre_run = False
else:
    raise ValueError("""
        Usage: either:
            (A) python block_driver.py dmrg.conf
            (B) Step 1: python block_driver.py dmrg.conf pre
                Step 2: python block_driver.py dmrg.conf run
    """)

# MPI
from block2.sz import MPICommunicator
MPI = MPICommunicator()
from mpi4py import MPI as PYMPI
comm = PYMPI.COMM_WORLD
def _print(*args, **kwargs):
    if MPI.rank == 0:
        print(*args, **kwargs)

tx = time.perf_counter()

# input parameters
Random.rand_seed(1234)
dic = parse(fin)
if DEBUG:
    _print("\n" + "*" * 34 + " INPUT START " + "*" * 34)
    for key, val in dic.items():
        _print ("%-25s %40s" % (key, val))
    _print("*" * 34 + " INPUT END   " + "*" * 34 + "\n")

scratch = dic.get("prefix", "./node0/")
n_threads = int(dic.get("num_thrds", 28))
bond_dims, dav_thrds, noises = dic["schedule"]
sweep_tol = float(dic.get("sweep_tol", 1e-6))

if MPI is not None and MPI.rank == 0:
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    os.environ['TMPDIR'] = scratch
if MPI is not None:
    MPI.barrier()

# global settings
memory = int(int(dic.get("mem", "40").split()[0]) * 1e9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
# ZHC NOTE nglobal_threads, nop_threads, MKL_NUM_THREADS
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, n_threads, n_threads, 1)
Global.threading.seq_type = SeqTypes.Tasked
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

# prepare hamiltonian
if pre_run or not no_pre_run:
    nelec = int(dic["nelec"])
    spin = int(dic["spin"])
    fints = dic["orbitals"]
    if fints[-7:] == "FCIDUMP":
        fcidump = FCIDUMP()
        fcidump.read(fints)
        fcidump.params["nelec"] = str(nelec)
        fcidump.params["ms2"] = str(spin)
    else:
        fcidump = read_integral(fints, nelec, spin)

    _print("read integral finished", time.perf_counter() - tx)

    vacuum = SZ(0)
    target = SZ(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))
    n_sites = fcidump.n_sites
    orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
    hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)

if dic.get("warmup", None) == "occ":
    _print("using occ init")
    assert "occ" in dic
    occs = VectorDouble([float(occ) for occ in dic["occ"].split() if len(occ) != 0])
    bias = float(dic.get("bias", 1.0))
else:
    occs = None

# prepare mps
if "fullrestart" in dic:
    _print("full restart")
    mps_info = MPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = "KET"
    mps_info.load_mutable()
    mps = MPS(mps_info)
    mps.load_data()
    mps.load_mutable()
elif pre_run or not no_pre_run:
    mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
    mps_info.tag = "KET"
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    if MPI is None or MPI.rank == 0:
        mps_info.save_data(scratch + '/mps_info.bin')
    mps = MPS(n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()
else:
    mps_info = MPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = "KET"
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    mps = MPS(mps_info.n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

# prepare mpo
if pre_run or not no_pre_run:
    # mpo for dmrg
    _print("build mpo", time.perf_counter() - tx)
    mpo = MPOQC(hamil, QCTypes.Conventional)
    _print("simpl mpo", time.perf_counter() - tx)
    mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    _print("simpl mpo finished", time.perf_counter() - tx)

    _print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

    if MPI is None or MPI.rank == 0:
        mpo.save_data(scratch + '/mpo.bin')

    # mpo for 1pdm
    _print("build 1pdm mpo", time.perf_counter() - tx)
    pmpo = PDM1MPOQC(hamil)
    _print("simpl 1pdm mpo", time.perf_counter() - tx)
    pmpo = SimplifiedMPO(pmpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    _print("simpl 1pdm mpo finished", time.perf_counter() - tx)

    _print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in pmpo.right_operator_names]))

    if MPI is None or MPI.rank == 0:
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
    
if not pre_run:
    if MPI is not None:
        from block2.sz import ParallelMPO
        mpo = ParallelMPO(mpo, prule)
        pmpo = ParallelMPO(pmpo, prule_npdm)

    _print("para mpo finished", time.perf_counter() - tx)

    mps.save_mutable()
    mps.deallocate()
    mps_info.save_mutable()
    mps_info.deallocate_mutable()

    # GS DMRG
    if "restart_onepdm" not in dic and "restart_oh" not in dic:
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
        sweep_energies = []
        discarded_weights = []
        if "twodot_to_onedot" not in dic:
            E_dmrg = dmrg.solve(len(bond_dims), mps.center == 0, sweep_tol)
        else:
            tto = int(dic["twodot_to_onedot"])
            assert len(bond_dims) > tto
            dmrg.solve(tto, mps.center == 0, 0)
            # save the twodot part energies and discarded weights 
            sweep_energies.append(np.array(dmrg.energies))
            discarded_weights.append(np.array(dmrg.discarded_weights))
            dmrg.me.dot = 1
            dmrg.bond_dims = VectorUBond(bond_dims[tto:])
            dmrg.noises = VectorDouble(noises[tto:])
            dmrg.davidson_conv_thrds = VectorDouble(dav_thrds[tto:])
            E_dmrg = dmrg.solve(len(bond_dims) - tto, mps.center == 0, sweep_tol)
            mps.dot = 1
            if MPI is None or MPI.rank == 0:
                mps.save_data()
        
        sweep_energies.append(np.array(dmrg.energies))
        discarded_weights.append(np.array(dmrg.discarded_weights))
        sweep_energies = np.vstack(sweep_energies)
        discarded_weights = np.hstack(discarded_weights)

        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/E_dmrg.npy", E_dmrg)
            np.save(scratch + "/bond_dims.npy", bond_dims[:len(discarded_weights)])
            np.save(scratch + "/sweep_energies.npy", sweep_energies)
            np.save(scratch + "/discarded_weights.npy", discarded_weights)
        _print("DMRG Energy = %20.15f" % E_dmrg)

        if MPI is None or MPI.rank == 0:
            mps_info.save_data(scratch + '/mps_info.bin')

    # ONEPDM
    if "restart_onepdm" in dic or "onepdm" in dic:
        me = MovingEnvironment(pmpo, mps, mps, "1PDM")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
        expect.solve(True, mps.center == 0)

        if MPI is None or MPI.rank == 0:
            dmr = expect.get_1pdm(me.n_sites)
            dm = np.array(dmr).copy()
            dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))
            dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

            np.save(scratch + "/1pdm.npy", dm)
            mps_info.save_data(scratch + '/mps_info.bin')
        
            _print("DMRG OCC = ", "".join(["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))

    # OH
    if "restart_oh" in dic:
        me = MovingEnvironment(mpo, mps, mps, "OH")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
        E_oh = expect.solve(False, mps.center == 0)

        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/E_oh.npy", E_oh)
            mps_info.save_data(scratch + '/mps_info.bin')
        _print("OH Energy = %20.15f" % E_oh)

    mps_info.deallocate()
