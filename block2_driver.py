#! /usr/bin/env python
"""
block2 wrapper.

Author:
    Huanchen Zhai
    Zhi-Hao Cui
"""

from block2 import SZ, SU2, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, DoubleFPCodec
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, OpNames, VectorInt
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

dic = parse(fin)
if "nonspinadapted" in dic:
    from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, MPICommunicator
    from block2.sz import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
    from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC, ParallelMPO, ParallelMPS
    SX = SZ
else:
    from block2.su2 import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, MPICommunicator
    from block2.su2 import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
    from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelMPO, ParallelMPS
    SX = SU2

# MPI
MPI = MPICommunicator()
from mpi4py import MPI as PYMPI
comm = PYMPI.COMM_WORLD
def _print(*args, **kwargs):
    if MPI.rank == 0:
        print(*args, **kwargs)

tx = time.perf_counter()

# input parameters
Random.rand_seed(1234)
if DEBUG:
    _print("\n" + "*" * 34 + " INPUT START " + "*" * 34)
    for key, val in dic.items():
        _print ("%-25s %40s" % (key, val))
    _print("*" * 34 + " INPUT END   " + "*" * 34 + "\n")

scratch = dic.get("prefix", "./node0/")
restart_dir = dic.get("restart_dir", None)
n_threads = int(dic.get("num_thrds", 28))
bond_dims, dav_thrds, noises = dic["schedule"]
sweep_tol = float(dic.get("sweep_tol", 1e-6))

if MPI is not None and MPI.rank == 0:
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    if restart_dir is not None and not os.path.isdir(restart_dir):
        os.mkdir(restart_dir)
    os.environ['TMPDIR'] = scratch
if MPI is not None:
    MPI.barrier()

# global settings
memory = int(int(dic.get("mem", "40").split()[0]) * 1e9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
# ZHC NOTE nglobal_threads, nop_threads, MKL_NUM_THREADS
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, n_threads, n_threads, 1)
Global.threading.seq_type = SeqTypes.Tasked
Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
Global.frame.load_buffering = False
Global.frame.save_buffering = False
Global.frame.use_main_stack = False
if restart_dir is not None:
    Global.frame.restart_dir = restart_dir
_print(Global.frame)
_print(Global.threading)

if MPI is not None:
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

    vacuum = SX(0)
    target = SX(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))
    n_sites = fcidump.n_sites
    orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
    hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)

# parallelization over sites
# use keyword: conn_centers auto 5      (5 is number of procs)
#          or  conn_centers 10 20 30 40 (list of connection site indices)
if "conn_centers" in dic:
    assert MPI is not None
    cc = dic["conn_centers"].split()
    if cc[0] == "auto":
        ncc = int(cc[1])
        conn_centers = list(np.arange(0, n_sites * ncc, n_sites, dtype=int) // ncc)[1:]
        assert len(conn_centers) == ncc - 1
    else:
        conn_centers = [int(xcc) for xcc in cc]
    _print("using connection sites: ", conn_centers)
    assert MPI.size % (len(conn_centers) + 1) == 0
    mps_prule = prule
    prule = prule.split(MPI.size // (len(conn_centers) + 1))
else:
    conn_centers = None

if dic.get("warmup", None) == "occ":
    _print("using occ init")
    assert "occ" in dic
    if len(dic["occ"].split()) == 1:
        with open(dic["occ"], 'r') as ofin:
            dic["occ"] = ofin.readlines()[0]
    occs = VectorDouble([float(occ) for occ in dic["occ"].split() if len(occ) != 0])
    bias = float(dic.get("bias", 1.0))
else:
    occs = None

dot = 1 if "onedot" in dic else 2

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
    forward = mps.center == 0
    if mps.canonical_form[mps.center] == 'L' and mps.center != mps.n_sites - mps.dot:
        mps.center += 1
        forward = True
    elif mps.canonical_form[mps.center] == 'C' and mps.center != 0:
        mps.center -= 1
        forward = False
elif pre_run or not no_pre_run:
    mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
    mps_info.tag = "KET"
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    if MPI is None or MPI.rank == 0:
        mps_info.save_data(scratch + '/mps_info.bin')
    if conn_centers is not None:
        mps = ParallelMPS(mps_info.n_sites, 0, dot, mps_prule)
    else:
        mps = MPS(n_sites, 0, dot)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    forward = mps.center == 0
else:
    mps_info = MPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = "KET"
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    if conn_centers is not None:
        mps = ParallelMPS(mps_info.n_sites, 0, dot, mps_prule)
    else:
        mps = MPS(mps_info.n_sites, 0, dot)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    forward = mps.center == 0

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

if conn_centers is not None and "fullrestart" in dic:
    assert mps.dot == 2
    mps = ParallelMPS(mps, mps_prule)
    if mps.canonical_form[0] == 'C' and mps.canonical_form[1] == 'R':
        mps.canonical_form = 'K' + mps.canonical_form[1:]
    elif mps.canonical_form[-1] == 'C' and mps.canonical_form[-2] == 'L':
        mps.canonical_form = mps.canonical_form[:-1] + 'S'
        mps.center = mps.n_sites - 1

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

    _print('1PDM MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in pmpo.right_operator_names]))

    if MPI is None or MPI.rank == 0:
        pmpo.save_data(scratch + '/mpo-1pdm.bin')
    
    # mpo for particle number correlation
    _print("build 1npc mpo", time.perf_counter() - tx)
    nmpo = NPC1MPOQC(hamil)
    _print("simpl 1npc mpo", time.perf_counter() - tx)
    nmpo = SimplifiedMPO(nmpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    _print("simpl 1npc mpo finished", time.perf_counter() - tx)

    _print('1NPC MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in nmpo.right_operator_names]))

    if MPI is None or MPI.rank == 0:
        nmpo.save_data(scratch + '/mpo-1npc.bin')
else:
    mpo = MPO(0)
    mpo.load_data(scratch + '/mpo.bin')
    cg = CG(200)
    cg.initialize()
    mpo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
    
    pmpo = MPO(0)
    pmpo.load_data(scratch + '/mpo-1pdm.bin')
    pmpo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('1PDM MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in pmpo.left_operator_names]))

    nmpo = MPO(0)
    nmpo.load_data(scratch + '/mpo-1npc.bin')
    nmpo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('1NPC MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in nmpo.left_operator_names]))
    
if not pre_run:
    if MPI is not None:
        mpo = ParallelMPO(mpo, prule)
        pmpo = ParallelMPO(pmpo, prule_npdm)
        nmpo = ParallelMPO(nmpo, prule_npdm)

    _print("para mpo finished", time.perf_counter() - tx)

    mps.save_mutable()
    mps.deallocate()
    mps_info.save_mutable()
    mps_info.deallocate_mutable()

    if conn_centers is not None:
        mps.conn_centers = VectorInt(conn_centers)

    # GS DMRG
    if "restart_onepdm" not in dic and "restart_correlation" not in dic and "restart_oh" not in dic:
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        if conn_centers is not None:
            forward = mps.center == 0

        _print("env init finished", time.perf_counter() - tx)

        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        if "lowmem_noise" in dic:
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
        else:
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
        dmrg.cutoff = float(dic.get("cutoff", 1E-14))
        dmrg.decomp_type = DecompositionTypes.DensityMatrix
        dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
        sweep_energies = []
        discarded_weights = []
        if "twodot_to_onedot" not in dic:
            E_dmrg = dmrg.solve(len(bond_dims), forward, sweep_tol)
        else:
            tto = int(dic["twodot_to_onedot"])
            assert len(bond_dims) > tto
            dmrg.solve(tto, forward, 0)
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

        if conn_centers is not None:
            me.finalize_environments()

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
            if SX == SZ:
                dmr = expect.get_1pdm(me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))
                dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
                _print("DMRG OCC = ", "".join(["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
            else:
                dmr = expect.get_1pdm_spatial(me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                _print("DMRG OCC = ", "".join(["%6.3f" % x for x in np.diag(dm)]))

            np.save(scratch + "/1pdm.npy", dm)
            mps_info.save_data(scratch + '/mps_info.bin')
    

    # Particle Number Correlation
    if "restart_correlation" in dic or "correlation" in dic:
        me = MovingEnvironment(nmpo, mps, mps, "1NPC")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
        expect.solve(True, mps.center == 0)

        if MPI is None or MPI.rank == 0:
            if SX == SZ:
                dmr = expect.get_1npc(0, me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))
                dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
            else:
                dmr = expect.get_1npc_spatial(0, me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()

            np.save(scratch + "/1npc.npy", dm)
            mps_info.save_data(scratch + '/mps_info.bin')
        

    # OH
    if "restart_oh" in dic:
        me = MovingEnvironment(mpo, mps, mps, "OH")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
        E_oh = expect.solve(False, forward)

        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/E_oh.npy", E_oh)
            mps_info.save_data(scratch + '/mps_info.bin')
        _print("OH Energy = %20.15f" % E_oh)

    mps_info.deallocate()
