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
    from block2 import VectorSZ as VectorSL
    from block2.sz import MultiMPS, MultiMPSInfo
    from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, MPICommunicator
    from block2.sz import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC, NoTransposeRule
    from block2.sz import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
    from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC, ParallelMPO, ParallelMPS, IdentityMPO
    SX = SZ
else:
    from block2 import VectorSU2 as VectorSL
    from block2.su2 import MultiMPS, MultiMPSInfo
    from block2.su2 import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, MPICommunicator
    from block2.su2 import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC, NoTransposeRule
    from block2.su2 import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
    from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelMPO, ParallelMPS, IdentityMPO
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

scratch = dic.get("prefix", "./nodex/")
restart_dir = dic.get("restart_dir", None)
restart_dir_per_sweep = dic.get("restart_dir_per_sweep", None)
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
fp_cps_cutoff = float(dic.get("fp_cps_cutoff", 1E-16))
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
# ZHC NOTE nglobal_threads, nop_threads, MKL_NUM_THREADS
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, n_threads, n_threads, 1)
Global.threading.seq_type = SeqTypes.Tasked
Global.frame.fp_codec = DoubleFPCodec(fp_cps_cutoff, 1024)
Global.frame.load_buffering = False
Global.frame.save_buffering = False
Global.frame.use_main_stack = False
Global.frame.minimal_disk_usage = True
if restart_dir is not None:
    Global.frame.restart_dir = restart_dir
if restart_dir_per_sweep is not None:
    Global.frame.restart_dir_per_sweep = restart_dir_per_sweep
_print(Global.frame)
_print(Global.threading)

if MPI is not None:
    prule = ParallelRuleQC(MPI)
    prule_npdm = ParallelRuleNPDMQC(MPI)

# prepare hamiltonian
if pre_run or not no_pre_run:
    nelec = [int(x) for x in dic["nelec"].split()]
    spin = [int(x) for x in dic.get("spin", "0").split()]
    isym = [int(x) for x in dic.get("irrep", "1").split()]
    fints = dic["orbitals"]
    if open(fints, 'rb').read(4) != b'\x89HDF':
        fcidump = FCIDUMP()
        fcidump.read(fints)
        fcidump.params["nelec"] = str(nelec[0])
        fcidump.params["ms2"] = str(spin[0])
        fcidump.params["isym"] = str(isym[0])
    else:
        fcidump = read_integral(fints, nelec[0], spin[0], isym=isym[0])
    swap_pg = getattr(PointGroup, "swap_" + dic.get("sym", "d2h"))

    _print("read integral finished", time.perf_counter() - tx)

    vacuum = SX(0)
    target = SX(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    targets = []
    for inelec in nelec:
        for ispin in spin:
            for iisym in isym:
                targets.append(SX(inelec, ispin, swap_pg(iisym)))
    targets = VectorSL(targets)
    n_sites = fcidump.n_sites
    orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
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
nroots = int(dic.get("nroots", 1))
mps_tags = dic.get("mps_tags", "KET").split()
soc = "soc" in dic
overlap = "overlap" in dic

# prepare mps
if "fullrestart" in dic:
    _print("full restart")
    mps_info = MPSInfo(0) if nroots == 1 and len(targets) == 1 else MultiMPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = mps_tags[0]
    mps_info.load_mutable()
    max_bdim = max([x.n_states_total for x in mps_info.left_dims])
    if mps_info.bond_dim < max_bdim:
        mps_info.bond_dim = max_bdim
    max_bdim = max([x.n_states_total for x in mps_info.right_dims])
    if mps_info.bond_dim < max_bdim:
        mps_info.bond_dim = max_bdim
    mps = MPS(mps_info) if nroots == 1 and len(targets) == 1 else MultiMPS(mps_info)
    mps.load_data()
    if nroots != 1:
        mps.nroots = nroots
        mps.wfns = mps.wfns[:nroots]
        mps.weights = mps.weights[:nroots]
    weights = dic.get("weights", None)
    if weights is not None:
        mps.weights = VectorDouble([float(x) for x in weights.split()])
    mps.load_mutable()
    forward = mps.center == 0
    if mps.canonical_form[mps.center] == 'L' and mps.center != mps.n_sites - mps.dot:
        mps.center += 1
        forward = True
    elif mps.canonical_form[mps.center] == 'C' and mps.center != 0:
        mps.center -= 1
        forward = False
    elif mps.center == mps.n_sites - 1 and mps.dot == 2:
        if mps.canonical_form[mps.center] == 'K':
            cg = CG(200)
            cg.initialize()
            mps.move_left(cg, prule)
        mps.center = mps.n_sites - 2
        mps.save_data()
        forward = False
elif pre_run or not no_pre_run:
    if nroots == 1 and len(targets) == 1:
        mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
    else:
        print('TARGETS = ', list(targets), flush=True)
        mps_info = MultiMPSInfo(n_sites, vacuum, targets, hamil.basis)
    if "full_fci_space" in dic:
        mps_info.set_bond_dimension_full_fci()
    mps_info.tag = mps_tags[0]
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    if MPI is None or MPI.rank == 0:
        mps_info.save_data(scratch + '/mps_info.bin')
    if conn_centers is not None:
        assert nroots == 1
        mps = ParallelMPS(mps_info.n_sites, 0, dot, mps_prule)
    elif nroots != 1 or len(targets) != 1:
        mps = MultiMPS(n_sites, 0, dot, nroots)
        weights = dic.get("weights", None)
        if weights is not None:
            mps.weights = VectorDouble([float(x) for x in weights.split()])
    else:
        mps = MPS(n_sites, 0, dot)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    forward = mps.center == 0
else:
    mps_info = MPSInfo(0) if nroots == 1 and len(targets) == 1 else MultiMPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = mps_tags[0]
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    if conn_centers is not None:
        assert nroots == 1
        mps = ParallelMPS(mps_info.n_sites, 0, dot, mps_prule)
    elif nroots != 1 or len(targets) != 1:
        mps = MultiMPS(n_sites, 0, dot, nroots)
        weights = dic.get("weights", None)
        if weights is not None:
            mps.weights = VectorDouble([float(x) for x in weights.split()])
    else:
        mps = MPS(mps_info.n_sites, 0, dot)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    forward = mps.center == 0

_print("MPS = ", mps.canonical_form, mps.center, mps.dot)
_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

if conn_centers is not None and "fullrestart" in dic:
    assert mps.dot == 2
    mps = ParallelMPS(mps, mps_prule)
    if mps.canonical_form[0] == 'C' and mps.canonical_form[1] == 'R':
        mps.canonical_form = 'K' + mps.canonical_form[1:]
    elif mps.canonical_form[-1] == 'C' and mps.canonical_form[-2] == 'L':
        mps.canonical_form = mps.canonical_form[:-1] + 'S'
        mps.center = mps.n_sites - 1

has_tran = "restart_tran_onepdm" in dic or "tran_onepdm" in dic \
    or "restart_tran_oh" in dic or "tran_oh" in dic

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
    pmpo = PDM1MPOQC(hamil, 1 if soc else 0)
    pmpo = SimplifiedMPO(pmpo,
        NoTransposeRule(RuleQC()) if has_tran else RuleQC(),
        True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    if MPI is None or MPI.rank == 0:
        pmpo.save_data(scratch + '/mpo-1pdm.bin')
    
    # mpo for particle number correlation
    _print("build 1npc mpo", time.perf_counter() - tx)
    nmpo = NPC1MPOQC(hamil)
    nmpo = SimplifiedMPO(nmpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    if MPI is None or MPI.rank == 0:
        nmpo.save_data(scratch + '/mpo-1npc.bin')

    # mpo for identity operator
    _print("build identity mpo", time.perf_counter() - tx)
    impo = IdentityMPO(hamil)
    impo = SimplifiedMPO(impo,
        NoTransposeRule(RuleQC()) if has_tran else RuleQC(),
        True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    if MPI is None or MPI.rank == 0:
        impo.save_data(scratch + '/mpo-ident.bin')

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

    impo = MPO(0)
    impo.load_data(scratch + '/mpo-ident.bin')
    impo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('IDENT MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in impo.left_operator_names]))

def split_mps(iroot, mps, mps_info):
    mps.load_data() # this will avoid memory sharing
    mps_info.load_mutable()
    mps.load_mutable()

    # break up a MultiMPS to single MPSs
    if len(mps_info.targets) != 1:
        smps_info = MultiMPSInfo(mps_info.n_sites, mps_info.vacuum,
                                mps_info.targets, mps_info.basis)
        if "full_fci_space" in dic:
            smps_info.set_bond_dimension_full_fci()
        smps_info.tag = mps_info.tag + "-%d" % iroot
        smps_info.bond_dim = mps_info.bond_dim
        for i in range(0, smps_info.n_sites + 1):
            smps_info.left_dims[i] = mps_info.left_dims[i]
            smps_info.right_dims[i] = mps_info.right_dims[i]
        smps_info.save_mutable()
        smps = MultiMPS(smps_info)
        smps.n_sites = mps.n_sites
        smps.center = mps.center
        smps.dot = mps.dot
        smps.canonical_form = '' + mps.canonical_form
        smps.tensors = mps.tensors[:]
        smps.wfns = mps.wfns[iroot:iroot+1]
        smps.weights = mps.weights[iroot:iroot+1]
        smps.weights[0] = 1
        smps.nroots = 1
        smps.save_mutable()
    else:
        smps_info = MPSInfo(mps_info.n_sites, mps_info.vacuum,
                                mps_info.targets[0], mps_info.basis)
        if "full_fci_space" in dic:
            smps_info.set_bond_dimension_full_fci()
        smps_info.tag = mps_info.tag + "-%d" % iroot
        smps_info.bond_dim = mps_info.bond_dim
        for i in range(0, smps_info.n_sites + 1):
            smps_info.left_dims[i] = mps_info.left_dims[i]
            smps_info.right_dims[i] = mps_info.right_dims[i]
        smps_info.save_mutable()
        smps = MPS(smps_info)
        smps.n_sites = mps.n_sites
        smps.center = mps.center
        smps.dot = mps.dot
        smps.canonical_form = '' + mps.canonical_form
        smps.tensors = mps.tensors[:]
        if smps.tensors[smps.center] is None:
            smps.tensors[smps.center] = mps.wfns[iroot][0]
        else:
            assert smps.center + 1 < smps.n_sites
            assert smps.tensors[smps.center + 1] is None
            smps.tensors[smps.center + 1] = mps.wfns[iroot][0]
        smps.save_mutable()
    
    smps.dot = dot
    forward = smps.center == 0
    if smps.canonical_form[smps.center] == 'L' and smps.center != smps.n_sites - smps.dot:
        smps.center += 1
        forward = True
    elif (smps.canonical_form[smps.center] == 'C' or smps.canonical_form[smps.center] == 'M') and smps.center != 0:
        smps.center -= 1
        forward = False
    if smps.canonical_form[smps.center] == 'M' and not isinstance(smps, MultiMPS):
        smps.canonical_form = smps.canonical_form[:smps.center] + 'C' + smps.canonical_form[smps.center + 1:]
    if smps.canonical_form[-1] == 'M' and not isinstance(smps, MultiMPS):
        smps.canonical_form = smps.canonical_form[:-1] + 'C'
    if dot == 1:
        if smps.canonical_form[0] == 'C' and smps.canonical_form[1] == 'R':
            smps.canonical_form = 'K' + smps.canonical_form[1:]
        elif smps.canonical_form[-1] == 'C' and smps.canonical_form[-2] == 'L':
            smps.canonical_form = smps.canonical_form[:-1] + 'S'
            smps.center = smps.n_sites - 1
        if smps.canonical_form[0] == 'M' and smps.canonical_form[1] == 'R':
            smps.canonical_form = 'J' + smps.canonical_form[1:]
        elif smps.canonical_form[-1] == 'M' and smps.canonical_form[-2] == 'L':
            smps.canonical_form = smps.canonical_form[:-1] + 'T'
            smps.center = smps.n_sites - 1

    mps.deallocate()
    mps_info.deallocate_mutable()
    smps.save_data()
    return smps, smps_info, forward

if not pre_run:
    if MPI is not None:
        mpo = ParallelMPO(mpo, prule)
        pmpo = ParallelMPO(pmpo, prule_npdm)
        nmpo = ParallelMPO(nmpo, prule_npdm)
        impo = ParallelMPO(impo, prule)

    _print("para mpo finished", time.perf_counter() - tx)

    mps.save_mutable()
    mps.deallocate()
    mps_info.save_mutable()
    mps_info.deallocate_mutable()

    if conn_centers is not None:
        mps.conn_centers = VectorInt(conn_centers)

    # state-specific DMRG (experimental)
    if "statespecific" in dic:
        assert isinstance(mps, MultiMPS)
        assert nroots != 1
        for iroot in range(mps.nroots):
            _print('----- root = %3d / %3d -----' % (iroot, mps.nroots), flush=True)
            smps, smps_info, forward = split_mps(iroot, mps, mps_info)
            
            me = MovingEnvironment(mpo, smps, smps, "DMRG")
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
            me.save_partition_info = True
            me.init_environments(True)

            if conn_centers is not None:
                forward = smps.center == 0

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
                E_dmrg = dmrg.solve(len(bond_dims) - tto, smps.center == 0, sweep_tol)
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
                np.save(scratch + "/E_dmrg-%d.npy" % iroot, E_dmrg)
                np.save(scratch + "/bond_dims-%d.npy" % iroot, bond_dims[:len(discarded_weights)])
                np.save(scratch + "/sweep_energies-%d.npy" % iroot, sweep_energies)
                np.save(scratch + "/discarded_weights-%d.npy" % iroot, discarded_weights)
            _print("DMRG Energy for root %4d = %20.15f" % (iroot, E_dmrg))

            if MPI is None or MPI.rank == 0:
                smps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

    # GS DMRG
    if "restart_onepdm" not in dic and "restart_correlation" not in dic \
        and "restart_oh" not in dic and "statespecific" not in dic \
        and "restart_tran_onepdm" not in dic and "restart_tran_oh" not in dic:
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

    def do_onepdm(bmps, kmps):
        me = MovingEnvironment(pmpo, bmps, kmps, "1PDM")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, bmps.info.bond_dim, kmps.info.bond_dim)
        expect.solve(True, kmps.center == 0)

        if MPI is None or MPI.rank == 0:
            if SX == SZ:
                dmr = expect.get_1pdm(me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))
                dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
            else:
                dmr = expect.get_1pdm_spatial(me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
        
            return dm
        else:
            return None

    # ONEPDM
    if "restart_onepdm" in dic or "onepdm" in dic:

        if nroots == 1:
            dm = do_onepdm(mps, mps)
            if MPI is None or MPI.rank == 0:
                if SX == SZ:
                    _print("DMRG OCC = ", "".join(["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
                else:
                    _print("DMRG OCC = ", "".join(["%6.3f" % x for x in np.diag(dm)]))
                np.save(scratch + "/1pdm.npy", dm)
        else:
            for iroot in range(mps.nroots):
                _print('----- root = %3d / %3d -----' % (iroot, mps.nroots), flush=True)
                smps, smps_info, forward = split_mps(iroot, mps, mps_info)
                dm = do_onepdm(smps, smps)
                if MPI is None or MPI.rank == 0:
                    if SX == SZ:
                        _print("DMRG OCC (state %4d) = " % iroot, "".join(["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
                    else:
                        _print("DMRG OCC (state %4d) = " % iroot, "".join(["%6.3f" % x for x in np.diag(dm)]))
                    np.save(scratch + "/1pdm-%d-%d.npy" % (iroot, iroot), dm)
                    smps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

        if MPI is None or MPI.rank == 0:
            mps_info.save_data(scratch + '/mps_info.bin')
    
    # Transition ONEPDM
    # note that there can be a undetermined +1/-1 factor due to the relative phase in two MPSs
    if "restart_tran_onepdm" in dic or "tran_onepdm" in dic:

        assert nroots != 1
        for iroot in range(mps.nroots):
            for jroot in range(mps.nroots):
                _print('----- root = %3d -> %3d / %3d -----' % (jroot, iroot, mps.nroots), flush=True)
                simps, simps_info, _ = split_mps(iroot, mps, mps_info)
                sjmps, sjmps_info, _ = split_mps(jroot, mps, mps_info)
                dm = do_onepdm(simps, sjmps)
                if SX == SU2:
                    qsbra = simps.info.targets[0].twos
                    # fix different Wigner–Eckart theorem convention
                    dm *= np.sqrt(qsbra + 1)
                dm = dm / np.sqrt(2)
                if MPI is None or MPI.rank == 0:
                    np.save(scratch + "/1pdm-%d-%d.npy" % (iroot, jroot), dm)
            if MPI is None or MPI.rank == 0:
                if SX == SZ:
                    _print("DMRG OCC (state %4d) = " % iroot, "".join(["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
                else:
                    _print("DMRG OCC (state %4d) = " % iroot, "".join(["%6.3f" % x for x in np.diag(dm)]))
                simps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

        if MPI is None or MPI.rank == 0:
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

    def do_oh(bmps, kmps):
        me = MovingEnvironment(impo if overlap else mpo, bmps, kmps, "OH")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(True)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, bmps.info.bond_dim, kmps.info.bond_dim)
        E_oh = expect.solve(False, kmps.center == 0)

        if MPI is None or MPI.rank == 0:
            return E_oh
        else:
            return None

    # OH (Hamiltonian expectation on MPS)
    if "restart_oh" in dic or "oh" in dic:

        if nroots == 1:
            E_oh = do_oh(mps, mps)
            if MPI is None or MPI.rank == 0:
                np.save(scratch + "/E_oh.npy", E_oh)
                print("OH Energy = %20.15f" % E_oh)
        else:
            mat_oh = np.zeros((mps.nroots, ))
            for iroot in range(mps.nroots):
                _print('----- root = %3d / %3d -----' % (iroot, mps.nroots), flush=True)
                smps, smps_info, forward = split_mps(iroot, mps, mps_info)
                E_oh = do_oh(smps, smps)
                if MPI is None or MPI.rank == 0:
                    mat_oh[iroot] = E_oh
                    print("OH Energy %4d - %4d = %20.15f" % (iroot, iroot, E_oh))
                    smps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)
            if MPI is None or MPI.rank == 0:
                np.save(scratch + "/E_oh.npy", mat_oh)

        if MPI is None or MPI.rank == 0:
            mps_info.save_data(scratch + '/mps_info.bin')
    
    # Transition OH (OH between different MPS roots)
    # note that there can be a undetermined +1/-1 factor due to the relative phase in two MPSs
    # only mat_oh[i, j] with i >= j are filled
    if "restart_tran_oh" in dic or "tran_oh" in dic:
        
        assert nroots != 1
        mat_oh = np.zeros((mps.nroots, mps.nroots))
        for iroot in range(mps.nroots):
            for jroot in range(iroot + 1):
                _print('----- root = %3d -> %3d / %3d -----' % (jroot, iroot, mps.nroots), flush=True)
                simps, simps_info, _ = split_mps(iroot, mps, mps_info)
                sjmps, sjmps_info, _ = split_mps(jroot, mps, mps_info)
                E_oh = do_oh(simps, sjmps)
                if MPI is None or MPI.rank == 0:
                    mat_oh[iroot, jroot] = E_oh
                    print("OH Energy %4d - %4d = %20.15f" % (iroot, jroot, E_oh))
            if MPI is None or MPI.rank == 0:
                simps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)
        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/E_oh.npy", mat_oh)

        if MPI is None or MPI.rank == 0:
            mps_info.save_data(scratch + '/mps_info.bin')

    mps_info.deallocate()
