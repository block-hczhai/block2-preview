
import sys
sys.path[:0] = ["../../../build"]

from block2 import SZ, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, OpNames
from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import DMRG, MovingEnvironment
import numpy as np
import time

MPI = None
_print = print

tx = time.perf_counter()

Random.rand_seed(0)
scratch = './tmp'
n_threads = 28
bond_dims = [100] * 5 + [250] * 5 + [400] * 5 + [800] * 5
noises = [1E-4] * 5 + [1E-5] * 5 + [1E-6] * 5 + [1E-7] * 5 + [0]

import os
if not os.path.isdir(scratch):
    os.mkdir(scratch)
os.environ['TMPDIR'] = scratch

memory = int(20 * 1E9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, 28, 28, 1)
Global.threading.seq_type = SeqTypes.Nothing

if MPI is not None:
    from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
    prule = ParallelRuleQC(MPI)

fcidump = FCIDUMP()
# occs = np.diag(np.load('rdm1_ccsd_mp2_basis.npy')[0] + np.load('rdm1_ccsd_mp2_basis.npy')[1])
# occs = VectorDouble(np.array(occs))
# _print(["%.4f" % x for x in occs])
# _print(sum(occs))
fcidump.read('FCIDUMP')

hf_occ = VectorUInt8([int(x) for x in 
    "0 0 0 0 0 0 0 0 0 0 2 0 2 0 0 0 2 2 0 2 0 2 2 2 2 0 0 2 2 0 0 0 2 2 2 2 0 2 0 2 0 2 0 2 0 0 0 0 2 2 2 0 2 2 2 2 2 2 2 2".split()])
print(fcidump.det_energy(hf_occ, 0, fcidump.n_sites) + fcidump.const_e)

_print("read integral finished", time.perf_counter() - tx)

vacuum = SZ(0)
target = SZ(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))
n_sites = fcidump.n_sites
orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)
hamil.opf.seq.mode = SeqTypes.Nothing

bond_dim = 500
mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
mps_info.tag = "KET"
mps_info.set_bond_dimension(bond_dim)
mps_info.save_data('mps_info.bin')
mps = MPS(n_sites, 0, 2)
mps.initialize(mps_info)
mps.random_canonicalize()

_print("GS INIT MPS BOND DIMS = ", ''.join(["%6d" % x.n_states_total for x in mps_info.left_dims]))

_print("build mpo", time.perf_counter() - tx)
mpo = MPOQC(hamil, QCTypes.Conventional)
_print("simpl mpo", time.perf_counter() - tx)
mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
_print("simpl mpo finished", time.perf_counter() - tx)

_print('GS MPO BOND DIMS = ', ''.join(["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

mpo_prev = mpo
mpo.save_data('mpo.bin')

# if MPI is not None:
#     from block2.sz import ParallelMPO
#     mpo = ParallelMPO(mpo, prule)

_print("para mpo finished", time.perf_counter() - tx)

mps.save_mutable()
mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()
