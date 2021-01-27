
import sys
sys.path[:0] = ["../../build"]

from block2 import SU2, Global, ThreadingTypes, Threading, DoubleFPCodec
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, NoiseTypes, DecompositionTypes
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, OpNamesSet, OpNames
from block2.su2 import HamiltonianQC, MPS, MPSInfo
from block2.su2 import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.su2 import DMRG, MovingEnvironment
Random.rand_seed(0)
scratch = './node0'
n_threads = 4
bond_dims = [250] * 5 + [500] * 5
noises = [1E-6] * 5

import os
if not os.path.isdir(scratch):
    os.mkdir(scratch)
os.environ['TMPDIR'] = scratch

memory = int(2 * 1E9)
init_memory(isize=int(1E8), dsize=int(memory), save_dir=scratch)
Global.threading = Threading(ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, n_threads, n_threads, 1)
Global.threading.seq_type = SeqTypes.Nothing
Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
Global.frame.use_main_stack = False
print(Global.frame)
print(Global.threading)

fcidump = FCIDUMP()
occs = read_occ('../../data/CR2.SVP.OCC')
fcidump.read('../../data/CR2.SVP.FCIDUMP')

vacuum = SU2(0)
target = SU2(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))
n_sites = fcidump.n_sites
orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)

mpo = MPOQC(hamil, QCTypes.Conventional)
mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

bond_dim = 250
mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
mps_info.tag = "KET"
mps_info.set_bond_dimension_using_occ(bond_dim, occs)
mps = MPS(n_sites, 0, 2)
mps.initialize(mps_info)
mps.random_canonicalize()

mps.save_mutable()
mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()

me = MovingEnvironment(mpo, mps, mps, "DMRG")
me.delayed_contraction = OpNamesSet.normal_ops()
me.cached_contraction = True
me.init_environments(True)

dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
dmrg.decomp_type = DecompositionTypes.DensityMatrix
dmrg.solve(10, mps.center == 0)

mps_info.deallocate()
mpo.deallocate()
hamil.deallocate()
fcidump.deallocate()
