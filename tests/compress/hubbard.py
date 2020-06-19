import sys
sys.path[:0] = ["../../build"]

from block2 import SU2, Global
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUInt16, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, NoiseTypes
from block2.su2 import HamiltonianQC, MPS, MPSInfo, IdentityMPO, Compress
from block2.su2 import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.su2 import DMRG, MovingEnvironment, NoTransposeRule

Random.rand_seed(0)
scratch = './my_tmp'
n_threads = 4
bond_dims = [250]
noises = [1E-6]

memory = int(1 * 1E9) # 1G memory
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
set_mkl_num_threads(n_threads)

fcidump = FCIDUMP()
fcidump.read('../../data/HUBBARD-L16.FCIDUMP')

vacuum = SU2(0)
target = SU2(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))

n_sites = fcidump.n_sites
orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
hamil = HamiltonianQC(vacuum, target, n_sites, orb_sym, fcidump)
hamil.opf.seq.mode = SeqTypes.Simple

mpo = MPOQC(hamil, QCTypes.Conventional)
mpo = SimplifiedMPO(mpo, RuleQC(), True)

mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis, hamil.orb_sym)
mps_info.tag = 'KET'
mps_info.set_bond_dimension(bond_dims[0])
mps = MPS(n_sites, 0, 2)
mps.initialize(mps_info)
mps.random_canonicalize()

mps.save_mutable()
mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()

me = MovingEnvironment(mpo, mps, mps, "DMRG")
me.init_environments(True)
dmrg = DMRG(me, VectorUInt16(bond_dims), VectorDouble(noises))
dmrg.noise_type = NoiseTypes.DensityMatrix
dmrg.solve(10, mps.center == 0)

bra_info = MPSInfo(n_sites, vacuum, target, hamil.basis, hamil.orb_sym)
bra_info.tag = 'BRA'
bra_info.set_bond_dimension(bond_dims[0] // 2)
bra = MPS(n_sites, mps.center, 2)
bra.initialize(bra_info)
bra.random_canonicalize()

bra.save_mutable()
bra.deallocate()
bra_info.save_mutable()
bra_info.deallocate_mutable()

# impo = IdentityMPO(hamil)
# impo = SimplifiedMPO(impo, Rule(), True)

cmpo = MPOQC(hamil, QCTypes.Conventional)
cmpo = SimplifiedMPO(cmpo, NoTransposeRule(RuleQC()), True)

cps_me = MovingEnvironment(cmpo, bra, mps, "COMPRESS")
cps_me.init_environments(True)
cps = Compress(cps_me, VectorUInt16([bond_dims[0] // 2]), VectorUInt16(bond_dims), VectorDouble([0.0]))
cps.noise_type = NoiseTypes.DensityMatrix
cps.solve(10, mps.center == 0)

cmpo.deallocate()
bra_info.deallocate()
mps_info.deallocate()
mpo.deallocate()
hamil.deallocate()
fcidump.deallocate()

release_memory()
