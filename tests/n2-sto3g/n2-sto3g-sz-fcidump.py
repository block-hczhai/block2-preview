
import sys
sys.path[:0] = ["../../build"]

from block2 import SZ, init_memory, release_memory, set_mkl_num_threads
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes
from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC

scratch = './my_tmp'
n_threads = 1

memory = int(1 * 1E9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
set_mkl_num_threads(n_threads)

fcidump = FCIDUMP()
fcidump.read('../../data/N2.STO3G.FCIDUMP')

vacuum = SZ(0)
target = SZ(fcidump.n_elec, fcidump.twos, PointGroup.swap_d2h(fcidump.isym))
n_sites = fcidump.n_sites
orb_sym = VectorUInt8(map(PointGroup.swap_d2h, fcidump.orb_sym))
hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)
hamil.opf.seq.mode = SeqTypes.Simple

mpo = MPOQC(hamil, QCTypes.NC)
mpo = SimplifiedMPO(mpo, RuleQC(), True)
print(mpo.get_blocking_formulas())

mpo.deallocate()
hamil.deallocate()
fcidump.deallocate()

release_memory()
