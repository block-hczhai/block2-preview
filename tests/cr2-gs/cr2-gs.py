
import sys
sys.path[:0] = ["../../build"]

from block2 import SU2, Global
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, NoiseTypes
from block2.su2 import HamiltonianQC, MPS, MPSInfo
from block2.su2 import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.su2 import DMRG, MovingEnvironment

Random.rand_seed(0)
scratch = './my_tmp'
n_threads = 8
bond_dims = [250] * 5 + [500] * 5 + [750] * 5
noises = [1E-5] * 5 + [1E-5] * 5 + [1E-5] * 15 + [0] * 5

import os
if not os.path.isdir(scratch):
    os.mkdir(scratch)
os.environ['TMPDIR'] = scratch

memory = int(20 * 1E9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
set_mkl_num_threads(n_threads)

from pyscf import gto, scf, cc, symm, ao2mo
import numpy as np

b = 1.68
with open('SVP','r') as f:
    mol = gto.Mole()
    mol.build(
        verbose = 3,
        symmetry = 'D2h',
        basis = {'Cr': gto.basis.parse(f.read())},
        atom = [['Cr',(0, 0, -b / 2)],
                ['Cr',(0, 0,  b / 2)]]
    )

m = scf.RHF(mol)
m.kernel()

mo_coeff = m.mo_coeff
n_ao = mo_coeff.shape[0]
n_mo = mo_coeff.shape[1]
n_elec = mol.nelectron

orb_coeff = mo_coeff
pg_reorder = True

fcidump_sym_d2h = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, orb_coeff)
orb_sym = np.array([fcidump_sym_d2h.index(i) + 1 for i in orb_sym_str])

# orbital reorder
if pg_reorder:
    optimal_reorder_d2h = ["Ag", "B1u", "B3u", "B2g", "B2u", "B3g", "B1g", "Au"]
    idx = np.argsort([optimal_reorder_d2h.index(i) for i in orb_sym_str])
    orb_sym = orb_sym[idx]
    orb_coeff = orb_coeff[:, idx]

g2e = ao2mo.restore(8, ao2mo.kernel(mol, orb_coeff), n_mo)
h1e = orb_coeff.T @ m.get_hcore() @ orb_coeff
ecore = mol.energy_nuc()

fcidump = FCIDUMP()
n_sites = n_mo
twos = 0
isym = 1
tol = 1E-13
mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
k = 0
for i in range(0, n_sites):
    for j in range(0, i + 1):
        assert abs(h1e[i, j] - h1e[j, i]) < tol
        mh1e[k] = h1e[i, j]
        k += 1
mg2e = g2e.flatten().copy()
mh1e[np.abs(mh1e) < tol] = 0.0
mg2e[np.abs(mg2e) < tol] = 0.0

fcidump.initialize_su2(n_sites, n_elec, twos, isym, ecore, mh1e, mg2e)

vacuum = SU2(0)
target = SU2(n_elec, twos, PointGroup.swap_d2h(isym))

orb_sym = VectorUInt8(map(PointGroup.swap_d2h, orb_sym))
hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)
hamil.opf.seq.mode = SeqTypes.Simple

print('CCSD ...')
mcc = cc.CCSD(m)
mcc.kernel()
dmmo = mcc.make_rdm1()
occ = np.diag(dmmo)
print('CCSD finished ...')

if pg_reorder:
    occ = occ[idx]

mpo = MPOQC(hamil, QCTypes.Conventional)
mpo = SimplifiedMPO(mpo, RuleQC(), True)

mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
mps_info.set_bond_dimension_using_occ(bond_dims[0], VectorDouble(occ))
mps = MPS(n_sites, 0, 2)
mps.initialize(mps_info)
mps.random_canonicalize()

mps.save_mutable()
mps.deallocate()
mps_info.save_mutable()
mps_info.deallocate_mutable()

me = MovingEnvironment(mpo, mps, mps, "DMRG")
me.init_environments(True)
dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
dmrg.noise_type = NoiseTypes.DensityMatrix
dmrg.solve(50, mps.center == 0, 0.0)

mps_info.deallocate()
mpo.deallocate()
hamil.deallocate()
fcidump.deallocate()
