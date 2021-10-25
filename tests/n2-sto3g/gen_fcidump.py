
from pyscf import gto, scf, ci, mcscf, ao2mo, symm
import numpy as np
from block2 import *

mol = gto.M(
    atom=[['N', (0, 0, 0)], ['N', (0, 0, 1.1)]],
    basis='sto3g',
    verbose=3, symmetry='d2h', spin=0)

mf = scf.RHF(mol).run()

n_sites = mol.nao
n_elec = sum(mol.nelec)
tol = 1E-13

fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
optimal_reorder = ["Ag", "B1u", "B3u", "B2g", "B2u", "B3g", "B1g", "Au"]
orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])
idx = np.argsort([optimal_reorder.index(i) for i in orb_sym_str])
orb_sym = orb_sym[idx]
mf.mo_coeff = mf.mo_coeff[:, idx]

print(orb_sym)

mc = mcscf.CASCI(mf, n_sites, n_elec)
h1e, e_core = mc.get_h1cas()
g2e = mc.get_h2cas()

h1e = h1e[np.tril_indices(n_sites)]
h1e[np.abs(h1e) < tol] = 0
g2e = ao2mo.restore(8, g2e, n_sites)
g2e[(g2e < tol) & (g2e > -tol)] = 0

n = n_sites
fd2 = FCIDUMP()
fd2.initialize_su2(n, n_elec, 0, 1, e_core, h1e, g2e)
fd2.orb_sym = VectorUInt8(orb_sym)
xh1e = np.array(fd2.h1e_matrix()).reshape((n, n))
xg2e = np.array(fd2.g2e_1fold()).reshape((n, n, n, n))

fd = FCIDUMP()
fd.read("../../data/N2.STO3G.FCIDUMP")
assert n == fd.n_sites
ph1e = np.array(fd.h1e_matrix()).reshape((n, n))
pg2e = np.array(fd.g2e_1fold()).reshape((n, n, n, n))
print(np.linalg.norm(ph1e - xh1e))
print(np.linalg.norm(pg2e - xg2e))

gh1e = np.zeros((n * 2, n * 2))
gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2))

for i in range(n * 2):
    for j in range(n * 2):
        if i % 2 == j % 2:
            gh1e[i, j] = xh1e[i // 2, j // 2]

for i in range(n * 2):
    for j in range(n * 2):
        for k in range(n * 2):
            for l in range(n * 2):
                if i % 2 == j % 2 and k % 2 == l % 2:
                    gg2e[i, j, k, l] = xg2e[i // 2, j // 2, k // 2, l // 2]

print('h1e diff = ', np.linalg.norm(gh1e - gh1e.T))

qh1e = gh1e[np.tril_indices(n * 2)]
qg2e = ao2mo.restore(8, gg2e, n * 2)

fd3 = FCIDUMP()
fd3.initialize_su2(n * 2, n_elec, n_elec, 1, e_core, qh1e, qg2e)
fd3.orb_sym = VectorUInt8([orb_sym[i // 2] for i in range(n * 2)])
fd3.write('N2.STO3G.G.FCIDUMP')
