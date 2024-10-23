import sys, os

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -107.654122447525) < 1E-6

if os.name == "nt":
    quit() # pyscf not available on windows

from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf import fci
from pyscf.tools import fcidump
import numpy as np

driver = DMRGDriver(scratch="./nodex", symm_type=SymmetryTypes.SU2, n_threads=4)
driver.read_fcidump(filename='../../data/N2.STO3G.FCIDUMP', pg='d2h', iprint=0)
driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec,
    spin=driver.spin, orb_sym=driver.orb_sym)
h1e, g2e, ecore = driver.h1e, driver.g2e, driver.ecore

mf = fcidump.to_scf('../../data/N2.STO3G.FCIDUMP', molpro_orbsym=True)
mf.mol.symmetry = True

mx = fci.FCI(mf)
mx.kernel(h1e, g2e, driver.n_sites, driver.n_elec, tol=1e-12,
    wfnsym=0, orbsym=np.array(driver.orb_sym))
assert abs(mx.e_tot + ecore - -107.654122447525) < 1E-6

dm1, dm2 = mx.make_rdm12(mx.ci, mf.mol.nao, mf.mol.nelec)

xdm1 = np.load('./node0/1pdm.npy')
xdm1 = xdm1[0] + xdm1[1]

assert np.linalg.norm(dm1 - xdm1) < 1E-5

_e_pqqp = np.load('./node0/e_pqqp.npy')
_e_pqpq = np.load('./node0/e_pqpq.npy')
_2pdm_spat = dm2.transpose(0, 2, 3, 1)
_2pdm_spat_pqqp = np.einsum('pqqp->pq', _2pdm_spat)
_2pdm_spat_pqpq = np.einsum('pqpq->pq', _2pdm_spat)

assert np.linalg.norm(_e_pqqp - _2pdm_spat_pqqp) < 1E-5
assert np.linalg.norm(_e_pqpq - _2pdm_spat_pqpq) < 1E-5
