
import os

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

xdm2 = np.load('./node0/2pdm.npy')
xdm2 = xdm2[0] + xdm2[2] + xdm2[1] + xdm2[1].transpose(1, 0, 3, 2)

assert np.linalg.norm(dm2 - xdm2.transpose(0, 3, 1, 2)) < 1E-4
