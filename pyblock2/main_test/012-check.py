import sys, struct, os

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -107.654122447525) < 1E-6

with open('node0/dmrg.e', 'rb') as f:
    a, b = struct.unpack('dd', f.read())
    assert abs(a - -107.654122447525) < 1E-5
    assert abs(b - -106.959626154680) < 1E-5

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

mx = fci.addons.fix_spin_(fci.FCI(mf), ss=0)
mx.kernel(h1e, g2e, driver.n_sites, driver.n_elec, tol=1e-12,
    wfnsym=0, orbsym=np.array(driver.orb_sym), nroots=4)
assert abs(mx.e_tot[0] + ecore - -107.654122447525) < 1E-6
assert abs(mx.e_tot[1] + ecore - -106.959626154680) < 1E-6

for ir in range(2):
    dm1, dm2 = mx.make_rdm12(mx.ci[ir], mf.mol.nao, mf.mol.nelec)

    xdm1 = np.load('./node0/1pdm-%d-%d.npy' % (ir, ir))
    xdm1 = xdm1[0] + xdm1[1]

    assert np.linalg.norm(dm1 - xdm1) < 1E-5

    xdm2 = np.load('./node0/2pdm-%d-%d.npy' % (ir, ir))
    xdm2 = xdm2[0] + xdm2[2] + xdm2[1] + xdm2[1].transpose(1, 0, 3, 2)

    assert np.linalg.norm(dm2 - xdm2.transpose(0, 3, 1, 2)) < 1E-4
