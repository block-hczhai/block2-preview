import sys
import numpy as np

energies = np.zeros((4, 4), dtype=complex)
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('OH Energy'):
            a, b = [int(c) for c in l.split()[2:5:2]]
            energies[a, b] = float(l.split()[-1]) * 1j + float(l.split()[-4])

ovlps = np.load('node0/ovlps.npy')

assert abs(energies[0, 0] / ovlps[0, 0] - -107.654122447525) < 1E-5
assert abs(energies[1, 1] / ovlps[1, 1] - -107.654122447525) < 1E-5
assert abs(energies[2, 2] / ovlps[2, 2] - -106.959626154680) < 1E-5
assert abs(energies[3, 3] / ovlps[3, 3] - -106.959626154680) < 1E-5
