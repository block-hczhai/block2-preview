import sys
import numpy as np

energies = np.zeros((4, 4), dtype=complex)
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('OH Energy'):
            a, b = [int(c) for c in l.split()[2:5:2]]
            energies[a, b] = float(l.split()[-1]) * 1j + float(l.split()[-4])

np.save('node0/ovlps.npy', energies)

x = energies[1, 0] / (energies[0, 0] * energies[1, 1]) ** 0.5
ang = -107.654122447525 * 0.2 % (2 * np.pi)

assert abs(np.abs(x) - 1.0) < 1E-6
assert abs(abs(np.angle(x) - ang) - 2 * np.pi) < 1E-6

x = energies[3, 2] / (energies[2, 2] * energies[3, 3]) ** 0.5
ang = -106.959626154680 * 0.2 % (2 * np.pi)

assert abs(np.abs(x) - 1.0) < 1E-6
assert abs(abs(np.angle(x) - ang) - 2 * np.pi) < 1E-6
