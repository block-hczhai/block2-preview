import sys

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -107.654122447525) < 1E-6

import numpy as np

dets = np.load('node0/sample-dets.npy')
vals = np.load('node0/sample-vals.npy')

idx = np.argsort(np.abs(vals))[-1]
assert np.all(dets[idx] == [3] * 5 + [0, 3] * 2 + [0])
assert abs(abs(vals[idx]) - 0.957506527063957) < 1E-4
