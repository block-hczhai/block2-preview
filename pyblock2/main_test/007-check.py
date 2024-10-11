import sys

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -106.94375693899154) < 1E-6
