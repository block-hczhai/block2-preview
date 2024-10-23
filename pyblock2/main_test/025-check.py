import sys

energy = 0
nat_occ = []
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])
        if l.startswith('REORDERED NAT OCC ='):
            nat_occ = [float(x) for x in l.split()[4:]]

assert abs(energy - -107.654122447525) < 1E-6
assert abs(nat_occ[0] - 1.999995) < 1E-4
