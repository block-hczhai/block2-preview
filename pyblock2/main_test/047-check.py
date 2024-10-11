import sys, struct

energies = []
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('T = RE'):
            energies.append(float(l.split()[9]) / float(l.split()[-1]))

for energy in energies:
    assert abs(energy - -106.959626154680) < 1E-6
