import sys, struct

energies = []
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('T = RE'):
            energies.append(float(l.split()[9]) / float(l.split()[-1]))

for energy in energies:
    assert abs(energy - -107.654122447525) < 1E-6
