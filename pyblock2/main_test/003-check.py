import sys

energies = [0.0] * 2
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy for root'):
            idx = int(l.split()[-3])
            energies[idx] = float(l.split()[-1])

assert abs(energies[0] - -107.654122447525) < 1E-5
assert abs(energies[1] - -106.959626154680) < 1E-5
