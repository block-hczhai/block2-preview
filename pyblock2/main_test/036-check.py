import sys

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('OH Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -106.939132859667396) < 1E-6
