import sys

energy = 0
quantum = ''
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])
        elif l.startswith('MPS ='):
            quantum = l.split('< ')[-1].split(' >')[0]

assert abs(energy - -106.939132859667396) < 1E-6
assert quantum == 'N=16 S=0 PG=0'
