import sys, struct

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -106.94375693899154) < 1E-6

with open('node0/dmrg.e', 'rb') as f:
    a, b, c = struct.unpack('ddd', f.read())
    assert abs(a - -106.9437569390) < 1E-5
    assert abs(b - -106.9304278080) < 1E-5
    assert abs(c - -106.8426967564) < 1E-5
