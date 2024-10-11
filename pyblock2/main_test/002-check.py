import sys, struct

energy = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('DMRG Energy ='):
            energy = float(l.split()[-1])

assert abs(energy - -107.654122447525) < 1E-6

with open('node0/dmrg.e', 'rb') as f:
    a, b = struct.unpack('dd', f.read())
    assert abs(a - -107.654122447525) < 1E-5
    assert abs(b - -106.959626154680) < 1E-5
