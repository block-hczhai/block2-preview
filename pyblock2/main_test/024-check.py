import sys

eners, dws = [], []

with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if "DW" in l:
            eners.append(float(l.split()[7]))
            dws.append(float(l.split()[-1]))

eners, dws = eners[3::4], dws[3::4]

import scipy.stats
reg = scipy.stats.linregress(dws, eners)

assert abs(eners[0] - -107.6541224474) < 1E-6
assert abs(eners[1] - -107.6541224425) < 1E-6
assert abs(eners[2] - -107.6541223606) < 1E-6
assert abs(eners[3] - -107.6541219540) < 1E-6
assert abs(reg.intercept - -107.654122447525) < 1E-6
