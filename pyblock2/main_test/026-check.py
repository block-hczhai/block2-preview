import sys

with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if '<Norm^2>' in l:
            norm_sq = float(l.split()[-1])
            assert abs(norm_sq - 1.0) < 1E-6
