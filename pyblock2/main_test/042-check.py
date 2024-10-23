import sys

ovlp = 0
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('Compression overlap ='):
            ovlp = float(l.split()[-1])

assert abs(ovlp - 0.957506527014452) < 1E-6
