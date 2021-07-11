#!/usr/bin/env bash

# Test for the block2 driver using input file format
# dmrg.conf/gaopt.conf from StackBlock 1.5

DMRG=../../pyblock2/driver/block2.py
GAOPT=../../pyblock2/driver/gaopt.py

echo TEST DMRG ...
python3 -u $DMRG dmrg.conf

echo TEST DMRG MPI ...
mpirun --bind-to core --map-by ppr:4:node:pe=4 python3 -u $DMRG dmrg.conf

# DMRG Energy =   -2.121631794832947

echo TEST GAOPT ...
python3 -u $GAOPT -config gaopt.conf -integral FCIDUMP
python3 -u $GAOPT -config gaopt.conf -integral ints.h5
python3 -u $GAOPT -config gaopt.conf -s -integral kmat

echo TEST GAOPT MPI ...
mpirun --bind-to core --map-by ppr:4:node:pe=4 python3 -u $GAOPT -config gaopt.conf -integral FCIDUMP
mpirun --bind-to core --map-by ppr:4:node:pe=4 python3 -u $GAOPT -config gaopt.conf -integral ints.h5
mpirun --bind-to core --map-by ppr:4:node:pe=4 python3 -u $GAOPT -config gaopt.conf -s -integral kmat

# MINIMUM / f =       1.701655172562
# DMRG REORDER FORMAT
# 1,3,2,4,5,6
