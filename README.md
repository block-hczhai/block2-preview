
block2
======

Efficient MPO implementation of quantum chemistry DMRG

Copyright (C) 2020 Huanchen Zhai

Features
--------

* State symmetry
    * U(1) particle number symmetry
    * SU(2) or U(1) spin symmetry
    * Abelian point group symmetry
* Sweep algorithms (2-site algorithm)
    * Ground-State DMRG
    * Multi-Target State-Averaged Excited-State DMRG
    * MPS compression
    * Expectation
    * Imaginary time evolution (tangent space and RK4 method)
* Finite-Temperature DMRG (ancilla approach)
* Low-Temperature DMRG (partition function approach)
* One Particle Density Matrix
* Quantum Chemistry MPO
    * Normal-Complementary (NC) partition
    * Complementary-Normal (CN) partition
    * Conventional scheme (switch between NC and CN near the middle site)
* Symbolic MPO simplification
* MPS initialization using occupation number
* Fermionic MPS algebra (U(1) spin symmetry)

Installation
------------

Dependence: `pybind11`, `python3`, and `mkl`. For unit tests, `googletest` is required.

`cmake` (version >= 3.0) can be used to compile C++ part of the code, as follows:

    mkdir build
    cd build
    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON
    make -j 10

This will build the python extension (using 10 CPU cores).

By default, the C++ templates will be explicitly instantiated in different compilation units, so that parallel
compilation is possible.

Alternatively, one can do single-file compilation using `-DEXP_TMPL=NONE`, then total compilation time can be
saved by avoiding unnecessary template instantiation, as follows:

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DEXP_TMPL=NONE
    make -j 1

This may take 5 minutes, need 7 to 10 GB memory.

To build unit tests and binary executable (instead of python extension), use the following:

    cmake .. -DUSE_MKL=ON -DBUILD_TEST=ON

If intel openMP library `libiomp5` is not available, one can use gnu openMP library.
The following will switch to gnu openMP library:

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=GNU

The following will use sequential mkl library:

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=SEQ

FT-DMRG
-------

Test Finite-Temperature (FT)-DMRG (need `pyscf` module):

    python3 -m pyblock2.ftdmrg

LT-DMRG
-------

Test Low-Temperature (LT)-DMRG (need `pyscf` module):

    python3 -m pyblock2.ltdmrg

Input File
----------

The code can either be used as a binary executable or through python interface.

Example input file for binary executable:

    rand_seed = 1000
    memory = 4E9
    scratch = ./scratch

    pg = c1
    fcidump = data/HUBBARD-L16.FCIDUMP
    mkl_threads = 1
    qc_type = conventional

    # print_mpo
    print_mpo_dims
    print_fci_dims
    print_mps_dims

    bond_dims = 500
    noises = 1E-6 1E-6 0.0

    center = 0
    dot = 2

    n_sweeps = 10
    tol = 1E-7
    forward = 1

    noise_type = density_matrix
    trunc_type = physical

To run this example:

    ./build/block2 input.txt

Known Bugs
----------

* Imaginary TE not working with `SeqTypes::None`.
* MPS algebra not working with MPS containing 2-site tensor.
