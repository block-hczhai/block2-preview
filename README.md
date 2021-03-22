
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
* Sweep algorithms (1-site / 2-site / 2-site to 1-site transition)
    * Ground-State DMRG
        * Decomposition types: density matrix / SVD
        * Noise types: wavefunction / density matrix / perturbative
    * Multi-Target State-Averaged Excited-State DMRG
    * MPS compression
    * Expectation
    * Imaginary time evolution (tangent space / RK4)
    * Green's function
* Finite-Temperature DMRG (ancilla approach)
* Low-Temperature DMRG (partition function approach)
* Particle Density Matrix (1-site / 2-site)
    * 1PDM
    * 2PDM (non-spin-adapted only)
    * Spin / charge correlation
* Quantum Chemistry MPO
    * Normal-Complementary (NC) partition
    * Complementary-Normal (CN) partition
    * Conventional scheme (switch between NC and CN near the middle site)
* Symbolic MPO simplification
* MPS initialization using occupation number
* Supported matrix representation of site operators
    * Block-sparse (outer) / dense (inner)
    * Block-sparse (outer) / elementwise-sparse (CSR, inner)
* Fermionic MPS algebra (non-spin-adapted only)
* Determinant overlap (non-spin-adapted only)
* Multi-level parallel DMRG
    * Parallelism over sites (2-site only)
    * Parallelism over sum of MPOs (non-spin-adapted only)
    * Parallelism over operators (distributed/shared memory)
    * Parallelism over symmetry sectors (shared memory)
    * Parallelism over dense matrix multiplications (MKL)
* Orbital Reordering
    * Fiedler
    * Genetic algorithm

Installation
------------

Dependence: `pybind11`, `python3`, and `mkl` (or `blas + lapack`).

For distributed parallel calculation, `mpi` library is required.

For unit tests, `googletest` is required.

`cmake` (version >= 3.0) can be used to compile C++ part of the code, as follows:

    mkdir build
    cd build
    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DLARGE_BOND=ON
    make -j 10

This will build the python extension (using 10 CPU cores) (serial code).

### MKL

If `-DUSE_MKL=ON` is not given, `blas` and `lapack` are required (with limited support for multi-threading).

Use `-DUSE_MKL64=ON` instead of `-DUSE_MKL=ON` to enable using matrices with 64-bit integer type.

### Serial compilation

By default, the C++ templates will be explicitly instantiated in different compilation units, so that parallel
compilation is possible.

Alternatively, one can do single-file compilation using `-DEXP_TMPL=NONE`, then total compilation time can be
saved by avoiding unnecessary template instantiation, as follows:

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DEXP_TMPL=NONE
    make -j 1

This may take 5 minutes, need 7 to 10 GB memory.

### MPI version

Adding option `-DMPI=ON` will build MPI parallel version. The C++ compiler and MPI library must be matched.
If necessary, environment variables `CC`, `CXX`, and `MPIHOME` can be used to explicitly set the path.

For mixed `openMP/MPI`, use `mpirun --bind-to none -n ...` or `mpirun --bind-to core --map-by ppr:$NPROC:node:pe=$NOMPT ...` to execute binary.

### Binary build

To build unit tests and binary executable (instead of python extension), use the following:

    cmake .. -DUSE_MKL=ON -DBUILD_TEST=ON

### TBB (Intel Threading Building Blocks)

Adding (optional) option `-DTBB=ON` will utilize `malloc` from `tbbmalloc`.
This can improve multi-threading performance.

### openMP

If gnu openMP library `libgomp` is not available, one can use intel openMP library.

The following will switch to intel openMP library (incompatible with `-fopenmp`):

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=INTEL

The following will use sequential mkl library:

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=SEQ

The following will use tbb mkl library:

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=TBB -DTBB=ON

Note: for `CSR sparse MKL + ThreadingTypes::Operator`, if `-DOMP_LIB=GNU`,
it is not possible to set both `n_threads_mkl` not equal to 1 and `n_threads_op` not equal to 1.
In other words, nested openMP is not possible for CSR sparse matrix (generating wrong result/non-convergence).
For `-DOMP_LIB=SEQ`, CSR sparse matrix is okay (non-nested openMP).
For `-DOMP_LIB=TBB`, nested openMP + TBB MKL is okay.

`-DTBB=ON` can be combined with any `-DOMP_LIB=...`.

### Maximal bond dimension

The default maximal allowed bond dimension per symmetry block is `65535`.
Adding option `-DSMALL_BOND=ON` will change this value to `255`.
Adding option `-DLARGE_BOND=ON` will change this value to `4294967295`.

### Release build

The release mode is controlled by CMAKE_BUILD_TYPE:

    cmake .. -DCMAKE_BUILD_TYPE=Release

will use optimization flags such as -O3 (default).

    cmake .. -DCMAKE_BUILD_TYPE=Debug

enables debug flags.

### Supported operating systems and compilers

* Linux + gcc 9.2.0 + MKL 2019
* MacOS 10.15 + Apple clang 12.0 + MKL 2021
* MacOS 10.15 + icpc 2021.1 + MKL 2021
* Windows 10 + Visual Studio 2019 (MSVC 14.28) + MKL 2021

GS-DMRG
-------

Test Ground-State DMRG (need `pyscf` module):

    python3 -m pyblock2.gsdmrg

FT-DMRG
-------

Test Finite-Temperature (FT)-DMRG (need `pyscf` module):

    python3 -m pyblock2.ftdmrg

LT-DMRG
-------

Test Low-Temperature (LT)-DMRG (need `pyscf` module):

    python3 -m pyblock2.ltdmrg

GF-DMRG
-------

Test Green's-Function (GF)-DMRG (DDMRG++) (need `pyscf` module):

    python3 -m pyblock2.gfdmrg

Input File
----------

The code can either be used as a binary executable or through python interface.

Example input file for binary executable:

    rand_seed = 1000
    memory = 4E9
    scratch = ./scratch

    pg = c1
    fcidump = data/HUBBARD-L16.FCIDUMP
    n_threads = 4
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

    noise_type = perturbative
    trunc_type = physical

To run this example:

    ./build/block2 input.txt
