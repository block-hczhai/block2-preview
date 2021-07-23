
[![Documentation Status](https://readthedocs.org/projects/block2/badge/?version=latest)](https://block2.readthedocs.io/en/latest/?badge=latest)

block2
======

The block2 code provides an efficient highly scalable
implementation of the Density Matrix Renormalization Group (DMRG) for quantum chemistry,
based on Matrix Product Operator (MPO) formalism.

The block2 code is developed as an improved version of [StackBlock](https://sanshar.github.io/Block/),
where the low-level structure of the code has been completely rewritten.
The block2 code is developed and maintained in Garnet Chan group at Caltech.

Main contributors:

* Huanchen Zhai [@hczhai](https://github.com/hczhai): DMRG and parallelization
* Henrik R. Larsson [@h-larsson](https://github.com/h-larsson): DMRG-MRCI and big site
* Zhi-Hao Cui [@zhcui](https://github.com/zhcui): user interface

If you find this package useful for your scientific research, please cite the work as:

Zhai, H., Chan, G. K. Low communication high performance ab initio density matrix renormalization group algorithms. *The Journal of Chemical Physics* 2021, **154**, 224116.

One can install ``block2`` using:

    pip install block2

To run a DMRG calculation, please use the following command:

    block2main dmrg.conf > dmrg.out

where ``dmrg.conf`` is the ``StackBlock`` style input file and ``dmrg.out`` contains the outputs.

Documentation: https://block2.readthedocs.io/en/latest/

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
    * Multi-Target Excited-State DMRG
        * State-averaged / state-specific
    * MPS compression / addition
    * Expectation
    * Imaginary / real time evolution
        * Hermitian / non-Hermitian Hamiltonian
        * Time-step targeting method
        * Time dependent variational principle method
    * Green's function
* Finite-Temperature DMRG (ancilla approach)
* Low-Temperature DMRG (partition function approach)
* Particle Density Matrix (1-site / 2-site)
    * 1PDM / 2PDM
    * Transition 1PDM
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
* Determinant/CSF overlap sampling
* Multi-level parallel DMRG
    * Parallelism over sites (2-site only)
    * Parallelism over sum of MPOs (non-spin-adapted only)
    * Parallelism over operators (distributed/shared memory)
    * Parallelism over symmetry sectors (shared memory)
    * Parallelism within dense matrix multiplications (MKL)
* Orbital Reordering
    * Fiedler
    * Genetic algorithm
* MPS Transformation
    * SU2 to SZ mapping
    * Point group mapping
    * Orbital basis rotation

References
----------

### Qauntum Chemisty DMRG

* Chan, G. K.-L.; Head-Gordon, M. Highly correlated calculations with a polynomial cost algorithm: A study of the density matrix renormalization group. *The Journal of Chemical Physics* 2002, **116**, 4462–4476.
* Sharma, S.; Chan, G. K.-L. Spin-adapted density matrix renormalization group algorithms for quantum chemistry. *The Journalof Chemical Physics* 2012, **136**, 124121.
* Wouters, S.; Van Neck, D. The density matrix renormalization group for ab initio quantum chemistry. *The European Physical Journal D* 2014, **68**, 272.

### Parallelization

* Chan, G. K.-L. An algorithm for large scale density matrix renormalization group calculations. *The Journal of Chemical Physics* 2004, **120**, 3172–3178.
* Chan, G. K.-L.; Keselman, A.; Nakatani, N.; Li, Z.; White, S. R. Matrix product operators, matrix product states, and ab initio density matrix renormalization group  algorithms. *The Journal of Chemical Physics* 2016, **145**, 014102.
* Stoudenmire, E.; White, S. R. Real-space parallel density matrix renormalization group. *Physical Review B* 2013, **87**, 155137.
* Zhai, H., Chan, G. K. Low communication high performance ab initio density matrix renormalization group algorithms. *The Journal of Chemical Physics* 2021, **154**, 224116.

### Spin-Orbit Coupling

* Sayfutyarova, E. R., Chan, G. K. L. A state interaction spin-orbit coupling density matrix renormalization group method. *The Journal of Chemical Physics* 2016, **144**, 234301.
* Sayfutyarova, E. R., Chan, G. K. L. Electron paramagnetic resonance g-tensors from state interaction spin-orbit coupling density matrix renormalization group. *The Journal of Chemical Physics* 2018, **148**, 184103.

### Green's Function

* Ronca, E., Li, Z., Jimenez-Hoyos, C. A., Chan, G. K. L. Time-step targeting time-dependent and dynamical density matrix renormalization group algorithms with ab initio Hamiltonians. *Journal of Chemical Theory and Computation* 2017, **13**, 5560-5571.

### Finite-Temperature DMRG

* Feiguin, A. E., White, S. R. Finite-temperature density matrix renormalization using an enlarged Hilbert space. *Physical Review B* 2005, **72**, 220401.
* Feiguin, A. E., White, S. R. Time-step targeting methods for real-time dynamics using the density matrix renormalization group. *Physical Review B* 2005, **72**, 020404.

### Linear Response

* Sharma, S., Chan, G. K. Communication: A flexible multi-reference perturbation theory by minimizing the Hylleraas functional with matrix product states. *Journal of Chemical Physics* 2014, **141**, 111101.

### Perturbative Noise

* White, S. R. Density matrix renormalization group algorithms with a single center site. *Physical Review B* 2005, **72**, 180403.
* Hubig, C., McCulloch, I. P., Schollwöck, U., Wolf, F. A. Strictly single-site DMRG algorithm with subspace expansion. *Physical Review B* 2015, **91**, 155115.

### Particle Density Matrix

* Ghosh, D., Hachmann, J., Yanai, T., & Chan, G. K. L. Orbital optimization in the density matrix renormalization group, with applications to polyenes and β-carotene. *The Journal of Chemical Physics* 2008, **128**, 144117.

### Determinant Coefficients

* Lee, S., Zhai, H., Sharma, S., Umrigar, C. J., Chan, G. K. Externally corrected CCSD with renormalized perturbative triples (R-ecCCSD (T)) and density matrix renormalization group and selected configuration interaction external sources. 2021, arXiv preprint arXiv:2102.12703.

### Orbital Reordering

* Olivares-Amaya, R.; Hu, W.; Nakatani, N.; Sharma, S.; Yang, J.;Chan, G. K.-L. The ab-initio density  matrix renormalization group in practice. *The Journal of Chemical Physics* 2015, **142**, 034102.

Manual Installation
-------------------

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

This may take 11 minutes, requiring 14 GB memory.

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

Sometimes, when you have to use `block2` together with other python modules (such as `pyscf` or `pyblock`),
it may have some problem coexisting with each other.
In general, change the import order may help.
For `pyscf`, `import block2` at the very beginning of the script may help.
For `pyblock`, recompiling `block2` use `cmake .. -DUSE_MKL=OFF -DBUILD_LIB=ON -OMP_LIB=SEQ -DLARGE_BOND=ON` may help.

Usage
-----

The code can either be used as a binary executable or through python interface.

The following are some examples using the python interface.

### GS-DMRG

Test Ground-State DMRG (need `pyscf` module):

    python3 -m pyblock2.gsdmrg

###  FT-DMRG

Test Finite-Temperature (FT)-DMRG (need `pyscf` module):

    python3 -m pyblock2.ftdmrg

### LT-DMRG

Test Low-Temperature (LT)-DMRG (need `pyscf` module):

    python3 -m pyblock2.ltdmrg

### GF-DMRG

Test Green's-Function (GF)-DMRG (DDMRG++) (need `pyscf` module):

    python3 -m pyblock2.gfdmrg

### SI-DMRG

Test State-Interaction (SI)-DMRG (need `pyscf` module):

    python3 -m pyblock2.sidmrg

### StackBlock Compatibility

A StackBlock 1.5 compatible user interface can be found at `pyblock2/driver/block2main`.
This script can work as a replacement of the StackBlock binary, with a few limitations and some extensions.
The format of the input file `dmrg.conf` is identical to that of StackBlock 1.5.
See `docs/driver.md` for detailed documentations for this interface.
Examples using this interface can be found at `tests/driver`.

### Input File (block2 style)

Example input file for binary executable `build/block2`:

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

### Using C++ Interpreter cling

Since `block2` is designed as a header-only C++ library, it can be conveniently executed
using C++ interpreter [cling](https://github.com/root-project/cling)
(which can be installed via [anaconda](https://anaconda.org/conda-forge/cling))
without any compilation. This can be useful for testing samll changes in the C++ code.

Example C++ code for `cling` can be found at `tests/cling/hubbard.cl`.
