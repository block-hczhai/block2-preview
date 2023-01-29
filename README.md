
[![Documentation Status](https://readthedocs.org/projects/block2/badge/?version=latest)](https://block2.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/block-hczhai/block2-preview/workflows/build/badge.svg)](https://github.com/block-hczhai/block2-preview/actions/workflows/build.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/block2.svg)](https://badge.fury.io/py/block2)

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
* Henrik R. Larsson [@h-larsson](https://github.com/h-larsson): DMRG-MRCI/MRPT, large site, Green's function in frequency and time for finite temp.  
* Seunghoon Lee [@seunghoonlee89](https://github.com/seunghoonlee89): Stochastic perturbative DMRG
* Zhi-Hao Cui [@zhcui](https://github.com/zhcui): User interface

If you find this package useful for your scientific research, please cite the work as:

 - Zhai, H., Chan, G. K. Low communication high performance ab initio density matrix renormalization group algorithms. *The Journal of Chemical Physics* 2021, **154**, 224116, doi: [10.1063/5.0050902](https://doi.org/10.1063/5.0050902).

For the large site code, please cite

 - Larsson, H. R., Zhai, H., Gunst, K., Chan, G. K. L. Matrix product states with large sites. *Journal of Chemical Theory and Computation* 2022, **18**, 749-762,  doi: [10.1021/acs.jctc.1c00957](https://doi.org/10.1021/acs.jctc.1c00957).

You can find a bibtex file in `CITATIONS.bib`.


One can install ``block2`` using ``pip``:

* OpenMP-only version (no MPI dependence)

      pip install block2

* Hybrid openMP/MPI version (requiring openMPI 4.0.x and ``mpi4py`` based on the same openMPI library installed)

      pip install block2-mpi

* Binary format is prepared via ``pip`` for python 3.6, 3.7, 3.8, and 3.9 with macOS (no-MPI) or Linux (no-MPI/openMPI). If these binaries have some problems, you can use the ``--no-binary`` option of ``pip`` to force building from source (for example, ``pip install block2 --no-binary block2``).

* One should only install one of ``block2`` and ``block2-mpi``. ``block2-mpi`` covers all features in ``block2``, but its dependence on mpi library can sometimes be difficult to deal with. Some guidance for resolving environment problems can be found in issue [#7](https://github.com/block-hczhai/block2-preview/issues/7) and [here](https://block2.readthedocs.io/en/latest/user/installation.html#installation-with-anaconda).

* To install the most recent development version, use:

      pip install block2==<version> --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/
      pip install block2-mpi==<version> --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/

  where ``<version>`` can be some development version number like ``0.5.1rc14``.

To run a DMRG calculation using the cmmand line interface, please use the following command:

    block2main dmrg.conf > dmrg.out

where ``dmrg.conf`` is the ``StackBlock`` style input file and ``dmrg.out`` contains the outputs.

For DMRGSCF calculation, please have a look at the [documentation](https://block2.readthedocs.io/en/latest/user/dmrg-scf.html).

Documentation: https://block2.readthedocs.io/en/latest/

Tutorial (python interface): https://block2.readthedocs.io/en/latest/tutorial/hubbard.html

Source code: https://github.com/block-hczhai/block2-preview

Features
--------

* State symmetry
    * U(1) particle number symmetry
    * SU(2) or U(1) spin symmetry (spatial orbital)
    * No spin symmetry (general spin orbital)
    * Abelian point group symmetry
    * Translational (K point) / Lz symmetry
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
    * Green's function
    * Time evolution
* Low-Temperature DMRG (partition function approach)
* Particle Density Matrix (1-site / 2-site)
    * 1PDM / 2PDM / 3PDM / 4PDM
    * Transition 1PDM / 2PDM / 3PDM / 4PDM
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
* Determinant/CSF coefficients of MPS
* Multi-level parallel DMRG
    * Parallelism over sites (2-site only)
    * Parallelism over sum of MPOs (distributed)
    * Parallelism over operators (distributed/shared memory)
    * Parallelism over symmetry sectors (shared memory)
    * Parallelism within dense matrix multiplications (MKL)
* DMRG-CASSCF and contracted dynamic correlation
    * DMRG-CASSCF (pyscf / openMOLCAS / forte interface)
    * DMRG-CASSCF nuclear gradients and geometry optimization (pyscf interface, RHF reference only)
    * DMRG-sc-NEVPT2 (pyscf interface, classical approach)
    * DMRG-sc-MPS-NEVPT2 (pyscf interface, MPS compression approximation)
    * DMRG-CASPT2 (openMOLCAS interface)
    * DMRG-cu-CASPT2 (openMOLCAS interface)
    * DMRG-MRDSRG (forte interface)
* Stochastic perturbative DMRG
* DMRG with Spin-Orbit Coupling (SOC)
    * 1-step approach (full complex one-MPO and hybrid real/complex two-MPO schemes)
    * 2-step approach
* Uncontracted dynamic correlation
    * DMRG Multi-Reference Configuration Interaction (MRCI) of arbitrary order
    * DMRG Multi-Reference Averaged Quadratic Coupled Cluster (AQCC)/ Coupled Pair Functional (ACPF)
    * DMRG NEVPT2/3/..., REPT2/3/..., MR-LCC, ...
* Orbital Reordering
    * Fiedler
    * Genetic algorithm
* MPS Transformation
    * SU2 to SZ mapping
    * Point group mapping
    * Orbital basis rotation

References
----------

### Quantum Chemisty DMRG

* Chan, G. K.-L.; Head-Gordon, M. Highly correlated calculations with a polynomial cost algorithm: A study of the density matrix renormalization group. *The Journal of Chemical Physics* 2002, **116**, 4462–4476. https://doi.org/10.1063/1.1449459.
* Sharma, S.; Chan, G. K.-L. Spin-adapted density matrix renormalization group algorithms for quantum chemistry. *The Journalof Chemical Physics* 2012, **136**, 124121. https://doi.org/10.1063/1.3695642.
* Wouters, S.; Van Neck, D. The density matrix renormalization group for ab initio quantum chemistry. *The European Physical Journal D* 2014, **68**, 272. https://doi.org/10.1140/epjd/e2014-50500-1.

### Parallelization

* Chan, G. K.-L. An algorithm for large scale density matrix renormalization group calculations. *The Journal of Chemical Physics* 2004, **120**, 3172–3178. https://doi.org/10.1063/1.1638734.
* Chan, G. K.-L.; Keselman, A.; Nakatani, N.; Li, Z.; White, S. R. Matrix product operators, matrix product states, and ab initio density matrix renormalization group  algorithms. *The Journal of Chemical Physics* 2016, **145**, 014102. https://doi.org/10.1063/1.4955108.
* Stoudenmire, E.; White, S. R. Real-space parallel density matrix renormalization group. *Physical Review B* 2013, **87**, 155137. https://doi.org/10.1103/PhysRevB.87.155137.
* Zhai, H., Chan, G. K. Low communication high performance ab initio density matrix renormalization group algorithms. *The Journal of Chemical Physics* 2021, **154**, 224116. https://doi.org/10.1063/5.0050902.

### Spin-Orbit Coupling

* Sayfutyarova, E. R., Chan, G. K. L. A state interaction spin-orbit coupling density matrix renormalization group method. *The Journal of Chemical Physics* 2016, **144**, 234301. https://doi.org/10.1063/1.4953445.
* Sayfutyarova, E. R., Chan, G. K. L. Electron paramagnetic resonance g-tensors from state interaction spin-orbit coupling density matrix renormalization group. *The Journal of Chemical Physics* 2018, **148**, 184103. https://doi.org/10.1063/1.5020079.
* Zhai, H., Chan, G. K. A comparison between the one- and two-step spin-orbit coupling approaches based on the ab initio Density Matrix Renormalization Group. *The Journal of Chemical Physics* 2022, **157**, 164108. https://doi.org/10.1063/5.0107805.

### Green's Function

* Ronca, E., Li, Z., Jimenez-Hoyos, C. A., Chan, G. K. L. Time-step targeting time-dependent and dynamical density matrix renormalization group algorithms with ab initio Hamiltonians. *Journal of Chemical Theory and Computation* 2017, **13**, 5560-5571. https://doi.org/10.1021/acs.jctc.7b00682.

### Finite-Temperature DMRG

* Feiguin, A. E., White, S. R. Finite-temperature density matrix renormalization using an enlarged Hilbert space. *Physical Review B* 2005, **72**, 220401. https://doi.org/10.1103/PhysRevB.72.220401.
* Feiguin, A. E., White, S. R. Time-step targeting methods for real-time dynamics using the density matrix renormalization group. *Physical Review B* 2005, **72**, 020404. https://doi.org/10.1103/PhysRevB.72.020404.

### Linear Response

* Sharma, S., Chan, G. K. Communication: A flexible multi-reference perturbation theory by minimizing the Hylleraas functional with matrix product states. *Journal of Chemical Physics* 2014, **141**, 111101. https://doi.org/10.1063/1.4895977.

### Perturbative Noise

* White, S. R. Density matrix renormalization group algorithms with a single center site. *Physical Review B* 2005, **72**, 180403. https://doi.org/10.1103/PhysRevB.72.180403.
* Hubig, C., McCulloch, I. P., Schollwöck, U., Wolf, F. A. Strictly single-site DMRG algorithm with subspace expansion. *Physical Review B* 2015, **91**, 155115. https://doi.org/10.1103/PhysRevB.91.155115.

### Particle Density Matrix

* Ghosh, D., Hachmann, J., Yanai, T., Chan, G. K. L. Orbital optimization in the density matrix renormalization group, with applications to polyenes and β-carotene. *The Journal of Chemical Physics* 2008, **128**, 144117. https://doi.org/10.1063/1.2883976.
* Guo, S., Watson, M. A., Hu, W., Sun, Q., Chan, G. K. L. N-electron valence state perturbation theory based on a density matrix renormalization group reference function, with applications to the chromium dimer and a trimer model of poly (p-phenylenevinylene). *Journal of Chemical Theory and Computation* 2016, **12**, 1583-1591. https://doi.org/10.1021/acs.jctc.5b01225.

### DMRG-SC-NEVPT2

* Roemelt, M., Guo, S., Chan, G. K. L. A projected approximation to strongly contracted N-electron valence perturbation theory for DMRG wavefunctions. *The Journal of Chemical Physics* 2016, **144**, 204113. https://doi.org/10.1063/1.4950757.
* Sokolov, A. Y., Guo, S., Ronca, E., Chan, G. K. L. Time-dependent N-electron valence perturbation theory with matrix product state reference wavefunctions for large active spaces and basis sets: Applications to the chromium dimer and all-trans polyenes. *The Journal of Chemical Physics* 2017, **146**, 244102. https://doi.org/10.1063/1.4986975.

### DMRG-CASPT2

* Kurashige, Y., Yanai, T. Second-order perturbation theory with a density matrix renormalization group self-consistent field reference function: Theory and application to the study of chromium dimer. *The Journal of Chemical Physics* 2011, **135**, 094104. https://doi.org/10.1063/1.3629454.
* Wouters, S., Van Speybroeck, V., Van Neck, D. DMRG-CASPT2 study of the longitudinal static second hyperpolarizability of all-trans polyenes. *The Journal of Chemical Physics* 2016, **145**, 054120. https://doi.org/10.1063/1.4959817.
* Nakatani, N., Guo, S. Density matrix renormalization group (DMRG) method as a common tool for large active-space CASSCF/CASPT2 calculations. *The Journal of Chemical Physics* 2017, **146**, 094102. https://doi.org/10.1063/1.4976644.

### Multi-Reference Correlation Theories

* Szalay, P. G.; Müller, T.; Gidofalvi, G.; Lischka, H.; Shepard, R. Multiconfiguration Self-Consistent Field and Multireference Configuration Interaction Methods and Applications. *Chemical Reviews* 2012, **112**, 108-181. https://doi.org/10.1021/cr200137a.
* Gdanitz, R. J., Ahlrichs, R. The Averaged Coupled-Pair Functional (ACPF): A Size-Extensive Modification of MR CI(SD). *Chemical Physics Letters* 1988, **143**, 413-420. https://doi.org/10.1016/0009-2614(88)87388-3.
* Szalay, P. G., Bartlett, R. J. Multi-Reference Averaged Quadratic Coupled-Cluster Method: A Size-Extensive Modification of Multi-Reference CI. *Chemical Physics Letters* 1993, **214**, 481-488. https://doi.org/10.1016/0009-2614(93)85670-J.

* Laidig, W. D.; Bartlett, R. J. A Multi-Reference Coupled-Cluster Method for Molecular Applications. *Chemical Physics Letters* 1984, **104**, 424-430. https://doi.org/10.1016/0009-2614(84)85617-1.
* Laidig, W. D., Saxe, P., Bartlett, R. J. The Description of N 2 and F 2 Potential Energy Surfaces Using Multireference Coupled Cluster Theory. *The Journal of Chemical Physics* 1987, **86**, 887-907. https://doi.org/10.1063/1.452291.

* Angeli, C., Cimiraglia, R., Evangelisti, S., Leininger, T., Malrieu, J.-P. Introduction of N-Electron Valence States for Multireference Perturbation Theory. *J. Chem. Phys.* 2001, **114**, 10252–10264. https://doi.org/10.1063/1.1361246.
* Angeli, C., Cimiraglia, R., Malrieu J.-P. N-electron valence state perturbation theory: A spinless formulation and an efficient implementation of the strongly contracted and of the partially contracted variants. *The Journal of chemical physics* 2002, **117**, 9138-9153. https://doi.org/10.1063/1.1515317.
* Angeli, C., Pastore, M., Cimiraglia, R. New Perspectives in Multireference Perturbation Theory: The n-Electron Valence State Approach. *Theor Chem Account* 2007, **117**,  743–754. https://doi.org/10.1007/s00214-006-0207-0.

* Fink, R. F. The Multi-Reference Retaining the Excitation Degree Perturbation Theory: A Size-Consistent, Unitary Invariant, and Rapidly Convergent Wavefunction Based Ab Initio Approach. *Chemical Physics* 2009, **356**, 39-46. https://doi.org/10.1016/j.chemphys.2008.10.004.
* Fink, R. F. Two New Unitary-Invariant and Size-Consistent Perturbation Theoretical Approaches to the Electron Correlation Energy. *Chemical Physics Letters* 2006, **428**, 461–466. https://doi.org/10.1016/j.cplett.2006.07.081.

* Sharma, S., Chan, G. K.-L. Communication: A Flexible Multi-Reference Perturbation Theory by Minimizing the Hylleraas Functional with Matrix Product States. *The Journal of Chemical Physics* 2014, **141**, 111101. https://doi.org/10.1063/1.4895977.
* Sharma, S., Alavi, A. Multireference Linearized Coupled Cluster Theory for Strongly Correlated Systems Using Matrix Product States. *The Journal of Chemical Physics* 2015, **143**, 102815. https://doi.org/10.1063/1.4928643.
* Sharma, S., Jeanmairet, G., Alavi, A. Quasi-Degenerate Perturbation Theory Using Matrix Product States. *The Journal of Chemical Physics* 2016, **144**, 034103. https://doi.org/10.1063/1.4939752.

* Larsson, H. R., Zhai, H., Gunst, K., Chan, G. K. L. Matrix product states with large sites. *Journal of Chemical Theory and Computation* 2022, **18**, 749-762. https://doi.org/10.1021/acs.jctc.1c00957.

### Determinant Coefficients

* Lee, S., Zhai, H., Sharma, S., Umrigar, C. J., Chan, G. K. L. Externally Corrected CCSD with Renormalized Perturbative Triples (R-ecCCSD (T)) and the Density Matrix Renormalization Group and Selected Configuration Interaction External Sources. *Journal of Chemical Theory and Computation* 2021, **17**, 3414-3425. https://doi.org/10.1021/acs.jctc.1c00205.

### Perturbative DMRG

* Guo, S., Li, Z., Chan, G. K. L. Communication: An efficient stochastic algorithm for the perturbative density matrix renormalization group in large active spaces. *The Journal of chemical physics* 2018, **148**, 221104. https://doi.org/10.1063/1.5031140.
* Guo, S., Li, Z., Chan, G. K. L. A perturbative density matrix renormalization group algorithm for large active spaces. *Journal of chemical theory and computation* 2018, **14**, 4063-4071. https://doi.org/10.1021/acs.jctc.8b00273.

### Orbital Reordering

* Olivares-Amaya, R.; Hu, W.; Nakatani, N.; Sharma, S.; Yang, J.;Chan, G. K.-L. The ab-initio density  matrix renormalization group in practice. *The Journal of Chemical Physics* 2015, **142**, 034102. https://doi.org/10.1063/1.4905329.

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

One can add both the repo root directory and the ``build`` directory into ``PYTHONPATH`` so that ``import block2`` and ``import pyblock2`` will work.

### MKL

If `-DUSE_MKL=ON` is not given, `blas` and `lapack` are required. Sometimes, the `blas` and `lapack` function names can contain the extra underscore. Therefore, it is recommended to use `-DUSE_MKL=OFF` and `-DF77UNDERSCORE=ON` together to prevent this underscore problem. If this generates the undefined reference error, one should try `-DUSE_MKL=OFF -DF77UNDERSCORE=OFF` instead.

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

Note for developers : for `CSR sparse MKL + ThreadingTypes::Operator`, if `-DOMP_LIB=GNU`,
it is not possible to set both `n_threads_mkl` not equal to 1 and `n_threads_op` not equal to 1.
In other words, nested openMP is not possible for CSR sparse matrix (generating wrong result/non-convergence).
For `-DOMP_LIB=SEQ`, CSR sparse matrix is okay (non-nested openMP).
For `-DOMP_LIB=TBB`, nested openMP + TBB MKL is okay.

`-DTBB=ON` can be combined with any `-DOMP_LIB=...`.

### Complex mode

For complex integrals / spin-orbit coupling (SOC), extra options ``-DUSE_COMPLEX=ON`` and ``-DUSE_SG=ON`` are required (and the compilation time will increase).

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

* Linux + gcc 9.2.0 + MKL 2021.4
* MacOS 10.15 + Apple clang 12.0 + MKL 2021 (MKL 2021.4 required for `pip install`)
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
See `docs/driver.md` and `docs/source/user/basic.rst` for detailed documentations for this interface.
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
without any compilation. This can be useful for testing small changes in the C++ code.

Example C++ code for `cling` can be found at `tests/cling/hubbard.cl`.
