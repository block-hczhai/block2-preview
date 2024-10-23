
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

 - H. Zhai, H. R. Larsson, S. Lee, Z.-H. Cui, T. Zhu, C. Sun, L. Peng, R. Peng, K. Liao, J. Tölle, J. Yang, S. Li, and G. K.-L. Chan. Block2: A comprehensive open source framework to develop and apply state-of-the-art DMRG algorithms in electronic structure and beyond. *The Journal of Chemical Physics* **159**, 234801 (2023). doi: [10.1063/5.0180424](https://doi.org/10.1063/5.0180424)

For parallel ab initio DMRG, please cite

 - H. Zhai, and G. K.-L. Chan. Low communication high performance ab initio density matrix renormalization group algorithms. *The Journal of Chemical Physics* **154**, 224116 (2021). doi: [10.1063/5.0050902](https://doi.org/10.1063/5.0050902).

For large site DMRG-MRCI/MRPT, please cite

 - H. R. Larsson, H. Zhai, K. Gunst, and G. K.-L. Chan. Matrix product states with large sites. *Journal of Chemical Theory and Computation* **18**, 749-762 (2022). doi: [10.1021/acs.jctc.1c00957](https://doi.org/10.1021/acs.jctc.1c00957).

For DMRG with spin-orbit-coupling, please cite

 - H. Zhai, and G. K.-L. Chan. A comparison between the one- and two-step spin-orbit coupling approaches based on the ab initio Density Matrix Renormalization Group. *The Journal of Chemical Physics* **157**, 164108 (2022). doi: [10.1063/5.0107805](https://doi.org/10.1063/5.0107805).

You can find a bibtex file in `CITATIONS.bib`.

One can install ``block2`` using ``pip`` (note: for very new Python versions, the ``--extra-index-url`` option of ``pip`` is required, see below for installing the developement version of ``block2``):

* OpenMP-only version (no MPI dependence)

      pip install block2

* Hybrid openMP/MPI version (requiring openMPI 5.0.x for ``block2-mpi >= 0.5.3`` or 4.1.x for ``block2-mpi <= 0.5.2`` and ``block2-mpi <= 0.5.3rc19``)

      pip install block2-mpi

* Binary format is prepared via ``pip`` for python 3.8, 3.9, 3.10, 3.11, 3.12, and 3.13 with macOS (x86 and arm64, no-MPI), Linux (no-MPI/openMPI), or Windows (x86, no-MPI). If these binaries have some problems, you can use the ``--no-binary`` option of ``pip`` to force building from source (for example, ``pip install block2 --no-binary block2``).

* One should only install one of ``block2`` and ``block2-mpi``. ``block2-mpi`` covers all features in ``block2``, but its dependence on mpi library can sometimes be difficult to deal with. Some guidance for resolving environment problems can be found in issue [#7](https://github.com/block-hczhai/block2-preview/issues/7) and [here](https://block2.readthedocs.io/en/latest/user/installation.html#installation-with-anaconda).

* To install the most recent development version, use:

      pip install block2==<version> --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/
      pip install block2-mpi==<version> --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/

  where ``<version>`` can be some development version number like ``0.5.3rc20`` (see https://github.com/block-hczhai/block2-preview/tags for a complete list of version numbers. The letter ``p`` is not needed). To force reinstalling an updated version, you may consider ``pip`` options ``--upgrade --force-reinstall --no-deps --no-cache-dir``.

The detailed instructions on manual installation can be found [here](https://block2.readthedocs.io/en/latest/user/installation.html#manual-installation).

To run a DMRG calculation using the command line interface, please use the following command:

    block2main dmrg.conf > dmrg.out

where ``dmrg.conf`` is the ``StackBlock`` style input file and ``dmrg.out`` contains the outputs.
Example input files can be found [here](https://block2.readthedocs.io/en/latest/user/basic.html).

For DMRGSCF calculation, please have a look at [here](https://block2.readthedocs.io/en/latest/user/dmrg-scf.html).

For a list of DMRG references for methods implemented in ``block2``, see: https://block2.readthedocs.io/en/latest/user/references.html

Documentation: https://block2.readthedocs.io/en/latest/

Tutorial (python interface): https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html

Example script for models: [Fermi-Hubbard](https://block2.readthedocs.io/en/latest/tutorial/hubbard.html), [Bose-Hubbard](https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#Bose-Hubbard-Model), [Hubbard-Holstein](https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#The-Hubbard-Holstein-Model), [SU(2) Heisenberg](https://block2.readthedocs.io/en/latest/tutorial/heisenberg.html), [SU(3) Heisenberg](https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#SU(3)-Heisenberg-Model), [t-J](https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#SU(2)-t-J-Model).

Source code: https://github.com/block-hczhai/block2-preview

For a simplified implementation of ab initio DMRG, see [pyblock3](https://github.com/block-hczhai/pyblock3-preview). Data can be imported and exported between ``block2`` and ``pyblock3``, see https://github.com/block-hczhai/block2-preview/discussions/35.

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
    * Extracting Determinant/CSF coefficients from MPS
    * Constructing MPS from Determinant/CSF coefficients
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

StackBlock Compatibility
------------------------

A [StackBlock 1.5](https://github.com/sanshar/StackBlock) compatible user interface can be found at `pyblock2/driver/block2main`.
This script can work as a replacement of the StackBlock binary, with a few limitations and some extensions.
The format of the input file `dmrg.conf` is identical to that of StackBlock 1.5.
See `docs/driver.md` and `docs/source/user/basic.rst` for detailed documentations for this interface.
Examples using this interface can be found at `tests/driver`.

Instuctions for installing the StackBlock code can be found in [here](https://block2.readthedocs.io/en/latest/user/mps-io.html#stackblock-installation). A list of precompiled binaries of StackBlock can be found in [here](https://github.com/hczhai/StackBlock/releases/tag/v1.5.3).
