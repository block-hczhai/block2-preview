
.. figure:: _static/block2-logo.png
   :width: 200
   :alt: block2 logo

.. only:: latex

   block2
   ######

**block2** is an efficient and highly scalable implementation of the Density Matrix Renormalization Group (DMRG)
for quantum chemistry, based on Matrix Product Operator (MPO) formalism.
The code is highly optimized for production level calculation of realistic systems.
It also provides plenty of options for tuning performance and new algorithm development.

The block2 code is developed as an improved version of `StackBlock <https://sanshar.github.io/Block/>`_,
where the low-level structure of the code has been completely rewritten.
The block2 code is developed and maintained in Garnet Chan group at Caltech
and Initiative for Computational Catalysis at Flatiron Institute.

Documentation: https://block2.readthedocs.io/en/latest/

Tutorial (python interface): https://block2.readthedocs.io/en/latest/tutorial/hubbard.html

Custom model Hamiltonians can be supported via a Python interface:
`Fermi-Hubbard <https://block2.readthedocs.io/en/latest/tutorial/hubbard.html>`_,
`Bose-Hubbard <https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#Bose-Hubbard-Model>`_,
`Hubbard-Holstein <https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#The-Hubbard-Holstein-Model>`_,
`SU(2) Heisenberg <https://block2.readthedocs.io/en/latest/tutorial/heisenberg.html>`_,
`SU(3) Heisenberg <https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#SU(3)-Heisenberg-Model>`_,
`t-J <https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#SU(2)-t-J-Model>`_,
`Correlation functions <https://block2.readthedocs.io/en/latest/tutorial/custom-hamiltonians.html#Correlation-Functions>`_.

Source code: https://github.com/block-hczhai/block2-preview

Contributors
""""""""""""

* Huanchen Zhai `@hczhai <https://github.com/hczhai>`_: DMRG and parallelization
* Henrik R. Larsson `@h-larsson <https://github.com/h-larsson>`_: DMRG-MRCI/MRPT and big site
* Seunghoon Lee `@seunghoonlee89 <https://github.com/seunghoonlee89>`_: Stochastic perturbative DMRG
* Zhi-Hao Cui `@zhcui <https://github.com/zhcui>`_: user interface

Features
""""""""

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
* DMRG with Spin-Orbit Coupling (SOC)
    * 1-step approach (full complex one-MPO and hybrid real/complex two-MPO schemes)
    * 2-step approach
* Stochastic perturbative DMRG
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

.. raw:: latex

   \chapter{User Guide}

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/installation
   user/interfaces
   user/basic
   user/advanced
   user/keywords
   user/dmrg-scf
   user/open-molcas
   user/forte
   user/mps-io
   user/references

.. raw:: latex

   \chapter{Python Interface Tutorial}

.. toctree::
   :maxdepth: 2
   :caption: Python Interface Tutorial

   tutorial/qc-hamiltonians
   tutorial/energy-extrapolation
   tutorial/restarting-dmrg
   tutorial/dmrg-soc
   tutorial/greens-function
   tutorial/custom-hamiltonians
   tutorial/hubbard
   tutorial/heisenberg
   tutorial/vibrational-hamiltonians
   tutorial/mpo-mps-quimb

.. raw:: latex

   \chapter{Developer Guide}

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/dmrg
   developer/orbital-rotation
   developer/pg-mapping
   developer/mpo-reloading
   developer/hints
   developer/notes

.. raw:: latex

   \chapter{API Reference}

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/pyblock2
   api/global
   api/sparse_matrix
   api/tensor_functions
   api/tools

.. raw:: latex

   \chapter{Theory}

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/hamiltonian
