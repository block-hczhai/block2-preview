
.. figure:: _static/block2-logo.png
   :width: 200
   :alt: block2 logo

.. only:: latex

   block2
   ======

**block2** is an efficient and highly scalable implementation of the Density Matrix Renormalization Group (DMRG) for quantum chemistry, based on Matrix Product Operator (MPO) formalism. The code is highly optimized for production level calculation of realistic systems. It also provides plenty of options for tuning performance and new algorithm development.

Contributors
------------

* Huanchen Zhai `@hczhai <https://github.com/hczhai>`_: DMRG and parallelization
* Henrik R. Larsson `@h-larsson <https://github.com/h-larsson>`_: DMRG-MRCI
* Zhi-Hao Cui `@zhcui <https://github.com/zhcui>`_: user interface

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
    * MPS compression / addition
    * Expectation
    * Imaginary/Real time evolution
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
* Determinant overlap (non-spin-adapted only)
* Multi-level parallel DMRG
    * Parallelism over sites (2-site only)
    * Parallelism over sum of MPOs (non-spin-adapted only)
    * Parallelism over operators (distributed/shared memory)
    * Parallelism over symmetry sectors (shared memory)
    * Parallelism within dense matrix multiplications (MKL)
* Orbital Reordering
    * Fiedler
    * Genetic algorithm

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user/installation
   user/references

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer/dmrg

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/global
   api/fft