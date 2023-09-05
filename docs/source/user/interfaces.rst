
Interfaces
==========

``block2`` can be used via many different interfaces.

Input File
----------

Like many quantum chemistry packages, ``block2`` can be used by reading parameters and instrutions from a formatted input file.
This interface is ``StackBlock`` compatible.
See :ref:`user_basic`, :ref:`user_advanced`, and :ref:`user_keywords`.

Interfaces for DMRGSCF
----------------------

To do DMRGSCF, we need to connect ``block2`` to some external softwares for the CASSCF part.
See :ref:`user_dmrgscf`, :ref:`user_open_molcas`, and :ref:`user_forte`.

Python Interface (low level)
----------------------------

General examples:

1. GS-DMRG

Test Ground-State DMRG (need `pyscf` module): ::

    python3 -m pyblock2.gsdmrg

2. FT-DMRG

Test Finite-Temperature (FT)-DMRG (need `pyscf` module): ::

    python3 -m pyblock2.ftdmrg

3. LT-DMRG

Test Low-Temperature (LT)-DMRG (need `pyscf` module): ::

    python3 -m pyblock2.ltdmrg

4. GF-DMRG

Test Green's-Function (GF)-DMRG (DDMRG++) (need `pyscf` module): ::

    python3 -m pyblock2.gfdmrg

5. SI-DMRG

Test State-Interaction (SI)-DMRG (need `pyscf` module): ::

    python3 -m pyblock2.sidmrg

For special topics, see :ref:`dev_mpo_reloading`, :ref:`dev_orbital_rotation`, :ref:`dev_pg_mapping`.

Python Interface (high level)
-----------------------------

See https://block2.readthedocs.io/en/latest/tutorial/qc-hamiltonians.html.

Input File (C++ Executable)
---------------------------

Example input file for binary executable ``build/block2``: ::

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

To run this example: ::

    ./build/block2 input.txt

C++ Interpreter
---------------

Since ``block2`` is designed as a header-only C++ library, it can be conveniently executed
using C++ interpreter [cling](https://github.com/root-project/cling)
(which can be installed via [anaconda](https://anaconda.org/conda-forge/cling))
without any compilation. This can be useful for testing small changes in the C++ code.

Example C++ code for ``cling`` can be found at ``tests/cling/hubbard.cl``.
