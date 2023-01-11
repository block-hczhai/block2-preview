
.. highlight:: bash

.. _user_open_molcas:

DMRGSCF (OpenMOLCAS)
====================

In this section we explain how to use ``block2`` and ``OpenMOLCAS`` for CASSCF and CASPT2 with DMRG as the active space solver.

Preparation
-----------

First, make sure ``block2`` is installed correctly (either compiled manually or installed using ``pip``,
and for ``pip`` the version of block2 should be ``>=0.5.1rc17``),
so that the command ``which block2main`` can print a valid file path to ``block2main``.

For example, the required ``block2`` can be installed using ``pip`` as: ::

    pip install block2>=0.5.1rc17 --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/

Then we need to compile an OpenMOLCAS with the ``block2`` interface.
The source code of the required OpenMOLCAS code can be found in https://github.com/hczhai/OpenMolcas, which is
a slightly modified version of https://github.com/quanp/OpenMolcas (the OpenMOLCAS interface for the ``block 1.5`` and ``StackBlock``).
To activate the ``blcok2`` interface, run ``cmake`` for this OpenMOLCAS with the option ``-DBLOCK2=ON``. The detailed procedure is as follows: ::

    git clone https://github.com/hczhai/OpenMolcas
    export MOLCASHOME=$PWD/OpenMolcas
    cd OpenMolcas
    mkdir build
    cd build
    CC=gcc CXX=g++ FC=gfortran MKLROOT=/usr/local cmake .. -DCMAKE_INSTALL_PREFIX=../install -DLINALG=MKL -DOPENMP=ON -DBLOCK2=ON
    make -j 10
    make install

Remember to change the ``MKLROOT`` variable in the above example for your case.

Then one can run OpenMolcas using the following command: ::

    MOLCAS=$MOLCASHOME/install MOLCAS_WORKDIR=/content/tmp pymolcas test.in

Where ``test.in`` is an OpenMolcas input file.

DMRGSCF
-------

The following is an example input file for DMRGSCF for a O2 triplet state (see :ref:`user_dmrgscf` for the similar calculation using ``pyscf``): ::

    &GATEWAY
        Title
        O2 Molecule
        Coord
        2

        O 0 0 -0.6035
        O 0 0 0.6035
        Basis set
        CC-PVDZ

    &SEWARD

    &SCF
        Spin = 1

    &RASSCF
        Spin
        3
        Symmetry
        4
        nActEl
        2 0 0
        Inactive
        3 1 1 0 2 0 0 0
        Ras2
        0 0 0 0 0 1 1 0
        CIROOT = 1 1 ; 1

    &RASSCF
        Spin
        3
        Symmetry
        4
        nActEl
        8 0 0
        Inactive
        2 0 0 0 2 0 0 0
        Ras2
        1 1 1 0 1 1 1 0
        CIROOT = 1 1 ; 1
        CISOlver = BLOCK
        DMRG = 1000

Note that the first ``RASSCF`` is actually a ROHF mean-field calculation.

.. highlight:: python

The same calculation in ``pyscf`` is: ::

    from pyscf import gto, scf, mcscf, mrpt, dmrgscf, lib, symm
    from pyblock2._pyscf.ao2mo import integrals as itg
    import os

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, symmetry='d2h', cart=False, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, g2e_symm=8)

    print(orb_sym)
    print(mf.mo_occ)
    orb_sym_name = [symm.irrep_id2name(mol.groupname, ir) for ir in orb_sym]
    print(orb_sym_name)

    mc = mcscf.CASSCF(mf, 6, 8)

    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.natorb = True
    mc.run()

From the ``pyscf`` output we can see the occupation number and orbtial irreps are : ::

    [0, 5, 0, 5, 0, 6, 7, 2, 3, 5, 5, 6, 7, 0, 2, 3, 0, 5, 6, 7, 0, 1, 4, 5, 0, 2, 3, 5] # XOR irreps
    [2. 2. 2. 2. 2. 2. 2. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # occ
    ['Ag', 'B1u', 'Ag', 'B1u', 'Ag', 'B2u', 'B3u', 'B2g', 'B3g', 'B1u', 'B1u', 'B2u', 'B3u', 'Ag', 'B2g', 'B3g', 'Ag', 'B1u', 'B2u', 'B3u', 'Ag', 'B1g', 'Au', 'B1u', 'Ag', 'B2g', 'B3g', 'B1u']

.. highlight:: text

The MOLCAS ordering of irreps of D2h is: ::

    ag b3u b2u b1g b1u b2g b3g au

This information can help us setting the ``Inactive`` and ``Ras2`` in the MOLCAS inputfile.

From the ``pyscf`` output we have: ::

    $ grep 'converged SCF energy' pyscf.out
    converged SCF energy = -149.608181589162
    $ grep 'CASSCF energy' pyscf.out
    CASSCF energy = -149.708657770064

From the ``openMOLCAS`` output we have: ::

    $ grep '::    RASSCF' o2.out
    ::    RASSCF root number  1 Total energy:   -149.60818159
    ::    RASSCF root number  1 Total energy:   -149.70865773

Which is consistent.

DMRG-CASPT2
-----------

The following is an example input file for CASPT2 calculation after DMRGSCF for a O2 triplet state.
The cumulant approximation of 4PDM is used for CASPT2. Note that the IPEA shift = 0.25 is used by default. ::

    &GATEWAY
        Title
        O2 Molecule
        Coord
        2

        O 0 0 -0.6035
        O 0 0 0.6035
        Basis set
        CC-PVDZ

    &SEWARD

    &SCF
        Spin = 1

    &RASSCF
        Spin
        3
        Symmetry
        4
        nActEl
        2 0 0
        Inactive
        3 1 1 0 2 0 0 0
        Ras2
        0 0 0 0 0 1 1 0
        CIROOT = 1 1 ; 1

    &RASSCF
        Spin
        3
        Symmetry
        4
        nActEl
        8 0 0
        Inactive
        2 0 0 0 2 0 0 0
        Ras2
        1 1 1 0 1 1 1 0
        CIROOT = 1 1 ; 1
        CISOlver = BLOCK
        DMRG = 1000
        3RDM

    &CASPT2
        MULT = 1 1
        CUMU

This will generate the following output: ::

    $ grep '::    CASPT2' o2.out 
    ::    CASPT2 Root  1     Total energy:   -149.97055932
