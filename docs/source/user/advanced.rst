
.. highlight:: bash

Advanced Usage
==============

Orbital Rotation
----------------

In this calculation we illustrate how to compute the grond state MPS in the given set of orbitals,
find the (new) DMRG natural orbitals, transform integrals to new orbitals,
transform the ground state MPS to new orbitals, and finally evaluate the energy of the transformed MPS in
the new orbitals to verify the quality of the transformed MPS.

First, we compute the energy and 1-particle density matrix for the ground state using the following input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    onepdm
    irrep_reorder

Note that we use the keyword ``irrep_reorder`` to reorder the orbitals so that orbitals belonging to the same
point group irrep are grouped together. This can make the orbital rotation more local.

The DMRG occupation number (in original ordering) will be printed at the end of the calculation: ::

    $ grep OCC dmrg-1.out
    DMRG OCC =   1.957 1.625 1.870 1.870 0.361 0.098 0.098 0.006 0.008 0.008 0.008 0.013 0.014 0.014 0.011 0.006 0.006 0.006 0.005 0.005 0.002 0.002 0.002 0.001 0.001 0.001
    $ grep Energy dmrg-1.out
    DMRG Energy =  -75.728467269121111

Second, we use the keyword ``nat_orbs`` to compute the natural orbitals. The value of the keyword ``nat_orbs``
specifies the filename for storing the rotated integrals (FCIDUMP).
If no value is associated with the keyword ``nat_orbs``, the rotated integrals will not be computed.
The keyword ``nat_orbs`` can only be used together with ``restart_onepdm`` or ``onepdm``, since natural orbitals
are found by diagonalizing 1-particle density matrix.

The following input file is used for this step (it can also be combined with the previous calculation): ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    restart_onepdm
    nat_orbs C2.NAT.FCIDUMP
    irrep_reorder

The occupation number in natural orbitals will be printed at the end of the calculation: ::

    $ grep OCC dmrg-2.out
    DMRG OCC =   1.957 1.625 1.870 1.870 0.361 0.098 0.098 0.006 0.008 0.008 0.008 0.013 0.014 0.014 0.011 0.006 0.006 0.006 0.005 0.005 0.002 0.002 0.002 0.001 0.001 0.001
    REORDERED OCC =   1.957 0.002 0.361 0.006 0.013 0.008 0.002 0.006 0.011 0.001 0.006 1.625 0.008 1.870 0.005 0.098 0.001 0.014 0.005 1.870 0.008 0.001 0.014 0.098 0.006 0.002
    NAT OCC =   0.000465 0.003017 0.006424 0.007848 0.360936 1.968407 0.000081 0.000916 0.001991 0.004082 0.015623 1.628182 0.003669 0.008706 1.870680 0.000424 0.002862 0.110463 0.003667 0.008705 1.870678 0.000424 0.002862 0.110480 0.006422 0.001989

The rotation matrix for natural orbitals, the logarithm of the rotation matrix, and the occupation number in natural orbitals
are stored as ``nat_rotation.npy``, ``nat_kappa.npy``, ``nat_occs.npy`` in scartch folder, respectively. In this example,
the rotated integral is stored as ``C2.NAT.FCIDUMP`` in the working directory.

Third, we load the MPS in the old orbitals and transform it into the new orbitals. This is done using time evolution.
The keyword ``delta_t`` is used to set a time step and indicate that this is a time evolution calculation.
The keyword ``orbital_rotation`` is used to indicate that the operator (exponentiated) applied into the MPS should
be the orbital rotation operator (constructed from ``nat_kappa.npy`` saved in the previous step).

Typically, a large bond dimension should be used depending how non-local the orbital rotation operator is.
The ``target_t`` for orbital rotation is automatically set to 1.

The following input file is used for this step: ::

    sym d2h

    nelec 8
    spin 0
    irrep 1

    schedule
        0 1000 0 0
    end

    orbital_rotation
    delta_t 0.02
    outputlevel 1
    noreorder

Note that ``noreorder`` must be used for orbital rotation. The orbital reordering
in previous step has already been taken into account.

The output looks like the following: ::

    $ grep DW dmrg-3.out 
    Time elapsed =      1.183 | E =      -0.0000000000 | Norm^2 =       0.9999999933 | DW = 5.19e-09
    Time elapsed =      2.727 | E =      -0.0000000000 | Norm^2 =       0.9999999878 | DW = 4.37e-09
    Time elapsed =      1.546 | E =       0.0000000000 | Norm^2 =       0.9999999678 | DW = 1.05e-08
    Time elapsed =      3.108 | E =      -0.0000000000 | Norm^2 =       0.9999999579 | DW = 6.31e-09
    ... ...
    Time elapsed =      1.665 | E =      -0.0000000000 | Norm^2 =       0.9999906353 | DW = 1.04e-07
    Time elapsed =      3.321 | E =      -0.0000000000 | Norm^2 =       0.9999904773 | DW = 5.86e-08
    Time elapsed =      1.646 | E =      -0.0000000000 | Norm^2 =       0.9999902248 | DW = 1.08e-07
    Time elapsed =      3.289 | E =      -0.0000000000 | Norm^2 =       0.9999900580 | DW = 6.19e-08

Since in every time step an orthogonal transformation is applied on the MPS,
the expectation value of the orthogonal transformation
(printed as the energy expectation) calculated on the MPS should always be zero.

Note that largest discarded weight is ``6.19e-08``, and the norm of MPS is not far away from 1.
So the transormation should be relatively accurate.

Finally, we calculate the energy expectation value using the transformed integral (``C2.NAT.FCIDUMP``)
and the transformed MPS (stored in the scratch folder), using the following input file: ::

    sym d2h
    orbitals C2.NAT.FCIDUMP

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    restart_oh
    noreorder

Note that ``noreorder`` must be used, since the MPS generated in the previous step is in
unreordered natural orbitals.
The keyword ``restart_oh`` will calculate the expectation value of the given Hamiltonian
loaded from integrals on the MPS loaded from scartch folder.

We have the following output: ::

    $ grep Energy dmrg-4.out
    OH Energy =  -75.726795187335256

One can increase the bond dimension in the evolution to make this closer to the value printed
in the first step.
