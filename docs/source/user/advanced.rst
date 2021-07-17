
.. highlight:: bash

.. _user_advanced:

Advanced Usage
==============

Orbital Rotation
----------------

In this calculation we illustrate how to compute the ground state MPS in the given set of orbitals,
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
    DMRG Energy =  -75.728467269121097

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
    nat_km_reorder
    nat_positive_def
    irrep_reorder

Where the optional keyword ``nat_km_reorder`` can be used to remove the artificial reordering in the natural orbitals
using Kuhn-Munkres algorithm. The optional keyword ``nat_positive_def`` can be used to avoid artificial rotation in the
logarithm of the rotation matrix, by make the rotation matrix quasi-positive-definite, with "quasi" in the sense that
the rotation matrix is not Hermitian. The two options may be good for weakly correlated systems, but have limited effects
for highly correlated systems (but for highly correlated systems it is also recommended to be used).

The occupation number in natural orbitals will be printed at the end of the calculation: ::

    $ grep OCC dmrg-2.out
    DMRG OCC =   1.957 1.625 1.870 1.870 0.361 0.098 0.098 0.006 0.008 0.008 0.008 0.013 0.014 0.014 0.011 0.006 0.006 0.006 0.005 0.005 0.002 0.002 0.002 0.001 0.001 0.001
    REORDERED OCC =   1.957 0.002 0.361 0.006 0.013 0.008 0.002 0.006 0.011 0.001 0.006 1.625 0.008 1.870 0.005 0.098 0.001 0.014 0.005 1.870 0.008 0.001 0.014 0.098 0.006 0.002
    NAT OCC =   0.000465 0.003017 0.006424 0.007848 0.360936 1.968407 0.000081 0.000916 0.001991 0.004082 0.015623 1.628182 0.003669 0.008706 1.870680 0.000424 0.002862 0.110463 0.003667 0.008705 1.870678 0.000424 0.002862 0.110480 0.006422 0.001989

With the optional keyword ``nat_km_reorder`` there will be an extra line: ::

    REORDERED NAT OCC =   1.968407 0.000465 0.360936 0.006424 0.007848 0.003017 0.001991 0.000081 0.004082 0.000916 0.015623 1.628182 0.008706 1.870680 0.003669 0.110463 0.000424 0.002862 0.003667 1.870678 0.008705 0.000424 0.002862 0.110480 0.006422 0.001989

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
    delta_t 0.05
    outputlevel 1
    noreorder

Note that ``noreorder`` must be used for orbital rotation. The orbital reordering
in previous step has already been taken into account.

The keyword ``te_type`` can be used to set the time-evolution algorithm. The default is ``rk4``,
which is the original time-step-targeting (TST) method. Another possible choice is ``tdvp``,
which is the time dependent variational principle with the projector-splitting (TDVP-PS) algorithm.

The output looks like the following: ::

    $ grep DW dmrg-3.out 
    Time elapsed =      2.263 | E =       0.0000000000 | Norm^2 =       0.9999999999 | DW = 1.76e-10
    Time elapsed =      4.910 | E =      -0.0000000000 | Norm^2 =       0.9999999997 | DW = 1.43e-10
    Time elapsed =      1.663 | E =      -0.0000000000 | Norm^2 =       0.9999999988 | DW = 4.46e-10
    Time elapsed =      3.475 | E =       0.0000000000 | Norm^2 =       0.9999999983 | DW = 2.50e-10
    ... ...
    Time elapsed =      3.011 | E =       0.0000000000 | Norm^2 =       0.9999999315 | DW = 1.04e-09
    Time elapsed =      4.753 | E =       0.0000000000 | Norm^2 =       0.9999999284 | DW = 8.68e-10
    Time elapsed =      1.786 | E =       0.0000000000 | Norm^2 =       0.9999999245 | DW = 1.07e-09
    Time elapsed =      3.835 | E =       0.0000000000 | Norm^2 =       0.9999999213 | DW = 9.09e-10

Since in every time step an orthogonal transformation is applied on the MPS,
the expectation value of the orthogonal transformation
(printed as the energy expectation) calculated on the MPS should always be zero.

Note that largest discarded weight is ``1.07e-09``, and the norm of MPS is not far away from 1.
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
    restart_onepdm
    noreorder

Note that ``noreorder`` must be used, since the MPS generated in the previous step is in
unreordered natural orbitals.
The keyword ``restart_oh`` will calculate the expectation value of the given Hamiltonian
loaded from integrals on the MPS loaded from scartch folder.

We have the following output: ::

    $ grep Energy dmrg-4.out
    OH Energy =  -75.728457535820155

The difference compared to the energy generated in the first step
``DMRG Energy =  -75.728467269121097`` is only 9.7E-6.
One can increase the bond dimension in the evolution to make this closer to the value printed
in the first step.

MPS Transform
-------------

The MPS can be copied and saved using another tag.
For SU2 (spin-adapted) MPS, it can also be transformed to SZ (non-spin-adapted) MPS and saved using another tag.

Limitations:

* Total spin zero spin-adapted MPS can be transformed directly.
* For non-zero total spin, the spin-adapted MPS must be in singlet embedding format. See next section.

First, we compute the energy for the spin-adapted ground state using the following input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags KET

The following script will read the spin-adapted MPS and tranform it to a non-spin-adapted MPS: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags KET
    restart_copy_mps ZKET
    trans_mps_to_sz

Here the keyword ``restart_copy_mps`` indicates that the MPS will be copied, associated with a value
indicating the new tag for saving the copied MPS.
If the keyword ``trans_mps_to_sz`` is present, the MPS will be transformed to non-spin-adapted before
being saved.

Finally, we calculate the energy expectation value using non-spin-adapted formalism
and the transformed MPS (stored in the scratch folder), using the following input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags ZKET
    restart_oh
    nonspinadapted

Some reference outputs for this example: ::

    $ grep Energy dmrg-1.out
    DMRG Energy =  -75.728467269121083
    $ grep MPS dmrg-2.out
    MPS =  KRRRRRRRRRRRRRRRRRRRRRRRRR 0 2
    GS INIT MPS BOND DIMS =       1     3    10    35   120   263   326   500   500   500   500   500   500   500   500   500   500   500   498   500   407   219    94    32    10     3     1
    $ grep 'MPS\|Energy' dmrg-3.out 
    MPS =  KRRRRRRRRRRRRRRRRRRRRRRRRR 0 2
    GS INIT MPS BOND DIMS =       1     4    16    64   246   578   712  1114  1097  1102  1110  1121  1126  1130  1116  1111  1111  1107  1074  1103   895   444   186    59    16     4     1
    OH Energy =  -75.728467269120898

We can see that the transformation from SU2 to SZ is nearly exact, and the required bond dimension for the SZ MPS
is roughly two times of the SU2 bond dimension.

Singlet Embedding
-----------------

For spin-adapted calculation with total spin not equal to zero, there can be some convergence problem
even if in one-site algorithm. One way to solve this problem is to use singlet embedding.
In ``StackBlock`` singlet embedding is used by default.
In ``block2``, by default singlet embedding is not used. If one adds the keyword ``singlet_embedding`` to the input file,
the singlet embedding scheme will be used. For most total spin not equal to zero calculation,
singlet embedding may be more stable. One cannot calculate transition density matrix between states with different total spins
using singlet embedding. To do that one can translate the MPS between singlet embedding format and non-singlet-embedding format.

When total spin is equal to zero, the keyword ``singlet_embedding`` will not have any effect.
If restarting a calculation, normally, the keyword ``singlet_embedding`` is not required since the format of the MPS
can be automatically recognized.

For translating SU2 MPS to SZ MPS with total spin not equal to zero, the SU2 MPS must be in singlet embedding format.

First, we compute the energy for the spin-adapted with non-zero total spin using the following input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 2
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags KET

The above input file indicates that singlet embedding is not used. The output is: ::

    $ grep 'MPS = ' dmrg-1.out
    MPS =  CCRRRRRRRRRRRRRRRRRRRRRRRR 0 2 < N=8 S=1 PG=0 >
    $ grep Energy dmrg-1.out
    DMRG Energy =  -75.423916647509742

Here the printed target quantum number of the MPS indicates that it is a triplet.

We can add the keyword ``singlet_embedding`` to do a singlet embedding calculation: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 2
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags SEKET
    singlet_embedding

When singlet embedding is used, the output is: ::

    $ grep 'MPS = ' dmrg-2.out
    MPS =  CCRRRRRRRRRRRRRRRRRRRRRRRR 0 2 < N=10 S=0 PG=0 >
    $ grep Energy dmrg-2.out
    DMRG Energy =  -75.423879916245895

Here the printed target quantum number of the MPS indicates that it is a singlet (including some ghost particles).

One can use the keywords ``trans_mps_to_singlet_embedding`` and ``trans_mps_from_singlet_embedding``
combined with ``restart_copy_mps`` or ``copy_mps`` to translate between singlet embedding and normal formats.

The following script transforms the MPS from singlet embedding to normal format: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 2
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags SEKET
    restart_copy_mps TKET
    trans_mps_from_singlet_embedding

We can verify that the transformed non-singlet-embedding MPS has the same energy as the singlet embedding MPS: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 2
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags TKET
    restart_oh

With the outputs: ::

    $ grep 'MPS = ' dmrg-4.out
    MPS =  KRRRRRRRRRRRRRRRRRRRRRRRRR 0 2 < N=8 S=1 PG=0 >
    $ grep Energy dmrg-4.out
    OH Energy =  -75.423879916245824

The following script will read the spin-adapted singlet embedding MPS and tranform it to a non-spin-adapted MPS: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 2
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags SEKET
    restart_copy_mps ZKETM2
    trans_mps_to_sz
    resolve_twosz -2
    normalize_mps

Here the keyword ``resolve_twosz`` indicates that the transformed SZ MPS will have projected spin ``2 * SZ = 2``.
For this case since ``2 * S = 2``, the possible values for ``resolve_twosz`` are ``-2, 0, 2``.
If the keyword ``resolve_twosz`` is not given, an MPS with ensemble of all possible projected spins will be produced
(which is often not very useful).
Getting one component of the SU2 MPS means that the SZ MPS will not have the same norm as the SU2 MPS.
If the keyword ``normalize_mps`` is added, the transformed SZ MPS will be normalized. The keyword ``normalize_mps``
can only have effect when ``trans_mps_to_sz`` is present.

Finally, we calculate the energy expectation value using non-spin-adapted formalism
and the transformed MPS (stored in the scratch folder), using the following input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin -2
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags ZKETM2
    restart_oh
    nonspinadapted

Some reference outputs for this example: ::

    $ grep MPS dmrg-6.out
    MPS =  KRRRRRRRRRRRRRRRRRRRRRRRRR 0 2 < N=8 SZ=-1 PG=0 >
    GS INIT MPS BOND DIMS =       1    12    48   192   601  1145  1398  1474  1476  1468  1466  1441  1356  1316  1255  1240  1217  1206  1198  1176   904   422   183    59    16     4     1
    $ grep Energy dmrg-6.out
    OH Energy =  -75.423879916245909

We can see that the transformation from SU2 to SZ is nearly exact. The other two components of the SU2 MPS
will also have the same energy as this one.

CSF or Determinant Sampling
---------------------------

The overlap between the spin-adapted MPS and Configuration State Functions (CSFs),
or between the non-spin-adapted MPS and determinants can be calculated.
Since there are exponentially many CSFs or determinants (when the number of electrons
is close to the number of orbitals), normally it only makes sense to sample
CSFs or determinants with (absolute value of) the overlap larger than a threshold.
The sampling is deterministic, meaning that all overlap above the given threshold will be printed.

The keyword ``sample`` or ``restart_sample`` can be used to sample CSFs or determinants
after DMRG or from an MPS loaded from disk. The value associated with the keyword
``sample`` or ``restart_sample`` is the threshold for sampling.

Setting the threshold to zero is allowed, but this may only be useful for some very small systems.

Limitations: For non-zero total spin CSF sampling,
the spin-adapted MPS must be in singlet embedding format. See the previous section.

The following is an example of the input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    irrep_reorder
    mps_tags KET
    sample 0.05

Some reference outputs for this example: ::

    $ grep CSF dmrg-1.out
    Number of CSF =         17 (cutoff =      0.05)
    Sum of weights of sampled CSF =    0.909360149891891
    CSF          0 20000000000202000002000000  =    0.828657540546610
    CSF          1 20200000000002000002000000  =   -0.330323898091116
    CSF          2 20+00000000+0200000-000-00  =   -0.140063445607095
    CSF          3 20+00000000+0-0-0002000000  =   -0.140041987646036
    ... ...
    CSF         16 200000000002000+0-02000000  =    0.050020205617060

When there are more than 50 determinants, only the first 50 with largest weights
will be printed. The complete list of determinants and coefficients are stored in
``sample-dets.npy`` and ``sample-vals.npy`` in the scratch folder, respectively.

So the restricted Hartree-Fock determinant/CSF has a very large coefficient (0.83).

To verify this, we can also directly compress the ground-state MPS to bond dimension 1,
to get the CSF with the largest coefficient. Note that the compression method may
converge to some other CSFs if there are many determinants with similar coefficients.

MPS Compression
---------------

MPS compression can be used to compress or fit a given MPS to a different
(larger or smaller) bond dimension.

The following is an example of the input file for the compression
(which will load the MPS obtailed from the previous ground-state DMRG): ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule
    0  250  0 0
    2  125  0 0
    4   62  0 0
    6   31  0 0
    8   15  0 0
    10   7  0 0
    12   3  0 0
    14   1  0 0
    end
    maxiter 16

    compression
    overlap
    read_mps_tags KET
    mps_tags BRA

    irrep_reorder

Here the keyword ``compression`` indicates that this is a compression calculation.
When the keyword ``overlap`` is given, the loaded MPS will be compressed,
otherwise, the result of H|MPS> will be compressed.
The tag of the input MPS is given by ``read_mps_tags``,
and the tag of the output MPS is given by ``mps_tags``.

Some reference outputs for this example: ::

    $ grep 'Compression overlap' dmrg-2.out
    Compression overlap =    0.828657540546619

We can see that the value obtained from compression is very close to the sampled value.
But when a lower bound of the overlap is known, the sampling method should be
more reliable and efficient for obtaining the CSF with the largest weight.

If the CSF or determinat pattern is required, one can do a quick sampling on the compressed
MPS using the keyword ``restart_sample 0``.

If the given MPS has a very small bond dimension, or the target (output) MPS has a very large bond dimension
(namely, "decompression"), one should use the keyword ``random_mps_init`` to allow a better random
initial guess for the target MPS. Otherwise, the generated output MPS may be inaccurate.
