
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

    mps_tags BRA
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

    mps_tags BRA
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

Here the keyword ``resolve_twosz`` indicates that the transformed SZ MPS will have projected spin ``2 * SZ = -2``.
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

LZ Symmetry
-----------

For diatomic molecules or model Hamiltonian with translational symmetry (such as 1D Hubbard model in momentum space),
it is possible to utilize additional K space symmetry.
To support the K space symmetry, the code must be compiled with the option ``-DUSE_KSYMM=ON`` (default).

One can add the keyword ``k_symmetry`` in the input file to use this additional symmetry.
Point group symmetry can be used together with k symmetry.
Therefore, even for system without K space symmetry, the calculation can still run as normal when the keyword ``k_symmetry`` is added.
Note, however, the MPS or MPO generated from an input file with/without the keyword ``k_symmetry``,
cannot be reloaded with an input file without/with the keyword ``k_symmetry``.

.. highlight:: python3

For molecules, the integral file (FCIDUMP file) must be generated in a special way so that the K/LZ symmetry can be used.
the following python script can be used to generate the integral with :math:`C_2 \otimes L_z` symmetry: ::

    import numpy as np
    from functools import reduce
    from pyscf import gto, scf, ao2mo, symm, tools, lib
    from block2 import FCIDUMP, VectorUInt8, VectorInt

    # adapted from https://github.com/hczhai/pyscf/blob/1.6/examples/symm/33-lz_adaption.py
    # with the sign of lz
    def lz_symm_adaptation(mol):
        z_irrep_map = {} # map from dooh to lz
        g_irrep_map = {} # map from dooh to c2
        symm_orb_map = {} # orbital rotation
        for ix in mol.irrep_id:
            rx, qx = ix % 10, ix // 10
            g_irrep_map[ix] = rx & 4
            z_irrep_map[ix] = (-1) ** ((rx & 1) == ((rx & 4) >> 2)) * ((qx << 1) + ((rx & 2) >> 1))
            if z_irrep_map[ix] == 0:
                symm_orb_map[(ix, ix)] = 1
            else:
                if (rx & 1) == ((rx & 4) >> 2):
                    symm_orb_map[(ix, ix)] = -np.sqrt(0.5) * ((rx & 2) - 1)
                else:
                    symm_orb_map[(ix, ix)] = -np.sqrt(0.5) * 1j
                symm_orb_map[(ix, ix ^ 1)] = symm_orb_map[(ix, ix)] * 1j

        z_irrep_map = [z_irrep_map[ix] for ix in mol.irrep_id]
        g_irrep_map = [g_irrep_map[ix] for ix in mol.irrep_id]
        rev_symm_orb = [np.zeros_like(x) for x in mol.symm_orb]
        for iix, ix in enumerate(mol.irrep_id):
            for iiy, iy in enumerate(mol.irrep_id):
                if (ix, iy) in symm_orb_map:
                    rev_symm_orb[iix] = rev_symm_orb[iix] + symm_orb_map[(ix, iy)] * mol.symm_orb[iiy]
        return rev_symm_orb, z_irrep_map, g_irrep_map

    # copied from https://github.com/hczhai/pyscf/blob/1.6/pyscf/symm/addons.py#L29
    # with the support for complex orbitals
    def label_orb_symm(mol, irrep_name, symm_orb, mo, s=None, check=True, tol=1e-9):
        nmo = mo.shape[1]
        if s is None:
            s = mol.intor_symmetric('int1e_ovlp')
        s_mo = np.dot(s, mo)
        norm = np.zeros((len(irrep_name), nmo))
        for i, csym in enumerate(symm_orb):
            moso = np.dot(csym.conj().T, s_mo)
            ovlpso = reduce(np.dot, (csym.conj().T, s, csym))
            try:
                s_moso = lib.cho_solve(ovlpso, moso)
            except:
                ovlpso[np.diag_indices(csym.shape[1])] += 1e-12
                s_moso = lib.cho_solve(ovlpso, moso)
            norm[i] = np.einsum('ki,ki->i', moso.conj(), s_moso).real
        norm /= np.sum(norm, axis=0)  # for orbitals which are not normalized
        iridx = np.argmax(norm, axis=0)
        orbsym = np.asarray([irrep_name[i] for i in iridx])

        if check:
            largest_norm = norm[iridx,np.arange(nmo)]
            orbidx = np.where(largest_norm < 1-tol)[0]
            if orbidx.size > 0:
                idx = np.where(largest_norm < 1-tol*1e2)[0]
                if idx.size > 0:
                    raise ValueError('orbitals %s not symmetrized, norm = %s' %
                                    (idx, largest_norm[idx]))
                else:
                    raise ValueError('orbitals %s not strictly symmetrized.',
                                np.unique(orbidx))
        return orbsym

    mol = gto.M(
        atom=[["C", (0, 0, 0)],
              ["C", (0, 0, 1.2425)]],
        basis='ccpvdz',
        symmetry='dooh')

    mol.symm_orb, z_irrep, g_irrep = lz_symm_adaptation(mol)
    mf = scf.RHF(mol)
    mf.run()

    h1e = mf.mo_coeff.conj().T @ mf.get_hcore() @ mf.mo_coeff
    print('h1e imag = ', np.linalg.norm(h1e.imag))
    assert np.linalg.norm(h1e.imag) < 1E-14
    e_core = mol.energy_nuc()
    h1e = h1e.real.flatten()
    _eri = ao2mo.restore(1, mf._eri, mol.nao)
    g2e = np.einsum('pqrs,pi,qj,rk,sl->ijkl', _eri,
        mf.mo_coeff.conj(), mf.mo_coeff, mf.mo_coeff.conj(), mf.mo_coeff, optimize=True)
    print('g2e imag = ', np.linalg.norm(g2e.imag))
    assert np.linalg.norm(g2e.imag) < 1E-14
    print('g2e symm = ', np.linalg.norm(g2e - g2e.transpose((1, 0, 3, 2))))
    print('g2e symm = ', np.linalg.norm(g2e - g2e.transpose((2, 3, 0, 1))))
    print('g2e symm = ', np.linalg.norm(g2e - g2e.transpose((3, 2, 1, 0))))
    g2e = g2e.real.flatten()

    fcidump_tol = 1E-13
    na = nb = mol.nelectron // 2
    n_mo = mol.nao
    h1e[np.abs(h1e) < fcidump_tol] = 0
    g2e[np.abs(g2e) < fcidump_tol] = 0

    orb_sym_z = label_orb_symm(mol, z_irrep, mol.symm_orb, mf.mo_coeff, check=True)
    orb_sym_g = label_orb_symm(mol, g_irrep, mol.symm_orb, mf.mo_coeff, check=True)
    print(orb_sym_z)

    fcidump = FCIDUMP()
    fcidump.initialize_su2(n_mo, na + nb, na - nb, 1, e_core, h1e, g2e)

    orb_sym_mp = VectorUInt8([tools.fcidump.ORBSYM_MAP['D2h'][i] for i in orb_sym_g])
    fcidump.orb_sym = VectorUInt8(orb_sym_mp)
    print('g symm error = ', fcidump.symmetrize(VectorUInt8(orb_sym_g)))

    fcidump.k_sym = VectorInt(orb_sym_z)
    fcidump.k_mod = 0
    print('z symm error = ', fcidump.symmetrize(fcidump.k_sym, fcidump.k_mod))

    fcidump.write('FCIDUMP')

.. highlight:: text

Note that, if only the LZ symmetry is required, one can simply set ``orb_sym_g[:] = 0``.

The following input file can be used to perform the calculation with :math:`C_2 \otimes L_z` symmetry: ::

    sym d2h
    orbitals FCIDUMP
    k_symmetry
    k_irrep 0

    nelec 12
    spin 0
    irrep 1

    hf_occ integral
    schedule
    0  500 1E-8 1E-3
    4  500 1E-8 1E-4
    8  500 1E-9 1E-5
    12 500 1E-9 0
    end
    maxiter 30

Where the ``k_irrep`` can be used to set the eigenvalue of LZ in the target state.
Note that it can be easier for the Davidson procedure to get stuck in local minima with high symmetry.
It is therefore recommended to use a custom schedule with larger noise and smaller Davidson threshold.

Some reference outputs for this input file: ::

    $ grep 'Time elapsed' dmrg-1.out | tail -1
    Time elapsed =     73.529 | E =     -75.7291544157 | DE = -6.31e-07 | DW = 1.28e-05
    $ grep 'DMRG Energy' dmrg-1.out
    DMRG Energy =  -75.729154415733063

When there are too many orbitals, and the default ``warmup fci`` initial guess is used,
the initial MPS can have very large bond dimension
(especially when the LZ symmetry is used, since LZ is not a finite group)
and the first sweep will take very long time.

One way to solve this is to limit the LZ to a finite group, using modular arithmetic.
We can limit LZ to Z4 or Z2. The efficiency gain will be smaller, but the convergence may be more stable.
The keyword ``k_mod`` can be used to set the modulus. When ``k_mod = 0``, it is the original infinite LZ group.

The following input file can be used to perform the calculation with :math:`C_2 \otimes Z_4` symmetry: ::

    sym d2h
    orbitals FCIDUMP
    k_symmetry
    k_irrep 0
    k_mod 4

    nelec 12
    spin 0
    irrep 1

    hf_occ integral
    schedule
    0  500 1E-8 1E-3
    4  500 1E-8 1E-4
    8  500 1E-9 1E-5
    12 500 1E-9 0
    end
    maxiter 30

Some reference outputs for this input file: ::

    $ grep 'Time elapsed' dmrg-2.out | tail -1
    Time elapsed =    111.491 | E =     -75.7292222457 | DE = -8.17e-08 | DW = 1.28e-05
    $ grep 'DMRG Energy' dmrg-2.out
    DMRG Energy =  -75.729222245693876

Similarly, setting ``k_mod 2`` gives the following output: ::

    $ grep 'Time elapsed' dmrg-3.out | tail -1
    Time elapsed =    135.394 | E =     -75.7314583188 | DE = -3.97e-07 | DW = 1.49e-05
    $ grep 'DMRG Energy' dmrg-3.out
    DMRG Energy =  -75.731458318751280

Initial Guess with Occupation Numbers
-------------------------------------

Once can use ``warmup occ`` initial guess to solve the initial guess problem, where another keywrod ``occ`` should be used,
followed by a list of (fractional) occupation numbers separated by the space character, to set the occupation numbers.
The occupation numbers can be obtained from a DMRG calculation using the same integral with/without K symmetry (or some other methods like CCSD and MP2).
If ``onepdm`` is in the input file, the occupation numbers will be printed at the end of the output.

The following input file will perform the DMRG calculation using the same integral without the K symmetry (but with C2 symmetry): ::

    sym d2h
    orbitals FCIDUMP

    nelec 12
    spin 0
    irrep 1

    hf_occ integral
    schedule
    0  500 1E-8 1E-3
    4  500 1E-8 1E-4
    8  500 1E-9 1E-5
    12 500 1E-9 0
    end
    maxiter 30
    onepdm

Some reference outputs for this input file: ::

    $ grep 'Time elapsed' dmrg-1.out | tail -2 | head -1
    Time elapsed =    190.549 | E =     -75.7314655815 | DE = -1.88e-07 | DW = 1.53e-05
    $ grep 'DMRG Energy' dmrg-1.out
    DMRG Energy =  -75.731465581478815
    $ grep 'DMRG OCC' dmrg-1.out
    DMRG OCC =   2.000 2.000 1.957 1.626 1.870 1.870 0.360 0.098 0.098 0.006 0.008 0.008 0.008 0.013 0.014 0.014 0.011 0.006 0.006 0.006 0.005 0.005 0.002 0.002 0.002 0.001 0.001 0.001

The following input file will perform the DMRG calculation using the K symmetry, but with initial guess generated from occupation numbers: ::

    sym d2h
    orbitals FCIDUMP
    k_symmetry
    k_irrep 0
    warmup occ
    occ 2.000 2.000 1.957 1.626 1.870 1.870 0.360 0.098 0.098 0.006 0.008 0.008 0.008 0.013 0.014 0.014 0.011 0.006 0.006 0.006 0.005 0.005 0.002 0.002 0.002 0.001 0.001 0.001
    cbias 0.2

    nelec 12
    spin 0
    irrep 1

    hf_occ integral
    schedule
    0  500 1E-8 1E-3
    4  500 1E-8 1E-4
    8  500 1E-9 1E-5
    12 500 1E-9 0
    end
    maxiter 30

Here ``cbias`` is the keyword to add a constant bias to the occ, so that 2.0 becomes 2.0 - cbias, and 0.098 becomes 0.098 + cbias.
Without the bias it is also easy to converge to a local minima.

Some reference outputs for this input file: ::

    $ grep 'Time elapsed' dmrg-3.out | tail -1
    Time elapsed =     55.938 | E =     -75.7244716369 | DE = -5.25e-07 | DW = 7.45e-06
    $ grep 'DMRG Energy' dmrg-3.out
    DMRG Energy =  -75.724471636942383

Here the calculation runs faster because the better initial guess, but the energy becomes worse.

Time Evolution
--------------

Now we give an example on how to do time evolution.
The computation will apply :math:`|MPS_{out}\rangle = \exp (-t H) |MPS_{in}\rangle` (with multiple steps).
When :math:`t` is a real floating point value, we will do imaginary time evolution of the MPS (namely, optimizing to ground state or finite-temperature state).
When :math:`t` is a pure imaginary value, we will do real time evolution of the MPS (namely, solving the time dependent Schrodinger equation).

To get accurate results, the time step has to be sufficiently small. The keyword ``delta_t`` is used to set a time step :math:`\Delta t` and indicate that this is a time evolution calculation. The keyword ``target_t`` is used to set a target "stopping" time, namely, the :math:`t`. The "starting" time is considered as zero. Therefore, the number of time steps is computed as :math:`nsteps = t / \Delta t` and printed.

If ``delta_t`` is too big, the time step error will be large. If ``delta_t`` is small, for fixed target time we have to do more time steps, with MPS bond dimension truncation happening after each sweep. So if ``delta_t`` is too small, the accumulated bond dimension truncation error will be large. Some meaningful time steps may be 0.01 to 0.1.

Real Time Evolution
^^^^^^^^^^^^^^^^^^^

First, we do a state-averaged calculation for the lowest two states using the following input file: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG
    nroots 2

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    noreorder

Note that the orbital reordering is disabled. The output: ::

    $ grep elapsed dmrg-1.out | tail -1
    Time elapsed =      5.762 | E[  2] =     -75.7268133875    -75.6376794953 | DE = -8.89e-08 | DW = 6.38e-05
    $ grep Final dmrg-1.out
    Final canonical form =  LLLLLLLLLLLLLLLLLLLLLLLLLJ 25

The energy of the MPS at the last site is actually -75.72629673 and -75.63717415, which are slightly different from the above values.

Second, we can use the following input file to load the
state-averaged MPS and then split it into individual MPSs: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG
    nroots 2

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30

    restart_copy_mps
    split_states
    trans_mps_to_complex
    noreorder

Note that here ``nroots`` must be the same as the previous case (or smaller, but larger than one),
otherwise the state-averaged MPS cannot be correctly loaded. The state-averaged MPS has the default tag KET.
We use calculation type keyword ``restart_copy_mps`` to do this transformation.
The new keyword ``split_states`` indicates that we want to split the MPS, this keyword should only be used
together with ``restart_copy_mps``.
The extra keyword ``trans_mps_to_complex`` will further make the MPS a complex MPS. This is required for
real time evolution, where ``delta_t`` can be imaginary.

For imaginary time evolution and real ``delta_t`` and real ``target_t``, everything will be real during the time evolution, so normally we do not need this extra keyword ``trans_mps_to_complex`` (but if you add it it is also okay).

The output looks like : ::
    
    $ tail -7 dmrg-2.out 
    ----- root =   0 /   2 -----
        final tag = KET-CPX-0
        final canonical form = LLLLLLLLLLLLLLLLLLLLLLLLLT
    ----- root =   1 /   2 -----
        final tag = KET-CPX-1
        final canonical form = LLLLLLLLLLLLLLLLLLLLLLLLLT
    MPI FINALIZE: rank 0 of 1

By default, the tranformed MPS will have tags ``KET-0``, ``KET-1`` etc, if it is real, or
``KET-CPX-0``, ``KET-CPX-1`` etc if it is complex.
If you set a custom tag, for example, when the input is like ``restart_copy_mps SKET``, the
tranformed MPS will have tags ``SKET-0``, ``SKET-1``, etc, no matter it is real or complex.

Third, we use the following script to do real time evolution: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    hf_occ integral
    schedule
    0 500 0 0
    end
    maxiter 10

    read_mps_tags KET-CPX-0
    mps_tags BRA
    delta_t 0.05i
    target_t 0.20i
    complex_mps
    noreorder

Note that a custom sweep schedule has to be used, to set the bond dimension to ``500`` (for example).
The keyword ``maxiter`` and ``noise`` in the sweep schedule are ignored.

For every time step, there can be multiple sweeps, called "sub sweeps". The total number of sweeps is ``n_sweeps = nsteps * n_sub_sweeps``. The keyword ``n_sub_sweeps`` can be used to set the number of sub sweeps. Default value is 2.

For real time evolution, ``delta_t`` and ``target_t`` should be pure imaginary values.
But they can also be general complex values.
When doing imaginary time evolution, ``delta_t`` and ``target_t`` should be all real.

The tag of the input MPS (old MPS) is given by ``read_mps_tags``.
The tag of the output MPS (new MPS) is given by ``mps_tags``. The two tags cannot be the same.
They should (better) not have common prefix. For example, ``KET`` and ``KET-1`` may not be used together, as ``-1`` may be used by the code internally which will lead to confusion.

For this example, ``target_t`` is four times ``delta_t``, so we will have 4 steps. Each time step has 2 sweeps. In total there will be 8 sweeps. The output is the result of applying ``\exp(-0.2i H)`` to the input.

Whenever a complex MPS is used, the keyword ``complex_mps`` should be used, otherwise the code will load the MPS incorrectly.

The output : ::

    $ grep 'final' dmrg-3.out
        mps final tag = BRA
        mps final canonical form = MRRRRRRRRRRRRRRRRRRRRRRRRR
    $ grep '<E>' dmrg-3.out
    T = RE    0.00000 + IM    0.05000 <E> =  -75.726309692728165 <Norm^2> =    0.999999608946318
    T = RE    0.00000 + IM    0.10000 <E> =  -75.726336818185246 <Norm^2> =    0.999994467614067
    T = RE    0.00000 + IM    0.15000 <E> =  -75.726364807114123 <Norm^2> =    0.999990200387707
    T = RE    0.00000 + IM    0.20000 <E> =  -75.726389514836484 <Norm^2> =    0.999986418355937

Here we see that the expectation value is printed after each time step.
The energy is roughly conserved (similar to the DMRG output -75.72629673), and the norm is roughly one.
Decreasing the time step may give more accurate results.

We can do the same for the excited state: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    hf_occ integral
    schedule
    0 500 0 0
    end
    maxiter 10

    read_mps_tags KET-CPX-1
    mps_tags BRAEX
    delta_t 0.05i
    target_t 0.20i
    complex_mps
    noreorder

The output : ::

    $ grep 'final' dmrg-4.out
        mps final tag = BRAEX
        mps final canonical form = MRRRRRRRRRRRRRRRRRRRRRRRRR
    $ grep '<E>' dmrg-4.out
    T = RE    0.00000 + IM    0.05000 <E> =  -75.637185795841717 <Norm^2> =    0.999999661398567
    T = RE    0.00000 + IM    0.10000 <E> =  -75.637212093724074 <Norm^2> =    0.999995415040728
    T = RE    0.00000 + IM    0.15000 <E> =  -75.637238086798163 <Norm^2> =    0.999991630799571
    T = RE    0.00000 + IM    0.20000 <E> =  -75.637260508028248 <Norm^2> =    0.999988252849994

The energy is close to the DMRG value -75.63717415.

For imaginary time evolution, since the propagator is not unitary, the norm will increase exponentially.
You may use the extra keyword ``normalize_mps`` to normalize MPS after each time step. The norm will still be computed and printed, but it will not be accumulated.

Finally, we can verify the energy at ``T = 0.0`` and ``T = 0.2`` and compute the overlap for these states.
The overlap between the all four states can be computed using the following input : ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP.ORIG

    hf_occ integral
    schedule
    0 500 0 0
    end
    maxiter 10

    mps_tags KET-CPX-0 BRA KET-CPX-1 BRAEX
    restart_tran_oh
    complex_mps
    overlap
    noreorder

The output is: ::

    $ grep 'OH' dmrg-5.out
    OH Energy    0 -    0 = RE    1.000000000000002 + IM    0.000000000000000
    OH Energy    1 -    0 = RE   -0.845792004408687 + IM   -0.533433527528264
    OH Energy    1 -    1 = RE    0.999986418355938 + IM    0.000000000000000
    OH Energy    2 -    0 = RE   -0.000000000000000 + IM    0.000000000000000
    OH Energy    2 -    1 = RE   -0.000000827506956 + IM   -0.000000742303613
    OH Energy    2 -    2 = RE    1.000000000000004 + IM    0.000000000000000
    OH Energy    3 -    0 = RE    0.000001731091412 + IM   -0.000000316659748
    OH Energy    3 -    1 = RE   -0.000001122421894 + IM    0.000002348984005
    OH Energy    3 -    2 = RE   -0.836158473098047 + IM   -0.548435696470209
    OH Energy    3 -    3 = RE    0.999988252849993 + IM    0.000000000000000

Here in the output each MPS gets a number, according to the order of tags in ``mps_tags``.
We have ``0 (KET-CPX-0), 1 (BRA), 2 (KET-CPX-1)`` and ``3 (BRAEX)``.

Note that state 1 (not normalized) is time evolved from state 0 (normalized).
We see that the overlap ``<1|1>`` is exactly 1. To get the overlap between the normalized states, we have: ::

    < normlized(0) | normlized(1) >
    = <0|1> / sqrt(<0|0> * <1|1>)
    = (-0.845792004408687 -0.533433527528264j) / sqrt( 0.999986418355938 * 1.000000000000002)
    = -0.8457977480901698 -0.5334371500173138j

The absolute value and the angle of this complex overlap is : ::

       np.abs( -0.8457977480901698 -0.5334371500173138j ) =  0.9999645112167714
    np.angle ( -0.8457977480901698 -0.5334371500173138j ) = -2.578911293480138

The absolute value is close to one. So the time evolution simply introduced a complex phase factor for the state, as expected. The complex phase factor can be computed as the remainder of ``E t`` divided by ``2 pi``: ::

    -75.72638951483646 * 0.2 % (2 * np.pi) - 2 * np.pi = -2.5789072886081197

Which is close to the printed value.

Also note that the overlap between the ground state and the excited state ``<2|0>`` is exactly zero. The corresponding overlap between the time evolved states ``<3|1>`` is slightly different from zero, mainly due to the time step error and truncation error.

We can also get the energy expetation, by removing the keyword ``overlap``: ::

    $ grep 'OH' dmrg-6.out
    OH Energy    0 -    0 = RE  -75.726296730204453 + IM    0.000000000000000
    OH Energy    1 -    0 = RE   64.049088006450049 + IM   40.394772180607831
    OH Energy    1 -    1 = RE  -75.725361025967970 + IM   -0.000000000000007
    OH Energy    2 -    0 = RE    0.000000000000008 + IM    0.000000000000000
    OH Energy    2 -    1 = RE    0.000061050951670 + IM    0.000056012958492
    OH Energy    2 -    2 = RE  -75.637174152353893 + IM    0.000000000000000
    OH Energy    3 -    0 = RE   -0.000132735557064 + IM    0.000024638559206
    OH Energy    3 -    1 = RE    0.000086585167013 + IM   -0.000178008928209
    OH Energy    3 -    2 = RE   63.244928578558032 + IM   41.482021915322555
    OH Energy    3 -    3 = RE  -75.636371985782972 + IM    0.000000000000000

Note that here not all states are normalized, the printed value is not directly the energy.
The printed value is ``<A|H|B>``, but the energy is ``<A|H|B>/<A|B>``.
So the printed value should be divided by the square of the norm of the MPS (see previous output). For example, for state 1 we have : ::

    -75.725361025967970 / 0.999986418355938 = -75.72638951483646

Which is the same as the number ``<E>`` printed by the time evolution (-75.726389514836484).
