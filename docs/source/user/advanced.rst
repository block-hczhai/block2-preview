
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
