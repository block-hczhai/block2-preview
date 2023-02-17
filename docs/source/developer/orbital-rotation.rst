
.. _dev_orbital_rotation:

MPS Orbital Rotation
====================

In this part we explain how to tranform an MPS with one orbital basis to another orbital basis.
For the case when the new basis is the one with natural orbitals, please see :ref:`user_advanced` for a simple solution.

We assume that the orbital rotation only happens within each irrep. If this is not the case, you need to
first transform MPS from a higher-order point group to a lower-order point group, according to :ref:`dev_pg_mapping`.

Example
-------

.. highlight:: python3

We consider, for example, the rotation from Hartree-Fock orbitals to localized orbitals within each irrep.
As a first step, we construct these orbitals using ``pyscf``: ::

    from block2 import FCIDUMP, VectorUInt8
    from pyscf import gto, scf, mcscf, lo, tools, ao2mo
    from pyscf.mcscf import casci_symm
    import scipy.linalg
    import scipy.optimize
    import numpy as np
    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz', symmetry='d2h')
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 26, 8)

    ncore = mc.ncore
    nactorb = mc.ncas


    # localize orbitals
    def scdm(coeff, overlap):
        aux = lo.orth.lowdin(overlap)
        no = coeff.shape[1]
        ova = coeff.T @ overlap @ aux
        piv = scipy.linalg.qr(ova, pivoting=True)[2]
        bc = ova[:, piv[:no]]
        ova = np.dot(bc.T, bc)
        s12inv = lo.orth.lowdin(ova)
        return coeff @ bc @ s12inv


    # sort orbitals by irrep
    def irrep_sort(coeff):
        optimal_reorder = [0, 6, 3, 5, 7, 1, 4, 2]  # d2h
        orb_sym = casci_symm.label_symmetry_(mc, mo_coeff_act).orbsym
        orb_opt = [optimal_reorder[x] for x in orb_sym]
        idx = np.argsort(orb_opt)
        return coeff[:, idx], orb_sym[idx]


    # HF orbitals (old basis)
    mo_coeff_act = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas].copy()
    mo_coeff_act, mo_orb_sym = irrep_sort(mo_coeff_act)

    # Symmetrized localized orbitals (new basis)
    lmo_coeff_act = mo_coeff_act.copy()
    for isym in set(mo_orb_sym):
        mask = np.array(mo_orb_sym) == isym
        lmo_coeff_act[:, mask] = scdm(
            mo_coeff_act[:, mask], mol.intor('cint1e_ovlp_sph'))

where ``mo_coeff_act`` represents the AO to MO coefficients for the old orbitals,
and ``lmo_coeff_act`` represents the AO to MO coefficients for the new orbitals.
The two sets of orbitals share the same irrep labels ``mo_orb_sym``.

It is not necessary that the orbitals should be sorted according to irrep.
But if orbitals with the same irrep are far from each other, the orbital rotation may be
likely non-local.

Next, we construct the rotation matrix between the two sets of orbitals: ::

    # orbital transform rot[old, new]
    orb_rot = np.linalg.pinv(mo_coeff_act) @ lmo_coeff_act
    assert np.linalg.norm(orb_rot.T - np.linalg.inv(orb_rot)) < 1E-12
    assert np.linalg.norm(lmo_coeff_act - mo_coeff_act @ orb_rot) < 1E-12

To make the transformation as local as possible (so that the required MPS bond dimension
for time evolution can be lower), we need to do some premutation and flipping of signs
in the rotation matrix and consequently the new orbitals: ::

    # change det sign and reorder rot within each irrep
    def regularize_rot_mat(rot, orb_sym, iprint=False):
        rot = rot.copy()
        for isym in set(orb_sym):
            mask = np.array(orb_sym) == isym
            # orbital matching (reordering within irrep)
            kmidx = scipy.optimize.linear_sum_assignment(
                1 - rot[mask, :][:, mask] ** 2)[1]
            if iprint:
                print("overlap before matching = ", np.sum(
                    np.diag(rot[mask, :][:, mask]) ** 2))
            rot[:, mask] = rot[:, mask][:, kmidx]
            if iprint:
                print("overlap after matching = ", np.sum(
                    np.diag(rot[mask, :][:, mask]) ** 2))
            # change sign to make it quasi-positive-definite
            for j in range(len(np.arange(len(mask))[mask])):
                mrot = rot[mask, :][:j + 1, :][:, mask][:, :j + 1]
                mrot_det = np.linalg.det(mrot)
                if iprint:
                    print("ISYM = %d J = %d MDET = %15.10f" % (isym, j, mrot_det))
                if mrot_det < 0:
                    mask0 = np.arange(len(mask), dtype=int)[mask][j]
                    rot[:, mask0] = -rot[:, mask0]
        return rot


    reg_orb_rot = regularize_rot_mat(orb_rot, mo_orb_sym)
    assert np.linalg.det(reg_orb_rot) > 0

    # regularized new basis
    lmo_coeff_act = mo_coeff_act @ reg_orb_rot

Note that ``reg_orb_rot`` must have a +1 determinant, because otherwise
the logarithm of it will have to be complex.

Now we can calculate the logarithm of the rotation matrix, namely, ``kappa``: ::

    # get logarithm of the rotation matrix
    def get_kappa(rot, orb_sym):
        kappa = np.zeros_like(rot)
        for isym in set(orb_sym):
            mask = np.array(orb_sym) == isym
            mrot = rot[mask, :][:, mask]
            # scipy.linalg.logm works perfectly for
            # quasi-positive-definite matrices
            mkappa = scipy.linalg.logm(mrot)
            assert mkappa.dtype == float
            gkappa = np.zeros((kappa.shape[0], mkappa.shape[1]))
            gkappa[mask, :] = mkappa
            kappa[:, mask] = gkappa
        assert np.linalg.norm(
            scipy.linalg.expm(kappa) - rot) < 1E-10
        assert np.linalg.norm(kappa + kappa.T) < 1E-10
        return kappa

    kappa = get_kappa(reg_orb_rot, mo_orb_sym)

Next, The ``FCIDUMP`` objects for DMRG and time evolution can be constructed
from the orbitals and ``kappa``, respectively: ::

    def get_fcidump(coeff, orb_sym, fname=None, tol=1E-13):
        mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas] = coeff
        mp_orb_sym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orb_sym]
        h1e, e_core = mc.get_h1cas()
        h1e = h1e.flatten()
        g2e = ao2mo.restore(8, mc.get_h2cas(), mc.ncas)
        h1e[np.abs(h1e) < tol] = 0
        g2e[np.abs(g2e) < tol] = 0
        na, nb = mc.nelecas
        fcidump = FCIDUMP()
        fcidump.initialize_su2(mc.ncas, na + nb, na - nb, 1, e_core, h1e, g2e)
        fcidump.orb_sym = VectorUInt8(mp_orb_sym)
        assert fcidump.symmetrize(VectorUInt8(orb_sym)) < 1E-10
        if fname is not None:
            fcidump.write(fname)
        return fcidump


    def get_kappa_fcidump(kappa, orb_sym, fname=None, tol=1E-13):
        mp_orb_sym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orb_sym]
        na, nb = mc.nelecas
        fcidump = FCIDUMP()
        kappa = kappa.flatten()
        kappa[np.abs(kappa) < tol] = 0
        fcidump.initialize_h1e(mc.ncas, na + nb, na - nb, 1, 0.0, kappa)
        fcidump.orb_sym = VectorUInt8(mp_orb_sym)
        assert fcidump.symmetrize(VectorUInt8(orb_sym)) < 1E-10
        if fname is not None:
            fcidump.write(fname)
        return fcidump


    fd_old = get_fcidump(mo_coeff_act, mo_orb_sym)
    fd_new = get_fcidump(lmo_coeff_act, mo_orb_sym)
    fd_kappa = get_kappa_fcidump(kappa, mo_orb_sym)

where ``fd_old`` is for the DMRG in the old basis, and ``fd_new`` is for the DMRG
in the new basis, and ``fd_kappa`` is for the orbital transform.

Now we are ready to do a DMRG in the old basis to find the ground-state MPS in this basis: ::

    from block2 import *
    from block2.su2 import *
    import numpy as np
    SX = SU2

    Global.frame = DoubleDataFrame(10 * 1024 ** 2, 10 * 1024 ** 3, "nodex")
    n_threads = Global.threading.n_threads_global
    Global.threading = Threading(
        ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
        n_threads, n_threads, 1)
    Global.threading.seq_type = SeqTypes.Tasked
    Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
    Global.frame.minimal_disk_usage = True
    Global.frame.use_main_stack = False
    print(Global.frame)
    print(Global.threading)



    # Hamiltonian in old basis
    fcidump = fd_old
    pg = "d2h"
    swap_pg = getattr(PointGroup, "swap_" + pg)
    vacuum = SX(0)
    target = SX(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    n_sites = fcidump.n_sites
    orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
    hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)
    print("D2H ORB SYM = ", hamil.orb_sym)

    # MPS
    mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
    mps_info.tag = 'KET'
    mps_info.set_bond_dimension(250)
    mps_info.save_data('./mps_info.bin')
    mps = MPS(n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    mps.save_mutable()
    mps_info.save_mutable()

    # MPO
    mpo = MPOQC(hamil, QCTypes.Conventional)
    mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # DMRG
    me = MovingEnvironment(mpo, mps, mps, "DMRG")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    dmrg = DMRG(me, VectorUBond([250, 500]), VectorDouble([1E-5] * 5 + [1E-6] * 5 + [0]))
    dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
    dmrg.davidson_conv_thrds = VectorDouble([1E-6] * 5 + [1E-7] * 5)
    ener = dmrg.solve(20, mps.center == 0, 1E-8)
    print('DMRG Energy = %20.15f' % ener)

The following script can be used to transform the ground-state MPS to the new basis: ::

    # Hamiltonain for orbital transform
    hamil_kappa = HamiltonianQC(vacuum, n_sites, orb_sym, fd_kappa)

    # MPO (anti-Hermitian)
    mpo_kappa = MPOQC(hamil_kappa, QCTypes.Conventional)
    mpo_kappa = SimplifiedMPO(mpo_kappa, AntiHermitianRuleQC(RuleQC()),
        True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # Time Step
    dt = 0.05
    # Target time
    tt = 1.0
    n_steps = int(abs(tt) / abs(dt) + 0.1)
    assert np.abs(abs(n_steps * dt) - abs(tt)) < 1E-10
    print("Time Evolution NSTEPS = %d" % n_steps)
    me_kappa = MovingEnvironment(mpo_kappa, mps, mps, "DMRG")
    me_kappa.delayed_contraction = OpNamesSet.normal_ops()
    me_kappa.cached_contraction = True
    me_kappa.init_environments(True)

    # Time Evolution (anti-Hermitian)
    # te_type can be TETypes.RK4 or TETypes.TangentSpace (TDVP)
    te_type = TETypes.RK4
    te = TimeEvolution(me_kappa, VectorUBond([1000]), te_type)
    te.hermitian = False
    te.iprint = 2
    te.n_sub_sweeps = 1 if te.mode == TETypes.TangentSpace else 2
    te.normalize_mps = False
    for i in range(n_steps):
        if te.mode == TETypes.TangentSpace:
            te.solve(2, dt / 2, mps.center == 0)
        else:
            te.solve(1, dt, mps.center == 0)
        print("T = %10.5f <E> = %20.15f <Norm^2> = %20.15f" %
                ((i + 1) * dt, te.energies[-1], te.normsqs[-1]))

Note that when constructing MPO, ``AntiHermitianRuleQC`` has to be used.
Also ``te.hermitian`` must be set to ``False`` for anti-Hermitian "Hamiltonian",
otherwise it will be assumed Hermitian.

.. note::

    ``TimeEvolution`` can support both one-site and two-site algorithm, but
    we highly recommend the two-site algorithm as there is no noise,
    and the one-site algorithm may have severe problem with losing quantum numbers.

Since every step in time evolution is a unitary transform, the "energy" expectation
should always be zero, and the "norm" of the MPS should be close to one.
Normally, a too large discarded weight or "norm" far from 1 indicates that
the error during the transform is too large.

Finally, we can check the energy expectation of the transformed MPS in the new basis: ::

    # Hamiltonain in new basis
    hamil_new = HamiltonianQC(vacuum, n_sites, orb_sym, fd_new)

    # MPO
    mpo_new = MPOQC(hamil_new, QCTypes.Conventional)
    mpo_new = SimplifiedMPO(mpo_new, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # Energy Expectation
    me_new = MovingEnvironment(mpo_new, mps, mps, "OVL")
    me_new.delayed_contraction = OpNamesSet.normal_ops()
    me_new.cached_contraction = True
    me_new.init_environments(True)

    expect = Expect(me_new, mps.info.bond_dim, mps.info.bond_dim)
    ener_new = expect.solve(False, mps.center == 0)

    print('Energy expectation = %20.15f' % ener_new)

.. highlight:: text

Some reference outputs for this example: ::

    D2H ORB SYM =  VectorUInt8[ 0 0 0 0 0 0 5 5 5 5 5 5 7 7 7 2 2 2 6 6 6 3 3 3 1 4 ]
    DMRG Energy =  -75.728487321653233
    Time Evolution NSTEPS = 20
    T =    0.05000 <E> =   -0.000000000000000 <Norm^2> =    0.999999979398520
    T =    0.10000 <E> =   -0.000000000000000 <Norm^2> =    0.999999926838107
    ... ...
    T =    0.95000 <E> =    0.000000000000000 <Norm^2> =    0.999996763879923
    Time elapsed =      5.738 | E =       0.0000000000 | Norm^2 =       0.9999964412 | DW = 3.83e-08
    T =    1.00000 <E> =    0.000000000000000 <Norm^2> =    0.999996441150652
    Energy expectation =  -75.728011987963555

Distributed Parallelization
---------------------------

.. highlight:: python3

Since the "Hamiltonian" used in orbital rotation has only one-body term, it is more efficient
to use a different parallelization rule. The normal two-body parallelization rule can still be used,
but it will not provide any speed-up when more than one MPI processes are used.

The one-body only parallelization rule can be used in the following way: ::

    MPI = MPICommunicator()
    prule_one_body = ParallelRuleOneBodyQC(MPI)
    mpo_kappa = ParallelMPO(mpo_kappa, prule_one_body)

MRCI (Big-Site) Example
-----------------------

The same procedure can be easily applied to the big-site MPO and MPS for MRCI calculation, with very little change.
The above script for normal MPS can be reused without change for big-site until line ``from block2 import *``.

Then, for big-site MPO/MPS, the following script can be used: ::

    from block2 import *
    from block2.su2 import *
    import numpy as np
    SX = SU2

    Global.frame = DoubleDataFrame(10 * 1024 ** 2, 10 * 1024 ** 3, "nodex")
    n_threads = Global.threading.n_threads_global
    Global.threading = Threading(
        ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
        n_threads, n_threads, 1)
    Global.threading.seq_type = SeqTypes.Nothing
    Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
    Global.frame.minimal_disk_usage = True
    Global.frame.use_main_stack = False
    print(Global.frame)
    print(Global.threading)

    # create a big site in MPO
    n_ext, ci_order = 5, 2
    def create_big_site(hamil, mpo):
        mrci_mps_info = MRCIMPSInfo(hamil.n_sites, n_ext, ci_order, hamil.vacuum, target, hamil.basis)
        mpo.basis = hamil.basis
        for i in range(n_ext):
            mpo = FusedMPO(mpo, mpo.basis, mpo.n_sites - 2, mpo.n_sites - 1, mrci_mps_info.right_dims_fci[mpo.n_sites - 2])
        for k, op in mpo.tensors[-1].ops.items():
            smat = CSRSparseMatrix()
            if op.sparsity() > 0.75:
                smat.from_dense(op)
                op.deallocate()
            else:
                smat.wrap_dense(op)
            mpo.tensors[-1].ops[k] = smat
        mpo.sparse_form = mpo.sparse_form[:-1] + 'S'
        mpo.tf = TensorFunctions(CSROperatorFunctions(hamil.opf.cg))
        return mpo

    # Hamiltonian in old basis
    fcidump = fd_old
    pg = "d2h"
    swap_pg = getattr(PointGroup, "swap_" + pg)
    vacuum = SX(0)
    target = SX(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    n_sites = fcidump.n_sites
    orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
    hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)
    print("D2H ORB SYM = ", hamil.orb_sym)

    # MPO
    mpo = MPOQC(hamil, QCTypes.Conventional)
    mpo = create_big_site(hamil, mpo)
    mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # MPS
    mps_info = MPSInfo(mpo.n_sites, vacuum, target, mpo.basis)
    mps_info.tag = 'KET'
    mps_info.set_bond_dimension(250)
    mps_info.save_data('./mps_info.bin')
    mps = MPS(mpo.n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    mps.save_mutable()
    mps_info.save_mutable()

    # DMRG
    me = MovingEnvironment(mpo, mps, mps, "DMRG")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    dmrg = DMRG(me, VectorUBond([250, 500]), VectorDouble([1E-5] * 5 + [1E-6] * 5 + [0]))
    dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
    dmrg.davidson_conv_thrds = VectorDouble([1E-6] * 5 + [1E-7] * 5)
    ener = dmrg.solve(20, mps.center == 0, 1E-8)
    print('MRCI DMRG Energy = %20.15f' % ener)


    # Hamiltonain for orbital transform
    hamil_kappa = HamiltonianQC(vacuum, n_sites, orb_sym, fd_kappa)

    # MPO (anti-Hermitian)
    mpo_kappa = MPOQC(hamil_kappa, QCTypes.Conventional)
    mpo_kappa = create_big_site(hamil_kappa, mpo_kappa)
    mpo_kappa = SimplifiedMPO(mpo_kappa, AntiHermitianRuleQC(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # Time Step
    dt = 0.05
    # Target time
    tt = 1.0
    n_steps = int(abs(tt) / abs(dt) + 0.1)
    assert np.abs(abs(n_steps * dt) - abs(tt)) < 1E-10
    print("Time Evolution NSTEPS = %d" % n_steps)
    me_kappa = MovingEnvironment(mpo_kappa, mps, mps, "DMRG")
    me_kappa.delayed_contraction = OpNamesSet.normal_ops()
    me_kappa.cached_contraction = True
    me_kappa.init_environments(True)

    # Time Evolution (anti-Hermitian)
    # te_type can be TETypes.RK4 or TETypes.TangentSpace (TDVP)
    te_type = TETypes.RK4
    te = TimeEvolution(me_kappa, VectorUBond([1000]), te_type)
    te.hermitian = False
    te.iprint = 2
    te.n_sub_sweeps = 1 if te.mode == TETypes.TangentSpace else 2
    te.normalize_mps = False
    for i in range(n_steps):
        if te.mode == TETypes.TangentSpace:
            te.solve(2, dt / 2, mps.center == 0)
        else:
            te.solve(1, dt, mps.center == 0)
        print("T = %10.5f <E> = %20.15f <Norm^2> = %20.15f" %
                ((i + 1) * dt, te.energies[-1], te.normsqs[-1]))


    # Hamiltonain in new basis
    hamil_new = HamiltonianQC(vacuum, n_sites, orb_sym, fd_new)

    # MPO
    mpo_new = MPOQC(hamil_new, QCTypes.Conventional)
    mpo_new = create_big_site(hamil_new, mpo_new)
    mpo_new = SimplifiedMPO(mpo_new, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # Energy Expectation
    me_new = MovingEnvironment(mpo_new, mps, mps, "OVL")
    me_new.delayed_contraction = OpNamesSet.normal_ops()
    me_new.cached_contraction = True
    me_new.init_environments(True)

    expect = Expect(me_new, mps.info.bond_dim, mps.info.bond_dim)
    ener_new = expect.solve(False, mps.center == 0)

    print('Energy expectation = %20.15f' % ener_new)

where the big-site MPO is created using the function ``create_big_site``, where the right-boundary sites
in the MPO are folded to a big site using the ``FusedMPO`` class. Other more efficient methods for creating
a big site can be used, but note that, the big site in the three MPOs ``mpo``, ``mpo_kappa``, and ``mpo_new``
must be created using the same method. This is to ensure that the quantum number fusing order is consistent
among different MPOs. This is required because the same MPS is used with all these MPOs.

Also note that ``SeqTypes.Nothing`` (instead of ``SeqTypes.Tasked``) should be used for big-site with CSR matrices.

.. highlight:: text

Some reference outputs for this example: ::

    D2H ORB SYM =  VectorUInt8[ 0 0 0 0 0 0 5 5 5 5 5 5 7 7 7 2 2 2 6 6 6 3 3 3 1 4 ]
    MRCI DMRG Energy =  -75.727859086194130
    Time Evolution NSTEPS = 20
    T =    0.05000 <E> =    0.000000000000000 <Norm^2> =    0.999999980443349
    T =    0.10000 <E> =   -0.000000000000000 <Norm^2> =    0.999999930992521
    ... ...
    T =    0.95000 <E> =    0.000000000000000 <Norm^2> =    0.999996944337650
    Time elapsed =      6.035 | E =      -0.0000000000 | Norm^2 =       0.9999966399 | DW = 3.84e-08
    T =    1.00000 <E> =   -0.000000000000000 <Norm^2> =    0.999996639941846
    Energy expectation =  -75.727409014459965
