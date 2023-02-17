
.. _dev_pg_mapping:

Point Group Mapping
===================

Here we discuss how to transform an MPS with a point group (PG) (such as D\ :sub:`2h`) to
an MPS with another PG (such as C\ :sub:`2v` or C\ :sub:`s`).

Limitations:

* Can transform from high-order PG (ket) to low-order PG (bra) or low-order PG (ket) to high-order PG (bra);
* The mapping between the two PG must be a homomorphism. As long as it is a homomorphism, any mapping can be used.
* For normal MPS, the transformation only have a tiny fitting error.
  For MPS with big site, the matrix elements in the big-site tensor in the MPS will be artificially reordered.
  (Because the "fusing order" inside the big site can have an influence on the order of states within each symmetry block.
  Since "fusing order" in the big site can be arbitrary, the mapping code itself cannot figure out the correct mapping for
  the order of states within each symmetry block. For normal site, there is only one state for each symmetry block so
  there is no problem.)
  So the transformed big-site tensor will not be accurate (but the normal site tensors in a big-site MPS will still be accurate).
* The integral (FCIDUMP) with the two PG must be exactly the same.
  Consider the following case: if you generate the integral from ``pyscf``, you calculate one integral with ``D2h``
  symmetry in the molecule, and another integral with ``C2v`` symmetry in the molecule.
  Then there can be small (or big) changes in the integral. Then you cannot use this feature,
  since point group symmetry is not the only thing that changed.

Example
-------

The example integral file ``C2.CAS.PVDZ.FCIDUMP`` can be found in the ``data`` folder.

.. highlight:: python3

First we do a ground state calculation using D\ :sub:`2h` point group: ::

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

    fcidump = FCIDUMP()
    fcidump.read('C2.CAS.PVDZ.FCIDUMP')

    # D2H Hamiltonian
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

Then we define a Hamiltonian with a different PG (but the same integral). The mapping should be based on the XOR notation.
Here we create ``hamil_c2v`` by mapping D2h irreps to C2v irreps.
The mapping is not unique, you may need to figure out the actual mapping based on
how you will need to mix orbitals with different irreps.
Note that something like ``pg_map = lambda x: x & 6`` or ``pg_map = lambda x: x & 3`` should also work. ::

    # C2V Hamiltonian
    pg_map = lambda x: (x & 6) >> 1
    orb_sym_c2v = VectorUInt8([pg_map(x) for x in orb_sym]) # the mapping is not unique
    hamil_c2v = HamiltonianQC(vacuum, n_sites, orb_sym_c2v, fcidump)
    target_c2v = SX(target.n, target.twos, pg_map(target.pg))
    print("C2V ORB SYM = ", hamil_c2v.orb_sym)

To transform MPS, we need a special identity MPO. This identity will not have bond dimension 1
since it has to mix different PG irreps. If the MPS does not have any big-site,
the last two parameters ``orb_sym_c2v, orb_sym`` can be omitted. ::

    # Identity MPO for PG mapping
    delta_target = (target_c2v - target)[0]
    impo = IdentityMPO(hamil_c2v.basis, hamil.basis, vacuum,
        delta_target, hamil.opf, orb_sym_c2v, orb_sym)
    impo = SimplifiedMPO(impo, NoTransposeRule(RuleQC()))

Next, we can perform the transformation of MPS using fitting. ::

    # C2V MPS
    mps_info_c2v = MPSInfo(n_sites, vacuum, target_c2v, hamil_c2v.basis)
    mps_info_c2v.tag = 'KET-C2V'
    mps_info_c2v.set_bond_dimension(500)
    mps_info_c2v.save_data('./mps_info_c2v.bin')
    mps_c2v = MPS(n_sites, mps.center, 2)
    mps_c2v.initialize(mps_info_c2v)
    mps_c2v.random_canonicalize()
    mps_c2v.save_mutable()
    mps_info_c2v.save_mutable()

    # Linear
    me = MovingEnvironment(impo, mps_c2v, mps, "LIN")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    cps = Linear(me, VectorUBond([500]), VectorUBond([500]))
    norm = cps.solve(20, mps.center == 0, 1E-8)
    print('Norm = %20.15f' % norm)

Finally, we can check whether the MPS gives the correct energy in the new C2v basis: ::

    # C2V MPO
    mpo_c2v = MPOQC(hamil_c2v, QCTypes.Conventional)
    mpo_c2v = SimplifiedMPO(mpo_c2v, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # Expectation
    me = MovingEnvironment(mpo_c2v, mps_c2v, mps_c2v, "DMRG")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    ex = Expect(me, 500, 500)
    ener_c2v = ex.solve(False)
    print('C2V Energy = %20.15f' % ener_c2v)

The printed energy should be very close to the D2h sweep energy at the last site of the last sweep.
Note that this may not be the same as the DMRG energy, which is the lowest energy in the last sweep,
because here the MPS is transformed from the previous D2h MPS with the center at the last site.

.. highlight:: text

If the MPS contains big-site, there can be a much larger error in the energy due to the reordering
of states in the big-site MPS tensor. Re-optimizing the big-site tensor may solve this problem.
In addition, ``me.delayed_contraction = OpNamesSet.normal_ops()`` *must not* be set. 
Otherwise, the following assertion occurs: ::

    Assertion`a->get_type() == SparseMatrixTypes::Normal && b->get_type() == SparseMatrixTypes::Normal && c->get_type() == SparseMatrixTypes::Normal && v->get_type() == SparseMatrixTypes::Normal && da->get_type() == SparseMatrixTypes::Normal && db->get_type() == SparseMatrixTypes::Normal' failed.

Some reference output for this example: ::

    D2H ORB SYM =  VectorUInt8[ 5 0 6 5 3 5 0 0 5 0 3 6 5 0 3 6 7 2 7 2 7 2 1 4 0 5 ]
    <-- Site =    0-   1 .. Mmps =    3 Ndav =   1 E =    -75.7284493902 Error = 1.14e-16 FLOPS = 8.66e+05 Tdav = 0.00 T = 0.01
    DMRG Energy =  -75.728475543752168
    C2V ORB SYM =  VectorUInt8[ 2 0 3 2 1 2 0 0 2 0 1 3 2 0 1 3 3 1 3 1 3 1 0 2 0 2 ]
    Norm =    1.000000000000001
    C2V Energy =  -75.728449390238850

Inverse Mapping
---------------

.. highlight:: python3

The inverse mapping from C\ :sub:`2v` to D\ :sub:`2h` is also supported.
The script is basically the same (except the exchange between C\ :sub:`2v` and D\ :sub:`2h`): ::

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

    fcidump = FCIDUMP()
    fcidump.read('C2.CAS.PVDZ.FCIDUMP')

    # C2V Hamiltonian
    pg = "d2h"
    pg_map = lambda x: (x & 6) >> 1
    swap_pg = getattr(PointGroup, "swap_" + pg)
    vacuum = SX(0)
    target_d2h = SX(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    target_c2v = SX(target_d2h.n, target_d2h.twos, pg_map(target_d2h.pg))
    n_sites = fcidump.n_sites
    orb_sym_d2h = VectorUInt8(map(swap_pg, fcidump.orb_sym))
    orb_sym_c2v = VectorUInt8([pg_map(x) for x in orb_sym_d2h]) # the mapping is not unique
    hamil = HamiltonianQC(vacuum, n_sites, orb_sym_c2v, fcidump)
    print("C2V ORB SYM = ", hamil.orb_sym)

    # C2V MPS
    mps_info = MPSInfo(n_sites, vacuum, target_c2v, hamil.basis)
    mps_info.tag = 'KET'
    mps_info.set_bond_dimension(250)
    mps_info.save_data('./mps_info.bin')
    mps = MPS(n_sites, 0, 2)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    mps.save_mutable()
    mps_info.save_mutable()

    # C2V MPO
    mpo = MPOQC(hamil, QCTypes.Conventional)
    mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # C2V DMRG
    me = MovingEnvironment(mpo, mps, mps, "DMRG")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    dmrg = DMRG(me, VectorUBond([250, 500]), VectorDouble([1E-5] * 5 + [1E-6] * 5 + [0]))
    dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
    dmrg.davidson_conv_thrds = VectorDouble([1E-6] * 5 + [1E-7] * 5)
    ener = dmrg.solve(20, mps.center == 0, 1E-8)
    print('DMRG Energy = %20.15f' % ener)

    # D2H Hamiltonian
    hamil_d2h = HamiltonianQC(vacuum, n_sites, orb_sym_d2h, fcidump)
    print("D2H ORB SYM = ", hamil_d2h.orb_sym)

    # Identity MPO for PG mapping
    delta_target = (target_d2h - target_c2v)[0]
    impo = IdentityMPO(hamil_d2h.basis, hamil.basis, vacuum,
        delta_target, hamil.opf, orb_sym_d2h, orb_sym_c2v)
    impo = SimplifiedMPO(impo, NoTransposeRule(RuleQC()))

    # D2H MPS
    mps_info_d2h = MPSInfo(n_sites, vacuum, target_d2h, hamil_d2h.basis)
    mps_info_d2h.tag = 'KET-D2H'
    mps_info_d2h.set_bond_dimension(500)
    mps_info_d2h.save_data('./mps_info_d2h.bin')
    mps_d2h = MPS(n_sites, mps.center, 2)
    mps_d2h.initialize(mps_info_d2h)
    mps_d2h.random_canonicalize()
    mps_d2h.save_mutable()
    mps_info_d2h.save_mutable()

    # Linear
    me = MovingEnvironment(impo, mps_d2h, mps, "LIN")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    cps = Linear(me, VectorUBond([500]), VectorUBond([500]))
    norm = cps.solve(20, mps.center == 0, 1E-8)
    print('Norm = %20.15f' % norm)

    # D2H MPO
    mpo_d2h = MPOQC(hamil_d2h, QCTypes.Conventional)
    mpo_d2h = SimplifiedMPO(mpo_d2h, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    # D2H Expectation
    me = MovingEnvironment(mpo_d2h, mps_d2h, mps_d2h, "DMRG")
    me.delayed_contraction = OpNamesSet.normal_ops()
    me.cached_contraction = True
    me.init_environments(True)
    ex = Expect(me, 500, 500)
    ener_d2h = ex.solve(False) / norm ** 2
    print('D2H Energy = %20.15f' % ener_d2h)

.. highlight:: text

Some reference outputs for this example: ::

    C2V ORB SYM =  VectorUInt8[ 2 0 3 2 1 2 0 0 2 0 1 3 2 0 1 3 3 1 3 1 3 1 0 2 0 2 ]
    --> Site =   24-  25 .. Mmps =    3 Ndav =   1 E =    -75.7284490538 Error = 1.62e-19 FLOPS = 3.87e+05 Tdav = 0.00 T = 0.01
    DMRG Energy =  -75.728475021520978
    D2H ORB SYM =  VectorUInt8[ 5 0 6 5 3 5 0 0 5 0 3 6 5 0 3 6 7 2 7 2 7 2 1 4 0 5 ]
    Norm =    0.999999999998821
    D2H Energy =  -75.728449053829152

Initial Guess for Compression
-----------------------------

For large systems, the initial guess for ``Linear`` (``mps_c2v`` or ``mps_d2h`` in the above examples)
may be too bad, and very small overlap (``F`` value) with ``mps`` can be observed.
The MPS bond dimension will be kept as 1 or a very small number (it should be at least one, since
by default the random FCI initial guess is used, where at least one state is kept for each quantum number
in the initial guess).

To solve this problem, one can add ``cps.cutoff = 0`` before the line ``norm = cps.solve(...)``.
Alternatively, one can add ``cps.trunc_type = TruncationTypes.KeepOne * n`` before the line ``norm = cps.solve(...)``,
where ``n`` is a small positive integer.

Generating initial guess using occupation numbers may also alleviate this problem, but using the
above settings, better initial guess with occupation numbers is not mandatory.
