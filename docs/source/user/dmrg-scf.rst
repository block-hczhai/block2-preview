
.. highlight:: bash

.. _user_dmrgscf:

DMRGSCF (pyscf)
===============

In this section we explain how to use ``block2`` (and optionally ``StackBlock``) and ``pyscf`` for ``DMRGSCF`` (CASSCF with DMRG as the active space solver).

Preparation
-----------

``pyscf`` can be installed using ``pip install pyscf``.
One also needs to install the pyscf extension called ``dmrgscf``, which can be obtained from
`https://github.com/pyscf/dmrgscf <https://github.com/pyscf/dmrgscf>`_.
If it is installed using ``pip``, one also needs to create a file named ``settings.py`` under the ``dmrgscf`` folder, as follows: ::

    $ pip install git+https://github.com/pyscf/dmrgscf
    $ PYSCFHOME=$(pip show pyscf-dmrgscf | grep 'Location' | tr ' ' '\n' | tail -n 1)
    $ wget https://raw.githubusercontent.com/pyscf/dmrgscf/master/pyscf/dmrgscf/settings.py.example
    $ mv settings.py.example ${PYSCFHOME}/pyscf/dmrgscf/settings.py
    $ chmod +x ${PYSCFHOME}/pyscf/dmrgscf/nevpt_mpi.py

Here we also assume that you have installed ``block2`` either using ``pip`` or manually.

DMRGSCF (serial)
----------------

.. highlight:: python3

The following is an example python script for DMRGSCF using ``block2`` running in a single node without MPI parallelism: ::

    from pyscf import gto, scf, lib, dmrgscf
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz',
        symmetry='d2h', verbose=4, max_memory=10000) # mem in MB
    mf = scf.RHF(mol)
    mf.kernel()

    from pyscf.mcscf import avas
    nactorb, nactelec, coeff = avas.avas(mf, ["C 2p", "C 3p", "C 2s", "C 3s"])
    print('CAS = ', nactorb, nactelec)

    mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM=1000, tol=1E-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4))
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    mc.canonicalization = True
    mc.natorb = True
    mc.kernel(coeff)

.. note ::

    Alternatively, to use ``StackBlock`` instead of ``block2`` as the DMRG solver, one can change the line involving ``dmrgscf.settings.BLOCKEXE`` to: ::

        dmrgscf.settings.BLOCKEXE = os.popen("which block.spin_adapted").read().strip()
    
    Please see :ref:`user_mps_io` for the instruction for the installation of ``StackBlock``.

.. note ::

    It is important to set a suitable ``mc.fcisolver.threads`` if you have multiple CPU cores in the node,
    to get high efficiency.

.. highlight:: text

This will generate the following output: ::

    $ grep 'CASSCF energy' cas1.out
    CASSCF energy = -75.6231442712648

DMRGSCF (distributed parallel)
------------------------------

.. highlight:: python3

The following example is DMRGSCF in hybrid MPI (distributed) and openMP (shared memory) parallelism.
For example, we can use 7 MPI processors and each processor uses 4 threads
(so in total the calculation will be done with 28 CPU cores): ::

    from pyscf import gto, scf, lib, dmrgscf
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = 'mpirun -n 7 --bind-to none'

    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz',
        symmetry='d2h', verbose=4, max_memory=10000) # mem in MB
    mf = scf.RHF(mol)
    mf.kernel()

    from pyscf.mcscf import avas
    nactorb, nactelec, coeff = avas.avas(mf, ["C 2p", "C 3p", "C 2s", "C 3s"])
    print('CAS = ', nactorb, nactelec)

    mc = dmrgscf.DMRGSCF(mf, nactorb, nactelec, maxM=1000, tol=1E-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = 4
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    mc.canonicalization = True
    mc.natorb = True
    mc.kernel(coeff)

.. note ::

    To use MPI with ``block2``, the block2 must be either (a) installed using ``pip install block2-mpi``
    or (b) manually built with ``-DMPI=ON``. Note that the ``block2`` installed using ``pip install block2``
    cannot be used together with ``mpirun`` if there are more than one processors (if this happens,
    it will generate wrong results and undefined behavior).

    If you have already ``pip install block2``, you must first ``pip uninstall block2`` then ``pip install block2-mpi``.

.. note ::

    If you do not have the ``--bind-to`` option in the ``mpirun`` command, sometimes every processor will only
    be able to use one thread (even if you set a larger number in the script), which will decrease the CPU usage
    and efficiency.

.. highlight:: text

This will generate the following output: ::

    $ grep 'CASSCF energy' cas2.out
    CASSCF energy = -75.6231442712753

CASSCF Reference
----------------

.. highlight:: python3

For this small (8, 8) active space, we can also compare the above DMRG results with the CASSCF result: ::

    from pyscf import gto, scf, lib, mcscf
    import os

    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz',
        symmetry='d2h', verbose=4, max_memory=10000) # mem in MB
    mf = scf.RHF(mol)
    mf.kernel()

    from pyscf.mcscf import avas
    nactorb, nactelec, coeff = avas.avas(mf, ["C 2p", "C 3p", "C 2s", "C 3s"])
    print('CAS = ', nactorb, nactelec)

    mc = mcscf.CASSCF(mf, nactorb, nactelec)
    mc.fcisolver.conv_tol = 1E-10
    mc.canonicalization = True
    mc.natorb = True
    mc.kernel(coeff)

.. highlight:: text

This will generate the following output: ::

    $ grep 'CASSCF energy' cas3.out
    CASSCF energy = -75.6231442712446

State-Average with Different Spins
----------------------------------

.. highlight:: python3

The following is an example python script for state-averaged DMRGSCF with singlet and triplet: ::

    from pyscf import gto, scf, lib, dmrgscf, mcscf
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz',
        symmetry='d2h', verbose=4, max_memory=10000) # mem in MB
    mf = scf.RHF(mol)
    mf.kernel()

    from pyscf.mcscf import avas
    nactorb, nactelec, coeff = avas.avas(mf, ["C 2p", "C 3p", "C 2s", "C 3s"])
    print('CAS = ', nactorb, nactelec)

    lib.param.TMPDIR = os.path.abspath(lib.param.TMPDIR)

    solvers = [dmrgscf.DMRGCI(mol, maxM=1000, tol=1E-10) for _ in range(2)]
    weights = [1.0 / len(solvers)] * len(solvers)

    solvers[0].spin = 0
    solvers[1].spin = 2

    for i, mcf in enumerate(solvers):
        mcf.runtimeDir = lib.param.TMPDIR + "/%d" % i
        mcf.scratchDirectory = lib.param.TMPDIR + "/%d" % i
        mcf.threads = 8
        mcf.memory = int(mol.max_memory / 1000) # mem in GB

    mc = mcscf.CASSCF(mf, nactorb, nactelec)
    mcscf.state_average_mix_(mc, solvers, weights)

    mc.canonicalization = True
    mc.natorb = True
    mc.kernel(coeff)

.. note ::

    The ``mc`` parameter in the function ``state_average_mix_`` must be a ``CASSCF`` object.
    It cannot be a ``DMRGSCF`` object (will produce a runtime error).

.. highlight:: text

This will generate the following output: ::

    $ grep 'State ' cas4.out
    State 0 weight 0.5  E = -75.6175232350073 S^2 = 0.0000000
    State 1 weight 0.5  E = -75.298522666384  S^2 = 2.0000000

Unrestricted DMRGSCF
--------------------

.. highlight:: python3

One can also perform Unrestricted CASSCF (UCASSCF) with ``block2`` using a UHF reference.
Currently this is not directly supported by the ``pyscf/dmrgscf`` package, but here we can add some small modifications.
The following is an example: ::

    from pyscf import gto, scf, lib, dmrgscf, mcscf, fci
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz',
        symmetry=False, verbose=4, max_memory=10000) # mem in MB
    mf = scf.UHF(mol)
    mf.kernel()

    def write_uhf_fcidump(DMRGCI, h1e, g2e, n_sites, nelec, ecore=0, tol=1E-15):

        import numpy as np
        from pyscf import ao2mo
        from subprocess import check_call
        from block2 import FCIDUMP, VectorUInt8

        if isinstance(nelec, (int, np.integer)):
            na = nelec // 2 + nelec % 2
            nb = nelec - na
        else:
            na, nb = nelec

        assert isinstance(h1e, tuple) and len(h1e) == 2
        assert isinstance(g2e, tuple) and len(g2e) == 3

        mh1e_a = h1e[0][np.tril_indices(n_sites)]
        mh1e_b = h1e[1][np.tril_indices(n_sites)]
        mh1e_a[np.abs(mh1e_a) < tol] = 0.0
        mh1e_b[np.abs(mh1e_b) < tol] = 0.0

        g2e_aa = ao2mo.restore(8, g2e[0], n_sites)
        g2e_bb = ao2mo.restore(8, g2e[2], n_sites)
        g2e_ab = ao2mo.restore(4, g2e[1], n_sites)
        g2e_aa[np.abs(g2e_aa) < tol] = 0.0
        g2e_bb[np.abs(g2e_bb) < tol] = 0.0
        g2e_ab[np.abs(g2e_ab) < tol] = 0.0

        mh1e = (mh1e_a, mh1e_b)
        mg2e = (g2e_aa, g2e_bb, g2e_ab)

        cmd = ' '.join((DMRGCI.mpiprefix, "mkdir -p", DMRGCI.scratchDirectory))
        check_call(cmd, shell=True)
        if not os.path.exists(DMRGCI.runtimeDir):
            os.makedirs(DMRGCI.runtimeDir)

        fd = FCIDUMP()
        fd.initialize_sz(n_sites, na + nb, na - nb, 1, ecore, mh1e, mg2e)
        fd.orb_sym = VectorUInt8([1] * n_sites)
        integral_file = os.path.join(DMRGCI.runtimeDir, DMRGCI.integralFile)
        fd.write(integral_file)
        DMRGCI.groupname = None
        DMRGCI.nonspinAdapted = True
        return integral_file

    def make_rdm12s(DMRGCI, state, norb, nelec, **kwargs):

        import numpy as np

        if isinstance(nelec, (int, np.integer)):
            na = nelec // 2 + nelec % 2
            nb = nelec - na
        else:
            na, nb = nelec

        file2pdm = "2pdm-%d-%d.npy" % (state, state) if DMRGCI.nroots > 1 else "2pdm.npy"
        dm2 = np.load(os.path.join(DMRGCI.scratchDirectory, "node0", file2pdm))
        dm2 = dm2.transpose(0, 1, 4, 2, 3)
        dm1a = np.einsum('ikjj->ki', dm2[0]) / (na - 1)
        dm1b = np.einsum('ikjj->ki', dm2[2]) / (nb - 1)

        return (dm1a, dm1b), dm2

    dmrgscf.dmrgci.writeIntegralFile = write_uhf_fcidump
    dmrgscf.DMRGCI.make_rdm12s = make_rdm12s

    mc = mcscf.UCASSCF(mf, 8, 8)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000, tol=1E-7)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ["OMP_NUM_THREADS"])
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    mc.canonicalization = True
    mc.natorb = True
    mc.kernel()

.. note ::

    In the above example, ``mf`` is the ``UHF`` object and ``mc`` is the ``UCASSCF`` object.
    It is important to ensure that both of them are with unrestricted orbitals.
    Otherwise the calculation may be done with only restricted orbitals.
    ``DMRGSCF`` wrapper cannot be used for this example.

.. note ::

    Due to limitations in ``pyscf/UCASCI``, currently the point group symmetry is not supported
    in UCASSCF/UCASCI with DMRG solver.
    ``pyscf/avas`` does not support creating active space with unrestricted orbtials
    so here we did not use ``avas``. The above example will not work with ``StackBlock``
    (the compatibility with ``StackBlock`` will be considered in future).

.. highlight:: text

This will generate the following output: ::

    $ grep 'UCASSCF energy' cas5.out
    UCASSCF energy = -75.6231442541606

UCASSCF Reference
-----------------

.. highlight:: python3

We compare the above DMRG results with the UCASSCF result using the FCI solver: ::

    mc = mcscf.UCASSCF(mf, 8, 8)
    mc.fcisolver.conv_tol = 1E-10
    mc.canonicalization = True
    mc.natorb = True
    mc.kernel(coeff)

.. highlight:: text

This will generate the following output: ::

    $ grep 'UCASSCF energy' cas6.out
    UCASSCF energy = -75.6231442706386

DMRGSCF Nuclear Gradients and Geometry Optimization
---------------------------------------------------

.. highlight:: python3

The following is an example python script for computing DMRGSCF nuclear gradients and geometry optimization using ``block2``: ::

    from pyscf import gto, scf, lib, dmrgscf
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='ccpvdz',
        symmetry='d2h', verbose=4, max_memory=10000) # mem in MB
    mf = scf.RHF(mol)
    mf.kernel()

    from pyscf.mcscf import avas
    nactorb, nactelec, coeff = avas.avas(mf, ["C 2p", "C 3p", "C 2s", "C 3s"])
    print('CAS = ', nactorb, nactelec)

    mc = mcscf.CASSCF(mf, nactorb, nactelec)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000, tol=1E-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 4))
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    mc.canonicalization = True
    mc.natorb = True
    mc.kernel(coeff)

    grad = mc.nuc_grad_method().kernel()

    mol_eq = mc.nuc_grad_method().optimizer(solver='geomeTRIC').kernel()
    print(mol_eq.atom_coords())

.. highlight:: text

This will generate the following output (the nuclear gradient at the initial geometry and the optimized geometry): ::

    $ grep -A 4 'SymAdaptedCASSCF gradients' cas7.out
    --------------- SymAdaptedCASSCF gradients ---------------
            x                y                z
    0 C     0.0000000000     0.0000000000     0.0388202961
    1 C     0.0000000000     0.0000000000    -0.0388202961
    ----------------------------------------------
    $ tail -n 3 cas7.out
    cycle 3: E = -75.6240204052  dE = -5.51573e-07  norm(grad) = 9.37108e-05
    [[ 0.          0.         -1.19709701]
    [ 0.          0.          1.19709701]]

.. note ::

    Currently, gradients for UCASSCF is not supported in ``pyscf``.
    The geometry optimization part requires an additional module called ``geomeTRIC``,
    which can be installed via ``pip install geometric``.

DMRG-SC-NEVPT2
--------------

.. highlight:: python3

The following is an example python script for a DMRG-SC-NEVPT2 calculation (with explicit 4pdm) using ``block2``: ::

    from pyscf import gto, scf, mcscf, mrpt, dmrgscf, lib
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    mc = mcscf.CASSCF(mf, 6, 8)

    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-10)
    mc.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.threads = 8
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.natorb = True
    mc.run()

    sc = mrpt.NEVPT(mc).run()

The alternative faster ``compress_approx`` approach using MPS compression is also supported: ::

    from pyscf import gto, scf, mcscf, mrpt, dmrgscf, lib
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    mc = mcscf.CASSCF(mf, 6, 8)

    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-10)
    mc.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.threads = 8
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.natorb = True
    mc.run()

    sc = mrpt.NEVPT(mc).compress_approx(maxM=200).run()

.. highlight:: text

This will generate the following output (for ``compress_approx`` approach): ::

    $ grep 'CASSCF energy' sc-nevpt2.out
    CASSCF energy = -149.708657771219
    $ grep 'Nevpt2 Energy' sc-nevpt2.out
    Nevpt2 Energy = -0.249182302692906

So the total NEVPT2 energy using the ``compress_approx`` approach is ``-149.708657771219 + -0.249182302692906 = -149.9578400739119``.

.. note ::

    The first "4pdm" approach is not supported by ``StackBlock``, but it is supported in the old ``Block`` code.
    The second "compression" approach is supported by ``StackBlock``.
    ``Block2`` supports both approaches.

    When using the second approach, it will generate a warning saying that ``WARN: DMRG executable file for
    nevptsolver is the same to the executable file for DMRG solver. If they are both compiled by MPI compilers,
    they may cause error or random results in DMRG-NEVPT calculation.``. Please ignore this warning for ``block2``.
    For ``block2``, it is okay to set ``BLOCKEXE`` and ``BLOCKEXE_COMPRESS_NEVPT`` to the same file.
    ``BLOCKEXE_COMPRESS_NEVPT`` can be compiled with or without MPI.
    So only a single version of ``block2main`` is required. If you want to use MPI, please set both
    ``BLOCKEXE`` and ``BLOCKEXE_COMPRESS_NEVPT`` to the same ``block2main`` and compile ``block2`` with MPI,
    or use ``pip install block2-mpi``, and then set an appropriate ``MPIPREFIX``.

    The second "compression" approach requires the ``mpi4py`` python package. Make sure ``import mpi4py`` works in
    python before trying this example. Also, make sure that the file ``${PYSCFHOME}/pyscf/dmrgscf/nevpt_mpi.py``
    has the ``execute`` permission. You can do ``chmod +x ${PYSCFHOME}/pyscf/dmrgscf/nevpt_mpi.py``
    to fix the permission.

    Note that for the second "compression" approach, if you need to add any extra keywords for the DMRG solver,
    such as ``singlet_embedding``, you need to add it using ``mc.fcisolver.block_extra_keyword`` instead of
    ``mc.fcisolver.extraline``.

DMRG-SC-NEVPT2 (Multi-State)
----------------------------

.. highlight:: python3

The following is an example input file for state-averaged DMRGSCF for three states,
and then the SC-NEVPT2 treatment of each of the three states. ::

    import numpy as np
    from pyscf import gto, scf, mcscf, mrpt, dmrgscf, lib
    import os

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    # state average casscf
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-10)
    mc.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.threads = 8
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.nroots = 3
    mc = mcscf.state_average_(mc, [1.0 / 3] * 3)
    mc.kernel()
    mf.mo_coeff = mc.mo_coeff

    # need an extra casci before calling mrpt
    mc = mcscf.CASCI(mf, 6, 8)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-10)
    mc.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.threads = 8
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.nroots = 3
    mc.natorb = True
    mc.kernel()

    # canonicalization for each state
    ms = [None] * mc.fcisolver.nroots
    cs = [None] * mc.fcisolver.nroots
    es = [None] * mc.fcisolver.nroots
    for ir in range(mc.fcisolver.nroots):
        ms[ir], cs[ir], es[ir] = mc.canonicalize(mc.mo_coeff, ci=mc.ci[ir], cas_natorb=False)

    refs = [-149.956650684550, -149.725338427894, -149.725338427894]

    # mrpt
    for ir in range(mc.fcisolver.nroots):
        mc.mo_coeff, mc.ci, mc.mo_energy = ms[ir], cs, es[ir]
        mr = mrpt.nevpt2.NEVPT(mc).set(canonicalized=True).compress_approx(maxM=200).run(root=ir)
        print('root =', ir, 'E =', mc.e_tot[ir] + mr.e_corr, 'diff =', mc.e_tot[ir] + mr.e_corr - refs[ir])

.. highlight:: text

This will generate the following output: ::

    $ grep 'diff' multi.out
    root = 0 E = -149.95664910937998 diff = 1.5751700175314909e-06
    root = 1 E = -149.72529848179465 diff = 3.994609934920845e-05
    root = 2 E = -149.7252985999243 diff = 3.9827969715133804e-05

.. note ::

    The above script should generate the same result if the explicit 4PDM approach is used,
    by removing ``.compress_approx(maxM=200)``.

    Changing ``mc.fcisolver`` to the default FCI active space solver should also generate the same result
    (note that ``.compress_approx(maxM=200)`` is not supported by the FCI active space solver).

    When the FCI active space solver is used, explicit canonicalization is also optional, namely,
    one can also remove ``.set(canonicalized=True)`` and ``mc.mo_coeff, mc.ci, mc.mo_energy = ms[ir], cs, es[ir]``
    and the result will still be the same.


DMRG-IC-NEVPT2
--------------

.. highlight:: python3

The following is an example python script for SC-NEVPT2 / IC-NEVPT2 with equations derived on the fly
(using the FCI solver): ::

    import numpy
    from pyscf import gto, scf, mcscf

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-11
    mc.canonicalization = True
    mc.run()

    from pyblock2.icmr.scnevpt2 import WickSCNEVPT2
    wsc = WickSCNEVPT2(mc).run()

    from pyblock2.icmr.icnevpt2_full import WickICNEVPT2
    wic = WickICNEVPT2(mc).run()

.. highlight:: text

This will generate the following output: ::

    $ grep 'E(WickSCNEVPT2)' nevpt2.out
    E(WickSCNEVPT2) = -149.9578403403482  E_corr_pt = -0.2491825691128931
    $ grep 'E(WickICNEVPT2)' nevpt2.out
    E(WickICNEVPT2) = -149.9601376470851  E_corr_pt = -0.2514798758497859

.. highlight:: python3

The above example can also run with the ``block2`` DMRG solver: ::

    import numpy
    from pyscf import gto, scf, mcscf, dmrgscf, lib
    import os

    if not os.path.exists(lib.param.TMPDIR):
        os.mkdir(lib.param.TMPDIR)

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    mc = mcscf.CASSCF(mf, 6, 8)

    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-14)
    mc.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.threads = 28
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    # set very tight thresholds for small system
    mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16]
    mc.fcisolver.scheduleMaxMs = [250, 500, 500, 500, 500]
    mc.fcisolver.scheduleTols = [1e-08, 1e-10, 1e-12, 1e-12, 1e-12]
    mc.fcisolver.scheduleNoises = [0.0001, 0.0001, 5e-05, 5e-05, 0.0]
    mc.fcisolver.maxIter = 30
    mc.fcisolver.twodot_to_onedot = 20
    mc.fcisolver.block_extra_keyword = ['singlet_embedding', 'full_fci_space', 'fp_cps_cutoff 0', 'cutoff 0']

    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-11
    mc.canonicalization = True
    mc.run()

    from pyblock2.icmr.scnevpt2 import WickSCNEVPT2
    wsc = WickSCNEVPT2(mc).run()

    from pyblock2.icmr.icnevpt2_full import WickICNEVPT2
    wic = WickICNEVPT2(mc).run()

.. highlight:: text

This will generate the following output: ::

    $ grep 'E(WickSCNEVPT2)' dmrg-nevpt2.out
    E(WickSCNEVPT2) = -149.9578400627551  E_corr_pt = -0.2491822915198339
    $ grep 'E(WickICNEVPT2)' dmrg-nevpt2.out
    E(WickICNEVPT2) = -149.9601376425396  E_corr_pt = -0.2514798713043632

DMRG-FIC-MRCISD
---------------

.. highlight:: python3

The following is an example python script for fully internally contracted MRCISD with equations derived on the fly
(using the FCI solver): ::

    # need first import numpy (before pyblock2)
    # otherwise the numpy multi-threading may not work
    import numpy

    from pyscf import gto, scf, mcscf
    from pyblock2.icmr.icmrcisd_full import WickICMRCISD

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='6-31g', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-11
    mc.run()

    mol.verbose = 5
    wsc = WickICMRCISD(mc).run()

.. highlight:: text

This will generate the following output: ::

    $ grep 'CASSCF energy' mrci.out 
    CASSCF energy = -149.636563280267
    $ grep 'WickICMRCISD' mrci.out
    E(WickICMRCISD)   = -149.7792742741091  E_corr_ci = -0.1427109938418027
    E(WickICMRCISD+Q) = -149.7858102349944  E_corr_ci = -0.1492469547270254

.. highlight:: python3

Similarly, we can do DMRG-FIC-MRCISD: ::

    # need first import numpy (before pyblock2)
    # otherwise the numpy multi-threading may not work
    import numpy

    from pyscf import gto, scf, mcscf, dmrgscf, lib
    from pyblock2.icmr.icmrcisd_full import WickICMRCISD
    import os

    if not os.path.exists(lib.param.TMPDIR):
        os.mkdir(lib.param.TMPDIR)

    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='6-31g', spin=2, verbose=4)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    mc = mcscf.CASSCF(mf, 6, 8)

    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1E-14)
    mc.fcisolver.runtimeDir = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.scratchDirectory = os.path.abspath(lib.param.TMPDIR)
    mc.fcisolver.threads = 28
    mc.fcisolver.memory = int(mol.max_memory / 1000) # mem in GB

    # set very tight thresholds for small system
    mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16]
    mc.fcisolver.scheduleMaxMs = [250, 500, 500, 500, 500]
    mc.fcisolver.scheduleTols = [1e-08, 1e-10, 1e-12, 1e-12, 1e-12]
    mc.fcisolver.scheduleNoises = [0.0001, 0.0001, 5e-05, 5e-05, 0.0]
    mc.fcisolver.maxIter = 30
    mc.fcisolver.twodot_to_onedot = 20
    mc.fcisolver.block_extra_keyword = ['singlet_embedding', 'full_fci_space', 'fp_cps_cutoff 0', 'cutoff 0']

    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-11
    mc.run()

    mol.verbose = 5
    wsc = WickICMRCISD(mc).run()

.. highlight:: text

This will generate the following output: ::

    $ grep 'CASSCF energy' dmrg-mrci.out 
    CASSCF energy = -149.636563280264
    $ grep 'WickICMRCISD' dmrg-mrci.out
    E(WickICMRCISD)   = -149.7792742857885  E_corr_ci = -0.1427110055241769
    E(WickICMRCISD+Q) = -149.785810250064  E_corr_ci = -0.1492469697996863

.. note ::

    The current FIC-MRCI / DMRG-FIC-MRCI implementation requires the explicit construction of the MRCI Hamiltonian,
    which is not practical for production runs.
