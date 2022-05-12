
.. highlight:: bash

DMRGSCF
=======

In this section we explain how to use ``block2`` (and optionally ``StackBlock``) and ``pyscf`` for ``DMRGSCF`` (CASSCF with DMRG as the active space solver).

Preparation
-----------

One should first install the pyscf extension called ``dmrgscf``, which can be obtained from
`https://github.com/pyscf/dmrgscf <https://github.com/pyscf/dmrgscf>`_.
If it is installed using ``pip``, one also needs to create a file named ``settings.py`` under the ``dmrgscf`` folder, as follows: ::

    $ pip install git+https://github.com/pyscf/dmrgscf
    $ PYSCFHOME=$(pip show pyscf-dmrgscf | grep 'Location' | tr ' ' '\n' | tail -n 1)
    $ wget https://raw.githubusercontent.com/pyscf/dmrgscf/master/pyscf/dmrgscf/settings.py.example
    $ mv settings.py.example ${PYSCFHOME}/pyscf/dmrgscf/settings.py

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
    in UCASSCF/UCASCI. ``pyscf/avas`` does not support creating active space with unrestricted orbtials
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
