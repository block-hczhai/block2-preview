
.. _dev_mpo_reloading:

MPO Reloading
=============

For systems with large number of orbitals, it is sometimes beneficial to save/reload the MPO object
to reduce memory fragmentation. The step of creation of the Hamiltonian and FCIDUMP
object can also be done only once and all Hamiltonian information can be kept in the MPO object,
which can be saved in disk storage. This can save computational cost (if creation of
the Hamiltonian/MPO is expensive) and memory cost (if the FCIDUMP object is big) for restarting.

For even larger number of orbitals, keeping the whole MPO object during the DMRG calculation
may still be memory-demanding. To solve this problem, the MPO can be reloaded in a minimal memory mode.
In this mode, only the essential data in MPO is loaded in the beginning.
Then, during the DMRG calculation, blocking formulae and definition of single-site operators will
be loaded for each site only. After the iteration for one site, the memory consumed by the
blocking formulae and single-site operators can be released.
Therefore, even if the MPO object itself can be big, only a small part (for each current site)
is loaded into memory (dynamically) at any instant during the DMRG calculation.

Limitations:

* If an MPO is loaded in the minimal memory mode, the MPO file must be kept in the file system
  (namely, not deleted / overwritten) during any subsequent algorithms using this MPO.
* If an MPO is loaded in the minimal memory mode, it is read-only. This means, you cannot simplify
  or parallelize such an MPO. As a result, if you use distributed parallelism,
  you have to save the already parallelized MPO (for each rank as separate files), and reload
  them in the minimal memory mode.
* Inspecting some site-related contents inside the minimal-memory MPO can be
  more complicated (requiring ``mpo.load_*`` before the operation) since these contents are not
  in memory by default.

Example
-------

The example integral file ``C2.CAS.PVDZ.FCIDUMP`` can be found in the ``data`` folder.

.. highlight:: python3

Saving a Serial MPO
^^^^^^^^^^^^^^^^^^^

First we save a non-parallelized MPO using the following script: ::

    from block2 import *
    from block2.su2 import *
    import numpy as np
    import psutil
    import os
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
    print("ORB SYM = ", hamil.orb_sym)

    mem = psutil.Process(os.getpid()).memory_info().rss
    print(" pre-mpo memory usage = %10s" % Parsing.to_size_string(mem))

    # MPO
    mpo = MPOQC(hamil, QCTypes.Conventional)
    mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    mpo.basis = hamil.basis

    mem = psutil.Process(os.getpid()).memory_info().rss
    print("post-mpo memory usage = %10s" % Parsing.to_size_string(mem))

    mpo.reduce_data()
    mpo.save_data('mpo.bin')

    fsize = os.path.getsize('mpo.bin')
    print("mpo size = %10s" % Parsing.to_size_string(fsize))

.. highlight:: text

Some reference outputs (the memory information can be different for each run): ::

    $ grep 'usage\|size' dmrg-1.out
     pre-mpo memory usage =    58.5 MB
    post-mpo memory usage =     126 MB
    mpo size =    2.35 MB

So without saving and reloading the MPO, the MPO object needs roughly 67.5 MB memory.

.. highlight:: python3

Loading a Serial MPO
^^^^^^^^^^^^^^^^^^^^

We can now load the saved ``mpo.bin`` to do DMRG, and skip the step for creating ``HamiltonianQC``
and ``FCIDUMP``: ::

    from block2 import *
    from block2.su2 import *
    import numpy as np
    import psutil
    import os
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

    mem = psutil.Process(os.getpid()).memory_info().rss
    print(" pre-load-mpo memory usage = %10s" % Parsing.to_size_string(mem))

    mpo = MPO(0)
    mpo.load_data('mpo.bin')

    mem = psutil.Process(os.getpid()).memory_info().rss
    print("post-load-mpo memory usage = %10s" % Parsing.to_size_string(mem))

    n_sites = mpo.n_sites
    vacuum = SX(0)
    target = SX(8, 0, 0)

    mps_info = MPSInfo(mpo.n_sites, vacuum, target, mpo.basis)
    mps_info.tag = 'KET'
    mps_info.set_bond_dimension(250)
    mps = MPS(n_sites, 0, 2)
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
    print('DMRG Energy = %20.15f' % ener)

.. highlight:: text

Some reference outputs (the memory information can be different for each run): ::

    $ grep 'usage\|Energy' dmrg-2.out
     pre-load-mpo memory usage =    42.6 MB
    post-load-mpo memory usage =    53.5 MB
    DMRG Energy =  -75.728475321395166

So the reloaded MPO object is smaller, which needs only 10.9 MB memory. The DMRG takes 70.581 seconds.

.. highlight:: python3

Loading a Serial MPO with Minimal Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can change the line in the above script: ::

    mpo.load_data('mpo.bin')

to: ::

    mpo.load_data('mpo.bin', minimal=True)

Then rerun the script. Now the MPO is loaded in the minimal memory mode.

.. highlight:: text

Some reference outputs (the memory information can be different for each run): ::

    $ grep 'usage\|Energy' dmrg-2.out
     pre-load-mpo memory usage =    40.7 MB
    post-load-mpo memory usage =    43.0 MB
    DMRG Energy =  -75.728475329694518

Now the reloaded MPO object occupies only 2.3 MB memory before the DMRG calculation.
The DMRG takes 70.688 seconds (which is not greatly affected by dynamically reloading MPO parts).

.. highlight:: python3

Saving Parallelized MPO
^^^^^^^^^^^^^^^^^^^^^^^

For distributed calculations, we can still reload the serial MPO and parallelize it.
But this way is only compatible to the non-minimal-memory mode.
To save the memory for distributed calculations, we need to save the parallelized MPO.
The parallelization script for MPO does not have to be run in parallel (but you still can run
it in parallel, which has a lower wall time cost but a higher memory cost).

The following script generates and saves the parallelized MPO for 7 mpi processsors
(note that this script should be run in serial, namely, no ``mpirun``): ::

    from block2 import *
    from block2.su2 import *
    import numpy as np
    import psutil
    import os

    Global.frame = DoubleDataFrame(10 * 1024 ** 2, 10 * 1024 ** 3, "nodex")

    mpo = MPO(0)
    mpo.load_data('mpo.bin')

    # size, rank, root
    comm = ParallelCommunicator(7, 0, 0)
    prule = ParallelRuleQC(comm)

    for irank in range(comm.size):
        comm.rank = irank
        para_mpo = ParallelMPO(mpo, prule)
        para_mpo.save_data('mpo.bin.%d' % irank)
        fsize = os.path.getsize('mpo.bin.%d' % irank)
        print("mpo.%d size = %10s" % (irank, Parsing.to_size_string(fsize)))

Here we assume a serial MPO ``mpo.bin`` has already been saved in the disk.
The ``ParallelCommunicator`` is a fake object for distributed parallelism.
We can manually change the ``rank`` of ``ParallelCommunicator`` to generate
parallelized MPOs for different ranks.

.. highlight:: text

Some reference outputs: ::

    mpo.0 size =    2.74 MB
    mpo.1 size =    2.75 MB
    mpo.2 size =    2.73 MB
    mpo.3 size =    2.74 MB
    mpo.4 size =    2.77 MB
    mpo.5 size =    2.78 MB
    mpo.6 size =    2.77 MB

Note that each parallelized MPO is larger than the serial MPO. Actually,
each of them includes both the "local" part and "global" part.
The "global" part then has the same size as the serial MPO.
(For big site code the "global" part for parallelized MPO can be smaller than
the full MPO).

.. highlight:: python3

Reloading Parallelized MPO
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following script is used for parallel DMRG with 7 mpi processsors
(namely, ``mpirun -n 7 --bind-to none python -u dmrg.py``, for example): ::

    from block2 import *
    from block2.su2 import *
    import numpy as np
    import psutil
    import os
    SX = SU2

    MPI = MPICommunicator()

    Global.frame = DoubleDataFrame(10 * 1024 ** 2, 10 * 1024 ** 3, "nodex")
    n_threads = Global.threading.n_threads_global // MPI.size
    Global.threading = Threading(
        ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
        n_threads, n_threads, 1)
    Global.threading.seq_type = SeqTypes.Tasked
    Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
    Global.frame.minimal_disk_usage = True
    Global.frame.use_main_stack = False
    print(Global.frame)
    print(Global.threading)

    prule = ParallelRuleQC(MPI)

    mem = psutil.Process(os.getpid()).memory_info().rss
    print(" pre-load-mpo memory usage = %10s" % Parsing.to_size_string(mem))

    mpo = ParallelMPO(0, prule)
    mpo.load_data('mpo.bin.%d' % MPI.rank, minimal=False)

    mem = psutil.Process(os.getpid()).memory_info().rss
    print("post-load-mpo memory usage = %10s" % Parsing.to_size_string(mem))

    n_sites = mpo.n_sites
    vacuum = SX(0)
    target = SX(8, 0, 0)

    mps_info = MPSInfo(mpo.n_sites, vacuum, target, mpo.basis)
    mps_info.tag = 'KET'
    mps_info.set_bond_dimension(250)
    mps = MPS(n_sites, 0, 2)
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
    print('DMRG Energy = %20.15f' % ener)

.. highlight:: text

Some reference outputs (the memory information can be different for each run): ::

    $ grep 'post-\|Energy' dmrg-3.out
    post-load-mpo memory usage =    59.6 MB
    post-load-mpo memory usage =    61.6 MB
    post-load-mpo memory usage =    59.4 MB
    post-load-mpo memory usage =    63.6 MB
    post-load-mpo memory usage =    59.4 MB
    post-load-mpo memory usage =    59.4 MB
    post-load-mpo memory usage =    59.4 MB
    DMRG Energy =  -75.728475146585453
    DMRG Energy =  -75.728475146585453
    DMRG Energy =  -75.728475146585453
    DMRG Energy =  -75.728475146585453
    DMRG Energy =  -75.728475146585453
    DMRG Energy =  -75.728475146585453
    DMRG Energy =  -75.728475146585453

.. highlight:: python3

Reloading Parallelized MPO with Minimal Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can change the line in the above script: ::

    mpo.load_data('mpo.bin.%d' % MPI.rank, minimal=False)

to: ::

    mpo.load_data('mpo.bin.%d' % MPI.rank, minimal=True)

Then rerun the script. Now the MPO is loaded in the minimal memory mode.

.. highlight:: text

Some reference outputs (the memory information can be different for each run): ::

    $ grep 'post-\|Energy' dmrg-3.out
    post-load-mpo memory usage =    52.8 MB
    post-load-mpo memory usage =    48.8 MB
    post-load-mpo memory usage =    50.8 MB
    post-load-mpo memory usage =    50.8 MB
    post-load-mpo memory usage =    52.8 MB
    post-load-mpo memory usage =    48.9 MB
    post-load-mpo memory usage =    48.8 MB
    DMRG Energy =  -75.728475151371001
    DMRG Energy =  -75.728475151371001
    DMRG Energy =  -75.728475151371001
    DMRG Energy =  -75.728475151371001
    DMRG Energy =  -75.728475151371001
    DMRG Energy =  -75.728475151371001
    DMRG Energy =  -75.728475151371001

We can see that the memory usage after loading MPO is smaller,
compared to the non-minimal-memory-usage mode.
