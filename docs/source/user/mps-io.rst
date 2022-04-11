
.. highlight:: bash

.. _user_mps_io:

MPS Import/Export
=================

The ``block2`` MPS can be translated into ``StackBlock`` format for restarting the calculation in ``StackBlock``.
Alternatively, the ``StackBlock`` rotation matrices and wavefunction can be translated into ``block2`` MPS.
Since different initial guess for MPS is generated in ``StackBlock`` and ``block2``, this feature
can be useful for sharing MPS initial guess among different codes, debugging, or performing some DMRG
methods not implemented in one of the code.

The translation itself should be exact, with the support for both spin-adapted and non-spin-adapted case.
If the canonical form is not ``LLL...KR``, some small error may occur during the canonical form translation.

The script ``${BLOCK2HOME}/pyblock2/driver/readwfn.py`` can be used to translate from ``StackBlock`` to ``block2``.
The script ``${BLOCK2HOME}/pyblock2/driver/writewfn.py`` can be used to translate from ``block2`` to ``StackBlock``.
These two scripts depend on ``block2``, ``pyblock`` and ``StackBlock``.
To install ``pyblock`` and ``StackBlock``, the ``boost`` package is required.
We will first explain the installation of these extra dependencies.

Boost Installation
------------------

One can download the most recent version of boost in
`https://www.boost.org/users/download/ <https://www.boost.org/users/download/>`_
Assuming the downloaded file is named ``boost_1_76_0.tar.gz``, stored in ``~/program/boost-1.76``
(you can choose any other directory).
One can then install ``boost`` in the following way. Please make sure the correct version of C++
compiler is set in the environment. The same C++ compiler should be used for compiling ``boost``
and ``block2``, ``pyblock`` and ``StackBlock``. ::

    $ mkdir ~/program/boost-1.76
    $ cd ~/program/boost-1.76
    $ PREFIX=$PWD
    $ wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
    $ tar zxf boost_1_76_0.tar.gz
    $ cd boost_1_76_0
    $ gcc --version
    gcc (GCC) 9.2.0
    $ bash bootstrap.sh
    $ echo 'using mpi ;' >> project-config.jam
    $ ./b2 install --prefix=$PREFIX
    $ echo $PREFIX
    /home/.../program/boost-1.76

Now an environment variable ``BOOSTROOT`` should be added.
This will ensure that this ``boost`` installation can be found by ``cmake``
for compiling ``StackBlock`` and ``pyblock``.
For example, one can add the following line into ``~/.bashrc``. ::

    export BOOSTROOT=~/program/boost-1.76

.. note ::

    The ``--prefix`` parameter cannot be set to a path beginning with ``~``.
    If you need such a path, please use an absolute path instead, namely,
    setting ``--prefix=/home/<user>/...``.

StackBlock Installation
-----------------------

The default vresion of ``StackBlock`` has some bugs for some non-popular features.
It is recommended to use `this fork <https://github.com/hczhai/StackBlock>`_,
which can be compiled using ``cmake``. First, make sure a working MPI library such as ``openmpi 4.0``
can be found in the system, a C++ compiler with the correct version can be found,
and ``BOOSTROOT`` is set. Then ``StackBlock`` can be compiled in the following way
(starting from the cloned ``StackBlock`` repo directory): ::

    export STACKBLOCK=$PWD
    mkdir build
    cd build
    cmake ..
    make -j 10
    rm CMakeCache.txt
    cmake .. -DBUILD_OH=ON
    make -j 10
    rm CMakeCache.txt
    cmake .. -DBUILD_READROT=ON
    make -j 10
    rm CMakeCache.txt
    cmake .. -DBUILD_GAOPT=ON
    make -j 10

This will generate four executables in the build direcotry.

``block.spin_adapted`` is the main ``StackBlock`` program. One can optionally add the following to ``~/.bashrc``: ::

    export PATH=${STACKBLOCK}/build:$PATH

``OH`` is the program to compute the expectation value on an MPS (or between two MPSs),
of Hamiltonian and/or the identity operator.

``read_rot`` is the program to translate the intermediate rotation matrix format
(from the spin-projected ``zmpo_dmrg`` code) to the ``StackBlock`` rotation matrix
and wavefunction format.

``gaopt`` is the program for orbital reordering using genetic algorithm.
Note that for this purpose, the ``block2`` driver ``${BLOCK2HOME}/pyblock2/driver/gaopt.py``
should provide much better performance.

pyblock Installation
--------------------

``pyblock`` is a python3 wrapper for the StackBlock code.
``pyblock`` contains a slightly revised version of StackBlock, which must be compiled.
The code can be obtained `here <https://github.com/hczhai/pyblock>`_.
First, make sure that a C++ compiler with the correct version can be found,
and ``BOOSTROOT`` is set. Then ``pyblock`` can be compiled in the following way
(starting from the cloned ``pyblock`` repo directory): ::

    export PYBLOCKHOME=$PWD
    mkdir build
    cd build
    cmake .. -DBUILD_LIB=ON -DUSE_MKL=ON
    make -j 10

Import MPS to block2
--------------------

Now we are ready to show how to translate a ``StackBlock`` MPS to ``block2`` MPS.

First, make sure a testing integral file ``C2.CAS.PVDZ.FCIDUMP`` is in the working directory.
The integral file can be found in ``${BLOCK2HOME}/data/C2.CAS.PVDZ.FCIDUMP``.

.. note ::

    Normally, orbital reordering can create some unnecessary complexities.
    It is recommended to use a already reordered FCIDUMP file across different codes.
    If the MPS has to be adjusted for orbital reordering, see :ref:`dev_orbital_rotation`.

We will first perform a DMRG ground-state calculation using the following input file ``dmrg.conf``: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30
    prefix ./tmp
    noreorder

The following command can be used to run ``StackBlock`` with this input file: ::

    mkdir ./tmp
    ${STACKBLOCK}/build/block.spin_adapted dmrg.conf > dmrg.out

The DMRG ground-state energy can be obtained from the output file: ::

    $ grep 'Sweep Energy' dmrg.out | tail -1
    M = 500     state = 0     Largest Discarded Weight = 0.000000000000  Sweep Energy = -75.728442606745

The energy for the MPS that will be translated is the energy at the last site of the last sweep: ::

    $ grep 'sweep energy' dmrg.out | tail -1
    Finished Sweep with 500 states and sweep energy for State [ 0 ] with Spin [ 0 ] :: -75.728442606745

Since in the default schedule the one-site algorithm is used for the last sweep. This two energies are identical.

Now the MPS in ``StackBlock`` format is stored in the scratch folder ``./tmp/node0``.
We will only need files in this folder with file names ``Rotation-*``, ``StateInfo-*``, ``wave-*``.
The other files ``Block-b-*`` and ``Block-f-*`` (with renormalized operators stored)
are not part of the MPS, which can be deleted.

The folowing commands can be used to translate the MPS.
Please make sure that the environment variables ``${STACKBLOCK}``, ``${PYBLOCKHOME}``, and ``${BLOCK2HOME}``
are correctly set. ::

    $ PYTHONPATH=${BLOCK2HOME}/build:$PYTHONPATH
    $ PYTHONPATH=${PYBLOCKHOME}:$PYTHONPATH
    $ PYTHONPATH=${PYBLOCKHOME}/build:$PYTHONPATH
    $ READWFN=${BLOCK2HOME}/pyblock2/driver/readwfn.py
    $ python3 $READWFN dmrg.conf -expect
    -75.72844260674495

.. note ::
    Here we use a special build of ``block2`` python extension,
    which was built using the ``cmake`` option ``-DTBB=OFF`` (the default is ``OFF``).
    On some systems ``-DUSE_MKL=OFF -OMP_LIB=SEQ`` may be required.
    This is to solve the conflicts for importing ``pyblock`` and ``block2`` in the same script.

Note that ``-expect`` option is optional. With this option, the energy of the translated MPS
will be evaluted in ``block2`` and printed.
We can see that the printed ``block2`` energy is almost exactly the same as the one obtained from
``StackBlock``.
By default, the translated ``block2`` MPS will be put in the output directory named
``./out`` with the tag ``KET``.

Export MPS from block2
----------------------

Now we show how to translate a ``block2`` MPS to ``StackBlock`` MPS.

We will first perform a DMRG ground-state calculation using the following input file ``dmrg2.conf``: ::

    sym d2h
    orbitals C2.CAS.PVDZ.FCIDUMP

    nelec 8
    spin 0
    irrep 1

    hf_occ integral
    schedule default
    maxM 500
    maxiter 30
    prefix ./tmp2
    noreorder

Note that the only difference between ``dmrg.conf`` and ``dmrg2.conf`` is the ``prefix``.
The following command can be used to run ``block2`` with this input file: ::

    ${BLOCK2HOME}/pyblock2/driver/block2main dmrg2.conf > dmrg2.out

The energy for the MPS that will be translated is the energy at the last site of the last sweep: ::

    $ grep 'DW' dmrg2.out | tail -1
    Time elapsed =      3.883 | E =     -75.7284436933 | DE = -3.85e-07 | DW = 3.76e-16

The folowing commands can be used to translate the MPS.
Please make sure that the environment variables ``${STACKBLOCK}``, ``${PYBLOCKHOME}``, and ``${BLOCK2HOME}``
are correctly set. ::

    $ PYTHONPATH=${BLOCK2HOME}/build:$PYTHONPATH
    $ PYTHONPATH=${PYBLOCKHOME}:$PYTHONPATH
    $ PYTHONPATH=${PYBLOCKHOME}/build:$PYTHONPATH
    $ WRITEWFN=${BLOCK2HOME}/pyblock2/driver/writewfn.py
    $ python3 $WRITEWFN dmrg2.conf -out out2
    load MPSInfo from ././tmp2/KET-mps_info.bin
    SRRRRRRRRRRRRRRRRRRRRRRRRR -> LLLLLLLLLLLLLLLLLLLLLLLLKR 24

From the print we can see that the canonical form of MPS has been changed,
which may cause some small error in the translated MPS.
The translated MPS in ``StackBlock`` format is now stored in the ``out2`` directory.
We can now evaluate the energy of the translated MPS using the ``OH`` program in ``StackBlock``: ::

    $ sed -i "s|^prefix.*|prefix ./out2|" dmrg2.conf
    $ ${STACKBLOCK}/build/OH dmrg2.conf | grep -A 1 'printing hamiltonian' | tail -1
    -75.7284436933

We can see that the printed ``StackBlock`` energy is exactly the same as the one obtained from
``block2``.

.. note ::

    The ``OH`` program in ``StackBlock`` can only evalute the ``onedot`` MPS
    (namely, MPS used in 1-site DMRG algorithm).
    The MPS can be spin-adapted or non-spin-adapted.
    If you use the ``OH`` in the default standard version of ``StackBlock``,
    the non-spin-adapted MPS is not supported and you need an extra argument
    for a file including the MPS ids. For example, you should use
    ``/path/to/default/StackBlock/OH dmrg2.conf wavenum`` where a file named
    ``wavenum`` should be set with contents ``0``
    (or any space-separated list of integers, if you have multiple MPSs).

Alternatively, we can also translate back to ``block2`` and evaluate the energy: ::

    $ sed -i "s|^prefix.*|prefix ./out2|" dmrg2.conf
    $ READWFN=${BLOCK2HOME}/pyblock2/driver/readwfn.py
    $ python3 $READWFN dmrg2.conf -dot 1 -expect -out out3
    -75.72844369332921

Which also prints the same energy.
