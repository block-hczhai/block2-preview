
.. highlight:: bash

.. _user_open_molcas:

DMRGSCF (forte)
====================

In this section we explain how to use ``block2`` and ``forte`` for CASSCF and DSRG calculations with DMRG as the active space solver.

``forte`` is an open-source package for strongly correlated methods, developed by Evangelista group.
The detailed instruction for the installation and the usage of the code can be found in
https://forte.readthedocs.io/.

Preparation
-----------

First, we need to build and install the C++ library of ``block2``. This can be done using the ``-DBUILD_CLIB=ON`` option: ::

    git clone https://github.com/block-hczhai/block2-preview
    cd block2-preview
    mkdir build
    cd build
    cmake .. -DUSE_MKL=ON -DBUILD_CLIB=ON -DLARGE_BOND=ON -DMPI=OFF -DCMAKE_INSTALL_PREFIX=../install
    make -j 10
    make install
    export BLOCK2_DIR=$PWD/../install
    cd ../..

After this, you will be able to find the block2 include files in ``${BLOCK2_DIR}/include/`` and ``libblock2.so`` in ``${BLOCK2_DIR}/lib64/``.
The ``block2Config.cmake`` file can be found in ``${BLOCK2_DIR}/share/cmake/block2/``.

Second, we need to build `psi4 <https://github.com/psi4/psi4>`_. Make sure an ``eigen3`` library is available in the system,
which can be installed using ``apt install libeigen3-dev`` or ``conda install -c omnia eigen3``. If you use the ``conda`` package,
you may need to add the ``cmake`` option ``-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}`` so that ``cmake`` can find it.

Then we can build ``psi4`` as follows: ::

    git clone https://github.com/psi4/psi4
    cd psi4
    mkdir build
    cd build
    export MATH_ROOT=${CONDA_PREFIX}
    cmake .. -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}
    make -j 10
    export PSI4_DIR=$PWD/stage
    cd ../..

Then we can add the following environment variables: ::

    export PATH=${PSI4_DIR}/bin:$PATH
    export PYTHONPATH=${PSI4_DIR}/lib:$PYTHONPATH
    export PSI_SCRATCH=/scratch/.../psi4      # use a valid scratch folder here

Then the command ``psi4`` should be available in the terminal. And ``python -c 'import psi4'`` should work.

Third, we need to build `abmit <https://github.com/jturney/ambit>`_ as follows: ::

    git clone https://github.com/jturney/ambit
    cd ambit
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    make -j 10
    make install
    export AMBIT_DIR=$PWD/../install
    cd ../..

Finally, we can build `forte <https://github.com/evangelistalab/forte>`_.
Here we use a revised version with the ``block2`` interface,
which can be found in the ``block2_dmrg`` branch of the forked repo https://github.com/hczhai/forte. To build it, we can: ::

    git clone https://github.com/hczhai/forte
    cd forte
    git checkout block2_dmrg
    $(psi4 --plugin-compile) -Dambit_DIR=${AMBIT_DIR}/share/cmake/ambit \
        -DENABLE_block2=ON \
        -Dblock2_DIR=${AMBIT_DIR}/share/cmake/block2 \
        -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}
    make -j 10
    export FORTE_DIR=$PWD
    cd ..

Then we can add the following environment variables: ::

    export PYTHONPATH=${FORTE_DIR}:$PYTHONPATH

Then ``python -c 'import forte'`` should work.

DMRG
----

.. highlight:: python3

The following is an example python script for DMRG for N2 in the minimal basis set: ::

    import psi4
    import forte

    psi4.geometry("""
    0 1
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """)

    psi4.set_options(
        {
            'basis': 'sto-3g',
            'scf_type': 'pk',
            'e_convergence': 14,
            'reference': 'rhf',
            'forte__active_space_solver': 'block2',
            'forte__block2_sweep_davidson_tols': [1E-15],
        } 
    )

    psi4.energy('forte')

.. highlight:: text

This will generate the following output: ::

    $ grep 'Energy Summary' -A 4 dmrg.out | tail -1
    1  (  0)    Ag     0     -107.654122447812   0.000000

DMRGSCF
-------

.. highlight:: python3

The following is an example python script for DMRGSCF for an O2 triplet state (see :ref:`user_dmrgscf` for the similar calculation using ``pyscf``): ::

    import psi4
    import forte

    psi4.geometry("""
    0 3
    O 0.0 0.0 -0.6035
    O 0.0 0.0 0.6035
    """)

    psi4.set_options(
        {
            'basis': 'cc-pvdz',
            'scf_type': 'direct',
            'e_convergence': 20,
            'reference': 'rohf',
            'forte__job_type': 'casscf',
            'forte__casscf_ci_solver': 'block2',
            'forte__block2_sweep_davidson_tols': [1E-15],
            'forte__restricted_docc': [2, 0, 0, 0, 0, 2, 0, 0],
            'forte__active': [1, 0, 1, 1, 0, 1, 1, 1],
            'forte__root_sym': 1, # B1g
        } 
    )

    psi4.energy('forte')

.. highlight:: text

This will generate the following output: ::

    $ grep 'Energy Summary' -A 4 dmrg.out | grep B1g
    3  (  0)   B1g     0     -149.671533509344   2.000000
    3  (  0)   B1g     0     -149.689293451723   2.000000
    3  (  0)   B1g     0     -149.703603100002   2.000000
    3  (  0)   B1g     0     -149.708080545113   2.000000
    3  (  0)   B1g     0     -149.708521258412   2.000000
    3  (  0)   B1g     0     -149.708617815460   2.000000
    3  (  0)   B1g     0     -149.708645817441   2.000000
    3  (  0)   B1g     0     -149.708654215054   2.000000
    3  (  0)   B1g     0     -149.708656716926   2.000000
    3  (  0)   B1g     0     -149.708657458784   2.000000
    3  (  0)   B1g     0     -149.708657678545   2.000000
    3  (  0)   B1g     0     -149.708657743713   2.000000
    3  (  0)   B1g     0     -149.708657763065   2.000000
    3  (  0)   B1g     0     -149.708657768818   2.000000
    3  (  0)   B1g     0     -149.708657770529   2.000000
    3  (  0)   B1g     0     -149.708657771038   2.000000
    3  (  0)   B1g     0     -149.708657771254   2.000000

DMRG-DSRG
---------

.. highlight:: python3

The following is an example python script for DMRG-DSRG for an O2 triplet state, using the DMRGSCF state as the reference state: ::

    import psi4
    import forte

    psi4.geometry("""
    0 3
    O 0.0 0.0 -0.6035
    O 0.0 0.0 0.6035
    """)

    psi4.set_options(
        {
            'basis': 'cc-pvdz',
            'scf_type': 'direct',
            'e_convergence': 20,
            'reference': 'rohf',
            'forte__job_type': 'casscf',
            'forte__casscf_ci_solver': 'block2',
            'forte__block2_sweep_davidson_tols': [1E-15],
            'forte__restricted_docc': [2, 0, 0, 0, 0, 2, 0, 0],
            'forte__active': [1, 0, 1, 1, 0, 1, 1, 1],
            'forte__root_sym': 1, # B1g
        } 
    )

    e, wfn = psi4.energy('forte', return_wfn=True)

    psi4.set_options(
        {
            'forte__job_type': 'newdriver',
            'forte__active_space_solver': 'block2',
            'forte__correlation_solver': 'sa-mrdsrg',
            'forte__dsrg_s': 0.5,
        } 
    )

    psi4.energy('forte', ref_wfn=wfn)

.. highlight:: text

This will generate the following output: ::

    $ grep 'E0 (reference)' dsrg.out
    E0 (reference)                 =   -149.708657771253996
    $ grep 'DSRG-MRPT2 correlation' -A 1 dsrg.out
    DSRG-MRPT2 correlation energy  =     -0.263404857500777
    DSRG-MRPT2 total energy        =   -149.972062628754770

State-Average
-------------

.. highlight:: python3

The following is an example python script for state-averaged DMRGSCF for three states: ::

    import psi4
    import forte

    psi4.geometry("""
    0 3
    O 0.0 0.0 -0.6035
    O 0.0 0.0 0.6035
    """)

    psi4.set_options(
        {
            'basis': 'cc-pvdz',
            'scf_type': 'direct',
            'e_convergence': 20,
            'reference': 'rohf',
            'forte__job_type': 'casscf',
            'forte__casscf_ci_solver': 'block2',
            'forte__block2_sweep_davidson_tols': [1E-15],
            'forte__restricted_docc': [2, 0, 0, 0, 0, 2, 0, 0],
            'forte__active': [1, 0, 1, 1, 0, 1, 1, 1],
            'forte__avg_state': [[1, 3, 3]], # (B1g, triplet, 3 states)
        } 
    )

    psi4.energy('forte')

.. highlight:: text

This will generate the following output: ::

    $ grep '==> Energy Summary <==' -A 6 03.out | tail -3
    3  (  0)   B1g     0     -149.690635774964   2.000000
    3  (  0)   B1g     1     -149.093708503131   2.000000
    3  (  0)   B1g     2     -148.861580599165   2.000000

.. highlight:: python3

.. note ::

    For realistic calculations one should not rely on the default settings for the DMRG schedule.
    Customized schedule can be set using for example: ::

        'forte__block2_sweep_n_sweeps': [4, 4, 4, 6],
        'forte__block2_sweep_bond_dims': [250, 500, 1000, 1000],
        'forte__block2_sweep_noises': [1E-4, 1E-5, 1E-5, 0],
        'forte__block2_sweep_davidson_tols': [1E-5, 1E-7, 1E-7, 1E-9],
        'forte__block2_energy_convergence': 1E-8,
        'forte__block2_n_total_sweeps': 18,
        'forte__block2_verbose': 2
