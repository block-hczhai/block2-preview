
Notes
=====

.. highlight:: text

Build block2 in ``manylinux2010`` docker image
----------------------------------------------

The docker image named ``quay.io/pypa/manylinux2010_x86_64`` is used.

First we need to select one python version: ::

    export PATHBAK=$PATH
    export PATH=/opt/python/cp37-cp37m/bin:$PATHBAK
    export PATH=/opt/python/cp38-cp38/bin:$PATHBAK
    export PATH=/opt/python/cp39-cp39/bin:$PATHBAK
    export PATH=/opt/python/cp310-cp310/bin:$PATHBAK
    which python3

Clone the block2 repo: ::

    git clone https://github.com/block-hczhai/block2

Edit the ``setup.py``: ::

    '-DPYTHON_EXECUTABLE={}'.format('/opt/python/cp37-cp37m/bin/python3'),

Instal dependencies and build: ::

    python3 -m pip install pip build twine --upgrade
    python3 -m pip install mkl==2019 mkl-include intel-openmp numpy cmake==3.17 pybind11
    python3 -m build

Change linux tag and upload: ::

    mv dist/block2-0.1.10-cp38-cp38-linux_x86_64.whl dist/block2-0.1.10-cp38-cp38-manylinux2010_x86_64.whl
    python3 -m twine upload dist/block2-0.1.10-cp38-cp38-manylinux2010_x86_64.whl

Installing ``block2`` using python virtual environment
------------------------------------------------------

This guide shows how one can manually build ``block2`` with openmpi without using Anaconda and Intel oneapi.

First we assume a suitable python3, openmpi library, and gcc compiler can be found in the system. For example, I have ::

    $ which gcc
    /opt/gcc/11.2.0/bin/gcc
    $ which mpirun
    /opt/openmpi/4.1.2/gnu/bin/mpirun
    $ which python3

First we install the python virtualenv ::

    $ python3 -m pip install --user --upgrade pip
    $ python3 -m pip install --user virtualenv

Then we create a virtualenv called ``base`` and activate it, which will create a folder called ``base`` in the current folder ::

    $ python3 -m venv base
    $ source base/bin/activate

Then we install necessary packages in this virtualenv ::

    $ pip install mkl mkl-include pybind11 numpy scipy psutil
    $ pip install mpi4py --no-binary mpi4py

Then we can build block as the following ::

    $ mkdir build
    $ cd build
    $ export MKLROOT=/path/to/base
    $ export PATH=/path/to/base:$PATH
    $ cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DMPI=ON -DLARGE_BOND=ON -DTBB=ON -DUSE_DMRG=ON -DUSE_BIG_SITE=ON -DUSE_SP_DMRG=ON -DUSE_IC=ON -DUSE_KSYMM=ON -DUSE_COMPLEX=ON

We can test that ``mpi4py`` is working ::

    $ mpirun -n 2 python -c 'from mpi4py import MPI;print(MPI.COMM_WORLD.rank)'
