
.. _user_installation:

Installation
============

.. highlight:: bash

Using ``pip``
-------------

One can install ``block2`` using ``pip``:

* OpenMP-only version (no MPI dependence) ::

      pip install block2

* Hybrid openMP/MPI version (requiring openMPI 4.0.x) ::

      pip install block2-mpi

* Binary format are prepared via ``pip`` for python 3.7, 3.8, 3.9, and 3.10 with macOS (x86 and arm64, no-MPI) or Linux (no-MPI/openMPI).
  If these binaries have some problems, you can use the ``--no-binary`` option of ``pip`` to force building from source
  (for example, ``pip install block2 --no-binary block2``).

* One should only install one of ``block2`` and ``block2-mpi``. ``block2-mpi`` covers all features in ``block2``,
  but its dependence on mpi library can sometimes be difficult to deal with.
  Some guidance for resolving environment problems can be found in github issue
  `#7 <https://github.com/block-hczhai/block2-preview/issues/7>`_.

* To install the most recent development version, use: ::

      pip install block2 --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/
      pip install block2-mpi --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/

Manual Installation
-------------------

Dependence: ``pybind11``, ``python3``, and ``mkl`` (or ``blas + lapack``).

For distributed parallel calculation, ``mpi`` library is required.

``cmake`` (version >= 3.0) can be used to compile C++ part of the code, as follows ::

    mkdir build
    cd build
    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DLARGE_BOND=ON -DMPI=ON
    make -j 10

Which will build the python extension library.

You may need to add the ``build`` directory to your environment ::

    export PYTHONPATH=/path/to/block2/build:${PYTHONPATH}

Options
-------

MKL
^^^

If ``-DUSE_MKL=ON`` is not given, ``blas`` and ``lapack`` are required. Sometimes, the ``blas`` and ``lapack`` function names can contain the extra underscore.
Therefore, it is recommended to use ``-DUSE_MKL=OFF`` and ``-DF77UNDERSCORE=ON`` together to prevent this underscore problem.
If this generates the undefined reference error, one should try ``-DUSE_MKL=OFF -DF77UNDERSCORE=OFF`` instead.

Use ``-DUSE_MKL64=ON`` instead of ``-DUSE_MKL=ON`` to enable using matrices with 64-bit integer type.

Serial compilation
^^^^^^^^^^^^^^^^^^

By default, the C++ templates will be explicitly instantiated in different compilation units, so that parallel
compilation is possible.

Alternatively, one can do single-file compilation using ``-DEXP_TMPL=NONE``, then total compilation time can be
saved by avoiding unnecessary template instantiation, as follows ::

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DEXP_TMPL=NONE
    make -j 1

This may take 5 minutes, need 7 to 10 GB memory.

MPI version
^^^^^^^^^^^

Adding option ``-DMPI=ON`` will build MPI parallel version. The C++ compiler and MPI library must be matched.
If necessary, environment variables ``CC``, ``CXX``, and ``MPIHOME`` can be used to explicitly set the path.

For mixed ``openMP/MPI``, use ``mpirun --bind-to none -n ...`` or ``mpirun --bind-to core --map-by ppr:$NPROC:node:pe=$NOMPT ...`` to execute binary.

Binary build
^^^^^^^^^^^^

To build unit tests and binary executable (instead of python extension), use the following ::

    cmake .. -DUSE_MKL=ON -DBUILD_TEST=ON

TBB (Intel Threading Building Blocks)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding (optional) option ``-DTBB=ON`` will utilize ``malloc`` from ``tbbmalloc``.
This can improve multi-threading performance.

openMP
^^^^^^

If gnu openMP library ``libgomp`` is not available, one can use intel openMP library.

The following will switch to intel openMP library (incompatible with ``-fopenmp``) ::

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=INTEL

The following will use sequential mkl library ::

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=SEQ

The following will use tbb mkl library ::

    cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DOMP_LIB=TBB -DTBB=ON

.. note::

    (For developers. ) For ``CSR sparse MKL + ThreadingTypes::Operator``, if ``-DOMP_LIB=GNU``,
    it is not possible to set both ``n_threads_mkl`` not equal to 1 and ``n_threads_op`` not equal to 1.
    In other words, nested openMP is not possible for CSR sparse matrix (generating wrong result/non-convergence).
    For ``-DOMP_LIB=SEQ``, CSR sparse matrix is okay (non-nested openMP).
    For ``-DOMP_LIB=TBB``, nested openMP + TBB MKL is okay.

``-DTBB=ON`` can be combined with any ``-DOMP_LIB=...``.

Complex mode
^^^^^^^^^^^^

For complex integrals / spin-orbit coupling (SOC), extra options ``-DUSE_COMPLEX=ON`` and ``-DUSE_SG=ON`` are required (and the compilation time will increase).

Maximal bond dimension
^^^^^^^^^^^^^^^^^^^^^^

The default maximal allowed bond dimension per symmetry block is ``65535``.
Adding option ``-DSMALL_BOND=ON`` will change this value to ``255``.
Adding option ``-DLARGE_BOND=ON`` will change this value to ``4294967295``.

Release build
^^^^^^^^^^^^^

The release mode is controlled by CMAKE_BUILD_TYPE.

The following option will use optimization flags such as -O3 (default) ::

    cmake .. -DCMAKE_BUILD_TYPE=Release

The following enables debug flags ::

    cmake .. -DCMAKE_BUILD_TYPE=Debug

Installation with ``anaconda``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An incorrectly installed ``mpi4py`` may produce this error: ::

    undefined symbol: ompi_mpi_logical8

when you execute ``from mpi4py import MPI`` in a ``python`` interpreter.

When using ``anaconda``, please make sure that ``mpi4py`` is linked with the same ``mpi`` library as the one used for compiling ``block2``.
We can create an ``anaconda`` virtual environment (optional): ::

    conda create -n block2 python=3.8 anaconda
    conda activate block2

Then make sure that a working ``mpi`` library is in the environment, using, for example: ::

    module load openmpi/4.0.4
    module load gcc/9.2.0

Then we should install ``mpi4py`` using this ``mpi`` library via ``--no-binary`` option of ``pip``: ::

    python -m pip install --no-binary :all: mpi4py

Sometimes, the above procedure may still give the ``undefined symbol: ompi_mpi_logical8`` error.
Then it is possible that the ``mpi4py`` is still linked to the ``mpich`` (version 3 or lower) library installed in ``anaconda``.
If this is the case, one should first ``conda uninstall mpich`` and then ``python -m pip -v install --no-binary :all: mpi4py``
and if the installation is successful, we can ``ldd $(python -c 'from mpi4py import MPI;print(MPI.__file__)')``
to check the linkage of the ``libmpi.so``. Ideally it should points to the ``openmpi/4.0.4`` library or any other version 4.0 mpi
library. Alternatively, if you do not want to uninstall the ``mpich`` in ``anaconda``, you may install ``block2`` from source using
the same ``mpich`` library.

Supported operating systems and compilers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Linux + gcc 9.2.0 + MKL 2021.4
* MacOS 10.15 + Apple clang 12.0 + MKL 2021
* MacOS 10.15 + icpc 2021.1 + MKL 2021
* Windows 10 + Visual Studio 2019 (MSVC 14.28) + MKL 2021

Using ``block2`` together with other python extensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, when you have to use ``block2`` together with other python modules (such as ``pyscf`` or ``pyblock``),
it may have some problem coexisting with each other.
In general, change the import order may help.
For ``pyscf``, ``import block2`` at the very beginning of the script may help.
For ``pyblock``, recompiling ``block2`` use ``cmake .. -DUSE_MKL=OFF -DBUILD_LIB=ON -OMP_LIB=SEQ -DLARGE_BOND=ON`` may help.

Using C++ Interpreter cling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since ``block2`` is designed as a header-only C++ library, it can be conveniently executed
using C++ interpreter `cling <https://github.com/root-project/cling>`_
(which can be installed via `anaconda <https://anaconda.org/conda-forge/cling>`_)
without any compilation. This can be useful for testing samll changes in the C++ code.

Example C++ code for ``cling`` can be found at ``tests/cling/hubbard.cl``.
