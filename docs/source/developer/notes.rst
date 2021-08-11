
Notes
=====

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
