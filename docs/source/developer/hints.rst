
Debugging Hints
===============

Here we list some of common assertion failure, errors, wrong outputs, and the solutions.

.. highlight:: text

Ground State Calculation
------------------------

[2021-05-09]
^^^^^^^^^^^^

**Assertion:** ::

    block2/parallel_mpi.hpp:330: void block2::MPICommunicator<S>::reduce_sum(double*, size_t, int) [with S = block2::SU2Long; size_t = long unsigned int]: Assertion `ierr == 0' failed.

**Conditions:** More than one MPI processors, ``QCTypes.Conventional``, ``Random.rand_seed(0)``, and ``gaopt``. Random assertion failure.

**Reason:** A different gaopt reordering was used in different mpi processors. Then the error happens during the initialization of environments.
Then there will be an array-size mismatching due to the difference in integrals.

**Solution:** Broadcast the orbital reordering indices before reordering the integral.

[2021-05-10]
^^^^^^^^^^^^

**Assertion:** ::

    block2/operator_functions.hpp:185: void block2::OperatorFunctions<S>::tensor_rotate(const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, bool, double) const [with S = block2::SZLong]: Assertion `a->get_type() == SparseMatrixTypes::Normal && c->get_type() == SparseMatrixTypes::Normal && rot_bra->get_type() == SparseMatrixTypes::Normal && rot_ket->get_type() == SparseMatrixTypes::Normal' failed.

**Conditions:** Loaded MPO, CSR.

**Reason:** The non-CSR ``OperatorFunctions`` is used for calculation requiring CSR matrices, after loading MPO.

**Solution:** Change ``csr_opf = OperatorFunctions(cg)`` to ``csr_opf = CSROperatorFunctions(cg)``.

[2021-05-11]
^^^^^^^^^^^^

**Output:** ::

    Sweep =   15 | Direction = backward | Bond dimension = 2000 | Noise =  1.00e-07 | Dav threshold =  1.00e-08
    <-- Site =   11-  12 .. Mmps =    3 Ndav =   1 E =    -36.8356589402 Error = 0.00e+00 FLOPS = 4.12e+06 Tdav = 0.02 T = 0.17
    <-- Site =   10-  11 .. Mmps =   10 Ndav =   1 E =    -36.8356589402 Error = 0.00e+00 FLOPS = 2.91e+08 Tdav = 0.02 T = 0.18
    <-- Site =    9-  10 .. Mmps =   35 Ndav =   1 E =    -36.8356589402 Error = 0.00e+00 FLOPS = 9.70e+09 Tdav = 0.02 T = 0.20
    <-- Site =    8-   9 .. Mmps =  126 Ndav =   1 E =    -36.8356589402 Error = 0.00e+00 FLOPS = 6.96e+10 Tdav = 0.06 T = 0.40
    <-- Site =    7-   8 .. Mmps =  462 Ndav =   1 E =    -36.8356589402 Error = 0.00e+00 FLOPS = 1.52e+11 Tdav = 0.28 T = 1.08
    <-- Site =    6-   7 .. Mmps = 1454 Ndav =   1 E =    -36.8356589402 Error = 4.57e-13 FLOPS = 2.27e+11 Tdav = 0.79 T = 2.54
    <-- Site =    5-   6 .. Mmps = 1679 Ndav =  12 E =    -37.0888587109 Error = 1.41e-12 FLOPS = 2.83e+11 Tdav = 7.21 T = 12.32
    <-- Site =    4-   5 .. Mmps =  904 Ndav =   1 E =    -37.0888587109 Error = 1.53e-12 FLOPS = 1.95e+11 Tdav = 0.27 T = 1.91
    <-- Site =    3-   4 .. Mmps =  490 Ndav =   1 E =    -37.0888587109 Error = 8.62e-13 FLOPS = 7.32e+10 Tdav = 0.05 T = 0.60
    <-- Site =    2-   3 .. Mmps =  209 Ndav =   1 E =    -37.0888587109 Error = 2.47e-13 FLOPS = 9.69e+09 Tdav = 0.02 T = 0.28
    <-- Site =    1-   2 .. Mmps =   64 Ndav =   1 E =    -37.0888587109 Error = 9.69e-15 FLOPS = 1.06e+09 Tdav = 0.01 T = 0.26
    <-- Site =    0-   1 .. Mmps =   11 Ndav =   1 E =    -37.0888587109 Error = 5.58e-15 FLOPS = 5.78e+06 Tdav = 0.02 T = 0.16
    Time elapsed =    187.772 | E =     -37.0888587109 | DE = -6.18e-12 | DW = 1.53e-12
    Time sweep =       20.100 | 2.11 TFLOP/SWP
    | Tcomm = 7.916 | Tidle = 3.657 | Twait = 0.000 | Dmem = 89.2 MB (11%) | Imem = 93.8 KB (96%) | Hmem = 736 MB | Pmem = 50.8 MB
    | Tread = 0.505 | Twrite = 0.553 | Tfpread = 0.462 | Tfpwrite = 0.090 | Tasync = 0.000
    | Trot = 0.368 | Tctr = 0.055 | Tint = 0.016 | Tmid = 2.304 | Tdctr = 0.033 | Tdiag = 0.310 | Tinfo = 0.039
    | Teff = 1.591 | Tprt = 2.578 | Teig = 8.760 | Tblk = 16.722 | Tmve = 3.376 | Tdm = 0.000 | Tsplt = 0.000 | Tsvd = 1.678

    Sweep =   16 | Direction =  forward | Bond dimension = 2000 | Noise =  1.00e-07 | Dav threshold =  1.00e-08
    --> Site =    0-   1 .. Mmps =    3 Ndav =   1 E =    -37.0888587109 Error = 0.00e+00 FLOPS = 4.51e+06 Tdav = 0.02 T = 0.18
    --> Site =    1-   2 .. Mmps =   10 Ndav =   1 E =    -37.0888587109 Error = 0.00e+00 FLOPS = 8.65e+08 Tdav = 0.01 T = 0.09
    --> Site =    2-   3 .. Mmps =   35 Ndav =   1 E =    -37.0888587109 Error = 0.00e+00 FLOPS = 1.03e+10 Tdav = 0.02 T = 0.11
    --> Site =    3-   4 .. Mmps =  126 Ndav =   1 E =    -37.0888587109 Error = 0.00e+00 FLOPS = 7.65e+10 Tdav = 0.05 T = 0.35
    --> Site =    4-   5 .. Mmps =  462 Ndav =   1 E =    -37.0888587109 Error = 0.00e+00 FLOPS = 1.61e+11 Tdav = 0.32 T = 1.25
    --> Site =    5-   6 .. Mmps = 1511 Ndav =   1 E =    -37.0888587109 Error = 3.25e-13 FLOPS = 2.24e+11 Tdav = 0.76 T = 2.50
    --> Site =    6-   7 .. Mmps = 1805 Ndav =  17 E =    -36.8356599462 Error = 1.65e-12 FLOPS = 3.10e+11 Tdav = 10.53 T = 15.73
    --> Site =    7-   8 .. Mmps =  975 Ndav =   1 E =    -36.8356599462 Error = 1.51e-12 FLOPS = 1.59e+11 Tdav = 0.38 T = 2.13
    --> Site =    8-   9 .. Mmps =  408 Ndav =   1 E =    -36.8356599462 Error = 8.11e-13 FLOPS = 7.52e+10 Tdav = 0.06 T = 0.53
    --> Site =    9-  10 .. Mmps =  156 Ndav =   1 E =    -36.8356599462 Error = 2.57e-13 FLOPS = 1.16e+10 Tdav = 0.02 T = 0.13
    --> Site =   10-  11 .. Mmps =   57 Ndav =   1 E =    -36.8356599462 Error = 1.46e-14 FLOPS = 4.29e+08 Tdav = 0.01 T = 0.18
    --> Site =   11-  12 .. Mmps =   12 Ndav =   1 E =    -36.8356599462 Error = 4.83e-15 FLOPS = 4.13e+06 Tdav = 0.02 T = 0.06
    Time elapsed =    211.003 | E =     -37.0888587109 | DE = 4.21e-12 | DW = 1.65e-12
    Time sweep =       23.231 | 3.23 TFLOP/SWP
    | Tcomm = 8.521 | Tidle = 2.996 | Twait = 0.000 | Dmem = 95.4 MB (10%) | Imem = 93.8 KB (96%) | Hmem = 736 MB | Pmem = 52.5 MB
    | Tread = 0.550 | Twrite = 0.624 | Tfpread = 0.504 | Tfpwrite = 0.092 | Tasync = 0.000
    | Trot = 0.385 | Tctr = 0.039 | Tint = 0.023 | Tmid = 2.480 | Tdctr = 0.052 | Tdiag = 0.323 | Tinfo = 0.035
    | Teff = 1.734 | Tprt = 2.656 | Teig = 12.197 | Tblk = 19.563 | Tmve = 3.667 | Tdm = 0.000 | Tsplt = 0.000 | Tsvd = 1.508

**Conditions:** More than one MPI processors, and ``QCTypes.Conventional``.

**Reason:** We see from the output that the energy jumps between two values even in very large bond dimension.
If only one MPI is used, there is no such behavior.
This is because the input integrals ``h1e`` and ``g2e`` are not synchronized.
In ``QCTypes.Conventional``, communication between MPI procs only happens at the middle site.
After this communication, the inconsistentcy between integrals can cause an artificial change in energy.
Note that inside ``block2``, we do not explicitly synchronize integral. In future, for larger systems,
the integral can even be distributed, such that synchronization will not be meaningful.

**Solution:** Synchronizing the input integrals ``h1e`` and ``g2e`` can solve this problem.

[2021-05-12]
^^^^^^^^^^^^

**Error Message:** (note that this problem in ``main.py`` has been fixed in commit 4f87784) ::

    Traceback (most recent call last):
    File "block2/pyblock2/driver/main.py", line 302, in <module>
        mps.load_data()
    RuntimeError: MPS::load_data on '/central/scratch/.../F.MPS.KET.-1' failed.

or ::

    Traceback (most recent call last):
    File "block2/pyblock2/driver/main.py", line 313, in <module>
        mps.load_mutable()
    RuntimeError: SparseMatrix:load_data on '/central/scratch/.../F.MPS.KET.14' failed.

or ::

    Traceback (most recent call last):
    File "block2/pyblock2/driver/main.py", line 313, in <module>
        mps.load_mutable()
    ValueError: cannot create std::vector larger than max_size()

**Conditions:** More than one MPI processors, python driver, happening with a very low probablity.

**Reason:** The problematic code is: ::

    mps.load_data()
    if mps.dot != dot and nroots == 1:
        mps.dot = dot
        mps.save_data()

And the non-root MPI proc can load data before or after the root proc saves the data. The wrong loaded data can cause the
subsequent ``mps.load_mutable()`` to fail.

**Solution:** Adding ``MPI.barrier()`` around ``mps.save_data()``.

Linear
------

[2021-05-14]
^^^^^^^^^^^^

**Assertion:** ::

    block2/moving_environment.hpp:110: block2::MovingEnvironment<S>::MovingEnvironment(const std::shared_ptr<block2::MPO<S> >&, const std::shared_ptr<block2::MPS<S> >&, const std::shared_ptr<block2::MPS<S> >&, const string&) [with S = block2::SU2Long; std::string = std::__cxx11::basic_string<char>]: Assertion `bra->center == ket->center && bra->dot == ket->dot' failed.

**Conditions:** Different bra and ket.

**Reason:** The bra and ket for initialization of MovingEnvironment do not have the same center.

**Solution:** Initializing bra or ket with consistent center, or do a sweep to align the MPS center.

[2021-05-14]
^^^^^^^^^^^^

**Assertion:** ::

    block2/operator_functions.hpp:194: void block2::OperatorFunctions<S>::tensor_rotate(const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, bool, double) const [with S = block2::SU2Long]: Assertion `adq == cdq && a->info->n >= c->info->n' failed.

**Conditions:** Different bra and ket.

**Reason:** The bra and ket has different MPSInfo, but the two MPSInfo has the same tag. When saving to/loading from disk,
the information stored in the two MPSInfo can interfere with each other.

**Solution:** Use different tags for different MPSInfo.

[2021-05-14]
^^^^^^^^^^^^

**Assertion:** ::

    block2/csr_matrix_functions.hpp:387: static void block2::CSRMatrixFunctions::multiply(const MatrixRef&, bool, const block2::CSRMatrixRef&, bool, const MatrixRef&, double, double): Assertion `(conja ? a.m : a.n) == (conjb ? b.n : b.m)' failed.

**Conditions:** Different bra and ket, CSR, IdentityMPO with bra and ket with different bases.

**Reason:** Wrong basis was used in the constructor of IdentityMPO.

**Solution:** Change ``IdentityMPO(mpo_bra.basis, mpo_bra.basis, ...)`` to ``IdentityMPO(mpo_bra.basis, mpo_ket.basis, ...)``.

[2021-05-18]
^^^^^^^^^^^^

**Assertion:** ::

    block2/csr_matrix_functions.hpp:396: static void block2::CSRMatrixFunctions::multiply(const MatrixRef&, bool, const block2::CSRMatrixRef&, bool, const MatrixRef&, double, double): Assertion `st == SPARSE_STATUS_SUCCESS' failed.

**Conditions:** CSR, ``SeqTypes.Tasked``.

**Reason:** ``SeqTypes.Tasked`` cannot be used together with CSR.

**Solution:** Change ``Global.threading.seq_type = SeqTypes.Tasked`` to ``Global.threading.seq_type = SeqTypes.Nothing``.

[2021-05-22]

**Assertion:** ::

    block2/sparse_matrix.hpp:552: void block2::SparseMatrixInfo<S, typename std::enable_if<std::integral_constant<bool, (sizeof (S) == sizeof (unsigned int))>::value>::type>::save_data(std::ostream&, bool) const [with S = block2::SU2Long; typename std::enable_if<std::integral_constant<bool, (sizeof (S) == sizeof (unsigned int))>::value>::type = void; std::ostream = std::basic_ostream<char>]: Assertion `n != -1' failed.

**Conditions:** ``mps.save_mutable``.

**Reason:** Some MPS tensors are deallocated (unloaded) after ``mps.flip_fused_form(...)`` or ``mps.move_left(...)``.

**Solution:** Call ``mps.load_mutable()`` after using ``mps.flip_fused_form(...)`` or ``mps.move_left(...)``,
so that ``mps.save_mutable()`` will be successful.
