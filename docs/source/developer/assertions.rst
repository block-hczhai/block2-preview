
Meaning of Assertions
=====================

Here we list some of common assertion failure and the solution.

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
