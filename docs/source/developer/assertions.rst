
Meaning of Assertions
=====================

Here we list some of common assertion failure and the solution.

.. highlight:: text

Ground State Calculation
------------------------

**Assertion:** ::

    block2/parallel_mpi.hpp:330: void block2::MPICommunicator<S>::reduce_sum(double*, size_t, int) [with S = block2::SU2Long; size_t = long unsigned int]: Assertion `ierr == 0' failed.

**Conditions:** More than one MPI processors, ``QCTypes.Conventional``, ``Random.rand_seed(0)``, and ``gaopt``. Random assertion failure.

**Reason:** A different gaopt reordering was used in different mpi processors. Then the error happens during the initialization of environments.
Then there will be an array-size mismatching due to the difference in integrals.

**Solution:** Broadcast the orbital reordering indices before reordering the integral.
