
Global Settings
===============

In **block2**, we try to minimize the use of global variables.
Two global variables (``frame_()`` and ``threading_()``) have been used for controlling
global settings such as stack memory, scartch folder and threading schemes.

Note that in ``block2`` the distributed parallelization scheme is handled
locally.

Threading
---------

.. doxygenenum:: block2::ThreadingTypes

.. doxygenenum:: block2::SeqTypes

.. doxygenstruct:: block2::Threading
    :members:

.. doxygenfunction:: block2::threading_

.. doxygendefine:: threading

Allocators
----------

.. doxygenstruct:: block2::Allocator
    :members:

.. doxygenstruct:: block2::StackAllocator
    :members:

.. doxygenstruct:: block2::VectorAllocator
    :members:

.. doxygenfunction:: block2::ialloc_

.. doxygenfunction:: block2::dalloc_

.. doxygendefine:: ialloc

Data Frame
----------

.. doxygenstruct:: block2::DataFrame
    :members:

.. doxygenfunction:: block2::frame_

Miscellanies
------------

.. doxygenfunction:: block2::check_signal_

.. doxygenfunction:: block2::print_trace
