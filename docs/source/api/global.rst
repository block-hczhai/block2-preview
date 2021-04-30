
Global Settings
===============

In **block2**, we try to minimize the use of global variables.
Two global variables have been used for controlling global settings such as stack memory,
scartch folder and threading schemes.

.. doxygendefine:: frame

.. doxygendefine:: threading

Threading
---------

.. doxygenenum:: block2::ThreadingTypes

.. doxygenenum:: block2::SeqTypes

.. doxygenstruct:: block2::Threading
    :members:

.. doxygenfunction:: block2::threading_
