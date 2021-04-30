
Number Theory Algorithms
========================

The :class:`Prime` class includes some number theory algorithms necessary for
integer factorization and finding primitive roots, which is used in some
Fast Fourier Transform algorithms.

.. doxygenstruct:: block2::Prime
    :members:

Fast Fourier Transform (FFT)
============================

A collection of FFT algorithms is implemented here. The implementation focuses more on
flexiablily and readability, rather than optimal performance (but the performance should
be acceptable).

The ``block2`` implementation is faster than ``numpy`` implementation for some special
array lengths, such as ``5929741 = 181 ** 3`` and ``5929742 = 2 * 7 * 13 * 31 * 1051``.

.. doxygenstruct:: block2::BasicFFT
    :members:

.. doxygenstruct:: block2::BasicFFT< 2 >
    :members:

.. doxygenstruct:: block2::RaderFFT
    :members:

.. doxygenstruct:: block2::BluesteinFFT
    :members:

.. doxygenstruct:: block2::DFT
    :members:

.. doxygenstruct:: block2::FactorizedFFT
    :members:

.. doxygenstruct:: block2::FactorizedFFT< F, P >
    :members:

.. doxygentypedef:: block2::FFT2

.. doxygentypedef:: block2::FFT
