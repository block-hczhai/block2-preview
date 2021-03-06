
Tools
=====

Floating-Point Number Compression
---------------------------------

For data like ``FCIDUMP`` integrals, the array elements do not need to be stored in its full binary form.
Storage or memory can be saved by using flexible bit representation for numbers of different magnitudes.
The :class:`FPCodec` class implements the lossless and lossy compression and decompression of floating-point
number arrays. Lossless compression is possible when all numebrs are close in their order of magnitudes.

.. doxygenstruct:: block2::FPtraits
    :members:

.. doxygenstruct:: block2::FPtraits< float >
    :members:

.. doxygenstruct:: block2::FPtraits< double >
    :members:

.. doxygenfunction:: block2::binary_repr

.. doxygenstruct:: block2::BitsCodec
    :members:

.. doxygenstruct:: block2::FPCodec
    :members:

.. doxygenstruct:: block2::CompressedVector
    :members:

.. doxygenstruct:: block2::CompressedVectorMT
    :members:

Kuhn-Munkres Algorithm
----------------------

The :class:`KuhnMunkres` class implements the Kuhn-Munkres algorithm for
finding the best matching in a bipartite with the lowest cost.

.. doxygenstruct:: block2::KuhnMunkres
    :members:

Number Theory Algorithms
------------------------

The :class:`Prime` class includes some number theory algorithms necessary for
integer factorization and finding primitive roots, which is used in some
Fast Fourier Transform algorithms.

.. doxygenstruct:: block2::Prime
    :members:

Fast Fourier Transform (FFT)
----------------------------

A collection of FFT algorithms is implemented here. The implementation focuses more on
flexibility and readability, rather than optimal performance (but the performance should
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
