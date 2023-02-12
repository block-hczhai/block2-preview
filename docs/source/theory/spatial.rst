
DMRG Quantum Chemistry Hamiltonian in Spatial Orbitals
======================================================

Hamiltonian
-----------

The quantum chemistry Hamiltonian is written as follows

.. math::
    \hat{H} = \sum_{ij,\sigma} t_{ij} \ a_{i\sigma}^\dagger a_{j\sigma}
    + \frac{1}{2} \sum_{ijkl, \sigma\sigma'} v_{ijkl}\
    a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'}a_{j\sigma}

where

.. math::
    t_{ij} =&\ t_{(ij)} = \int \mathrm{d}\mathbf{x} \
    \phi_i^*(\mathbf{x}) \left( -\frac{1}{2}\nabla^2 - \sum_a \frac{Z_a}{r_a} \right)
    \phi_j(\mathbf{x}) \\
    v_{ijkl} =&\ v_{(ij)(kl)} = v_{(kl)(ij)} =
    \int \mathrm{d} \mathbf{x}_1 \mathrm{d} \mathbf{x}_2 \ \frac{\phi_i^*(\mathbf{x}_1)\phi_k^*(\mathbf{x}_2)
    \phi_l(\mathbf{x}_2)\phi_j(\mathbf{x}_1)}{r_{12}}

Note that here the order of :math:`ijkl` is the same as that in ``FCIDUMP`` (chemist's notation :math:`[ij|kl]`).

Partitioning in Spatial Orbitals
--------------------------------

The partitioning of Hamiltonian in left (:math:`L`) and right (:math:`R`) blocks is given by

.. math::
    \hat{H} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R} \\
    &\ + \Big( \sum_{i\in L,\sigma} a_{i\sigma}^\dagger \hat{S}_{i\sigma}^{R} + h.c. \Big)
    + \Big( \sum_{i\in L,\sigma} a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{R} + h.c.
        + \sum_{i\in R,\sigma} a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{L} + h.c. \Big) \\
    &\ +\frac{1}{2} \Big( \sum_{ik\in L,\sigma\sigma'} \hat{A}_{ik,\sigma\sigma'}^{L} \hat{P}_{ik,\sigma\sigma'}^{R} + h.c. \Big)
    + \sum_{ij\in L} \hat{B}_{ij} \hat{Q}_{ij}^{R}
    - \sum_{il\in L,\sigma\sigma'} \hat{B}'_{il\sigma\sigma'} {\hat{Q}}^{\prime R}_{il\sigma\sigma'}

where the normal and complementary operators are defined by

.. math::
    \hat{S}_{i\sigma}^{L/R} =&\ \sum_{j\in L/R} t_{ij}a_{j\sigma}, \\
    \hat{R}_{i\sigma}^{L/R} =&\ \sum_{jkl\in L/R,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}, \\
    \hat{A}_{ik,\sigma\sigma'} =&\ a_{i\sigma}^\dagger a_{k\sigma'}^\dagger, \\
    \hat{B}_{ij} =&\ \sum_{\sigma} a_{i\sigma}^\dagger a_{j\sigma}, \\
    \hat{B}'_{il,\sigma\sigma'} =&\ a_{i\sigma}^\dagger a_{l\sigma'}, \\
    \hat{P}_{ik,\sigma\sigma'}^{R} =&\ \sum_{jl\in R} v_{ijkl} a_{l\sigma'} a_{j\sigma}, \\
    \hat{Q}_{ij}^{R} =&\ \sum_{kl\in R,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}, \\
    {\hat{Q}}_{il,\sigma\sigma'}^{\prime R} =&\ \sum_{jk\in R} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma}

Note that we need to move all on-site interaction into local Hamiltonian, so that when construction interaction terms in Hamiltonian,
operators anticommute (without giving extra constant terms).

Derivation
^^^^^^^^^^

First consider one-electron term. :math:`ij` indices have only two possibilities: :math:`i` left, :math:`j` right,
or :math:`i` right, :math:`j` left. Index :math:`i` must be associated with creation operator. So the second case
is the Hermitian conjugate of the first case. Namely,

.. math::
    \sum_{i\in L,\sigma} a_{i\sigma}^\dagger \hat{S}_{i\sigma}^{R} + h.c.
        = \sum_{i\in L,\sigma} a_{i\sigma}^\dagger \hat{S}_{i\sigma}^{R}
            + \sum_{j\in L,\sigma} \hat{S}_{j\sigma}^{R\dagger }a_{j\sigma}
        = \sum_{i\in L/R,j \in R/L,\sigma} t_{ij} a_{i\sigma}^\dagger a_{j\sigma}

Next consider one of :math:`ijkl` in left, and three of them in right. These terms are

.. math::
    \hat{H}_{1L, 3R} =&\ \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{j\in L, ikl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{k\in L, ijl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{l\in L, ijk \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma} \\
    =&\ \left[ \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{k\in L, ijl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma} \right]
    + \frac{1}{2}\sum_{j\in L, ikl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{l\in L, ijk \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}

where the terms in bracket equal to first and third terms in left-hand-side. Outside the bracket are second, forth
terms.

The conjugate of third term in rhs is second term in rhs

.. math::
    \frac{1}{2}\sum_{j\in L, ikl \in R ,\sigma\sigma'}
        v_{ijkl}  a_{j\sigma}^\dagger a_{l\sigma'}^\dagger  a_{k\sigma'} a_{i\sigma}
    = \frac{1}{2}\sum_{k\in L, ijl \in R ,\sigma\sigma'}
        v_{lkji}  a_{k\sigma}^\dagger a_{i\sigma'}^\dagger  a_{j\sigma'} a_{l\sigma}
    = \frac{1}{2}\sum_{k\in L, ijl \in R ,\sigma\sigma'}
        v_{ijkl}  a_{i\sigma'}^\dagger a_{k\sigma}^\dagger a_{l\sigma} a_{j\sigma'}

The conjugate of forth term in rhs is first term in rhs

.. math::
    \frac{1}{2}\sum_{l\in L, ijk \in R ,\sigma\sigma'}
        v_{ijkl}  a_{j\sigma}^\dagger a_{l\sigma'}^\dagger  a_{k\sigma'} a_{i\sigma}
    = \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{lkji}  a_{k\sigma}^\dagger a_{i\sigma'}^\dagger  a_{j\sigma'} a_{l\sigma}
    = \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl}  a_{i\sigma'}^\dagger a_{k\sigma}^\dagger a_{l\sigma}  a_{j\sigma'}

Therefore, using :math:`v_{ijkl} = v_{klij}`

.. math::
    \hat{H}_{1L, 3R} =&\ \left[ \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{k\in L, ijl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma} \right] + h.c. \\
    =&\ \left[ \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{k\in L, ijl \in R ,\sigma\sigma'}
        v_{ijkl} a_{k\sigma'}^\dagger a_{i\sigma}^\dagger a_{j\sigma} a_{l\sigma'} \right] + h.c. \\
    =&\ \left[ \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2}\sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{klij} a_{i\sigma'}^\dagger a_{k\sigma}^\dagger a_{l\sigma} a_{j\sigma'} \right] + h.c. \\
    =&\ \sum_{i\in L, jkl \in R ,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma} + h.c. \\
    =&\ \sum_{i\in L,\sigma} a_{i\sigma}^\dagger \sum_{jkl \in R,\sigma'}
        v_{ijkl}  a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma} + h.c. =
        \sum_{i\in L,\sigma} a_{i\sigma}^\dagger R_{i\sigma}^{R} + h.c.

Next consider the two creation operators together in left or in together in right. There are two cases.
The second case is the conjugate of the first case, namely,

.. math::
    \sum_{ik\in R, jl \in L, \sigma\sigma'} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger
        v_{ijkl} a_{l\sigma'} a_{j\sigma}
    = \sum_{jl\in R, ik \in L, \sigma\sigma'} a_{j\sigma}^\dagger a_{l\sigma'}^\dagger
        v_{jilk} a_{k\sigma'} a_{i\sigma}
    = \sum_{ik \in L, jl\in R, \sigma\sigma'} v_{jilk} a_{j\sigma}^\dagger a_{l\sigma'}^\dagger
        a_{k\sigma'} a_{i\sigma}
    = \sum_{ik \in L, jl\in R, \sigma\sigma'} v_{ijkl} \Big( a_{i\sigma}^\dagger a_{k\sigma'}^\dagger
        a_{l\sigma'} a_{j\sigma} \Big)^\dagger

This explains the :math:`\hat{A}\hat{P}` term. The last situation is, one creation in left and one creation in right.
Note that when exchange two elementary operators, one creation and one annihilation, one in left and one in right,
they must anticommute.

.. math::
    \hat{H}_{2L,2R} =&\ \frac{1}{2} \sum_{il\in L, jk\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2} \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2} \sum_{kl\in L, ij\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}
    + \frac{1}{2} \sum_{jk\in L, il\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma} \\
    =&\ 
    -\frac{1}{2} \sum_{il\in L, jk\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{l\sigma'} a_{k\sigma'}^\dagger a_{j\sigma}
    + \frac{1}{2} \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'}
    + \frac{1}{2} \sum_{kl\in L, ij\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'}
    - \frac{1}{2} \sum_{jk\in L, il\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{l\sigma'} a_{k\sigma'}^\dagger a_{j\sigma}

where the first, forth terms are combing different spins. The second, third terms are for the same spin.
First consider the same-spin case

.. math::
    &\ \frac{1}{2} \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'}
    + \frac{1}{2} \sum_{kl\in L, ij\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'} \\
    =&\ \frac{1}{2} \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'}
    + \frac{1}{2} \sum_{kl\in L, ij\in R,\sigma\sigma'}
        v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} a_{i\sigma}^\dagger a_{j\sigma} \\
    =&\ \frac{1}{2} \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'}
    + \frac{1}{2} \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{klij} a_{i\sigma'}^\dagger a_{j\sigma'} a_{k\sigma}^\dagger a_{l\sigma} \\
    =&\ \sum_{ij\in L, kl\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{j\sigma} a_{k\sigma'}^\dagger a_{l\sigma'}
    = \sum_{ij\in L} \sum_{\sigma} a_{i\sigma}^\dagger a_{j\sigma} \sum_{kl\in R_k}\sum_{\sigma'}
        v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}
    = \sum_{ij\in L} \hat{B}_{ij} \hat{Q}_{ij}^{R}

For the different-spin case,

.. math::
    &\ -\frac{1}{2} \sum_{il\in L, jk\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{l\sigma'} a_{k\sigma'}^\dagger a_{j\sigma}
    - \frac{1}{2} \sum_{jk\in L, il\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{l\sigma'} a_{k\sigma'}^\dagger a_{j\sigma}
    = -\sum_{il\in L, jk\in R,\sigma\sigma'}
        v_{ijkl} a_{i\sigma}^\dagger a_{l\sigma'} a_{k\sigma'}^\dagger a_{j\sigma} \\
    =&\ - \sum_{il\in L\sigma\sigma'} a_{i\sigma}^\dagger a_{l\sigma'} \sum_{jk\in R}
        v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma}
    = - \sum_{il\in L\sigma\sigma'} \hat{B}'_{il\sigma\sigma'} {\hat{Q}'}_{il\sigma\sigma'}^{R}

Normal/Complementary Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above version is used when left block is short in length. Note that all terms should be written in a way that operators
for particles in left block should appear in the left side of operator string, and operators for particles in right block
should appear in the right side of operator string. To write the Hermitian conjugate explicitly, we have

.. math::
    \hat{H}^{NC} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R} \\
    &\ +  \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{S}_{i\sigma}^{R} - a_{i\sigma} \hat{S}_{i\sigma}^{R\dagger} \Big)
    +  \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{R} - a_{i\sigma} \hat{R}_{i\sigma}^{R\dagger} \Big)
        + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{ik\in L,\sigma\sigma'} \Big( \hat{A}_{ik,\sigma\sigma'} \hat{P}_{ik,\sigma\sigma'}^{R} +
    \hat{A}_{ik,\sigma\sigma'}^{\dagger} \hat{P}_{ik,\sigma\sigma'}^{R\dagger}
     \Big)
    + \sum_{ij\in L} \hat{B}_{ij} \hat{Q}_{ij}^{R}
    - \sum_{il\in L,\sigma\sigma'} \hat{B}'_{il\sigma\sigma'} {\hat{Q}}^{\prime R}_{il\sigma\sigma'}

Note that no minus sign for Hermitian conjugate terms with :math:`A, P` because these are not Fermion operators.

Also note that

.. math::
    \sum_{i\in L,\sigma} a_{i\sigma}^\dagger \hat{S}_{i\sigma}^{R}
    = \sum_{i\in L,j\in R,\sigma} t_{ij} a_{i\sigma}^\dagger a_{j\sigma}
    = \sum_{j\in R,\sigma} S_{j\sigma}^{L\dagger} a_{j\sigma}

Define

.. math::
    \hat{R}_{i\sigma}^{\prime L/R} = \frac{1}{2} \hat{S}_{i\sigma}^{L/R} + \hat{R}_{i\sigma}^{L/R}
        = \frac{1}{2} \sum_{j\in L/R} t_{ij}a_{j\sigma}
        + \sum_{jkl\in L/R,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}

we have

.. math::
    \hat{H}^{NC} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{\prime R} - a_{i\sigma} \hat{R}_{i\sigma}^{\prime R\dagger} \Big)
        + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{\prime L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{\prime L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{ik\in L,\sigma\sigma'} \Big( \hat{A}_{ik,\sigma\sigma'} \hat{P}_{ik,\sigma\sigma'}^{R} +
    \hat{A}_{ik,\sigma\sigma'}^{\dagger} \hat{P}_{ik,\sigma\sigma'}^{R\dagger}
     \Big)
    + \sum_{ij\in L} \hat{B}_{ij} \hat{Q}_{ij}^{R}
    - \sum_{il\in L,\sigma\sigma'} \hat{B}'_{il\sigma\sigma'} {\hat{Q}}^{\prime R}_{il\sigma\sigma'}

With this normal/complementary partitioning, the operators required in left block are

.. math::
    \big\{ \hat{H}^{L}, \hat{1}^L, a_{i\sigma}^\dagger, a_{i\sigma}, \hat{R}_{k\sigma}^{\prime L\dagger},
    \hat{R}_{k\sigma}^{\prime L}, \hat{A}_{ij,\sigma\sigma'}, \hat{A}_{ij,\sigma\sigma'}^{\dagger},
    \hat{B}_{ij}, \hat{B}_{ij,\sigma\sigma'}^{\prime} \big\}\quad (i,j\in L, \ k \in R)

The operators required in right block are

.. math::
    \big\{ \hat{1}^{R}, \hat{H}^R, \hat{R}_{i\sigma}^{\prime R}, \hat{R}_{i\sigma}^{\prime R\dagger},
    a_{k\sigma}, a_{k\sigma}^\dagger, \hat{P}_{ij,\sigma\sigma'}^R, \hat{P}_{ij,\sigma\sigma'}^{R\dagger},
    \hat{Q}_{ij}^R, \hat{Q}_{ij,\sigma\sigma'}^{\prime R} \big\}\quad (i,j\in L, \ k \in R)

Assuming that there are :math:`K` sites in total, and :math:`K_L/K_R` sites in left/right block (optimally, :math:`K_L \le K_R`),
the total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{NC} = 1 + 1 + 4K_L + 4K_R + 8K_L^2 + K_L^2 + 4K_L^2 = 13K_L^2 + 4K + 2

Complementary/Normal Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
    \hat{H}^{CN} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{\prime R} - a_{i\sigma} \hat{R}_{i\sigma}^{\prime R\dagger} \Big)
    + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{\prime L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{\prime L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{jl\in R,\sigma\sigma'} \Big( \hat{P}_{jl,\sigma\sigma'}^{L} \hat{A}_{jl,\sigma\sigma'} +
        \hat{P}_{jl,\sigma\sigma'}^{L\dagger} \hat{A}_{jl,\sigma\sigma'}^{\dagger}
     \Big)
    + \sum_{kl\in R} \hat{Q}_{kl}^{L} \hat{B}_{kl}
    - \sum_{jk\in R, \sigma\sigma'} {\hat{Q}}^{\prime L}_{jk\sigma\sigma'} \hat{B}'_{jk\sigma\sigma'}

Now the operators required in left block are

.. math::
    \big\{ \hat{H}^L, \hat{1}^{L}, a_{i\sigma}^\dagger, a_{i\sigma}, \hat{R}_{k\sigma}^{\prime L\dagger},
    \hat{R}_{k\sigma}^{\prime L}, \hat{P}_{kl,\sigma\sigma'}^L, \hat{P}_{kl,\sigma\sigma'}^{L\dagger},
    \hat{Q}_{kl}^L, \hat{Q}_{kl,\sigma\sigma'}^{\prime L} \big\}\quad (k,l\in R, \ i \in L)

The operators required in right block are

.. math::
    \big\{ \hat{1}^R, \hat{H}^{R}, \hat{R}_{i\sigma}^{\prime R}, \hat{R}_{i\sigma}^{\prime R\dagger},
    a_{k\sigma}, a_{k\sigma}^\dagger, \hat{A}_{kl,\sigma\sigma'}, \hat{A}_{kl,\sigma\sigma'}^{\dagger},
    \hat{B}_{kl}, \hat{B}_{kl,\sigma\sigma'}^{\prime} \big\}\quad (k,l\in R, \ i \in L)

The total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{CN} = 1 + 1 + 4K_R + 4K_L + 8K_R^2 + K_R^2 + 4K_R^2 = 13K_R^2 + 4K + 2

Blocking
--------

The enlarged left/right block is denoted as :math:`L*/R*`.
Make sure that all :math:`L` operators are to the left of :math:`*` operators.

.. math::
    \hat{R}_{i\sigma}^{\prime L*} =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L} \left( \sum_{kl \in *,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
            a_{j\sigma}
        + \sum_{j\in *} \left( \sum_{kl \in L,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
            a_{j\sigma} \\
        &\ + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \left( \sum_{jl \in *} v_{ijkl} a_{l\sigma'}
            a_{j\sigma} \right)
        + \sum_{k\in *,\sigma'} a_{k\sigma'}^\dagger \left( \sum_{jl \in L} v_{ijkl} a_{l\sigma'}
            a_{j\sigma} \right)
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \left( \sum_{jk \in *} v_{ijkl} a_{k\sigma'}^\dagger
            a_{j\sigma} \right)
        - \sum_{l \in *,\sigma'} a_{l\sigma'} \left( \sum_{jk \in L} v_{ijkl} a_{k\sigma'}^\dagger
            a_{j\sigma} \right) \\
        =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L} a_{j\sigma} \left( \sum_{kl \in *,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
        + \sum_{j\in *} \left( \sum_{kl \in L,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
            a_{j\sigma} \\
        &\ + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \left( \sum_{jl \in *} v_{ijkl} a_{l\sigma'}
            a_{j\sigma} \right)
        + \sum_{k\in *,\sigma'} \left( \sum_{jl \in L} v_{ijkl} a_{l\sigma'} a_{j\sigma} \right) a_{k\sigma'}^\dagger
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \left( \sum_{jk \in *} v_{ijkl} a_{k\sigma'}^\dagger
            a_{j\sigma} \right)
        - \sum_{l \in *,\sigma'} \left( \sum_{jk \in L} v_{ijkl} a_{k\sigma'}^\dagger
            a_{j\sigma} \right) a_{l\sigma'}

Now there are two possibilities. In NC partition, in :math:`L` we have :math:`A,A^\dagger, B, B'`
and in :math:`*` we have :math:`P,P^\dagger,Q, Q'`. In CN partition, the opposite is true. Therefore, we have

.. math::
    \hat{R}_{i\sigma}^{\prime L*,NC} =&\
        \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L} a_{j\sigma} \hat{Q}_{ij}^*
        + \sum_{j\in *, kl \in L} v_{ijkl} \hat{B}_{kl} a_{j\sigma} \\
        &\ + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \hat{P}_{ik,\sigma\sigma'}^*
        + \sum_{k\in *,jl \in L, \sigma'} v_{ijkl} \hat{A}_{jl,\sigma\sigma'}^{\dagger} a_{k\sigma'}^\dagger
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime *}
        - \sum_{l \in *,jk \in L,\sigma'} v_{ijkl} \hat{B}_{kj,\sigma'\sigma}^{\prime} a_{l\sigma'} \\
    =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \hat{P}_{ik,\sigma\sigma'}^*
        + \sum_{j\in L} a_{j\sigma} \hat{Q}_{ij}^*
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime *} \\
    &\ + \sum_{k\in *,jl \in L, \sigma'} v_{ijkl} \hat{A}_{jl,\sigma\sigma'}^{\dagger} a_{k\sigma'}^\dagger
        + \sum_{j\in *, kl \in L} v_{ijkl} \hat{B}_{kl} a_{j\sigma}
        - \sum_{l \in *,jk \in L,\sigma'} v_{ijkl} \hat{B}_{kj,\sigma'\sigma}^{\prime} a_{l\sigma'} \\

.. math::
    \hat{R}_{i\sigma}^{\prime L*,CN} =&\
        \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L,kl \in *} v_{ijkl} a_{j\sigma} \hat{B}_{kl}
        + \sum_{j\in *} \hat{Q}_{ij}^{L} a_{j\sigma} \\
        &\ + \sum_{k\in L,jl \in *, \sigma'} v_{ijkl} a_{k\sigma'}^\dagger \hat{A}_{jl,\sigma\sigma'}^\dagger
        + \sum_{k\in *,\sigma'} \hat{P}_{ik,\sigma\sigma'}^L a_{k\sigma'}^\dagger
        - \sum_{l \in L,jk \in *,\sigma'} v_{ijkl} a_{l\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime}
        - \sum_{l \in *,\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime L} a_{l\sigma'} \\
        =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{k\in L,jl \in *, \sigma'} v_{ijkl} a_{k\sigma'}^\dagger \hat{A}_{jl,\sigma\sigma'}^\dagger
        + \sum_{j\in L,kl \in *} v_{ijkl} a_{j\sigma} \hat{B}_{kl}
        - \sum_{l \in L,jk \in *,\sigma'} v_{ijkl} a_{l\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime} \\
        &\ + \sum_{k\in *,\sigma'} \hat{P}_{ik,\sigma\sigma'}^L a_{k\sigma'}^\dagger
        + \sum_{j\in *} \hat{Q}_{ij}^{L} a_{j\sigma}
        - \sum_{l \in *,\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime L} a_{l\sigma'}

Similarly,

.. math::
    \hat{R}_{i\sigma}^{\prime R*,NC}
    =&\ \hat{R}_{i\sigma}^{\prime *} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{R}_{i\sigma}^{\prime R}
        + \sum_{k\in *,\sigma'} a_{k\sigma'}^\dagger \hat{P}_{ik,\sigma\sigma'}^R
        + \sum_{j\in *} a_{j\sigma} \hat{Q}_{ij}^R
        - \sum_{l \in *,\sigma'} a_{l\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime R} \\
    &\ + \sum_{k\in R,jl \in *, \sigma'} v_{ijkl} \hat{A}_{jl,\sigma\sigma'}^{\dagger} a_{k\sigma'}^\dagger
        + \sum_{j\in R, kl \in *} v_{ijkl} \hat{B}_{kl} a_{j\sigma}
        - \sum_{l \in R,jk \in *,\sigma'} v_{ijkl} \hat{B}_{kj,\sigma'\sigma}^{\prime} a_{l\sigma'} \\
    \hat{R}_{i\sigma}^{\prime R*,CN}
        =&\ \hat{R}_{i\sigma}^{\prime *} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{R}_{i\sigma}^{\prime R}
        + \sum_{k\in *,jl \in R, \sigma'} v_{ijkl} a_{k\sigma'}^\dagger \hat{A}_{jl,\sigma\sigma'}^\dagger
        + \sum_{j\in *,kl \in R} v_{ijkl} a_{j\sigma} \hat{B}_{kl}
        - \sum_{l \in *,jk \in R,\sigma'} v_{ijkl} a_{l\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime} \\
        &\ + \sum_{k\in R,\sigma'} \hat{P}_{ik,\sigma\sigma'}^* a_{k\sigma'}^\dagger
        + \sum_{j\in R} \hat{Q}_{ij}^{*} a_{j\sigma}
        - \sum_{l \in R,\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime *} a_{l\sigma'}

Number of terms

.. math::
    N_{R',NC} =&\ (2 + 5K_L + 5 K_L^2) K_R + (2 + 5 + 5K_R) K_L = 5K_L^2 K_R + 10 K_L K_R + 2K + 5K_L \\
    N_{R',CN} =&\ (2 + 5K_L + 5) K_R + (2 + 5K_R^2 + 5 K_R) K_L = 5K_R^2 K_L + 10 K_R K_L + 2K + 5K_R

Blocking of other complementary operators is straightforward

.. math::
    \hat{P}_{ik,\sigma\sigma'}^{L*,CN} =&\ \hat{P}_{ik,\sigma\sigma'}^{L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{P}_{ik,\sigma\sigma'}^*
        + \sum_{j\in L,l \in *} v_{ijkl} a_{l\sigma'} a_{j\sigma}
        + \sum_{j\in *,l \in L} v_{ijkl} a_{l\sigma'} a_{j\sigma} \\
    =&\ \hat{P}_{ik,\sigma\sigma'}^{L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{P}_{ik,\sigma\sigma'}^*
        - \sum_{j\in L,l \in *} v_{ijkl} a_{j\sigma} a_{l\sigma'}
        + \sum_{j\in *,l \in L} v_{ijkl} a_{l\sigma'} a_{j\sigma} \\
    \hat{P}_{ik,\sigma\sigma'}^{R*,NC} =&\ \hat{P}_{ik,\sigma\sigma'}^{*} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{P}_{ik,\sigma\sigma'}^R
        + \sum_{j\in *,l \in R} v_{ijkl} a_{l\sigma'} a_{j\sigma}
        + \sum_{j\in R,l \in *} v_{ijkl} a_{l\sigma'} a_{j\sigma} \\
    =&\ \hat{P}_{ik,\sigma\sigma'}^{*} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{P}_{ik,\sigma\sigma'}^R
        - \sum_{j\in *,l \in R} v_{ijkl} a_{j\sigma} a_{l\sigma'}
        + \sum_{j\in R,l \in *} v_{ijkl} a_{l\sigma'} a_{j\sigma}

and

.. math::
    \hat{Q}_{ij}^{L*,CN} =&\ \hat{Q}_{ij}^{L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{ij}^*
        + \sum_{k\in L, l \in *,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}
        + \sum_{k\in *, l \in L,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} \\
    =&\ \hat{Q}_{ij}^{L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{ij}^*
        + \sum_{k\in L, l \in *,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}
        - \sum_{k\in *, l \in L,\sigma'} v_{ijkl} a_{l\sigma'} a_{k\sigma'}^\dagger  \\
    \hat{Q}_{ij}^{R*,NC} =&\ \hat{Q}_{ij}^{*} \otimes \hat{1}^R + \hat{1}^* \otimes \hat{Q}_{ij}^R
        + \sum_{k\in *, l \in R,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}
        + \sum_{k\in R, l \in *,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'} \\
    =&\ \hat{Q}_{ij}^{*} \otimes \hat{1}^R + \hat{1}^* \otimes \hat{Q}_{ij}^R
        + \sum_{k\in *, l \in R,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}
        - \sum_{k\in R, l \in *,\sigma'} v_{ijkl} a_{l\sigma'} a_{k\sigma'}^\dagger

and

.. math::
    \hat{Q}_{il,\sigma\sigma'}^{\prime L*,CN} =&\
        \hat{Q}_{il,\sigma\sigma'}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^L \otimes \hat{Q}_{il,\sigma\sigma'}^{\prime *}
        + \sum_{j\in L, k \in *} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma}
        + \sum_{j\in *, k \in L} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma} \\
    =&\ \hat{Q}_{il,\sigma\sigma'}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^L \otimes \hat{Q}_{il,\sigma\sigma'}^{\prime *}
        - \sum_{j\in L, k \in *} v_{ijkl} a_{j\sigma} a_{k\sigma'}^\dagger
        + \sum_{j\in *, k \in L} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma} \\
    \hat{Q}_{il,\sigma\sigma'}^{\prime R*,NC} =&\
        \hat{Q}_{il,\sigma\sigma'}^{\prime *} \otimes \hat{1}^R
        + \hat{1}^* \otimes \hat{Q}_{il,\sigma\sigma'}^{\prime R}
        + \sum_{j\in *, k \in R} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma}
        + \sum_{j\in R, k \in *} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma} \\
    =&\ \hat{Q}_{il,\sigma\sigma'}^{\prime *} \otimes \hat{1}^R
        + \hat{1}^* \otimes \hat{Q}_{il,\sigma\sigma'}^{\prime R}
        - \sum_{j\in *, k \in R} v_{ijkl} a_{j\sigma} a_{k\sigma'}^\dagger
        + \sum_{j\in R, k \in *} v_{ijkl} a_{k\sigma'}^\dagger a_{j\sigma}

Middle-Site Transformation
--------------------------

When the sweep is performed from left to right, passing the middle site, we need to switch from NC partition
to CN partition. The cost is :math:`O(K^4/16)`. This happens only once in the sweep. The cost of one blocking procedure is
:math:`O(K_<^2K_>)`, but there are :math:`K` blocking steps in one sweep. So the cost for blocking in one sweep is
:math:`O(KK_<^2K_>)`. Note that the most expensive part in the program should be the Hamiltonian step in Davidson,
which scales as :math:`O(K_<^2)`.

.. math::
    \hat{P}_{ik,\sigma\sigma'}^{L,NC\to CN} =&\ \sum_{jl\in L} v_{ijkl} a_{l\sigma'} a_{j\sigma}
        = \sum_{jl\in L} v_{ijkl} \hat{A}_{jl,\sigma\sigma'}^{\dagger} \\
    \hat{Q}_{ij}^{L,NC\to CN} =&\ \sum_{kl\in L,\sigma'} v_{ijkl} a_{k\sigma'}^\dagger a_{l\sigma'}
        = \sum_{kl\in L} v_{ijkl} \hat{B}_{kl} \\
    \hat{Q}_{il,\sigma\sigma'}^{\prime L,NC\to CN} =&\ \sum_{jk \in L} v_{ijkl}
        a_{k\sigma'}^\dagger a_{j\sigma} = \sum_{jk \in L} v_{ijkl} \hat{B}_{kj,\sigma'\sigma}^{\prime}
