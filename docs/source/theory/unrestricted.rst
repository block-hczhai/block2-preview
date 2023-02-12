
DMRG Quantum Chemistry Hamiltonian in Unrestricted Spatial Orbitals
===================================================================

Hamiltonian
-----------

The quantum chemistry Hamiltonian is written as follows

.. math::
    \hat{H} = \sum_{ij,\sigma} t_{ij,\sigma} \ a_{i\sigma}^\dagger a_{j\sigma}
    + \frac{1}{2} \sum_{ijkl, \sigma\sigma'} v_{ijkl, \sigma\sigma'}\
    a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'}a_{j\sigma}

where

.. math::
    t_{ij,\sigma} =&\ t_{(ij),\sigma} = \int \mathrm{d}\mathbf{x} \
    \phi_{i\sigma}^*(\mathbf{x}) \left( -\frac{1}{2}\nabla^2 - \sum_a \frac{Z_a}{r_a} \right)
    \phi_{j\sigma}(\mathbf{x}) \\
    v_{ijkl,\sigma\sigma'} =&\ v_{(ij)(kl),\sigma\sigma'} = v_{(kl)(ij),\sigma\sigma'} =
    \int \mathrm{d} \mathbf{x}_1 \mathrm{d} \mathbf{x}_2 \ \frac{\phi_{i\sigma}^*(\mathbf{x}_1)\phi_{k\sigma'}^*(\mathbf{x}_2)
    \phi_{l\sigma'}(\mathbf{x}_2)\phi_{j\sigma}(\mathbf{x}_1)}{r_{12}}

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
    + \sum_{ij\in L,\sigma} \hat{B}_{ij\sigma} \hat{Q}_{ij\sigma}^{R}
    - \sum_{il\in L,\sigma\sigma'} \hat{B}'_{il\sigma\sigma'} {\hat{Q}}^{\prime R}_{il\sigma\sigma'}

where the normal and complementary operators are defined by

.. math::
    \hat{S}_{i\sigma}^{L/R} =&\ \sum_{j\in L/R} t_{ij,\sigma}a_{j\sigma}, \\
    \hat{R}_{i\sigma}^{L/R} =&\ \sum_{jkl\in L/R,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}, \\
    \hat{A}_{ik,\sigma\sigma'} =&\ a_{i\sigma}^\dagger a_{k\sigma'}^\dagger, \\
    \hat{B}_{ij,\sigma} =&\ a_{i\sigma}^\dagger a_{j\sigma}, \\
    \hat{B}'_{il,\sigma\sigma'} =&\ a_{i\sigma}^\dagger a_{l\sigma'}, \\
    \hat{P}_{ik,\sigma\sigma'}^{R} =&\ \sum_{jl\in R} v_{ijkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma}, \\
    \hat{Q}_{ij,\sigma}^{R} =&\ \sum_{kl\in R,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'}, \\
    {\hat{Q}}_{il,\sigma\sigma'}^{\prime R} =&\ \sum_{jk\in R} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{j\sigma}

Note that we need to move all on-site interaction into local Hamiltonian, so that when construction interaction terms in Hamiltonian,
operators anticommute (without giving extra constant terms).

Define

.. math::
    \hat{R}_{i\sigma}^{\prime L/R} = \frac{1}{2} \hat{S}_{i\sigma}^{L/R} + \hat{R}_{i\sigma}^{L/R}
        = \frac{1}{2} \sum_{j\in L/R} t_{ij,\sigma}a_{j\sigma}
        + \sum_{jkl\in L/R,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}

Then we have

.. math::
    \hat{H}^{NC} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{\prime R} - a_{i\sigma} \hat{R}_{i\sigma}^{\prime R\dagger} \Big)
        + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{\prime L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{\prime L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{ik\in L,\sigma\sigma'} \Big( \hat{A}_{ik,\sigma\sigma'} \hat{P}_{ik,\sigma\sigma'}^{R} +
    \hat{A}_{ik,\sigma\sigma'}^{\dagger} \hat{P}_{ik,\sigma\sigma'}^{R\dagger}
     \Big)
    + \sum_{ij\in L,\sigma} \hat{B}_{ij,\sigma} \hat{Q}_{ij,\sigma}^{R}
    - \sum_{il\in L,\sigma\sigma'} \hat{B}'_{il\sigma\sigma'} {\hat{Q}}^{\prime R}_{il\sigma\sigma'}

Normal/Complementary Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With this normal/complementary partitioning, the operators required in left block are

.. math::
    \big\{ \hat{H}^{L}, \hat{1}^L, a_{i\sigma}^\dagger, a_{i\sigma}, \hat{R}_{k\sigma}^{\prime L\dagger},
    \hat{R}_{k\sigma}^{\prime L}, \hat{A}_{ij,\sigma\sigma'}, \hat{A}_{ij,\sigma\sigma'}^{\dagger},
    \hat{B}_{ij,\sigma}, \hat{B}_{ij,\sigma\sigma'}^{\prime} \big\}\quad (i,j\in L, \ k \in R)

The operators required in right block are

.. math::
    \big\{ \hat{1}^{R}, \hat{H}^R, \hat{R}_{i\sigma}^{\prime R}, \hat{R}_{i\sigma}^{\prime R\dagger},
    a_{k\sigma}, a_{k\sigma}^\dagger, \hat{P}_{ij,\sigma\sigma'}^R, \hat{P}_{ij,\sigma\sigma'}^{R\dagger},
    \hat{Q}_{ij,\sigma}^R, \hat{Q}_{ij,\sigma\sigma'}^{\prime R} \big\}\quad (i,j\in L, \ k \in R)

Assuming that there are :math:`K` sites in total, and :math:`K_L/K_R` sites in left/right block (optimally, :math:`K_L \le K_R`),
the total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{NC} = 1 + 1 + 4K_L + 4K_R + 8K_L^2 + 2K_L^2 + 4K_L^2 = 14K_L^2 + 4K + 2

Complementary/Normal Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
    \hat{H}^{CN} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{\prime R} - a_{i\sigma} \hat{R}_{i\sigma}^{\prime R\dagger} \Big)
    + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{\prime L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{\prime L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{jl\in R,\sigma\sigma'} \Big( \hat{P}_{jl,\sigma\sigma'}^{L} \hat{A}_{jl,\sigma\sigma'} +
        \hat{P}_{jl,\sigma\sigma'}^{L\dagger} \hat{A}_{jl,\sigma\sigma'}^{\dagger}
     \Big)
    + \sum_{kl\in R,\sigma} \hat{Q}_{kl,\sigma}^{L} \hat{B}_{kl,\sigma}
    - \sum_{jk\in R, \sigma\sigma'} {\hat{Q}}^{\prime L}_{jk\sigma\sigma'} \hat{B}'_{jk\sigma\sigma'}

Now the operators required in left block are

.. math::
    \big\{ \hat{H}^L, \hat{1}^{L}, a_{i\sigma}^\dagger, a_{i\sigma}, \hat{R}_{k\sigma}^{\prime L\dagger},
    \hat{R}_{k\sigma}^{\prime L}, \hat{P}_{kl,\sigma\sigma'}^L, \hat{P}_{kl,\sigma\sigma'}^{L\dagger},
    \hat{Q}_{kl,\sigma}^L, \hat{Q}_{kl,\sigma\sigma'}^{\prime L} \big\}\quad (k,l\in R, \ i \in L)

The operators required in right block are

.. math::
    \big\{ \hat{1}^R, \hat{H}^{R}, \hat{R}_{i\sigma}^{\prime R}, \hat{R}_{i\sigma}^{\prime R\dagger},
    a_{k\sigma}, a_{k\sigma}^\dagger, \hat{A}_{kl,\sigma\sigma'}, \hat{A}_{kl,\sigma\sigma'}^{\dagger},
    \hat{B}_{kl,\sigma}, \hat{B}_{kl,\sigma\sigma'}^{\prime} \big\}\quad (k,l\in R, \ i \in L)

The total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{CN} = 1 + 1 + 4K_R + 4K_L + 8K_R^2 + 2K_R^2 + 4K_R^2 = 14K_R^2 + 4K + 2

Blocking
--------

The enlarged left/right block is denoted as :math:`L*/R*`.
Make sure that all :math:`L` operators are to the left of :math:`*` operators.

.. math::
    \hat{R}_{i\sigma}^{\prime L*} =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L} \left( \sum_{kl \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
            a_{j\sigma}
        + \sum_{j\in *} \left( \sum_{kl \in L,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
            a_{j\sigma} \\
        &\ + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \left( \sum_{jl \in *} v_{ijkl,\sigma\sigma'} a_{l\sigma'}
            a_{j\sigma} \right)
        + \sum_{k\in *,\sigma'} a_{k\sigma'}^\dagger \left( \sum_{jl \in L} v_{ijkl,\sigma\sigma'} a_{l\sigma'}
            a_{j\sigma} \right)
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \left( \sum_{jk \in *} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger
            a_{j\sigma} \right)
        - \sum_{l \in *,\sigma'} a_{l\sigma'} \left( \sum_{jk \in L} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger
            a_{j\sigma} \right) \\
        =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L} a_{j\sigma} \left( \sum_{kl \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
        + \sum_{j\in *} \left( \sum_{kl \in L,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'} \right)
            a_{j\sigma} \\
        &\ + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \left( \sum_{jl \in *} v_{ijkl,\sigma\sigma'} a_{l\sigma'}
            a_{j\sigma} \right)
        + \sum_{k\in *,\sigma'} \left( \sum_{jl \in L} v_{ijkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma} \right) a_{k\sigma'}^\dagger
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \left( \sum_{jk \in *} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger
            a_{j\sigma} \right)
        - \sum_{l \in *,\sigma'} \left( \sum_{jk \in L} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger
            a_{j\sigma} \right) a_{l\sigma'}

Now there are two possibilities. In NC partition, in :math:`L` we have :math:`A,A^\dagger, B, B'`
and in :math:`*` we have :math:`P,P^\dagger,Q, Q'`. In CN partition, the opposite is true. Therefore, we have

.. math::
    \hat{R}_{i\sigma}^{\prime L*,NC} =&\
        \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L} a_{j\sigma} \hat{Q}_{ij,\sigma}^*
        + \sum_{j\in *, kl \in L,\sigma'} v_{ijkl,\sigma\sigma'} \hat{B}_{kl,\sigma'} a_{j\sigma} \\
        &\ + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \hat{P}_{ik,\sigma\sigma'}^*
        + \sum_{k\in *,jl \in L, \sigma'} v_{ijkl,\sigma\sigma'} \hat{A}_{jl,\sigma\sigma'}^{\dagger} a_{k\sigma'}^\dagger
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime *}
        - \sum_{l \in *,jk \in L,\sigma'} v_{ijkl,\sigma\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime} a_{l\sigma'} \\
    =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \hat{P}_{ik,\sigma\sigma'}^*
        + \sum_{j\in L} a_{j\sigma} \hat{Q}_{ij,\sigma}^*
        - \sum_{l \in L,\sigma'} a_{l\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime *} \\
    &\ + \sum_{k\in *,jl \in L, \sigma'} v_{ijkl,\sigma\sigma'} \hat{A}_{jl,\sigma\sigma'}^{\dagger} a_{k\sigma'}^\dagger
        + \sum_{j\in *, kl \in L,\sigma'} v_{ijkl,\sigma\sigma'} \hat{B}_{kl,\sigma'} a_{j\sigma}
        - \sum_{l \in *,jk \in L,\sigma'} v_{ijkl,\sigma\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime} a_{l\sigma'} \\

.. math::
    \hat{R}_{i\sigma}^{\prime L*,CN} =&\
        \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{j\in L,kl \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{j\sigma} \hat{B}_{kl,\sigma'}
        + \sum_{j\in *} \hat{Q}_{ij,\sigma}^{L} a_{j\sigma} \\
        &\ + \sum_{k\in L,jl \in *, \sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger \hat{A}_{jl,\sigma\sigma'}^\dagger
        + \sum_{k\in *,\sigma'} \hat{P}_{ik,\sigma\sigma'}^L a_{k\sigma'}^\dagger
        - \sum_{l \in L,jk \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{l\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime}
        - \sum_{l \in *,\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime L} a_{l\sigma'} \\
        =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{k\in L,jl \in *, \sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger \hat{A}_{jl,\sigma\sigma'}^\dagger
        + \sum_{j\in L,kl \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{j\sigma} \hat{B}_{kl,\sigma'}
        - \sum_{l \in L,jk \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{l\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime} \\
        &\ + \sum_{k\in *,\sigma'} \hat{P}_{ik,\sigma\sigma'}^L a_{k\sigma'}^\dagger
        + \sum_{j\in *} \hat{Q}_{ij,\sigma}^{L} a_{j\sigma}
        - \sum_{l \in *,\sigma'} \hat{Q}_{il,\sigma\sigma'}^{\prime L} a_{l\sigma'}

Simplified Form
---------------

Define

.. math::
    {\hat{Q}}_{ij,\sigma\sigma'}^{\prime\prime R} = \delta_{\sigma\sigma'} \hat{Q}^{R}_{ij\sigma}
        - \hat{Q}^{\prime R}_{ij\sigma\sigma'}

we have N/C form

.. math::
    \hat{H}^{NC} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{\prime R} - a_{i\sigma} \hat{R}_{i\sigma}^{\prime R\dagger} \Big)
        + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{\prime L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{\prime L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{ik\in L,\sigma\sigma'} \Big( \hat{A}_{ik,\sigma\sigma'} \hat{P}_{ik,\sigma\sigma'}^{R} +
    \hat{A}_{ik,\sigma\sigma'}^{\dagger} \hat{P}_{ik,\sigma\sigma'}^{R\dagger}
     \Big)
    + \sum_{ij\in L,\sigma\sigma'} \hat{B}'_{ij\sigma\sigma'} {\hat{Q}}^{\prime\prime R}_{ij\sigma\sigma'}

With this normal/complementary partitioning, the operators required in left block are

.. math::
    \big\{ \hat{H}^{L}, \hat{1}^L, a_{i\sigma}^\dagger, a_{i\sigma}, \hat{R}_{k\sigma}^{\prime L\dagger},
    \hat{R}_{k\sigma}^{\prime L}, \hat{A}_{ij,\sigma\sigma'}, \hat{A}_{ij,\sigma\sigma'}^{\dagger},
    \hat{B}_{ij,\sigma\sigma'}^{\prime} \big\}\quad (i,j\in L, \ k \in R)

The operators required in right block are

.. math::
    \big\{ \hat{1}^{R}, \hat{H}^R, \hat{R}_{i\sigma}^{\prime R}, \hat{R}_{i\sigma}^{\prime R\dagger},
    a_{k\sigma}, a_{k\sigma}^\dagger, \hat{P}_{ij,\sigma\sigma'}^R, \hat{P}_{ij,\sigma\sigma'}^{R\dagger},
    \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime R} \big\}\quad (i,j\in L, \ k \in R)

Assuming that there are :math:`K` sites in total, and :math:`K_L/K_R` sites in left/right block (optimally, :math:`K_L \le K_R`),
the total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{NC} = 1 + 1 + 4K_L + 4K_R + 8K_L^2 + 4K_L^2 = 12K_L^2 + 4K + 2

and C/N form

.. math::
    \hat{H}^{CN} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L,\sigma} \Big( a_{i\sigma}^\dagger \hat{R}_{i\sigma}^{\prime R} - a_{i\sigma} \hat{R}_{i\sigma}^{\prime R\dagger} \Big)
    + \sum_{i\in R,\sigma} \Big( \hat{R}_{i\sigma}^{\prime L\dagger} a_{i\sigma} - \hat{R}_{i\sigma}^{\prime L} a_{i\sigma}^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{jl\in R,\sigma\sigma'} \Big( \hat{P}_{jl,\sigma\sigma'}^{L} \hat{A}_{jl,\sigma\sigma'} +
        \hat{P}_{jl,\sigma\sigma'}^{L\dagger} \hat{A}_{jl,\sigma\sigma'}^{\dagger}
     \Big)
    + \sum_{kl\in R, \sigma\sigma'} {\hat{Q}}^{\prime\prime L}_{kl\sigma\sigma'} \hat{B}'_{kl\sigma\sigma'}

Now the operators required in left block are

.. math::
    \big\{ \hat{H}^L, \hat{1}^{L}, a_{i\sigma}^\dagger, a_{i\sigma}, \hat{R}_{k\sigma}^{\prime L\dagger},
    \hat{R}_{k\sigma}^{\prime L}, \hat{P}_{kl,\sigma\sigma'}^L, \hat{P}_{kl,\sigma\sigma'}^{L\dagger},
    \hat{Q}_{kl,\sigma\sigma'}^{\prime\prime L} \big\}\quad (k,l\in R, \ i \in L)

The operators required in right block are

.. math::
    \big\{ \hat{1}^R, \hat{H}^{R}, \hat{R}_{i\sigma}^{\prime R}, \hat{R}_{i\sigma}^{\prime R\dagger},
    a_{k\sigma}, a_{k\sigma}^\dagger, \hat{A}_{kl,\sigma\sigma'}, \hat{A}_{kl,\sigma\sigma'}^{\dagger},
    \hat{B}_{kl,\sigma\sigma'}^{\prime} \big\}\quad (k,l\in R, \ i \in L)

The total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{CN} = 1 + 1 + 4K_R + 4K_L + 8K_R^2 + 4K_R^2 = 12K_R^2 + 4K + 2

Then for blocking

.. math::
    \hat{R}_{i\sigma}^{\prime L*,NC}
    =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{k\in L,\sigma'} a_{k\sigma'}^\dagger \hat{P}_{ik,\sigma\sigma'}^*
        + \sum_{j \in L,\sigma'} a_{j\sigma'} \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime *} \\
    &\ + \sum_{k\in *,jl \in L, \sigma'} v_{ijkl,\sigma\sigma'} \hat{A}_{jl,\sigma\sigma'}^{\dagger} a_{k\sigma'}^\dagger
        + \sum_{j\in *, kl \in L,\sigma'} v_{ijkl,\sigma\sigma'} \hat{B}'_{kl,\sigma'\sigma'} a_{j\sigma}
        - \sum_{l \in *,jk \in L,\sigma'} v_{ijkl,\sigma\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime} a_{l\sigma'} \\

.. math::
    \hat{R}_{i\sigma}^{\prime L*,CN}
        =&\ \hat{R}_{i\sigma}^{\prime L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_{i\sigma}^{\prime *}
        + \sum_{k\in L,jl \in *, \sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger \hat{A}_{jl,\sigma\sigma'}^\dagger
        + \sum_{j\in L,kl \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{j\sigma} \hat{B}'_{kl,\sigma'\sigma'} \\
        &\ - \sum_{l \in L,jk \in *,\sigma'} v_{ijkl,\sigma\sigma'} a_{l\sigma'} \hat{B}_{kj,\sigma'\sigma}^{\prime}
        + \sum_{k\in *,\sigma'} \hat{P}_{ik,\sigma\sigma'}^L a_{k\sigma'}^\dagger
        + \sum_{j \in *,\sigma'} \hat{Q}_{ij,\sigma\sigma'}^{\prime L} a_{j\sigma'}
