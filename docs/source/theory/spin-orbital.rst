
DMRG Quantum Chemistry Hamiltonian in Spin Orbitals
===================================================

Hamiltonian
-----------

The quantum chemistry Hamiltonian is written as follows

.. math::
    \hat{H} = \sum_{ij} t_{ij} \ a_i^\dagger a_j
    + \frac{1}{2} \sum_{ijkl} v_{ijkl}\ a_i^\dagger a_k^\dagger a_l a_j

where :math:`ijkl` are spin orbital indices, and

.. math::
    t_{ij} =&\ \int \mathrm{d}\mathbf{x} \
    \phi_i^*(\mathbf{x}) \left( -\frac{1}{2}\nabla^2 - \sum_a \frac{Z_a}{r_a} \right) \phi_j(\mathbf{x}) \\
    v_{ijkl} =&\ \int \mathrm{d} \mathbf{x}_1 \mathrm{d} \mathbf{x}_2 \ \frac{\phi_i^*(\mathbf{x}_1)\phi_k^*(\mathbf{x}_2)
    \phi_l(\mathbf{x}_2)\phi_j(\mathbf{x}_1)}{r_{12}}

Note that here the order of :math:`ijkl` is the same as that in ``FCIDUMP`` (chemist's notation :math:`[ij|kl]`).

When spin index is given, we have

.. math::
    t_{i\sigma,j\tau} =&\ t_{ij}\delta_{\sigma\tau} \\
    v_{i\sigma,j\tau,k\mu,l\nu} =&\ v_{ijkl} \delta_{\sigma\tau} \delta_{\mu\nu}

For complex orbitals, we have

.. math::
    t_{ij} =&\ t_{ji}^* \\
    v_{ijkl} =&\ v_{klij} = v_{jilk}^* = v_{lkji}^*

For real orbitals, we have

.. math::
    t_{ij} =&\ t_{ji} \\
    v_{ijkl} =&\ v_{klij} = v_{jilk} = v_{lkji} = v_{jikl} = v_{ijlk} = v_{lkij} = v_{klji}

Partitioning in Spin Orbitals
-----------------------------

The partitioning of Hamiltonian in left (:math:`L`) and right (:math:`R`) blocks is given by

.. math::
    \hat{H} = \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
     +\Big( \sum_{i\in L} a_i^\dagger \hat{R}_i^{R} + h.c. + \sum_{i\in R} a_i^\dagger \hat{R}_i^{L} + h.c. \Big)
     +\frac{1}{2} \Big( \sum_{ik\in L} \hat{A}_{ik}^L \hat{P}_{ik}^{R} + h.c. \Big)
       + \sum_{ij\in L} \hat{B}_{ij}^L \hat{Q}_{ij}^{R}

where the normal and complementary operators are defined by

.. math::
    \hat{R}_i^{L/R} =&\ \frac{1}{2} \sum_{j\in L/R} t_{ij} a_j + \sum_{jkl\in L/R} v_{ijkl} a_k^\dagger a_l a_j, \\
    \hat{A}_{ik} =&\ a_i^\dagger a_k^\dagger, \\
    \hat{B}_{ij} =&\ a_i^\dagger a_j, \\
    \hat{P}_{ik}^{R} =&\ \sum_{jl\in R} v_{ijkl} a_l a_j, \\
    \hat{Q}_{ij}^{R} =&\ \sum_{kl\in R} \big( v_{ijkl} - v_{ilkj} \big) a_k^\dagger a_l

Note that we need to move all on-site interaction into local Hamiltonian,
so that when construction interaction terms in Hamiltonian,
operators anticommute (without giving extra constant terms).

Derivation
^^^^^^^^^^

First consider one-electron term. :math:`ij` indices have only two possibilities: :math:`i` left, :math:`j` right,
or :math:`i` right, :math:`j` left. Index :math:`i` must be associated with creation operator. So the second case
is the Hermitian conjugate of the first case. Namely, consider :math:`\hat{S}_i^{L/R}` as the one-body part of
:math:`\hat{R}_i^{L/R}`, we have

.. math::
    &\ \Big( \sum_{i\in L} a_i^\dagger \hat{S}_i^{R} + h.c.
        + \sum_{i\in R} a_i^\dagger \hat{S}_i^{L} + h.c. \Big) \\
    =&\ \Big( \sum_{i\in L} a_i^\dagger \hat{S}_i^{R} + \sum_{i\in L} \hat{S}_i^{R\dagger} a_i
        + \sum_{i\in R} a_i^\dagger \hat{S}_i^{L} + \sum_{i\in R} \hat{S}_i^{L\dagger} a_i \Big) \\
    =&\ \frac{1}{2} \Big( \sum_{i\in L,j\in R} t_{ij} a_i^\dagger a_j + \sum_{i\in L,j\in R} t_{ij}^* a_j^\dagger a_i
        + \sum_{i\in R,j \in L} t_{ij} a_i^\dagger a_j + \sum_{i\in R,j\in L}t_{ij}^* a_j^\dagger a_i \Big)

Using :math:`t_{ij}^* = t_{ji}` and swap the indices :math:`ij` we have

.. math::
    \cdots =&\ \frac{1}{2} \Big( \sum_{i\in L,j\in R} t_{ij} a_i^\dagger a_j + \sum_{i\in R,j\in L} t_{ij} a_i^\dagger a_j
        + \sum_{i\in R,j \in L} t_{ij} a_i^\dagger a_j + \sum_{i\in L,j\in R}t_{ij} a_i^\dagger a_j \Big) \\
    =&\ \sum_{i\in L,j\in R} t_{ij} a_i^\dagger a_j  + \sum_{i\in R,j \in L} t_{ij} a_i^\dagger a_j

Next consider one of :math:`ijkl` in left, and three of them in right. These terms are

.. math::
    \hat{H}_{1L,3R} =&\
      \frac{1}{2}\sum_{i\in L, jkl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{j\in L, ikl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{k\in L, ijl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{l\in L, ijk \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j \\
    =&\ \left[
      \frac{1}{2}\sum_{i\in L, jkl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{k\in L, ijl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j \right]
    + \frac{1}{2}\sum_{j\in L, ikl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{l\in L, ijk \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j

where the terms in bracket equal to first and third terms in left-hand-side. Outside the bracket are second, forth
terms.

The conjugate of third term in rhs is second term in rhs

.. math::
      \frac{1}{2}\sum_{j\in L, ikl \in R} v_{ijkl}^* a_j^\dagger a_l^\dagger a_k a_i
    = \frac{1}{2}\sum_{k\in L, ijl \in R} v_{lkji}^* a_k^\dagger a_i^\dagger a_j a_l
    = \frac{1}{2}\sum_{k\in L, ijl \in R} v_{ijkl}   a_i^\dagger a_k^\dagger a_l a_j

The conjugate of forth term in rhs is first term in rhs

.. math::
      \frac{1}{2}\sum_{l\in L, ijk \in R} v_{ijkl}^* a_j^\dagger a_l^\dagger a_k a_i
    = \frac{1}{2}\sum_{i\in L, jkl \in R} v_{lkji}^* a_k^\dagger a_i^\dagger a_j a_l
    = \frac{1}{2}\sum_{i\in L, jkl \in R} v_{ijkl}   a_i^\dagger a_k^\dagger a_l a_j

Therefore, using :math:`v_{ijkl} = v_{klij}`

.. math::
    \hat{H}_{1L,3R} =&\ \left[
      \frac{1}{2}\sum_{i\in L, jkl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{k\in L, ijl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j \right] + h.c. \\
    =&\ \left[
      \frac{1}{2}\sum_{i\in L, jkl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{k\in L, ijl \in R} v_{ijkl} a_k^\dagger a_i^\dagger a_j a_l \right] + h.c. \\
    =&\ \left[
      \frac{1}{2}\sum_{i\in L, jkl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2}\sum_{i\in L, jkl \in R} v_{klij} a_i^\dagger a_k^\dagger a_l a_j \right] + h.c. \\
    =&\ \sum_{i\in L, jkl \in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j + h.c. \\
    =&\ \sum_{i\in L} a_i^\dagger \sum_{jkl \in R} v_{ijkl} a_k^\dagger a_l a_j + h.c.
    =   \sum_{i\in L} a_i^\dagger R_i^{R} + h.c.

Next consider the two creation operators together in left or in together in right. There are two cases.
The second case is the conjugate of the first case, namely,

.. math::
      \sum_{ik\in R, jl \in L} a_i^\dagger a_k^\dagger v_{ijkl} a_l a_j
    = \sum_{jl\in R, ik \in L} a_j^\dagger a_l^\dagger v_{jilk} a_k a_i
    = \sum_{ik \in L, jl\in R} v_{jilk} a_j^\dagger a_l^\dagger a_k a_i
    = \sum_{ik \in L, jl\in R} v_{ijkl}^* \Big( a_i^\dagger a_k^\dagger a_l a_j \Big)^\dagger

This explains the :math:`\hat{A}\hat{P}` term. The last situation is, one creation in left and one creation in right.
Note that when exchange two elementary operators, one creation and one annihilation, one in left and one in right,
they must anticommute.

.. math::
    \hat{H}_{2L,2R} =&\
      \frac{1}{2} \sum_{il\in L, jk\in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2} \sum_{ij\in L, kl\in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2} \sum_{kl\in L, ij\in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j
    + \frac{1}{2} \sum_{jk\in L, il\in R} v_{ijkl} a_i^\dagger a_k^\dagger a_l a_j \\
    =&\
    - \frac{1}{2} \sum_{il\in L, jk\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j
    + \frac{1}{2} \sum_{ij\in L, kl\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
    + \frac{1}{2} \sum_{kl\in L, ij\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
    - \frac{1}{2} \sum_{jk\in L, il\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j

First consider the second and third terms

.. math::
    &\  \frac{1}{2} \sum_{ij\in L, kl\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
      + \frac{1}{2} \sum_{kl\in L, ij\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l \\
    =&\ \frac{1}{2} \sum_{ij\in L, kl\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
      + \frac{1}{2} \sum_{kl\in L, ij\in R} v_{ijkl} a_k^\dagger a_l a_i^\dagger a_j \\
    =&\ \frac{1}{2} \sum_{ij\in L, kl\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
      + \frac{1}{2} \sum_{ij\in L, kl\in R} v_{klij} a_i^\dagger a_j a_k^\dagger a_l \\
    =&\ \sum_{ij\in L, kl\in R} v_{ijkl} a_i^\dagger a_j a_k^\dagger a_l
     = \sum_{ij\in L} a_i^\dagger a_j \sum_{kl\in R} v_{ijkl} a_k^\dagger a_l
     = \sum_{ij\in L} \hat{B}_{ij} \hat{Q}_{ij\prime}^{R}

For the other two terms,

.. math::
    &\ -\frac{1}{2} \sum_{il\in L, jk\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j
       -\frac{1}{2} \sum_{jk\in L, il\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j \\
    =&\ -\frac{1}{2} \sum_{il\in L, jk\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j
        -\frac{1}{2} \sum_{jk\in L, il\in R} v_{ijkl} a_k^\dagger a_j a_i^\dagger a_l \\
    =&\ -\frac{1}{2} \sum_{il\in L, jk\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j
        -\frac{1}{2} \sum_{il\in L, jk\in R} v_{klij} a_i^\dagger a_l a_k^\dagger a_j \\
    =&\ -\sum_{il\in L, jk\in R} v_{ijkl} a_i^\dagger a_l a_k^\dagger a_j \\
    =&\ -\sum_{il\in L} a_i^\dagger a_l \sum_{jk\in R} v_{ijkl} a_k^\dagger a_j
    =  \sum_{il\in L} \hat{B}_{il} \hat{Q}_{il\prime\prime}^{R}

Then

.. math::
     \hat{Q}_{ij}^{R} =  \hat{Q}_{ij\prime}^{R} + \hat{Q}_{ij\prime\prime}^{R}
    = \sum_{kl\in R} \big( v_{ijkl} - v_{ilkj} \big) a_k^\dagger a_l

Normal/Complementary Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above version is used when left block is short in length. Note that all terms should be written in a way that operators
for particles in left block should appear in the left side of operator string, and operators for particles in right block
should appear in the right side of operator string. To write the Hermitian conjugate explicitly, we have

.. math::
    \hat{H}^{NC} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R} \\
    &\ + \sum_{i\in L} \Big( a_i^\dagger \hat{R}_i^{R} - a_i \hat{R}_i^{R\dagger}  \Big)
       + \sum_{i\in R} \Big( \hat{R}_i^{L\dagger} a_i  - \hat{R}_i^{L} a_i^\dagger \Big) \\
    &\ + \frac{1}{2}  \sum_{ik\in L} \Big( \hat{A}_{ik} \hat{P}_{ik}^{R} +
         \hat{A}_{ik}^{\dagger} \hat{P}_{ik}^{R\dagger} \Big)
    + \sum_{ij\in L} \hat{B}_{ij} \hat{Q}_{ij}^{R}

Note that no minus sign for Hermitian conjugate terms with :math:`A, P` because these are not Fermion operators.

With this normal/complementary partitioning, the operators required in left block are

.. math::
    \big\{ \hat{H}^{L}, \hat{1}^L, a_i^\dagger, a_i, \hat{R}_k^{L\dagger},
    \hat{R}_k^{L}, \hat{A}_{ij}, \hat{A}_{ij}^{\dagger}, \hat{B}_{ij} \big\} \quad (i,j\in L, \ k \in R)

The operators required in right block are

.. math::
    \big\{ \hat{1}^{R}, \hat{H}^R, \hat{R}_i^{R}, \hat{R}_i^{R\dagger},
    a_k, a_k^\dagger, \hat{P}_{ij}^R, \hat{P}_{ij}^{R\dagger}, \hat{Q}_{ij}^R \big\} \quad (i,j\in L, \ k \in R)

Assuming that there are :math:`K` sites in total, and :math:`K_L/K_R` sites in left/right block (optimally, :math:`K_L \le K_R`),
the total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{NC} = 1 + 1 + 2 K_L + 2 K_R + 2 K_L^2 + K_L^2 = 3K_L^2 + 2K + 2

Complementary/Normal Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
    \hat{H}^{CN} =&\ \hat{H}^{L} \otimes \hat{1}^{R} + \hat{1}^{L} \otimes \hat{H}^{R}
    + \sum_{i\in L} \Big( a_i^\dagger \hat{R}_i^{R} - a_i \hat{R}_i^{R\dagger} \Big)
    + \sum_{i\in R} \Big( \hat{R}_i^{L\dagger} a_i - \hat{R}_i^{L} a_i^\dagger \Big) \\
    &\ +\frac{1}{2}  \sum_{jl\in R} \Big( \hat{P}_{jl}^{L} \hat{A}_{jl} +
        \hat{P}_{jl}^{L\dagger} \hat{A}_{jl}^{\dagger} \Big)
    + \sum_{kl\in R} \hat{Q}_{kl}^{L} \hat{B}_{kl}

Now the operators required in left block are

.. math::
    \big\{ \hat{H}^L, \hat{1}^{L}, a_i^\dagger, a_i, \hat{R}_k^{L\dagger},
    \hat{R}_k^{L}, \hat{P}_{kl}^L, \hat{P}_{kl}^{L\dagger},
    \hat{Q}_{kl}^L \big\}\quad (k,l\in R, \ i \in L)

The operators required in right block are

.. math::
    \big\{ \hat{1}^R, \hat{H}^{R}, \hat{R}_i^{R}, \hat{R}_i^{R\dagger},
    a_k, a_k^\dagger, \hat{A}_{kl}, \hat{A}_{kl}^{\dagger}, \hat{B}_{kl} \big\}\quad (k,l\in R, \ i \in L)

The total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{CN} = 1 + 1 + 2K_R + 2K_L + 2K_R^2 + K_R^2 = 3K_R^2 + 2K + 2

Blocking
--------

The enlarged left/right block is denoted as :math:`L*/R*`.
Make sure that all :math:`L` operators are to the left of :math:`*` operators.

.. math::
    \hat{R}_i^{L*} =&\ \hat{R}_i^{L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_i^{*}
        + \sum_{j\in L} \left( \sum_{kl \in *} v_{ijkl} a_k^\dagger a_l \right) a_j
        + \sum_{j\in *} \left( \sum_{kl \in L} v_{ijkl} a_k^\dagger a_l \right) a_j \\
        &\ + \sum_{k\in L} a_k^\dagger \left( \sum_{jl \in *} v_{ijkl} a_l a_j \right)
        + \sum_{k\in *} a_k^\dagger \left( \sum_{jl \in L} v_{ijkl} a_l a_j \right)
        - \sum_{l \in L} a_l \left( \sum_{jk \in *} v_{ijkl} a_k^\dagger a_j \right)
        - \sum_{l \in *} a_l \left( \sum_{jk \in L} v_{ijkl} a_k^\dagger a_j \right) \\
        =&\ \hat{R}_i^{ L} \otimes \hat{1}^* + \hat{1}^{L} \otimes \hat{R}_i^{*}
        + \sum_{j\in L} a_j \left( \sum_{kl \in *} v_{ijkl} a_k^\dagger a_l \right)
        + \sum_{j\in *} \left( \sum_{kl \in L} v_{ijkl} a_k^\dagger a_l \right) a_j \\
        &\ + \sum_{k\in L} a_k^\dagger \left( \sum_{jl \in *} v_{ijkl} a_l a_j \right)
        + \sum_{k\in *} \left( \sum_{jl \in L} v_{ijkl} a_l a_j \right) a_k^\dagger
        - \sum_{l \in L} a_l \left( \sum_{jk \in *} v_{ijkl} a_k^\dagger a_j \right)
        - \sum_{l \in *} \left( \sum_{jk \in L} v_{ijkl} a_k^\dagger a_j \right) a_l

Now there are two possibilities. In NC partition, in :math:`L` we have :math:`A,A^\dagger, B, B'`
and in :math:`*` we have :math:`P,P^\dagger,Q, Q'`. In CN partition, the opposite is true. Therefore, we have

.. math::
    \hat{R}_i^{ L*,NC} =&\
        \hat{R}_i^{ L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_i^{ *}
        + \sum_{j\in L} a_j \hat{Q}_{ij}^*
        + \sum_{j\in *, kl \in L} \big( v_{ijkl} - v_{ilkj} \big) \hat{B}_{kl} a_j
         + \sum_{k\in L} a_k^\dagger \hat{P}_{ik}^*
        + \sum_{k\in *,jl \in L, } v_{ijkl} \hat{A}_{jl}^{\dagger} a_k^\dagger \\
    =&\ \hat{R}_i^{ L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_i^{ *}
        + \sum_{k\in L} a_k^\dagger \hat{P}_{ik}^*
        + \sum_{j\in L} a_j \hat{Q}_{ij}^* 
    + \sum_{k\in *,jl \in L, } v_{ijkl} \hat{A}_{jl}^{\dagger} a_k^\dagger
        + \sum_{j\in *, kl \in L} \big( v_{ijkl} - v_{ilkj} \big) \hat{B}_{kl} a_j \\

.. math::
    \hat{R}_i^{ L*,CN} =&\
        \hat{R}_i^{ L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_i^{ *}
        + \sum_{j\in L,kl \in *} \big( v_{ijkl} - v_{ilkj} \big) a_j \hat{B}_{kl}
        + \sum_{j\in *} \hat{Q}_{ij}^{L} a_j
         + \sum_{k\in L,jl \in *, } v_{ijkl} a_k^\dagger \hat{A}_{jl}^\dagger
        + \sum_{k\in *} \hat{P}_{ik}^L a_k^\dagger \\
        =&\ \hat{R}_i^{ L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{R}_i^{ *}
        + \sum_{k\in L,jl \in *, } v_{ijkl} a_k^\dagger \hat{A}_{jl}^\dagger
        + \sum_{j\in L,kl \in *} \big( v_{ijkl} - v_{ilkj} \big) a_j \hat{B}_{kl} 
         + \sum_{k\in *} \hat{P}_{ik}^L a_k^\dagger
        + \sum_{j\in *} \hat{Q}_{ij}^{L} a_j

Similarly,

.. math::
    \hat{R}_i^{ R*,NC}
    =&\ \hat{R}_i^{ *} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{R}_i^{ R}
        + \sum_{k\in *} a_k^\dagger \hat{P}_{ik}^R
        + \sum_{j\in *} a_j \hat{Q}_{ij}^R
        + \sum_{k\in R,jl \in *, } v_{ijkl} \hat{A}_{jl}^{\dagger} a_k^\dagger
        + \sum_{j\in R, kl \in *} \big( v_{ijkl} - v_{ilkj} \big) \hat{B}_{kl} a_j \\
    \hat{R}_i^{ R*,CN}
        =&\ \hat{R}_i^{ *} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{R}_i^{ R}
        + \sum_{k\in *,jl \in R, } v_{ijkl} a_k^\dagger \hat{A}_{jl}^\dagger
        + \sum_{j\in *,kl \in R} \big( v_{ijkl} - v_{ilkj} \big) a_j \hat{B}_{kl}
        + \sum_{k\in R} \hat{P}_{ik}^* a_k^\dagger
        + \sum_{j\in R} \hat{Q}_{ij}^{*} a_j

Number of terms

.. math::
    N_{R,NC} =&\ (2 + 2K_L + 2 K_L^2) K_R + (2 + 2 + 2K_R) K_L = 2K_L^2 K_R + 4 K_L K_R + 2K + 2K_L \\
    N_{R,CN} =&\ (2 + 2K_L + 2) K_R + (2 + 2K_R^2 + 2 K_R) K_L = 2K_R^2 K_L + 4 K_R K_L + 2K + 2K_R

Blocking of other complementary operators is straightforward

.. math::
    \hat{P}_{ik}^{L*,CN} =&\ \hat{P}_{ik}^{L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{P}_{ik}^*
        + \sum_{j\in L,l \in *} v_{ijkl} a_l a_j
        + \sum_{j\in *,l \in L} v_{ijkl} a_l a_j \\
    =&\ \hat{P}_{ik}^{L} \otimes \hat{1}^*
        + \hat{1}^{L} \otimes \hat{P}_{ik}^*
        - \sum_{j\in L,l \in *} v_{ijkl} a_j a_l
        + \sum_{j\in *,l \in L} v_{ijkl} a_l a_j \\
    \hat{P}_{ik}^{R*,NC} =&\ \hat{P}_{ik}^{*} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{P}_{ik}^R
        + \sum_{j\in *,l \in R} v_{ijkl} a_l a_j
        + \sum_{j\in R,l \in *} v_{ijkl} a_l a_j \\
    =&\ \hat{P}_{ik}^{*} \otimes \hat{1}^R
        + \hat{1}^{*} \otimes \hat{P}_{ik}^R
        - \sum_{j\in *,l \in R} v_{ijkl} a_j a_l
        + \sum_{j\in R,l \in *} v_{ijkl} a_l a_j

and

.. math::
    \hat{Q}_{ij}^{L*,CN} =&\ \hat{Q}_{ij}^{L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{ij}^*
        + \sum_{k\in L, l \in *} v_{ijkl} a_k^\dagger a_l
        + \sum_{k\in *, l \in L} v_{ijkl} a_k^\dagger a_l \\
    =&\ \hat{Q}_{ij}^{L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{ij}^*
        + \sum_{k\in L, l \in *} v_{ijkl} a_k^\dagger a_l
        - \sum_{k\in *, l \in L} v_{ijkl} a_l a_k^\dagger  \\
    \hat{Q}_{ij}^{R*,NC} =&\ \hat{Q}_{ij}^{*} \otimes \hat{1}^R + \hat{1}^* \otimes \hat{Q}_{ij}^R
        + \sum_{k\in *, l \in R} v_{ijkl} a_k^\dagger a_l
        + \sum_{k\in R, l \in *} v_{ijkl} a_k^\dagger a_l \\
    =&\ \hat{Q}_{ij}^{*} \otimes \hat{1}^R + \hat{1}^* \otimes \hat{Q}_{ij}^R
        + \sum_{k\in *, l \in R} v_{ijkl} a_k^\dagger a_l
        - \sum_{k\in R, l \in *} v_{ijkl} a_l a_k^\dagger

Middle-Site Transformation
--------------------------

When the sweep is performed from left to right, passing the middle site, we need to switch from NC partition
to CN partition. The cost is :math:`O(K^4/16)`. This happens only once in the sweep. The cost of one blocking procedure is
:math:`O(K_<^2K_>)`, but there are :math:`K` blocking steps in one sweep. So the cost for blocking in one sweep is
:math:`O(KK_<^2K_>)`. Note that the most expensive part in the program should be the Hamiltonian step in Davidson,
which scales as :math:`O(K_<^2)`.

.. math::
    \hat{P}_{ik}^{L,NC\to CN} =&\ \sum_{jl\in L} v_{ijkl} a_l a_j
        = \sum_{jl\in L} v_{ijkl} \hat{A}_{jl}^{\dagger} \\
    \hat{Q}_{ij}^{L,NC\to CN} =&\ \sum_{kl\in L} v_{ijkl} a_k^\dagger a_l
        = \sum_{kl\in L} v_{ijkl} \hat{B}_{kl} \\
