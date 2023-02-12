
Sum MPO Formalism in Unrestricted Spatial Orbitals
==================================================

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

Derivation
----------

Sum of MPO

.. math::
    \hat{H} = \sum_{m\sigma} a_{m\sigma}^\dagger \hat{H}_{m\sigma} =
        \sum_{m\sigma} a_{m\sigma}^\dagger \left[ \sum_{j} t_{mj,\sigma} \ a_{j\sigma}
    + \frac{1}{2} \sum_{jkl, \sigma'} v_{mjkl, \sigma\sigma'}\
     a_{k\sigma'}^\dagger a_{l\sigma'}a_{j\sigma} \right]

Now consider :math:`LR` partition. There are 8 possibilities: :math:`LLL, LRR, RLR, RRL, LLR, LRL, RLL, RRR`.

.. math::
    \hat{H}_{m\sigma} =&\ 
        \left[ \sum_{j \in L} t_{mj,\sigma} \ a_{j\sigma}
        + \frac{1}{2} \sum_{jkl\in L, \sigma'} v_{mjkl, \sigma\sigma'} \ a_{k\sigma'}^\dagger a_{l\sigma'}a_{j\sigma}
         \right]
        + \left[  \sum_{j \in R} t_{mj,\sigma} \ a_{j\sigma}
        + \frac{1}{2} \sum_{jkl\in R, \sigma'} v_{mjkl, \sigma\sigma'} \ a_{k\sigma'}^\dagger a_{l\sigma'}a_{j\sigma} \right] \\
        +&\ \left[ \frac{1}{2} \sum_{j \in L} a_{j\sigma} \sum_{kl\in R, \sigma'} v_{mjkl, \sigma\sigma'}\
        a_{k\sigma'}^\dagger a_{l\sigma'}
        + \frac{1}{2} \sum_{k \in L, \sigma'} a_{k\sigma'}^\dagger \sum_{jl \in R} v_{mjkl, \sigma\sigma'}\
             a_{l\sigma'}a_{j\sigma}
        -\frac{1}{2} \sum_{l \in L, \sigma'} a_{l\sigma'} \sum_{jk\in R} v_{mjkl, \sigma\sigma'}\
     a_{k\sigma'}^\dagger a_{j\sigma} 
             \right]\\ 
        +&\ \left[ \frac{1}{2} \sum_{j\in R} \left( \sum_{kl \in L, \sigma'} v_{mjkl, \sigma\sigma'}\
     a_{k\sigma'}^\dagger a_{l\sigma'} \right) a_{j\sigma}
     + \frac{1}{2} \sum_{k\in R, \sigma'} \left( \sum_{jl \in L} v_{mjkl, \sigma\sigma'}\
      a_{l\sigma'}a_{j\sigma}  \right) a_{k\sigma'}^\dagger
      - \frac{1}{2} \sum_{l\in R, \sigma'} \left( \sum_{jk \in L} v_{mjkl, \sigma\sigma'}\
     a_{k\sigma'}^\dagger a_{j\sigma} \right) a_{l\sigma'}
     \right]

Let

.. math::
    \hat{H}^{L/R}_{m\sigma} =&\ \sum_{j \in L/R} t_{mj,\sigma} \ a_{j\sigma}
        + \frac{1}{2} \sum_{jkl\in L/R, \sigma'} v_{mjkl, \sigma\sigma'} \ a_{k\sigma'}^\dagger a_{l\sigma'}a_{j\sigma} \\
    \hat{P}_{ik,\sigma\sigma'}^{L/R} =&\ \sum_{jl\in L/R} v_{ijkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma}, \\
    \hat{Q}_{ij,\sigma}^{L/R} =&\ \sum_{kl\in L/R,\sigma'} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma'}, \\
    {\hat{Q}}_{il,\sigma\sigma'}^{\prime L/R} =&\ \sum_{jk\in L/R} v_{ijkl,\sigma\sigma'} a_{k\sigma'}^\dagger a_{j\sigma} \\
    {\hat{Q}}_{ij,\sigma\sigma'}^{\prime\prime R} =&\ \delta_{\sigma\sigma'} \hat{Q}^{R}_{ij\sigma}
        - \hat{Q}^{\prime R}_{ij\sigma\sigma'}

we have

.. math::
    \hat{H}_{m\sigma} =&\ \hat{H}^{L}_{m\sigma} \otimes \hat{1}^R + \hat{1}^L \otimes \hat{H}^{R}_{m\sigma}
        + \frac{1}{2} \sum_{j \in L} a_{j\sigma} \hat{Q}_{mj,\sigma}^{R}
        + \frac{1}{2} \sum_{k \in L, \sigma'} a_{k\sigma'}^\dagger  \hat{P}_{mk,\sigma\sigma'}^{R}
        - \frac{1}{2} \sum_{l \in L, \sigma'} a_{l\sigma'} {\hat{Q}}_{ml,\sigma\sigma'}^{\prime R}
        + \frac{1}{2} \sum_{j \in R} \hat{Q}_{mj,\sigma}^{L} a_{j\sigma}
        + \frac{1}{2} \sum_{k \in R, \sigma'} \hat{P}_{mk,\sigma\sigma'}^{L} a_{k\sigma'}^\dagger
        - \frac{1}{2} \sum_{l \in R, \sigma'} {\hat{Q}}_{ml,\sigma\sigma'}^{\prime L} a_{l\sigma'} \\
        =&\ \hat{H}^{L}_{m\sigma} \otimes \hat{1}^R + \hat{1}^L \otimes \hat{H}^{R}_{m\sigma}
        + \frac{1}{2} \sum_{k \in L, \sigma'} a_{k\sigma'}^\dagger  \hat{P}_{mk,\sigma\sigma'}^{R}
        + \frac{1}{2} \sum_{j \in L, \sigma'} a_{j\sigma'}
            \left( \delta_{\sigma\sigma'} \hat{Q}_{mj,\sigma}^{R} - {\hat{Q}}_{mj,\sigma\sigma'}^{\prime R} \right)
        + \frac{1}{2} \sum_{k \in R, \sigma'} \hat{P}_{mk,\sigma\sigma'}^{L} a_{k\sigma'}^\dagger
        + \frac{1}{2} \sum_{j \in R, \sigma'}
            \left( \delta_{\sigma\sigma'} \hat{Q}_{mj,\sigma}^{L} - {\hat{Q}}_{mj,\sigma\sigma'}^{\prime L} \right)
            a_{j\sigma'} \\
        =&\ \hat{H}^{L}_{m\sigma} \otimes \hat{1}^R + \hat{1}^L \otimes \hat{H}^{R}_{m\sigma}
        + \frac{1}{2} \sum_{k \in L, \sigma'} a_{k\sigma'}^\dagger  \hat{P}_{mk,\sigma\sigma'}^{R}
        + \frac{1}{2} \sum_{j \in L, \sigma'} a_{j\sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R}
        + \frac{1}{2} \sum_{k \in R, \sigma'} \hat{P}_{mk,\sigma\sigma'}^{L} a_{k\sigma'}^\dagger
        + \frac{1}{2} \sum_{j \in R, \sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L} a_{j\sigma'}

Now consider :math:`m \in L` or :math:`m \in R`. For :math:`m \in L`:

.. math::
     \sum_{m\in L, \sigma} a_{m\sigma}^\dagger \hat{H}_{m\sigma} =&\
    \left( \sum_{m\in L, \sigma} a_{m\sigma}^\dagger \hat{H}^L_{m\sigma} \right) \otimes \hat{1}^R
     + \sum_{m\in  L, \sigma} a_{m\sigma}^\dagger  \otimes \hat{H}^{R}_{m\sigma} \\
     +&\ \frac{1}{2} \sum_{mk \in L, \sigma\sigma'} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger  \hat{P}_{mk,\sigma\sigma'}^{R}
     + \frac{1}{2} \sum_{mj \in L, \sigma\sigma'} a_{m\sigma}^\dagger a_{j\sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R}
     + \frac{1}{2} \sum_{k \in R, \sigma'} \left( \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{L} \right)
        a_{k\sigma'}^\dagger
    + \frac{1}{2} \sum_{j \in R, \sigma'} \left( \sum_{m\in L,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L}
        \right) a_{j\sigma'} \\
    =&\ \hat{H}^{ML} \otimes \hat{1}^R + \sum_{m\in  L, \sigma} a_{m\sigma}^\dagger  \otimes \hat{H}^{R}_{m\sigma}
        + \frac{1}{2} \sum_{mk \in L, \sigma\sigma'}  \hat{A}_{mk,\sigma\sigma'}  \hat{P}_{mk,\sigma\sigma'}^{R}
     + \frac{1}{2} \sum_{mj \in L, \sigma\sigma'}  \hat{B}_{mj,\sigma\sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R}
     + \frac{1}{2} \sum_{k \in R, \sigma'} \hat{P}_{k\sigma'}^{ML} a_{k\sigma'}^\dagger
     + \frac{1}{2} \sum_{j \in R, \sigma'} \hat{Q}_{j\sigma'}^{ML} a_{j\sigma'}

where

.. math::
    \hat{A}_{ik,\sigma\sigma'} =&\ a_{i\sigma}^\dagger a_{k\sigma'}^\dagger, \\
    \hat{B}_{il,\sigma\sigma'} =&\ a_{i\sigma}^\dagger a_{l\sigma'}, \\
    \hat{H}^{ML/R} =&\ \sum_{m\in L/R, \sigma} a_{m\sigma}^\dagger \hat{H}^{L/R}_{m\sigma} \\
    \hat{P}_{k\sigma'}^{ML/R} =&\ \sum_{m\in L/R,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{L/R} \\
    \hat{Q}_{j\sigma'}^{ML/R} =&\ \sum_{m\in L/R,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L/R}

For :math:`m \in R`:

.. math::
    \sum_{m\in R, \sigma} a_{m\sigma}^\dagger \hat{H}_{m\sigma} =&\
        -\sum_{m \in R,\sigma} \hat{H}^{L}_{m\sigma} \otimes a_{m\sigma}^\dagger
        + \hat{1}^L \otimes \left( \sum_{m \in R,\sigma} a_{m\sigma}^\dagger \hat{H}^{R}_{m\sigma} \right) \\
      -&\ \frac{1}{2} \sum_{k \in L, \sigma'} a_{k\sigma'}^\dagger
        \left( \sum_{m \in R,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{R} \right)
        - \frac{1}{2} \sum_{j \in L, \sigma'} a_{j\sigma'}
        \left( \sum_{m \in R,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R} \right)
        + \frac{1}{2} \sum_{mk \in R, \sigma\sigma'} \hat{P}_{mk,\sigma\sigma'}^{L} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger
        + \frac{1}{2} \sum_{mj \in R, \sigma\sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L} a_{m\sigma}^\dagger a_{j\sigma'} \\
    =&\ -\sum_{m \in R,\sigma} \hat{H}^{L}_{m\sigma} \otimes a_{m\sigma}^\dagger + \hat{1}^L \otimes \hat{H}^{MR}
    - \frac{1}{2} \sum_{k \in L, \sigma'} a_{k\sigma'}^\dagger \hat{P}_{k,\sigma'}^{MR}
        - \frac{1}{2} \sum_{j \in L, \sigma'} a_{j\sigma'} {\hat{Q}}_{j,\sigma'}^{MR}
        + \frac{1}{2} \sum_{mk \in R, \sigma\sigma'} \hat{P}_{mk,\sigma\sigma'}^{L} \hat{A}_{mk,\sigma\sigma'}
        + \frac{1}{2} \sum_{mj \in R, \sigma\sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L} \hat{B}_{mj,\sigma\sigma'}

In summary

.. math::
    \hat{H} =&\ \hat{H}^{ML} \otimes \hat{1}^R + \sum_{m\in  L, \sigma} a_{m\sigma}^\dagger  \otimes \hat{H}^{R}_{m\sigma}
        + \frac{1}{2} \sum_{mj \in L, \sigma\sigma'}  \hat{A}_{mj,\sigma\sigma'}  \hat{P}_{mj,\sigma\sigma'}^{R}
     + \frac{1}{2} \sum_{mj \in L, \sigma\sigma'}  \hat{B}_{mj,\sigma\sigma'} {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R}
     + \frac{1}{2} \sum_{k \in R, \sigma'} \hat{P}_{k\sigma'}^{ML} a_{k\sigma'}^\dagger
     + \frac{1}{2} \sum_{k \in R, \sigma'} \hat{Q}_{k\sigma'}^{ML} a_{k\sigma'} \\
     -&\ \sum_{n \in R,\sigma} \hat{H}^{L}_{n\sigma} \otimes a_{n\sigma}^\dagger + \hat{1}^L \otimes \hat{H}^{MR}
    - \frac{1}{2} \sum_{j \in L, \sigma'} a_{j\sigma'}^\dagger \hat{P}_{j,\sigma'}^{MR}
        - \frac{1}{2} \sum_{j \in L, \sigma'} a_{j\sigma'} {\hat{Q}}_{j,\sigma'}^{MR}
        + \frac{1}{2} \sum_{nk \in R, \sigma\sigma'} \hat{P}_{nk,\sigma\sigma'}^{L} \hat{A}_{nk,\sigma\sigma'}
        + \frac{1}{2} \sum_{nk \in R, \sigma\sigma'} {\hat{Q}}_{nk,\sigma\sigma'}^{\prime\prime L} \hat{B}_{nk,\sigma\sigma'}

The operators required in left block are

.. math::
    \big\{ \hat{H}^{ML}, a_{m\sigma}^\dagger, \hat{A}_{mj,\sigma\sigma'}, \hat{B}_{mj,\sigma\sigma'},
        \hat{P}_{k\sigma'}^{ML}, \hat{Q}_{k\sigma'}^{ML},
        \hat{H}^{L}_{n\sigma} ,\hat{1}^L, a_{j\sigma'}^\dagger, a_{j\sigma'},
        \hat{P}_{nk,\sigma\sigma'}^{L}, {\hat{Q}}_{nk,\sigma\sigma'}^{\prime\prime L} \big\} \quad (m,j\in L, \ n,k \in R)

The total number of operators is

.. math::
    N =&\ 1 + 2 K_{ML} + 4 K_{ML} K_{L} + 4 K_{ML} K_{L} + 2 K_{R} + 2 K_{R}
      + 2 K_{MR} + 1 + 2 K_{L} + 2 K_{L} + 4 K_{MR} K_{R} + 4 K_{MR} K_{R} \\
      =&\ 2 + 2 K_M + 4 K + 8 K_{ML} K_{L} + 8 K_{MR} K_{R}

Reordered left and right block operators

.. math::
    L =&\ \big\{ \hat{H}^{ML}, \hat{1}^L, a_{m\sigma}^\dagger, \hat{H}^{L}_{n\sigma} ,a_{j\sigma'}^\dagger, a_{j\sigma'},
        \hat{P}_{k\sigma'}^{ML}, \hat{Q}_{k\sigma'}^{ML},
        \hat{A}_{mj,\sigma\sigma'}, \hat{B}_{mj,\sigma\sigma'},
        \hat{P}_{nk,\sigma\sigma'}^{L}, {\hat{Q}}_{nk,\sigma\sigma'}^{\prime \prime L} \big\} \quad (m,j\in L, \ n,k \in R) \\
    R =&\ \big\{ \hat{1}^R, \hat{H}^{MR}, \hat{H}^{R}_{m\sigma}, a_{n\sigma}^\dagger,
        \hat{P}_{j,\sigma'}^{MR}, {\hat{Q}}_{j,\sigma'}^{MR},  a_{k\sigma'}^\dagger,  a_{k\sigma'},
        \hat{P}_{mj,\sigma\sigma'}^{R}, {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R},
        \hat{A}_{nk,\sigma\sigma'}, \hat{B}_{nk,\sigma\sigma'} \big \}

Now let

.. math::
    \hat{R}_{k\sigma}^{ML/R} =&\ -2 \delta(k\in M) \hat{H}^{L/R}_{k\sigma} + \hat{P}_{k\sigma'}^{ML/R} \\
    \hat{S}_{k\sigma}^{ML/R} =&\ \hat{Q}_{k\sigma'}^{ML/R}

we have

.. math::
    L =&\ \big\{ \hat{H}^{ML}, \hat{1}^L, a_{j\sigma'}^\dagger, a_{j\sigma'},
        \hat{R}_{k\sigma'}^{ML}, \hat{S}_{k\sigma'}^{ML},
        \hat{A}_{mj,\sigma\sigma'}, \hat{B}_{mj,\sigma\sigma'},
        \hat{P}_{nk,\sigma\sigma'}^{L}, {\hat{Q}}_{nk,\sigma\sigma'}^{\prime \prime L} \big\} \quad (m,j\in L, \ n,k \in R) \\
    R =&\ \big\{ \hat{1}^R, \hat{H}^{MR},
        \hat{R}_{j,\sigma'}^{MR}, \hat{S}_{j,\sigma'}^{MR},  a_{k\sigma'}^\dagger,  a_{k\sigma'},
        \hat{P}_{mj,\sigma\sigma'}^{R}, {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime R},
        \hat{A}_{nk,\sigma\sigma'}, \hat{B}_{nk,\sigma\sigma'} \big \}
    
The total number of operators is

.. math::
    N = 2 + 4 K + 8 K_{ML} K_{L} + 8 K_{MR} K_{R}

Blocking
--------

.. math::
    \hat{P}_{k\sigma'}^{ML*} =&\ \sum_{m\in L*,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{L*}
        = \sum_{m\in L*,\sigma} a_{m\sigma}^\dagger \sum_{jl\in L*} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma} \\
        =&\ \hat{P}_{k\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{P}_{k\sigma'}^{M*}
        + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{L}
        + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \sum_{j\in *, l\in L} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma}
        + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \sum_{j\in L, l\in *} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma} \\
        &\ + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{*}
        + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger \sum_{j\in *, l\in L} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma}
        + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger \sum_{j\in L, l\in *} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma} \\
        =&\ \hat{P}_{k\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{P}_{k\sigma'}^{M*}
        + \sum_{m\in *,\sigma} \hat{P}_{mk,\sigma\sigma'}^{L} a_{m\sigma}^\dagger
        + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{*}
        + \sum_{ml\in L,j\in *,\sigma} v_{mjkl,\sigma\sigma'} a_{m\sigma}^\dagger a_{l\sigma'} a_{j\sigma}
        - \sum_{mj\in L,l\in *,\sigma} v_{mjkl,\sigma\sigma'} a_{m\sigma}^\dagger a_{j\sigma} a_{l\sigma'} \\
        &\ - \sum_{mj\in *,l\in L,\sigma} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{m\sigma}^\dagger a_{j\sigma}
        + \sum_{ml\in *,j\in L,\sigma} v_{mjkl,\sigma\sigma'} a_{j\sigma} a_{m\sigma}^\dagger a_{l\sigma'} \\
        =&\ \hat{P}_{k\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{P}_{k\sigma'}^{M*}
        + \sum_{m\in *,\sigma} \hat{P}_{mk,\sigma\sigma'}^{L} a_{m\sigma}^\dagger
        + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{*}
        + \sum_{ml\in L,j\in *,\sigma} v_{mjkl,\sigma\sigma'}  a_{m\sigma}^\dagger a_{l\sigma'} a_{j\sigma}
        - \sum_{ml\in L,j\in *,\sigma} v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{l\sigma} a_{j\sigma'} \\
        &\ - \sum_{mj\in *,l\in L,\sigma} v_{mjkl,\sigma\sigma'} a_{l\sigma'} a_{m\sigma}^\dagger a_{j\sigma}
        + \sum_{mj\in *,l\in L,\sigma} v_{mlkj,\sigma\sigma'} a_{l\sigma} a_{m\sigma}^\dagger a_{j\sigma'} \\
        =&\ \hat{P}_{k\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{P}_{k\sigma'}^{M*}
        + \sum_{m\in *,\sigma} \hat{P}_{mk,\sigma\sigma'}^{L} a_{m\sigma}^\dagger
        + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger \hat{P}_{mk,\sigma\sigma'}^{*} \\
        &\ + \sum_{ml\in L,j\in *,\sigma} v_{mjkl,\sigma\sigma'} \hat{B}_{ml,\sigma\sigma'} a_{j\sigma}
        - \sum_{ml\in L,j\in *,\sigma} v_{mlkj,\sigma\sigma'} \hat{B}_{ml,\sigma\sigma} a_{j\sigma'}
        + \sum_{mj\in *,l\in L,\sigma} v_{mlkj,\sigma\sigma'} a_{l\sigma} \hat{B}_{mj,\sigma\sigma'}
        - \sum_{mj\in *,l\in L,\sigma} v_{mjkl,\sigma\sigma'} a_{l\sigma'} \hat{B}_{mj,\sigma\sigma}

and

.. math::
    \hat{Q}_{j\sigma'}^{ML*} =&\ \sum_{m\in L*,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L*}
        = \sum_{m\in L*,\sigma} a_{m\sigma}^\dagger 
        \sum_{kl\in L*} \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
        - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \right) \\
        =&\ \hat{Q}_{j\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{j\sigma'}^{M*}
            + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L}
            + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime *} \\
        &\ + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger 
            \sum_{k\in *, l\in L} \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \right)
            + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger 
            \sum_{k\in L, l\in *} \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \right) \\
            &\ + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger 
            \sum_{k\in *, l\in L} \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \right)
            + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger 
            \sum_{k\in L, l\in *} \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \right) \\
        =&\ \hat{Q}_{j\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{j\sigma'}^{M*}
            + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L}
            + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime *} \\
        &\ + \sum_{ml\in L,k\in *, \sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{m\sigma}^\dagger  a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma} \right)
            + \sum_{mk\in L, l\in *,\sigma}
             \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{m\sigma}^\dagger  a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma} \right) \\
            &\ + \sum_{mk\in *, l\in L,\sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{m\sigma}^\dagger  a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma} \right)
            + \sum_{ml\in *,k\in L,\sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{m\sigma}^\dagger  a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma} \right) \\
        =&\ \hat{Q}_{j\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{j\sigma'}^{M*}
            + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L}
            + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime *} \\
        &\ - \sum_{ml\in L,k\in *, \sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{m\sigma}^\dagger a_{l\sigma''} a_{k\sigma''}^\dagger 
            - v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{l\sigma} a_{k\sigma'}^\dagger \right)
            + \sum_{mk\in L, l\in *,\sigma}
             \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{m\sigma}^\dagger  a_{k\sigma''}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma} \right) \\
            &\ + \sum_{mk\in *, l\in L,\sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{l\sigma''} a_{m\sigma}^\dagger a_{k\sigma''}^\dagger 
            - v_{mlkj,\sigma\sigma'} a_{l\sigma} a_{m\sigma}^\dagger a_{k\sigma'}^\dagger \right)
            - \sum_{ml\in *,k\in L,\sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{m\sigma}^\dagger a_{l\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{m\sigma}^\dagger a_{l\sigma} \right)

and

    .. math::
        =&\ \hat{Q}_{j\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{j\sigma'}^{M*}
            + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L}
            + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime *} \\
        &\ - \sum_{ml\in L,k\in *, \sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} \hat{B}_{ml\sigma\sigma''} a_{k\sigma''}^\dagger 
            - v_{mlkj,\sigma\sigma'} \hat{B}_{ml\sigma\sigma} a_{k\sigma'}^\dagger \right)
            + \sum_{mk\in L, l\in *,\sigma}
             \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} \hat{A}_{mk\sigma\sigma''} a_{l\sigma''} 
            - v_{mlkj,\sigma\sigma'} \hat{A}_{mk\sigma\sigma'} a_{l\sigma} \right) \\
        &\ + \sum_{mk\in *, l\in L,\sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{l\sigma''} \hat{A}_{mk\sigma\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{l\sigma} \hat{A}_{mk\sigma\sigma'} \right)
            - \sum_{ml\in *,k\in L,\sigma}
            \left( \delta_{\sigma\sigma'} \sum_{\sigma''} v_{mjkl,\sigma\sigma''} a_{k\sigma''}^\dagger \hat{B}_{ml\sigma\sigma''}
            - v_{mlkj,\sigma\sigma'} a_{k\sigma'}^\dagger \hat{B}_{ml\sigma\sigma} \right)

after simplification

    .. math::
        =&\ \hat{Q}_{j\sigma'}^{ML} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{j\sigma'}^{M*}
            + \sum_{m\in *,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime L}
            + \sum_{m\in L,\sigma} a_{m\sigma}^\dagger {\hat{Q}}_{mj,\sigma\sigma'}^{\prime\prime *} \\
        &\ - \sum_{ml\in L,k\in *, \sigma}
            \left( v_{mjkl,\sigma'\sigma} \hat{B}_{ml\sigma'\sigma} a_{k\sigma}^\dagger 
            - v_{mlkj,\sigma\sigma'} \hat{B}_{ml\sigma\sigma} a_{k\sigma'}^\dagger \right)
            + \sum_{ml\in L, k\in *,\sigma}
             \left( v_{mjlk,\sigma'\sigma} \hat{A}_{ml\sigma'\sigma} a_{k\sigma} 
            - v_{mklj,\sigma\sigma'} \hat{A}_{ml\sigma\sigma'} a_{k\sigma} \right) \\
        &\ + \sum_{mk\in *, l\in L,\sigma}
            \left( v_{mjkl,\sigma'\sigma} a_{l\sigma} \hat{A}_{mk\sigma'\sigma}
            - v_{mlkj,\sigma\sigma'} a_{l\sigma} \hat{A}_{mk\sigma\sigma'} \right)
            - \sum_{mk\in *,l\in L,\sigma}
            \left( v_{mjlk,\sigma'\sigma} a_{l\sigma}^\dagger \hat{B}_{mk\sigma'\sigma}
            - v_{mklj,\sigma\sigma'} a_{l\sigma'}^\dagger \hat{B}_{mk\sigma\sigma} \right)

For :math:`P, Q`, we have

.. math::
    \hat{P}_{ik,\sigma\sigma'}^{L*} =&\ \sum_{jl\in L*} v_{ijkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma}
        = \hat{P}_{ik,\sigma\sigma'}^{L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{P}_{ik,\sigma\sigma'}^{*}
        + \sum_{j\in L, l \in *} v_{ijkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma}
        + \sum_{j\in *, l \in L} v_{ijkl,\sigma\sigma'} a_{l\sigma'} a_{j\sigma} \\
    =&\ \hat{P}_{ik,\sigma\sigma'}^{L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{P}_{ik,\sigma\sigma'}^{*}
        - \sum_{j\in L, l \in *} v_{ijkl,\sigma\sigma'} a_{j\sigma} a_{l\sigma'}
        + \sum_{j\in L, l \in *} v_{ilkj,\sigma\sigma'} a_{j\sigma'} a_{l\sigma} \\
    \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime L*} =&\ \delta_{\sigma\sigma'} \hat{Q}^{L*}_{ij\sigma}
        - \hat{Q}^{\prime L*}_{ij\sigma\sigma'}
        = \delta_{\sigma\sigma'} \sum_{kl\in L*,\sigma''} v_{ijkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
        - \sum_{kl\in L*} v_{ilkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \\
    =&\ \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime *}
        + \delta_{\sigma\sigma'} \sum_{k\in L, l\in *,\sigma''} v_{ijkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
        - \sum_{k\in L, l\in *} v_{ilkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma}
        + \delta_{\sigma\sigma'} \sum_{k\in *, l\in L,\sigma''} v_{ijkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
        - \sum_{k\in *, l\in L} v_{ilkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma} \\
    =&\ \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime L} \otimes \hat{1}^* + \hat{1}^L \otimes \hat{Q}_{ij,\sigma\sigma'}^{\prime\prime *}
        + \delta_{\sigma\sigma'} \sum_{k\in L, l\in *,\sigma''} v_{ijkl,\sigma\sigma''} a_{k\sigma''}^\dagger a_{l\sigma''}
        - \sum_{k\in L, l\in *} v_{ilkj,\sigma\sigma'} a_{k\sigma'}^\dagger a_{l\sigma}
        - \delta_{\sigma\sigma'} \sum_{k\in L, l\in *,\sigma''} v_{ijlk,\sigma\sigma''} a_{k\sigma''} a_{l\sigma''}^\dagger
        + \sum_{k\in L, l\in *} v_{iklj,\sigma\sigma'} a_{k\sigma} a_{l\sigma'}^\dagger
