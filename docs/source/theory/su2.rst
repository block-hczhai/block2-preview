
Spin-Adapted DMRG Quantum Chemistry Hamiltonian
===============================================

Partitioning in SU(2)
---------------------

The partitioning of Hamiltonian in left (:math:`L`) and right (:math:`R`) blocks is given by

.. math::
    (\hat{H})^{[0]} =&\ \big( \hat{H}^{L} \big)^{[0]} \otimes_{[0]} \big( \hat{1}^{R} \big)^{[0]}
    + \big( \hat{1}^{L} \big)^{[0]} \otimes_{[0]} \big( \hat{H}^{R} \big)^{[0]} \\
    &\ + \sqrt{2} \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right] \\
    &\ + 2 \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right]
    + 2 \sum_{i\in R} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{L} \big)^{[\frac{1}{2}]}
    + h.c. \right] \\
    &\ - \frac{1}{2} \sum_{ik\in L} \left[
    \sqrt{3}
    \big(\hat{A}_{ik} \big)^{[1]} \otimes_{[0]}
    \big(\hat{P}_{ik}^{R} \big)^{[1]}
    + \big(\hat{A}_{ik} \big)^{[0]} \otimes_{[0]}
    \big(\hat{P}_{ik}^{R} \big)^{[0]} + h.c. \right] \\
    &\ +\sum_{ij\in L} \left[
        \big( \hat{B}_{ij} \big)^{[0]} \otimes_{[0]} \left( 2\big( \hat{Q}_{ij}^{R} \big)^{[0]}
        - \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[0]} \right)
        + \sqrt{3} \big( {\hat{B}'}_{ij} \big)^{[1]} \otimes_{[0]} \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[1]}
        \right]

where the normal and complementary operators are defined by

.. math::
    \big( \hat{S}_{i}^{L/R} \big)^{[\frac{1}{2}]} =&\ \sum_{j\in L/R} t_{ij} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    \big( \hat{R}_{i}^{L/R} \big)^{[\frac{1}{2}]} =&\ \sum_{jkl\in L/R} v_{ijkl}
    \left[ \Big( a_{k}^\dagger \Big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
    \otimes_{[\frac{1}{2}]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    \big( \hat{A}_{ik} \big)^{[0/1]} =&\
    \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \\
    \big( \hat{P}_{ik}^{R} \big)^{[0/1]} =&\
        \sum_{jl\in R} v_{ijkl} \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{l} \big)^{[\frac{1}{2}]} \\
    \big( \hat{B}_{ij} \big)^{[0]} =&\
        \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    \big( {\hat{B}'}_{ij} \big)^{[1]} =&\
        \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{j} \big)^{[\frac{1}{2}]}\\
    \big( \hat{Q}_{ij}^{R} \big)^{[0]} =&\
        \sum_{kl\in R} v_{ijkl}
        \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \\
    \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[0/1]} =&\
        \sum_{kl\in R} v_{ilkj}
        \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{l} \big)^{[\frac{1}{2}]} \\
    \big( {\hat{Q}}_{ij}^{\prime \prime R} \big)^{[0]} :=&\
        2 \big( {\hat{Q}}_{ij}^{R} \big)^{[0]} - \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[0]}
    = \sum_{kl\in R} (2v_{ijkl} - v_{ilkj})
        \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]}

Derivation
^^^^^^^^^^

CG Factors
**********

From :math:`j_2 = 1/2` CG factors

.. math::
    \bigg\langle j_1\ \left(M - \frac{1}{2} \right)\ \frac{1}{2}\ \frac{1}{2} \bigg| \left( j_1 \pm \frac{1}{2} \right)\ M
    \bigg\rangle =&\ \pm \sqrt{\frac{1}{2} \left( 1 \pm \frac{M}{j_1 + \frac{1}{2}} \right)} \\
    \bigg\langle j_1\ \left(M + \frac{1}{2} \right)\ \frac{1}{2}\ \left( -\frac{1}{2}\right) \bigg| \left( j_1 \pm \frac{1}{2} \right)\ M
    \bigg\rangle =&\ \sqrt{\frac{1}{2} \left( 1 \mp \frac{M}{j_1 + \frac{1}{2}} \right)}

and symmetry relation

.. math::
    \langle j_1\ m_1\ j_2\ m_2 |J\ M\rangle = (-1)^{j_1+j_2-J} \langle j_2\ m_2\ j_1\ m_1 |J\ M\rangle

and

.. math::
    (-1)^{j_1+\frac{1}{2}-j_1\mp\frac{1}{2}} = (-1)^{\frac{1}{2}\mp\frac{1}{2}} = \pm 1

we have

.. math::
    \bigg\langle \frac{1}{2}\ \frac{1}{2}\ j_1\ \left(M - \frac{1}{2} \right) \bigg| \left( j_1 \pm \frac{1}{2} \right)\ M
    \bigg\rangle =&\ \sqrt{\frac{1}{2} \left( 1 \pm \frac{M}{j_1 + \frac{1}{2}} \right)} \\
    \bigg\langle \frac{1}{2}\ \left( -\frac{1}{2}\right)\ j_1\ \left(M + \frac{1}{2} \right) \bigg| \left( j_1 \pm \frac{1}{2} \right)\ M
    \bigg\rangle =&\ \pm \sqrt{\frac{1}{2} \left( 1 \mp \frac{M}{j_1 + \frac{1}{2}} \right)}

let :math:`j_1 = 1`, we have

.. math::
    \langle \tfrac{1}{2}\ \tfrac{1}{2}\ 1\ (M - \tfrac{1}{2}) | \tfrac{1}{2}\ M \rangle =&\ \sqrt{\tfrac{1}{2} ( 1-\frac{M}{\tfrac{3}{2}} )} \\
    \langle \tfrac{1}{2}\ (-\tfrac{1}{2})\ 1\ (M + \tfrac{1}{2}) | \tfrac{1}{2}\ M \rangle =&\ -\sqrt{\tfrac{1}{2} ( 1+\frac{M}{\tfrac{3}{2}} )}

So the coefficients for :math:`[\tfrac{1}{2}] \otimes_{[\tfrac{1}{2}]} [1]` are

.. math::
    [\tfrac{1}{2} + 0 = \tfrac{1}{2}] = \sqrt{\tfrac{1}{3}},\quad [-\tfrac{1}{2} + 1 = \tfrac{1}{2}] = -\sqrt{\tfrac{2}{3}} \\
    [\tfrac{1}{2} + (-1) = -\tfrac{1}{2}] = \sqrt{\tfrac{2}{3}},\quad [-\tfrac{1}{2} + 0 = -\tfrac{1}{2}] = -\sqrt{\tfrac{1}{3}}

The coefficients for :math:`[1] \otimes_{[\tfrac{1}{2}]} [\tfrac{1}{2}]` are

.. math::
    [0 + \tfrac{1}{2} = \tfrac{1}{2}] = -\sqrt{\tfrac{1}{3}},\quad [1 -\tfrac{1}{2} = \tfrac{1}{2}] = \sqrt{\tfrac{2}{3}} \\
    [(-1) + \tfrac{1}{2} = -\tfrac{1}{2}] = -\sqrt{\tfrac{2}{3}},\quad [0 -\tfrac{1}{2} = -\tfrac{1}{2}] = \sqrt{\tfrac{1}{3}}

This means that the SU(2) operator exchange factor for :math:`[\tfrac{1}{2}] \otimes_{[\tfrac{1}{2}]} [1] \to [1] \otimes_{[\tfrac{1}{2}]} [\tfrac{1}{2}]`
is :math:`-1`. The fermion factor is :math:`+1`. So the overall exchange factor for this case is :math:`-1`.

Tensor Product Formulas
***********************


Singlet

.. math::
    \big(a_p^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_q^\dagger\big)^{[1/2]}
        =&\ \begin{pmatrix} a_{p\alpha}^\dagger \\ a_{p\beta}^\dagger \end{pmatrix}^{[1/2]}
        \otimes_{[0]}
        \begin{pmatrix} a_{q\alpha}^\dagger \\ a_{q\beta}^\dagger \end{pmatrix}^{[1/2]}
        = \frac{1}{\sqrt{2}} \begin{pmatrix} a_{p\alpha}^\dagger a_{q\beta}^\dagger - a_{p\beta}^\dagger a_{q\alpha}^\dagger
        \end{pmatrix}^{[0]} \\
    \big(a_p^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_q\big)^{[1/2]}
        =&\ \begin{pmatrix} a_{p\alpha}^\dagger \\ a_{p\beta}^\dagger \end{pmatrix}^{[1/2]}
        \otimes_{[0]}
        \begin{pmatrix} -a_{q\beta} \\ a_{q\alpha} \end{pmatrix}^{[1/2]}
        = \frac{1}{\sqrt{2}} \begin{pmatrix} a_{p\alpha}^\dagger a_{q\alpha}+ a_{p\beta}^\dagger a_{q\beta}
        \end{pmatrix}^{[0]} \\
    \big(a_p\big)^{[1/2]} \otimes_{[0]} \big(a_q\big)^{[1/2]}
        =&\ \begin{pmatrix} -a_{p\beta} \\ a_{p\alpha} \end{pmatrix}^{[1/2]}
        \otimes_{[0]}
        \begin{pmatrix} -a_{q\beta} \\ a_{q\alpha} \end{pmatrix}^{[1/2]}
        = \frac{1}{\sqrt{2}} \begin{pmatrix} -a_{p\beta} a_{q\alpha} + a_{p\alpha} a_{q\beta}
        \end{pmatrix}^{[0]}

Triplet

.. math::
    \big(a_p^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_q^\dagger\big)^{[1/2]}
        =&\ \begin{pmatrix} a_{p\alpha}^\dagger \\ a_{p\beta}^\dagger \end{pmatrix}^{[1/2]}
        \otimes_{[1]}
        \begin{pmatrix} a_{q\alpha}^\dagger \\ a_{q\beta}^\dagger \end{pmatrix}^{[1/2]}
        = \begin{pmatrix}
            a_{p\alpha}^\dagger a_{q\alpha}^\dagger \\
            \frac{1}{\sqrt{2}} \Big(
                a_{p\alpha}^\dagger a_{q\beta}^\dagger + a_{p\beta}^\dagger a_{q\alpha}^\dagger \Big) \\
            a_{p\beta}^\dagger a_{q\beta}^\dagger
        \end{pmatrix}^{[1]} \\
    \big(a_p^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_q\big)^{[1/2]}
        =&\ \begin{pmatrix} a_{p\alpha}^\dagger \\ a_{p\beta}^\dagger \end{pmatrix}^{[1/2]}
        \otimes_{[1]}
        \begin{pmatrix} -a_{q\beta} \\ a_{q\alpha} \end{pmatrix}^{[1/2]}
        = \begin{pmatrix}
            -a_{p\alpha}^\dagger a_{q\beta} \\
            \frac{1}{\sqrt{2}} \Big(
                a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta} \Big) \\
            a_{p\beta}^\dagger a_{q\alpha}
        \end{pmatrix}^{[1]} \\
    \big(a_p\big)^{[1/2]} \otimes_{[1]} \big(a_q\big)^{[1/2]}
        =&\ \begin{pmatrix} -a_{p\beta} \\ a_{p\alpha} \end{pmatrix}^{[1/2]}
        \otimes_{[1]}
        \begin{pmatrix} -a_{q\beta} \\ a_{q\alpha} \end{pmatrix}^{[1/2]}
        = \begin{pmatrix}
            a_{p\beta} a_{q\beta} \\
            -\frac{1}{\sqrt{2}} \Big( a_{p\beta} a_{q\alpha} + a_{p\alpha} a_{q\beta} \Big) \\
            a_{p\alpha} a_{q\alpha}
        \end{pmatrix}^{[1]}

Doublet times singlet/triplet

.. math::
    U^{[1/2]} = &\ \big(a_p^\dagger\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_r\big)^{[1/2]} \otimes_{[1]} \big(a_s\big)^{[1/2]} \Big]
    = \begin{pmatrix} a_{p\alpha}^\dagger \\ a_{p\beta}^\dagger \end{pmatrix}^{[1/2]} \otimes_{[1/2]} \begin{pmatrix}
            a_{r\beta} a_{s\beta} \\
            -\frac{1}{\sqrt{2}} \Big( a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta} \Big) \\
            a_{r\alpha} a_{s\alpha}
        \end{pmatrix}^{[1]} \\
    =&\ \begin{pmatrix}
        -\frac{1}{\sqrt{2}}\frac{1}{\sqrt{3}} a_{p\alpha}^\dagger \Big( a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta} \Big)
        -\frac{\sqrt{2}}{\sqrt{3}} a_{p\beta}^\dagger a_{r\beta} a_{s\beta} \\
        \frac{\sqrt{2}}{\sqrt{3}} a_{p\alpha}^\dagger a_{r\alpha} a_{s\alpha}
        +\big( -\frac{1}{\sqrt{3}}\big) \big( -\frac{1}{\sqrt{2}} \big) a_{p\beta}^\dagger \Big( a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta} \Big)
         \end{pmatrix}^{[1/2]}
    = \frac{1}{\sqrt{6}} \begin{pmatrix}
        - a_{p\alpha}^\dagger a_{r\beta} a_{s\alpha} - a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta}
        -2 a_{p\beta}^\dagger a_{r\beta} a_{s\beta} \\
        2 a_{p\alpha}^\dagger a_{r\alpha} a_{s\alpha}
        +a_{p\beta}^\dagger a_{r\beta} a_{s\alpha} + a_{p\beta}^\dagger a_{r\alpha} a_{s\beta} \end{pmatrix}^{[1/2]} \\
    V^{[1/2]} =&\ \big(a_p^\dagger\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_r\big)^{[1/2]} \otimes_{[0]} \big(a_s\big)^{[1/2]} \Big]
    = \frac{1}{\sqrt{2}} \begin{pmatrix} a_{p\alpha}^\dagger \\ a_{p\beta}^\dagger \end{pmatrix}^{[1/2]} \otimes_{[1/2]}
        \begin{pmatrix} -a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta}
        \end{pmatrix}^{[0]} \\
    =&\ \frac{1}{\sqrt{2}}
        \begin{pmatrix} -a_{p\alpha}^\dagger a_{r\beta} a_{s\alpha} + a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta}\\
            -a_{p\beta}^\dagger a_{r\beta} a_{s\alpha} + a_{p\beta}^\dagger a_{r\alpha} a_{s\beta}\end{pmatrix}^{[1/2]}

Therefore,

    .. math::
        \sqrt{3} U^{[1/2]} - V^{[1/2]} =&\  \frac{1}{\sqrt{2}} \begin{pmatrix}
        - a_{p\alpha}^\dagger a_{r\beta} a_{s\alpha} - a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta}
        -2 a_{p\beta}^\dagger a_{r\beta} a_{s\beta} \\
        2 a_{p\alpha}^\dagger a_{r\alpha} a_{s\alpha}
        +a_{p\beta}^\dagger a_{r\beta} a_{s\alpha} + a_{p\beta}^\dagger a_{r\alpha} a_{s\beta} \end{pmatrix}^{[1/2]}
        - \frac{1}{\sqrt{2}}
        \begin{pmatrix} -a_{p\alpha}^\dagger a_{r\beta} a_{s\alpha} + a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta}\\
            -a_{p\beta}^\dagger a_{r\beta} a_{s\alpha} + a_{p\beta}^\dagger a_{r\alpha} a_{s\beta}\end{pmatrix}^{[1/2]} \\
        =&\ \frac{1}{\sqrt{2}}
        \begin{pmatrix}
        -a_{p\alpha}^\dagger a_{r\beta} a_{s\alpha} - a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta} -2 a_{p\beta}^\dagger a_{r\beta} a_{s\beta}
        +a_{p\alpha}^\dagger a_{r\beta} a_{s\alpha} - a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta}\\
        2 a_{p\alpha}^\dagger a_{r\alpha} a_{s\alpha} +a_{p\beta}^\dagger a_{r\beta} a_{s\alpha} + a_{p\beta}^\dagger a_{r\alpha} a_{s\beta}
        +a_{p\beta}^\dagger a_{r\beta} a_{s\alpha} - a_{p\beta}^\dagger a_{r\alpha} a_{s\beta}\end{pmatrix}^{[1/2]} \\
        =&\ \sqrt{2}
        \begin{pmatrix}
        - a_{p\alpha}^\dagger a_{r\alpha} a_{s\beta} - a_{p\beta}^\dagger a_{r\beta} a_{s\beta} \\
        a_{p\alpha}^\dagger a_{r\alpha} a_{s\alpha} + a_{p\beta}^\dagger a_{r\beta} a_{s\alpha}
        \end{pmatrix}^{[1/2]}

Another case

.. math::
    S^{[1/2]} = &\ \big(a_r\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_p^\dagger \big)^{[1/2]} \otimes_{[1]} \big(a_q\big)^{[1/2]} \Big]
    = \begin{pmatrix} -a_{r\beta} \\ a_{r\alpha} \end{pmatrix}^{[1/2]} \otimes_{[1/2]}
        \begin{pmatrix}
            -a_{p\alpha}^\dagger a_{q\beta} \\
            \frac{1}{\sqrt{2}} \Big( a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta} \Big) \\
            a_{p\beta}^\dagger a_{q\alpha}
        \end{pmatrix}^{[1]} \\
    =&\ \begin{pmatrix}
        \frac{1}{\sqrt{2}} \frac{1}{\sqrt{3}} (-a_{r\beta}) \Big( a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta} \Big)
        +\frac{\sqrt{2}}{\sqrt{3}} a_{r\alpha} a_{p\alpha}^\dagger a_{q\beta} \\
        -\frac{\sqrt{2}}{\sqrt{3}} a_{r\beta} a_{p\beta}^\dagger a_{q\alpha}
        -\frac{1}{\sqrt{2}} \frac{1}{\sqrt{3}} a_{r\alpha} \Big( a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta} \Big)
        \end{pmatrix}^{[1/2]}
    = \frac{1}{\sqrt{6}} \begin{pmatrix}
        -a_{r\beta} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\beta} a_{p\beta}^\dagger a_{q\beta} +2 a_{r\alpha} a_{p\alpha}^\dagger a_{q\beta}\\
        -2a_{r\beta} a_{p\beta}^\dagger a_{q\alpha} -a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\alpha} a_{p\beta}^\dagger a_{q\beta}
        \end{pmatrix}^{[1/2]} \\
    T^{[1/2]} = &\ \big(a_r\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_p^\dagger \big)^{[1/2]} \otimes_{[0]} \big(a_q\big)^{[1/2]} \Big]
        = \frac{1}{\sqrt{2}} \begin{pmatrix} -a_{r\beta} \\ a_{r\alpha} \end{pmatrix}^{[1/2]} \otimes_{[1/2]}
        \begin{pmatrix} a_{p\alpha}^\dagger a_{q\alpha}+ a_{p\beta}^\dagger a_{q\beta} \end{pmatrix}^{[0]} \\
        =&\ \frac{1}{\sqrt{2}}
        \begin{pmatrix} -a_{r\beta} a_{p\alpha}^\dagger a_{q\alpha} - a_{r\beta}a_{p\beta}^\dagger a_{q\beta} \\
        a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\alpha}a_{p\beta}^\dagger a_{q\beta}\end{pmatrix}^{[1/2]}

Therefore,

.. math::
    \sqrt{3} S^{[1/2]} - T^{[1/2]} =&\
        \frac{1}{\sqrt{6}} \begin{pmatrix}
        -a_{r\beta} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\beta} a_{p\beta}^\dagger a_{q\beta} +2 a_{r\alpha} a_{p\alpha}^\dagger a_{q\beta}\\
        -2a_{r\beta} a_{p\beta}^\dagger a_{q\alpha} -a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\alpha} a_{p\beta}^\dagger a_{q\beta}
        \end{pmatrix}^{[1/2]}-\frac{1}{\sqrt{2}}
        \begin{pmatrix} -a_{r\beta} a_{p\alpha}^\dagger a_{q\alpha} - a_{r\beta}a_{p\beta}^\dagger a_{q\beta} \\
        a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\alpha}a_{p\beta}^\dagger a_{q\beta}\end{pmatrix}^{[1/2]} \\
        =&\ \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            -a_{r\beta} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\beta} a_{p\beta}^\dagger a_{q\beta} +2 a_{r\alpha} a_{p\alpha}^\dagger a_{q\beta}
            +a_{r\beta} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\beta}a_{p\beta}^\dagger a_{q\beta} \\
            -2a_{r\beta} a_{p\beta}^\dagger a_{q\alpha} -a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha} + a_{r\alpha} a_{p\beta}^\dagger a_{q\beta}
            -a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha} - a_{r\alpha}a_{p\beta}^\dagger a_{q\beta}
        \end{pmatrix}^{[1/2]} \\
        =&\ \sqrt{2}
        \begin{pmatrix}
            a_{r\beta}a_{p\beta}^\dagger a_{q\beta} +a_{r\alpha} a_{p\alpha}^\dagger a_{q\beta} \\
            -a_{r\beta} a_{p\beta}^\dagger a_{q\alpha} -a_{r\alpha} a_{p\alpha}^\dagger a_{q\alpha}
        \end{pmatrix}^{[1/2]}

Triplet times triplet

.. math::
    X^{[0]} = &\ \Big[ \big(a_p^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_q^\dagger\big)^{[1/2]} \Big]
    \otimes_{[0]}
    \Big[ \big(a_r\big)^{[1/2]} \otimes_{[1]} \big(a_s\big)^{[1/2]} \Big] \\
    =&\ \begin{pmatrix}
        a_{p\alpha}^\dagger a_{q\alpha}^\dagger \\
        \frac{1}{\sqrt{2}} \Big(
            a_{p\alpha}^\dagger a_{q\beta}^\dagger + a_{p\beta}^\dagger a_{q\alpha}^\dagger \Big) \\
        a_{p\beta}^\dagger a_{q\beta}^\dagger
    \end{pmatrix}^{[1]}
    \otimes_{[0]}
    \begin{pmatrix}
        a_{r\beta} a_{s\beta} \\
        -\frac{1}{\sqrt{2}} \Big( a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta} \Big) \\
        a_{r\alpha} a_{s\alpha}
    \end{pmatrix}^{[1]} \\
    =&\ \frac{1}{\sqrt{3}} \begin{pmatrix}
    a_{p\alpha}^\dagger a_{q\alpha}^\dagger a_{r\alpha} s_{s\alpha}
    + \frac{1}{2} \Big(
            a_{p\alpha}^\dagger a_{q\beta}^\dagger + a_{p\beta}^\dagger a_{q\alpha}^\dagger \Big)
    \Big( a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta} \Big)
    + a_{p\beta}^\dagger a_{q\beta}^\dagger a_{r\beta} a_{s\beta}
    \end{pmatrix} \\
    Y^{[0]} = &\ \Big[ \big(a_p^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_q^\dagger\big)^{[1/2]} \Big]
    \otimes_{[0]}
    \Big[ \big(a_r\big)^{[1/2]} \otimes_{[0]} \big(a_s\big)^{[1/2]} \Big] \\
    =&\ \frac{1}{\sqrt{2}} \begin{pmatrix} a_{p\alpha}^\dagger a_{q\beta}^\dagger - a_{p\beta}^\dagger a_{q\alpha}^\dagger
    \end{pmatrix}^{[0]} \otimes_{[0]}
    \frac{1}{\sqrt{2}} \begin{pmatrix} -a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta}
    \end{pmatrix}^{[0]} \\
    =&\ \frac{1}{2} \Big( a_{p\alpha}^\dagger a_{q\beta}^\dagger - a_{p\beta}^\dagger a_{q\alpha}^\dagger \Big)
    \Big( -a_{r\beta} a_{s\alpha} + a_{r\alpha} a_{s\beta} \Big)

Using

.. math::
    (a+b)(c+d) + (a-b)(-c+d) = (a+b)(2d) -2b(-c+d) = 2 (ad+bc)

we have

.. math::
    \sqrt{3} X^{[0]} + Y^{[0]} =&\
    a_{p\alpha}^\dagger a_{q\alpha}^\dagger a_{r\alpha} s_{s\alpha}
    + a_{p\beta}^\dagger a_{q\beta}^\dagger a_{r\beta} a_{s\beta}
    + a_{p\alpha}^\dagger a_{q\beta}^\dagger a_{r\alpha} a_{s\beta}
    + a_{p\beta}^\dagger a_{q\alpha}^\dagger a_{r\beta} a_{s\alpha} \\
    =&\ \sum_{\sigma\sigma'} a_{p\sigma}^\dagger a_{q\sigma'}^\dagger a_{r\sigma} s_{s\sigma'}

Another case

.. math::
    Z^{[0]} = &\ \Big[ \big(a_p^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_q\big)^{[1/2]} \Big]
    \otimes_{[0]}
    \Big[ \big(a_r^\dagger \big)^{[1/2]} \otimes_{[1]} \big(a_s\big)^{[1/2]} \Big] \\
    =&\ \begin{pmatrix}
        -a_{p\alpha}^\dagger a_{q\beta} \\
        \frac{1}{\sqrt{2}} \Big(
            a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta} \Big) \\
        a_{p\beta}^\dagger a_{q\alpha}
    \end{pmatrix}^{[1]}
    \otimes_{[0]}
    \begin{pmatrix}
        -a_{r\alpha}^\dagger a_{s\beta} \\
        \frac{1}{\sqrt{2}} \Big(
            a_{r\alpha}^\dagger a_{s\alpha} - a_{r\beta}^\dagger a_{s\beta} \Big) \\
        a_{r\beta}^\dagger a_{s\alpha}
    \end{pmatrix}^{[1]} \\
    =&\ \frac{1}{\sqrt{3}} \begin{pmatrix}
    -a_{p\alpha}^\dagger a_{q\beta} a_{r\beta}^\dagger a_{s\alpha}
    -\frac{1}{2} \Big(
            a_{p\alpha}^\dagger a_{q\alpha} - a_{p\beta}^\dagger a_{q\beta} \Big)
        \Big(
            a_{r\alpha}^\dagger a_{s\alpha} - a_{r\beta}^\dagger a_{s\beta} \Big)
    - a_{p\beta}^\dagger a_{q\alpha} a_{r\alpha}^\dagger a_{s\beta}
    \end{pmatrix} \\
    W^{[0]} =&\
    \Big[ \big(a_p^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_q\big)^{[1/2]} \Big]
    \otimes_{[0]}
    \Big[ \big(a_r^\dagger \big)^{[1/2]} \otimes_{[0]} \big(a_s\big)^{[1/2]} \Big] \\
    =&\ \frac{1}{\sqrt{2}} \begin{pmatrix} a_{p\alpha}^\dagger a_{q\alpha}+ a_{p\beta}^\dagger a_{q\beta}
    \end{pmatrix}^{[0]} \otimes_{[0]}
    \frac{1}{\sqrt{2}} \begin{pmatrix} a_{r\alpha}^\dagger a_{s\alpha}+ a_{r\beta}^\dagger a_{s\beta}
    \end{pmatrix}^{[0]} \\
    =&\ \frac{1}{2} \Big( a_{p\alpha}^\dagger a_{q\alpha}+ a_{p\beta}^\dagger a_{q\beta}\Big)
    \Big( a_{r\alpha}^\dagger a_{s\alpha}+ a_{r\beta}^\dagger a_{s\beta} \Big)

Using

.. math::
    (a-b)(c-d) + (a+b)(c+d) = (a+b)(2c) - (2b)(c-d) = 2(ac+bd)

we have

.. math::
    -\sqrt{3} Z^{[0]} + W^{[0]} =&\
     a_{p\alpha}^\dagger a_{q\beta} a_{r\beta}^\dagger a_{s\alpha}
    + a_{p\beta}^\dagger a_{q\alpha} a_{r\alpha}^\dagger a_{s\beta}
    + a_{p\alpha}^\dagger a_{q\alpha} a_{r\alpha}^\dagger a_{s\alpha}
    + a_{p\beta}^\dagger a_{q\beta} a_{r\beta}^\dagger a_{s\beta} \\
    =&\ \sum_{\sigma\sigma'} a_{p\sigma}^\dagger a_{q\sigma'} a_{r\sigma'}^\dagger a_{s\sigma}

S Term
******

From second singlet formula we have

.. math::
    \sqrt{2} \sum_{i\in L} \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{R} \big)^{[\frac{1}{2}]}
        = \sum_{i\in L} \big( t_{ij} a_{i\alpha}^\dagger a_{j\alpha} + t_{ij} a_{i\beta}^\dagger a_{j\beta} \big)

R Term
******

This is the same as the S term. Note that in the expression for :math:`\hat{R}`, we have a :math:`\otimes_{[0]}`,
this is because in the original spatial expression there is a summation over :math:`\sigma`. Then there is a
:math:`[0] \otimes_{[1/2]} [1/2]`, which will not produce any extra coefficients.

AP Term
*******

Using definition

.. math::
    \big( \hat{A}_{ik} \big)^{[0/1]} =&\
    \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \\
    \big( \hat{P}_{ik}^{R} \big)^{[0/1]} =&\
        -\sum_{jl\in R} v_{ijkl} \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{l} \big)^{[\frac{1}{2}]}

We have

.. math::
    &\ \sum_{ik\in L} \left[ \sqrt{3} \big(\hat{A}_{ik} \big)^{[1]} \otimes_{[0]}
    \big(\hat{P}_{ik}^{R} \big)^{[1]} + \big(\hat{A}_{ik} \big)^{[0]} \otimes_{[0]} \big(\hat{P}_{ik}^{R} \big)^{[0]} \right] \\
    =&\ \sum_{ik\in L,jl\in R} v_{ijkl} \left[ \sqrt{3}
    \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]}\right]
    \otimes_{[0]} \left[ \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
    + \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]}\right]
    \otimes_{[0]} \left[ \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
    \right] \\
    =&\ \sum_{ik\in L,jl\in R} v_{ijkl} \left[ \sum_{\sigma\sigma'} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger
        a_{j\sigma} a_{l\sigma'} \right]
    = -\sum_{ik\in L,jl\in R,\sigma\sigma'} v_{ijkl} a_{i\sigma}^\dagger a_{k\sigma'}^\dagger a_{l\sigma'} a_{j\sigma}

Note that in last step, we can anticommute :math:`a_{l\sigma'}, a_{j\sigma}` because it's assumed that in the :math:`\sigma`
summation, when :math:`j=l`, :math:`\sigma \neq \sigma'`. Otherwise there will be two :math:`a` operators acting on the same site
and the contribution is zero.

BQ Term
*******

In spatial expression, this term is :math:`BQ - B'Q'`. Now :math:`-\sqrt{3} Z^{[0]} + W^{[0]}` gives
:math:`B'Q'`. And :math:`2 W^{[0]}` gives :math:`BQ`. Therefore,

.. math::
    2 W^{[0]} - \big(-\sqrt{3} Z^{[0]} + W^{[0]}\big) = \sqrt{3} Z^{[0]} + W^{[0]}

This looks like :math:`\hat{A}\hat{P}` term, but without :math:`\frac{1}{2}` and :math:`h.c.`.
But this is not correct, because the definition of :math:`Q, Q'` is not equivalent due to the index order in
:math:`v_{ijkl}`. So they will give different :math:`W^{[0]}`. Instead we have (note that
:math:`\big( \hat{B}_{ij} \big)^{[0]} = \big( {\hat{B}'}_{ij} \big)^{[0]}`)

.. math::
    &\ \sum_{ij\in L} \left[
        2\Big( \hat{B}_{ij} \Big)^{[0]} \otimes_{[0]} \Big( \hat{Q}_{ij}^{R} \Big)^{[0]}
        - \Big( {\hat{B}'}_{ij} \Big)^{[0]} \otimes_{[0]} \Big( {\hat{Q}'}_{ij}^{R} \Big)^{[0]}
        + \sqrt{3} \Big( {\hat{B}'}_{ij} \Big)^{[1]} \otimes_{[0]} \Big( {\hat{Q}'}_{ij}^{R} \Big)^{[1]}
        \right] \\
    =&\ \sum_{ij\in L} \left[
        \Big( \hat{B}_{ij} \Big)^{[0]} \otimes_{[0]} \left( \Big( 2\hat{Q}_{ij}^{R} \Big)^{[0]}
        - \Big( {\hat{Q}'}_{ij}^{R} \Big)^{[0]} \right)
        + \sqrt{3} \Big( {\hat{B}'}_{ij} \Big)^{[1]} \otimes_{[0]} \Big( {\hat{Q}'}_{ij}^{R} \Big)^{[1]}
        \right]

Note that :math:`B, Q` do not have :math:`[1]` form.

Normal/Complementary Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that

.. math::
    \sqrt{2} \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right]
    = \sqrt{2} \sum_{i\in R} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{L} \big)^{[\frac{1}{2}]}
    + h.c. \right]

Therefore,

.. math::
    &\ \sqrt{2} \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right]
     + 2 \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right]
    + 2 \sum_{i\in R} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{L} \big)^{[\frac{1}{2}]}
    + h.c. \right] \\
    =&\ \frac{\sqrt{2}}{2} \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right]
    + \frac{\sqrt{2}}{2} \sum_{i\in R} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{S}_{i}^{L} \big)^{[\frac{1}{2}]}
    + h.c. \right] \\
    &\ + 2 \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{R} \big)^{[\frac{1}{2}]}
    + h.c. \right]
    + 2 \sum_{i\in R} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{L} \big)^{[\frac{1}{2}]}
    + h.c. \right] \\
    =&\ 2 \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]}
        \Big[ \big( \hat{R}_{i}^{R} \big)^{[\frac{1}{2}]} + \frac{\sqrt{2}}{4}
            \big( \hat{S}_{i}^{R} \big)^{[\frac{1}{2}]} \Big]
    + h.c. \right]
    + 2 \sum_{i\in R} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]}
        \Big[ \big( \hat{R}_{i}^{L} \big)^{[\frac{1}{2}]} + \frac{\sqrt{2}}{4}
            \big( \hat{S}_{i}^{L} \big)^{[\frac{1}{2}]} \Big]
    + h.c. \right]

So define

.. math::
    \big( \hat{R}_{i}^{\prime L/R} \big)^{[\frac{1}{2}]} :=
        \frac{\sqrt{2}}{4} \big( \hat{S}_{i}^{L} \big)^{[\frac{1}{2}]}
        + \big( \hat{R}_{i}^{L} \big)^{[\frac{1}{2}]} =
    \frac{\sqrt{2}}{4} \sum_{j\in L/R} t_{ij} \big( a_{j} \big)^{[\frac{1}{2}]} + \sum_{jkl\in L/R} v_{ijkl}
    \left[ \Big( a_{k}^\dagger \Big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
    \otimes_{[\frac{1}{2}]} \big( a_{j} \big)^{[\frac{1}{2}]}

Here :math:`\frac{\sqrt{2}}{4}` should be understood as :math:`\frac{1}{2} \cdot \frac{1}{\sqrt{2}}`.
The :math:`\frac{1}{2}` is the same as spatial case, and :math:`\frac{1}{\sqrt{2}}`
is because the expected :math:`\sqrt{2}` factor is not added for the :math:`\hat{R}` term.

Operator Exchange factors
*************************

Here we consider fermion and SU(2) exchange factors together. From :math:`j_2 = 1/2` CG factors

.. math::
    \bigg\langle j_1\ \left(M - \frac{1}{2} \right)\ \frac{1}{2}\ \frac{1}{2} \bigg| \left( j_1 \pm \frac{1}{2} \right)\ M
    \bigg\rangle =&\ \pm \sqrt{\frac{1}{2} \left( 1 \pm \frac{M}{j_1 + \frac{1}{2}} \right)} \\
    \bigg\langle j_1\ \left(M + \frac{1}{2} \right)\ \frac{1}{2}\ \left( -\frac{1}{2}\right) \bigg| \left( j_1 \pm \frac{1}{2} \right)\ M
    \bigg\rangle =&\ \sqrt{\frac{1}{2} \left( 1 \mp \frac{M}{j_1 + \frac{1}{2}} \right)}

Let :math:`j_1 = \frac{1}{2}` we have

.. math::
    \bigg\langle \frac{1}{2}\ \left( - \frac{1}{2} \right)\ \frac{1}{2}\ \frac{1}{2} \bigg| \left( \frac{1}{2} \pm \frac{1}{2} \right)\ 0
    \bigg\rangle =&\ \pm \sqrt{\frac{1}{2} } \\
    \bigg\langle \frac{1}{2} \ \frac{1}{2} \ \frac{1}{2}\ \left( -\frac{1}{2}\right) \bigg| \left( \frac{1}{2} \pm \frac{1}{2} \right)\ 0
    \bigg\rangle =&\ \sqrt{\frac{1}{2} }

The exchange factor formula is

.. math::
    \left( \hat{X}_1^{[S_1]} \otimes_{[S]} \hat{X}_2^{[S_2]} \right)^{[S_z]}
        =&\ \sum_{S_{1z},S_{2z}} \hat{X}_1^{[S_1][S_{1z}]} \hat{X}_2^{[S_2][S_{2z}]}
            \langle SS_z| S_1S_{1z},\ S_2 S_{2z} \rangle \\
        =&\ \mathrm{P}_{\mathrm{fermi}}^{\mathrm{exchange}}(N_1,N_2)
            \sum_{S_{1z},S_{2z}} \hat{X}_2^{[S_2][S_{2z}]} \hat{X}_1^{[S_1][S_{1z}]}
            \langle SS_z| S_1S_{1z},\ S_2 S_{2z} \rangle \\
        =&\ \mathrm{P}_{\mathrm{fermi}}^{\mathrm{exchange}}(N_1,N_2)
            \frac{\langle SS_z| S_1S_{1z},\ S_2 S_{2z} \rangle}
            {\langle SS_z| S_2S_{2z},\ S_1 S_{1z} \rangle}
            \left( \hat{X}_2^{[S_2]} \otimes_{[S]} \hat{X}_1^{[S_1]} \right)^{[S_z]} \\
    \hat{X}_1^{[S_1]} \otimes_{[S]} \hat{X}_2^{[S_2]}
        =&\ \mathrm{P}_{\mathrm{fermi}}^{\mathrm{exchange}}(N_1,N_2)
        \mathrm{P}_{\mathrm{SU(2)}}^{\mathrm{exchange}}(S_1, S_2, S)
        \hat{X}_2^{[S_2]} \otimes_{[S]} \hat{X}_1^{[S_1]}

For :math:`[1/2] \otimes_{[0]} [1/2]`, this is

.. math::
    \mathrm{P}^{\mathrm{exchange}}(\tfrac{1}{2}, \tfrac{1}{2}, 0) = (-1) \frac{\big\langle \frac{1}{2} \ \frac{1}{2} \ \frac{1}{2}\ \left( -\frac{1}{2}\right) \big| 0\ 0
    \big\rangle}{\big\langle \frac{1}{2} \ \left( -\frac{1}{2}\right) \ \frac{1}{2}\ \frac{1}{2} \big| 0\ 0
    \big\rangle} = (-1) \frac{\sqrt{\frac{1}{2}}}{-\sqrt{\frac{1}{2}}} = 1

For :math:`[1/2] \otimes_{[1]} [1/2]`, this is

.. math::
    \mathrm{P}^{\mathrm{exchange}}(\tfrac{1}{2}, \tfrac{1}{2}, 1) = (-1) \frac{\big\langle \frac{1}{2} \ \frac{1}{2} \ \frac{1}{2}\ \left( -\frac{1}{2}\right) \big| 1\ 0
    \big\rangle}{\big\langle \frac{1}{2} \ \left( -\frac{1}{2}\right) \ \frac{1}{2}\ \frac{1}{2} \big| 1\ 0
    \big\rangle} = (-1) \frac{\sqrt{\frac{1}{2}}}{\sqrt{\frac{1}{2}}} = -1

From CG factors

.. math::
    \langle 1\ m_1 \ 1 \ (-m_1) | 0 \ 0 \rangle = \frac{(-1)^{1-m_1}}{\sqrt{3}}

we have

.. math::
    \mathrm{P}^{\mathrm{exchange}}(1, 1, 0) = (+1) \frac{\big\langle 1 \ 1 \ \ 1\ -1 \big| 0\ 0
    \big\rangle}{\big\langle 1 \ -1 \ 1\ 1 \big| 0\ 0
    \big\rangle} = (+1) \frac{\frac{(-1)^{0}}{\sqrt{3}}}{\frac{(-1)^{2}}{\sqrt{3}}} = 1

we have

.. math::
    (\hat{H})^{[0], NC} =&\ \big( \hat{H}^{L} \big)^{[0]} \otimes_{[0]} \big( \hat{1}^{R} \big)^{[0]}
    + \big( \hat{1}^{L} \big)^{[0]} \otimes_{[0]} \big( \hat{H}^{R} \big)^{[0]} \\
    &\ + 2 \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{\prime R} \big)^{[\frac{1}{2}]}
    + \big( a_{i}\big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{\prime R\dagger} \big)^{[\frac{1}{2}]} \right]
    + 2 \sum_{i\in R} \left[ \big( \hat{R}_{i}^{\prime L\dagger} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{i} \big)^{[\frac{1}{2}]}
    + \big( \hat{R}_{i}^{\prime L} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{i}^\dagger \big)^{[\frac{1}{2}]}\right] \\
    &\ - \frac{1}{2} \sum_{ik\in L} \left[
    \big(\hat{A}_{ik} \big)^{[0]} \otimes_{[0]} \big(\hat{P}_{ik}^{R} \big)^{[0]}
    + \sqrt{3} \big(\hat{A}_{ik} \big)^{[1]} \otimes_{[0]} \big(\hat{P}_{ik}^{R} \big)^{[1]}
    + \big(\hat{A}_{ik}^\dagger \big)^{[0]} \otimes_{[0]} \big(\hat{P}_{ik}^{R\dagger} \big)^{[0]}
    + \sqrt{3} \big(\hat{A}_{ik}^\dagger \big)^{[1]} \otimes_{[0]} \big(\hat{P}_{ik}^{R\dagger} \big)^{[1]}
    \right] \\
    &\ +\sum_{ij\in L} \left[
        \big( \hat{B}_{ij} \big)^{[0]} \otimes_{[0]} \big( {\hat{Q}}_{ij}^{\prime\prime R} \big)^{[0]}
        + \sqrt{3} \big( {\hat{B}'}_{ij} \big)^{[1]} \otimes_{[0]} \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[1]}
        \right]

With this normal/complementary partitioning, the operators required in left block are

.. math::
    \big\{ \big( \hat{H}^L \big)^{[0]}, \big( \hat{1}^{L} \big)^{[0]}, \big( a_{i}^\dagger \big)^{[\frac{1}{2}]}, \big( a_{i} \big)^{[\frac{1}{2}]},
        \big( \hat{R}_{k}^{\prime L\dagger} \big)^{[\frac{1}{2}]}, \big( \hat{R}_{k}^{\prime L} \big)^{[\frac{1}{2}]},
        \big(\hat{A}_{ij} \big)^{[0]}, \big(\hat{A}_{ij} \big)^{[1]}, \big(\hat{A}_{ij}^\dagger \big)^{[0]}, \big(\hat{A}_{ij}^\dagger \big)^{[1]},
        \big( \hat{B}_{ij} \big)^{[0]}, \big( {\hat{B}'}_{ij} \big)^{[1]}
    \big\}\quad (i,j\in L, k\in R)

The operators required in right block are

.. math::
    \big\{ \big( \hat{1}^{R} \big)^{[0]}, \big( \hat{H}^{R} \big)^{[0]}, \big( \hat{R}_{i}^{\prime R} \big)^{[\frac{1}{2}]},
        \big( \hat{R}_{i}^{\prime R\dagger} \big)^{[\frac{1}{2}]}, \big( a_{k} \big)^{[\frac{1}{2}]}, \big( a_{k}^\dagger \big)^{[\frac{1}{2}]},
        \big(\hat{P}_{ij}^{R} \big)^{[0]}, \big(\hat{P}_{ij}^{R} \big)^{[1]}, \big(\hat{P}_{ij}^{R\dagger} \big)^{[0]},
        \big(\hat{P}_{ij}^{R\dagger} \big)^{[1]}, \big( {\hat{Q}}_{ij}^{\prime\prime R} \big)^{[0]}, \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[1]}
    \big\}\quad (i,j\in L, k\in R)

Assuming that there are :math:`K` sites in total, and :math:`K_L/K_R` sites in left/right block (optimally, :math:`K_L \le K_R`),
the total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{NC} = 1 + 1 + 2K_L + 2K_R + 4K_L^2 + 2K_L^2 = 6K_L^2 + 2K + 2

Complementary/Normal Partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that due the CG factors, exchange any :math:`\otimes_{[0]}` product will not produce extra sign.

.. math::
    (\hat{H})^{[0], CN} =&\ \big( \hat{H}^{L} \big)^{[0]} \otimes_{[0]} \big( \hat{1}^{R} \big)^{[0]}
    + \big( \hat{1}^{L} \big)^{[0]} \otimes_{[0]} \big( \hat{H}^{R} \big)^{[0]} \\
    &\ + 2 \sum_{i\in L} \left[ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{\prime R} \big)^{[\frac{1}{2}]}
    + \big( a_{i}\big)^{[\frac{1}{2}]} \otimes_{[0]} \big( \hat{R}_{i}^{\prime R\dagger} \big)^{[\frac{1}{2}]} \right]
    + 2 \sum_{i\in R} \left[ \big( \hat{R}_{i}^{\prime L\dagger} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{i} \big)^{[\frac{1}{2}]}
    + \big( \hat{R}_{i}^{\prime L} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{i}^\dagger \big)^{[\frac{1}{2}]}\right] \\
    &\ - \frac{1}{2} \sum_{jl\in R} \left[
    \big(\hat{P}_{jl}^{L} \big)^{[0]} \otimes_{[0]} \big(\hat{A}_{jl} \big)^{[0]}
    + \sqrt{3} \big(\hat{P}_{jl}^{L} \big)^{[1]} \otimes_{[0]} \big(\hat{A}_{jl} \big)^{[1]}
    + \big(\hat{P}_{jl}^{L\dagger} \big)^{[0]} \otimes_{[0]} \big(\hat{A}_{jl}^\dagger \big)^{[0]}
    + \sqrt{3} \big(\hat{P}_{jl}^{L\dagger} \big)^{[1]} \otimes_{[0]} \big(\hat{A}_{jl}^\dagger \big)^{[1]}
    \right] \\
    &\ +\sum_{kl\in R} \left[
        \big( {\hat{Q}}_{kl}^{\prime\prime L} \big)^{[0]} \otimes_{[0]} \big( \hat{B}_{kl} \big)^{[0]}
        + \sqrt{3} \big( {\hat{Q}}_{kl}^{\prime L} \big)^{[1]} \otimes_{[0]} \big( {\hat{B}'}_{kl} \big)^{[1]}
        \right]

Now the operators required in left block are

.. math::
    \big\{ \big( \hat{H}^L \big)^{[0]}, \big( \hat{1}^{L} \big)^{[0]}, \big( a_{i}^\dagger \big)^{[\frac{1}{2}]}, \big( a_{i} \big)^{[\frac{1}{2}]},
        \big( \hat{R}_{k}^{\prime L\dagger} \big)^{[\frac{1}{2}]}, \big( \hat{R}_{k}^{\prime L} \big)^{[\frac{1}{2}]},
        \big(\hat{P}_{kl}^{L} \big)^{[0]}, \big(\hat{P}_{kl}^{L} \big)^{[1]}, \big(\hat{P}_{kl}^{L\dagger} \big)^{[0]},
        \big(\hat{P}_{kl}^{L\dagger} \big)^{[1]}, \big( {\hat{Q}}_{kl}^{\prime\prime L} \big)^{[0]}, \big( {\hat{Q}}_{kl}^{\prime L} \big)^{[1]}
    \big\}\quad (k,l\in R, i\in L)

The operators required in right block are

.. math::
    \big\{ \big( \hat{1}^{R} \big)^{[0]}, \big( \hat{H}^{R} \big)^{[0]}, \big( \hat{R}_{i}^{\prime R} \big)^{[\frac{1}{2}]},
        \big( \hat{R}_{i}^{\prime R\dagger} \big)^{[\frac{1}{2}]}, \big( a_{k} \big)^{[\frac{1}{2}]}, \big( a_{k}^\dagger \big)^{[\frac{1}{2}]},
        \big(\hat{A}_{kl} \big)^{[0]}, \big(\hat{A}_{kl} \big)^{[1]}, \big(\hat{A}_{kl}^\dagger \big)^{[0]}, \big(\hat{A}_{kl}^\dagger \big)^{[1]},
        \big( \hat{B}_{kl} \big)^{[0]}, \big( {\hat{B}'}_{kl} \big)^{[1]}
    \big\}\quad (k,l\in R, i\in L)

The total number of operators (and also the number of terms in Hamiltonian with partition)
in left or right block is

.. math::
    N_{CN} = 1 + 1 + 2K_L + 2K_R + 4K_R^2 + 2K_R^2 = 6K_R^2 + 2K + 2

Blocking
--------

The enlarged left/right block is denoted as :math:`L*/R*`.
Make sure that all :math:`L` operators are to the left of :math:`*` operators.
(The exchange factor for this is -1 for doublet :math:`\otimes` triplet and +1 doublet :math:`\otimes` singlet.)

First we have

.. math::
    \big( \hat{R}_{i}^{L/R} \big)^{[1/2]} =&\ \sum_{jkl\in L/R} v_{ijkl}
    \left[ \big( a_{k}^\dagger \big)^{[1/2]} \otimes_{[0]} \big( a_{l} \big)^{[1/2]} \right]
    \otimes_{[1/2]} \big( a_{j} \big)^{[1/2]} \\
    =&\ \frac{1}{\sqrt{2}} \sum_{jkl\in L/R} v_{ijkl} \begin{pmatrix} a_{k\alpha}^\dagger a_{l\alpha}+ a_{k\beta}^\dagger a_{l\beta}
        \end{pmatrix}^{[0]} \otimes_{[1/2]} \big( a_{j} \big)^{[1/2]} \\
    =&\ \frac{1}{\sqrt{2}} \sum_{jkl\in L/R} v_{ijkl} \begin{pmatrix}
        -a_{k\alpha}^\dagger a_{l\alpha}a_{j\beta} - a_{k\beta}^\dagger a_{l\beta}a_{j\beta} \\
        a_{k\alpha}^\dagger a_{l\alpha}a_{j\alpha}+ a_{k\beta}^\dagger a_{l\beta}a_{j\alpha}
        \end{pmatrix}^{[1/2]}

From the formula :math:`\sqrt{3} U^{[1/2]} - V^{[1/2]}` we have

.. math::
    \big( \hat{R}_{i}^{L/R} \big)^{[1/2]} = \frac{\sqrt{3}}{2} \sum_{jkl\in L/R} v_{ijkl}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \Big]
        - \frac{1}{2} \sum_{jkl\in L/R} v_{ijkl}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \Big]

From the formula :math:`\sqrt{3} S^{[1/2]} - T^{[1/2]}` we have (for :math:`k\neq l`)

.. math::
    \big( \hat{R}_{i}^{L/R} \big)^{[1/2]} = \frac{\sqrt{3}}{2} \sum_{jkl\in L/R} v_{ijkl}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \Big]
        - \frac{1}{2} \sum_{jkl\in L/R} v_{ijkl}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \Big[ \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \Big]

We have

.. math::
    \big( \hat{R}_{i}^{\prime L*} \big)^{[1/2]} =&\
        \big( \hat{R}_{i}^{\prime L} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \\
        &\ + \sum_{j \in L}  \left[ \sum_{kl\in *} v_{ijkl} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
          \otimes_{[\frac{1}{2}]} \big( a_{j} \big)^{[\frac{1}{2}]}
        + \sum_{j \in *}  \left[ \sum_{kl\in L} v_{ijkl} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
          \otimes_{[\frac{1}{2}]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
        &\ - \frac{1}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right] \\
        &\ - \frac{1}{2} \sum_{k \in *}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{k \in *}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]\\
        &\ - \frac{1}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]\\
        &\ - \frac{1}{2} \sum_{l\in *}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{l\in *}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right] \\
    =&\ \big( \hat{R}_{i}^{\prime L} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \\
        &\ + \sum_{j \in L}  \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[\frac{1}{2}]}
            \left[ \sum_{kl\in *} v_{ijkl} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
        + \sum_{j \in *}  \left[ \sum_{kl\in L} v_{ijkl} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
          \otimes_{[\frac{1}{2}]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
        &\ - \frac{1}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right] \\
        &\ - \frac{1}{2} \sum_{k \in *} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in *} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ - \frac{1}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]\\
        &\ - \frac{1}{2} \sum_{l\in *} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_l\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in *} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_l\big)^{[1/2]}

After reordering of terms

.. math::
    \big( \hat{R}_{i}^{\prime L*} \big)^{[1/2]} =&\
        \big( \hat{R}_{i}^{\prime L} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \\
        &\ - \frac{1}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right] \\
        &\ + \sum_{j \in L}  \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[\frac{1}{2}]}
            \left[ \sum_{kl\in *} v_{ijkl} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right] \\
        &\ - \frac{1}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]\\
        &\ - \frac{1}{2} \sum_{k \in *} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in *} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ + \sum_{j \in *}  \left[ \sum_{kl\in L} v_{ijkl} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \right]
          \otimes_{[\frac{1}{2}]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
        &\ - \frac{1}{2} \sum_{l\in *} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_l\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in *} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_l\big)^{[1/2]} \\
    =&\ \big( \hat{R}_{i}^{\prime L} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \\
        &\ - \frac{1}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{k \in L}
        \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jl\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right] \\
        &\ + \frac{1}{2} \sum_{j\in L} \big(a_j\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{kl\in *} (2 v_{ijkl} - v_{ilkj}) \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_l\big)^{[1/2]} \right]
        +\frac{\sqrt{3}}{2} \sum_{l\in L}
        \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \left[ \sum_{jk\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]\\
        &\ - \frac{1}{2} \sum_{k \in *} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[0]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in *} \left[ \sum_{jl\in L} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ + \frac{1}{2} \sum_{j\in *} \left[ \sum_{kl\in L} (2v_{ijkl} - v_{ilkj}) \big(a_k^\dagger\big)^{[1/2]} \otimes_{[0]} \big(a_l\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_j\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in *} \left[ \sum_{jk\in L} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1]} \big(a_j\big)^{[1/2]} \right]
        \otimes_{[1/2]} \big(a_l\big)^{[1/2]}

By definition (The overall exchange factor for :math:`[1/2] \otimes_{[0]} [1/2]` is 1, and for :math:`[1/2] \otimes_{[1]} [1/2]` is -1)

.. math::
    \big( \hat{A}_{ik} \big)^{[0/1]} =&\ \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \\
    \big( \hat{A}_{ik}^\dagger \big)^{[0]} =&\ \big( a_{i} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{k} \big)^{[\frac{1}{2}]}
    = \big( a_{k} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{i} \big)^{[\frac{1}{2}]} \\
    \big( \hat{A}_{ik}^\dagger \big)^{[1]} =&\ -\big( a_{i} \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{k} \big)^{[\frac{1}{2}]}
    = \big( a_{k} \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{i} \big)^{[\frac{1}{2}]} \\
    \big( \hat{P}_{ik}^{R} \big)^{[0/1]} =&\
        \sum_{jl\in R} v_{ijkl} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    \big( \hat{B}_{ij} \big)^{[0]} =&\
        \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    \big( {\hat{B}'}_{ij} \big)^{[1]} =&\
        \big( a_{i}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{j} \big)^{[\frac{1}{2}]}\\
    \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[1]} =&\
        \sum_{kl\in R} v_{ilkj}
        \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]} \\
    \big( {\hat{Q}}_{ij}^{\prime \prime R} \big)^{[0]} =&\ \sum_{kl\in R} (2v_{ijkl} - v_{ilkj})
        \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]}

we have

.. math::
    \big( \hat{R}_{i}^{\prime L*,NC} \big)^{[1/2]} =&\
        \big( \hat{R}_{i}^{\prime L} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \\
        &\ - \frac{1}{2} \sum_{k \in L} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{P}_{ik}^{*} \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{k \in L} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{P}_{ik}^{*} \big)^{[1]} \\
        &\ + \frac{1}{2} \sum_{j\in L} \big(a_j\big)^{[1/2]} \otimes_{[1/2]} \big( {\hat{Q}}_{ij}^{\prime \prime *} \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{l\in L} \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \big( {\hat{Q}}_{il}^{\prime *} \big)^{[1]}\\
        &\ - \frac{1}{2} \sum_{k \in *,jl\in L} v_{ijkl} \big( \hat{A}_{jl}^\dagger \big)^{[0]} \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in *,jl\in L} v_{ijkl} \big( \hat{A}_{jl}^\dagger \big)^{[1]} \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ + \frac{1}{2} \sum_{j\in *,kl\in L} (2v_{ijkl} - v_{ilkj}) \big( \hat{B}_{kl} \big)^{[0]} \otimes_{[1/2]} \big(a_j\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in *,jk\in L} v_{ijkl} \big( {\hat{B}'}_{kj} \big)^{[1]} \otimes_{[1/2]} \big(a_l\big)^{[1/2]} \\
    \big( \hat{R}_{i}^{\prime L*,CN} \big)^{[1/2]} =&\
        \big( \hat{R}_{i}^{\prime L} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \\
        &\ - \frac{1}{2} \sum_{k \in L,jl\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{A}_{jl}^\dagger \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{k \in L,jl\in *} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{A}_{jl}^\dagger \big)^{[1]} \\
        &\ + \frac{1}{2} \sum_{j\in L,kl\in *} (2 v_{ijkl} - v_{ilkj}) \big(a_j\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{B}_{kl} \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{l\in L,jk\in *} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \big( {\hat{B}'}_{kj} \big)^{[1]} \\
        &\ - \frac{1}{2} \sum_{k \in *} \big( \hat{P}_{ik}^{L} \big)^{[0]} \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in *} \big( \hat{P}_{ik}^{L} \big)^{[1]} \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ + \frac{1}{2} \sum_{j\in *} \big( {\hat{Q}}_{ij}^{\prime \prime L} \big)^{[0]} \otimes_{[1/2]} \big(a_j\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in *} \big( {\hat{Q}}_{il}^{ \prime L} \big)^{[1]} \otimes_{[1/2]} \big(a_l\big)^{[1/2]}

To generate symmetrized :math:`P`, we need to change the :math:`A` line to the following

.. math::
    - \frac{1}{4} \sum_{k \in *,jl\in L} (v_{ijkl} + v_{ilkj}) \big( \hat{A}_{jl}^\dagger \big)^{[0]} \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{4} \sum_{k \in *,jl\in L} (v_{ijkl} - v_{ilkj}) \big( \hat{A}_{jl}^\dagger \big)^{[1]} \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}

Similarly,

.. math::
    \big( \hat{R}_{i}^{\prime R*,NC} \big)^{[1/2]} =&\
        \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^R \big)^{[0]}
        + \big( \hat{1}^* \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime R} \big)^{[1/2]} \\
        &\ - \frac{1}{2} \sum_{k \in *} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{P}_{ik}^{R} \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{k \in *} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{P}_{ik}^{R} \big)^{[1]} \\
        &\ + \frac{1}{2} \sum_{j\in *} \big(a_j\big)^{[1/2]} \otimes_{[1/2]} \big( {\hat{Q}}_{ij}^{\prime \prime R} \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{l\in *} \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \big( {\hat{Q}}_{il}^{\prime R} \big)^{[1]}\\
        &\ - \frac{1}{2} \sum_{k \in R,jl\in *} v_{ijkl} \big( \hat{A}_{jl}^\dagger \big)^{[0]} \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in R,jl\in *} v_{ijkl} \big( \hat{A}_{jl}^\dagger \big)^{[1]} \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ + \frac{1}{2} \sum_{j\in R,kl\in *} (2v_{ijkl} - v_{ilkj}) \big( \hat{B}_{kl} \big)^{[0]} \otimes_{[1/2]} \big(a_j\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in R,jk\in *} v_{ijkl} \big( {\hat{B}'}_{kj} \big)^{[1]} \otimes_{[1/2]} \big(a_l\big)^{[1/2]} \\
    \big( \hat{R}_{i}^{\prime R*,CN} \big)^{[1/2]} =&\
        \big( \hat{R}_{i}^{\prime *} \big)^{[1/2]} \otimes_{[1/2]} \big( \hat{1}^R \big)^{[0]}
        + \big( \hat{1}^* \big)^{[0]} \otimes_{[1/2]} \big( \hat{R}_{i}^{\prime R} \big)^{[1/2]} \\
        &\ - \frac{1}{2} \sum_{k \in *,jl\in R} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{A}_{jl}^\dagger \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{k \in *,jl\in R} v_{ijkl} \big(a_k^\dagger\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{A}_{jl}^\dagger \big)^{[1]} \\
        &\ + \frac{1}{2} \sum_{j\in *,kl\in R} (2 v_{ijkl} - v_{ilkj}) \big(a_j\big)^{[1/2]} \otimes_{[1/2]} \big( \hat{B}_{kl} \big)^{[0]}
        +\frac{\sqrt{3}}{2} \sum_{l\in *,jk\in R} v_{ijkl} \big(a_l\big)^{[1/2]} \otimes_{[1/2]} \big( {\hat{B}'}_{kj} \big)^{[1]} \\
        &\ - \frac{1}{2} \sum_{k \in R} \big( \hat{P}_{ik}^{*} \big)^{[0]} \otimes_{[1/2]}  \big(a_k^\dagger\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{k \in R} \big( \hat{P}_{ik}^{*} \big)^{[1]} \otimes_{[1/2]} \big(a_k^\dagger\big)^{[1/2]}  \\
        &\ + \frac{1}{2} \sum_{j\in R} \big( {\hat{Q}}_{ij}^{\prime \prime *} \big)^{[0]} \otimes_{[1/2]} \big(a_j\big)^{[1/2]}
        -\frac{\sqrt{3}}{2} \sum_{l\in R} \big( {\hat{Q}}_{il}^{ \prime *} \big)^{[1]} \otimes_{[1/2]} \big(a_l\big)^{[1/2]}

Number of terms

.. math::
    N_{R',NC} =&\ (2 + 4K_L + 4K_L^2) K_R + (2 + 4 + 4K_R) K_L = 4K_L^2 K_R + 8K_L K_R + 2K + 4 K_L \\
    N_{R',CN} =&\ (2 + 4K_L + 4) K_R + (2 + 4K_R^2 + 4K_R) K_L = 4K_R^2 K_L + 8K_R K_L + 2K + 4 K_R

Blocking of other complementary operators is straightforward

.. math::
    \big( \hat{P}_{ik}^{L*,CN} \big)^{[0/1]} =&\ \big( \hat{P}_{ik}^{L} \big)^{[0/1]} \otimes_{[0/1]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[0/1]} \big( \hat{P}_{ik}^{*} \big)^{[0/1]}
        + \sum_{j \in L, l \in *} v_{ijkl} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{j} \big)^{[\frac{1}{2}]}
        + \sum_{j \in *, l \in L} v_{ijkl} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    =&\ \big( \hat{P}_{ik}^{L} \big)^{[0/1]} \otimes_{[0/1]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[0/1]} \big( \hat{P}_{ik}^{*} \big)^{[0/1]}
        \pm \sum_{j \in L, l \in *} v_{ijkl} \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{l} \big)^{[\frac{1}{2}]}
        + \sum_{j \in *, l \in L} v_{ijkl} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{j} \big)^{[\frac{1}{2}]} \\
    \big( \hat{P}_{ik}^{R*,NC} \big)^{[0/1]} =&\ \big( \hat{P}_{ik}^{*} \big)^{[0/1]} \otimes_{[0/1]} \big( \hat{1}^R \big)^{[0]}
        + \big( \hat{1}^* \big)^{[0]} \otimes_{[0/1]} \big( \hat{P}_{ik}^{R} \big)^{[0/1]}
        \pm \sum_{j \in *, l \in R} v_{ijkl} \big( a_{j} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{l} \big)^{[\frac{1}{2}]}
        + \sum_{j \in R, l \in *} v_{ijkl} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{j} \big)^{[\frac{1}{2}]}

and

.. math::
    \big( {\hat{Q}}_{ij}^{\prime \prime L*,CN} \big)^{[0]} =&\ \big( {\hat{Q}}_{ij}^{\prime \prime L} \big)^{[0]} \otimes_{[0]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[0]} \big( {\hat{Q}}_{ij}^{\prime \prime *} \big)^{[0]}
        + \sum_{k\in L, l \in *} (2v_{ijkl} - v_{ilkj}) \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]}
        + \sum_{k\in *, l \in L} (2v_{ijkl} - v_{ilkj}) \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]} \\
    =&\ \big( {\hat{Q}}_{ij}^{\prime \prime L} \big)^{[0]} \otimes_{[0]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[0]} \big( {\hat{Q}}_{ij}^{\prime \prime *} \big)^{[0]}
        + \sum_{k\in L, l \in *} (2v_{ijkl} - v_{ilkj}) \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]}
        + \sum_{k\in *, l \in L} (2v_{ijkl} - v_{ilkj}) \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \\
    \big( {\hat{Q}}_{ij}^{\prime \prime R*,NC} \big)^{[0]} =&\ \big( {\hat{Q}}_{ij}^{\prime \prime *} \big)^{[0]} \otimes_{[0]} \big( \hat{1}^R \big)^{[0]}
        + \big( \hat{1}^* \big)^{[0]} \otimes_{[0]} \big( {\hat{Q}}_{ij}^{\prime \prime R} \big)^{[0]}
        + \sum_{k\in *, l \in R} (2v_{ijkl} - v_{ilkj}) \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]}
        + \sum_{k\in R, l \in *} (2v_{ijkl} - v_{ilkj}) \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]}

and

.. math::
    \big( {\hat{Q}}_{ij}^{\prime L*,CN} \big)^{[1]} =&\ \big( {\hat{Q}}_{ij}^{\prime L} \big)^{[1]} \otimes_{[1]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1]} \big( {\hat{Q}}_{ij}^{\prime *} \big)^{[1]}
        + \sum_{k\in L, l \in *} v_{ilkj} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]}
        + \sum_{k\in *, l \in L} v_{ilkj} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]} \\
    =&\ \big( {\hat{Q}}_{ij}^{\prime L} \big)^{[1]} \otimes_{[1]} \big( \hat{1}^* \big)^{[0]}
        + \big( \hat{1}^L \big)^{[0]} \otimes_{[1]} \big( {\hat{Q}}_{ij}^{\prime *} \big)^{[1]}
        + \sum_{k\in L, l \in *} v_{ilkj} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]}
        - \sum_{k\in *, l \in L} v_{ilkj} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \\
    \big( {\hat{Q}}_{ij}^{\prime R*,CN} \big)^{[1]} =&\ \big( {\hat{Q}}_{ij}^{\prime *} \big)^{[1]} \otimes_{[1]} \big( \hat{1}^R \big)^{[0]}
        + \big( \hat{1}^* \big)^{[0]} \otimes_{[1]} \big( {\hat{Q}}_{ij}^{\prime R} \big)^{[1]}
        + \sum_{k\in *, l \in R} v_{ilkj} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]}
        - \sum_{k\in R, l \in *} v_{ilkj} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]}

Middle-Site Transformation
--------------------------

.. math::
    \big( \hat{P}_{ik}^{L,NC\to CN} \big)^{[0/1]} =&\
        \sum_{jl\in L} v_{ijkl} \big( a_{l} \big)^{[\frac{1}{2}]} \otimes_{[0/1]} \big( a_{j} \big)^{[\frac{1}{2}]}
        = \sum_{jl\in L} v_{ijkl} \big( \hat{A}_{jl}^\dagger \big)^{[0/1]} \\
    \big( {\hat{Q}}_{ij}^{\prime \prime L,NC\to CN} \big)^{[0]} =&\
        \sum_{kl\in R} (2v_{ijkl} - v_{ilkj}) \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[0]} \big( a_{l} \big)^{[\frac{1}{2}]}
        = \sum_{kl\in R} (2v_{ijkl} - v_{ilkj}) \big( \hat{B}_{kl} \big)^{[0]} \\
    \big( {\hat{Q}}_{ij}^{\prime L,NC\to CN} \big)^{[1]} =&\
        \sum_{kl\in R} v_{ilkj} \big( a_{k}^\dagger \big)^{[\frac{1}{2}]} \otimes_{[1]} \big( a_{l} \big)^{[\frac{1}{2}]}
        = \sum_{kl\in R} v_{ilkj} \big( {\hat{B}'}_{kl} \big)^{[1]}
