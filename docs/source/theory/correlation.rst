
Diagonal Two-Particle Density Matrix
====================================

PDM Definition
--------------

One-particle density matrix

.. math::
    \langle a_{p\sigma}^\dagger a_{q\tau} \rangle

Two-particle density matrix

.. math::
    \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{r\gamma} a_{s\lambda} \rangle

Spatial one-particle density matrix

.. math::
    E_{pq} \equiv \sum_{\sigma} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle

Spatial two-particle density matrix

.. math::
    e_{pqrs} \equiv \sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{r\tau} a_{s\sigma} \rangle

Spatial two-spin density matrix

.. math::
    s_{pqrs} \equiv \sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}}
        \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{r\tau} a_{s\sigma} \rangle

where

.. math::
    (-1)^{1+\delta_{\sigma\tau}} = \begin{cases} 1 & \sigma = \tau \\ -1 & \sigma \neq \tau \end{cases}

NPC Definition
--------------

Number of particle correlation (pure spin)

.. math::
    \langle n_{p\sigma} n_{q\tau} \rangle = \langle a_{p\sigma}^\dagger a_{p\sigma} a_{q\tau}^\dagger a_{q\tau} \rangle

Number of particle correlation (mixed spin)

.. math::
    \langle a_{p\sigma}^\dagger a_{p\tau} a_{q\tau}^\dagger a_{q\sigma} \rangle

Spin/Charge Correlation
-----------------------

Spin correlation

.. math::
    S_{pq} = \langle (n_{p\alpha} - n_{p\beta}) (n_{q\alpha} - n_{q\beta}) \rangle
        = \langle n_{p\alpha} n_{q\alpha} \rangle - \langle n_{p\alpha} n_{q\beta} \rangle
            - \langle n_{p\beta} n_{q\alpha} \rangle + \langle n_{p\beta} n_{q\beta} \rangle
        = \sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}} \langle n_{p\sigma} n_{q\tau} \rangle

Charge correlation

.. math::
    C_{pq} = \langle (n_{p\alpha} + n_{p\beta}) (n_{q\alpha} + n_{q\beta}) \rangle
        = \langle n_{p\alpha} n_{q\alpha} \rangle + \langle n_{p\alpha} n_{q\beta} \rangle
            + \langle n_{p\beta} n_{q\alpha} \rangle + \langle n_{p\beta} n_{q\beta} \rangle
        = \sum_{\sigma\tau} \langle n_{p\sigma} n_{q\tau} \rangle

Diagonal Spatial Two-Particle Density Matrix (Pure Spin)
--------------------------------------------------------

Using anticommutation relation

.. math::
    a_{q\tau}^\dagger a_{p\sigma} = - a_{p\sigma} a_{q\tau}^\dagger + \delta_{pq}\delta_{\sigma\tau}

We have

.. math::
    \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{q\tau} a_{p\sigma} \rangle
        = -\langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{p\sigma} a_{q\tau} \rangle
        = \langle a_{p\sigma}^\dagger a_{p\sigma} a_{q\tau}^\dagger a_{q\tau} \rangle
            - \delta_{pq} \delta_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{q\tau} \rangle

Then

.. math::
    e_{pqqp} \equiv&\ \sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{q\tau} a_{p\sigma} \rangle
        = -\sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{p\sigma} a_{q\tau} \rangle
        = \sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{p\sigma} a_{q\tau}^\dagger a_{q\tau} \rangle
            - \delta_{pq} \sum_{\sigma} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle \\
        =&\ \sum_{\sigma\tau} \langle n_{p\sigma} n_{q\tau} \rangle
            - \delta_{pq} \sum_{\sigma} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle

Therefore,

.. math::
    \boxed{C_{pq} \equiv \sum_{\sigma\tau} \langle n_{p\sigma} n_{q\tau} \rangle = e_{pqqp} + \delta_{pq} E_{pq}}

Similarly,

.. math::
    s_{pqqp} \equiv&\ \sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}} \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{q\tau} a_{p\sigma} \rangle
        = -\sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}} \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{p\sigma} a_{q\tau} \rangle \\
        =&\ \sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}} \langle a_{p\sigma}^\dagger a_{p\sigma} a_{q\tau}^\dagger a_{q\tau} \rangle
            - \delta_{pq} \sum_{\sigma} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle \\
        =&\ \sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}} \langle n_{p\sigma} n_{q\tau} \rangle
            - \delta_{pq} \sum_{\sigma} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle

Therefore,

.. math::
    \boxed{S_{pq} \equiv \sum_{\sigma\tau} (-1)^{1+\delta_{\sigma\tau}} \langle n_{p\sigma} n_{q\tau} \rangle
        = s_{pqqp} + \delta_{pq} E_{pq} }

Diagonal Spatial Two-Particle Density Matrix (Mixed Spin)
---------------------------------------------------------

Using anticommutation relation

.. math::
    a_{q\tau}^\dagger a_{p\tau} = - a_{p\tau} a_{q\tau}^\dagger + \delta_{pq}

we have

.. math::
    e_{pqpq} \equiv&\ \sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{q\tau}^\dagger a_{p\tau} a_{q\sigma} \rangle
        = -\sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{p\tau} a_{q\tau}^\dagger a_{q\sigma} \rangle
            + \delta_{pq} \sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle \\
        =&\ -\sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{p\tau} a_{q\tau}^\dagger a_{q\sigma} \rangle
            + 2\delta_{pq} \sum_{\sigma} \langle a_{p\sigma}^\dagger a_{q\sigma} \rangle \\

Therefore,

.. math::
    \boxed{\sum_{\sigma\tau} \langle a_{p\sigma}^\dagger a_{p\tau} a_{q\tau}^\dagger a_{q\sigma} \rangle
        = -e_{pqpq} + 2\delta_{pq} E_{pq}}
