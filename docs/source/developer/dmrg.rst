
DMRG Options
============

``me->dot``
-----------

- **Values:** 1 or 2.
- **Meaning:** Select 2-site or 1-site algorithm, unless affected by ``last_site_1site``.

``decomp_last_site``
--------------------

- **Meaning:** If ``false``, decomposition for *affected* sites will be skipped.
- **Default:** ``true``.
- **Affected sites:**
  - For ``me->dot = 1``, only affect site ``n - 1`` for forward and site ``0`` for backward sweep.
  - For ``me->dot = 2`` and ``last_site_1site = true``, only affect site ``n - 1`` for forward sweep.
- **Side effect:** *some* sites will have a canonical form different from the typical one.
- **Restriction:** Only active for ``me->dot = 1`` (or when ``me->dot = 2`` but ``last_site_1site = true``).
- **Accuracy:** Should not affect accuracy.
- **Efficiency:** ``decomp_last_site = false`` provides faster speed.
- **Indicator:** When ``decomp_last_site = false``, the affected site will print ``Mmps = 0``.

``last_site_svd``
-----------------

- **Meaning:** If ``true``, for *affected* sites:
  - Davidson step will be skipped
  - *and* decomposition method will be changed to SVD
  - *and* if ``noise_type = DensityMatrix``, it will be changed to ``Wavefunction``.
- **Default:** ``false``.
- **Affected sites:** only affect site ``n - 1`` for backward sweep.
- **Restriction:** Only active for ``me->dot = 1`` (or when ``me->dot = 2`` but ``last_site_1site = true``).
- **Accuracy:** If ``true``:
  - Skipping davidson step should not affect accuracy.
  - Using SVD instead of density matrix may decrease accuracy.
- **Efficiency:** ``last_site_svd = true`` provides faster speed.
- **Indicator:** When ``last_site_svd = true``, the affected site will print ``Ndav = 0 E = 0.0``.
- **Requirement:** Need ``DMRGSCI``.

``last_site_1site``
-------------------

- **Meaning:** If ``true``, for *affected* sites:
  - In forward sweep, 2-site iteration for sites ``n - 2`` and ``n - 1`` will be changed to 1-site iteration for site ``n - 1``.
  - In backward sweep, 2-site iteration for sites ``n - 2`` and ``n - 1`` will be changed to 1-site iteration for site ``n - 1``.
- **Default:** ``false``.
- **Affected sites:** only affect site ``n - 2`` and ``n - 1`` for forward and backward sweep.
- **Side effect:** MPS bond between site ``n - 2`` and site ``n - 1`` will not be updated in forward sweep.
- **Restriction:** Only active for ``me->dot = 2``.
- **Accuracy:** If ``true``:
  - Accuracy decreased because of 1-site algorithm.
  - Accuracy decreased because of the side effect.
- **Efficiency:** ``last_site_1site = true`` provides faster speed.
- **Indicator:** When ``last_site_1site = true``, the affected site will print ``Site = <n - 1> LAST``.
- **Requirement:** Need ``DMRGSCI``.

Early DMRG stop
-----------------

To stop a DMRG run gracefully, e.g., in case of non-convergence, 
create a file named ``BLOCK_STOP_CALCULATION`` with the text ``STOP``.
The DMRG run will then stop as it would be converged after the current sweep is over.
