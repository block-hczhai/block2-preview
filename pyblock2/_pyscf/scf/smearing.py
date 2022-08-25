#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2022 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""
Smearing function adapted from:
    https://github.com/gkclab/libdmet_preview/blob/main/libdmet/routine/pbc_helper.py
    https://github.com/gkclab/libdmet_preview/blob/main/libdmet/routine/mfd.py
    https://github.com/gkclab/libdmet_preview/blob/main/libdmet/routine/ftsystem.py

    Author: Zhihao Cui <zhcui0408@gmail.com>
"""

import numpy as np

FIT_TOL = 1e-12

def fermi_smearing_occ(mu, mo_energy, beta, ncore=0, nvirt=0):
    """
    Fermi smearing function for mo_occ.
    By using broadcast, mu can be a list of values 
    (i.e. each sector can have different mu)
    
    Args:
        mu: chemical potential, can have shape (), (1,), (spin,).
        mo_energy: orbital energy, can be (nmo,) or (s, k, ..., nmo) array.
        beta: inverse temperature, float.
        ncore: number of core orbitals with occupation 1.
        nvirt: number of virt orbitals with occupation 0.

    Returns:
        occ: orbital occupancy, the same shape as mo_energy.
    """
    mo_energy = np.asarray(mo_energy)
    mu = np.asarray(mu).reshape(-1, *([1] * (mo_energy.ndim - 1)))
    de = beta * (mo_energy - mu)
    occ = np.zeros_like(mo_energy)
    idx = de < 100
    if ncore != 0:
        assert mo_energy.ndim == 1
        idx[:ncore] = False
        occ[:ncore] = 1.0
    if nvirt != 0:
        assert mo_energy.ndim == 1
        idx[-nvirt:] = False

    occ[idx] = 1.0 / (np.exp(de[idx]) + 1.0)
    return occ


def gaussian_smearing_occ(mu, mo_energy, beta, ncore=0, nvirt=0):
    """
    Gaussian smearing function for mo_occ.

    Args:
        mu: chemical potential, can have shape (), (1,), (spin,).
        mo_energy: orbital energy, can be (nmo,) or (s, k, ..., nmo) array.
        beta: inverse temperature, float.

    Returns:
        occ: orbital occupancy, the same shape as mo_energy.
    """
    import scipy
    mo_energy = np.asarray(mo_energy)
    mu = np.asarray(mu).reshape(-1, *([1] * (mo_energy.ndim - 1)))
    return 0.5 * scipy.special.erfc((mo_energy - mu) * beta)


def find_mu(
    nelec,
    mo_energy,
    beta,
    mu0=None,
    f_occ=fermi_smearing_occ,
    tol=FIT_TOL,
    ncore=0,
    nvirt=0,
):
    """
    Find chemical potential mu for a target nelec.
    Assume mo_energy has no spin dimension.
    
    Returns:
        mu: chemical potential.
    """
    from scipy.optimize import brentq
    from pyscf.lib import logger

    def nelec_cost_fn_brentq(mu):
        mo_occ = f_occ(mu, mo_energy, beta, ncore=ncore, nvirt=nvirt)
        return mo_occ.sum() - nelec

    nelec_int = int(np.round(nelec))
    if nelec_int >= len(mo_energy):
        lval = mo_energy[-1] - (1.0 / beta)
        rval = mo_energy[-1] + max(10.0, 1.0 / beta)
    elif nelec_int <= 0:
        lval = mo_energy[0] - max(10.0, 1.0 / beta)
        rval = mo_energy[0] + (1.0 / beta)
    else:
        lval = mo_energy[nelec_int - 1] - (1.0 / beta)
        rval = mo_energy[nelec_int] + (1.0 / beta)

    # for the corner case where all empty or all occupied
    if nelec_cost_fn_brentq(lval) * nelec_cost_fn_brentq(rval) > 0:
        lval -= max(100.0, 1.0 / beta)
        rval += max(100.0, 1.0 / beta)
    res = brentq(
        nelec_cost_fn_brentq,
        lval,
        rval,
        xtol=tol,
        rtol=tol,
        maxiter=10000,
        full_output=True,
        disp=False,
    )
    if not res[1].converged:
        logger.warn("fitting mu (fermi level) brentq fails.")
    mu = res[0]
    return mu


def check_nelec(nelec, ncells=None, tol=1e-5):
    """
    Round off the nelec to its nearest integer.

    Args:
        nelec: number of electrons for the whole lattice.
        ncells: number of cells, if not None, will check nelec / ncells.

    Returns:
        nelec: rounded nelec, int.
        nelec_per_cell: number of elecrtons per cell, float.
    """
    from pyscf.lib import logger

    nelec_round = int(np.round(nelec))
    if abs(nelec - nelec_round) > tol:
        logger.warn(
            "HF: nelec is rounded to integer nelec = %d (original %.2f)",
            nelec_round,
            nelec,
        )
    nelec = nelec_round
    if ncells is None:
        nelec_per_cell = None
    else:
        nelec_per_cell = nelec / float(ncells)
        if abs(nelec_per_cell - np.round(nelec_per_cell)) > tol:
            logger.warn("HF: nelec per cell (%.5f) is not an integer.", nelec_per_cell)
        else:
            nelec_per_cell = int(np.round(nelec_per_cell))
    return nelec, nelec_per_cell


def assignocc(
    ew,
    nelec,
    beta,
    mu0=0.0,
    fix_mu=False,
    thr_deg=1e-6,
    Sz=None,
    fit_tol=1e-12,
    f_occ=fermi_smearing_occ,
    ncore=0,
    nvirt=0,
):
    """
    Assign the occupation number of a mean-field.
    nelec is per spin for RHF, total for UHF. 
    """

    try:
        from collections.abc import Iterable
    except ImportError:
        from collections import Iterable
    from pyscf.lib import logger

    ew = np.asarray(ew)
    if (Sz is None) and (not isinstance(nelec, Iterable)):
        if beta < np.inf:
            if ncore == 0 and nvirt == 0:
                ew_sorted = np.sort(ew, axis=None, kind="mergesort")
                if fix_mu:
                    mu = mu0
                else:
                    mu = find_mu(
                        nelec, ew_sorted, beta, mu0=mu0, tol=fit_tol, f_occ=f_occ
                    )
                ewocc = f_occ(mu, ew, beta)
                nerr = abs(np.sum(ewocc) - nelec)
            else:
                idx = np.argsort(ew, axis=None, kind="mergesort")
                ew_sorted = ew.ravel()[idx]
                idx_re = np.argsort(idx, kind="mergesort")
                idx = None
                if fix_mu:
                    mu = mu0
                else:
                    mu = find_mu(
                        nelec,
                        ew_sorted,
                        beta,
                        mu0=mu0,
                        tol=fit_tol,
                        f_occ=f_occ,
                        ncore=ncore,
                        nvirt=nvirt,
                    )
                ewocc = f_occ(mu, ew_sorted, beta, ncore=ncore, nvirt=nvirt)[idx_re]
                ewocc = ewocc.reshape(ew.shape)
                nerr = abs(np.sum(ewocc) - nelec)
        else:  # zero T
            ew_sorted = np.sort(ew, axis=None, kind="mergesort")
            nelec = check_nelec(nelec, None)[0]
            if (
                np.sum(ew < mu0 - thr_deg) <= nelec
                and np.sum(ew <= mu0 + thr_deg) >= nelec
            ):
                # we prefer not to change mu
                mu = mu0
            else:
                mu = 0.5 * (ew_sorted[nelec - 1] + ew_sorted[nelec])

            ewocc = 1.0 * (ew < mu - thr_deg)
            nremain_elec = nelec - np.sum(ewocc)
            if nremain_elec > 0:
                # fractional occupation
                remain_orb = np.logical_and(ew <= mu + thr_deg, ew >= mu - thr_deg)
                nremain_orb = np.sum(remain_orb)
                logger.warn(
                    "degenerate HOMO-LUMO, assign fractional occupation\n"
                    "%d electrons assigned to %d orbitals",
                    nremain_elec,
                    nremain_orb,
                )
                ewocc += (float(nremain_elec) / nremain_orb) * remain_orb
            nerr = 0.0
    else:  # allow specify Sz
        spin = ew.shape[0]
        assert spin == 2
        if not isinstance(nelec, Iterable):
            nelec = [(nelec + Sz) * 0.5, (nelec - Sz) * 0.5]
        if not isinstance(mu0, Iterable):
            mu0 = [mu0 for s in range(spin)]
        ewocc = np.empty_like(ew)
        mu = np.zeros((spin,))
        nerr = np.zeros((spin,))
        ewocc[0], mu[0], nerr[0] = assignocc(
            ew[0],
            nelec[0],
            beta,
            mu0[0],
            fix_mu=fix_mu,
            thr_deg=thr_deg,
            fit_tol=fit_tol,
            f_occ=f_occ,
            ncore=ncore,
            nvirt=nvirt,
        )
        ewocc[1], mu[1], nerr[1] = assignocc(
            ew[1],
            nelec[1],
            beta,
            mu0[1],
            fix_mu=fix_mu,
            thr_deg=thr_deg,
            fit_tol=fit_tol,
            f_occ=f_occ,
            ncore=ncore,
            nvirt=nvirt,
        )
    return ewocc, mu, nerr


def smearing_(mf, sigma=None, method="fermi", mu0=None, tol=1e-13, fit_spin=False):
    """
    Fermi-Dirac or Gaussian smearing.
    This version support Sz for UHF smearing.
    Args:
        mf: kmf object.
        sigma: smearing parameter, ~ 1/beta, unit in Hartree.
        method: fermi or gaussian
        mu0: initial mu
        tol: tolerance for fitting nelec
        fit_spin: if True, will fit each spin channel seprately.
    Returns:
        mf: modified mf object.
    """
    from pyscf.scf import uhf
    from pyscf.scf import ghf
    from pyscf.pbc.scf import khf
    from pyscf.lib import logger

    mf_class = mf.__class__
    is_uhf = isinstance(mf, uhf.UHF)
    is_ghf = isinstance(mf, ghf.GHF)
    is_rhf = (not is_uhf) and (not is_ghf)
    is_khf = isinstance(mf, khf.KSCF)
    if hasattr(mf, "cell"):
        Sz = mf.cell.spin
    else:
        Sz = mf.mol.spin
    tol = min(mf.conv_tol * 0.01, tol)

    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        """
        Label the occupancies for each orbital for sampled k-points.
        This is a k-point version of scf.hf.SCF.get_occ
        """
        if hasattr(mf, "kpts") and getattr(mf.kpts, "kpts_ibz", None) is not None:
            mo_energy_kpts = mf.kpts.transform_mo_energy(mo_energy_kpts)
        mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mo_occ_kpts

        if is_khf:
            nkpts = getattr(mf.kpts, "nkpts", None)
            if nkpts is None:
                nkpts = len(mf.kpts)
        else:
            nkpts = 1

        # find nelec_target
        from pyscf.pbc import gto as pbcgto
        if isinstance(mf.mol, pbcgto.Cell):
            nelectron = mf.mol.tot_electrons(nkpts)
        else:
            nelectron = mf.mol.tot_electrons()
        if is_uhf:
            if fit_spin:
                nelec_target = [(nelectron + Sz) * 0.5, (nelectron - Sz) * 0.5]
            else:
                nelec_target = nelectron
        elif is_ghf:
            nelec_target = nelectron
        else:
            nelec_target = nelectron * 0.5

        if mf.smearing_method.lower() == "fermi":  # Fermi-Dirac smearing
            f_occ = fermi_smearing_occ
        else:  # Gaussian smearing
            f_occ = gaussian_smearing_occ

        mo_energy = np.asarray(mo_energy_kpts)
        mo_occ, mf.mu, nerr = assignocc(
            mo_energy, nelec_target, 1.0 / mf.sigma, mf.mu, fit_tol=tol, f_occ=f_occ
        )
        mo_occ = mo_occ.reshape(mo_energy.shape)

        # See https://www.vasp.at/vasp-workshop/slides/k-points.pdf
        if mf.smearing_method.lower() == "fermi":
            f = mo_occ[(mo_occ > 0) & (mo_occ < 1)]
            mf.entropy = -(f * np.log(f) + (1 - f) * np.log(1 - f)).sum() / nkpts
        else:
            if is_uhf and fit_spin:
                mf.entropy = (
                    np.exp(-(((mo_energy[0] - mf.mu[0]) / mf.sigma) ** 2)).sum()
                    / (2 * np.sqrt(np.pi))
                    / nkpts
                ) + (
                    np.exp(-(((mo_energy[1] - mf.mu[1]) / mf.sigma) ** 2)).sum()
                    / (2 * np.sqrt(np.pi))
                    / nkpts
                )
            else:
                mf.entropy = (
                    np.exp(-(((mo_energy - mf.mu) / mf.sigma) ** 2)).sum()
                    / (2 * np.sqrt(np.pi))
                    / nkpts
                )
        if is_rhf:
            mo_occ *= 2
            mf.entropy *= 2

        nelec_now = mo_occ.sum()
        logger.debug(
            mf,
            "    Fermi level %s  Sum mo_occ_kpts = %s  should equal nelec = %s",
            mf.mu,
            nelec_now,
            nelectron,
        )
        if abs(nelec_now - nelectron) > 1e-8:
            logger.warn(
                "Occupancy (nelec_now %s) is not equal to cell.nelectron (%s).",
                nelec_now,
                nelectron,
            )
        logger.info(
            mf,
            "    sigma = %g  Optimized mu = %s  entropy = %.12g",
            mf.sigma,
            mf.mu,
            mf.entropy,
        )

        if hasattr(mf, "kpts") and getattr(mf.kpts, "kpts_ibz", None) is not None:
            if is_uhf:
                mo_occ = (
                    mf.kpts.check_mo_occ_symmetry(mo_occ[0]),
                    mf.kpts.check_mo_occ_symmetry(mo_occ[1]),
                )
            else:
                mo_occ = mf.kpts.check_mo_occ_symmetry(mo_occ)
        return mo_occ

    def get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
        if is_khf:
            grad_kpts = []
            for k, mo in enumerate(mo_coeff_kpts):
                f_mo = mo.T.conj() @ fock[k] @ mo
                nmo = f_mo.shape[0]
                grad_kpts.append(f_mo[np.tril_indices(nmo, -1)])
            return np.hstack(grad_kpts)
        else:
            f_mo = mo_coeff_kpts.T.conj() @ fock @ mo_coeff_kpts
            nmo = f_mo.shape[0]
            return f_mo[np.tril_indices(nmo, -1)]

    def get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None):
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mf_class.get_grad(mf, mo_coeff_kpts, mo_occ_kpts, fock)
        if fock is None:
            dm1 = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = mf.get_hcore() + mf.get_veff(mf.cell, dm1)
        if is_uhf:
            ga = get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return np.hstack((ga, gb))
        else:  # rhf and ghf
            return get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

    def energy_tot(dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        e_tot = mf.energy_elec(dm_kpts, h1e_kpts, vhf_kpts)[0] + mf.energy_nuc()
        if mf.sigma and mf.smearing_method and mf.entropy is not None:
            mf.e_free = e_tot - mf.sigma * mf.entropy
            mf.e_zero = e_tot - mf.sigma * mf.entropy * 0.5
            logger.info(
                mf,
                "    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g",
                e_tot,
                mf.e_free,
                mf.e_zero,
            )
        return e_tot

    mf.sigma = sigma
    mf.smearing_method = method
    mf.entropy = None
    mf.e_free = None
    mf.e_zero = None
    if mu0 is None:
        mf.mu = 0.0
    else:
        mf.mu = mu0
    mf._keys = mf._keys.union(
        ["sigma", "smearing_method", "entropy", "e_free", "e_zero", "mu"]
    )

    mf.get_occ = get_occ
    mf.energy_tot = energy_tot
    mf.get_grad = get_grad
    return mf
