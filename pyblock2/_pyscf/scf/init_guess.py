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
RDM1 transforms adapted from:
    https://github.com/gkclab/libdmet_preview/blob/main/libdmet/basis_transform/make_basis.py
    Author: Zhihao Cui <zhcui0408@gmail.com>
Mulliken population adapted from:
    https://github.com/gkclab/libdmet_preview/blob/main/libdmet/system/analyze.py
    Author: Zhihao Cui <zhcui0408@gmail.com>
"""

import numpy as np


def add_spin_dim(H, spin, non_spin_dim=3):
    H = np.asarray(H)
    if H.ndim == non_spin_dim:
        H = H[None]
    assert H.ndim == (non_spin_dim + 1)
    if H.shape[0] < spin:
        H = np.asarray((H[0],) * spin)
    return H


def transform_rdm1_to_ao(dm_mo_mo, c_ao_mo):
    dm_mo_mo = np.asarray(dm_mo_mo)
    c_ao_mo = np.asarray(c_ao_mo)

    nao = c_ao_mo.shape[-2]
    if c_ao_mo.ndim < dm_mo_mo.ndim:
        c_ao_mo = add_spin_dim(c_ao_mo, dm_mo_mo.shape[0], non_spin_dim=2)
    if c_ao_mo.ndim == 2:
        dm_ao_ao = c_ao_mo @ dm_mo_mo @ c_ao_mo.conj().T
    else:
        spin = c_ao_mo.shape[0]
        dm_mo_mo = add_spin_dim(dm_mo_mo, spin, non_spin_dim=2)
        assert dm_mo_mo.ndim == c_ao_mo.ndim
        dm_ao_ao = np.zeros((spin, nao, nao), dtype=c_ao_mo.dtype)
        for s in range(spin):
            dm_ao_ao[s] = c_ao_mo[s] @ dm_mo_mo[s] @ c_ao_mo[s].conj().T
    return dm_ao_ao


def transform_rdm1_to_mo(rdm1_ao_ao, c_ao_mo, s_ao_ao):
    rdm1_ao_ao = np.asarray(rdm1_ao_ao)
    c_ao_mo = np.asarray(c_ao_mo)
    nao = c_ao_mo.shape[-2]
    if c_ao_mo.ndim < rdm1_ao_ao.ndim:
        c_ao_mo = add_spin_dim(c_ao_mo, rdm1_ao_ao.shape[0], non_spin_dim=2)
    if c_ao_mo.ndim == 2:
        c_inv = c_ao_mo.conj().T.dot(s_ao_ao)
        rdm1_mo_mo = c_inv @ rdm1_ao_ao @ c_inv.conj().T
    else:
        spin = c_ao_mo.shape[0]
        rdm1_ao_ao = add_spin_dim(rdm1_ao_ao, spin, non_spin_dim=2)
        assert rdm1_ao_ao.ndim == c_ao_mo.ndim
        rdm1_mo_mo = np.zeros((spin, nao, nao), dtype=c_ao_mo.dtype)
        for s in range(spin):
            c_inv = c_ao_mo[s].conj().T.dot(s_ao_ao)
            rdm1_mo_mo[s] = c_inv @ rdm1_ao_ao[s] @ c_inv.conj().T
    return rdm1_mo_mo


def mulliken_pop_dmao(mol, dmao):
    """
    Mulliken population analysis, UHF case.
    Include local magnetic moment.
    """
    from pyscf import scf, lo
    from pyscf.lib import logger

    rmf = scf.RHF(mol)
    ld = lo.orth_ao(mol, "lowdin", pre_orth_ao="SCF")
    dm = transform_rdm1_to_mo(dmao, ld, rmf.get_ovlp())
    s = np.eye(dm.shape[-1])
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm * 0.5, dm * 0.5))
    pop_a = np.einsum("ij,ji->i", dm[0], s).real
    pop_b = np.einsum("ij,ji->i", dm[1], s).real

    logger.info(
        rmf, " ** Mulliken pop            alpha | beta      %12s **" % ("magnetism")
    )
    for i, s in enumerate(mol.ao_labels()):
        logger.info(
            rmf,
            "pop of  %-14s %10.5f | %-10.5f  %10.5f",
            s.strip(),
            pop_a[i],
            pop_b[i],
            pop_a[i] - pop_b[i],
        )
    logger.info(
        rmf,
        "In total               %10.5f | %-10.5f  %10.5f",
        sum(pop_a),
        sum(pop_b),
        sum(pop_a) - sum(pop_b),
    )

    logger.note(
        rmf,
        " ** Mulliken atomic charges    ( Nelec_alpha | Nelec_beta )"
        " %12s **" % ("magnetism"),
    )
    nelec_a = np.zeros(mol.natm)
    nelec_b = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        nelec_a[s[0]] += pop_a[i]
        nelec_b[s[0]] += pop_b[i]
    chg = mol.atom_charges() - (nelec_a + nelec_b)
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        logger.note(
            rmf,
            "charge of %4d%2s =   %10.5f  (  %10.5f   %10.5f )   %10.5f",
            ia,
            symb,
            chg[ia],
            nelec_a[ia],
            nelec_b[ia],
            nelec_a[ia] - nelec_b[ia],
        )
    return (pop_a, pop_b), chg


def get_metal_init_guess(
    mol, orb="3d", atom_idxs=[0], coupling="+", atomic_spin=None, nelec_corr=None
):
    from pyscf import scf, lo
    from pyscf.lib import logger

    rmf = scf.RHF(mol)
    dm0 = rmf.get_init_guess(key="atom")
    if dm0.ndim == 2:
        dm0 = np.array([dm0, dm0]) / 2.0
    nelec = np.einsum("sij,ji->s", dm0, rmf.get_ovlp())
    logger.info(scf.RHF(mol), "NELEC BEFORE = %.1f %.1f", nelec[0], nelec[1])

    idxs = []
    assert len(atom_idxs) == len(coupling)
    for ia in atom_idxs:
        orbs = orb if isinstance(orb, list) else [orb]
        for orx in orbs:
            idx = mol.search_ao_label("%d %s %s.*" % (ia, mol.atom_symbol(ia), orx))
            idxs.append(idx)

    ld = lo.orth_ao(mol, "lowdin", pre_orth_ao="SCF")
    dl0 = transform_rdm1_to_mo(dm0, ld, rmf.get_ovlp())

    if atomic_spin is None and coupling == "+" * len(atom_idxs):
        atomic_spin = mol.spin // len(atom_idxs)
    elif atomic_spin is None and coupling == ("+-" * (len(atom_idxs) // 2) + "+"):
        atomic_spin = mol.spin
    else:
        assert atomic_spin is not None

    dax, dbx = 0, 0
    if nelec_corr == "all":
        dl0[0] += (mol.nelectron / 2.0 - nelec[0]) / len(dl0[0])
        dl0[1] += (mol.nelectron / 2.0 - nelec[1]) / len(dl0[1])
    elif nelec_corr == "metal":
        dax = (mol.nelectron / 2.0 - nelec[0]) / len(atom_idxs)
        dbx = (mol.nelectron / 2.0 - nelec[1]) / len(atom_idxs)
    dspin = atomic_spin / 2.0

    for i in range(len(atom_idxs)):
        idx = idxs[i]
        if coupling[i] == "+":
            dl0[0][idx, idx] += (dax + dspin) / len(idx)
            dl0[1][idx, idx] += (dbx - dspin) / len(idx)
        else:
            dl0[0][idx, idx] += (dax - dspin) / len(idx)
            dl0[1][idx, idx] += (dbx + dspin) / len(idx)

    dm0 = transform_rdm1_to_ao(dl0, ld)
    nelec = np.einsum("sij,ji->s", dm0, rmf.get_ovlp())
    logger.info(scf.RHF(mol), "NELEC AFTER  = %.1f %.1f", nelec[0], nelec[1])
    return dm0
