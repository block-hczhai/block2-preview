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

import numpy as np


def get_rhf_integrals(mf, ncore=0, ncas=None, pg_symm=True, g2e_symm=1):
    mol = mf.mol
    mo = mf.mo_coeff

    from pyscf import symm, ao2mo

    if ncas is None:
        ncas = mo.shape[1] - ncore

    if pg_symm and mol.symmetry:
        orb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo, tol=1e-2)
        orb_sym = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_sym]
        orb_sym = orb_sym[ncore : ncore + ncas]
    else:
        orb_sym = [0] * ncas

    ecore = mol.energy_nuc()
    mo_core = mo[:, :ncore]
    mo_cas = mo[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = 0

    if ncore != 0:
        core_dmao = 2 * mo_core @ mo_core.T
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj - 0.5 * vk
        ecore += np.einsum(
            "ij,ji->", core_dmao, hcore_ao + 0.5 * hveff_ao, optimize=True
        )

    h1e = mo_cas.T @ (hcore_ao + hveff_ao) @ mo_cas

    eri_ao = mol if mf._eri is None else mf._eri
    g2e = ao2mo.full(eri_ao, mo_cas)
    g2e = ao2mo.restore(g2e_symm, g2e, ncas)

    n_elec = mol.nelectron - ncore * 2
    spin = mol.spin

    return ncas, n_elec, spin, ecore, h1e, g2e, orb_sym


def get_uhf_integrals(mf, ncore=0, ncas=None, pg_symm=True, g2e_symm=1):
    mol = mf.mol
    mo_a, mo_b = mf.mo_coeff

    from pyscf import symm, ao2mo

    if ncas is None:
        ncas = mo_a.shape[1] - ncore

    if pg_symm and mol.symmetry:
        orb_syma = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_a, tol=1e-2
        )
        orb_syma = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_syma]
        orb_syma = orb_syma[ncore : ncore + ncas]
        orb_symb = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_b, tol=1e-2
        )
        orb_symb = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_symb]
        orb_symb = orb_symb[ncore : ncore + ncas]
        if orb_syma == orb_symb:
            orb_sym = orb_syma
        else:
            orb_sym = (orb_syma, orb_symb)
    else:
        orb_sym = [0] * ncas

    ecore = mol.energy_nuc()
    mo_core = mo_a[:, :ncore], mo_b[:, :ncore]
    mo_cas = mo_a[:, ncore : ncore + ncas], mo_b[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = (0, 0)

    if ncore != 0:
        core_dmao = mo_core[0] @ mo_core[0].T, mo_core[1] @ mo_core[1].T
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj[0] + vj[1] - vk
        ecore += np.einsum(
            "ij,ji->", core_dmao[0], hcore_ao + 0.5 * hveff_ao[0], optimize=True
        )
        ecore += np.einsum(
            "ij,ji->", core_dmao[1], hcore_ao + 0.5 * hveff_ao[1], optimize=True
        )

    h1e_a = mo_cas[0].T @ (hcore_ao + hveff_ao[0]) @ mo_cas[0]
    h1e_b = mo_cas[1].T @ (hcore_ao + hveff_ao[1]) @ mo_cas[1]

    eri_ao = mol if mf._eri is None else mf._eri
    mo_a, mo_b = mo_cas
    g2e_aa = ao2mo.restore(g2e_symm, ao2mo.full(eri_ao, mo_a), ncas)
    g2e_ab = ao2mo.restore(
        min(g2e_symm, 4), ao2mo.general(eri_ao, (mo_a, mo_a, mo_b, mo_b)), ncas
    )
    g2e_bb = ao2mo.restore(g2e_symm, ao2mo.full(eri_ao, mo_b), ncas)

    n_elec = mol.nelectron - ncore * 2
    spin = mol.spin

    return ncas, n_elec, spin, ecore, (h1e_a, h1e_b), (g2e_aa, g2e_ab, g2e_bb), orb_sym


# ncas counts number of spatial orbitals
def get_ghf_integrals(mf, ncore=0, ncas=None, pg_symm=True, g2e_symm=1):
    mol = mf.mol
    mo = mf.mo_coeff

    from pyscf import symm, ao2mo

    if ncas is None:
        ncas = mo.shape[1] // 2 - ncore

    if pg_symm and mol.symmetry:
        s = np.kron(np.eye(2, dtype=int), mol.intor_symmetric("int1e_ovlp"))
        symm_orb = [np.kron(np.eye(2, dtype=int), c) for c in mol.symm_orb]
        orb_sym = symm.label_orb_symm(mol, mol.irrep_name, symm_orb, mo, s=s, tol=1e-2)
        orb_sym = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_sym]
        orb_sym = orb_sym[ncore * 2 : ncore * 2 + ncas * 2]
    else:
        orb_sym = [0] * (ncas * 2)

    ecore = mol.energy_nuc()
    mo_core = mo[:, : ncore * 2]
    mo_cas = mo[:, ncore * 2 : ncore * 2 + ncas * 2]
    hcore_ao = mf.get_hcore()
    hveff_ao = 0

    if ncore != 0:
        core_dmao = mo_core @ mo_core.T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj - vk
        ecore += np.einsum(
            "ij,ji->", core_dmao, hcore_ao + 0.5 * hveff_ao, optimize=True
        )

    h1e = mo_cas.T.conj() @ (hcore_ao + hveff_ao) @ mo_cas

    eri_ao = mol if mf._eri is None else mf._eri
    mo_a, mo_b = mo_cas[: mol.nao], mo_cas[mol.nao :]
    g2e = ao2mo.restore(4, ao2mo.general(eri_ao, (mo_a, mo_a, mo_b, mo_b)), ncas * 2)
    g2e = g2e + g2e.transpose(1, 0)
    g2e += ao2mo.restore(4, ao2mo.full(eri_ao, mo_a), ncas * 2)
    g2e += ao2mo.restore(4, ao2mo.full(eri_ao, mo_b), ncas * 2)

    g2e = ao2mo.restore(g2e_symm, g2e, ncas * 2)

    n_elec = mol.nelectron - ncore * 2
    spin = mol.spin

    return ncas * 2, n_elec, spin, ecore, h1e, g2e, orb_sym


# ncas counts number of spatial orbitals
def get_dhf_integrals(mf, ncore=0, ncas=None, pg_symm=True):
    mol = mf.mol
    mo = mf.mo_coeff

    from pyscf import symm, ao2mo, lib

    if ncas is None:
        ncas = mo.shape[1] // 4 - ncore

    nneg = mo.shape[1] // 4
    ncore += nneg

    if pg_symm and mol.symmetry:
        s = np.kron(np.eye(4, dtype=int), mol.intor_symmetric("int1e_ovlp"))
        symm_orb = [np.kron(np.eye(4, dtype=int), c) for c in mol.symm_orb]
        orb_sym = symm.label_orb_symm(mol, mol.irrep_name, symm_orb, mo, s=s)
        orb_sym = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_sym]
        orb_sym = orb_sym[ncore * 2 : ncore * 2 + ncas * 2]
    else:
        orb_sym = [0] * (ncas * 2)

    ecore = mol.energy_nuc().real
    mo_core = mo[:, nneg * 2 : ncore * 2]
    mo_cas = mo[:, ncore * 2 : ncore * 2 + ncas * 2]
    hcore_ao = mf.get_hcore()
    hveff_ao = 0

    assert (
        not mf.with_ssss and mf._coulomb_level.upper() == "SSLL"
    ) or mf._coulomb_level.upper() == "SSSS"

    if ncore != 0:
        core_dmao = mo_core @ mo_core.T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj - vk
        ecore += np.einsum(
            "ij,ji->", core_dmao, hcore_ao + 0.5 * hveff_ao, optimize=True
        ).real

    c1 = 0.5 / lib.param.LIGHT_SPEED

    h1e = mo_cas.T.conj() @ (hcore_ao + hveff_ao) @ mo_cas

    mo_l, mo_s = mo_cas[: mol.nao_2c()], mo_cas[mol.nao_2c() :]
    g2e = ao2mo.general(
        mol, (mo_l, mo_l, mo_s, mo_s), intor="int2e_spsp2_spinor", aosym=4
    )
    g2e = (g2e + g2e.transpose(1, 0)) * c1 ** 2
    g2e += ao2mo.full(mol, mo_l, intor="int2e_spinor", aosym=4)
    g2e += ao2mo.full(mol, mo_s, intor="int2e_spsp1spsp2_spinor", aosym=4) * c1 ** 4

    if mf.with_gaunt:
        p = "int2e_breit_" if mf.with_breit else "int2e_"
        g2e_lsls = ao2mo.general(
            mol, (mo_l, mo_s, mo_l, mo_s), intor=p + "ssp1ssp2_spinor", aosym=1, comp=1
        )
        g2e_slsl = (
            g2e_lsls.reshape((ncas * 2,) * 4)
            .transpose(3, 2, 1, 0)
            .conj()
            .reshape((ncas * ncas * 4,) * 2)
        )
        g2e_lssl = ao2mo.general(
            mol, (mo_l, mo_s, mo_s, mo_l), intor=p + "ssp1sps2_spinor", aosym=1, comp=1
        )
        g2e_slls = g2e_lssl.transpose(1, 0)

        if mf.with_breit:
            g2e += (g2e_lsls + g2e_slsl + g2e_lssl + g2e_slls) * c1 ** 2
        else:
            g2e -= (g2e_lsls + g2e_slsl + g2e_lssl + g2e_slls) * c1 ** 2

    g2e = g2e.reshape((ncas * 2,) * 4)
    # assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1E-10
    # assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1E-10
    # assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1E-10

    n_elec = mol.nelectron - (ncore - nneg) * 2
    spin = mol.spin

    return ncas * 2, n_elec, spin, ecore, h1e, g2e, orb_sym
