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
Spin-orbit coupling Hamiltonian integrals.

Partially adapted from:
    (1) https://github.com/xubwa/socutils/blob/master/somf.py
        Author: Xubo Wang
    (2) https://github.com/pyscf/properties/blob/master/pyscf/prop/zfs/uhf.py
        Author: Qiming Sun
"""

import numpy as np

from .integrals import get_rhf_integrals, get_uhf_integrals


def get_bp_hso2e(mol, dm0):
    hso2e = mol.intor("int2e_p1vxp1", 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum("yijkl,lk->yij", hso2e, dm0, optimize=True)
    vk = np.einsum("yijkl,jk->yil", hso2e, dm0, optimize=True)
    vk += np.einsum("yijkl,li->ykj", hso2e, dm0, optimize=True)
    return vj, vk


def get_bp_hso2e_amfi(mol, dm0):
    """atomic-mean-field approximation"""
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    import copy

    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_bp_hso2e(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def get_x2c_hso2e(mol, gamma_ll, gamma_ls, gamma_ss):
    hso2e = mol.intor("int2e_ip1ip2_sph", 9).reshape(
        3, 3, mol.nao, mol.nao, mol.nao, mol.nao
    )
    idx = np.arange(3)
    kint = (
        hso2e[np.roll(idx, 2), np.roll(idx, 1)]
        - hso2e[np.roll(idx, 1), np.roll(idx, 2)]
    )
    hso_ll = -2 * np.einsum("ipmqn,pq->imn", kint, gamma_ss, optimize=True)
    hso_ls = -1 * (
        np.einsum("impqr,pq->imr", kint, gamma_ls, optimize=True)
        + np.einsum("ipmqr,pq->imr", kint, gamma_ls, optimize=True)
    )
    hso_ss = -2 * (
        np.einsum("irspq,pq->irs", kint, gamma_ll, optimize=True)
        + np.einsum("irsqp,pq->irs", kint, gamma_ll, optimize=True)
        - np.einsum("irpsq,pq->irs", kint, gamma_ll, optimize=True)
    )
    return hso_ll, hso_ls, hso_ss


def get_x2c_hso2e_amfi(mol, gamma_ll, gamma_ls, gamma_ss):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    hso_ll = np.zeros((3, nao, nao))
    hso_ls = np.zeros((3, nao, nao))
    hso_ss = np.zeros((3, nao, nao))
    import copy

    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        hll, hls, hss = get_x2c_hso2e(
            atom, gamma_ll[p0:p1, p0:p1], gamma_ls[p0:p1, p0:p1], gamma_ss[p0:p1, p0:p1]
        )
        hso_ll[:, p0:p1, p0:p1] = hll
        hso_ls[:, p0:p1, p0:p1] = hls
        hso_ss[:, p0:p1, p0:p1] = hss
    return hso_ll, hso_ls, hso_ss


def get_somf_hsoao(mf, dmao=None, amfi=True, x2c1e=True, x2c2e=True, soecp=True):
    from pyscf.data import nist
    from pyscf import gto, lib

    mol = mf.mol

    if dmao is None:
        dmao = mf.make_rdm1()
        if dmao.ndim == 3:
            dmao = dmao[0] + dmao[1]
        elif dmao.ndim == 2 and dmao.shape[0] == mol.nao_nr() * 2:
            dmao = dmao[0::2, 0::2] + dmao[1::2, 1::2]

    if hasattr(mf, "with_x2c") and mf.with_x2c and (x2c1e or x2c2e):
        xmol, ctr = mf.with_x2c.get_xmol(mol)
        xdmao = ctr @ dmao @ ctr.conj().T
        x = mf.with_x2c.get_xmat()
        r = mf.with_x2c._get_rmat(x=x)
    if hasattr(mf, "with_x2c") and mf.with_x2c and x2c1e:
        hso1e = xmol.intor_asymmetric("int1e_pnucxp", 3)
        hso1e = np.einsum(
            "pq,qr,irs,sk,kl->ipl", r.conj().T, x.conj().T, hso1e, x, r, optimize=True
        )
    elif x2c1e is not None:
        hso1e = mol.intor_asymmetric("int1e_pnucxp", 3)
    else:
        hso1e = 0
    if hasattr(mf, "with_x2c") and mf.with_x2c and x2c2e == "x2camf":
        try:
            import x2camf
        except ImportError:
            raise RuntimeError(
                """x2camf library can be found in https://github.com/Warlocat/x2camf ;
                binaries can be found in https://github.com/hczhai/x2camf/releases/latest .
                """
            )
        hso2e = x2camf.amfi(
            mf.with_x2c, spin_free=False, two_c=False, with_gaunt=True, with_gauge=True
        )
        hso2e = hso2e / (nist.ALPHA ** 2 / 4)
        sphsp = xmol.sph2spinor_coeff()
        p_repr = np.einsum(
            "ixp,pq,jyq->ijxy", sphsp, hso2e, sphsp.conj(), optimize=True
        )
        hso2e = (
            p_repr[0, np.array([1, 1, 0])] * np.array([-1j, 1, -1j])[:, None, None]
        ).real
    elif hasattr(mf, "with_x2c") and mf.with_x2c and x2c2e:
        gamma_sa = xdmao * 0.5
        gamma_ll = r @ gamma_sa @ r.T.conj()
        gamma_ls = gamma_ll @ x.T.conj()
        gamma_ss = x @ gamma_ll @ x.T.conj()
        hso_ll, hso_ls, hso_ss = (
            get_x2c_hso2e_amfi(xmol, gamma_ll, gamma_ls, gamma_ss)
            if amfi
            else get_x2c_hso2e(xmol, gamma_ll, gamma_ls, gamma_ss)
        )
        hso_ls = np.einsum("imr,rn->imn", hso_ls, x, optimize=True)
        hso_sl = -hso_ls.transpose(0, 2, 1).conj()
        hso_ss = np.einsum("irs,rm,sn->imn", hso_ss, x.conj(), x, optimize=True)
        hso2e = np.einsum(
            "pr,irs,sq->ipq",
            r.conj().T,
            hso_ll + hso_ls + hso_sl + hso_ss,
            r,
            optimize=True,
        )
    elif x2c2e is not None:
        vj, vk = get_bp_hso2e_amfi(mol, dmao) if amfi else get_bp_hso2e(mol, dmao)
        hso2e = vj - vk * 1.5
    else:
        hso2e = 0.0
    if hasattr(mf, "with_x2c") and mf.with_x2c and x2c1e:
        if mf.with_x2c.basis is not None:
            s22 = xmol.intor_symmetric("int1e_ovlp")
            s21 = gto.intor_cross("int1e_ovlp", xmol, mol)
            c = lib.cho_solve(s22, s21)
            hso1e = c.conj().T @ hso1e @ c
        elif mf.with_x2c.xuncontract and ctr is not None:
            hso1e = ctr.conj().T @ hso1e @ ctr
    if hasattr(mf, "with_x2c") and mf.with_x2c and x2c2e:
        if mf.with_x2c.basis is not None:
            s22 = xmol.intor_symmetric("int1e_ovlp")
            s21 = gto.intor_cross("int1e_ovlp", xmol, mol)
            c = lib.cho_solve(s22, s21)
            hso2e = c.conj().T @ hso2e @ c
        elif mf.with_x2c.xuncontract and ctr is not None:
            hso2e = ctr.conj().T @ hso2e @ ctr
    hso = (1j * nist.ALPHA ** 2 / 2) * (hso1e + hso2e)
    if mol.has_ecp_soc() and soecp:
        hso -= mol.intor("ECPso") * 1j
    return hso


def get_rhf_somf_integrals(
    mf,
    ncore=0,
    ncas=None,
    pg_symm=True,
    dmao=None,
    amfi=True,
    x2c1e=True,
    x2c2e=True,
    soecp=True,
):

    from pyscf import scf

    assert isinstance(mf, scf.hf.RHF)
    n_elec, spin, ecore, hsf1e, gsf2e, orb_sym = get_rhf_integrals(
        mf, ncore, ncas, pg_symm, g2e_symm=1
    )[1:]

    mo = mf.mo_coeff

    if ncas is None:
        ncas = mo.shape[1] - ncore

    gh1e = np.zeros((ncas * 2, ncas * 2), dtype=complex)
    gg2e = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2), dtype=complex)

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            gh1e[i, j] = hsf1e[i // 2, j // 2]

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            for k in range(ncas * 2):
                for l in range(k % 2, ncas * 2, 2):
                    gg2e[i, j, k, l] = gsf2e[i // 2, j // 2, k // 2, l // 2]

    hsoao = get_somf_hsoao(mf, dmao, amfi, x2c1e, x2c2e, soecp)

    if isinstance(hsoao, np.ndarray):
        hso = np.einsum(
            "rij,ip,jq->rpq",
            hsoao,
            mo[:, ncore : ncore + ncas],
            mo[:, ncore : ncore + ncas],
        )

        for i in range(ncas * 2):
            for j in range(ncas * 2):
                if i % 2 == 0 and j % 2 == 0:  # aa
                    gh1e[i, j] += hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 1 and j % 2 == 1:  # bb
                    gh1e[i, j] -= hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 0 and j % 2 == 1:  # ab
                    gh1e[i, j] += (
                        hso[0, i // 2, j // 2] - hso[1, i // 2, j // 2] * 1j
                    ) * 0.5
                elif i % 2 == 1 and j % 2 == 0:  # ba
                    gh1e[i, j] += (
                        hso[0, i // 2, j // 2] + hso[1, i // 2, j // 2] * 1j
                    ) * 0.5

    assert np.linalg.norm(gh1e - gh1e.T.conj()) < 1e-10

    orb_sym = [orb_sym[x // 2] for x in range(ncas * 2)]

    return ncas * 2, n_elec, spin, ecore, gh1e, gg2e, orb_sym


def get_uhf_somf_integrals(
    mf,
    ncore=0,
    ncas=None,
    pg_symm=True,
    dmao=None,
    amfi=True,
    x2c1e=True,
    x2c2e=True,
    soecp=True,
):

    from pyscf import scf

    assert isinstance(mf, scf.uhf.UHF)
    n_elec, spin, ecore, hsf1e, gsf2e, orb_sym = get_uhf_integrals(
        mf, ncore, ncas, pg_symm, g2e_symm=1
    )[1:]

    mo = mf.mo_coeff

    if ncas is None:
        ncas = mo.shape[1] - ncore

    gh1e = np.zeros((ncas * 2, ncas * 2), dtype=complex)
    gg2e = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2), dtype=complex)

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            if i % 2 == 0:
                gh1e[i, j] = hsf1e[0][i // 2, j // 2]
            else:
                gh1e[i, j] = hsf1e[1][i // 2, j // 2]

    for i in range(ncas * 2):
        for j in range(i % 2, ncas * 2, 2):
            for k in range(ncas * 2):
                for l in range(k % 2, ncas * 2, 2):
                    if i % 2 == 0 and k % 2 == 0:
                        gg2e[i, j, k, l] = gsf2e[0][i // 2, j // 2, k // 2, l // 2]
                    elif i % 2 == 0 and k % 2 != 0:
                        gg2e[i, j, k, l] = gsf2e[1][i // 2, j // 2, k // 2, l // 2]
                    elif i % 2 != 0 and k % 2 == 0:
                        gg2e[i, j, k, l] = gsf2e[1][k // 2, l // 2, i // 2, j // 2]
                    else:
                        gg2e[i, j, k, l] = gsf2e[2][i // 2, j // 2, k // 2, l // 2]

    hsoao = get_somf_hsoao(mf, dmao, amfi, x2c1e, x2c2e, soecp)

    if isinstance(hsoao, np.ndarray):
        hso = np.einsum(
            "rij,xip,yjq->xyrpq",
            hsoao,
            np.array(mo)[:, :, ncore : ncore + ncas],
            np.array(mo)[:, :, ncore : ncore + ncas],
        )

        for i in range(ncas * 2):
            for j in range(ncas * 2):
                if i % 2 == 0 and j % 2 == 0:  # aa
                    gh1e[i, j] += hso[0, 0, 2, i // 2, j // 2] * 0.5
                elif i % 2 == 1 and j % 2 == 1:  # bb
                    gh1e[i, j] -= hso[1, 1, 2, i // 2, j // 2] * 0.5
                elif i % 2 == 0 and j % 2 == 1:  # ab
                    gh1e[i, j] += (
                        hso[0, 1, 0, i // 2, j // 2] - hso[0, 1, 1, i // 2, j // 2] * 1j
                    ) * 0.5
                elif i % 2 == 1 and j % 2 == 0:  # ba
                    gh1e[i, j] += (
                        hso[1, 0, 0, i // 2, j // 2] + hso[1, 0, 1, i // 2, j // 2] * 1j
                    ) * 0.5

    assert np.linalg.norm(gh1e - gh1e.T.conj()) < 1e-10

    orb_sym = [orb_sym[x // 2] for x in range(ncas * 2)]

    return ncas * 2, n_elec, spin, ecore, gh1e, gg2e, orb_sym

