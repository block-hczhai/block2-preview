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
Unrestricted Natural Orbital (UNO) for active space.

Partially adapted from
    https://github.com/pyscf/dmrgscf/blob/master/examples/32-dmrg_casscf_nevpt2_for_FeS.py
    Author: Zhendong Li <zhendongli2008@gmail.com>
            Qiming Sun <osirpt.sun@gmail.com>
"""

import numpy as np
import scipy.linalg


def sqrtm(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v * np.sqrt(e), v.T.conj())


def lowdin(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v / np.sqrt(e), v.T.conj())


def loc(mol, mocoeff, tol=1e-6, maxcycle=1000, iop=0):
    part = {}
    for iatom in range(mol.natm):
        part[iatom] = []
    ncgto = 0
    for binfo in mol._bas:
        atom_id = binfo[0]
        lang = binfo[1]
        ncntr = binfo[3]
        nbas = ncntr * (2 * lang + 1)
        part[atom_id] += range(ncgto, ncgto + nbas)
        ncgto += nbas
    partition = []
    for iatom in range(mol.natm):
        partition.append(part[iatom])
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    print()
    print("[pm_loc_kernel]")
    print(" mocoeff.shape=", mocoeff.shape)
    print(" tol=", tol)
    print(" maxcycle=", maxcycle)
    print(" partition=", len(partition), "\\n", partition)
    k = mocoeff.shape[0]
    n = mocoeff.shape[1]
    natom = len(partition)

    def genPaij(mol, mocoeff, ova, partition, iop):
        c = mocoeff.copy()
        # Mulliken matrix
        if iop == 0:
            cts = c.T.dot(ova)
            natom = len(partition)
            pija = np.zeros((natom, n, n))
            for iatom in range(natom):
                idx = partition[iatom]
                tmp = np.dot(cts[:, idx], c[idx, :])
                pija[iatom] = 0.5 * (tmp + tmp.T)
        # Lowdin
        elif iop == 1:
            s12 = sqrtm(ova)
            s12c = s12.T.dot(c)
            natom = len(partition)
            pija = np.zeros((natom, n, n))
            for iatom in range(natom):
                idx = partition[iatom]
                pija[iatom] = np.dot(s12c[idx, :].T, s12c[idx, :])
        # Boys
        elif iop == 2:
            rmat = mol.intor_symmetric("cint1e_r_sph", 3)
            pija = np.zeros((3, n, n))
            for icart in range(3):
                pija[icart] = c.T @ rmat[icart] @ c
        # P[i,j,a]
        pija = pija.transpose(1, 2, 0).copy()
        return pija

    u = np.identity(n)
    pija = genPaij(mol, mocoeff, ova, partition, iop)

    # Start
    def funval(pija):
        return np.einsum("iia,iia", pija, pija)

    fun = funval(pija)
    print(" initial funval = ", fun)
    for icycle in range(maxcycle):
        delta = 0.0
        # i>j
        ijdx = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                bij = abs(np.sum(pija[i, j] * (pija[i, i] - pija[j, j])))
                ijdx.append((i, j, bij))
        ijdx = sorted(ijdx, key=lambda x: x[2], reverse=True)
        for i, j, bij in ijdx:
            # determine angle
            vij = pija[i, i] - pija[j, j]
            aij = np.dot(pija[i, j], pija[i, j]) - 0.25 * np.dot(vij, vij)
            bij = np.dot(pija[i, j], vij)
            if abs(aij) < 1.0e-10 and abs(bij) < 1.0e-10:
                continue
            p1 = np.sqrt(aij ** 2 + bij ** 2)
            cos4a = -aij / p1
            sin4a = bij / p1
            cos2a = np.sqrt((1 + cos4a) * 0.5)
            sin2a = np.sqrt((1 - cos4a) * 0.5)
            cosa = np.sqrt((1 + cos2a) * 0.5)
            sina = np.sqrt((1 - cos2a) * 0.5)
            # Why? Because we require alpha in [0,pi/2]
            if sin4a < 0.0:
                cos2a = -cos2a
                sina, cosa = cosa, sina
            # stationary condition
            if abs(cosa - 1.0) < 1.0e-10:
                continue
            if abs(sina - 1.0) < 1.0e-10:
                continue
            # incremental value
            delta += p1 * (1 - cos4a)
            # Transformation
            # Urot
            ui = u[:, i] * cosa + u[:, j] * sina
            uj = -u[:, i] * sina + u[:, j] * cosa
            u[:, i] = ui.copy()
            u[:, j] = uj.copy()
            # Bra-transformation of Integrals
            tmp_ip = pija[i, :, :] * cosa + pija[j, :, :] * sina
            tmp_jp = -pija[i, :, :] * sina + pija[j, :, :] * cosa
            pija[i, :, :] = tmp_ip.copy()
            pija[j, :, :] = tmp_jp.copy()
            # Ket-transformation of Integrals
            tmp_ip = pija[:, i, :] * cosa + pija[:, j, :] * sina
            tmp_jp = -pija[:, i, :] * sina + pija[:, j, :] * cosa
            pija[:, i, :] = tmp_ip.copy()
            pija[:, j, :] = tmp_jp.copy()
        fun = fun + delta
        print("icycle=", icycle, "delta=", delta, "fun=", fun)
        if delta < tol:
            break

    # Check
    ierr = 0
    if delta < tol:
        print("CONG: PMloc converged!")
    else:
        ierr = 1
        print("WARNING: PMloc not converged")
    return ierr, u


def get_uno(mf, iprint=True):

    mol = mf.mol

    # 1. Read UHF-alpha/beta orbitals from chkfile

    ma, mb = mf.mo_coeff
    norb = ma.shape[1]
    nalpha = (mol.nelectron + mol.spin) // 2
    nbeta = (mol.nelectron - mol.spin) // 2
    if iprint:
        print(
            "Nalpha = %d, Nbeta %d, Sz = %d, Norb = %d"
            % (nalpha, nbeta, mol.spin, norb)
        )

    # 2. Sanity check, using orthogonality

    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    diff = ma.conj().T @ ova @ ma - np.identity(norb)
    assert np.linalg.norm(diff) < 1e-7
    diff = mb.conj().T @ ova @ mb - np.identity(norb)
    assert np.linalg.norm(diff) < 1e-7

    pta = np.dot(ma[:, :nalpha], ma[:, :nalpha].conj().T)
    ptb = np.dot(mb[:, :nbeta], mb[:, :nbeta].conj().T)
    pt = 0.5 * (pta + ptb)

    # Lowdin basis
    s12 = sqrtm(ova)
    s12inv = lowdin(ova)
    pt = s12 @ pt @ s12
    if iprint:
        print("Idemponency of DM: %s" % np.linalg.norm(pt.dot(pt) - pt))
    enorb = mf.mo_energy

    fa = ma @ np.diag(enorb[0]) @ ma.conj().T
    fb = mb @ np.diag(enorb[1]) @ mb.conj().T
    fav = (fa + fb) / 2

    # 'natural' occupations and orbitals
    eig, coeff = np.linalg.eigh(pt)
    eig = 2 * eig
    eig[abs(eig) < 1e-14] = 0.0

    # Rotate back to AO representation and check orthogonality
    coeff = np.dot(s12inv, coeff)
    diff = coeff.conj().T @ ova @ coeff - np.identity(norb)
    assert np.linalg.norm(diff) < 1e-7

    # 3. Search for active space

    # 3.1 Transform the entire MO space into core, active, and external space
    # based on natural occupancy

    fexpt = coeff.conj().T @ ova @ fav @ ova @ coeff
    enorb = np.diag(fexpt)
    index = np.argsort(-eig)
    enorb = enorb[index]
    nocc = eig[index]
    coeff = coeff[:, index]

    # Reordering and define active space according to thresh

    thresh = 0.05
    active = (thresh <= nocc) & (nocc <= 2 - thresh)
    act_idx = np.where(active)[0]
    if iprint:
        print("active orbital indices %s" % act_idx)
        print("Num active orbitals %d" % len(act_idx))
    c_orbs = coeff[:, : act_idx[0]]
    a_orbs = coeff[:, act_idx]
    v_orbs = coeff[:, act_idx[-1] + 1 :]
    norb = c_orbs.shape[0]
    nc = c_orbs.shape[1]
    na = a_orbs.shape[1]
    nv = v_orbs.shape[1]
    if iprint:
        print("core orbs:", c_orbs.shape)
        print("act  orbs:", a_orbs.shape)
        print("vir  orbs:", v_orbs.shape)
    assert nc + na + nv == norb

    # 3.2 Localizing core, active, external space separately, based on certain
    # local orbitals.

    c_orbs_oao = np.dot(s12, c_orbs)
    a_orbs_oao = np.dot(s12, a_orbs)
    v_orbs_oao = np.dot(s12, v_orbs)
    assert "Ortho-c_oao", (
        np.linalg.norm(np.dot(c_orbs_oao.conj().T, c_orbs_oao) - np.identity(nc)) < 1e-7
    )
    assert "Ortho-a_oao", (
        np.linalg.norm(np.dot(a_orbs_oao.conj().T, a_orbs_oao) - np.identity(na)) < 1e-7
    )
    assert "Ortho-v_oao", (
        np.linalg.norm(np.dot(v_orbs_oao.conj().T, v_orbs_oao) - np.identity(nv)) < 1e-7
    )

    def scdm(coeff, overlap, aux):
        no = coeff.shape[1]
        ova = coeff.conj().T @ overlap @ aux
        q, r, piv = scipy.linalg.qr(ova, pivoting=True)
        bc = ova[:, piv[:no]]
        ova = np.dot(bc.T, bc)
        s12inv = lowdin(ova)
        cnew = coeff @ bc @ s12inv
        return cnew

    clmo = c_orbs
    almo = a_orbs
    ierr, uc = loc(mol, clmo)
    ierr, ua = loc(mol, almo)
    clmo = clmo.dot(uc)
    almo = almo.dot(ua)

    # clmo = scdm(c_orbs, ova, s12inv)  # local "AOs" in core space
    # almo = scdm(a_orbs, ova, s12inv)  # local "AOs" in active space
    vlmo = scdm(v_orbs, ova, s12inv)  # local "AOs" in external space

    # 3.3 Sorting each space (core, active, external) based on "orbital energy" to
    # prevent high-lying orbitals standing in valence space.

    # Get <i|F|i>

    def psort(ova, fav, pt, s12, coeff):
        ptnew = 2.0 * (coeff.conj().T @ s12 @ pt @ s12 @ coeff)
        nocc = np.diag(ptnew)
        index = np.argsort(-nocc)
        ncoeff = coeff[:, index]
        nocc = nocc[index]
        enorb = np.diag(coeff.conj().T @ ova @ fav @ ova @ coeff)
        enorb = enorb[index]
        return ncoeff, nocc, enorb

    # E-SORT

    mo_c, n_c, e_c = psort(ova, fav, pt, s12, clmo)
    mo_o, n_o, e_o = psort(ova, fav, pt, s12, almo)
    mo_v, n_v, e_v = psort(ova, fav, pt, s12, vlmo)

    # coeff is the local molecular orbitals

    coeff = np.hstack((mo_c, mo_o, mo_v))
    mo_occ = np.hstack((n_c, n_o, n_v))

    # Test orthogonality for the localize MOs as before

    diff = coeff.conj().T @ ova @ coeff - np.identity(norb)
    assert np.linalg.norm(diff) < 1e-7

    # Population analysis to confirm that our LMO (coeff) make sense

    lcoeff = s12.dot(coeff)

    diff = lcoeff.conj().T @ lcoeff - np.identity(norb)
    assert np.linalg.norm(diff) < 1e-7

    if iprint:
        print("\\nLowdin population for LMOs:")

    labels = mol.ao_labels(None)
    texts = [None] * norb
    for iorb in range(norb):
        vec = lcoeff[:, iorb] ** 2
        ivs = np.argsort(vec)
        if iorb < nc:
            text = "[C %3d] occ = %8.5f" % (iorb, mo_occ[iorb])
            ftext = " fii = %10.3f" % e_c[iorb]
        elif iorb >= nc and iorb < nc + na:
            text = "[A %3d] occ = %8.5f" % (iorb, mo_occ[iorb])
            ftext = " fii = %10.3f" % e_o[iorb - nc]
        else:
            text = "[V %3d] occ = %8.5f" % (iorb, mo_occ[iorb])
            ftext = " fii = %10.3f" % e_v[iorb - nc - na]
        gtext = ""
        for iao in ivs[::-1][:3]:
            gtext += "(%3d-%2s-%7s = %5.3f) " % (
                labels[iao][0],
                labels[iao][1],
                labels[iao][2] + labels[iao][3],
                vec[iao],
            )
        if iprint:
            print(text + ftext + " " + gtext)
        texts[iorb] = text + "\\n" + gtext

    return coeff, mo_occ
