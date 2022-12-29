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
UHF/EOM-CCSD (EE/IP/EA) in spatial orbitals with equations derived on the fly.
"""

try:
    from .uccsd import h4, ex1a, ex1b, ex2aa, ex2ab, ex2ba, ex2bb, P, PT, NR, FC, fix_eri_permutations
except ImportError:
    from uccsd import h4, ex1a, ex1b, ex2aa, ex2ab, ex2ba, ex2bb, P, PT, NR, FC, fix_eri_permutations
import numpy as np
import functools

SP = lambda x: x.simplify()

exip1a = P("C[i]")
exip1b = P("C[I]")
exip2aa = P("C[i] C[j] D[b]")
exip2ab = P("C[i] C[J] D[B]")
exip2ba = P("C[I] C[j] D[b]")
exip2bb = P("C[I] C[J] D[B]")

exea1a = P("D[a]")
exea1b = P("D[A]")
exea2aa = P("C[j] D[b] D[a]")
exea2ab = P("C[J] D[B] D[a]")
exea2ba = P("C[j] D[b] D[A]")
exea2bb = P("C[J] D[B] D[A]")

ree1 = P("SUM <ai> ra[ia] C[a] D[i]\n + SUM <AI> rb[IA] C[A] D[I]")
ree2 = P(
    """
    0.25 SUM <aibj> raa[ijab] C[a] C[b] D[j] D[i]
    0.50 SUM <aiBJ> rab[iJaB] C[a] C[B] D[J] D[i]
    0.50 SUM <AIbj> rab[jIbA] C[A] C[b] D[j] D[I]
    0.25 SUM <AIBJ> rbb[IJAB] C[A] C[B] D[J] D[I]
"""
)

rip1 = P("SUM <i> ra[i] D[i]\n + SUM <I> rb[I] D[I]")
rip2 = P(
    """
    0.50 SUM <ibj> raaa[ijb] C[b] D[j] D[i]
    1.00 SUM <iBJ> rabb[iJB] C[B] D[J] D[i]
    1.00 SUM <Ibj> rbaa[Ijb] C[b] D[j] D[I]
    0.50 SUM <BIJ> rbbb[IJB] C[B] D[J] D[I]
"""
)

rea1 = P("SUM <a> ra[a] C[a]\n + SUM <A> rb[A] C[A]")
rea2 = P(
    """
    0.50 SUM <abj> raaa[jab] C[a] C[b] D[j]
    1.00 SUM <aBJ> rbab[JaB] C[a] C[B] D[J]
    1.00 SUM <Abj> raba[jAb] C[A] C[b] D[j]
    0.50 SUM <ABJ> rbbb[JAB] C[A] C[B] D[J]
"""
)

ree = NR(ree1 + ree2)
rip = NR(rip1 + rip2)
rea = NR(rea1 + rea2)

# eom-ee-ccsd
h4r = SP((h4 ^ ree).expand(4))
eomee_r1a_eq = FC(ex1a * h4r)
eomee_r1b_eq = FC(ex1b * h4r)
eomee_r2aa_eq = FC(ex2aa * h4r)
eomee_r2ab_eq = FC(ex2ab * h4r)
eomee_r2ba_eq = FC(ex2ba * h4r)
eomee_r2bb_eq = FC(ex2bb * h4r)
eomee_r1a_diag_eq = FC(ex1a * (h4 ^ ex1a.conjugate()))
eomee_r1b_diag_eq = FC(ex1b * (h4 ^ ex1b.conjugate()))
eomee_r2aa_diag_eq = FC(ex2aa * (h4 ^ ex2aa.conjugate()))
eomee_r2ab_diag_eq = FC(ex2ab * (h4 ^ ex2ab.conjugate()))
eomee_r2ba_diag_eq = FC(ex2ba * (h4 ^ ex2ba.conjugate()))
eomee_r2bb_diag_eq = FC(ex2bb * (h4 ^ ex2bb.conjugate()))

# eom-ip-ccsd
h4r = SP((h4 ^ rip).expand(4))
rh4 = SP((rip.conjugate() * h4).expand(4))
eomip_r1a_eq = FC(exip1a * h4r)
eomip_r1b_eq = FC(exip1b * h4r)
eomip_r2aa_eq = FC(exip2aa * h4r)
eomip_r2ab_eq = FC(exip2ab * h4r)
eomip_r2ba_eq = FC(exip2ba * h4r)
eomip_r2bb_eq = FC(exip2bb * h4r)
eomip_r1a_left_eq = FC(rh4 ^ exip1a.conjugate())
eomip_r1b_left_eq = FC(rh4 ^ exip1b.conjugate())
eomip_r2aa_left_eq = FC(rh4 ^ exip2aa.conjugate())
eomip_r2ab_left_eq = FC(rh4 ^ exip2ab.conjugate())
eomip_r2ba_left_eq = FC(rh4 ^ exip2ba.conjugate())
eomip_r2bb_left_eq = FC(rh4 ^ exip2bb.conjugate())
eomip_r1a_diag_eq = FC(exip1a * (h4 ^ exip1a.conjugate()))
eomip_r1b_diag_eq = FC(exip1b * (h4 ^ exip1b.conjugate()))
eomip_r2aa_diag_eq = FC(exip2aa * (h4 ^ exip2aa.conjugate()))
eomip_r2ab_diag_eq = FC(exip2ab * (h4 ^ exip2ab.conjugate()))
eomip_r2ba_diag_eq = FC(exip2ba * (h4 ^ exip2ba.conjugate()))
eomip_r2bb_diag_eq = FC(exip2bb * (h4 ^ exip2bb.conjugate()))

# eom-ea-ccsd
h4r = SP((h4 ^ rea).expand(4))
rh4 = SP((rea.conjugate() * h4).expand(4))
eomea_r1a_eq = FC(exea1a * h4r)
eomea_r1b_eq = FC(exea1b * h4r)
eomea_r2aa_eq = FC(exea2aa * h4r)
eomea_r2ab_eq = FC(exea2ab * h4r)
eomea_r2ba_eq = FC(exea2ba * h4r)
eomea_r2bb_eq = FC(exea2bb * h4r)
eomea_r1a_left_eq = FC(rh4 ^ exea1a.conjugate())
eomea_r1b_left_eq = FC(rh4 ^ exea1b.conjugate())
eomea_r2aa_left_eq = FC(rh4 ^ exea2aa.conjugate())
eomea_r2ab_left_eq = FC(rh4 ^ exea2ab.conjugate())
eomea_r2ba_left_eq = FC(rh4 ^ exea2ba.conjugate())
eomea_r2bb_left_eq = FC(rh4 ^ exea2bb.conjugate())
eomea_r1a_diag_eq = FC(exea1a * (h4 ^ exea1a.conjugate()))
eomea_r1b_diag_eq = FC(exea1b * (h4 ^ exea1b.conjugate()))
eomea_r2aa_diag_eq = FC(exea2aa * (h4 ^ exea2aa.conjugate()))
eomea_r2ab_diag_eq = FC(exea2ab * (h4 ^ exea2ab.conjugate()))
eomea_r2ba_diag_eq = FC(exea2ba * (h4 ^ exea2ba.conjugate()))
eomea_r2bb_diag_eq = FC(exea2bb * (h4 ^ exea2bb.conjugate()))

for eq in [
    eomee_r1a_eq,
    eomee_r1b_eq,
    eomee_r2aa_eq,
    eomee_r2ab_eq,
    eomee_r2ba_eq,
    eomee_r2bb_eq,
    eomee_r1a_diag_eq,
    eomee_r1b_diag_eq,
    eomee_r2aa_diag_eq,
    eomee_r2ab_diag_eq,
    eomee_r2ba_diag_eq,
    eomee_r2bb_diag_eq,
    eomip_r1a_eq,
    eomip_r1b_eq,
    eomip_r2aa_eq,
    eomip_r2ab_eq,
    eomip_r2ba_eq,
    eomip_r2bb_eq,
    eomip_r1a_left_eq,
    eomip_r1b_left_eq,
    eomip_r2aa_left_eq,
    eomip_r2ab_left_eq,
    eomip_r2ba_left_eq,
    eomip_r2bb_left_eq,
    eomip_r1a_diag_eq,
    eomip_r1b_diag_eq,
    eomip_r2aa_diag_eq,
    eomip_r2ab_diag_eq,
    eomip_r2ba_diag_eq,
    eomip_r2bb_diag_eq,
    eomea_r1a_eq,
    eomea_r1b_eq,
    eomea_r2aa_eq,
    eomea_r2ab_eq,
    eomea_r2ba_eq,
    eomea_r2bb_eq,
    eomea_r1a_left_eq,
    eomea_r1b_left_eq,
    eomea_r2aa_left_eq,
    eomea_r2ab_left_eq,
    eomea_r2ba_left_eq,
    eomea_r2bb_left_eq,
    eomea_r1a_diag_eq,
    eomea_r1b_diag_eq,
    eomea_r2aa_diag_eq,
    eomea_r2ab_diag_eq,
    eomea_r2ba_diag_eq,
    eomea_r2bb_diag_eq,
]:
    fix_eri_permutations(eq)

from pyscf.cc import eom_uccsd


def wick_make_imds(eom, eris=None):
    from pyscf import ao2mo

    if eris is None:
        eris = eom._cc.ao2mo(eom._cc.mo_coeff)
    t1a, t1b = eom._cc.t1
    t2aa, t2ab, t2bb = eom._cc.t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    eris_vvVV = np.zeros((nvira ** 2, nvirb ** 2), dtype=np.asarray(eris.vvVV).dtype)
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    eris_vvVV[
        (vtrila[0] * nvira + vtrila[1])[:, None], vtrilb[0] * nvirb + vtrilb[1]
    ] = np.asarray(eris.vvVV)
    eris_vvVV[
        (vtrila[1] * nvira + vtrila[0])[:, None], vtrilb[1] * nvirb + vtrilb[0]
    ] = np.asarray(eris.vvVV)
    eris_vvVV[
        (vtrila[0] * nvira + vtrila[1])[:, None], vtrilb[1] * nvirb + vtrilb[0]
    ] = np.asarray(eris.vvVV)
    eris_vvVV[
        (vtrila[1] * nvira + vtrila[0])[:, None], vtrilb[0] * nvirb + vtrilb[1]
    ] = np.asarray(eris.vvVV)
    eris_vvVV = eris_vvVV.reshape(nvira, nvira, nvirb, nvirb)
    imds = {
        "haie": eris.focka[:nocca, nocca:],
        "haei": eris.focka[nocca:, :nocca],
        "haee": eris.focka[nocca:, nocca:],
        "haii": eris.focka[:nocca, :nocca],
        "hbIE": eris.fockb[:noccb, noccb:],
        "hbEI": eris.fockb[noccb:, :noccb],
        "hbEE": eris.fockb[noccb:, noccb:],
        "hbII": eris.fockb[:noccb, :noccb],
        "vaaiiii": np.asarray(eris.oooo),
        "vaaieii": np.asarray(eris.ovoo),
        "vaaieie": np.asarray(eris.ovov),
        "vaaiiee": np.asarray(eris.oovv),
        "vaaieei": np.asarray(eris.ovvo),
        "vaaieee": eris.get_ovvv(slice(None)),
        "vaaeeee": ao2mo.restore(1, np.asarray(eris.vvvv), nvira),
        "vbbIIII": np.asarray(eris.OOOO),
        "vbbIEII": np.asarray(eris.OVOO),
        "vbbIEIE": np.asarray(eris.OVOV),
        "vbbIIEE": np.asarray(eris.OOVV),
        "vbbIEEI": np.asarray(eris.OVVO),
        "vbbIEEE": eris.get_OVVV(slice(None)),
        "vbbEEEE": ao2mo.restore(1, np.asarray(eris.VVVV), nvirb),
        "vabiiII": np.asarray(eris.ooOO),
        "vabieII": np.asarray(eris.ovOO),
        "vabieIE": np.asarray(eris.ovOV),
        "vabiiEE": np.asarray(eris.ooVV),
        "vabieEI": np.asarray(eris.ovVO),
        "vabieEE": eris.get_ovVV(slice(None)),
        "vabeeEE": eris_vvVV,
        "vbaIEii": np.asarray(eris.OVoo),
        "vbaIIee": np.asarray(eris.OOvv),
        "vbaIEei": np.asarray(eris.OVvo),
        "vbaIEee": eris.get_OVvv(slice(None)),
        "taie": t1a,
        "tbIE": t1b,
        "taaiiee": t2aa,
        "tabiIeE": t2ab,
        "tbbIIEE": t2bb,
        "deltaii": np.eye(nocca),
        "deltaII": np.eye(noccb),
        "deltaee": np.eye(nvira),
        "deltaEE": np.eye(nvirb),
        **{"ident%d" % d: np.ones((1,) * d) for d in [1, 2, 3]},
    }
    return imds


def wick_eomccsd_diag(eom, eq_type, imds=None):
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = eom._cc.t1, eom._cc.t2
    nocca, noccb, nvira, nvirb = t2[1].shape
    if eq_type == "ee":
        hr1a = np.zeros((nocca, nvira), dtype=t1[0].dtype)
        hr1b = np.zeros((noccb, nvirb), dtype=t1[1].dtype)
        hr2aa = np.zeros((nocca, nocca, nvira, nvira), dtype=t2[0].dtype)
        hr2ab = np.zeros((nocca, noccb, nvira, nvirb), dtype=t2[1].dtype)
        hr2bb = np.zeros((noccb, noccb, nvirb, nvirb), dtype=t2[2].dtype)
        eom_eq = eomee_r1a_diag_eq.to_einsum(PT("hra[ia]"))
        eom_eq += eomee_r1b_diag_eq.to_einsum(PT("hrb[IA]"))
        eom_eq += eomee_r2aa_diag_eq.to_einsum(PT("hraa[ijab]"))
        eom_eq += eomee_r2ab_diag_eq.to_einsum(PT("hrab[iJaB]"))
        eom_eq += eomee_r2bb_diag_eq.to_einsum(PT("hrbb[IJAB]"))
        hr_amps = {
            "hra": hr1a,
            "hrb": hr1b,
            "hraa": hr2aa,
            "hrab": hr2ab,
            "hrbb": hr2bb,
        }
        hr1 = hr1a, hr1b
        hr2 = hr2aa, hr2ab, hr2bb
    elif eq_type == "ip":
        hr1a = np.zeros((nocca, ), dtype=t1[0].dtype)
        hr1b = np.zeros((noccb, ), dtype=t1[1].dtype)
        hr2aaa = np.zeros((nocca, nocca, nvira), dtype=t2[0].dtype)
        hr2abb = np.zeros((nocca, noccb, nvirb), dtype=t2[1].dtype)
        hr2baa = np.zeros((noccb, nocca, nvira), dtype=t2[1].dtype)
        hr2bbb = np.zeros((noccb, noccb, nvirb), dtype=t2[2].dtype)
        eom_eq = eomip_r1a_diag_eq.to_einsum(PT("hra[i]"))
        eom_eq += eomip_r1b_diag_eq.to_einsum(PT("hrb[I]"))
        eom_eq += eomip_r2aa_diag_eq.to_einsum(PT("hraaa[ijb]"))
        eom_eq += eomip_r2ab_diag_eq.to_einsum(PT("hrabb[iJB]"))
        eom_eq += eomip_r2ba_diag_eq.to_einsum(PT("hrbaa[Ijb]"))
        eom_eq += eomip_r2bb_diag_eq.to_einsum(PT("hrbbb[IJB]"))
        hr_amps = {
            "hra": hr1a,
            "hrb": hr1b,
            "hraaa": hr2aaa,
            "hrabb": hr2abb,
            "hrbaa": hr2baa,
            "hrbbb": hr2bbb,
        }
        hr1 = hr1a, hr1b
        hr2 = hr2aaa, hr2baa, hr2abb, hr2bbb
    elif eq_type == "ea":
        hr1a = np.zeros((nvira, ), dtype=t1[0].dtype)
        hr1b = np.zeros((nvirb, ), dtype=t1[1].dtype)
        hr2aaa = np.zeros((nocca, nvira, nvira), dtype=t2[0].dtype)
        hr2aba = np.zeros((nocca, nvirb, nvira), dtype=t2[1].dtype)
        hr2bab = np.zeros((noccb, nvira, nvirb), dtype=t2[1].dtype)
        hr2bbb = np.zeros((noccb, nvirb, nvirb), dtype=t2[2].dtype)
        eom_eq = eomea_r1a_diag_eq.to_einsum(PT("hra[a]"))
        eom_eq += eomea_r1b_diag_eq.to_einsum(PT("hrb[A]"))
        eom_eq += eomea_r2aa_diag_eq.to_einsum(PT("hraaa[jab]"))
        eom_eq += eomea_r2ba_diag_eq.to_einsum(PT("hraba[jAb]"))
        eom_eq += eomea_r2ab_diag_eq.to_einsum(PT("hrbab[JaB]"))
        eom_eq += eomea_r2bb_diag_eq.to_einsum(PT("hrbbb[JAB]"))
        hr_amps = {
            "hra": hr1a,
            "hrb": hr1b,
            "hraaa": hr2aaa,
            "hraba": hr2aba,
            "hrbab": hr2bab,
            "hrbbb": hr2bbb,
        }
        hr1 = hr1a, hr1b
        hr2 = hr2aaa, hr2aba, hr2bab, hr2bbb
    exec(eom_eq, globals(), {**imds, **hr_amps})
    if eq_type == "ee":
        return eom.amplitudes_to_vector(hr1, hr2), None
    else:
        return eom.amplitudes_to_vector(hr1, hr2)


def wick_eomccsd_matvec(eom, eq_type, vector, imds=None, diag=None):
    if imds is None:
        imds = eom.make_imds()
    nocca, noccb, nvira, nvirb = eom._cc.t2[1].shape
    nmoa, nmob = nocca + nvira, noccb + nvirb
    r1, r2 = eom.vector_to_amplitudes(vector, (nmoa, nmob), (nocca, noccb))
    if eq_type == "ee":
        r1a, r1b = r1
        r2aa, r2ab, r2bb = r2
        hr1a = np.zeros_like(r1a)
        hr1b = np.zeros_like(r1b)
        hr2aa = np.zeros_like(r2aa)
        hr2ab = np.zeros_like(r2ab)
        hr2bb = np.zeros_like(r2bb)
        eom_eq = eomee_r1a_eq.to_einsum(PT("hra[ia]"))
        eom_eq += eomee_r1b_eq.to_einsum(PT("hrb[IA]"))
        eom_eq += eomee_r2aa_eq.to_einsum(PT("hraa[ijab]"))
        eom_eq += eomee_r2ab_eq.to_einsum(PT("hrab[iJaB]"))
        eom_eq += eomee_r2bb_eq.to_einsum(PT("hrbb[IJAB]"))
        r_amps = {
            "raie": r1a,
            "rbIE": r1b,
            "raaiiee": r2aa,
            "rabiIeE": r2ab,
            "rbbIIEE": r2bb,
        }
        hr_amps = {
            "hra": hr1a,
            "hrb": hr1b,
            "hraa": hr2aa,
            "hrab": hr2ab,
            "hrbb": hr2bb,
        }
        hr1 = hr1a, hr1b
        hr2 = hr2aa, hr2ab, hr2bb
    elif eq_type in ["ip", "lip"]:
        r1a, r1b = r1
        r2aaa, r2baa, r2abb, r2bbb = r2
        hr1a = np.zeros_like(r1a)
        hr1b = np.zeros_like(r1b)
        hr2aaa = np.zeros_like(r2aaa)
        hr2abb = np.zeros_like(r2abb)
        hr2baa = np.zeros_like(r2baa)
        hr2bbb = np.zeros_like(r2bbb)
        if eq_type == "ip":
            eom_eq = eomip_r1a_eq.to_einsum(PT("hra[i]"))
            eom_eq += eomip_r1b_eq.to_einsum(PT("hrb[I]"))
            eom_eq += eomip_r2aa_eq.to_einsum(PT("hraaa[ijb]"))
            eom_eq += eomip_r2ab_eq.to_einsum(PT("hrabb[iJB]"))
            eom_eq += eomip_r2ba_eq.to_einsum(PT("hrbaa[Ijb]"))
            eom_eq += eomip_r2bb_eq.to_einsum(PT("hrbbb[IJB]"))
        else:
            eom_eq = eomip_r1a_left_eq.to_einsum(PT("hra[i]"))
            eom_eq += eomip_r1b_left_eq.to_einsum(PT("hrb[I]"))
            eom_eq += eomip_r2aa_left_eq.to_einsum(PT("hraaa[ijb]"))
            eom_eq += eomip_r2ab_left_eq.to_einsum(PT("hrabb[iJB]"))
            eom_eq += eomip_r2ba_left_eq.to_einsum(PT("hrbaa[Ijb]"))
            eom_eq += eomip_r2bb_left_eq.to_einsum(PT("hrbbb[IJB]"))
        r_amps = {
            "rai": r1a,
            "rbI": r1b,
            "raaaiie": r2aaa,
            "rabbiIE": r2abb,
            "rbaaIie": r2baa,
            "rbbbIIE": r2bbb,
        }
        hr_amps = {
            "hra": hr1a,
            "hrb": hr1b,
            "hraaa": hr2aaa,
            "hrabb": hr2abb,
            "hrbaa": hr2baa,
            "hrbbb": hr2bbb,
        }
        hr1 = hr1a, hr1b
        hr2 = hr2aaa, hr2baa, hr2abb, hr2bbb
    elif eq_type in ["ea", "lea"]:
        r1a, r1b = r1
        r2aaa, r2aba, r2bab, r2bbb = r2
        hr1a = np.zeros_like(r1a)
        hr1b = np.zeros_like(r1b)
        hr2aaa = np.zeros_like(r2aaa)
        hr2aba = np.zeros_like(r2aba)
        hr2bab = np.zeros_like(r2bab)
        hr2bbb = np.zeros_like(r2bbb)
        if eq_type == "ea":
            eom_eq = eomea_r1a_eq.to_einsum(PT("hra[a]"))
            eom_eq += eomea_r1b_eq.to_einsum(PT("hrb[A]"))
            eom_eq += eomea_r2aa_eq.to_einsum(PT("hraaa[jab]"))
            eom_eq += eomea_r2ba_eq.to_einsum(PT("hraba[jAb]"))
            eom_eq += eomea_r2ab_eq.to_einsum(PT("hrbab[JaB]"))
            eom_eq += eomea_r2bb_eq.to_einsum(PT("hrbbb[JAB]"))
        else:
            eom_eq = eomea_r1a_left_eq.to_einsum(PT("hra[a]"))
            eom_eq += eomea_r1b_left_eq.to_einsum(PT("hrb[A]"))
            eom_eq += eomea_r2aa_left_eq.to_einsum(PT("hraaa[jab]"))
            eom_eq += eomea_r2ba_left_eq.to_einsum(PT("hraba[jAb]"))
            eom_eq += eomea_r2ab_left_eq.to_einsum(PT("hrbab[JaB]"))
            eom_eq += eomea_r2bb_left_eq.to_einsum(PT("hrbbb[JAB]"))
        r_amps = {
            "rae": r1a,
            "rbE": r1b,
            "raaaiee": r2aaa,
            "rabaiEe": r2aba,
            "rbabIeE": r2bab,
            "rbbbIEE": r2bbb,
        }
        hr_amps = {
            "hra": hr1a,
            "hrb": hr1b,
            "hraaa": hr2aaa,
            "hraba": hr2aba,
            "hrbab": hr2bab,
            "hrbbb": hr2bbb,
        }
        hr1 = hr1a, hr1b
        hr2 = hr2aaa, hr2aba, hr2bab, hr2bbb
    exec(eom_eq, globals(), {**imds, **r_amps, **hr_amps})
    return eom.amplitudes_to_vector(hr1, hr2)


class WickUEOMEESpinKeep(eom_uccsd.EOMEESpinKeep):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ee")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ee")
    make_imds = wick_make_imds


class WickUEOMIP(eom_uccsd.EOMIP):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ip")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lip")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ip")
    make_imds = wick_make_imds


class WickUEOMEA(eom_uccsd.EOMEA):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ea")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lea")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ea")
    make_imds = wick_make_imds

