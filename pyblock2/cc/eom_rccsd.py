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
Spin-free EOM-CCSD (EE/IP/EA) with equations derived on the fly.

Authors:  Huanchen Zhai
          Johannes Tölle    Dec 4-6, 2022
"""
try:
    from .rccsd import hbar, ex1, ex2, P, PT, SP, FC, fix_eri_permutations, MapStrStr, WickGraph
except ImportError:
    from rccsd import hbar, ex1, ex2, P, PT, SP, FC, fix_eri_permutations, MapStrStr, WickGraph
import numpy as np
import functools

exip1 = P("C1[i]")
exip2 = P("C1[i] E1[j,b]")
exea1 = P("D1[a]")
exea2 = P("E1[j,b] D1[a]")

exee2_d = P("E1[k,c] E1[l,d]")
exip2_d = P("C1[k] E1[l,d]")
exea2_d = P("E1[l,d] D1[c]")

ree1 = P("SUM <ai> r[ia] E1[a,i]")
ree2 = 0.5 * P("SUM <abij> r[ijab] E1[a,i] E1[b,j]")

rip1 = P("SUM <i> r[i] D1[i]")
rip2 = P("SUM <bij> r[ijb] E1[b,j] D1[i]")

rea1 = P("SUM <a> r[a] C1[a]")
rea2 = P("SUM <abj> r[jab] C1[a] E1[b,j]")

ree = SP(ree1 + ree2)
rip = SP(rip1 + rip2)
rea = SP(rea1 + rea2)

# eom-ee-ccsd
eomee_r1_eq = FC(ex1 * (hbar ^ ree))
eomee_r2_eq = FC(ex2 * (hbar ^ ree))
eomee_r1_left_eq = FC(ree.conjugate() * (hbar ^ ex1.conjugate()))
eomee_r2_left_eq = FC(ree.conjugate() * (hbar ^ ex2.conjugate()))
eomee_r1_diag_eq = FC(ex1 * (hbar ^ ex1.conjugate()))
eomee_r2_diag_eq = FC(ex2 * (hbar ^ exee2_d.conjugate()))

# eom-ip-ccsd
eomip_r1_eq = FC(exip1 * (hbar ^ rip))
eomip_r2_eq = FC(exip2 * (hbar ^ rip))
eomip_r1_left_eq = FC(rip.conjugate() * (hbar ^ exip1.conjugate()))
eomip_r2_left_eq = FC(rip.conjugate() * (hbar ^ exip2.conjugate()))
eomip_r1_diag_eq = FC(exip1 * (hbar ^ exip1.conjugate()))
eomip_r2_diag_eq = FC(exip2 * (hbar ^ exip2_d.conjugate()))

# eom-ea-ccsd
eomea_r1_eq = FC(exea1 * (hbar ^ rea))
eomea_r2_eq = FC(exea2 * (hbar ^ rea))
eomea_r1_left_eq = FC(rea.conjugate() * (hbar ^ exea1.conjugate()))
eomea_r2_left_eq = FC(rea.conjugate() * (hbar ^ exea2.conjugate()))
eomea_r1_diag_eq = FC(exea1 * (hbar ^ exea1.conjugate()))
eomea_r2_diag_eq = FC(exea2 * (hbar ^ exea2_d.conjugate()))

# need some rearrangements
ijmap = MapStrStr({"i": "j", "j": "i"})
abmap = MapStrStr({"a": "b", "b": "a"})
ijbmap = MapStrStr({"k": "i", "l": "j", "d": "b"})
jabmap = MapStrStr({"l": "j", "c": "a", "d": "b"})
ijabmap = MapStrStr({"k": "i", "l": "j", "c": "a", "d": "b"})

eomee_r1_eq = 0.5 * eomee_r1_eq
eomee_r2_eq = SP((1.0 / 3.0) * (eomee_r2_eq + 0.5 * eomee_r2_eq.index_map(ijmap)))
eomee_r1_left_eq = 0.5 * eomee_r1_left_eq
eomee_r2_left_eq = SP((1.0 / 3.0) * (eomee_r2_left_eq + 0.5 * eomee_r2_left_eq.index_map(ijmap)))
eomee_r1_diag_eq = 0.5 * eomee_r1_diag_eq
eomee_r2_diag_eq = SP(
    (1.0 / 3.0) * (eomee_r2_diag_eq + 0.5 * eomee_r2_diag_eq.index_map(ijmap))
)
eomee_r2_diag_eq = SP(
    P("1 - 0.5 delta[ij] delta[ab]") * eomee_r2_diag_eq.index_map(ijabmap)
)

eomip_r1_eq = 0.5 * eomip_r1_eq
eomip_r2_eq = SP((1.0 / 3.0) * (eomip_r2_eq + 0.5 * eomip_r2_eq.index_map(ijmap)))
eomip_r1_left_eq = 0.5 * eomip_r1_left_eq
eomip_r2_left_eq = SP(
    (1.0 / 3.0) * (eomip_r2_left_eq + 0.5 * eomip_r2_left_eq.index_map(ijmap))
)
eomip_r1_diag_eq = 0.5 * eomip_r1_diag_eq
eomip_r2_diag_eq = SP(
    (1.0 / 3.0) * (eomip_r2_diag_eq + 0.5 * eomip_r2_diag_eq.index_map(ijmap))
)
eomip_r2_diag_eq = SP(eomip_r2_diag_eq.index_map(ijbmap))

eomea_r1_eq = 0.5 * eomea_r1_eq
eomea_r2_eq = SP((1.0 / 3.0) * (eomea_r2_eq + 0.5 * eomea_r2_eq.index_map(abmap)))
eomea_r1_left_eq = 0.5 * eomea_r1_left_eq
eomea_r2_left_eq = SP(
    (1.0 / 3.0) * (eomea_r2_left_eq + 0.5 * eomea_r2_left_eq.index_map(abmap))
)
eomea_r1_diag_eq = 0.5 * eomea_r1_diag_eq
eomea_r2_diag_eq = SP(
    (1.0 / 3.0) * (eomea_r2_diag_eq + 0.5 * eomea_r2_diag_eq.index_map(abmap))
)
eomea_r2_diag_eq = SP(eomea_r2_diag_eq.index_map(jabmap))

gr_eomee_eq = WickGraph().add_term(PT("hr1[ia]"), eomee_r1_eq).add_term(PT("hr2[ijab]"), eomee_r2_eq).simplify()
gr_eomee_left_eq = WickGraph().add_term(PT("hr1[ia]"), eomee_r1_left_eq).add_term(PT("hr2[ijab]"), eomee_r2_left_eq).simplify()
gr_eomee_diag_eq = WickGraph().add_term(PT("hr1[ia]"), eomee_r1_diag_eq).add_term(PT("hr2[ijab]"), eomee_r2_diag_eq).simplify()

gr_eomip_eq = WickGraph().add_term(PT("hr1[i]"), eomip_r1_eq).add_term(PT("hr2[ijb]"), eomip_r2_eq).simplify()
gr_eomip_left_eq = WickGraph().add_term(PT("hr1[i]"), eomip_r1_left_eq).add_term(PT("hr2[ijb]"), eomip_r2_left_eq).simplify()
gr_eomip_diag_eq = WickGraph().add_term(PT("hr1[i]"), eomip_r1_diag_eq).add_term(PT("hr2[ijb]"), eomip_r2_diag_eq).simplify()

gr_eomea_eq = WickGraph().add_term(PT("hr1[a]"), eomea_r1_eq).add_term(PT("hr2[jab]"), eomea_r2_eq).simplify()
gr_eomea_left_eq = WickGraph().add_term(PT("hr1[a]"), eomea_r1_left_eq).add_term(PT("hr2[jab]"), eomea_r2_left_eq).simplify()
gr_eomea_diag_eq = WickGraph().add_term(PT("hr1[a]"), eomea_r1_diag_eq).add_term(PT("hr2[jab]"), eomea_r2_diag_eq).simplify()

for eq in [
    gr_eomee_eq,
    gr_eomee_left_eq,
    gr_eomee_diag_eq,
    gr_eomip_eq,
    gr_eomip_left_eq,
    gr_eomip_diag_eq,
    gr_eomea_eq,
    gr_eomea_left_eq,
    gr_eomea_diag_eq,
]:
    fix_eri_permutations(eq)

from pyscf.cc import eom_rccsd


def wick_eomccsd_diag(eom, eq_type, imds=None):
    if not hasattr(eom._cc, "eris"):
        eom._cc.eris = eom._cc.ao2mo(eom._cc.mo_coeff)
    t1, t2, eris = eom._cc.t1, eom._cc.t2, eom._cc.eris
    nocc, nvir = t1.shape
    if eq_type == "ee":
        hr1 = np.zeros((nocc, nvir), dtype=t1.dtype)
        hr2 = np.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        eom_eq = gr_eomee_diag_eq.to_einsum()
    elif eq_type == "ip":
        hr1 = np.zeros((nocc,), dtype=t1.dtype)
        hr2 = np.zeros((nocc, nocc, nvir), dtype=t2.dtype)
        eom_eq = gr_eomip_diag_eq.to_einsum()
    elif eq_type == "ea":
        hr1 = np.zeros((nvir,), dtype=t1.dtype)
        hr2 = np.zeros((nocc, nvir, nvir), dtype=t2.dtype)
        eom_eq = gr_eomea_diag_eq.to_einsum()
    exec(
        eom_eq,
        globals(),
        {
            "fIE": eris.fock[:nocc, nocc:],
            "fEI": eris.fock[nocc:, :nocc],
            "fEE": eris.fock[nocc:, nocc:],
            "fII": eris.fock[:nocc, :nocc],
            "vIIII": np.array(eris.oooo),
            "vIEII": np.array(eris.ovoo),
            "vIIEE": np.array(eris.oovv),
            "vIEEI": np.array(eris.ovvo),
            "vIEIE": np.array(eris.ovov),
            "vIEEE": np.array(eris.ovvv),
            "vEEEE": np.array(eris.vvvv),
            "tIE": t1,
            "tIIEE": t2,
            "hr1": hr1,
            "hr2": hr2,
            "deltaII": np.eye(nocc),
            "deltaEE": np.eye(nvir),
            **{"ident%d" % d: np.ones((1,) * d) for d in [1, 2, 3]},
        },
    )
    if eq_type == "ee":
        return eom.amplitudes_to_vector(hr1, hr2), None, None
    else:
        return eom.amplitudes_to_vector(hr1, hr2)


def wick_eomccsd_matvec(eom, eq_type, vector, imds=None, diag=None):
    if not hasattr(eom._cc, "eris"):
        eom._cc.eris = eom._cc.ao2mo(eom._cc.mo_coeff)
    t1, t2, eris = eom._cc.t1, eom._cc.t2, eom._cc.eris
    r1, r2 = eom.vector_to_amplitudes(vector, eom.nmo, eom.nocc)
    nocc, nvir = t1.shape
    hr1 = np.zeros_like(r1)
    hr2 = np.zeros_like(r2)
    if eq_type == "ee":
        eom_eq = gr_eomee_eq.to_einsum()
        r_amps = {"rIE": r1, "rIIEE": r2}
    elif eq_type == "lee":
        eom_eq = gr_eomee_left_eq.to_einsum()
        r_amps = {"rIE": r1, "rIIEE": r2}
    elif eq_type == "ip":
        eom_eq = gr_eomip_eq.to_einsum()
        r_amps = {"rI": r1, "rIIE": r2}
    elif eq_type == "lip":
        eom_eq = gr_eomip_left_eq.to_einsum()
        r_amps = {"rI": r1, "rIIE": r2}
    elif eq_type == "ea":
        eom_eq = gr_eomea_eq.to_einsum()
        r_amps = {"rE": r1, "rIEE": r2}
    elif eq_type == "lea":
        eom_eq = gr_eomea_left_eq.to_einsum()
        r_amps = {"rE": r1, "rIEE": r2}
    exec(
        eom_eq,
        globals(),
        {
            "fIE": eris.fock[:nocc, nocc:],
            "fEI": eris.fock[nocc:, :nocc],
            "fEE": eris.fock[nocc:, nocc:],
            "fII": eris.fock[:nocc, :nocc],
            "vIIII": np.array(eris.oooo),
            "vIEII": np.array(eris.ovoo),
            "vIIEE": np.array(eris.oovv),
            "vIEEI": np.array(eris.ovvo),
            "vIEIE": np.array(eris.ovov),
            "vIEEE": np.array(eris.ovvv),
            "vEEEE": np.array(eris.vvvv),
            "tIE": t1,
            "tIIEE": t2,
            "hr1": hr1,
            "hr2": hr2,
            "deltaII": np.eye(nocc),
            "deltaEE": np.eye(nvir),
            **r_amps,
        },
    )
    return eom.amplitudes_to_vector(hr1, hr2)


class WickREOMEESinglet(eom_rccsd.EOMEESinglet):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ee")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lee")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ee")
    def gen_matvec(self, imds=None, diag=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[0]
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
        return matvec, diag


class WickREOMIP(eom_rccsd.EOMIP):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ip")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lip")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ip")


class WickREOMEA(eom_rccsd.EOMEA):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ea")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lea")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ea")
