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
EOM-CCSD (EE/IP/EA) in general orbitals with equations derived on the fly.
"""
try:
    from .gccsd import hbar, ex1, ex2, P, PT, NR, FC, fix_eri_permutations, WickGraph
except ImportError:
    from gccsd import hbar, ex1, ex2, P, PT, NR, FC, fix_eri_permutations, WickGraph
import numpy as np
import functools

exip1 = P("C[i]")
exip2 = P("C[i] C[j] D[b]")
exea1 = P("D[a]")
exea2 = P("C[j] D[b] D[a]")

ree1 = P("SUM <ai> r[ia] C[a] D[i]")
ree2 = 0.25 * P("SUM <abij> r[ijab] C[a] C[b] D[j] D[i]")

rip1 = P("SUM <i> r[i] D[i]")
rip2 = 0.5 * P("SUM <bij> r[ijb] C[b] D[j] D[i]")

rea1 = P("SUM <a> r[a] C[a]")
rea2 = 0.5 * P("SUM <abj> r[jab] C[a] C[b] D[j]")

ree = NR(ree1 + ree2)
rip = NR(rip1 + rip2)
rea = NR(rea1 + rea2)

# eom-ee-ccsd
eomee_r1_eq = FC(ex1 * (hbar ^ ree))
eomee_r2_eq = FC(ex2 * (hbar ^ ree))
eomee_r1_diag_eq = FC(ex1 * (hbar ^ ex1.conjugate()))
eomee_r2_diag_eq = FC(ex2 * (hbar ^ ex2.conjugate()))

gr_eomee_eq = WickGraph().add_term(PT("hr1[ia]"), eomee_r1_eq).add_term(PT("hr2[ijab]"), eomee_r2_eq).simplify()
gr_eomee_diag_eq = WickGraph().add_term(PT("hr1[ia]"), eomee_r1_diag_eq).add_term(PT("hr2[ijab]"), eomee_r2_diag_eq).simplify()

# eom-ip-ccsd
eomip_r1_eq = FC(exip1 * (hbar ^ rip))
eomip_r2_eq = FC(exip2 * (hbar ^ rip))
eomip_r1_left_eq = FC(rip.conjugate() * (hbar ^ exip1.conjugate()))
eomip_r2_left_eq = FC(rip.conjugate() * (hbar ^ exip2.conjugate()))
eomip_r1_diag_eq = FC(exip1 * (hbar ^ exip1.conjugate()))
eomip_r2_diag_eq = FC(exip2 * (hbar ^ exip2.conjugate()))

gr_eomip_eq = WickGraph().add_term(PT("hr1[i]"), eomip_r1_eq).add_term(PT("hr2[ijb]"), eomip_r2_eq).simplify()
gr_eomip_left_eq = WickGraph().add_term(PT("hr1[i]"), eomip_r1_left_eq).add_term(PT("hr2[ijb]"), eomip_r2_left_eq).simplify()
gr_eomip_diag_eq = WickGraph().add_term(PT("hr1[i]"), eomip_r1_diag_eq).add_term(PT("hr2[ijb]"), eomip_r2_diag_eq).simplify()

# eom-ea-ccsd
eomea_r1_eq = FC(exea1 * (hbar ^ rea))
eomea_r2_eq = FC(exea2 * (hbar ^ rea))
eomea_r1_left_eq = FC(rea.conjugate() * (hbar ^ exea1.conjugate()))
eomea_r2_left_eq = FC(rea.conjugate() * (hbar ^ exea2.conjugate()))
eomea_r1_diag_eq = FC(exea1 * (hbar ^ exea1.conjugate()))
eomea_r2_diag_eq = FC(exea2 * (hbar ^ exea2.conjugate()))

gr_eomea_eq = WickGraph().add_term(PT("hr1[a]"), eomea_r1_eq).add_term(PT("hr2[jab]"), eomea_r2_eq).simplify()
gr_eomea_left_eq = WickGraph().add_term(PT("hr1[a]"), eomea_r1_left_eq).add_term(PT("hr2[jab]"), eomea_r2_left_eq).simplify()
gr_eomea_diag_eq = WickGraph().add_term(PT("hr1[a]"), eomea_r1_diag_eq).add_term(PT("hr2[jab]"), eomea_r2_diag_eq).simplify()

for eq in [
    gr_eomee_eq,
    gr_eomee_diag_eq,
    gr_eomip_eq,
    gr_eomip_left_eq,
    gr_eomip_diag_eq,
    gr_eomea_eq,
    gr_eomea_left_eq,
    gr_eomea_diag_eq,
]:
    fix_eri_permutations(eq)

from pyscf.cc import eom_gccsd


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
            "hIE": eris.fock[:nocc, nocc:],
            "hEI": eris.fock[nocc:, :nocc],
            "hEE": eris.fock[nocc:, nocc:],
            "hII": eris.fock[:nocc, :nocc],
            "vIIII": np.array(eris.oooo),
            "vIIIE": np.array(eris.ooov),
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
    return eom.amplitudes_to_vector(hr1, hr2)


def wick_eomccsd_matvec(eom, eq_type, vector, imds=None, diag=None):
    if not hasattr(eom._cc, "eris"):
        eom._cc.eris = eom._cc.ao2mo(eom._cc.mo_coeff)
    t1, t2, eris = eom._cc.t1, eom._cc.t2, eom._cc.eris
    r1, r2 = eom.vector_to_amplitudes(vector, eom.nmo, eom.nocc)
    nocc = t1.shape[0]
    hr1 = np.zeros_like(r1)
    hr2 = np.zeros_like(r2)
    if eq_type == "ee":
        eom_eq = gr_eomee_eq.to_einsum()
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
            "hIE": eris.fock[:nocc, nocc:],
            "hEI": eris.fock[nocc:, :nocc],
            "hEE": eris.fock[nocc:, nocc:],
            "hII": eris.fock[:nocc, :nocc],
            "vIIII": np.array(eris.oooo),
            "vIIIE": np.array(eris.ooov),
            "vIIEE": np.array(eris.oovv),
            "vIEEI": np.array(eris.ovvo),
            "vIEIE": np.array(eris.ovov),
            "vIEEE": np.array(eris.ovvv),
            "vEEEE": np.array(eris.vvvv),
            "tIE": t1,
            "tIIEE": t2,
            "hr1": hr1,
            "hr2": hr2,
            **r_amps,
        },
    )
    return eom.amplitudes_to_vector(hr1, hr2)


class WickGEOMEE(eom_gccsd.EOMEE):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ee")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ee")


class WickGEOMIP(eom_gccsd.EOMIP):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ip")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lip")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ip")


class WickGEOMEA(eom_gccsd.EOMEA):
    matvec = functools.partialmethod(wick_eomccsd_matvec, "ea")
    l_matvec = functools.partialmethod(wick_eomccsd_matvec, "lea")
    get_diag = functools.partialmethod(wick_eomccsd_diag, "ea")

