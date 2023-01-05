#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2023 Huanchen Zhai <hczhai@caltech.edu>
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
UHF/CCSD rdm in spatial orbitals with equations derived on the fly.
"""

try:
    from .uccsd import HBar, t, P, PT, NR, FC
except ImportError:
    from uccsd import HBar, t, P, PT, NR, FC
import numpy as np

l1 = P("SUM <ai> la[ia] C[i] D[a]\n + SUM <AI> lb[IA] C[I] D[A]")
l2 = P("""
    0.25 SUM <aibj> laa[ijab] C[i] C[j] D[b] D[a]
    0.50 SUM <aiBJ> lab[iJaB] C[i] C[J] D[B] D[a]
    0.50 SUM <AIbj> lab[jIbA] C[I] C[j] D[b] D[A]
    0.25 SUM <AIBJ> lbb[IJAB] C[I] C[J] D[B] D[A]
""")

l = NR(l1 + l2)
I = P("1")

# dm1[p,q] = <q^\dagger p>
doo, dOO = NR(P("C[i] D[j]")), NR(P("C[I] D[J]"))
dov, dOV = NR(P("C[a] D[j]")), NR(P("C[A] D[J]"))
dvo, dVO = NR(P("C[i] D[b]")), NR(P("C[I] D[B]"))
dvv, dVV = NR(P("C[a] D[b]")), NR(P("C[A] D[B]"))

# dm2[q,p,s,r] = <p^\dagger r^\dagger s q>
dovov = NR(0.50 * P("C[a] C[b] D[j] D[i] + C[i] C[j] D[b] D[a]"))
dovOV = NR(0.50 * P("C[a] C[B] D[J] D[i] + C[i] C[J] D[B] D[a]"))
dOVOV = NR(0.50 * P("C[A] C[B] D[J] D[I] + C[I] C[J] D[B] D[A]"))

dvvvv = NR(0.50 * P("C[b] C[d] D[c] D[a] + C[a] C[c] D[d] D[b]"))
dvvVV = NR(0.50 * P("C[b] C[D] D[C] D[a] + C[a] C[C] D[D] D[b]"))
dVVVV = NR(0.50 * P("C[B] C[D] D[C] D[A] + C[A] C[C] D[D] D[B]"))

doooo = NR(0.50 * P("C[j] C[l] D[k] D[i] + C[i] C[k] D[l] D[j]"))
dooOO = NR(0.50 * P("C[j] C[L] D[K] D[i] + C[i] C[K] D[L] D[j]"))
dOOOO = NR(0.50 * P("C[J] C[L] D[K] D[I] + C[I] C[K] D[L] D[J]"))

doovv = NR(0.50 * P("C[j] C[b] D[a] D[i] + C[i] C[a] D[b] D[j]"))
dooVV = NR(0.50 * P("C[j] C[B] D[A] D[i] + C[i] C[A] D[B] D[j]"))
dOOvv = NR(0.50 * P("C[J] C[b] D[a] D[I] + C[I] C[a] D[b] D[J]"))
dOOVV = NR(0.50 * P("C[J] C[B] D[A] D[I] + C[I] C[A] D[B] D[J]"))

dovvo = NR(0.50 * P("C[a] C[j] D[b] D[i] + C[i] C[b] D[j] D[a]"))
dovVO = NR(0.50 * P("C[a] C[J] D[B] D[i] + C[i] C[B] D[J] D[a]"))
dOVvo = NR(0.50 * P("C[A] C[j] D[b] D[I] + C[I] C[b] D[j] D[A]"))
dOVVO = NR(0.50 * P("C[A] C[J] D[B] D[I] + C[I] C[B] D[J] D[A]"))

dovvv = NR(0.25 * P("C[a] C[c] D[b] D[i] + C[i] C[b] D[c] D[a] + C[c] C[a] D[i] D[b] + C[b] C[i] D[a] D[c]"))
dovVV = NR(0.25 * P("C[a] C[C] D[B] D[i] + C[i] C[B] D[C] D[a] + C[C] C[a] D[i] D[B] + C[B] C[i] D[a] D[C]"))
dOVvv = NR(0.25 * P("C[A] C[c] D[b] D[I] + C[I] C[b] D[c] D[A] + C[c] C[A] D[I] D[b] + C[b] C[I] D[A] D[c]"))
dOVVV = NR(0.25 * P("C[A] C[C] D[B] D[I] + C[I] C[B] D[C] D[A] + C[C] C[A] D[I] D[B] + C[B] C[I] D[A] D[C]"))

dooov = NR(0.25 * P("C[j] C[a] D[k] D[i] + C[i] C[k] D[a] D[j] + C[a] C[j] D[i] D[k] + C[k] C[i] D[j] D[a]"))
dooOV = NR(0.25 * P("C[j] C[A] D[K] D[i] + C[i] C[K] D[A] D[j] + C[A] C[j] D[i] D[K] + C[K] C[i] D[j] D[A]"))
dOOov = NR(0.25 * P("C[J] C[a] D[k] D[I] + C[I] C[k] D[a] D[J] + C[a] C[J] D[I] D[k] + C[k] C[I] D[J] D[a]"))
dOOOV = NR(0.25 * P("C[J] C[A] D[K] D[I] + C[I] C[K] D[A] D[J] + C[A] C[J] D[I] D[K] + C[K] C[I] D[J] D[A]"))

doo_eq, dOO_eq = (FC((I + l) * HBar(d, t, 4)) for d in [doo, dOO])
dov_eq, dOV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dov, dOV])
dvo_eq, dVO_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dvo, dVO])
dvv_eq, dVV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dvv, dVV])

dovov_eq, dovOV_eq, dOVOV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dovov, dovOV, dOVOV])
dvvvv_eq, dvvVV_eq, dVVVV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dvvvv, dvvVV, dVVVV])
doooo_eq, dooOO_eq, dOOOO_eq = (FC((I + l) * HBar(d, t, 4)) for d in [doooo, dooOO, dOOOO])
doovv_eq, dooVV_eq, dOOvv_eq, dOOVV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [doovv, dooVV, dOOvv, dOOVV])
dovvo_eq, dovVO_eq, dOVvo_eq, dOVVO_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dovvo, dovVO, dOVvo, dOVVO])
dovvv_eq, dovVV_eq, dOVvv_eq, dOVVV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dovvv, dovVV, dOVvv, dOVVV])
dooov_eq, dooOV_eq, dOOov_eq, dOOOV_eq = (FC((I + l) * HBar(d, t, 4)) for d in [dooov, dooOV, dOOov, dOOOV])

from pyscf.cc import uccsd_rdm

def wick_gamma1_intermediates(mycc, t1, t2, l1, l2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    dooa = np.zeros((nocca, nocca), dtype=t1a.dtype)
    dova = np.zeros((nocca, nvira), dtype=t1a.dtype)
    dvoa = np.zeros((nvira, nocca), dtype=t1a.dtype)
    dvva = np.zeros((nvira, nvira), dtype=t1a.dtype)
    doob = np.zeros((noccb, noccb), dtype=t1b.dtype)
    dovb = np.zeros((noccb, nvirb), dtype=t1b.dtype)
    dvob = np.zeros((nvirb, noccb), dtype=t1b.dtype)
    dvvb = np.zeros((nvirb, nvirb), dtype=t1b.dtype)
    rdm_eq =  doo_eq.to_einsum(PT("dooa[ji]"))
    rdm_eq += dov_eq.to_einsum(PT("dova[ja]"))
    rdm_eq += dvo_eq.to_einsum(PT("dvoa[bi]"))
    rdm_eq += dvv_eq.to_einsum(PT("dvva[ba]"))
    rdm_eq += dOO_eq.to_einsum(PT("doob[JI]"))
    rdm_eq += dOV_eq.to_einsum(PT("dovb[JA]"))
    rdm_eq += dVO_eq.to_einsum(PT("dvob[BI]"))
    rdm_eq += dVV_eq.to_einsum(PT("dvvb[BA]"))
    exec(rdm_eq, globals(), {
        "taie": t1a,
        "tbIE": t1b,
        "taaiiee": t2aa,
        "tabiIeE": t2ab,
        "tbbIIEE": t2bb,
        "laie": l1a,
        "lbIE": l1b,
        "laaiiee": l2aa,
        "labiIeE": l2ab,
        "lbbIIEE": l2bb,
        "dooa": dooa,
        "dova": dova,
        "dvoa": dvoa,
        "dvva": dvva,
        "doob": doob,
        "dovb": dovb,
        "dvob": dvob,
        "dvvb": dvvb,
    })
    return ((dooa, doob), (dova, dovb), (dvoa, dvob), (dvva, dvvb))

def wick_gamma2_intermediates(mycc, t1, t2, l1, l2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    dovovaa = np.zeros((nocca, nvira, nocca, nvira), dtype=t1a.dtype)
    dvvvvaa = np.zeros((nvira, nvira, nvira, nvira), dtype=t1a.dtype)
    dooooaa = np.zeros((nocca, nocca, nocca, nocca), dtype=t1a.dtype)
    doovvaa = np.zeros((nocca, nocca, nvira, nvira), dtype=t1a.dtype)
    dovvoaa = np.zeros((nocca, nvira, nvira, nocca), dtype=t1a.dtype)
    dovvvaa = np.zeros((nocca, nvira, nvira, nvira), dtype=t1a.dtype)
    dooovaa = np.zeros((nocca, nocca, nocca, nvira), dtype=t1a.dtype)

    dovovab = np.zeros((nocca, nvira, noccb, nvirb), dtype=t1a.dtype)
    dvvvvab = np.zeros((nvira, nvira, nvirb, nvirb), dtype=t1a.dtype)
    dooooab = np.zeros((nocca, nocca, noccb, noccb), dtype=t1a.dtype)
    doovvab = np.zeros((nocca, nocca, nvirb, nvirb), dtype=t1a.dtype)
    dovvoab = np.zeros((nocca, nvira, nvirb, noccb), dtype=t1a.dtype)
    dovvvab = np.zeros((nocca, nvira, nvirb, nvirb), dtype=t1a.dtype)
    dooovab = np.zeros((nocca, nocca, noccb, nvirb), dtype=t1a.dtype)

    doovvba = np.zeros((noccb, noccb, nvira, nvira), dtype=t1b.dtype)
    dovvoba = np.zeros((noccb, nvirb, nvira, nocca), dtype=t1b.dtype)
    dovvvba = np.zeros((noccb, nvirb, nvira, nvira), dtype=t1b.dtype)
    dooovba = np.zeros((noccb, noccb, nocca, nvira), dtype=t1b.dtype)

    dovovbb = np.zeros((noccb, nvirb, noccb, nvirb), dtype=t1b.dtype)
    dvvvvbb = np.zeros((nvirb, nvirb, nvirb, nvirb), dtype=t1b.dtype)
    doooobb = np.zeros((noccb, noccb, noccb, noccb), dtype=t1b.dtype)
    doovvbb = np.zeros((noccb, noccb, nvirb, nvirb), dtype=t1a.dtype)
    dovvobb = np.zeros((noccb, nvirb, nvirb, noccb), dtype=t1b.dtype)
    dovvvbb = np.zeros((noccb, nvirb, nvirb, nvirb), dtype=t1b.dtype)
    dooovbb = np.zeros((noccb, noccb, noccb, nvirb), dtype=t1b.dtype)

    rdm_eq =  dovov_eq.to_einsum(PT("dovovaa[iajb]"))
    rdm_eq += dovOV_eq.to_einsum(PT("dovovab[iaJB]"))
    rdm_eq += dOVOV_eq.to_einsum(PT("dovovbb[IAJB]"))

    rdm_eq += dvvvv_eq.to_einsum(PT("dvvvvaa[abcd]"))
    rdm_eq += dvvVV_eq.to_einsum(PT("dvvvvab[abCD]"))
    rdm_eq += dVVVV_eq.to_einsum(PT("dvvvvbb[ABCD]"))

    rdm_eq += doooo_eq.to_einsum(PT("dooooaa[ijkl]"))
    rdm_eq += dooOO_eq.to_einsum(PT("dooooab[ijKL]"))
    rdm_eq += dOOOO_eq.to_einsum(PT("doooobb[IJKL]"))

    rdm_eq += doovv_eq.to_einsum(PT("doovvaa[ijab]"))
    rdm_eq += dooVV_eq.to_einsum(PT("doovvab[ijAB]"))
    rdm_eq += dOOvv_eq.to_einsum(PT("doovvba[IJab]"))
    rdm_eq += dOOVV_eq.to_einsum(PT("doovvbb[IJAB]"))

    rdm_eq += dovvo_eq.to_einsum(PT("dovvoaa[iabj]"))
    rdm_eq += dovVO_eq.to_einsum(PT("dovvoab[iaBJ]"))
    rdm_eq += dOVvo_eq.to_einsum(PT("dovvoba[IAbj]"))
    rdm_eq += dOVVO_eq.to_einsum(PT("dovvobb[IABJ]"))

    rdm_eq += dovvv_eq.to_einsum(PT("dovvvaa[iabc]"))
    rdm_eq += dovVV_eq.to_einsum(PT("dovvvab[iaBC]"))
    rdm_eq += dOVvv_eq.to_einsum(PT("dovvvba[IAbc]"))
    rdm_eq += dOVVV_eq.to_einsum(PT("dovvvbb[IABC]"))

    rdm_eq += dooov_eq.to_einsum(PT("dooovaa[ijka]"))
    rdm_eq += dooOV_eq.to_einsum(PT("dooovab[ijKA]"))
    rdm_eq += dOOov_eq.to_einsum(PT("dooovba[IJka]"))
    rdm_eq += dOOOV_eq.to_einsum(PT("dooovbb[IJKA]"))

    exec(rdm_eq, globals(), {
        "taie": t1a,
        "tbIE": t1b,
        "taaiiee": t2aa,
        "tabiIeE": t2ab,
        "tbbIIEE": t2bb,
        "laie": l1a,
        "lbIE": l1b,
        "laaiiee": l2aa,
        "labiIeE": l2ab,
        "lbbIIEE": l2bb,
        "dovovaa": dovovaa,
        "dovovab": dovovab,
        "dovovbb": dovovbb,
        "dvvvvaa": dvvvvaa,
        "dvvvvab": dvvvvab,
        "dvvvvbb": dvvvvbb,
        "dooooaa": dooooaa,
        "dooooab": dooooab,
        "doooobb": doooobb,
        "doovvaa": doovvaa,
        "doovvab": doovvab,
        "doovvba": doovvba,
        "doovvbb": doovvbb,
        "dovvoaa": dovvoaa,
        "dovvoab": dovvoab,
        "dovvoba": dovvoba,
        "dovvobb": dovvobb,
        "dovvvaa": dovvvaa,
        "dovvvab": dovvvab,
        "dovvvba": dovvvba,
        "dovvvbb": dovvvbb,
        "dooovaa": dooovaa,
        "dooovab": dooovab,
        "dooovba": dooovba,
        "dooovbb": dooovbb,
    })
    return (
        (dovovaa, dovovab, None, dovovbb), (dvvvvaa, dvvvvab, None, dvvvvbb),
        (dooooaa, dooooab, None, doooobb), (doovvaa, doovvab, doovvba, doovvbb),
        (dovvoaa, dovvoab, dovvoba, dovvobb), (None, None, None, None),
        (dovvvaa, dovvvab, dovvvba, dovvvbb), (dooovaa, dooovab, dooovba, dooovbb)
    )

def wick_make_rdm1(mycc, t1, t2, l1, l2, **kwargs):
    d1 = wick_gamma1_intermediates(mycc, t1, t2, l1, l2)
    return uccsd_rdm._make_rdm1(mycc, d1, **kwargs)

def wick_make_rdm2(mycc, t1, t2, l1, l2, **kwargs):
    d1 = wick_gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = wick_gamma2_intermediates(mycc, t1, t2, l1, l2)
    return uccsd_rdm._make_rdm2(mycc, d1, d2, **kwargs)
