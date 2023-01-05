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
Spin-free CCSD rdm with equations derived on the fly.
"""

try:
    from .rccsd import HBarTerms, Z, P, PT, SP, NR, FC
except ImportError:
    from rccsd import HBarTerms, Z, P, PT, SP, NR, FC
import numpy as np

l1 = P("SUM <ai> l[ia] E1[i,a]")
l2 = 0.5 * P("SUM <abij> l[ijab] E1[i,a] E1[j,b]")

l = SP(l1 + l2)
I = P("1")

# dm1[p,q] = <q^\dagger p>
doo = 0.5 * NR(P("C1[i] D1[j]"))
dov = 0.5 * NR(P("C1[a] D1[j]"))
dvo = 0.5 * NR(P("C1[i] D1[b]"))
dvv = 0.5 * NR(P("C1[a] D1[b]"))

# dm2[q,p,s,r] = <p^\dagger r^\dagger s q>
dovov = NR(0.250 * P("C1[a] C2[b] D2[j] D1[i] + C1[i] C2[j] D2[b] D1[a]"))
dvvvv = NR(0.125 * P("C1[b] C2[d] D2[c] D1[a] + C1[a] C2[c] D2[d] D1[b]"))
doooo = NR(0.125 * P("C1[j] C2[l] D2[k] D1[i] + C1[i] C2[k] D2[l] D1[j]"))
doovv = NR(0.250 * P("C1[j] C2[b] D2[a] D1[i] + C1[i] C2[a] D2[b] D1[j]"))
dovvo = NR(0.250 * P("C1[a] C2[j] D2[b] D1[i] + C1[i] C2[b] D2[j] D1[a]"))
dovvv = NR(0.250 * P("C1[a] C2[c] D2[b] D1[i] + C1[i] C2[b] D2[c] D1[a] + C1[c] C2[a] D2[i] D1[b] + C1[b] C2[i] D2[a] D1[c]"))
dooov = NR(0.250 * P("C1[j] C2[a] D2[k] D1[i] + C1[i] C2[k] D2[a] D1[j] + C1[a] C2[j] D2[i] D1[k] + C1[k] C2[i] D2[j] D1[a]"))

doo_eq = FC((I + l) * sum(HBarTerms(doo)[:5], Z))
dov_eq = FC((I + l) * sum(HBarTerms(dov)[:5], Z))
dvo_eq = FC((I + l) * sum(HBarTerms(dvo)[:5], Z))
dvv_eq = FC((I + l) * sum(HBarTerms(dvv)[:5], Z))

dovov_eq = FC((I + l) * sum(HBarTerms(dovov)[:5], Z))
dvvvv_eq = FC((I + l) * sum(HBarTerms(dvvvv)[:5], Z))
doooo_eq = FC((I + l) * sum(HBarTerms(doooo)[:5], Z))
doovv_eq = FC((I + l) * sum(HBarTerms(doovv)[:5], Z))
dovvo_eq = FC((I + l) * sum(HBarTerms(dovvo)[:5], Z))
dovvv_eq = FC((I + l) * sum(HBarTerms(dovvv)[:5], Z))
dooov_eq = FC((I + l) * sum(HBarTerms(dooov)[:5], Z))

from pyscf.cc import ccsd_rdm

def wick_gamma1_intermediates(mycc, t1, t2, l1, l2):
    nocc, nvir = t1.shape
    doo = np.zeros((nocc, nocc), dtype=t1.dtype)
    dov = np.zeros((nocc, nvir), dtype=t1.dtype)
    dvo = np.zeros((nvir, nocc), dtype=t1.dtype)
    dvv = np.zeros((nvir, nvir), dtype=t1.dtype)
    rdm_eq =  doo_eq.to_einsum(PT("doo[ji]"))
    rdm_eq += dov_eq.to_einsum(PT("dov[ja]"))
    rdm_eq += dvo_eq.to_einsum(PT("dvo[bi]"))
    rdm_eq += dvv_eq.to_einsum(PT("dvv[ba]"))
    exec(rdm_eq, globals(), {
        "tIE": t1,
        "tIIEE": t2,
        "lIE": l1,
        "lIIEE": l2,
        "doo": doo,
        "dov": dov,
        "dvo": dvo,
        "dvv": dvv,
    })
    return doo, dov, dvo, dvv

def wick_gamma2_intermediates(mycc, t1, t2, l1, l2):
    nocc, nvir = t1.shape
    dovov = np.zeros((nocc, nvir, nocc, nvir), dtype=t1.dtype)
    dvvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=t1.dtype)
    doooo = np.zeros((nocc, nocc, nocc, nocc), dtype=t1.dtype)
    doovv = np.zeros((nocc, nocc, nvir, nvir), dtype=t1.dtype)
    dovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=t1.dtype)
    dovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=t1.dtype)
    dooov = np.zeros((nocc, nocc, nocc, nvir), dtype=t1.dtype)
    rdm_eq =  dovov_eq.to_einsum(PT("dovov[iajb]"))
    rdm_eq += dvvvv_eq.to_einsum(PT("dvvvv[abcd]"))
    rdm_eq += doooo_eq.to_einsum(PT("doooo[ijkl]"))
    rdm_eq += doovv_eq.to_einsum(PT("doovv[ijab]"))
    rdm_eq += dovvo_eq.to_einsum(PT("dovvo[iabj]"))
    rdm_eq += dovvv_eq.to_einsum(PT("dovvv[iabc]"))
    rdm_eq += dooov_eq.to_einsum(PT("dooov[ijka]"))
    exec(rdm_eq, globals(), {
        "tIE": t1,
        "tIIEE": t2,
        "lIE": l1,
        "lIIEE": l2,
        "dovov": dovov,
        "dvvvv": dvvvv,
        "doooo": doooo,
        "doovv": doovv,
        "dovvo": dovvo,
        "dovvv": dovvv,
        "dooov": dooov,
    })
    return dovov, dvvvv, doooo, doovv, dovvo, None, dovvv, dooov

def wick_make_rdm1(mycc, t1, t2, l1, l2, **kwargs):
    d1 = wick_gamma1_intermediates(mycc, t1, t2, l1, l2)
    return ccsd_rdm._make_rdm1(mycc, d1, **kwargs)

def wick_make_rdm2(mycc, t1, t2, l1, l2, **kwargs):
    d1 = wick_gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = wick_gamma2_intermediates(mycc, t1, t2, l1, l2)
    return ccsd_rdm._make_rdm2(mycc, d1, d2, **kwargs)
