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
CCSD rdm in general orbitals with equations derived on the fly.
"""

try:
    from .gccsd import HBar, t, P, PT, NR, FC, WickGraph
except ImportError:
    from gccsd import HBar, t, P, PT, NR, FC, WickGraph
import numpy as np

l1 = P("SUM <ai> l[ia] C[i] D[a]")
l2 = 0.25 * P("SUM <abij> l[ijab] C[i] C[j] D[b] D[a]")

l = NR(l1 + l2)
I = P("1")

# dm1[p,q] = <q^\dagger p>
doo = NR(P("C[i] D[j]"))
dov = NR(P("C[a] D[j]"))
dvo = NR(P("C[i] D[b]"))
dvv = NR(P("C[a] D[b]"))

# dm2[q,p,s,r] = <p^\dagger r^\dagger s q>
dovov = NR(0.50 * P("C[a] C[b] D[j] D[i] + C[i] C[j] D[b] D[a]"))
dvvvv = NR(0.50 * P("C[b] C[d] D[c] D[a] + C[a] C[c] D[d] D[b]"))
doooo = NR(0.50 * P("C[j] C[l] D[k] D[i] + C[i] C[k] D[l] D[j]"))
dovvo = NR(0.50 * P("C[a] C[j] D[b] D[i] + C[i] C[b] D[j] D[a]"))
dovvv = NR(0.25 * P("C[a] C[c] D[b] D[i] + C[i] C[b] D[c] D[a] + C[c] C[a] D[i] D[b] + C[b] C[i] D[a] D[c]"))
dooov = NR(0.25 * P("C[j] C[a] D[k] D[i] + C[i] C[k] D[a] D[j] + C[a] C[j] D[i] D[k] + C[k] C[i] D[j] D[a]"))

doo_eq = FC((I + l) * HBar(doo, t, 4))
dov_eq = FC((I + l) * HBar(dov, t, 4))
dvo_eq = FC((I + l) * HBar(dvo, t, 4))
dvv_eq = FC((I + l) * HBar(dvv, t, 4))

dovov_eq = FC((I + l) * HBar(dovov, t, 4))
dvvvv_eq = FC((I + l) * HBar(dvvvv, t, 4))
doooo_eq = FC((I + l) * HBar(doooo, t, 4))
dovvo_eq = FC((I + l) * HBar(dovvo, t, 4))
dovvv_eq = FC((I + l) * HBar(dovvv, t, 4))
dooov_eq = FC((I + l) * HBar(dooov, t, 4))

gr_rdm1_eq = WickGraph()
for tn, eq in zip(["doo[ji]", "dov[ja]", "dvo[bi]", "dvv[ba]"], [doo_eq, dov_eq, dvo_eq, dvv_eq]):
    gr_rdm1_eq.add_term(PT(tn), eq)
gr_rdm1_eq = gr_rdm1_eq.simplify()

gr_rdm2_eq = WickGraph()
for tn, eq in zip(["dovov[iajb]", "dvvvv[abcd]", "doooo[ijkl]", "dovvo[iabj]", "dovvv[iabc]",
    "dooov[ijka]"], [dovov_eq, dvvvv_eq, doooo_eq, dovvo_eq, dovvv_eq, dooov_eq]):
    gr_rdm2_eq.add_term(PT(tn), eq)
gr_rdm2_eq = gr_rdm2_eq.simplify()

from pyscf.cc import gccsd_rdm

def wick_gamma1_intermediates(mycc, t1, t2, l1, l2):
    nocc, nvir = t1.shape
    doo = np.zeros((nocc, nocc), dtype=t1.dtype)
    dov = np.zeros((nocc, nvir), dtype=t1.dtype)
    dvo = np.zeros((nvir, nocc), dtype=t1.dtype)
    dvv = np.zeros((nvir, nvir), dtype=t1.dtype)
    exec(gr_rdm1_eq.to_einsum(), globals(), {
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
    dovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=t1.dtype)
    dovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=t1.dtype)
    dooov = np.zeros((nocc, nocc, nocc, nvir), dtype=t1.dtype)
    exec(gr_rdm2_eq.to_einsum(), globals(), {
        "tIE": t1,
        "tIIEE": t2,
        "lIE": l1,
        "lIIEE": l2,
        "dovov": dovov,
        "dvvvv": dvvvv,
        "doooo": doooo,
        "dovvo": dovvo,
        "dovvv": dovvv,
        "dooov": dooov,
    })
    return dovov, dvvvv, doooo, None, dovvo, None, dovvv, dooov

def wick_make_rdm1(mycc, t1, t2, l1, l2, **kwargs):
    d1 = wick_gamma1_intermediates(mycc, t1, t2, l1, l2)
    return gccsd_rdm._make_rdm1(mycc, d1, **kwargs)

def wick_make_rdm2(mycc, t1, t2, l1, l2, **kwargs):
    d1 = wick_gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = wick_gamma2_intermediates(mycc, t1, t2, l1, l2)
    return gccsd_rdm._make_rdm2(mycc, d1, d2, **kwargs)
