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
UHF/CCSD lambda equation in spatial orbitals with equations derived on the fly.
"""

try:
    from .uccsd import h4, en_eq, ex1a, ex1b, ex2aa, ex2ab, ex2ba, ex2bb, P, PT, NR, FC, fix_eri_permutations
except ImportError:
    from uccsd import h4, en_eq, ex1a, ex1b, ex2aa, ex2ab, ex2ba, ex2bb, P, PT, NR, FC, fix_eri_permutations
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

l1a_eq = FC((I + l) * (h4 - en_eq) * ex1a.conjugate())
l1b_eq = FC((I + l) * (h4 - en_eq) * ex1b.conjugate())
l2aa_eq = FC((I + l) * (h4 - en_eq) * ex2aa.conjugate())
l2bb_eq = FC((I + l) * (h4 - en_eq) * ex2bb.conjugate())
l2ab_eq = FC((I + l) * (h4 - en_eq) * ex2ab.conjugate())
l2ba_eq = FC((I + l) * (h4 - en_eq) * ex2ba.conjugate())

# add diag fock term to lhs and rhs of the equation
l1a_eq = l1a_eq + P("ha[ii]\n - ha[aa]") * P("la[ia]")
l1b_eq = l1b_eq + P("hb[II]\n - hb[AA]") * P("lb[IA]")
l2aa_eq = l2aa_eq + P("ha[ii]\n + ha[jj]\n - ha[aa]\n - ha[bb]") * P("laa[ijab]")
l2bb_eq = l2bb_eq + P("hb[II]\n + hb[JJ]\n - hb[AA]\n - hb[BB]") * P("lbb[IJAB]")
l2ab_eq = l2ab_eq + P("ha[ii]\n + hb[JJ]\n - ha[aa]\n - hb[BB]") * P("lab[iJaB]")
l2ba_eq = l2ba_eq + P("hb[II]\n + ha[jj]\n - hb[AA]\n - ha[bb]") * P("lab[jIbA]")

for eq in [l1a_eq, l1b_eq, l2aa_eq, l2bb_eq, l2ab_eq, l2ba_eq]:
    fix_eri_permutations(eq)

from pyscf.cc import uccsd, ccsd_lambda
from pyscf.lib import logger
from pyscf import ao2mo

def wick_update_lambda(cc, t1, t2, l1, l2, eris, imds):
    assert isinstance(eris, uccsd._ChemistsERIs)
    assert cc.level_shift == 0
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, noccb, nvira, nvirb = t2ab.shape
    l1anew = np.zeros_like(l1a)
    l1bnew = np.zeros_like(l1b)
    l2aanew = np.zeros_like(l2aa)
    l2bbnew = np.zeros_like(l2bb)
    l2abnew = np.zeros_like(l2ab)
    lambda_eq = l1a_eq.to_einsum(PT("lanew[ia]")) + l1b_eq.to_einsum(PT("lbnew[IA]"))
    lambda_eq += l2aa_eq.to_einsum(PT("laanew[ijab]")) + l2bb_eq.to_einsum(PT("lbbnew[IJAB]"))
    lambda_eq += l2ab_eq.to_einsum(PT("labnew[iJaB]")) + l2ba_eq.to_einsum(PT("labnew[jIbA]"))
    eris_vvVV = np.zeros((nvira**2, nvirb**2), dtype=np.asarray(eris.vvVV).dtype)
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    eris_vvVV[(vtrila[0]*nvira+vtrila[1])[:, None], vtrilb[0]*nvirb+vtrilb[1]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[1]*nvira+vtrila[0])[:, None], vtrilb[1]*nvirb+vtrilb[0]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[0]*nvira+vtrila[1])[:, None], vtrilb[1]*nvirb+vtrilb[0]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[1]*nvira+vtrila[0])[:, None], vtrilb[0]*nvirb+vtrilb[1]] = np.asarray(eris.vvVV)
    eris_vvVV = eris_vvVV.reshape(nvira, nvira, nvirb, nvirb)
    exec(lambda_eq, globals(), {
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
        "laie": l1a,
        "lbIE": l1b,
        "laaiiee": l2aa,
        "labiIeE": l2ab,
        "lbbIIEE": l2bb,
        "lanew": l1anew,
        "lbnew": l1bnew,
        "laanew": l2aanew,
        "labnew": l2abnew,
        "lbbnew": l2bbnew
    })
    faii, faee = np.diag(eris.focka)[:nocca], np.diag(eris.focka)[nocca:]
    fbii, fbee = np.diag(eris.fockb)[:noccb], np.diag(eris.fockb)[noccb:]
    eaia = faii[:, None] - faee[None, :]
    ebia = fbii[:, None] - fbee[None, :]
    eaaijab = eaia[:, None, :, None] + eaia[None, :, None, :]
    ebbijab = ebia[:, None, :, None] + ebia[None, :, None, :]
    eabijab = eaia[:, None, :, None] + ebia[None, :, None, :]
    l1anew /= eaia
    l1bnew /= ebia
    l2aanew /= eaaijab
    l2bbnew /= ebbijab
    l2abnew /= eabijab + eabijab
    l1new = l1anew, l1bnew
    l2new = l2aanew, l2abnew, l2bbnew
    return l1new, l2new

def wick_kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, lambda *args, **kwargs: None, wick_update_lambda)
