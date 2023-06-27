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
Spin-free CCSD lambda equation with equations derived on the fly.
"""

try:
    from .rccsd import hbar, ex1, ex2, en_eq, P, PT, SP, FC, fix_eri_permutations, MapStrStr, WickGraph
except ImportError:
    from rccsd import hbar, ex1, ex2, en_eq, P, PT, SP, FC, fix_eri_permutations, MapStrStr, WickGraph
import numpy as np

l1 = P("SUM <ai> l[ia] E1[i,a]")
l2 = 0.5 * P("SUM <abij> l[ijab] E1[i,a] E1[j,b]")

l = SP(l1 + l2)
I = P("1")

l1_eq = FC((I + l) * (hbar - en_eq) * ex1.conjugate())
l2_eq = FC((I + l) * (hbar - en_eq) * ex2.conjugate())

# need some rearrangements
l1_eq = 0.5 * l1_eq
ijmap = MapStrStr({ 'i': 'j', 'j': 'i' })
l2_eq = SP((1.0 / 3.0) * (l2_eq + 0.5 * l2_eq.index_map(ijmap)))

# move diag fock to lhs
l1_eq = SP(l1_eq + P("f[ii]\n - f[aa]") * P("l[ia]"))
l2_eq = SP(l2_eq + P("f[ii]\n + f[jj]\n - f[aa]\n - f[bb]") * P("l[ijab]"))

gr_lambda_eq = WickGraph().add_term(PT("l1new[ia]"), l1_eq).add_term(PT("l2new[ijab]"), l2_eq).simplify()

fix_eri_permutations(gr_lambda_eq)

from pyscf.cc import rccsd, ccsd_lambda
from pyscf.lib import logger

def wick_update_lambda(cc, t1, t2, l1, l2, eris, imds):
    assert isinstance(eris, rccsd._ChemistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    l1new = np.zeros_like(l1)
    l2new = np.zeros_like(l2)
    exec(gr_lambda_eq.to_einsum(), globals(), {
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
        "lIE": l1,
        "lIIEE": l2,
        "l1new": l1new,
        "l2new": l2new
    })
    fii, faa = np.diag(eris.fock)[:nocc], np.diag(eris.fock)[nocc:]
    eia = fii[:, None] - faa[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    l1new /= eia
    l2new /= eijab
    return l1new, l2new

def wick_kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, lambda *args, **kwargs: None, wick_update_lambda)
