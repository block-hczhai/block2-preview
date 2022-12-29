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
CCSD lambda equation in general orbitals with equations derived on the fly.
"""

try:
    from .gccsd import hbar, ex1, ex2, P, PT, NR, FC, en_eq, fix_eri_permutations
except ImportError:
    from gccsd import hbar, ex1, ex2, P, PT, NR, FC, en_eq, fix_eri_permutations
import numpy as np

l1 = P("SUM <ai> l[ia] C[i] D[a]")
l2 = 0.25 * P("SUM <abij> l[ijab] C[i] C[j] D[b] D[a]")

l = NR(l1 + l2)
I = P("1")

l1_eq = FC((I + l) * (hbar - en_eq) * ex1.conjugate())
l2_eq = FC((I + l) * (hbar - en_eq) * ex2.conjugate())

# add diag fock term to lhs and rhs of the equation
l1_eq = l1_eq + P("h[ii]\n - h[aa]") * P("l[ia]")
l2_eq = l2_eq + P("h[ii]\n + h[jj]\n - h[aa]\n - h[bb]") * P("l[ijab]")

fix_eri_permutations(l1_eq)
fix_eri_permutations(l2_eq)

from pyscf.cc import gccsd, ccsd_lambda
from pyscf.lib import logger

def wick_update_lambda(cc, t1, t2, l1, l2, eris, imds):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    l1new = np.zeros_like(l1)
    l2new = np.zeros_like(l2)
    lambda_eq = l1_eq.to_einsum(PT("l1new[ia]")) + l2_eq.to_einsum(PT("l2new[ijab]"))
    exec(lambda_eq, globals(), {
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
