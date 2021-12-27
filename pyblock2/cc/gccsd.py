
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
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
CCSD in general orbitals with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

import itertools
import numpy as np

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.Inactive] = WickIndex.parse_set("pqrsijklmno")
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("pqrsabcdefg")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("v", 4)] = WickPermutation.four_anti()
    perm_map[("t", 2)] = WickPermutation.non_symmetric()
    perm_map[("t", 4)] = WickPermutation.four_anti()

    p = lambda x: WickExpr.parse(x, idx_map, perm_map)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    return p, pt

P, PT = init_parsers() # parsers
NR = lambda x: x.expand(-1, True).simplify() # normal order
FC = lambda x: x.expand(0).simplify() # fully contracted
Z = P("") # zero

def CommT(t, d): # commutator with t (at order d)
    return lambda h, i: (1.0 / i) * (h ^ t).expand((d - i) * 4).simplify()

def HBar(h, t, d): # exp(-t) h exp(t) (order d)
    return sum(itertools.accumulate([h, *range(1, d + 1)], CommT(t, d)), Z)

h1 = P("SUM <pq> h[pq] C[p] D[q]")
h2 = 0.25 * P("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
t1 = P("SUM <ai> t[ia] C[a] D[i]")
t2 = 0.25 * P("SUM <abij> t[ijab] C[a] C[b] D[j] D[i]")
ex1 = P("C[i] D[a]")
ex2 = P("C[i] C[j] D[b] D[a]")

h = NR(h1 + h2)
t = NR(t1 + t2)

en_eq = FC(HBar(h, t, 2))
t1_eq = FC(ex1 * HBar(h, t, 3))
t2_eq = FC(ex2 * HBar(h, t, 4))

# add diag fock term to lhs and rhs of the equation
t1_eq = t1_eq + P("h[ii]\n - h[aa]") * P("t[ia]")
t2_eq = t2_eq + P("h[ii]\n + h[jj]\n - h[aa]\n - h[bb]") * P("t[ijab]")

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Inactive: "I"}
    allowed_perms = {"IIII", "IIIE", "IIEE", "IEEI", "IEIE", "IEEE", "EEEE"}
    for term in eq.terms:
        for wt in term.tensors:
            if wt.name == "v":
                k = ''.join([imap[wi.types] for wi in wt.indices])
                if k not in allowed_perms:
                    found = False
                    for perm in wt.perms:
                        wtt = wt * perm
                        k = ''.join([imap[wi.types] for wi in wtt.indices])
                        if k in allowed_perms:
                            wt.indices = wtt.indices
                            if perm.negative:
                                term.factor = -term.factor
                            found = True
                            break
                    assert found

fix_eri_permutations(t1_eq)
fix_eri_permutations(t2_eq)

from pyscf.cc import gccsd

def wick_energy(cc, t1, t2, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    E = np.array(0.0)
    exec(en_eq.to_einsum(PT("E")), globals(), {
        "hIE": eris.fock[:nocc, nocc:],
        "vIIEE": np.array(eris.oovv),
        "tIE": t1,
        "tIIEE": t2,
        "E": E
    })
    return E

def wick_update_amps(cc, t1, t2, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    t1new = np.zeros_like(t1)
    t2new = np.zeros_like(t2)
    amps_eq = t1_eq.to_einsum(PT("t1new[ia]")) + t2_eq.to_einsum(PT("t2new[ijab]"))
    exec(amps_eq, globals(), {
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
        "t1new": t1new,
        "t2new": t2new
    })
    fii, faa = np.diag(eris.fock)[:nocc], np.diag(eris.fock)[nocc:]
    eia = fii[:, None] - faa[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1new /= eia
    t2new /= eijab
    return t1new, t2new

class WickGCCSD(gccsd.GCCSD):
    def __init__(self, mf, **kwargs):
        gccsd.GCCSD.__init__(self, mf, **kwargs)
    energy = wick_energy
    update_amps = wick_update_amps

GCCSD = WickGCCSD

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='cc-pvdz')
    mf = scf.GHF(mol).run(conv_tol=1E-14)
    ccsd = gccsd.GCCSD(mf).run()
    wccsd = WickGCCSD(mf).run()
