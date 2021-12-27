
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
UGA-CCSD [J. Chem. Phys. 89, 7382 (1988)] with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")
import numpy as np

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.Inactive] = WickIndex.parse_set("pqrsijklmno")
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("pqrsabcdefg")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("v", 4)] = WickPermutation.qc_chem()
    perm_map[("t", 2)] = WickPermutation.non_symmetric()
    perm_map[("t", 4)] = WickPermutation.pair_symmetric(2, False)

    defs = MapStrPWickTensorExpr()
    p = lambda x: WickExpr.parse(x, idx_map, perm_map).substitute(defs)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    pd = lambda x: WickExpr.parse_def(x, idx_map, perm_map)

    return p, pt, pd, defs

P, PT, PD, DEF = init_parsers() # parsers
SP = lambda x: x.simplify() # just simplify
FC = lambda x: x.expand(0).simplify() # fully contracted
Z = P("") # zero

# definitions

DEF["h"] = PD("h[pq] = f[pq] \n - 2.0 SUM <j> v[pqjj] \n + SUM <j> v[pjjq]")
h1 = P("SUM <pq> h[pq] E1[p,q]")
h2 = P("0.5 SUM <pqrs> v[prqs] E2[pq,rs]")
t1 = P("SUM <ai> t[ia] E1[a,i]")
t2 = P("0.5 SUM <abij> t[ijab] E1[a,i] E1[b,j]")
ex1 = P("E1[i,a]")
ex2 = P("E1[i,a] E1[j,b]")
ehf = P("2 SUM <i> h[ii] \n + 2 SUM <ij> v[iijj] \n - SUM <ij> v[ijji]")

h = SP(h1 + h2 - ehf)
t = SP(t1 + t2)

HBarTerms = [
    h, h ^ t,
    0.5 * ((h ^ t1) ^ t1) + ((h ^ t2) ^ t1) + 0.5 * ((h ^ t2) ^ t2),
    (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1) + 0.5 * (((h ^ t2) ^ t1) ^ t1),
    (1 / 24.0) * ((((h ^ t1) ^ t1) ^ t1) ^ t1)
]

en_eq = FC(sum(HBarTerms[:3], Z))
t1_eq = FC(ex1 * sum(HBarTerms[:4], Z))
t2_eq = FC(ex2 * sum(HBarTerms[:5], Z))

# need some rearrangements
t1_eq = 0.5 * t1_eq
ijmap = MapStrStr({ 'i': 'j', 'j': 'i' })
t2_eq = SP((1.0 / 3.0) * (t2_eq + 0.5 * t2_eq.index_map(ijmap)))

# move diag fock to lhs
t1_eq = SP(t1_eq + P("f[ii]\n - f[aa]") * P("t[ia]"))
t2_eq = SP(t2_eq + P("f[ii]\n + f[jj]\n - f[aa]\n - f[bb]") * P("t[ijab]"))

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Inactive: "I"}
    allowed_perms = {"IIII", "IEII", "IIEE", "IEEI", "IEIE", "IEEE", "EEEE"}
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
                            found = True
                            break
                    assert found

fix_eri_permutations(t1_eq)
fix_eri_permutations(t2_eq)

from pyscf.cc import rccsd

def wick_energy(cc, t1, t2, eris):
    assert isinstance(eris, rccsd._ChemistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    E = np.array(0.0)
    exec(en_eq.to_einsum(PT("E")), globals(), {
        "fIE": eris.fock[:nocc, nocc:],
        "vIEIE": np.array(eris.ovov),
        "tIE": t1,
        "tIIEE": t2,
        "E": E
    })
    return E

def wick_update_amps(cc, t1, t2, eris):
    assert isinstance(eris, rccsd._ChemistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    t1new = np.zeros_like(t1)
    t2new = np.zeros_like(t2)
    amps_eq = t1_eq.to_einsum(PT("t1new[ia]")) + t2_eq.to_einsum(PT("t2new[ijab]"))
    exec(amps_eq, globals(), {
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
        "t1new": t1new,
        "t2new": t2new
    })
    fii, faa = np.diag(eris.fock)[:nocc], np.diag(eris.fock)[nocc:]
    eia = fii[:, None] - faa[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1new /= eia
    t2new /= eijab
    return t1new, t2new

class WickRCCSD(rccsd.RCCSD):
    def __init__(self, mf, **kwargs):
        rccsd.RCCSD.__init__(self, mf, **kwargs)
    energy = wick_energy
    update_amps = wick_update_amps

RCCSD = WickRCCSD

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='cc-pvdz')
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ccsd = rccsd.RCCSD(mf).run()
    wccsd = WickRCCSD(mf).run()
