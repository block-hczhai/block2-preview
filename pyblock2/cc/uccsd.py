
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2023 Huanchen Zhai <hczhai@caltech.edu>
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
UHF/CCSD and UHF/CCSD(T) in spatial orbitals with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation, VectorInt16
    from block2 import VectorWickPermutation, MapStrPWickTensorExpr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

import itertools
import numpy as np

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.InactiveAlpha] = WickIndex.parse_set("pqrsijklmno")
    idx_map[WickIndexTypes.InactiveBeta] = WickIndex.parse_set("PQRSIJKLMNO")
    idx_map[WickIndexTypes.ExternalAlpha] = WickIndex.parse_set("pqrsabcdefg")
    idx_map[WickIndexTypes.ExternalBeta] = WickIndex.parse_set("PQRSABCDEFG")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("vaa", 4)] = WickPermutation.qc_chem()
    perm_map[("vbb", 4)] = WickPermutation.qc_chem()
    perm_map[("vab", 4)] = WickPermutation.qc_chem()[1:]
    perm_map[("vba", 4)] = WickPermutation.qc_chem()[1:]
    perm_map[("ta", 2)] = WickPermutation.non_symmetric()
    perm_map[("tb", 2)] = WickPermutation.non_symmetric()
    perm_map[("taa", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("tbb", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("tab", 4)] = WickPermutation.non_symmetric()
    perm_01 = WickPermutation.pair_anti_symmetric(3)[0::2]
    perm_02 = WickPermutation.pair_anti_symmetric(3)[1::2]
    perm_12 = VectorWickPermutation([x * y * x for x, y in zip(perm_01, perm_02)])
    perm_map[("taaa", 6)] = WickPermutation.pair_anti_symmetric(3)
    perm_map[("taab", 6)] = perm_01
    perm_map[("tabb", 6)] = perm_12
    perm_map[("tbbb", 6)] = WickPermutation.pair_anti_symmetric(3)
    perm_map[("ra", 2)] = WickPermutation.non_symmetric()
    perm_map[("rb", 2)] = WickPermutation.non_symmetric()
    perm_map[("raa", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("rbb", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("rab", 4)] = WickPermutation.non_symmetric()
    perm_map[("la", 2)] = WickPermutation.non_symmetric()
    perm_map[("lb", 2)] = WickPermutation.non_symmetric()
    perm_map[("laa", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("lbb", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("lab", 4)] = WickPermutation.non_symmetric()

    p = lambda x: WickExpr.parse(x, idx_map, perm_map)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    def px(*xs):
        defs = MapStrPWickTensorExpr()
        for x in xs:
            name = x.split("=")[0].split("[")[0].strip()
            defs[name] = WickExpr.parse_def(x, idx_map, perm_map)
        return defs
    return p, pt, px

P, PT, PX = init_parsers() # parsers
NR = lambda x: x.expand(-1, True).simplify() # normal order
FC = lambda x: x.expand(0).simplify() # fully contracted
Z = P("") # zero

def CommT(t, d): # commutator with t (at order d)
    return lambda h, i: (1.0 / i) * (h ^ t).expand((d - i) * 4).simplify()

def HBar(h, t, d): # exp(-t) h exp(t) (order d)
    return sum(itertools.accumulate([h, *range(1, d + 1)], CommT(t, d)), Z)

h1 = P("SUM <pq> ha[pq] C[p] D[q]\n + SUM <PQ> hb[PQ] C[P] D[Q]")
h2 = P("""
    0.5 SUM <prqs> vaa[prqs] C[p] C[q] D[s] D[r]
    0.5 SUM <prQS> vab[prQS] C[p] C[Q] D[S] D[r]
    0.5 SUM <PRqs> vba[PRqs] C[P] C[q] D[s] D[R]
    0.5 SUM <PRQS> vbb[PRQS] C[P] C[Q] D[S] D[R]
""")
t1 = P("SUM <ai> ta[ia] C[a] D[i]\n + SUM <AI> tb[IA] C[A] D[I]")
# this def is consistent with pyscf init t amps
t2 = P("""
    0.25 SUM <aibj> taa[ijab] C[a] C[b] D[j] D[i]
    0.50 SUM <aiBJ> tab[iJaB] C[a] C[B] D[J] D[i]
    0.50 SUM <AIbj> tab[jIbA] C[A] C[b] D[j] D[I]
    0.25 SUM <AIBJ> tbb[IJAB] C[A] C[B] D[J] D[I]
""")
t3 = (1.0 / 36.0) * P("""
    SUM <abcijk> taaa[ijkabc] C[a] C[b] C[c] D[k] D[j] D[i]
    SUM <abCijK> taab[ijKabC] C[a] C[b] C[C] D[K] D[j] D[i]
    SUM <aBciJk> taab[ikJacB] C[a] C[B] C[c] D[k] D[J] D[i]
    SUM <aBCiJK> tabb[iJKaBC] C[a] C[B] C[C] D[K] D[J] D[i]
    SUM <AbcIjk> taab[jkIbcA] C[A] C[b] C[c] D[k] D[j] D[I]
    SUM <AbCIjK> tabb[jIKbAC] C[A] C[b] C[C] D[K] D[j] D[I]
    SUM <ABcIJk> tabb[kIJcAB] C[A] C[B] C[c] D[k] D[J] D[I]
    SUM <ABCIJK> tbbb[IJKABC] C[A] C[B] C[C] D[K] D[J] D[I]
""")
ex1a = P("C[i] D[a]")
ex1b = P("C[I] D[A]")
ex2aa = P("C[i] C[j] D[b] D[a]")
ex2bb = P("C[I] C[J] D[B] D[A]")
ex2ab = P("C[i] C[J] D[B] D[a]")
ex2ba = P("C[I] C[j] D[b] D[A]")
ex3aaa = 1.0 * P("C[i] C[j] C[k] D[c] D[b] D[a]")
ex3aab = 3.0 * P("C[i] C[j] C[K] D[C] D[b] D[a]")
ex3aba = 3.0 * P("C[i] C[J] C[k] D[c] D[B] D[a]")
ex3abb = 3.0 * P("C[i] C[J] C[K] D[C] D[B] D[a]")
ex3baa = 3.0 * P("C[I] C[j] C[k] D[c] D[b] D[A]")
ex3bab = 3.0 * P("C[I] C[j] C[K] D[C] D[b] D[A]")
ex3bba = 3.0 * P("C[I] C[J] C[k] D[c] D[B] D[A]")
ex3bbb = 1.0 * P("C[I] C[J] C[K] D[C] D[B] D[A]")

h = NR(h1 + h2)
t = NR(t1 + t2)

h2 = HBar(h, t, 2)
h3 = HBar(h, t, 3)
h4 = HBar(h, t, 4)

en_eq = FC(h2)
t1a_eq = FC(ex1a * h3)
t1b_eq = FC(ex1b * h3)
t2aa_eq = FC(ex2aa * h4)
t2bb_eq = FC(ex2bb * h4)
t2ab_eq = FC(ex2ab * h4)
t2ba_eq = FC(ex2ba * h4)

# move diag fock to lhs
t1a_eq = t1a_eq + P("ha[ii]\n - ha[aa]") * P("ta[ia]")
t1b_eq = t1b_eq + P("hb[II]\n - hb[AA]") * P("tb[IA]")
t2aa_eq = t2aa_eq + P("ha[ii]\n + ha[jj]\n - ha[aa]\n - ha[bb]") * P("taa[ijab]")
t2bb_eq = t2bb_eq + P("hb[II]\n + hb[JJ]\n - hb[AA]\n - hb[BB]") * P("tbb[IJAB]")
t2ab_eq = t2ab_eq + P("ha[ii]\n + hb[JJ]\n - ha[aa]\n - hb[BB]") * P("tab[iJaB]")
t2ba_eq = t2ba_eq + P("hb[II]\n + ha[jj]\n - hb[AA]\n - ha[bb]") * P("tab[jIbA]")

# non-iterative perturbative triples
pt3_eqs = []
ht123 = NR((h + (h ^ NR(t1 + t2 + t3))).expand(6))
for ex3 in [ex3aaa, ex3aab, ex3aba, ex3abb, ex3baa, ex3bab, ex3bba, ex3bbb]:
    pt3_eq = FC(FC(ex3 * ht123).substitute(PX(
        "taaa[ijkabc] = 0", "taab[ijKabC] = 0",
        "tabb[iJKaBC] = 0", "tbbb[IJKABC] = 0",
    )))
    pt3_eqs.append(pt3_eq)
pt3_en_eq = FC(t.conjugate() * NR((h ^ t3).expand(4)))

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.ExternalAlpha: "v",  WickIndexTypes.InactiveAlpha: "o",
        WickIndexTypes.ExternalBeta: "V",  WickIndexTypes.InactiveBeta: "O"}
    allowed_perms = {"oooo", "ovoo", "ovov", "oovv", "ovvo", "ovvv", "vvvv",
        "OOOO", "OVOO", "OVOV", "OOVV", "OVVO", "OVVV", "VVVV",
        "ooOO", "ovOO", "ovOV", "ooVV", "ovVO", "ovVV", "vvVV",
        "OVoo", "OOvv", "OVvo", "OVvv"}
    for term in eq.terms:
        for wt in term.tensors:
            if wt.name.startswith("v"):
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
                    if not found and wt.name in ["vab", "vba"]:
                        wt.name = "vba" if wt.name == "vab" else "vab"
                        wtx = wt * WickPermutation(VectorInt16([2, 3, 0, 1]))
                        for perm in wt.perms:
                            wtt = wtx * perm
                            k = ''.join([imap[wi.types] for wi in wtt.indices])
                            if k in allowed_perms:
                                wt.indices = wtt.indices
                                found = True
                                break
                    assert found

for eq in [en_eq, t1a_eq, t1b_eq, t2aa_eq, t2bb_eq, t2ab_eq, t2ba_eq, pt3_en_eq]:
    fix_eri_permutations(eq)
for eq in pt3_eqs:
    fix_eri_permutations(eq)

from pyscf import ao2mo
from pyscf.cc import uccsd

def wick_energy(cc, t1, t2, eris):
    assert isinstance(eris, uccsd._ChemistsERIs)
    assert cc.level_shift == 0
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    E = np.array(0.0)
    exec(en_eq.to_einsum(PT("E")), globals(), {
        "haie": eris.focka[:nocca, nocca:],
        "hbIE": eris.fockb[:noccb, noccb:],
        "vaaieie": np.asarray(eris.ovov),
        "vabieIE": np.asarray(eris.ovOV),
        "vbaIEei": np.asarray(eris.OVvo),
        "vbbIEIE": np.asarray(eris.OVOV),
        "taie": t1a,
        "tbIE": t1b,
        "taaiiee": t2aa,
        "tabiIeE": t2ab,
        "tbbIIEE": t2bb,
        "E": E
    })
    return E

def wick_update_amps(cc, t1, t2, eris):
    assert isinstance(eris, uccsd._ChemistsERIs)
    assert cc.level_shift == 0
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    t1anew = np.zeros_like(t1a)
    t1bnew = np.zeros_like(t1b)
    t2aanew = np.zeros_like(t2aa)
    t2bbnew = np.zeros_like(t2bb)
    t2abnew = np.zeros_like(t2ab)
    amps_eq = t1a_eq.to_einsum(PT("tanew[ia]")) + t1b_eq.to_einsum(PT("tbnew[IA]"))
    amps_eq += t2aa_eq.to_einsum(PT("taanew[ijab]")) + t2bb_eq.to_einsum(PT("tbbnew[IJAB]"))
    amps_eq += t2ab_eq.to_einsum(PT("tabnew[iJaB]")) + t2ba_eq.to_einsum(PT("tabnew[jIbA]"))
    eris_vvVV = np.zeros((nvira**2, nvirb**2), dtype=np.asarray(eris.vvVV).dtype)
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    eris_vvVV[(vtrila[0]*nvira+vtrila[1])[:, None], vtrilb[0]*nvirb+vtrilb[1]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[1]*nvira+vtrila[0])[:, None], vtrilb[1]*nvirb+vtrilb[0]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[0]*nvira+vtrila[1])[:, None], vtrilb[1]*nvirb+vtrilb[0]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[1]*nvira+vtrila[0])[:, None], vtrilb[0]*nvirb+vtrilb[1]] = np.asarray(eris.vvVV)
    eris_vvVV = eris_vvVV.reshape(nvira, nvira, nvirb, nvirb)
    exec(amps_eq, globals(), {
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
        "tanew": t1anew,
        "tbnew": t1bnew,
        "taanew": t2aanew,
        "tabnew": t2abnew,
        "tbbnew": t2bbnew
    })
    faii, faee = np.diag(eris.focka)[:nocca], np.diag(eris.focka)[nocca:]
    fbii, fbee = np.diag(eris.fockb)[:noccb], np.diag(eris.fockb)[noccb:]
    eaia = faii[:, None] - faee[None, :]
    ebia = fbii[:, None] - fbee[None, :]
    eaaijab = eaia[:, None, :, None] + eaia[None, :, None, :]
    ebbijab = ebia[:, None, :, None] + ebia[None, :, None, :]
    eabijab = eaia[:, None, :, None] + ebia[None, :, None, :]
    t1anew /= eaia
    t1bnew /= ebia
    t2aanew /= eaaijab
    t2bbnew /= ebbijab
    t2abnew /= eabijab + eabijab
    t1new = t1anew, t1bnew
    t2new = t2aanew, t2abnew, t2bbnew
    return t1new, t2new

def wick_t3_amps(cc, t1=None, t2=None, eris=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    assert isinstance(eris, uccsd._ChemistsERIs)
    assert cc.level_shift == 0
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    t3aaa = np.zeros((nocca, ) * 3 + (nvira, ) * 3)
    t3aab = np.zeros((nocca, ) * 2 + (noccb, ) + (nvira, ) * 2 + (nvirb, ))
    t3abb = np.zeros((nocca, ) + (noccb, ) * 2 + (nvira, ) + (nvirb, ) * 2)
    t3bbb = np.zeros((noccb, ) * 3 + (nvirb, ) * 3)

    eqaaa, eqaab, eqaba, eqabb, eqbaa, eqbab, eqbba, eqbbb = pt3_eqs
    amps_eq = "".join([
        eqaaa.to_einsum(PT("taaa[ijkabc]")), eqaab.to_einsum(PT("taab[ijKabC]")),
        eqaba.to_einsum(PT("taab[ikJacB]")), eqabb.to_einsum(PT("tabb[iJKaBC]")),
        eqbaa.to_einsum(PT("taab[jkIbcA]")), eqbab.to_einsum(PT("tabb[jIKbAC]")),
        eqbba.to_einsum(PT("tabb[kIJcAB]")), eqbbb.to_einsum(PT("tbbb[IJKABC]")),
    ])
    eris_vvVV = np.zeros((nvira**2, nvirb**2), dtype=np.asarray(eris.vvVV).dtype)
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    eris_vvVV[(vtrila[0]*nvira+vtrila[1])[:, None], vtrilb[0]*nvirb+vtrilb[1]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[1]*nvira+vtrila[0])[:, None], vtrilb[1]*nvirb+vtrilb[0]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[0]*nvira+vtrila[1])[:, None], vtrilb[1]*nvirb+vtrilb[0]] = np.asarray(eris.vvVV)
    eris_vvVV[(vtrila[1]*nvira+vtrila[0])[:, None], vtrilb[0]*nvirb+vtrilb[1]] = np.asarray(eris.vvVV)
    eris_vvVV = eris_vvVV.reshape(nvira, nvira, nvirb, nvirb)
    exec(amps_eq, globals(), {
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
        "taaiiee": t2aa,
        "tabiIeE": t2ab,
        "tbbIIEE": t2bb,
        "taaa": t3aaa,
        "taab": t3aab,
        "tabb": t3abb,
        "tbbb": t3bbb,
    })
    faii, faee = np.diag(eris.focka)[:nocca], np.diag(eris.focka)[nocca:]
    fbii, fbee = np.diag(eris.fockb)[:noccb], np.diag(eris.fockb)[noccb:]
    eaia = faii[:, None] - faee[None, :]
    ebia = fbii[:, None] - fbee[None, :]
    eaaijab = eaia[:, None, :, None] + eaia[None, :, None, :]
    ebbijab = ebia[:, None, :, None] + ebia[None, :, None, :]
    eabijab = eaia[:, None, :, None] + ebia[None, :, None, :]
    eaaaijkabc = eaaijab[:, :, None, :, :, None] + eaia[None, None, :, None, None, :]
    eaabijkabc = eaaijab[:, :, None, :, :, None] + ebia[None, None, :, None, None, :]
    eabbijkabc = eabijab[:, :, None, :, :, None] + ebia[None, None, :, None, None, :]
    ebbbijkabc = ebbijab[:, :, None, :, :, None] + ebia[None, None, :, None, None, :]
    t3aaa /= eaaaijkabc
    t3aab /= eaabijkabc + eaabijkabc + eaabijkabc
    t3abb /= eabbijkabc + eabbijkabc + eabbijkabc
    t3bbb /= ebbbijkabc
    t3 = t3aaa, t3aab, t3abb, t3bbb
    return t3

def wick_ccsd_t(cc, t1=None, t2=None, eris=None, t3=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    assert isinstance(eris, uccsd._ChemistsERIs)
    assert cc.level_shift == 0
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb = t2ab.shape[:2]
    if t3 is None:
        t3 = wick_t3_amps(cc, t1=t1, t2=t2, eris=eris)
    t3aaa, t3aab, t3abb, t3bbb = t3
    e_t = np.array(0.0)
    exec(pt3_en_eq.to_einsum(PT("E")), globals(), {
        "haie": eris.focka[:nocca, nocca:],
        "hbIE": eris.fockb[:noccb, noccb:],
        "vaaieii": np.asarray(eris.ovoo),
        "vaaieie": np.asarray(eris.ovov),
        "vaaieee": eris.get_ovvv(slice(None)),
        "vabieII": np.asarray(eris.ovOO),
        "vabieIE": np.asarray(eris.ovOV),
        "vabieEE": eris.get_ovVV(slice(None)),
        "vbaIEei": np.asarray(eris.OVvo),
        "vbaIEee": eris.get_OVvv(slice(None)),
        "vbaIEii": np.asarray(eris.OVoo),
        "vbbIEIE": np.asarray(eris.OVOV),
        "vbbIEEE": eris.get_OVVV(slice(None)),
        "vbbIEII": np.asarray(eris.OVOO),
        "taie": t1a,
        "tbIE": t1b,
        "taaiiee": t2aa,
        "tabiIeE": t2ab,
        "tbbIIEE": t2bb,
        "taaaiiieee": t3aaa,
        "taabiiIeeE": t3aab,
        "tabbiIIeEE": t3abb,
        "tbbbIIIEEE": t3bbb,
        "E": e_t
    })
    return e_t

class WickUCCSD(uccsd.UCCSD):
    def __init__(self, mf, **kwargs):
        uccsd.UCCSD.__init__(self, mf, **kwargs)
    energy = wick_energy
    update_amps = wick_update_amps
    ccsd_t = wick_ccsd_t

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyblock2.cc.eom_uccsd import WickUEOMIP
        return WickUEOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyblock2.cc.eom_uccsd import WickUEOMEA
        return WickUEOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eomee_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyblock2.cc.eom_uccsd import WickUEOMEESpinKeep
        return WickUEOMEESpinKeep(self).kernel(nroots, koopmans, guess, eris)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        from pyblock2.cc.lambda_uccsd import wick_kernel
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
            wick_kernel(self, eris, t1, t2, l1, l2,
                                max_cycle=self.max_cycle,
                                tol=self.conv_tol_normt,
                                verbose=self.verbose)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, **kwargs):
        from pyblock2.cc.rdm_uccsd import wick_make_rdm1
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return wick_make_rdm1(self, t1, t2, l1, l2, **kwargs)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, **kwargs):
        from pyblock2.cc.rdm_uccsd import wick_make_rdm2
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return wick_make_rdm2(self, t1, t2, l1, l2, **kwargs)

UCCSD = WickUCCSD

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='cc-pvdz')
    mf = scf.UHF(mol).run(conv_tol=1E-14)
    ccsd = uccsd.UCCSD(mf).run()
    print('E(T) = ', ccsd.ccsd_t())
    print('E-ee = ', ccsd.eomee_ccsd()[0])
    print('E-ip (right) = ', ccsd.ipccsd()[0])
    print('E-ea (right) = ', ccsd.eaccsd()[0])
    l1, l2 = ccsd.solve_lambda()
    dm1 = ccsd.make_rdm1()
    dm2 = ccsd.make_rdm2()
    wccsd = WickUCCSD(mf).run()
    print('E(T) = ', wccsd.ccsd_t())
    print('E-ee = ', wccsd.eomee_ccsd()[0])
    print('E-ip (right) = ', wccsd.ipccsd()[0])
    print('E-ea (right) = ', wccsd.eaccsd()[0])
    wl1, wl2 = wccsd.solve_lambda()
    print('lambda diff1 = ', [np.linalg.norm(np.array(l) - np.array(wl)) for l, wl in zip(l1, wl1)])
    print('lambda diff2 = ', [np.linalg.norm(np.array(l) - np.array(wl)) for l, wl in zip(l2, wl2)])
    wdm1 = wccsd.make_rdm1()
    wdm2 = wccsd.make_rdm2()
    print('dm diff1 = ', [np.linalg.norm(np.array(dm) - np.array(wdm)) for dm, wdm in zip(dm1, wdm1)])
    print('dm diff2 = ', [np.linalg.norm(np.array(dm) - np.array(wdm)) for dm, wdm in zip(dm2, wdm2)])
