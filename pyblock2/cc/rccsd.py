
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
UGA-CCSD [J. Chem. Phys. 89, 7382 (1988)] / CCSD(T) with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr, WickGraph
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
    perm_map[("t", 6)] = WickPermutation.pair_symmetric(3, False)
    perm_map[("r", 2)] = WickPermutation.non_symmetric()
    perm_map[("r", 4)] = WickPermutation.pair_symmetric(2, False)
    perm_map[("l", 2)] = WickPermutation.non_symmetric()
    perm_map[("l", 4)] = WickPermutation.pair_symmetric(2, False)

    defs = MapStrPWickTensorExpr()
    p = lambda x: WickExpr.parse(x, idx_map, perm_map).substitute(defs)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    pd = lambda x: WickExpr.parse_def(x, idx_map, perm_map)
    def px(x):
        defs = MapStrPWickTensorExpr()
        name = x.split("=")[0].split("[")[0].strip()
        defs[name] = WickExpr.parse_def(x, idx_map, perm_map)
        return defs

    return p, pt, pd, px, defs

P, PT, PD, PX, DEF = init_parsers() # parsers
SP = lambda x: x.simplify() # just simplify
FC = lambda x: x.expand(0).simplify() # fully contracted
NR = lambda x: x.expand(-1, True, False).simplify() # normal order
Z = P("") # zero

# definitions

DEF["h"] = PD("h[pq] = f[pq] \n - 2.0 SUM <j> v[pqjj] \n + SUM <j> v[pjjq]")
ehf = P("2 SUM <i> h[ii] \n + 2 SUM <ij> v[iijj] \n - SUM <ij> v[ijji]")

h1 = P("SUM <pq> h[pq] E1[p,q]")
h2 = 0.5 * P("SUM <pqrs> v[prqs] E2[pq,rs]")
t1 = P("SUM <ai> t[ia] E1[a,i]")
t2 = 0.5 * P("SUM <abij> t[ijab] E1[a,i] E1[b,j]")
t3 = (1.0 / 6.0) * P("SUM <abcijk> t[ijkabc] E1[a,i] E1[b,j] E1[c,k]")
ex1 = P("E1[i,a]")
ex2 = P("E1[i,a] E1[j,b]")
ex3 = P("E1[i,a] E1[j,b] E1[k,c]")

h = SP(h1 + h2 - ehf)
t = SP(t1 + t2)

HBarTerms = lambda h: [
    h, h ^ t,
    0.5 * ((h ^ t1) ^ t1) + ((h ^ t2) ^ t1) + 0.5 * ((h ^ t2) ^ t2),
    (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1) + 0.5 * (((h ^ t2) ^ t1) ^ t1),
    (1 / 24.0) * ((((h ^ t1) ^ t1) ^ t1) ^ t1)
]

hbar = sum(HBarTerms(h)[:5], Z)
en_eq = FC(sum(HBarTerms(h)[:3], Z))
t1_eq = FC(ex1 * sum(HBarTerms(h)[:4], Z))
t2_eq = FC(ex2 * hbar)

# need some rearrangements
t1_eq = 0.5 * t1_eq
ijmap = MapStrStr({ 'i': 'j', 'j': 'i' })
t2_eq = SP((1.0 / 3.0) * (t2_eq + 0.5 * t2_eq.index_map(ijmap)))

# move diag fock to lhs
t1_eq = SP(t1_eq + P("f[ii]\n - f[aa]") * P("t[ia]"))
t2_eq = SP(t2_eq + P("f[ii]\n + f[jj]\n - f[aa]\n - f[bb]") * P("t[ijab]"))

# non-iterative perturbative triples
pt3_eq = FC(ex3 * (h + (h ^ SP(t1 + t2 + t3))))
pt3_eq = FC(pt3_eq.substitute(PX("t[ijkabc] = 0")))
pt3_en_eq = FC(t.conjugate() * (h ^ t3))

def purify_pt3_eq(pt3_eq):
    main_terms = []
    rels = {
        MapStrStr({ 'i': 'j', 'j': 'k', 'k': 'i' }): 4,
        MapStrStr({ 'i': 'k', 'j': 'i', 'k': 'j' }): 4,
        MapStrStr({ 'i': 'j', 'j': 'i' }): -2,
        MapStrStr({ 'j': 'k', 'k': 'j' }): -2,
        MapStrStr({ 'k': 'i', 'i': 'k' }): -2,
    }
    f = 8.0
    for t in pt3_eq.terms:
        ok = True
        teq = WickExpr(t)
        for k, v in rels.items():
            peq = pt3_eq.index_map(k) * -v
            if len(SP(teq + peq).terms) != len(peq.terms) - 1:
                ok = False
                break
        if ok:
            main_terms.append(t * (1 / f))
    pt3_eq.terms = pt3_eq.terms.__class__(main_terms)

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Inactive: "I"}
    allowed_perms = {"IIII", "IEII", "IIEE", "IEEI", "IEIE", "IEEE", "EEEE"}
    for term in (eq.terms if isinstance(eq, WickExpr) else [t for g in eq.right for t in g.terms]):
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

purify_pt3_eq(pt3_eq)

gr_en_eq = WickGraph().add_term(PT("E"), en_eq).simplify()
gr_amps_eq = WickGraph().add_term(PT("t1new[ia]"), t1_eq).add_term(PT("t2new[ijab]"), t2_eq).simplify()
gr_pt3_eq = WickGraph().add_term(PT("t3[ijkabc]"), pt3_eq).simplify()
gr_pt3_en_eq = WickGraph().add_term(PT("E"), pt3_en_eq).simplify()

fix_eri_permutations(gr_amps_eq)
fix_eri_permutations(gr_pt3_eq)
fix_eri_permutations(gr_pt3_en_eq)

from pyscf.cc import rccsd

def wick_energy(cc, t1, t2, eris):
    assert isinstance(eris, rccsd._ChemistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    E = np.array(0.0)
    exec(gr_en_eq.to_einsum(), globals(), {
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
    exec(gr_amps_eq.to_einsum(), globals(), {
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

def wick_t3_amps(cc, t1=None, t2=None, eris=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    assert isinstance(eris, rccsd._ChemistsERIs)
    assert cc.level_shift == 0
    nocc, nvir = t1.shape

    t3 = np.zeros((nocc, ) * 3 + (nvir, ) * 3)
    exec(gr_pt3_eq.to_einsum(), globals(), {
        "vIIII": np.array(eris.oooo),
        "vIEII": np.array(eris.ovoo),
        "vIIEE": np.array(eris.oovv),
        "vIEEI": np.array(eris.ovvo),
        "vIEIE": np.array(eris.ovov),
        "vIEEE": np.array(eris.ovvv),
        "vEEEE": np.array(eris.vvvv),
        "tIIEE": t2,
        "t3": t3,
    })
    fii, faa = np.diag(eris.fock)[:nocc], np.diag(eris.fock)[nocc:]
    eia = fii[:, None] - faa[None, :]
    eiiaa = eia[:, None, :, None] + eia[None, :, None, :]
    eiiiaaa = eiiaa[:, :, None, :, :, None] + eia[None, None, :, None, None, :]
    t3 /= eiiiaaa
    return t3

def wick_ccsd_t(cc, t1=None, t2=None, eris=None, t3=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    assert isinstance(eris, rccsd._ChemistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    if t3 is None:
        t3 = wick_t3_amps(cc, t1=t1, t2=t2, eris=eris)
    e_t = np.array(0.0)
    exec(gr_pt3_en_eq.to_einsum(), globals(), {
        "fIE": eris.fock[:nocc, nocc:],
        "vIIEE": np.array(eris.oovv),
        "vIEII": np.array(eris.ovoo),
        "vIEIE": np.array(eris.ovov),
        "vIEEE": np.array(eris.ovvv),
        "tIE": t1,
        "tIIEE": t2,
        "tIIIEEE": t3,
        "E": e_t
    })
    return e_t

class WickRCCSD(rccsd.RCCSD):
    def __init__(self, mf, **kwargs):
        rccsd.RCCSD.__init__(self, mf, **kwargs)
    energy = wick_energy
    update_amps = wick_update_amps
    ccsd_t = wick_ccsd_t

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyblock2.cc.eom_rccsd import WickREOMIP
        return WickREOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyblock2.cc.eom_rccsd import WickREOMEA
        return WickREOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyblock2.cc.eom_rccsd import WickREOMEESinglet
        return WickREOMEESinglet(self).kernel(nroots, koopmans, guess, eris)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        from pyblock2.cc.lambda_rccsd import wick_kernel
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
        from pyblock2.cc.rdm_rccsd import wick_make_rdm1
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return wick_make_rdm1(self, t1, t2, l1, l2, **kwargs)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, **kwargs):
        from pyblock2.cc.rdm_rccsd import wick_make_rdm2
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return wick_make_rdm2(self, t1, t2, l1, l2, **kwargs)

RCCSD = WickRCCSD

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='cc-pvdz')
    mf = scf.RHF(mol).run(conv_tol=1E-14)
    ccsd = rccsd.RCCSD(mf).run()
    print('E(T) = ', ccsd.ccsd_t())
    print('E-ee (right) = ', ccsd.eomee_ccsd_singlet()[0])
    print('E-ip (right) = ', ccsd.ipccsd()[0])
    print('E-ip ( left) = ', ccsd.ipccsd(left=True)[0])
    print('E-ea (right) = ', ccsd.eaccsd()[0])
    print('E-ea ( left) = ', ccsd.eaccsd(left=True)[0])
    l1, l2 = ccsd.solve_lambda()
    dm1 = ccsd.make_rdm1()
    dm2 = ccsd.make_rdm2()
    wccsd = WickRCCSD(mf).run()
    print('E(T) = ', wccsd.ccsd_t())
    print('E-ee (right) = ', wccsd.eomee_ccsd_singlet()[0])
    print('E-ip (right) = ', wccsd.ipccsd()[0])
    print('E-ip ( left) = ', wccsd.ipccsd(left=True)[0])
    print('E-ea (right) = ', wccsd.eaccsd()[0])
    print('E-ea ( left) = ', wccsd.eaccsd(left=True)[0])
    wl1, wl2 = wccsd.solve_lambda()
    print('lambda diff = ', np.linalg.norm(l1 - wl1), np.linalg.norm(l2 - wl2))
    wdm1 = wccsd.make_rdm1()
    wdm2 = wccsd.make_rdm2()
    print('dm diff = ', np.linalg.norm(dm1 - wdm1), np.linalg.norm(dm2 - wdm2))
