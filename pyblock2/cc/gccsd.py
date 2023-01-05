
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
CCSD and CCSD(T) in general orbitals with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr
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
    perm_map[("t", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("t", 6)] = WickPermutation.pair_anti_symmetric(3)
    perm_map[("r", 2)] = WickPermutation.non_symmetric()
    perm_map[("r", 4)] = WickPermutation.pair_anti_symmetric(2)
    perm_map[("l", 2)] = WickPermutation.non_symmetric()
    perm_map[("l", 4)] = WickPermutation.pair_anti_symmetric(2)

    p = lambda x: WickExpr.parse(x, idx_map, perm_map)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    def px(x):
        defs = MapStrPWickTensorExpr()
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

h1 = P("SUM <pq> h[pq] C[p] D[q]")
h2 = 0.25 * P("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
t1 = P("SUM <ai> t[ia] C[a] D[i]")
t2 = 0.25 * P("SUM <abij> t[ijab] C[a] C[b] D[j] D[i]")
t3 = (1.0 / 36.0) * P("SUM <abcijk> t[ijkabc] C[a] C[b] C[c] D[k] D[j] D[i]")
ex1 = P("C[i] D[a]")
ex2 = P("C[i] C[j] D[b] D[a]")
ex3 = P("C[i] C[j] C[k] D[c] D[b] D[a]")

h = NR(h1 + h2)
t = NR(t1 + t2)
hbar = HBar(h, t, 4)

en_eq = FC(HBar(h, t, 2))
t1_eq = FC(ex1 * HBar(h, t, 3))
t2_eq = FC(ex2 * hbar)

# add diag fock term to lhs and rhs of the equation
t1_eq = t1_eq + P("h[ii]\n - h[aa]") * P("t[ia]")
t2_eq = t2_eq + P("h[ii]\n + h[jj]\n - h[aa]\n - h[bb]") * P("t[ijab]")

# non-iterative perturbative triples
pt3_eq = FC(ex3 * (h + (h ^ NR(t1 + t2 + t3))))
pt3_eq = FC(pt3_eq.substitute(PX("t[ijkabc] = 0")))
pt3_en_eq = FC(t.conjugate() * (h ^ t3))

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
fix_eri_permutations(pt3_eq)
fix_eri_permutations(pt3_en_eq)

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

def wick_t3_amps(cc, t1=None, t2=None, eris=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    assert isinstance(eris, gccsd._PhysicistsERIs)
    nocc, nvir = t1.shape
    t3 = np.zeros((nocc, ) * 3 + (nvir, ) * 3)
    amps_eq = pt3_eq.to_einsum(PT("t3[ijkabc]"))
    exec(amps_eq, globals(), {
        "vIIII": np.array(eris.oooo),
        "vIIIE": np.array(eris.ooov),
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
    assert isinstance(eris, gccsd._PhysicistsERIs)
    assert cc.level_shift == 0
    nocc = t1.shape[0]
    if t3 is None:
        t3 = wick_t3_amps(cc, t1=t1, t2=t2, eris=eris)
    e_t = np.array(0.0)
    exec(pt3_en_eq.to_einsum(PT("E")), globals(), {
        "hIE": eris.fock[:nocc, nocc:],
        "vIIEE": np.array(eris.oovv),
        "vIIIE": np.array(eris.ooov),
        "vIEEE": np.array(eris.ovvv),
        "tIE": t1,
        "tIIEE": t2,
        "tIIIEEE": t3,
        "E": e_t
    })
    return e_t

class WickGCCSD(gccsd.GCCSD):
    def __init__(self, mf, **kwargs):
        gccsd.GCCSD.__init__(self, mf, **kwargs)
    energy = wick_energy
    update_amps = wick_update_amps
    ccsd_t = wick_ccsd_t

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyblock2.cc.eom_gccsd import WickGEOMIP
        return WickGEOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyblock2.cc.eom_gccsd import WickGEOMEA
        return WickGEOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyblock2.cc.eom_gccsd import WickGEOMEE
        return WickGEOMEE(self).kernel(nroots, koopmans, guess, eris)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        from pyblock2.cc.lambda_gccsd import wick_kernel
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
        from pyblock2.cc.rdm_gccsd import wick_make_rdm1
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return wick_make_rdm1(self, t1, t2, l1, l2, **kwargs)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, **kwargs):
        from pyblock2.cc.rdm_gccsd import wick_make_rdm2
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return wick_make_rdm2(self, t1, t2, l1, l2, **kwargs)

GCCSD = WickGCCSD

if __name__ == "__main__":

    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='cc-pvdz')
    mf = scf.GHF(mol).run(conv_tol=1E-14)
    ccsd = gccsd.GCCSD(mf).run()
    print('E(T) = ', ccsd.ccsd_t())
    print('E-ee = ', ccsd.eeccsd()[0])
    print('E-ip (right) = ', ccsd.ipccsd()[0])
    print('E-ip ( left) = ', ccsd.ipccsd(left=True)[0])
    print('E-ea (right) = ', ccsd.eaccsd()[0])
    print('E-ea ( left) = ', ccsd.eaccsd(left=True)[0])
    l1, l2 = ccsd.solve_lambda()
    dm1 = ccsd.make_rdm1()
    dm2 = ccsd.make_rdm2()
    wccsd = WickGCCSD(mf).run()
    print('E(T) = ', wccsd.ccsd_t())
    print('E-ee = ', wccsd.eeccsd()[0])
    print('E-ip (right) = ', wccsd.ipccsd()[0])
    print('E-ip ( left) = ', wccsd.ipccsd(left=True)[0])
    print('E-ea (right) = ', wccsd.eaccsd()[0])
    print('E-ea ( left) = ', wccsd.eaccsd(left=True)[0])
    wl1, wl2 = wccsd.solve_lambda()
    print('lambda diff = ', np.linalg.norm(l1 - wl1), np.linalg.norm(l2 - wl2))
    wdm1 = wccsd.make_rdm1()
    wdm2 = wccsd.make_rdm2()
    print('dm diff = ', np.linalg.norm(dm1 - wdm1), np.linalg.norm(dm2 - wdm2))
