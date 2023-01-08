
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
Internally-Contracted MR-REPT2 (MRLCC2) [J. Chem. Theory Comput. 13, 488 (2017)]
with equations derived on the fly.
need internal contraction module of block2.

A orthogonal basis is used in CG method.
"""

import numpy as np
import time

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation, WickTensorTypes
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

try:
    from . import eri_helper, dmrg_helper
except ImportError:
    import eri_helper, dmrg_helper

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.Inactive] = WickIndex.parse_set("mnxyijkl")
    idx_map[WickIndexTypes.Active] = WickIndex.parse_set("mnxyabcdefghpq")
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("mnxyrstu")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("w", 4)] = WickPermutation.qc_phys()

    defs = MapStrPWickTensorExpr()
    p = lambda x: WickExpr.parse(x, idx_map, perm_map).substitute(defs)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    pd = lambda x: WickExpr.parse_def(x, idx_map, perm_map)

    return p, pt, pd, defs

def _sum_indices(expr, idxs):
    for term in expr.terms:
        for idx in idxs:
            term.ctr_indices.add(idx)
    return expr

P, PT, PD, DEF = init_parsers() # parsers
SP = lambda x: x.expand().add_spin_free_trans_symm().remove_external().simplify()
SPR = lambda x: x.expand().add_spin_free_trans_symm().remove_external().remove_inactive().simplify()
Comm = lambda b, h, k, idxs: SP(_sum_indices(b.conjugate() * (h ^ k), idxs))
Norm = lambda b, k: SP(b.conjugate() * k)
Rhs = lambda b, k: SPR(b.conjugate() * k)

# See J. Chem. Theory Comput. 15, 2291 (2019) Eq. (11)
# Fink's Hamiltonian
h1 = P("""
    SUM <ij> h[ij] E1[i,j]
    SUM <ab> h[ab] E1[a,b]
    SUM <rs> h[rs] E1[r,s]
""")
h2 = P("""
    0.5 SUM <ijkl> w[ijkl] E2[ij,kl]
    0.5 SUM <abcd> w[abcd] E2[ab,cd]
    0.5 SUM <rstu> w[rstu] E2[rs,tu]
    0.5 SUM <iajb> w[iajb] E2[ia,jb] \n + 0.5 SUM <iajb> w[iabj] E2[ia,bj]
    0.5 SUM <irjs> w[irjs] E2[ir,js] \n + 0.5 SUM <irjs> w[irsj] E2[ir,sj]
    0.5 SUM <aibj> w[aibj] E2[ai,bj] \n + 0.5 SUM <aibj> w[aijb] E2[ai,jb]
    0.5 SUM <arbs> w[arbs] E2[ar,bs] \n + 0.5 SUM <arbs> w[arsb] E2[ar,sb]
    0.5 SUM <risj> w[risj] E2[ri,sj] \n + 0.5 SUM <risj> w[rijs] E2[ri,js]
    0.5 SUM <rasb> w[rasb] E2[ra,sb] \n + 0.5 SUM <rasb> w[rabs] E2[ra,bs]
""")
hd = h1 + h2
hfull = P("SUM <mn> h[mn] E1[m,n] \n + 0.5 SUM <mnxy> w[mnxy] E2[mn,xy]")

# convert < E1[p,a] E1[q,b] > ("dm2") to < E2[pq,ab] > ("E2"), etc.
# E2[pq,ab] = E1[p,a] E1[q,b] - delta[aq] E1[p,b]
pdm_eqs = [
    "E1[p,a] = E1[p,a]\n - E1[p,a]\n + dm1[pa]",
    "E2[pq,ab] = E2[pq,ab]\n - E1[p,a] E1[q,b]\n + dm2[paqb]",
    "E3[pqg,abc] = E3[pqg,abc]\n - E1[p,a] E1[q,b] E1[g,c]\n + dm3[paqbgc]",
    "E4[abcd,efgh] = E4[abcd,efgh]\n - E1[a,e] E1[b,f] E1[c,g] E1[d,h]\n + dm4[aebfcgdh]"
]

for k, eq in enumerate(pdm_eqs):
    name, expr = PD(eq)
    pdm_eqs[k] = SP(expr).to_einsum(name)


# def of ic-mrrept2 sub-spaces
sub_spaces = {
    "ijrskltu*": "E1[r,i] E1[s,j]",
    "rsiatukp*": "E1[r,i] E1[s,a]",
    "ijrakltp*": "E1[r,j] E1[a,i]",
    "rsabtupq*": "E1[r,b] E1[s,a]",
    "ijabklpq*": "E1[b,i] E1[a,j]",
    "irabktpq1": "E1[r,i] E1[a,b]",
    "irabktpq2": "E1[a,i] E1[r,b]",
    "rabctdef*": "E1[r,b] E1[a,c]",
    "iabckdef*": "E1[b,i] E1[a,c]"
}

ener_eqs = {} # Hamiltonian expectations
ener2_eqs = {} # Hamiltonian expectations
rhhk_eqs = {} # rhs equations
norm_eqs = {} # overlap equations
norm2_eqs = {} # overlap equations

for key, expr in sub_spaces.items():
    l = len(key)
    ket_bra_map = { k: v for k, v in zip(key[9 - l:4 - l], key[4 - l:-1]) }
    x = P("x%s[%s]" % ('2' if key[-1] == '2' else '', key[:4]))
    ket = P(expr)
    bra = ket.index_map(MapStrStr(ket_bra_map))
    rhhk_eqs[key] = Rhs(bra, hfull)
    norm_eqs[key] = Norm(bra, ket)
    x_idxs = x.terms[0].tensors[0].indices
    ener_eqs[key] = Comm(bra, hd, ket * x, x_idxs)
    if key[-1] in "12":
        ket2 = P(sub_spaces[key[:-1] + ('1' if key[-1] == '2' else '2')])
        x2 = P("x%s[%s]" % ('' if key[-1] == '2' else '2', key[:4]))
        ener2_eqs[key] = Comm(bra, hd, ket2 * x2, x_idxs)
        norm2_eqs[key] = Norm(bra, ket2)

allowed_perms = {"AAAA", "EAAA", "EAIA", "EAAI", "AAIA", "EEIA",
    "EAII", "EEAA", "AAII", "EEII", "EEEE", "IIII",
    "EIEI", "EAEA", "AIAI", "IIIE", "IAIE", "IIIA"}

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Active: "A",
        WickIndexTypes.Inactive: "I"}
    for term in eq.terms:
        for wt in term.tensors:
            if wt.name == "w":
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

for eq in [*ener_eqs.values(), *ener2_eqs.values(), *rhhk_eqs.values()]:
    fix_eri_permutations(eq)

def _key_idx(key):
    t = [WickIndexTypes.Inactive, WickIndexTypes.Active, WickIndexTypes.External]
    return [t.index(wi.types) for wi in PT("x[%s]" % key).indices]

def _conjugate_gradient(axop, x, b, xdot=np.dot, max_iter=1000, conv_thrd=5E-4, iprint=False):
    r = -axop(x) + b
    p = r.copy()
    error = xdot(p, r)
    if np.sqrt(np.abs(error)) < conv_thrd:
        func = xdot(x, b)
        if iprint:
            print("%5d %15.8f %9.2E" % (0, func, error))
        return func, x, 1
    old_error = error
    xiter = 0
    while xiter < max_iter:
        t = time.perf_counter()
        xiter += 1
        hp = axop(p)
        alpha = old_error / xdot(p, hp)
        x += alpha * p
        r -= alpha * hp
        z = r.copy()
        error = xdot(z, r)
        func = xdot(x, b)
        if iprint:
            print("%5d %15.8f %9.2E T = %.3f" % (xiter, func, error, time.perf_counter() - t))
        if np.sqrt(np.abs(error)) < conv_thrd:
            break
        else:
            beta = error / old_error
            old_error = error
            p[:] = beta * p + z
    if xiter == max_iter:
        print("Error : linear solver (cg) not converged!")
    return func, x, xiter + 1

from pyscf import lib

def kernel(ic, mc=None, mo_coeff=None, pdms=None, eris=None, root=None, iprint=None):
    if mc is None:
        mc = ic._mc
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if root is None and hasattr(ic, 'root'):
        root = ic.root
    if iprint is None and hasattr(ic, 'iprint'):
        iprint = ic.iprint
    ic.root = root
    ic.mo_coeff = mo_coeff
    ic.ci = mc.ci
    ic.mo_energy = mc.mo_energy
    tt = time.perf_counter()
    if pdms is None:
        t = time.perf_counter()
        pdms = eri_helper.init_pdms(mc=mc, pdm_eqs=pdm_eqs, root=root)
        tpdms = time.perf_counter() - t
    if eris is None:
        t = time.perf_counter()
        eris = eri_helper.init_eris(mc=mc, mo_coeff=mo_coeff, mrci=True)
        teris = time.perf_counter() - t
    ic.eris = eris
    assert isinstance(eris, eri_helper._ChemistsERIs)
    E1, E2, E3, E4 = pdms
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nvirt = len(ic.mo_energy) - nocc
    mdict = {
        "E1": E1, "E2": E2, "E3": E3, "E4": E4,
        "deltaII": np.eye(ncore), "deltaEE": np.eye(nvirt),
        "deltaAA": np.eye(ncas), 
        "ident1": np.ones((1, )),
        "ident2": np.ones((1, 1, )),
        "ident3": np.ones((1, 1, 1, )),
        **{ 'h' + a + b: eris.get_h1(a + b) for a in 'IAE' for b in 'IAE' },
        **{ 'w' + k: eris.get_phys(k) for k in allowed_perms }
    }

    ic.sub_eners = {}
    ic.sub_times = {'pdms': tpdms, 'eris': teris}
    niter = 0
    for key in sub_spaces:
        t = time.perf_counter()
        skey = key[:4].split('a')[0]
        if E4 is None and "abc" in key:
            irkmap = {'i': 'aaac', 'r': 'aaav'}
            ic.sub_eners[skey] = dmrg_helper.dmrg_response_singles(mc, eris, E1,
                irkmap[key[0]], theory='mrrept2', root=root)
            lib.logger.note(ic, "E(%s-%4s) = %20.14f",
                ic.__class__.__name__.replace("IC", "UC"), 'mps' + skey, ic.sub_eners[skey])
            ic.sub_times[skey] = time.perf_counter() - t
            continue
        if key[-1] == '2':
            continue
        rkey = key[:-1][4:]
        nkey = key[:-1][4:] + key[:-1][:4]
        xindex = "".join(["IAE"[i] for i in _key_idx(rkey)])
        s = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(nkey)])
        b = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(rkey)])
        pt2_seqs = norm_eqs[key].to_einsum(PT("s[%s]" % nkey))
        pt2_beqs = rhhk_eqs[key].to_einsum(PT("b[%s]" % rkey))
        pt2_axeqs = ener_eqs[key].to_einsum(PT("ax[%s]" % rkey))
        if key[-1] == '1':
            key2 = key[:-1] + '2'
            s12 = np.zeros_like(s)
            s21 = np.zeros_like(s)
            s22 = np.zeros_like(s)
            b2 = np.zeros_like(b)
            pt2_seqs += norm2_eqs[key].to_einsum(PT("s12[%s]" % nkey))
            pt2_seqs += norm_eqs[key2].to_einsum(PT("s22[%s]" % nkey))
            pt2_seqs += norm2_eqs[key2].to_einsum(PT("s21[%s]" % nkey))
            pt2_beqs += rhhk_eqs[key2].to_einsum(PT("b2[%s]" % rkey))
            pt2_axeqs += ener2_eqs[key].to_einsum(PT("ax[%s]" % rkey))
            pt2_axeqs += ener_eqs[key2].to_einsum(PT("ax2[%s]" % rkey))
            pt2_axeqs += ener2_eqs[key2].to_einsum(PT("ax2[%s]" % rkey))
            exec(pt2_seqs, globals(), { "s": s, "s12": s12, "s21": s21, "s22": s22, **mdict })
            ss = np.concatenate((s[..., None], s12[..., None], s21[..., None], s22[..., None]), axis=-1)
            ns = np.prod(s.shape[:len(s.shape) // 2], dtype=int)
            ss = ss.reshape((ns, ns, 2, 2)).transpose((2, 0, 3, 1))
            sw, su = np.linalg.eigh(ss.reshape((ns * 2, ns * 2)))
            idx = sw > ic.trunc_thrds
            sf = su[:, idx] * (sw[idx] ** (-0.5))
            sb = su[:, idx] * (sw[idx] ** 0.5)
            sf = sf.reshape((2, *s.shape[:len(s.shape) // 2], -1))
            sb = sb.reshape((2, *s.shape[:len(s.shape) // 2], -1))
            exec(pt2_beqs, globals(), { "b": b, "b2": b2, **mdict })
            def trans_forth(g, sf=sf):
                return np.einsum("v%s,v%sx->x" % (rkey, rkey), g, sf, optimize=True)
            def trans_back(g, sb=sb):
                return np.einsum("x,v%sx->v%s" % (rkey, rkey), g, sb, optimize=True)
            xdot = lambda a, b: (trans_back(a) * trans_back(b)).sum()
            def axop(ppx, eqs=pt2_axeqs, xid=xindex):
                px = trans_back(ppx)
                pax = np.zeros_like(px)
                x, x2 = px[0], px[1]
                ax, ax2 = pax[0], pax[1]
                exec(eqs, globals(), { "x" + xid: x, "x2" + xid: x2,
                    "ax": ax, "ax2": ax2, **mdict })
                return trans_forth(pax)
            pb = trans_forth(np.concatenate((b[None], b2[None]), axis=0))
        else:
            exec(pt2_seqs, globals(), { "s": s, **mdict })
            ns = np.prod(s.shape[:len(s.shape) // 2], dtype=int)
            sw, su = np.linalg.eigh(s.reshape((ns, ns)))
            idx = sw > ic.trunc_thrds
            sf = su[:, idx] * (sw[idx] ** (-0.5))
            sb = su[:, idx] * (sw[idx] ** 0.5)
            sf = sf.reshape((*s.shape[:len(s.shape) // 2], -1))
            sb = sb.reshape((*s.shape[:len(s.shape) // 2], -1))
            exec(pt2_beqs, globals(), { "b": b, **mdict })
            def trans_forth(g, sf=sf):
                return np.einsum("%s,%sx->x" % (rkey, rkey), g, sf, optimize=True)
            def trans_back(g, sb=sb):
                return np.einsum("x,%sx->%s" % (rkey, rkey), g, sb, optimize=True)
            xdot = lambda a, b: (trans_back(a) * trans_back(b)).sum()
            def axop(px, eqs=pt2_axeqs, xid=xindex):
                x = trans_back(px)
                ax = np.zeros_like(x)
                exec(eqs, globals(), { "x" + xid: x, "ax": ax, **mdict })
                return trans_forth(ax)
            pb = trans_forth(b)
        func, _, niterx = _conjugate_gradient(axop, pb.copy(), pb, xdot, iprint=iprint)
        niter += niterx
        if skey not in ic.sub_eners:
            ic.sub_eners[skey] = -func
            ic.sub_times[skey] = time.perf_counter() - t
        else:
            ic.sub_eners[skey] += -func
            ic.sub_times[skey] += time.perf_counter() - t
        if key[-1] in "-1*":
            lib.logger.note(ic, "E(%s-%4s) = %20.14f Niter = %5d",
                ic.__class__.__name__, skey, ic.sub_eners[skey], niter)
            niter = 0
    ic.e_corr = sum(ic.sub_eners.values())
    ic.sub_times['total'] = time.perf_counter() - tt
    lib.logger.note(ic, 'E(%s) = %.16g  E_corr_pt = %.16g',
        ic.__class__.__name__, ic.e_tot, ic.e_corr)
    lib.logger.note(ic, "Timings = %s",
        " | ".join(["%s = %7.2f" % (k, v) for k, v in ic.sub_times.items()]))

class WickICMRREPT2(lib.StreamObject):
    def __init__(self, mc):
        self._mc = mc
        assert mc.canonicalization
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.e_corr = None
        self.trunc_thrds = 1E-4

    @property
    def e_tot(self):
        if hasattr(self._mc, 'e_states') and hasattr(self, 'root'):
            return np.asarray(self.e_corr) + self._mc.e_states[self.root]
        elif hasattr(self, 'root') and isinstance(self._mc.e_tot, np.ndarray):
            return np.asarray(self.e_corr) + self._mc.e_tot[self.root]
        else:
            return np.asarray(self.e_corr) + self._mc.e_tot

    kernel = kernel

ICMRREPT2 = WickICMRREPT2

if __name__ == "__main__":

    from pyscf import gto, scf, mcscf

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='6-31g', spin=2)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    # Example 1 - single state
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.run()
    wsc = WickICMRREPT2(mc).set(iprint=True).run()
    # converged SCF energy = -149.528026672327
    # CASSCF energy = -149.63656327982
    # E(WickICMRREPT2) = -149.7947264253112  E_corr_pt = -0.15816314549156
