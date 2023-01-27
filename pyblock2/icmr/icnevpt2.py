
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
Internally-Contracted NEVPT2 [J. Chem. Phys. 117, 9138 (2002)]
with equations derived on the fly.
need internal contraction module of block2.

CG method is used for solving the sparse linear problem.
"""

import numpy as np
import time

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
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
Rhs = lambda b, k: SPR(b.conjugate() * k)

hi = P("SUM <i> orbe[i] E1[i,i]\n + SUM <r> orbe[r] E1[r,r]")
h1 = P("SUM <ab> f[ab] E1[a,b]")
h2 = P("0.5 SUM <abcd> w[abcd] E2[ab,cd]")
hd = hi + h1 + h2
hfull = P("SUM <mn> f[mn] E1[m,n] \n - 2.0 SUM <mnj> w[mjnj] E1[m,n]\n"
    "+ 1.0 SUM <mnj> w[mjjn] E1[m,n]\n + 0.5 SUM <mnxy> w[mnxy] E2[mn,xy]")

# convert < E1[p,a] E1[q,b] > ("dm2") to < E2[pq,ab] > ("E2"), etc.
pdm_eqs = [
    "E1[p,a] = E1[p,a]\n - E1[p,a]\n + dm1[pa]",
    "E2[pq,ab] = E2[pq,ab]\n - E1[p,a] E1[q,b]\n + dm2[paqb]",
    "E3[pqg,abc] = E3[pqg,abc]\n - E1[p,a] E1[q,b] E1[g,c]\n + dm3[paqbgc]",
    "E4[abcd,efgh] = E4[abcd,efgh]\n - E1[a,e] E1[b,f] E1[c,g] E1[d,h]\n + dm4[aebfcgdh]"
]

for k, eq in enumerate(pdm_eqs):
    name, expr = PD(eq)
    pdm_eqs[k] = SP(expr).to_einsum(name)

# def of ic-nevpt2 sub-spaces
sub_spaces = {
    "ijrs+": "E1[r,i] E1[s,j] \n + E1[s,i] E1[r,j]",
    "ijrs-": "E1[r,i] E1[s,j] \n - E1[s,i] E1[r,j]",
    "rsiap+": "E1[r,i] E1[s,a] \n + E1[s,i] E1[r,a]",
    "rsiap-": "E1[r,i] E1[s,a] \n - E1[s,i] E1[r,a]",
    "ijrap+": "E1[r,j] E1[a,i] \n + E1[r,i] E1[a,j]",
    "ijrap-": "E1[r,j] E1[a,i] \n - E1[r,i] E1[a,j]",
    "rsabpq+": "E1[r,b] E1[s,a] \n + E1[s,b] E1[r,a]",
    "rsabpq-": "E1[r,b] E1[s,a] \n - E1[s,b] E1[r,a]",
    "ijabpq+": "E1[b,i] E1[a,j] \n + E1[b,j] E1[a,i]",
    "ijabpq-": "E1[b,i] E1[a,j] \n - E1[b,j] E1[a,i]",
    "irabpq1": "E1[r,i] E1[a,b]",
    "irabpq2": "E1[a,i] E1[r,b]",
    "rabcpqg*": "E1[r,b] E1[a,c]",
    "iabcpqg*": "E1[b,i] E1[a,c]"
}

ener_eqs = {} # Hamiltonian expectations
ener2_eqs = {} # Hamiltonian expectations
rhhk_eqs = {} # rhs equations

for key, expr in sub_spaces.items():
    l = len(key)
    ket_bra_map = { k: v for k, v in zip(key[9 - l:4 - l], key[4 - l:-1]) }
    x = P("x%s[%s]" % ('2' if key[-1] == '2' else '', key[:4]))
    ket = P(expr)
    bra = ket.index_map(MapStrStr(ket_bra_map))
    rhhk_eqs[key] = Rhs(bra, hfull)
    act_idxs = x.terms[0].tensors[0].indices[9 - l:4]
    ener_eqs[key] = Comm(bra, hd, ket * x, act_idxs)
    if key[-1] in "12":
        ket2 = P(sub_spaces[key[:-1] + ('1' if key[-1] == '2' else '2')])
        x2 = P("x%s[%s]" % ('' if key[-1] == '2' else '2', key[:4]))
        ener2_eqs[key] = Comm(bra, hd, ket2 * x2, act_idxs)

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Active: "A",
        WickIndexTypes.Inactive: "I"}
    allowed_perms = {"AAAA", "EAAA", "EAIA", "EAAI", "AAIA",
                     "EEIA", "EAII", "EEAA", "AAII", "EEII"}
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

def _grid_restrict(key, grid, restrict_cas, no_eq):
    ixx = []
    wis = PT("x[%s]" % key).indices
    contig = lambda a, b: b == chr(ord(a) + 1)
    for i, wi in enumerate(wis):
        if not restrict_cas and wi.types == WickIndexTypes.Active:
            continue
        if i != 0 and wis[i].types == wis[i - 1].types:
            if wi.types == WickIndexTypes.Active:
                if not contig(wis[i - 1].name, wis[i].name):
                    continue
                if i + 1 < len(wis) and contig(wis[i].name, wis[i + 1].name):
                    continue
                if i - 2 >= 0 and contig(wis[i - 2].name, wis[i - 1].name):
                    continue
            if no_eq:
                ixx.append(grid[i - 1] < grid[i])
            else:
                ixx.append(grid[i - 1] <= grid[i])
    return np.bitwise_and.reduce(ixx)

def _conjugate_gradient(axop, x, b, xdot=np.dot, max_iter=5000, conv_thrd=5E-4, iprint=False):
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

from pyscf import lib, mcscf

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
    if not ic.canonicalized:
        xci = mc.ci
        if root is not None and (isinstance(xci, list) or isinstance(xci, range)):
            if isinstance(mc.fcisolver, mcscf.addons.StateAverageFCISolver):
                dm1 = mc.fcisolver.states_make_rdm1(xci, mc.ncas, mc.nelecas)[root]
            else:
                dm1 = mc.fcisolver.make_rdm1(xci[root], mc.ncas, mc.nelecas)
            xci = xci[root]
        else:
            dm1 = mc.fcisolver.make_rdm1(xci, mc.ncas, mc.nelecas)
        ic.mo_coeff, xci, ic.mo_energy = mc.canonicalize(
            ic.mo_coeff, ci=xci, cas_natorb=False, casdm1=dm1,
            verbose=ic.verbose)
        if root is not None and isinstance(ic.ci, list):
            ic.ci[root] = xci
        elif root is not None:
            ic.ci = xci
    tt = time.perf_counter()
    if pdms is None:
        t = time.perf_counter()
        pdms = eri_helper.init_pdms(mc=mc, pdm_eqs=pdm_eqs, root=root)
        tpdms = time.perf_counter() - t
    if eris is None:
        t = time.perf_counter()
        eris = eri_helper.init_eris(mc=mc, mo_coeff=ic.mo_coeff)
        teris = time.perf_counter() - t
    ic.eris = eris
    assert isinstance(eris, eri_helper._ChemistsERIs)
    E1, E2, E3, E4 = pdms
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nvirt = len(ic.mo_energy) - nocc
    orbeI = ic.mo_energy[:ncore]
    orbeE = ic.mo_energy[nocc:]
    wkeys = ["wAAAA", "wEAAA", "wEAIA", "wEAAI", "wAAIA", 
             "wEEIA", "wEAII", "wEEAA", "wAAII", "wEEII"]
    mdict = {
        "E1": E1, "E2": E2, "E3": E3, "E4": E4,
        "deltaII": np.eye(ncore), "deltaEE": np.eye(nvirt),
        "deltaAA": np.eye(ncas), 
        "ident1": np.ones((1, )),
        "ident2": np.ones((1, 1, )),
        "ident3": np.ones((1, 1, 1, )),
        "fAA": eris.h1eff[ncore:nocc, ncore:nocc],
        "fAI": eris.h1eff[ncore:nocc, :ncore],
        "fEI": eris.h1eff[nocc:, :ncore],
        "fEA": eris.h1eff[nocc:, ncore:nocc],
        "orbeE": orbeE, "orbeI": orbeI,
        **{ k: eris.get_phys(k[1:]) for k in wkeys }
    }

    ic.sub_eners = {}
    ic.sub_times = {'pdms': tpdms, 'eris': teris}
    niter = 0
    for key in sub_spaces:
        t = time.perf_counter()
        l = len(key)
        skey = key[:9 - l]
        if E4 is None and "abc" in key:
            irkmap = {'i': 'aaac', 'r': 'aaav'}
            ic.sub_eners[skey] = dmrg_helper.dmrg_response_singles(mc, eris, E1,
                irkmap[key[0]], theory='nevpt2', root=root)
            lib.logger.note(ic, "E(%s-%4s) = %20.14f",
                ic.__class__.__name__.replace("IC", "UC"), 'mps' + skey, ic.sub_eners[skey])
            ic.sub_times[skey] = time.perf_counter() - t
            continue
        if key[-1] == '2':
            continue
        rkey = key[:9 - l] + key[4 - l:-1]
        xindex = "".join(["IAE"[i] for i in _key_idx(rkey)])
        b = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(rkey)])
        pt2_beqs = rhhk_eqs[key].to_einsum(PT("b[%s]" % rkey))
        pt2_axeqs = ener_eqs[key].to_einsum(PT("ax[%s]" % rkey))
        if key[-1] == '1':
            key2 = key[:-1] + '2'
            b2 = np.zeros_like(b)
            pt2_beqs += rhhk_eqs[key2].to_einsum(PT("b2[%s]" % rkey))
            pt2_axeqs += ener2_eqs[key].to_einsum(PT("ax[%s]" % rkey))
            pt2_axeqs += ener_eqs[key2].to_einsum(PT("ax2[%s]" % rkey))
            pt2_axeqs += ener2_eqs[key2].to_einsum(PT("ax2[%s]" % rkey))
            exec(pt2_beqs, globals(), { "b": b, "b2": b2, **mdict })
            xdot = lambda a, b: (a * b).sum()
            def axop(px, eqs=pt2_axeqs, xid=xindex):
                pax = np.zeros_like(px)
                x, x2 = px[0], px[1]
                ax, ax2 = pax[0], pax[1]
                exec(eqs, globals(), { "x" + xid: x, "x2" + xid: x2,
                    "ax": ax, "ax2": ax2, **mdict })
                return pax
            pb = np.concatenate((b[None], b2[None]), axis=0)
        else:
            exec(pt2_beqs, globals(), { "b": b, **mdict })
            if 9 - len(key) >= 2:
                grid = np.indices(b.shape, dtype=np.int16)
                idx = _grid_restrict(rkey, grid, key[-1] in '+-', key[-1] == '-')
            else:
                idx = slice(None)
            if len(key) - 5 == 2 and key[-1] in '+-':
                ridx = ~idx
            else:
                ridx = slice(0)
            xdot = lambda a, b, idx=idx: (a[idx] * b[idx]).sum()
            def axop(x, eqs=pt2_axeqs, xid=xindex, ridx=ridx):
                x[ridx] = 0
                ax = np.zeros_like(x)
                exec(eqs, globals(), { "x" + xid: x, "ax": ax, **mdict })
                ax[ridx] = 0
                return ax
            pb = b
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

class WickICNEVPT2(lib.StreamObject):
    def __init__(self, mc):
        self._mc = mc
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.canonicalized = False
        self.e_corr = None

    @property
    def e_tot(self):
        if hasattr(self._mc, 'e_states') and hasattr(self, 'root'):
            return np.asarray(self.e_corr) + self._mc.e_states[self.root]
        elif hasattr(self, 'root') and isinstance(self._mc.e_tot, np.ndarray):
            return np.asarray(self.e_corr) + self._mc.e_tot[self.root]
        else:
            return np.asarray(self.e_corr) + self._mc.e_tot

    kernel = kernel

ICNEVPT2 = WickICNEVPT2

if __name__ == "__main__":

    from pyscf import gto, scf, mcscf

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    # Example 1 - single state
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.run()
    wsc = WickICNEVPT2(mc).run()
    # converged SCF energy = -149.608181589162
    # CASSCF energy = -149.708657770028
    # E(WickICNEVPT2) = -149.9601360650626  E_corr_pt = -0.2514782950349269
    # ref -149.708657773372 (casscf) -0.2514798828 (pc) -149.960137653992 (tot-pc)

    # Example 2 - CASCI multi-state
    mc2 = mcscf.CASCI(mf, 6, 8)
    mc2.fcisolver.nroots = 3
    mc2.fcisolver.conv_tol = 1e-14
    mc2.canonicalization = True
    mc2.kernel(mc.mo_coeff)
    # [ -149.708657770072 -149.480535581399 -149.480535581399 ]

    wsc = WickICNEVPT2(mc2).run(root=0)
    wsc = WickICNEVPT2(mc2).run(root=1)
    wsc = WickICNEVPT2(mc2).run(root=2)
    # [ -0.2514783034161103 -0.2468313797875435 -0.2468313797875419 ]

    # Example 3 - CASSCF state-average
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.state_average_([1 / 3] * 3)
    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.run()
    # [ -149.706807301761 -149.484319802309 -149.484319802303 ]
    wsc = WickICNEVPT2(mc).run(root=0)
    wsc = WickICNEVPT2(mc).run(root=1)
    wsc = WickICNEVPT2(mc).run(root=2)
    # E(WickICNEVPT2) = -149.9591789667880  E_corr_pt = -0.252371665027283
    # E(WickICNEVPT2) = -149.7264213542754  E_corr_pt = -0.2421015519665067
    # E(WickICNEVPT2) = -149.7264213421491  E_corr_pt = -0.2421015398465075

    # ref casscf = [ -149.706807583050 -149.484319661994 -149.484319661994 ]
    # ref 0 =  -0.2498431015 (SC) -0.2522554873 (PC) -149.959063070398 (TOT-PC)
    # ref 1 =  -0.2410187659 (SC) -0.2422012621 (PC) -149.726520924054 (TOT-PC)
    # ref 2 =  -0.2410187659 (SC) -0.2422012621 (PC) -149.726520924054 (TOT-PC)
