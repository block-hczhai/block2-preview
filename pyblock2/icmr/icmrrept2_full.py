
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

Full Hamiltonian matrix is build for solving the linear problem.
May consume large amount of memory.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

import numpy as np
import time

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

P, PT, PD, DEF = init_parsers() # parsers
SP = lambda x: x.expand().add_spin_free_trans_symm().remove_external().simplify()
SPR = lambda x: x.expand().add_spin_free_trans_symm().remove_external().remove_inactive().simplify()
Comm = lambda b, h, k: SP(b.conjugate() * (h ^ k))
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
norm_eqs = {} # overlap equations
norm2_eqs = {} # overlap equations
rhhk_eqs = {} # rhs equations

for key, expr in sub_spaces.items():
    l = len(key)
    ket_bra_map = { k: v for k, v in zip(key[9 - l:4 - l], key[4 - l:-1]) }
    ket = P(expr)
    bra = ket.index_map(MapStrStr(ket_bra_map))
    rhhk_eqs[key] = Rhs(bra, hfull)
    ener_eqs[key] = Comm(bra, hd, ket)
    norm_eqs[key] = Norm(bra, ket)
    if key[-1] in "12":
        ket2 = P(sub_spaces[key[:-1] + ('1' if key[-1] == '2' else '2')])
        ener2_eqs[key] = Comm(bra, hd, ket2)
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

def _linear_solve(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]

from pyscf import lib

def kernel(ic, mc=None, mo_coeff=None, pdms=None, eris=None, root=None):
    if mc is None:
        mc = ic._mc
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if root is None and hasattr(ic, 'root'):
        root = ic.root
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
        hkey = key[:-1][4:] + key[:-1][:4]
        rhhk = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(rkey)])
        ener = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(hkey)])
        norm = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(hkey)])
        nr_eqs = norm_eqs[key].to_einsum(PT("norm[%s]" % hkey))
        pt2_eqs = rhhk_eqs[key].to_einsum(PT("rhhk[%s]" % rkey))
        pt2_eqs += ener_eqs[key].to_einsum(PT("ener[%s]" % hkey))
        if key[-1] == '1':
            key2 = key[:-1] + '2'
            rhhk2 = np.zeros_like(rhhk)
            ener12 = np.zeros_like(ener)
            ener21 = np.zeros_like(ener)
            ener22 = np.zeros_like(ener)
            norm12 = np.zeros_like(norm)
            norm21 = np.zeros_like(norm)
            norm22 = np.zeros_like(norm)
            pt2_eqs += rhhk_eqs[key2].to_einsum(PT("rhhk2[%s]" % rkey))
            pt2_eqs += ener2_eqs[key].to_einsum(PT("ener12[%s]" % hkey))
            pt2_eqs += ener_eqs[key2].to_einsum(PT("ener22[%s]" % hkey))
            pt2_eqs += ener2_eqs[key2].to_einsum(PT("ener21[%s]" % hkey))
            nr_eqs += norm2_eqs[key].to_einsum(PT("norm12[%s]" % hkey))
            nr_eqs += norm_eqs[key2].to_einsum(PT("norm22[%s]" % hkey))
            nr_eqs += norm2_eqs[key2].to_einsum(PT("norm21[%s]" % hkey))
            exec(nr_eqs, globals(), {
                "norm": norm, "norm12": norm12, "norm21": norm21,
                "norm22": norm22, **mdict
            })
            exec(pt2_eqs, globals(), {
                "rhhk": rhhk, "rhhk2": rhhk2, "ener": ener, "ener12": ener12,
                "ener22": ener22, "ener21": ener21, **mdict
            })
            norm = np.concatenate((norm[..., None], norm12[..., None],
                norm21[..., None], norm22[..., None]), axis=-1)
            rhhk = np.concatenate((rhhk[..., None], rhhk2[..., None]), axis=-1)
            ener = np.concatenate((ener[..., None], ener12[..., None],
                ener21[..., None], ener22[..., None]), axis=-1)
            xr = rhhk.ravel()
            xh = ener.reshape(xr.size // 2, xr.size // 2, 2, 2).transpose(0, 2, 1, 3)
            xh = xh.reshape(xr.size, xr.size)
            xn = norm.reshape(xr.size // 2, xr.size // 2, 2, 2).transpose(0, 2, 1, 3)
            xn = xn.reshape(xr.size, xr.size)
        else:
            exec(nr_eqs, globals(), { "norm": norm, **mdict })
            exec(pt2_eqs, globals(), { "rhhk": rhhk, "ener": ener, **mdict })
            xr = rhhk.ravel()
            xh = ener.reshape(xr.size, xr.size)
            xn = norm.reshape(xr.size, xr.size)
        xw, xu = np.linalg.eigh(xn)
        idx = xw > ic.trunc_thrds
        xf = xu[:, idx] * (xw[idx] ** (-0.5))
        xb = xu[:, idx] * (xw[idx] ** 0.5)
        th = xf.T @ xh @ xb
        tr = xf.T @ xr
        tx = _linear_solve(th, tr)
        xx = xb @ tx
        if skey not in ic.sub_eners:
            ic.sub_eners[skey] = -(xx * xr).sum()
            ic.sub_times[skey] = time.perf_counter() - t
        else:
            ic.sub_eners[skey] += -(xx * xr).sum()
            ic.sub_times[skey] += time.perf_counter() - t
        if key[-1] in "-1*":
            lib.logger.note(ic, "E(%s-%4s) = %20.14f",
                ic.__class__.__name__, skey, ic.sub_eners[skey])
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

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    # Example 1 - single state
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-12
    mc.run()
    wsc = WickICMRREPT2(mc).run()
    # converged SCF energy = -149.608181589162
    # CASSCF energy = -149.708657771235
    # CASCI E = -149.708657771235  E(CI) = -21.7431933131011  S^2 = 2.0000000
    # E(WickICMRREPT2-ijrs) =    -0.01740840148861
    # E(WickICMRREPT2- rsi) =    -0.04787731501533
    # E(WickICMRREPT2- ijr) =    -0.00422896694136
    # E(WickICMRREPT2-  rs) =    -0.10776433245143
    # E(WickICMRREPT2-  ij) =    -0.00204217954728
    # E(WickICMRREPT2-  ir) =    -0.06371598465781
    # E(WickICMRREPT2-   r) =    -0.04952457287085
    # E(WickICMRREPT2-   i) =    -0.00306754772523
    # E(WickICMRREPT2) = -150.0042870719332  E_corr_pt = -0.2956293006979092
    # Timings = ... total =  334.05

    # Example 2 - CASCI multi-state
    mc2 = mcscf.CASCI(mf, 6, 8)
    mc2.fcisolver.nroots = 3
    mc2.fcisolver.conv_tol = 1e-14
    mc2.canonicalization = True
    mc2.kernel(mc.mo_coeff)
    # [ -149.708657771235 -149.480534641702 -149.480534641702 ]

    wsc = WickICMRREPT2(mc2).run(root=0)
    wsc = WickICMRREPT2(mc2).run(root=1)
    wsc = WickICMRREPT2(mc2).run(root=2)
    # [ -0.2956293045124755 -0.2916934296640428 -0.291693429664039 ]

    # Example 3 - CASSCF state-average
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.state_average_([1 / 3] * 3)
    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.run()
    # [ -149.706807298862 -149.48431980376 -149.484319803759 ]
    wsc = WickICMRREPT2(mc).run(root=0)
    wsc = WickICMRREPT2(mc).run(root=1)
    wsc = WickICMRREPT2(mc).run(root=2)
    # E(WickICMRREPT2) = -150.0037522554753  E_corr_pt = -0.2969449566132465
    # E(WickICMRREPT2) = -149.771390135199  E_corr_pt = -0.2870703314391356
    # E(WickICMRREPT2) = -149.7713901309621  E_corr_pt = -0.2870703272025853
