
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
Strongly-Contracted NEVPT2 [J. Chem. Phys. 117, 9138 (2002)]
with equations derived on the fly.
need internal contraction module of block2.
"""

import numpy as np
import time

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr
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

P, PT, PD, DEF = init_parsers() # parsers
SP = lambda x: x.expand().remove_external().add_spin_free_trans_symm().simplify()
Comm = lambda h, k: SP(k.conjugate() * (h ^ k))
Norm = lambda k: SP(k.conjugate() * k)

DEF["gamma"] = PD("gamma[mn] = 1.0 \n - 0.5 delta[mn]")
h1 = P("SUM <ab> h[ab] E1[a,b]")
h2 = P("0.5 SUM <abcd> w[abcd] E2[ab,cd]")
hd = h1 + h2

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

# def of sc-nevpt2 sub-spaces
sub_spaces = {
    "ijrs": "gamma[ij] gamma[rs] w[rsij] E1[r,i] E1[s,j]\n"
            "gamma[ij] gamma[rs] w[rsji] E1[s,i] E1[r,j]",
    "rsi": "SUM <a> gamma[rs] w[rsia] E1[r,i] E1[s,a]\n"
           "SUM <a> gamma[rs] w[sria] E1[s,i] E1[r,a]",
    "ijr": "SUM <a> gamma[ij] w[raji] E1[r,j] E1[a,i]\n"
           "SUM <a> gamma[ij] w[raij] E1[r,i] E1[a,j]",
    "rs": "SUM <ab> gamma[rs] w[rsba] E1[r,b] E1[s,a]",
    "ij": "SUM <ab> gamma[ij] w[baij] E1[b,i] E1[a,j]",
    "ir": "SUM <ab> w[raib] E1[r,i] E1[a,b]\n"
          "SUM <ab> w[rabi] E1[a,i] E1[r,b]\n"
          "h[ri] E1[r,i]",
    "r": "SUM <abc> w[rabc] E1[r,b] E1[a,c]\n"
         "SUM <a> h[ra] E1[r,a]\n"
         "- SUM <ab> w[rbba] E1[r,a]",
    "i": "SUM <abc> w[baic] E1[b,i] E1[a,c]\n"
         "SUM <a> h[ai] E1[a,i]"
}

norm_eqs = {} # norm equations
ener_eqs = {} # Hamiltonian expectations
deno_fns = {} # denominators

for key, expr in sub_spaces.items():
    ket = P(expr)
    norm_eqs[key] = Norm(ket).to_einsum(PT("norm[%s]" % key))
    ener_eqs[key] = Comm(hd, ket).to_einsum(PT("ener[%s]" % key))
    denos = []
    for ix, wi in enumerate(PT("deno[%s]" % key).indices):
        sl = tuple([slice(None) if i == ix else None for i in range(len(key))])
        itx = 0 if (wi.types & WickIndexTypes.Inactive) != WickIndexTypes.Nothing else 1
        denos.append(lambda fii, fee, itx=itx, sl=sl: [-fii, fee][itx][sl])
    deno_fns[key] = lambda fii, fee, denos=denos: sum([dfn(fii, fee) for dfn in denos])


from pyscf import lib, mcscf

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
    wdict = { k: eris.get_phys(k[1:]) for k in wkeys }

    ic.sub_norms = {}
    ic.sub_eners = {}
    ic.sub_times = {'pdms': tpdms, 'eris': teris}
    for key in sub_spaces:
        t = time.perf_counter()
        if E4 is None and key in ["i", "r"]:
            irkmap = {'i': 'aaac', 'r': 'aaav'}
            ic.sub_eners[key] = dmrg_helper.dmrg_response_singles(mc, eris, E1,
                irkmap[key[0]], theory='nevpt2', root=root)
            lib.logger.note(ic, "E(%s-%4s) = %20.14f",
                ic.__class__.__name__.replace("SC", "UC"), 'mps' + key, ic.sub_eners[key])
            ic.sub_times[key] = time.perf_counter() - t
            continue
        deno = deno_fns[key](orbeI, orbeE)
        norm = np.zeros_like(deno)
        ener = np.zeros_like(deno)
        exec(norm_eqs[key] + ener_eqs[key], globals(), {
            "E1": E1, "E2": E2, "E3": E3, "E4": E4,
            "norm": norm, "ener": ener,
            "deltaII": np.eye(ncore), "deltaEE": np.eye(nvirt),
            "hAA": eris.h1eff[ncore:nocc, ncore:nocc],
            "hAI": eris.h1eff[ncore:nocc, :ncore],
            "hEI": eris.h1eff[nocc:, :ncore],
            "hEA": eris.h1eff[nocc:, ncore:nocc],
            **wdict
        })
        idx = (abs(norm) > 1E-14)
        grid = np.indices(deno.shape, dtype=np.int16)
        for sym_pair in ["ij", "rs"]:
            ix = key.find(sym_pair)
            if ix != -1:
                idx &= grid[ix] <= grid[ix + 1]
        ener[idx] = deno[idx] + ener[idx] / norm[idx]
        ic.sub_eners[key] = -(norm[idx] / ener[idx]).sum()
        ic.sub_norms[key] = norm[idx].sum()
        ic.sub_times[key] = time.perf_counter() - t
        lib.logger.note(ic, "E(%s-%4s) = %20.14f",
            ic.__class__.__name__, key, ic.sub_eners[key])
    ic.e_corr = sum(ic.sub_eners.values())
    lib.logger.note(ic, 'E(%s) = %.16g  E_corr_pt = %.16g',
        ic.__class__.__name__, ic.e_tot, ic.e_corr)
    lib.logger.note(ic, "Timings = %s",
        " | ".join(["%s = %7.2f" % (k, v) for k, v in ic.sub_times.items()]))

class WickSCNEVPT2(lib.StreamObject):
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

SCNEVPT2 = WickSCNEVPT2

if __name__ == "__main__":

    from pyscf import gto, scf, mcscf, mrpt

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    # Example 1 - single state
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.run()
    sc = mrpt.NEVPT(mc).run()
    wsc = WickSCNEVPT2(mc).run()
    # converged SCF energy = -149.608181589162
    # CASSCF energy = -149.708657770004
    # E(WickSCNEVPT2) = -149.9578381852564  E_corr_pt = -0.2491804151760018
    # E_corr(pyscf) = -0.249180419805814
    # ref -149.708657773372 (casscf) -0.2491831156 (sc) -149.95784088897202 (tot-sc)

    # Example 2 - CASCI multi-state
    mc2 = mcscf.CASCI(mf, 6, 8)
    mc2.fcisolver.nroots = 3
    mc2.fcisolver.conv_tol = 1e-14
    mc2.canonicalization = True
    mc2.kernel(mc.mo_coeff)
    # [ -149.708657770004 -149.480535614204 -149.480535614204 ]

    sc = mrpt.NEVPT(mc2, root=0).run()
    sc = mrpt.NEVPT(mc2, root=1).run()
    sc = mrpt.NEVPT(mc2, root=2).run()
    # [ -0.249180123136442 -0.245294722085918 -0.245294767244671 ]

    wsc = WickSCNEVPT2(mc2).run(root=0)
    wsc = WickSCNEVPT2(mc2).run(root=1)
    wsc = WickSCNEVPT2(mc2).run(root=2)
    # [ -0.2491801231364459 -0.2452947220859165 -0.2452947672446744 ]

    # Example 3 - CASSCF state-average
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.state_average_([1 / 3] * 3)
    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.run()
    # [ -149.706807303572 -149.484319801403 -149.484319801397 ]
    wsc = WickSCNEVPT2(mc).run(root=0)
    wsc = WickSCNEVPT2(mc).run(root=1)
    wsc = WickSCNEVPT2(mc).run(root=2)
    # E(WickSCNEVPT2) = -149.9566474118755  E_corr_pt = -0.2498401043435796
    # E(WickSCNEVPT2) = -149.7252997698935  E_corr_pt = -0.2409799703938773
    # E(WickSCNEVPT2) = -149.7252996696915  E_corr_pt = -0.2409798701919107

    # ref casscf = [ -149.706807583050 -149.484319661994 -149.484319661994 ]
    # ref 0 =  -0.2498431015 (SC) -0.2522554873 (PC) -149.956650684550 (TOT-SC)
    # ref 1 =  -0.2410187659 (SC) -0.2422012621 (PC) -149.725338427894 (TOT-SC)
    # ref 2 =  -0.2410187659 (SC) -0.2422012621 (PC) -149.725338427894 (TOT-SC)
