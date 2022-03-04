
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
Fully Internally-Contracted MRCISD [J. Chem. Phys. 145, 054104 (2016)]
with equations derived on the fly (will take ~ 1 min).
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

import numpy as np

try:
    from . import eri_helper
except ImportError:
    import eri_helper

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.Inactive] = WickIndex.parse_set("mnxyijkl")
    idx_map[WickIndexTypes.Active] = WickIndex.parse_set("mnxyabcdefghpq")
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("mnxyrstu")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("w", 4)] = WickPermutation.qc_phys()

    p = lambda x: WickExpr.parse(x, idx_map, perm_map)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    pd = lambda x: WickExpr.parse_def(x, idx_map, perm_map)

    return p, pt, pd

P, PT, PD = init_parsers() # parsers
SP = lambda x: x.expand().add_spin_free_trans_symm().remove_external().remove_inactive().simplify()
Comm = lambda b, h, k: SP(b.conjugate() * (h ^ k))
Expt = lambda b, h, k: SP(b.conjugate() * (h * k))
Norm = lambda b, k: SP(b.conjugate() * k)
Z = P("")

h1 = P("SUM <mn> h[mn] E1[m,n]")
h2 = P("0.5 SUM <mnxy> w[mnxy] E2[mn,xy]")
h = h1 + h2

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

# def of fic-mrcisd sub-spaces
sub_spaces = {
    "ref-ref-*": "1.0",
    "ijrskltu*": "E1[r,i] E1[s,j]",
    "rsiatukp*": "E1[r,i] E1[s,a]",
    "ijrakltp*": "E1[r,j] E1[a,i]",
    "rsabtupq*": "E1[r,b] E1[s,a]",
    "ijabklpq*": "E1[b,i] E1[a,j]",
    "irabktpq1": "E1[r,i] E1[a,b]",
    "irabktpq2": "E1[a,i] E1[r,b]",
    "rabctpqg*": "E1[r,b] E1[a,c]",
    "iabckpqg*": "E1[b,i] E1[a,c]"
}

ener_eqs = {} # Hamiltonian expectations
norm_eqs = {} # Overlap equations

for bkey, bexpr in sub_spaces.items():
    bra = P(bexpr)
    ket_bra_map = { k: v for k, v in zip(bkey[:4], bkey[4:8]) }
    bra = bra.index_map(MapStrStr(ket_bra_map))
    for kkey, kexpr in sub_spaces.items():
        ket = P(kexpr)
        if bkey[:-1] == kkey[:-1]:
            ener_eqs[(bkey, kkey)] = Comm(bra, h, ket)
        else:
            ener_eqs[(bkey, kkey)] = Expt(bra, h, ket)
        norm_eqs[(bkey, kkey)] = Norm(bra, ket)
        if bkey != kkey and not bkey[-1] in '12' and not kkey[-1] in '12':
            assert norm_eqs[(bkey, kkey)] == Z

allowed_perms = {'EEEE', 'AIEE', 'EAAI', 'EIEE', 'IIEE', 'EAEA', 'IAEA',
                 'AAEA', 'EIEI', 'AAII', 'AEEA', 'AIEI', 'EAEE', 'IAEI',
                 'AIAI', 'AAAI', 'AAAA', 'IIII', 'EIEA', 'AIII', 'IIEI'}

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

for eq in [*ener_eqs.values(), *norm_eqs.values()]:
    fix_eri_permutations(eq)

def _key_idx(key):
    t = [WickIndexTypes.Inactive, WickIndexTypes.Active,
         WickIndexTypes.External, WickIndexTypes.Nothing]
    return [t.index(wi.types) for wi in PT("x[%s]" % key).indices]

from pyscf import lib

def kernel(ic, mc=None, mo_coeff=None, pdms=None, eris=None, nroots=1):
    if mc is None:
        mc = ic._mc
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    ic.mo_coeff = mo_coeff
    ic.ci = mc.ci
    ic.mo_energy = mc.mo_energy
    if pdms is None:
        pdms = eri_helper.init_pdms(mc=mc, pdm_eqs=pdm_eqs)
    if eris is None:
        eris = eri_helper.init_eris(mc=mc, mo_coeff=mo_coeff, mrci=True)
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
        **{ "ident%d" % d: np.ones((1, ) * d) for d in [1, 2, 3] },
        **{ 'h' + a + b: eris.get_h1(a + b) for a in 'IAE' for b in 'IAE' },
        **{ 'w' + k: eris.get_phys(k) for k in allowed_perms }
    }
    umats = {} # ortho matrices
    ntr = 0
    for key in [k for k in sub_spaces if k[-1] != '2']:
        nkey = key[:-1] if key[:3] != 'ref' else '??'
        norm = np.zeros([[ncore, ncas, nvirt, 1][ix] for ix in _key_idx(nkey)])
        b_eqs = norm_eqs[(key, key)].to_einsum(PT("norm[%s]" % nkey))
        dtot = np.prod([[ncore, ncas, nvirt, 1][ix] for ix in _key_idx(nkey[:4])], dtype=int)
        if key[-1] == '1':
            key2 = key[:-1] + '2'
            norm12 = np.zeros_like(norm)
            norm21 = np.zeros_like(norm)
            norm22 = np.zeros_like(norm)
            b_eqs += norm_eqs[(key, key2)].to_einsum(PT("norm12[%s]" % nkey))
            b_eqs += norm_eqs[(key2, key)].to_einsum(PT("norm21[%s]" % nkey))
            b_eqs += norm_eqs[(key2, key2)].to_einsum(PT("norm22[%s]" % nkey))
            exec(b_eqs, globals(), { "norm": norm, "norm12": norm12,
                "norm21": norm21, "norm22": norm22, **mdict })
            norm = np.concatenate((norm[..., None], norm12[..., None],
                norm21[..., None], norm22[..., None]), axis=-1)
            xn = norm.reshape(dtot, dtot, 2, 2).transpose(0, 2, 1, 3)
            xn = xn.reshape(dtot * 2, dtot * 2)
        else:
            exec(b_eqs, globals(), { "norm": norm, **mdict })
            xn = norm.reshape(dtot, dtot)
        lib.logger.debug(ic, 'diag overlap %s size = %d', key, len(xn))
        w, v = np.linalg.eigh(xn)
        idx = w > ic.mrci_thrds
        umats[key] = v[:, idx] * (w[idx] ** (-0.5))
        ntr += umats[key].shape[1]
    lib.logger.info(ic, 'HMAT basis size = %d thrds = %g', ntr, ic.mrci_thrds)
    pre_hmats = {}
    for bkey in sub_spaces:
        bk = bkey[4:8] if bkey[:3] != 'ref' else ''
        for kkey in sub_spaces:
            lib.logger.debug(ic, 'hamil eq (%s, %s)', bkey, kkey)
            kk = kkey[:4] if kkey[:3] != 'ref' else ''
            nkey = bk + kk
            ener = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(nkey)])
            a_eqs = ener_eqs[(bkey, kkey)].to_einsum(PT("ener[%s]" % nkey))
            dbra = np.prod([[ncore, ncas, nvirt][ix] for ix in _key_idx(bk)], dtype=int)
            dket = np.prod([[ncore, ncas, nvirt][ix] for ix in _key_idx(kk)], dtype=int)
            exec(a_eqs, globals(), { "ener": ener, **mdict })
            pre_hmats[(bkey, kkey)] = ener.reshape(dbra, dket)
    for bkey in [k for k in sub_spaces if k[-1] == '1']:
        for kkey in sub_spaces:
            m1 = pre_hmats[(bkey, kkey)]
            m2 = pre_hmats[(bkey[:-1] + "2", kkey)]
            m12 = np.concatenate((m1[..., None], m2[..., None]), axis=-1).transpose(0, 2, 1)
            pre_hmats[(bkey, kkey)] = m12.reshape(m1.shape[0] * 2, m1.shape[1])
    for kkey in [k for k in sub_spaces if k[-1] == '1']:
        for bkey in [k for k in sub_spaces if k[-1] != '2']:
            m1 = pre_hmats[(bkey, kkey)]
            m2 = pre_hmats[(bkey, kkey[:-1] + "2")]
            m12 = np.concatenate((m1[..., None], m2[..., None]), axis=-1)
            pre_hmats[(bkey, kkey)] = m12.reshape(m1.shape[0], m1.shape[1] * 2)
    keys = [k for k in sub_spaces if k[-1] != '2' and k[:3] != 'ref']
    keys = [k for k in sub_spaces if k[:3] == 'ref'] + keys
    lib.logger.debug(ic, 'keys = %r', keys)
    hmat = np.zeros((ntr, ntr))
    ib = 0
    for bkey in keys:
        ik = 0
        for kkey in keys:
            hx = pre_hmats[(bkey, kkey)]
            lib.logger.debug(ic, 'pre mat (%s, %s) %7d%7d symm error = %15.10f',
                bkey, kkey, ib, ik, np.linalg.norm(hx - pre_hmats[(kkey, bkey)].T))
            htr = np.einsum("ij,ia,jb->ab", hx, umats[bkey], umats[kkey], optimize=True)
            hmat[ib:ib + htr.shape[0], ik:ik + htr.shape[1]] = htr
            ik += umats[kkey].shape[1]
        ib += umats[bkey].shape[1]
    lib.logger.info(ic, 'HMAT symm error = %15.10f', np.linalg.norm(hmat - hmat.T))
    hmat = (hmat + hmat.T) / 2
    w, v = np.linalg.eigh(hmat)
    ic.e_states = w[:nroots] + ic._mc.e_tot
    ic.ci = v[:, :nroots]
    ic.e_corr = ic.e_states[0] - ic._mc.e_tot
    ic.de_dav_q = ic.e_corr * (1 - v[0, 0] ** 2) / v[0, 0] ** 2
    lib.logger.note(ic, 'E(MRCI) - E(ref) = %.16g DC = %.16g', ic.e_corr, ic.de_dav_q)
    lib.logger.note(ic, 'E(%s)   = %.16g  E_corr_ci = %.16g',
        ic.__class__.__name__, ic.e_tot, ic.e_corr)
    lib.logger.note(ic, 'E(%s+Q) = %.16g  E_corr_ci = %.16g',
        ic.__class__.__name__, ic.e_tot + ic.de_dav_q, ic.e_corr + ic.de_dav_q)

class WickICMRCISD(lib.StreamObject):
    def __init__(self, mc):
        self._mc = mc
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.e_corr = None
        self.mrci_thrds = 1E-10

    @property
    def e_tot(self):
        return np.asarray(self.e_corr) + self._mc.e_tot

    kernel = kernel

ICMRCISD = WickICMRCISD

if __name__ == "__main__":

    from pyscf import gto, scf, mcscf

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2)
    mf = scf.RHF(mol).run(conv_tol=1E-20)
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.run()
    mol.verbose = 5
    wmrci = WickICMRCISD(mc).run()
    # converged SCF energy = -149.608181589162
    # CASSCF energy = -149.708657770062
    # HMAT symm error =    0.0026672717
    # E(MRCI) - E(ref) = -0.2643537344241054 DC = -0.01633618724661364
    # E(WickICMRCISD)   = -149.9730115044681  E_corr_ci = -0.2643537344241054
    # E(WickICMRCISD+Q) = -149.9893476917148  E_corr_ci = -0.280689921670719
