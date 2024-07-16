
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2024 Huanchen Zhai <hczhai@caltech.edu>
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
UGA-CCSDT with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")
import numpy as np

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.Inactive] = WickIndex.parse_set("pqrsijklmno")
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("pqrsabcdefg")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("V", 4)] = WickPermutation.qc_phys()
    perm_map[("T", 2)] = WickPermutation.non_symmetric()
    perm_map[("T", 4)] = WickPermutation.pair_symmetric(2, False)
    perm_map[("T", 6)] = WickPermutation.pair_symmetric(3, False)

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

h1 = P("SUM <pq> H[pq] E1[p,q]")
h2 = 0.5 * P("SUM <pqrs> V[pqrs] E2[pq,rs]")
t1 = P("SUM <ai> T[ia] E1[a,i]")
t2 = 0.5 * P("SUM <abij> T[ijab] E1[a,i] E1[b,j]")
t3 = (1.0 / 6.0) * P("SUM <abcijk> T[ijkabc] E1[a,i] E1[b,j] E1[c,k]")
ex1 = P("E1[i,a]")
ex2 = P("E1[i,a] E1[j,b]")
ex3 = P("E1[i,a] E1[j,b] E1[k,c]")

hf = PX("H[pq] = F[pq] \n - 2.0 SUM <j> V[pjqj] \n + SUM <j> V[pjjq]")
ehf = P("2 SUM <i> H[ii] \n + 2 SUM <ij> V[ijij] \n - SUM <ij> V[ijji]")

HBarTermsSD = lambda h, t: [
    h, h ^ t,
    0.5 * ((h ^ t1) ^ t1) + ((h ^ t2) ^ t1) + 0.5 * ((h ^ t2) ^ t2),
    (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1) + 0.5 * (((h ^ t2) ^ t1) ^ t1),
    (1 / 24.0) * ((((h ^ t1) ^ t1) ^ t1) ^ t1)
]

HBarTermsSDT = lambda h, t: [
    h, h ^ t,
    0.5 * ((h ^ t) ^ t),
    (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1) + 0.5 * (((h ^ t2) ^ t1) ^ t1)
        + 0.5 * (((h ^ t2) ^ t2) ^ t1) + 0.5 * (((h ^ t3) ^ t1) ^ t1),
    (1 / 24.0) * ((((h ^ t1) ^ t1) ^ t1) ^ t1) + (1 / 6.0) * ((((h ^ t2) ^ t1) ^ t1) ^ t1)
]

def purify(eq, outt):
    pure_terms = []
    for term in eq.terms:
        nt, mp = len(term.tensors) + 1, {}
        mi = max([len(wt.indices) // 2 for wt in list(term.tensors) + [outt]])
        for it, wt in enumerate(list(term.tensors) + [outt]):
            for ii, wi in enumerate(wt.indices):
                mp[wi] = mp.get(wi, []) + [it * mi + ii % (len(wt.indices) // 2)]
        parex = list(range(nt * mi))
        cnt = [0] * nt * mi
        def findx(x):
            if parex[x] != x:
                parex[x] = findx(parex[x])
            return parex[x]
        def unionx(x, y):
            x, y = findx(x), findx(y)
            if x != y:
                parex[x] = y
        for zc, zd in mp.values():
            cnt[zc], cnt[zd] = 1, 1
            unionx(zc, zd)
        cnt1 = len(set([findx((nt - 1) * mi + i) for i in range(mi) if cnt[(nt - 1) * mi + i]]))
        cnt2 = sum([1 for i in range(mi) if cnt[(nt - 1) * mi + i]])
        if cnt1 == cnt2:
            pure_terms.append(term * (1 / 2 ** (len(outt.indices) // 2)))
    return eq.__class__(eq.terms.__class__(pure_terms))

def to_tensor_eqs(eqs, outt):
    xstr = eqs.to_einsum(outt).split('\n')[:-1]
    tensor_eqs = []
    for xx in xstr:
        zx = xx.replace('+= np.', '+= 1 * np.').split(' += ')[1].split(' * np.einsum(')
        f = float(zx[0])
        zz = zx[1].split(', ')
        script = zz[0][1:-1]
        nmls = [len(x) for x in script.split('->')[0].split(',')]
        nms = [x[:-l] for x, l in zip(zz[1:-1], nmls)]
        idxs = [x[-l:] for x, l in zip(zz[1:-1], nmls)]
        tensor_eqs.append((f, script, idxs, nms))
    return tensor_eqs

def get_cc_eqs(t_order, normal_ord=True):
    t = [SP(t1 + t2), SP(t1 + t2 + t3)][t_order - 2]
    h = [SP(h1 + h2), SP(h1 + h2 - ehf).substitute(hf)][normal_ord]
    HBarTerms = [HBarTermsSD, HBarTermsSDT][t_order - 2]

    en_eq = FC(sum(HBarTerms(h, t)[:3], Z))
    t1_eq = purify(FC(ex1 * sum(HBarTerms(h, t)[:4], Z)), PT("X[ia]"))
    t2_eq = purify(FC(ex2 * sum(HBarTerms(h, t)[:5], Z)), PT("X[ijab]"))
    cc_tensor_eqs = [
        to_tensor_eqs(en_eq, PT("X[]")), to_tensor_eqs(t1_eq, PT("X[ia]")),
        to_tensor_eqs(t2_eq, PT("X[ijab]")),
    ]
    if t_order == 3:
        t3_eq = purify(FC(ex3 * sum(HBarTerms(h, t)[:5], Z)), PT("X[ijkabc]"))
        cc_tensor_eqs += [to_tensor_eqs(t3_eq, PT("X[ijkabc]"))]
    return cc_tensor_eqs

def rt_normal_order(n_occ, ints, ecore, phys_g2e=True):
    import numpy as np
    h1e, g2e = ints
    if phys_g2e:
        g2e = g2e.transpose(0, 2, 1, 3)
    ecore += 2.0 * np.einsum('ii->', h1e[:n_occ, :n_occ], optimize='optimal')
    ecore += 2.0 * np.einsum('iijj->', g2e[:n_occ, :n_occ, :n_occ, :n_occ], optimize='optimal')
    ecore -= 1.0 * np.einsum('ijji->', g2e[:n_occ, :n_occ, :n_occ, :n_occ], optimize='optimal')
    h1e = h1e.copy()
    h1e += 2.0 * np.einsum('pqjj->pq', g2e[:, :, :n_occ, :n_occ], optimize='optimal')
    h1e -= 1.0 * np.einsum('pjjq->pq', g2e[:, :n_occ, :n_occ, :], optimize='optimal')
    if phys_g2e:
        g2e = g2e.transpose(0, 2, 1, 3)
    return (h1e, g2e), ecore

def rt_ao2mo(mf, nfrozen=0, normal_ord=True, ncas=None, dtype=float, ecore_target=None):
    from pyscf import ao2mo
    import numpy as np

    mol = mf.mol
    ncore = nfrozen
    ncas = mf.mo_coeff.shape[1] - ncore if ncas is None else ncas

    n_occ = (mol.nelectron - mol.spin) // 2 - ncore
    n_virt = ncas - n_occ

    mo = mf.mo_coeff
    mo_core = mo[:, :ncore]
    mo_cas = mo[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = 0

    if ncore != 0:
        core_dmao = 2 * mo_core @ mo_core.T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj - 0.5 * vk
        ecore0 = np.einsum("ij,ji->", core_dmao, hcore_ao + 0.5 * hveff_ao, optimize='optimal')
    else:
        ecore0 = 0.0

    h1e = mo_cas.T.conj() @ (hcore_ao + hveff_ao) @ mo_cas
    eri_ao = mol if mf._eri is None else mf._eri
    g2e = ao2mo.restore(1, ao2mo.full(eri_ao, mo_cas), ncas)

    h1e = np.asarray(h1e, dtype=dtype)
    g2e = np.asarray(g2e, dtype=dtype)

    enuc = mol.energy_nuc() + ecore0
    ecore = enuc

    if normal_ord:
        (h1e, g2e), ecore = rt_normal_order(n_occ, (h1e, g2e), ecore, phys_g2e=False)

    g2e = g2e.transpose(0, 2, 1, 3)

    if ecore_target is not None:
        assert not normal_ord
        h1e[np.mgrid[:ncas], np.mgrid[:ncas]] -= ((ecore_target - ecore) / (n_occ * 2))
        ecore = ecore_target

    return ecore, (h1e, g2e), n_occ, n_virt

def rcc_init_amps(cc):
    import numpy as np
    from pyscf.lib import logger
    from pyscf import lib

    mo_e = cc.mo_energy
    nocc = cc.n_occ
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    eijab = lib.direct_sum("ia,jb->ijab", eia, eia)
    t1 = cc.ints[0][:nocc, nocc:] / eia
    eris_oovv = np.array(cc.ints[1][:nocc, :nocc, nocc:, nocc:])
    t2 = eris_oovv / eijab
    cc.emp2 = 2.0 * np.einsum("ijab,ijab", t2, eris_oovv.conj(), optimize='optimal').real
    cc.emp2 -= np.einsum("ijab,ijab", t2, eris_oovv.conj(), optimize='optimal').real
    logger.info(cc, "Init t2, MP2 energy = %.15g", cc.emp2)
    tamps = [t1, t2]
    for it in range(3, cc.order + 1):
        tamps.append(np.zeros((*([t1.shape[0]] * it), *([t1.shape[1]] * it)), dtype=t1.dtype))
    return tamps[:cc.order]

def cc_run_diis(cc, tamps, istep, normt, de, adiis):
    from pyscf.lib import logger
    if adiis and istep >= cc.diis_start_cycle:
        vec = cc.amplitudes_to_vector(tamps)
        tamps = cc.vector_to_amplitudes(adiis.update(vec))
        logger.debug1(cc, "DIIS for step %d", istep)
    return tamps

def cc_kernel(cc, tamps=None, max_cycle=50, tol=1e-8, tolnormt=1e-6):
    from pyscf.lib import logger
    from pyscf import lib
    import numpy as np

    log = logger.new_logger(cc, cc.verbose)
    if tamps is None:
        tamps = cc.get_init_guess()

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    ecc = cc.energy(tamps)
    log.info("Init E_corr(%s) = %.15g", cc.name, ecc)

    if cc.diis:
        adiis = lib.diis.DIIS(cc, None, incore=True)
        adiis.space = cc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        tamps_new = cc.update_amps(tamps)
        tmpvec = cc.amplitudes_to_vector(tamps_new)
        tmpvec -= cc.amplitudes_to_vector(tamps)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if cc.iterative_damping < 1.0:
            alpha = cc.iterative_damping
            for tx, tx_new in zip(tamps, tamps_new):
                tx_new *= alpha
                tx_new += (1 - alpha) * tx
        tamps = tamps_new
        tamps_new = None
        tamps = cc.run_diis(tamps, istep, normt, ecc - eold, adiis)
        eold, ecc = ecc, cc.energy(tamps)
        log.info("cycle =%3d  E_corr(%s) = %22.15f  dE = %10.3e  norm(t amps) = %10.3e",
            istep + 1, cc.name, ecc, ecc - eold, normt)
        cput1 = log.timer("%s iter" % cc.name, *cput1)
        if abs(ecc - eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer(cc.name, *cput0)
    cc.converged, cc.e_corr, cc.tamps = conv, ecc, tamps
    logger.info(cc, "%s %sconverged", cc.name, "" if cc.converged else "not ")
    logger.note(cc, "E(%s) = %.16g  E_corr = %.16g", cc.name, cc.e_tot, cc.e_corr)
    return cc.e_tot

def gcc_energy(cc, tamps):
    import numpy as np
    ints, n_occ, eqs = cc.ints, cc.n_occ, cc.eqs
    sli = {'I': slice(n_occ), 'E': slice(n_occ, None)}
    ener = np.array(0.0, dtype=ints[0].dtype)
    for f, script, idxs, nm in eqs[0]:
        tensors = []
        for nmx, idx in zip(nm, idxs):
            tensors.append(tamps[len(idx) // 2 - 1] if nmx == 'T' else ints[len(idx) // 2 - 1][tuple(sli[x] for x in idx)])
        ener += f * np.einsum(script, *tensors, optimize='optimal')
    return ener

def gcc_update_amps(cc, tamps):
    import numpy as np
    ints, n_occ, eqs = cc.ints, cc.n_occ, cc.eqs
    sli = {'I': slice(n_occ), 'E': slice(n_occ, None)}
    ts_new = [np.zeros_like(t, dtype=ints[0].dtype) for t in tamps]
    fii, faa = np.diag(ints[0])[:n_occ], np.diag(ints[0])[n_occ:]
    if not cc.normal_ord:
        fii = fii + np.einsum("mjmj->m", ints[1][:n_occ, :n_occ, :n_occ, :n_occ], optimize='optimal')
        faa = faa + np.einsum("mjmj->m", ints[1][n_occ:, :n_occ, n_occ:, :n_occ], optimize='optimal')
    eia, exx = fii[:, None] - faa[None, :], np.array(0)
    for kk in range(0, cc.order):
        for f, script, idxs, nm in eqs[kk + 1]:
            tensors = []
            for nmx, idx in zip(nm, idxs):
                tensors.append(tamps[len(idx) // 2 - 1] if nmx == 'T' else ints[len(idx) // 2 - 1][tuple(sli[x] for x in idx)])
            ts_new[kk] += f * np.einsum(script, *tensors, optimize='optimal')
        xi, xa, xt = 'ijklmnop'[:kk + 1], 'abcdefgh'[:kk + 1], tamps[kk]
        for ii, aa in zip(xi, xa):
            ts_new[kk] += np.einsum('%s,%s%s->%s%s' % (ii, xi, xa, xi, xa), fii, xt, optimize='optimal')
            ts_new[kk] -= np.einsum('%s,%s%s->%s%s' % (aa, xi, xa, xi, xa), faa, xt, optimize='optimal')
        exx = exx[((slice(None), ) * kk + (None, )) * 2] + eia[((None, ) * kk + (slice(None), )) * 2]
        ts_new[kk] /= exx
    return ts_new

def gcc_amplitudes_to_vector(cc, tamps, out=None):
    import numpy as np
    nov = cc.n_occ * cc.n_virt
    size = 0
    for it, _ in enumerate(tamps):
        size += nov ** (it + 1)
    vector = np.ndarray(size, tamps[0].dtype, buffer=out)
    size = 0
    for it, t in enumerate(tamps):
        vector[size : size + nov ** (it + 1)] = t.ravel()
        size += nov ** (it + 1)
    return vector

def gcc_vector_to_amplitudes(cc, vector):
    nocc, nvir, nov = cc.n_occ, cc.n_virt, cc.n_occ * cc.n_virt
    z, it, tamps = 0, 0, []
    while z < vector.size:
        t = vector[z:z + nov ** (it + 1)].reshape((*([nocc] * (it + 1)), *([nvir] * (it + 1))))
        tamps.append(t)
        z += nov ** (it + 1)
        it += 1
    return tamps

class RCC:
    def __init__(self, mf, t_order=3, nfrozen=0, normal_ord=True, ecore_target=None, verbose=None, diis=True, dtype=float):
        import sys
        self.order = t_order
        self.normal_ord = normal_ord
        self.eqs = get_cc_eqs(t_order, normal_ord=normal_ord)
        self.e_hf, self.ints, self.n_occ, self.n_virt = rt_ao2mo(mf,
            nfrozen=nfrozen, normal_ord=normal_ord, dtype=dtype, ecore_target=ecore_target)
        self.mo_energy = mf.mo_energy
        self.stdout = sys.stdout
        self.verbose = mf.verbose if verbose is None else verbose
        self.diis = diis
        self.diis_space = 6
        self.diis_start_cycle = 0
        self.iterative_damping = 1.0
        self.name = "RCC%s" % "SDTQPH789"[:self.order]
        self.converged = False
    energy = gcc_energy
    update_amps = lambda cc, tamps: gcc_update_amps(cc, tamps)
    init_amps = rcc_init_amps
    amplitudes_to_vector = gcc_amplitudes_to_vector
    vector_to_amplitudes = gcc_vector_to_amplitudes
    get_init_guess = rcc_init_amps
    run_diis = cc_run_diis
    kernel = cc_kernel
    e_tot = property(lambda self: self.e_hf + self.e_corr)

def RCCSD(*args, **kwargs):
    kwargs["t_order"] = 2
    return RCC(*args, **kwargs)

def RCCSDT(*args, **kwargs):
    kwargs["t_order"] = 3
    return RCC(*args, **kwargs)

if __name__ == "__main__":

    from pyscf import gto, scf, fci
    import numpy as np

    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g', verbose=3)
    mf = scf.RHF(mol).run(conv_tol=1E-14)

    mc = fci.FCI(mf)
    e_fci = mc.kernel()[0]
    print('E[FCI] = %18.12f' % e_fci)

    assert abs(mf.e_tot - -74.9611711378677) < 1E-10
    assert abs(e_fci - -75.019275996606) < 1E-10

    mcc = RCCSD(mf, verbose=4, diis=False)
    mcc.kernel()
    assert abs(mcc.e_tot - -75.01913630605036) < 1E-6

    mcc = RCCSDT(mf, verbose=4, diis=False)
    mcc.kernel()
    assert abs(mcc.e_tot - -75.01922418629152) < 1E-6
