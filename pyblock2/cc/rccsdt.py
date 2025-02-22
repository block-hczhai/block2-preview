
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2025 Huanchen Zhai <hczhai.ok@gmail.com>
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
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation, VectorWickTensor
    from block2 import MapStrPWickTensorExpr, VectorWickIndex, WickTensorTypes
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
    perm_map[("L", 2)] = WickPermutation.non_symmetric()
    perm_map[("L", 4)] = WickPermutation.pair_symmetric(2, False)
    perm_map[("L", 6)] = WickPermutation.pair_symmetric(3, False)
    perm_map[("R", 2)] = WickPermutation.non_symmetric()
    perm_map[("R", 4)] = WickPermutation.pair_symmetric(2, False)
    perm_map[("R", 6)] = WickPermutation.pair_symmetric(3, False)

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
l1 = P("SUM <ai> L[ia] E1[i,a]")
l2 = 0.5 * P("SUM <abij> L[ijab] E1[i,a] E1[j,b]")
l3 = (1.0 / 6.0) * P("SUM <abcijk> L[ijkabc] E1[i,a] E1[j,b] E1[k,c]")
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

def make_diag(eq, name, outt):
    diag_terms = []
    for term in eq.terms:
        deltas, tensors = [], []
        if any(wt.name == name and len(wt.indices) != len(outt.indices) for wt in term.tensors):
            continue
        for wt in term.tensors:
            if wt.name == name:
                for wi, oi in zip(wt.indices, outt.indices):
                    deltas.append(WickTensor("delta", VectorWickIndex([wi, oi]), WickPermutation.two_symmetric(),
                          WickTensorTypes.KroneckerDelta))
            else:
                tensors.append(wt)
        diag_terms.append(term.__class__(VectorWickTensor(tensors + deltas), term.ctr_indices, term.factor))
    return eq.__class__(eq.terms.__class__(diag_terms))

def purify(eq, outt):
    pure_terms = []
    for term in eq.terms:
        nt, mp = len(term.tensors) + 1, {}
        mi = max([(len(wt.indices) + 1) // 2 for wt in list(term.tensors) + [outt]])
        for it, wt in enumerate(list(term.tensors) + [outt]):
            for ii, wi in enumerate(wt.indices):
                x = (ii - len(wt.indices) % 2) % max(1, len(wt.indices) // 2)
                mp[wi] = mp.get(wi, []) + [it * mi + (x if len(wt.indices) % 2 == 0 else
                    (-it * mi + (nt - 1) * mi if ii == 0 else x + 1))]
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
            pure_terms.append(term * (1 / 2 ** ((len(outt.indices) + 1) // 2)))
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
        nms = ['=' if x.startswith('ident') else x[:-l] for x, l in zip(zz[1:-1], nmls)]
        idxs = ['?' * l if x.startswith('ident') else x[-l:] for x, l in zip(zz[1:-1], nmls)]
        tensor_eqs.append((f, script, idxs, nms))
    return tensor_eqs

def get_cc_eqs(t_order, normal_ord=True):
    t = [SP(t1 + t2), SP(t1 + t2 + t3)][t_order - 2]
    h = [SP(h1 + h2), SP(h1 + h2 - ehf).substitute(hf)][normal_ord]
    HBarTerms = [HBarTermsSD, HBarTermsSDT][t_order - 2]

    en_eq = FC(sum(HBarTerms(h, t)[:3], Z))
    t1_eq = purify(FC(ex1 * sum(HBarTerms(h, t)[:4], Z)), PT("X[ia]"))
    t2_eq = purify(FC(ex2 * sum(HBarTerms(h, t)[:5], Z)), PT("X[ijab]"))
    tensor_eqs = [
        to_tensor_eqs(en_eq, PT("X[]")), to_tensor_eqs(t1_eq, PT("X[ia]")),
        to_tensor_eqs(t2_eq, PT("X[ijab]")),
    ]
    if t_order == 3:
        t3_eq = purify(FC(ex3 * sum(HBarTerms(h, t)[:5], Z)), PT("X[ijkabc]"))
        tensor_eqs += [to_tensor_eqs(t3_eq, PT("X[ijkabc]"))]
    return tensor_eqs

def get_cc_lambda_eqs(t_order, normal_ord=True):
    I = P("1")
    t = [SP(t1 + t2), SP(t1 + t2 + t3)][t_order - 2]
    l = [SP(l1 + l2), SP(l1 + l2 + l3)][t_order - 2]
    h = [SP(h1 + h2), SP(h1 + h2 - ehf).substitute(hf)][normal_ord]
    HBarTerms = [HBarTermsSD, HBarTermsSDT][t_order - 2]
    hbar = sum(HBarTerms(h, t)[:5], Z)

    en_eq = FC(sum(HBarTerms(h, t)[:3], Z))
    l1_eq = purify(FC((I + l) * (hbar - en_eq) * ex1.conjugate()), PT("X[ia]"))
    l2_eq = purify(FC((I + l) * (hbar - en_eq) * ex2.conjugate()), PT("X[ijab]"))
    tensor_eqs = [to_tensor_eqs(l1_eq, PT("X[ia]")), to_tensor_eqs(l2_eq, PT("X[ijab]"))]
    if t_order == 3:
        l3_eq = purify(FC((I + l) * (hbar - en_eq) * ex3.conjugate()), PT("X[ijkabc]"))
        tensor_eqs += [to_tensor_eqs(l3_eq, PT("X[ijkabc]"))]
    return tensor_eqs

def get_eomcc_eqs(t_order, eom_t, normal_ord=True):
    if eom_t == 'ee':
        r1 = P("SUM <ai> R[ia] E1[a,i]")
        r2 = 0.5 * P("SUM <abij> R[ijab] E1[a,i] E1[b,j]")
        r3 = (1.0 / 6.0) * P("SUM <abcijk> R[ijkabc] E1[a,i] E1[b,j] E1[c,k]")
        pt1, pt2, pt3 = PT("X[ia]"), PT("X[ijab]"), PT("X[ijkabc]")
        ex1, ex2, ex3 = P("E1[i,a]"), P("E1[i,a] E1[j,b]"), P("E1[i,a] E1[j,b] E1[k,c]")
    elif eom_t == 'ip':
        r1 = P("SUM <i> R[i] D1[i]")
        r2 = P("SUM <bij> R[ijb] E1[b,j] D1[i]")
        r3 = P("SUM <bcijk> R[ijkbc] E1[b,j] E1[c,k] D1[i]")
        pt1, pt2, pt3 = PT("X[i]"), PT("X[ijb]"), PT("X[ijkbc]")
        ex1, ex2, ex3 = P("C1[i]"), P("C1[i] E1[j,b]"), P("C1[i] E1[j,b] E1[k,c]")
    elif eom_t == 'ea':
        r1 = P("SUM <a> R[a] C1[a]")
        r2 = P("SUM <abj> R[abj] C1[a] E1[b,j]")
        r3 = P("SUM <abcjk> R[abcjk] C1[a] E1[b,j] E1[c,k]")
        pt1, pt2, pt3 = PT("X[a]"), PT("X[abj]"), PT("X[abcjk]")
        ex1, ex2, ex3 = P("D1[a]"), P("E1[j,b] D1[a]"), P("E1[j,b] E1[k,c] D1[a]")
    t = [SP(t1 + t2), SP(t1 + t2 + t3)][t_order - 2]
    r = [SP(r1 + r2), SP(r1 + r2 + r3)][t_order - 2]
    h = [SP(h1 + h2), SP(h1 + h2 - ehf).substitute(hf)][normal_ord]
    HBarTerms = [HBarTermsSD, HBarTermsSDT][t_order - 2]
    hbar = sum(HBarTerms(h, t)[:5], Z)
    r1_eq = purify(FC(ex1 * (hbar ^ r)), pt1)
    r2_eq = purify(FC(ex2 * (hbar ^ r)), pt2)
    r1_left_eq = purify(FC(r.conjugate() * (hbar ^ ex1.conjugate())), pt1)
    r2_left_eq = purify(FC(r.conjugate() * (hbar ^ ex2.conjugate())), pt2)
    r1_diag_eq = SP(make_diag(r1_eq, 'R', pt1))
    r2_diag_eq = SP(make_diag(r2_eq, 'R', pt2))
    r1_left_diag_eq = SP(make_diag(r1_left_eq, 'R', pt1))
    r2_left_diag_eq = SP(make_diag(r2_left_eq, 'R', pt2))
    tensor_eqs = [
        [to_tensor_eqs(r1_eq, pt1), to_tensor_eqs(r2_eq, pt2)],
        [to_tensor_eqs(r1_left_eq, pt1), to_tensor_eqs(r2_left_eq, pt2)],
        [to_tensor_eqs(r1_diag_eq, pt1), to_tensor_eqs(r2_diag_eq, pt2)],
        [to_tensor_eqs(r1_left_diag_eq, pt1), to_tensor_eqs(r2_left_diag_eq, pt2)],
    ]
    if t_order == 3:
        r3_eq = purify(FC(ex3 * (hbar ^ r)), pt3)
        r3_left_eq = purify(FC(r.conjugate() * (hbar ^ ex3.conjugate())), pt3)
        r3_diag_eq = SP(make_diag(r3_eq, 'R', pt3))
        r3_left_diag_eq = SP(make_diag(r3_left_eq, 'R', pt3))
        tensor_eqs[0] += [to_tensor_eqs(r3_eq, pt3)]
        tensor_eqs[1] += [to_tensor_eqs(r3_left_eq, pt3)]
        tensor_eqs[2] += [to_tensor_eqs(r3_diag_eq, pt3)]
        tensor_eqs[3] += [to_tensor_eqs(r3_left_diag_eq, pt3)]
    return tensor_eqs

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
    cc.emp2 -= np.einsum("jiab,ijab", t2, eris_oovv.conj(), optimize='optimal').real
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

def cc_lamb_kernel(cc, tamps=None, lamps=None, max_cycle=50, tol=1e-8, tolnormt=1e-6):
    from pyscf.lib import logger
    from pyscf import lib
    import numpy as np

    log = logger.new_logger(cc, cc.verbose)
    if tamps is None:
        tamps = cc.tamps
    if lamps is None:
        lamps = [np.copy(x) for x in cc.tamps]
    if cc.lamb_eqs is None:
        cc.lamb_eqs = get_cc_lambda_eqs(cc.order, normal_ord=cc.normal_ord)

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())

    if cc.diis:
        adiis = lib.diis.DIIS(cc, None, incore=True)
        adiis.space = cc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        lamps_new = cc.update_lambda(lamps, tamps)
        tmpvec = cc.amplitudes_to_vector(lamps_new)
        tmpvec -= cc.amplitudes_to_vector(lamps)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if cc.iterative_damping < 1.0:
            alpha = cc.iterative_damping
            for tx, tx_new in zip(lamps, lamps_new):
                tx_new *= alpha
                tx_new += (1 - alpha) * tx
        lamps = lamps_new
        lamps_new = None
        lamps = cc.run_diis(lamps, istep, normt, 0.0, adiis)
        log.info("cycle =%3d  Lambda(%s)  norm(l amps) = %10.3e", istep + 1, cc.name, normt)
        cput1 = log.timer("%s iter" % cc.name, *cput1)
        if normt < tolnormt:
            conv = True
            break
    log.timer(cc.name, *cput0)
    cc.lamb_converged, cc.lamps = conv, lamps
    logger.info(cc, "%s %sconverged", cc.name, "lambda " + ("" if cc.converged else "not "))
    return lamps

def rcc_energy(cc, tamps):
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

def rt_purify(r):
    import itertools, numpy as np
    r, ip, n = np.copy(r), r.ndim % 2, r.ndim // 2
    for perm in itertools.combinations(range(n), 3):
        idxl, idxr = [slice(None)] * n, [slice(None)] * n
        for p in perm:
            idxl[p] = np.mgrid[:r.shape[p]]
        for p in perm:
            idxr[p] = np.mgrid[:r.shape[p + n]]
        r[(slice(None), ) * ip + tuple(idxl) + (slice(None), ) * n] = 0.0
        r[(slice(None), ) * ip + (slice(None), ) * n + tuple(idxr)] = 0.0
    return r

def rcc_update_amps(cc, tamps):
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
        ts_new[kk] = rt_purify(ts_new[kk])
        xi, xa, xt = 'ijklmnop'[:kk + 1], 'abcdefgh'[:kk + 1], tamps[kk]
        for ii, aa in zip(xi, xa):
            ts_new[kk] += np.einsum('%s,%s%s->%s%s' % (ii, xi, xa, xi, xa), fii, xt, optimize='optimal')
            ts_new[kk] -= np.einsum('%s,%s%s->%s%s' % (aa, xi, xa, xi, xa), faa, xt, optimize='optimal')
        exx = exx[((slice(None), ) * kk + (None, )) * 2] + eia[((None, ) * kk + (slice(None), )) * 2]
        ts_new[kk] /= exx
    return ts_new

def rcc_update_lambda(cc, lamps, tamps):
    import numpy as np
    ints, n_occ, eqs = cc.ints, cc.n_occ, cc.lamb_eqs
    sli = {'I': slice(n_occ), 'E': slice(n_occ, None)}
    ls_new = [np.zeros_like(t, dtype=ints[0].dtype) for t in lamps]
    fii, faa = np.diag(ints[0])[:n_occ], np.diag(ints[0])[n_occ:]
    if not cc.normal_ord:
        fii = fii + np.einsum("mjmj->m", ints[1][:n_occ, :n_occ, :n_occ, :n_occ], optimize='optimal')
        faa = faa + np.einsum("mjmj->m", ints[1][n_occ:, :n_occ, n_occ:, :n_occ], optimize='optimal')
    eia, exx = fii[:, None] - faa[None, :], np.array(0)
    for kk in range(0, cc.order):
        for f, script, idxs, nm in eqs[kk]:
            tensors = []
            for nmx, idx in zip(nm, idxs):
                tensors.append({'L': lamps, 'T': tamps}[nmx][len(idx) // 2 - 1] if nmx in 'LT'
                    else ints[len(idx) // 2 - 1][tuple(sli[x] for x in idx)])
            ls_new[kk] += f * np.einsum(script, *tensors, optimize='optimal')
        ls_new[kk] = rt_purify(ls_new[kk])
        xi, xa, xt = 'ijklmnop'[:kk + 1], 'abcdefgh'[:kk + 1], lamps[kk]
        for ii, aa in zip(xi, xa):
            ls_new[kk] += np.einsum('%s,%s%s->%s%s' % (ii, xi, xa, xi, xa), fii, xt, optimize='optimal')
            ls_new[kk] -= np.einsum('%s,%s%s->%s%s' % (aa, xi, xa, xi, xa), faa, xt, optimize='optimal')
        exx = exx[((slice(None), ) * kk + (None, )) * 2] + eia[((None, ) * kk + (slice(None), )) * 2]
        ls_new[kk] /= exx
    return ls_new

def rcc_eom_matmul(cc, eom_t, ramps, tamps, left=False):
    import numpy as np
    ints, n_occ, eqs = cc.ints, cc.n_occ, cc.eom_eqs[eom_t][left]
    sli = {'I': slice(n_occ), 'E': slice(n_occ, None)}
    rs_new = [np.zeros_like(r, dtype=ints[0].dtype) for r in ramps]
    for kk in range(0, cc.order):
        for f, script, idxs, nm in eqs[kk]:
            tensors = []
            for nmx, idx in zip(nm, idxs):
                tensors.append({'R': ramps, 'T': tamps}[nmx][(len(idx) + 1) // 2 - 1] if nmx in 'RT'
                    else ints[len(idx) // 2 - 1][tuple(sli[x] for x in idx)])
            rs_new[kk] += f * np.einsum(script, *tensors, optimize='optimal')
        rs_new[kk] = rt_purify(rs_new[kk])
    return rs_new

def rcc_eom_matdiag(cc, eom_t, tamps, left=False):
    import numpy as np
    ints, nocc, nvir, eqs = cc.ints, cc.n_occ, cc.n_virt, cc.eom_eqs[eom_t][2 + left]
    sli = {'I': slice(nocc), 'E': slice(nocc, None)}
    diags = [0 for _ in tamps]
    for it in range(cc.order):
        t_shape = [[nocc] * (it + (eom_t != 'ea')), [nvir] * (it + (eom_t != 'ip'))][::2 * (eom_t != 'ea') - 1]
        t_shape = (*t_shape[0], *t_shape[1])
        diags[it] = np.zeros(t_shape)
    for kk in range(0, cc.order):
        for f, script, idxs, nm in eqs[kk]:
            tensors = []
            for nmx, idx in zip(nm, idxs):
                if nmx == '=':
                    tensors.append(np.ones((1, ) * len(idx)))
                elif nmx == 'delta':
                    tensors.append({'II': np.identity(nocc), 'EE': np.identity(nvir)}[idx])
                else:
                    tensors.append(tamps[(len(idx) + 1) // 2 - 1] if nmx in 'T'
                        else ints[len(idx) // 2 - 1][tuple(sli[x] for x in idx)])
            diags[kk] += f * np.einsum(script, *tensors, optimize='optimal')
    return diags

def rt_symm_schemes(n_occ, n_vir, t_ord):
    from pyscf.fci import cistring
    import numpy as np
    n_fci_ts, cistrs = [], [[], []]
    for na in range(0, t_ord + 1):
        n_fci_ts.append((cistring.num_strings(n_occ, na), cistring.num_strings(n_vir, na)))
        cistrs[0].append(cistring.addrs2str(n_occ, na, np.mgrid[:n_fci_ts[na][0]]))
        cistrs[1].append(cistring.addrs2str(n_vir, na, np.mgrid[:n_fci_ts[na][1]]))
    tamp_addrs, tamp_masks = [[()[:0] for _ in range(t_ord + 1)] for _ in range(2)]
    for na, xshapes in enumerate(n_fci_ts):
        for nn, xcistr, xshape in zip([n_occ, n_vir], cistrs, xshapes):
            xmask = np.nonzero(xcistr[na][:, None] >> np.arange(nn)[None, :] & 1)[1].reshape(xcistr[na].shape[0], na).T
            tamp_addrs[na] += (np.mgrid[:xshape], )
            tamp_masks[na] += (xmask, )
    return (n_fci_ts, tamp_masks, tamp_addrs), cistrs

def rt_symm_patterns(n, reduced_pats):
    import itertools, functools
    dp = [[[()[:0]] if i == 0 and j == 0 else [] for i in range(n + 1)] for j in range(n + 1)]
    for p, k, i in [(p, k, i) for p in range(1, n + 1) for k in range(n + 1) for i in range(1, min(k + 1, 3) if reduced_pats else k + 1)]:
        dp[p][k].extend([(i, ) + r for r in dp[p - 1][k - i]])
    xp = [{()[:0]: [[]]} if i == 0 else {} for i in range(n + 1)]
    for i, g in [(i, g) for i in range(1, n + 1) for p in range(1, n + 1) for k in range(n + 1) for g in dp[p][k]]:
        xp[i][g] = []
        for x in (x for x in itertools.product(*[range(x + 1) for x in g]) if x != g):
            qx = tuple(ig - ix for ig, ix in zip(g, x))
            if not reduced_pats or sum(qx) <= 2:
                for ps in xp[i - 1].get(tuple(x for x in x if x != 0), []):
                    xs = [functools.reduce(lambda r, t: (r[0] + (0 if t == 0 else r[1][0], ), r[1][t != 0:]), x, [(), p])[0] for p in ps]
                    xp[i][g].append([qx] + xs)
    r = [[[[] for _ in range(m)] for _ in range(m)] for m in range(n + 1)]
    for ri, j, k in [(r[m][i], j, k) for m in range(n + 1) for i in range(m) for j in range(m) for k in dp[i + 1][m]]:
        lx = tuple(ix for ix, x in enumerate(k) for _ in range(x))
        rxs = [tuple(ix for k in range(len(k)) for ix, kr in enumerate(rg) for _ in range(kr[k])) for rg in xp[j + 1][k]]
        ri[j] += [(lx, rx) for rx in rxs]
    r[0] = [[[((), ())]]]
    return r

def rt_symm_take(x, symm_t, pats, shapes, masks, addrs):
    import numpy as np
    addr, shape = (), ()
    x = x.transpose(tuple(p for x in symm_t for p in x[2:]))
    for it, (ov, na) in enumerate([(x[1], x[0]) for x in symm_t if x[0] != 0]):
        assert na % 2 == 0
        xparts, xaddrs, xshape, xa, xb = [], [], 0, ov not in [-1, -2], ov in [-1, -3]
        for pi, pj, px in [(pi + 1, pj + 1, x) for pi, ps in enumerate(pats[na // 2]) for pj, px in enumerate(ps) for x in px]:
            xp = x[(slice(None), ) * it + tuple(masks[pi][xa][p] for p in px[0])][(slice(None), ) * (it + 1)
                + tuple(masks[pj][xb][p] for p in px[1])]
            xparts.append(xp.reshape((*xp.shape[:it], xp.shape[it] * xp.shape[it + 1], *xp.shape[it + 2:])))
            xaddrs.append((addrs[pi][xa][:, None] * shapes[pj][xb] + addrs[pj][xb][None, :]).ravel() + xshape)
            xshape += shapes[pi][xa] * shapes[pj][xb]
        x = np.concatenate(xparts, axis=it)
        addr = tuple(ix[..., None] for ix in addr) + (np.concatenate(xaddrs)[(None, ) * it], )
        shape = shape + (xshape, )
    xx = np.zeros(shape, dtype=x.dtype)
    xx[addr] = x
    return xx[tuple([None, slice(None)][x[0] != 0] for x in symm_t)]

def rt_symm_untake(x, symm_t, pats, shapes, masks):
    import itertools, numpy as np
    nt, xx = 0, x
    for it, (ov, na) in enumerate([(x[1], x[0]) for x in symm_t if x[0] != 0]):
        assert na % 2 == 0
        nn, xi, xj = na // 2, int(ov not in [-1, -2]), int(ov in [-1, -3])
        z = np.zeros(xx.shape[:nt] + (shapes[1][xi], ) * nn + (shapes[1][xj], ) * nn + x.shape[it + 1:])
        prebk = [(pi + (nn != 0), pj + (nn != 0), x[0], x[1]) for pi, ps in enumerate(pats[nn]) for pj, px in enumerate(ps) for x in px]
        presh = np.array(shapes)[np.array([x[0] for x in prebk]), xi] * np.array(shapes)[np.array([x[1] for x in prebk]), xj]
        preic = np.cumsum(np.insert(presh, 0, 0)[:-1], dtype=int)
        perms = np.array(list(itertools.permutations(range(nn))))
        for (pci, pcj, pxi, pxj), ic, nnc in zip(prebk, preic, presh):
            mask_i, mask_j = masks[pci][xi][np.array(pxi)], masks[pcj][xj][np.array(pxj)]
            for mi, mj in [(mask_i[p], mask_j[p]) for p in perms]:
                shape = xx.shape[:nt] + (mi.shape[1], mj.shape[1]) + x.shape[it + 1:]
                z[(slice(None), ) * nt + (*mi[:, :, None], *mj[:, None, :], ...)] = \
                    xx[(slice(None), ) * nt + (slice(ic, ic + nnc), ...)].reshape(shape)
        xx = z.transpose(*range(nt), *[nt + p for p in np.argsort(symm_t[it][2:])], *range(nt + na, nt + na + x.ndim - it - 1))
        nt += na
    return xx

def rt_symm_amps(eom_t, ts, symm_schemes):
    import numpy as np
    ts = list(ts)
    for it, xt in enumerate(ts):
        nn = xt.ndim // 2
        if eom_t == 'ee':
            ts[it] = rt_symm_take(xt, ((nn * 2, -1, *range(nn * 2)), ), *symm_schemes)
        else:
            if eom_t == 'ea':
                xt = xt.transpose((0, *range(1 + it, 1 + it + it), *range(1, 1 + it)))
            ts[it] = np.concatenate([rt_symm_take(x, ((nn * 2, -1, *range(nn * 2)), ), *symm_schemes)[None] for x in xt])
    return ts

def rt_unsymm_amps(eom_t, ts, symm_schemes):
    import numpy as np
    ts = list(ts)
    for it, xt in enumerate(ts):
        if eom_t == 'ee':
            ts[it] = rt_symm_untake(xt, (((it + 1) * 2, -1, *range((it + 1) * 2)), ), *symm_schemes[:3])
        else:
            ts[it] = np.concatenate([rt_symm_untake(x, ((it * 2, -1, *range(it * 2)), ), *symm_schemes[:3])[None] for x in xt])
            ts[it] = ts[it].reshape(ts[it].shape[:1 + it + it])
            if eom_t == 'ea':
                ts[it] = ts[it].transpose((0, *range(1 + it, 1 + it + it), *range(1, 1 + it)))
    return ts

def rcc_symm_amplitudes_to_vector(cc, tamps, out=None):
    import numpy as np
    vector = np.ndarray(np.sum([t.size for t in tamps]), tamps[0].dtype, buffer=out)
    size = 0
    for t in tamps:
        vector[size : size + t.size] = t.ravel()
        size += t.size
    return vector

def rcc_vector_to_symm_amplitudes(cc, eom_t, vector):
    z, it, tamps = 0, 0, []
    while len(tamps) < cc.order:
        pats, shapes = cc.symm_schemes[:2]
        if eom_t == 'ee':
            nz = sum(shapes[pi + 1][0] * shapes[pj + 1][1] * len(px)
                for pi, ps in enumerate(pats[it + 1]) for pj, px in enumerate(ps))
            t = vector[z:z + nz]
        else:
            nw = {'ip': cc.n_occ, 'ea':cc.n_virt}[eom_t]
            nz = nw * sum(shapes[pi + (it != 0)][0] * shapes[pj + (it != 0)][1] * len(px)
                for pi, ps in enumerate(pats[it]) for pj, px in enumerate(ps))
            t = vector[z:z + nz].reshape((nw, nz // nw))
        z += nz
        tamps.append(t)
        it += 1
    return tamps

def rcc_eom_amplitudes_to_vector(cc, eom_t, tamps, out=None):
    return rcc_symm_amplitudes_to_vector(cc, rt_symm_amps(eom_t, tamps, cc.symm_schemes), out=out)

def rcc_eom_vector_to_amplitudes(cc, eom_t, vector):
    return rt_unsymm_amps(eom_t, rcc_vector_to_symm_amplitudes(cc, eom_t, vector), cc.symm_schemes)

def eomcc_kernel(cc, eom_t, rampss=None, tamps=None, left=False, nroots=1, max_cycle=200, tol=1e-6):
    import numpy as np
    if eom_t not in cc.eom_eqs:
        cc.eom_eqs[eom_t] = get_eomcc_eqs(cc.order, eom_t)
    if tamps is None:
        tamps = cc.tamps
    diag = cc.eom_amplitudes_to_vector(eom_t, cc.eom_matdiag(eom_t, tamps=tamps, left=left))
    if rampss is None:
        rampss = [[0 for _ in range(cc.order)] for _ in range(nroots)]
        nocc, nvir = cc.n_occ, cc.n_virt
        for ir in range(nroots):
            for it in range(cc.order):
                t_shape = [[nocc] * (it + (eom_t != 'ea')), [nvir] * (it + (eom_t != 'ip'))][::2 * (eom_t != 'ea') - 1]
                t_shape = (*t_shape[0], *t_shape[1])
                rampss[ir][it] = np.zeros(t_shape)
        rampss = [cc.eom_amplitudes_to_vector(eom_t, ramps) for ramps in rampss]
        for ir, idx in enumerate(np.argsort(diag)[:nroots]):
            rampss[ir][idx] = 1.0
    def hop(vector):
        ramps = cc.eom_vector_to_amplitudes(eom_t, vector)
        ramps = cc.eom_matmul(eom_t, ramps, tamps, left=left)
        return cc.eom_amplitudes_to_vector(eom_t, ramps)
    eners, cc.rampss, _ = davidson_non_hermi(hop, rampss, diag, iprint=cc.verbose >= 5, conv_thrd=tol, max_iter=max_cycle * nroots)
    return eners[0] if nroots == 1 else eners, cc.rampss

def rcc_amplitudes_to_vector(cc, tamps, out=None):
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

def rcc_vector_to_amplitudes(cc, vector):
    nocc, nvir, nov = cc.n_occ, cc.n_virt, cc.n_occ * cc.n_virt
    z, it, tamps = 0, 0, []
    while z < vector.size:
        t = vector[z:z + nov ** (it + 1)].reshape((*([nocc] * (it + 1)), *([nvir] * (it + 1))))
        tamps.append(t)
        z += nov ** (it + 1)
        it += 1
    return tamps

def davidson_non_hermi(hop, ket, adiag=None, max_iter=5000, conv_thrd=1E-7, deflation_min_size=2,
                       deflation_max_size=30, iprint=False, imag_cutoff=1E-3):
    import numpy as np, time, scipy.linalg
    k = len(ket)
    if deflation_min_size < k:
        deflation_min_size = k
    for i in range(k):
        for j in range(i):
            ket[i] += -np.dot(ket[j].conj(), ket[i]) * ket[j]
        ket[i] /= np.linalg.norm(ket[i])
    sigma = [None] * k
    q, ck, msig, m, xiter = ket[0], 0, 0, k, 0
    while xiter < max_iter:
        tx = time.perf_counter()
        xiter += 1
        for i in range(msig, m):
            sigma[i] = hop(ket[i])
            msig += 1
        atilde = np.zeros((m, m), dtype=sigma[0].dtype)
        for i in range(m):
            for j in range(m):
                atilde[i, j] = np.dot(ket[i].conj(), sigma[j])
        eigv, alpha = scipy.linalg.eig(atilde)
        ixv = np.argsort(np.abs(eigv.imag))
        max_imag_tol = max(imag_cutoff, np.abs(eigv[min(m, k) - 1].imag))
        ixv = np.array(sorted(np.mgrid[:m], key=lambda i: (abs(eigv[i].imag) >= max_imag_tol,
            eigv[i].real if abs(eigv[i].imag) < max_imag_tol else abs(eigv[i].imag))))
        eigv, alpha = eigv[ixv], alpha[:, ixv]
        if ket[0].dtype != alpha.dtype:
            degen_idx = np.where(eigv.imag != 0)[0]
            if degen_idx.size > 0:
                alpha[:, degen_idx[1::2]] = alpha[:, degen_idx[1::2]].imag
            alpha = alpha.real
            for kk in range(k):
                alpha[:, kk] = alpha[:, kk] / np.linalg.norm(alpha[:, kk])
        eigv = eigv.real
        bx = [ib.copy() for ib in ket[:k]]
        for j in range(k):
            bx[j] = ket[j] * alpha[j, j]
        for j in range(k):
            for i in range(m):
                if i != j:
                    bx[j] += alpha[i, j] * ket[i]
        sigmax = [ib.copy() for ib in sigma[:k]]
        for j in range(k):
            sigmax[j] = sigma[j] * alpha[j, j]
        for j in range(k):
            for i in range(m):
                if i != j:
                    sigmax[j] += alpha[i, j] * sigma[i]
        for i in range(ck):
            q = sigmax[i].copy()
            q += (-eigv[i]) * bx[i]
            qq = np.dot(q.conj(), q)
            if np.sqrt(qq) >= conv_thrd:
                ck = i
                break
        qs = [None] * min(k - ck, 1)
        assert k > ck
        for ick in range(0, min(k - ck, 1)):
            qs[ick] = sigmax[ck + ick].copy()
            qs[ick] += (-eigv[ck + ick]) * bx[ck + ick]
        q = qs[0]
        qq = np.dot(q.conj(), q)
        if iprint:
            print("%5d %5d %5d %15.8f %9.2E T = %.3f" % (xiter, m, ck, eigv[ck], qq, time.perf_counter() - tx))
        if adiag is not None:
            for iq in range(len(qs)):
                t = bx[iq].copy()
                mask = np.abs(eigv[ck] - adiag) > 1E-12
                t[mask] = t[mask] / (eigv[ck] - adiag[mask])
                numerator = np.dot(t.conj(), qs[iq])
                denominator = np.dot(bx[iq].conj(), t)
                qs[iq] += (-numerator / denominator) * bx[iq]
                qs[iq][mask] = qs[iq][mask] / (eigv[ck] - adiag[mask])
        if qq < 0 or np.sqrt(qq) < conv_thrd:
            ck += 1
            if ck == k:
                break
        else:
            if m >= deflation_max_size:
                m = 0
                msig = 0
                qs = [x.copy() for x in bx]
            for iq in range(len(qs)):
                for j in range(m):
                    qs[iq] += (-np.dot(ket[j].conj(), qs[iq])) * ket[j]
                for j in range(iq):
                    if qs[j] is not None:
                        qs[iq] += (-np.dot(qs[j].conj(), qs[iq])) * qs[j]
                if np.linalg.norm(qs[iq]) > 1E-14:
                    qs[iq] /= np.linalg.norm(qs[iq])
                else:
                    qs[iq] = None
            qs = [x for x in qs if x is not None]
            if m + len(qs) > len(ket):
                for _ in range(len(ket) - m + len(qs)):
                    ket.append(None)
                    sigma.append(None)
            for q in qs:
                ket[m] = q
                m += 1
        if xiter == max_iter:
            break
    ket[:k] = bx[:k]
    return eigv[:k], ket[:k], xiter

class RCC:
    def __init__(self, mf, t_order=3, nfrozen=0, normal_ord=True, ecore_target=None, verbose=None, diis=True, dtype=float):
        import sys
        self.order = t_order
        self.normal_ord = normal_ord
        self.eqs = get_cc_eqs(t_order, normal_ord=normal_ord)
        self.lamb_eqs = None
        self.eom_eqs = {}
        self.e_hf, self.ints, self.n_occ, self.n_virt = rt_ao2mo(mf,
            nfrozen=nfrozen, normal_ord=normal_ord, dtype=dtype, ecore_target=ecore_target)
        pats = rt_symm_patterns(self.order, True)
        shapes, masks, addrs = rt_symm_schemes(self.n_occ, self.n_virt, self.order)[0]
        self.symm_schemes = pats, shapes, masks, addrs
        self.mo_energy = mf.mo_energy
        self.stdout = sys.stdout
        self.verbose = mf.verbose if verbose is None else verbose
        self.diis = diis
        self.diis_space = 6
        self.diis_start_cycle = 0
        self.iterative_damping = 1.0
        self.name = "RCC%s" % "SDTQPH789"[:self.order]
        self.converged = False
    energy = rcc_energy
    update_amps = rcc_update_amps
    update_lambda = rcc_update_lambda
    init_amps = rcc_init_amps
    amplitudes_to_vector = rcc_amplitudes_to_vector
    vector_to_amplitudes = rcc_vector_to_amplitudes
    eom_amplitudes_to_vector = rcc_eom_amplitudes_to_vector
    eom_vector_to_amplitudes = rcc_eom_vector_to_amplitudes
    eom_matdiag = rcc_eom_matdiag
    eom_matmul = rcc_eom_matmul
    get_init_guess = rcc_init_amps
    run_diis = cc_run_diis
    kernel = cc_kernel
    solve_lambda = cc_lamb_kernel
    ipccsd = lambda cc, *args, **kwargs: eomcc_kernel(cc, 'ip', *args, **kwargs)
    eaccsd = lambda cc, *args, **kwargs: eomcc_kernel(cc, 'ea', *args, **kwargs)
    eeccsd = lambda cc, *args, **kwargs: eomcc_kernel(cc, 'ee', *args, **kwargs)
    e_tot = property(lambda self: self.e_hf + self.e_corr)

def RCCSD(*args, **kwargs):
    kwargs["t_order"] = 2
    return RCC(*args, **kwargs)

def RCCSDT(*args, **kwargs):
    kwargs["t_order"] = 3
    return RCC(*args, **kwargs)

if __name__ == "__main__":

    from pyscf import gto, scf, cc
    import numpy as np

    mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='cc-pvdz')
    mf = scf.RHF(mol).run(conv_tol=1E-14)

    from pyscf.cc import rccsd
    ccsd = rccsd.RCCSD(mf).run()
    print('E-ee (right) = ', ccsd.eomee_ccsd_singlet()[0])
    print('E-ip (right) = ', ccsd.ipccsd()[0])
    print('E-ip ( left) = ', ccsd.ipccsd(left=True)[0])
    print('E-ea (right) = ', ccsd.eaccsd()[0])
    print('E-ea ( left) = ', ccsd.eaccsd(left=True)[0])
    xl1, xl2 = ccsd.solve_lambda()

    mcc = RCCSD(mf, verbose=4, diis=False)
    mcc.kernel()
    assert abs(mcc.e_tot - -76.23486335601116) < 1E-6
    wl1, wl2 = mcc.solve_lambda()
    print('lambda diff = ', np.linalg.norm(xl1 - wl1), np.linalg.norm(xl2 - wl2))
    print('E-ee (right) = ', mcc.eeccsd(max_cycle=1000)[0])
    print('E-ip (right) = ', mcc.ipccsd()[0])
    print('E-ip ( left) = ', mcc.ipccsd(left=True)[0])
    print('E-ea (right) = ', mcc.eaccsd()[0])
    print('E-ea ( left) = ', mcc.eaccsd(left=True)[0])

    mcc = RCCSDT(mf, verbose=4, diis=False)
    mcc.kernel()
    assert abs(mcc.e_tot - -76.2385041073466) < 1E-6
    wl1, wl2, wl3 = mcc.solve_lambda()
    print('lambda diff = ', np.linalg.norm(xl1 - wl1), np.linalg.norm(xl2 - wl2))
    print('E-ee (right) = ', mcc.eeccsd(max_cycle=1000)[0])
    print('E-ip (right) = ', mcc.ipccsd()[0])
    print('E-ip ( left) = ', mcc.ipccsd(left=True)[0])
    print('E-ea (right) = ', mcc.eaccsd()[0])
    print('E-ea ( left) = ', mcc.eaccsd(left=True)[0])
