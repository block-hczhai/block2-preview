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
CCSDT in general orbitals with equations derived on the fly.
need internal contraction module of block2.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import WickGraph
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
    perm_map[("t", 4)] = WickPermutation.four_anti()
    perm_map[("t", 6)] = WickPermutation.pair_anti_symmetric(3)
    # perm_map[("t", 8)] = WickPermutation.pair_anti_symmetric(4)

    p = lambda x: WickExpr.parse(x, idx_map, perm_map)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    return p, pt


P, PT = init_parsers()  # parsers
NR = lambda x: x.expand(-1, True).simplify()  # normal order
FC = lambda x: x.expand(0).simplify()  # fully contracted
Z = P("")  # zero


def CommT(t, d):  # commutator with t (at order d)
    return lambda h, i: (1.0 / i) * (h ^ t).expand((d - i) * 4).simplify()


def HBar(h, t, d):  # exp(-t) h exp(t) (order d)
    return sum(itertools.accumulate([h, *range(1, d + 1)], CommT(t, d)), Z)


h1 = P("SUM <pq> h[pq] C[p] D[q]")
h2 = 0.25 * P("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
t1 = P("SUM <ai> t[ia] C[a] D[i]")
t2 = 0.25 * P("SUM <abij> t[ijab] C[a] C[b] D[j] D[i]")
t3 = (1.0 / 36.0) * P("SUM <abcijk> t[ijkabc] C[a] C[b] C[c] D[k] D[j] D[i]")
# t4 = (1.0 / 24.0 ** 2) * P("SUM <abcdijkl> t[ijklabcd] C[a] C[b] C[c] C[d] D[l] D[k] D[j] D[i]")
ex1 = P("C[i] D[a]")
ex2 = P("C[i] C[j] D[b] D[a]")
ex3 = P("C[i] C[j] C[k] D[c] D[b] D[a]")
# ex4 = P("C[i] C[j] C[k] C[l] D[d] D[c] D[b] D[a]")

h = NR(h1 + h2)
t12 = NR(t1 + t2)
t123 = NR(t1 + t2 + t3)
# t1234 = NR(t1 + t2 + t3 + t4)

en_eq = FC(HBar(h, t12, 2))


def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E", WickIndexTypes.Inactive: "I"}
    allowed_perms = {"IIII", "IIIE", "IIEE", "IEEI", "IEIE", "IEEE", "EEEE"}
    for term in (eq.terms if isinstance(eq, WickExpr) else [t for g in eq.right for t in g.terms]):
        for wt in term.tensors:
            if wt.name == "v":
                k = "".join([imap[wi.types] for wi in wt.indices])
                if k not in allowed_perms:
                    found = False
                    for perm in wt.perms:
                        wtt = wt * perm
                        k = "".join([imap[wi.types] for wi in wtt.indices])
                        if k in allowed_perms:
                            wt.indices = wtt.indices
                            if perm.negative:
                                term.factor = -term.factor
                            found = True
                            break
                    assert found

gr_en_eq = WickGraph().add_term(PT("E"), en_eq).simplify()

def get_cc_amps_eqs(order):
    import time

    t = [None, None, t12, t123, None][order]
    eqs = []
    if order >= 1:
        print("1...", end="", flush=True)
        tt = time.perf_counter()
        t1_eq = FC(ex1 * HBar(h, t, 3))
        t1_eq = t1_eq + P("h[ii]\n - h[aa]") * P("t[ia]")
        fix_eri_permutations(t1_eq)
        # eqs.append(t1_eq.to_einsum(PT("t1new[ia]")))
        eqs.append(WickGraph().add_term(PT("t1new[ia]"), t1_eq).simplify().to_einsum())
        print("%8.3f sec" % (time.perf_counter() - tt))
    if order >= 2:
        print("2...", end="", flush=True)
        tt = time.perf_counter()
        t2_eq = FC(ex2 * HBar(h, t, 4))
        t2_eq = t2_eq + P("h[ii]\n + h[jj]\n - h[aa]\n - h[bb]") * P("t[ijab]")
        fix_eri_permutations(t2_eq)
        # eqs.append(t2_eq.to_einsum(PT("t2new[ijab]")))
        eqs.append(WickGraph().add_term(PT("t2new[ijab]"), t2_eq).simplify().to_einsum())
        print("%8.3f sec" % (time.perf_counter() - tt))
    if order >= 3:
        print("3...", end="", flush=True)
        tt = time.perf_counter()
        t3_eq = FC(ex3 * HBar(h, t, 5))
        t3_eq = t3_eq + P(
            "h[ii]\n + h[jj]\n + h[kk]\n - h[aa]\n - h[bb]\n - h[cc]"
        ) * P("t[ijkabc]")
        fix_eri_permutations(t3_eq)
        # eqs.append(t3_eq.to_einsum(PT("t3new[ijkabc]")))
        eqs.append(WickGraph().add_term(PT("t3new[ijkabc]"), t3_eq).simplify().to_einsum())
        print("%8.3f sec" % (time.perf_counter() - tt))
    # if order >= 4:
    #     print('4...', end='', flush=True)
    #     tt = time.perf_counter()
    #     t4_eq = FC(ex4 * HBar(h, t, 6))
    #     t4_eq = t4_eq + P("h[ii]\n + h[jj]\n + h[kk]\n + h[ll]\n - h[aa]\n - h[bb]\n - h[cc]\n - h[dd]") * P("t[ijklabcd]")
    #     fix_eri_permutations(t4_eq)
    #     # eqs.append(t4_eq.to_einsum(PT("t4new[ijklabcd]")))
    #     eqs.append(WickGraph().add_term(PT("t4new[ijklabcd]"), t4_eq).simplify().to_einsum())
    #     print("%8.3f sec" % (time.perf_counter() - tt))
    return eqs


eqs = [None] * 6

from pyscf.cc import gccsd


def wick_energy(cc, tamps, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    assert cc.level_shift == 0
    nocc = tamps[0].shape[0]
    E = np.array(0.0)
    tdics = {}
    for it, t in enumerate(tamps):
        tdics["t" + "I" * (it + 1) + "E" * (it + 1)] = t
    exec(
        gr_en_eq.to_einsum(),
        globals(),
        {"hIE": eris.fock[:nocc, nocc:], "vIIEE": np.array(eris.oovv), **tdics, "E": E},
    )
    return E


def wick_update_amps(cc, tamps, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    assert cc.level_shift == 0
    nocc = tamps[0].shape[0]
    tamps_new = [None] * len(tamps)
    tdics = {}
    for it, t in enumerate(tamps):
        tamps_new[it] = np.zeros_like(t)
        tdics["t" + "I" * (it + 1) + "E" * (it + 1)] = t
        tdics["t%dnew" % (it + 1)] = tamps_new[it]
    if eqs[cc.order] is None:
        eqs[cc.order] = get_cc_amps_eqs(cc.order)
    amps_eq = "".join(eqs[cc.order])
    exec(
        amps_eq,
        globals(),
        {
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
            **tdics,
        },
    )
    fii, faa = np.diag(eris.fock)[:nocc], np.diag(eris.fock)[nocc:]
    eia = fii[:, None] - faa[None, :]
    tamps_new[0] /= eia
    if cc.order >= 2:
        eiiaa = eia[:, None, :, None] + eia[None, :, None, :]
        tamps_new[1] /= eiiaa
    if cc.order >= 3:
        eiiiaaa = eiiaa[:, :, None, :, :, None] + eia[None, None, :, None, None, :]
        tamps_new[2] /= eiiiaaa
    # if cc.order >= 4:
    #     eiiiiaaaa = (
    #         eiiiaaa[:, :, :, None, :, :, :, None]
    #         + eia[None, None, None, :, None, None, None, :]
    #     )
    #     tamps_new[3] /= eiiiiaaaa
    # if cc.order >= 5:
    #     eiiiiiaaaaa = (
    #         eiiiiaaaa[:, :, :, :, None, :, :, :, :, None]
    #         + eia[None, None, None, None, :, None, None, None, None, :]
    #     )
    #     tamps_new[4] /= eiiiiiaaaaa
    # if cc.order >= 6:
    #     eiiiiiiaaaaaa = (
    #         eiiiiiaaaaa[:, :, :, :, :, None, :, :, :, :, :, None]
    #         + eia[None, None, None, None, None, :, None, None, None, None, None, :]
    #     )
    #     tamps_new[5] /= eiiiiiiaaaaaa
    return tamps_new


def wick_amplitudes_to_vector(tamps, out=None):
    nocc, nvir = tamps[0].shape
    nov = nocc * nvir
    size = 0
    for it, _ in enumerate(tamps):
        size += nov ** (it + 1)
    vector = np.ndarray(size, tamps[0].dtype, buffer=out)
    size = 0
    for it, t in enumerate(tamps):
        vector[size : size + nov ** (it + 1)] = t.ravel()
        size += nov ** (it + 1)
    return vector


def wick_vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    size = 0
    tamps = []
    it = 0
    while size < vector.size:
        t = vector[size : size + nov ** (it + 1)].reshape(
            (*([nocc] * (it + 1)), *([nvir] * (it + 1)))
        )
        tamps.append(t)
        size += nov ** (it + 1)
        it += 1
    return tamps


def wick_run_diis(mycc, tamps, istep, normt, de, adiis):
    from pyscf.lib import logger

    if (
        adiis
        and istep >= mycc.diis_start_cycle
        and abs(de) < mycc.diis_start_energy_diff
    ):
        vec = mycc.amplitudes_to_vector(tamps)
        tamps = mycc.vector_to_amplitudes(adiis.update(vec))
        logger.debug1(mycc, "DIIS for step %d", istep)
    return tamps


def wick_init_amps(mycc, eris=None):
    from pyscf.lib import logger
    from pyscf import lib

    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    mo_e = eris.mo_energy
    nocc = mycc.nocc
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    eijab = lib.direct_sum("ia,jb->ijab", eia, eia)
    t1 = eris.fock[:nocc, nocc:] / eia
    eris_oovv = np.array(eris.oovv)
    t2 = eris_oovv / eijab
    mycc.emp2 = 0.25 * np.einsum("ijab,ijab", t2, eris_oovv.conj(), optimize=True).real
    logger.info(mycc, "Init t2, MP2 energy = %.15g", mycc.emp2)
    tamps = [t1, t2]
    for it in range(3, mycc.order + 1):
        tamps.append(
            np.zeros((*([t1.shape[0]] * it), *([t1.shape[1]] * it)), dtype=t1.dtype)
        )
    return mycc.emp2, tamps[: mycc.order]


def wick_kernel(
    mycc,
    eris=None,
    tamps=None,
    max_cycle=50,
    tol=1e-8,
    tolnormt=1e-6,
    verbose=None,
    callback=None,
):
    from pyscf.lib import logger
    from pyscf import lib

    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if tamps is None:
        tamps = mycc.get_init_guess(eris)

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    ecc = mycc.energy(tamps, eris)
    suf = "SDTQPH789"[: mycc.order]
    log.info("Init E_corr(CC%s) = %.15g", suf, ecc)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        tamps_new = mycc.update_amps(tamps, eris)
        if callback is not None:
            callback(locals())
        tmpvec = mycc.amplitudes_to_vector(tamps_new)
        tmpvec -= mycc.amplitudes_to_vector(tamps)
        normt = np.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            for tx, tx_new in zip(tamps, tamps_new):
                tx_new *= alpha
                tx_new += (1 - alpha) * tx
        tamps = tamps_new
        tamps_new = None
        tamps = mycc.run_diis(tamps, istep, normt, ecc - eold, adiis)
        eold, ecc = ecc, mycc.energy(tamps, eris)
        log.info(
            "cycle = %d  E_corr(CC%s) = %.15g  dE = %.9g  norm(t amps) = %.6g",
            istep + 1,
            suf,
            ecc,
            ecc - eold,
            normt,
        )
        cput1 = log.timer("CC%s iter" % suf, *cput1)
        if abs(ecc - eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer("CC%s" % suf, *cput0)
    return conv, ecc, tamps


class WickGCC(gccsd.GCCSD):
    def __init__(self, mf, order=3, **kwargs):
        self.order = order
        gccsd.GCCSD.__init__(self, mf, **kwargs)
        self.e_hf = mf.e_tot

    energy = wick_energy
    update_amps = wick_update_amps
    init_amps = wick_init_amps
    run_diis = wick_run_diis

    def amplitudes_to_vector(self, tamps, out=None):
        return wick_amplitudes_to_vector(tamps, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None:
            nocc = self.nocc
        if nmo is None:
            nmo = self.nmo
        return wick_vector_to_amplitudes(vec, nmo, nocc)

    def get_init_guess(self, eris=None):
        return self.init_amps(eris)[1]

    def kernel(self, tamps=None, eris=None):
        self.converged, self.e_corr, self.tamps = wick_kernel(
            self, eris=eris, tamps=tamps
        )
        self._finalize()

    def _finalize(self):
        from pyscf.lib import logger

        name = "GCC%s" % "SDTQPH789"[: self.order]
        if self.converged:
            logger.info(self, "%s converged", name)
        else:
            logger.note(self, "%s not converged", name)
        logger.note(
            self, "E(%s) = %.16g  E_corr = %.16g", name, self.e_tot, self.e_corr
        )
        return self


def GCCSD(*args, **kwargs):
    kwargs["order"] = 2
    return WickGCC(*args, **kwargs)


def GCCSDT(*args, **kwargs):
    kwargs["order"] = 3
    return WickGCC(*args, **kwargs)


# def GCCSDTQ(*args, **kwargs):
#     kwargs['order'] = 4
#     return WickGCC(*args, **kwargs)

if __name__ == "__main__":

    from pyscf import gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="cc-pvdz", verbose=4)
    mf = scf.GHF(mol).run(conv_tol=1e-14)
    ccsd = gccsd.GCCSD(mf).run()
    wccsd = GCCSD(mf).run()
    wccsdt = GCCSDT(mf).run()
    # wccsdtq = GCCSDTQ(mf).run()

    # 1...   3.880 sec ->  0.647 sec ->  0.533 sec
    # 2...  15.781 sec ->  2.584 sec ->  2.194 sec
    # 3... 102.674 sec -> 14.604 sec -> 12.316 sec
    # pvdz basis / 204 sec -> 53 sec per iter
    # E(HF)     = -76.0167894720743
    # E(GCCSD)  = -76.23486336279412
    # E(T)(ref) =  -0.003466431834820524
    # E(GCCSDT) = -76.2385041072569
