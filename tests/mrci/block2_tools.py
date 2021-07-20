
"""
Some tools for DMRG with python

:author: Henrik R. Larsson
:author: Huanchen Zhai

"""

# ATTENTION: block2 import has to come before numpy import in order not to mess up openmp
import block2
from block2 import Random, SZ, PointGroup, FCIDUMP, Global
from block2 import QCTypes, FuseTypes, NoiseTypes, TruncationTypes
from block2 import VectorDouble, VectorUInt8, VectorUInt16, VectorVectorInt, VectorUBond, VectorInt
from block2 import OpNames, SiteIndex, DecompositionTypes
from block2.sz import MovingEnvironment, DMRG, DMRGBigSite, DMRGBigSiteAQCC, DMRGBigSiteAQCCOLD
from block2.sz import MPS, MPSInfo, MPOQC, SimplifiedMPO, HamiltonianQC, RuleQC, Rule
# from block2.sz import DelayedOperatorTensor, OpElement, SymbolicColumnVector
from block2.sz import StateInfo, SparseMatrix, SparseMatrixInfo, UnfusedMPS
from block2.sz import HamiltonianQCBigSite, SCIFockBigSite
from block2.sz import MPO, IdentityMPO, DiagonalMPO
from block2 import TruncationTypes, OpNamesSet
try:
    # parallel
    from block2.sz import MPICommunicator, ParallelRuleQC, ParallelTensorFunctions, ParallelMPO
    hasMPI = True
except ImportError:
    MPICommunicator = ParallelRuleQC = None
    hasMPI = False
MPICommunicator = ParallelRuleQC = ParallelRuleIdentity = None
hasMPI = False

DMRGSCI = DMRGBigSite
DMRGSCIAQCC = DMRGBigSiteAQCC
DMRGSCIAQCCOLD = DMRGBigSiteAQCCOLD

import itertools
from typing import List, Tuple
from pyscf import tools, ao2mo
import numpy as np
import time
import sys
import subprocess
import shutil
import os  # copy
import psutil


def getVerbosePrinter(verbose, indent="", flush=False):
    if verbose:
        if flush:
            def _print(*args, **kwargs):
                kwargs["flush"] = True
                print(indent, *args, **kwargs)
        else:
            def _print(*args, **kwargs):
                print(indent, *args, **kwargs)
    else:
        _print = lambda *args, **kwargs: None
    return _print


def fmt_size(i, suffix='B'):
    if i < 1000:
        return "%d %s" % (i, suffix)
    else:
        a = 1024
        for pf in "KMGTPEZY":
            p = 2
            for k in [10, 100, 1000]:
                if i < k * a:
                    return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                p -= 1
            a *= 1024
    return "??? " + suffix

def get_QNs(big_site):
    while not isinstance(big_site, SCIFockBigSite):
        big_site = big_site.big_site
    return big_site.quantumNumbers

def restrictMPSInfo(info: MPSInfo, hamil: HamiltonianQCBigSite,
                    target: SZ,
                    qnRestriction, iSite=None):
    """ Restrict MPSinfo `info` via qnRestriction
    :param qnRestriction: Function with SZ as input.
            If returns true, the quantum numbers are disabled on fci dims
    :param iSite: Site of restriction; defaults to last site
    """
    n_sites = info.n_sites
    iSite = n_sites - 1 if iSite is None else iSite
    if iSite == 0:
        dims = info.left_dims_fci[iSite + 1]  # vacuum site is included => +1
    else:
        dims = info.right_dims_fci[iSite]
    for i in range(dims.n):
        if qnRestriction(dims.quanta[i]):
            #print("restrict",dims.quanta[i], dims.n_states[i])
            dims.n_states[i] = 0
    for ldf, rdf in zip(info.left_dims_fci, info.right_dims_fci):
        StateInfo.filter(ldf, rdf, target)
        StateInfo.filter(rdf, ldf, target)
        ldf.collect()
        rdf.collect()
    for k in range(0, n_sites):
        info.left_dims_fci[k + 1] = StateInfo.tensor_product_ref(
            info.left_dims_fci[k], hamil.basis[k], info.left_dims_fci[k + 1])
    for k in range(0, n_sites)[::-1]:
        info.right_dims_fci[k] = StateInfo.tensor_product_ref(
            hamil.basis[k], info.right_dims_fci[k + 1], info.right_dims_fci[k])
    for ldf, rdf in zip(info.left_dims_fci, info.right_dims_fci):
        StateInfo.filter(ldf, rdf, target)
        StateInfo.filter(rdf, ldf, target)
        ldf.collect()
        rdf.collect()
    for ldf, rdf in zip(info.left_dims, info.right_dims):  # zero states?
        ldf.collect()
        rdf.collect()


def saveMPStoDir(mps: MPS, mpsSaveDir: str, MPI: MPICommunicator = None):
    mps.save_data()  # Important! Saves canonical form
    if not os.path.exists(mpsSaveDir):
        os.makedirs(mpsSaveDir)
    mps.info.save_data(f"{mpsSaveDir}/mps_info.bin")

    def copyIt(fnam: str):
        if MPI is not None and MPI.rank != 0:
            return
        lastName = os.path.split(fnam)[-1]
        fst = f"cp -p {fnam} {mpsSaveDir}/{lastName}"
        try:
            subprocess.call(fst.split())
        except:  # May problem due to allocate memory
            print(f"# ATTENTION: saveMPStoDir with command'{fst}' failed!")
            print(f"# Error message: {sys.exc_info()[0]}")
            print(f"# Error message: {sys.exc_info()[1]}")
            print(f"# Try again with shutil")
            try:
                # vv does not copy metadata -.-
                shutil.copyfile(fnam, mpsSaveDir + "/" + lastName)
            except:
                print(f"\t# ATTENTION: saveMPStoDir with shutil also failed")
                print(f"\t# Error message: {sys.exc_info()[0]}")
                print(f"\t# Error message: {sys.exc_info()[1]}")
                print(f"\t# Try again with syscal")
                os.system(fst)

    for iSite in range(mps.n_sites + 1):
        fnam = mps.info.get_filename(False, iSite)
        copyIt(fnam)
        if MPI is not None:
            MPI.barrier()
        fnam = mps.info.get_filename(True, iSite)
        copyIt(fnam)
        if MPI is not None:
            MPI.barrier()
    for iSite in range(-1, mps.n_sites):  # -1 is data
        fnam = mps.get_filename(iSite)
        copyIt(fnam)
        if MPI is not None:
            MPI.barrier()


def loadMPSfromDir(mps_info: MPSInfo, mpsSaveDir: str, MPI: MPICommunicator = None):
    if mps_info is None:  # new way
        mps_info = MPSInfo(0)
        mps_info.load_data(f"{mpsSaveDir}/mps_info.bin")
    else:
        # TODO: It would be good to check if mps_info.bin is available
        #       and then compare it again mps_info input to see any mismatch
        pass

    def copyItRev(fnam: str):
        if MPI is not None and MPI.rank != 0:
            # ATTENTION: For multi node stuff, I assume that all nodes have one global scratch dir
            return
        lastName = os.path.split(fnam)[-1]
        fst = f"cp -p {mpsSaveDir}/{lastName} {fnam}"
        try:
            subprocess.call(fst.split())
        except:  # May problem due to allocate memory, but why???
            print(f"# ATTENTION: loadMPSfromDir with command'{fst}' failed!")
            print(f"# Error message: {sys.exc_info()[0]}")
            print(f"# Error message: {sys.exc_info()[1]}")
            print(f"# Try again with shutil")
            try:
                # vv does not copy metadata -.-
                shutil.copyfile(mpsSaveDir + "/" + lastName, fnam)
            except:
                print(f"\t# ATTENTION: loadMPSfromDir with shutil also failed")
                print(f"\t# Error message: {sys.exc_info()[0]}")
                print(f"\t# Error message: {sys.exc_info()[1]}")
                print(f"\t# Try again with syscal")
                os.system(fst)
    for iSite in range(mps_info.n_sites + 1):
        fnam = mps_info.get_filename(False, iSite)
        copyItRev(fnam)
        if MPI is not None:
            MPI.barrier()
        fnam = mps_info.get_filename(True, iSite)
        copyItRev(fnam)
        if MPI is not None:
            MPI.barrier()
    mps_info.load_mutable()
    mps = MPS(mps_info)
    for iSite in range(-1, mps_info.n_sites):  # -1 is data
        fnam = mps.get_filename(iSite)
        copyItRev(fnam)
        if MPI is not None:
            MPI.barrier()
    mps.load_data()
    mps.load_mutable()
    return mps


# a collection of quantum states (StateInfo) and DET associated with each state
class StateInfoDET:
    def __init__(self, info, dets):
        self.info = info
        self.dets = dets

    # tensor product
    def __matmul__(self, other):
        new_quanta = []
        new_n_states = []
        new_dets = []
        # state_info.hpp L191
        for i in range(self.info.n):
            for j in range(other.info.n):
                qc = self.info.quanta[i] + other.info.quanta[j]
                new_quanta.append(qc)
                nprod = self.info.n_states[i] * other.info.n_states[j]
                assert nprod <= 65535
                new_n_states.append(nprod)
                new_dets.append([da + db for da in self.dets[i]
                                 for db in other.dets[j]])
        sorted_idx = list(range(0, len(new_quanta)))
        sorted_idx.sort(key=lambda x: new_quanta[x])
        sorted_quanta = []
        sorted_n_states = []
        sorted_dets = []
        for q, grp in itertools.groupby(sorted_idx, key=lambda x: new_quanta[x]):
            grp = list(grp)
            sorted_quanta.append(q)
            sorted_n_states.append(sum([new_n_states[i] for i in grp]))
            sorted_dets.append(sum([new_dets[i] for i in grp], []))
        info = StateInfo()
        info.allocate(len(sorted_quanta))
        for i, (q, n) in enumerate(zip(sorted_quanta, sorted_n_states)):
            info.quanta[i] = q
            info.n_states[i] = n
        return StateInfoDET(info, sorted_dets)

    def __repr__(self):
        r = ""
        dstr = "0ab2"
        ii = 0
        for i in range(self.info.n):
            for j in range(self.info.n_states[i]):
                r += "I=%2d (IQ=%2d IN=%2d) %20r N = %5d DET = %s\n" % \
                    (ii, i, j, self.info.quanta[i], self.info.n_states[i],
                        ''.join(map(lambda x: dstr[x], self.dets[i][j])))
                ii += 1
        return r


# contract last `n_ctr_sites` with 'R..R' form to 'R' form
# mps should be in deallocated state before entering this function
def contract_right_fused(mps, n_ctr_sites=2):
    for i in range(mps.n_sites - n_ctr_sites, mps.n_sites):
        assert mps.canonical_form[i] == 'R'
    mps.info.load_right_dims(mps.n_sites)
    r = mps.info.right_dims[mps.n_sites]
    m = mps.info.get_basis(mps.n_sites - 1)
    mr = StateInfo.tensor_product_ref(
        m, r, mps.info.right_dims_fci[mps.n_sites - 1])
    r.reallocate(0)
    mr.reallocate(mr.n)
    mps.load_tensor(mps.n_sites - 1)
    ts = mps.tensors[mps.n_sites - 1]
    for i in range(mps.n_sites - 2, mps.n_sites - n_ctr_sites - 1, -1):
        # i is the index of left of the two tensors to be contracted
        mx = mps.info.get_basis(i)
        xmr = StateInfo.tensor_product_ref(mx, mr, mps.info.right_dims_fci[i])
        xmrc = StateInfo.get_connection_info(mx, mr, xmr)
        mps.info.load_right_dims(i)
        l = mps.info.right_dims[i]
        new_info = SparseMatrixInfo()
        new_info.initialize(l, xmr, mps.info.vacuum, False, False)
        new = SparseMatrix()
        new.allocate(new_info)
        mps.load_tensor(i)
        unfused = UnfusedMPS.transform_right_fused(i, mps, False)
        mps.unload_tensor(i)
        rmats = {}
        for j in range(ts.info.n):
            ql = ts.info.quanta[j].get_bra(ts.info.delta_quantum)
            rmats[ql] = np.array(ts[j], copy=True)
        mp = {}
        for iph, d in enumerate(unfused.data):
            qph = mx.quanta[iph]
            for ((ql, qr), mat) in d:
                # in unfused, all quantum numbers are in left form
                # we need to recover the right quantum numbers
                xql = mps.info.target - ql
                xqr = mps.info.target - qr
                if xql not in mp:
                    mp[xql] = {}
                if xqr in rmats:
                    mp[xql][(qph, xqr)] = np.array(mat.ref) @ rmats[xqr]
        for j in range(new_info.n):
            ql = new_info.quanta[j].get_ket()
            iq = xmr.find_state(ql)
            qqed = xmrc.n if iq == xmr.n - 1 else xmrc.n_states[iq + 1]
            ip = 0
            mat = np.array(new[j], copy=False)
            mat[:] = 0.0
            for qq in range(xmrc.n_states[iq], qqed):
                iqqa = xmrc.quanta[qq].data >> 16
                iqqb = xmrc.quanta[qq].data & 0xFFFF
                qa = mx.quanta[iqqa]
                qb = mr.quanta[iqqb]
                lp = mx.n_states[iqqa] * mr.n_states[iqqb]
                if (qa, qb) in mp[ql]:
                    mat[:, ip:ip + lp] = mp[ql][(qa, qb)]
                ip += lp
            assert ip == mat.shape[1]
        mr.reallocate(0)
        ts.info.reallocate(0)
        ts.reallocate(0)
        xmr.reallocate(xmr.n)
        xmrc.reallocate(0)
        l.reallocate(0)
        new_info.reallocate(new_info.n)
        new.reallocate(new.total_memory)
        ts = new
        r = mr
        m = mx
        mr = xmr
    for i in range(mps.n_sites + 1):
        mps.info.left_dims[i].reallocate(0)
    for i in range(mps.n_sites, -1, -1):
        mps.info.right_dims[i].reallocate(0)
    mr.reallocate(0)
    ts.info.reallocate(ts.info.n)
    ts.reallocate(ts.total_memory)
    return ts


def getEnergy(me):
    me.prepare()
    i = me.center
    me.move_to(i)
    if me.ket.tensors[i] is not None and me.ket.tensors[i + 1] is not None:
        MovingEnvironment.contract_two_dot(i, me.ket)
    else:
        # here tensor[i] is already two-site tensor
        me.ket.load_tensor(i)
        me.ket.tensors[i + 1] = None

    # Effective Hamiltonian
    # diagonal Hamiltonian is computed and used for Davidson preconditioning
    h_eff = me.eff_ham(FuseTypes.FuseLR, compute_diag=True)
    energy = h_eff.expect()[0][0]
    assert energy[0].name == OpNames.H
    return energy[1] + me.mpo.const_e * me.ket.tensors[i].norm() ** 2


def generate_dmrg_schedule(maxM=500, startM=None, tol=1e-8, expand=True, maxIter=100, pyscfWay=True):
    # pyscfWay is apparently better for Cr2!
    # hrl: taken from dmrgci class of pyscf
    if startM is None:
        if maxM < 200:
            startM = 50
        else:
            startM = 200
    scheduleSweeps = []
    scheduleMaxMs = []
    scheduleTols = []
    scheduleNoises = []
    startM = startM
    N_sweep = 0
    Tol = 1.0e-4
    if pyscfWay:  # from pyscf.dmrgscf. does not make sense to have davTol 1e-10 or so and noise still 5e-5
        Noise = Tol
        while startM < int(maxM):
            scheduleSweeps.append(N_sweep)
            N_sweep += 4
            scheduleMaxMs.append(startM)
            startM *= 2
            scheduleTols.append(max(Tol, Noise / 10))  # hrl changed this
            scheduleNoises.append(Noise)
        while Tol > float(tol):
            scheduleSweeps.append(N_sweep)
            N_sweep += 2
            scheduleMaxMs.append(maxM)
            scheduleTols.append(Tol)
            Tol /= 10.0
            scheduleNoises.append(5e-5)
    else:
        # from stackblock; see also J. Chem. Phys. 142, 034102 (2015); table 1
        _maxMs = [50, 100, 250, 500, 1000, 2000,
                  4000, 5000, 6000, 7000, 8000, 10000, maxM]
        _maxIter = [8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4]
        _noise = [1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5,
                  5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 1e-5]
        N_sweep = 0
        if startM < _maxMs[0]:
            N_sweep += 8
            scheduleSweeps.append(N_sweep)
            scheduleMaxMs.append(startM)
            scheduleNoises.append(1e-4)
            scheduleTols.append(max(1e-5, tol))
        for m, i, n in zip(_maxMs, _maxIter, _noise):
            if m > maxM:
                break
            if m < startM:
                continue
            N_sweep += i
            scheduleSweeps.append(N_sweep)
            scheduleMaxMs.append(m)
            scheduleNoises.append(n)
            scheduleTols.append(max(n / 10, tol))
            N_sweep += i

    scheduleSweeps.append(N_sweep)
    scheduleMaxMs.append(maxM)
    scheduleTols.append(tol)
    scheduleNoises.append(0.0)
    N_sweep += 2
    twodot_to_onedot = N_sweep + 2
    maxIter = twodot_to_onedot + 12 if maxIter <= 0 else maxIter
    if not expand:
        return scheduleSweeps, scheduleNoises, scheduleTols, scheduleMaxMs, maxIter
    # hrl: expand
    noises = []
    tols = []
    maxMs = []
    scheduleSweeps.append(maxIter)
    scheduleNoises.append(scheduleNoises[-1])
    scheduleTols.append(scheduleTols[-1])
    scheduleMaxMs.append(scheduleMaxMs[-1])
    for N, n, t, m in zip(scheduleSweeps, scheduleNoises, scheduleTols, scheduleMaxMs):
        nStart = len(noises)
        for i in range(nStart, N):
            noises.append(n)
            tols.append(t)
            maxMs.append(m)
    return noises, tols, maxMs


def align_mps_center(ket: MPS, ref: MPS, pRule=None, opf=None):
    """ Make sure that ket and ref have same canonical form
    by changing center of ket

    Taken from block2/pyblock2/gfdmrg.py (Written by H. Zhai)
    """
    raise NotImplementedError("This does not work in general...; "
                              "use changeCanonincalForm")
    if ket.canonical_form == ref.canonical_form:
        return
    if opf is None:
        # Assume csr
        opf = block2.sz.CSROperatorFunctions(block2.sz.CG())
    if ref.center == 0:
        ket.center += 1
        ket.canonical_form = ket.canonical_form[:-1] + 'S'
        while ket.center != 0:
            ket.move_left(opf.cg, pRule)
    else:
        ket.canonical_form = 'K' + ket.canonical_form[1:]
        while ket.center != ket.n_sites - 1:
            ket.move_right(opf.cg, pRule)
        ket.center -= 1
    assert ket.canonical_form == ref.canonical_form, f"{ket.canonical_form} vs {ref.canonical_form}"
    return


def changeCanonicalForm(mps: MPS,
                        parallelRule: ParallelRuleQC = None,
                        keepStates=None
                        ):
    """ Right now, this only change from left (right) canonical form to right (left) one and cannot change dot mode"""
    opf = block2.sz.OperatorFunctions(block2.sz.CG())
    mpo = IdentityMPO(mps.info.basis, mps.info.basis, SZ(0, 0, 0), opf)
    if parallelRule is not None:
        # ParallelMPO always assumes simplification
        mpo = SimplifiedMPO(mpo, RuleQC(), True, False)
        mpo = ParallelMPO(mpo, parallelRule)
    # vv does not work. tensors[i] != nullptr assertion
    #mps.dot = 1
    if mps.dot == 2 and mps.center == mps.n_sites - 1:
        mps.center -= 1
    me = MovingEnvironment(mpo, mps, mps, "bondDimChange")
    me.init_environments(False)
    bondDim = mps.info.bond_dim
    bondDim = bondDim if bondDim != 0 else max(
        [x.n_states_total for x in mps.info.left_dims])  # after read in, bond dim is 0 -.-
    bondDim += 100  # for 2s, just for numerics
    dmrg = DMRGSCI(me, VectorUBond([bondDim]), VectorDouble([0]))
    if keepStates is not None:
        assert isinstance(keepStates, int)
        dmrg.trunc_type = TruncationTypes.KeepOne * keepStates
    dmrg.cutoff = 0.0
    dmrg.quanta_cutoff = 0.0
    dmrg.decomp_last_site = False
    dmrg.last_site_svd = True
    dmrg.last_site_1site = mps.dot == 2
    dmrg.forward = mps.center == 0 or mps.center == 1
    dmrg.davidson_soft_max_iter = 0
    dmrg.iprint = 0
    dmrg.solve(1, dmrg.forward, 1e-16)
    mpo.deallocate()
    return


def isFirstCanForm(mps: MPS):
    return mps.canonical_form[0] == "C" or mps.canonical_form[0] == "K" or mps.canonical_form[0] == "S"


def isLastCanForm(mps: MPS):
    # mps.center depends on dot algorithm; one could also check canonical_form != "R" / != "L"
    return mps.canonical_form[-1] == "C" or mps.canonical_form[-1] == "K" or mps.canonical_form[-1] == "S"


def _getWeight(tensor, referenceQNs: List[int]):
    nQuanta = tensor.info.n
    otherWeight = weightRef = 0
    for iQ in referenceQNs:
        arr = np.array(tensor[iQ], copy=False)
        #sz = tensor.info.quanta[iQ]
        weightRef += np.vdot(arr, arr)
    for iQ in np.setdiff1d(range(nQuanta), referenceQNs):
        arr = np.array(tensor[iQ], copy=False)
        otherWeight += np.vdot(arr, arr)
    return weightRef, otherWeight


def getDavidsonCorrectionRAS(mps: MPS, correlationEnergy, nElec, referenceQNs: List[int], firstSiteQNs: List[int] = None, allFirstSiteSZs: List[SZ] = None,
                             parallelRule=None):
    """ ATTENTION: This is the PROPER way to get davidson correlation for RAS partition. It *modifies* the MPS
        xQNs is the list of quantum number INDICES to be taken as reference.

    ATTENTION: THIS ONLY WORKS IF *ALL* QUANTA ARE INCLUDED IN THE STATE! 
        Thus, I added a keepStates = 1 in changeCanonicalForm. However, this does not seem to work always.
        For the first site, I already added "allFirstSiteSZs" that is a list of *all* available quantum numbers on first site , which is then used together with firstSiteQNs for using only those tensors that are still in list.
        However, for the last site, I have not yet added this.

    The last site has blocks x and y, where x is the vacuum (weightRef) and b the rest
    The first site has blocks a and b, where a is the non-closed determinants and b is the rest

    x,y contain contributions of a and b
    a,b contain contributions of x and y
    As a+b=1 and x+y=1, it is hard to separate them.
    So here I remove the contributions of a to get proper weight of reference
    """
    # vv make sure that states are kept; does work always?
    changeCanonicalForm(mps, parallelRule=parallelRule, keepStates=1)
    changeCanonicalForm(mps, parallelRule=parallelRule, keepStates=1)
    if isLastCanForm(mps):
        changeCanonicalForm(mps, parallelRule=parallelRule, keepStates=1)
    mps.load_tensor(0)
    tensor = mps.tensors[0]
    nQuanta = tensor.info.n
    weightFirstSite = 0
    if allFirstSiteSZs is not None:
        assert mps.canonical_form[0] == "C"
        info = tensor.info
        for iQ in firstSiteQNs:
            iiQ = info.find_state(allFirstSiteSZs[iQ] - info.delta_quantum)
            # ^^ actual number
            if iiQ < 0:
                continue
            arr = np.array(mps.tensors[0][iiQ], copy=False)
            weightFirstSite += np.vdot(arr, arr)
        for iQ in np.setdiff1d(range(len(allFirstSiteSZs)), firstSiteQNs):
            iiQ = info.find_state(allFirstSiteSZs[iQ] - info.delta_quantum)
            if iiQ < 0:
                continue
            arr = np.array(mps.tensors[0][iiQ], copy=False)
            arr.fill(0)
    else:
        for iQ in firstSiteQNs:
            assert iQ < nQuanta, f"nQuanta={nQuanta} but iQ={iQ}; are not all states included?"
            arr = np.array(mps.tensors[0][iQ], copy=False)
            weightFirstSite += np.vdot(arr, arr)
        for iQ in np.setdiff1d(range(nQuanta), firstSiteQNs):
            arr = np.array(mps.tensors[0][iQ], copy=False)
            arr.fill(0)
    mps.save_tensor(0)
    mps.unload_tensor(0)
    changeCanonicalForm(mps, parallelRule=parallelRule, keepStates=1)
    mps.load_tensor(mps.n_sites - 1)
    weightRef, otherWeight = _getWeight(
        mps.tensors[mps.n_sites - 1], referenceQNs)
    mps.unload_tensor(mps.n_sites - 1)
    # vv  changeCanonicalForm (the DMRG there) normalizes -.- so have to multiply by norm again
    weightRef *= weightFirstSite
    otherWeight *= weightFirstSite
    result = _getDavidsonCorrectionResult(
        weightRef, otherWeight, correlationEnergy, nElec)
    result["weightRefLast"] = 0
    result["otherWeightLast"] = 0
    result["weightRefFirst"] = 0
    result["otherWeightFirst"] = 0
    return result


def getDavidsonCorrection(mps: MPS, correlationEnergy, nElec, referenceQNs: List[int], firstSiteQNs: List[int] = None):
    """ xQNs is the list of quantum number INDICES to be taken as reference.  """
    # Davidson correction
    if not isLastCanForm(mps):
        changeCanonicalForm(mps)
    assert isLastCanForm(mps), f"{mps.canonical_form}"
    mps.load_tensor(mps.n_sites - 1)
    result = {}
    weightRef, otherWeight = _getWeight(
        mps.tensors[mps.n_sites - 1], referenceQNs)
    mps.unload_tensor(mps.n_sites - 1)
    if firstSiteQNs is not None:
        print("ATTENTION: THIS DOES NOT WORK! Use getDavidsonCorreactionRAS")
        # TODO: Remove this once the scripts are changed accordingly
        changeCanonicalForm(mps)
        assert isFirstCanForm(mps), f"{mps.canonical_form}"
        mps.load_tensor(0)
        weightRefFirst, otherWeightFirst = _getWeight(
            mps.tensors[0], firstSiteQNs)
        mps.unload_tensor(0)
        result["weightRefLast"] = weightRef
        result["otherWeightLast"] = otherWeight
        result["weightRefFirst"] = weightRefFirst
        result["otherWeightFirst"] = otherWeightFirst
        weightRef -= otherWeightFirst
        otherWeight -= weightRefFirst
    result = _getDavidsonCorrectionResult(
        weightRef, otherWeight, correlationEnergy, nElec)
    if firstSiteQNs is not None:
        result["weightRefLast"] = weightRef
        result["otherWeightLast"] = otherWeight
        result["weightRefFirst"] = weightRefFirst
        result["otherWeightFirst"] = otherWeightFirst
    return result


def _getDavidsonCorrectionResult(weightRef, otherWeight, correlationEnergy, nElec):
    result = {}
    result["weightRef"] = weightRef
    result["otherWeight"] = otherWeight
    davCorr = correlationEnergy * (1 - weightRef)
    davCorrNormalized = davCorr / weightRef
    # vv 10.1016/0009-2614(88)87431-1  non-iterative MR-AQCC
    meissnerCorr = davCorrNormalized * \
        (nElec - 2) * (nElec - 3) / (nElec * (nElec - 1))
    # Pople
    cos2 = 2 * weightRef - 1
    tan22 = (1 - cos2**2) / cos2**2
    popleCorr = correlationEnergy * \
        (nElec * (np.sqrt(1 + 2 * tan22 / nElec) - 1) / (2 / cos2 - 2) - 1)
    result["davCorr"] = davCorr
    result["davCorrNormalized"] = davCorrNormalized
    result["meissnerCorr"] = meissnerCorr
    result["popleCorr"] = popleCorr
    return result


def _getDavidsonCorrectionResultMod(weightRef, otherWeight, correlationEnergy, nElec):
    """ Modified davidson correction, see Eqs 49-51,
     Yuriy G. Khait, Wanyi Jiang, Mark R. Hoffmann, Chemical Physics Letters 493 (2010) 1–10
     "On the inclusion of triple and quadruple electron excitations into MRCISD for multiple states"
    """
    result = {}
    result["weightRef"] = weightRef
    result["otherWeight"] = otherWeight
    davCorr = correlationEnergy / weightRef
    davCorrNormalized = davCorr
    rAQCC = 1 - (nElec - 2) * (nElec - 3) / (nElec * (nElec - 1))
    rACPF = 2 / nElec
    meissnerCorr = correlationEnergy / (weightRef + rAQCC * (1 - weightRef))
    popleCorr = correlationEnergy / (weightRef + rACPF * (1 - weightRef))
#                           vvv Their definition is slightly different ; so substract correlationEnergy again
    result["davCorr"] = davCorr - correlationEnergy
    result["davCorrNormalized"] = davCorrNormalized - correlationEnergy
    result["meissnerCorr"] = meissnerCorr - correlationEnergy
    result["popleCorr"] = popleCorr - correlationEnergy
    return result


def getFCIDUMP(file, mol, wfnSym, h1e, eri, nElec: Tuple[int], eCore, orbsym, tol):
    assert hasattr(nElec, "__len__"), "nElec needs to be nela,nelb tuple"
    assert len(nElec) == 2, "nElec needs to be nela,nelb tuple"
    nTot = h1e.shape[0]
    assert np.all(np.array(orbsym) > 0), f"molpro orbsym needed!{orbsym}"
    assert len(orbsym) == nTot, f"nTot={nTot} vs |orbsym|={len(orbsym)}"
    if file is not None:
        tools.fcidump.from_integrals(file, h1e, eri, nTot, nElec, eCore,
                                     orbsym=orbsym, tol=tol)
        fcidump = FCIDUMP()
        fcidump.read(file)
    else:
        eri = ao2mo.restore(8, eri, nTot)
        h1e = h1e[np.tril_indices_from(h1e)].ravel()
        h1e[abs(h1e) < tol] = 0.0
        eri = eri.ravel()
        eri[abs(eri) < tol] = 0.0
        wfnSym_molpro = tools.fcidump.ORBSYM_MAP[mol.groupname][wfnSym]
        fcidump = FCIDUMP()
        nela, nelb = nElec
        fcidump.initialize_su2(
            nTot, nela + nelb, abs(nela - nelb), wfnSym_molpro, eCore, h1e, eri)
        fcidump.orb_sym = VectorUInt8(orbsym)
    return fcidump


class AQCC_Params:
    def __init__(self, nEl: int = 2, mode="AQCC", eRef=0.,
                 QNs=None,
                 old=False,
                 start_delta_e=0,
                 hamilAQCC: HamiltonianQCBigSite = None,
                 hamilAQCC2: HamiltonianQCBigSite = None):
        self.eRef = eRef
        self.old = old
        self.QNs = QNs  # Quantum numbers for old mode
        self.hamilAQCC = hamilAQCC  # for new mode
        self.hamilAQCC2 = hamilAQCC2  # for ACPF2
        self.nEl = nEl
        self.mode = mode
        self.start_delta_e = start_delta_e
        #
        self.G2 = 0
        if mode == "AQCC":
            self.G = 1 - (nEl - 3) * (nEl - 2) / (nEl * (nEl - 1))
        elif mode == "CEPA":
            self.G = 0  # TODO. Not quite; we have gA = 0 (see ACPF paper)
        elif mode == "ACPF":
            self.G = 2 / nEl
        else:
            assert mode == "ACPF2"
            self.G = 1 - (nEl - 3) * (nEl - 2) / (nEl * (nEl - 1))
            self.G2 = 2 / nEl


def doDMRGopt(mps: MPS, hamil: HamiltonianQC, noises, davTols, bond_dims, tol,
              oneSiteSweeps=0,
              oneSiteTol=None,
              enforceCanonicalForm=False,
              dav_max_iter=None,
              keepStates=None,
              cutoff=1e-12,
              initSweep=True,
              only1S=False,
              lastSite1Site=False,
              unscaledNoise=True,
              cachedContraction=False,
              aqcc=None,
              mpiComm: MPICommunicator = None,
              parallelRule: ParallelRuleQC = None,
              threeIdxMode=1,
              # 0: no   : most memory, slower runtime
              # 1:  normal_ops: fewer  memory , less runtime than 0
              # 2: all_ops: even fewer memory (almost half, cmp. to 0), more runtime than 0
              justSaveEverything=False,
              verbose=True):
    n_sweeps = len(noises)
    verboseOrig = verbose
    if mpiComm is not None and verbose:
        verbose = mpiComm.rank == 0
    if verbose:
        mem = psutil.Process(os.getpid()).memory_info().rss
        print("# before MPOQC; current memory: ", fmt_size(mem), flush=True)
    # MPO initialization
    if hasattr(hamil, 'n_orbs_left'):
        ntg = Global.threading.n_threads_global
        Global.threading.n_threads_global = 1
        mpo = MPOQC(hamil, QCTypes.NC)
        Global.threading.n_threads_global = ntg
    else:
        mpo = MPOQC(hamil, QCTypes.NC)
    if verbose:
        mem = psutil.Process(os.getpid()).memory_info().rss
        print("# after MPOQC; current memory: ", fmt_size(mem), flush=True)
    if threeIdxMode != 0 and not lastSite1Site:
        raise NotImplementedError(
            "threeIdxMode != 0 only works with lastSite1Site")

    #mpo = SimplifiedMPO(mpo, Rule(), False)
    if threeIdxMode != 0:
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
    else:
        mpo = SimplifiedMPO(mpo, RuleQC(), True, False)
    if justSaveEverything:
        _dir = "." if not isinstance(
            justSaveEverything, str) else justSaveEverything
        mps.info.save_data(_dir + '/mps_info.bin')
        mpo.basis = hamil.basis
        mpo.save_data(_dir + "/mpo.bin")
        print("saved")
        quit()

    if verbose:
        print('# left mpo dims = ', ''.join(
            ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
        print('# right mpo dims = ', ''.join(
            ["%6d" % (x.m * x.n) for x in mpo.right_operator_names]))
        # estimate memory
        maxD = max(bond_dims)
        mps_info2 = MPSInfo(mps.info.n_sites, mps.info.vacuum,
                            mps.info.target, mps.info.basis)
        mps_info2.set_bond_dimension(maxD)
        if only1S or oneSiteTol is not None:
            peakLowMemMode, peakMemNormalMode, diskStorage = mpo.estimate_storage(
                mps_info2, 1)
            print("# Estimate storage MPO 1 dot:", peakMemNormalMode /
                  1e9, "GB; disk:", diskStorage / 1e9, "GB")
        if not only1S:
            peakLowMemMode, peakMemNormalMode, diskStorage = mpo.estimate_storage(
                mps_info2, 2)
            print("# Estimate storage MPO 2 dot:", peakMemNormalMode /
                  1e9, "GB; disk:", diskStorage / 1e9, "GB")
        peakMemNormalMode, diskStorage = mps.estimate_storage(mps_info2)
        print("# Estimate storage MPS :", peakMemNormalMode / 1e9,
              "GB; disk:", diskStorage / 1e9, "GB", flush=True)
        mps_info2.deallocate_mutable()
        del mps_info2

    if hasMPI:
        assert mpiComm is not None, f"{mpiComm}  {mpiComm.__class__}"
        mpo = ParallelMPO(mpo, parallelRule)

    toDealloc = [mpo]

    # print(mpo.get_blocking_formulas())
    if verbose:
        mem = psutil.Process(os.getpid()).memory_info().rss
        print("# after SimplifiedMPO; current memory: ", fmt_size(mem), flush=True)
    if not only1S and lastSite1Site:
        if mps.center == mps.n_sites - 1:  # happens after readin
            if mpiComm is None or mpiComm.rank == 0:
                print("# ATTENTION: DMRG: mps canonical_form form:",
                      mps.canonical_form)
                print("# ATTENTION: DMRG: Setting mps.center to ",
                      mps.center - 1, "for hybrid 2s algorithm")
            mps.center -= 1

    me = MovingEnvironment(mpo, mps, mps, "DMRG")
    me.init_environments(False)  # verbose) # does not matter
    if cachedContraction:
        # ATTENTION: Need to set Global.frame.use_main_stack = False; not tested with linear/DMRGSCIAQCC
        me.cached_contraction = True
    if threeIdxMode == 1:
        # ATTENTION: Does not work for multiple MEs
        me.delayed_contraction = OpNamesSet.normal_ops()
    elif threeIdxMode == 2:
        # ATTENTION: Does not work for multiple MEs
        me.delayed_contraction = OpNamesSet.all_ops()
    else:
        assert threeIdxMode == 0, f"{threeIdxMode}"
    if verbose:
        mem = psutil.Process(os.getpid()).memory_info().rss
        print("# after MovingEnvironment; current memory: ", fmt_size(mem), flush=True)

    # DMRG: First 2 sweeps with normal noise and then actual sweep

    if aqcc is not None:
        if verbose:
            print("# DMRG; Do AQCC!")
        if mpiComm is not None:
            raise NotImplementedError
        assert isinstance(aqcc, AQCC_Params)
        if aqcc.old:
            dmrg = DMRGSCIAQCCOLD(me, VectorUBond(bond_dims), VectorDouble(noises),
                                  aqcc.G, aqcc.eRef, aqcc.QNs)
        else:
            hamilAQCC = aqcc.hamilAQCC
            dmpo = IdentityMPO(hamilAQCC)
            dmpo = SimplifiedMPO(dmpo, RuleQC(), True)
            toDealloc.append(dmpo)
            dme = MovingEnvironment(dmpo, mps, mps, "AQCC_DIAG")
            dme.init_environments(False)  # verbose) # does not matter
            if aqcc.mode == "ACPF2":
                dmpo2 = IdentityMPO(aqcc.hamilAQCC2)
                dmpo2 = SimplifiedMPO(dmpo2, RuleQC(), True)
                toDealloc.append(dmpo2)
                dme2 = MovingEnvironment(dmpo2, mps, mps, "AQCC_DIAG2")
                dme2.init_environments(False)  # verbose) # does not matter
            if hamilAQCC.get_n_orbs_left() == 0:  # no RAS
                if aqcc.mode != "ACPF2":
                    dmrg = DMRGSCIAQCC(me, aqcc.G, dme,
                                       VectorUBond(bond_dims), VectorDouble(noises), aqcc.eRef)
                else:
                    dmrg = DMRGSCIAQCC(me, aqcc.G, dme, aqcc.G2, dme2,
                                       VectorUBond(bond_dims), VectorDouble(noises), aqcc.eRef)
            else:
                mpoID = IdentityMPO(hamil)  # full Identity
                mpoID = SimplifiedMPO(mpoID, RuleQC(), True)
                toDealloc.append(mpoID)
                meID = MovingEnvironment(mpoID, mps, mps, "AQCC_ID")
                meID.init_environments(False)  # verbose) # does not matter

                if aqcc.mode == "ACPF2":  # TODO necessary?
                    mpoID2 = IdentityMPO(hamil)  # full Identity
                    mpoID2 = SimplifiedMPO(mpoID2, RuleQC())
                    toDealloc.append(mpoID2)
                    meID2 = MovingEnvironment(mpoID2, mps, mps, "AQCC_ID2")
                    # verbose) # does not matter
                    meID2.init_environments(False)
                if aqcc.mode != "ACPF2":
                    dmrg = DMRGSCIAQCC(me, aqcc.G, meID, dme,
                                       VectorUBond(bond_dims), VectorDouble(noises), aqcc.eRef)
                else:
                    dmrg = DMRGSCIAQCC(me, aqcc.G, meID, dme, aqcc.G2, meID2, dme2,
                                       VectorUBond(bond_dims), VectorDouble(noises), aqcc.eRef)
            dmrg.delta_e = aqcc.start_delta_e
    else:
        dmrg = DMRGSCI(me, VectorUBond(bond_dims), VectorDouble(noises))
    if not only1S and lastSite1Site:
        dmrg.last_site_1site = True
        if verbose:
            print("# DMRG: enable lastSite1Site", flush=True)
        # ATTENTION: For this option, I got slightly higher energies for a toy problem
        #   The energy was worse if dmrg.last_site_svd=False
        # dmrg.last_site_svd = False and dmrg.decomp_last_site = False: -107.6639582540
        # dmrg.last_site_svd = True and dmrg.decomp_last_site = False:  -107.6639854076
        # dmrg.last_site_svd = False and dmrg.decomp_last_site = True:  -107.6639582542
        # without this option and no decomp_last_site  and svd:          -107.6639924967
        #  I assume that this is some numerical noise
        assert mps.dot == 2
    elif mps.dot == 2:
        assert mps.center != mps.n_sites - \
            1, f"mps center = {mps.center} for n_sites={mps.n_sites} but for 2 dot w/o lastSite1Site, it cannot be a corner site"
    dmrg.decomp_last_site = False  # IMPORTANT! For speed and memory
    dmrg.last_site_svd = True
    if not verboseOrig:
        dmrg.iprint = 0

    dmrg.forward = mps.center == 0 or mps.center == 1
    if keepStates is not None:
        assert isinstance(keepStates, int)
        dmrg.trunc_type = TruncationTypes.KeepOne * keepStates
    dmrg.davidson_conv_thrds = VectorDouble(davTols)
    # dmrg.decomp_type = DecompositionTypes.SVD # does not work with ReducedPerturbativeXXXCollected

    if dav_max_iter is not None:
        dmrg.davidson_soft_max_iter = dav_max_iter
    if only1S:
        oneSiteSweeps = 0
        dmrg.me.dot = 1
        if unscaledNoise:
            # dmrg.noise_type = NoiseTypes.ReducedPerturbativeUnscaled # Perturbative is too time-consuming (see Mail from Oct 2,2020)
            # does not work with SVD decomp (unless in last site)
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeUnscaledCollectedLowMem
        else:
            # Sometimes much better performance, at least for RAS-type (nThawed != 0)
            #dmrg.noise_type = NoiseTypes.ReducedPerturbative
            # does not work with SVD decomp (unless in last site)
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
        dmrg.cutoff = cutoff
        assert not dmrg.decomp_last_site
        energy = dmrg.solve(n_sweeps, dmrg.forward, tol)
        discWeight = dmrg.discarded_weights[-1]
    else:
        # A) DMRG with random noise
        if abs(noises[0]) > 1e-14 and initSweep:
            # dmrg.noise_type = NoiseTypes.Wavefunction  # not able to add qnums
            dmrg.noise_type = NoiseTypes.DensityMatrix
            # dmrg.davidson_max_iter = 6 # throws an error
            energy = dmrg.solve(2, dmrg.forward, tol)
            discWeight = dmrg.discarded_weights[-1]
        else:
            uenergies, uweights, ubond_dims = [], [], []

        # B) DMRG with White's correction
        #dmrg.noise_type = NoiseTypes.Perturbative
        if unscaledNoise:
            # dmrg.noise_type = NoiseTypes.ReducedPerturbativeUnscaled # Perturbative is too time-consuming (see Mail from Oct 2,2020)
            # does not work with SVD decomp (unless in last site)
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeUnscaledCollectedLowMem
        else:
            # Sometimes much better performance, at least for RAS-type (nThawed != 0)
            #dmrg.noise_type = NoiseTypes.ReducedPerturbative
            # does not work with SVD decomp (unless in last site)
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
        dmrg.cutoff = cutoff
        #dmrg.davidson_max_iter = 20
        energy = dmrg.solve(n_sweeps, dmrg.forward, tol)
        discWeight = dmrg.discarded_weights[-1]
        if oneSiteSweeps > 0:
            if verbose:
                print("#------------------------")
                print("#-- one-site algorithm --")
                print("#------------------------")
                print("# two-site energy:", energy)
                print("# two-site disc. weight:", discWeight)
            dmrg.bond_dims = VectorUBond([dmrg.bond_dims[-1]])
            dmrg.davidson_conv_thrds = VectorDouble(
                [dmrg.davidson_conv_thrds[-1]])
            dmrg.noises = VectorDouble([0.])
            dmrg.decomp_last_site = False  # IMPORTANT!
            dmrg.me.dot = 1
            oneSiteTol = oneSiteTol if oneSiteTol is not None else tol / 2
            energy1s = dmrg.solve(oneSiteSweeps, dmrg.forward, oneSiteTol)
            if verbose:
                print("# one-site energy:", energy1s)
    # K for 1 site
    isCanonicalForm = mps.canonical_form[-1] == "C" or mps.canonical_form[-1] == "K" or mps.canonical_form[-1] == "S"
    if enforceCanonicalForm and not isCanonicalForm:
        if verbose:
            print("# additional sweep for canonical form")
        dmrg.noises = VectorDouble([0.])
        # Davidson tolerance not important.. actually yes as the 2site can deterioriate it
        dmrg.davidson_conv_thrds = VectorDouble([1e-7])
        dmrg.bond_dims = VectorUBond([dmrg.bond_dims[-1]])
        energy = dmrg.solve(1, dmrg.forward, tol)
    if verbose:
        print("# DMRG MovingEnvironment: tctr =",
              dmrg.me.tctr, "trot=", dmrg.me.trot)
    for _mpo in toDealloc[::-1]:
        _mpo.deallocate()  # ATTENTION: This makes returning the dmrg object not useful
    if oneSiteSweeps > 0:
        return (energy, energy1s), discWeight, dmrg.bond_dims[-1], dmrg
    else:
        return energy, discWeight, dmrg.bond_dims[-1], dmrg


if __name__ == "__main__":
    sweeps, noises, tols, maxMs, maxIter = generate_dmrg_schedule(
        2000, 200, 5e-8, False)
    for s, n, t, m in zip(sweeps, noises, tols, maxMs):
        print(s, n, t, m)
    noises, tols, maxMs = generate_dmrg_schedule(2000, 200, 5e-8, True)
    print("------")
    for n, t, m in zip(noises, tols, maxMs):
        print(n, t, m)
    print("maxIter=", maxIter)
