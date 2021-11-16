"""
Sandeep/Sharma MR-LCC driver
 J. Chem. Phys. 143, 102815 (2015)

reproducing Table II (slightly different energy)

Adapted from sciblock2

:author: Henrik R. Larsson
Modified by Huanchen Zhai (JUL 19 2021)
"""
import os
import tempfile
import unittest
from pyscf import gto, scf, tools
from pyscf import mcscf, symm
from pyscf.mcscf import casci_symm
import sys
import numpy as np
from block2 import Random
from block2 import SZ, init_memory, release_memory, set_mkl_num_threads
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, VectorUBond, VectorInt, VectorChar
from block2 import VectorDouble, NoiseTypes, TruncationTypes
from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC, CASCIMPSInfo
from block2.sz import MPS, MPSInfo, DMRG, MovingEnvironment, StateInfo
from block2.sz import NoTransposeRule, Expect, MRCIMPSInfo, Rule

from block2.sz import LinearBigSite
import block2
from block2.sz import SimplifiedBigSite, SCIFockBigSite
from block2_tools import *

LinearSCI = LinearBigSite

def SciWrapper(*args, **kwargs):
    x = SCIFockBigSite(*args, **kwargs)
    return SimplifiedBigSite(x, RuleQC())

def SciWrapperExcludeQNs(*args, **kwargs):
    x = SCIFockBigSite(*args, **kwargs)
    x.excludeQNs = True
    return SimplifiedBigSite(x, RuleQC())

# unwrap
UW = lambda x : x if x is None or isinstance(x, SCIFockBigSite) else UW(x.big_site)

def MPOQCSCI(*args, **kwargs):
    ntg = Global.threading.n_threads_global
    Global.threading.n_threads_global = 1
    x = MPOQC(*args, **kwargs)
    Global.threading.n_threads_global = ntg
    return x

HamiltonianQCSCI = HamiltonianQCBigSite
IdentityMPOSCI = IdentityMPO


_print = getVerbosePrinter(False, flush=True)

def test_rept():
    scratchDir = './nodex'
    saveDir = './save_mps'

    DOT = 2
    nCore =    0
    nThawed = 2
    nElCas = 8
    nCas = 8
    USE_FCIDUMP=False
    FCIDUMP_TOL = 1e-12

    tol = 1E-6
    noises, davTols, bond_dims = generate_dmrg_schedule(maxM=500,
                                                        startM=100,
                                                        tol=tol)

    """# Set up the System
    Global parameters:
    """
    rand_seed = 1234
    #memory = int(20E9)
    memory = int(6E9)
    if "OMP_NUM_THREADS" in os.environ:
        n_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        n_threads = 2
    _print("# used threads=", n_threads)

    Random.rand_seed(rand_seed)
    set_mkl_num_threads(n_threads)
    block2.set_omp_num_threads(n_threads)

    ####################################################################################
    #Pyscf
    ####################################################################################
    bondDistance = 1.8
    init_memory(isize=int(1e7), dsize=int(memory), save_dir=scratchDir)
    _print("----------------------")
    _print("BOND DISTANCE",bondDistance)
    _print("----------------------",flush=True)
    mol = gto.Mole()
    mol.build(
            atom=[["C", (0., 0., 0.)],
                ["C", (0., 0., bondDistance)]],
            spin=0,
        basis="cc-pvdz",
        unit="bohr",
        symmetry=True,
        symmetry_subgroup="d2h",
        verbose=0
    )
    pg = 'c1' if not mol.symmetry else mol.symmetry_subgroup.lower()
    _print("## pg=",pg)
    myhf = scf.RHF(mol)
    myhf.kernel()
    mc = mcscf.CASSCF(myhf, ncas=nCas, nelecas=nElCas)
    mc.kernel()
    MO_COEFF = mc.mo_coeff
    E_CASSCF = mc.e_tot
    _print("# E_CASSCF=",E_CASSCF)

    _mcCI = mcscf.CASCI(myhf, ncas=mol.nao - nCore, nelecas=nElCas + nThawed * 2, ncore=nCore)
    _mcCI.mo_coeff = MO_COEFF
    _mcCI.mo_coeff = casci_symm.label_symmetry_(_mcCI, MO_COEFF)
    MO_COEFF = _mcCI.mo_coeff
    wfnSym = _mcCI.fcisolver.wfnsym
    wfnSym = wfnSym if wfnSym is not None else 0
    h1e, eCore = _mcCI.get_h1cas()
    orbSym = np.array(_mcCI.mo_coeff.orbsym)[nCore:]
    nTot = h1e.shape[0]
    nExt = nTot - nCas - nThawed
    _print(f"# nTot= {nTot}, nCas={nCas}, nExt={nExt}")
    assert nTot == mol.nao - nCore
    _print("# groupName=", mol.groupname)
    _print("# orbSym=", orbSym)
    _print("# orbSym=", [symm.irrep_id2name(mol.groupname, s) for s in orbSym])
    _print("# wfnSym=", wfnSym, symm.irrep_id2name(mol.groupname, wfnSym), flush=True)
    assert wfnSym == 0, "Want A1g state"
    molpro_orbsym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbSym]
    assert orbSym.size == nTot

    eri = _mcCI.get_h2cas()
    eri = np.require(eri, dtype=np.float64)
    h1e = np.require(h1e, dtype=np.float64)
    nelecas = _mcCI.nelecas
    del _mcCI, myhf

    ####################################################################################
    #Transfer
    ####################################################################################
    fcidump = getFCIDUMP("fcidump.tmp" if USE_FCIDUMP else None,
                            mol, wfnSym, h1e, eri, nelecas, eCore, molpro_orbsym, FCIDUMP_TOL)
    del h1e, eri

    swap_pg = getattr(PointGroup, "swap_" + pg)
    orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
    vacuum = SZ(0, 0, 0)
    target = SZ(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    n_orb = fcidump.n_sites

    nExt = n_orb - nCas - nThawed

    _print("# -------------------------")
    _print("# Psi0 site")
    _print("# -------------------------")
    if nThawed != 0:
        closed = list(range(2 * nThawed))
        # vv create excitations from core orbitals  -> holes
        RAS_space = [closed.copy()]
        for i in range(2 * nThawed):
            c = closed.copy()
            del c[i]
            RAS_space.append(c)
        for i in range(2 * nThawed):
            for j in range(i + 1, 2 * nThawed):
                c = closed.copy()
                del c[j]  # j > i so delete it first to have same order
                del c[i]
                RAS_space.append(c)
        _print("# RAS space:", len(RAS_space))
        RAS_space = [block2.VectorInt(R) for R in RAS_space]  # vacuum
        RAS_space = block2.VectorVectorInt(RAS_space)
        sciWrapLeftPsi0 = SciWrapperExcludeQNs(n_orb, nThawed, False, fcidump, orb_sym, RAS_space, False)
        UW(sciWrapLeftPsi0).setQnIdxBra(VectorInt(np.arange(len(UW(sciWrapLeftPsi0).quantumNumbers))),
                                    VectorChar(["H", "I", "B", "Q"]))
    else:
        sciWrapLeftPsi0 = None
    sciWrapRightPsi0 = SciWrapperExcludeQNs(n_orb, nExt, True, fcidump, orb_sym, 2,2,2, False)
    UW(sciWrapRightPsi0).setQnIdxBra(VectorInt(np.arange(len(UW(sciWrapRightPsi0).quantumNumbers))),
                                    VectorChar(["H", "I", "B", "Q"]))

    hamilPsi0 = HamiltonianQCSCI(vacuum, n_orb, orb_sym, fcidump, sciWrapLeftPsi0, sciWrapRightPsi0)
    hamilPsi0.opf.seq.mode = block2.SeqTypes.Simple

    mps_infoPsi0 = MPSInfo(hamilPsi0.n_sites, vacuum, target, hamilPsi0.basis)
    ##################
    ##################
    # ATTENTION vv not needed; just for performance purposes
    restrictMPSInfo(mps_infoPsi0, hamilPsi0, target, lambda x: x.n > 0)
    if nThawed != 0:
        restrictMPSInfo(mps_infoPsi0, hamilPsi0, target, lambda x: x.n != nThawed*2, iSite=0)
    mps_infoPsi0.tag = "Psi0"
    mps_infoPsi0.set_bond_dimension(bond_dims[0])
    mpsPsi0 = MPS(hamilPsi0.n_sites, 0, DOT)
    mpsPsi0.initialize(mps_infoPsi0)
    mpsPsi0.random_canonicalize()
    mpsPsi0.info.tag = "Psi0"
    mpsPsi0.save_mutable()
    mpsPsi0.save_data()
    mpsPsi0.deallocate()
    mps_infoPsi0.save_mutable()
    mps_infoPsi0.deallocate_mutable()

    energyPsi0, discWeightPsi0, bondDimPsi0, dmrgPsi0 = doDMRGopt(mpsPsi0, hamilPsi0, noises, davTols, bond_dims, tol,
                                                                threeIdxMode=1,
                                                                enforceCanonicalForm=True,
                                                                verbose=True,
                                                                only1S=DOT == 1,
                                                                #justSaveEverything=True,
                                                                unscaledNoise=False,
                                                                keepStates=5,
                                                                lastSite1Site=True)
    # saveMPStoDir(mpsPsi0, tmpDirName+f"/mpsPsi0")
    if DOT == 2 and mpsPsi0.center == mpsPsi0.n_sites -1:
        mpsPsi0.center -= 1 # this is so annoying... leads to problems later

    _print(f"# CASCI E = {energyPsi0:20.15f}")
    _print(f"# DIFFERENCE TO CASSCF ENERGY: {energyPsi0-E_CASSCF:20.15f}")
    assert abs(energyPsi0 - E_CASSCF) < 1E-6
    _print("# -------------------------")
    _print("# Correction site")
    _print("# -------------------------")
    if nThawed != 0:
        sciWrapLeftPsi1 = SciWrapper(n_orb, nThawed, False, fcidump, orb_sym, RAS_space, False)
    else:
        sciWrapLeftPsi1 = None

    sciWrapRightPsi1 = SciWrapper(n_orb, nExt, True, fcidump, orb_sym, 2, 2, 2, False)

    hamilFull = HamiltonianQCSCI(vacuum, n_orb, orb_sym, fcidump, sciWrapLeftPsi1, sciWrapRightPsi1)
    hamilFull.opf.seq.mode = block2.SeqTypes.Simple

    mps_infoPsi1 = MPSInfo(hamilFull.n_sites, vacuum, target, hamilFull.basis)
    mps_infoPsi1.set_bond_dimension(bond_dims[0])
    mps_infoPsi1.tag = "Psi1"

    if mpsPsi0.center != 0: # last site is somewhat troublesome for twosite and last_1_site
        changeCanonicalForm(mpsPsi0)
    mpsPsi1 = MPS(hamilFull.n_sites, mpsPsi0.center, mpsPsi0.dot)
    mpsPsi1.initialize(mps_infoPsi1)
    mpsPsi1.random_canonicalize()

    mpsPsi1.save_mutable()
    mpsPsi1.save_data()
    mpsPsi1.deallocate()
    mps_infoPsi1.save_mutable()
    mps_infoPsi1.deallocate_mutable()

    # vv Annoying but calling this twice aligns with mpsPsi0
    changeCanonicalForm(mpsPsi1)
    changeCanonicalForm(mpsPsi1)

    if DOT == 2 and mpsPsi1.center == mpsPsi1.n_sites -1:
        mpsPsi1.center -= 1 # leads to problems later
    if DOT == 2 and mpsPsi0.center == mpsPsi0.n_sites -1:
        mpsPsi0.center -= 1 # leads to problems later

    if mpsPsi1.canonical_form != mpsPsi0.canonical_form:
        changeCanonicalForm(mpsPsi0)
    assert mpsPsi1.canonical_form == mpsPsi0.canonical_form, f"{mpsPsi1.canonical_form} vs {mpsPsi0.canonical_form}"
    assert mpsPsi1.center == mpsPsi0.center, \
        f"{mpsPsi1.canonical_form} vs {mpsPsi0.canonical_form}; {mpsPsi1.center} vs. {mpsPsi0.center}"
    # ^^ can be changed by using either changeCanonical form or mpsPsi1.move_right


    # Verify <psi1|psi0>
    xme = MovingEnvironment(IdentityMPOSCI(hamilFull), mpsPsi1, mpsPsi0, "EX")
    xme.init_environments(False)
    ex = Expect(xme, bond_dims[0], bond_dims[0]).solve(False)
    _print(f'# <psi1|psi0> = {ex:20.15f} (should be 0)')

    # Psi1: H - E0 (ecore cancelled); LHS
    lmpo = MPOQCSCI(hamilPsi0, QCTypes.NC)
    lmpo = SimplifiedMPO(lmpo, RuleQC(), True)
    lmpo.const_e -= energyPsi0
    lme = MovingEnvironment(lmpo, mpsPsi1, mpsPsi1, "Psi1")
    lme.init_environments(False)

    # Psi0: -H (ecore has no effect since <bra|ket> = 0)
    # hamilFull.ruleQC = NoTransposeRule(RuleQC())
    hamilFull.big_left = SimplifiedBigSite(UW(hamilFull.big_left), NoTransposeRule(RuleQC()))
    hamilFull.big_right = SimplifiedBigSite(UW(hamilFull.big_right), NoTransposeRule(RuleQC()))
    rmpo = MPOQCSCI(hamilFull, QCTypes.NC)
    rmpo = SimplifiedMPO(rmpo, NoTransposeRule(RuleQC()), True)
    rmpo.const_e -= energyPsi0
    rme = MovingEnvironment(rmpo, mpsPsi1, mpsPsi0, "Psi0")
    rme.init_environments(False)

    # ex = <psi1|-H|psi0>
    cps = LinearSCI(lme, rme, None, VectorUBond(bond_dims), VectorUBond([bond_dims[-1]+400]*len(bond_dims)), VectorDouble(noises))
    cps.noise_type = NoiseTypes.ReducedPerturbative
    if DOT == 2:
        cps.last_site_1site = True
    cps.decomp_last_site = False  # IMPORTANT! For speed and memory
    cps.trunc_type = TruncationTypes.KeepOne * 1
    cps.last_site_svd = True
    cps.cutoff = 1e-12
    cps.linear_conv_thrds = VectorDouble([x / 50 for x in davTols])


    # Need extra quantum numbers to represent -H|bra> accurately
    cps.trunc_type = TruncationTypes.KeepOne * 1
    ex = cps.solve(len(bond_dims), mpsPsi1.center == 0, tol)
    _print(f"# LCC Energy = {energyPsi0-ex:20.15f}")
    eLCC = energyPsi0-ex
    print(f"# ELCC = %20.15f" % eLCC)
    print(f"# EREF = %20.15f" % -75.455183935505)
    assert abs(eLCC - -75.455183935505) < 1E-4

    rmpo.deallocate()
    lmpo.deallocate()
    hamilFull.deallocate()
    hamilPsi0.deallocate()
    release_memory()

if __name__ == "__main__":
    test_rept()
