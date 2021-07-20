"""
Test AQCC (new driver)

Adapted from sciblock2

:author: Henrik R. Larsson
Modified by Huanchen Zhai (JUL 19 2021)
"""
import numpy as np
import tempfile
import unittest
from functools import reduce
from pyscf import gto, scf, tools
from pyscf import mcscf
from block2 import Random, SZ, PointGroup, FCIDUMP
from block2 import VectorDouble, VectorUInt8, VectorUInt16, VectorVectorInt, VectorUBond, VectorInt
from block2 import init_memory, release_memory, set_mkl_num_threads
from block2.sz import HamiltonianQC, MPSInfo, MPS, MPOQC, SimplifiedMPO, RuleQC, DiagonalMPO, SiteMPO
from block2.sz import IdentityMPO
from block2.sz import SCIFockBigSite, SimplifiedBigSite, HamiltonianQCBigSite
import block2
from block2_tools import *


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

def test_normal_mode():
    scratchDir = './nodex'
    # Compare old AQCC vs. new AQCC
    memory = int(10E9)
    Random.rand_seed(1234)
    init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9),
                save_dir=scratchDir)
    set_mkl_num_threads(1)
    coords = (0., 1.39, 2.5)
    mol = gto.Mole()
    mol.build(
        atom=[["Be", (0., 0., 0.)],
                ["H", coords],
                ["H", (0., -coords[1], coords[2])]],
        spin=0,
        basis={
            "Be": gto.parse(
                """
                Be S
                1267.07    0.001940
                190.356    0.014786
                43.2959    0.071795
                12.1442    0.236348
                3.80923    0.471763
                1.26847    0.355183
                Be S
                5.693880  -0.028876
                1.555630  -0.177565
                0.171855   1.071630
                Be S
                0.057181   1.00000
                Be P
                5.693880  0.004836
                1.555630   0.144045
                0.171855   0.949692
                """
            ),
            "H": gto.parse(
                """
    Be S
    19.2406   0.032828
    2.8992    0.231208
    0.6534    0.817238
    Be S
    0.17760   1.000
                """
            )
        },
        symmetry=True,
        symmetry_subgroup="c2v",
        unit="bohr",
        verbose=0
    )
    pg = 'c1' if not mol.symmetry else mol.symmetry_subgroup.lower()
    myhf = scf.RHF(mol)
    myhf.kernel()
    mc = mcscf.CASSCF(myhf, ncas=2, nelecas=2)
    mc.kernel()
    MO_COEFF = mc.mo_coeff
    if mol.symmetry:
        orbsym = MO_COEFF.orbsym
        orbsym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    else:
        orbsym = [1] * mol.na
    tools.fcidump.from_mo(mol, scratchDir+"/fcidump.tmp", MO_COEFF, orbsym=orbsym)
    fcidump = FCIDUMP()
    fcidump.read(scratchDir+"/fcidump.tmp")
    # DMRG parameters
    tol = 1E-12
    noises, davTols, bond_dims = generate_dmrg_schedule(maxM=2000,
                                                        startM=2000,
                                                        tol=tol)
    swap_pg = getattr(PointGroup, "swap_" + pg)
    vacuum = SZ(0, 0, 0)
    target = SZ(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))

    nExt = 6
    sciWrapRight = SciWrapper(mol.nao, nExt, True, fcidump, orb_sym, 2, 2, 2, False)
    hamilSCI = HamiltonianQCSCI(vacuum, mol.nao, orb_sym, fcidump, None, sciWrapRight)
    hamilSCI.opf.seq.mode = block2.SeqTypes.Simple
    # orbSym should be empty
    mps_infoSCI = MPSInfo(hamilSCI.n_sites, vacuum, target, hamilSCI.basis)
    mps_infoSCI.set_bond_dimension(bond_dims[0])
    mpsSCI = MPS(hamilSCI.n_sites, 0, 2)
    mpsSCI.initialize(mps_infoSCI)
    mpsSCI.random_canonicalize()
    mpsSCI.save_mutable()
    mpsSCI.save_data()
    mpsSCI.deallocate()
    mps_infoSCI.save_mutable()
    mps_infoSCI.deallocate_mutable()

    #################################
    nElCas = mol.nelectron
    CAS_ENERGY = -15.571389329181489
    QNs = UW(sciWrapRight).quantumNumbers[1:]
    aqcc = AQCC_Params(nEl=nElCas, eRef=CAS_ENERGY, mode="AQCC", old=True, QNs=QNs)
    energyAQCC1, discWeightSCI, bondDimSCI, dmrgSCI = doDMRGopt(mpsSCI, hamilSCI, noises, davTols, bond_dims, tol,
                                                                enforceCanonicalForm=True, lastSite1Site=True,
                                                                only1S=False,
                                                                threeIdxMode=0,
                                                                verbose=False,
                                                                aqcc=aqcc)
    #################################
    # AQCCnew
    #################################
    sciWrapRightAQCC = SciWrapperExcludeQNs(mol.nao, nExt, True, fcidump, orb_sym, 2, 2, 2, False)
    UW(sciWrapRightAQCC).qnIdxBra = VectorInt(np.arange(1, len(UW(sciWrapRightAQCC).quantumNumbers)))
    hamilSCIAQCC = HamiltonianQCSCI(vacuum, mol.nao, orb_sym, fcidump, None, sciWrapRightAQCC)
    aqccParams = AQCC_Params(nEl=nElCas, eRef=CAS_ENERGY, mode="AQCC",
                                hamilAQCC=hamilSCIAQCC)
    energyAQCC2, discWeightSCI, bondDimSCI, dmrgSCI = doDMRGopt(mpsSCI, hamilSCI, noises, davTols, bond_dims, tol,
                                                                enforceCanonicalForm=True, lastSite1Site=True,
                                                                only1S=False,
                                                                threeIdxMode=0,
                                                                verbose=False,
                                                                aqcc=aqccParams)

    print(f"# EAQCC1 = %20.15f" % energyAQCC1)
    print(f"# EAQCC2 = %20.15f" % energyAQCC2)
    print(f"# EREF   = %20.15f" % -15.623309683712941)
    assert abs(energyAQCC1 - energyAQCC2) < 1E-7
    assert abs(energyAQCC1 - -15.623309683712941) < 1E-7

def test_RAS_mode():
    scratchDir = './nodex'
    # Compare old AQCC vs. new AQCC
    memory = int(10E9)
    Random.rand_seed(1234)
    init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9),
                save_dir=scratchDir)
    set_mkl_num_threads(1)
    coords = (0., 1.39, 2.5)
    mol = gto.Mole()
    mol.build(
        atom=[["Be", (0., 0., 0.)],
                ["H", coords],
                ["H", (0., -coords[1], coords[2])]],
        spin=0,
        basis={
            "Be": gto.parse(
                """
                Be S
                1267.07    0.001940
                190.356    0.014786
                43.2959    0.071795
                12.1442    0.236348
                3.80923    0.471763
                1.26847    0.355183
                Be S
                5.693880  -0.028876
                1.555630  -0.177565
                0.171855   1.071630
                Be S
                0.057181   1.00000
                Be P
                5.693880  0.004836
                1.555630   0.144045
                0.171855   0.949692
                """
            ),
            "H": gto.parse(
                """
    Be S
    19.2406   0.032828
    2.8992    0.231208
    0.6534    0.817238
    Be S
    0.17760   1.000
                """
            )
        },
        symmetry=True,
        symmetry_subgroup="c2v",
        unit="bohr",
        verbose=0
    )

    pg = 'c1' if not mol.symmetry else mol.symmetry_subgroup.lower()
    myhf = scf.RHF(mol)
    enuc = myhf.energy_nuc()
    myhf.kernel()
    assert mol.nao < 14
    mc = mcscf.CASSCF(myhf, ncas=2, nelecas=2)
    mc.kernel()
    MO_COEFF = mc.mo_coeff
    if mol.symmetry:
        orbsym = MO_COEFF.orbsym
        orbsym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    else:
        orbsym = [1] * mol.nao
    # DMRG parameters
    tol = 1E-12
    noises, davTols, bond_dims = generate_dmrg_schedule(maxM=2000,
                                                        startM=2000,
                                                        tol=tol)

    n_sweeps = len(noises)
    print("n_sweeps=", n_sweeps)

    # Hamiltonian initialization:

    tools.fcidump.from_mo(mol, scratchDir+"/fcidump.tmp", MO_COEFF, orbsym=orbsym)
    fcidump = FCIDUMP()
    fcidump.read(scratchDir+"/fcidump.tmp")

    swap_pg = getattr(PointGroup, "swap_" + pg)
    vacuum = SZ(0, 0, 0)
    target = SZ(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    n_sites = fcidump.n_sites
    n_orb = n_sites
    orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
    #print("# fcidump symmetrize error:", fcidump.symmetrize(orb_sym), flush=True)

    ##################################################
    #######
    # build determinants
    nRAS = mc.ncore
    closed = list(range(2 * nRAS))
    # vv create excitations from core orbitals  -> holes
    RAS_space = [closed.copy()]
    for i in range(2 * nRAS):
        c = closed.copy()
        del c[i]
        RAS_space.append(c)
    for i in range(2 * nRAS):
        for j in range(i + 1, 2 * nRAS):
            c = closed.copy()
            del c[j]  # j > i so delete it first to have same order
            del c[i]
            RAS_space.append(c)
    #print("# RAS space:", len(RAS_space))
    #
    RAS_space = [block2.VectorInt(R) for R in RAS_space]  # vacuum
    RAS_space = block2.VectorVectorInt(RAS_space)
    sciWrapLeft = SciWrapper(n_orb, nRAS, False, fcidump, orb_sym, RAS_space,False)
    nExt = mol.nao - mc.ncas - nRAS
    sciWrapRight = SciWrapper(n_orb, nExt, True, fcidump, orb_sym, 2, 2, 2,False)
    hamilSCI = HamiltonianQCSCI(vacuum, n_orb, orb_sym, fcidump, sciWrapLeft, sciWrapRight)
    # orbSym should be empty
    mps_infoSCI = MPSInfo(hamilSCI.n_sites, vacuum, target, hamilSCI.basis)
    mps_infoSCI.set_bond_dimension(bond_dims[0])
    mpsSCI = MPS(hamilSCI.n_sites, 0, 1)
    mpsSCI.initialize(mps_infoSCI)
    mpsSCI.random_canonicalize()
    print(mpsSCI.canonical_form)
    mpsSCI.save_mutable()
    mpsSCI.save_data()
    mpsSCI.deallocate()
    mps_infoSCI.save_mutable()
    mps_infoSCI.deallocate_mutable()

    #################################
    # AQCC
    #################################
    nElCas = mol.nelectron
    sciWrapLeftAQCC = SciWrapperExcludeQNs(n_orb, nRAS, False, fcidump, orb_sym, RAS_space, False)
    UW(sciWrapLeftAQCC).qnIdxBra = VectorInt([len(UW(sciWrapLeftAQCC).quantumNumbers) - 1])
    sciWrapRightAQCC = SciWrapperExcludeQNs(n_orb, nExt, True, fcidump, orb_sym, 2, 2, 2, False)
    UW(sciWrapRightAQCC).qnIdxBra = VectorInt([0])

    hamilSCIAQCC = HamiltonianQCSCI(vacuum, n_orb, orb_sym, fcidump, sciWrapLeftAQCC, sciWrapRightAQCC)

    aqccParams = AQCC_Params(nEl=nElCas, eRef=mc.e_tot, mode="AQCC",
                                hamilAQCC=hamilSCIAQCC)
    # CI for warmup
    doDMRGopt(mpsSCI, hamilSCI, noises, davTols, bond_dims, tol,
                lastSite1Site=False, only1S=True,
                threeIdxMode=0, verbose=False)

    energyAQCC, discWeightSCI, bondDimSCI, dmrgSCI = doDMRGopt(mpsSCI, hamilSCI, noises, davTols, bond_dims, tol,
                                                                lastSite1Site=False, only1S=True,
                                                                threeIdxMode=0,
                                                                verbose=False,
                                                                aqcc=aqccParams)

    print(f"# EAQCC = %20.15f" % energyAQCC)
    print(f"# EREF  = %20.15f" % -15.623480954149368)
    assert abs(energyAQCC - -15.623480954149368) < 1E-6


if __name__ == "__main__":
    test_normal_mode()
    test_RAS_mode()
