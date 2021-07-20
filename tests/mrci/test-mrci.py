"""
Test DMRG as MRCI solver with big sites
Adapted from sciblock2

:author: Henrik R. Larsson
Modified by Huanchen Zhai (JUL 19 2021)
"""
import numpy as np
import tempfile
import unittest
from functools import reduce
from pyscf import gto
from main_driver import BLOCK2_SCI

def test_NormalVersusBig(PG):
    print("-- normal versus big",flush=True)
    nCas = 10
    nElCas = 14
    nCore = 2
    atString ='N 0 0 0.0; N 0 0 1.1208; '
    basis = "6-31G"
    nCas -= nCore
    nElCas -= 2 * nCore
    scratchDir = './nodex'
    saveDir = './save_mps'

    print("PG=",PG,flush=True)
    mol = gto.Mole()
    mol.build(
    atom=atString,
    spin=0,
    basis=basis,
    symmetry=False if PG is None else True,
    symmetry_subgroup = PG,
    verbose=0
    )
    driver = BLOCK2_SCI(mol, 0, nCas, mol.nelectron, 800,
                        fciMode=False, doCASSCF=True,
                        verbose=False,
                        memory=int(6e9),
                        scratchDir=scratchDir)
    dmrgArgs = {"threeIdxMode":0}
    eNormal = driver.runNormalMPS(dmrgArgs=dmrgArgs)["energy"]
    eBig = driver.runBigMPS(saveDir=saveDir,dmrgArgs=dmrgArgs)["energy"]
    errMsg = f"{mol.symmetry_subgroup}  eNormal={eNormal} eBig={eBig}"
    # -109.0934767147 # -109.0936128932
    print('eBig  = %20.10f' % eBig)
    print('eNor  = %20.10f' % eNormal)
    if PG == "c2v":
        # test readin; only for one
        driver.noises = [0]
        driver.davTols = [1e-12]
        driver.bondDims = [driver.bondDims[-1]]
        eBig2 = driver.runBigMPS(loadDir=saveDir, dmrgArgs=dmrgArgs)["energy"]
        print('eBig2 = %20.10f' % eBig2)
    del driver


def test_ThawedMode():
    print("-- thawed",flush=True)
    nCas = 8
    nElCas = 10
    nThawed = 2
    atString ='N 0 0 0.0; N 0 0 1.1208; '
    basis = "6-31G"
    scratchDir = './nodex'
    saveDir = './save_mps'

    mol = gto.Mole()
    mol.build(
        atom=atString,
        spin=0,
        basis=basis,
        symmetry=False,
        verbose=0
    )
    driver = BLOCK2_SCI(mol, nThawed, nCas, nElCas, 1300,
                        fciMode=False, doCASSCF=True,
                        verbose=True,
                        memory=int(12e9),
                        scratchDir=scratchDir)
    driver.verbose = True # for dmrg
    eCAS_orca = -109.031573025
    print('eorca = %20.10f' % eCAS_orca)
    print('eScf  = %20.10f' % driver.eScf)
    res = driver.runBigMPS(dmrgArgs = {"lastSite1Site":True})
    energy = res["energy"]
    energy_q = res["energy+q"]
    errMsg = f"{mol.symmetry_subgroup}"
    eRef = -109.106360989 # orca
    eRef_q = -109.108379361
    print('eRef    = %20.10f' % eRef)
    print('energy  = %20.10f' % energy)
    print('eRef_q    = %20.10f' % eRef_q)
    print('energy_q  = %20.10f' % energy_q)
    del driver

def test_mrci_aqcc():
    print("-- aqcc",flush=True)
    nCas = 8
    nElCas = 10
    nThawed = 0
    atString ='N 0 0 0.0; N 0 0 1.1208; '
    basis = "6-31G"
    scratchDir = './nodex'
    saveDir = './save_mps'

    mol = gto.Mole()
    mol.build(
        atom=atString,
        spin=0,
        basis=basis,
        symmetry=False,
        verbose=0
    )
    driver = BLOCK2_SCI(mol, nThawed, nCas, nElCas, 1700, # 1900: -109.1041756105
                        fciMode=False, doCASSCF=True,
                        verbose=False,
                        memory=int(10e9),
                        scratchDir=scratchDir)
    driver.verbose = True # for dmrg
    eCAS_orca = -109.031573025
    print('eorca = %20.10f' % eCAS_orca)
    print('eScf  = %20.10f' % driver.eScf)
    res = driver.runBigMPS(saveDir=saveDir, dmrgArgs = {"lastSite1Site":True})
    energy = res["energy"]
    energy_q = res["energy+q"]
    errMsg = f"{mol.symmetry_subgroup}"
    eRef = -109.104186639
    eRef_q = -109.106124080928
    print('eRef    = %20.10f' % eRef)
    print('energy  = %20.10f' % energy)
    print('eRef_q    = %20.10f' % eRef_q)
    print('energy_q  = %20.10f' % energy_q)
    driver.modDMRGparamsForRestart()
    resAQCC = driver.runBigMPS(loadDir=saveDir,aqcc="AQCC", dmrgArgs = {"lastSite1Site":True})
    energyAQCC = resAQCC["energy"]
    eRefAQCC = -109.105435669024
    print('eRefAQCC    = %20.10f' % eRefAQCC)
    print('energyAQCC  = %20.10f' % energyAQCC)
    del driver

if __name__ == "__main__":
    test_NormalVersusBig('d2h')
    test_NormalVersusBig('c2v')
    test_ThawedMode()
    test_mrci_aqcc()
