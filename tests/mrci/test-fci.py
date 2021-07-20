
"""
FCI test block2-mrci
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

def doIt(mol, maxM, nThawed, nCas, extraNoise=False, shuffle=False,
            places=7):
    scratchDir = './nodex'
    saveDir = './save_mps'
    if shuffle:
        def mo_coeff_shuffler(mol,mo_coeff,myhf,mc, mcCI):
            # change irrep ordering for testing py bugs
            sym = mo_coeff.orbsym[mcCI.ncore:]
            idx = np.argsort(-sym)
            idx = [i+mcCI.ncore for i in idx]
            mo_coeff[:,mcCI.ncore:] = mo_coeff[:,idx]
            mo_coeff.orbsym[mcCI.ncore:] = mo_coeff.orbsym[idx]
            return mo_coeff
    else:
        mo_coeff_shuffler=None
    driver = BLOCK2_SCI(mol, nThawed, nCas, mol.nelectron-nThawed*2, maxM,
                        fciMode=True, doCASSCF=False,
                        verbose=False,
                        mo_coeff_shuffler=mo_coeff_shuffler,
                        scratchDir=scratchDir)
    if extraNoise:
        # those small systems can be quite annoying
        for i in range(10):
            driver.noises.insert(0,1e-3)
            driver.davTols.insert(0, 1e-4)
            driver.bondDims.insert(0,50)
    eFCI = driver.runFCI()
    dmrgArgs = {"keepStates":1, "threeIdxMode":0}
    eNormal = driver.runNormalMPS(dmrgArgs=dmrgArgs)["energy"]
    if abs(eNormal -eFCI) > 1e-5:
        # for some symmetries, some initial guesses get stuck in local minima
        eNormal = driver.runNormalMPS(dmrgArgs=dmrgArgs)["energy"]
    eBig = driver.runBigMPS(saveDir=saveDir, dmrgArgs=dmrgArgs)["energy"]
    if abs(eBig-eNormal) > 1e-2:
        # for some symmetries, some initial guesses get stuck in local minima
        eBig = driver.runBigMPS(saveDir=saveDir,dmrgArgs=dmrgArgs)["energy"]
        if abs(eBig-eNormal) > 1e-2:
            eBig = driver.runBigMPS(loadDir=saveDir, dmrgArgs=dmrgArgs)["energy"]
    # subtest are important so that the stuff still runs through in order to fully deallocate everythign!
    assert abs(eFCI - eNormal) < 1E-7
    assert abs(eBig - eNormal) < 1E-7
    print('eFCI  = %20.10f' % eFCI)
    print('eNor  = %20.10f' % eNormal)
    print('eBig  = %20.10f' % eBig)
    # test readin
    driver.noises = [0]
    driver.davTols = [1e-12]
    driver.bondDims = [maxM]
    eBig2 = driver.runBigMPS(loadDir=saveDir, dmrgArgs=dmrgArgs)["energy"]
    print('eBig2 = %20.10f' % eBig2)
    print('-' * 20)
    assert abs(eBig - eBig2) < 1E-7
    del driver

def run_h2(PG):
    print("H2: ",PG)
    mol = gto.Mole()
    mol.build(
        atom="H 0 0 0; H 0 0 1",
        spin=0,
        basis="6-31G**",
        symmetry=False if PG is None else True,
        symmetry_subgroup=PG,
        verbose=0
    )

    # doIt(mol, 500, 0, 2)
    doIt(mol, 500, 1, 2)
    doIt(mol, 500, 0, 2, shuffle=True)
    doIt(mol, 500, 1, 2, shuffle=True)

def run_lih(PG):
    PG = 'c2v'

    print("LiH: ",PG)
    mol = gto.Mole()
    mol.build(
        atom="H 0 0 0; Li 0 0 1.5",
        spin=0,
        basis="6-31G",
        symmetry=False if PG is None else True,
        symmetry_subgroup=PG,
        verbose=0
    )
    places = 7 if PG != "c2v" else 5 # c2v is annoying here
    doIt(mol, 500, 0, 4, extraNoise=True, places=places)  # no RAS mode
    doIt(mol, 500, 1, 4, extraNoise=True, places=places)  # no RAS mode

if __name__ == "__main__":
    run_h2('d2h')
    run_h2('cs')
    run_lih('c2v')
    run_lih('c2')
