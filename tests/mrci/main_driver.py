
"""
Main driver for block2-mrci
Adapted from sciblock2

:author: Henrik R. Larsson
Modified by Huanchen Zhai (JUL 19 2021)
"""

from pyscf import gto, scf, tools
from pyscf import mcscf, fci
from pyscf.mcscf import casci_symm
import numpy as np
import os

from block2 import Random, SZ, PointGroup, FCIDUMP, Global, Threading, ThreadingTypes, SeqTypes
from block2 import VectorDouble, VectorUInt8, VectorUInt16, VectorVectorInt, VectorUBond
from block2 import init_memory, release_memory, set_mkl_num_threads, set_omp_num_threads
from block2.sz import HamiltonianQC, MPSInfo, MRCIMPSInfo, MPS, MPOQC, SimplifiedMPO, RuleQC, Rule
import block2
import time
from block2.sz import SCIFockBigSite, SimplifiedBigSite, HamiltonianQCBigSite
from block2_tools import *


class BLOCK2_SCI:
    def __init__(self, mol: gto.Mole,
                 nThawed: int, nCas: int, nElCas:int,
                 maxM,
                 fciMode=False, # no MRCI-SD but FCI
                 doCASSCF=True,
                 scratchDir=None,
                 verbose=True,memory=int(2e9),
                 dmrgTol = 1e-8,
                 cas_frozen = 0,
                 mo_coeff_shuffler = None,
                 fcidumpTol=1e-12):
        self.toDeallocate = []
        if verbose:
            def _print(*args, **kwargs):
                kwargs["flush"] = True
                print(*args, **kwargs)
        else:
            def _print(*args, **kwargs):
                kwargs["flush"] = True
        randSeed = 1234
        if "OMP_NUM_THREADS" in os.environ:
            n_threads = int(os.environ["OMP_NUM_THREADS"])
        else:
            n_threads = 28
        _print("# used threads=", n_threads)
        Random.rand_seed(randSeed)
        if scratchDir is None:
            scratchDir = './nodex'
        init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratchDir)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, n_threads, n_threads, 1)
        Global.threading.seq_type = SeqTypes.Nothing
        pg = 'c1' if not mol.symmetry else mol.symmetry_subgroup.lower()
        _print("## pg=", pg)
        ##############
        # HF
        ##############
        myhf = scf.RHF(mol)
        eNuc = myhf.energy_nuc()
        myhf.kernel()
        ##############
        # CASSCF
        ##############
        if doCASSCF:
            assert nElCas > 0
            mc = mcscf.CASSCF(myhf, ncas=nCas, nelecas=nElCas, frozen=cas_frozen)
            mc.with_dep4 = True
            mc.max_cycle_micro = 10
            mc.fix_spin_()
            eScf = mc.kernel()[0]
            mo_coeff = mc.mo_coeff
            assert mc.ncore >= nThawed, f"{mc.ncore} vs {nThawed}; nCas={nCas}, nao={mol.nao}"
            nCore = mc.ncore - nThawed
        else:
            mc = None
            mo_coeff = myhf.mo_coeff
            eScf  = myhf.e_tot
            nCore = 0
        ##############
        # Transfer
        ##############
        RAS_MODE = nThawed != 0
        nExt = mol.nao - nCore - nThawed - nCas
        mcCI = mcscf.CASCI(myhf, ncas=mol.nao - nCore, nelecas=nElCas + nThawed*2, ncore=nCore)
        assert sum(mcCI.nelecas) <= mol.nelectron, f"Wrong nThawed={nThawed}? nElCas={nElCas}, mol.nelectron={mol.nelectron}"
        if mo_coeff_shuffler is not None:
            mo_coeff = casci_symm.label_symmetry_(mcCI, mo_coeff)
            mo_coeff = mo_coeff_shuffler(mol, mo_coeff, myhf, mc, mcCI)
        if mol.symmetry:
            mcCI.mo_coeff = casci_symm.label_symmetry_(mcCI, mo_coeff)
        else:
            mcCI.mo_coeff = mo_coeff
        if mol.symmetry:
            orbsym = mcCI.mo_coeff.orbsym
            orbsym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbsym[nCore:]]
        else:
            orbsym = [1] * (mol.nao - nCore)
        _print("# molpro orbsym:", orbsym)
        wfnSym = mcCI.fcisolver.wfnsym
        wfnSym = wfnSym if wfnSym is not None else 0
        _print("# wfnSym:", wfnSym)
        # to fcidump
        h1e, eCore = mcCI.get_h1cas()
        eri = mcCI.get_h2cas()
        nTot = h1e.shape[0]
        assert nTot == mol.nao - nCore
        ##############
        # block2
        ##############
        fcidump = getFCIDUMP(None, mol, wfnSym, h1e, eri, mcCI.nelecas, eCore, orbsym, fcidumpTol)
        self.toDeallocate.append(fcidump)
        del h1e, eri
        swap_pg = getattr(PointGroup, "swap_" + pg)
        vacuum = SZ(0, 0, 0)
        target = SZ(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
        orbsym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
        noises, davTols, bondDims = generate_dmrg_schedule(maxM=maxM,
                                                            startM=min(50,maxM),
                                                            tol=dmrgTol)

        # Do some selfies
        self.verbose = verbose
        self.fciMode = fciMode
        self.dmrgTol = dmrgTol
        self._print = _print
        self.dot = 2
        self.nElec = mcCI.nelecas
        self.noises = noises
        self.davTols = davTols
        self.bondDims = bondDims
        self.orbsym = orbsym
        self.target = target
        self.vacuum = vacuum
        self.fcidump = fcidump
        self.casCI = mcCI
        self.mo_coeff = mo_coeff
        self.eScf = eScf
        self.randSeed = randSeed
        self.memory = memory
        self.RAS_MODE = RAS_MODE
        self.nExt = nExt
        self.nTot = nTot
        self.nCas = nCas
        self.nThawed = nThawed
        self.mol = mol
        self.hf = myhf
        self.mc = mc
        self.pg = pg
        self.eNuc = eNuc
        self.mps = self.mpsSCI = self.hamil = self.hamilSCI = None

    def __del__(self):
        for x in self.toDeallocate[::-1]:
            x.deallocate()
        if Global.frame is not None: # happens in unit test, for whatever reasons
            Global.frame.reset(0)
            release_memory()
    
    def modDMRGparamsForRestart(self):
        n = len(self.bondDims)
        self.bondDims =  [self.bondDims[-1]] * n
        self.noises = [0] * n
        self.davTols = [self.davTols[-1]] * n

    def runFCI(self):
        mol = self.mol
        assert mol.nelectron <= 8 and mol.nao < 20
        cisolver = fci.FCI(self.mol, self.hf.mo_coeff)
        return cisolver.kernel()[0]

    def runNormalMPS(self, saveDir=None, loadDir=None, dmrgArgs={}):
        self._print("- Do normal MPS calc -")
        if self.RAS_MODE and not self.fciMode:
            raise NotImplementedError
        hamil = HamiltonianQC(self.vacuum, self.fcidump.n_sites, self.orbsym, self.fcidump)
        hamil.opf.seq.mode = block2.SeqTypes.Simple
        self.toDeallocate.append(hamil)

        runtime = time.time()

        if self.fciMode:
            mps_info = MPSInfo(hamil.n_sites, self.vacuum, self.target, hamil.basis)
        else:
            mps_info = MRCIMPSInfo(hamil.n_sites, self.nExt, 2, self.vacuum, self.target, hamil.basis)
        if loadDir is not None:
            self._print("# load MPS from ",loadDir)
            mps = loadMPSfromDir(mps_info, loadDir)
            mps.dot = self.dot # TODO: is this save?
        else:
            mps = MPS(hamil.n_sites, 0, self.dot)
            mps_info.set_bond_dimension(self.bondDims[0])
            mps.initialize(mps_info)
            mps.random_canonicalize()
        self._print("# Canonical form=", mps.canonical_form)
        mps.save_mutable()
        mps.save_data()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        energy, discWeight, bondDim, dmrg = doDMRGopt(mps, hamil, self.noises, self.davTols, self.bondDims, self.dmrgTol,
                                                      verbose=self.verbose, **dmrgArgs)
        runtime = time.time() - runtime
        if saveDir is not None:
            self._print("# save MPS to ",saveDir)
            saveMPStoDir(mps, saveDir)
        res = {"energy":energy, "discWeight":discWeight, "bondDim":bondDim, "runtime":runtime}
        return res

    def runBigMPS(self, saveDir=None, loadDir=None,
                  aqcc :str=None, dmrgArgs={}):
        self._print("- Do Big Site MPS calc -")
        if self.RAS_MODE:
            if self.fciMode:
                nA, nB = self.nElec
                sciWrapLeft = SCIFockBigSite(self.nTot, self.nThawed, False, self.fcidump, self.orbsym,
                                         min(self.nThawed, nA), min(self.nThawed, nB), min(2 * self.nThawed, nA + nB),
                                         verbose=self.verbose)
            else:
                # build determinants
                nThawed = self.nThawed
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
                self._print("# RAS space:", len(RAS_space))
                RAS_space = [block2.VectorInt(R) for R in RAS_space]  # vacuum
                RAS_space = block2.VectorVectorInt(RAS_space)
                sciWrapLeft = SCIFockBigSite(self.nTot, self.nThawed, False, self.fcidump, self.orbsym, RAS_space,
                                         verbose=self.verbose)
            sciWrapLeft = SimplifiedBigSite(sciWrapLeft, RuleQC())
        else:
            sciWrapLeft = None
        if self.fciMode:
            nA,nB = self.nElec
            sciWrapRight = SCIFockBigSite(self.nTot, self.nExt, True, self.fcidump, self.orbsym,
                                      min(self.nExt,nA), min(self.nExt,nB), min(2*self.nExt,nA+nB),
                                      verbose = self.verbose)
        else:
            sciWrapRight = SCIFockBigSite(self.nTot, self.nExt, True, self.fcidump, self.orbsym, 2, 2, 2,
                                      verbose = self.verbose)
        sciWrapRight = SimplifiedBigSite(sciWrapRight, RuleQC())
        hamil = HamiltonianQCBigSite(self.vacuum, self.fcidump.n_sites, self.orbsym, self.fcidump,
                                 sciWrapLeft, sciWrapRight)
        hamil.opf.seq.mode = block2.SeqTypes.Simple

        runtime = time.time()

        mps_info = MPSInfo(hamil.n_sites, self.vacuum, self.target, hamil.basis)
        if loadDir is not None:
            self._print("# load MPS from ",loadDir)
            mps = loadMPSfromDir(mps_info, loadDir)
            mps.dot = self.dot # TODO: is this save?
        else:
            mps = MPS(hamil.n_sites, 0, self.dot)
            mps_info.set_bond_dimension(self.bondDims[0])
            mps.initialize(mps_info)
            mps.random_canonicalize()
        self._print("# Canonical form=", mps.canonical_form)
        mps.save_mutable()
        mps.save_data()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()
        if aqcc is not None:
            nElCas = self.fcidump.n_elec - 2 * self.nThawed
            aqcc = AQCC_Params(nEl=nElCas,
                                     QNs = get_QNs(sciWrapRight)[1:],
                                     eRef=self.eScf,
                                     mode=aqcc, old=True)
            self._print("# AQCCRef G=:", aqcc.G)
            self._print("# AQCCRef QN:", get_QNs(sciWrapRight)[0])
            self._print("# AQCCRef CASSCF:", self.eScf)


        energy, discWeight, bondDim, dmrg = doDMRGopt(mps, hamil, self.noises, self.davTols, self.bondDims,
                                                      self.dmrgTol,
                                                      verbose=self.verbose,aqcc=aqcc, **dmrgArgs)
        runtime = time.time() - runtime
        if saveDir is not None:
            self._print("# save MPS to ",saveDir)
            saveMPStoDir(mps, saveDir)
        res = {"energy":energy, "discWeight":discWeight, "bondDim":bondDim, "runtime":runtime}
        if not self.fciMode and aqcc is None:
            # Davidson correction
            corrE = energy - self.eScf
            if self.RAS_MODE:
                qnsLeft = [len(get_QNs(sciWrapLeft)) - 1]
            else:
                qnsLeft = None
            davCorrs = getDavidsonCorrection(mps, corrE, self.fcidump.n_elec, [0], qnsLeft)
            weightRef = davCorrs["weightRef"]
            otherWeight = davCorrs["otherWeight"]
            davCorrection = davCorrs["davCorr"]
            for correction in "davCorr davCorrNormalized meissnerCorr popleCorr".split():
                corr = davCorrs[correction]
                self._print(f"# {correction:20s} correction: {corr:20.16f}  new energy: {energy+ corr:20.16f}")
            self._print("weight of CAS reference:", weightRef, "other:", otherWeight, "sum=1?", otherWeight + weightRef)
            self._print("# Davidson correction:", davCorrection, "new Energy:", energy + davCorrection)
            res["davCorrs"] = davCorrs
            res["energy+q"] = energy + davCorrs["davCorr"]
        hamil.deallocate()
        return res
