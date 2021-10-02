
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2021 Henrik R. Larsson <larsson@caltech.edu>
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

""" Various tools for block2

These need to be initialized first, depending on the used spin.
Example:
    > from block2 import SZ
    > import tools
    > tools.init(SZ)
    > from tools import saveMPStoDir, loadMPSfromDir, changeCanonicalForm # optional
Note: "from tools import ..." must come after the init statement

:author: Henrik R. Larsson
:author: Huanchen Zhai
"""
from typing import Union, Tuple, List
import warnings

import block2
from block2 import SZ, SU2
from block2 import FCIDUMP, TruncationTypes
from block2 import VectorDouble, VectorUInt8, VectorUBond
from block2 import TruncationTypes

import itertools
import numpy as np
import sys
import subprocess, shutil, os
try:
    from pyscf import tools, ao2mo, gto
    hasPySCF = True
except ImportError:
    hasPySCF = False

# Function definitions (set in init)
hasMPI = False
saveMPStoDir = None
loadMPSfromDir = None
changeCanonicalForm = None

def mkDir(folder: str):
    """ mkdir -p folder """
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except FileExistsError: # don't ask...
            pass


def init(SpinLabel: Union[SZ, SU2]):
    """ Initialize this module by setting the function definitions, depending on the used spin symmetry"""
    global hasMPI, saveMPStoDir, loadMPSfromDir, changeCanonicalForm
    if SpinLabel == SZ:
        from block2.sz import MovingEnvironment, DMRG
        from block2.sz import MPS, MPSInfo
        from block2.sz import MPO, IdentityMPO, SimplifiedMPO, RuleQC
        try:
            # parallel
            from block2.sz import MPICommunicator, ParallelMPO, ParallelRuleIdentity
        except ImportError:
            MPICommunicator = ParallelRuleIdentity = None
    else:
        assert SpinLabel == SU2
        from block2.su2 import MovingEnvironment, DMRG
        from block2.su2 import MPS, MPSInfo
        from block2.su2 import MPO, IdentityMPO, SimplifiedMPO, RuleQC
        try:
            # parallel
            from block2.su2 import MPICommunicator, ParallelMPO, ParallelRuleIdentity
        except ImportError:
            MPICommunicator = ParallelRuleIdentity = None

    # function definitions

    def saveMPStoDir(mps:MPS, mpsSaveDir:str, MPI:MPICommunicator=None):
        mps.save_data() # Important! Saves canonical form
        mkDir(mpsSaveDir)
        mps.info.save_data(f"{mpsSaveDir}/mps_info.bin")
        def copyIt(fnam:str):
            if MPI is not None and MPI.rank != 0:
                # ATTENTION: For multi node  calcs, I assume that all nodes have one global scratch dir
                return
            lastName = os.path.split(fnam)[-1]
            fst = f"cp -p {fnam} {mpsSaveDir}/{lastName}"
            # subprocess is favored but sometimes there is a problem due to memory allocation
            try:
                subprocess.call(fst.split())
            except: # May problem due to allocate memory
                print(f"# ATTENTION: saveMPStoDir with command'{fst}' failed!")
                print(f"# Error message: {sys.exc_info()[0]}")
                print(f"# Error message: {sys.exc_info()[1]}")
                print(f"# Try again with shutil")
                try:
                    # vv does not copy metadata 
                    shutil.copyfile(fnam, mpsSaveDir+"/"+lastName)
                except:
                    print(f"\t# ATTENTION: saveMPStoDir with shutil also failed")
                    print(f"\t# Error message: {sys.exc_info()[0]}")
                    print(f"\t# Error message: {sys.exc_info()[1]}")
                    print(f"\t# Try again with syscal")
                    os.system(fst)

        for iSite in range(mps.n_sites+1):
            fnam = mps.info.get_filename(False,iSite)
            copyIt(fnam)
            if MPI is not None:
                MPI.barrier()
            fnam = mps.info.get_filename(True, iSite)
            copyIt(fnam)
            if MPI is not None:
                MPI.barrier()
        for iSite in range(-1,mps.n_sites): # -1 is data
            fnam = mps.get_filename(iSite)
            copyIt(fnam)
            if MPI is not None:
                MPI.barrier()
        return

    def loadMPSfromDir(mps_info: MPSInfo,  mpsSaveDir:str, MPI:MPICommunicator=None) -> MPS:
        """  Load MPS from directory
        :param mps_info: If None, MPSInfo will be read from mpsSaveDir/mps_info.bin
        :param mpsSaveDir: Directory where the MPS has been saved
        :param MPI: MPI class (or None)
        :return: MPS state
        """
        if mps_info is None: # use mps_info.bin
            mps_info = MPSInfo(0)
            mps_info.load_data(f"{mpsSaveDir}/mps_info.bin")
        else:
            # TODO: It would be good to check if mps_info.bin is available
            #       and then compare it again mps_info input to see any mismatch
            pass 
        def copyItRev(fnam: str):
            if MPI is not None and MPI.rank != 0:
                # ATTENTION: For multi node  calcs, I assume that all nodes have one global scratch dir
                return
            lastName = os.path.split(fnam)[-1]
            fst = f"cp -p {mpsSaveDir}/{lastName} {fnam}"
            try:
                subprocess.call(fst.split())
            except: # May problem due to allocate memory, but why??? 
                print(f"# ATTENTION: loadMPSfromDir with command'{fst}' failed!")
                print(f"# Error message: {sys.exc_info()[0]}")
                print(f"# Error message: {sys.exc_info()[1]}")
                print(f"# Try again with shutil")
                try:
                    # vv does not copy metadata -.-
                    shutil.copyfile(mpsSaveDir+"/"+lastName, fnam)
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
        if MPI is not None:
            MPI.barrier()
        mps_info.bond_dim = mps.info.get_max_bond_dimension() # is not initalized
        return mps

    def changeCanonicalForm(mps: MPS,
                            parallelRule:ParallelRuleIdentity = None,
                            keepStates=None
                            ):
        """ Change the canonical form of the MPS *via* a sweep to one of the end sites of the MPS

        Useful for aligning two MPSs to completely identical canonical form.
        ATTENTION: This normalizes the MPO.

        :param mps: MPS to
        :param parallelRule: Instance of ParallelRuleIdentity, otherwise, it will not be normalized when MPI is used
        :param keepStates: Number of additional states to keep, besides maximal bond dimension
        """
        opf = block2.sz.OperatorFunctions(block2.sz.CG())
        mpo = IdentityMPO(mps.info.basis, mps.info.basis, SZ(0,0,0), opf)
        if parallelRule is not None:
            mpo = SimplifiedMPO(mpo, RuleQC(), True, False) # ParallelMPO always assumes simplification
            mpo = ParallelMPO(mpo, parallelRule)
        # vv does not work. tensors[i] != nullptr assertion
        #mps.dot = 1
        if mps.dot == 2 and mps.center == mps.n_sites-1:
            mps.center -= 1
        me = MovingEnvironment(mpo, mps, mps, "bondDimChange")
        me.init_environments(False)
        bondDim = mps.info.get_max_bond_dimension() + 100 # +100 for numerics in 2s
        dmrg = DMRG(me, VectorUBond([bondDim]), VectorDouble([0]))
        if keepStates is not None:
            assert isinstance(keepStates,int)
            dmrg.trunc_type = TruncationTypes.KeepOne * keepStates
        dmrg.cutoff = 0.0
        dmrg.quanta_cutoff = 0.0
        dmrg.forward = mps.center == 0 or mps.center == 1
        dmrg.davidson_soft_max_iter = 0
        dmrg.iprint = 0
        dmrg.solve(1, dmrg.forward, 1e-16)
        mpo.deallocate()
        return

def printDummyFunction(*args, **kwargs):
    """ Does nothing"""
    pass

def getVerbosePrinter(verbose, indent="",flush=False):
    """ Substitute of the print statement. Will be ignored if verbose is False. """

    if verbose:
        if flush:
            def _print(*args, **kwargs):
                kwargs["flush"] = True
                print(indent,*args,**kwargs)
        else:
            def _print(*args, **kwargs):
                print(indent, *args, **kwargs)
    else:
        _print = printDummyFunction
    return _print