
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2021 Seunghoon Lee <seunghoonlee89@gmail.com>
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
stochastic perturbative DMRG

Author: Seunghoon Lee, 2021
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
import time
import numpy as np

# Set spin-adapted or non-spin-adapted here
#SpinLabel = SU2
SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC, CG
    from block2.su2 import MPSInfo, MPS, UnfusedMPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
else:
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC, CG
    from block2.sz import MPSInfo, MPS, UnfusedMPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect
    from block2.sz import StochasticPDMRG

# MPI
from mpi4py import MPI as MPI
comm = MPI.COMM_WORLD
mrank = MPI.COMM_WORLD.Get_rank()
msize = MPI.COMM_WORLD.Get_size()

def _print(*args, **kwargs):
    if mrank == 0:
        print(*args, **kwargs)

class SPDMRGError(Exception):
    pass

class SPDMRG:
    """
    stochastic perturbative DMRG for molecules.
    """

    def __init__(self, scratch='./nodex', fcidump=None, mps_tags=[], verbose=0):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """
        self.fcidump = None 
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None

        self.Edmrg = float(np.load(self.scratch + '/E_dmrg.npy'))

        if len(mps_tags) == 1 and mps_tags[0] == "KET":
            mps_tags=['ZKET', 'ZBRA']
        else:
            assert len(mps_tags) == 2

        mps1_info = MPSInfo(0)
        mps1_info.load_data(self.scratch + '/%s-mps_info.bin'%(mps_tags[0]))
        mps1 = MPS(mps1_info)
        comm.barrier()
        print (mrank)
        if mrank == 0:
            self.change_mps_center(mps1, 0)
        comm.barrier()
        mps1.load_data()
        self.mps_psi0 = UnfusedMPS(mps1)

        mps2_info = MPSInfo(0)
        mps2_info.load_data(self.scratch + '/%s-mps_info.bin'%(mps_tags[1]))
        mps2 = MPS(mps2_info)
        comm.barrier()
        print (mrank)
        if mrank == 0:
            self.change_mps_center(mps2, 0)
        comm.barrier()
        mps2.load_data()
        self.mps_qvpsi0 = UnfusedMPS(mps2)

        self.norm_qvpsi0  = float(np.load(self.scratch + '/cps_overlap.npy'))
        self.norm_qvpsi0  = self.norm_qvpsi0*self.norm_qvpsi0

        self.SPDMRG = StochasticPDMRG(self.mps_psi0, self.mps_qvpsi0, self.norm_qvpsi0) 
        self.n_sites = self.SPDMRG.n_sites
        if fcidump is not None:
            self.fcidump = fcidump
            E_dmrg = float(np.load(scratch + '/E_dmrg.npy'))
            E_cas  = E_dmrg - self.fcidump.const_e 
            dm_e_pqqp = np.load(scratch + '/e_pqqp.npy')
            dm_e_pqpq = np.load(scratch + '/e_pqpq.npy')
            one_pdm = np.load(scratch + "/1pdm.npy")

            E_0 = self.SPDMRG.E0(fcidump, dm_e_pqqp, dm_e_pqpq, one_pdm[0]+one_pdm[1])
            self.fcidump.const_e = - 0.5 * ( E_cas + E_0 ) 

    def change_mps_center(self, ket, center):
        ket.load_data()
        cf = ket.canonical_form
        print(cf)
        cg = CG(200)
        cg.initialize()
        if ket.center == center:
            return
        if center == 0:
            if ket.center == ket.n_sites - 2:
                ket.center += 1
            ket.canonical_form = ket.canonical_form[:-1] + 'S'
            while ket.center != 0:
                ket.move_left(cg, None)
        else:
            ket.canonical_form = 'K' + ket.canonical_form[1:]
            while ket.center != ket.n_sites - 1:
                ket.move_right(cg, None)
            ket.center -= 1
        if self.verbose >= 2:
            _print('CF = %s --> %s' % (cf, ket.canonical_form))
        ket.save_data()

    def init_hamiltonian_fcidump(self, pg, filename):
        """Read integrals from FCIDUMP file.
        pg : point group, pg = 'c1' or 'd2h'
        filename : FCIDUMP filename
        """
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                PointGroup.swap_d2h(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
        assert pg in ["d2h", "c1"]

    def kernel(self, max_samp):
# 1] Importance Sampling of Determinant
# 1-1] C term
        if self.verbose >= 4:
            _print("1] IMPORTANT SAMPLING")
            _print("1-1] C term")
            _print("     sampling D_p with P_p = |<Phi_0|D_p>|^2")
            _print("     & computing <1/(E_d-E_0)>")
        #TODO: canonicalinze mps_psi0 as CRR...R
        Cterm = []
        H00   = 0.0
        H00_2 = 0.0
        H11   = 0.0
        H11_2 = 0.0
        H10   = 0.0
        H10_2 = 0.0

        max_samp_per_rank = max_samp // msize
        for num_samp in range(mrank*max_samp_per_rank, (mrank+1)*max_samp_per_rank):
            # sample | D_p > with P_p = |< Psi_0 | D_p >|^2
            self.SPDMRG.sampling_c()
            # calculate dE_p = < D_p | H_d | D_p > - E0
            dE_p = self.fcidump.det_energy(self.SPDMRG.det_string, 0, self.n_sites) + self.fcidump.const_e
            #print(self.SPDMRG.det_string)
            #print(dE_p, self.fcidump.const_e)
            # sampling 1/(E_p-E_0)
            H00   += 1.0 / (dE_p*float(max_samp))
            H00_2 += 1.0 / (dE_p*dE_p*float(max_samp))
            self.SPDMRG.clear()

        print(mrank, H00)
        if msize != 0:
            comm.barrier()
            H00   = comm.reduce(H00,   op=MPI.SUM, root=0)  
            H00_2 = comm.reduce(H00_2, op=MPI.SUM, root=0)  

        if mrank == 0: 
            print('reduced ', H00 / msize)
            avg_Cterm = H00 / msize 
            std_Cterm = np.sqrt(( H00_2 / msize - avg_Cterm*avg_Cterm)/float(max_samp))

        if mrank == 0 and self.verbose >= 4: 
            _print(" C term = %15.10f (%15.10f)" % (avg_Cterm, std_Cterm))
            _print("")

            _print("1-2] A & B term")
            _print("     sampling D_p with P_p = |<Psi_0| VQ |D_p>|^2")
            _print("     & computing <1/(E_d-E_0)> and <<D_p|Psi_0> / {(E_d-E_0) <D_p|QV|Psi_0>}>")

        Aterm = []
        Bterm = []
        max_samp_per_rank = max_samp // msize
        print_samp = max_samp // 10 
        for num_samp in range(mrank*max_samp_per_rank, (mrank+1)*max_samp_per_rank):
            if num_samp % print_samp ==0:
                _print( '%d processor: sampling %d %% done'%(mrank, (num_samp//print_samp+1)*10) )
            # sample | D_p > with P_p = |< Psi_0 | VQ | D_p >|^2
            self.SPDMRG.sampling_ab()
            # calculate dE_p = < D_p | H_d | D_p > - E0
            dE_p = self.fcidump.det_energy(self.SPDMRG.det_string, 0, self.n_sites) + self.fcidump.const_e
            #print(self.SPDMRG.det_string)
            #print(dE_p, self.fcidump.const_e)
            # sampling 1/(E_p-E_0)
            H11   += self.norm_qvpsi0 / (dE_p*float(max_samp))
            H11_2 += self.norm_qvpsi0*self.norm_qvpsi0 / (dE_p*dE_p*float(max_samp))

            Sqv_p  = self.SPDMRG.overlap_c() 
            S_p    = self.SPDMRG.overlap_ab()
            tmp    = self.norm_qvpsi0*S_p / (Sqv_p*dE_p)
            H10   += tmp / float(max_samp)
            H10_2 += tmp*tmp / float(max_samp)
            self.SPDMRG.clear()

        print(mrank, H11, H10)
        if msize != 0:
            comm.barrier()
            H11   = comm.reduce(H11,   op=MPI.SUM, root=0)
            H11_2 = comm.reduce(H11_2, op=MPI.SUM, root=0)
            H10   = comm.reduce(H10,   op=MPI.SUM, root=0)
            H10_2 = comm.reduce(H10_2, op=MPI.SUM, root=0)

        if mrank == 0: 
            print('reduced ', H11 / msize, H10 / msize)
            avg_Aterm = H11 / msize 
            std_Aterm = np.sqrt(( H11_2 / msize - avg_Aterm*avg_Aterm)/float(max_samp))
            avg_Bterm = H10 / msize 
            std_Bterm = np.sqrt(( H10_2 / msize - avg_Bterm*avg_Bterm)/float(max_samp))
            Emp2 = - avg_Aterm + avg_Bterm**2 / avg_Cterm
            std_Emp2 = std_Aterm + avg_Bterm ** 2 / abs(avg_Cterm) \
                       * ( 2 * std_Bterm / abs(avg_Bterm) + std_Cterm / abs(avg_Cterm) ) 

        if mrank == 0 and self.verbose >= 4: 
            _print("")
            _print("         ===============")
            _print("         === SUMMARY ===")
            _print("         ===============")
            _print("")
            _print(" A   term = %15.10f (%15.10f)" % (avg_Aterm, std_Aterm))
            _print(" B^2 term = %15.10f (%15.10f)" % (avg_Bterm*avg_Bterm, std_Bterm))
            _print(" C   term = %15.10f (%15.10f)" % (avg_Cterm, std_Cterm))
            _print("")
            _print("    DMRG Energy = %15.10f"%(self.Edmrg))
            _print("     MP2 Energy = %15.10f (%15.10f)"%(Emp2, std_Emp2))
            _print(" --------------------------------")
            _print(" sp-DMRG Energy = %15.10f (%15.10f)"%(self.Edmrg+Emp2, std_Emp2))

        return [Emp2, std_Emp2]

    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        release_memory()

if __name__ == "__main__":

    import os
    # parameters
    n_threads = 4
    pg = 'c1'  # point group: d2h or c1
    FCIDUMP='H0'
    scratch = '%s'%os.environ['SCRATCHDIR']
    os.environ['TMPDIR'] = scratch

    memory = 1E7  # 1Gb 
    omp_threads=8
    verbose = 3

    assert SpinLabel == SZ
    _print(("NON-" if SpinLabel == SZ else "") + "SPIN-ADAPTED")

    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
    else:
        raise SPDMRGError("Point group %d not supported yet!" % pg)

    Random.rand_seed(0)
    Global.threading = Threading(
        ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
    Global.threading.seq_type = SeqTypes.Simple

    init_memory(isize=int(memory * 0.1),
                dsize=int(memory * 0.9), save_dir=scratch)

    # init: load MPS for |Psi0> and QV|Psi0> as 3-legs sparse tensors
    SPDMRG = SPDMRG(scratch=scratch, verbose=verbose)
    SPDMRG.init_hamiltonian_fcidump(pg, FCIDUMP)
    nsample=1
    SPDMRG.kernel(nsample)

    del SPDMRG  # IMPORTANT!!! --> to release stack memory
