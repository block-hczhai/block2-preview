
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
Revised: Huanchen Zhai
    Aug 13, 2021 (added openmp)
    Aug 20, 2021 (added su2)
"""

from block2 import VectorUInt8
import time
import numpy as np

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

    def __init__(self, su2, scratch='./nodex', fcidump=None, mps_tags=[], verbose=0, use_threading=True, n_steps=20):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """
        self.fcidump = None 
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.use_threading = use_threading
        self.n_steps = n_steps
        self.bdims = [0, 0]
        self.su2 = su2

        if self.su2:
            from block2.su2 import MPSInfo, MPS, UnfusedMPS, StochasticPDMRG
        else:
            from block2.sz import MPSInfo, MPS, UnfusedMPS, StochasticPDMRG

        self.Edmrg = np.float64(np.load(self.scratch + '/E_dmrg.npy'))

        if len(mps_tags) == 1 and mps_tags[0] == "KET":
            mps_tags=['ZKET', 'ZBRA']
        else:
            assert len(mps_tags) == 2

        mps1_info = MPSInfo(0)
        mps1_info.load_data(self.scratch + '/%s-mps_info.bin'%(mps_tags[0]))
        mps1 = MPS(mps1_info)
        mps1_info.load_mutable()
        max_bdim_l = max([x.n_states_total for x in mps1_info.left_dims])
        max_bdim_r = max([x.n_states_total for x in mps1_info.right_dims])
        self.bdims[0] = max(max_bdim_l, max_bdim_r)
        comm.barrier()
        if mrank == 0:
            self.change_mps_center(mps1, 0)
        comm.barrier()

        mps1.load_data()
        self.mps_psi0 = UnfusedMPS(mps1)

        mps2_info = MPSInfo(0)
        mps2_info.load_data(self.scratch + '/%s-mps_info.bin'%(mps_tags[1]))
        mps2 = MPS(mps2_info)
        mps2_info.load_mutable()
        max_bdim_l = max([x.n_states_total for x in mps2_info.left_dims])
        max_bdim_r = max([x.n_states_total for x in mps2_info.right_dims])
        self.bdims[1] = max(max_bdim_l, max_bdim_r)
        comm.barrier()
        if mrank == 0:
            self.change_mps_center(mps2, 0)
        comm.barrier()
        mps2.load_data()
        self.mps_qvpsi0 = UnfusedMPS(mps2)

        self.norm_qvpsi0  = np.float64(np.load(self.scratch + '/cps_overlap.npy'))
        self.norm_qvpsi0  = self.norm_qvpsi0*self.norm_qvpsi0

        self.SPDMRG = StochasticPDMRG(self.mps_psi0, self.mps_qvpsi0, self.norm_qvpsi0) 
        self.n_sites = self.SPDMRG.n_sites
        if fcidump is not None:
            self.fcidump = fcidump
            E_dmrg = np.float64(np.load(scratch + '/E_dmrg.npy'))
            E_cas  = E_dmrg - self.fcidump.const_e 
            dm_e_pqqp = np.load(scratch + '/e_pqqp.npy')
            dm_e_pqpq = np.load(scratch + '/e_pqpq.npy')
            one_pdm = np.load(scratch + "/1pdm.npy")

            E_0 = self.SPDMRG.energy_zeroth(fcidump, dm_e_pqqp, dm_e_pqpq, one_pdm[0]+one_pdm[1])
            self.fcidump.const_e = - 0.5 * ( E_cas + E_0 ) 

    def change_mps_center(self, ket, center):
        if self.su2:
            from block2.su2 import CG
        else:
            from block2.sz import CG
        ket.load_data()
        cf = ket.canonical_form
        _print('CF = %s CENTER = %d' % (cf, ket.center))
        cg = CG()
        if ket.center == center:
            if ket.canonical_form[0] == 'S':
                ket.move_right(cg, None)
                ket.move_left(cg, None)
        elif center == 0:
            if ket.center == ket.n_sites - 2:
                ket.center += 1
            if ket.canonical_form[-1] == 'C':
                ket.canonical_form = ket.canonical_form[:-1] + 'S'
            while ket.center != 0:
                ket.move_left(cg, None)
        else:
            if ket.canonical_form[0] == 'C':
                ket.canonical_form = 'K' + ket.canonical_form[1:]
            while ket.center != ket.n_sites - 1:
                ket.move_right(cg, None)
            ket.center -= 1
        if self.verbose >= 2:
            _print('--> %s' % (ket.canonical_form))
        ket.save_data()
    
    def compute_correction(self, max_samp_c, max_samp, H00, H00_2, H11, H11_2, H10, H10_2):
        max_samp_c = np.float64(max_samp_c)
        max_samp = np.float64(max_samp)
        avg_Cterm = H00
        std_Cterm = np.sqrt(abs(H00_2 - avg_Cterm * avg_Cterm) / max_samp_c)
        avg_Aterm = H11
        std_Aterm = np.sqrt(abs(H11_2 - avg_Aterm * avg_Aterm) / max_samp)
        avg_Bterm = H10
        std_Bterm = np.sqrt(abs(H10_2 - avg_Bterm * avg_Bterm) / max_samp)
        Emp2 = - avg_Aterm + avg_Bterm ** 2 / avg_Cterm
        with np.errstate(divide='ignore', invalid='ignore'):
            if abs(avg_Bterm) > 1e-10:
                std_Emp2 = std_Aterm + avg_Bterm ** 2 / abs(avg_Cterm) \
                            * (2 * std_Bterm / abs(avg_Bterm) + std_Cterm / abs(avg_Cterm) ) 
            else:
                std_Emp2 = std_Aterm
        if self.verbose >= 4:
            _print(" A   term = %15.10f (%15.10f)" % (avg_Aterm, std_Aterm))
            _print(" B   term = %15.10f (%15.10f)" % (avg_Bterm, std_Bterm))
            _print(" C   term = %15.10f (%15.10f)" % (avg_Cterm, std_Cterm))
        return Emp2, std_Emp2

    def kernel(self, max_samp):
        comm.barrier()
# 1] Importance Sampling of Determinant
# 1-1] C term
        if self.verbose >= 4:
            _print("1] IMPORTANT SAMPLING")
            _print("1-1] C term")
            _print("     sampling D_p with P_p = |<Phi_0|D_p>|^2")
            _print("     & computing <1/(E_d-E_0)>")
        Cterm = []
        H00   = 0.0
        H00_2 = 0.0
        H11   = 0.0
        H11_2 = 0.0
        H10   = 0.0
        H10_2 = 0.0
        tx = time.perf_counter()

        max_samp_per_rank = max_samp // msize
        det_string = VectorUInt8([0] * self.n_sites * 2)
        if msize == mrank - 1:
            n_samp_per_rank = max_samp - mrank * max_samp_per_rank
        else:
            n_samp_per_rank = max_samp_per_rank
        if self.use_threading:
            H00, H00_2 = self.SPDMRG.parallel_sampling(n_samp_per_rank, 0, self.fcidump)
        else:
            for num_samp in range(n_samp_per_rank):
                # sample | D_p > with P_p = |< Psi_0 | D_p >|^2
                self.SPDMRG.sampling(0, det_string)
                # calculate dE_p = < D_p | H_d | D_p > - E0
                dE_p = self.fcidump.det_energy(det_string, 0, self.n_sites) + self.fcidump.const_e
                # sampling 1/(E_p-E_0)
                H00   += 1.0 / (dE_p*np.float64(n_samp_per_rank))
                H00_2 += 1.0 / (dE_p*dE_p*np.float64(n_samp_per_rank))

        if msize != 0:
            comm.barrier()
            H00, H00_2 = [x * np.float64(n_samp_per_rank) for x in [H00, H00_2]]
            H00   = comm.allreduce(H00,   op=MPI.SUM)
            H00_2 = comm.allreduce(H00_2, op=MPI.SUM)
            H00, H00_2 = [x / np.float64(max_samp) for x in [H00, H00_2]]

        if mrank == 0 and self.verbose >= 4:
            _print("1-2] A & B term")
            _print("     sampling D_p with P_p = |<Psi_0| VQ |D_p>|^2")
            _print("     & computing <1/(E_d-E_0)> and <<D_p|Psi_0> / {(E_d-E_0) <D_p|QV|Psi_0>}>")

        Aterm = []
        Bterm = []
        if self.use_threading:
            _print("Sampling | BRA bond dimension = %d | KET bond dimension = %d | Nsample = %d"
                % (self.bdims[1], self.bdims[0], max_samp), flush=True)
            n_steps = 10
            sub_results = [0, 0, 0, 0]
            current_max_samp = 0
            tg = time.perf_counter()
            for i_samp in range(self.n_steps):
                if i_samp == self.n_steps - 1:
                    n_sub_samp = n_samp_per_rank - i_samp * (n_samp_per_rank // self.n_steps)
                else:
                    n_sub_samp = n_samp_per_rank // self.n_steps
                tx = time.perf_counter()
                results = self.SPDMRG.parallel_sampling(n_sub_samp, 1, self.fcidump)
                sub_results = [ra + rb * n_sub_samp for ra, rb in zip(sub_results,results)]
                current_max_samp += n_sub_samp
                fcms = np.float64(current_max_samp)
                part_f, part_err = self.compute_correction(max_samp, current_max_samp,
                    H00, H00_2, sub_results[0] / fcms, sub_results[1] / fcms,
                    sub_results[2] / fcms, sub_results[3] / fcms)
                print('Rank = %3d / %3d Step = %3d / %3d .. Nsample = %10d F = %18.10f Error = %9.2E T = %.2f' %
                    (mrank, msize, i_samp, self.n_steps, n_sub_samp, part_f, part_err, time.perf_counter() - tx), flush=True)
            _print("Time elapsed = %.3f" % (time.perf_counter() - tg))

            H11, H11_2, H10, H10_2 = [rx / n_samp_per_rank for rx in sub_results]
        else:
            if max_samp_per_rank > 10:
                print_samp = max_samp_per_rank // 10 
            else:
                print_samp = max_samp_per_rank  
            for num_samp in range(n_samp_per_rank):
                if num_samp % print_samp ==0:
                    print( 'processor %2d: sampling %2d %% done'%(mrank, (num_samp//print_samp+0)*10) )
                # sample | D_p > with P_p = |< Psi_0 | VQ | D_p >|^2
                Sqv_p = self.SPDMRG.sampling(1, det_string)
                # calculate dE_p = < D_p | H_d | D_p > - E0
                dE_p = self.fcidump.det_energy(det_string, 0, self.n_sites) + self.fcidump.const_e
                # sampling 1/(E_p-E_0)
                H11   += self.norm_qvpsi0 / (dE_p*np.float64(max_samp_per_rank))
                H11_2 += self.norm_qvpsi0*self.norm_qvpsi0 / (dE_p*dE_p*np.float64(max_samp_per_rank))

                S_p    = self.SPDMRG.overlap(1, det_string)
                tmp    = self.norm_qvpsi0*S_p / (Sqv_p*dE_p)
                H10   += tmp / np.float64(max_samp_per_rank)
                H10_2 += tmp*tmp / np.float64(max_samp_per_rank)

        if msize != 0:
            comm.barrier()
            H11, H11_2, H10, H10_2 = [x * np.float64(n_samp_per_rank) for x in [H11, H11_2, H10, H10_2]]
            H11   = comm.allreduce(H11,   op=MPI.SUM)
            H11_2 = comm.allreduce(H11_2, op=MPI.SUM)
            H10   = comm.allreduce(H10,   op=MPI.SUM)
            H10_2 = comm.allreduce(H10_2, op=MPI.SUM)
            H11, H11_2, H10, H10_2 = [x / np.float64(max_samp) for x in [H11, H11_2, H10, H10_2]]

        if mrank == 0: 
            Emp2, std_Emp2 = self.compute_correction(max_samp, max_samp, H00, H00_2, H11, H11_2, H10, H10_2)
        else: 
            Emp2, std_Emp2 = 0.0, 0.0 

        if mrank == 0 and self.verbose >= 4: 
            _print("")
            _print("         ===============")
            _print("         === SUMMARY ===")
            _print("         ===============")
            _print("")
            _print("    DMRG Energy = %15.10f"%(self.Edmrg))
            _print("     MP2 Energy = %15.10f (%15.10f)"%(Emp2, std_Emp2))
            _print(" --------------------------------")
            _print(" sp-DMRG Energy = %15.10f (%15.10f)"%(self.Edmrg+Emp2, std_Emp2))

        return [Emp2, std_Emp2] 
