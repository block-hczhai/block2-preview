
import contextlib
import numpy as np
import shutil

from .io import MPSTools, MPOTools
from .core import MPS as PYMPS, Tensor, SubTensor

from block2 import SZ, Global
from block2 import init_memory, release_memory, set_mkl_num_threads
from block2 import VectorUInt8, VectorUInt16, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, NoiseTypes
from block2.sz import HamiltonianQC, MPS, MPSInfo, IdentityMPO, Compress
from block2.sz import PDM1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import DMRG, MovingEnvironment, NoTransposeRule

class Hamiltonian:
    """
    Initialization of system, MPS, and MPO.
    """
    def __init__(self, hamil):
        self.hamil = hamil

    @staticmethod
    @contextlib.contextmanager
    def _init(scratch='./my_tmp', rand_seed=1234):
        Random.rand_seed(rand_seed)
        memory = int(1 * 1E9)  # 1G memory
        init_memory(isize=int(memory * 0.1),
                    dsize=int(memory * 0.9), save_dir=scratch)
        set_mkl_num_threads(1)

        yield ()

        release_memory()
        shutil.rmtree(scratch)

    @staticmethod
    @contextlib.contextmanager
    def _from_fcidump(fcidump, pg='d2h'):

        swap_pg = getattr(PointGroup, "swap_" + pg)
        vacuum = SZ(0, 0, 0)
        target = SZ(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))

        n_sites = fcidump.n_sites
        orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
        hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)

        yield Hamiltonian(hamil)

        hamil.deallocate()
        fcidump.deallocate()
    
    @staticmethod
    @contextlib.contextmanager
    def from_fcidump(filename, pg='d2h', scratch='./my_tmp', rand_seed=1234):
        with Hamiltonian._init(scratch, rand_seed) as ():
            fcidump = FCIDUMP()
            fcidump.read(filename)
            with Hamiltonian._from_fcidump(fcidump, pg=pg) as hamil:
                yield hamil
    
    @staticmethod
    @contextlib.contextmanager
    def hubbard(n_sites, u=2, t=1, scratch='./my_tmp', rand_seed=1234):
        with Hamiltonian._init(scratch, rand_seed) as ():
            fcidump = FCIDUMP()
            h1e = np.zeros((n_sites * (n_sites + 1) // 2), dtype=float)
            g2e = np.zeros((n_sites * (n_sites + 1) // 2 * (n_sites * (n_sites + 1) // 2 + 1) // 2), dtype=float)
            ij, kl, ijkl = 0, 0, 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    if abs(i - j) == 1:
                        h1e[ij] = t
                    kl = 0
                    for k in range(0, n_sites):
                        for l in range(0, k + 1):
                            if ij >= kl:
                                if i == j and k == l and i == k:
                                    g2e[ijkl] = u
                                ijkl += 1
                            kl += 1
                    ij += 1

            fcidump.initialize_su2(n_sites, n_sites, 0, 1, 0, h1e, g2e)
            fcidump.orb_sym = VectorUInt8([1] * n_sites)
            with Hamiltonian._from_fcidump(fcidump, pg='c1') as hamil:
                yield hamil

    def get_mpo(self, mode="NC"):
        bmpo = MPOQC(self.hamil, QCTypes.NC if mode == "NC" else QCTypes.CN)
        mpo = MPOTools.from_block2(bmpo)
        bmpo.deallocate()

        return mpo
    
    def get_init_mps(self, bond_dim=250):

        hamil = self.hamil
        vacuum = hamil.vacuum
        target = SZ(hamil.fcidump.n_elec, hamil.fcidump.twos, 0)

        mps_info = MPSInfo(hamil.n_sites, vacuum, target, hamil.basis, hamil.orb_sym)
        mps_info.set_bond_dimension(bond_dim)
        mps = MPS(hamil.n_sites, 0, 1)
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        xmps = MPSTools.from_block2(mps)
        mps_info.deallocate()

        return xmps

    @staticmethod
    def get_determinant_mps(det, orb_sym=None):
        n_sites = len(det) // 2
        tensors = []
        pq = None
        for i in range(n_sites):
            if det[i * 2] == 1 and det[i * 2 + 1] == 1:
                q = SZ(2, 0, 0)
            elif det[i * 2] == 1 and det[i * 2 + 1] == 0:
                q = SZ(1, 1, 0 if orb_sym is None else orb_sym[i])
            elif det[i * 2] == 0 and det[i * 2 + 1] == 1:
                q = SZ(1, -1, 0 if orb_sym is None else orb_sym[i])
            else:
                q = SZ(0, 0, 0)
            if i == 0:
                ql = (q, q)
                rd = np.array([[1.0]])
            elif i == n_sites - 1:
                ql = (pq, q)
                rd = np.array([[1.0]])
            else:
                ql = (pq, q, pq + q)
                rd = np.array([[[1.0]]])
            pq = pq + q if pq is not None else q
            ts = Tensor(blocks=[SubTensor(q_labels=ql, reduced=rd)])
            tensors.append(ts)
        return PYMPS(tensors=tensors)

    def get_ground_state_mps(self, bond_dim=250):
        
        hamil = self.hamil
        vacuum = hamil.vacuum
        target = SZ(hamil.fcidump.n_elec, hamil.fcidump.twos, 0)

        mps_info = MPSInfo(hamil.n_sites, vacuum, target, hamil.basis, hamil.orb_sym)
        mps_info.set_bond_dimension(bond_dim)
        mps = MPS(hamil.n_sites, 0, 2)
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()
        
        mpo = MPOQC(hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True)

        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.init_environments(True)
        dmrg = DMRG(me, VectorUInt16([bond_dim]), VectorDouble([1E-6, 0]))
        energy = dmrg.solve(20, mps.center == 0)

        if mps.center == 0:
            mps.dot = 1
        elif mps.center == mps.n_sites - 2:
            mps.center = mps.n_sites - 1
            mps.dot = 1
        
        xmps = MPSTools.from_block2(mps)

        mpo.deallocate()
        mps_info.deallocate()

        print("Ground State Energy = ", energy)
        return xmps
