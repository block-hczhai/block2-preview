
#  block2: Efficient MPO implementation of quantum chemistry DMRG
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

"""
Initialization of Hamiltonian, MPS, and MPO.
"""

import contextlib
import numpy as np
import shutil

from .io import MPSTools, MPOTools
from .core import MPS as PYMPS, Tensor, SubTensor

from block2 import SZ, Global
from block2 import init_memory, release_memory, set_mkl_num_threads
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, NoiseTypes
from block2.sz import HamiltonianQC, MPS, MPSInfo, AncillaMPSInfo
from block2.sz import PDM1MPOQC, AncillaMPO, SimplifiedMPO, Rule, RuleQC, MPOQC
from block2.sz import DMRG, MovingEnvironment, NoTransposeRule

class HamilTools:
    """
    Initialization of different type of systems.
    """
    def __init__(self, hamil):
        self.hamil = hamil

    @staticmethod
    @contextlib.contextmanager
    def _init(scratch='./my_tmp', rand_seed=1234, memory=int(1E9), n_threads=1):
        Random.rand_seed(rand_seed)
        init_memory(isize=int(memory * 0.1),
                    dsize=int(memory * 0.9), save_dir=scratch)
        import os
        empty_scratch = len(os.listdir(scratch)) == 0
        set_mkl_num_threads(n_threads)

        yield ()

        release_memory()
        if empty_scratch:
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

        yield HamilTools(hamil)

        hamil.deallocate()
        fcidump.deallocate()
    
    @staticmethod
    @contextlib.contextmanager
    def from_fcidump(filename, pg='d2h', **kwargs):
        """
        Read quantum chemistry system from FCIDUMP file.
        """
        with HamilTools._init(**kwargs) as ():
            fcidump = FCIDUMP()
            fcidump.read(filename)
            with HamilTools._from_fcidump(fcidump, pg=pg) as hamil:
                yield hamil
    
    @staticmethod
    @contextlib.contextmanager
    def hchain(n_sites, r=1.8, pg_reorder=True, **kwargs):
        """
        1D Hydrogen chain model. r in bohr.
        """
        if 'scratch' in kwargs:
            import os
            if not os.path.isdir(kwargs['scratch']):
                os.mkdir(kwargs['scratch'])
            os.environ['TMPDIR'] = kwargs['scratch']

        from pyscf import gto, scf, symm, ao2mo

        BOHR = 0.52917721092  # Angstroms
        r = r * BOHR
        mol = gto.M(atom=[['H', (i * r, 0, 0)] for i in range(n_sites)],
            basis='sto6g', verbose=0, symmetry='d2h')
        pg = mol.symmetry.lower()

        # Reorder
        if pg == 'd2h':
            fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
            optimal_reorder = ["Ag", "B1u", "B3u",
                            "B2g", "B2u", "B3g", "B1g", "Au"]
        elif pg == 'c1':
            fcidump_sym = ["A"]
            optimal_reorder = ["A"]
        else:
            assert False
        
        # SCF
        m = scf.RHF(mol)
        m.kernel()
        mo_coeff = m.mo_coeff
        n_ao = mo_coeff.shape[0]
        n_mo = mo_coeff.shape[1]

        orb_sym_str = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeff)
        orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])

        # Sort the orbitals by symmetry for more efficient DMRG
        if pg_reorder:
            idx = np.argsort([optimal_reorder.index(i) for i in orb_sym_str])
            orb_sym = orb_sym[idx]
            mo_coeff = mo_coeff[:, idx]
            ridx = np.argsort(idx)
        else:
            ridx = np.array(list(range(n_mo)), dtype=int)

        h1e = mo_coeff.T @ m.get_hcore() @ mo_coeff
        g2e = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff), n_mo)
        ecore = mol.energy_nuc()
        del m, mol

        with HamilTools._init(**kwargs) as ():
            fcidump = FCIDUMP()
            n_elec = n_sites
            tol = 1E-13
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.flatten().copy()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
            fcidump.initialize_su2(n_sites, n_elec, 0, 1, ecore, mh1e, mg2e)
            fcidump.orb_sym = VectorUInt8(orb_sym)
            with HamilTools._from_fcidump(fcidump, pg=pg) as hamil:
                yield hamil

    @staticmethod
    @contextlib.contextmanager
    def hubbard(n_sites, u=2, t=1, **kwargs):
        """
        1D Hubbard model.
        """
        with HamilTools._init(**kwargs) as ():
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
            with HamilTools._from_fcidump(fcidump, pg='c1') as hamil:
                yield hamil

    def get_mpo(self, mode="NC", mu=0.0, ancilla=False):
        self.hamil.mu = mu
        bmpo = MPOQC(self.hamil, QCTypes.NC if mode == "NC" else QCTypes.CN)
        self.hamil.mu = 0.0
        if ancilla:
            bmpo = AncillaMPO(bmpo)
        mpo = MPOTools.from_block2(bmpo)
        bmpo.deallocate()
        return mpo
    
    @contextlib.contextmanager
    def get_thermal_limit_mps_block2(self, dot=2):
        hamil = self.hamil
        vacuum = hamil.vacuum
        target = SZ(hamil.n_sites * 2, hamil.fcidump.twos, 0)

        mps_info = AncillaMPSInfo(hamil.n_sites, vacuum, target, hamil.basis)
        mps_info.set_thermal_limit()
        mps = MPS(hamil.n_sites * 2, 0, dot)
        mps.initialize(mps_info)
        mps.fill_thermal_limit()
        mps.canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        yield mps
        mps_info.deallocate()
    
    @contextlib.contextmanager
    def get_init_mps_block2(self, bond_dim=250, dot=2):

        hamil = self.hamil
        vacuum = hamil.vacuum
        target = SZ(hamil.fcidump.n_elec, hamil.fcidump.twos, 0)

        mps_info = MPSInfo(hamil.n_sites, vacuum, target, hamil.basis)
        mps_info.set_bond_dimension(bond_dim)
        mps = MPS(hamil.n_sites, 0, dot)
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        yield mps
        mps_info.deallocate()

    def get_init_mps(self, bond_dim=250):
        with self.get_init_mps_block2(bond_dim=bond_dim, dot=1) as mps:
            xmps = MPSTools.from_block2(mps)
        return xmps
    
    def get_thermal_limit_mps(self):
        with self.get_thermal_limit_mps_block2(dot=1) as mps:
            xmps = MPSTools.from_block2(mps)
        return xmps

    @staticmethod
    def get_determinant_mps(det, orb_sym=None):
        """
        MPS from SZ determinant.
        det = [0a, 0b, 1a, 1b, ...] where ia/ib = 0 (empty) or 1 (occupied).
        """
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

    @contextlib.contextmanager
    def get_ground_state_mps_block2(self, bond_dim=250, n_sweeps=20):
        hamil = self.hamil
        with self.get_init_mps_block2(bond_dim=bond_dim, dot=2) as mps:
            mpo = MPOQC(hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(mpo, RuleQC(), True)

            me = MovingEnvironment(mpo, mps, mps, "DMRG")
            me.init_environments(True)
            dmrg = DMRG(me, VectorUBond([bond_dim]), VectorDouble([1E-6, 0]))
            energy = dmrg.solve(n_sweeps, mps.center == 0)

            print("Ground State Energy = ", energy)
            yield mps
            mpo.deallocate()

    def get_ground_state_mps(self, bond_dim=250, n_sweeps=20):
        with self.get_ground_state_mps_block2(bond_dim=bond_dim, n_sweeps=n_sweeps) as mps:
            if mps.center == 0:
                mps.dot = 1
            elif mps.center == mps.n_sites - 2:
                mps.center = mps.n_sites - 1
                mps.dot = 1
            xmps = MPSTools.from_block2(mps)
        return xmps
