
#
# FT-DMRG using pyscf and block2
#
# Author: Huanchen Zhai, May 14, 2020
#

import sys
sys.path[:0] = ["./build"]

import numpy as np
import time
from block2 import init_memory, release_memory, set_mkl_num_threads
from block2 import VectorUInt8, VectorUInt16, VectorDouble
from block2 import Random, FCIDUMP, Hamiltonian, SpinLabel
from block2 import AncillaMPSInfo, MPS
from block2 import IdentityMPO, AncillaMPO, PDM1MPOQCSU2, SimplifiedMPO, Rule, RuleQCSU2
from block2 import MPOQCSU2, QCTypes, SeqTypes
from block2 import MovingEnvironment, Compress, ImaginaryTE, TETypes, Expect

class FTDMRGError(Exception):
    pass

class FTDMRG:
    """
    Finite-temperature DMRG for molecules.
    """

    def __init__(self, su2=True, scratch='./nodex', memory=4 * 1E9, omp_threads=8, verbose=2):
        
        assert su2
        Random.rand_seed(0)
        init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
        set_mkl_num_threads(omp_threads)
        self.fcidump = None
        self.hamil = None
        self.verbose = verbose

    def init_hamiltonian_fcidump(self, pg, filename):
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        self.orb_sym = VectorUInt8(map(Hamiltonian.swap_d2h, self.fcidump.orb_sym))
        n_elec = self.fcidump.n_sites * 2
        
        vaccum = SpinLabel(0)
        target = SpinLabel(n_elec, self.fcidump.twos, Hamiltonian.swap_d2h(self.fcidump.isym))
        self.n_physical_sites = self.fcidump.n_sites
        self.n_sites = self.fcidump.n_sites * 2

        self.hamil = Hamiltonian(vaccum, target, self.n_physical_sites, True, self.fcidump, self.orb_sym)
        self.hamil.opf.seq.mode = SeqTypes.Simple
        assert pg in ["d2h", "c1"]
        
    def init_hamiltonian_su2(self, pg, n_sites, twos, isym, orb_sym, e_core, h1e, g2e, tol=1E-13):
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        n_elec = n_sites * 2
        mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
        k = 0
        for i in range(0, n_sites):
            for j in range(0, i + 1):
                assert abs(h1e[i, j] - h1e[j, i]) < 1E-10
                mh1e[k] = h1e[i, j]
                k += 1
        mg2e = g2e.flatten().copy()
        mh1e[np.abs(mh1e) < tol] = 0.0
        mg2e[np.abs(mg2e) < tol] = 0.0
        self.fcidump.initialize_su2(n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        self.orb_sym = VectorUInt8(map(Hamiltonian.swap_d2h, orb_sym))

        vaccum = SpinLabel(0)
        target = SpinLabel(n_elec, twos, Hamiltonian.swap_d2h(isym))
        self.n_physical_sites = n_sites
        self.n_sites = n_sites * 2

        self.hamil = Hamiltonian(vaccum, target, self.n_physical_sites, True, self.fcidump, self.orb_sym)
        self.hamil.opf.seq.mode = SeqTypes.Simple

        k = 0
        for i in range(0, n_sites):
            for j in range(0, i + 1):
                assert self.hamil.t(i, j) == mh1e[k]
                k += 1
        ij = 0
        ijkl = 0
        for i in range(n_sites):
            for j in range(0, i + 1):
                kl = 0
                for k in range(0, i + 1):
                    for l in range(0, k + 1):
                        if ij >= kl:
                            assert self.hamil.v(i, j, k, l) == mg2e[ijkl]
                            ijkl += 1
                        kl += 1
                ij += 1
        assert pg in ["d2h", "c1"]
    
    def generate_initial_mps(self, bond_dim):
        if self.verbose >= 2:
            print('>>> START generate initial mps <<<')
        t = time.perf_counter()
        assert self.hamil is not None

        # Ancilla MPSInfo (thermal)
        mps_info_thermal = AncillaMPSInfo(self.n_physical_sites, self.hamil.vaccum, self.hamil.target, self.hamil.basis, self.hamil.orb_sym, self.hamil.n_syms)
        mps_info_thermal.set_thermal_limit()
        mps_info_thermal.tag = "THERMAL"
        mps_info_thermal.save_mutable()
        mps_info_thermal.deallocate_mutable()

        # Ancilla MPSInfo (initial)
        mps_info = AncillaMPSInfo(self.n_physical_sites, self.hamil.vaccum, self.hamil.target, self.hamil.basis, self.hamil.orb_sym, self.hamil.n_syms)
        mps_info.set_bond_dimension(bond_dim)
        mps_info.tag = "INIT"
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        if self.verbose >= 2:
            print("left dims = ", [p.n_states_total for p in mps_info.left_dims])
            print("right dims = ", [p.n_states_total for p in mps_info.right_dims])

        # Ancilla MPS (thermal)
        mps_thermal = MPS(self.n_sites, self.n_sites - 2, 2)
        mps_info_thermal.load_mutable()
        mps_thermal.initialize(mps_info_thermal)
        mps_thermal.fill_thermal_limit()
        mps_thermal.canonicalize()

        # Ancilla MPS (initial)
        mps = MPS(self.n_sites, self.n_sites - 2, 2)
        mps_info.load_mutable()
        mps.initialize(mps_info)
        mps.random_canonicalize()

        # MPS/MPSInfo save mutable
        mps.save_mutable()
        mps.deallocate()
        mps_info.deallocate_mutable()
        mps_thermal.save_mutable()
        mps_thermal.deallocate()
        mps_info_thermal.deallocate_mutable()

        # Identity MPO
        impo = IdentityMPO(self.hamil)
        impo = AncillaMPO(impo)
        impo = SimplifiedMPO(impo, Rule())

        # ME
        ime = MovingEnvironment(impo, mps, mps_thermal, "COMPRESS")
        ime.init_environments()

        # Compress
        cps = Compress(ime, VectorUInt16([bond_dim]), VectorUInt16([10]), VectorDouble([0.0]))
        cps.solve(30, False)

        mps.save_data()
        impo.deallocate()
        mps_info.deallocate()
        mps_info_thermal.deallocate()

        if self.verbose >= 2:
            print('>>> COMPLETE generate initial mps | Time = %.2f <<<' % (time.perf_counter() - t))
    
    def imaginary_time_evolution(self, beta, beta_step, mu, bond_dims):
        if self.verbose >= 2:
            print('>>> START imaginary time evolution <<<')
        t = time.perf_counter()

        self.hamil.mu = mu

        # Ancilla MPSInfo (initial)
        mps_info = AncillaMPSInfo(self.n_physical_sites, self.hamil.vaccum, self.hamil.target, self.hamil.basis, self.hamil.orb_sym, self.hamil.n_syms)
        mps_info.tag = "INIT"
        mps_info.load_mutable()

        # Ancilla MPS (initial)
        mps = MPS(mps_info)
        mps.load_data()
        mps.load_mutable()

        # MPS/MPSInfo save mutable
        mps_info.tag = "FINAL"
        mps_info.save_mutable()
        mps.save_mutable()
        mps.deallocate()
        mps_info.deallocate_mutable()

        # MPO
        mpo = MPOQCSU2(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(AncillaMPO(mpo), RuleQCSU2())

        # TE
        me = MovingEnvironment(mpo, mps, mps, "TE")
        me.init_environments()
        te = ImaginaryTE(me, VectorUInt16(bond_dims), TETypes.RK4)
        n_steps = int(round(beta / beta_step) + 0.1)
        te.solve(n_steps, beta_step, mps.center == 0)

        self.bond_dim = bond_dims[-1]
        mps.save_data()
        mpo.deallocate()
        mps_info.deallocate()

        if self.verbose >= 2:
            print('>>> COMPLETE imaginary time evolution | Time = %.2f <<<' % (time.perf_counter() - t))

    def get_one_pdm(self, ridx=None):
        if self.verbose >= 2:
            print('>>> START one-pdm <<<')
        t = time.perf_counter()
        
        self.hamil.mu = 0.0

        # Ancilla MPSInfo (initial)
        mps_info = AncillaMPSInfo(self.n_physical_sites, self.hamil.vaccum, self.hamil.target, self.hamil.basis, self.hamil.orb_sym, self.hamil.n_syms)
        mps_info.tag = "FINAL"

        # Ancilla MPS (initial)
        mps = MPS(mps_info)
        mps.load_data()

        # 1PDM MPO
        pmpo = PDM1MPOQCSU2(self.hamil)
        pmpo = AncillaMPO(pmpo, True)
        pmpo = SimplifiedMPO(pmpo, Rule())

        # 1PDM
        pme = MovingEnvironment(pmpo, mps, mps, "1PDM")
        pme.init_environments()
        expect = Expect(pme, self.bond_dim, self.bond_dim)
        expect.solve(True, mps.center == 0)
        dmr = expect.get_1pdm_spatial(self.n_physical_sites)
        dm = np.array(dmr).copy()

        if ridx is not None:
            dm[:, :] = dm[ridx, :][:, ridx]

        dmr.deallocate()
        pmpo.deallocate()
        mps_info.deallocate()
        
        if self.verbose >= 2:
            print('>>> COMPLETE one-pdm | Time = %.2f <<<' % (time.perf_counter() - t))

        return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
    
    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        release_memory()

if __name__ == "__main__":

    # parameters
    bond_dim = 300
    beta = 0.01
    beta_step = 0.0002
    mu = -1.0
    bond_dims = [bond_dim]
    scratch = './nodex'

    import os
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf, symm, ao2mo

    # H chain
    N = 8
    BOHR = 0.52917721092  # Angstroms
    R = 1.8 * BOHR
    mol = gto.M(atom = [['H', (i * R, 0, 0)] for i in range(N)],
        basis='sto6g', verbose=0, symmetry='D2h')
    
    # SCF
    pg = mol.symmetry.lower()
    m = scf.RHF(mol)
    m.kernel()
    mo_coeff = m.mo_coeff
    n_ao = mo_coeff.shape[0]
    n_mo = mo_coeff.shape[1]

    # Reorder
    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
        optimal_reorder = ["Ag", "B1u", "B3u", "B2g", "B2u", "B3g", "B1g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
        optimal_reorder = ["A"]
    else:
        raise FTDMRGError("Point group %d not supported yet!" % pg)
    
    orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])

    pg_reorder = True

    # Sort the orbitals by symmetry for more efficient DMRG
    if pg_reorder:
        idx = np.argsort([optimal_reorder.index(i) for i in orb_sym_str])
        orb_sym = orb_sym[idx]
        mo_coeff = mo_coeff[:, idx]
        ridx = np.argsort(idx)
    else:
        ridx = np.array(list(range(n_mo), dtype=int))
    
    h1e = mo_coeff.T @ m.get_hcore() @ mo_coeff
    g2e = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff), n_mo)
    ecore = mol.energy_nuc()
    ecore = 0.0

    ft = FTDMRG(scratch=scratch)
    ft.init_hamiltonian_su2(pg, n_sites=n_mo, twos=0, isym=1, orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e)
    ft.generate_initial_mps(bond_dim)
    ft.imaginary_time_evolution(beta, beta_step, mu, bond_dims)
    print(ft.get_one_pdm(ridx))
