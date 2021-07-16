
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
Low-Temperature (LT)-dmrg.
Multi-Target State-Averaged Excited-State Approach;
using pyscf and block2.

Author: Huanchen Zhai, Jun 21, 2020
"""

from block2 import init_memory, release_memory, Threading, ThreadingTypes, Global
from block2 import VectorUInt8, VectorUBond, VectorInt, VectorDouble, PointGroup
from block2 import Random, FCIDUMP, QCTypes, SeqTypes
from block2 import SU2, SZ, get_partition_weights, OpNamesSet, OpNames
import numpy as np
import time

# Set spin-adapted or non-spin-adapted here
# SpinLabel = SU2
SpinLabel = SZ

if SpinLabel == SU2:
    from block2 import VectorSU2 as VectorSL
    from block2.su2 import HamiltonianQC, MultiMPSInfo, MultiMPS
    from block2.su2 import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import MovingEnvironment, Expect, DMRG
else:
    from block2 import VectorSZ as VectorSL
    from block2.sz import HamiltonianQC, MultiMPSInfo, MultiMPS
    from block2.sz import PDM1MPOQC, PDM2MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import MovingEnvironment, Expect, DMRG


class LTDMRGError(Exception):
    pass


class LTDMRG:
    """
    Low-temperature DMRG for molecules.
    """

    def __init__(self, scratch='./nodex', memory=1 * 1E9, omp_threads=2, verbose=2):

        Random.rand_seed(0)
        init_memory(isize=int(memory * 0.1),
                    dsize=int(memory * 0.9), save_dir=scratch)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
        Global.threading.seq_type = SeqTypes.Tasked
        Global.frame.use_main_stack = False
        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.nroots = 10

    def init_hamiltonian_fcidump(self, pg, filename):
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.targets = VectorSL([SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                           PointGroup.swap_d2h(self.fcidump.isym))])
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
        assert pg in ["d2h", "c1"]

    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym, e_core,
                         h1e, g2e, tol=1E-13, save_fcidump=None):
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        if not isinstance(h1e, tuple):
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.flatten()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
            self.fcidump.initialize_su2(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        else:
            assert SpinLabel == SZ
            assert isinstance(h1e, tuple) and len(h1e) == 2
            assert isinstance(g2e, tuple) and len(g2e) == 3
            mh1e_a = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e_b = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e = (mh1e_a, mh1e_b)
            for xmh1e, xh1e in zip(mh1e, h1e):
                k = 0
                for i in range(0, n_sites):
                    for j in range(0, i + 1):
                        assert abs(xh1e[i, j] - xh1e[j, i]) < tol
                        xmh1e[k] = xh1e[i, j]
                        k += 1
                xmh1e[np.abs(xmh1e) < tol] = 0.0
            mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        self.orb_sym = VectorUInt8(map(PointGroup.swap_d2h, orb_sym))

        vacuum = SpinLabel(0)
        self.targets = VectorSL(
            [SpinLabel(n_elec, twos, PointGroup.swap_d2h(isym))])
        self.n_sites = n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        if save_fcidump is not None:
            self.fcidump.orb_sym = VectorUInt8(orb_sym)
            self.fcidump.write(save_fcidump)
        assert pg in ["d2h", "c1"]

    # State-averaged DMRG for multipile targets and multiple roots
    def dmrg(self, mu, bond_dims, noises, n_steps=30, conv_tol=1E-7):

        if self.verbose >= 2:
            print('>>> START DMRG <<<')
        t = time.perf_counter()

        self.hamil.mu = mu

        # MultiMPSInfo
        mps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                self.targets, self.hamil.basis)
        mps_info.tag = "FINAL"
        mps_info.set_bond_dimension(bond_dims[0])

        # MultiMPS
        mps = MultiMPS(self.n_sites, self.n_sites - 2, 2, self.nroots)
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

        # MPO
        tx = time.perf_counter()
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
        if self.verbose >= 2:
            print('MPO time = ', time.perf_counter() - tx)

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        tx = time.perf_counter()
        me.init_environments(self.verbose >= 3)
        if self.verbose >= 2:
            print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_max_iter = 1000 * self.nroots
        dmrg.iprint = self.verbose
        dmrg.solve(n_steps, mps.center == 0)

        self.energies = dmrg.energies[-1]
        self.quanta = dmrg.mps_quanta[-1]
        self.multiplicities = VectorInt(
            [qs[0][0].multiplicity for qs in self.quanta])

        self.bond_dim = bond_dims[-1]
        mps.save_data()
        mpo.deallocate()
        mps_info.deallocate()

        if self.verbose >= 2:
            print('>>> COMPLETE DMRG | Time = %.2f <<<' %
                  (time.perf_counter() - t))

    # particle number correlation
    # return value (SU2):
    #     npc[0, :, :] -> <(N_{i,alpha}+N_{i,beta}) (N_{j,alpha}+N_{j,beta})>
    # return value (SZ):
    #     npc[0, :, :] -> <N_{i,alpha} N_{j,alpha}>
    #     npc[1, :, :] -> <N_{i,alpha}  N_{j,beta}>
    #     npc[2, :, :] -> < N_{i,beta} N_{j,alpha}>
    #     npc[3, :, :] -> < N_{i,beta}  N_{j,beta}>
    def get_one_npc(self, beta, ridx=None):
        if self.verbose >= 2:
            print('>>> START one-npc <<<')
        t = time.perf_counter()

        self.hamil.mu = 0.0
        assert isinstance(beta, float)

        # MultiMPSInfo
        mps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                self.targets, self.hamil.basis)
        mps_info.tag = "FINAL"
        mps_info.load_mutable()

        # MultiMPS (final)
        mps = MultiMPS(mps_info)
        mps.load_data()

        # 1NPC MPO
        pmpo = NPC1MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, Rule())

        dms = []

        for iroot in range(mps.nroots):
            if self.verbose >= 2:
                print('>>> root = %3d / %3d <<<' % (iroot, mps.nroots))

            smps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                     self.targets, self.hamil.basis)
            smps_info.tag = "1NPC"
            for i in range(0, smps_info.n_sites + 1):
                smps_info.left_dims[i] = mps_info.left_dims[i]
                smps_info.right_dims[i] = mps_info.right_dims[i]
            smps_info.save_mutable()

            mps.load_mutable()

            smps = MultiMPS(smps_info)
            smps.n_sites = mps.n_sites
            smps.center = mps.center
            smps.dot = mps.dot
            smps.canonical_form = '' + mps.canonical_form
            smps.tensors = mps.tensors[:]
            smps.wfns = mps.wfns[iroot:iroot+1]
            smps.weights = mps.weights[iroot:iroot+1]
            smps.nroots = 1
            smps.save_mutable()

            mps.deallocate()

            # 1NPC
            pme = MovingEnvironment(pmpo, smps, smps, "1NPC")
            pme.init_environments(self.verbose >= 3)
            expect = Expect(pme, self.bond_dim, self.bond_dim)
            expect.iprint = self.verbose
            expect.solve(True, smps.center == 0)

            if SpinLabel == SU2:
                dmr = expect.get_1npc_spatial(0, self.n_sites)
                dm = np.array(dmr).copy()
            else:
                dmr = expect.get_1npc(0, self.n_sites)
                dm = np.array(dmr).copy()
                dm = dm.reshape((self.n_sites, 2, self.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))

            dmr.deallocate()
            smps_info.deallocate()

            if ridx is not None:
                dm[:, :] = dm[ridx, :][:, ridx]

            if SpinLabel == SU2:
                dm = dm[None, :, :]
            else:
                dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 0, 1],
                                     dm[None, :, :, 1, 0], dm[None, :, :, 1, 1]], axis=0)

            dms.append(dm)

        pmpo.deallocate()
        mps_info.deallocate_mutable()
        mps_info.deallocate()

        pws = get_partition_weights(beta, self.energies, self.multiplicities)
        dm = np.add.reduce([w * dm for w, dm in zip(pws, dms)])

        if self.verbose >= 2:
            print('>>> COMPLETE one-npc | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        return dm

    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, beta, ridx=None):
        if self.verbose >= 2:
            print('>>> START one-pdm <<<')
        t = time.perf_counter()

        self.hamil.mu = 0.0
        assert isinstance(beta, float)

        # MultiMPSInfo
        mps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                self.targets, self.hamil.basis)
        mps_info.tag = "FINAL"
        mps_info.load_mutable()

        # MultiMPS (final)
        mps = MultiMPS(mps_info)
        mps.load_data()

        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        dms = []

        for iroot in range(mps.nroots):
            if self.verbose >= 2:
                print('>>> root = %3d / %3d <<<' % (iroot, mps.nroots))

            smps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                     self.targets, self.hamil.basis)
            smps_info.tag = "1PDM"
            for i in range(0, smps_info.n_sites + 1):
                smps_info.left_dims[i] = mps_info.left_dims[i]
                smps_info.right_dims[i] = mps_info.right_dims[i]
            smps_info.save_mutable()

            mps.load_mutable()

            smps = MultiMPS(smps_info)
            smps.n_sites = mps.n_sites
            smps.center = mps.center
            smps.dot = mps.dot
            smps.canonical_form = '' + mps.canonical_form
            smps.tensors = mps.tensors[:]
            smps.wfns = mps.wfns[iroot:iroot+1]
            smps.weights = mps.weights[iroot:iroot+1]
            smps.nroots = 1
            smps.save_mutable()

            mps.deallocate()

            # 1PDM
            pme = MovingEnvironment(pmpo, smps, smps, "1PDM")
            pme.init_environments(self.verbose >= 3)
            expect = Expect(pme, self.bond_dim, self.bond_dim)
            expect.iprint = self.verbose
            expect.solve(True, smps.center == 0)

            if SpinLabel == SU2:
                dmr = expect.get_1pdm_spatial(self.n_sites)
                dm = np.array(dmr).copy()
            else:
                dmr = expect.get_1pdm(self.n_sites)
                dm = np.array(dmr).copy()
                dm = dm.reshape((self.n_sites, 2, self.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))

            dmr.deallocate()
            smps_info.deallocate()

            if ridx is not None:
                dm[:, :] = dm[ridx, :][:, ridx]

            if SpinLabel == SU2:
                dm = np.concatenate(
                    [dm[None, :, :], dm[None, :, :]], axis=0) / 2
            else:
                dm = np.concatenate(
                    [dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

            dms.append(dm)

        pmpo.deallocate()
        mps_info.deallocate_mutable()
        mps_info.deallocate()

        pws = get_partition_weights(beta, self.energies, self.multiplicities)
        dm = np.add.reduce([w * dm for w, dm in zip(pws, dms)])

        if self.verbose >= 2:
            print('>>> COMPLETE one-pdm | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        return dm

    # two-particle density matrix (SZ only)
    # return value:
    #     pdm[0, i, j, k, l] -> <AD_{i,alpha} AD_{j,alpha} A_{k,alpha} A_{l,alpha}>
    #     pdm[1, i, j, k, l] -> <AD_{i,alpha} AD_{j, beta} A_{k, beta} A_{l,alpha}>
    #     pdm[2, i, j, k, l] -> <AD_{i, beta} AD_{j, beta} A_{k, beta} A_{l, beta}>
    def get_two_pdm(self, beta, ridx=None):
        if self.verbose >= 2:
            print('>>> START two-pdm <<<')
        t = time.perf_counter()

        self.hamil.mu = 0.0

        # only support SZ
        assert SpinLabel == SZ

        assert isinstance(beta, float)

        # MultiMPSInfo
        mps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                self.targets, self.hamil.basis)
        mps_info.tag = "FINAL"
        mps_info.load_mutable()

        # MultiMPS (final)
        mps = MultiMPS(mps_info)
        mps.load_data()

        # 2PDM MPO
        pmpo = PDM2MPOQC(self.hamil, mask=PDM2MPOQC.s_minimal)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        dms = []

        for iroot in range(mps.nroots):
            if self.verbose >= 2:
                print('>>> root = %3d / %3d <<<' % (iroot, mps.nroots))

            smps_info = MultiMPSInfo(self.n_sites, self.hamil.vacuum,
                                     self.targets, self.hamil.basis)
            smps_info.tag = "2PDM"
            for i in range(0, smps_info.n_sites + 1):
                smps_info.left_dims[i] = mps_info.left_dims[i]
                smps_info.right_dims[i] = mps_info.right_dims[i]
            smps_info.save_mutable()

            mps.load_mutable()

            smps = MultiMPS(smps_info)
            smps.n_sites = mps.n_sites
            smps.center = mps.center
            smps.dot = mps.dot
            smps.canonical_form = '' + mps.canonical_form
            smps.tensors = mps.tensors[:]
            smps.wfns = mps.wfns[iroot:iroot+1]
            smps.weights = mps.weights[iroot:iroot+1]
            smps.nroots = 1
            smps.save_mutable()

            mps.deallocate()

            # 2PDM
            pme = MovingEnvironment(pmpo, smps, smps, "2PDM")
            pme.init_environments(self.verbose >= 3)
            expect = Expect(pme, self.bond_dim, self.bond_dim)
            expect.iprint = self.verbose
            expect.solve(True, smps.center == 0)

            dmr = expect.get_2pdm(self.n_sites)
            dm = np.array(dmr, copy=True)
            dm = dm.reshape((self.n_sites, 2, self.n_sites, 2,
                             self.n_sites, 2, self.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 4, 6, 1, 3, 5, 7))

            smps_info.deallocate()

            if ridx is not None:
                dm[:, :, :, :] = dm[ridx, :, :, :] \
                    [:, ridx, :, :][:, :, ridx, :][:, :, :, ridx]

            dm = np.concatenate([dm[None, :, :, :, :, 0, 0, 0, 0], dm[None, :, :, :, :, 0, 1, 1, 0],
                                 dm[None, :, :, :, :, 1, 1, 1, 1]], axis=0)

            dms.append(dm)

        pmpo.deallocate()
        mps_info.deallocate_mutable()
        mps_info.deallocate()

        pws = get_partition_weights(beta, self.energies, self.multiplicities)
        dm = np.add.reduce([w * dm for w, dm in zip(pws, dms)])

        if self.verbose >= 2:
            print('>>> COMPLETE two-pdm | Time = %.2f <<<' %
                  (time.perf_counter() - t))

        return dm

    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        release_memory()


if __name__ == "__main__":

    # parameters
    bond_dims = [250, 250, 500, 500, 500, 500, 750]
    noises = [1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-9, 1E-9, 1E-9, 0]
    sweep_tol = 1E-8
    max_dmrg_steps = 30

    nroots = 30
    beta = 20.0
    mu = -1.0
    n_threads = 8
    hf_type = "RHF"
    mpg = 'd2h'
    pg_reorder = True
    scratch = './nodex'
    scratch = '/scratch/local/hczhai/lt-hchain'

    import os
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    os.environ['TMPDIR'] = scratch

    from pyscf import gto, scf, symm, ao2mo

    # H chain
    N = 8
    BOHR = 0.52917721092  # Angstroms
    R = 1.8 * BOHR
    mol = gto.M(atom=[['H', (i * R, 0, 0)] for i in range(N)],
                basis='sto6g', verbose=0, symmetry=mpg)
    pg = mol.symmetry.lower()
    nelec = mol.nelectron

    # Reorder
    if pg == 'd2h':
        fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
        optimal_reorder = ["Ag", "B1u", "B3u",
                           "B2g", "B2u", "B3g", "B1g", "Au"]
    elif pg == 'c1':
        fcidump_sym = ["A"]
        optimal_reorder = ["A"]
    else:
        raise LTDMRGError("Point group %d not supported yet!" % pg)

    if hf_type == "RHF":
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
        ecore = 0.0

    elif hf_type == "UHF":
        assert SpinLabel == SZ
        # SCF
        m = scf.UHF(mol)
        m.kernel()
        mo_coeff_a, mo_coeff_b = m.mo_coeff[0], m.mo_coeff[1]
        n_ao = mo_coeff_a.shape[0]
        n_mo = mo_coeff_b.shape[1]

        orb_sym_str_a = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeff_a)
        orb_sym_str_b = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeff_b)
        orb_sym_a = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str_a])
        orb_sym_b = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str_b])

        # Sort the orbitals by symmetry for more efficient DMRG
        if pg_reorder:
            idx_a = np.argsort([optimal_reorder.index(i)
                                for i in orb_sym_str_a])
            orb_sym_a = orb_sym_a[idx_a]
            mo_coeff_a = mo_coeff_a[:, idx_a]
            idx_b = np.argsort([optimal_reorder.index(i)
                                for i in orb_sym_str_b])
            orb_sym_b = orb_sym_b[idx_b]
            mo_coeff_b = mo_coeff_b[:, idx_b]
            assert np.allclose(idx_a, idx_b)
            assert np.allclose(orb_sym_a, orb_sym_b)
            orb_sym = orb_sym_a
            ridx = np.argsort(idx_a)
        else:
            orb_sym = orb_sym_a
            ridx = np.array(list(range(n_mo)), dtype=int)

        h1ea = mo_coeff_a.T @ m.get_hcore() @ mo_coeff_a
        h1eb = mo_coeff_b.T @ m.get_hcore() @ mo_coeff_b
        g2eaa = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff_a), n_mo)
        g2ebb = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff_b), n_mo)
        g2eab = ao2mo.kernel(
            mol, [mo_coeff_a, mo_coeff_a, mo_coeff_b, mo_coeff_b])
        h1e = (h1ea, h1eb)
        g2e = (g2eaa, g2ebb, g2eab)
        ecore = mol.energy_nuc()
        ecore = 0.0

    lt = LTDMRG(scratch=scratch, memory=2E9, verbose=2, omp_threads=n_threads)
    lt.init_hamiltonian(pg, n_sites=n_mo, n_elec=nelec, twos=0, isym=1,
                        orb_sym=orb_sym, e_core=ecore, h1e=h1e, g2e=g2e)

    # Set multiple targets
    lt.targets = VectorSL([])
    pgs = list(set(list(lt.orb_sym)))
    for xpg in pgs:
        for na in range(nelec // 2 - 3, nelec // 2 + 4):
            for nb in range(nelec // 2 - 3, nelec // 2 + 4):
                xn = na + nb
                xs2 = na - nb
                if SpinLabel == SU2 and xs2 < 0:
                    continue
                lt.targets.append(SpinLabel(xn, xs2, xpg))
    # Numebr of states
    lt.nroots = nroots
    print("Targets : LEN = ", len(lt.targets), lt.targets)
    print("Nroots = ", lt.nroots)

    lt.dmrg(mu, bond_dims, noises, n_steps=max_dmrg_steps, conv_tol=sweep_tol)

    partition_weights = get_partition_weights(
        beta, lt.energies, lt.multiplicities)

    print("Partition Function Weights (beta = %9.5f, mu = %9.5f):" % (beta, mu))
    for ii, (e, qs, pw) in enumerate(zip(lt.energies, lt.quanta, partition_weights)):
        print("   [State %3d] = %20.15f %20r %10.5g" % (ii, e, qs[0][0], pw))

    print("Multiplicities = ", lt.multiplicities)
    print("Average Energy = %20.15f" % sum([(e + mu * qs[0][0].n) * pw
                                            for e, qs, pw in zip(lt.energies, lt.quanta, partition_weights)]))
    print("Average Particle Number = %20.15f" % sum([qs[0][0].n * pw
                                                     for qs, pw in zip(lt.quanta, partition_weights)]))

    pdm1 = lt.get_one_pdm(beta, ridx)
    npc1 = lt.get_one_npc(beta, ridx)
    pdm2 = lt.get_two_pdm(beta, ridx)

    import pickle
    pickle.dump(pdm1, open('ltpdm1', "wb"))
    pickle.dump(pdm2, open('ltpdm2', "wb"))

    del lt
