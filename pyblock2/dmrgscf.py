
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

import block2
from block2 import *
from block2.su2 import *
from pyscf import lib, ao2mo, gto, mcscf, tools
from pyscf.mcscf import casci_symm
import numpy as np

class DMRGCI(lib.StreamObject):
    """DMRG FCI solver."""

    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.e_tot = None
        self.dmrg_args = {
            "startM": 250, "maxM": 500, "schedule": "default",
            "sweep_tol": 1E-6, "cutoff": 1E-14
        }
    
    @staticmethod
    def get_schedule(kwargs):
        start_m = int(kwargs.get("startM", 250))
        max_m = int(kwargs.get("maxM", 0))
        if max_m <= 0:
            raise ValueError("A positive maxM must be set for default schedule, "
                            + "current value : %d" % max_m)
        elif max_m < start_m:
            raise ValueError(
                "maxM %d cannot be smaller than startM %d" % (max_m, start_m))
        sch_type = kwargs.get("schedule", "")
        sweep_tol = float(kwargs.get("sweep_tol", 0))
        if sweep_tol == 0:
            kwargs["sweep_tol"] = "1E-5"
            sweep_tol = 1E-5
        schedule = []
        if sch_type == "default":
            def_m = [50, 100, 250, 500] + [1000 * x for x in range(1, 11)]
            def_iter = [8] * 5 + [4] * 9
            def_noise = [1E-3] * 3 + [1E-4] * 2 + [5E-5] * 9
            def_tol = [1E-4] * 3 + [1E-5] * 2 + [5E-6] * 9
            if start_m == max_m:
                schedule.append([0, start_m, 1E-5, 1E-4])
                schedule.append([8, start_m, 5E-6, 5E-5])
            elif start_m < def_m[0]:
                def_m.insert(0, start_m)
                for x in [def_iter, def_noise, def_tol]:
                    x.insert(0, x[0])
            elif start_m > def_m[-1]:
                while start_m > def_m[-1]:
                    def_m.append(def_m[-1] + 1000)
                    for x in [def_iter, def_noise, def_tol]:
                        x.append(x[-1])
            else:
                for i in range(1, len(def_m)):
                    if start_m < def_m[i]:
                        def_m[i - 1] = start_m
                        break
            isweep = 0
            for i in range(len(def_m)):
                if def_m[i] >= max_m:
                    schedule.append([isweep, max_m, def_tol[i], def_noise[i]])
                    isweep += def_iter[i]
                    break
                elif def_m[i] >= start_m:
                    schedule.append([isweep, def_m[i], def_tol[i], def_noise[i]])
                    isweep += def_iter[i]
            schedule.append([schedule[-1][0] + 8, max_m, sweep_tol / 10, 0.0])
            last_iter = schedule[-1][0]
            if "twodot" not in kwargs and "onedot" not in kwargs \
                and "twodot_to_onedot" not in kwargs:
                kwargs["twodot_to_onedot"] = str(last_iter + 2)
            max_iter = int(kwargs.get("maxiter", 0))
            if max_iter <= schedule[-1][0]:
                kwargs["maxiter"] = str(last_iter + 4)
                max_iter = last_iter + 4
        else:
            raise ValueError("Unrecognized schedule type (%s)" % sch_type)
        
        tmp = list(zip(*schedule))
        nsweeps = np.diff(tmp[0]).tolist()
        maxiter = int(kwargs.get("maxiter", 1)) - int(np.sum(nsweeps))
        assert maxiter > 0
        nsweeps.append(maxiter)

        schedule = [[], [], []]
        for nswp, M, tol, noise in zip(nsweeps, *tmp[1:]):
            schedule[0].extend([M] * nswp)
            schedule[1].extend([tol] * nswp)
            schedule[2].extend([noise] * nswp)
        kwargs["schedule"] = schedule
        return schedule
    
    def kernel(self, h1e, g2e, norb, nelec, ecore=0, ci0=None, **kwargs):

        # Global
        Random.rand_seed(123456)
        scratch = './nodex'
        n_threads = lib.num_threads()
        fcidump_tol = 1E-13
        memory = 50E9
        init_memory(isize=int(memory * 0.05),
            dsize=int(memory * 0.95), save_dir=scratch)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
            n_threads, n_threads, 1)
        Global.threading.seq_type = SeqTypes.Tasked
        Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
        Global.frame.load_buffering = False
        Global.frame.save_buffering = False
        Global.frame.use_main_stack = False
        Global.frame.minimal_disk_usage = True
        if self.verbose >= 6:
            print(Global.frame)
            print(Global.threading)

        if 'orbsym' in kwargs:
            orb_sym = kwargs['orbsym']
        else:
            orb_sym = [0] * norb

        n_orbs = norb
        self.n_orbs = n_orbs
        mp_orb_sym = [tools.fcidump.ORBSYM_MAP[self.mol.groupname][i] for i in orb_sym]
        h1e = h1e.flatten()
        g2e = ao2mo.restore(8, g2e, n_orbs)
        h1e[np.abs(h1e) < fcidump_tol] = 0
        g2e[np.abs(g2e) < fcidump_tol] = 0
        na, nb = nelec
        fcidump = FCIDUMP()
        fcidump.initialize_su2(n_orbs, na + nb, na - nb, 1, ecore, h1e, g2e)
        fcidump.orb_sym = VectorUInt8(mp_orb_sym)
        error = fcidump.symmetrize(VectorUInt8(orb_sym))
        if self.verbose >= 6:
            print('symm error = ', error)

        if self.dmrg_args["schedule"] == "default":
            DMRGCI.get_schedule(self.dmrg_args)
        
        bond_dims, dav_thrds, noises = self.dmrg_args["schedule"]
        sweep_tol = self.dmrg_args["sweep_tol"]
        
        # Hamiltonian
        vacuum = SU2(0)
        target = SU2(na + nb, abs(na - nb), 0)
        hamil = HamiltonianQC(vacuum, fcidump.n_sites, VectorUInt8(orb_sym), fcidump)
        n_sites = hamil.n_sites

        self.hamil = hamil

        # MPS
        info = MPSInfo(n_sites, vacuum, target, hamil.basis)
        info.set_bond_dimension(bond_dims[0])
        mps = MPS(n_sites, 0, 2)
        mps.initialize(info)
        mps.random_canonicalize()
        mps.tensors[mps.center].normalize()
        mps.save_mutable()
        info.save_mutable()
        forward = mps.center == 0

        # MPO
        mpo = MPOQC(hamil, QCTypes.NC)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(self.verbose >= 5)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
        dmrg.cutoff = self.dmrg_args["cutoff"]
        dmrg.iprint = max(min(self.verbose - 4, 3), 0)
        dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
        if "twodot_to_onedot" not in self.dmrg_args:
            self.e_tot = dmrg.solve(len(bond_dims), forward, sweep_tol)
        else:
            tto = int(self.dmrg_args["twodot_to_onedot"])
            assert len(bond_dims) > tto
            dmrg.solve(tto, forward, 0)
            dmrg.bond_dims = VectorUBond(bond_dims[tto:])
            dmrg.noises = VectorDouble(noises[tto:])
            dmrg.davidson_conv_thrds = VectorDouble(dav_thrds[tto:])
            dmrg.me.dot = 1
            self.e_tot = dmrg.solve(len(bond_dims) - tto, mps.center == 0, sweep_tol)
            mps.dot = 1
            mps.save_data()

        self.mps = mps
        self.converged = True
        return self.e_tot, mps

    def make_rdm1(self, state=None, norb=None, nelec=None):
        '''
        Spin-traced one-particle density matrix in MO basis.
        dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>
        '''

        if state is None:
            state = self.mps
            state.dot = 2
        
        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil, 0)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        # 1PDM
        pme = MovingEnvironment(pmpo, state, state, "1PDM")
        pme.init_environments(self.verbose >= 5)
        expect = Expect(pme, state.info.bond_dim, state.info.bond_dim)
        expect.iprint = max(min(self.verbose - 4, 3), 0)
        expect.solve(True, state.center == 0)
        dmr = expect.get_1pdm_spatial(self.n_orbs)
        dm = np.array(dmr).copy()
        dmr.deallocate()

        return dm.transpose((1, 0))

    def make_rdm2(self, state=None, norb=None, nelec=None):
        '''
        Spin-traced two-particle density matrix in MO basis
        dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>
        '''

        if state is None:
            state = self.mps
            state.dot = 2
        
        # 2PDM MPO
        pmpo = PDM2MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())

        # 2PDM
        pme = MovingEnvironment(pmpo, state, state, "2PDM")
        pme.init_environments(self.verbose >= 5)
        expect = Expect(pme, state.info.bond_dim, state.info.bond_dim)
        expect.iprint = max(min(self.verbose - 4, 3), 0)
        expect.solve(True, state.center == 0)
        dmr = expect.get_2pdm_spatial(self.n_orbs)
        dm = np.array(dmr, copy=True)

        return dm.transpose((0, 3, 1, 2))
    
    def make_rdm12(self, state=None, norb=None, nelec=None):
        dm1 = self.make_rdm1(state, norb, nelec)
        dm2 = self.make_rdm2(state, norb, nelec)
        return dm1, dm2


class DMRGCASCI(lib.StreamObject):
    """CASCI using DMRG as fcisolver"""

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None):
        self._casci = mcscf.CASCI(mf_or_mol, ncas, nelecas, ncore)
        self.mol = self._casci._scf.mol
        self._scf = self._casci._scf
        self.converged = False
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.fcisolver = DMRG(self._scf)

        self.e_tot = 0
        self.e_cas = None

    def kernel(self, mo_coeff=None):

        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        
        self.mo_coeff = mo_coeff

        mc = self._casci
        mo_coeff_act = mo_coeff[:, mc.ncore:mc.ncore + mc.ncas].copy()
        if self.mol.groupname.lower() == 'c1':
            orb_sym = [0] * mo_coeff_act.shape[1]
        else:
            orb_sym = casci_symm.label_symmetry_(mc, mo_coeff_act).orbsym
        mc.mo_coeff = mo_coeff

        h1e, e_core = mc.get_h1cas()
        g2e = mc.get_h2cas()
        self.e_tot, self.ci = self.fcisolver.kernel(h1e, g2e,
            mc.ncas, mc.nelecas, e_core, orb_sym)

        self.converged = True
        self.e_corr = self.e_tot - self._scf.e_tot

        lib.logger.note(self, 'DMRGCI E = %.16g  E_corr = %.16g',
            self.e_tot, self.e_corr)
        
        return self.e_corr, self.ci
    
    def make_rdm1(self, state=None, norb=None, nelec=None):
        '''
        Spin-traced one-particle density matrix in AO basis.
        dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>
        '''
        
        mc = self._casci
        casdm1 = self.fcisolver.make_rdm1(state, norb, nelec)
        mocore = self.mo_coeff[:,:mc.ncore]
        mocas = self.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
        dm1 = np.dot(mocore, mocore.T) * 2
        dm1 = dm1 + mocas @ casdm1 @ mocas.T
        return dm1


if __name__ == '__main__':

    from pyscf import gto, scf, fci

    mol = gto.M(
        atom='''
            O  0.000000  0.000000  0.000000
            H  0.758602  0.000000  0.504284
            H  0.758602  0.000000  -0.504284
        ''',
        basis='ccpvdz',
        verbose=4, symmetry=False, spin=0)

    # Mean Field
    mf = scf.RHF(mol).run()

    # DMRG FCI solver
    # mc = mcscf.CASCI(mf, 8, 8)
    # h1e, e_core = mc.get_h1cas()
    # g2e = mc.get_h2cas()
    # dmrg = DMRGCI(mf)
    # dmrg.dmrg_args['maxM'] = 2000
    # dmrg.dmrg_args['sweep_tol'] = 1E-16
    # ener, ci = dmrg.kernel(h1e, g2e, mc.ncas, mc.nelecas, e_core)
    # dm1, dm2 = dmrg.make_rdm12(ci, mc.ncas, mc.nelecas)
    # print(ener)

    # Reference FCI solver
    # xfci = fci.FCI(mol)
    # ener, ci = xfci.kernel(h1e, g2e, mc.ncas, mc.nelecas, ecore=e_core)
    # dm1x, dm2x = xfci.make_rdm12(ci, mc.ncas, mc.nelecas)
    # print(ener)
    # print(np.linalg.norm(dm1 - dm1x))
    # print(np.linalg.norm(dm2 - dm2x))
    
    # DMRG CASCI
    # myci = mcscf.CASCI(mf, 8, 8)
    # myci.fcisolver = DMRGCI(mf)
    # myci.fcisolver.dmrg_args['maxM'] = 2000
    # myci.fcisolver.dmrg_args['sweep_tol'] = 1E-16
    # myci = myci.run()
    # dm1 = myci.make_rdm1()

    # Reference CASCI
    # myci = mcscf.CASCI(mf, 8, 8)
    # myci.fcisolver.conv_tol = 1E-16
    # myci.run()
    # dm1x = myci.make_rdm1()
    # print(np.linalg.norm(dm1 - dm1x))

    # DMRG CASSCF
    mc = mcscf.CASSCF(mf, 6, 6)
    mc.fcisolver = DMRGCI(mf)
    mc.fcisolver.dmrg_args['maxM'] = 2000
    mc.fcisolver.dmrg_args['sweep_tol'] = 1E-16
    mc.run()

    # Reference CASSCF
    # mc = mcscf.CASSCF(mf, 6, 6)
    # mc.fcisolver.kernel()
    # mc.run()
