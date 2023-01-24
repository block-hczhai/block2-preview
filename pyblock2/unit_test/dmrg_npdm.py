import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, NPDMAlgorithmTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module", params=["N2"])
def system_def(request):
    from pyscf import gto

    if request.param == "N2":
        mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
        return mol, 0, None, "N2"
    elif request.param == "C2":
        mol = gto.M(
            atom="C 0 0 0; C 0 0 1.2425", basis="ccpvdz", symmetry="d2h", verbose=0
        )
        return mol, 2, 8, "C2"


@pytest.fixture(scope="module", params=["Normal", "Fast", "SF", "SFLM"])
def algo_type(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 1, 2])
def site_type(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def fuse_ctrrot_type(request):
    return request.param


class TestNPDM:
    def test_rhf(self, tmp_path, system_def, site_type, algo_type, fuse_ctrrot_type):
        from pyscf import scf, fci

        mol, ncore, ncas, name = system_def
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        if name == "N2":
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif name == "C2":
            assert abs(mf.e_tot - -75.386902377706) < 1e-10
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            mf, ncore, ncas, g2e_symm=8
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SU2, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )

        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        bond_dims = [250] * 4 + [500] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=10 + np.random.randint(0, 2),
            tol=1E-12,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energies - -107.654122447523) < 1e-6
        elif name == "C2": # may stuck in local minima
            assert abs(energies - -75.552895292451) < 1e-6

        if algo_type == "Normal":
            n_algo_type = NPDMAlgorithmTypes.Normal
        elif algo_type == "Fast":
            n_algo_type = NPDMAlgorithmTypes.Fast
        elif algo_type == "SF":
            n_algo_type = NPDMAlgorithmTypes.SymbolFree | NPDMAlgorithmTypes.Compressed
        elif algo_type == "SFLM":
            n_algo_type = NPDMAlgorithmTypes.SymbolFree | NPDMAlgorithmTypes.Compressed | NPDMAlgorithmTypes.LowMem

        porder = 4 if algo_type == "SFLM" else 3
        if site_type != 0:
            porder = 2

        pdms = []
        for ip in range(1, porder + 1):
            pdms.append(driver.get_npdm(ket, pdm_type=ip, algo_type=n_algo_type,
                site_type=site_type, iprint=2,
                fused_contraction_rotation=fuse_ctrrot_type))

        driver.finalize()

        mx = fci.FCI(mf)
        mx.kernel(h1e, g2e, ncas, n_elec, tol=1E-12)

        if porder <= 3:
            fdms = fci.rdm.make_dm123('FCI3pdm_kern_sf', mx.ci, mx.ci, ncas, n_elec)
            E1, E2, E3 = [np.zeros_like(dm) for dm in fdms]
        else:
            fdms = fci.rdm.make_dm1234('FCI4pdm_kern_sf', mx.ci, mx.ci, ncas, n_elec)
            E1, E2, E3, E4 = [np.zeros_like(dm) for dm in fdms]

        deltaAA = np.eye(ncas)
        E1 += np.einsum('pa->pa', fdms[0], optimize=True)
        E2 += np.einsum('paqb->pqab', fdms[1], optimize=True)
        E2 += -1 * np.einsum('aq,pb->pqab', deltaAA, E1, optimize=True)
        E3 += np.einsum('paqbgc->pqgabc', fdms[2], optimize=True)
        E3 += -1 * np.einsum('ag,pqcb->pqgabc', deltaAA, E2, optimize=True)
        E3 += -1 * np.einsum('aq,pgbc->pqgabc', deltaAA, E2, optimize=True)
        E3 += -1 * np.einsum('bg,pqac->pqgabc', deltaAA, E2, optimize=True)
        E3 += -1 * np.einsum('aq,bg,pc->pqgabc', deltaAA, deltaAA, E1, optimize=True)

        if porder == 4:
            E4 += np.einsum('aebfcgdh->abcdefgh', fdms[3], optimize=True)
            E4 += -1 * np.einsum('eb,acdfgh->abcdefgh', deltaAA, E3, optimize=True)
            E4 += -1 * np.einsum('ec,abdgfh->abcdefgh', deltaAA, E3, optimize=True)
            E4 += -1 * np.einsum('ed,abchfg->abcdefgh', deltaAA, E3, optimize=True)
            E4 += -1 * np.einsum('fc,abdegh->abcdefgh', deltaAA, E3, optimize=True)
            E4 += -1 * np.einsum('fd,abcehg->abcdefgh', deltaAA, E3, optimize=True)
            E4 += -1 * np.einsum('gd,abcefh->abcdefgh', deltaAA, E3, optimize=True)
            E4 += -1 * np.einsum('eb,fc,adgh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('eb,fd,achg->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('eb,gd,acfh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('ec,fd,abgh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('ec,gd,abhf->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('ed,fc,abhg->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('fc,gd,abeh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
            E4 += -1 * np.einsum('eb,fc,gd,ah->abcdefgh', deltaAA, deltaAA, deltaAA, E1, optimize=True)

        E2 = E2.transpose(0, 1, 3, 2)
        E3 = E3.transpose(0, 1, 2, 5, 4, 3)

        if porder == 4:
            E4 = E4.transpose(0, 1, 2, 3, 7, 6, 5, 4)

        ddm1 = np.max(np.abs(pdms[0] - E1))
        ddm2 = np.max(np.abs(pdms[1] - E2))
        print('pdm1 diff = %9.2g' % ddm1)
        print('pdm2 diff = %9.2g' % ddm2)

        assert ddm1 < 1E-5
        assert ddm2 < 1E-5

        if porder >= 3:
            ddm3 = np.max(np.abs(pdms[2] - E3))
            print('pdm3 diff = %9.2g' % ddm3)
            assert ddm3 < 1E-5

        if porder >= 4:
            ddm4 = np.max(np.abs(pdms[3] - E4))
            print('pdm4 diff = %9.2g' % ddm4)
            assert ddm4 < 1E-5

    def test_uhf(self, tmp_path, system_def, site_type, algo_type):
        from pyscf import scf, fci

        mol, ncore, ncas, name = system_def
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        umf = mf.to_uhf()
        if name == "N2":
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif name == "C2":
            assert abs(mf.e_tot - -75.386902377706) < 1e-10

        ncas, n_elec, spin, ecore, h1es, g2es, orb_sym = itg.get_uhf_integrals(
            umf, ncore, ncas, g2e_symm=8
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SZ, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )

        mpo = driver.get_qc_mpo(h1e=h1es, g2e=g2es, ecore=ecore, iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        bond_dims = [250] * 4 + [500] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=10 + np.random.randint(0, 2),
            tol=1E-12,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energies - -107.654122447523) < 1e-6
        elif name == "C2":
            assert abs(energies - -75.552895292345) < 1e-6

        if algo_type == "Normal":
            n_algo_type = NPDMAlgorithmTypes.Normal
        elif algo_type == "Fast":
            n_algo_type = NPDMAlgorithmTypes.Fast
        elif algo_type == "SF":
            n_algo_type = NPDMAlgorithmTypes.SymbolFree | NPDMAlgorithmTypes.Compressed
        elif algo_type == "SFLM":
            n_algo_type = NPDMAlgorithmTypes.SymbolFree | NPDMAlgorithmTypes.Compressed | NPDMAlgorithmTypes.LowMem

        porder = 3 if site_type == 0 else 2

        pdms = []
        for ip in range(1, porder + 1):
            pdms.append(driver.get_npdm(ket, pdm_type=ip, algo_type=n_algo_type, site_type=site_type, iprint=2))

        driver.finalize()

        mx = fci.FCI(mf)
        h1e, g2e = itg.get_rhf_integrals(mf, ncore, ncas, g2e_symm=8)[-3:-1]
        mx.kernel(h1e, g2e, ncas, n_elec, tol=1E-12)

        fdms = fci.rdm.make_dm123('FCI3pdm_kern_sf', mx.ci, mx.ci, ncas, n_elec)
        E1, E2, E3 = [np.zeros_like(dm) for dm in fdms]

        deltaAA = np.eye(ncas)
        E1 += np.einsum('pa->pa', fdms[0], optimize=True)
        E2 += np.einsum('paqb->pqab', fdms[1], optimize=True)
        E2 += -1 * np.einsum('aq,pb->pqab', deltaAA, E1, optimize=True)
        E3 += np.einsum('paqbgc->pqgabc', fdms[2], optimize=True)
        E3 += -1 * np.einsum('ag,pqcb->pqgabc', deltaAA, E2, optimize=True)
        E3 += -1 * np.einsum('aq,pgbc->pqgabc', deltaAA, E2, optimize=True)
        E3 += -1 * np.einsum('bg,pqac->pqgabc', deltaAA, E2, optimize=True)
        E3 += -1 * np.einsum('aq,bg,pc->pqgabc', deltaAA, deltaAA, E1, optimize=True)
        E2 = E2.transpose(0, 1, 3, 2)
        E3 = E3.transpose(0, 1, 2, 5, 4, 3)

        _1pdm, _2pdm = pdms[:2]
        _1pdm = _1pdm[0] + _1pdm[1]
        _2pdm = _2pdm[0] + _2pdm[1] + _2pdm[1].transpose(1, 0, 3, 2) + _2pdm[2]
        pdms[:2] = _1pdm, _2pdm

        ddm1 = np.max(np.abs(pdms[0] - E1))
        ddm2 = np.max(np.abs(pdms[1] - E2))
        print('pdm1 diff = %9.2g' % ddm1)
        print('pdm2 diff = %9.2g' % ddm2)

        assert ddm1 < 1E-5
        assert ddm2 < 1E-5

        if porder >= 3:
            _3pdm = pdms[2]
            _3pdm = _3pdm[0] \
                + _3pdm[1] + _3pdm[1].transpose(0, 2, 1, 4, 3, 5) + _3pdm[1].transpose(2, 0, 1, 4, 5, 3) \
                + _3pdm[2] + _3pdm[2].transpose(1, 0, 2, 3, 5, 4) + _3pdm[2].transpose(1, 2, 0, 5, 3, 4) \
                + _3pdm[3]
            ddm3 = np.max(np.abs(_3pdm - E3))
            print('pdm3 diff = %9.2g' % ddm3)
            assert ddm3 < 1E-5

    def test_ghf(self, tmp_path, system_def, site_type, algo_type):
        from pyscf import scf, fci

        mol, xncore, xncas, name = system_def
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        gmf = mf.to_ghf()
        if name == "N2":
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif name == "C2":
            assert abs(mf.e_tot - -75.386902377706) < 1e-10
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_ghf_integrals(
            gmf, xncore, xncas, g2e_symm=8
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SGF, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )

        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        bond_dims = [250] * 4 + [500] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=10 + np.random.randint(0, 2),
            tol=0,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energies - -107.654122447523) < 1e-6
        elif name == "C2":
            assert abs(energies - -75.552895292345) < 1e-6

        if algo_type == "Normal":
            n_algo_type = NPDMAlgorithmTypes.Normal
        elif algo_type == "Fast":
            n_algo_type = NPDMAlgorithmTypes.Fast
        elif algo_type == "SF":
            n_algo_type = NPDMAlgorithmTypes.SymbolFree | NPDMAlgorithmTypes.Compressed
        elif algo_type == "SFLM":
            n_algo_type = (
                NPDMAlgorithmTypes.SymbolFree
                | NPDMAlgorithmTypes.Compressed
                | NPDMAlgorithmTypes.LowMem
            )

        porder = 2

        pdms = []
        for ip in range(1, porder + 1):
            pdms.append(
                driver.get_npdm(
                    ket,
                    pdm_type=ip,
                    algo_type=n_algo_type,
                    site_type=site_type,
                    iprint=2,
                )
            )

        driver.finalize()

        ncas = ncas // 2
        xncore = xncore // 2
        mx = fci.FCI(mf)
        h1e, g2e = itg.get_rhf_integrals(mf, xncore, xncas, g2e_symm=8)[-3:-1]
        mx.kernel(h1e, g2e, ncas, n_elec, tol=1e-12)

        fdms = fci.rdm.make_rdm12("FCIrdm12kern_sf", mx.ci, mx.ci, ncas, n_elec)
        E1, E2 = [np.zeros_like(dm) for dm in fdms]

        deltaAA = np.eye(ncas)
        E1 += np.einsum("pa->pa", fdms[0], optimize=True)
        E2 += np.einsum("paqb->pqab", fdms[1], optimize=True)
        E2 += -1 * np.einsum("aq,pb->pqab", deltaAA, E1, optimize=True)
        E2 = E2.transpose(0, 1, 3, 2)

        def take(x, i):
            gg = []
            for _ in range(x.ndim // 2):
                gg.append(i % 2)
                i = i // 2
            sl = tuple(slice(g, None, 2) for g in gg)
            return x[sl + sl[::-1]]

        _1pdm, _2pdm = pdms[:2]
        _1pdm = sum(take(_1pdm, i) for i in range(2 ** 1))
        _2pdm = sum(take(_2pdm, i) for i in range(2 ** 2))

        ddm1 = np.max(np.abs(_1pdm - E1))
        ddm2 = np.max(np.abs(_2pdm - E2))
        print("pdm1 diff = %9.2g" % ddm1)
        print("pdm2 diff = %9.2g" % ddm2)

        assert ddm1 < 1e-5
        assert ddm2 < 1e-5
