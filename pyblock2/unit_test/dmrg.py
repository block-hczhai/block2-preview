import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

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


@pytest.fixture(scope="module", params=["Coulomb", "Gaunt", "Breit"])
def dhf_type(request):
    return request.param


class TestDMRG:
    def test_rhf(self, tmp_path, system_def):
        from pyscf import scf

        mol, ncore, ncas, name = system_def
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        if name == "N2":
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif name == "C2":
            assert abs(mf.e_tot - -75.386902377706) < 1e-10
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            mf, ncore, ncas
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SU2, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        b = driver.expr_builder()
        b.add_sum_term("(C+D)0", np.sqrt(2) * h1e)
        b.add_sum_term("((C+(C+D)0)1+D)0", (2 * 0.5) * g2e.transpose(0, 2, 3, 1))
        b.add_const(ecore)

        mpo = driver.get_mpo(b.finalize(), iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=3)
        bond_dims = [250] * 8
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energies[0] - -107.654122447523) < 1e-6
            assert abs(energies[1] - -106.959626154679) < 1e-6
            assert abs(energies[2] - -106.943756938989) < 1e-6
        elif name == "C2": # may stuck in local minima
            assert abs(energies[0] - -75.552895292451) < 1e-6
            assert abs(energies[1] - -75.536490900344) < 1e-6
            assert abs(energies[2] - -75.536490900079) < 1e-6

        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energy - -107.654122447523) < 1e-6
        elif name == "C2":
            assert abs(energy - -75.552895292451) < 1e-6

        driver.target.pg = 2
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        print(energy)
        if name == "N2":
            assert abs(energy - -107.30674473475638) < 1e-6
        elif name == "C2":
            assert abs(energy - -75.36070319318232) < 1e-6

    def test_uhf(self, tmp_path, system_def):
        from pyscf import scf

        mol, ncore, ncas, name = system_def
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        if name == "N2":
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif name == "C2":
            assert abs(mf.e_tot - -75.386902377706) < 1e-10
        ncas, n_elec, spin, ecore, h1es, g2es, orb_sym = itg.get_uhf_integrals(
            mf, ncore, ncas
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SZ, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        for h1e in h1es:
            h1e[np.abs(h1e) < 1e-7] = 0
            assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        for ig, g2e in enumerate(g2es):
            g2e[np.abs(g2e) < 1e-7] = 0
            if ig != 1:
                assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
            assert np.linalg.norm(g2e - g2e.transpose(0, 1, 3, 2).conj()) < 1e-7
            assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        b = driver.expr_builder()
        b.add_sum_term("cd", h1es[0])
        b.add_sum_term("CD", h1es[1])
        b.add_sum_term("ccdd", 0.5 * g2es[0].transpose(0, 2, 3, 1))
        b.add_sum_term("cCDd", 0.5 * g2es[1].transpose(0, 2, 3, 1))
        b.add_sum_term("CcdD", 0.5 * g2es[1].transpose(2, 0, 1, 3))
        b.add_sum_term("CCDD", 0.5 * g2es[2].transpose(0, 2, 3, 1))
        b.add_const(ecore)

        mpo = driver.get_mpo(b.finalize(), iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=3)
        bond_dims = [400] * 8
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energies[0] - -107.654122447523) < 1e-6
            assert abs(energies[1] - -107.031449471625) < 1e-6
            assert abs(energies[2] - -106.959626154679) < 1e-6
        elif name == "C2":
            assert abs(energies[0] - -75.552895292345) < 1e-6
            assert abs(energies[1] - -75.536490900060) < 1e-6
            assert abs(energies[2] - -75.536490900060) < 1e-6

        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energy - -107.654122447523) < 1e-6
        elif name == "C2":
            assert abs(energy - -75.552895292345) < 1e-6

    def test_ghf(self, tmp_path, system_def):
        from pyscf import scf

        mol, ncore, ncas, name = system_def
        mf = scf.GHF(mol).run(conv_tol=1e-14)
        if name == "N2":
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif name == "C2":
            assert abs(mf.e_tot - -75.386902377706) < 1e-10
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_ghf_integrals(
            mf, ncore, ncas
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SGF, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0
        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        b = driver.expr_builder()
        b.add_sum_term("CD", h1e)
        b.add_sum_term("CCDD", 0.5 * g2e.transpose(0, 2, 3, 1))
        b.add_const(ecore)

        mpo = driver.get_mpo(b.finalize(), iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=3)
        bond_dims = [400] * 8
        noises = [1e-3] * 4 + [1e-4] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )
        if name == "N2":
            assert abs(energies[0] - -107.654122447523) < 1e-6
            assert abs(energies[1] - -107.031449471625) < 1e-6
            assert abs(energies[2] - -107.031449471625) < 1e-6
        elif name == "C2": # may stuck in local minima
            assert abs(energies[0] - -75.552895292344) < 1e-6
            assert abs(energies[1] - -75.536490899999) < 1e-6
            assert abs(energies[2] - -75.536490899999) < 1e-6

    def test_dhf(self, tmp_path, system_def, dhf_type):
        from pyscf import scf

        mol, ncore, ncas, name = system_def
        if dhf_type == "Coulomb":
            mf = (
                scf.DHF(mol).set(with_gaunt=False, with_breit=False).run(conv_tol=1e-12)
            )
            if name == "N2":
                assert abs(mf.e_tot - -107.544314972723) < 1e-10
            elif name == "C2":
                assert abs(mf.e_tot - -75.419258871568) < 1e-10
        elif dhf_type == "Gaunt":
            mf = scf.DHF(mol).set(with_gaunt=True, with_breit=False).run(conv_tol=1e-12)
            if name == "N2":
                assert abs(mf.e_tot - -107.535004358049) < 1e-10
            elif name == "C2":
                assert abs(mf.e_tot - -75.413568302252) < 1e-10
        elif dhf_type == "Breit":
            mf = scf.DHF(mol).set(with_gaunt=True, with_breit=True).run(conv_tol=1e-12)
            if name == "N2":
                assert abs(mf.e_tot - -107.535230665795) < 1e-10
            elif name == "C2":
                assert abs(mf.e_tot - -75.413692277327) < 1e-10
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_dhf_integrals(
            mf, ncore, ncas, pg_symm=False
        )
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SGFCPX, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0
        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        b = driver.expr_builder()
        b.add_sum_term("CD", h1e)
        b.add_sum_term("CCDD", 0.5 * g2e.transpose(0, 2, 3, 1))
        b.add_const(ecore)

        mpo = driver.get_mpo(b.finalize(), iprint=1)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=2)
        bond_dims = [400] * 8
        noises = [1e-3] * 4 + [1e-4] * 4 + [0]
        thrds = [1e-10] * 8
        energies = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
            dav_max_iter=250,
        )
        if name == "N2":
            if dhf_type == "Coulomb":
                assert abs(energies[0] - -107.701986146895) < 1e-6
                assert abs(energies[1] - -107.404877016101) < 1e-6
            elif dhf_type == "Gaunt":
                assert abs(energies[0] - -107.692699451270) < 1e-6
                assert abs(energies[1] - -107.395534308264) < 1e-6
            elif dhf_type == "Breit":
                assert abs(energies[0] - -107.692920949172) < 1e-6
                assert abs(energies[1] - -107.395767689285) < 1e-6
        elif name == "C2":
            if dhf_type == "Coulomb":
                assert abs(energies[0] - -75.585051021867) < 1e-6
                assert abs(energies[1] - -75.568910613425) < 1e-6
            elif dhf_type == "Gaunt":
                assert abs(energies[0] - -75.579351391472) < 1e-6
                assert abs(energies[1] - -75.563183490113) < 1e-6
            elif dhf_type == "Breit":
                assert abs(energies[0] - -75.579482776888) < 1e-6
                assert abs(energies[1] - -75.563315259912) < 1e-6
