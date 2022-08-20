import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module", params=["Cl"])
def system_def(request):
    from pyscf import gto

    if request.param == "Cl":
        mol = gto.M(atom="Cl 0 0 0", basis='cc-pvdz-dk', verbose=0, spin=1)
        return mol, 5, 4, "Cl"


@pytest.fixture(scope="module", params=["bp", "bp-amfi", "x2c", "x2c-amfi"])
def soc_type(request):
    return request.param


class TestDMRG:
    def test_rhf(self, tmp_path, system_def, soc_type):
        from pyscf import scf, mcscf

        mol, ncore, ncas, _ = system_def

        if "x2c" in soc_type:
            mf = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-14)
            assert abs(mf.e_tot - -460.87496073796086) < 1e-10
        else:
            mf = scf.RHF(mol).run(conv_tol=1e-14)
            assert abs(mf.e_tot - -459.1079678030042) < 1e-10

        ncaselec = mol.nelectron - ncore * 2
        mc = mcscf.CASSCF(mf, ncas, ncaselec).state_average_(np.ones(3) / 3.0)
        mc.kernel()
        mf.mo_coeff = mc.mo_coeff

        amfi = "amfi" in soc_type
        x2c = "x2c" in soc_type
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_rhf_somf_integrals(
            mf, ncore, ncas, pg_symm=False, amfi=amfi, x2c1e=x2c, x2c2e=x2c
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
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=6)
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
        from pyscf.data import nist
        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        zfs = np.average(energies[4:6]) - np.average(energies[0:4])

        if "x2c" in soc_type:
            assert abs(zfs * au2cm - 823.00213) < 1e-2
        else:
            assert abs(zfs * au2cm - 837.29645) < 1e-2

    def test_uhf(self, tmp_path, system_def, soc_type):
        from pyscf import scf, mcscf

        mol, ncore, ncas, _ = system_def

        if "x2c" in soc_type:
            mf = scf.UHF(mol).sfx2c1e().run(conv_tol=1e-14)
            assert abs(mf.e_tot - -460.8789016293768) < 1e-10
        else:
            mf = scf.UHF(mol).run(conv_tol=1e-14)
            assert abs(mf.e_tot - -459.11192524585005) < 1e-10

        ncaselec = mol.nelectron - ncore * 2
        mc = mcscf.UCASSCF(mf, ncas, ncaselec).state_average_(np.ones(3) / 3.0)
        try:
            mc.kernel()
        except AssertionError:
            pytest.skip("this pyscf version does not support UCASSCF state average.")
        mf.mo_coeff = mc.mo_coeff

        amfi = "amfi" in soc_type
        x2c = "x2c" in soc_type
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_rhf_somf_integrals(
            mf, ncore, ncas, pg_symm=False, amfi=amfi, x2c1e=x2c, x2c2e=x2c
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
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=6)
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
        from pyscf.data import nist
        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        zfs = np.average(energies[4:6]) - np.average(energies[0:4])

        if "x2c" in soc_type:
            assert abs(zfs * au2cm - 843.50084) < 1e-2
        else:
            assert abs(zfs * au2cm - 857.67109) < 1e-2
