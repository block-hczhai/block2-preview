import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import soc_integrals as itgsoc
from pyblock2.driver.core import SOCDMRGDriver, SymmetryTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def symm_type(pytestconfig):
    return pytestconfig.getoption("symm")


@pytest.fixture(scope="module", params=["Cl"])
def system_def(request):
    from pyscf import gto

    if request.param == "Cl":
        mol = gto.M(atom="Cl 0 0 0", basis="cc-pvdz-dk", verbose=0, spin=1, max_memory=8000)
        return mol, 5, 4, "Cl"


@pytest.fixture(
    scope="module",
    params=["bp", "bp-amfi", "bp-amfi-hybrid", "x2c", "x2c-amfi", "x2c-amfi-hybrid"],
)
def soc_type(request):
    return request.param


class TestDMRG:
    def test_rhf(self, tmp_path, system_def, soc_type, symm_type):
        from pyscf import scf, mcscf

        mol, ncore, ncas, _ = system_def

        if "x2c" in soc_type:
            mf = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-14)
            assert abs(mf.e_tot - -460.87496073796086) < 1e-7
        else:
            mf = scf.RHF(mol).run(conv_tol=1e-14)
            assert abs(mf.e_tot - -459.1079678030042) < 1e-7

        ncaselec = mol.nelectron - ncore * 2
        mc = mcscf.CASSCF(mf, ncas, ncaselec).state_average_(np.ones(3) / 3.0)
        mc.kernel()
        mf.mo_coeff = mc.mo_coeff

        amfi = "amfi" in soc_type
        x2c = "x2c" in soc_type
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_rhf_somf_integrals(
            mf, ncore, ncas, pg_symm=False, amfi=amfi, x2c1e=x2c, x2c2e=x2c
        )

        symm = SymmetryTypes.SAnySGFCPX if symm_type == "sany" else SymmetryTypes.SGFCPX
        driver = SOCDMRGDriver(scratch=str(tmp_path / "nodex"), symm_type=symm, n_threads=4)
        driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym)
        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        bond_dims = [400] * 8
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8

        if "hybrid" in soc_type:
            assert np.linalg.norm(g2e.imag) < 1e-7
            mpo_cpx = driver.get_qc_mpo(h1e=h1e - h1e.real, g2e=None, ecore=0, iprint=1)
            # fd = driver.write_fcidump(h1e=np.ascontiguousarray(h1e - h1e.real), g2e=None, ecore=0)
            # mpo_cpx = driver.get_conventional_qc_mpo(fd)
            symm = SymmetryTypes.SAnySGF if symm_type == "sany" else SymmetryTypes.SGF
            driver.set_symm_type(symm_type=symm, reset_frame=False)
            driver.initialize_system(
                n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym
            )
            mpo = driver.get_qc_mpo(h1e=h1e.real, g2e=g2e.real, ecore=ecore, iprint=1)
            # fd = driver.write_fcidump(h1e=np.ascontiguousarray(h1e.real), g2e=np.ascontiguousarray(g2e.real), ecore=ecore)
            # mpo = driver.get_conventional_qc_mpo(fd)
            ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=6 * 2)
            energies = driver.hybrid_mpo_dmrg(
                mpo,
                mpo_cpx,
                ket,
                n_sweeps=20,
                bond_dims=bond_dims,
                noises=noises,
                thrds=thrds,
                dav_max_iter=250,
                iprint=1,
            )
        else:
            mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
            # fd = driver.write_fcidump(h1e, g2e, ecore=ecore)
            # mpo = driver.get_conventional_qc_mpo(fd)
            ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=6)
            energies = driver.dmrg(
                mpo,
                ket,
                n_sweeps=20,
                bond_dims=bond_dims,
                noises=noises,
                thrds=thrds,
                dav_max_iter=250,
                iprint=1,
            )

        from pyscf.data import nist

        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        zfs = np.average(energies[4:6]) - np.average(energies[0:4])

        if "x2c" in soc_type:
            assert abs(zfs * au2cm - 823.00213) < 1
        else:
            assert abs(zfs * au2cm - 837.29645) < 1

        driver.finalize()

    def test_uhf(self, tmp_path, system_def, soc_type, symm_type):
        from pyscf import scf, mcscf

        mol, ncore, ncas, _ = system_def

        if "x2c" in soc_type:
            mf = scf.UHF(mol).sfx2c1e().run(conv_tol=1e-14)
            assert abs(mf.e_tot - -460.8789016293768) < 1e-7
        else:
            mf = scf.UHF(mol).run(conv_tol=1e-14)
            assert abs(mf.e_tot - -459.11192524585005) < 1e-7

        ncaselec = mol.nelectron - ncore * 2
        mc = mcscf.UCASSCF(mf, ncas, ncaselec).state_average_(np.ones(3) / 3.0)
        try:
            mc.kernel()
        except AssertionError:
            pytest.skip("this pyscf version does not support UCASSCF state average.")
        mf.mo_coeff = mc.mo_coeff

        amfi = "amfi" in soc_type
        x2c = "x2c" in soc_type
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itgsoc.get_uhf_somf_integrals(
            mf, ncore, ncas, pg_symm=False, amfi=amfi, x2c1e=x2c, x2c2e=x2c
        )

        symm = SymmetryTypes.SAnySGFCPX if symm_type == "sany" else SymmetryTypes.SGFCPX
        driver = SOCDMRGDriver(scratch=str(tmp_path / "nodex"), symm_type=symm, n_threads=4)
        driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym)
        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        bond_dims = [400] * 8
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8

        if "hybrid" in soc_type:
            assert np.linalg.norm(g2e.imag) < 1e-7
            mpo_cpx = driver.get_qc_mpo(h1e=h1e - h1e.real, g2e=None, ecore=0, iprint=1)
            symm = SymmetryTypes.SAnySGF if symm_type == "sany" else SymmetryTypes.SGF
            driver.set_symm_type(symm_type=symm, reset_frame=False)
            driver.initialize_system(
                n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym
            )
            mpo = driver.get_qc_mpo(h1e=h1e.real, g2e=g2e.real, ecore=ecore, iprint=1)
            ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=6 * 2)
            energies = driver.hybrid_mpo_dmrg(
                mpo,
                mpo_cpx,
                ket,
                n_sweeps=20,
                bond_dims=bond_dims,
                noises=noises,
                thrds=thrds,
                dav_max_iter=250,
                iprint=1,
            )
        else:
            mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
            ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=6)
            energies = driver.dmrg(
                mpo,
                ket,
                n_sweeps=20,
                bond_dims=bond_dims,
                noises=noises,
                thrds=thrds,
                dav_max_iter=250,
                iprint=1,
            )

        from pyscf.data import nist

        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        zfs = np.average(energies[4:6]) - np.average(energies[0:4])

        if "x2c" in soc_type:
            assert abs(zfs * au2cm - 843.50084) < 1
        else:
            assert abs(zfs * au2cm - 857.67109) < 1

        driver.finalize()
