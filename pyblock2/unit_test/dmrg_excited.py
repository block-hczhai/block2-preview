import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def symm_type(pytestconfig):
    return pytestconfig.getoption("symm")


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


@pytest.fixture(scope="module", params=[True, False])
def singlet_embedding_type(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 2, 4])
def spin_type(request):
    return request.param


class TestExcitedDMRG:
    def test_rhf(
        self, tmp_path, system_def, singlet_embedding_type, spin_type, symm_type
    ):
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

        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        symm = SymmetryTypes.SAnySU2 if symm_type == "sany" else SymmetryTypes.SU2
        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=symm, n_threads=4
        )

        driver.initialize_system(
            n_sites=ncas,
            n_elec=n_elec,
            spin=spin_type,
            orb_sym=orb_sym,
            singlet_embedding=singlet_embedding_type,
        )
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1)
        # fd = driver.write_fcidump(h1e, g2e, ecore=ecore)
        # mpo = driver.get_conventional_qc_mpo(fd)
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=5)
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
            iprint=2,
        )
        if name == "N2":
            if spin_type == 0:
                assert abs(energies[0] - -107.65412244752470) < 1e-6
                assert abs(energies[1] - -106.95962615467998) < 1e-6
                assert abs(energies[2] - -106.94375693899154) < 1e-6
            elif spin_type == 2:
                assert abs(energies[0] - -106.93913285966788) < 1e-6
                assert abs(energies[1] - -106.71173801496816) < 1e-6
                assert abs(energies[2] - -106.70055113334190) < 1e-6
            elif spin_type == 4:
                assert abs(energies[0] - -107.03144947162717) < 1e-6
                assert abs(energies[1] - -106.63379058932087) < 1e-6
                assert abs(energies[2] - -106.62753894625438) < 1e-6

        driver.finalize()
