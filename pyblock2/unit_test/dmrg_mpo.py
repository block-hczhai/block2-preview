import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

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


mpo_algo_names = [
    "Bipartite",
    "SVD",
    "BlockedSVD",
    "FastBlockedSVD",
    "BlockedRescaledSVD",
    "FastBlockedRescaledSVD",
    "BlockedBipartite",
    "FastBlockedBipartite",
    "RescaledSVD",
    "FastSVD",
    "FastRescaledSVD",
    "FastBipartite",
    "Conventional",
    "ConventionalNC",
    "ConventionalCN",
    "NoTransConventional",
    "NoTransConventionalNC",
    "NoTransConventionalCN",
    "NoRIntermedConventional",
    "NoTransNoRIntermedConventional",
]


@pytest.fixture(scope="module", params=mpo_algo_names)
def mpo_algo_type(request):
    return request.param


class TestDMRGMPO:
    def test_rhf(self, tmp_path, system_def, mpo_algo_type):
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

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SU2, n_threads=4
        )

        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo_algo_type = getattr(MPOAlgorithmTypes, mpo_algo_type)
        mpo = driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            reorder='irrep',
            cutoff=1e-12,
            algo_type=mpo_algo_type,
            iprint=1,
        )
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        bond_dims = [250] * 8
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=2,
        )
        if name == "N2":
            assert abs(energy - -107.65412244752470) < 1e-6

        driver.finalize()
