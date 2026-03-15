import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def symm_type(pytestconfig):
    return pytestconfig.getoption("symm")


@pytest.fixture(scope="module")
def fd_data(pytestconfig):
    return pytestconfig.getoption("fd_data")


@pytest.fixture(scope="module", params=["N2"])
def name(request):
    return request.param


@pytest.fixture(scope="module", params=["Gaunt"])
def dhf_type(request):
    return request.param


mpo_algo_names = [
    "Bipartite",
    "SVD",
    "DisjointSVD",
    "FastDisjointSVD",
    "DisjointRRQR",
    "FastDisjointRRQR",
    # "BlockedSumDisjointSVD",
    # "FastBlockedSumDisjointSVD",
    # "BlockedRescaledSumDisjointSVD",
    # "FastBlockedRescaledSumDisjointSVD",
    # "BlockedDisjointSVD",
    # "FastBlockedDisjointSVD",
    # "BlockedRescaledDisjointSVD",
    # "FastBlockedRescaledDisjointSVD",
    # "RescaledDisjointSVD",
    # "FastDisjointSVD",
    # "FastRescaledDisjointSVD",
    "BlockedSumSVD",
    "FastBlockedSumSVD",
    "BlockedRescaledSumSVD",
    "FastBlockedRescaledSumSVD",
    "BlockedSumBipartite",
    "FastBlockedSumBipartite",
    "BlockedSVD",
    "FastBlockedSVD",
    "BlockedRescaledSVD",
    "FastBlockedRescaledSVD",
    "RRQR",
    "BlockedSumRRQR",
    "FastBlockedSumRRQR",
    "BlockedRescaledSumRRQR",
    "FastBlockedRescaledSumRRQR",
    "BlockedRRQR",
    "FastBlockedRRQR",
    "BlockedRescaledRRQR",
    "FastBlockedRescaledRRQR",
    "RescaledRRQR",
    "FastRRQR",
    "FastRescaledRRQR",
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


@pytest.fixture(
    scope="module",
    params=["FastBipartite", "FastBlockedSVD", "DisjointSVD", "DisjointRRQR", "FastRRQR", "FastBlockedRRQR"],
)
def dhf_mpo_algo_type(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def block_max_length(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def disjoint_all_blocks(request):
    return request.param


class TestDMRGMPO:
    def test_rhf(
        self, tmp_path, name, mpo_algo_type, block_max_length, disjoint_all_blocks, symm_type, fd_data
    ):

        symm = SymmetryTypes.SAnySU2 if symm_type == "sany" else SymmetryTypes.SU2
        driver = DMRGDriver(scratch=str(tmp_path / "nodex"), symm_type=symm, n_threads=4)

        if fd_data == "":
            from pyscf import gto, scf

            if name == "N2":
                mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)
                mf = scf.RHF(mol).run(conv_tol=1e-14)
                assert abs(mf.e_tot - -107.49650051179789) < 1e-10
                ncore, ncas = 0, None
            elif name == "C2":
                mol = gto.M(
                    atom="C 0 0 0; C 0 0 1.2425", basis="ccpvdz", symmetry="d2h", verbose=0
                )
                mf = scf.RHF(mol).run(conv_tol=1e-14)
                assert abs(mf.e_tot - -75.386902377706) < 1e-10
                ncore, ncas = 2, 8
            else:
                assert False

            ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, ncore, ncas)
            driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
        else:
            assert name == "N2"
            driver.read_fcidump(filename=fd_data + '/N2.STO3G.RHF.FCIDUMP', pg='d2h')
            driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec,
                spin=driver.spin, orb_sym=driver.orb_sym)
            h1e, g2e, ecore = driver.h1e, driver.g2e, driver.ecore

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        mpo_algo_type = getattr(MPOAlgorithmTypes, mpo_algo_type)

        if MPOAlgorithmTypes.Disjoint not in mpo_algo_type and disjoint_all_blocks:
            pytest.skip()

        if MPOAlgorithmTypes.Conventional in mpo_algo_type and block_max_length:
            pytest.skip()

        if MPOAlgorithmTypes.Conventional in mpo_algo_type and symm_type == "sany":
            pytest.skip()

        mpo = driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            reorder="irrep",
            cutoff=1e-12,
            algo_type=mpo_algo_type,
            block_max_length=block_max_length,
            disjoint_all_blocks=disjoint_all_blocks,
            iprint=2,
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

    def test_dhf(self, tmp_path, dhf_type, dhf_mpo_algo_type, symm_type, fd_data):

        symm = SymmetryTypes.SAnySGFCPX if symm_type == "sany" else SymmetryTypes.SGFCPX
        driver = DMRGDriver(scratch=str(tmp_path / "nodex"), symm_type=symm, n_threads=4)

        if fd_data == "":
            from pyscf import gto, scf

            mol = gto.M(
                atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
                basis="sto3g",
                symmetry=False,
                verbose=0,
            )
            ncore, ncas = 0, None

            if dhf_type == "Coulomb":
                mf = scf.DHF(mol).set(with_gaunt=False, with_breit=False).run(conv_tol=1e-12)
                assert abs(mf.e_tot - -75.0052749296693) < 1e-10
            elif dhf_type == "Gaunt":
                mf = scf.DHF(mol).set(with_gaunt=True, with_breit=False).run(conv_tol=1e-12)
                assert abs(mf.e_tot - -74.9978573265168) < 1e-10
            elif dhf_type == "Breit":
                mf = scf.DHF(mol).set(with_gaunt=True, with_breit=True).run(conv_tol=1e-12)
                assert abs(mf.e_tot - -74.9980861167505) < 1e-10
            else:
                assert False
            ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_dhf_integrals(
                mf, ncore, ncas, pg_symm=False
            )
            driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
        else:
            if dhf_type == "Coulomb":
                driver.read_fcidump(filename=fd_data + '/H2O.STO3G.DHF-C.FCIDUMP', pg='d2h')
            elif dhf_type == "Gaunt":
                driver.read_fcidump(filename=fd_data + '/H2O.STO3G.DHF-G.FCIDUMP', pg='d2h')
            elif dhf_type == "Breit":
                driver.read_fcidump(filename=fd_data + '/H2O.STO3G.DHF-B.FCIDUMP', pg='d2h')
            else:
                assert False
            driver.initialize_system(
                n_sites=driver.n_sites,
                n_elec=driver.n_elec,
                spin=driver.spin,
                orb_sym=driver.orb_sym,
            )
            h1e, g2e, ecore = driver.h1e, driver.g2e, driver.ecore

        h1e[np.abs(h1e) < 1e-7] = 0
        g2e[np.abs(g2e) < 1e-7] = 0

        assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(2, 3, 0, 1)) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-7
        assert np.linalg.norm(g2e - g2e.transpose(3, 2, 1, 0).conj()) < 1e-7

        mpo = driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            cutoff=1e-12,
            algo_type=getattr(MPOAlgorithmTypes, dhf_mpo_algo_type),
            iprint=2,
        )
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        bond_dims = [400] * 8
        noises = [1e-3] * 4 + [1e-4] * 4 + [0]
        thrds = [1e-10] * 8
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=2,
            dav_max_iter=250,
        )
        if dhf_type == "Coulomb":
            assert abs(energy - -75.05489216789145) < 1e-5
        elif dhf_type == "Gaunt":
            assert abs(energy - -75.04749505314540) < 1e-5
        elif dhf_type == "Breit":
            assert abs(energy - -75.04772059258008) < 1e-5

        driver.finalize()
