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


@pytest.fixture(scope="module", params=["none", "manual", "wick", "wick-os"])
def nord_algo_type(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 2, 4])
def spin_type(request):
    return request.param


class TestDMRGMPO:
    def test_rhf(self, tmp_path, system_def, nord_algo_type, spin_type):
        from pyscf import scf

        mol, ncore, ncas, name = system_def
        mol.spin = spin_type
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        if spin_type == 0:
            assert abs(mf.e_tot - -107.49650051179789) < 1e-10
        elif spin_type == 2:
            assert abs(mf.e_tot - -107.21998874754469) < 1e-10
        elif spin_type == 4:
            assert abs(mf.e_tot - -107.01879450793324) < 1e-10
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

        cidx = np.arange(ncas) < (n_elec - spin) // 2
        midx = (np.arange(ncas) >= (n_elec - spin) // 2) & (
            np.arange(ncas) < (n_elec + spin) // 2
        )

        if nord_algo_type == "none":
            cref, mref, wick = None, None, False
        elif nord_algo_type == "manual":
            cref, mref, wick = cidx, None, False
        elif nord_algo_type == "wick":
            cref, mref, wick = cidx, None, True
        elif nord_algo_type == "wick-os":
            cref, mref, wick = cidx, midx, True
        else:
            assert False

        mpo = driver.get_qc_mpo(
            h1e=h1e,
            g2e=g2e,
            ecore=ecore,
            # reorder="irrep",
            normal_order_ref=cref,
            normal_order_single_ref=mref,
            normal_order_wick=wick,
            cutoff=1e-12,
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
        if spin_type == 0:
            assert abs(energy - -107.654122447525) < 1e-6
        elif spin_type == 2:
            assert abs(energy - -106.939132859668) < 1e-6
        elif spin_type == 4:
            assert abs(energy - -107.031449471627) < 1e-6

        driver.finalize()
