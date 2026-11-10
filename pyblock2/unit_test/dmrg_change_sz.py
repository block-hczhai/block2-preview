import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def symm_type(pytestconfig):
    return pytestconfig.getoption("symm")


@pytest.fixture(scope="module", params=[0, 2], ids=["singlet", "triplet"])
def spin(request):
    return request.param


def get_isometry_error(driver, mps):
    """Largest deviation from orthonormality of the L/R site tensors of an MPS."""
    umps = driver.bw.bs.UnfusedMPS(mps)
    cf = umps.canonical_form
    err = 0.0
    for i in range(umps.n_sites):
        if cf[i] not in "LR":
            continue
        ovlps = {}
        for ip in range(len(umps.tensors[i].data)):
            for (lq, rq), ts in umps.tensors[i].data[ip]:
                ts = np.array(ts)
                if cf[i] == "R":
                    k, ts = str(lq), ts.reshape(ts.shape[0], -1)
                    ovlps[k] = ovlps.get(k, 0) + ts @ ts.T
                else:
                    k, ts = str(rq), ts.reshape(-1, ts.shape[2])
                    ovlps[k] = ovlps.get(k, 0) + ts.T @ ts
        for x in ovlps.values():
            err = max(err, np.max(np.abs(x - np.eye(x.shape[0]))))
    return err


class TestChangeToSZ:
    def test_mps_change_to_sz(self, tmp_path, symm_type, spin):
        if symm_type == "sany":
            pytest.skip("mps_change_to_sz requires the SU2 mode")

        from pyscf import gto, scf

        mol = gto.M(
            atom=[["H", (0, 0, 1.6 * i)] for i in range(8)], basis="sto3g", verbose=0
        )
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        ncas, n_elec, _, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            mf, ncore=0, ncas=None, g2e_symm=8
        )

        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SU2, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
        driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=[50] * 20,
            noises=[1e-4] * 6 + [0] * 14,
            thrds=[1e-10] * 20,
            iprint=0,
        )
        driver.align_mps_center(ket, ref=0)
        impo = driver.get_identity_mpo()
        energy = driver.expectation(ket, mpo, ket) / driver.expectation(ket, impo, ket)

        # for non-singlet, the extracted Sz component has norm**2 = 1 / (2S + 1)
        zket = driver.mps_change_to_sz(ket, "ZKET", sz=None if spin == 0 else spin)
        if spin != 0:
            zket0 = driver.mps_change_to_sz(ket, "ZKET0", sz=0)
        norm2 = 1.0 / (spin + 1)

        driver.symm_type = SymmetryTypes.SZ
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        impo = driver.get_identity_mpo()

        assert abs(driver.expectation(zket, impo, zket) - norm2) < 1e-10
        assert abs(driver.expectation(zket, mpo, zket) / norm2 - energy) < 1e-8

        # site tensors are orthonormal, as the canonical form claims
        assert get_isometry_error(driver, zket) < 1e-10

        # a sweep without truncation started from the converted MPS
        # cannot raise the energy
        zket.iscale(1 / np.sqrt(norm2))
        driver.dmrg(
            mpo, zket, n_sweeps=1, bond_dims=[300], noises=[0], thrds=[1e-10], iprint=0
        )
        ener = driver.expectation(zket, mpo, zket) / driver.expectation(
            zket, impo, zket
        )
        assert ener < energy + 1e-8

        if spin != 0:
            driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=0, orb_sym=orb_sym)
            mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
            impo = driver.get_identity_mpo()
            assert abs(driver.expectation(zket0, impo, zket0) - norm2) < 1e-10
            assert abs(driver.expectation(zket0, mpo, zket0) / norm2 - energy) < 1e-8
            assert get_isometry_error(driver, zket0) < 1e-10
