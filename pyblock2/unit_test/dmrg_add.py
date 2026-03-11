import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def symm_type(pytestconfig):
    return pytestconfig.getoption("symm")


@pytest.fixture(scope="module")
def fd_data(pytestconfig):
    return pytestconfig.getoption("fd_data")

@pytest.fixture(scope="module", params=[2, 1])
def dot_type(request):
    return request.param

def _init_driver(tmp_path, symm_type, fd_data):
    symm = SymmetryTypes.SAnySU2 if symm_type == "sany" else SymmetryTypes.SU2
    driver = DMRGDriver(
        scratch=str(tmp_path / "nodex"), symm_type=symm, n_threads=4
    )

    if fd_data == "":
        from pyscf import gto, scf

        mol = gto.M(
            atom="N 0 0 0; N 0 0 1.1",
            basis="sto3g",
            symmetry="d2h",
            verbose=0,
        )
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            mf, 0, None
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
    else:
        driver.read_fcidump(filename=fd_data + "/N2.STO3G.RHF.FCIDUMP", pg="d2h")
        driver.initialize_system(
            n_sites=driver.n_sites,
            n_elec=driver.n_elec,
            spin=driver.spin,
            orb_sym=driver.orb_sym,
        )

    return driver


class TestDMRGAddition:
    def test_multi_addition_matches_sequential(self, tmp_path, symm_type, fd_data, dot_type):
        driver = _init_driver(tmp_path, symm_type, fd_data)

        try:
            driver.bw.b.Random.rand_seed(1234)
            impo = driver.get_identity_mpo()

            kets = [
                driver.get_random_mps(tag="ADD-KET-0", bond_dim=50, dot=dot_type),
                driver.get_random_mps(tag="ADD-KET-1", bond_dim=50, dot=dot_type),
                driver.get_random_mps(tag="ADD-KET-2", bond_dim=50, dot=dot_type),
            ]
            bra_template = driver.get_random_mps(tag="ADD-BRA-TPL", bond_dim=100, dot=dot_type)
            bra_multi = driver.copy_mps(bra_template, "ADD-BRA-MULTI")
            bra_seq01 = driver.copy_mps(bra_template, "ADD-BRA-SEQ-01")

            bra_bond_dims = [100] * 6
            ket_bond_dimss = [[100] * 6, [100], [100]]
            coeffs = [1.0, -0.5, 0.25]

            driver.multi_addition(
                bra_multi,
                kets,
                mpos=coeffs,
                n_sweeps=6,
                tol=1e-10,
                bra_bond_dims=bra_bond_dims,
                ket_bond_dimss=ket_bond_dimss,
                cutoff=1e-24,
                iprint=0,
            )
            driver.addition(
                bra_seq01,
                kets[0],
                kets[1],
                mpo_a=coeffs[0],
                mpo_b=coeffs[1],
                n_sweeps=6,
                tol=1e-10,
                bra_bond_dims=bra_bond_dims,
                ket_a_bond_dims=ket_bond_dimss[0],
                ket_b_bond_dims=ket_bond_dimss[1],
                cutoff=1e-24,
                iprint=0,
            )

            bra_seq = driver.copy_mps(bra_seq01, "ADD-BRA-SEQ")
            driver.addition(
                bra_seq,
                bra_seq01,
                kets[2],
                mpo_a=1.0,
                mpo_b=coeffs[2],
                n_sweeps=6,
                tol=1e-10,
                bra_bond_dims=bra_bond_dims,
                ket_a_bond_dims=bra_bond_dims,
                ket_b_bond_dims=ket_bond_dimss[2],
                cutoff=1e-24,
                iprint=0,
            )

            overlap = driver.expectation(bra_multi, impo, bra_seq, iprint=0)
            norm_multi = driver.expectation(bra_multi, impo, bra_multi, iprint=0)
            norm_seq = driver.expectation(bra_seq, impo, bra_seq, iprint=0)
            fidelity = abs(overlap) / np.sqrt(abs(norm_multi * norm_seq))
            rel_norm_diff = abs(norm_multi - norm_seq) / max(1.0, abs(norm_seq))

            assert fidelity > 1 - [0, 1E-4, 1E-8][dot_type]
            assert rel_norm_diff < [0, 1E-4, 1E-8][dot_type]

            driver.bw.b.Random.rand_seed(4321)
            smoke_kets = [
                driver.get_random_mps(tag="ADD-SMOKE-0", bond_dim=50, dot=dot_type),
                driver.get_random_mps(tag="ADD-SMOKE-1", bond_dim=50, dot=dot_type),
            ]
            bra_smoke = driver.copy_mps(bra_template, "ADD-BRA-SMOKE")
            norm_smoke = driver.multi_addition(
                bra_smoke,
                smoke_kets,
                n_sweeps=2,
                tol=1e-8,
                bra_bond_dims=[100] * 2,
                ket_bond_dimss=[[100] * 2, [100]],
                cutoff=1e-24,
                iprint=0,
            )

            assert np.isfinite(np.real(norm_smoke))
        finally:
            driver.finalize()

    def test_multi_addition_rejects_mismatched_dot(self, tmp_path, symm_type, fd_data):
        driver = _init_driver(tmp_path, symm_type, fd_data)

        try:
            driver.bw.b.Random.rand_seed(2468)
            bra = driver.get_random_mps(tag="DOT-BRA", bond_dim=100, dot=2)
            kets = [
                driver.get_random_mps(tag="DOT-KET-0", bond_dim=100, dot=2),
                driver.get_random_mps(tag="DOT-KET-1", bond_dim=100, dot=1),
            ]

            with pytest.raises(RuntimeError, match="same dot"):
                driver.multi_addition(bra, kets, n_sweeps=1, iprint=0)
        finally:
            driver.finalize()
