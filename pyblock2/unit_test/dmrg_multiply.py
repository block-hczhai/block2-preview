import numpy as np
import pytest
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


def _init_driver_and_integrals(tmp_path, symm_type, fd_data):
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
        h1e, g2e, ecore = driver.h1e, driver.g2e, driver.ecore

    return driver, h1e, g2e, ecore


def _compare_states(driver, impo, lhs, rhs, dot_type):
    overlap = driver.expectation(lhs, impo, rhs, iprint=0)
    norm_lhs = driver.expectation(lhs, impo, lhs, iprint=0)
    norm_rhs = driver.expectation(rhs, impo, rhs, iprint=0)
    fidelity = abs(overlap) / np.sqrt(abs(norm_lhs * norm_rhs))
    rel_norm_diff = abs(norm_lhs - norm_rhs) / max(1.0, abs(norm_rhs))

    assert fidelity > 1 - [0, 1e-4, 1e-8][dot_type]
    assert rel_norm_diff < [0, 1e-4, 1e-8][dot_type]


def _run_multiply(
    driver,
    bra,
    mpo,
    ket,
    sweep_bond_dims,
    right_kernel=None,
    proj_mpss=None,
    proj_bond_dim=-1,
):
    kwargs = dict(
        n_sweeps=6,
        tol=1e-10,
        bond_dims=sweep_bond_dims,
        bra_bond_dims=sweep_bond_dims,
        cutoff=1e-24,
        iprint=0,
    )
    if right_kernel is not None:
        kwargs["right_kernel"] = right_kernel
    if proj_mpss is not None:
        kwargs["proj_mpss"] = proj_mpss
        kwargs["proj_weights"] = [1.0]
        kwargs["proj_bond_dim"] = proj_bond_dim
    return driver.multiply(bra, mpo, ket, **kwargs)


def _run_addition(driver, bra, ket_a, ket_b, sweep_bond_dims):
    return driver.addition(
        bra,
        ket_a,
        ket_b,
        mpo_a=1.0,
        mpo_b=1.0,
        n_sweeps=6,
        tol=1e-10,
        bra_bond_dims=sweep_bond_dims,
        ket_a_bond_dims=sweep_bond_dims,
        ket_b_bond_dims=sweep_bond_dims,
        cutoff=1e-24,
        iprint=0,
    )


class TestDMRGMultiply:
    def test_multiply_kernel_and_projection(
        self, tmp_path, symm_type, fd_data, dot_type
    ):
        driver, h1e, g2e, ecore = _init_driver_and_integrals(
            tmp_path, symm_type, fd_data
        )

        try:
            driver.bw.b.Random.rand_seed(1357)
            mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
            impo = driver.get_identity_mpo()

            ket_base = driver.get_random_mps(tag="MUL-KET-BASE", bond_dim=50, dot=dot_type)
            proj_base = driver.get_random_mps(tag="MUL-PROJ-BASE", bond_dim=50, dot=dot_type)
            bra_template = driver.get_random_mps(
                tag="MUL-BRA-TPL", bond_dim=200, dot=dot_type
            )

            work_bond_dim = max(
                bra_template.info.bond_dim,
                3 * ket_base.info.bond_dim,
                3 * proj_base.info.bond_dim,
            )
            sweep_bond_dims = [work_bond_dim] * 6

            ket_plain = driver.copy_mps(ket_base, "MUL-KET-PLAIN")
            bra_plain = driver.copy_mps(bra_template, "MUL-BRA-PLAIN")
            plain_norm = _run_multiply(
                driver, bra_plain, mpo, ket_plain, sweep_bond_dims
            )
            assert np.isfinite(np.abs(plain_norm))

            def scale_kernel(beta, hop, a, b, xs):
                hop(a, b, 1.5 * beta)

            ket_scale = driver.copy_mps(ket_base, "MUL-KET-SCALE")
            bra_scale = driver.copy_mps(bra_template, "MUL-BRA-SCALE")
            _run_multiply(
                driver,
                bra_scale,
                mpo,
                ket_scale,
                sweep_bond_dims,
                right_kernel=scale_kernel,
            )

            plain_scale_a = driver.copy_mps(bra_plain, "MUL-PLAIN-SCALE-A")
            plain_scale_b = driver.copy_mps(bra_plain, "MUL-PLAIN-SCALE-B")
            expected_scale_half = driver.copy_mps(
                bra_template, "MUL-EXPECTED-SCALE-HALF"
            )
            driver.addition(
                expected_scale_half,
                plain_scale_a,
                plain_scale_b,
                mpo_a=1.0,
                mpo_b=0.5,
                n_sweeps=6,
                tol=1e-10,
                bra_bond_dims=sweep_bond_dims,
                ket_a_bond_dims=sweep_bond_dims,
                ket_b_bond_dims=sweep_bond_dims,
                cutoff=1e-24,
                iprint=0,
            )
            _compare_states(driver, impo, bra_scale, expected_scale_half, dot_type)

            def proj_kernel(beta, hop, a, b, xs):
                hop(a, b, beta)
                b += beta * xs[0]

            ket_proj = driver.copy_mps(ket_base, "MUL-KET-PROJ")
            proj_kernel_mps = driver.copy_mps(proj_base, "MUL-PROJ-KERNEL")
            bra_proj = driver.copy_mps(bra_template, "MUL-BRA-PROJ")
            _run_multiply(
                driver,
                bra_proj,
                mpo,
                ket_proj,
                sweep_bond_dims,
                right_kernel=proj_kernel,
                proj_mpss=[proj_kernel_mps],
                proj_bond_dim=work_bond_dim,
            )

            plain_proj = driver.copy_mps(bra_plain, "MUL-PLAIN-PROJ")
            proj_expected = driver.copy_mps(proj_base, "MUL-PROJ-EXPECTED")
            expected_proj = driver.copy_mps(bra_template, "MUL-EXPECTED-PROJ")
            _run_addition(
                driver, expected_proj, plain_proj, proj_expected, sweep_bond_dims
            )
            _compare_states(driver, impo, bra_proj, expected_proj, dot_type)

            ket_ignore = driver.copy_mps(ket_base, "MUL-KET-IGNORE")
            proj_ignore = driver.copy_mps(proj_base, "MUL-PROJ-IGNORE")
            bra_ignore = driver.copy_mps(bra_template, "MUL-BRA-IGNORE")
            _run_multiply(
                driver,
                bra_ignore,
                mpo,
                ket_ignore,
                sweep_bond_dims,
                proj_mpss=[proj_ignore],
                proj_bond_dim=work_bond_dim,
            )
            _compare_states(driver, impo, bra_ignore, bra_plain, dot_type)
        finally:
            driver.finalize()
