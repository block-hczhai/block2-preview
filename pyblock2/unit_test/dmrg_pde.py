import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from pyblock2.algebra.io import MPOTools, MPSTools
from pyblock2.algebra.pde import PDETools1D, PDEToolsND


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _laplacian_1d_matrix(npts, dx):
    lap = sparse.diags(
        [np.ones(npts - 1), -2 * np.ones(npts), np.ones(npts - 1)],
        [-1, 0, 1],
        shape=(npts, npts),
        format="lil",
    )
    lap[0, -1] = 1.0
    lap[-1, 0] = 1.0
    return lap.tocsr() / dx ** 2


def _periodic_laplacian(values, dxs):
    result = np.zeros_like(values)
    for axis, dx in enumerate(dxs):
        result += (
            np.roll(values, -1, axis=axis)
            - 2 * values
            + np.roll(values, 1, axis=axis)
        ) / dx ** 2
    return result


def _rk4_evolve(operator, initial, dt, nt, source=None):
    source = np.zeros_like(initial) if source is None else source
    state = initial.copy()
    for _ in range(nt):
        k1 = operator @ state + source
        k2 = operator @ (state + 0.5 * dt * k1) + source
        k3 = operator @ (state + 0.5 * dt * k2) + source
        k4 = operator @ (state + dt * k3) + source
        state = state + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
    return state


def _to_block2_mps(pde, pyket, tag, dot=2):
    ket = MPSTools.to_block2(pyket, pde.driver.basis, tag=tag)
    return pde.driver.adjust_mps(ket, dot=dot)[0]


def _run_td_dmrg(driver, mpo, ket, dt, nt, kernel, ext_mpss=None, bond_dim=40):
    tket = driver.copy_mps(ket, tag="TKET")
    for _ in range(nt):
        tket = driver.td_dmrg(
            mpo,
            tket,
            -dt,
            -dt,
            final_mps_tag="TKET",
            n_sub_sweeps=2,
            te_type="rk4",
            kernel=kernel,
            ext_mpss=ext_mpss,
            cutoff=1e-24,
            hermitian=False,
            normalize_mps=False,
            bond_dims=[bond_dim],
            iprint=0,
        )
    tket = driver.adjust_mps(tket, dot=1)[0]
    return MPSTools.from_block2(tket)


def _morse_potential(coords, beta, de):
    return de * (np.exp(-2 * beta * coords) - 2 * np.exp(-beta * coords))


def test_pdetoolsnd_site_layouts():
    interleaved = PDEToolsND(
        n_pts=2,
        nd=2,
        xi=[0.0, -1.0],
        xf=[1.0, 2.0],
        bases=[[2, 3], [5, 7]],
        site_order="interleaved",
    )
    blocked = PDEToolsND(
        n_pts=2,
        nd=2,
        xi=[0.0, -1.0],
        xf=[1.0, 2.0],
        bases=[[2, 3], [5, 7]],
        site_order="blocked",
    )

    assert interleaved.n_sites == 4
    assert interleaved.bases == [2, 5, 3, 7]
    assert interleaved.site_schedule == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert interleaved.axis_sites == [[0, 2], [1, 3]]
    np.testing.assert_allclose(interleaved.dx, [1.0 / 6.0, 3.0 / 35.0])

    assert blocked.bases == [2, 3, 5, 7]
    assert blocked.site_schedule == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert blocked.axis_sites == [[0, 1], [2, 3]]
    np.testing.assert_allclose(blocked.dx, [1.0 / 6.0, 3.0 / 35.0])


def test_pdetoolsnd_range_and_exponential_are_layout_independent():
    interleaved = PDEToolsND(
        n_pts=2,
        nd=2,
        xi=[0.0, -1.0],
        xf=[1.0, 1.0],
        bases=2,
        site_order="interleaved",
    )
    blocked = PDEToolsND(
        n_pts=2,
        nd=2,
        xi=[0.0, -1.0],
        xf=[1.0, 1.0],
        bases=2,
        site_order="blocked",
    )

    range_args = ([0.25, -0.5], [0.75, 0.0], 3.0)
    icoords, irange = interleaved.pymps_rasterize(interleaved.pymps_from_range(*range_args))
    bcoords, brange = blocked.pymps_rasterize(blocked.pymps_from_range(*range_args))
    for xa, xb in zip(icoords, bcoords):
        np.testing.assert_allclose(xa, xb)
    np.testing.assert_allclose(irange, brange)

    expected_range = np.where(
        (
            (icoords[0][:, None] >= range_args[0][0])
            & (icoords[0][:, None] <= range_args[1][0])
            & (icoords[1][None, :] >= range_args[0][1])
            & (icoords[1][None, :] <= range_args[1][1])
        ),
        range_args[2],
        0.0,
    )
    np.testing.assert_allclose(irange, expected_range)

    exp_args = (np.e, [0.5, -0.75])
    _, iexp = interleaved.pymps_rasterize(interleaved.pymps_from_exponential(*exp_args))
    _, bexp = blocked.pymps_rasterize(blocked.pymps_from_exponential(*exp_args))
    np.testing.assert_allclose(iexp, bexp)

    expected_exp = np.e ** (
        exp_args[1][0] * icoords[0][:, None] + exp_args[1][1] * icoords[1][None, :]
    )
    np.testing.assert_allclose(iexp, expected_exp)


@pytest.mark.parametrize("site_order", ["interleaved", "blocked"])
def test_pdetoolsnd_laplacian_matches_numpy(tmp_path, site_order):
    pde = PDEToolsND(
        n_pts=2,
        nd=2,
        xi=[0.0, 0.0],
        xf=[1.0, 1.0],
        bases=2,
        site_order=site_order,
    )
    pde.init_dmrg_driver(scratch=str(tmp_path / site_order))
    try:
        state = pde.pymps_from_exponential(np.e, [0.5, -0.75])
        _, values = pde.pymps_rasterize(state)
        mpo = pde.pympo_from_differential([[0, 0, 1], [0, 0, 1]], pbc=True)
        _, applied = pde.pymps_rasterize(mpo @ state)
    finally:
        pde.driver.finalize()

    expected = _periodic_laplacian(values, pde.dx)
    np.testing.assert_allclose(applied, expected)


def test_dmrg_pde_time_dependent_example_matches_finite_difference(tmp_path):
    pde = PDETools1D(5, xi=-1.0, xf=1.0, bases=2)
    pde.init_dmrg_driver(scratch=str(tmp_path / "td"))
    driver = pde.driver
    nu = 0.1
    dt = 0.002
    nt = 8

    try:
        pympo = pde.pympo_from_differential([0, 0, 1], pbc=True) * nu
        mpo = MPOTools.to_block2(pympo, driver.basis, add_ident=True)

        pyket = pde.pymps_from_range(-0.2, 0.2, 1.0)
        ket = _to_block2_mps(pde, pyket, "KET")

        def kernel_a(beta, hop, a, b, xs):
            hop(a, b, beta)

        pyout_a = _run_td_dmrg(driver, mpo, ket, dt, nt, kernel_a)
    finally:
        driver.finalize()

    x, initial = pde.pymps_rasterize(pyket)
    lap = nu * _laplacian_1d_matrix(len(x), pde.dx)
    expected_a = _rk4_evolve(lap, initial, dt, nt)
    _, actual_a = pde.pymps_rasterize(pyout_a)

    np.testing.assert_allclose(actual_a, expected_a, atol=1e-6, rtol=1e-6)


def test_dmrg_pde_schrodinger_example_matches_sparse_eigensolver(tmp_path):
    pde = PDETools1D(5, xi=-2.0, xf=7.0, bases=2)
    pde.init_dmrg_driver(scratch=str(tmp_path / "eig"))
    driver = pde.driver
    beta = 0.5
    mu = 6.0
    de = 12.0
    nroots = 2

    try:
        potential_mps = de * (
            pde.pymps_from_exponential(np.e, -2 * beta)
            - 2 * pde.pymps_from_exponential(np.e, -beta)
        )
        pympo = (-1.0 / (2.0 * mu)) * pde.pympo_from_differential([0, 0, 1], pbc=True)
        pympo = pympo + potential_mps.diag()
        mpo = MPOTools.to_block2(pympo, driver.basis, add_ident=True)

        kets = [driver.get_random_mps(tag="KET%d" % i, bond_dim=1) for i in range(nroots)]
        energies = []
        for ir in range(nroots):
            energy = driver.dmrg(
                mpo,
                kets[ir],
                n_sweeps=8,
                bond_dims=[24] * 8,
                noises=[0] * 8,
                thrds=[0] * 8,
                dav_max_iter=200,
                iprint=0,
                proj_mpss=kets[:ir],
                proj_weights=[10.0] * ir,
                dav_rel_conv_thrd=1e-7,
                tol=1e-9,
                cutoff=1e-30,
            )
            energies.append(energy)
    finally:
        driver.finalize()

    x = np.linspace(pde.xi, pde.xf, 2 ** pde.n_sites + 1)[:-1]
    lap = _laplacian_1d_matrix(len(x), pde.dx)
    potential = sparse.diags(_morse_potential(x, beta, de), format="csr")
    dense_h = (-1.0 / (2.0 * mu)) * lap + potential
    expected, _ = spla.eigsh(dense_h, k=nroots, which="SA", tol=1e-10)
    expected = np.sort(expected)

    np.testing.assert_allclose(np.sort(energies), expected, atol=5e-7, rtol=1e-6)


def test_dmrg_pde_poisson_example_matches_constrained_sparse_solve(tmp_path):
    pde = PDETools1D(5, xi=0.0, xf=1.0, bases=2)
    pde.init_dmrg_driver(scratch=str(tmp_path / "poisson"))
    driver = pde.driver

    try:
        pympo = pde.pympo_from_differential([0, 0, 1], pbc=True)
        mpo = MPOTools.to_block2(pympo, driver.basis, add_ident=True)
        pyket = pde.pymps_from_range(0.3, 0.3, 1.0) + pde.pymps_from_range(0.60, 0.60, 1.0)
        pypket = pde.pymps_from_range(0.0, 0.0, 1.0)
        ket = _to_block2_mps(pde, pyket, "KET")
        pket = _to_block2_mps(pde, pypket, "PKET")
        bra = driver.get_random_mps(tag="BRA", bond_dim=4, nroots=1)
        impo = driver.get_identity_mpo()
        driver.multiply(
            bra,
            impo,
            ket,
            n_sweeps=8,
            bond_dims=[24] * 8,
            noises=[0] * 8,
            thrds=[0] * 8,
            tol=0.0,
            left_mpo=mpo,
            linear_max_iter=200,
            linear_rel_conv_thrd=1e-8,
            cutoff=1e-24,
            solver_type="CG",
            proj_mpss=[pket],
            proj_weights=[1.0],
            proj_bond_dim=48,
            iprint=0,
        )
        bra = driver.adjust_mps(bra, dot=1)[0]
        pybra = MPSTools.from_block2(bra)
    finally:
        driver.finalize()

    _, source = pde.pymps_rasterize(pyket)
    _, actual = pde.pymps_rasterize(pybra)
    lap = _laplacian_1d_matrix(len(source), pde.dx).tolil()
    rhs = source.copy()
    lap[0, :] = 0.0
    lap[0, 0] = 1.0
    rhs[0] = 0.0
    expected = spla.spsolve(lap.tocsr(), rhs)

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
