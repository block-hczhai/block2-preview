import pytest
import numpy as np
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, OrbitalEntropy

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def symm_type(pytestconfig):
    return pytestconfig.getoption("symm")


@pytest.fixture(scope="module", params=[0, 2], ids=["singlet", "triplet"])
def spin(request):
    return request.param


# bra/ket one-orbital operators (see OrbitalEntropy.ops) of the 36 entries
# of the two-orbital RDM (Table 3. J. Chem. Theory Comput. 2013, 9, 2959-2973)
LAYOUT = (
    "1/1 1/6 2/5 5/2 6/1 1/11 3/9 9/3 11/1 6/6 1/16 2/15 3/14 4/13 5/12 6/11 7/10 "
    + "8/9 9/8 10/7 11/6 12/5 13/4 14/3 15/2 16/1 11/11 6/16 8/14 14/8 16/6 11/16 "
    + "12/15 15/12 16/11 16/16"
)
LAYOUT = [[int(v) for v in u.split("/")] for u in LAYOUT.split()]
# the 26 entries used with block_symm=True are the upper triangles of the blocks
BLOCK_SYMM_IDX = []
ip = 0
for d in [1, 2, 2, 1, 4, 1, 2, 2, 1]:
    BLOCK_SYMM_IDX += [ip + i * d + j for i in range(d) for j in range(i, d)]
    ip += d * d


def get_fci_two_orb_rdm(civec, norb, nelec, i, j):
    """Two-orbital RDM entries <psi| |a_i a_j><b_i b_j| |psi> in the basis
    |s_i s_j> = C(s_i)_i C(s_j)_j |vac>, s = 0, dn, up, updn,
    with C(0) = 1, C(dn) = a+_dn, C(up) = a+_up, C(updn) = a+_up a+_dn."""
    from pyscf.fci import addons, cistring

    fops = {
        ("c", "a"): addons.cre_a,
        ("c", "b"): addons.cre_b,
        ("d", "a"): addons.des_a,
        ("d", "b"): addons.des_b,
    }
    cre = [[], ["b"], ["a"], ["a", "b"]]

    def apply(t, s, p, x):
        ci, na, nb = x
        dna, dnb = (1 if t == "c" else -1) * (s == "a"), (1 if t == "c" else -1) * (s == "b")
        if ci is None or not (0 <= na + dna <= norb and 0 <= nb + dnb <= norb):
            return None, na + dna, nb + dnb
        return fops[(t, s)](ci, norb, (na, nb), p), na + dna, nb + dnb

    def project_vac(x):
        ci, na, nb = x
        if ci is None:
            return x
        ci = ci.copy()
        for k, nx in enumerate([na, nb]):
            strs = cistring.make_strings(range(norb), nx)
            occ = ((strs >> i) & 1) | ((strs >> j) & 1)
            if k == 0:
                ci[occ != 0, :] = 0
            else:
                ci[:, occ != 0] = 0
        return ci, na, nb

    rdm = np.zeros(len(LAYOUT))
    for k, (ix, iy) in enumerate(LAYOUT):
        (ai, bi), (aj, bj) = divmod(ix - 1, 4), divmod(iy - 1, 4)
        x = (civec, *nelec)
        for p, b in [(i, bi), (j, bj)]:
            for s in cre[b]:
                x = apply("d", s, p, x)
        x = project_vac(x)
        for p, a in [(j, aj), (i, ai)]:
            for s in cre[a][::-1]:
                x = apply("c", s, p, x)
        if x[0] is not None and x[1:] == tuple(nelec):
            rdm[k] = np.dot(civec.ravel(), x[0].ravel())
    return rdm


class TestOrbRDM:
    def test_two_orb_rdm(self, tmp_path, symm_type, spin):
        if symm_type == "sany":
            pytest.skip("two-orbital RDMs are tested in the SZ mode")

        from pyscf import gto, scf, fci

        mol = gto.M(
            atom=[["H", (0, 0, x)] for x in [0.0, 1.0, 2.3, 3.2]],
            basis="sto3g",
            verbose=0,
        )
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        ncas, n_elec, _, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
            mf, ncore=0, ncas=None, g2e_symm=1
        )
        nelec = ((n_elec + spin) // 2, (n_elec - spin) // 2)
        efci, civec = fci.direct_spin1.FCI().kernel(
            h1e, g2e, ncas, nelec, ecore=ecore, conv_tol=1e-14
        )

        driver = DMRGDriver(
            scratch=str(tmp_path / "nodex"), symm_type=SymmetryTypes.SZ, n_threads=4
        )
        driver.initialize_system(
            n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        ket = driver.get_random_mps(tag="GS", bond_dim=50, nroots=1)
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=[50] * 20,
            noises=[1e-4] * 6 + [0] * 14,
            thrds=[1e-12] * 20,
            iprint=0,
        )
        assert abs(energy - efci) < 1e-10

        # NPDM-based two-orbital RDMs (as in get_orbital_entropies_use_npdm)
        exprs, nx = OrbitalEntropy.get_two_orb_rdm_exprs()
        pdms = driver.get_npdm(
            ket,
            pdm_type=[len(k) // 2 for (k, _), _ in exprs.items()],
            npdm_expr=[k for (k, _), _ in exprs.items()],
            mask=[list(m) for (_, m), _ in exprs.items()],
            iprint=0,
        )
        rdms = np.zeros((ncas, ncas, nx))
        for ((_, m), v), pdm in zip(exprs.items(), pdms):
            for ix, f in v:
                if len(set(m)) == 0:
                    rdms[..., ix] += pdm[None, None] * f
                elif len(set(m)) == 1 and m[0] == 0:
                    rdms[..., ix] += pdm[:, None] * f
                elif len(set(m)) == 1 and m[0] == 1:
                    rdms[..., ix] += pdm[None, :] * f
                else:
                    rdms[..., ix] += pdm * f

        # MPO-based two-orbital RDMs (as in get_orbital_entropies)
        mpos = driver.get_n_orb_rdm_mpos(orb_type=2, ij_symm=False)

        for i in range(ncas):
            for j in range(ncas):
                if i == j:
                    continue
                ref = get_fci_two_orb_rdm(civec, ncas, nelec, i, j)
                assert np.max(np.abs(rdms[i, j] - ref)) < 1e-6
                x = np.array([driver.expectation(ket, m, ket) for m in mpos[(i, j)]])
                assert np.max(np.abs(x - ref[BLOCK_SYMM_IDX])) < 1e-6
