#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2023 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

from pyscf import lib
import numpy as np

"""Arbitrary order Møller-Plesset Perturbation Theory."""


class MP(lib.StreamObject):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mp_order=2):
        from pyscf.scf import hf, addons

        if isinstance(mf, hf.KohnShamDFT):
            raise RuntimeError("MP Warning: The first argument mf is a DFT object.")
        if isinstance(mf, scf.rohf.ROHF):
            lib.logger.warn(
                mf,
                "RMP method does not support ROHF method. ROHF object "
                "is converted to UHF object and UMP method is called.",
            )
            mf = addons.convert_to_uhf(mf)
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.nroots = 1
        self.frozen = frozen
        self.converged = False
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_hf = None
        self.e_corr = None
        self.ci = None
        self.mp_order = mp_order

    @property
    def e_tot(self):
        return np.asarray(self.e_corr) + self.e_hf

    def kernel(self, max_cycle=50, tol=1e-8):
        is_rhf = self.mo_coeff is None or (
            isinstance(self.mo_coeff, np.ndarray)
            and self.mo_coeff.ndim == 2
            and self.mo_coeff.shape[0] == self.mol.nao
        )
        from pyblock2._pyscf.ao2mo import integrals as itg
        import block2 as b

        if is_rhf:
            ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
                self._scf,
                ncore=0 if self.frozen is None else self.frozen,
                ncas=None,
                g2e_symm=1,
            )
            nc, nv = (n_elec - spin) // 2, ncas - (n_elec + spin) // 2
            idx = np.concatenate(
                [np.mgrid[:ncas][ncas - nv :], np.mgrid[:ncas][: ncas - nv]]
            )
            orb_sym = np.array(orb_sym)[idx]
            h1e = h1e[idx, :][:, idx]
            g2e = g2e[idx, :][:, idx][:, :, idx][:, :, :, idx]
            bs = b.su2
            VectorSX = b.VectorSU2
            target = b.SU2(n_elec, spin, 0)
            vacuum = b.SU2(0)
            fd = b.FCIDUMP()
            fd.initialize_su2(ncas, n_elec, spin, 0, ecore, h1e, g2e)

            f1e = (
                h1e
                + 2.0 * np.einsum("mnjj->mn", g2e[:, :, nv:, nv:])
                - 1.0 * np.einsum("mjjn->mn", g2e[:, nv:, nv:, :])
            )
            e0 = ecore + 2.0 * np.einsum("jj->", f1e[nv:, nv:])
            e1 = -2.0 * np.einsum("iijj->", g2e[nv:, nv:, nv:, nv:]) + 1.0 * np.einsum(
                "ijji->", g2e[nv:, nv:, nv:, nv:]
            )

            fd0 = b.FCIDUMP()
            fd0.initialize_su2(ncas, n_elec, spin, 0, ecore - e0, f1e, 0.0 * g2e)
            fd1 = b.FCIDUMP()
            fd1.initialize_su2(ncas, n_elec, spin, 0, 0.0, h1e - f1e, g2e)
        else:
            ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_uhf_integrals(
                self._scf,
                ncore=0 if self.frozen is None else self.frozen,
                ncas=None,
                g2e_symm=1,
            )
            nc, nv = (n_elec - abs(spin)) // 2, ncas - (n_elec + abs(spin)) // 2
            idx = np.concatenate(
                [np.mgrid[:ncas][ncas - nv :], np.mgrid[:ncas][: ncas - nv]]
            )
            orb_sym = np.array(orb_sym)[idx]
            h1e = tuple(x[idx, :][:, idx] for x in h1e)
            g2e_aa, g2e_ab, g2e_bb = tuple(
                x[idx, :][:, idx][:, :, idx][:, :, :, idx] for x in g2e
            )
            bs = b.sz
            VectorSX = b.VectorSZ
            target = b.SZ(n_elec, spin, 0)
            vacuum = b.SZ(0)
            g2e = g2e_aa, g2e_bb, g2e_ab
            h1e = tuple(np.ascontiguousarray(x) for x in h1e)
            g2e = tuple(
                np.ascontiguousarray(x) for x in g2e
            )
            fd = b.FCIDUMP()
            fd.initialize_sz(ncas, n_elec, spin, 0, ecore, h1e, g2e)

            if spin == 0:
                nva = slice(nv, None)
                nvb = slice(nv, None)
            elif spin > 0:
                nva = slice(nv, None)
                nvb = slice(nv, nv + nc)
            else:
                nva = slice(nv, nv + nc)
                nvb = slice(nv, None)

            f1e = (
                h1e[0]
                + np.einsum("mnjj->mn", g2e[0][:, :, nva, nva])
                + np.einsum("mnjj->mn", g2e[2][:, :, nvb, nvb])
                - np.einsum("mjjn->mn", g2e[0][:, nva, nva, :]),
                h1e[1]
                + np.einsum("mnjj->mn", g2e[1][:, :, nvb, nvb])
                + np.einsum("jjmn->mn", g2e[2][nva, nva, :, :])
                - np.einsum("mjjn->mn", g2e[1][:, nvb, nvb, :]),
            )
            e0 = (
                ecore
                + np.einsum("jj->", f1e[0][nva, nva])
                + np.einsum("jj->", f1e[1][nvb, nvb])
            )
            e1 = (
                -0.5 * np.einsum("iijj->", g2e[0][nva, nva, nva, nva])
                - 0.5 * np.einsum("iijj->", g2e[1][nvb, nvb, nvb, nvb])
                - 1.0 * np.einsum("iijj->", g2e[2][nva, nva, nvb, nvb])
                + 0.5 * np.einsum("ijji->", g2e[0][nva, nva, nva, nva])
                + 0.5 * np.einsum("ijji->", g2e[1][nvb, nvb, nvb, nvb])
            )

            f1e = tuple(np.ascontiguousarray(x) for x in f1e)

            fd0 = b.FCIDUMP()
            fd0.initialize_sz(
                ncas, n_elec, spin, 0, ecore - e0, f1e, tuple(0.0 * x for x in g2e)
            )
            fd1 = b.FCIDUMP()
            fd1.initialize_sz(
                ncas, n_elec, spin, 0, 0.0, (h1e[0] - f1e[0], h1e[1] - f1e[1]), g2e
            )

        fd.symmetrize(b.VectorUInt8(orb_sym))
        fd0.symmetrize(b.VectorUInt8(orb_sym))
        fd1.symmetrize(b.VectorUInt8(orb_sym))
        big = bs.DRTBigSite(
            VectorSX([target]), False, ncas, b.VectorUInt8(orb_sym), fd, 0
        )
        big.drt = bs.DRT(
            big.drt.n_sites,
            big.drt.get_init_qs(),
            big.drt.orb_sym,
            nc,
            nv,
            n_ex=n_elec,
            nc_ref=0,
            single_ref=True,
        )

        big0 = bs.DRTBigSite(
            VectorSX([target]), False, ncas, b.VectorUInt8(orb_sym), fd0, 0
        )
        big0.drt = big.drt
        big1 = bs.DRTBigSite(
            VectorSX([target]), False, ncas, b.VectorUInt8(orb_sym), fd1, 0
        )
        big1.drt = big.drt

        if self.verbose >= 5:
            print(big.drt)

        class Hamiltonian:
            def __init__(self, big, drt):
                big.drt = drt
                hamil_op = bs.OpElement(b.OpNames.H, b.SiteIndex(), vacuum)
                hmat = big.get_site_op(0, hamil_op)[0]
                self.hmat = hmat

            def __matmul__(self, kci: np.ndarray) -> np.ndarray:
                bci = np.zeros((self.hmat.n,), dtype=kci.dtype)
                b.CSRMatrixFunctions.multiply(
                    self.hmat,
                    0,
                    b.Matrix(kci.reshape(-1, 1)),
                    0,
                    b.Matrix(bci.reshape(-1, 1)),
                    1.0,
                    0.0,
                )
                return bci

            def diag(self) -> np.ndarray:
                dci = np.zeros((self.hmat.n,), dtype=float)
                self.hmat.diag(b.Matrix(dci.reshape(-1, 1)))
                return dci

        ket0 = big.drt
        pkcis, pket = [np.ones((len(ket0 ^ 0),))], ket0 ^ 0
        self.e_hf = np.dot(pkcis[0], Hamiltonian(big, pket) @ pkcis[0]) + ecore
        self.e_corr = 0
        conv = False

        ket0 = big.drt
        ecorrs = [e0, e1]
        kets = [ket0 ^ 0]
        wfns = [np.ones((len(kets[0]),))]
        for n in range(1, self.mp_order // 2 + 1):
            new_ket = ket0 ^ (2 * n)
            new_wfns = []
            for wfn in wfns:
                nwfn = np.zeros((len(new_ket),))
                nwfn[new_ket >> kets[-1]] = wfn
                new_wfns.append(nwfn)
            wfns = new_wfns
            kets.append(new_ket)
            h0 = Hamiltonian(big0, new_ket)
            h1 = Hamiltonian(big1, new_ket)
            rwfn = h1 @ wfns[-1]
            rwfn = rwfn - np.dot(rwfn, wfns[0]) * wfns[0]
            for k in range(1, n):
                rwfn -= ecorrs[k] * wfns[-k]
            nwfn = wfns[-1].copy()
            _, niter = b.IterativeMatrixFunctions.conjugate_gradient(
                h0.__matmul__,
                h0.diag() + fd0.const_e,
                nwfn,
                rwfn,
                consta=fd0.const_e,
                iprint=self.verbose >= 4,
                conv_thrd=tol ** 2,
                soft_max_iter=max_cycle,
                max_iter=max_cycle + 100,
            )
            nwfn = -nwfn + np.dot(nwfn, wfns[0]) * wfns[0]
            ecorr = [np.dot(wfns[-1], h1 @ nwfn), np.dot(nwfn, h1 @ nwfn)]
            wfns.append(nwfn)
            for i in [0, 1]:
                for k in range(1, 2 * n + i - 1):
                    for m in range(
                        max(1, n - k + i), min(n + i - 1, 2 * n + i - k - 1) + 1
                    ):
                        ecorr[i] -= ecorrs[k] * np.dot(wfns[m], wfns[2 * n + i - k - m])
            ecorrs += ecorr
            lib.logger.note(
                self,
                "E(%4s) =%s  E_corr =%s",
                "MP%d" % (2 * n),
                "%20.16g" % sum(ecorrs[: 2 * n + 1]),
                "%20.16g" % (sum(ecorrs[: 2 * n + 1]) - self.e_hf),
            )
            if 2 * n + 1 <= self.mp_order:
                lib.logger.note(
                    self,
                    "E(%4s) =%s  E_corr =%s",
                    "MP%d" % (2 * n + 1),
                    "%20.16g" % sum(ecorrs[: 2 * n + 2]),
                    "%20.16g" % (sum(ecorrs[: 2 * n + 2]) - self.e_hf),
                )

            self.e_corr = sum(ecorrs[: self.mp_order + 1]) - self.e_hf
            conv = niter < max_cycle
            self.ci = wfns
        
        if self.mp_order == 1:
            lib.logger.note(
                self,
                "E(%4s) =%s",
                "MP%d" % 1,
                "%20.16g" % sum(ecorrs[:2]),
            )

        return conv, self.e_corr + self.e_hf, self.ci


if __name__ == "__main__":

    from pyscf import gto, scf, mp

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", symmetry="d2h", verbose=3)
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mp.MP2(mf).run()
    MP(mf, mp_order=2).run()

    mol = gto.M(
        atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", symmetry="c1", verbose=3, spin=2
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mp.MP2(mf).run()
    MP(mf, mp_order=2).run() # not okay

    mol = gto.M(
        atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", symmetry="c1", verbose=3, spin=2
    )
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    mp.MP2(mf).run()
    MP(mf, mp_order=2).run()
