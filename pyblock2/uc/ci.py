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

"""Arbitrary order Configuration Interaction."""


class CI(lib.StreamObject):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, ci_order=2):
        from pyscf.scf import hf, rohf, addons

        if isinstance(mf, hf.KohnShamDFT):
            raise RuntimeError("CI Warning: The first argument mf is a DFT object.")
        if isinstance(mf, rohf.ROHF):
            lib.logger.warn(mf, 'RCI method does not support ROHF method. ROHF object '
                            'is converted to UHF object and UCI method is called.')
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
        self.ci_order = ci_order

    @property
    def e_tot(self):
        return np.asarray(self.e_corr) + self.e_hf

    def kernel(self, ci0=None, max_cycle=50, tol=1e-8):
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
            h1e = tuple(np.ascontiguousarray(x) for x in h1e)
            g2e_aa, g2e_ab, g2e_bb = tuple(
                np.ascontiguousarray(x) for x in (g2e_aa, g2e_ab, g2e_bb)
            )
            bs = b.sz
            VectorSX = b.VectorSZ
            target = b.SZ(n_elec, spin, 0)
            vacuum = b.SZ(0)
            fd = b.FCIDUMP()
            fd.initialize_sz(
                ncas, n_elec, spin, 0, ecore, h1e, (g2e_aa, g2e_bb, g2e_ab)
            )

        fd.symmetrize(b.VectorUInt8(orb_sym))
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
        for n in range(2, self.ci_order + 1):
            ket = ket0 ^ n
            if ci0 is not None and n == 2:
                kcis = (
                    list(ci0)[: self.nroots]
                    if isinstance(ci0, list) or isinstance(ci0, tuple)
                    else [ci0]
                )
            else:
                kcis = [np.zeros((len(ket),)) for _ in pkcis]
                for kci, pkci in zip(kcis, pkcis):
                    kci[ket >> pket] = pkci
            if len(kcis) < self.nroots:
                for _ in range(1, self.nroots):
                    kcis.append(np.concatenate([kcis[-1][-1:], kcis[-1][:-1]]))
            hamil = Hamiltonian(big, ket)
            eners, ndav = b.IterativeMatrixFunctions.davidson(
                hamil.__matmul__,
                hamil.diag(),
                kcis,
                self.verbose >= 4,
                conv_thrd=tol ** 2,
                soft_max_iter=max_cycle,
                max_iter=max_cycle + 100,
            )
            lib.logger.note(
                self,
                "E(%4s) =%s  E_corr =%s",
                "CI%d" % n,
                "%20.16g" * self.nroots % tuple(x + ecore for x in eners),
                "%20.16g" * self.nroots % tuple(x + ecore - self.e_hf for x in eners),
            )
            pkcis, pket = kcis, ket
            self.e_corr = [x + ecore - self.e_hf for x in eners]
            if self.nroots == 1:
                self.e_corr = self.e_corr[0]
            self.ci = kcis if self.nroots > 1 else kcis[0]
            conv = ndav < max_cycle

        if self.ci_order == 0:
            lib.logger.note(
                self,
                "E(%4s) =%s",
                "CI%d" % 0,
                "%20.16g" * self.nroots % ((self.e_hf,) * self.nroots),
            )

        return conv, self.e_corr + self.e_hf, self.ci


if __name__ == "__main__":

    from pyscf import gto, scf, ci

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", symmetry="d2h", verbose=3)
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    ci.CISD(mf).run()
    CI(mf, ci_order=2).run()

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="ccpvdz", symmetry="c1", verbose=3, spin=2)
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    ci.CISD(mf).run()
    CI(mf, ci_order=2).run()

    mf = scf.UHF(mol).run(conv_tol=1e-14)
    ci.CISD(mf).run()
    CI(mf, ci_order=2).run()
