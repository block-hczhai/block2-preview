
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2024 Huanchen Zhai <hczhai@caltech.edu>
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

"""
PDE Tools.
"""

class PDETools1D:
    def __init__(self, n_sites, xi=0.0, xf=1.0, bases=2):
        from functools import reduce
        if isinstance(bases, int):
            bases = [bases]
        if len(bases) != n_sites:
            bases = [bases[i % len(bases)] for i in range(n_sites)]
        self.n_sites = n_sites
        self.bases = bases
        self.xi = xi
        self.xf = xf
        self.dx = (xf - xi) / reduce(lambda x, y: x * y, bases)
        self.driver = None

    def init_dmrg_driver(self, **kwargs):
        import numpy as np
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        driver = DMRGDriver(symm_type=SymmetryTypes.SAny, **kwargs)
        driver.set_symmetry_groups()
        Q = driver.bw.SX
        site_ops = []
        for bz in self.bases:
            ops = {"": np.identity(bz), 'z': np.identity(bz)}
            for i in range(1, bz):
                ops[chr(ord('O') + i)] = np.diag(np.ones(bz - i), -i)
                ops[chr(ord('N') - i)] = np.diag(np.ones(bz - i), i)
            for i in range(0, bz):
                ops[chr(ord('a') + i)] = np.zeros((bz, bz))
                ops[chr(ord('a') + i)][i, i] = 1
            site_ops.append(ops)
        driver.initialize_system(n_sites=self.n_sites, vacuum=Q(), target=Q(), hamil_init=False)
        driver.ghamil = driver.get_custom_hamiltonian([[(Q(), bz)] for bz in self.bases], site_ops)
        self.driver = driver

    @staticmethod
    def trans_tensors_to_pymps(tensors):
        from pyblock2.algebra.core import SubTensor, Tensor, MPS
        from block2 import SAny
        tensors[0], tensors[-1] = tensors[0][0], tensors[-1][..., 0]
        q_labels = [(SAny(), ) * len(ts.shape) for ts in tensors]
        return MPS([Tensor([SubTensor(qs, ts)]) for qs, ts in zip(q_labels, tensors)])

    @staticmethod
    def trans_pymps_to_tensors(pyket):
        return [t.blocks[0].reduced for t in pyket.tensors]
    
    @staticmethod
    def shift_polynomial_x(coeffs, dx):
        import math
        new_coeffs = [0.0] * len(coeffs)
        for k, c in enumerate(coeffs):
            for j in range(k + 1):
                new_coeffs[j] += math.comb(k, j) * dx ** (k - j) * c
        return new_coeffs

    @staticmethod
    def solve_finite_diff(k, x=0):
        n, w = k + 1, -2 * x - k
        a, b = [[0] * n for _ in range(n)], [0] * n
        for i in range(n):
            b[i] = (x + n) ** i * (-w)
            for j in range(0, n):
                a[i][j] = (x + j) ** i
        b[n - 1] = (x + n) ** n * (-w)
        for j in range(0, n):
            a[n - 1][j] = (x + j) ** n
        for i in range(n):
            assert b[i] % a[i][i] == 0
            b[i] = b[i] // a[i][i]
            for j in range(i + 1, n):
                assert a[i][j] % a[i][i] == 0
                a[i][j] = a[i][j] // a[i][i]
            for j in range(i + 1, n):
                for k in range(i + 1, n):
                    a[j][k] -= a[j][i] * a[i][k]
                b[j] -= a[j][i] * b[i]
                a[j][i] = 0
        for i in range(n)[::-1]:
            for j in range(0, i):
                b[j] -= a[j][i] * b[i]
        return b + [w]

    def pymps_step_function(self, x0, y, forward=True):
        """f(x) = y * (x >= x0) (forward)
           f(x) = y * (x <= x0) (backward)"""
        import numpy as np
        dx = self.dx
        tensors = [np.zeros(((2, bz, 2))) for bz in self.bases]
        for i, bz in list(enumerate(self.bases))[::-1]:
            xx = int((x0 - self.xi) / dx) % bz
            tensors[i][1, :, 1] = 1
            tensors[i][0, xx, 0] = 1
            tensors[i][0, slice(xx + 1, None) if forward else slice(xx), 1] = 1
            if i == self.n_sites - 1:
                tensors[i] = tensors[i] @ np.array([y, y])[:, None]
            dx *= bz
        return PDETools1D.trans_tensors_to_pymps(tensors)

    def pymps_from_range(self, xa, xb, y):
        """f(x) = y * (xa <= x <= xb)"""
        keta = self.pymps_step_function(xa, y, forward=True)
        ketb = self.pymps_step_function(xb, 1.0, forward=False)
        return keta.diag() @ ketb

    def pymps_from_polynomial(self, coeffs):
        """f(x) = sum_i (coeffs[i] * x^i)"""
        import numpy as np, math
        dx, coeffs = self.dx, PDETools1D.shift_polynomial_x(coeffs, self.xi)
        tensors = [np.zeros(((len(coeffs), bz, len(coeffs)))) for bz in self.bases]
        for i, bz in list(enumerate(self.bases))[::-1]:
            for j in range(1 if i == 0 else len(coeffs)):
                for k in range(j, len(coeffs)):
                    tensors[i][j, :, k] += math.comb(k, j) * (np.mgrid[:bz] * dx) ** (k - j)
            if i == self.n_sites - 1:
                tensors[i] = tensors[i] @ np.array(coeffs)[:, None]
            dx *= bz
        return PDETools1D.trans_tensors_to_pymps(tensors)

    def pymps_from_trigonometric(self, alpha, phi=0.0):
        """f(x) = sin(alpha * x + phi)"""
        import numpy as np
        dx, phi = self.dx, phi + alpha * self.xi
        tensors = [np.zeros(((2, bz, 2))) for bz in self.bases]
        for i, bz in list(enumerate(self.bases))[::-1]:
            tensors[i][0, :, 0] = tensors[i][1, :, 1] = np.cos(np.mgrid[:bz] * alpha * dx)
            tensors[i][0, :, 1] = np.sin(np.mgrid[:bz] * alpha * dx)
            tensors[i][1, :, 0] = -np.sin(np.mgrid[:bz] * alpha * dx)
            if i == self.n_sites - 1:
                tensors[i] = tensors[i] @ np.array([np.sin(phi), np.cos(phi)])[:, None]
            dx *= bz
        return PDETools1D.trans_tensors_to_pymps(tensors)

    def pymps_from_exponential(self, z, alpha):
        """f(x) = z^(alpha * x)"""
        import numpy as np
        dx = self.dx
        tensors = [np.zeros(((1, bz, 1))) for bz in self.bases]
        for i, bz in list(enumerate(self.bases))[::-1]:
            tensors[i][0, :, 0] = z ** (np.mgrid[:bz] * alpha * dx)
            if i == self.n_sites - 1:
                tensors[i][0, :, 0] *= z ** (alpha * self.xi)
            dx *= bz
        return PDETools1D.trans_tensors_to_pymps(tensors)

    def pymps_rasterize(self, pyket, n_pts=4096):
        """Sample f(x) as (xi, fi)"""
        import numpy as np
        tensors = PDETools1D.trans_pymps_to_tensors(pyket)
        p = tensors[0]
        for x in tensors[1:]:
            if p.size // p.shape[-1] >= n_pts:
                x = x[:, 0]
            p = np.tensordot(p, x, axes=((-1), (0)))
        p = p.reshape(-1)
        return np.linspace(self.xi, self.xf, len(p) + 1)[:-1], p

    def pympo_from_differential(self, coeffs, cutoff=1E-24, pbc=True):
        """H = sum_i (coeffs[i] * d^i/dx^i)"""
        import math, numpy as np
        fxs = {}
        for k, cc in enumerate(coeffs):
            if k % 2 == 0:
                rs = [{k // 2 - i: (-1) ** i * math.comb(k, i) for i in range(k + 1)}]
            else:
                rs = [{k // 2 - i - 1: (-1) ** (i + 1) * math.comb(k - 1, i) / 2 for i in range(k)},
                      {k // 2 - i + 1: (-1) ** i * math.comb(k - 1, i) / 2 for i in range(k)}]
            for x, c in [(x, c) for r in rs for x, c in r.items()]:
                fxs[x] = fxs.get(x, 0) + c * cc / self.dx ** k
        fxs = {k: v for k, v in fxs.items() if abs(v) > cutoff}
        d = {}
        for dx, c in fxs.items():
            rs = [((), dx)]
            for b in self.bases[::-1]:
                ts = []
                for term, x in rs:
                    fx, ax = 1 if x >= 0 else -1, abs(x)
                    z = fx * (ax % b)
                    ts.append((term + (z, ), fx * (ax // b)))
                    if z != 0:
                        ts.append((term + (z - fx * b, ), fx * ((ax + b) // b)))
                rs = ts
            for r in [tuple(-x for x in r[0][::-1]) for r in rs]:
                d[r] = d.get(r, 0.0) + c
        terms = {k: v for k, v in d.items() if abs(v) > cutoff}
        assert self.driver is not None
        b = self.driver.expr_builder()
        idxs = np.arange(self.n_sites, dtype=int)
        tps = ''.join(chr(ord('O') + i) for i in range(1, 12))
        tms = ''.join(chr(ord('N') - i) for i in range(1, 12)[::-1])
        for term, val in terms.items():
            b.add_term(''.join(('z' + tps + tms)[x] for x in term), idxs, val)
        if not pbc:
            return NotImplemented
            from functools import reduce
            idxs2 = np.array([x for x in idxs for _ in range(2)], dtype=int)
            for k, cc in enumerate(coeffs):
                if k % 2 == 0:
                    rs = [{k // 2 - i: (-1) ** i * math.comb(k, i) for i in range(k + 1)}]
                else:
                    rs = [{k // 2 - i - 1: (-1) ** (i + 1) * math.comb(k - 1, i) / 2 for i in range(k)},
                        {k // 2 - i + 1: (-1) ** i * math.comb(k - 1, i) / 2 for i in range(k)}]
                min_x = min(kk for r in rs for kk in r)
                max_x = max(kk for r in rs for kk in r)
                xbz = reduce(lambda x, y: x * y, self.bases)
                for ix in range(abs(min_x) + max_x):
                    x0 = -ix if ix < abs(min_x) else abs(min_x) - ix
                    xs = PDETools1D.solve_finite_diff(k, x0)
                    if k % 2 == 1 and ix >= abs(min_x):
                        xs = [-xx for xx in xs]
                    gr = {x0 + kk if ix < abs(min_x) else -x0 - kk: -vv / 2 for kk, vv in enumerate(xs)}
                    x = ix if ix < abs(min_x) else xbz - 1 - (ix - abs(min_x))
                    for r in rs + [gr]:
                        for kk, vv in r.items():
                            kx, dx, term = x, kk, ''
                            for bz in self.bases[::-1]:
                                fx, ax = 1 if dx >= 0 else -1, abs(dx)
                                z = fx * (ax % bz)
                                if z == 0 or (z < 0 and kx % bz + z >= 0) or (z > 0 and kx % bz + z < bz):
                                    term += ('z' + tps + tms)[-z]
                                    dx = fx * (ax // bz)
                                else:
                                    term += ('z' + tps + tms)[-z + fx * bz]
                                    dx = fx * ((ax + bz) // bz)
                                term += chr(ord('a') + kx % bz)
                                kx = kx // bz
                            b.add_term(term[::-1], idxs2, -cc * vv / self.dx ** k)
        mpo = self.driver.get_mpo(b.finalize(adjust_order=False, fermionic_ops=''), add_ident=False, iprint=0)
        from pyblock2.algebra.io import MPOTools
        return MPOTools.from_block2(mpo.prim_mpo)
