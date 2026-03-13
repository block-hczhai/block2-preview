
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

from functools import reduce
import math

import numpy as np


class PDEToolsND:
    _SITE_ORDER_INTERLEAVED = "interleaved"
    _SITE_ORDER_BLOCKED = "blocked"

    def __init__(self, n_pts, nd, xi=0.0, xf=1.0, bases=2, site_order="interleaved"):
        if nd < 1:
            raise ValueError("nd must be positive")
        if n_pts < 1:
            raise ValueError("n_pts must be positive")
        if site_order not in (self._SITE_ORDER_INTERLEAVED, self._SITE_ORDER_BLOCKED):
            raise ValueError("site_order must be 'interleaved' or 'blocked'")

        self.nd = int(nd)
        self.n_pts = int(n_pts)
        self.n_sites = self.nd * self.n_pts
        self.site_order = site_order
        self._xi = tuple(float(x) for x in self._broadcast_axis_values(xi, "xi"))
        self._xf = tuple(float(x) for x in self._broadcast_axis_values(xf, "xf"))
        self._bases_nd = tuple(
            tuple(int(x) for x in row) for row in self._normalize_bases(bases)
        )
        self._dx = tuple(
            (xf_d - xi_d) / reduce(lambda x, y: x * y, bases_d, 1)
            for xi_d, xf_d, bases_d in zip(self._xi, self._xf, self._bases_nd)
        )
        self._site_schedule = tuple(self._build_site_schedule())
        self._axis_sites = tuple(
            tuple(i for i, (dim, _) in enumerate(self._site_schedule) if dim == axis)
            for axis in range(self.nd)
        )
        self._bases = [self._bases_nd[dim][pt] for dim, pt in self._site_schedule]
        self._grid_shape = tuple(
            reduce(lambda x, y: x * y, bases_d, 1) for bases_d in self._bases_nd
        )
        self.driver = None

    @property
    def xi(self):
        return self._xi

    @property
    def xf(self):
        return self._xf

    @property
    def dx(self):
        return self._dx

    @property
    def bases(self):
        return list(self._bases)

    @property
    def bases_nd(self):
        return [list(row) for row in self._bases_nd]

    @property
    def site_schedule(self):
        return list(self._site_schedule)

    @property
    def axis_sites(self):
        return [list(row) for row in self._axis_sites]

    @staticmethod
    def trans_tensors_to_pymps(tensors):
        from pyblock2.algebra.core import MPS, SubTensor, Tensor
        from block2 import SAny

        tensors = [np.asarray(ts) for ts in tensors]
        tensors[0], tensors[-1] = tensors[0][0], tensors[-1][..., 0]
        q_labels = [(SAny(),) * len(ts.shape) for ts in tensors]
        return MPS([Tensor([SubTensor(qs, ts)]) for qs, ts in zip(q_labels, tensors)])

    @staticmethod
    def trans_pymps_to_tensors(pyket):
        return [t.blocks[0].reduced for t in pyket.tensors]

    @staticmethod
    def shift_polynomial_x(coeffs, dx):
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
                for kk in range(i + 1, n):
                    a[j][kk] -= a[j][i] * a[i][kk]
                b[j] -= a[j][i] * b[i]
                a[j][i] = 0
        for i in range(n)[::-1]:
            for j in range(0, i):
                b[j] -= a[j][i] * b[i]
        return b + [w]

    def init_dmrg_driver(self, **kwargs):
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        driver = DMRGDriver(symm_type=SymmetryTypes.SAny, **kwargs)
        driver.set_symmetry_groups()
        Q = driver.bw.SX
        site_ops = [self._build_site_ops(bz) for bz in self._bases]
        driver.initialize_system(
            n_sites=self.n_sites, vacuum=Q(), target=Q(), hamil_init=False
        )
        driver.ghamil = driver.get_custom_hamiltonian(
            [[(Q(), bz)] for bz in self._bases], site_ops
        )
        self.driver = driver

    def pymps_step_function(self, x0, y, forward=True, dim=0):
        """f(x) = y * (x_dim >= x0) (forward), y * (x_dim <= x0) (backward)."""
        axis_states = [self._axis_constant_pymps(axis) for axis in range(self.nd)]
        axis_states[dim] = self._axis_step_pymps(dim, x0, y, forward)
        return self._combine_axis_pymps(axis_states)

    def pymps_from_range(self, xa, xb, y):
        """f(x) = y * prod_dim (xa[dim] <= x_dim <= xb[dim])."""
        xa = self._broadcast_axis_values(xa, "xa")
        xb = self._broadcast_axis_values(xb, "xb")
        axis_states = [
            self._axis_range_pymps(axis, xa[axis], xb[axis], 1.0)
            for axis in range(self.nd)
        ]
        return self._combine_axis_pymps(axis_states) * y

    def pymps_from_exponential(self, z, alpha):
        """f(x) = z^(sum_dim alpha[dim] * x_dim)."""
        alpha = self._broadcast_axis_values(alpha, "alpha")
        axis_states = [
            self._axis_exponential_pymps(axis, z, alpha[axis])
            for axis in range(self.nd)
        ]
        return self._combine_axis_pymps(axis_states)

    def pymps_rasterize(self, pyket, n_pts=4096):
        """Sample f(x) as (coords, values)."""
        tensors = self.trans_pymps_to_tensors(pyket)
        p = tensors[0]
        for x in tensors[1:]:
            p = np.tensordot(p, x, axes=((-1), (0)))
        values = np.asarray(p).reshape(tuple(self._bases))
        axis_order = [site for sites in self._axis_sites for site in sites]
        values = values.transpose(axis_order).reshape(self._grid_shape)
        coords = [
            np.linspace(xi_d, xf_d, size + 1)[:-1]
            for xi_d, xf_d, size in zip(self._xi, self._xf, self._grid_shape)
        ]
        if values.size > n_pts:
            step = max(1, int(math.ceil((values.size / float(n_pts)) ** (1.0 / self.nd))))
            slicer = tuple(slice(None, None, step) for _ in range(self.nd))
            values = values[slicer]
            coords = [coord[::step] for coord in coords]
        return coords, values

    def pympo_from_differential(self, coeffs, cutoff=1e-24, pbc=True):
        """H = sum_dim sum_k coeffs[dim][k] * d^k / d x_dim^k."""
        if not pbc:
            return NotImplemented

        coeffs = self._normalize_differential_coeffs(coeffs)
        global_terms = {}
        for axis, axis_coeffs in enumerate(coeffs):
            axis_terms = self._differential_terms_1d(
                self._bases_nd[axis], self._dx[axis], axis_coeffs, cutoff
            )
            for term, val in axis_terms.items():
                gterm = [0] * self.n_sites
                for site, shift in zip(self._axis_sites[axis], term):
                    gterm[site] = shift
                gterm = tuple(gterm)
                global_terms[gterm] = global_terms.get(gterm, 0.0) + val
        global_terms = {
            term: val for term, val in global_terms.items() if abs(val) > cutoff
        }

        assert self.driver is not None
        builder = self.driver.expr_builder()
        idxs = np.arange(self.n_sites, dtype=int)
        tps = "".join(chr(ord("O") + i) for i in range(1, 12))
        tms = "".join(chr(ord("N") - i) for i in range(1, 12)[::-1])
        charset = "z" + tps + tms
        for term, val in global_terms.items():
            builder.add_term("".join(charset[x] for x in term), idxs, val)
        mpo = self.driver.get_mpo(
            builder.finalize(adjust_order=False, fermionic_ops=""),
            add_ident=False,
            iprint=0,
        )
        from pyblock2.algebra.io import MPOTools

        return MPOTools.from_block2(mpo.prim_mpo)

    def _build_site_schedule(self):
        if self.site_order == self._SITE_ORDER_INTERLEAVED:
            return [(axis, pt) for pt in range(self.n_pts) for axis in range(self.nd)]
        return [(axis, pt) for axis in range(self.nd) for pt in range(self.n_pts)]

    def _broadcast_axis_values(self, value, name):
        if np.isscalar(value):
            return [value] * self.nd
        value = list(value)
        if len(value) != self.nd:
            raise ValueError("%s must be a scalar or have length nd" % name)
        return value

    def _normalize_bases(self, bases):
        if np.isscalar(bases):
            row = [int(bases)] * self.n_pts
            rows = [row[:] for _ in range(self.nd)]
        else:
            bases = list(bases)
            if len(bases) == self.n_pts and all(np.isscalar(x) for x in bases):
                row = [int(x) for x in bases]
                rows = [row[:] for _ in range(self.nd)]
            elif len(bases) == self.nd and all(not np.isscalar(row) for row in bases):
                rows = []
                for row in bases:
                    row = [int(x) for x in row]
                    if len(row) != self.n_pts:
                        raise ValueError("bases rows must have length n_pts")
                    rows.append(row)
            else:
                raise ValueError(
                    "bases must be an int, length n_pts, or shape (nd, n_pts)"
                )
        if any(base < 1 for row in rows for base in row):
            raise ValueError("bases must be positive")
        return rows

    @staticmethod
    def _build_site_ops(bz):
        ops = {"": np.identity(bz), "z": np.identity(bz)}
        for i in range(1, bz):
            ops[chr(ord("O") + i)] = np.diag(np.ones(bz - i), -i)
            ops[chr(ord("N") - i)] = np.diag(np.ones(bz - i), i)
        for i in range(0, bz):
            ops[chr(ord("a") + i)] = np.zeros((bz, bz))
            ops[chr(ord("a") + i)][i, i] = 1
        return ops

    @staticmethod
    def _expand_mps_tensors(tensors):
        raw = []
        for i, tensor in enumerate(tensors):
            if tensor.ndim == 2:
                if i == 0:
                    raw.append(tensor[None, ...])
                else:
                    raw.append(tensor[..., None])
            else:
                raw.append(tensor)
        return raw

    @staticmethod
    def _axis_bond_dim(tensors, processed):
        if processed == 0:
            return tensors[0].shape[0]
        return tensors[processed - 1].shape[2]

    @staticmethod
    def _embed_axis_tensor(active_tensor, dims_before, dims_after, active_axis):
        left_dim = reduce(lambda x, y: x * y, dims_before, 1)
        right_dim = reduce(lambda x, y: x * y, dims_after, 1)
        tensor = np.zeros((left_dim, active_tensor.shape[1], right_dim), dtype=active_tensor.dtype)
        for phys in range(active_tensor.shape[1]):
            mat = np.array([[1]], dtype=active_tensor.dtype)
            for axis, (lb, rb) in enumerate(zip(dims_before, dims_after)):
                if axis == active_axis:
                    factor = active_tensor[:, phys, :]
                else:
                    assert lb == rb
                    factor = np.identity(lb, dtype=active_tensor.dtype)
                mat = np.kron(mat, factor)
            tensor[:, phys, :] = mat
        return tensor

    @classmethod
    def _axis_step_raw_tensors(cls, bases, xi, dx, x0, y, forward):
        tensors = [np.zeros((2, bz, 2), dtype=float) for bz in bases]
        for i, bz in list(enumerate(bases))[::-1]:
            xx = int((x0 - xi) / dx) % bz
            tensors[i][1, :, 1] = 1
            tensors[i][0, xx, 0] = 1
            tensors[i][0, slice(xx + 1, None) if forward else slice(xx), 1] = 1
            if i == len(bases) - 1:
                tensors[i] = tensors[i] @ np.array([y, y], dtype=float)[:, None]
            dx *= bz
        return tensors

    @classmethod
    def _axis_exponential_raw_tensors(cls, bases, xi, dx, z, alpha):
        dtype = np.result_type(z, alpha, float)
        tensors = [np.zeros((1, bz, 1), dtype=dtype) for bz in bases]
        for i, bz in list(enumerate(bases))[::-1]:
            tensors[i][0, :, 0] = z ** (np.mgrid[:bz] * alpha * dx)
            if i == len(bases) - 1:
                tensors[i][0, :, 0] *= z ** (alpha * xi)
            dx *= bz
        return tensors

    @classmethod
    def _axis_polynomial_raw_tensors(cls, bases, xi, dx, coeffs):
        coeffs = cls.shift_polynomial_x(coeffs, xi)
        tensors = [np.zeros((len(coeffs), bz, len(coeffs)), dtype=float) for bz in bases]
        for i, bz in list(enumerate(bases))[::-1]:
            for j in range(1 if i == 0 else len(coeffs)):
                for k in range(j, len(coeffs)):
                    tensors[i][j, :, k] += math.comb(k, j) * (np.mgrid[:bz] * dx) ** (k - j)
            if i == len(bases) - 1:
                tensors[i] = tensors[i] @ np.array(coeffs, dtype=float)[:, None]
            dx *= bz
        return tensors

    @classmethod
    def _axis_trigonometric_raw_tensors(cls, bases, xi, dx, alpha, phi):
        phi = phi + alpha * xi
        tensors = [np.zeros((2, bz, 2), dtype=float) for bz in bases]
        for i, bz in list(enumerate(bases))[::-1]:
            angles = np.mgrid[:bz] * alpha * dx
            tensors[i][0, :, 0] = tensors[i][1, :, 1] = np.cos(angles)
            tensors[i][0, :, 1] = np.sin(angles)
            tensors[i][1, :, 0] = -np.sin(angles)
            if i == len(bases) - 1:
                tensors[i] = tensors[i] @ np.array([np.sin(phi), np.cos(phi)], dtype=float)[:, None]
            dx *= bz
        return tensors

    def _axis_constant_pymps(self, dim):
        return self.trans_tensors_to_pymps(
            self._axis_exponential_raw_tensors(
                self._bases_nd[dim], self._xi[dim], self._dx[dim], 1.0, 0.0
            )
        )

    def _axis_step_pymps(self, dim, x0, y, forward):
        return self.trans_tensors_to_pymps(
            self._axis_step_raw_tensors(
                self._bases_nd[dim], self._xi[dim], self._dx[dim], x0, y, forward
            )
        )

    def _axis_range_pymps(self, dim, xa, xb, y):
        keta = self._axis_step_pymps(dim, xa, y, True)
        ketb = self._axis_step_pymps(dim, xb, 1.0, False)
        return keta.diag() @ ketb

    def _axis_exponential_pymps(self, dim, z, alpha):
        return self.trans_tensors_to_pymps(
            self._axis_exponential_raw_tensors(
                self._bases_nd[dim], self._xi[dim], self._dx[dim], z, alpha
            )
        )

    def _combine_axis_pymps(self, axis_states):
        axis_tensors = [self._expand_mps_tensors(self.trans_pymps_to_tensors(state)) for state in axis_states]
        axis_pos = [0] * self.nd
        tensors = []
        for active_axis, pt in self._site_schedule:
            assert axis_pos[active_axis] == pt
            active_tensor = axis_tensors[active_axis][pt]
            dims_before = []
            dims_after = []
            for axis in range(self.nd):
                processed = axis_pos[axis]
                if axis == active_axis:
                    dims_before.append(active_tensor.shape[0])
                    dims_after.append(active_tensor.shape[2])
                else:
                    bond_dim = self._axis_bond_dim(axis_tensors[axis], processed)
                    dims_before.append(bond_dim)
                    dims_after.append(bond_dim)
            tensors.append(
                self._embed_axis_tensor(
                    active_tensor, dims_before, dims_after, active_axis
                )
            )
            axis_pos[active_axis] += 1
        assert axis_pos == [self.n_pts] * self.nd
        return self.trans_tensors_to_pymps(tensors)

    def _normalize_differential_coeffs(self, coeffs):
        coeffs = list(coeffs)
        if self.nd == 1 and coeffs and np.isscalar(coeffs[0]):
            return [coeffs]
        if len(coeffs) != self.nd:
            raise ValueError("coeffs must provide one coefficient list per dimension")
        if any(np.isscalar(axis_coeffs) for axis_coeffs in coeffs):
            raise ValueError("each dimension must have a coefficient list")
        return [list(axis_coeffs) for axis_coeffs in coeffs]

    @classmethod
    def _differential_terms_1d(cls, bases, dx, coeffs, cutoff):
        fxs = {}
        for k, cc in enumerate(coeffs):
            if k % 2 == 0:
                rs = [{k // 2 - i: (-1) ** i * math.comb(k, i) for i in range(k + 1)}]
            else:
                rs = [
                    {
                        k // 2 - i - 1: (-1) ** (i + 1) * math.comb(k - 1, i) / 2
                        for i in range(k)
                    },
                    {
                        k // 2 - i + 1: (-1) ** i * math.comb(k - 1, i) / 2
                        for i in range(k)
                    },
                ]
            for r in rs:
                for x, c in r.items():
                    fxs[x] = fxs.get(x, 0.0) + c * cc / dx ** k
        fxs = {k: v for k, v in fxs.items() if abs(v) > cutoff}
        terms = {}
        for delta, c in fxs.items():
            rs = [((), delta)]
            for base in bases[::-1]:
                next_rs = []
                for term, x in rs:
                    fx = 1 if x >= 0 else -1
                    ax = abs(x)
                    z = fx * (ax % base)
                    next_rs.append((term + (z,), fx * (ax // base)))
                    if z != 0:
                        next_rs.append((term + (z - fx * base,), fx * ((ax + base) // base)))
                rs = next_rs
            for digits, _ in rs:
                term = tuple(-x for x in digits[::-1])
                terms[term] = terms.get(term, 0.0) + c
        return {k: v for k, v in terms.items() if abs(v) > cutoff}


class PDETools1D(PDEToolsND):
    def __init__(self, n_sites, xi=0.0, xf=1.0, bases=2):
        super().__init__(n_pts=n_sites, nd=1, xi=xi, xf=xf, bases=bases)

    @property
    def xi(self):
        return self._xi[0]

    @property
    def xf(self):
        return self._xf[0]

    @property
    def dx(self):
        return self._dx[0]

    def pymps_step_function(self, x0, y, forward=True):
        return super().pymps_step_function(x0, y, forward=forward, dim=0)

    def pymps_from_polynomial(self, coeffs):
        return self.trans_tensors_to_pymps(
            self._axis_polynomial_raw_tensors(
                self._bases_nd[0], self.xi, self.dx, coeffs
            )
        )

    def pymps_from_trigonometric(self, alpha, phi=0.0):
        return self.trans_tensors_to_pymps(
            self._axis_trigonometric_raw_tensors(
                self._bases_nd[0], self.xi, self.dx, alpha, phi
            )
        )

    def pymps_rasterize(self, pyket, n_pts=4096):
        """Sample f(x) as (xi, fi)."""
        tensors = self.trans_pymps_to_tensors(pyket)
        p = tensors[0]
        for x in tensors[1:]:
            if p.size // p.shape[-1] >= n_pts:
                x = x[:, 0]
            p = np.tensordot(p, x, axes=((-1), (0)))
        p = p.reshape(-1)
        return np.linspace(self.xi, self.xf, len(p) + 1)[:-1], p

    def pympo_from_differential(self, coeffs, cutoff=1e-24, pbc=True):
        return super().pympo_from_differential([coeffs], cutoff=cutoff, pbc=pbc)
