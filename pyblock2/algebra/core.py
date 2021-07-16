
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
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
MPS/MPO algebra (fermionic, block-sparse).
"""

import numpy as np
from collections import Counter
from itertools import accumulate, groupby


class SubTensor:
    """
    A block in block-sparse tensor.

    Attributes:
        q_labels : tuple(SZ..)
            Quantum labels for this sub-tensor block.
            Each element in the tuple corresponds one rank of the tensor.
        reduced : numpy.ndarray
            Rank-:attr:`rank` dense reduced matrix.
        rank : int
            Rank of the tensor. ``rank == len(q_labels)``.
    """

    def __init__(self, q_labels=None, reduced=None):
        self.q_labels = tuple(q_labels) if q_labels is not None else ()
        self.rank = len(q_labels)
        self.reduced = reduced
        if self.rank != 0:
            if reduced is not None:
                assert len(self.reduced.shape) == self.rank

    def build_random(self):
        """Set reduced matrix with random numbers in [0, 1)."""
        self.reduced = np.random.random(self.reduced.shape)

    def build_zero(self):
        """Set reduced matrix to zero."""
        self.reduced = np.zeros(self.reduced.shape)
    
    def copy(self):
        """Shallow copy."""
        return SubTensor(q_labels=self.q_labels, reduced=self.reduced)

    def __mul__(self, other):
        """Scalar multiplication."""
        return SubTensor(q_labels=self.q_labels, reduced=other * self.reduced)

    def __neg__(self):
        """Times (-1)."""
        return SubTensor(q_labels=self.q_labels, reduced=-self.reduced)

    def equal_shape(self, other):
        """Test if two blocks have equal shape and quantum labels."""
        return self.q_labels == other.q_labels and self.reduced.shape == other.reduced.shape

    def __eq__(self, other):
        return self.q_labels == other.q_labels and np.allclose(self.reduced, other.reduced)

    def __repr__(self):
        return "(Q=) %r (R=) %r" % (self.q_labels, self.reduced)


class Tensor:
    """
    Block-sparse tensor.

    Attributes:
        blocks : list(SubTensor)
            A list of (non-zero) blocks.
    """

    def __init__(self, blocks=None):
        self.blocks = blocks if blocks is not None else []

    def copy(self):
        """Shallow copy."""
        return Tensor(blocks=self.blocks)

    def deep_copy(self):
        """Deep copy."""
        return Tensor(blocks=[b.copy() for b in self.blocks])

    def zero_copy(self):
        """A deep copy with zero reduced matrices."""
        blocks = [SubTensor(q_labels=b.q_labels, reduced=np.zeros_like(b.reduced))
                  for b in self.blocks]
        return Tensor(blocks=blocks)

    @property
    def rank(self):
        """Rank of the tensor."""
        return 0 if len(self.blocks) == 0 else self.blocks[0].rank

    @property
    def n_blocks(self):
        """Number of (non-zero) blocks."""
        return len(self.blocks)

    def modify(self, other):
        """Modify the blocks according to another Tensor's blocks."""
        self.blocks[:] = other.blocks

    def get_state_info(self, idx):
        """Get dict from quanta to number of states for left or right bond dimension."""
        mp = Counter()
        for block in self.blocks:
            q = block.q_labels[idx]
            if q in mp:
                assert block.reduced.shape[idx] == mp[q]
            else:
                mp[q] = block.reduced.shape[idx]
        return mp

    @staticmethod
    def contract(tsa, tsb, idxa, idxb, fidxa=None, fidxb=None, out_trans=None):
        """
        Contract two Tensor to form a new Tensor.

        Args:
            tsa : Tensor
                Tensor a, as left operand.
            tsb : Tensor
                Tensor b, as right operand.
            idxa : list(int)
                Indices of rank to be contracted in tensor a.
            idxb : list(int)
                Indices of rank to be contracted in tensor b.
            fidxa : None or int or (int, int)
                Index 'bra' in tensor a (operator), for determining fermion phase for 'op x st'.
                Or 'ket' in tensor a (operator), for determining fermion phase for 'op x op'.
                Or '(bra, bra)' in tensor a (operator product), for determining fermion phase for 'op x st'.
                This index should not be contracted.
                'fidxa' and 'fidxb' are only required for operator tensor applied on state tensor.
            fidxb : None or int or (int, int)
                Index 'virtual left' in tensor b (state), for determining fermion phase for 'op x st'.
                Or '(bra, ket)' in tensor b (operator), for determining fermion phase for 'op x op'.
                This index should not be contracted.
            out_trans : None or tuple
                Permutation of output indices.

        Returns:
            tensor : Tensor
                The contracted Tensor.
        """
        assert len(idxa) == len(idxb)
        idxa = [x if x >= 0 else tsa.rank + x for x in idxa]
        idxb = [x if x >= 0 else tsb.rank + x for x in idxb]
        out_idx_a = list(set(range(0, tsa.rank)) - set(idxa))
        out_idx_b = list(set(range(0, tsb.rank)) - set(idxb))

        # tuple of mutable object cannot be used as key
        map_idx_b = {}
        for block in tsb.blocks:
            subg = tuple(block.q_labels[id] for id in idxb)
            if subg not in map_idx_b:
                map_idx_b[subg] = []
            map_idx_b[subg].append(block)

        map_idx_out = {}
        for block_a in tsa.blocks:
            subg = tuple(block_a.q_labels[id] for id in idxa)
            if subg in map_idx_b:
                outga = tuple(block_a.q_labels[id] for id in out_idx_a)
                for block_b in map_idx_b[subg]:
                    outg = outga + \
                        tuple(block_b.q_labels[id] for id in out_idx_b)
                    outd = tuple(x for x in outg)
                    mat = np.tensordot(
                        block_a.reduced, block_b.reduced, axes=(idxa, idxb))
                    # fermionic phase factor
                    if isinstance(fidxa, tuple):
                        # operator product x two-site state
                        assert len(idxa) == 2 and len(fidxa) == 2
                        if (block_a.q_labels[fidxa[0]] + block_a.q_labels[idxa[0]] +
                            block_a.q_labels[fidxa[1]] + block_a.q_labels[idxa[1]]).is_fermion and \
                                block_b.q_labels[fidxb].is_fermion:
                            mat *= -1
                    elif fidxa is not None:
                        assert len(idxa) == 1
                        if isinstance(fidxb, tuple):
                            # operator x operator
                            if block_a.q_labels[fidxa].is_fermion and (
                                    block_b.q_labels[fidxb[0]] + block_b.q_labels[fidxb[1]]).is_fermion:
                                mat *= -1
                        else:
                            # operator x one-site state
                            if (block_a.q_labels[fidxa] + block_a.q_labels[idxa[0]]).is_fermion and \
                                    block_b.q_labels[fidxb].is_fermion:
                                mat *= -1
                    if out_trans is not None:
                        outg = tuple(outg[t] for t in out_trans)
                        mat = mat.transpose(out_trans)
                    if outd not in map_idx_out:
                        map_idx_out[outd] = SubTensor(
                            q_labels=outg, reduced=mat)
                    else:
                        map_idx_out[outd].reduced += mat
        if len(out_idx_a) + len(out_idx_b) == 0:
            if len(map_idx_out) == 0:
                return 0.0
            return map_idx_out[()].reduced.item()
        else:
            return Tensor(blocks=list(map_idx_out.values()))

    def left_canonicalize(self, mode='reduced'):
        """
        Left canonicalization (using QR factorization).
        Left canonicalization needs to collect all left indices for each specific right index.
        So that we will only have one R, but left dim of q is unchanged.

        Returns:
            r_blocks : dict(q_label_r -> numpy.ndarray)
                The R matrix for each right-index quantum label.
        """
        collected_rows = {}
        for block in self.blocks:
            q_label_r = block.q_labels[-1]
            if q_label_r not in collected_rows:
                collected_rows[q_label_r] = []
            collected_rows[q_label_r].append(block)
        r_blocks_map = {}
        for q_label_r, blocks in collected_rows.items():
            l_shapes = [np.prod(b.reduced.shape[:-1]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((sh, -1))
                                  for sh, b in zip(l_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat, mode)
            r_blocks_map[q_label_r] = r
            qs = np.split(q, list(accumulate(l_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, b in zip(qs, blocks):
                b.reduced = q.reshape(b.reduced.shape[:-1] + (r.shape[0], ))
        return r_blocks_map

    def right_canonicalize(self, mode='reduced'):
        """
        Right canonicalization (using QR factorization).

        Returns:
            l_blocks : dict(q_label_l -> numpy.ndarray)
                The L matrix for each left-index quantum label.
        """
        collected_cols = {}
        for block in self.blocks:
            q_label_l = block.q_labels[0]
            if q_label_l not in collected_cols:
                collected_cols[q_label_l] = []
            collected_cols[q_label_l].append(block)
        l_blocks_map = {}
        for q_label_l, blocks in collected_cols.items():
            r_shapes = [np.prod(b.reduced.shape[1:]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((-1, sh)).T
                                  for sh, b in zip(r_shapes, blocks)], axis=0)
            q, r = np.linalg.qr(mat, mode)
            l_blocks_map[q_label_l] = r.T
            qs = np.split(q, list(accumulate(r_shapes[:-1])), axis=0)
            assert(len(qs) == len(blocks))
            for q, b in zip(qs, blocks):
                b.reduced = q.T.reshape((r.shape[0], ) + b.reduced.shape[1:])
        return l_blocks_map

    def left_multiply(self, mats):
        """
        Left Multiplication.
        Currently only used for multiplying R obtained from right-canonicalization/compression.

        Args:
            mats : dict(q_label_r -> numpy.ndarray)
                The R matrix for each right-index quantum label.
        """
        blocks = []
        for block in self.blocks:
            q_label_r = block.q_labels[0]
            if q_label_r in mats:
                mat = np.tensordot(
                    mats[q_label_r], block.reduced, axes=([1], [0]))
                blocks.append(SubTensor(q_labels=block.q_labels, reduced=mat))
        self.blocks = blocks

    def right_multiply(self, mats):
        """
        Right Multiplication.
        Currently only used for multiplying L obtained from right-canonicalization/compression.

        Args:
            mats : dict(q_label_l -> numpy.ndarray)
                The L matrix for each left-index quantum label.
        """
        blocks = []
        for block in self.blocks:
            q_label_l = block.q_labels[-1]
            if q_label_l in mats:
                mat = np.tensordot(
                    block.reduced, mats[q_label_l], axes=([block.rank - 1], [0]))
                blocks.append(SubTensor(q_labels=block.q_labels, reduced=mat))
        self.blocks = blocks

    def truncate_singular_values(self, svd_s, k=-1, cutoff=0.0):
        """
        Internal method for truncation.

        Args:
            svd_s : list(numpy.ndarray)
                Singular value array for each quantum number.
            k : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singluar value.

        Returns:
            svd_r : list(numpy.ndarray)
                Truncated list of singular value arrays.
            gls : list(numpy.ndarray)
                List of kept singular value indices.
            error : double
                Truncation error (same unit as singular value).
        """
        ss = [(i, j, v) for i, ps in enumerate(svd_s)
              for j, v in enumerate(ps)]
        ss.sort(key=lambda x: -x[2])
        ss_trunc = [x for x in ss if x[2] >= cutoff]
        ss_trunc = ss_trunc[:k] if k != -1 else ss_trunc
        ss_trunc.sort(key=lambda x: (x[0], x[1]))
        svd_r = [None] * len(svd_s)
        gls = [None] * len(svd_s)
        error = 0.0
        for ik, g in groupby(ss_trunc, key=lambda x: x[0]):
            gl = np.array([ig[1] for ig in g], dtype=int)
            gl_inv = np.array(
                list(set(range(0, len(svd_s[ik]))) - set(gl)), dtype=int)
            gls[ik] = gl
            error += (svd_s[ik][gl_inv] ** 2).sum()
            svd_r[ik] = svd_s[ik][gl]
        for ik in range(len(svd_s)):
            if gls[ik] is None:
                error += (svd_s[ik] ** 2).sum()
        return svd_r, gls, np.sqrt(error)

    def left_compress(self, k=-1, cutoff=0.0):
        """
        Left compression needs to collect all left indices for each specific right index.
        Bond dimension of rightmost index is compressed.

        Args:
            k : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singluar value.

        Returns:
            compressed tensor, dict of right blocks, compression error
        """
        collected_rows = {}
        for block in self.blocks:
            q_label_r = block.q_labels[-1]
            if q_label_r not in collected_rows:
                collected_rows[q_label_r] = []
            collected_rows[q_label_r].append(block)
        svd_s, blocks_l, blocks_r = [], [], []
        for q_label_r, blocks in collected_rows.items():
            l_shapes = [np.prod(b.reduced.shape[:-1]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((sh, -1))
                                  for sh, b in zip(l_shapes, blocks)], axis=0)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            svd_s.append(s)
            blocks_l.append(u)
            blocks_r.append(vh)
        svd_r, gls, error = self.truncate_singular_values(svd_s, k, cutoff)
        for ik, gl in enumerate(gls):
            if gl is not None and len(gl) != len(svd_s[ik]):
                blocks_l[ik] = blocks_l[ik][:, gl]
                blocks_r[ik] = blocks_r[ik][gl, :]
        r_blocks_map = {}
        l_blocks = []
        for ik, (q_label_r, blocks) in enumerate(collected_rows.items()):
            if svd_r[ik] is not None:
                l_shapes = [np.prod(b.reduced.shape[:-1]) for b in blocks]
                qs = np.split(blocks_l[ik], list(
                    accumulate(l_shapes[:-1])), axis=0)
                for q, b in zip(qs, blocks):
                    mat = q.reshape(
                        b.reduced.shape[:-1] + (blocks_l[ik].shape[1], ))
                    l_blocks.append(
                        SubTensor(q_labels=b.q_labels, reduced=mat))
                r_blocks_map[q_label_r] = svd_r[ik][:, None] * blocks_r[ik]
        return Tensor(blocks=l_blocks), r_blocks_map, error

    def right_compress(self, k=-1, cutoff=0.0):
        """
        Right compression needs to collect all right indices for each specific left index.
        Bond dimension of leftmost index is compressed.

        Args:
            k : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singluar value.

        Returns:
            compressed tensor, dict of left blocks, compression error
        """
        collected_cols = {}
        for block in self.blocks:
            q_label_l = block.q_labels[0]
            if q_label_l not in collected_cols:
                collected_cols[q_label_l] = []
            collected_cols[q_label_l].append(block)
        svd_s, blocks_l, blocks_r = [], [], []
        for q_label_l, blocks in collected_cols.items():
            r_shapes = [np.prod(b.reduced.shape[1:]) for b in blocks]
            mat = np.concatenate([b.reduced.reshape((-1, sh))
                                  for sh, b in zip(r_shapes, blocks)], axis=1)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            svd_s.append(s)
            blocks_l.append(u)
            blocks_r.append(vh)
        svd_r, gls, error = self.truncate_singular_values(svd_s, k, cutoff)
        for ik, gl in enumerate(gls):
            if gl is not None and len(gl) != len(svd_s[ik]):
                blocks_l[ik] = blocks_l[ik][:, gl]
                blocks_r[ik] = blocks_r[ik][gl, :]
        l_blocks_map = {}
        r_blocks = []
        for ik, (q_label_l, blocks) in enumerate(collected_cols.items()):
            if svd_r[ik] is not None:
                r_shapes = [np.prod(b.reduced.shape[1:]) for b in blocks]
                qs = np.split(blocks_r[ik], list(
                    accumulate(r_shapes[:-1])), axis=1)
                assert(len(qs) == len(blocks))
                for q, b in zip(qs, blocks):
                    mat = q.reshape(
                        (blocks_r[ik].shape[0], ) + b.reduced.shape[1:])
                    r_blocks.append(
                        SubTensor(q_labels=b.q_labels, reduced=mat))
                l_blocks_map[q_label_l] = svd_r[ik][None, :] * blocks_l[ik]
        return Tensor(blocks=r_blocks), l_blocks_map, error

    def __mul__(self, other):
        """Scalar multiplication."""
        return Tensor(blocks=[block * other for block in self.blocks])

    def __neg__(self):
        """Times (-1)."""
        return Tensor(blocks=[-block for block in self.blocks])

    def __repr__(self):
        return "\n".join("%3d %r" % (ib, b) for ib, b in enumerate(self.blocks))


class MPS:
    """
    Matrix Product State.

    Attributes:
        tensors : list(Tensor)
            A list of MPS tensors.
        n_sites : int
            Number of sites.
    """

    def __init__(self, tensors=None):
        self.tensors = tensors if tensors is not None else []

    @property
    def n_sites(self):
        """Number of sites."""
        return len(self.tensors)

    def deep_copy(self):
        """Deep copy."""
        return MPS(tensors=[ts.deep_copy() if ts is not None else None for ts in self.tensors])

    def get_left_dims(self, idx=0):
        """
        Get list of dict from quanta to number of states for left bond dimension of each site tensor.

        Args:
            idx : int
                Index of left bond.
        """
        left_dims = []
        for i in range(1, self.n_sites):
            if self.tensors[i] is None:
                left_dims.append(Counter())
            else:
                left_dims.append(self.tensors[i].get_state_info(idx))
        return left_dims

    def get_right_dims(self, idx=-1):
        """
        Get list of dict from quanta to number of states for right bond dimension of each site tensor.

        Args:
            idx : int
                Index of right bond.
        """
        right_dims = []
        for i in range(self.n_sites - 1):
            if self.tensors[i] is None:
                right_dims.append(Counter())
            else:
                right_dims.append(self.tensors[i].get_state_info(idx))
        return right_dims

    def get_bond_dims(self, idxl=0, idxr=-1):
        """
        Get list of dict from quanta to number of states for bond dimension of each site tensor.
        This is the union of left and right dimensions at each bond.

        Args:
            idxl : int
                Index of left bond.
            idxr : int
                Index of right bond (can be negative).
        """
        left_dims = self.get_left_dims(idxl)
        right_dims = self.get_right_dims(idxr)
        bond_dims = []
        for ld, rd in zip(left_dims, right_dims):
            bond_dims.append(ld | rd)
        return bond_dims

    def __mul__(self, other):
        """Scalar multiplication."""
        return MPS(tensors=[self.tensors[0] * other] + self.tensors[1:])

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        """Times (-1)."""
        return MPS(tensors=[-self.tensors[0]] + self.tensors[1:])

    def __add__(self, other):
        """Add two MPS. data in `other` MPS will be put in larger reduced indices."""
        assert isinstance(other, MPS)
        assert self.n_sites == other.n_sites

        bonds = self.get_bond_dims(), other.get_bond_dims()
        sum_bonds = [bondx + bondy for bondx, bondy in zip(*bonds)]

        tensors = []
        for i in range(self.n_sites):
            if self.tensors[i] is None:
                assert other.tensors[i] is None
                tensors.append(None)
                continue
            if i != 0:
                lb = sum_bonds[i - 1]
            if i != self.n_sites - 1:
                rb = sum_bonds[i]
            sub_mp = {}
            # find required new blocks and their shapes
            for block in self.tensors[i].blocks + other.tensors[i].blocks:
                q = block.q_labels
                sh = block.reduced.shape
                if q not in sub_mp:
                    mshape = list(sh)
                    if i != 0:
                        mshape[0] = lb[q[0]]
                    if i != self.n_sites - 1:
                        mshape[-1] = rb[q[-1]]
                    sub_mp[q] = SubTensor(q, np.zeros(tuple(mshape)))
            # copy block self.blocks to smaller index in new block
            for block in self.tensors[i].blocks:
                q = block.q_labels
                sh = block.reduced.shape
                if i == 0:
                    sub_mp[q].reduced[..., : sh[-1]] += block.reduced
                elif i == self.n_sites - 1:
                    sub_mp[q].reduced[: sh[0], ...] += block.reduced
                else:
                    sub_mp[q].reduced[: sh[0], ..., : sh[-1]] += block.reduced
            # copy block other.blocks to greater index in new block
            for block in other.tensors[i].blocks:
                q = block.q_labels
                sh = block.reduced.shape
                if i == 0:
                    sub_mp[q].reduced[..., -sh[-1]:] += block.reduced
                elif i == self.n_sites - 1:
                    sub_mp[q].reduced[-sh[0]:, ...] += block.reduced
                else:
                    sub_mp[q].reduced[-sh[0]:, ..., -sh[-1]:] += block.reduced
            tensors.append(Tensor(blocks=list(sub_mp.values())))
        return MPS(tensors=tensors)

    def __sub__(self, other):
        return self + (-other)

    def __or__(self, other):
        """
        Contraction of two general MPS to a number. <MPS|MPS>.
        A general MPS is MPS/MPO with one or more physical indices in the middle,
        but two-site tensor is not allowed.
        """

        assert self.__class__ == other.__class__
        assert self.n_sites == other.n_sites
        left = 0.0
        for i in range(self.n_sites):
            assert self.tensors[i] is not None
            assert other.tensors[i] is not None
            assert self.tensors[i].rank == other.tensors[i].rank
            if self.tensors[i].n_blocks == 0 or other.tensors[i].n_blocks == 0:
                return 0.0
            if i != self.n_sites - 1:
                cidx = list(range(0, self.tensors[i].rank - 1))
            else:
                cidx = list(range(0, self.tensors[i].rank))
            if i == 0:
                left = Tensor.contract(self.tensors[i], other.tensors[i], cidx, cidx)
            else:
                lbra = Tensor.contract(left, self.tensors[i], [0], [0])
                left = Tensor.contract(lbra, other.tensors[i], cidx, cidx)
        assert isinstance(left, float)

        return left

    def __matmul__(self, other):
        """Contraction of two MPS. <MPS|MPS>."""

        assert isinstance(other, MPS)
        if isinstance(other, MPO):
            return other @ self
        assert self.n_sites == other.n_sites

        left = 0.0
        for i in range(self.n_sites):
            if self.tensors[i] is None:
                assert other.tensors[i] is None
                continue
            if self.tensors[i].n_blocks == 0 or other.tensors[i].n_blocks == 0:
                continue
            if i == 0:
                if self.tensors[i].rank == 3 and self.tensors[i + 1] is None:
                    left = Tensor.contract(
                        self.tensors[i], other.tensors[i], [0, 1], [0, 1])
                else:
                    left = Tensor.contract(
                        self.tensors[i], other.tensors[i], [0], [0])
            else:
                lbra = Tensor.contract(left, self.tensors[i], [0], [0])
                if self.tensors[i].rank == 4 or (
                        i == self.n_sites - 2 and self.tensors[i + 1]
                        is None and self.tensors[i].rank == 3):
                    left = Tensor.contract(lbra, other.tensors[i], [
                                           0, 1, 2], [0, 1, 2])
                else:
                    left = Tensor.contract(
                        lbra, other.tensors[i], [0, 1], [0, 1])
        assert isinstance(left, float)

        return left

    def canonicalize(self, center):
        """
        MPS/MPO canonicalization.

        Args:
            center : int
                Site index of canonicalization center.
        """
        for i in range(0, center):
            if self.tensors[i] is None:
                continue
            rs = self.tensors[i].left_canonicalize()
            if i + 1 < self.n_sites and self.tensors[i + 1] is not None:
                self.tensors[i + 1].left_multiply(rs)
            elif i + 2 < self.n_sites:
                self.tensors[i + 2].left_multiply(rs)
        for i in range(self.n_sites - 1, center, -1):
            if self.tensors[i] is None:
                continue
            ls = self.tensors[i].right_canonicalize()
            if i - 1 >= 0 and self.tensors[i - 1] is not None:
                self.tensors[i - 1].right_multiply(ls)
            elif i - 2 >= 0:
                self.tensors[i - 2].right_multiply(ls)

    def compress(self, k=-1, cutoff=0.0, left=True):
        """
        MPS/MPO bond dimension compression.

        Args:
            k : int
                Maximal total bond dimension.
                If `k == -1`, no restriction in total bond dimension.
            cutoff : double
                Minimal kept singluar value.
            left : bool
                If left, canonicalize to right boundary and then svd to left.
                Otherwise, canonicalize to left boundary and then svd to right.
        """
        merror = 0.0
        if left:
            self.canonicalize(self.n_sites - 1)
            for i in range(self.n_sites - 1, 0, -1):
                if self.tensors[i] is None:
                    continue
                r, ls, err = self.tensors[i].right_compress(k, cutoff)
                merror = max(merror, err)
                self.tensors[i] = r
                if self.tensors[i - 1] is not None:
                    self.tensors[i - 1].right_multiply(ls)
                elif i - 2 >= 0:
                    self.tensors[i - 2].right_multiply(ls)
        else:
            self.canonicalize(0)
            for i in range(0, self.n_sites - 1):
                if self.tensors[i] is None:
                    continue
                l, rs, err = self.tensors[i].left_compress(k, cutoff)
                merror = max(merror, err)
                self.tensors[i] = l
                if self.tensors[i + 1] is not None:
                    self.tensors[i + 1].left_multiply(rs)
                elif i + 2 < self.n_sites:
                    self.tensors[i + 2].left_multiply(rs)
        return merror

    def merge_virtual_dims(self):
        """Merge double left and right virtual dims to single left and right virtual dim."""
        bonds_up = self.get_bond_dims(idxl=0, idxr=-2)
        bonds_dn = self.get_bond_dims(idxl=1, idxr=-1)

        # tensor product of virtual dimensions
        bonds_maps = []
        for bup, bdn in zip(bonds_up, bonds_dn):
            bonds_map = {}
            for u, nu in bup.items():
                for d, nd in bdn.items():
                    q = u + d
                    if q not in bonds_map:
                        bonds_map[q] = [0, {}]
                    assert (u, d) not in bonds_map[q][1]
                    bonds_map[q][1][(u, d)] = bonds_map[q][0]
                    bonds_map[q][0] += nu * nd
            bonds_maps.append(bonds_map)

        # merge virtual dimensions
        for i in range(self.n_sites):
            if self.tensors[i] is None:
                continue
            map_blocks = {}
            for block in self.tensors[i].blocks:
                if i != 0:
                    qla = block.q_labels[0]
                    qlb = block.q_labels[1]
                    ql = qla + qlb
                    nl = bonds_maps[i - 1][ql][0]
                    ill = bonds_maps[i - 1][ql][1][(qla, qlb)]
                    nll = block.reduced.shape[0] * block.reduced.shape[1]
                if i != self.n_sites - 1:
                    qra = block.q_labels[-2]
                    qrb = block.q_labels[-1]
                    qr = qra + qrb
                    nr = bonds_maps[i][qr][0]
                    irr = bonds_maps[i][qr][1][(qra, qrb)]
                    nrr = block.reduced.shape[-2] * block.reduced.shape[-1]
                if i == 0:
                    qm = block.q_labels[:-2]
                    q = qm + (qr,)
                    sh = block.reduced.shape[:-2] + (nr,)
                    xsh = block.reduced.shape[:-2] + (nrr,)
                    if q not in map_blocks:
                        map_blocks[q] = SubTensor(
                            q_labels=q, reduced=np.zeros(sh))
                    map_blocks[q].reduced[..., irr: irr
                                          + nrr] = block.reduced.reshape(xsh)
                elif i == self.n_sites - 1:
                    qm = block.q_labels[2:]
                    q = (ql, ) + qm
                    sh = (nl,) + block.reduced.shape[2:]
                    xsh = (nll,) + block.reduced.shape[2:]
                    if q not in map_blocks:
                        map_blocks[q] = SubTensor(
                            q_labels=q, reduced=np.zeros(sh))
                    map_blocks[q].reduced[ill:ill
                                          + nll, ...] = block.reduced.reshape(xsh)
                else:
                    qm = block.q_labels[2:-2]
                    q = (ql, ) + qm + (qr, )
                    sh = (nl,) + block.reduced.shape[2:-2] + (nr,)
                    xsh = (nll,) + block.reduced.shape[2:-2] + (nrr,)
                    if q not in map_blocks:
                        map_blocks[q] = SubTensor(
                            q_labels=q, reduced=np.zeros(sh))
                    map_blocks[q].reduced[ill:ill + nll, ...,
                                          irr:irr + nrr] = block.reduced.reshape(xsh)
            self.tensors[i] = Tensor(blocks=list(map_blocks.values()))

    def show_bond_dims(self):
        bonds = self.get_bond_dims(idxl=0, idxr=-1)
        return '|'.join([str(sum(x.values())) for x in bonds])

    def __getitem__(self, i):
        return self.tensors[i]

    def __setitem__(self, i, ts):
        self.tensors[i] = ts

    def norm(self):
        return np.sqrt(self @ self)


class MPO(MPS):
    """
    Matrix Product Operator.

    Attributes:
        tensors : list(Tensor)
            A list of MPO tensors.
        const_e : float
            constant energy term.
        n_sites : int
            Number of sites.
    """

    def __init__(self, tensors=None, const_e=0.0):
        self.const_e = const_e
        super().__init__(tensors=tensors)

    def deep_copy(self):
        """Deep copy."""
        return MPO(tensors=[ts.deep_copy() for ts in self.tensors], const_e=self.const_e)

    def __mul__(self, other):
        """Scalar multiplication."""
        return MPO(tensors=[self.tensors[0] * other] + self.tensors[1:], const_e=other * self.const_e)


    def __neg__(self):
        """Times (-1)."""
        return MPO(tensors=[-self.tensors[0]] + self.tensors[1:], const_e=-self.const_e)

    def __add__(self, other):
        """Add two MPO. data in `other` MPO will be put in larger reduced indices."""
        assert isinstance(other, MPO)
        return MPO(tensors=super().__add__(other).tensors, const_e=self.const_e + other.const_e)

    def __matmul__(self, other):
        """
        (a) Contraction of MPO and MPS. MPO |MPS>. (other : MPS)
        (b) Contraction of MPO and MPO. MPO * MPO. (other : MPO)
        """

        tensors = [None] * self.n_sites
        if isinstance(other, MPO):
            assert self.n_sites == other.n_sites

            for i in range(self.n_sites):
                if i == 0:
                    ot = (0, 2, 1, 3)
                    tensors[i] = Tensor.contract(
                        self.tensors[i], other.tensors[i], [1], [0], out_trans=ot)
                else:
                    ot = (0, 2, 1, 3) if i == self.n_sites - \
                        1 else (0, 3, 1, 4, 2, 5)
                    tensors[i] = Tensor.contract(
                        self.tensors[i], other.tensors[i],
                        [2], [1], 1, 0, out_trans=ot)

            mpo = MPO(tensors=tensors)
            mpo.merge_virtual_dims()
            if self.const_e == 0 and other.const_e == 0:
                return mpo
            elif self.const_e == 0:
                return mpo + self * other.const_e
            elif other.const_e == 0:
                return mpo + other * self.const_e
            else:
                mpo = mpo + (other * self.const_e + self * other.const_e)
                mpo.const_e -= self.const_e * other.const_e
                return mpo
        elif isinstance(other, MPS):
            assert self.n_sites == other.n_sites

            for i in range(self.n_sites):
                if other.tensors[i] is None:
                    continue
                if i == 0:
                    if i < self.n_sites - 1 and other.tensors[i + 1] is None:
                        # 2-site case (first tensor or the only tensor)
                        op = Tensor.contract(
                            self.tensors[i], self.tensors[i + 1], [2], [0], 1, (1, 2))
                        tensors[i] = Tensor.contract(
                            op, other.tensors[i], [1, 3], [0, 1])
                    else:
                        # 1-site case (first tensor)
                        tensors[i] = Tensor.contract(
                            self.tensors[i], other.tensors[i], [1], [0])
                else:
                    if other.tensors[i].rank == 4 or \
                            (i == self.n_sites - 2 and other.tensors[i + 1] is None and other.tensors[i].rank == 3):
                        if other.tensors[i].rank == 4:
                            assert i < self.n_sites - 1
                            assert other.tensors[i + 1] is None
                            ot = (0, 4, 1, 2, 3, 5)
                        else:
                            ot = (0, 3, 1, 2)
                        # 2-site case
                        op = Tensor.contract(
                            self.tensors[i], self.tensors[i + 1], [3], [0], 2, (1, 2))
                        tensors[i] = Tensor.contract(op, other.tensors[i],
                                                     [2, 4], [1, 2], (1, 3), 0, out_trans=ot)
                    else:
                        # 1-site case
                        ot = (0, 2, 1) if i == self.n_sites - \
                            1 else (0, 3, 1, 2, 4)
                        tensors[i] = Tensor.contract(
                            self.tensors[i], other.tensors[i],
                            [2], [1], 1, 0, out_trans=ot)

            mps = MPS(tensors=tensors)
            mps.merge_virtual_dims()
            return mps + self.const_e * other if self.const_e != 0 else mps
        else:
            assert False
