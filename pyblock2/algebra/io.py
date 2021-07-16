
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
MPS/MPO format transform between block2 and pyblock2.
"""

import numpy as np
from .core import MPS, MPO, Tensor, SubTensor
from block2 import OpTypes, QCTypes, SZ
from block2.sz import StateInfo, MPOQC


class TensorTools:
    @staticmethod
    def from_block2_fused(bspmat, l, r, lr, clr):
        """Translate block2 rank2 left-fused right-boundary SparseMatrix to pyblock2 rank2 tensor."""
        blocks = []
        for i in range(bspmat.info.n):
            ql = bspmat.info.quanta[i].get_bra(bspmat.info.delta_quantum)
            ib = lr.find_state(ql)
            bbed = clr.n if ib == lr.n - 1 else clr.n_states[ib + 1]
            pmat = np.array(bspmat[i]).flatten()
            ip = 0
            for bb in range(clr.n_states[ib], bbed):
                ibba = clr.quanta[bb].data >> 16
                ibbb = clr.quanta[bb].data & 0xFFFF
                ql, nl = l.quanta[ibba], l.n_states[ibba]
                qr, nr = r.quanta[ibbb], r.n_states[ibbb]
                rmat = pmat[ip: ip + nl * nr].reshape((nl, nr))
                blocks.append(
                    SubTensor(q_labels=(ql, qr), reduced=rmat.copy()))
                ip += nl * nr
            assert ip == pmat.shape[0]
        return Tensor(blocks=blocks)

    @staticmethod
    def from_block2_no_fused(bspmat):
        """Translate block2 rank2 unfused SparseMatrix to pyblock2 rank2 tensor."""
        blocks = []
        for i in range(bspmat.info.n):
            ql = bspmat.info.quanta[i].get_bra(bspmat.info.delta_quantum)
            qr = bspmat.info.quanta[i].get_ket()
            if bspmat.info.is_wavefunction:
                qr = -qr
            pmat = np.array(bspmat[i])
            blocks.append(
                SubTensor(q_labels=(ql, qr), reduced=pmat.copy()))
        return Tensor(blocks=blocks)

    @staticmethod
    def from_block2_left_fused(bspmat, l, m, lm, clm):
        """Translate block2 rank2 left-fused SparseMatrix to pyblock2 rank3 tensor."""
        blocks = []
        for i in range(bspmat.info.n):
            qlm = bspmat.info.quanta[i].get_bra(bspmat.info.delta_quantum)
            qr = bspmat.info.quanta[i].get_ket()
            if bspmat.info.is_wavefunction:
                qr = -qr
            ib = lm.find_state(qlm)
            bbed = clm.n if ib == lm.n - 1 else clm.n_states[ib + 1]
            pmat = np.array(bspmat[i])
            nr = pmat.shape[1]
            ip = 0
            for bb in range(clm.n_states[ib], bbed):
                ibba = clm.quanta[bb].data >> 16
                ibbb = clm.quanta[bb].data & 0xFFFF
                ql, nl = l.quanta[ibba], l.n_states[ibba]
                qm, nm = m.quanta[ibbb], m.n_states[ibbb]
                rmat = pmat[ip: ip + nl * nm, :].reshape((nl, nm, nr))
                blocks.append(
                    SubTensor(q_labels=(ql, qm, qr), reduced=rmat.copy()))
                ip += nl * nm
            assert ip == pmat.shape[0]
        return Tensor(blocks=blocks)

    @staticmethod
    def from_block2_right_fused(bspmat, m, r, mr, cmr):
        """Translate block2 rank2 right-fused SparseMatrix to pyblock2 rank3 tensor."""
        blocks = []
        for i in range(bspmat.info.n):
            ql = bspmat.info.quanta[i].get_bra(bspmat.info.delta_quantum)
            qmr = bspmat.info.quanta[i].get_ket()
            if bspmat.info.is_wavefunction:
                qmr = -qmr
            ik = mr.find_state(qmr)
            kked = cmr.n if ik == mr.n - 1 else cmr.n_states[ik + 1]
            pmat = np.array(bspmat[i])
            nl = pmat.shape[0]
            ip = 0
            for kk in range(cmr.n_states[ik], kked):
                ikka = cmr.quanta[kk].data >> 16
                ikkb = cmr.quanta[kk].data & 0xFFFF
                qm, nm = m.quanta[ikka], m.n_states[ikka]
                qr, nr = r.quanta[ikkb], r.n_states[ikkb]
                rmat = pmat[:, ip:ip + nm * nr].reshape((nl, nm, nr))
                blocks.append(
                    SubTensor(q_labels=(ql, qm, qr), reduced=rmat.copy()))
                ip += nm * nr
            assert ip == pmat.shape[1]
        return Tensor(blocks=blocks)

    @staticmethod
    def from_block2_left_and_right_fused(bspmat, l, ma, mb, r, lm, clm, mr, cmr):
        """Translate block2 rank2 left-and-right-fused SparseMatrix to pyblock2 rank4 tensor."""
        blocks = []
        for i in range(bspmat.info.n):
            qlm = bspmat.info.quanta[i].get_bra(bspmat.info.delta_quantum)
            qmr = bspmat.info.quanta[i].get_ket()
            if bspmat.info.is_wavefunction:
                qmr = -qmr
            ib = lm.find_state(qlm)
            bbed = clm.n if ib == lm.n - 1 else clm.n_states[ib + 1]
            ik = mr.find_state(qmr)
            kked = cmr.n if ik == mr.n - 1 else cmr.n_states[ik + 1]
            pmat = np.array(bspmat[i])
            ipl = 0
            for bb in range(clm.n_states[ib], bbed):
                ibba = clm.quanta[bb].data >> 16
                ibbb = clm.quanta[bb].data & 0xFFFF
                ql, nl = l.quanta[ibba], l.n_states[ibba]
                qma, nma = ma.quanta[ibbb], ma.n_states[ibbb]
                ipr = 0
                for kk in range(cmr.n_states[ik], kked):
                    ikka = cmr.quanta[kk].data >> 16
                    ikkb = cmr.quanta[kk].data & 0xFFFF
                    qmb, nmb = mb.quanta[ikka], mb.n_states[ikka]
                    qr, nr = r.quanta[ikkb], r.n_states[ikkb]
                    rmat = pmat[ipl: ipl + nl * nma, ipr: ipr
                                + nmb * nr].reshape((nl, nma, nmb, nr))
                    blocks.append(
                        SubTensor(q_labels=(ql, qma, qmb, qr), reduced=rmat.copy()))
                    ipr += nmb * nr
                assert ipr == pmat.shape[1]
                ipl += nl * nma
            assert ipl == pmat.shape[0]
        return Tensor(blocks=blocks)


class MPSTools:
    @staticmethod
    def from_block2(bmps):
        """Translate block2 MPS to pyblock2 MPS."""
        tensors = [None] * bmps.n_sites
        for i in range(0, bmps.n_sites):
            if bmps.tensors[i] is None:
                continue
            if (i == 0 and i < bmps.center) or (
                    i == bmps.n_sites - 1
                    and i >= bmps.center + bmps.dot) or (
                    i == 0 and i == bmps.center and bmps.dot == 1) or (
                    i == 0 and i == bmps.center
                    and i == bmps.n_sites - 2 and bmps.dot == 2):
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_no_fused(bmps.tensors[i])
                bmps.unload_tensor(i)
            elif i < bmps.center or (
                    i == bmps.center and i == bmps.n_sites - 2 and
                    bmps.dot == 2) or (
                    i == bmps.center and bmps.dot == 1):
                bmps.info.load_left_dims(i)
                l = bmps.info.left_dims[i]
                m = bmps.info.basis[i]
                lm = StateInfo.tensor_product_ref(
                    l, m, bmps.info.left_dims_fci[i + 1])
                clm = StateInfo.get_connection_info(l, m, lm)
                bmps.load_tensor(i)
                if i == bmps.n_sites - 1 and i == bmps.center and bmps.dot == 1:
                    if bmps.tensors[i].info.n == 1 and \
                            bmps.tensors[i].info.quanta[0].get_ket() == -bmps.target:
                        tensors[i] = TensorTools.from_block2_fused(
                            bmps.tensors[i], l, m, lm, clm)
                    else:
                        tensors[i] = TensorTools.from_block2_no_fused(
                            bmps.tensors[i])
                else:
                    tensors[i] = TensorTools.from_block2_left_fused(
                        bmps.tensors[i], l, m, lm, clm)
                bmps.unload_tensor(i)
                clm.deallocate()
                lm.deallocate()
                l.deallocate()
            elif i >= bmps.center + bmps.dot or (
                    i == bmps.center and i == 0 and bmps.dot == 2):
                if i >= bmps.center + bmps.dot:
                    bmps.info.load_right_dims(i + 1)
                    m = bmps.info.basis[i]
                    r = bmps.info.right_dims[i + 1]
                    mr = StateInfo.tensor_product_ref(
                        m, r, bmps.info.right_dims_fci[i])
                else:
                    bmps.info.load_right_dims(i + 2)
                    m = bmps.info.basis[i + 1]
                    r = bmps.info.right_dims[i + 2]
                    mr = StateInfo.tensor_product_ref(
                        m, r, bmps.info.right_dims_fci[i + 1])
                cmr = StateInfo.get_connection_info(m, r, mr)
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_right_fused(
                    bmps.tensors[i], m, r, mr, cmr)
                bmps.unload_tensor(i)
                cmr.deallocate()
                mr.deallocate()
                r.deallocate()
            elif i == bmps.center and i != 0 and i != bmps.n_sites - 2 and bmps.dot == 2:
                bmps.info.load_left_dims(i)
                bmps.info.load_right_dims(i + 2)
                l = bmps.info.left_dims[i]
                ma = bmps.info.basis[i]
                mb = bmps.info.basis[i + 1]
                r = bmps.info.right_dims[i + 2]
                lm = StateInfo.tensor_product_ref(
                    l, ma, bmps.info.left_dims_fci[i + 1])
                mr = StateInfo.tensor_product_ref(
                    mb, r, bmps.info.right_dims_fci[i + 1])
                clm = StateInfo.get_connection_info(l, m, lm)
                cmr = StateInfo.get_connection_info(m, r, mr)
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_left_and_right_fused(
                    bmps.tensors[i], l, ma, mb, r, lm, clm, mr, cmr)
                bmps.unload_tensor(i)
                cmr.deallocate()
                clm.deallocate()
                mr.deallocate()
                lm.deallocate()
                r.deallocate()
                l.deallocate()
            else:
                assert False
        if bmps.center != bmps.n_sites - 1:
            for block in tensors[bmps.center].blocks:
                block.q_labels = block.q_labels[:-1] + \
                    (bmps.info.target - block.q_labels[-1], )
        for i in range(bmps.center + bmps.dot, bmps.n_sites):
            for block in tensors[i].blocks:
                if block.rank == 3:
                    block.q_labels = (bmps.info.target - block.q_labels[0],
                                      block.q_labels[1], bmps.info.target - block.q_labels[2])
                elif block.rank == 2:
                    block.q_labels = (bmps.info.target - block.q_labels[0],
                                      block.q_labels[1])
                else:
                    assert False
        return MPS(tensors=tensors)


class MPOTools:
    @staticmethod
    def from_block2(bmpo):
        """Translate block2 (un-simplified) MPO to pyblock2 MPO."""
        assert bmpo.schemer is None
        if isinstance(bmpo, MPOQC):
            assert bmpo.mode == QCTypes.NC or bmpo.mode == QCTypes.CN
        tensors = [None] * bmpo.n_sites
        # tranlate operator name symbols to quantum labels
        idx_mps, idx_qss, idx_imps = [], [], []
        for i in range(0, bmpo.n_sites - 1):
            lidx_mp = {}
            lidx_qs = [op.q_label for op in bmpo.left_operator_names[i].data]
            for ip, q in enumerate(lidx_qs):
                if q not in lidx_mp:
                    lidx_mp[q] = []
                lidx_mp[q].append(ip)
            limp = {iv: iiv for _, v in lidx_mp.items()
                    for iiv, iv in enumerate(v)}
            idx_mps.append(lidx_mp)
            idx_qss.append(lidx_qs)
            idx_imps.append(limp)
        for i in range(0, bmpo.n_sites):
            assert bmpo.tensors[i].lmat == bmpo.tensors[i].rmat
            mat = bmpo.tensors[i].lmat
            ops = bmpo.tensors[i].ops
            map_blocks = {}
            if i == 0:
                for k, expr in enumerate(mat.data):
                    if expr.get_type() == OpTypes.Zero:
                        continue
                    elif expr.get_type() == OpTypes.Elem:
                        spmat = ops[expr.abs()]
                        if spmat.factor == 0 or spmat.info.n == 0:
                            continue
                        qr = idx_qss[i][k]
                        nr = len(idx_mps[i][qr])
                        ir = idx_imps[i][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(
                                spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (qu, qd, qr)
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nu, nd, nr)))
                            map_blocks[qx].reduced[:, :, ir] += expr.factor * \
                                spmat.factor * np.array(spmat[p])
                    else:
                        assert False
            elif i == bmpo.n_sites - 1:
                for k, expr in enumerate(mat.data):
                    if expr.get_type() == OpTypes.Zero:
                        continue
                    elif expr.get_type() == OpTypes.Elem:
                        spmat = ops[expr.abs()]
                        if spmat.factor == 0 or spmat.info.n == 0:
                            continue
                        ql = idx_qss[i - 1][k]
                        nl = len(idx_mps[i - 1][ql])
                        il = idx_imps[i - 1][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(
                                spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (ql, qu, qd)
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nl, nu, nd)))
                            map_blocks[qx].reduced[il, :, :] += expr.factor * \
                                spmat.factor * np.array(spmat[p])
                    else:
                        assert False
            else:
                for (j, k), expr in zip(mat.indices, mat.data):
                    if expr.get_type() == OpTypes.Zero:
                        continue
                    elif expr.get_type() == OpTypes.Elem:
                        spmat = ops[expr.abs()]
                        if spmat.factor == 0 or spmat.info.n == 0:
                            continue
                        ql, qr = idx_qss[i - 1][j], idx_qss[i][k]
                        nl, nr = len(idx_mps[i - 1][ql]
                                     ), len(idx_mps[i][qr])
                        il, ir = idx_imps[i - 1][j], idx_imps[i][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(
                                spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (ql, qu, qd, qr)
                            if np.linalg.norm(np.array(spmat[p])) == 0:
                                continue
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nl, nu, nd, nr)))
                            map_blocks[qx].reduced[il, :, :,
                                                   ir] += expr.factor * spmat.factor * np.array(spmat[p])
                    else:
                        assert False
            tensors[i] = Tensor(blocks=list(map_blocks.values()))
        return MPO(tensors=tensors, const_e=bmpo.const_e)
