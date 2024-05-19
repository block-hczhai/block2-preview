
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


class TensorTools:
    @staticmethod
    def from_block2_fused(bspmat, l, r, lr, clr):
        """Translate block2 rank2 left-fused right-boundary SparseMatrix to pyblock2 rank2 tensor."""
        blocks = []
        for i in range(bspmat.info.n):
            ql = bspmat.info.quanta[i].get_bra(bspmat.info.delta_quantum)
            ib = lr.find_state(ql)
            pmat = np.array(bspmat[i]).ravel()
            ip = 0
            for bb in range(clr.acc_n_states[ib], clr.acc_n_states[ib + 1]):
                ibba, ibbb = clr.ij_indices[bb]
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
            pmat = np.array(bspmat[i])
            nr = pmat.shape[1]
            ip = 0
            for bb in range(clm.acc_n_states[ib], clm.acc_n_states[ib + 1]):
                ibba, ibbb = clm.ij_indices[bb]
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
            pmat = np.array(bspmat[i])
            nl = pmat.shape[0]
            ip = 0
            for kk in range(cmr.acc_n_states[ik], cmr.acc_n_states[ik + 1]):
                ikka, ikkb = cmr.ij_indices[kk]
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
            ik = mr.find_state(qmr)
            pmat = np.array(bspmat[i])
            ipl = 0
            for bb in range(clm.acc_n_states[ib], clm.acc_n_states[ib + 1]):
                ibba, ibbb = clm.ij_indices[bb]
                ql, nl = l.quanta[ibba], l.n_states[ibba]
                qma, nma = ma.quanta[ibbb], ma.n_states[ibbb]
                ipr = 0
                for kk in range(cmr.acc_n_states[ik], cmr.acc_n_states[ik + 1]):
                    ikka, ikkb = cmr.ij_indices[kk]
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
                lm = m.__class__.tensor_product_ref(
                    l, m, bmps.info.left_dims_fci[i + 1])
                clm = m.__class__.get_connection_info(l, m, lm)
                bmps.load_tensor(i)
                if i == bmps.n_sites - 1 and i == bmps.center and bmps.dot == 1:
                    if bmps.tensors[i].info.n == 1 and \
                            bmps.tensors[i].info.quanta[0].get_ket() == -bmps.info.target:
                        tensors[i] = TensorTools.from_block2_fused(
                            bmps.tensors[i], l, m, lm, clm)
                    else:
                        tensors[i] = TensorTools.from_block2_no_fused(
                            bmps.tensors[i])
                else:
                    tensors[i] = TensorTools.from_block2_left_fused(
                        bmps.tensors[i], l, m, lm, clm)
                bmps.unload_tensor(i)
                lm.deallocate()
                l.deallocate()
            elif i >= bmps.center + bmps.dot or (
                    i == bmps.center and i == 0 and bmps.dot == 2):
                if i >= bmps.center + bmps.dot:
                    bmps.info.load_right_dims(i + 1)
                    m = bmps.info.basis[i]
                    r = bmps.info.right_dims[i + 1]
                    mr = m.__class__.tensor_product_ref(
                        m, r, bmps.info.right_dims_fci[i])
                else:
                    bmps.info.load_right_dims(i + 2)
                    m = bmps.info.basis[i + 1]
                    r = bmps.info.right_dims[i + 2]
                    mr = m.__class__.tensor_product_ref(
                        m, r, bmps.info.right_dims_fci[i + 1])
                cmr = m.__class__.get_connection_info(m, r, mr)
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_right_fused(
                    bmps.tensors[i], m, r, mr, cmr)
                bmps.unload_tensor(i)
                mr.deallocate()
                r.deallocate()
            elif i == bmps.center and i != 0 and i != bmps.n_sites - 2 and bmps.dot == 2:
                bmps.info.load_left_dims(i)
                bmps.info.load_right_dims(i + 2)
                l = bmps.info.left_dims[i]
                ma = bmps.info.basis[i]
                mb = bmps.info.basis[i + 1]
                r = bmps.info.right_dims[i + 2]
                lm = ma.__class__.tensor_product_ref(
                    l, ma, bmps.info.left_dims_fci[i + 1])
                mr = ma.__class__.tensor_product_ref(
                    mb, r, bmps.info.right_dims_fci[i + 1])
                clm = ma.__class__.get_connection_info(l, ma, lm)
                cmr = ma.__class__.get_connection_info(mb, r, mr)
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_left_and_right_fused(
                    bmps.tensors[i], l, ma, mb, r, lm, clm, mr, cmr)
                bmps.unload_tensor(i)
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

    @staticmethod
    def to_block2(mps, basis, center=0, tag='KET'):
        """
        Translate pyblock2 MPS to block2 MPS.
        
        Args:
            mps : pyblock2 MPS
                More than one physical index is not supported.
                But fused index can be supported.
            center : int
                The pyblock2 MPS is transformed after
                canonicalization at the given center site.
            basis : List(Counter)
                Phyiscal basis infomation at each site.
            tag : str
                Tag of the block2 MPS. Default is "KET".
        
        Returns:
            bmps : block2 MPS
                To inspect this MPS, please make sure that the block2 global
                scratch folder and stack memory are properly initialized.
        """
        import block2 as b
        Q = mps.tensors[0].blocks[0].q_labels[0].__class__
        DT = mps.tensors[0].blocks[0].reduced.dtype
        if Q == b.SZ and DT == np.complex128:
            import block2.cpx.sz as bs, block2.sz as brs, block2.cpx as bx
        elif Q == b.SZ and DT == np.float64:
            import block2.sz as bs, block2.sz as brs, block2 as bx
        elif Q == b.SU2 and DT == np.complex128:
            import block2.cpx.su2 as bs, block2.su2 as brs, block2.cpx as bx
        elif Q == b.SU2 and DT == np.float64:
            import block2.su2 as bs, block2.su2 as brs, block2 as bx
        elif Q == b.SGF and DT == np.complex128:
            import block2.cpx.sgf as bs, block2.sgf as brs, block2.cpx as bx
        elif Q == b.SGF and DT == np.float64:
            import block2.sgf as bs, block2.sgf as brs, block2 as bx
        elif Q == b.SGB and DT == np.complex128:
            import block2.cpx.sgb as bs, block2.sgb as brs, block2.cpx as bx
        elif Q == b.SGB and DT == np.float64:
            import block2.sgb as bs, block2.sgb as brs, block2 as bx
        else:
            raise RuntimeError("Q = %s DT = %s not supported!" % (Q, DT))
        if b.Global.frame is None:
            raise RuntimeError("block2 is not initialized!")
        save_dir = b.Global.frame.save_dir
        mps.canonicalize(center)
        n_sites = len(mps.tensors)
        ql = mps.tensors[0].blocks[0].q_labels
        qr = mps.tensors[-1].blocks[0].q_labels
        vacuum = (ql[1] - ql[0])[0]
        target = (qr[0] + qr[1])[0]
        info = brs.MPSInfo(n_sites, vacuum, target, basis)
        info.tag = tag
        info.set_bond_dimension_full_fci(vacuum, vacuum)
        info.left_dims[0] = brs.StateInfo(vacuum)
        for i, xinfo in enumerate(mps.get_left_dims()):
            p = info.left_dims[i + 1]
            p.allocate(len(xinfo))
            for ix, (k, v) in enumerate(xinfo.items()):
                p.quanta[ix] = k
                p.n_states[ix] = v
            p.sort_states()
        info.left_dims[n_sites] = brs.StateInfo(target)
        info.right_dims[0] = brs.StateInfo(target)
        for i, xinfo in enumerate(mps.get_right_dims()):
            p = info.right_dims[i + 1]
            p.allocate(len(xinfo))
            for ix, (k, v) in enumerate(xinfo.items()):
                p.quanta[ix] = (target - k)[0]
                p.n_states[ix] = v
            p.sort_states()
        info.right_dims[n_sites] = brs.StateInfo(vacuum)
        info.save_mutable()
        info.save_data("%s/%s-mps_info.bin" % (save_dir, tag))
        tensors = [bs.SparseTensor() for _ in range(n_sites)]
        for i, bb in enumerate(basis):
            tensors[i].data = bs.VectorVectorPSSTensor([bs.VectorPSSTensor() for _ in range(bb.n)])
            for block in mps[i].blocks:
                if i == 0:
                    ql = vacuum
                    qm, qr = block.q_labels
                    blk = block.reduced.reshape((1, *block.reduced.shape))
                elif i == n_sites - 1:
                    ql, qm = block.q_labels
                    qr = target
                    blk = block.reduced.reshape((*block.reduced.shape, 1))
                else:
                    ql, qm, qr = block.q_labels
                    blk = block.reduced
                im = bb.find_state(qm)
                assert im != -1
                tensors[i].data[im].append(((ql, qr), bx.Tensor(b.VectorMKLInt(blk.shape))))
                np.array(tensors[i].data[im][-1][1], copy=False)[:] = blk

        umps = bs.UnfusedMPS()
        umps.info = info
        umps.n_sites = n_sites
        umps.canonical_form = "L" * center + ("S" if center == n_sites - 1 else "K") + \
            "R" * (n_sites - center - 1)
        umps.center = center
        umps.dot = 1
        umps.tensors = bs.VectorSpTensor(tensors)
        return umps.finalize()

class MPOTools:
    @staticmethod
    def from_block2(bmpo):
        """Translate block2 (un-simplified) MPO to pyblock2 MPO."""
        assert bmpo.schemer is None
        from block2 import OpTypes, QCTypes
        if bmpo.__class__.__name__ == "MPOQC":
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
                            spm = np.array(spmat[p])
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nu, nd, nr), dtype=spm.dtype))
                            map_blocks[qx].reduced[:, :, ir] += expr.factor * \
                                spmat.factor * spm
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
                            spm = np.array(spmat[p])
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nl, nu, nd), dtype=spm.dtype))
                            map_blocks[qx].reduced[il, :, :] += expr.factor * \
                                spmat.factor * spm
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
                            spm = np.array(spmat[p])
                            if np.linalg.norm(spm) == 0:
                                continue
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx, reduced=np.zeros((nl, nu, nd, nr), dtype=spm.dtype))
                            map_blocks[qx].reduced[il, :, :,
                                                   ir] += expr.factor * spmat.factor * spm
                    elif expr.get_type() == OpTypes.Sum:
                        for xexpr in expr.strings:
                            spmat = ops[xexpr.a.abs()]
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
                                spm = np.array(spmat[p])
                                if np.linalg.norm(spm) == 0:
                                    continue
                                if qx not in map_blocks:
                                    map_blocks[qx] = SubTensor(
                                        q_labels=qx, reduced=np.zeros((nl, nu, nd, nr), dtype=spm.dtype))
                                map_blocks[qx].reduced[il, :, :,
                                                    ir] += xexpr.factor * spmat.factor * spm
                    else:
                        assert False
            tensors[i] = Tensor(blocks=list(map_blocks.values()))
        return MPO(tensors=tensors, const_e=bmpo.const_e)
