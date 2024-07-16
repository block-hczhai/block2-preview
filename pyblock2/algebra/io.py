
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
                rmat = pmat[ip : ip + nl * nr].reshape((nl, nr))
                blocks.append(SubTensor(q_labels=(ql, qr), reduced=rmat.copy()))
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
            blocks.append(SubTensor(q_labels=(ql, qr), reduced=pmat.copy()))
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
                rmat = pmat[ip : ip + nl * nm, :].reshape((nl, nm, nr))
                blocks.append(SubTensor(q_labels=(ql, qm, qr), reduced=rmat.copy()))
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
                rmat = pmat[:, ip : ip + nm * nr].reshape((nl, nm, nr))
                blocks.append(SubTensor(q_labels=(ql, qm, qr), reduced=rmat.copy()))
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
                    rmat = pmat[ipl : ipl + nl * nma, ipr : ipr + nmb * nr].reshape(
                        (nl, nma, nmb, nr)
                    )
                    blocks.append(
                        SubTensor(q_labels=(ql, qma, qmb, qr), reduced=rmat.copy())
                    )
                    ipr += nmb * nr
                assert ipr == pmat.shape[1]
                ipl += nl * nma
            assert ipl == pmat.shape[0]
        return Tensor(blocks=blocks)


def init_block2_types(Q, DT):
    import block2 as b

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
    elif Q == b.SAny and DT == np.complex128:
        import block2.cpx.sany as bs, block2.sany as brs, block2.cpx as bx
    elif Q == b.SAny and DT == np.float64:
        import block2.sany as bs, block2.sany as brs, block2 as bx
    else:
        raise RuntimeError("Q = %s DT = %s not supported!" % (Q, DT))
    if b.Global.frame is None:
        raise RuntimeError("block2 is not initialized!")
    return b, bs, brs, bx


class MPSTools:
    @staticmethod
    def from_block2(bmps):
        """Translate block2 MPS to pyblock2 MPS."""
        tensors = [None] * bmps.n_sites
        for i in range(0, bmps.n_sites):
            if bmps.tensors[i] is None:
                continue
            if (
                (i == 0 and i < bmps.center)
                or (i == bmps.n_sites - 1 and i >= bmps.center + bmps.dot)
                or (i == 0 and i == bmps.center and bmps.dot == 1)
                or (
                    i == 0
                    and i == bmps.center
                    and i == bmps.n_sites - 2
                    and bmps.dot == 2
                )
            ):
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_no_fused(bmps.tensors[i])
                bmps.unload_tensor(i)
            elif (
                i < bmps.center
                or (i == bmps.center and i == bmps.n_sites - 2 and bmps.dot == 2)
                or (
                    i == bmps.center and bmps.dot == 1 and bmps.canonical_form[i] != "S"
                )
            ):
                bmps.info.load_left_dims(i)
                l = bmps.info.left_dims[i]
                m = bmps.info.basis[i]
                lm = m.__class__.tensor_product_ref(
                    l, m, bmps.info.left_dims_fci[i + 1]
                )
                clm = m.__class__.get_connection_info(l, m, lm)
                bmps.load_tensor(i)
                if i == bmps.n_sites - 1 and i == bmps.center and bmps.dot == 1:
                    if (
                        bmps.tensors[i].info.n == 1
                        and bmps.tensors[i].info.quanta[0].get_ket()
                        == -bmps.info.target
                    ):
                        tensors[i] = TensorTools.from_block2_fused(
                            bmps.tensors[i], l, m, lm, clm
                        )
                    else:
                        tensors[i] = TensorTools.from_block2_no_fused(bmps.tensors[i])
                else:
                    tensors[i] = TensorTools.from_block2_left_fused(
                        bmps.tensors[i], l, m, lm, clm
                    )
                bmps.unload_tensor(i)
                lm.deallocate()
                l.deallocate()
            elif (
                i >= bmps.center + bmps.dot
                or (i == bmps.center and i == 0 and bmps.dot == 2)
                or bmps.canonical_form[i] == "S"
            ):
                if i >= bmps.center + bmps.dot or bmps.canonical_form[i] == "S":
                    bmps.info.load_right_dims(i + 1)
                    m = bmps.info.basis[i]
                    r = bmps.info.right_dims[i + 1]
                    mr = m.__class__.tensor_product_ref(
                        m, r, bmps.info.right_dims_fci[i]
                    )
                else:
                    bmps.info.load_right_dims(i + 2)
                    m = bmps.info.basis[i + 1]
                    r = bmps.info.right_dims[i + 2]
                    mr = m.__class__.tensor_product_ref(
                        m, r, bmps.info.right_dims_fci[i + 1]
                    )
                cmr = m.__class__.get_connection_info(m, r, mr)
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_right_fused(
                    bmps.tensors[i], m, r, mr, cmr
                )
                bmps.unload_tensor(i)
                mr.deallocate()
                r.deallocate()
            elif (
                i == bmps.center and i != 0 and i != bmps.n_sites - 2 and bmps.dot == 2
            ):
                bmps.info.load_left_dims(i)
                bmps.info.load_right_dims(i + 2)
                l = bmps.info.left_dims[i]
                ma = bmps.info.basis[i]
                mb = bmps.info.basis[i + 1]
                r = bmps.info.right_dims[i + 2]
                lm = ma.__class__.tensor_product_ref(
                    l, ma, bmps.info.left_dims_fci[i + 1]
                )
                mr = ma.__class__.tensor_product_ref(
                    mb, r, bmps.info.right_dims_fci[i + 1]
                )
                clm = ma.__class__.get_connection_info(l, ma, lm)
                cmr = ma.__class__.get_connection_info(mb, r, mr)
                bmps.load_tensor(i)
                tensors[i] = TensorTools.from_block2_left_and_right_fused(
                    bmps.tensors[i], l, ma, mb, r, lm, clm, mr, cmr
                )
                bmps.unload_tensor(i)
                mr.deallocate()
                lm.deallocate()
                r.deallocate()
                l.deallocate()
            else:
                assert False
        if bmps.center != bmps.n_sites - 1:
            for block in tensors[bmps.center].blocks:
                block.q_labels = block.q_labels[:-1] + (
                    bmps.info.target - block.q_labels[-1],
                )
        for i in range(bmps.center + bmps.dot, bmps.n_sites):
            for block in tensors[i].blocks:
                if block.rank == 3:
                    block.q_labels = (
                        bmps.info.target - block.q_labels[0],
                        block.q_labels[1],
                        bmps.info.target - block.q_labels[2],
                    )
                elif block.rank == 2:
                    block.q_labels = (
                        bmps.info.target - block.q_labels[0],
                        block.q_labels[1],
                    )
                else:
                    assert False
        return MPS(tensors=tensors)

    @staticmethod
    def to_block2(mps, basis, center=0, tag="KET", left_vacuum=None):
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
            left_vacuum : None or SU2
                Left vacuum for singlet embedding

        Returns:
            bmps : block2 MPS
                To inspect this MPS, please make sure that the block2 global
                scratch folder and stack memory are properly initialized.
        """
        Q = mps.tensors[0].blocks[0].q_labels[0].__class__
        DT = mps.tensors[0].blocks[0].reduced.dtype
        b, bs, brs, bx = init_block2_types(Q, DT)
        save_dir = b.Global.frame.save_dir
        mps.canonicalize(center)
        n_sites = len(mps.tensors)
        ql = mps.tensors[0].blocks[0].q_labels
        qr = mps.tensors[-1].blocks[0].q_labels
        if left_vacuum is None:
            left_vacuum = (ql[1] - ql[0])[0]
        target = (qr[0] + qr[1])[0]
        vacuum = (target - target)[0]
        for ib, x in enumerate(basis):
            p = brs.StateInfo()
            p.allocate(len(x))
            for ix, (k, v) in enumerate(x.items()):
                p.quanta[ix] = k
                p.n_states[ix] = v
            basis[ib] = p
            p.sort_states()
        info = brs.MPSInfo(n_sites, vacuum, target, brs.VectorStateInfo(basis))
        info.tag = tag
        info.set_bond_dimension_full_fci(left_vacuum, vacuum)
        info.left_dims[0] = brs.StateInfo(left_vacuum)
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
        info.bond_dim = info.get_max_bond_dimension()
        info.save_mutable()
        info.save_data("%s/%s-mps_info.bin" % (save_dir, tag))
        tensors = [bs.SparseTensor() for _ in range(n_sites)]
        for i, bb in enumerate(basis):
            tensors[i].data = bs.VectorVectorPSSTensor(
                [bs.VectorPSSTensor() for _ in range(bb.n)]
            )
            for block in mps[i].blocks:
                if i == 0:
                    ql = left_vacuum
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
                tensors[i].data[im].append(
                    ((ql, qr), bx.Tensor(b.VectorMKLInt(blk.shape)))
                )
                np.array(tensors[i].data[im][-1][1], copy=False)[:] = blk

        umps = bs.UnfusedMPS()
        umps.info = info
        umps.n_sites = n_sites
        umps.canonical_form = (
            "L" * center
            + ("S" if center == n_sites - 1 else "K")
            + "R" * (n_sites - center - 1)
        )
        umps.center = center
        umps.dot = 1
        umps.tensors = bs.VectorSpTensor(tensors)
        return umps.finalize()

    @staticmethod
    def trans_sz_to_su2(
        pyzket, sz_basis, sz_target, target_twos=0, left_vacuum=None, cutoff=1e-10
    ):
        """Translate SZ pyblock2 MPS to SU2 MPS."""
        from block2 import SU2, SU2CG

        n_sites = len(pyzket.tensors)
        if left_vacuum is None:
            left_vacuum = SU2(target_twos % 2, target_twos, 0)
        sz_shift = -sz_target
        sz_shift.n = left_vacuum.n
        sz_shift.pg = 0
        su2_target = SU2(sz_target.n + left_vacuum.n, 0, sz_target.pg)
        trr = Tensor(
            [
                SubTensor(
                    q_labels=(sz_target + sz_shift, su2_target),
                    reduced=np.array([[1.0]]),
                )
            ]
        )
        cg = SU2CG()
        u_tensors = []
        for k, bz in list(enumerate(sz_basis))[::-1]:
            subz, kzm, finfo = {}, {}, {}
            for zk, v in bz.items():
                uk = SU2(zk.n, abs(zk.twos), zk.pg)
                subz[uk] = subz.get(uk, 0) + v
                kzm[uk] = kzm.get(uk, []) + [(zk, v)]
            kzm = {uk: sorted(v, key=lambda x: x[0].twos) for uk, v in kzm.items()}
            for uk, v in kzm.items():
                ip = 0
                finfo[uk] = {}
                for zk, vv in v:
                    finfo[uk][zk] = ip
                    ip += vv
            trm_blocks = []
            for zk, v in bz.items():
                uk = SU2(zk.n, abs(zk.twos), zk.pg)
                red = np.zeros((subz[uk], v))
                for i in range(v):
                    red[finfo[uk][zk] + i, i] = 1
                trm_blocks.append(SubTensor(q_labels=(uk, zk), reduced=red))
            trm = Tensor(trm_blocks)
            tensor = pyzket.tensors[k]
            blocks = []
            for b in tensor.blocks:
                qs = [xq for xq in b.q_labels]
                if k == 0:
                    qs[-1] = qs[-1] + sz_shift
                    blocks.append(
                        SubTensor(
                            q_labels=(sz_shift,) + tuple(qs), reduced=b.reduced[None]
                        )
                    )
                elif k == n_sites - 1:
                    qs[0] = qs[0] + sz_shift
                    blocks.append(
                        SubTensor(
                            q_labels=tuple(qs) + (sz_target + sz_shift,),
                            reduced=b.reduced[..., None],
                        )
                    )
                else:
                    qs[0] = qs[0] + sz_shift
                    qs[-1] = qs[-1] + sz_shift
                    blocks.append(SubTensor(q_labels=tuple(qs), reduced=b.reduced))
            tensor = Tensor(blocks=blocks)
            tt = Tensor.contract(tensor, trr, idxa=[2], idxb=[0])
            tt = Tensor.contract(trm, tt, idxa=[1], idxb=[1], out_trans=(1, 0, 2))
            subz, kzm, finfo = {}, {}, {}
            for b in tt.blocks:
                ja, jb = b.q_labels[2], b.q_labels[1]
                jcs = [(ja - jb)[ij] for ij in range((ja - jb).count)]
                nja, njb = ja.multiplicity, jb.multiplicity
                sh = b.reduced.shape
                for jc in jcs:
                    if jc not in kzm:
                        kzm[jc] = {}
                    if (ja, jb) not in kzm[jc]:
                        kzm[jc][(ja, jb)] = (sh[2] // nja, sh[1] // njb)
                        subz[jc] = subz.get(jc, 0) + (sh[2] // nja) * (sh[1] // njb)
            for uk, v in kzm.items():
                ip = 0
                finfo[uk] = {}
                for zk, (va, vb) in sorted(v.items()):
                    finfo[uk][zk] = ip
                    ip += va * vb
            new_blocks_map = {}
            for b in tt.blocks:
                ja, jb, mc = b.q_labels[2], b.q_labels[1], b.q_labels[0]
                jcs = [(ja - jb)[ij] for ij in range((ja - jb).count)]
                nja, njb = ja.multiplicity, jb.multiplicity
                sh = b.reduced.shape
                red = b.reduced.reshape((sh[0], njb, sh[1] // njb, nja, sh[2] // nja))
                for jc in jcs:
                    cgmat = np.zeros((nja, njb))
                    for ija in range(nja):
                        for ijb in range(njb):
                            cgmat[ija, ijb] = cg.cg(
                                ja.twos,
                                jb.twos,
                                jc.twos,
                                -ja.twos + 2 * ija,
                                jb.twos - 2 * ijb,
                                mc.twos,
                            )
                    jred = np.einsum("ij,cjbia->cab", cgmat, red, optimize="optimal")
                    jred = jred.reshape((sh[0], (sh[2] // nja) * (sh[1] // njb))).T
                    if (jc, mc) not in new_blocks_map:
                        new_blocks_map[(jc, mc)] = np.zeros((subz[jc], sh[0]))
                    xmat = new_blocks_map[(jc, mc)]
                    va, vb = kzm[jc][(ja, jb)]
                    xmat[finfo[jc][(ja, jb)] : finfo[jc][(ja, jb)] + va * vb] += jred
            newt = Tensor(
                blocks=[
                    SubTensor(q_labels=k, reduced=v) for k, v in new_blocks_map.items()
                ]
            )
            r, q, error = newt.right_compress(cutoff=cutoff, sv_on_l=False)
            reduced_q_blocks = []
            mq_blocks = []
            for jc, mat in q.items():
                assert mat.shape[0] == subz[jc]
                for (ja, jb), ix in finfo[jc].items():
                    va, vb = kzm[jc][(ja, jb)]
                    rmat = (
                        mat[ix : ix + va * vb, :]
                        .reshape((va, vb, -1))
                        .transpose(2, 1, 0)
                    )
                    reduced_q_blocks.append(
                        SubTensor(q_labels=(jc, jb, ja), reduced=rmat)
                    )
                    nja, njb, njc = ja.multiplicity, jb.multiplicity, jc.multiplicity
                    cgmat = np.zeros((nja, njb, njc))
                    for ija in range(nja):
                        for ijb in range(njb):
                            for ijc in range(njc):
                                cgmat[ija, ijb, ijc] = cg.cg(
                                    ja.twos,
                                    jb.twos,
                                    jc.twos,
                                    -ja.twos + 2 * ija,
                                    jb.twos - 2 * ijb,
                                    -jc.twos + 2 * ijc,
                                )
                    mmat = np.einsum(
                        "ijk,cba->kcjbia", cgmat, rmat, optimize="optimal"
                    ).reshape((-1, vb * njb, va * nja))
                    mq_blocks.append(SubTensor(q_labels=(jc, jb, ja), reduced=mmat))
            qred = Tensor(reduced_q_blocks)
            trr = Tensor.contract(tt, Tensor(mq_blocks), idxa=[1, 2], idxb=[1, 2])
            if k == n_sites - 1:
                blocks = []
                for b in qred.blocks:
                    blocks.append(
                        SubTensor(q_labels=b.q_labels[:-1], reduced=b.reduced[..., 0])
                    )
                qred = Tensor(blocks=blocks)
            elif k == 0:
                pref = 0
                for b in trr.blocks:
                    if b.q_labels[1].twos == target_twos:
                        pref = np.sum(b.reduced)
                blocks = []
                for b in qred.blocks:
                    if b.q_labels[0].twos != target_twos:
                        continue
                    blocks.append(
                        SubTensor(q_labels=b.q_labels[1:], reduced=b.reduced[0] * pref)
                    )
                qred = Tensor(blocks=blocks)
            u_tensors.append(qred)
        return MPS(tensors=u_tensors[::-1])


class MPOTools:
    @staticmethod
    def from_block2(bmpo):
        """Translate block2 (un-simplified) MPO to pyblock2 MPO."""
        assert bmpo.schemer is None
        from block2 import OpTypes, QCTypes

        if bmpo.__class__.__name__ == "MPOQC":
            assert bmpo.mode == QCTypes.NC or bmpo.mode == QCTypes.CN
        tensors = [None] * bmpo.n_sites
        # translate operator name symbols to quantum labels
        idx_mps, idx_qss, idx_imps = [], [], []
        for i in range(0, bmpo.n_sites - 1):
            lidx_mp = {}
            lidx_qs = [op.q_label for op in bmpo.left_operator_names[i].data]
            for ip, q in enumerate(lidx_qs):
                if q not in lidx_mp:
                    lidx_mp[q] = []
                lidx_mp[q].append(ip)
            limp = {iv: iiv for _, v in lidx_mp.items() for iiv, iv in enumerate(v)}
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
                            qu = spmat.info.quanta[p].get_bra(spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (qu, qd, qr)
                            spm = np.array(spmat[p])
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx,
                                    reduced=np.zeros((nu, nd, nr), dtype=spm.dtype),
                                )
                            map_blocks[qx].reduced[:, :, ir] += (
                                expr.factor * spmat.factor * spm
                            )
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
                            qu = spmat.info.quanta[p].get_bra(spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (ql, qu, qd)
                            spm = np.array(spmat[p])
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx,
                                    reduced=np.zeros((nl, nu, nd), dtype=spm.dtype),
                                )
                            map_blocks[qx].reduced[il, :, :] += (
                                expr.factor * spmat.factor * spm
                            )
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
                        nl, nr = len(idx_mps[i - 1][ql]), len(idx_mps[i][qr])
                        il, ir = idx_imps[i - 1][j], idx_imps[i][k]
                        for p in range(spmat.info.n):
                            qu = spmat.info.quanta[p].get_bra(spmat.info.delta_quantum)
                            qd = spmat.info.quanta[p].get_ket()
                            nu = spmat.info.n_states_bra[p]
                            nd = spmat.info.n_states_ket[p]
                            qx = (ql, qu, qd, qr)
                            spm = np.array(spmat[p])
                            if np.linalg.norm(spm) == 0:
                                continue
                            if qx not in map_blocks:
                                map_blocks[qx] = SubTensor(
                                    q_labels=qx,
                                    reduced=np.zeros((nl, nu, nd, nr), dtype=spm.dtype),
                                )
                            map_blocks[qx].reduced[il, :, :, ir] += (
                                expr.factor * spmat.factor * spm
                            )
                    elif expr.get_type() == OpTypes.Sum:
                        for xexpr in expr.strings:
                            spmat = ops[xexpr.a.abs()]
                            if spmat.factor == 0 or spmat.info.n == 0:
                                continue
                            ql, qr = idx_qss[i - 1][j], idx_qss[i][k]
                            nl, nr = len(idx_mps[i - 1][ql]), len(idx_mps[i][qr])
                            il, ir = idx_imps[i - 1][j], idx_imps[i][k]
                            for p in range(spmat.info.n):
                                qu = spmat.info.quanta[p].get_bra(
                                    spmat.info.delta_quantum
                                )
                                qd = spmat.info.quanta[p].get_ket()
                                nu = spmat.info.n_states_bra[p]
                                nd = spmat.info.n_states_ket[p]
                                qx = (ql, qu, qd, qr)
                                spm = np.array(spmat[p])
                                if np.linalg.norm(spm) == 0:
                                    continue
                                if qx not in map_blocks:
                                    map_blocks[qx] = SubTensor(
                                        q_labels=qx,
                                        reduced=np.zeros(
                                            (nl, nu, nd, nr), dtype=spm.dtype
                                        ),
                                    )
                                map_blocks[qx].reduced[il, :, :, ir] += (
                                    xexpr.factor * spmat.factor * spm
                                )
                    else:
                        assert False
            tensors[i] = Tensor(blocks=list(map_blocks.values()))
        return MPO(tensors=tensors, const_e=bmpo.const_e)

    @staticmethod
    def to_block2(mpo, basis, tag="PYMPO", add_ident=True):
        """
        Translate pyblock2 MPO to block2 MPO.
        """
        from collections import Counter

        Q = mpo.tensors[0].blocks[0].q_labels[0].__class__
        DT = mpo.tensors[0].blocks[0].reduced.dtype
        b, bs, brs, _ = init_block2_types(Q, DT)
        n_sites = len(mpo.tensors)
        bmpo = bs.MPO(n_sites, tag)
        tensors, lops, rops, site_op_infos = [], [], [], []
        site_basis = [None] * n_sites
        for ib, x in enumerate(basis):
            p = brs.StateInfo()
            p.allocate(len(x))
            for ix, (k, v) in enumerate(x.items()):
                p.quanta[ix] = k
                p.n_states[ix] = v
            site_basis[ib] = p
            p.sort_states()
        vacuum = (
            mpo.tensors[0].blocks[0].q_labels[0] - mpo.tensors[0].blocks[0].q_labels[0]
        )[0]
        mid_dims = mpo.get_bond_dims()
        left_dims = [Counter({vacuum: 1})] + mid_dims
        right_dims = mid_dims + [Counter({vacuum: 1})]
        for i in range(n_sites):
            ts = mpo.tensors[i]
            tensors.append(bs.OperatorTensor())
            site_op_infos.append({})
            dalloc = b.DoubleVectorAllocator()
            ialloc = b.IntVectorAllocator()
            n_rows = sum([v for v in left_dims[i].values()])
            n_cols = sum([v for v in right_dims[i].values()])
            left_qs = [k for k, v in sorted(left_dims[i].items()) for _ in range(v)]
            right_qs = [k for k, v in sorted(right_dims[i].items()) for _ in range(v)]
            left_acc = Counter()
            right_acc = Counter()
            iv = 0
            for k, nv in sorted(left_dims[i].items()):
                left_acc[k] = iv
                iv += nv
            iv = 0
            for k, nv in sorted(right_dims[i].items()):
                right_acc[k] = iv
                iv += nv
            data_dict = {}
            for block in ts.blocks:
                xqq, xmm = block.q_labels, block.reduced
                qnr, qnc = vacuum, vacuum
                if i != n_sites - 1:
                    qnc, xqq = block.q_labels[-1], xqq[:-1]
                else:
                    xmm = xmm[..., None]
                if i != 0:
                    qnr, xqq = block.q_labels[0], xqq[1:]
                else:
                    xmm = xmm[None, ...]
                nnr, nnc = left_dims[i][qnr], right_dims[i][qnc]
                inr, inc = left_acc[qnr], right_acc[qnc]
                for iir in range(inr, inr + nnr):
                    for iic in range(inc, inc + nnc):
                        zmm = xmm[iir - inr, ..., iic - inc]
                        if np.linalg.norm(zmm) < 1e-12:
                            continue
                        if (iir, iic) not in data_dict:
                            data_dict[(iir, iic)] = []
                        data_dict[(iir, iic)].append((xqq, zmm))
            for kk, (k, v) in enumerate(sorted(data_dict.items())):
                dq = (v[0][0][0] - v[0][0][1])[0]
                if i == 0:
                    xexpr = bs.OpElement(
                        b.OpNames.XL,
                        b.SiteIndex((k[1] // 1000, k[1] % 1000), ()),
                        dq,
                        1.0,
                    )
                elif i == n_sites - 1:
                    xexpr = bs.OpElement(
                        b.OpNames.XR,
                        b.SiteIndex((k[0] // 1000, k[0] % 1000), ()),
                        dq,
                        1.0,
                    )
                else:
                    xexpr = bs.OpElement(
                        b.OpNames.X, b.SiteIndex((kk // 1000, kk % 1000), ()), dq, 1.0
                    )
                if dq not in site_op_infos[i]:
                    site_op_infos[i][dq] = brs.SparseMatrixInfo(ialloc)
                    site_op_infos[i][dq].initialize(
                        site_basis[i], site_basis[i], dq, dq.is_fermion
                    )
                minfo = site_op_infos[i][dq]
                xmat = bs.SparseMatrix(dalloc)
                xmat.allocate(minfo)
                for (ql, qr), mm in v:
                    iq = xmat.info.find_state(dq.combine(ql, qr))
                    xmat[iq] = np.asarray(mm).ravel()
                tensors[i].ops[xexpr] = xmat
            idq = vacuum
            iop = bs.OpElement(b.OpNames.I, b.SiteIndex(), idq, 1.0)
            if iop not in tensors[i].ops:
                if idq not in site_op_infos[i]:
                    site_op_infos[i][idq] = brs.SparseMatrixInfo(ialloc)
                    site_op_infos[i][idq].initialize(
                        site_basis[i], site_basis[i], idq, idq.is_fermion
                    )
                minfo = site_op_infos[i][idq]
                xmat = bs.SparseMatrix(dalloc)
                xmat.allocate(minfo)
                for ix in range(minfo.n):
                    xmat[ix] = np.identity(minfo.n_states_ket[ix]).ravel()
                tensors[i].ops[iop] = xmat
            lopd = [
                bs.OpElement(
                    b.OpNames.XL, b.SiteIndex((k // 1000, k % 1000), ()), q, 1.0
                )
                for k, q in enumerate(right_qs)
            ]
            ropd = [
                bs.OpElement(
                    b.OpNames.XR, b.SiteIndex((k // 1000, k % 1000), ()), -q, 1.0
                )
                for k, q in enumerate(left_qs)
            ]
            if i == 0:
                tensors[i].lmat = brs.SymbolicRowVector(n_cols)
                tensors[i].lmat.data = brs.VectorOpExpr(lopd)
                for iz, zz in enumerate(lopd):
                    if zz not in tensors[i].ops:
                        tensors[i].lmat.data[iz] = brs.OpExpr()
            elif i == n_sites - 1:
                tensors[i].lmat = brs.SymbolicColumnVector(n_rows)
                tensors[i].lmat.data = brs.VectorOpExpr(ropd)
                for iz, zz in enumerate(ropd):
                    if zz not in tensors[i].ops:
                        tensors[i].lmat.data[iz] = brs.OpExpr()
            else:
                matx = [
                    bs.OpElement(
                        b.OpNames.X,
                        b.SiteIndex((kk // 1000, kk % 1000), ()),
                        (v[0][0][0] - v[0][0][1])[0],
                        1.0,
                    )
                    for kk, (_, v) in enumerate(sorted(data_dict.items()))
                ]
                tensors[i].lmat = brs.SymbolicMatrix(n_rows, n_cols)
                tensors[i].lmat.indices = b.VectorPIntInt(
                    [k for k in sorted(data_dict)]
                )
                tensors[i].lmat.data = brs.VectorOpExpr(matx)
            tensors[i].rmat = tensors[i].lmat
            rops.append(brs.SymbolicColumnVector(len(ropd)))
            lops.append(brs.SymbolicRowVector(len(lopd)))
            rops[i].data = brs.VectorOpExpr(ropd)
            lops[i].data = brs.VectorOpExpr(lopd)
            site_op_infos[i] = brs.VectorPLMatInfo(sorted(site_op_infos[i].items()))
        bmpo.const_e = mpo.const_e
        bmpo.tf = bs.TensorFunctions(bs.OperatorFunctions(brs.CG()))
        bmpo.site_op_infos = brs.VectorVectorPLMatInfo(site_op_infos)
        bmpo.basis = brs.VectorStateInfo(site_basis)
        bmpo.sparse_form = "N" * n_sites
        bmpo.op = bs.OpElement(b.OpNames.H, b.SiteIndex(), lops[-1][0].q_label, 1.0)
        bmpo.right_operator_names = brs.VectorSymbolic(rops)
        bmpo.left_operator_names = brs.VectorSymbolic(lops)
        bmpo.tensors = bs.VectorOpTensor(tensors)
        bmpo.left_vacuum = vacuum
        # sanity check
        for ii in range(0, bmpo.n_sites):
            for k, v in bmpo.tensors[ii].ops.items():
                assert k.q_label == v.info.delta_quantum
            mat = bmpo.tensors[ii].lmat
            lop = bmpo.left_operator_names[ii].data
            rop = bmpo.right_operator_names[ii].data
            if ii == 0:
                for iop in range(len(lop)):
                    if mat.data[iop].get_type() != b.OpTypes.Zero:
                        assert mat.data[iop].q_label == lop[iop].q_label
            elif ii == bmpo.n_sites - 1:
                for iop in range(len(rop)):
                    if mat.data[iop].get_type() != b.OpTypes.Zero:
                        assert mat.data[iop].q_label == rop[iop].q_label
            else:
                llop = bmpo.left_operator_names[ii - 1].data
                rrop = bmpo.right_operator_names[ii + 1].data
                assert len(lop) == len(rrop)
                for iop in range(len(lop)):
                    assert lop[iop].q_label == -rrop[iop].q_label
                for ig in range(len(mat.data)):
                    sl = llop[mat.indices[ig][0]].q_label
                    sr = lop[mat.indices[ig][1]].q_label
                    sm = mat.data[ig].q_label
                    assert sl + sm == sr
        for ii in range(0, bmpo.n_sites):
            bmpo.save_tensor(ii)
            bmpo.unload_tensor(ii)
            bmpo.save_left_operators(ii)
            bmpo.unload_left_operators(ii)
            bmpo.save_right_operators(ii)
            bmpo.unload_right_operators(ii)
        bmpo = bs.SimplifiedMPO(bmpo, bs.Rule(), False, False)
        if add_ident:
            bmpo = bs.IdentityAddedMPO(bmpo)
        return bmpo
