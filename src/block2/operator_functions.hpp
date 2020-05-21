
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include "batch_gemm.hpp"
#include "cg.hpp"
#include "matrix_functions.hpp"
#include "sparse_matrix.hpp"
#include <cassert>
#include <memory>

using namespace std;

namespace block2 {

enum struct NoiseTypes : uint8_t { Wavefunction, DensityMatrix, Perturbative };

// SparseMatrix operations
template <typename S> struct OperatorFunctions {
    shared_ptr<CG<S>> cg;
    shared_ptr<BatchGEMMSeq> seq = nullptr;
    OperatorFunctions(const shared_ptr<CG<S>> &cg) : cg(cg) {
        seq = make_shared<BatchGEMMSeq>(0, SeqTypes::None);
    }
    // a += b * scale
    void iadd(SparseMatrix<S> &a, const SparseMatrix<S> &b, double scale = 1.0,
              bool conj = false) const {
        if (a.info == b.info && !conj) {
            if (seq->mode != SeqTypes::None) {
                seq->iadd(MatrixRef(a.data, 1, a.total_memory),
                          MatrixRef(b.data, 1, b.total_memory),
                          scale * b.factor, a.factor);
                a.factor = 1.0;
            } else {
                if (a.factor != 1.0) {
                    MatrixFunctions::iscale(
                        MatrixRef(a.data, 1, a.total_memory), a.factor);
                    a.factor = 1.0;
                }
                if (scale != 0.0)
                    MatrixFunctions::iadd(MatrixRef(a.data, 1, a.total_memory),
                                          MatrixRef(b.data, 1, b.total_memory),
                                          scale * b.factor);
            }
        } else {
            S bdq = b.info->delta_quantum;
            for (int ia = 0, ib; ia < a.info->n; ia++) {
                S bra = a.info->quanta[ia].get_bra(a.info->delta_quantum);
                S ket = a.info->quanta[ia].get_ket();
                S bq = conj ? bdq.combine(ket, bra) : bdq.combine(bra, ket);
                if (bq != S(0xFFFFFFFF) &&
                    ((ib = b.info->find_state(bq)) != -1)) {
                    double factor = scale * b.factor;
                    if (conj)
                        factor *= cg->transpose_cg(bdq.twos(), bra.twos(),
                                                   ket.twos());
                    if (seq->mode != SeqTypes::None)
                        seq->iadd(a[ia], b[ib], factor, a.factor, conj);
                    else {
                        if (a.factor != 1.0) {
                            MatrixFunctions::iscale(a[ia], a.factor);
                        }
                        if (scale != 0.0)
                            MatrixFunctions::iadd(a[ia], b[ib], factor, conj);
                    }
                }
            }
            a.factor = 1;
        }
    }
    void tensor_rotate(const SparseMatrix<S> &a, const SparseMatrix<S> &c,
                       const SparseMatrix<S> &rot_bra,
                       const SparseMatrix<S> &rot_ket, bool trans,
                       double scale = 1.0) const {
        scale = scale * a.factor * rot_bra.factor * rot_ket.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, cdq = c.info->delta_quantum;
        assert(adq == cdq && a.info->n >= c.info->n);
        for (int ic = 0, ia = 0; ic < c.info->n; ia++, ic++) {
            while (a.info->quanta[ia] != c.info->quanta[ic])
                ia++;
            S cq = c.info->quanta[ic].get_bra(cdq);
            S cqprime = c.info->quanta[ic].get_ket();
            int ibra = rot_bra.info->find_state(cq);
            int iket = rot_ket.info->find_state(cqprime);
            if (seq->mode != SeqTypes::None)
                seq->rotate(a[ia], c[ic], rot_bra[ibra], !trans, rot_ket[iket],
                            trans, scale);
            else
                MatrixFunctions::rotate(a[ia], c[ic], rot_bra[ibra], !trans,
                                        rot_ket[iket], trans, scale);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    void tensor_product_diagonal(uint8_t conj, const SparseMatrix<S> &a,
                                 const SparseMatrix<S> &b,
                                 const SparseMatrix<S> &c, S opdq,
                                 double scale = 1.0) const {
        scale = scale * a.factor * b.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, bdq = b.info->delta_quantum;
        assert(c.info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c.info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = lower_bound(cinfo->quanta + cinfo->n[conj],
                             cinfo->quanta + cinfo->n[conj + 1], abdq) -
                 cinfo->quanta;
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->tensor_product_diagonal(a[ia], b[ib], c[ic],
                                             scale * factor);
            else
                MatrixFunctions::tensor_product_diagonal(a[ia], b[ib], c[ic],
                                                         scale * factor);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    void tensor_product_multiply(uint8_t conj, const SparseMatrix<S> &a,
                                 const SparseMatrix<S> &b,
                                 const SparseMatrix<S> &c,
                                 const SparseMatrix<S> &v, S opdq,
                                 double scale = 1.0) const {
        scale = scale * a.factor * b.factor * c.factor;
        assert(v.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, bdq = b.info->delta_quantum;
        assert(c.info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c.info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = lower_bound(cinfo->quanta + cinfo->n[conj],
                             cinfo->quanta + cinfo->n[conj + 1], abdq) -
                 cinfo->quanta;
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = cinfo->stride[il];
            if (seq->mode == SeqTypes::Simple && il != ixa &&
                iv <= cinfo->stride[il - 1])
                seq->simple_perform();
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->rotate(c[ic], v[iv], a[ia], conj & 1, b[ib], !(conj & 2),
                            scale * factor);
            else {
                seq->cumulative_nflop += (size_t)c[ic].m * c[ic].n *
                                             ((conj & 2) ? b[ib].n : b[ib].n) +
                                         (size_t)a[ia].m * a[ia].n *
                                             ((conj & 2) ? b[ib].n : b[ib].n);
                MatrixFunctions::rotate(c[ic], v[iv], a[ia], conj & 1, b[ib],
                                        !(conj & 2), scale * factor);
            }
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    void tensor_product(uint8_t conj, const SparseMatrix<S> &a,
                        const SparseMatrix<S> &b, SparseMatrix<S> &c,
                        double scale = 1.0) const {
        scale = scale * a.factor * b.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, bdq = b.info->delta_quantum,
          cdq = c.info->delta_quantum;
        assert(c.info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c.info->cinfo;
        S abdq = cdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = lower_bound(cinfo->quanta + cinfo->n[conj],
                             cinfo->quanta + cinfo->n[conj + 1], abdq) -
                 cinfo->quanta;
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            uint32_t stride = cinfo->stride[il];
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->tensor_product(a[ia], conj & 1, b[ib], (conj & 2) >> 1,
                                    c[ic], scale * factor, stride);
            else
                MatrixFunctions::tensor_product(a[ia], conj & 1, b[ib],
                                                (conj & 2) >> 1, c[ic],
                                                scale * factor, stride);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    // c = a * b * scale
    void product(const SparseMatrix<S> &a, const SparseMatrix<S> &b,
                 const SparseMatrix<S> &c, double scale = 1.0) const {
        scale = scale * a.factor * b.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        int adq = a.info->delta_quantum.multiplicity() - 1,
            bdq = b.info->delta_quantum.multiplicity() - 1,
            cdq = c.info->delta_quantum.multiplicity() - 1;
        for (int ic = 0; ic < c.info->n; ic++) {
            S cq = c.info->quanta[ic].get_bra(c.info->delta_quantum);
            S cqprime = c.info->quanta[ic].get_ket();
            S aps = cq - a.info->delta_quantum;
            for (int k = 0; k < aps.count(); k++) {
                S aqprime = aps[k];
                int ia = a.info->find_state(
                    a.info->delta_quantum.combine(cq, aps[k]));
                if (ia != -1) {
                    S bl = b.info->delta_quantum.combine(aqprime, cqprime);
                    if (bl != S(0xFFFFFFFFU)) {
                        int ib = b.info->find_state(bl);
                        if (ib != -1) {
                            int aqpj = aqprime.multiplicity() - 1,
                                cqj = cq.multiplicity() - 1,
                                cqpj = cqprime.multiplicity() - 1;
                            double factor =
                                cg->racah(cqpj, bdq, cqj, adq, aqpj, cdq);
                            factor *= sqrt((cdq + 1) * (aqpj + 1)) *
                                      (((adq + bdq - cdq) & 2) ? -1 : 1);
                            MatrixFunctions::multiply(a[ia], false, b[ib],
                                                      false, c[ic],
                                                      scale * factor, 1.0);
                        }
                    }
                }
            }
        }
    }
    // Product with transposed tensor: [a] x [b]^T or [a]^T x [b]
    static void
    trans_product(const SparseMatrix<S> &a, const SparseMatrix<S> &b,
                  bool trace_right, double noise = 0.0,
                  NoiseTypes noise_type = NoiseTypes::DensityMatrix) {
        double scale = a.factor * a.factor, noise_scale = 0;
        assert(b.factor == 1.0);
        if (abs(scale) < TINY && noise == 0.0)
            return;
        SparseMatrix<S> tmp;
        if (noise != 0 && noise_type == NoiseTypes::Wavefunction) {
            tmp.allocate(a.info);
            tmp.randomize(-0.5, 0.5);
            noise_scale = noise / tmp.norm();
            noise_scale *= noise_scale;
        } else if (noise != 0 && noise_type == NoiseTypes::DensityMatrix) {
            tmp.allocate(b.info);
            tmp.randomize(0.0, 1.0);
            noise_scale = noise * noise / tmp.norm();
        }
        if (trace_right)
            for (int ia = 0; ia < a.info->n; ia++) {
                S qb = a.info->quanta[ia].get_bra(a.info->delta_quantum);
                int ib = b.info->find_state(qb);
                MatrixFunctions::multiply(a[ia], false, a[ia], true, b[ib],
                                          scale, 1.0);
                if (noise_scale != 0 && noise_type == NoiseTypes::Wavefunction)
                    MatrixFunctions::multiply(tmp[ia], false, tmp[ia], true,
                                              b[ib], noise_scale, 1.0);
                else if (noise_scale != 0 &&
                         noise_type == NoiseTypes::DensityMatrix)
                    MatrixFunctions::iadd(b[ib], tmp[ib], noise_scale);
            }
        else
            for (int ia = 0; ia < a.info->n; ia++) {
                S qb = -a.info->quanta[ia].get_ket();
                int ib = b.info->find_state(qb);
                MatrixFunctions::multiply(a[ia], true, a[ia], false, b[ib],
                                          scale, 1.0);
                if (noise_scale != 0 && noise_type == NoiseTypes::Wavefunction)
                    MatrixFunctions::multiply(tmp[ia], true, tmp[ia], false,
                                              b[ib], noise_scale, 1.0);
                else if (noise_scale != 0 &&
                         noise_type == NoiseTypes::DensityMatrix)
                    MatrixFunctions::iadd(b[ib], tmp[ib], noise_scale);
            }
        if (noise != 0)
            tmp.deallocate();
    }
};

} // namespace block2
