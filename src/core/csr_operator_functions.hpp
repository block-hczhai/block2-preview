
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
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

#include "csr_matrix_functions.hpp"
#include "csr_sparse_matrix.hpp"
#include "operator_functions.hpp"
#include <cassert>
#include <memory>

using namespace std;

namespace block2 {

// CSR Block-sparse Matrix operations
template <typename S, typename FL>
struct CSROperatorFunctions : OperatorFunctions<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using OperatorFunctions<S, FL>::cg;
    using OperatorFunctions<S, FL>::seq;
    CSROperatorFunctions(const shared_ptr<CG<S>> &cg)
        : OperatorFunctions<S, FL>(cg) {}
    SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::CSR;
    }
    shared_ptr<OperatorFunctions<S, FL>> copy() const override {
        shared_ptr<OperatorFunctions<S, FL>> opf =
            make_shared<CSROperatorFunctions<S, FL>>(this->cg);
        opf->seq = this->seq->copy();
        return opf;
    }
    // a += b * scale
    void iadd(const shared_ptr<SparseMatrix<S, FL>> &a,
              const shared_ptr<SparseMatrix<S, FL>> &b, FL scale = 1.0,
              bool conj = false) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::iadd(a, b, scale, conj);
        assert(a->get_type() == SparseMatrixTypes::CSR &&
               b->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S, FL>> ca =
            dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a);
        shared_ptr<CSRSparseMatrix<S, FL>> cb =
            dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b);
        if (a->info == b->info && !conj) {
            if (a->factor != (FL)1.0) {
                for (int i = 0; i < ca->info->n; i++)
                    GCSRMatrixFunctions<FL>::iscale((*ca)[i], a->factor);
                a->factor = 1.0;
            }
            if (scale != (FL)0.0)
                for (int i = 0; i < a->info->n; i++)
                    GCSRMatrixFunctions<FL>::iadd((*ca)[i], (*cb)[i],
                                                  scale * b->factor);
        } else {
            S bdq = b->info->delta_quantum;
            for (int ia = 0, ib; ia < a->info->n; ia++) {
                S bra = a->info->quanta[ia].get_bra(a->info->delta_quantum);
                S ket = a->info->quanta[ia].get_ket();
                S bq = conj ? bdq.combine(ket, bra) : bdq.combine(bra, ket);
                if (bq != S(S::invalid) &&
                    ((ib = b->info->find_state(bq)) != -1)) {
                    FL factor = scale * b->factor;
                    if (conj)
                        factor *= (FP)cg->transpose_cg(bdq, bra, ket);
                    if (a->factor != (FP)1.0)
                        GCSRMatrixFunctions<FL>::iscale((*ca)[ia], a->factor);
                    if (factor != (FP)0.0)
                        GCSRMatrixFunctions<FL>::iadd((*ca)[ia], (*cb)[ib],
                                                      factor, conj);
                }
            }
            a->factor = (FP)1;
        }
    }
    void tensor_rotate(const shared_ptr<SparseMatrix<S, FL>> &a,
                       const shared_ptr<SparseMatrix<S, FL>> &c,
                       const shared_ptr<SparseMatrix<S, FL>> &rot_bra,
                       const shared_ptr<SparseMatrix<S, FL>> &rot_ket,
                       bool trans, FL scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            rot_bra->get_type() == SparseMatrixTypes::Normal &&
            rot_ket->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::tensor_rotate(
                a, c, rot_bra, rot_ket, trans, scale);
        assert(a->get_type() == SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(rot_bra->get_type() != SparseMatrixTypes::CSR);
        assert(rot_ket->get_type() != SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S, FL>> ca =
            dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a);
        scale = scale * a->factor * rot_bra->factor * rot_ket->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, cdq = c->info->delta_quantum;
        assert(adq == cdq && a->info->n >= c->info->n);
        for (int ic = 0, ia = 0; ic < c->info->n; ia++, ic++) {
            while (a->info->quanta[ia] != c->info->quanta[ic])
                ia++;
            S cq = c->info->quanta[ic].get_bra(cdq);
            S cqprime = c->info->quanta[ic].get_ket();
            int ibra = rot_bra->info->find_state(cq);
            int iket = rot_ket->info->find_state(cqprime);
            GCSRMatrixFunctions<FL>::rotate((*ca)[ia], (*c)[ic],
                                            (*rot_bra)[ibra], !trans,
                                            (*rot_ket)[iket], trans, scale);
        }
    }
    void tensor_product_diagonal(uint8_t conj,
                                 const shared_ptr<SparseMatrix<S, FL>> &a,
                                 const shared_ptr<SparseMatrix<S, FL>> &b,
                                 const shared_ptr<SparseMatrix<S, FL>> &c,
                                 S opdq, FL scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::tensor_product_diagonal(
                conj, a, b, c, opdq, scale);
        shared_ptr<CSRSparseMatrix<S, FL>> ca, cb;
        int idiag = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a), idiag |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b), idiag |= 2;
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(idiag == 1 || idiag == 2 || idiag == 3);
        scale = scale * a->factor * b->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            double factor = cinfo->factor[il];
            switch (idiag) {
            case 1:
                GCSRMatrixFunctions<FL>::tensor_product_diagonal(
                    conj, (*ca)[ia], (*b)[ib], (*c)[ic], scale * (FP)factor);
                break;
            case 2:
                GCSRMatrixFunctions<FL>::tensor_product_diagonal(
                    conj, (*a)[ia], (*cb)[ib], (*c)[ic], scale * (FP)factor);
                break;
            case 3:
                GCSRMatrixFunctions<FL>::tensor_product_diagonal(
                    conj, (*ca)[ia], (*cb)[ib], (*c)[ic], scale * (FP)factor);
                break;
            default:
                assert(false);
            }
        }
    }
    // b = < v | a | c >
    void
    tensor_left_partial_expectation(uint8_t conj,
                                    const shared_ptr<SparseMatrix<S, FL>> &a,
                                    const shared_ptr<SparseMatrix<S, FL>> &b,
                                    const shared_ptr<SparseMatrix<S, FL>> &c,
                                    const shared_ptr<SparseMatrix<S, FL>> &v,
                                    S opdq, FL scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            v->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::tensor_left_partial_expectation(
                conj, a, b, c, v, opdq, scale);
        shared_ptr<CSRSparseMatrix<S, FL>> ca, cb;
        int irot = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a), irot |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b), irot |= 2;
        assert(v->get_type() != SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(irot == 1);
        scale = scale * a->factor * v->factor * c->factor;
        assert(b->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = cinfo->stride[il];
            double factor = cinfo->factor[il];
            seq->cumulative_nflop +=
                (size_t)(*c)[ic].m * (*c)[ic].n *
                    ((conj & 1) ? (*a)[ia].m : (*a)[ia].n) +
                (size_t)(*v)[iv].m * (*v)[iv].n *
                    ((conj & 1) ? (*a)[ia].n : (*a)[ia].m);
            GCSRMatrixFunctions<FL>::left_partial_rotate(
                (*ca)[ia], conj & 1, (*b)[ib], (conj & 2) >> 1, (*v)[iv],
                (*c)[ic], scale * (FP)factor);
        }
    }
    // a = < v | b | c >
    void
    tensor_right_partial_expectation(uint8_t conj,
                                     const shared_ptr<SparseMatrix<S, FL>> &a,
                                     const shared_ptr<SparseMatrix<S, FL>> &b,
                                     const shared_ptr<SparseMatrix<S, FL>> &c,
                                     const shared_ptr<SparseMatrix<S, FL>> &v,
                                     S opdq, FL scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            v->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::tensor_right_partial_expectation(
                conj, a, b, c, v, opdq, scale);
        shared_ptr<CSRSparseMatrix<S, FL>> ca, cb;
        int irot = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a), irot |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b), irot |= 2;
        assert(v->get_type() != SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(irot == 1);
        scale = scale * b->factor * v->factor * c->factor;
        assert(a->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = cinfo->stride[il];
            double factor = cinfo->factor[il];
            seq->cumulative_nflop +=
                (size_t)(*c)[ic].m * (*c)[ic].n *
                    ((conj & 2) ? (*b)[ib].m : (*b)[ib].n) +
                (size_t)(*v)[iv].m * (*v)[iv].n *
                    ((conj & 2) ? (*b)[ib].n : (*b)[ib].m);
            GCSRMatrixFunctions<FL>::right_partial_rotate(
                (*cb)[ib], (conj & 2) >> 1, (*a)[ia], conj & 1, (*v)[iv],
                (*c)[ic], scale * (FP)factor);
        }
    }
    // v = (a x b) @ c
    void tensor_product_multiply(
        uint8_t conj, const shared_ptr<SparseMatrix<S, FL>> &a,
        const shared_ptr<SparseMatrix<S, FL>> &b,
        const shared_ptr<SparseMatrix<S, FL>> &c,
        const shared_ptr<SparseMatrix<S, FL>> &v, S opdq, FL scale = 1.0,
        TraceTypes tt = TraceTypes::None) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            v->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::tensor_product_multiply(
                conj, a, b, c, v, opdq, scale, tt);
        shared_ptr<CSRSparseMatrix<S, FL>> ca, cb;
        int irot = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a), irot |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b), irot |= 2;
        assert(v->get_type() != SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(irot == 1 || irot == 2 || irot == 3);
        scale = scale * a->factor * b->factor * c->factor;
        assert(v->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = cinfo->stride[il];
            double factor = cinfo->factor[il];
            seq->cumulative_nflop +=
                (size_t)(*c)[ic].m * (*c)[ic].n *
                    ((conj & 2) ? (*b)[ib].n : (*b)[ib].n) +
                (size_t)(*a)[ia].m * (*a)[ia].n *
                    ((conj & 2) ? (*b)[ib].n : (*b)[ib].n);
            switch (irot | ((int)tt << 2)) {
            case 1:
                GCSRMatrixFunctions<FL>::rotate((*c)[ic], (*v)[iv], (*ca)[ia],
                                                conj & 1, (*b)[ib], !(conj & 2),
                                                scale * (FP)factor);
                break;
            case 2:
                GCSRMatrixFunctions<FL>::rotate(
                    (*c)[ic], (*v)[iv], (*a)[ia], conj & 1, (*cb)[ib],
                    !(conj & 2), scale * (FP)factor);
                break;
            case 3:
                GCSRMatrixFunctions<FL>::rotate(
                    (*c)[ic], (*v)[iv], (*ca)[ia], conj & 1, (*cb)[ib],
                    !(conj & 2), scale * (FP)factor);
                break;
            case 1 | ((int)TraceTypes::Left << 2):
                GMatrixFunctions<FL>::multiply((*c)[ic], false, (*b)[ib],
                                               !(conj & 2), (*v)[iv],
                                               scale * (FP)factor, 1.0);
                break;
            case 2 | ((int)TraceTypes::Left << 2):
            case 3 | ((int)TraceTypes::Left << 2):
                GCSRMatrixFunctions<FL>::multiply((*c)[ic], false, (*cb)[ib],
                                                  !(conj & 2), (*v)[iv],
                                                  scale * (FP)factor, 1.0);
                break;
            case 1 | ((int)TraceTypes::Right << 2):
            case 3 | ((int)TraceTypes::Right << 2):
                GCSRMatrixFunctions<FL>::multiply((*ca)[ia], conj & 1, (*c)[ic],
                                                  false, (*v)[iv],
                                                  scale * (FP)factor, 1.0);
                break;
            case 2 | ((int)TraceTypes::Right << 2):
                GMatrixFunctions<FL>::multiply((*a)[ia], conj & 1, (*c)[ic],
                                               false, (*v)[iv],
                                               scale * (FP)factor, 1.0);
                break;
            default:
                assert(false);
            }
        }
    }
    void tensor_product(uint8_t conj, const shared_ptr<SparseMatrix<S, FL>> &a,
                        const shared_ptr<SparseMatrix<S, FL>> &b,
                        const shared_ptr<SparseMatrix<S, FL>> &c,
                        FL scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::tensor_product(conj, a, b, c,
                                                            scale);
        shared_ptr<CSRSparseMatrix<S, FL>> ca, cb, cc;
        int itp = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a), itp |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b), itp |= 2;
        if (c->get_type() == SparseMatrixTypes::CSR)
            cc = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(c), itp |= 4;
        assert(itp == 5 || itp == 6 || itp == 7);
        scale = scale * a->factor * b->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum,
          cdq = c->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = cdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            uint32_t stride = cinfo->stride[il];
            double factor = cinfo->factor[il];
            switch (itp) {
            case 5:
                GCSRMatrixFunctions<FL>::tensor_product(
                    (*ca)[ia], conj & 1, (*b)[ib], (conj & 2) >> 1, (*cc)[ic],
                    scale * (FP)factor, stride);
                break;
            case 6:
                GCSRMatrixFunctions<FL>::tensor_product(
                    (*a)[ia], conj & 1, (*cb)[ib], (conj & 2) >> 1, (*cc)[ic],
                    scale * (FP)factor, stride);
                break;
            case 7:
                GCSRMatrixFunctions<FL>::tensor_product(
                    (*ca)[ia], conj & 1, (*cb)[ib], (conj & 2) >> 1, (*cc)[ic],
                    scale * (FP)factor, stride);
                break;
            default:
                assert(false);
            }
        }
    }
    // dot product with no complex conj
    FL dot_product(const shared_ptr<SparseMatrix<S, FL>> &a,
                   const shared_ptr<SparseMatrix<S, FL>> &b,
                   FL scale = 1.0) override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S, FL>::dot_product(a, b, scale);
        shared_ptr<CSRSparseMatrix<S, FL>> ca, cb;
        int itp = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(a), itp |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(b), itp |= 2;
        assert(itp == 1 || itp == 2 || itp == 3);
        assert(a->info->n == b->info->n &&
               a->info->delta_quantum == b->info->delta_quantum);
        FL r = 0;
        for (int i = 0; i < a->info->n; i++) {
            GCSRMatrix<FL> ma, mb;
            switch (itp) {
            case 1:
                ma = (*ca)[i];
                mb = GCSRMatrix<FL>((*b)[i].m, (*b)[i].n,
                                    (MKL_INT)(*b)[i].size(), (*b)[i].data,
                                    nullptr, nullptr);
                break;
            case 2:
                ma = GCSRMatrix<FL>((*a)[i].m, (*a)[i].n,
                                    (MKL_INT)(*a)[i].size(), (*a)[i].data,
                                    nullptr, nullptr);
                mb = (*cb)[i];
                break;
            case 3:
                ma = (*ca)[i];
                mb = (*cb)[i];
                break;
            default:
                assert(false);
            }
            r += GCSRMatrixFunctions<FL>::sparse_dot(ma, mb);
        }
        seq->cumulative_nflop += a->info->get_total_memory();
        return r * scale;
    }
};

} // namespace block2
