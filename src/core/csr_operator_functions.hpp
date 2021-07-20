
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
template <typename S> struct CSROperatorFunctions : OperatorFunctions<S> {
    using OperatorFunctions<S>::cg;
    using OperatorFunctions<S>::seq;
    CSROperatorFunctions(const shared_ptr<CG<S>> &cg)
        : OperatorFunctions<S>(cg) {}
    SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::CSR;
    }
    shared_ptr<OperatorFunctions<S>> copy() const override {
        shared_ptr<OperatorFunctions<S>> opf =
            make_shared<CSROperatorFunctions<S>>(this->cg);
        opf->seq = this->seq->copy();
        return opf;
    }
    // a += b * scale
    void iadd(const shared_ptr<SparseMatrix<S>> &a,
              const shared_ptr<SparseMatrix<S>> &b, double scale = 1.0,
              bool conj = false) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::iadd(a, b, scale, conj);
        assert(a->get_type() == SparseMatrixTypes::CSR &&
               b->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S>> ca =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(a);
        shared_ptr<CSRSparseMatrix<S>> cb =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(b);
        if (a->info == b->info && !conj) {
            if (a->factor != 1.0) {
                for (int i = 0; i < ca->info->n; i++)
                    CSRMatrixFunctions::iscale((*ca)[i], a->factor);
                a->factor = 1.0;
            }
            if (scale != 0.0)
                for (int i = 0; i < a->info->n; i++)
                    CSRMatrixFunctions::iadd((*ca)[i], (*cb)[i],
                                             scale * b->factor);
        } else {
            S bdq = b->info->delta_quantum;
            for (int ia = 0, ib; ia < a->info->n; ia++) {
                S bra = a->info->quanta[ia].get_bra(a->info->delta_quantum);
                S ket = a->info->quanta[ia].get_ket();
                S bq = conj ? bdq.combine(ket, bra) : bdq.combine(bra, ket);
                if (bq != S(S::invalid) &&
                    ((ib = b->info->find_state(bq)) != -1)) {
                    double factor = scale * b->factor;
                    if (conj)
                        factor *= cg->transpose_cg(bdq.twos(), bra.twos(),
                                                   ket.twos());
                    if (a->factor != 1.0)
                        CSRMatrixFunctions::iscale((*ca)[ia], a->factor);
                    if (factor != 0.0)
                        CSRMatrixFunctions::iadd((*ca)[ia], (*cb)[ib], factor,
                                                 conj);
                }
            }
            a->factor = 1;
        }
    }
    void tensor_rotate(const shared_ptr<SparseMatrix<S>> &a,
                       const shared_ptr<SparseMatrix<S>> &c,
                       const shared_ptr<SparseMatrix<S>> &rot_bra,
                       const shared_ptr<SparseMatrix<S>> &rot_ket, bool trans,
                       double scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            rot_bra->get_type() == SparseMatrixTypes::Normal &&
            rot_ket->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::tensor_rotate(a, c, rot_bra, rot_ket,
                                                       trans, scale);
        assert(a->get_type() == SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(rot_bra->get_type() != SparseMatrixTypes::CSR);
        assert(rot_ket->get_type() != SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S>> ca =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(a);
        scale = scale * a->factor * rot_bra->factor * rot_ket->factor;
        assert(c->factor == 1.0);
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
            CSRMatrixFunctions::rotate((*ca)[ia], (*c)[ic], (*rot_bra)[ibra],
                                       !trans, (*rot_ket)[iket], trans, scale);
        }
    }
    void tensor_product_diagonal(uint8_t conj,
                                 const shared_ptr<SparseMatrix<S>> &a,
                                 const shared_ptr<SparseMatrix<S>> &b,
                                 const shared_ptr<SparseMatrix<S>> &c, S opdq,
                                 double scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::tensor_product_diagonal(conj, a, b, c,
                                                                 opdq, scale);
        shared_ptr<CSRSparseMatrix<S>> ca, cb;
        int idiag = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S>>(a), idiag |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S>>(b), idiag |= 2;
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(idiag == 1 || idiag == 2 || idiag == 3);
        scale = scale * a->factor * b->factor;
        assert(c->factor == 1.0);
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
                CSRMatrixFunctions::tensor_product_diagonal(
                    (*ca)[ia], (*b)[ib], (*c)[ic], scale * factor);
                break;
            case 2:
                CSRMatrixFunctions::tensor_product_diagonal(
                    (*a)[ia], (*cb)[ib], (*c)[ic], scale * factor);
                break;
            case 3:
                CSRMatrixFunctions::tensor_product_diagonal(
                    (*ca)[ia], (*cb)[ib], (*c)[ic], scale * factor);
                break;
            default:
                assert(false);
            }
        }
    }
    // b = < v | a | c >
    void tensor_partial_expectation(uint8_t conj,
                                    const shared_ptr<SparseMatrix<S>> &a,
                                    const shared_ptr<SparseMatrix<S>> &b,
                                    const shared_ptr<SparseMatrix<S>> &c,
                                    const shared_ptr<SparseMatrix<S>> &v,
                                    S opdq, double scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            v->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::tensor_partial_expectation(
                conj, a, b, c, v, opdq, scale);
        shared_ptr<CSRSparseMatrix<S>> ca, cb;
        int irot = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S>>(a), irot |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S>>(b), irot |= 2;
        assert(v->get_type() != SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(irot == 1);
        scale = scale * a->factor * v->factor * c->factor;
        assert(b->factor == 1.0);
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
            CSRMatrixFunctions::rotate((*ca)[ia], conj & 1, (*b)[ib], conj & 2,
                                       (*v)[iv], (*c)[ic], scale * factor);
        }
    }
    // v = (a x b) @ c
    void
    tensor_product_multiply(uint8_t conj, const shared_ptr<SparseMatrix<S>> &a,
                            const shared_ptr<SparseMatrix<S>> &b,
                            const shared_ptr<SparseMatrix<S>> &c,
                            const shared_ptr<SparseMatrix<S>> &v, S opdq,
                            double scale = 1.0,
                            TraceTypes tt = TraceTypes::None) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal &&
            v->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::tensor_product_multiply(
                conj, a, b, c, v, opdq, scale, tt);
        shared_ptr<CSRSparseMatrix<S>> ca, cb;
        int irot = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S>>(a), irot |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S>>(b), irot |= 2;
        assert(v->get_type() != SparseMatrixTypes::CSR);
        assert(c->get_type() != SparseMatrixTypes::CSR);
        assert(irot == 1 || irot == 2 || irot == 3);
        scale = scale * a->factor * b->factor * c->factor;
        assert(v->factor == 1.0);
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
                CSRMatrixFunctions::rotate((*c)[ic], (*v)[iv], (*ca)[ia],
                                           conj & 1, (*b)[ib], !(conj & 2),
                                           scale * factor);
                break;
            case 2:
                CSRMatrixFunctions::rotate((*c)[ic], (*v)[iv], (*a)[ia],
                                           conj & 1, (*cb)[ib], !(conj & 2),
                                           scale * factor);
                break;
            case 3:
                CSRMatrixFunctions::rotate((*c)[ic], (*v)[iv], (*ca)[ia],
                                           conj & 1, (*cb)[ib], !(conj & 2),
                                           scale * factor);
                break;
            case 1 | ((int)TraceTypes::Left << 2):
                MatrixFunctions::multiply((*c)[ic], false, (*b)[ib],
                                          !(conj & 2), (*v)[iv], scale * factor,
                                          1.0);
                break;
            case 2 | ((int)TraceTypes::Left << 2):
            case 3 | ((int)TraceTypes::Left << 2):
                CSRMatrixFunctions::multiply((*c)[ic], false, (*cb)[ib],
                                             !(conj & 2), (*v)[iv],
                                             scale * factor, 1.0);
                break;
            case 1 | ((int)TraceTypes::Right << 2):
            case 3 | ((int)TraceTypes::Right << 2):
                CSRMatrixFunctions::multiply((*ca)[ia], conj & 1, (*c)[ic],
                                             false, (*v)[iv], scale * factor,
                                             1.0);
                break;
            case 2 | ((int)TraceTypes::Right << 2):
                MatrixFunctions::multiply((*a)[ia], conj & 1, (*c)[ic], false,
                                          (*v)[iv], scale * factor, 1.0);
                break;
            default:
                assert(false);
            }
        }
    }
    void tensor_product(uint8_t conj, const shared_ptr<SparseMatrix<S>> &a,
                        const shared_ptr<SparseMatrix<S>> &b,
                        const shared_ptr<SparseMatrix<S>> &c,
                        double scale = 1.0) const override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal &&
            c->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::tensor_product(conj, a, b, c, scale);
        shared_ptr<CSRSparseMatrix<S>> ca, cb, cc;
        int itp = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S>>(a), itp |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S>>(b), itp |= 2;
        if (c->get_type() == SparseMatrixTypes::CSR)
            cc = dynamic_pointer_cast<CSRSparseMatrix<S>>(c), itp |= 4;
        assert(itp == 5 || itp == 6 || itp == 7);
        scale = scale * a->factor * b->factor;
        assert(c->factor == 1.0);
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
                CSRMatrixFunctions::tensor_product(
                    (*ca)[ia], conj & 1, (*b)[ib], (conj & 2) >> 1, (*cc)[ic],
                    scale * factor, stride);
                break;
            case 6:
                CSRMatrixFunctions::tensor_product(
                    (*a)[ia], conj & 1, (*cb)[ib], (conj & 2) >> 1, (*cc)[ic],
                    scale * factor, stride);
                break;
            case 7:
                CSRMatrixFunctions::tensor_product(
                    (*ca)[ia], conj & 1, (*cb)[ib], (conj & 2) >> 1, (*cc)[ic],
                    scale * factor, stride);
                break;
            default:
                assert(false);
            }
        }
    }
    double dot_product(const shared_ptr<SparseMatrix<S>> &a,
                       const shared_ptr<SparseMatrix<S>> &b,
                       double scale = 1.0) override {
        if (a->get_type() == SparseMatrixTypes::Normal &&
            b->get_type() == SparseMatrixTypes::Normal)
            return OperatorFunctions<S>::dot_product(a, b, scale);
        shared_ptr<CSRSparseMatrix<S>> ca, cb;
        int itp = 0;
        if (a->get_type() == SparseMatrixTypes::CSR)
            ca = dynamic_pointer_cast<CSRSparseMatrix<S>>(a), itp |= 1;
        if (b->get_type() == SparseMatrixTypes::CSR)
            cb = dynamic_pointer_cast<CSRSparseMatrix<S>>(b), itp |= 2;
        assert(itp == 1 || itp == 2 || itp == 3);
        assert(a->info->n == b->info->n &&
               a->info->delta_quantum == b->info->delta_quantum);
        double r = 0;
        for (int i = 0; i < a->info->n; i++) {
            CSRMatrixRef ma, mb;
            switch (itp) {
            case 1:
                ma = (*ca)[i];
                mb = CSRMatrixRef((*b)[i].m, (*b)[i].n, (MKL_INT)(*b)[i].size(),
                                  (*b)[i].data, nullptr, nullptr);
                break;
            case 2:
                ma = CSRMatrixRef((*a)[i].m, (*a)[i].n, (MKL_INT)(*a)[i].size(),
                                  (*a)[i].data, nullptr, nullptr);
                mb = (*cb)[i];
                break;
            case 3:
                ma = (*ca)[i];
                mb = (*cb)[i];
                break;
            default:
                assert(false);
            }
            r += CSRMatrixFunctions::sparse_dot(ma, mb);
        }
        seq->cumulative_nflop += a->info->get_total_memory();
        return r * scale;
    }
};

} // namespace block2
