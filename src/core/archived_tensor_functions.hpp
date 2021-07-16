
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

#include "archived_sparse_matrix.hpp"
#include "tensor_functions.hpp"

using namespace std;

namespace block2 {

// Operations for operator tensors
template <typename S> struct ArchivedTensorFunctions : TensorFunctions<S> {
    using TensorFunctions<S>::opf;
    string filename = "";
    mutable int64_t offset = 0;
    ArchivedTensorFunctions(const shared_ptr<OperatorFunctions<S>> &opf)
        : TensorFunctions<S>(opf) {}
    TensorFunctionsTypes get_type() const override {
        return TensorFunctionsTypes::Archived;
    }
    void archive_tensor(const shared_ptr<OperatorTensor<S>> &a) const {
        map<double *, vector<shared_ptr<SparseMatrix<S>>>> mp;
        for (auto &op : a->ops) {
            shared_ptr<SparseMatrix<S>> mat = op.second;
            shared_ptr<ArchivedSparseMatrix<S>> arc =
                make_shared<ArchivedSparseMatrix<S>>(filename, offset);
            arc->save_archive(mat);
            mp[op.second->data].push_back(op.second);
            op.second = arc;
            offset += arc->total_memory;
        }
        for (auto it = mp.crbegin(); it != mp.crend(); it++)
            for (const auto &t : it->second)
                t->deallocate();
    }
    // c = a
    void left_assign(const shared_ptr<OperatorTensor<S>> &a,
                     shared_ptr<OperatorTensor<S>> &c) const override {
        assert(a->lmat != nullptr);
        assert(a->lmat->get_type() == SymTypes::RVec);
        assert(c->lmat != nullptr);
        assert(c->lmat->get_type() == SymTypes::RVec);
        assert(a->lmat->data.size() == c->lmat->data.size());
        for (size_t i = 0; i < a->lmat->data.size(); i++) {
            if (a->lmat->data[i]->get_type() == OpTypes::Zero)
                c->lmat->data[i] = a->lmat->data[i];
            else {
                assert(a->lmat->data[i] == c->lmat->data[i]);
                shared_ptr<OpExpr<S>> pa = abs_value(a->lmat->data[i]),
                                      pc = abs_value(c->lmat->data[i]);
                shared_ptr<SparseMatrix<S>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(a->ops[pa])
                        ->load_archive();
                shared_ptr<SparseMatrix<S>> matc = c->ops.at(pc);
                matc->allocate(matc->info);
                if (matc->info->n == mata->info->n)
                    matc->copy_data_from(mata, true);
                else
                    matc->selective_copy_from(mata, true);
                matc->factor = mata->factor;
                shared_ptr<ArchivedSparseMatrix<S>> arc =
                    make_shared<ArchivedSparseMatrix<S>>(filename, offset);
                arc->save_archive(matc);
                matc->deallocate();
                c->ops.at(pc) = arc;
                offset += arc->total_memory;
                mata->deallocate();
            }
        }
    }
    // c = a
    void right_assign(const shared_ptr<OperatorTensor<S>> &a,
                      shared_ptr<OperatorTensor<S>> &c) const override {
        assert(a->rmat != nullptr);
        assert(a->rmat->get_type() == SymTypes::CVec);
        assert(c->rmat != nullptr);
        assert(c->rmat->get_type() == SymTypes::CVec);
        assert(a->rmat->data.size() == c->rmat->data.size());
        for (size_t i = 0; i < a->rmat->data.size(); i++) {
            if (a->rmat->data[i]->get_type() == OpTypes::Zero)
                c->rmat->data[i] = a->rmat->data[i];
            else {
                assert(a->rmat->data[i] == c->rmat->data[i]);
                shared_ptr<OpExpr<S>> pa = abs_value(a->rmat->data[i]),
                                      pc = abs_value(c->rmat->data[i]);
                shared_ptr<SparseMatrix<S>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(a->ops[pa])
                        ->load_archive();
                shared_ptr<SparseMatrix<S>> matc = c->ops.at(pc);
                matc->allocate(matc->info);
                if (matc->info->n == mata->info->n)
                    matc->copy_data_from(mata, true);
                else
                    matc->selective_copy_from(mata, true);
                matc->factor = mata->factor;
                shared_ptr<ArchivedSparseMatrix<S>> arc =
                    make_shared<ArchivedSparseMatrix<S>>(filename, offset);
                arc->save_archive(matc);
                matc->deallocate();
                c->ops.at(pc) = arc;
                offset += arc->total_memory;
                mata->deallocate();
            }
        }
    }
    // vmat = expr[L part | R part] x cmat (for perturbative noise)
    void tensor_product_partial_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S>> &lopt,
        const shared_ptr<OperatorTensor<S>> &ropt, bool trace_right,
        const shared_ptr<SparseMatrix<S>> &cmat,
        const vector<pair<uint8_t, S>> &psubsl,
        const vector<
            vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
            &cinfos,
        const vector<S> &vdqs, const shared_ptr<SparseMatrixGroup<S>> &vmats,
        int &vidx, int tvidx, bool do_reduce) const override {
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), S());
        if ((!trace_right && lopt->ops.count(i_op) == 0) ||
            (trace_right && ropt->ops.count(i_op) == 0))
            return;
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->b != nullptr);
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> old_cinfo =
                cmat->info->cinfo;
            if (trace_right) {
                assert(lopt->ops.count(op->a) != 0 &&
                       ropt->ops.count(i_op) != 0);
                shared_ptr<SparseMatrix<S>> lmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                        lopt->ops.at(op->a))
                        ->load_archive();
                shared_ptr<SparseMatrix<S>> rmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                        ropt->ops.at(i_op))
                        ->load_archive();
                S opdq = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                S pks = cmat->info->delta_quantum + opdq;
                int ij = (int)(lower_bound(
                                   psubsl.begin(), psubsl.end(),
                                   make_pair((uint8_t)(op->conj & 1), opdq)) -
                               psubsl.begin());
                for (int k = 0; k < pks.count(); k++) {
                    S vdq = pks[k];
                    int iv = (int)(lower_bound(vdqs.begin(), vdqs.end(), vdq) -
                                   vdqs.begin());
                    shared_ptr<SparseMatrix<S>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    opf->tensor_product_multiply(op->conj & 1, lmat, rmat, cmat,
                                                 vmat, opdq, op->factor,
                                                 TraceTypes::Right);
                }
                rmat->deallocate();
                lmat->deallocate();
            } else {
                assert(lopt->ops.count(i_op) != 0 &&
                       ropt->ops.count(op->b) != 0);
                shared_ptr<SparseMatrix<S>> lmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                        lopt->ops.at(i_op))
                        ->load_archive();
                shared_ptr<SparseMatrix<S>> rmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                        ropt->ops.at(op->b))
                        ->load_archive();
                S opdq = (op->conj & 2) ? -op->b->q_label : op->b->q_label;
                S pks = cmat->info->delta_quantum + opdq;
                int ij =
                    (int)(lower_bound(
                              psubsl.begin(), psubsl.end(),
                              make_pair((uint8_t)(!!(op->conj & 2)), opdq)) -
                          psubsl.begin());
                for (int k = 0; k < pks.count(); k++) {
                    S vdq = pks[k];
                    int iv = (int)(lower_bound(vdqs.begin(), vdqs.end(), vdq) -
                                   vdqs.begin());
                    shared_ptr<SparseMatrix<S>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    opf->tensor_product_multiply(op->conj & 2, lmat, rmat, cmat,
                                                 vmat, opdq, op->factor,
                                                 TraceTypes::Left);
                }
                rmat->deallocate();
                lmat->deallocate();
            }
            cmat->info->cinfo = old_cinfo;
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product_partial_multiply(x, lopt, ropt, trace_right,
                                                cmat, psubsl, cinfos, vdqs,
                                                vmats, vidx, tvidx, false);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // vmats = expr x cmats
    void tensor_product_multi_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S>> &lopt,
        const shared_ptr<OperatorTensor<S>> &ropt,
        const shared_ptr<SparseMatrixGroup<S>> &cmats,
        const shared_ptr<SparseMatrixGroup<S>> &vmats,
        const unordered_map<
            S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>
            &cinfos,
        S opdq, double factor, bool all_reduce) const override {
        unordered_map<S, int> vdqs;
        vdqs.reserve(vmats->n);
        for (int iv = 0; iv < vmats->n; iv++)
            vdqs[vmats->infos[iv]->delta_quantum] = iv;
        for (int i = 0; i < cmats->n; i++) {
            shared_ptr<SparseMatrix<S>> pcmat = (*cmats)[i];
            pcmat->factor = factor;
            shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                make_shared<SparseMatrixInfo<S>>(*pcmat->info);
            pcmat->info = pcmat_info;
            S cdq = pcmat->info->delta_quantum;
            S vdq = opdq + cdq;
            for (int iv = 0; iv < vdq.count(); iv++)
                if (vdqs.count(vdq[iv])) {
                    pcmat->info->cinfo = cinfos.at(opdq.combine(vdq[iv], cdq));
                    tensor_product_multiply(expr, lopt, ropt, pcmat,
                                            (*vmats)[vdqs[vdq[iv]]], opdq,
                                            false);
                }
        }
    }
    // vmat = expr x cmat
    void tensor_product_multiply(const shared_ptr<OpExpr<S>> &expr,
                                 const shared_ptr<OperatorTensor<S>> &lopt,
                                 const shared_ptr<OperatorTensor<S>> &ropt,
                                 const shared_ptr<SparseMatrix<S>> &cmat,
                                 const shared_ptr<SparseMatrix<S>> &vmat,
                                 S opdq, bool all_reduce) const override {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->b != nullptr);
            assert(
                !(lopt->ops.count(op->a) == 0 || ropt->ops.count(op->b) == 0));
            shared_ptr<SparseMatrix<S>> lmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                    lopt->ops.at(op->a))
                    ->load_archive();
            shared_ptr<SparseMatrix<S>> rmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                    ropt->ops.at(op->b))
                    ->load_archive();
            opf->tensor_product_multiply(op->conj, lmat, rmat, cmat, vmat, opdq,
                                         op->factor);
            rmat->deallocate();
            lmat->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product_multiply(x, lopt, ropt, cmat, vmat, opdq, false);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // mat = diag(expr)
    void tensor_product_diagonal(const shared_ptr<OpExpr<S>> &expr,
                                 const shared_ptr<OperatorTensor<S>> &lopt,
                                 const shared_ptr<OperatorTensor<S>> &ropt,
                                 const shared_ptr<SparseMatrix<S>> &mat,
                                 S opdq) const override {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->b != nullptr);
            assert(
                !(lopt->ops.count(op->a) == 0 || ropt->ops.count(op->b) == 0));
            shared_ptr<SparseMatrix<S>> lmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                    lopt->ops.at(op->a))
                    ->load_archive();
            shared_ptr<SparseMatrix<S>> rmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                    ropt->ops.at(op->b))
                    ->load_archive();
            opf->tensor_product_diagonal(op->conj, lmat, rmat, mat, opdq,
                                         op->factor);
            rmat->deallocate();
            lmat->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product_diagonal(x, lopt, ropt, mat, opdq);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // mat = eval(expr)
    void tensor_product(const shared_ptr<OpExpr<S>> &expr,
                        const unordered_map<shared_ptr<OpExpr<S>>,
                                            shared_ptr<SparseMatrix<S>>> &lop,
                        const unordered_map<shared_ptr<OpExpr<S>>,
                                            shared_ptr<SparseMatrix<S>>> &rop,
                        shared_ptr<SparseMatrix<S>> &mat) const override {
        shared_ptr<ArchivedSparseMatrix<S>> aromat = nullptr;
        shared_ptr<SparseMatrix<S>> omat;
        if (mat->get_type() == SparseMatrixTypes::Archived) {
            aromat = dynamic_pointer_cast<ArchivedSparseMatrix<S>>(mat);
            omat = aromat->load_archive();
        } else
            omat = mat;
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->b != nullptr);
            assert(lop.count(op->a) != 0 && rop.count(op->b) != 0);
            shared_ptr<SparseMatrix<S>> lmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(lop.at(op->a))
                    ->load_archive();
            shared_ptr<SparseMatrix<S>> rmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(rop.at(op->b))
                    ->load_archive();
            opf->tensor_product(op->conj, lmat, rmat, omat, op->factor);
            rmat->deallocate();
            lmat->deallocate();
        } break;
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            assert((op->a == nullptr) ^ (op->b == nullptr));
            assert(op->ops.size() != 0);
            shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
            if (op->c != nullptr && ((op->b == nullptr && rop.count(op->c)) ||
                                     (op->a == nullptr && lop.count(op->c)))) {
                if (op->b == nullptr && rop.count(op->c))
                    tmp = dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                              rop.at(op->c))
                              ->load_archive();
                else
                    tmp = dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                              lop.at(op->c))
                              ->load_archive();
            } else if (op->b == nullptr) {
                shared_ptr<OpExpr<S>> opb =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(op->a) != 0 && rop.count(opb) != 0);
                tmp->allocate(rop.at(opb)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    shared_ptr<SparseMatrix<S>> rmat =
                        dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                            rop.at(
                                abs_value((shared_ptr<OpExpr<S>>)op->ops[i])))
                            ->load_archive();
                    opf->iadd(tmp, rmat, op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode == SeqTypes::Simple)
                        opf->seq->simple_perform();
                    rmat->deallocate();
                }
            } else {
                shared_ptr<OpExpr<S>> opa =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(opa) != 0 && rop.count(op->b) != 0);
                tmp->allocate(lop.at(opa)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    shared_ptr<SparseMatrix<S>> lmat =
                        dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                            lop.at(
                                abs_value((shared_ptr<OpExpr<S>>)op->ops[i])))
                            ->load_archive();
                    opf->iadd(tmp, lmat, op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode == SeqTypes::Simple)
                        opf->seq->simple_perform();
                    lmat->deallocate();
                }
            }
            if (op->b == nullptr) {
                shared_ptr<SparseMatrix<S>> lmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(lop.at(op->a))
                        ->load_archive();
                opf->tensor_product(op->conj, lmat, tmp, omat, op->factor);
                lmat->deallocate();
            } else {
                shared_ptr<SparseMatrix<S>> rmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(rop.at(op->b))
                        ->load_archive();
                opf->tensor_product(op->conj, tmp, rmat, omat, op->factor);
                rmat->deallocate();
            }
            tmp->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product(x, lop, rop, omat);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
        if (omat != mat) {
            aromat->save_archive(omat);
            omat->deallocate();
        }
    }
    // c = mpst_bra x a x mpst_ket
    void left_rotate(const shared_ptr<OperatorTensor<S>> &a,
                     const shared_ptr<SparseMatrix<S>> &mpst_bra,
                     const shared_ptr<SparseMatrix<S>> &mpst_ket,
                     shared_ptr<OperatorTensor<S>> &c) const override {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
            shared_ptr<ArchivedSparseMatrix<S>> arc =
                make_shared<ArchivedSparseMatrix<S>>(filename, offset);
            arc->save_archive(c->ops.at(op));
            c->ops.at(op)->deallocate();
            c->ops.at(op) = arc;
            offset += arc->total_memory;
        }
        for (size_t i = 0; i < a->lmat->data.size(); i++)
            if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->lmat->data[i]);
                shared_ptr<SparseMatrix<S>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(a->ops.at(pa))
                        ->load_archive();
                shared_ptr<ArchivedSparseMatrix<S>> armatc =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                        c->ops.at(pa));
                shared_ptr<SparseMatrix<S>> matc = armatc->load_archive();
                opf->tensor_rotate(mata, matc, mpst_bra, mpst_ket, false);
                armatc->save_archive(matc);
                matc->deallocate();
                mata->deallocate();
            }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    // c = mpst_bra x a x mpst_ket
    void right_rotate(const shared_ptr<OperatorTensor<S>> &a,
                      const shared_ptr<SparseMatrix<S>> &mpst_bra,
                      const shared_ptr<SparseMatrix<S>> &mpst_ket,
                      shared_ptr<OperatorTensor<S>> &c) const override {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
            shared_ptr<ArchivedSparseMatrix<S>> arc =
                make_shared<ArchivedSparseMatrix<S>>(filename, offset);
            arc->save_archive(c->ops.at(op));
            c->ops.at(op)->deallocate();
            c->ops.at(op) = arc;
            offset += arc->total_memory;
        }
        for (size_t i = 0; i < a->rmat->data.size(); i++)
            if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->rmat->data[i]);
                shared_ptr<SparseMatrix<S>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(a->ops.at(pa))
                        ->load_archive();
                shared_ptr<ArchivedSparseMatrix<S>> armatc =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                        c->ops.at(pa));
                shared_ptr<SparseMatrix<S>> matc = armatc->load_archive();
                opf->tensor_rotate(mata, matc, mpst_bra, mpst_ket, true);
                armatc->save_archive(matc);
                matc->deallocate();
                mata->deallocate();
            }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    void intermediates(const shared_ptr<Symbolic<S>> &names,
                       const shared_ptr<Symbolic<S>> &exprs,
                       const shared_ptr<OperatorTensor<S>> &a,
                       bool left) const override {
        for (size_t i = 0; i < exprs->data.size(); i++)
            if (exprs->data[i]->get_type() == OpTypes::Sum) {
                shared_ptr<OpSum<S>> expr =
                    dynamic_pointer_cast<OpSum<S>>(exprs->data[i]);
                for (size_t j = 0; j < expr->strings.size(); j++)
                    if (expr->strings[j]->get_type() == OpTypes::SumProd) {
                        shared_ptr<OpSumProd<S>> ex =
                            dynamic_pointer_cast<OpSumProd<S>>(
                                expr->strings[j]);
                        if ((left && ex->b == nullptr) ||
                            (!left && ex->a == nullptr) || ex->c == nullptr)
                            continue;
                        if (a->ops.count(ex->c) != 0)
                            continue;
                        shared_ptr<SparseMatrix<S>> tmp =
                            make_shared<SparseMatrix<S>>();
                        shared_ptr<OpExpr<S>> opb =
                            abs_value((shared_ptr<OpExpr<S>>)ex->ops[0]);
                        assert(a->ops.count(opb) != 0);
                        tmp->allocate(a->ops.at(opb)->info);
                        for (size_t k = 0; k < ex->ops.size(); k++) {
                            shared_ptr<SparseMatrix<S>> xmat =
                                dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                                    a->ops.at(abs_value(
                                        (shared_ptr<OpExpr<S>>)ex->ops[k])))
                                    ->load_archive();
                            opf->iadd(tmp, xmat, ex->ops[k]->factor,
                                      ex->conjs[k]);
                            if (opf->seq->mode == SeqTypes::Simple)
                                opf->seq->simple_perform();
                            xmat->deallocate();
                        }
                        shared_ptr<ArchivedSparseMatrix<S>> arc =
                            make_shared<ArchivedSparseMatrix<S>>(filename,
                                                                 offset);
                        arc->save_archive(tmp);
                        tmp->deallocate();
                        a->ops[ex->c] = arc;
                        offset += arc->total_memory;
                    }
            }
    }
    // Numerical transform from normal operators
    // to complementary operators near the middle site
    void
    numerical_transform(const shared_ptr<OperatorTensor<S>> &a,
                        const shared_ptr<Symbolic<S>> &names,
                        const shared_ptr<Symbolic<S>> &exprs) const override {
        assert(names->data.size() == exprs->data.size());
        assert((a->lmat == nullptr) ^ (a->rmat == nullptr));
        if (a->lmat == nullptr)
            a->rmat = names;
        else
            a->lmat = names;
        for (size_t i = 0; i < a->ops.size(); i++) {
            bool found = false;
            for (size_t k = 0; k < names->data.size(); k++) {
                if (exprs->data[k]->get_type() == OpTypes::Zero)
                    continue;
                shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
                shared_ptr<OpExpr<S>> expr =
                    exprs->data[k] *
                    (1 / dynamic_pointer_cast<OpElement<S>>(names->data[k])
                             ->factor);
                assert(a->ops.count(nop) != 0);
                switch (expr->get_type()) {
                case OpTypes::Sum: {
                    shared_ptr<OpSum<S>> op =
                        dynamic_pointer_cast<OpSum<S>>(expr);
                    found |= i < op->strings.size();
                    if (i < op->strings.size()) {
                        shared_ptr<OpElement<S>> nexpr =
                            op->strings[i]->get_op();
                        assert(a->ops.count(nexpr) != 0);
                        shared_ptr<SparseMatrix<S>> imat =
                            dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                                a->ops.at(nexpr))
                                ->load_archive();
                        shared_ptr<ArchivedSparseMatrix<S>> aromat =
                            dynamic_pointer_cast<ArchivedSparseMatrix<S>>(
                                a->ops.at(nop));
                        shared_ptr<SparseMatrix<S>> omat =
                            aromat->load_archive();
                        opf->iadd(omat, imat, op->strings[i]->factor,
                                  op->strings[i]->conj != 0);
                        if (opf->seq->mode == SeqTypes::Simple)
                            opf->seq->simple_perform();
                        aromat->save_archive(omat);
                        omat->deallocate();
                        imat->deallocate();
                    }
                } break;
                case OpTypes::Zero:
                    break;
                default:
                    assert(false);
                    break;
                }
            }
            if (!found)
                break;
        }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    // c = a x b (dot)
    void left_contract(const shared_ptr<OperatorTensor<S>> &a,
                       const shared_ptr<OperatorTensor<S>> &b,
                       shared_ptr<OperatorTensor<S>> &c,
                       const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                       OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr)
            left_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == c->lmat->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpElement<S>> cop =
                    dynamic_pointer_cast<OpElement<S>>(c->lmat->data[i]);
                shared_ptr<OpExpr<S>> op = abs_value(c->lmat->data[i]);
                shared_ptr<OpExpr<S>> expr = exprs->data[i] * (1 / cop->factor);
                if (!delayed(cop->name)) {
                    c->ops.at(op)->allocate(c->ops.at(op)->info);
                    tensor_product(expr, a->ops, b->ops, c->ops.at(op));
                    shared_ptr<ArchivedSparseMatrix<S>> arc =
                        make_shared<ArchivedSparseMatrix<S>>(filename, offset);
                    arc->save_archive(c->ops.at(op));
                    c->ops.at(op)->deallocate();
                    c->ops.at(op) = arc;
                    offset += arc->total_memory;
                }
            }
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
    // c = b (dot) x a
    void right_contract(const shared_ptr<OperatorTensor<S>> &a,
                        const shared_ptr<OperatorTensor<S>> &b,
                        shared_ptr<OperatorTensor<S>> &c,
                        const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                        OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr)
            right_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == c->rmat->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpElement<S>> cop =
                    dynamic_pointer_cast<OpElement<S>>(c->rmat->data[i]);
                shared_ptr<OpExpr<S>> op = abs_value(c->rmat->data[i]);
                shared_ptr<OpExpr<S>> expr = exprs->data[i] * (1 / cop->factor);
                if (!delayed(cop->name)) {
                    c->ops.at(op)->allocate(c->ops.at(op)->info);
                    tensor_product(expr, b->ops, a->ops, c->ops.at(op));
                    shared_ptr<ArchivedSparseMatrix<S>> arc =
                        make_shared<ArchivedSparseMatrix<S>>(filename, offset);
                    arc->save_archive(c->ops.at(op));
                    c->ops.at(op)->deallocate();
                    c->ops.at(op) = arc;
                    offset += arc->total_memory;
                }
            }
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
};

} // namespace block2
