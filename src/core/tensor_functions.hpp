
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

#include "operator_functions.hpp"
#include "operator_tensor.hpp"
#include "sparse_matrix.hpp"
#include "symbolic.hpp"
#include <cassert>
#include <map>
#include <memory>
#include <unordered_map>

using namespace std;

namespace block2 {

enum struct TensorFunctionsTypes : uint8_t {
    Normal = 0,
    Archived = 1,
    Delayed = 2,
    Parallel = 3
};

// Operations for operator tensors
template <typename S> struct TensorFunctions {
    shared_ptr<OperatorFunctions<S>> opf;
    TensorFunctions(const shared_ptr<OperatorFunctions<S>> &opf) : opf(opf) {}
    virtual ~TensorFunctions() = default;
    virtual TensorFunctionsTypes get_type() const {
        return TensorFunctionsTypes::Normal;
    }
    virtual shared_ptr<TensorFunctions<S>> copy() const {
        return make_shared<TensorFunctions<S>>(opf->copy());
    }
    virtual void operator()(const MatrixRef &b, const MatrixRef &c,
                            double scale = 1.0) {
        opf->seq->operator()(b, c, scale);
    }
    template <typename T> void serial_for(size_t n, T op) const {
        shared_ptr<TensorFunctions<S>> tf =
            make_shared<TensorFunctions<S>>(*this);
        for (size_t i = 0; i < n; i++)
            op(tf, i);
    }
    template <typename T> void parallel_for(size_t n, T op) const {
        shared_ptr<TensorFunctions<S>> tf =
            make_shared<TensorFunctions<S>>(*this);
        int ntop = threading->activate_operator();
        if (ntop == 1) {
            for (size_t i = 0; i < n; i++)
                op(tf, i);
        } else {
            vector<shared_ptr<TensorFunctions<S>>> tfs(1, tf);
            vector<pair<size_t, size_t>> tf_sz(ntop + 1);
            for (int i = 1; i < ntop; i++) {
                tfs.push_back(this->copy());
                tfs[i]->opf->seq->cumulative_nflop = 0;
            }
#pragma omp parallel for schedule(dynamic) num_threads(ntop)
            for (int i = 0; i < (int)n; i++) {
                int tid = threading->get_thread_id();
                op(tfs[tid], (size_t)i);
            }
            tf_sz[1].first = opf->seq->batch[0]->gp.size();
            tf_sz[1].second = opf->seq->batch[1]->gp.size();
            for (int i = 1; i < ntop; i++) {
                tf_sz[i + 1].first =
                    tf_sz[i].first + tfs[i]->opf->seq->batch[0]->gp.size();
                tf_sz[i + 1].second =
                    tf_sz[i].second + tfs[i]->opf->seq->batch[1]->gp.size();
                opf->seq->batch[0]->nflop += tfs[i]->opf->seq->batch[0]->nflop;
                opf->seq->batch[1]->nflop += tfs[i]->opf->seq->batch[1]->nflop;
                opf->seq->cumulative_nflop +=
                    tfs[i]->opf->seq->cumulative_nflop;
                opf->seq->max_work =
                    max(opf->seq->max_work, tfs[i]->opf->seq->max_work);
            }
            if (tf_sz[ntop].second != 0) {
                if (tf_sz[ntop].first != 0)
                    opf->seq->batch[0]->resize(tf_sz[ntop].first);
                opf->seq->batch[1]->resize(tf_sz[ntop].second);
#pragma omp parallel num_threads(ntop)
                {
                    int tid = threading->get_thread_id();
                    if (tid != 0) {
                        if (tf_sz[ntop].first != 0)
                            opf->seq->batch[0]->copy_from(
                                tf_sz[tid].first, tfs[tid]->opf->seq->batch[0]);
                        opf->seq->batch[1]->copy_from(
                            tf_sz[tid].second, tfs[tid]->opf->seq->batch[1]);
                    }
                }
            }
        }
        threading->activate_normal();
    }
    template <typename T, typename SM>
    void serial_reduce(size_t n, const shared_ptr<SM> &mat, T op) const {
        shared_ptr<TensorFunctions<S>> tf =
            make_shared<TensorFunctions<S>>(*this);
        for (size_t i = 0; i < n; i++)
            op(tf, mat, i);
    }
    template <typename T, typename SM>
    void parallel_reduce(size_t n, const shared_ptr<SM> &mat, T op) const {
        if (opf->seq->mode == SeqTypes::Auto ||
            (opf->seq->mode & SeqTypes::Tasked)) {
            auto xop = [&mat, &op](const shared_ptr<TensorFunctions<S>> &tf,
                                   size_t i) { op(tf, mat, i); };
            return parallel_for(n, xop);
        }
        shared_ptr<TensorFunctions<S>> tf =
            make_shared<TensorFunctions<S>>(*this);
        int ntop = threading->activate_operator();
        if (ntop == 1) {
            for (size_t i = 0; i < n; i++)
                op(tf, mat, i);
        } else {
            vector<shared_ptr<SM>> mats(1, mat);
            vector<shared_ptr<TensorFunctions<S>>> tfs(1, tf);
            mats.resize(ntop, nullptr);
            for (int i = 1; i < ntop; i++) {
                tfs.push_back(this->copy());
                tfs[i]->opf->seq->cumulative_nflop = 0;
            }
#pragma omp parallel num_threads(ntop)
            {
                int tid = threading->get_thread_id();
                if (tid != 0) {
                    shared_ptr<VectorAllocator<double>> d_alloc =
                        make_shared<VectorAllocator<double>>();
                    mats[tid] = make_shared<SM>(d_alloc);
                    mats[tid]->allocate_like(mat);
                }
#pragma omp for schedule(dynamic)
                for (int i = 0; i < (int)n; i++)
                    op(tfs[tid], mats[tid], (size_t)i);
#pragma omp single
                tfs[tid]->opf->parallel_reduce(mats, 0, ntop);
                if (tid != 0) {
                    mats[tid]->deallocate();
                    mats[tid] = nullptr;
                }
            }
            for (int i = 1; i < ntop; i++)
                opf->seq->cumulative_nflop +=
                    tfs[i]->opf->seq->cumulative_nflop;
        }
        threading->activate_normal();
    }
    // c = a
    virtual void left_assign(const shared_ptr<OperatorTensor<S>> &a,
                             shared_ptr<OperatorTensor<S>> &c) const {
        assert(a->lmat != nullptr);
        assert(a->lmat->get_type() == SymTypes::RVec);
        assert(c->lmat != nullptr);
        assert(c->lmat->get_type() == SymTypes::RVec);
        assert(a->lmat->data.size() == c->lmat->data.size());
        parallel_for(
            a->lmat->data.size(),
            [&a, &c](const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                if (a->lmat->data[i]->get_type() == OpTypes::Zero)
                    c->lmat->data[i] = a->lmat->data[i];
                else {
                    assert(a->lmat->data[i] == c->lmat->data[i]);
                    shared_ptr<OpExpr<S>> pa = abs_value(a->lmat->data[i]),
                                          pc = abs_value(c->lmat->data[i]);
                    if (!frame->use_main_stack) {
                        // skip cached part
                        if (c->ops[pc]->alloc != nullptr)
                            return;
                        c->ops[pc]->alloc =
                            make_shared<VectorAllocator<double>>();
                        c->ops[pc]->allocate(c->ops[pc]->info);
                    }
                    if (c->ops[pc]->info->n == a->ops[pa]->info->n)
                        c->ops[pc]->copy_data_from(a->ops[pa], true);
                    else
                        c->ops[pc]->selective_copy_from(a->ops[pa], true);
                    c->ops[pc]->factor = a->ops[pa]->factor;
                }
            });
    }
    // c = a
    virtual void right_assign(const shared_ptr<OperatorTensor<S>> &a,
                              shared_ptr<OperatorTensor<S>> &c) const {
        assert(a->rmat != nullptr);
        assert(a->rmat->get_type() == SymTypes::CVec);
        assert(c->rmat != nullptr);
        assert(c->rmat->get_type() == SymTypes::CVec);
        assert(a->rmat->data.size() == c->rmat->data.size());
        parallel_for(
            a->rmat->data.size(),
            [&a, &c](const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                if (a->rmat->data[i]->get_type() == OpTypes::Zero)
                    c->rmat->data[i] = a->rmat->data[i];
                else {
                    assert(a->rmat->data[i] == c->rmat->data[i]);
                    shared_ptr<OpExpr<S>> pa = abs_value(a->rmat->data[i]),
                                          pc = abs_value(c->rmat->data[i]);
                    if (!frame->use_main_stack) {
                        // skip cached part
                        if (c->ops[pc]->alloc != nullptr)
                            return;
                        c->ops[pc]->alloc =
                            make_shared<VectorAllocator<double>>();
                        c->ops[pc]->allocate(c->ops[pc]->info);
                    }
                    if (c->ops[pc]->info->n == a->ops[pa]->info->n)
                        c->ops[pc]->copy_data_from(a->ops[pa], true);
                    else
                        c->ops[pc]->selective_copy_from(a->ops[pa], true);
                    c->ops[pc]->factor = a->ops[pa]->factor;
                }
            });
    }
    // vmat = expr[L part | R part] x cmat (for perturbative noise)
    virtual void tensor_product_partial_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S>> &lopt,
        const shared_ptr<OperatorTensor<S>> &ropt, bool trace_right,
        const shared_ptr<SparseMatrix<S>> &cmat,
        const vector<pair<uint8_t, S>> &psubsl,
        const vector<
            vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
            &cinfos,
        const vector<S> &vdqs, const shared_ptr<SparseMatrixGroup<S>> &vmats,
        int &vidx, int tvidx, bool do_reduce) const {
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), S());
        // if no identity operator found in one side,
        // then the site does not have to be optimized.
        // perturbative noise can be skipped
        if ((!trace_right && lopt->ops.count(i_op) == 0) ||
            (trace_right && ropt->ops.count(i_op) == 0))
            return;
        switch (expr->get_type()) {
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            assert(op->a != nullptr && op->b != nullptr && op->ops.size() == 2);
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> old_cinfo =
                cmat->info->cinfo;
            bool dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
            assert((dleft ? lopt : ropt)->get_type() ==
                   OperatorTensorTypes::Delayed);
            shared_ptr<DelayedOperatorTensor<S>> dopt =
                dynamic_pointer_cast<DelayedOperatorTensor<S>>(dleft ? lopt
                                                                     : ropt);
            shared_ptr<SparseMatrix<S>> dlmat, drmat;
            uint8_t dconj = 0;
            if (dleft == trace_right) {
                assert(dopt->lopt->ops.count(op->ops[0]) != 0);
                assert(dopt->ropt->ops.count(op->ops[1]) != 0);
                dlmat = dopt->lopt->ops.at(op->ops[0]);
                drmat = dopt->ropt->ops.at(op->ops[1]);
                dconj = (uint8_t)op->conjs[0] | (op->conjs[1] << 1);
            } else {
                // for mixed 2-index/3-index, i_op can be delayed or undelayed
                if ((trace_right && ropt->ops.at(i_op)->data != nullptr) ||
                    (!trace_right && lopt->ops.at(i_op)->data != nullptr))
                    dlmat = drmat = nullptr;
                else {
                    assert(dopt->lopt->ops.count(i_op) != 0);
                    assert(dopt->ropt->ops.count(i_op) != 0);
                    dlmat = dopt->lopt->ops.at(i_op);
                    drmat = dopt->ropt->ops.at(i_op);
                }
            }
            if (trace_right) {
                assert(lopt->ops.count(op->a) != 0 &&
                       ropt->ops.count(i_op) != 0);
                shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
                shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(i_op);
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
                    if (tvidx >= 0 && tvidx != iv)
                        continue;
                    shared_ptr<SparseMatrix<S>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    if (dlmat != nullptr)
                        opf->three_tensor_product_multiply(
                            op->conj & 1, lmat, rmat, cmat, vmat, dconj, dlmat,
                            drmat, dleft, opdq, op->factor, TraceTypes::Right);
                    else
                        opf->tensor_product_multiply(
                            op->conj & 1, lmat, rmat, cmat, vmat, opdq,
                            op->factor, TraceTypes::Right);
                }
            } else {
                assert(lopt->ops.count(i_op) != 0 &&
                       ropt->ops.count(op->b) != 0);
                shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(i_op);
                shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
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
                    if (tvidx >= 0 && tvidx != iv)
                        continue;
                    shared_ptr<SparseMatrix<S>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    if (dlmat != nullptr)
                        opf->three_tensor_product_multiply(
                            op->conj & 2, lmat, rmat, cmat, vmat, dconj, dlmat,
                            drmat, dleft, opdq, op->factor, TraceTypes::Left);
                    else
                        opf->tensor_product_multiply(
                            op->conj & 2, lmat, rmat, cmat, vmat, opdq,
                            op->factor, TraceTypes::Left);
                }
            }
            cmat->info->cinfo = old_cinfo;
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->a != nullptr && op->b != nullptr);
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> old_cinfo =
                cmat->info->cinfo;
            shared_ptr<SparseMatrix<S>> dlmat, drmat;
            uint8_t dconj = 0;
            bool dleft = false;
            if (lopt->get_type() == OperatorTensorTypes::Delayed ||
                ropt->get_type() == OperatorTensorTypes::Delayed) {
                dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
                shared_ptr<DelayedOperatorTensor<S>> dopt =
                    dynamic_pointer_cast<DelayedOperatorTensor<S>>(
                        dleft ? lopt : ropt);
                if (dleft != trace_right) {
                    if ((trace_right && ropt->ops.at(i_op)->data != nullptr) ||
                        (!trace_right && lopt->ops.at(i_op)->data != nullptr))
                        dlmat = drmat = nullptr;
                    else {
                        assert(dopt->lopt->ops.count(i_op) != 0);
                        assert(dopt->ropt->ops.count(i_op) != 0);
                        dlmat = dopt->lopt->ops.at(i_op);
                        drmat = dopt->ropt->ops.at(i_op);
                    }
                }
            }
            if (trace_right) {
                assert(lopt->ops.count(op->a) != 0 &&
                       ropt->ops.count(i_op) != 0);
                shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
                shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(i_op);
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
                    if (tvidx >= 0 && tvidx != iv)
                        continue;
                    shared_ptr<SparseMatrix<S>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    if (dlmat != nullptr)
                        opf->three_tensor_product_multiply(
                            op->conj & 1, lmat, rmat, cmat, vmat, dconj, dlmat,
                            drmat, dleft, opdq, op->factor, TraceTypes::Right);
                    else
                        opf->tensor_product_multiply(
                            op->conj & 1, lmat, rmat, cmat, vmat, opdq,
                            op->factor, TraceTypes::Right);
                }
            } else {
                assert(lopt->ops.count(i_op) != 0 &&
                       ropt->ops.count(op->b) != 0);
                shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(i_op);
                shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
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
                    if (tvidx >= 0 && tvidx != iv)
                        continue;
                    shared_ptr<SparseMatrix<S>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    if (dlmat != nullptr)
                        opf->three_tensor_product_multiply(
                            op->conj & 2, lmat, rmat, cmat, vmat, dconj, dlmat,
                            drmat, dleft, opdq, op->factor, TraceTypes::Left);
                    else
                        opf->tensor_product_multiply(
                            op->conj & 2, lmat, rmat, cmat, vmat, opdq,
                            op->factor, TraceTypes::Left);
                }
            }
            cmat->info->cinfo = old_cinfo;
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            // non-reduced noise
            if (vidx != -1)
                for (auto &x : op->strings)
                    tensor_product_partial_multiply(x, lopt, ropt, trace_right,
                                                    cmat, psubsl, cinfos, vdqs,
                                                    vmats, vidx, tvidx, false);
            // copy vmats to every thread, high memory mode
            else if (tvidx == -1)
                parallel_reduce(
                    op->strings.size(), vmats,
                    [&op, &lopt, &ropt, trace_right, &cmat, &psubsl, &cinfos,
                     &vdqs](const shared_ptr<TensorFunctions<S>> &tf,
                            const shared_ptr<SparseMatrixGroup<S>> &vmats,
                            size_t i) {
                        shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                            make_shared<SparseMatrixInfo<S>>(*cmat->info);
                        shared_ptr<SparseMatrix<S>> pcmat =
                            make_shared<SparseMatrix<S>>(*cmat);
                        pcmat->info = pcmat_info;
                        int vidx = -1;
                        tf->tensor_product_partial_multiply(
                            op->strings[i], lopt, ropt, trace_right, pcmat,
                            psubsl, cinfos, vdqs, vmats, vidx, -1, false);
                    });
            // every thread works on a single vmat, low memory mode
            else if (tvidx == -2)
                parallel_for(
                    vmats->n,
                    [&op, &lopt, &ropt, trace_right, &cmat, &vmats, &psubsl,
                     &cinfos, &vdqs](const shared_ptr<TensorFunctions<S>> &tf,
                                     size_t i) {
                        shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                            make_shared<SparseMatrixInfo<S>>(*cmat->info);
                        shared_ptr<SparseMatrix<S>> pcmat =
                            make_shared<SparseMatrix<S>>(*cmat);
                        pcmat->info = pcmat_info;
                        int vidx = -1;
                        for (auto &x : op->strings)
                            tf->tensor_product_partial_multiply(
                                x, lopt, ropt, trace_right, pcmat, psubsl,
                                cinfos, vdqs, vmats, vidx, (int)i, false);
                    });
            else
                assert(false);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // vmats = expr x cmats
    virtual void tensor_product_multi_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S>> &lopt,
        const shared_ptr<OperatorTensor<S>> &ropt,
        const shared_ptr<SparseMatrixGroup<S>> &cmats,
        const shared_ptr<SparseMatrixGroup<S>> &vmats,
        const unordered_map<
            S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>
            &cinfos,
        S opdq, double factor, bool all_reduce) const {
        unordered_map<S, int> vdqs;
        vdqs.reserve(vmats->n);
        for (int iv = 0; iv < vmats->n; iv++)
            vdqs[vmats->infos[iv]->delta_quantum] = iv;
        switch (expr->get_type()) {
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            parallel_reduce(
                op->strings.size() * cmats->n, vmats,
                [&op, &lopt, &ropt, &cmats, &opdq, &vdqs, &cinfos, factor](
                    const shared_ptr<TensorFunctions<S>> &tf,
                    const shared_ptr<SparseMatrixGroup<S>> &vmats, size_t idx) {
                    const size_t i = idx % op->strings.size(),
                                 j = idx / op->strings.size();
                    shared_ptr<SparseMatrix<S>> pcmat = (*cmats)[(int)j];
                    pcmat->factor = factor;
                    shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                        make_shared<SparseMatrixInfo<S>>(*pcmat->info);
                    pcmat->info = pcmat_info;
                    S cdq = pcmat->info->delta_quantum;
                    S vdq = opdq + cdq;
                    for (int iv = 0; iv < vdq.count(); iv++)
                        if (vdqs.count(vdq[iv])) {
                            pcmat->info->cinfo =
                                cinfos.at(opdq.combine(vdq[iv], cdq));
                            tf->tensor_product_multiply(
                                op->strings[i], lopt, ropt, pcmat,
                                (*vmats)[vdqs[vdq[iv]]], opdq, false);
                        }
                });
        } break;
        case OpTypes::Zero:
            break;
        default:
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
                        pcmat->info->cinfo =
                            cinfos.at(opdq.combine(vdq[iv], cdq));
                        tensor_product_multiply(expr, lopt, ropt, pcmat,
                                                (*vmats)[vdqs[vdq[iv]]], opdq,
                                                false);
                    }
            }
            break;
        }
    }
    // fast expectation algorithm for NPDM, by reusing partially contracted
    // left part, assuming there are smaller number of unique left operators
    virtual vector<pair<shared_ptr<OpExpr<S>>, double>>
    tensor_product_expectation(const vector<shared_ptr<OpExpr<S>>> &names,
                               const vector<shared_ptr<OpExpr<S>>> &exprs,
                               const shared_ptr<OperatorTensor<S>> &lopt,
                               const shared_ptr<OperatorTensor<S>> &ropt,
                               const shared_ptr<SparseMatrix<S>> &cmat,
                               const shared_ptr<SparseMatrix<S>> &vmat) const {
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations(names.size());
        map<tuple<uint8_t, S, S>,
            unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>
            partials;
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>(d_alloc);
        assert(names.size() == exprs.size());
        S ket_dq = cmat->info->delta_quantum;
        S bra_dq = vmat->info->delta_quantum;
        for (size_t k = 0; k < exprs.size(); k++) {
            // may happen for NPDM with ancilla
            assert(dynamic_pointer_cast<OpElement<S>>(names[k])->name !=
                   OpNames::Zero);
            shared_ptr<OpExpr<S>> expr = exprs[k];
            S opdq = dynamic_pointer_cast<OpElement<S>>(names[k])->q_label;
            if (opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                continue;
            switch (expr->get_type()) {
            case OpTypes::SumProd:
                throw runtime_error("Tensor product expectation with delayed "
                                    "contraction not yet supported.");
                break;
            case OpTypes::Prod: {
                shared_ptr<OpProduct<S>> op =
                    dynamic_pointer_cast<OpProduct<S>>(expr);
                assert(op->a != nullptr && op->b != nullptr);
                assert(lopt->ops.count(op->a) != 0 &&
                       ropt->ops.count(op->b) != 0);
                if (lopt->get_type() == OperatorTensorTypes::Delayed ||
                    ropt->get_type() == OperatorTensorTypes::Delayed)
                    throw runtime_error(
                        "Tensor product expectation with delayed "
                        "contraction not yet supported.");
                shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
                shared_ptr<SparseMatrix<S>> rmat =
                    make_shared<SparseMatrix<S>>(d_alloc);
                rmat->info = ropt->ops.at(op->b)->info;
                if (!partials[make_tuple(op->conj, rmat->info->delta_quantum,
                                         opdq)]
                         .count(op->a)) {
                    rmat->allocate(rmat->info);
                    partials[make_tuple(op->conj, rmat->info->delta_quantum,
                                        opdq)][op->a] = rmat;
                }
            } break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S>> sop = dynamic_pointer_cast<OpSum<S>>(expr);
                for (size_t j = 0; j < sop->strings.size(); j++) {
                    shared_ptr<OpProduct<S>> op = sop->strings[j];
                    assert(op->a != nullptr && op->b != nullptr);
                    assert(lopt->ops.count(op->a) != 0 &&
                           ropt->ops.count(op->b) != 0);
                    if (lopt->get_type() == OperatorTensorTypes::Delayed ||
                        ropt->get_type() == OperatorTensorTypes::Delayed)
                        throw runtime_error(
                            "Tensor product expectation with delayed "
                            "contraction not yet supported.");
                    shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
                    shared_ptr<SparseMatrix<S>> rmat =
                        make_shared<SparseMatrix<S>>(d_alloc);
                    rmat->info = ropt->ops.at(op->b)->info;
                    if (!partials[make_tuple(op->conj,
                                             rmat->info->delta_quantum, opdq)]
                             .count(op->a)) {
                        partials[make_tuple(op->conj, rmat->info->delta_quantum,
                                            opdq)][op->a] = rmat;
                    }
                }
            } break;
            case OpTypes::Zero:
                break;
            default:
                assert(false);
                break;
            }
        }
        vector<tuple<uint8_t, S, shared_ptr<OpExpr<S>>,
                     shared_ptr<SparseMatrix<S>>>>
            vparts;
        for (auto &m : partials)
            for (auto &mm : m.second) {
                vparts.push_back(make_tuple(get<0>(m.first), get<2>(m.first),
                                            mm.first, mm.second));
                mm.second->allocate(mm.second->info);
            }
        parallel_for(vparts.size(),
                     [&vparts, &lopt, &cmat, &vmat](
                         const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                         uint8_t conj = get<0>(vparts[i]);
                         S opdq = get<1>(vparts[i]);
                         shared_ptr<SparseMatrix<S>> lmat =
                             lopt->ops.at(get<2>(vparts[i]));
                         shared_ptr<SparseMatrix<S>> rmat = get<3>(vparts[i]);
                         tf->opf->tensor_partial_expectation(conj, lmat, rmat,
                                                             cmat, vmat, opdq);
                     });
        vector<size_t> prod_idxs;
        prod_idxs.reserve(exprs.size());
        for (size_t k = 0; k < exprs.size(); k++) {
            shared_ptr<OpExpr<S>> expr = exprs[k];
            S opdq = dynamic_pointer_cast<OpElement<S>>(names[k])->q_label;
            expectations[k] = make_pair(names[k], 0.0);
            if (opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                continue;
            switch (expr->get_type()) {
            case OpTypes::Prod:
                prod_idxs.push_back(k);
                break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S>> sop = dynamic_pointer_cast<OpSum<S>>(expr);
                int ntop = threading->activate_operator();
                double r = 0;
#pragma omp parallel for schedule(dynamic) num_threads(ntop) reduction(+ : r)
                for (int j = 0; j < (int)sop->strings.size(); j++) {
                    shared_ptr<OpProduct<S>> op = sop->strings[j];
                    shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
                    shared_ptr<SparseMatrix<S>> lmat =
                        partials
                            .at(make_tuple(op->conj, rmat->info->delta_quantum,
                                           opdq))
                            .at(op->a);
                    r += opf->dot_product(lmat, rmat, op->factor);
                }
                threading->activate_normal();
                expectations[k] = make_pair(names[k], r);
            } break;
            case OpTypes::Zero:
                break;
            default:
                assert(false);
                break;
            }
        }
        parallel_for(
            prod_idxs.size(),
            [&prod_idxs, &ropt, &partials, &exprs, &names, &expectations](
                const shared_ptr<TensorFunctions<S>> &tf, size_t pk) {
                size_t k = prod_idxs[pk];
                shared_ptr<OpExpr<S>> expr = exprs[k];
                S opdq = dynamic_pointer_cast<OpElement<S>>(names[k])->q_label;
                shared_ptr<OpProduct<S>> op =
                    dynamic_pointer_cast<OpProduct<S>>(expr);
                shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
                shared_ptr<SparseMatrix<S>> lmat =
                    partials
                        .at(make_tuple(op->conj, rmat->info->delta_quantum,
                                       opdq))
                        .at(op->a);
                expectations[k] = make_pair(
                    names[k], tf->opf->dot_product(lmat, rmat, op->factor));
            });
        for (auto &vpart : vparts)
            get<3>(vpart)->deallocate();
        return expectations;
    }
    // vmat = expr x cmat
    virtual void
    tensor_product_multiply(const shared_ptr<OpExpr<S>> &expr,
                            const shared_ptr<OperatorTensor<S>> &lopt,
                            const shared_ptr<OperatorTensor<S>> &ropt,
                            const shared_ptr<SparseMatrix<S>> &cmat,
                            const shared_ptr<SparseMatrix<S>> &vmat, S opdq,
                            bool all_reduce) const {
        switch (expr->get_type()) {
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            assert(op->a != nullptr && op->b != nullptr && op->ops.size() == 2);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            bool dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
            assert((dleft ? lopt : ropt)->get_type() ==
                   OperatorTensorTypes::Delayed);
            shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
            shared_ptr<DelayedOperatorTensor<S>> dopt =
                dynamic_pointer_cast<DelayedOperatorTensor<S>>(dleft ? lopt
                                                                     : ropt);
            assert(dopt->lopt->ops.count(op->ops[0]) != 0);
            assert(dopt->ropt->ops.count(op->ops[1]) != 0);
            shared_ptr<SparseMatrix<S>> dlmat = dopt->lopt->ops.at(op->ops[0]);
            shared_ptr<SparseMatrix<S>> drmat = dopt->ropt->ops.at(op->ops[1]);
            uint8_t dconj = (uint8_t)op->conjs[0] | (op->conjs[1] << 1);
            opf->three_tensor_product_multiply(op->conj, lmat, rmat, cmat, vmat,
                                               dconj, dlmat, drmat, dleft, opdq,
                                               op->factor);
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->a != nullptr && op->b != nullptr);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
            opf->tensor_product_multiply(op->conj, lmat, rmat, cmat, vmat, opdq,
                                         op->factor);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            parallel_reduce(
                op->strings.size(), vmat,
                [&op, &lopt, &ropt, &cmat,
                 &opdq](const shared_ptr<TensorFunctions<S>> &tf,
                        const shared_ptr<SparseMatrix<S>> &vmat, size_t i) {
                    tf->tensor_product_multiply(op->strings[i], lopt, ropt,
                                                cmat, vmat, opdq, false);
                });
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // mat = diag(expr)
    virtual void
    tensor_product_diagonal(const shared_ptr<OpExpr<S>> &expr,
                            const shared_ptr<OperatorTensor<S>> &lopt,
                            const shared_ptr<OperatorTensor<S>> &ropt,
                            const shared_ptr<SparseMatrix<S>> &mat,
                            S opdq) const {
        switch (expr->get_type()) {
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            assert(op->a != nullptr && op->b != nullptr && op->ops.size() == 2);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            bool dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
            assert((dleft ? lopt : ropt)->get_type() ==
                   OperatorTensorTypes::Delayed);
            shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
            shared_ptr<DelayedOperatorTensor<S>> dopt =
                dynamic_pointer_cast<DelayedOperatorTensor<S>>(dleft ? lopt
                                                                     : ropt);
            assert(dopt->lopt->ops.count(op->ops[0]) != 0);
            assert(dopt->ropt->ops.count(op->ops[1]) != 0);
            shared_ptr<SparseMatrix<S>> dlmat = dopt->lopt->ops.at(op->ops[0]);
            shared_ptr<SparseMatrix<S>> drmat = dopt->ropt->ops.at(op->ops[1]);
            uint8_t dconj = (uint8_t)op->conjs[0] | (op->conjs[1] << 1);
            opf->three_tensor_product_diagonal(op->conj, lmat, rmat, mat, dconj,
                                               dlmat, drmat, dleft, opdq,
                                               op->factor);
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->a != nullptr && op->b != nullptr);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            shared_ptr<SparseMatrix<S>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = ropt->ops.at(op->b);
            opf->tensor_product_diagonal(op->conj, lmat, rmat, mat, opdq,
                                         op->factor);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            parallel_reduce(
                op->strings.size(), mat,
                [&op, &lopt, &ropt,
                 &opdq](const shared_ptr<TensorFunctions<S>> &tf,
                        const shared_ptr<SparseMatrix<S>> &mat, size_t i) {
                    tf->tensor_product_diagonal(op->strings[i], lopt, ropt, mat,
                                                opdq);
                });
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
            else if (opf->seq->mode & SeqTypes::Tasked)
                opf->seq->auto_perform(
                    MatrixRef(mat->data, (MKL_INT)mat->total_memory, 1));
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // mat = eval(expr)
    virtual void tensor_product(
        const shared_ptr<OpExpr<S>> &expr,
        const unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>
            &lop,
        const unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>
            &rop,
        shared_ptr<SparseMatrix<S>> &mat) const {
        switch (expr->get_type()) {
        case OpTypes::Elem: {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(expr);
            assert((rop.count(op) != 0) ^ (lop.count(op) != 0));
            shared_ptr<SparseMatrix<S>> lmat =
                lop.count(op) != 0 ? lop.at(op)
                                   : lop.at(make_shared<OpExpr<S>>());
            shared_ptr<SparseMatrix<S>> rmat =
                rop.count(op) != 0 ? rop.at(op)
                                   : rop.at(make_shared<OpExpr<S>>());
            opf->tensor_product(0, lmat, rmat, mat, op->factor);
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->b != nullptr);
            assert(lop.count(op->a) != 0 && rop.count(op->b) != 0);
            shared_ptr<SparseMatrix<S>> lmat = lop.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = rop.at(op->b);
            // here in parallel when not allocated mat->factor = 0, mat->data =
            // 0 but if mat->info->n = 0, mat->data also = 0
            opf->tensor_product(op->conj, lmat, rmat, mat, op->factor);
        } break;
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            assert((op->a == nullptr) ^ (op->b == nullptr));
            assert(op->ops.size() != 0);
            bool has_intermediate = false;
            shared_ptr<VectorAllocator<double>> d_alloc =
                make_shared<VectorAllocator<double>>();
            shared_ptr<SparseMatrix<S>> tmp =
                make_shared<SparseMatrix<S>>(d_alloc);
            if (op->c != nullptr && ((op->b == nullptr && rop.count(op->c)) ||
                                     (op->a == nullptr && lop.count(op->c)))) {
                has_intermediate = true;
                if (op->b == nullptr && rop.count(op->c))
                    tmp = rop.at(op->c);
                else
                    tmp = lop.at(op->c);
            } else if (op->b == nullptr) {
                shared_ptr<OpExpr<S>> opb =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(op->a) != 0 && rop.count(opb) != 0);
                tmp->allocate(rop.at(opb)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    opf->iadd(
                        tmp,
                        rop.at(abs_value((shared_ptr<OpExpr<S>>)op->ops[i])),
                        op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode & SeqTypes::Simple)
                        opf->seq->simple_perform();
                }
            } else {
                shared_ptr<OpExpr<S>> opa =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(opa) != 0 && rop.count(op->b) != 0);
                tmp->allocate(lop.at(opa)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    opf->iadd(
                        tmp,
                        lop.at(abs_value((shared_ptr<OpExpr<S>>)op->ops[i])),
                        op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode & SeqTypes::Simple)
                        opf->seq->simple_perform();
                }
            }
            if (op->b == nullptr)
                opf->tensor_product(op->conj, lop.at(op->a), tmp, mat,
                                    op->factor);
            else
                opf->tensor_product(op->conj, tmp, rop.at(op->b), mat,
                                    op->factor);
            if (!has_intermediate)
                tmp->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product(x, lop, rop, mat);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // c = mpst_bra x a x mpst_ket
    virtual void left_rotate(const shared_ptr<OperatorTensor<S>> &a,
                             const shared_ptr<SparseMatrix<S>> &mpst_bra,
                             const shared_ptr<SparseMatrix<S>> &mpst_ket,
                             shared_ptr<OperatorTensor<S>> &c) const {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
        }
        parallel_for(a->lmat->data.size(),
                     [&a, &c, &mpst_bra, &mpst_ket](
                         const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                         if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                             auto pa = abs_value(a->lmat->data[i]);
                             tf->opf->tensor_rotate(a->ops.at(pa),
                                                    c->ops.at(pa), mpst_bra,
                                                    mpst_ket, false);
                         }
                     });
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    // c = mpst_bra x a x mpst_ket
    virtual void right_rotate(const shared_ptr<OperatorTensor<S>> &a,
                              const shared_ptr<SparseMatrix<S>> &mpst_bra,
                              const shared_ptr<SparseMatrix<S>> &mpst_ket,
                              shared_ptr<OperatorTensor<S>> &c) const {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
        }
        parallel_for(a->rmat->data.size(),
                     [&a, &c, &mpst_bra, &mpst_ket](
                         const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                         if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                             auto pa = abs_value(a->rmat->data[i]);
                             tf->opf->tensor_rotate(a->ops.at(pa),
                                                    c->ops.at(pa), mpst_bra,
                                                    mpst_ket, true);
                         }
                     });
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    virtual void intermediates(const shared_ptr<Symbolic<S>> &names,
                               const shared_ptr<Symbolic<S>> &exprs,
                               const shared_ptr<OperatorTensor<S>> &a,
                               bool left) const {
        vector<vector<shared_ptr<OpSumProd<S>>>> exs;
        vector<int> maxk;
        exs.reserve(exprs->data.size());
        maxk.reserve(exprs->data.size());
        for (size_t i = 0; i < exprs->data.size(); i++)
            if (exprs->data[i] != nullptr &&
                exprs->data[i]->get_type() == OpTypes::Sum) {
                shared_ptr<OpSum<S>> expr =
                    dynamic_pointer_cast<OpSum<S>>(exprs->data[i]);
                exs.push_back(vector<shared_ptr<OpSumProd<S>>>());
                maxk.push_back(0);
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
                        a->ops[ex->c] = tmp;
                        exs.back().push_back(ex);
                        maxk.back() = max(maxk.back(), (int)ex->ops.size());
                    }
            }
        parallel_for(exs.size(), [&maxk, &exs,
                                  &a](const shared_ptr<TensorFunctions<S>> &tf,
                                      size_t i) {
            for (int k = 0; k < maxk[i]; k++) {
                for (auto &ex : exs[i]) {
                    if (k < ex->ops.size()) {
                        shared_ptr<SparseMatrix<S>> xmat = a->ops.at(
                            abs_value((shared_ptr<OpExpr<S>>)ex->ops[k]));
                        assert(xmat->get_type() != SparseMatrixTypes::Delayed);
                        tf->opf->iadd(a->ops.at(ex->c), xmat,
                                      ex->ops[k]->factor, ex->conjs[k]);
                    }
                }
                if (tf->opf->seq->mode & SeqTypes::Simple)
                    tf->opf->seq->simple_perform();
            }
        });
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    // Numerical transform from normal operators
    // to complementary operators near the middle site
    virtual void
    numerical_transform(const shared_ptr<OperatorTensor<S>> &a,
                        const shared_ptr<Symbolic<S>> &names,
                        const shared_ptr<Symbolic<S>> &exprs) const {
        assert(names->data.size() == exprs->data.size());
        assert((a->lmat == nullptr) ^ (a->rmat == nullptr));
        if (a->lmat == nullptr)
            a->rmat = names;
        else
            a->lmat = names;
        vector<pair<shared_ptr<SparseMatrix<S>>, shared_ptr<OpSum<S>>>> trs;
        trs.reserve(names->data.size());
        int maxi = 0;
        for (size_t k = 0; k < names->data.size(); k++) {
            if (exprs->data[k]->get_type() == OpTypes::Zero)
                continue;
            shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
            shared_ptr<OpExpr<S>> expr =
                exprs->data[k] *
                (1 /
                 dynamic_pointer_cast<OpElement<S>>(names->data[k])->factor);
            assert(a->ops.count(nop) != 0);
            shared_ptr<SparseMatrix<S>> anop = a->ops.at(nop);
            switch (expr->get_type()) {
            case OpTypes::Sum:
                trs.push_back(
                    make_pair(anop, dynamic_pointer_cast<OpSum<S>>(expr)));
                maxi = max(
                    maxi,
                    (int)dynamic_pointer_cast<OpSum<S>>(expr)->strings.size());
                break;
            case OpTypes::Zero:
                break;
            default:
                assert(false);
                break;
            }
        }
        parallel_for(
            trs.size(),
            [&trs, &a](const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                shared_ptr<OpSum<S>> op = trs[i].second;
                for (size_t j = 0; j < op->strings.size(); j++) {
                    shared_ptr<OpElement<S>> nexpr = op->strings[j]->get_op();
                    assert(a->ops.count(nexpr) != 0);
                    tf->opf->iadd(trs[i].first, a->ops.at(nexpr),
                                  op->strings[j]->factor,
                                  op->strings[j]->conj != 0);
                    if (tf->opf->seq->mode & SeqTypes::Simple)
                        tf->opf->seq->simple_perform();
                }
            });
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    // delete unnecessary operators after numerical_transform
    virtual void
    post_numerical_transform(const shared_ptr<OperatorTensor<S>> &a,
                             const shared_ptr<Symbolic<S>> &names,
                             const shared_ptr<Symbolic<S>> &new_names) const {
        set<shared_ptr<OpExpr<S>>, op_expr_less<S>> del_ops;
        for (size_t j = 0; j < names->data.size(); j++)
            del_ops.insert(names->data[j]);
        for (size_t j = 0; j < new_names->data.size(); j++)
            if (del_ops.count(new_names->data[j]))
                del_ops.erase(new_names->data[j]);
        vector<tuple<double *, shared_ptr<SparseMatrix<S>>, uint8_t>> mp;
        vector<tuple<double *, shared_ptr<SparseMatrix<S>>, uint8_t>> mp_ext;
        mp.reserve(a->ops.size());
        mp_ext.reserve(a->ops.size());
        for (auto it = a->ops.cbegin(); it != a->ops.cend(); it++)
            if (it->second->total_memory != 0) {
                if (it->second->alloc == dalloc)
                    mp.emplace_back(it->second->data, it->second,
                                    del_ops.count(it->first));
                else
                    mp_ext.emplace_back(it->second->data, it->second,
                                        del_ops.count(it->first));
            }
        sort(
            mp.begin(), mp.end(),
            [](const tuple<double *, shared_ptr<SparseMatrix<S>>, uint8_t> &a,
               const tuple<double *, shared_ptr<SparseMatrix<S>>, uint8_t> &b) {
                return get<0>(a) < get<0>(b);
            });
        sort(
            mp_ext.begin(), mp_ext.end(),
            [](const tuple<double *, shared_ptr<SparseMatrix<S>>, uint8_t> &a,
               const tuple<double *, shared_ptr<SparseMatrix<S>>, uint8_t> &b) {
                return get<0>(a) > get<0>(b);
            });
        for (const auto &t : mp)
            get<1>(t)->reallocate(get<2>(t) ? 0 : get<1>(t)->total_memory);
        for (const auto &t : mp_ext)
            if (get<2>(t))
                get<1>(t)->deallocate();
            else
                get<1>(t)->reallocate(dalloc);
    }
    // Substituing delayed left experssions
    // Return sum of three-operator tensor products
    virtual shared_ptr<Symbolic<S>>
    substitute_delayed_exprs(const shared_ptr<Symbolic<S>> &exprs,
                             const shared_ptr<DelayedOperatorTensor<S>> &a,
                             bool left, OpNamesSet delayed,
                             bool use_orig = true) const {
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<OpExpr<S>>> aops;
        shared_ptr<Symbolic<S>> amat = left ? a->lmat : a->rmat;
        assert(amat->data.size() == a->mat->data.size());
        aops.reserve(amat->data.size());
        for (size_t i = 0; i < amat->data.size(); i++) {
            shared_ptr<OpElement<S>> aop =
                dynamic_pointer_cast<OpElement<S>>(amat->data[i]);
            shared_ptr<OpExpr<S>> op = abs_value(amat->data[i]);
            shared_ptr<OpExpr<S>> expr = a->mat->data[i] * (1 / aop->factor);
            aops[op] = expr;
        }
        vector<shared_ptr<OpExpr<S>>> rexpr(exprs->data.size());
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static, 20) num_threads(ntg)
        for (int i = 0; i < (int)exprs->data.size(); i++) {
            shared_ptr<OpExpr<S>> expr = exprs->data[i];
            if (expr->get_type() == OpTypes::ExprRef)
                expr = use_orig ? dynamic_pointer_cast<OpExprRef<S>>(expr)->orig
                                : dynamic_pointer_cast<OpExprRef<S>>(expr)->op;
            vector<shared_ptr<OpProduct<S>>> prods;
            switch (expr->get_type()) {
            case OpTypes::Prod:
                prods.push_back(dynamic_pointer_cast<OpProduct<S>>(expr));
                break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
                prods.insert(prods.end(), op->strings.begin(),
                             op->strings.end());
            } break;
            case OpTypes::Zero:
                break;
            default:
                assert(false);
                break;
            }
            vector<shared_ptr<OpProduct<S>>> rr;
            rr.reserve(prods.size());
            for (auto &prod : prods) {
                shared_ptr<OpExpr<S>> hexpr;
                bool required =
                    left ? delayed(prod->a->name) : delayed(prod->b->name);
                if (!required) {
                    rr.push_back(prod);
                    continue;
                }
                if (left) {
                    assert(aops.count(prod->a) != 0);
                    hexpr = aops.at(prod->a);
                } else {
                    assert(aops.count(prod->b) != 0);
                    hexpr = aops.at(prod->b);
                }
                // sometimes, terms in TEMP expr can be evaluated in different
                // procs, which will change the conj of TEMP
                // we need to use the conj from the localized expr of TEMP
                unordered_map<shared_ptr<OpExpr<S>>, uint8_t> mpc;
                if (hexpr->get_type() == OpTypes::ExprRef && use_orig) {
                    shared_ptr<OpExpr<S>> op_expr =
                        dynamic_pointer_cast<OpExprRef<S>>(hexpr)->op;
                    vector<shared_ptr<OpSumProd<S>>> mrs;
                    switch (op_expr->get_type()) {
                    case OpTypes::SumProd:
                        mrs.push_back(
                            dynamic_pointer_cast<OpSumProd<S>>(op_expr));
                    case OpTypes::Sum: {
                        shared_ptr<OpSum<S>> sop =
                            dynamic_pointer_cast<OpSum<S>>(op_expr);
                        for (auto &op : sop->strings)
                            if (op->get_type() == OpTypes::SumProd)
                                mrs.push_back(
                                    dynamic_pointer_cast<OpSumProd<S>>(op));
                    } break;
                    case OpTypes::Zero:
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    mpc.reserve(mrs.size());
                    for (auto &op : mrs)
                        if (op->c != nullptr) {
                            uint8_t cj = op->b == nullptr ? (op->conj >> 1)
                                                          : (op->conj & 1);
                            if (mpc.count(op->c))
                                assert(mpc.at(op->c) == cj);
                            else
                                mpc[op->c] = cj;
                        }
                    hexpr = dynamic_pointer_cast<OpExprRef<S>>(hexpr)->orig;
                } else if (hexpr->get_type() == OpTypes::ExprRef && !use_orig)
                    hexpr = dynamic_pointer_cast<OpExprRef<S>>(hexpr)->op;
                vector<shared_ptr<OpProduct<S>>> rk;
                vector<shared_ptr<OpSumProd<S>>> rs;
                switch (hexpr->get_type()) {
                case OpTypes::SumProd:
                    rs.push_back(dynamic_pointer_cast<OpSumProd<S>>(hexpr));
                case OpTypes::Prod:
                    rk.push_back(dynamic_pointer_cast<OpProduct<S>>(hexpr));
                    break;
                case OpTypes::Sum: {
                    shared_ptr<OpSum<S>> sop =
                        dynamic_pointer_cast<OpSum<S>>(hexpr);
                    for (auto &op : sop->strings)
                        if (op->get_type() == OpTypes::Prod)
                            rk.push_back(op);
                        else if (op->get_type() == OpTypes::SumProd)
                            rs.push_back(
                                dynamic_pointer_cast<OpSumProd<S>>(op));
                } break;
                case OpTypes::Zero:
                    break;
                default:
                    assert(false);
                    break;
                }
                for (auto &op : rs) {
                    if (op->c != nullptr) {
                        uint8_t cj = op->conj;
                        if (mpc.size() != 0 && mpc.count(op->c))
                            cj = op->b == nullptr
                                     ? ((op->conj & 1) | (mpc.at(op->c) << 1))
                                     : ((op->conj & 2) | mpc.at(op->c));
                        if (op->b == nullptr) {
                            // TEMP may not have terms in all procs
                            if (a->ropt->ops.count(op->c))
                                rk.push_back(make_shared<OpProduct<S>>(
                                    op->a, op->c, op->factor, cj));
                        } else {
                            // TEMP may not have terms in all procs
                            if (a->lopt->ops.count(op->c))
                                rk.push_back(make_shared<OpProduct<S>>(
                                    op->c, op->b, op->factor, cj));
                        }
                    } else if (op->b == nullptr) {
                        for (size_t j = 0; j < op->ops.size(); j++)
                            rk.push_back(make_shared<OpProduct<S>>(
                                op->a, op->ops[j], op->factor,
                                op->conj ^ (op->conjs[j] << 1)));
                    } else {
                        for (size_t j = 0; j < op->ops.size(); j++)
                            rk.push_back(make_shared<OpProduct<S>>(
                                op->ops[j], op->b, op->factor,
                                op->conj ^ (uint8_t)op->conjs[j]));
                    }
                }
                for (auto &op : rk)
                    rr.push_back(make_shared<OpSumProd<S>>(
                        prod->a, prod->b,
                        vector<shared_ptr<OpElement<S>>>{op->a, op->b},
                        vector<bool>{(bool)(op->conj & 1),
                                     (bool)(op->conj & 2)},
                        prod->factor * op->factor, prod->conj));
            }
            rexpr[i] = make_shared<OpSum<S>>(rr);
            if (!use_orig && exprs->data[i]->get_type() == OpTypes::ExprRef) {
                shared_ptr<OpExprRef<S>> pexpr =
                    dynamic_pointer_cast<OpExprRef<S>>(exprs->data[i]);
                rexpr[i] = make_shared<OpExprRef<S>>(rexpr[i], pexpr->is_local,
                                                     pexpr->orig);
            }
        }
        threading->activate_normal();
        shared_ptr<Symbolic<S>> r = exprs->copy();
        r->data = rexpr;
        if (r->get_type() == SymTypes::RVec)
            r->n = (int)r->data.size();
        else
            r->m = (int)r->data.size();
        return r;
    }
    // delayed left and right block contraction (for effective hamil)
    virtual shared_ptr<DelayedOperatorTensor<S>>
    delayed_contract(const shared_ptr<OperatorTensor<S>> &a,
                     const shared_ptr<OperatorTensor<S>> &b,
                     const shared_ptr<OpExpr<S>> &op,
                     OpNamesSet delayed) const {
        shared_ptr<DelayedOperatorTensor<S>> dopt =
            make_shared<DelayedOperatorTensor<S>>();
        dopt->lopt = a, dopt->ropt = b;
        dopt->dops.push_back(op);
        assert(a->lmat->data.size() == b->rmat->data.size());
        shared_ptr<Symbolic<S>> exprs = a->lmat * b->rmat;
        assert(exprs->data.size() == 1);
        if (a->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S>>(a), true,
                delayed);
        else if (b->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S>>(b), false,
                delayed);
        else
            dopt->mat = exprs;
        return dopt;
    }
    // delayed left and right block contraction (for effective hamil)
    // using the pre-computed exprs
    virtual shared_ptr<DelayedOperatorTensor<S>>
    delayed_contract(const shared_ptr<OperatorTensor<S>> &a,
                     const shared_ptr<OperatorTensor<S>> &b,
                     const shared_ptr<Symbolic<S>> &ops,
                     const shared_ptr<Symbolic<S>> &exprs,
                     OpNamesSet delayed) const {
        shared_ptr<DelayedOperatorTensor<S>> dopt =
            make_shared<DelayedOperatorTensor<S>>();
        dopt->lopt = a, dopt->ropt = b;
        dopt->dops = ops->data;
        if (a->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S>>(a), true,
                delayed);
        else if (b->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S>>(b), false,
                delayed);
        else
            dopt->mat = exprs;
        return dopt;
    }
    // c = a x b (dot) (delayed for 3-operator operations)
    virtual void delayed_left_contract(
        const shared_ptr<OperatorTensor<S>> &a,
        const shared_ptr<OperatorTensor<S>> &b,
        shared_ptr<OperatorTensor<S>> &c,
        const shared_ptr<Symbolic<S>> &cexprs = nullptr) const {
        if (a == nullptr)
            return left_contract(a, b, c, cexprs);
        shared_ptr<DelayedOperatorTensor<S>> dopt =
            make_shared<DelayedOperatorTensor<S>>();
        dopt->mat = cexprs == nullptr ? a->lmat * b->lmat : cexprs;
        dopt->lopt = a, dopt->ropt = b;
        dopt->ops = c->ops;
        dopt->lmat = c->lmat, dopt->rmat = c->rmat;
        c = dopt;
    }
    // c = b (dot) x a (delayed for 3-operator operations)
    virtual void delayed_right_contract(
        const shared_ptr<OperatorTensor<S>> &a,
        const shared_ptr<OperatorTensor<S>> &b,
        shared_ptr<OperatorTensor<S>> &c,
        const shared_ptr<Symbolic<S>> &cexprs = nullptr) const {
        if (a == nullptr)
            return right_contract(a, b, c, cexprs);
        shared_ptr<DelayedOperatorTensor<S>> dopt =
            make_shared<DelayedOperatorTensor<S>>();
        dopt->mat = cexprs == nullptr ? b->rmat * a->rmat : cexprs;
        dopt->lopt = b, dopt->ropt = a;
        dopt->ops = c->ops;
        dopt->lmat = c->lmat, dopt->rmat = c->rmat;
        c = dopt;
    }
    // c = a x b (dot)
    virtual void left_contract(const shared_ptr<OperatorTensor<S>> &a,
                               const shared_ptr<OperatorTensor<S>> &b,
                               shared_ptr<OperatorTensor<S>> &c,
                               const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                               OpNamesSet delayed = OpNamesSet()) const {
        if (frame->use_main_stack)
            for (auto &p : c->ops) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(p.first);
                if (a == nullptr || !delayed(op->name))
                    c->ops.at(op)->allocate(c->ops.at(op)->info);
            }
        if (a == nullptr)
            left_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == c->lmat->data.size());
            parallel_for(
                exprs->data.size(),
                [&a, &b, &c, &exprs,
                 &delayed](const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                    shared_ptr<OpElement<S>> cop =
                        dynamic_pointer_cast<OpElement<S>>(c->lmat->data[i]);
                    shared_ptr<OpExpr<S>> op = abs_value(c->lmat->data[i]);
                    shared_ptr<OpExpr<S>> expr =
                        exprs->data[i] * (1 / cop->factor);
                    if (!delayed(cop->name)) {
                        if (!frame->use_main_stack) {
                            // skip cached part
                            if (c->ops.at(op)->alloc != nullptr)
                                return;
                            c->ops.at(op)->alloc =
                                make_shared<VectorAllocator<double>>();
                            c->ops.at(op)->allocate(c->ops.at(op)->info);
                        }
                        tf->tensor_product(expr, a->ops, b->ops, c->ops.at(op));
                    }
                });
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
    // c = b (dot) x a
    virtual void right_contract(const shared_ptr<OperatorTensor<S>> &a,
                                const shared_ptr<OperatorTensor<S>> &b,
                                shared_ptr<OperatorTensor<S>> &c,
                                const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                                OpNamesSet delayed = OpNamesSet()) const {
        if (frame->use_main_stack)
            for (auto &p : c->ops) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(p.first);
                if (a == nullptr || !delayed(op->name))
                    c->ops.at(op)->allocate(c->ops.at(op)->info);
            }
        if (a == nullptr)
            right_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == c->rmat->data.size());
            parallel_for(
                exprs->data.size(),
                [&a, &b, &c, &exprs,
                 &delayed](const shared_ptr<TensorFunctions<S>> &tf, size_t i) {
                    shared_ptr<OpElement<S>> cop =
                        dynamic_pointer_cast<OpElement<S>>(c->rmat->data[i]);
                    shared_ptr<OpExpr<S>> op = abs_value(c->rmat->data[i]);
                    shared_ptr<OpExpr<S>> expr =
                        exprs->data[i] * (1 / cop->factor);
                    if (!delayed(cop->name)) {
                        if (!frame->use_main_stack) {
                            // skip cached part
                            if (c->ops.at(op)->alloc != nullptr)
                                return;
                            c->ops.at(op)->alloc =
                                make_shared<VectorAllocator<double>>();
                            c->ops.at(op)->allocate(c->ops.at(op)->info);
                        }
                        tf->tensor_product(expr, b->ops, a->ops, c->ops.at(op));
                    }
                });
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
};

} // namespace block2
