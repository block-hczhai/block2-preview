
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
#include "spin_permutation.hpp"
#include "symbolic.hpp"
#include <array>
#include <cassert>
#include <map>
#include <memory>
#include <string>
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
template <typename S, typename FL> struct TensorFunctions {
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<OperatorFunctions<S, FL>> opf;
    TensorFunctions(const shared_ptr<OperatorFunctions<S, FL>> &opf)
        : opf(opf) {}
    virtual ~TensorFunctions() = default;
    virtual TensorFunctionsTypes get_type() const {
        return TensorFunctionsTypes::Normal;
    }
    virtual shared_ptr<TensorFunctions> copy() const {
        return make_shared<TensorFunctions>(opf->copy());
    }
    virtual void operator()(const GMatrix<FL> &b, const GMatrix<FL> &c,
                            FL scale = 1.0) {
        opf->seq->operator()(b, c, scale);
    }
    template <typename T> void serial_for(size_t n, T op) const {
        shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(*this);
        for (size_t i = 0; i < n; i++)
            op(tf, i);
    }
    template <typename T> void parallel_for(size_t n, T op) const {
        shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(*this);
        int ntop = threading->activate_operator();
        if (ntop == 1) {
            for (size_t i = 0; i < n; i++)
                op(tf, i);
        } else {
            vector<shared_ptr<TensorFunctions>> tfs(1, tf);
            vector<array<size_t, 4>> tf_sz(ntop + 1);
            for (int i = 1; i < ntop; i++) {
                tfs.push_back(this->copy());
                tfs[i]->opf->seq->cumulative_nflop = 0;
            }
#pragma omp parallel for schedule(dynamic) num_threads(ntop)
            for (int i = 0; i < (int)n; i++) {
                int tid = threading->get_thread_id();
                op(tfs[tid], (size_t)i);
            }
            tf_sz[1][0] = opf->seq->batch[0]->gp.size();
            tf_sz[1][1] = opf->seq->batch[0]->c.size();
            tf_sz[1][2] = opf->seq->batch[1]->gp.size();
            tf_sz[1][3] = opf->seq->batch[1]->c.size();
            bool has_acidxs = opf->seq->batch[0]->acidxs.size() != 0;
            for (int i = 1; i < ntop; i++) {
                tf_sz[i + 1][0] =
                    tf_sz[i][0] + tfs[i]->opf->seq->batch[0]->gp.size();
                tf_sz[i + 1][1] =
                    tf_sz[i][1] + tfs[i]->opf->seq->batch[0]->c.size();
                tf_sz[i + 1][2] =
                    tf_sz[i][2] + tfs[i]->opf->seq->batch[1]->gp.size();
                tf_sz[i + 1][3] =
                    tf_sz[i][3] + tfs[i]->opf->seq->batch[1]->c.size();
                has_acidxs = has_acidxs ||
                             tfs[i]->opf->seq->batch[0]->acidxs.size() != 0;
                opf->seq->batch[0]->nflop += tfs[i]->opf->seq->batch[0]->nflop;
                opf->seq->batch[1]->nflop += tfs[i]->opf->seq->batch[1]->nflop;
                opf->seq->cumulative_nflop +=
                    tfs[i]->opf->seq->cumulative_nflop;
                opf->seq->max_work =
                    max(opf->seq->max_work, tfs[i]->opf->seq->max_work);
            }
            if (tf_sz[ntop][2] != 0) {
                if (tf_sz[ntop][0] != 0)
                    opf->seq->batch[0]->resize(tf_sz[ntop][0], tf_sz[ntop][1]);
                opf->seq->batch[1]->resize(tf_sz[ntop][2], tf_sz[ntop][3]);
                if (has_acidxs)
                    opf->seq->batch[0]->acidxs.resize(tf_sz[ntop][0]);
#pragma omp parallel num_threads(ntop)
                {
                    int tid = threading->get_thread_id();
                    if (tid != 0) {
                        if (tf_sz[ntop][0] != 0)
                            opf->seq->batch[0]->copy_from(
                                tf_sz[tid][0], tf_sz[tid][1],
                                tfs[tid]->opf->seq->batch[0]);
                        opf->seq->batch[1]->copy_from(
                            tf_sz[tid][2], tf_sz[tid][3],
                            tfs[tid]->opf->seq->batch[1]);
                    }
                }
            }
        }
        threading->activate_normal();
    }
    template <typename T, typename SM>
    void serial_reduce(size_t n, const shared_ptr<SM> &mat, T op) const {
        shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(*this);
        for (size_t i = 0; i < n; i++)
            op(tf, mat, i);
    }
    template <typename T, typename SM>
    void parallel_reduce(size_t n, const shared_ptr<SM> &mat, T op) const {
        if (opf->seq->mode == SeqTypes::Auto ||
            (opf->seq->mode & SeqTypes::Tasked)) {
            auto xop = [&mat, &op](const shared_ptr<TensorFunctions> &tf,
                                   size_t i) { op(tf, mat, i); };
            return parallel_for(n, xop);
        }
        shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(*this);
        int ntop = threading->activate_operator();
        if (ntop == 1) {
            for (size_t i = 0; i < n; i++)
                op(tf, mat, i);
        } else {
            vector<shared_ptr<SM>> mats(1, mat);
            vector<shared_ptr<TensorFunctions>> tfs(1, tf);
            mats.resize(ntop, nullptr);
            for (int i = 1; i < ntop; i++) {
                tfs.push_back(this->copy());
                tfs[i]->opf->seq->cumulative_nflop = 0;
            }
#pragma omp parallel num_threads(ntop)
            {
                int tid = threading->get_thread_id();
                if (tid != 0) {
                    shared_ptr<VectorAllocator<FP>> d_alloc =
                        make_shared<VectorAllocator<FP>>();
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
    virtual void left_assign(const shared_ptr<OperatorTensor<S, FL>> &a,
                             shared_ptr<OperatorTensor<S, FL>> &c) const {
        assert(a->lmat != nullptr);
        assert(a->lmat->get_type() == SymTypes::RVec);
        assert(c->lmat != nullptr);
        assert(c->lmat->get_type() == SymTypes::RVec);
        assert(a->lmat->data.size() == c->lmat->data.size());
        parallel_for(
            a->lmat->data.size(),
            [&a, &c](const shared_ptr<TensorFunctions> &tf, size_t i) {
                if (a->lmat->data[i]->get_type() == OpTypes::Zero)
                    c->lmat->data[i] = a->lmat->data[i];
                else {
                    assert(a->lmat->data[i] == c->lmat->data[i]);
                    shared_ptr<OpExpr<S>> pa = abs_value(a->lmat->data[i]),
                                          pc = abs_value(c->lmat->data[i]);
                    if (!frame_<FP>()->use_main_stack) {
                        // skip cached part
                        if (c->ops[pc]->alloc != nullptr)
                            return;
                        c->ops[pc]->alloc = make_shared<VectorAllocator<FP>>();
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
    virtual void right_assign(const shared_ptr<OperatorTensor<S, FL>> &a,
                              shared_ptr<OperatorTensor<S, FL>> &c) const {
        assert(a->rmat != nullptr);
        assert(a->rmat->get_type() == SymTypes::CVec);
        assert(c->rmat != nullptr);
        assert(c->rmat->get_type() == SymTypes::CVec);
        assert(a->rmat->data.size() == c->rmat->data.size());
        parallel_for(
            a->rmat->data.size(),
            [&a, &c](const shared_ptr<TensorFunctions> &tf, size_t i) {
                if (a->rmat->data[i]->get_type() == OpTypes::Zero)
                    c->rmat->data[i] = a->rmat->data[i];
                else {
                    assert(a->rmat->data[i] == c->rmat->data[i]);
                    shared_ptr<OpExpr<S>> pa = abs_value(a->rmat->data[i]),
                                          pc = abs_value(c->rmat->data[i]);
                    if (!frame_<FP>()->use_main_stack) {
                        // skip cached part
                        if (c->ops[pc]->alloc != nullptr)
                            return;
                        c->ops[pc]->alloc = make_shared<VectorAllocator<FP>>();
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
        const shared_ptr<OperatorTensor<S, FL>> &lopt,
        const shared_ptr<OperatorTensor<S, FL>> &ropt, bool trace_right,
        const shared_ptr<SparseMatrix<S, FL>> &cmat,
        const vector<pair<uint8_t, S>> &psubsl,
        const vector<
            vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
            &cinfos,
        const vector<S> &vdqs,
        const shared_ptr<SparseMatrixGroup<S, FL>> &vmats, int &vidx, int tvidx,
        bool do_reduce) const {
        const shared_ptr<OpElement<S, FL>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), S());
        // if no identity operator found in one side,
        // then the site does not have to be optimized.
        // perturbative noise can be skipped
        if ((!trace_right && lopt->ops.count(i_op) == 0) ||
            (trace_right && ropt->ops.count(i_op) == 0))
            return;
        switch (expr->get_type()) {
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S, FL>> op =
                dynamic_pointer_cast<OpSumProd<S, FL>>(expr);
            assert(op->a != nullptr && op->b != nullptr && op->ops.size() == 2);
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> old_cinfo =
                cmat->info->cinfo;
            bool dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
            assert((dleft ? lopt : ropt)->get_type() ==
                   OperatorTensorTypes::Delayed);
            shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(
                    dleft ? lopt : ropt);
            shared_ptr<SparseMatrix<S, FL>> dlmat, drmat;
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
                shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
                shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(i_op);
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
                    shared_ptr<SparseMatrix<S, FL>> vmat =
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
                shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(i_op);
                shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
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
                    shared_ptr<SparseMatrix<S, FL>> vmat =
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
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->a != nullptr && op->b != nullptr);
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> old_cinfo =
                cmat->info->cinfo;
            shared_ptr<SparseMatrix<S, FL>> dlmat, drmat;
            uint8_t dconj = 0;
            bool dleft = false;
            if (lopt->get_type() == OperatorTensorTypes::Delayed ||
                ropt->get_type() == OperatorTensorTypes::Delayed) {
                dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
                shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                    dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(
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
                shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
                shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(i_op);
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
                    shared_ptr<SparseMatrix<S, FL>> vmat =
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
                shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(i_op);
                shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
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
                    shared_ptr<SparseMatrix<S, FL>> vmat =
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
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
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
                     &vdqs](const shared_ptr<TensorFunctions> &tf,
                            const shared_ptr<SparseMatrixGroup<S, FL>> &vmats,
                            size_t i) {
                        shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                            make_shared<SparseMatrixInfo<S>>(*cmat->info);
                        shared_ptr<SparseMatrix<S, FL>> pcmat =
                            make_shared<SparseMatrix<S, FL>>(*cmat);
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
                     &cinfos,
                     &vdqs](const shared_ptr<TensorFunctions> &tf, size_t i) {
                        shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                            make_shared<SparseMatrixInfo<S>>(*cmat->info);
                        shared_ptr<SparseMatrix<S, FL>> pcmat =
                            make_shared<SparseMatrix<S, FL>>(*cmat);
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
        const shared_ptr<OperatorTensor<S, FL>> &lopt,
        const shared_ptr<OperatorTensor<S, FL>> &ropt,
        const shared_ptr<SparseMatrixGroup<S, FL>> &cmats,
        const shared_ptr<SparseMatrixGroup<S, FL>> &vmats,
        const unordered_map<
            S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>
            &cinfos,
        S opdq, FL factor, bool all_reduce) const {
        unordered_map<S, int> vdqs;
        vdqs.reserve(vmats->n);
        for (int iv = 0; iv < vmats->n; iv++)
            vdqs[vmats->infos[iv]->delta_quantum] = iv;
        switch (expr->get_type()) {
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            parallel_reduce(
                op->strings.size() * cmats->n, vmats,
                [&op, &lopt, &ropt, &cmats, &opdq, &vdqs, &cinfos,
                 factor](const shared_ptr<TensorFunctions> &tf,
                         const shared_ptr<SparseMatrixGroup<S, FL>> &vmats,
                         size_t idx) {
                    const size_t i = idx % op->strings.size(),
                                 j = idx / op->strings.size();
                    shared_ptr<SparseMatrix<S, FL>> pcmat = (*cmats)[(int)j];
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
                shared_ptr<SparseMatrix<S, FL>> pcmat = (*cmats)[i];
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
    void npdm_middle_intermediates(
        const shared_ptr<NPDMScheme> &scheme,
        const shared_ptr<NPDMCounter> &counter, int n_sites, int center,
        uint64_t &mshape,
        vector<vector<vector<uint64_t>>> &mshape_presum) const {
        mshape = 0;
        mshape_presum.resize(1);
        mshape_presum[0] =
            vector<vector<uint64_t>>(scheme->middle_blocking.size());
        for (int i = 0; i < scheme->middle_blocking.size(); i++) {
            mshape_presum[0][i] =
                vector<uint64_t>(scheme->middle_blocking[i].size() + 1);
            mshape_presum[0][i][0] =
                i == 0 ? 0 : mshape_presum[0][i - 1].back();
            for (int j = 0; j < scheme->middle_blocking[i].size(); j++) {
                uint32_t lx = scheme->middle_blocking[i][j].first;
                uint32_t rx = scheme->middle_blocking[i][j].second;
                uint64_t lcnt = counter->count_left(
                    scheme->left_terms[lx].first.first, center, true);
                uint64_t rcnt = counter->count_right(
                    scheme->right_terms[rx].first.first, center + 1);
                mshape_presum[0][i][j + 1] =
                    mshape_presum[0][i][j] + lcnt * rcnt;
                mshape += lcnt * rcnt;
            }
        }
        if (center == n_sites - 2) {
            mshape_presum.push_back(
                vector<vector<uint64_t>>(scheme->last_middle_blocking.size()));
            for (int i = 0; i < scheme->last_middle_blocking.size(); i++) {
                mshape_presum[1][i] = vector<uint64_t>(
                    scheme->last_middle_blocking[i].size() + 1);
                mshape_presum[1][i][0] = i == 0
                                             ? mshape_presum[0].back().back()
                                             : mshape_presum[1][i - 1].back();
                if (scheme->last_middle_blocking[i].size() != 0)
                    for (int j = 0; j < scheme->last_middle_blocking[i].size();
                         j++) {
                        uint32_t lx = scheme->last_middle_blocking[i][j].first;
                        uint32_t rx = scheme->last_middle_blocking[i][j].second;
                        assert(scheme->left_terms[lx].second == false ||
                               scheme->left_terms[lx].first.first.size() == 0);
                        uint64_t lcnt = counter->count_left(
                            scheme->left_terms[lx].first.first, center, false);
                        uint64_t rcnt = counter->count_right(
                            rx < scheme->right_terms.size()
                                ? scheme->right_terms[rx].first.first
                                : scheme
                                      ->last_right_terms
                                          [rx - scheme->right_terms.size()]
                                      .first.first,
                            center + 1);
                        mshape_presum[1][i][j + 1] =
                            mshape_presum[1][i][j] + lcnt * rcnt;
                        mshape += lcnt * rcnt;
                    }
            }
        }
        assert(mshape_presum.back().back().back() == mshape);
    }
    virtual shared_ptr<GTensor<FL, uint64_t>>
    npdm_sort_load_file(const string &filename, bool compressed) const {
        shared_ptr<GTensor<FL, uint64_t>> p =
            make_shared<GTensor<FL, uint64_t>>();
        string fn = filename + (compressed ? ".fpc" : ".npy");
        ifstream ifs(fn.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("TensorFunctions::npdm_sort load on '" + fn +
                                "' failed.");
        if (compressed) {
            size_t arr_len;
            ifs >> arr_len;
            p->data =
                make_shared<vector<FL>>(arr_len / (sizeof(FL) / sizeof(FP)));
            p->shape =
                vector<uint64_t>{(uint64_t)arr_len / (sizeof(FL) / sizeof(FP))};
            make_shared<FPCodec<FP>>()->read_array(ifs, (FP *)p->data->data(),
                                                   arr_len);
        } else
            p->read_array(ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("TensorFunctions::npdm_sort load on '" + fn +
                                "' failed.");
        ifs.close();
        return p;
    }
    template <typename FLX>
    void npdm_sort(const shared_ptr<NPDMScheme> &scheme,
                   const vector<shared_ptr<GTensor<FLX>>> &npdm,
                   const string &filename, int n_sites, int center,
                   bool compressed) const {
        shared_ptr<NPDMCounter> counter =
            make_shared<NPDMCounter>(scheme->n_max_ops, n_sites);
        shared_ptr<GTensor<FL, uint64_t>> p =
            npdm_sort_load_file(filename, compressed);
        if (p == nullptr)
            return;
        uint64_t mshape = 0;
        vector<vector<vector<uint64_t>>> mshape_presum;
        npdm_middle_intermediates(scheme, counter, n_sites, center, mshape,
                                  mshape_presum);
        map<vector<uint16_t>, vector<pair<int, int>>> middle_patterns;
        for (int i = 0; i < (int)scheme->perms.size(); i++)
            for (int j = 0; j < (int)scheme->perms[i]->index_patterns.size();
                 j++)
                middle_patterns[scheme->perms[i]->index_patterns[j]].push_back(
                    make_pair(i, j));
        int middle_count = (int)scheme->middle_blocking.size();
        int middle_base_count = middle_count;
        if (center == n_sites - 2)
            middle_count += (int)scheme->last_middle_blocking.size();
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int ii = 0; ii < middle_count; ii++) {
            bool is_last = ii >= middle_base_count;
            int i = is_last ? ii - middle_base_count : ii;
            if (is_last && scheme->last_middle_blocking[i].size() == 0)
                continue;
            map<string, int> middle_cd_map;
            for (int j = 0; j < (int)scheme->middle_terms[i].size(); j++)
                middle_cd_map[scheme->middle_terms[i][j]] = j;
            for (auto &r : middle_patterns.at(scheme->middle_perm_patterns[i]))
                for (auto &pr : scheme->perms[r.first]->data[r.second]) {
                    const vector<uint16_t> &perm = pr.first;
                    for (auto &prr : pr.second) {
                        int jj = middle_cd_map[prr.second];
                        const uint32_t lx =
                            is_last ? scheme->last_middle_blocking[i][jj].first
                                    : scheme->middle_blocking[i][jj].first;
                        const uint32_t rx =
                            is_last ? scheme->last_middle_blocking[i][jj].second
                                    : scheme->middle_blocking[i][jj].second;
                        const vector<uint16_t> &rpat =
                            rx < scheme->right_terms.size()
                                ? scheme->right_terms[rx].first.first
                                : scheme
                                      ->last_right_terms
                                          [rx - scheme->right_terms.size()]
                                      .first.first;
                        const uint64_t lcnt = counter->count_left(
                            scheme->left_terms[lx].first.first, center,
                            !is_last);
                        const uint64_t rcnt =
                            counter->count_right(rpat, center + 1);
                        if (lcnt == 0 || rcnt == 0)
                            continue;
                        // left / right index linearlization multiplier
                        vector<uint64_t> lmx(
                            scheme->left_terms[lx].first.first.size());
                        vector<uint64_t> rmx(rpat.size());
                        uint64_t mxx = 1;
                        for (int k = (int)perm.size() - 1; k >= 0; k--) {
                            if (perm[k] < lmx.size())
                                lmx[perm[k]] = mxx;
                            else
                                rmx[perm[k] - lmx.size()] = mxx;
                            mxx *= n_sites;
                        }
                        // left / right indices
                        vector<uint16_t> lxx, rxx;
                        // left / right linearlized indices
                        vector<uint64_t> lixx(lcnt), rixx(rcnt);
                        counter->init_left(scheme->left_terms[lx].first.first,
                                           center, !is_last, lxx);
                        for (uint64_t il = 0; il < lcnt; il++) {
                            for (int k = 0; k < (int)lmx.size(); k++)
                                lixx[il] += lxx[k] * lmx[k];
                            counter->next_left(
                                scheme->left_terms[lx].first.first, center,
                                lxx);
                        }
                        counter->init_right(rpat, center + 1, rxx);
                        for (uint64_t ir = 0; ir < rcnt; ir++) {
                            for (int k = 0; k < (int)rmx.size(); k++)
                                rixx[ir] += rxx[k] * rmx[k];
                            counter->next_right(rpat, center + 1, rxx);
                        }
                        // sorting
                        const uint64_t ip = mshape_presum[is_last][i][jj];
                        for (uint64_t il = 0; il < lcnt; il++)
                            for (uint64_t ir = 0; ir < rcnt; ir++)
                                (*npdm[r.first]->data)[lixx[il] + rixx[ir]] +=
                                    (FLX)prr.first *
                                    (FLX)(*p->data)[ip + il * rcnt + ir];
                    }
                }
        }
        threading->activate_normal();
    }
    struct NPDMIndexer {
        vector<uint64_t> plidxs, pridxs;
        shared_ptr<NPDMScheme> scheme;
        shared_ptr<NPDMCounter> counter;
        int center;
        S vacuum;
        NPDMIndexer(const shared_ptr<NPDMScheme> &scheme,
                    const shared_ptr<NPDMCounter> &counter, int center,
                    S vacuum)
            : scheme(scheme), counter(counter), center(center), vacuum(vacuum) {
            plidxs.resize(scheme->left_terms.size() + 1, 0);
            pridxs.resize(scheme->right_terms.size() +
                              scheme->last_right_terms.size() + 1,
                          0);
            for (int k = 0; k < (int)scheme->left_terms.size(); k++)
                plidxs[k + 1] =
                    plidxs[k] +
                    counter->count_left(scheme->left_terms[k].first.first,
                                        center, scheme->left_terms[k].second);
            for (int k = 0; k < (int)scheme->right_terms.size(); k++)
                pridxs[k + 1] =
                    pridxs[k] +
                    counter->count_right(scheme->right_terms[k].first.first,
                                         center + 1);
            for (int k = 0; k < (int)scheme->last_right_terms.size(); k++)
                pridxs[k + 1 + scheme->right_terms.size()] =
                    pridxs[k + scheme->right_terms.size()] +
                    counter->count_right(
                        scheme->last_right_terms[k].first.first, center + 1);
        }
        inline pair<uint64_t, uint64_t> get_left_count(uint32_t lx,
                                                       bool is_last) const {
            uint64_t lcnt = (uint64_t)counter->count_left(
                scheme->left_terms[lx].first.first, center, !is_last);
            uint64_t lshift = !scheme->left_terms[lx].second && !is_last
                                  ? (uint64_t)counter->count_left(
                                        scheme->left_terms[lx].first.first,
                                        center, scheme->left_terms[lx].second) -
                                        lcnt
                                  : 0;
            return make_pair(lcnt, plidxs[lx] + lshift);
        }
        inline pair<uint64_t, uint64_t> get_right_count(uint32_t rx) const {
            const vector<uint16_t> &rpat =
                rx < scheme->right_terms.size()
                    ? scheme->right_terms[rx].first.first
                    : scheme->last_right_terms[rx - scheme->right_terms.size()]
                          .first.first;
            uint64_t rcnt = counter->count_right(rpat, center + 1);
            return make_pair(rcnt, pridxs[rx]);
        }
        inline void set_parallel_right_skip(uint32_t rx, uint64_t rcnt,
                                            int msize, int mrank,
                                            vector<uint8_t> &r) const {
            uint64_t imj = 0;
            const vector<uint16_t> &rpat = scheme->right_terms[rx].first.first;
            for (int mj = center + 1; mj < counter->n_sites; mj++) {
                uint64_t mjcnt = counter->count_right(rpat, mj) -
                                 (mj == counter->n_sites - 1
                                      ? 0
                                      : counter->count_right(rpat, mj + 1));
                if (mj % msize != mrank)
                    memset(&r[imj], 1, sizeof(uint8_t) * mjcnt);
                imj += mjcnt;
            }
            assert(imj == rcnt);
        }
        inline shared_ptr<SparseMatrix<S, FL>>
        get_mat(uint64_t ixx, const shared_ptr<OperatorTensor<S, FL>> &xopt,
                OpNames op_name) const {
            SiteIndex sx(
                {(uint16_t)(ixx >> 36), (uint16_t)((ixx >> 24) & 0xFFFLL),
                 (uint16_t)((ixx >> 12) & 0xFFFLL), (uint16_t)(ixx & 0xFFFLL)},
                {});
            shared_ptr<OpElement<S, FL>> op =
                make_shared<OpElement<S, FL>>(op_name, sx, vacuum);
            return xopt->ops.count(op) ? xopt->ops.at(op) : nullptr;
        }
    };
    // fast symbol-free expectation algorithm for NPDM
    // cache_left should be used if wfn is smaller in the left side.
    // so it is irrelavent for the zero-dot algorithm.
    // but for one-dot it should be aligned with fuse_left
    // better cache_left can save time for the M^3 part
    virtual vector<pair<shared_ptr<OpExpr<S>>, FL>>
    tensor_product_npdm_fragment(const shared_ptr<NPDMScheme> &scheme, S vacuum,
                                 const string &filename, int n_sites,
                                 int center, int parallel_center,
                                 const shared_ptr<OperatorTensor<S, FL>> &lopt,
                                 const shared_ptr<OperatorTensor<S, FL>> &ropt,
                                 const shared_ptr<SparseMatrix<S, FL>> &cmat,
                                 const shared_ptr<SparseMatrix<S, FL>> &vmat,
                                 bool cache_left, bool compressed,
                                 bool low_mem) const {
        vector<pair<shared_ptr<OpExpr<S>>, FL>> expectations(1);
        if (center == n_sites - 1) {
            expectations[0] = make_pair(make_shared<OpCounter<S>>(0), (FL)0.0);
            return expectations;
        }
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        S ket_dq = cmat->info->delta_quantum;
        S bra_dq = vmat->info->delta_quantum;
        shared_ptr<NPDMCounter> counter =
            make_shared<NPDMCounter>(scheme->n_max_ops, n_sites);
        uint64_t mshape = 0;
        vector<vector<vector<uint64_t>>> mshape_presum;
        npdm_middle_intermediates(scheme, counter, n_sites, center, mshape,
                                  mshape_presum);
        shared_ptr<NPDMIndexer> indexer =
            make_shared<NPDMIndexer>(scheme, counter, center, vacuum);
        expectations[0] = make_pair(make_shared<OpCounter<S>>(mshape), (FL)0.0);
        shared_ptr<GTensor<FL, uint64_t>> result =
            make_shared<GTensor<FL, uint64_t>>(vector<uint64_t>{mshape});
        int middle_count = (int)scheme->middle_blocking.size();
        int middle_base_count = middle_count;
        if (center == n_sites - 2)
            middle_count += (int)scheme->last_middle_blocking.size();
        map<pair<uint32_t, bool>, vector<pair<int, int>>> cache_to_middle;
        for (int ii = 0; ii < middle_count; ii++) {
            bool is_last = ii >= middle_base_count;
            int i = is_last ? ii - middle_base_count : ii;
            if (is_last && scheme->last_middle_blocking[i].size() == 0)
                continue;
            for (int j = 0; j < (int)scheme->middle_blocking[i].size(); j++) {
                uint32_t cx =
                    cache_left
                        ? (is_last ? scheme->last_middle_blocking[i][j].first
                                   : scheme->middle_blocking[i][j].first)
                        : (is_last ? scheme->last_middle_blocking[i][j].second
                                   : scheme->middle_blocking[i][j].second);
                cache_to_middle[make_pair(cx, is_last)].push_back(
                    make_pair(i, j));
            }
        }
        for (auto &ml : cache_to_middle) {
            uint32_t cx = ml.first.first;
            bool is_last = ml.first.second;
            uint64_t ccnt, cshift;
            tie(ccnt, cshift) = cache_left
                                    ? indexer->get_left_count(cx, is_last)
                                    : indexer->get_right_count(cx);
            if (ccnt == 0)
                continue;
            if (low_mem) {
                // low mem: do not parallelize / store M^3 contraction
                // intermediates
                if (cache_left) {
                    // do M^3 contraction for each left operator
                    uint64_t lcnt = ccnt, lshift = cshift;
                    map<S, shared_ptr<SparseMatrix<S, FL>>> left_partials;
                    for (uint64_t il = 0; il < lcnt; il++) {
                        shared_ptr<SparseMatrix<S, FL>> lmat =
                            indexer->get_mat(il + lshift, lopt, OpNames::XL);
                        if (lmat != nullptr) {
                            left_partials.clear();
                            for (auto &mr : ml.second) {
                                int i = mr.first, j = mr.second;
                                uint32_t rx =
                                    is_last
                                        ? scheme->last_middle_blocking[i][j]
                                              .second
                                        : scheme->middle_blocking[i][j].second;
                                uint64_t rcnt, rshift;
                                tie(rcnt, rshift) =
                                    indexer->get_right_count(rx);
                                for (uint64_t ir = 0; ir < rcnt; ir++) {
                                    shared_ptr<SparseMatrix<S, FL>> rmat =
                                        indexer->get_mat(ir + rshift, ropt,
                                                         OpNames::XR);
                                    if (rmat == nullptr)
                                        continue;
                                    // FIXME: not working for non-singlet
                                    // operators
                                    S opdq = (lmat->info->delta_quantum +
                                              rmat->info->delta_quantum)[0];
                                    if (opdq.combine(bra_dq, ket_dq) ==
                                        S(S::invalid))
                                        continue;
                                    if (!left_partials.count(
                                            rmat->info->delta_quantum)) {
                                        shared_ptr<SparseMatrix<S, FL>> xmat =
                                            make_shared<SparseMatrix<S, FL>>(
                                                d_alloc);
                                        xmat->info = rmat->info;
                                        xmat->allocate(xmat->info);
                                        left_partials[rmat->info
                                                          ->delta_quantum] =
                                            xmat;
                                        opf->tensor_left_partial_expectation(
                                            0, lmat, xmat, cmat, vmat, opdq);
                                    }
                                }
                                uint64_t iresult =
                                    mshape_presum[is_last][i][j] + rcnt * il;
                                parallel_for(
                                    (size_t)rcnt,
                                    [&left_partials, &ropt, &result, &indexer,
                                     rshift, iresult](
                                        const shared_ptr<TensorFunctions> &tf,
                                        size_t pk) {
                                        uint64_t ir = (uint64_t)pk;
                                        shared_ptr<SparseMatrix<S, FL>> rmat =
                                            indexer->get_mat(ir + rshift, ropt,
                                                             OpNames::XR);
                                        if (rmat != nullptr &&
                                            left_partials.count(
                                                rmat->info->delta_quantum)) {
                                            shared_ptr<SparseMatrix<S, FL>>
                                                lmat = left_partials.at(
                                                    rmat->info->delta_quantum);
                                            (*result->data)[iresult + ir] =
                                                tf->opf->dot_product(lmat,
                                                                     rmat);
                                        }
                                    });
                            }
                        }
                    }
                } else {
                    // do M^3 contraction for each right operator
                    uint64_t rcnt = ccnt, rshift = cshift;
                    map<S, shared_ptr<SparseMatrix<S, FL>>> right_partials;
                    for (uint64_t ir = 0; ir < rcnt; ir++) {
                        shared_ptr<SparseMatrix<S, FL>> rmat =
                            indexer->get_mat(ir + rshift, ropt, OpNames::XR);
                        if (rmat != nullptr) {
                            right_partials.clear();
                            for (auto &mr : ml.second) {
                                int i = mr.first, j = mr.second;
                                uint32_t lx =
                                    is_last
                                        ? scheme->last_middle_blocking[i][j]
                                              .first
                                        : scheme->middle_blocking[i][j].first;
                                uint64_t lcnt, lshift;
                                tie(lcnt, lshift) =
                                    indexer->get_left_count(lx, is_last);
                                for (uint64_t il = 0; il < lcnt; il++) {
                                    shared_ptr<SparseMatrix<S, FL>> lmat =
                                        indexer->get_mat(il + lshift, lopt,
                                                         OpNames::XL);
                                    if (lmat == nullptr)
                                        continue;
                                    // FIXME: not working for non-singlet
                                    // operators
                                    S opdq = (lmat->info->delta_quantum +
                                              rmat->info->delta_quantum)[0];
                                    if (opdq.combine(bra_dq, ket_dq) ==
                                        S(S::invalid))
                                        continue;
                                    if (!right_partials.count(
                                            lmat->info->delta_quantum)) {
                                        shared_ptr<SparseMatrix<S, FL>> xmat =
                                            make_shared<SparseMatrix<S, FL>>(
                                                d_alloc);
                                        xmat->info = lmat->info;
                                        xmat->allocate(xmat->info);
                                        right_partials[lmat->info
                                                           ->delta_quantum] =
                                            xmat;
                                        opf->tensor_right_partial_expectation(
                                            0, xmat, rmat, cmat, vmat, opdq);
                                    }
                                }
                                uint64_t iresult =
                                    mshape_presum[is_last][i][j] + ir;
                                parallel_for(
                                    (size_t)lcnt,
                                    [&right_partials, &lopt, &result, &indexer,
                                     lshift, iresult, rcnt](
                                        const shared_ptr<TensorFunctions> &tf,
                                        size_t pk) {
                                        uint64_t il = (uint64_t)pk;
                                        shared_ptr<SparseMatrix<S, FL>> lmat =
                                            indexer->get_mat(il + lshift, lopt,
                                                             OpNames::XL);
                                        if (lmat != nullptr &&
                                            right_partials.count(
                                                lmat->info->delta_quantum)) {
                                            shared_ptr<SparseMatrix<S, FL>>
                                                rmat = right_partials.at(
                                                    lmat->info->delta_quantum);
                                            (*result
                                                  ->data)[iresult + rcnt * il] =
                                                tf->opf->dot_product(lmat,
                                                                     rmat);
                                        }
                                    });
                            }
                        }
                    }
                }
            } else {
                // parallelize and store M^3 contraction intermediates
                vector<pair<shared_ptr<SparseMatrix<S, FL>>,
                            map<S, shared_ptr<SparseMatrix<S, FL>>>>>
                    c_partials;
                vector<pair<size_t, S>> c_compute;
                c_partials.reserve(ccnt);
                if (cache_left) {
                    uint64_t lcnt = ccnt, lshift = cshift;
                    vector<shared_ptr<SparseMatrix<S, FL>>> r_partials;
                    vector<uint64_t> left_idxs;
                    vector<pair<int, uint64_t>> right_idxs;
                    vector<uint64_t> right_cnts;
                    left_idxs.reserve(lcnt);
                    right_cnts.reserve(ml.second.size());
                    int im = 0;
                    for (auto &mr : ml.second) {
                        int i = mr.first, j = mr.second;
                        uint32_t rx =
                            is_last ? scheme->last_middle_blocking[i][j].second
                                    : scheme->middle_blocking[i][j].second;
                        uint64_t rcnt, rshift;
                        tie(rcnt, rshift) = indexer->get_right_count(rx);
                        r_partials.reserve(rcnt);
                        right_idxs.reserve(rcnt);
                        right_cnts.push_back(rcnt);
                        for (uint64_t ir = 0; ir < rcnt; ir++) {
                            shared_ptr<SparseMatrix<S, FL>> rmat =
                                indexer->get_mat(ir + rshift, ropt,
                                                 OpNames::XR);
                            if (rmat != nullptr) {
                                right_idxs.push_back(make_pair(im, ir));
                                r_partials.push_back(rmat);
                            }
                        }
                        im++;
                    }
                    if (right_idxs.size() == 0)
                        continue;
                    for (uint64_t il = 0; il < lcnt; il++) {
                        shared_ptr<SparseMatrix<S, FL>> lmat =
                            indexer->get_mat(il + lshift, lopt, OpNames::XL);
                        if (lmat != nullptr) {
                            left_idxs.push_back(il);
                            c_partials.push_back(make_pair(
                                lmat,
                                map<S, shared_ptr<SparseMatrix<S, FL>>>()));
                            for (size_t ixr = 0; ixr < right_idxs.size();
                                 ixr++) {
                                auto &mr = ml.second[right_idxs[ixr].first];
                                uint64_t ir = right_idxs[ixr].second;
                                shared_ptr<SparseMatrix<S, FL>> rmat =
                                    r_partials[ixr];
                                // FIXME: not working for non-singlet operators
                                S opdq = (lmat->info->delta_quantum +
                                          rmat->info->delta_quantum)[0];
                                if (opdq.combine(bra_dq, ket_dq) ==
                                    S(S::invalid))
                                    continue;
                                if (!c_partials.back().second.count(
                                        rmat->info->delta_quantum)) {
                                    shared_ptr<SparseMatrix<S, FL>> xmat =
                                        make_shared<SparseMatrix<S, FL>>(
                                            d_alloc);
                                    xmat->info = rmat->info;
                                    xmat->allocate(xmat->info);
                                    c_partials.back()
                                        .second[rmat->info->delta_quantum] =
                                        xmat;
                                    c_compute.push_back(
                                        make_pair(c_partials.size() - 1,
                                                  rmat->info->delta_quantum));
                                }
                            }
                        }
                    }
                    parallel_for(
                        c_compute.size(),
                        [&c_compute, &c_partials, &cmat, &vmat](
                            const shared_ptr<TensorFunctions> &tf, size_t i) {
                            shared_ptr<SparseMatrix<S, FL>> lmat =
                                c_partials[c_compute[i].first].first;
                            shared_ptr<SparseMatrix<S, FL>> rmat =
                                c_partials[c_compute[i].first].second.at(
                                    c_compute[i].second);
                            // FIXME: not working for non-singlet operators
                            S opdq = (lmat->info->delta_quantum +
                                      rmat->info->delta_quantum)[0];
                            tf->opf->tensor_left_partial_expectation(
                                0, lmat, rmat, cmat, vmat, opdq);
                        });
                    parallel_for(
                        left_idxs.size() * right_idxs.size(),
                        [&c_partials, &r_partials, &left_idxs, &right_idxs,
                         &right_cnts, &cmat, &vmat, &result, &mshape_presum,
                         &ml, is_last](const shared_ptr<TensorFunctions> &tf,
                                       size_t k) {
                            size_t ixl = k / right_idxs.size(),
                                   ixr = k % right_idxs.size();
                            uint64_t il = left_idxs[ixl],
                                     ir = right_idxs[ixr].second;
                            auto &mr = ml.second[right_idxs[ixr].first];
                            int i = mr.first, j = mr.second;
                            uint64_t rcnt = right_cnts[right_idxs[ixr].first];
                            uint64_t iresult =
                                mshape_presum[is_last][i][j] + rcnt * il + ir;
                            shared_ptr<SparseMatrix<S, FL>> rmat =
                                r_partials[ixr];
                            if (c_partials[ixl].second.count(
                                    rmat->info->delta_quantum)) {
                                shared_ptr<SparseMatrix<S, FL>> lmat =
                                    c_partials[ixl].second.at(
                                        rmat->info->delta_quantum);
                                (*result->data)[iresult] =
                                    tf->opf->dot_product(lmat, rmat);
                            }
                        });
                } else {
                    uint64_t rcnt = ccnt, rshift = cshift;
                    vector<shared_ptr<SparseMatrix<S, FL>>> l_partials;
                    vector<uint64_t> right_idxs;
                    vector<pair<int, uint64_t>> left_idxs;
                    right_idxs.reserve(rcnt);
                    int im = 0;
                    for (auto &mr : ml.second) {
                        int i = mr.first, j = mr.second;
                        uint32_t lx =
                            is_last ? scheme->last_middle_blocking[i][j].first
                                    : scheme->middle_blocking[i][j].first;
                        uint64_t lcnt, lshift;
                        tie(lcnt, lshift) =
                            indexer->get_left_count(lx, is_last);
                        l_partials.reserve(lcnt);
                        left_idxs.reserve(lcnt);
                        for (uint64_t il = 0; il < lcnt; il++) {
                            shared_ptr<SparseMatrix<S, FL>> lmat =
                                indexer->get_mat(il + lshift, lopt,
                                                 OpNames::XL);
                            if (lmat != nullptr) {
                                left_idxs.push_back(make_pair(im, il));
                                l_partials.push_back(lmat);
                            }
                        }
                        im++;
                    }
                    if (left_idxs.size() == 0)
                        continue;
                    for (uint64_t ir = 0; ir < rcnt; ir++) {
                        shared_ptr<SparseMatrix<S, FL>> rmat =
                            indexer->get_mat(ir + rshift, ropt, OpNames::XR);
                        if (rmat != nullptr) {
                            right_idxs.push_back(ir);
                            c_partials.push_back(make_pair(
                                rmat,
                                map<S, shared_ptr<SparseMatrix<S, FL>>>()));
                            for (size_t ixl = 0; ixl < left_idxs.size();
                                 ixl++) {
                                auto &mr = ml.second[left_idxs[ixl].first];
                                uint64_t il = left_idxs[ixl].second;
                                shared_ptr<SparseMatrix<S, FL>> lmat =
                                    l_partials[ixl];
                                // FIXME: not working for non-singlet operators
                                S opdq = (lmat->info->delta_quantum +
                                          rmat->info->delta_quantum)[0];
                                if (opdq.combine(bra_dq, ket_dq) ==
                                    S(S::invalid))
                                    continue;
                                if (!c_partials.back().second.count(
                                        lmat->info->delta_quantum)) {
                                    shared_ptr<SparseMatrix<S, FL>> xmat =
                                        make_shared<SparseMatrix<S, FL>>(
                                            d_alloc);
                                    xmat->info = lmat->info;
                                    xmat->allocate(xmat->info);
                                    c_partials.back()
                                        .second[lmat->info->delta_quantum] =
                                        xmat;
                                    c_compute.push_back(
                                        make_pair(c_partials.size() - 1,
                                                  lmat->info->delta_quantum));
                                }
                            }
                        }
                    }
                    parallel_for(
                        c_compute.size(),
                        [&c_compute, &c_partials, &cmat, &vmat](
                            const shared_ptr<TensorFunctions> &tf, size_t i) {
                            shared_ptr<SparseMatrix<S, FL>> rmat =
                                c_partials[c_compute[i].first].first;
                            shared_ptr<SparseMatrix<S, FL>> lmat =
                                c_partials[c_compute[i].first].second.at(
                                    c_compute[i].second);
                            // FIXME: not working for non-singlet operators
                            S opdq = (lmat->info->delta_quantum +
                                      rmat->info->delta_quantum)[0];
                            tf->opf->tensor_right_partial_expectation(
                                0, lmat, rmat, cmat, vmat, opdq);
                        });
                    parallel_for(
                        left_idxs.size() * right_idxs.size(),
                        [&c_partials, &l_partials, &left_idxs, &right_idxs,
                         &cmat, &vmat, &result, &mshape_presum, &ml, is_last,
                         rcnt](const shared_ptr<TensorFunctions> &tf,
                               size_t k) {
                            size_t ixl = k / right_idxs.size(),
                                   ixr = k % right_idxs.size();
                            uint64_t il = left_idxs[ixl].second,
                                     ir = right_idxs[ixr];
                            auto &mr = ml.second[left_idxs[ixl].first];
                            int i = mr.first, j = mr.second;
                            uint64_t iresult =
                                mshape_presum[is_last][i][j] + rcnt * il + ir;
                            shared_ptr<SparseMatrix<S, FL>> lmat =
                                l_partials[ixl];
                            if (c_partials[ixr].second.count(
                                    lmat->info->delta_quantum)) {
                                shared_ptr<SparseMatrix<S, FL>> rmat =
                                    c_partials[ixr].second.at(
                                        lmat->info->delta_quantum);
                                (*result->data)[iresult] =
                                    tf->opf->dot_product(lmat, rmat);
                            }
                        });
                }
            }
        };
        string fn = filename + (compressed ? ".fpc" : ".npy");
        ofstream ofs(fn.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error(
                "TensorFunctions::tensor_product_npdm_fragment save on '" + fn +
                "' failed.");
        if (compressed) {
            ofs << result->data->size() * (sizeof(FL) / sizeof(FP));
            make_shared<FPCodec<FP>>()->write_array(
                ofs, (FP *)result->data->data(),
                result->data->size() * (sizeof(FL) / sizeof(FP)));
        } else
            result->write_array(ofs);
        if (!ofs.good())
            throw runtime_error(
                "TensorFunctions::tensor_product_npdm_fragment save on '" + fn +
                "' failed.");
        ofs.close();
        return expectations;
    }
    // fast expectation algorithm for NPDM, by reusing partially contracted
    // left part, assuming there are smaller number of unique left operators
    virtual vector<pair<shared_ptr<OpExpr<S>>, FL>>
    tensor_product_expectation(const vector<shared_ptr<OpExpr<S>>> &names,
                               const vector<shared_ptr<OpExpr<S>>> &exprs,
                               const shared_ptr<OperatorTensor<S, FL>> &lopt,
                               const shared_ptr<OperatorTensor<S, FL>> &ropt,
                               const shared_ptr<SparseMatrix<S, FL>> &cmat,
                               const shared_ptr<SparseMatrix<S, FL>> &vmat,
                               bool all_reduce) const {
        vector<pair<shared_ptr<OpExpr<S>>, FL>> expectations(names.size());
        map<tuple<uint8_t, S, S>,
            unordered_map<shared_ptr<OpExpr<S>>,
                          shared_ptr<SparseMatrix<S, FL>>>>
            partials;
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(names.size() == exprs.size());
        S ket_dq = cmat->info->delta_quantum;
        S bra_dq = vmat->info->delta_quantum;
        for (size_t k = 0; k < exprs.size(); k++) {
            // may happen for NPDM with ancilla
            using OESF = OpElement<S, FL>;
            assert(dynamic_pointer_cast<OESF>(names[k])->name != OpNames::Zero);
            shared_ptr<OpExpr<S>> expr = exprs[k];
            S opdq = dynamic_pointer_cast<OpElement<S, FL>>(names[k])->q_label;
            if (opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                continue;
            switch (expr->get_type()) {
            case OpTypes::SumProd:
                throw runtime_error("Tensor product expectation with delayed "
                                    "contraction not yet supported.");
                break;
            case OpTypes::Prod: {
                shared_ptr<OpProduct<S, FL>> op =
                    dynamic_pointer_cast<OpProduct<S, FL>>(expr);
                assert(op->a != nullptr && op->b != nullptr);
                assert(lopt->ops.count(op->a) != 0 &&
                       ropt->ops.count(op->b) != 0);
                if (lopt->get_type() == OperatorTensorTypes::Delayed ||
                    ropt->get_type() == OperatorTensorTypes::Delayed)
                    throw runtime_error(
                        "Tensor product expectation with delayed "
                        "contraction not yet supported.");
                shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
                shared_ptr<SparseMatrix<S, FL>> rmat =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
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
                shared_ptr<OpSum<S, FL>> sop =
                    dynamic_pointer_cast<OpSum<S, FL>>(expr);
                for (size_t j = 0; j < sop->strings.size(); j++) {
                    shared_ptr<OpProduct<S, FL>> op = sop->strings[j];
                    assert(op->a != nullptr && op->b != nullptr);
                    assert(lopt->ops.count(op->a) != 0 &&
                           ropt->ops.count(op->b) != 0);
                    if (lopt->get_type() == OperatorTensorTypes::Delayed ||
                        ropt->get_type() == OperatorTensorTypes::Delayed)
                        throw runtime_error(
                            "Tensor product expectation with delayed "
                            "contraction not yet supported.");
                    shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
                    shared_ptr<SparseMatrix<S, FL>> rmat =
                        make_shared<SparseMatrix<S, FL>>(d_alloc);
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
                     shared_ptr<SparseMatrix<S, FL>>>>
            vparts;
        for (auto &m : partials)
            for (auto &mm : m.second) {
                vparts.push_back(make_tuple(get<0>(m.first), get<2>(m.first),
                                            mm.first, mm.second));
                mm.second->allocate(mm.second->info);
            }
        parallel_for(vparts.size(), [&vparts, &lopt, &cmat, &vmat](
                                        const shared_ptr<TensorFunctions> &tf,
                                        size_t i) {
            uint8_t conj = get<0>(vparts[i]);
            S opdq = get<1>(vparts[i]);
            shared_ptr<SparseMatrix<S, FL>> lmat =
                lopt->ops.at(get<2>(vparts[i]));
            shared_ptr<SparseMatrix<S, FL>> rmat = get<3>(vparts[i]);
            tf->opf->tensor_left_partial_expectation(conj, lmat, rmat, cmat,
                                                     vmat, opdq);
        });
        vector<size_t> prod_idxs;
        prod_idxs.reserve(exprs.size());
        for (size_t k = 0; k < exprs.size(); k++) {
            shared_ptr<OpExpr<S>> expr = exprs[k];
            S opdq = dynamic_pointer_cast<OpElement<S, FL>>(names[k])->q_label;
            expectations[k] = make_pair(names[k], 0.0);
            if (opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                continue;
            switch (expr->get_type()) {
            case OpTypes::Prod:
                prod_idxs.push_back(k);
                break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S, FL>> sop =
                    dynamic_pointer_cast<OpSum<S, FL>>(expr);
                int ntop = threading->activate_operator();
                vector<FL> rs(ntop, 0.0);
#pragma omp parallel for schedule(dynamic) num_threads(ntop)
                for (int j = 0; j < (int)sop->strings.size(); j++) {
                    shared_ptr<OpProduct<S, FL>> op = sop->strings[j];
                    shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
                    shared_ptr<SparseMatrix<S, FL>> lmat =
                        partials
                            .at(make_tuple(op->conj, rmat->info->delta_quantum,
                                           opdq))
                            .at(op->a);
                    int tid = threading->get_thread_id();
                    rs[tid] += (op->conj & 2)
                                   ? xconj<FL>(opf->dot_product(
                                         lmat, rmat, xconj<FL>(op->factor)))
                                   : opf->dot_product(lmat, rmat, op->factor);
                }
                FL r = accumulate(rs.begin(), rs.end(), (FL)0.0);
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
            [&prod_idxs, &ropt, &partials, &exprs, &names,
             &expectations](const shared_ptr<TensorFunctions> &tf, size_t pk) {
                size_t k = prod_idxs[pk];
                shared_ptr<OpExpr<S>> expr = exprs[k];
                S opdq =
                    dynamic_pointer_cast<OpElement<S, FL>>(names[k])->q_label;
                shared_ptr<OpProduct<S, FL>> op =
                    dynamic_pointer_cast<OpProduct<S, FL>>(expr);
                shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
                shared_ptr<SparseMatrix<S, FL>> lmat =
                    partials
                        .at(make_tuple(op->conj, rmat->info->delta_quantum,
                                       opdq))
                        .at(op->a);
                expectations[k] = make_pair(
                    names[k],
                    (op->conj & 2)
                        ? xconj<FL>(tf->opf->dot_product(lmat, rmat,
                                                         xconj<FL>(op->factor)))
                        : tf->opf->dot_product(lmat, rmat, op->factor));
            });
        for (auto &vpart : vparts)
            get<3>(vpart)->deallocate();
        return expectations;
    }
    // vmat = expr x cmat
    virtual void
    tensor_product_multiply(const shared_ptr<OpExpr<S>> &expr,
                            const shared_ptr<OperatorTensor<S, FL>> &lopt,
                            const shared_ptr<OperatorTensor<S, FL>> &ropt,
                            const shared_ptr<SparseMatrix<S, FL>> &cmat,
                            const shared_ptr<SparseMatrix<S, FL>> &vmat, S opdq,
                            bool all_reduce) const {
        switch (expr->get_type()) {
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S, FL>> op =
                dynamic_pointer_cast<OpSumProd<S, FL>>(expr);
            assert(op->a != nullptr && op->b != nullptr && op->ops.size() == 2);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            bool dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
            assert((dleft ? lopt : ropt)->get_type() ==
                   OperatorTensorTypes::Delayed);
            shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
            shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(
                    dleft ? lopt : ropt);
            assert(dopt->lopt->ops.count(op->ops[0]) != 0);
            assert(dopt->ropt->ops.count(op->ops[1]) != 0);
            shared_ptr<SparseMatrix<S, FL>> dlmat =
                dopt->lopt->ops.at(op->ops[0]);
            shared_ptr<SparseMatrix<S, FL>> drmat =
                dopt->ropt->ops.at(op->ops[1]);
            uint8_t dconj = (uint8_t)op->conjs[0] | (op->conjs[1] << 1);
            opf->three_tensor_product_multiply(op->conj, lmat, rmat, cmat, vmat,
                                               dconj, dlmat, drmat, dleft, opdq,
                                               op->factor);
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->a != nullptr && op->b != nullptr);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
            opf->tensor_product_multiply(op->conj, lmat, rmat, cmat, vmat, opdq,
                                         op->factor);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            parallel_reduce(
                op->strings.size(), vmat,
                [&op, &lopt, &ropt, &cmat,
                 &opdq](const shared_ptr<TensorFunctions> &tf,
                        const shared_ptr<SparseMatrix<S, FL>> &vmat, size_t i) {
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
                            const shared_ptr<OperatorTensor<S, FL>> &lopt,
                            const shared_ptr<OperatorTensor<S, FL>> &ropt,
                            const shared_ptr<SparseMatrix<S, FL>> &mat,
                            S opdq) const {
        switch (expr->get_type()) {
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S, FL>> op =
                dynamic_pointer_cast<OpSumProd<S, FL>>(expr);
            assert(op->a != nullptr && op->b != nullptr && op->ops.size() == 2);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            bool dleft = lopt->get_type() == OperatorTensorTypes::Delayed;
            assert((dleft ? lopt : ropt)->get_type() ==
                   OperatorTensorTypes::Delayed);
            shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
            shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(
                    dleft ? lopt : ropt);
            assert(dopt->lopt->ops.count(op->ops[0]) != 0);
            assert(dopt->ropt->ops.count(op->ops[1]) != 0);
            shared_ptr<SparseMatrix<S, FL>> dlmat =
                dopt->lopt->ops.at(op->ops[0]);
            shared_ptr<SparseMatrix<S, FL>> drmat =
                dopt->ropt->ops.at(op->ops[1]);
            uint8_t dconj = (uint8_t)op->conjs[0] | (op->conjs[1] << 1);
            opf->three_tensor_product_diagonal(op->conj, lmat, rmat, mat, dconj,
                                               dlmat, drmat, dleft, opdq,
                                               op->factor);
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->a != nullptr && op->b != nullptr);
            assert(lopt->ops.count(op->a) != 0 && ropt->ops.count(op->b) != 0);
            shared_ptr<SparseMatrix<S, FL>> lmat = lopt->ops.at(op->a);
            shared_ptr<SparseMatrix<S, FL>> rmat = ropt->ops.at(op->b);
            opf->tensor_product_diagonal(op->conj, lmat, rmat, mat, opdq,
                                         op->factor);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            parallel_reduce(
                op->strings.size(), mat,
                [&op, &lopt, &ropt,
                 &opdq](const shared_ptr<TensorFunctions> &tf,
                        const shared_ptr<SparseMatrix<S, FL>> &mat, size_t i) {
                    tf->tensor_product_diagonal(op->strings[i], lopt, ropt, mat,
                                                opdq);
                });
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
            else if (opf->seq->mode & SeqTypes::Tasked)
                opf->seq->auto_perform(
                    GMatrix<FL>(mat->data, (MKL_INT)mat->total_memory, 1));
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    // mat = eval(expr)
    virtual void
    tensor_product(const shared_ptr<OpExpr<S>> &expr,
                   const unordered_map<shared_ptr<OpExpr<S>>,
                                       shared_ptr<SparseMatrix<S, FL>>> &lop,
                   const unordered_map<shared_ptr<OpExpr<S>>,
                                       shared_ptr<SparseMatrix<S, FL>>> &rop,
                   shared_ptr<SparseMatrix<S, FL>> &mat) const {
        switch (expr->get_type()) {
        case OpTypes::Elem: {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(expr);
            assert((rop.count(op) != 0) ^ (lop.count(op) != 0));
            shared_ptr<SparseMatrix<S, FL>> lmat =
                lop.count(op) != 0 ? lop.at(op)
                                   : lop.at(make_shared<OpExpr<S>>());
            shared_ptr<SparseMatrix<S, FL>> rmat =
                rop.count(op) != 0 ? rop.at(op)
                                   : rop.at(make_shared<OpExpr<S>>());
            opf->tensor_product(0, lmat, rmat, mat, op->factor);
        } break;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->b != nullptr);
            // if (lop.count(op->a) == 0)
            //     cout << "missing op->a : " << *op->a << endl;
            // if (rop.count(op->b) == 0)
            //     cout << "missing op->b : " << *op->b << endl;
            assert(lop.count(op->a) != 0 && rop.count(op->b) != 0);
            shared_ptr<SparseMatrix<S, FL>> lmat = lop.at(op->a);
            shared_ptr<SparseMatrix<S, FL>> rmat = rop.at(op->b);
            // here in parallel when not allocated mat->factor = 0, mat->data =
            // 0 but if mat->info->n = 0, mat->data also = 0
            opf->tensor_product(op->conj, lmat, rmat, mat, op->factor);
        } break;
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S, FL>> op =
                dynamic_pointer_cast<OpSumProd<S, FL>>(expr);
            assert((op->a == nullptr) ^ (op->b == nullptr));
            assert(op->ops.size() != 0);
            bool has_intermediate = false;
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            shared_ptr<SparseMatrix<S, FL>> tmp =
                make_shared<SparseMatrix<S, FL>>(d_alloc);
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
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            for (auto &x : op->strings)
                if (x->get_type() == OpTypes::Prod && x->b == nullptr)
                    tensor_product(x->get_op(), lop, rop, mat);
                else
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
    virtual void left_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                             const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                             const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                             shared_ptr<OperatorTensor<S, FL>> &c) const {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
        }
        parallel_for(a->lmat->data.size(),
                     [&a, &c, &mpst_bra, &mpst_ket](
                         const shared_ptr<TensorFunctions> &tf, size_t i) {
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
    virtual void right_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                              const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                              const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                              shared_ptr<OperatorTensor<S, FL>> &c) const {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
        }
        parallel_for(a->rmat->data.size(),
                     [&a, &c, &mpst_bra, &mpst_ket](
                         const shared_ptr<TensorFunctions> &tf, size_t i) {
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
                               const shared_ptr<OperatorTensor<S, FL>> &a,
                               bool left) const {
        vector<vector<shared_ptr<OpSumProd<S, FL>>>> exs;
        vector<int> maxk;
        exs.reserve(exprs->data.size());
        maxk.reserve(exprs->data.size());
        for (size_t i = 0; i < exprs->data.size(); i++)
            if (exprs->data[i] != nullptr &&
                exprs->data[i]->get_type() == OpTypes::Sum) {
                shared_ptr<OpSum<S, FL>> expr =
                    dynamic_pointer_cast<OpSum<S, FL>>(exprs->data[i]);
                exs.push_back(vector<shared_ptr<OpSumProd<S, FL>>>());
                maxk.push_back(0);
                for (size_t j = 0; j < expr->strings.size(); j++)
                    if (expr->strings[j]->get_type() == OpTypes::SumProd) {
                        shared_ptr<OpSumProd<S, FL>> ex =
                            dynamic_pointer_cast<OpSumProd<S, FL>>(
                                expr->strings[j]);
                        if ((left && ex->b == nullptr) ||
                            (!left && ex->a == nullptr) || ex->c == nullptr)
                            continue;
                        if (a->ops.count(ex->c) != 0)
                            continue;
                        shared_ptr<SparseMatrix<S, FL>> tmp =
                            make_shared<SparseMatrix<S, FL>>();
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
                                  &a](const shared_ptr<TensorFunctions> &tf,
                                      size_t i) {
            for (int k = 0; k < maxk[i]; k++) {
                for (auto &ex : exs[i]) {
                    if (k < ex->ops.size()) {
                        shared_ptr<SparseMatrix<S, FL>> xmat = a->ops.at(
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
    numerical_transform(const shared_ptr<OperatorTensor<S, FL>> &a,
                        const shared_ptr<Symbolic<S>> &names,
                        const shared_ptr<Symbolic<S>> &exprs) const {
        assert(names->data.size() == exprs->data.size());
        assert((a->lmat == nullptr) ^ (a->rmat == nullptr));
        if (a->lmat == nullptr)
            a->rmat = names;
        else
            a->lmat = names;
        vector<pair<shared_ptr<SparseMatrix<S, FL>>, shared_ptr<OpSum<S, FL>>>>
            trs;
        trs.reserve(names->data.size());
        int maxi = 0;
        for (size_t k = 0; k < names->data.size(); k++) {
            if (exprs->data[k]->get_type() == OpTypes::Zero)
                continue;
            shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
            shared_ptr<OpExpr<S>> expr =
                exprs->data[k] *
                ((FP)1.0 /
                 dynamic_pointer_cast<OpElement<S, FL>>(names->data[k])
                     ->factor);
            assert(a->ops.count(nop) != 0);
            shared_ptr<SparseMatrix<S, FL>> anop = a->ops.at(nop);
            switch (expr->get_type()) {
            case OpTypes::Sum:
                trs.push_back(
                    make_pair(anop, dynamic_pointer_cast<OpSum<S, FL>>(expr)));
                maxi = max(maxi, (int)dynamic_pointer_cast<OpSum<S, FL>>(expr)
                                     ->strings.size());
                break;
            case OpTypes::Zero:
                break;
            default:
                assert(false);
                break;
            }
        }
        parallel_for(trs.size(), [&trs,
                                  &a](const shared_ptr<TensorFunctions> &tf,
                                      size_t i) {
            shared_ptr<OpSum<S, FL>> op = trs[i].second;
            for (size_t j = 0; j < op->strings.size(); j++) {
                shared_ptr<OpElement<S, FL>> nexpr = op->strings[j]->get_op();
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
    post_numerical_transform(const shared_ptr<OperatorTensor<S, FL>> &a,
                             const shared_ptr<Symbolic<S>> &names,
                             const shared_ptr<Symbolic<S>> &new_names) const {
        set<shared_ptr<OpExpr<S>>, op_expr_less<S>> del_ops;
        for (size_t j = 0; j < names->data.size(); j++)
            del_ops.insert(names->data[j]);
        for (size_t j = 0; j < new_names->data.size(); j++)
            if (del_ops.count(new_names->data[j]))
                del_ops.erase(new_names->data[j]);
        vector<tuple<FL *, shared_ptr<SparseMatrix<S, FL>>, uint8_t>> mp;
        vector<tuple<FL *, shared_ptr<SparseMatrix<S, FL>>, uint8_t>> mp_ext;
        mp.reserve(a->ops.size());
        mp_ext.reserve(a->ops.size());
        for (auto it = a->ops.cbegin(); it != a->ops.cend(); it++)
            if (it->second->total_memory != 0) {
                if (it->second->alloc == dalloc_<FP>())
                    mp.emplace_back(it->second->data, it->second,
                                    del_ops.count(it->first));
                else
                    mp_ext.emplace_back(it->second->data, it->second,
                                        del_ops.count(it->first));
            }
        sort(
            mp.begin(), mp.end(),
            [](const tuple<FL *, shared_ptr<SparseMatrix<S, FL>>, uint8_t> &a,
               const tuple<FL *, shared_ptr<SparseMatrix<S, FL>>, uint8_t> &b) {
                return get<0>(a) < get<0>(b);
            });
        sort(
            mp_ext.begin(), mp_ext.end(),
            [](const tuple<FL *, shared_ptr<SparseMatrix<S, FL>>, uint8_t> &a,
               const tuple<FL *, shared_ptr<SparseMatrix<S, FL>>, uint8_t> &b) {
                return get<0>(a) > get<0>(b);
            });
        for (const auto &t : mp)
            get<1>(t)->reallocate(get<2>(t) ? 0 : get<1>(t)->total_memory);
        for (const auto &t : mp_ext)
            if (get<2>(t))
                get<1>(t)->deallocate();
            else
                get<1>(t)->reallocate(dalloc_<FP>());
    }
    // Substituing delayed left experssions
    // Return sum of three-operator tensor products
    virtual shared_ptr<Symbolic<S>>
    substitute_delayed_exprs(const shared_ptr<Symbolic<S>> &exprs,
                             const shared_ptr<DelayedOperatorTensor<S, FL>> &a,
                             bool left, OpNamesSet delayed,
                             bool use_orig = true) const {
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<OpExpr<S>>> aops;
        shared_ptr<Symbolic<S>> amat = left ? a->lmat : a->rmat;
        assert(amat->data.size() == a->mat->data.size());
        aops.reserve(amat->data.size());
        for (size_t i = 0; i < amat->data.size(); i++) {
            shared_ptr<OpElement<S, FL>> aop =
                dynamic_pointer_cast<OpElement<S, FL>>(amat->data[i]);
            shared_ptr<OpExpr<S>> op = abs_value(amat->data[i]);
            shared_ptr<OpExpr<S>> expr =
                a->mat->data[i] * ((FP)1.0 / aop->factor);
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
            vector<shared_ptr<OpProduct<S, FL>>> prods;
            switch (expr->get_type()) {
            case OpTypes::Prod:
                prods.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(expr));
                break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S, FL>> op =
                    dynamic_pointer_cast<OpSum<S, FL>>(expr);
                prods.insert(prods.end(), op->strings.begin(),
                             op->strings.end());
            } break;
            case OpTypes::Zero:
                break;
            default:
                assert(false);
                break;
            }
            vector<shared_ptr<OpProduct<S, FL>>> rr;
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
                    vector<shared_ptr<OpSumProd<S, FL>>> mrs;
                    switch (op_expr->get_type()) {
                    case OpTypes::SumProd:
                        mrs.push_back(
                            dynamic_pointer_cast<OpSumProd<S, FL>>(op_expr));
                    case OpTypes::Sum: {
                        shared_ptr<OpSum<S, FL>> sop =
                            dynamic_pointer_cast<OpSum<S, FL>>(op_expr);
                        for (auto &op : sop->strings)
                            if (op->get_type() == OpTypes::SumProd)
                                mrs.push_back(
                                    dynamic_pointer_cast<OpSumProd<S, FL>>(op));
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
                vector<shared_ptr<OpProduct<S, FL>>> rk;
                vector<shared_ptr<OpSumProd<S, FL>>> rs;
                switch (hexpr->get_type()) {
                case OpTypes::SumProd:
                    rs.push_back(dynamic_pointer_cast<OpSumProd<S, FL>>(hexpr));
                case OpTypes::Prod:
                    rk.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(hexpr));
                    break;
                case OpTypes::Sum: {
                    shared_ptr<OpSum<S, FL>> sop =
                        dynamic_pointer_cast<OpSum<S, FL>>(hexpr);
                    for (auto &op : sop->strings)
                        if (op->get_type() == OpTypes::Prod)
                            rk.push_back(op);
                        else if (op->get_type() == OpTypes::SumProd)
                            rs.push_back(
                                dynamic_pointer_cast<OpSumProd<S, FL>>(op));
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
                                rk.push_back(make_shared<OpProduct<S, FL>>(
                                    op->a, op->c, op->factor, cj));
                        } else {
                            // TEMP may not have terms in all procs
                            if (a->lopt->ops.count(op->c))
                                rk.push_back(make_shared<OpProduct<S, FL>>(
                                    op->c, op->b, op->factor, cj));
                        }
                    } else if (op->b == nullptr) {
                        for (size_t j = 0; j < op->ops.size(); j++)
                            rk.push_back(make_shared<OpProduct<S, FL>>(
                                op->a, op->ops[j], op->factor,
                                op->conj ^ (op->conjs[j] << 1)));
                    } else {
                        for (size_t j = 0; j < op->ops.size(); j++)
                            rk.push_back(make_shared<OpProduct<S, FL>>(
                                op->ops[j], op->b, op->factor,
                                op->conj ^ (uint8_t)op->conjs[j]));
                    }
                }
                for (auto &op : rk)
                    rr.push_back(make_shared<OpSumProd<S, FL>>(
                        prod->a, prod->b,
                        vector<shared_ptr<OpElement<S, FL>>>{op->a, op->b},
                        vector<bool>{(bool)(op->conj & 1),
                                     (bool)(op->conj & 2)},
                        prod->factor * op->factor, prod->conj));
            }
            rexpr[i] = make_shared<OpSum<S, FL>>(rr);
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
    virtual shared_ptr<DelayedOperatorTensor<S, FL>>
    delayed_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                     const shared_ptr<OperatorTensor<S, FL>> &b,
                     const shared_ptr<OpExpr<S>> &op,
                     OpNamesSet delayed) const {
        shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
            make_shared<DelayedOperatorTensor<S, FL>>();
        dopt->lopt = a, dopt->ropt = b;
        dopt->dops.push_back(op);
        assert(a->lmat->data.size() == b->rmat->data.size());
        shared_ptr<Symbolic<S>> exprs = a->lmat * b->rmat;
        assert(exprs->data.size() == 1);
        if (a->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(a),
                true, delayed);
        else if (b->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(b),
                false, delayed);
        else
            dopt->mat = exprs;
        return dopt;
    }
    // delayed left and right block contraction (for effective hamil)
    // using the pre-computed exprs
    virtual shared_ptr<DelayedOperatorTensor<S, FL>>
    delayed_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                     const shared_ptr<OperatorTensor<S, FL>> &b,
                     const shared_ptr<Symbolic<S>> &ops,
                     const shared_ptr<Symbolic<S>> &exprs,
                     OpNamesSet delayed) const {
        shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
            make_shared<DelayedOperatorTensor<S, FL>>();
        dopt->lopt = a, dopt->ropt = b;
        dopt->dops = ops->data;
        if (a->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(a),
                true, delayed);
        else if (b->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(b),
                false, delayed);
        else
            dopt->mat = exprs;
        return dopt;
    }
    // c = a x b (dot) (delayed for 3-operator operations)
    virtual void delayed_left_contract(
        const shared_ptr<OperatorTensor<S, FL>> &a,
        const shared_ptr<OperatorTensor<S, FL>> &b,
        shared_ptr<OperatorTensor<S, FL>> &c,
        const shared_ptr<Symbolic<S>> &cexprs = nullptr) const {
        if (a == nullptr)
            return left_contract(a, b, c, cexprs);
        shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
            make_shared<DelayedOperatorTensor<S, FL>>();
        dopt->mat = cexprs == nullptr ? a->lmat * b->lmat : cexprs;
        dopt->lopt = a, dopt->ropt = b;
        dopt->ops = c->ops;
        dopt->lmat = c->lmat, dopt->rmat = c->rmat;
        c = dopt;
    }
    // c = b (dot) x a (delayed for 3-operator operations)
    virtual void delayed_right_contract(
        const shared_ptr<OperatorTensor<S, FL>> &a,
        const shared_ptr<OperatorTensor<S, FL>> &b,
        shared_ptr<OperatorTensor<S, FL>> &c,
        const shared_ptr<Symbolic<S>> &cexprs = nullptr) const {
        if (a == nullptr)
            return right_contract(a, b, c, cexprs);
        shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
            make_shared<DelayedOperatorTensor<S, FL>>();
        dopt->mat = cexprs == nullptr ? b->rmat * a->rmat : cexprs;
        dopt->lopt = b, dopt->ropt = a;
        dopt->ops = c->ops;
        dopt->lmat = c->lmat, dopt->rmat = c->rmat;
        c = dopt;
    }
    // c = a x b (dot)
    // dot means it is from the dot block
    virtual void left_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                               const shared_ptr<OperatorTensor<S, FL>> &b,
                               shared_ptr<OperatorTensor<S, FL>> &c,
                               const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                               OpNamesSet delayed = OpNamesSet()) const {
        if (frame_<FP>()->use_main_stack)
            for (auto &p : c->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
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
                 &delayed](const shared_ptr<TensorFunctions> &tf, size_t i) {
                    shared_ptr<OpElement<S, FL>> cop =
                        dynamic_pointer_cast<OpElement<S, FL>>(
                            c->lmat->data[i]);
                    shared_ptr<OpExpr<S>> op = abs_value(c->lmat->data[i]);
                    shared_ptr<OpExpr<S>> expr =
                        exprs->data[i] * ((FP)1.0 / cop->factor);
                    if (!delayed(cop->name)) {
                        if (!frame_<FP>()->use_main_stack) {
                            // skip cached part
                            if (c->ops.at(op)->alloc != nullptr)
                                return;
                            c->ops.at(op)->alloc =
                                make_shared<VectorAllocator<FP>>();
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
    // dot means it is from the dot block
    virtual void right_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                                const shared_ptr<OperatorTensor<S, FL>> &b,
                                shared_ptr<OperatorTensor<S, FL>> &c,
                                const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                                OpNamesSet delayed = OpNamesSet()) const {
        if (frame_<FP>()->use_main_stack)
            for (auto &p : c->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
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
                 &delayed](const shared_ptr<TensorFunctions> &tf, size_t i) {
                    shared_ptr<OpElement<S, FL>> cop =
                        dynamic_pointer_cast<OpElement<S, FL>>(
                            c->rmat->data[i]);
                    shared_ptr<OpExpr<S>> op = abs_value(c->rmat->data[i]);
                    shared_ptr<OpExpr<S>> expr =
                        exprs->data[i] * ((FP)1.0 / cop->factor);
                    if (!delayed(cop->name)) {
                        if (!frame_<FP>()->use_main_stack) {
                            // skip cached part
                            if (c->ops.at(op)->alloc != nullptr)
                                return;
                            c->ops.at(op)->alloc =
                                make_shared<VectorAllocator<FP>>();
                            c->ops.at(op)->allocate(c->ops.at(op)->info);
                        }
                        tf->tensor_product(expr, b->ops, a->ops, c->ops.at(op));
                    }
                });
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
    // c = mpst_bra x [ a x b (dot) ] x mpst_ket
    // without consuming large memory of blocking
    // need to make sure a and c are not in the same frame
    virtual void
    left_contract_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                         const shared_ptr<OperatorTensor<S, FL>> &b,
                         const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                         const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                         shared_ptr<OperatorTensor<S, FL>> &ab,
                         shared_ptr<OperatorTensor<S, FL>> &c,
                         const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                         OpNamesSet delayed = OpNamesSet()) const {
        if (frame_<FP>()->use_main_stack)
            for (auto &p : ab->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                if (a == nullptr || !delayed(op->name))
                    ab->ops.at(op)->allocate(ab->ops.at(op)->info);
            }
        if (a == nullptr) {
            left_assign(b, ab);
            left_rotate(ab, mpst_bra, mpst_ket, c);
        } else {
            for (auto &p : c->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                c->ops.at(op)->allocate(c->ops.at(op)->info);
            }
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == ab->lmat->data.size());
            // because of deallocation of the ab, we cannot use auto
            assert(opf->seq->mode != SeqTypes::Auto);
            parallel_for(
                exprs->data.size(),
                [&a, &b, &ab, &c, &mpst_bra, &mpst_ket, &exprs,
                 &delayed](const shared_ptr<TensorFunctions> &tf, size_t i) {
                    shared_ptr<OpElement<S, FL>> cop =
                        dynamic_pointer_cast<OpElement<S, FL>>(
                            ab->lmat->data[i]);
                    shared_ptr<OpExpr<S>> op = abs_value(ab->lmat->data[i]);
                    shared_ptr<OpExpr<S>> expr =
                        exprs->data[i] * ((FP)1.0 / cop->factor);
                    if (!delayed(cop->name)) {
                        if (!frame_<FP>()->use_main_stack) {
                            // skip cached part
                            if (ab->ops.at(op)->alloc != nullptr)
                                return;
                            ab->ops.at(op)->alloc =
                                make_shared<VectorAllocator<FP>>();
                            ab->ops.at(op)->allocate(ab->ops.at(op)->info);
                        }
                        tf->tensor_product(expr, a->ops, b->ops,
                                           ab->ops.at(op));
                        tf->opf->tensor_rotate(ab->ops.at(op), c->ops.at(op),
                                               mpst_bra, mpst_ket, false);
                        if (!frame_<FP>()->use_main_stack)
                            ab->ops.at(op)->deallocate();
                    }
                });
        }
    }
    // c = mpst_bra x [ b (dot) x a ] x mpst_ket
    // without consuming large memory of blocking
    // need to make sure a and c are not in the same frame
    virtual void
    right_contract_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                          const shared_ptr<OperatorTensor<S, FL>> &b,
                          const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                          const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                          shared_ptr<OperatorTensor<S, FL>> &ab,
                          shared_ptr<OperatorTensor<S, FL>> &c,
                          const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                          OpNamesSet delayed = OpNamesSet()) const {
        if (frame_<FP>()->use_main_stack)
            for (auto &p : ab->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                if (a == nullptr || !delayed(op->name))
                    ab->ops.at(op)->allocate(ab->ops.at(op)->info);
            }
        if (a == nullptr) {
            right_assign(b, ab);
            right_rotate(ab, mpst_bra, mpst_ket, c);
        } else {
            for (auto &p : c->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                c->ops.at(op)->allocate(c->ops.at(op)->info);
            }
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == ab->rmat->data.size());
            // because of deallocation of the ab, we cannot use auto
            assert(opf->seq->mode != SeqTypes::Auto);
            parallel_for(
                exprs->data.size(),
                [&a, &b, &ab, &c, &mpst_bra, &mpst_ket, &exprs,
                 &delayed](const shared_ptr<TensorFunctions> &tf, size_t i) {
                    shared_ptr<OpElement<S, FL>> cop =
                        dynamic_pointer_cast<OpElement<S, FL>>(
                            ab->rmat->data[i]);
                    shared_ptr<OpExpr<S>> op = abs_value(ab->rmat->data[i]);
                    shared_ptr<OpExpr<S>> expr =
                        exprs->data[i] * ((FP)1.0 / cop->factor);
                    if (!delayed(cop->name)) {
                        if (!frame_<FP>()->use_main_stack) {
                            // skip cached part
                            if (ab->ops.at(op)->alloc != nullptr)
                                return;
                            ab->ops.at(op)->alloc =
                                make_shared<VectorAllocator<FP>>();
                            ab->ops.at(op)->allocate(ab->ops.at(op)->info);
                        }
                        tf->tensor_product(expr, b->ops, a->ops,
                                           ab->ops.at(op));
                        tf->opf->tensor_rotate(ab->ops.at(op), c->ops.at(op),
                                               mpst_bra, mpst_ket, true);
                        if (!frame_<FP>()->use_main_stack)
                            ab->ops.at(op)->deallocate();
                    }
                });
        }
    }
};

} // namespace block2
