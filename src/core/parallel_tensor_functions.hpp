
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

#include "parallel_rule.hpp"
#include "tensor_functions.hpp"
#include <cassert>
#include <map>
#include <memory>
#include <set>

using namespace std;

namespace block2 {

// Operations for operator tensors (distributed parallel)
template <typename S, typename FL>
struct ParallelTensorFunctions : TensorFunctions<S, FL> {
    using typename TensorFunctions<S, FL>::FP;
    using TensorFunctions<S, FL>::opf;
    using TensorFunctions<S, FL>::parallel_for;
    using TensorFunctions<S, FL>::substitute_delayed_exprs;
    shared_ptr<ParallelRule<S, FL>> rule;
    ParallelTensorFunctions(const shared_ptr<OperatorFunctions<S, FL>> &opf,
                            const shared_ptr<ParallelRule<S, FL>> &rule)
        : TensorFunctions<S, FL>(opf), rule(rule) {}
    shared_ptr<TensorFunctions<S, FL>> copy() const override {
        return make_shared<ParallelTensorFunctions<S, FL>>(opf->copy(), rule);
    }
    TensorFunctionsTypes get_type() const override {
        return TensorFunctionsTypes::Parallel;
    }
    void operator()(const GMatrix<FL> &b, const GMatrix<FL> &c,
                    FL scale = (FL)1.0) override {
        opf->seq->operator()(b, c, scale);
        rule->comm->allreduce_sum(c.data, c.size());
    }
    // c = a
    void left_assign(const shared_ptr<OperatorTensor<S, FL>> &a,
                     shared_ptr<OperatorTensor<S, FL>> &c) const override {
        assert(a->lmat != nullptr);
        assert(a->lmat->get_type() == SymTypes::RVec);
        assert(c->lmat != nullptr);
        assert(c->lmat->get_type() == SymTypes::RVec);
        assert(a->lmat->data.size() == c->lmat->data.size());
        vector<size_t> idxs;
        idxs.reserve(a->lmat->data.size());
        for (size_t i = 0; i < a->lmat->data.size(); i++) {
            if (a->lmat->data[i]->get_type() == OpTypes::Zero)
                c->lmat->data[i] = a->lmat->data[i];
            else {
                assert(a->lmat->data[i] == c->lmat->data[i]);
                shared_ptr<OpExpr<S>> pa = abs_value(a->lmat->data[i]),
                                      pc = abs_value(c->lmat->data[i]);
                if (rule->available(pc)) {
                    assert(rule->available(pa));
                    // skip cached part
                    if (c->ops[pc]->alloc != nullptr)
                        return;
                    assert(c->ops[pc]->data == nullptr);
                    if (frame_<FP>()->use_main_stack)
                        c->ops[pc]->allocate(c->ops[pc]->info);
                    idxs.push_back(i);
                    c->ops[pc]->factor = a->ops[pa]->factor;
                } else if (rule->partial(pc) && (rule->get_parallel_type() &
                                                 ParallelTypes::NewScheme))
                    c->ops[pc]->factor = 0;
            }
        }
        parallel_for(
            idxs.size(),
            [&a, &c, &idxs](const shared_ptr<TensorFunctions<S, FL>> &tf,
                            size_t ii) {
                size_t i = idxs[ii];
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
            });
    }
    // c = a
    void right_assign(const shared_ptr<OperatorTensor<S, FL>> &a,
                      shared_ptr<OperatorTensor<S, FL>> &c) const override {
        assert(a->rmat != nullptr);
        assert(a->rmat->get_type() == SymTypes::CVec);
        assert(c->rmat != nullptr);
        assert(c->rmat->get_type() == SymTypes::CVec);
        assert(a->rmat->data.size() == c->rmat->data.size());
        vector<size_t> idxs;
        idxs.reserve(a->lmat->data.size());
        for (size_t i = 0; i < a->rmat->data.size(); i++) {
            if (a->rmat->data[i]->get_type() == OpTypes::Zero)
                c->rmat->data[i] = a->rmat->data[i];
            else {
                assert(a->rmat->data[i] == c->rmat->data[i]);
                shared_ptr<OpExpr<S>> pa = abs_value(a->rmat->data[i]),
                                      pc = abs_value(c->rmat->data[i]);
                if (rule->available(pc)) {
                    assert(rule->available(pa));
                    // skip cached part
                    if (c->ops[pc]->alloc != nullptr)
                        return;
                    assert(c->ops[pc]->data == nullptr);
                    if (frame_<FP>()->use_main_stack)
                        c->ops[pc]->allocate(c->ops[pc]->info);
                    idxs.push_back(i);
                    c->ops[pc]->factor = a->ops[pa]->factor;
                } else if (rule->partial(pc) && (rule->get_parallel_type() &
                                                 ParallelTypes::NewScheme))
                    c->ops[pc]->factor = 0;
            }
        }
        parallel_for(
            idxs.size(),
            [&a, &c, &idxs](const shared_ptr<TensorFunctions<S, FL>> &tf,
                            size_t ii) {
                size_t i = idxs[ii];
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
            });
    }
    // vmat = expr[L part | R part] x cmat (for perturbative noise)
    void tensor_product_partial_multiply(
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
        bool do_reduce) const override {
        if (expr->get_type() == OpTypes::ExprRef) {
            shared_ptr<OpExprRef<S>> op =
                dynamic_pointer_cast<OpExprRef<S>>(expr);
            TensorFunctions<S, FL>::tensor_product_partial_multiply(
                op->op, lopt, ropt, trace_right, cmat, psubsl, cinfos, vdqs,
                vmats, vidx, tvidx, false);
            if (opf->seq->mode != SeqTypes::Auto &&
                !(opf->seq->mode & SeqTypes::Tasked) && do_reduce)
                rule->comm->reduce_sum(vmats, rule->comm->root);
        } else
            TensorFunctions<S, FL>::tensor_product_partial_multiply(
                expr, lopt, ropt, trace_right, cmat, psubsl, cinfos, vdqs,
                vmats, vidx, tvidx, false);
    }
    // vmats = expr x cmats
    void tensor_product_multi_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S, FL>> &lopt,
        const shared_ptr<OperatorTensor<S, FL>> &ropt,
        const shared_ptr<SparseMatrixGroup<S, FL>> &cmats,
        const shared_ptr<SparseMatrixGroup<S, FL>> &vmats,
        const unordered_map<
            S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>
            &cinfos,
        S opdq, FL factor, bool all_reduce) const override {
        if (expr->get_type() == OpTypes::ExprRef) {
            shared_ptr<OpExprRef<S>> op =
                dynamic_pointer_cast<OpExprRef<S>>(expr);
            TensorFunctions<S, FL>::tensor_product_multi_multiply(
                op->op, lopt, ropt, cmats, vmats, cinfos, opdq, factor, false);
            if (all_reduce)
                rule->comm->allreduce_sum(vmats);
        } else
            TensorFunctions<S, FL>::tensor_product_multi_multiply(
                expr, lopt, ropt, cmats, vmats, cinfos, opdq, factor, false);
    }
    shared_ptr<GTensor<FL, uint64_t>>
    npdm_sort_load_file(const string &filename,
                        bool compressed) const override {
        shared_ptr<GTensor<FL, uint64_t>> p =
            TensorFunctions<S, FL>::npdm_sort_load_file(filename, compressed);
        rule->comm->allreduce_sum(p->data->data(), p->data->size());
        return p;
    }
    vector<pair<shared_ptr<OpExpr<S>>, FL>> tensor_product_npdm_fragment(
        const shared_ptr<NPDMScheme> &scheme, S vacuum, const string &filename,
        int n_sites, int center, int parallel_center,
        const shared_ptr<OperatorTensor<S, FL>> &lopt,
        const shared_ptr<OperatorTensor<S, FL>> &ropt,
        const shared_ptr<SparseMatrix<S, FL>> &cmat,
        const shared_ptr<SparseMatrix<S, FL>> &vmat, bool cache_left,
        bool compressed, bool low_mem) const override {
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
        TensorFunctions<S, FL>::npdm_middle_intermediates(
            scheme, counter, n_sites, center, mshape, mshape_presum);
        shared_ptr<typename TensorFunctions<S, FL>::NPDMIndexer> indexer =
            make_shared<typename TensorFunctions<S, FL>::NPDMIndexer>(
                scheme, counter, center, vacuum);
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
            if (center >= parallel_center && cache_left) {
                if (cx % rule->comm->size != rule->comm->rank)
                    continue;
            } else if (center < parallel_center && is_last && !cache_left) {
                if ((cx - scheme->right_terms.size()) % rule->comm->size !=
                    rule->comm->rank)
                    continue;
            }
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
                                if (center < parallel_center && is_last) {
                                    if ((rx - scheme->right_terms.size()) %
                                            rule->comm->size !=
                                        rule->comm->rank)
                                        continue;
                                }
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
                                vector<uint8_t> rskip(rcnt, 0);
                                if (center < parallel_center && !is_last)
                                    indexer->set_parallel_right_skip(
                                        rx, rcnt, rule->comm->size,
                                        rule->comm->rank, rskip);
                                parallel_for(
                                    (size_t)rcnt,
                                    [&left_partials, &ropt, &result, &indexer,
                                     &rskip, rshift, iresult](
                                        const shared_ptr<TensorFunctions<S, FL>>
                                            &tf,
                                        size_t pk) {
                                        uint64_t ir = (uint64_t)pk;
                                        shared_ptr<SparseMatrix<S, FL>> rmat =
                                            indexer->get_mat(ir + rshift, ropt,
                                                             OpNames::XR);
                                        if (!rskip[ir] && rmat != nullptr &&
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
                    vector<uint8_t> rskip(rcnt, 0);
                    if (center < parallel_center && !is_last)
                        indexer->set_parallel_right_skip(
                            cx, rcnt, rule->comm->size, rule->comm->rank,
                            rskip);
                    for (uint64_t ir = 0; ir < rcnt; ir++) {
                        shared_ptr<SparseMatrix<S, FL>> rmat =
                            indexer->get_mat(ir + rshift, ropt, OpNames::XR);
                        if (!rskip[ir] && rmat != nullptr) {
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
                                if (center >= parallel_center) {
                                    if (lx % rule->comm->size !=
                                        rule->comm->rank)
                                        continue;
                                }
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
                                        const shared_ptr<TensorFunctions<S, FL>>
                                            &tf,
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
                        if (center < parallel_center && is_last) {
                            if ((rx - scheme->right_terms.size()) %
                                    rule->comm->size !=
                                rule->comm->rank) {
                                im++;
                                continue;
                            }
                        }
                        vector<uint8_t> rskip(rcnt, 0);
                        if (center < parallel_center && !is_last)
                            indexer->set_parallel_right_skip(
                                rx, rcnt, rule->comm->size, rule->comm->rank,
                                rskip);
                        r_partials.reserve(rcnt);
                        right_idxs.reserve(rcnt);
                        right_cnts.push_back(rcnt);
                        for (uint64_t ir = 0; ir < rcnt; ir++) {
                            shared_ptr<SparseMatrix<S, FL>> rmat =
                                indexer->get_mat(ir + rshift, ropt,
                                                 OpNames::XR);
                            if (!rskip[ir] && rmat != nullptr) {
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
                        [&c_compute, &c_partials, &cmat,
                         &vmat](const shared_ptr<TensorFunctions<S, FL>> &tf,
                                size_t i) {
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
                         &ml,
                         is_last](const shared_ptr<TensorFunctions<S, FL>> &tf,
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
                        if (center >= parallel_center) {
                            if (lx % rule->comm->size != rule->comm->rank) {
                                im++;
                                continue;
                            }
                        }
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
                    vector<uint8_t> rskip(rcnt, 0);
                    if (center < parallel_center && !is_last)
                        indexer->set_parallel_right_skip(
                            cx, rcnt, rule->comm->size, rule->comm->rank,
                            rskip);
                    for (uint64_t ir = 0; ir < rcnt; ir++) {
                        shared_ptr<SparseMatrix<S, FL>> rmat =
                            indexer->get_mat(ir + rshift, ropt, OpNames::XR);
                        if (!rskip[ir] && rmat != nullptr) {
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
                        [&c_compute, &c_partials, &cmat,
                         &vmat](const shared_ptr<TensorFunctions<S, FL>> &tf,
                                size_t i) {
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
                         rcnt](const shared_ptr<TensorFunctions<S, FL>> &tf,
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
            throw runtime_error("ParallelTensorFunctions::tensor_product_npdm_"
                                "fragment save on '" +
                                fn + "' failed.");
        if (compressed) {
            ofs << result->data->size();
            make_shared<FPCodec<FP>>()->write_array(
                ofs, (FP *)result->data->data(),
                result->data->size() * (sizeof(FL) / sizeof(FP)));
        } else
            result->write_array(ofs);
        if (!ofs.good())
            throw runtime_error("ParallelTensorFunctions::tensor_product_npdm_"
                                "fragment save on '" +
                                fn + "' failed.");
        ofs.close();
        return expectations;
    }
    vector<pair<shared_ptr<OpExpr<S>>, FL>>
    tensor_product_expectation(const vector<shared_ptr<OpExpr<S>>> &names,
                               const vector<shared_ptr<OpExpr<S>>> &exprs,
                               const shared_ptr<OperatorTensor<S, FL>> &lopt,
                               const shared_ptr<OperatorTensor<S, FL>> &ropt,
                               const shared_ptr<SparseMatrix<S, FL>> &cmat,
                               const shared_ptr<SparseMatrix<S, FL>> &vmat,
                               bool all_reduce) const override {
        vector<pair<shared_ptr<OpExpr<S>>, FL>> expectations(names.size());
        vector<FL> results(names.size(), 0);
        assert(names.size() == exprs.size());
        S ket_dq = cmat->info->delta_quantum;
        S bra_dq = vmat->info->delta_quantum;
        rule->set_partition(ParallelRulePartitionTypes::Middle);
        vector<shared_ptr<OpExpr<S>>> pnames;
        vector<shared_ptr<OpExpr<S>>> pexprs;
        vector<size_t> pidxs;
        for (size_t k = 0; k < exprs.size(); k++) {
            expectations[k] = make_pair(names[k], 0.0);
            S opdq = dynamic_pointer_cast<OpElement<S, FL>>(names[k])->q_label;
            if (opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                continue;
            shared_ptr<OpExpr<S>> expr = exprs[k];
            if (!rule->number(names[k]) || rule->own(names[k]))
                pidxs.push_back(k);
        }
        pnames.reserve(pidxs.size());
        pexprs.reserve(pidxs.size());
        for (size_t kk = 0; kk < pidxs.size(); kk++) {
            pnames.push_back(names[pidxs[kk]]);
            shared_ptr<OpExpr<S>> expr = exprs[pidxs[kk]];
            if (expr->get_type() == OpTypes::ExprRef) {
                shared_ptr<OpExprRef<S>> op =
                    dynamic_pointer_cast<OpExprRef<S>>(expr);
                expr = dynamic_pointer_cast<OpExprRef<S>>(expr)->op;
            }
            pexprs.push_back(expr);
        }
        vector<pair<shared_ptr<OpExpr<S>>, FL>> pexpectations =
            TensorFunctions<S, FL>::tensor_product_expectation(
                pnames, pexprs, lopt, ropt, cmat, vmat, all_reduce);
        for (size_t kk = 0; kk < pidxs.size(); kk++)
            results[pidxs[kk]] = pexpectations[kk].second;
        // !all_reduce is the general npdm (with symbol case)
        if (all_reduce)
            rule->comm->allreduce_sum(results.data(), results.size());
        else
            rule->comm->barrier();
        for (size_t i = 0; i < names.size(); i++)
            expectations[i].second = results[i];
        return expectations;
    }
    // vmat = expr x cmat
    void tensor_product_multiply(const shared_ptr<OpExpr<S>> &expr,
                                 const shared_ptr<OperatorTensor<S, FL>> &lopt,
                                 const shared_ptr<OperatorTensor<S, FL>> &ropt,
                                 const shared_ptr<SparseMatrix<S, FL>> &cmat,
                                 const shared_ptr<SparseMatrix<S, FL>> &vmat,
                                 S opdq, bool all_reduce) const override {
        if (expr->get_type() == OpTypes::ExprRef) {
            shared_ptr<OpExprRef<S>> op =
                dynamic_pointer_cast<OpExprRef<S>>(expr);
            TensorFunctions<S, FL>::tensor_product_multiply(
                op->op, lopt, ropt, cmat, vmat, opdq, false);
            if (all_reduce)
                rule->comm->allreduce_sum(vmat);
        } else
            TensorFunctions<S, FL>::tensor_product_multiply(
                expr, lopt, ropt, cmat, vmat, opdq, false);
    }
    // mat = diag(expr)
    void tensor_product_diagonal(const shared_ptr<OpExpr<S>> &expr,
                                 const shared_ptr<OperatorTensor<S, FL>> &lopt,
                                 const shared_ptr<OperatorTensor<S, FL>> &ropt,
                                 const shared_ptr<SparseMatrix<S, FL>> &mat,
                                 S opdq) const override {
        if (expr->get_type() == OpTypes::ExprRef) {
            shared_ptr<OpExprRef<S>> op =
                dynamic_pointer_cast<OpExprRef<S>>(expr);
            TensorFunctions<S, FL>::tensor_product_diagonal(op->op, lopt, ropt,
                                                            mat, opdq);
            rule->comm->allreduce_sum(mat);
        } else
            TensorFunctions<S, FL>::tensor_product_diagonal(expr, lopt, ropt,
                                                            mat, opdq);
    }
    // c = mpst_bra x a x mpst_ket
    void left_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                     const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                     const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                     shared_ptr<OperatorTensor<S, FL>> &c) const override {
        for (size_t i = 0; i < a->lmat->data.size(); i++)
            if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->lmat->data[i]);
                bool req = rule->available(pa);
                if (rule->get_parallel_type() & ParallelTypes::NewScheme)
                    req = req || this->rule->partial(pa);
                if (req) {
                    assert(c->ops.at(pa)->data == nullptr);
                    c->ops.at(pa)->allocate(c->ops.at(pa)->info);
                }
            }
        bool repeat = true,
             no_repeat = !(rule->comm_type & ParallelCommTypes::NonBlocking);
        auto f = [&a, &c, &mpst_bra, &mpst_ket, this, &repeat, &no_repeat](
                     const shared_ptr<TensorFunctions<S, FL>> &tf, size_t i) {
            if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->lmat->data[i]);
                bool req = true;
                if (this->rule->get_parallel_type() & ParallelTypes::NewScheme)
                    req = this->rule->own(pa) || this->rule->repeat(pa) ||
                          this->rule->partial(pa);
                else
                    req = this->rule->own(pa) &&
                          ((repeat && this->rule->repeat(pa)) ||
                           (no_repeat && !this->rule->repeat(pa)));
                if (req)
                    tf->opf->tensor_rotate(a->ops.at(pa), c->ops.at(pa),
                                           mpst_bra, mpst_ket, false);
            }
        };
        parallel_for(a->lmat->data.size(), f);
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
        if (rule->get_parallel_type() & ParallelTypes::NewScheme)
            return;
        for (size_t i = 0; i < a->lmat->data.size(); i++)
            if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->lmat->data[i]);
                if (rule->repeat(pa)) {
                    if (!(rule->comm_type & ParallelCommTypes::NonBlocking))
                        rule->comm->broadcast(c->ops.at(pa), rule->owner(pa));
                    else
                        rule->comm->ibroadcast(c->ops.at(pa), rule->owner(pa));
                }
            }
        if (rule->comm_type & ParallelCommTypes::NonBlocking) {
            repeat = false, no_repeat = true;
            parallel_for(a->lmat->data.size(), f);
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
            rule->comm->waitall();
        }
    }
    // c = mpst_bra x a x mpst_ket
    void right_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                      const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                      const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                      shared_ptr<OperatorTensor<S, FL>> &c) const override {
        for (size_t i = 0; i < a->rmat->data.size(); i++)
            if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->rmat->data[i]);
                bool req = rule->available(pa);
                if (rule->get_parallel_type() & ParallelTypes::NewScheme)
                    req = req || this->rule->partial(pa);
                if (req) {
                    assert(c->ops.at(pa)->data == nullptr);
                    c->ops.at(pa)->allocate(c->ops.at(pa)->info);
                }
            }
        bool repeat = true,
             no_repeat = !(rule->comm_type & ParallelCommTypes::NonBlocking);
        auto f = [&a, &c, &mpst_bra, &mpst_ket, this, &repeat, &no_repeat](
                     const shared_ptr<TensorFunctions<S, FL>> &tf, size_t i) {
            if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->rmat->data[i]);
                bool req = true;
                if (this->rule->get_parallel_type() & ParallelTypes::NewScheme)
                    req = this->rule->own(pa) || this->rule->repeat(pa) ||
                          this->rule->partial(pa);
                else
                    req = this->rule->own(pa) &&
                          ((repeat && this->rule->repeat(pa)) ||
                           (no_repeat && !this->rule->repeat(pa)));
                if (req)
                    tf->opf->tensor_rotate(a->ops.at(pa), c->ops.at(pa),
                                           mpst_bra, mpst_ket, true);
            }
        };
        parallel_for(a->rmat->data.size(), f);
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
        if (rule->get_parallel_type() & ParallelTypes::NewScheme)
            return;
        for (size_t i = 0; i < a->rmat->data.size(); i++)
            if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->rmat->data[i]);
                if (rule->repeat(pa)) {
                    if (!(rule->comm_type & ParallelCommTypes::NonBlocking))
                        rule->comm->broadcast(c->ops.at(pa), rule->owner(pa));
                    else
                        rule->comm->ibroadcast(c->ops.at(pa), rule->owner(pa));
                }
            }
        if (rule->comm_type & ParallelCommTypes::NonBlocking) {
            repeat = false, no_repeat = true;
            parallel_for(a->rmat->data.size(), f);
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
            rule->comm->waitall();
        }
    }
    void intermediates(const shared_ptr<Symbolic<S>> &names,
                       const shared_ptr<Symbolic<S>> &exprs,
                       const shared_ptr<OperatorTensor<S, FL>> &a,
                       bool left) const override {
        auto f = [&names, &a, left,
                  this](const vector<shared_ptr<OpExpr<S>>> &local_exprs) {
            shared_ptr<Symbolic<S>> ex = names->copy();
            ex->data = local_exprs;
            this->TensorFunctions<S, FL>::intermediates(names, ex, a, left);
        };
        vector<shared_ptr<SparseMatrix<S, FL>>> mats;
        rule->distributed_apply(f, names->data, exprs->data, mats);
    }
    // Numerical transform from normal operators
    // to complementary operators near the middle site
    void
    numerical_transform(const shared_ptr<OperatorTensor<S, FL>> &a,
                        const shared_ptr<Symbolic<S>> &names,
                        const shared_ptr<Symbolic<S>> &exprs) const override {
        assert(names->data.size() == exprs->data.size());
        assert((a->lmat == nullptr) ^ (a->rmat == nullptr));
        if (a->lmat == nullptr)
            a->rmat = names;
        else
            a->lmat = names;
        int comm_size = rule->get_parallel_type() & ParallelTypes::Simple
                            ? 1
                            : rule->comm->size;
        vector<vector<
            pair<shared_ptr<SparseMatrix<S, FL>>, shared_ptr<OpSum<S, FL>>>>>
            trs(comm_size);
        const shared_ptr<OpSum<S, FL>> zero =
            make_shared<OpSum<S, FL>>(vector<shared_ptr<OpProduct<S, FL>>>());
        for (int ip = 0; ip < comm_size; ip++)
            trs[ip].reserve(names->data.size() / comm_size);
        int maxi = 0;
        for (size_t k = 0; k < names->data.size(); k++) {
            if (exprs->data[k]->get_type() == OpTypes::Zero)
                continue;
            shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
            shared_ptr<OpExpr<S>> expr =
                exprs->data[k] *
                ((FL)1.0 /
                 dynamic_pointer_cast<OpElement<S, FL>>(names->data[k])
                     ->factor);
            shared_ptr<OpExprRef<S>> lexpr;
            int ip = rule->owner(nop);
            if (expr->get_type() != OpTypes::ExprRef)
                lexpr = rule->localize_expr(expr, ip);
            else
                lexpr = dynamic_pointer_cast<OpExprRef<S>>(expr);
            if (rule->get_parallel_type() & ParallelTypes::Simple)
                ip = 0;
            expr = lexpr->op;
            assert(a->ops.count(nop) != 0);
            // can be normal operator or zero complementary operator
            // if zero complementary operator, set factor to zero, skip
            // allocation
            // in non-parallel case, all normal/compelementary operators
            // are allocated
            if (lexpr->orig->get_type() == OpTypes::Zero) {
                if (a->ops.at(nop)->data == nullptr)
                    a->ops.at(nop)->factor = 0;
                continue;
            }
            shared_ptr<SparseMatrix<S, FL>> anop = a->ops.at(nop);
            switch (expr->get_type()) {
            case OpTypes::Sum:
                trs[ip].push_back(
                    make_pair(anop, dynamic_pointer_cast<OpSum<S, FL>>(expr)));
                maxi = max(maxi, (int)dynamic_pointer_cast<OpSum<S, FL>>(expr)
                                     ->strings.size());
                break;
            case OpTypes::Zero:
                trs[ip].push_back(make_pair(anop, zero));
                break;
            default:
                assert(false);
                break;
            }
        }
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<VectorAllocator<FP>> d_alloc_local =
            make_shared<VectorAllocator<FP>>();
        for (int ip = 0; ip < comm_size; ip++) {
            for (size_t k = 0; k < trs[ip].size(); k++) {
                assert(trs[ip][k].first->data == nullptr);
                if (ip != rule->comm->rank &&
                    !(rule->get_parallel_type() & ParallelTypes::Simple))
                    trs[ip][k].first->alloc = d_alloc;
                else
                    trs[ip][k].first->alloc = d_alloc_local;
                trs[ip][k].first->allocate(trs[ip][k].first->info);
            }
            parallel_for(
                trs[ip].size(),
                [&trs, &a, ip](const shared_ptr<TensorFunctions<S, FL>> &tf,
                               size_t i) {
                    shared_ptr<OpSum<S, FL>> op = trs[ip][i].second;
                    for (size_t j = 0; j < op->strings.size(); j++) {
                        shared_ptr<OpElement<S, FL>> nexpr =
                            op->strings[j]->get_op();
                        assert(a->ops.count(nexpr) != 0);
                        tf->opf->iadd(trs[ip][i].first, a->ops.at(nexpr),
                                      op->strings[j]->factor,
                                      op->strings[j]->conj != 0);
                        if (tf->opf->seq->mode & SeqTypes::Simple)
                            tf->opf->seq->simple_perform();
                    }
                });
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
            if (!(rule->get_parallel_type() & ParallelTypes::Simple)) {
                for (size_t k = 0; k < names->data.size(); k++) {
                    shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
                    if (exprs->data[k]->get_type() == OpTypes::Zero)
                        continue;
                    if (rule->owner(nop) != ip)
                        continue;
                    shared_ptr<OpExpr<S>> expr = exprs->data[k];
                    if (rule->get_parallel_type() & ParallelTypes::NewScheme)
                        assert(expr->get_type() == OpTypes::ExprRef);
                    shared_ptr<OpExprRef<S>> lexpr;
                    if (expr->get_type() != OpTypes::ExprRef)
                        lexpr = rule->localize_expr(expr, rule->owner(nop));
                    else
                        lexpr = dynamic_pointer_cast<OpExprRef<S>>(expr);
                    if (lexpr->orig->get_type() == OpTypes::Zero)
                        continue;
                    rule->comm->reduce_sum(a->ops.at(nop), rule->owner(nop));
                }
                if (ip != rule->comm->rank) {
                    for (int k = (int)trs[ip].size() - 1; k >= 0; k--)
                        trs[ip][k].first->deallocate();
                }
            }
        }
    }
    // delayed left and right block contraction
    shared_ptr<DelayedOperatorTensor<S, FL>>
    delayed_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                     const shared_ptr<OperatorTensor<S, FL>> &b,
                     const shared_ptr<OpExpr<S>> &op,
                     OpNamesSet delayed) const override {
        shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
            make_shared<DelayedOperatorTensor<S, FL>>();
        dopt->lopt = a, dopt->ropt = b;
        dopt->dops.push_back(op);
        assert(a->lmat->data.size() == b->rmat->data.size());
        shared_ptr<Symbolic<S>> exprs = a->lmat * b->rmat;
        assert(exprs->data.size() == 1);
        bool use_orig = !(rule->get_parallel_type() & ParallelTypes::NewScheme);
        if (a->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(a),
                true, delayed, use_orig);
        else if (b->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(b),
                false, delayed, use_orig);
        else
            dopt->mat = exprs;
        if (use_orig) {
            bool dleft = a->get_type() == OperatorTensorTypes::Delayed;
            dopt->mat->data[0] = rule->localize_expr(
                dopt->mat->data[0], rule->owner(dopt->dops[0]), dleft);
        }
        return dopt;
    }
    // delayed left and right block contraction
    // using the pre-computed exprs
    shared_ptr<DelayedOperatorTensor<S, FL>>
    delayed_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                     const shared_ptr<OperatorTensor<S, FL>> &b,
                     const shared_ptr<Symbolic<S>> &ops,
                     const shared_ptr<Symbolic<S>> &exprs,
                     OpNamesSet delayed) const override {
        shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
            make_shared<DelayedOperatorTensor<S, FL>>();
        dopt->lopt = a, dopt->ropt = b;
        dopt->dops = ops->data;
        bool use_orig = !(rule->get_parallel_type() & ParallelTypes::NewScheme);
        if (a->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(a),
                true, delayed, use_orig);
        else if (b->get_type() == OperatorTensorTypes::Delayed)
            dopt->mat = substitute_delayed_exprs(
                exprs, dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(b),
                false, delayed, use_orig);
        else
            dopt->mat = exprs;
        if (use_orig) {
            bool dleft = a->get_type() == OperatorTensorTypes::Delayed;
            for (size_t i = 0; i < dopt->mat->data.size(); i++)
                if (dopt->mat->data[i]->get_type() != OpTypes::ExprRef)
                    dopt->mat->data[i] = rule->localize_expr(
                        dopt->mat->data[i], rule->owner(dopt->dops[i]), dleft);
        }
        return dopt;
    }
    // c = a x b (dot)
    void left_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                       const shared_ptr<OperatorTensor<S, FL>> &b,
                       shared_ptr<OperatorTensor<S, FL>> &c,
                       const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                       OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr)
            left_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == c->lmat->data.size());
            vector<shared_ptr<SparseMatrix<S, FL>>> mats(exprs->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpExpr<S>> op = abs_value(c->lmat->data[i]);
                if (!delayed(
                        dynamic_pointer_cast<OpElement<S, FL>>(op)->name)) {
                    if (!frame_<FP>()->use_main_stack) {
                        // skip cached part
                        if (c->ops.at(op)->alloc != nullptr)
                            continue;
                        c->ops.at(op)->alloc =
                            make_shared<VectorAllocator<FP>>();
                    }
                    mats[i] = c->ops.at(op);
                }
            }
            auto f = [&a, &b, &mats,
                      this](const vector<shared_ptr<OpExpr<S>>> &local_exprs) {
                for (size_t i = 0; i < local_exprs.size(); i++)
                    if (frame_<FP>()->use_main_stack &&
                        local_exprs[i] != nullptr) {
                        assert(mats[i]->data == nullptr);
                        mats[i]->allocate(mats[i]->info);
                    }
                this->parallel_for(
                    local_exprs.size(),
                    [&a, &b, &mats,
                     &local_exprs](const shared_ptr<TensorFunctions<S, FL>> &tf,
                                   size_t i) {
                        if (local_exprs[i] != nullptr) {
                            if (!frame_<FP>()->use_main_stack)
                                mats[i]->allocate(mats[i]->info);
                            tf->tensor_product(local_exprs[i], a->ops, b->ops,
                                               mats[i]);
                        }
                    });
                if (this->opf->seq->mode == SeqTypes::Auto)
                    this->opf->seq->auto_perform();
            };
            rule->distributed_apply(f, c->lmat->data, exprs->data, mats);
        }
    }
    // c = b (dot) x a
    void right_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                        const shared_ptr<OperatorTensor<S, FL>> &b,
                        shared_ptr<OperatorTensor<S, FL>> &c,
                        const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                        OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr)
            right_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == c->rmat->data.size());
            vector<shared_ptr<SparseMatrix<S, FL>>> mats(exprs->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpExpr<S>> op = abs_value(c->rmat->data[i]);
                if (!delayed(
                        dynamic_pointer_cast<OpElement<S, FL>>(op)->name)) {
                    if (!frame_<FP>()->use_main_stack) {
                        // skip cached part
                        if (c->ops.at(op)->alloc != nullptr)
                            continue;
                        c->ops.at(op)->alloc =
                            make_shared<VectorAllocator<FP>>();
                    }
                    mats[i] = c->ops.at(op);
                }
            }
            auto f = [&a, &b, &mats,
                      this](const vector<shared_ptr<OpExpr<S>>> &local_exprs) {
                for (size_t i = 0; i < local_exprs.size(); i++)
                    if (frame_<FP>()->use_main_stack &&
                        local_exprs[i] != nullptr) {
                        assert(mats[i]->data == nullptr);
                        mats[i]->allocate(mats[i]->info);
                    }
                this->parallel_for(
                    local_exprs.size(),
                    [&a, &b, &mats,
                     &local_exprs](const shared_ptr<TensorFunctions<S, FL>> &tf,
                                   size_t i) {
                        if (local_exprs[i] != nullptr) {
                            if (!frame_<FP>()->use_main_stack)
                                mats[i]->allocate(mats[i]->info);
                            tf->tensor_product(local_exprs[i], b->ops, a->ops,
                                               mats[i]);
                        }
                    });
                if (this->opf->seq->mode == SeqTypes::Auto)
                    this->opf->seq->auto_perform();
            };
            rule->distributed_apply(f, c->rmat->data, exprs->data, mats);
        }
    }
    // c = mpst_bra x [ a x b (dot) ] x mpst_ket
    void
    left_contract_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                         const shared_ptr<OperatorTensor<S, FL>> &b,
                         const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                         const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                         shared_ptr<OperatorTensor<S, FL>> &ab,
                         shared_ptr<OperatorTensor<S, FL>> &c,
                         const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                         OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr) {
            left_assign(b, ab);
            left_rotate(ab, mpst_bra, mpst_ket, c);
        } else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == c->lmat->data.size());
            assert(rule->get_parallel_type() & ParallelTypes::NewScheme);
            assert(opf->seq->mode != SeqTypes::Auto);
            vector<shared_ptr<SparseMatrix<S, FL>>> mats(exprs->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpExpr<S>> op = abs_value(ab->lmat->data[i]);
                if (!delayed(
                        dynamic_pointer_cast<OpElement<S, FL>>(op)->name)) {
                    if (!frame_<FP>()->use_main_stack) {
                        // skip cached part
                        if (ab->ops.at(op)->alloc != nullptr)
                            continue;
                        ab->ops.at(op)->alloc =
                            make_shared<VectorAllocator<FP>>();
                    }
                    mats[i] = ab->ops.at(op);
                }
            }
            vector<shared_ptr<OpExpr<S>>> local_exprs;
            auto f =
                [&local_exprs](const vector<shared_ptr<OpExpr<S>>> &exprs) {
                    local_exprs = exprs;
                };
            rule->distributed_apply(f, ab->lmat->data, exprs->data, mats);
            if (frame_<FP>()->use_main_stack)
                for (size_t i = 0; i < local_exprs.size(); i++)
                    if (local_exprs[i] != nullptr) {
                        assert(mats[i]->data == nullptr);
                        mats[i]->allocate(mats[i]->info);
                    }
            for (size_t i = 0; i < ab->lmat->data.size(); i++)
                if (ab->lmat->data[i]->get_type() != OpTypes::Zero) {
                    auto pa = abs_value(ab->lmat->data[i]);
                    bool req = rule->available(pa);
                    if (rule->get_parallel_type() & ParallelTypes::NewScheme)
                        req = req || rule->partial(pa);
                    if (req) {
                        assert(c->ops.at(pa)->data == nullptr);
                        c->ops.at(pa)->allocate(c->ops.at(pa)->info);
                    }
                }
            bool repeat = true, no_repeat = !(rule->comm_type &
                                              ParallelCommTypes::NonBlocking);
            auto g = [&a, &b, &ab, &local_exprs, &mats, &c, &mpst_bra,
                      &mpst_ket, this, &repeat,
                      &no_repeat](const shared_ptr<TensorFunctions<S, FL>> &tf,
                                  size_t i) {
                if (ab->lmat->data[i]->get_type() != OpTypes::Zero) {
                    auto pa = abs_value(ab->lmat->data[i]);
                    bool req = true;
                    if (this->rule->get_parallel_type() &
                        ParallelTypes::NewScheme)
                        req = this->rule->own(pa) || this->rule->repeat(pa) ||
                              this->rule->partial(pa);
                    else
                        req = this->rule->own(pa) &&
                              ((repeat && this->rule->repeat(pa)) ||
                               (no_repeat && !this->rule->repeat(pa)));
                    if (req) {
                        assert(local_exprs[i] != nullptr);
                        if (!frame_<FP>()->use_main_stack)
                            mats[i]->allocate(mats[i]->info);
                        tf->tensor_product(local_exprs[i], a->ops, b->ops,
                                           mats[i]);
                        tf->opf->tensor_rotate(mats[i], c->ops.at(pa), mpst_bra,
                                               mpst_ket, false);
                        if (!frame_<FP>()->use_main_stack)
                            mats[i]->deallocate();
                    }
                }
            };
            parallel_for(c->lmat->data.size(), g);
        }
    }
    // c = mpst_bra x [ b (dot) x a ] x mpst_ket
    void
    right_contract_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                          const shared_ptr<OperatorTensor<S, FL>> &b,
                          const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                          const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                          shared_ptr<OperatorTensor<S, FL>> &ab,
                          shared_ptr<OperatorTensor<S, FL>> &c,
                          const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                          OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr) {
            right_assign(b, ab);
            right_rotate(ab, mpst_bra, mpst_ket, c);
        } else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == c->rmat->data.size());
            assert(rule->get_parallel_type() & ParallelTypes::NewScheme);
            assert(opf->seq->mode != SeqTypes::Auto);
            vector<shared_ptr<SparseMatrix<S, FL>>> mats(exprs->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpExpr<S>> op = abs_value(ab->rmat->data[i]);
                if (!delayed(
                        dynamic_pointer_cast<OpElement<S, FL>>(op)->name)) {
                    if (!frame_<FP>()->use_main_stack) {
                        // skip cached part
                        if (ab->ops.at(op)->alloc != nullptr)
                            continue;
                        ab->ops.at(op)->alloc =
                            make_shared<VectorAllocator<FP>>();
                    }
                    mats[i] = ab->ops.at(op);
                }
            }
            vector<shared_ptr<OpExpr<S>>> local_exprs;
            auto f =
                [&local_exprs](const vector<shared_ptr<OpExpr<S>>> &exprs) {
                    local_exprs = exprs;
                };
            rule->distributed_apply(f, ab->rmat->data, exprs->data, mats);
            if (frame_<FP>()->use_main_stack)
                for (size_t i = 0; i < local_exprs.size(); i++)
                    if (local_exprs[i] != nullptr) {
                        assert(mats[i]->data == nullptr);
                        mats[i]->allocate(mats[i]->info);
                    }
            for (size_t i = 0; i < ab->rmat->data.size(); i++)
                if (ab->rmat->data[i]->get_type() != OpTypes::Zero) {
                    auto pa = abs_value(ab->rmat->data[i]);
                    bool req = rule->available(pa);
                    if (rule->get_parallel_type() & ParallelTypes::NewScheme)
                        req = req || rule->partial(pa);
                    if (req) {
                        assert(c->ops.at(pa)->data == nullptr);
                        c->ops.at(pa)->allocate(c->ops.at(pa)->info);
                    }
                }
            bool repeat = true, no_repeat = !(rule->comm_type &
                                              ParallelCommTypes::NonBlocking);
            auto g = [&a, &b, &ab, &local_exprs, &mats, &c, &mpst_bra,
                      &mpst_ket, this, &repeat,
                      &no_repeat](const shared_ptr<TensorFunctions<S, FL>> &tf,
                                  size_t i) {
                if (ab->rmat->data[i]->get_type() != OpTypes::Zero) {
                    auto pa = abs_value(ab->rmat->data[i]);
                    bool req = true;
                    if (this->rule->get_parallel_type() &
                        ParallelTypes::NewScheme)
                        req = this->rule->own(pa) || this->rule->repeat(pa) ||
                              this->rule->partial(pa);
                    else
                        req = this->rule->own(pa) &&
                              ((repeat && this->rule->repeat(pa)) ||
                               (no_repeat && !this->rule->repeat(pa)));
                    if (req) {
                        assert(local_exprs[i] != nullptr);
                        if (!frame_<FP>()->use_main_stack)
                            mats[i]->allocate(mats[i]->info);
                        tf->tensor_product(local_exprs[i], b->ops, a->ops,
                                           mats[i]);
                        tf->opf->tensor_rotate(mats[i], c->ops.at(pa), mpst_bra,
                                               mpst_ket, true);
                        if (!frame_<FP>()->use_main_stack)
                            mats[i]->deallocate();
                    }
                }
            };
            parallel_for(c->rmat->data.size(), g);
        }
    }
};

} // namespace block2
