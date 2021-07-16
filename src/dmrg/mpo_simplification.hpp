
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

#include "mpo.hpp"
#include "../core/rule.hpp"
#include "../core/threading.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#define TINY (1E-20)

using namespace std;

namespace block2 {

// Simplify MPO expression according to symmetry rules
template <typename S> struct SimplifiedMPO : MPO<S> {
    // Original MPO
    shared_ptr<MPO<S>> prim_mpo;
    shared_ptr<Rule<S>> rule;
    // Collect terms means that sum of products will be changed to
    // product of one symbol times a sum of symbols
    // (if there are common factors)
    // A x B + A x C + A x D => A x (B + C + D)
    bool collect_terms, use_intermediate;
    OpNamesSet intermediate_ops;
    SimplifiedMPO(const shared_ptr<MPO<S>> &mpo,
                  const shared_ptr<Rule<S>> &rule, bool collect_terms = true,
                  bool use_intermediate = false,
                  OpNamesSet intermediate_ops = OpNamesSet::all_ops())
        : prim_mpo(mpo), rule(rule), MPO<S>(mpo->n_sites),
          collect_terms(collect_terms), use_intermediate(use_intermediate),
          intermediate_ops(intermediate_ops) {
        if (!collect_terms)
            use_intermediate = false;
        static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        MPO<S>::const_e = mpo->const_e;
        MPO<S>::tensors = mpo->tensors;
        MPO<S>::basis = mpo->basis;
        MPO<S>::op = mpo->op;
        MPO<S>::schemer = mpo->schemer;
        MPO<S>::tf = mpo->tf;
        MPO<S>::site_op_infos = mpo->site_op_infos;
        MPO<S>::left_operator_names = mpo->left_operator_names;
        MPO<S>::sparse_form = mpo->sparse_form;
        for (auto &x : MPO<S>::left_operator_names)
            x = x->copy();
        MPO<S>::right_operator_names = mpo->right_operator_names;
        for (auto &x : MPO<S>::right_operator_names)
            x = x->copy();
        MPO<S>::left_operator_exprs.resize(MPO<S>::n_sites);
        MPO<S>::right_operator_exprs.resize(MPO<S>::n_sites);
        // for comp operators created in the middle site,
        // if all integrals related to the comp operators are zero,
        // label this comp operator as zero
        if (MPO<S>::schemer != nullptr) {
            MPO<S>::schemer = mpo->schemer->copy();
            int i = MPO<S>::schemer->left_trans_site;
            for (size_t j = 0;
                 j < MPO<S>::schemer->left_new_operator_names->data.size();
                 j++) {
                if (j < MPO<S>::left_operator_names[i]->data.size() &&
                    MPO<S>::left_operator_names[i]->data[j] ==
                        MPO<S>::schemer->left_new_operator_names->data[j])
                    continue;
                else if (MPO<S>::schemer->left_new_operator_exprs->data[j]
                             ->get_type() == OpTypes::Zero)
                    MPO<S>::schemer->left_new_operator_names->data[j] =
                        MPO<S>::schemer->left_new_operator_exprs->data[j];
            }
            i = MPO<S>::schemer->right_trans_site;
            for (size_t j = 0;
                 j < MPO<S>::schemer->right_new_operator_names->data.size();
                 j++) {
                if (j < MPO<S>::right_operator_names[i]->data.size() &&
                    MPO<S>::right_operator_names[i]->data[j] ==
                        MPO<S>::schemer->right_new_operator_names->data[j])
                    continue;
                else if (MPO<S>::schemer->right_new_operator_exprs->data[j]
                             ->get_type() == OpTypes::Zero)
                    MPO<S>::schemer->right_new_operator_names->data[j] =
                        MPO<S>::schemer->right_new_operator_exprs->data[j];
            }
        }
        // construct blocking formulas by contration of op name (vector) and mpo
        // matrix; if left/right trans, by contration of new (comp) op name
        // (vector) and mpo matrix
        // if contracted expr is zero, label the blocked operator as zero
        int ntg = threading->activate_global();
        // left blocking
        for (int i = 0; i < MPO<S>::n_sites; i++) {
            if (i == 0) {
                MPO<S>::tensors[i] = MPO<S>::tensors[i]->copy();
                if (MPO<S>::tensors[i]->lmat == MPO<S>::tensors[i]->rmat)
                    MPO<S>::tensors[i]->lmat = MPO<S>::tensors[i]->rmat =
                        MPO<S>::tensors[i]->rmat->copy();
                else
                    MPO<S>::tensors[i]->lmat = MPO<S>::tensors[i]->lmat->copy();
                MPO<S>::left_operator_exprs[i] = MPO<S>::tensors[i]->lmat;
            } else if (MPO<S>::schemer == nullptr ||
                       i - 1 != MPO<S>::schemer->left_trans_site)
                MPO<S>::left_operator_exprs[i] =
                    MPO<S>::left_operator_names[i - 1] *
                    MPO<S>::tensors[i]->lmat;
            else
                MPO<S>::left_operator_exprs[i] =
                    (shared_ptr<Symbolic<S>>)
                        MPO<S>::schemer->left_new_operator_names *
                    MPO<S>::tensors[i]->lmat;
            if (MPO<S>::schemer != nullptr &&
                i == MPO<S>::schemer->left_trans_site) {
                for (size_t j = 0;
                     j < MPO<S>::left_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::left_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero) {
                        if (j < MPO<S>::schemer->left_new_operator_names->data
                                    .size() &&
                            MPO<S>::left_operator_names[i]->data[j] ==
                                MPO<S>::schemer->left_new_operator_names
                                    ->data[j])
                            MPO<S>::schemer->left_new_operator_names->data[j] =
                                MPO<S>::left_operator_exprs[i]->data[j];
                        MPO<S>::left_operator_names[i]->data[j] =
                            MPO<S>::left_operator_exprs[i]->data[j];
                    }
            } else {
                for (size_t j = 0;
                     j < MPO<S>::left_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::left_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero)
                        MPO<S>::left_operator_names[i]->data[j] =
                            MPO<S>::left_operator_exprs[i]->data[j];
            }
        }
        // right blocking
        for (int i = MPO<S>::n_sites - 1; i >= 0; i--) {
            if (i == MPO<S>::n_sites - 1) {
                MPO<S>::tensors[i] = MPO<S>::tensors[i]->copy();
                if (MPO<S>::tensors[i]->lmat == MPO<S>::tensors[i]->rmat)
                    MPO<S>::tensors[i]->rmat = MPO<S>::tensors[i]->lmat =
                        MPO<S>::tensors[i]->lmat->copy();
                else
                    MPO<S>::tensors[i]->rmat = MPO<S>::tensors[i]->rmat->copy();
                MPO<S>::right_operator_exprs[i] = MPO<S>::tensors[i]->rmat;
            } else if (MPO<S>::schemer == nullptr ||
                       i + 1 != MPO<S>::schemer->right_trans_site)
                MPO<S>::right_operator_exprs[i] =
                    MPO<S>::tensors[i]->rmat *
                    MPO<S>::right_operator_names[i + 1];
            else
                MPO<S>::right_operator_exprs[i] =
                    MPO<S>::tensors[i]->rmat *
                    (shared_ptr<Symbolic<S>>)
                        MPO<S>::schemer->right_new_operator_names;
            if (MPO<S>::schemer != nullptr &&
                i == MPO<S>::schemer->right_trans_site) {
                for (size_t j = 0;
                     j < MPO<S>::right_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::right_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero) {
                        if (j < MPO<S>::schemer->right_new_operator_names->data
                                    .size() &&
                            MPO<S>::right_operator_names[i]->data[j] ==
                                MPO<S>::schemer->right_new_operator_names
                                    ->data[j])
                            MPO<S>::schemer->right_new_operator_names->data[j] =
                                MPO<S>::right_operator_exprs[i]->data[j];
                        MPO<S>::right_operator_names[i]->data[j] =
                            MPO<S>::right_operator_exprs[i]->data[j];
                    }
            } else {
                for (size_t j = 0;
                     j < MPO<S>::right_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::right_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero)
                        MPO<S>::right_operator_names[i]->data[j] =
                            MPO<S>::right_operator_exprs[i]->data[j];
            }
        }
        // construct super blocking contraction formula
        // first case is that the blocking formula is already given
        // for example in the npdm code
        if (mpo->middle_operator_exprs.size() != 0) {
            MPO<S>::middle_operator_names = mpo->middle_operator_names;
            MPO<S>::middle_operator_exprs = mpo->middle_operator_exprs;
            assert(MPO<S>::schemer == nullptr);
            // if some operators are erased in left/right operator names
            // they should not appear in middle operator exprs
            // and the expr should be zero
            for (size_t i = 0; i < MPO<S>::middle_operator_names.size(); i++) {
                set<shared_ptr<OpExpr<S>>, op_expr_less<S>> left_zero_ops,
                    right_zero_ops;
                for (size_t j = 0;
                     j < MPO<S>::left_operator_names[i]->data.size(); j++)
                    if (MPO<S>::left_operator_names[i]->data[j]->get_type() ==
                        OpTypes::Zero)
                        left_zero_ops.insert(
                            mpo->left_operator_names[i]->data[j]);
                for (size_t j = 0;
                     j < MPO<S>::right_operator_names[i + 1]->data.size(); j++)
                    if (MPO<S>::right_operator_names[i + 1]
                            ->data[j]
                            ->get_type() == OpTypes::Zero)
                        right_zero_ops.insert(
                            mpo->right_operator_names[i + 1]->data[j]);
                for (size_t j = 0;
                     j < MPO<S>::middle_operator_exprs[i]->data.size(); j++) {
                    shared_ptr<OpExpr<S>> &x =
                        MPO<S>::middle_operator_exprs[i]->data[j];
                    switch (x->get_type()) {
                    case OpTypes::Zero:
                        break;
                    case OpTypes::Prod:
                        if (left_zero_ops.count(
                                dynamic_pointer_cast<OpProduct<S>>(x)->a) ||
                            right_zero_ops.count(
                                dynamic_pointer_cast<OpProduct<S>>(x)->b))
                            MPO<S>::middle_operator_exprs[i]->data[j] = zero;
                        break;
                    case OpTypes::Sum:
                        for (auto &r :
                             dynamic_pointer_cast<OpSum<S>>(x)->strings)
                            if (left_zero_ops.count(
                                    dynamic_pointer_cast<OpProduct<S>>(r)->a) ||
                                right_zero_ops.count(
                                    dynamic_pointer_cast<OpProduct<S>>(r)->b)) {
                                MPO<S>::middle_operator_exprs[i]->data[j] =
                                    zero;
                                break;
                            }
                        break;
                    default:
                        assert(false);
                    }
                }
            }
        } else {
            vector<uint8_t> px[2];
            // figure out the mutual dependence of from right to left
            // px[.][j] is 1 if left operator is useful in next blocking
            for (int i = MPO<S>::n_sites - 1; i >= 0; i--) {
                if (i != MPO<S>::n_sites - 1) {
                    // if a left operator is not useful in next blocking
                    // and not useful in super block
                    // then it is labelled as not useful
                    // when it is not useful, set it to zero
                    if (MPO<S>::schemer == nullptr ||
                        i != MPO<S>::schemer->left_trans_site) {
                        for (size_t j = 0;
                             j < MPO<S>::left_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::right_operator_names[i + 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::left_operator_names[i]->data[j] =
                                    MPO<S>::right_operator_names[i + 1]
                                        ->data[j];
                            else if (MPO<S>::left_operator_names[i]
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else if (MPO<S>::schemer->right_trans_site -
                                   MPO<S>::schemer->left_trans_site >
                               1) {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->left_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::schemer->left_new_operator_names
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->left_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::right_operator_names[i + 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::schemer->left_new_operator_names
                                    ->data[j] =
                                    MPO<S>::right_operator_names[i + 1]
                                        ->data[j];
                            else if (MPO<S>::schemer->left_new_operator_names
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    }
                    if (MPO<S>::schemer != nullptr &&
                        i == MPO<S>::schemer->left_trans_site) {
                        px[!(i & 1)].resize(px[i & 1].size());
                        memcpy(px[!(i & 1)].data(), px[i & 1].data(),
                               sizeof(uint8_t) * px[!(i & 1)].size());
                        px[i & 1].resize(
                            MPO<S>::left_operator_names[i]->data.size());
                        memset(px[i & 1].data(), 0,
                               sizeof(uint8_t) * px[i & 1].size());
                        unordered_map<shared_ptr<OpExpr<S>>, int> mp;
                        mp.reserve(MPO<S>::left_operator_names[i]->data.size());
                        for (size_t j = 0;
                             j < MPO<S>::left_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::left_operator_names[i]
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                mp[abs_value(
                                    MPO<S>::left_operator_names[i]->data[j])] =
                                    (int)j;
                        shared_ptr<SymbolicRowVector<S>> &exprs =
                            MPO<S>::schemer->left_new_operator_exprs;
                        for (size_t j = 0; j < exprs->data.size(); j++) {
                            if (px[!(i & 1)][j] &&
                                j < MPO<S>::left_operator_names[i]
                                        ->data.size() &&
                                MPO<S>::left_operator_names[i]->data[j] ==
                                    MPO<S>::schemer->left_new_operator_names
                                        ->data[j])
                                px[i & 1][j] = 1;
                            else if (px[!(i & 1)][j] &&
                                     exprs->data[j]->get_type() !=
                                         OpTypes::Zero) {
                                shared_ptr<OpSum<S>> op = make_shared<OpSum<S>>(
                                    dynamic_pointer_cast<OpSum<S>>(
                                        exprs->data[j])
                                        ->strings);
                                for (size_t k = 0; k < op->strings.size();
                                     k++) {
                                    shared_ptr<OpExpr<S>> expr = abs_value(
                                        (shared_ptr<OpExpr<S>>)op->strings[k]
                                            ->a);
                                    if (mp.count(expr) == 0)
                                        op->strings[k] =
                                            make_shared<OpProduct<S>>(
                                                *op->strings[k] * 0.0);
                                    else
                                        px[i & 1][mp[expr]] = 1;
                                    assert(op->strings[k]->b == nullptr);
                                }
                                exprs->data[j] = op;
                            }
                        }
                        for (size_t j = 0;
                             j < MPO<S>::left_operator_names[i]->data.size();
                             j++)
                            if (!px[i & 1][j])
                                MPO<S>::left_operator_names[i]->data[j] = zero;
                    }
                }
                // at the beginning, all px values are zero
                // then, set the required op px = 1 based on mpo matrix
                if (i != 0) {
                    if (MPO<S>::schemer == nullptr ||
                        i - 1 != MPO<S>::schemer->left_trans_site)
                        px[!(i & 1)].resize(
                            MPO<S>::left_operator_names[i - 1]->data.size());
                    else
                        px[!(i & 1)].resize(
                            MPO<S>::schemer->left_new_operator_names->data
                                .size());
                    memset(&px[!(i & 1)][0], 0,
                           sizeof(uint8_t) * px[!(i & 1)].size());
                    if (MPO<S>::tensors[i]->lmat->get_type() == SymTypes::Mat) {
                        assert(px[i & 1].size() != 0);
                        shared_ptr<SymbolicMatrix<S>> mat =
                            dynamic_pointer_cast<SymbolicMatrix<S>>(
                                MPO<S>::tensors[i]->lmat);
                        for (size_t j = 0; j < mat->data.size(); j++)
                            if (px[i & 1][mat->indices[j].second] &&
                                mat->data[j]->get_type() != OpTypes::Zero)
                                px[!(i & 1)][mat->indices[j].first] = 1;
                    }
                }
            }
            // figure out the mutual dependence of from left to right
            for (int i = 0; i < MPO<S>::n_sites; i++) {
                if (i != 0) {
                    if (MPO<S>::schemer == nullptr ||
                        i != MPO<S>::schemer->right_trans_site) {
                        for (size_t j = 0;
                             j < MPO<S>::right_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::left_operator_names[i - 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::right_operator_names[i]->data[j] =
                                    MPO<S>::left_operator_names[i - 1]->data[j];
                            else if (MPO<S>::right_operator_names[i]
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else if (MPO<S>::schemer->right_trans_site -
                                   MPO<S>::schemer->left_trans_site >
                               1) {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->right_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::schemer->right_new_operator_names
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->right_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::left_operator_names[i - 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::schemer->right_new_operator_names
                                    ->data[j] =
                                    MPO<S>::left_operator_names[i - 1]->data[j];
                            else if (MPO<S>::schemer->right_new_operator_names
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    }
                    if (MPO<S>::schemer != nullptr &&
                        i == MPO<S>::schemer->right_trans_site) {
                        px[!(i & 1)].resize(px[i & 1].size());
                        memcpy(px[!(i & 1)].data(), px[i & 1].data(),
                               sizeof(uint8_t) * px[!(i & 1)].size());
                        px[i & 1].resize(
                            MPO<S>::right_operator_names[i]->data.size());
                        memset(px[i & 1].data(), 0,
                               sizeof(uint8_t) * px[i & 1].size());
                        unordered_map<shared_ptr<OpExpr<S>>, int> mp;
                        mp.reserve(
                            MPO<S>::right_operator_names[i]->data.size());
                        for (size_t j = 0;
                             j < MPO<S>::right_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::right_operator_names[i]
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                mp[abs_value(
                                    MPO<S>::right_operator_names[i]->data[j])] =
                                    (int)j;
                        shared_ptr<SymbolicColumnVector<S>> &exprs =
                            MPO<S>::schemer->right_new_operator_exprs;
                        for (size_t j = 0; j < exprs->data.size(); j++) {
                            if (px[!(i & 1)][j] &&
                                j < MPO<S>::right_operator_names[i]
                                        ->data.size() &&
                                MPO<S>::right_operator_names[i]->data[j] ==
                                    MPO<S>::schemer->right_new_operator_names
                                        ->data[j])
                                px[i & 1][j] = 1;
                            else if (px[!(i & 1)][j] &&
                                     exprs->data[j]->get_type() !=
                                         OpTypes::Zero) {
                                shared_ptr<OpSum<S>> op = make_shared<OpSum<S>>(
                                    dynamic_pointer_cast<OpSum<S>>(
                                        exprs->data[j])
                                        ->strings);
                                for (size_t k = 0; k < op->strings.size();
                                     k++) {
                                    shared_ptr<OpExpr<S>> expr = abs_value(
                                        (shared_ptr<OpExpr<S>>)op->strings[k]
                                            ->a);
                                    if (mp.count(expr) == 0)
                                        op->strings[k] =
                                            make_shared<OpProduct<S>>(
                                                *op->strings[k] * 0.0);
                                    else
                                        px[i & 1][mp[expr]] = 1;
                                    assert(op->strings[k]->b == nullptr);
                                }
                                exprs->data[j] = op;
                            }
                        }
                        for (size_t j = 0;
                             j < MPO<S>::right_operator_names[i]->data.size();
                             j++)
                            if (!px[i & 1][j])
                                MPO<S>::right_operator_names[i]->data[j] = zero;
                    }
                }
                if (i != MPO<S>::n_sites - 1) {
                    if (MPO<S>::schemer == nullptr ||
                        i + 1 != MPO<S>::schemer->right_trans_site)
                        px[!(i & 1)].resize(
                            MPO<S>::right_operator_names[i + 1]->data.size());
                    else
                        px[!(i & 1)].resize(
                            MPO<S>::schemer->right_new_operator_names->data
                                .size());
                    memset(px[!(i & 1)].data(), 0,
                           sizeof(uint8_t) * px[!(i & 1)].size());
                    if (MPO<S>::tensors[i]->rmat->get_type() == SymTypes::Mat) {
                        shared_ptr<SymbolicMatrix<S>> mat =
                            dynamic_pointer_cast<SymbolicMatrix<S>>(
                                MPO<S>::tensors[i]->rmat);
                        for (size_t j = 0; j < mat->data.size(); j++)
                            if (px[i & 1][mat->indices[j].first] &&
                                mat->data[j]->get_type() != OpTypes::Zero)
                                px[!(i & 1)][mat->indices[j].second] = 1;
                    }
                }
            }
            MPO<S>::middle_operator_names.resize(MPO<S>::n_sites - 1);
            MPO<S>::middle_operator_exprs.resize(MPO<S>::n_sites - 1);
            shared_ptr<SymbolicColumnVector<S>> mpo_op =
                make_shared<SymbolicColumnVector<S>>(1);
            (*mpo_op)[0] = mpo->op;
            for (int i = 0; i < MPO<S>::n_sites - 1; i++) {
                MPO<S>::middle_operator_names[i] = mpo_op;
                if (MPO<S>::schemer == nullptr ||
                    i != MPO<S>::schemer->left_trans_site ||
                    MPO<S>::schemer->right_trans_site -
                            MPO<S>::schemer->left_trans_site >
                        1)
                    MPO<S>::middle_operator_exprs[i] =
                        MPO<S>::left_operator_names[i] *
                        MPO<S>::right_operator_names[i + 1];
                else
                    MPO<S>::middle_operator_exprs[i] =
                        (shared_ptr<Symbolic<S>>)
                            MPO<S>::schemer->left_new_operator_names *
                        MPO<S>::right_operator_names[i + 1];
            }
        }
        simplify();
        threading->activate_normal();
    }
    shared_ptr<OpExpr<S>> simplify_expr(const shared_ptr<OpExpr<S>> &expr,
                                        S op = S(S::invalid)) {
        static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            assert(op->b != nullptr);
            shared_ptr<OpElementRef<S>> opl = rule->operator()(op->a);
            shared_ptr<OpElementRef<S>> opr = rule->operator()(op->b);
            shared_ptr<OpElement<S>> a = opl == nullptr ? op->a : opl->op;
            shared_ptr<OpElement<S>> b = opr == nullptr ? op->b : opr->op;
            uint8_t conj = (opl != nullptr && opl->trans) |
                           ((opr != nullptr && opr->trans) << 1);
            double factor = (opl != nullptr ? opl->factor : 1.0) *
                            (opr != nullptr ? opr->factor : 1.0) * op->factor;
            return make_shared<OpProduct<S>>(a, b, factor, conj);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> ops = dynamic_pointer_cast<OpSum<S>>(expr);
            unordered_map<shared_ptr<OpExpr<S>>,
                          vector<shared_ptr<OpProduct<S>>>>
                mp;
            mp.reserve(ops->strings.size());
            for (auto &x : ops->strings) {
                if (x->factor == 0)
                    continue;
                shared_ptr<OpElementRef<S>> opl = rule->operator()(x->a);
                shared_ptr<OpElementRef<S>> opr =
                    x->b == nullptr ? nullptr : rule->operator()(x->b);
                shared_ptr<OpElement<S>> a = opl == nullptr ? x->a : opl->op;
                shared_ptr<OpElement<S>> b = opr == nullptr ? x->b : opr->op;
                uint8_t conj = (opl != nullptr && opl->trans) |
                               ((opr != nullptr && opr->trans) << 1);
                double factor = (opl != nullptr ? opl->factor : 1.0) *
                                (opr != nullptr ? opr->factor : 1.0) *
                                x->factor;
                if (!mp.count(a))
                    mp[a] = vector<shared_ptr<OpProduct<S>>>();
                vector<shared_ptr<OpProduct<S>>> &px = mp.at(a);
                int g = -1;
                for (size_t k = 0; k < px.size(); k++)
                    if (px[k]->b == b && px[k]->conj == conj) {
                        g = (int)k;
                        break;
                    }
                if (g == -1)
                    px.push_back(make_shared<OpProduct<S>>(a, b, factor, conj));
                else {
                    px[g]->factor += factor;
                    if (abs(px[g]->factor) < TINY)
                        px.erase(px.begin() + g);
                }
            }
            vector<shared_ptr<OpProduct<S>>> terms;
            terms.reserve(mp.size());
            for (auto &r : mp)
                terms.insert(terms.end(), r.second.begin(), r.second.end());
            if (terms.size() == 0)
                return zero;
            else if (terms[0]->b == nullptr || terms.size() <= 2)
                return make_shared<OpSum<S>>(terms);
            else if (collect_terms && op != S(S::invalid)) {
                unordered_map<shared_ptr<OpExpr<S>>,
                              map<int, vector<shared_ptr<OpProduct<S>>>>>
                    mpa[2], mpb[2];
                for (int i = 0; i < 2; i++) {
                    mpa[i].reserve(terms.size());
                    mpb[i].reserve(terms.size());
                }
                for (auto &x : terms) {
                    assert(x->a != nullptr && x->b != nullptr);
                    if (x->conj & 1)
                        mpa[1][x->a][x->b->q_label.multiplicity()].push_back(x);
                    else
                        mpa[0][x->a][x->b->q_label.multiplicity()].push_back(x);
                    if (x->conj & 2)
                        mpb[1][x->b][x->a->q_label.multiplicity()].push_back(x);
                    else
                        mpb[0][x->b][x->a->q_label.multiplicity()].push_back(x);
                }
                terms.clear();
                if (mpa[0].size() + mpa[1].size() <=
                    mpb[0].size() + mpb[1].size()) {
                    for (int i = 0; i < 2; i++)
                        for (auto &r : mpa[i]) {
                            int pg = dynamic_pointer_cast<OpElement<S>>(r.first)
                                         ->q_label.pg() ^
                                     op.pg();
                            for (auto &rr : r.second) {
                                if (rr.second.size() == 1)
                                    terms.push_back(rr.second[0]);
                                else {
                                    vector<bool> conjs;
                                    vector<shared_ptr<OpElement<S>>> ops;
                                    conjs.reserve(rr.second.size());
                                    ops.reserve(rr.second.size());
                                    for (auto &s : rr.second) {
                                        if (s->b->q_label.pg() != pg)
                                            continue;
                                        bool cj = (s->conj & 2) != 0,
                                             found = false;
                                        OpElement<S> op = s->b->abs();
                                        for (size_t j = 0; j < ops.size(); j++)
                                            if (conjs[j] == cj &&
                                                op == ops[j]->abs()) {
                                                found = true;
                                                ops[j]->factor +=
                                                    s->b->factor * s->factor;
                                                break;
                                            }
                                        if (!found) {
                                            conjs.push_back((s->conj & 2) != 0);
                                            ops.push_back(dynamic_pointer_cast<
                                                          OpElement<S>>(
                                                (shared_ptr<OpExpr<S>>)s->b *
                                                s->factor));
                                        }
                                    }
                                    uint8_t cjx = i;
                                    if (conjs[0])
                                        conjs.flip(), cjx |= 1 << 1;
                                    if (ops.size() == 1)
                                        terms.push_back(
                                            make_shared<OpProduct<S>>(
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                ops[0], 1.0, cjx));
                                    else if (ops.size() != 0)
                                        terms.push_back(
                                            make_shared<OpSumProd<S>>(
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                ops, conjs, 1.0, cjx));
                                }
                            }
                        }
                } else {
                    for (int i = 0; i < 2; i++)
                        for (auto &r : mpb[i]) {
                            int pg = dynamic_pointer_cast<OpElement<S>>(r.first)
                                         ->q_label.pg() ^
                                     op.pg();
                            for (auto &rr : r.second) {
                                if (rr.second.size() == 1)
                                    terms.push_back(rr.second[0]);
                                else {
                                    vector<bool> conjs;
                                    vector<shared_ptr<OpElement<S>>> ops;
                                    conjs.reserve(rr.second.size());
                                    ops.reserve(rr.second.size());
                                    for (auto &s : rr.second) {
                                        if (s->a->q_label.pg() != pg)
                                            continue;
                                        bool cj = (s->conj & 1) != 0,
                                             found = false;
                                        OpElement<S> op = s->a->abs();
                                        for (size_t j = 0; j < ops.size(); j++)
                                            if (conjs[j] == cj &&
                                                op == ops[j]->abs()) {
                                                found = true;
                                                ops[j]->factor +=
                                                    s->a->factor * s->factor;
                                                break;
                                            }
                                        if (!found) {
                                            conjs.push_back((s->conj & 1) != 0);
                                            ops.push_back(dynamic_pointer_cast<
                                                          OpElement<S>>(
                                                (shared_ptr<OpExpr<S>>)s->a *
                                                s->factor));
                                        }
                                    }
                                    uint8_t cjx = i << 1;
                                    if (conjs[0])
                                        conjs.flip(), cjx |= 1;
                                    if (ops.size() == 1)
                                        terms.push_back(
                                            make_shared<OpProduct<S>>(
                                                ops[0],
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                1.0, cjx));
                                    else if (ops.size() != 0)
                                        terms.push_back(
                                            make_shared<OpSumProd<S>>(
                                                ops,
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                conjs, 1.0, cjx));
                                }
                            }
                        }
                }
                return make_shared<OpSum<S>>(terms);
            } else
                return make_shared<OpSum<S>>(terms);
        } break;
        case OpTypes::Zero:
        case OpTypes::Elem:
            return expr;
        default:
            assert(false);
            break;
        }
        return expr;
    }
    void simplify_symbolic(const shared_ptr<Symbolic<S>> &name,
                           const shared_ptr<Symbolic<S>> &expr,
                           const shared_ptr<Symbolic<S>> &ref = nullptr) {
        assert(name->data.size() == expr->data.size());
        size_t k = 0;
        for (size_t j = 0; j < name->data.size(); j++) {
            if (name->data[j]->get_type() == OpTypes::Zero)
                continue;
            else if (expr->data[j]->get_type() == OpTypes::Zero &&
                     (ref == nullptr || j >= ref->data.size() ||
                      ref->data[j] != name->data[j]))
                continue;
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(name->data[j]);
            if (rule->operator()(op) != nullptr)
                continue;
            name->data[k] = name->data[j];
            expr->data[k] = expr->data[j];
            k++;
        }
        name->data.resize(k);
        expr->data.resize(k);
        int ntg = ref != nullptr ? threading->activate_global() : 1;
#pragma omp parallel for schedule(static, 20) num_threads(ntg)
        for (int j = 0; j < (int)name->data.size(); j++) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(name->data[j]);
            name->data[j] = abs_value(name->data[j]);
            expr->data[j] =
                simplify_expr(expr->data[j], op->q_label) * (1 / op->factor);
        }
        if (use_intermediate) {
            uint16_t idxi = 0, idxj = 0;
            for (size_t j = 0; j < expr->data.size(); j++) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(name->data[j]);
                if (expr->data[j]->get_type() == OpTypes::Sum &&
                    intermediate_ops(op->name)) {
                    shared_ptr<OpSum<S>> ex =
                        dynamic_pointer_cast<OpSum<S>>(expr->data[j]);
                    for (size_t k = 0; k < ex->strings.size(); k++)
                        if (ex->strings[k]->get_type() == OpTypes::SumProd) {
                            shared_ptr<OpSumProd<S>> op =
                                dynamic_pointer_cast<OpSumProd<S>>(
                                    ex->strings[k]);
                            assert(op->ops.size() != 0);
                            op->c = make_shared<OpElement<S>>(
                                OpNames::TEMP, SiteIndex({idxj, idxi}, {}),
                                op->ops[0]->q_label);
                            idxi++;
                            if (idxi == 1000)
                                idxi = 0, idxj++;
                        }
                }
            }
        }
        if (name->get_type() == SymTypes::RVec)
            name->n = expr->n = (int)name->data.size();
        else
            name->m = expr->m = (int)name->data.size();
    }
    void simplify() {
        if (MPO<S>::schemer != nullptr) {
            simplify_symbolic(
                MPO<S>::schemer->left_new_operator_names,
                MPO<S>::schemer->left_new_operator_exprs,
                MPO<S>::left_operator_names[MPO<S>::schemer->left_trans_site]);
            simplify_symbolic(
                MPO<S>::schemer->right_new_operator_names,
                MPO<S>::schemer->right_new_operator_exprs,
                MPO<S>::right_operator_names[MPO<S>::schemer
                                                 ->right_trans_site]);
        }
        int ntg = threading->activate_global();
        vector<int> gidx(MPO<S>::n_sites);
        for (int i = 0; i < MPO<S>::n_sites; i++)
            gidx[i] = i;
        if (ntg != 1)
            sort(gidx.begin(), gidx.end(), [this](int i, int j) {
                return this->left_operator_names[i]->data.size() >
                       this->left_operator_names[j]->data.size();
            });
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int ii = 0; ii < MPO<S>::n_sites; ii++) {
            int i = gidx[ii];
            simplify_symbolic(MPO<S>::left_operator_names[i],
                              MPO<S>::left_operator_exprs[i]);
            simplify_symbolic(MPO<S>::right_operator_names[i],
                              MPO<S>::right_operator_exprs[i]);
            if (i < MPO<S>::n_sites - 1) {
                shared_ptr<Symbolic<S>> mexpr =
                    MPO<S>::middle_operator_exprs[i];
                for (size_t j = 0; j < mexpr->data.size(); j++)
                    mexpr->data[j] = simplify_expr(mexpr->data[j]);
            }
        }
    }
    AncillaTypes get_ancilla_type() const override {
        return prim_mpo->get_ancilla_type();
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

} // namespace block2
