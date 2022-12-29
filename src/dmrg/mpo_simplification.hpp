
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

#include "../core/rule.hpp"
#include "../core/threading.hpp"
#include "mpo.hpp"
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
template <typename S, typename FL> struct SimplifiedMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    // Original MPO
    shared_ptr<MPO<S, FL>> prim_mpo;
    shared_ptr<Rule<S, FL>> rule;
    // Collect terms means that sum of products will be changed to
    // product of one symbol times a sum of symbols
    // (if there are common factors)
    // A x B + A x C + A x D => A x (B + C + D)
    bool collect_terms, use_intermediate;
    // on some cases, non-redundant operators are removed because
    // themselves do not appear, but then they are used
    // indirectly from simplified redundant operators
    // this attribute will check and keep non-redundant operators
    // most mpo is written without the need to check this
    // currently, only the general spin mpo needs this
    bool check_indirect_ref;
    OpNamesSet intermediate_ops;
    SimplifiedMPO(const shared_ptr<MPO<S, FL>> &mpo,
                  const shared_ptr<Rule<S, FL>> &rule,
                  bool collect_terms = true, bool use_intermediate = false,
                  OpNamesSet intermediate_ops = OpNamesSet::all_ops(),
                  const string &tag = "", bool check_indirect_ref = true)
        : prim_mpo(mpo),
          rule(rule), MPO<S, FL>(mpo->n_sites, tag == "" ? mpo->tag : tag),
          collect_terms(collect_terms), use_intermediate(use_intermediate),
          intermediate_ops(intermediate_ops),
          check_indirect_ref(check_indirect_ref) {
        if (!collect_terms)
            use_intermediate = false;
        static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        MPO<S, FL>::hamil = mpo->hamil;
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::tensors = mpo->tensors;
        MPO<S, FL>::basis = mpo->basis;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::left_vacuum = mpo->left_vacuum;
        MPO<S, FL>::npdm_scheme = mpo->npdm_scheme;
        MPO<S, FL>::npdm_parallel_center = mpo->npdm_parallel_center;
        MPO<S, FL>::tf = mpo->tf;
        MPO<S, FL>::site_op_infos = mpo->site_op_infos;
        MPO<S, FL>::sparse_form = mpo->sparse_form;
        vector<size_t> left_op_sizes(MPO<S, FL>::n_sites);
        vector<size_t> right_op_sizes(MPO<S, FL>::n_sites);
        MPO<S, FL>::left_operator_names = mpo->left_operator_names;
        for (size_t i = 0; i < MPO<S, FL>::left_operator_names.size(); i++) {
            mpo->load_left_operators(i);
            MPO<S, FL>::left_operator_names[i] =
                mpo->left_operator_names[i]->copy();
            left_op_sizes[i] = MPO<S, FL>::left_operator_names[i]->data.size();
            mpo->unload_left_operators(i);
            MPO<S, FL>::save_left_operators(i);
            MPO<S, FL>::unload_left_operators(i);
        }
        MPO<S, FL>::right_operator_names = mpo->right_operator_names;
        for (size_t i = 0; i < MPO<S, FL>::right_operator_names.size(); i++) {
            mpo->load_right_operators(i);
            MPO<S, FL>::right_operator_names[i] =
                mpo->right_operator_names[i]->copy();
            right_op_sizes[i] =
                MPO<S, FL>::right_operator_names[i]->data.size();
            mpo->unload_right_operators(i);
            MPO<S, FL>::save_right_operators(i);
            MPO<S, FL>::unload_right_operators(i);
        }
        MPO<S, FL>::left_operator_exprs.resize(MPO<S, FL>::n_sites);
        MPO<S, FL>::right_operator_exprs.resize(MPO<S, FL>::n_sites);
        // for comp operators created in the middle site,
        // if all integrals related to the comp operators are zero,
        // label this comp operator as zero
        MPO<S, FL>::schemer = mpo->schemer;
        size_t left_new_size = 0, right_new_size = 0;
        if (MPO<S, FL>::schemer != nullptr) {
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO SIM load schemer ... " << endl;
            mpo->load_schemer();
            MPO<S, FL>::schemer =
                frame_<FP>()->minimal_memory_usage
                    ? make_shared<MPOSchemer<S>>(*mpo->schemer)
                    : mpo->schemer->copy();
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO SIM unload schemer ... " << endl;
            mpo->unload_schemer();
            int i = MPO<S, FL>::schemer->left_trans_site;
            left_new_size =
                MPO<S, FL>::schemer->left_new_operator_names->data.size();
            MPO<S, FL>::load_left_operators(i);
            for (size_t j = 0; j < left_new_size; j++) {
                if (j < MPO<S, FL>::left_operator_names[i]->data.size() &&
                    MPO<S, FL>::left_operator_names[i]->data[j] ==
                        MPO<S, FL>::schemer->left_new_operator_names->data[j])
                    continue;
                else if (MPO<S, FL>::schemer->left_new_operator_exprs->data[j]
                             ->get_type() == OpTypes::Zero)
                    MPO<S, FL>::schemer->left_new_operator_names->data[j] =
                        MPO<S, FL>::schemer->left_new_operator_exprs->data[j];
            }
            MPO<S, FL>::unload_left_operators(i);
            i = MPO<S, FL>::schemer->right_trans_site;
            right_new_size =
                MPO<S, FL>::schemer->right_new_operator_names->data.size();
            MPO<S, FL>::load_right_operators(i);
            for (size_t j = 0; j < right_new_size; j++) {
                if (j < MPO<S, FL>::right_operator_names[i]->data.size() &&
                    MPO<S, FL>::right_operator_names[i]->data[j] ==
                        MPO<S, FL>::schemer->right_new_operator_names->data[j])
                    continue;
                else if (MPO<S, FL>::schemer->right_new_operator_exprs->data[j]
                             ->get_type() == OpTypes::Zero)
                    MPO<S, FL>::schemer->right_new_operator_names->data[j] =
                        MPO<S, FL>::schemer->right_new_operator_exprs->data[j];
            }
            MPO<S, FL>::unload_right_operators(i);
            MPO<S, FL>::save_schemer();
            MPO<S, FL>::unload_schemer();
        }
        if (MPO<S, FL>::tag != mpo->tag)
            for (int i = 0; i < MPO<S, FL>::n_sites; i++) {
                mpo->load_tensor(i);
                MPO<S, FL>::tensors[i] = mpo->tensors[i];
                mpo->unload_tensor(i);
                MPO<S, FL>::save_tensor(i);
                MPO<S, FL>::unload_tensor(i);
            }
        if (mpo->middle_operator_exprs.size() != 0 &&
            MPO<S, FL>::tag == mpo->tag) {
            for (size_t i = 0; i < mpo->middle_operator_names.size(); i++) {
                mpo->load_left_operators(i);
                mpo->load_right_operators(i + 1);
            }
        }
        // construct blocking formulas by contration of op name (vector) and mpo
        // matrix; if left/right trans, by contration of new (comp) op name
        // (vector) and mpo matrix
        // if contracted expr is zero, label the blocked operator as zero
        int ntg = threading->activate_global();
        // left blocking
        for (int i = 0; i < MPO<S, FL>::n_sites; i++) {
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO SIM LEFT BLK ... " << setw(4) << i << " / "
                     << setw(4) << MPO<S, FL>::n_sites << endl;
            MPO<S, FL>::load_tensor(i, true);
            if (i == 0) {
                MPO<S, FL>::tensors[i] = MPO<S, FL>::tensors[i]->copy();
                if (MPO<S, FL>::tensors[i]->lmat ==
                    MPO<S, FL>::tensors[i]->rmat)
                    MPO<S, FL>::tensors[i]->lmat =
                        MPO<S, FL>::tensors[i]->rmat =
                            MPO<S, FL>::tensors[i]->rmat->copy();
                else
                    MPO<S, FL>::tensors[i]->lmat =
                        MPO<S, FL>::tensors[i]->lmat->copy();
                MPO<S, FL>::left_operator_exprs[i] =
                    MPO<S, FL>::tensors[i]->lmat;
            } else if (MPO<S, FL>::schemer == nullptr ||
                       i - 1 != MPO<S, FL>::schemer->left_trans_site) {
                MPO<S, FL>::load_left_operators(i - 1);
                MPO<S, FL>::left_operator_exprs[i] =
                    MPO<S, FL>::left_operator_names[i - 1] *
                    MPO<S, FL>::tensors[i]->lmat;
                MPO<S, FL>::unload_left_operators(i - 1);
            } else {
                MPO<S, FL>::load_schemer();
                MPO<S, FL>::left_operator_exprs[i] =
                    (shared_ptr<Symbolic<S>>)
                        MPO<S, FL>::schemer->left_new_operator_names *
                    MPO<S, FL>::tensors[i]->lmat;
                MPO<S, FL>::unload_schemer();
            }
            MPO<S, FL>::unload_tensor(i);
            MPO<S, FL>::load_left_operators(i);
            if (MPO<S, FL>::schemer != nullptr &&
                i == MPO<S, FL>::schemer->left_trans_site) {
                MPO<S, FL>::load_schemer();
                for (size_t j = 0;
                     j < MPO<S, FL>::left_operator_exprs[i]->data.size(); j++)
                    if (MPO<S, FL>::left_operator_exprs[i]
                            ->data[j]
                            ->get_type() == OpTypes::Zero) {
                        if (j < MPO<S, FL>::schemer->left_new_operator_names
                                    ->data.size() &&
                            MPO<S, FL>::left_operator_names[i]->data[j] ==
                                MPO<S, FL>::schemer->left_new_operator_names
                                    ->data[j])
                            MPO<S, FL>::schemer->left_new_operator_names
                                ->data[j] =
                                MPO<S, FL>::left_operator_exprs[i]->data[j];
                        MPO<S, FL>::left_operator_names[i]->data[j] =
                            MPO<S, FL>::left_operator_exprs[i]->data[j];
                    }
                MPO<S, FL>::save_schemer();
                MPO<S, FL>::unload_schemer();
            } else {
                for (size_t j = 0;
                     j < MPO<S, FL>::left_operator_exprs[i]->data.size(); j++)
                    if (MPO<S, FL>::left_operator_exprs[i]
                            ->data[j]
                            ->get_type() == OpTypes::Zero)
                        MPO<S, FL>::left_operator_names[i]->data[j] =
                            MPO<S, FL>::left_operator_exprs[i]->data[j];
            }
            MPO<S, FL>::save_left_operators(i);
            MPO<S, FL>::unload_left_operators(i);
        }
        // right blocking
        for (int i = MPO<S, FL>::n_sites - 1; i >= 0; i--) {
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO SIM RIGHT BLK ... " << setw(4) << i << " / "
                     << setw(4) << MPO<S, FL>::n_sites << endl;
            MPO<S, FL>::load_tensor(i, true);
            if (i == MPO<S, FL>::n_sites - 1) {
                MPO<S, FL>::tensors[i] = MPO<S, FL>::tensors[i]->copy();
                if (MPO<S, FL>::tensors[i]->lmat ==
                    MPO<S, FL>::tensors[i]->rmat)
                    MPO<S, FL>::tensors[i]->rmat =
                        MPO<S, FL>::tensors[i]->lmat =
                            MPO<S, FL>::tensors[i]->lmat->copy();
                else
                    MPO<S, FL>::tensors[i]->rmat =
                        MPO<S, FL>::tensors[i]->rmat->copy();
                MPO<S, FL>::right_operator_exprs[i] =
                    MPO<S, FL>::tensors[i]->rmat;
            } else if (MPO<S, FL>::schemer == nullptr ||
                       i + 1 != MPO<S, FL>::schemer->right_trans_site) {
                MPO<S, FL>::load_right_operators(i + 1);
                MPO<S, FL>::right_operator_exprs[i] =
                    MPO<S, FL>::tensors[i]->rmat *
                    MPO<S, FL>::right_operator_names[i + 1];
                MPO<S, FL>::unload_right_operators(i + 1);
            } else {
                MPO<S, FL>::load_schemer();
                MPO<S, FL>::right_operator_exprs[i] =
                    MPO<S, FL>::tensors[i]->rmat *
                    (shared_ptr<Symbolic<S>>)
                        MPO<S, FL>::schemer->right_new_operator_names;
                MPO<S, FL>::unload_schemer();
            }
            MPO<S, FL>::unload_tensor(i);
            MPO<S, FL>::load_right_operators(i);
            if (MPO<S, FL>::schemer != nullptr &&
                i == MPO<S, FL>::schemer->right_trans_site) {
                MPO<S, FL>::load_schemer();
                for (size_t j = 0;
                     j < MPO<S, FL>::right_operator_exprs[i]->data.size(); j++)
                    if (MPO<S, FL>::right_operator_exprs[i]
                            ->data[j]
                            ->get_type() == OpTypes::Zero) {
                        if (j < MPO<S, FL>::schemer->right_new_operator_names
                                    ->data.size() &&
                            MPO<S, FL>::right_operator_names[i]->data[j] ==
                                MPO<S, FL>::schemer->right_new_operator_names
                                    ->data[j])
                            MPO<S, FL>::schemer->right_new_operator_names
                                ->data[j] =
                                MPO<S, FL>::right_operator_exprs[i]->data[j];
                        MPO<S, FL>::right_operator_names[i]->data[j] =
                            MPO<S, FL>::right_operator_exprs[i]->data[j];
                    }
                MPO<S, FL>::save_schemer();
                MPO<S, FL>::unload_schemer();
            } else {
                for (size_t j = 0;
                     j < MPO<S, FL>::right_operator_exprs[i]->data.size(); j++)
                    if (MPO<S, FL>::right_operator_exprs[i]
                            ->data[j]
                            ->get_type() == OpTypes::Zero)
                        MPO<S, FL>::right_operator_names[i]->data[j] =
                            MPO<S, FL>::right_operator_exprs[i]->data[j];
            }
            MPO<S, FL>::save_right_operators(i);
            MPO<S, FL>::unload_right_operators(i);
        }
        // construct super blocking contraction formula
        // first case is that the blocking formula is already given
        // for example in the npdm code
        if (mpo->middle_operator_exprs.size() != 0) {
            MPO<S, FL>::middle_operator_names = mpo->middle_operator_names;
            MPO<S, FL>::middle_operator_exprs = mpo->middle_operator_exprs;
            using MSF = MPO<S, FL>;
            assert(MSF::schemer == nullptr);
            // if some operators are erased in left/right operator names
            // they should not appear in middle operator exprs
            // and the expr should be zero
            for (size_t i = 0; i < MPO<S, FL>::middle_operator_names.size();
                 i++) {
                if (frame_<FP>()->minimal_memory_usage)
                    cout << "MPO SIM MIDDLE DEP ... " << setw(4) << i << " / "
                         << setw(4) << MPO<S, FL>::n_sites << endl;
                mpo->load_middle_operators(i);
                MPO<S, FL>::middle_operator_names[i] =
                    mpo->middle_operator_names[i]->copy();
                MPO<S, FL>::middle_operator_exprs[i] =
                    mpo->middle_operator_exprs[i]->copy();
                mpo->unload_middle_operators(i);
                set<shared_ptr<OpExpr<S>>, op_expr_less<S>> left_zero_ops,
                    right_zero_ops;
                MPO<S, FL>::load_left_operators(i);
                for (size_t j = 0;
                     j < MPO<S, FL>::left_operator_names[i]->data.size(); j++)
                    if (MPO<S, FL>::left_operator_names[i]
                            ->data[j]
                            ->get_type() == OpTypes::Zero)
                        left_zero_ops.insert(
                            mpo->left_operator_names[i]->data[j]);
                mpo->unload_left_operators(i);
                MPO<S, FL>::unload_left_operators(i);
                MPO<S, FL>::load_right_operators(i + 1);
                for (size_t j = 0;
                     j < MPO<S, FL>::right_operator_names[i + 1]->data.size();
                     j++)
                    if (MPO<S, FL>::right_operator_names[i + 1]
                            ->data[j]
                            ->get_type() == OpTypes::Zero)
                        right_zero_ops.insert(
                            mpo->right_operator_names[i + 1]->data[j]);
                mpo->unload_right_operators(i + 1);
                MPO<S, FL>::unload_right_operators(i + 1);
                for (size_t j = 0;
                     j < MPO<S, FL>::middle_operator_exprs[i]->data.size();
                     j++) {
                    shared_ptr<OpExpr<S>> &x =
                        MPO<S, FL>::middle_operator_exprs[i]->data[j];
                    switch (x->get_type()) {
                    case OpTypes::Zero:
                        break;
                    case OpTypes::Prod:
                        if (left_zero_ops.count(
                                dynamic_pointer_cast<OpProduct<S, FL>>(x)->a) ||
                            right_zero_ops.count(
                                dynamic_pointer_cast<OpProduct<S, FL>>(x)->b))
                            MPO<S, FL>::middle_operator_exprs[i]->data[j] =
                                zero;
                        break;
                    case OpTypes::Sum:
                        for (auto &r :
                             dynamic_pointer_cast<OpSum<S, FL>>(x)->strings)
                            if (left_zero_ops.count(
                                    dynamic_pointer_cast<OpProduct<S, FL>>(r)
                                        ->a) ||
                                right_zero_ops.count(
                                    dynamic_pointer_cast<OpProduct<S, FL>>(r)
                                        ->b)) {
                                MPO<S, FL>::middle_operator_exprs[i]->data[j] =
                                    zero;
                                break;
                            }
                        break;
                    default:
                        assert(false);
                    }
                }
                MPO<S, FL>::save_middle_operators(i);
                MPO<S, FL>::unload_middle_operators(i);
            }
        } else {
            vector<uint8_t> px[2];
            unordered_map<shared_ptr<OpExpr<S>>, int> xmp;
            // figure out the mutual dependence from right to left
            // px[.][j] is 1 if left operator is useful in next blocking
            for (int i = MPO<S, FL>::n_sites - 1; i >= 0; i--) {
                if (frame_<FP>()->minimal_memory_usage)
                    cout << "MPO SIM LEFT DEP ... " << setw(4) << i << " / "
                         << setw(4) << MPO<S, FL>::n_sites << endl;
                MPO<S, FL>::load_left_operators(i);
                if (i != MPO<S, FL>::n_sites - 1) {
                    // if a left operator is not useful in next blocking
                    // and not useful in super block
                    // then it is labelled as not useful
                    // when it is not useful, set it to zero
                    if (MPO<S, FL>::schemer == nullptr ||
                        i != MPO<S, FL>::schemer->left_trans_site) {
                        MPO<S, FL>::load_right_operators(i + 1);
                        for (size_t j = 0;
                             j <
                             MPO<S, FL>::left_operator_names[i]->data.size();
                             j++)
                            if (MPO<S, FL>::right_operator_names[i + 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j]) {
                                if (check_indirect_ref)
                                    xmp[abs_value(
                                        MPO<S, FL>::left_operator_names[i]
                                            ->data[j])] = j;
                                else
                                    MPO<S, FL>::left_operator_names[i]
                                        ->data[j] = zero;
                            } else if (MPO<S, FL>::left_operator_names[i]
                                           ->data[j]
                                           ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                        MPO<S, FL>::unload_right_operators(i + 1);
                        if (xmp.size() != 0) {
                            for (size_t j = 0;
                                 j < MPO<S, FL>::left_operator_names[i]
                                         ->data.size();
                                 j++)
                                if (px[i & 1][j] &&
                                    MPO<S, FL>::left_operator_names[i]
                                            ->data[j]
                                            ->get_type() != OpTypes::Zero) {
                                    auto xref = rule->operator()(
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            MPO<S, FL>::left_operator_names[i]
                                                ->data[j]));
                                    if (xref != nullptr &&
                                        xmp.count(xref->op) != 0)
                                        xmp.at(xref->op) = -1;
                                }
                            for (auto &j : xmp)
                                if (j.second != -1)
                                    MPO<S, FL>::left_operator_names[i]
                                        ->data[j.second] = zero;
                            xmp.clear();
                        }
                    } else if (MPO<S, FL>::schemer->right_trans_site -
                                   MPO<S, FL>::schemer->left_trans_site >
                               1) {
                        MPO<S, FL>::load_schemer();
                        for (size_t j = 0;
                             j < MPO<S, FL>::schemer->left_new_operator_names
                                     ->data.size();
                             j++)
                            if (MPO<S, FL>::schemer->left_new_operator_names
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                        MPO<S, FL>::unload_schemer();
                    } else {
                        MPO<S, FL>::load_right_operators(i + 1);
                        MPO<S, FL>::load_schemer();
                        for (size_t j = 0;
                             j < MPO<S, FL>::schemer->left_new_operator_names
                                     ->data.size();
                             j++)
                            if (MPO<S, FL>::right_operator_names[i + 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j]) {
                                if (check_indirect_ref)
                                    xmp[abs_value(MPO<S, FL>::schemer
                                                      ->left_new_operator_names
                                                      ->data[j])] = j;
                                else
                                    MPO<S, FL>::schemer->left_new_operator_names
                                        ->data[j] = zero;
                            } else if (MPO<S, FL>::schemer
                                           ->left_new_operator_names->data[j]
                                           ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                        MPO<S, FL>::unload_right_operators(i + 1);
                        if (xmp.size() != 0) {
                            for (size_t j = 0;
                                 j < MPO<S, FL>::schemer
                                         ->left_new_operator_names->data.size();
                                 j++)
                                if (px[i & 1][j] &&
                                    MPO<S, FL>::schemer->left_new_operator_names
                                            ->data[j]
                                            ->get_type() != OpTypes::Zero) {
                                    auto xref = rule->operator()(
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            MPO<S, FL>::schemer
                                                ->left_new_operator_names
                                                ->data[j]));
                                    if (xref != nullptr &&
                                        xmp.count(xref->op) != 0)
                                        xmp.at(xref->op) = -1;
                                }
                            for (auto &j : xmp)
                                if (j.second != -1)
                                    MPO<S, FL>::schemer->left_new_operator_names
                                        ->data[j.second] = zero;
                            xmp.clear();
                        }
                        MPO<S, FL>::save_schemer();
                        MPO<S, FL>::unload_schemer();
                    }
                    if (MPO<S, FL>::schemer != nullptr &&
                        i == MPO<S, FL>::schemer->left_trans_site) {
                        MPO<S, FL>::load_schemer();
                        px[!(i & 1)].resize(px[i & 1].size());
                        memcpy(px[!(i & 1)].data(), px[i & 1].data(),
                               sizeof(uint8_t) * px[!(i & 1)].size());
                        px[i & 1].resize(
                            MPO<S, FL>::left_operator_names[i]->data.size());
                        memset(px[i & 1].data(), 0,
                               sizeof(uint8_t) * px[i & 1].size());
                        unordered_map<shared_ptr<OpExpr<S>>, int> mp;
                        mp.reserve(
                            MPO<S, FL>::left_operator_names[i]->data.size());
                        for (size_t j = 0;
                             j <
                             MPO<S, FL>::left_operator_names[i]->data.size();
                             j++)
                            if (MPO<S, FL>::left_operator_names[i]
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                mp[abs_value(MPO<S, FL>::left_operator_names[i]
                                                 ->data[j])] = (int)j;
                        shared_ptr<SymbolicRowVector<S>> &exprs =
                            MPO<S, FL>::schemer->left_new_operator_exprs;
                        for (size_t j = 0; j < exprs->data.size(); j++) {
                            if (px[!(i & 1)][j] &&
                                j < MPO<S, FL>::left_operator_names[i]
                                        ->data.size() &&
                                MPO<S, FL>::left_operator_names[i]->data[j] ==
                                    MPO<S, FL>::schemer->left_new_operator_names
                                        ->data[j])
                                px[i & 1][j] = 1;
                            else if (px[!(i & 1)][j] &&
                                     exprs->data[j]->get_type() !=
                                         OpTypes::Zero) {
                                shared_ptr<OpSum<S, FL>> op =
                                    make_shared<OpSum<S, FL>>(
                                        dynamic_pointer_cast<OpSum<S, FL>>(
                                            exprs->data[j])
                                            ->strings);
                                for (size_t k = 0; k < op->strings.size();
                                     k++) {
                                    shared_ptr<OpExpr<S>> expr = abs_value(
                                        (shared_ptr<OpExpr<S>>)op->strings[k]
                                            ->a);
                                    if (mp.count(expr) == 0)
                                        op->strings[k] =
                                            make_shared<OpProduct<S, FL>>(
                                                *op->strings[k] * 0.0);
                                    else {
                                        px[i & 1][mp[expr]] = 1;
                                        if (check_indirect_ref) {
                                            auto xref = rule->operator()(
                                                dynamic_pointer_cast<
                                                    OpElement<S, FL>>(expr));
                                            if (xref != nullptr &&
                                                mp.count(xref->op))
                                                px[i & 1][mp[xref->op]] = 1;
                                        }
                                    }
                                    assert(op->strings[k]->b == nullptr);
                                }
                                exprs->data[j] = op;
                            }
                        }
                        for (size_t j = 0;
                             j <
                             MPO<S, FL>::left_operator_names[i]->data.size();
                             j++)
                            if (!px[i & 1][j])
                                MPO<S, FL>::left_operator_names[i]->data[j] =
                                    zero;
                        MPO<S, FL>::save_schemer();
                        MPO<S, FL>::unload_schemer();
                    }
                }
                MPO<S, FL>::save_left_operators(i);
                MPO<S, FL>::unload_left_operators(i);
                // at the beginning, all px values are zero
                // then, set the required op px = 1 based on mpo matrix
                if (i != 0) {
                    if (MPO<S, FL>::schemer == nullptr ||
                        i - 1 != MPO<S, FL>::schemer->left_trans_site) {
                        px[!(i & 1)].resize(left_op_sizes[i - 1]);
                    } else {
                        px[!(i & 1)].resize(left_new_size);
                    }
                    memset(&px[!(i & 1)][0], 0,
                           sizeof(uint8_t) * px[!(i & 1)].size());
                    MPO<S, FL>::load_tensor(i, true);
                    if (MPO<S, FL>::tensors[i]->lmat->get_type() ==
                        SymTypes::Mat) {
                        assert(px[i & 1].size() != 0);
                        shared_ptr<SymbolicMatrix<S>> mat =
                            dynamic_pointer_cast<SymbolicMatrix<S>>(
                                MPO<S, FL>::tensors[i]->lmat);
                        for (size_t j = 0; j < mat->data.size(); j++)
                            if (px[i & 1][mat->indices[j].second] &&
                                mat->data[j]->get_type() != OpTypes::Zero)
                                px[!(i & 1)][mat->indices[j].first] = 1;
                    }
                    MPO<S, FL>::unload_tensor(i);
                }
            }
            // figure out the mutual dependence from left to right
            for (int i = 0; i < MPO<S, FL>::n_sites; i++) {
                if (frame_<FP>()->minimal_memory_usage)
                    cout << "MPO SIM RIGHT DEP ... " << setw(4) << i << " / "
                         << setw(4) << MPO<S, FL>::n_sites << endl;
                MPO<S, FL>::load_right_operators(i);
                if (i != 0) {
                    if (MPO<S, FL>::schemer == nullptr ||
                        i != MPO<S, FL>::schemer->right_trans_site) {
                        MPO<S, FL>::load_left_operators(i - 1);
                        for (size_t j = 0;
                             j <
                             MPO<S, FL>::right_operator_names[i]->data.size();
                             j++)
                            if (MPO<S, FL>::left_operator_names[i - 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j]) {
                                if (check_indirect_ref)
                                    xmp[abs_value(
                                        MPO<S, FL>::right_operator_names[i]
                                            ->data[j])] = j;
                                else
                                    MPO<S, FL>::right_operator_names[i]
                                        ->data[j] = zero;
                            } else if (MPO<S, FL>::right_operator_names[i]
                                           ->data[j]
                                           ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                        MPO<S, FL>::unload_left_operators(i - 1);
                        if (xmp.size() != 0) {
                            for (size_t j = 0;
                                 j < MPO<S, FL>::right_operator_names[i]
                                         ->data.size();
                                 j++)
                                if (px[i & 1][j] &&
                                    MPO<S, FL>::right_operator_names[i]
                                            ->data[j]
                                            ->get_type() != OpTypes::Zero) {
                                    auto xref = rule->operator()(
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            MPO<S, FL>::right_operator_names[i]
                                                ->data[j]));
                                    if (xref != nullptr &&
                                        xmp.count(xref->op) != 0)
                                        xmp.at(xref->op) = -1;
                                }
                            for (auto &j : xmp)
                                if (j.second != -1)
                                    MPO<S, FL>::right_operator_names[i]
                                        ->data[j.second] = zero;
                            xmp.clear();
                        }
                    } else if (MPO<S, FL>::schemer->right_trans_site -
                                   MPO<S, FL>::schemer->left_trans_site >
                               1) {
                        MPO<S, FL>::load_schemer();
                        for (size_t j = 0;
                             j < MPO<S, FL>::schemer->right_new_operator_names
                                     ->data.size();
                             j++)
                            if (MPO<S, FL>::schemer->right_new_operator_names
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                        MPO<S, FL>::unload_schemer();
                    } else {
                        MPO<S, FL>::load_left_operators(i - 1);
                        MPO<S, FL>::load_schemer();
                        for (size_t j = 0;
                             j < MPO<S, FL>::schemer->right_new_operator_names
                                     ->data.size();
                             j++)
                            if (MPO<S, FL>::left_operator_names[i - 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j]) {
                                if (check_indirect_ref)
                                    xmp[abs_value(MPO<S, FL>::schemer
                                                      ->right_new_operator_names
                                                      ->data[j])] = j;
                                else
                                    MPO<S, FL>::schemer
                                        ->right_new_operator_names->data[j] =
                                        zero;
                            } else if (MPO<S, FL>::schemer
                                           ->right_new_operator_names->data[j]
                                           ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                        MPO<S, FL>::unload_left_operators(i - 1);
                        if (xmp.size() != 0) {
                            for (size_t j = 0;
                                 j <
                                 MPO<S, FL>::schemer->right_new_operator_names
                                     ->data.size();
                                 j++)
                                if (px[i & 1][j] &&
                                    MPO<S, FL>::schemer
                                            ->right_new_operator_names->data[j]
                                            ->get_type() != OpTypes::Zero) {
                                    auto xref = rule->operator()(
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            MPO<S, FL>::schemer
                                                ->right_new_operator_names
                                                ->data[j]));
                                    if (xref != nullptr &&
                                        xmp.count(xref->op) != 0)
                                        xmp.at(xref->op) = -1;
                                }
                            for (auto &j : xmp)
                                if (j.second != -1)
                                    MPO<S, FL>::schemer
                                        ->right_new_operator_names
                                        ->data[j.second] = zero;
                            xmp.clear();
                        }
                        MPO<S, FL>::save_schemer();
                        MPO<S, FL>::unload_schemer();
                    }
                    if (MPO<S, FL>::schemer != nullptr &&
                        i == MPO<S, FL>::schemer->right_trans_site) {
                        MPO<S, FL>::load_schemer();
                        px[!(i & 1)].resize(px[i & 1].size());
                        memcpy(px[!(i & 1)].data(), px[i & 1].data(),
                               sizeof(uint8_t) * px[!(i & 1)].size());
                        px[i & 1].resize(
                            MPO<S, FL>::right_operator_names[i]->data.size());
                        memset(px[i & 1].data(), 0,
                               sizeof(uint8_t) * px[i & 1].size());
                        unordered_map<shared_ptr<OpExpr<S>>, int> mp;
                        mp.reserve(
                            MPO<S, FL>::right_operator_names[i]->data.size());
                        for (size_t j = 0;
                             j <
                             MPO<S, FL>::right_operator_names[i]->data.size();
                             j++)
                            if (MPO<S, FL>::right_operator_names[i]
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                mp[abs_value(MPO<S, FL>::right_operator_names[i]
                                                 ->data[j])] = (int)j;
                        shared_ptr<SymbolicColumnVector<S>> &exprs =
                            MPO<S, FL>::schemer->right_new_operator_exprs;
                        for (size_t j = 0; j < exprs->data.size(); j++) {
                            if (px[!(i & 1)][j] &&
                                j < MPO<S, FL>::right_operator_names[i]
                                        ->data.size() &&
                                MPO<S, FL>::right_operator_names[i]->data[j] ==
                                    MPO<S, FL>::schemer
                                        ->right_new_operator_names->data[j])
                                px[i & 1][j] = 1;
                            else if (px[!(i & 1)][j] &&
                                     exprs->data[j]->get_type() !=
                                         OpTypes::Zero) {
                                shared_ptr<OpSum<S, FL>> op =
                                    make_shared<OpSum<S, FL>>(
                                        dynamic_pointer_cast<OpSum<S, FL>>(
                                            exprs->data[j])
                                            ->strings);
                                for (size_t k = 0; k < op->strings.size();
                                     k++) {
                                    shared_ptr<OpExpr<S>> expr = abs_value(
                                        (shared_ptr<OpExpr<S>>)op->strings[k]
                                            ->a);
                                    if (mp.count(expr) == 0)
                                        op->strings[k] =
                                            make_shared<OpProduct<S, FL>>(
                                                *op->strings[k] * 0.0);
                                    else {
                                        px[i & 1][mp[expr]] = 1;
                                        if (check_indirect_ref) {
                                            auto xref = rule->operator()(
                                                dynamic_pointer_cast<
                                                    OpElement<S, FL>>(expr));
                                            if (xref != nullptr &&
                                                mp.count(xref->op))
                                                px[i & 1][mp[xref->op]] = 1;
                                        }
                                    }
                                    assert(op->strings[k]->b == nullptr);
                                }
                                exprs->data[j] = op;
                            }
                        }
                        for (size_t j = 0;
                             j <
                             MPO<S, FL>::right_operator_names[i]->data.size();
                             j++)
                            if (!px[i & 1][j])
                                MPO<S, FL>::right_operator_names[i]->data[j] =
                                    zero;
                        MPO<S, FL>::save_schemer();
                        MPO<S, FL>::unload_schemer();
                    }
                }
                MPO<S, FL>::save_right_operators(i);
                MPO<S, FL>::unload_right_operators(i);
                if (i != MPO<S, FL>::n_sites - 1) {
                    if (MPO<S, FL>::schemer == nullptr ||
                        i + 1 != MPO<S, FL>::schemer->right_trans_site)
                        px[!(i & 1)].resize(right_op_sizes[i + 1]);
                    else
                        px[!(i & 1)].resize(right_new_size);
                    memset(px[!(i & 1)].data(), 0,
                           sizeof(uint8_t) * px[!(i & 1)].size());
                    MPO<S, FL>::load_tensor(i, true);
                    if (MPO<S, FL>::tensors[i]->rmat->get_type() ==
                        SymTypes::Mat) {
                        shared_ptr<SymbolicMatrix<S>> mat =
                            dynamic_pointer_cast<SymbolicMatrix<S>>(
                                MPO<S, FL>::tensors[i]->rmat);
                        for (size_t j = 0; j < mat->data.size(); j++)
                            if (px[i & 1][mat->indices[j].first] &&
                                mat->data[j]->get_type() != OpTypes::Zero)
                                px[!(i & 1)][mat->indices[j].second] = 1;
                    }
                    MPO<S, FL>::unload_tensor(i);
                }
            }
            MPO<S, FL>::middle_operator_names.resize(MPO<S, FL>::n_sites - 1);
            MPO<S, FL>::middle_operator_exprs.resize(MPO<S, FL>::n_sites - 1);
            shared_ptr<SymbolicColumnVector<S>> mpo_op =
                make_shared<SymbolicColumnVector<S>>(1);
            (*mpo_op)[0] = mpo->op;
            for (int i = 0; i < MPO<S, FL>::n_sites - 1; i++) {
                if (frame_<FP>()->minimal_memory_usage)
                    cout << "MPO SIM MID ... " << setw(4) << i << " / "
                         << setw(4) << MPO<S, FL>::n_sites << endl;
                MPO<S, FL>::middle_operator_names[i] = mpo_op;
                if (MPO<S, FL>::schemer == nullptr ||
                    i != MPO<S, FL>::schemer->left_trans_site ||
                    MPO<S, FL>::schemer->right_trans_site -
                            MPO<S, FL>::schemer->left_trans_site >
                        1) {
                    MPO<S, FL>::load_left_operators(i);
                    MPO<S, FL>::load_right_operators(i + 1);
                    MPO<S, FL>::middle_operator_exprs[i] =
                        MPO<S, FL>::left_operator_names[i] *
                        MPO<S, FL>::right_operator_names[i + 1];
                    MPO<S, FL>::unload_right_operators(i + 1);
                    MPO<S, FL>::unload_left_operators(i);
                } else {
                    MPO<S, FL>::load_schemer();
                    MPO<S, FL>::load_right_operators(i + 1);
                    MPO<S, FL>::middle_operator_exprs[i] =
                        (shared_ptr<Symbolic<S>>)
                            MPO<S, FL>::schemer->left_new_operator_names *
                        MPO<S, FL>::right_operator_names[i + 1];
                    MPO<S, FL>::unload_right_operators(i + 1);
                    MPO<S, FL>::unload_schemer();
                }
                MPO<S, FL>::save_middle_operators(i);
                MPO<S, FL>::unload_middle_operators(i);
            }
        }
        simplify(left_op_sizes);
        // sync left assign
        MPO<S, FL>::load_tensor(0);
        MPO<S, FL>::load_left_operators(0);
        MPO<S, FL>::tensors[0]->lmat = MPO<S, FL>::left_operator_exprs[0];
        MPO<S, FL>::unload_left_operators(0);
        MPO<S, FL>::save_tensor(0);
        MPO<S, FL>::unload_tensor(0);
        // sync right assign
        MPO<S, FL>::load_tensor(MPO<S, FL>::n_sites - 1);
        MPO<S, FL>::load_right_operators(MPO<S, FL>::n_sites - 1);
        MPO<S, FL>::tensors[MPO<S, FL>::n_sites - 1]->rmat =
            MPO<S, FL>::right_operator_exprs[MPO<S, FL>::n_sites - 1];
        MPO<S, FL>::unload_right_operators(MPO<S, FL>::n_sites - 1);
        MPO<S, FL>::save_tensor(MPO<S, FL>::n_sites - 1);
        MPO<S, FL>::unload_tensor(MPO<S, FL>::n_sites - 1);
        threading->activate_normal();
    }
    shared_ptr<OpExpr<S>> simplify_expr(const shared_ptr<OpExpr<S>> &expr,
                                        S op = S(S::invalid)) {
        static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->b != nullptr);
            shared_ptr<OpElementRef<S, FL>> opl = rule->operator()(op->a);
            shared_ptr<OpElementRef<S, FL>> opr = rule->operator()(op->b);
            shared_ptr<OpElement<S, FL>> a = opl == nullptr ? op->a : opl->op;
            shared_ptr<OpElement<S, FL>> b = opr == nullptr ? op->b : opr->op;
            uint8_t conj = (opl != nullptr && opl->trans) |
                           ((opr != nullptr && opr->trans) << 1);
            FL factor = (opl != nullptr ? opl->factor : (FL)1.0) *
                        (opr != nullptr ? opr->factor : (FL)1.0) * op->factor;
            return make_shared<OpProduct<S, FL>>(a, b, factor, conj);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> ops =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            // merge terms that differ only by coefficients
            unordered_map<shared_ptr<OpExpr<S>>,
                          vector<shared_ptr<OpProduct<S, FL>>>>
                mp;
            mp.reserve(ops->strings.size());
            for (auto &x : ops->strings) {
                if (x->factor == (FL)0.0)
                    continue;
                shared_ptr<OpElementRef<S, FL>> opl = rule->operator()(x->a);
                shared_ptr<OpElementRef<S, FL>> opr =
                    x->b == nullptr ? nullptr : rule->operator()(x->b);
                shared_ptr<OpElement<S, FL>> a =
                    opl == nullptr ? x->a : opl->op;
                shared_ptr<OpElement<S, FL>> b =
                    opr == nullptr ? x->b : opr->op;
                uint8_t conj = (opl != nullptr && opl->trans) |
                               ((opr != nullptr && opr->trans) << 1);
                FL factor = (opl != nullptr ? opl->factor : (FL)1.0) *
                            (opr != nullptr ? opr->factor : (FL)1.0) *
                            x->factor;
                if (!mp.count(a))
                    mp[a] = vector<shared_ptr<OpProduct<S, FL>>>();
                vector<shared_ptr<OpProduct<S, FL>>> &px = mp.at(a);
                int g = -1;
                for (size_t k = 0; k < px.size(); k++)
                    if (px[k]->b == b && px[k]->conj == conj) {
                        g = (int)k;
                        break;
                    }
                if (g == -1)
                    px.push_back(
                        make_shared<OpProduct<S, FL>>(a, b, factor, conj));
                else {
                    px[g]->factor += factor;
                    if (abs(px[g]->factor) < TINY)
                        px.erase(px.begin() + g);
                }
            }
            vector<shared_ptr<OpProduct<S, FL>>> terms;
            terms.reserve(mp.size());
            for (auto &r : mp)
                terms.insert(terms.end(), r.second.begin(), r.second.end());
            if (terms.size() == 0)
                return zero;
            else if (terms[0]->b == nullptr || terms.size() <= 2)
                return make_shared<OpSum<S, FL>>(terms);
            else if (collect_terms && op != S(S::invalid)) {
                // extract common factors from terms
                unordered_map<shared_ptr<OpExpr<S>>,
                              map<int, vector<shared_ptr<OpProduct<S, FL>>>>>
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
                    // merge right part
                    for (int i = 0; i < 2; i++)
                        for (auto &r : mpa[i]) {
                            // pgb = op.pg - pga
                            int pga =
                                dynamic_pointer_cast<OpElement<S, FL>>(r.first)
                                    ->q_label.pg();
                            int pg =
                                S::pg_mul(i ? pga : S::pg_inv(pga), op.pg());
                            for (auto &rr : r.second) {
                                if (rr.second.size() == 1)
                                    terms.push_back(rr.second[0]);
                                else {
                                    vector<bool> conjs;
                                    vector<shared_ptr<OpElement<S, FL>>> ops;
                                    conjs.reserve(rr.second.size());
                                    ops.reserve(rr.second.size());
                                    for (auto &s : rr.second) {
                                        if (!S::pg_equal(s->b->q_label.pg(),
                                                         (s->conj & 2)
                                                             ? S::pg_inv(pg)
                                                             : pg))
                                            continue;
                                        bool cj = (s->conj & 2) != 0,
                                             found = false;
                                        OpElement<S, FL> op = s->b->abs();
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
                                                          OpElement<S, FL>>(
                                                (shared_ptr<OpExpr<S>>)s->b *
                                                s->factor));
                                        }
                                    }
                                    uint8_t cjx = i;
                                    if (conjs[0])
                                        conjs.flip(), cjx |= 1 << 1;
                                    if (ops.size() == 1)
                                        terms.push_back(
                                            make_shared<OpProduct<S, FL>>(
                                                dynamic_pointer_cast<
                                                    OpElement<S, FL>>(r.first),
                                                ops[0], 1.0, cjx));
                                    else if (ops.size() != 0)
                                        terms.push_back(
                                            make_shared<OpSumProd<S, FL>>(
                                                dynamic_pointer_cast<
                                                    OpElement<S, FL>>(r.first),
                                                ops, conjs, 1.0, cjx));
                                }
                            }
                        }
                } else {
                    // merge left part
                    for (int i = 0; i < 2; i++)
                        for (auto &r : mpb[i]) {
                            // pga = op.pg - pgb
                            int pgb =
                                dynamic_pointer_cast<OpElement<S, FL>>(r.first)
                                    ->q_label.pg();
                            int pg =
                                S::pg_mul(i ? pgb : S::pg_inv(pgb), op.pg());
                            for (auto &rr : r.second) {
                                if (rr.second.size() == 1)
                                    terms.push_back(rr.second[0]);
                                else {
                                    vector<bool> conjs;
                                    vector<shared_ptr<OpElement<S, FL>>> ops;
                                    conjs.reserve(rr.second.size());
                                    ops.reserve(rr.second.size());
                                    for (auto &s : rr.second) {
                                        if (!S::pg_equal(s->a->q_label.pg(),
                                                         (s->conj & 1)
                                                             ? S::pg_inv(pg)
                                                             : pg))
                                            continue;
                                        bool cj = (s->conj & 1) != 0,
                                             found = false;
                                        OpElement<S, FL> op = s->a->abs();
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
                                                          OpElement<S, FL>>(
                                                (shared_ptr<OpExpr<S>>)s->a *
                                                s->factor));
                                        }
                                    }
                                    uint8_t cjx = i << 1;
                                    if (conjs[0])
                                        conjs.flip(), cjx |= 1;
                                    if (ops.size() == 1)
                                        terms.push_back(
                                            make_shared<OpProduct<S, FL>>(
                                                ops[0],
                                                dynamic_pointer_cast<
                                                    OpElement<S, FL>>(r.first),
                                                1.0, cjx));
                                    else if (ops.size() != 0)
                                        terms.push_back(
                                            make_shared<OpSumProd<S, FL>>(
                                                ops,
                                                dynamic_pointer_cast<
                                                    OpElement<S, FL>>(r.first),
                                                conjs, 1.0, cjx));
                                }
                            }
                        }
                }
                return make_shared<OpSum<S, FL>>(terms);
            } else
                return make_shared<OpSum<S, FL>>(terms);
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
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(name->data[j]);
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
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(name->data[j]);
            name->data[j] = abs_value(name->data[j]);
            expr->data[j] = simplify_expr(expr->data[j], op->q_label) *
                            ((FL)1.0 / op->factor);
        }
        if (use_intermediate) {
            uint16_t idxi = 0, idxj = 0;
            for (size_t j = 0; j < expr->data.size(); j++) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(name->data[j]);
                if (expr->data[j]->get_type() == OpTypes::Sum &&
                    intermediate_ops(op->name)) {
                    shared_ptr<OpSum<S, FL>> ex =
                        dynamic_pointer_cast<OpSum<S, FL>>(expr->data[j]);
                    for (size_t k = 0; k < ex->strings.size(); k++)
                        if (ex->strings[k]->get_type() == OpTypes::SumProd) {
                            shared_ptr<OpSumProd<S, FL>> op =
                                dynamic_pointer_cast<OpSumProd<S, FL>>(
                                    ex->strings[k]);
                            assert(op->ops.size() != 0);
                            op->c = make_shared<OpElement<S, FL>>(
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
    void simplify(const vector<size_t> &left_op_sizes) {
        if (MPO<S, FL>::schemer != nullptr) {
            MPO<S, FL>::load_schemer();
            MPO<S, FL>::load_left_operators(
                MPO<S, FL>::schemer->left_trans_site);
            simplify_symbolic(
                MPO<S, FL>::schemer->left_new_operator_names,
                MPO<S, FL>::schemer->left_new_operator_exprs,
                MPO<S, FL>::left_operator_names[MPO<S, FL>::schemer
                                                    ->left_trans_site]);
            MPO<S, FL>::unload_left_operators(
                MPO<S, FL>::schemer->left_trans_site);
            MPO<S, FL>::load_right_operators(
                MPO<S, FL>::schemer->right_trans_site);
            simplify_symbolic(
                MPO<S, FL>::schemer->right_new_operator_names,
                MPO<S, FL>::schemer->right_new_operator_exprs,
                MPO<S, FL>::right_operator_names[MPO<S, FL>::schemer
                                                     ->right_trans_site]);
            MPO<S, FL>::unload_right_operators(
                MPO<S, FL>::schemer->right_trans_site);
            MPO<S, FL>::save_schemer();
            MPO<S, FL>::unload_schemer();
        }
        int ntg = threading->activate_global();
        vector<int> gidx(MPO<S, FL>::n_sites);
        for (int i = 0; i < MPO<S, FL>::n_sites; i++)
            gidx[i] = i;
        if (ntg != 1)
            sort(gidx.begin(), gidx.end(),
                 [this, &left_op_sizes](int i, int j) {
                     return left_op_sizes[i] > left_op_sizes[j];
                 });
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int ii = 0; ii < MPO<S, FL>::n_sites; ii++) {
            int i = gidx[ii];
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO SIM ... " << setw(4) << ii << " / " << setw(4)
                     << MPO<S, FL>::n_sites << endl;
            MPO<S, FL>::load_left_operators(i);
            simplify_symbolic(MPO<S, FL>::left_operator_names[i],
                              MPO<S, FL>::left_operator_exprs[i]);
            MPO<S, FL>::save_left_operators(i);
            MPO<S, FL>::unload_left_operators(i);
            MPO<S, FL>::load_right_operators(i);
            simplify_symbolic(MPO<S, FL>::right_operator_names[i],
                              MPO<S, FL>::right_operator_exprs[i]);
            MPO<S, FL>::save_right_operators(i);
            MPO<S, FL>::unload_right_operators(i);
            if (i < MPO<S, FL>::n_sites - 1) {
                MPO<S, FL>::load_middle_operators(i);
                shared_ptr<Symbolic<S>> mexpr =
                    MPO<S, FL>::middle_operator_exprs[i];
                for (size_t j = 0; j < mexpr->data.size(); j++)
                    mexpr->data[j] = simplify_expr(mexpr->data[j]);
                MPO<S, FL>::save_middle_operators(i);
                MPO<S, FL>::unload_middle_operators(i);
            }
        }
    }
    AncillaTypes get_ancilla_type() const override {
        return prim_mpo->get_ancilla_type();
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

} // namespace block2
