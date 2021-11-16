
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

#include "../core/parallel_rule.hpp"
#include "../core/parallel_tensor_functions.hpp"
#include "mpo.hpp"
#include <memory>

using namespace std;

namespace block2 {

template <typename S, typename FL> struct ClassicParallelMPO : MPO<S, FL> {
    using MPO<S, FL>::n_sites;
    shared_ptr<ParallelRule<S, FL>> rule;
    shared_ptr<MPO<S, FL>> prim_mpo;
    ClassicParallelMPO(const shared_ptr<MPO<S, FL>> &mpo,
                       const shared_ptr<ParallelRule<S, FL>> &rule)
        : MPO<S, FL>(mpo->n_sites), prim_mpo(mpo), rule(rule) {
        if (rule->comm->para_type & ParallelTypes::NewScheme)
            rule->comm->para_type =
                rule->comm->para_type ^ ParallelTypes::NewScheme;
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::tensors = mpo->tensors;
        MPO<S, FL>::basis = mpo->basis;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::tf =
            make_shared<ParallelTensorFunctions<S, FL>>(mpo->tf->opf, rule);
        MPO<S, FL>::site_op_infos = mpo->site_op_infos;
        MPO<S, FL>::schemer = mpo->schemer;
        MPO<S, FL>::sparse_form = mpo->sparse_form;
        MPO<S, FL>::left_operator_names = mpo->left_operator_names;
        MPO<S, FL>::right_operator_names = mpo->right_operator_names;
        MPO<S, FL>::middle_operator_names = mpo->middle_operator_names;
        MPO<S, FL>::left_operator_exprs = mpo->left_operator_exprs;
        MPO<S, FL>::right_operator_exprs = mpo->right_operator_exprs;
        MPO<S, FL>::middle_operator_exprs = mpo->middle_operator_exprs;
        rule->set_partition(ParallelRulePartitionTypes::Left);
        for (size_t ix = 0; ix < MPO<S, FL>::left_operator_exprs.size(); ix++) {
            auto &x = MPO<S, FL>::left_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->localize_expr(
                    x->data[j],
                    rule->owner(MPO<S, FL>::left_operator_names[ix]->data[j]));
            }
        }
        rule->set_partition(ParallelRulePartitionTypes::Right);
        for (size_t ix = 0; ix < MPO<S, FL>::right_operator_exprs.size();
             ix++) {
            auto &x = MPO<S, FL>::right_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->localize_expr(
                    x->data[j],
                    rule->owner(MPO<S, FL>::right_operator_names[ix]->data[j]));
            }
        }
        rule->set_partition(ParallelRulePartitionTypes::Middle);
        for (size_t ix = 0; ix < MPO<S, FL>::middle_operator_exprs.size();
             ix++) {
            auto &x = MPO<S, FL>::middle_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->localize_expr(
                    x->data[j],
                    rule->owner(
                        MPO<S, FL>::middle_operator_names[ix]->data[j]));
            }
        }
        // this will change schemer in original mpo
        if (MPO<S, FL>::schemer != nullptr) {
            MPO<S, FL>::schemer = MPO<S, FL>::schemer->copy();
            auto lx = MPO<S, FL>::schemer->left_new_operator_exprs;
            for (size_t j = 0; j < lx->data.size(); j++) {
                assert(lx->data[j]->get_type() != OpTypes::ExprRef);
                lx->data[j] = rule->localize_expr(
                    lx->data[j],
                    rule->owner(
                        MPO<S, FL>::schemer->left_new_operator_names->data[j]));
            }
            auto rx = MPO<S, FL>::schemer->right_new_operator_exprs;
            for (size_t j = 0; j < rx->data.size(); j++) {
                assert(rx->data[j]->get_type() != OpTypes::ExprRef);
                rx->data[j] = rule->localize_expr(
                    rx->data[j],
                    rule->owner(MPO<S, FL>::schemer->right_new_operator_names
                                    ->data[j]));
            }
        }
    }
    ParallelTypes get_parallel_type() const override {
        return rule->get_parallel_type();
    }
    void deallocate() override { prim_mpo->deallocate(); }
    shared_ptr<MPO<S, FL>> scalar_multiply(FL d) const override {
        shared_ptr<MPO<S, FL>> rmpo = make_shared<ClassicParallelMPO>(*this);
        assert(rmpo->middle_operator_exprs.size() != 0);
        for (size_t ix = 0; ix < rmpo->middle_operator_exprs.size(); ix++) {
            auto &x = rmpo->middle_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++)
                x->data[j] = d * x->data[j];
        }
        rmpo->const_e = d * rmpo->const_e;
        return rmpo;
    }
};

template <typename S, typename FL> struct ParallelMPO : MPO<S, FL> {
    using MPO<S, FL>::n_sites;
    shared_ptr<ParallelRule<S, FL>> rule;
    shared_ptr<MPO<S, FL>> prim_mpo;
    ParallelMPO(int n_sites, const shared_ptr<ParallelRule<S, FL>> &rule)
        : MPO<S, FL>(n_sites), rule(rule), prim_mpo(nullptr) {
        rule->comm->para_type =
            rule->comm->para_type | ParallelTypes::NewScheme;
    }
    ParallelMPO(const shared_ptr<MPO<S, FL>> &mpo,
                const shared_ptr<ParallelRule<S, FL>> &rule)
        : MPO<S, FL>(mpo->n_sites), prim_mpo(mpo), rule(rule) {
        rule->comm->para_type =
            rule->comm->para_type | ParallelTypes::NewScheme;
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        // cannot parallelize archived mpo
        // one should archive parallelized mpo instead
        assert(mpo->archive_filename == "");
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::tensors = mpo->tensors;
        MPO<S, FL>::basis = mpo->basis;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::tf =
            make_shared<ParallelTensorFunctions<S, FL>>(mpo->tf->opf, rule);
        MPO<S, FL>::site_op_infos = mpo->site_op_infos;
        MPO<S, FL>::schemer = mpo->schemer;
        MPO<S, FL>::sparse_form = mpo->sparse_form;
        MPO<S, FL>::left_operator_names = mpo->left_operator_names;
        MPO<S, FL>::right_operator_names = mpo->right_operator_names;
        MPO<S, FL>::middle_operator_names = mpo->middle_operator_names;
        MPO<S, FL>::left_operator_exprs = mpo->left_operator_exprs;
        MPO<S, FL>::right_operator_exprs = mpo->right_operator_exprs;
        MPO<S, FL>::middle_operator_exprs = mpo->middle_operator_exprs;
        rule->set_partition(ParallelRulePartitionTypes::Left);
        for (size_t ix = 0; ix < MPO<S, FL>::left_operator_exprs.size(); ix++) {
            auto &x = MPO<S, FL>::left_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->modern_localize_expr(
                    x->data[j], MPO<S, FL>::left_operator_names[ix]->data[j],
                    ix <= 1, true);
            }
        }
        rule->set_partition(ParallelRulePartitionTypes::Right);
        for (size_t ix = 0; ix < MPO<S, FL>::right_operator_exprs.size();
             ix++) {
            auto &x = MPO<S, FL>::right_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->modern_localize_expr(
                    x->data[j], MPO<S, FL>::right_operator_names[ix]->data[j],
                    true, ix >= MPO<S, FL>::right_operator_exprs.size() - 2);
            }
        }
        rule->set_partition(ParallelRulePartitionTypes::Middle);
        for (size_t ix = 0; ix < MPO<S, FL>::middle_operator_exprs.size();
             ix++) {
            auto &x = MPO<S, FL>::middle_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->modern_localize_expr(
                    x->data[j], MPO<S, FL>::middle_operator_names[ix]->data[j],
                    false, false);
            }
        }
        // this will change schemer in original mpo
        if (MPO<S, FL>::schemer != nullptr) {
            MPO<S, FL>::schemer = MPO<S, FL>::schemer->copy();
            auto lx = MPO<S, FL>::schemer->left_new_operator_exprs;
            for (size_t j = 0; j < lx->data.size(); j++) {
                assert(lx->data[j]->get_type() != OpTypes::ExprRef);
                lx->data[j] = rule->modern_localize_expr(
                    lx->data[j],
                    MPO<S, FL>::schemer->left_new_operator_names->data[j],
                    false, false);
            }
            auto rx = MPO<S, FL>::schemer->right_new_operator_exprs;
            for (size_t j = 0; j < rx->data.size(); j++) {
                assert(rx->data[j]->get_type() != OpTypes::ExprRef);
                rx->data[j] = rule->modern_localize_expr(
                    rx->data[j],
                    MPO<S, FL>::schemer->right_new_operator_names->data[j],
                    false, false);
            }
        }
    }
    ParallelTypes get_parallel_type() const override {
        return rule->get_parallel_type();
    }
    void deallocate() override {
        if (prim_mpo != nullptr)
            prim_mpo->deallocate();
    }
    void load_data(istream &ifs, bool minimal = false) override {
        MPO<S, FL>::load_data(ifs, minimal);
        MPO<S, FL>::tf = make_shared<ParallelTensorFunctions<S, FL>>(
            MPO<S, FL>::tf->opf, rule);
    }
    shared_ptr<MPO<S, FL>> scalar_multiply(FL d) const override {
        shared_ptr<MPO<S, FL>> rmpo = make_shared<ParallelMPO>(*this);
        assert(rmpo->middle_operator_exprs.size() != 0);
        for (size_t ix = 0; ix < rmpo->middle_operator_exprs.size(); ix++) {
            auto &x = rmpo->middle_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++)
                x->data[j] = d * x->data[j];
        }
        rmpo->const_e = d * rmpo->const_e;
        return rmpo;
    }
};

} // namespace block2
