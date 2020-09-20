
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

#include "mpo.hpp"
#include "parallel_rule.hpp"
#include "parallel_tensor_functions.hpp"
#include <memory>

using namespace std;

namespace block2 {

template <typename S> struct ParallelMPO : MPO<S> {
    using MPO<S>::n_sites;
    shared_ptr<ParallelRule<S>> rule;
    shared_ptr<MPO<S>> prim_mpo;
    ParallelMPO(const shared_ptr<MPO<S>> &mpo,
                const shared_ptr<ParallelRule<S>> &rule)
        : MPO<S>(mpo->n_sites), prim_mpo(mpo), rule(rule) {
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        MPO<S>::const_e = mpo->const_e;
        MPO<S>::tensors = mpo->tensors;
        MPO<S>::basis = mpo->basis;
        MPO<S>::op = mpo->op;
        MPO<S>::tf =
            make_shared<ParallelTensorFunctions<S>>(mpo->tf->opf, rule);
        MPO<S>::site_op_infos = mpo->site_op_infos;
        MPO<S>::schemer = mpo->schemer;
        MPO<S>::sparse_form = mpo->sparse_form;
        MPO<S>::left_operator_names = mpo->left_operator_names;
        MPO<S>::right_operator_names = mpo->right_operator_names;
        MPO<S>::middle_operator_names = mpo->middle_operator_names;
        MPO<S>::left_operator_exprs = mpo->left_operator_exprs;
        MPO<S>::right_operator_exprs = mpo->right_operator_exprs;
        MPO<S>::middle_operator_exprs = mpo->middle_operator_exprs;
        for (size_t ix = 0; ix < MPO<S>::left_operator_exprs.size(); ix++) {
            auto &x = MPO<S>::left_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->localize_expr(
                    x->data[j],
                    rule->owner(MPO<S>::left_operator_names[ix]->data[j]));
            }
        }
        for (size_t ix = 0; ix < MPO<S>::right_operator_exprs.size(); ix++) {
            auto &x = MPO<S>::right_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->localize_expr(
                    x->data[j],
                    rule->owner(MPO<S>::right_operator_names[ix]->data[j]));
            }
        }
        for (size_t ix = 0; ix < MPO<S>::middle_operator_exprs.size(); ix++) {
            auto &x = MPO<S>::middle_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++) {
                assert(x->data[j]->get_type() != OpTypes::ExprRef);
                x->data[j] = rule->localize_expr(
                    x->data[j],
                    rule->owner(MPO<S>::middle_operator_names[ix]->data[j]));
            }
        }
        // this will change schemer in original mpo
        if (MPO<S>::schemer != nullptr) {
            MPO<S>::schemer = MPO<S>::schemer->copy();
            auto lx = MPO<S>::schemer->left_new_operator_exprs;
            for (size_t j = 0; j < lx->data.size(); j++) {
                assert(lx->data[j]->get_type() != OpTypes::ExprRef);
                lx->data[j] = rule->localize_expr(
                    lx->data[j],
                    rule->owner(
                        MPO<S>::schemer->left_new_operator_names->data[j]));
            }
            auto rx = MPO<S>::schemer->right_new_operator_exprs;
            for (size_t j = 0; j < rx->data.size(); j++) {
                assert(rx->data[j]->get_type() != OpTypes::ExprRef);
                rx->data[j] = rule->localize_expr(
                    rx->data[j],
                    rule->owner(
                        MPO<S>::schemer->right_new_operator_names->data[j]));
            }
        }
    }
    ParallelTypes get_parallel_type() const override {
        return rule->get_parallel_type();
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

} // namespace block2
