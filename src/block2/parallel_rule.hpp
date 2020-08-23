
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

#include "expr.hpp"
#include "sparse_matrix.hpp"
#include <memory>

using namespace std;

namespace block2 {

enum ParallelOpTypes : uint8_t {
    None = 0,
    Repeated = 1,
    Number = 2,
    Partial = 4
};

template <typename S> struct ParallelCommunicator {
    int size, rank, root;
    double tcomm = 0.0;
    ParallelCommunicator() : size(1), rank(0), root(0) {}
    ParallelCommunicator(int size, int rank, int root)
        : size(size), rank(rank), root(root) {}
    virtual ParallelTypes get_parallel_type() const {
        return ParallelTypes::Serial;
    }
    virtual void barrier() { assert(size == 1); }
    virtual void broadcast(const shared_ptr<SparseMatrix<S>> &mat, int owner) {
        assert(size == 1);
    }
    virtual void broadcast(double *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void allreduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat) {
        assert(size == 1);
    }
    virtual void allreduce_sum(const shared_ptr<SparseMatrix<S>> &mat) {
        assert(size == 1);
    }
    virtual void allreduce_sum(vector<S> &vs) { assert(size == 1); }
    virtual void reduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat,
                            int owner) {
        assert(size == 1);
    }
    virtual void reduce_sum(const shared_ptr<SparseMatrix<S>> &mat, int owner) {
        assert(size == 1);
    }
    virtual void reduce_sum(double *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void allreduce_logical_or(bool &v) { assert(size == 1); }
};

struct ParallelProperty {
    int owner;
    ParallelOpTypes ptype;
    ParallelProperty() : owner(0), ptype(ParallelOpTypes::Repeated) {}
    ParallelProperty(int owner, ParallelOpTypes ptype)
        : owner(owner), ptype(ptype) {}
};

// Rule for parallel dispatcher
template <typename S> struct ParallelRule {
    shared_ptr<ParallelCommunicator<S>> comm;
    ParallelRule(const shared_ptr<ParallelCommunicator<S>> &comm) : comm(comm) {
        assert(frame != nullptr);
        frame->prefix_distri = frame->prefix + Parsing::to_string(comm->rank);
        if (comm->rank != comm->root)
            frame->prefix_can_write = false;
    }
    ParallelTypes get_parallel_type() const {
        return comm->get_parallel_type();
    }
    virtual ParallelProperty
    operator()(const shared_ptr<OpElement<S>> &op) const {
        return ParallelProperty();
    }
    bool is_root() const noexcept { return comm->rank == comm->root; }
    bool available(const shared_ptr<OpExpr<S>> &op, int node = -1) const
        noexcept {
        return (node == -1 ? own(op) : owner(op) == node) || repeat(op);
    }
    bool own(const shared_ptr<OpExpr<S>> &op) const noexcept {
        assert(op->get_type() == OpTypes::Elem);
        ParallelProperty pp = (*this)(dynamic_pointer_cast<OpElement<S>>(op));
        return pp.owner == comm->rank;
    }
    int owner(const shared_ptr<OpExpr<S>> &op) const noexcept {
        assert(op->get_type() == OpTypes::Elem);
        ParallelProperty pp = (*this)(dynamic_pointer_cast<OpElement<S>>(op));
        return pp.owner;
    }
    bool repeat(const shared_ptr<OpExpr<S>> &op) const noexcept {
        assert(op->get_type() == OpTypes::Elem);
        ParallelProperty pp = (*this)(dynamic_pointer_cast<OpElement<S>>(op));
        return pp.ptype & ParallelOpTypes::Repeated;
    }
    bool partial(const shared_ptr<OpExpr<S>> &op) const noexcept {
        assert(op->get_type() == OpTypes::Elem);
        ParallelProperty pp = (*this)(dynamic_pointer_cast<OpElement<S>>(op));
        return pp.ptype & ParallelOpTypes::Partial;
    }
    template <typename EvalOp, typename PostOp>
    void parallel_apply(EvalOp f, PostOp g,
                        const vector<shared_ptr<OpExpr<S>>> &ops,
                        const vector<shared_ptr<OpExpr<S>>> &exprs,
                        vector<shared_ptr<SparseMatrix<S>>> &mats) const {
        assert(ops.size() == exprs.size());
        vector<pair<shared_ptr<OpElement<S>>, shared_ptr<OpExprRef<S>>>>
            op_exprs(exprs.size());
        for (size_t i = 0; i < exprs.size(); i++) {
            shared_ptr<OpElement<S>> cop =
                dynamic_pointer_cast<OpElement<S>>(ops[i]);
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(abs_value(ops[i]));
            shared_ptr<OpExpr<S>> expr = exprs[i] * (1 / cop->factor);
            if (expr->get_type() != OpTypes::ExprRef) {
                op_exprs[i] = make_pair(
                    op, partial(op) ? localize_expr(expr, owner(op))
                                    : make_shared<OpExprRef<S>>(expr, true));
            } else
                op_exprs[i] =
                    make_pair(op, dynamic_pointer_cast<OpExprRef<S>>(expr));
        }
        for (size_t i = 0; i < exprs.size(); i++) {
            shared_ptr<OpElement<S>> op = op_exprs[i].first;
            shared_ptr<OpExprRef<S>> expr_ref = op_exprs[i].second;
            bool req =
                partial(op) ? (expr_ref->is_local ? own(op) : true) : own(op);
            if (req) {
                f(expr_ref->op, mats[i]);
            }
        }
        g();
        for (size_t i = 0; i < exprs.size(); i++) {
            shared_ptr<OpElement<S>> op = op_exprs[i].first;
            shared_ptr<OpExprRef<S>> expr_ref = op_exprs[i].second;
            if (partial(op) && !expr_ref->is_local)
                comm->reduce_sum(mats[i], (*this)(op).owner);
            if (repeat(op)) {
                if (mats[i]->data == nullptr)
                    mats[i]->allocate(mats[i]->info);
                comm->broadcast(mats[i], (*this)(op).owner);
            }
        }
    }
    shared_ptr<OpExprRef<S>> localize_expr(const shared_ptr<OpExpr<S>> &expr,
                                           int owner) const {
        const shared_ptr<OpExprRef<S>> zero_ref =
            make_shared<OpExprRef<S>>(make_shared<OpExpr<S>>(), true);
        shared_ptr<OpExprRef<S>> r = localize_expr_owner(expr, owner);
        if (comm->rank == owner)
            return r;
        else if (r->is_local)
            return zero_ref;
        else {
            r = localize_expr_owner(expr, comm->rank);
            // some R operator only have components on other nodes
            // under this case, the components should be transferred
            // to its owner node
            r->is_local = false;
            return r;
        }
    }
    shared_ptr<OpExprRef<S>>
    localize_expr_owner(const shared_ptr<OpExpr<S>> &expr, int owner) const {
        const shared_ptr<OpExprRef<S>> zero_ref =
            make_shared<OpExprRef<S>>(make_shared<OpExpr<S>>(), false);
        if (expr->get_type() == OpTypes::Zero)
            return make_shared<OpExprRef<S>>(expr, true);
        else if (expr->get_type() == OpTypes::Elem) {
            if (available(dynamic_pointer_cast<OpElement<S>>(expr), owner))
                return make_shared<OpExprRef<S>>(expr, true);
            else
                return zero_ref;
        } else if (expr->get_type() == OpTypes::Prod) {
            shared_ptr<OpString<S>> op =
                dynamic_pointer_cast<OpString<S>>(expr);
            bool aa = available(op->a, owner),
                 ab = op->b == nullptr ? true : available(op->b, owner);
            if (aa && ab)
                return make_shared<OpExprRef<S>>(expr, true);
            else
                return zero_ref;
        } else if (expr->get_type() == OpTypes::SumProd) {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            if (op->a != nullptr) {
                if (!available(op->a, owner))
                    return zero_ref;
                else {
                    vector<shared_ptr<OpElement<S>>> ops;
                    vector<bool> conjs;
                    for (size_t i = 0; i < op->ops.size(); i++)
                        if (available(op->ops[i], owner))
                            ops.push_back(op->ops[i]),
                                conjs.push_back(op->conjs[i]);
                    if (ops.size() == 0)
                        return zero_ref;
                    else if (ops.size() == 1)
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpString<S>>(op->a, ops[0], op->factor,
                                                     op->conj ^
                                                         (conjs[0] << 1)),
                            ops.size() == op->ops.size());
                    else {
                        uint8_t cjx = op->conj;
                        if (conjs[0])
                            conjs.flip(), cjx ^= 1 << 1;
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpSumProd<S>>(op->a, ops, conjs,
                                                      op->factor, cjx),
                            ops.size() == op->ops.size());
                    }
                }
            } else {
                if (!available(op->b, owner))
                    return zero_ref;
                else {
                    vector<shared_ptr<OpElement<S>>> ops;
                    vector<bool> conjs;
                    for (size_t i = 0; i < op->ops.size(); i++)
                        if (available(op->ops[i], owner))
                            ops.push_back(op->ops[i]),
                                conjs.push_back(op->conjs[i]);
                    if (ops.size() == 0)
                        return zero_ref;
                    else if (ops.size() == 1)
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpString<S>>(ops[0], op->b, op->factor,
                                                     op->conj ^ conjs[0]),
                            ops.size() == op->ops.size());
                    else {
                        uint8_t cjx = op->conj;
                        if (conjs[0])
                            conjs.flip(), cjx ^= 1;
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpSumProd<S>>(ops, op->b, conjs,
                                                      op->factor, cjx),
                            ops.size() == op->ops.size());
                    }
                }
            }
        } else if (expr->get_type() == OpTypes::Sum) {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            vector<shared_ptr<OpExpr<S>>> strings;
            bool is_local = true;
            for (size_t i = 0; i < op->strings.size(); i++) {
                shared_ptr<OpExprRef<S>> r =
                    localize_expr_owner(op->strings[i], owner);
                is_local = is_local && r->is_local;
                strings.push_back(r->op);
            }
            return make_shared<OpExprRef<S>>(sum(strings), is_local);
        } else {
            assert(false);
            return nullptr;
        }
    }
};

} // namespace block2
