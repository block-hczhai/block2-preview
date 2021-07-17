
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
    int size, rank, root, group, grank, gsize, ngroup;
    ParallelTypes para_type = ParallelTypes::Serial;
    double tcomm = 0.0, tidle = 0.0, twait = 0.0; // Runtime for communication
    ParallelCommunicator()
        : size(1), rank(0), root(0), group(0), grank(0), gsize(1), ngroup(1) {}
    ParallelCommunicator(int size, int rank, int root)
        : size(size), rank(rank), root(root), group(0), grank(rank),
          gsize(size), ngroup(1) {}
    virtual ~ParallelCommunicator() = default;
    ParallelTypes get_parallel_type() const { return para_type; }
    virtual shared_ptr<ParallelCommunicator<S>> split(int igroup, int irank) {
        assert(false);
        return nullptr;
    }
    virtual void barrier() { assert(size == 1); }
    virtual void broadcast(const shared_ptr<SparseMatrix<S>> &mat, int owner) {
        assert(size == 1);
    }
    virtual void broadcast(double *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void broadcast(complex<double> *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void ibroadcast(const shared_ptr<SparseMatrix<S>> &mat, int owner) {
        assert(size == 1);
    }
    virtual void ibroadcast(double *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void broadcast(int *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void broadcast(long long int *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void allreduce_sum(double *data, size_t len) { assert(size == 1); }
    virtual void allreduce_sum(complex<double> *data, size_t len) {
        assert(size == 1);
    }
    virtual void allreduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat) {
        assert(size == 1);
    }
    virtual void allreduce_sum(const shared_ptr<SparseMatrix<S>> &mat) {
        assert(size == 1);
    }
    virtual void allreduce_sum(vector<S> &vs) { assert(size == 1); }
    virtual void allreduce_logical_or(char *data, size_t len) { assert(size == 1); }
    virtual void allreduce_xor(char *data, size_t len) { assert(size == 1); }
    virtual void allreduce_min(double *data, size_t len) { assert(size == 1); }
    virtual void allreduce_min(vector<vector<double>> &vs) {
        assert(size == 1);
    }
    virtual void allreduce_min(vector<double> &vs) { assert(size == 1); }
    virtual void allreduce_max(double *data, size_t len) { assert(size == 1); }
    virtual void allreduce_max(vector<double> &vs) { assert(size == 1); }
    virtual void reduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat,
                            int owner) {
        assert(size == 1);
    }
    virtual void reduce_sum(const shared_ptr<SparseMatrix<S>> &mat, int owner) {
        assert(size == 1);
    }
    virtual void ireduce_sum(const shared_ptr<SparseMatrix<S>> &mat,
                             int owner) {
        assert(size == 1);
    }
    virtual void reduce_sum(double *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void ireduce_sum(double *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void reduce_sum(uint64_t *data, size_t len, int owner) {
        assert(size == 1);
    }
    virtual void allreduce_logical_or(bool &v) { assert(size == 1); }
    virtual void waitall() { assert(size == 1); }
};

struct ParallelProperty {
    int owner;
    ParallelOpTypes ptype;
    ParallelProperty() : owner(0), ptype(ParallelOpTypes::Repeated) {}
    ParallelProperty(int owner, ParallelOpTypes ptype)
        : owner(owner), ptype(ptype) {}
};

enum struct ParallelCommTypes : uint8_t { None = 0, NonBlocking = 1 };

enum struct ParallelRulePartitionTypes : uint8_t { Left, Right, Middle };

inline bool operator&(ParallelCommTypes a, ParallelCommTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline ParallelCommTypes operator|(ParallelCommTypes a, ParallelCommTypes b) {
    return ParallelCommTypes((uint8_t)a | (uint8_t)b);
}

// Rule for parallel dispatcher
template <typename S> struct ParallelRule {
    shared_ptr<ParallelCommunicator<S>> comm;
    ParallelCommTypes comm_type;
    ParallelRule(const shared_ptr<ParallelCommunicator<S>> &comm,
                 ParallelCommTypes comm_type = ParallelCommTypes::None)
        : comm(comm), comm_type(comm_type) {
        assert(frame != nullptr);
        frame->prefix_distri = frame->prefix + Parsing::to_string(comm->rank);
        frame->prefix_can_write = comm->rank == comm->root;
    }
    virtual ~ParallelRule() = default;
    ParallelTypes get_parallel_type() const {
        return comm->get_parallel_type();
    }
    // For NPDM, the parallel rule can be different for different partition
    virtual void set_partition(ParallelRulePartitionTypes partition) const {}
    virtual shared_ptr<ParallelRule<S>> split(int gsize) const {
        int igroup = comm->rank / gsize, irank = comm->rank % gsize;
        assert(comm->size % gsize == 0);
        comm->ngroup = comm->size / gsize;
        comm->gsize = gsize;
        comm->group = igroup;
        comm->grank = irank;
        return make_shared<ParallelRule<S>>(comm->split(igroup, irank),
                                            comm_type);
    }
    virtual ParallelProperty
    operator()(const shared_ptr<OpElement<S>> &op) const {
        return ParallelProperty();
    }
    bool is_root() const noexcept { return comm->rank == comm->root; }
    bool available(const shared_ptr<OpExpr<S>> &op,
                   int node = -1) const noexcept {
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
    bool number(const shared_ptr<OpExpr<S>> &op) const noexcept {
        assert(op->get_type() == OpTypes::Elem);
        ParallelProperty pp = (*this)(dynamic_pointer_cast<OpElement<S>>(op));
        return pp.ptype & ParallelOpTypes::Number;
    }
    template <typename T>
    void distributed_apply(T f, const vector<shared_ptr<OpExpr<S>>> &ops,
                           const vector<shared_ptr<OpExpr<S>>> &exprs,
                           vector<shared_ptr<SparseMatrix<S>>> &mats) const {
        assert(ops.size() == exprs.size());
        vector<pair<shared_ptr<OpElement<S>>, shared_ptr<OpExprRef<S>>>>
            op_exprs(exprs.size());
        for (size_t i = 0; i < exprs.size(); i++) {
            if (i < mats.size() && mats[i] == nullptr)
                continue;
            shared_ptr<OpElement<S>> cop =
                dynamic_pointer_cast<OpElement<S>>(ops[i]);
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(abs_value(ops[i]));
            shared_ptr<OpExpr<S>> expr = exprs[i] * (1 / cop->factor);
            if (get_parallel_type() & ParallelTypes::NewScheme)
                assert(expr->get_type() == OpTypes::ExprRef);
            if (expr->get_type() != OpTypes::ExprRef) {
                op_exprs[i] = make_pair(
                    op, partial(op)
                            ? localize_expr(expr, owner(op))
                            : make_shared<OpExprRef<S>>(expr, true, expr));
            } else
                op_exprs[i] =
                    make_pair(op, dynamic_pointer_cast<OpExprRef<S>>(expr));
        }
        vector<shared_ptr<OpExpr<S>>> local_exprs(exprs.size());
        vector<shared_ptr<OpExpr<S>>> post_exprs = local_exprs;
        for (size_t i = 0; i < exprs.size(); i++) {
            if (i < mats.size() && mats[i] == nullptr)
                continue;
            shared_ptr<OpElement<S>> op = op_exprs[i].first;
            shared_ptr<OpExprRef<S>> expr_ref = op_exprs[i].second;
            bool req =
                partial(op) ? (expr_ref->is_local ? own(op) : true) : own(op);
            bool comm_req = (partial(op) && !expr_ref->is_local) || repeat(op);
            if (get_parallel_type() & ParallelTypes::NewScheme)
                req = comm_req = partial(op) || own(op) || repeat(op);
            if (req) {
                if (!(comm_type & ParallelCommTypes::NonBlocking) || comm_req)
                    local_exprs[i] = expr_ref->op;
                else
                    post_exprs[i] = expr_ref->op;
            }
        }
        f(local_exprs);
        if (get_parallel_type() & ParallelTypes::NewScheme)
            return;
        for (size_t i = 0; i < mats.size(); i++) {
            if (mats[i] == nullptr)
                continue;
            shared_ptr<OpElement<S>> op = op_exprs[i].first;
            shared_ptr<OpExprRef<S>> expr_ref = op_exprs[i].second;
            if (!(comm_type & ParallelCommTypes::NonBlocking)) {
                if (partial(op) && !expr_ref->is_local)
                    comm->reduce_sum(mats[i], (*this)(op).owner);
                if (repeat(op)) {
                    if (mats[i]->data == nullptr)
                        mats[i]->allocate(mats[i]->info);
                    comm->broadcast(mats[i], (*this)(op).owner);
                }
            } else {
                if (partial(op) && !expr_ref->is_local)
                    comm->ireduce_sum(mats[i], (*this)(op).owner);
                if (repeat(op)) {
                    if (mats[i]->data == nullptr)
                        mats[i]->allocate(mats[i]->info);
                    comm->ibroadcast(mats[i], (*this)(op).owner);
                }
            }
        }
        if (comm_type & ParallelCommTypes::NonBlocking) {
            f(post_exprs);
            comm->waitall();
        }
    }
    shared_ptr<OpExprRef<S>>
    modern_localize_expr(const shared_ptr<OpExpr<S>> &expr,
                         const shared_ptr<OpExpr<S>> &op, bool llocal,
                         bool rlocal) const {
        int xowner = owner(op);
        const shared_ptr<OpExprRef<S>> zero_ref =
            make_shared<OpExprRef<S>>(make_shared<OpExpr<S>>(), true, expr);
        shared_ptr<OpExprRef<S>> r =
            modern_localize_expr_owner(expr, xowner, llocal, rlocal);
        if (comm->rank == xowner || repeat(op))
            return r;
        else if (r->is_local)
            return zero_ref;
        else {
            r = modern_localize_expr_owner(expr, comm->rank, llocal, rlocal);
            return r;
        }
    }
    shared_ptr<OpExprRef<S>>
    modern_localize_expr_owner(const shared_ptr<OpExpr<S>> &expr, int owner,
                               bool llocal, bool rlocal) const {
        const shared_ptr<OpExprRef<S>> zero_ref =
            make_shared<OpExprRef<S>>(make_shared<OpExpr<S>>(), false, expr);
        switch (expr->get_type()) {
        case OpTypes::Zero:
            return zero_ref;
        case OpTypes::Elem:
            if (available(dynamic_pointer_cast<OpElement<S>>(expr), owner))
                return make_shared<OpExprRef<S>>(expr, true, expr);
            else
                return zero_ref;
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            if (op->b == nullptr) {
                if (available(dynamic_pointer_cast<OpElement<S>>(op->a), owner))
                    return make_shared<OpExprRef<S>>(expr, true, expr);
                else
                    return zero_ref;
            }
            if ((partial(op->a) && !llocal) || (partial(op->b) && !rlocal))
                return make_shared<OpExprRef<S>>(expr, false, expr);
            bool aa = available(op->a, owner), ab = available(op->b, owner);
            if (aa && ab)
                return make_shared<OpExprRef<S>>(expr, true, expr);
            else
                return zero_ref;
        }
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            if (op->a != nullptr && op->b != nullptr)
                // when using modern scheme, we do not need to re-parallelize
                // 3-operator operation formulas
                assert(false);
            else if (op->a != nullptr) {
                bool pa = partial(op->a) && !llocal,
                     aa = available(op->a, owner);
                if (!aa && !pa)
                    return zero_ref;
                else {
                    vector<shared_ptr<OpElement<S>>> ops;
                    vector<bool> conjs;
                    for (size_t i = 0; i < op->ops.size(); i++)
                        if (available(op->ops[i], owner) ||
                            (partial(op->ops[i]) && !rlocal))
                            ops.push_back(op->ops[i]),
                                conjs.push_back(op->conjs[i]);
                    if (ops.size() == 0)
                        return zero_ref;
                    else if (ops.size() == 1 && op->c == nullptr)
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpProduct<S>>(op->a, ops[0], op->factor,
                                                      op->conj ^
                                                          (conjs[0] << 1)),
                            ops.size() == op->ops.size(), expr);
                    else {
                        uint8_t cjx = op->conj;
                        if (conjs[0])
                            conjs.flip(), cjx ^= 1 << 1;
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpSumProd<S>>(op->a, ops, conjs,
                                                      op->factor, cjx, op->c),
                            ops.size() == op->ops.size(), expr);
                    }
                }
            } else {
                bool pb = partial(op->b) && !rlocal,
                     ab = available(op->b, owner);
                if (!ab && !pb)
                    return zero_ref;
                else {
                    vector<shared_ptr<OpElement<S>>> ops;
                    vector<bool> conjs;
                    for (size_t i = 0; i < op->ops.size(); i++)
                        if (available(op->ops[i], owner) ||
                            (partial(op->ops[i]) && !llocal))
                            ops.push_back(op->ops[i]),
                                conjs.push_back(op->conjs[i]);
                    if (ops.size() == 0)
                        return zero_ref;
                    else if (ops.size() == 1 && op->c == nullptr)
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpProduct<S>>(ops[0], op->b, op->factor,
                                                      op->conj ^
                                                          (uint8_t)conjs[0]),
                            ops.size() == op->ops.size(), expr);
                    else {
                        uint8_t cjx = op->conj;
                        if (conjs[0])
                            conjs.flip(), cjx ^= 1;
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpSumProd<S>>(ops, op->b, conjs,
                                                      op->factor, cjx, op->c),
                            ops.size() == op->ops.size(), expr);
                    }
                }
            }
        }
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            vector<shared_ptr<OpExpr<S>>> strings;
            strings.reserve(op->strings.size());
            bool is_local = true;
            for (size_t i = 0; i < op->strings.size(); i++) {
                shared_ptr<OpExprRef<S>> r = modern_localize_expr_owner(
                    op->strings[i], owner, llocal, rlocal);
                is_local = is_local && r->is_local;
                strings.push_back(r->op);
            }
            return make_shared<OpExprRef<S>>(sum(strings), is_local, expr);
        }
        default:
            assert(false);
        }
        return nullptr;
    }
    shared_ptr<OpExprRef<S>> localize_expr(const shared_ptr<OpExpr<S>> &expr,
                                           int owner, bool dleft = true) const {
        const shared_ptr<OpExprRef<S>> zero_ref =
            make_shared<OpExprRef<S>>(make_shared<OpExpr<S>>(), true, expr);
        shared_ptr<OpExprRef<S>> r = localize_expr_owner(expr, owner, dleft);
        if (comm->rank == owner)
            return r;
        else if (r->is_local)
            return zero_ref;
        else {
            r = localize_expr_owner(expr, comm->rank, dleft);
            // some R operator only have components on other nodes
            // under this case, the components should be transferred
            // to its owner node
            r->is_local = false;
            return r;
        }
    }
    shared_ptr<OpExprRef<S>>
    localize_expr_owner(const shared_ptr<OpExpr<S>> &expr, int owner,
                        bool dleft = true) const {
        const shared_ptr<OpExprRef<S>> zero_ref =
            make_shared<OpExprRef<S>>(make_shared<OpExpr<S>>(), false, expr);
        if (expr->get_type() == OpTypes::Zero)
            return make_shared<OpExprRef<S>>(expr, true, expr);
        else if (expr->get_type() == OpTypes::Elem) {
            if (available(dynamic_pointer_cast<OpElement<S>>(expr), owner))
                return make_shared<OpExprRef<S>>(expr, true, expr);
            else
                return zero_ref;
        } else if (expr->get_type() == OpTypes::Prod) {
            shared_ptr<OpProduct<S>> op =
                dynamic_pointer_cast<OpProduct<S>>(expr);
            bool aa = available(op->a, owner),
                 ab = op->b == nullptr ? true : available(op->b, owner);
            if (aa && ab)
                return make_shared<OpExprRef<S>>(expr, true, expr);
            else
                return zero_ref;
        } else if (expr->get_type() == OpTypes::SumProd) {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            if (op->a != nullptr && op->b != nullptr) {
                bool aa = available(op->a, owner), ab = available(op->b, owner);
                bool pda = op->ops[0]->name == OpNames::TEMP,
                     pdb = op->ops[1]->name == OpNames::TEMP;
                bool ada = available(op->ops[0], owner) || pda,
                     adb = available(op->ops[1], owner) || pdb;
                if ((dleft && ada && adb && ab) || (!dleft && ada && adb && aa))
                    return make_shared<OpExprRef<S>>(expr, !(pda || pdb), expr);
                else
                    return zero_ref;
            } else if (op->a != nullptr) {
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
                    else if (ops.size() == 1 && op->c == nullptr)
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpProduct<S>>(op->a, ops[0], op->factor,
                                                      op->conj ^
                                                          (conjs[0] << 1)),
                            ops.size() == op->ops.size(), expr);
                    else {
                        uint8_t cjx = op->conj;
                        if (conjs[0])
                            conjs.flip(), cjx ^= 1 << 1;
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpSumProd<S>>(op->a, ops, conjs,
                                                      op->factor, cjx, op->c),
                            ops.size() == op->ops.size(), expr);
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
                    else if (ops.size() == 1 && op->c == nullptr)
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpProduct<S>>(ops[0], op->b, op->factor,
                                                      op->conj ^
                                                          (uint8_t)conjs[0]),
                            ops.size() == op->ops.size(), expr);
                    else {
                        uint8_t cjx = op->conj;
                        if (conjs[0])
                            conjs.flip(), cjx ^= 1;
                        return make_shared<OpExprRef<S>>(
                            make_shared<OpSumProd<S>>(ops, op->b, conjs,
                                                      op->factor, cjx, op->c),
                            ops.size() == op->ops.size(), expr);
                    }
                }
            }
        } else if (expr->get_type() == OpTypes::Sum) {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            vector<shared_ptr<OpExpr<S>>> strings;
            bool is_local = true;
            for (size_t i = 0; i < op->strings.size(); i++) {
                shared_ptr<OpExprRef<S>> r =
                    localize_expr_owner(op->strings[i], owner, dleft);
                is_local = is_local && r->is_local;
                strings.push_back(r->op);
            }
            return make_shared<OpExprRef<S>>(sum(strings), is_local, expr);
        } else {
            assert(false);
            return nullptr;
        }
    }
};

} // namespace block2
