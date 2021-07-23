
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "../core/expr.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/rule.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/delayed_sparse_matrix.hpp"
#include "../core/state_info.hpp"
#include <memory>
#include <unordered_map>

using namespace std;

namespace block2 {

/** The interface for a big site. */
template <typename S> struct BigSite {
    int n_orbs; //!< Spatial orbitals in the big site
    shared_ptr<StateInfo<S>> basis;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> op_infos;
    BigSite(int n_orbs) : n_orbs(n_orbs) {}
    virtual ~BigSite() = default;
    virtual void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const {}
    // Find sparse matrix info for site operator with the given delta quantum q
    shared_ptr<SparseMatrixInfo<S>> find_site_op_info(S q) const {
        auto p = lower_bound(op_infos.begin(), op_infos.end(), q,
                             SparseMatrixInfo<S>::cmp_op_info);
        if (p == op_infos.end() || p->first != q)
            return nullptr;
        else
            return p->second;
    }
};

template <typename S> struct SimplifiedBigSite : BigSite<S> {
    shared_ptr<BigSite<S>> big_site;
    shared_ptr<Rule<S>> rule;
    SimplifiedBigSite(const shared_ptr<BigSite<S>> &big_site,
                      const shared_ptr<Rule<S>> &rule)
        : BigSite<S>(*big_site), big_site(big_site), rule(rule) {}
    virtual ~SimplifiedBigSite() = default;
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override {
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> kops;
        kops.reserve(ops.size());
        for (auto &p : ops) {
            assert(p.second == nullptr);
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            if ((*rule)(op) != nullptr)
                p.second = make_shared<DelayedSparseMatrix<S, OpExpr<S>>>(
                    m, p.first, big_site->find_site_op_info(op->q_label));
            else
                kops[p.first] = nullptr;
        }
        big_site->get_site_ops(m, kops);
        for (auto &p : ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            if (p.second != nullptr) {
                shared_ptr<OpElement<S>> ref_op = (*rule)(op)->op;
                if (kops.count(ref_op) && (kops.at(ref_op)->factor == 0 ||
                                           kops.at(ref_op)->norm() < TINY))
                    p.second->factor = 0;
            } else
                p.second = kops.at(p.first);
        }
    }
};

template <typename S> struct ParallelBigSite : BigSite<S> {
    shared_ptr<BigSite<S>> big_site;
    shared_ptr<ParallelRule<S>> rule;
    ParallelBigSite(const shared_ptr<BigSite<S>> &big_site,
                    const shared_ptr<ParallelRule<S>> &rule)
        : BigSite<S>(*big_site), big_site(big_site), rule(rule) {}
    virtual ~ParallelBigSite() = default;
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override {
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> kops;
        kops.reserve(ops.size());
        for (auto &p : ops) {
            assert(p.second == nullptr);
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            if (!rule->available(op))
                p.second = make_shared<DelayedSparseMatrix<S, OpExpr<S>>>(
                    m, p.first, big_site->find_site_op_info(op->q_label));
            else
                kops[p.first] = nullptr;
        }
        big_site->get_site_ops(m, kops);
        for (auto &p : kops)
            ops[p.first] = p.second;
        vector<char> is_zero(ops.size());
        int ii = 0;
        for (auto &p : ops)
            is_zero[ii++] = p.second->factor == 0;
        rule->comm->allreduce_logical_or(is_zero.data(), is_zero.size());
        ii = 0;
        for (auto &p : ops)
            if (is_zero[ii++])
                p.second->factor = 0;
    }
};

} // namespace block2
