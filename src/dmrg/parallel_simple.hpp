
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

#include "../core/integral.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/rule.hpp"
#include <algorithm>
#include <memory>

using namespace std;

namespace block2 {

// Operator names
enum struct ParallelSimpleTypes : uint8_t { None, I, J, IJ, KL };

// Rule for parallel dispatcher for quantum chemistry MPO
template <typename S, typename FL>
struct ParallelRuleSimple : ParallelRule<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using ParallelRule<S, FL>::comm;
    ParallelSimpleTypes mode;
    ParallelRuleSimple(ParallelSimpleTypes mode,
                       const shared_ptr<ParallelCommunicator<S>> &comm,
                       ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S, FL>(comm, comm_type), mode(mode) {
        comm->para_type = comm->para_type | ParallelTypes::Simple;
    }
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S, FL>::split(gsize);
        return make_shared<ParallelRuleSimple>(mode, r->comm, r->comm_type);
    }
    ParallelProperty
    operator()(const shared_ptr<OpElement<S, FL>> &op) const override {
        return ParallelProperty(comm->rank, ParallelOpTypes::None);
    }
    FL index_prefactor(uint16_t i, uint16_t j) const noexcept {
        switch (mode) {
        case ParallelSimpleTypes::I:
            return (FL)(FP)(comm->rank == i % comm->size);
        case ParallelSimpleTypes::J:
            return (FL)(FP)(comm->rank == j % comm->size);
        case ParallelSimpleTypes::IJ:
        case ParallelSimpleTypes::KL:
            return (FL)0.5 * ((FL)(FP)(comm->rank == i % comm->size) +
                              (FL)(FP)(comm->rank == j % comm->size));
        case ParallelSimpleTypes::None:
            return (FL)(FP)(1.0);
        }
        return (FL)0.0;
    }
    FL index_prefactor(uint16_t i, uint16_t j, uint16_t k,
                       uint16_t l) const noexcept {
        vector<uint16_t> idx{i, j, k, l};
        sort(idx.begin(), idx.end());
        const uint16_t ii = idx[0], jj = idx[1], kk = idx[2], ll = idx[3];
        switch (mode) {
        case ParallelSimpleTypes::I:
            return (FL)(FP)(comm->rank == i % comm->size);
        case ParallelSimpleTypes::J:
            return (FL)(FP)(comm->rank == j % comm->size);
        case ParallelSimpleTypes::IJ:
            return jj == kk
                       ? (FL)(FP)(comm->rank == jj % comm->size)
                       : (FL)(FP)(comm->rank ==
                                  (ii <= jj ? (int)jj * (jj + 1) / 2 + ii
                                            : (int)ii * (ii + 1) / 2 + jj) %
                                      comm->size);
        case ParallelSimpleTypes::KL:
            return jj == kk
                       ? (FL)(FP)(comm->rank == kk % comm->size)
                       : (FL)(FP)(comm->rank ==
                                  (kk <= ll ? (int)ll * (ll + 1) / 2 + kk
                                            : (int)kk * (kk + 1) / 2 + ll) %
                                      comm->size);
        case ParallelSimpleTypes::None:
            return (FL)(FP)(1.0);
        }
        return (FL)0.0;
    }
};

// One- and two-electron integrals
// distriubed over mpi procs
template <typename S, typename FL> struct ParallelFCIDUMP : FCIDUMP<FL> {
    shared_ptr<FCIDUMP<FL>> fcidump;
    shared_ptr<ParallelRuleSimple<S, FL>> rule;
    using FCIDUMP<FL>::params;
    ParallelFCIDUMP(const shared_ptr<FCIDUMP<FL>> &fcidump,
                    const shared_ptr<ParallelRuleSimple<S, FL>> &rule)
        : fcidump(fcidump), rule(rule) {
        params = fcidump->params;
    }
    virtual ~ParallelFCIDUMP() = default;
    // One-electron integral element (SU(2))
    FL t(uint16_t i, uint16_t j) const override {
        return rule->index_prefactor(i, j) * fcidump->t(i, j);
    }
    // One-electron integral element (SZ)
    FL t(uint8_t s, uint16_t i, uint16_t j) const override {
        return rule->index_prefactor(i, j) * fcidump->t(s, i, j);
    }
    // Two-electron integral element (SU(2))
    FL v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        return rule->index_prefactor(i, j, k, l) * fcidump->v(i, j, k, l);
    }
    // Two-electron integral element (SZ)
    FL v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
         uint16_t l) const override {
        return rule->index_prefactor(i, j, k, l) *
               fcidump->v(sl, sr, i, j, k, l);
    }
    typename const_fl_type<FL>::FL e() const override { return fcidump->e(); }
    void deallocate() override { fcidump->deallocate(); }
};

// Symmetry rules for simplifying quantum chemistry sum MPO (non-spin-adapted)
template <typename S, typename FL> struct SumMPORule : Rule<S, FL> {
    shared_ptr<Rule<S, FL>> prim_rule;
    shared_ptr<ParallelRuleSimple<S, FL>> para_rule;
    SumMPORule(const shared_ptr<Rule<S, FL>> &rule,
               const shared_ptr<ParallelRuleSimple<S, FL>> &para_rule)
        : prim_rule(rule), para_rule(para_rule) {}
    shared_ptr<OpElementRef<S, FL>>
    operator()(const shared_ptr<OpElement<S, FL>> &op) const override {
        if (op->site_index.size() == 1 ||
            (op->site_index.size() == 2 &&
             para_rule->index_prefactor(op->site_index[0], op->site_index[1]) !=
                 (FL)0.0 &&
             para_rule->index_prefactor(op->site_index[1], op->site_index[0]) !=
                 (FL)0.0))
            return prim_rule->operator()(op);
        else
            return nullptr;
    }
};

} // namespace block2
