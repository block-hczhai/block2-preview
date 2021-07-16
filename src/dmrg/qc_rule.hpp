
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

#include "../core/expr.hpp"
#include "../core/rule.hpp"
#include <memory>

using namespace std;

namespace block2 {

template <typename, typename = void> struct RuleQC;

// Symmetry rules for simplifying quantum chemistry MPO (non-spin-adapted)
template <typename S> struct RuleQC<S, typename S::is_sz_t> : Rule<S> {
    uint8_t mask;
    const static uint8_t D = 0U, R = 1U, A = 2U, P = 3U, B = 4U, Q = 5U;
    RuleQC(bool d = true, bool r = true, bool a = true, bool p = true,
           bool b = true, bool q = true)
        : mask((d << D) | (r << R) | (a << A) | (p << P) | (b << B) |
               (q << Q)) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        switch (op->name) {
        case OpNames::D:
            return (mask & (1 << D)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::C, op->site_index,
                                               -op->q_label, op->factor),
                                           true, 1)
                                     : nullptr;
        case OpNames::RD:
            return (mask & (1 << R)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::R, op->site_index,
                                               -op->q_label, op->factor),
                                           true, 1)
                                     : nullptr;
        case OpNames::A:
            return (mask & (1 << A)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(OpNames::A,
                                                       op->site_index.flip(),
                                                       op->q_label, op->factor),
                             false, -1)
                       : nullptr;
        case OpNames::AD:
            return (mask & (1 << A))
                       ? (op->site_index[0] <= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A, op->site_index,
                                        -op->q_label, op->factor),
                                    true, 1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A, op->site_index.flip(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::P:
            return (mask & (1 << P)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(OpNames::P,
                                                       op->site_index.flip(),
                                                       op->q_label, op->factor),
                             false, -1)
                       : nullptr;
        case OpNames::PD:
            return (mask & (1 << P))
                       ? (op->site_index[0] <= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P, op->site_index,
                                        -op->q_label, op->factor),
                                    true, 1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P, op->site_index.flip(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::B:
            return (mask & (1 << B)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::B, op->site_index.flip(),
                                 -op->q_label, op->factor),
                             true, 1)
                       : nullptr;
        // BD with site index i == j cannot be represented by B
        case OpNames::BD:
            return ((mask & (1 << B)) &&
                    (op->site_index[0] != op->site_index[1]))
                       ? (op->site_index[0] < op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::B, op->site_index,
                                        -op->q_label, op->factor),
                                    true, -1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::B, op->site_index.flip(),
                                        op->q_label, op->factor),
                                    false, -1))
                       : nullptr;
        case OpNames::Q:
            return (mask & (1 << Q)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::Q, op->site_index.flip(),
                                 -op->q_label, op->factor),
                             true, 1)
                       : nullptr;
        default:
            return nullptr;
        }
    }
};

// Symmetry rules for simplifying quantum chemistry MPO (spin-adapted)
template <typename S> struct RuleQC<S, typename S::is_su2_t> : Rule<S> {
    uint8_t mask;
    const static uint8_t D = 0U, R = 1U, A = 2U, P = 3U, B = 4U, Q = 5U;
    RuleQC(bool d = true, bool r = true, bool a = true, bool p = true,
           bool b = true, bool q = true)
        : mask((d << D) | (r << R) | (a << A) | (p << P) | (b << B) |
               (q << Q)) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        switch (op->name) {
        case OpNames::D:
            return (mask & (1 << D)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::C, op->site_index,
                                               -op->q_label, op->factor),
                                           true, 1)
                                     : nullptr;
        case OpNames::RD:
            return (mask & (1 << R)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::R, op->site_index,
                                               -op->q_label, op->factor),
                                           true, -1)
                                     : nullptr;
        // Aij[S] = mul('Ci', 'Cj', S)
        case OpNames::A:
            return (mask & (1 << A)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::A, op->site_index.flip_spatial(),
                                 op->q_label, op->factor),
                             false, op->site_index.s() ? -1 : 1)
                       : nullptr;
        // ADij[S] = mul('Dj', 'Di', S)
        case OpNames::AD:
            return (mask & (1 << A))
                       ? (op->site_index[0] <= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A, op->site_index,
                                        -op->q_label, op->factor),
                                    true, op->site_index.s() ? 1 : -1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A,
                                        op->site_index.flip_spatial(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::P:
            return (mask & (1 << P)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::P, op->site_index.flip_spatial(),
                                 op->q_label, op->factor),
                             false, op->site_index.s() ? -1 : 1)
                       : nullptr;
        case OpNames::PD:
            return (mask & (1 << P))
                       ? (op->site_index[0] <= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P, op->site_index,
                                        -op->q_label, op->factor),
                                    true, op->site_index.s() ? 1 : -1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P,
                                        op->site_index.flip_spatial(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        // Bij[S] = mul('Ci', 'Dj', S)
        case OpNames::B:
            return (mask & (1 << B)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::B, op->site_index.flip_spatial(),
                                 -op->q_label, op->factor),
                             true, op->site_index.s() ? -1 : 1)
                       : nullptr;
        // BDij[S] = mul('Di', 'Cj', S)
        case OpNames::BD:
            return ((mask & (1 << B)) &&
                    (op->site_index[0] != op->site_index[1]))
                       ? (op->site_index[0] < op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::B, op->site_index,
                                        -op->q_label, op->factor),
                                    true, 1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::B,
                                        op->site_index.flip_spatial(),
                                        op->q_label, op->factor),
                                    false, op->site_index.s() ? -1 : 1))
                       : nullptr;
        case OpNames::Q:
            return (mask & (1 << Q)) && op->site_index[0] > op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::Q, op->site_index.flip_spatial(),
                                 -op->q_label, op->factor),
                             true, op->site_index.s() ? -1 : 1)
                       : nullptr;
        default:
            return nullptr;
        }
    }
};

// For anti-Hermitian Hamiltonian with only one-body terms
template <typename S> struct AntiHermitianRuleQC : Rule<S> {
    shared_ptr<Rule<S>> prim_rule;
    AntiHermitianRuleQC(const shared_ptr<Rule<S>> &rule) : prim_rule(rule) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        shared_ptr<OpElementRef<S>> r = prim_rule->operator()(op);
        return op->name == OpNames::RD
                   ? make_shared<OpElementRef<S>>(r->op, r->trans, -r->factor)
                   : r;
    }
};

} // namespace block2
