
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
#include <memory>

using namespace std;

namespace block2 {

// Rule for MPO simplification
template <typename S> struct Rule {
    Rule() {}
    virtual ~Rule() = default;
    virtual shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const {
        return nullptr;
    }
};

// Remove rules involving transposed operator from a rule
// The original rule is not changed
template <typename S> struct NoTransposeRule : Rule<S> {
    shared_ptr<Rule<S>> prim_rule;
    NoTransposeRule(const shared_ptr<Rule<S>> &rule) : prim_rule(rule) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        shared_ptr<OpElementRef<S>> r = prim_rule->operator()(op);
        return r == nullptr || r->trans ? nullptr : r;
    }
};

} // namespace block2
