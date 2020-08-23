
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

#include "parallel_rule.hpp"
#include <memory>

using namespace std;

namespace block2 {

// Rule for parallel dispatcher for quantum chemisty MPO
template <typename S> struct ParallelRuleQC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    ParallelRuleQC(const shared_ptr<ParallelCommunicator<S>> &comm)
        : ParallelRule<S>(comm) {}
    static int find_index(uint16_t i, uint16_t j) {
        return i < j ? ((int)j * (j + 1) >> 1) + i
                     : ((int)i * (i + 1) >> 1) + j;
    }
    ParallelProperty
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        SiteIndex si = op->site_index;
        switch (op->name) {
        case OpNames::I:
            return ParallelProperty(0, ParallelOpTypes::Repeated);
        case OpNames::C:
        case OpNames::D:
        case OpNames::N:
        case OpNames::NN:
            return ParallelProperty(si[0] % comm->size,
                                    ParallelOpTypes::Repeated);
        case OpNames::H:
            return ParallelProperty(0, ParallelOpTypes::Partial);
        case OpNames::R:
        case OpNames::RD:
            return ParallelProperty(si[0] % comm->size,
                                    ParallelOpTypes::Partial);
        case OpNames::PDM1:
            return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                    ParallelOpTypes::Number);
        case OpNames::A:
        case OpNames::AD:
        case OpNames::P:
        case OpNames::PD:
        case OpNames::B:
        case OpNames::BD:
        case OpNames::Q:
            return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                    ParallelOpTypes::None);
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

} // namespace block2
