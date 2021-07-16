
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
#include <memory>

using namespace std;

namespace block2 {

// Rule for parallel dispatcher for quantum chemistry MPO
template <typename S> struct ParallelRuleQC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    ParallelRuleQC(const shared_ptr<ParallelCommunicator<S>> &comm,
                   ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRuleQC<S>>(r->comm, r->comm_type);
    }
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
        case OpNames::A:
        case OpNames::AD:
        case OpNames::P:
        case OpNames::PD:
        case OpNames::B:
        case OpNames::BD:
        case OpNames::Q:
        case OpNames::TEMP:
            return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                    ParallelOpTypes::None);
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

// Rule for parallel dispatcher for quantum chemistry MPO with only one-body
// term
template <typename S> struct ParallelRuleOneBodyQC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    ParallelRuleOneBodyQC(const shared_ptr<ParallelCommunicator<S>> &comm,
                          ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRuleOneBodyQC<S>>(r->comm, r->comm_type);
    }
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
        case OpNames::R:
        case OpNames::RD:
            return ParallelProperty(si[0] % comm->size, ParallelOpTypes::None);
        case OpNames::H:
            return ParallelProperty(0, ParallelOpTypes::Partial);
        case OpNames::TEMP:
            return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                    ParallelOpTypes::None);
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

// Rule for parallel dispatcher for quantum chemistry 1PDM
// this one should provide better scalability than ParallelRuleNPDMQC
template <typename S> struct ParallelRulePDM1QC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    mutable ParallelRulePartitionTypes partition;
    ParallelRulePDM1QC(const shared_ptr<ParallelCommunicator<S>> &comm,
                       ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRulePDM1QC<S>>(r->comm, r->comm_type);
    }
    void set_partition(ParallelRulePartitionTypes partition) const override {
        this->partition = partition;
    }
    static uint64_t find_index(uint32_t i, uint32_t j) { return i < j ? j : i; }
    ParallelProperty
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        SiteIndex si = op->site_index;
        switch (partition) {
        case ParallelRulePartitionTypes::Left:
            return ParallelProperty(0, ParallelOpTypes::Repeated);
        case ParallelRulePartitionTypes::Right:
            switch (op->name) {
            case OpNames::I:
                return ParallelProperty(0, ParallelOpTypes::Repeated);
            case OpNames::C:
            case OpNames::D:
            case OpNames::N:
            case OpNames::NN:
                return ParallelProperty(si[0] % comm->size,
                                        ParallelOpTypes::None);
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
            case OpNames::BD:
                return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                        ParallelOpTypes::None);
            default:
                assert(false);
            }
        case ParallelRulePartitionTypes::Middle:
            switch (op->name) {
            case OpNames::PDM1:
                return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                        ParallelOpTypes::Number);
            default:
                return ParallelProperty(0, ParallelOpTypes::Repeated);
            }
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

// Rule for parallel dispatcher for quantum chemistry 2PDM
// this one should provide better scalability than ParallelRuleNPDMQC
template <typename S> struct ParallelRulePDM2QC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    mutable ParallelRulePartitionTypes partition;
    ParallelRulePDM2QC(const shared_ptr<ParallelCommunicator<S>> &comm,
                       ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRulePDM2QC<S>>(r->comm, r->comm_type);
    }
    void set_partition(ParallelRulePartitionTypes partition) const override {
        this->partition = partition;
    }
    static uint64_t find_index(uint32_t i, uint32_t j) {
        return i < j ? ((int)j * (j + 1) >> 1) + i
                     : ((int)i * (i + 1) >> 1) + j;
    }
    static uint64_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        array<uint16_t, 4> arr = {i, j, k, l};
        sort(arr.begin(), arr.end());
        if (arr[1] == arr[2])
            return arr[3];
        else
            return find_index(arr[2], arr[3]);
    }
    ParallelProperty
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        SiteIndex si = op->site_index;
        switch (partition) {
        case ParallelRulePartitionTypes::Left:
            return ParallelProperty(0, ParallelOpTypes::Repeated);
        case ParallelRulePartitionTypes::Right:
            switch (op->name) {
            case OpNames::I:
                return ParallelProperty(0, ParallelOpTypes::Repeated);
            case OpNames::C:
            case OpNames::D:
            case OpNames::N:
            case OpNames::NN:
                return ParallelProperty(si[0] % comm->size,
                                        ParallelOpTypes::Repeated);
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
            case OpNames::BD:
                return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                        ParallelOpTypes::None);
            case OpNames::CCD:
            case OpNames::CDC:
            case OpNames::CDD:
            case OpNames::DCC:
            case OpNames::DCD:
            case OpNames::DDC:
            case OpNames::CCDD:
                return ParallelProperty(si[0] % comm->size,
                                        ParallelOpTypes::None);
            default:
                assert(false);
            }
        case ParallelRulePartitionTypes::Middle:
            switch (op->name) {
            case OpNames::PDM2:
                return ParallelProperty(find_index(si[0], si[1], si[2], si[3]) %
                                            comm->size,
                                        ParallelOpTypes::Number);
            default:
                return ParallelProperty(0, ParallelOpTypes::Repeated);
            }
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

// Rule for parallel dispatcher for quantum chemistry NPDM
template <typename S> struct ParallelRuleNPDMQC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    ParallelRuleNPDMQC(const shared_ptr<ParallelCommunicator<S>> &comm,
                       ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRuleNPDMQC<S>>(r->comm, r->comm_type);
    }
    static uint64_t find_index(uint32_t i, uint32_t j) {
        return i < j ? ((int)j * (j + 1) >> 1) + i
                     : ((int)i * (i + 1) >> 1) + j;
    }
    static uint64_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        uint32_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return find_index(p, q);
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
        case OpNames::A:
        case OpNames::AD:
        case OpNames::B:
        case OpNames::BD:
            return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                    ParallelOpTypes::Repeated);
        case OpNames::CCD:
        case OpNames::CDC:
        case OpNames::CDD:
        case OpNames::DCC:
        case OpNames::DCD:
        case OpNames::DDC:
            return ParallelProperty(find_index(si[0], si[1], si[2], 0) %
                                        comm->size,
                                    ParallelOpTypes::Repeated);
        case OpNames::CCDD:
            return ParallelProperty(find_index(si[0], si[1], si[2], si[3]) %
                                        comm->size,
                                    ParallelOpTypes::Repeated);
        case OpNames::PDM1:
            return ParallelProperty(find_index(si[0], si[1]) % comm->size,
                                    ParallelOpTypes::Number);
        case OpNames::PDM2:
            return ParallelProperty(find_index(si[0], si[1], si[2], si[3]) %
                                        comm->size,
                                    ParallelOpTypes::Number);
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

// Rule for parallel dispatcher for SiteMPO/LocalMPO
template <typename S> struct ParallelRuleSiteQC : ParallelRule<S> {
    using ParallelRule<S>::comm;
    ParallelRuleSiteQC(const shared_ptr<ParallelCommunicator<S>> &comm,
                       ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRuleSiteQC<S>>(r->comm, r->comm_type);
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
            return ParallelProperty(0, ParallelOpTypes::None);
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

// Rule for parallel dispatcher for IdentityMPO
template <typename S> struct ParallelRuleIdentity : ParallelRule<S> {
    using ParallelRule<S>::comm;
    ParallelRuleIdentity(const shared_ptr<ParallelCommunicator<S>> &comm,
                         ParallelCommTypes comm_type = ParallelCommTypes::None)
        : ParallelRule<S>(comm, comm_type) {}
    shared_ptr<ParallelRule<S>> split(int gsize) const override {
        shared_ptr<ParallelRule<S>> r = ParallelRule<S>::split(gsize);
        return make_shared<ParallelRuleIdentity<S>>(r->comm, r->comm_type);
    }
    ParallelProperty
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        SiteIndex si = op->site_index;
        switch (op->name) {
        case OpNames::I:
            return ParallelProperty(0, ParallelOpTypes::None);
        default:
            assert(false);
        }
        return ParallelRule<S>::operator()(op);
    }
};

} // namespace block2
