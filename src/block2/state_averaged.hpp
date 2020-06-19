
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

#include "mps.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace std;

namespace block2 {

// MPSInfo for multiple targets
template <typename S> struct MultiMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::left_dims_fci;
    using MPSInfo<S>::right_dims_fci;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::get_basis;
    vector<S> targets;
    MultiMPSInfo(int n_sites, S vacuum, const vector<S> &targets,
                 StateInfo<S> *basis, const vector<uint8_t> orbsym,
                 bool init_fci = true)
        : targets(targets), MPSInfo<S>(n_sites, vacuum, targets[0], basis,
                                       orbsym, false) {
        set_bond_dimension_fci();
    }
    vector<S> get_complementary(S q) const override {
        vector<S> r;
        for (auto target : targets) {
            S qs = target - q;
            for (int i = 0; i < qs.count(); i++)
                r.push_back(qs[i]);
        }
        sort(r.begin(), r.end());
        r.resize(distance(r.begin(), unique(r.begin(), r.end())));
        return r;
    }
    void set_bond_dimension_fci() override {
        S max_target = *max_element(targets.begin(), targets.end());
        left_dims_fci[0] = StateInfo<S>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] = StateInfo<S>::tensor_product(
                left_dims_fci[i], get_basis(i), max_target);
        right_dims_fci[n_sites] = StateInfo<S>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] = StateInfo<S>::tensor_product(
                get_basis(i), right_dims_fci[i + 1], max_target);
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<S>::multi_target_filter(left_dims_fci[i],
                                              right_dims_fci[i], targets);
            StateInfo<S>::multi_target_filter(right_dims_fci[i],
                                              left_dims_fci[i], targets);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i].collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i].collect();
    }
};

} // namespace block2