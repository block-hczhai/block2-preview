
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

#include "integral.hpp"

using namespace std;

namespace block2 {

struct HubbardFCIDUMP : FCIDUMP {
    double const_u, const_t;
    HubbardFCIDUMP(uint16_t n_sites, double t = 1, double u = 2)
        : FCIDUMP(), const_u(u), const_t(t) {
        params.clear();
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_sites);
        params["ms2"] = Parsing::to_string(0);
        params["isym"] = Parsing::to_string(1);
        params["iuhf"] = "0";
        stringstream ss;
        for (uint16_t i = 0; i < n_sites; i++) {
            ss << "1";
            if (i != n_sites - 1)
                ss << ",";
        }
        params["orbsym"] = ss.str();
    }
    double t(uint16_t i, uint16_t j) const override {
        return abs(i - j) == 1 ? const_t : 0;
    }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        return abs(i - j) == 1 ? const_t : 0;
    }
    // Two-electron integral element (SU(2))
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        return i == j && j == k && k == l ? const_u : 0;
    }
    // Two-electron integral element (SZ)
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        return i == j && j == k && k == l ? const_u : 0;
    }
    double e() const override { return 0.0; }
    void deallocate() override {}
};

} // namespace block2
