
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
#include "utils.hpp"
#include <cmath>

using namespace std;

namespace block2 {

struct HeisenbergFCIDUMP : FCIDUMP<double> {
    vector<double> couplings;
    HeisenbergFCIDUMP(const shared_ptr<FCIDUMP<double>> &fd)
        : FCIDUMP<double>() {
        uint16_t n_sites = fd->n_sites();
        params.clear();
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_sites);
        params["ms2"] = Parsing::to_string(n_sites & 1);
        params["isym"] = Parsing::to_string(1);
        params["iuhf"] = "0";
        stringstream ss;
        for (uint16_t i = 0; i < n_sites; i++) {
            ss << "1";
            if (i != n_sites - 1)
                ss << ",";
        }
        params["orbsym"] = ss.str();
        couplings.resize((size_t)n_sites * n_sites, 0);
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++)
                couplings[i * (int)n_sites + j] = fd->t(i, j);
    }
    double t(uint16_t i, uint16_t j) const override { return 0; }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override { return 0; }
    // Two-electron integral element (SU(2))
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        const int n = n_sites();
        if (i == j && k == l)
            return -0.25 * couplings[i * n * k];
        else if (i == l && j == k)
            return -0.5 * couplings[i * n * j];
        else
            return 0;
    }
    // Two-electron integral element (SZ)
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        const int n = n_sites();
        if (i == j && k == l)
            return -0.25 * couplings[i * n * k];
        else if (i == l && j == k)
            return -0.5 * couplings[i * n * j];
        else
            return 0;
    }
    long double e() const override { return (long double)0.0; }
    void deallocate() override {}
};

} // namespace block2
