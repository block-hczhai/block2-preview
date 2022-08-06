
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

struct HubbardFCIDUMP : FCIDUMP<double> {
    double const_u, const_t;
    bool periodic;
    HubbardFCIDUMP(uint16_t n_sites, double t = 1, double u = 2,
                   bool periodic = false)
        : FCIDUMP<double>(), const_u(u), const_t(t), periodic(periodic) {
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
    }
    double t(uint16_t i, uint16_t j) const override {
        const uint16_t n = n_sites();
        if (periodic)
            return ((i + n - j) % n == 1 ? const_t : 0) +
                   ((j + n - i) % n == 1 ? const_t : 0);
        else
            return (i > j && i - j == 1) || (i < j && j - i == 1) ? const_t : 0;
    }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        const int n = n_sites();
        if (periodic)
            return ((i + n - j) % n == 1 ? const_t : 0) +
                   ((j + n - i) % n == 1 ? const_t : 0);
        else
            return (i > j && i - j == 1) || (i < j && j - i == 1) ? const_t : 0;
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
    long double e() const override { return (long double)0.0; }
    void deallocate() override {}
};

struct HubbardKSpaceFCIDUMP : FCIDUMP<double> {
    double const_u, const_t;
    const double _pi = acos(-1);
    HubbardKSpaceFCIDUMP(uint16_t n_sites, double t = 1, double u = 2)
        : FCIDUMP<double>(), const_u(u), const_t(t) {
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
        ss = stringstream();
        for (uint16_t i = 0; i < n_sites; i++) {
            ss << i;
            if (i != n_sites - 1)
                ss << ",";
        }
        params["ksym"] = ss.str();
        params["kmod"] = Parsing::to_string(n_sites);
        params["kisym"] = Parsing::to_string(n_sites / 2);
    }
    double t(uint16_t i, uint16_t j) const override {
        return i == j ? -2 * const_t * cos(2 * _pi * i / n_sites() + _pi) : 0;
    }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        return i == j ? -2 * const_t * cos(2 * _pi * i / n_sites() + _pi) : 0;
    }
    // Two-electron integral element (SU(2))
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        const int n = n_sites();
        return (i + n - j + k + n - l) % n == 0 ? const_u / n : 0;
    }
    // Two-electron integral element (SZ)
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        const int n = n_sites();
        return (i + n - j + k + n - l) % n == 0 ? const_u / n : 0;
    }
    long double e() const override { return (long double)0.0; }
    void deallocate() override {}
};

} // namespace block2
