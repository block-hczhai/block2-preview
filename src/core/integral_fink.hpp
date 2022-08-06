
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
#include "matrix.hpp"

using namespace std;

namespace block2 {

struct FinkFCIDUMP : FCIDUMP<double> {
    using FCIDUMP<double>::params;
    using FCIDUMP<double>::data;
    using FCIDUMP<double>::uhf;
    using FCIDUMP<double>::n_sites;
    using FCIDUMP<double>::vs;
    using FCIDUMP<double>::vgs;
    using FCIDUMP<double>::vabs;
    shared_ptr<FCIDUMP<double>> fcidump;
    uint16_t n_inactive, n_virtual, n_active;
    FinkFCIDUMP(const shared_ptr<FCIDUMP<double>> &fcidump, uint16_t n_inactive,
                uint16_t n_virtual)
        : fcidump(fcidump), n_inactive(n_inactive), n_virtual(n_virtual),
          n_active(fcidump->n_sites() - n_inactive - n_virtual) {
        params = fcidump->params;
        data = fcidump->data;
        uhf = fcidump->uhf;
    }
    virtual ~FinkFCIDUMP() = default;
    uint16_t sub_space(uint16_t x) const {
        if (x < n_inactive)
            return 2;
        else if (x >= n_inactive && x < n_inactive + n_active)
            return 1;
        else
            return 0;
    };
    bool is_fink(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return (sub_space(i) == sub_space(j) && sub_space(k) == sub_space(l)) ||
               (sub_space(i) == sub_space(l) && sub_space(k) == sub_space(j));
    }
    shared_ptr<FCIDUMP<double>> deep_copy() const override {
        shared_ptr<FCIDUMP<double>> fd = fcidump->deep_copy();
        uint16_t n = n_sites();
        for (size_t s = 0; s < fd->ts.size(); s++) {
            fd->ts[s].clear();
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    fd->ts[s](i, j) = t(s, i, j);
        }
        for (size_t s = 0; s < vgs.size(); s++) {
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint16_t k = 0; k < n; k++)
                        for (uint16_t l = 0; l < n; l++)
                            if (!is_fink(i, j, k, l))
                                fd->vgs[s](i, j, k, l) = 0;
        }
        for (size_t s = 0; s < vabs.size(); s++) {
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint16_t k = 0; k < n; k++)
                        for (uint16_t l = 0; l < n; l++)
                            if (!is_fink(i, j, k, l))
                                fd->vabs[s](i, j, k, l) = 0;
        }
        for (size_t s = 0; s < vs.size(); s++) {
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint16_t k = 0; k < n; k++)
                        for (uint16_t l = 0; l < n; l++)
                            if (!is_fink(i, j, k, l))
                                fd->vs[s](i, j, k, l) = 0;
        }
        fd->const_e = e();
        return fd;
    }
    double t(uint16_t i, uint16_t j) const override {
        return sub_space(i) == sub_space(j) ? fcidump->t(i, j) : 0;
    }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        return sub_space(i) == sub_space(j) ? fcidump->t(s, i, j) : 0;
    }
    // Two-electron integral element (SU(2))
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        if (!is_fink(i, j, k, l))
            return 0;
        else
            return fcidump->v(i, j, k, l);
    }
    // Two-electron integral element (SZ)
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        if (!is_fink(i, j, k, l))
            return 0;
        else
            return fcidump->v(sl, sr, i, j, k, l);
    }
    long double e() const override { return fcidump->const_e; }
    void deallocate() override {
        data = nullptr;
        fcidump->deallocate();
    }
};

} // namespace block2
