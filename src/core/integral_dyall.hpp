
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

#include "matrix.hpp"
#include "integral.hpp"

using namespace std;

namespace block2 {

struct DyallFCIDUMP : FCIDUMP {
    shared_ptr<vector<double>> vdata_fock, vdata_heff;
    vector<TInt> fock, heff;
    shared_ptr<FCIDUMP> fcidump;
    uint16_t n_inactive, n_virtual, n_active;
    bool fock_uhf = false;
    double const_e_dyall;
    DyallFCIDUMP(const shared_ptr<FCIDUMP> &fcidump, uint16_t n_inactive,
                 uint16_t n_virtual)
        : fcidump(fcidump), n_inactive(n_inactive), n_virtual(n_virtual),
          n_active(fcidump->n_sites() - n_inactive - n_virtual) {
        params = fcidump->params;
        data = fcidump->data;
    }
    virtual ~DyallFCIDUMP() = default;
    void initialize_su2(const double *f, size_t lf) {
        initialize_fock_su2(f, lf);
        initialize_heff();
        initialize_const();
    }
    void initialize_sz(const double *fa, size_t lfa, const double *fb,
                       size_t lfb) {
        initialize_fock_sz(fa, lfa, fb, lfb);
        initialize_heff();
        initialize_const();
    }
    void initialize_from_1pdm_su2(MatrixRef pdm1) {
        initialize_fock_su2(pdm1);
        initialize_heff();
        initialize_const();
    }
    void initialize_from_1pdm_sz(MatrixRef pdm1) {
        initialize_fock_sz(pdm1);
        initialize_heff();
        initialize_const();
    }
    void read(const string &filename) override {
        shared_ptr<FCIDUMP> fd = make_shared<FCIDUMP>();
        fd->read(filename);
        vdata_fock = fd->vdata;
        fock = fd->ts;
        fd = nullptr;
        initialize_heff();
        initialize_const();
    }
    void initialize_fock_su2(MatrixRef pdm1) {
        fock.push_back(TInt(fcidump->n_sites(), true));
        assert(pdm1.size() == fock[0].size());
        vdata_fock = make_shared<vector<double>>(fock[0].size());
        fock[0].data = vdata_fock->data();
        for (uint16_t p = 0; p < fcidump->n_sites(); p++)
            for (uint16_t q = 0; q < fcidump->n_sites(); q++) {
                double v = fcidump->t(p, q);
                for (uint16_t r = 0; r < fcidump->n_sites(); r++)
                    for (uint16_t s = 0; s < fcidump->n_sites(); s++)
                        v += pdm1(r, s) * (fcidump->v(p, q, r, s) -
                                           0.5 * fcidump->v(p, r, s, q));
                fock[0](p, q) = v;
            }
        fock_uhf = false;
    }
    void initialize_fock_sz(MatrixRef pdm1) {
        fock.push_back(TInt(fcidump->n_sites(), true));
        fock.push_back(fock[0]);
        assert(pdm1.size() == fock[0].size() * 4);
        vdata_fock =
            make_shared<vector<double>>(fock[0].size() + fock[1].size());
        fock[0].data = vdata_fock->data();
        fock[1].data = vdata_fock->data() + fock[0].size();
        for (uint8_t x = 0; x < 2; x++)
            for (uint16_t p = 0; p < fcidump->n_sites(); p++)
                for (uint16_t q = 0; q < fcidump->n_sites(); q++) {
                    double v = 0.5 * fcidump->t(x, p, q);
                    for (uint8_t y = 0; y < 2; y++)
                        for (uint16_t r = 0; r < fcidump->n_sites(); r++)
                            for (uint16_t s = 0; s < fcidump->n_sites(); s++) {
                                v += pdm1(r * 2 + y, s * 2 + y) *
                                     fcidump->v(x, y, p, q, r, s);
                                if (x == y)
                                    v -= pdm1(r * 2 + y, s * 2 + y) *
                                         fcidump->v(x, y, p, r, s, q);
                            }
                    fock[x](p, q) = v;
                }
        fock_uhf = true;
    }
    void initialize_fock_su2(const double *f, size_t lf) {
        fock.push_back(TInt(fcidump->n_sites(), false));
        if (lf != fock[0].size())
            fock[0].general = true;
        assert(lf == fock[0].size());
        vdata_fock = make_shared<vector<double>>(fock[0].size());
        fock[0].data = vdata_fock->data();
        memcpy(fock[0].data, f, sizeof(double) * lf);
        fock_uhf = false;
    }
    void initialize_fock_sz(const double *fa, size_t lfa, const double *fb,
                            size_t lfb) {
        fock.push_back(TInt(fcidump->n_sites(), false));
        if (lfa != fock[0].size())
            fock[0].general = true;
        fock.push_back(fock[0]);
        assert(lfa == fock[0].size() && lfb == fock[1].size());
        vdata_fock =
            make_shared<vector<double>>(fock[0].size() + fock[1].size());
        fock[0].data = vdata_fock->data();
        fock[1].data = vdata_fock->data() + lfa;
        memcpy(fock[0].data, fa, sizeof(double) * lfa);
        memcpy(fock[1].data, fb, sizeof(double) * lfb);
        fock_uhf = true;
    }
    void initialize_heff() {
        if (!fcidump->uhf) {
            heff.push_back(TInt(n_active, true));
            vdata_heff = make_shared<vector<double>>(heff[0].size());
            heff[0].data = vdata_heff->data();
            for (uint16_t a = 0; a < n_active; a++)
                for (uint16_t b = 0; b < n_active; b++) {
                    double v = fcidump->t(a + n_inactive, b + n_inactive);
                    for (uint16_t i = 0; i < n_inactive; i++)
                        v += 2 * fcidump->v(a + n_inactive, b + n_inactive, i,
                                            i) -
                             fcidump->v(a + n_inactive, i, i, b + n_inactive);
                    heff[0](a, b) = v;
                }
        } else {
            heff.push_back(TInt(n_active, true));
            heff.push_back(TInt(n_active, true));
            vdata_heff =
                make_shared<vector<double>>(heff[0].size() + heff[1].size());
            heff[0].data = vdata_heff->data();
            heff[1].data = vdata_heff->data() + heff[0].size();
            for (uint8_t s = 0; s < 2; s++)
                for (uint16_t a = 0; a < n_active; a++)
                    for (uint16_t b = 0; b < n_active; b++) {
                        double v =
                            0.5 * fcidump->t(s, a + n_inactive, b + n_inactive);
                        for (uint8_t si = 0; si < 2; si++)
                            for (uint16_t i = 0; i < n_inactive; i++) {
                                v += 0.5 * fcidump->v(s, si, a + n_inactive,
                                                      b + n_inactive, i, i);
                                if (si == s)
                                    v -= 0.5 * fcidump->v(s, si, a + n_inactive,
                                                          i, i, b + n_inactive);
                            }
                        heff[s](a, b) = v;
                    }
        }
        uhf = fcidump->uhf;
    }
    void initialize_const() {
        const_e_dyall = 0;
        if (!fock_uhf) {
            for (uint16_t i = 0; i < n_inactive; i++)
                const_e_dyall -= 2 * fock[0](i, i);
        } else {
            for (uint8_t s = 0; s < 2; s++)
                for (uint16_t i = 0; i < n_inactive; i++)
                    const_e_dyall -= fock[s](i, i);
        }
        if (!fcidump->uhf) {
            for (uint16_t i = 0; i < n_inactive; i++) {
                const_e_dyall += 2 * fcidump->t(i, i);
                for (uint16_t j = 0; j < n_inactive; j++)
                    const_e_dyall +=
                        2 * fcidump->v(i, i, j, j) - fcidump->v(i, j, i, j);
            }
        } else {
            for (uint8_t s = 0; s < 2; s++)
                for (uint16_t i = 0; i < n_inactive; i++) {
                    const_e_dyall += fcidump->t(s, i, i);
                    for (uint8_t sj = 0; sj < 2; sj++)
                        for (uint16_t j = 0; j < n_inactive; j++) {
                            const_e_dyall +=
                                0.5 * fcidump->v(s, sj, i, i, j, j);
                            if (sj == s)
                                const_e_dyall -=
                                    0.5 * fcidump->v(s, sj, i, j, i, j);
                        }
                }
        }
    }
    shared_ptr<FCIDUMP> deep_copy() const override {
        shared_ptr<FCIDUMP> fd = fcidump->deep_copy();
        uint16_t n = n_sites();
        for (size_t s = 0; s < fd->ts.size(); s++) {
            fd->ts[s].clear();
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    fd->ts[s](i, j) = t(s, i, j);
        }
        auto is_ext = [this](uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
            return i < n_inactive || j < n_inactive || k < n_inactive ||
                   l < n_inactive || i >= n_inactive + n_active ||
                   j >= n_inactive + n_active || k >= n_inactive + n_active ||
                   l >= n_inactive + n_active;
        };
        for (size_t s = 0; s < vgs.size(); s++) {
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint16_t k = 0; k < n; k++)
                        for (uint16_t l = 0; l < n; l++)
                            if (is_ext(i, j, k, l))
                                fd->vgs[s](i, j, k, l) = 0;
        }
        for (size_t s = 0; s < vabs.size(); s++) {
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint16_t k = 0; k < n; k++)
                        for (uint16_t l = 0; l < n; l++)
                            if (is_ext(i, j, k, l))
                                fd->vabs[s](i, j, k, l) = 0;
        }
        for (size_t s = 0; s < vs.size(); s++) {
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint16_t k = 0; k < n; k++)
                        for (uint16_t l = 0; l < n; l++)
                            if (is_ext(i, j, k, l))
                                fd->vs[s](i, j, k, l) = 0;
        }
        fd->const_e = e();
        return fd;
    }
    // Remove integral elements that violate point group symmetry
    // orbsym: in XOR convention
    double symmetrize(const vector<uint8_t> &orbsym) override {
        uint16_t n = n_sites();
        assert((int)orbsym.size() == n);
        double error = 0.0;
        for (auto &x : fock)
            for (int i = 0; i < x.n; i++)
                for (int j = 0; j < (x.general ? x.n : i + 1); j++)
                    if (orbsym[i] ^ orbsym[j])
                        error += abs(x(i, j)), x(i, j) = 0;
        for (auto &x : heff)
            for (int i = 0; i < x.n; i++)
                for (int j = 0; j < (x.general ? x.n : i + 1); j++)
                    if (orbsym[i + n_inactive] ^ orbsym[j + n_inactive])
                        error += abs(x(i, j)), x(i, j) = 0;
        error += fcidump->symmetrize(orbsym);
        return error;
    }
    double t(uint16_t i, uint16_t j) const override {
        if ((i < n_inactive && j < n_inactive) ||
            (i >= n_inactive + n_active && j >= n_inactive + n_active))
            return fock[0](i, j);
        else if ((i >= n_inactive && i < n_inactive + n_active) &&
                 (j >= n_inactive && j < n_inactive + n_active))
            return heff[0](i - n_inactive, j - n_inactive);
        else
            return 0;
    }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        if ((i < n_inactive && j < n_inactive) ||
            (i >= n_inactive + n_active && j >= n_inactive + n_active))
            return fock_uhf ? fock[s](i, j) : fock[0](i, j);
        else if ((i >= n_inactive && i < n_inactive + n_active) &&
                 (j >= n_inactive && j < n_inactive + n_active))
            return uhf ? heff[s](i - n_inactive, j - n_inactive)
                       : heff[0](i - n_inactive, j - n_inactive);
        else
            return 0;
    }
    // Two-electron integral element (SU(2))
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        if (i < n_inactive || j < n_inactive || k < n_inactive ||
            l < n_inactive || i >= n_inactive + n_active ||
            j >= n_inactive + n_active || k >= n_inactive + n_active ||
            l >= n_inactive + n_active)
            return 0;
        else
            return fcidump->v(i, j, k, l);
    }
    // Two-electron integral element (SZ)
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        if (i < n_inactive || j < n_inactive || k < n_inactive ||
            l < n_inactive || i >= n_inactive + n_active ||
            j >= n_inactive + n_active || k >= n_inactive + n_active ||
            l >= n_inactive + n_active)
            return 0;
        else
            return fcidump->v(sl, sr, i, j, k, l);
    }
    double e() const override { return const_e_dyall + fcidump->const_e; }
    void deallocate() override {
        vdata_fock = nullptr;
        vdata_heff = nullptr;
        data = nullptr;
        fcidump->deallocate();
    }
};

} // namespace block2
