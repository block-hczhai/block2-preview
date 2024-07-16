
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2023 Huanchen Zhai <hczhai@caltech.edu>
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

#include "drt_big_site.hpp"
#include <algorithm>

using namespace std;

namespace block2 {

template <typename S, typename FL, ElemOpTypes T = ElemT<S>::value>
struct DRTMPS {
    typedef long long LL;
    shared_ptr<DRT<S, T>> drt;
    vector<LL> shapes;
    vector<vector<LL>> offsets;
    vector<vector<FL>> data;
    static int get_k(const shared_ptr<DRT<S, T>> &drt, int j) {
        return j < drt->n_rows() && j >= 0
                   ? drt->abc[j][0] + abs(drt->abc[j][1]) + drt->abc[j][2]
                   : -1;
    }
    static vector<vector<LL>> get_offsets(const shared_ptr<DRT<S, T>> &drt,
                                          const vector<LL> &shapes) {
        vector<vector<LL>> r(drt->n_sites);
        for (int k = drt->n_sites - 1, j = 0; k >= 0; k--) {
            r[k].push_back(0);
            for (; get_k(drt, j) == k + 1; j++)
                for (int16_t d = 0; d < 4; d++) {
                    LL x = shapes[j] * shapes[drt->jds[j][d]] *
                           (drt->jds[j][d] != 0);
                    r[k].push_back(r[k].back() + x);
                }
        }
        return r;
    }
    static vector<array<int, 4>> get_inv_jds(const shared_ptr<DRT<S, T>> &drt) {
        vector<array<int, 4>> inv_jds(drt->jds.size(),
                                      array<int, 4>{-1, -1, -1, -1});
        for (size_t j = 0; j < drt->jds.size(); j++)
            for (int16_t d = 0; d < 4; d++)
                if (drt->jds[j][d] != 0)
                    inv_jds[drt->jds[j][d]][d] = (int)j;
        return inv_jds;
    }
    DRTMPS(const shared_ptr<DRT<S, T>> &drt, const vector<LL> &shapes,
           const vector<vector<FL>> &data)
        : drt(drt), shapes(shapes), data(data) {
        if (this->shapes.size() == 0) {
            this->shapes.resize(drt->xs.size());
            for (size_t j = 0; j < drt->xs.size(); j++)
                this->shapes[j] = drt->xs[j][4];
        }
        offsets = get_offsets(drt, this->shapes);
        if (this->data.size() == 0) {
            this->data.resize(drt->n_sites);
            for (int k = 0; k < drt->n_sites; k++)
                this->data[k] = vector<FL>(offsets[k].back());
        }
    }
    virtual ~DRTMPS() = default;
    static shared_ptr<DRTMPS> from_ci_vector(const shared_ptr<DRT<S, T>> &drt,
                                             const FL *ci) {
        shared_ptr<DRTMPS<S, FL>> r =
            make_shared<DRTMPS<S, FL>>(drt, vector<LL>(), vector<vector<FL>>());
        vector<LL> ij(drt->n_init_qs + 1, 0);
        for (int i = 0; i < drt->n_init_qs; i++)
            ij[i + 1] = ij[i] + drt->xs[i].back();
        for (int k = drt->n_sites - 1, j = 0, jz; k >= 0; k--)
            for (jz = j; get_k(drt, j) == k + 1; j++)
                for (int16_t d = 0; d < 4; d++)
                    if (drt->jds[j][d] != 0) {
                        GMatrix<FL> pd(r->data[k].data() +
                                           r->offsets[k][(j - jz) * 4 + d],
                                       (MKL_INT)r->shapes[j],
                                       (MKL_INT)r->shapes[drt->jds[j][d]]);
                        for (LL x = drt->xs[j][d]; x < drt->xs[j][d + 1]; x++)
                            pd((MKL_INT)x, (MKL_INT)(x - drt->xs[j][d])) =
                                j >= drt->n_init_qs ? (FL)1.0 : ci[x + ij[j]];
                    }
        return r;
    }
    vector<FL> to_ci_vector() const {
        vector<vector<int>> pkr(2, vector<int>());
        vector<vector<LL>> pk(2, vector<LL>());
        vector<vector<FL>> mats(2, vector<FL>());
        vector<vector<LL>> idxs(2, vector<LL>());
        vector<LL> bdims = get_bond_dimensions();
        LL max_bdim = *max_element(bdims.begin(), bdims.end());
        pkr[0].reserve(drt->size()), pk[0].reserve(drt->size());
        pkr[1].reserve(drt->size()), pk[1].reserve(drt->size());
        mats[0].reserve(max_bdim), mats[1].reserve(max_bdim);
        idxs[0].reserve(drt->size()), idxs[1].reserve(drt->size());
        LL x = 0, sx = 0;
        for (int i = 0; i < drt->n_init_qs; i++) {
            pkr[0].push_back(i);
            pk[0].push_back(x);
            idxs[0].push_back(sx);
            x += drt->xs[i].back(), sx += shapes[i];
        }
        vector<FL> ci(x);
        idxs[0].push_back(sx);
        for (LL p = 0; p < bdims.back(); p++)
            mats[0].push_back((FL)1.0);
        int pi = 0, pj = pi ^ 1;
        for (int k = drt->n_sites - 1, jz = 0; k >= 0; k--, pi ^= 1, pj ^= 1) {
            pkr[pj].clear(), pk[pj].clear();
            LL mat_sz = 0, idx_sz = 0;
            for (int j = 0; j < pkr[pi].size(); j++)
                for (int16_t d = 0; d < 4; d++)
                    if (drt->jds[pkr[pi][j]][d] != 0 &&
                        shapes[pkr[pi][j]] != 0 &&
                        shapes[drt->jds[pkr[pi][j]][d]] != 0)
                        mat_sz += shapes[drt->jds[pkr[pi][j]][d]], idx_sz++;
            mats[pj].resize(mat_sz);
            idxs[pj].resize(idx_sz + 1);
            idxs[pj][idx_sz] = mat_sz;
            mat_sz = 0, idx_sz = 0;
            for (int j = 0; j < pkr[pi].size(); j++)
                for (int16_t d = 0; d < 4; d++)
                    if (drt->jds[pkr[pi][j]][d] != 0 &&
                        shapes[pkr[pi][j]] != 0 &&
                        shapes[drt->jds[pkr[pi][j]][d]] != 0) {
                        GMatrix<FL> ma(
                            mats[pi].data() + idxs[pi][j], 1,
                            (MKL_INT)(idxs[pi][j + 1] - idxs[pi][j]));
                        GMatrix<FL> mb(
                            (FL *)data[k].data() +
                                offsets[k][(pkr[pi][j] - jz) * 4 + d],
                            (MKL_INT)shapes[pkr[pi][j]],
                            (MKL_INT)shapes[drt->jds[pkr[pi][j]][d]]);
                        GMatrix<FL> mc(
                            mats[pj].data() + mat_sz, 1,
                            (MKL_INT)shapes[drt->jds[pkr[pi][j]][d]]);
                        GMatrixFunctions<FL>::multiply(ma, 0, mb, 0, mc,
                                                       (FL)1.0, (FL)0.0);
                        idxs[pj][idx_sz++] = mat_sz;
                        mat_sz += shapes[drt->jds[pkr[pi][j]][d]];
                        pkr[pj].push_back(drt->jds[pkr[pi][j]][d]);
                        pk[pj].push_back(pk[pi][j] + drt->xs[pkr[pi][j]][d]);
                    }
            for (; get_k(drt, jz) == k + 1; jz++)
                ;
        }
        for (int j = 0; j < pk[pi].size(); j++)
            ci[pk[pi][j]] += mats[pi][j];
        return ci;
    }
    vector<LL> get_bond_dimensions() const {
        vector<LL> r(drt->n_sites + 1, 0);
        for (int k = drt->n_sites, j = 0; k >= 0; k--)
            for (; get_k(drt, j) == k; j++)
                r[k] += shapes[j];
        return r;
    }
    shared_ptr<DRTMPS> qr() const {
        vector<array<int, 4>> inv_jds = get_inv_jds(drt);
        vector<LL> new_shapes(drt->xs.size(), 1);
        for (int j = drt->n_init_qs; j < (int)drt->xs.size(); j++) {
            LL x = 0;
            for (int16_t d = 0; d < 4; d++)
                if (inv_jds[j][d] != -1)
                    x += new_shapes[inv_jds[j][d]];
            new_shapes[j] = min(x, shapes[j]);
        }
        shared_ptr<DRTMPS> r =
            make_shared<DRTMPS<S, FL>>(drt, new_shapes, vector<vector<FL>>());
        vector<vector<FL>> gauges(drt->xs.size());
        vector<FL> matx, matq;
        for (int i = 0; i < drt->n_init_qs; i++)
            gauges[i] = vector<FL>(shapes[i], (FL)1.0);
        for (int k = drt->n_sites - 1, j = drt->n_init_qs, jz = 0, pz, l;
             k >= 0; k--) {
            for (pz = jz, jz = j; get_k(drt, j) == k; j++) {
                LL mz = 0;
                for (int16_t d = 0; d < 4; d++)
                    if (inv_jds[j][d] != -1)
                        mz += new_shapes[inv_jds[j][d]];
                matx.resize(mz * shapes[j]);
                mz = 0;
                for (int16_t d = 0; d < 4; d++)
                    if ((l = inv_jds[j][d]) != -1 && new_shapes[l] != 0) {
                        if (shapes[j] != 0)
                            GMatrixFunctions<FL>::multiply(
                                GMatrix<FL>(gauges[l].data(),
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)shapes[l]),
                                false,
                                GMatrix<FL>((FL *)data[k].data() +
                                                offsets[k][(l - pz) * 4 + d],
                                            (MKL_INT)shapes[l],
                                            (MKL_INT)shapes[j]),
                                false,
                                GMatrix<FL>(matx.data() + mz * shapes[j],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)shapes[j]),
                                (FL)1.0, (FL)0.0);
                        mz += new_shapes[l];
                    }
                matq.resize(mz * new_shapes[j]);
                gauges[j].resize(new_shapes[j] * shapes[j]);
                if (new_shapes[j] == 0 || shapes[j] == 0)
                    continue;
                if (k != 0) {
                    GMatrixFunctions<FL>::qr(
                        GMatrix<FL>(matx.data(), (MKL_INT)mz,
                                    (MKL_INT)shapes[j]),
                        GMatrix<FL>(matq.data(), (MKL_INT)mz,
                                    (MKL_INT)new_shapes[j]),
                        GMatrix<FL>(gauges[j].data(), (MKL_INT)new_shapes[j],
                                    (MKL_INT)shapes[j]));
                    mz = 0;
                    for (int16_t d = 0; d < 4; d++)
                        if ((l = inv_jds[j][d]) != -1) {
                            GMatrixFunctions<FL>::copy(
                                GMatrix<FL>(r->data[k].data() +
                                                r->offsets[k][(l - pz) * 4 + d],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)new_shapes[j]),
                                GMatrix<FL>(matq.data() + mz * new_shapes[j],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)new_shapes[j]));
                            mz += new_shapes[l];
                        }
                } else {
                    assert(shapes[j] == new_shapes[j]);
                    mz = 0;
                    for (int16_t d = 0; d < 4; d++)
                        if ((l = inv_jds[j][d]) != -1) {
                            GMatrixFunctions<FL>::copy(
                                GMatrix<FL>(r->data[k].data() +
                                                r->offsets[k][(l - pz) * 4 + d],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)new_shapes[j]),
                                GMatrix<FL>(matx.data() + mz * new_shapes[j],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)new_shapes[j]));
                            mz += new_shapes[l];
                        }
                }
            }
            for (l = pz; l < jz; l++)
                gauges[l] = vector<FL>();
        }
        return r;
    }
    shared_ptr<DRTMPS> svd(int max_bond_dim = -1, FL cutoff = (FL)0.0) const {
        vector<vector<FL>> gauges(drt->n_rows());
        vector<LL> new_shapes(drt->n_rows(), 1);
        vector<vector<int>> jx(drt->n_sites + 1);
        for (int k = drt->n_sites, j = 0; k >= 0; k--)
            for (; get_k(drt, j) == k; j++)
                jx[k].push_back(j);
        int mjx = 0;
        for (int k = 0; k < drt->n_sites; k++)
            mjx = max(mjx, (int)jx[k + 1].size());
        for (int i = 0; i < jx[0].size(); i++)
            gauges[jx[0][i]] = vector<FL>(shapes[jx[0][i]], (FL)1.0);
        int jz = jx[0][0];
        vector<vector<FL>> tmps(mjx), tmpl(mjx), tmpr(mjx);
        vector<FL> tmpx, s_vals;
        vector<vector<FL>> new_data(drt->n_sites);
        vector<vector<LL>> new_offsets(drt->n_sites);
        for (int k = 0, l; k < drt->n_sites; k++) {
            jz = jx[k + 1][0];
            LL total_s = 0;
            for (int j : jx[k + 1]) {
                LL nx = 0, mz = 0, mm;
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0)
                        nx += new_shapes[l];
                tmpx.reserve(nx * shapes[j]);
                mm = min(nx, shapes[j]);
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0 && new_shapes[l] != 0) {
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>((FL *)data[k].data() +
                                            offsets[k][(j - jz) * 4 + d],
                                        (MKL_INT)shapes[j], (MKL_INT)shapes[l]),
                            false,
                            GMatrix<FL>(gauges[l].data(), (MKL_INT)shapes[l],
                                        (MKL_INT)new_shapes[l]),
                            false,
                            GMatrix<FL>(tmpx.data() + mz, (MKL_INT)shapes[j],
                                        (MKL_INT)nx),
                            (FL)1.0, (FL)0.0);
                        mz += new_shapes[l];
                    }
                if (nx != 0) {
                    tmps[j - jz].reserve(mm);
                    tmpl[j - jz].reserve(shapes[j] * mm);
                    tmpr[j - jz].reserve(mm * nx);
                    if (mm != 0) {
                        GMatrixFunctions<FL>::svd(
                            GMatrix<FL>(tmpx.data(), (MKL_INT)shapes[j],
                                        (MKL_INT)nx),
                            GMatrix<FL>(tmpl[j - jz].data(), (MKL_INT)shapes[j],
                                        (MKL_INT)mm),
                            GMatrix<FL>(tmps[j - jz].data(), 1, (MKL_INT)mm),
                            GMatrix<FL>(tmpr[j - jz].data(), (MKL_INT)mm,
                                        (MKL_INT)nx));
                        if (k == drt->n_sites - 1) {
                            assert(mm == 1 && shapes[j] == 1);
                            for (LL i = 0; i < nx; i++)
                                tmpr[j - jz][i] *=
                                    tmps[j - jz][0] * tmpl[j - jz][0];
                        }
                    }
                }
                total_s += mm;
            }
            s_vals.reserve(total_s);
            total_s = 0;
            for (int j : jx[k + 1]) {
                LL nx = 0;
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0)
                        nx += new_shapes[l];
                LL mm = min(nx, shapes[j]);
                if (mm != 0)
                    GMatrixFunctions<FL>::copy(
                        GMatrix<FL>(s_vals.data() + total_s, 1, (MKL_INT)mm),
                        GMatrix<FL>(tmps[j - jz].data(), 1, (MKL_INT)mm));
                total_s += mm;
            }
            vector<LL> srt_idx(total_s);
            for (LL i = 0; i < total_s; i++)
                srt_idx[i] = i;
            sort(srt_idx.begin(), srt_idx.end(),
                 [&s_vals](LL a, LL b) { return s_vals[a] > s_vals[b]; });
            if (max_bond_dim != -1 && (int)srt_idx.size() > max_bond_dim)
                srt_idx.resize(max_bond_dim);
            LL jxx = 0;
            for (LL ix = 0; ix < (LL)srt_idx.size(); ix++) {
                srt_idx[jxx] = srt_idx[ix];
                jxx += (s_vals[srt_idx[ix]] >= cutoff);
            }
            srt_idx.resize(jxx);
            sort(srt_idx.begin(), srt_idx.end());
            new_offsets[k].push_back(0);
            LL isrt = 0, isrtz;
            total_s = 0;
            for (int j : jx[k + 1]) {
                LL nx = 0;
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0)
                        nx += new_shapes[l];
                LL mm = min(nx, shapes[j]);
                for (isrtz = isrt;
                     isrt < (LL)srt_idx.size() && srt_idx[isrt] - total_s < mm;)
                    isrt++;
                new_shapes[j] = isrt - isrtz;
                gauges[j].resize(shapes[j] * new_shapes[j]);
                for (LL g = 0; g < shapes[j]; g++)
                    for (LL jsrt = isrtz; jsrt < isrt; jsrt++)
                        gauges[j][g * new_shapes[j] + (jsrt - isrtz)] =
                            tmpl[j - jz][g * mm + (srt_idx[jsrt] - total_s)] *
                            s_vals[srt_idx[jsrt]];
                for (int16_t d = 0; d < 4; d++) {
                    LL x = new_shapes[j] * new_shapes[drt->jds[j][d]] *
                           (drt->jds[j][d] != 0);
                    new_offsets[k].push_back(new_offsets[k].back() + x);
                }
                total_s += mm;
            }
            new_data[k] = vector<FL>(new_offsets[k].back());
            total_s = 0, isrt = 0;
            for (int j : jx[k + 1]) {
                isrtz = isrt;
                isrt += new_shapes[j];
                LL mz = 0, nx = 0;
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0)
                        nx += new_shapes[l];
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0 && new_shapes[l] != 0) {
                        for (LL jsrt = isrtz; jsrt < isrt; jsrt++)
                            GMatrixFunctions<FL>::copy(
                                GMatrix<FL>(
                                    new_data[k].data() +
                                        new_offsets[k][(j - jz) * 4 + d] +
                                        (jsrt - isrtz) * new_shapes[l],
                                    1, (MKL_INT)new_shapes[l]),
                                GMatrix<FL>(tmpr[j - jz].data() +
                                                (srt_idx[jsrt] - total_s) * nx +
                                                mz,
                                            1, (MKL_INT)new_shapes[l]));
                        mz += new_shapes[l];
                    }
                total_s += min(nx, shapes[j]);
            }
        }
        return make_shared<DRTMPS<S, FL>>(drt, new_shapes, new_data);
    }
    FL dot(const shared_ptr<DRTMPS> &other) const {
        assert(drt->n_sites == other->drt->n_sites);
        vector<vector<FL>> mats(drt->xs.size());
        for (int i = 0; i < drt->n_init_qs; i++)
            mats[i] = vector<FL>(shapes[i] * other->shapes[i], (FL)1.0);
        vector<FL> tmp;
        for (int k = drt->n_sites - 1, j = 0, jz, l; k >= 0; k--) {
            for (jz = j; get_k(drt, j) == k + 1; j++)
                for (int16_t d = 0; d < 4; d++)
                    if ((l = drt->jds[j][d]) != 0 && shapes[l] != 0 &&
                        other->shapes[l] != 0) {
                        if (mats[l].size() == 0)
                            mats[l] = vector<FL>(shapes[l] * other->shapes[l],
                                                 (FL)0.0);
                        if (shapes[j] == 0 || other->shapes[j] == 0)
                            continue;
                        tmp.reserve(shapes[j] * other->shapes[l]);
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>(mats[j].data(), (MKL_INT)shapes[j],
                                        (MKL_INT)other->shapes[j]),
                            false,
                            GMatrix<FL>((FL *)other->data[k].data() +
                                            other->offsets[k][(j - jz) * 4 + d],
                                        (MKL_INT)other->shapes[j],
                                        (MKL_INT)other->shapes[l]),
                            false,
                            GMatrix<FL>(tmp.data(), (MKL_INT)shapes[j],
                                        (MKL_INT)other->shapes[l]),
                            (FL)1.0, (FL)0.0);
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>((FL *)data[k].data() +
                                            offsets[k][(j - jz) * 4 + d],
                                        (MKL_INT)shapes[j], (MKL_INT)shapes[l]),
                            3,
                            GMatrix<FL>(tmp.data(), (MKL_INT)shapes[j],
                                        (MKL_INT)other->shapes[l]),
                            false,
                            GMatrix<FL>(mats[l].data(), (MKL_INT)shapes[l],
                                        (MKL_INT)other->shapes[l]),
                            (FL)1.0, (FL)1.0);
                    }
            for (l = jz; l < j; l++)
                mats[l] = vector<FL>();
        }
        return accumulate(mats.back().begin(), mats.back().end(), (FL)0.0,
                          plus<FL>());
    }
    FL expect(const shared_ptr<DRTMPS> &other, const shared_ptr<HDRT<S>> &hdrt,
              const vector<vector<ElemMat<S, FL>>> &site_matrices,
              const shared_ptr<vector<FL>> &ints) const {
        assert(drt->n_sites == other->drt->n_sites);
        vector<vector<int>> jx(drt->n_sites + 1);
        for (int k = drt->n_sites, j = 0; k >= 0; k--)
            for (; get_k(drt, j) == k; j++)
                jx[k].push_back(j);
        FL r = 0;
        for (LL ihx = 0, ih; ihx < (LL)ints->size(); ihx++) {
            if (abs((*ints)[ihx]) < TINY)
                continue;
            int jh = 0;
            for (ih = ihx; ih >= hdrt->xs[jh * (hdrt->nd + 1) + hdrt->nd]; jh++)
                ih -= hdrt->xs[jh * (hdrt->nd + 1) + hdrt->nd];
            map<pair<int, int>, vector<FL>> mats;
            for (int i = 0; i < drt->n_init_qs; i++)
                mats[make_pair(i, i)] =
                    vector<FL>(shapes[i] * other->shapes[i], (FL)1.0);
            vector<FL> tmp;
            for (int k = drt->n_sites - 1, j = 0; k >= 0; k--) {
                int16_t dh =
                    (int16_t)(upper_bound(
                                  hdrt->xs.begin() + jh * (hdrt->nd + 1),
                                  hdrt->xs.begin() + (jh + 1) * (hdrt->nd + 1),
                                  ih) -
                              1 - (hdrt->xs.begin() + jh * (hdrt->nd + 1)));
                const int jhv = hdrt->jds[jh * hdrt->nd + dh];
                const ElemMat<S, FL> &smat = site_matrices[k][dh];
                for (size_t md = 0; md < smat.data.size(); md++) {
                    const int16_t dbra = smat.indices[md].first;
                    const int16_t dket = smat.indices[md].second;
                    int jz = jx[k + 1][0];
                    for (int jbra : jx[k + 1])
                        for (int jket : jx[k + 1]) {
                            if (!mats.count(make_pair(jbra, jket)))
                                continue;
                            if (shapes[jbra] == 0 || other->shapes[jket] == 0)
                                continue;
                            const int lbra = drt->jds[jbra][dbra],
                                      lket = other->drt->jds[jket][dket];
                            if (lbra == 0 || shapes[lbra] == 0 || lket == 0 ||
                                other->shapes[lket] == 0)
                                continue;
                            const int16_t kiq = other->drt->abc[lket][1];
                            const int16_t mdq = smat.dq;
                            const FL f =
                                (FL)(1 - (((kiq & 1) & (mdq & 1)) << 1));
                            if (!mats.count(make_pair(lbra, lket)))
                                mats[make_pair(lbra, lket)] = vector<FL>(
                                    shapes[lbra] * other->shapes[lket],
                                    (FL)0.0);
                            tmp.reserve(shapes[jbra] * other->shapes[lket]);
                            GMatrixFunctions<FL>::multiply(
                                GMatrix<FL>(mats[make_pair(jbra, jket)].data(),
                                            (MKL_INT)shapes[jbra],
                                            (MKL_INT)other->shapes[jket]),
                                false,
                                GMatrix<FL>(
                                    (FL *)other->data[k].data() +
                                        other->offsets[k]
                                                      [(jket - jz) * 4 + dket],
                                    (MKL_INT)other->shapes[jket],
                                    (MKL_INT)other->shapes[lket]),
                                false,
                                GMatrix<FL>(tmp.data(), (MKL_INT)shapes[jbra],
                                            (MKL_INT)other->shapes[lket]),
                                f * smat.data[md], (FL)0.0);
                            GMatrixFunctions<FL>::multiply(
                                GMatrix<FL>(
                                    (FL *)data[k].data() +
                                        offsets[k][(jbra - jz) * 4 + dbra],
                                    (MKL_INT)shapes[jbra],
                                    (MKL_INT)shapes[lbra]),
                                3,
                                GMatrix<FL>(tmp.data(), (MKL_INT)shapes[jbra],
                                            (MKL_INT)other->shapes[lket]),
                                false,
                                GMatrix<FL>(mats[make_pair(lbra, lket)].data(),
                                            (MKL_INT)shapes[lbra],
                                            (MKL_INT)other->shapes[lket]),
                                (FL)1.0, (FL)1.0);
                        }
                }
                ih -= hdrt->xs[jh * (hdrt->nd + 1) + dh];
                jh = jhv;
                for (int jbra : jx[k + 1])
                    for (int jket : jx[k + 1])
                        if (mats.count(make_pair(jbra, jket)))
                            mats[make_pair(jbra, jket)] = vector<FL>();
            }
            int mz = drt->n_rows() - 1;
            if (mats.count(make_pair(mz, mz)))
                r += (*ints)[ihx] * accumulate(mats[make_pair(mz, mz)].begin(),
                                               mats[make_pair(mz, mz)].end(),
                                               (FL)0.0, plus<FL>());
        }
        return r;
    }
};

template <typename S, typename FL, ElemOpTypes T = ElemT<S>::value>
struct HDRTMPO {
    typedef long long LL;
    shared_ptr<HDRT<S, T>> hdrt;
    vector<LL> shapes;
    vector<vector<LL>> offsets;
    vector<vector<FL>> data;
    static vector<vector<LL>> get_offsets(const shared_ptr<HDRT<S, T>> &hdrt,
                                          const vector<LL> &shapes) {
        vector<vector<LL>> r(hdrt->n_sites);
        for (int k = hdrt->n_sites - 1, j = 0; k >= 0; k--) {
            r[k].push_back(0);
            for (; j < hdrt->n_rows() && hdrt->qs[j][0] == k + 1; j++)
                for (int16_t d = 0; d < hdrt->nd; d++) {
                    LL x = shapes[j] * shapes[hdrt->jds[j * hdrt->nd + d]] *
                           (hdrt->jds[j * hdrt->nd + d] != 0);
                    r[k].push_back(r[k].back() + x);
                }
        }
        return r;
    }
    static vector<int> get_inv_jds(const shared_ptr<HDRT<S, T>> &hdrt) {
        vector<int> inv_jds(hdrt->n_rows() * hdrt->nd, -1);
        for (size_t j = 0; j < hdrt->n_rows(); j++)
            for (int16_t d = 0; d < hdrt->nd; d++)
                if (hdrt->jds[j * hdrt->nd + d] != 0)
                    inv_jds[hdrt->jds[j * hdrt->nd + d] * hdrt->nd + d] =
                        (int)j;
        return inv_jds;
    }
    HDRTMPO(const shared_ptr<HDRT<S, T>> &hdrt, const vector<LL> &shapes,
            const vector<vector<FL>> &data)
        : hdrt(hdrt), shapes(shapes), data(data) {
        if (this->shapes.size() == 0) {
            this->shapes.resize(hdrt->n_rows());
            for (size_t j = 0; j < hdrt->n_rows(); j++)
                this->shapes[j] = hdrt->xs[j * (hdrt->nd + 1) + hdrt->nd];
        }
        offsets = get_offsets(hdrt, this->shapes);
        if (this->data.size() == 0) {
            this->data.resize(hdrt->n_sites);
            for (int k = 0; k < hdrt->n_sites; k++)
                this->data[k] = vector<FL>(offsets[k].back());
        }
    }
    virtual ~HDRTMPO() = default;
    static shared_ptr<HDRTMPO>
    from_ci_vector(const shared_ptr<HDRT<S, T>> &hdrt, const FL *ci) {
        shared_ptr<HDRTMPO<S, FL>> r = make_shared<HDRTMPO<S, FL>>(
            hdrt, vector<LL>(), vector<vector<FL>>());
        vector<LL> ij(hdrt->n_init_qs + 1, 0);
        for (int i = 0; i < hdrt->n_init_qs; i++)
            ij[i + 1] = ij[i] + hdrt->xs[i * (hdrt->nd + 1) + hdrt->nd];
        for (int k = hdrt->n_sites - 1, j = 0, jz; k >= 0; k--)
            for (jz = j; j < hdrt->n_rows() && hdrt->qs[j][0] == k + 1; j++)
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if (hdrt->jds[j * hdrt->nd + d] != 0) {
                        GMatrix<FL> pd(
                            r->data[k].data() +
                                r->offsets[k][(j - jz) * hdrt->nd + d],
                            (MKL_INT)r->shapes[j],
                            (MKL_INT)r->shapes[hdrt->jds[j * hdrt->nd + d]]);
                        for (LL x = hdrt->xs[j * (hdrt->nd + 1) + d];
                             x < hdrt->xs[j * (hdrt->nd + 1) + d + 1]; x++)
                            pd((MKL_INT)x,
                               (MKL_INT)(x -
                                         hdrt->xs[j * (hdrt->nd + 1) + d])) =
                                j >= hdrt->n_init_qs ? (FL)1.0 : ci[x + ij[j]];
                    }
        return r;
    }
    vector<FL> to_ci_vector() const {
        vector<vector<int>> pkr(2, vector<int>());
        vector<vector<LL>> pk(2, vector<LL>());
        vector<vector<FL>> mats(2, vector<FL>());
        vector<vector<LL>> idxs(2, vector<LL>());
        vector<LL> bdims = get_bond_dimensions();
        LL max_bdim = *max_element(bdims.begin(), bdims.end());
        pkr[0].reserve(hdrt->size()), pk[0].reserve(hdrt->size());
        pkr[1].reserve(hdrt->size()), pk[1].reserve(hdrt->size());
        mats[0].reserve(max_bdim), mats[1].reserve(max_bdim);
        idxs[0].reserve(hdrt->size()), idxs[1].reserve(hdrt->size());
        LL x = 0, sx = 0;
        for (int i = 0; i < hdrt->n_init_qs; i++) {
            pkr[0].push_back(i);
            pk[0].push_back(x);
            idxs[0].push_back(sx);
            x += hdrt->xs[i * (hdrt->nd + 1) + hdrt->nd], sx += shapes[i];
        }
        vector<FL> ci(x);
        idxs[0].push_back(sx);
        for (LL p = 0; p < bdims.back(); p++)
            mats[0].push_back((FL)1.0);
        int pi = 0, pj = pi ^ 1;
        for (int k = hdrt->n_sites - 1, jz = 0; k >= 0; k--, pi ^= 1, pj ^= 1) {
            pkr[pj].clear(), pk[pj].clear();
            LL mat_sz = 0, idx_sz = 0;
            for (int j = 0; j < pkr[pi].size(); j++)
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if (hdrt->jds[pkr[pi][j] * hdrt->nd + d] != 0 &&
                        shapes[pkr[pi][j]] != 0 &&
                        shapes[hdrt->jds[pkr[pi][j] * hdrt->nd + d]] != 0)
                        mat_sz += shapes[hdrt->jds[pkr[pi][j] * hdrt->nd + d]],
                            idx_sz++;
            mats[pj].resize(mat_sz);
            idxs[pj].resize(idx_sz + 1);
            idxs[pj][idx_sz] = mat_sz;
            mat_sz = 0, idx_sz = 0;
            for (int j = 0; j < pkr[pi].size(); j++)
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if (hdrt->jds[pkr[pi][j] * hdrt->nd + d] != 0 &&
                        shapes[pkr[pi][j]] != 0 &&
                        shapes[hdrt->jds[pkr[pi][j] * hdrt->nd + d]] != 0) {
                        GMatrix<FL> ma(
                            mats[pi].data() + idxs[pi][j], 1,
                            (MKL_INT)(idxs[pi][j + 1] - idxs[pi][j]));
                        GMatrix<FL> mb(
                            (FL *)data[k].data() +
                                offsets[k][(pkr[pi][j] - jz) * hdrt->nd + d],
                            (MKL_INT)shapes[pkr[pi][j]],
                            (MKL_INT)
                                shapes[hdrt->jds[pkr[pi][j] * hdrt->nd + d]]);
                        GMatrix<FL> mc(
                            mats[pj].data() + mat_sz, 1,
                            (MKL_INT)
                                shapes[hdrt->jds[pkr[pi][j] * hdrt->nd + d]]);
                        GMatrixFunctions<FL>::multiply(ma, 0, mb, 0, mc,
                                                       (FL)1.0, (FL)0.0);
                        idxs[pj][idx_sz++] = mat_sz;
                        mat_sz += shapes[hdrt->jds[pkr[pi][j] * hdrt->nd + d]];
                        pkr[pj].push_back(hdrt->jds[pkr[pi][j] * hdrt->nd + d]);
                        pk[pj].push_back(
                            pk[pi][j] +
                            hdrt->xs[pkr[pi][j] * (hdrt->nd + 1) + d]);
                    }
            for (; jz < hdrt->n_rows() && hdrt->qs[jz][0] == k + 1; jz++)
                ;
        }
        for (int j = 0; j < pk[pi].size(); j++)
            ci[pk[pi][j]] += mats[pi][j];
        return ci;
    }
    vector<LL> get_bond_dimensions() const {
        vector<LL> r(hdrt->n_sites + 1, 0);
        for (int k = hdrt->n_sites, j = 0; k >= 0; k--)
            for (; j < hdrt->n_rows() && hdrt->qs[j][0] == k; j++)
                r[k] += shapes[j];
        return r;
    }
    shared_ptr<HDRTMPO> qr() const {
        vector<int> inv_jds = get_inv_jds(hdrt);
        vector<LL> new_shapes(hdrt->n_rows(), 1);
        for (int j = hdrt->n_init_qs; j < hdrt->n_rows(); j++) {
            LL x = 0;
            for (int16_t d = 0; d < hdrt->nd; d++)
                if (inv_jds[j * hdrt->nd + d] != -1)
                    x += new_shapes[inv_jds[j * hdrt->nd + d]];
            new_shapes[j] = min(x, shapes[j]);
        }
        shared_ptr<HDRTMPO> r =
            make_shared<HDRTMPO<S, FL>>(hdrt, new_shapes, vector<vector<FL>>());
        vector<vector<FL>> gauges(hdrt->n_rows());
        vector<FL> matx, matq;
        for (int i = 0; i < hdrt->n_init_qs; i++)
            gauges[i] = vector<FL>(shapes[i], (FL)1.0);
        for (int k = hdrt->n_sites - 1, j = hdrt->n_init_qs, jz = 0, pz, l;
             k >= 0; k--) {
            for (pz = jz, jz = j; j < hdrt->n_rows() && hdrt->qs[j][0] == k;
                 j++) {
                LL mz = 0;
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if (inv_jds[j * hdrt->nd + d] != -1)
                        mz += new_shapes[inv_jds[j * hdrt->nd + d]];
                matx.resize(mz * shapes[j]);
                mz = 0;
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = inv_jds[j * hdrt->nd + d]) != -1 &&
                        new_shapes[l] != 0) {
                        if (shapes[j] != 0)
                            GMatrixFunctions<FL>::multiply(
                                GMatrix<FL>(gauges[l].data(),
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)shapes[l]),
                                false,
                                GMatrix<FL>(
                                    (FL *)data[k].data() +
                                        offsets[k][(l - pz) * hdrt->nd + d],
                                    (MKL_INT)shapes[l], (MKL_INT)shapes[j]),
                                false,
                                GMatrix<FL>(matx.data() + mz * shapes[j],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)shapes[j]),
                                (FL)1.0, (FL)0.0);
                        mz += new_shapes[l];
                    }
                matq.resize(mz * new_shapes[j]);
                gauges[j].resize(new_shapes[j] * shapes[j]);
                if (new_shapes[j] == 0 || shapes[j] == 0)
                    continue;
                if (k != 0) {
                    GMatrixFunctions<FL>::qr(
                        GMatrix<FL>(matx.data(), (MKL_INT)mz,
                                    (MKL_INT)shapes[j]),
                        GMatrix<FL>(matq.data(), (MKL_INT)mz,
                                    (MKL_INT)new_shapes[j]),
                        GMatrix<FL>(gauges[j].data(), (MKL_INT)new_shapes[j],
                                    (MKL_INT)shapes[j]));
                    mz = 0;
                    for (int16_t d = 0; d < hdrt->nd; d++)
                        if ((l = inv_jds[j * hdrt->nd + d]) != -1) {
                            GMatrixFunctions<FL>::copy(
                                GMatrix<FL>(
                                    r->data[k].data() +
                                        r->offsets[k][(l - pz) * hdrt->nd + d],
                                    (MKL_INT)new_shapes[l],
                                    (MKL_INT)new_shapes[j]),
                                GMatrix<FL>(matq.data() + mz * new_shapes[j],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)new_shapes[j]));
                            mz += new_shapes[l];
                        }
                } else {
                    assert(shapes[j] == new_shapes[j]);
                    mz = 0;
                    for (int16_t d = 0; d < hdrt->nd; d++)
                        if ((l = inv_jds[j * hdrt->nd + d]) != -1) {
                            GMatrixFunctions<FL>::copy(
                                GMatrix<FL>(
                                    r->data[k].data() +
                                        r->offsets[k][(l - pz) * hdrt->nd + d],
                                    (MKL_INT)new_shapes[l],
                                    (MKL_INT)new_shapes[j]),
                                GMatrix<FL>(matx.data() + mz * new_shapes[j],
                                            (MKL_INT)new_shapes[l],
                                            (MKL_INT)new_shapes[j]));
                            mz += new_shapes[l];
                        }
                }
            }
            for (l = pz; l < jz; l++)
                gauges[l] = vector<FL>();
        }
        return r;
    }
    shared_ptr<HDRTMPO> svd(int max_bond_dim = -1, FL cutoff = (FL)0.0) const {
        vector<vector<FL>> gauges(hdrt->n_rows());
        vector<LL> new_shapes(hdrt->n_rows(), 1);
        vector<vector<int>> jx(hdrt->n_sites + 1);
        for (int k = hdrt->n_sites, j = 0; k >= 0; k--)
            for (; j < hdrt->n_rows() && hdrt->qs[j][0] == k; j++)
                jx[k].push_back(j);
        int mjx = 0;
        for (int k = 0; k < hdrt->n_sites; k++)
            mjx = max(mjx, (int)jx[k + 1].size());
        for (int i = 0; i < jx[0].size(); i++)
            gauges[jx[0][i]] = vector<FL>(shapes[jx[0][i]], (FL)1.0);
        int jz = jx[0][0];
        vector<vector<FL>> tmps(mjx), tmpl(mjx), tmpr(mjx);
        vector<FL> tmpx, s_vals;
        vector<vector<FL>> new_data(hdrt->n_sites);
        vector<vector<LL>> new_offsets(hdrt->n_sites);
        for (int k = 0, l; k < hdrt->n_sites; k++) {
            jz = jx[k + 1][0];
            LL total_s = 0;
            for (int j : jx[k + 1]) {
                LL nx = 0, mz = 0, mm;
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = hdrt->jds[j * hdrt->nd + d]) != 0)
                        nx += new_shapes[l];
                tmpx.reserve(nx * shapes[j]);
                mm = min(nx, shapes[j]);
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = hdrt->jds[j * hdrt->nd + d]) != 0 &&
                        new_shapes[l] != 0) {
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>((FL *)data[k].data() +
                                            offsets[k][(j - jz) * hdrt->nd + d],
                                        (MKL_INT)shapes[j], (MKL_INT)shapes[l]),
                            false,
                            GMatrix<FL>(gauges[l].data(), (MKL_INT)shapes[l],
                                        (MKL_INT)new_shapes[l]),
                            false,
                            GMatrix<FL>(tmpx.data() + mz, (MKL_INT)shapes[j],
                                        (MKL_INT)nx),
                            (FL)1.0, (FL)0.0);
                        mz += new_shapes[l];
                    }
                if (nx != 0) {
                    tmps[j - jz].reserve(mm);
                    tmpl[j - jz].reserve(shapes[j] * mm);
                    tmpr[j - jz].reserve(mm * nx);
                    if (mm != 0) {
                        GMatrixFunctions<FL>::svd(
                            GMatrix<FL>(tmpx.data(), (MKL_INT)shapes[j],
                                        (MKL_INT)nx),
                            GMatrix<FL>(tmpl[j - jz].data(), (MKL_INT)shapes[j],
                                        (MKL_INT)mm),
                            GMatrix<FL>(tmps[j - jz].data(), 1, (MKL_INT)mm),
                            GMatrix<FL>(tmpr[j - jz].data(), (MKL_INT)mm,
                                        (MKL_INT)nx));
                        if (k == hdrt->n_sites - 1) {
                            assert(mm == 1 && shapes[j] == 1);
                            for (LL i = 0; i < nx; i++)
                                tmpr[j - jz][i] *=
                                    tmps[j - jz][0] * tmpl[j - jz][0];
                        }
                    }
                }
                total_s += mm;
            }
            s_vals.reserve(total_s);
            total_s = 0;
            for (int j : jx[k + 1]) {
                LL nx = 0;
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = hdrt->jds[j * hdrt->nd + d]) != 0)
                        nx += new_shapes[l];
                LL mm = min(nx, shapes[j]);
                if (mm != 0)
                    GMatrixFunctions<FL>::copy(
                        GMatrix<FL>(s_vals.data() + total_s, 1, (MKL_INT)mm),
                        GMatrix<FL>(tmps[j - jz].data(), 1, (MKL_INT)mm));
                total_s += mm;
            }
            vector<LL> srt_idx(total_s);
            for (LL i = 0; i < total_s; i++)
                srt_idx[i] = i;
            sort(srt_idx.begin(), srt_idx.end(),
                 [&s_vals](LL a, LL b) { return s_vals[a] > s_vals[b]; });
            if (max_bond_dim != -1 && (int)srt_idx.size() > max_bond_dim)
                srt_idx.resize(max_bond_dim);
            LL jxx = 0;
            for (LL ix = 0; ix < (LL)srt_idx.size(); ix++) {
                srt_idx[jxx] = srt_idx[ix];
                jxx += (s_vals[srt_idx[ix]] >= cutoff);
            }
            srt_idx.resize(jxx);
            sort(srt_idx.begin(), srt_idx.end());
            new_offsets[k].push_back(0);
            LL isrt = 0, isrtz;
            total_s = 0;
            for (int j : jx[k + 1]) {
                LL nx = 0;
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = hdrt->jds[j * hdrt->nd + d]) != 0)
                        nx += new_shapes[l];
                LL mm = min(nx, shapes[j]);
                for (isrtz = isrt;
                     isrt < (LL)srt_idx.size() && srt_idx[isrt] - total_s < mm;)
                    isrt++;
                new_shapes[j] = isrt - isrtz;
                gauges[j].resize(shapes[j] * new_shapes[j]);
                for (LL g = 0; g < shapes[j]; g++)
                    for (LL jsrt = isrtz; jsrt < isrt; jsrt++)
                        gauges[j][g * new_shapes[j] + (jsrt - isrtz)] =
                            tmpl[j - jz][g * mm + (srt_idx[jsrt] - total_s)] *
                            s_vals[srt_idx[jsrt]];
                for (int16_t d = 0; d < hdrt->nd; d++) {
                    LL x = new_shapes[j] *
                           new_shapes[hdrt->jds[j * hdrt->nd + d]] *
                           (hdrt->jds[j * hdrt->nd + d] != 0);
                    new_offsets[k].push_back(new_offsets[k].back() + x);
                }
                total_s += mm;
            }
            new_data[k] = vector<FL>(new_offsets[k].back());
            total_s = 0, isrt = 0;
            for (int j : jx[k + 1]) {
                isrtz = isrt;
                isrt += new_shapes[j];
                LL mz = 0, nx = 0;
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = hdrt->jds[j * hdrt->nd + d]) != 0)
                        nx += new_shapes[l];
                for (int16_t d = 0; d < hdrt->nd; d++)
                    if ((l = hdrt->jds[j * hdrt->nd + d]) != 0 &&
                        new_shapes[l] != 0) {
                        for (LL jsrt = isrtz; jsrt < isrt; jsrt++)
                            GMatrixFunctions<FL>::copy(
                                GMatrix<FL>(
                                    new_data[k].data() +
                                        new_offsets[k]
                                                   [(j - jz) * hdrt->nd + d] +
                                        (jsrt - isrtz) * new_shapes[l],
                                    1, (MKL_INT)new_shapes[l]),
                                GMatrix<FL>(tmpr[j - jz].data() +
                                                (srt_idx[jsrt] - total_s) * nx +
                                                mz,
                                            1, (MKL_INT)new_shapes[l]));
                        mz += new_shapes[l];
                    }
                total_s += min(nx, shapes[j]);
            }
        }
        return make_shared<HDRTMPO<S, FL>>(hdrt, new_shapes, new_data);
    }
};

} // namespace block2
