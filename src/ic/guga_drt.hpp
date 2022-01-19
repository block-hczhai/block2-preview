
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

/** Distinct Row Table in The Graphical Unitary Group Approach (GUGA). */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

using namespace std;

namespace block2 {

struct PaldusTable {
    vector<array<int16_t, 3>> abc;
    PaldusTable() {}
    PaldusTable(int n) : PaldusTable() { abc.resize(n); }
    virtual ~PaldusTable() = default;
    bool sanity_check() const {
        int n = n_rows();
        // QCDES Eq. (6.41)
        for (int i = 0; i < n; i++)
            if (abc[i][0] + abc[i][1] + abc[i][2] != n - i)
                return false;
            else if (abc[i][0] < 0 || abc[i][1] < 0 || abc[i][2] < 0)
                return false;
        PaldusTable d = diff();
        for (int i = 0; i < n; i++)
            if (!((d.abc[i][0] == 1 && d.abc[i][1] == 0 && d.abc[i][2] == 0) ||
                  (d.abc[i][0] == 0 && d.abc[i][1] == 1 && d.abc[i][2] == 0) ||
                  (d.abc[i][0] == 0 && d.abc[i][1] == 0 && d.abc[i][2] == 1) ||
                  (d.abc[i][0] == 1 && d.abc[i][1] == -1 && d.abc[i][2] == 1)))
                return false;
        return true;
    }
    int n_rows() const { return (int)abc.size(); }
    // QCDES Eq. (6.45)
    int16_t n_elec() const { return 2 * abc[0][0] + abc[0][1]; }
    // QCDES Eq. (6.46)
    int16_t twos() const { return abc[0][1]; }
    // PaldusTable to variation-tables QCDES Eq. (6.48)
    PaldusTable diff() const {
        int n = n_rows();
        PaldusTable r(n);
        r.abc.back() = abc.back();
        for (int i = n - 2; i >= 0; i--)
            for (int j = 0; j < 3; j++)
                r.abc[i][j] = abc[i][j] - abc[i + 1][j];
        return r;
    }
    // variation-tables to PaldusTable
    PaldusTable accu() const {
        int n = n_rows();
        PaldusTable r(n);
        r.abc.back() = abc.back();
        for (int i = n - 2; i >= 0; i--)
            for (int j = 0; j < 3; j++)
                r.abc[i][j] = r.abc[i + 1][j] + abc[i][j];
        return r;
    }
    // QCDES Eq. (6.52)
    // +: 1  -: 2
    // asssume current table is diff
    vector<uint8_t> to_step_vector() const {
        int n = n_rows();
        vector<uint8_t> r(n);
        for (int i = n - 1; i >= 0; i--)
            r[n - 1 - i] = abc[i][0] + abc[i][0] - abc[i][2] + 1;
        return r;
    }
    static PaldusTable from_step_vector(const vector<uint8_t> &ds) {
        int n = (int)ds.size();
        PaldusTable r(n);
        for (int i = n - 1; i >= 0; i--)
            switch (ds[n - 1 - i]) {
            case 0:
                r.abc[i][0] = 0, r.abc[i][1] = 0, r.abc[i][2] = 1;
                break;
            case 1:
                r.abc[i][0] = 0, r.abc[i][1] = 1, r.abc[i][2] = 0;
                break;
            case 2:
                r.abc[i][0] = 1, r.abc[i][1] = -1, r.abc[i][2] = 1;
                break;
            case 3:
                r.abc[i][0] = 1, r.abc[i][1] = 0, r.abc[i][2] = 0;
                break;
            default:
                assert(false);
            }
        return r;
    }
    virtual string to_str() const {
        stringstream ss;
        ss << setw(4) << "" << setw(4) << "A" << setw(4) << "B" << setw(4)
           << "C" << endl;
        int n = n_rows();
        for (int i = 0; i < n; i++)
            ss << setw(4) << n - i << setw(4) << abc[i][0] << setw(4)
               << abc[i][1] << setw(4) << abc[i][2] << endl;
        return ss.str();
    }
};

template <typename S = void, typename = void> struct DistinctRowTable;

template <typename S>
struct DistinctRowTable<S, typename enable_if<is_void<S>::value>::type>
    : PaldusTable {
    typedef long long LL;
    vector<array<int, 4>> jd;
    vector<array<LL, 4>> xs;
    DistinctRowTable() {}
    DistinctRowTable(int16_t a, int16_t b, int16_t c) : DistinctRowTable() {
        abc.push_back(array<int16_t, 3>{a, b, c});
    }
    virtual ~DistinctRowTable() = default;
    static bool cmp_row_ab(const array<int16_t, 3> &p,
                           const pair<int16_t, int16_t> &q) {
        return q.first == p[0] ? q.second < p[1] : q.first < p[0];
    }
    int find_row(int16_t a, int16_t b, int start = 0) const {
        auto p = lower_bound(abc.begin() + start, abc.end(), make_pair(a, b),
                             cmp_row_ab);
        if (p == abc.end() || (*p)[0] != a || (*p)[1] != b)
            return -1;
        else
            return (int)(p - abc.begin());
    }
    int n_drt() const {
        int a = abc[0][0], b = abc[0][1], c = abc[0][2];
        int d = min(a, c);
        return (a + 1) * (c + 1) * (b + b + 2 + d) / 2 -
               d * (d + 1) * (d + 2) / 6;
    }
    virtual void initialize() {
        int nd = n_drt();
        abc.resize(1);
        jd.clear();
        abc.reserve(nd);
        jd.reserve(nd);
        vector<array<int16_t, 2>> tmp;
        tmp.reserve(nd * 4);
        auto tmp_cmp = [](array<int16_t, 2> p, array<int16_t, 2> q) {
            return q[0] == p[0] ? q[1] < p[1] : q[0] < p[0];
        };
        int16_t k = abc[0][0] + abc[0][1] + abc[0][2];
        // d = 0: ; d = 1: b--; d = 2:a--,b++; d = 3:a--;
        for (int ir = 0, nr = 1, it = 0; ir < abc.size(); ir += nr, k--) {
            nr = (int)abc.size() - ir;
            it = 0;
            for (int i = 0; i < nr; i++) {
                array<int16_t, 3> p = abc[i + ir];
                for (int16_t dk = 0; dk < 4; dk++) {
                    if ((p[2] == 0 && (dk == 0 || dk == 2)) ||
                        (p[1] == 0 && dk == 1) ||
                        (p[0] == 0 && (dk == 2 || dk == 3)))
                        continue;
                    const int16_t xa = (int16_t)(p[0] - (dk >> 1));
                    const int16_t xb = (int16_t)(p[1] - (dk & 1) + (dk >> 1));
                    tmp[it++] = array<int16_t, 2>{xa, xb};
                }
            }
            sort(tmp.begin(), tmp.begin() + it, tmp_cmp);
            for (int i = 0; i < it; i++)
                if (i == 0 || tmp[i][0] != tmp[i - 1][0] ||
                    tmp[i][1] != tmp[i - 1][1])
                    abc.push_back(array<int16_t, 3>{
                        tmp[i][0], tmp[i][1],
                        (int16_t)(k - 1 - tmp[i][0] - tmp[i][1])});
            for (int i = 0; i < nr; i++) {
                array<int16_t, 3> p = abc[i + ir];
                jd.push_back(array<int, 4>{0, 0, 0, 0});
                for (int16_t dk = 0; dk < 4; dk++) {
                    if ((p[2] == 0 && (dk == 0 || dk == 2)) ||
                        (p[1] == 0 && dk == 1) ||
                        (p[0] == 0 && (dk == 2 || dk == 3)))
                        continue;
                    const int16_t xa = (int16_t)(p[0] - (dk >> 1));
                    const int16_t xb = (int16_t)(p[1] - (dk & 1) + (dk >> 1));
                    const int16_t xc = (int16_t)(p[2] - !(dk & 1));
                    const int px = find_row(xa, xb, ir + nr);
                    assert(px != -1);
                    jd.back()[dk] = px;
                }
            }
        }
        int n = n_rows();
        xs.resize(n);
        xs.back() = array<LL, 4>{0, 0, 0, 1};
        for (int i = n - 2; i >= 0; i--) {
            xs[i] = array<LL, 4>{0, 0, 0, 0};
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    xs[i][dk] = xs[jd[i][dk]][3];
            for (int16_t dk = 1; dk < 4; dk++)
                xs[i][dk] += xs[i][dk - 1];
        }
    }
    string to_str() const override {
        if (jd.size() == 0)
            return PaldusTable::to_str();
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "A" << setw(4)
           << "B" << setw(4) << "C" << setw(6) << "JD0" << setw(6) << "JD1"
           << setw(6) << "JD2" << setw(6) << "JD3"
           << " " << setw(12) << "X0"
           << " " << setw(12) << "X1"
           << " " << setw(12) << "X2"
           << " " << setw(12) << "X3" << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            k = abc[i][0] + abc[i][1] + abc[i][2];
            ss << setw(4) << (i + 1);
            if (k == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << k;
            ss << setw(4) << abc[i][0] << setw(4) << abc[i][1] << setw(4)
               << abc[i][2];
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] == 0)
                    ss << setw(6) << "";
                else
                    ss << setw(6) << jd[i][dk] + 1;
            for (int16_t dk = 0; dk < 4; dk++)
                ss << " " << setw(12) << xs[i][dk];
            ss << endl;
        }
        return ss.str();
    }
    vector<int> step_vector_to_arc(const vector<uint8_t> &ds) const {
        vector<int> r;
        r.reserve(ds.size() + 1);
        r.push_back(0);
        for (int i = (int)ds.size() - 1; i >= 0; i--)
            r.push_back(jd[r.back()][ds[i]]);
        return r;
    }
    LL index_of_step_vector(const vector<uint8_t> &ds) const {
        LL r = 0;
        int x = 0;
        for (int i = (int)ds.size() - 1; i >= 0; i--) {
            r += ds[i] == 0 ? 0 : xs[x][ds[i] - 1];
            x = jd[x][ds[i]];
        }
        return r;
    }
};

template <typename S>
struct DistinctRowTable<S, typename enable_if<!is_void<S>::value>::type>
    : DistinctRowTable<void> {
    vector<typename S::pg_t> orb_sym;
    vector<typename S::pg_t> pgs;
    DistinctRowTable(const vector<typename S::pg_t> &orb_sym)
        : orb_sym(orb_sym), DistinctRowTable<void>() {}
    DistinctRowTable(int16_t a, int16_t b, int16_t c, typename S::pg_t pg,
                     const vector<typename S::pg_t> &orb_sym)
        : DistinctRowTable(orb_sym) {
        abc.push_back(array<int16_t, 3>{a, b, c});
        pgs.push_back(pg);
    }
    virtual ~DistinctRowTable() = default;
    void initialize() override {
        int nd = n_drt();
        abc.resize(1);
        pgs.resize(1);
        jd.clear();
        abc.reserve(nd);
        jd.reserve(nd);
        pgs.reserve(nd);
        vector<pair<array<int16_t, 2>, typename S::pg_t>> tmp;
        vector<int> idx, ridx;
        auto tmp_cmp = [&tmp](int ip, int iq) {
            return tmp[ip].first[0] != tmp[iq].first[0]
                       ? tmp[ip].first[0] > tmp[iq].first[0]
                       : (tmp[ip].first[1] != tmp[iq].first[1]
                              ? tmp[ip].first[1] > tmp[iq].first[1]
                              : (int)tmp[ip].second > (int)tmp[iq].second);
        };
        int16_t k = abc[0][0] + abc[0][1] + abc[0][2];
        // d = 0: ; d = 1: b--; d = 2:a--,b++; d = 3:a--;
        for (int ir = 0, nr = 1, it = 0; ir < abc.size(); ir += nr, k--) {
            nr = (int)abc.size() - ir;
            it = 0;
            tmp.reserve(nr * 4);
            for (int i = 0; i < nr; i++) {
                array<int16_t, 3> p = abc[i + ir];
                jd.push_back(array<int, 4>{0, 0, 0, 0});
                for (int16_t dk = 0; dk < 4; dk++) {
                    if ((p[2] == 0 && (dk == 0 || dk == 2)) ||
                        (p[1] == 0 && dk == 1) ||
                        (p[0] == 0 && (dk == 2 || dk == 3)))
                        continue;
                    const int16_t xa = (int16_t)(p[0] - (dk >> 1));
                    const int16_t xb = (int16_t)(p[1] - (dk & 1) + (dk >> 1));
                    const typename S::pg_t xpg =
                        dk == 0 || dk == 3
                            ? pgs[i + ir]
                            : S::pg_mul(pgs[i + ir], S::pg_inv(orb_sym[k - 1]));
                    tmp[it] = make_pair(array<int16_t, 2>{xa, xb}, xpg);
                    if (k - 1 == 0 && xpg != 0)
                        continue;
                    jd.back()[dk] = ++it;
                }
            }
            idx.reserve(it);
            for (int i = 0; i < it; i++)
                idx[i] = i;
            sort(idx.begin(), idx.begin() + it, tmp_cmp);
            ridx.reserve(it + 1);
            ridx[0] = 0;
            for (int i = 0; i < it; i++)
                if (i == 0 || tmp_cmp(idx[i - 1], idx[i])) {
                    int ii = idx[i];
                    abc.push_back(
                        array<int16_t, 3>{tmp[ii].first[0], tmp[ii].first[1],
                                          (int16_t)(k - 1 - tmp[ii].first[0] -
                                                    tmp[ii].first[1])});
                    pgs.push_back(tmp[ii].second);
                    ridx[idx[i] + 1] = (int)abc.size() - 1;
                } else
                    ridx[idx[i] + 1] = ridx[idx[i - 1] + 1];
            for (int i = 0; i < nr; i++)
                for (int16_t dk = 0; dk < 4; dk++)
                    jd[i + ir][dk] = ridx[jd[i + ir][dk]];
        }
        int n = n_rows();
        idx.reserve(n);
        for (int i = n - 1; i >= 0; i--) {
            idx[i] = 0;
            if (abc[i][0] == 0 && abc[i][1] == 0 && abc[i][2] == 0)
                idx[i] = 1;
            else
                for (int16_t dk = 0; dk < 4; dk++)
                    if (jd[i][dk] != 0 && idx[jd[i][dk]])
                        idx[i] = 1;
        }
        int xn = 0;
        for (int i = 0; i < n; i++)
            if (idx[i]) {
                if (i != xn) {
                    abc[xn] = abc[i];
                    jd[xn] = jd[i];
                    pgs[xn] = pgs[i];
                }
                idx[i] = xn++;
            }
        n = xn;
        for (int i = 0; i < n; i++)
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    jd[i][dk] = idx[jd[i][dk]];
        abc.resize(n);
        jd.resize(n);
        pgs.resize(n);
        xs.resize(n);
        for (int i = n - 1; i >= 0; i--) {
            xs[i] = array<LL, 4>{0, 0, 0, 0};
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    xs[i][dk] = xs[jd[i][dk]][3];
            for (int16_t dk = 1; dk < 4; dk++)
                xs[i][dk] += xs[i][dk - 1];
            if (abc[i][0] == 0 && abc[i][1] == 0 && abc[i][2] == 0)
                xs[i][3] = 1;
        }
    }
    string to_str() const override {
        if (jd.size() == 0)
            return PaldusTable::to_str();
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "A" << setw(4)
           << "B" << setw(4) << "C" << setw(6) << "PG" << setw(6) << "JD0"
           << setw(6) << "JD1" << setw(6) << "JD2" << setw(6) << "JD3"
           << " " << setw(12) << "X0"
           << " " << setw(12) << "X1"
           << " " << setw(12) << "X2"
           << " " << setw(12) << "X3" << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            k = abc[i][0] + abc[i][1] + abc[i][2];
            ss << setw(4) << (i + 1);
            if (k == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << k;
            ss << setw(4) << abc[i][0] << setw(4) << abc[i][1] << setw(4)
               << abc[i][2];
            ss << setw(6) << (int)pgs[i];
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] == 0)
                    ss << setw(6) << "";
                else
                    ss << setw(6) << jd[i][dk] + 1;
            for (int16_t dk = 0; dk < 4; dk++)
                ss << " " << setw(12) << xs[i][dk];
            ss << endl;
        }
        return ss.str();
    }
};

template <typename S, typename = void> struct MRCIDistinctRowTable;

template <typename S>
struct MRCIDistinctRowTable<S, typename enable_if<is_void<S>::value>::type>
    : DistinctRowTable<S> {
    using typename DistinctRowTable<S>::LL;
    using DistinctRowTable<S>::abc;
    using DistinctRowTable<S>::jd;
    using DistinctRowTable<S>::xs;
    using DistinctRowTable<S>::n_rows;
    using DistinctRowTable<S>::n_drt;
    vector<int16_t> ts;
    vector<vector<uint8_t>> refs;
    int nref = 0, nex = 0, nvirt = 0, ncore = 0;
    MRCIDistinctRowTable() : DistinctRowTable<S>() {}
    MRCIDistinctRowTable(int16_t a, int16_t b, int16_t c)
        : MRCIDistinctRowTable() {
        abc.push_back(array<int16_t, 3>{a, b, c});
    }
    virtual ~MRCIDistinctRowTable() = default;
    virtual void initialize_mrci(int ci_order,
                                 const vector<vector<uint8_t>> &refs) {
        nref = (int)refs.size();
        assert(nref > 0);
        nex = ci_order;
        nvirt = 0;
        for (int i = 0; i < (int)refs[0].size(); i++)
            if (all_of(refs.begin(), refs.end(),
                       [i](const vector<uint8_t> &ref) { return ref[i] == 0; }))
                nvirt++;
            else
                break;
        ncore = 0;
        for (int i = nvirt; i < (int)refs[0].size(); i++)
            if (all_of(refs.begin(), refs.end(),
                       [i](const vector<uint8_t> &ref) { return ref[i] == 3; }))
                ncore++;
            else
                break;
        cout << "nv nex = " << nvirt << " " << nex << endl;
        int nd = n_drt();
        abc.resize(1);
        ts.clear();
        ts.reserve(nd);
        for (int i = 0; i < nref; i++)
            ts.push_back(0);
        jd.clear();
        abc.reserve(nd);
        jd.reserve(nd);
        int tt = 2 + nref;
        vector<int16_t> tmp;
        vector<int> idx, ridx;
        tmp.reserve(nd * 4);
        auto tmp_cmp = [&tmp, &tt](int ip, int iq) {
            for (int i = 0; i < tt; i++)
                if (tmp[ip * tt + i] != tmp[iq * tt + i])
                    return tmp[ip * tt + i] > tmp[iq * tt + i];
            return false;
        };
        // rd == 3 -> 0
        // rd == 12 d == 3 || d == 12 rd == 0 -> 1
        // rd == 0 d == 3 -> 2
        auto arc_t = [](int16_t rd, int16_t d) {
            return (((rd >> 1) ^ (rd & 1)) & (d == 3)) |
                   (((!rd) & (d >> 1)) + ((!rd) & (d & 1)));
        };
        int16_t k = abc[0][0] + abc[0][1] + abc[0][2];
        // d = 0: ; d = 1: b--; d = 2:a--,b++; d = 3:a--;
        vector<int> ref_idx(nref * 2, 0);
        for (int ir = 0, nr, it; ir < abc.size(); ir += nr, k--) {
            nr = (int)abc.size() - ir;
            it = 0;
            tmp.reserve(nr * 4 * tt);
            for (int i = 0; i < nr; i++) {
                array<int16_t, 3> p = abc[i + ir];
                jd.push_back(array<int, 4>{0, 0, 0, 0});
                for (int16_t dk = 0; dk < 4; dk++) {
                    if ((p[2] == 0 && (dk == 0 || dk == 2)) ||
                        (p[1] == 0 && dk == 1) ||
                        (p[0] == 0 && (dk == 2 || dk == 3)))
                        continue;
                    tmp[it * tt] = (int16_t)(p[0] - (dk >> 1));
                    tmp[it * tt + 1] = (int16_t)(p[1] - (dk & 1) + (dk >> 1));
                    const int16_t xc =
                        (int16_t)(k - 1 - tmp[it * tt] - tmp[it * tt + 1]);
                    if (k - 1 <= nvirt && xc < k - 1 - nex)
                        continue;
                    // at the act/core boundry, 2 spin cannot be larger than
                    // the excitation level
                    if (k - 1 == ncore + nvirt && tmp[it * tt + 1] > nex)
                        continue;
                    bool t_valid = false;
                    if (k - 1 >= nvirt && k - 1 < ncore + nvirt) {
                        const int16_t mt = ts[(i + ir) * nref];
                        const int16_t xt = mt + arc_t(refs[0][k - 1], dk);
                        t_valid = t_valid | (xt <= nex);
                        for (int j = 0; j < nref; j++)
                            tmp[it * tt + 2 + j] = xt;
                    } else {
                        for (int j = 0; j < nref; j++) {
                            if (ref_idx[j] == i + ir &&
                                ((refs[j][k - 1] ^ dk) == 3) &&
                                ((dk >> 1) ^ (dk & 1)))
                                assert(ts[(i + ir) * nref + j] == 0);
                            // here if 1/2 2/1 diff outside the occ change
                            // it should be count as one excitation
                            const int16_t xt = ts[(i + ir) * nref + j] +
                                               ((ref_idx[j] == i + ir) &
                                                ((refs[j][k - 1] ^ dk) == 3) &
                                                ((dk >> 1) ^ (dk & 1))) +
                                               arc_t(refs[j][k - 1], dk);
                            t_valid = t_valid | (xt <= nex);
                            tmp[it * tt + 2 + j] = xt;
                        }
                        if (k - 1 == ncore + nvirt) {
                            const int16_t mt = *min_element(
                                &tmp[it * tt + 2], &tmp[(it + 1) * tt]);
                            for (int j = 0; j < nref; j++)
                                tmp[it * tt + 2 + j] = mt;
                        }
                    }
                    if (t_valid) {
                        for (int j = 0; j < nref; j++)
                            if (ref_idx[j] == i + ir && refs[j][k - 1] == dk)
                                ref_idx[j + nref] = it + 1;
                        jd.back()[dk] = ++it;
                    }
                }
            }
            idx.reserve(it);
            for (int i = 0; i < it; i++)
                idx[i] = i;
            sort(idx.begin(), idx.begin() + it, tmp_cmp);
            ridx.reserve(it + 1);
            ridx[0] = 0;
            for (int i = 0; i < it; i++)
                if (i == 0 || tmp_cmp(idx[i - 1], idx[i])) {
                    int ii = idx[i] * tt;
                    abc.push_back(array<int16_t, 3>{
                        tmp[ii + 0], tmp[ii + 1],
                        (int16_t)(k - 1 - tmp[ii + 0] - tmp[ii + 1])});
                    for (int j = 0; j < nref; j++)
                        ts.push_back(tmp[ii + 2 + j]);
                    ridx[idx[i] + 1] = (int)abc.size() - 1;
                } else
                    ridx[idx[i] + 1] = ridx[idx[i - 1] + 1];
            for (int i = 0; i < nr; i++)
                for (int16_t dk = 0; dk < 4; dk++)
                    jd[i + ir][dk] = ridx[jd[i + ir][dk]];
            for (int j = 0; j < nref; j++)
                ref_idx[j] = ridx[ref_idx[j + nref]];
        }
        int n = n_rows();
        idx.reserve(n);
        for (int i = n - 1; i >= 0; i--) {
            idx[i] = 0;
            if (abc[i][0] == 0 && abc[i][1] == 0 && abc[i][2] == 0)
                idx[i] = 1;
            else
                for (int16_t dk = 0; dk < 4; dk++)
                    if (jd[i][dk] != 0 && idx[jd[i][dk]])
                        idx[i] = 1;
        }
        int xn = 0;
        for (int i = 0; i < n; i++)
            if (idx[i]) {
                if (i != xn) {
                    abc[xn] = abc[i];
                    jd[xn] = jd[i];
                    memcpy(&ts[xn * nref], &ts[i * nref],
                           sizeof(int16_t) * nref);
                }
                if (xn == 0 || abc[xn][0] + abc[xn][1] + abc[xn][2] > nvirt ||
                    abc[xn - 1][0] != abc[xn][0] ||
                    abc[xn - 1][1] != abc[xn][1] ||
                    abc[xn - 1][2] != abc[xn][2])
                    idx[i] = xn++;
                else
                    idx[i] = xn - 1;
            }
        n = xn;
        for (int i = 0; i < n; i++)
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    jd[i][dk] = idx[jd[i][dk]];
        abc.resize(n);
        jd.resize(n);
        ts.resize(n * nref);
        xs.resize(n);
        for (int i = n - 1; i >= 0; i--) {
            xs[i] = array<LL, 4>{0, 0, 0, 0};
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    xs[i][dk] = xs[jd[i][dk]][3];
            for (int16_t dk = 1; dk < 4; dk++)
                xs[i][dk] += xs[i][dk - 1];
            if (abc[i][0] == 0 && abc[i][1] == 0 && abc[i][2] == 0)
                xs[i][3] = 1;
        }
    }
    string to_str() const override {
        if (jd.size() == 0)
            return PaldusTable::to_str();
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "A" << setw(4)
           << "B" << setw(4) << "C";
        for (int j = 0; j < nref; j++) {
            stringstream sr;
            sr << "T" << j;
            ss << setw(4) << sr.str();
        }
        ss << setw(6) << "JD0" << setw(6) << "JD1" << setw(6) << "JD2"
           << setw(6) << "JD3"
           << " " << setw(12) << "X0"
           << " " << setw(12) << "X1"
           << " " << setw(12) << "X2"
           << " " << setw(12) << "X3" << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            k = abc[i][0] + abc[i][1] + abc[i][2];
            ss << setw(4) << (i + 1);
            if (k == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << k;
            ss << setw(4) << abc[i][0] << setw(4) << abc[i][1] << setw(4)
               << abc[i][2];
            for (int j = 0; j < nref; j++)
                ss << setw(4) << ts[i * nref + j];
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] == 0)
                    ss << setw(6) << "";
                else
                    ss << setw(6) << jd[i][dk] + 1;
            for (int16_t dk = 0; dk < 4; dk++)
                ss << " " << setw(12) << xs[i][dk];
            ss << endl;
        }
        return ss.str();
    }
};

template <typename S>
struct MRCIDistinctRowTable<S, typename enable_if<!is_void<S>::value>::type>
    : MRCIDistinctRowTable<void> {
    vector<typename S::pg_t> orb_sym;
    vector<typename S::pg_t> pgs;
    MRCIDistinctRowTable(const vector<typename S::pg_t> &orb_sym)
        : orb_sym(orb_sym), MRCIDistinctRowTable<void>() {}
    MRCIDistinctRowTable(int16_t a, int16_t b, int16_t c, typename S::pg_t pg,
                         const vector<typename S::pg_t> &orb_sym)
        : MRCIDistinctRowTable(orb_sym) {
        abc.push_back(array<int16_t, 3>{a, b, c});
        pgs.push_back(pg);
    }
    virtual ~MRCIDistinctRowTable() = default;
    void initialize_mrci(int ci_order,
                         const vector<vector<uint8_t>> &refs) override {
        nref = (int)refs.size();
        assert(nref > 0);
        nex = ci_order;
        nvirt = 0;
        for (int i = 0; i < (int)refs[0].size(); i++)
            if (all_of(refs.begin(), refs.end(),
                       [i](const vector<uint8_t> &ref) { return ref[i] == 0; }))
                nvirt++;
            else
                break;
        ncore = 0;
        for (int i = nvirt; i < (int)refs[0].size(); i++)
            if (all_of(refs.begin(), refs.end(),
                       [i](const vector<uint8_t> &ref) { return ref[i] == 3; }))
                ncore++;
            else
                break;
        cout << "nv nex = " << nvirt << " " << nex << endl;
        int nd = n_drt();
        abc.resize(1);
        pgs.resize(1);
        ts.clear();
        ts.reserve(nd);
        for (int i = 0; i < nref; i++)
            ts.push_back(0);
        jd.clear();
        abc.reserve(nd);
        jd.reserve(nd);
        pgs.reserve(nd);
        int tt = 2 + nref;
        vector<int16_t> tmp;
        vector<typename S::pg_t> tmp_pg;
        vector<int> idx, ridx;
        auto tmp_cmp = [&tmp, &tmp_pg, &tt](int ip, int iq) {
            for (int i = 0; i < tt; i++)
                if (tmp[ip * tt + i] != tmp[iq * tt + i])
                    return tmp[ip * tt + i] > tmp[iq * tt + i];
            return (int)tmp_pg[ip] > (int)tmp_pg[iq];
        };
        // rd == 3 -> 0
        // rd == 12 d == 3 || d == 12 rd == 0 -> 1
        // rd == 0 d == 3 -> 2
        auto arc_t = [](int16_t rd, int16_t d) {
            return (((rd >> 1) ^ (rd & 1)) & (d == 3)) |
                   (((!rd) & (d >> 1)) + ((!rd) & (d & 1)));
        };
        int16_t k = abc[0][0] + abc[0][1] + abc[0][2];
        // d = 0: ; d = 1: b--; d = 2:a--,b++; d = 3:a--;
        vector<int> ref_idx(nref * 2, 0);
        for (int ir = 0, nr, it; ir < abc.size(); ir += nr, k--) {
            nr = (int)abc.size() - ir;
            it = 0;
            tmp.reserve(nr * 4 * tt);
            tmp_pg.reserve(nr * 4);
            for (int i = 0; i < nr; i++) {
                array<int16_t, 3> p = abc[i + ir];
                jd.push_back(array<int, 4>{0, 0, 0, 0});
                for (int16_t dk = 0; dk < 4; dk++) {
                    if ((p[2] == 0 && (dk == 0 || dk == 2)) ||
                        (p[1] == 0 && dk == 1) ||
                        (p[0] == 0 && (dk == 2 || dk == 3)))
                        continue;
                    tmp[it * tt] = (int16_t)(p[0] - (dk >> 1));
                    tmp[it * tt + 1] = (int16_t)(p[1] - (dk & 1) + (dk >> 1));
                    const int16_t xc =
                        (int16_t)(k - 1 - tmp[it * tt] - tmp[it * tt + 1]);
                    tmp_pg[it] =
                        dk == 0 || dk == 3
                            ? pgs[i + ir]
                            : S::pg_mul(pgs[i + ir], S::pg_inv(orb_sym[k - 1]));
                    if (k - 1 == 0 && tmp_pg[it] != 0)
                        continue;
                    if (k - 1 <= nvirt && xc < k - 1 - nex)
                        continue;
                    // at the act/core boundry, 2 spin cannot be larger than
                    // the excitation level
                    if (k - 1 == ncore + nvirt && tmp[it * tt + 1] > nex)
                        continue;
                    bool t_valid = false;
                    if (k - 1 >= nvirt && k - 1 < ncore + nvirt) {
                        const int16_t mt = ts[(i + ir) * nref];
                        const int16_t xt = mt + arc_t(refs[0][k - 1], dk);
                        t_valid = t_valid | (xt <= nex);
                        for (int j = 0; j < nref; j++)
                            tmp[it * tt + 2 + j] = xt;
                    } else {
                        for (int j = 0; j < nref; j++) {
                            if (ref_idx[j] == i + ir &&
                                ((refs[j][k - 1] ^ dk) == 3) &&
                                ((dk >> 1) ^ (dk & 1)))
                                assert(ts[(i + ir) * nref + j] == 0);
                            // here if 1/2 2/1 diff outside the occ change
                            // it should be count as one excitation
                            const int16_t xt = ts[(i + ir) * nref + j] +
                                               ((ref_idx[j] == i + ir) &
                                                ((refs[j][k - 1] ^ dk) == 3) &
                                                ((dk >> 1) ^ (dk & 1))) +
                                               arc_t(refs[j][k - 1], dk);
                            t_valid = t_valid | (xt <= nex);
                            tmp[it * tt + 2 + j] = xt;
                        }
                        if (k - 1 == ncore + nvirt) {
                            const int16_t mt = *min_element(
                                &tmp[it * tt + 2], &tmp[(it + 1) * tt]);
                            for (int j = 0; j < nref; j++)
                                tmp[it * tt + 2 + j] = mt;
                        }
                    }
                    if (t_valid) {
                        for (int j = 0; j < nref; j++)
                            if (ref_idx[j] == i + ir && refs[j][k - 1] == dk)
                                ref_idx[j + nref] = it + 1;
                        jd.back()[dk] = ++it;
                    }
                }
            }
            idx.reserve(it);
            for (int i = 0; i < it; i++)
                idx[i] = i;
            sort(idx.begin(), idx.begin() + it, tmp_cmp);
            ridx.reserve(it + 1);
            ridx[0] = 0;
            for (int i = 0; i < it; i++)
                if (i == 0 || tmp_cmp(idx[i - 1], idx[i])) {
                    int ii = idx[i] * tt;
                    abc.push_back(array<int16_t, 3>{
                        tmp[ii + 0], tmp[ii + 1],
                        (int16_t)(k - 1 - tmp[ii + 0] - tmp[ii + 1])});
                    for (int j = 0; j < nref; j++)
                        ts.push_back(tmp[ii + 2 + j]);
                    pgs.push_back(tmp_pg[idx[i]]);
                    ridx[idx[i] + 1] = (int)abc.size() - 1;
                } else
                    ridx[idx[i] + 1] = ridx[idx[i - 1] + 1];
            for (int i = 0; i < nr; i++)
                for (int16_t dk = 0; dk < 4; dk++)
                    jd[i + ir][dk] = ridx[jd[i + ir][dk]];
            for (int j = 0; j < nref; j++)
                ref_idx[j] = ridx[ref_idx[j + nref]];
        }
        int n = n_rows();
        idx.reserve(n);
        for (int i = n - 1; i >= 0; i--) {
            idx[i] = 0;
            if (abc[i][0] == 0 && abc[i][1] == 0 && abc[i][2] == 0)
                idx[i] = 1;
            else
                for (int16_t dk = 0; dk < 4; dk++)
                    if (jd[i][dk] != 0 && idx[jd[i][dk]])
                        idx[i] = 1;
        }
        int xn = 0;
        for (int i = 0; i < n; i++)
            if (idx[i]) {
                if (i != xn) {
                    abc[xn] = abc[i];
                    jd[xn] = jd[i];
                    pgs[xn] = pgs[i];
                    memcpy(&ts[xn * nref], &ts[i * nref],
                           sizeof(int16_t) * nref);
                }
                if (xn == 0 || abc[xn][0] + abc[xn][1] + abc[xn][2] > nvirt ||
                    abc[xn - 1][0] != abc[xn][0] ||
                    abc[xn - 1][1] != abc[xn][1] ||
                    abc[xn - 1][2] != abc[xn][2] || pgs[xn - 1] != pgs[xn])
                    idx[i] = xn++;
                else
                    idx[i] = xn - 1;
            }
        n = xn;
        for (int i = 0; i < n; i++)
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    jd[i][dk] = idx[jd[i][dk]];
        abc.resize(n);
        jd.resize(n);
        pgs.resize(n);
        ts.resize(n * nref);
        xs.resize(n);
        for (int i = n - 1; i >= 0; i--) {
            xs[i] = array<LL, 4>{0, 0, 0, 0};
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] != 0)
                    xs[i][dk] = xs[jd[i][dk]][3];
            for (int16_t dk = 1; dk < 4; dk++)
                xs[i][dk] += xs[i][dk - 1];
            if (abc[i][0] == 0 && abc[i][1] == 0 && abc[i][2] == 0)
                xs[i][3] = 1;
        }
    }
    string to_str() const override {
        if (jd.size() == 0)
            return PaldusTable::to_str();
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "A" << setw(4)
           << "B" << setw(4) << "C" << setw(6) << "PG";
        for (int j = 0; j < nref; j++) {
            stringstream sr;
            sr << "T" << j;
            ss << setw(4) << sr.str();
        }
        ss << setw(6) << "JD0" << setw(6) << "JD1" << setw(6) << "JD2"
           << setw(6) << "JD3"
           << " " << setw(12) << "X0"
           << " " << setw(12) << "X1"
           << " " << setw(12) << "X2"
           << " " << setw(12) << "X3" << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            k = abc[i][0] + abc[i][1] + abc[i][2];
            ss << setw(4) << (i + 1);
            if (k == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << k;
            ss << setw(4) << abc[i][0] << setw(4) << abc[i][1] << setw(4)
               << abc[i][2];
            ss << setw(6) << (int)pgs[i];
            for (int j = 0; j < nref; j++)
                ss << setw(4) << ts[i * nref + j];
            for (int16_t dk = 0; dk < 4; dk++)
                if (jd[i][dk] == 0)
                    ss << setw(6) << "";
                else
                    ss << setw(6) << jd[i][dk] + 1;
            for (int16_t dk = 0; dk < 4; dk++)
                ss << " " << setw(12) << xs[i][dk];
            ss << endl;
        }
        return ss.str();
    }
};

} // namespace block2

// #include "../core/symmetry.hpp"

// using namespace block2;

// int main() {
//     // DistinctRowTable<void> drt(3, 0, 2);
//     MRCIDistinctRowTable<SU2> drt(
//         5, 0, 9, 0,
//         vector<SU2::pg_t>{0, 0, 0, 0, 0, 2, 3, 2, 2, 2, 0, 3, 0, 0});
//     vector<vector<uint8_t>> refs;
//     refs.push_back(vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3,
//     3});
//     // refs.push_back(vector<uint8_t> {0, 0, 3, 3, 0, 1, 2});
//     // refs.push_back(vector<uint8_t> {0, 0, 3, 3, 1, 0, 2});
//     drt.initialize_mrci(2, refs);
//     // drt.initialize();
//     cout << drt.to_str() << endl;
//     return 0;
// }
