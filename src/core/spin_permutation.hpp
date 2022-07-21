
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2022 Huanchen Zhai <hczhai@caltech.edu>
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

/** Spin permutation for NPDM and GeneralMPO. */

#pragma once

#include "cg.hpp"
#include "matrix_functions.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

enum struct SpinOperator : uint8_t {
    C = 4,
    D = 0,
    CA = 4,
    CB = 5,
    DA = 0,
    DB = 1
};

inline bool operator&(SpinOperator a, SpinOperator b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline SpinOperator operator^(SpinOperator a, SpinOperator b) {
    return (SpinOperator)((uint8_t)a ^ (uint8_t)b);
}

inline ostream &operator<<(ostream &os, const SpinOperator c) {
    const static string repr[] = {"DA", "DB", "C", "D", "CA", "CB"};
    os << repr[(uint8_t)c];
    return os;
}

struct SpinPermTerm {
    double factor;
    vector<pair<SpinOperator, uint16_t>> ops;
    SpinPermTerm() : factor(0) {}
    SpinPermTerm(SpinOperator op, uint16_t index, double factor = 1.0)
        : factor(factor), ops{make_pair(op, index)} {}
    SpinPermTerm(const vector<pair<SpinOperator, uint16_t>> &ops,
                 double factor = 1.0)
        : factor(factor), ops(ops) {}
    SpinPermTerm operator-() const { return SpinPermTerm(ops, -factor); }
    bool operator<(const SpinPermTerm &other) const {
        if (ops.size() != other.ops.size())
            return ops.size() < other.ops.size();
        for (size_t i = 0; i < ops.size(); i++)
            if (ops[i].first != other.ops[i].first)
                return ops[i].first < other.ops[i].first;
            else if (ops[i].second != other.ops[i].second)
                return ops[i].second < other.ops[i].second;
        if (abs(factor - other.factor) >= 1E-12)
            return factor < other.factor;
        return false;
    }
    bool ops_equal_to(const SpinPermTerm &other) const {
        if (ops.size() != other.ops.size())
            return false;
        for (size_t i = 0; i < ops.size(); i++)
            if (ops[i].first != other.ops[i].first)
                return false;
            else if (ops[i].second != other.ops[i].second)
                return false;
        return true;
    }
    bool operator==(const SpinPermTerm &other) const {
        return ops_equal_to(other) && abs(factor - other.factor) < 1E-12;
    }
    bool operator!=(const SpinPermTerm &other) const {
        return !ops_equal_to(other) || abs(factor - other.factor) >= 1E-12;
    }
    string to_str() const {
        stringstream ss;
        if (factor != 1.0)
            ss << factor << " ";
        for (auto &op : ops)
            ss << op.first << op.second << " ";
        return ss.str();
    }
};

struct SpinPermTensor {
    vector<vector<SpinPermTerm>> data;
    SpinPermTensor() : data{vector<SpinPermTerm>()} {}
    SpinPermTensor(const vector<vector<SpinPermTerm>> &data) : data(data) {}
    static SpinPermTensor C(uint16_t index) {
        vector<SpinPermTerm> a = {SpinPermTerm(SpinOperator::CB, index)};
        vector<SpinPermTerm> b = {SpinPermTerm(SpinOperator::CA, index)};
        return SpinPermTensor(vector<vector<SpinPermTerm>>{a, b});
    }
    static SpinPermTensor D(uint16_t index) {
        vector<SpinPermTerm> a = {SpinPermTerm(SpinOperator::DA, index)};
        vector<SpinPermTerm> b = {-SpinPermTerm(SpinOperator::DB, index)};
        return SpinPermTensor(vector<vector<SpinPermTerm>>{a, b});
    }
    SpinPermTensor simplify() const {
        vector<vector<SpinPermTerm>> zd = data;
        for (auto &jz : zd) {
            sort(jz.begin(), jz.end());
            int j = 0;
            for (int i = 0; i < (int)jz.size(); i++)
                if (j == 0 || !jz[j - 1].ops_equal_to(jz[i]))
                    jz[j == 0 || abs(jz[j - 1].factor) >= 1E-12 ? j++ : j - 1] =
                        jz[i];
                else
                    jz[j - 1].factor += jz[i].factor;
            if (j != 0 && abs(jz[j - 1].factor) < 1E-12)
                j--;
            jz.resize(j);
        }
        return SpinPermTensor(zd);
    }
    // return 1 if number of even cycles is odd
    static uint8_t permutation_parity(const vector<uint16_t> &perm) {
        uint8_t n = 0;
        vector<uint8_t> tag(perm.size(), 0);
        for (uint16_t i = 0, j; i < (uint16_t)perm.size(); i++) {
            j = i, n ^= !tag[j];
            while (!tag[j])
                n ^= 1, tag[j] = 1, j = perm[j];
        }
        return n;
    }
    static pair<string, int> auto_sort_string(const vector<uint16_t> &x,
                                              const string &xops) {
        vector<uint16_t> perm, pcnt;
        string z(xops.length(), '.');
        perm.resize(x.size());
        pcnt.resize(x.size() + 1, 0);
        for (uint16_t i = 0; i < x.size(); i++)
            pcnt[x[i] + 1]++;
        for (uint16_t i = 0; i < x.size(); i++)
            pcnt[i + 1] += pcnt[i];
        for (uint16_t i = 0; i < x.size(); i++)
            z[pcnt[x[i]]] = xops[i], perm[i] = pcnt[x[i]]++;
        return make_pair(z, permutation_parity(perm) ? -1 : 1);
    }
    SpinPermTensor auto_sort() const {
        SpinPermTensor r = *this;
        vector<uint16_t> perm, pcnt;
        for (auto &jx : r.data)
            for (auto &tx : jx) {
                vector<pair<SpinOperator, uint16_t>> new_ops = tx.ops;
                perm.resize(tx.ops.size());
                pcnt.resize(tx.ops.size() + 1);
                for (uint16_t i = 0; i < tx.ops.size() + 1; i++)
                    pcnt[i] = 0;
                for (uint16_t i = 0; i < tx.ops.size(); i++)
                    pcnt[tx.ops[i].second + 1]++;
                for (uint16_t i = 0; i < tx.ops.size(); i++)
                    pcnt[i + 1] += pcnt[i];
                for (uint16_t i = 0; i < tx.ops.size(); i++)
                    new_ops[pcnt[tx.ops[i].second]] = tx.ops[i],
                    perm[i] = pcnt[tx.ops[i].second]++;
                tx.factor *= permutation_parity(perm) ? -1 : 1;
                tx.ops = new_ops;
            }
        return r;
    }
    SpinPermTensor operator*(double d) const {
        SpinPermTensor r = *this;
        for (auto &jx : r.data)
            for (auto &tx : jx)
                tx.factor *= d;
        return r;
    }
    SpinPermTensor operator+(const SpinPermTensor &other) const {
        assert(data.size() == other.data.size());
        vector<vector<SpinPermTerm>> zd = data;
        for (size_t i = 0; i < zd.size(); i++)
            zd[i].insert(zd[i].end(), other.data[i].begin(),
                         other.data[i].end());
        return SpinPermTensor(zd).simplify();
    }
    bool operator==(const SpinPermTensor &other) const {
        SpinPermTensor a = simplify(), b = other.simplify();
        if (a.data.size() != b.data.size())
            return false;
        for (int i = 0; i < (int)a.data.size(); i++) {
            if (a.data[i].size() != b.data[i].size())
                return false;
            for (int j = 0; j < a.data[i].size(); j++)
                if (a.data[i][j] != b.data[i][j])
                    return false;
        }
        return true;
    }
    double equal_to_scaled(const SpinPermTensor &other) const {
        SpinPermTensor a = simplify(), b = other.simplify();
        double fac = 0;
        if (a.data.size() != b.data.size())
            return 0;
        for (int i = 0; i < (int)a.data.size(); i++) {
            if (a.data[i].size() != b.data[i].size())
                return 0;
            for (int j = 0; j < a.data[i].size(); j++)
                if (!a.data[i][j].ops_equal_to(b.data[i][j]))
                    return 0;
                else if (fac == 0)
                    fac = a.data[i][j].factor / b.data[i][j].factor;
                else if (abs(a.data[i][j].factor / b.data[i][j].factor - fac) >=
                         1E-12)
                    return 0;
        }
        return fac;
    }
    static SpinPermTensor mul(const SpinPermTensor &x, const SpinPermTensor &y,
                              int16_t tjz, const SU2CG &cg) {
        vector<vector<SpinPermTerm>> z(tjz + 1);
        int16_t tjx = (int16_t)x.data.size() - 1,
                tjy = (int16_t)y.data.size() - 1;
        for (int16_t iz = 0, mz = -tjz; mz <= tjz; mz += 2, iz++)
            for (int16_t ix = 0, mx = -tjx; mx <= tjx; mx += 2, ix++)
                for (int16_t iy = 0, my = -tjy; my <= tjy; my += 2, iy++)
                    if (mx + my == mz) {
                        for (auto &tx : x.data[ix])
                            for (auto &ty : y.data[iy]) {
                                double factor =
                                    tx.factor * ty.factor *
                                    cg.cg(tjx, tjy, tjz, mx, my, mz);
                                vector<pair<SpinOperator, uint16_t>> ops =
                                    tx.ops;
                                ops.reserve(tx.ops.size() + ty.ops.size());
                                ops.insert(ops.end(), ty.ops.begin(),
                                           ty.ops.end());
                                z[iz].push_back(SpinPermTerm(ops, factor));
                            }
                    }
        return SpinPermTensor(z).simplify();
    }
    string to_str() const {
        stringstream ss;
        for (auto &dxx : data) {
            bool first = true;
            for (auto &dx : dxx) {
                string x = dx.to_str();
                if (x[0] != '-' && !first)
                    ss << "+ " << x;
                else
                    ss << x;
                first = false;
            }
            ss << endl;
        }
        return ss.str();
    }
};

struct SpinPermRecoupling {
    static string to_str(uint16_t x) {
        stringstream ss;
        ss << x;
        return ss.str();
    }
    static vector<uint8_t> make_cds(const string &x) {
        vector<uint8_t> r;
        for (auto &c : x)
            r.push_back(c == 'C');
        return r;
    }
    static string make_with_cds(const string &x, const vector<uint8_t> &cds) {
        int icd = 0;
        stringstream ss;
        for (auto &c : x)
            if (c == '.')
                ss << (cds[icd++] ? "C" : "D");
            else
                ss << c;
        return ss.str();
    }
    static string split_cds(const string &x, vector<uint8_t> &cds) {
        int icd = 0;
        stringstream ss;
        cds.clear();
        for (auto &c : x)
            if (c == 'C' || c == 'c')
                ss << '.', cds.push_back(1);
            else if (c == 'D' || c == 'd')
                ss << '.', cds.push_back(0);
            else
                ss << c;
        return ss.str();
    }
    static int count_cds(const string &x) {
        int ncd = 0;
        for (auto &c : x)
            if (!(c == '.' || c == '(' || c == ')' || c == '+' ||
                  (c >= '0' && c <= '9')))
                ncd++;
        return ncd;
    }
    static SpinPermTensor make_tensor(const string &x,
                                      const vector<uint16_t> &indices,
                                      const vector<uint8_t> &cds,
                                      const SU2CG &cg) {
        if (x == ".") {
            assert(indices.size() == 1 && cds.size() == 1);
            return cds[0] ? SpinPermTensor::C(indices[0])
                          : SpinPermTensor::D(indices[0]);
        } else {
            int ix = 0, dot_cnt = 0, depth = 0;
            for (auto &c : x) {
                if (c == '(')
                    depth++;
                else if (c == ')')
                    depth--;
                else if (c == '.')
                    dot_cnt++;
                else if (c == '+' && depth == 1)
                    break;
                ix++;
            }
            int twos = 0, iy = 0;
            for (int i = (int)x.length() - 1, k = 1; i >= 0; i--, k *= 10)
                if (x[i] >= '0' && x[i] <= '9')
                    twos += (x[i] - '0') * k;
                else {
                    iy = i;
                    break;
                }
            SpinPermTensor a = make_tensor(
                x.substr(1, ix - 1),
                vector<uint16_t>(indices.begin(), indices.begin() + dot_cnt),
                vector<uint8_t>(cds.begin(), cds.begin() + dot_cnt), cg);
            SpinPermTensor b = make_tensor(
                x.substr(ix + 1, iy - ix - 1),
                vector<uint16_t>(indices.begin() + dot_cnt, indices.end()),
                vector<uint8_t>(cds.begin() + dot_cnt, cds.end()), cg);
            return SpinPermTensor::mul(a, b, twos, cg);
        }
    }
    static uint16_t find_split_index(const string &x) {
        int dot_cnt = 0, depth = 0;
        for (auto &c : x)
            if (c == '(')
                depth++;
            else if (c == ')')
                depth--;
            else if (c == '.')
                dot_cnt++;
            else if (c == '+' && depth == 1)
                break;
        return dot_cnt;
    }
    // (.+(.+(.+.)0)0)0 -> 1 2 3
    // ((.+.)0+(.+.)0) -> 2 3
    static vector<uint16_t> find_split_indices_from_left(const string &x,
                                                         int start_depth = 1) {
        int dot_cnt = 0, depth = 0;
        vector<uint16_t> r;
        for (size_t ic = 0; ic < x.length(); ic++) {
            auto &c = x[ic];
            if (c == '(')
                depth++;
            else if (c == ')')
                depth--;
            else if (c == '.')
                dot_cnt++;
            else if (c == '+' && depth == start_depth) {
                r.push_back(dot_cnt);
                start_depth++;
            }
        }
        return r;
    }
    static vector<string> initialize(uint16_t n, uint16_t twos) {
        map<pair<uint16_t, uint16_t>, vector<string>> mp;
        mp[make_pair(0, 0)] = vector<string>{""};
        mp[make_pair(1, 1)] = vector<string>{"."};
        for (int k = 2; k <= n; k++)
            for (int j = k % 2; j <= k; j += 2) {
                mp[make_pair(k, j)] = vector<string>();
                vector<string> &mpz = mp.at(make_pair(k, j));
                for (int p = 1; p < k; p++)
                    for (int jl = p % 2; jl <= p; jl += 2)
                        for (int jr = abs(j - jl); jr <= j + jl; jr += 2)
                            if (mp.count(make_pair(p, jl)) &&
                                mp.count(make_pair(k - p, jr)))
                                for (auto &xl : mp.at(make_pair(p, jl)))
                                    for (auto &xr : mp.at(make_pair(k - p, jr)))
                                        mpz.push_back("(" + xl + "+" + xr +
                                                      ")" + to_str(j));
            }
        return mp[make_pair(n, twos)];
    }
};

struct SpinPermPattern {
    uint16_t n;
    vector<uint16_t> data;
    SpinPermPattern(uint16_t n) : n(n), data(initialize(n)) {}
    static vector<uint16_t> all_reordering(const vector<uint16_t> &x) {
        vector<pair<uint16_t, uint16_t>> pp;
        for (auto &ix : x)
            if (pp.size() == 0 || ix != pp.back().first)
                pp.push_back(make_pair(ix, 1));
            else
                pp.back().second++;
        uint16_t maxx = pp.back().first + 1;
        vector<uint16_t> ha(x.size(), maxx);
        for (uint16_t i = 1; i <= pp.back().second; i++) {
            vector<uint16_t> hb;
            for (int j = 0, k; j < (int)ha.size(); j += x.size()) {
                for (k = (int)x.size() - 1; k >= 0; k--)
                    if (ha[j + k] != maxx)
                        break;
                    else {
                        hb.insert(hb.end(), ha.begin() + j,
                                  ha.begin() + (j + x.size()));
                        hb[hb.size() - x.size() + k] = pp.back().first;
                    }
            }
            ha = hb;
        }
        if (pp.size() == 1)
            return ha;
        vector<uint16_t> g = all_reordering(
            vector<uint16_t>(x.begin(), x.end() - pp.back().second));
        vector<uint16_t> r(ha.size() * g.size() /
                           (x.size() - pp.back().second));
        for (int h = 0, ir = 0; h < (int)ha.size(); h += x.size())
            for (int j = 0; j < (int)g.size();
                 j += x.size() - pp.back().second) {
                memcpy(r.data() + ir, ha.data() + h,
                       x.size() * sizeof(uint16_t));
                for (int k = 0, kk = 0; k < (int)x.size(); k++)
                    if (r[ir + k] == maxx)
                        r[ir + k] = g[j + kk++];
                ir += x.size();
            }
        return r;
    }
    static vector<uint16_t> initialize(uint16_t n) {
        map<pair<uint16_t, uint16_t>, vector<vector<uint16_t>>> mp;
        for (uint16_t i = 0; i <= n; i++)
            mp[make_pair(0, i)] = vector<vector<uint16_t>>();
        mp.at(make_pair(0, 0)).push_back(vector<uint16_t>());
        // i = distinct numbers, j = length
        for (uint16_t i = 1; i <= n; i++) {
            for (uint16_t j = i; j <= n; j++) {
                mp[make_pair(i, j)] = vector<vector<uint16_t>>();
                vector<vector<uint16_t>> &mpv = mp.at(make_pair(i, j));
                for (uint16_t k = 1; k <= j - i + 1; k++) {
                    vector<vector<uint16_t>> &mpx =
                        mp.at(make_pair(i - 1, j - k));
                    for (auto &x : mpx) {
                        mpv.push_back(x);
                        mpv.back().insert(mpv.back().end(), k, i - 1);
                    }
                }
            }
        }
        size_t cnt = 0;
        for (uint16_t i = 1; i <= n; i++)
            cnt += mp.at(make_pair(i, n)).size();
        vector<uint16_t> r(cnt * (n + 2));
        size_t ic = 0;
        for (uint16_t i = 1; i <= n; i++) {
            vector<vector<uint16_t>> &mpx = mp.at(make_pair(i, n));
            for (auto &x : mpx) {
                r[ic++] = i;
                memcpy(r.data() + ic, x.data(), sizeof(uint16_t) * n);
                r[ic + n] = n;
                for (int16_t k = (n - 1) >> 1; k >= 0; k--) {
                    if (r[ic + k] != r[ic + k + 1]) {
                        r[ic + n] = k;
                        break;
                    }
                    if (r[ic + n - 1 - k] != r[ic + n - 1 - k + 1]) {
                        r[ic + n] = n - 1 - k;
                        break;
                    }
                }
                ic += n + 1;
            }
        }
        return r;
    }
    size_t count() const { return data.size() / (n + 2); }
    vector<uint16_t> operator[](size_t i) {
        return vector<uint16_t>(data.begin() + i * (n + 2) + 1,
                                data.begin() + i * (n + 2) + n + 1);
    }
    string to_str() const {
        stringstream ss;
        size_t cnt = data.size() / (n + 2);
        ss << "N = " << n << " COUNT = " << data.size() / (n + 2) << endl;
        for (size_t ic = 0; ic < cnt; ic++) {
            for (uint16_t j = 0; j < n + 1; j++)
                ss << data[ic * (n + 2) + j] << (j == 0 ? " : " : " ");
            ss << "| " << data[ic * (n + 2) + n + 1] << endl;
        }
        return ss.str();
    }
    static vector<string> get_unique(const vector<uint8_t> &cds,
                                     const vector<uint16_t> &ref_indices,
                                     int split_idx = -1,
                                     bool ref_split = false) {
        SU2CG cg(100);
        cg.initialize();
        int nn = cds.size();
        vector<uint16_t> indices = ref_indices;
        if (indices.size() == 0)
            for (int i = 0; i < nn; i++)
                indices.push_back(i);
        vector<uint16_t> ref_split_idx;
        for (int i = 1; i < nn; i++)
            if (indices[i] != indices[i - 1])
                ref_split_idx.push_back(i);
        vector<string> pp = SpinPermRecoupling::initialize(nn, 0);
        vector<SpinPermTensor> ts(pp.size());
        for (int i = 0; i < (int)pp.size(); i++) {
            if (split_idx != -1 &&
                SpinPermRecoupling::find_split_index(pp[i]) != split_idx)
                continue;
            if (ref_split) {
                vector<uint16_t> act_split_idx =
                    SpinPermRecoupling::find_split_indices_from_left(pp[i]);
                if (act_split_idx.size() < ref_split_idx.size() ||
                    !equal(ref_split_idx.begin(), ref_split_idx.end(),
                           act_split_idx.begin()))
                    continue;
            }
            ts[i] =
                SpinPermRecoupling::make_tensor(pp[i], ref_indices, cds, cg);
            assert(ts[i].data.size() != 0);
        }
        vector<int> selected_pp_idx;
        for (int i = 0; i < (int)pp.size(); i++) {
            if (ts[i].data.size() == 1 && ts[i].data[0].size() == 0)
                continue;
            bool found = false;
            for (auto j : selected_pp_idx) {
                double x = ts[i].equal_to_scaled(ts[j]);
                if (x != 0)
                    found = true;
            }
            if (!found)
                selected_pp_idx.push_back(i);
        }
        vector<string> r;
        for (auto &ip : selected_pp_idx)
            r.push_back(pp[ip]);
        return r;
    }
    static vector<vector<double>> make_matrix(const vector<SpinPermTensor> &x,
                                              const SpinPermTensor &std) {
        vector<vector<double>> r(x.size());
        int ixx = 0;
        for (int ixx = 0; ixx < x.size(); ixx++) {
            auto &pg = r[ixx];
            pg = vector<double>(std.data[0].size(), 0);
            for (auto &t : x[ixx].data[0])
                for (int i = 0; i < pg.size(); i++)
                    if (std.data[0][i].ops_equal_to(t)) {
                        pg[i] += t.factor;
                        break;
                    }
        }
        return r;
    }
};

// generate appropriate spin recoupling formulae after reordering
struct SpinPermScheme {
    vector<vector<uint16_t>> index_patterns;
    vector<map<vector<uint16_t>, vector<pair<double, string>>>> data;
    SpinPermScheme() {}
    SpinPermScheme(string spin_str, bool su2 = true) {
        int nn = SpinPermRecoupling::count_cds(spin_str);
        SpinPermScheme r = su2 ? SpinPermScheme::initialize_su2(nn, spin_str)
                               : SpinPermScheme::initialize_sz(nn, spin_str);
        index_patterns = r.index_patterns;
        data = r.data;
    }
    static SpinPermScheme initialize_sz(int nn, string spin_str) {
        using T = SpinPermTensor;
        using R = SpinPermRecoupling;
        SpinPermPattern spat(nn);
        vector<double> mptr;
        SpinPermScheme r;
        r.index_patterns.resize(spat.count());
        r.data.resize(spat.count());
        for (size_t i = 0; i < spat.count(); i++) {
            vector<uint16_t> irr = spat[i];
            r.index_patterns[i] = irr;
            vector<uint16_t> rr = SpinPermPattern::all_reordering(irr);
            for (int j = 0; j < rr.size(); j += irr.size()) {
                vector<uint16_t> indices(rr.begin() + j,
                                         rr.begin() + j + irr.size());
                r.data[i][indices] = vector<pair<double, string>>();
                vector<pair<double, string>> &rec_formula =
                    r.data[i].at(indices);
                auto pis = SpinPermTensor::auto_sort_string(indices, spin_str);
                rec_formula.push_back(make_pair((double)pis.second, pis.first));
            }
        }
        return r;
    }
    static SpinPermScheme initialize_su2(int nn, string spin_str) {
        using T = SpinPermTensor;
        using R = SpinPermRecoupling;
        SU2CG cg(100);
        cg.initialize();
        vector<uint8_t> cds;
        spin_str = SpinPermRecoupling::split_cds(spin_str, cds);
        SpinPermPattern spat(nn);
        vector<double> mptr;
        SpinPermScheme r;
        r.index_patterns.resize(spat.count());
        r.data.resize(spat.count());
        for (size_t i = 0; i < spat.count(); i++) {
            vector<uint16_t> irr = spat[i];
            r.index_patterns[i] = irr;
            vector<uint16_t> rr = SpinPermPattern::all_reordering(irr);
            for (int j = 0; j < rr.size(); j += irr.size()) {
                vector<uint16_t> indices(rr.begin() + j,
                                         rr.begin() + j + irr.size());
                r.data[i][indices] = vector<pair<double, string>>();
                vector<pair<double, string>> &rec_formula =
                    r.data[i].at(indices);
                T x = R::make_tensor(spin_str, indices, cds, cg) * 2;
                T xs = x.auto_sort();
                vector<uint8_t> target_cds = cds;
                for (int j = 0; j < irr.size(); j++)
                    target_cds[j] =
                        xs.data[0][0].ops[j].first & SpinOperator::C;
                vector<string> ttp =
                    SpinPermPattern::get_unique(target_cds, irr, -1, true);
                vector<T> tts(ttp.size());
                bool found = false;
                for (int j = 0; j < tts.size() && !found; j++) {
                    tts[j] = R::make_tensor(ttp[j], irr, target_cds, cg) * 2;
                    double x = xs.equal_to_scaled(tts[j]);
                    if (x != 0) {
                        found = true;
                        rec_formula.push_back(
                            make_pair(x, R::make_with_cds(ttp[j], target_cds)));
                    }
                }
                if (found)
                    continue;
                SpinPermTensor std = SpinPermTensor();
                int cxx = 117;
                for (int j = 0; j < tts.size(); j++)
                    std = std + tts[j] * (cxx++);
                std = std + xs * (cxx++);
                vector<double> pgv(std.data[0].size(), 0);
                for (auto &t : xs.data[0])
                    for (int i = 0; i < pgv.size(); i++)
                        if (std.data[0][i].ops_equal_to(t)) {
                            pgv[i] += t.factor;
                            break;
                        }
                auto pgg = SpinPermPattern::make_matrix(tts, std);
                mptr.resize(pgg[0].size() * tts.size());
                for (int ja = 0; ja < (int)pgg.size() && !found; ja++)
                    for (int jb = ja + 1; jb < (int)pgg.size() && !found;
                         jb++) {
                        MatrixRef a(mptr.data(), pgg[0].size(), 2);
                        MatrixRef x(mptr.data() + a.size(), 2, 1);
                        MatrixRef b(mptr.data() + a.size() + x.size(),
                                    pgg[0].size(), 1);
                        for (int k = 0; k < (int)pgg[0].size(); k++)
                            b(k, 0) = pgv[k], a(k, 0) = pgg[ja][k],
                                 a(k, 1) = pgg[jb][k];
                        double c = MatrixFunctions::least_squares(a, b, x);
                        if (abs(c) > 1E-12)
                            continue;
                        found = true;
                        rec_formula.push_back(make_pair(
                            x.data[0], R::make_with_cds(ttp[ja], target_cds)));
                        rec_formula.push_back(make_pair(
                            x.data[1], R::make_with_cds(ttp[jb], target_cds)));
                    }
                for (int ja = 0; ja < (int)pgg.size() && !found; ja++)
                    for (int jb = ja + 1; jb < (int)pgg.size() && !found; jb++)
                        for (int jc = jb + 1; jc < (int)pgg.size() && !found;
                             jc++) {
                            MatrixRef a(mptr.data(), pgg[0].size(), 3);
                            MatrixRef x(mptr.data() + a.size(), 3, 1);
                            MatrixRef b(mptr.data() + a.size() + x.size(),
                                        pgg[0].size(), 1);
                            for (int k = 0; k < (int)pgg[0].size(); k++)
                                b(k, 0) = pgv[k], a(k, 0) = pgg[ja][k],
                                     a(k, 1) = pgg[jb][k], a(k, 2) = pgg[jc][k];
                            double c = MatrixFunctions::least_squares(a, b, x);
                            if (abs(c) > 1E-12)
                                continue;
                            found = true;
                            rec_formula.push_back(make_pair(
                                x.data[0],
                                R::make_with_cds(ttp[ja], target_cds)));
                            rec_formula.push_back(make_pair(
                                x.data[1],
                                R::make_with_cds(ttp[jb], target_cds)));
                            rec_formula.push_back(make_pair(
                                x.data[2],
                                R::make_with_cds(ttp[jc], target_cds)));
                        }
                for (int ja = 0; ja < (int)pgg.size() && !found; ja++)
                    for (int jb = ja + 1; jb < (int)pgg.size() && !found; jb++)
                        for (int jc = jb + 1; jc < (int)pgg.size() && !found;
                             jc++)
                            for (int jd = jc + 1;
                                 jd < (int)pgg.size() && !found; jd++) {
                                MatrixRef a(mptr.data(), pgg[0].size(), 4);
                                MatrixRef x(mptr.data() + a.size(), 4, 1);
                                MatrixRef b(mptr.data() + a.size() + x.size(),
                                            pgg[0].size(), 1);
                                for (int k = 0; k < (int)pgg[0].size(); k++)
                                    b(k, 0) = pgv[k], a(k, 0) = pgg[ja][k],
                                         a(k, 1) = pgg[jb][k],
                                         a(k, 2) = pgg[jc][k],
                                         a(k, 3) = pgg[jd][k];
                                double c =
                                    MatrixFunctions::least_squares(a, b, x);
                                if (abs(c) > 1E-12)
                                    continue;
                                found = true;
                                rec_formula.push_back(make_pair(
                                    x.data[0],
                                    R::make_with_cds(ttp[ja], target_cds)));
                                rec_formula.push_back(make_pair(
                                    x.data[1],
                                    R::make_with_cds(ttp[jb], target_cds)));
                                rec_formula.push_back(make_pair(
                                    x.data[2],
                                    R::make_with_cds(ttp[jc], target_cds)));
                                rec_formula.push_back(make_pair(
                                    x.data[3],
                                    R::make_with_cds(ttp[jd], target_cds)));
                            }
                for (int ja = 0; ja < (int)pgg.size() && !found; ja++)
                    for (int jb = ja + 1; jb < (int)pgg.size() && !found; jb++)
                        for (int jc = jb + 1; jc < (int)pgg.size() && !found;
                             jc++)
                            for (int jd = jc + 1;
                                 jd < (int)pgg.size() && !found; jd++)
                                for (int je = jd + 1;
                                     je < (int)pgg.size() && !found; je++) {
                                    MatrixRef a(mptr.data(), pgg[0].size(), 5);
                                    MatrixRef x(mptr.data() + a.size(), 5, 1);
                                    MatrixRef b(mptr.data() + a.size() +
                                                    x.size(),
                                                pgg[0].size(), 1);
                                    for (int k = 0; k < (int)pgg[0].size(); k++)
                                        b(k, 0) = pgv[k], a(k, 0) = pgg[ja][k],
                                             a(k, 1) = pgg[jb][k],
                                             a(k, 2) = pgg[jc][k],
                                             a(k, 3) = pgg[jd][k],
                                             a(k, 4) = pgg[je][k];
                                    double c =
                                        MatrixFunctions::least_squares(a, b, x);
                                    if (abs(c) > 1E-12)
                                        continue;
                                    found = true;
                                    rec_formula.push_back(make_pair(
                                        x.data[0],
                                        R::make_with_cds(ttp[ja], target_cds)));
                                    rec_formula.push_back(make_pair(
                                        x.data[1],
                                        R::make_with_cds(ttp[jb], target_cds)));
                                    rec_formula.push_back(make_pair(
                                        x.data[2],
                                        R::make_with_cds(ttp[jc], target_cds)));
                                    rec_formula.push_back(make_pair(
                                        x.data[3],
                                        R::make_with_cds(ttp[jd], target_cds)));
                                    rec_formula.push_back(make_pair(
                                        x.data[4],
                                        R::make_with_cds(ttp[je], target_cds)));
                                }
                assert(found);
            }
        }
        return r;
    }
    string to_str() const {
        stringstream ss;
        int cnt = index_patterns.size();
        int nn = cnt == 0 ? 0 : index_patterns[0].size();
        ss << "N = " << nn << " COUNT = " << cnt << endl;
        for (size_t ic = 0; ic < cnt; ic++) {
            for (uint16_t j = 0; j < nn; j++)
                ss << index_patterns[ic][j] << " ";
            ss << " :: " << endl;
            for (auto &r : data[ic]) {
                for (uint16_t j = 0; j < nn; j++)
                    ss << r.first[j] << " ";
                ss << " = ";
                for (auto &g : r.second) {
                    ss << setw(10) << setprecision(6) << fixed << g.first
                       << " * [ " << g.second << " ] ";
                }
                ss << endl;
            }
        }
        return ss.str();
    }
};

} // namespace block2

// using namespace block2;

// int main() {
//     using T = SpinPermTensor;
//     using R = SpinPermRecoupling;
//     SpinPermScheme x(6);
//     cout << x.to_str() << endl;
//     abort();
//     SpinPermPattern spat(4);
//     cout << spat.to_str() << endl;
//     for (size_t i = 0; i < spat.count(); i++) {
//         vector<uint16_t> irr = spat[i];
//         vector<uint16_t> rr = SpinPermPattern::all_reordering(irr);
//         for (int j = 0; j < rr.size(); j += irr.size()) {
//             for (int k = 0; k < irr.size(); k++)
//                 cout << setw(4) << rr[j + k];
//             cout << endl;
//         }
//     }
//     abort();
//     SU2CG cg(100);
//     cg.initialize();
//     uint16_t p = 0, q = 1, r = 2, s = 3;
//     // a = mul(mul('Cp', 'Cq', 2), mul('Dr', 'Ds', 2), 0)
//     // b = mul(mul(mul('Cp', 'Cq', 2), 'Dr', 1), 'Ds', 0)
//     auto a = T::mul(T::mul(T::C(p), T::C(q), 2, cg),
//                     T::mul(T::D(r), T::D(s), 2, cg), 0, cg);
//     auto b = T::mul(T::mul(T::mul(T::C(p), T::C(q), 2, cg), T::D(r), 1, cg),
//                     T::D(s), 0, cg);
//     cout << a.to_str() << endl;
//     cout << b.to_str() << endl;
//     cout << (a == b) << endl;
//     int nn = 4;
//     vector<string> pp = SpinPermRecoupling::initialize(nn, 0);
//     cout << pp.size() << endl;
//     for (auto &xp : pp)
//         cout << xp << " | " << SpinPermRecoupling::find_split_index(xp) <<
//         endl;
//     vector<uint8_t> cds;
//     vector<uint16_t> indices;
//     for (int i = 0; i < nn; i++)
//         indices.push_back(i), cds.push_back(i < nn / 2);
//     // vector<T> ts(pp.size());
//     // for (int i = 0; i < (int)pp.size(); i++)
//     //     ts[i] = SpinPermRecoupling::make_tensor(pp[i], indices, cds, cg);
//     // vector<int> selected_pp_idx;
//     // for (int i = 0; i < (int)pp.size(); i++) {
//     //     cout << i << " / " << pp.size() << " " << ts[i].to_str() << endl;
//     //     bool found = false;
//     //     for (auto j : selected_pp_idx) {
//     //         double x = ts[i].equal_to_scaled(ts[j]);
//     //         if (x != 0)
//     //             found = true;
//     //         // cout << "[" << i << "] = " << x << " * [" << j << "]" <<
//     endl;
//     //     }
//     //     if (!found)
//     //         selected_pp_idx.push_back(i);
//     // }
//     // cout << " selected count = " << selected_pp_idx.size() << endl;
//     // cout << " selected = ";
//     // for (auto &ix : selected_pp_idx)
//     //     cout << ix << " ";
//     // cout << endl;
//     // SpinPermTensor std;
//     // int cxx = 117;
//     // for (auto &ix : selected_pp_idx) {
//     //     std = std + ts[ix] * (cxx++);
//     // }
//     // cout << std.data[0].size() << endl;
//     // vector<vector<double>> pgg(selected_pp_idx.size());
//     // int ixx = 0;
//     // for (auto &ix : selected_pp_idx) {
//     //     auto &pg = pgg[ixx++];
//     //     pg = vector<double>(std.data[0].size(), 0);
//     //     for (auto &t : ts[ix].data[0])
//     //         for (int i = 0; i < pg.size(); i++)
//     //             if (std.data[0][i].ops_equal_to(t)) {
//     //                 pg[i] += t.factor;
//     //                 break;
//     //             }
//     //     // for (auto &px : pg)
//     //     //     cout << setw(10) << setprecision(6) << fixed << px;
//     //     // cout << endl;
//     // }
//     vector<double> ppp(100000000);
//     // for (int i = 0; i < (int)pgg.size(); i++)
//     //     for (int ja = 0; ja < (int)pgg.size(); ja++)
//     //         for (int jb = ja + 1; jb < (int)pgg.size(); jb++) {
//     //             if (ja == i || jb == i)
//     //                 continue;
//     //             MatrixRef a(ppp.data(), pgg[0].size(), 2);
//     //             MatrixRef x(ppp.data() + a.size(), 2, 1);
//     //             MatrixRef b(ppp.data() + a.size() + x.size(),
//     pgg[0].size(),
//     //             1); for (int k = 0; k < (int)pgg[0].size(); k++)
//     //                 b(k, 0) = pgg[i][k], a(k, 0) = pgg[ja][k],
//     //                      a(k, 1) = pgg[jb][k];
//     //             double c = MatrixFunctions::least_squares(a, b, x);
//     //             if (abs(c) > 1E-12)
//     //                 continue;
//     //             cout << "[" << i << "] = ";
//     //             cout << setw(10) << setprecision(6) << x.data[0] << " * ["
//     <<
//     //             ja
//     //                  << "] ";
//     //             cout << setw(10) << setprecision(6) << x.data[1] << " * ["
//     <<
//     //             jb
//     //                  << "] ";
//     //             cout << endl;
//     //         }
//     vector<uint16_t> gg = indices;
//     vector<uint16_t> rr = SpinPermPattern::all_reordering(gg);
//     cout << "reorder count = " << rr.size() / gg.size() << endl;
//     for (int i = 0; i < rr.size(); i += gg.size()) {
//         for (int k = 0; k < gg.size(); k++)
//             cout << setw(4) << rr[i + k];
//         // cout << endl;
//         vector<uint16_t> indices(rr.begin() + i, rr.begin() + i + gg.size());
//         string xpre = "(.+.)0";
//         for (int inn = 4; inn <= nn; inn += 2)
//             xpre = "((.+" + xpre + ")1+.)0";
//         SpinPermTensor x =
//             SpinPermRecoupling::make_tensor(xpre, indices, cds, cg) * 2;
//         // cout << x.to_str() << endl;
//         SpinPermTensor xs = x.auto_sort();
//         // cout << xs.to_str() << endl;
//         vector<uint8_t> target_cds = cds;
//         for (int j = 0; j < gg.size(); j++)
//             target_cds[j] = xs.data[0][0].ops[j].first & SpinOperator::C;
//         // cout << "target - cds = ";
//         // for (int j = 0; j < gg.size(); j++)
//         //     cout << (target_cds[j] ? "C" : "D");
//         // cout << endl;
//         vector<string> ttp =
//             SpinPermPattern::get_unique(target_cds, vector<uint16_t>(), nn /
//             2);
//         vector<T> tts(ttp.size());
//         bool found = false;
//         for (int j = 0; j < tts.size(); j++) {
//             tts[j] = R::make_tensor(ttp[j], gg, target_cds, cg) * 2;
//             double x = xs.equal_to_scaled(tts[j]);
//             if (x != 0)
//                 found = true, cout << " = " << setw(10) << setprecision(6)
//                                    << fixed << x << " * [" << j << "]" <<
//                                    endl;
//         }
//         if (found)
//             continue;
//         SpinPermTensor std = SpinPermTensor();
//         int cxx = 117;
//         for (int j = 0; j < tts.size(); j++)
//             std = std + tts[j] * (cxx++);
//         std = std + xs * (cxx++);
//         // cout << std.data[0].size() << endl;
//         vector<double> pgv(std.data[0].size(), 0);
//         for (auto &t : xs.data[0])
//             for (int i = 0; i < pgv.size(); i++)
//                 if (std.data[0][i].ops_equal_to(t)) {
//                     pgv[i] += t.factor;
//                     break;
//                 }
//         auto pgg = SpinPermPattern::make_matrix(tts, std);
//         for (int ja = 0; ja < (int)pgg.size() && !found; ja++)
//             for (int jb = ja + 1; jb < (int)pgg.size() && !found; jb++) {
//                 MatrixRef a(ppp.data(), pgg[0].size(), 2);
//                 MatrixRef x(ppp.data() + a.size(), 2, 1);
//                 MatrixRef b(ppp.data() + a.size() + x.size(), pgg[0].size(),
//                 1); for (int k = 0; k < (int)pgg[0].size(); k++)
//                     b(k, 0) = pgv[k], a(k, 0) = pgg[ja][k],
//                          a(k, 1) = pgg[jb][k];
//                 double c = MatrixFunctions::least_squares(a, b, x);
//                 if (abs(c) > 1E-12)
//                     continue;
//                 cout << " = ";
//                 cout << setw(10) << setprecision(6) << fixed << x.data[0]
//                      << " * [" << ja << "] ";
//                 cout << setw(10) << setprecision(6) << fixed << x.data[1]
//                      << " * [" << jb << "] ";
//                 cout << endl;
//                 found = true;
//             }
//         for (int ja = 0; ja < (int)pgg.size() && !found; ja++)
//             for (int jb = ja + 1; jb < (int)pgg.size() && !found; jb++)
//                 for (int jc = jb + 1; jc < (int)pgg.size() && !found; jc++) {
//                     MatrixRef a(ppp.data(), pgg[0].size(), 3);
//                     MatrixRef x(ppp.data() + a.size(), 3, 1);
//                     MatrixRef b(ppp.data() + a.size() + x.size(),
//                     pgg[0].size(),
//                                 1);
//                     for (int k = 0; k < (int)pgg[0].size(); k++)
//                         b(k, 0) = pgv[k], a(k, 0) = pgg[ja][k],
//                              a(k, 1) = pgg[jb][k], a(k, 2) = pgg[jc][k];
//                     double c = MatrixFunctions::least_squares(a, b, x);
//                     if (abs(c) > 1E-12)
//                         continue;
//                     cout << " = ";
//                     cout << setw(10) << setprecision(6) << fixed << x.data[0]
//                          << " * [" << ja << "] ";
//                     cout << setw(10) << setprecision(6) << fixed << x.data[1]
//                          << " * [" << jb << "] ";
//                     cout << setw(10) << setprecision(6) << fixed << x.data[2]
//                          << " * [" << jc << "] ";
//                     cout << endl;
//                     found = true;
//                 }
//         assert(found);
//     }
//     abort();
//     // cout << "---------" << endl;
//     // SpinPermTensor zstd =
//     //     R::make_tensor("((.+(.+.)0)1+.)0", indices, cds, cg) * 2;
//     // cout << zstd.to_str() << endl;
//     // vector<SpinPermTensor> A = {
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, q, r, s},
//     //                    R::make_cds("CCDD"), cg),
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, q, s, r},
//     //                    R::make_cds("CCDD"), cg),
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, r, q, s},
//     //                    R::make_cds("CDCD"), cg),
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, r, s, q},
//     //                    R::make_cds("CDDC"), cg),
//     // };
//     // vector<SpinPermTensor> B = {
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, q, r, s},
//     //                    R::make_cds("CCDD"), cg),
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, q, s, r},
//     //                    R::make_cds("CCDD"), cg) *
//     //         -1,
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, r, q, s},
//     //                    R::make_cds("CDCD"), cg),
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, r, s, q},
//     //                    R::make_cds("CDDC"), cg) *
//     //         -1,
//     // };
//     // for (int i = 0; i < A.size(); i++) {
//     //     SpinPermTensor zz = A[i] * (-1) + B[i] * sqrt(3);
//     //     zz = zz.auto_sort();
//     //     cout << (zz == zstd) << " === " << zz.to_str() << endl;
//     // }
//     // SpinPermTensor ZA =
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{0, 1, 2, 3},
//     //                    R::make_cds("CDDC"), cg);
//     // SpinPermTensor ZB =
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{0, 1, 2, 3},
//     //                    R::make_cds("CDDC"), cg) *
//     //     -1;
//     // SpinPermTensor ZC =
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{0, 1, 2, 3},
//     //                    R::make_cds("CCDD"), cg) *
//     //     -1;
//     // cout << "ZA = " << ZA.to_str() << endl;
//     // cout << "ZB = " << ZB.to_str() << endl;
//     // cout << "ZC = " << ZC.to_str() << endl;
//     // SpinPermTensor ZAB = ZA * (-1) + ZB * sqrt(3);
//     // cout << "ZAB = " << ZAB.to_str() << endl;
//     return 0;
// }
