
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
#include "threading.hpp"
#include <algorithm>
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
#include <vector>

using namespace std;

namespace block2 {

enum struct SpinOperator : uint8_t {
    C = 4,
    D = 0,
    CA = 4,
    CB = 5,
    DA = 0,
    DB = 1,
    S = 2,
    SP = 6,
    SZ = 3,
    SM = 7,
};

inline bool operator&(SpinOperator a, SpinOperator b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline SpinOperator operator^(SpinOperator a, SpinOperator b) {
    return (SpinOperator)((uint8_t)a ^ (uint8_t)b);
}

inline ostream &operator<<(ostream &os, const SpinOperator c) {
    const static string repr[] = {"DA", "DB", "S",  "SZ",
                                  "CA", "CB", "SP", "SM"};
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
    SpinPermTerm operator*(double d) const {
        return SpinPermTerm(ops, d * factor);
    }
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
    // outer vector is projected spin components
    // inner vector is sum of terms
    vector<vector<SpinPermTerm>> data;
    SpinPermTensor() : data{vector<SpinPermTerm>()} {}
    SpinPermTensor(const vector<vector<SpinPermTerm>> &data) : data(data) {}
    static SpinPermTensor I() {
        return SpinPermTensor(vector<vector<SpinPermTerm>>{vector<SpinPermTerm>{
            SpinPermTerm(vector<pair<SpinOperator, uint16_t>>())}});
    }
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
    static SpinPermTensor T(uint16_t index) {
        vector<SpinPermTerm> sp = {-SpinPermTerm(SpinOperator::SP, index)};
        vector<SpinPermTerm> sz = {SpinPermTerm(SpinOperator::SZ, index) *
                                   sqrt(2)};
        vector<SpinPermTerm> sm = {SpinPermTerm(SpinOperator::SM, index)};
        return SpinPermTensor(vector<vector<SpinPermTerm>>{sp, sz, sm});
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
    // old -> new
    static vector<uint16_t> find_pattern_perm(const vector<uint16_t> &x) {
        vector<uint16_t> perm(x.size()), pcnt(x.size() + 1, 0);
        for (uint16_t i = 0; i < x.size(); i++)
            pcnt[x[i] + 1]++;
        for (uint16_t i = 0; i < x.size(); i++)
            pcnt[i + 1] += pcnt[i];
        for (uint16_t i = 0; i < x.size(); i++)
            perm[i] = pcnt[x[i]]++;
        return perm;
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
    static vector<double> dot_product(const SpinPermTensor &a,
                                      const SpinPermTensor &b) {
        assert(a.data.size() == b.data.size());
        vector<vector<SpinPermTerm>> ad = a.data;
        vector<vector<SpinPermTerm>> bd = b.data;
        vector<double> r(a.data.size(), 0);
        for (int iz = 0; iz < (int)a.data.size(); iz++) {
            sort(ad[iz].begin(), ad[iz].end());
            sort(bd[iz].begin(), bd[iz].end());
            int ia = 0, ib = 0, na = (int)ad[iz].size(),
                nb = (int)bd[iz].size();
            while (ia < na && ib < nb) {
                if (ad[iz][ia].ops_equal_to(bd[iz][ib]))
                    r[iz] += ad[iz][ia].factor * bd[iz][ib].factor, ia++, ib++;
                else if (ad[iz][ia] < bd[iz][ib])
                    ia++;
                else
                    ib++;
            }
        }
        return r;
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
                if (tx.ops.size() != 0 && !(tx.ops[0].first & SpinOperator::S))
                    tx.factor *= permutation_parity(perm) ? -1 : 1;
                tx.ops = new_ops;
            }
        return r;
    }
    vector<uint8_t> get_cds() const {
        if (data.size() == 0 || data[0].size() == 0)
            return vector<uint8_t>();
        vector<uint8_t> r(data[0][0].ops.size(), 0);
        for (int j = 0; j < (int)r.size(); j++)
            r[j] =
                ((uint8_t)data[0][0].ops[j].first & (uint8_t)SpinOperator::S) |
                (data[0][0].ops[j].first & SpinOperator::C);
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
        if (data.size() == 1 && data[0].size() == 0)
            return other;
        else if (other.data.size() == 1 && other.data[0].size() == 0)
            return *this;
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
            r.push_back(c == 'T' ? 2 : (c == 'C'));
        return r;
    }
    static string make_with_cds(const string &x, const vector<uint8_t> &cds) {
        int icd = 0;
        stringstream ss;
        for (auto &c : x)
            if (c == '.')
                ss << (cds[icd] == 2 ? "T" : (cds[icd] ? "C" : "D")), icd++;
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
            else if (c == 'T')
                ss << '.', cds.push_back(2);
            else
                ss << c;
        return ss.str();
    }
    static int get_target_twos(const string &xstr) {
        vector<pair<int, char>> pex;
        pex.reserve(xstr.length());
        for (char x : xstr)
            if (x >= '0' && x <= '9') {
                if (pex.back().first != -1)
                    pex.back().first = pex.back().first * 10 + (int)(x - '0');
                else
                    pex.push_back(make_pair((int)(x - '0'), ' '));
            } else {
                if (x == '+' && pex.back().second == ' ')
                    pex.back().second = '*';
                pex.push_back(make_pair(-1, x));
            }
        if (pex.size() == 0)
            return 0;
        else if (pex.back().second != ' ')
            return 1;
        else
            return pex.back().first;
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
            return cds[0] == 2 ? SpinPermTensor::T(indices[0])
                               : (cds[0] ? SpinPermTensor::C(indices[0])
                                         : SpinPermTensor::D(indices[0]));
        } else if (x == "") {
            assert(indices.size() == 0 && cds.size() == 0);
            return SpinPermTensor::I();
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
    static string get_sub_expr(const string &expr, int i, int j) {
        int cnt = 0, depth = 0, start = -1, extra = 0;
        stringstream ss;
        for (auto &c : expr) {
            if (c == 'C' || c == 'D' || c == 'c' || c == 'd' || c == 'T') {
                if (cnt >= i && cnt < j)
                    ss << c;
                cnt++;
            } else if (c == '(') {
                depth++;
                if (cnt == i && start == -1 && j > i + 1)
                    start = depth;
                if (cnt >= i && cnt < j && start != -1 && depth >= start)
                    ss << c, extra++;
            } else if (c == ')') {
                depth--;
                if (cnt >= i && cnt <= j && start != -1 && depth + 1 >= start)
                    ss << c, extra--;
            } else if (c >= '0' && c <= '9') {
                if (cnt >= i && cnt <= j && start != -1 && depth + 1 >= start)
                    ss << c;
            } else if (c == '+') {
                if (cnt >= i && cnt < j && start != -1 && depth >= start)
                    ss << c;
            }
        }
        return ss.str().substr(extra);
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
        for (int ic = 0; ic < (int)x.length(); ic++) {
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
    // (((.+.)0+.)0+.)0 -> 1 2 3
    // ((.+.)0+(.+.)0) -> 1 2
    static vector<uint16_t> find_split_indices_from_right(const string &x,
                                                          int start_depth = 1) {
        int dot_cnt = 0, depth = 0;
        vector<uint16_t> r, rx;
        for (int ic = (int)x.length() - 1; ic >= 0; ic--) {
            auto &c = x[ic];
            if (c == ')')
                depth++;
            else if (c == '(')
                depth--;
            else if (c == '.')
                dot_cnt++;
            else if (c == '+' && depth == start_depth) {
                r.push_back(dot_cnt);
                start_depth++;
            }
        }
        for (int i = (int)r.size() - 1; i >= 0; i--)
            rx.push_back(dot_cnt - r[i]);
        return rx;
    }
    // site_dq = 2 -> heisenberg spin model
    static vector<string> initialize(uint16_t n, uint16_t twos,
                                     uint16_t site_dq = 1) {
        map<pair<uint16_t, uint16_t>, vector<string>> mp;
        mp[make_pair(0, 0)] = vector<string>{""};
        mp[make_pair(1, site_dq)] = vector<string>{"."};
        for (int k = 2; k <= n; k++)
            for (int j = site_dq == 2 ? 0 : k % 2; j <= k * site_dq; j += 2) {
                mp[make_pair(k, j)] = vector<string>();
                vector<string> &mpz = mp.at(make_pair(k, j));
                for (int p = 1; p < k; p++)
                    for (int jl = site_dq == 2 ? 0 : p % 2; jl <= p * site_dq;
                         jl += 2)
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

struct SpinRecoupling {
    struct Level {
        int8_t left_idx, mid_idx, right_idx, left_cnt, right_cnt;
    };
    static Level get_level(const string &x, int8_t i_start) {
        Level r;
        if (x[i_start] != '(') {
            r.right_idx = -1;
            return r;
        }
        r.left_idx = i_start + 1;
        int8_t dot_cnt = 0, depth = 0;
        for (int8_t i = i_start; i < (int8_t)x.length(); i++) {
            auto &c = x[i];
            if (c == '(')
                depth++;
            else if (c == ')') {
                depth--;
                if (depth == 0) {
                    r.right_idx = i + 1;
                    break;
                }
            } else if (c == '+' && depth == 1)
                r.mid_idx = i + 1, r.left_cnt = dot_cnt;
            else if (c == '.' || c == 'C' || c == 'D' || c == 'T')
                dot_cnt++;
        }
        r.right_cnt = dot_cnt - r.left_cnt;
        return r;
    }
    static int8_t get_twos(const string &x, Level h, bool heis) {
        if (h.right_idx == -1)
            return heis ? 2 : 1;
        int8_t g = 0;
        for (int8_t i = h.right_idx; i < (int8_t)x.length(); i++)
            if (x[i] >= '0' && x[i] <= '9')
                g = g * 10 + (int8_t)(x[i] - '0');
            else
                break;
        return g;
    }
    static map<string, double> recouple(const map<string, double> &x,
                                        int8_t i_start, int8_t left_cnt,
                                        SU2CG cg, bool heis) {
        const string &x0 = x.cbegin()->first;
        Level h = get_level(x0, i_start);
        if (left_cnt == h.left_cnt)
            return x;
        Level hl = get_level(x0, h.left_idx);
        Level hr = get_level(x0, h.mid_idx);
        map<string, double> v;
        if (left_cnt > h.left_cnt) {
            if (hr.left_cnt != left_cnt - h.left_cnt)
                return recouple(
                    recouple(x, h.mid_idx, left_cnt - h.left_cnt, cg, heis),
                    i_start, left_cnt, cg, heis);
            Level hrl = get_level(x0, hr.left_idx);
            Level hrr = get_level(x0, hr.mid_idx);
            for (auto &xm : x) {
                const string &xx = xm.first;
                // 1+(2+3) -> (1+2)+3
                int8_t j1 = get_twos(xx, hl, heis),
                       j23 = get_twos(xx, hr, heis),
                       j2 = get_twos(xx, hrl, heis),
                       j3 = get_twos(xx, hrr, heis), j = get_twos(xx, h, heis);
                for (int8_t j12 = abs(j1 - j2); j12 <= j1 + j2; j12 += 2)
                    if (j >= abs(j12 - j3) && j <= j12 + j3) {
                        stringstream ss;
                        ss << xx.substr(0, h.left_idx) << "("
                           << xx.substr(h.left_idx, h.mid_idx - 1 - h.left_idx)
                           << "+"
                           << xx.substr(hr.left_idx,
                                        hr.mid_idx - 1 - hr.left_idx)
                           << ")" << (int)j12 << "+"
                           << xx.substr(hr.mid_idx,
                                        hr.right_idx - 1 - hr.mid_idx)
                           << xx.substr(h.right_idx - 1);
                        v[ss.str()] += xm.second *
                                       cg.racah(j1, j2, j, j3, j12, j23) *
                                       sqrt((j12 + 1) * (j23 + 1));
                    }
            }
        } else {
            if (hl.right_cnt != h.left_cnt - left_cnt)
                return recouple(recouple(x, h.left_idx, left_cnt, cg, heis),
                                i_start, left_cnt, cg, heis);
            Level hll = get_level(x0, hl.left_idx);
            Level hlr = get_level(x0, hl.mid_idx);
            for (auto &xm : x) {
                const string &xx = xm.first;
                // (1+2)+3 -> 1+(2+3)
                int8_t j1 = get_twos(xx, hll, heis),
                       j2 = get_twos(xx, hlr, heis),
                       j12 = get_twos(xx, hl, heis),
                       j3 = get_twos(xx, hr, heis), j = get_twos(xx, h, heis);
                for (int8_t j23 = abs(j2 - j3); j23 <= j2 + j3; j23 += 2)
                    if (j >= abs(j1 - j23) && j <= j1 + j23) {
                        stringstream ss;
                        ss << xx.substr(0, h.left_idx)
                           << xx.substr(hl.left_idx,
                                        hl.mid_idx - 1 - hl.left_idx)
                           << "+("
                           << xx.substr(hl.mid_idx,
                                        hl.right_idx - 1 - hl.mid_idx)
                           << "+"
                           << xx.substr(h.mid_idx, h.right_idx - 1 - h.mid_idx)
                           << ")" << (int)j23 << xx.substr(h.right_idx - 1);
                        v[ss.str()] += xm.second *
                                       cg.racah(j3, j2, j, j1, j23, j12) *
                                       sqrt((j23 + 1) * (j12 + 1));
                    }
            }
        }
        for (auto it = v.cbegin(); it != v.cend();) {
            if (abs(it->second) < 1E-12)
                it = v.erase(it);
            else
                it++;
        }
        return v;
    }
    static map<string, double>
    recouple_split(const map<string, double> &x,
                   const vector<uint16_t> &ref_indices, int split_idx, SU2CG cg,
                   bool heis) {
        int nn = ref_indices.size();
        vector<uint16_t> ref_split_idx;
        int ref_mid = -1;
        for (int i = 1; i < nn; i++)
            if (ref_indices[i] != ref_indices[i - 1]) {
                ref_split_idx.push_back(i);
                if (i == split_idx)
                    ref_mid = (int)ref_split_idx.size() - 1;
            }
        if (ref_split_idx.size() == 0)
            return x;
        if (split_idx == nn)
            ref_mid = (int)ref_split_idx.size() - 1;
        map<string, double> r = x;
        if (split_idx != -1) {
            // handle npdm split
            int ii = 0, ir = 0;
            for (int i = ref_mid; i < (int)ref_split_idx.size(); i++) {
                r = recouple(r, ii, ref_split_idx[i] - ir, cg, heis);
                ii = get_level(r.begin()->first, ii).mid_idx;
                ir = ref_split_idx[i];
            }
            Level h = get_level(r.begin()->first, 0);
            ii = h.left_idx;
            for (int i = ref_mid - 1; i >= 0; i--) {
                r = recouple(r, ii, ref_split_idx[i], cg, heis);
                ii = get_level(r.begin()->first, ii).left_idx;
            }
        } else {
            // handle hamiltonian split
            int ii = 0, ir = 0;
            for (auto &rr : ref_split_idx) {
                r = recouple(r, ii, rr - ir, cg, heis);
                ii = get_level(r.begin()->first, ii).mid_idx, ir = rr;
            }
        }
        return r;
    }
};

struct SpinPermPattern {
    uint16_t n;
    vector<uint16_t> data;
    SpinPermPattern(uint16_t n) : n(n), data(initialize(n)) {}
    static vector<uint16_t> all_reordering(const vector<uint16_t> &x) {
        if (x.size() == 0)
            return x;
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
        for (uint16_t i = 0; i <= n; i++)
            cnt += mp.at(make_pair(i, n)).size();
        vector<uint16_t> r(cnt * (n + 2));
        size_t ic = 0;
        for (uint16_t i = 0; i <= n; i++) {
            vector<vector<uint16_t>> &mpx = mp.at(make_pair(i, n));
            for (auto &x : mpx) {
                r[ic++] = i;
                memcpy(r.data() + ic, x.data(), sizeof(uint16_t) * n);
                for (r[ic + n] = (n - 1) >> 1;
                     r[ic + n] < n - 1 && x[r[ic + n]] == x[r[ic + n] + 1];
                     r[ic + n]++)
                    ;
                r[ic + n]++;
                ic += n + 1;
            }
        }
        return r;
    }
    size_t count() const { return data.size() / (n + 2); }
    vector<uint16_t> operator[](size_t i) const {
        return vector<uint16_t>(data.begin() + i * (n + 2) + 1,
                                data.begin() + i * (n + 2) + n + 1);
    }
    uint16_t get_split_index(size_t i) const {
        return data[i * (n + 2) + n + 1];
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
    // if split_idx != -1: npdm case. else: hamiltonian mpo case
    static vector<string> get_unique(const vector<uint8_t> &cds,
                                     const vector<uint16_t> &ref_indices,
                                     int target_twos = 0, int split_idx = -1,
                                     bool ref_split = false) {
        SU2CG cg;
        int nn = cds.size();
        vector<uint16_t> indices = ref_indices;
        if (indices.size() == 0)
            for (int i = 0; i < nn; i++)
                indices.push_back(i);
        vector<uint16_t> ref_split_idx;
        int ref_mid = -1;
        for (int i = 1; i < nn; i++)
            if (indices[i] != indices[i - 1]) {
                ref_split_idx.push_back(i);
                if (i == split_idx)
                    ref_mid = (int)ref_split_idx.size() - 1;
            }
        if (split_idx == nn)
            ref_mid = (int)ref_split_idx.size() - 1;
        bool heis = cds.size() != 0 && cds[0] == 2;
        vector<string> pp =
            SpinPermRecoupling::initialize(nn, target_twos, heis ? 2 : 1);
        vector<SpinPermTensor> ts(pp.size());
        for (int i = 0; i < (int)pp.size(); i++) {
            if (split_idx != -1 && !ref_split) {
                if (SpinPermRecoupling::find_split_index(pp[i]) != split_idx)
                    continue;
            } else if (split_idx != -1) {
                // handle npdm split
                if (ref_split_idx.size() != 0) {
                    if (split_idx < nn && SpinPermRecoupling::find_split_index(
                                              pp[i]) != split_idx)
                        continue;
                    vector<uint16_t> act_split_left_idx =
                        SpinPermRecoupling::find_split_indices_from_right(
                            pp[i]);
                    vector<uint16_t> act_split_right_idx =
                        SpinPermRecoupling::find_split_indices_from_left(pp[i]);
                    if (act_split_left_idx.size() < ref_mid + 1 ||
                        !equal(ref_split_idx.begin(),
                               ref_split_idx.begin() + ref_mid + 1,
                               act_split_left_idx.begin() +
                                   (act_split_left_idx.size() - ref_mid - 1)))
                        continue;
                    else if (act_split_right_idx.size() <
                                 (int)ref_split_idx.size() - ref_mid ||
                             !equal(ref_split_idx.begin() + ref_mid,
                                    ref_split_idx.end(),
                                    act_split_right_idx.begin()))
                        continue;
                }
            } else if (ref_split) {
                vector<uint16_t> act_split_idx =
                    SpinPermRecoupling::find_split_indices_from_left(pp[i]);
                if (act_split_idx.size() < ref_split_idx.size() ||
                    !equal(ref_split_idx.begin(), ref_split_idx.end(),
                           act_split_idx.begin()))
                    continue;
            }
            ts[i] = SpinPermRecoupling::make_tensor(pp[i], indices, cds, cg);
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
    bool is_su2;
    int8_t left_vacuum;
    SpinPermScheme() {}
    SpinPermScheme(string spin_str, bool su2 = true, bool is_fermion = true,
                   bool is_npdm = false) {
        int nn = SpinPermRecoupling::count_cds(spin_str);
        SpinPermScheme r =
            su2 ? SpinPermScheme::initialize_su2(nn, spin_str, is_npdm)
                : SpinPermScheme::initialize_sz(nn, spin_str, is_fermion);
        index_patterns = r.index_patterns;
        data = r.data;
        is_su2 = r.is_su2;
        left_vacuum = r.left_vacuum;
    }
    static SpinPermScheme initialize_sz(int nn, string spin_str,
                                        bool is_fermion = true) {
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
            int nj = irr.size() == 0 ? 1 : rr.size() / irr.size();
            for (int jj = 0; jj < nj; jj++) {
                vector<uint16_t> indices(rr.begin() + jj * irr.size(),
                                         rr.begin() + (jj + 1) * irr.size());
                vector<uint16_t> perm =
                    SpinPermTensor::find_pattern_perm(indices);
                r.data[i][perm] = vector<pair<double, string>>();
                vector<pair<double, string>> &rec_formula = r.data[i].at(perm);
                auto pis = SpinPermTensor::auto_sort_string(indices, spin_str);
                rec_formula.push_back(make_pair(
                    is_fermion ? (double)pis.second : 1.0, pis.first));
            }
        }
        r.is_su2 = false;
        r.left_vacuum = 0;
        return r;
    }
    static SpinPermScheme initialize_su2_old(int nn, string spin_str,
                                             bool is_npdm = false) {
        using T = SpinPermTensor;
        using R = SpinPermRecoupling;
        SU2CG cg;
        vector<uint8_t> cds;
        spin_str = SpinPermRecoupling::split_cds(spin_str, cds);
        int target_twos = SpinPermRecoupling::get_target_twos(spin_str);
        SpinPermPattern spat(nn);
        vector<double> mptr;
        SpinPermScheme r;
        r.index_patterns.resize(spat.count());
        r.data.resize(spat.count());
        for (size_t i = 0; i < spat.count(); i++) {
            vector<uint16_t> irr = spat[i];
            int split_idx = is_npdm ? spat.get_split_index(i) : -1;
            r.index_patterns[i] = irr;
            vector<uint16_t> rr = SpinPermPattern::all_reordering(irr);
            int nj = irr.size() == 0 ? 1 : rr.size() / irr.size();
            vector<string> ttp = SpinPermPattern::get_unique(
                cds, irr, target_twos, split_idx, true);
            for (int jj = 0; jj < nj; jj++) {
                vector<uint16_t> indices(rr.begin() + jj * irr.size(),
                                         rr.begin() + (jj + 1) * irr.size());
                vector<uint16_t> perm =
                    SpinPermTensor::find_pattern_perm(indices);
                r.data[i][perm] = vector<pair<double, string>>();
                vector<pair<double, string>> &rec_formula = r.data[i].at(perm);
                T x = R::make_tensor(spin_str, indices, cds, cg) * 2;
                T xs = x.auto_sort();
                vector<uint8_t> target_cds = xs.get_cds();
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
                mptr.resize((pgg[0].size() + 1) *
                            max(tts.size() + 1, (size_t)5));
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
        r.is_su2 = true;
        r.left_vacuum = (int8_t)target_twos;
        return r;
    }
    static SpinPermScheme initialize_su2(int nn, string spin_str,
                                         bool is_npdm = false) {
        using T = SpinPermTensor;
        using R = SpinPermRecoupling;
        SU2CG cg;
        vector<uint8_t> cds;
        if (spin_str.find('T') != string::npos)
            return initialize_su2_old(nn, spin_str, is_npdm);
        spin_str = R::split_cds(spin_str, cds);
        bool heis = cds.size() != 0 && cds[0] == 2;
        int target_twos = R::get_target_twos(spin_str);
        SpinPermPattern spat(nn);
        SpinPermScheme r;
        r.index_patterns.resize(spat.count());
        r.data.resize(spat.count());
        map<pair<int, vector<uint16_t>>, int> unique_data;
        int xi = spat.count() - 1;
        vector<uint16_t> xrr = spat[xi];
        int xq = is_npdm ? spat.get_split_index(xi) : -1;
        vector<string> unique_strs =
            SpinPermPattern::get_unique(cds, xrr, target_twos, xq, true);
        r.index_patterns[xi] = xrr;
        vector<uint16_t> rr = SpinPermPattern::all_reordering(xrr);
        int xnj = xrr.size() == 0 ? 1 : rr.size() / xrr.size();
        for (int jj = 0; jj < xnj; jj++) {
            vector<uint16_t> indices(rr.begin() + jj * xrr.size(),
                                     rr.begin() + (jj + 1) * xrr.size());
            vector<uint16_t> perm = SpinPermTensor::find_pattern_perm(indices);
            r.data[xi][perm] = vector<pair<double, string>>();
        }
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static, 50) num_threads(ntg)
        for (int jj = 0; jj < xnj; jj++) {
            vector<uint16_t> indices(rr.begin() + jj * xrr.size(),
                                     rr.begin() + (jj + 1) * xrr.size());
            vector<uint16_t> perm = SpinPermTensor::find_pattern_perm(indices);
            vector<pair<double, string>> &udq = r.data[xi].at(perm);
            T x = R::make_tensor(spin_str, indices, cds, cg);
            T xs = x.auto_sort().simplify();
            vector<uint8_t> target_cds = xs.get_cds();
            vector<T> tts;
            vector<int> idxs;
            vector<double> dp;
            tts.reserve(unique_strs.size());
            idxs.reserve(unique_strs.size());
            dp.reserve(unique_strs.size());
            for (int j = 0; j < unique_strs.size(); j++) {
                T ts = R::make_tensor(unique_strs[j], xrr, target_cds, cg);
                double tdp = T::dot_product(ts, xs)[0];
                if (abs(tdp) > 1E-12)
                    tts.push_back(ts), idxs.push_back(j), dp.push_back(tdp);
            }
            bool found = false;
            int l = (int)idxs.size();
            vector<int> xx(l);
            for (int il = 1; il <= l; il++) {
                for (int j = 0; j < il; j++)
                    xx[j] = j;
                for (;;) {
                    double tdp = 0;
                    for (int j = 0; j < il; j++)
                        tdp += dp[xx[j]] * dp[xx[j]];
                    if (abs(tdp - 1) < 1E-12) {
                        SpinPermTensor std = SpinPermTensor();
                        for (int j = 0; j < il; j++)
                            std = std + tts[xx[j]] * dp[xx[j]];
                        if (std.simplify() == xs) {
                            found = true;
                            for (int j = 0; j < il; j++)
                                udq.push_back(make_pair(
                                    dp[xx[j]],
                                    R::make_with_cds(unique_strs[idxs[xx[j]]],
                                                     target_cds)));
                            break;
                        }
                    }
                    bool has_next = true;
                    for (int j = 0; j < il; j++)
                        if ((j == il - 1 && xx[j] < l - 1) ||
                            (j < il - 1 && xx[j] + 1 < xx[j + 1])) {
                            xx[j]++;
                            for (int k = 0; k < j; k++)
                                xx[k] = k;
                        } else if (j == il - 1)
                            has_next = false;
                    if (!has_next)
                        break;
                }
                if (found)
                    break;
            }
            assert(found);
        }
        for (int i = 0; i < spat.count() - 1; i++) {
            vector<uint16_t> irr = spat[i];
            r.index_patterns[i] = irr;
            vector<uint16_t> rr = SpinPermPattern::all_reordering(irr);
            int nj = irr.size() == 0 ? 1 : rr.size() / irr.size();
            int iq = is_npdm ? spat.get_split_index(i) : -1;
            vector<map<string, double>> recouples(unique_strs.size());
            map<string, int> map_unique_strs;
            for (int iu = 0; iu < (int)unique_strs.size(); iu++) {
                map<string, double> p =
                    map<string, double>{make_pair(unique_strs[iu], 1.0)};
                recouples[iu] =
                    SpinRecoupling::recouple_split(p, irr, iq, cg, heis);
                map_unique_strs[unique_strs[iu]] = iu;
            }
            for (int jj = 0; jj < nj; jj++) {
                vector<uint16_t> indices(rr.begin() + jj * irr.size(),
                                         rr.begin() + (jj + 1) * irr.size());
                vector<uint16_t> perm =
                    SpinPermTensor::find_pattern_perm(indices);
                r.data[i][perm] = vector<pair<double, string>>();
            }
#pragma omp parallel for schedule(static, 50) num_threads(ntg)
            for (int jj = 0; jj < nj; jj++) {
                vector<uint16_t> indices(rr.begin() + jj * irr.size(),
                                         rr.begin() + (jj + 1) * irr.size());
                vector<uint16_t> perm =
                    SpinPermTensor::find_pattern_perm(indices);
                vector<pair<double, string>> &udq = r.data[i].at(perm);
                vector<pair<double, string>> &ref_udq = r.data[xi].at(perm);
                vector<uint8_t> tcds;
                map<string, double> r;
                for (auto &mr : ref_udq) {
                    string k = R::split_cds(mr.second, tcds);
                    for (auto &rr : recouples[map_unique_strs.at(k)])
                        r[rr.first] += rr.second * mr.first;
                }
                for (auto &mr : r)
                    if (abs(mr.second) > 1E-12)
                        udq.push_back(make_pair(
                            mr.second, R::make_with_cds(mr.first, tcds)));
                assert(udq.size() != 0);
            }
        }
        threading->activate_normal();
        r.is_su2 = true;
        r.left_vacuum = (int8_t)target_twos;
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

struct NPDMCounter {
    int n_ops, n_sites;
    vector<vector<uint32_t>> dp;
    NPDMCounter(int n_ops, int n_sites) : n_ops(n_ops), n_sites(n_sites) {
        // dp[n][k] = term count of n distinct ops with index range [0, k)
        dp.resize(n_ops + 1);
        for (int i = 0; i < n_ops + 1; i++) {
            dp[i] = vector<uint32_t>(n_sites + 1, 1);
            if (i != 0) {
                dp[i][0] = 0;
                for (int kk = 1; kk <= n_sites; kk++)
                    dp[i][kk] = dp[i][kk - 1] + dp[i - 1][kk - 1];
            }
        }
    }
    uint32_t count_left(const vector<uint16_t> &pattern, int k, bool f) const {
        set<uint16_t> g(pattern.begin(), pattern.end());
        if (k == n_sites - 1 || k == -1)
            return pattern.size() == 0 ? 1 : 0;
        if (f && g.size() != 0)
            return dp[(int)g.size() - 1][k];
        else
            return dp[(int)g.size()][k + 1];
    }
    bool init_left(const vector<uint16_t> &pattern, int k, bool f,
                   vector<uint16_t> &r) const {
        r.clear();
        r.resize(pattern.size(), 0);
        if (pattern.size() == 0)
            return true;
        for (int i = 1; i < (int)pattern.size(); i++)
            r[i] = r[i - 1] + (pattern[i - 1] != pattern[i]);
        if (r.back() > k)
            return false;
        if (f && r.size() > 0) {
            r.back() = k;
            for (int i = (int)r.size() - 2; i >= 0; i--)
                if (pattern[i] == pattern[i + 1])
                    r[i] = k;
                else
                    break;
        }
        return true;
    }
    bool next_left(const vector<uint16_t> &pattern, int k,
                   vector<uint16_t> &r) const {
        if (pattern.size() == 0)
            return false;
        vector<uint16_t> kk(1, 0);
        for (int i = 1; i < (int)pattern.size(); i++)
            if (pattern[i - 1] != pattern[i])
                kk.push_back(i);
        for (int i = 0; i < (int)kk.size(); i++)
            if ((i == kk.size() - 1 && r[kk[i]] < k) ||
                (i < kk.size() - 1 && r[kk[i]] + 1 < r[kk[i + 1]])) {
                int j = i == kk.size() - 1 ? pattern.size() : kk[i + 1];
                for (int m = kk[i]; m < j; m++)
                    r[m]++;
                for (int m = 0; m < kk[i]; m++)
                    r[m] =
                        m == 0 ? 0 : r[m - 1] + (pattern[m - 1] != pattern[m]);
                return true;
            } else if (i == kk.size() - 1)
                return false;
        return false;
    }
    uint32_t count_right(const vector<uint16_t> &pattern, int k) const {
        set<uint16_t> g(pattern.begin(), pattern.end());
        if (k == 0)
            return pattern.size() == 0 ? 1 : 0;
        else
            return dp[(int)g.size()][n_sites - k];
    }
    bool init_right(const vector<uint16_t> &pattern, int k,
                    vector<uint16_t> &r) const {
        r.clear();
        if (pattern.size() == 0)
            return true;
        r.resize(pattern.size(), k);
        for (int i = 1; i < (int)pattern.size(); i++)
            r[i] = r[i - 1] + (pattern[i - 1] != pattern[i]);
        return r.back() < n_sites;
    }
    bool index_right(const vector<uint16_t> &pattern, int k, int ix,
                     vector<uint16_t> &r) const {
        r.clear();
        if (pattern.size() == 0)
            return true;
        r.resize(pattern.size(), 0);
        int gz = 1;
        for (int i = 1; i < (int)pattern.size(); i++)
            gz += (pattern[i - 1] != pattern[i]);
        for (int i = 0; i < (int)pattern.size(); i++)
            if (i != 0 && pattern[i - 1] == pattern[i])
                r[i] = r[i - 1];
            else {
                int cnt = dp[gz][n_sites - k];
                r[i] = n_sites -
                       (uint16_t)(lower_bound(dp[gz].begin(),
                                              dp[gz].begin() + n_sites - k + 1,
                                              cnt - ix) -
                                  dp[gz].begin());
                ix -= cnt - dp[gz][n_sites - r[i]];
                k = r[i] + 1;
                gz--;
            }
        assert(ix == 0);
        return r.back() < n_sites;
    }
    bool next_right(const vector<uint16_t> &pattern, int k,
                    vector<uint16_t> &r) const {
        if (pattern.size() == 0)
            return false;
        vector<uint16_t> kk(1, 0);
        for (int i = 1; i < (int)pattern.size(); i++)
            if (pattern[i - 1] != pattern[i])
                kk.push_back(i);
        for (int i = (int)kk.size() - 1; i >= 0; i--)
            if (r[kk[i]] < n_sites - kk.size() + i) {
                int j = i == kk.size() - 1 ? pattern.size() : kk[i + 1];
                for (int m = kk[i]; m < j; m++)
                    r[m]++;
                for (int m = j; m < (int)r.size(); m++)
                    r[m] = r[m - 1] + (pattern[m - 1] != pattern[m]);
                return true;
            } else if (i == 0)
                return false;
        return false;
    }
};

struct NPDMScheme {
    vector<pair<pair<vector<uint16_t>, string>, bool>> left_terms, right_terms;
    vector<vector<uint32_t>> left_blocking, right_blocking;
    vector<vector<uint16_t>> middle_perm_patterns;
    vector<vector<string>> middle_terms;
    vector<vector<pair<uint32_t, uint32_t>>> middle_blocking;
    vector<pair<pair<vector<uint16_t>, string>, bool>> last_right_terms;
    vector<vector<uint32_t>> last_right_blocking;
    vector<vector<pair<uint32_t, uint32_t>>> last_middle_blocking;
    vector<string> local_terms;
    vector<shared_ptr<SpinPermScheme>> perms;
    int n_max_ops;
    NPDMScheme(const shared_ptr<SpinPermScheme> &perm)
        : NPDMScheme(vector<shared_ptr<SpinPermScheme>>{perm}) {}
    NPDMScheme(const vector<shared_ptr<SpinPermScheme>> &perms) : perms(perms) {
        n_max_ops = 0;
        for (auto perm : perms)
            for (int i = 0; i < (int)perm->index_patterns.size(); i++)
                n_max_ops = max(n_max_ops, (int)perm->index_patterns[i].size());
        initialize();
    }
    void initialize() {
        set<string> locals;
        map<vector<uint16_t>, map<string, int>> left_patterns, right_patterns;
        map<vector<uint16_t>, map<string, int>> last_right_patterns;
        left_patterns[vector<uint16_t>()][""] = true;
        right_patterns[vector<uint16_t>()][""] = true;
        locals.insert("");
        map<vector<uint16_t>, vector<pair<int, int>>> middle_patterns;
        for (int i = 0; i < (int)perms.size(); i++)
            for (int j = 0; j < (int)perms[i]->index_patterns.size(); j++)
                middle_patterns[perms[i]->index_patterns[j]].push_back(
                    make_pair(i, j));
        middle_perm_patterns.clear();
        middle_perm_patterns.reserve(middle_patterns.size());
        for (auto pr : middle_patterns)
            middle_perm_patterns.push_back(pr.first);
        sort(middle_perm_patterns.begin(), middle_perm_patterns.end(),
             [](const vector<uint16_t> &a, const vector<uint16_t> &b) {
                 return a.size() != b.size() ? a.size() < b.size() : a < b;
             });
        vector<set<string>> cds(middle_perm_patterns.size());
        for (int i = 0; i < (int)middle_perm_patterns.size(); i++)
            for (auto &r : middle_patterns.at(middle_perm_patterns[i]))
                for (auto &dt : perms[r.first]->data[r.second])
                    for (auto x : dt.second)
                        cds[i].insert(x.second);
        middle_terms.clear();
        for (int i = 0; i < (int)cds.size(); i++) {
            middle_terms.push_back(
                vector<string>(cds[i].begin(), cds[i].end()));
            sort(middle_terms[i].begin(), middle_terms[i].end(),
                 [](const string &a, const string &b) {
                     return a.length() != b.length() ? a.length() < b.length()
                                                     : a < b;
                 });
        }
        // find required left / right terms
        for (int i = 0; i < (int)middle_perm_patterns.size(); i++) {
            vector<uint16_t> &pat = middle_perm_patterns[i];
            int n_ops = (int)pat.size();
            int ii = n_ops - n_ops / 2 - 1, kk;
            while (ii < n_ops - 1 && pat[ii] == pat[ii + 1])
                ii++;
            vector<pair<int, bool>> left_sub_pats;
            for (int jj = 0; jj < ii; jj++)
                if (pat[jj] != pat[jj + 1])
                    left_sub_pats.push_back(make_pair(jj + 1, false));
            left_sub_pats.push_back(make_pair(ii + 1, true));
            for (auto &k : left_sub_pats) {
                vector<uint16_t> spat(pat.begin(), pat.begin() + k.first);
                kk = k.first - 1;
                while (kk > 0 && pat[kk - 1] == pat[kk])
                    kk--;
                for (auto &cd : cds[i]) {
                    locals.insert(
                        SpinPermRecoupling::get_sub_expr(cd, kk, k.first));
                    string xcd =
                        SpinPermRecoupling::get_sub_expr(cd, 0, k.first);
                    if (left_patterns[spat].count(xcd) == 0 || !k.second)
                        left_patterns[spat][xcd] = k.second;
                }
            }
            vector<uint16_t> rpat(pat.begin() + (ii + 1), pat.end());
            const uint16_t rref = rpat.size() == 0 ? 0 : rpat[0];
            for (auto &r : rpat)
                r -= rref;
            kk = ii + 1;
            while (kk < n_ops - 1 && pat[kk] == pat[kk + 1])
                kk++;
            for (auto &cd : cds[i]) {
                locals.insert(
                    SpinPermRecoupling::get_sub_expr(cd, ii + 1, kk + 1));
                string xcd =
                    SpinPermRecoupling::get_sub_expr(cd, ii + 1, n_ops);
                right_patterns[rpat][xcd] = true;
            }
            if (ii == n_ops - 1) {
                while (ii > 0 && pat[ii - 1] == pat[ii])
                    ii--;
                vector<uint16_t> rpat(pat.begin() + ii, pat.end());
                const uint16_t rref = rpat.size() == 0 ? 0 : rpat[0];
                for (auto &r : rpat)
                    r -= rref;
                for (auto &cd : cds[i]) {
                    string xcd =
                        SpinPermRecoupling::get_sub_expr(cd, ii, n_ops);
                    locals.insert(xcd);
                    last_right_patterns[rpat][xcd] = true;
                }
            }
        }
        for (auto &lr : last_right_patterns)
            if (right_patterns.count(lr.first))
                for (auto &llr : lr.second)
                    if (right_patterns.at(lr.first).count(llr.first))
                        llr.second = false;
        // local sorting
        local_terms = vector<string>(locals.begin(), locals.end());
        sort(local_terms.begin(), local_terms.end(),
             [](const string &a, const string &b) {
                 return a.length() != b.length() ? a.length() < b.length()
                                                 : a < b;
             });
        // left terms sorting
        vector<vector<uint16_t>> map_keys;
        map_keys.reserve(left_patterns.size());
        for (auto &p : left_patterns)
            map_keys.push_back(p.first);
        sort(map_keys.begin(), map_keys.end(),
             [](const vector<uint16_t> &a, const vector<uint16_t> &b) {
                 return a.size() != b.size() ? a.size() < b.size() : a < b;
             });
        left_terms.clear();
        for (auto &k : map_keys)
            for (auto &r : left_patterns.at(k)) {
                left_terms.push_back(
                    make_pair(make_pair(k, r.first), (bool)r.second));
                r.second = left_terms.size() - 1;
            }
        // right terms sorting
        map_keys.clear();
        map_keys.reserve(right_patterns.size());
        for (auto &p : right_patterns)
            map_keys.push_back(p.first);
        sort(map_keys.begin(), map_keys.end(),
             [](const vector<uint16_t> &a, const vector<uint16_t> &b) {
                 return a.size() != b.size() ? a.size() < b.size() : a < b;
             });
        right_terms.clear();
        for (auto &k : map_keys)
            for (auto &r : right_patterns.at(k)) {
                right_terms.push_back(
                    make_pair(make_pair(k, r.first), (bool)r.second));
                r.second = right_terms.size() - 1;
            }
        // last right terms sorting
        map_keys.clear();
        map_keys.reserve(last_right_patterns.size());
        for (auto &p : last_right_patterns)
            map_keys.push_back(p.first);
        sort(map_keys.begin(), map_keys.end(),
             [](const vector<uint16_t> &a, const vector<uint16_t> &b) {
                 return a.size() != b.size() ? a.size() < b.size() : a < b;
             });
        last_right_terms.clear();
        for (auto &k : map_keys)
            for (auto &r : last_right_patterns.at(k))
                if (r.second) {
                    last_right_terms.push_back(
                        make_pair(make_pair(k, r.first), (bool)r.second));
                    r.second = last_right_terms.size() - 1 + right_terms.size();
                } else
                    r.second = right_patterns.at(k).at(r.first);
        // middle blocking
        middle_blocking.clear();
        for (int i = 0; i < (int)middle_perm_patterns.size(); i++) {
            middle_blocking.push_back(
                vector<pair<uint32_t, uint32_t>>(middle_terms[i].size()));
            vector<uint16_t> &pat = middle_perm_patterns[i];
            int n_ops = (int)pat.size();
            int ii = n_ops - n_ops / 2 - 1;
            while (ii < n_ops - 1 && pat[ii] == pat[ii + 1])
                ii++;
            vector<uint16_t> lpat(pat.begin(), pat.begin() + ii + 1);
            vector<uint16_t> rpat(pat.begin() + (ii + 1), pat.end());
            const uint16_t rref = rpat.size() == 0 ? 0 : rpat[0];
            for (auto &r : rpat)
                r -= rref;
            for (int j = 0; j < (int)middle_terms[i].size(); j++) {
                uint32_t lp =
                    left_patterns.at(lpat).at(SpinPermRecoupling::get_sub_expr(
                        middle_terms[i][j], 0, ii + 1));
                uint32_t rp =
                    right_patterns.at(rpat).at(SpinPermRecoupling::get_sub_expr(
                        middle_terms[i][j], ii + 1, n_ops));
                middle_blocking[i][j] = make_pair(lp, rp);
            }
        }
        // last middle blocking
        last_middle_blocking.clear();
        for (int i = 0; i < (int)middle_perm_patterns.size(); i++) {
            vector<uint16_t> &pat = middle_perm_patterns[i];
            int n_ops = (int)pat.size();
            int ii = n_ops - n_ops / 2 - 1;
            while (ii < n_ops - 1 && pat[ii] == pat[ii + 1])
                ii++;
            if (ii != n_ops - 1)
                last_middle_blocking.push_back(
                    vector<pair<uint32_t, uint32_t>>());
            else {
                last_middle_blocking.push_back(
                    vector<pair<uint32_t, uint32_t>>(middle_terms[i].size()));
                while (ii > 0 && pat[ii - 1] == pat[ii])
                    ii--;
                vector<uint16_t> lpat(pat.begin(), pat.begin() + ii);
                vector<uint16_t> rpat(pat.begin() + ii, pat.end());
                const uint16_t rref = rpat.size() == 0 ? 0 : rpat[0];
                for (auto &r : rpat)
                    r -= rref;
                for (int j = 0; j < (int)middle_terms[i].size(); j++) {
                    uint32_t lp = left_patterns.at(lpat).at(
                        SpinPermRecoupling::get_sub_expr(middle_terms[i][j], 0,
                                                         ii));
                    uint32_t rp = last_right_patterns.at(rpat).at(
                        SpinPermRecoupling::get_sub_expr(middle_terms[i][j], ii,
                                                         n_ops));
                    last_middle_blocking[i][j] = make_pair(lp, rp);
                }
            }
        }
        map<string, uint32_t> local_map;
        for (int i = 0; i < (int)local_terms.size(); i++)
            local_map[local_terms[i]] = i;
        // left blocking
        left_blocking.resize(left_terms.size());
        for (int ir = 0; ir < (int)left_terms.size(); ir++) {
            auto &r = left_terms[ir];
            left_blocking[ir].clear();
            if (!r.second || r.first.first.size() == 0) {
                left_blocking[ir].push_back(ir);
                left_blocking[ir].push_back(local_map.at(""));
            }
            if (r.first.first.size() != 0) {
                int kk = (int)r.first.first.size() - 1;
                while (kk > 0 && r.first.first[kk - 1] == r.first.first[kk])
                    kk--;
                vector<uint16_t> ppat(r.first.first.begin(),
                                      r.first.first.begin() + kk);
                left_blocking[ir].push_back(left_patterns.at(ppat).at(
                    SpinPermRecoupling::get_sub_expr(r.first.second, 0, kk)));
                left_blocking[ir].push_back(
                    local_map.at(SpinPermRecoupling::get_sub_expr(
                        r.first.second, kk, (int)r.first.first.size())));
            }
        }
        // right blocking
        right_blocking.resize(right_terms.size());
        for (int ir = 0; ir < (int)right_terms.size(); ir++) {
            auto &r = right_terms[ir];
            right_blocking[ir].clear();
            if (r.first.first.size() == 0) {
                right_blocking[ir].push_back(local_map.at(""));
                right_blocking[ir].push_back(ir);
            } else {
                if (r.first.first.size() != 0) {
                    int kk = 0;
                    while (kk < (int)r.first.first.size() - 1 &&
                           r.first.first[kk + 1] == r.first.first[kk])
                        kk++;
                    vector<uint16_t> ppat(r.first.first.begin() + kk + 1,
                                          r.first.first.end());
                    const uint16_t rref = ppat.size() == 0 ? 0 : ppat[0];
                    for (auto &r : ppat)
                        r -= rref;
                    right_blocking[ir].push_back(
                        local_map.at(SpinPermRecoupling::get_sub_expr(
                            r.first.second, 0, kk + 1)));
                    right_blocking[ir].push_back(right_patterns.at(ppat).at(
                        SpinPermRecoupling::get_sub_expr(
                            r.first.second, kk + 1,
                            (int)r.first.first.size())));
                }
                right_blocking[ir].push_back(local_map.at(""));
                right_blocking[ir].push_back(ir);
            }
        }
        // last right blocking
        last_right_blocking.resize(last_right_terms.size());
        for (int ir = 0; ir < (int)last_right_terms.size(); ir++) {
            auto &r = last_right_terms[ir];
            assert(r.first.first.size() != 0);
            last_right_blocking[ir].clear();
            last_right_blocking[ir].push_back(local_map.at(r.first.second));
            last_right_blocking[ir].push_back(0);
        }
    }
    string to_str() const {
        stringstream ss;
        ss << "N_MAX_OPS = " << n_max_ops << endl;
        ss << "N_LOCAL = " << local_terms.size() << endl;
        for (int i = 0; i < (int)local_terms.size(); i++)
            ss << "[" << setw(4) << i << "] = " << local_terms[i] << endl;
        ss << endl;
        ss << " N_L = " << left_terms.size() << endl;
        for (int i = 0; i < (int)left_terms.size(); i++) {
            ss << "[" << setw(4) << i << "] = ";
            for (auto &r : left_terms[i].first.first)
                ss << r << " ";
            ss << "- " << left_terms[i].first.second;
            ss << " " << (left_terms[i].second ? "T" : "F");
            ss << " :: ";
            for (int j = 0; j < left_blocking[i].size(); j += 2)
                ss << left_blocking[i][j] << "+" << left_blocking[i][j + 1]
                   << " / ";
            ss << endl;
        }
        ss << endl;
        ss << " N_R = " << right_terms.size() << endl;
        for (int i = 0; i < (int)right_terms.size(); i++) {
            ss << "[" << setw(4) << i << "] = ";
            for (auto &r : right_terms[i].first.first)
                ss << r << " ";
            ss << "- " << right_terms[i].first.second;
            ss << " " << (right_terms[i].second ? "T" : "F");
            ss << " :: ";
            for (int j = 0; j < right_blocking[i].size(); j += 2)
                ss << right_blocking[i][j] << "+" << right_blocking[i][j + 1]
                   << " / ";
            ss << endl;
        }
        ss << endl;
        ss << " N_R_LAST = " << last_right_terms.size() << endl;
        for (int i = 0; i < (int)last_right_terms.size(); i++) {
            ss << "[" << setw(4) << i << "] = ";
            for (auto &r : last_right_terms[i].first.first)
                ss << r << " ";
            ss << "- " << last_right_terms[i].first.second;
            ss << " " << (last_right_terms[i].second ? "T" : "F");
            ss << " :: ";
            for (int j = 0; j < last_right_blocking[i].size(); j += 2)
                ss << last_right_blocking[i][j] << "+"
                   << last_right_blocking[i][j + 1] << " / ";
            ss << endl;
        }
        ss << endl;
        ss << " N_M = " << middle_terms.size() << endl;
        for (int i = 0; i < (int)middle_terms.size(); i++) {
            ss << "[" << setw(4) << i << "] = ";
            for (auto &r : middle_perm_patterns[i])
                ss << r << " ";
            ss << ":: ";
            for (int j = 0; j < (int)middle_terms[i].size(); j++) {
                ss << middle_terms[i][j] << " ";
                ss << middle_blocking[i][j].first << "+"
                   << middle_blocking[i][j].second << " / ";
            }
            ss << endl;
        }
        ss << endl;
        int m_last_count = 0;
        for (int i = 0; i < (int)middle_terms.size(); i++)
            m_last_count += (last_middle_blocking[i].size() != 0);
        ss << " N_M_LAST = " << m_last_count << endl;
        for (int i = 0; i < (int)middle_terms.size(); i++) {
            if (last_middle_blocking[i].size() == 0)
                continue;
            ss << "[" << setw(4) << i << "] = ";
            for (auto &r : middle_perm_patterns[i])
                ss << r << " ";
            ss << ":: ";
            for (int j = 0; j < (int)last_middle_blocking[i].size(); j++) {
                ss << middle_terms[i][j] << " ";
                ss << last_middle_blocking[i][j].first << "+"
                   << last_middle_blocking[i][j].second << " / ";
            }
            ss << endl;
        }
        ss << endl;
        return ss.str();
    }
};

} // namespace block2

// using namespace block2;

// int main() {
// SU2CG cg;
// map<string, double> p;
// p["((C+(D+C)0)1+D)0"] = 1.0;
// vector<uint16_t> ref = {0, 1, 2, 3};

// // p["(C+(D+C)0)1"] = 1.0;
// for (auto &x : p)
//     cout << x.first << " = " << x.second << endl;
// cout << endl;
// map<string, double> q = SpinRecoupling::recouple_split(p, ref, 2, cg);
// // map<string, double> q = SpinRecoupling::recouple(p, 0, 2, cg);
// for (auto &x : q)
//     cout << x.first << " = " << x.second << endl;
// cout << endl;
// p = SpinRecoupling::recouple(q, 0, 1, cg);
// for (auto &x : p)
//     cout << x.first << " = " << x.second << endl;

// shared_ptr<SpinPermScheme> x = make_shared<SpinPermScheme>(
//     SpinPermScheme::initialize_su2_new(4, "((C+D)0+(C+D)0)0", true));
// shared_ptr<SpinPermScheme> x = make_shared<SpinPermScheme>(
//     SpinPermScheme::initialize_su2_new(4, "((C+(C+D)0)1+D)0", true));
// shared_ptr<SpinPermScheme> x = make_shared<SpinPermScheme>(
//     SpinPermScheme::initialize_su2_new(6, "((C+((C+(C+D)0)1+D)0)1+D)0",
//     true));
// shared_ptr<SpinPermScheme> x = make_shared<SpinPermScheme>(
//     SpinPermScheme::initialize_sz(6, "CCCDDD", true));
// auto pp = x->index_patterns[12];
// for (auto &gg : pp)
//     cout << gg << " ";
// cout << endl;
// NPDMCounter ct(6, 7);
// int k = 2;
//     bool kf = true;
//     int cnt = ct.count_left(pp, k, kf);
//     cout << "left = " << cnt << endl;
// vector<uint16_t> xx;
//     cout << ct.init_left(pp, k, kf, xx) << endl;
//     for (int i = 0; i < cnt; i++) {
//         for(auto & gg:xx )
//             cout <<  gg << " ";
//         cout << ">" << ct.next_left(pp, k, xx) << endl;
//     }
//     k = 2;
// int cnt = ct.count_right(pp, k);
// cout << "right = " << cnt << endl;
// cout << ct.init_right(pp, k, xx) << endl;
// for (int i = 0; i < cnt; i++) {
//     cout << "[ " << i << " ] ";
//     for (auto &gg : xx)
//         cout << gg << " ";
//     cout << ">" << ct.next_right(pp, k, xx) << endl;
// }
// cout << endl;
// for (int i = 0; i < cnt; i++) {
//     cout << "[ " << i << " ] ";
//     bool bb = ct.index_right(pp, k, i, xx);
//     for (auto &gg : xx)
//         cout << gg << " ";
//     cout << ">" << bb << endl;
// }
// cout << x->to_str() << endl;
// NPDMScheme y(x);
// cout << y.to_str() << endl;
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
//     SU2CG cg;
//     uint16_t p = 0, q = 1, r = 2, s = 3;
//     // a = mul(mul('Cp', 'Cq', 2), mul('Dr', 'Ds', 2), 0)
//     // b = mul(mul(mul('Cp', 'Cq', 2), 'Dr', 1), 'Ds', 0)
//     auto a = T::mul(T::mul(T::C(p), T::C(q), 2, cg),
//                     T::mul(T::D(r), T::D(s), 2, cg), 0, cg);
//     auto b = T::mul(T::mul(T::mul(T::C(p), T::C(q), 2, cg), T::D(r), 1,
//     cg),
//                     T::D(s), 0, cg);
//     cout << a.to_str() << endl;
//     cout << b.to_str() << endl;
//     cout << (a == b) << endl;
//     int nn = 4;
//     vector<string> pp = SpinPermRecoupling::initialize(nn, 0);
//     cout << pp.size() << endl;
//     for (auto &xp : pp)
//         cout << xp << " | " << SpinPermRecoupling::find_split_index(xp)
//         << endl;
//     vector<uint8_t> cds;
//     vector<uint16_t> indices;
//     for (int i = 0; i < nn; i++)
//         indices.push_back(i), cds.push_back(i < nn / 2);
//     // vector<T> ts(pp.size());
//     // for (int i = 0; i < (int)pp.size(); i++)
//     //     ts[i] = SpinPermRecoupling::make_tensor(pp[i], indices, cds,
//     cg);
//     // vector<int> selected_pp_idx;
//     // for (int i = 0; i < (int)pp.size(); i++) {
//     //     cout << i << " / " << pp.size() << " " << ts[i].to_str() <<
//     endl;
//     //     bool found = false;
//     //     for (auto j : selected_pp_idx) {
//     //         double x = ts[i].equal_to_scaled(ts[j]);
//     //         if (x != 0)
//     //             found = true;
//     //         // cout << "[" << i << "] = " << x << " * [" << j << "]"
//     << endl;
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
//     //             cout << setw(10) << setprecision(6) << x.data[0] << "
//     * ["
//     <<
//     //             ja
//     //                  << "] ";
//     //             cout << setw(10) << setprecision(6) << x.data[1] << "
//     * ["
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
//         vector<uint16_t> indices(rr.begin() + i, rr.begin() + i +
//         gg.size()); string xpre = "(.+.)0"; for (int inn = 4; inn <= nn;
//         inn += 2)
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
//             SpinPermPattern::get_unique(target_cds, vector<uint16_t>(),
//             nn / 2);
//         vector<T> tts(ttp.size());
//         bool found = false;
//         for (int j = 0; j < tts.size(); j++) {
//             tts[j] = R::make_tensor(ttp[j], gg, target_cds, cg) * 2;
//             double x = xs.equal_to_scaled(tts[j]);
//             if (x != 0)
//                 found = true, cout << " = " << setw(10) <<
//                 setprecision(6)
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
//                 MatrixRef b(ppp.data() + a.size() + x.size(),
//                 pgg[0].size(), 1); for (int k = 0; k <
//                 (int)pgg[0].size(); k++)
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
//                 for (int jc = jb + 1; jc < (int)pgg.size() && !found;
//                 jc++) {
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
//                     cout << setw(10) << setprecision(6) << fixed <<
//                     x.data[0]
//                          << " * [" << ja << "] ";
//                     cout << setw(10) << setprecision(6) << fixed <<
//                     x.data[1]
//                          << " * [" << jb << "] ";
//                     cout << setw(10) << setprecision(6) << fixed <<
//                     x.data[2]
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
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, q, r,
//     s},
//     //                    R::make_cds("CCDD"), cg),
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, q, s,
//     r},
//     //                    R::make_cds("CCDD"), cg),
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, r, q,
//     s},
//     //                    R::make_cds("CDCD"), cg),
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{p, r, s,
//     q},
//     //                    R::make_cds("CDDC"), cg),
//     // };
//     // vector<SpinPermTensor> B = {
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, q, r,
//     s},
//     //                    R::make_cds("CCDD"), cg),
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, q, s,
//     r},
//     //                    R::make_cds("CCDD"), cg) *
//     //         -1,
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, r, q,
//     s},
//     //                    R::make_cds("CDCD"), cg),
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{p, r, s,
//     q},
//     //                    R::make_cds("CDDC"), cg) *
//     //         -1,
//     // };
//     // for (int i = 0; i < A.size(); i++) {
//     //     SpinPermTensor zz = A[i] * (-1) + B[i] * sqrt(3);
//     //     zz = zz.auto_sort();
//     //     cout << (zz == zstd) << " === " << zz.to_str() << endl;
//     // }
//     // SpinPermTensor ZA =
//     //     R::make_tensor("((.+.)0+(.+.)0)0", vector<uint16_t>{0, 1, 2,
//     3},
//     //                    R::make_cds("CDDC"), cg);
//     // SpinPermTensor ZB =
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{0, 1, 2,
//     3},
//     //                    R::make_cds("CDDC"), cg) *
//     //     -1;
//     // SpinPermTensor ZC =
//     //     R::make_tensor("((.+.)2+(.+.)2)0", vector<uint16_t>{0, 1, 2,
//     3},
//     //                    R::make_cds("CCDD"), cg) *
//     //     -1;
//     // cout << "ZA = " << ZA.to_str() << endl;
//     // cout << "ZB = " << ZB.to_str() << endl;
//     // cout << "ZC = " << ZC.to_str() << endl;
//     // SpinPermTensor ZAB = ZA * (-1) + ZB * sqrt(3);
//     // cout << "ZAB = " << ZAB.to_str() << endl;
//     return 0;
// }
