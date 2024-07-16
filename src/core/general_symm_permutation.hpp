
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

/** General symmetry permutation for operators. */

#pragma once

#include "clebsch_gordan.hpp"
#include "threading.hpp"
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
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

enum struct GeneralSymmOperator : uint16_t {
    C = 1,
    D = 2,
    E = 4,
    F = 8,
    G = 16,
    S = 32
};

inline bool operator&(GeneralSymmOperator a, GeneralSymmOperator b) {
    return ((uint16_t)a & (uint16_t)b) != 0;
}

inline GeneralSymmOperator operator^(GeneralSymmOperator a,
                                     GeneralSymmOperator b) {
    return (GeneralSymmOperator)((uint16_t)a ^ (uint16_t)b);
}

inline ostream &operator<<(ostream &os, const GeneralSymmOperator c) {
    const static string repr[] = {"C", "D", "E", "F", "G", "S"};
    for (uint16_t p = 1, r = 0; p <= 256; p <<= 1, r++)
        if ((uint16_t)c == p) {
            os << repr[r];
            break;
        }
    return os;
}

struct GeneralSymmElement {
    GeneralSymmOperator op;
    uint16_t index;
    vector<int16_t> tms;
    GeneralSymmElement(GeneralSymmOperator op, uint16_t index, int16_t tm)
        : op(op), index(index), tms{tm} {}
    GeneralSymmElement(GeneralSymmOperator op, uint16_t index,
                       const vector<int16_t> &tms)
        : op(op), index(index), tms(tms) {}
    bool operator<(const GeneralSymmElement &other) const {
        if (op != other.op)
            return op < other.op;
        if (tms.size() != other.tms.size())
            return tms.size() < other.tms.size();
        for (size_t i = 0; i < tms.size(); i++)
            if (tms[i] != other.tms[i])
                return tms[i] < other.tms[i];
        if (index != other.index)
            return index < other.index;
        return false;
    }
    bool operator==(const GeneralSymmElement &other) const {
        if (op != other.op)
            return false;
        if (tms.size() != other.tms.size())
            return false;
        for (size_t i = 0; i < tms.size(); i++)
            if (tms[i] != other.tms[i])
                return false;
        if (index != other.index)
            return false;
        return true;
    }
    bool operator!=(const GeneralSymmElement &other) const {
        return !(*this == other);
    }
    size_t hash() const {
        size_t h = (size_t)tms.size();
        h ^= (uint16_t)op + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= (uint16_t)index + 0x9E3779B9 + (h << 6) + (h >> 2);
        for (int8_t i = 0; i < tms.size(); i++)
            h ^= (int16_t)tms[i] + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    string to_str() const {
        stringstream ss;
        ss << op << index << "<";
        for (size_t i = 0; i < tms.size(); i++)
            ss << tms[i] << (i == tms.size() - 1 ? ">" : ",");
        return ss.str();
    }
};

template <typename FL> struct GeneralSymmTerm {
    typedef decltype(abs((FL)0.0)) FP;
    FL factor;
    vector<GeneralSymmElement> ops;
    GeneralSymmTerm() : factor((FL)0.0) {}
    GeneralSymmTerm(GeneralSymmElement elem, FL factor = (FL)1.0)
        : factor(factor), ops{elem} {}
    GeneralSymmTerm(const vector<GeneralSymmElement> &ops, FL factor = (FL)1.0)
        : factor(factor), ops(ops) {}
    GeneralSymmTerm operator-() const { return GeneralSymmTerm(ops, -factor); }
    GeneralSymmTerm operator*(FL d) const {
        return GeneralSymmTerm(ops, d * factor);
    }
    bool operator<(const GeneralSymmTerm &other) const {
        if (ops.size() != other.ops.size())
            return ops.size() < other.ops.size();
        for (size_t i = 0; i < ops.size(); i++)
            if (ops[i] != other.ops[i])
                return ops[i] < other.ops[i];
        if (abs(factor - other.factor) >= (FP)1E-12)
            return abs(factor) < abs(other.factor);
        return false;
    }
    bool ops_equal_to(const GeneralSymmTerm &other) const {
        if (ops.size() != other.ops.size())
            return false;
        for (size_t i = 0; i < ops.size(); i++)
            if (ops[i] != other.ops[i])
                return false;
        return true;
    }
    bool operator==(const GeneralSymmTerm &other) const {
        return ops_equal_to(other) && abs(factor - other.factor) < (FP)1E-12;
    }
    bool operator!=(const GeneralSymmTerm &other) const {
        return !ops_equal_to(other) || abs(factor - other.factor) >= (FP)1E-12;
    }
    string to_str() const {
        stringstream ss;
        if (factor != (FL)1.0)
            ss << factor << " ";
        for (auto &op : ops)
            ss << op.to_str() << " ";
        return ss.str();
    }
};

template <typename FL> struct GeneralSymmTensor {
    typedef decltype(abs((FL)0.0)) FP;
    // outer vector is projected symm components
    // inner vector is sum of terms
    vector<vector<GeneralSymmTerm<FL>>> data;
    vector<int16_t> tjs;
    GeneralSymmTensor() : data{vector<GeneralSymmTerm<FL>>()} {}
    GeneralSymmTensor(const vector<vector<GeneralSymmTerm<FL>>> &data)
        : data(data), tjs{(int16_t)(data.size() - 1)} {}
    GeneralSymmTensor(const vector<vector<GeneralSymmTerm<FL>>> &data,
                      const vector<int16_t> &tjs)
        : data(data), tjs(tjs) {}
    static GeneralSymmTensor i() {
        return GeneralSymmTensor(
            vector<vector<GeneralSymmTerm<FL>>>{vector<GeneralSymmTerm<FL>>{
                GeneralSymmTerm<FL>(vector<GeneralSymmElement>())}});
    }
    static GeneralSymmTensor c(uint16_t index) {
        vector<GeneralSymmTerm<FL>> a = {GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::C, index, -1))};
        vector<GeneralSymmTerm<FL>> b = {GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::C, index, 1))};
        return GeneralSymmTensor(vector<vector<GeneralSymmTerm<FL>>>{a, b});
    }
    static GeneralSymmTensor d(uint16_t index) {
        vector<GeneralSymmTerm<FL>> a = {GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::D, index, 1))};
        vector<GeneralSymmTerm<FL>> b = {-GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::D, index, -1))};
        return GeneralSymmTensor(vector<vector<GeneralSymmTerm<FL>>>{a, b});
    }
    static GeneralSymmTensor t(uint16_t index) {
        vector<GeneralSymmTerm<FL>> sp = {-GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::S, index, 2))};
        vector<GeneralSymmTerm<FL>> sz = {GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::S, index, 0),
            (FL)sqrt(2.0))};
        vector<GeneralSymmTerm<FL>> sm = {GeneralSymmTerm<FL>(
            GeneralSymmElement(GeneralSymmOperator::S, index, -2))};
        return GeneralSymmTensor(
            vector<vector<GeneralSymmTerm<FL>>>{sp, sz, sm});
    }
    static GeneralSymmTensor c_angular(int16_t twol, uint16_t index) {
        vector<vector<GeneralSymmTerm<FL>>> r;
        r.reserve(twol + 1);
        for (int16_t twom = -twol; twom <= twol; twom += 2)
            r.push_back(vector<GeneralSymmTerm<FL>>{GeneralSymmTerm<FL>(
                GeneralSymmElement(GeneralSymmOperator::C, index, twom))});
        return GeneralSymmTensor(r);
    }
    static GeneralSymmTensor d_angular(int16_t twol, uint16_t index) {
        vector<vector<GeneralSymmTerm<FL>>> r;
        r.reserve(twol + 1);
        for (int16_t twom = -twol; twom <= twol; twom += 2)
            r.push_back(vector<GeneralSymmTerm<FL>>{GeneralSymmTerm<FL>(
                GeneralSymmElement(GeneralSymmOperator::D, index, twom))});
        return GeneralSymmTensor(r);
    }
    GeneralSymmTensor simplify() const {
        vector<vector<GeneralSymmTerm<FL>>> zd = data;
        for (auto &jz : zd) {
            sort(jz.begin(), jz.end());
            int j = 0;
            for (int i = 0; i < (int)jz.size(); i++)
                if (j == 0 || !jz[j - 1].ops_equal_to(jz[i]))
                    jz[j == 0 || abs(jz[j - 1].factor) >= (FP)1E-12 ? j++
                                                                    : j - 1] =
                        jz[i];
                else
                    jz[j - 1].factor += jz[i].factor;
            if (j != 0 && abs(jz[j - 1].factor) < (FP)1E-12)
                j--;
            jz.resize(j);
        }
        return GeneralSymmTensor(zd, tjs);
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
    GeneralSymmTensor auto_sort() const {
        GeneralSymmTensor r = *this;
        vector<uint16_t> perm, pcnt;
        for (auto &jx : r.data)
            for (auto &tx : jx) {
                vector<GeneralSymmElement> new_ops = tx.ops;
                perm.resize(tx.ops.size());
                pcnt.resize(tx.ops.size() + 1);
                for (uint16_t i = 0; i < tx.ops.size() + 1; i++)
                    pcnt[i] = 0;
                for (uint16_t i = 0; i < tx.ops.size(); i++)
                    pcnt[tx.ops[i].index + 1]++;
                for (uint16_t i = 0; i < tx.ops.size(); i++)
                    pcnt[i + 1] += pcnt[i];
                for (uint16_t i = 0; i < tx.ops.size(); i++)
                    new_ops[pcnt[tx.ops[i].index]] = tx.ops[i],
                    perm[i] = pcnt[tx.ops[i].index]++;
                if (tx.ops.size() != 0 &&
                    !(tx.ops[0].op & GeneralSymmOperator::S))
                    tx.factor *= permutation_parity(perm) ? -1 : 1;
                tx.ops = new_ops;
            }
        return r;
    }
    GeneralSymmTensor normal_sort() const {
        GeneralSymmTensor r = this->auto_sort();
        bool found = true;
        while (found) {
            found = false;
            for (auto &jx : r.data) {
                for (auto &tx : jx) {
                    for (size_t i = 1; i < tx.ops.size(); i++)
                        if (tx.ops[i].index == tx.ops[i - 1].index &&
                            tx.ops[i].op == GeneralSymmOperator::C &&
                            tx.ops[i - 1].op == GeneralSymmOperator::D) {
                            vector<GeneralSymmElement> ex_ops;
                            bool has_ex = true;
                            if (tx.ops[i].tms.size() !=
                                tx.ops[i - 1].tms.size())
                                has_ex = false;
                            for (size_t j = 0;
                                 has_ex && j < tx.ops[i].tms.size(); j++)
                                has_ex = has_ex && (tx.ops[i].tms[j] ==
                                                    tx.ops[i - 1].tms[j]);
                            if (has_ex) {
                                for (size_t j = 0; j < tx.ops.size(); j++)
                                    if (j != i && j != i - 1)
                                        ex_ops.push_back(tx.ops[j]);
                            }
                            auto tmp = tx.ops[i - 1];
                            tx.ops[i - 1] = tx.ops[i];
                            tx.ops[i] = tmp;
                            tx.factor = -tx.factor;
                            found = true;
                            if (has_ex)
                                jx.push_back(
                                    GeneralSymmTerm<FL>(ex_ops, -tx.factor));
                            break;
                        }
                    if (found)
                        break;
                }
                if (found)
                    break;
            }
        }
        return r.simplify();
    }
    vector<uint8_t> get_cds() const {
        if (data.size() == 0 || data[0].size() == 0)
            return vector<uint8_t>();
        vector<uint8_t> r(data[0][0].ops.size(), 0);
        for (int j = 0; j < (int)r.size(); j++)
            r[j] = (uint8_t)(((uint8_t)data[0][0].ops[j].op &
                              (uint8_t)GeneralSymmOperator::S) |
                             ((uint8_t)data[0][0].ops[j].op &
                              (uint8_t)GeneralSymmOperator::C));
        return r;
    }
    GeneralSymmTensor operator*(FL d) const {
        GeneralSymmTensor r = *this;
        for (auto &jx : r.data)
            for (auto &tx : jx)
                tx.factor *= d;
        return r;
    }
    GeneralSymmTensor operator-(const GeneralSymmTensor &other) const {
        return *this + other * (FL)-1.0;
    }
    GeneralSymmTensor operator+(const GeneralSymmTensor &other) const {
        if (data.size() == 1 && data[0].size() == 0)
            return other;
        else if (other.data.size() == 1 && other.data[0].size() == 0)
            return *this;
        assert(tjs == other.tjs);
        vector<vector<GeneralSymmTerm<FL>>> zd = data;
        for (size_t i = 0; i < zd.size(); i++)
            zd[i].insert(zd[i].end(), other.data[i].begin(),
                         other.data[i].end());
        return GeneralSymmTensor(zd, tjs).simplify();
    }
    bool operator==(const GeneralSymmTensor &other) const {
        GeneralSymmTensor a = simplify(), b = other.simplify();
        if (a.tjs != b.tjs || a.data.size() != b.data.size())
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
    FL equal_to_scaled(const GeneralSymmTensor &other) const {
        GeneralSymmTensor a = simplify(), b = other.simplify();
        FL fac = (FL)0.0;
        if (a.tjs != b.tjs || a.data.size() != b.data.size())
            return (FL)0.0;
        for (int i = 0; i < (int)a.data.size(); i++) {
            if (a.data[i].size() != b.data[i].size())
                return (FL)0.0;
            for (int j = 0; j < a.data[i].size(); j++)
                if (!a.data[i][j].ops_equal_to(b.data[i][j]))
                    return (FL)0.0;
                else if (fac == (FL)0.0)
                    fac = a.data[i][j].factor / b.data[i][j].factor;
                else if (abs(a.data[i][j].factor / b.data[i][j].factor - fac) >=
                         (FP)1E-12)
                    return (FL)0.0;
        }
        return fac;
    }
    static GeneralSymmTensor mul(const GeneralSymmTensor &x,
                                 const GeneralSymmTensor &y,
                                 const vector<int16_t> &tjzs,
                                 const vector<shared_ptr<AnyCG<FL>>> &cgs) {
        int16_t mt = 1;
        for (int16_t tjz : tjzs)
            mt = mt * (tjz + 1);
        vector<vector<GeneralSymmTerm<FL>>> z(mt);
        // ix iy iz mx my mz
        vector<pair<array<int16_t, 3>, FL>> mxyzs{
            make_pair(array<int16_t, 3>{0, 0, 0}, (FL)1.0)};
        for (int16_t ik = 0; ik < (int16_t)tjzs.size(); ik++) {
            vector<pair<array<int16_t, 3>, FL>> nxyzs;
            for (int im = 0; im < (int)mxyzs.size(); im++) {
                int16_t tjx = x.tjs[ik], tjy = y.tjs[ik], tjz = tjzs[ik];
                for (int16_t iz = 0, mz = -tjz; mz <= tjz; mz += 2, iz++)
                    for (int16_t ix = 0, mx = -tjx; mx <= tjx; mx += 2, ix++)
                        for (int16_t iy = 0, my = -tjy; my <= tjy;
                             my += 2, iy++) {
                            FL factor = cgs[ik]->cg(tjx, tjy, tjz, mx, my, mz);
                            if (abs(factor) >= (FP)1E-12)
                                nxyzs.push_back(make_pair(
                                    array<int16_t, 3>{
                                        (int16_t)(mxyzs[im].first[0] *
                                                      (tjx + 1) +
                                                  ix),
                                        (int16_t)(mxyzs[im].first[1] *
                                                      (tjy + 1) +
                                                  iy),
                                        (int16_t)(mxyzs[im].first[2] *
                                                      (tjz + 1) +
                                                  iz)},
                                    mxyzs[im].second * factor));
                        }
            }
            mxyzs = nxyzs;
        }
        for (const auto &mxyz : mxyzs) {
            int16_t ix = mxyz.first[0], iy = mxyz.first[1], iz = mxyz.first[2];
            for (auto &tx : x.data[ix])
                for (auto &ty : y.data[iy]) {
                    FL factor = tx.factor * ty.factor * mxyz.second;
                    if (abs(factor) < (FP)1E-12)
                        continue;
                    vector<GeneralSymmElement> ops = tx.ops;
                    ops.reserve(tx.ops.size() + ty.ops.size());
                    ops.insert(ops.end(), ty.ops.begin(), ty.ops.end());
                    z[iz].push_back(GeneralSymmTerm<FL>(ops, factor));
                }
        }
        return GeneralSymmTensor(z, tjzs).simplify();
    }
    // ((.+.)[0]+.)[0]
    //  ^        ^ ^
    struct Level {
        int8_t left_idx, mid_idx, right_idx, left_cnt, right_cnt;
    };
    static Level get_level(const string &x, int8_t i_start) {
        Level r;
        if (x[i_start] != '(') {
            r.left_idx = i_start;
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
            else if (c == '.' || (c >= 'A' && c <= 'Z') ||
                     (c >= 'a' && c <= 'z'))
                dot_cnt++;
        }
        r.right_cnt = dot_cnt - r.left_cnt;
        return r;
    }
    static vector<int16_t> get_quanta(const string &expr, const Level &l) {
        vector<int16_t> tjs;
        int p = l.right_idx == -1 ? l.left_idx + 1 : l.right_idx;
        if (p >= (int)expr.length())
            return vector<int16_t>();
        if (expr[p] != '[')
            tjs.push_back(0);
        for (int i = p; i < (int)expr.length(); i++)
            if (expr[i] == '[' || expr[i] == ',')
                tjs.push_back(0);
            else if (expr[i] >= '0' && expr[i] <= '9')
                tjs.back() = tjs.back() * 10 + (int)(expr[i] - '0');
            else if (expr[i] == '?')
                tjs.back() = -1;
            else
                break;
        return tjs;
    }
    static int count_cds(const string &x) {
        int ncd = 0;
        for (auto &c : x)
            if (!(c == '.' || c == '(' || c == ')' || c == '+' ||
                  (c >= '0' && c <= '9') || c == '[' || c == ']' || c == ','))
                ncd++;
        return ncd;
    }
    static map<vector<int>, string>
    parse_expr_angular_expr(const string &expr, const vector<int16_t> &idxs,
                            int ii = 0) {
        Level l = get_level(expr, 0);
        if (l.right_idx == -1) {
            vector<int16_t> xjs = get_quanta(expr, l);
            stringstream ss;
            ss << expr[0] << (xjs.size() == 0 ? idxs[ii] * 2 : xjs[0]);
            return map<vector<int>, string>{make_pair(vector<int>(), ss.str())};
        } else {
            string lexpr = expr.substr(l.left_idx, l.mid_idx - 1 - l.left_idx);
            string rexpr = expr.substr(l.mid_idx, l.right_idx - 1 - l.mid_idx);
            map<vector<int>, string> lm =
                parse_expr_angular_expr(lexpr, idxs, ii);
            map<vector<int>, string> rm =
                parse_expr_angular_expr(rexpr, idxs, ii + l.left_cnt);
            Level gl = get_level(lexpr, 0);
            Level gr = get_level(rexpr, 0);
            vector<int16_t> tjs = get_quanta(expr, l);
            vector<int16_t> ltjs = get_quanta(lexpr, gl);
            vector<int16_t> rtjs = get_quanta(rexpr, gr);
            vector<int> mml(ltjs.size() + 1, 0), mmr(rtjs.size() + 1, 0);
            int ixx, ix;
            for (ix = 0, ixx = 0; ix < (int)ltjs.size(); ix++)
                if (ltjs[ix] == -1)
                    mml[ix] = ixx++;
            mml[ltjs.size()] = ixx;
            for (ix = 0, ixx = 0; ix < (int)rtjs.size(); ix++)
                if (rtjs[ix] == -1)
                    mmr[ix] = ixx++;
            mmr[rtjs.size()] = ixx;
            map<vector<int>, string> r;
            for (const auto &xl : lm) {
                vector<int16_t> xltjs =
                    get_quanta(xl.second, get_level(xl.second, 0));
                for (const auto &xr : rm) {
                    vector<int16_t> xrtjs =
                        get_quanta(xr.second, get_level(xr.second, 0));
                    vector<pair<int16_t, int16_t>> mjs_sz;
                    size_t tot_sz = 1;
                    for (size_t it = 0; it < tjs.size(); it++)
                        if (tjs[it] == -1) {
                            int16_t ixl = xltjs[it], ixr = xrtjs[it];
                            mjs_sz.push_back(
                                make_pair(abs(ixl - ixr), ixl + ixr));
                            tot_sz *= (ixl + ixr - abs(ixl - ixr)) / 2 + 1;
                        }
                    for (size_t ix = 0, im, ixx; ix < tot_sz; ix++) {
                        ixx = ix;
                        vector<int16_t> xtjs = tjs;
                        vector<int> rkl = xl.first, rkr = xr.first, rk;
                        bool okay = true;
                        im = 0;
                        for (size_t it = 0; it < xtjs.size(); it++)
                            if (xtjs[it] != -1) {
                                int16_t ixl = xltjs[it], ixr = xrtjs[it];
                                if (!(xtjs[it] >= abs(ixl - ixr) &&
                                      xtjs[it] <= ixl + ixr &&
                                      ((xtjs[it] + ixl + ixr) & 1) == 0)) {
                                    okay = false;
                                    break;
                                }
                                if (xtjs[it] == 0) {
                                    if (rtjs.size() != 0 && rtjs[it] == -1)
                                        rkr[rkr.size() - mmr.back() + mmr[it]] =
                                            -1;
                                    else if (ltjs.size() != 0 && ltjs[it] == -1)
                                        rkl[rkl.size() - mml.back() + mml[it]] =
                                            -1;
                                }
                            } else {
                                int16_t dm =
                                    (mjs_sz[im].second - mjs_sz[im].first) / 2 +
                                    1;
                                xtjs[it] = ixx % dm * 2 + mjs_sz[im].first;
                                rk.push_back(xtjs[it]);
                                ixx /= dm;
                                im++;
                            }
                        if (okay) {
                            vector<int> rkk;
                            for (int x : rkl)
                                if (x != -1)
                                    rkk.push_back(x);
                            for (int x : rkr)
                                if (x != -1)
                                    rkk.push_back(x);
                            rkk.insert(rkk.end(), rk.begin(), rk.end());
                            stringstream ss;
                            ss << "(" << xl.second << "+" << xr.second << ")";
                            if (xtjs.size() == 1)
                                ss << xtjs[0];
                            else {
                                ss << "[";
                                for (size_t p = 0; p < xtjs.size(); p++)
                                    ss << xtjs[p]
                                       << (p == xtjs.size() - 1 ? "]" : ",");
                            }
                            r[rkk] = ss.str();
                        }
                    }
                }
            }
            return r;
        }
    }
    static map<vector<int>, GeneralSymmTensor>
    parse_expr_angular(const string &expr, const vector<int16_t> &idxs,
                       const vector<shared_ptr<AnyCG<FL>>> &cgs, int ii = 0) {
        Level l = get_level(expr, 0);
        if (l.right_idx == -1) {
            vector<int16_t> xjs = get_quanta(expr, l);
            if (expr[0] == 'C')
                return map<vector<int>, GeneralSymmTensor>{
                    make_pair(vector<int>(),
                              c_angular(xjs.size() == 0 ? idxs[ii] * 2 : xjs[0],
                                        idxs[ii]))};
            else if (expr[0] == 'D')
                return map<vector<int>, GeneralSymmTensor>{
                    make_pair(vector<int>(),
                              d_angular(xjs.size() == 0 ? idxs[ii] * 2 : xjs[0],
                                        idxs[ii]))};
            else {
                assert(false);
                return map<vector<int>, GeneralSymmTensor>();
            }
        } else {
            string lexpr = expr.substr(l.left_idx, l.mid_idx - 1 - l.left_idx);
            string rexpr = expr.substr(l.mid_idx, l.right_idx - 1 - l.mid_idx);
            map<vector<int>, GeneralSymmTensor> lm =
                parse_expr_angular(lexpr, idxs, cgs, ii);
            map<vector<int>, GeneralSymmTensor> rm =
                parse_expr_angular(rexpr, idxs, cgs, ii + l.left_cnt);
            Level gl = get_level(lexpr, 0);
            Level gr = get_level(rexpr, 0);
            vector<int16_t> tjs = get_quanta(expr, l);
            vector<int16_t> ltjs = get_quanta(lexpr, gl);
            vector<int16_t> rtjs = get_quanta(rexpr, gr);
            vector<int> mml(ltjs.size() + 1, 0), mmr(rtjs.size() + 1, 0);
            int ixx, ix;
            for (ix = 0, ixx = 0; ix < (int)ltjs.size(); ix++)
                if (ltjs[ix] == -1)
                    mml[ix] = ixx++;
            mml[ltjs.size()] = ixx;
            for (ix = 0, ixx = 0; ix < (int)rtjs.size(); ix++)
                if (rtjs[ix] == -1)
                    mmr[ix] = ixx++;
            mmr[rtjs.size()] = ixx;
            map<vector<int>, GeneralSymmTensor> r;
            for (const auto &xl : lm)
                for (const auto &xr : rm) {
                    vector<pair<int16_t, int16_t>> mjs_sz;
                    size_t tot_sz = 1;
                    for (size_t it = 0; it < tjs.size(); it++)
                        if (tjs[it] == -1) {
                            int16_t ixl = xl.second.tjs[it],
                                    ixr = xr.second.tjs[it];
                            mjs_sz.push_back(
                                make_pair(abs(ixl - ixr), ixl + ixr));
                            tot_sz *= (ixl + ixr - abs(ixl - ixr)) / 2 + 1;
                        }
                    for (size_t ix = 0, im, ixx; ix < tot_sz; ix++) {
                        ixx = ix;
                        vector<int16_t> xtjs = tjs;
                        vector<int> rkl = xl.first, rkr = xr.first, rk;
                        bool okay = true;
                        im = 0;
                        for (size_t it = 0; it < xtjs.size(); it++)
                            if (xtjs[it] != -1) {
                                int16_t ixl = xl.second.tjs[it],
                                        ixr = xr.second.tjs[it];
                                if (!(xtjs[it] >= abs(ixl - ixr) &&
                                      xtjs[it] <= ixl + ixr &&
                                      ((xtjs[it] + ixl + ixr) & 1) == 0)) {
                                    okay = false;
                                    break;
                                }
                                if (xtjs[it] == 0) {
                                    if (rtjs.size() != 0 && rtjs[it] == -1)
                                        rkr[rkr.size() - mmr.back() + mmr[it]] =
                                            -1;
                                    else if (ltjs.size() != 0 && ltjs[it] == -1)
                                        rkl[rkl.size() - mml.back() + mml[it]] =
                                            -1;
                                }
                            } else {
                                int16_t dm =
                                    (mjs_sz[im].second - mjs_sz[im].first) / 2 +
                                    1;
                                xtjs[it] = ixx % dm * 2 + mjs_sz[im].first;
                                rk.push_back(xtjs[it]);
                                ixx /= dm;
                                im++;
                            }
                        if (okay) {
                            vector<int> rkk;
                            for (int x : rkl)
                                if (x != -1)
                                    rkk.push_back(x);
                            for (int x : rkr)
                                if (x != -1)
                                    rkk.push_back(x);
                            rkk.insert(rkk.end(), rk.begin(), rk.end());
                            r[rkk] = mul(xl.second, xr.second, xtjs, cgs);
                        }
                    }
                }
            return r;
        }
    }
    string to_str() const {
        stringstream ss;
        ss << "[";
        for (auto tj : tjs)
            ss << " " << tj;
        ss << " ] ";
        if (data.size() > 1)
            ss << endl;
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

template <typename FL> struct GeneralSymmExpr {
    typedef decltype(abs((FL)0.0)) FP;
    int n_sites;
    int n_ops;
    string expr;
    int n_reduced_sites;
    int max_l;
    vector<string> orb_sym;
    vector<int> site_sym;
    vector<pair<vector<GeneralSymmElement>, vector<pair<vector<int>, FL>>>>
        data;
    static const string &orb_names() {
        const static string _orb_names = "spdfghiklmnoqrtuvwxyz";
        return _orb_names;
    }
    struct vector_elem_hasher {
        size_t operator()(const vector<GeneralSymmElement> &x) const {
            size_t r = x.size();
            for (auto &i : x)
                r ^= i.hash() + 0x9e3779b9 + (r << 6) + (r >> 2);
            return r;
        }
    };
    // expr includes ?. when construct, generate multiple dynamicly
    GeneralSymmExpr(const vector<string> &orb_sym, const string &expr)
        : expr(expr), orb_sym(orb_sym) {
        using T = GeneralSymmTensor<FL>;
        n_sites = (int)orb_sym.size();
        set<int> dist_site;
        max_l = 0;
        for (auto &ir : orb_sym) {
            size_t l = orb_names().find_first_of(ir[0]);
            assert(l != string::npos);
            dist_site.insert((int)l);
            max_l = max((int)l, max_l);
            if (ir.substr(1) == "+0")
                site_sym.push_back((int)l);
        }
        n_reduced_sites = (int)site_sym.size();
        assert(dist_site.size() != 0);
        vector<shared_ptr<AnyCG<FL>>> cgs(1, make_shared<AnySO3RSHCG<FL>>());
        unordered_map<vector<GeneralSymmElement>, vector<pair<vector<int>, FL>>,
                      vector_elem_hasher>
            mp;
        n_ops = 0;
        size_t ng = 1;
        for (auto &c : expr)
            if (c >= 'A' && c <= 'Z')
                n_ops++, ng *= dist_site.size();
        vector<int> vdist(dist_site.begin(), dist_site.end());
        for (size_t ig = 0, igv; ig < ng; ig++) {
            vector<int16_t> ls;
            igv = ig;
            for (int il = 0; il < n_ops; il++)
                ls.push_back((int16_t)vdist[igv % vdist.size()]),
                    igv /= vdist.size();
            auto pex = T::parse_expr_angular(expr, ls, cgs);
            for (const auto &mex : pex) {
                const T &ex = mex.second;
                if (ex.data[0].size() == 0)
                    continue;
                for (auto &k : ex.data[0])
                    mp[k.ops].push_back(make_pair(mex.first, k.factor));
            }
        }
        data = vector<
            pair<vector<GeneralSymmElement>, vector<pair<vector<int>, FL>>>>(
            mp.begin(), mp.end());
        sort(data.begin(), data.end(),
             [](const pair<vector<GeneralSymmElement>,
                           vector<pair<vector<int>, FL>>> &i,
                const pair<vector<GeneralSymmElement>,
                           vector<pair<vector<int>, FL>>> &j) {
                 if (i.second.size() != j.second.size())
                     return i.second.size() < j.second.size();
                 else {
                     for (int im = (int)i.second.size() - 1; im >= 0; im--) {
                         if (i.second[im].first.size() !=
                             j.second[im].first.size())
                             return i.second[im].first.size() <
                                    j.second[im].first.size();
                         for (int jm = 0; jm < (int)i.second[im].first.size();
                              jm++)
                             if (i.second[im].first[jm] !=
                                 j.second[im].first[jm])
                                 return i.second[im].first[jm] >
                                        j.second[im].first[jm];
                     }
                     return i.first < j.first;
                 }
             });
    }
    void reduce(const FL *int_data, FL *reduced_data, FP cutoff = (FP)1E-12) {
        map<string, vector<int>> orb_idx_mp;
        map<char, vector<int>> site_idx_mp;
        for (size_t ix = 0; ix < orb_sym.size(); ix++)
            orb_idx_mp[orb_sym[ix]].push_back((int)ix);
        for (size_t ix = 0; ix < site_sym.size(); ix++)
            site_idx_mp[orb_names()[site_sym[ix]]].push_back((int)ix);
        size_t ml_stride = 1, ml_size = 1;
        int ml_len = data.size() == 0 ? 0 : (int)data[0].second[0].first.size();
        for (int i = 0; i < n_ops; i++)
            ml_stride *= (size_t)n_reduced_sites;
        for (int i = 0; i < ml_len; i++)
            ml_size *= (size_t)(max_l + max_l + 1);
        vector<uint8_t> solved(ml_stride * ml_size, 0);
        memset(reduced_data, 0, solved.size() * sizeof(FL));
        for (auto &mx : data) {
            vector<string> kg;
            vector<int> n_orbs;
            size_t np = 1;
            for (int p = 0; p < n_ops; p++) {
                stringstream ss;
                ss << orb_names()[mx.first[p].index];
                ss << (mx.first[p].tms[0] >= 0 ? "+" : "");
                ss << mx.first[p].tms[0] / 2;
                kg.push_back(ss.str());
                n_orbs.push_back((int)orb_idx_mp.at(kg.back()).size());
                np *= n_orbs.back();
            }
            for (size_t ip = 0; ip < np; ip++) {
                size_t ipv = ip, ipx = 0, ipz = 0;
                for (auto &g : kg) {
                    ipx = ipx * n_sites +
                          orb_idx_mp.at(g)[ipv % orb_idx_mp.at(g).size()];
                    ipz = ipz * n_reduced_sites +
                          site_idx_mp.at(g[0])[ipv % orb_idx_mp.at(g).size()];
                    ipv = ipv / orb_idx_mp.at(g).size();
                }
                FL f = int_data[ipx];
                for (int iv = (int)mx.second.size() - 1; iv >= 0; iv--) {
                    size_t ipg = 0;
                    for (auto &mf : mx.second[iv].first)
                        ipg = ipg * (max_l + max_l + 1) + mf / 2;
                    ipg *= ml_stride;
                    if (!solved[ipg + ipz]) {
                        reduced_data[ipg + ipz] = f / mx.second[iv].second;
                        solved[ipg + ipz] = 1;
                    }
                    f -= reduced_data[ipg + ipz] * mx.second[iv].second;
                }
                assert(abs(f) < cutoff);
            }
        }
    }
    void reduce_expr(const FL *reduced_data, vector<string> &exprs,
                     vector<vector<uint16_t>> &indices,
                     vector<vector<FL>> &idata, FP cutoff = (FP)1E-12) {
        using T = GeneralSymmTensor<FL>;
        map<char, vector<int>> site_idx_mp;
        for (size_t ix = 0; ix < site_sym.size(); ix++)
            site_idx_mp[orb_names()[site_sym[ix]]].push_back((int)ix);
        size_t ml_stride = 1, ml_size = 1;
        int ml_len = data.size() == 0 ? 0 : (int)data[0].second[0].first.size();
        for (int i = 0; i < n_ops; i++)
            ml_stride *= (size_t)n_reduced_sites;
        for (int i = 0; i < ml_len; i++)
            ml_size *= (size_t)(max_l + max_l + 1);
        vector<int16_t> ls(n_ops, 0);
        vector<uint16_t> idx(n_ops, 0);
        map<string, size_t> mp;
        for (size_t it = 0; it < ml_stride; it++) {
            size_t itv = it;
            for (int il = 0; il < n_ops; il++) {
                ls[n_ops - 1 - il] = (int16_t)site_sym[itv % n_reduced_sites];
                idx[n_ops - 1 - il] = (uint16_t)(itv % n_reduced_sites);
                itv /= n_reduced_sites;
            }
            auto pex = T::parse_expr_angular_expr(expr, ls);
            for (const auto &mex : pex) {
                size_t ipg = 0;
                for (auto &mf : mex.first)
                    ipg = ipg * (max_l + max_l + 1) + mf / 2;
                ipg *= ml_stride;
                if (abs(reduced_data[ipg + it]) <= cutoff)
                    continue;
                size_t ik =
                    mp.count(mex.second) ? mp.at(mex.second) : exprs.size();
                if (ik == exprs.size()) {
                    mp[mex.second] = exprs.size();
                    exprs.push_back(mex.second);
                    indices.push_back(vector<uint16_t>());
                    idata.push_back(vector<FL>());
                }
                indices[ik].insert(indices[ik].end(), idx.begin(), idx.end());
                idata[ik].push_back(reduced_data[ipg + it]);
            }
        }
    }
    string to_str() const {
        stringstream ss;
        ss << "N-SITES = " << n_sites << " -> " << n_reduced_sites
           << " MAX-L = " << max_l << " PATTERN = " << expr
           << " N-OPS = " << n_ops << endl;
        ss << "ORB-SYM = [";
        for (auto ir : orb_sym)
            ss << " " << ir;
        ss << " ]" << endl;
        ss << "RED-SYM = [";
        for (auto ir : site_sym)
            ss << " " << orb_names()[ir];
        ss << " ]" << endl;
        ss << "DATA = " << data.size() << endl;
        return ss.str();
    }
};

template <typename FL> struct GeneralSymmRecoupling {
    typedef decltype(abs((FL)0.0)) FP;
    using T = GeneralSymmTensor<FL>;
    static map<string, FL> recouple(const map<string, FL> &x, int8_t i_start,
                                    int8_t left_cnt,
                                    const vector<shared_ptr<AnyCG<FL>>> &cgs) {
        const string &x0 = x.cbegin()->first;
        typename T::Level h = T::get_level(x0, i_start);
        if (left_cnt == h.left_cnt)
            return x;
        typename T::Level hl = T::get_level(x0, h.left_idx);
        typename T::Level hr = T::get_level(x0, h.mid_idx);
        map<string, FL> v;
        if (left_cnt > h.left_cnt) {
            if (hr.left_cnt != left_cnt - h.left_cnt)
                return recouple(
                    recouple(x, h.mid_idx, left_cnt - h.left_cnt, cgs), i_start,
                    left_cnt, cgs);
            typename T::Level hrl = T::get_level(x0, hr.left_idx);
            typename T::Level hrr = T::get_level(x0, hr.mid_idx);
            for (auto &xm : x) {
                const string &xx = xm.first;
                // 1+(2+3) -> (1+2)+3
                vector<int16_t> j1s = T::get_quanta(xx, hl),
                                j23s = T::get_quanta(xx, hr),
                                j2s = T::get_quanta(xx, hrl),
                                j3s = T::get_quanta(xx, hrr),
                                js = T::get_quanta(xx, h);
                assert(j1s.size() == j23s.size() && j1s.size() == j2s.size() &&
                       j1s.size() == j3s.size() && j1s.size() == js.size());
                vector<pair<vector<int16_t>, FL>> mcgs{
                    make_pair(vector<int16_t>(), xm.second)};
                for (int16_t ik = 0; ik < (int16_t)j1s.size(); ik++) {
                    vector<pair<vector<int16_t>, FL>> ncgs;
                    for (int im = 0; im < (int)mcgs.size(); im++) {
                        int16_t j1 = j1s[ik], j23 = j23s[ik], j2 = j2s[ik],
                                j3 = j3s[ik], j = js[ik];
                        for (int16_t j12 = abs(j1 - j2); j12 <= j1 + j2;
                             j12 += 2) {
                            FL f = (FL)cgs[ik]->racah(j1, j2, j, j3, j12, j23) *
                                   (FL)(FP)sqrt((j12 + 1) * (j23 + 1));
                            if (abs(f) < 1E-12)
                                continue;
                            vector<int16_t> k = mcgs[im].first;
                            k.push_back(j12);
                            ncgs.push_back(make_pair(k, mcgs[im].second * f));
                        }
                    }
                    mcgs = ncgs;
                }
                for (int im = 0; im < (int)mcgs.size(); im++) {
                    stringstream ss;
                    ss << xx.substr(0, h.left_idx) << "("
                       << xx.substr(h.left_idx, h.mid_idx - 1 - h.left_idx)
                       << "+"
                       << xx.substr(hr.left_idx, hr.mid_idx - 1 - hr.left_idx)
                       << ")";
                    if (mcgs[im].first.size() > 1)
                        ss << "[";
                    for (int ik = 0; ik < (int)mcgs[im].first.size(); ik++)
                        ss << (int)mcgs[im].first[ik]
                           << (ik == (int)mcgs[im].first.size() - 1 ? "" : ",");
                    if (mcgs[im].first.size() > 1)
                        ss << "]";
                    ss << "+"
                       << xx.substr(hr.mid_idx, hr.right_idx - 1 - hr.mid_idx)
                       << xx.substr(h.right_idx - 1);
                    v[ss.str()] += mcgs[im].second;
                }
            }
        } else {
            if (hl.right_cnt != h.left_cnt - left_cnt)
                return recouple(recouple(x, h.left_idx, left_cnt, cgs), i_start,
                                left_cnt, cgs);
            typename T::Level hll = T::get_level(x0, hl.left_idx);
            typename T::Level hlr = T::get_level(x0, hl.mid_idx);
            for (auto &xm : x) {
                const string &xx = xm.first;
                // (1+2)+3 -> 1+(2+3)
                vector<int16_t> j1s = T::get_quanta(xx, hll),
                                j2s = T::get_quanta(xx, hlr),
                                j12s = T::get_quanta(xx, hl),
                                j3s = T::get_quanta(xx, hr),
                                js = T::get_quanta(xx, h);
                assert(j1s.size() == j2s.size() && j1s.size() == j12s.size() &&
                       j1s.size() == j3s.size() && j1s.size() == js.size());
                vector<pair<vector<int16_t>, FL>> mcgs{
                    make_pair(vector<int16_t>(), xm.second)};
                for (int16_t ik = 0; ik < (int16_t)j1s.size(); ik++) {
                    vector<pair<vector<int16_t>, FL>> ncgs;
                    for (int im = 0; im < (int)mcgs.size(); im++) {
                        int16_t j1 = j1s[ik], j2 = j2s[ik], j12 = j12s[ik],
                                j3 = j3s[ik], j = js[ik];
                        for (int16_t j23 = abs(j2 - j3); j23 <= j2 + j3;
                             j23 += 2) {
                            FL f = (FL)cgs[ik]->racah(j3, j2, j, j1, j23, j12) *
                                   (FL)(FP)sqrt((j23 + 1) * (j12 + 1));
                            if (abs(f) < 1E-12)
                                continue;
                            vector<int16_t> k = mcgs[im].first;
                            k.push_back(j23);
                            ncgs.push_back(make_pair(k, mcgs[im].second * f));
                        }
                    }
                    mcgs = ncgs;
                }
                for (int im = 0; im < (int)mcgs.size(); im++) {
                    stringstream ss;
                    ss << xx.substr(0, h.left_idx)
                       << xx.substr(hl.left_idx, hl.mid_idx - 1 - hl.left_idx)
                       << "+("
                       << xx.substr(hl.mid_idx, hl.right_idx - 1 - hl.mid_idx)
                       << "+"
                       << xx.substr(h.mid_idx, h.right_idx - 1 - h.mid_idx)
                       << ")";
                    if (mcgs[im].first.size() > 1)
                        ss << "[";
                    for (int ik = 0; ik < (int)mcgs[im].first.size(); ik++)
                        ss << (int)mcgs[im].first[ik]
                           << (ik == (int)mcgs[im].first.size() - 1 ? "" : ",");
                    if (mcgs[im].first.size() > 1)
                        ss << "]";
                    ss << xx.substr(h.right_idx - 1);
                    v[ss.str()] += mcgs[im].second;
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
    static map<string, FL> exchange(const map<string, FL> &x, int8_t n,
                                    int8_t i,
                                    const vector<shared_ptr<AnyCG<FL>>> &cgs) {
        map<string, FL> r = x;
        if (i != 0)
            r = recouple(r, 0, i, cgs);
        int8_t ii = i == 0 ? 0 : T::get_level(r.cbegin()->first, 0).mid_idx;
        if (i + 2 < n)
            r = recouple(r, ii, 2, cgs);
        typename T::Level h = T::get_level(r.cbegin()->first, ii);
        while (h.left_cnt != 1 || h.right_cnt != 1)
            h = T::get_level(r.cbegin()->first, h.left_idx);
        map<string, FL> rr;
        for (const auto &mr : r) {
            stringstream ss;
            string lexpr =
                mr.first.substr(h.left_idx, h.mid_idx - 1 - h.left_idx);
            string rexpr =
                mr.first.substr(h.mid_idx, h.right_idx - 1 - h.mid_idx);
            ss << mr.first.substr(0, h.left_idx) << rexpr << '+' << lexpr
               << mr.first.substr(h.right_idx - 1);
            vector<int16_t> tjs = T::get_quanta(mr.first, h);
            vector<int16_t> ltjs = T::get_quanta(lexpr, T::get_level(lexpr, 0));
            vector<int16_t> rtjs = T::get_quanta(rexpr, T::get_level(rexpr, 0));
            FL fx = mr.second * (FL)-1.0;
            for (size_t j = 0; j < cgs.size(); j++)
                fx *= cgs[j]->phase(ltjs[j], rtjs[j], tjs[j]);
            if (abs(fx) >= 1E-12)
                rr[ss.str()] = fx;
        }
        return rr;
    }
    static map<string, FL>
    sort_indices(const map<string, FL> &x, const vector<uint16_t> &indices,
                 const vector<shared_ptr<AnyCG<FL>>> &cgs) {
        int16_t n = (int16_t)indices.size();
        map<string, FL> r = x;
        vector<uint16_t> idx = indices;
        for (int16_t i = 0; i < n; i++)
            for (int16_t j = n - 2; j >= i; j--)
                if (idx[j] > idx[j + 1]) {
                    r = exchange(r, (int8_t)n, (int8_t)j, cgs);
                    swap(idx[j], idx[j + 1]);
                }
        return r;
    }
    static map<string, FL>
    recouple_split(const map<string, FL> &x,
                   const vector<uint16_t> &ref_indices, int split_idx,
                   const vector<shared_ptr<AnyCG<FL>>> &cgs) {
        int nn = (int)ref_indices.size();
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
        map<string, FL> r = x;
        if (split_idx != -1 && split_idx != -2) {
            // handle npdm split
            int ii = 0, ir = 0;
            for (int i = ref_mid; i < (int)ref_split_idx.size(); i++) {
                r = recouple(r, (int8_t)ii, (int8_t)(ref_split_idx[i] - ir),
                             cgs);
                ii = T::get_level(r.begin()->first, ii).mid_idx;
                ir = ref_split_idx[i];
            }
            typename T::Level h = T::get_level(r.begin()->first, 0);
            ii = h.left_idx;
            for (int i = ref_mid - 1; i >= 0; i--) {
                r = recouple(r, (int8_t)ii, (int8_t)ref_split_idx[i], cgs);
                ii = T::get_level(r.begin()->first, ii).left_idx;
            }
        } else if (split_idx == -1) {
            // handle hamiltonian split
            int ii = 0, ir = 0;
            for (auto &rr : ref_split_idx) {
                r = recouple(r, (int8_t)ii, (int8_t)(rr - ir), cgs);
                ii = T::get_level(r.begin()->first, ii).mid_idx, ir = rr;
            }
        } else if (split_idx == -2) {
            // handle drt hamiltonian split
            int ii = 0;
            for (int i = (int)ref_split_idx.size() - 1; i >= 0; i--) {
                r = recouple(r, (int8_t)ii, (int8_t)ref_split_idx[i], cgs);
                ii = T::get_level(r.begin()->first, ii).left_idx;
            }
        }
        return r;
    }
};

struct GeneralSymmPermPattern {
    uint16_t n; // string length
    vector<uint16_t> mask;
    vector<uint16_t> data;
    GeneralSymmPermPattern(uint16_t n,
                           const vector<uint16_t> &mask = vector<uint16_t>())
        : n(n), mask(mask), data(initialize(n, mask)) {}
    static vector<uint16_t>
    all_reordering(const vector<uint16_t> &x,
                   const vector<uint16_t> &mask = vector<uint16_t>()) {
        if (x.size() == 0)
            return x;
        vector<pair<uint16_t, uint16_t>> pp;
        for (auto &ix : x)
            if (pp.size() == 0 || ix != pp.back().first)
                pp.push_back(make_pair(ix, 1));
            else
                pp.back().second++;
        uint16_t maxx = pp.back().first + 1;
        // find the max one and its count
        vector<uint16_t> ha(x.size(), maxx);
        // i = number of max one
        // ha = all possible lists of the max one and undetermined ones (maxx)
        for (uint16_t i = 1; i <= pp.back().second; i++) {
            vector<uint16_t> hb;
            for (int j = 0, k; j < (int)ha.size(); j += (int)x.size()) {
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
        size_t ir = 0;
        for (int h = 0; h < (int)ha.size(); h += (int)x.size())
            for (int j = 0; j < (int)g.size();
                 j += (int)(x.size() - pp.back().second)) {
                memcpy(r.data() + ir, ha.data() + h,
                       x.size() * sizeof(uint16_t));
                for (int k = 0, kk = 0; k < (int)x.size(); k++)
                    if (r[ir + k] == maxx)
                        r[ir + k] = g[j + kk++];
                if (mask.size() != 0) {
                    bool skip = false;
                    for (int k = 1; k < (int)x.size(); k++)
                        skip = skip || (mask[k] == mask[k - 1] &&
                                        r[ir + k] != r[ir + k - 1]);
                    if (skip)
                        continue;
                }
                ir += x.size();
            }
        r.resize(ir);
        return r;
    }
    static vector<uint16_t>
    initialize(uint16_t n, const vector<uint16_t> &mask = vector<uint16_t>()) {
        int mi = n;
        if (mask.size() != 0) {
            mi = 1;
            for (uint16_t j = 1; j < n; j++)
                mi += mask[j] != mask[j - 1];
        }
        map<pair<uint16_t, uint16_t>, vector<vector<uint16_t>>> mp;
        for (uint16_t i = 0; i <= n; i++)
            mp[make_pair(0, i)] = vector<vector<uint16_t>>();
        mp.at(make_pair(0, 0)).push_back(vector<uint16_t>());
        // i = distinct numbers, j = length
        for (uint16_t i = 1; i <= mi; i++) {
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
        for (uint16_t i = 0; i <= mi; i++)
            cnt += mp.at(make_pair(i, n)).size();
        vector<uint16_t> r(cnt * (n + 2));
        size_t ic = 0;
        for (uint16_t i = 0; i <= mi; i++) {
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
};

// generate appropriate spin recoupling formulae after reordering
template <typename FL> struct GeneralSymmPermScheme {
    vector<vector<uint16_t>> index_patterns;
    vector<map<vector<uint16_t>, vector<pair<FL, string>>>> data;
    vector<uint16_t> mask;
    vector<int16_t> targets;
    vector<shared_ptr<AnyCG<FL>>> cgs;
    GeneralSymmPermScheme() {}
    GeneralSymmPermScheme(string spin_str,
                          const vector<shared_ptr<AnyCG<FL>>> &cgs,
                          bool is_npdm = false, bool is_drt = false,
                          const vector<uint16_t> &mask = vector<uint16_t>()) {
        int nn = GeneralSymmTensor<FL>::count_cds(spin_str);
        GeneralSymmPermScheme r = GeneralSymmPermScheme::initialize(
            nn, spin_str, cgs, is_npdm, is_drt, mask);
        index_patterns = r.index_patterns;
        this->mask = r.mask;
        data = r.data;
        targets = r.targets;
        this->cgs = r.cgs;
    }
    static GeneralSymmPermScheme
    initialize(int nn, const string &spin_str,
               const vector<shared_ptr<AnyCG<FL>>> &cgs, bool is_npdm = false,
               bool is_drt = false,
               const vector<uint16_t> &mask = vector<uint16_t>()) {
        using R = GeneralSymmRecoupling<FL>;
        using T = GeneralSymmTensor<FL>;
        GeneralSymmPermPattern spat(nn, mask);
        GeneralSymmPermScheme r;
        r.cgs = cgs;
        r.index_patterns.resize(spat.count());
        r.data.resize(spat.count());
        int ntg = threading->activate_global();
        map<vector<uint16_t>, map<string, FL>> ref_ps;
        map<string, FL> p = map<string, FL>{make_pair(spin_str, (FL)1.0)};
        for (int i = (int)spat.count() - 1; i >= 0; i--) {
            vector<uint16_t> irr = spat[i];
            r.index_patterns[i] = irr;
            vector<uint16_t> rr =
                GeneralSymmPermPattern::all_reordering(irr, mask);
            int nj = irr.size() == 0 ? 1 : (int)(rr.size() / irr.size());
            int iq = is_npdm ? spat.get_split_index(i) : (is_drt ? -2 : -1);
            vector<map<string, FL>> ps(nj);
#pragma omp parallel for schedule(static, 20) num_threads(ntg)
            for (int jj = 0; jj < nj; jj++) {
                vector<uint16_t> indices(rr.begin() + jj * irr.size(),
                                         rr.begin() + (jj + 1) * irr.size());
                vector<uint16_t> perm = T::find_pattern_perm(indices);
                ps[jj] = R::recouple_split(
                    ref_ps.count(perm) ? ref_ps.at(perm)
                                       : R::sort_indices(p, indices, cgs),
                    irr, iq, cgs);
            }
            for (int jj = 0; jj < nj; jj++) {
                vector<uint16_t> indices(rr.begin() + jj * irr.size(),
                                         rr.begin() + (jj + 1) * irr.size());
                vector<uint16_t> perm = T::find_pattern_perm(indices);
                r.data[i][perm] = vector<pair<FL, string>>();
                vector<pair<FL, string>> &udq = r.data[i].at(perm);
                udq.reserve(ps[jj].size());
                for (auto &mr : ps[jj])
                    udq.push_back(make_pair(mr.second, mr.first));
                assert(udq.size() != 0);
                if (i == spat.count() - 1)
                    ref_ps[perm] = ps[jj];
            }
        }
        threading->activate_normal();
        r.targets = T::get_quanta(spin_str, T::get_level(spin_str, 0));
        r.mask = mask;
        return r;
    }
    string to_str() const {
        stringstream ss;
        int cnt = (int)index_patterns.size();
        int nn = cnt == 0 ? 0 : (int)index_patterns[0].size();
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
