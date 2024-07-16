
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

/** Symbolic algebra using Wick's theorem. */

#pragma once

#include "../core/threading.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

namespace block2 {

enum struct WickIndexTypes : uint8_t {
    None = 0,
    Inactive = 1,
    Active = 2, // active ordered as external
    Single = 4, // single ordered as inactive
    External = 8,
    Alpha = 16,
    Beta = 32,
    ActiveSingle = 2 | 4,
    AlphaBeta = 16 | 32,
    InactiveAlpha = 16 | 1,
    ActiveAlpha = 16 | 2,
    ExternalAlpha = 16 | 8,
    InactiveBeta = 32 | 1,
    ActiveBeta = 32 | 2,
    ExternalBeta = 32 | 8,
};

inline string to_str(const WickIndexTypes c) {
    const static string repr[] = {
        "N",   "I",    "A",  "IA",  "S",   "IS",   "AS",  "IAS", "E",  "EI",
        "EA",  "EIA",  "ES", "EIS", "EAS", "EIAS", "A",   "i",   "a",  "ia",
        "s",   "is",   "as", "ias", "e",   "ei",   "ea",  "eia", "es", "eis",
        "eas", "eias", "B",  "I",   "A",   "IA",   "S",   "IS",  "AS", "IAS",
        "E",   "EI",   "EA", "EIA", "ES",  "EIS",  "EAS", "EIAS"};
    return repr[(uint8_t)c];
}

inline WickIndexTypes operator|(WickIndexTypes a, WickIndexTypes b) {
    return WickIndexTypes((uint8_t)a | (uint8_t)b);
}

inline WickIndexTypes operator&(WickIndexTypes a, WickIndexTypes b) {
    return WickIndexTypes((uint8_t)a & (uint8_t)b);
}

inline WickIndexTypes operator~(WickIndexTypes a) {
    return WickIndexTypes(~(uint8_t)a);
}

enum struct WickTensorTypes : uint8_t {
    CreationOperator = 0,
    DestroyOperator = 1,
    SpinFreeOperator = 2,
    KroneckerDelta = 3,
    Tensor = 4
};

struct WickIndex {
    string name;
    WickIndexTypes types;
    WickIndex() : WickIndex("") {}
    explicit WickIndex(const char name[])
        : name(name), types(WickIndexTypes::None) {}
    explicit WickIndex(const string &name)
        : name(name), types(WickIndexTypes::None) {}
    WickIndex(const string &name, WickIndexTypes types)
        : name(name), types(types) {}
    bool operator==(const WickIndex &other) const noexcept {
        return name == other.name && types == other.types;
    }
    bool operator!=(const WickIndex &other) const noexcept {
        return name != other.name || types != other.types;
    }
    bool operator<(const WickIndex &other) const noexcept {
        return types == other.types ? name < other.name : types < other.types;
    }
    size_t hash() const noexcept { return std::hash<string>{}(name); }
    friend ostream &operator<<(ostream &os, const WickIndex &wi) {
        os << wi.name;
        return os;
    }
    bool has_types() const { return types != WickIndexTypes::None; }
    bool is_short() const { return name.length() == 1; }
    WickIndex with_no_types() const { return WickIndex(name); }
    static vector<WickIndex> parse(const string &x) {
        size_t index = x.find_first_of(" ", 0);
        vector<WickIndex> r;
        if (index == string::npos) {
            r.resize(x.size());
            for (int i = 0; i < (int)x.length(); i++)
                r[i] = WickIndex(string(1, x[i]));
        } else {
            size_t last = 0;
            while (index != string::npos) {
                if (index > last)
                    r.push_back(WickIndex(x.substr(last, index - last)));
                last = index + 1;
                index = x.find_first_of(" ", last);
            }
            if (x.length() > last)
                r.push_back(WickIndex(x.substr(last, x.length() - last)));
        }
        return r;
    }
    static vector<WickIndex>
    add_types(vector<WickIndex> r,
              const map<WickIndexTypes, set<WickIndex>> &type_map) {
        for (auto &rr : r)
            for (auto &m : type_map)
                if (m.second.count(rr.with_no_types()))
                    rr.types = rr.types | m.first;
        return r;
    }
    static vector<WickIndex>
    parse_with_types(const string &x,
                     const map<WickIndexTypes, set<WickIndex>> &type_map) {
        return add_types(parse(x), type_map);
    }
    static set<WickIndex> parse_set(const string &x) {
        vector<WickIndex> r = parse(x);
        sort(r.begin(), r.end());
        return set<WickIndex>(r.begin(), r.end());
    }
    static set<WickIndex>
    parse_set_with_types(const string &x,
                         const map<WickIndexTypes, set<WickIndex>> &type_map) {
        vector<WickIndex> r = parse_with_types(x, type_map);
        sort(r.begin(), r.end());
        return set<WickIndex>(r.begin(), r.end());
    }
    void save(ostream &ofs) const {
        size_t lname = (size_t)name.length();
        ofs.write((char *)&lname, sizeof(lname));
        ofs.write((char *)&name[0], sizeof(char) * lname);
        ofs.write((char *)&types, sizeof(types));
    }
    void load(istream &ifs) {
        size_t lname = 0;
        ifs.read((char *)&lname, sizeof(lname));
        name = string(lname, ' ');
        ifs.read((char *)&name[0], sizeof(char) * lname);
        ifs.read((char *)&types, sizeof(types));
    }
};

struct WickPermutation {
    vector<int16_t> data;
    bool negative;
    WickPermutation() : negative(false) {}
    explicit WickPermutation(const vector<int16_t> &data, bool negative = false)
        : data(data), negative(negative) {}
    bool operator==(const WickPermutation &other) const noexcept {
        return negative == other.negative && data == other.data;
    }
    bool operator!=(const WickPermutation &other) const noexcept {
        return negative != other.negative || data != other.data;
    }
    bool operator<(const WickPermutation &other) const noexcept {
        return negative == other.negative ? data < other.data
                                          : negative < other.negative;
    }
    WickPermutation operator*(const WickPermutation &other) const noexcept {
        vector<int16_t> r(data.size());
        for (int i = 0; i < (int)data.size(); i++)
            r[i] = data[other.data[i]];
        return WickPermutation(r, negative ^ other.negative);
    }
    friend ostream &operator<<(ostream &os, const WickPermutation &wp) {
        os << "< " << (wp.negative ? "- " : "+ ");
        for (int i = 0; i < (int)wp.data.size(); i++)
            os << wp.data[i] << " ";
        os << ">";
        return os;
    }
    size_t hash() const noexcept {
        size_t h = std::hash<bool>{}(negative);
        h ^= data.size() + 0x9E3779B9 + (h << 6) + (h >> 2);
        for (int i = 0; i < data.size(); i++)
            h ^= (std::hash<int16_t>{}(data[i])) + 0x9E3779B9 + (h << 6) +
                 (h >> 2);
        return h;
    }
    static vector<WickPermutation>
    complete_set(int n, const vector<WickPermutation> &def) {
        vector<int16_t> ident(n);
        for (int i = 0; i < n; i++)
            ident[i] = i;
        auto hx = [](const WickPermutation &wp) { return wp.hash(); };
        unordered_set<WickPermutation, decltype(hx)> swp(def.size(), hx);
        vector<WickPermutation> vwp;
        vwp.push_back(WickPermutation(ident, false));
        swp.insert(vwp[0]);
        for (int k = 0; k < vwp.size(); k++) {
            WickPermutation g = vwp[k];
            for (auto &d : def) {
                WickPermutation h = g * d;
                if (!swp.count(h))
                    vwp.push_back(h), swp.insert(h);
            }
        }
        return vwp;
    }
    static vector<WickPermutation> non_symmetric() {
        return vector<WickPermutation>();
    }
    static vector<WickPermutation> two_symmetric() {
        return vector<WickPermutation>{WickPermutation({1, 0}, false)};
    }
    // symmetry of amplitudes of canonical transformation theory
    static vector<WickPermutation> two_anti_symmetric() {
        return vector<WickPermutation>{WickPermutation({1, 0}, true)};
    }
    // symmetry of amplitudes of canonical transformation theory
    static vector<WickPermutation> four_anti_symmetric() {
        return vector<WickPermutation>{WickPermutation({2, 3, 0, 1}, true),
                                       WickPermutation({1, 0, 3, 2}, false)};
    }
    // chem =  vijkl Ci Ck Dl Dj
    static vector<WickPermutation> qc_chem() {
        return vector<WickPermutation>{WickPermutation({2, 3, 0, 1}, false),
                                       WickPermutation({1, 0, 2, 3}, false),
                                       WickPermutation({0, 1, 3, 2}, false)};
    }
    // phys = vijkl Ci Cj Dl Dk
    // chem -> phys : i = 0 j = 2 l = 3 k = 1 (eph = eri.transpose(0, 2, 1, 3))
    // phys -> chem : eri = eph.transpose(0, 2, 1, 3)
    static vector<WickPermutation> qc_phys() {
        return vector<WickPermutation>{WickPermutation({0, 3, 2, 1}, false),
                                       WickPermutation({2, 1, 0, 3}, false),
                                       WickPermutation({1, 0, 3, 2}, false)};
    }
    // anti = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    static vector<WickPermutation> four_anti() {
        return vector<WickPermutation>{WickPermutation({2, 3, 0, 1}, false),
                                       WickPermutation({1, 0, 2, 3}, true),
                                       WickPermutation({0, 1, 3, 2}, true)};
    }
    static vector<WickPermutation> pair_anti_symmetric(int n) {
        vector<WickPermutation> r(max(n - 1, 0) * 2);
        vector<int16_t> x(n * 2);
        for (int i = 0; i < n + n; i++)
            x[i] = i;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n; j++)
                x[j] = j == 0 ? i : (j == i ? 0 : j);
            r[i - 1] = WickPermutation(x, true);
        }
        for (int i = 0; i < n + n; i++)
            x[i] = i;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n; j++)
                x[j + n] = j == 0 ? i + n : (j == i ? n : j + n);
            r[i - 1 + n - 1] = WickPermutation(x, true);
        }
        return r;
    }
    static vector<WickPermutation> pair_symmetric(int n,
                                                  bool hermitian = false) {
        vector<WickPermutation> r(max(n - 1, 0));
        vector<int16_t> x(n * 2);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                x[j] = j == 0 ? i : (j == i ? 0 : j);
                x[j + n] = j == 0 ? i + n : (j == i ? n : j + n);
            }
            r[i - 1] = WickPermutation(x, false);
        }
        if (hermitian && n != 0) {
            for (int j = 0; j < n; j++)
                x[j] = j + n, x[j + n] = j;
            r.push_back(WickPermutation(x, false));
        }
        return r;
    }
    static vector<WickPermutation> all(int n) {
        size_t nr = 1;
        vector<int16_t> x(n);
        for (int i = 0; i < n; i++)
            x[i] = (int16_t)i, nr *= (i + 1);
        vector<WickPermutation> r(nr);
        size_t ir = 0;
        r[ir++] = WickPermutation(x, false);
        while (next_permutation(x.begin(), x.end()))
            r[ir++] = WickPermutation(x, false);
        assert(ir == nr);
        return r;
    }
    void save(ostream &ofs) const {
        size_t ldata = (size_t)data.size();
        ofs.write((char *)&ldata, sizeof(ldata));
        ofs.write((char *)&data[0], sizeof(int16_t) * ldata);
        ofs.write((char *)&negative, sizeof(negative));
    }
    void load(istream &ifs) {
        size_t ldata = 0;
        ifs.read((char *)&ldata, sizeof(ldata));
        data.resize(ldata);
        ifs.read((char *)&data[0], sizeof(int16_t) * ldata);
        ifs.read((char *)&negative, sizeof(negative));
    }
};

struct WickTensor {
    string name;
    vector<WickIndex> indices;
    vector<WickPermutation> perms;
    WickTensorTypes type;
    WickTensor() : name(""), type(WickTensorTypes::Tensor) {}
    WickTensor(
        const string &name, const vector<WickIndex> &indices,
        const vector<WickPermutation> &perms = WickPermutation::non_symmetric(),
        WickTensorTypes type = WickTensorTypes::Tensor)
        : name(name), indices(indices),
          perms(WickPermutation::complete_set((int)indices.size(), perms)),
          type(type) {}
    static vector<WickPermutation>
    reset_permutations(const vector<WickIndex> &indices,
                       const vector<WickPermutation> &perms) {
        vector<WickPermutation> rperms;
        set<WickPermutation> sperms;
        for (auto &perm : perms) {
            if (sperms.count(perm))
                continue;
            bool valid = true;
            for (int i = 0; i < (int)indices.size() && valid; i++) {
                auto &ia = indices[perm.data[i]], &ib = indices[i];
                if ((ia.types != WickIndexTypes::None ||
                     ib.types != WickIndexTypes::None) &&
                    ((((ia.types & (~WickIndexTypes::AlphaBeta)) !=
                           WickIndexTypes::None ||
                       (ib.types & (~WickIndexTypes::AlphaBeta)) !=
                           WickIndexTypes::None) &&
                      ((ia.types & ib.types) & (~WickIndexTypes::AlphaBeta)) ==
                          WickIndexTypes::None) ||
                     (ia.types & WickIndexTypes::AlphaBeta) !=
                         (ib.types & WickIndexTypes::AlphaBeta)))
                    valid = false;
            }
            if (valid) {
                sperms.insert(perm);
                rperms.push_back(perm);
            }
        }
        return rperms;
    }
    static WickTensor
    parse(const string &tex_expr,
          const map<WickIndexTypes, set<WickIndex>> &idx_map,
          const map<pair<string, int>, vector<WickPermutation>> &perm_map) {
        string name, indices;
        bool is_name = true, has_sq = tex_expr.find('[') != string::npos;
        for (char c : tex_expr)
            if ((!has_sq && c == '_') || c == '[')
                is_name = false;
            else if (c == ',')
                continue;
            else if (c == ' ' && !is_name)
                indices.push_back(c);
            else if (string("{}]").find(c) == string::npos && is_name) {
                if (c != ' ')
                    name.push_back(c);
            } else if (string("{}]").find(c) == string::npos && !is_name)
                indices.push_back(c);
            else if (c == ']' || c == '}')
                break;
        int index_len = 0;
        if (indices.find(' ') != string::npos) {
            size_t last = 0;
            size_t index = indices.find_first_of(' ', last);
            while (index != string::npos) {
                if (index > last)
                    index_len++;
                last = index + 1;
                index = indices.find_first_of(' ', last);
            }
            if (indices.length() > last)
                index_len++;
        } else
            index_len = (int)indices.size();
        vector<WickPermutation> perms;
        if (perm_map.count(make_pair(name, index_len)))
            perms = perm_map.at(make_pair(name, index_len));
        WickTensorTypes tensor_type = WickTensorTypes::Tensor;
        if (name == "C" && index_len == 1)
            tensor_type = WickTensorTypes::CreationOperator;
        else if (name == "D" && index_len == 1)
            tensor_type = WickTensorTypes::DestroyOperator;
        // the number indicates the summed spin label
        else if (name[0] == 'C' && name.length() >= 2 && name[1] >= '0' &&
                 name[1] <= '9' && index_len == 1)
            tensor_type = WickTensorTypes::CreationOperator;
        else if (name[0] == 'D' && name.length() >= 2 && name[1] >= '0' &&
                 name[1] <= '9' && index_len == 1)
            tensor_type = WickTensorTypes::DestroyOperator;
        // for external usage
        else if ((name == "Ca" || name == "Cb") && index_len == 1)
            tensor_type = WickTensorTypes::CreationOperator;
        else if ((name == "Da" || name == "Db") && index_len == 1)
            tensor_type = WickTensorTypes::DestroyOperator;
        else if (name[0] == 'E' && name.length() == 2 &&
                 index_len == (int)(name[1] - '0') * 2) {
            tensor_type = WickTensorTypes::SpinFreeOperator;
            perms = WickPermutation::pair_symmetric((int)(name[1] - '0'));
        } else if (name[0] == 'E' && name.length() == index_len + 1 &&
                   all_of(name.begin() + 1, name.end(),
                          [](const char &c) { return c == 'C' || c == 'D'; })) {
            tensor_type = WickTensorTypes::SpinFreeOperator;
            perms = WickPermutation::non_symmetric();
        } else if (name[0] == 'R' && name.length() == 2 &&
                   index_len == (int)(name[1] - '0') * 2) {
            tensor_type = WickTensorTypes::SpinFreeOperator;
            perms = WickPermutation::pair_symmetric((int)(name[1] - '0'), true);
        } else if (name == "delta" && index_len == 2) {
            tensor_type = WickTensorTypes::KroneckerDelta;
            perms = WickPermutation::two_symmetric();
        }
        return WickTensor(name, WickIndex::parse_with_types(indices, idx_map),
                          perms, tensor_type);
    }
    bool operator==(const WickTensor &other) const noexcept {
        return type == other.type && name == other.name &&
               indices == other.indices;
    }
    bool operator!=(const WickTensor &other) const noexcept {
        return type != other.type || name != other.name ||
               indices != other.indices;
    }
    bool operator<(const WickTensor &other) const noexcept {
        int fc = fermi_type_compare(other);
        return fc != 0 ? (fc == -1)
                       : (name != other.name
                              ? name < other.name
                              : (type == other.type ? indices < other.indices
                                                    : type < other.type));
    }
    // -1 = less than; +1 = greater than; 0 = equal to
    int fermi_type_compare(const WickTensor &other) const noexcept {
        const WickIndexTypes mask =
            WickIndexTypes::Inactive | WickIndexTypes::Active |
            WickIndexTypes::Single | WickIndexTypes::External;
        WickIndexTypes x_type = indices.size() == 0 ? WickIndexTypes::None
                                                    : indices[0].types & mask;
        WickIndexTypes y_type = other.indices.size() == 0
                                    ? WickIndexTypes::None
                                    : other.indices[0].types & mask;
        WickIndexTypes occ_type =
            WickIndexTypes(min((uint8_t)x_type, (uint8_t)y_type));
        WickIndexTypes max_type =
            WickIndexTypes(max((uint8_t)x_type, (uint8_t)y_type));
        if (occ_type == WickIndexTypes::None ||
            occ_type == WickIndexTypes::External ||
            (occ_type == WickIndexTypes::Active &&
             max_type == WickIndexTypes::Active))
            occ_type = WickIndexTypes::Inactive;
        return fermi_type(occ_type) != other.fermi_type(occ_type)
                   ? (fermi_type(occ_type) < other.fermi_type(occ_type) ? -1
                                                                        : 1)
                   : 0;
    }
    WickTensor operator*(const WickPermutation &perm) const noexcept {
        vector<WickIndex> xindices(indices.size());
        for (int i = 0; i < (int)indices.size(); i++)
            xindices[i] = indices[perm.data[i]];
        return WickTensor(name, xindices, perms, type);
    }
    // Ca [00] < Di [01] < Ci [10] < Da [11]
    // Ca [00] < Du [01] < Cu [10] < Da [11]
    // Cu [00] < Di [01] < Ci [10] < Du [11]
    int fermi_type(WickIndexTypes occ_type) const noexcept {
        const int x = type == WickTensorTypes::DestroyOperator;
        const int y = indices.size() != 0 &&
                      (indices[0].types & occ_type) != WickIndexTypes::None;
        return x | ((x ^ y) << 1);
    }
    string to_str(const WickPermutation &perm) const {
        string d = " ";
        if (all_of(indices.begin(), indices.end(),
                   [](const WickIndex &idx) { return idx.is_short(); }))
            d = "";
        stringstream ss;
        ss << (perm.negative ? "-" : "") << name << "[" << d;
        for (int i = 0; i < (int)indices.size(); i++) {
            if (type == WickTensorTypes::SpinFreeOperator &&
                i * 2 == (int)indices.size())
                ss << "," << d;
            ss << indices[perm.data[i]] << d;
        }
        ss << "]";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, const WickTensor &wt) {
        os << wt.to_str(wt.perms[0]);
        return os;
    }
    static WickTensor kronecker_delta(const vector<WickIndex> &indices) {
        assert(indices.size() == 2);
        return WickTensor("delta", indices, WickPermutation::two_symmetric(),
                          WickTensorTypes::KroneckerDelta);
    }
    // GUGA book P66 EQ21 E[ij] = x_{i\sigma}^\dagger x_{j\sigma}
    // e[ik,jl] = E[ij]E[kl] - delta[kj]E[il] = e[ki,lj] ==> e[ij,kl] in P66
    // e[ijk...abc...] = SUM <stu...> C[is] C[jt] C[ku] ... D[cu] D[bt] D[as]
    // ...
    static WickTensor spin_free(const vector<WickIndex> &indices) {
        assert(indices.size() % 2 == 0);
        const int k = (int)(indices.size() / 2);
        stringstream name;
        name << "E" << k;
        return WickTensor(name.str(), indices,
                          WickPermutation::pair_symmetric(k),
                          WickTensorTypes::SpinFreeOperator);
    }
    static WickTensor general_spin_free(const vector<WickIndex> &indices,
                                        const vector<uint8_t> &cds) {
        assert(indices.size() % 2 == 0);
        const int k = (int)(indices.size() / 2);
        stringstream name;
        name << "E";
        for (auto &c : cds)
            name << (c ? 'C' : 'D');
        return WickTensor(name.str(), indices, WickPermutation::non_symmetric(),
                          WickTensorTypes::SpinFreeOperator);
    }
    // with additional pq,rs -> rs,pq symmetry
    static WickTensor
    spin_free_density_matrix(const vector<WickIndex> &indices) {
        assert(indices.size() % 2 == 0);
        const int k = (int)(indices.size() / 2);
        stringstream name;
        name << "R" << k;
        return WickTensor(name.str(), indices,
                          WickPermutation::pair_symmetric(k, true),
                          WickTensorTypes::SpinFreeOperator);
    }
    static WickTensor cre(const WickIndex &index, const string &name = "C") {
        return WickTensor(name, vector<WickIndex>{index},
                          WickPermutation::non_symmetric(),
                          WickTensorTypes::CreationOperator);
    }
    static WickTensor cre(const WickIndex &index,
                          const map<WickIndexTypes, set<WickIndex>> &idx_map,
                          const string &name = "C") {
        return WickTensor(
            name, WickIndex::add_types(vector<WickIndex>{index}, idx_map),
            WickPermutation::non_symmetric(),
            WickTensorTypes::CreationOperator);
    }
    static WickTensor des(const WickIndex &index, const string &name = "D") {
        return WickTensor(name, vector<WickIndex>{index},
                          WickPermutation::non_symmetric(),
                          WickTensorTypes::DestroyOperator);
    }
    static WickTensor des(const WickIndex &index,
                          const map<WickIndexTypes, set<WickIndex>> &idx_map,
                          const string &name = "D") {
        return WickTensor(
            name, WickIndex::add_types(vector<WickIndex>{index}, idx_map),
            WickPermutation::non_symmetric(), WickTensorTypes::DestroyOperator);
    }
    int get_spin_tag() const {
        if ((type == WickTensorTypes::CreationOperator ||
             type == WickTensorTypes::DestroyOperator) &&
            name.length() >= 2 && name[1] >= '0' && name[1] <= '9') {
            int spin_tag = 0;
            for (int i = 1; i < name.length(); i++)
                spin_tag = spin_tag * 10 + (name[i] - '0');
            return spin_tag;
        } else
            return -1;
    }
    void set_spin_tag(int tag) {
        if ((type == WickTensorTypes::CreationOperator ||
             type == WickTensorTypes::DestroyOperator) &&
            (name.length() == 1 ||
             (name.length() >= 2 && name[1] >= '0' && name[1] <= '9'))) {
            stringstream ss;
            if (tag >= 0)
                ss << tag;
            name = string(1, name[0]) + ss.str();
        }
    }
    WickTensor sort(double &factor) const {
        WickTensor x = *this;
        bool neg = false;
        for (auto &perm : perms) {
            WickTensor z = *this * perm;
            if (z.indices < x.indices)
                x = z, neg = perm.negative;
        }
        if (neg)
            factor = -factor;
        return x;
    }
    // ctr_maps : first is a map from given index name to abstract index
    // second int is 1 / -1 indicating whether it is negative
    // new_idx is the number of abstract indices already used in any map in
    // ctr_maps
    vector<pair<map<WickIndex, int>, int>>
    sort_gen_maps(const WickTensor &ref, const set<WickIndex> &ctr_idxs,
                  const vector<pair<map<WickIndex, int>, int>> &ctr_maps,
                  int new_idx) {
        set<pair<map<WickIndex, int>, int>> new_maps;
        assert(perms.size() != 0);
        for (auto &perm : perms) {
            WickTensor zz = *this * perm;
            for (auto &ctr_map : ctr_maps) {
                WickTensor z = zz;
                map<WickIndex, int> new_map;
                int kidx = new_idx;
                for (auto &wi : z.indices)
                    if (ctr_idxs.count(wi)) {
                        if (!ctr_map.first.count(wi) && !new_map.count(wi))
                            new_map[wi] = kidx++;
                        wi.name = string(1, (ctr_map.first.count(wi)
                                                 ? ctr_map.first.at(wi)
                                                 : new_map.at(wi)) +
                                                '0');
                    }
                // ref is a already sorted tensor, we want to find all possible
                // index name to abstract index maps to get that tensor
                // but only contracted indices are allowed to be freely changed
                if (z.indices == ref.indices) {
                    new_map.insert(ctr_map.first.begin(), ctr_map.first.end());
                    new_maps.insert(make_pair(new_map, perm.negative
                                                           ? -ctr_map.second
                                                           : ctr_map.second));
                }
            }
        }
        return vector<pair<map<WickIndex, int>, int>>(new_maps.begin(),
                                                      new_maps.end());
    }
    WickTensor sort(const set<WickIndex> &ctr_idxs,
                    const vector<pair<map<WickIndex, int>, int>> &ctr_maps,
                    int &new_idx) const {
        int kidx = new_idx;
        // x is the final output
        // first, we construct a tensor with no perm
        WickTensor x = *this;
        map<WickIndex, int> new_map;
        assert(ctr_maps.size() != 0);
        for (auto &wi : x.indices)
            if (ctr_idxs.count(wi)) {
                if (!ctr_maps[0].first.count(wi) && !new_map.count(wi))
                    new_map[wi] = kidx++;
                wi.name = string(1, (ctr_maps[0].first.count(wi)
                                         ? ctr_maps[0].first.at(wi)
                                         : new_map.at(wi)) +
                                        '0');
            }
        // second, try all perms, get the tensor with abstract indices
        // x = min(z)
        // get the one with the smallest abstract indices
        // but here we do not need to know the name to abstract map
        // the name to abstract map may have multiple possibilities
        // which will be generaated in the next step
        for (auto &perm : perms) {
            WickTensor zz = *this * perm;
            for (auto &ctr_map : ctr_maps) {
                WickTensor z = zz;
                new_map.clear();
                kidx = new_idx;
                for (auto &wi : z.indices)
                    if (ctr_idxs.count(wi)) {
                        if (!ctr_map.first.count(wi) && !new_map.count(wi))
                            new_map[wi] = kidx++;
                        wi.name = string(1, (ctr_map.first.count(wi)
                                                 ? ctr_map.first.at(wi)
                                                 : new_map.at(wi)) +
                                                '0');
                    }
                if (z.indices < x.indices)
                    x = z;
            }
        }
        new_idx = kidx;
        return x;
    }
    string get_permutation_rules() const {
        stringstream ss;
        for (int i = 0; i < (int)perms.size(); i++)
            ss << to_str(perms[i])
               << (i == (int)perms.size() - 1 ? "" : " == ");
        return ss.str();
    }
    static map<string, string> get_index_map(const vector<WickIndex> &idxa,
                                             const vector<WickIndex> &idxb) {
        map<string, string> r;
        if (idxa.size() != idxb.size())
            return r;
        map<WickIndexTypes, pair<vector<uint16_t>, vector<uint16_t>>> mp;
        for (int i = 0; i < (int)idxa.size(); i++)
            mp[idxa[i].types].first.push_back(i);
        for (int i = 0; i < (int)idxb.size(); i++)
            mp[idxb[i].types].second.push_back(i);
        bool ok = true;
        for (auto &x : mp) {
            if (x.second.first.size() != x.second.second.size()) {
                ok = false;
                r.clear();
                break;
            }
            for (int j = 0; j < (int)x.second.first.size(); j++)
                r[idxa[x.second.first[j]].name] = idxb[x.second.second[j]].name;
        }
        return r;
    }
    static vector<map<string, string>>
    get_all_index_permutations(const vector<WickIndex> &indices) {
        map<WickIndexTypes, pair<vector<uint16_t>, vector<WickPermutation>>> mp;
        for (int i = 0; i < (int)indices.size(); i++)
            mp[indices[i].types].first.push_back(i);
        size_t np = 1;
        for (auto &x : mp) {
            x.second.second = WickPermutation::all((int)x.second.first.size());
            np *= x.second.second.size();
        }
        map<string, string> mx;
        for (auto &x : indices)
            mx[x.name] = x.name;
        vector<map<string, string>> r(np, mx);
        for (size_t ip = 0; ip < np; ip++) {
            map<string, string> &mt = r[ip];
            size_t jp = ip;
            for (auto &x : mp) {
                WickPermutation &wp =
                    x.second.second[jp % x.second.second.size()];
                jp /= x.second.second.size();
                for (int j = 0; j < (int)x.second.first.size(); j++)
                    mt[indices[x.second.first[j]].name] =
                        indices[x.second.first[wp.data[j]]].name;
            }
        }
        return r;
    }
    void save(ostream &ofs) const {
        size_t lname = (size_t)name.length();
        ofs.write((char *)&lname, sizeof(lname));
        ofs.write((char *)&name[0], sizeof(char) * lname);
        size_t lindices = (size_t)indices.size();
        ofs.write((char *)&lindices, sizeof(lindices));
        for (size_t i = 0; i < lindices; i++)
            indices[i].save(ofs);
        size_t lperms = (size_t)perms.size();
        ofs.write((char *)&lperms, sizeof(lperms));
        for (size_t i = 0; i < lperms; i++)
            perms[i].save(ofs);
        ofs.write((char *)&type, sizeof(type));
    }
    void load(istream &ifs) {
        size_t lname = 0;
        ifs.read((char *)&lname, sizeof(lname));
        name = string(lname, ' ');
        ifs.read((char *)&name[0], sizeof(char) * lname);
        size_t lindices = 0;
        ifs.read((char *)&lindices, sizeof(lindices));
        indices.resize(lindices);
        for (size_t i = 0; i < lindices; i++)
            indices[i].load(ifs);
        size_t lperms = 0;
        ifs.read((char *)&lperms, sizeof(lperms));
        perms.resize(lperms);
        for (size_t i = 0; i < lperms; i++)
            perms[i].load(ifs);
        ifs.read((char *)&type, sizeof(type));
    }
};

struct WickString {
    vector<WickTensor> tensors;
    set<WickIndex> ctr_indices;
    double factor;
    WickString() : factor(0.0) {}
    explicit WickString(const WickTensor &tensor, double factor = 1.0)
        : factor(factor), tensors({tensor}), ctr_indices() {}
    explicit WickString(const vector<WickTensor> &tensors)
        : factor(1.0), tensors(tensors), ctr_indices() {}
    WickString(const vector<WickTensor> &tensors,
               const set<WickIndex> &ctr_indices, double factor = 1.0)
        : factor(factor), tensors(tensors), ctr_indices(ctr_indices) {}
    bool abs_equal_to(const WickString &other) const noexcept {
        return tensors.size() == other.tensors.size() &&
               ctr_indices.size() == other.ctr_indices.size() &&
               tensors == other.tensors && ctr_indices == other.ctr_indices;
    }
    bool operator==(const WickString &other) const noexcept {
        return factor == other.factor && tensors == other.tensors &&
               ctr_indices == other.ctr_indices;
    }
    bool operator!=(const WickString &other) const noexcept {
        return factor != other.factor || tensors != other.tensors ||
               ctr_indices != other.ctr_indices;
    }
    bool operator<(const WickString &other) const noexcept {
        if (tensors.size() != other.tensors.size())
            return tensors.size() < other.tensors.size();
        else if (ctr_indices.size() != other.ctr_indices.size())
            return ctr_indices.size() < other.ctr_indices.size();
        else if (tensors != other.tensors)
            return tensors < other.tensors;
        else if (ctr_indices != other.ctr_indices)
            return ctr_indices < other.ctr_indices;
        else
            return factor < other.factor;
    }
    static WickString
    parse(const string &tex_expr,
          const map<WickIndexTypes, set<WickIndex>> &idx_map,
          const map<pair<string, int>, vector<WickPermutation>> &perm_map) {
        vector<WickTensor> tensors;
        set<WickIndex> ctr_idxs;
        string sum_expr, fac_expr, tensor_expr;
        int idx = 0;
        for (; idx < tex_expr.length(); idx++) {
            char c = tex_expr[idx];
            if (c == ' ' || c == '(')
                continue;
            else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+')
                fac_expr.push_back(c);
            else
                break;
        }
        for (; idx < tex_expr.length() &&
               (tex_expr[idx] == ')' || tex_expr[idx] == ' ');
             idx++)
            ;
        bool has_sum = false;
        if (tex_expr.substr(idx, 6) == "\\sum_{")
            idx += 6, has_sum = true;
        else if (tex_expr.substr(idx, 5) == "SUM <")
            idx += 5, has_sum = true;
        for (; idx < tex_expr.length() && has_sum; idx++) {
            char c = tex_expr[idx];
            if (c == '}' || c == '|' || c == '>')
                break;
            else
                sum_expr.push_back(c);
        }
        vector<WickIndexTypes> ctr_idx_types;
        if (idx < tex_expr.length() && tex_expr[idx] == '|') {
            for (idx++; idx < tex_expr.length(); idx++)
                if (tex_expr[idx] == '>')
                    break;
                else if (tex_expr[idx] == 'I')
                    ctr_idx_types.push_back(WickIndexTypes::Inactive);
                else if (tex_expr[idx] == 'A')
                    ctr_idx_types.push_back(WickIndexTypes::Active);
                else if (tex_expr[idx] == 'S')
                    ctr_idx_types.push_back(WickIndexTypes::Single);
                else if (tex_expr[idx] == 'E')
                    ctr_idx_types.push_back(WickIndexTypes::External);
        }
        if (idx < tex_expr.length() &&
            (tex_expr[idx] == '}' || tex_expr[idx] == '>'))
            idx++;
        for (; idx < tex_expr.length(); idx++) {
            char c = tex_expr[idx];
            if (c == '}' || c == ']') {
                tensor_expr.push_back(c);
                tensors.push_back(
                    WickTensor::parse(tensor_expr, idx_map, perm_map));
                tensor_expr = "";
            } else
                tensor_expr.push_back(c);
        }
        map<WickIndex, WickIndex> ctr_idx_type_adjust;
        if (sum_expr != "") {
            vector<WickIndex> v_ctr_idxs =
                WickIndex::parse_with_types(sum_expr, idx_map);
            if (ctr_idx_types.size() == v_ctr_idxs.size())
                for (size_t i = 0; i < v_ctr_idxs.size(); i++) {
                    auto &xct = ctr_idx_type_adjust[v_ctr_idxs[i]];
                    v_ctr_idxs[i].types = ctr_idx_types[i];
                    xct = v_ctr_idxs[i];
                }
            sort(v_ctr_idxs.begin(), v_ctr_idxs.end());
            ctr_idxs = set<WickIndex>(v_ctr_idxs.begin(), v_ctr_idxs.end());
        }
        while (tensor_expr != "" && tensor_expr[0] == ' ')
            tensor_expr = tensor_expr.substr(1);
        if (tensor_expr != "")
            tensors.push_back(
                WickTensor::parse(tensor_expr, idx_map, perm_map));
        for (auto &wt : tensors)
            for (auto &wi : wt.indices)
                if (ctr_idx_type_adjust.count(wi))
                    wi = ctr_idx_type_adjust.at(wi);
        double xfac = 1.0;
        if (fac_expr == "-")
            xfac = -1;
        else if (fac_expr != "" && fac_expr != "+")
            xfac = atof(fac_expr.c_str());
        return WickString(tensors, ctr_idxs, xfac);
    }
    void fix_index_names() {
        set<WickIndex> new_ctr_idxs;
        for (auto wi : ctr_indices) {
            if ((int)(wi.name[0]) >= 123)
                wi.name[0] -= 58;
            else if ((int)(wi.name[0]) < 0)
                wi.name[0] = (int)wi.name[0] + 256 - 58;
            new_ctr_idxs.insert(wi);
        }
        ctr_indices = new_ctr_idxs;
        for (auto &wt : tensors)
            for (auto &wi : wt.indices)
                if ((int)(wi.name[0]) >= 123)
                    wi.name[0] -= 58;
                else if ((int)(wi.name[0]) < 0)
                    wi.name[0] = (int)wi.name[0] + 256 - 58;
    }
    vector<WickString> substitute(
        const map<string, pair<WickTensor, vector<WickString>>> &defs) const {
        vector<WickString> r = {*this};
        set<WickIndex> orig_idxs = used_indices();
        r[0].tensors.clear();
        for (auto &wt : tensors) {
            if (!defs.count(wt.name) ||
                defs.at(wt.name).first.indices.size() != wt.indices.size()) {
                for (auto &rr : r)
                    rr.tensors.push_back(wt);
            } else {
                auto &p = defs.at(wt.name);
                vector<WickString> rx;
                for (auto &rr : r) {
                    for (auto &dx : p.second) {
                        WickString rg = rr;
                        set<WickIndex> used_idxs = rr.used_indices();
                        used_idxs.insert(orig_idxs.begin(), orig_idxs.end());
                        used_idxs.insert(wt.indices.begin(), wt.indices.end());
                        map<WickIndex, WickIndex> idx_map;
                        assert(p.first.indices.size() == wt.indices.size());
                        for (int i = 0; i < (int)wt.indices.size(); i++)
                            idx_map[p.first.indices[i]] = wt.indices[i];
                        for (auto &wi : dx.ctr_indices) {
                            WickIndex g = wi;
                            for (int i = 0; i < 100; i++) {
                                g.name[0] = wi.name[0] + i;
                                if ((int)(g.name[0]) >= 123)
                                    g.name[0] -= 58;
                                if (!used_idxs.count(g))
                                    break;
                            }
                            rg.ctr_indices.insert(g);
                            used_idxs.insert(g);
                            idx_map[wi] = g;
                        }
                        for (auto wx : dx.tensors) {
                            for (auto &wi : wx.indices)
                                wi = idx_map.at(wi);
                            rg.tensors.push_back(wx);
                        }
                        rg.factor *= dx.factor;
                        rx.push_back(rg);
                    }
                }
                r = rx;
            }
        }
        return r;
    }
    WickString index_map(const map<string, string> &maps) const {
        WickString r = *this;
        for (auto &wt : r.tensors)
            for (auto &wi : wt.indices)
                if (maps.count(wi.name))
                    wi.name = maps.at(wi.name);
        return r;
    }
    set<WickIndex> used_indices() const {
        set<WickIndex> r;
        for (auto &ts : tensors)
            r.insert(ts.indices.begin(), ts.indices.end());
        return r;
    }
    set<string> used_index_names() const {
        set<string> r;
        for (auto &ts : tensors)
            for (auto &idx : ts.indices)
                r.insert(idx.name);
        return r;
    }
    set<string> ctr_index_names() const {
        set<string> r;
        for (auto &idx : ctr_indices)
            r.insert(idx.name);
        return r;
    }
    map<int, int> used_spin_tags() const {
        map<int, int> r;
        for (auto &ts : tensors) {
            int spin_tag = ts.get_spin_tag();
            if (spin_tag != -1)
                r[spin_tag]++;
        }
        return r;
    }
    WickString operator*(const WickString &other) const noexcept {
        vector<WickTensor> xtensors = tensors;
        xtensors.insert(xtensors.end(), other.tensors.begin(),
                        other.tensors.end());
        set<WickIndex> xctr_indices = ctr_indices;
        xctr_indices.insert(other.ctr_indices.begin(), other.ctr_indices.end());
        // resolve conflicts in summation indices
        set<string> a_idxs = used_index_names(),
                    b_idxs = other.used_index_names();
        vector<string> used_idxs_v(a_idxs.size() + b_idxs.size());
        auto it = set_union(a_idxs.begin(), a_idxs.end(), b_idxs.begin(),
                            b_idxs.end(), used_idxs_v.begin());
        set<string> used_idxs(used_idxs_v.begin(), it);
        vector<string> a_rep(ctr_indices.size()),
            b_rep(other.ctr_indices.size()), c_rep(ctr_indices.size());
        set<string> ctr_names = ctr_index_names();
        it = set_intersection(ctr_names.begin(), ctr_names.end(),
                              b_idxs.begin(), b_idxs.end(), a_rep.begin());
        a_rep.resize(it - a_rep.begin());
        set<string> other_ctr_names = other.ctr_index_names();
        it = set_intersection(other_ctr_names.begin(), other_ctr_names.end(),
                              a_idxs.begin(), a_idxs.end(), b_rep.begin());
        b_rep.resize(it - b_rep.begin());
        it = set_intersection(ctr_names.begin(), ctr_names.end(),
                              other_ctr_names.begin(), other_ctr_names.end(),
                              c_rep.begin());
        c_rep.resize(it - c_rep.begin());
        set<string> xa_rep(a_rep.begin(), a_rep.end()),
            xb_rep(b_rep.begin(), b_rep.end()),
            xc_rep(c_rep.begin(), c_rep.end());
        map<string, string> mp_idxs;
        for (auto &idx : used_idxs)
            if (xa_rep.count(idx) || xb_rep.count(idx))
                for (int i = 1; i < 100; i++) {
                    string g = idx;
                    g[0] += i;
                    if ((int)(g[0]) >= 123)
                        g[0] -= 58;
                    if (!used_idxs.count(g)) {
                        used_idxs.insert(g);
                        mp_idxs[idx] = g;
                        break;
                    }
                }
        // change contraction index in a, if it is also free index in b
        for (int i = 0; i < tensors.size(); i++)
            for (auto &wi : xtensors[i].indices)
                if (mp_idxs.count(wi.name) && xa_rep.count(wi.name) &&
                    !xc_rep.count(wi.name))
                    wi.name = mp_idxs[wi.name];
        // change contraction index in b,
        // if it is also free index or contraction index in a
        for (int i = (int)tensors.size(); i < (int)xtensors.size(); i++)
            for (auto &wi : xtensors[i].indices)
                if (mp_idxs.count(wi.name) && xb_rep.count(wi.name))
                    wi.name = mp_idxs[wi.name];
        // resolve conflicts in spin tags
        map<int, int> a_tags = used_spin_tags(),
                      b_tags = other.used_spin_tags();
        int new_tag = 0;
        for (auto &atg : a_tags)
            if (b_tags.count(atg.first)) {
                int atgv = atg.second;
                int btgv = b_tags.at(atg.first);
                if (btgv > 1) {
                    while (a_tags.count(new_tag) || b_tags.count(new_tag))
                        new_tag++;
                    b_tags[new_tag] = btgv;
                    b_tags.erase(atg.first);
                    for (int i = (int)tensors.size(); i < (int)xtensors.size();
                         i++)
                        if (xtensors[i].get_spin_tag() == atg.first)
                            xtensors[i].set_spin_tag(new_tag);
                }
            }
        for (auto &btg : b_tags)
            if (a_tags.count(btg.first)) {
                int btgv = btg.second;
                int atgv = a_tags.at(btg.first);
                if (atgv > 1) {
                    while (a_tags.count(new_tag) || b_tags.count(new_tag))
                        new_tag++;
                    a_tags[new_tag] = atgv;
                    a_tags.erase(btg.first);
                    for (int i = 0; i < (int)tensors.size(); i++)
                        if (xtensors[i].get_spin_tag() == btg.first)
                            xtensors[i].set_spin_tag(new_tag);
                }
            }
        xctr_indices.clear();
        for (auto &wi : ctr_indices)
            if (mp_idxs.count(wi.name) && xa_rep.count(wi.name) &&
                !xc_rep.count(wi.name))
                xctr_indices.insert(WickIndex(mp_idxs[wi.name], wi.types));
            else
                xctr_indices.insert(wi);
        for (auto &wi : other.ctr_indices)
            if (mp_idxs.count(wi.name) && xb_rep.count(wi.name))
                xctr_indices.insert(WickIndex(mp_idxs[wi.name], wi.types));
            else
                xctr_indices.insert(wi);
        return WickString(xtensors, xctr_indices, factor * other.factor);
    }
    WickString operator*(double d) const noexcept {
        return WickString(tensors, ctr_indices, factor * d);
    }
    WickString abs() const { return WickString(tensors, ctr_indices); }
    bool group_less(const WickString &other) const noexcept {
        const static vector<WickTensorTypes> wtts = {
            WickTensorTypes::KroneckerDelta, WickTensorTypes::Tensor,
            WickTensorTypes::CreationOperator, WickTensorTypes::DestroyOperator,
            WickTensorTypes::SpinFreeOperator};
        if (tensors.size() != other.tensors.size())
            return tensors.size() < other.tensors.size();
        if (ctr_indices.size() != other.ctr_indices.size())
            return ctr_indices.size() < other.ctr_indices.size();
        for (auto &wtt : wtts) {
            int xi = 0, xii = 0, xj = 0, xjj = 0;
            for (auto &wt : tensors)
                if (wt.type == wtt)
                    xi += (int)wt.indices.size(), xii++;
            for (auto &wt : other.tensors)
                if (wt.type == wtt)
                    xj += (int)wt.indices.size(), xjj++;
            if (xi != xj)
                return xi < xj;
            if (xii != xjj)
                return xii < xjj;
        }
        return false;
    }
    bool has_inactive_ops() const {
        for (auto &wt : tensors)
            if (wt.type == WickTensorTypes::SpinFreeOperator ||
                wt.type == WickTensorTypes::CreationOperator ||
                wt.type == WickTensorTypes::DestroyOperator)
                for (auto &wi : wt.indices)
                    if ((uint8_t)(wi.types & WickIndexTypes::Inactive))
                        return true;
        return false;
    }
    bool has_external_ops() const {
        for (auto &wt : tensors)
            if (wt.type == WickTensorTypes::SpinFreeOperator ||
                wt.type == WickTensorTypes::CreationOperator ||
                wt.type == WickTensorTypes::DestroyOperator)
                for (auto &wi : wt.indices)
                    if ((uint8_t)(wi.types & WickIndexTypes::External))
                        return true;
        return false;
    }
    WickString simple_sort() const {
        vector<WickTensor> cd_tensors, ot_tensors;
        double xfactor = factor;
        map<WickIndex, int> ctr_map;
        int ip = 0;
        for (auto &wt : tensors)
            if (wt.type == WickTensorTypes::KroneckerDelta ||
                wt.type == WickTensorTypes::Tensor)
                ot_tensors.push_back(wt.sort(xfactor));
            else
                cd_tensors.push_back(wt.sort(xfactor));
        vector<WickTensor> f_tensors = ot_tensors;
        f_tensors.insert(f_tensors.end(), cd_tensors.begin(), cd_tensors.end());
        std::sort(f_tensors.begin(), f_tensors.begin() + ot_tensors.size());
        return WickString(f_tensors, ctr_indices, xfactor);
    }
    WickString quick_sort() const {
        vector<WickTensor> cd_tensors, ot_tensors;
        double xfactor = factor;
        for (auto &wt : tensors)
            if (wt.type == WickTensorTypes::KroneckerDelta ||
                wt.type == WickTensorTypes::Tensor)
                ot_tensors.push_back(wt.sort(xfactor));
            else
                cd_tensors.push_back(wt.sort(xfactor));
        std::sort(ot_tensors.begin(), ot_tensors.end(),
                  [](const WickTensor &a, const WickTensor &b) {
                      return a.name != b.name
                                 ? a.name < b.name
                                 : a.indices.size() < b.indices.size();
                  });
        // ot_tensor_groups is the accumulated count of size of each group
        // each group has the tensors with the same name
        vector<int> ot_tensor_groups;
        for (int i = 0; i < (int)ot_tensors.size(); i++)
            if (i == 0 || (ot_tensors[i].name != ot_tensors[i - 1].name ||
                           ot_tensors[i].indices.size() !=
                               ot_tensors[i - 1].indices.size()))
                ot_tensor_groups.push_back(i);
        ot_tensor_groups.push_back((int)ot_tensors.size());
        int kidx = 0;
        vector<WickTensor> ot_sorted(ot_tensors.size());
        vector<pair<map<WickIndex, int>, int>> ctr_maps = {
            make_pair(map<WickIndex, int>(), 1)};
        // loop over each group
        for (int ig = 0; ig < (int)ot_tensor_groups.size() - 1; ig++) {
            // wta has the same size as the group
            vector<int> wta(ot_tensor_groups[ig + 1] - ot_tensor_groups[ig]);
            for (int j = 0; j < (int)wta.size(); j++)
                wta[j] = ot_tensor_groups[ig] + j;
            // wtb is the already sorted tensors with abstract contracted
            // indices
            // wta are the original tensors, wtb are the sorted tensors
            WickTensor *wtb = ot_sorted.data() + ot_tensor_groups[ig];
            // we have j tensors with the same name
            // we need to determine the best order of these tensors
            for (int j = 0; j < (int)wta.size(); j++) {
                int jxx = -1, jixx = -1;
                // we try all tensors to be put in the jth position
                // the one with the smallest sorted index name will win
                // jxx is the index of the selected tensor in the original list
                // jixx / jidx / kidx denotes how many ctr_idx were consumed
                for (int k = j; k < (int)wta.size(); k++) {
                    int jidx = kidx;
                    wtb[k] =
                        ot_tensors[wta[k]].sort(ctr_indices, ctr_maps, jidx);
                    // if this one is smaller, use it
                    if (k == j || wtb[k].indices < wtb[j].indices)
                        wtb[j] = wtb[k], jxx = k, jixx = jidx;
                }
                // get original tensor in jxx of wta
                // make it to be the sorted tensor in j of wtb
                // but this time get all possible maps
                ctr_maps = ot_tensors[wta[jxx]].sort_gen_maps(
                    wtb[j], ctr_indices, ctr_maps, kidx);
                kidx = jixx;
                if (jxx != j)
                    swap(wta[jxx], wta[j]);
            }
        }
        // now all ot_tensors are already in ot_sorted
        // we add the remaining cd_tensors
        bool is_sf_cd = cd_tensors.size() != 0;
        for (auto &wt : cd_tensors)
            if (wt.get_spin_tag() == -1) {
                is_sf_cd = false;
                break;
            }
        if (is_sf_cd) {
            // sf_pri = map from undetermined ctr index to its appearance in
            // which group and appeared how many times apppear earlier and then
            // more times should have lower abstract index number
            map<WickIndex, vector<pair<int, int>>> sf_pri;
            // a list of undetermined ctr indices
            vector<WickIndex> sf_ctr_idx;
            // sf_idx_sorted = arg sort of sf_ctr_idx
            // sf_idx_group_mp = from cd_tensor index to group index
            vector<int> sf_idx_sorted, sf_idx_group_mp;
            // first determine group index
            for (int i = 0, k = 0; i < (int)cd_tensors.size(); i++) {
                if (i != 0 &&
                    (cd_tensors[i].name[0] != cd_tensors[i - 1].name[0] ||
                     cd_tensors[i].indices[0].types !=
                         cd_tensors[i - 1].indices[0].types))
                    k++;
                sf_idx_group_mp.push_back(k);
            }
            // fill sf_pri
            for (int i = 0; i < (int)cd_tensors.size(); i++) {
                auto &ix = cd_tensors[i].indices[0];
                if (ctr_indices.count(ix) && !ctr_maps[0].first.count(ix)) {
                    if (!sf_pri.count(ix)) {
                        sf_pri[ix].push_back(make_pair(sf_idx_group_mp[i], 1));
                        sf_ctr_idx.push_back(ix);
                    } else if (sf_pri.at(ix).back().first == sf_idx_group_mp[i])
                        sf_pri.at(ix).back().second++;
                    else
                        sf_pri.at(ix).push_back(
                            make_pair(sf_idx_group_mp[i], 1));
                }
            }
            // argsort sf_pri
            for (size_t i = 0; i < sf_ctr_idx.size(); i++)
                sf_idx_sorted[i] = (int)i;
            stable_sort(
                sf_idx_sorted.begin(), sf_idx_sorted.end(),
                [&sf_pri, &sf_ctr_idx](int i, int j) {
                    vector<pair<int, int>> &fi = sf_pri.at(sf_ctr_idx[i]);
                    vector<pair<int, int>> &fj = sf_pri.at(sf_ctr_idx[j]);
                    for (size_t k = 0; k < min(fi.size(), fj.size()); k++)
                        if (fi[k].first != fj[k].first)
                            return fi[k].first < fj[k].first;
                        else if (fi[k].second != fj[k].second)
                            return fi[k].second > fj[k].second;
                    return fi.size() > fj.size();
                });
            // determine abstract index based on sorted sf_pri
            int jidx = kidx;
            for (int i = 0; i < (int)sf_idx_sorted.size(); i++) {
                auto &ix = sf_ctr_idx[sf_idx_sorted[i]];
                ctr_maps[0].first[ix] = jidx;
            }
            // change indices in the cd_tensors
            for (int i = 0; i < (int)cd_tensors.size(); i++) {
                auto &wi = cd_tensors[i].indices[0];
                if (ctr_maps[0].first.count(wi))
                    wi.name = string(1, ctr_maps[0].first.at(wi) + '0');
            }
            // cd_tensor_groups is the accumulated count of size of each group
            // each group has the tensors with the same name and same sub_space
            vector<int> cd_tensor_groups;
            for (int i = 0; i < (int)cd_tensors.size(); i++)
                if (i == 0 ||
                    (cd_tensors[i].name[0] != cd_tensors[i - 1].name[0] ||
                     cd_tensors[i].indices[0].types !=
                         cd_tensors[i - 1].indices[0].types))
                    cd_tensor_groups.push_back(i);
            cd_tensor_groups.push_back((int)cd_tensors.size());
            // loop over each group
            uint8_t arg_sign = 0;
            int new_spin_tag = 0;
            map<int, int> spin_tag_mp;
            for (int ig = 0; ig < (int)cd_tensor_groups.size() - 1; ig++) {
                // wta has the same size as the group
                vector<int> arg_idx(cd_tensor_groups[ig + 1] -
                                    cd_tensor_groups[ig]);
                // arg sort within each group
                for (size_t i = 0; i < arg_idx.size(); i++)
                    arg_idx[i] = (int)i + cd_tensor_groups[ig];
                stable_sort(arg_idx.begin(), arg_idx.end(),
                            [&cd_tensors](int i, int j) {
                                return cd_tensors[i].indices <
                                       cd_tensors[j].indices;
                            });
                // fermion sign
                for (int i = 0; i < (int)arg_idx.size(); i++)
                    for (int j = i + 1; j < (int)arg_idx.size(); j++)
                        arg_sign ^= (arg_idx[j] < arg_idx[i]);
                // spin tag and push into sorted list
                for (int i = 0; i < (int)arg_idx.size(); i++) {
                    WickTensor &wt = cd_tensors[arg_idx[i]];
                    int old_spin_tag = wt.get_spin_tag();
                    if (!spin_tag_mp.count(old_spin_tag))
                        spin_tag_mp[old_spin_tag] = new_spin_tag++;
                    wt.set_spin_tag(spin_tag_mp.at(old_spin_tag));
                    ot_sorted.push_back(wt);
                }
            }
            kidx = jidx;
            ctr_maps[0].second =
                arg_sign ? -ctr_maps[0].second : ctr_maps[0].second;
        } else {
            for (auto &wt : cd_tensors) {
                int jidx = kidx;
                // now ctr_maps should contain all possible info to determine
                // the abstract index unless there is ctr_idx only in cd_tensors
                ot_sorted.push_back(wt.sort(ctr_indices, ctr_maps, kidx));
                ctr_maps = wt.sort_gen_maps(ot_sorted.back(), ctr_indices,
                                            ctr_maps, jidx);
            }
        }
        // this should always be true
        assert(kidx == (int)ctr_maps[0].first.size());
        set<WickIndex> xctr_idxs;
        for (auto wi : ctr_indices) {
            if (ctr_maps[0].first.count(wi))
                wi.name = string(1, ctr_maps[0].first.at(wi) + '0');
            xctr_idxs.insert(wi);
        }
        // here we assume that ctr_indices do not have more indices than all
        // indices in the tensors
        if (kidx != (int)ctr_indices.size()) {
            cout << "string = " << *this << endl;
            cout << "ctr indices = ";
            for (auto &x : xctr_idxs)
                cout << x << " ";
            cout << endl;
            cout << "used ctr indices = " << kidx << endl;
            cout << "sorted tensors = ";
            for (auto &x : ot_sorted)
                cout << x << " ";
            cout << endl;
            assert(false);
        }
        return WickString(ot_sorted, xctr_idxs, xfactor * ctr_maps[0].second);
    }
    WickString old_sort() const {
        vector<WickTensor> cd_tensors, ot_tensors;
        map<WickIndex, int> ctr_map;
        double xfactor = factor;
        int ip = 0;
        for (auto &wt : tensors)
            if (wt.type == WickTensorTypes::KroneckerDelta ||
                wt.type == WickTensorTypes::Tensor)
                ot_tensors.push_back(wt.sort(xfactor));
            else
                cd_tensors.push_back(wt.sort(xfactor));
        for (auto &wt : ot_tensors)
            for (auto &wi : wt.indices)
                if (ctr_indices.count(wi) && !ctr_map.count(wi))
                    ctr_map[wi] = ip++;
        for (auto &wt : cd_tensors)
            for (auto &wi : wt.indices)
                if (ctr_indices.count(wi) && !ctr_map.count(wi))
                    ctr_map[wi] = ip++;
        vector<WickTensor> f_tensors = ot_tensors;
        f_tensors.insert(f_tensors.end(), cd_tensors.begin(), cd_tensors.end());
        WickString ex(f_tensors, set<WickIndex>(), xfactor);
        for (auto &wt : ex.tensors) {
            for (auto &wi : wt.indices)
                if (ctr_indices.count(wi))
                    wi.name = string(1, ctr_map[wi] + '0');
            wt = wt.sort(ex.factor);
        }
        vector<WickIndex> ex_ctr(ctr_indices.begin(), ctr_indices.end());
        for (auto &wi : ex_ctr)
            wi.name = string(1, ctr_map[wi] + '0');
        std::sort(ex.tensors.begin(), ex.tensors.begin() + ot_tensors.size());
        vector<int> ip_map(ip);
        for (int i = 0; i < ip; i++)
            ip_map[i] = i;
        while (next_permutation(ip_map.begin(), ip_map.end())) {
            WickString ez(f_tensors, set<WickIndex>(), xfactor);
            for (auto &wt : ez.tensors) {
                for (auto &wi : wt.indices)
                    if (ctr_indices.count(wi))
                        wi.name = string(1, ip_map[ctr_map[wi]] + '0');
                wt = wt.sort(ez.factor);
            }
            std::sort(ez.tensors.begin(),
                      ez.tensors.begin() + ot_tensors.size());
            if (ez < ex) {
                ex = ez;
                ex_ctr =
                    vector<WickIndex>(ctr_indices.begin(), ctr_indices.end());
                for (auto &wi : ex_ctr)
                    wi.name = string(1, ip_map[ctr_map[wi]] + '0');
            }
        }
        return WickString(ex.tensors,
                          set<WickIndex>(ex_ctr.begin(), ex_ctr.end()),
                          ex.factor);
    }
    friend ostream &operator<<(ostream &os, const WickString &ws) {
        os << "(" << fixed << setprecision(10) << setw(16) << ws.factor << ") ";
        if (ws.ctr_indices.size() != 0) {
            string d = " ";
            if (all_of(ws.ctr_indices.begin(), ws.ctr_indices.end(),
                       [](const WickIndex &idx) { return idx.is_short(); }))
                d = "";
            os << "SUM <" << d;
            for (auto &ci : ws.ctr_indices)
                os << ci << d;
            if (any_of(ws.ctr_indices.begin(), ws.ctr_indices.end(),
                       [](const WickIndex &wi) { return wi.has_types(); })) {
                os << "|";
                for (auto &ci : ws.ctr_indices)
                    os << to_str(ci.types)
                       << (to_str(ci.types).length() > 1 ? " " : "");
            }
            os << "> ";
        }
        for (int i = 0; i < (int)ws.tensors.size(); i++)
            os << ws.tensors[i] << (i == (int)ws.tensors.size() - 1 ? "" : " ");
        return os;
    }
    WickString simplify_delta() const {
        vector<WickTensor> xtensors = tensors;
        set<WickIndex> xctr_indices = ctr_indices;
        double xfactor = factor;
        vector<int> xidxs;
        for (int i = 0; i < (int)xtensors.size(); i++)
            if (xtensors[i].type == WickTensorTypes::KroneckerDelta) {
                const WickIndex &ia = xtensors[i].indices[0],
                                &ib = xtensors[i].indices[1];
                if (ia != ib) {
                    if ((ia.types != WickIndexTypes::None ||
                         ib.types != WickIndexTypes::None) &&
                        ((((ia.types & (~WickIndexTypes::AlphaBeta)) !=
                               WickIndexTypes::None ||
                           (ib.types & (~WickIndexTypes::AlphaBeta)) !=
                               WickIndexTypes::None) &&
                          ((ia.types & ib.types) &
                           (~WickIndexTypes::AlphaBeta)) ==
                              WickIndexTypes::None) ||
                         (ia.types & WickIndexTypes::AlphaBeta) !=
                             (ib.types & WickIndexTypes::AlphaBeta)))
                        xfactor = 0;
                    else if (!xctr_indices.count(ia) &&
                             !xctr_indices.count(ib)) {
                        bool found = false;
                        for (int j = 0; j < (int)xidxs.size() && !found; j++)
                            if (xtensors[xidxs[j]].type ==
                                    WickTensorTypes::KroneckerDelta &&
                                ((xtensors[xidxs[j]].indices[0] == ia &&
                                  xtensors[xidxs[j]].indices[1] == ib) ||
                                 (xtensors[xidxs[j]].indices[0] == ib &&
                                  xtensors[xidxs[j]].indices[1] == ia)))
                                found = true;
                        if (!found) {
                            xidxs.push_back(i);
                            WickIndex ic = min(ia, ib);
                            for (int j = 0; j < (int)xtensors.size(); j++)
                                if (j != i)
                                    for (int k = 0;
                                         k < (int)xtensors[j].indices.size();
                                         k++)
                                        if (xtensors[j].indices[k] == ia ||
                                            xtensors[j].indices[k] == ib)
                                            xtensors[j].indices[k] = ic;
                        }
                    } else {
                        WickIndex ic;
                        if (xctr_indices.count(ia)) {
                            ic = WickIndex(ib.name, ia.types & ib.types);
                            xctr_indices.erase(ia);
                        } else {
                            ic = WickIndex(ia.name, ia.types & ib.types);
                            xctr_indices.erase(ib);
                        }
                        for (int j = 0; j < (int)xtensors.size(); j++)
                            if (j != i)
                                for (int k = 0;
                                     k < (int)xtensors[j].indices.size(); k++)
                                    if (xtensors[j].indices[k] == ia ||
                                        xtensors[j].indices[k] == ib)
                                        xtensors[j].indices[k] = ic;
                    }
                }
            } else
                xidxs.push_back(i);
        for (int i = 0; i < (int)xidxs.size(); i++)
            xtensors[i] = xtensors[xidxs[i]];
        xtensors.resize(xidxs.size());
        return WickString(xtensors, xctr_indices, xfactor);
    }
    void save(ostream &ofs) const {
        size_t ltensors = (size_t)tensors.size();
        ofs.write((char *)&ltensors, sizeof(ltensors));
        for (size_t i = 0; i < ltensors; i++)
            tensors[i].save(ofs);
        size_t lctr_indices = (size_t)ctr_indices.size();
        ofs.write((char *)&lctr_indices, sizeof(lctr_indices));
        for (const auto &wi : ctr_indices)
            wi.save(ofs);
        ofs.write((char *)&factor, sizeof(factor));
    }
    void load(istream &ifs) {
        size_t ltensors = 0;
        ifs.read((char *)&ltensors, sizeof(ltensors));
        tensors.resize(ltensors);
        for (size_t i = 0; i < ltensors; i++)
            tensors[i].load(ifs);
        size_t lctr_indices = 0;
        ifs.read((char *)&lctr_indices, sizeof(lctr_indices));
        for (size_t i = 0; i < lctr_indices; i++) {
            WickIndex wi;
            wi.load(ifs);
            ctr_indices.insert(wi);
        }
        ifs.read((char *)&factor, sizeof(factor));
    }
};

struct WickExpr {
    vector<WickString> terms;
    WickExpr() {}
    explicit WickExpr(const WickString &term)
        : terms(vector<WickString>{term}) {}
    explicit WickExpr(const vector<WickString> &terms) : terms(terms) {}
    bool operator==(const WickExpr &other) const noexcept {
        return terms == other.terms;
    }
    bool operator!=(const WickExpr &other) const noexcept {
        return terms != other.terms;
    }
    bool operator<(const WickExpr &other) const noexcept {
        return terms < other.terms;
    }
    WickExpr operator*(const WickExpr &other) const noexcept {
        vector<WickString> xterms;
        xterms.reserve(terms.size() * other.terms.size());
        for (auto &ta : terms)
            for (auto &tb : other.terms)
                xterms.push_back(ta * tb);
        return WickExpr(xterms);
    }
    WickExpr operator+(const WickExpr &other) const noexcept {
        vector<WickString> xterms = terms;
        xterms.insert(xterms.end(), other.terms.begin(), other.terms.end());
        return WickExpr(xterms);
    }
    WickExpr operator-(const WickExpr &other) const noexcept {
        vector<WickString> xterms = terms;
        WickExpr mx = other * (-1.0);
        xterms.insert(xterms.end(), mx.terms.begin(), mx.terms.end());
        return WickExpr(xterms);
    }
    WickExpr operator*(double d) const noexcept {
        vector<WickString> xterms = terms;
        for (auto &term : xterms)
            term = term * d;
        return WickExpr(xterms);
    }
    friend ostream &operator<<(ostream &os, const WickExpr &we) {
        os << "EXPR /" << we.terms.size() << "/";
        if (we.terms.size() != 0)
            os << endl;
        for (int i = 0; i < (int)we.terms.size(); i++)
            os << we.terms[i] << endl;
        return os;
    }
    string to_einsum(const WickTensor &x, bool first_eq = false,
                     const string &intermediate_name = "") const {
        stringstream ss;
        bool first = true;
        bool has_br = false;
        for (auto &term : terms) {
            set<WickIndex> gstr;
            for (auto &wt : term.tensors)
                for (auto &wi : wt.indices)
                    gstr.insert(wi);
            int n_br = 0;
            for (auto &wi : x.indices)
                n_br += !gstr.count(wi);
            if ((has_br = (n_br > 0)))
                break;
        }
        for (auto &term : terms) {
            map<WickIndex, string> mp;
            set<string> pstr, gstr;
            for (int i = 0; i < (int)term.tensors.size(); i++)
                for (auto &wi : term.tensors[i].indices)
                    if (!term.ctr_indices.count(wi) && !mp.count(wi)) {
                        string x = wi.name;
                        while (pstr.count(x))
                            x[0]++;
                        mp[wi] = x, pstr.insert(x);
                    }
            for (int i = 0; i < (int)term.tensors.size(); i++)
                for (auto &wi : term.tensors[i].indices)
                    if (!mp.count(wi)) {
                        string x = wi.name;
                        while (pstr.count(x))
                            x[0]++;
                        mp[wi] = x, pstr.insert(x);
                    }
            for (auto &wi : x.indices)
                if (!mp.count(wi)) {
                    string x = wi.name;
                    while (pstr.count(x))
                        x[0]++;
                    mp[wi] = x, pstr.insert(x);
                }
            if (first_eq)
                ss << x.name << " = ";
            else if (!has_br || intermediate_name == "" ||
                     x.name.substr(0, intermediate_name.length()) !=
                         intermediate_name)
                ss << x.name << " += ";
            else
                ss << x.name << " = " << x.name << " + ";
            first_eq = false;
            int n_const = 0;
            for (auto &wt : term.tensors)
                if (wt.indices.size() == 0)
                    n_const++;
            if (term.tensors.size() == 0 || n_const == term.tensors.size()) {
                if (n_const == 0)
                    ss << term.factor << "\n";
                else {
                    if (term.factor != 1.0)
                        ss << term.factor << " * ";
                    int i_const = 0;
                    for (auto &wt : term.tensors)
                        if (wt.indices.size() == 0)
                            ss << wt.name
                               << (++i_const == n_const ? "\n" : " * ");
                }
                continue;
            }
            if (term.factor != 1.0)
                ss << term.factor << " * ";
            for (auto &wt : term.tensors)
                if (wt.indices.size() == 0)
                    ss << wt.name << " * ";
            ss << "np.einsum('";
            first = false;
            for (auto &wt : term.tensors) {
                if (wt.indices.size() == 0)
                    continue;
                if (first)
                    ss << ",";
                for (auto &wi : wt.indices)
                    ss << mp[wi], gstr.insert(mp[wi]);
                first = true;
            }
            int n_br = 0;
            for (auto &wi : x.indices)
                n_br += !gstr.count(mp[wi]);
            if (n_br != 0) {
                ss << ",";
                for (auto &wi : x.indices)
                    if (!gstr.count(mp[wi]))
                        ss << mp[wi];
            }
            ss << "->";
            for (auto &wi : x.indices)
                ss << mp[wi];
            ss << "'";
            for (auto &wt : term.tensors) {
                if (wt.indices.size() == 0)
                    continue;
                ss << ", " << wt.name;
                if (wt.type == WickTensorTypes::KroneckerDelta ||
                    wt.type == WickTensorTypes::Tensor &&
                        (intermediate_name == "" ||
                         wt.name.substr(0, intermediate_name.length()) !=
                             intermediate_name))
                    for (auto &wi : wt.indices)
                        ss << to_str(wi.types);
            }
            if (n_br != 0)
                ss << ", ident" << n_br;
            ss << ", optimize=True)\n";
        }
        return ss.str();
    }
    static string to_einsum_add_indent(const string &x, int indent = 4) {
        stringstream ss;
        for (size_t i = 0, j = 0; j != string::npos; i = j + 1) {
            ss << string(indent, ' ');
            j = x.find_first_of("\n", i);
            if (j > i)
                ss << x.substr(i, j - i);
            ss << endl;
        }
        return ss.str();
    }
    static WickExpr
    parse(const string &expr,
          const map<WickIndexTypes, set<WickIndex>> &idx_map,
          const map<pair<string, int>, vector<WickPermutation>> &perm_map =
              map<pair<string, int>, vector<WickPermutation>>()) {
        vector<WickString> terms;
        stringstream exx;
        for (int ic = 0; ic < (int)expr.length(); ic++) {
            auto &c = expr[ic];
            if (c == '+' || c == '-')
                exx << "\n" << c;
            else if (c == '(') {
                exx << "\n" << c;
                for (ic++; ic < (int)expr.length() && expr[ic] != ')'; ic++)
                    if (expr[ic] != ' ')
                        exx << expr[ic];
                ic--;
            } else
                exx << c;
        }
        string tex_expr = exx.str();
        size_t index = tex_expr.find_first_of("\n\r", 0);
        size_t last = 0;
        while (index != string::npos) {
            if (index > last && tex_expr.substr(last, index - last) !=
                                    string(index - last, ' '))
                terms.push_back(WickString::parse(
                    tex_expr.substr(last, index - last), idx_map, perm_map));
            last = index + 1;
            index = tex_expr.find_first_of("\n\r", last);
        }
        if (tex_expr.length() > last &&
            tex_expr.substr(last, tex_expr.length() - last) !=
                string(tex_expr.length() - last, ' '))
            terms.push_back(WickString::parse(
                tex_expr.substr(last, tex_expr.length() - last), idx_map,
                perm_map));
        return WickExpr(terms);
    }
    static pair<WickTensor, WickExpr>
    parse_def(const string &tex_expr,
              const map<WickIndexTypes, set<WickIndex>> &idx_map,
              const map<pair<string, int>, vector<WickPermutation>> &perm_map =
                  map<pair<string, int>, vector<WickPermutation>>()) {
        size_t index = tex_expr.find_first_of("=", 0);
        assert(index != string::npos);
        WickTensor name =
            WickTensor::parse(tex_expr.substr(0, index), idx_map, perm_map);
        WickExpr expr =
            WickExpr::parse(tex_expr.substr(index + 1), idx_map, perm_map);
        return make_pair(name, expr);
    }
    WickExpr
    substitute(const map<string, pair<WickTensor, WickExpr>> &defs) const {
        WickExpr r;
        map<string, pair<WickTensor, vector<WickString>>> xdefs;
        for (auto &dd : defs)
            xdefs[dd.first] =
                make_pair(dd.second.first, dd.second.second.terms);
        for (auto &ws : terms) {
            vector<WickString> rws = ws.substitute(xdefs);
            r.terms.insert(r.terms.end(), rws.begin(), rws.end());
        }
        return r;
    }
    WickExpr index_map(const map<string, string> &maps) const {
        WickExpr r = *this;
        for (auto &ws : r.terms)
            ws = ws.index_map(maps);
        return r;
    }
    static WickExpr split_index_types(const WickString &x) {
        vector<WickIndex> vidxs(x.ctr_indices.begin(), x.ctr_indices.end());
        vector<vector<WickIndex>> xctr_idxs = {vidxs};
        WickIndexTypes check_mask =
            WickIndexTypes::Inactive | WickIndexTypes::Active |
            WickIndexTypes::Single | WickIndexTypes::External;
        vector<WickIndexTypes> check_types = {
            WickIndexTypes::Inactive, WickIndexTypes::Active,
            WickIndexTypes::Single, WickIndexTypes::External};
        for (int i = 0; i < (int)vidxs.size(); i++) {
            int k = 0, nk = (int)xctr_idxs.size();
            for (int j = 0; j < (int)check_types.size(); j++)
                if ((vidxs[i].types & check_types[j]) != WickIndexTypes::None &&
                    (vidxs[i].types & check_mask) != check_types[j]) {
                    if (k != 0) {
                        xctr_idxs.reserve(xctr_idxs.size() + nk);
                        for (int l = 0; l < nk; l++)
                            xctr_idxs.push_back(xctr_idxs[l]);
                    }
                    for (int l = 0; l < nk; l++) {
                        xctr_idxs[k * nk + l][i].types =
                            xctr_idxs[k * nk + l][i].types & (~check_mask);
                        xctr_idxs[k * nk + l][i].types =
                            xctr_idxs[k * nk + l][i].types | check_types[j];
                    }
                    k++;
                }
        }
        WickExpr r;
        for (int i = 0; i < (int)xctr_idxs.size(); i++) {
            r.terms.push_back(WickString(
                x.tensors,
                set<WickIndex>(xctr_idxs[i].begin(), xctr_idxs[i].end()),
                x.factor));
            for (auto &wt : r.terms.back().tensors) {
                for (auto &wi : wt.indices)
                    for (int j = 0; j < (int)xctr_idxs[i].size(); j++) {
                        auto &wii = xctr_idxs[i][j];
                        if (wi == vidxs.at(j) &&
                            (wi.types & wii.types) != WickIndexTypes::None)
                            wi = wii;
                    }
                if (wt.perms.size() == 0)
                    r.terms.back().factor = 0;
            }
            if (r.terms.back().factor == 0)
                r.terms.pop_back();
        }
        return r;
    }
    WickExpr split_index_types() const {
        WickExpr r;
        for (auto &term : terms) {
            WickExpr rr = split_index_types(term);
            r.terms.insert(r.terms.end(), rr.terms.begin(), rr.terms.end());
        }
        return r;
    }
    WickExpr expand(int max_unctr = -1, bool no_ctr = false,
                    bool no_unctr_sf_inact = true) const {
        return split_index_types().normal_order_impl(max_unctr, no_ctr,
                                                     no_unctr_sf_inact);
    }
    WickExpr normal_order_impl(int max_unctr = -1, bool no_ctr = false,
                               bool no_unctr_sf_inact = true) const {
        int ntg = threading->activate_global();
        vector<WickExpr> r(ntg);
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int k = 0; k < (int)terms.size(); k++) {
            int tid = threading->get_thread_id();
            WickExpr rr = normal_order_impl_new(terms[k], max_unctr, no_ctr,
                                                no_unctr_sf_inact);
            r[tid].terms.insert(r[tid].terms.end(), rr.terms.begin(),
                                rr.terms.end());
        }
        threading->activate_normal();
        WickExpr rx;
        size_t nr = 0;
        for (auto &rr : r)
            nr += rr.terms.size();
        rx.terms.reserve(nr);
        for (auto &rr : r)
            rx.terms.insert(rx.terms.end(), rr.terms.begin(), rr.terms.end());
        return rx;
    }
    static WickExpr normal_order_impl(const WickString &x, int max_unctr = -1,
                                      bool no_ctr = false) {
        WickExpr r;
        bool cd_type = any_of(
            x.tensors.begin(), x.tensors.end(), [](const WickTensor &wt) {
                return wt.type == WickTensorTypes::CreationOperator ||
                       wt.type == WickTensorTypes::DestroyOperator;
            });
        bool sf_type = any_of(
            x.tensors.begin(), x.tensors.end(), [](const WickTensor &wt) {
                return wt.type == WickTensorTypes::SpinFreeOperator;
            });
        assert(!cd_type || !sf_type);
        vector<WickTensor> cd_tensors, ot_tensors;
        vector<int> cd_idx_map;
        cd_tensors.reserve(x.tensors.size());
        ot_tensors.reserve(x.tensors.size());
        for (auto &wt : x.tensors)
            if (wt.type == WickTensorTypes::CreationOperator ||
                wt.type == WickTensorTypes::DestroyOperator)
                cd_tensors.push_back(wt);
            else if (wt.type == WickTensorTypes::SpinFreeOperator) {
                int sf_n = (int)wt.indices.size() / 2;
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::cre(wt.indices[i]));
                    cd_idx_map.push_back((int)cd_idx_map.size() + sf_n);
                }
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::des(wt.indices[i + sf_n]));
                    cd_idx_map.push_back((int)cd_idx_map.size() - sf_n);
                }
            } else
                ot_tensors.push_back(wt);
        int ot_count = (int)ot_tensors.size();
        // all possible pairs
        vector<pair<int, int>> ctr_idxs;
        // starting index in ctr_idxs for the given first index in the pair
        vector<int> ctr_cd_idxs(cd_tensors.size() + 1);
        for (int i = 0; i < (int)cd_tensors.size(); i++) {
            ctr_cd_idxs[i] = (int)ctr_idxs.size();
            if (sf_type) {
                for (int j = i + 1; j < (int)cd_tensors.size(); j++)
                    if (cd_tensors[j].type < cd_tensors[i].type)
                        ctr_idxs.push_back(make_pair(i, j));
            } else {
                for (int j = i + 1; j < (int)cd_tensors.size(); j++)
                    if (cd_tensors[i].type != cd_tensors[j].type &&
                        cd_tensors[j] < cd_tensors[i])
                        ctr_idxs.push_back(make_pair(i, j));
            }
        }
        ctr_cd_idxs[cd_tensors.size()] = (int)ctr_idxs.size();
        vector<pair<int, int>> que;
        vector<pair<int, int>> cur_idxs(cd_tensors.size());
        vector<int8_t> cur_idxs_mask(cd_tensors.size(), 0);
        vector<int> tensor_idxs(cd_tensors.size());
        vector<int> cd_idx_map_rev(cd_tensors.size());
        vector<int> acc_sign(cd_tensors.size() + 1);
        if (max_unctr != 0 || cd_tensors.size() % 2 == 0) {
            que.push_back(make_pair(-1, -1));
            acc_sign[0] = 0; // even
            for (int i = 0; i < (int)cd_tensors.size(); i++)
                tensor_idxs[i] = i;
            if (sf_type) {
                stable_sort(tensor_idxs.begin(), tensor_idxs.end(),
                            [&cd_tensors](int i, int j) {
                                return cd_tensors[i].type < cd_tensors[j].type;
                            });
                assert(all_of(tensor_idxs.begin(),
                              tensor_idxs.begin() + tensor_idxs.size() / 2,
                              [&cd_tensors](int i) {
                                  return cd_tensors[i].type ==
                                         WickTensorTypes::CreationOperator;
                              }));
            } else {
                // sign for reordering tensors to the normal order
                for (int i = 0; i < (int)cd_tensors.size(); i++)
                    for (int j = i + 1; j < (int)cd_tensors.size(); j++)
                        acc_sign[0] ^= (int)(cd_tensors[j] < cd_tensors[i]);
                // arg sort of tensors in the normal order
                stable_sort(tensor_idxs.begin(), tensor_idxs.end(),
                            [&cd_tensors](int i, int j) {
                                return cd_tensors[i] < cd_tensors[j];
                            });
            }
        }
        // depth-first tree traverse
        while (!que.empty()) {
            int l = que.back().first, j = que.back().second, k = 0;
            que.pop_back();
            int a, b, c, d;
            if (l != -1) {
                cur_idxs[l] = ctr_idxs[j];
                k = ctr_cd_idxs[ctr_idxs[j].first + 1];
            }
            acc_sign[l + 2] = acc_sign[l + 1];
            ot_tensors.resize(ot_count + l + 1);
            memset(cur_idxs_mask.data(), 0,
                   sizeof(int8_t) * cur_idxs_mask.size());
            if (sf_type)
                memcpy(cd_idx_map_rev.data(), cd_idx_map.data(),
                       sizeof(int) * cd_idx_map.size());
            if (l != -1) {
                tie(c, d) = cur_idxs[l];
                bool skip = false;
                acc_sign[l + 2] ^= ((c ^ d) & 1) ^ 1;
                // add contraction crossing sign from c/d
                for (int i = 0; i < l && !skip; i++) {
                    tie(a, b) = cur_idxs[i];
                    skip |= (b == d || b == c || a == d);
                    cur_idxs_mask[a] = cur_idxs_mask[b] = 1;
                    acc_sign[l + 2] ^= ((a < c && b > c && b < d) ||
                                        (a > c && a < d && b > d));
                }
                if (skip)
                    continue;
                cur_idxs_mask[c] = cur_idxs_mask[d] = 1;
                if (sf_type) {
                    for (int i = 0; i < l; i++) {
                        tie(a, b) = cur_idxs[i];
                        cd_idx_map_rev[cd_idx_map_rev[a]] = cd_idx_map_rev[b];
                        cd_idx_map_rev[cd_idx_map_rev[b]] = cd_idx_map_rev[a];
                    }
                    cd_idx_map_rev[cd_idx_map_rev[c]] = cd_idx_map_rev[d];
                    cd_idx_map_rev[cd_idx_map_rev[d]] = cd_idx_map_rev[c];
                    acc_sign[l + 2] = 0;
                } else {
                    // remove tensor reorder sign for c/d
                    acc_sign[l + 2] ^= (int)(cd_tensors[d] < cd_tensors[c]);
                    for (int i = 0; i < (int)cd_tensors.size(); i++)
                        if (!cur_idxs_mask[i]) {
                            acc_sign[l + 2] ^= (int)(cd_tensors[max(c, i)] <
                                                     cd_tensors[min(c, i)]);
                            acc_sign[l + 2] ^= (int)(cd_tensors[max(d, i)] <
                                                     cd_tensors[min(d, i)]);
                        }
                }
                ot_tensors[ot_count + l] =
                    WickTensor::kronecker_delta(vector<WickIndex>{
                        cd_tensors[c].indices[0], cd_tensors[d].indices[0]});
            }
            // push next contraction order to queue
            if (!no_ctr)
                for (; k < (int)ctr_idxs.size(); k++)
                    que.push_back(make_pair(l + 1, k));
            if (max_unctr != -1 && cd_tensors.size() - (l + l + 2) > max_unctr)
                continue;
            if (sf_type) {
                int sf_n = (int)cd_tensors.size() / 2, tn = sf_n - l - 1;
                vector<WickIndex> wis(tn * 2);
                for (int i = 0, k = 0; i < (int)tensor_idxs.size(); i++)
                    if (!cur_idxs_mask[tensor_idxs[i]] &&
                        cd_tensors[tensor_idxs[i]].type ==
                            WickTensorTypes::CreationOperator) {
                        wis[k] = cd_tensors[tensor_idxs[i]].indices[0];
                        wis[k + tn] = cd_tensors[cd_idx_map_rev[tensor_idxs[i]]]
                                          .indices[0];
                        k++;
                    }
                ot_tensors.push_back(WickTensor::spin_free(wis));
            } else {
                for (int i = 0; i < (int)tensor_idxs.size(); i++)
                    if (!cur_idxs_mask[tensor_idxs[i]])
                        ot_tensors.push_back(cd_tensors[tensor_idxs[i]]);
            }
            r.terms.push_back(
                WickString(ot_tensors, x.ctr_indices,
                           acc_sign[l + 2] ? -x.factor : x.factor));
        }
        return r;
    }
    // no_unctr_sf_inact: if false, will keep spin-free operators with
    // destroy operators before creation operators (will then not using E1/E2)
    // note that under this case there can be extra terms due to the
    // ambiguity in the definition of normal order for spin-free operators
    static WickExpr normal_order_impl_new(const WickString &x,
                                          int max_unctr = -1,
                                          bool no_ctr = false,
                                          bool no_unctr_sf_inact = true) {
        WickExpr r;
        bool cd_type = any_of(
            x.tensors.begin(), x.tensors.end(), [](const WickTensor &wt) {
                return wt.type == WickTensorTypes::CreationOperator ||
                       wt.type == WickTensorTypes::DestroyOperator;
            });
        bool sf_type = any_of(
            x.tensors.begin(), x.tensors.end(), [](const WickTensor &wt) {
                return wt.type == WickTensorTypes::SpinFreeOperator;
            });
        sf_type = sf_type || any_of(x.tensors.begin(), x.tensors.end(),
                                    [](const WickTensor &wt) {
                                        return wt.get_spin_tag() != -1;
                                    });
        vector<WickTensor> cd_tensors, ot_tensors;
        vector<int> cd_idx_map, n_inactive_idxs_a, n_inactive_idxs_b;
        vector<uint8_t> single_idxs_mask, cur_single_idxs_mask;
        int init_sign = 0, final_sign = 0;
        cd_tensors.reserve(x.tensors.size());
        ot_tensors.reserve(x.tensors.size());
        map<int, vector<int>> cd_spin_map;
        for (auto &wt : x.tensors)
            if (wt.type == WickTensorTypes::CreationOperator ||
                wt.type == WickTensorTypes::DestroyOperator) {
                cd_tensors.push_back(wt);
                if (sf_type) {
                    cd_idx_map.push_back(-1);
                    cd_spin_map[wt.get_spin_tag()].push_back(
                        (int)cd_tensors.size() - 1);
                }
            } else if (wt.type == WickTensorTypes::SpinFreeOperator) {
                int sf_n = (int)wt.indices.size() / 2;
                // sign from reverse destroy operator
                init_sign ^= ((sf_n - 1) & 1) ^ (((sf_n - 1) & 2) >> 1);
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::cre(wt.indices[i]));
                    cd_idx_map.push_back((int)cd_idx_map.size() + sf_n);
                }
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::des(wt.indices[i + sf_n]));
                    cd_idx_map.push_back((int)cd_idx_map.size() - sf_n);
                }
            } else
                ot_tensors.push_back(wt);
        if (cd_type && sf_type) {
            for (auto &x : cd_spin_map) {
                assert(x.second.size() == 2);
                cd_idx_map[x.second[0]] = x.second[1];
                cd_idx_map[x.second[1]] = x.second[0];
            }
        }
        if (sf_type) {
            int max_tag = (int)cd_tensors.size() + 1, spin_tag = 0;
            for (int i = 0; i < (int)cd_tensors.size(); i++)
                cd_tensors[i].set_spin_tag(max_tag);
            vector<pair<int, int>> spin_tag_cd_idx;
            for (int i = 0; i < (int)cd_tensors.size(); i++)
                if (cd_tensors[i].get_spin_tag() == max_tag) {
                    cd_tensors[i].set_spin_tag(spin_tag);
                    cd_tensors[cd_idx_map[i]].set_spin_tag(spin_tag);
                    if (cd_tensors[i].name[0] == 'C')
                        spin_tag_cd_idx.push_back(make_pair(i, cd_idx_map[i]));
                    else
                        spin_tag_cd_idx.push_back(make_pair(cd_idx_map[i], i));
                    spin_tag++;
                }
            vector<int> spin_tag_idx(spin_tag), rev_spin_tag_idx(spin_tag);
            for (int i = 0; i < spin_tag; i++)
                spin_tag_idx[i] = i;
            stable_sort(
                spin_tag_idx.begin(), spin_tag_idx.end(),
                [&spin_tag_cd_idx, &cd_tensors](int i, int j) {
                    return cd_tensors[spin_tag_cd_idx[i].first].indices !=
                                   cd_tensors[spin_tag_cd_idx[j].first].indices
                               ? cd_tensors[spin_tag_cd_idx[i].first].indices <
                                     cd_tensors[spin_tag_cd_idx[j].first]
                                         .indices
                               : cd_tensors[spin_tag_cd_idx[i].second].indices <
                                     cd_tensors[spin_tag_cd_idx[j].second]
                                         .indices;
                });
            for (int i = 0; i < spin_tag; i++)
                rev_spin_tag_idx[spin_tag_idx[i]] = i;
            for (int i = 0; i < (int)cd_tensors.size(); i++)
                cd_tensors[i].set_spin_tag(
                    rev_spin_tag_idx[cd_tensors[i].get_spin_tag()]);
        }
        int ot_count = (int)ot_tensors.size();
        // all possible pairs
        vector<pair<int, int>> ctr_idxs;
        // starting index in ctr_idxs for the given first index in the pair
        vector<int> ctr_cd_idxs(cd_tensors.size() + 1);
        if (sf_type) {
            n_inactive_idxs_a.resize(cd_tensors.size() + 1, 0);
            n_inactive_idxs_b.resize(cd_tensors.size() + 1, 0);
            single_idxs_mask.resize(cd_tensors.size());
            cur_single_idxs_mask.resize(cd_tensors.size());
        }
        for (int i = 0; i < (int)cd_tensors.size(); i++) {
            ctr_cd_idxs[i] = (int)ctr_idxs.size();
            if (sf_type) {
                for (int j = i + 1; j < (int)cd_tensors.size(); j++) {
                    const WickIndexTypes tij = cd_tensors[i].indices[0].types &
                                               cd_tensors[j].indices[0].types;
                    if (tij != WickIndexTypes::None &&
                        cd_tensors[i].type != cd_tensors[j].type &&
                        cd_tensors[j] < cd_tensors[i]) {
                        ctr_idxs.push_back(make_pair(i, j));
                        if ((tij & WickIndexTypes::Inactive) !=
                            WickIndexTypes::None)
                            n_inactive_idxs_a[i] = n_inactive_idxs_b[j] = 1;
                        if ((tij & WickIndexTypes::Single) !=
                            WickIndexTypes::None)
                            single_idxs_mask[i] = single_idxs_mask[j] = 1;
                    }
                }
            } else {
                for (int j = i + 1; j < (int)cd_tensors.size(); j++) {
                    const WickIndexTypes &ia = cd_tensors[i].indices[0].types,
                                         &ib = cd_tensors[j].indices[0].types;
                    if (((ia == WickIndexTypes::None &&
                          ib == WickIndexTypes::None) ||
                         ((((ia & (~WickIndexTypes::AlphaBeta)) ==
                                WickIndexTypes::None &&
                            (ib & (~WickIndexTypes::AlphaBeta)) ==
                                WickIndexTypes::None) ||
                           ((ia & ib) & (~WickIndexTypes::AlphaBeta)) !=
                               WickIndexTypes::None) &&
                          (ia & WickIndexTypes::AlphaBeta) ==
                              (ib & WickIndexTypes::AlphaBeta))) &&
                        cd_tensors[i].type != cd_tensors[j].type &&
                        cd_tensors[j] < cd_tensors[i])
                        ctr_idxs.push_back(make_pair(i, j));
                }
            }
        }
        ctr_cd_idxs[cd_tensors.size()] = (int)ctr_idxs.size();
        for (int i = (int)n_inactive_idxs_a.size() - 2; i >= 0; i--)
            n_inactive_idxs_a[i] += n_inactive_idxs_a[i + 1];
        for (int i = (int)n_inactive_idxs_b.size() - 2; i >= 0; i--)
            n_inactive_idxs_b[i] += n_inactive_idxs_b[i + 1];
        vector<pair<int, int>> que;
        vector<pair<int, int>> cur_idxs(cd_tensors.size());
        vector<int8_t> cur_idxs_mask(cd_tensors.size(), 0),
            level_single_mask(cd_tensors.size(), 0);
        vector<int> tensor_idxs(cd_tensors.size()), rev_idxs(cd_tensors.size());
        vector<int> cd_idx_map_rev(cd_tensors.size());
        vector<int> acc_sign(cd_tensors.size() + 1);
        if (max_unctr != 0 || cd_tensors.size() % 2 == 0) {
            que.push_back(make_pair(-1, -1));
            acc_sign[0] = init_sign; // even
            for (int i = 0; i < (int)cd_tensors.size(); i++)
                tensor_idxs[i] = i;
            // arg sort of tensors in the normal order
            if (sf_type && no_unctr_sf_inact) {
                stable_sort(tensor_idxs.begin(), tensor_idxs.end(),
                            [&cd_tensors](int i, int j) {
                                return cd_tensors[i].type < cd_tensors[j].type;
                            });
                assert(all_of(tensor_idxs.begin(),
                              tensor_idxs.begin() + tensor_idxs.size() / 2,
                              [&cd_tensors](int i) {
                                  return cd_tensors[i].type ==
                                         WickTensorTypes::CreationOperator;
                              }));
            } else {
                stable_sort(tensor_idxs.begin(), tensor_idxs.end(),
                            [&cd_tensors](int i, int j) {
                                return cd_tensors[i] < cd_tensors[j];
                            });
                if (sf_type && !no_unctr_sf_inact) {
                    // the previous guarentees that we can sort
                    // the first and the second half separately
                    int n_sf = (int)tensor_idxs.size() / 2;
                    stable_sort(tensor_idxs.begin(), tensor_idxs.end(),
                                [&cd_tensors](int i, int j) {
                                    int ki = (cd_tensors[i].indices[0].types &
                                              WickIndexTypes::ActiveSingle) !=
                                             WickIndexTypes::None;
                                    int kj = (cd_tensors[j].indices[0].types &
                                              WickIndexTypes::ActiveSingle) !=
                                             WickIndexTypes::None;
                                    return ki < kj;
                                });
                    int n_act = 0;
                    for (auto &ti : tensor_idxs)
                        n_act += (cd_tensors[ti].indices[0].types &
                                  WickIndexTypes::ActiveSingle) !=
                                 WickIndexTypes::None;
                    // only when there are no active indices, fermi_type_compare
                    // is stable
                    stable_sort(
                        tensor_idxs.begin(),
                        tensor_idxs.begin() + min(n_sf * 2 - n_act, n_sf),
                        [&cd_tensors](int i, int j) {
                            int fc =
                                cd_tensors[i].fermi_type_compare(cd_tensors[j]);
                            int ispin = cd_tensors[i].get_spin_tag();
                            int jspin = cd_tensors[j].get_spin_tag();
                            return fc != 0
                                       ? (fc == -1)
                                       : (ispin != jspin
                                              ? ispin < jspin
                                              : cd_tensors[i] < cd_tensors[j]);
                        });
                    stable_sort(
                        tensor_idxs.begin() + min(n_sf * 2 - n_act, n_sf),
                        tensor_idxs.begin() + (n_sf * 2 - n_act),
                        [&cd_tensors](int i, int j) {
                            int fc =
                                cd_tensors[i].fermi_type_compare(cd_tensors[j]);
                            int ispin = cd_tensors[i].get_spin_tag();
                            int jspin = cd_tensors[j].get_spin_tag();
                            return fc != 0
                                       ? (fc == -1)
                                       : (ispin != jspin
                                              ? ispin > jspin
                                              : cd_tensors[i] < cd_tensors[j]);
                        });
                    stable_sort(
                        tensor_idxs.begin() + (n_sf * 2 - n_act),
                        tensor_idxs.end(), [&cd_tensors](int i, int j) {
                            int fc =
                                cd_tensors[i].fermi_type_compare(cd_tensors[j]);
                            int ispin = cd_tensors[i].get_spin_tag();
                            int jspin = cd_tensors[j].get_spin_tag();
                            return fc != 0
                                       ? (fc == -1)
                                       : (ispin != jspin
                                              ? ispin > jspin
                                              : cd_tensors[i] < cd_tensors[j]);
                        });
                }
                // sign for reordering tensors to the normal order
                for (int i = 0; i < (int)tensor_idxs.size(); i++)
                    rev_idxs[tensor_idxs[i]] = i;
                for (int i = 0; i < (int)rev_idxs.size(); i++)
                    for (int j = i + 1; j < (int)rev_idxs.size(); j++)
                        acc_sign[0] ^= (rev_idxs[j] < rev_idxs[i]);
            }
        }
        // depth-first tree traverse
        while (!que.empty()) {
            int l = que.back().first, j = que.back().second, k = 0;
            que.pop_back();
            int a, b, c, d, n_inact_a = 0, n_inact_b = 0;
            int inact_fac = 1, single_inact_fac = 1;
            if (l != -1) {
                cur_idxs[l] = ctr_idxs[j];
                k = ctr_cd_idxs[ctr_idxs[j].first + 1];
            }
            acc_sign[l + 2] = acc_sign[l + 1];
            ot_tensors.resize(ot_count + l + 1);
            memset(cur_idxs_mask.data(), 0,
                   sizeof(int8_t) * cur_idxs_mask.size());
            memset(level_single_mask.data(), 0,
                   sizeof(int8_t) * level_single_mask.size());
            if (sf_type) {
                memcpy(cd_idx_map_rev.data(), cd_idx_map.data(),
                       sizeof(int) * cd_idx_map.size());
                memcpy(cur_single_idxs_mask.data(), single_idxs_mask.data(),
                       sizeof(uint8_t) * single_idxs_mask.size());
            }
            if (l != -1) {
                tie(c, d) = cur_idxs[l];
                bool skip = false;
                acc_sign[l + 2] ^= ((c ^ d) & 1) ^ 1;
                // add contraction crossing sign from c/d
                for (int i = 0; i < l && !skip; i++) {
                    tie(a, b) = cur_idxs[i];
                    skip |= (b == d || b == c || a == d);
                    cur_idxs_mask[a] = cur_idxs_mask[b] = 1;
                    acc_sign[l + 2] ^= ((a < c && b > c && b < d) ||
                                        (a > c && a < d && b > d));
                }
                if (skip)
                    continue;
                cur_idxs_mask[c] = cur_idxs_mask[d] = 1;
                if (sf_type) {
                    n_inact_a = n_inact_b = 0;
                    for (int i = 0; i < l; i++) {
                        tie(a, b) = cur_idxs[i];
                        n_inact_a +=
                            n_inactive_idxs_a[a] - n_inactive_idxs_a[a + 1];
                        n_inact_b +=
                            n_inactive_idxs_b[b] - n_inactive_idxs_b[b + 1];
                        inact_fac *= 1 << ((cd_idx_map_rev[a] == b) &
                                           !cur_single_idxs_mask[a]);
                        single_inact_fac *= 1 << (level_single_mask[i] =
                                                      (cd_idx_map_rev[a] == b) &
                                                      cur_single_idxs_mask[a]);
                        cur_single_idxs_mask[cd_idx_map_rev[a]] |=
                            cur_single_idxs_mask[b];
                        cur_single_idxs_mask[cd_idx_map_rev[b]] |=
                            cur_single_idxs_mask[a];
                        cd_idx_map_rev[cd_idx_map_rev[a]] = cd_idx_map_rev[b];
                        cd_idx_map_rev[cd_idx_map_rev[b]] = cd_idx_map_rev[a];
                    }
                    n_inact_a +=
                        n_inactive_idxs_a[c] - n_inactive_idxs_a[c + 1];
                    n_inact_b +=
                        n_inactive_idxs_b[d] - n_inactive_idxs_b[d + 1];
                    // paired inactive must be all contracted
                    // otherwise it cannot be represented as spin-free operators
                    if (no_unctr_sf_inact &&
                        n_inact_a + n_inactive_idxs_a[c + 1] <
                            n_inactive_idxs_a[0] &&
                        n_inact_b + n_inactive_idxs_b[d + 1] <
                            n_inactive_idxs_b[0])
                        continue;
                    inact_fac *= 1 << ((cd_idx_map_rev[c] == d) &
                                       !cur_single_idxs_mask[c]);
                    single_inact_fac *=
                        1 << (level_single_mask[l] = (cd_idx_map_rev[c] == d) &
                                                     cur_single_idxs_mask[c]);
                    cur_single_idxs_mask[cd_idx_map_rev[c]] |=
                        cur_single_idxs_mask[d];
                    cur_single_idxs_mask[cd_idx_map_rev[d]] |=
                        cur_single_idxs_mask[c];
                    cd_idx_map_rev[cd_idx_map_rev[c]] = cd_idx_map_rev[d];
                    cd_idx_map_rev[cd_idx_map_rev[d]] = cd_idx_map_rev[c];
                }
                if (!sf_type || (sf_type && !no_unctr_sf_inact)) {
                    // remove tensor reorder sign for c/d
                    acc_sign[l + 2] ^= (rev_idxs[d] < rev_idxs[c]);
                    for (int i = 0; i < (int)rev_idxs.size(); i++)
                        if (!cur_idxs_mask[i]) {
                            acc_sign[l + 2] ^=
                                (rev_idxs[max(c, i)] < rev_idxs[min(c, i)]);
                            acc_sign[l + 2] ^=
                                (rev_idxs[max(d, i)] < rev_idxs[min(d, i)]);
                        }
                }
                ot_tensors[ot_count + l] =
                    WickTensor::kronecker_delta(vector<WickIndex>{
                        cd_tensors[c].indices[0], cd_tensors[d].indices[0]});
            }
            // push next contraction order to queue
            if (!no_ctr)
                for (; k < (int)ctr_idxs.size(); k++)
                    que.push_back(make_pair(l + 1, k));
            if (max_unctr != -1 && cd_tensors.size() - (l + l + 2) > max_unctr)
                continue;
            if (sf_type && no_unctr_sf_inact) {
                if (n_inact_a < n_inactive_idxs_a[0] &&
                    n_inact_b < n_inactive_idxs_b[0])
                    continue;
                int sf_n = (int)cd_tensors.size() / 2, tn = sf_n - l - 1;
                vector<WickIndex> wis(tn * 2);
                for (int i = 0, k = 0; i < (int)tensor_idxs.size(); i++)
                    if (!cur_idxs_mask[tensor_idxs[i]] &&
                        cd_tensors[tensor_idxs[i]].type ==
                            WickTensorTypes::CreationOperator) {
                        rev_idxs[k] = tensor_idxs[i];
                        rev_idxs[k + tn] = cd_idx_map_rev[tensor_idxs[i]];
                        k++;
                    }
                for (int i = 0; i < tn + tn; i++)
                    wis[i] = cd_tensors[rev_idxs[i]].indices[0];
                // sign for reversing destroy operator
                final_sign = ((tn - 1) & 1) ^ (((tn - 1) & 2) >> 1);
                // sign for reordering tensors to the normal order
                for (int i = 0; i < (int)(tn + tn); i++)
                    for (int j = i + 1; j < (int)(tn + tn); j++)
                        final_sign ^= (rev_idxs[j] < rev_idxs[i]);
                if (wis.size() != 0)
                    ot_tensors.push_back(WickTensor::spin_free(wis));
            } else {
                if (sf_type && !no_unctr_sf_inact) {
                    int max_tag = (int)tensor_idxs.size() + 1;
                    for (int i = 0; i < (int)tensor_idxs.size(); i++)
                        if (!cur_idxs_mask[tensor_idxs[i]])
                            cd_tensors[tensor_idxs[i]].set_spin_tag(max_tag);
                    int spin_tag = 0;
                    for (int i = 0; i < (int)tensor_idxs.size(); i++)
                        if (!cur_idxs_mask[tensor_idxs[i]])
                            if (cd_tensors[tensor_idxs[i]].get_spin_tag() ==
                                max_tag) {
                                cd_tensors[tensor_idxs[i]].set_spin_tag(
                                    spin_tag);
                                cd_tensors[cd_idx_map_rev[tensor_idxs[i]]]
                                    .set_spin_tag(spin_tag);
                                spin_tag++;
                            }
                    for (int i = 0; i <= l; i++) {
                        tie(a, b) = cur_idxs[i];
                        if (level_single_mask[i]) {
                            cd_tensors[a].set_spin_tag(spin_tag);
                            cd_tensors[b].set_spin_tag(spin_tag);
                            spin_tag++;
                        }
                    }
                }
                for (int i = 0; i < (int)tensor_idxs.size(); i++)
                    if (!cur_idxs_mask[tensor_idxs[i]])
                        ot_tensors.push_back(cd_tensors[tensor_idxs[i]]);
            }
            if (single_inact_fac != 1) {
                // 0 = delta; 1 = 0.5 ab + 0.5 ba
                for (int ix = 1, lx = 1; ix < single_inact_fac; ix++) {
                    for (int i = 0, kx = ix; i <= l; i++)
                        if (level_single_mask[i])
                            lx <<= (kx & 1), kx >>= 1;
                    for (int jx = 0; jx < lx; jx++) {
                        vector<WickTensor> f_tensors(
                            ot_tensors.begin(), ot_tensors.begin() + ot_count);
                        for (int i = 0, kx = ix, lx = jx; i <= l;
                             kx >>= level_single_mask[i++])
                            if (level_single_mask[i]) {
                                if (!(kx & 1))
                                    f_tensors.push_back(
                                        ot_tensors[ot_count + i]);
                            } else
                                f_tensors.push_back(ot_tensors[ot_count + i]);
                        for (int i = 0, kx = ix, mx = jx; i <= l;
                             kx >>= level_single_mask[i++])
                            if (level_single_mask[i] && (kx & 1)) {
                                tie(a, b) = cur_idxs[i];
                                if (mx & 1) {
                                    f_tensors.push_back(cd_tensors[a]);
                                    f_tensors.push_back(cd_tensors[b]);
                                } else {
                                    f_tensors.push_back(cd_tensors[b]);
                                    f_tensors.push_back(cd_tensors[a]);
                                }
                                mx >>= 1;
                            }
                        for (int i = 0; i < (int)tensor_idxs.size(); i++)
                            if (!cur_idxs_mask[tensor_idxs[i]])
                                f_tensors.push_back(cd_tensors[tensor_idxs[i]]);
                        r.terms.push_back(WickString(
                            f_tensors, x.ctr_indices,
                            (1.0 / lx) * inact_fac *
                                ((acc_sign[l + 2] ^ final_sign) ? -x.factor
                                                                : x.factor)));
                    }
                }
                single_inact_fac = 1;
            }
            r.terms.push_back(WickString(
                ot_tensors, x.ctr_indices,
                inact_fac * single_inact_fac *
                    ((acc_sign[l + 2] ^ final_sign) ? -x.factor : x.factor)));
        }
        return r;
    }
    WickExpr simple_sort() const {
        WickExpr r = *this;
        for (auto &rr : r.terms)
            rr = rr.simple_sort();
        return r;
    }
    WickExpr simplify_delta() const {
        WickExpr r = *this;
        for (auto &rr : r.terms)
            rr = rr.simplify_delta();
        return r;
    }
    WickExpr simplify_zero() const {
        WickExpr r;
        for (auto &rr : terms)
            if (abs(rr.factor) > 1E-12)
                r.terms.push_back(rr);
        return r;
    }
    WickExpr remove_external() const {
        WickExpr r;
        for (auto &rr : terms)
            if (!rr.has_external_ops())
                r.terms.push_back(rr);
        return r;
    }
    WickExpr remove_inactive() const {
        WickExpr r;
        for (auto &rr : terms)
            if (!rr.has_inactive_ops())
                r.terms.push_back(rr);
        return r;
    }
    // when there is only one spin free operator
    // it can be considered as density matrix
    // on the ref state with trans symmetry
    WickExpr add_spin_free_trans_symm() const {
        WickExpr r = *this;
        for (auto &rr : r.terms) {
            int found = 0;
            WickTensor *xwt;
            for (auto &wt : rr.tensors)
                if (wt.type == WickTensorTypes::SpinFreeOperator)
                    found++, xwt = &wt;
            if (found == 1)
                xwt->perms = WickPermutation::complete_set(
                    (int)xwt->indices.size(),
                    WickPermutation::pair_symmetric(
                        (int)xwt->indices.size() / 2, true));
        }
        return r;
    }
    WickExpr conjugate() const {
        WickExpr r = *this;
        for (auto &rr : r.terms) {
            vector<WickTensor> tensors;
            for (auto &wt : rr.tensors)
                if (wt.type == WickTensorTypes::SpinFreeOperator) {
                    int k = (int)wt.indices.size() / 2;
                    for (int i = 0; i < k; i++)
                        swap(wt.indices[i], wt.indices[i + k]);
                    tensors.push_back(wt);
                } else if (wt.type == WickTensorTypes::CreationOperator) {
                    wt.type = WickTensorTypes::DestroyOperator;
                    wt.name[0] = wt.name[0] == 'C' ? 'D' : wt.name[0];
                    tensors.push_back(wt);
                } else if (wt.type == WickTensorTypes::DestroyOperator) {
                    wt.type = WickTensorTypes::CreationOperator;
                    wt.name[0] = wt.name[0] == 'D' ? 'C' : wt.name[0];
                    tensors.push_back(wt);
                }
            for (auto &wt : rr.tensors)
                if (wt.type == WickTensorTypes::SpinFreeOperator ||
                    wt.type == WickTensorTypes::CreationOperator ||
                    wt.type == WickTensorTypes::DestroyOperator)
                    wt = tensors.back(), tensors.pop_back();
        }
        return r;
    }
    WickExpr simplify_merge() const {
        vector<WickString> sorted(terms.size());
        vector<pair<int, double>> ridxs;
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int k = 0; k < (int)terms.size(); k++)
            sorted[k] = terms[k].abs().quick_sort();
        threading->activate_normal();
        for (int i = 0; i < (int)terms.size(); i++) {
            bool found = false;
            for (int j = 0; j < (int)ridxs.size() && !found; j++)
                if (sorted[i].abs_equal_to(sorted[ridxs[j].first])) {
                    found = true;
                    ridxs[j].second += terms[i].factor * sorted[i].factor *
                                       sorted[ridxs[j].first].factor;
                }
            if (!found)
                ridxs.push_back(make_pair(i, terms[i].factor));
        }
        WickExpr r;
        for (auto &m : ridxs) {
            r.terms.push_back(terms[m.first]);
            r.terms.back().factor = m.second;
        }
        r = r.simplify_zero();
        sort(r.terms.begin(), r.terms.end());
        return r;
    }
    WickExpr simplify() const {
        return simplify_delta().simplify_zero().simplify_merge();
    }
    void save(ostream &ofs) const {
        size_t lterms = (size_t)terms.size();
        ofs.write((char *)&lterms, sizeof(lterms));
        for (size_t i = 0; i < lterms; i++)
            terms[i].save(ofs);
    }
    void load(istream &ifs) {
        size_t lterms = 0;
        ifs.read((char *)&lterms, sizeof(lterms));
        terms.resize(lterms);
        for (size_t i = 0; i < lterms; i++)
            terms[i].load(ifs);
    }
};

inline WickExpr operator+(const WickString &a, const WickString &b) noexcept {
    return WickExpr({a, b});
}

inline WickExpr operator*(double d, const WickExpr &x) noexcept {
    return x * d;
}

// commutator
inline WickExpr operator^(const WickExpr &a, const WickExpr &b) noexcept {
    return a * b + b * a * (-1.0);
}

// multiply and contract all
inline WickExpr operator&(const WickExpr &a, const WickExpr &b) noexcept {
    WickExpr c = a * b;
    for (auto &ws : c.terms)
        for (auto &wt : ws.tensors)
            for (auto &wi : wt.indices)
                ws.ctr_indices.insert(wi);
    return c;
}

struct WickGraph {
    vector<WickTensor> left;
    vector<WickExpr> right;
    vector<vector<pair<double, map<string, string>>>> index_maps;
    map<WickIndexTypes, double> idx_scales;
    double multiply_scale = 5, add_scale = 3;
    string intermediate_name = "_x";
    static map<WickIndexTypes, double> init_idx_scales() {
        map<WickIndexTypes, double> r;
        r[WickIndexTypes::External] = 12;
        r[WickIndexTypes::ExternalAlpha] = 12;
        r[WickIndexTypes::ExternalBeta] = 12;
        r[WickIndexTypes::Active] = 8;
        r[WickIndexTypes::ActiveAlpha] = 8;
        r[WickIndexTypes::ActiveBeta] = 8;
        r[WickIndexTypes::Inactive] = 4;
        r[WickIndexTypes::InactiveAlpha] = 4;
        r[WickIndexTypes::InactiveBeta] = 4;
        r[WickIndexTypes::None] = 4;
        return r;
    }
    WickGraph()
        : left(vector<WickTensor>()), right(vector<WickExpr>()),
          index_maps(vector<vector<pair<double, map<string, string>>>>()) {
        idx_scales = init_idx_scales();
    }
    WickGraph(
        const vector<WickTensor> &left, const vector<WickExpr> &right,
        const vector<vector<pair<double, map<string, string>>>> &index_maps)
        : left(left), right(right), index_maps(index_maps) {
        idx_scales = init_idx_scales();
    }
    static WickGraph
    from_expr(const vector<pair<WickTensor, WickExpr>> &terms) {
        WickGraph gr;
        for (auto &wt : terms) {
            gr.left.push_back(wt.first);
            gr.right.push_back(wt.second);
            gr.index_maps.push_back(vector<pair<double, map<string, string>>>{
                make_pair(1.0, map<string, string>())});
        }
        return gr;
    }
    static map<string, string> merge_index_maps(const map<string, string> &a,
                                                const map<string, string> &b) {
        if (a.size() == 0)
            return b;
        else if (b.size() == 0)
            return a;
        map<string, string> r = a;
        for (auto &m : r)
            if (b.count(m.second))
                m.second = b.at(m.second);
        for (auto &m : b)
            if (!r.count(m.first))
                r[m.first] = m.second;
        return r;
    }
    int n_terms() const { return (int)right.size(); }
    void add_term(const WickTensor &wt, const WickExpr &wx) {
        left.push_back(wt);
        right.push_back(wx);
        index_maps.push_back(vector<pair<double, map<string, string>>>{
            make_pair(1.0, map<string, string>())});
    }
    // figure out next useable intermediate name index
    int get_intermediate_start() const {
        int tmp_start = 0;
        // figure out next useable intermediate name index
        const size_t inl = intermediate_name.length();
        for (int ix = 0; ix < n_terms(); ix++) {
            for (const WickString &ws : right[ix].terms)
                for (const WickTensor &wt : ws.tensors)
                    if (wt.name.substr(0, inl) == intermediate_name) {
                        int tmp_num = 0;
                        for (size_t it = inl; it < wt.name.length(); it++)
                            if (wt.name[it] >= '0' || wt.name[it] <= '9')
                                tmp_num = tmp_num * 10 + (wt.name[it] - '0');
                            else {
                                tmp_num = -1;
                                break;
                            }
                        tmp_start = max(tmp_start, tmp_num);
                    }
            if (left[ix].name.substr(0, inl) == intermediate_name) {
                int tmp_num = 0;
                for (size_t it = inl; it < left[ix].name.length(); it++)
                    if (left[ix].name[it] >= '0' || left[ix].name[it] <= '9')
                        tmp_num = tmp_num * 10 + (left[ix].name[it] - '0');
                    else {
                        tmp_num = -1;
                        break;
                    }
                tmp_start = max(tmp_start, tmp_num);
            }
        }
        return tmp_start;
    }
    // sort each term in each expr according to best contraction order
    WickGraph simplify_binary_sort() const {
        WickGraph gr = *this;
        for (int ix = 0; ix < gr.n_terms(); ix++) {
            for (WickString &wt : gr.right[ix].terms) {
                // cout << "orig str = " << wt << endl;
                vector<int> xord(wt.tensors.size()), mord;
                for (int it = 0; it < (int)xord.size(); it++)
                    xord[it] = it;
                double msca = 0.0;
                vector<set<WickIndex>> out_idxs(wt.tensors.size());
                for (auto &wi : gr.left[ix].indices)
                    out_idxs[wt.tensors.size() - 1].insert(wi);
                do {
                    if (xord.size() >= 2 && xord[0] > xord[1])
                        continue;
                    // cout << ":: ord = ";
                    // for (auto &xxf : xord)
                    //     cout << xxf << " ";
                    // cout << endl;
                    for (int ii = (int)xord.size() - 1; ii > 0; ii--) {
                        out_idxs[ii - 1] = out_idxs[ii];
                        for (auto &wi : wt.tensors[xord[ii]].indices)
                            out_idxs[ii - 1].insert(wi);
                    }
                    set<WickIndex> p;
                    for (auto &wi : wt.tensors[xord[0]].indices)
                        p.insert(wi);
                    double tsca = 0, psca;
                    for (int ii = 1; ii < (int)xord.size(); ii++) {
                        for (auto &wi : wt.tensors[xord[ii]].indices)
                            p.insert(wi);
                        psca = 1;
                        for (auto &x : p)
                            psca *= idx_scales.count(x.types)
                                        ? idx_scales.at(x.types)
                                        : idx_scales.at(WickIndexTypes::None);
                        tsca += psca;
                        // cout << "out " << ii << " = ";
                        // for (auto &pp : out_idxs[ii])
                        //     cout << pp << " ";
                        //     cout << endl;
                        for (auto px = p.begin(); px != p.end();) {
                            if (!out_idxs[ii].count(*px))
                                px = p.erase(px);
                            else
                                ++px;
                        }
                        // cout << "ctr" << ii << " -> " << tsca << endl;
                        // for (auto &pp : p)
                        //     cout << pp << " ";
                        //     cout << endl;
                    }
                    if (msca == 0.0 || tsca < msca)
                        msca = tsca, mord = xord;
                } while (next_permutation(xord.begin(), xord.end()));
                vector<WickTensor> wts(mord.size());
                for (int ii = 0; ii < (int)mord.size(); ii++)
                    wts[ii] = wt.tensors[mord[ii]];
                wt = WickString(wts, wt.ctr_indices, wt.factor);
                // cout << "final wt = " << wt << endl;
            }
        }
        return gr;
    }
    // split terms into binary contractions
    WickGraph simplify_binary_split() const {
        WickGraph gr = *this;
        int tmp_start = gr.get_intermediate_start();
        vector<pair<WickTensor, WickExpr>> new_terms;
        for (int ix = 0; ix < gr.n_terms(); ix++) {
            for (WickString &wt : gr.right[ix].terms) {
                if (wt.tensors.size() <= 2)
                    continue;
                vector<set<WickIndex>> out_idxs(wt.tensors.size());
                for (auto &wi : gr.left[ix].indices)
                    out_idxs[wt.tensors.size() - 1].insert(wi);
                for (int ii = (int)wt.tensors.size() - 1; ii > 0; ii--) {
                    out_idxs[ii - 1] = out_idxs[ii];
                    for (auto &wi : wt.tensors[ii].indices)
                        out_idxs[ii - 1].insert(wi);
                }
                set<WickIndex> p, pctr;
                for (auto &wi : wt.tensors[0].indices)
                    p.insert(wi);
                WickTensor pt = wt.tensors[0], pr;
                for (int ii = 1; ii < (int)wt.tensors.size(); ii++) {
                    const WickTensor &ps = wt.tensors[ii];
                    pctr.clear();
                    for (auto &wi : ps.indices)
                        p.insert(wi);
                    for (auto px = p.begin(); px != p.end();) {
                        if (!out_idxs[ii].count(*px))
                            pctr.insert(*px), px = p.erase(px);
                        else
                            ++px;
                    }
                    if (ii != (int)wt.tensors.size() - 1) {
                        // get the perm for the intermediate
                        vector<int16_t> amap(pt.indices.size());
                        vector<int16_t> bmap(ps.indices.size());
                        vector<WickIndex> out_p;
                        map<WickIndex, int16_t> pctrl;
                        int16_t imx = 0;
                        for (auto &pr : pctr)
                            pctrl[pr] = imx++;
                        for (size_t ig = 0; ig < pt.indices.size(); ig++)
                            if (pctr.count(pt.indices[ig]))
                                amap[ig] =
                                    (int16_t)(-1 - pctrl.at(pt.indices[ig]));
                            else
                                amap[ig] = (int16_t)out_p.size(),
                                out_p.push_back(pt.indices[ig]);
                        for (size_t ig = 0; ig < ps.indices.size(); ig++)
                            if (pctr.count(ps.indices[ig]))
                                bmap[ig] =
                                    (int16_t)(-1 - pctrl.at(ps.indices[ig]));
                            else
                                bmap[ig] = (int16_t)out_p.size(),
                                out_p.push_back(ps.indices[ig]);
                        vector<WickPermutation> perms;
                        vector<pair<int16_t, int16_t>> awmap(amap.size());
                        vector<pair<int16_t, int16_t>> bwmap(bmap.size());
                        for (auto &wp : pt.perms) {
                            // cout << "1" << wp << endl;
                            for (int ig = 0; ig < (int)amap.size(); ig++)
                                awmap[ig] =
                                    make_pair(amap[ig], amap[wp.data[ig]]);
                            sort(awmap.begin(), awmap.end(),
                                 [](const pair<int16_t, int16_t> &a,
                                    const pair<int16_t, int16_t> &b) {
                                     return a.first < b.first;
                                 });
                            int imd = 0;
                            bool ctr_ok = true;
                            for (; imd < (int)amap.size(); imd++)
                                if (awmap[imd].first >= 0)
                                    break;
                                else if (awmap[imd].second >= 0)
                                    ctr_ok = false;
                            if (!ctr_ok)
                                continue;
                            for (auto &wq : ps.perms) {
                                // cout << "2" << wq << endl;
                                for (int ig = 0; ig < (int)bmap.size(); ig++)
                                    bwmap[ig] =
                                        make_pair(bmap[ig], bmap[wq.data[ig]]);
                                sort(bwmap.begin(), bwmap.end(),
                                     [](const pair<int16_t, int16_t> &a,
                                        const pair<int16_t, int16_t> &b) {
                                         return a.first < b.first;
                                     });
                                assert(imd <= (int)bwmap.size());
                                ctr_ok = true;
                                for (int imx = 0; imx < imd; imx++)
                                    if (!(ctr_ok = (awmap[imx].first ==
                                                        bwmap[imx].first &&
                                                    awmap[imx].second ==
                                                        bwmap[imx].second)))
                                        break;
                                if (!ctr_ok)
                                    continue;
                                vector<int16_t> pxr;
                                for (int imx = imd; imx < (int)awmap.size();
                                     imx++)
                                    pxr.push_back(awmap[imx].second);
                                for (int imx = imd; imx < (int)bwmap.size();
                                     imx++)
                                    pxr.push_back(bwmap[imx].second);
                                perms.push_back(WickPermutation(
                                    pxr, wp.negative ^ wq.negative));
                                if (pt.name == ps.name &&
                                    awmap.size() == bwmap.size()) {
                                    ctr_ok = true;
                                    for (int imx = imd; imx < (int)awmap.size();
                                         imx++)
                                        if (!(ctr_ok = (awmap[imx].second ==
                                                        bwmap[imx].second -
                                                            (int)awmap.size() +
                                                            imd)))
                                            break;
                                    if (ctr_ok) {
                                        pxr.clear();
                                        for (int imx = imd;
                                             imx < (int)bwmap.size(); imx++)
                                            pxr.push_back(bwmap[imx].second);
                                        for (int imx = imd;
                                             imx < (int)awmap.size(); imx++)
                                            pxr.push_back(awmap[imx].second);
                                        perms.push_back(WickPermutation(
                                            pxr, wp.negative ^ wq.negative));
                                    }
                                }
                            }
                        }
                        // remove repeated indices
                        map<WickIndex, int> out_s;
                        vector<WickIndex> out_pp;
                        for (auto &wi : out_p)
                            if (!out_s.count(wi))
                                out_s[wi] = (int)out_pp.size(),
                                out_pp.push_back(wi);
                        if (out_pp.size() != out_p.size()) {
                            vector<WickPermutation> xperms;
                            for (auto &perm : perms) {
                                vector<int16_t> data(out_pp.size(), -1);
                                bool ok = true;
                                for (int irp = 0;
                                     irp < (int)perm.data.size() && ok; irp++)
                                    if (data[out_s.at(out_p[irp])] == -1)
                                        data[out_s.at(out_p[irp])] =
                                            out_s.at(out_p[perm.data[irp]]);
                                    else if (!(ok =
                                                   (data[out_s.at(
                                                        out_p[irp])] ==
                                                    out_s.at(
                                                        out_p
                                                            [perm.data[irp]]))))
                                        break;
                                if (ok)
                                    xperms.push_back(
                                        WickPermutation(data, perm.negative));
                            }
                            out_p = out_pp;
                            perms = xperms;
                        }
                        stringstream name;
                        name << intermediate_name << ++tmp_start;
                        perms = WickTensor::reset_permutations(out_p, perms);
                        pr = WickTensor(name.str(), out_p, perms);
                        new_terms.push_back(make_pair(
                            pr, WickExpr(WickString(vector<WickTensor>{pt, ps},
                                                    pctr))));
                        pt = pr;
                    } else
                        wt = WickString(vector<WickTensor>{pt, ps}, pctr,
                                        wt.factor);
                }
            }
        }
        for (auto &wt : new_terms) {
            gr.left.push_back(wt.first);
            gr.right.push_back(wt.second);
            gr.index_maps.push_back(vector<pair<double, map<string, string>>>{
                make_pair(1.0, map<string, string>())});
        }
        return gr;
    }
    // delete/merge duplicate binary contractions
    WickGraph simplify_binary_unique(int iprint = 0) const {
        vector<pair<pair<WickTensor, WickExpr>,
                    vector<pair<double, map<string, string>>>>>
            new_terms;
        vector<pair<WickTensor, int>> uniq_terms;
        map<pair<string, int>, pair<pair<string, vector<int>>, double>>
            dup_index_perms;
        vector<vector<WickString>> sorted(n_terms());
        vector<pair<int, int>> sorted_idx;
        for (int ix = 0; ix < n_terms(); ix++) {
            if (index_maps[ix].size() != 1)
                continue;
            for (int ig = 0; ig < (int)right[ix].terms.size(); ig++)
                sorted_idx.push_back(make_pair(ix, ig));
            sorted[ix].resize(right[ix].terms.size());
        }
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int k = 0; k < (int)sorted_idx.size(); k++) {
            int ix = sorted_idx[k].first, ig = sorted_idx[k].second;
            sorted[ix][ig] = right[ix].terms[ig].abs().quick_sort();
        }
        threading->activate_normal();
        for (int ix = 0; ix < n_terms(); ix++) {
            if (index_maps[ix].size() != 1) {
                new_terms.push_back(
                    make_pair(make_pair(left[ix], right[ix]), index_maps[ix]));
                continue;
            }
            const vector<WickString> &gg = sorted[ix];
            bool ufound = false;
            for (auto &ut : uniq_terms) {
                const vector<WickString> &uu = sorted[ut.second];
                bool ok = true;
                for (size_t igu = 0; igu < gg.size() && ok; igu++) {
                    const WickString &g = gg[igu], &u = uu[igu];
                    if (right[ix].terms.size() != uu.size() ||
                        g.ctr_indices.size() != u.ctr_indices.size() ||
                        g.tensors.size() != u.tensors.size())
                        ok = false;
                    if (ut.first.indices.size() != left[ix].indices.size())
                        ok = false;
                    for (auto ig = g.ctr_indices.begin(),
                              iu = u.ctr_indices.begin();
                         ig != g.ctr_indices.end() && ok; ++ig, ++iu)
                        if (!(ok = (*ig == *iu)))
                            break;
                    for (int ip = 0; ip < (int)g.tensors.size() && ok; ip++)
                        if (!(ok = (g.tensors[ip].name == u.tensors[ip].name &&
                                    g.tensors[ip].indices.size() ==
                                        u.tensors[ip].indices.size())))
                            break;
                }
                if (!ok)
                    continue;
                // FIXME: here need to consider tensor permutations
                map<WickIndex, WickIndex> gtu, gtuctr;
                if (gg.size() > 0) {
                    const WickString &g = gg[0], &u = uu[0];
                    for (int ip = 0; ip < (int)g.tensors.size() && ok; ip++)
                        for (size_t ig = 0;
                             ig < g.tensors[ip].indices.size() && ok; ig++)
                            if (!(ok = (g.ctr_indices.count(
                                            g.tensors[ip].indices[ig]) ==
                                        u.ctr_indices.count(
                                            u.tensors[ip].indices[ig]))))
                                break;
                            else {
                                if (!(ok = (g.tensors[ip].indices[ig].types ==
                                            u.tensors[ip].indices[ig].types)))
                                    break;
                                if (!g.ctr_indices.count(
                                        g.tensors[ip].indices[ig])) {
                                    if (!gtu.count(g.tensors[ip].indices[ig]))
                                        gtu[g.tensors[ip].indices[ig]] =
                                            u.tensors[ip].indices[ig];
                                    else if (!(ok =
                                                   (gtu.at(g.tensors[ip]
                                                               .indices[ig]) ==
                                                    u.tensors[ip].indices[ig])))
                                        break;
                                } else {
                                    if (!gtuctr.count(
                                            g.tensors[ip].indices[ig]))
                                        gtuctr[g.tensors[ip].indices[ig]] =
                                            u.tensors[ip].indices[ig];
                                    else if (!(ok =
                                                   (gtuctr.at(
                                                        g.tensors[ip]
                                                            .indices[ig]) ==
                                                    u.tensors[ip].indices[ig])))
                                        break;
                                }
                            }
                }
                if (!ok || gtu.size() != left[ix].indices.size())
                    continue;
                for (size_t igu = 1; igu < gg.size() && ok; igu++) {
                    const WickString &g = gg[igu], &u = uu[igu];
                    for (int ip = 0; ip < (int)g.tensors.size() && ok; ip++)
                        for (size_t ig = 0;
                             ig < g.tensors[ip].indices.size() && ok; ig++)
                            if (!(ok = (g.ctr_indices.count(
                                            g.tensors[ip].indices[ig]) ==
                                        u.ctr_indices.count(
                                            u.tensors[ip].indices[ig]))))
                                break;
                            else if (!g.ctr_indices.count(
                                         g.tensors[ip].indices[ig])) {
                                if (!(ok = (gtu.count(
                                                g.tensors[ip].indices[ig]) &&
                                            gtu.at(g.tensors[ip].indices[ig]) ==
                                                u.tensors[ip].indices[ig])))
                                    break;
                            } else {
                                if (!(ok = (gtuctr.count(
                                                g.tensors[ip].indices[ig]) &&
                                            gtuctr.at(
                                                g.tensors[ip].indices[ig]) ==
                                                u.tensors[ip].indices[ig])))
                                    break;
                            }
                    ok = ok || (abs(right[ix].terms[0].factor *
                                        right[ut.second].terms[igu].factor *
                                        gg[0].factor * uu[0].factor -
                                    right[ix].terms[igu].factor *
                                        right[ut.second].terms[0].factor *
                                        g.factor * u.factor) < 1E-12);
                }
                if (!ok)
                    continue;
                vector<int> pgw(ut.first.indices.size());
                for (int i = 0, j; i < (int)ut.first.indices.size(); i++) {
                    for (j = 0; j < (int)left[ix].indices.size(); j++)
                        if (gtu.at(left[ix].indices[j]) == ut.first.indices[i])
                            break;
                    pgw[i] = j;
                }
                dup_index_perms[make_pair(left[ix].name,
                                          (int)left[ix].indices.size())] =
                    make_pair(make_pair(ut.first.name, pgw),
                              right[ix].terms[0].factor /
                                  right[ut.second].terms[0].factor *
                                  gg[0].factor * uu[0].factor);
                if (iprint) {
                    cout << "found equal = " << left[ix].name << " -> "
                         << ut.first.name << " ";
                    for (int i = 0; i < pgw.size(); i++)
                        cout << pgw[i] << " ";
                    cout << " f = "
                         << (right[ix].terms[0].factor /
                             right[ut.second].terms[0].factor * gg[0].factor *
                             uu[0].factor)
                         << endl;
                }
                ufound = true;
                break;
            }
            if (!ufound)
                uniq_terms.push_back(make_pair(left[ix], ix));
        }
        WickGraph gr;
        for (auto &wt : uniq_terms) {
            gr.left.push_back(wt.first);
            gr.right.push_back(right[wt.second]);
            gr.index_maps.push_back(vector<pair<double, map<string, string>>>{
                make_pair(1.0, map<string, string>())});
        }
        for (auto &wt : new_terms) {
            gr.left.push_back(wt.first.first);
            gr.right.push_back(wt.first.second);
            gr.index_maps.push_back(wt.second);
        }
        for (auto &wx : gr.right)
            for (auto &ws : wx.terms)
                for (auto &wt : ws.tensors)
                    if (dup_index_perms.count(
                            make_pair(wt.name, (int)wt.indices.size()))) {
                        pair<pair<string, vector<int>>, double> &dp =
                            dup_index_perms.at(
                                make_pair(wt.name, (int)wt.indices.size()));
                        wt.name = dp.first.first;
                        ws.factor *= dp.second;
                        vector<WickIndex> nidx = wt.indices;
                        for (int i = 0; i < (int)wt.indices.size(); i++)
                            nidx[i] = wt.indices[dp.first.second[i]];
                        wt.indices = nidx;
                    }
        return dup_index_perms.size() == 0 ? gr : gr.simplify_binary_unique();
    }
    // for each expr, detect common factors
    WickGraph simplify_binary_factor(int iprint = 0) const {
        WickGraph gr;
        int tmp_start = get_intermediate_start();
        for (int ix = 0; ix < n_terms(); ix++) {
            if (right[ix].terms.size() == 1) {
                gr.left.push_back(left[ix]);
                gr.right.push_back(right[ix]);
                gr.index_maps.push_back(index_maps[ix]);
                continue;
            }
            vector<WickString> other_strings;
            vector<WickString> unique_strings;
            vector<pair<WickTensor, WickExpr>> new_terms;
            for (auto ws : right[ix].terms) {
                if (ws.tensors.size() != 2) {
                    other_strings.push_back(ws);
                    continue;
                }
                set<WickIndex> wwa(ws.tensors[0].indices.begin(),
                                   ws.tensors[0].indices.end());
                set<WickIndex> wwb(ws.tensors[1].indices.begin(),
                                   ws.tensors[1].indices.end());
                bool skip = false;
                for (auto &wa : wwa)
                    if (wwb.count(wa) && !ws.ctr_indices.count(wa))
                        skip = true;
                if (skip || wwa.size() != ws.tensors[0].indices.size() ||
                    wwb.size() != ws.tensors[1].indices.size()) {
                    other_strings.push_back(ws);
                    continue;
                }
                bool found = false;
                for (auto &wr : unique_strings) {
                    if (wr.ctr_indices.size() != ws.ctr_indices.size() || found)
                        continue;
                    for (int iws = 0; iws < 2 && !found; iws++)
                        for (int iwr = 0; iwr < 2 && !found; iwr++)
                            if (ws.tensors[iws].name == wr.tensors[iwr].name &&
                                ws.tensors[iws].indices.size() ==
                                    wr.tensors[iwr].indices.size() &&
                                ws.tensors[!iws].indices.size() ==
                                    wr.tensors[!iwr].indices.size()) {
                                for (auto &zperm : ws.tensors[iws].perms) {
                                    WickString zs = ws;
                                    zs.tensors[iws] = ws.tensors[iws] * zperm;
                                    map<string, string> idx_mp =
                                        WickTensor::get_index_map(
                                            zs.tensors[iws].indices,
                                            wr.tensors[iwr].indices);
                                    if (idx_mp.size() == 0)
                                        continue;
                                    set<string> zsc, wrc;
                                    for (auto &wc : zs.ctr_indices)
                                        zsc.insert(wc.name);
                                    for (auto &wc : wr.ctr_indices)
                                        wrc.insert(wc.name);
                                    for (auto px = idx_mp.begin();
                                         px != idx_mp.end();) {
                                        if (!zsc.count(px->first) ||
                                            !wrc.count(px->second))
                                            px = idx_mp.erase(px);
                                        else
                                            ++px;
                                    }
                                    WickString wt = zs.index_map(idx_mp);
                                    // cout << zperm << " " << wt.tensors[iws]
                                    //      << " " << wr.tensors[iwr] << endl;
                                    if (wt.tensors[iws] != wr.tensors[iwr])
                                        continue;
                                    set<WickPermutation> pa(
                                        wt.tensors[!iws].perms.begin(),
                                        wt.tensors[!iws].perms.end());
                                    vector<WickPermutation> perms;
                                    for (auto &perm : wr.tensors[!iwr].perms)
                                        if (pa.count(perm))
                                            perms.push_back(perm);
                                    stringstream name;
                                    name << intermediate_name << ++tmp_start;
                                    WickTensor wc(name.str(),
                                                  wr.tensors[!iwr].indices,
                                                  perms);
                                    vector<WickString> rw;
                                    bool fr = false, fs = false;
                                    for (auto &term : new_terms) {
                                        if (term.first == wr.tensors[!iwr]) {
                                            fr = true;
                                            WickExpr we =
                                                term.second * wr.factor;
                                            rw.insert(rw.end(),
                                                      we.terms.begin(),
                                                      we.terms.end());
                                            term.second.terms.clear();
                                        }
                                        if (term.first == wt.tensors[!iws]) {
                                            fs = true;
                                            WickExpr we =
                                                term.second * (zperm.negative
                                                                   ? -wt.factor
                                                                   : wt.factor);
                                            rw.insert(rw.end(),
                                                      we.terms.begin(),
                                                      we.terms.end());
                                            term.second.terms.clear();
                                        }
                                    }
                                    if (!fr)
                                        rw.push_back(WickString(
                                            wr.tensors[!iwr], wr.factor));
                                    if (!fs)
                                        rw.push_back(WickString(
                                            wt.tensors[!iws], zperm.negative
                                                                  ? -wt.factor
                                                                  : wt.factor));
                                    new_terms.push_back(
                                        make_pair(wc, WickExpr(rw)));
                                    wr.factor = 1.0;
                                    wr.tensors[!iwr] = wc;
                                    if (iprint)
                                        cout << "found " << wc << " = "
                                             << new_terms.back().second << endl;
                                    found = true;
                                    break;
                                }
                            }
                }
                if (!found)
                    unique_strings.push_back(ws);
            }
            for (auto &wt : new_terms)
                if (wt.second.terms.size() != 0) {
                    gr.left.push_back(wt.first);
                    gr.right.push_back(wt.second);
                    gr.index_maps.push_back(
                        vector<pair<double, map<string, string>>>{
                            make_pair(1.0, map<string, string>())});
                }
            for (auto &x : unique_strings)
                other_strings.push_back(x);
            gr.left.push_back(left[ix]);
            gr.right.push_back(WickExpr(other_strings));
            gr.index_maps.push_back(index_maps[ix]);
        }
        return gr;
    }
    // merge terms in each expr related with index permutations
    WickGraph simplify_permutations() const {
        WickGraph gr;
        int tmp_start = get_intermediate_start();
        int ntg = threading->activate_global();
        for (int ix = 0; ix < n_terms(); ix++) {
            vector<map<string, string>> all_perms =
                WickTensor::get_all_index_permutations(left[ix].indices);
            vector<WickString> unique_terms;
            vector<vector<pair<double, int>>> unique_perms;
            vector<vector<WickString>> unique_sorted;
            vector<WickString> sorted(right[ix].terms.size());
#pragma omp parallel for schedule(static) num_threads(ntg)
            for (int k = 0; k < (int)right[ix].terms.size(); k++)
                sorted[k] = right[ix].terms[k].abs().quick_sort();
            // get unique terms, and other terms can be expressed as index perm
            // of unique terms
            for (int k = 0; k < (int)right[ix].terms.size(); k++) {
                const WickString &wt = right[ix].terms[k];
                int ip = -1, iu = -1;
                double factor = 0.0;
                bool out_found = false;
                for (iu = 0; iu < (int)unique_terms.size(); iu++) {
                    bool found = false;
                    for (ip = 0; ip < all_perms.size(); ip++) {
                        const WickString &pd = unique_sorted[iu][ip];
                        if (pd.abs_equal_to(sorted[k])) {
                            found = true;
                            factor = wt.factor * sorted[k].factor * pd.factor;
                            break;
                        }
                    }
                    if (found) {
                        out_found = true;
                        break;
                    }
                }
                if (!out_found) {
                    unique_terms.push_back(wt.abs());
                    unique_sorted.push_back(
                        vector<WickString>(all_perms.size()));
#pragma omp parallel for schedule(static) num_threads(ntg)
                    for (int ik = 0; ik < all_perms.size(); ik++)
                        unique_sorted.back()[ik] = unique_terms.back()
                                                       .index_map(all_perms[ik])
                                                       .abs()
                                                       .quick_sort();
                    unique_perms.push_back(
                        vector<pair<double, int>>{make_pair(wt.factor, -1)});
                } else
                    unique_perms[iu].push_back(make_pair(factor, ip));
            }
            // normalize perm coefficients
            for (int it = 0; it < (int)unique_perms.size(); it++) {
                double tf = unique_perms[it][0].first;
                unique_terms[it] = unique_terms[it] * tf;
                for (auto &x : unique_perms[it])
                    x.first /= tf;
            }
            // for each unique term, sort all its perms
            for (int it = 0; it < (int)unique_perms.size(); it++)
                sort(
                    unique_perms[it].begin(), unique_perms[it].end(),
                    [](const pair<double, int> &a, const pair<double, int> &b) {
                        return a.second < b.second;
                    });
            // sort all unique terms
            vector<int> unique_idxs(unique_perms.size());
            for (int it = 0; it < (int)unique_perms.size(); it++)
                unique_idxs[it] = it;
            auto fup = [&unique_perms](int a, int b) {
                if (unique_perms[a].size() != unique_perms[b].size())
                    return unique_perms[a].size() < unique_perms[b].size();
                for (int ip = 0; ip < unique_perms[a].size(); ip++)
                    if (unique_perms[a][ip].second !=
                        unique_perms[b][ip].second)
                        return unique_perms[a][ip].second <
                               unique_perms[b][ip].second;
                for (int ip = 0; ip < unique_perms[a].size(); ip++)
                    if (abs(unique_perms[a][ip].first -
                            unique_perms[b][ip].first) > 1E-12)
                        return unique_perms[a][ip].first <
                               unique_perms[b][ip].first;
                return false;
            };
            sort(unique_idxs.begin(), unique_idxs.end(), fup);
            vector<WickString> vw;
            for (int ii = 0; ii < (int)unique_idxs.size(); ii++) {
                if (ii == 0 || fup(unique_idxs[ii - 1], unique_idxs[ii])) {
                    WickTensor lt = left[ix];
                    stringstream name;
                    name << intermediate_name << ++tmp_start;
                    lt.name = name.str();
                    vw.push_back(WickString(vector<WickTensor>{lt}));
                    gr.left.push_back(lt);
                    gr.right.push_back(WickExpr(unique_terms[unique_idxs[ii]]));
                    vector<pair<double, map<string, string>>> imx;
                    for (auto &px : index_maps[ix])
                        for (auto &x : unique_perms[unique_idxs[ii]])
                            imx.push_back(make_pair(
                                x.first * px.first,
                                x.second == -1
                                    ? px.second
                                    : merge_index_maps(all_perms[x.second],
                                                       px.second)));
                    gr.index_maps.push_back(imx);
                } else
                    gr.right.back().terms.push_back(
                        unique_terms[unique_idxs[ii]]);
            }
            if (vw.size() != 1) {
                gr.left.push_back(left[ix]);
                gr.right.push_back(WickExpr(vw));
                gr.index_maps.push_back(
                    vector<pair<double, map<string, string>>>{
                        make_pair(1.0, map<string, string>())});
            } else
                gr.left.back() = left[ix];
        }
        threading->activate_normal();
        return gr;
    }
    WickGraph expand_permutations() const {
        WickGraph gr;
        for (int ix = 0; ix < n_terms(); ix++) {
            WickExpr wx;
            for (int i = 0; i < index_maps[ix].size(); i++)
                wx = wx + right[ix].index_map(index_maps[ix][i].second) *
                              index_maps[ix][i].first;
            gr.left.push_back(left[ix]);
            gr.right.push_back(wx);
            gr.index_maps.push_back(vector<pair<double, map<string, string>>>{
                make_pair(1.0, map<string, string>())});
        }
        return gr;
    }
    WickGraph expand_binary() const {
        WickGraph gr;
        map<string, pair<WickTensor, WickExpr>> defs;
        for (int ix = 0; ix < n_terms(); ix++) {
            assert(index_maps[ix].size() == 1);
            WickExpr r = right[ix].substitute(defs);
            if (left[ix].name.substr(0, intermediate_name.length()) ==
                intermediate_name)
                defs[left[ix].name] = make_pair(left[ix], r);
            else {
                gr.left.push_back(left[ix]);
                gr.right.push_back(r);
                gr.index_maps.push_back(index_maps[ix]);
            }
        }
        return gr;
    }
    WickGraph topological_sort() const {
        map<pair<string, int>, int> mvert;
        for (int ix = 0; ix < n_terms(); ix++)
            mvert[make_pair(left[ix].name, (int)left[ix].indices.size())] = ix;
        vector<set<int>> edges(n_terms());
        vector<set<int>> inv_edges(n_terms());
        vector<int> uroots(n_terms(), 0), uout(n_terms(), 0);
        vector<double> tscale(n_terms(), 1);
        for (int ix = 0; ix < n_terms(); ix++)
            for (auto &ws : right[ix].terms)
                for (auto &wt : ws.tensors)
                    if (wt.name.substr(0, intermediate_name.length()) ==
                            intermediate_name &&
                        mvert.count(make_pair(wt.name, (int)wt.indices.size())))
                        edges[mvert.at(
                                  make_pair(wt.name, (int)wt.indices.size()))]
                            .insert(ix);
        for (int ix = 0; ix < n_terms(); ix++)
            for (auto &ig : edges[ix])
                uroots[ig]++, uout[ix]++, inv_edges[ig].insert(ix);
        for (int ix = 0; ix < n_terms(); ix++)
            for (auto &x : left[ix].indices)
                tscale[ix] *= idx_scales.count(x.types)
                                  ? idx_scales.at(x.types)
                                  : idx_scales.at(WickIndexTypes::None);
        double mscale = tscale.size() == 0
                            ? 0.0
                            : accumulate(tscale.begin(), tscale.end(), 0.0);
        vector<int> g;
        vector<int> tt;
        for (int i = 0; i < n_terms(); i++)
            if (uroots[i] == 0)
                tt.push_back(i);
        size_t tx = 0;
        auto fw = [&tscale, &uout, &inv_edges, &edges, &uroots,
                   &mscale](int a) -> vector<double> {
            double rs = 0.0, rd = 0.0, rp = 0.0, rg = 0.0;
            for (auto &g : inv_edges[a])
                rs -= (uout[g] == 1) * tscale[g],
                    rd -= (uroots[g] == 0) * tscale[g];
            for (auto &g : edges[a]) {
                for (auto &gg : inv_edges[g])
                    rp -= (gg != a && uroots[gg] == 0) * tscale[g];
                rg -= (uout[g] == 1) * tscale[g];
            }
            return vector<double>{rs == 0.0 ? mscale : rs + tscale[a],
                                  rd == 0.0 ? mscale : rd + tscale[a],
                                  rp == 0.0 ? mscale : rp + tscale[a],
                                  rg == 0.0 ? mscale : rg + tscale[a]};
        };
        while (tx != tt.size()) {
            sort(tt.begin() + tx, tt.end(),
                 [&fw](int a, int b) { return fw(a) < fw(b); });
            int ti = tt[tx++];
            g.push_back(ti);
            for (auto p : inv_edges[ti])
                uout[p]--;
            for (auto p : edges[ti]) {
                uroots[p]--;
                if (uroots[p] == 0)
                    tt.push_back(p);
            }
        }
        WickGraph gr;
        if ((int)g.size() != n_terms())
            throw runtime_error("Cannot find any topological order.");
        else {
            for (auto p : g) {
                gr.left.push_back(left[p]);
                gr.right.push_back(right[p]);
                gr.index_maps.push_back(index_maps[p]);
            }
        }
        return gr;
    }
    WickGraph simplify() const {
        return simplify_permutations()
            .simplify_binary_sort()
            .simplify_binary_split()
            .simplify_binary_unique()
            .simplify_binary_factor()
            .topological_sort();
    }
    string to_einsum() const {
        stringstream ss;
        int tmp_start = get_intermediate_start();
        auto fsuf = [this](const WickTensor &term) -> string {
            if (term.name.substr(0, this->intermediate_name.length()) ==
                this->intermediate_name)
                return "";
            stringstream ss;
            if (term.type == WickTensorTypes::KroneckerDelta ||
                term.type == WickTensorTypes::Tensor)
                for (auto &wi : term.indices)
                    ss << to_str(wi.types);
            return ss.str();
        };
        auto long_term = [](const WickExpr &expr,
                            const vector<WickIndex> &left_idx) -> bool {
            bool need_einsum = false;
            for (auto &term : expr.terms) {
                int nsz = 0;
                for (auto &wt : term.tensors)
                    nsz += wt.indices.size() != 0;
                if ((need_einsum = nsz > 2))
                    break;
                if (term.ctr_indices.size() == 0)
                    continue;
                map<WickIndex, int> idx_cnt;
                for (auto &wt : term.tensors)
                    for (auto &wi : wt.indices)
                        idx_cnt[wi]++;
                for (auto &wi : idx_cnt)
                    if ((need_einsum = (wi.second > 2 ||
                                        (wi.second == 2 &&
                                         !term.ctr_indices.count(wi.first)))))
                        break;
                if (need_einsum)
                    break;
                for (auto &wi : left_idx)
                    if ((need_einsum = (idx_cnt.count(wi) != 1)))
                        break;
                if (need_einsum)
                    break;
            }
            return need_einsum;
        };
        // creation time of each temp
        map<string, int> xcre;
        vector<vector<pair<int, int>>> extra(n_terms());
        for (int ix = 0; ix < n_terms(); ix++)
            if (left[ix].name.substr(0, intermediate_name.length()) ==
                intermediate_name)
                xcre[left[ix].name] = ix;
        // new graph: use each temp as soon as possible
        for (int ix = 0; ix < n_terms(); ix++)
            if (!long_term(right[ix], left[ix].indices) &&
                right[ix].terms.size() > 1) {
                vector<int> pps(right[ix].terms.size(), -1);
                int mps = -1;
                for (int it = 0; it < (int)right[ix].terms.size(); it++) {
                    for (auto &wt : right[ix].terms[it].tensors)
                        if (xcre.count(wt.name))
                            pps[it] = max(pps[it], xcre.at(wt.name));
                    mps = mps == -1 ? pps[it] : min(pps[it], mps);
                }
                mps = mps == -1 ? ix : mps;
                for (int it = 0; it < (int)right[ix].terms.size(); it++)
                    extra[pps[it] == -1 ? mps : pps[it]].push_back(
                        make_pair(ix, it));
            }
        WickGraph nwg;
        vector<int> tcount(n_terms(), 0);
        vector<int8_t> long_first;
        for (int ix = 0; ix < n_terms(); ix++) {
            if (long_term(right[ix], left[ix].indices) ||
                right[ix].terms.size() <= 1) {
                nwg.left.push_back(left[ix]);
                nwg.right.push_back(right[ix]);
                nwg.index_maps.push_back(index_maps[ix]);
                long_first.push_back(true);
            }
            for (int iex = 0; iex < (int)extra[ix].size(); iex++) {
                int ixx = extra[ix][iex].first, itx = extra[ix][iex].second;
                nwg.left.push_back(left[ixx]);
                nwg.right.push_back(WickExpr(right[ixx].terms[itx]));
                long_first.push_back(tcount[ixx] == 0);
                tcount[ixx]++;
                if (tcount[ixx] == (int)right[ixx].terms.size())
                    nwg.index_maps.push_back(index_maps[ixx]);
                else
                    nwg.index_maps.push_back(
                        vector<pair<double, map<string, string>>>{
                            make_pair(1.0, map<string, string>())});
            }
        }
        // last use time for each temp
        map<string, int> xdes;
        for (int ix = 0; ix < nwg.n_terms(); ix++)
            for (auto &term : nwg.right[ix].terms)
                for (auto &wt : term.tensors)
                    if (wt.name.substr(0, intermediate_name.length()) ==
                        intermediate_name)
                        xdes[wt.name] = xdes.count(wt.name)
                                            ? max(xdes.at(wt.name), ix)
                                            : ix;
        for (int ix = 0; ix < nwg.n_terms(); ix++) {
            bool need_einsum = long_term(nwg.right[ix], nwg.left[ix].indices);
            map<string, int> lidx_map;
            for (int i = 0; i < (int)nwg.left[ix].indices.size(); i++)
                lidx_map[nwg.left[ix].indices[i].name] = i;
            bool is_int =
                nwg.left[ix].name.substr(0, intermediate_name.length()) ==
                intermediate_name;
            if (need_einsum)
                ss << nwg.right[ix].to_einsum(
                    nwg.left[ix], is_int && long_first[ix], intermediate_name);
            else {
                for (int ip = 0; ip < (int)nwg.right[ix].terms.size(); ip++) {
                    bool is_eq = ip == 0 && is_int && long_first[ix];
                    ss << nwg.left[ix].name;
                    ss << (is_eq ? " = " : " += ");
                    const auto &term = nwg.right[ix].terms[ip];
                    // prefactor
                    int n_const = 0;
                    for (auto &wt : term.tensors)
                        if (wt.indices.size() == 0)
                            n_const++;
                    if (term.tensors.size() == 0 ||
                        n_const == term.tensors.size()) {
                        if (n_const == 0)
                            ss << term.factor << "\n";
                        else {
                            if (term.factor != 1.0)
                                ss << term.factor << " * ";
                            int i_const = 0;
                            for (auto &wt : term.tensors)
                                if (wt.indices.size() == 0)
                                    ss << wt.name
                                       << (++i_const == n_const ? "\n" : " * ");
                        }
                        continue;
                    }
                    if (term.factor != 1.0)
                        ss << term.factor << " * ", is_eq = false;
                    vector<WickTensor> eff_tensors;
                    vector<WickIndex> out_idx;
                    for (auto &wt : term.tensors)
                        if (wt.indices.size() == 0)
                            ss << wt.name << " * ", is_eq = false;
                        else
                            eff_tensors.push_back(wt);
                    // contraction
                    if (eff_tensors.size() == 1) {
                        ss << eff_tensors[0].name << fsuf(eff_tensors[0]);
                        out_idx = eff_tensors[0].indices;
                    } else if (term.ctr_indices.size() == 0) {
                        vector<string> names = vector<string>{
                            eff_tensors[0].name + fsuf(eff_tensors[0]),
                            eff_tensors[1].name + fsuf(eff_tensors[1])};
                        // partial trace
                        for (int ik = 0; ik < 2; ik++)
                            while (
                                set<WickIndex>(eff_tensors[ik].indices.begin(),
                                               eff_tensors[ik].indices.end())
                                    .size() != eff_tensors[ik].indices.size()) {
                                map<WickIndex, int> mwi;
                                int axa = 0, axb = 0;
                                for (int iw = 0;
                                     iw < (int)eff_tensors[ik].indices.size();
                                     iw++) {
                                    if (mwi.count(
                                            eff_tensors[ik].indices[iw])) {
                                        axa =
                                            mwi.at(eff_tensors[ik].indices[iw]),
                                        axb = iw;
                                        break;
                                    }
                                    mwi[eff_tensors[ik].indices[iw]] = iw;
                                }
                                vector<WickIndex> nww;
                                for (int iw = 0;
                                     iw < (int)eff_tensors[ik].indices.size();
                                     iw++)
                                    if (iw != axa && iw != axb)
                                        nww.push_back(
                                            eff_tensors[ik].indices[iw]);
                                nww.push_back(eff_tensors[ik].indices[axa]);
                                stringstream wss;
                                wss << "np.diagonal(" << names[ik] << ", 0, "
                                    << axa << ", " << axb << ")";
                                names[ik] = wss.str();
                                eff_tensors[ik].indices = nww;
                            }
                        // match dims for elementwise multiplication
                        out_idx = nwg.left[ix].indices;
                        for (int ik = 0; ik < 2; ik++) {
                            map<WickIndex, int> mwi;
                            for (int iw = 0;
                                 iw < (int)eff_tensors[ik].indices.size(); iw++)
                                mwi[eff_tensors[ik].indices[iw]] = iw;
                            vector<int> tr;
                            vector<string> ntr;
                            bool need_tr = false;
                            for (int iw = 0, iwg = 0; iw < (int)out_idx.size();
                                 iw++)
                                if (mwi.count(out_idx[iw])) {
                                    tr.push_back(mwi.at(out_idx[iw]));
                                    need_tr = need_tr ||
                                              tr.back() != (int)(tr.size() - 1);
                                    ntr.push_back(":");
                                } else
                                    ntr.push_back("None");
                            stringstream wss;
                            wss << names[ik];
                            if (need_tr) {
                                wss << ".transpose(";
                                for (int iw = 0; iw < (int)tr.size(); iw++)
                                    wss << tr[iw]
                                        << (iw == (int)tr.size() - 1 ? ")"
                                                                     : ", ");
                            }
                            if (eff_tensors[ik].indices.size() <
                                out_idx.size()) {
                                wss << "[";
                                for (int iw = 0; iw < (int)ntr.size(); iw++)
                                    wss << ntr[iw]
                                        << (iw == (int)ntr.size() - 1 ? "]"
                                                                      : ", ");
                            }
                            names[ik] = wss.str();
                        }
                        ss << "np.multiply(" << names[0] << ", " << names[1]
                           << ")";
                    } else {
                        ss << "np.tensordot(" << eff_tensors[0].name
                           << fsuf(eff_tensors[0]) << ", "
                           << eff_tensors[1].name << fsuf(eff_tensors[1])
                           << ", axes=(";
                        map<WickIndex, int> lla, llb;
                        for (int i = 0; i < (int)eff_tensors[0].indices.size();
                             i++)
                            lla[eff_tensors[0].indices[i]] = i;
                        for (int i = 0; i < (int)eff_tensors[1].indices.size();
                             i++)
                            llb[eff_tensors[1].indices[i]] = i;
                        ss << (term.ctr_indices.size() == 1 ? "" : "(");
                        int ib = 0, ibg = (int)term.ctr_indices.size() - 1;
                        for (auto &mc : term.ctr_indices)
                            ss << lla[mc] << (ib++ == ibg ? "" : ", ");
                        ss << (term.ctr_indices.size() == 1 ? ", " : "), (");
                        ib = 0;
                        for (auto &mc : term.ctr_indices)
                            ss << llb[mc] << (ib++ == ibg ? "" : ", ");
                        ss << (term.ctr_indices.size() == 1 ? "))" : ")))");
                        for (int j = 0; j < 2; j++)
                            for (int i = 0;
                                 i < (int)eff_tensors[j].indices.size(); i++)
                                if (!term.ctr_indices.count(
                                        eff_tensors[j].indices[i]))
                                    out_idx.push_back(
                                        eff_tensors[j].indices[i]);
                        assert(out_idx.size() == nwg.left[ix].indices.size());
                    }
                    // final transpose
                    map<WickIndex, int> mtr;
                    for (int ig = 0; ig < (int)out_idx.size(); ig++)
                        mtr[out_idx[ig]] = ig;
                    vector<int> tr(out_idx.size());
                    bool no_trans = true;
                    for (int ig = 0; ig < (int)nwg.left[ix].indices.size();
                         ig++)
                        tr[ig] = mtr.at(nwg.left[ix].indices[ig]),
                        no_trans = no_trans && (tr[ig] == ig);
                    if (!no_trans) {
                        ss << ".transpose(";
                        for (int ir = 0; ir < (int)tr.size(); ir++)
                            ss << tr[ir]
                               << (ir == (int)tr.size() - 1 ? "" : ", ");
                        ss << ")";
                    }
                    if (eff_tensors.size() == 1 && is_eq)
                        ss << ".copy()";
                    // ss << " # " << left[ix] << " :: " << term;
                    ss << "\n";
                }
            }
            // handle permutations
            if (nwg.index_maps[ix].size() != 1) {
                ++tmp_start;
                ss << intermediate_name << tmp_start << " = "
                   << nwg.left[ix].name << ".copy()\n";
                for (int ii = 0; ii < (int)nwg.index_maps[ix].size(); ii++) {
                    if (ii == 0)
                        ss << nwg.left[ix].name << "[:] = ";
                    else if (is_int && need_einsum)
                        ss << nwg.left[ix].name << " = " << nwg.left[ix].name
                           << " + ";
                    else
                        ss << nwg.left[ix].name << " += ";
                    if (nwg.index_maps[ix][ii].first != 1.0)
                        ss << nwg.index_maps[ix][ii].first << " * ";
                    ss << intermediate_name << tmp_start;
                    if (nwg.index_maps[ix][ii].second.size() != 0) {
                        ss << ".transpose(";
                        vector<int> tr(nwg.left[ix].indices.size());
                        for (auto &mr : nwg.index_maps[ix][ii].second)
                            tr[lidx_map.at(mr.second)] = lidx_map.at(mr.first);
                        for (int ir = 0; ir < (int)tr.size(); ir++)
                            ss << tr[ir]
                               << (ir == (int)tr.size() - 1 ? "" : ", ");
                        ss << ")";
                    }
                    ss << "\n";
                }
                ss << intermediate_name << tmp_start << " = None\n";
            }
            // release temp memory
            set<string> names;
            for (auto &term : nwg.right[ix].terms)
                for (auto &wt : term.tensors)
                    if (xdes.count(wt.name) && xdes.at(wt.name) == ix &&
                        !names.count(wt.name))
                        ss << wt.name << " = None\n", names.insert(wt.name);
        }
        return ss.str();
    }
    WickGraph expand() const { return expand_permutations().expand_binary(); }
    friend ostream &operator<<(ostream &os, const WickGraph &wg) {
        os << "GRAPH /" << wg.n_terms() << "/";
        if (wg.n_terms() != 0)
            os << endl;
        for (int ig = 0; ig < wg.n_terms(); ig++) {
            os << setw(4) << ig << " :: " << wg.left[ig] << " ";
            for (auto &p : wg.left[ig].perms)
                os << p << " ";
            os << "= EXPR /" << wg.right[ig].terms.size() << "/ ";
            if (wg.index_maps[ig].size() != 1) {
                os << "{ ";
                for (int igg = 0; igg < (int)wg.index_maps[ig].size(); igg++) {
                    const auto &mx = wg.index_maps[ig][igg];
                    if (abs(mx.first - 1.0) < 1E-12)
                        os << (igg == 0 ? "" : " + ")
                           << (mx.second.size() == 0 ? "1" : "");
                    else if (abs(mx.first + 1.0) < 1E-12)
                        os << " - " << (mx.second.size() == 0 ? "1" : "");
                    else
                        os << (igg == 0 ? "" : " + ") << "(" << fixed
                           << setprecision(10) << setw(16) << mx.first << ") ";
                    if (mx.second.size() != 0) {
                        os << "P[";
                        for (auto &mmx : mx.second)
                            os << mmx.first;
                        os << "->";
                        for (auto &mmx : mx.second)
                            os << mmx.second;
                        os << "]";
                    }
                }
                os << " }";
            }
            os << endl;
            for (int i = 0; i < (int)wg.right[ig].terms.size(); i++)
                os << setw(4) << " " << wg.right[ig].terms[i] << endl;
        }
        return os;
    }
};

} // namespace block2

namespace std {

template <> struct hash<block2::WickPermutation> {
    size_t operator()(const block2::WickPermutation &x) const noexcept {
        return x.hash();
    }
};

} // namespace std
