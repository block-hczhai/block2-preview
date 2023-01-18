
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
    Active = 2,
    External = 4,
    Alpha = 8,
    Beta = 16,
    AlphaBeta = 8 | 16,
    InactiveAlpha = 8 | 1,
    ActiveAlpha = 8 | 2,
    ExternalAlpha = 8 | 4,
    InactiveBeta = 16 | 1,
    ActiveBeta = 16 | 2,
    ExternalBeta = 16 | 4,
};

inline string to_str(const WickIndexTypes c) {
    const static string repr[] = {"N", "I", "A", "IA", "E", "EI", "EA", "EIA",
                                  "A", "i", "a", "ia", "e", "ei", "ea", "eia",
                                  "B", "I", "A", "IA", "E", "EI", "EA", "EIA"};
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
    WickIndex(const char name[]) : name(name), types(WickIndexTypes::None) {}
    WickIndex(const string &name) : name(name), types(WickIndexTypes::None) {}
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
};

struct WickPermutation {
    vector<int16_t> data;
    bool negative;
    WickPermutation() : negative(false) {}
    WickPermutation(const vector<int16_t> &data, bool negative = false)
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
        for (auto &perm : perms) {
            bool valid = true;
            for (int i = 0; i < (int)indices.size() && valid; i++)
                if ((indices[perm.data[i]].types & indices[i].types) ==
                        WickIndexTypes::None &&
                    indices[perm.data[i]].types != WickIndexTypes::None &&
                    indices[i].types != WickIndexTypes::None)
                    valid = false;
            if (valid)
                rperms.push_back(perm);
        }
        return rperms;
    }
    static WickTensor
    parse(const string &tex_expr,
          const map<WickIndexTypes, set<WickIndex>> &idx_map,
          const map<pair<string, int>, vector<WickPermutation>> &perm_map) {
        string name, indices;
        bool is_name = true;
        for (char c : tex_expr)
            if (c == '_' || c == '[')
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
        const WickIndexTypes mask = WickIndexTypes::Inactive |
                                    WickIndexTypes::Active |
                                    WickIndexTypes::External;
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
};

struct WickString {
    vector<WickTensor> tensors;
    set<WickIndex> ctr_indices;
    double factor;
    WickString() : factor(0.0) {}
    WickString(const WickTensor &tensor, double factor = 1.0)
        : factor(factor), tensors({tensor}), ctr_indices() {}
    WickString(const vector<WickTensor> &tensors)
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
    WickString index_map(const map<string, string> &maps) {
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
        set<WickIndex> a_idxs = used_indices(), b_idxs = other.used_indices();
        vector<WickIndex> used_idxs_v(a_idxs.size() + b_idxs.size());
        auto it = set_union(a_idxs.begin(), a_idxs.end(), b_idxs.begin(),
                            b_idxs.end(), used_idxs_v.begin());
        set<WickIndex> used_idxs(used_idxs_v.begin(), it);
        vector<WickIndex> a_rep(ctr_indices.size()),
            b_rep(other.ctr_indices.size()), c_rep(ctr_indices.size());
        it = set_intersection(ctr_indices.begin(), ctr_indices.end(),
                              b_idxs.begin(), b_idxs.end(), a_rep.begin());
        a_rep.resize(it - a_rep.begin());
        it =
            set_intersection(other.ctr_indices.begin(), other.ctr_indices.end(),
                             a_idxs.begin(), a_idxs.end(), b_rep.begin());
        b_rep.resize(it - b_rep.begin());
        it = set_intersection(ctr_indices.begin(), ctr_indices.end(),
                              other.ctr_indices.begin(),
                              other.ctr_indices.end(), c_rep.begin());
        c_rep.resize(it - c_rep.begin());
        set<WickIndex> xa_rep(a_rep.begin(), a_rep.end()),
            xb_rep(b_rep.begin(), b_rep.end()),
            xc_rep(c_rep.begin(), c_rep.end());
        map<WickIndex, WickIndex> mp_idxs;
        for (auto &idx : used_idxs)
            if (xa_rep.count(idx) || xb_rep.count(idx))
                for (int i = 1; i < 100; i++) {
                    WickIndex g = idx;
                    g.name[0] += i;
                    if (!used_idxs.count(g)) {
                        used_idxs.insert(g);
                        mp_idxs[idx] = g;
                        break;
                    }
                }
        // change contraction index in a, if it is also free index in b
        for (int i = 0; i < tensors.size(); i++)
            for (auto &wi : xtensors[i].indices)
                if (mp_idxs.count(wi) && xa_rep.count(wi) && !xc_rep.count(wi))
                    wi = mp_idxs[wi];
        // change contraction index in b,
        // if it is also free index or contraction index in a
        for (int i = tensors.size(); i < (int)xtensors.size(); i++)
            for (auto &wi : xtensors[i].indices)
                if (mp_idxs.count(wi) && xb_rep.count(wi))
                    wi = mp_idxs[wi];
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
                    for (int i = tensors.size(); i < (int)xtensors.size(); i++)
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
            if (mp_idxs.count(wi) && xa_rep.count(wi) && !xc_rep.count(wi))
                xctr_indices.insert(mp_idxs[wi]);
            else
                xctr_indices.insert(wi);
        for (auto &wi : other.ctr_indices)
            if (mp_idxs.count(wi) && xb_rep.count(wi))
                xctr_indices.insert(mp_idxs[wi]);
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
        ot_tensor_groups.push_back(ot_tensors.size());
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
                sf_idx_sorted[i] = i;
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
            cd_tensor_groups.push_back(cd_tensors.size());
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
                    arg_idx[i] = i + cd_tensor_groups[ig];
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
};

struct WickExpr {
    vector<WickString> terms;
    WickExpr() {}
    WickExpr(const WickString &term) : terms(vector<WickString>{term}) {}
    WickExpr(const vector<WickString> &terms) : terms(terms) {}
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
    string to_einsum(const WickTensor &x) const {
        stringstream ss;
        bool first = true;
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
            ss << x.name << (first ? " += " : " += ");
            first = false;
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
                    wt.type == WickTensorTypes::Tensor)
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
        return terms;
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
        WickIndexTypes check_mask = WickIndexTypes::Inactive |
                                    WickIndexTypes::Active |
                                    WickIndexTypes::External;
        vector<WickIndexTypes> check_types = {WickIndexTypes::Inactive,
                                              WickIndexTypes::Active,
                                              WickIndexTypes::External};
        for (int i = 0; i < (int)vidxs.size(); i++) {
            int k = 0, nk = xctr_idxs.size();
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
                    for (auto &wii : xctr_idxs[i])
                        if (wi.with_no_types() == wii.with_no_types() &&
                            (wi.types & wii.types) != WickIndexTypes::None)
                            wi = wii;
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
                int sf_n = wt.indices.size() / 2;
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::cre(wt.indices[i]));
                    cd_idx_map.push_back(cd_idx_map.size() + sf_n);
                }
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::des(wt.indices[i + sf_n]));
                    cd_idx_map.push_back(cd_idx_map.size() - sf_n);
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
                        acc_sign[0] ^= (cd_tensors[j] < cd_tensors[i]);
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
                    acc_sign[l + 2] ^= (cd_tensors[d] < cd_tensors[c]);
                    for (int i = 0; i < (int)cd_tensors.size(); i++)
                        if (!cur_idxs_mask[i]) {
                            acc_sign[l + 2] ^=
                                (cd_tensors[max(c, i)] < cd_tensors[min(c, i)]);
                            acc_sign[l + 2] ^=
                                (cd_tensors[max(d, i)] < cd_tensors[min(d, i)]);
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
                int sf_n = cd_tensors.size() / 2, tn = sf_n - l - 1;
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
                    cd_spin_map[wt.get_spin_tag()].push_back(cd_tensors.size() -
                                                             1);
                }
            } else if (wt.type == WickTensorTypes::SpinFreeOperator) {
                int sf_n = wt.indices.size() / 2;
                // sign from reverse destroy operator
                init_sign ^= ((sf_n - 1) & 1) ^ (((sf_n - 1) & 2) >> 1);
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::cre(wt.indices[i]));
                    cd_idx_map.push_back(cd_idx_map.size() + sf_n);
                }
                for (int i = 0; i < sf_n; i++) {
                    cd_tensors.push_back(WickTensor::des(wt.indices[i + sf_n]));
                    cd_idx_map.push_back(cd_idx_map.size() - sf_n);
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
        vector<int8_t> cur_idxs_mask(cd_tensors.size(), 0);
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
                    int n_sf = tensor_idxs.size() / 2;
                    stable_sort(tensor_idxs.begin(), tensor_idxs.end(),
                                [&cd_tensors](int i, int j) {
                                    int ki = (cd_tensors[i].indices[0].types &
                                              WickIndexTypes::Active) !=
                                             WickIndexTypes::None;
                                    int kj = (cd_tensors[j].indices[0].types &
                                              WickIndexTypes::Active) !=
                                             WickIndexTypes::None;
                                    return ki < kj;
                                });
                    int n_act = 0;
                    for (auto &ti : tensor_idxs)
                        n_act +=
                            (cd_tensors[ti].indices[0].types &
                             WickIndexTypes::Active) != WickIndexTypes::None;
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
            double inact_fac = 1.0;
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
                    n_inact_a = n_inact_b = 0;
                    for (int i = 0; i < l; i++) {
                        tie(a, b) = cur_idxs[i];
                        n_inact_a +=
                            n_inactive_idxs_a[a] - n_inactive_idxs_a[a + 1];
                        n_inact_b +=
                            n_inactive_idxs_b[b] - n_inactive_idxs_b[b + 1];
                        inact_fac *= 1 << (cd_idx_map_rev[a] == b);
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
                    inact_fac *= 1 << (cd_idx_map_rev[c] == d);
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
                int sf_n = cd_tensors.size() / 2, tn = sf_n - l - 1;
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
                }
                for (int i = 0; i < (int)tensor_idxs.size(); i++)
                    if (!cur_idxs_mask[tensor_idxs[i]])
                        ot_tensors.push_back(cd_tensors[tensor_idxs[i]]);
            }
            r.terms.push_back(WickString(
                ot_tensors, x.ctr_indices,
                inact_fac *
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

struct WickGHF {
    vector<map<WickIndexTypes, set<WickIndex>>> idx_map; // aa, bb, ab, ba
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    WickGHF() {
        idx_map.resize(4);
        idx_map[0][WickIndexTypes::Alpha] = WickIndex::parse_set("ijkl");
        idx_map[1][WickIndexTypes::Beta] = WickIndex::parse_set("ijkl");
        idx_map[2][WickIndexTypes::Alpha] = WickIndex::parse_set("ij");
        idx_map[2][WickIndexTypes::Beta] = WickIndex::parse_set("kl");
        idx_map[3][WickIndexTypes::Beta] = WickIndex::parse_set("ij");
        idx_map[3][WickIndexTypes::Alpha] = WickIndex::parse_set("kl");
        perm_map[make_pair("v", 4)] = WickPermutation::qc_chem();
    }
    WickExpr make_h1b() const {
        WickExpr expr =
            WickExpr::parse("SUM <ij> h[ij] D[i] C[j]", idx_map[1], perm_map);
        return expr.expand().simplify();
    }
    WickExpr make_h2aa() const {
        WickExpr expr =
            0.5 * WickExpr::parse("SUM <ijkl> v[ijkl] C[i] C[k] D[l] D[j]",
                                  idx_map[0], perm_map);
        return expr.expand().simplify();
    }
    WickExpr make_h2bb() const {
        WickExpr expr =
            0.5 * WickExpr::parse("SUM <ijkl> v[ijkl] D[i] D[k] C[l] C[j]",
                                  idx_map[1], perm_map);
        return expr.expand().simplify();
    }
    WickExpr make_h2ab() const {
        WickExpr expr =
            0.5 * WickExpr::parse("SUM <ijkl> v[ijkl] C[i] D[k] C[l] D[j]",
                                  idx_map[2], perm_map);
        return expr.expand().simplify();
    }
    WickExpr make_h2ba() const {
        WickExpr expr =
            0.5 * WickExpr::parse("SUM <ijkl> v[ijkl] D[i] C[k] D[l] C[j]",
                                  idx_map[3], perm_map);
        return expr.expand().simplify();
    }
};

struct WickCCSD {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    WickExpr h1, h2, h, t1, t2, t, ex1, ex2;
    WickCCSD(bool anti_integral = true) {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("pqrsijklmno");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("pqrsabcdefg");
        perm_map[make_pair("v", 4)] = anti_integral
                                          ? WickPermutation::four_anti()
                                          : WickPermutation::qc_phys();
        perm_map[make_pair("t", 2)] = WickPermutation::non_symmetric();
        perm_map[make_pair("t", 4)] = WickPermutation::four_anti();
        h1 = WickExpr::parse("SUM <pq> h[pq] C[p] D[q]", idx_map, perm_map);
        h2 = (anti_integral ? 0.25 : 0.5) *
             WickExpr::parse("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]", idx_map,
                             perm_map);
        t1 = WickExpr::parse("SUM <ai> t[ai] C[a] D[i]", idx_map, perm_map);
        t2 = 0.25 * WickExpr::parse("SUM <abij> t[abij] C[a] C[b] D[j] D[i]",
                                    idx_map, perm_map);
        ex1 = WickExpr::parse("C[i] D[a]", idx_map, perm_map);
        ex2 = WickExpr::parse("C[i] C[j] D[b] D[a]", idx_map, perm_map);
        h = (h1 + h2).expand(-1, true).simplify();
        t = (t1 + t2).expand(-1, true).simplify();
    }
    // h + [h, t] + 0.5 [[h, t1], t1]
    WickExpr energy_equations(int order = 2) const {
        vector<WickExpr> hx(5, h);
        WickExpr amp = h;
        for (int i = 0; i < order; amp = amp + hx[++i])
            hx[i + 1] = (1.0 / (i + 1)) *
                        (hx[i] ^ t).expand((order - 1 - i) * 2).simplify();
        return amp.expand(0).simplify();
    }
    // ex1 * (h + [h, t] + 0.5 [[h, t], t] + (1/6) [[[h2, t1], t1], t1])
    WickExpr t1_equations(int order = 4) const {
        vector<WickExpr> hx(5, h);
        WickExpr amp = h;
        for (int i = 0; i < order; amp = amp + hx[++i])
            hx[i + 1] = (1.0 / (i + 1)) *
                        (hx[i] ^ t).expand((order - i) * 2).simplify();
        return (ex1 * amp).expand(0).simplify();
    }
    // MEST Eq. (5.7.16)
    WickExpr t2_equations(int order = 4) const {
        vector<WickExpr> hx(5, h);
        WickExpr amp = h;
        for (int i = 0; i < order; amp = amp + hx[++i])
            hx[i + 1] = (1.0 / (i + 1)) *
                        (hx[i] ^ t).expand((order - i) * 4).simplify();
        return (ex2 * amp).expand(0).simplify();
    }
};

struct WickUGACCSD {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    map<string, pair<WickTensor, WickExpr>> defs;
    WickExpr h1, h2, e0, h, t1, t2, t, ex1, ex2;
    WickUGACCSD() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("pqrsijklmno");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("pqrsabcdefg");
        perm_map[make_pair("v", 4)] = WickPermutation::qc_phys();
        perm_map[make_pair("t", 2)] = WickPermutation::non_symmetric();
        perm_map[make_pair("t", 4)] = WickPermutation::pair_symmetric(2, false);
        // def of fock matrix
        defs["h"] = WickExpr::parse_def(
            "h[pq] = f[pq] \n - 2.0 SUM <j> v[pjqj] \n + SUM <j> v[pjjq]",
            idx_map, perm_map);
        h1 = WickExpr::parse("SUM <pq> h[pq] E1[p,q]", idx_map, perm_map)
                 .substitute(defs);
        h2 = WickExpr::parse("0.5 SUM <pqrs> v[pqrs] E2[pq,rs]", idx_map,
                             perm_map);
        t1 = WickExpr::parse("SUM <ai> t[ai] E1[a,i]", idx_map, perm_map);
        t2 = WickExpr::parse("0.5 SUM <abij> t[abij] E1[a,i] E1[b,j]", idx_map,
                             perm_map);
        ex1 = WickExpr::parse("E1[i,a]", idx_map, perm_map);
        ex2 = WickExpr::parse("E1[i,a] E1[j,b]", idx_map, perm_map);
        // hartree-fock reference
        e0 =
            WickExpr::parse(
                "2 SUM <i> h[ii] \n + 2 SUM <ij> v[ijij] \n - SUM <ij> v[ijji]",
                idx_map, perm_map)
                .substitute(defs);
        h = (h1 + h2 - e0).simplify();
        t = (t1 + t2).simplify();
    }
    // h + [h, t] + 0.5 [[h, t1], t1]
    WickExpr energy_equations(int order = 2) const {
        vector<WickExpr> hx(5, h);
        WickExpr amp = h;
        for (int i = 0; i < order; amp = amp + hx[++i])
            hx[i + 1] = (1.0 / (i + 1)) * (hx[i] ^ t);
        return amp.expand(0).simplify();
    }
    // ex1 * (h + [h, t] + 0.5 [[h, t], t] + (1/6) [[[h2, t1], t1], t1])
    WickExpr t1_equations(int order = 4) const {
        vector<WickExpr> hx(5, h);
        WickExpr amp = h;
        if (order >= 1)
            amp = amp + (h ^ t);
        if (order >= 2)
            amp = amp + 0.5 * ((h ^ t1) ^ t1) + ((h ^ t2) ^ t1);
        if (order >= 3)
            amp = amp + (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1);
        return (ex1 * amp).expand(0).simplify();
    }
    // J. Chem. Phys. 89, 7382 (1988) Eq. (17)
    WickExpr t2_equations(int order = 4) const {
        vector<WickExpr> hx(5, h);
        WickExpr amp = h;
        if (order >= 1)
            amp = amp + (h ^ t);
        if (order >= 2)
            amp = amp + 0.5 * ((h ^ t1) ^ t1) + ((h ^ t2) ^ t1) +
                  0.5 * ((h ^ t2) ^ t2);
        if (order >= 3)
            amp = amp + (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1) +
                  0.5 * (((h ^ t2) ^ t1) ^ t1);
        if (order >= 4)
            amp = amp + (1 / 24.0) * ((((h ^ t1) ^ t1) ^ t1) ^ t1);
        return (ex2 * amp).expand(0).simplify();
    }
};

struct WickSCNEVPT2 {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    map<string, pair<WickTensor, WickExpr>> defs;
    vector<pair<string, string>> sub_spaces;
    WickExpr heff, hw, hd;
    WickSCNEVPT2() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("mnxyijkl");
        idx_map[WickIndexTypes::Active] =
            WickIndex::parse_set("mnxyabcdefghpq");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("mnxyrstu");
        perm_map[make_pair("w", 4)] = WickPermutation::qc_phys();
        heff = WickExpr::parse("SUM <ab> h[ab] E1[a,b]", idx_map, perm_map);
        hw = WickExpr::parse("0.5 SUM <abcd> w[abcd] E2[ab,cd]", idx_map,
                             perm_map);
        hd = heff + hw;
        sub_spaces = {{"ijrs", "gamma[ij] gamma[rs] w[rsij] E1[r,i] E1[s,j] \n"
                               "gamma[ij] gamma[rs] w[rsji] E1[s,i] E1[r,j]"},
                      {"rsi", "SUM <a> gamma[rs] w[rsia] E1[r,i] E1[s,a] \n"
                              "SUM <a> gamma[rs] w[sria] E1[s,i] E1[r,a]"},
                      {"ijr", "SUM <a> gamma[ij] w[raji] E1[r,j] E1[a,i] \n"
                              "SUM <a> gamma[ij] w[raij] E1[r,i] E1[a,j]"},
                      {"rs", "SUM <ab> gamma[rs] w[rsba] E1[r,b] E1[s,a]"},
                      {"ij", "SUM <ab> gamma[ij] w[baij] E1[b,i] E1[a,j]"},
                      {"ir", "SUM <ab> w[raib] E1[r,i] E1[a,b] \n"
                             "SUM <ab> w[rabi] E1[a,i] E1[r,b] \n"
                             "h[ri] E1[r,i]"},
                      {"r", "SUM <abc> w[rabc] E1[r,b] E1[a,c] \n"
                            "SUM <a> h[ra] E1[r,a] \n"
                            "- SUM <ab> w[rbba] E1[r,a]"},
                      {"i", "SUM <abc> w[baic] E1[b,i] E1[a,c] \n"
                            "SUM <a> h[ai] E1[a,i]"}};
        defs["gamma"] = WickExpr::parse_def(
            "gamma[mn] = 1.0 \n - 0.5 delta[mn]", idx_map, perm_map);
    }
    WickExpr build_communicator(const string &bra, const string &ket,
                                bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xbra.conjugate() & (hd ^ xket).expand().simplify())
                   : (xbra.conjugate() * (hd ^ xket).expand().simplify());
        return expr.expand()
            .remove_external()
            .add_spin_free_trans_symm()
            .simplify();
    }
    WickExpr build_communicator(const string &ket, bool do_sum = true) const {
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xket.conjugate() & (hd ^ xket).expand().simplify())
                   : (xket.conjugate() * (hd ^ xket).expand().simplify());
        return expr.expand()
            .remove_external()
            .add_spin_free_trans_symm()
            .simplify();
    }
    WickExpr build_norm(const string &ket, bool do_sum = true) const {
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xket.conjugate() & xket) : (xket.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .simplify();
    }
    string to_einsum_orb_energies(const WickTensor &tensor) const {
        stringstream ss;
        ss << tensor.name << " = ";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (wi.types == WickIndexTypes::Inactive)
                ss << "(-1) * ";
            ss << "orbe" << to_str(wi.types);
            ss << "[";
            for (int j = 0; j < (int)tensor.indices.size(); j++) {
                ss << (i == j ? ":" : "None");
                if (j != (int)tensor.indices.size() - 1)
                    ss << ", ";
            }
            ss << "]";
            if (i != (int)tensor.indices.size() - 1)
                ss << " + ";
        }
        return ss.str();
    }
    string to_einsum_sum_restriction(const WickTensor &tensor) const {
        stringstream ss, sr;
        ss << "grid = np.indices((";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            ss << (wi.types == WickIndexTypes::Inactive ? "ncore" : "nvirt");
            if (i != (int)tensor.indices.size() - 1 || i == 0)
                ss << ", ";
            if (i != 0 &&
                tensor.indices[i].types == tensor.indices[i - 1].types)
                sr << "idx &= grid[" << i - 1 << "] <= grid[" << i << "]"
                   << endl;
        }
        ss << "))" << endl;
        return ss.str() + sr.str();
    }
    string to_einsum() const {
        stringstream ss;
        WickTensor norm, ener, deno;
        for (int i = 0; i < (int)sub_spaces.size(); i++) {
            string key = sub_spaces[i].first, expr = sub_spaces[i].second;
            stringstream sr;
            ss << "def compute_" << key << "():" << endl;
            norm = WickTensor::parse("norm[" + key + "]", idx_map, perm_map);
            ener = WickTensor::parse("hexp[" + key + "]", idx_map, perm_map);
            deno = WickTensor::parse("deno[" + key + "]", idx_map, perm_map);
            sr << to_einsum_orb_energies(deno) << endl;
            sr << "norm = np.zeros_like(deno)" << endl;
            sr << build_norm(expr, false).to_einsum(norm) << endl;
            sr << "hexp = np.zeros_like(deno)" << endl;
            sr << build_communicator(expr, false).to_einsum(ener) << endl;
            sr << "idx = abs(norm) > 1E-14" << endl;
            if (key.length() >= 2)
                sr << to_einsum_sum_restriction(deno) << endl;
            sr << "hexp[idx] = deno[idx] + hexp[idx] / norm[idx]" << endl;
            sr << "xener = -(norm[idx] / hexp[idx]).sum()" << endl;
            sr << "xnorm = norm[idx].sum()" << endl;
            sr << "return xnorm, xener" << endl;
            ss << WickExpr::to_einsum_add_indent(sr.str()) << endl;
        }
        return ss.str();
    }
};

struct WickICNEVPT2 : WickSCNEVPT2 {
    WickICNEVPT2() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("mnxyijkl");
        idx_map[WickIndexTypes::Active] =
            WickIndex::parse_set("mnxyabcdefghpq");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("mnxyrstu");
        perm_map[make_pair("w", 4)] = WickPermutation::qc_phys();
        WickExpr hi = WickExpr::parse("SUM <i> orbe[i] E1[i,i]\n"
                                      "SUM <r> orbe[r] E1[r,r]",
                                      idx_map, perm_map);
        heff = WickExpr::parse("SUM <ab> h[ab] E1[a,b]", idx_map, perm_map);
        hw = WickExpr::parse("0.5 SUM <abcd> w[abcd] E2[ab,cd]", idx_map,
                             perm_map);
        hd = hi + heff + hw;
        sub_spaces = {{"ijrs+", "E1[r,i] E1[s,j] \n + E1[s,i] E1[r,j]"},
                      {"ijrs-", "E1[r,i] E1[s,j] \n - E1[s,i] E1[r,j]"},
                      {"rsiap+", "E1[r,i] E1[s,a] \n + E1[s,i] E1[r,a]"},
                      {"rsiap-", "E1[r,i] E1[s,a] \n - E1[s,i] E1[r,a]"},
                      {"ijrap+", "E1[r,j] E1[a,i] \n + E1[r,i] E1[a,j]"},
                      {"ijrap-", "E1[r,j] E1[a,i] \n - E1[r,i] E1[a,j]"},
                      {"rsabpq+", "E1[r,b] E1[s,a] \n + E1[s,b] E1[r,a]"},
                      {"rsabpq-", "E1[r,b] E1[s,a] \n - E1[s,b] E1[r,a]"},
                      {"ijabpq+", "E1[b,i] E1[a,j] \n + E1[b,j] E1[a,i]"},
                      {"ijabpq-", "E1[b,i] E1[a,j] \n - E1[b,j] E1[a,i]"},
                      {"irabpq1", "E1[r,i] E1[a,b]"},
                      {"irabpq2", "E1[a,i] E1[r,b]"},
                      {"rabcpqg", "E1[r,b] E1[a,c]"},
                      {"iabcpqg", "E1[b,i] E1[a,c]"}};
    }
    WickExpr build_norm(const string &bra, const string &ket,
                        bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xbra.conjugate() & xket) : (xbra.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .simplify();
    }
    WickExpr build_rhs(const string &bra, const string &ket,
                       bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xbra.conjugate() & xket) : (xbra.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .remove_inactive()
            .simplify();
    }
    string to_einsum_orb_energies(const WickTensor &tensor) const {
        stringstream ss;
        ss << tensor.name << " = np.zeros((";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (wi.types == WickIndexTypes::Inactive)
                ss << "ncore, ";
            else if (wi.types == WickIndexTypes::Active)
                ss << "ncas, ";
            else if (wi.types == WickIndexTypes::External)
                ss << "nvirt, ";
        }
        ss << "))";
        return ss.str();
    }
    string to_einsum_sum_restriction(const WickTensor &tensor,
                                     bool restrict_cas = true,
                                     bool no_eq = false) const {
        stringstream ss, sr;
        ss << "grid = np.indices((";
        bool first_and = false;
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (!restrict_cas && wi.types == WickIndexTypes::Active)
                continue;
            ss << (wi.types == WickIndexTypes::Inactive
                       ? "ncore"
                       : (wi.types == WickIndexTypes::External ? "nvirt"
                                                               : "ncas"));
            if (i != (int)tensor.indices.size() - 1 || i == 0)
                ss << ", ";
            if (i != 0 &&
                tensor.indices[i].types == tensor.indices[i - 1].types) {
                if (wi.types == WickIndexTypes::Active) {
                    if (tensor.indices[i].name[0] !=
                        tensor.indices[i - 1].name[0] + 1)
                        continue;
                    if (i + 1 < (int)tensor.indices.size() &&
                        tensor.indices[i + 1].name[0] ==
                            tensor.indices[i].name[0] + 1)
                        continue;
                    if (i - 2 >= 0 && tensor.indices[i - 1].name[0] ==
                                          tensor.indices[i - 2].name[0] + 1)
                        continue;
                }
                sr << "idx " << (first_and ? "&" : "") << "= grid[" << i - 1
                   << "] <" << (no_eq ? "" : "=") << " grid[" << i << "]"
                   << endl;
                first_and = true;
            }
        }
        ss << "))" << endl;
        return ss.str() + sr.str();
    }
    string to_einsum() const {
        stringstream ss;
        WickTensor hexp, deno, rheq;
        for (int i = 0; i < (int)sub_spaces.size(); i++) {
            string key = sub_spaces[i].first, ket_expr = sub_spaces[i].second;
            string mkey = key, skey = key;
            // this h is actually heff
            string hfull = "SUM <mn> h[mn] E1[m,n] \n"
                           "-2.0 SUM <mnj> w[mjnj] E1[m,n]\n"
                           "+1.0 SUM <mnj> w[mjjn] E1[m,n]\n"
                           "0.5 SUM <mnxy> w[mnxy] E2[mn,xy] \n";
            if (key.back() == '+')
                skey = key.substr(0, key.length() - 1), mkey = skey + "_plus";
            else if (key.back() == '-')
                skey = key.substr(0, key.length() - 1), mkey = skey + "_minus";
            else if (key.back() == '1')
                skey = key.substr(0, key.length() - 1), mkey = skey;
            else if (key.back() == '2')
                continue;
            string rkey = skey;
            map<char, char> ket_bra_map;
            if (skey.length() > 4) {
                for (int j = 4; j < (int)skey.length(); j++)
                    ket_bra_map[skey[j + 4 - skey.length()]] = skey[j];
                rkey = skey.substr(0, skey.length() - ket_bra_map.size() * 2) +
                       skey.substr(skey.length() - ket_bra_map.size());
            }
            string bra_expr = ket_expr;
            for (int j = 0; j < (int)bra_expr.length(); j++)
                if (ket_bra_map.count(bra_expr[j]))
                    bra_expr[j] = ket_bra_map[bra_expr[j]];
            stringstream sr;
            ss << "def compute_" << mkey << "():" << endl;
            hexp = WickTensor::parse("hexp[" + skey + "]", idx_map, perm_map);
            deno = WickTensor::parse("deno[" + skey + "]", idx_map, perm_map);
            rheq = WickTensor::parse("rheq[" + rkey + "]", idx_map, perm_map);
            sr << to_einsum_orb_energies(rheq) << endl;
            sr << build_rhs(bra_expr, hfull, false).to_einsum(rheq) << endl;
            sr << to_einsum_orb_energies(hexp) << endl;
            sr << build_communicator(bra_expr, ket_expr, false).to_einsum(hexp)
               << endl;
            bool restrict_cas = key.back() == '+' || key.back() == '-';
            bool non_ortho = key.back() == '1' || key.back() == '2';
            if (non_ortho) {
                string ket_expr_2 = sub_spaces[i + 1].second;
                string bra_expr_2 = ket_expr_2;
                for (int j = 0; j < (int)bra_expr_2.length(); j++)
                    if (ket_bra_map.count(bra_expr_2[j]))
                        bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                sr << "rheq12 = np.zeros(rheq.shape + (2, ))" << endl;
                sr << "rheq12[..., 0] = rheq" << endl << endl;
                sr << to_einsum_orb_energies(rheq) << endl;
                sr << build_rhs(bra_expr_2, hfull, false).to_einsum(rheq)
                   << endl;
                sr << "rheq12[..., 1] = rheq" << endl << endl;

                sr << "hexp12 = np.zeros(hexp.shape + (2, 2, ))" << endl;
                sr << "hexp12[..., 0, 0] = hexp" << endl << endl;
                sr << to_einsum_orb_energies(hexp) << endl;
                sr << build_communicator(bra_expr, ket_expr_2, false)
                          .to_einsum(hexp)
                   << endl;
                sr << "hexp12[..., 0, 1] = hexp" << endl << endl;
                sr << to_einsum_orb_energies(hexp) << endl;
                sr << build_communicator(bra_expr_2, ket_expr, false)
                          .to_einsum(hexp)
                   << endl;
                sr << "hexp12[..., 1, 0] = hexp" << endl << endl;
                sr << to_einsum_orb_energies(hexp) << endl;
                sr << build_communicator(bra_expr_2, ket_expr_2, false)
                          .to_einsum(hexp)
                   << endl;
                sr << "hexp12[..., 1, 1] = hexp" << endl << endl;
                sr << "dcas = ncas ** " << ket_bra_map.size() << endl;
                sr << "xr = rheq12.reshape((-1, dcas * 2))" << endl;
                sr << "xh = hexp12.reshape((-1, dcas, dcas, 2, 2))" << endl;
                sr << "xh = xh.transpose(0, 1, 3, 2, 4)" << endl;
                sr << "xh = xh.reshape((-1, dcas * 2, dcas * 2))" << endl;
            } else {
                if (ket_bra_map.size() == 2 && restrict_cas)
                    sr << "dcas = ncas * (ncas " << key.back() << " 1) // 2 "
                       << endl;
                else
                    sr << "dcas = ncas ** " << ket_bra_map.size() << endl;
                if (skey.length() - ket_bra_map.size() * 2 >= 2) {
                    sr << to_einsum_sum_restriction(rheq, restrict_cas,
                                                    key.back() == '-');
                    sr << "xr = rheq[idx].reshape((-1, dcas))" << endl;
                    sr << to_einsum_sum_restriction(hexp, restrict_cas,
                                                    key.back() == '-');
                    sr << "xh = hexp[idx].reshape((-1, dcas, dcas))" << endl
                       << endl;
                } else {
                    sr << "xr = rheq.reshape((-1, dcas))" << endl << endl;
                    sr << "xh = hexp.reshape((-1, dcas, dcas))" << endl << endl;
                }
            }
            sr << "return -(np.linalg.solve(xh, xr) * xr).sum()" << endl;
            ss << WickExpr::to_einsum_add_indent(sr.str()) << endl;
        }
        return ss.str();
    }
};

struct WickICMRCI {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    vector<pair<string, string>> sub_spaces;
    WickExpr h1, h2, h;
    WickICMRCI() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("mnxyijkl");
        idx_map[WickIndexTypes::Active] =
            WickIndex::parse_set("mnxyabcdefghpq");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("mnxyrstu");
        perm_map[make_pair("w", 4)] = WickPermutation::qc_phys();
        h1 = WickExpr::parse("SUM <mn> h[mn] E1[m,n]", idx_map, perm_map);
        h2 = WickExpr::parse("0.5 SUM <mnxy> w[mnxy] E2[mn,xy]", idx_map,
                             perm_map);
        h = h1 + h2;
        sub_spaces = {{"reference", ""},
                      {"ijrskltu+", "E1[r,i] E1[s,j] \n + E1[s,i] E1[r,j]"},
                      {"ijrskltu-", "E1[r,i] E1[s,j] \n - E1[s,i] E1[r,j]"},
                      {"rsiatukp+", "E1[r,i] E1[s,a] \n + E1[s,i] E1[r,a]"},
                      {"rsiatukp-", "E1[r,i] E1[s,a] \n - E1[s,i] E1[r,a]"},
                      {"ijrakltp+", "E1[r,j] E1[a,i] \n + E1[r,i] E1[a,j]"},
                      {"ijrakltp-", "E1[r,j] E1[a,i] \n - E1[r,i] E1[a,j]"},
                      {"rsabtupq+", "E1[r,b] E1[s,a] \n + E1[s,b] E1[r,a]"},
                      {"rsabtupq-", "E1[r,b] E1[s,a] \n - E1[s,b] E1[r,a]"},
                      {"ijabklpq+", "E1[b,i] E1[a,j] \n + E1[b,j] E1[a,i]"},
                      {"ijabklpq-", "E1[b,i] E1[a,j] \n - E1[b,j] E1[a,i]"},
                      {"irabktpq1", "E1[r,i] E1[a,b]"},
                      {"irabktpq2", "E1[a,i] E1[r,b]"},
                      {"rabctpqg", "E1[r,b] E1[a,c]"},
                      {"iabckpqg", "E1[b,i] E1[a,c]"}};
    }
    // only block diagonal term will use communicator
    WickExpr build_hamiltonian(const string &bra, const string &ket,
                               bool do_sum = true, bool do_comm = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map);
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map);
        WickExpr expr;
        if (bra == "" && ket == "")
            ;
        else if (bra == "")
            expr = (h * xket);
        else if (ket == "")
            expr = (xbra.conjugate() * h);
        else if (do_comm)
            expr = do_sum ? (xbra.conjugate() & (h ^ xket))
                          : (xbra.conjugate() * (h ^ xket));
        else
            expr = do_sum ? (xbra.conjugate() & (h * xket))
                          : (xbra.conjugate() * (h * xket));
        return expr.expand()
            .remove_external()
            .remove_inactive()
            .add_spin_free_trans_symm()
            .simplify();
    }
    WickExpr build_overlap(const string &bra, const string &ket,
                           bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map);
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map);
        WickExpr expr =
            do_sum ? (xbra.conjugate() & xket) : (xbra.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .remove_inactive()
            .simplify();
    }
    string to_einsum_zeros(const WickTensor &tensor) const {
        stringstream ss;
        ss << tensor.name << " = np.zeros((";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (wi.types == WickIndexTypes::Inactive)
                ss << "ncore, ";
            else if (wi.types == WickIndexTypes::Active)
                ss << "ncas, ";
            else if (wi.types == WickIndexTypes::External)
                ss << "nvirt, ";
        }
        ss << "))";
        return ss.str();
    }
    pair<string, bool>
    to_einsum_sum_restriction(const WickTensor &tensor,
                              string eq_pattern = "+") const {
        stringstream ss, sr;
        ss << "grid = np.indices((";
        bool first_and = false;
        bool has_idx = false;
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            ss << (wi.types == WickIndexTypes::Inactive
                       ? "ncore"
                       : (wi.types == WickIndexTypes::External ? "nvirt"
                                                               : "ncas"));
            if (i != (int)tensor.indices.size() - 1 || i == 0)
                ss << ", ";
            if (i != 0 && wi.types == tensor.indices[i - 1].types) {
                if (wi.name[0] != tensor.indices[i - 1].name[0] + 1)
                    continue;
                if (wi.types == WickIndexTypes::Active) {
                    if (i + 1 < (int)tensor.indices.size() &&
                        tensor.indices[i + 1].types == wi.types &&
                        tensor.indices[i + 1].name[0] == wi.name[0] + 1)
                        continue;
                    if (i - 2 >= 0 && tensor.indices[i - 2].types == wi.types &&
                        tensor.indices[i - 1].name[0] ==
                            tensor.indices[i - 2].name[0] + 1)
                        continue;
                    if (eq_pattern.length() == 2) {
                        if (eq_pattern[0] != '+' && eq_pattern[0] != '-' &&
                            i < tensor.indices.size() / 2)
                            continue;
                        else if (eq_pattern[1] != '+' && eq_pattern[1] != '-' &&
                                 i >= tensor.indices.size() / 2)
                            continue;
                    } else {
                        if (eq_pattern[0] != '+' && eq_pattern[0] != '-')
                            continue;
                    }
                }
                has_idx = true;
                sr << "idx " << (first_and ? "&" : "") << "= grid[" << i - 1
                   << "] <";
                if (eq_pattern.length() == 1)
                    sr << (eq_pattern[0] == '+' ? "=" : "");
                else if (i < tensor.indices.size() / 2)
                    sr << (eq_pattern[0] == '+' ? "=" : "");
                else
                    sr << (eq_pattern[1] == '+' ? "=" : "");
                sr << " grid[" << i << "]" << endl;
                first_and = true;
            }
        }
        ss << "))" << endl;
        return make_pair(ss.str() + sr.str(), has_idx);
    }
    string to_einsum() const {
        stringstream ss, sk;
        WickTensor hmat, deno, norm;
        ss << "xnorms = {}" << endl << endl;
        ss << "xhmats = {}" << endl << endl;
        sk << "keys = [" << endl;
        vector<string> norm_keys(sub_spaces.size());
        for (int i = 0; i < (int)sub_spaces.size(); i++) {
            string key = sub_spaces[i].first, ket_expr = sub_spaces[i].second;
            string mkey = key, skey = key;
            if (key.back() == '+')
                skey = key.substr(0, key.length() - 1), mkey = key;
            else if (key.back() == '-')
                skey = key.substr(0, key.length() - 1), mkey = key;
            else if (key.back() == '1')
                skey = key.substr(0, key.length() - 1), mkey = skey;
            else if (key.back() == '2')
                continue;
            norm_keys[i] = mkey;
            ss << "# compute : overlap " << mkey << endl << endl;
            sk << "    '" << mkey << "'," << endl;
            stringstream sr;
            if (mkey == "reference") {
                sr << "xn = np.ones((1, 1, 1))" << endl;
                sr << "xnorms['" << mkey << "'] = xn" << endl;
                ss << WickExpr::to_einsum_add_indent(sr.str(), 0) << endl;
                continue;
            }
            string nkey = skey;
            map<char, char> ket_bra_map;
            int pidx = skey.length();
            for (int j = 4; j < (int)skey.length(); j++) {
                if (skey[j] == 'p')
                    pidx = j;
                // norm is only non-zero between diff act indices
                if (j >= pidx)
                    ket_bra_map[skey[j + 4 - skey.length()]] = skey[j];
            }
            nkey = skey.substr(0, 4) + skey.substr(pidx);
            int nact = skey.length() - pidx;
            string bra_expr = ket_expr;
            for (int j = 0; j < (int)bra_expr.length(); j++)
                if (ket_bra_map.count(bra_expr[j]))
                    bra_expr[j] = ket_bra_map[bra_expr[j]];
            norm = WickTensor::parse("norm[" + nkey + "]", idx_map, perm_map);
            sr << to_einsum_zeros(norm) << endl;
            sr << build_overlap(bra_expr, ket_expr, false).to_einsum(norm)
               << endl;
            bool restrict_cas = key.back() == '+' || key.back() == '-';
            bool non_ortho = key.back() == '1' || key.back() == '2';
            if (non_ortho) {
                string ket_expr_2 = sub_spaces[i + 1].second;
                string bra_expr_2 = ket_expr_2;
                for (int j = 0; j < (int)bra_expr_2.length(); j++)
                    if (ket_bra_map.count(bra_expr_2[j]))
                        bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                sr << "norm12 = np.zeros(norm.shape + (2, 2))" << endl;
                sr << "norm12[..., 0, 0] = norm" << endl << endl;
                sr << to_einsum_zeros(norm) << endl;
                sr << build_overlap(bra_expr, ket_expr_2, false).to_einsum(norm)
                   << endl;
                sr << "norm12[..., 0, 1] = norm" << endl << endl;
                sr << to_einsum_zeros(norm) << endl;
                sr << build_overlap(bra_expr_2, ket_expr, false).to_einsum(norm)
                   << endl;
                sr << "norm12[..., 1, 0] = norm" << endl << endl;
                sr << to_einsum_zeros(norm) << endl;
                sr << build_overlap(bra_expr_2, ket_expr_2, false)
                          .to_einsum(norm)
                   << endl;
                sr << "norm12[..., 1, 1] = norm" << endl << endl;

                sr << "dcas = ncas ** " << nact << endl;
                sr << "xn = norm12.reshape((-1, dcas, dcas, 2, 2))" << endl;
                sr << "xn = xn.transpose(0, 1, 3, 2, 4)" << endl;
                sr << "xn = xn.reshape((-1, dcas * 2, dcas * 2))" << endl;
            } else {
                if (nact == 2 && restrict_cas)
                    sr << "dcas = ncas * (ncas " << key.back() << " 1) // 2 "
                       << endl;
                else
                    sr << "dcas = ncas ** " << nact << endl;
                auto si =
                    to_einsum_sum_restriction(norm, string(1, key.back()));
                if (si.second) {
                    sr << si.first;
                    sr << "xn = norm[idx].reshape((-1, dcas, dcas))" << endl;
                } else
                    sr << "xn = norm.reshape((-1, dcas, dcas))" << endl << endl;
            }
            sr << "xnorms['" << mkey << "'] = xn" << endl;
            sr << "print(np.linalg.norm(xn))" << endl;
            sr << "assert np.linalg.norm(xn - xn.transpose(0, 2, 1)) < 1E-10"
               << endl;
            ss << WickExpr::to_einsum_add_indent(sr.str(), 0) << endl;
        }
        sk << "]" << endl << endl;
        for (int k = 0; k < (int)sub_spaces.size(); k++) {
            string xkkey = sub_spaces[k].first;
            string ket_expr = sub_spaces[k].second;
            string kmkey = "";
            if (xkkey.back() == '+')
                kmkey = "+";
            else if (xkkey.back() == '-')
                kmkey = "-";
            else if (xkkey.back() == '2')
                continue;
            string kkey = xkkey == "reference" ? xkkey : xkkey.substr(0, 4);
            string ikkey = xkkey == "reference" ? "" : kkey;
            bool kref = xkkey == "reference";
            bool krestrict_cas = xkkey.back() == '+' || xkkey.back() == '-';
            bool knon_ortho = xkkey.back() == '1' || xkkey.back() == '2';
            for (int b = 0; b < (int)sub_spaces.size(); b++) {
                string xbkey = sub_spaces[b].first;
                string bra_expr = sub_spaces[b].second;
                string bmkey = "";
                if (xbkey.back() == '+')
                    bmkey = "+";
                else if (xbkey.back() == '-')
                    bmkey = "-";
                else if (xbkey.back() == '2')
                    continue;
                string bkey = xbkey == "reference" ? xbkey : xbkey.substr(4, 4);
                string ibkey = xbkey == "reference" ? "" : bkey;
                string mkey = bkey + bmkey + " | H - E0 | " + kkey + kmkey;
                map<char, char> ket_bra_map;
                for (int j = 4; j < 8; j++)
                    ket_bra_map[xbkey[j - 4]] = xbkey[j];
                for (int j = 0; j < (int)bra_expr.length(); j++)
                    if (ket_bra_map.count(bra_expr[j]))
                        bra_expr[j] = ket_bra_map[bra_expr[j]];
                stringstream sr;
                cerr << mkey << endl;
                ss << "# compute : hmat = " << mkey << endl << endl;
                ss << "print('compute : hmat = < " << mkey << " >')" << endl
                   << endl;
                hmat = WickTensor::parse("hmat[" + ibkey + ikkey + "]", idx_map,
                                         perm_map);
                sr << to_einsum_zeros(hmat) << endl;
                bool bref = xbkey == "reference";
                bool brestrict_cas = xbkey.back() == '+' || xbkey.back() == '-';
                bool bnon_ortho = xbkey.back() == '1' || xbkey.back() == '2';
                sr << build_hamiltonian(bra_expr, ket_expr, false, b == k)
                          .to_einsum(hmat)
                   << endl;
                sr << "bdsub, bdcas = xnorms['" << norm_keys[b]
                   << "'].shape[:2]" << endl;
                sr << "kdsub, kdcas = xnorms['" << norm_keys[k]
                   << "'].shape[:2]" << endl;
                auto si = to_einsum_sum_restriction(
                    hmat, (bref ? "" : string(1, xbkey.back())) +
                              (kref ? "" : string(1, xkkey.back())));
                if (bnon_ortho && knon_ortho) {
                    assert(k == b);
                    string ket_expr_2 = sub_spaces[k + 1].second;
                    string bra_expr_2 = sub_spaces[b + 1].second;
                    for (int j = 0; j < (int)bra_expr_2.length(); j++)
                        if (ket_bra_map.count(bra_expr_2[j]))
                            bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                    sr << "hmat12 = np.zeros(hmat.shape + (2, 2))" << endl;
                    sr << "hmat12[..., 0, 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr, ket_expr_2, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 0, 1] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr_2, ket_expr, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1, 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr_2, ket_expr_2, false,
                                            b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1, 1] = hmat" << endl << endl;
                    sr << "hmat = hmat12.reshape((bdsub, bdcas // 2, kdsub, "
                          "kdcas // 2, 2, 2))"
                       << endl;
                    sr << "hmat = hmat.transpose((0, 1, 4, 2, 3, 5))" << endl;
                    sr << "xh = hmat.reshape((bdsub, bdcas, kdsub, kdcas))"
                       << endl;
                } else if (bnon_ortho) {
                    string bra_expr_2 = sub_spaces[b + 1].second;
                    for (int j = 0; j < (int)bra_expr_2.length(); j++)
                        if (ket_bra_map.count(bra_expr_2[j]))
                            bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                    sr << "hmat12 = np.zeros(hmat.shape + (2, ))" << endl;
                    sr << "hmat12[..., 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr_2, ket_expr, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1] = hmat" << endl << endl;
                    if (si.second) {
                        sr << si.first;
                        sr << "hmat = hmat12[idx].reshape((bdsub, bdcas // 2, "
                              "kdsub, kdcas, 2))"
                           << endl;
                    } else
                        sr << "hmat = hmat12.reshape((bdsub, bdcas // 2, "
                              "kdsub, kdcas, 2))"
                           << endl;
                    sr << "hmat = hmat.transpose((0, 1, 4, 2, 3))" << endl;
                    sr << "xh = hmat.reshape((bdsub, bdcas, kdsub, kdcas))"
                       << endl;
                } else if (knon_ortho) {
                    string ket_expr_2 = sub_spaces[k + 1].second;

                    sr << "hmat12 = np.zeros(hmat.shape + (2, ))" << endl;
                    sr << "hmat12[..., 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr, ket_expr_2, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1] = hmat" << endl << endl;
                    if (si.second) {
                        sr << si.first;
                        sr << "xh = hmat12[idx].reshape((bdsub, bdcas, "
                              "kdsub, kdcas))"
                           << endl;
                    } else
                        sr << "xh = hmat12.reshape((bdsub, bdcas, "
                              "kdsub, kdcas))"
                           << endl;
                } else {
                    if (si.second) {
                        sr << si.first;
                        sr << "xh = hmat[idx].reshape((bdsub, bdcas, kdsub, "
                              "kdcas))"
                           << endl;
                    } else
                        sr << "xh = hmat.reshape((bdsub, bdcas, kdsub, kdcas))"
                           << endl;
                }
                sr << "xhmats[('" << norm_keys[b] << "', '" << norm_keys[k]
                   << "')] = xh" << endl;
                ss << WickExpr::to_einsum_add_indent(sr.str(), 0) << endl;
            }
        }
        return sk.str() + ss.str();
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
