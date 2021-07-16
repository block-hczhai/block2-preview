
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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

using namespace std;

namespace block2 {

#ifdef _LARGE_BOND
typedef uint32_t ubond_t;
typedef int64_t total_bond_t;
#define _SI_MEM_SIZE(n) ((n) << 1)
#define _DBL_MEM_SIZE(n) ((n) << 1)
#else
#ifdef _SMALL_BOND
typedef uint8_t ubond_t;
typedef int32_t total_bond_t;
#define _SI_MEM_SIZE(n) ((n) + (((n) + 3) >> 2))
#define _DBL_MEM_SIZE(n) ((n) - ((n) >> 1))
#else
typedef uint16_t ubond_t;
typedef int32_t total_bond_t;
#define _SI_MEM_SIZE(n) (((n) << 1) - ((n) >> 1))
#define _DBL_MEM_SIZE(n) (n)
#endif
#endif

template <typename, typename = void> struct StateInfo;

// A collection of quantum symmetry labels and their quantity
template <typename S>
struct StateInfo<S, typename enable_if<integral_constant<
                        bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    shared_ptr<vector<uint32_t>> vdata;
    // Array for symmetry labels
    S *quanta;
    // Array for number of states
    ubond_t *n_states;
    int n;
    total_bond_t n_states_total;
    StateInfo()
        : quanta(nullptr), n_states(nullptr), n_states_total(0), n(0),
          vdata(nullptr) {}
    StateInfo(S q) : vdata(nullptr) {
        allocate(1);
        quanta[0] = q, n_states[0] = 1, n_states_total = 1;
    }
    void load_data(istream &ifs) {
        ifs.read((char *)&n_states_total, sizeof(n_states_total));
        ifs.read((char *)&n, sizeof(n));
        vdata = make_shared<vector<uint32_t>>(_SI_MEM_SIZE(n));
        uint32_t *ptr = vdata->data();
        ifs.read((char *)ptr, sizeof(uint32_t) * _SI_MEM_SIZE(n));
        quanta = (S *)ptr;
        n_states = (ubond_t *)(ptr + n);
    }
    void load_data(const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("StateInfo::load_data on '" + filename +
                                "' failed.");
        load_data(ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("StateInfo::load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    void save_data(ostream &ofs) const {
        ofs.write((char *)&n_states_total, sizeof(n_states_total));
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)quanta, sizeof(uint32_t) * _SI_MEM_SIZE(n));
    }
    void save_data(const string &filename) const {
        if (Parsing::link_exists(filename))
            Parsing::remove_file(filename);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("StateInfo::save_data on '" + filename +
                                "' failed.");
        save_data(ofs);
        if (!ofs.good())
            throw runtime_error("StateInfo::save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    // need length * 2
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0) {
            vdata = make_shared<vector<uint32_t>>(_SI_MEM_SIZE(length));
            ptr = vdata->data();
        }
        n = length;
        quanta = (S *)ptr;
        n_states = (ubond_t *)(ptr + length);
    }
    void reallocate(int length) {
        if (length < n) {
            memmove(quanta + length, n_states, length * sizeof(ubond_t));
            vdata->resize(_SI_MEM_SIZE(length));
            quanta = (S *)vdata->data();
            n_states = (ubond_t *)(quanta + length);
        } else if (length > n) {
            vdata->resize(_SI_MEM_SIZE(length));
            quanta = (S *)vdata->data();
            n_states = (ubond_t *)(quanta + n);
            memmove(quanta + length, n_states, length * sizeof(ubond_t));
            n_states = (ubond_t *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        vdata = nullptr;
        quanta = nullptr;
        n_states = nullptr;
    }
    StateInfo deep_copy() const {
        StateInfo other;
        other.allocate(n);
        copy_data_to(other);
        other.n_states_total = n_states_total;
        return other;
    }
    void copy_data_to(StateInfo &other) const {
        assert(other.n == n);
        memcpy(other.quanta, quanta, _SI_MEM_SIZE(n) * sizeof(uint32_t));
    }
    void sort_states() {
        vector<int> idx(n);
        vector<S> q(quanta, quanta + n);
        vector<ubond_t> nq(n_states, n_states + n);
        for (int i = 0; i < n; i++)
            idx[i] = i;
        sort(idx.begin(), idx.end(),
             [&q](int i, int j) { return q[i] < q[j]; });
        for (int i = 0; i < n; i++)
            quanta[i] = q[idx[i]], n_states[i] = nq[idx[i]];
        n_states_total = 0;
        for (int i = 0; i < n; i++)
            n_states_total += n_states[i];
    }
    // Remove quanta larger than target and quanta with zero n_states
    void collect(S target = S(S::invalid)) {
        int k = -1;
        for (int i = 0; i < n; i++)
            if (n_states[i] == 0)
                continue;
            else if (quanta[i].n() > target.n())
                continue;
            else if (k != -1 && quanta[i] == quanta[k])
                n_states[k] =
                    (ubond_t)min((uint32_t)n_states[k] + n_states[i],
                                 (uint32_t)numeric_limits<ubond_t>::max());
            else {
                k++;
                quanta[k] = quanta[i];
                n_states[k] = n_states[i];
            }
        reallocate(k + 1);
        n_states_total = 0;
        for (int i = 0; i < n; i++)
            n_states_total += n_states[i];
    }
    int find_state(S q) const {
        auto p = lower_bound(quanta, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return (int)(p - quanta);
    }
    void reduce_n_states(int m) {
        bool can_reduce = true;
        while (can_reduce && n_states_total > m) {
            can_reduce = false;
            for (int k = 0; k < n; k++)
                if (n_states[k] > 1) {
                    can_reduce = true;
                    n_states_total -= n_states[k];
                    n_states[k] >>= 1;
                    n_states_total += n_states[k];
                }
        }
    }
    static StateInfo complementary(const StateInfo &a, S target) {
        StateInfo c;
        c.allocate(a.n);
        memcpy(c.quanta, a.quanta, c.n * sizeof(S));
        memcpy(c.n_states, a.n_states, c.n * sizeof(ubond_t));
        for (int i = 0; i < a.n; i++)
            c.quanta[i] = target - a.quanta[i];
        c.sort_states();
        c.collect();
        return c;
    }
    // Tensor product of StateInfo a and b
    // If resulting state does not appear in cref, it will be removed
    static StateInfo tensor_product(const StateInfo &a, const StateInfo &b,
                                    const StateInfo &cref) {
        StateInfo c;
        c.allocate(cref.n);
        memcpy(c.quanta, cref.quanta, c.n * sizeof(S));
        memset(c.n_states, 0, c.n * sizeof(ubond_t));
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    int ic = c.find_state(qc[k]);
                    if (ic != -1) {
                        uint32_t nprod =
                            (uint32_t)a.n_states[i] * (uint32_t)b.n_states[j] +
                            (uint32_t)c.n_states[ic];
                        c.n_states[ic] = (ubond_t)min(
                            nprod, (uint32_t)numeric_limits<ubond_t>::max());
                    }
                }
            }
        c.collect();
        return c;
    }
    // Tensor product of StateInfo a and b
    // Resulting state that larger than target will be removed
    static StateInfo tensor_product(const StateInfo &a, const StateInfo &b,
                                    S target) {
        int nc = 0;
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++)
                nc += (a.quanta[i] + b.quanta[j]).count();
        StateInfo c;
        c.allocate(nc);
        for (int i = 0, ic = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    c.quanta[ic + k] = qc[k];
                    uint32_t nprod =
                        (uint32_t)a.n_states[i] * (uint32_t)b.n_states[j];
                    c.n_states[ic + k] = (ubond_t)min(
                        nprod, (uint32_t)numeric_limits<ubond_t>::max());
                }
                ic += qc.count();
            }
        c.sort_states();
        c.collect(target);
        return c;
    }
    // Connection info for tensor product c of StateInfo a and b
    // For determining stride in tensor product of two SparseMatrix
    static StateInfo get_connection_info(const StateInfo &a, const StateInfo &b,
                                         const StateInfo &c) {
        map<S, vector<uint32_t>> mp;
        int nc = 0, iab = 0;
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                nc += qc.count();
                for (int k = 0; k < qc.count(); k++)
                    mp[qc[k]].push_back((i << 16) + j);
            }
        StateInfo ci;
        ci.allocate(nc);
        for (int ic = 0; ic < c.n; ic++) {
            vector<uint32_t> &v = mp.at(c.quanta[ic]);
            ci.n_states[ic] = iab;
            memcpy(ci.quanta + iab, v.data(), v.size() * sizeof(uint32_t));
            iab += (int)v.size();
        }
        ci.reallocate(iab);
        ci.n_states_total = c.n;
        return ci;
    }
    // Remove unmatched quantum numbers in left or right blocks
    // Using the target quantum number as the constraint
    // b is unchanged
    static void filter(StateInfo &a, const StateInfo &b, S target) {
        a.n_states_total = 0;
        for (int i = 0; i < a.n; i++) {
            S qb = target - a.quanta[i];
            uint32_t x = 0;
            for (int k = 0; k < qb.count(); k++) {
                int idx = b.find_state(qb[k]);
                x += idx == -1 ? 0 : (uint32_t)b.n_states[idx];
            }
            a.n_states[i] = (ubond_t)min(x, (uint32_t)a.n_states[i]);
            a.n_states_total += a.n_states[i];
        }
    }
    static void multi_target_filter(StateInfo &a, const StateInfo &b,
                                    const vector<S> &targets) {
        a.n_states_total = 0;
        for (int i = 0; i < a.n; i++) {
            set<int> idxs;
            for (S target : targets) {
                S qb = target - a.quanta[i];
                for (int k = 0, idx; k < qb.count(); k++)
                    if ((idx = b.find_state(qb[k])) != -1)
                        idxs.insert(idx);
            }
            uint32_t x = 0;
            for (auto idx : idxs)
                x += (uint32_t)b.n_states[idx];
            a.n_states[i] = (ubond_t)min(x, (uint32_t)a.n_states[i]);
            a.n_states_total += a.n_states[i];
        }
    }
    friend ostream &operator<<(ostream &os, const StateInfo<S> &c) {
        for (int i = 0; i < c.n; i++)
            os << c.quanta[i].to_str() << " : " << (uint32_t)c.n_states[i]
               << endl;
        return os;
    }
};

template <typename, typename = void> struct StateProbability;

// A collection of quantum symmetry labels and their probability
template <typename S>
struct StateProbability<
    S, typename enable_if<integral_constant<
           bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    shared_ptr<vector<uint32_t>> vdata;
    S *quanta;
    double *probs;
    int n;
    StateProbability() : quanta(0), probs(0), n(0) {}
    StateProbability(S q) {
        allocate(1);
        quanta[0] = q, probs[0] = 1;
    }
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0) {
            vdata = make_shared<vector<uint32_t>>((length << 1) + length + 1);
            ptr = vdata->data();
        }
        n = length;
        quanta = (S *)ptr;
        // double must be 8-aligned
        probs = (double *)(ptr + length + !!((size_t)(ptr + length) & 7));
    }
    void reallocate(int length) {
        if (length < n) {
            memmove(quanta + length + !!((size_t)(quanta + length) & 7), probs,
                    length * sizeof(double));
            vdata->resize((length << 1) + length + 1);
            quanta = (S *)vdata->data();
            probs =
                (double *)(quanta + length + !!((size_t)(quanta + length) & 7));
        } else if (length > n) {
            vdata->resize((length << 1) + length + 1);
            quanta = (S *)vdata->data();
            memmove(quanta + length + !!((size_t)(quanta + length) & 7), probs,
                    length * sizeof(double));
            probs = (double *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        assert(n != 0);
        vdata = nullptr;
        quanta = nullptr;
        probs = nullptr;
    }
    void collect(S target = S(S::invalid)) {
        int k = -1;
        for (int i = 0; i < n; i++)
            if (probs[i] == 0.0)
                continue;
            else if (quanta[i].n() > target.n())
                continue;
            else if (k != -1 && quanta[i] == quanta[k])
                probs[k] = probs[k] + probs[i];
            else {
                k++;
                quanta[k] = quanta[i];
                probs[k] = probs[i];
            }
        reallocate(k + 1);
    }
    int find_state(S q) const {
        auto p = lower_bound(quanta, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return (int)(p - quanta);
    }
    static StateProbability<S> tensor_product_no_collect(
        const StateProbability<S> &a, const StateProbability<S> &b,
        const StateInfo<S> &cref, const vector<vector<double>> &pref) {
        StateProbability<S> c;
        c.allocate(cref.n);
        memcpy(c.quanta, cref.quanta, c.n * sizeof(S));
        memset(c.probs, 0, c.n * sizeof(double));
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                if (pref.size() == 0)
                    for (int k = 0; k < qc.count(); k++) {
                        int ic = c.find_state(qc[k]);
                        if (ic != -1)
                            c.probs[ic] += a.probs[i] * b.probs[j];
                    }
                else
                    for (int k = 0; k < qc.count(); k++) {
                        int ic = c.find_state(qc[k]);
                        if (ic != -1)
                            c.probs[ic] +=
                                a.probs[i] * b.probs[j] * pref[qc.count()][k];
                    }
            }
        return c;
    }
    friend ostream &operator<<(ostream &os, const StateProbability<S> &c) {
        for (int i = 0; i < c.n; i++)
            os << c.quanta[i].to_str() << " : " << c.probs[i] << endl;
        return os;
    }
};

template <typename S1, typename S2, typename = void, typename = void>
struct TransStateInfo {
    static shared_ptr<StateInfo<S2>>
    forward(const shared_ptr<StateInfo<S1>> &si) {
        return TransStateInfo<S2, S1>::backward(si);
    }
    static shared_ptr<StateInfo<S1>>
    backward(const shared_ptr<StateInfo<S2>> &si) {
        return TransStateInfo<S2, S1>::forward(si);
    }
};

// Translation between SU2 and SZ StateInfo
template <typename S1, typename S2>
struct TransStateInfo<S1, S2, typename S1::is_sz_t, typename S2::is_su2_t> {
    // from sz to su2 is not exact
    static shared_ptr<StateInfo<S2>>
    forward(const shared_ptr<StateInfo<S1>> &si) {
        vector<pair<S2, ubond_t>> vso;
        map<pair<int, int>, vector<S1>> mp;
        for (int i = 0; i < si->n; i++) {
            S1 q = si->quanta[i];
            mp[make_pair(q.n(), q.pg())].push_back(q);
        }
        for (auto &m : mp) {
            sort(m.second.begin(), m.second.end(),
                 [](S1 a, S1 b) { return abs(a.twos()) > abs(b.twos()); });
            vector<pair<ubond_t, int>> nst(abs(m.second[0].twos()) + 1);
            for (auto &mm : m.second) {
                int iq = si->find_state(mm);
                nst[abs(mm.twos())].first += si->n_states[iq];
                nst[abs(mm.twos())].second++;
            }
            for (int lims = abs(m.second[0].twos()), twos = lims; twos >= 0;
                 twos -= 2) {
                S2 q(m.first.first, twos, m.first.second);
                ubond_t a = nst[twos].second == 0
                                ? 0
                                : nst[twos].first / nst[twos].second;
                if (twos != lims) {
                    ubond_t b =
                        nst[twos + 2].second == 0
                            ? 0
                            : nst[twos + 2].first / nst[twos + 2].second;
                    if (a > b)
                        vso.push_back(make_pair(q, a - b));
                } else if (a > 0)
                    vso.push_back(make_pair(q, a));
            }
        }
        shared_ptr<StateInfo<S2>> so = make_shared<StateInfo<S2>>();
        so->allocate((int)vso.size());
        for (int i = 0; i < (int)vso.size(); i++)
            so->quanta[i] = vso[i].first, so->n_states[i] = vso[i].second;
        so->sort_states();
        return so;
    }
    static shared_ptr<StateInfo<S1>>
    backward(const shared_ptr<StateInfo<S2>> &si) {
        map<S1, ubond_t> mp;
        for (int i = 0; i < si->n; i++) {
            S2 q = si->quanta[i];
            for (int j = -q.twos(); j <= q.twos(); j += 2)
                mp[S1(q.n(), j, q.pg())] += si->n_states[i];
        }
        shared_ptr<StateInfo<S1>> so = make_shared<StateInfo<S1>>();
        so->allocate((int)mp.size());
        int i = 0;
        for (auto m : mp) {
            so->quanta[i] = m.first, so->n_states[i] = m.second;
            i++;
        }
        so->sort_states();
        return so;
    }
    static shared_ptr<StateInfo<S2>>
    backward_connection(const shared_ptr<StateInfo<S2>> &si,
                        const shared_ptr<StateInfo<S1>> &bsi) {
        shared_ptr<StateInfo<S2>> so = make_shared<StateInfo<S2>>();
        map<S1, vector<S2>> mp;
        int nb = 0, ii = 0;
        for (int i = 0; i < si->n; i++) {
            S2 q = si->quanta[i];
            nb += q.twos() + 1;
            for (int j = -q.twos(); j <= q.twos(); j += 2)
                mp[S1(q.n(), j, q.pg())].push_back(q);
        }
        so->allocate(nb);
        for (int ib = 0; ib < bsi->n; ib++) {
            vector<S2> &v = mp.at(bsi->quanta[ib]);
            so->n_states[ib] = ii;
            memcpy(so->quanta + ii, v.data(), v.size() * sizeof(S2));
            ii += (int)v.size();
        }
        so->reallocate(ii);
        so->n_states_total = bsi->n;
        return so;
    }
};

} // namespace block2
