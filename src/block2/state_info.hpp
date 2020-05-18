
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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

#include "allocator.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <type_traits>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct StateInfo;

// A collection of quantum symmetry labels and their quantity
template <typename S>
struct StateInfo<S, typename enable_if<integral_constant<
                        bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    // Array for symmetry labels
    S *quanta;
    // Array for number of states
    uint16_t *n_states;
    int n_states_total, n;
    StateInfo() : quanta(0), n_states(0), n_states_total(0), n(0) {}
    StateInfo(S q) {
        allocate(1);
        quanta[0] = q, n_states[0] = 1, n_states_total = 1;
    }
    void load_data(const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        ifs.read((char *)&n_states_total, sizeof(n_states_total));
        ifs.read((char *)&n, sizeof(n));
        uint32_t *ptr = ialloc->allocate((n << 1) - (n >> 1));
        ifs.read((char *)ptr, sizeof(uint32_t) * ((n << 1) - (n >> 1)));
        ifs.close();
        quanta = (S *)ptr;
        n_states = (uint16_t *)(ptr + n);
    }
    void save_data(const string &filename) const {
        ofstream ofs(filename.c_str(), ios::binary);
        ofs.write((char *)&n_states_total, sizeof(n_states_total));
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)quanta, sizeof(uint32_t) * ((n << 1) - (n >> 1)));
        ofs.close();
    }
    // need length * 2
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) - (length >> 1));
        n = length;
        quanta = (S *)ptr;
        n_states = (uint16_t *)(ptr + length);
    }
    void reallocate(int length) {
        uint32_t *ptr =
            ialloc->reallocate((uint32_t *)quanta, (n << 1) - (n >> 1),
                               (length << 1) - (length >> 1));
        if (ptr == (uint32_t *)quanta) {
            memmove(ptr + length, n_states, length * sizeof(uint16_t));
            n_states = (uint16_t *)(quanta + length);
        } else {
            memmove(ptr, quanta, length * sizeof(uint32_t));
            memmove(ptr + length, n_states, length * sizeof(uint16_t));
            quanta = (S *)ptr;
            n_states = (uint16_t *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        assert(n != 0);
        ialloc->deallocate((uint32_t *)quanta, (n << 1) - (n >> 1));
        quanta = 0;
        n_states = 0;
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
        memcpy(other.quanta, quanta, ((n << 1) - (n >> 1)) * sizeof(uint32_t));
    }
    void sort_states() {
        int idx[n];
        S q[n];
        uint16_t nq[n];
        memcpy(q, quanta, n * sizeof(S));
        memcpy(nq, n_states, n * sizeof(uint16_t));
        for (int i = 0; i < n; i++)
            idx[i] = i;
        sort(idx, idx + n, [&q](int i, int j) { return q[i] < q[j]; });
        for (int i = 0; i < n; i++)
            quanta[i] = q[idx[i]], n_states[i] = nq[idx[i]];
        n_states_total = 0;
        for (int i = 0; i < n; i++)
            n_states_total += n_states[i];
    }
    // Remove quanta larger than target and quanta with zero n_states
    void collect(S target = 0x7FFFFFFF) {
        int k = -1;
        int nn = upper_bound(quanta, quanta + n, target) - quanta;
        for (int i = 0; i < nn; i++)
            if (n_states[i] == 0)
                continue;
            else if (k != -1 && quanta[i] == quanta[k])
                n_states[k] =
                    (uint16_t)min((uint32_t)n_states[k] + n_states[i], 65535U);
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
            return p - quanta;
    }
    // Tensor product of StateInfo a and b
    // If resulting state does not appear in cref, it will be removed
    static StateInfo tensor_product(const StateInfo &a, const StateInfo &b,
                                    const StateInfo &cref) {
        StateInfo c;
        c.allocate(cref.n);
        memcpy(c.quanta, cref.quanta, c.n * sizeof(S));
        memset(c.n_states, 0, c.n * sizeof(uint16_t));
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    int ic = c.find_state(qc[k]);
                    if (ic != -1) {
                        uint32_t nprod =
                            (uint32_t)a.n_states[i] * (uint32_t)b.n_states[j] +
                            (uint32_t)c.n_states[ic];
                        c.n_states[ic] = (uint16_t)min(nprod, 65535U);
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
                    c.n_states[ic + k] = (uint16_t)min(nprod, 65535U);
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
            memcpy(ci.quanta + iab, &v[0], v.size() * sizeof(uint32_t));
            iab += v.size();
        }
        ci.reallocate(iab);
        ci.n_states_total = c.n;
        return ci;
    }
    // Remove unmatched quantum numbers in left or right blocks
    // Using the target quantum number as the constraint
    static void filter(StateInfo &a, StateInfo &b, S target) {
        a.n_states_total = 0;
        for (int i = 0; i < a.n; i++) {
            S qb = target - a.quanta[i];
            int x = 0;
            for (int k = 0; k < qb.count(); k++) {
                int idx = b.find_state(qb[k]);
                x += idx == -1 ? 0 : b.n_states[idx];
            }
            a.n_states[i] = (uint16_t)min(x, (int)a.n_states[i]);
            a.n_states_total += a.n_states[i];
        }
        b.n_states_total = 0;
        for (int i = 0; i < b.n; i++) {
            S qa = target - b.quanta[i];
            int x = 0;
            for (int k = 0; k < qa.count(); k++) {
                int idx = a.find_state(qa[k]);
                x += idx == -1 ? 0 : a.n_states[idx];
            }
            b.n_states[i] = (uint16_t)min(x, (int)b.n_states[i]);
            b.n_states_total += b.n_states[i];
        }
    }
    friend ostream &operator<<(ostream &os, const StateInfo<S> &c) {
        for (int i = 0; i < c.n; i++)
            os << c.quanta[i].to_str() << " : " << c.n_states[i] << endl;
        return os;
    }
};

template <typename, typename = void> struct StateProbability;

// A collection of quantum symmetry labels and their probability
template <typename S>
struct StateProbability<
    S, typename enable_if<integral_constant<
           bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    S *quanta;
    double *probs;
    int n;
    StateProbability() : quanta(0), probs(0), n(0) {}
    StateProbability(S q) {
        allocate(1);
        quanta[0] = q, probs[0] = 1;
    }
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) + length);
        n = length;
        quanta = (S *)ptr;
        probs = (double *)(ptr + length);
    }
    void reallocate(int length) {
        uint32_t *ptr = ialloc->reallocate((uint32_t *)quanta, (n << 1) + n,
                                           (length << 1) + length);
        if (ptr == (uint32_t *)quanta) {
            memmove(ptr + length, probs, length * sizeof(double));
            probs = (double *)(quanta + length);
        } else {
            memmove(ptr, quanta, length * sizeof(uint32_t));
            memmove(ptr + length, probs, length * sizeof(double));
            quanta = (S *)ptr;
            probs = (double *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        assert(n != 0);
        ialloc->deallocate((uint32_t *)quanta, (n << 1) + n);
        quanta = 0;
        probs = 0;
    }
    void collect(S target = 0x7FFFFFFF) {
        int k = -1;
        int nn = upper_bound(quanta, quanta + n, target) - quanta;
        for (int i = 0; i < nn; i++)
            if (probs[i] == 0.0)
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
            return p - quanta;
    }
    static StateProbability<S>
    tensor_product_no_collect(const StateProbability<S> &a,
                              const StateProbability<S> &b,
                              const StateInfo<S> &cref) {
        StateProbability<S> c;
        c.allocate(cref.n);
        memcpy(c.quanta, cref.quanta, c.n * sizeof(uint32_t));
        memset(c.probs, 0, c.n * sizeof(double));
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    int ic = c.find_state(qc[k]);
                    if (ic != -1)
                        c.probs[ic] += a.probs[i] * b.probs[j];
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

} // namespace block2
