
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

#include "threading.hpp"
#include "utils.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

// Symmetric/general 2D array for storage of one-electron integrals
struct TInt {
    // Number of orbitals
    uint16_t n;
    double *data;
    bool general;
    TInt(uint16_t n, bool general = false)
        : n(n), data(nullptr), general(general) {}
    uint32_t find_index(uint16_t i, uint16_t j) const {
        return general ? (uint32_t)i * n + j
                       : (i < j ? ((uint32_t)j * (j + 1) >> 1) + i
                                : ((uint32_t)i * (i + 1) >> 1) + j);
    }
    size_t size() const {
        return general ? (size_t)n * n : ((size_t)n * (n + 1) >> 1);
    }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j) {
        return *(data + find_index(i, j));
    }
    double operator()(uint16_t i, uint16_t j) const {
        return *(data + find_index(i, j));
    }
    void reorder(const TInt &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint16_t i = 0; i < n; i++)
            for (uint16_t j = 0; j < (general ? n : i + 1); j++)
                (*this)(i, j) = other(ord[i], ord[j]);
    }
    void rotate(const TInt &other, const vector<double> &rot_mat) {
        assert(n == other.n);
        vector<double> tmp((size_t)n * n);
        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
        {
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ij = 0; ij < n * n; ij++) {
                int i = ij / n, j = ij % n;
#else
#pragma omp for schedule(dynamic) collapse(2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += other(i, q) * rot_mat[q * n + j];
                tmp[(size_t)i * n + j] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ij = 0; ij < n * n; ij++) {
                int i = ij / n, j = ij % n;
#else
#pragma omp for schedule(dynamic) collapse(2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
#endif
                if (!general && j > i)
                    continue;
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(size_t)q * n + j] * rot_mat[q * n + i];
                (*this)(i, j) = x;
            }
        }
    }
    friend ostream &operator<<(ostream &os, TInt x) {
        os << fixed << setprecision(16);
        for (uint16_t i = 0; i < x.n; i++)
            for (uint16_t j = 0; j < (x.general ? x.n : i + 1); j++)
                if (x(i, j) != 0.0)
                    os << setw(20) << x(i, j) << setw(4) << i + 1 << setw(4)
                       << j + 1 << setw(4) << 0 << setw(4) << 0 << endl;
        return os;
    }
};

// General 4D array for storage of two-electron integrals
struct V1Int {
    // Number of orbitals
    uint32_t n;
    size_t m;
    double *data;
    V1Int(uint32_t n) : n(n), m((size_t)n * n * n * n), data(nullptr) {}
    size_t size() const { return m; }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + (((size_t)i * n + j) * n + k) * n + l);
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return *(data + (((size_t)i * n + j) * n + k) * n + l);
    }
    void reorder(const V1Int &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < n; j++)
                for (uint32_t k = 0; k < n; k++)
                    for (uint32_t l = 0; l < n; l++)
                        (*this)(i, j, k, l) =
                            other(ord[i], ord[j], ord[k], ord[l]);
    }
    void rotate(const V1Int &other, const vector<double> &rot_mat) {
        assert(n == other.n);
        vector<double> tmp(size());
#ifdef _MSC_VER
        assert((size_t)n * n * n * n <= (size_t)numeric_limits<int>::max());
#endif
        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
        {
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += other(i, j, k, q) * rot_mat[q * n + l];
                tmp[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)i * n + j) * n + q) * n + l] *
                         rot_mat[q * n + k];
                (*this)(i, j, k, l) = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += (*this)(i, q, k, l) * rot_mat[q * n + j];
                tmp[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)q * n + j) * n + k) * n + l] *
                         rot_mat[q * n + i];
                (*this)(i, j, k, l) = x;
            }
        }
    }
    friend ostream &operator<<(ostream &os, V1Int x) {
        os << fixed << setprecision(16);
        for (uint32_t i = 0; i < x.n; i++)
            for (uint32_t j = 0; j < x.n; j++)
                for (uint32_t k = 0; k < x.n; k++)
                    for (uint32_t l = 0; l < x.n; l++)
                        if (x(i, j, k, l) != 0.0)
                            os << setw(20) << x(i, j, k, l) << setw(4) << i + 1
                               << setw(4) << j + 1 << setw(4) << k + 1
                               << setw(4) << l + 1 << endl;
        return os;
    }
};

// 4D array with 4-fold symmetry for storage of two-electron integrals
// [ijkl] = [jikl] = [jilk] = [ijlk]
struct V4Int {
    // n: number of orbitals
    uint32_t n, m;
    double *data;
    V4Int(uint32_t n) : n(n), m(n * (n + 1) >> 1), data(nullptr) {}
    size_t find_index(uint32_t i, uint32_t j) const {
        return i < j ? ((size_t)j * (j + 1) >> 1) + i
                     : ((size_t)i * (i + 1) >> 1) + j;
    }
    size_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        size_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return p * m + q;
    }
    size_t size() const { return (size_t)m * m; }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + find_index(i, j, k, l));
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return *(data + find_index(i, j, k, l));
    }
    void reorder(const V4Int &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j <= i; j++)
                for (uint32_t k = 0; k < n; k++)
                    for (uint32_t l = 0; l <= k; l++)
                        (*this)(i, j, k, l) =
                            other(ord[i], ord[j], ord[k], ord[l]);
    }
    void rotate(const V4Int &other, const vector<double> &rot_mat) {
        assert(n == other.n);
        vector<double> tmp((size_t)n * n * n * n), tmp2((size_t)n * n * n * n);
#ifdef _MSC_VER
        assert((size_t)n * n * n * n <= (size_t)numeric_limits<int>::max());
#endif
        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
        {
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += other(i, j, k, q) * rot_mat[q * n + l];
                tmp[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)i * n + j) * n + q) * n + l] *
                         rot_mat[q * n + k];
                tmp2[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp2[(((size_t)i * n + q) * n + k) * n + l] *
                         rot_mat[q * n + j];
                tmp[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
                            if (j > i || l > k)
                                continue;
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)q * n + j) * n + k) * n + l] *
                         rot_mat[q * n + i];
                (*this)(i, j, k, l) = x;
            }
        }
    }
    friend ostream &operator<<(ostream &os, V4Int x) {
        os << fixed << setprecision(16);
        for (uint32_t i = 0; i < x.n; i++)
            for (uint32_t j = 0; j <= i; j++)
                for (uint32_t k = 0; k < x.n; k++)
                    for (uint32_t l = 0; l <= k; l++)
                        if (x(i, j, k, l) != 0.0)
                            os << setw(20) << x(i, j, k, l) << setw(4) << i + 1
                               << setw(4) << j + 1 << setw(4) << k + 1
                               << setw(4) << l + 1 << endl;
        return os;
    }
};

// 4D array with 8-fold symmetry for storage of two-electron integrals
// [ijkl] = [jikl] = [jilk] = [ijlk] = [klij] = [klji] = [lkji] = [lkij]
struct V8Int {
    // n: number of orbitals
    uint32_t n, m;
    double *data;
    V8Int(uint32_t n) : n(n), m(n * (n + 1) >> 1), data(nullptr) {}
    size_t find_index(uint32_t i, uint32_t j) const {
        return i < j ? ((size_t)j * (j + 1) >> 1) + i
                     : ((size_t)i * (i + 1) >> 1) + j;
    }
    size_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        uint32_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return find_index(p, q);
    }
    size_t size() const { return ((size_t)m * (m + 1) >> 1); }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + find_index(i, j, k, l));
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return *(data + find_index(i, j, k, l));
    }
    void reorder(const V8Int &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint32_t i = 0, ij = 0; i < n; i++)
            for (uint32_t j = 0; j <= i; j++, ij++)
                for (uint32_t k = 0, kl = 0; k <= i; k++)
                    for (uint32_t l = 0; l <= k; l++, kl++)
                        if (ij >= kl)
                            (*this)(i, j, k, l) =
                                other(ord[i], ord[j], ord[k], ord[l]);
    }
    void rotate(const V8Int &other, const vector<double> &rot_mat) {
        assert(n == other.n);
        vector<double> tmp((size_t)n * n * n * n), tmp2((size_t)n * n * n * n);
#ifdef _MSC_VER
        assert((size_t)n * n * n * n <= (size_t)numeric_limits<int>::max());
#endif
        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
        {
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += other(i, j, k, q) * rot_mat[q * n + l];
                tmp[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)i * n + j) * n + q) * n + l] *
                         rot_mat[q * n + k];
                tmp2[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp2[(((size_t)i * n + q) * n + k) * n + l] *
                         rot_mat[q * n + j];
                tmp[(((size_t)i * n + j) * n + k) * n + l] = x;
            }
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
            for (int ijkl = 0; ijkl < n * n * n * n; ijkl++) {
                int i = ijkl / (n * n * n), j = (ijkl / (n * n)) % n,
                    k = (ijkl / n) % n, l = ijkl % n;
#else
#pragma omp for schedule(dynamic) collapse(4)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++) {
                            if (j > i || l > k ||
                                (k * (k + 1) >> 1) + l > (i * (i + 1) >> 1) + j)
                                continue;
#endif
                double x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)q * n + j) * n + k) * n + l] *
                         rot_mat[q * n + i];
                (*this)(i, j, k, l) = x;
            }
        }
    }
    friend ostream &operator<<(ostream &os, V8Int x) {
        os << fixed << setprecision(16);
        for (uint32_t i = 0, ij = 0; i < x.n; i++)
            for (uint32_t j = 0; j <= i; j++, ij++)
                for (uint32_t k = 0, kl = 0; k <= i; k++)
                    for (uint32_t l = 0; l <= k; l++, kl++)
                        if (ij >= kl && x(i, j, k, l) != 0.0)
                            os << setw(20) << x(i, j, k, l) << setw(4) << i + 1
                               << setw(4) << j + 1 << setw(4) << k + 1
                               << setw(4) << l + 1 << endl;
        return os;
    }
};

// One- and two-electron integrals
struct FCIDUMP {
    shared_ptr<vector<double>> vdata;
    map<string, string> params;
    vector<TInt> ts;
    vector<V8Int> vs;
    vector<V4Int> vabs;
    vector<V1Int> vgs;
    double const_e;
    double *data;
    size_t total_memory;
    bool uhf, general;
    FCIDUMP() : const_e(0.0), uhf(false), total_memory(0), vdata(nullptr) {}
    // Initialize integrals: U(1) case
    // Two-electron integrals can be three general rank-4 arrays
    // or 8-fold, 8-fold, 4-fold rank-1 arrays
    virtual ~FCIDUMP() = default;
    virtual void initialize_sz(uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                               uint16_t isym, double e, const double *ta,
                               size_t lta, const double *tb, size_t ltb,
                               const double *va, size_t lva, const double *vb,
                               size_t lvb, const double *vab, size_t lvab) {
        params.clear();
        ts.clear();
        vs.clear();
        vabs.clear();
        vgs.clear();
        this->const_e = e;
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_elec);
        params["ms2"] = Parsing::to_string(twos);
        params["isym"] = Parsing::to_string(isym);
        params["iuhf"] = "1";
        ts.push_back(TInt(n_sites));
        ts.push_back(TInt(n_sites));
        if (lta != ts[0].size())
            ts[0].general = ts[1].general = true;
        assert(lta == ts[0].size() && ltb == ts[1].size());
        vs.push_back(V8Int(n_sites));
        vs.push_back(V8Int(n_sites));
        vabs.push_back(V4Int(n_sites));
        if (vs[0].size() == lva) {
            assert(vs[1].size() == lvb);
            assert(vabs[0].size() == lvab);
            general = false;
            total_memory = lta + ltb + lva + lvb + lvab;
            vdata = make_shared<vector<double>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            ts[1].data = data + lta;
            vs[0].data = data + lta + ltb;
            vs[1].data = data + lta + ltb + lva;
            vabs[0].data = data + lta + ltb + lva + lvb;
            memcpy(vs[0].data, va, sizeof(double) * lva);
            memcpy(vs[1].data, vb, sizeof(double) * lvb);
            memcpy(vabs[0].data, vab, sizeof(double) * lvab);
        } else {
            general = true;
            vs.clear();
            vabs.clear();
            vgs.push_back(V1Int(n_sites));
            vgs.push_back(V1Int(n_sites));
            vgs.push_back(V1Int(n_sites));
            assert(vgs[0].size() == lva);
            assert(vgs[1].size() == lvb);
            assert(vgs[2].size() == lvab);
            total_memory = lta + ltb + lva + lvb + lvab;
            vdata = make_shared<vector<double>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            ts[1].data = data + lta;
            vgs[0].data = data + lta + ltb;
            vgs[1].data = data + lta + ltb + lva;
            vgs[2].data = data + lta + ltb + lva + lvb;
            memcpy(vgs[0].data, va, sizeof(double) * lva);
            memcpy(vgs[1].data, vb, sizeof(double) * lvb);
            memcpy(vgs[2].data, vab, sizeof(double) * lvab);
        }
        memcpy(ts[0].data, ta, sizeof(double) * lta);
        memcpy(ts[1].data, tb, sizeof(double) * ltb);
        uhf = true;
    }
    // Initialize integrals: SU(2) case
    // Two-electron integrals can be general rank-4 array or 8-fold rank-1 array
    virtual void initialize_su2(uint16_t n_sites, uint16_t n_elec,
                                uint16_t twos, uint16_t isym, double e,
                                const double *t, size_t lt, const double *v,
                                size_t lv) {
        params.clear();
        ts.clear();
        vs.clear();
        vabs.clear();
        vgs.clear();
        this->const_e = e;
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_elec);
        params["ms2"] = Parsing::to_string(twos);
        params["isym"] = Parsing::to_string(isym);
        params["iuhf"] = "0";
        ts.push_back(TInt(n_sites));
        if (lt != ts[0].size())
            ts[0].general = true;
        assert(lt == ts[0].size());
        vs.push_back(V8Int(n_sites));
        if (vs[0].size() == lv) {
            general = false;
            total_memory = ts[0].size() + vs[0].size();
            vdata = make_shared<vector<double>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            vs[0].data = data + ts[0].size();
            memcpy(vs[0].data, v, sizeof(double) * lv);
        } else {
            general = true;
            vs.clear();
            vgs.push_back(V1Int(n_sites));
            assert(lv == vgs[0].size());
            total_memory = ts[0].size() + vgs[0].size();
            vdata = make_shared<vector<double>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            vgs[0].data = data + ts[0].size();
            memcpy(vgs[0].data, v, sizeof(double) * lv);
        }
        memcpy(ts[0].data, t, sizeof(double) * lt);
        uhf = false;
    }
    // Initialize with only h1e integral
    virtual void initialize_h1e(uint16_t n_sites, uint16_t n_elec,
                                uint16_t twos, uint16_t isym, double e,
                                const double *t, size_t lt) {
        params.clear();
        ts.clear();
        vs.clear();
        vabs.clear();
        vgs.clear();
        this->const_e = e;
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_elec);
        params["ms2"] = Parsing::to_string(twos);
        params["isym"] = Parsing::to_string(isym);
        params["iuhf"] = "0";
        ts.push_back(TInt(n_sites));
        if (lt != ts[0].size())
            ts[0].general = true;
        assert(lt == ts[0].size());
        vs.push_back(V8Int(n_sites));
        general = false;
        total_memory = ts[0].size() + vs[0].size();
        vdata = make_shared<vector<double>>(total_memory);
        data = vdata->data();
        ts[0].data = data;
        vs[0].data = data + ts[0].size();
        memset(vs[0].data, 0, sizeof(double) * vs[0].size());
        memcpy(ts[0].data, t, sizeof(double) * lt);
        uhf = false;
    }
    // Writing FCIDUMP file to disk
    virtual void write(const string &filename) const {
        ofstream ofs(filename.c_str());
        if (!ofs.good())
            throw runtime_error("FCIDUMP::write on '" + filename + "' failed.");
        ofs << " &FCI NORB=" << setw(4) << (int)n_sites()
            << ",NELEC=" << setw(4) << (int)n_elec() << ",MS2=" << setw(4)
            << (int)twos() << "," << endl;
        assert(params.count("orbsym") != 0);
        ofs << "  ORBSYM=" << params.at("orbsym") << "," << endl;
        ofs << "  ISYM=" << setw(4) << (int)isym() << "," << endl;
        if (uhf)
            ofs << "  IUHF=1," << endl;
        if (general)
            ofs << "  IGENERAL=1," << endl;
        if (ts[0].general)
            ofs << "  ITGENERAL=1," << endl;
        ofs << " &END" << endl;
        auto write_const = [](ofstream &os, double x) {
            os << fixed << setprecision(16);
            os << setw(20) << x << setw(4) << 0 << setw(4) << 0 << setw(4) << 0
               << setw(4) << 0 << endl;
        };
        if (!uhf) {
            if (general)
                ofs << vgs[0];
            else
                ofs << vs[0];
            ofs << ts[0];
            write_const(ofs, this->const_e);
        } else {
            if (general) {
                for (size_t i = 0; i < vgs.size(); i++)
                    ofs << vgs[i], write_const(ofs, 0.0);
            } else {
                for (size_t i = 0; i < vs.size(); i++)
                    ofs << vs[i], write_const(ofs, 0.0);
                ofs << vabs[0], write_const(ofs, 0.0);
            }
            for (size_t i = 0; i < ts.size(); i++)
                ofs << ts[i], write_const(ofs, 0.0);
            write_const(ofs, this->const_e);
        }
        if (!ofs.good())
            throw runtime_error("FCIDUMP::write on '" + filename + "' failed.");
        ofs.close();
    }
    // Parsing a FCIDUMP file
    virtual void read(const string &filename) {
        params.clear();
        ts.clear();
        vs.clear();
        vabs.clear();
        const_e = 0.0;
        ifstream ifs(filename.c_str());
        if (!ifs.good())
            throw runtime_error("FCIDUMP::read on '" + filename + "' failed.");
        vector<string> lines = Parsing::readlines(&ifs);
        if (ifs.bad())
            throw runtime_error("FCIDUMP::read on '" + filename + "' failed.");
        ifs.close();
        vector<string> pars;
        size_t il = 0;
        for (; il < lines.size(); il++) {
            string l(Parsing::lower(lines[il]));
            if (l.find("&fci") != string::npos)
                l.replace(l.find("&fci"), 4, "");
            if (l.find("/") != string::npos || l.find("&end") != string::npos)
                break;
            else
                pars.push_back(l);
        }
        il++;
        string par = Parsing::join(pars.begin(), pars.end(), ",");
        for (size_t ip = 0; ip < par.length(); ip++)
            if (par[ip] == ' ')
                par[ip] = ',';
        pars = Parsing::split(par, ",", true);
        string p_key = "";
        for (auto &c : pars) {
            if (c.find("=") != string::npos || p_key.length() == 0) {
                vector<string> cs = Parsing::split(c, "=", true);
                p_key = Parsing::trim(cs[0]);
                params[p_key] = cs.size() == 2 ? Parsing::trim(cs[1]) : "";
            } else {
                string cc = Parsing::trim(c);
                if (cc.length() != 0)
                    params[p_key] = params[p_key].length() == 0
                                        ? cc
                                        : params[p_key] + "," + cc;
            }
        }
        size_t int_sz = lines.size() > il ? lines.size() - il : 0;
        vector<array<uint16_t, 4>> int_idx(int_sz);
        vector<double> int_val(int_sz);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int64_t ill = 0; ill < (int64_t)int_sz; ill++) {
            string ll = Parsing::trim(lines[il + ill]);
            if (ll.length() == 0 || ll[0] == '!') {
                int_idx[ill][0] = numeric_limits<uint16_t>::max();
                continue;
            }
            vector<string> ls = Parsing::split(ll, " ", true);
            assert(ls.size() == 5);
            int_idx[ill] = array<uint16_t, 4>{(uint16_t)Parsing::to_int(ls[1]),
                                              (uint16_t)Parsing::to_int(ls[2]),
                                              (uint16_t)Parsing::to_int(ls[3]),
                                              (uint16_t)Parsing::to_int(ls[4])};
            int_val[ill] = Parsing::to_double(ls[0]);
        }
        threading->activate_normal();
        uint16_t n = (uint16_t)Parsing::to_int(params["norb"]);
        uhf = params.count("iuhf") != 0 && Parsing::to_int(params["iuhf"]) == 1;
        general = params.count("igeneral") != 0 &&
                  Parsing::to_int(params["igeneral"]) == 1;
        if (!uhf) {
            ts.push_back(TInt(n));
            ts[0].general = params.count("itgeneral") != 0 &&
                            Parsing::to_int(params["itgeneral"]) == 1;
            if (!general) {
                vs.push_back(V8Int(n));
                total_memory = ts[0].size() + vs[0].size();
                vdata = make_shared<vector<double>>(total_memory);
                data = vdata->data();
                ts[0].data = data;
                vs[0].data = data + ts[0].size();
                ts[0].clear();
                vs[0].clear();
            } else {
                vgs.push_back(V1Int(n));
                total_memory = ts[0].size() + vgs[0].size();
                vdata = make_shared<vector<double>>(total_memory);
                data = vdata->data();
                ts[0].data = data;
                vgs[0].data = data + ts[0].size();
                ts[0].clear();
                vgs[0].clear();
            }
            for (size_t i = 0; i < int_val.size(); i++) {
                if (int_idx[i][0] == numeric_limits<uint16_t>::max())
                    continue;
                if (int_idx[i][0] + int_idx[i][1] + int_idx[i][2] +
                        int_idx[i][3] ==
                    0)
                    const_e = int_val[i];
                else if (int_idx[i][2] + int_idx[i][3] == 0)
                    ts[0](int_idx[i][0] - 1, int_idx[i][1] - 1) = int_val[i];
                else if (!general)
                    vs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                          int_idx[i][2] - 1, int_idx[i][3] - 1) = int_val[i];
                else
                    vgs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                           int_idx[i][2] - 1, int_idx[i][3] - 1) = int_val[i];
            }
        } else {
            ts.push_back(TInt(n));
            ts.push_back(TInt(n));
            ts[0].general = ts[1].general =
                params.count("itgeneral") != 0 &&
                Parsing::to_int(params["itgeneral"]) == 1;
            if (!general) {
                vs.push_back(V8Int(n));
                vs.push_back(V8Int(n));
                vabs.push_back(V4Int(n));
                total_memory =
                    ((ts[0].size() + vs[0].size()) << 1) + vabs[0].size();
                vdata = make_shared<vector<double>>(total_memory);
                data = vdata->data();
                ts[0].data = data;
                ts[1].data = data + ts[0].size();
                vs[0].data = data + (ts[0].size() << 1);
                vs[1].data = data + (ts[0].size() << 1) + vs[0].size();
                vabs[0].data = data + ((ts[0].size() + vs[0].size()) << 1);
                ts[0].clear(), ts[1].clear();
                vs[0].clear(), vs[1].clear(), vabs[0].clear();
            } else {
                for (int i = 0; i < 3; i++)
                    vgs.push_back(V1Int(n));
                total_memory = ts[0].size() * 2 + vgs[0].size() * 3;
                vdata = make_shared<vector<double>>(total_memory);
                data = vdata->data();
                ts[0].data = data;
                ts[1].data = data + ts[0].size();
                vgs[0].data = data + (ts[0].size() << 1);
                vgs[1].data = data + (ts[0].size() << 1) + vgs[0].size();
                vgs[2].data = data + (ts[0].size() << 1) + (vgs[0].size() << 1);
                ts[0].clear(), ts[1].clear();
                vgs[0].clear(), vgs[1].clear(), vgs[2].clear();
            }
            int ip = 0;
            for (size_t i = 0; i < int_val.size(); i++) {
                if (int_idx[i][0] == numeric_limits<uint16_t>::max())
                    continue;
                if (int_idx[i][0] + int_idx[i][1] + int_idx[i][2] +
                        int_idx[i][3] ==
                    0) {
                    ip++;
                    if (ip == 6)
                        const_e = int_val[i];
                } else if (int_idx[i][2] + int_idx[i][3] == 0) {
                    ts[ip - 3](int_idx[i][0] - 1, int_idx[i][1] - 1) =
                        int_val[i];
                } else {
                    assert(ip <= 2);
                    if (!general) {
                        if (ip < 2)
                            vs[ip](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                   int_idx[i][2] - 1, int_idx[i][3] - 1) =
                                int_val[i];
                        else
                            vabs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                    int_idx[i][2] - 1, int_idx[i][3] - 1) =
                                int_val[i];
                    } else {
                        vgs[ip](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                int_idx[i][2] - 1, int_idx[i][3] - 1) =
                            int_val[i];
                    }
                }
            }
        }
    }
    // Remove integral elements that violate point group symmetry
    // orbsym: in XOR convention
    virtual double symmetrize(const vector<uint8_t> &orbsym) {
        uint16_t n = n_sites();
        assert((int)orbsym.size() == n);
        double error = 0.0;
        for (auto &t : ts)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < (t.general ? n : i + 1); j++)
                    if (orbsym[i] ^ orbsym[j])
                        error += abs(t(i, j)), t(i, j) = 0;
        for (auto &v : vgs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            if (orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l])
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vabs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            if (orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l])
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vs)
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl &&
                                (orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l]))
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        return error;
    }
    // Target 2S or 2Sz
    uint16_t twos() const {
        return (uint16_t)Parsing::to_int(params.at("ms2"));
    }
    // Number of sites
    uint16_t n_sites() const {
        return (uint16_t)Parsing::to_int(params.at("norb"));
    }
    // Number of electrons
    uint16_t n_elec() const {
        return (uint16_t)Parsing::to_int(params.at("nelec"));
    }
    // Target point group irreducible representation (counting from 1)
    uint8_t isym() const { return (uint8_t)Parsing::to_int(params.at("isym")); }
    // Set point group irreducible representation for each site
    void set_orb_sym(const vector<uint8_t> &x) {
        stringstream ss;
        for (size_t i = 0; i < x.size(); i++) {
            ss << (int)x[i];
            if (i != x.size() - 1)
                ss << ",";
        }
        params["orbsym"] = ss.str();
    }
    // Point group irreducible representation for each site
    vector<uint8_t> orb_sym() const {
        vector<string> x = Parsing::split(params.at("orbsym"), ",", true);
        vector<uint8_t> r;
        r.reserve(x.size());
        for (auto &xx : x)
            r.push_back((uint8_t)Parsing::to_int(xx));
        return r;
    }
    // energy of a determinant
    double det_energy(const vector<uint8_t> iocc, uint16_t i_begin,
                      uint16_t i_end) const {
        double energy = 0;
        uint16_t n_block_sites = i_end - i_begin;
        vector<uint8_t> spin_occ;
        assert(iocc.size() == n_block_sites ||
               iocc.size() == n_block_sites * 2);
        if (iocc.size() == n_block_sites) {
            spin_occ = vector<uint8_t>(n_block_sites * 2, 0);
            for (uint16_t i = 0; i < n_block_sites; i++) {
                spin_occ[i * 2] = iocc[i] >= 1;
                spin_occ[i * 2 + 1] = iocc[i] == 2;
            }
        } else
            spin_occ = iocc;
        for (uint16_t i = 0; i < n_block_sites; i++)
            for (uint8_t si = 0; si < 2; si++)
                if (spin_occ[i * 2 + si]) {
                    energy += t(si, i + i_begin, i + i_begin);
                    for (uint16_t j = 0; j < n_block_sites; j++)
                        for (uint8_t sj = 0; sj < 2; sj++)
                            if (spin_occ[j * 2 + sj]) {
                                energy +=
                                    0.5 * v(si, sj, i + i_begin, i + i_begin,
                                            j + i_begin, j + i_begin);
                                if (si == sj)
                                    energy -= 0.5 * v(si, sj, i + i_begin,
                                                      j + i_begin, j + i_begin,
                                                      i + i_begin);
                            }
                }
        return energy;
    }
    vector<double> h1e_energy() const {
        vector<double> r(n_sites());
        for (uint16_t i = 0; i < n_sites(); i++)
            r[i] = t(i, i);
        return r;
    }
    // h1e matrix
    vector<double> h1e_matrix() const {
        uint16_t n = n_sites();
        vector<double> r(n * n, 0);
        for (uint16_t i = 0; i < n; i++)
            for (uint16_t j = 0; j < n; j++)
                r[i * n + j] += t(i, j);
        return r;
    }
    // exchange matrix
    vector<double> exchange_matrix() const {
        uint16_t n = n_sites();
        vector<double> r(n * n, 0);
        for (uint16_t i = 0; i < n; i++)
            for (uint16_t j = 0; j < n; j++)
                r[i * n + j] += v(i, j, j, i);
        return r;
    }
    // abs h1e matrix
    vector<double> abs_h1e_matrix() const {
        uint16_t n = n_sites();
        vector<double> r(n * n, 0);
        for (uint8_t si = 0; si < 2; si++)
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    r[i * n + j] += 0.5 * abs(t(si, i, j));
        return r;
    }
    // abs exchange matrix
    vector<double> abs_exchange_matrix() const {
        uint16_t n = n_sites();
        vector<double> r(n * n, 0);
        for (uint16_t i = 0; i < n; i++)
            for (uint8_t si = 0; si < 2; si++)
                for (uint16_t j = 0; j < n; j++)
                    for (uint8_t sj = 0; sj < 2; sj++)
                        r[i * n + j] += 0.25 * abs(v(si, sj, i, j, j, i));
        return r;
    }
    template <typename T>
    static vector<T> reorder(const vector<T> &data,
                             const vector<uint16_t> &ord) {
        vector<T> rdata(data.size());
        assert(data.size() % ord.size() == 0);
        size_t nn = data.size() / ord.size();
        for (size_t i = 0; i < ord.size(); i++)
            for (size_t k = 0; k < nn; k++)
                rdata[i * nn + k] = data[ord[i] * nn + k];
        return rdata;
    }
    virtual void reorder(const vector<uint16_t> &ord) {
        uint16_t n = n_sites();
        assert(ord.size() == n);
        shared_ptr<vector<double>> rdata =
            make_shared<vector<double>>(total_memory);
        vector<TInt> rts(ts);
        vector<V1Int> rvgs(vgs);
        vector<V4Int> rvabs(vabs);
        vector<V8Int> rvs(vs);
        for (size_t i = 0; i < ts.size(); i++) {
            rts[i].data = ts[i].data - data + rdata->data();
            rts[i].reorder(ts[i], ord);
        }
        for (size_t i = 0; i < vgs.size(); i++) {
            rvgs[i].data = vgs[i].data - data + rdata->data();
            rvgs[i].reorder(vgs[i], ord);
        }
        for (size_t i = 0; i < vabs.size(); i++) {
            rvabs[i].data = vabs[i].data - data + rdata->data();
            rvabs[i].reorder(vabs[i], ord);
        }
        for (size_t i = 0; i < vs.size(); i++) {
            rvs[i].data = vs[i].data - data + rdata->data();
            rvs[i].reorder(vs[i], ord);
        }
        vdata = rdata;
        data = rdata->data();
        ts = rts, vgs = rvgs, vabs = rvabs, vs = rvs;
        if (params.count("orbsym"))
            set_orb_sym(reorder(orb_sym(), ord));
    }
    // orbital rotation
    // rot_mat: (old, new)
    virtual void rotate(const vector<double> &rot_mat) {
        uint16_t n = n_sites();
        assert((int)rot_mat.size() == (int)n * n);
        shared_ptr<vector<double>> rdata =
            make_shared<vector<double>>(total_memory);
        vector<TInt> rts(ts);
        vector<V1Int> rvgs(vgs);
        vector<V4Int> rvabs(vabs);
        vector<V8Int> rvs(vs);
        for (size_t i = 0; i < ts.size(); i++) {
            rts[i].data = ts[i].data - data + rdata->data();
            rts[i].rotate(ts[i], rot_mat);
        }
        for (size_t i = 0; i < vgs.size(); i++) {
            rvgs[i].data = vgs[i].data - data + rdata->data();
            rvgs[i].rotate(vgs[i], rot_mat);
        }
        for (size_t i = 0; i < vabs.size(); i++) {
            rvabs[i].data = vabs[i].data - data + rdata->data();
            rvabs[i].rotate(vabs[i], rot_mat);
        }
        for (size_t i = 0; i < vs.size(); i++) {
            rvs[i].data = vs[i].data - data + rdata->data();
            rvs[i].rotate(vs[i], rot_mat);
        }
        vdata = rdata;
        data = rdata->data();
        ts = rts, vgs = rvgs, vabs = rvabs, vs = rvs;
    }
    virtual shared_ptr<FCIDUMP> deep_copy() const {
        shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>(*this);
        fcidump->vdata = make_shared<vector<double>>(*vdata);
        fcidump->data = fcidump->vdata->data();
        vector<TInt> rts(ts);
        vector<V1Int> rvgs(vgs);
        vector<V4Int> rvabs(vabs);
        vector<V8Int> rvs(vs);
        for (size_t i = 0; i < ts.size(); i++)
            fcidump->ts[i].data = ts[i].data - data + fcidump->data;
        for (size_t i = 0; i < vgs.size(); i++)
            fcidump->vgs[i].data = vgs[i].data - data + fcidump->data;
        for (size_t i = 0; i < vabs.size(); i++)
            fcidump->vabs[i].data = vabs[i].data - data + fcidump->data;
        for (size_t i = 0; i < vs.size(); i++)
            fcidump->vs[i].data = vs[i].data - data + fcidump->data;
        return fcidump;
    }
    // One-electron integral element (SU(2))
    virtual double t(uint16_t i, uint16_t j) const { return ts[0](i, j); }
    // One-electron integral element (SZ)
    virtual double t(uint8_t s, uint16_t i, uint16_t j) const {
        return uhf ? ts[s](i, j) : ts[0](i, j);
    }
    // Two-electron integral element (SU(2))
    virtual double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return general ? vgs[0](i, j, k, l) : vs[0](i, j, k, l);
    }
    // Two-electron integral element (SZ)
    virtual double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
                     uint16_t l) const {
        if (uhf) {
            if (sl == sr)
                return general ? vgs[sl](i, j, k, l) : vs[sl](i, j, k, l);
            else if (sl == 0 && sr == 1)
                return general ? vgs[2](i, j, k, l) : vabs[0](i, j, k, l);
            else
                return general ? vgs[2](k, l, i, j) : vabs[0](k, l, i, j);
        } else
            return general ? vgs[0](i, j, k, l) : vs[0](i, j, k, l);
    }
    virtual double e() const { return const_e; }
    virtual void deallocate() {
        assert(total_memory != 0);
        vdata = nullptr;
        data = nullptr;
        ts.clear();
        vs.clear();
        vabs.clear();
        vgs.clear();
    }
};

} // namespace block2
