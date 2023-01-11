
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
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

template <typename FL>
inline void fd_write_line(ostream &os, FL x, uint16_t i = 0, uint16_t j = 0,
                          uint16_t k = 0, uint16_t l = 0) {
    os << fixed << setprecision(16);
    os << setw(24) << x << setw(4) << i << setw(4) << j << setw(4) << k
       << setw(4) << l << endl;
};

template <typename FL>
inline void fd_write_line(ostream &os, complex<FL> x, uint16_t i = 0,
                          uint16_t j = 0, uint16_t k = 0, uint16_t l = 0) {
    os << fixed << setprecision(16);
    os << setw(24) << real(x) << setw(20) << imag(x) << setw(4) << i << setw(4)
       << j << setw(4) << k << setw(4) << l << endl;
};

template <typename FL>
inline void fd_read_line(array<uint16_t, 4> &idx, FL &d,
                         const vector<string> &x) {
    assert(x.size() == 5);
    idx = array<uint16_t, 4>{
        (uint16_t)Parsing::to_int(x[1]), (uint16_t)Parsing::to_int(x[2]),
        (uint16_t)Parsing::to_int(x[3]), (uint16_t)Parsing::to_int(x[4])};
    d = (FL)Parsing::to_double(x[0]);
}

template <typename FL>
inline void fd_read_line(array<uint16_t, 4> &idx, complex<FL> &d,
                         const vector<string> &x) {
    if (x.size() == 6) {
        idx = array<uint16_t, 4>{
            (uint16_t)Parsing::to_int(x[2]), (uint16_t)Parsing::to_int(x[3]),
            (uint16_t)Parsing::to_int(x[4]), (uint16_t)Parsing::to_int(x[5])};
        d = complex<FL>((FL)Parsing::to_double(x[0]),
                        (FL)Parsing::to_double(x[1]));
    } else if (x.size() == 5) {
        idx = array<uint16_t, 4>{
            (uint16_t)Parsing::to_int(x[1]), (uint16_t)Parsing::to_int(x[2]),
            (uint16_t)Parsing::to_int(x[3]), (uint16_t)Parsing::to_int(x[4])};
        d = (complex<FL>)(FL)Parsing::to_double(x[0]);
    } else
        assert(false);
}

// Symmetric/general 2D array for storage of one-electron integrals
template <typename FL> struct TInt {
    // Number of orbitals
    uint16_t n;
    FL *data;
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
    void clear() { memset(data, 0, sizeof(FL) * size()); }
    FL &operator()(uint16_t i, uint16_t j) {
        return *(data + find_index(i, j));
    }
    FL operator()(uint16_t i, uint16_t j) const {
        return *(data + find_index(i, j));
    }
    void reorder(const TInt &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint16_t i = 0; i < n; i++)
            for (uint16_t j = 0; j < (general ? n : i + 1); j++)
                (*this)(i, j) = other(ord[i], ord[j]);
    }
    void rotate(const TInt &other, const vector<FL> &rot_mat) {
        assert(n == other.n);
        vector<FL> tmp((size_t)n * n);
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(size_t)q * n + j] * xconj(rot_mat[q * n + i]);
                (*this)(i, j) = x;
            }
        }
    }
    friend ostream &operator<<(ostream &os, TInt x) {
        os << fixed << setprecision(16);
        for (uint16_t i = 0; i < x.n; i++)
            for (uint16_t j = 0; j < (x.general ? x.n : i + 1); j++)
                if (x(i, j) != (FL)0.0)
                    fd_write_line(os, x(i, j), i + 1, j + 1);
        return os;
    }
};

// General 4D array for storage of two-electron integrals
template <typename FL> struct V1Int {
    // Number of orbitals
    uint32_t n;
    size_t m;
    FL *data;
    V1Int(uint32_t n) : n(n), m((size_t)n * n * n * n), data(nullptr) {}
    size_t size() const { return m; }
    void clear() { memset(data, 0, sizeof(FL) * size()); }
    FL &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + (((size_t)i * n + j) * n + k) * n + l);
    }
    FL operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
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
    void rotate(const V1Int &other, const vector<FL> &rot_mat) {
        assert(n == other.n);
        vector<FL> tmp(size());
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)i * n + j) * n + q) * n + l] *
                         xconj(rot_mat[q * n + k]);
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)q * n + j) * n + k) * n + l] *
                         xconj(rot_mat[q * n + i]);
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
                        if (x(i, j, k, l) != (FL)0.0)
                            fd_write_line(os, x(i, j, k, l), i + 1, j + 1,
                                          k + 1, l + 1);
        return os;
    }
};

// 4D array with 4-fold symmetry for storage of two-electron integrals
// [ijkl] = [jikl] = [jilk] = [ijlk]
template <typename FL> struct V4Int {
    // n: number of orbitals
    uint32_t n, m;
    FL *data;
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
    void clear() { memset(data, 0, sizeof(FL) * size()); }
    FL &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + find_index(i, j, k, l));
    }
    FL operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
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
    void rotate(const V4Int &other, const vector<FL> &rot_mat) {
        assert(n == other.n);
        vector<FL> tmp((size_t)n * n * n * n), tmp2((size_t)n * n * n * n);
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)i * n + j) * n + q) * n + l] *
                         xconj(rot_mat[q * n + k]);
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)q * n + j) * n + k) * n + l] *
                         xconj(rot_mat[q * n + i]);
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
                        if (x(i, j, k, l) != (FL)0.0)
                            fd_write_line(os, x(i, j, k, l), i + 1, j + 1,
                                          k + 1, l + 1);
        return os;
    }
};

// 4D array with 8-fold symmetry for storage of two-electron integrals
// [ijkl] = [jikl] = [jilk] = [ijlk] = [klij] = [klji] = [lkji] = [lkij]
template <typename FL> struct V8Int {
    // n: number of orbitals
    uint32_t n, m;
    FL *data;
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
    void clear() { memset(data, 0, sizeof(FL) * size()); }
    FL &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + find_index(i, j, k, l));
    }
    FL operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
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
    void rotate(const V8Int &other, const vector<FL> &rot_mat) {
        assert(n == other.n);
        vector<FL> tmp((size_t)n * n * n * n), tmp2((size_t)n * n * n * n);
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)i * n + j) * n + q) * n + l] *
                         xconj(rot_mat[q * n + k]);
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
                FL x = 0;
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
                FL x = 0;
                for (int q = 0; q < n; q++)
                    x += tmp[(((size_t)q * n + j) * n + k) * n + l] *
                         xconj(rot_mat[q * n + i]);
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
                        if (ij >= kl && x(i, j, k, l) != (FL)0.0)
                            fd_write_line(os, x(i, j, k, l), i + 1, j + 1,
                                          k + 1, l + 1);
        return os;
    }
};

// One- and two-electron integrals
template <typename FL> struct FCIDUMP {
    typedef decltype(abs((FL)0.0)) FP;
    shared_ptr<vector<FL>> vdata;
    map<string, string> params;
    vector<TInt<FL>> ts;
    vector<V8Int<FL>> vs;
    vector<V4Int<FL>> vabs;
    vector<V1Int<FL>> vgs;
    typename const_fl_type<FL>::FL const_e;
    FL *data;
    size_t total_memory;
    bool uhf, general;
    FCIDUMP() : const_e(0.0), uhf(false), total_memory(0), vdata(nullptr) {}
    // Initialize integrals: U(1) case
    // Two-electron integrals can be three general rank-4 arrays
    // or 8-fold, 8-fold, 4-fold rank-1 arrays
    virtual ~FCIDUMP() = default;
    virtual void initialize_sz(uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                               uint16_t isym, typename const_fl_type<FL>::FL e,
                               const FL *ta, size_t lta, const FL *tb,
                               size_t ltb, const FL *va, size_t lva,
                               const FL *vb, size_t lvb, const FL *vab,
                               size_t lvab) {
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
        ts.push_back(TInt<FL>(n_sites));
        ts.push_back(TInt<FL>(n_sites));
        if (lta != ts[0].size())
            ts[0].general = ts[1].general = true;
        assert(lta == ts[0].size() && ltb == ts[1].size());
        vs.push_back(V8Int<FL>(n_sites));
        vs.push_back(V8Int<FL>(n_sites));
        vabs.push_back(V4Int<FL>(n_sites));
        if (vs[0].size() == lva) {
            assert(vs[1].size() == lvb);
            assert(vabs[0].size() == lvab);
            general = false;
            total_memory = lta + ltb + lva + lvb + lvab;
            vdata = make_shared<vector<FL>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            ts[1].data = data + lta;
            vs[0].data = data + lta + ltb;
            vs[1].data = data + lta + ltb + lva;
            vabs[0].data = data + lta + ltb + lva + lvb;
            memcpy(vs[0].data, va, sizeof(FL) * lva);
            memcpy(vs[1].data, vb, sizeof(FL) * lvb);
            memcpy(vabs[0].data, vab, sizeof(FL) * lvab);
        } else {
            general = true;
            vs.clear();
            vabs.clear();
            vgs.push_back(V1Int<FL>(n_sites));
            vgs.push_back(V1Int<FL>(n_sites));
            vgs.push_back(V1Int<FL>(n_sites));
            assert(vgs[0].size() == lva);
            assert(vgs[1].size() == lvb);
            assert(vgs[2].size() == lvab);
            total_memory = lta + ltb + lva + lvb + lvab;
            vdata = make_shared<vector<FL>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            ts[1].data = data + lta;
            vgs[0].data = data + lta + ltb;
            vgs[1].data = data + lta + ltb + lva;
            vgs[2].data = data + lta + ltb + lva + lvb;
            memcpy(vgs[0].data, va, sizeof(FL) * lva);
            memcpy(vgs[1].data, vb, sizeof(FL) * lvb);
            memcpy(vgs[2].data, vab, sizeof(FL) * lvab);
        }
        memcpy(ts[0].data, ta, sizeof(FL) * lta);
        memcpy(ts[1].data, tb, sizeof(FL) * ltb);
        uhf = true;
    }
    // Initialize integrals: SU(2) case
    // Two-electron integrals can be general rank-4 array or 8-fold rank-1 array
    virtual void initialize_su2(uint16_t n_sites, uint16_t n_elec,
                                uint16_t twos, uint16_t isym,
                                typename const_fl_type<FL>::FL e, const FL *t,
                                size_t lt, const FL *v, size_t lv) {
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
        ts.push_back(TInt<FL>(n_sites));
        if (lt != ts[0].size())
            ts[0].general = true;
        assert(lt == ts[0].size());
        vs.push_back(V8Int<FL>(n_sites));
        if (vs[0].size() == lv) {
            general = false;
            total_memory = ts[0].size() + vs[0].size();
            vdata = make_shared<vector<FL>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            vs[0].data = data + ts[0].size();
            memcpy(vs[0].data, v, sizeof(FL) * lv);
        } else {
            general = true;
            vs.clear();
            vgs.push_back(V1Int<FL>(n_sites));
            assert(lv == vgs[0].size());
            total_memory = ts[0].size() + vgs[0].size();
            vdata = make_shared<vector<FL>>(total_memory);
            data = vdata->data();
            ts[0].data = data;
            vgs[0].data = data + ts[0].size();
            memcpy(vgs[0].data, v, sizeof(FL) * lv);
        }
        memcpy(ts[0].data, t, sizeof(FL) * lt);
        uhf = false;
    }
    // Initialize with only h1e integral
    virtual void initialize_h1e(uint16_t n_sites, uint16_t n_elec,
                                uint16_t twos, uint16_t isym,
                                typename const_fl_type<FL>::FL e, const FL *t,
                                size_t lt) {
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
        ts.push_back(TInt<FL>(n_sites));
        if (lt != ts[0].size())
            ts[0].general = true;
        assert(lt == ts[0].size());
        vs.push_back(V8Int<FL>(n_sites));
        general = false;
        total_memory = ts[0].size() + vs[0].size();
        vdata = make_shared<vector<FL>>(total_memory);
        data = vdata->data();
        ts[0].data = data;
        vs[0].data = data + ts[0].size();
        memset(vs[0].data, 0, sizeof(FL) * vs[0].size());
        memcpy(ts[0].data, t, sizeof(FL) * lt);
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
        if (params.count("ksym") != 0)
            ofs << "  KSYM=" << params.at("ksym") << "," << endl;
        if (params.count("kmod") != 0)
            ofs << "  KMOD=" << (int)k_mod() << "," << endl;
        ofs << "  ISYM=" << setw(4) << (int)isym() << "," << endl;
        if (uhf)
            ofs << "  IUHF=1," << endl;
        if (general)
            ofs << "  IGENERAL=1," << endl;
        if (ts[0].general)
            ofs << "  ITGENERAL=1," << endl;
        ofs << " &END" << endl;
        if (!uhf) {
            if (general)
                ofs << vgs[0];
            else
                ofs << vs[0];
            ofs << ts[0];
            fd_write_line(ofs, this->const_e);
        } else {
            if (general) {
                for (size_t i = 0; i < vgs.size(); i++)
                    ofs << vgs[i], fd_write_line(ofs, 0.0);
            } else {
                for (size_t i = 0; i < vs.size(); i++)
                    ofs << vs[i], fd_write_line(ofs, 0.0);
                ofs << vabs[0], fd_write_line(ofs, 0.0);
            }
            for (size_t i = 0; i < ts.size(); i++)
                ofs << ts[i], fd_write_line(ofs, 0.0);
            fd_write_line(ofs, (FL)this->const_e);
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
        const_e = (typename const_fl_type<FL>::FL)0.0;
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
        vector<FL> int_val(int_sz);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int64_t ill = 0; ill < (int64_t)int_sz; ill++) {
            string ll = Parsing::trim(lines[il + ill]);
            if (ll.length() == 0 || ll[0] == '!') {
                int_idx[ill][0] = numeric_limits<uint16_t>::max();
                continue;
            }
            vector<string> ls = Parsing::split(ll, " ", true);
            fd_read_line(int_idx[ill], int_val[ill], ls);
            if (int_idx[ill][0] + int_idx[ill][1] + int_idx[ill][2] +
                    int_idx[ill][3] ==
                0) {
                typename const_fl_type<FL>::FL tmp_const_e;
                fd_read_line(int_idx[ill], tmp_const_e, ls);
                if (tmp_const_e != (typename const_fl_type<FL>::FL)0.0)
                    const_e = tmp_const_e;
            }
        }
        threading->activate_normal();
        uint16_t n = (uint16_t)Parsing::to_int(params["norb"]);
        uhf = params.count("iuhf") != 0 && Parsing::to_int(params["iuhf"]) == 1;
        general = params.count("igeneral") != 0 &&
                  Parsing::to_int(params["igeneral"]) == 1;
        if (!uhf) {
            ts.push_back(TInt<FL>(n));
            ts[0].general = params.count("itgeneral") != 0 &&
                            Parsing::to_int(params["itgeneral"]) == 1;
            if (!general) {
                vs.push_back(V8Int<FL>(n));
                total_memory = ts[0].size() + vs[0].size();
                vdata = make_shared<vector<FL>>(total_memory);
                data = vdata->data();
                ts[0].data = data;
                vs[0].data = data + ts[0].size();
                ts[0].clear();
                vs[0].clear();
            } else {
                vgs.push_back(V1Int<FL>(n));
                total_memory = ts[0].size() + vgs[0].size();
                vdata = make_shared<vector<FL>>(total_memory);
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
                    ;
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
            ts.push_back(TInt<FL>(n));
            ts.push_back(TInt<FL>(n));
            ts[0].general = ts[1].general =
                params.count("itgeneral") != 0 &&
                Parsing::to_int(params["itgeneral"]) == 1;
            if (!general) {
                vs.push_back(V8Int<FL>(n));
                vs.push_back(V8Int<FL>(n));
                vabs.push_back(V4Int<FL>(n));
                total_memory =
                    ((ts[0].size() + vs[0].size()) << 1) + vabs[0].size();
                vdata = make_shared<vector<FL>>(total_memory);
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
                    vgs.push_back(V1Int<FL>(n));
                total_memory = ts[0].size() * 2 + vgs[0].size() * 3;
                vdata = make_shared<vector<FL>>(total_memory);
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
                        ;
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
    // Remove small integral elements
    virtual FP truncate_small(FP tol) {
        uint16_t n = n_sites();
        FP error = 0.0;
        for (auto &t : ts)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < (t.general ? n : i + 1); j++)
                    if (abs(t(i, j)) < tol)
                        error += abs(t(i, j)), t(i, j) = 0;
        for (auto &v : vgs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            if (abs(v(i, j, k, l)) < tol)
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vabs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            if (abs(v(i, j, k, l)) < tol)
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vs)
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl && abs(v(i, j, k, l)) < tol)
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        return error;
    }
    // Remove integral elements that violate point group symmetry
    // orbsym: in XOR convention
    virtual FP symmetrize(const vector<uint8_t> &orbsym) {
        uint16_t n = n_sites();
        assert((int)orbsym.size() == n);
        FP error = 0.0;
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
    // Remove integral elements that violate point group symmetry
    // orbsym: in Lz convention
    virtual FP symmetrize(const vector<int16_t> &orbsym) {
        uint16_t n = n_sites();
        assert((int)orbsym.size() == n);
        FP error = 0.0;
        for (auto &t : ts)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < (t.general ? n : i + 1); j++)
                    if (orbsym[i] - orbsym[j])
                        error += abs(t(i, j)), t(i, j) = 0;
        for (auto &v : vgs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            if (orbsym[i] - orbsym[j] + orbsym[k] - orbsym[l])
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vabs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            if (orbsym[i] - orbsym[j] + orbsym[k] - orbsym[l])
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vs)
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl &&
                                (orbsym[i] - orbsym[j] + orbsym[k] - orbsym[l]))
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        return error;
    }
    // Remove integral elements that violate point group symmetry
    // ksym: k symmetry
    virtual FP symmetrize(const vector<int> &ksym, int kmod) {
        uint16_t n = n_sites();
        assert((int)ksym.size() == n);
        if (vabs.size() != 0 || vs.size() != 0)
            cout << "WARNING: k symmetry should not be used together with "
                    "4-fold or 8-fold symmetry."
                 << endl;
        FP error = 0.0;
        for (auto &t : ts)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < (t.general ? n : i + 1); j++)
                    if ((kmod == 0 && ksym[i] - ksym[j]) ||
                        (kmod != 0 && (ksym[i] + kmod - ksym[j]) % kmod))
                        error += abs(t(i, j)), t(i, j) = 0;
        for (auto &v : vgs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            if ((kmod == 0 &&
                                 ksym[i] - ksym[j] + ksym[k] - ksym[l]) ||
                                (kmod != 0 && (ksym[i] + kmod - ksym[j] +
                                               ksym[k] + kmod - ksym[l]) %
                                                  kmod))
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vabs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            if ((kmod == 0 &&
                                 ksym[i] - ksym[j] + ksym[k] - ksym[l]) ||
                                (kmod != 0 && (ksym[i] + kmod - ksym[j] +
                                               ksym[k] + kmod - ksym[l]) %
                                                  kmod))
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : vs)
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl &&
                                ((kmod == 0 &&
                                  ksym[i] - ksym[j] + ksym[k] - ksym[l]) ||
                                 (kmod != 0 && (ksym[i] + kmod - ksym[j] +
                                                ksym[k] + kmod - ksym[l]) %
                                                   kmod)))
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
    template <typename T> void set_orb_sym(const vector<T> &x) {
        stringstream ss;
        for (size_t i = 0; i < x.size(); i++) {
            ss << (int)x[i];
            if (i != x.size() - 1)
                ss << ",";
        }
        params["orbsym"] = ss.str();
    }
    // Point group irreducible representation for each site
    template <typename T> vector<T> orb_sym() const {
        vector<string> x = Parsing::split(params.at("orbsym"), ",", true);
        vector<T> r;
        r.reserve(x.size());
        for (auto &xx : x)
            r.push_back((T)Parsing::to_int(xx));
        return r;
    }
    // Set k symmetry for each site
    template <typename T> void set_k_sym(const vector<T> &x) {
        stringstream ss;
        for (size_t i = 0; i < x.size(); i++) {
            ss << (int)x[i];
            if (i != x.size() - 1)
                ss << ",";
        }
        params["ksym"] = ss.str();
    }
    // k symmetry for each site
    template <typename T> vector<T> k_sym() const {
        if (!params.count("ksym"))
            return vector<T>(n_sites(), (T)0);
        vector<string> x = Parsing::split(params.at("ksym"), ",", true);
        vector<T> r;
        r.reserve(x.size());
        for (auto &xx : x)
            r.push_back((T)Parsing::to_int(xx));
        return r;
    }
    // Set modulus for k symmetry
    void set_k_mod(int kmod) { params["kmod"] = Parsing::to_string(kmod); }
    // Modulus for k symmetry
    int k_mod() const {
        return params.count("kmod") ? Parsing::to_int(params.at("kmod")) : 0;
    }
    // Set target state for k symmetry
    void set_k_isym(int k_isym) {
        params["kisym"] = Parsing::to_string(k_isym);
    }
    // Target state for k symmetry
    int k_isym() const {
        return params.count("kisym") ? Parsing::to_int(params.at("kisym")) : 0;
    }
    // return number of non-zero terms
    virtual size_t count_non_zero() const {
        uint16_t nn = n_sites();
        size_t cnt = 0;
        for (uint16_t i = 0; i < nn; i++)
            for (uint16_t j = 0; j < nn; j++)
                for (uint16_t k = 0; k < nn; k++)
                    for (uint16_t l = 0; l < nn; l++)
                        cnt += (v(i, j, k, l) != (FL)0.0);
        for (uint16_t i = 0; i < nn; i++)
            for (uint16_t j = 0; j < nn; j++)
                cnt += (t(i, j) != (FL)0.0);
        return cnt;
    }
    // energy of a determinant
    FL det_energy(const vector<uint8_t> iocc, uint16_t i_begin,
                  uint16_t i_end) const {
        FL energy = 0;
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
                                energy += (FP)0.5 * v(si, sj, i + i_begin,
                                                      i + i_begin, j + i_begin,
                                                      j + i_begin);
                                if (si == sj)
                                    energy -=
                                        (FP)0.5 * v(si, sj, i + i_begin,
                                                    j + i_begin, j + i_begin,
                                                    i + i_begin);
                            }
                }
        return energy;
    }
    vector<FL> h1e_energy() const {
        vector<FL> r(n_sites());
        for (uint16_t i = 0; i < n_sites(); i++)
            r[i] = t(i, i);
        return r;
    }
    // h1e matrix
    vector<FL> h1e_matrix(int8_t s = -1) const {
        uint16_t n = n_sites();
        vector<FL> r(n * n, 0);
        if (s == -1)
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    r[i * n + j] += t(i, j);
        else
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    r[i * n + j] += t(s, i, j);
        return r;
    }
    // g2e 1-fold
    vector<FL> g2e_1fold(int8_t sl = -1, int8_t sr = -1) const {
        const int n = n_sites();
        const size_t m = (size_t)n * n;
        vector<FL> r(m * m, 0);
        size_t ijkl = 0;
        if (sl == -1 || sr == -1) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            r[ijkl++] = v(i, j, k, l);
        } else {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            r[ijkl++] = v(sl, sr, i, j, k, l);
        }
        assert(ijkl == r.size());
        return r;
    }
    // g2e 4-fold
    vector<FL> g2e_4fold(int8_t sl = -1, int8_t sr = -1) const {
        const int n = n_sites();
        const size_t m = (size_t)n * (n + 1) >> 1;
        vector<FL> r(m * m, 0);
        size_t ijkl = 0;
        if (sl == -1 || sr == -1) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            r[ijkl++] = v(i, j, k, l);
        } else {
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            r[ijkl++] = v(sl, sr, i, j, k, l);
        }
        assert(ijkl == r.size());
        return r;
    }
    // g2e 8-fold
    vector<FL> g2e_8fold(int8_t sl = -1, int8_t sr = -1) const {
        const int n = n_sites();
        const size_t m = (size_t)n * (n + 1) >> 1;
        vector<FL> r(m * (m + 1) >> 1, 0);
        size_t ijkl = 0;
        if (sl == -1 || sr == -1) {
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl)
                                r[ijkl++] = v(i, j, k, l);
        } else {
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl)
                                r[ijkl++] = v(sl, sr, i, j, k, l);
        }
        assert(ijkl == r.size());
        return r;
    }
    // exchange matrix
    vector<FL> exchange_matrix() const {
        uint16_t n = n_sites();
        vector<FL> r(n * n, 0);
        for (uint16_t i = 0; i < n; i++)
            for (uint16_t j = 0; j < n; j++)
                r[i * n + j] += v(i, j, j, i);
        return r;
    }
    // abs h1e matrix
    vector<FP> abs_h1e_matrix() const {
        uint16_t n = n_sites();
        vector<FP> r(n * n, 0);
        for (uint8_t si = 0; si < 2; si++)
            for (uint16_t i = 0; i < n; i++)
                for (uint16_t j = 0; j < n; j++)
                    r[i * n + j] += 0.5 * abs(t(si, i, j));
        return r;
    }
    // abs exchange matrix
    vector<FP> abs_exchange_matrix() const {
        uint16_t n = n_sites();
        vector<FP> r(n * n, 0);
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
        shared_ptr<vector<FL>> rdata = make_shared<vector<FL>>(total_memory);
        vector<TInt<FL>> rts(ts);
        vector<V1Int<FL>> rvgs(vgs);
        vector<V4Int<FL>> rvabs(vabs);
        vector<V8Int<FL>> rvs(vs);
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
            set_orb_sym(reorder(orb_sym<int>(), ord));
        if (params.count("ksym"))
            set_k_sym(reorder(k_sym<int>(), ord));
    }
    virtual void rescale(typename const_fl_type<FL>::FL shift = 0) {
        typename const_fl_type<FL>::FL x = 0;
        uint16_t xn = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            xn += ts[i].n;
            for (uint16_t j = 0; j < ts[i].n; j++)
                x += ts[i](j, j);
        }
        if (shift == (typename const_fl_type<FL>::FL)0.0)
            x = x / (typename const_fl_type<FP>::FL)xn;
        else
            x = (shift - const_e) / (typename const_fl_type<FP>::FL)n_elec();
        for (size_t i = 0; i < ts.size(); i++)
            for (uint16_t j = 0; j < ts[i].n; j++)
                ts[i](j, j) = ts[i](j, j) - (FL)x;
        const_e = const_e + x * (typename const_fl_type<FP>::FL)n_elec();
    }
    // orbital rotation
    // rot_mat: (old, new)
    virtual void rotate(const vector<FL> &rot_mat) {
        uint16_t n = n_sites();
        assert((int)rot_mat.size() == (int)n * n);
        shared_ptr<vector<FL>> rdata = make_shared<vector<FL>>(total_memory);
        vector<TInt<FL>> rts(ts);
        vector<V1Int<FL>> rvgs(vgs);
        vector<V4Int<FL>> rvabs(vabs);
        vector<V8Int<FL>> rvs(vs);
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
        fcidump->vdata = make_shared<vector<FL>>(*vdata);
        fcidump->data = fcidump->vdata->data();
        vector<TInt<FL>> rts(ts);
        vector<V1Int<FL>> rvgs(vgs);
        vector<V4Int<FL>> rvabs(vabs);
        vector<V8Int<FL>> rvs(vs);
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
    virtual FL t(uint16_t i, uint16_t j) const { return ts[0](i, j); }
    // One-electron integral element (SZ)
    virtual FL t(uint8_t s, uint16_t i, uint16_t j) const {
        return uhf ? ts[s](i, j) : ts[0](i, j);
    }
    // Two-electron integral element (SU(2))
    virtual FL v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return general ? vgs[0](i, j, k, l) : vs[0](i, j, k, l);
    }
    // Two-electron integral element (SZ)
    virtual FL v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
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
    virtual typename const_fl_type<FL>::FL e() const { return const_e; }
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

template <typename FL> struct MRCISFCIDUMP : FCIDUMP<FL> {
    using FCIDUMP<FL>::params;
    shared_ptr<FCIDUMP<FL>> prim_fcidump;
    uint16_t n_inactive, n_virtual, n_active;
    MRCISFCIDUMP(const shared_ptr<FCIDUMP<FL>> &fcidump, uint16_t n_inactive,
                 uint16_t n_virtual)
        : FCIDUMP<FL>(), prim_fcidump(fcidump), n_inactive(n_inactive),
          n_virtual(n_virtual),
          n_active(fcidump->n_sites() - n_inactive - n_virtual) {
        params = fcidump->params;
    }
    virtual ~MRCISFCIDUMP() = default;
    // One-electron integral element (SU(2))
    FL t(uint16_t i, uint16_t j) const override {
        return prim_fcidump->t(i, j);
    }
    // One-electron integral element (SZ)
    FL t(uint8_t s, uint16_t i, uint16_t j) const override {
        return prim_fcidump->t(s, i, j);
    }
    // Two-electron integral element (SU(2))
    FL v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        const uint16_t nocc = n_inactive + n_active;
        const int cnt = (i >= nocc) + (j >= nocc) + (k >= nocc) + (l >= nocc);
        return cnt <= 2 ? prim_fcidump->v(i, j, k, l) : 0;
    }
    // Two-electron integral element (SZ)
    FL v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
         uint16_t l) const override {
        const uint16_t nocc = n_inactive + n_active;
        const int cnt = (i >= nocc) + (j >= nocc) + (k >= nocc) + (l >= nocc);
        return cnt <= 2 ? prim_fcidump->v(sl, sr, i, j, k, l) : 0;
    }
    typename const_fl_type<FL>::FL e() const override {
        return prim_fcidump->e();
    }
    void deallocate() override {}
};

template <typename FL> struct SpinOrbitalFCIDUMP : FCIDUMP<FL> {
    using FCIDUMP<FL>::params;
    typedef typename FCIDUMP<FL>::FP FP;
    shared_ptr<FCIDUMP<FL>> prim_fcidump;
    SpinOrbitalFCIDUMP(const shared_ptr<FCIDUMP<FL>> &fcidump)
        : FCIDUMP<FL>(), prim_fcidump(fcidump) {
        params = fcidump->params;
        uint16_t n_spin_sites = fcidump->n_sites() * 2;
        params["norb"] = Parsing::to_string(n_spin_sites);
        vector<string> x =
            Parsing::split(fcidump->params.at("orbsym"), ",", true);
        stringstream ss;
        for (uint16_t i = 0; i < n_spin_sites; i++) {
            ss << x[i / 2];
            if (i != n_spin_sites - 1)
                ss << ",";
        }
        params["orbsym"] = ss.str();
    }
    virtual ~SpinOrbitalFCIDUMP() = default;
    // Remove integral elements that violate point group symmetry
    // orbsym: in XOR convention
    FP symmetrize(const vector<uint8_t> &orbsym) override {
        vector<uint8_t> prim_orbsym(orbsym.size() / 2);
        for (size_t i = 0; i < orbsym.size() / 2; i++)
            prim_orbsym[i] = orbsym[i * 2];
        return prim_fcidump->symmetrize(prim_orbsym);
    }
    // One-electron integral element (SGF)
    FL t(uint16_t i, uint16_t j) const override {
        if ((i ^ j) & 1)
            return 0.0;
        else
            return prim_fcidump->t(i & 1, i >> 1, j >> 1);
    }
    // One-electron integral element (SZ)
    FL t(uint8_t s, uint16_t i, uint16_t j) const override {
        assert(false);
        return t(i, j);
    }
    // Two-electron integral element (SGF)
    FL v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        if (((i ^ j) & 1) || ((k ^ l) & 1))
            return 0.0;
        else
            return prim_fcidump->v(i & 1, k & 1, i >> 1, j >> 1, k >> 1,
                                   l >> 1);
    }
    // Two-electron integral element (SZ)
    FL v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
         uint16_t l) const override {
        assert(false);
        return v(i, j, k, l);
    }
    typename const_fl_type<FL>::FL e() const override {
        return prim_fcidump->e();
    }
    void deallocate() override {}
};

} // namespace block2
