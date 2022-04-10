
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

#include "allocator.hpp"
#include "threading.hpp"
#ifdef _HAS_INTEL_MKL
#ifndef MKL_Complex16
#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#endif
#include "mkl.h"
#endif
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace std;

#define _MINTSZ (sizeof(MKL_INT) / sizeof(int32_t))

namespace block2 {

// General Matrix
template <typename FL> struct GMatrix;

// 2D dense matrix stored in stack memory
template <> struct GMatrix<float> {
    typedef double FL;
    typedef float FP;
    typedef complex<float> FC;
    MKL_INT m, n; // m is rows, n is cols
    float *data;
    GMatrix(float *data, MKL_INT m, MKL_INT n) : data(data), m(m), n(n) {}
    float &operator()(MKL_INT i, MKL_INT j) const {
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc_<float>() : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<float>() : alloc)->deallocate(data, size());
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(float)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    float trace() const {
        assert(m == n);
        float r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

// 2D dense matrix stored in stack memory
template <> struct GMatrix<double> {
    typedef long double FL;
    typedef double FP;
    typedef complex<double> FC;
    MKL_INT m, n; // m is rows, n is cols
    double *data;
    GMatrix(double *data, MKL_INT m, MKL_INT n) : data(data), m(m), n(n) {}
    double &operator()(MKL_INT i, MKL_INT j) const {
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc_<double>() : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<double>() : alloc)
            ->deallocate(data, size());
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(double)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    double trace() const {
        assert(m == n);
        double r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

typedef GMatrix<double> MatrixRef;

template <typename FL, typename = void> struct GDiagonalMatrix;

// Diagonal matrix
template <typename FL>
struct GDiagonalMatrix<FL,
                       typename enable_if<is_floating_point<FL>::value>::type>
    : GMatrix<FL> {
    using GMatrix<FL>::data;
    using GMatrix<FL>::m;
    using GMatrix<FL>::n;
    FL zero = 0.0;
    GDiagonalMatrix(FL *data, MKL_INT n) : GMatrix<FL>(data, n, n) {}
    FL &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? *(data + i) : const_cast<FL &>(zero);
    }
    size_t size() const { return (size_t)m; }
    // need override since size() is changed (which is not virtual)
    void allocate(const shared_ptr<Allocator<FL>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc_<FL>() : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<FL>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<FL>() : alloc)->deallocate(data, size());
        data = nullptr;
    }
    void clear() { memset(data, 0, size() * sizeof(FL)); }
    friend ostream &operator<<(ostream &os, const GDiagonalMatrix &mat) {
        os << "DIAG MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        os << "[ ";
        for (MKL_INT j = 0; j < mat.n; j++)
            os << setw(20) << setprecision(14) << mat(j, j) << " ";
        os << "]" << endl;
        return os;
    }
};

typedef GDiagonalMatrix<double> DiagonalMatrix;

template <typename FL, typename = void> struct GIdentityMatrix;

// Identity matrix
template <typename FL>
struct GIdentityMatrix<FL,
                       typename enable_if<is_floating_point<FL>::value>::type>
    : GDiagonalMatrix<FL> {
    using GDiagonalMatrix<FL>::zero;
    FL one = 1.0;
    GIdentityMatrix(MKL_INT n) : GDiagonalMatrix<FL>(nullptr, n) {}
    FL &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? const_cast<FL &>(one) : const_cast<FL &>(zero);
    }
    void allocate() {}
    void deallocate() {}
    void clear() {}
    friend ostream &operator<<(ostream &os, const GIdentityMatrix &mat) {
        os << "IDENT MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        return os;
    }
};

typedef GIdentityMatrix<double> IdentityMatrix;

// complex dense matrix
template <> struct GMatrix<complex<float>> {
    typedef float FP;
    typedef complex<float> FC;
    MKL_INT m, n; // m is rows, n is cols
    complex<float> *data;
    GMatrix(complex<float> *data, MKL_INT m, MKL_INT n)
        : data(data), m(m), n(n) {}
    complex<float> &operator()(MKL_INT i, MKL_INT j) const {
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        data = (complex<float> *)(alloc == nullptr ? dalloc_<float>() : alloc)
                   ->allocate(size() * 2);
    }
    void deallocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<float>() : alloc)
            ->deallocate((float *)data, size() * 2);
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(complex<float>)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "CPX-MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    complex<float> trace() const {
        assert(m == n);
        complex<float> r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

// complex dense matrix
template <> struct GMatrix<complex<double>> {
    typedef double FP;
    typedef complex<double> FC;
    MKL_INT m, n; // m is rows, n is cols
    complex<double> *data;
    GMatrix(complex<double> *data, MKL_INT m, MKL_INT n)
        : data(data), m(m), n(n) {}
    complex<double> &operator()(MKL_INT i, MKL_INT j) const {
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        data = (complex<double> *)(alloc == nullptr ? dalloc_<FP>() : alloc)
                   ->allocate(size() * 2);
    }
    void deallocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<FP>() : alloc)
            ->deallocate((double *)data, size() * 2);
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(complex<double>)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "CPX-MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    complex<double> trace() const {
        assert(m == n);
        complex<double> r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

typedef GMatrix<complex<double>> ComplexMatrixRef;

// Diagonal complex matrix
template <typename FL>
struct GDiagonalMatrix<FL, typename enable_if<is_complex<FL>::value>::type>
    : GMatrix<FL> {
    using GMatrix<FL>::data;
    using GMatrix<FL>::m;
    using GMatrix<FL>::n;
    FL zero = 0.0;
    GDiagonalMatrix(FL *data, MKL_INT n) : GMatrix<FL>(data, n, n) {}
    FL &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? *(data + i) : const_cast<FL &>(zero);
    }
    size_t size() const { return (size_t)m; }
    // need override since size() is changed (which is not virtual)
    void allocate(const shared_ptr<Allocator<typename GMatrix<FL>::FP>> &alloc =
                      nullptr) {
        data = (FL *)(alloc == nullptr ? dalloc_<typename GMatrix<FL>::FP>()
                                       : alloc)
                   ->allocate(size() * 2);
    }
    void deallocate(const shared_ptr<Allocator<typename GMatrix<FL>::FP>>
                        &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<typename GMatrix<FL>::FP>() : alloc)
            ->deallocate((typename GMatrix<FL>::FP *)data, size() * 2);
        data = nullptr;
    }
    void clear() { memset(data, 0, size() * sizeof(FL)); }
    friend ostream &operator<<(ostream &os, const GDiagonalMatrix &mat) {
        os << "DIAG CPX-MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        os << "[ ";
        for (MKL_INT j = 0; j < mat.n; j++)
            os << setw(20) << setprecision(14) << mat(j, j) << " ";
        os << "]" << endl;
        return os;
    }
};

typedef GDiagonalMatrix<complex<double>> ComplexDiagonalMatrix;

// General rank-n dense tensor
template <typename FL> struct GTensor {
    vector<MKL_INT> shape;
    vector<FL> data;
    GTensor(MKL_INT m, MKL_INT k, MKL_INT n) : shape{m, k, n} {
        data.resize((size_t)m * k * n);
    }
    GTensor(const vector<MKL_INT> &shape) : shape(shape) {
        size_t x = 1;
        for (MKL_INT sh : shape)
            x = x * (size_t)sh;
        data.resize(x);
    }
    size_t size() const { return data.size(); }
    void clear() { memset(data.data(), 0, size() * sizeof(FL)); }
    void truncate(MKL_INT n) {
        assert(shape.size() == 1);
        data.resize(n);
        shape[0] = n;
    }
    void truncate_left(MKL_INT nl) {
        assert(shape.size() == 2);
        data.resize(nl * shape[1]);
        shape[0] = nl;
    }
    void truncate_right(MKL_INT nr) {
        assert(shape.size() == 2);
        for (MKL_INT i = 1; i < shape[0]; i++)
            memmove(data.data() + i * nr, data.data() + i * shape[1],
                    nr * sizeof(FL));
        data.resize(shape[0] * nr);
        shape[1] = nr;
    }
    GMatrix<FL> ref() {
        if (shape.size() == 3 && shape[1] == 1)
            return GMatrix<FL>(data.data(), shape[0], shape[2]);
        else if (shape.size() == 2)
            return GMatrix<FL>(data.data(), shape[0], shape[1]);
        else if (shape.size() == 1)
            return GMatrix<FL>(data.data(), shape[0], 1);
        else {
            assert(false);
            return GMatrix<FL>(data.data(), 0, 1);
        }
    }
    FL &operator()(initializer_list<MKL_INT> idx) {
        size_t i = 0;
        int k = 0;
        for (auto &ix : idx)
            i = i * shape[k++] + ix;
        return data.at(i);
    }
    friend ostream &operator<<(ostream &os, const GTensor &ts) {
        os << "TENSOR ( ";
        for (auto sh : ts.shape)
            os << sh << " ";
        os << ")" << endl;
        os << "   DATA [";
        for (auto x : ts.data)
            os << fixed << setw(20) << setprecision(14) << x << " ";
        os << "]" << endl;
        return os;
    }
};

typedef GTensor<double> Tensor;
typedef GTensor<complex<double>> ComplexTensor;

} // namespace block2
