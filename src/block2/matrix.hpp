
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
#include "threading.hpp"
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace std;

#define _MINTSZ (sizeof(MKL_INT) / sizeof(int32_t))

namespace block2 {

// 2D dense matrix stored in stack memory
struct MatrixRef {
    MKL_INT m, n; // m is rows, n is cols
    double *data;
    MatrixRef(double *data, MKL_INT m, MKL_INT n) : data(data), m(m), n(n) {}
    double &operator()(MKL_INT i, MKL_INT j) const {
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc : alloc)->deallocate(data, size());
        data = nullptr;
    }
    void clear() { memset(data, 0, size() * sizeof(double)); }
    MatrixRef flip_dims() const { return MatrixRef(data, n, m); }
    MatrixRef shift_ptr(size_t l) const { return MatrixRef(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const MatrixRef &mat) {
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

// Diagonal matrix
struct DiagonalMatrix : MatrixRef {
    double zero = 0.0;
    DiagonalMatrix(double *data, MKL_INT n) : MatrixRef(data, n, n) {}
    double &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? *(data + i) : const_cast<double &>(zero);
    }
    size_t size() const { return (size_t)m; }
    // need override since size() is changed (which is not virtual)
    void allocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc : alloc)->deallocate(data, size());
        data = nullptr;
    }
    void clear() { memset(data, 0, size() * sizeof(double)); }
    friend ostream &operator<<(ostream &os, const DiagonalMatrix &mat) {
        os << "DIAG MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        os << "[ ";
        for (MKL_INT j = 0; j < mat.n; j++)
            os << setw(20) << setprecision(14) << mat(j, j) << " ";
        os << "]" << endl;
        return os;
    }
};

// Identity matrix
struct IdentityMatrix : DiagonalMatrix {
    double one = 1.0;
    IdentityMatrix(MKL_INT n) : DiagonalMatrix(nullptr, n) {}
    double &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? const_cast<double &>(one) : const_cast<double &>(zero);
    }
    void allocate() {}
    void deallocate() {}
    void clear() {}
    friend ostream &operator<<(ostream &os, const IdentityMatrix &mat) {
        os << "IDENT MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        return os;
    }
};

// General rank-n dense tensor
struct Tensor {
    vector<MKL_INT> shape;
    vector<double> data;
    Tensor(MKL_INT m, MKL_INT k, MKL_INT n) : shape{m, k, n} {
        data.resize(m * k * n);
    }
    Tensor(const vector<MKL_INT> &shape) : shape(shape) {
        data.resize(
            accumulate(shape.begin(), shape.end(), 1, multiplies<double>()));
    }
    size_t size() const { return data.size(); }
    void clear() { memset(data.data(), 0, size() * sizeof(double)); }
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
            memcpy(data.data() + i * nr, data.data() + i * shape[1],
                   nr * sizeof(double));
        data.resize(shape[0] * nr);
        shape[1] = nr;
    }
    MatrixRef ref() {
        if (shape.size() == 3 && shape[1] == 1)
            return MatrixRef(data.data(), shape[0], shape[2]);
        else if (shape.size() == 2)
            return MatrixRef(data.data(), shape[0], shape[1]);
        else if (shape.size() == 1)
            return MatrixRef(data.data(), shape[0], 1);
        else {
            assert(false);
            return MatrixRef(data.data(), 0, 1);
        }
    }
    double &operator()(initializer_list<MKL_INT> idx) {
        size_t i = 0;
        int k = 0;
        for (auto &ix : idx) {
            if (k != 0)
                i *= shape[k - 1];
            i += ix, k++;
        }
        return data[i];
    }
    friend ostream &operator<<(ostream &os, const Tensor &ts) {
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

} // namespace block2
