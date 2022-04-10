
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
#include "complex_matrix_functions.hpp"
#include "matrix.hpp"
#include "matrix_functions.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

#define TINY (1E-20)

using namespace std;

namespace block2 {

// Compressed-Sparse-Row matrix
template <typename FL> struct GCSRMatrix {
    typedef typename GMatrix<FL>::FP FP;
    static const int cpx_sz = sizeof(FL) / sizeof(FP);
    shared_ptr<Allocator<FP>> alloc = nullptr;
    MKL_INT m, n, nnz; // m is rows, n is cols, nnz is number of nonzeros
    FL *data;
    MKL_INT *rows, *cols;
    GCSRMatrix()
        : m(0), n(0), nnz(0), data(nullptr), rows(nullptr), cols(nullptr) {}
    GCSRMatrix(MKL_INT m, MKL_INT n, MKL_INT nnz = 0) : m(m), n(n), nnz(nnz) {
        alloc = make_shared<VectorAllocator<FP>>();
        allocate();
        if (nnz != size())
            memset(rows, 0, (m + 1) * sizeof(MKL_INT));
    }
    GCSRMatrix(MKL_INT m, MKL_INT n, MKL_INT nnz, FL *data, MKL_INT *rows,
               MKL_INT *cols)
        : m(m), n(n), nnz(nnz), data(data), rows(rows), cols(cols) {}
    size_t size() const { return (size_t)m * n; }
    MKL_INT memory_size() const {
        if (sizeof(MKL_INT) + sizeof(MKL_INT) == sizeof(FL))
            return nnz == m * n ? nnz : nnz + ((nnz + m + 2) >> 1);
        else if (sizeof(MKL_INT) == sizeof(FL))
            return nnz == m * n ? nnz : nnz + nnz + m + 1;
        else if ((sizeof(MKL_INT) << 2) == sizeof(FL))
            return nnz == m * n ? nnz : nnz + ((nnz + m + 4) >> 1);
        else {
            assert(false);
            return 0;
        }
    }
    void load_data(istream &ifs) {
        ifs.read((char *)&m, sizeof(m));
        ifs.read((char *)&n, sizeof(n));
        ifs.read((char *)&nnz, sizeof(nnz));
        if (alloc == nullptr)
            alloc = make_shared<VectorAllocator<FP>>();
        allocate();
        ifs.read((char *)data, sizeof(FL) * nnz);
        if (nnz != size()) {
            ifs.read((char *)cols, sizeof(MKL_INT) * nnz);
            ifs.read((char *)rows, sizeof(MKL_INT) * (m + 1));
        } else
            cols = rows = nullptr;
    }
    void save_data(ostream &ofs) const {
        ofs.write((char *)&m, sizeof(m));
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)&nnz, sizeof(nnz));
        ofs.write((char *)data, sizeof(FL) * nnz);
        if (nnz != size()) {
            ofs.write((char *)cols, sizeof(MKL_INT) * nnz);
            ofs.write((char *)rows, sizeof(MKL_INT) * m);
            ofs.write((char *)&nnz, sizeof(MKL_INT));
        }
    }
    // conj transpose
    GCSRMatrix
    transpose(const shared_ptr<Allocator<FP>> &alloc = nullptr) const {
        GCSRMatrix r(n, m, nnz, nullptr, nullptr, nullptr);
        r.alloc = alloc;
        r.allocate();
        if (r.nnz != r.size()) {
            memset(r.rows, 0, sizeof(MKL_INT) * (n + 1));
            for (MKL_INT ia = 0; ia < nnz; ia++)
                r.rows[cols[ia] + 1]++;
            for (MKL_INT ia = 0; ia < n; ia++)
                r.rows[ia + 1] += r.rows[ia];
            for (MKL_INT ia = 0; ia < m; ia++) {
                MKL_INT jap = rows[ia], jar = ia == m - 1 ? nnz : rows[ia + 1];
                for (MKL_INT ja = jap; ja < jar; ja++) {
                    r.cols[r.rows[cols[ja]]] = ia;
                    r.data[r.rows[cols[ja]]] = xconj<FL>(data[ja]);
                    r.rows[cols[ja]]++;
                }
            }
            for (MKL_INT ia = n - 1; ia >= 0; ia--)
                r.rows[ia] -= r.rows[ia] - (ia == 0 ? 0 : r.rows[ia - 1]);
        } else
            GMatrixFunctions<FL>::iadd(GMatrix<FL>(r.data, n, m),
                                       GMatrix<FL>(data, m, n), 1.0, true, 0.0);
        return r;
    }
    FP sparsity() const { return 1.0 - (FP)nnz / (m * n); }
    void allocate(FL *ptr = nullptr) {
        if (ptr == nullptr) {
            if (alloc == nullptr)
                alloc = dalloc_<FP>();
            data = (FL *)alloc->allocate(memory_size() * cpx_sz);
        } else
            data = ptr;
        if (nnz == size())
            cols = rows = nullptr;
        else {
            cols = (MKL_INT *)(data + nnz);
            rows = (MKL_INT *)(cols + nnz);
        }
    }
    void deallocate() {
        if (alloc == nullptr)
            data = nullptr;
        else {
            assert(data != nullptr);
            alloc->deallocate(data, memory_size() * cpx_sz);
            alloc = nullptr;
            data = nullptr;
            cols = rows = nullptr;
        }
    }
    GMatrix<FL> dense_ref() const {
        assert(nnz == size());
        return GMatrix<FL>(data, m, n);
    }
    GCSRMatrix deep_copy() const {
        GCSRMatrix r(m, n, nnz, nullptr, nullptr, nullptr);
        r.alloc = make_shared<VectorAllocator<FP>>();
        r.allocate();
        memcpy(r.data, data, nnz * sizeof(FL));
        if (nnz != size()) {
            memcpy(r.cols, cols, nnz * sizeof(MKL_INT));
            memcpy(r.rows, rows, m * sizeof(MKL_INT));
            r.rows[m] = nnz;
        }
        return r;
    }
    friend ostream &operator<<(ostream &os, const GCSRMatrix &mat) {
        if (mat.nnz == mat.size())
            os << "CSR-DENSE-" << mat.dense_ref();
        else {
            os << "CSR-MAT ( " << mat.m << "x" << mat.n
               << " ) NNZ = " << mat.nnz << endl;
            for (MKL_INT i = 0; i < mat.m; i++)
                if ((i == mat.m - 1 ? mat.nnz : mat.rows[i + 1]) >
                    mat.rows[i]) {
                    os << "ROW [ " << setw(5) << i << " ] = ";
                    for (MKL_INT j = mat.rows[i];
                         j < (i == mat.m - 1 ? mat.nnz : mat.rows[i + 1]); j++)
                        os << setw(5) << mat.cols[j] << " : " << setw(20)
                           << setprecision(14) << mat.data[j] << ", ";
                    os << endl;
                }
        }
        return os;
    }
    // copy
    void from_dense(const GMatrix<FL> &mat, FP cutoff = TINY) {
        alloc = make_shared<VectorAllocator<FP>>();
        m = mat.m, n = mat.n, nnz = 0;
        for (MKL_INT i = 0; i < mat.size(); i++)
            nnz += abs(mat.data[i]) > cutoff;
        allocate();
        if (nnz == size())
            memcpy(data, mat.data, sizeof(FL) * size());
        else {
            for (MKL_INT i = 0, k = 0; i < m; i++) {
                rows[i] = k;
                for (MKL_INT j = 0; j < n; j++)
                    if (abs(mat(i, j)) > cutoff)
                        cols[k] = j, data[k] = mat(i, j), k++;
            }
            rows[m] = nnz;
        }
    }
    void to_dense(GMatrix<FL> mat) const {
        if (nnz == size())
            memcpy(mat.data, data, sizeof(FL) * size());
        else {
            mat.clear();
            for (MKL_INT i = 0; i < m; i++) {
                MKL_INT rows_end = i == m - 1 ? nnz : rows[i + 1];
                for (MKL_INT j = rows[i]; j < rows_end; j++)
                    mat(i, cols[j]) = data[j];
            }
        }
    }
    void diag(GMatrix<FL> x) const {
        assert(m == n);
        if (nnz == size()) {
            const MKL_INT inc = 1, ind = n + 1;
            xcopy<FL>(&m, data, &ind, x.data, &inc);
        } else {
            x.clear();
            if (nnz != 0)
                for (MKL_INT i = 0; i < m; i++) {
                    MKL_INT rows_end = i == m - 1 ? nnz : rows[i + 1];
                    MKL_INT ic = (MKL_INT)(lower_bound(cols + rows[i],
                                                       cols + rows_end, i) -
                                           cols);
                    if (ic != rows_end && cols[ic] == i)
                        x.data[i] = data[ic];
                }
        }
    }
    FL trace() const {
        assert(m == n);
        if (nnz == size())
            return dense_ref().trace();
        else {
            FL r = 0;
            for (MKL_INT i = 0; i < m; i++) {
                MKL_INT rows_end = i == m - 1 ? nnz : rows[i + 1];
                MKL_INT ic =
                    (MKL_INT)(lower_bound(cols + rows[i], cols + rows_end, i) -
                              cols);
                if (ic != rows_end && cols[ic] == i)
                    r += data[ic];
            }
            return r;
        }
    }
};

} // namespace block2
