
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
struct CSRMatrixRef {
    shared_ptr<Allocator<double>> alloc = nullptr;
    MKL_INT m, n, nnz; // m is rows, n is cols, nnz is number of nonzeros
    double *data;
    MKL_INT *rows, *cols;
    CSRMatrixRef()
        : m(0), n(0), nnz(0), data(nullptr), rows(nullptr), cols(nullptr) {}
    CSRMatrixRef(MKL_INT m, MKL_INT n, MKL_INT nnz = 0) : m(m), n(n), nnz(nnz) {
        alloc = make_shared<VectorAllocator<double>>();
        allocate();
        if (nnz != size())
            memset(rows, 0, (m + 1) * sizeof(MKL_INT));
    }
    CSRMatrixRef(MKL_INT m, MKL_INT n, MKL_INT nnz, double *data, MKL_INT *rows,
                 MKL_INT *cols)
        : m(m), n(n), nnz(nnz), data(data), rows(rows), cols(cols) {}
    size_t size() const { return (size_t)m * n; }
    MKL_INT memory_size() const {
        if (sizeof(MKL_INT) == 4)
            return nnz == m * n ? nnz : nnz + ((nnz + m + 2) >> 1);
        else
            return nnz == m * n ? nnz : nnz + nnz + m + 1;
    }
    void load_data(istream &ifs) {
        ifs.read((char *)&m, sizeof(m));
        ifs.read((char *)&n, sizeof(n));
        ifs.read((char *)&nnz, sizeof(nnz));
        if (alloc == nullptr)
            alloc = make_shared<VectorAllocator<double>>();
        allocate();
        ifs.read((char *)data, sizeof(double) * nnz);
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
        ofs.write((char *)data, sizeof(double) * nnz);
        if (nnz != size()) {
            ofs.write((char *)cols, sizeof(MKL_INT) * nnz);
            ofs.write((char *)rows, sizeof(MKL_INT) * m);
            ofs.write((char *)&nnz, sizeof(MKL_INT));
        }
    }
    CSRMatrixRef
    transpose(const shared_ptr<Allocator<double>> &alloc = nullptr) const {
        CSRMatrixRef r(n, m, nnz, nullptr, nullptr, nullptr);
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
                    r.data[r.rows[cols[ja]]] = data[ja];
                    r.rows[cols[ja]]++;
                }
            }
            for (MKL_INT ia = n - 1; ia >= 0; ia--)
                r.rows[ia] -= r.rows[ia] - (ia == 0 ? 0 : r.rows[ia - 1]);
        } else
            for (MKL_INT i = 0, inc = 1; i < n; i++)
                dcopy(&m, data + i, &n, r.data + i * m, &inc);
        return r;
    }
    double sparsity() const { return 1.0 - (double)nnz / (m * n); }
    void allocate(double *ptr = nullptr) {
        if (ptr == nullptr) {
            if (alloc == nullptr)
                alloc = dalloc;
            data = alloc->allocate(memory_size());
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
            alloc->deallocate(data, memory_size());
            alloc = nullptr;
            data = nullptr;
            cols = rows = nullptr;
        }
    }
    MatrixRef dense_ref() const {
        assert(nnz == size());
        return MatrixRef(data, m, n);
    }
    CSRMatrixRef deep_copy() const {
        CSRMatrixRef r(m, n, nnz, nullptr, nullptr, nullptr);
        r.alloc = make_shared<VectorAllocator<double>>();
        r.allocate();
        memcpy(r.data, data, nnz * sizeof(double));
        if (nnz != size()) {
            memcpy(r.cols, cols, nnz * sizeof(MKL_INT));
            memcpy(r.rows, rows, m * sizeof(MKL_INT));
            r.rows[m] = nnz;
        }
        return r;
    }
    friend ostream &operator<<(ostream &os, const CSRMatrixRef &mat) {
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
    void from_dense(const MatrixRef &mat, double cutoff = TINY) {
        alloc = make_shared<VectorAllocator<double>>();
        m = mat.m, n = mat.n, nnz = 0;
        for (MKL_INT i = 0; i < mat.size(); i++)
            nnz += abs(mat.data[i]) > cutoff;
        allocate();
        if (nnz == size())
            memcpy(data, mat.data, sizeof(double) * size());
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
    void to_dense(MatrixRef mat) const {
        if (nnz == size())
            memcpy(mat.data, data, sizeof(double) * size());
        else {
            mat.clear();
            for (MKL_INT i = 0; i < m; i++) {
                MKL_INT rows_end = i == m - 1 ? nnz : rows[i + 1];
                for (MKL_INT j = rows[i]; j < rows_end; j++)
                    mat(i, cols[j]) = data[j];
            }
        }
    }
    void diag(MatrixRef x) const {
        assert(m == n);
        if (nnz == size()) {
            const MKL_INT inc = 1, ind = n + 1;
            dcopy(&m, data, &ind, x.data, &inc);
        } else {
            x.clear();
            if (nnz != 0)
                for (MKL_INT i = 0; i < m; i++) {
                    MKL_INT rows_end = i == m - 1 ? nnz : rows[i + 1];
                    MKL_INT ic =
                        (MKL_INT)(lower_bound(cols + rows[i], cols + rows_end, i) - cols);
                    if (ic != rows_end && cols[ic] == i)
                        x.data[i] = data[ic];
                }
        }
    }
    double trace() const {
        assert(m == n);
        if (nnz == size())
            return dense_ref().trace();
        else {
            double r = 0;
            for (MKL_INT i = 0; i < m; i++) {
                MKL_INT rows_end = i == m - 1 ? nnz : rows[i + 1];
                MKL_INT ic =
                    (MKL_INT)(lower_bound(cols + rows[i], cols + rows_end, i) - cols);
                if (ic != rows_end && cols[ic] == i)
                    r += data[ic];
            }
            return r;
        }
    }
};

} // namespace block2
