
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
    int m, n, nnz; // m is rows, n is cols, nnz is number of nonzeros
    double *data;
    int *rows, *cols;
    CSRMatrixRef()
        : m(0), n(0), nnz(0), data(nullptr), rows(nullptr), cols(nullptr) {}
    CSRMatrixRef(int m, int n, int nnz = 0) : m(m), n(n), nnz(nnz) {
        alloc = make_shared<VectorAllocator<double>>();
        allocate();
        if (nnz != size())
            memset(rows, 0, (m + 1) * sizeof(int));
    }
    CSRMatrixRef(int m, int n, int nnz, double *data, int *rows, int *cols)
        : m(m), n(n), nnz(nnz), data(data), rows(rows), cols(cols) {}
    size_t size() const { return (size_t)m * n; }
    int memory_size() const {
        return nnz == m * n ? nnz : nnz + ((nnz + m + 2) >> 1);
    }
    void load_data(ifstream &ifs) {
        ifs.read((char *)&m, sizeof(m));
        ifs.read((char *)&n, sizeof(n));
        ifs.read((char *)&nnz, sizeof(nnz));
        if (alloc == nullptr)
            alloc = make_shared<VectorAllocator<double>>();
        allocate();
        ifs.read((char *)data, sizeof(double) * nnz);
        if (nnz != size()) {
            ifs.read((char *)cols, sizeof(int) * nnz);
            ifs.read((char *)rows, sizeof(int) * (m + 1));
        } else
            cols = rows = nullptr;
    }
    void save_data(ofstream &ofs) const {
        ofs.write((char *)&m, sizeof(m));
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)&nnz, sizeof(nnz));
        ofs.write((char *)data, sizeof(double) * nnz);
        if (nnz != size()) {
            ofs.write((char *)cols, sizeof(int) * nnz);
            ofs.write((char *)rows, sizeof(int) * (m + 1));
        }
    }
    CSRMatrixRef
    transpose(const shared_ptr<Allocator<double>> &alloc = nullptr) const {
        CSRMatrixRef r(n, m, nnz, nullptr, nullptr, nullptr);
        r.alloc = alloc;
        r.allocate();
        if (r.nnz != r.size()) {
            memset(r.rows, 0, sizeof(int) * (n + 1));
            for (int ia = 0; ia < nnz; ia++)
                r.rows[cols[ia] + 1]++;
            for (int ia = 0; ia < n; ia++)
                r.rows[ia + 1] += r.rows[ia];
            for (int ia = 0; ia < m; ia++) {
                int jap = rows[ia], jar = ia == m - 1 ? nnz : rows[ia + 1];
                for (int ja = jap; ja < jar; ja++) {
                    r.cols[r.rows[cols[ja]]] = ia;
                    r.data[r.rows[cols[ja]]] = data[ja];
                    r.rows[cols[ja]]++;
                }
            }
            for (int ia = n - 1; ia >= 0; ia--)
                r.rows[ia] -= r.rows[ia] - (ia == 0 ? 0 : r.rows[ia - 1]);
        } else
            for (int i = 0, inc = 1; i < n; i++)
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
            cols = (int *)(data + nnz);
            rows = (int *)(cols + nnz);
        }
    }
    void deallocate() {
        if (alloc == nullptr) {
            assert(cols == nullptr && rows == nullptr);
            data = nullptr;
        } else {
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
            memcpy(r.cols, cols, nnz * sizeof(int));
            memcpy(r.rows, rows, m * sizeof(int));
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
            for (int i = 0; i < mat.m; i++)
                if ((i == mat.m - 1 ? mat.nnz : mat.rows[i + 1]) >
                    mat.rows[i]) {
                    os << "ROW [ " << setw(5) << i << " ] = ";
                    for (int j = mat.rows[i];
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
        for (int i = 0; i < mat.size(); i++)
            nnz += abs(mat.data[i]) > cutoff;
        allocate();
        if (nnz == size())
            memcpy(data, mat.data, sizeof(double) * size());
        else {
            for (int i = 0, k = 0; i < m; i++) {
                rows[i] = k;
                for (int j = 0; j < n; j++)
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
            for (int i = 0; i < m; i++) {
                int rows_end = i == m - 1 ? nnz : rows[i + 1];
                for (int j = rows[i]; j < rows_end; j++)
                    mat(i, cols[j]) = data[j];
            }
        }
    }
    void diag(MatrixRef x) const {
        assert(m == n);
        if (nnz == size()) {
            const int inc = 1, ind = n + 1;
            dcopy(&m, data, &ind, x.data, &inc);
        } else {
            x.clear();
            for (int i = 0; i < m; i++) {
                int rows_end = i == m - 1 ? nnz : rows[i + 1];
                int ic = lower_bound(cols + rows[i], cols + rows_end, i) - cols;
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
            for (int i = 0; i < m; i++) {
                int rows_end = i == m - 1 ? nnz : rows[i + 1];
                int ic = lower_bound(cols + rows[i], cols + rows_end, i) - cols;
                if (ic != rows_end && cols[ic] == i)
                    r += data[ic];
            }
            return r;
        }
    }
};

} // namespace block2
