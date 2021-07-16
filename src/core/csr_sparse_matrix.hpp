
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

#include "csr_matrix.hpp"
#include "csr_matrix_functions.hpp"
#include "sparse_matrix.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#define TINY (1E-20)

using namespace std;

namespace block2 {

// CSR Block-sparse Matrix
// Representing sparse operator
template <typename S> struct CSRSparseMatrix : SparseMatrix<S> {
    using SparseMatrix<S>::alloc;
    using SparseMatrix<S>::info;
    using SparseMatrix<S>::data;
    using SparseMatrix<S>::total_memory;
    vector<shared_ptr<CSRMatrixRef>> csr_data;
    CSRSparseMatrix(const shared_ptr<Allocator<double>> &alloc = nullptr)
        : SparseMatrix<S>(alloc) {}
    SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::CSR;
    }
    void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                  double *ptr = 0) override {
        this->info = info;
        assert(ptr == 0);
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++)
            csr_data[i] = make_shared<CSRMatrixRef>(
                (MKL_INT)info->n_states_bra[i], (MKL_INT)info->n_states_ket[i]);
    }
    // initialize csr_data without allocating memory
    void initialize(const shared_ptr<SparseMatrixInfo<S>> &info) {
        this->info = info;
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++) {
            csr_data[i] = make_shared<CSRMatrixRef>(
                (MKL_INT)info->n_states_bra[i], (MKL_INT)info->n_states_ket[i],
                0, nullptr, nullptr, nullptr);
            csr_data[i]->alloc = make_shared<VectorAllocator<double>>();
        }
    }
    void deallocate() override {
        if (csr_data.size() != 0) {
            for (int i = info->n - 1; i >= 0; i--)
                csr_data[i]->deallocate();
            csr_data.clear();
        }
        if (alloc != nullptr) {
            if (total_memory == 0)
                assert(data == nullptr);
            else
                alloc->deallocate(data, total_memory);
            alloc = nullptr;
        }
        total_memory = 0;
        data = nullptr;
    }
    void load_data(istream &ifs, bool pointer_only = false) override {
        assert(pointer_only == false);
        SparseMatrix<S>::load_data(ifs);
        csr_data.resize(info->n);
        if (total_memory != 0) {
            for (int i = 0; i < info->n; i++) {
                MatrixRef dmat = SparseMatrix<S>::operator[](i);
                csr_data[i] = make_shared<CSRMatrixRef>(
                    dmat.m, dmat.n, (MKL_INT)dmat.size(), dmat.data, nullptr, nullptr);
            }
        } else {
            for (int i = 0; i < info->n; i++) {
                csr_data[i] = make_shared<CSRMatrixRef>();
                csr_data[i]->load_data(ifs);
            }
        }
    }
    void save_data(ostream &ofs, bool pointer_only = false) const override {
        assert(pointer_only == false);
        SparseMatrix<S>::save_data(ofs);
        if (total_memory == 0) {
            assert((int)csr_data.size() == info->n);
            for (int i = 0; i < info->n; i++)
                csr_data[i]->save_data(ofs);
        }
    }
    CSRMatrixRef &operator[](S q) const { return (*this)[info->find_state(q)]; }
    CSRMatrixRef &operator[](int idx) const {
        assert(idx != -1 && idx < csr_data.size());
        return *csr_data[idx];
    }
    void copy_data_from(const shared_ptr<SparseMatrix<S>> &other,
                        bool ref = false) override {
        assert(other->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S>> cother =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(other);
        assert(info->n == other->info->n);
        deallocate();
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++)
            if (!ref)
                csr_data[i] =
                    make_shared<CSRMatrixRef>(cother->csr_data[i]->deep_copy());
            else {
                shared_ptr<CSRMatrixRef> mat = cother->csr_data[i];
                csr_data[i] = make_shared<CSRMatrixRef>(
                    mat->m, mat->n, mat->nnz, mat->data, mat->rows, mat->cols);
            }
    }
    void selective_copy_from(const shared_ptr<SparseMatrix<S>> &other,
                             bool ref = false) override {
        assert(other->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S>> cother =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(other);
        deallocate();
        csr_data.resize(info->n);
        for (int i = 0, k; i < other->info->n; i++)
            if ((k = info->find_state(other->info->quanta[i])) != -1) {
                if (!ref)
                    csr_data[k] = make_shared<CSRMatrixRef>(
                        cother->csr_data[i]->deep_copy());
                else {
                    shared_ptr<CSRMatrixRef> mat = cother->csr_data[i];
                    csr_data[k] = make_shared<CSRMatrixRef>(
                        mat->m, mat->n, mat->nnz, mat->data, mat->rows,
                        mat->cols);
                }
            }
    }
    void clear() override {
        deallocate();
        allocate(info);
    }
    double norm() const override {
        double r = 0;
        for (int i = 0; i < info->n; i++) {
            double rn = CSRMatrixFunctions::norm(*csr_data[i]);
            r += rn * rn;
        }
        return sqrt(r);
    }
    // ratio of zero elements to total size
    double sparsity() const override {
        size_t nnz = 0, size = 0;
        for (int i = 0; i < info->n; i++)
            nnz += csr_data[i]->nnz, size += csr_data[i]->size();
        return 1.0 - (double)nnz / size;
    }
    // set "csr" sparse matrix as a wrapper for dense sparse mat
    // this will ref memory allocated in mat
    // mat should not be deallocated afterwards
    void wrap_dense(const shared_ptr<SparseMatrix<S>> &mat) {
        alloc = mat->alloc;
        total_memory = mat->total_memory;
        data = mat->data;
        info = mat->info;
        assert(csr_data.size() == 0);
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++) {
            MatrixRef dmat = (*mat)[i];
            csr_data[i] = make_shared<CSRMatrixRef>(
                dmat.m, dmat.n, (MKL_INT)dmat.size(), dmat.data, nullptr, nullptr);
        }
    }
    // construct real csr sparse matrix from dense sparse mat
    // this will allocate memory for csr matrix
    // mat should be deallocated afterwards
    void from_dense(const shared_ptr<SparseMatrix<S>> &mat) {
        assert(csr_data.size() == 0);
        info = mat->info;
        csr_data.resize(info->n);
        if (mat->get_type() == SparseMatrixTypes::Normal) {
            for (int i = 0; i < info->n; i++) {
                csr_data[i] = make_shared<CSRMatrixRef>();
                csr_data[i]->from_dense((*mat)[i]);
            }
        } else {
            shared_ptr<CSRSparseMatrix<S>> smat =
                dynamic_pointer_cast<CSRSparseMatrix<S>>(mat);
            for (int i = 0; i < info->n; i++) {
                csr_data[i] = make_shared<CSRMatrixRef>();
                csr_data[i]->from_dense((*smat)[i].dense_ref());
            }
        }
    }
    // this will not allocate dense matrix
    // mat must be pre-allocated
    void to_dense(const shared_ptr<SparseMatrix<S>> &mat) {
        assert(mat->data != nullptr);
        assert(mat->info == info);
        assert(csr_data.size() == info->n);
        for (int i = 0; i < info->n; i++)
            csr_data[i]->to_dense((*mat)[i]);
    }
};

} // namespace block2
