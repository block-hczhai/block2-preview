
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
template <typename S, typename FL>
struct CSRSparseMatrix : SparseMatrix<S, FL> {
    using typename SparseMatrix<S, FL>::FP;
    using SparseMatrix<S, FL>::alloc;
    using SparseMatrix<S, FL>::info;
    using SparseMatrix<S, FL>::data;
    using SparseMatrix<S, FL>::total_memory;
    using SparseMatrix<S, FL>::cpx_sz;
    vector<shared_ptr<GCSRMatrix<FL>>> csr_data;
    CSRSparseMatrix(const shared_ptr<Allocator<FP>> &alloc = nullptr)
        : SparseMatrix<S, FL>(alloc) {}
    SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::CSR;
    }
    void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                  FL *ptr = 0) override {
        this->info = info;
        assert(ptr == 0);
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++)
            csr_data[i] = make_shared<GCSRMatrix<FL>>(
                (MKL_INT)info->n_states_bra[i], (MKL_INT)info->n_states_ket[i]);
    }
    // initialize csr_data without allocating memory
    void initialize(const shared_ptr<SparseMatrixInfo<S>> &info) {
        this->info = info;
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++) {
            csr_data[i] = make_shared<GCSRMatrix<FL>>(
                (MKL_INT)info->n_states_bra[i], (MKL_INT)info->n_states_ket[i],
                0, nullptr, nullptr, nullptr);
            csr_data[i]->alloc = make_shared<VectorAllocator<FP>>();
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
                alloc->deallocate(data, total_memory * cpx_sz);
            alloc = nullptr;
        }
        total_memory = 0;
        data = nullptr;
    }
    void load_data(istream &ifs, bool pointer_only = false) override {
        assert(pointer_only == false);
        SparseMatrix<S, FL>::load_data(ifs);
        csr_data.resize(info->n);
        if (total_memory != 0) {
            for (int i = 0; i < info->n; i++) {
                GMatrix<FL> dmat = SparseMatrix<S, FL>::operator[](i);
                csr_data[i] = make_shared<GCSRMatrix<FL>>(
                    dmat.m, dmat.n, (MKL_INT)dmat.size(), dmat.data, nullptr,
                    nullptr);
            }
        } else {
            for (int i = 0; i < info->n; i++) {
                csr_data[i] = make_shared<GCSRMatrix<FL>>();
                csr_data[i]->load_data(ifs);
            }
        }
    }
    void save_data(ostream &ofs, bool pointer_only = false) const override {
        assert(pointer_only == false);
        SparseMatrix<S, FL>::save_data(ofs);
        if (total_memory == 0) {
            assert((int)csr_data.size() == info->n);
            for (int i = 0; i < info->n; i++)
                csr_data[i]->save_data(ofs);
        }
    }
    GCSRMatrix<FL> &operator[](S q) const {
        return (*this)[info->find_state(q)];
    }
    GCSRMatrix<FL> &operator[](int idx) const {
        assert(idx != -1 && idx < csr_data.size());
        return *csr_data[idx];
    }
    void copy_data_from(const shared_ptr<SparseMatrix<S, FL>> &other,
                        bool ref = false) override {
        assert(other->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix> cother =
            dynamic_pointer_cast<CSRSparseMatrix>(other);
        assert(info->n == other->info->n);
        deallocate();
        csr_data.resize(info->n);
        if (ref && frame_<FP>()->minimal_memory_usage)
            alloc = cother->alloc;
        for (int i = 0; i < info->n; i++)
            if (!ref)
                csr_data[i] = make_shared<GCSRMatrix<FL>>(
                    cother->csr_data[i]->deep_copy());
            else {
                shared_ptr<GCSRMatrix<FL>> mat = cother->csr_data[i];
                csr_data[i] = make_shared<GCSRMatrix<FL>>(
                    mat->m, mat->n, mat->nnz, mat->data, mat->rows, mat->cols);
                // under this case, the site tensor will be dynamically loaded
                // the unload will not explicitly deallocate
                // so double deallocate will not happen
                // but if site tensor is persistent
                // we should set alloc to nullptr to prevent double deallcoate
                if (frame_<FP>()->minimal_memory_usage)
                    csr_data[i]->alloc = mat->alloc;
            }
    }
    void selective_copy_from(const shared_ptr<SparseMatrix<S, FL>> &other,
                             bool ref = false) override {
        assert(other->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix> cother =
            dynamic_pointer_cast<CSRSparseMatrix>(other);
        deallocate();
        csr_data.resize(info->n);
        if (ref && frame_<FP>()->minimal_memory_usage)
            alloc = cother->alloc;
        for (int i = 0, k; i < other->info->n; i++)
            if ((k = info->find_state(other->info->quanta[i])) != -1) {
                if (!ref)
                    csr_data[k] = make_shared<GCSRMatrix<FL>>(
                        cother->csr_data[i]->deep_copy());
                else {
                    shared_ptr<GCSRMatrix<FL>> mat = cother->csr_data[i];
                    csr_data[k] = make_shared<GCSRMatrix<FL>>(
                        mat->m, mat->n, mat->nnz, mat->data, mat->rows,
                        mat->cols);
                    if (frame_<FP>()->minimal_memory_usage)
                        csr_data[k]->alloc = mat->alloc;
                }
            }
    }
    void clear() override {
        deallocate();
        allocate(info);
    }
    FP norm() const override {
        FP r = 0;
        for (int i = 0; i < info->n; i++) {
            FP rn = GCSRMatrixFunctions<FL>::norm(*csr_data[i]);
            r += rn * rn;
        }
        return sqrt(r);
    }
    // ratio of zero elements to total size
    FP sparsity() const override {
        size_t nnz = 0, size = 0;
        for (int i = 0; i < info->n; i++)
            nnz += csr_data[i]->nnz, size += csr_data[i]->size();
        return 1.0 - (FP)nnz / size;
    }
    // set "csr" sparse matrix as a wrapper for dense sparse mat
    // this will ref memory allocated in mat
    // mat should not be deallocated afterwards
    void wrap_dense(const shared_ptr<SparseMatrix<S, FL>> &mat) {
        alloc = mat->alloc;
        total_memory = mat->total_memory;
        data = mat->data;
        info = mat->info;
        assert(csr_data.size() == 0);
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++) {
            GMatrix<FL> dmat = (*mat)[i];
            csr_data[i] = make_shared<GCSRMatrix<FL>>(
                dmat.m, dmat.n, (MKL_INT)dmat.size(), dmat.data, nullptr,
                nullptr);
        }
    }
    // construct real csr sparse matrix from dense sparse mat
    // this will allocate memory for csr matrix
    // mat should be deallocated afterwards
    void from_dense(const shared_ptr<SparseMatrix<S, FL>> &mat) {
        assert(csr_data.size() == 0);
        info = mat->info;
        csr_data.resize(info->n);
        if (mat->get_type() == SparseMatrixTypes::Normal) {
            for (int i = 0; i < info->n; i++) {
                csr_data[i] = make_shared<GCSRMatrix<FL>>();
                csr_data[i]->from_dense((*mat)[i]);
            }
        } else {
            shared_ptr<CSRSparseMatrix> smat =
                dynamic_pointer_cast<CSRSparseMatrix>(mat);
            for (int i = 0; i < info->n; i++) {
                csr_data[i] = make_shared<GCSRMatrix<FL>>();
                csr_data[i]->from_dense((*smat)[i].dense_ref());
            }
        }
    }
    // this will not allocate dense matrix
    // mat must be pre-allocated
    void to_dense(const shared_ptr<SparseMatrix<S, FL>> &mat) {
        assert(mat->data != nullptr);
        assert(mat->info == info);
        assert(csr_data.size() == info->n);
        for (int i = 0; i < info->n; i++)
            csr_data[i]->to_dense((*mat)[i]);
    }
};

} // namespace block2
