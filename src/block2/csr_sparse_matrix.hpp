
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
    vector<shared_ptr<CSRMatrixRef>> csr_data;
    CSRSparseMatrix(const shared_ptr<Allocator<double>> &alloc = nullptr)
        : SparseMatrix<S>(alloc) {}
    const SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::CSR;
    }
    void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                  double *ptr = 0) override {
        this->info = info;
        assert(ptr == 0);
        csr_data.resize(info->n);
        for (int i = 0; i < info->n; i++)
            csr_data[i] = make_shared<CSRMatrixRef>(info->n_states_bra[i],
                                                    info->n_states_ket[i]);
    }
    void deallocate() override {
        for (int i = this->info->n - 1; i >= 0; i--)
            csr_data[i]->deallocate();
        csr_data.clear();
        this->total_memory = 0;
        this->data = nullptr;
    }
    CSRMatrixRef &operator[](S q) const {
        return (*this)[this->info->find_state(q)];
    }
    CSRMatrixRef &operator[](int idx) {
        assert(idx != -1);
        return *csr_data[idx];
    }
    void copy_data_from(const shared_ptr<SparseMatrix<S>> &other) override {
        assert(other->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S>> cother =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(other);
        assert(this->info->n == other->info->n);
        deallocate();
        csr_data.resize(this->info->n);
        for (int i = 0; i < this->info->n; i++)
            csr_data[i] =
                make_shared<CSRMatrixRef>(cother->csr_data[i]->deep_copy());
    }
    void
    selective_copy_from(const shared_ptr<SparseMatrix<S>> &other) override {
        assert(other->get_type() == SparseMatrixTypes::CSR);
        shared_ptr<CSRSparseMatrix<S>> cother =
            dynamic_pointer_cast<CSRSparseMatrix<S>>(other);
        deallocate();
        csr_data.resize(this->info->n);
        for (int i = 0, k; i < other->info->n; i++)
            if ((k = this->info->find_state(other->info->quanta[i])) != -1)
                csr_data[k] =
                    make_shared<CSRMatrixRef>(cother->csr_data[i]->deep_copy());
    }
    void clear() override {
        deallocate();
        allocate(this->info);
    }
    double norm() const override {
        double r = 0;
        for (int i = 0; i < this->info->n; i++) {
            double rn = CSRMatrixFunctions::norm(*csr_data[i]);
            r += rn * rn;
        }
        return sqrt(r);
    }
    // ratio of zero elements to total size
    double sparsity() const override {
        size_t nnz = 0, size = 0;
        for (int i = 0; i < this->info->n; i++)
            nnz += csr_data[i]->nnz, size += csr_data[i]->size();
        return 1.0 - (double)nnz / size;
    }
    // this will allocate csr matrix
    void from_dense(const shared_ptr<SparseMatrix<S>> &mat) {
        assert(csr_data.size() == 0);
        this->info = mat->info;
        csr_data.resize(this->info->n);
        for (int i = 0; i < this->info->n; i++) {
            csr_data[i] = make_shared<CSRMatrixRef>();
            csr_data[i]->from_dense((*mat)[i]);
        }
    }
    // this will not allocate sparse matrix
    void to_dense(const shared_ptr<SparseMatrix<S>> &mat) {
        assert(mat->data != nullptr);
        assert(mat->info == this->info);
        assert(csr_data.size() == this->info->n);
        for (int i = 0; i < this->info->n; i++)
            csr_data[i]->to_dense((*mat)[i]);
    }
};

} // namespace block2
