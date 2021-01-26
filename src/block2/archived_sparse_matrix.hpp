
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

#include "csr_sparse_matrix.hpp"
#include "sparse_matrix.hpp"

using namespace std;

namespace block2 {

// Block-sparse Matrix associated with disk storage
// Representing sparse operator
template <typename S> struct ArchivedSparseMatrix : SparseMatrix<S> {
    using SparseMatrix<S>::alloc;
    using SparseMatrix<S>::info;
    using SparseMatrix<S>::data;
    using SparseMatrix<S>::factor;
    using SparseMatrix<S>::total_memory;
    using SparseMatrix<S>::allocate;
    string filename;
    int64_t offset = 0;
    SparseMatrixTypes sparse_type;
    ArchivedSparseMatrix(const string &filename, int64_t offset,
                         const shared_ptr<Allocator<double>> &alloc = nullptr)
        : SparseMatrix<S>(alloc), filename(filename), offset(offset) {}
    SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::Archived;
    }
    void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                  double *ptr = 0) override {
        assert(false);
    }
    void deallocate() override {}
    shared_ptr<SparseMatrix<S>> load_archive() {
        if (alloc == nullptr)
            alloc = dalloc;
        if (sparse_type == SparseMatrixTypes::Normal) {
            shared_ptr<SparseMatrix<S>> mat =
                make_shared<SparseMatrix<S>>(alloc);
            mat->info = info;
            mat->total_memory = info->get_total_memory();
            total_memory = mat->total_memory;
            mat->factor = factor;
            if (total_memory != 0) {
                mat->data = alloc->allocate(mat->total_memory);
                ifstream ifs(filename.c_str(), ios::binary);
                ifs.seekg(sizeof(double) * offset);
                ifs.read((char *)mat->data, sizeof(double) * mat->total_memory);
                ifs.close();
            } else
                mat->data = nullptr;
            return mat;
        } else if (sparse_type == SparseMatrixTypes::CSR) {
            shared_ptr<CSRSparseMatrix<S>> mat =
                make_shared<CSRSparseMatrix<S>>(nullptr);
            mat->info = info;
            mat->csr_data.resize(info->n);
            mat->factor = factor;
            mat->total_memory = 0;
            if (info->n != 0) {
                ifstream ifs(filename.c_str(), ios::binary);
                ifs.seekg(sizeof(double) * offset);
                for (int i = 0; i < info->n; i++) {
                    mat->csr_data[i] = make_shared<CSRMatrixRef>();
                    mat->csr_data[i]->load_data(ifs);
                }
                ifs.close();
            }
            return mat;
        } else
            throw runtime_error("Unknown SparseType");
    }
    void save_archive(const shared_ptr<SparseMatrix<S>> &mat) {
        sparse_type = mat->get_type();
        alloc = mat->alloc;
        info = mat->info;
        factor = mat->factor;
        total_memory = mat->total_memory;
        if (sparse_type == SparseMatrixTypes::Normal) {
            sparse_type = SparseMatrixTypes::Normal;
            if (total_memory != 0) {
                ofstream ofs(filename.c_str(),
                             ios::binary | ios::out | ios::app);
                ofs.close();
                ofs.open(filename.c_str(), ios::binary | ios::in);
                ofs.seekp(sizeof(double) * offset);
                ofs.write((char *)mat->data, sizeof(double) * total_memory);
                ofs.close();
            }
        } else if (sparse_type == SparseMatrixTypes::CSR) {
            alloc = nullptr;
            if (info->n != 0) {
                shared_ptr<CSRSparseMatrix<S>> smat =
                    dynamic_pointer_cast<CSRSparseMatrix<S>>(mat);
                ofstream ofs(filename.c_str(),
                             ios::binary | ios::out | ios::app);
                ofs.close();
                ofs.open(filename.c_str(), ios::binary | ios::in);
                ofs.seekp(sizeof(double) * offset);
                for (int i = 0; i < info->n; i++)
                    smat->csr_data[i]->save_data(ofs);
                total_memory = ((size_t)ofs.tellp() - sizeof(double) * offset +
                                sizeof(double) - 1) /
                               sizeof(double);
                assert((size_t)ofs.tellp() <=
                       sizeof(double) * (offset + total_memory));
                ofs.close();
            } else
                total_memory = 0;
        } else
            assert(false);
    }
};

} // namespace block2
