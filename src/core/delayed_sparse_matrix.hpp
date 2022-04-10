
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

#include "csr_sparse_matrix.hpp"
#include "expr.hpp"
#include "sparse_matrix.hpp"

using namespace std;

namespace block2 {

// A delayed wrapper for normal/CSR SparseMatrix
template <typename S, typename FL, typename T = void>
struct DelayedSparseMatrix : DelayedSparseMatrix<S, FL> {
    shared_ptr<T> mat;
    DelayedSparseMatrix(const shared_ptr<T> &mat)
        : DelayedSparseMatrix<S, FL>(), mat(mat) {}
    void deallocate() override { mat->deallocate(); }
    shared_ptr<SparseMatrix<S, FL>> build() override {
        shared_ptr<T> rmat = make_shared<T>();
        rmat->allocate(mat->info);
        rmat->copy_data_from(mat, true);
        rmat->factor = mat->factor;
        return rmat;
    }
    shared_ptr<DelayedSparseMatrix<S, FL>> copy() override {
        return make_shared<DelayedSparseMatrix>(this->mat);
    }
    shared_ptr<DelayedSparseMatrix<S, FL>>
    selective_copy(const shared_ptr<SparseMatrixInfo<S>> &info) override {
        shared_ptr<T> new_mat = make_shared<T>();
        new_mat->allocate(info);
        new_mat->selective_copy_from(mat, true);
        new_mat->factor = mat->factor;
        return make_shared<DelayedSparseMatrix>(new_mat);
    }
};

// Block-sparse Matrix created on-the-fly (base class)
// Representing sparse operator
template <typename S, typename FL>
struct DelayedSparseMatrix<S, FL> : SparseMatrix<S, FL> {
    DelayedSparseMatrix<S, FL>() : SparseMatrix<S, FL>() {}
    virtual ~DelayedSparseMatrix<S, FL>() = default;
    SparseMatrixTypes get_type() const override {
        return SparseMatrixTypes::Delayed;
    }
    void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                  FL *ptr = 0) override {
        assert(false);
    }
    virtual void deallocate() override {}
    // return a SparseMatrix or CSRSparseMatrix
    // the returned matrix will be appropriately deallocated after using
    virtual shared_ptr<SparseMatrix<S, FL>> build() { return nullptr; }
    // shallow copy
    virtual shared_ptr<DelayedSparseMatrix<S, FL>> copy() { return nullptr; }
    // selective shallow copy
    virtual shared_ptr<DelayedSparseMatrix<S, FL>>
    selective_copy(const shared_ptr<SparseMatrixInfo<S>> &info) {
        return nullptr;
    }
};

// Delayed site operator
template <typename S, typename FL>
struct DelayedSparseMatrix<S, FL, OpExpr<S>> : DelayedSparseMatrix<S, FL> {
    uint16_t m;
    shared_ptr<OpExpr<S>> op;
    DelayedSparseMatrix(uint16_t m, const shared_ptr<OpExpr<S>> &op,
                        const shared_ptr<SparseMatrixInfo<S>> &info = nullptr)
        : DelayedSparseMatrix<S, FL>(), m(m), op(op) {
        this->info = info;
    }
    void load_data(istream &ifs, bool pointer_only = false) override {
        ifs.read((char *)&m, sizeof(m));
        op = load_expr<S, FL>(ifs);
    }
    void save_data(ostream &ofs, bool pointer_only = false) const override {
        ofs.write((char *)&m, sizeof(m));
        assert(op != nullptr);
        save_expr<S>(op, ofs);
    }
    shared_ptr<SparseMatrix<S, FL>> build() override {
        assert(false);
        return nullptr;
    }
    typename GMatrix<FL>::FP norm() const override { return 1.0; }
    shared_ptr<DelayedSparseMatrix<S, FL>> copy() override {
        return make_shared<DelayedSparseMatrix>(*this);
    }
    shared_ptr<DelayedSparseMatrix<S, FL>>
    selective_copy(const shared_ptr<SparseMatrixInfo<S>> &info) override {
        shared_ptr<DelayedSparseMatrix> mat =
            make_shared<DelayedSparseMatrix>(*this);
        mat->info = info;
        return mat;
    }
};

} // namespace block2
