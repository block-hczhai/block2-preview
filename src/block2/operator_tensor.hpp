
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

#include "sparse_matrix.hpp"
#include "symbolic.hpp"
#include <map>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

// Matrix/Vector of symbols representation a tensor in MPO or contracted MPO
template <typename S> struct OperatorTensor {
    // Symbolic tensor for left blocking and right blocking
    // For normal MPO, lmat and rmat are the same
    shared_ptr<Symbolic<S>> lmat, rmat;
    // SparseMatrix representation of symbols
    map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
        ops;
    OperatorTensor() : lmat(nullptr), rmat(nullptr) {}
    void reallocate(bool clean) {
        for (auto &p : ops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
    }
    void deallocate() {
        // need to check order in parallel mode
        map<double *, shared_ptr<SparseMatrix<S>>> mp;
        for (auto it = ops.cbegin(); it != ops.cend(); it++)
            mp[it->second->data] = it->second;
        for (auto it = mp.crbegin(); it != mp.crend(); it++)
            it->second->deallocate();
    }
    shared_ptr<OperatorTensor> copy() const {
        shared_ptr<OperatorTensor> r = make_shared<OperatorTensor>();
        r->lmat = lmat, r->rmat = rmat;
        r->ops = ops;
        return r;
    }
    shared_ptr<OperatorTensor> deep_copy() const {
        shared_ptr<OperatorTensor> r = make_shared<OperatorTensor>();
        r->lmat = lmat, r->rmat = rmat;
        for (auto &p : ops) {
            shared_ptr<SparseMatrix<S>> mat = make_shared<SparseMatrix<S>>();
            if (p.second->total_memory == 0)
                mat->info = p.second->info;
            else {
                mat->allocate(p.second->info);
                mat->copy_data_from(p.second);
            }
            mat->factor = p.second->factor;
            r->ops[p.first] = mat;
        }
        return r;
    }
};

// Delayed contraction of left and right block MPO tensors
template <typename S> struct DelayedOperatorTensor {
    // Symbol of super block operator(s)
    vector<shared_ptr<OpExpr<S>>> ops;
    // Symbolic expression of super block operator(s)
    shared_ptr<Symbolic<S>> mat;
    // SparseMatrix representation of symbols from left and right block
    map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
        lops, rops;
    DelayedOperatorTensor() {}
    void reallocate(bool clean) {
        for (auto &p : lops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
        for (auto &p : rops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
    }
    void deallocate() {
        // need to check order in parallel mode
        map<double *, shared_ptr<SparseMatrix<S>>> mp;
        for (auto it = rops.cbegin(); it != rops.cend(); it++)
            mp[it->second->data] = it->second;
        for (auto it = mp.crbegin(); it != mp.crend(); it++)
            it->second->deallocate();
        mp.clear();
        for (auto it = lops.cbegin(); it != lops.cend(); it++)
            mp[it->second->data] = it->second;
        for (auto it = mp.crbegin(); it != mp.crend(); it++)
            it->second->deallocate();
    }
};

} // namespace block2
