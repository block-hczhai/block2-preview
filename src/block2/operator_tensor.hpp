
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
    void load_data(ifstream &ifs) {
        bool lr;
        ifs.read((char *)&lr, sizeof(lr));
        if (lr) {
            lmat = load_symbolic<S>(ifs);
            rmat = lmat;
        } else {
            lmat = load_symbolic<S>(ifs);
            rmat = load_symbolic<S>(ifs);
        }
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        for (int i = 0; i < sz; i++) {
            shared_ptr<OpExpr<S>> expr = load_expr<S>(ifs);
            SparseMatrixTypes tp;
            ifs.read((char *)&tp, sizeof(tp));
            assert(tp == SparseMatrixTypes::Normal ||
                   tp == SparseMatrixTypes::CSR);
            shared_ptr<SparseMatrix<S>> mat =
                tp == SparseMatrixTypes::Normal
                    ? make_shared<SparseMatrix<S>>(d_alloc)
                    : make_shared<CSRSparseMatrix<S>>(d_alloc);
            mat->info = make_shared<SparseMatrixInfo<S>>(i_alloc);
            mat->info->load_data(ifs);
            mat->load_data(ifs);
            ops[expr] = mat;
        }
    }
    void save_data(ofstream &ofs) const {
        bool lr = lmat == rmat;
        ofs.write((char *)&lr, sizeof(lr));
        if (lr)
            save_symbolic(lmat, ofs);
        else {
            save_symbolic(lmat, ofs);
            save_symbolic(rmat, ofs);
        }
        int sz = (int)ops.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (auto &op : ops) {
            save_expr(op.first, ofs);
            assert(op.second != nullptr);
            SparseMatrixTypes tp = op.second->get_type();
            ofs.write((char *)&tp, sizeof(tp));
            op.second->info->save_data(ofs);
            op.second->save_data(ofs);
        }
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
            if (p.second->get_type() == SparseMatrixTypes::Normal) {
                shared_ptr<SparseMatrix<S>> mat =
                    make_shared<SparseMatrix<S>>();
                if (p.second->total_memory == 0)
                    mat->info = p.second->info;
                else {
                    mat->allocate(p.second->info);
                    mat->copy_data_from(p.second);
                }
                mat->factor = p.second->factor;
                r->ops[p.first] = mat;
            } else if (p.second->get_type() == SparseMatrixTypes::Archived) {
                shared_ptr<ArchivedSparseMatrix<S>> pmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S>>(p.second);
                shared_ptr<ArchivedSparseMatrix<S>> mat =
                    make_shared<ArchivedSparseMatrix<S>>(
                        pmat->filename, pmat->offset, pmat->alloc);
                mat->info = pmat->info;
                mat->factor = pmat->factor;
                mat->sparse_type = pmat->sparse_type;
                r->ops[p.first] = mat;
            } else
                assert(false);
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
