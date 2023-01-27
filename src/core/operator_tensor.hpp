
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

#include "archived_sparse_matrix.hpp"
#include "csr_sparse_matrix.hpp"
#include "delayed_sparse_matrix.hpp"
#include "sparse_matrix.hpp"
#include "symbolic.hpp"
#include "threading.hpp"
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

enum struct OperatorTensorTypes : uint8_t { Normal, Delayed };

// Matrix/Vector of symbols representation a tensor in MPO or contracted MPO
template <typename S, typename FL> struct OperatorTensor {
    typedef typename GMatrix<FL>::FP FP;
    // Symbolic tensor for left blocking and right blocking
    // For normal MPO, lmat and rmat are the same
    shared_ptr<Symbolic<S>> lmat, rmat;
    // SparseMatrix representation of symbols
    unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>> ops;
    OperatorTensor() : lmat(nullptr), rmat(nullptr) {}
    virtual ~OperatorTensor() = default;
    virtual OperatorTensorTypes get_type() const {
        return OperatorTensorTypes::Normal;
    }
    virtual size_t get_total_memory() const {
        size_t r = 0;
        for (auto it = ops.cbegin(); it != ops.cend(); it++)
            if (it->second != nullptr && it->second->data != nullptr)
                r += it->second->total_memory;
        return r;
    }
    virtual void reallocate(bool clean) {
        for (auto &p : ops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
    }
    virtual void deallocate() {
        // need to check order in parallel mode
        vector<pair<FL *, shared_ptr<SparseMatrix<S, FL>>>> mp;
        mp.reserve(ops.size());
        for (auto it = ops.cbegin(); it != ops.cend(); it++)
            mp.emplace_back(it->second->data, it->second);
        sort(mp.begin(), mp.end(),
             [](const pair<FL *, shared_ptr<SparseMatrix<S, FL>>> &a,
                const pair<FL *, shared_ptr<SparseMatrix<S, FL>>> &b) {
                 return a.first > b.first;
             });
        for (const auto &t : mp)
            t.second->deallocate();
    }
    void load_data(istream &ifs, bool pointer_only = false,
                   bool no_ops = false) {
        uint8_t lr;
        ifs.read((char *)&lr, sizeof(lr));
        if (lr == 1) {
            lmat = load_symbolic<S, FL>(ifs);
            rmat = lmat;
        } else if (lr != 4) {
            if (lr == 0 || lr == 2)
                lmat = load_symbolic<S, FL>(ifs);
            if (lr == 0 || lr == 3)
                rmat = load_symbolic<S, FL>(ifs);
        }
        if (no_ops)
            return;
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        ops.reserve(sz);
        for (int i = 0; i < sz; i++) {
            shared_ptr<OpExpr<S>> expr = load_expr<S, FL>(ifs);
            SparseMatrixTypes tp;
            ifs.read((char *)&tp, sizeof(tp));
            shared_ptr<SparseMatrix<S, FL>> mat;
            if (tp == SparseMatrixTypes::Normal)
                mat = make_shared<SparseMatrix<S, FL>>(d_alloc);
            else if (tp == SparseMatrixTypes::CSR)
                mat = make_shared<CSRSparseMatrix<S, FL>>(d_alloc);
            else if (tp == SparseMatrixTypes::Delayed)
                mat = make_shared<DelayedSparseMatrix<S, FL, OpExpr<S>>>(
                    0, nullptr, nullptr);
            else
                assert(false);
            mat->info = make_shared<SparseMatrixInfo<S>>(i_alloc);
            if (pointer_only)
                mat->alloc = dalloc_<FP>(), mat->info->alloc = ialloc;
            mat->info->load_data(ifs, pointer_only);
            mat->load_data(ifs, pointer_only);
            ops[expr] = mat;
        }
    }
    void save_data(ostream &ofs, bool pointer_only = false) const {
        uint8_t lr = lmat == rmat
                         ? (lmat == nullptr ? 4 : 1)
                         : (rmat == nullptr ? 2 : (lmat == nullptr ? 3 : 0));
        ofs.write((char *)&lr, sizeof(lr));
        if (lr == 1 || lr == 2)
            save_symbolic(lmat, ofs);
        else if (lr == 3)
            save_symbolic(rmat, ofs);
        else if (lr == 0) {
            save_symbolic(lmat, ofs);
            save_symbolic(rmat, ofs);
        }
        int sz = (int)ops.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (auto &op : ops) {
            save_expr(op.first, ofs);
            assert(op.second != nullptr);
            SparseMatrixTypes tp = op.second->get_type();
            assert(tp == SparseMatrixTypes::Normal ||
                   tp == SparseMatrixTypes::CSR ||
                   tp == SparseMatrixTypes::Delayed);
            ofs.write((char *)&tp, sizeof(tp));
            op.second->info->save_data(ofs, pointer_only);
            op.second->save_data(ofs, pointer_only);
        }
    }
    shared_ptr<OperatorTensor> copy() const {
        shared_ptr<OperatorTensor> r = make_shared<OperatorTensor>();
        r->lmat = lmat, r->rmat = rmat;
        r->ops = ops;
        return r;
    }
    shared_ptr<OperatorTensor>
    deep_copy(const shared_ptr<Allocator<FP>> &alloc = nullptr,
              const shared_ptr<Allocator<FP>> &ref_alloc = nullptr) const {
        shared_ptr<OperatorTensor> r = make_shared<OperatorTensor>();
        r->lmat = lmat, r->rmat = rmat;
        vector<shared_ptr<OpExpr<S>>> op_keys;
        r->ops.reserve(ops.size());
        op_keys.reserve(ops.size());
        for (auto &p : ops) {
            if (p.second->get_type() == SparseMatrixTypes::Normal) {
                shared_ptr<SparseMatrix<S, FL>> mat =
                    make_shared<SparseMatrix<S, FL>>(alloc);
                if (p.second->total_memory == 0)
                    mat->info = p.second->info;
                else if (p.second->alloc != ref_alloc) {
                    mat->total_memory = p.second->total_memory;
                    mat->info = p.second->info;
                    mat->data = p.second->data;
                    mat->alloc = nullptr;
                } else {
                    mat->allocate(p.second->info);
                    op_keys.push_back(p.first);
                }
                mat->factor = p.second->factor;
                r->ops[p.first] = mat;
            } else if (p.second->get_type() == SparseMatrixTypes::Archived) {
                shared_ptr<ArchivedSparseMatrix<S, FL>> pmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(p.second);
                shared_ptr<ArchivedSparseMatrix<S, FL>> mat =
                    make_shared<ArchivedSparseMatrix<S, FL>>(
                        pmat->filename, pmat->offset, pmat->alloc);
                mat->info = pmat->info;
                mat->factor = pmat->factor;
                mat->sparse_type = pmat->sparse_type;
                r->ops[p.first] = mat;
            } else if (p.second->get_type() == SparseMatrixTypes::Delayed)
                r->ops[p.first] =
                    dynamic_pointer_cast<DelayedSparseMatrix<S, FL>>(p.second)
                        ->copy();
            else
                assert(false);
        }
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static, 20) num_threads(ntg)
        for (int i = 0; i < (int)op_keys.size(); i++)
            r->ops.at(op_keys[i])->copy_data_from(ops.at(op_keys[i]));
        threading->activate_normal();
        return r;
    }
};

// Delayed contraction of left and right block MPO tensors
// or left and dot / dot and right (for 3-index operations)
template <typename S, typename FL>
struct DelayedOperatorTensor : OperatorTensor<S, FL> {
    typedef typename SparseMatrix<S, FL>::FP FP;
    // Symbol of super block operator(s)
    vector<shared_ptr<OpExpr<S>>> dops;
    // Symbolic expression of super block operator(s)
    shared_ptr<Symbolic<S>> mat;
    // SparseMatrix representation of symbols from left and right block
    shared_ptr<OperatorTensor<S, FL>> lopt, ropt;
    DelayedOperatorTensor() : OperatorTensor<S, FL>() {}
    OperatorTensorTypes get_type() const override {
        return OperatorTensorTypes::Delayed;
    }
    size_t get_total_memory() const override {
        size_t r = 0;
        for (auto it = ropt->ops.cbegin(); it != ropt->ops.cend(); it++)
            if (it->second->data != nullptr)
                r += it->second->total_memory;
        for (auto it = lopt->ops.cbegin(); it != lopt->ops.cend(); it++)
            if (it->second->data != nullptr)
                r += it->second->total_memory;
        return r;
    }
    void reallocate(bool clean) override {
        for (auto &p : lopt->ops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
        for (auto &p : ropt->ops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
    }
    void deallocate() override {
        // do not free contracted operators for future reuse in rotation
        if (!frame_<FP>()->use_main_stack)
            return;
        vector<pair<FL *, shared_ptr<SparseMatrix<S, FL>>>> mp;
        mp.reserve(lopt->ops.size() + ropt->ops.size());
        for (auto it = ropt->ops.cbegin(); it != ropt->ops.cend(); it++)
            mp.emplace_back(it->second->data, it->second);
        for (auto it = lopt->ops.cbegin(); it != lopt->ops.cend(); it++)
            mp.emplace_back(it->second->data, it->second);
        sort(mp.begin(), mp.end(),
             [](const pair<FL *, shared_ptr<SparseMatrix<S, FL>>> &a,
                const pair<FL *, shared_ptr<SparseMatrix<S, FL>>> &b) {
                 return a.first > b.first;
             });
        for (const auto &t : mp)
            t.second->deallocate();
    }
};

} // namespace block2
