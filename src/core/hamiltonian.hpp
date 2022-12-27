
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "delayed_sparse_matrix.hpp"
#include "expr.hpp"
#include "operator_functions.hpp"
#include "sparse_matrix.hpp"
#include "symbolic.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

// Delayed Operator names
enum struct DelayedOpNames : uint32_t {
    None = 0,
    H = 1,
    Normal = 2,
    R = 4,
    RD = 8,
    P = 16,
    PD = 32,
    Q = 64,
    CCDD = 128,
    CCD = 256,
    CDD = 512,
    TR = 1024,
    TS = 2048,
    LeftBig = 4096,
    RightBig = 8192
};

inline DelayedOpNames operator|(DelayedOpNames a, DelayedOpNames b) {
    return DelayedOpNames((uint32_t)a | (uint32_t)b);
}

inline uint32_t operator&(DelayedOpNames a, DelayedOpNames b) {
    return (uint32_t)a & (uint32_t)b;
}

template <typename S, typename = void> struct SiteBasis;

template <typename S> struct SiteBasis<S, typename S::is_sz_t> {
    static shared_ptr<StateInfo<S>> get(int isyma, int isymb = -1) {
        if (isymb == -1)
            isymb = isyma;
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(4);
        b->quanta[0] = S(0, 0, 0);
        b->quanta[1] = S(1, 1, isyma);
        b->quanta[2] = S(1, -1, isymb);
        b->quanta[3] = S(2, 0, S::pg_mul(isyma, isymb));
        b->n_states[0] = b->n_states[1] = b->n_states[2] = b->n_states[3] = 1;
        b->sort_states();
        return b;
    }
};

template <typename S> struct SiteBasis<S, typename S::is_su2_t> {
    static shared_ptr<StateInfo<S>> get(int isym) {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(3);
        b->quanta[0] = S(0, 0, 0);
        b->quanta[1] = S(1, 1, isym);
        b->quanta[2] = S(2, 0, S::pg_mul(isym, isym));
        b->n_states[0] = b->n_states[1] = b->n_states[2] = 1;
        b->sort_states();
        return b;
    }
};

template <typename S> struct SiteBasis<S, typename S::is_sg_t> {
    static shared_ptr<StateInfo<S>> get(int isym) {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(2);
        b->quanta[0] = S(0, 0);
        b->quanta[1] = S(1, isym);
        b->n_states[0] = b->n_states[1] = 1;
        b->sort_states();
        return b;
    }
};

// Hamiltonian includes sparse matrix info and matrix representations
// of site operators
template <typename S, typename FL> struct Hamiltonian {
    S vacuum;
    // Site basis
    vector<shared_ptr<StateInfo<S>>> basis;
    // Sparse matrix info for site operators
    vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>> site_op_infos;
    // Number of orbitals
    uint16_t n_sites;
    // Point group symmetry of orbitals
    vector<typename S::pg_t> orb_sym;
    // For storing pre-computed CG factors for sparse matrix functions
    shared_ptr<OperatorFunctions<S, FL>> opf = nullptr;
    DelayedOpNames delayed = DelayedOpNames::None;
    static vector<typename S::pg_t>
    combine_orb_sym(const vector<uint8_t> &orb_sym, const vector<int> &k_sym,
                    int k_mod) {
        assert(orb_sym.size() == k_sym.size());
        vector<typename S::pg_t> pg_sym(orb_sym.size());
        for (size_t i = 0; i < pg_sym.size(); i++)
            pg_sym[i] = S::pg_combine(orb_sym[i], k_sym[i], k_mod);
        return pg_sym;
    }
    Hamiltonian(S vacuum, int n_sites, const vector<typename S::pg_t> &orb_sym)
        : vacuum(vacuum), n_sites((uint16_t)n_sites), orb_sym(orb_sym) {
        assert((int)this->n_sites == n_sites);
    }
    virtual ~Hamiltonian() = default;
    virtual int get_n_orbs_left() const { return 0; }
    virtual int get_n_orbs_right() const { return 0; }
    // Fill the map with sparse matrix representation of site operators
    // The keys in map should be already set by filter_site_ops
    virtual void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>>
            &ops) const {};
    // Fill the map with sparse matrix representation of site operators
    // Trivial sparse matrices are removed from symbolic operator tensor and map
    void filter_site_ops(
        uint16_t m, const vector<shared_ptr<Symbolic<S>>> &mats,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>>
            &ops) const {
        vector<shared_ptr<Symbolic<S>>> pmats = mats;
        // hrl: ops is empty initially. It will be filled here. First by
        // specifying the keys, then by declaring the value
        if (pmats.size() == 2 && pmats[0] == pmats[1])
            pmats.resize(1);
        if (pmats.size() >= 1)
            ops.reserve(pmats[0]->data.size());
        // hrl: specifying key
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    ops[abs_value(x)] = nullptr;
                    break;
                case OpTypes::Sum:
                    for (auto &r :
                         dynamic_pointer_cast<OpSum<S, FL>>(x)->strings)
                        ops[abs_value((shared_ptr<OpExpr<S>>)r->get_op())] =
                            nullptr;
                    break;
                default:
                    assert(false);
                }
            }
        const shared_ptr<OpElement<S, FL>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), vacuum);
        ops[i_op] = nullptr;
        // hrl: specifying value
        get_site_ops(m, ops);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        // hrl: now check whether some keys are eq. to zero etc.; simplification
        size_t kk;
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                shared_ptr<OpExpr<S>> xx;
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem: {
                    shared_ptr<SparseMatrix<S, FL>> &mat = ops.at(abs_value(x));
                    if (mat->factor == (FL)0.0 || mat->info->n == 0 ||
                        mat->norm() < TINY)
                        x = zero;
                } break;
                case OpTypes::Sum:
                    kk = 0;
                    for (size_t i = 0;
                         i <
                         dynamic_pointer_cast<OpSum<S, FL>>(x)->strings.size();
                         i++) {
                        // hrl why abs_value? => to remove "phase" for getting
                        // key in ops
                        xx = abs_value((shared_ptr<OpExpr<S>>)
                                           dynamic_pointer_cast<OpSum<S, FL>>(x)
                                               ->strings[i]
                                               ->get_op());
                        shared_ptr<SparseMatrix<S, FL>> &mat = ops.at(xx);
                        if (!(mat->factor == (FL)0.0 || mat->info->n == 0 ||
                              mat->norm() < TINY)) {
                            if (i != kk)
                                dynamic_pointer_cast<OpSum<S, FL>>(x)
                                    ->strings[kk] =
                                    dynamic_pointer_cast<OpSum<S, FL>>(x)
                                        ->strings[i];
                            kk++;
                        }
                    }
                    if (kk == 0)
                        x = zero;
                    else if (kk != dynamic_pointer_cast<OpSum<S, FL>>(x)
                                       ->strings.size())
                        dynamic_pointer_cast<OpSum<S, FL>>(x)->strings.resize(
                            kk);
                    break;
                default:
                    assert(false);
                }
            }
        for (auto pmat : pmats)
            if (pmat->get_type() == SymTypes::Mat) {
                shared_ptr<SymbolicMatrix<S>> smat =
                    dynamic_pointer_cast<SymbolicMatrix<S>>(pmat);
                size_t j = 0;
                for (size_t i = 0; i < smat->indices.size(); i++)
                    if (smat->data[i]->get_type() != OpTypes::Zero) {
                        if (i != j)
                            smat->data[j] = smat->data[i],
                            smat->indices[j] = smat->indices[i];
                        j++;
                    }
                smat->data.resize(j);
                smat->indices.resize(j);
            }
        for (auto it = ops.cbegin(); it != ops.cend();) {
            if (it->second->factor == (FL)0.0 || it->second->info->n == 0)
                ops.erase(it++);
            else
                it++;
        }
    }
    // Find sparse matrix info for site operator with the given delta quantum q
    shared_ptr<SparseMatrixInfo<S>> find_site_op_info(uint16_t i, S q) const {
        auto p = lower_bound(site_op_infos[i].begin(), site_op_infos[i].end(),
                             q, SparseMatrixInfo<S>::cmp_op_info);
        if (p == site_op_infos[i].end() || p->first != q)
            return nullptr;
        else
            return p->second;
    }
    // get the delta quantum of an operator string
    virtual S get_string_quantum(const string &expr,
                                 const uint16_t *idxs) const {
        return vacuum;
    }
    virtual void deallocate() {}
};

// Delayed site operator
template <typename S, typename FL>
struct DelayedSparseMatrix<S, FL, Hamiltonian<S, FL>>
    : DelayedSparseMatrix<S, FL, OpExpr<S>> {
    using DelayedSparseMatrix<S, FL, OpExpr<S>>::m;
    using DelayedSparseMatrix<S, FL, OpExpr<S>>::op;
    shared_ptr<Hamiltonian<S, FL>> hamil;
    DelayedSparseMatrix(const shared_ptr<Hamiltonian<S, FL>> &hamil, uint16_t m,
                        const shared_ptr<OpExpr<S>> &op,
                        const shared_ptr<SparseMatrixInfo<S>> &info = nullptr)
        : DelayedSparseMatrix<S, FL, OpExpr<S>>(m, op, info), hamil(hamil) {}
    shared_ptr<SparseMatrix<S, FL>> build() override {
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>>
            ops;
        assert(hamil != nullptr);
        assert(hamil->delayed == DelayedOpNames::None);
        ops[op] = nullptr;
        hamil->get_site_ops(m, ops);
        if (this->info->n == ops.at(op)->info->n)
            return ops.at(op);
        else {
            shared_ptr<SparseMatrix<S, FL>> new_mat;
            if (ops.at(op)->get_type() == SparseMatrixTypes::Normal)
                new_mat = make_shared<SparseMatrix<S, FL>>();
            else
                new_mat = make_shared<CSRSparseMatrix<S, FL>>();
            new_mat->allocate(this->info);
            new_mat->selective_copy_from(ops.at(op), false);
            new_mat->factor = ops.at(op)->factor;
            ops.at(op)->deallocate();
            return new_mat;
        }
    }
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
