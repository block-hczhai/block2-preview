
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

// Hamiltonian includes sparse matrix info and matrix representations
// of site operators
template <typename S> struct Hamiltonian {
    S vacuum, target;
    // Site basis
    StateInfo<S> *basis;
    // Sparse matrix info for site operators
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> *site_op_infos;
    // Sparse matrix representation for normal site operators
    vector<pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>
        *site_norm_ops;
    // Number of orbitals and point group symmetry irreducible representations
    uint8_t n_sites, n_syms;
    shared_ptr<OperatorFunctions<S>> opf;
    // Point group symmetry of orbitals
    vector<uint8_t> orb_sym;
    Hamiltonian(S vacuum, S target, int n_sites, const vector<uint8_t> &orb_sym)
        : vacuum(vacuum), target(target), n_sites((uint8_t)n_sites),
          orb_sym(orb_sym) {
        assert((int)this->n_sites == n_sites);
        n_syms = *max_element(orb_sym.begin(), orb_sym.end()) + 1;
        basis = new StateInfo<S>[n_syms];
        site_op_infos =
            new vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>[n_syms];
        site_norm_ops = new vector<
            pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>[n_syms];
        opf = make_shared<OperatorFunctions<S>>(make_shared<CG<S>>(100));
        opf->cg->initialize();
    }
    // Fill the map with sparse matrix representation of site operators
    // The keys in map should be already set by filter_site_ops
    virtual void get_site_ops(
        uint8_t m,
        map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
            &ops) const {};
    // Fill the map with sparse matrix representation of site operators
    // Trivial sparse matrices are removed from symbolic operator tensor and map
    void filter_site_ops(uint8_t m, const vector<shared_ptr<Symbolic<S>>> &mats,
                         map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                             op_expr_less<S>> &ops) const {
        vector<shared_ptr<Symbolic<S>>> pmats = mats;
        if (pmats.size() == 2 && pmats[0] == pmats[1])
            pmats.resize(1);
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    ops[abs_value(x)] = nullptr;
                    break;
                case OpTypes::Sum:
                    for (auto &r : dynamic_pointer_cast<OpSum<S>>(x)->strings)
                        ops[abs_value((shared_ptr<OpExpr<S>>)r->get_op())] =
                            nullptr;
                    break;
                default:
                    assert(false);
                }
            }
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), vacuum);
        ops[i_op] = nullptr;
        get_site_ops(m, ops);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        size_t kk;
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                shared_ptr<OpExpr<S>> xx;
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    xx = abs_value(x);
                    if (ops[xx]->factor == 0.0 || ops[xx]->info->n == 0)
                        x = zero;
                    break;
                case OpTypes::Sum:
                    kk = 0;
                    for (size_t i = 0;
                         i < dynamic_pointer_cast<OpSum<S>>(x)->strings.size();
                         i++) {
                        xx = abs_value((shared_ptr<OpExpr<S>>)
                                           dynamic_pointer_cast<OpSum<S>>(x)
                                               ->strings[i]
                                               ->get_op());
                        shared_ptr<SparseMatrix<S>> &mat = ops[xx];
                        if (!(mat->factor == 0.0 || mat->info->n == 0)) {
                            if (i != kk)
                                dynamic_pointer_cast<OpSum<S>>(x)->strings[kk] =
                                    dynamic_pointer_cast<OpSum<S>>(x)
                                        ->strings[i];
                            kk++;
                        }
                    }
                    if (kk == 0)
                        x = zero;
                    else if (kk !=
                             dynamic_pointer_cast<OpSum<S>>(x)->strings.size())
                        dynamic_pointer_cast<OpSum<S>>(x)->strings.resize(kk);
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
            if (it->second->factor == 0.0 || it->second->info->n == 0)
                ops.erase(it++);
            else
                it++;
        }
    }
    static bool cmp_site_norm_op(
        const pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &p,
        const shared_ptr<OpExpr<S>> &q) {
        return op_expr_less<S>()(p.first, q);
    }
    // Find sparse matrix info for site operator with the given delta quantum q
    shared_ptr<SparseMatrixInfo<S>> find_site_op_info(S q,
                                                      uint8_t i_sym) const {
        auto p = lower_bound(site_op_infos[i_sym].begin(),
                             site_op_infos[i_sym].end(), q,
                             SparseMatrixInfo<S>::cmp_op_info);
        if (p == site_op_infos[i_sym].end() || p->first != q)
            return nullptr;
        else
            return p->second;
    }
    // Find site normal operator with the given symbol q
    shared_ptr<SparseMatrix<S>>
    find_site_norm_op(const shared_ptr<OpExpr<S>> &q, uint8_t i_sym) const {
        auto p = lower_bound(site_norm_ops[i_sym].begin(),
                             site_norm_ops[i_sym].end(), q, cmp_site_norm_op);
        if (p == site_norm_ops[i_sym].end() || !(p->first == q))
            return nullptr;
        else
            return p->second;
    }
    virtual void deallocate() {
        opf->cg->deallocate();
        delete[] site_norm_ops;
        delete[] site_op_infos;
        delete[] basis;
    }
};

} // namespace block2
