
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "../core/delayed_sparse_matrix.hpp"
#include "../core/expr.hpp"
#include "../core/operator_functions.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/symbolic.hpp"
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>
/** SCI Hamiltonian wrapper.
 */

using namespace std;

namespace block2 {

enum struct DelayedSCIOpNames : uint32_t {
    None = 0,
    H = 1,
    Normal = 2,
    R = 4,
    RD = 8,
    P = 16,
    PD = 32,
    Q = 64,
    LeftBig = 128,
    RightBig = 256
};

inline DelayedSCIOpNames operator|(DelayedSCIOpNames a, DelayedSCIOpNames b) {
    return DelayedSCIOpNames((uint32_t)a | (uint32_t)b);
}

inline uint32_t operator&(DelayedSCIOpNames a, DelayedSCIOpNames b) {
    return (uint32_t)a & (uint32_t)b;
}

/** Hamiltonian includes sparse matrix info and matrix representations
 * of site operators for SCI
 *
 * Copied from Hamiltonian with minor modifications right now:
 *  - basis is a vector of size n_sites.
 *     @attention: For point group symmetry, make sure that MPSInfo::orbsym is
 * empty!
 *
 * */
template <typename S> struct HamiltonianSCI {
    S vacuum;
    // Site basis
    vector<shared_ptr<StateInfo<S>>> basis; //!< Vector of size n_sites
    // Sparse matrix info for site operators
    vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>> site_op_infos;
    // Sparse matrix representation for normal site operators
    vector<pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>
        *site_norm_ops;
    // Number of sites and point group symmetry irreducible representations
    uint16_t n_sites, n_syms;
    shared_ptr<OperatorFunctions<S>> opf;
    // Point group symmetry of orbitals
    vector<uint8_t> orb_sym;
    DelayedSCIOpNames delayed = DelayedSCIOpNames::None;

    HamiltonianSCI(S vacuum, int n_sites, const vector<uint8_t> &orb_sym)
        : vacuum(vacuum), n_sites((uint16_t)n_sites), orb_sym(orb_sym) {
        assert((int)this->n_sites == n_sites);
        n_syms = *max_element(orb_sym.begin(), orb_sym.end()) + 1;
        basis = vector<shared_ptr<StateInfo<S>>>(n_sites);
        site_op_infos =
            vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(n_sites);
        site_norm_ops = new vector<
            pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>[n_sites];
        opf = make_shared<OperatorFunctions<S>>(make_shared<CG<S>>());
        opf->cg->initialize();
    }
    virtual ~HamiltonianSCI() = default;

    // Fill the map with sparse matrix representation of site operators
    // Trivial sparse matrices are removed from symbolic operator tensor and map
    void filter_site_ops(uint16_t m,
                         const vector<shared_ptr<Symbolic<S>>> &mats,
                         unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops) const {
        vector<shared_ptr<Symbolic<S>>> pmats = mats;
        // hrl: ops is empty initially. It will be filled here. First by
        // specifying the keys, then by declaring the value
        if (pmats.size() == 2 && pmats[0] == pmats[1])
            pmats.resize(1);
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
        // hrl: specifying value
        get_site_ops(m, ops);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        //  hrl: Now check whether some keys are eq. to zero etc.;
        //  simplification
        size_t kk;
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                shared_ptr<OpExpr<S>> xx;
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    xx = abs_value(x);
                    if (ops[xx]->factor == 0.0 || ops[xx]->info->n == 0 ||
                        ops[xx]->norm() < TINY)
                        x = zero;
                    break;
                case OpTypes::Sum:
                    kk = 0;
                    for (size_t i = 0;
                         i < dynamic_pointer_cast<OpSum<S>>(x)->strings.size();
                         i++) {
                        // TODO hrl why abs_value? => to remove "phase" for
                        // getting key in ops
                        xx = abs_value((shared_ptr<OpExpr<S>>)
                                           dynamic_pointer_cast<OpSum<S>>(x)
                                               ->strings[i]
                                               ->get_op());
                        shared_ptr<SparseMatrix<S>> &mat = ops[xx];
                        if (!(mat->factor == 0.0 || mat->info->n == 0 ||
                              mat->norm() < TINY)) {
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

    virtual void deallocate() {
        opf->cg->deallocate();
        delete[] site_norm_ops;
    }

    // Fill the map with sparse matrix representation of site operators
    // The keys in map should be already set by filter_site_ops
    /* @param m: site */
    virtual void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>
            &ops) const {};

  protected:
    // Find sparse matrix info for site operator with the given delta quantum q
    shared_ptr<SparseMatrixInfo<S>> find_site_op_info(S q,
                                                      uint16_t iSite) const {
        auto p = lower_bound(site_op_infos[iSite].begin(),
                             site_op_infos[iSite].end(), q,
                             SparseMatrixInfo<S>::cmp_op_info);
        if (p == site_op_infos[iSite].end() || p->first != q) {
            cerr << "find_site_op_info cant find q:" << q << "iSite=" << iSite
                 << endl;
            cerr << "last site =" << n_sites - 1 << endl;
            throw std::runtime_error("oops");
            return nullptr;
        } else
            return p->second;
    }

    // Find site normal operator with the given symbol q
    shared_ptr<SparseMatrix<S>>
    find_site_norm_op(const shared_ptr<OpExpr<S>> &q, uint16_t iSite) const {
        auto p = lower_bound(site_norm_ops[iSite].begin(),
                             site_norm_ops[iSite].end(), q, cmp_site_norm_op);
        if (p == site_norm_ops[iSite].end() || !(p->first == q)) {
            cout << "FIND SITE NORM OP FOR" << iSite << endl;
            cout << "fail for site" << iSite << endl;
            auto opQ = dynamic_pointer_cast<OpElement<S>>(q);
            cout << " that is:" << *opQ << endl;
            throw std::runtime_error("Fail in find_site_norm_op");
            assert(false);
            return nullptr;
        } else {
            return p->second;
        }
    }
};

// Delayed site operator
template <typename S>
struct DelayedSparseMatrix<S, HamiltonianSCI<S>> : DelayedSparseMatrix<S, OpExpr<S>> {
    shared_ptr<HamiltonianSCI<S>> hamil;
    using DelayedSparseMatrix<S, OpExpr<S>>::m;
    using DelayedSparseMatrix<S, OpExpr<S>>::op;
    DelayedSparseMatrix(const shared_ptr<HamiltonianSCI<S>> &hamil, uint16_t m,
                        const shared_ptr<OpExpr<S>> &op,
                        const shared_ptr<SparseMatrixInfo<S>> &info = nullptr)
        : DelayedSparseMatrix<S, OpExpr<S>>(m, op, info), hamil(hamil) {
    }
    shared_ptr<SparseMatrix<S>> build() override {
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>
            ops;
        assert(hamil != nullptr);
        assert(hamil->delayed == DelayedSCIOpNames::None);
        ops[op] = nullptr;
        hamil->get_site_ops(m, ops);
        if (this->info->n == ops.at(op)->info->n)
            return ops.at(op);
        else {
            shared_ptr<SparseMatrix<S>> new_mat;
            if (ops.at(op)->get_type() == SparseMatrixTypes::Normal)
                new_mat = make_shared<SparseMatrix<S>>();
            else
                new_mat = make_shared<CSRSparseMatrix<S>>();
            new_mat->allocate(this->info);
            new_mat->selective_copy_from(ops.at(op), false);
            new_mat->factor = ops.at(op)->factor;
            ops.at(op)->deallocate();
            return new_mat;
        }
    }
    shared_ptr<DelayedSparseMatrix<S>> copy() override {
        return make_shared<DelayedSparseMatrix>(*this);
    }
    shared_ptr<DelayedSparseMatrix<S>>
    selective_copy(const shared_ptr<SparseMatrixInfo<S>> &info) override {
        shared_ptr<DelayedSparseMatrix> mat =
            make_shared<DelayedSparseMatrix>(*this);
        mat->info = info;
        return mat;
    }
};

} // namespace block2
