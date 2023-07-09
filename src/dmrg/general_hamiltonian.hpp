
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2023 Huanchen Zhai <hczhai@caltech.edu>
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

/** Hilbert space definition for any symmetry. */

#pragma once

#include "../core/hamiltonian.hpp"
#include "../core/integral_general.hpp"
#include "../core/iterative_matrix_functions.hpp"
#include "../core/spin_permutation.hpp"
#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename, typename = void> struct GeneralHamiltonian;

// General Hamiltonian (non-spin-adapted)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename S::is_sz_t> : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    struct pair_hasher {
        size_t
        operator()(const pair<typename S::pg_t, typename S::pg_t> &x) const {
            size_t r = x.first;
            r ^= x.second + 0x9e3779b9 + (r << 6) + (r >> 2);
            return r;
        }
    };
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<pair<typename S::pg_t, typename S::pg_t>,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>,
                  pair_hasher>
        op_prims;
    const static int max_n = 10, max_s = 10;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        // alpha and beta orbitals can have different pg symmetries
        return orb_sym.size() != n_sites * 2
                   ? SiteBasis<S>::get(orb_sym[m])
                   : SiteBasis<S>::get(orb_sym[m], orb_sym[m + n_sites]);
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = -max_s_odd; s <= max_s_odd; s += 2) {
                    info[S(n, s, orb_sym[m])] = nullptr;
                    info[S(n, s, S::pg_inv(orb_sym[m]))] = nullptr;
                    if (orb_sym.size() == n_sites * 2) {
                        info[S(n, s, orb_sym[m + n_sites])] = nullptr;
                        info[S(n, s, S::pg_inv(orb_sym[m + n_sites]))] =
                            nullptr;
                    }
                }
            for (int n = -max_n_even; n <= max_n_even; n += 2)
                for (int s = -max_s_even; s <= max_s_even; s += 2) {
                    info[S(n, s, S::pg_mul(orb_sym[m], orb_sym[m]))] = nullptr;
                    info[S(n, s,
                           S::pg_mul(orb_sym[m], S::pg_inv(orb_sym[m])))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]), orb_sym[m]))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]),
                                     S::pg_inv(orb_sym[m])))] = nullptr;
                    if (orb_sym.size() == n_sites * 2) {
                        info[S(n, s,
                               S::pg_mul(orb_sym[m], orb_sym[m + n_sites]))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(orb_sym[m],
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m]),
                                         orb_sym[m + n_sites]))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m]),
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(orb_sym[m + n_sites],
                                         orb_sym[m + n_sites]))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(orb_sym[m + n_sites],
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m + n_sites]),
                                         orb_sym[m + n_sites]))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m + n_sites]),
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                    }
                }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            typename S::pg_t ipga = orb_sym[m], ipgb = ipga;
            if (orb_sym.size() == n_sites * 2)
                ipgb = orb_sym[m + n_sites];
            pair<typename S::pg_t, typename S::pg_t> ipg =
                make_pair(ipga, ipgb);
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
            (*op_prims[""])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims[""])[S(1, 1, ipga)](0, 0) = 1.0;
            (*op_prims[""])[S(1, -1, ipgb)](0, 0) = 1.0;
            (*op_prims[""])[S(2, 0, S::pg_mul(ipga, ipgb))](0, 0) = 1.0;

            op_prims["c"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["c"]->allocate(find_site_op_info(m, S(1, 1, ipga)));
            (*op_prims["c"])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims["c"])[S(1, -1, ipgb)](0, 0) = 1.0;

            op_prims["d"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["d"]->allocate(
                find_site_op_info(m, S(-1, -1, S::pg_inv(ipga))));
            (*op_prims["d"])[S(1, 1, ipga)](0, 0) = 1.0;
            (*op_prims["d"])[S(2, 0, S::pg_mul(ipga, ipgb))](0, 0) = 1.0;

            op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["C"]->allocate(find_site_op_info(m, S(1, -1, ipgb)));
            (*op_prims["C"])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims["C"])[S(1, 1, ipga)](0, 0) = -1.0;

            op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["D"]->allocate(
                find_site_op_info(m, S(-1, 1, S::pg_inv(ipgb))));
            (*op_prims["D"])[S(1, -1, ipgb)](0, 0) = 1.0;
            (*op_prims["D"])[S(2, 0, S::pg_mul(ipga, ipgb))](0, 0) = -1.0;
        }
        // site norm operators
        const string stx[5] = {"", "c", "C", "d", "D"};
        for (uint16_t m = 0; m < n_sites; m++) {
            typename S::pg_t ipga = orb_sym[m], ipgb = ipga;
            if (orb_sym.size() == n_sites * 2)
                ipgb = orb_sym[m + n_sites];
            pair<typename S::pg_t, typename S::pg_t> ipg =
                make_pair(ipga, ipgb);
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(m,
                                      op_prims.at(ipg)[t]->info->delta_quantum),
                    op_prims.at(ipg)[t]->data);
            }
        }
    }
    virtual shared_ptr<SparseMatrix<S, FL>>
    get_site_string_op(uint16_t m, const string &expr) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tx,
            tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
        if (site_norm_ops[m].count(expr))
            return site_norm_ops[m].at(expr);
        else {
            tx = site_norm_ops[m].at(string(1, expr[0]));
            for (size_t i = 1; i < expr.length(); i++) {
                S q = tx->info->delta_quantum + site_norm_ops[m]
                                                    .at(string(1, expr[i]))
                                                    ->info->delta_quantum;
                tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                tmp->allocate(find_site_op_info(m, q));
                opf->product(0, tx, site_norm_ops[m].at(string(1, expr[i])),
                             tmp);
                tx = tmp;
            }
            return (site_norm_ops[m][expr] = tx);
        }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1);
            r[ix][0] = S(0, 0, 0);
            for (int i = 0; i < term_l[ix]; i++)
                switch (exprs[ix][i]) {
                case 'c':
                    r[ix][i + 1] = r[ix][i] + S(1, 1, 0);
                    break;
                case 'C':
                    r[ix][i + 1] = r[ix][i] + S(1, -1, 0);
                    break;
                case 'd':
                    r[ix][i + 1] = r[ix][i] + S(-1, -1, 0);
                    break;
                case 'D':
                    r[ix][i + 1] = r[ix][i] + S(-1, 1, 0);
                    break;
                default:
                    assert(false);
                }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (orb_sym.size() == n_sites * 2 &&
                (expr[j] == 'C' || expr[j] == 'D'))
                ipg = orb_sym[idxs[j] + n_sites];
            if (expr[j] == 'd' || expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0, 0);
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
            if (idxs != nullptr && orb_sym.size() == n_sites * 2 &&
                (expr[j] == 'C' || expr[j] == 'D'))
                ipg = orb_sym[idxs[j] + n_sites];
            if (expr[j] == 'c')
                r = r + S(1, 1, ipg);
            else if (expr[j] == 'C')
                r = r + S(1, -1, ipg);
            else if (expr[j] == 'd')
                r = r + S(-1, -1, S::pg_inv(ipg));
            else if (expr[j] == 'D')
                r = r + S(-1, 1, S::pg_inv(ipg));
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return expr.substr(i, j - i);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (spin-adapted)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename S::is_su2_t> : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 10, max_s = 10;
    int twos;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym), twos(twos) {
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        if (twos == -1) // fermion model
            return SiteBasis<S>::get(orb_sym[m]);
        else // heisenberg spin model
            return make_shared<StateInfo<S>>(S(twos, twos, orb_sym[m]));
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = 1; s <= max_s_odd; s += 2) {
                    info[S(n, s, orb_sym[m])] = nullptr;
                    info[S(n, s, S::pg_inv(orb_sym[m]))] = nullptr;
                }
            for (int n = -max_n_even; n <= max_n_even; n += 2)
                for (int s = 0; s <= max_s_even; s += 2) {
                    info[S(n, s, S::pg_mul(orb_sym[m], orb_sym[m]))] = nullptr;
                    info[S(n, s,
                           S::pg_mul(orb_sym[m], S::pg_inv(orb_sym[m])))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]), orb_sym[m]))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]),
                                     S::pg_inv(orb_sym[m])))] = nullptr;
                }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        vector<string> stx = vector<string>{"", "C", "D"};
        if (twos == -1) { // fermion model
            for (uint16_t m = 0; m < n_sites; m++) {
                const typename S::pg_t ipg = orb_sym[m];
                if (this->op_prims.count(ipg) == 0)
                    this->op_prims[ipg] =
                        unordered_map<string,
                                      shared_ptr<SparseMatrix<S, FL>>>();
                else
                    continue;
                unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>
                    &op_prims = this->op_prims.at(ipg);
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
                (*op_prims[""])[S(0, 0, 0, 0)](0, 0) = 1.0;
                (*op_prims[""])[S(1, 1, 1, ipg)](0, 0) = 1.0;
                (*op_prims[""])[S(2, 0, 0, S::pg_mul(ipg, ipg))](0, 0) = 1.0;

                op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["C"]->allocate(find_site_op_info(m, S(1, 1, ipg)));
                (*op_prims["C"])[S(0, 1, 0, 0)](0, 0) = 1.0;
                (*op_prims["C"])[S(1, 0, 1, ipg)](0, 0) = -sqrt(2);

                op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["D"]->allocate(
                    find_site_op_info(m, S(-1, 1, S::pg_inv(ipg))));
                (*op_prims["D"])[S(1, 0, 1, ipg)](0, 0) = sqrt(2);
                (*op_prims["D"])[S(2, 1, 0, S::pg_mul(ipg, ipg))](0, 0) = 1.0;
            }
        } else { // heisenberg spin model
            stx = vector<string>{"", "T"};
            for (uint16_t m = 0; m < n_sites; m++) {
                const typename S::pg_t ipg = orb_sym[m];
                if (this->op_prims.count(ipg) == 0)
                    this->op_prims[ipg] =
                        unordered_map<string,
                                      shared_ptr<SparseMatrix<S, FL>>>();
                else
                    continue;
                unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>
                    &op_prims = this->op_prims.at(ipg);
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
                (*op_prims[""])[S(twos, twos, twos, 0)](0, 0) = 1.0;

                op_prims["T"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["T"]->allocate(find_site_op_info(m, S(0, 2, 0)));
                (*op_prims["T"])[S(twos, twos, twos, 0)](0, 0) =
                    sqrt((FL)twos * ((FL)twos + (FL)2.0) / (FL)2.0);
            }
        }
        // site norm operators
        for (uint16_t m = 0; m < n_sites; m++) {
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
        }
    }
    virtual shared_ptr<SparseMatrix<S, FL>>
    get_site_string_op(uint16_t m, const string &expr) {
        if (site_norm_ops[m].count(expr))
            return site_norm_ops[m].at(expr);
        else {
            int ix = 0, depth = 0;
            for (auto &c : expr) {
                if (c == '(')
                    depth++;
                else if (c == ')')
                    depth--;
                else if (c == '+' && depth == 1)
                    break;
                ix++;
            }
            int twos = 0, iy = 0;
            for (int i = (int)expr.length() - 1, k = 1; i >= 0; i--, k *= 10)
                if (expr[i] >= '0' && expr[i] <= '9')
                    twos += (expr[i] - '0') * k;
                else {
                    iy = i;
                    break;
                }
            shared_ptr<SparseMatrix<S, FL>> a =
                get_site_string_op(m, expr.substr(1, ix - 1));
            shared_ptr<SparseMatrix<S, FL>> b =
                get_site_string_op(m, expr.substr(ix + 1, iy - ix - 1));
            S dq = a->info->delta_quantum + b->info->delta_quantum;
            dq.set_twos(twos);
            dq.set_twos_low(twos);
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            shared_ptr<SparseMatrix<S, FL>> r =
                make_shared<SparseMatrix<S, FL>>(d_alloc);
            r->allocate(find_site_op_info(m, dq));
            opf->product(0, a, b, r);
            site_norm_ops[m][expr] = r;
            return r;
        }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1, S(0, 0, 0));
            vector<pair<int, char>> pex;
            pex.reserve(exprs[ix].length());
            bool is_heis = false;
            for (char x : exprs[ix])
                if (x >= '0' && x <= '9') {
                    if (pex.back().first != -1)
                        pex.back().first =
                            pex.back().first * 10 + (int)(x - '0');
                    else
                        pex.push_back(make_pair((int)(x - '0'), ' '));
                } else {
                    if (x == '+' && pex.back().second == ' ')
                        pex.back().second = '*';
                    else if (x == 'T')
                        is_heis = true;
                    pex.push_back(make_pair(-1, x));
                }
            const int site_dq = is_heis ? 2 : 1;
            if (pex.size() == 0)
                continue;
            else if (pex.size() == 1 && pex.back().first == -1) {
                // single C/D
                pex.insert(pex.begin(), make_pair(-1, '('));
                pex.push_back(make_pair(site_dq, ' '));
            }
            assert(pex.back().first != -1);
            int cnt = 0;
            // singlet embedding (twos will be set later)
            if (left_vacuum == S(S::invalid))
                r[ix][0].set_n(pex.back().first);
            else {
                assert(left_vacuum.twos() == pex.back().first);
                r[ix][0].set_n(left_vacuum.n());
                r[ix][0].set_pg(left_vacuum.pg());
            }
            vector<uint16_t> stk;
            for (auto &p : pex) {
                if (p.second == '(')
                    stk.push_back(cnt);
                // numbers in exprs like (()0+.) will not be used
                else if (p.second == '*')
                    stk.pop_back();
                // use the right part to define the twos for the left part
                // because the right part is not affected by the singlet
                // embedding
                else if (p.second == ' ') {
                    r[ix][stk.back()].set_twos(p.first);
                    r[ix][stk.back()].set_twos_low(p.first);
                    stk.pop_back();
                } else if (p.second == 'C')
                    r[ix][cnt + 1].set_n(r[ix][cnt].n() + 1), cnt++;
                else if (p.second == 'D')
                    r[ix][cnt + 1].set_n(r[ix][cnt].n() - 1), cnt++;
                else if (p.second == 'T')
                    cnt++, is_heis = true;
            }
            if (r[ix].size() >= 2) {
                r[ix][r[ix].size() - 2].set_twos(site_dq);
                r[ix][r[ix].size() - 2].set_twos_low(site_dq);
            }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0, i = 0; j < (uint16_t)expr.length(); j++) {
            if (expr[j] != 'C' && expr[j] != 'D')
                continue;
            typename S::pg_t ipg = orb_sym[idxs[i]];
            if (expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (i < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
            i++;
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0, 0);
        for (uint16_t j = 0, i = 0; j < (uint16_t)expr.length(); j++) {
            if (expr[j] != 'C' && expr[j] != 'D')
                continue;
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[i]] : 0;
            if (expr[j] == 'C')
                r = r + S(1, 1, ipg);
            else if (expr[j] == 'D')
                r = r + S(-1, 1, S::pg_inv(ipg));
            i++;
        }
        int rr = SpinPermRecoupling::get_target_twos(expr);
        r.set_twos(rr);
        r.set_twos_low(rr);
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return SpinPermRecoupling::get_sub_expr(expr, i, j);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (general spin, fermionic)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename enable_if<S::GIF>::type>
    : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 10, max_s = 10;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        return SiteBasis<S>::get(orb_sym[m]);
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n_odd; n <= max_n_odd; n += 2) {
                info[S(n, orb_sym[m])] = nullptr;
                info[S(n, S::pg_inv(orb_sym[m]))] = nullptr;
            }
            for (int n = -max_n_even; n <= max_n_even; n += 2) {
                info[S(n, S::pg_mul(orb_sym[m], orb_sym[m]))] = nullptr;
                info[S(n, S::pg_mul(orb_sym[m], S::pg_inv(orb_sym[m])))] =
                    nullptr;
                info[S(n, S::pg_mul(S::pg_inv(orb_sym[m]), orb_sym[m]))] =
                    nullptr;
                info[S(n, S::pg_mul(S::pg_inv(orb_sym[m]),
                                    S::pg_inv(orb_sym[m])))] = nullptr;
            }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            const typename S::pg_t ipg = orb_sym[m];
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0)));
            (*op_prims[""])[S(0, 0)](0, 0) = 1.0;
            (*op_prims[""])[S(1, ipg)](0, 0) = 1.0;

            op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["C"]->allocate(find_site_op_info(m, S(1, ipg)));
            (*op_prims["C"])[S(0, 0)](0, 0) = 1.0;

            op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["D"]->allocate(
                find_site_op_info(m, S(-1, S::pg_inv(ipg))));
            (*op_prims["D"])[S(1, ipg)](0, 0) = 1.0;
        }
        // site norm operators
        const string stx[3] = {"", "C", "D"};
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
    }
    virtual shared_ptr<SparseMatrix<S, FL>>
    get_site_string_op(uint16_t m, const string &expr) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tx,
            tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
        if (site_norm_ops[m].count(expr))
            return site_norm_ops[m].at(expr);
        else {
            tx = site_norm_ops[m].at(string(1, expr[0]));
            for (size_t i = 1; i < expr.length(); i++) {
                S q = tx->info->delta_quantum + site_norm_ops[m]
                                                    .at(string(1, expr[i]))
                                                    ->info->delta_quantum;
                tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                tmp->allocate(find_site_op_info(m, q));
                opf->product(0, tx, site_norm_ops[m].at(string(1, expr[i])),
                             tmp);
                tx = tmp;
            }
            return (site_norm_ops[m][expr] = tx);
        }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1);
            r[ix][0] = S(0, 0);
            for (int i = 0; i < term_l[ix]; i++)
                switch (exprs[ix][i]) {
                case 'C':
                    r[ix][i + 1] = r[ix][i] + S(1, 0);
                    break;
                case 'D':
                    r[ix][i + 1] = r[ix][i] + S(-1, 0);
                    break;
                default:
                    assert(false);
                }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0);
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
            if (expr[j] == 'C')
                r = r + S(1, ipg);
            else if (expr[j] == 'D')
                r = r + S(-1, S::pg_inv(ipg));
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return expr.substr(i, j - i);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (general spin, bosonic or spin)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename enable_if<!S::GIF>::type>
    : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 20;
    int twos;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym), twos(twos) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        if (orb_sym.size() == 0)
            Hamiltonian<S, FL>::orb_sym.resize(n_sites);
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(twos + 1);
        for (int i = 0, tm = -twos; tm < twos + 1; i++, tm += 2)
            b->quanta[i] = S(tm, tm == twos ? orb_sym[m] : 0),
            b->n_states[i] = 1;
        b->sort_states();
        return b;
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n; n <= max_n; n++) {
                info[S(n, orb_sym[m])] = nullptr;
                info[S(n, S::pg_inv(orb_sym[m]))] = nullptr;
                info[S(n, 0)] = nullptr;
            }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            const typename S::pg_t ipg = orb_sym[m];
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0)));
            for (int tm = -twos; tm < twos + 1; tm += 2)
                (*op_prims[""])[S(tm, tm == twos ? ipg : 0)](0, 0) = 1.0;

            op_prims["P"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["P"]->allocate(find_site_op_info(m, S(2, ipg)));
            for (int tm = -twos; tm < twos - 1; tm += 2)
                (*op_prims["P"])[S(tm, tm == twos ? ipg : 0)](0, 0) =
                    sqrt((twos - tm) * (twos + tm + 2) / 4);

            op_prims["M"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["M"]->allocate(
                find_site_op_info(m, S(-2, S::pg_inv(ipg))));
            for (int tm = -twos + 2; tm < twos + 1; tm += 2)
                (*op_prims["M"])[S(tm, tm == twos ? ipg : 0)](0, 0) =
                    sqrt((twos + tm) * (twos - tm + 2) / 4);

            op_prims["Z"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["Z"]->allocate(find_site_op_info(m, S(0, 0)));
            for (int tm = -twos; tm < twos + 1; tm += 2)
                (*op_prims["Z"])[S(tm, tm == twos ? ipg : 0)](0, 0) =
                    (FL)tm / (FL)2.0;
        }
        // site norm operators
        const string stx[4] = {"", "P", "M", "Z"};
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
    }
    virtual shared_ptr<SparseMatrix<S, FL>>
    get_site_string_op(uint16_t m, const string &expr) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tx,
            tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
        if (site_norm_ops[m].count(expr))
            return site_norm_ops[m].at(expr);
        else {
            tx = site_norm_ops[m].at(string(1, expr[0]));
            for (size_t i = 1; i < expr.length(); i++) {
                S q = tx->info->delta_quantum + site_norm_ops[m]
                                                    .at(string(1, expr[i]))
                                                    ->info->delta_quantum;
                tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                tmp->allocate(find_site_op_info(m, q));
                opf->product(0, tx, site_norm_ops[m].at(string(1, expr[i])),
                             tmp);
                tx = tmp;
            }
            return (site_norm_ops[m][expr] = tx);
        }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1);
            r[ix][0] = S(0, 0);
            for (int i = 0; i < term_l[ix]; i++)
                switch (exprs[ix][i]) {
                case 'P':
                    r[ix][i + 1] = r[ix][i] + S(2, 0);
                    break;
                case 'M':
                    r[ix][i + 1] = r[ix][i] + S(-2, 0);
                    break;
                case 'Z':
                    r[ix][i + 1] = r[ix][i];
                    break;
                default:
                    assert(false);
                }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (expr[j] == 'Z')
                continue;
            else if (expr[j] == 'M')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0);
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
            if (expr[j] == 'Z')
                continue;
            else if (expr[j] == 'P')
                r = r + S(2, ipg);
            else if (expr[j] == 'M')
                r = r + S(-2, S::pg_inv(ipg));
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return expr.substr(i, j - i);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (any symmetry)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename S::is_sany_t> : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 10, max_s = 10;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        if (vacuum == S::init_su2()) {
            b->allocate(3);
            b->quanta[0] = S::init_su2(0, 0, 0);
            b->quanta[1] = S::init_su2(1, 1, orb_sym[m]);
            b->quanta[2] = S::init_su2(2, 0, 0);
            b->n_states[0] = b->n_states[1] = b->n_states[2] = 1;
        } else if (vacuum == S::init_sz()) {
            b->allocate(4);
            b->quanta[0] = S::init_sz(0, 0, 0);
            b->quanta[1] = S::init_sz(1, 1, orb_sym[m]);
            b->quanta[2] = S::init_sz(1, -1, orb_sym[m]);
            b->quanta[3] = S::init_sz(2, 0, 0);
            b->n_states[0] = b->n_states[1] = b->n_states[2] = b->n_states[3] =
                1;
        } else if (vacuum == S::init_sgf()) {
            b->allocate(2);
            b->quanta[0] = S::init_sgf(0, 0);
            b->quanta[1] = S::init_sgf(1, orb_sym[m]);
            b->n_states[0] = b->n_states[1] = 1;
        } else {
            stringstream ss;
            ss << vacuum;
            throw runtime_error("Symmetry not implemented: " + ss.str());
        }
        b->sort_states();
        return b;
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            if (vacuum == S::init_su2()) {
                for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                    for (int s = 1; s <= max_s_odd; s += 2)
                        info[S::init_su2(n, s, orb_sym[m])] = nullptr;
                for (int n = -max_n_even; n <= max_n_even; n += 2)
                    for (int s = 0; s <= max_s_even; s += 2)
                        info[S::init_su2(n, s, 0)] = nullptr;
            } else if (vacuum == S::init_sz()) {
                for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                    for (int s = -max_s_odd; s <= max_s_odd; s += 2)
                        info[S::init_sz(n, s, orb_sym[m])] = nullptr;
                for (int n = -max_n_even; n <= max_n_even; n += 2)
                    for (int s = -max_s_even; s <= max_s_even; s += 2)
                        info[S::init_sz(n, s, 0)] = nullptr;
            } else if (vacuum == S::init_sgf()) {
                for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                    info[S::init_sgf(n, orb_sym[m])] = nullptr;
                for (int n = -max_n_even; n <= max_n_even; n += 2)
                    info[S::init_sgf(n, 0)] = nullptr;
            }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        vector<string> stx;
        if (vacuum == S::init_su2())
            stx = vector<string>{"", "C", "D"};
        else if (vacuum == S::init_sz())
            stx = vector<string>{"", "c", "C", "d", "D"};
        else if (vacuum == S::init_sgf())
            stx = vector<string>{"", "C", "D"};
        for (uint16_t m = 0; m < n_sites; m++) {
            typename S::pg_t ipg = orb_sym[m];
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            if (vacuum == S::init_su2()) {
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(
                    find_site_op_info(m, S::init_su2(0, 0, 0)));
                (*op_prims[""])[S::init_su2(0, 0, 0, 0)](0, 0) = 1.0;
                (*op_prims[""])[S::init_su2(1, 1, 1, ipg)](0, 0) = 1.0;
                (*op_prims[""])[S::init_su2(2, 0, 0, S::pg_mul(ipg, ipg))](
                    0, 0) = 1.0;

                op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["C"]->allocate(
                    find_site_op_info(m, S::init_su2(1, 1, ipg)));
                (*op_prims["C"])[S::init_su2(0, 1, 0, 0)](0, 0) = 1.0;
                (*op_prims["C"])[S::init_su2(1, 0, 1, ipg)](0, 0) = -sqrt(2);

                op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["D"]->allocate(
                    find_site_op_info(m, S::init_su2(-1, 1, S::pg_inv(ipg))));
                (*op_prims["D"])[S::init_su2(1, 0, 1, ipg)](0, 0) = sqrt(2);
                (*op_prims["D"])[S::init_su2(2, 1, 0, S::pg_mul(ipg, ipg))](
                    0, 0) = 1.0;
            } else if (vacuum == S::init_sz()) {
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(
                    find_site_op_info(m, S::init_sz(0, 0, 0)));
                (*op_prims[""])[S::init_sz(0, 0, 0)](0, 0) = 1.0;
                (*op_prims[""])[S::init_sz(1, 1, ipg)](0, 0) = 1.0;
                (*op_prims[""])[S::init_sz(1, -1, ipg)](0, 0) = 1.0;
                (*op_prims[""])[S::init_sz(2, 0, S::pg_mul(ipg, ipg))](0, 0) =
                    1.0;

                op_prims["c"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["c"]->allocate(
                    find_site_op_info(m, S::init_sz(1, 1, ipg)));
                (*op_prims["c"])[S::init_sz(0, 0, 0)](0, 0) = 1.0;
                (*op_prims["c"])[S::init_sz(1, -1, ipg)](0, 0) = 1.0;

                op_prims["d"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["d"]->allocate(
                    find_site_op_info(m, S::init_sz(-1, -1, S::pg_inv(ipg))));
                (*op_prims["d"])[S::init_sz(1, 1, ipg)](0, 0) = 1.0;
                (*op_prims["d"])[S::init_sz(2, 0, S::pg_mul(ipg, ipg))](0, 0) =
                    1.0;

                op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["C"]->allocate(
                    find_site_op_info(m, S::init_sz(1, -1, ipg)));
                (*op_prims["C"])[S::init_sz(0, 0, 0)](0, 0) = 1.0;
                (*op_prims["C"])[S::init_sz(1, 1, ipg)](0, 0) = -1.0;

                op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["D"]->allocate(
                    find_site_op_info(m, S::init_sz(-1, 1, S::pg_inv(ipg))));
                (*op_prims["D"])[S::init_sz(1, -1, ipg)](0, 0) = 1.0;
                (*op_prims["D"])[S::init_sz(2, 0, S::pg_mul(ipg, ipg))](0, 0) =
                    -1.0;
            } else if (vacuum == S::init_sgf()) {
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(find_site_op_info(m, S::init_sgf(0, 0)));
                (*op_prims[""])[S::init_sgf(0, 0)](0, 0) = 1.0;
                (*op_prims[""])[S::init_sgf(1, ipg)](0, 0) = 1.0;

                op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["C"]->allocate(
                    find_site_op_info(m, S::init_sgf(1, ipg)));
                (*op_prims["C"])[S::init_sgf(0, 0)](0, 0) = 1.0;

                op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["D"]->allocate(
                    find_site_op_info(m, S::init_sgf(-1, S::pg_inv(ipg))));
                (*op_prims["D"])[S::init_sgf(1, ipg)](0, 0) = 1.0;
            }
        }
        // site norm operators
        for (uint16_t m = 0; m < n_sites; m++) {
            typename S::pg_t ipg = orb_sym[m];
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(m,
                                      op_prims.at(ipg)[t]->info->delta_quantum),
                    op_prims.at(ipg)[t]->data);
            }
        }
    }
    virtual shared_ptr<SparseMatrix<S, FL>>
    get_site_string_op(uint16_t m, const string &expr) {
        if (site_norm_ops[m].count(expr))
            return site_norm_ops[m].at(expr);
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> r =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        shared_ptr<SparseMatrix<S, FL>> a, b;
        S dq;
        if (vacuum == S::init_su2()) {
            int ix = 0, depth = 0;
            for (auto &c : expr) {
                if (c == '(')
                    depth++;
                else if (c == ')')
                    depth--;
                else if (c == '+' && depth == 1)
                    break;
                ix++;
            }
            int iy = 0;
            for (int i = (int)expr.length() - 1, k = 1; i >= 0; i--, k *= 10)
                if (!(expr[i] >= '0' && expr[i] <= '9')) {
                    iy = i;
                    break;
                }
            a = get_site_string_op(m, expr.substr(1, ix - 1));
            b = get_site_string_op(m, expr.substr(ix + 1, iy - ix - 1));
            int qa =
                SpinPermRecoupling::get_target_twos(expr.substr(1, ix - 1));
            int qb = SpinPermRecoupling::get_target_twos(
                expr.substr(ix + 1, iy - ix - 1));
            int qc = SpinPermRecoupling::get_target_twos(expr);
            dq = (a->info->delta_quantum +
                  b->info->delta_quantum)[(qc - abs(qa - qb)) / 2];
        } else {
            a = get_site_string_op(m, expr.substr(0, expr.length() - 1));
            b = get_site_string_op(m, expr.substr(expr.length() - 1, 1));
            dq = a->info->delta_quantum + b->info->delta_quantum;
        }
        r->allocate(find_site_op_info(m, dq));
        opf->product(0, a, b, r);
        site_norm_ops[m][expr] = r;
        return r;
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1, vacuum);
            if (vacuum == S::init_su2()) {
                vector<pair<int, char>> pex;
                pex.reserve(exprs[ix].length());
                for (char x : exprs[ix])
                    if (x >= '0' && x <= '9') {
                        if (pex.back().first != -1)
                            pex.back().first =
                                pex.back().first * 10 + (int)(x - '0');
                        else
                            pex.push_back(make_pair((int)(x - '0'), ' '));
                    } else {
                        if (x == '+' && pex.back().second == ' ')
                            pex.back().second = '*';
                        pex.push_back(make_pair(-1, x));
                    }
                const int site_dq = 1;
                if (pex.size() == 0)
                    continue;
                else if (pex.size() == 1 && pex.back().first == -1) {
                    // single C/D
                    pex.insert(pex.begin(), make_pair(-1, '('));
                    pex.push_back(make_pair(site_dq, ' '));
                }
                assert(pex.back().first != -1);
                int cnt = 0;
                vector<int> qn(r[ix].size());
                // singlet embedding (twos will be set later)
                assert(left_vacuum == S(S::invalid));
                r[ix][0] = r[ix][0] + S::init_su2(pex.back().first, 0, 0);
                qn[0] = pex.back().first;
                vector<uint16_t> stk;
                for (auto &p : pex) {
                    if (p.second == '(')
                        stk.push_back(cnt);
                    // numbers in exprs like (()0+.) will not be used
                    else if (p.second == '*')
                        stk.pop_back();
                    // use the right part to define the twos for the left part
                    // because the right part is not affected by the singlet
                    // embedding
                    else if (p.second == ' ') {
                        r[ix][stk.back()].set_twos(p.first);
                        r[ix][stk.back()].set_twos_low(p.first);
                        stk.pop_back();
                    } else if (p.second == 'C')
                        r[ix][cnt + 1] =
                            r[ix][cnt + 1] + S::init_su2(qn[cnt] + 1, 0, 0),
                                    qn[cnt + 1] = qn[cnt] + 1, cnt++;
                    else if (p.second == 'D')
                        r[ix][cnt + 1] =
                            r[ix][cnt + 1] + S::init_su2(qn[cnt] - 1, 0, 0),
                                    qn[cnt + 1] = qn[cnt] - 1, cnt++;
                }
                if (r[ix].size() >= 2) {
                    r[ix][r[ix].size() - 2].set_twos(site_dq);
                    r[ix][r[ix].size() - 2].set_twos_low(site_dq);
                }
            } else if (vacuum == S::init_sz()) {
                for (int i = 0; i < term_l[ix]; i++)
                    switch (exprs[ix][i]) {
                    case 'c':
                        r[ix][i + 1] = r[ix][i] + S::init_sz(1, 1, 0);
                        break;
                    case 'C':
                        r[ix][i + 1] = r[ix][i] + S::init_sz(1, -1, 0);
                        break;
                    case 'd':
                        r[ix][i + 1] = r[ix][i] + S::init_sz(-1, -1, 0);
                        break;
                    case 'D':
                        r[ix][i + 1] = r[ix][i] + S::init_sz(-1, 1, 0);
                        break;
                    default:
                        assert(false);
                    }
            } else if (vacuum == S::init_sgf()) {
                for (int i = 0; i < term_l[ix]; i++)
                    switch (exprs[ix][i]) {
                    case 'C':
                        r[ix][i + 1] = r[ix][i] + S::init_sgf(1, 0);
                        break;
                    case 'D':
                        r[ix][i + 1] = r[ix][i] + S::init_sgf(-1, 0);
                        break;
                    default:
                        assert(false);
                    }
            }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0, i = 0; j < (uint16_t)expr.length(); j++) {
            if (vacuum == S::init_su2()) {
                if (expr[j] != 'C' && expr[j] != 'D')
                    continue;
                typename S::pg_t ipg = orb_sym[idxs[i]];
                if (expr[j] == 'D')
                    ipg = S::pg_inv(ipg);
                if (i < k)
                    l = l + S::init_su2(0, 0, ipg);
                else
                    r = r + S::init_su2(0, 0, ipg);
                i++;
            } else if (vacuum == S::init_sz()) {
                typename S::pg_t ipg = orb_sym[idxs[j]];
                if (expr[j] == 'd' || expr[j] == 'D')
                    ipg = S::pg_inv(ipg);
                if (j < k)
                    l = l + S::init_sz(0, 0, ipg);
                else
                    r = r + S::init_sz(0, 0, ipg);
            } else if (vacuum == S::init_sgf()) {
                typename S::pg_t ipg = orb_sym[idxs[j]];
                if (expr[j] == 'D')
                    ipg = S::pg_inv(ipg);
                if (j < k)
                    l = l + S::init_sgf(0, ipg);
                else
                    r = r + S::init_sgf(0, ipg);
            }
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r = vacuum;
        for (uint16_t j = 0, i = 0; j < (uint16_t)expr.length(); j++) {
            if (vacuum == S::init_su2()) {
                if (expr[j] != 'C' && expr[j] != 'D')
                    continue;
                typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[i]] : 0;
                if (expr[j] == 'C')
                    r = r + S::init_su2(1, 0, ipg);
                else if (expr[j] == 'D')
                    r = r + S::init_su2(-1, 0, S::pg_inv(ipg));
                i++;
            } else if (vacuum == S::init_sz()) {
                typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
                if (expr[j] == 'c')
                    r = r + S::init_sz(1, 1, ipg);
                else if (expr[j] == 'C')
                    r = r + S::init_sz(1, -1, ipg);
                else if (expr[j] == 'd')
                    r = r + S::init_sz(-1, -1, S::pg_inv(ipg));
                else if (expr[j] == 'D')
                    r = r + S::init_sz(-1, 1, S::pg_inv(ipg));
            } else if (vacuum == S::init_sgf()) {
                typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
                if (expr[j] == 'C')
                    r = r + S::init_sgf(1, ipg);
                else if (expr[j] == 'D')
                    r = r + S::init_sgf(-1, S::pg_inv(ipg));
            }
        }
        if (vacuum == S::init_su2()) {
            int rr = SpinPermRecoupling::get_target_twos(expr);
            r = r + S::init_su2(0, rr, 0);
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        if (expr.find('+') == string::npos && expr.find('(') == string::npos &&
            expr.find('[') == string::npos)
            return expr.substr(i, j - i);
        else
            return SpinPermRecoupling::get_sub_expr(expr, i, j);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

} // namespace block2
