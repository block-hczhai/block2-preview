
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

#include "../core/expr.hpp"
#include "../core/hamiltonian.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/symbolic.hpp"
#include "../core/tensor_functions.hpp"
#include "mpo.hpp"
#include <cassert>
#include <memory>

using namespace std;

namespace block2 {

template <typename, typename = void> struct PDM1MPOQC;

// "MPO" for one particle density matrix (non-spin-adapted)
template <typename S> struct PDM1MPOQC<S, typename S::is_sz_t> : MPO<S> {
    PDM1MPOQC(const shared_ptr<Hamiltonian<S>> &hamil, uint8_t ds = 0)
        : MPO<S>(hamil->n_sites) {
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil->vacuum);
        shared_ptr<OpElement<S>> zero_op = make_shared<OpElement<S>>(
            OpNames::Zero, SiteIndex(), hamil->vacuum);
        assert(ds == 0);
        const uint16_t n_sites = MPO<S>::n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> c_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> d_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> b_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> pdm1_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
#else
        shared_ptr<OpExpr<S>> c_op[n_orbs][2], d_op[n_orbs][2];
        shared_ptr<OpExpr<S>> b_op[n_orbs][n_orbs][4];
        shared_ptr<OpExpr<S>> pdm1_op[n_orbs][n_orbs][4];
#endif
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint16_t m = 0; m < n_orbs; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil->orb_sym[m]));
                d_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil->orb_sym[m]));
            }
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s],
                          hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                    pdm1_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PDM1, sidx,
                        S(0, sz_minus[s],
                          hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil->opf);
        MPO<S>::site_op_infos = hamil->site_op_infos;
        for (uint16_t pm = 0; pm < n_sites; pm++) {
            uint16_t m = pm + n_orbs_big_left - 1;
            // left operator names
            //   1 : identity
            //   1*4 : mm / cd
            //   2*2 : m / c d
            // right operator names
            //   1 : identity
            //   2*2*(n-m) : j / c d (j >= m)
            //   1*4 : mm / cd (only last site)
            int lshape, rshape;
            if (pm == 0)
                lshape = 1 + 4 * (m + 1) + 4 * (m + 1) * (m + 1);
            else if (pm == n_sites - 1)
                lshape = 1;
            else
                lshape = 1 + 4 + 4;
            if (pm == 0)
                rshape = 1;
            else if (pm == n_sites - 1)
                rshape = 1 + 4 * (n_orbs - m) + 4 * (n_orbs - m) * (n_orbs - m);
            else
                rshape = 1 + 4 * (n_orbs - m);
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            // 1 : identity
            (*plop)[0] = i_op;
            int p = 1;
            if (pm == 0) {
                // 1*4 : mm / cd
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m + 1; j++)
                        for (uint16_t k = 0; k < m + 1; k++)
                            (*plop)[p++] = b_op[j][k][s];
                // 2*2 : m / c d
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        (*plop)[p + j] = c_op[j][s];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        (*plop)[p + j] = d_op[j][s];
                    p += m + 1;
                }
            } else if (pm != n_sites - 1) {
                // 1*4 : mm / cd
                for (uint8_t s = 0; s < 4; s++)
                    (*plop)[p + s] = b_op[m][m][s];
                p += 4;
                // 2*2 : m / c d
                for (uint8_t s = 0; s < 2; s++) {
                    (*plop)[p++] = c_op[m][s];
                    (*plop)[p++] = d_op[m][s];
                }
            }
            assert(p == lshape);
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            // 1 : identity
            (*prop)[0] = i_op;
            p = 1;
            if (pm != 0) {
                // 2*2*(n-m) : j / c d (j >= m)
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_orbs; j++)
                        (*prop)[p + j - m] = c_op[j][s];
                    p += n_orbs - m;
                    for (uint16_t j = m; j < n_orbs; j++)
                        (*prop)[p + j - m] = d_op[j][s];
                    p += n_orbs - m;
                }
            }
            if (pm == n_sites - 1) {
                // 1*4 : mm / cd (only last site)
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = m; j < n_orbs; j++)
                        for (uint16_t k = m; k < n_orbs; k++)
                            (*prop)[p++] = b_op[j][k][s];
            }
            assert(p == rshape);
            this->right_operator_names.push_back(prop);
            // middle operators
            //   1*4*1 : mm / cd
            //   2*4*(n-m-1) : mj(-jm) / cd dc (j > m)
            //   1*4*1 : jj / cd (j > m) (last site only)
            if (pm != n_sites - 1) {
                int mshape;
                if (pm == 0)
                    mshape =
                        4 * (m + 1) * (m + 1) + 8 * (m + 1) * (n_orbs - m - 1);
                else
                    mshape = 4 + 8 * (n_orbs - m - 1);
                if (pm == n_sites - 2)
                    mshape += 4 * (n_orbs - m - 1) * (n_orbs - m - 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                p = 0;
                if (pm == 0) {
                    for (uint8_t s = 0; s < 4; s++) {
                        // 1*4*1 : mm / cd
                        for (uint16_t j = 0; j < m + 1; j++)
                            for (uint16_t k = 0; k < m + 1; k++) {
                                (*pmop)[p] = pdm1_op[j][k][s];
                                (*pmexpr)[p] = b_op[j][k][s] * i_op;
                                p++;
                            }
                        // 2*4*(n-m-1) : mj(-jm) / cd dc (j > m)
                        for (uint16_t k = 0; k < m + 1; k++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                (*pmop)[p] = pdm1_op[k][j][s];
                                (*pmexpr)[p] = c_op[k][s & 1] * d_op[j][s >> 1];
                                p++;
                                (*pmop)[p] = pdm1_op[j][k][s];
                                (*pmexpr)[p] =
                                    -1.0 * (d_op[k][s >> 1] * c_op[j][s & 1]);
                                p++;
                            }
                    }
                } else {
                    for (uint8_t s = 0; s < 4; s++) {
                        // 1*4*1 : mm / cd
                        (*pmop)[p] = pdm1_op[m][m][s];
                        (*pmexpr)[p] = b_op[m][m][s] * i_op;
                        p++;
                        // 2*4*(n-m-1) : mj(-jm) / cd dc (j > m)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            (*pmop)[p] = pdm1_op[m][j][s];
                            (*pmexpr)[p] = c_op[m][s & 1] * d_op[j][s >> 1];
                            p++;
                            (*pmop)[p] = pdm1_op[j][m][s];
                            (*pmexpr)[p] =
                                -1.0 * (d_op[m][s >> 1] * c_op[j][s & 1]);
                            p++;
                        }
                    }
                }
                // 1*4*1 : jj / cd (j > m) (last site only)
                if (pm == n_sites - 2)
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            for (uint16_t k = m + 1; k < n_orbs; k++) {
                                (*pmop)[p] = pdm1_op[j][k][s];
                                (*pmexpr)[p] = i_op * b_op[j][k][s];
                                p++;
                            }
                assert(p == mshape);
                this->middle_operator_names.push_back(pmop);
                this->middle_operator_exprs.push_back(pmexpr);
            }
            // site tensors
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            // left operator names
            //   1 : identity
            //   1*4 : mm / cd
            //   2*2 : m / c d
            int llshape = pm == 1 ? 1 + 4 * m + 4 * m * m : 1 + 4 + 4;
            int lrshape = pm == 0 ? 1 + 4 * (m + 1) + 4 * (m + 1) * (m + 1)
                                  : (pm != n_sites - 1 ? 1 + 4 + 4 : 1);
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (pm == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (pm == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            p = 1;
            if (pm == 0) {
                int pi = 0;
                // 1*4 : mm / cd
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m + 1; j++)
                        for (uint16_t k = 0; k < m + 1; k++)
                            (*plmat)[{pi, p++}] = b_op[j][k][s];
                // 2*2 : m / c d
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        (*plmat)[{pi, p++}] = c_op[j][s];
                    for (uint16_t j = 0; j < m + 1; j++)
                        (*plmat)[{pi, p++}] = d_op[j][s];
                }
            } else if (pm != n_sites - 1) {
                int pi = 0;
                // 1*4 : mm / cd
                for (uint8_t s = 0; s < 4; s++)
                    (*plmat)[{pi, p + s}] = b_op[m][m][s];
                p += 4;
                // 2*2 : m / c d
                for (uint8_t s = 0; s < 2; s++) {
                    (*plmat)[{pi, p++}] = c_op[m][s];
                    (*plmat)[{pi, p++}] = d_op[m][s];
                }
            }
            assert(p == lrshape);
            // right operator names
            //   1 : identity
            //   2*2*(n-m) : j / c d (j >= m)
            //   1*4 : mm / cd (only last site)
            int rlshape = pm == 0 ? 1
                                  : (pm != n_sites - 1
                                         ? (n_orbs - m) * 4 + 1
                                         : (n_orbs - m) * 4 + 1 +
                                               (n_orbs - m) * (n_orbs - m) * 4);
            int rrshape = pm != n_sites - 2
                              ? (n_orbs - m - 1) * 4 + 1
                              : (n_orbs - m - 1) * 4 + 1 +
                                    (n_orbs - m - 1) * (n_orbs - m - 1) * 4;
            if (pm == 0)
                prmat = make_shared<SymbolicRowVector<S>>(rrshape);
            else if (pm == n_sites - 1)
                prmat = make_shared<SymbolicColumnVector<S>>(rlshape);
            else
                prmat = make_shared<SymbolicMatrix<S>>(rlshape, rrshape);
            (*prmat)[{0, 0}] = i_op;
            p = 1;
            if (pm != 0 && pm != n_sites - 1) {
                int pi = 0;
                int pc[2] = {1 - (m + 1), 1 + 2 * (n_orbs - m - 1) - (m + 1)},
                    pd[2] = {1 + (n_orbs - m - 1) - (m + 1),
                             1 + 3 * (n_orbs - m - 1) - (m + 1)};
                // 2*2*(n-m) : j / c d (j >= m)
                for (uint8_t s = 0; s < 2; s++) {
                    (*prmat)[{p, pi}] = c_op[m][s];
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        (*prmat)[{p + j - m, pc[s] + j}] = i_op;
                    p += n_orbs - m;
                    (*prmat)[{p, pi}] = d_op[m][s];
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        (*prmat)[{p + j - m, pd[s] + j}] = i_op;
                    p += n_orbs - m;
                }
            } else if (pm == n_sites - 1) {
                int pi = 0;
                // 2*2*(n-m) : j / c d (j >= m)
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_orbs; j++)
                        (*prmat)[{p + j - m, pi}] = c_op[j][s];
                    p += n_orbs - m;
                    for (uint16_t j = m; j < n_orbs; j++)
                        (*prmat)[{p + j - m, pi}] = d_op[j][s];
                    p += n_orbs - m;
                }
                // 1*4 : mm / cd (only last site)
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = m; j < n_orbs; j++)
                        for (uint16_t k = m; k < n_orbs; k++)
                            (*prmat)[{p++, pi}] = b_op[j][k][s];
            }
            assert(p == rlshape);
            opt->lmat = plmat, opt->rmat = prmat;
            hamil->filter_site_ops(pm, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
    template <typename FL>
    static GMatrix<FL> get_matrix(
        const vector<vector<pair<shared_ptr<OpExpr<S>>, FL>>> &expectations,
        uint16_t n_orbs) {
        GMatrix<FL> r(nullptr, n_orbs * 2, n_orbs * 2);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                r(2 * op->site_index[0] + op->site_index.s(0),
                  2 * op->site_index[1] + op->site_index.s(1)) = x.second;
            }
        return r;
    }
    template <typename FL>
    static GMatrix<FL> get_matrix_spatial(
        const vector<vector<pair<shared_ptr<OpExpr<S>>, FL>>> &expectations,
        uint16_t n_orbs) {
        GMatrix<FL> r(nullptr, n_orbs, n_orbs);
        r.allocate();
        r.clear();
        GMatrix<FL> t = get_matrix(expectations, n_orbs);
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++)
                r(i, j) = t(2 * i + 0, 2 * j + 0) + t(2 * i + 1, 2 * j + 1);
        t.deallocate();
        return r;
    }
};

// "MPO" for one particle density matrix (spin-adapted)
// ds = 0 (default) normal 1pdm:
//     dm[i, j]    = < a^\dagger_{ia} a^{ja} + a^\dagger_{ib} a^{jb} >
//                 = sqrt(2) < a^{\dagger[1/2]}_i \otimes_[0] a^{[1/2]}_j >
//                 = sqrt(2) < a^{[1/2]}_j \otimes_[0] a^{\dagger[1/2]}_i >
// ds = 1 (spin-orbit) triplet excitation operators
//     dm[i, j]    =      sqrt(2) < a^{\dagger[1/2]}_i \otimes_[1] a^{[1/2]}_j >
//                 = (-1) sqrt(2) < a^{[1/2]}_j \otimes_[1] a^{\dagger[1/2]}_i >
template <typename S> struct PDM1MPOQC<S, typename S::is_su2_t> : MPO<S> {
    PDM1MPOQC(const shared_ptr<Hamiltonian<S>> &hamil, uint8_t ds = 0)
        : MPO<S>(hamil->n_sites) {
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil->vacuum);
        shared_ptr<OpElement<S>> zero_op = make_shared<OpElement<S>>(
            OpNames::Zero, SiteIndex(), S(0, ds * 2, 0));
        assert(ds == 0 || ds == 1);
        const auto n_sites = MPO<S>::n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
#ifdef _MSC_VER
        vector<shared_ptr<OpExpr<S>>> c_op(n_orbs), d_op(n_orbs);
        vector<vector<shared_ptr<OpExpr<S>>>> b_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(n_orbs));
        vector<vector<shared_ptr<OpExpr<S>>>> pdm1_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(n_orbs));
#else
        shared_ptr<OpExpr<S>> c_op[n_orbs], d_op[n_orbs];
        shared_ptr<OpExpr<S>> b_op[n_orbs][n_orbs];
        shared_ptr<OpExpr<S>> pdm1_op[n_orbs][n_orbs];
#endif
        for (uint16_t m = 0; m < n_orbs; m++) {
            c_op[m] = make_shared<OpElement<S>>(OpNames::C, SiteIndex(m),
                                                S(1, 1, hamil->orb_sym[m]));
            d_op[m] = make_shared<OpElement<S>>(OpNames::D, SiteIndex(m),
                                                S(-1, 1, hamil->orb_sym[m]));
        }
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++) {
                b_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(i, j, ds),
                    S(0, ds * 2, hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                pdm1_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::PDM1, SiteIndex(i, j),
                    S(0, ds * 2, hamil->orb_sym[i] ^ hamil->orb_sym[j]));
            }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil->opf);
        MPO<S>::site_op_infos = hamil->site_op_infos;
        for (uint16_t pm = 0; pm < n_sites; pm++) {
            uint16_t m = pm + n_orbs_big_left - 1;
            // left operator names
            //   1 : identity
            //   1 : mm / cd
            //   2 : m / c d
            // right operator names
            //   1 : identity
            //   2*(n-m) : j / c d (j >= m)
            //   1 : mm / cd (only last site)
            int lshape, rshape;
            if (pm == 0)
                lshape = 1 + 2 * (m + 1) + 1 * (m + 1) * (m + 1);
            else if (pm == n_sites - 1)
                lshape = 1;
            else
                lshape = 1 + 1 + 2;
            if (pm == 0)
                rshape = 1;
            else if (pm == n_sites - 1)
                rshape = 1 + 2 * (n_orbs - m) + 1 * (n_orbs - m) * (n_orbs - m);
            else
                rshape = 1 + 2 * (n_orbs - m);
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            // 1 : identity
            (*plop)[0] = i_op;
            int p = 1;
            if (pm == 0) {
                // 1 : mm / cd
                for (uint16_t j = 0; j < m + 1; j++)
                    for (uint16_t k = 0; k < m + 1; k++)
                        (*plop)[p++] = b_op[j][k];
                // 2 : m / c d
                for (uint16_t j = 0; j < m + 1; j++)
                    (*plop)[p++] = c_op[j];
                for (uint16_t j = 0; j < m + 1; j++)
                    (*plop)[p++] = d_op[j];
            } else if (pm != n_sites - 1) {
                // 1 : mm / cd
                (*plop)[p++] = b_op[m][m];
                // 2 : m / c d
                (*plop)[p++] = c_op[m];
                (*plop)[p++] = d_op[m];
            }
            assert(p == lshape);
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            // 1 : identity
            (*prop)[0] = i_op;
            p = 1;
            if (pm != 0) {
                // 2*(n-m) : j / c d (j >= m)
                for (uint16_t j = m; j < n_orbs; j++)
                    (*prop)[p + j - m] = c_op[j];
                p += n_orbs - m;
                for (uint16_t j = m; j < n_orbs; j++)
                    (*prop)[p + j - m] = d_op[j];
                p += n_orbs - m;
            }
            if (pm == n_sites - 1) {
                // 1 : mm / cd (only last site)
                for (uint16_t j = m; j < n_orbs; j++)
                    for (uint16_t k = m; k < n_orbs; k++)
                        (*prop)[p++] = b_op[j][k];
            }
            assert(p == rshape);
            this->right_operator_names.push_back(prop);
            // middle operators
            //   1*1 : mm / cd
            //   2*(n-m-1) : mj(-jm) / cd dc (j > m)
            //   1*1 : jj / cd (j > m) (last site only)
            if (pm != n_sites - 1) {
                int mshape;
                if (pm == 0)
                    mshape =
                        1 * (m + 1) * (m + 1) + 2 * (m + 1) * (n_orbs - m - 1);
                else
                    mshape = 1 + 2 * (n_orbs - m - 1);
                if (pm == n_sites - 2)
                    mshape += 1 * (n_orbs - m - 1) * (n_orbs - m - 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                p = 0;
                if (pm == 0) {
                    // 1*1 : mm / cd
                    for (uint16_t j = 0; j < m + 1; j++)
                        for (uint16_t k = 0; k < m + 1; k++) {
                            (*pmop)[p] = pdm1_op[j][k];
                            (*pmexpr)[p] = sqrt(2.0) * (b_op[j][k] * i_op);
                            p++;
                        }
                    // 2*(n-m-1) : mj(-jm) / cd dc (j > m)
                    for (uint16_t k = 0; k < m + 1; k++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            (*pmop)[p] = pdm1_op[k][j];
                            (*pmexpr)[p] = sqrt(2.0) * (c_op[k] * d_op[j]);
                            p++;
                            (*pmop)[p] = pdm1_op[j][k];
                            (*pmexpr)[p] = (ds ? -sqrt(2.0) : sqrt(2.0)) *
                                           (d_op[k] * c_op[j]);
                            p++;
                        }
                } else {
                    // 1*1 : mm / cd
                    (*pmop)[p] = pdm1_op[m][m];
                    (*pmexpr)[p] = sqrt(2.0) * (b_op[m][m] * i_op);
                    p++;
                    // 2*(n-m-1) : mj(-jm) / cd dc (j > m)
                    for (uint16_t j = m + 1; j < n_orbs; j++) {
                        (*pmop)[p] = pdm1_op[m][j];
                        (*pmexpr)[p] = sqrt(2.0) * (c_op[m] * d_op[j]);
                        p++;
                        (*pmop)[p] = pdm1_op[j][m];
                        (*pmexpr)[p] =
                            (ds ? -sqrt(2.0) : sqrt(2.0)) * (d_op[m] * c_op[j]);
                        p++;
                    }
                }
                // 1*1 : jj / cd (j > m) (last site only)
                if (pm == n_sites - 2)
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        for (uint16_t k = m + 1; k < n_orbs; k++) {
                            (*pmop)[p] = pdm1_op[j][k];
                            (*pmexpr)[p] = sqrt(2.0) * (i_op * b_op[j][k]);
                            p++;
                        }
                assert(p == mshape);
                this->middle_operator_names.push_back(pmop);
                this->middle_operator_exprs.push_back(pmexpr);
            }
            // site tensors
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            // left operator names
            //   1 : identity
            //   1 : mm / cd
            //   2 : m / c d
            int llshape = pm == 1 ? 1 + 2 * m + 1 * m * m : 1 + 2 + 1;
            int lrshape = pm == 0 ? 1 + 2 * (m + 1) + 1 * (m + 1) * (m + 1)
                                  : (pm != n_sites - 1 ? 1 + 2 + 1 : 1);
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (pm == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (pm == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            p = 1;
            if (pm == 0) {
                int pi = 0;
                // 1*4 : mm / cd
                for (uint16_t j = 0; j < m + 1; j++)
                    for (uint16_t k = 0; k < m + 1; k++)
                        (*plmat)[{pi, p++}] = b_op[j][k];
                // 2*2 : m / c d
                for (uint16_t j = 0; j < m + 1; j++)
                    (*plmat)[{pi, p++}] = c_op[j];
                for (uint16_t j = 0; j < m + 1; j++)
                    (*plmat)[{pi, p++}] = d_op[j];
            } else if (pm != n_sites - 1) {
                int pi = 0;
                // 1 : mm / cd
                (*plmat)[{pi, p++}] = b_op[m][m];
                // 2 : m / c d
                (*plmat)[{pi, p++}] = c_op[m];
                (*plmat)[{pi, p++}] = d_op[m];
            }
            assert(p == lrshape);
            // right operator names
            //   1 : identity
            //   2*(n-m) : j / c d (j >= m)
            //   1 : mm / cd (only last site)
            int rlshape = pm == 0 ? 1
                                  : (pm != n_sites - 1
                                         ? (n_orbs - m) * 2 + 1
                                         : (n_orbs - m) * 2 + 1 +
                                               (n_orbs - m) * (n_orbs - m) * 1);
            int rrshape = pm != n_sites - 2
                              ? (n_orbs - m - 1) * 2 + 1
                              : (n_orbs - m - 1) * 2 + 1 +
                                    (n_orbs - m - 1) * (n_orbs - m - 1) * 1;
            if (pm == 0)
                prmat = make_shared<SymbolicRowVector<S>>(rrshape);
            else if (pm == n_sites - 1)
                prmat = make_shared<SymbolicColumnVector<S>>(rlshape);
            else
                prmat = make_shared<SymbolicMatrix<S>>(rlshape, rrshape);
            (*prmat)[{0, 0}] = i_op;
            p = 1;
            if (pm != 0 && pm != n_sites - 1) {
                int pi = 0;
                int pc = 1 - (m + 1);
                int pd = 1 + (n_orbs - m - 1) - (m + 1);
                // 2*(n-m) : j / c d (j >= m)
                (*prmat)[{p, pi}] = c_op[m];
                for (uint16_t j = m + 1; j < n_orbs; j++)
                    (*prmat)[{p + j - m, pc + j}] = i_op;
                p += n_orbs - m;
                (*prmat)[{p, pi}] = d_op[m];
                for (uint16_t j = m + 1; j < n_orbs; j++)
                    (*prmat)[{p + j - m, pd + j}] = i_op;
                p += n_orbs - m;
            } else if (pm == n_sites - 1) {
                int pi = 0;
                // 2*(n-m) : j / c d (j >= m)
                for (uint16_t j = m; j < n_orbs; j++)
                    (*prmat)[{p + j - m, pi}] = c_op[j];
                p += n_orbs - m;
                for (uint16_t j = m; j < n_orbs; j++)
                    (*prmat)[{p + j - m, pi}] = d_op[m];
                p += n_orbs - m;
                // 1 : mm / cd (only last site)
                for (uint16_t j = m; j < n_orbs; j++)
                    for (uint16_t k = m; k < n_orbs; k++)
                        (*prmat)[{p++, pi}] = b_op[j][k];
            }
            assert(p == rlshape);
            opt->lmat = plmat, opt->rmat = prmat;
            hamil->filter_site_ops(pm, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
    // only for singlet
    template <typename FL>
    static GMatrix<FL> get_matrix(
        const vector<vector<pair<shared_ptr<OpExpr<S>>, FL>>> &expectations,
        uint16_t n_orbs) {
        GMatrix<FL> r(nullptr, n_orbs * 2, n_orbs * 2);
        r.allocate();
        r.clear();
        GMatrix<FL> t = get_matrix_spatial(expectations, n_orbs);
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++) {
                r(2 * i + 0, 2 * j + 0) = t(i, j) / 2.0;
                r(2 * i + 1, 2 * j + 1) = t(i, j) / 2.0;
            }
        t.deallocate();
        return r;
    }
    template <typename FL>
    static GMatrix<FL> get_matrix_spatial(
        const vector<vector<pair<shared_ptr<OpExpr<S>>, FL>>> &expectations,
        uint16_t n_orbs) {
        GMatrix<FL> r(nullptr, n_orbs, n_orbs);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                r(op->site_index[0], op->site_index[1]) = x.second;
            }
        return r;
    }
};

} // namespace block2
