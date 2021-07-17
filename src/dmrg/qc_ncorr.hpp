
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

template <typename, typename = void> struct NPC1MPOQC;

// "MPO" for charge/spin correlation (non-spin-adapted)
// NN[0~3] = n_{p,sp} x n_{q,sq}
// NN[4] = ad_{pa} a_{pb} x ad_{qb} a_{qa}
// NN[5] = ad_{pb} a_{pa} x ad_{qa} a_{qb}
template <typename S> struct NPC1MPOQC<S, typename S::is_sz_t> : MPO<S> {
    NPC1MPOQC(const shared_ptr<Hamiltonian<S>> &hamil)
        : MPO<S>(hamil->n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil->vacuum);
        shared_ptr<OpElement<S>> zero_op = make_shared<OpElement<S>>(
            OpNames::Zero, SiteIndex(), hamil->vacuum);
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> b_op(
            n_sites, vector<shared_ptr<OpExpr<S>>>(4));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> nn_op(
            n_sites, vector<vector<shared_ptr<OpExpr<S>>>>(
                         n_sites, vector<shared_ptr<OpExpr<S>>>(6)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> pdm1_op(
            n_sites, vector<vector<shared_ptr<OpExpr<S>>>>(
                         n_sites, vector<shared_ptr<OpExpr<S>>>(6)));
#else
        shared_ptr<OpExpr<S>> b_op[n_sites][4];
        shared_ptr<OpExpr<S>> nn_op[n_sites][n_sites][6];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites][6];
#endif
        const int sz_minus[4] = {0, -2, 2, 0};
        for (uint16_t m = 0; m < n_sites; m++)
            for (uint8_t s = 0; s < 4; s++)
                b_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::B,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, sz_minus[s], 0));
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++) {
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    nn_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::NN, sidx, hamil->vacuum);
                    pdm1_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PDM1, sidx, hamil->vacuum);
                }
                for (uint8_t s = 0; s < 2; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)s, (uint8_t)0, (uint8_t)1});
                    nn_op[i][j][4 + s] = make_shared<OpElement<S>>(
                        OpNames::NN, sidx, hamil->vacuum);
                    pdm1_op[i][j][4 + s] = make_shared<OpElement<S>>(
                        OpNames::PDM1, sidx, hamil->vacuum);
                }
            }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil->opf);
        MPO<S>::site_op_infos = hamil->site_op_infos;
        for (uint16_t m = 0; m < n_sites; m++) {
            int lshape = m != n_sites - 1 ? 1 + 10 * (m + 1) : 1;
            int rshape = m != n_sites - 1 ? 1 : 11;
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            (*plop)[0] = i_op;
            if (m != n_sites - 1)
                for (uint16_t j = 0; j <= m; j++) {
                    for (uint8_t s = 0; s < 4; s++)
                        (*plop)[1 + (m + 1) * s + j] = b_op[j][s];
                    for (uint8_t s = 0; s < 6; s++)
                        (*plop)[1 + (m + 1) * (4 + s) + j] = nn_op[j][m][s];
                }
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            (*prop)[0] = i_op;
            if (m == n_sites - 1) {
                for (uint8_t s = 0; s < 6; s++)
                    (*prop)[1 + s] = nn_op[m][m][s];
                for (uint8_t s = 0; s < 4; s++)
                    (*prop)[7 + s] = b_op[m][s];
            }
            this->right_operator_names.push_back(prop);
            // middle operators
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 6 * (2 * m + 1) : 24 * (m + 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                int p = 0;
                for (uint8_t s = 0; s < 6; s++) {
                    for (uint16_t j = 0; j <= m; j++) {
                        shared_ptr<OpExpr<S>> expr = nn_op[j][m][s] * i_op;
                        (*pmop)[p + 2 * j] = pdm1_op[j][m][s];
                        (*pmexpr)[p + 2 * j] = expr;
                        if (j != m) {
                            (*pmop)[p + 2 * j + 1] =
                                s < 4 ? pdm1_op[m][j][((s & 1) << 1) | (s >> 1)]
                                      : pdm1_op[m][j][s ^ 1];
                            (*pmexpr)[p + 2 * j + 1] = expr;
                        }
                    }
                    p += 2 * m + 1;
                }
                if (m == n_sites - 2) {
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint16_t j = 0; j <= m; j++) {
                            shared_ptr<OpExpr<S>> expr =
                                b_op[j][(s & 1) | ((s & 1) << 1)] *
                                b_op[m + 1][(s >> 1) | ((s >> 1) << 1)];
                            (*pmop)[p + 2 * j] = pdm1_op[j][m + 1][s];
                            (*pmop)[p + 2 * j + 1] =
                                pdm1_op[m + 1][j][((s & 1) << 1) | (s >> 1)];
                            (*pmexpr)[p + 2 * j] = (*pmexpr)[p + 2 * j + 1] =
                                expr;
                        }
                        (*pmop)[p + 2 * (m + 1)] = pdm1_op[m + 1][m + 1][s];
                        (*pmexpr)[p + 2 * (m + 1)] =
                            i_op * nn_op[m + 1][m + 1][s];
                        p += 2 * m + 3;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j <= m; j++) {
                            shared_ptr<OpExpr<S>> expr =
                                b_op[j][s | ((!s) << 1)] *
                                b_op[m + 1][(!s) | (s << 1)];
                            (*pmop)[p + 2 * j] = pdm1_op[j][m + 1][s + 4];
                            (*pmop)[p + 2 * j + 1] =
                                pdm1_op[m + 1][j][(s + 4) ^ 1];
                            (*pmexpr)[p + 2 * j] = (*pmexpr)[p + 2 * j + 1] =
                                expr;
                        }
                        (*pmop)[p + 2 * (m + 1)] = pdm1_op[m + 1][m + 1][s + 4];
                        (*pmexpr)[p + 2 * (m + 1)] =
                            i_op * nn_op[m + 1][m + 1][s + 4];
                        p += 2 * m + 3;
                    }
                    assert(p == mshape);
                }
                this->middle_operator_names.push_back(pmop);
                this->middle_operator_exprs.push_back(pmexpr);
            }
            // site tensors
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            int llshape = 1 + 10 * m;
            int lrshape = m != n_sites - 1 ? 1 + 10 * (m + 1) : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            if (m != n_sites - 1) {
                int pi = 0, pb[4] = {1, 1 + m, 1 + m + m, 1 + m + m + m}, p = 1;
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        (*plmat)[{pb[s] + i, p + i}] = i_op;
                    (*plmat)[{pi, p + m}] = b_op[m][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        (*plmat)[{pb[(s & 1) | ((s & 1) << 1)] + i, p + i}] =
                            b_op[m][(s >> 1) | ((s >> 1) << 1)];
                    (*plmat)[{pi, p + m}] = nn_op[m][m][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        (*plmat)[{pb[s | ((!s) << 1)] + i, p + i}] =
                            b_op[m][(!s) | (s << 1)];
                    (*plmat)[{pi, p + m}] = nn_op[m][m][s + 4];
                    p += m + 1;
                }
                assert(p == lrshape);
            }
            if (m == n_sites - 1) {
                prmat = make_shared<SymbolicColumnVector<S>>(11);
                prmat->data[0] = i_op;
                for (uint8_t s = 0; s < 6; s++)
                    prmat->data[1 + s] = nn_op[m][m][s];
                for (uint8_t s = 0; s < 4; s++)
                    prmat->data[7 + s] = b_op[m][s];
            } else {
                if (m == n_sites - 2)
                    prmat = make_shared<SymbolicMatrix<S>>(1, 11);
                else if (m == 0)
                    prmat = make_shared<SymbolicRowVector<S>>(1);
                else
                    prmat = make_shared<SymbolicMatrix<S>>(1, 1);
                (*prmat)[{0, 0}] = i_op;
            }
            opt->lmat = plmat, opt->rmat = prmat;
            hamil->filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
};

// "MPO" for charge/spin correlation (spin-adapted)
// NN[0] = 2 * B0 x B0
//   e_pqqp = NN[0] - delta_pq Epq
// NN[1] = -sqrt(3) * B1 x B1 + B0 x B0
//   e_pqpq = -NN[1] + 2 * delta_pq Epq
// where Epq = 1pdm spatial
template <typename S> struct NPC1MPOQC<S, typename S::is_su2_t> : MPO<S> {
    NPC1MPOQC(const shared_ptr<Hamiltonian<S>> &hamil)
        : MPO<S>(hamil->n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil->vacuum);
        shared_ptr<OpElement<S>> zero_op = make_shared<OpElement<S>>(
            OpNames::Zero, SiteIndex(), hamil->vacuum);
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> b_op(
            n_sites, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> nn_op(
            n_sites, vector<vector<shared_ptr<OpExpr<S>>>>(
                         n_sites, vector<shared_ptr<OpExpr<S>>>(2)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> pdm1_op(
            n_sites, vector<vector<shared_ptr<OpExpr<S>>>>(
                         n_sites, vector<shared_ptr<OpExpr<S>>>(2)));
#else
        shared_ptr<OpExpr<S>> b_op[n_sites][2];
        shared_ptr<OpExpr<S>> nn_op[n_sites][n_sites][2];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites][2];
#endif
        for (uint16_t m = 0; m < n_sites; m++)
            for (uint8_t s = 0; s < 2; s++)
                b_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(m, m, s), S(0, s * 2, 0));
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++)
                for (uint8_t s = 0; s < 2; s++) {
                    nn_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::NN, SiteIndex(i, j, s), hamil->vacuum);
                    pdm1_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PDM1, SiteIndex(i, j, s), hamil->vacuum);
                }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil->opf);
        MPO<S>::site_op_infos = hamil->site_op_infos;
        for (uint16_t m = 0; m < n_sites; m++) {
            int lshape = m != n_sites - 1 ? 1 + 4 * (m + 1) : 1;
            int rshape = m != n_sites - 1 ? 1 : 5;
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            (*plop)[0] = i_op;
            if (m != n_sites - 1)
                for (uint16_t j = 0; j <= m; j++) {
                    for (uint8_t s = 0; s < 2; s++)
                        (*plop)[1 + (m + 1) * s + j] = b_op[j][s];
                    for (uint8_t s = 0; s < 2; s++)
                        (*plop)[1 + (m + 1) * (2 + s) + j] = nn_op[j][m][s];
                }
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            (*prop)[0] = i_op;
            if (m == n_sites - 1) {
                (*prop)[1] = nn_op[m][m][0];
                (*prop)[2] = nn_op[m][m][1];
                (*prop)[3] = b_op[m][0];
                (*prop)[4] = b_op[m][1];
            }
            this->right_operator_names.push_back(prop);
            // middle operators
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 2 * (2 * m + 1) : 8 * (m + 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                int p = 0;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j <= m; j++) {
                        shared_ptr<OpExpr<S>> expr = nn_op[j][m][s] * i_op;
                        (*pmop)[p + 2 * j] = pdm1_op[m][j][s],
                                        (*pmexpr)[p + 2 * j] = expr;
                        if (j != m)
                            (*pmop)[p + 2 * j + 1] = pdm1_op[j][m][s],
                                                (*pmexpr)[p + 2 * j + 1] = expr;
                    }
                    p += 2 * m + 1;
                }
                if (m == n_sites - 2) {
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j <= m; j++) {
                            shared_ptr<OpExpr<S>> expr =
                                s == 0 ? 2.0 * (b_op[j][0] * b_op[m + 1][0])
                                       : (-sqrt(3.0)) *
                                                 (b_op[j][1] * b_op[m + 1][1]) +
                                             b_op[j][0] * b_op[m + 1][0];
                            (*pmop)[p + 2 * j] = pdm1_op[j][m + 1][s];
                            (*pmop)[p + 2 * j + 1] = pdm1_op[m + 1][j][s];
                            (*pmexpr)[p + 2 * j] = (*pmexpr)[p + 2 * j + 1] =
                                expr;
                        }
                        p += 2 * (m + 1);
                        (*pmop)[p] = pdm1_op[m + 1][m + 1][s];
                        (*pmexpr)[p] = i_op * nn_op[m + 1][m + 1][s];
                        p++;
                    }
                }
                this->middle_operator_names.push_back(pmop);
                this->middle_operator_exprs.push_back(pmexpr);
            }
            // site tensors
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            int llshape = 1 + 4 * m;
            int lrshape = m != n_sites - 1 ? 1 + 4 * (m + 1) : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            if (m != n_sites - 1) {
                int pi = 0, pb[2] = {1, 1 + m}, p = 1;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        (*plmat)[{pb[s] + i, p + i}] = i_op;
                    (*plmat)[{pi, p + m}] = b_op[m][s];
                    p += m + 1;
                }
                for (uint16_t i = 0; i < m; i++)
                    (*plmat)[{pb[0] + i, p + i}] = 2.0 * b_op[m][0];
                (*plmat)[{pi, p + m}] = nn_op[m][m][0];
                p += m + 1;
                for (uint16_t i = 0; i < m; i++) {
                    (*plmat)[{pb[0] + i, p + i}] = b_op[m][0];
                    (*plmat)[{pb[1] + i, p + i}] = (-sqrt(3.0)) * b_op[m][1];
                }
                (*plmat)[{pi, p + m}] = nn_op[m][m][1];
                p += m + 1;
                assert(p == lrshape);
            }
            if (m == n_sites - 1) {
                prmat = make_shared<SymbolicColumnVector<S>>(5);
                prmat->data[0] = i_op;
                prmat->data[1] = nn_op[m][m][0];
                prmat->data[2] = nn_op[m][m][1];
                prmat->data[3] = b_op[m][0];
                prmat->data[4] = b_op[m][1];
            } else {
                if (m == n_sites - 2)
                    prmat = make_shared<SymbolicMatrix<S>>(1, 5);
                else if (m == 0)
                    prmat = make_shared<SymbolicRowVector<S>>(1);
                else
                    prmat = make_shared<SymbolicMatrix<S>>(1, 1);
                (*prmat)[{0, 0}] = i_op;
            }
            opt->lmat = plmat, opt->rmat = prmat;
            hamil->filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
};

} // namespace block2
