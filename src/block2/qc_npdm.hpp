
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
#include "hamiltonian.hpp"
#include "mpo.hpp"
#include "operator_tensor.hpp"
#include "symbolic.hpp"
#include "tensor_functions.hpp"
#include <cassert>
#include <memory>

using namespace std;

namespace block2 {

template <typename, typename = void> struct PDM1MPOQC;

// "MPO" for one particle density matrix (non-spin-adapted)
template <typename S> struct PDM1MPOQC<S, typename S::is_sz_t> : MPO<S> {
    PDM1MPOQC(const Hamiltonian<S> &hamil) : MPO<S>(hamil.n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vacuum);
        shared_ptr<OpElement<S>> zero_op =
            make_shared<OpElement<S>>(OpNames::Zero, SiteIndex(), hamil.vacuum);
        shared_ptr<OpExpr<S>> c_op[n_sites][2], d_op[n_sites][2];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites][4];
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t m = 0; m < n_sites; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil.orb_sym[m]));
                d_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil.orb_sym[m]));
            }
        for (uint8_t i = 0; i < n_sites; i++)
            for (uint8_t j = 0; j < n_sites; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    // for i > j, use spin of i <= j to make the quantum number matched
                    // tranpose is not important since only expectation value matters
                    // not working for diff bra ket!
                    // TODO: keep more terms here and use simplification
                    pdm1_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PDM1, sidx,
                        S(0, i <= j ? sz_minus[s] : -sz_minus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        for (uint8_t m = 0; m < n_sites; m++) {
            int lshape = m != n_sites - 1 ? 1 + 6 * (m + 1) : 1;
            int rshape = m != n_sites - 1 ? 1 : 7;
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            (*plop)[0] = i_op;
            if (m != n_sites - 1)
                for (uint8_t j = 0; j <= m; j++) {
                    for (uint8_t s = 0; s < 2; s++)
                        (*plop)[1 + (m + 1) * s + j] = c_op[j][s];
                    for (uint8_t s = 0; s < 4; s++)
                        (*plop)[1 + (m + 1) * (2 + s) + j] = b_op[j][m][s];
                }
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            (*prop)[0] = i_op;
            if (m == n_sites - 1) {
                for (uint8_t s = 0; s < 4; s++)
                    (*prop)[1 + s] = b_op[m][m][s];
                for (uint8_t s = 0; s < 2; s++)
                    (*prop)[5 + s] = d_op[m][s];
            }
            this->right_operator_names.push_back(prop);
            // middle operators
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 4 * (2 * m + 1) : 16 * (m + 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                int p = 0;
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint8_t j = 0; j <= m; j++) {
                        shared_ptr<OpExpr<S>> expr = b_op[j][m][s] * i_op;
                        (*pmop)[p + 2 * j] = pdm1_op[j][m][s];
                        (*pmexpr)[p + 2 * j] = expr;
                        if (j != m) {
                            (*pmop)[p + 2 * j + 1] =
                                pdm1_op[m][j][((s & 1) << 1) | (s >> 1)];
                            (*pmexpr)[p + 2 * j + 1] = expr;
                        }
                    }
                    p += 2 * m + 1;
                }
                if (m == n_sites - 2) {
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint8_t j = 0; j <= m; j++) {
                            shared_ptr<OpExpr<S>> expr =
                                c_op[j][s & 1] * d_op[m + 1][s >> 1];
                            (*pmop)[p + 2 * j] = pdm1_op[j][m + 1][s];
                            (*pmop)[p + 2 * j + 1] =
                                pdm1_op[m + 1][j][((s & 1) << 1) | (s >> 1)];
                            (*pmexpr)[p + 2 * j] = (*pmexpr)[p + 2 * j + 1] =
                                expr;
                        }
                        (*pmop)[p + 2 * (m + 1)] = pdm1_op[m + 1][m + 1][s];
                        (*pmexpr)[p + 2 * (m + 1)] =
                            i_op * b_op[m + 1][m + 1][s];
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
            int llshape = 1 + 6 * m;
            int lrshape = m != n_sites - 1 ? 1 + 6 * (m + 1) : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            if (m != n_sites - 1) {
                int pi = 0, pc[2] = {1, 1 + m}, p = 1;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint8_t i = 0; i < m; i++)
                        (*plmat)[{pc[s] + i, p + i}] = i_op;
                    (*plmat)[{pi, p + m}] = c_op[m][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint8_t i = 0; i < m; i++)
                        (*plmat)[{pc[s & 1] + i, p + i}] = d_op[m][s >> 1];
                    (*plmat)[{pi, p + m}] = b_op[m][m][s];
                    p += m + 1;
                }
                assert(p == lrshape);
            }
            if (m == n_sites - 1) {
                prmat = make_shared<SymbolicColumnVector<S>>(7);
                prmat->data[0] = i_op;
                for (uint8_t s = 0; s < 4; s++)
                    prmat->data[1 + s] = b_op[m][m][s];
                for (uint8_t s = 0; s < 2; s++)
                    prmat->data[5 + s] = d_op[m][s];
            } else {
                if (m == n_sites - 2)
                    prmat = make_shared<SymbolicMatrix<S>>(1, 7);
                else if (m == 0)
                    prmat = make_shared<SymbolicRowVector<S>>(1);
                else
                    prmat = make_shared<SymbolicMatrix<S>>(1, 1);
                (*prmat)[{0, 0}] = i_op;
            }
            opt->lmat = plmat, opt->rmat = prmat;
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
};

// "MPO" for one particle density matrix (spin-adapted)
template <typename S> struct PDM1MPOQC<S, typename S::is_su2_t> : MPO<S> {
    PDM1MPOQC(const Hamiltonian<S> &hamil) : MPO<S>(hamil.n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vacuum);
        shared_ptr<OpElement<S>> zero_op =
            make_shared<OpElement<S>>(OpNames::Zero, SiteIndex(), hamil.vacuum);
        shared_ptr<OpExpr<S>> c_op[n_sites], d_op[n_sites];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites];
        for (uint8_t m = 0; m < n_sites; m++) {
            c_op[m] = make_shared<OpElement<S>>(OpNames::C, SiteIndex(m),
                                                S(1, 1, hamil.orb_sym[m]));
            d_op[m] = make_shared<OpElement<S>>(OpNames::D, SiteIndex(m),
                                                S(-1, 1, hamil.orb_sym[m]));
        }
        for (uint8_t i = 0; i < n_sites; i++)
            for (uint8_t j = 0; j < n_sites; j++) {
                b_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(i, j, 0),
                    S(0, 0, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                pdm1_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::PDM1, SiteIndex(i, j),
                    S(0, 0, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
            }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        for (uint8_t m = 0; m < n_sites; m++) {
            int lshape = m != n_sites - 1 ? 1 + 2 * (m + 1) : 1;
            int rshape = m != n_sites - 1 ? 1 : 3;
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            (*plop)[0] = i_op;
            if (m != n_sites - 1)
                for (uint8_t j = 0; j <= m; j++)
                    (*plop)[1 + j] = c_op[j],
                                (*plop)[1 + (m + 1) + j] = b_op[j][m];
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            (*prop)[0] = i_op;
            if (m == n_sites - 1)
                (*prop)[1] = b_op[m][m], (*prop)[2] = d_op[m];
            this->right_operator_names.push_back(prop);
            // middle operators
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 2 * m + 1 : 4 * (m + 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                for (uint8_t j = 0; j <= m; j++) {
                    shared_ptr<OpExpr<S>> expr =
                        sqrt(2.0) * (b_op[j][m] * i_op);
                    (*pmop)[2 * j] = pdm1_op[m][j], (*pmexpr)[2 * j] = expr;
                    if (j != m)
                        (*pmop)[2 * j + 1] = pdm1_op[j][m],
                                        (*pmexpr)[2 * j + 1] = expr;
                }
                if (m == n_sites - 2) {
                    int p = 2 * m + 1;
                    for (uint8_t j = 0; j <= m; j++) {
                        shared_ptr<OpExpr<S>> expr =
                            sqrt(2.0) * (c_op[j] * d_op[m + 1]);
                        (*pmop)[p + 2 * j] = pdm1_op[j][m + 1];
                        (*pmop)[p + 2 * j + 1] = pdm1_op[m + 1][j];
                        (*pmexpr)[p + 2 * j] = (*pmexpr)[p + 2 * j + 1] = expr;
                    }
                    p += 2 * (m + 1);
                    (*pmop)[p] = pdm1_op[m + 1][m + 1];
                    (*pmexpr)[p] = sqrt(2.0) * (i_op * b_op[m + 1][m + 1]);
                }
                this->middle_operator_names.push_back(pmop);
                this->middle_operator_exprs.push_back(pmexpr);
            }
            // site tensors
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            int llshape = 1 + 2 * m;
            int lrshape = m != n_sites - 1 ? 1 + 2 * (m + 1) : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            if (m != n_sites - 1) {
                int pi = 0, pc = 1, p = 1;
                for (uint8_t i = 0; i < m; i++)
                    (*plmat)[{pc + i, p + i}] = i_op;
                (*plmat)[{pi, p + m}] = c_op[m];
                p += m + 1;
                for (uint8_t i = 0; i < m; i++)
                    (*plmat)[{pc + i, p + i}] = d_op[m];
                (*plmat)[{pi, p + m}] = b_op[m][m];
                p += m + 1;
                assert(p == lrshape);
            }
            if (m == n_sites - 1) {
                prmat = make_shared<SymbolicColumnVector<S>>(3);
                prmat->data[0] = i_op;
                prmat->data[1] = b_op[m][m];
                prmat->data[2] = d_op[m];
            } else {
                if (m == n_sites - 2)
                    prmat = make_shared<SymbolicMatrix<S>>(1, 3);
                else if (m == 0)
                    prmat = make_shared<SymbolicRowVector<S>>(1);
                else
                    prmat = make_shared<SymbolicMatrix<S>>(1, 1);
                (*prmat)[{0, 0}] = i_op;
            }
            opt->lmat = plmat, opt->rmat = prmat;
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
};

} // namespace block2