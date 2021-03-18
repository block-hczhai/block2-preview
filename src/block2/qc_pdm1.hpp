
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
    PDM1MPOQC(const Hamiltonian<S> &hamil, uint8_t ds = 0)
        : MPO<S>(hamil.n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vacuum);
        shared_ptr<OpElement<S>> zero_op =
            make_shared<OpElement<S>>(OpNames::Zero, SiteIndex(), hamil.vacuum);
        assert(ds == 0);
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> c_op(
            n_sites, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> d_op(
            n_sites, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> b_op(
            n_sites, vector<vector<shared_ptr<OpExpr<S>>>>(
                         n_sites, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> pdm1_op(
            n_sites, vector<vector<shared_ptr<OpExpr<S>>>>(
                         n_sites, vector<shared_ptr<OpExpr<S>>>(4)));
#else
        shared_ptr<OpExpr<S>> c_op[n_sites][2], d_op[n_sites][2];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites][4];
#endif
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint16_t m = 0; m < n_sites; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil.orb_sym[m]));
                d_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil.orb_sym[m]));
            }
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    pdm1_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PDM1, sidx,
                        S(0, sz_minus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        for (uint16_t m = 0; m < n_sites; m++) {
            // left operator names
            //   1 : identity
            //   1*4 : mm / cd
            //   2*2 : m / c d
            // right operator names
            //   1 : identity
            //   2*2*(n-m) : j / c d (j >= m)
            //   1*4 : mm / cd (only last site)
            int lshape = m != n_sites - 1 ? 9 : 1;
            int rshape = m == 0 ? 1
                                : (m != n_sites - 1 ? (n_sites - m) * 4 + 1
                                                    : (n_sites - m) * 4 + 5);
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            // 1 : identity
            (*plop)[0] = i_op;
            int p = 1;
            if (m != n_sites - 1) {
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
            if (m != 0) {
                // 2*2*(n-m) : j / c d (j >= m)
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_sites; j++)
                        (*prop)[p + j - m] = c_op[j][s];
                    p += n_sites - m;
                    for (uint16_t j = m; j < n_sites; j++)
                        (*prop)[p + j - m] = d_op[j][s];
                    p += n_sites - m;
                }
            }
            if (m == n_sites - 1) {
                // 1*4 : mm / cd (only last site)
                for (uint8_t s = 0; s < 4; s++)
                    (*prop)[p + s] = b_op[m][m][s];
                p += 4;
            }
            assert(p == rshape);
            this->right_operator_names.push_back(prop);
            // middle operators
            //   1*4*1 : mm / cd
            //   2*4*(n-m-1) : mj(-jm) / cd dc (j > m)
            //   1*4*1 : jj / cd (j > m) (last site only)
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 4 + 8 * (n_sites - m - 1) : 16;
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                p = 0;
                for (uint8_t s = 0; s < 4; s++) {
                    // 1*4*1 : mm / cd
                    (*pmop)[p] = pdm1_op[m][m][s];
                    (*pmexpr)[p] = b_op[m][m][s] * i_op;
                    p++;
                    // 2*4*(n-m-1) : mj(-jm) / cd dc (j > m)
                    for (uint16_t j = m + 1; j < n_sites; j++) {
                        (*pmop)[p] = pdm1_op[m][j][s];
                        (*pmexpr)[p] = c_op[m][s & 1] * d_op[j][s >> 1];
                        p++;
                        (*pmop)[p] = pdm1_op[j][m][s];
                        (*pmexpr)[p] =
                            -1.0 * (d_op[m][s >> 1] * c_op[j][s & 1]);
                        p++;
                    }
                }
                // 1*4*1 : jj / cd (j > m) (last site only)
                if (m == n_sites - 2)
                    for (uint8_t s = 0; s < 4; s++) {
                        (*pmop)[p] = pdm1_op[m + 1][m + 1][s];
                        (*pmexpr)[p] = i_op * b_op[m + 1][m + 1][s];
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
            int llshape = 9;
            int lrshape = m != n_sites - 1 ? 9 : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            p = 1;
            if (m != n_sites - 1) {
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
            int rlshape = m == 0 ? 1
                                 : (m != n_sites - 1 ? (n_sites - m) * 4 + 1
                                                     : (n_sites - m) * 4 + 5);
            int rrshape = m != n_sites - 2 ? (n_sites - m - 1) * 4 + 1
                                           : (n_sites - m - 1) * 4 + 5;
            if (m == 0)
                prmat = make_shared<SymbolicRowVector<S>>(rrshape);
            else if (m == n_sites - 1)
                prmat = make_shared<SymbolicColumnVector<S>>(rlshape);
            else
                prmat = make_shared<SymbolicMatrix<S>>(rlshape, rrshape);
            (*prmat)[{0, 0}] = i_op;
            p = 1;
            if (m != 0) {
                int pi = 0;
                int pc[2] = {1 - (m + 1), 1 + 2 * (n_sites - m - 1) - (m + 1)},
                    pd[2] = {1 + (n_sites - m - 1) - (m + 1),
                             1 + 3 * (n_sites - m - 1) - (m + 1)};
                // 2*2*(n-m) : j / c d (j >= m)
                for (uint8_t s = 0; s < 2; s++) {
                    (*prmat)[{p, pi}] = c_op[m][s];
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        (*prmat)[{p + j - m, pc[s] + j}] = i_op;
                    p += n_sites - m;
                    (*prmat)[{p, pi}] = d_op[m][s];
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        (*prmat)[{p + j - m, pd[s] + j}] = i_op;
                    p += n_sites - m;
                }
            }
            if (m == n_sites - 1) {
                // 1*4 : mm / cd (only last site)
                for (uint8_t s = 0; s < 4; s++)
                    (*prmat)[{p + s, 0}] = b_op[m][m][s];
                p += 4;
            }
            assert(p == rlshape);
            opt->lmat = plmat, opt->rmat = prmat;
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
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
    PDM1MPOQC(const Hamiltonian<S> &hamil, uint8_t ds = 0)
        : MPO<S>(hamil.n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vacuum);
        shared_ptr<OpElement<S>> zero_op = make_shared<OpElement<S>>(
            OpNames::Zero, SiteIndex(), S(0, ds * 2, 0));
        assert(ds == 0 || ds == 1);
#ifdef _MSC_VER
        vector<shared_ptr<OpExpr<S>>> c_op(n_sites), d_op(n_sites);
        vector<vector<shared_ptr<OpExpr<S>>>> b_op(
            n_sites, vector<shared_ptr<OpExpr<S>>>(n_sites));
        vector<vector<shared_ptr<OpExpr<S>>>> pdm1_op(
            n_sites, vector<shared_ptr<OpExpr<S>>>(n_sites));
#else
        shared_ptr<OpExpr<S>> c_op[n_sites], d_op[n_sites];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites];
#endif
        for (uint16_t m = 0; m < n_sites; m++) {
            c_op[m] = make_shared<OpElement<S>>(OpNames::C, SiteIndex(m),
                                                S(1, 1, hamil.orb_sym[m]));
            d_op[m] = make_shared<OpElement<S>>(OpNames::D, SiteIndex(m),
                                                S(-1, 1, hamil.orb_sym[m]));
        }
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++) {
                b_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(i, j, ds),
                    S(0, ds * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                pdm1_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::PDM1, SiteIndex(i, j),
                    S(0, ds * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
            }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        for (uint16_t m = 0; m < n_sites; m++) {
            // left operator names
            //   1 : identity
            //   1 : mm / cd
            //   2 : m / c d
            // right operator names
            //   1 : identity
            //   2*(n-m) : j / c d (j >= m)
            //   1 : mm / cd (only last site)
            int lshape = m != n_sites - 1 ? 4 : 1;
            int rshape = m == 0 ? 1
                                : (m != n_sites - 1 ? 2 * (n_sites - m) + 1
                                                    : 2 * (n_sites - m) + 2);
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            // 1 : identity
            (*plop)[0] = i_op;
            int p = 1;
            if (m != n_sites - 1) {
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
            if (m != 0) {
                // 2*(n-m) : j / c d (j >= m)
                for (uint16_t j = m; j < n_sites; j++)
                    (*prop)[p + j - m] = c_op[j];
                p += n_sites - m;
                for (uint16_t j = m; j < n_sites; j++)
                    (*prop)[p + j - m] = d_op[j];
                p += n_sites - m;
            }
            if (m == n_sites - 1)
                // 1 : mm / cd (only last site)
                (*prop)[p++] = b_op[m][m];
            assert(p == rshape);
            this->right_operator_names.push_back(prop);
            // middle operators
            //   1*1 : mm / cd
            //   2*(n-m-1) : mj(-jm) / cd dc (j > m)
            //   1*1 : jj / cd (j > m) (last site only)
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 1 + 2 * (n_sites - m - 1) : 4;
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                p = 0;
                // 1*1 : mm / cd
                (*pmop)[p] = pdm1_op[m][m];
                (*pmexpr)[p] = sqrt(2.0) * (b_op[m][m] * i_op);
                p++;
                // 2*(n-m-1) : mj(-jm) / cd dc (j > m)
                for (uint16_t j = m + 1; j < n_sites; j++) {
                    (*pmop)[p] = pdm1_op[m][j];
                    (*pmexpr)[p] = sqrt(2.0) * (c_op[m] * d_op[j]);
                    p++;
                    (*pmop)[p] = pdm1_op[j][m];
                    (*pmexpr)[p] =
                        (ds ? -sqrt(2.0) : sqrt(2.0)) * (d_op[m] * c_op[j]);
                    p++;
                }
                // 1*1 : jj / cd (j > m) (last site only)
                if (m == n_sites - 2) {
                    (*pmop)[p] = pdm1_op[m + 1][m + 1];
                    (*pmexpr)[p] = sqrt(2.0) * (i_op * b_op[m + 1][m + 1]);
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
            int llshape = 4;
            int lrshape = m != n_sites - 1 ? 4 : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            p = 1;
            if (m != n_sites - 1) {
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
            int rlshape = m == 0 ? 1
                                 : (m != n_sites - 1 ? (n_sites - m) * 2 + 1
                                                     : (n_sites - m) * 2 + 2);
            int rrshape = m != n_sites - 2 ? (n_sites - m - 1) * 2 + 1
                                           : (n_sites - m - 1) * 2 + 2;
            if (m == 0)
                prmat = make_shared<SymbolicRowVector<S>>(rrshape);
            else if (m == n_sites - 1)
                prmat = make_shared<SymbolicColumnVector<S>>(rlshape);
            else
                prmat = make_shared<SymbolicMatrix<S>>(rlshape, rrshape);
            (*prmat)[{0, 0}] = i_op;
            p = 1;
            if (m != 0) {
                int pi = 0;
                int pc = 1 - (m + 1);
                int pd = 1 + (n_sites - m - 1) - (m + 1);
                // 2*(n-m) : j / c d (j >= m)
                (*prmat)[{p, pi}] = c_op[m];
                for (uint16_t j = m + 1; j < n_sites; j++)
                    (*prmat)[{p + j - m, pc + j}] = i_op;
                p += n_sites - m;
                (*prmat)[{p, pi}] = d_op[m];
                for (uint16_t j = m + 1; j < n_sites; j++)
                    (*prmat)[{p + j - m, pd + j}] = i_op;
                p += n_sites - m;
            }
            if (m == n_sites - 1)
                // 1 : mm / cd (only last site)
                (*prmat)[{p++, 0}] = b_op[m][m];
            assert(p == rlshape);
            opt->lmat = plmat, opt->rmat = prmat;
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
};

} // namespace block2
