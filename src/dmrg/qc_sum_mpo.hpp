
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

#include "../core/delayed_tensor_functions.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/symbolic.hpp"
#include "../core/tensor_functions.hpp"
#include "mpo.hpp"
#include "qc_hamiltonian.hpp"
#include "qc_mpo.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct SumMPOQC;

// Quantum chemistry MPO (non-spin-adapted)
template <typename S> struct SumMPOQC<S, typename S::is_sz_t> : MPO<S> {
    using MPO<S>::n_sites;
    vector<uint16_t> ts;
    SumMPOQC(const shared_ptr<HamiltonianQC<S>> &hamil,
             const vector<uint16_t> &pts)
        : MPO<S>(hamil->n_sites), ts(pts) {
        assert(ts.size() > 0);
        sort(ts.begin(), ts.end());
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S>>(OpNames::H, SiteIndex(), hamil->vacuum);
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil->vacuum);
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> c_op(
            hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(2)),
            d_op(hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> tr_op(
            hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(2)),
            ts_op(hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> a_op(
            hamil->n_sites,
            vector<vector<shared_ptr<OpExpr<S>>>>(
                hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> b_op(
            hamil->n_sites,
            vector<vector<shared_ptr<OpExpr<S>>>>(
                hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> p_op(
            hamil->n_sites,
            vector<vector<shared_ptr<OpExpr<S>>>>(
                hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> q_op(
            hamil->n_sites,
            vector<vector<shared_ptr<OpExpr<S>>>>(
                hamil->n_sites, vector<shared_ptr<OpExpr<S>>>(4)));
#else
        shared_ptr<OpExpr<S>> c_op[n_sites][2], d_op[n_sites][2];
        shared_ptr<OpExpr<S>> tr_op[n_sites][2], ts_op[n_sites][2];
        shared_ptr<OpExpr<S>> a_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> p_op[n_sites][n_sites][4];
        shared_ptr<OpExpr<S>> q_op[n_sites][n_sites][4];
#endif
        MPO<S>::op = dynamic_pointer_cast<OpElement<S>>(h_op);
        MPO<S>::const_e = hamil->e();
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil->opf);
        else
            MPO<S>::tf = make_shared<DelayedTensorFunctions<S>>(hamil->opf);
        MPO<S>::site_op_infos = hamil->site_op_infos;
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint16_t m = 0; m < n_sites; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil->orb_sym[m]));
                d_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil->orb_sym[m]));
                tr_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::TR, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil->orb_sym[m]));
                ts_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::TS, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil->orb_sym[m]));
            }
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++) {
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    a_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::A, sidx,
                        S(2, sz_plus[s],
                          hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s],
                          hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                    p_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::P, sidx,
                        S(-2, -sz_plus[s],
                          hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                    q_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::Q, sidx,
                        S(0, -sz_minus[s],
                          hamil->orb_sym[i] ^ hamil->orb_sym[j]));
                }
            }
        int p, lt = (int)ts.size();
        vector<uint16_t> mt(n_sites + 1, 0);
        for (uint16_t m = 0; m <= n_sites; m++)
            for (uint16_t t : ts)
                mt[m] += (t < m);
        for (uint16_t m = 0; m < n_sites; m++) {
            shared_ptr<Symbolic<S>> pmat;
            int lshape = 2 + 4 * n_sites + 8 * mt[m] * m +
                         8 * (lt - mt[m]) * (n_sites - m);
            int rshape = 2 + 4 * n_sites + 8 * mt[m + 1] * (m + 1) +
                         8 * (lt - mt[m + 1]) * (n_sites - m - 1);
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (m == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            int pit = (int)count(ts.begin(), ts.end(), m);
            Symbolic<S> &mat = *pmat;
            if (m == 0) {
                p = 0;
                if (pit)
                    mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++)
                    for (uint16_t j = 0; j <= m; j++)
                        mat[{0, p++}] = c_op[j][s];
                for (uint8_t s = 0; s < 2; s++)
                    for (uint16_t j = 0; j <= m; j++)
                        mat[{0, p++}] = d_op[j][s];
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        mat[{0, p + j - m - 1}] = 0.5 * tr_op[j][s];
                    p += n_sites - m - 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; pit && j < n_sites; j++)
                        mat[{0, p + j - m - 1}] = 0.5 * ts_op[j][s];
                    p += n_sites - m - 1;
                }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t <= m)
                            for (uint16_t j = 0; j <= m; j++)
                                mat[{0, p++}] = a_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t <= m)
                            for (uint16_t j = 0; j <= m; j++)
                                mat[{0, p++}] = b_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t > m)
                            for (uint16_t j = m + 1; j < n_sites; j++)
                                mat[{0, p++}] = 0.5 * p_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t > m)
                            for (uint16_t j = m + 1; j < n_sites; j++)
                                mat[{0, p++}] = 0.5 * q_op[t][j][s];
                assert(p == mat.n);
            } else {
                mat[{0, 0}] = i_op;
                if (pit)
                    mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = (-0.5) * tr_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; pit && j < m; j++)
                        mat[{p + j, 0}] = (-0.5) * ts_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++)
                    for (uint16_t j = m; j < n_sites; j++, p++)
                        if (j == m)
                            mat[{p, 0}] = c_op[j][s];
                for (uint8_t s = 0; s < 2; s++)
                    for (uint16_t j = m; j < n_sites; j++, p++)
                        if (j == m)
                            mat[{p, 0}] = d_op[j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t < m)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{p++, 0}] = 0.5 * p_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t < m)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{p++, 0}] = 0.5 * q_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t == m) {
                            for (uint16_t j = m; j < n_sites; j++, p++)
                                if (j == m)
                                    mat[{p, 0}] = a_op[t][j][s];
                        } else if (t > m)
                            p += n_sites - m;
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t == m) {
                            for (uint16_t j = m; j < n_sites; j++, p++)
                                if (j == m)
                                    mat[{p, 0}] = b_op[t][j][s];
                        } else if (t > m)
                            p += n_sites - m;
                assert(p == mat.m);
            }
            if (m != 0 && m != hamil->n_sites - 1) {
                mat[{1, 1}] = i_op;
                p = 2;
                // pointers
                int pi = 1;
                int pc[2] = {2, 2 + m};
                int pd[2] = {2 + m * 2, 2 + m * 3};
                int ptr[2] = {2 + m * 4 - m, 2 + m * 3 + n_sites - m};
                int pts[2] = {2 + m * 2 + n_sites * 2 - m,
                              2 + m + n_sites * 3 - m};
                int pa[4] = {2 + n_sites * 4 + mt[m] * m * 0,
                             2 + n_sites * 4 + mt[m] * m * 1,
                             2 + n_sites * 4 + mt[m] * m * 2,
                             2 + n_sites * 4 + mt[m] * m * 3};
                int pb[4] = {2 + n_sites * 4 + mt[m] * m * 4,
                             2 + n_sites * 4 + mt[m] * m * 5,
                             2 + n_sites * 4 + mt[m] * m * 6,
                             2 + n_sites * 4 + mt[m] * m * 7};
                int pp[4] = {2 + n_sites * 4 + mt[m] * m * 8,
                             2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m),
                             2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m) * 2,
                             2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m) * 3};
                int pq[4] = {2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m) * 4,
                             2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m) * 5,
                             2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m) * 6,
                             2 + n_sites * 4 + mt[m] * m * 8 +
                                 (lt - mt[m]) * (n_sites - m) * 7};
                // C
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pc[s] + j, p++}] = i_op;
                    mat[{pi, p++}] = c_op[m][s];
                }
                // D
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pd[s] + j, p++}] = i_op;
                    mat[{pi, p++}] = d_op[m][s];
                }
                // TR
                for (uint8_t sp = 0; sp < 2; sp++) {
                    int pik = 0;
                    for (uint16_t k = m + 1; k < n_sites; k++) {
                        mat[{ptr[sp] + k, p}] = i_op;
                        mat[{pi, p}] = 0.5 * tr_op[k][sp];
                        if (count(ts.begin(), ts.end(), k)) {
                            for (uint8_t s = 0; s < 2; s++)
                                for (uint16_t j = 0; j < m; j++)
                                    mat[{pc[s] + j, p}] =
                                        -0.5 * p_op[k][j][sp | (s << 1)];
                            for (uint8_t s = 0; s < 2; s++)
                                for (uint16_t j = 0; j < m; j++)
                                    mat[{pd[s] + j, p}] =
                                        -0.5 * q_op[k][j][sp | (s << 1)];
                            for (uint8_t s = 0; s < 2; s++)
                                mat[{pp[sp | (s << 1)] +
                                         (pit + pik) * (n_sites - m),
                                     p}] = -1.0 * c_op[m][s];
                            for (uint8_t s = 0; s < 2; s++)
                                mat[{pq[sp | (s << 1)] +
                                         (pit + pik) * (n_sites - m),
                                     p}] = -1.0 * d_op[m][s];
                            pik++;
                        }
                        if (pit) {
                            for (uint8_t s = 0; s < 2; s++)
                                mat[{pp[s | (sp << 1)] + k - m, p}] +=
                                    c_op[m][s];
                            for (uint8_t s = 0; s < 2; s++)
                                for (uint16_t l = 0; l < m; l++)
                                    mat[{pd[s] + l, p}] +=
                                        (0.5 * hamil->v(s, sp, m, l, k, m)) *
                                        b_op[m][m][s | (sp << 1)];
                            for (uint8_t s = 0; s < 2; s++)
                                for (uint16_t l = 0; l < m; l++)
                                    mat[{pd[sp] + l, p}] +=
                                        (-0.5 * hamil->v(s, sp, m, m, k, l)) *
                                        b_op[m][m][s | (s << 1)];
                        }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t i : ts)
                                if (i < m)
                                    mat[{pc[s] + i, p}] +=
                                        0.5 * p_op[i][k][s | (sp << 1)];
                        for (uint8_t s = 0; s < 2; s++) {
                            int pii = 0;
                            for (uint16_t i : ts)
                                if (i < m) {
                                    for (uint16_t l = 0; l < m; l++)
                                        mat[{pb[s | (sp << 1)] + pii * m + l,
                                             p}] = (0.5 * hamil->v(s, sp, i, m,
                                                                   k, l)) *
                                                   d_op[m][s];
                                    pii++;
                                }
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            int pii = 0;
                            for (uint16_t i : ts)
                                if (i < m) {
                                    for (uint16_t l = 0; l < m; l++)
                                        mat[{pb[s | (s << 1)] + pii * m + l,
                                             p}] += (-0.5 * hamil->v(s, sp, i,
                                                                     l, k, m)) *
                                                    d_op[m][sp];
                                    pii++;
                                }
                        }
                        p++;
                    }
                }
                // TS
                for (uint8_t sp = 0; sp < 2; sp++) {
                    for (uint16_t j = m + 1; j < n_sites; j++) {
                        mat[{pts[sp] + j, p}] = i_op;
                        if (pit) {
                            mat[{pi, p}] = 0.5 * ts_op[j][sp];
                            for (uint8_t s = 0; s < 2; s++)
                                mat[{pq[s | (sp << 1)] + j - m, p}] =
                                    c_op[m][s];
                            for (uint8_t s = 0; s < 2; s++)
                                for (uint16_t l = 0; l < m; l++) {
                                    mat[{pd[s] + l, p}] +=
                                        (0.5 * hamil->v(sp, s, m, j, m, l)) *
                                        a_op[m][m][sp | (s << 1)];
                                    mat[{pd[s] + l, p}] +=
                                        (-0.5 * hamil->v(s, sp, m, l, m, j)) *
                                        a_op[m][m][s | (sp << 1)];
                                }
                            for (uint8_t s = 0; s < 2; s++)
                                for (uint16_t l = 0; l < m; l++) {
                                    mat[{pc[s] + l, p}] +=
                                        (-0.5 * hamil->v(sp, s, m, j, l, m)) *
                                        b_op[m][m][sp | (s << 1)];
                                    mat[{pc[sp] + l, p}] +=
                                        (0.5 * hamil->v(s, sp, m, m, l, j)) *
                                        b_op[m][m][s | (s << 1)];
                                }
                        }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t i : ts)
                                if (i < m)
                                    mat[{pc[s] + i, p}] =
                                        0.5 * q_op[i][j][s | (sp << 1)];
                        for (uint8_t s = 0; s < 2; s++) {
                            int pii = 0;
                            for (uint16_t i : ts)
                                if (i < m) {
                                    for (uint16_t l = 0; l < m; l++) {
                                        mat[{pb[sp | (s << 1)] + pii * m + l,
                                             p}] = (-0.5 * hamil->v(sp, s, i, j,
                                                                    m, l)) *
                                                   c_op[m][s];
                                        mat[{pb[s | (s << 1)] + pii * m + l,
                                             p}] += (0.5 * hamil->v(s, sp, i, l,
                                                                    m, j)) *
                                                    c_op[m][sp];
                                    }
                                    pii++;
                                }
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            int pii = 0;
                            for (uint16_t i : ts)
                                if (i < m) {
                                    for (uint16_t l = 0; l < m; l++) {
                                        mat[{pa[sp | (s << 1)] + pii * m + l,
                                             p}] = (0.5 * hamil->v(sp, s, i, j,
                                                                   l, m)) *
                                                   d_op[m][s];
                                        mat[{pa[s | (sp << 1)] + pii * m + l,
                                             p}] += (-0.5 * hamil->v(s, sp, i,
                                                                     m, l, j)) *
                                                    d_op[m][s];
                                    }
                                    pii++;
                                }
                        }
                        p++;
                    }
                }
                // A
                for (uint8_t s = 0; s < 4; s++) {
                    int pii = 0;
                    for (uint16_t i : ts)
                        if (i < m) {
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pa[s] + pii * m + j,
                                     p + pii * (m + 1) + j}] = i_op;
                            mat[{pc[s & 1] + i, p + pii * (m + 1) + m}] =
                                c_op[m][s >> 1];
                            pii++;
                        } else if (i == m) {
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pc[s >> 1] + j, p + pii * (m + 1) + j}] =
                                    (-1.0) * c_op[m][s & 1];
                            mat[{pi, p + pii * (m + 1) + m}] = a_op[m][m][s];
                        }
                    assert(mt[m] == pii);
                    p += mt[m + 1] * (m + 1);
                }
                // B
                for (uint8_t s = 0; s < 4; s++) {
                    int pii = 0;
                    for (uint16_t i : ts)
                        if (i < m) {
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pb[s] + pii * m + j,
                                     p + pii * (m + 1) + j}] = i_op;
                            mat[{pc[s & 1] + i, p + pii * (m + 1) + m}] =
                                d_op[m][s >> 1];
                            pii++;
                        } else if (i == m) {
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pd[s >> 1] + j, p + pii * (m + 1) + j}] =
                                    (-1.0) * c_op[m][s & 1];
                            mat[{pi, p + pii * (m + 1) + m}] = b_op[m][m][s];
                        }
                    assert(mt[m] == pii);
                    p += mt[m + 1] * (m + 1);
                }
                // P
                for (uint8_t s = 0; s < 4; s++) {
                    int pii = 0;
                    for (uint16_t i : ts)
                        if (i > m) {
                            for (uint16_t k = m + 1; k < n_sites; k++) {
                                mat[{pp[s] + (pit + pii) * (n_sites - m) + k -
                                         m,
                                     p + pii * (n_sites - m - 1) + k - m - 1}] =
                                    i_op;
                                mat[{pi, p + pii * (n_sites - m - 1) + k - m -
                                             1}] = 0.5 * p_op[i][k][s];
                                for (uint16_t j = 0; j < m; j++) {
                                    mat[{pd[s & 1] + j,
                                         p + pii * (n_sites - m - 1) + k - m -
                                             1}] +=
                                        (-0.5 *
                                         hamil->v(s & 1, s >> 1, i, j, k, m)) *
                                        d_op[m][s >> 1];
                                    mat[{pd[s >> 1] + j,
                                         p + pii * (n_sites - m - 1) + k - m -
                                             1}] +=
                                        (0.5 *
                                         hamil->v(s & 1, s >> 1, i, m, k, j)) *
                                        d_op[m][s & 1];
                                }
                            }
                            pii++;
                        }
                    assert(pii == lt - mt[m + 1]);
                    p += (lt - mt[m + 1]) * (n_sites - m - 1);
                }
                // Q
                for (uint8_t s = 0; s < 4; s++) {
                    int pii = 0;
                    for (uint16_t i : ts)
                        if (i > m) {
                            for (uint16_t j = m + 1; j < n_sites; j++) {
                                mat[{pq[s] + (pit + pii) * (n_sites - m) + j -
                                         m,
                                     p + pii * (n_sites - m - 1) + j - m - 1}] =
                                    i_op;
                                mat[{pi, p + pii * (n_sites - m - 1) + j - m -
                                             1}] = 0.5 * q_op[i][j][s];
                                for (uint16_t k = 0; k < m; k++) {
                                    if ((s & 1) == (s >> 1))
                                        for (uint8_t spp = 0; spp < 2; spp++) {
                                            mat[{pc[spp] + k,
                                                 p + pii * (n_sites - m - 1) +
                                                     j - m - 1}] +=
                                                (0.5 * hamil->v(s & 1, spp, i,
                                                                j, k, m)) *
                                                d_op[m][spp];
                                            mat[{pd[spp] + k,
                                                 p + pii * (n_sites - m - 1) +
                                                     j - m - 1}] +=
                                                (-0.5 * hamil->v(s & 1, spp, i,
                                                                 j, m, k)) *
                                                c_op[m][spp];
                                        }
                                    mat[{pc[s >> 1] + k,
                                         p + pii * (n_sites - m - 1) + j - m -
                                             1}] +=
                                        (-0.5 *
                                         hamil->v(s & 1, s >> 1, i, m, k, j)) *
                                        d_op[m][s & 1];
                                    mat[{pd[s & 1] + k,
                                         p + pii * (n_sites - m - 1) + j - m -
                                             1}] +=
                                        (0.5 *
                                         hamil->v(s & 1, s >> 1, i, k, m, j)) *
                                        c_op[m][s >> 1];
                                }
                            }
                            pii++;
                        }
                    assert(pii == lt - mt[m + 1]);
                    p += (lt - mt[m + 1]) * (n_sites - m - 1);
                }
                assert(p == mat.n);
            }
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            opt->lmat = opt->rmat = pmat;
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop;
            if (m == hamil->n_sites - 1)
                plop = make_shared<SymbolicRowVector<S>>(1);
            else
                plop = make_shared<SymbolicRowVector<S>>(rshape);
            SymbolicRowVector<S> &lop = *plop;
            lop[0] = h_op;
            if (m != hamil->n_sites - 1) {
                lop[1] = i_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j <= m; j++)
                        lop[p + j] = c_op[j][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j <= m; j++)
                        lop[p + j] = d_op[j][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        lop[p + j - (m + 1)] = 0.5 * tr_op[j][s];
                    p += n_sites - (m + 1);
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_sites; j++)
                        lop[p + j - (m + 1)] = 0.5 * ts_op[j][s];
                    p += n_sites - (m + 1);
                }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t <= m)
                            for (uint16_t j = 0; j <= m; j++)
                                lop[p++] = a_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t <= m)
                            for (uint16_t j = 0; j <= m; j++)
                                lop[p++] = b_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t > m)
                            for (uint16_t j = m + 1; j < n_sites; j++)
                                lop[p++] = 0.5 * p_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t > m)
                            for (uint16_t j = m + 1; j < n_sites; j++)
                                lop[p++] = 0.5 * q_op[t][j][s];
                assert(p == rshape);
            }
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop;
            if (m == 0)
                prop = make_shared<SymbolicColumnVector<S>>(1);
            else
                prop = make_shared<SymbolicColumnVector<S>>(lshape);
            SymbolicColumnVector<S> &rop = *prop;
            if (m == 0)
                rop[0] = h_op;
            else {
                rop[0] = i_op;
                rop[1] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = (-0.5) * tr_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = (-0.5) * ts_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_sites; j++)
                        rop[p + j - m] = c_op[j][s];
                    p += n_sites - m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_sites; j++)
                        rop[p + j - m] = d_op[j][s];
                    p += n_sites - m;
                }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t < m)
                            for (uint16_t j = 0; j < m; j++)
                                rop[p++] = 0.5 * p_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t < m)
                            for (uint16_t j = 0; j < m; j++)
                                rop[p++] = 0.5 * q_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t >= m)
                            for (uint16_t j = m; j < n_sites; j++)
                                rop[p++] = a_op[t][j][s];
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t t : ts)
                        if (t >= m)
                            for (uint16_t j = m; j < n_sites; j++)
                                rop[p++] = b_op[t][j][s];
                assert(p == lshape);
            }
            this->right_operator_names.push_back(prop);
            hamil->filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            this->tensors[m]->deallocate();
    }
};

} // namespace block2
