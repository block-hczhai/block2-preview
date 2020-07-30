
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "mpo.hpp"
#include "operator_tensor.hpp"
#include "qc_mpo.hpp"
#include "qc_hamiltonian_SCI.hpp"
#include "symbolic.hpp"
#include "tensor_functions.hpp"
#include <cassert>
#include <memory>
#include <vector>
#include <limits>

using namespace std;

namespace block2 {


template <typename, typename = void> struct MPOQCSCI;

// Quantum chemistry MPO (non-spin-adapted)
template <typename S> struct MPOQCSCI<S, typename S::is_sz_t> : MPO<S> {
    using MPO<S>::n_sites;
    QCTypes mode;
    bool symmetrized_p; //!> If true, conventional P operator; symmetrized P
    MPOQCSCI(const HamiltonianQCSCI<S> &hamil, QCTypes mode = QCTypes::NC,
          bool symmetrized_p = true)
        : MPO<S>(hamil.n_sites), mode(mode), symmetrized_p(symmetrized_p) {
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S>>(OpNames::H, SiteIndex(), hamil.vacuum);
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vacuum);
        const auto nOrb = hamil.nOrbCas + hamil.nOrbExt;
        if(nOrb > numeric_limits<uint16_t>::max()){
            cerr << "value of nOrb " << nOrb << endl;
            cerr << "max value of uint16_t" << numeric_limits<uint16_t>::max() << endl;
            throw std::runtime_error("SiteIndex and others require int16 type...");
        }
        const auto ciSite = hamil.n_sites - 1;
        shared_ptr<OpExpr<S>> c_op[nOrb][2], d_op[nOrb][2]; // hrl: site; sz value
        shared_ptr<OpExpr<S>> mc_op[nOrb][2], md_op[nOrb][2]; // hrl: mc stands for minus C
        shared_ptr<OpExpr<S>> rd_op[nOrb][2], r_op[nOrb][2];
        shared_ptr<OpExpr<S>> mrd_op[nOrb][2], mr_op[nOrb][2];
        shared_ptr<OpExpr<S>> a_op[nOrb][nOrb][4];
        shared_ptr<OpExpr<S>> ad_op[nOrb][nOrb][4];
        shared_ptr<OpExpr<S>> b_op[nOrb][nOrb][4];
        shared_ptr<OpExpr<S>> p_op[nOrb][nOrb][4];
        shared_ptr<OpExpr<S>> pd_op[nOrb][nOrb][4];
        shared_ptr<OpExpr<S>> q_op[nOrb][nOrb][4];
        this->op = dynamic_pointer_cast<OpElement<S>>(h_op);
        this->const_e = hamil.e();
        this->tf = make_shared<TensorFunctions<S>>(hamil.opf);
        this->site_op_infos = hamil.site_op_infos;
        if(mode != QCTypes::NC){
            throw std::invalid_argument("Currently, only NC is implemented as mode. "
                                        "This is what typically makes most sense in MRCI");
        }
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        //////////////////
        // hrl: initialization of array elements
        for (uint16_t iOrb = 0; iOrb < nOrb; iOrb++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[iOrb][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({iOrb}, {s}),
                                              S(1, sz[s], hamil.orb_sym[iOrb]));
                d_op[iOrb][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({iOrb}, {s}),
                                              S(-1, -sz[s], hamil.orb_sym[iOrb]));
                mc_op[iOrb][s] = make_shared<OpElement<S>>(
                        OpNames::C, SiteIndex({iOrb}, {s}),
                        S(1, sz[s], hamil.orb_sym[iOrb]), -1.0);
                md_op[iOrb][s] = make_shared<OpElement<S>>(
                        OpNames::D, SiteIndex({iOrb}, {s}),
                        S(-1, -sz[s], hamil.orb_sym[iOrb]), -1.0);
                rd_op[iOrb][s] =
                    make_shared<OpElement<S>>(OpNames::RD, SiteIndex({iOrb}, {s}),
                                              S(1, sz[s], hamil.orb_sym[iOrb]));
                r_op[iOrb][s] =
                    make_shared<OpElement<S>>(OpNames::R, SiteIndex({iOrb}, {s}),
                                              S(-1, -sz[s], hamil.orb_sym[iOrb]));
                mrd_op[iOrb][s] = make_shared<OpElement<S>>(
                        OpNames::RD, SiteIndex({iOrb}, {s}),
                        S(1, sz[s], hamil.orb_sym[iOrb]), -1.0);
                mr_op[iOrb][s] = make_shared<OpElement<S>>(
                        OpNames::R, SiteIndex({iOrb}, {s}),
                        S(-1, -sz[s], hamil.orb_sym[iOrb]), -1.0);
            }
        for (uint16_t i = 0; i < nOrb; i++)
            for (uint16_t j = 0; j < nOrb; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    a_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::A, sidx,
                        S(2, sz_plus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    ad_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::AD, sidx,
                        S(-2, -sz_plus[s],
                          hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    p_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::P, sidx,
                        S(-2, -sz_plus[s],
                          hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    pd_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PD, sidx,
                        S(2, sz_plus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    q_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::Q, sidx,
                        S(0, -sz_minus[s],
                          hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                }
        // hrl initialization done
        //////////////////
        int p; // pointer
        //////////////////// vvv hrl giant loop for each site, filling this->tensors and left/right operator names
        // hrl: First the matrix elements are set (pmat/mat)
        //      Then, the operator names
        for (uint16_t m = 0; m < hamil.n_sites; m++) {
            shared_ptr<Symbolic<S>> pmat;
            /* hrl: left operators:
             *  H, 1, a', a, R', R, A, A', B
             *  right operators:
             *  1, H, R,  R', a, a',P, P', Q
             *  This is the order used here
             */
            int lshape = 2 + 4 * nOrb + 12 * m * m; // left bond dimension of MPO site
            int rshape = 2 + 4 * nOrb + 12 * (m+1) * (m+1); // right bond dimension of MPO site
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (m == ciSite) // last site
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            ///////////////////////////
            // H, 1, a', a, R', -R
            ///////////////////////////
            if (m == 0) {
                /////////////////////////////
                // First site right blocking
                // H, 1, a', a, R', -R
                /////////////////////////////
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                mat[{0, 2}] = c_op[m][0];
                mat[{0, 3}] = c_op[m][1];
                mat[{0, 4}] = d_op[m][0];
                mat[{0, 5}] = d_op[m][1];
                p = 6;
                for (uint8_t s = 0; s < 2; s++) { // R
                    for (uint16_t j = m + 1; j < nOrb; j++) {
                        mat[{0, p++}] = rd_op[j][s];
                    }
                }
                for (uint8_t s = 0; s < 2; s++) { // -R
                    for (uint16_t j = m + 1; j < nOrb; j++){
                        mat[{0, p++}] = mr_op[j][s];
                    }
                }
            } else if (m == ciSite) {
                /////////////////////////////
                // Last site left blocking
                // 1 H, R, -R', a, a'
                /////////////////////////////
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p++, 0}] = r_op[j][s];
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p++, 0}] = mrd_op[j][s];
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for(int x = m; x < nOrb; ++x){
                        mat[{p++, 0}] = d_op[x][s];
                    }
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for(int x = m; x < nOrb; ++x) {
                        mat[{p++, 0}] = c_op[x][s];
                    }
                }
            }
            if (m == 0) {
                /////////////////////////////
                // First site right blocking
                // A, A', B
                /////////////////////////////
                for (uint8_t s = 0; s < 4; s++)
                    mat[{0, p++}] = a_op[m][m][s];
                for (uint8_t s = 0; s < 4; s++)
                    mat[{0, p++}] = ad_op[m][m][s];
                for (uint8_t s = 0; s < 4; s++)
                    mat[{0, p++}] = b_op[m][m][s];
                assert(p == mat.n);
            } else {
                if (m != hamil.n_sites - 1) {
                    /////////////////////////////
                    // Normal site left blocking
                    // R', -R, a, a'
                    /////////////////////////////
                    mat[{0, 0}] = i_op;
                    mat[{1, 0}] = h_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p++, 0}] = r_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p++, 0}] = mrd_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p, 0}] = d_op[m][s];
                        p += nOrb - m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p, 0}] = c_op[m][s];
                        p += nOrb - m;
                    }
                }
                /////////////////////////////
                // Normal AND last site left blocking
                // P, P', Q
                /////////////////////////////
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p++, 0}] = 0.5 * p_op[j][k][s];
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p++, 0}] = 0.5 * pd_op[j][k][s];
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p++, 0}] = q_op[j][k][s];
                    }
                assert(p == mat.m);
            }
            if (m != 0 and m != hamil.n_sites - 1) {
                /////////////////////////////
                // Normal sites
                /////////////////////////////
                mat[{1, 1}] = i_op;
                p = 2;
                /////////////////////////////
                // pointers
                /////////////////////////////
                int pi = 1;
                int pc[2] = {2, 2 + m};
                int pd[2] = {2 + m * 2, 2 + m * 3};
                int prd[2] = {2 + m * 4 - m, 2 + m * 3 + nOrb - m};
                int pr[2] = {2 + m * 2 + nOrb * 2 - m,
                             2 + m + nOrb * 3 - m};
                int pa[4] = {2 + nOrb * 4 + m * m * 0,
                             2 + nOrb * 4 + m * m * 1,
                             2 + nOrb * 4 + m * m * 2,
                             2 + nOrb * 4 + m * m * 3};
                int pad[4] = {2 + nOrb * 4 + m * m * 4,
                              2 + nOrb * 4 + m * m * 5,
                              2 + nOrb * 4 + m * m * 6,
                              2 + nOrb * 4 + m * m * 7};
                int pb[4] = {2 + nOrb * 4 + m * m * 8,
                             2 + nOrb * 4 + m * m * 9,
                             2 + nOrb * 4 + m * m * 10,
                             2 + nOrb * 4 + m * m * 11};
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
                // RD
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = m + 1; i < nOrb; i++) {
                        mat[{prd[s] + i, p}] = i_op;
                        mat[{pi, p }] = rd_op[i][s];
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++) {
                                mat[{pd[sp] + k, p}] =
                                    -1.0 * pd_op[k][i][sp | (s << 1)]; // HRL: new commit by HC on 2020-07-15 ( 82c7cb334f3cae06de5ec0cbfe0122bb550b59cc )
                                    //pd_op[i][k][s | (sp << 1)]; // old commit ; just symmetry
                                mat[{pc[sp] + k, p}] =
                                    q_op[k][i][sp | (s << 1)];
                            }
                        if (!symmetrized_p)
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f = hamil.v(s, sp, i, j, m, l);
                                        mat[{pa[s | (sp << 1)] + j * m + l, p}] =
                                                f * d_op[m][sp];
                                    }
                        else
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f0 =  0.5 * hamil.v(s, sp, i, j, m, l),
                                               f1 = -0.5 * hamil.v(s, sp, i, l, m, j);
                                        mat[{pa[s | (sp << 1)] + j * m + l, p}] +=
                                            f0 * d_op[m][sp];
                                        mat[{pa[sp | (s << 1)] + j * m + l, p}] +=
                                            f1 * d_op[m][sp];
                                    }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++)
                                for (uint16_t l = 0; l < m; l++) {
                                    double f = hamil.v(s, sp, i, m, k, l);
                                    mat[{pb[sp | (sp << 1)] + l * m + k, p}] =
                                            f * c_op[m][s];
                                }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t k = 0; k < m; k++) {
                                    double f =
                                        -1.0 * hamil.v(s, sp, i, j, k, m);
                                    mat[{pb[s | (sp << 1)] + j * m + k, p}] +=
                                        f * c_op[m][sp];
                                }
                        ++p;
                    }
                }
                // R
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = m + 1; i < nOrb; i++) {
                        mat[{pr[s] + i, p}] = i_op;
                        mat[{pi, p}] = mr_op[i][s];
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++) {
                                mat[{pc[sp] + k, p}] =
                                    p_op[k][i][sp | (s << 1)]; // HRL: new commit by HC on 2020-07-15 ( 82c7cb334f3cae06de5ec0cbfe0122bb550b59cc )
                                    //-1.0 * p_op[i][k][s | (sp << 1)]; //old HRL ; just symmetry
                                mat[{pd[sp] + k, p}] =
                                    -1.0 * q_op[i][k][s | (sp << 1)];
                            }
                        if (!symmetrized_p)
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f = -1.0 * hamil.v(s, sp, i,
                                                                  j, m, l);
                                        mat[{pad[s | (sp << 1)] + j * m + l, p}] =
                                            f * c_op[m][sp];
                                    }
                        else
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        double f0 = -0.5 * hamil.v(s, sp, i,
                                                                   j, m, l),
                                               f1 = 0.5 * hamil.v(s, sp, i,
                                                                  l, m, j);
                                        mat[{pad[s | (sp << 1)] + j * m + l, p}] +=
                                            f0 * c_op[m][sp];
                                        mat[{pad[sp | (s << 1)] + j * m + l, p}] +=
                                            f1 * c_op[m][sp];
                                    }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < m; k++)
                                for (uint16_t l = 0; l < m; l++) {
                                    double f =
                                            -1.0 * hamil.v(s, sp, i, m, k, l);
                                    mat[{pb[sp | (sp << 1)] + k * m + l, p}]
                                            = f * d_op[m][s];
                                }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t k = 0; k < m; k++) {
                                    double f = (-1.0) * (-1.0) *
                                               hamil.v(s, sp, i, j, k, m);
                                    mat[{pb[sp | (s << 1)] + k * m + j, p}] =
                                            f * d_op[m][sp];
                                }
                        ++p;
                    }
                }
                // A
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pa[s] + i * m + j, p + i * (m + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pc[s & 1] + i, p + i * (m + 1) + m}] = c_op[m][s >> 1];
                        mat[{pc[s >> 1] + i, p + m * (m + 1) + i}] = mc_op[m][s & 1];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = a_op[m][m][s];
                    p += (m + 1) * (m + 1);
                }
                // AD
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pad[s] + i * m + j, p + i * (m + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pd[s & 1] + i, p + i * (m + 1) + m}] = md_op[m][s >> 1];
                        mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] = d_op[m][s & 1];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m][s];
                    p += (m + 1) * (m + 1);
                }
                // B
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pb[s] + i * m + j, p + i * (m + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pc[s & 1] + i, p + i * (m + 1) + m}] = d_op[m][s >> 1];
                        mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] = mc_op[m][s & 1];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = b_op[m][m][s];
                    p += (m + 1) * (m + 1);
                }
                assert(p == mat.n);
            }
            shared_ptr<OperatorTensor<S>> opt = make_shared<OperatorTensor<S>>();
            if (not (m == 0 and m == hamil.n_sites - 1)) {
                opt->lmat = opt->rmat = pmat;
            } else{ // should only occur for n_sites = 2
                opt->rmat = pmat;
            }
            /////////////////////// hrl vv
            // operator names
            if (opt->lmat == pmat) {
                /////////////////////////////
                // left operator names
                /////////////////////////////
                shared_ptr<SymbolicRowVector<S>> plop;
                if (m == hamil.n_sites - 1)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (m != hamil.n_sites - 1) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            lop[p++] = c_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            lop[p++] = d_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m + 1; j < nOrb; j++)
                            lop[p++] = rd_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m + 1; j < nOrb; j++)
                            lop[p++] = mr_op[j][s];
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint8_t k = 0; k < m + 1; k++)
                                lop[p++] = a_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint8_t k = 0; k < m + 1; k++)
                                lop[p++] = ad_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                lop[p++] = b_op[j][k][s];
                        }
                    assert(p == rshape);
                }
                this->left_operator_names.push_back(plop);
            }
            if (opt->rmat == pmat) {
                /////////////////////////////
                // right operator names
                /////////////////////////////
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
                            rop[p++] = r_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            rop[p++] = mrd_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m; j < nOrb; j++)
                            rop[p++] = d_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m; j < nOrb; j++)
                            rop[p++] = c_op[j][s];
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                rop[p++] = 0.5 * p_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                rop[p++] = 0.5 * pd_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                rop[p++] = q_op[j][k][s];
                        }
                    assert(p == lshape);
                }
                this->right_operator_names.push_back(prop);
            }
            // hrl: opt->ops is empty. It will be filled in filter_site_opts based on lmat,rmat
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
        /////////////////////// hrl giant loop ^^^^
    }
    void deallocate() override {
        for (uint16_t m = n_sites - 1; m < n_sites; m--) {
            for (auto it = this->tensors[m]->ops.crbegin();
                 it != this->tensors[m]->ops.crend(); ++it) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(it->first);
                //cout << "m == " << (int) m << "deallocate" << op.name << "s" << (int) op.site_index[0] << ","
                //     << (int) op.site_index[1] << "ss" << (int) op.site_index.s(0) << (int) op.site_index.s(1) << endl;
                if (m == n_sites - 1) { //ATTENTION hrl: I assume that all operators are allocated on the big site
                    it->second->deallocate();
                } else if (op.name == OpNames::R || op.name == OpNames::RD ||
                           op.name == OpNames::H ||
                           (op.name == OpNames::Q &&
                            op.site_index.s(0) == op.site_index.s(1)))
                    it->second->deallocate();
            }
        }
    }
    };

    template <typename S> struct MPOQCSCI<S, typename S::is_su2_t> : MPO<S> {
        QCTypes mode;
        bool symmetrized_p; //!> If true, conventional P operator; symmetrized P
        MPOQCSCI(const HamiltonianQCSCI<S> &hamil, QCTypes mode = QCTypes::NC,
                 bool symmetrized_p = true):
                MPO<S>(hamil.n_sites), mode(mode), symmetrized_p(symmetrized_p) {
            throw std::runtime_error("not yet implemented");
        }
    };

} // namespace block2
