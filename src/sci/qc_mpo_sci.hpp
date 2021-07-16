
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

#include "../dmrg/mpo.hpp"
#include "../core/operator_tensor.hpp"
#include "../dmrg/qc_mpo.hpp"
#include "../core/delayed_tensor_functions.hpp"
#include "qc_hamiltonian_sci.hpp"
#include "../core/symbolic.hpp"
#include "../core/tensor_functions.hpp"
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
    using MPO<S>::sparse_form;
    using MPO<S>::site_op_infos;
    using MPO<S>::op;
    using MPO<S>::const_e;
    using MPO<S>::tf;
    QCTypes mode;
    bool symmetrized_p; //!> If true, conventional P operator; symmetrized P
    bool firstSiteIsSCI, lastSiteIsSCI;
    MPOQCSCI(const HamiltonianQCSCI<S> &hamil, QCTypes mode = QCTypes::NC,
          bool symmetrized_p = true)
        : MPO<S>(hamil.n_sites), mode(mode), symmetrized_p(symmetrized_p) {
        int nOrbFirst = 1; // #Orbitals of first site
        if(hamil.sciWrapperLeft != nullptr){
            sparse_form[0] = 'S'; // Big site will be sparse
            firstSiteIsSCI = true;
            nOrbFirst = hamil.sciWrapperLeft->nOrbThis;
        }else{
            firstSiteIsSCI = false;
        }
        if(hamil.sciWrapperRight != nullptr){
            sparse_form[hamil.n_sites-1] = 'S'; // Big site will be sparse
            lastSiteIsSCI = true;
        }else{
            lastSiteIsSCI = false;
        }
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S>>(OpNames::H, SiteIndex(), hamil.vacuum);
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vacuum);
        const auto nOrb = hamil.nOrbLeft + hamil.nOrbCas + hamil.nOrbRight;
        if(nOrb > numeric_limits<uint16_t>::max()){
            cerr << "value of nOrb " << nOrb << endl;
            cerr << "max value of uint16_t" << numeric_limits<uint16_t>::max() << endl;
            throw std::runtime_error("SiteIndex and others require int16 type...");
        }
        const auto lastSite = hamil.n_sites - 1;
        // vv nOrb x 2 matrices
        using OpExprMat2 = vector< array<shared_ptr<OpExpr<S>>,2> >;
        OpExprMat2 c_op(nOrb), d_op(nOrb); // hrl: site; sz value
        OpExprMat2 mc_op(nOrb), md_op(nOrb); // hrl: mc stands for minus C
        OpExprMat2 rd_op(nOrb), r_op(nOrb);
        OpExprMat2 mrd_op(nOrb), mr_op(nOrb);
        // vv nOrb x nOrb x 4 tensors
        using OpExprMat4 = vector< array<shared_ptr<OpExpr<S>>,4> >;
        using OpExprTens4 = vector<OpExprMat4>;
        OpExprTens4 a_op(nOrb, OpExprMat4(nOrb)); //It would be so easy in fortran...
        OpExprTens4 ad_op(nOrb, OpExprMat4(nOrb));
        OpExprTens4 b_op(nOrb, OpExprMat4(nOrb));
        OpExprTens4 p_op(nOrb, OpExprMat4(nOrb));
        OpExprTens4 pd_op(nOrb, OpExprMat4(nOrb));
        OpExprTens4 q_op(nOrb, OpExprMat4(nOrb));
        op = dynamic_pointer_cast<OpElement<S>>(h_op);
        const_e = hamil.e();
        //tf = make_shared<TensorFunctions<S>>(hamil.opf);
        tf = make_shared<TensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        if (hamil.delayed == DelayedSCIOpNames::None)
            tf = make_shared<TensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        else
            tf = make_shared<DelayedTensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        tf->opf->seq = hamil.opf->seq; // seq_type
        site_op_infos = hamil.site_op_infos;
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
            const auto mm = m + nOrbFirst - 1; // First site may be big
            const int lshape = 2 + 4 * nOrb + 12 * mm * mm; // left bond dimension of MPO site
            const int rshape = 2 + 4 * nOrb + 12 * (mm+1) * (mm+1); // right bond dimension of MPO site
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (m == lastSite) // last site
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
                p = 2;
                if(not firstSiteIsSCI) {
                    mat[{0, 2}] = c_op[m][0];
                    mat[{0, 3}] = c_op[m][1];
                    mat[{0, 4}] = d_op[m][0];
                    mat[{0, 5}] = d_op[m][1];
                    p = 6;
                }else{
                    for (uint8_t s = 0; s < 2; s++) {
                        for (int iOrb = 0; iOrb < nOrbFirst; ++iOrb) {
                            mat[{0, p++}] = c_op[iOrb][s];
                        }
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (int iOrb = 0; iOrb < nOrbFirst; ++iOrb) {
                            mat[{0, p++}] = d_op[iOrb][s];
                        }
                    }
                }
                for (uint8_t s = 0; s < 2; s++) { // R'
                    for (uint16_t j = mm + 1; j < nOrb; j++) {
                        mat[{0, p++}] = rd_op[j][s];
                    }
                }
                for (uint8_t s = 0; s < 2; s++) { // -R
                    for (uint16_t j = mm + 1; j < nOrb; j++){
                        mat[{0, p++}] = mr_op[j][s];
                    }
                }
            } else if (m == lastSite) {
                /////////////////////////////
                // Last site left blocking
                // 1 H, R, -R', a, a'
                /////////////////////////////
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < mm; j++)
                        mat[{p++, 0}] = r_op[j][s];
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < mm; j++)
                        mat[{p++, 0}] = mrd_op[j][s];
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for(int x = mm; x < nOrb; ++x){
                        mat[{p++, 0}] = d_op[x][s];
                    }
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for(int x = mm; x < nOrb; ++x) {
                        mat[{p++, 0}] = c_op[x][s];
                    }
                }
            }
            if (m == 0) {
                /////////////////////////////
                // First site right blocking
                // A, A', B
                /////////////////////////////
                if(not firstSiteIsSCI) {
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{0, p++}] = a_op[m][m][s];
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{0, p++}] = ad_op[m][m][s];
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{0, p++}] = b_op[m][m][s];
                }else{
                    for (uint8_t s = 0; s < 4; s++)
                        for(int iOrb = 0; iOrb < nOrbFirst; ++iOrb)
                            for(int jOrb = 0; jOrb < nOrbFirst; ++jOrb)
                                mat[{0, p++}] = a_op[iOrb][jOrb][s];
                    for (uint8_t s = 0; s < 4; s++)
                        for(int iOrb = 0; iOrb < nOrbFirst; ++iOrb)
                            for(int jOrb = 0; jOrb < nOrbFirst; ++jOrb)
                                mat[{0, p++}] = ad_op[iOrb][jOrb][s];
                    for (uint8_t s = 0; s < 4; s++)
                        for(int iOrb = 0; iOrb < nOrbFirst; ++iOrb)
                            for(int jOrb = 0; jOrb < nOrbFirst; ++jOrb)
                                mat[{0, p++}] = b_op[iOrb][jOrb][s];
                }
                assert(p == mat.n);
            } else {
                if (m != lastSite) {
                    /////////////////////////////
                    // Normal site left blocking
                    // R', -R, a, a'
                    /////////////////////////////
                    mat[{0, 0}] = i_op;
                    mat[{1, 0}] = h_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < mm; j++)
                            mat[{p++, 0}] = r_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < mm; j++)
                            mat[{p++, 0}] = mrd_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p, 0}] = d_op[mm][s];
                        p += nOrb - mm;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p, 0}] = c_op[mm][s];
                        p += nOrb - mm;
                    }
                }
                /////////////////////////////
                // Normal AND last site left blocking
                // P, P', Q
                /////////////////////////////
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < mm; j++) {
                        for (uint16_t k = 0; k < mm; k++)
                            mat[{p++, 0}] = 0.5 * p_op[j][k][s];
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < mm; j++) {
                        for (uint16_t k = 0; k < mm; k++)
                            mat[{p++, 0}] = 0.5 * pd_op[j][k][s];
                    }
                for (uint8_t s = 0; s < 4; s++)
                    for (uint16_t j = 0; j < mm; j++) {
                        for (uint16_t k = 0; k < mm; k++)
                            mat[{p++, 0}] = q_op[j][k][s];
                    }
                assert(p == mat.m);
            }
            if (m != 0 and m != lastSite) {
                /////////////////////////////
                // Normal sites
                /////////////////////////////
                mat[{1, 1}] = i_op;
                p = 2;
                /////////////////////////////
                // pointers
                /////////////////////////////
                int pi = 1;
                int pc[2] = {2, 2 + mm};
                int pd[2] = {2 + mm * 2, 2 + mm * 3};
                int prd[2] = {2 + mm * 4 - mm, 2 + mm * 3 + nOrb - mm};
                int pr[2] = {2 + mm * 2 + nOrb * 2 - mm,
                             2 + mm + nOrb * 3 - mm};
                int pa[4] = {2 + nOrb * 4 + mm * mm * 0,
                             2 + nOrb * 4 + mm * mm * 1,
                             2 + nOrb * 4 + mm * mm * 2,
                             2 + nOrb * 4 + mm * mm * 3};
                int pad[4] = {2 + nOrb * 4 + mm * mm * 4,
                              2 + nOrb * 4 + mm * mm * 5,
                              2 + nOrb * 4 + mm * mm * 6,
                              2 + nOrb * 4 + mm * mm * 7};
                int pb[4] = {2 + nOrb * 4 + mm * mm * 8,
                             2 + nOrb * 4 + mm * mm * 9,
                             2 + nOrb * 4 + mm * mm * 10,
                             2 + nOrb * 4 + mm * mm * 11};
                // C
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < mm; j++)
                        mat[{pc[s] + j, p++}] = i_op;
                    mat[{pi, p++}] = c_op[mm][s];
                }
                // D
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < mm; j++)
                        mat[{pd[s] + j, p++}] = i_op;
                    mat[{pi, p++}] = d_op[mm][s];
                }
                // RD
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = mm + 1; i < nOrb; i++) {
                        mat[{prd[s] + i, p}] = i_op;
                        mat[{pi, p }] = rd_op[i][s];
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < mm; k++) {
                                mat[{pd[sp] + k, p}] =
                                    -1.0 * pd_op[k][i][sp | (s << 1)]; // HRL: new commit by HC on 2020-07-15 ( 82c7cb334f3cae06de5ec0cbfe0122bb550b59cc )
                                    //pd_op[i][k][s | (sp << 1)]; // old commit ; just symmetry
                                mat[{pc[sp] + k, p}] =
                                    q_op[k][i][sp | (s << 1)];
                            }
                        if (!symmetrized_p)
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < mm; j++)
                                    for (uint16_t l = 0; l < mm; l++) {
                                        double f = hamil.v(s, sp, i, j, mm, l);
                                        mat[{pa[s | (sp << 1)] + j * mm + l, p}] =
                                                f * d_op[mm][sp];
                                    }
                        else
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < mm; j++)
                                    for (uint16_t l = 0; l < mm; l++) {
                                        double f0 =  0.5 * hamil.v(s, sp, i, j, mm, l),
                                               f1 = -0.5 * hamil.v(s, sp, i, l, mm, j);
                                        mat[{pa[s | (sp << 1)] + j * mm + l, p}] +=
                                            f0 * d_op[mm][sp];
                                        mat[{pa[sp | (s << 1)] + j * mm + l, p}] +=
                                            f1 * d_op[mm][sp];
                                    }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < mm; k++)
                                for (uint16_t l = 0; l < mm; l++) {
                                    double f = hamil.v(s, sp, i, mm, k, l);
                                    mat[{pb[sp | (sp << 1)] + l * mm + k, p}] =
                                            f * c_op[mm][s];
                                }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t j = 0; j < mm; j++)
                                for (uint16_t k = 0; k < mm; k++) {
                                    double f =
                                        -1.0 * hamil.v(s, sp, i, j, k, mm);
                                    mat[{pb[s | (sp << 1)] + j * mm + k, p}] +=
                                        f * c_op[mm][sp];
                                }
                        ++p;
                    }
                }
                // R
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t i = mm + 1; i < nOrb; i++) {
                        mat[{pr[s] + i, p}] = i_op;
                        mat[{pi, p}] = mr_op[i][s];
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < mm; k++) {
                                mat[{pc[sp] + k, p}] =
                                    p_op[k][i][sp | (s << 1)]; // HRL: new commit by HC on 2020-07-15 ( 82c7cb334f3cae06de5ec0cbfe0122bb550b59cc )
                                    //-1.0 * p_op[i][k][s | (sp << 1)]; //old HRL ; just symmetry
                                mat[{pd[sp] + k, p}] =
                                    -1.0 * q_op[i][k][s | (sp << 1)];
                            }
                        if (!symmetrized_p)
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < mm; j++)
                                    for (uint16_t l = 0; l < mm; l++) {
                                        double f = -1.0 * hamil.v(s, sp, i,
                                                                  j, mm, l);
                                        mat[{pad[s | (sp << 1)] + j * mm + l, p}] =
                                            f * c_op[mm][sp];
                                    }
                        else
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < mm; j++)
                                    for (uint16_t l = 0; l < mm; l++) {
                                        double f0 = -0.5 * hamil.v(s, sp, i,
                                                                   j, mm, l),
                                               f1 = 0.5 * hamil.v(s, sp, i,
                                                                  l, mm, j);
                                        mat[{pad[s | (sp << 1)] + j * mm + l, p}] +=
                                            f0 * c_op[mm][sp];
                                        mat[{pad[sp | (s << 1)] + j * mm + l, p}] +=
                                            f1 * c_op[mm][sp];
                                    }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t k = 0; k < mm; k++)
                                for (uint16_t l = 0; l < mm; l++) {
                                    double f =
                                            -1.0 * hamil.v(s, sp, i, mm, k, l);
                                    mat[{pb[sp | (sp << 1)] + k * mm + l, p}]
                                            = f * d_op[mm][s];
                                }
                        for (uint8_t sp = 0; sp < 2; sp++)
                            for (uint16_t j = 0; j < mm; j++)
                                for (uint16_t k = 0; k < mm; k++) {
                                    double f = (-1.0) * (-1.0) *
                                               hamil.v(s, sp, i, j, k, mm);
                                    mat[{pb[sp | (s << 1)] + k * mm + j, p}] =
                                            f * d_op[mm][sp];
                                }
                        ++p;
                    }
                }
                // A
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < mm; i++)
                        for (uint16_t j = 0; j < mm; j++)
                            mat[{pa[s] + i * mm + j, p + i * (mm + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < mm; i++) {
                        mat[{pc[s & 1] + i, p + i * (mm + 1) + mm}] = c_op[mm][s >> 1];
                        mat[{pc[s >> 1] + i, p + mm * (mm + 1) + i}] = mc_op[mm][s & 1];
                    }
                    mat[{pi, p + mm * (mm + 1) + mm}] = a_op[mm][mm][s];
                    p += (mm + 1) * (mm + 1);
                }
                // AD
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < mm; i++)
                        for (uint16_t j = 0; j < mm; j++)
                            mat[{pad[s] + i * mm + j, p + i * (mm + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < mm; i++) {
                        mat[{pd[s & 1] + i, p + i * (mm + 1) + mm}] = md_op[mm][s >> 1];
                        mat[{pd[s >> 1] + i, p + mm * (mm + 1) + i}] = d_op[mm][s & 1];
                    }
                    mat[{pi, p + mm * (mm + 1) + mm}] = ad_op[mm][mm][s];
                    p += (mm + 1) * (mm + 1);
                }
                // B
                for (uint8_t s = 0; s < 4; s++) {
                    for (uint16_t i = 0; i < mm; i++)
                        for (uint16_t j = 0; j < mm; j++)
                            mat[{pb[s] + i * mm + j, p + i * (mm + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < mm; i++) {
                        mat[{pc[s & 1] + i, p + i * (mm + 1) + mm}] = d_op[mm][s >> 1];
                        mat[{pd[s >> 1] + i, p + mm * (mm + 1) + i}] = mc_op[mm][s & 1];
                    }
                    mat[{pi, p + mm * (mm + 1) + mm}] = b_op[mm][mm][s];
                    p += (mm + 1) * (mm + 1);
                }
                assert(p == mat.n);
            }
            shared_ptr<OperatorTensor<S>> opt = make_shared<OperatorTensor<S>>();
            if (not (m == 0 and m == lastSite)) {
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
                if (m == lastSite)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (m != lastSite) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < mm + 1; j++)
                            lop[p++] = c_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < mm+ 1; j++)
                            lop[p++] = d_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = mm + 1; j < nOrb; j++)
                            lop[p++] = rd_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = mm + 1; j < nOrb; j++)
                            lop[p++] = mr_op[j][s];
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < mm + 1; j++) {
                            for (uint8_t k = 0; k < mm + 1; k++)
                                lop[p++] = a_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < mm + 1; j++) {
                            for (uint8_t k = 0; k < mm + 1; k++)
                                lop[p++] = ad_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < mm + 1; j++) {
                            for (uint16_t k = 0; k < mm + 1; k++)
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
                        for (uint16_t j = 0; j < mm; j++)
                            rop[p++] = r_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < mm; j++)
                            rop[p++] = mrd_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = mm; j < nOrb; j++)
                            rop[p++] = d_op[j][s];
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = mm; j < nOrb; j++)
                            rop[p++] = c_op[j][s];
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < mm; j++) {
                            for (uint16_t k = 0; k < mm; k++)
                                rop[p++] = 0.5 * p_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < mm; j++) {
                            for (uint16_t k = 0; k < mm; k++)
                                rop[p++] = 0.5 * pd_op[j][k][s];
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < mm; j++) {
                            for (uint16_t k = 0; k < mm; k++)
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
            vector<pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>> vps(
                this->tensors[m]->ops.cbegin(), this->tensors[m]->ops.cend());
            for (auto it = vps.crbegin(); it != vps.crend(); ++it) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(it->first);
                //cout << "m == " << (int) m << "deallocate" << op.name << "s" << (int) op.site_index[0] << ","
                //     << (int) op.site_index[1] << "ss" << (int) op.site_index.s(0) << (int) op.site_index.s(1) << endl;
                if ( (m == n_sites - 1 and lastSiteIsSCI) or (m==0 and firstSiteIsSCI)) {
                    //ATTENTION hrl: I assume that all operators are allocated on the big site
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

// MPO of single site operator
template <typename S> struct SiteMPOSCI : MPO<S> {
    using MPO<S>::sparse_form;
    using MPO<S>::n_sites;
    using MPO<S>::site_op_infos;
    using MPO<S>::op;
    using MPO<S>::const_e;
    using MPO<S>::tf;
    SiteMPOSCI(const HamiltonianQCSCI<S> &hamil, const shared_ptr<OpElement<S>> &op_,
               int k = -1)
        : MPO<S>(hamil.n_sites) {
            shared_ptr<OpElement<S>> i_op = make_shared<OpElement<S>>(
                    OpNames::I, SiteIndex(), hamil.vacuum);
        if (hamil.sciWrapperLeft != nullptr)
            sparse_form[0] = 'S';
        if (hamil.sciWrapperRight != nullptr)
            sparse_form[hamil.n_sites - 1] = 'S';
        op = op_;
        const_e = 0.0;
        tf = make_shared<TensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        if (hamil.delayed == DelayedSCIOpNames::None)
            tf = make_shared<TensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        else
            tf = make_shared<DelayedTensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        tf->opf->seq = hamil.opf->seq; // seq_type
        if (k == -1) {
            assert(op->site_index.size() >= 1);
            k = op->site_index[0];
        }
        site_op_infos = hamil.site_op_infos;
        for (uint16_t m = 0; m < hamil.n_sites; m++) {
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(1);
            else if (m == hamil.n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(1);
            else
                pmat = make_shared<SymbolicMatrix<S>>(1, 1);
            (*pmat)[{0, 0}] = m == k ? op : i_op;
            shared_ptr<OperatorTensor<S>> opt = make_shared<OperatorTensor<S>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop = make_shared<SymbolicRowVector<S>>(1);
            (*plop)[0] = m >= k ? op : i_op;
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop = make_shared<SymbolicColumnVector<S>>(1);
            (*prop)[0] = m <= k ? op : i_op;
            this->right_operator_names.push_back(prop);
            // site operators
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
};

// MPO of single site operator
template <typename S> struct IdentityMPOSCI : MPO<S> {
    using MPO<S>::sparse_form;
    using MPO<S>::n_sites;
    using MPO<S>::site_op_infos;
    using MPO<S>::op;
    using MPO<S>::const_e;
    using MPO<S>::tf;
    IdentityMPOSCI(const HamiltonianQCSCI<S> &hamil): MPO<S>(hamil.n_sites) {
        shared_ptr<OpElement<S>> i_op = make_shared<OpElement<S>>(
                OpNames::I, SiteIndex(), hamil.vacuum);
        if (hamil.sciWrapperLeft != nullptr)
            sparse_form[0] = 'S';
        if (hamil.sciWrapperRight != nullptr)
            sparse_form[hamil.n_sites - 1] = 'S';
        op = i_op;
        const_e = 0.0;
        tf = make_shared<TensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        if (hamil.delayed == DelayedSCIOpNames::None)
            tf = make_shared<TensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        else
            tf = make_shared<DelayedTensorFunctions<S>>(
                make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
        tf->opf->seq = hamil.opf->seq; // seq_type
        site_op_infos = hamil.site_op_infos;
        for (uint16_t m = 0; m < hamil.n_sites; m++) {
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(1);
            else if (m == hamil.n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(1);
            else
            pmat = make_shared<SymbolicMatrix<S>>(1, 1);
            (*pmat)[{0, 0}] = i_op;
            shared_ptr<OperatorTensor<S>> opt = make_shared<OperatorTensor<S>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop = make_shared<SymbolicRowVector<S>>(1);
            (*plop)[0] = i_op;
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop = make_shared<SymbolicColumnVector<S>>(1);
            (*prop)[0] = i_op;
            this->right_operator_names.push_back(prop);
            // site operators
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
};

} // namespace block2
