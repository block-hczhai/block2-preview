
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2022 Huanchen Zhai <hczhai@caltech.edu>
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

/** Automatic construction of MPO for N particle density matrix. */

#pragma once

#include "../core/spin_permutation.hpp"
#include "general_mpo.hpp"
#include "mpo.hpp"
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

template <typename S, typename FL> struct GeneralNPDMMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    typedef long long int LL;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::tensors;
    using MPO<S, FL>::left_operator_names;
    using MPO<S, FL>::right_operator_names;
    using MPO<S, FL>::middle_operator_names;
    using MPO<S, FL>::middle_operator_exprs;
    using MPO<S, FL>::basis;
    shared_ptr<NPDMScheme> scheme;
    shared_ptr<NPDMCounter> counter;
    FP cutoff;
    int iprint;
    S left_vacuum = S(S::invalid);
    GeneralNPDMMPO(const shared_ptr<GeneralHamiltonian<S, FL>> &hamil,
                   const shared_ptr<NPDMScheme> &scheme, FP cutoff = (FP)0.0,
                   int iprint = 1, const string &tag = "NPDM")
        : MPO<S, FL>(hamil->n_sites, tag), scheme(scheme), cutoff(cutoff),
          iprint(iprint) {
        MPO<S, FL>::hamil = hamil;
    }
    void build() override {
        shared_ptr<GeneralHamiltonian<S, FL>> hamil =
            dynamic_pointer_cast<GeneralHamiltonian<S, FL>>(MPO<S, FL>::hamil);
        shared_ptr<OpElement<S, FL>> zero_op = make_shared<OpElement<S, FL>>(
            OpNames::Zero, SiteIndex(), hamil->vacuum);
        MPO<S, FL>::const_e = (typename const_fl_type<FL>::FL)0.0;
        MPO<S, FL>::op = zero_op;
        MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        n_sites = (int)hamil->n_sites;
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        basis = hamil->basis;
        left_operator_names.resize(n_sites, nullptr);
        right_operator_names.resize(n_sites, nullptr);
        tensors.resize(n_sites, nullptr);
        for (uint16_t m = 0; m < n_sites; m++)
            tensors[m] = make_shared<OperatorTensor<S, FL>>();
        counter = make_shared<NPDMCounter>(scheme->n_max_ops, n_sites);
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        if (iprint) {
            cout << "Build NPDMMPO | Nsites = " << setw(5) << n_sites
                 << " | Nmaxops = " << setw(2) << scheme->n_max_ops
                 << " | Cutoff = " << scientific << setw(8) << setprecision(2)
                 << cutoff << endl;
        }
        vector<LL> lshapes(n_sites, 0), rshapes(n_sites, 0);
        map<vector<uint16_t>, vector<pair<int, int>>> middle_patterns;
        for (int i = 0; i < (int)scheme->perms.size(); i++)
            for (int j = 0; j < (int)scheme->perms[i]->index_patterns.size();
                 j++)
                middle_patterns[scheme->perms[i]->index_patterns[j]].push_back(
                    make_pair(i, j));
        for (uint16_t m = 0; m < n_sites; m++) {
            LL &lshape = lshapes[m], &rshape = rshapes[m];
            for (int i = 0; i < (int)scheme->left_terms.size(); i++)
                lshape += counter->count_left(scheme->left_terms[i].first.first,
                                              m, scheme->left_terms[i].second);
            for (int i = 0; i < (int)scheme->right_terms.size(); i++)
                rshape +=
                    counter->count_right(scheme->right_terms[i].first.first, m);
            if (m == n_sites - 1)
                for (int i = 0; i < (int)scheme->last_right_terms.size(); i++)
                    rshape += counter->count_right(
                        scheme->last_right_terms[i].first.first, m);
        }
        LL ixx, acc_cnt;
        Timer _t, _t2;
        double tsite, tnmid, tsite_total = 0;
        for (int m = 0; m < n_sites; m++) {
            if (iprint) {
                cout << " Site = " << setw(5) << m << " / " << setw(5)
                     << n_sites << " ..";
                cout << " L = " << setw(6) << lshapes[m];
                cout << " R = " << setw(6) << rshapes[m];
                cout.flush();
            }
            _t.get_time();
            tsite = tnmid = 0;
            LL lshape = lshapes[m], rshape = rshapes[m];
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            vector<uint16_t> idx;
            ixx = 0;
            if (iprint >= 2)
                cout << endl;
            for (int i = 0; i < (int)scheme->left_terms.size(); i++) {
                int cnt = counter->count_left(scheme->left_terms[i].first.first,
                                              m, scheme->left_terms[i].second);
                if (cnt == 0)
                    continue;
                bool has_next =
                    counter->init_left(scheme->left_terms[i].first.first, m,
                                       scheme->left_terms[i].second, idx);
                if (iprint >= 2)
                    cout << " -  LEFT  [" << setw(5) << i << "] :: ";
                assert(has_next);
                for (int j = 0; j < cnt; j++) {
                    SiteIndex si({(uint16_t)(ixx / 1000 / 1000),
                                  (uint16_t)(ixx / 1000 % 1000),
                                  (uint16_t)(ixx % 1000)},
                                 {});
                    S q = hamil->get_string_quantum(
                        scheme->left_terms[i].first.second, &idx[0]);
                    (*plop)[ixx] =
                        make_shared<OpElement<S, FL>>(OpNames::XL, si, q);
                    if (iprint >= 2) {
                        cout << "(" << ixx << ") "
                             << scheme->left_terms[i].first.second << " ";
                        for (auto &g : idx)
                            cout << g << " ";
                        cout << q << " / ";
                    }
                    ixx++;
                    has_next = counter->next_left(
                        scheme->left_terms[i].first.first, m, idx);
                }
                if (iprint >= 2)
                    cout << endl;
                assert(!has_next);
            }
            assert(ixx == lshape);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            ixx = 0;
            for (int i = 0; i < (int)scheme->right_terms.size(); i++) {
                int cnt =
                    counter->count_right(scheme->right_terms[i].first.first, m);
                if (cnt == 0)
                    continue;
                bool has_next = counter->init_right(
                    scheme->right_terms[i].first.first, m, idx);
                if (iprint >= 2)
                    cout << " - RIGHT  [" << setw(5) << i << "] :: ";
                assert(has_next);
                for (int j = 0; j < cnt; j++) {
                    SiteIndex si({(uint16_t)(ixx / 1000 / 1000),
                                  (uint16_t)(ixx / 1000 % 1000),
                                  (uint16_t)(ixx % 1000)},
                                 {});
                    S q = hamil->get_string_quantum(
                        scheme->right_terms[i].first.second, &idx[0]);
                    (*prop)[ixx] =
                        make_shared<OpElement<S, FL>>(OpNames::XR, si, q);
                    if (iprint >= 2) {
                        cout << "(" << ixx << ") "
                             << scheme->right_terms[i].first.second << " ";
                        for (auto &g : idx)
                            cout << g << " ";
                        cout << q << " / ";
                    }
                    ixx++;
                    has_next = counter->next_right(
                        scheme->right_terms[i].first.first, m, idx);
                }
                if (iprint >= 2)
                    cout << endl;
                assert(!has_next);
            }
            if (m == n_sites - 1)
                for (int i = 0; i < (int)scheme->last_right_terms.size(); i++) {
                    int cnt = counter->count_right(
                        scheme->last_right_terms[i].first.first, m);
                    bool has_next = counter->init_right(
                        scheme->last_right_terms[i].first.first, m, idx);
                    if (iprint >= 2)
                        cout << " - RIGHT* [" << setw(5) << i << "] :: ";
                    assert(has_next);
                    for (int j = 0; j < cnt; j++) {
                        SiteIndex si({(uint16_t)(ixx / 1000 / 1000),
                                      (uint16_t)(ixx / 1000 % 1000),
                                      (uint16_t)(ixx % 1000)},
                                     {});
                        S q = hamil->get_string_quantum(
                            scheme->last_right_terms[i].first.second, &idx[0]);
                        (*prop)[ixx] =
                            make_shared<OpElement<S, FL>>(OpNames::XR, si, q);
                        if (iprint >= 2) {
                            cout << "(" << ixx << ") "
                                 << scheme->last_right_terms[i].first.second
                                 << " ";
                            for (auto &g : idx)
                                cout << g << " ";
                            cout << q << " / ";
                        }
                        ixx++;
                        has_next = counter->next_right(
                            scheme->last_right_terms[i].first.first, m, idx);
                    }
                    if (iprint >= 2)
                        cout << endl;
                    assert(!has_next);
                }
            assert(ixx == rshape);
            left_operator_names[m] = plop;
            right_operator_names[m] = prop;
            // construct local operators
            shared_ptr<OperatorTensor<S, FL>> opt = tensors[m];
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            int llshape = m == 0 ? 1 : lshapes[m - 1], lrshape = lshapes[m];
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            int rlshape = rshapes[m],
                rrshape = m == n_sites - 1 ? 1 : rshapes[m + 1];
            if (m == 0)
                prmat = make_shared<SymbolicRowVector<S>>(rrshape);
            else if (m == n_sites - 1)
                prmat = make_shared<SymbolicColumnVector<S>>(rlshape);
            else
                prmat = make_shared<SymbolicMatrix<S>>(rlshape, rrshape);
            opt->lmat = plmat, opt->rmat = prmat;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> site_ops;
            unordered_map<string, shared_ptr<OpElement<S, FL>>> site_op_names;
            for (string &cd : scheme->local_terms)
                site_ops[cd] = nullptr;
            site_ops[""] = nullptr;
            hamil->get_site_string_ops(m, site_ops);
            site_op_names.reserve(site_ops.size());
            ixx = 0;
            for (string &cd : scheme->local_terms) {
                shared_ptr<SparseMatrix<S, FL>> xm = site_ops.at(cd);
                if (cd.length() == 0 && m != 0 && m != n_sites - 1) {
                    site_op_names[cd] = make_shared<OpElement<S, FL>>(
                        OpNames::I, SiteIndex(), xm->info->delta_quantum);
                } else {
                    site_op_names[cd] = make_shared<OpElement<S, FL>>(
                        m == 0 || m == n_sites - 1
                            ? (m == 0 ? OpNames::XL : OpNames::XR)
                            : OpNames::X,
                        SiteIndex({(uint16_t)(ixx / 1000 / 1000),
                                   (uint16_t)(ixx / 1000 % 1000),
                                   (uint16_t)(ixx % 1000)},
                                  {}),
                        xm->info->delta_quantum);
                    ixx++;
                }
                if (xm->factor == (FL)0.0 || xm->info->n == 0 ||
                    xm->norm() < TINY)
                    site_op_names[cd] = nullptr;
                else
                    opt->ops[site_op_names.at(cd)] = xm;
            }
            vector<shared_ptr<OpElement<S, FL>>> site_mp(site_ops.size());
            for (int i = 0; i < (int)scheme->local_terms.size(); i++) {
                site_mp[i] = site_op_names[scheme->local_terms[i]];
                if (iprint >= 2) {
                    cout << " - LOCAL  [" << setw(5) << i << "] :: ";
                    cout << scheme->local_terms[i];
                    if (site_op_names[scheme->local_terms[i]] != nullptr)
                        cout << " "
                             << site_op_names[scheme->local_terms[i]]->q_label;
                    cout << endl;
                }
            }
            // construct left mpo tensor
            vector<LL> prev_idxs;
            prev_idxs.resize(scheme->left_terms.size() + 1, 0);
            for (int i = 0; i < (int)scheme->left_terms.size(); i++)
                prev_idxs[i + 1] =
                    prev_idxs[i] +
                    counter->count_left(scheme->left_terms[i].first.first,
                                        m - 1, scheme->left_terms[i].second);
            acc_cnt = 0;
            for (int i = 0; i < (int)scheme->left_terms.size(); i++) {
                int cnt = counter->count_left(scheme->left_terms[i].first.first,
                                              m, scheme->left_terms[i].second);
                if (cnt == 0)
                    continue;
                LL jacc_cnt = 0;
                if (iprint >= 2)
                    cout << " -  LEFT  BLOCKING  [" << setw(5) << i << "] :: ";
                for (int j = 0; j < (int)scheme->left_blocking[i].size();
                     j += 2) {
                    uint32_t lidx = scheme->left_blocking[i][j];
                    uint32_t ridx = scheme->left_blocking[i][j + 1];
                    LL jcnt = prev_idxs[lidx + 1] - prev_idxs[lidx];
                    if (site_mp[ridx] != nullptr)
                        for (LL k = 0; k < jcnt; k++)
                            (*plmat)[{(int)(prev_idxs[lidx] + k),
                                      (int)(jacc_cnt + acc_cnt + k)}] =
                                site_mp[ridx];
                    if (iprint >= 2)
                        for (LL k = 0; k < jcnt; k++)
                            cout << prev_idxs[lidx] + k << "+" << ridx << "="
                                 << jacc_cnt + acc_cnt + k << " / ";
                    jacc_cnt += jcnt;
                }
                if (iprint >= 2)
                    cout << endl;
                assert(jacc_cnt == cnt);
                acc_cnt += cnt;
            }
            // construct right mpo tensor
            prev_idxs.resize(scheme->right_terms.size() + 1, 0);
            for (int i = 0; i < (int)scheme->right_terms.size(); i++)
                prev_idxs[i + 1] =
                    prev_idxs[i] +
                    counter->count_right(scheme->right_terms[i].first.first,
                                         m + 1);
            acc_cnt = 0;
            for (int i = 0; i < (int)scheme->right_terms.size(); i++) {
                int cnt =
                    counter->count_right(scheme->right_terms[i].first.first, m);
                if (cnt == 0)
                    continue;
                if (iprint >= 2)
                    cout << " - RIGHT  BLOCKING  [" << setw(5) << i << "] :: ";
                LL jacc_cnt = 0;
                for (int j = 0; j < (int)scheme->right_blocking[i].size();
                     j += 2) {
                    uint32_t lidx = scheme->right_blocking[i][j];
                    uint32_t ridx = scheme->right_blocking[i][j + 1];
                    LL jcnt = prev_idxs[ridx + 1] - prev_idxs[ridx];
                    if (site_mp[lidx] != nullptr)
                        for (LL k = 0; k < jcnt; k++)
                            (*prmat)[{(int)(jacc_cnt + acc_cnt + k),
                                      (int)(prev_idxs[ridx] + k)}] =
                                site_mp[lidx];
                    if (iprint >= 2)
                        for (LL k = 0; k < jcnt; k++)
                            cout << lidx << "+" << prev_idxs[ridx] + k << "="
                                 << jacc_cnt + acc_cnt + k << " / ";
                    jacc_cnt += jcnt;
                }
                if (iprint >= 2)
                    cout << endl;
                assert(jacc_cnt == cnt);
                acc_cnt += cnt;
            }
            if (m == n_sites - 1)
                for (int i = 0; i < (int)scheme->last_right_terms.size(); i++) {
                    int cnt = counter->count_right(
                        scheme->last_right_terms[i].first.first, m);
                    if (iprint >= 2)
                        cout << " - RIGHT* BLOCKING  [" << setw(5) << i
                             << "] :: ";
                    LL jacc_cnt = 0;
                    for (int j = 0;
                         j < (int)scheme->last_right_blocking[i].size();
                         j += 2) {
                        uint32_t ridx = scheme->last_right_blocking[i][j];
                        if (site_mp[ridx] != nullptr)
                            (*prmat)[{(int)(jacc_cnt + acc_cnt), 0}] =
                                site_mp[ridx];
                        if (iprint >= 2)
                            cout << ridx << "+" << 0 << "="
                                 << jacc_cnt + acc_cnt << " / ";
                        jacc_cnt++;
                    }
                    if (iprint >= 2)
                        cout << endl;
                    assert(jacc_cnt == cnt);
                    acc_cnt += cnt;
                }
            tsite = tnmid = _t.get_time();
            if (m == 0) {
                if (iprint) {
                    cout << " M = " << setw(12) << 0;
                    cout << " Tmid = " << fixed << setprecision(3)
                         << tsite - tnmid;
                    cout << " T = " << fixed << setprecision(3) << tsite;
                    cout << endl;
                    tsite_total += tsite;
                }
                continue;
            }
            // middle operators
            LL mshape = 0;
            for (int i = 0; i < scheme->middle_blocking.size(); i++) {
                uint32_t lx = scheme->middle_blocking[i][0].first;
                uint32_t rx = scheme->middle_blocking[i][0].second;
                LL lcnt = counter->count_left(
                    scheme->left_terms[lx].first.first, m - 1, true);
                LL rcnt = counter->count_right(
                    scheme->right_terms[rx].first.first, m);
                for (auto &r :
                     middle_patterns.at(scheme->middle_perm_patterns[i]))
                    for (auto &rr : scheme->perms[r.first]->data[r.second])
                        mshape += (lcnt * rcnt) * rr.second.size();
            }
            if (m == n_sites - 1)
                for (int i = 0; i < scheme->last_middle_blocking.size(); i++)
                    if (scheme->last_middle_blocking[i].size() != 0) {
                        uint32_t lx = scheme->last_middle_blocking[i][0].first;
                        uint32_t rx = scheme->last_middle_blocking[i][0].second;
                        assert(scheme->left_terms[lx].second == false ||
                               scheme->left_terms[lx].first.first.size() == 0);
                        LL lcnt = counter->count_left(
                            scheme->left_terms[lx].first.first, m - 1, false);
                        LL rcnt = counter->count_right(
                            rx < scheme->right_terms.size()
                                ? scheme->right_terms[rx].first.first
                                : scheme
                                      ->last_right_terms
                                          [rx - scheme->right_terms.size()]
                                      .first.first,
                            m);
                        for (auto &r : middle_patterns.at(
                                 scheme->middle_perm_patterns[i]))
                            for (auto &rr :
                                 scheme->perms[r.first]->data[r.second])
                                mshape += (lcnt * rcnt) * rr.second.size();
                    }
            if (iprint) {
                cout << " M = " << setw(12) << mshape;
                cout.flush();
            }
            if (iprint >= 2)
                cout << endl;
            shared_ptr<SymbolicColumnVector<S>> pmop =
                make_shared<SymbolicColumnVector<S>>(mshape);
            shared_ptr<SymbolicColumnVector<S>> pmexpr =
                make_shared<SymbolicColumnVector<S>>(mshape);
            LL im = 0;
            shared_ptr<SymbolicRowVector<S>> pmlop =
                dynamic_pointer_cast<SymbolicRowVector<S>>(
                    left_operator_names[m - 1]);
            shared_ptr<SymbolicColumnVector<S>> pmrop =
                dynamic_pointer_cast<SymbolicColumnVector<S>>(
                    right_operator_names[m]);
            int middle_count = (int)scheme->middle_blocking.size();
            int middle_base_count = middle_count;
            if (m == n_sites - 1)
                middle_count += (int)scheme->last_middle_blocking.size();
            for (int ii = 0; ii < middle_count; ii++) {
                bool is_last = ii >= middle_base_count;
                int i = is_last ? ii - middle_base_count : ii;
                if (is_last && scheme->last_middle_blocking[i].size() == 0)
                    continue;
                map<string, int> middle_cd_map;
                for (int j = 0; j < (int)scheme->middle_terms[i].size(); j++)
                    middle_cd_map[scheme->middle_terms[i][j]] = j;
                for (auto &r :
                     middle_patterns.at(scheme->middle_perm_patterns[i]))
                    for (auto &pr : scheme->perms[r.first]->data[r.second])
                        for (auto &prr : pr.second) {
                            if (iprint >= 2) {
                                cout << (is_last ? " - MIDDLE*" : " - MIDDLE ")
                                     << " BLOCKING  [" << setw(5) << r.first
                                     << "/" << setw(5) << r.second << "] ";
                                for (auto &g : scheme->middle_perm_patterns[i])
                                    cout << g << " ";
                                cout << prr.second;
                                cout << " -> ";
                                for (auto &g : pr.first)
                                    cout << g << " ";
                                cout << ":: ";
                            }
                            int jj = middle_cd_map[prr.second];
                            uint32_t lx =
                                is_last
                                    ? scheme->last_middle_blocking[i][jj].first
                                    : scheme->middle_blocking[i][jj].first;
                            uint32_t rx =
                                is_last
                                    ? scheme->last_middle_blocking[i][jj].second
                                    : scheme->middle_blocking[i][jj].second;
                            vector<LL> plidxs, pridxs;
                            plidxs.resize(scheme->left_terms.size() + 1, 0);
                            for (int k = 0; k < (int)scheme->left_terms.size();
                                 k++)
                                plidxs[k + 1] =
                                    plidxs[k] +
                                    counter->count_left(
                                        scheme->left_terms[k].first.first,
                                        m - 1, scheme->left_terms[k].second);
                            pridxs.resize(scheme->right_terms.size() + 1, 0);
                            for (int k = 0; k < (int)scheme->right_terms.size();
                                 k++)
                                pridxs[k + 1] =
                                    pridxs[k] +
                                    counter->count_right(
                                        scheme->right_terms[k].first.first, m);
                            if (is_last) {
                                pridxs.resize(
                                    scheme->right_terms.size() +
                                        scheme->last_right_terms.size() + 1,
                                    0);
                                for (int k = 0;
                                     k < (int)scheme->last_right_terms.size();
                                     k++)
                                    pridxs[k + 1 + scheme->right_terms.size()] =
                                        pridxs[k + scheme->right_terms.size()] +
                                        counter->count_right(
                                            scheme->last_right_terms[k]
                                                .first.first,
                                            m);
                            }
                            vector<uint16_t> perm =
                                SpinPermTensor::find_pattern_perm(pr.first);
                            vector<uint16_t> lxx, rxx, mxx(pr.first.size());
                            vector<uint16_t> rpat =
                                rx < scheme->right_terms.size()
                                    ? scheme->right_terms[rx].first.first
                                    : scheme
                                          ->last_right_terms
                                              [rx - scheme->right_terms.size()]
                                          .first.first;
                            LL lcnt = counter->count_left(
                                scheme->left_terms[lx].first.first, m - 1,
                                !is_last);
                            LL rcnt = counter->count_right(rpat, m);
                            LL lshift =
                                !scheme->left_terms[lx].second && !is_last
                                    ? counter->count_left(
                                          scheme->left_terms[lx].first.first,
                                          m - 1,
                                          scheme->left_terms[lx].second) -
                                          lcnt
                                    : 0;
                            counter->init_left(
                                scheme->left_terms[lx].first.first, m - 1,
                                !is_last, lxx);
                            for (LL il = 0; il < lcnt; il++) {
                                counter->init_right(rpat, m, rxx);
                                for (LL ir = 0; ir < rcnt; ir++) {
                                    LL ixx = 0;
                                    for (int k = 0; k < (int)pr.first.size();
                                         k++) {
                                        mxx[k] =
                                            perm[k] < lxx.size()
                                                ? lxx[perm[k]]
                                                : rxx[perm[k] - lxx.size()];
                                        ixx = ixx * n_sites + mxx[k];
                                    }
                                    S q = hamil->get_string_quantum(
                                        scheme->middle_terms[i][jj], &mxx[0]);
                                    if (iprint >= 2) {
                                        for (auto &g : mxx)
                                            cout << g << " ";
                                        cout << ": " << plidxs[lx] + lshift + il
                                             << "+" << pridxs[rx] + ir << " (*"
                                             << fixed << setprecision(3)
                                             << setw(6) << prr.first << ") ";
                                    }
                                    (*pmop)[im] = make_shared<OpElement<S, FL>>(
                                        OpNames::XPDM,
                                        SiteIndex(
                                            {(uint16_t)(ixx >> 36),
                                             (uint16_t)((ixx >> 24) & 0xFFFLL),
                                             (uint16_t)((ixx >> 12) & 0xFFFLL),
                                             (uint16_t)(ixx & 0xFFFLL)},
                                            {(uint8_t)r.first}),
                                        q);
                                    (*pmexpr)[im] =
                                        (FL)prr.first *
                                        (*pmlop)[plidxs[lx] + lshift + il] *
                                        (*pmrop)[pridxs[rx] + ir];
                                    counter->next_right(rpat, m, rxx);
                                    im++;
                                }
                                counter->next_left(
                                    scheme->left_terms[lx].first.first, m - 1,
                                    lxx);
                            }
                            if (iprint >= 2)
                                cout << endl;
                        }
            }
            assert(im == mshape);
            middle_operator_names.push_back(pmop);
            middle_operator_exprs.push_back(pmexpr);
            if (iprint) {
                tsite += _t.get_time();
                cout << " Tmid = " << fixed << setprecision(3) << tsite - tnmid;
                cout << " T = " << fixed << setprecision(3) << tsite;
                cout << endl;
                tsite_total += tsite;
            }
        }
        if (iprint)
            cout << "Ttotal = " << fixed << setprecision(3) << setw(10)
                 << tsite_total << endl;
    }
    virtual ~GeneralNPDMMPO() = default;
};

} // namespace block2
