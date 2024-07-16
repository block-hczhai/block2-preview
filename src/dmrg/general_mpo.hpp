
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

/** Automatic construction of MPO. */

#pragma once

#include "../core/flow.hpp"
#include "../core/fp_codec.hpp"
#include "general_hamiltonian.hpp"
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

enum struct MPOAlgorithmTypes : uint16_t {
    None = 0,
    Bipartite = 1,
    SVD = 2,
    Rescaled = 4,
    Fast = 8,
    Blocked = 16,
    Sum = 32,
    Constrained = 64,
    Disjoint = 128,
    NC = 256,
    CN = 512,
    Length = 1024,
    DisjointSVD = 128 | 2,
    BlockedSumDisjointSVD = 128 | 32 | 16 | 2,
    FastBlockedSumDisjointSVD = 128 | 32 | 16 | 8 | 2,
    BlockedRescaledSumDisjointSVD = 128 | 32 | 16 | 4 | 2,
    FastBlockedRescaledSumDisjointSVD = 128 | 32 | 16 | 8 | 4 | 2,
    BlockedDisjointSVD = 128 | 16 | 2,
    FastBlockedDisjointSVD = 128 | 16 | 8 | 2,
    BlockedRescaledDisjointSVD = 128 | 16 | 4 | 2,
    FastBlockedRescaledDisjointSVD = 128 | 16 | 8 | 4 | 2,
    RescaledDisjointSVD = 128 | 4 | 2,
    FastDisjointSVD = 128 | 8 | 2,
    FastRescaledDisjointSVD = 128 | 8 | 4 | 2,
    ConstrainedSVD = 64 | 2,
    BlockedSumConstrainedSVD = 64 | 32 | 16 | 2,
    FastBlockedSumConstrainedSVD = 64 | 32 | 16 | 8 | 2,
    BlockedRescaledSumConstrainedSVD = 64 | 32 | 16 | 4 | 2,
    FastBlockedRescaledSumConstrainedSVD = 64 | 32 | 16 | 8 | 4 | 2,
    BlockedConstrainedSVD = 64 | 16 | 2,
    FastBlockedConstrainedSVD = 64 | 16 | 8 | 2,
    BlockedRescaledConstrainedSVD = 64 | 16 | 4 | 2,
    FastBlockedRescaledConstrainedSVD = 64 | 16 | 8 | 4 | 2,
    RescaledConstrainedSVD = 64 | 4 | 2,
    FastConstrainedSVD = 64 | 8 | 2,
    FastRescaledConstrainedSVD = 64 | 8 | 4 | 2,
    BlockedSumSVD = 32 | 16 | 2,
    FastBlockedSumSVD = 32 | 16 | 8 | 2,
    BlockedRescaledSumSVD = 32 | 16 | 4 | 2,
    FastBlockedRescaledSumSVD = 32 | 16 | 8 | 4 | 2,
    BlockedSumBipartite = 32 | 16 | 1,
    FastBlockedSumBipartite = 32 | 16 | 8 | 1,
    BlockedSVD = 16 | 2,
    FastBlockedSVD = 16 | 8 | 2,
    BlockedRescaledSVD = 16 | 4 | 2,
    FastBlockedRescaledSVD = 16 | 8 | 4 | 2,
    BlockedLengthSVD = 16 | 2 | 1024,
    FastBlockedLengthSVD = 16 | 8 | 2 | 1024,
    BlockedBipartite = 16 | 1,
    FastBlockedBipartite = 16 | 8 | 1,
    RescaledSVD = 4 | 2,
    FastSVD = 8 | 2,
    FastRescaledSVD = 8 | 4 | 2,
    FastBipartite = 8 | 1,
};

inline bool operator&(MPOAlgorithmTypes a, MPOAlgorithmTypes b) {
    return ((uint16_t)a & (uint16_t)b) != 0;
}

inline MPOAlgorithmTypes operator|(MPOAlgorithmTypes a, MPOAlgorithmTypes b) {
    return MPOAlgorithmTypes((uint16_t)a | (uint16_t)b);
}

inline ostream &operator<<(ostream &os, const MPOAlgorithmTypes c) {
    if (c == MPOAlgorithmTypes::NC)
        os << "NC";
    else if (c == MPOAlgorithmTypes::CN)
        os << "CN";
    else if (c == MPOAlgorithmTypes::None)
        os << "None";
    else if (c == MPOAlgorithmTypes::Rescaled)
        os << "Res";
    else if (c == MPOAlgorithmTypes::Fast)
        os << "Fast";
    else if (c == MPOAlgorithmTypes::Blocked)
        os << "Blocked";
    else if (c == MPOAlgorithmTypes::Sum)
        os << "Sum";
    else if (c == MPOAlgorithmTypes::Constrained)
        os << "Constrained";
    else if (c == MPOAlgorithmTypes::Disjoint)
        os << "Disjoint";
    else if (c == MPOAlgorithmTypes::Length)
        os << "Length";
    else {
        if (c & MPOAlgorithmTypes::Fast)
            os << "Fast";
        if (c & MPOAlgorithmTypes::Blocked)
            os << "Blocked";
        if (c & MPOAlgorithmTypes::Rescaled)
            os << "Res";
        if (c & MPOAlgorithmTypes::Sum)
            os << "Sum";
        if (c & MPOAlgorithmTypes::Constrained)
            os << "Cons";
        if (c & MPOAlgorithmTypes::Disjoint)
            os << "Dis";
        if (c & MPOAlgorithmTypes::Length)
            os << "Len";
        if (c & MPOAlgorithmTypes::Bipartite)
            os << "BIP";
        if (c & MPOAlgorithmTypes::SVD)
            os << "SVD";
    }
    return os;
}

template <typename S, typename FL> struct GeneralMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    typedef long long int LL;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::tensors;
    using MPO<S, FL>::left_operator_names;
    using MPO<S, FL>::right_operator_names;
    using MPO<S, FL>::basis;
    MPOAlgorithmTypes algo_type;
    vector<FP> discarded_weights;
    shared_ptr<GeneralFCIDUMP<FL>> afd;
    FP cutoff;
    int max_bond_dim;
    int iprint;
    S left_vacuum = S(S::invalid);
    int sum_mpo_mod = -1;
    bool compute_accurate_svd_error = true;
    FP csvd_sparsity = (FP)0.0;
    FP csvd_eps = (FP)1E-10;
    int csvd_max_iter = 1000;
    vector<FP> disjoint_levels;
    bool disjoint_all_blocks = false;
    FP disjoint_multiplier = (FP)1.0;
    bool block_max_length = false;   // separate 1e/2e terms
    bool fast_no_orb_dep_op = false; // fast mode for no orb_sym case
    static inline size_t expr_index_hash(const string &expr,
                                         const uint16_t *terms, int n,
                                         const uint16_t init = 0) noexcept {
        size_t h = (size_t)init;
        h ^= hash<string>{}(expr) + 0x9E3779B9 + (h << 6) + (h >> 2);
        for (int i = 0; i < n; i++)
            h ^= terms[i] + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    GeneralMPO(const shared_ptr<GeneralHamiltonian<S, FL>> &hamil,
               const shared_ptr<GeneralFCIDUMP<FL>> &afd,
               MPOAlgorithmTypes algo_type, FP cutoff = (FP)0.0,
               int max_bond_dim = -1, int iprint = 1, const string &tag = "HQC")
        : MPO<S, FL>(hamil->n_sites, tag), afd(afd), algo_type(algo_type),
          cutoff(cutoff), max_bond_dim(max_bond_dim), iprint(iprint) {
        MPO<S, FL>::hamil = hamil;
    }
    void build() override {
        bool rescale = algo_type & MPOAlgorithmTypes::Rescaled;
        bool fast = algo_type & MPOAlgorithmTypes::Fast;
        bool blocked = algo_type & MPOAlgorithmTypes::Blocked;
        bool sum_mpo = algo_type & MPOAlgorithmTypes::Sum;
        bool constrain = algo_type & MPOAlgorithmTypes::Constrained;
        bool disjoint = algo_type & MPOAlgorithmTypes::Disjoint;
        bool length = algo_type & MPOAlgorithmTypes::Length;
        if (!disjoint)
            disjoint_multiplier = (FP)1.0;
        if (!(algo_type & MPOAlgorithmTypes::SVD) && max_bond_dim != -1)
            throw runtime_error(
                "Max bond dimension can only be used together with SVD!");
        else if (!(algo_type & MPOAlgorithmTypes::SVD) && rescale)
            throw runtime_error(
                "Rescaling can only be used together with SVD!");
        else if (!(algo_type & MPOAlgorithmTypes::SVD) && constrain)
            throw runtime_error(
                "Constrained can only be used together with SVD!");
        else if (!(algo_type & MPOAlgorithmTypes::SVD) && disjoint)
            throw runtime_error("Disjoint can only be used together with SVD!");
        else if ((algo_type & MPOAlgorithmTypes::NC) &&
                 algo_type != MPOAlgorithmTypes::NC)
            throw runtime_error("Invalid MPO algorithm type with NC!");
        else if ((algo_type & MPOAlgorithmTypes::CN) &&
                 algo_type != MPOAlgorithmTypes::CN)
            throw runtime_error("Invalid MPO algorithm type with CN!");
        else if (algo_type == MPOAlgorithmTypes::None)
            throw runtime_error("Invalid MPO algorithm None!");
        shared_ptr<GeneralHamiltonian<S, FL>> hamil =
            dynamic_pointer_cast<GeneralHamiltonian<S, FL>>(MPO<S, FL>::hamil);
        MPO<S, FL>::const_e = afd->e();
        MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        n_sites = (int)hamil->n_sites;
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        basis = hamil->basis;
        left_operator_names.resize(n_sites, nullptr);
        right_operator_names.resize(n_sites, nullptr);
        tensors.resize(n_sites, nullptr);
        discarded_weights.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            tensors[m] = make_shared<OperatorTensor<S, FL>>();
        S vacuum = hamil->vacuum;
        // length of each term; starting index of each term
        // at the beginning, term_i is all zero
        vector<uint16_t> term_l(afd->exprs.size());
        vector<vector<uint16_t>> term_i(afd->exprs.size());
        vector<vector<uint16_t>> term_k(afd->exprs.size());
        LL n_terms = 0;
        int max_term_l = 0;
        for (int ix = 0; ix < (int)afd->exprs.size(); ix++) {
            const int nn = SpinPermRecoupling::count_cds(afd->exprs[ix]);
            term_l[ix] = nn;
            term_i[ix].resize(afd->data[ix].size(), 0);
            term_k[ix].resize(afd->data[ix].size(), -1);
            assert(afd->indices[ix].size() == nn * term_i[ix].size());
            n_terms += (LL)afd->data[ix].size();
            max_term_l = max(max_term_l, nn);
        }
        vector<vector<S>> quanta_ref =
            hamil->init_string_quanta(afd->exprs, term_l, left_vacuum);
        S qh = hamil->vacuum, actual_qh = qh;
        left_vacuum = hamil->vacuum;
        for (int ix = 0; ix < (int)afd->exprs.size(); ix++) {
            if (afd->data[ix].size() != 0) {
                qh = hamil
                         ->get_string_quanta(quanta_ref[ix], afd->exprs[ix],
                                             &afd->indices[ix][0], term_l[ix])
                         .first;
                auto pl = hamil->get_string_quanta(
                    quanta_ref[ix], afd->exprs[ix], &afd->indices[ix][0], 0);
                left_vacuum = pl.first, actual_qh = pl.second;
                break;
            }
        }
        // cout << "qh = " << qh << " left_vac = " << left_vacuum
        //      << " actual_qh = " << actual_qh << endl;
        assert(qh.combine(left_vacuum, -actual_qh) != S(S::invalid));
        vector<S> left_q = vector<S>{qh.combine(left_vacuum, -actual_qh)};
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S, FL>>(OpNames::H, SiteIndex(), qh);
        MPO<S, FL>::op = dynamic_pointer_cast<OpElement<S, FL>>(h_op);
        MPO<S, FL>::left_vacuum = left_vacuum;
        if (iprint) {
            cout << endl;
            cout << "Build MPO | Nsites = " << setw(5) << n_sites
                 << " | Nterms = " << setw(10) << n_terms
                 << " | Algorithm = " << algo_type
                 << " | Cutoff = " << scientific << setw(8) << setprecision(2)
                 << cutoff;
            if (algo_type & MPOAlgorithmTypes::SVD)
                cout << " | Max bond dimension = " << setw(5) << max_bond_dim;
            cout << endl;
            if (block_max_length)
                cout << " | BlockMaxLen = T";
            if (sum_mpo)
                cout << " | SumMPO : Mod = " << setw(5) << sum_mpo_mod;
            if (disjoint) {
                cout << " | Disjoint : All blocks = "
                     << (disjoint_all_blocks ? "T" : "F");
                cout << " Multiplier = " << setw(5) << fixed << setprecision(2)
                     << disjoint_multiplier;
                if (disjoint_levels.size() > 0) {
                    cout << " Levels =";
                    for (auto &dl : disjoint_levels)
                        cout << " " << scientific << setw(8) << setprecision(2)
                             << dl;
                }
            }
            if (block_max_length || sum_mpo || disjoint)
                cout << endl;
        }
        // index of current terms
        // in future, cur_terms should have an extra level
        // indicating the term length
        vector<vector<pair<int, LL>>> cur_terms(1);
        vector<vector<FL>> cur_values(1);
        cur_terms[0].resize(n_terms);
        cur_values[0].resize(n_terms);
        size_t ik = 0;
        for (int ix = 0; ix < (int)afd->exprs.size(); ix++)
            for (size_t it = 0; it < afd->data[ix].size(); it++) {
                cur_terms[0][ik] = make_pair(ix, (LL)it);
                cur_values[0][ik] = afd->data[ix][it];
                ik++;
            }
        assert(ik == n_terms);
        vector<pair<int, LL>> part_terms;
        vector<FL> part_values;
        vector<LL> part_indices;
        // to save time, divide O(K^4) terms into K groups
        // for each iteration on site k, only O(K^3) terms are processed
        if (fast) {
            vector<pair<int, LL>> ext_cur_terms;
            vector<FL> ext_cur_values;
            vector<LL> part_count(n_sites, 0);
            for (LL ik = 0; ik < n_terms; ik++) {
                int ix = cur_terms[0][ik].first;
                LL it = cur_terms[0][ik].second;
                if (term_l[ix] != 0)
                    part_count[afd->indices[ix][it * term_l[ix]]]++;
                else {
                    ext_cur_terms.push_back(cur_terms[0][ik]);
                    ext_cur_values.push_back(cur_values[0][ik]);
                }
            }
            LL part_n_terms = 0;
            for (int ii = 0; ii < n_sites; ii++)
                part_n_terms += part_count[ii];
            part_indices.resize(n_sites + 1, 0);
            // here part_indices[ii + 1] will later be increased to presum upto
            // part_count[ii]
            for (int ii = 1; ii < n_sites; ii++)
                part_indices[ii + 1] = part_indices[ii] + part_count[ii - 1];
            part_terms.resize(part_n_terms);
            part_values.resize(part_n_terms);
            for (LL ik = 0; ik < n_terms; ik++) {
                int ix = cur_terms[0][ik].first;
                LL it = cur_terms[0][ik].second;
                if (term_l[ix] != 0) {
                    int ii = afd->indices[ix][it * term_l[ix]];
                    LL x = part_indices[ii + 1]++;
                    part_terms[x] = cur_terms[0][ik];
                    part_values[x] = cur_values[0][ik];
                }
            }
            assert(part_indices[n_sites] == part_n_terms);
            cur_terms[0] = ext_cur_terms;
            cur_values[0] = ext_cur_values;
        }
        // do svd from left to right
        // time complexity: O(KDLN(log N))
        // K: n_sites, D: max_bond_dim, L: term_len, N: n_terms
        // using block-structure according to left q number
        // this is the map from left q number to its block index
        // pair(0 = left min 1 = right min >= 2 same min, (min term_l, max l))
        map<pair<pair<uint16_t, pair<uint16_t, uint16_t>>, S>, int> q_map;
        // for each iq block, a map from hashed repr of string of op in left
        // block to (mpo index, term index (in cur_terms), left block string of
        // op index)
        vector<unordered_map<size_t, vector<pair<pair<int, LL>, int>>>> map_ls;
        // for each iq block, a map from hashed repr of string of op in right
        // block to (mpo index, term index (in cur_terms), right block string of
        // op index)
        vector<unordered_map<size_t, vector<pair<pair<int, LL>, int>>>> map_rs;
        // sparse repr of the connection (edge) matrix for each block
        vector<vector<pair<pair<int, int>, FL>>> mats;
        // for each block, the nrow and ncol of the block
        vector<pair<LL, LL>> nms;
        // range of ip that should be svd/bip separately
        // only used in sum mode pair(start, end)
        vector<vector<int>> sparse_ranges;
        // cache of operator strings
        vector<map<pair<uint16_t, uint16_t>, string>> sub_exprs(
            afd->exprs.size());
        FL rsc_factor = 1;
        Timer _t, _t2;
        double tsite, tsvd, tsite_total = 0, tsvd_total = 0;
        FP dw_max = 0, error_total = 0;
        size_t nnz_total = 0, size_total = 0;
        int bond_max = 0;
        for (int ii = 0; ii < n_sites; ii++) {
            if (iprint) {
                cout << " Site = " << setw(5) << ii << " / " << setw(5)
                     << n_sites << " .. ";
                cout.flush();
            }
            _t.get_time();
            tsite = tsvd = 0;
            q_map.clear();
            map_ls.clear();
            map_rs.clear();
            mats.clear();
            nms.clear();
            LL delayed_term = -1, part_off = 0;
            vector<int> ip_sparse(cur_values.size(), ii);
            for (int isr = 0; isr < (int)sparse_ranges.size(); isr++) {
                auto &sr = sparse_ranges[isr];
                for (int j = 0; j < (int)sr.size(); j += 2)
                    for (int k = sr[j]; k < sr[j + 1]; k++)
                        ip_sparse[k] = isr;
            }
            FP eff_disjoint_multiplier =
                ii == n_sites - 1 ? (FP)1.0 : disjoint_multiplier;
            // Part 1: iter over all mpos
            for (int ip = 0; ip < (int)cur_values.size(); ip++) {
                LL cn = (LL)cur_terms[ip].size(), cnr = cn;
                // for part terms, we have two things:
                // (1) terms starting with the current index should be handled
                // (cnr ~ cn) (2) terms not starting with the current index
                // should be delayed here ip = 0 is fixed to be identity in the
                // left
                if (part_indices.size() != 0 && ip == 0) {
                    cn += part_indices[ii + 1] - part_indices[ii];
                    part_off = part_indices[ii] - cnr;
                    if (part_indices[ii + 1] != part_indices[n_sites]) {
                        // this represents all terms with starting index > ii
                        delayed_term = part_indices[ii + 1];
                        int ix = part_terms[delayed_term].first;
                        LL it = part_terms[delayed_term].second;
                        LL itt = it * term_l[ix];
                        term_k[ix][it] = 0;
                        if (!sub_exprs[ix].count(make_pair(0, 0)))
                            sub_exprs[ix][make_pair(0, 0)] =
                                GeneralHamiltonian<S, FL>::get_sub_expr(
                                    afd->exprs[ix], 0, 0);
                        if (!sub_exprs[ix].count(make_pair(0, term_l[ix])))
                            sub_exprs[ix][make_pair(0, term_l[ix])] =
                                GeneralHamiltonian<S, FL>::get_sub_expr(
                                    afd->exprs[ix], 0, term_l[ix]);
                        pair<S, S> pq =
                            fast_no_orb_dep_op
                                ? make_pair(quanta_ref[ix][0],
                                            quanta_ref[ix].back() -
                                                quanta_ref[ix][0])
                                : hamil->get_string_quanta(
                                      quanta_ref[ix], afd->exprs[ix],
                                      &afd->indices[ix][itt], 0);
                        q_map[make_pair(make_pair(0, make_pair(0, 0)),
                                        qh.combine(pq.first, -pq.second))] = 0;
                        map_ls.emplace_back();
                        map_rs.emplace_back();
                        mats.emplace_back();
                        nms.push_back(make_pair(1, 1));
                        map_ls[0][0].push_back(make_pair(make_pair(0, -1), 0));
                        map_rs[0][0].push_back(make_pair(make_pair(0, -1), 0));
                        mats[0].push_back(
                            make_pair(make_pair(0, 0),
                                      part_values[delayed_term] * rsc_factor));
                    }
                }
                for (LL ic = 0; ic < cn; ic++) {
                    LL ix, it;
                    FL itv;
                    if (ic < cnr) {
                        ix = cur_terms[ip][ic].first;
                        it = cur_terms[ip][ic].second;
                        itv = cur_values[ip][ic];
                    } else {
                        ix = part_terms[ic + part_off].first;
                        it = part_terms[ic + part_off].second;
                        itv = part_values[ic + part_off] * rsc_factor;
                    }
                    int ik = term_i[ix][it], k = ik, kmax = term_l[ix];
                    LL itt = it * kmax;
                    // separate the current product into two parts
                    // (left block part and right block part)
                    for (; k < kmax && afd->indices[ix][itt + k] <= ii; k++)
                        ;
                    int iw = k == kmax ? 0 : 1, iwl = kmax - k;
                    for (int iwk = k + 1; iwk < kmax; iwk++)
                        iw += (afd->indices[ix][itt + iwk] ==
                               afd->indices[ix][itt + iwk - 1]);
                    if (!sub_exprs[ix].count(make_pair(ik, k)))
                        sub_exprs[ix][make_pair(ik, k)] =
                            GeneralHamiltonian<S, FL>::get_sub_expr(
                                afd->exprs[ix], ik, k);
                    if (!sub_exprs[ix].count(make_pair(k, kmax)))
                        sub_exprs[ix][make_pair(k, kmax)] =
                            GeneralHamiltonian<S, FL>::get_sub_expr(
                                afd->exprs[ix], k, kmax);
                    // cout << "ip = " << ip << " ic = " << ic << " ix = " << ix
                    //      << " it = " << it << " ik = " << ik
                    //      << " kmax = " << kmax << " k = " << k
                    //      << " term = " << afd->exprs[ix];
                    // for (int gk = 0; gk < kmax; gk++)
                    //     cout << " " << afd->indices[ix][itt + gk];
                    // cout << " L = " << sub_exprs[ix].at(make_pair(ik, k))
                    //      << " R = " << sub_exprs[ix].at(make_pair(k, kmax));
                    // first right site position
                    term_k[ix][it] = k;
                    const string &lstr = sub_exprs[ix].at(make_pair(ik, k));
                    const string &rstr = sub_exprs[ix].at(make_pair(k, kmax));
                    size_t hl = expr_index_hash(
                        lstr, afd->indices[ix].data() + itt + ik, k - ik, ip);
                    size_t hr = expr_index_hash(
                        rstr, afd->indices[ix].data() + itt + k, kmax - k, 1);
                    pair<S, S> pq = fast_no_orb_dep_op
                                        ? make_pair(quanta_ref[ix][k],
                                                    quanta_ref[ix].back() -
                                                        quanta_ref[ix][k])
                                        : hamil->get_string_quanta(
                                              quanta_ref[ix], afd->exprs[ix],
                                              &afd->indices[ix][itt], k);
                    S qq = qh.combine(pq.first, -pq.second);
                    // possible error here due to unsymmetrized integral
                    assert(qq != S(S::invalid));
                    pair<uint16_t, uint16_t> pqq =
                        make_pair((uint16_t)0, (uint16_t)0);
                    if (constrain && kmax - k == k)
                        pqq = make_pair((uint16_t)2,
                                        min((uint16_t)k, (uint16_t)(kmax - k)));
                    if (blocked)
                        pqq = make_pair(kmax - k == k && kmax != 0
                                            ? (uint16_t)2
                                            : (uint16_t)(kmax - k > k),
                                        min((uint16_t)k, (uint16_t)(kmax - k)));
                    if (sum_mpo && kmax - k == k && k != 0)
                        pqq = make_pair(
                            (uint16_t)(2 + (sum_mpo_mod == -1
                                                ? ip_sparse[ip]
                                                : ip_sparse[ip] % sum_mpo_mod)),
                            (uint16_t)k);
                    pair<uint16_t, pair<uint16_t, uint16_t>> ppqq = make_pair(
                        pqq.first, make_pair(pqq.second, (uint16_t)0));
                    if (block_max_length && ii != n_sites - 1)
                        ppqq.second.second = (uint16_t)kmax;
                    if (length)
                        ppqq.second.second = (uint16_t)(iw * max_term_l + iwl);
                    if (q_map.count(make_pair(ppqq, qq)) == 0) {
                        const int nq = (int)q_map.size();
                        q_map[make_pair(ppqq, qq)] = nq;
                        map_ls.emplace_back();
                        map_rs.emplace_back();
                        mats.emplace_back();
                        nms.push_back(make_pair(0, 0));
                    }
                    int iq = q_map.at(make_pair(ppqq, qq));
                    LL il = -1, ir = -1;
                    LL &nml = nms[iq].first, &nmr = nms[iq].second;
                    auto &mpl = map_ls[iq];
                    auto &mpr = map_rs[iq];
                    if (mpl.count(hl)) {
                        int iq = 0;
                        auto &vq = mpl.at(hl);
                        for (; iq < (int)vq.size(); iq++) {
                            int vip = vq[iq].first.first, vix;
                            LL vic = vq[iq].first.second, vit;
                            if (vic >= (LL)cur_terms[vip].size()) {
                                vix = part_terms[vic + part_off].first;
                                vit = part_terms[vic + part_off].second;
                            } else if (vic != -1) {
                                vix = cur_terms[vip][vic].first;
                                vit = cur_terms[vip][vic].second;
                            } else {
                                vix = part_terms[delayed_term].first;
                                vit = part_terms[delayed_term].second;
                            }
                            LL vitt = vit * term_l[vix];
                            int vik = term_i[vix][vit], vk = term_k[vix][vit];
                            if (vip == ip && vk - vik == k - ik &&
                                equal(afd->indices[vix].data() + vitt + vik,
                                      afd->indices[vix].data() + vitt + vk,
                                      afd->indices[ix].data() + itt + ik) &&
                                ((vix == ix && vik == ik) ||
                                 lstr == sub_exprs[vix].at(make_pair(vik, vk))))
                                break;
                        }
                        if (iq == (int)vq.size())
                            vq.push_back(make_pair(make_pair(ip, ic),
                                                   (int)(il = nml++)));
                        else
                            il = vq[iq].second;
                    } else
                        mpl[hl].push_back(
                            make_pair(make_pair(ip, ic), (int)(il = nml++)));
                    if (mpr.count(hr)) {
                        int iq = 0;
                        auto &vq = mpr.at(hr);
                        for (; iq < (int)vq.size(); iq++) {
                            int vip = vq[iq].first.first, vix;
                            LL vic = vq[iq].first.second, vit;
                            if (vic >= (LL)cur_terms[vip].size()) {
                                vix = part_terms[vic + part_off].first;
                                vit = part_terms[vic + part_off].second;
                            } else if (vic != -1) {
                                vix = cur_terms[vip][vic].first;
                                vit = cur_terms[vip][vic].second;
                            } else {
                                vix = part_terms[delayed_term].first;
                                vit = part_terms[delayed_term].second;
                            }
                            LL vitt = vit * term_l[vix];
                            int vkmax = term_l[vix], vk = term_k[vix][vit];
                            if (vkmax - vk == kmax - k &&
                                equal(afd->indices[vix].data() + vitt + vk,
                                      afd->indices[vix].data() + vitt + vkmax,
                                      afd->indices[ix].data() + itt + k) &&
                                ((vix == ix && vk == k) ||
                                 rstr ==
                                     sub_exprs[vix].at(make_pair(vk, vkmax))))
                                break;
                        }
                        if (iq == (int)vq.size())
                            vq.push_back(make_pair(make_pair(ip, ic),
                                                   (int)(ir = nmr++)));
                        else
                            ir = vq[iq].second;
                    } else
                        mpr[hr].push_back(
                            make_pair(make_pair(ip, ic), (int)(ir = nmr++)));
                    // cout << "il = " << il << " ir = " << ir
                    //      << " ql = " << qq.get_bra(qh)
                    //      << " qr = " << -qq.get_ket() << endl;
                    mats[iq].push_back(
                        make_pair(make_pair((int)il, (int)ir), itv));
                }
            }
            // cout << "mats size = " << mats.size() << endl;
            // Part 2: svd or mvc
            vector<pair<array<vector<FL>, 2>, vector<FP>>> svds;
            vector<array<vector<int>, 2>> mvcs;
            if (algo_type & MPOAlgorithmTypes::Bipartite)
                mvcs.resize(q_map.size());
            else
                svds.resize(q_map.size());
            vector<S> qs(q_map.size());
            vector<int> pqx(q_map.size());
            for (auto &mq : q_map) {
                qs[mq.second] = mq.first.second;
                pqx[mq.second] = mq.first.first.first;
            }
            int s_kept_total = 0, nr_total = 0;
            FP res_s_sum = 0, res_factor = 1;
            size_t res_s_count = 0;
            for (auto &mq : q_map) {
                int iq = mq.second;
                auto &matvs = mats[iq];
                auto &nm = nms[iq];
                int szl = (int)nm.first, szr = (int)nm.second, szm;
                // cout << "iq = " << iq << " q = " << qs[iq] << " szl = " <<
                // szl
                //      << " szr = " << szr << endl;
                if (algo_type & MPOAlgorithmTypes::NC)
                    szm = ii == n_sites - 1 ? szr : szl;
                else if (algo_type & MPOAlgorithmTypes::CN)
                    szm = ii == 0 ? szl : szr;
                else // bipartitie / SVD
                    szm = (int)(min(szl, szr) * eff_disjoint_multiplier);
                if (!(algo_type & MPOAlgorithmTypes::Bipartite)) {
                    if (delayed_term != -1 && iq == 0)
                        szm =
                            (int)(min(szl - 1, szr) * eff_disjoint_multiplier) +
                            1;
                    svds[iq].first[0].resize((size_t)szm * szl);
                    svds[iq].second.resize(szm);
                    svds[iq].first[1].resize((size_t)szm * szr);
                }
                int s_kept = 0;
                if (algo_type & MPOAlgorithmTypes::Bipartite) { // bipartite
                    _t2.get_time();
                    Flow flow(szl + szr);
                    for (auto &lrv : matvs)
                        flow.resi[lrv.first.first][lrv.first.second + szl] = 1;
                    for (int i = 0; i < szl; i++)
                        flow.resi[szl + szr][i] = 1;
                    for (int i = 0; i < szr; i++)
                        flow.resi[szl + i][szl + szr + 1] = 1;
                    flow.mvc(0, szl, szl, szr, mvcs[iq][0], mvcs[iq][1]);
                    if (ii == n_sites - 1) {
                        assert(szr == 1);
                        mvcs[iq][0].resize(0);
                        mvcs[iq][1].resize(1);
                        mvcs[iq][1][0] = 0;
                    }
                    tsvd += _t2.get_time();
                    // delayed I * O(K^4) term must be of NC type
                    if (delayed_term != -1 && iq == 0) {
                        if ((mvcs[iq][0].size() == 0 || mvcs[iq][0][0] != 0))
                            mvcs[iq][0].push_back(0);
                        if (mvcs[iq][1].size() != 0 && mvcs[iq][1][0] == 0)
                            mvcs[iq][1] = vector<int>(mvcs[iq][1].begin() + 1,
                                                      mvcs[iq][1].end());
                    }
                    s_kept = (int)mvcs[iq][0].size() + (int)mvcs[iq][1].size();
                } else if (((algo_type & MPOAlgorithmTypes::NC) &&
                            ii != n_sites - 1) ||
                           ((algo_type & MPOAlgorithmTypes::CN) &&
                            ii == 0)) { // NC
                    memset(svds[iq].first[0].data(), 0,
                           sizeof(FL) * svds[iq].first[0].size());
                    memset(svds[iq].first[1].data(), 0,
                           sizeof(FL) * svds[iq].first[1].size());
                    for (auto &lrv : matvs)
                        svds[iq].first[1][(size_t)lrv.first.first * szr +
                                          lrv.first.second] += lrv.second;
                    for (int i = 0; i < szm; i++)
                        svds[iq].first[0][(size_t)i * szm + i] =
                            svds[iq].second[i] = 1;
                    s_kept = szm;
                } else if (((algo_type & MPOAlgorithmTypes::CN) && ii != 0) ||
                           ((algo_type & MPOAlgorithmTypes::NC) &&
                            ii == n_sites - 1)) { // CN
                    memset(svds[iq].first[0].data(), 0,
                           sizeof(FL) * svds[iq].first[0].size());
                    memset(svds[iq].first[1].data(), 0,
                           sizeof(FL) * svds[iq].first[1].size());
                    for (auto &lrv : matvs)
                        svds[iq].first[0][(size_t)lrv.first.first * szr +
                                          lrv.first.second] += lrv.second;
                    for (int i = 0; i < szm; i++)
                        svds[iq].first[1][(size_t)i * szr + i] =
                            svds[iq].second[i] = 1;
                    s_kept = szm;
                } else { // SVD
                    _t2.get_time();
                    vector<FL> mat((size_t)szl * szr, 0);
                    if (delayed_term != -1 && iq == 0) {
                        for (auto &lrv : matvs)
                            if (lrv.first.first == 0)
                                svds[iq].first[1][lrv.first.second] +=
                                    lrv.second;
                            else
                                mat[(size_t)(lrv.first.first - 1) * szr +
                                    lrv.first.second] += lrv.second;
                        szl--;
                        svds[iq].second[0] = 1;
                        svds[iq].first[0][0] = 1;
                        threading->activate_global_mkl();
                        if ((pqx[iq] >= 2 || disjoint_all_blocks) && disjoint)
                            IterativeMatrixFunctions<FL>::disjoint_svd(
                                GMatrix<FL>(mat.data(), szl, szr),
                                GMatrix<FL>(svds[iq].first[0].data() + 1 + szm,
                                            szl, szm),
                                GMatrix<FP>(svds[iq].second.data() + 1, 1,
                                            szm - 1),
                                GMatrix<FL>(svds[iq].first[1].data() + szr,
                                            szm - 1, szr),
                                disjoint_levels, false, iprint >= 2);
                        else
                            GMatrixFunctions<FL>::svd(
                                GMatrix<FL>(mat.data(), szl, szr),
                                GMatrix<FL>(svds[iq].first[0].data() + 1 + szm,
                                            szl, szm),
                                GMatrix<FP>(svds[iq].second.data() + 1, 1,
                                            szm - 1),
                                GMatrix<FL>(svds[iq].first[1].data() + szr,
                                            szm - 1, szr));
                        threading->activate_normal();
                        szl++;
                    } else {
                        for (auto &lrv : matvs)
                            mat[(size_t)lrv.first.first * szr +
                                lrv.first.second] += lrv.second;
                        // cout << "mat = " << GMatrix<FL>(mat.data(), szl, szr)
                        // << endl;
                        threading->activate_global_mkl();
                        if ((pqx[iq] >= 2 || disjoint_all_blocks) && disjoint)
                            IterativeMatrixFunctions<FL>::disjoint_svd(
                                GMatrix<FL>(mat.data(), szl, szr),
                                GMatrix<FL>(svds[iq].first[0].data(), szl, szm),
                                GMatrix<FP>(svds[iq].second.data(), 1, szm),
                                GMatrix<FL>(svds[iq].first[1].data(), szm, szr),
                                disjoint_levels, false, iprint >= 2);
                        else
                            GMatrixFunctions<FL>::svd(
                                GMatrix<FL>(mat.data(), szl, szr),
                                GMatrix<FL>(svds[iq].first[0].data(), szl, szm),
                                GMatrix<FP>(svds[iq].second.data(), 1, szm),
                                GMatrix<FL>(svds[iq].first[1].data(), szm,
                                            szr));
                        threading->activate_normal();
                        // cout << "l = " <<
                        // GMatrix<FL>(svds[iq].first[0].data(), szl, szm) <<
                        // endl; cout << "s = " <<
                        // GMatrix<FP>(svds[iq].second.data(), 1, szm) << endl;
                        // cout
                        // << "r = " << GMatrix<FL>(svds[iq].first[1].data(),
                        // szm, szr) << endl;
                    }
                    res_s_sum +=
                        accumulate(svds[iq].second.begin(),
                                   svds[iq].second.end(), (FP)0, plus<FP>());
                    res_s_count += svds[iq].second.size();
                    if (!rescale) {
                        for (int i = 0; i < szm; i++)
                            if (svds[iq].second[i] > sqrt(cutoff))
                                s_kept++;
                            else
                                discarded_weights[ii] +=
                                    svds[iq].second[i] * svds[iq].second[i];
                        if (max_bond_dim >= 1)
                            s_kept = min(s_kept, max_bond_dim);
                        svds[iq].second.resize(s_kept);
                    } else
                        s_kept = szm;
                    tsvd += _t2.get_time();
                }
                s_kept_total += s_kept;
                nr_total += szr;
            }
            if (ii == n_sites - 1 && s_kept_total == 0)
                throw runtime_error("Empty Hamiltonian!");
            else if (ii == n_sites - 1 && s_kept_total != 1)
                throw runtime_error(
                    "Hamiltonian may contain multiple total symmetry blocks "
                    "(small integral elements violating point group "
                    "symmetry)!");
            if (rescale) {
                s_kept_total = 0;
                res_factor = res_s_sum / res_s_count;
                // keep only 1 significant digit
                typename FPtraits<FP>::U rrepr =
                    (typename FPtraits<FP>::U &)res_factor &
                    ~(((typename FPtraits<FP>::U)1 << FPtraits<FP>::mbits) - 1);
                res_factor = (FP &)rrepr;
                if (res_factor == 0)
                    res_factor = 1;
                for (auto &mq : q_map) {
                    int s_kept = 0;
                    int iq = mq.second;
                    auto &nm = nms[iq];
                    int szl = (int)nm.first, szr = (int)nm.second;
                    int szm = (int)(min(szl, szr) * eff_disjoint_multiplier);
                    if (delayed_term != -1 && iq == 0)
                        szm =
                            (int)(min(szl - 1, szr) * eff_disjoint_multiplier) +
                            1;
                    for (int i = 0; i < szm; i++)
                        svds[iq].second[i] /= res_factor;
                    for (int i = 0; i < szm; i++)
                        if (svds[iq].second[i] > sqrt(cutoff))
                            s_kept++;
                        else
                            discarded_weights[ii] += svds[iq].second[i] *
                                                     svds[iq].second[i] *
                                                     res_factor * res_factor;
                    svds[iq].second.resize(s_kept);
                    s_kept_total += s_kept;
                }
            }
            if (max_bond_dim >= 1) {
                vector<pair<int, int>> idxs;
                for (auto &mq : q_map) {
                    int iq = mq.second;
                    for (size_t i = 0; i < svds[iq].second.size(); i++)
                        idxs.push_back(make_pair(iq, (int)i));
                }
                if (idxs.size() > max_bond_dim) {
                    s_kept_total = 0;
                    sort(idxs.begin(), idxs.end(),
                         [&svds](const pair<int, int> &a,
                                 const pair<int, int> &b) {
                             return svds[a.first].second[a.second] >
                                    svds[b.first].second[b.second];
                         });
                    for (size_t i = max_bond_dim; i < idxs.size(); i++) {
                        FP val = svds[idxs[i].first].second[idxs[i].second];
                        discarded_weights[ii] += val * val;
                        svds[idxs[i].first].second[idxs[i].second] = 0;
                    }
                    for (auto &mq : q_map) {
                        int s_kept = 0;
                        int iq = mq.second;
                        for (size_t i = 0; i < svds[iq].second.size(); i++)
                            svds[iq].second[s_kept] = svds[iq].second[i],
                            s_kept += svds[iq].second[i] != 0;
                        svds[iq].second.resize(s_kept);
                        s_kept_total += s_kept;
                    }
                }
            }
            if (constrain) {
                for (auto &mq : q_map) {
                    int s_kept = 0;
                    int iq = mq.second;
                    auto &nm = nms[iq];
                    auto &matvs = mats[iq];
                    if (pqx[iq] < 2)
                        continue;
                    int szl = (int)nm.first, szr = (int)nm.second;
                    int szm = (int)(min(szl, szr) * eff_disjoint_multiplier);
                    int rank = (int)svds[iq].second.size();
                    if ((szl == 1 && szr == 1) || rank == 0)
                        continue;
                    vector<FL> mat((size_t)szl * szr, 0);
                    if (delayed_term != -1 && iq == 0) {
                        szm =
                            (int)(min(szl - 1, szr) * eff_disjoint_multiplier) +
                            1;
                        for (auto &lrv : matvs)
                            if (lrv.first.first != 0)
                                mat[(size_t)(lrv.first.first - 1) * szr +
                                    lrv.first.second] += lrv.second;
                        szl--;
                        threading->activate_global_mkl();
                        IterativeMatrixFunctions<FL>::constrained_svd(
                            GMatrix<FL>(mat.data(), szl, szr), rank,
                            GMatrix<FL>(svds[iq].first[0].data() + 1 + szm, szl,
                                        szm),
                            GMatrix<FP>(svds[iq].second.data() + 1, 1,
                                        rank - 1),
                            GMatrix<FL>(svds[iq].first[1].data() + szr,
                                        rank - 1, szr),
                            csvd_sparsity, csvd_sparsity, csvd_max_iter,
                            csvd_max_iter, csvd_eps, csvd_eps, iprint >= 2);
                        threading->activate_normal();
                        szl++;
                    } else {
                        for (auto &lrv : matvs)
                            mat[(size_t)lrv.first.first * szr +
                                lrv.first.second] += lrv.second;
                        threading->activate_global_mkl();
                        IterativeMatrixFunctions<FL>::constrained_svd(
                            GMatrix<FL>(mat.data(), szl, szr), rank,
                            GMatrix<FL>(svds[iq].first[0].data(), szl, szm),
                            GMatrix<FP>(svds[iq].second.data(), 1, rank),
                            GMatrix<FL>(svds[iq].first[1].data(), rank, szr),
                            csvd_sparsity, csvd_sparsity, csvd_max_iter,
                            csvd_max_iter, csvd_eps, csvd_eps, iprint >= 2);
                        threading->activate_normal();
                    }
                    if (rescale)
                        for (int i = 0; i < rank; i++)
                            svds[iq].second[i] /= res_factor;
                }
            }
            FP accurate_svd_error = (FP)0.0;
            if (compute_accurate_svd_error &&
                (algo_type & MPOAlgorithmTypes::SVD)) {
                for (auto &mq : q_map) {
                    int iq = mq.second;
                    auto &nm = nms[iq];
                    auto &matvs = mats[iq];
                    int szl = (int)nm.first, szr = (int)nm.second,
                        szm = (int)(min(szl, szr) * eff_disjoint_multiplier);
                    if (delayed_term != -1 && iq == 0)
                        szm =
                            (int)(min(szl - 1, szr) * eff_disjoint_multiplier) +
                            1;
                    int s_kept = (int)svds[iq].second.size();
                    vector<FL> smat(
                        max((size_t)szl * szr, (size_t)s_kept * s_kept),
                        (FL)0.0);
                    vector<FL> stmp;
                    stmp.reserve((size_t)s_kept * szr);
                    for (int i = 0; i < s_kept; i++)
                        smat[(size_t)i * s_kept + i] =
                            svds[iq].second[i] * res_factor;
                    threading->activate_global_mkl();
                    if (s_kept > 0) {
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>(smat.data(), s_kept, s_kept), false,
                            GMatrix<FL>(svds[iq].first[1].data(), s_kept, szr),
                            false, GMatrix<FL>(stmp.data(), s_kept, szr),
                            (FL)1.0, (FL)0.0);
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>(svds[iq].first[0].data(), szl, szm),
                            false, GMatrix<FL>(stmp.data(), s_kept, szr), false,
                            GMatrix<FL>(smat.data(), szl, szr), (FL)1.0,
                            (FL)0.0);
                    }
                    for (auto &lrv : matvs)
                        smat[(size_t)lrv.first.first * szr +
                             lrv.first.second] -= lrv.second;
                    FP xnorm = GMatrixFunctions<FL>::norm(
                        GMatrix<FL>(smat.data(), szl, szr));
                    accurate_svd_error += xnorm * xnorm;
                    threading->activate_normal();
                }
            }
            if (iprint) {
                cout << "Mmpo = " << setw(5) << s_kept_total
                     << " DW = " << scientific << setw(8) << setprecision(2)
                     << discarded_weights[ii];
                if (compute_accurate_svd_error &&
                    (algo_type & MPOAlgorithmTypes::SVD))
                    cout << " Error = " << scientific << setw(8)
                         << setprecision(2) << sqrt(accurate_svd_error);
                cout.flush();
                bond_max = max(bond_max, s_kept_total);
                dw_max = max(dw_max, discarded_weights[ii]);
                error_total += accurate_svd_error;
            }
            // Part 3: construct mpo tensor
            shared_ptr<OperatorTensor<S, FL>> opt = tensors[ii];
            shared_ptr<Symbolic<S>> pmat;
            int lshape = (int)cur_terms.size();
            int rshape = s_kept_total;
            if (ii == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (ii == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            opt->lmat = opt->rmat = pmat;
            Symbolic<S> &mat = *pmat;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> site_ops;
            unordered_map<string, shared_ptr<OpElement<S, FL>>> site_op_names;
            for (auto &mq : q_map) {
                int iq = mq.second;
                for (auto &vls : map_ls[iq])
                    for (auto &vl : vls.second) {
                        int ip = vl.first.first, ix;
                        LL ic = vl.first.second, it;
                        if (ic >= (LL)cur_terms[ip].size()) {
                            ix = part_terms[ic + part_off].first;
                            it = part_terms[ic + part_off].second;
                        } else if (ic != -1) {
                            ix = cur_terms[ip][ic].first;
                            it = cur_terms[ip][ic].second;
                        } else {
                            ix = part_terms[delayed_term].first;
                            it = part_terms[delayed_term].second;
                        }
                        int ik = term_i[ix][it], k = term_k[ix][it];
                        // cout << "ix = " << ix << " expr = " << afd->exprs[ix]
                        //      << " ik = " << ik << " k = " << k
                        //      << " sub = " << sub_exprs[ix].at(make_pair(ik,
                        //      k))
                        //      << endl;
                        site_ops[sub_exprs[ix].at(make_pair(ik, k))] = nullptr;
                    }
            }
            site_ops[""] = nullptr;
            hamil->get_site_string_ops(ii, site_ops);
            site_op_names.reserve(site_ops.size());
            LL ixx = 0;
            for (auto &xm : site_ops) {
                if (xm.first.length() == 0)
                    site_op_names[xm.first] = make_shared<OpElement<S, FL>>(
                        OpNames::I, SiteIndex(),
                        xm.second->info->delta_quantum);
                else {
                    site_op_names[xm.first] = make_shared<OpElement<S, FL>>(
                        OpNames::X,
                        SiteIndex({(uint16_t)(ixx / 1000 / 1000),
                                   (uint16_t)(ixx / 1000 % 1000),
                                   (uint16_t)(ixx % 1000)},
                                  {}),
                        xm.second->info->delta_quantum);
                    ixx++;
                }
                if (xm.second->factor == (FL)0.0 || xm.second->info->n == 0 ||
                    xm.second->norm() < TINY)
                    site_op_names[xm.first] = nullptr;
                else
                    opt->ops[site_op_names.at(xm.first)] = xm.second;
            }
            int ppir = 0;
            for (int iq = 0; iq < (int)qs.size(); iq++) {
                S qq = qs[iq];
                auto &matvs = mats[iq];
                auto &mpl = map_ls[iq];
                auto &nm = nms[iq];
                int szl = (int)nm.first, szr = (int)nm.second, szm;
                if (algo_type & MPOAlgorithmTypes::NC)
                    szm = ii == n_sites - 1 ? szr : szl;
                else if (algo_type & MPOAlgorithmTypes::CN)
                    szm = ii == 0 ? szl : szr;
                else if (algo_type & MPOAlgorithmTypes::Bipartite)
                    szm = (int)mvcs[iq][0].size() + (int)mvcs[iq][1].size();
                else { // SVD
                    szm = (int)(min(szl, szr) * eff_disjoint_multiplier);
                    if (delayed_term != -1 && iq == 0)
                        szm =
                            (int)(min(szl - 1, szr) * eff_disjoint_multiplier) +
                            1;
                }
                vector<shared_ptr<OpElement<S, FL>>> site_mp(szl);
                for (auto &vls : mpl)
                    for (auto &vl : vls.second) {
                        int ip = vl.first.first, ix;
                        LL ic = vl.first.second, it;
                        if (ic >= (LL)cur_terms[ip].size()) {
                            ix = part_terms[ic + part_off].first;
                            it = part_terms[ic + part_off].second;
                        } else if (ic != -1) {
                            ix = cur_terms[ip][ic].first;
                            it = cur_terms[ip][ic].second;
                        } else {
                            ix = part_terms[delayed_term].first;
                            it = part_terms[delayed_term].second;
                        }
                        int il = vl.second;
                        int ik = term_i[ix][it], k = term_k[ix][it];
                        // if (site_op_names.at(
                        //         sub_exprs[ix].at(make_pair(ik, k))) !=
                        //         nullptr)
                        // cout << "iq = " << iq << " il = " << il
                        //      << " ix = " << ix << " it = " << it
                        //      << " ik = " << ik << " k = " << k
                        //      << " ql = " << qq.get_bra(qh)
                        //      << " qr = " << -qq.get_ket() << " sub = "
                        //      << sub_exprs[ix].at(make_pair(ik, k)) << " "
                        //      << site_op_names
                        //             .at(sub_exprs[ix].at(make_pair(ik, k)))
                        //             ->q_label
                        //      << endl;
                        site_mp[il] = site_op_names.at(
                            sub_exprs[ix].at(make_pair(ik, k)));
                    }
                if (algo_type & MPOAlgorithmTypes::Bipartite) {
                    vector<int> lip(szl), lix(szl, -1), rix(szr, -1);
                    int ixln = (int)mvcs[iq][0].size(),
                        ixrn = (int)mvcs[iq][1].size();
                    int szm = ixln + ixrn;
                    for (auto &vls : mpl)
                        for (auto &vl : vls.second) {
                            int il = vl.second, ip = vl.first.first;
                            lip[il] = ip;
                        }
                    for (int ixl = 0; ixl < ixln; ixl++)
                        lix[mvcs[iq][0][ixl]] = ixl;
                    for (int ixr = 0; ixr < ixrn; ixr++)
                        rix[mvcs[iq][1][ixr]] = ixr + ixln;
                    vector<vector<shared_ptr<OpExpr<S>>>> tterms(
                        cur_terms.size() * szm);
                    for (auto &lrv : matvs) {
                        int il = lrv.first.first, ir = lrv.first.second, irx;
                        FL factor = 1;
                        if (lix[il] == -2)
                            continue;
                        else if (lix[il] != -1)
                            irx = lix[il], lix[il] = -2;
                        else
                            irx = rix[ir], factor = lrv.second;
                        int ip = lip[il];
                        if (abs(factor) > cutoff && site_mp[il] != nullptr)
                            tterms[ip * szm + irx].push_back(
                                site_mp[il]->scalar_multiply(factor));
                    }
                    for (LL vix = 0; vix < (int)tterms.size(); vix++)
                        if (tterms[vix].size() != 0)
                            mat[{(int)(vix / szm), ppir + (int)(vix % szm)}] =
                                sum(tterms[vix]);
                } else {
                    int rszm = (int)svds[iq].second.size();
                    for (int ir = 0; ir < rszm; ir++) {
                        vector<vector<shared_ptr<OpExpr<S>>>> tterms(
                            cur_terms.size());
                        for (auto &vls : mpl)
                            for (auto &vl : vls.second) {
                                int il = vl.second, ip = vl.first.first;
                                assert(
                                    site_mp[il] == nullptr ||
                                    left_q[ip].get_bra(qh).combine(
                                        qq.get_bra(qh), site_mp[il]->q_label) !=
                                        S(S::invalid));
                                FL factor =
                                    svds[iq].first[0][(size_t)il * szm + ir] *
                                    res_factor;
                                if (ii == n_sites - 1)
                                    factor *= svds[iq].second[ir];
                                // cout << "iq = " << iq << " q = " << q
                                //      << " ip = " << ip << " l = " <<
                                //      left_q[ip]
                                //      << " m = " << site_mp[il]->q_label << "
                                //      factor = " << factor << endl;
                                if (abs(factor) > cutoff &&
                                    site_mp[il] != nullptr)
                                    tterms[ip].push_back(
                                        site_mp[il]->scalar_multiply(factor));
                            }
                        for (int vip = 0; vip < (int)tterms.size(); vip++)
                            if (tterms[vip].size() != 0) {
                                // cout << "mat idx l = " << vip << " "
                                //      << " r = " << ppir + ir
                                //      << " idx = " << mat.data.size() << endl;
                                mat[{vip, ppir + ir}] = sum(tterms[vip]);
                            }
                    }
                    szm = rszm;
                }
                ppir += szm;
            }
            assert(ppir == s_kept_total);
            if (iprint) {
                cout << " NNZ = " << setw(8) << mat.nnz() << " SPT = " << fixed
                     << setprecision(4) << setw(6)
                     << (double)(mat.size() - mat.nnz()) / mat.size();
                cout.flush();
                nnz_total += mat.nnz();
                size_total += mat.size();
            }
            // Part 4: evaluate sum expressions
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            for (size_t i = 0; i < mat.data.size(); i++) {
                // only happens for non-sparse boundary tensors
                if (mat.data[i]->get_type() == OpTypes::Zero)
                    continue;
                shared_ptr<OpSum<S, FL>> opx =
                    dynamic_pointer_cast<OpSum<S, FL>>(mat.data[i]);
                assert(opx->strings.size() != 0);
                shared_ptr<SparseMatrix<S, FL>> xmat =
                    opt->ops.at(opx->strings[0]->get_op());
                if (opx->strings.size() == 1) {
                    if (ii == 0 || ii == n_sites - 1) {
                        shared_ptr<OpElement<S, FL>> opel =
                            make_shared<OpElement<S, FL>>(
                                ii == 0 ? OpNames::XL : OpNames::XR,
                                SiteIndex({(uint16_t)(i / 1000),
                                           (uint16_t)(i % 1000)},
                                          {}),
                                xmat->info->delta_quantum);
                        mat.data[i] = opel;
                        assert(opx->strings[0]->get_op()->q_label ==
                               opel->q_label);
                        if (opx->strings[0]->factor != (FL)1.0) {
                            shared_ptr<SparseMatrix<S, FL>> gmat =
                                make_shared<SparseMatrix<S, FL>>(nullptr);
                            gmat->allocate(xmat->info, xmat->data);
                            gmat->factor =
                                xmat->factor * opx->strings[0]->factor;
                            opt->ops[opel] = gmat;
                        } else
                            opt->ops[opel] = xmat;
                    } else
                        mat.data[i] =
                            opx->strings[0]->get_op()->scalar_multiply(
                                (FL)opx->strings[0]->factor);
                } else {
                    // for SU2 there will be multiple possible gmats
                    // with different dq
                    map<S, shared_ptr<SparseMatrix<S, FL>>> gmats;
                    bool all_same_dq = true;
                    for (auto &x : opx->strings) {
                        shared_ptr<SparseMatrixInfo<S>> info =
                            opt->ops.at(x->get_op())->info;
                        if (!gmats.count(info->delta_quantum)) {
                            shared_ptr<SparseMatrix<S, FL>> gmat =
                                make_shared<SparseMatrix<S, FL>>(d_alloc);
                            gmat->allocate(info);
                            gmats[info->delta_quantum] = gmat;
                        }
                    }
                    for (auto &x : opx->strings) {
                        shared_ptr<SparseMatrix<S, FL>> mmat =
                            opt->ops.at(x->get_op());
                        shared_ptr<SparseMatrix<S, FL>> gmat =
                            gmats.at(mmat->info->delta_quantum);
                        hamil->opf->iadd(gmat, mmat, x->factor);
                        if (hamil->opf->seq->mode != SeqTypes::None)
                            hamil->opf->seq->simple_perform();
                    }
                    if (gmats.size() == 1) {
                        shared_ptr<SparseMatrix<S, FL>> gmat =
                            gmats.begin()->second;
                        shared_ptr<OpElement<S, FL>> opel;
                        if (ii == 0 || ii == n_sites - 1)
                            opel = make_shared<OpElement<S, FL>>(
                                ii == 0 ? OpNames::XL : OpNames::XR,
                                SiteIndex({(uint16_t)(i / 1000),
                                           (uint16_t)(i % 1000)},
                                          {}),
                                gmat->info->delta_quantum);
                        else {
                            opel = make_shared<OpElement<S, FL>>(
                                OpNames::X,
                                SiteIndex({(uint16_t)(ixx / 1000 / 1000),
                                           (uint16_t)(ixx / 1000 % 1000),
                                           (uint16_t)(ixx % 1000)},
                                          {}),
                                gmat->info->delta_quantum);
                            ixx++;
                        }
                        mat.data[i] = opel;
                        opt->ops[opel] = gmat;
                        assert(opx->strings[0]->get_op()->q_label ==
                               opel->q_label);
                    } else {
                        // for non-singlet Hamiltonian:
                        // ii != 0 && ii != n_sites - 1 may not be satisfied
                        // in fact for non singlet mps this is already supported
                        // since with non-zero left_vac, the left_assign will be
                        // replaced by left_contract which supports arbitrary
                        // expressions
                        vector<shared_ptr<OpExpr<S>>> opels;
                        opels.reserve(gmats.size());
                        for (auto &gmat : gmats) {
                            shared_ptr<OpElement<S, FL>> opel =
                                make_shared<OpElement<S, FL>>(
                                    OpNames::X,
                                    SiteIndex({(uint16_t)(ixx / 1000 / 1000),
                                               (uint16_t)(ixx / 1000 % 1000),
                                               (uint16_t)(ixx % 1000)},
                                              {}),
                                    gmat.second->info->delta_quantum);
                            ixx++;
                            opels.push_back(opel);
                            opt->ops[opel] = gmat.second;
                        }
                        mat.data[i] = sum(opels);
                    }
                }
            }
            assert(ixx < 1000 * 1000 * 1000);
            // Part 5: left and right operator names
            shared_ptr<SymbolicRowVector<S>> plop;
            shared_ptr<SymbolicColumnVector<S>> prop;
            if (ii == n_sites - 1)
                plop = make_shared<SymbolicRowVector<S>>(1);
            else
                plop = make_shared<SymbolicRowVector<S>>(rshape);
            if (ii == 0)
                prop = make_shared<SymbolicColumnVector<S>>(1);
            else
                prop = make_shared<SymbolicColumnVector<S>>(lshape);
            left_operator_names[ii] = plop;
            right_operator_names[ii] = prop;
            SymbolicRowVector<S> &lop = *plop;
            SymbolicColumnVector<S> &rop = *prop;
            for (int iop = 0; iop < rop.m; iop++)
                rop[iop] = make_shared<OpElement<S, FL>>(
                    OpNames::XR,
                    SiteIndex({(uint16_t)(iop / 1000), (uint16_t)(iop % 1000)},
                              {}),
                    -left_q[iop].get_ket());
            // Part 6: prepare for next
            vector<vector<FL>> new_cur_values(s_kept_total);
            vector<vector<pair<int, LL>>> new_cur_terms(s_kept_total);
            int isk = 0;
            left_q.resize(s_kept_total);
            sparse_ranges.clear();
            for (int iq = 0; iq < (int)qs.size(); iq++) {
                S qq = qs[iq];
                auto &mpr = map_rs[iq];
                auto &nm = nms[iq];
                int szr = (int)nm.second, szl = (int)nm.first;
                vector<pair<int, LL>> vct(szr);
                for (auto &vrs : mpr)
                    for (auto &vr : vrs.second) {
                        int vip = vr.first.first, vix;
                        LL vic = vr.first.second, vit;
                        if (vic >= (LL)cur_terms[vip].size()) {
                            vix = part_terms[vic + part_off].first;
                            vit = part_terms[vic + part_off].second;
                            vct[vr.second] = part_terms[vic + part_off];
                        } else if (vic != -1) {
                            vix = cur_terms[vip][vic].first;
                            vit = cur_terms[vip][vic].second;
                            vct[vr.second] = cur_terms[vip][vic];
                        } else {
                            vix = part_terms[delayed_term].first;
                            vit = part_terms[delayed_term].second;
                            vct[vr.second] = part_terms[delayed_term];
                        }
                        term_i[vix][vit] = term_k[vix][vit];
                    }
                int rszm;
                if (algo_type & MPOAlgorithmTypes::Bipartite) {
                    auto &matvs = mats[iq];
                    vector<int> lix(szl, -1), rix(szr, -1);
                    int ixln = (int)mvcs[iq][0].size(),
                        ixrn = (int)mvcs[iq][1].size();
                    rszm = ixln + ixrn;
                    for (int ixl = 0; ixl < ixln; ixl++)
                        lix[mvcs[iq][0][ixl]] = ixl;
                    // add right vertices in MVC
                    for (int ixr = 0; ixr < ixrn; ixr++) {
                        int ir = mvcs[iq][1][ixr];
                        rix[ir] = ixr + ixln;
                        new_cur_terms[rix[ir] + isk].push_back(vct[ir]);
                        new_cur_values[rix[ir] + isk].push_back((FL)1.0);
                    }
                    for (int ir = 0; ir < rszm; ir++)
                        left_q[ir + isk] = qq;
                    // add edges with right vertex not in MVC
                    // and edges with both left and right vertices in MVC
                    for (auto &lrv : matvs) {
                        int il = lrv.first.first, ir = lrv.first.second;
                        if (rix[ir] != -1 && lix[il] == -1)
                            continue;
                        assert(lix[il] != -1);
                        if (iq == 0 && delayed_term != -1 &&
                            vct[ir] == part_terms[delayed_term])
                            continue;
                        new_cur_terms[lix[il] + isk].push_back(vct[ir]);
                        new_cur_values[lix[il] + isk].push_back(lrv.second);
                    }
                } else {
                    rszm = (int)svds[iq].second.size();
                    if (iq == 0 && delayed_term != -1)
                        assert(rszm != 0);
                    bool has_pf = false;
                    FL pf_factor = 0;
                    for (int j = 0; j < rszm; j++) {
                        left_q[j + isk] = qq;
                        for (int ir = 0; ir < szr; ir++) {
                            // singular values multiplies to right
                            FL val =
                                ii == n_sites - 1
                                    ? svds[iq].first[1][(size_t)j * szr + ir]
                                    : svds[iq].first[1][(size_t)j * szr + ir] *
                                          svds[iq].second[j];
                            if (iq == 0 && delayed_term != -1 &&
                                vct[ir] == part_terms[delayed_term]) {
                                pf_factor += val;
                                has_pf = true;
                                continue;
                            }
                            if (abs(svds[iq].first[1][(size_t)j * szr + ir]) <
                                cutoff)
                                continue;
                            new_cur_terms[j + isk].push_back(vct[ir]);
                            new_cur_values[j + isk].push_back(val);
                        }
                    }
                    if (has_pf)
                        rsc_factor = pf_factor / part_values[delayed_term];
                }
                if (pqx[iq] >= 2) {
                    if (sparse_ranges.size() <= pqx[iq] - 2)
                        sparse_ranges.resize(pqx[iq] - 1);
                    sparse_ranges[pqx[iq] - 2].push_back(isk);
                    sparse_ranges[pqx[iq] - 2].push_back(isk + rszm);
                }
                isk += rszm;
            }
            for (int iop = 0; iop < lop.n; iop++)
                lop[iop] = make_shared<OpElement<S, FL>>(
                    OpNames::XL,
                    SiteIndex({(uint16_t)(iop / 1000), (uint16_t)(iop % 1000)},
                              {}),
                    left_q[iop].get_bra(qh));
            assert(isk == s_kept_total);
            cur_terms = new_cur_terms;
            cur_values = new_cur_values;
            if (cur_terms.size() == 0) {
                cur_terms.emplace_back();
                cur_values.emplace_back();
            }
            // Part 7: sanity check
            for (auto &op : opt->ops)
                assert((dynamic_pointer_cast<OpElement<S, FL>>(op.first)
                            ->q_label) == op.second->info->delta_quantum);
            if (ii == 0) {
                for (int iop = 0; iop < lop.n; iop++)
                    // singlet embedding
                    if (mat.data[iop]->get_type() != OpTypes::Zero) {
                        S sll = dynamic_pointer_cast<OpElement<S, FL>>(lop[iop])
                                    ->q_label;
                        if (mat.data[iop]->get_type() == OpTypes::Elem) {
                            S sl = dynamic_pointer_cast<OpElement<S, FL>>(
                                       mat.data[iop])
                                       ->q_label;
                            assert(sl.combine(sll, left_vacuum) !=
                                   S(S::invalid));
                        } else if (mat.data[iop]->get_type() == OpTypes::Sum) {
                            for (auto &x : dynamic_pointer_cast<OpSum<S, FL>>(
                                               mat.data[iop])
                                               ->strings) {
                                S sl = x->get_op()->q_label;
                                assert(sl.combine(sll, left_vacuum) !=
                                       S(S::invalid));
                            }
                        }
                    }
            } else if (ii == n_sites - 1) {
                for (int iop = 0; iop < rop.m; iop++)
                    if (mat.data[iop]->get_type() != OpTypes::Zero)
                        assert((dynamic_pointer_cast<OpElement<S, FL>>(rop[iop])
                                    ->q_label ==
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    mat.data[iop])
                                    ->q_label));
            } else {
                SymbolicRowVector<S> &llop =
                    *dynamic_pointer_cast<SymbolicRowVector<S>>(
                        left_operator_names[ii - 1]);
                auto gmat = dynamic_pointer_cast<SymbolicMatrix<S>>(pmat);
                for (size_t ig = 0; ig < gmat->data.size(); ig++) {
                    S sl = dynamic_pointer_cast<OpElement<S, FL>>(
                               llop[gmat->indices[ig].first])
                               ->q_label;
                    S sr = dynamic_pointer_cast<OpElement<S, FL>>(
                               lop[gmat->indices[ig].second])
                               ->q_label;
                    if (mat.data[ig]->get_type() == OpTypes::Elem) {
                        S sm =
                            dynamic_pointer_cast<OpElement<S, FL>>(mat.data[ig])
                                ->q_label;
                        assert(sl.combine(sr, sm) != S(S::invalid));
                    } else if (mat.data[ig]->get_type() == OpTypes::Sum) {
                        for (auto &x :
                             dynamic_pointer_cast<OpSum<S, FL>>(mat.data[ig])
                                 ->strings)
                            assert(sl.combine(sr, x->get_op()->q_label) !=
                                   S(S::invalid));
                    } else
                        assert(false);
                }
            }
            if (iprint) {
                tsite = _t.get_time();
                if (algo_type & MPOAlgorithmTypes::SVD)
                    cout << fixed << setprecision(3) << " Tsvd = " << tsvd;
                else if (algo_type & MPOAlgorithmTypes::Bipartite)
                    cout << fixed << setprecision(3) << " Tmvc = " << tsvd;
                cout << " T = " << tsite << endl;
                tsite_total += tsite;
                tsvd_total += tsvd;
            }
            this->save_tensor(ii);
            this->unload_tensor(ii);
        }
        if (n_terms != 0) {
            // end of loop; check last term is identity with cur_values = 1
            assert(cur_values.size() == 1 && cur_values[0].size() == 1);
            assert(cur_values[0][0] == (FL)1.0);
            int ix = cur_terms[0][0].first;
            LL it = cur_terms[0][0].second;
            assert(it != -1);
            assert(term_i[ix][it] == term_l[ix]);
        }
        if (iprint) {
            cout << "Ttotal = " << fixed << setprecision(3) << setw(10)
                 << tsite_total << fixed << setprecision(3);
            if (algo_type & MPOAlgorithmTypes::SVD)
                cout << " Tsvd-total = " << tsvd_total;
            else if (algo_type & MPOAlgorithmTypes::Bipartite)
                cout << " Tmvc-total = " << tsvd_total;
            cout << " MPO bond dimension = " << setw(5) << bond_max;
            cout << " MaxDW = " << scientific << setw(8) << setprecision(2)
                 << dw_max;
            if (compute_accurate_svd_error &&
                (algo_type & MPOAlgorithmTypes::SVD))
                cout << " Total error = " << scientific << setw(8)
                     << setprecision(2) << sqrt(error_total);
            cout << endl;
            cout << "NNZ = " << setw(12) << nnz_total;
            cout << " SIZE = " << setw(12) << size_total;
            cout << " SPT = " << fixed << setprecision(4) << setw(6)
                 << (double)(size_total - nnz_total) / size_total << endl
                 << endl;
        }
        for (int i = 0; i < (int)left_operator_names.size(); i++) {
            this->save_left_operators(i);
            this->unload_left_operators(i);
        }
        for (int i = 0; i < (int)right_operator_names.size(); i++) {
            this->save_right_operators(i);
            this->unload_right_operators(i);
        }
    }
    virtual ~GeneralMPO() = default;
};

} // namespace block2