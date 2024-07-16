
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2023 Huanchen Zhai <hczhai@caltech.edu>
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

/** Integral for Hamiltonian with arbitrary expressions. */

#pragma once

#include "flow.hpp"
#include "general_symm_permutation.hpp"
#include "integral.hpp"
#include "spin_permutation.hpp"
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

enum struct ElemOpTypes : uint8_t { SU2, SZ, SGF, SGB, SAny };

template <typename FL> struct GeneralFCIDUMP {
    typedef decltype(abs((FL)0.0)) FP;
    map<string, string> params;
    typename const_fl_type<FL>::FL const_e =
        (typename const_fl_type<FL>::FL)0.0;
    vector<string> exprs;
    vector<vector<uint16_t>> indices;
    vector<vector<FL>> data;
    ElemOpTypes elem_type;
    bool order_adjusted = false;
    GeneralFCIDUMP() : elem_type(ElemOpTypes::SU2) {}
    GeneralFCIDUMP(ElemOpTypes elem_type) : elem_type(elem_type) {}
    virtual ~GeneralFCIDUMP() = default;
    static shared_ptr<GeneralFCIDUMP>
    initialize_from_qc(const shared_ptr<FCIDUMP<FL>> &fcidump,
                       ElemOpTypes elem_type, FP cutoff = (FP)0.0) {
        shared_ptr<GeneralFCIDUMP> r = make_shared<GeneralFCIDUMP>();
        r->params = fcidump->params;
        r->const_e = fcidump->e();
        r->elem_type = elem_type;
        uint16_t n = fcidump->n_sites();
        if (elem_type == ElemOpTypes::SU2) {
            r->exprs.push_back("((C+(C+D)0)1+D)0");
            r->indices.push_back(vector<uint16_t>());
            r->data.push_back(vector<FL>());
            auto *idx = &r->indices.back();
            auto *dt = &r->data.back();
            array<uint16_t, 4> arr;
            for (arr[0] = 0; arr[0] < n; arr[0]++)
                for (arr[1] = 0; arr[1] < n; arr[1]++)
                    for (arr[2] = 0; arr[2] < n; arr[2]++)
                        for (arr[3] = 0; arr[3] < n; arr[3]++)
                            if (abs(fcidump->v(arr[0], arr[3], arr[1],
                                               arr[2])) > cutoff) {
                                idx->insert(idx->end(), arr.begin(), arr.end());
                                dt->push_back(
                                    fcidump->v(arr[0], arr[3], arr[1], arr[2]));
                            }
            r->exprs.push_back("(C+D)0");
            r->indices.push_back(vector<uint16_t>());
            r->data.push_back(vector<FL>());
            idx = &r->indices.back(), dt = &r->data.back();
            for (arr[0] = 0; arr[0] < n; arr[0]++)
                for (arr[1] = 0; arr[1] < n; arr[1]++)
                    if (abs(fcidump->t(arr[0], arr[1])) > cutoff) {
                        idx->insert(idx->end(), arr.begin(), arr.begin() + 2);
                        dt->push_back((FL)sqrt(2) * fcidump->t(arr[0], arr[1]));
                    }
        } else if (elem_type == ElemOpTypes::SZ) {
            r->exprs.push_back("ccdd");
            r->exprs.push_back("cCDd");
            r->exprs.push_back("CcdD");
            r->exprs.push_back("CCDD");
            for (uint8_t si = 0; si < 2; si++)
                for (uint8_t sj = 0; sj < 2; sj++) {
                    r->indices.push_back(vector<uint16_t>());
                    r->data.push_back(vector<FL>());
                    auto *idx = &r->indices.back();
                    auto *dt = &r->data.back();
                    array<uint16_t, 4> arr;
                    for (arr[0] = 0; arr[0] < n; arr[0]++)
                        for (arr[1] = 0; arr[1] < n; arr[1]++)
                            for (arr[2] = 0; arr[2] < n; arr[2]++)
                                for (arr[3] = 0; arr[3] < n; arr[3]++)
                                    if (abs(fcidump->v(si, sj, arr[0], arr[3],
                                                       arr[1], arr[2])) >
                                        cutoff) {
                                        idx->insert(idx->end(), arr.begin(),
                                                    arr.end());
                                        dt->push_back((FL)0.5 *
                                                      fcidump->v(si, sj, arr[0],
                                                                 arr[3], arr[1],
                                                                 arr[2]));
                                    }
                }
            r->exprs.push_back("cd");
            r->exprs.push_back("CD");
            for (uint8_t si = 0; si < 2; si++) {
                r->indices.push_back(vector<uint16_t>());
                r->data.push_back(vector<FL>());
                auto *idx = &r->indices.back();
                auto *dt = &r->data.back();
                array<uint16_t, 2> arr;
                for (arr[0] = 0; arr[0] < n; arr[0]++)
                    for (arr[1] = 0; arr[1] < n; arr[1]++)
                        if (abs(fcidump->t(si, arr[0], arr[1])) > cutoff) {
                            idx->insert(idx->end(), arr.begin(),
                                        arr.begin() + 2);
                            dt->push_back(fcidump->t(si, arr[0], arr[1]));
                        }
            }
        } else {
            r->exprs.push_back("CCDD");
            r->indices.push_back(vector<uint16_t>());
            r->data.push_back(vector<FL>());
            auto *idx = &r->indices.back();
            auto *dt = &r->data.back();
            array<uint16_t, 4> arr;
            for (arr[0] = 0; arr[0] < n; arr[0]++)
                for (arr[1] = 0; arr[1] < n; arr[1]++)
                    for (arr[2] = 0; arr[2] < n; arr[2]++)
                        for (arr[3] = 0; arr[3] < n; arr[3]++)
                            if (abs(fcidump->v(arr[0], arr[3], arr[1],
                                               arr[2])) > cutoff) {
                                idx->insert(idx->end(), arr.begin(), arr.end());
                                dt->push_back(
                                    (FL)0.5 *
                                    fcidump->v(arr[0], arr[3], arr[1], arr[2]));
                            }
            r->exprs.push_back("CD");
            r->indices.push_back(vector<uint16_t>());
            r->data.push_back(vector<FL>());
            idx = &r->indices.back(), dt = &r->data.back();
            for (arr[0] = 0; arr[0] < n; arr[0]++)
                for (arr[1] = 0; arr[1] < n; arr[1]++)
                    if (abs(fcidump->t(arr[0], arr[1])) > cutoff) {
                        idx->insert(idx->end(), arr.begin(), arr.begin() + 2);
                        dt->push_back(fcidump->t(arr[0], arr[1]));
                    }
        }
        return r;
    }
    void add_eight_fold_term(const FL *vals, size_t len, FP cutoff = (FP)0.0,
                             FL factor = (FL)1.0) {
        size_t n = 0, m = 0;
        for (n = 1; n < len; n++) {
            m = n * (n + 1) >> 1;
            if ((m * (m + 1) >> 1) >= len)
                break;
        }
        assert((m * (m + 1) >> 1) == len && m >= n);
        vector<size_t> xm(m + 1, 0), gm(m + 1, 2), pm(m + 1, 0), qm(m + 1, 0);
        for (size_t im = 1; im <= m; im++) {
            xm[im] = xm[im - 1] + im;
            if (xm[im] - 1 <= m) {
                gm[xm[im] - 1] = 1;
                for (size_t jm = xm[im - 1]; jm < xm[im]; jm++)
                    pm[jm] = im - 1, qm[jm] = jm - xm[im - 1];
            }
        }
        assert(xm[m] == len);
        int ntg = threading->activate_global();
        vector<size_t> ms(ntg + 1, 0);
        const size_t plm = m / ntg + !!(m % ntg);
#pragma omp parallel num_threads(ntg)
        {
            int tid = threading->get_thread_id();
            for (size_t im = plm * tid; im < min(m, plm * (tid + 1)); im++)
                for (size_t jm = 0; jm <= im; jm++)
                    ms[tid] += (abs(factor * vals[xm[im] + jm]) > cutoff) *
                               gm[im] * gm[jm] * (2 - (im == jm));
        }
        ms[ntg] = accumulate(&ms[0], &ms[ntg], (size_t)0);
        indices.push_back(vector<uint16_t>(ms[ntg] * 4));
        data.push_back(vector<FL>(ms[ntg]));
#pragma omp parallel num_threads(ntg)
        {
            int tid = threading->get_thread_id();
            size_t istart = 0;
            for (int i = 0; i < tid; i++)
                istart += ms[i];
            for (size_t im = plm * tid; im < min(m, plm * (tid + 1)); im++)
                for (size_t jm = 0; jm <= im; jm++)
                    if (abs(factor * vals[xm[im] + jm]) > cutoff) {
                        for (size_t xxm = 0, xim = im, xjm = jm,
                                    xs = istart * 4;
                             xxm < (2 - (im == jm));
                             xxm++, xim = jm, xjm = im) {
                            indices.back()[xs + 0] = (uint16_t)pm[xim];
                            indices.back()[xs + 1] = (uint16_t)pm[xjm];
                            indices.back()[xs + 2] = (uint16_t)qm[xjm];
                            indices.back()[xs + 3] = (uint16_t)qm[xim];
                            data.back()[istart] = factor * vals[xm[im] + jm];
                            istart++, xs += 4;
                            if (gm[xim] == 2) {
                                indices.back()[xs + 0] = (uint16_t)qm[xim];
                                indices.back()[xs + 1] = (uint16_t)pm[xjm];
                                indices.back()[xs + 2] = (uint16_t)qm[xjm];
                                indices.back()[xs + 3] = (uint16_t)pm[xim];
                                data.back()[istart] =
                                    factor * vals[xm[im] + jm];
                                istart++, xs += 4;
                            }
                            if (gm[xjm] == 2) {
                                indices.back()[xs + 0] = (uint16_t)pm[xim];
                                indices.back()[xs + 1] = (uint16_t)qm[xjm];
                                indices.back()[xs + 2] = (uint16_t)pm[xjm];
                                indices.back()[xs + 3] = (uint16_t)qm[xim];
                                data.back()[istart] =
                                    factor * vals[xm[im] + jm];
                                istart++, xs += 4;
                            }
                            if (gm[xim] == 2 && gm[xjm] == 2) {
                                indices.back()[xs + 0] = (uint16_t)qm[xim];
                                indices.back()[xs + 1] = (uint16_t)qm[xjm];
                                indices.back()[xs + 2] = (uint16_t)pm[xjm];
                                indices.back()[xs + 3] = (uint16_t)pm[xim];
                                data.back()[istart] =
                                    factor * vals[xm[im] + jm];
                                istart++, xs += 4;
                            }
                        }
                    }
            for (int i = 0; i < tid + 1; i++)
                istart -= ms[i];
            assert(istart == 0);
        }
        threading->activate_normal();
    }
    // array must have the min strides == 1
    void add_sum_term(const FL *vals, size_t len, const vector<int> &shape,
                      const vector<size_t> &strides, FP cutoff = (FP)0.0,
                      FL factor = (FL)1.0,
                      const vector<int> &orb_sym = vector<int>(),
                      vector<uint16_t> rperm = vector<uint16_t>(),
                      int target_irrep = 0) {
        int ntg = threading->activate_global();
        vector<size_t> lens(ntg + 1, 0);
        const size_t plen = len / ntg + !!(len % ntg);
        if (rperm.size() == 0)
            for (size_t i = 0; i < shape.size(); i++)
                rperm.push_back((uint16_t)i);
#pragma omp parallel num_threads(ntg)
        {
            int tid = threading->get_thread_id();
            for (size_t i = plen * tid; i < min(len, plen * (tid + 1)); i++)
                lens[tid] += (abs(factor * vals[i]) > cutoff);
        }
        lens[ntg] = accumulate(&lens[0], &lens[ntg], (size_t)0);
        indices.push_back(vector<uint16_t>(lens[ntg] * shape.size()));
        data.push_back(vector<FL>(lens[ntg]));
#pragma omp parallel num_threads(ntg)
        {
            int tid = threading->get_thread_id();
            size_t istart = 0;
            for (int i = 0; i < tid; i++)
                istart += lens[i];
            if (orb_sym.size() == 0) {
                for (size_t i = plen * tid; i < min(len, plen * (tid + 1)); i++)
                    if (abs(factor * vals[i]) > cutoff) {
                        for (int j = 0; j < (int)shape.size(); j++)
                            indices.back()[istart * shape.size() + rperm[j]] =
                                (uint16_t)(i / strides[j] % shape[j]);
                        data.back()[istart] = factor * vals[i];
                        istart++;
                    }
            } else {
                for (size_t i = plen * tid; i < min(len, plen * (tid + 1)); i++)
                    if (abs(factor * vals[i]) > cutoff) {
                        int irrep = target_irrep;
                        for (int j = 0; j < (int)shape.size(); j++) {
                            indices.back()[istart * shape.size() + rperm[j]] =
                                (uint16_t)(i / strides[j] % shape[j]);
                            irrep ^= orb_sym[i / strides[j] % shape[j]];
                        }
                        data.back()[istart] = factor * vals[i] * (FL)(!irrep);
                        istart++;
                    }
            }
            for (int i = 0; i < tid + 1; i++)
                istart -= lens[i];
            assert(istart == 0);
        }
        threading->activate_normal();
    }
    struct vector_uint16_hasher {
        size_t operator()(const vector<uint16_t> &x) const {
            size_t r = x.size();
            for (auto &i : x)
                r ^= i + 0x9e3779b9 + (r << 6) + (r >> 2);
            return r;
        }
    };
    // abelian symmetry case
    shared_ptr<GeneralFCIDUMP> adjust_order(const string &fermionic_ops,
                                            bool merge = true,
                                            FP cutoff = (FP)0.0) const {
        unordered_map<string, uint64_t> r_exprs;
        unordered_map<char, int8_t> is_op_f;
        for (size_t it = 0; it < fermionic_ops.length(); it++)
            is_op_f[fermionic_ops[it]] = 1;
        vector<vector<uint16_t>> r_indices;
        vector<vector<FL>> r_data;
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            int nn = (int)exprs[ix].length();
            vector<uint16_t> idx_idx(nn);
            for (size_t i = 0; i < (nn == 0 ? 1 : indices[ix].size());
                 i += (nn == 0 ? 1 : nn)) {
                for (int j = 0; j < nn; j++)
                    idx_idx[j] = j;
                string xex = exprs[ix];
                int8_t n = 0;
                for (int xi = 0; xi < (int)nn - 1; xi++)
                    for (int xj = xi; xj >= 0; xj--)
                        if (indices[ix][i + idx_idx[xj]] >
                            indices[ix][i + idx_idx[xj + 1]]) {
                            swap(idx_idx[xj], idx_idx[xj + 1]);
                            swap(xex[xj], xex[xj + 1]);
                            n ^= (is_op_f.count(xex[xj]) &&
                                  is_op_f.count(xex[xj + 1]));
                        }
                if (!r_exprs.count(xex)) {
                    r_exprs[xex] = r_data.size();
                    r_indices.push_back(vector<uint16_t>());
                    r_data.push_back(vector<FL>());
                }
                uint64_t ir = r_exprs.at(xex);
                for (int j = 0; j < nn; j++)
                    r_indices[ir].push_back(indices[ix][i + idx_idx[j]]);
                r_data[ir].push_back((FL)(FP)(1 - (n << 1)) *
                                     data[ix][nn == 0 ? i : i / nn]);
            }
        }
        shared_ptr<GeneralFCIDUMP> r = make_shared<GeneralFCIDUMP>();
        r->params = params;
        r->const_e = const_e;
        r->elem_type = elem_type;
        r->exprs.resize(r_exprs.size());
        for (auto &x : r_exprs)
            r->exprs[x.second] = x.first;
        r->indices = r_indices;
        r->data = r_data;
        if (merge)
            r->merge_terms(cutoff);
        return r;
    }
    template <typename T, typename FLX>
    shared_ptr<GeneralFCIDUMP>
    adjust_order_impl(const vector<shared_ptr<T>> &schemes, bool merge = true,
                      bool is_drt = false, FP cutoff = (FP)0.0) const {
        unordered_map<string, int> r_str_mp;
        vector<vector<uint16_t>> r_indices;
        vector<vector<FL>> r_data;
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            shared_ptr<T> scheme = schemes[ix];
            unordered_map<vector<uint16_t>, int, vector_uint16_hasher>
                idx_pattern_mp;
            vector<unordered_map<vector<uint16_t>, int, vector_uint16_hasher>>
                idx_perm_mp((int)scheme->data.size());
            int kk = 0, nn = (int)scheme->index_patterns[0].size();
            for (int i = 0; i < (int)scheme->data.size(); i++) {
                idx_pattern_mp[scheme->index_patterns[i]] = i;
                for (auto &j : scheme->data[i])
                    idx_perm_mp[i][j.first] = kk++;
            }
            vector<pair<int, vector<uint16_t>>> idx_pats(kk);
            for (int i = 0; i < (int)scheme->data.size(); i++)
                for (auto &x : idx_perm_mp[i])
                    idx_pats[x.second] = make_pair(i, x.first);
            // first divide all indices according to scheme classes
            vector<vector<size_t>> idx_patidx(kk);
            if (indices.size() == 0)
                continue;
            vector<uint16_t> idx_idx(nn), idx_pat(nn), idx_mat(nn);
            for (size_t i = 0; i < (nn == 0 ? 1 : indices[ix].size());
                 i += (nn == 0 ? 1 : nn)) {
                for (int j = 0; j < nn; j++)
                    idx_idx[j] = j;
                sort(idx_idx.begin(), idx_idx.begin() + nn,
                     [this, ix, i](uint16_t x, uint16_t y) {
                         return this->indices[ix][i + x] <
                                this->indices[ix][i + y];
                     });
                if (nn >= 1)
                    idx_mat[0] = 0;
                for (int j = 1; j < nn; j++)
                    idx_mat[j] =
                        idx_mat[j - 1] + (indices[ix][i + idx_idx[j]] !=
                                          indices[ix][i + idx_idx[j - 1]]);
                for (int j = 0; j < nn; j++)
                    idx_pat[idx_idx[j]] = j;
                idx_patidx[idx_perm_mp[idx_pattern_mp.at(idx_mat)].at(idx_pat)]
                    .push_back(i);
            }
            kk = (int)r_str_mp.size();
            // collect all reordered expr types
            for (auto i : scheme->data)
                for (auto &j : i)
                    for (auto &k : j.second)
                        if (!r_str_mp.count(k.second))
                            r_str_mp[k.second] = kk++;
            if (r_indices.size() < r_str_mp.size())
                r_indices.resize(r_str_mp.size());
            if (r_data.size() < r_str_mp.size())
                r_data.resize(r_str_mp.size());
            for (size_t ip = 0; ip < idx_patidx.size(); ip++) {
                const vector<pair<FLX, string>> &strd =
                    scheme->data[idx_pats[ip].first].at(idx_pats[ip].second);
                for (auto i : idx_patidx[ip]) {
                    idx_pat = vector<uint16_t>(indices[ix].begin() + i,
                                               indices[ix].begin() + i + nn);
                    sort(idx_pat.begin(), idx_pat.end());
                    for (int j = 0; j < (int)strd.size(); j++) {
                        vector<uint16_t> &iridx =
                            r_indices[r_str_mp.at(strd[j].second)];
                        vector<FL> &irdata =
                            r_data[r_str_mp.at(strd[j].second)];
                        iridx.insert(iridx.end(), idx_pat.begin(),
                                     idx_pat.end());
                        irdata.push_back((FL)(data[ix][nn == 0 ? i : i / nn] *
                                              (FL)strd[j].first));
                    }
                }
            }
        }
        shared_ptr<GeneralFCIDUMP> r = make_shared<GeneralFCIDUMP>();
        r->params = params;
        r->const_e = const_e;
        int dcnt = 0;
        for (auto &rx : r_data)
            dcnt += rx.size() != 0;
        r->elem_type = elem_type;
        r->exprs.resize(r_str_mp.size());
        r->indices.reserve(dcnt);
        r->data.reserve(dcnt);
        for (auto &x : r_str_mp)
            r->exprs[x.second] = x.first;
        for (int i = 0, j = 0; i < r->exprs.size(); i++) {
            r->exprs[j] = r->exprs[i];
            if (r_data[i].size() != 0) {
                r->indices.push_back(r_indices[i]);
                r->data.push_back(r_data[i]), j++;
            }
        }
        r->exprs.resize(dcnt);
        if (merge)
            r->merge_terms(cutoff);
        return r;
    }
    shared_ptr<GeneralFCIDUMP> adjust_order(bool merge = true,
                                            bool is_drt = false,
                                            FP cutoff = (FP)0.0) const {
        if (elem_type == ElemOpTypes::SAny)
            return adjust_order(vector<shared_ptr<GeneralSymmPermScheme<FL>>>(),
                                merge, is_drt, cutoff);
        else
            return adjust_order(vector<shared_ptr<SpinPermScheme>>(), merge,
                                is_drt, cutoff);
    }
    shared_ptr<GeneralFCIDUMP>
    adjust_order(const vector<shared_ptr<SpinPermScheme>> &schemes,
                 bool merge = true, bool is_drt = false,
                 FP cutoff = (FP)0.0) const {
        vector<shared_ptr<SpinPermScheme>> psch = schemes;
        if (psch.size() < exprs.size()) {
            psch.resize(exprs.size(), nullptr);
            for (size_t ix = 0; ix < exprs.size(); ix++) {
                vector<uint16_t> mask = find_mask(ix);
                psch[ix] = make_shared<SpinPermScheme>(
                    exprs[ix], elem_type == ElemOpTypes::SU2,
                    elem_type != ElemOpTypes::SGB, false, is_drt, mask);
            }
        }
        return adjust_order_impl<SpinPermScheme, double>(psch, merge, is_drt,
                                                         cutoff);
    }
    shared_ptr<GeneralFCIDUMP>
    adjust_order(const vector<shared_ptr<GeneralSymmPermScheme<FL>>> &schemes,
                 bool merge = true, bool is_drt = false,
                 FP cutoff = (FP)0.0) const {
        vector<shared_ptr<GeneralSymmPermScheme<FL>>> psch = schemes;
        vector<shared_ptr<AnyCG<FL>>> cgs(1, make_shared<AnySO3RSHCG<FL>>());
        if (psch.size() < exprs.size()) {
            psch.resize(exprs.size(), nullptr);
            for (size_t ix = 0; ix < exprs.size(); ix++) {
                vector<uint16_t> mask = find_mask(ix);
                psch[ix] = make_shared<GeneralSymmPermScheme<FL>>(
                    exprs[ix], cgs, false, is_drt, mask);
            }
        }
        return adjust_order_impl<GeneralSymmPermScheme<FL>, FL>(psch, merge,
                                                                is_drt, cutoff);
    }
    vector<uint16_t> find_mask(size_t ix) const {
        const int nn = SpinPermRecoupling::count_cds(exprs[ix]);
        size_t nidx = indices[ix].size() / (nn == 0 ? 1 : nn);
        int ntg = threading->activate_global();
        const size_t pidx = nidx / ntg + !!(nidx % ntg);
        vector<uint8_t> qq(nn * nn * ntg, 0);
#pragma omp parallel num_threads(ntg)
        {
            int tid = threading->get_thread_id();
            vector<uint8_t> q(nn * nn, 0);
            for (size_t im = pidx * tid; im < min(nidx, pidx * (tid + 1)); im++)
                for (int ii = 0; ii < nn; ii++)
                    for (int jj = ii + 1; jj < nn; jj++)
                        if (indices[ix][im * nn + ii] !=
                            indices[ix][im * nn + jj])
                            q[ii * nn + jj] = 1;
            for (int ii = 0; ii < nn; ii++)
                for (int jj = ii + 1; jj < nn; jj++)
                    if (q[ii * nn + jj])
                        qq[tid * nn * nn + ii * nn + jj] = 1;
        }
        threading->activate_normal();
        for (int it = 1; it < ntg; it++)
            for (int ii = 0; ii < nn; ii++)
                for (int jj = ii + 1; jj < nn; jj++)
                    if (qq[it * nn * nn + ii * nn + jj])
                        qq[ii * nn + jj] = 1;
        DSU dsu(nn);
        for (int ii = 0; ii < nn; ii++)
            for (int jj = ii + 1; jj < nn; jj++)
                if (!qq[ii * nn + jj])
                    dsu.unionx(ii, jj);
        vector<uint16_t> rr(nn);
        for (int ii = 0; ii < nn; ii++)
            rr[ii] = dsu.findx(ii);
        return rr;
    }
    void merge_terms(FP cutoff = (FP)0.0) {
        vector<size_t> idx;
        for (int ix = 0; ix < (int)exprs.size(); ix++) {
            idx.clear();
            idx.reserve(data[ix].size());
            for (size_t i = 0; i < data[ix].size(); i++)
                idx.push_back(i);
            const int nn = SpinPermRecoupling::count_cds(exprs[ix]);
            sort(idx.begin(), idx.end(), [nn, ix, this](size_t i, size_t j) {
                for (int ic = 0; ic < nn; ic++)
                    if (this->indices[ix][i * nn + ic] !=
                        this->indices[ix][j * nn + ic])
                        return this->indices[ix][i * nn + ic] <
                               this->indices[ix][j * nn + ic];
                return false;
            });
            for (size_t i = 1; i < idx.size(); i++) {
                bool eq = true;
                for (int ic = 0; ic < nn; ic++)
                    if (this->indices[ix][idx[i] * nn + ic] !=
                        this->indices[ix][idx[i - 1] * nn + ic]) {
                        eq = false;
                        break;
                    }
                if (eq)
                    data[ix][idx[i]] += data[ix][idx[i - 1]],
                        data[ix][idx[i - 1]] = 0;
            }
            size_t cnt = 0;
            for (size_t i = 0; i < idx.size(); i++)
                if (abs(data[ix][i]) > cutoff)
                    cnt++;
            vector<uint16_t> r_indices;
            vector<FL> r_data;
            r_indices.reserve(cnt * nn);
            r_data.reserve(cnt);
            for (size_t i = 0; i < idx.size(); i++)
                if (abs(data[ix][idx[i]]) > cutoff) {
                    r_indices.insert(r_indices.end(),
                                     indices[ix].begin() + idx[i] * nn,
                                     indices[ix].begin() + (idx[i] + 1) * nn);
                    r_data.push_back(data[ix][idx[i]]);
                }
            indices[ix] = r_indices;
            data[ix] = r_data;
        }
    }
    // Target 2S or 2Sz
    uint16_t twos() const {
        return (uint16_t)Parsing::to_int(params.at("ms2"));
    }
    // Number of sites
    uint16_t n_sites() const {
        return (uint16_t)Parsing::to_int(params.at("norb"));
    }
    // Number of electrons
    uint16_t n_elec() const {
        return (uint16_t)Parsing::to_int(params.at("nelec"));
    }
    virtual typename const_fl_type<FL>::FL e() const { return const_e; }
    template <typename T> vector<T> orb_sym() const {
        vector<string> x = Parsing::split(params.at("orbsym"), ",", true);
        vector<T> r;
        r.reserve(x.size());
        for (auto &xx : x)
            r.push_back((T)Parsing::to_int(xx));
        return r;
    }
    friend ostream &operator<<(ostream &os, GeneralFCIDUMP x) {
        if (x.params.size() != 0) {
            os << " NSITES = " << x.n_sites() << " NELEC = " << x.n_elec();
            os << " TWOS = " << x.twos() << endl;
        }
        os << " SU2 = " << (x.elem_type == ElemOpTypes::SU2);
        os << fixed << setprecision(16) << " CONST E = " << x.const_e << endl;
        for (int ix = 0; ix < (int)x.exprs.size(); ix++) {
            os << " TERM " << x.exprs[ix] << " ::" << endl;
            size_t lg = x.data[ix].size() == 0
                            ? 0
                            : x.indices[ix].size() / x.data[ix].size();
            for (size_t ic = 0, ig = 0; ic < x.data[ix].size(); ic++) {
                for (; ig < ic * lg + lg; ig++)
                    os << setw(7) << x.indices[ix][ig];
                os << " = " << fixed << setprecision(16) << x.data[ix][ic]
                   << endl;
            }
        }
        return os;
    }
    static shared_ptr<GeneralFCIDUMP> add(const shared_ptr<GeneralFCIDUMP> &a,
                                          const shared_ptr<GeneralFCIDUMP> &b) {
        shared_ptr<GeneralFCIDUMP> r =
            make_shared<GeneralFCIDUMP>(a->elem_type);
        assert(a->elem_type == b->elem_type);
        r->const_e = a->const_e + b->const_e;
        r->params = a->params;
        r->exprs = a->exprs;
        r->indices = a->indices;
        r->data = a->data;
        r->exprs.insert(r->exprs.end(), b->exprs.begin(), b->exprs.end());
        r->indices.insert(r->indices.end(), b->indices.begin(),
                          b->indices.end());
        r->data.insert(r->data.end(), b->data.begin(), b->data.end());
        r->order_adjusted = a->order_adjusted && b->order_adjusted;
        return r;
    }
};

} // namespace block2
