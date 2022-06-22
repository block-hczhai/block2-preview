
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

/** Automatic construction of MPO. */

#pragma once

#include "../core/integral.hpp"
#include "spin_permutation.hpp"
#include <array>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename FL> struct AutoFCIDUMP {
    typedef decltype(abs((FL)0.0)) FP;
    map<string, string> params;
    FL const_e;
    vector<string> exprs;
    vector<vector<uint16_t>> indices;
    vector<vector<FL>> data;
    bool order_adjusted = false;
    AutoFCIDUMP() {}
    static shared_ptr<AutoFCIDUMP>
    initialize_from_qc(const shared_ptr<FCIDUMP<FL>> &fcidump,
                       FP cutoff = (FP)0.0) {
        shared_ptr<AutoFCIDUMP> r = make_shared<AutoFCIDUMP>();
        r->params = fcidump->params;
        r->const_e = fcidump->e();
        uint16_t n = fcidump->n_sites();
        if (!fcidump->uhf) {
            r->exprs.push_back("((C+(C+D)0)1+D)0");
            r->indices.push_back(vector<uint16_t>());
            r->data.push_back(vector<FL>());
            auto *idx = &r->indices.back(), *dt = &r->data.back();
            array<uint16_t, 4> arr;
            for (arr[0] = 0; arr[0] < n; arr[0]++)
                for (arr[1] = 0; arr[1] < n; arr[1]++)
                    for (arr[2] = 0; arr[2] < n; arr[2]++)
                        for (arr[3] = 0; arr[3] < n; arr[3]++)
                            if (abs(fcidump->v(arr[0], arr[1], arr[2],
                                               arr[3])) > cutoff) {
                                idx->insert(idx->end(), arr.begin(), arr.end());
                                dt->push_back(
                                    fcidump->v(arr[0], arr[1], arr[2], arr[3]));
                            }
            r->exprs.push_back("(C+D)0");
            r->indices.push_back(vector<uint16_t>());
            r->data.push_back(vector<FL>());
            idx = &r->indices.back(), dt = &r->data.back();
            for (arr[0] = 0; arr[0] < n; arr[0]++)
                for (arr[1] = 0; arr[1] < n; arr[1]++)
                    if (abs(fcidump->t(arr[0], arr[1])) > cutoff) {
                        idx->insert(idx->end(), arr.begin(), arr.begin() + 2);
                        dt->push_back(fcidump->t(arr[0], arr[1]));
                    }
        } else {
            r->exprs.push_back("((c+(c+d)0)1+d)0");
            r->exprs.push_back("((c+(C+D)0)1+d)0");
            r->exprs.push_back("((C+(c+d)0)1+D)0");
            r->exprs.push_back("((C+(C+D)0)1+D)0");
            for (uint8_t si = 0; si < 2; si++)
                for (uint8_t sj = 0; sj < 2; sj++) {
                    r->indices.push_back(vector<uint16_t>());
                    r->data.push_back(vector<FL>());
                    auto *idx = &r->indices.back(), *dt = &r->data.back();
                    array<uint16_t, 4> arr;
                    for (arr[0] = 0; arr[0] < n; arr[0]++)
                        for (arr[1] = 0; arr[1] < n; arr[1]++)
                            for (arr[2] = 0; arr[2] < n; arr[2]++)
                                for (arr[3] = 0; arr[3] < n; arr[3]++)
                                    if (abs(fcidump->v(si, sj, arr[0], arr[1],
                                                       arr[2], arr[3])) >
                                        cutoff) {
                                        idx->insert(idx->end(), arr.begin(),
                                                    arr.end());
                                        dt->push_back(fcidump->v(si, sj, arr[0],
                                                                 arr[1], arr[2],
                                                                 arr[3]));
                                    }
                }
            r->exprs.push_back("(c+d)0");
            r->exprs.push_back("(C+D)0");
            for (uint8_t si = 0; si < 2; si++) {
                r->indices.push_back(vector<uint16_t>());
                r->data.push_back(vector<FL>());
                auto *idx = &r->indices.back(), *dt = &r->data.back();
                for (arr[0] = 0; arr[0] < n; arr[0]++)
                    for (arr[1] = 0; arr[1] < n; arr[1]++)
                        if (abs(fcidump->t(si, arr[0], arr[1])) > cutoff) {
                            idx->insert(idx->end(), arr.begin(),
                                        arr.begin() + 2);
                            dt->push_back(fcidump->t(si, arr[0], arr[1]));
                        }
            }
        }
        return r;
    }
    struct vector_uint16_hasher {
        size_t operator()(const vector<uint16_t> &x) const {
            size_t r = x.size();
            for (auto &i : x)
                r ^= i + 0x9e3779b9 + (r << 6) + (r >> 2);
            return r;
        }
    };
    shared_ptr<AutoFCIDUMP>
    adjust_order(const vector<shared_ptr<SpinPermScheme>> &schemes =
                     vector<shared_ptr<SpinPermScheme>>(),
                 bool merge = true, FP cutoff = (FP)0.0) const {
        vector<shared_ptr<SpinPermScheme>> psch = schemes;
        if (psch.size() < exprs.size()) {
            psch.resize(exprs.size(), nullptr);
            for (size_t ix = 0; ix < exprs.size(); ix++) {
                vector<uint8_t> cds;
                string xcd = SpinPermRecoupling::split_cds(exprs[ix], cds);
                psch[ix] =
                    make_shared<SpinPermScheme>((int)cds.size(), xcd, cds);
            }
        }
        unordered_map<string, int> r_str_mp;
        vector<vector<uint16_t>> r_indices;
        vector<vector<FL>> r_data;
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            shared_ptr<SpinPermScheme> scheme = psch[ix];
            unordered_map<vector<uint16_t>, pair<int, int>,
                          vector_uint16_hasher>
                idx_pattern_mp;
            int kk = 0, nn = (int)scheme->index_patterns[0].size();
            for (int i = 0; i < (int)scheme->data.size(); i++)
                for (auto &j : scheme->data[i])
                    if (!idx_pattern_mp.count(j.first))
                        idx_pattern_mp[j.first] = make_pair(kk++, i);
            vector<vector<uint16_t>> idx_pats(idx_pattern_mp.size());
            for (auto &x : idx_pattern_mp)
                idx_pats[x.second.first] = x.first;
            vector<vector<int>> idx_patidx(idx_pattern_mp.size());
            if (indices.size() == 0)
                continue;
            vector<uint16_t> idx_idx(nn);
            vector<uint16_t> idx_pat(nn);
            vector<uint16_t> idx_mat(nn);
            for (size_t i = 0; i < indices[ix].size(); i += nn) {
                for (int j = 0; j < nn; j++)
                    idx_idx[j] = j;
                sort(idx_idx.begin(), idx_idx.begin() + nn,
                     [this, ix, i](uint16_t x, uint16_t y) {
                         return this->indices[ix][i + x] <
                                this->indices[ix][i + y];
                     });
                idx_mat[0] = 0;
                for (int j = 1; j < nn; j++)
                    idx_mat[j] =
                        idx_mat[j - 1] + (indices[ix][i + idx_idx[j]] !=
                                          indices[ix][i + idx_idx[j - 1]]);
                for (int j = 0; j < nn; j++)
                    idx_pat[idx_idx[j]] = idx_mat[j];
                idx_patidx[idx_pattern_mp.at(idx_pat).first].push_back(i);
            }
            kk = 0;
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
                vector<uint16_t> &ipat = idx_pats[ip];
                int schi = idx_pattern_mp.at(ipat).second;
                vector<pair<double, string>> &strd =
                    shceme->data[idx_pattern_mp.at(ipat).second].at(ipat);
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
                        irdata.push_back(
                            (FL)(data[ix][i / nn] * strd[j].first));
                    }
                }
            }
        }
        shared_ptr<AutoFCIDUMP> r = make_shared<AutoFCIDUMP>();
        r->params = params;
        r->const_e = const_e;
        r->exprs.resize(r_str_mp.size());
        for (auto &x : r_str_mp)
            r->exprs[x.second] = x.first;
        r->indices = r_indices;
        r->data = r_data;
        if (merge)
            r->merge_terms(cutoff);
        return r;
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
            vector<vector<uint16_t>> r_indices;
            vector<vector<FL>> r_data;
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
};

} // namespace block2