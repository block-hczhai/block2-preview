
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2023 Huanchen Zhai <hczhai@caltech.edu>
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

/** Hilbert space definition for any symmetry. */

#pragma once

#include "../core/hamiltonian.hpp"
#include "../core/integral.hpp"
#include "../core/iterative_matrix_functions.hpp"
#include "../core/spin_permutation.hpp"
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

enum struct ElemOpTypes : uint8_t { SU2, SZ, SGF, SGB };

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
                            indices.back()[xs + 0] = pm[xim];
                            indices.back()[xs + 1] = pm[xjm];
                            indices.back()[xs + 2] = qm[xjm];
                            indices.back()[xs + 3] = qm[xim];
                            data.back()[istart] = factor * vals[xm[im] + jm];
                            istart++, xs += 4;
                            if (gm[xim] == 2) {
                                indices.back()[xs + 0] = qm[xim];
                                indices.back()[xs + 1] = pm[xjm];
                                indices.back()[xs + 2] = qm[xjm];
                                indices.back()[xs + 3] = pm[xim];
                                data.back()[istart] =
                                    factor * vals[xm[im] + jm];
                                istart++, xs += 4;
                            }
                            if (gm[xjm] == 2) {
                                indices.back()[xs + 0] = pm[xim];
                                indices.back()[xs + 1] = qm[xjm];
                                indices.back()[xs + 2] = pm[xjm];
                                indices.back()[xs + 3] = qm[xim];
                                data.back()[istart] =
                                    factor * vals[xm[im] + jm];
                                istart++, xs += 4;
                            }
                            if (gm[xim] == 2 && gm[xjm] == 2) {
                                indices.back()[xs + 0] = qm[xim];
                                indices.back()[xs + 1] = qm[xjm];
                                indices.back()[xs + 2] = pm[xjm];
                                indices.back()[xs + 3] = pm[xim];
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
                      vector<uint16_t> rperm = vector<uint16_t>()) {
        int ntg = threading->activate_global();
        vector<size_t> lens(ntg + 1, 0);
        const size_t plen = len / ntg + !!(len % ntg);
        if (rperm.size() == 0)
            for (size_t i = 0; i < shape.size(); i++)
                rperm.push_back(i);
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
                                i / strides[j] % shape[j];
                        data.back()[istart] = factor * vals[i];
                        istart++;
                    }
            } else {
                for (size_t i = plen * tid; i < min(len, plen * (tid + 1)); i++)
                    if (abs(factor * vals[i]) > cutoff) {
                        int irrep = 0;
                        for (int j = 0; j < (int)shape.size(); j++) {
                            indices.back()[istart * shape.size() + rperm[j]] =
                                i / strides[j] % shape[j];
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
    shared_ptr<GeneralFCIDUMP>
    adjust_order(const vector<shared_ptr<SpinPermScheme>> &schemes =
                     vector<shared_ptr<SpinPermScheme>>(),
                 bool merge = true, bool is_drt = false,
                 FP cutoff = (FP)0.0) const {
        vector<shared_ptr<SpinPermScheme>> psch = schemes;
        if (psch.size() < exprs.size()) {
            psch.resize(exprs.size(), nullptr);
            for (size_t ix = 0; ix < exprs.size(); ix++)
                psch[ix] = make_shared<SpinPermScheme>(
                    exprs[ix], elem_type == ElemOpTypes::SU2,
                    elem_type != ElemOpTypes::SGB, false, is_drt);
        }
        unordered_map<string, int> r_str_mp;
        vector<vector<uint16_t>> r_indices;
        vector<vector<FL>> r_data;
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            shared_ptr<SpinPermScheme> scheme = psch[ix];
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
            vector<vector<int>> idx_patidx(kk);
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
                vector<pair<double, string>> &strd =
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
                                              (FL)(FP)strd[j].first));
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
};

template <typename, typename, typename = void> struct GeneralHamiltonian;

// General Hamiltonian (non-spin-adapted)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename S::is_sz_t> : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    struct pair_hasher {
        size_t
        operator()(const pair<typename S::pg_t, typename S::pg_t> &x) const {
            size_t r = x.first;
            r ^= x.second + 0x9e3779b9 + (r << 6) + (r >> 2);
            return r;
        }
    };
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<pair<typename S::pg_t, typename S::pg_t>,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>,
                  pair_hasher>
        op_prims;
    const static int max_n = 10, max_s = 10;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        // alpha and beta orbitals can have different pg symmetries
        return orb_sym.size() != n_sites * 2
                   ? SiteBasis<S>::get(orb_sym[m])
                   : SiteBasis<S>::get(orb_sym[m], orb_sym[m + n_sites]);
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = -max_s_odd; s <= max_s_odd; s += 2) {
                    info[S(n, s, orb_sym[m])] = nullptr;
                    info[S(n, s, S::pg_inv(orb_sym[m]))] = nullptr;
                    if (orb_sym.size() == n_sites * 2) {
                        info[S(n, s, orb_sym[m + n_sites])] = nullptr;
                        info[S(n, s, S::pg_inv(orb_sym[m + n_sites]))] =
                            nullptr;
                    }
                }
            for (int n = -max_n_even; n <= max_n_even; n += 2)
                for (int s = -max_s_even; s <= max_s_even; s += 2) {
                    info[S(n, s, S::pg_mul(orb_sym[m], orb_sym[m]))] = nullptr;
                    info[S(n, s,
                           S::pg_mul(orb_sym[m], S::pg_inv(orb_sym[m])))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]), orb_sym[m]))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]),
                                     S::pg_inv(orb_sym[m])))] = nullptr;
                    if (orb_sym.size() == n_sites * 2) {
                        info[S(n, s,
                               S::pg_mul(orb_sym[m], orb_sym[m + n_sites]))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(orb_sym[m],
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m]),
                                         orb_sym[m + n_sites]))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m]),
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(orb_sym[m + n_sites],
                                         orb_sym[m + n_sites]))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(orb_sym[m + n_sites],
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m + n_sites]),
                                         orb_sym[m + n_sites]))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(orb_sym[m + n_sites]),
                                         S::pg_inv(orb_sym[m + n_sites])))] =
                            nullptr;
                    }
                }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            typename S::pg_t ipga = orb_sym[m], ipgb = ipga;
            if (orb_sym.size() == n_sites * 2)
                ipgb = orb_sym[m + n_sites];
            pair<typename S::pg_t, typename S::pg_t> ipg =
                make_pair(ipga, ipgb);
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
            (*op_prims[""])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims[""])[S(1, 1, ipga)](0, 0) = 1.0;
            (*op_prims[""])[S(1, -1, ipgb)](0, 0) = 1.0;
            (*op_prims[""])[S(2, 0, S::pg_mul(ipga, ipgb))](0, 0) = 1.0;

            op_prims["c"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["c"]->allocate(find_site_op_info(m, S(1, 1, ipga)));
            (*op_prims["c"])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims["c"])[S(1, -1, ipgb)](0, 0) = 1.0;

            op_prims["d"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["d"]->allocate(
                find_site_op_info(m, S(-1, -1, S::pg_inv(ipga))));
            (*op_prims["d"])[S(1, 1, ipga)](0, 0) = 1.0;
            (*op_prims["d"])[S(2, 0, S::pg_mul(ipga, ipgb))](0, 0) = 1.0;

            op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["C"]->allocate(find_site_op_info(m, S(1, -1, ipgb)));
            (*op_prims["C"])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims["C"])[S(1, 1, ipga)](0, 0) = -1.0;

            op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["D"]->allocate(
                find_site_op_info(m, S(-1, 1, S::pg_inv(ipgb))));
            (*op_prims["D"])[S(1, -1, ipgb)](0, 0) = 1.0;
            (*op_prims["D"])[S(2, 0, S::pg_mul(ipga, ipgb))](0, 0) = -1.0;
        }
        // site norm operators
        const string stx[5] = {"", "c", "C", "d", "D"};
        for (uint16_t m = 0; m < n_sites; m++) {
            typename S::pg_t ipga = orb_sym[m], ipgb = ipga;
            if (orb_sym.size() == n_sites * 2)
                ipgb = orb_sym[m + n_sites];
            pair<typename S::pg_t, typename S::pg_t> ipg =
                make_pair(ipga, ipgb);
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(m,
                                      op_prims.at(ipg)[t]->info->delta_quantum),
                    op_prims.at(ipg)[t]->data);
            }
        }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tmp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        for (auto &p : ops) {
            if (site_norm_ops[m].count(p.first))
                p.second = site_norm_ops[m].at(p.first);
            else {
                p.second = site_norm_ops[m].at(string(1, p.first[0]));
                for (size_t i = 1; i < p.first.length(); i++) {
                    S q = p.second->info->delta_quantum +
                          site_norm_ops[m]
                              .at(string(1, p.first[i]))
                              ->info->delta_quantum;
                    tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                    tmp->allocate(find_site_op_info(m, q));
                    opf->product(0, p.second,
                                 site_norm_ops[m].at(string(1, p.first[i])),
                                 tmp);
                    p.second = tmp;
                }
                site_norm_ops[m][p.first] = p.second;
            }
        }
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1);
            r[ix][0] = S(0, 0, 0);
            for (int i = 0; i < term_l[ix]; i++)
                switch (exprs[ix][i]) {
                case 'c':
                    r[ix][i + 1] = r[ix][i] + S(1, 1, 0);
                    break;
                case 'C':
                    r[ix][i + 1] = r[ix][i] + S(1, -1, 0);
                    break;
                case 'd':
                    r[ix][i + 1] = r[ix][i] + S(-1, -1, 0);
                    break;
                case 'D':
                    r[ix][i + 1] = r[ix][i] + S(-1, 1, 0);
                    break;
                default:
                    assert(false);
                }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (orb_sym.size() == n_sites * 2 &&
                (expr[j] == 'C' || expr[j] == 'D'))
                ipg = orb_sym[idxs[j] + n_sites];
            if (expr[j] == 'd' || expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0, 0);
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
            if (idxs != nullptr && orb_sym.size() == n_sites * 2 &&
                (expr[j] == 'C' || expr[j] == 'D'))
                ipg = orb_sym[idxs[j] + n_sites];
            if (expr[j] == 'c')
                r = r + S(1, 1, ipg);
            else if (expr[j] == 'C')
                r = r + S(1, -1, ipg);
            else if (expr[j] == 'd')
                r = r + S(-1, -1, S::pg_inv(ipg));
            else if (expr[j] == 'D')
                r = r + S(-1, 1, S::pg_inv(ipg));
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return expr.substr(i, j - i);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (spin-adapted)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename S::is_su2_t> : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 10, max_s = 10;
    int twos;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym), twos(twos) {
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        if (twos == -1) // fermion model
            return SiteBasis<S>::get(orb_sym[m]);
        else // heisenberg spin model
            return make_shared<StateInfo<S>>(S(twos, twos, orb_sym[m]));
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = 1; s <= max_s_odd; s += 2) {
                    info[S(n, s, orb_sym[m])] = nullptr;
                    info[S(n, s, S::pg_inv(orb_sym[m]))] = nullptr;
                }
            for (int n = -max_n_even; n <= max_n_even; n += 2)
                for (int s = 0; s <= max_s_even; s += 2) {
                    info[S(n, s, S::pg_mul(orb_sym[m], orb_sym[m]))] = nullptr;
                    info[S(n, s,
                           S::pg_mul(orb_sym[m], S::pg_inv(orb_sym[m])))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]), orb_sym[m]))] =
                        nullptr;
                    info[S(n, s,
                           S::pg_mul(S::pg_inv(orb_sym[m]),
                                     S::pg_inv(orb_sym[m])))] = nullptr;
                }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        vector<string> stx = vector<string>{"", "C", "D"};
        if (twos == -1) { // fermion model
            for (uint16_t m = 0; m < n_sites; m++) {
                const typename S::pg_t ipg = orb_sym[m];
                if (this->op_prims.count(ipg) == 0)
                    this->op_prims[ipg] =
                        unordered_map<string,
                                      shared_ptr<SparseMatrix<S, FL>>>();
                else
                    continue;
                unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>
                    &op_prims = this->op_prims.at(ipg);
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
                (*op_prims[""])[S(0, 0, 0, 0)](0, 0) = 1.0;
                (*op_prims[""])[S(1, 1, 1, ipg)](0, 0) = 1.0;
                (*op_prims[""])[S(2, 0, 0, S::pg_mul(ipg, ipg))](0, 0) = 1.0;

                op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["C"]->allocate(find_site_op_info(m, S(1, 1, ipg)));
                (*op_prims["C"])[S(0, 1, 0, 0)](0, 0) = 1.0;
                (*op_prims["C"])[S(1, 0, 1, ipg)](0, 0) = -sqrt(2);

                op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["D"]->allocate(
                    find_site_op_info(m, S(-1, 1, S::pg_inv(ipg))));
                (*op_prims["D"])[S(1, 0, 1, ipg)](0, 0) = sqrt(2);
                (*op_prims["D"])[S(2, 1, 0, S::pg_mul(ipg, ipg))](0, 0) = 1.0;
            }
        } else { // heisenberg spin model
            stx = vector<string>{"", "T"};
            for (uint16_t m = 0; m < n_sites; m++) {
                const typename S::pg_t ipg = orb_sym[m];
                if (this->op_prims.count(ipg) == 0)
                    this->op_prims[ipg] =
                        unordered_map<string,
                                      shared_ptr<SparseMatrix<S, FL>>>();
                else
                    continue;
                unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>
                    &op_prims = this->op_prims.at(ipg);
                op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
                (*op_prims[""])[S(twos, twos, twos, 0)](0, 0) = 1.0;

                op_prims["T"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims["T"]->allocate(find_site_op_info(m, S(0, 2, 0)));
                (*op_prims["T"])[S(twos, twos, twos, 0)](0, 0) =
                    sqrt((FL)twos * ((FL)twos + (FL)2.0) / (FL)2.0);
            }
        }
        // site norm operators
        for (uint16_t m = 0; m < n_sites; m++) {
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
        }
    }
    shared_ptr<SparseMatrix<S, FL>> get_site_string_op(uint16_t m,
                                                       const string &expr) {
        if (site_norm_ops[m].count(expr))
            return site_norm_ops[m].at(expr);
        else {
            int ix = 0, depth = 0;
            for (auto &c : expr) {
                if (c == '(')
                    depth++;
                else if (c == ')')
                    depth--;
                else if (c == '+' && depth == 1)
                    break;
                ix++;
            }
            int twos = 0, iy = 0;
            for (int i = (int)expr.length() - 1, k = 1; i >= 0; i--, k *= 10)
                if (expr[i] >= '0' && expr[i] <= '9')
                    twos += (expr[i] - '0') * k;
                else {
                    iy = i;
                    break;
                }
            shared_ptr<SparseMatrix<S, FL>> a =
                get_site_string_op(m, expr.substr(1, ix - 1));
            shared_ptr<SparseMatrix<S, FL>> b =
                get_site_string_op(m, expr.substr(ix + 1, iy - ix - 1));
            S dq = a->info->delta_quantum + b->info->delta_quantum;
            dq.set_twos(twos);
            dq.set_twos_low(twos);
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            shared_ptr<SparseMatrix<S, FL>> r =
                make_shared<SparseMatrix<S, FL>>(d_alloc);
            r->allocate(find_site_op_info(m, dq));
            opf->product(0, a, b, r);
            site_norm_ops[m][expr] = r;
            return r;
        }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1, S(0, 0, 0));
            vector<pair<int, char>> pex;
            pex.reserve(exprs[ix].length());
            bool is_heis = false;
            for (char x : exprs[ix])
                if (x >= '0' && x <= '9') {
                    if (pex.back().first != -1)
                        pex.back().first =
                            pex.back().first * 10 + (int)(x - '0');
                    else
                        pex.push_back(make_pair((int)(x - '0'), ' '));
                } else {
                    if (x == '+' && pex.back().second == ' ')
                        pex.back().second = '*';
                    else if (x == 'T')
                        is_heis = true;
                    pex.push_back(make_pair(-1, x));
                }
            const int site_dq = is_heis ? 2 : 1;
            if (pex.size() == 0)
                continue;
            else if (pex.size() == 1 && pex.back().first == -1) {
                // single C/D
                pex.insert(pex.begin(), make_pair(-1, '('));
                pex.push_back(make_pair(site_dq, ' '));
            }
            assert(pex.back().first != -1);
            int cnt = 0;
            // singlet embedding (twos will be set later)
            if (left_vacuum == S(S::invalid))
                r[ix][0].set_n(pex.back().first);
            else {
                assert(left_vacuum.twos() == pex.back().first);
                r[ix][0].set_n(left_vacuum.n());
                r[ix][0].set_pg(left_vacuum.pg());
            }
            vector<uint16_t> stk;
            for (auto &p : pex) {
                if (p.second == '(')
                    stk.push_back(cnt);
                // numbers in exprs like (()0+.) will not be used
                else if (p.second == '*')
                    stk.pop_back();
                // use the right part to define the twos for the left part
                // because the right part is not affected by the singlet
                // embedding
                else if (p.second == ' ') {
                    r[ix][stk.back()].set_twos(p.first);
                    r[ix][stk.back()].set_twos_low(p.first);
                    stk.pop_back();
                } else if (p.second == 'C')
                    r[ix][cnt + 1].set_n(r[ix][cnt].n() + 1), cnt++;
                else if (p.second == 'D')
                    r[ix][cnt + 1].set_n(r[ix][cnt].n() - 1), cnt++;
                else if (p.second == 'T')
                    cnt++, is_heis = true;
            }
            if (r[ix].size() >= 2) {
                r[ix][r[ix].size() - 2].set_twos(site_dq);
                r[ix][r[ix].size() - 2].set_twos_low(site_dq);
            }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0, i = 0; j < (uint16_t)expr.length(); j++) {
            if (expr[j] != 'C' && expr[j] != 'D')
                continue;
            typename S::pg_t ipg = orb_sym[idxs[i]];
            if (expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (i < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
            i++;
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0, 0);
        for (uint16_t j = 0, i = 0; j < (uint16_t)expr.length(); j++) {
            if (expr[j] != 'C' && expr[j] != 'D')
                continue;
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[i]] : 0;
            if (expr[j] == 'C')
                r = r + S(1, 1, ipg);
            else if (expr[j] == 'D')
                r = r + S(-1, 1, S::pg_inv(ipg));
            i++;
        }
        int rr = SpinPermRecoupling::get_target_twos(expr);
        r.set_twos(rr);
        r.set_twos_low(rr);
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return SpinPermRecoupling::get_sub_expr(expr, i, j);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (general spin, fermionic)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename enable_if<S::GIF>::type>
    : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 10, max_s = 10;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        return SiteBasis<S>::get(orb_sym[m]);
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n_odd; n <= max_n_odd; n += 2) {
                info[S(n, orb_sym[m])] = nullptr;
                info[S(n, S::pg_inv(orb_sym[m]))] = nullptr;
            }
            for (int n = -max_n_even; n <= max_n_even; n += 2) {
                info[S(n, S::pg_mul(orb_sym[m], orb_sym[m]))] = nullptr;
                info[S(n, S::pg_mul(orb_sym[m], S::pg_inv(orb_sym[m])))] =
                    nullptr;
                info[S(n, S::pg_mul(S::pg_inv(orb_sym[m]), orb_sym[m]))] =
                    nullptr;
                info[S(n, S::pg_mul(S::pg_inv(orb_sym[m]),
                                    S::pg_inv(orb_sym[m])))] = nullptr;
            }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            const typename S::pg_t ipg = orb_sym[m];
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0)));
            (*op_prims[""])[S(0, 0)](0, 0) = 1.0;
            (*op_prims[""])[S(1, ipg)](0, 0) = 1.0;

            op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["C"]->allocate(find_site_op_info(m, S(1, ipg)));
            (*op_prims["C"])[S(0, 0)](0, 0) = 1.0;

            op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["D"]->allocate(
                find_site_op_info(m, S(-1, S::pg_inv(ipg))));
            (*op_prims["D"])[S(1, ipg)](0, 0) = 1.0;
        }
        // site norm operators
        const string stx[3] = {"", "C", "D"};
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tmp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        for (auto &p : ops) {
            if (site_norm_ops[m].count(p.first))
                p.second = site_norm_ops[m].at(p.first);
            else {
                p.second = site_norm_ops[m].at(string(1, p.first[0]));
                for (size_t i = 1; i < p.first.length(); i++) {
                    S q = p.second->info->delta_quantum +
                          site_norm_ops[m]
                              .at(string(1, p.first[i]))
                              ->info->delta_quantum;
                    tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                    tmp->allocate(find_site_op_info(m, q));
                    opf->product(0, p.second,
                                 site_norm_ops[m].at(string(1, p.first[i])),
                                 tmp);
                    p.second = tmp;
                }
                site_norm_ops[m][p.first] = p.second;
            }
        }
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1);
            r[ix][0] = S(0, 0);
            for (int i = 0; i < term_l[ix]; i++)
                switch (exprs[ix][i]) {
                case 'C':
                    r[ix][i + 1] = r[ix][i] + S(1, 0);
                    break;
                case 'D':
                    r[ix][i + 1] = r[ix][i] + S(-1, 0);
                    break;
                default:
                    assert(false);
                }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0);
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
            if (expr[j] == 'C')
                r = r + S(1, ipg);
            else if (expr[j] == 'D')
                r = r + S(-1, S::pg_inv(ipg));
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return expr.substr(i, j - i);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

// General Hamiltonian (general spin, bosonic or spin)
template <typename S, typename FL>
struct GeneralHamiltonian<S, FL, typename enable_if<!S::GIF>::type>
    : Hamiltonian<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using Hamiltonian<S, FL>::vacuum;
    using Hamiltonian<S, FL>::n_sites;
    using Hamiltonian<S, FL>::basis;
    using Hamiltonian<S, FL>::site_op_infos;
    using Hamiltonian<S, FL>::orb_sym;
    using Hamiltonian<S, FL>::find_site_op_info;
    using Hamiltonian<S, FL>::opf;
    using Hamiltonian<S, FL>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    const static int max_n = 20;
    int twos;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(
        S vacuum, int n_sites,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int twos = -1)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym), twos(twos) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        if (orb_sym.size() == 0)
            Hamiltonian<S, FL>::orb_sym.resize(n_sites);
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual ~GeneralHamiltonian() = default;
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(twos + 1);
        for (int i = 0, tm = -twos; tm < twos + 1; i++, tm += 2)
            b->quanta[i] = S(tm, tm == twos ? orb_sym[m] : 0),
            b->n_states[i] = 1;
        b->sort_states();
        return b;
    }
    virtual void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n; n <= max_n; n++) {
                info[S(n, orb_sym[m])] = nullptr;
                info[S(n, S::pg_inv(orb_sym[m]))] = nullptr;
                info[S(n, 0)] = nullptr;
            }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            const typename S::pg_t ipg = orb_sym[m];
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] =
                    unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>();
            else
                continue;
            unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &op_prims =
                this->op_prims.at(ipg);
            op_prims[""] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0)));
            for (int tm = -twos; tm < twos + 1; tm += 2)
                (*op_prims[""])[S(tm, tm == twos ? ipg : 0)](0, 0) = 1.0;

            op_prims["P"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["P"]->allocate(find_site_op_info(m, S(2, ipg)));
            for (int tm = -twos; tm < twos - 1; tm += 2)
                (*op_prims["P"])[S(tm, tm == twos ? ipg : 0)](0, 0) =
                    sqrt((twos - tm) * (twos + tm + 2) / 4);

            op_prims["M"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["M"]->allocate(
                find_site_op_info(m, S(-2, S::pg_inv(ipg))));
            for (int tm = -twos + 2; tm < twos + 1; tm += 2)
                (*op_prims["M"])[S(tm, tm == twos ? ipg : 0)](0, 0) =
                    sqrt((twos + tm) * (twos - tm + 2) / 4);

            op_prims["Z"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["Z"]->allocate(find_site_op_info(m, S(0, 0)));
            for (int tm = -twos; tm < twos + 1; tm += 2)
                (*op_prims["Z"])[S(tm, tm == twos ? ipg : 0)](0, 0) =
                    (FL)tm / (FL)2.0;
        }
        // site norm operators
        const string stx[4] = {"", "P", "M", "Z"};
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
    }
    virtual void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tmp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        for (auto &p : ops) {
            if (site_norm_ops[m].count(p.first))
                p.second = site_norm_ops[m].at(p.first);
            else {
                p.second = site_norm_ops[m].at(string(1, p.first[0]));
                for (size_t i = 1; i < p.first.length(); i++) {
                    S q = p.second->info->delta_quantum +
                          site_norm_ops[m]
                              .at(string(1, p.first[i]))
                              ->info->delta_quantum;
                    tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                    tmp->allocate(find_site_op_info(m, q));
                    opf->product(0, p.second,
                                 site_norm_ops[m].at(string(1, p.first[i])),
                                 tmp);
                    p.second = tmp;
                }
                site_norm_ops[m][p.first] = p.second;
            }
        }
    }
    virtual vector<vector<S>> init_string_quanta(const vector<string> &exprs,
                                                 const vector<uint16_t> &term_l,
                                                 S left_vacuum) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1);
            r[ix][0] = S(0, 0);
            for (int i = 0; i < term_l[ix]; i++)
                switch (exprs[ix][i]) {
                case 'P':
                    r[ix][i + 1] = r[ix][i] + S(2, 0);
                    break;
                case 'M':
                    r[ix][i + 1] = r[ix][i] + S(-2, 0);
                    break;
                case 'Z':
                    r[ix][i + 1] = r[ix][i];
                    break;
                default:
                    assert(false);
                }
        }
        return r;
    }
    virtual pair<S, S> get_string_quanta(const vector<S> &ref,
                                         const string &expr,
                                         const uint16_t *idxs,
                                         uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (expr[j] == 'Z')
                continue;
            else if (expr[j] == 'M')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
    }
    S get_string_quantum(const string &expr,
                         const uint16_t *idxs) const override {
        S r(0, 0);
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = idxs != nullptr ? orb_sym[idxs[j]] : 0;
            if (expr[j] == 'Z')
                continue;
            else if (expr[j] == 'P')
                r = r + S(2, ipg);
            else if (expr[j] == 'M')
                r = r + S(-2, S::pg_inv(ipg));
        }
        return r;
    }
    static string get_sub_expr(const string &expr, int i, int j) {
        return expr.substr(i, j - i);
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_norm_ops : this->site_norm_ops)
            for (auto &p : site_norm_ops)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

} // namespace block2