
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
#include "../core/hamiltonian.hpp"
#include "../core/integral.hpp"
#include "../core/spin_permutation.hpp"
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

enum struct ElemOpTypes : uint8_t { SU2, SZ, SGF, SGB };

enum struct MPOAlgorithmTypes : uint16_t {
    None = 0,
    Bipartite = 1,
    SVD = 2,
    Rescaled = 4,
    Fast = 8,
    NC = 16,
    CN = 32,
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
    const static string repr[] = {
        "None", "BIP",     "SVD",     "", "Res", "", "ResSVD",     "", //
        "Fast", "FastBIP", "FastSVD", "", "",    "", "FastResSVD", "", //
        "NC",   "",        "",        "", "",    "", "",           "", //
        "",     "",        "",        "", "",    "", "",           "", "CN"};
    os << repr[(uint16_t)c];
    return os;
}

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
    GeneralFCIDUMP() {}
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
                 bool merge = true, FP cutoff = (FP)0.0) const {
        vector<shared_ptr<SpinPermScheme>> psch = schemes;
        if (psch.size() < exprs.size()) {
            psch.resize(exprs.size(), nullptr);
            for (size_t ix = 0; ix < exprs.size(); ix++)
                psch[ix] = make_shared<SpinPermScheme>(
                    exprs[ix], elem_type == ElemOpTypes::SU2,
                    elem_type != ElemOpTypes::SGB);
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
            // first divide all indices according to scheme classes
            vector<vector<int>> idx_patidx(idx_pattern_mp.size());
            if (indices.size() == 0)
                continue;
            vector<uint16_t> idx_idx(nn);
            vector<uint16_t> idx_pat(nn);
            vector<uint16_t> idx_mat(max(1, nn));
            for (size_t i = 0; i < (nn == 0 ? 1 : indices[ix].size());
                 i += (nn == 0 ? 1 : nn)) {
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
                vector<uint16_t> &ipat = idx_pats[ip];
                int schi = idx_pattern_mp.at(ipat).second;
                vector<pair<double, string>> &strd =
                    scheme->data[idx_pattern_mp.at(ipat).second].at(ipat);
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
        opf->cg->initialize();
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
    void init_site_ops() {
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
            op_prims[""]->allocate(find_site_op_info(m, S(0, 0, 0)));
            (*op_prims[""])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims[""])[S(1, -1, ipg)](0, 0) = 1.0;
            (*op_prims[""])[S(1, 1, ipg)](0, 0) = 1.0;
            (*op_prims[""])[S(2, 0, S::pg_mul(ipg, ipg))](0, 0) = 1.0;

            op_prims["c"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["c"]->allocate(find_site_op_info(m, S(1, 1, ipg)));
            (*op_prims["c"])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims["c"])[S(1, -1, ipg)](0, 0) = 1.0;

            op_prims["d"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["d"]->allocate(
                find_site_op_info(m, S(-1, -1, S::pg_inv(ipg))));
            (*op_prims["d"])[S(1, 1, ipg)](0, 0) = 1.0;
            (*op_prims["d"])[S(2, 0, S::pg_mul(ipg, ipg))](0, 0) = 1.0;

            op_prims["C"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["C"]->allocate(find_site_op_info(m, S(1, -1, ipg)));
            (*op_prims["C"])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims["C"])[S(1, 1, ipg)](0, 0) = -1.0;

            op_prims["D"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["D"]->allocate(
                find_site_op_info(m, S(-1, 1, S::pg_inv(ipg))));
            (*op_prims["D"])[S(1, -1, ipg)](0, 0) = 1.0;
            (*op_prims["D"])[S(2, 0, S::pg_mul(ipg, ipg))](0, 0) = -1.0;
        }
        // site norm operators
        const string stx[5] = {"", "c", "C", "d", "D"};
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->info->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
    }
    void get_site_string_ops(
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
    static vector<vector<S>>
    init_string_quanta(const vector<string> &exprs,
                       const vector<uint16_t> &term_l) {
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
    pair<S, S> get_string_quanta(const vector<S> &ref, const string &expr,
                                 const uint16_t *idxs, uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        for (uint16_t j = 0; j < (uint16_t)expr.length(); j++) {
            typename S::pg_t ipg = orb_sym[idxs[j]];
            if (expr[j] == 'd' || expr[j] == 'D')
                ipg = S::pg_inv(ipg);
            if (j < k)
                l.set_pg(S::pg_mul(l.pg(), ipg));
            else
                r.set_pg(S::pg_mul(r.pg(), ipg));
        }
        return make_pair(l, r);
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
        opf->cg->deallocate();
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
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>(100));
        opf->cg->initialize();
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
    void init_site_ops() {
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
    void get_site_string_ops(
        uint16_t m,
        unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        for (auto &p : ops)
            p.second = get_site_string_op(m, p.first);
    }
    static vector<vector<S>>
    init_string_quanta(const vector<string> &exprs,
                       const vector<uint16_t> &term_l) {
        vector<vector<S>> r(exprs.size());
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            r[ix].resize(term_l[ix] + 1, S(0, 0, 0));
            vector<pair<int, char>> pex;
            pex.reserve(exprs[ix].length());
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
                    pex.push_back(make_pair(-1, x));
                }
            assert(pex.back().first != -1);
            int cnt = 0;
            bool is_heis = false;
            // singlet embedding (twos will be set later)
            r[ix][0].set_n(pex.back().first);
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
            const int site_dq = is_heis ? 2 : 1;
            if (r[ix].size() >= 2) {
                r[ix][r[ix].size() - 2].set_twos(site_dq);
                r[ix][r[ix].size() - 2].set_twos_low(site_dq);
            }
        }
        return r;
    }
    pair<S, S> get_string_quanta(const vector<S> &ref, const string &expr,
                                 const uint16_t *idxs, uint16_t k) const {
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
    static string get_sub_expr(const string &expr, int i, int j) {
        int cnt = 0, depth = 0, start = -1, extra = 0;
        stringstream ss;
        for (auto &c : expr) {
            if (c == '(') {
                depth++;
                if (cnt == i && start == -1 && j > i + 1)
                    start = depth;
                if (cnt >= i && cnt < j && start != -1 && depth >= start)
                    ss << c, extra++;
            } else if (c == ')') {
                depth--;
                if (cnt >= i && cnt <= j && start != -1 && depth + 1 >= start)
                    ss << c, extra--;
            } else if (c >= '0' && c <= '9') {
                if (cnt >= i && cnt <= j && start != -1 && depth + 1 >= start)
                    ss << c;
            } else if (c == '+') {
                if (cnt >= i && cnt < j && start != -1 && depth >= start)
                    ss << c;
            } else if (c == 'C' || c == 'D' || c == 'T') {
                if (cnt >= i && cnt < j)
                    ss << c;
                cnt++;
            }
        }
        return ss.str().substr(extra);
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
        opf->cg->deallocate();
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
        opf->cg->initialize();
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
    void init_site_ops() {
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
    void get_site_string_ops(
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
    static vector<vector<S>>
    init_string_quanta(const vector<string> &exprs,
                       const vector<uint16_t> &term_l) {
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
    pair<S, S> get_string_quanta(const vector<S> &ref, const string &expr,
                                 const uint16_t *idxs, uint16_t k) const {
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
        opf->cg->deallocate();
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
        opf->cg->initialize();
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
            b->quanta[i] = S(tm, 0), b->n_states[i] = 1;
        b->sort_states();
        return b;
    }
    void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            for (int n = -max_n; n <= max_n; n++)
                info[S(n, 0)] = nullptr;
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            const typename S::pg_t ipg = 0;
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
                (*op_prims[""])[S(tm, 0)](0, 0) = 1.0;

            op_prims["P"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["P"]->allocate(find_site_op_info(m, S(2, 0)));
            for (int tm = -twos; tm < twos - 1; tm += 2)
                (*op_prims["P"])[S(tm, 0)](0, 0) =
                    sqrt((twos - tm) * (twos + tm + 2) / 4);

            op_prims["M"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["M"]->allocate(find_site_op_info(m, S(-2, 0)));
            for (int tm = -twos + 2; tm < twos + 1; tm += 2)
                (*op_prims["M"])[S(tm, 0)](0, 0) =
                    sqrt((twos + tm) * (twos - tm + 2) / 4);

            op_prims["Z"] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            op_prims["Z"]->allocate(find_site_op_info(m, S(0, 0)));
            for (int tm = -twos; tm < twos + 1; tm += 2)
                (*op_prims["Z"])[S(tm, 0)](0, 0) = (FL)tm / (FL)2.0;
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
    void get_site_string_ops(
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
    static vector<vector<S>>
    init_string_quanta(const vector<string> &exprs,
                       const vector<uint16_t> &term_l) {
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
    pair<S, S> get_string_quanta(const vector<S> &ref, const string &expr,
                                 const uint16_t *idxs, uint16_t k) const {
        S l = ref[k], r = ref.back() - l;
        return make_pair(l, r);
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
        opf->cg->deallocate();
        Hamiltonian<S, FL>::deallocate();
    }
};

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
               int max_bond_dim = -1, bool iprint = true,
               const string &tag = "HQC")
        : MPO<S, FL>(hamil->n_sites, tag), algo_type(algo_type) {
        bool rescale = algo_type & MPOAlgorithmTypes::Rescaled;
        bool fast = algo_type & MPOAlgorithmTypes::Fast;
        if (!(algo_type & MPOAlgorithmTypes::SVD) && max_bond_dim != -1)
            throw runtime_error(
                "Max bond dimension can only be used together with SVD!");
        else if (!(algo_type & MPOAlgorithmTypes::SVD) && rescale)
            throw runtime_error(
                "Rescaling can only be used together with SVD!");
        else if ((algo_type & MPOAlgorithmTypes::NC) &&
                 algo_type != MPOAlgorithmTypes::NC)
            throw runtime_error("Invalid MPO algorithm type with NC!");
        else if ((algo_type & MPOAlgorithmTypes::CN) &&
                 algo_type != MPOAlgorithmTypes::CN)
            throw runtime_error("Invalid MPO algorithm type with CN!");
        else if (algo_type == MPOAlgorithmTypes::None)
            throw runtime_error("Invalid MPO algorithm None!");
        vector<typename S::pg_t> orb_sym = hamil->orb_sym;
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::const_e = afd->e();
        MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        n_sites = (int)orb_sym.size();
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
        for (int ix = 0; ix < (int)afd->exprs.size(); ix++) {
            const int nn = SpinPermRecoupling::count_cds(afd->exprs[ix]);
            term_l[ix] = nn;
            term_i[ix].resize(afd->data[ix].size(), 0);
            term_k[ix].resize(afd->data[ix].size(), -1);
            assert(afd->indices[ix].size() == nn * term_i[ix].size());
            n_terms += (LL)afd->data[ix].size();
        }
        vector<vector<S>> quanta_ref =
            GeneralHamiltonian<S, FL>::init_string_quanta(afd->exprs, term_l);
        S qh = hamil->vacuum, left_vacuum = hamil->vacuum, actual_qh = qh;
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
        if (iprint) {
            cout << "Build MPO | Nsites = " << setw(5) << n_sites
                 << " | Nterms = " << setw(10) << n_terms
                 << " | Algorithm = " << algo_type
                 << " | Cutoff = " << scientific << setw(8) << setprecision(2)
                 << cutoff;
            if (algo_type & MPOAlgorithmTypes::SVD)
                cout << " | Max bond dimension = " << setw(5) << max_bond_dim;
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
        unordered_map<S, int> q_map;
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
        // cache of operator strings
        vector<map<pair<uint16_t, uint16_t>, string>> sub_exprs(
            afd->exprs.size());
        FL rsc_factor = 1;
        Timer _t, _t2;
        double tsite, tsvd;
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
                        pair<S, S> pq = hamil->get_string_quanta(
                            quanta_ref[ix], afd->exprs[ix],
                            &afd->indices[ix][itt], 0);
                        q_map[qh.combine(pq.first, -pq.second)] = 0;
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
                    pair<S, S> pq =
                        hamil->get_string_quanta(quanta_ref[ix], afd->exprs[ix],
                                                 &afd->indices[ix][itt], k);
                    S qq = qh.combine(pq.first, -pq.second);
                    // possible error here due to unsymmetrized integral
                    assert(qq != S(S::invalid));
                    if (q_map.count(qq) == 0) {
                        q_map[qq] = (int)q_map.size();
                        map_ls.emplace_back();
                        map_rs.emplace_back();
                        mats.emplace_back();
                        nms.push_back(make_pair(0, 0));
                    }
                    int iq = q_map.at(qq), il = -1, ir = -1;
                    LL &nml = nms[iq].first, &nmr = nms[iq].second;
                    auto &mpl = map_ls[iq];
                    auto &mpr = map_rs[iq];
                    if (mpl.count(hl)) {
                        int iq = 0;
                        auto &vq = mpl.at(hl);
                        for (; iq < vq.size(); iq++) {
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
                            vq.push_back(
                                make_pair(make_pair(ip, ic), il = nml++));
                        else
                            il = vq[iq].second;
                    } else
                        mpl[hl].push_back(
                            make_pair(make_pair(ip, ic), il = nml++));
                    if (mpr.count(hr)) {
                        int iq = 0;
                        auto &vq = mpr.at(hr);
                        for (; iq < vq.size(); iq++) {
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
                            vq.push_back(
                                make_pair(make_pair(ip, ic), ir = nmr++));
                        else
                            ir = vq[iq].second;
                    } else
                        mpr[hr].push_back(
                            make_pair(make_pair(ip, ic), ir = nmr++));
                    // cout << "il = " << il << " ir = " << ir
                    //      << " ql = " << qq.get_bra(qh)
                    //      << " qr = " << -qq.get_ket() << endl;
                    mats[iq].push_back(make_pair(make_pair(il, ir), itv));
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
            for (auto &mq : q_map)
                qs[mq.second] = mq.first;
            int s_kept_total = 0, nr_total = 0;
            FP res_s_sum = 0, res_factor = 1;
            size_t res_s_count = 0;
            for (auto &mq : q_map) {
                int iq = mq.second;
                auto &matvs = mats[iq];
                auto &nm = nms[iq];
                int szl = nm.first, szr = nm.second, szm;
                // cout << "iq = " << iq << " q = " << qs[iq] << " szl = " <<
                // szl
                //      << " szr = " << szr << endl;
                if (algo_type & MPOAlgorithmTypes::NC)
                    szm = ii == n_sites - 1 ? szr : szl;
                else if (algo_type & MPOAlgorithmTypes::CN)
                    szm = ii == 0 ? szl : szr;
                else // bipartitie / SVD
                    szm = min(szl, szr);
                if (!(algo_type & MPOAlgorithmTypes::Bipartite)) {
                    if (delayed_term != -1 && iq == 0)
                        szm = min(szl - 1, szr) + 1;
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
                                mat[(lrv.first.first - 1) * szr +
                                    lrv.first.second] += lrv.second;
                        szl--;
                        svds[iq].second[0] = 1;
                        svds[iq].first[0][0] = 1;
                        threading->activate_global_mkl();
                        GMatrixFunctions<FL>::svd(
                            GMatrix<FL>(mat.data(), szl, szr),
                            GMatrix<FL>(svds[iq].first[0].data() + 1 + szm, szl,
                                        szm),
                            GMatrix<FP>(svds[iq].second.data() + 1, 1, szm - 1),
                            GMatrix<FL>(svds[iq].first[1].data() + szr, szm - 1,
                                        szr));
                        threading->activate_normal();
                        szl++;
                    } else {
                        for (auto &lrv : matvs)
                            mat[lrv.first.first * szr + lrv.first.second] +=
                                lrv.second;
                        // cout << "mat = " << GMatrix<FL>(mat.data(), szl, szr)
                        // << endl;
                        threading->activate_global_mkl();
                        GMatrixFunctions<FL>::svd(
                            GMatrix<FL>(mat.data(), szl, szr),
                            GMatrix<FL>(svds[iq].first[0].data(), szl, szm),
                            GMatrix<FP>(svds[iq].second.data(), 1, szm),
                            GMatrix<FL>(svds[iq].first[1].data(), szm, szr));
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
                                   svds[iq].second.end(), 0, plus<FP>());
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
            if (ii == n_sites - 1 && s_kept_total != 1)
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
                    int szl = nm.first, szr = nm.second;
                    int szm = min(szl, szr);
                    if (delayed_term != -1 && iq == 0)
                        szm = min(szl - 1, szr) + 1;
                    for (int i = 0; i < szm; i++)
                        svds[iq].second[i] /= res_factor;
                    for (int i = 0; i < szm; i++)
                        if (svds[iq].second[i] > sqrt(cutoff))
                            s_kept++;
                        else
                            discarded_weights[ii] +=
                                svds[iq].second[i] * svds[iq].second[i];
                    svds[iq].second.resize(s_kept);
                    s_kept_total += s_kept;
                }
            }
            if (max_bond_dim >= 1) {
                vector<pair<int, int>> idxs;
                for (auto &mq : q_map) {
                    int iq = mq.second;
                    for (size_t i = 0; i < svds[iq].second.size(); i++)
                        idxs.push_back(make_pair(iq, i));
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
            if (iprint) {
                cout << "Mmpo = " << setw(5) << s_kept_total
                     << " Error = " << scientific << setw(8) << setprecision(2)
                     << discarded_weights[ii];
                cout.flush();
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
                int szl = nm.first, szr = nm.second, szm;
                if (algo_type & MPOAlgorithmTypes::NC)
                    szm = ii == n_sites - 1 ? szr : szl;
                else if (algo_type & MPOAlgorithmTypes::CN)
                    szm = ii == 0 ? szl : szr;
                else if (algo_type & MPOAlgorithmTypes::Bipartite)
                    szm = (int)mvcs[iq][0].size() + (int)mvcs[iq][1].size();
                else { // SVD
                    szm = min(szl, szr);
                    if (delayed_term != -1 && iq == 0)
                        szm = min(szl - 1, szr) + 1;
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
                        // for non-singlet Hamiltonian this may not be satisfied
                        assert(ii != 0 && ii != n_sites - 1);
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
            for (int iq = 0; iq < (int)qs.size(); iq++) {
                S qq = qs[iq];
                auto &mpr = map_rs[iq];
                auto &nm = nms[iq];
                int szr = nm.second, szl = nm.first;
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
                    if (mat.data[iop]->get_type() != OpTypes::Zero)
                        assert((dynamic_pointer_cast<OpElement<S, FL>>(lop[iop])
                                    ->q_label ==
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    mat.data[iop])
                                    ->q_label));
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
            }
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
    }
    virtual ~GeneralMPO() = default;
};

} // namespace block2