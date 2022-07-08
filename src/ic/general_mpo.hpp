
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

#include "../core/hamiltonian.hpp"
#include "../core/integral.hpp"
#include "../dmrg/mpo.hpp"
#include "spin_permutation.hpp"
#include <array>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename FL> struct GeneralFCIDUMP {
    typedef decltype(abs((FL)0.0)) FP;
    map<string, string> params;
    FL const_e;
    vector<string> exprs;
    vector<vector<uint16_t>> indices;
    vector<vector<FL>> data;
    bool order_adjusted = false;
    GeneralFCIDUMP() {}
    static shared_ptr<GeneralFCIDUMP>
    initialize_from_qc(const shared_ptr<FCIDUMP<FL>> &fcidump,
                       FP cutoff = (FP)0.0) {
        shared_ptr<GeneralFCIDUMP> r = make_shared<GeneralFCIDUMP>();
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
    shared_ptr<GeneralFCIDUMP>
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
        shared_ptr<GeneralFCIDUMP> r = make_shared<GeneralFCIDUMP>();
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
    template <typename T> vector<T> orb_sym() const {
        vector<string> x = Parsing::split(params.at("orbsym"), ",", true);
        vector<T> r;
        r.reserve(x.size());
        for (auto &xx : x)
            r.push_back((T)Parsing::to_int(xx));
        return r;
    }
};

template <typename, typename, typename = void> struct GeneralHamiltonian;

// Quantum chemistry Hamiltonian (non-spin-adapted)
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
    vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        site_long_ops;
    // Primitives for sparse matrix representation for normal site operators
    unordered_map<typename S::pg_t,
                  unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>
        op_prims;
    // Chemical potenital parameter in Hamiltonian
    FL mu = 0;
    GeneralHamiltonian()
        : Hamiltonian<S, FL>(S(), 0, vector<typename S::pg_t>()) {}
    GeneralHamiltonian(S vacuum, int n_sites,
                       const vector<typename S::pg_t> &orb_sym)
        : Hamiltonian<S, FL>(vacuum, n_sites, orb_sym) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S, FL>>(make_shared<CG<S>>());
        opf->cg->initialize();
        basis.resize(n_sites);
        site_op_infos.resize(n_sites);
        site_long_ops.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual void set_mu(FL mu) { this->mu = mu; }
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        return SiteBasis<S>::get(orb_sym[m]);
    }
    void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        const int max_n = 10, max_s = 10;
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vacuum] = nullptr;
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
        const int sz[2] = {1, -1};
        for (uint16_t m = 0; m < n_sites; m++) {
            const typename S::pg_t ipg = orb_sym[m];
            if (this->op_prims.count(ipg) == 0)
                this->op_prims[ipg] = vector<
                    unordered_map<OpNames, shared_ptr<SparseMatrix<S, FL>>>>(6);
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
            for (uint8_t s = 0; s < 2; s++) {
                op_prims[s == 0 ? "c" : "C"] =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[s == 0 ? "c" : "C"]->allocate(
                    find_site_op_info(m, S(1, sz[s], ipg)));
                (*op_prims[s == 0 ? "c" : "C"])[S(0, 0, 0)](0, 0) = 1.0;
                (*op_prims[s == 0 ? "c" : "C"])[S(1, -sz[s], ipg)](0, 0) =
                    s ? -1.0 : 1.0;
                op_prims[s == 0 ? "d" : "D"] =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
                op_prims[s == 0 ? "d" : "D"]->allocate(
                    find_site_op_info(m, S(-1, -sz[s], S::pg_inv(ipg))));
                (*op_prims[s == 0 ? "d" : "D"])[S(1, sz[s], ipg)](0, 0) = 1.0;
                (*op_prims[s == 0 ? "d" : "D"])[S(2, 0, S::pg_mul(ipg, ipg))](
                    0, 0) = s ? -1.0 : 1.0;
            }
        }
        // site norm operators
        const shared_ptr<OpElement<S, FL>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), vacuum);
        const string stx[5] = {"", "c", "C", "d", "D"};
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto t : stx) {
                site_norm_ops[m][t] = make_shared<SparseMatrix<S, FL>>(nullptr);
                site_norm_ops[m][t]->allocate(
                    find_site_op_info(
                        m, op_prims.at(orb_sym[m])[t]->delta_quantum),
                    op_prims.at(orb_sym[m])[t]->data);
            }
    }
    void
    get_site_ops(uint16_t m,
                 unordered_map<string, shared_ptr<SparseMatrix<S, FL>>> &ops) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> tmp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        zero->factor = 0.0;
        auto &op_prims = this->op_prims.at(orb_sym[m]);
        for (auto &p : ops) {
            if (site_norm_ops[m].count(p.first))
                p.second = site_norm_ops[m].at(p.first);
            else if (site_long_ops[m].count(p.first))
                p.second = site_long_ops[m].at(p.first);
            else {
                p.second = site_long_ops[m].at(string(1, p.first[0]));
                for (size_t i = 1; i < p.first.length(); i++) {
                    S q =
                        p.second->delta_quantum + site_norm_ops[m]
                                                      .at(string(1, p.first[i]))
                                                      ->delta_quantum;
                    tmp = make_shared<SparseMatrix<S, FL>>(d_alloc);
                    tmp->allocate(find_site_op_info(m, q));
                    opf->product(0, p.second,
                                 site_norm_ops[m].at(string(1, p.first[i])),
                                 tmp);
                    p.second = tmp;
                }
                site_long_ops[m][p.first] = p.second;
            }
        }
    }
    void deallocate() override {
        for (auto &op_prims : this->op_prims)
            for (auto &p : op_prims.second)
                p.second->deallocate();
        for (auto &site_long_ops : this->site_long_ops)
            for (auto &p : site_long_ops.second)
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

// max_bond_dim >= -1: SVD
// max_bond_dim = -2: NC
// max_bond_dim = -3: CN
// max_bond_dim = -4: bipartite O(K^5)
// max_bond_dim = -5: fast bipartite O(K^4)
// max_bond_dim = -6: SVD (rescale)
// max_bond_dim = -7: SVD (rescale, fast)
// max_bond_dim = -8: SVD (fast)
template <typename S, typename FL> struct GeneralMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::tensors;
    using MPO<S, FL>::site_op_infos;
    using MPO<S, FL>::left_operator_names;
    using MPO<S, FL>::right_operator_names;
    using MPO<S, FL>::middle_operator_names;
    using MPO<S, FL>::middle_operator_exprs;
    using MPO<S, FL>::basis;
    GeneralMPO(const shared_ptr<GeneralHamiltonian<S, FL>> &hamil,
               const shared_ptr<GeneralFCIDUMP<FL>> &afd, FP cutoff = (FL)0.0,
               int max_bond_dim = -1) {
        bool rescale = false, fast_k4 = false;
        if (max_bond_dim == -6)
            rescale = true, fast_k4 = false, max_bond_dim = -1;
        else if (max_bond_dim == -7)
            rescale = true, fast_k4 = true, max_bond_dim = -1;
        else if (max_bond_dim == -8)
            rescale = false, fast_k4 = true, max_bond_dim = -1;
        else if (max_bond_dim == -5)
            fast_k4 = true, max_bond_dim = -4;
        vector<typename S::pg_t> orb_sym = hamil->orb_sym;
        uint16_t n_sites = (int)orb_sym.size();
        S vacuum = hamil->vacuum;
        vector<S> left_q = {vacuum};
        unordered_map<S, uint32_t> info_l, info_r;
        info_l[vacuum] = 1;
        // length of each term; starting index of each term
        // at the beginning, term_i is all zero
        vector<int> term_l(afd->exprs.size()), term_i(afd->exprs.size());
        long long int n_terms = 0;
        for (int ix = 0; ix < (int)afd->exprs.size(); ix++) {
            const int nn = SpinPermRecoupling::count_cds(afd->exprs[ix]);
            term_l[ix] = nn;
            term_i[ix].resize(afd->data[ix].size());
            assert(afd->indices[ix].size() == nn * term_i[ix].size());
            n_terms += (long long int)afd->data[ix].size();
        }
        // index of current terms
        // in future, cur_terms should have an extra level
        // indicating the term length
        vector<vector<pair<int, long long int>>> cur_terms(1);
        vector<vector<FL>> cur_values(1);
        cur_terms[0].resize(n_terms);
        cur_values[0].resize(n_terms);
        size_t ik = 0;
        for (int ix = 0; ix < (int)afd->exprs.size(); ix++)
            for (size_t it = 0; it < afd->data[ix].size(); it++) {
                cur_terms[0][ik] = make_pair(ix, (long long int)it);
                cur_values[0][ik] = afd->data[ix][it];
            }
        assert(!fast_k4);
        // do svd from left to right
        // time complexity: O(KDLN(log N))
        // K: n_sites, D: max_bond_dim, L: term_len, N: n_terms
        // using block-structure according to left q number
        // this is the map from left q number to its block index
        unordered_map<S, int> q_map;
        // for each iq block, a map from hashed repr of string of op in left
        // block to (mpo index, term index, left block string of op index)
        vector<
            unordered_map<size_t, vector<pair<pair<int, long long int>, int>>>>
            map_ls;
        // for each iq block, a map from hashed repr of string of op in right
        // block to (term index, right block string of op index)
        vector<unordered_map<size_t, vector<pair<long long int, int>>>> map_rs;
        // sparse repr of the connection (edge) matrix for each block
        vector<vector<pair<pair<int, int>, FL>>> mats;
        // for each block, the nrow and ncol of the block
        vector<pair<long long int, long long int>> nms;
        vector<int> cur_term_i(n_terms, -1);
        FL rsc_factor = 1;
        for (int ii = 0; ii < n_sites; ii++) {
            cout << "MPO site" << setw(4) << ii << " / " << n_sites << endl;
            q_map.clear();
            map_ls.clear();
            map_rs.clear();
            mats.clear();
            nms.clear();
            info_r.clear();
            long long int pholder_term = -1;
            // iter over all mpos
            for (int ip = 0; ip < (int)cur_values.size(); ip++) {
                S qll = left_q[ip];
                long long int cn = (long long int)cur_terms[ip].size(),
                              cnr = cn;
                // if (prefix_part.size() != 0 && ip == 0) {
                //     cn += prefix_part[ii + 1] - prefix_part[ii];
                //     if (prefix_part[ii + 1] != prefix_part[n_sites]) {
                //         pholder_term = prefix_terms[prefix_part[ii + 1]];
                //         q_map[qll] = 0;
                //         map_ls.emplace_back();
                //         map_rs.emplace_back();
                //         mats.emplace_back();
                //         nms.push_back(make_pair(1, 1));
                //         map_ls[0][0].push_back(
                //             make_pair(make_pair(0, pholder_term), 0));
                //         map_rs[0][0].push_back(make_pair(pholder_term, 0));
                //         mats[0].push_back(make_pair(
                //             make_pair(0, 0),
                //             prefix_values[pholder_term] * rsc_factor));
                //         cur_term_i[pholder_term] = 0;
                //     }
                // }
                for (long long int ic = 0; ic < cn; ic++) {
                    long long int it =
                        ic < cnr ? cur_terms[ip][ic]
                                 : prefix_terms[ic - cnr + prefix_part[ii]];
                    FL itv = ic < cnr ? cur_values[ip][ic]
                                      : prefix_values[it] * rsc_factor;
                    int ik = term_i[it], k = ik, kmax = term_l[it];
                    long long int itt = it * term_len;
                    // separate the current product into two parts
                    // (left block part and right block part)
                    for (; k < kmax &&
                           (term_sorted[itt + k] % m_op) / m_site <= ii;
                         k++)
                        ;
                    // first right site position
                    cur_term_i[it] = k;
                    size_t hl =
                        op_hash(term_sorted.data() + itt + ik, k - ik, ip);
                    size_t hr = op_hash(term_sorted.data() + itt + k, kmax - k);
                    SZ ql = qll;
                    for (int i = ik; i < k; i++)
                        ql = ql +
                             from_op(term_sorted[itt + i], porb, m_site, m_op);
                    if (q_map.count(ql) == 0) {
                        q_map[ql] = (int)q_map.size();
                        map_ls.emplace_back();
                        map_rs.emplace_back();
                        mats.emplace_back();
                        nms.push_back(make_pair(0, 0));
                    }
                    int iq = q_map.at(ql), il = -1, ir = -1;
                    long long int &nml = nms[iq].first, &nmr = nms[iq].second;
                    auto &mpl = map_ls[iq];
                    auto &mpr = map_rs[iq];
                    if (mpl.count(hl)) {
                        int iq = 0;
                        auto &vq = mpl.at(hl);
                        for (; iq < vq.size(); iq++) {
                            int vip = vq[iq].first.first;
                            long long int vit = vq[iq].first.second;
                            long long int vitt = vit * term_len;
                            int vik = term_i[vit], vk = cur_term_i[vit];
                            if (vip == ip && vk - vik == k - ik &&
                                equal(term_sorted.data() + vitt + vik,
                                      term_sorted.data() + vitt + vk,
                                      term_sorted.data() + itt + ik))
                                break;
                        }
                        if (iq == (int)vq.size())
                            vq.push_back(
                                make_pair(make_pair(ip, it), il = nml++));
                        else
                            il = vq[iq].second;
                    } else
                        mpl[hl].push_back(
                            make_pair(make_pair(ip, it), il = nml++));
                    if (mpr.count(hr)) {
                        int iq = 0;
                        auto &vq = mpr.at(hr);
                        for (; iq < vq.size(); iq++) {
                            int vit = vq[iq].first, vitt = vit * term_len;
                            int vkmax = term_l[vit], vk = cur_term_i[vit];
                            if (vkmax - vk == kmax - k &&
                                equal(term_sorted.data() + vitt + vk,
                                      term_sorted.data() + vitt + vkmax,
                                      term_sorted.data() + itt + k))
                                break;
                        }
                        if (iq == (int)vq.size())
                            vq.push_back(make_pair(it, ir = nmr++));
                        else
                            ir = vq[iq].second;
                    } else
                        mpr[hr].push_back(make_pair(it, ir = nmr++));
                    mats[iq].push_back(make_pair(make_pair(il, ir), itv));
                }
            }
        }
    }
};

} // namespace block2