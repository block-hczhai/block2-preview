
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

#include "../core/integral.hpp"
#include "mps.hpp"
#include "mps_unfused.hpp"
#include <algorithm>
#include <array>
#include <set>
#include <stack>
#include <tuple>
#include <vector>

using namespace std;

namespace block2 {

// Prefix trie structure
// can be used as map<DET, FL>
// memory complexity:
//    (n_dets << 4^n_sites) : (4 * n_sites + 1) * n_dets * sizeof(int)
//    (n_dets  ~ 4^n_sites) : (19 / 3) * n_dets * sizeof(int)
// time complexity: (D = MPS bond dimension)
//    (n_dets << 4^n_sites) : n_sites * n_dets * D * D
//    (n_dets  ~ 4^n_sites) : (4 / 3) * n_dets * D * D
template <typename D, typename FL, uint8_t L = 4, typename IT = uint32_t>
struct TRIE {
    typedef typename GMatrix<FL>::FP FP;
    typedef IT XIT;
    vector<array<IT, L>> data;
    vector<IT> dets, invs;
    vector<FL> vals;
    int n_sites;
    bool enable_look_up;
    TRIE(int n_sites, bool enable_look_up = false)
        : n_sites(n_sites), enable_look_up(enable_look_up) {
        data.reserve(n_sites + 1);
        data.push_back(array<IT, L>());
    }
    // empty trie
    void clear() {
        data.clear(), dets.clear(), invs.clear(), vals.clear();
        data.push_back(array<IT, L>());
    }
    // deep copy
    shared_ptr<D> copy() {
        shared_ptr<D> dett = make_shared<D>(n_sites, enable_look_up);
        dett->data = vector<array<IT, L>>(data.begin(), data.end());
        dett->dets = vector<IT>(dets.begin(), dets.end());
        dett->invs = vector<IT>(invs.begin(), invs.end());
        dett->vals = vector<FL>(vals.begin(), vals.end());
        return dett;
    }
    // number of determinants
    size_t size() const noexcept { return dets.size(); }
    // add a determinant to trie
    void push_back(const vector<uint8_t> &det) {
        assert((int)det.size() == n_sites);
        IT cur = 0;
        for (int i = 0; i < n_sites; i++) {
            uint8_t j = det[i];
            if (data[cur][j] == 0) {
                assert(data.size() <= (size_t)numeric_limits<IT>::max());
                data[cur][j] = (IT)data.size();
                data.push_back(array<IT, L>());
            }
            cur = data[cur][j];
        }
        // cannot push_back repeated determinants
        assert(dets.size() == 0 || cur > dets.back());
        dets.push_back(cur);
        if (enable_look_up) {
            invs.resize(data.size());
            for (int i = 0, cur = 0; i < n_sites; i++) {
                uint8_t j = det[i];
                invs[data[cur][j]] = cur;
                cur = data[cur][j];
            }
        }
    }
    void sort_dets() {
        vector<int> gidx(dets.size());
        for (int i = 0; i < (int)gidx.size(); i++)
            gidx[i] = i;
        sort(gidx.begin(), gidx.end(),
             [this](int i, int j) { return this->dets[i] < this->dets[j]; });
        vector<IT> ndets = dets;
        vector<FL> nvals = vals;
        for (int i = 0; i < (int)gidx.size(); i++) {
            dets[i] = ndets[gidx[i]];
            vals[i] = nvals[gidx[i]];
        }
    }
    // find the index of a determinant
    // dets must be sorted
    int find(const vector<uint8_t> &det) {
        assert((int)det.size() == n_sites);
        IT cur = 0;
        for (int i = 0; i < n_sites; i++) {
            uint8_t j = det[i];
            if (data[cur][j] == 0)
                return -1;
            cur = data[cur][j];
        }
        int idx =
            (int)(lower_bound(dets.begin(), dets.end(), cur) - dets.begin());
        return idx < (int)dets.size() && dets[idx] == cur ? idx : -1;
    }
    // get a determinant in trie
    vector<uint8_t> operator[](int idx) const {
        assert(enable_look_up && idx < dets.size());
        vector<uint8_t> r(n_sites, 0);
        IT cur = dets[idx], ir;
        for (int i = n_sites - 1; i >= 0; i--, cur = ir) {
            ir = invs[cur];
            for (uint8_t j = 0; j < (uint8_t)data[ir].size(); j++)
                if (data[ir][j] == cur) {
                    r[i] = j;
                    break;
                }
        }
        return r;
    }
    vector<FP> get_state_occupation() const {
        int ntg = threading->activate_global();
        vector<vector<FP>> pop(ntg);
#pragma omp parallel num_threads(ntg)
        {
            int tid = threading->get_thread_id();
            pop[tid].resize(n_sites * L);
            vector<FP> &ipop = pop[tid];
#pragma omp for schedule(static)
            for (int i = 0; i < dets.size(); i++) {
                FP vsq = xreal<FL>(xconj<FL>(vals[i]) * vals[i]);
                vector<uint8_t> det = (*this)[i];
                for (int j = 0; j < n_sites; j++)
                    ipop[j * L + det[j]] += vsq;
            }
        }
        vector<FP> rpop(n_sites * L, 0);
        for (int itg = 0; itg < ntg; itg++)
            for (int k = 0; k < n_sites * L; k++)
                rpop[k] += pop[itg][k];
        return rpop;
    }
};

template <typename, typename, typename = void> struct DeterminantTRIE;

// Prefix trie structure of determinants (non-spin-adapted)
// det[i] = 0 (empty) 1 (alpha) 2 (beta) 3 (doubly)
template <typename S, typename FL>
struct DeterminantTRIE<S, FL, typename S::is_sz_t>
    : TRIE<DeterminantTRIE<S, FL>, FL> {
    typedef typename TRIE<DeterminantTRIE<S, FL>, FL>::XIT IT;
    typedef typename GMatrix<FL>::FP FP;
    using TRIE<DeterminantTRIE<S, FL>, FL>::data;
    using TRIE<DeterminantTRIE<S, FL>, FL>::dets;
    using TRIE<DeterminantTRIE<S, FL>, FL>::vals;
    using TRIE<DeterminantTRIE<S, FL>, FL>::invs;
    using TRIE<DeterminantTRIE<S, FL>, FL>::n_sites;
    using TRIE<DeterminantTRIE<S, FL>, FL>::enable_look_up;
    using TRIE<DeterminantTRIE<S, FL>, FL>::sort_dets;
    DeterminantTRIE(int n_sites, bool enable_look_up = false)
        : TRIE<DeterminantTRIE<S, FL>, FL>(n_sites, enable_look_up) {}
    shared_ptr<UnfusedMPS<S, FL>> construct_mps(
        const shared_ptr<MPSInfo<S>> &info,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) const {
        shared_ptr<UnfusedMPS<S, FL>> r = make_shared<UnfusedMPS<S, FL>>();
        r->info = info;
        r->canonical_form = string(n_sites - 1, 'L') + "K";
        r->center = n_sites - 1;
        r->n_sites = n_sites;
        r->dot = 1;
        r->tensors.resize(n_sites);
        S vacuum = info->left_dims_fci[0]->quanta[0];
        vector<pair<S, IT>> cur_nodes =
            vector<pair<S, IT>>{make_pair(vacuum, 0)};
        info->left_dims[0] =
            make_shared<StateInfo<S>>(info->left_dims_fci[0]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            info->left_dims[i + 1] = make_shared<StateInfo<S>>(
                info->left_dims_fci[i + 1]->deep_copy());
        info->right_dims[n_sites] = make_shared<StateInfo<S>>(
            info->right_dims_fci[n_sites]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            info->right_dims[i] =
                make_shared<StateInfo<S>>(info->right_dims_fci[i]->deep_copy());
        for (int k = 0; k < n_sites; k++) {
            vector<uint8_t> basis_iqs(4);
            for (uint8_t j = 0; j < info->basis[k]->n; j++)
                if (info->basis[k]->quanta[j].n() == 0)
                    basis_iqs[0] = j;
                else if (info->basis[k]->quanta[j].n() == 2)
                    basis_iqs[3] = j;
                else if (info->basis[k]->quanta[j].twos() == 1)
                    basis_iqs[1] = j;
                else if (info->basis[k]->quanta[j].twos() == -1)
                    basis_iqs[2] = j;
                else
                    assert(false);
            shared_ptr<SparseTensor<S, FL>> t =
                make_shared<SparseTensor<S, FL>>();
            vector<pair<S, IT>> next_nodes;
            map<S, MKL_INT> lsh, rsh;
            vector<map<pair<S, S>, vector<pair<MKL_INT, MKL_INT>>>> blocks(
                info->basis[k]->n);
            vector<map<pair<S, S>, vector<FL>>> coeffs(info->basis[k]->n);
            // determine shape
            for (const auto &irx : cur_nodes) {
                IT ir = irx.second;
                S pq = irx.first;
                for (uint8_t j = 0; j < (uint8_t)data[ir].size(); j++)
                    if (data[ir][j] != 0) {
                        S nq = pq + info->basis[k]->quanta[basis_iqs[j]];
                        next_nodes.push_back(make_pair(nq, data[ir][j]));
                        blocks[basis_iqs[j]][make_pair(pq, nq)].push_back(
                            make_pair(lsh[pq], rsh[nq]));
                        if (k == n_sites - 1) {
                            int idx =
                                (int)(lower_bound(dets.begin(), dets.end(),
                                                  data[ir][j]) -
                                      dets.begin());
                            assert(idx < (int)dets.size() &&
                                   dets[idx] == data[ir][j]);
                            coeffs[basis_iqs[j]][make_pair(pq, nq)].push_back(
                                vals[idx]);
                        } else
                            rsh[nq]++;
                    }
                lsh[pq]++;
            }
            if (k == n_sites - 1) {
                assert(rsh.size() == 1);
                rsh.begin()->second = 1;
            }
            // create tensor
            t->data.resize(blocks.size());
            for (uint8_t j = 0; j < (uint8_t)blocks.size(); j++)
                for (const auto &mp : blocks[j]) {
                    shared_ptr<GTensor<FL>> gt = make_shared<GTensor<FL>>(
                        lsh.at(mp.first.first), 1, rsh.at(mp.first.second));
                    t->data[j].push_back(make_pair(
                        make_pair(mp.first.first, mp.first.second), gt));
                    if (k == n_sites - 1)
                        for (size_t im = 0; im < mp.second.size(); im++)
                            (*gt)({mp.second[im].first, 0,
                                   mp.second[im].second}) =
                                coeffs[j].at(mp.first)[im];
                    else
                        for (const auto &mx : mp.second)
                            (*gt)({mx.first, 0, mx.second}) = (FL)(FP)1.0;
                }
            cur_nodes = next_nodes;
            // put shape in bond dims
            total_bond_t new_total = 0;
            for (int p = 0; p < info->left_dims[k + 1]->n; p++) {
                if (rsh.count(info->left_dims[k + 1]->quanta[p])) {
                    info->left_dims[k + 1]->n_states[p] =
                        (ubond_t)rsh.at(info->left_dims[k + 1]->quanta[p]);
                    new_total += info->left_dims[k + 1]->n_states[p];
                } else
                    info->left_dims[k + 1]->n_states[p] = 0;
            }
            info->left_dims[k + 1]->n_states_total = new_total;
            r->tensors[k] = t;
        }
        for (int i = n_sites - 1; i >= 0; i--)
            if (info->right_dims[i]->n_states_total >
                (total_bond_t)dets.size()) {
                total_bond_t new_total = 0;
                for (int k = 0; k < info->right_dims[i]->n; k++) {
                    uint64_t new_n_states =
                        (uint64_t)(ceil((double)info->right_dims[i]
                                            ->n_states[k] *
                                        dets.size() /
                                        info->right_dims[i]->n_states_total) +
                                   0.1);
                    assert(new_n_states != 0);
                    info->right_dims[i]->n_states[k] =
                        (ubond_t)min((uint64_t)new_n_states,
                                     (uint64_t)numeric_limits<ubond_t>::max());
                    new_total += info->right_dims[i]->n_states[k];
                }
                info->right_dims[i]->n_states_total = new_total;
            }
        for (int i = 0; i <= n_sites; i++)
            StateInfo<S>::filter(*info->right_dims[i], *info->left_dims[i],
                                 info->target);
        for (int i = 0; i <= n_sites; i++)
            info->left_dims[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            info->right_dims[i]->collect();
        info->check_bond_dimensions();
        if (para_rule == nullptr || para_rule->is_root())
            info->save_mutable();
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        return r;
    }
    // set the value for each determinant to the overlap between mps
    void evaluate(const shared_ptr<UnfusedMPS<S, FL>> &mps, FP cutoff = 0,
                  int max_rank = -1, const vector<uint8_t> &ref = {}) {
        assert(ref.size() == n_sites || ref.size() == 0);
        if (max_rank < 0)
            max_rank = mps->info->target.n();
        vals.resize(dets.size());
        memset(vals.data(), 0, sizeof(FL) * vals.size());
        bool has_dets = dets.size() != 0;
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<tuple<int, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>, int, int>>
            ptrs;

        vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos(n_sites + 1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(*mps->info->left_dims_fci[0],
                                 *mps->info->left_dims_fci[0],
                                 mps->info->vacuum, false);
        for (int d = 0; d < n_sites; d++) {
            pinfos[d + 1].resize(4);
            for (int j = 0; j < (int)pinfos[d + 1].size(); j++) {
                map<S, MKL_INT> qkets;
                for (auto &m : mps->tensors[d]->data[j]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (!qkets.count(ket))
                        qkets[ket] = m.second->shape[2];
                }
                StateInfo<S> ibra, iket;
                ibra.allocate((int)qkets.size());
                iket.allocate((int)qkets.size());
                int k = 0;
                for (auto &qm : qkets) {
                    ibra.quanta[k] = iket.quanta[k] = qm.first;
                    ibra.n_states[k] = 1;
                    iket.n_states[k] = (ubond_t)qm.second;
                    k++;
                }
                pinfos[d + 1][j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                pinfos[d + 1][j]->initialize(ibra, iket, mps->info->vacuum,
                                             false);
            }
        }
        if (!has_dets) {
            for (uint8_t j = 0; j < (uint8_t)data[0].size(); j++)
                if (data[0][j] == 0) {
                    assert(data.size() <= (size_t)numeric_limits<IT>::max());
                    data[0][j] = (IT)data.size();
                    data.push_back(array<IT, 4>{0, 0, 0, 0});
                }
        }
        shared_ptr<SparseMatrix<S, FL>> zmat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        zmat->allocate(pinfos[0][0]);
        for (size_t j = 0; j < zmat->total_memory; j++)
            zmat->data[j] = 1.0;
        vector<uint8_t> zdet(n_sites);
        for (uint8_t j = 0; j < (uint8_t)data[0].size(); j++)
            if (data[0][j] != 0)
                ptrs.push_back(make_tuple(data[0][j], j, 0, zmat, zdet, 0, 0));

        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>, int, int>>
            pptrs;
        int ntg = threading->activate_global();
        int ngroup = ntg * 4;
        vector<shared_ptr<SparseMatrix<S, FL>>> ccmp(ngroup);
#pragma omp parallel num_threads(ntg)
        // depth-first traverse of trie
        while (!ptrs.empty()) {
#pragma omp master
            check_signal_()();
            int pstart = max(0, (int)ptrs.size() - ngroup);
#pragma omp for schedule(static)
            for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                shared_ptr<VectorAllocator<FP>> pd_alloc =
                    make_shared<VectorAllocator<FP>>();
                auto &p = ptrs[ip];
                int j = get<1>(p), d = get<2>(p), nh = get<5>(p),
                    np = get<6>(p);
                shared_ptr<SparseMatrix<S, FL>> pmp = get<3>(p);
                shared_ptr<SparseMatrix<S, FL>> cmp =
                    make_shared<SparseMatrix<S, FL>>(pd_alloc);
                cmp->allocate(pinfos[d + 1][j]);
                for (auto &m : mps->tensors[d]->data[j]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                                   m.second->ref(), false,
                                                   (*cmp)[ket], 1.0, 1.0);
                }
                if (cmp->info->n == 0 || (cutoff != 0 && cmp->norm() < cutoff))
                    ccmp[ip - pstart] = nullptr;
                else if (ref.size() != 0 && (nh > max_rank || np > max_rank))
                    ccmp[ip - pstart] = nullptr;
                else
                    ccmp[ip - pstart] = cmp;
            }
#pragma omp single
            {
                for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                    if (ccmp[ip - pstart] == nullptr)
                        continue;
                    auto &p = ptrs[ip];
                    IT cur = get<0>(p);
                    int j = get<1>(p), d = get<2>(p);
                    int nh = get<5>(p), np = get<6>(p);
                    vector<uint8_t> det = get<4>(p);
                    det[d] = j;
                    shared_ptr<SparseMatrix<S, FL>> cmp = ccmp[ip - pstart];
                    if (d == n_sites - 1) {
                        assert(cmp->total_memory == 1 &&
                               cmp->info->find_state(mps->info->target) == 0);
                        if (!has_dets) {
                            dets.push_back(cur);
                            vals.push_back(cmp->data[0]);
                            if (enable_look_up) {
                                invs.resize(data.size());
                                IT curx = 0;
                                for (int i = 0; i < n_sites; i++) {
                                    uint8_t jj = det[i];
                                    invs[data[curx][jj]] = curx;
                                    curx = data[curx][jj];
                                }
                            }
                        } else
                            vals[lower_bound(dets.begin(), dets.end(), cur) -
                                 dets.begin()] = cmp->data[0];
                    } else {
                        if (!has_dets) {
                            for (uint8_t jj = 0; jj < (uint8_t)data[cur].size();
                                 jj++)
                                if (data[cur][jj] == 0) {
                                    assert(data.size() <=
                                           (size_t)numeric_limits<IT>::max());
                                    data[cur][jj] = (IT)data.size();
                                    data.push_back(array<IT, 4>{0, 0, 0, 0});
                                }
                        }
                        for (uint8_t jj = 0; jj < (uint8_t)data[cur].size();
                             jj++)
                            if (data[cur][jj] != 0) {
                                int nh_n = nh;
                                int np_n = np;
                                if (ref.size() != 0) {
                                    if (!(j & 1) && (ref[d] & 1))
                                        nh_n += 1;
                                    if (!(j & 2) && (ref[d] & 2))
                                        nh_n += 1;
                                    if ((j & 1) && !(ref[d] & 1))
                                        np_n += 1;
                                    if ((j & 2) && !(ref[d] & 2))
                                        np_n += 1;
                                }
                                pptrs.push_back(make_tuple(data[cur][jj], jj,
                                                           d + 1, cmp, det,
                                                           nh_n, np_n));
                            }
                    }
                    ccmp[ip - pstart] = nullptr;
                }
                ptrs.resize(pstart);
                ptrs.insert(ptrs.end(), pptrs.begin(), pptrs.end());
                pptrs.clear();
            }
        }
        pinfos.clear();
        sort_dets();
        threading->activate_normal();
    }
    // phase of moving all alpha elec before beta elec
    int phase_change_spin_order(const vector<uint8_t> &det) {
        uint8_t n = 0, j = 0;
        for (int i = 0; i < det.size(); j ^= det[i++])
            n ^= (det[i] << 1) & j;
        return 1 - (n & 2);
    }
    // return 1 if number of even cycles is odd
    uint8_t permutation_parity(const vector<int> &perm) {
        uint8_t n = 0;
        vector<uint8_t> tag(perm.size(), 0);
        for (int i = 0, j; i < perm.size(); i++) {
            j = i, n ^= !tag[j];
            while (!tag[j])
                n ^= 1, tag[j] = 1, j = perm[j];
        }
        return n;
    }
    int phase_change_orb_order(const vector<uint8_t> &det,
                               const vector<int> &reorder) {
        vector<int> alpha, beta, rev_det, reva, revb;
        alpha.reserve(det.size());
        beta.reserve(det.size());
        rev_det.reserve(det.size());
        reva.reserve(det.size());
        revb.reserve(det.size());
        for (int i = 0; i < det.size(); i++)
            rev_det[reorder[i]] = det[i];
        for (int i = 0, ja = 0, jb = 0; i < det.size(); i++) {
            reva[i] = ja, ja += (rev_det[i] & 1);
            revb[i] = jb, jb += ((rev_det[i] & 2) >> 1);
        }
        for (int i = 0; i < det.size(); i++) {
            if (det[i] & 1)
                alpha.push_back(reva[reorder[i]]);
            if (det[i] & 2)
                beta.push_back(revb[reorder[i]]);
        }
        uint8_t n = permutation_parity(alpha);
        n ^= permutation_parity(beta);
        return 1 - (n << 1);
    }
    void convert_phase(const vector<int> &reorder) {
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int i = 0; i < vals.size(); i++) {
            vector<uint8_t> det = (*this)[i];
            vals[i] *= phase_change_spin_order(det);
            if (reorder.size() != 0)
                vals[i] *= phase_change_orb_order(det, reorder);
        }
        threading->activate_normal();
    }
};

// Prefix trie structure of Configuration State Functions (CSFs) (spin-adapted)
// det[i] = 0 (empty) 1 (increase) 2 (decrease) 3 (doubly)
template <typename S, typename FL>
struct DeterminantTRIE<S, FL, typename S::is_su2_t>
    : TRIE<DeterminantTRIE<S, FL>, FL> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename TRIE<DeterminantTRIE<S, FL>, FL>::XIT IT;
    using TRIE<DeterminantTRIE<S, FL>, FL>::data;
    using TRIE<DeterminantTRIE<S, FL>, FL>::dets;
    using TRIE<DeterminantTRIE<S, FL>, FL>::vals;
    using TRIE<DeterminantTRIE<S, FL>, FL>::invs;
    using TRIE<DeterminantTRIE<S, FL>, FL>::n_sites;
    using TRIE<DeterminantTRIE<S, FL>, FL>::enable_look_up;
    using TRIE<DeterminantTRIE<S, FL>, FL>::sort_dets;
    DeterminantTRIE(int n_sites, bool enable_look_up = false)
        : TRIE<DeterminantTRIE<S, FL>, FL>(n_sites, enable_look_up) {}
    shared_ptr<UnfusedMPS<S, FL>> construct_mps(
        const shared_ptr<MPSInfo<S>> &info,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) const {
        shared_ptr<UnfusedMPS<S, FL>> r = make_shared<UnfusedMPS<S, FL>>();
        r->info = info;
        r->canonical_form = string(n_sites - 1, 'L') + "K";
        r->center = n_sites - 1;
        r->n_sites = n_sites;
        r->dot = 1;
        r->tensors.resize(n_sites);
        S vacuum = info->left_dims_fci[0]->quanta[0];
        vector<pair<S, IT>> cur_nodes =
            vector<pair<S, IT>>{make_pair(vacuum, 0)};
        info->left_dims[0] =
            make_shared<StateInfo<S>>(info->left_dims_fci[0]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            info->left_dims[i + 1] = make_shared<StateInfo<S>>(
                info->left_dims_fci[i + 1]->deep_copy());
        info->right_dims[n_sites] = make_shared<StateInfo<S>>(
            info->right_dims_fci[n_sites]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            info->right_dims[i] =
                make_shared<StateInfo<S>>(info->right_dims_fci[i]->deep_copy());
        for (int k = 0; k < n_sites; k++) {
            vector<uint8_t> basis_iqs(4);
            for (uint8_t j = 0; j < info->basis[k]->n; j++)
                if (info->basis[k]->quanta[j].n() == 0)
                    basis_iqs[0] = j;
                else if (info->basis[k]->quanta[j].n() == 2)
                    basis_iqs[3] = j;
                else if (info->basis[k]->quanta[j].twos() == 1)
                    basis_iqs[1] = basis_iqs[2] = j;
                else
                    assert(false);
            shared_ptr<SparseTensor<S, FL>> t =
                make_shared<SparseTensor<S, FL>>();
            vector<pair<S, IT>> next_nodes;
            map<S, MKL_INT> lsh, rsh;
            vector<map<pair<S, S>, vector<pair<MKL_INT, MKL_INT>>>> blocks(
                info->basis[k]->n);
            vector<map<pair<S, S>, vector<FL>>> coeffs(info->basis[k]->n);
            // determine shape
            for (const auto &irx : cur_nodes) {
                IT ir = irx.second;
                S pq = irx.first;
                for (uint8_t j = 0; j < (uint8_t)data[ir].size(); j++)
                    if (data[ir][j] != 0) {
                        S nq = pq + info->basis[k]->quanta[basis_iqs[j]];
                        if (nq.count() > 1)
                            nq = nq[j == 1];
                        next_nodes.push_back(make_pair(nq, data[ir][j]));
                        blocks[basis_iqs[j]][make_pair(pq, nq)].push_back(
                            make_pair(lsh[pq], rsh[nq]));
                        if (k == n_sites - 1) {
                            int idx =
                                (int)(lower_bound(dets.begin(), dets.end(),
                                                  data[ir][j]) -
                                      dets.begin());
                            assert(idx < (int)dets.size() &&
                                   dets[idx] == data[ir][j]);
                            coeffs[basis_iqs[j]][make_pair(pq, nq)].push_back(
                                vals[idx]);
                        } else
                            rsh[nq]++;
                    }
                lsh[pq]++;
            }
            if (k == n_sites - 1) {
                assert(rsh.size() == 1);
                rsh.begin()->second = 1;
            }
            // create tensor
            t->data.resize(blocks.size());
            for (uint8_t j = 0; j < (uint8_t)blocks.size(); j++)
                for (const auto &mp : blocks[j]) {
                    shared_ptr<GTensor<FL>> gt = make_shared<GTensor<FL>>(
                        lsh.at(mp.first.first), 1, rsh.at(mp.first.second));
                    t->data[j].push_back(make_pair(
                        make_pair(mp.first.first, mp.first.second), gt));
                    if (k == n_sites - 1)
                        for (size_t im = 0; im < mp.second.size(); im++)
                            (*gt)({mp.second[im].first, 0,
                                   mp.second[im].second}) =
                                coeffs[j].at(mp.first)[im];
                    else
                        for (const auto &mx : mp.second)
                            (*gt)({mx.first, 0, mx.second}) = (FL)(FP)1.0;
                }
            cur_nodes = next_nodes;
            // put shape in bond dims
            total_bond_t new_total = 0;
            for (int p = 0; p < info->left_dims[k + 1]->n; p++) {
                if (rsh.count(info->left_dims[k + 1]->quanta[p])) {
                    info->left_dims[k + 1]->n_states[p] =
                        (ubond_t)rsh.at(info->left_dims[k + 1]->quanta[p]);
                    new_total += info->left_dims[k + 1]->n_states[p];
                } else
                    info->left_dims[k + 1]->n_states[p] = 0;
            }
            info->left_dims[k + 1]->n_states_total = new_total;
            r->tensors[k] = t;
        }
        for (int i = n_sites - 1; i >= 0; i--)
            if (info->right_dims[i]->n_states_total >
                (total_bond_t)dets.size()) {
                total_bond_t new_total = 0;
                for (int k = 0; k < info->right_dims[i]->n; k++) {
                    uint64_t new_n_states =
                        (uint64_t)(ceil((double)info->right_dims[i]
                                            ->n_states[k] *
                                        dets.size() /
                                        info->right_dims[i]->n_states_total) +
                                   0.1);
                    assert(new_n_states != 0);
                    info->right_dims[i]->n_states[k] =
                        (ubond_t)min((uint64_t)new_n_states,
                                     (uint64_t)numeric_limits<ubond_t>::max());
                    new_total += info->right_dims[i]->n_states[k];
                }
                info->right_dims[i]->n_states_total = new_total;
            }
        for (int i = 0; i <= n_sites; i++)
            StateInfo<S>::filter(*info->right_dims[i], *info->left_dims[i],
                                 info->target);
        for (int i = 0; i <= n_sites; i++)
            info->left_dims[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            info->right_dims[i]->collect();
        info->check_bond_dimensions();
        if (para_rule == nullptr || para_rule->is_root())
            info->save_mutable();
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        return r;
    }
    // set the value for each CSF to the overlap between mps
    void evaluate(const shared_ptr<UnfusedMPS<S, FL>> &mps, FP cutoff = 0,
                  int max_rank = -1, const vector<uint8_t> &ref = {}) {
        if (max_rank < 0)
            max_rank = mps->info->target.n();
        vals.resize(dets.size());
        memset(vals.data(), 0, sizeof(FL) * vals.size());
        bool has_dets = dets.size() != 0;
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>>>
            ptrs;
        vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos(n_sites + 1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(*mps->info->left_dims_fci[0],
                                 *mps->info->left_dims_fci[0],
                                 mps->info->vacuum, false);
        for (int d = 0; d < n_sites; d++) {
            pinfos[d + 1].resize(4);
            for (int j = 0; j < pinfos[d + 1].size(); j++) {
                int jd = j >= 2 ? j - 1 : j;
                map<S, MKL_INT> qkets;
                for (auto &m : mps->tensors[d]->data[jd]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (jd == 1 && !((j == 1 && ket.twos() > bra.twos()) ||
                                     (j == 2 && ket.twos() < bra.twos())))
                        continue;
                    if (!qkets.count(ket))
                        qkets[ket] = m.second->shape[2];
                }
                StateInfo<S> ibra, iket;
                ibra.allocate((int)qkets.size());
                iket.allocate((int)qkets.size());
                int k = 0;
                for (auto &qm : qkets) {
                    ibra.quanta[k] = iket.quanta[k] = qm.first;
                    ibra.n_states[k] = 1;
                    iket.n_states[k] = (ubond_t)qm.second;
                    k++;
                }
                pinfos[d + 1][j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                pinfos[d + 1][j]->initialize(ibra, iket, mps->info->vacuum,
                                             false);
            }
        }
        if (!has_dets) {
            for (uint8_t j = 0; j < (uint8_t)data[0].size(); j++)
                if (data[0][j] == 0) {
                    assert(data.size() <= (size_t)numeric_limits<IT>::max());
                    data[0][j] = (IT)data.size();
                    data.push_back(array<IT, 4>{0, 0, 0, 0});
                }
        }
        shared_ptr<SparseMatrix<S, FL>> zmat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        zmat->allocate(pinfos[0][0]);
        for (size_t j = 0; j < zmat->total_memory; j++)
            zmat->data[j] = 1.0;
        vector<uint8_t> zdet(n_sites);
        for (uint8_t j = 0; j < (uint8_t)data[0].size(); j++)
            if (data[0][j] != 0)
                ptrs.push_back(make_tuple(data[0][j], j, 0, zmat, zdet));
        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>>>
            pptrs;
        int ntg = threading->activate_global();
        int ngroup = ntg * 4;
        vector<shared_ptr<SparseMatrix<S, FL>>> ccmp(ngroup);
#pragma omp parallel num_threads(ntg)
        // depth-first traverse of trie
        while (!ptrs.empty()) {
#pragma omp master
            check_signal_()();
            int pstart = max(0, (int)ptrs.size() - ngroup);
#pragma omp for schedule(static)
            for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                shared_ptr<VectorAllocator<FP>> pd_alloc =
                    make_shared<VectorAllocator<FP>>();
                auto &p = ptrs[ip];
                int j = get<1>(p), d = get<2>(p);
                shared_ptr<SparseMatrix<S, FL>> pmp = get<3>(p);
                shared_ptr<SparseMatrix<S, FL>> cmp =
                    make_shared<SparseMatrix<S, FL>>(pd_alloc);
                cmp->allocate(pinfos[d + 1][j]);
                int jd = j >= 2 ? j - 1 : j;
                for (auto &m : mps->tensors[d]->data[jd]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (jd == 1 && !((j == 1 && ket.twos() > bra.twos()) ||
                                     (j == 2 && ket.twos() < bra.twos())))
                        continue;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                                   m.second->ref(), false,
                                                   (*cmp)[ket], 1.0, 1.0);
                }
                if (cmp->info->n == 0 || (cutoff != 0 && cmp->norm() < cutoff))
                    ccmp[ip - pstart] = nullptr;
                else
                    ccmp[ip - pstart] = cmp;
            }
#pragma omp single
            {
                for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                    if (ccmp[ip - pstart] == nullptr)
                        continue;
                    auto &p = ptrs[ip];
                    IT cur = get<0>(p);
                    int j = get<1>(p), d = get<2>(p);
                    vector<uint8_t> det = get<4>(p);
                    det[d] = j;
                    shared_ptr<SparseMatrix<S, FL>> cmp = ccmp[ip - pstart];
                    if (d == n_sites - 1) {
                        assert(cmp->total_memory == 1 &&
                               cmp->info->find_state(mps->info->target) == 0);
                        if (!has_dets) {
                            dets.push_back(cur);
                            vals.push_back(cmp->data[0]);
                            if (enable_look_up) {
                                invs.resize(data.size());
                                IT curx = 0;
                                for (int i = 0; i < n_sites; i++) {
                                    uint8_t jj = det[i];
                                    invs[data[curx][jj]] = curx;
                                    curx = data[curx][jj];
                                }
                            }
                        } else
                            vals[lower_bound(dets.begin(), dets.end(), cur) -
                                 dets.begin()] = cmp->data[0];
                    } else {
                        if (!has_dets) {
                            for (uint8_t jj = 0; jj < (uint8_t)data[cur].size();
                                 jj++)
                                if (data[cur][jj] == 0) {
                                    assert(data.size() <=
                                           (size_t)numeric_limits<IT>::max());
                                    data[cur][jj] = (IT)data.size();
                                    data.push_back(array<IT, 4>{0, 0, 0, 0});
                                }
                        }
                        for (uint8_t jj = 0; jj < (uint8_t)data[cur].size();
                             jj++)
                            if (data[cur][jj] != 0)
                                pptrs.push_back(make_tuple(data[cur][jj], jj,
                                                           d + 1, cmp, det));
                    }
                    ccmp[ip - pstart] = nullptr;
                }
                ptrs.resize(pstart);
                ptrs.insert(ptrs.end(), pptrs.begin(), pptrs.end());
                pptrs.clear();
            }
        }
        pinfos.clear();
        sort_dets();
        threading->activate_normal();
    }
    void convert_phase(const vector<int> &reorder) {}
};

// Prefix trie structure of determinants (general spin)
// det[i] = 0 (empty) 1 (occ)
template <typename S, typename FL>
struct DeterminantTRIE<S, FL, typename S::is_sg_t>
    : TRIE<DeterminantTRIE<S, FL>, FL, 2> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename TRIE<DeterminantTRIE<S, FL>, FL, 2>::XIT IT;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::data;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::dets;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::vals;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::invs;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::n_sites;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::enable_look_up;
    using TRIE<DeterminantTRIE<S, FL>, FL, 2>::sort_dets;
    DeterminantTRIE(int n_sites, bool enable_look_up = false)
        : TRIE<DeterminantTRIE<S, FL>, FL, 2>(n_sites, enable_look_up) {}
    shared_ptr<UnfusedMPS<S, FL>> construct_mps(
        const shared_ptr<MPSInfo<S>> &info,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) const {
        shared_ptr<UnfusedMPS<S, FL>> r = make_shared<UnfusedMPS<S, FL>>();
        r->info = info;
        r->canonical_form = string(n_sites - 1, 'L') + "K";
        r->center = n_sites - 1;
        r->n_sites = n_sites;
        r->dot = 1;
        r->tensors.resize(n_sites);
        S vacuum = info->left_dims_fci[0]->quanta[0];
        vector<pair<S, IT>> cur_nodes =
            vector<pair<S, IT>>{make_pair(vacuum, 0)};
        info->left_dims[0] =
            make_shared<StateInfo<S>>(info->left_dims_fci[0]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            info->left_dims[i + 1] = make_shared<StateInfo<S>>(
                info->left_dims_fci[i + 1]->deep_copy());
        info->right_dims[n_sites] = make_shared<StateInfo<S>>(
            info->right_dims_fci[n_sites]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            info->right_dims[i] =
                make_shared<StateInfo<S>>(info->right_dims_fci[i]->deep_copy());
        for (int k = 0; k < n_sites; k++) {
            vector<uint8_t> basis_iqs(2);
            for (uint8_t j = 0; j < info->basis[k]->n; j++)
                if (info->basis[k]->quanta[j].n() == 0)
                    basis_iqs[0] = j;
                else if (info->basis[k]->quanta[j].n() == 1)
                    basis_iqs[1] = j;
                else
                    assert(false);
            shared_ptr<SparseTensor<S, FL>> t =
                make_shared<SparseTensor<S, FL>>();
            vector<pair<S, IT>> next_nodes;
            map<S, MKL_INT> lsh, rsh;
            vector<map<pair<S, S>, vector<pair<MKL_INT, MKL_INT>>>> blocks(
                info->basis[k]->n);
            vector<map<pair<S, S>, vector<FL>>> coeffs(info->basis[k]->n);
            // determine shape
            for (const auto &irx : cur_nodes) {
                IT ir = irx.second;
                S pq = irx.first;
                for (uint8_t j = 0; j < (uint8_t)data[ir].size(); j++)
                    if (data[ir][j] != 0) {
                        S nq = pq + info->basis[k]->quanta[basis_iqs[j]];
                        next_nodes.push_back(make_pair(nq, data[ir][j]));
                        blocks[basis_iqs[j]][make_pair(pq, nq)].push_back(
                            make_pair(lsh[pq], rsh[nq]));
                        if (k == n_sites - 1) {
                            int idx =
                                (int)(lower_bound(dets.begin(), dets.end(),
                                                  data[ir][j]) -
                                      dets.begin());
                            assert(idx < (int)dets.size() &&
                                   dets[idx] == data[ir][j]);
                            coeffs[basis_iqs[j]][make_pair(pq, nq)].push_back(
                                vals[idx]);
                        } else
                            rsh[nq]++;
                    }
                lsh[pq]++;
            }
            if (k == n_sites - 1) {
                assert(rsh.size() == 1);
                rsh.begin()->second = 1;
            }
            // create tensor
            t->data.resize(blocks.size());
            for (uint8_t j = 0; j < (uint8_t)blocks.size(); j++)
                for (const auto &mp : blocks[j]) {
                    shared_ptr<GTensor<FL>> gt = make_shared<GTensor<FL>>(
                        lsh.at(mp.first.first), 1, rsh.at(mp.first.second));
                    t->data[j].push_back(make_pair(
                        make_pair(mp.first.first, mp.first.second), gt));
                    if (k == n_sites - 1)
                        for (size_t im = 0; im < mp.second.size(); im++)
                            (*gt)({mp.second[im].first, 0,
                                   mp.second[im].second}) =
                                coeffs[j].at(mp.first)[im];
                    else
                        for (const auto &mx : mp.second)
                            (*gt)({mx.first, 0, mx.second}) = (FL)(FP)1.0;
                }
            cur_nodes = next_nodes;
            // put shape in bond dims
            total_bond_t new_total = 0;
            for (int p = 0; p < info->left_dims[k + 1]->n; p++) {
                if (rsh.count(info->left_dims[k + 1]->quanta[p])) {
                    info->left_dims[k + 1]->n_states[p] =
                        (ubond_t)rsh.at(info->left_dims[k + 1]->quanta[p]);
                    new_total += info->left_dims[k + 1]->n_states[p];
                } else
                    info->left_dims[k + 1]->n_states[p] = 0;
            }
            info->left_dims[k + 1]->n_states_total = new_total;
            r->tensors[k] = t;
        }
        for (int i = n_sites - 1; i >= 0; i--)
            if (info->right_dims[i]->n_states_total >
                (total_bond_t)dets.size()) {
                total_bond_t new_total = 0;
                for (int k = 0; k < info->right_dims[i]->n; k++) {
                    uint64_t new_n_states =
                        (uint64_t)(ceil((double)info->right_dims[i]
                                            ->n_states[k] *
                                        dets.size() /
                                        info->right_dims[i]->n_states_total) +
                                   0.1);
                    assert(new_n_states != 0);
                    info->right_dims[i]->n_states[k] =
                        (ubond_t)min((uint64_t)new_n_states,
                                     (uint64_t)numeric_limits<ubond_t>::max());
                    new_total += info->right_dims[i]->n_states[k];
                }
                info->right_dims[i]->n_states_total = new_total;
            }
        for (int i = 0; i <= n_sites; i++)
            StateInfo<S>::filter(*info->right_dims[i], *info->left_dims[i],
                                 info->target);
        for (int i = 0; i <= n_sites; i++)
            info->left_dims[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            info->right_dims[i]->collect();
        info->check_bond_dimensions();
        if (para_rule == nullptr || para_rule->is_root())
            info->save_mutable();
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        return r;
    }
    // set the value for each CSF to the overlap between mps
    void evaluate(const shared_ptr<UnfusedMPS<S, FL>> &mps, FP cutoff = 0,
                  int max_rank = -1, const vector<uint8_t> &ref = {}) {
        assert(ref.size() == n_sites || ref.size() == 0);
        if (max_rank < 0)
            max_rank = mps->info->target.n();
        vals.resize(dets.size());
        memset(vals.data(), 0, sizeof(FL) * vals.size());
        bool has_dets = dets.size() != 0;
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>, int, int>>
            ptrs;
        vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos(n_sites + 1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(*mps->info->left_dims_fci[0],
                                 *mps->info->left_dims_fci[0],
                                 mps->info->vacuum, false);
        for (int d = 0; d < n_sites; d++) {
            pinfos[d + 1].resize(2);
            for (int j = 0; j < pinfos[d + 1].size(); j++) {
                map<S, MKL_INT> qkets;
                for (auto &m : mps->tensors[d]->data[j]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (!qkets.count(ket))
                        qkets[ket] = m.second->shape[2];
                }
                StateInfo<S> ibra, iket;
                ibra.allocate((int)qkets.size());
                iket.allocate((int)qkets.size());
                int k = 0;
                for (auto &qm : qkets) {
                    ibra.quanta[k] = iket.quanta[k] = qm.first;
                    ibra.n_states[k] = 1;
                    iket.n_states[k] = (ubond_t)qm.second;
                    k++;
                }
                pinfos[d + 1][j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                pinfos[d + 1][j]->initialize(ibra, iket, mps->info->vacuum,
                                             false);
            }
        }
        if (!has_dets) {
            for (uint8_t j = 0; j < (uint8_t)data[0].size(); j++)
                if (data[0][j] == 0) {
                    assert(data.size() <= (size_t)numeric_limits<IT>::max());
                    data[0][j] = (IT)data.size();
                    data.push_back(array<IT, 2>{0, 0});
                }
        }
        shared_ptr<SparseMatrix<S, FL>> zmat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        zmat->allocate(pinfos[0][0]);
        for (size_t j = 0; j < zmat->total_memory; j++)
            zmat->data[j] = 1.0;
        vector<uint8_t> zdet(n_sites);
        for (uint8_t j = 0; j < (int)data[0].size(); j++)
            if (data[0][j] != 0)
                ptrs.push_back(make_tuple(data[0][j], j, 0, zmat, zdet, 0, 0));
        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>, int, int>>
            pptrs;
        int ntg = threading->activate_global();
        int ngroup = ntg * 2;
        vector<shared_ptr<SparseMatrix<S, FL>>> ccmp(ngroup);
#pragma omp parallel num_threads(ntg)
        // depth-first traverse of trie
        while (!ptrs.empty()) {
#pragma omp master
            check_signal_()();
            int pstart = max(0, (int)ptrs.size() - ngroup);
#pragma omp for schedule(static)
            for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                shared_ptr<VectorAllocator<FP>> pd_alloc =
                    make_shared<VectorAllocator<FP>>();
                auto &p = ptrs[ip];
                int j = get<1>(p), d = get<2>(p), nh = get<5>(p),
                    np = get<6>(p);
                shared_ptr<SparseMatrix<S, FL>> pmp = get<3>(p);
                shared_ptr<SparseMatrix<S, FL>> cmp =
                    make_shared<SparseMatrix<S, FL>>(pd_alloc);
                cmp->allocate(pinfos[d + 1][j]);
                for (auto &m : mps->tensors[d]->data[j]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                                   m.second->ref(), false,
                                                   (*cmp)[ket], 1.0, 1.0);
                }
                if (cmp->info->n == 0 || (cutoff != 0 && cmp->norm() < cutoff))
                    ccmp[ip - pstart] = nullptr;
                else if (ref.size() != 0 && (nh > max_rank || np > max_rank))
                    ccmp[ip - pstart] = nullptr;
                else
                    ccmp[ip - pstart] = cmp;
            }
#pragma omp single
            {
                for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                    if (ccmp[ip - pstart] == nullptr)
                        continue;
                    auto &p = ptrs[ip];
                    IT cur = get<0>(p);
                    int j = get<1>(p), d = get<2>(p);
                    int nh = get<5>(p), np = get<6>(p);
                    vector<uint8_t> det = get<4>(p);
                    det[d] = j;
                    shared_ptr<SparseMatrix<S, FL>> cmp = ccmp[ip - pstart];
                    if (d == n_sites - 1) {
                        assert(cmp->total_memory == 1 &&
                               cmp->info->find_state(mps->info->target) == 0);
                        if (!has_dets) {
                            dets.push_back(cur);
                            vals.push_back(cmp->data[0]);
                            if (enable_look_up) {
                                invs.resize(data.size());
                                IT curx = 0;
                                for (int i = 0; i < n_sites; i++) {
                                    uint8_t jj = det[i];
                                    invs[data[curx][jj]] = curx;
                                    curx = data[curx][jj];
                                }
                            }
                        } else
                            vals[lower_bound(dets.begin(), dets.end(), cur) -
                                 dets.begin()] = cmp->data[0];
                    } else {
                        if (!has_dets) {
                            for (uint8_t jj = 0; jj < (uint8_t)data[cur].size();
                                 jj++)
                                if (data[cur][jj] == 0) {
                                    assert(data.size() <=
                                           (size_t)numeric_limits<IT>::max());
                                    data[cur][jj] = (IT)data.size();
                                    data.push_back(array<IT, 2>{0, 0});
                                }
                        }
                        for (uint8_t jj = 0; jj < (uint8_t)data[cur].size();
                             jj++)
                            if (data[cur][jj] != 0) {
                                int nh_n = nh;
                                int np_n = np;
                                if (ref.size() != 0) {
                                    if (!j && ref[d])
                                        nh_n += 1;
                                    if (j && !ref[d])
                                        np_n += 1;
                                }
                                pptrs.push_back(make_tuple(data[cur][jj], jj,
                                                           d + 1, cmp, det,
                                                           nh_n, np_n));
                            }
                    }
                    ccmp[ip - pstart] = nullptr;
                }
                ptrs.resize(pstart);
                ptrs.insert(ptrs.end(), pptrs.begin(), pptrs.end());
                pptrs.clear();
            }
        }
        pinfos.clear();
        sort_dets();
        threading->activate_normal();
    }
    uint8_t permutation_parity(const vector<int> &perm) {
        uint8_t n = 0;
        vector<uint8_t> tag(perm.size(), 0);
        for (int i = 0, j; i < perm.size(); i++) {
            j = i, n ^= !tag[j];
            while (!tag[j])
                n ^= 1, tag[j] = 1, j = perm[j];
        }
        return n;
    }
    int phase_change_orb_order(const vector<uint8_t> &det,
                               const vector<int> &reorder) {
        vector<int> orb, rev_det, rev;
        orb.reserve(det.size());
        rev_det.reserve(det.size());
        rev.reserve(det.size());
        for (int i = 0; i < det.size(); i++)
            rev_det[reorder[i]] = det[i];
        for (int i = 0, j = 0; i < det.size(); i++)
            rev[i] = j, j += rev_det[i];
        for (int i = 0; i < det.size(); i++)
            if (det[i])
                orb.push_back(rev[reorder[i]]);
        uint8_t n = permutation_parity(orb);
        return 1 - (n << 1);
    }
    void convert_phase(const vector<int> &reorder) {
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int i = 0; i < vals.size(); i++) {
            vector<uint8_t> det = (*this)[i];
            if (reorder.size() != 0)
                vals[i] *= phase_change_orb_order(det, reorder);
        }
        threading->activate_normal();
    }
};

// Prefix trie structure of determinants (arbitrary symmetry)
template <typename S, typename FL>
struct DeterminantTRIE<S, FL, typename S::is_sany_t>
    : TRIE<DeterminantTRIE<S, FL>, FL, 16> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename TRIE<DeterminantTRIE<S, FL>, FL, 16>::XIT IT;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::data;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::dets;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::vals;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::invs;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::n_sites;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::enable_look_up;
    using TRIE<DeterminantTRIE<S, FL>, FL, 16>::sort_dets;
    DeterminantTRIE(int n_sites, bool enable_look_up = false)
        : TRIE<DeterminantTRIE<S, FL>, FL, 16>(n_sites, enable_look_up) {}
    shared_ptr<UnfusedMPS<S, FL>> construct_mps(
        const shared_ptr<MPSInfo<S>> &info,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) const {
        shared_ptr<UnfusedMPS<S, FL>> r = make_shared<UnfusedMPS<S, FL>>();
        r->info = info;
        r->canonical_form = string(n_sites - 1, 'L') + "K";
        r->center = n_sites - 1;
        r->n_sites = n_sites;
        r->dot = 1;
        r->tensors.resize(n_sites);
        S vacuum = info->left_dims_fci[0]->quanta[0];
        vector<pair<S, IT>> cur_nodes =
            vector<pair<S, IT>>{make_pair(vacuum, 0)};
        info->left_dims[0] =
            make_shared<StateInfo<S>>(info->left_dims_fci[0]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            info->left_dims[i + 1] = make_shared<StateInfo<S>>(
                info->left_dims_fci[i + 1]->deep_copy());
        info->right_dims[n_sites] = make_shared<StateInfo<S>>(
            info->right_dims_fci[n_sites]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            info->right_dims[i] =
                make_shared<StateInfo<S>>(info->right_dims_fci[i]->deep_copy());
        for (int k = 0; k < n_sites; k++) {
            vector<array<uint8_t, 3>> basis_iqs;
            for (uint8_t j = 0; j < info->basis[k]->n; j++)
                for (uint8_t jm = 0;
                     jm < info->basis[k]->quanta[j].multiplicity(); jm++)
                    for (uint8_t jj = 0; jj < info->basis[k]->n_states[j]; jj++)
                        basis_iqs.push_back(array<uint8_t, 3>{j, jm, jj});
            if (basis_iqs.size() > data[0].size())
                throw runtime_error("DeterminantTRIE<SAny>::construct_mps "
                                    "basis size too large!");
            shared_ptr<SparseTensor<S, FL>> t =
                make_shared<SparseTensor<S, FL>>();
            vector<pair<S, IT>> next_nodes;
            map<S, MKL_INT> lsh, rsh;
            vector<map<pair<S, S>, vector<array<MKL_INT, 3>>>> blocks(
                info->basis[k]->n);
            vector<map<pair<S, S>, vector<FL>>> coeffs(info->basis[k]->n);
            // determine shape
            for (const auto &irx : cur_nodes) {
                IT ir = irx.second;
                S pq = irx.first;
                for (uint8_t j = 0; j < (uint8_t)basis_iqs.size(); j++)
                    if (data[ir][j] != 0) {
                        S nq = pq + info->basis[k]->quanta[basis_iqs[j][0]];
                        assert(basis_iqs[j][1] < nq.count());
                        nq = nq[basis_iqs[j][1]];
                        next_nodes.push_back(make_pair(nq, data[ir][j]));
                        blocks[basis_iqs[j][0]][make_pair(pq, nq)].push_back(
                            array<MKL_INT, 3>{lsh[pq], basis_iqs[j][2],
                                              rsh[nq]});
                        if (k == n_sites - 1) {
                            int idx =
                                (int)(lower_bound(dets.begin(), dets.end(),
                                                  data[ir][j]) -
                                      dets.begin());
                            assert(idx < (int)dets.size() &&
                                   dets[idx] == data[ir][j]);
                            coeffs[basis_iqs[j][0]][make_pair(pq, nq)]
                                .push_back(vals[idx]);
                        } else
                            rsh[nq]++;
                    }
                lsh[pq]++;
            }
            if (k == n_sites - 1) {
                assert(rsh.size() == 1);
                rsh.begin()->second = 1;
            }
            // create tensor
            t->data.resize(blocks.size());
            for (uint8_t j = 0; j < (uint8_t)blocks.size(); j++) {
                uint8_t nmid = info->basis[k]->n_states[j];
                for (const auto &mp : blocks[j]) {
                    shared_ptr<GTensor<FL>> gt = make_shared<GTensor<FL>>(
                        lsh.at(mp.first.first), nmid, rsh.at(mp.first.second));
                    t->data[j].push_back(make_pair(
                        make_pair(mp.first.first, mp.first.second), gt));
                    if (k == n_sites - 1)
                        for (size_t im = 0; im < mp.second.size(); im++)
                            (*gt)({mp.second[im][0], mp.second[im][1],
                                   mp.second[im][2]}) =
                                coeffs[j].at(mp.first)[im];
                    else
                        for (const auto &mx : mp.second)
                            (*gt)({mx[0], mx[1], mx[2]}) = (FL)(FP)1.0;
                }
            }
            cur_nodes = next_nodes;
            // put shape in bond dims
            total_bond_t new_total = 0;
            for (int p = 0; p < info->left_dims[k + 1]->n; p++) {
                if (rsh.count(info->left_dims[k + 1]->quanta[p])) {
                    info->left_dims[k + 1]->n_states[p] =
                        (ubond_t)rsh.at(info->left_dims[k + 1]->quanta[p]);
                    new_total += info->left_dims[k + 1]->n_states[p];
                } else
                    info->left_dims[k + 1]->n_states[p] = 0;
            }
            info->left_dims[k + 1]->n_states_total = new_total;
            r->tensors[k] = t;
        }
        for (int i = n_sites - 1; i >= 0; i--)
            if (info->right_dims[i]->n_states_total >
                (total_bond_t)dets.size()) {
                total_bond_t new_total = 0;
                for (int k = 0; k < info->right_dims[i]->n; k++) {
                    uint64_t new_n_states =
                        (uint64_t)(ceil((double)info->right_dims[i]
                                            ->n_states[k] *
                                        dets.size() /
                                        info->right_dims[i]->n_states_total) +
                                   0.1);
                    assert(new_n_states != 0);
                    info->right_dims[i]->n_states[k] =
                        (ubond_t)min((uint64_t)new_n_states,
                                     (uint64_t)numeric_limits<ubond_t>::max());
                    new_total += info->right_dims[i]->n_states[k];
                }
                info->right_dims[i]->n_states_total = new_total;
            }
        for (int i = 0; i <= n_sites; i++)
            StateInfo<S>::filter(*info->right_dims[i], *info->left_dims[i],
                                 info->target);
        for (int i = 0; i <= n_sites; i++)
            info->left_dims[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            info->right_dims[i]->collect();
        info->check_bond_dimensions();
        if (para_rule == nullptr || para_rule->is_root())
            info->save_mutable();
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        return r;
    }
    // set the value for each DET to the overlap between mps
    void evaluate(const shared_ptr<UnfusedMPS<S, FL>> &mps, FP cutoff = 0,
                  int max_rank = -1, const vector<uint8_t> &ref = {}) {
        assert(max_rank == -1);
        vals.resize(dets.size());
        memset(vals.data(), 0, sizeof(FL) * vals.size());
        bool has_dets = dets.size() != 0;
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>>>
            ptrs;
        vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos(n_sites + 1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(*mps->info->left_dims_fci[0],
                                 *mps->info->left_dims_fci[0],
                                 mps->info->vacuum, false);
        vector<vector<array<uint8_t, 3>>> basis_iqs(n_sites);
        for (int d = 0; d < n_sites; d++) {
            for (uint8_t j = 0; j < mps->info->basis[d]->n; j++)
                for (uint8_t jm = 0;
                     jm < mps->info->basis[d]->quanta[j].multiplicity(); jm++)
                    for (uint8_t jj = 0; jj < mps->info->basis[d]->n_states[j];
                         jj++)
                        basis_iqs[d].push_back(array<uint8_t, 3>{j, jm, jj});
            if (basis_iqs[d].size() > data[0].size())
                throw runtime_error(
                    "DeterminantTRIE<SAny>::evaluate basis size too large!");

            pinfos[d + 1].resize(basis_iqs[d].size());
            for (int j = 0; j < pinfos[d + 1].size(); j++) {
                int jd = basis_iqs[d][j][0];
                map<S, MKL_INT> qkets;
                for (auto &m : mps->tensors[d]->data[jd]) {
                    S bra = m.first.first, ket = m.first.second;
                    S jket = bra + mps->info->basis[d]->quanta[jd];
                    if (basis_iqs[d][j][1] >= jket.count())
                        continue;
                    if (jket[basis_iqs[d][j][1]] == ket && !qkets.count(ket))
                        qkets[ket] = m.second->shape[2];
                }
                StateInfo<S> ibra, iket;
                ibra.allocate((int)qkets.size());
                iket.allocate((int)qkets.size());
                int k = 0;
                for (auto &qm : qkets) {
                    ibra.quanta[k] = iket.quanta[k] = qm.first;
                    ibra.n_states[k] = 1;
                    iket.n_states[k] = (ubond_t)qm.second;
                    k++;
                }
                pinfos[d + 1][j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                pinfos[d + 1][j]->initialize(ibra, iket, mps->info->vacuum,
                                             false);
            }
        }
        if (!has_dets) {
            for (uint8_t j = 0; j < (uint8_t)basis_iqs[0].size(); j++)
                if (data[0][j] == 0) {
                    assert(data.size() <= (size_t)numeric_limits<IT>::max());
                    data[0][j] = (IT)data.size();
                    data.push_back({});
                }
        }
        shared_ptr<SparseMatrix<S, FL>> zmat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        zmat->allocate(pinfos[0][0]);
        for (size_t j = 0; j < zmat->total_memory; j++)
            zmat->data[j] = 1.0;
        vector<uint8_t> zdet(n_sites);
        for (uint8_t j = 0; j < (uint8_t)basis_iqs[0].size(); j++)
            if (data[0][j] != 0)
                ptrs.push_back(make_tuple(data[0][j], j, 0, zmat, zdet));
        vector<tuple<IT, int, int, shared_ptr<SparseMatrix<S, FL>>,
                     vector<uint8_t>>>
            pptrs;
        int ntg = threading->activate_global();
        int ngroup = ntg * 4;
        vector<shared_ptr<SparseMatrix<S, FL>>> ccmp(ngroup);
#pragma omp parallel num_threads(ntg)
        // depth-first traverse of trie
        while (!ptrs.empty()) {
#pragma omp master
            check_signal_()();
            int pstart = max(0, (int)ptrs.size() - ngroup);
#pragma omp for schedule(static)
            for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                shared_ptr<VectorAllocator<FP>> pd_alloc =
                    make_shared<VectorAllocator<FP>>();
                auto &p = ptrs[ip];
                int j = get<1>(p), d = get<2>(p);
                shared_ptr<SparseMatrix<S, FL>> pmp = get<3>(p);
                shared_ptr<SparseMatrix<S, FL>> cmp =
                    make_shared<SparseMatrix<S, FL>>(pd_alloc);
                cmp->allocate(pinfos[d + 1][j]);
                int jd = basis_iqs[d][j][0];
                for (auto &m : mps->tensors[d]->data[jd]) {
                    S bra = m.first.first, ket = m.first.second;
                    S jket = bra + mps->info->basis[d]->quanta[jd];
                    if (basis_iqs[d][j][1] >= jket.count())
                        continue;
                    if (jket[basis_iqs[d][j][1]] != ket)
                        continue;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    GMatrixFunctions<FL>::multiply(
                        (*pmp)[bra], false,
                        GMatrix<FL>(m.second->data->data() +
                                        basis_iqs[d][j][2] * m.second->shape[2],
                                    m.second->shape[0], m.second->shape[2]),
                        false, (*cmp)[ket], 1.0, 1.0,
                        (MKL_INT)m.second->shape[1] * m.second->shape[2]);
                }
                if (cmp->info->n == 0 || (cutoff != 0 && cmp->norm() < cutoff))
                    ccmp[ip - pstart] = nullptr;
                else
                    ccmp[ip - pstart] = cmp;
            }
#pragma omp single
            {
                for (int ip = pstart; ip < (int)ptrs.size(); ip++) {
                    if (ccmp[ip - pstart] == nullptr)
                        continue;
                    auto &p = ptrs[ip];
                    IT cur = get<0>(p);
                    int j = get<1>(p), d = get<2>(p);
                    vector<uint8_t> det = get<4>(p);
                    det[d] = j;
                    shared_ptr<SparseMatrix<S, FL>> cmp = ccmp[ip - pstart];
                    if (d == n_sites - 1) {
                        assert(cmp->total_memory == 1 &&
                               cmp->info->find_state(mps->info->target) == 0);
                        if (!has_dets) {
                            dets.push_back(cur);
                            vals.push_back(cmp->data[0]);
                            if (enable_look_up) {
                                invs.resize(data.size());
                                IT curx = 0;
                                for (int i = 0; i < n_sites; i++) {
                                    uint8_t jj = det[i];
                                    invs[data[curx][jj]] = curx;
                                    curx = data[curx][jj];
                                }
                            }
                        } else
                            vals[lower_bound(dets.begin(), dets.end(), cur) -
                                 dets.begin()] = cmp->data[0];
                    } else {
                        if (!has_dets) {
                            for (uint8_t jj = 0;
                                 jj < (uint8_t)basis_iqs[d + 1].size(); jj++)
                                if (data[cur][jj] == 0) {
                                    assert(data.size() <=
                                           (size_t)numeric_limits<IT>::max());
                                    data[cur][jj] = (IT)data.size();
                                    data.push_back({});
                                }
                        }
                        for (uint8_t jj = 0;
                             jj < (uint8_t)basis_iqs[d + 1].size(); jj++)
                            if (data[cur][jj] != 0)
                                pptrs.push_back(make_tuple(data[cur][jj], jj,
                                                           d + 1, cmp, det));
                    }
                    ccmp[ip - pstart] = nullptr;
                }
                ptrs.resize(pstart);
                ptrs.insert(ptrs.end(), pptrs.begin(), pptrs.end());
                pptrs.clear();
            }
        }
        pinfos.clear();
        sort_dets();
        threading->activate_normal();
    }
    uint8_t permutation_parity(const vector<int> &perm) {
        throw runtime_error("Not implemented for arbitrary symmetry!");
        return 0;
    }
    int phase_change_orb_order(const vector<uint8_t> &det,
                               const vector<int> &reorder) {
        throw runtime_error("Not implemented for arbitrary symmetry!");
        return 0;
    }
    void convert_phase(const vector<int> &reorder) {
        throw runtime_error("Not implemented for arbitrary symmetry!");
    }
};

template <typename S, typename FL, typename = void> struct DeterminantQC {
    vector<uint8_t> hf_occ;
    vector<typename S::pg_t> orb_sym;
    vector<FL> h1e_energy;
    int n_trials = 20, n_outer_trials = 50000;
    DeterminantQC(const vector<uint8_t> &hf_occ,
                  const vector<typename S::pg_t> &orb_sym,
                  const vector<FL> &h1e_energy)
        : hf_occ(hf_occ), orb_sym(orb_sym), h1e_energy(h1e_energy) {}
    struct det_less {
        bool operator()(const vector<uint8_t> &a,
                        const vector<uint8_t> &b) const {
            assert(a.size() == b.size());
            for (size_t i = 0; i < a.size(); i++)
                if (a[i] != b[i])
                    return a[i] < b[i];
            return false;
        }
    };
    S det_quantum(const vector<uint8_t> &det, int i_begin, int i_end) const {
        int n_block_sites = i_end - i_begin;
        assert(det.size() == n_block_sites);
        int n = 0, twos = 0, ipg = 0;
        for (int i = 0; i < n_block_sites; i++) {
            n += det[i];
            if (det[i] == 1)
                ipg = S::pg_mul(ipg, orb_sym[i + i_begin]), twos++;
        }
        return S(n, twos, ipg);
    }
    // generate determinants for quantum number q for block [i_begin, i_end)
    vector<vector<uint8_t>> distribute(S q, int i_begin, int i_end) const {
        int n_block_sites = i_end - i_begin;
        vector<uint8_t> idx(n_block_sites, 0);
        for (int i = 0; i < n_block_sites; i++)
            idx[i] = i_begin + i;
        sort(idx.begin(), idx.end(), [this](int i, int j) {
            return hf_occ[i] != hf_occ[j]
                       ? (hf_occ[i] > hf_occ[j])
                       : (xreal<FL>(h1e_energy[i]) < xreal<FL>(h1e_energy[j]));
        });
        int n_alpha = (q.n() + q.twos()) >> 1, n_beta = (q.n() - q.twos()) >> 1;
        int n_docc = min(n_alpha, n_beta);
        assert(n_alpha >= 0 && n_beta >= 0 && n_alpha <= n_block_sites &&
               n_beta <= n_block_sites);
        vector<bool> mask(n_block_sites, true);
        for (int i = 0; i < max(n_alpha, n_beta); i++)
            mask[i] = false;
        vector<vector<uint8_t>> r;
        for (int jt = 0; jt < n_outer_trials && r.empty(); jt++)
            for (int it = 0; it < n_trials; it++) {
                next_permutation(mask.begin(), mask.end());
                vector<uint8_t> iocc(n_block_sites, 0);
                for (int i = 0, j = 0; i < n_block_sites; i++)
                    !mask[i] && (iocc[idx[i] - i_begin] = j++ < n_docc ? 2 : 1);
                if (det_quantum(iocc, i_begin, i_end).pg() == q.pg())
                    r.push_back(iocc);
            }
        return r;
    }
};

template <typename S, typename FL>
struct DeterminantQC<S, FL, typename S::is_sany_t> {
    vector<uint8_t> hf_occ;
    vector<typename S::pg_t> orb_sym;
    vector<FL> h1e_energy;
    int n_trials = 20, n_outer_trials = 50000;
    DeterminantQC(const vector<uint8_t> &hf_occ,
                  const vector<typename S::pg_t> &orb_sym,
                  const vector<FL> &h1e_energy)
        : hf_occ(hf_occ), orb_sym(orb_sym), h1e_energy(h1e_energy) {}
    struct det_less {
        bool operator()(const vector<uint8_t> &a,
                        const vector<uint8_t> &b) const {
            assert(a.size() == b.size());
            for (size_t i = 0; i < a.size(); i++)
                if (a[i] != b[i])
                    return a[i] < b[i];
            return false;
        }
    };
    S det_quantum(const vector<uint8_t> &det, int i_begin, int i_end) const {
        assert(false);
        return S();
    }
    // generate determinants for quantum number q for block [i_begin, i_end)
    vector<vector<uint8_t>> distribute(S q, int i_begin, int i_end) const {
        assert(false);
        return vector<vector<uint8_t>>();
    }
};

// Quantum number infomation in a MPS
// Generated from determinant, used for warm-up sweep
template <typename S, typename FL> struct DeterminantMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::basis;
    shared_ptr<FCIDUMP<FL>> fcidump;
    shared_ptr<DeterminantQC<S, FL>> det;
    vector<uint8_t> iocc;
    ubond_t n_det_states = 2; // number of states for each determinant
    DeterminantMPSInfo(int n_sites, S vacuum, S target,
                       const vector<shared_ptr<StateInfo<S>>> &basis,
                       const vector<typename S::pg_t> &orb_sym,
                       const vector<uint8_t> &iocc,
                       const shared_ptr<FCIDUMP<FL>> &fcidump)
        : iocc(iocc), fcidump(fcidump),
          det(make_shared<DeterminantQC<S, FL>>(iocc, orb_sym,
                                                fcidump->h1e_energy())),
          MPSInfo<S>(n_sites, vacuum, target, basis) {}
    void set_bond_dimension(ubond_t m) override {
        this->bond_dim = m;
        this->left_dims[0] = make_shared<StateInfo<S>>(this->vacuum);
        this->right_dims[this->n_sites] =
            make_shared<StateInfo<S>>(this->vacuum);
    }
    WarmUpTypes get_warm_up_type() const override {
        return WarmUpTypes::Determinant;
    }
    void set_left_bond_dimension(int i,
                                 const vector<vector<vector<uint8_t>>> &dets) {
        this->left_dims[0] = make_shared<StateInfo<S>>(this->vacuum);
        for (int j = 0; j < i; j++) {
            set<vector<uint8_t>, typename DeterminantQC<S, FL>::det_less> mp;
            for (auto &idets : dets)
                for (auto &jdet : idets)
                    mp.insert(
                        vector<uint8_t>(jdet.begin(), jdet.begin() + j + 1));
            this->left_dims[j + 1]->allocate((int)mp.size());
            auto it = mp.begin();
            for (int k = 0; k < this->left_dims[j + 1]->n; k++, it++) {
                this->left_dims[j + 1]->quanta[k] =
                    det->det_quantum(*it, 0, j + 1);
                this->left_dims[j + 1]->n_states[k] = 1;
            }
            this->left_dims[j + 1]->sort_states();
            this->left_dims[j + 1]->collect();
        }
        this->left_dims[i + 1]->allocate((int)dets.size());
        for (int k = 0; k < this->left_dims[i + 1]->n; k++) {
            this->left_dims[i + 1]->quanta[k] =
                det->det_quantum(dets[k][0], 0, i + 1);
            this->left_dims[i + 1]->n_states[k] = (ubond_t)dets[k].size();
        }
        this->left_dims[i + 1]->sort_states();
        for (int k = i + 1; k < this->n_sites; k++)
            this->left_dims[k + 1]->n = 0;
    }
    void set_right_bond_dimension(int i,
                                  const vector<vector<vector<uint8_t>>> &dets) {
        this->right_dims[this->n_sites] =
            make_shared<StateInfo<S>>(this->vacuum);
        for (int j = this->n_sites - 1; j > i; j--) {
            set<vector<uint8_t>, typename DeterminantQC<S, FL>::det_less> mp;
            for (auto &idets : dets)
                for (auto &jdet : idets)
                    mp.insert(
                        vector<uint8_t>(jdet.begin() + (j - i), jdet.end()));
            this->right_dims[j]->allocate((int)mp.size());
            auto it = mp.begin();
            for (int k = 0; k < this->right_dims[j]->n; k++, it++) {
                this->right_dims[j]->quanta[k] =
                    det->det_quantum(*it, j, this->n_sites);
                this->right_dims[j]->n_states[k] = 1;
            }
            this->right_dims[j]->sort_states();
            this->right_dims[j]->collect();
        }
        this->right_dims[i]->allocate((int)dets.size());
        for (int k = 0; k < this->right_dims[i]->n; k++) {
            this->right_dims[i]->quanta[k] =
                det->det_quantum(dets[k][0], i, this->n_sites);
            this->right_dims[i]->n_states[k] = (ubond_t)dets[k].size();
        }
        this->right_dims[i]->sort_states();
        for (int k = i - 1; k >= 0; k--)
            this->right_dims[k]->n = 0;
    }
    vector<vector<vector<uint8_t>>> get_determinants(StateInfo<S> &st,
                                                     int i_begin, int i_end) {
        vector<vector<vector<uint8_t>>> dets;
        dets.reserve(st.n);
        for (int j = 0; j < st.n; j++) {
            vector<vector<uint8_t>> dd =
                det->distribute(st.quanta[j], i_begin, i_end);
            if (dd.size() == 0)
                continue;
            int n_states = min((int)dd.size(), (int)st.n_states[j]);
            vector<FL> dd_energies(dd.size());
            vector<int> dd_idx(dd.size());
            for (size_t k = 0; k < dd.size(); k++)
                dd_energies[k] = fcidump->det_energy(dd[k], i_begin, i_end),
                dd_idx[k] = (int)k;
            sort(dd_idx.begin(), dd_idx.end(), [&dd_energies](int ii, int jj) {
                return xreal(dd_energies[ii]) < xreal(dd_energies[jj]);
            });
            dets.push_back(vector<vector<uint8_t>>());
            for (int k = 0; k < n_states; k++)
                dets.back().push_back(dd[dd_idx[k]]);
        }
        st.deallocate();
        return dets;
    }
    // generate quantum numbers based on determinant for left block [0, i]
    // right bond dimension at site i_right_ref is used as reference
    StateInfo<S> get_complementary_left_dims(int i, int i_right_ref,
                                             bool match_prev = false) {
        this->load_right_dims(i_right_ref);
        StateInfo<S> rref = *this->right_dims[i_right_ref];
        for (int k = i_right_ref - 1; k >= i + 1; k--) {
            StateInfo<S> rr = StateInfo<S>::tensor_product(
                *basis[k], rref, *this->right_dims_fci[k]);
            rref = rr;
        }
        // get complementary quantum numbers
        map<S, ubond_t> qs;
        for (int i = 0; i < rref.n; i++) {
            S qls = this->target - rref.quanta[i];
            for (int k = 0; k < qls.count(); k++)
                qs[qls[k]] += rref.n_states[i];
        }
        rref.deallocate();
        if (match_prev) {
            this->load_left_dims(i + 1);
            for (int l = 0; l < this->left_dims[i + 1]->n; l++) {
                S q = this->left_dims[i + 1]->quanta[l];
                if (qs.count(q) == 0)
                    qs[q] = this->left_dims[i + 1]->n_states[l];
                else
                    qs[q] = max(qs[q], this->left_dims[i + 1]->n_states[l]);
            }
            this->left_dims[i + 1]->deallocate();
        }
        StateInfo<S> lref;
        lref.allocate((int)qs.size());
        int k = 0;
        for (auto &q : qs) {
            lref.quanta[k] = q.first;
            lref.n_states[k] = min(q.second, n_det_states);
            k++;
        }
        lref.sort_states();
        return lref;
    }
    // generate quantum numbers based on determinant for right block [i,
    // n_sites) left bond dimension at site i_left_ref is used as reference
    StateInfo<S> get_complementary_right_dims(int i, int i_left_ref,
                                              bool match_prev = false) {
        this->load_left_dims(i_left_ref + 1);
        StateInfo<S> lref = *this->left_dims[i_left_ref + 1];
        for (int k = i_left_ref + 1; k < i; k++) {
            StateInfo<S> ll = StateInfo<S>::tensor_product(
                lref, *basis[k], *this->left_dims_fci[k + 1]);
            lref = ll;
        }
        // get complementary quantum numbers
        map<S, ubond_t> qs;
        for (int i = 0; i < lref.n; i++) {
            S qrs = this->target - lref.quanta[i];
            for (int k = 0; k < qrs.count(); k++)
                qs[qrs[k]] += lref.n_states[i];
        }
        lref.deallocate();
        if (match_prev) {
            this->load_right_dims(i);
            for (int l = 0; l < this->right_dims[i]->n; l++) {
                S q = this->right_dims[i]->quanta[l];
                if (qs.count(q) == 0)
                    qs[q] = this->right_dims[i]->n_states[l];
                else
                    qs[q] = max(qs[q], this->right_dims[i]->n_states[l]);
            }
            this->right_dims[i]->deallocate();
        }
        StateInfo<S> rref;
        rref.allocate((int)qs.size());
        int k = 0;
        for (auto &q : qs) {
            rref.quanta[k] = q.first;
            rref.n_states[k] = min(q.second, n_det_states);
            k++;
        }
        rref.sort_states();
        return rref;
    }
};

} // namespace block2
