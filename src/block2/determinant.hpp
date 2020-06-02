
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
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

#include "integral.hpp"
#include "mps.hpp"
#include <algorithm>
#include <set>
#include <vector>

using namespace std;

namespace block2 {

template <typename S> struct DeterminantQC {
    vector<uint8_t> hf_occ, orb_sym;
    vector<double> h1e_energy;
    int n_trials = 20, n_outer_trials = 50000;
    DeterminantQC(const vector<uint8_t> &hf_occ, const vector<uint8_t> &orb_sym,
                  const vector<double> &h1e_energy)
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
        uint16_t n = 0, twos = 0, ipg = 0;
        for (int i = 0; i < n_block_sites; i++) {
            n += det[i];
            if (det[i] == 1)
                ipg ^= orb_sym[i + i_begin], twos++;
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
            return hf_occ[i] != hf_occ[j] ? (hf_occ[i] > hf_occ[j])
                                          : (h1e_energy[i] < h1e_energy[j]);
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

// Quantum number infomation in a MPS
// Generated from determinant, used for warm-up sweep
template <typename S> struct DeterminantMPSInfo : MPSInfo<S> {
    shared_ptr<FCIDUMP> fcidump;
    shared_ptr<DeterminantQC<S>> det;
    vector<uint8_t> iocc;
    uint16_t n_det_states = 2; // number of states for each determinant
    DeterminantMPSInfo(int n_sites, S vaccum, S target, StateInfo<S> *basis,
                       const vector<uint8_t> orbsym, uint8_t n_syms,
                       const vector<uint8_t> &iocc,
                       const shared_ptr<FCIDUMP> &fcidump)
        : iocc(iocc), fcidump(fcidump),
          det(make_shared<DeterminantQC<S>>(iocc, orbsym,
                                            fcidump->h1e_energy())),
          MPSInfo<S>(n_sites, vaccum, target, basis, orbsym, n_syms) {}
    void set_bond_dimension(uint16_t m) override {
        this->bond_dim = m;
        this->left_dims[0] = StateInfo<S>(this->vaccum);
        this->right_dims[this->n_sites] = StateInfo<S>(this->vaccum);
    }
    WarmUpTypes get_warm_up_type() const override {
        return WarmUpTypes::Determinant;
    }
    void set_left_bond_dimension(int i,
                                 const vector<vector<vector<uint8_t>>> &dets) {
        this->left_dims[0] = StateInfo<S>(this->vaccum);
        for (int j = 0; j < i; j++) {
            set<vector<uint8_t>, typename DeterminantQC<S>::det_less> mp;
            for (auto &idets : dets)
                for (auto &jdet : idets)
                    mp.insert(
                        vector<uint8_t>(jdet.begin(), jdet.begin() + j + 1));
            this->left_dims[j + 1].allocate(mp.size());
            auto it = mp.begin();
            for (int k = 0; k < this->left_dims[j + 1].n; k++, it++) {
                this->left_dims[j + 1].quanta[k] =
                    det->det_quantum(*it, 0, j + 1);
                this->left_dims[j + 1].n_states[k] = 1;
            }
            this->left_dims[j + 1].sort_states();
            this->left_dims[j + 1].collect();
        }
        this->left_dims[i + 1].allocate(dets.size());
        for (int k = 0; k < this->left_dims[i + 1].n; k++) {
            this->left_dims[i + 1].quanta[k] =
                det->det_quantum(dets[k][0], 0, i + 1);
            this->left_dims[i + 1].n_states[k] = dets[k].size();
        }
        this->left_dims[i + 1].sort_states();
        for (int k = i + 1; k < this->n_sites; k++)
            this->left_dims[k + 1].n = 0;
    }
    void set_right_bond_dimension(int i,
                                  const vector<vector<vector<uint8_t>>> &dets) {
        this->right_dims[this->n_sites] = StateInfo<S>(this->vaccum);
        for (int j = this->n_sites - 1; j > i; j--) {
            set<vector<uint8_t>, typename DeterminantQC<S>::det_less> mp;
            for (auto &idets : dets)
                for (auto &jdet : idets)
                    mp.insert(
                        vector<uint8_t>(jdet.begin() + (j - i), jdet.end()));
            this->right_dims[j].allocate(mp.size());
            auto it = mp.begin();
            for (int k = 0; k < this->right_dims[j].n; k++, it++) {
                this->right_dims[j].quanta[k] =
                    det->det_quantum(*it, j, this->n_sites);
                this->right_dims[j].n_states[k] = 1;
            }
            this->right_dims[j].sort_states();
            this->right_dims[j].collect();
        }
        this->right_dims[i].allocate(dets.size());
        for (int k = 0; k < this->right_dims[i].n; k++) {
            this->right_dims[i].quanta[k] =
                det->det_quantum(dets[k][0], i, this->n_sites);
            this->right_dims[i].n_states[k] = dets[k].size();
        }
        this->right_dims[i].sort_states();
        for (int k = i - 1; k >= 0; k--)
            this->right_dims[k].n = 0;
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
            vector<double> dd_energies(dd.size());
            vector<int> dd_idx(dd.size());
            for (size_t k = 0; k < dd.size(); k++)
                dd_energies[k] = fcidump->det_energy(dd[k], i_begin, i_end),
                dd_idx[k] = k;
            sort(dd_idx.begin(), dd_idx.end(), [&dd_energies](int ii, int jj) {
                return dd_energies[ii] < dd_energies[jj];
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
        StateInfo<S> rref = this->right_dims[i_right_ref];
        for (int k = i_right_ref - 1; k >= i + 1; k--) {
            StateInfo<S> rr = StateInfo<S>::tensor_product(
                this->get_basis(k), rref, this->right_dims_fci[k]);
            rref.reallocate(0);
            rr.reallocate(rr.n);
            rref = rr;
        }
        // get complementary quantum numbers
        map<S, uint16_t> qs;
        for (int i = 0; i < rref.n; i++) {
            S qls = this->target - rref.quanta[i];
            for (int k = 0; k < qls.count(); k++)
                qs[qls[k]] += rref.n_states[i];
        }
        rref.deallocate();
        if (match_prev) {
            this->load_left_dims(i + 1);
            for (int l = 0; l < this->left_dims[i + 1].n; l++) {
                S q = this->left_dims[i + 1].quanta[l];
                if (qs.count(q) == 0)
                    qs[q] = this->left_dims[i + 1].n_states[l];
                else
                    qs[q] = max(qs[q], this->left_dims[i + 1].n_states[l]);
            }
            this->left_dims[i + 1].deallocate();
        }
        StateInfo<S> lref;
        lref.allocate(qs.size());
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
        StateInfo<S> lref = this->left_dims[i_left_ref + 1];
        for (int k = i_left_ref + 1; k < i; k++) {
            StateInfo<S> ll = StateInfo<S>::tensor_product(
                lref, this->get_basis(k), this->left_dims_fci[k + 1]);
            lref.reallocate(0);
            ll.reallocate(ll.n);
            lref = ll;
        }
        // get complementary quantum numbers
        map<S, uint16_t> qs;
        for (int i = 0; i < lref.n; i++) {
            S qrs = this->target - lref.quanta[i];
            for (int k = 0; k < qrs.count(); k++)
                qs[qrs[k]] += lref.n_states[i];
        }
        lref.deallocate();
        if (match_prev) {
            this->load_right_dims(i);
            for (int l = 0; l < this->right_dims[i].n; l++) {
                S q = this->right_dims[i].quanta[l];
                if (qs.count(q) == 0)
                    qs[q] = this->right_dims[i].n_states[l];
                else
                    qs[q] = max(qs[q], this->right_dims[i].n_states[l]);
            }
            this->right_dims[i].deallocate();
        }
        StateInfo<S> rref;
        rref.allocate(qs.size());
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