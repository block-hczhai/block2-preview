
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

#include "ancilla.hpp"
#include "sparse_matrix.hpp"
#include "state_info.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

// Read occupation numbers from a file
inline vector<double> read_occ(const string &filename) {
    assert(Parsing::file_exists(filename));
    ifstream ifs(filename.c_str());
    vector<string> lines = Parsing::readlines(&ifs);
    assert(lines.size() >= 1);
    vector<string> vals = Parsing::split(lines[0], " ", true);
    vector<double> r;
    transform(vals.begin(), vals.end(), back_inserter(r), Parsing::to_double);
    return r;
}

// Quantum number infomation in a MPS
template <typename S> struct MPSInfo {
    int n_sites;
    S vaccum;
    S target;
    vector<uint8_t> orbsym;
    uint8_t n_syms;
    uint16_t bond_dim;
    StateInfo<S> *basis, *left_dims_fci, *right_dims_fci;
    StateInfo<S> *left_dims, *right_dims;
    string tag = "KET";
    MPSInfo(int n_sites, S vaccum, S target, StateInfo<S> *basis,
            const vector<uint8_t> orbsym, uint8_t n_syms)
        : n_sites(n_sites), vaccum(vaccum), target(target), orbsym(orbsym),
          n_syms(n_syms), basis(basis), bond_dim(0) {
        left_dims_fci = new StateInfo<S>[n_sites + 1];
        left_dims_fci[0] = StateInfo<S>(vaccum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] = StateInfo<S>::tensor_product(
                left_dims_fci[i], basis[orbsym[i]], target);
        right_dims_fci = new StateInfo<S>[n_sites + 1];
        right_dims_fci[n_sites] = StateInfo<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] = StateInfo<S>::tensor_product(
                basis[orbsym[i]], right_dims_fci[i + 1], target);
        for (int i = 0; i <= n_sites; i++)
            StateInfo<S>::filter(left_dims_fci[i], right_dims_fci[i], target);
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i].collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i].collect();
        left_dims = new StateInfo<S>[n_sites + 1];
        right_dims = new StateInfo<S>[n_sites + 1];
    }
    virtual AncillaTypes get_ancilla_type() const { return AncillaTypes::None; }
    void set_bond_dimension_using_occ(uint16_t m, const vector<double> &occ,
                                      double bias = 1.0) {
        bond_dim = m;
        // site state probabilities
        StateProbability<S> *site_probs = new StateProbability<S>[n_sites];
        for (int i = 0; i < n_sites; i++) {
            double alpha_occ = occ[i];
            if (bias != 1.0) {
                if (alpha_occ > 1)
                    alpha_occ = 1 + pow(alpha_occ - 1, bias);
                else if (alpha_occ < 1)
                    alpha_occ = 1 - pow(1 - alpha_occ, bias);
            }
            alpha_occ /= 2;
            assert(0 <= alpha_occ && alpha_occ <= 1);
            vector<double> probs = {(1 - alpha_occ) * (1 - alpha_occ),
                                    (1 - alpha_occ) * alpha_occ,
                                    alpha_occ * alpha_occ};
            site_probs[i].allocate(basis[orbsym[i]].n);
            for (int j = 0; j < basis[orbsym[i]].n; j++) {
                site_probs[i].quanta[j] = basis[orbsym[i]].quanta[j];
                site_probs[i].probs[j] = probs[basis[orbsym[i]].quanta[j].n()];
            }
        }
        // left and right block probabilities
        StateProbability<S> *left_probs = new StateProbability<S>[n_sites + 1];
        StateProbability<S> *right_probs = new StateProbability<S>[n_sites + 1];
        left_probs[0] = StateProbability<S>(vaccum);
        for (int i = 0; i < n_sites; i++)
            left_probs[i + 1] = StateProbability<S>::tensor_product_no_collect(
                left_probs[i], site_probs[i], left_dims_fci[i + 1]);
        right_probs[n_sites] = StateProbability<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_probs[i] = StateProbability<S>::tensor_product_no_collect(
                site_probs[i], right_probs[i + 1], right_dims_fci[i]);
        // conditional probabilities
        for (int i = 0; i <= n_sites; i++) {
            double *lprobs = dalloc->allocate(left_probs[i].n);
            double *rprobs = dalloc->allocate(right_probs[i].n);
            for (int j = 0; j < left_probs[i].n; j++)
                lprobs[j] = left_probs[i].probs[j] *
                            left_probs[i].quanta[j].multiplicity();
            for (int j = 0; j < right_probs[i].n; j++)
                rprobs[j] = right_probs[i].probs[j] *
                            right_probs[i].quanta[j].multiplicity();
            for (int j = 0; i > 0 && j < left_probs[i].n; j++) {
                if (left_probs[i].probs[j] == 0)
                    continue;
                double x = 0;
                S rks = target - left_probs[i].quanta[j];
                for (int k = 0, ik; k < rks.count(); k++)
                    if ((ik = right_probs[i].find_state(rks[k])) != -1)
                        x += rprobs[ik];
                left_probs[i].probs[j] *= x;
            }
            for (int j = 0; i < n_sites && j < right_probs[i].n; j++) {
                if (right_probs[i].probs[j] == 0)
                    continue;
                double x = 0;
                S lks = target - right_probs[i].quanta[j];
                for (int k = 0, ik; k < lks.count(); k++)
                    if ((ik = left_probs[i].find_state(lks[k])) != -1)
                        x += lprobs[ik];
                right_probs[i].probs[j] *= x;
            }
            dalloc->deallocate(rprobs, right_probs[i].n);
            dalloc->deallocate(lprobs, left_probs[i].n);
        }
        // adjusted temparary fci dims
        StateInfo<S> *left_dims_fci_t = new StateInfo<S>[n_sites + 1];
        StateInfo<S> *right_dims_fci_t = new StateInfo<S>[n_sites + 1];
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims_fci_t[i] = left_dims_fci[i].deep_copy();
            right_dims_fci_t[i] = right_dims_fci[i].deep_copy();
        }
        // left and right block dims
        left_dims[0] = StateInfo<S>(vaccum);
        for (int i = 1; i <= n_sites; i++) {
            left_dims[i].allocate(left_probs[i].n);
            memcpy(left_dims[i].quanta, left_probs[i].quanta,
                   sizeof(S) * left_probs[i].n);
            double prob_sum =
                accumulate(left_probs[i].probs,
                           left_probs[i].probs + left_probs[i].n, 0.0);
            for (int j = 0; j < left_probs[i].n; j++)
                left_dims[i].n_states[j] =
                    min((uint16_t)round(left_probs[i].probs[j] / prob_sum * m),
                        left_dims_fci_t[i].n_states[j]);
            left_dims[i].collect();
            if (i != n_sites) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    left_dims[i], basis[orbsym[i]], left_dims_fci_t[i + 1]);
                for (int j = 0, k; j < left_dims_fci_t[i + 1].n; j++)
                    if ((k = tmp.find_state(
                             left_dims_fci_t[i + 1].quanta[j])) != -1)
                        left_dims_fci_t[i + 1].n_states[j] =
                            min(tmp.n_states[k],
                                left_dims_fci_t[i + 1].n_states[j]);
                for (int j = 0; j < left_probs[i + 1].n; j++)
                    if (tmp.find_state(left_probs[i + 1].quanta[j]) == -1)
                        left_probs[i + 1].probs[j] = 0;
                tmp.deallocate();
            }
        }
        right_dims[n_sites] = StateInfo<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--) {
            right_dims[i].allocate(right_probs[i].n);
            memcpy(right_dims[i].quanta, right_probs[i].quanta,
                   sizeof(S) * right_probs[i].n);
            double prob_sum =
                accumulate(right_probs[i].probs,
                           right_probs[i].probs + right_probs[i].n, 0.0);
            for (int j = 0; j < right_probs[i].n; j++)
                right_dims[i].n_states[j] =
                    min((uint16_t)round(right_probs[i].probs[j] / prob_sum * m),
                        right_dims_fci_t[i].n_states[j]);
            right_dims[i].collect();
            if (i != 0) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    basis[orbsym[i - 1]], right_dims[i],
                    right_dims_fci_t[i - 1]);
                for (int j = 0, k; j < right_dims_fci_t[i - 1].n; j++)
                    if ((k = tmp.find_state(
                             right_dims_fci_t[i - 1].quanta[j])) != -1)
                        right_dims_fci_t[i - 1].n_states[j] =
                            min(tmp.n_states[k],
                                right_dims_fci_t[i - 1].n_states[j]);
                for (int j = 0; j < right_probs[i - 1].n; j++)
                    if (tmp.find_state(right_probs[i - 1].quanta[j]) == -1)
                        right_probs[i - 1].probs[j] = 0;
                tmp.deallocate();
            }
        }
        for (int i = 0; i < n_sites; i++)
            site_probs[i].reallocate(0);
        for (int i = 0; i <= n_sites; i++)
            left_probs[i].reallocate(0);
        for (int i = n_sites; i >= 0; i--)
            right_probs[i].reallocate(0);
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims_fci_t[i].reallocate(0);
            right_dims_fci_t[i].reallocate(0);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims[i].reallocate(left_dims[i].n);
        for (int i = n_sites; i >= 0; i--)
            right_dims[i].reallocate(right_dims[i].n);
        assert(ialloc->shift == 0);
        delete[] right_dims_fci_t;
        delete[] left_dims_fci_t;
        delete[] right_probs;
        delete[] left_probs;
        delete[] site_probs;
    }
    void set_bond_dimension(uint16_t m) {
        bond_dim = m;
        left_dims[0] = StateInfo<S>(vaccum);
        for (int i = 0; i < n_sites; i++)
            left_dims[i + 1] = left_dims_fci[i + 1].deep_copy();
        for (int i = 0; i < n_sites; i++)
            if (left_dims[i + 1].n_states_total > m) {
                int new_total = 0;
                for (int k = 0; k < left_dims[i + 1].n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)left_dims[i + 1].n_states[k] *
                                        m / left_dims[i + 1].n_states_total) +
                                   0.1);
                    left_dims[i + 1].n_states[k] =
                        (uint16_t)min(new_n_states, 65535U);
                    new_total += left_dims[i + 1].n_states[k];
                }
                left_dims[i + 1].n_states_total = new_total;
            }
        right_dims[n_sites] = StateInfo<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims[i] = right_dims_fci[i].deep_copy();
        for (int i = n_sites - 1; i >= 0; i--)
            if (right_dims[i].n_states_total > m) {
                int new_total = 0;
                for (int k = 0; k < right_dims[i].n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)right_dims[i].n_states[k] * m /
                                        right_dims[i].n_states_total) +
                                   0.1);
                    right_dims[i].n_states[k] =
                        (uint16_t)min(new_n_states, 65535U);
                    new_total += right_dims[i].n_states[k];
                }
                right_dims[i].n_states_total = new_total;
            }
        for (int i = -1; i < n_sites - 1; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                left_dims[i + 1], basis[orbsym[i + 1]], target);
            int new_total = 0;
            for (int k = 0; k < left_dims[i + 2].n; k++) {
                int tk = t.find_state(left_dims[i + 2].quanta[k]);
                if (tk == -1)
                    left_dims[i + 2].n_states[k] = 0;
                else if (left_dims[i + 2].n_states[k] > t.n_states[tk])
                    left_dims[i + 2].n_states[k] = t.n_states[tk];
                new_total += left_dims[i + 2].n_states[k];
            }
            left_dims[i + 2].n_states_total = new_total;
            t.deallocate();
        }
        for (int i = n_sites; i > 0; i--) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                basis[orbsym[i - 1]], right_dims[i], target);
            int new_total = 0;
            for (int k = 0; k < right_dims[i - 1].n; k++) {
                int tk = t.find_state(right_dims[i - 1].quanta[k]);
                if (tk == -1)
                    right_dims[i - 1].n_states[k] = 0;
                else if (right_dims[i - 1].n_states[k] > t.n_states[tk])
                    right_dims[i - 1].n_states[k] = t.n_states[tk];
                new_total += right_dims[i - 1].n_states[k];
            }
            right_dims[i - 1].n_states_total = new_total;
            t.deallocate();
        }
    }
    string get_filename(bool left, int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MPS.INFO." << tag
           << (left ? ".LEFT." : ".RIGHT.") << Parsing::to_string(i);
        return ss.str();
    }
    void save_mutable() const {
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims[i].save_data(get_filename(true, i));
            right_dims[i].save_data(get_filename(false, i));
        }
    }
    void load_mutable() const {
        for (int i = 0; i <= n_sites; i++)
            left_dims[i].load_data(get_filename(true, i));
        for (int i = n_sites; i >= 0; i--)
            right_dims[i].load_data(get_filename(false, i));
    }
    void deallocate_mutable() {
        for (int i = 0; i <= n_sites; i++)
            right_dims[i].deallocate();
        for (int i = n_sites; i >= 0; i--)
            left_dims[i].deallocate();
    }
    void save_left_dims(int i) const {
        left_dims[i].save_data(get_filename(true, i));
    }
    void save_right_dims(int i) const {
        right_dims[i].save_data(get_filename(false, i));
    }
    void load_left_dims(int i) {
        left_dims[i].load_data(get_filename(true, i));
    }
    void load_right_dims(int i) {
        right_dims[i].load_data(get_filename(false, i));
    }
    void deallocate() {
        for (int i = 0; i <= n_sites; i++)
            right_dims_fci[i].deallocate();
        for (int i = n_sites; i >= 0; i--)
            left_dims_fci[i].deallocate();
    }
    ~MPSInfo() {
        delete[] left_dims;
        delete[] right_dims;
        delete[] left_dims_fci;
        delete[] right_dims_fci;
    }
};

// Adding tensors for ancilla sites to a MPS
// n_sites = 2 * n_physical_sites
template <typename S> struct AncillaMPSInfo : MPSInfo<S> {
    int n_physical_sites;
    static vector<uint8_t> trans_orbsym(const vector<uint8_t> &a, int n_sites) {
        vector<uint8_t> b(n_sites << 1, 0);
        for (int i = 0, j = 0; i < n_sites; i++, j += 2)
            b[j] = b[j + 1] = a[i];
        return b;
    }
    AncillaMPSInfo(int n_sites, S vaccum, S target, StateInfo<S> *basis,
                   const vector<uint8_t> &orbsym, uint8_t n_syms)
        : n_physical_sites(n_sites), MPSInfo<S>(n_sites << 1, vaccum, target,
                                                basis,
                                                trans_orbsym(orbsym, n_sites),
                                                n_syms) {}
    AncillaTypes get_ancilla_type() const override {
        return AncillaTypes::Ancilla;
    }
    void set_thermal_limit() {
        MPSInfo<S>::left_dims[0] = StateInfo<S>(MPSInfo<S>::vaccum);
        for (int i = 0; i < MPSInfo<S>::n_sites; i++)
            if (i & 1) {
                S q = MPSInfo<S>::left_dims[i]
                          .quanta[MPSInfo<S>::left_dims[i].n - 1] +
                      MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]].quanta[0];
                assert(q.count() == 1);
                MPSInfo<S>::left_dims[i + 1] = StateInfo<S>(q);
            } else
                MPSInfo<S>::left_dims[i + 1] = StateInfo<S>::tensor_product(
                    MPSInfo<S>::left_dims[i],
                    MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]],
                    MPSInfo<S>::target);
        MPSInfo<S>::right_dims[MPSInfo<S>::n_sites] =
            StateInfo<S>(MPSInfo<S>::vaccum);
        for (int i = MPSInfo<S>::n_sites - 1; i >= 0; i--)
            if (i & 1)
                MPSInfo<S>::right_dims[i] = StateInfo<S>::tensor_product(
                    MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]],
                    MPSInfo<S>::right_dims[i + 1], MPSInfo<S>::target);
            else {
                S q = MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]].quanta[0] +
                      MPSInfo<S>::right_dims[i + 1]
                          .quanta[MPSInfo<S>::right_dims[i + 1].n - 1];
                assert(q.count() == 1);
                MPSInfo<S>::right_dims[i] = StateInfo<S>(q);
            }
    }
};

// Matrix Product State
template <typename S> struct MPS {
    int n_sites, center, dot;
    shared_ptr<MPSInfo<S>> info;
    vector<shared_ptr<SparseMatrix<S>>> tensors;
    string canonical_form;
    MPS(const shared_ptr<MPSInfo<S>> &info)
        : n_sites(0), center(0), dot(0), info(info) {}
    MPS(int n_sites, int center, int dot)
        : n_sites(n_sites), center(center), dot(dot) {
        canonical_form.resize(n_sites);
        for (int i = 0; i < center; i++)
            canonical_form[i] = 'L';
        for (int i = center; i < center + dot; i++)
            canonical_form[i] = 'C';
        for (int i = center + dot; i < n_sites; i++)
            canonical_form[i] = 'R';
    }
    void initialize(const shared_ptr<MPSInfo<S>> &info) {
        this->info = info;
        vector<shared_ptr<SparseMatrixInfo<S>>> mat_infos;
        mat_infos.resize(n_sites);
        tensors.resize(n_sites);
        for (int i = 0; i < center; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                info->left_dims[i], info->basis[info->orbsym[i]],
                info->left_dims_fci[i + 1]);
            mat_infos[i] = make_shared<SparseMatrixInfo<S>>();
            mat_infos[i]->initialize(t, info->left_dims[i + 1], info->vaccum,
                                     false);
            t.reallocate(0);
            mat_infos[i]->reallocate(mat_infos[i]->n);
        }
        mat_infos[center] = make_shared<SparseMatrixInfo<S>>();
        if (dot == 1) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                info->left_dims[center], info->basis[info->orbsym[center]],
                info->left_dims_fci[center + dot]);
            mat_infos[center]->initialize(t, info->right_dims[center + dot],
                                          info->target, false, true);
            t.reallocate(0);
            mat_infos[center]->reallocate(mat_infos[center]->n);
        } else {
            StateInfo<S> tl = StateInfo<S>::tensor_product(
                info->left_dims[center], info->basis[info->orbsym[center]],
                info->left_dims_fci[center + 1]);
            StateInfo<S> tr = StateInfo<S>::tensor_product(
                info->basis[info->orbsym[center + 1]],
                info->right_dims[center + dot],
                info->right_dims_fci[center + 1]);
            mat_infos[center]->initialize(tl, tr, info->target, false, true);
            tl.reallocate(0);
            tr.reallocate(0);
            mat_infos[center]->reallocate(mat_infos[center]->n);
        }
        for (int i = center + dot; i < n_sites; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                info->basis[info->orbsym[i]], info->right_dims[i + 1],
                info->right_dims_fci[i]);
            mat_infos[i] = make_shared<SparseMatrixInfo<S>>();
            mat_infos[i]->initialize(info->right_dims[i], t, info->vaccum,
                                     false);
            t.reallocate(0);
            mat_infos[i]->reallocate(mat_infos[i]->n);
        }
        for (int i = 0; i < n_sites; i++)
            if (mat_infos[i] != nullptr) {
                tensors[i] = make_shared<SparseMatrix<S>>();
                tensors[i]->allocate(mat_infos[i]);
            }
    }
    void fill_thermal_limit() {
        assert(info->get_ancilla_type() == AncillaTypes::Ancilla);
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                if (i < center || i > center || (i == center && dot == 1)) {
                    int n = info->basis[info->orbsym[i]].n;
                    assert(tensors[i]->total_memory == n);
                    if (i & 1)
                        for (int j = 0; j < n; j++)
                            tensors[i]->data[j] = 1.0;
                    else {
                        double norm = 0;
                        for (int j = 0; j < n; j++)
                            norm += info->basis[info->orbsym[i]]
                                        .quanta[j]
                                        .multiplicity();
                        norm = sqrt(norm);
                        for (int j = 0; j < n; j++)
                            tensors[i]->data[j] =
                                sqrt(info->basis[info->orbsym[i]]
                                         .quanta[j]
                                         .multiplicity()) /
                                norm;
                    }
                } else {
                    assert(!(i & 1));
                    assert(info->basis[info->orbsym[i]].n ==
                           tensors[i]->info->n);
                    double norm = 0;
                    for (int j = 0; j < tensors[i]->info->n; j++)
                        norm += tensors[i]->info->quanta[j].multiplicity();
                    norm = sqrt(norm);
                    for (int j = 0; j < tensors[i]->info->n; j++) {
                        assert((*tensors[i])[j].size() == 1);
                        (*tensors[i])[j](0, 0) =
                            sqrt(tensors[i]->info->quanta[j].multiplicity()) /
                            norm;
                    }
                }
            }
    }
    void canonicalize() {
        for (int i = 0; i < center; i++) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> tmat = make_shared<SparseMatrix<S>>();
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>();
            tmat_info->initialize(info->left_dims[i + 1],
                                  info->left_dims[i + 1], info->vaccum, false);
            tmat->allocate(tmat_info);
            tensors[i]->left_canonicalize(tmat);
            StateInfo<S> l = info->left_dims[i + 1],
                         m = info->basis[info->orbsym[i + 1]];
            StateInfo<S> lm = StateInfo<S>::tensor_product(
                             l, m, info->left_dims_fci[i + 2]),
                         r;
            StateInfo<S> lmc = StateInfo<S>::get_connection_info(l, m, lm);
            if (i + 1 == center && dot == 1)
                r = info->right_dims[center + dot];
            else if (i + 1 == center && dot == 2)
                r = StateInfo<S>::tensor_product(
                    info->basis[info->orbsym[center + 1]],
                    info->right_dims[center + dot],
                    info->right_dims_fci[center + 1]);
            else
                r = info->left_dims[i + 2];
            tensors[i + 1]->left_multiply(tmat, l, m, r, lm, lmc);
            if (i + 1 == center && dot == 2)
                r.deallocate();
            lmc.deallocate();
            lm.deallocate();
            tmat_info->deallocate();
            tmat->deallocate();
        }
        for (int i = n_sites - 1; i >= center + dot; i--) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> tmat = make_shared<SparseMatrix<S>>();
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>();
            tmat_info->initialize(info->right_dims[i], info->right_dims[i],
                                  info->vaccum, false);
            tmat->allocate(tmat_info);
            tensors[i]->right_canonicalize(tmat);
            if (dot == 1 && i - 1 == center) {
                shared_ptr<SparseMatrix<S>> tmp =
                    make_shared<SparseMatrix<S>>();
                tmp->allocate(tensors[i - 1]->info);
                tmp->copy_data_from(*tensors[i - 1]);
                tensors[i - 1]->contract(tmp, tmat);
                tmp->deallocate();
            } else {
                StateInfo<S> m = info->basis[info->orbsym[i - 1]],
                             r = info->right_dims[i];
                StateInfo<S> mr = StateInfo<S>::tensor_product(
                    m, r, info->right_dims_fci[i - 1]);
                StateInfo<S> mrc = StateInfo<S>::get_connection_info(m, r, mr);
                StateInfo<S> l;
                if (i - 1 == center + 1 && dot == 2) {
                    l = StateInfo<S>::tensor_product(
                        info->left_dims[center],
                        info->basis[info->orbsym[center]],
                        info->left_dims_fci[center + 1]);
                    tensors[i - 2]->right_multiply(tmat, l, m, r, mr, mrc);
                } else {
                    l = info->right_dims[i - 1];
                    tensors[i - 1]->right_multiply(tmat, l, m, r, mr, mrc);
                }
                if (i - 1 == center + 1 && dot == 2)
                    l.deallocate();
                mrc.deallocate();
                mr.deallocate();
            }
            tmat_info->deallocate();
            tmat->deallocate();
        }
    }
    void random_canonicalize() {
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                shared_ptr<SparseMatrix<S>> tmat =
                    make_shared<SparseMatrix<S>>();
                shared_ptr<SparseMatrixInfo<S>> tmat_info =
                    make_shared<SparseMatrixInfo<S>>();
                tensors[i]->randomize();
                if (i < center) {
                    tmat_info->initialize(info->left_dims[i + 1],
                                          info->left_dims[i + 1], info->vaccum,
                                          false);
                    tmat->allocate(tmat_info);
                    tensors[i]->left_canonicalize(tmat);
                } else if (i > center) {
                    tmat_info->initialize(info->right_dims[i],
                                          info->right_dims[i], info->vaccum,
                                          false);
                    tmat->allocate(tmat_info);
                    tensors[i]->right_canonicalize(tmat);
                }
                if (i != center) {
                    tmat_info->deallocate();
                    tmat->deallocate();
                }
            }
    }
    string get_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MPS." << info->tag
           << "." << Parsing::to_string(i);
        return ss.str();
    }
    void load_data() {
        ifstream ifs(get_filename(-1).c_str(), ios::binary);
        ifs.read((char *)&n_sites, sizeof(n_sites));
        ifs.read((char *)&center, sizeof(center));
        ifs.read((char *)&dot, sizeof(dot));
        canonical_form = string(n_sites, ' ');
        ifs.read((char *)&canonical_form[0], sizeof(char) * n_sites);
        vector<uint8_t> bs(n_sites);
        ifs.read((char *)&bs[0], sizeof(uint8_t) * n_sites);
        ifs.close();
        tensors.resize(n_sites, nullptr);
        for (int i = 0; i < n_sites; i++)
            if (bs[i])
                tensors[i] = make_shared<SparseMatrix<S>>();
    }
    void save_data() const {
        ofstream ofs(get_filename(-1).c_str(), ios::binary);
        ofs.write((char *)&n_sites, sizeof(n_sites));
        ofs.write((char *)&center, sizeof(center));
        ofs.write((char *)&dot, sizeof(dot));
        ofs.write((char *)&canonical_form[0], sizeof(char) * n_sites);
        vector<uint8_t> bs(n_sites);
        for (int i = 0; i < n_sites; i++)
            bs[i] = uint8_t(tensors[i] != nullptr);
        ofs.write((char *)&bs[0], sizeof(uint8_t) * n_sites);
        ofs.close();
    }
    void load_mutable() const {
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr)
                tensors[i]->load_data(get_filename(i), true);
    }
    void save_mutable() const {
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr)
                tensors[i]->save_data(get_filename(i), true);
    }
    void save_tensor(int i) const {
        assert(tensors[i] != nullptr);
        tensors[i]->save_data(get_filename(i), true);
    }
    void load_tensor(int i) {
        assert(tensors[i] != nullptr);
        tensors[i]->load_data(get_filename(i), true);
    }
    void unload_tensor(int i) {
        assert(tensors[i] != nullptr);
        tensors[i]->info->deallocate();
        tensors[i]->deallocate();
    }
    void deallocate() {
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->deallocate();
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->info->deallocate();
    }
};

} // namespace block2
