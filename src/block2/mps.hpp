
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
    if (!ifs.good())
        throw runtime_error("read_occ on '" + filename + "' failed.");
    vector<string> lines = Parsing::readlines(&ifs);
    if (ifs.bad())
        throw runtime_error("read_occ on '" + filename + "' failed.");
    ifs.close();
    assert(lines.size() >= 1);
    vector<string> vals = Parsing::split(lines[0], " ", true);
    vector<double> r;
    transform(vals.begin(), vals.end(), back_inserter(r), Parsing::to_double);
    return r;
}

// Write occupation numbers to a file
inline void write_occ(const string &filename, const vector<double> &occ) {
    ofstream ofs(filename.c_str());
    if (!ofs.good())
        throw runtime_error("write_occ on '" + filename + "' failed.");
    ofs << fixed << setprecision(8);
    for (auto x : occ)
        ofs << setw(12) << x;
    ofs << endl;
    if (!ofs.good())
        throw runtime_error("write_occ on '" + filename + "' failed.");
    ofs.close();
}

enum struct WarmUpTypes : uint8_t { None, Local, Determinant };

enum struct MultiTypes : uint8_t { None, Multi };

// Quantum number information in a MPS
template <typename S> struct MPSInfo {
    int n_sites;
    S vacuum;
    S target;
    ubond_t bond_dim;
    // States in each site
    vector<shared_ptr<StateInfo<S>>> basis;
    // Maximal possible states for left/right block (may be equal to/smaller
    // than FCI space)
    vector<shared_ptr<StateInfo<S>>> left_dims_fci, right_dims_fci;
    // Actual (truncated) states for left/right block
    vector<shared_ptr<StateInfo<S>>> left_dims, right_dims;
    string tag = "KET";
    MPSInfo(int n_sites, S vacuum, S target,
            const vector<shared_ptr<StateInfo<S>>> &basis, bool init_fci = true)
        : n_sites(n_sites), vacuum(vacuum), target(target), basis(basis),
          bond_dim(0) {
        left_dims_fci.resize(n_sites + 1);
        right_dims_fci.resize(n_sites + 1);
        left_dims.resize(n_sites + 1);
        right_dims.resize(n_sites + 1);
        if (init_fci)
            set_bond_dimension_fci();
        for (int i = 0; i <= n_sites; i++)
            left_dims[i] = make_shared<StateInfo<S>>();
        for (int i = n_sites; i >= 0; i--)
            right_dims[i] = make_shared<StateInfo<S>>();
    }
    virtual ~MPSInfo() = default;
    virtual AncillaTypes get_ancilla_type() const { return AncillaTypes::None; }
    virtual WarmUpTypes get_warm_up_type() const { return WarmUpTypes::None; }
    virtual MultiTypes get_multi_type() const { return MultiTypes::None; }
    virtual vector<S> get_complementary(S q) const {
        return vector<S>{target - q};
    }
    virtual void set_bond_dimension_fci() {
        left_dims_fci[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *left_dims_fci[i], *basis[i], target));
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *basis[i], *right_dims_fci[i + 1], target));
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<S>::filter(*left_dims_fci[i], *right_dims_fci[i], target);
            StateInfo<S>::filter(*right_dims_fci[i], *left_dims_fci[i], target);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i]->collect();
    }
    // set up initial mps using integral HF occupation numbers
    // construct local FCI space using n_local nearest sites
    void set_bond_dimension_using_hf(uint16_t m, const vector<double> &occ,
                                     int n_local = 0) {
        bond_dim = m;
        assert(occ.size() == n_sites);
        S occupied;
        for (int i = 0; i < basis[0]->n; i++)
            if (basis[0]->quanta[i].n() == 2) {
                occupied = basis[0]->quanta[i];
                break;
            }
        for (auto x : occ)
            assert(x == 2 || x == 0);
        vector<shared_ptr<StateInfo<S>>> left_dims_hf(n_sites + 1);
        vector<shared_ptr<StateInfo<S>>> right_dims_hf(n_sites + 1);
        left_dims_hf[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_hf[i + 1] = make_shared<StateInfo<S>>(
                left_dims_hf[i]->quanta[0] + (occ[i] == 2 ? occupied : vacuum));
        right_dims_hf[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_hf[i] =
                make_shared<StateInfo<S>>((occ[i] == 2 ? occupied : vacuum) +
                                          right_dims_hf[i + 1]->quanta[0]);
        left_dims[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0, j; i < n_sites; i++) {
            vector<StateInfo<S>> tmps;
            j = max(0, i - n_local + 1);
            tmps.push_back(left_dims_hf[j]->deep_copy());
            for (; j <= i; j++) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    tmps.back(), *basis[j], *left_dims_fci[j + 1]);
                tmps.push_back(tmp);
                tmps.back().reduce_n_states(m);
            }
            for (size_t k = 0; k < tmps.size() - 1; k++)
                tmps[k].deallocate();
            left_dims[i + 1] = make_shared<StateInfo<S>>(tmps.back());
            if (i != n_sites - 1) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    *left_dims[i], *basis[i], *left_dims_fci[i + 1]);
                for (int j = 0; j < left_dims[i + 1]->n; j++) {
                    int k = tmp.find_state(left_dims[i + 1]->quanta[j]);
                    if (k == -1)
                        left_dims[i + 1]->n_states[j] = 0;
                    else
                        left_dims[i + 1]->n_states[j] =
                            min(tmp.n_states[k], left_dims[i + 1]->n_states[j]);
                }
                tmp.deallocate();
                left_dims[i + 1]->collect();
            }
        }
        right_dims[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1, j; i >= 0; i--) {
            vector<StateInfo<S>> tmps;
            j = min(n_sites - 1, i + n_local - 1);
            tmps.push_back(right_dims_hf[j + 1]->deep_copy());
            for (; j >= i; j--) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    *basis[j], tmps.back(), *right_dims_fci[j]);
                tmps.push_back(tmp);
                tmps.back().reduce_n_states(m);
            }
            for (size_t k = 0; k < tmps.size() - 1; k++)
                tmps[k].deallocate();
            right_dims[i] = make_shared<StateInfo<S>>(tmps.back());
            if (i != 0) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    *basis[i], *right_dims[i + 1], *right_dims_fci[i]);
                for (int j = 0; j < right_dims[i]->n; j++) {
                    int k = tmp.find_state(right_dims[i]->quanta[j]);
                    if (k == -1)
                        right_dims[i]->n_states[j] = 0;
                    else
                        right_dims[i]->n_states[j] =
                            min(tmp.n_states[k], right_dims[i]->n_states[j]);
                }
                tmp.deallocate();
                right_dims[i]->collect();
            }
        }
    }
    // set up initial mps using fractional occupation numbers
    void set_bond_dimension_using_occ(uint16_t m, const vector<double> &occ,
                                      double bias = 1.0) {
        bond_dim = m;
        // site state probabilities
        vector<shared_ptr<StateProbability<S>>> site_probs(n_sites);
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
            site_probs[i] = make_shared<StateProbability<S>>();
            site_probs[i]->allocate(basis[i]->n);
            for (int j = 0; j < basis[i]->n; j++) {
                site_probs[i]->quanta[j] = basis[i]->quanta[j];
                site_probs[i]->probs[j] = probs[basis[i]->quanta[j].n()];
            }
        }
        // left and right block probabilities
        vector<shared_ptr<StateProbability<S>>> left_probs(n_sites + 1);
        vector<shared_ptr<StateProbability<S>>> right_probs(n_sites + 1);
        left_probs[0] = make_shared<StateProbability<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_probs[i + 1] = make_shared<StateProbability<S>>(
                StateProbability<S>::tensor_product_no_collect(
                    *left_probs[i], *site_probs[i], *left_dims_fci[i + 1]));
        right_probs[n_sites] = make_shared<StateProbability<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_probs[i] = make_shared<StateProbability<S>>(
                StateProbability<S>::tensor_product_no_collect(
                    *site_probs[i], *right_probs[i + 1], *right_dims_fci[i]));
        // conditional probabilities
        for (int i = 0; i <= n_sites; i++) {
            double lprobs[left_probs[i]->n], rprobs[right_probs[i]->n];
            for (int j = 0; j < left_probs[i]->n; j++)
                lprobs[j] = left_probs[i]->probs[j] *
                            left_probs[i]->quanta[j].multiplicity();
            for (int j = 0; j < right_probs[i]->n; j++)
                rprobs[j] = right_probs[i]->probs[j] *
                            right_probs[i]->quanta[j].multiplicity();
            for (int j = 0; i > 0 && j < left_probs[i]->n; j++) {
                if (left_probs[i]->probs[j] == 0)
                    continue;
                double x = 0;
                vector<S> rkss = get_complementary(left_probs[i]->quanta[j]);
                for (auto rks : rkss)
                    for (int k = 0, ik; k < rks.count(); k++)
                        if ((ik = right_probs[i]->find_state(rks[k])) != -1)
                            x += rprobs[ik];
                left_probs[i]->probs[j] *= x;
            }
            for (int j = 0; i < n_sites && j < right_probs[i]->n; j++) {
                if (right_probs[i]->probs[j] == 0)
                    continue;
                double x = 0;
                vector<S> lkss = get_complementary(right_probs[i]->quanta[j]);
                for (auto lks : lkss)
                    for (int k = 0, ik; k < lks.count(); k++)
                        if ((ik = left_probs[i]->find_state(lks[k])) != -1)
                            x += lprobs[ik];
                right_probs[i]->probs[j] *= x;
            }
        }
        // adjusted temparary fci dims
        vector<shared_ptr<StateInfo<S>>> left_dims_fci_t(n_sites + 1);
        vector<shared_ptr<StateInfo<S>>> right_dims_fci_t(n_sites + 1);
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims_fci_t[i] =
                make_shared<StateInfo<S>>(left_dims_fci[i]->deep_copy());
            right_dims_fci_t[i] =
                make_shared<StateInfo<S>>(right_dims_fci[i]->deep_copy());
        }
        // left and right block dims
        left_dims[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 1; i <= n_sites; i++) {
            left_dims[i]->allocate(left_probs[i]->n);
            memcpy(left_dims[i]->quanta, left_probs[i]->quanta,
                   sizeof(S) * left_probs[i]->n);
            double prob_sum =
                accumulate(left_probs[i]->probs,
                           left_probs[i]->probs + left_probs[i]->n, 0.0);
            for (int j = 0; j < left_probs[i]->n; j++)
                left_dims[i]->n_states[j] =
                    min((ubond_t)round(left_probs[i]->probs[j] / prob_sum * m),
                        left_dims_fci_t[i]->n_states[j]);
            left_dims[i]->collect();
            if (i != n_sites) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    *left_dims[i], *basis[i], *left_dims_fci_t[i + 1]);
                for (int j = 0, k; j < left_dims_fci_t[i + 1]->n; j++)
                    if ((k = tmp.find_state(
                             left_dims_fci_t[i + 1]->quanta[j])) != -1)
                        left_dims_fci_t[i + 1]->n_states[j] =
                            min(tmp.n_states[k],
                                left_dims_fci_t[i + 1]->n_states[j]);
                for (int j = 0; j < left_probs[i + 1]->n; j++)
                    if (tmp.find_state(left_probs[i + 1]->quanta[j]) == -1)
                        left_probs[i + 1]->probs[j] = 0;
                tmp.deallocate();
            }
        }
        right_dims[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--) {
            right_dims[i]->allocate(right_probs[i]->n);
            memcpy(right_dims[i]->quanta, right_probs[i]->quanta,
                   sizeof(S) * right_probs[i]->n);
            double prob_sum =
                accumulate(right_probs[i]->probs,
                           right_probs[i]->probs + right_probs[i]->n, 0.0);
            for (int j = 0; j < right_probs[i]->n; j++)
                right_dims[i]->n_states[j] =
                    min((ubond_t)round(right_probs[i]->probs[j] / prob_sum * m),
                        right_dims_fci_t[i]->n_states[j]);
            right_dims[i]->collect();
            if (i != 0) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    *basis[i - 1], *right_dims[i], *right_dims_fci_t[i - 1]);
                for (int j = 0, k; j < right_dims_fci_t[i - 1]->n; j++)
                    if ((k = tmp.find_state(
                             right_dims_fci_t[i - 1]->quanta[j])) != -1)
                        right_dims_fci_t[i - 1]->n_states[j] =
                            min(tmp.n_states[k],
                                right_dims_fci_t[i - 1]->n_states[j]);
                for (int j = 0; j < right_probs[i - 1]->n; j++)
                    if (tmp.find_state(right_probs[i - 1]->quanta[j]) == -1)
                        right_probs[i - 1]->probs[j] = 0;
                tmp.deallocate();
            }
        }
    }
    // set up bond dimension using FCI quantum numbers
    // each FCI quantum number has at least one state kept
    virtual void set_bond_dimension(ubond_t m) {
        bond_dim = m;
        left_dims[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims[i + 1] =
                make_shared<StateInfo<S>>(left_dims_fci[i + 1]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            if (left_dims[i + 1]->n_states_total > m) {
                int new_total = 0;
                for (int k = 0; k < left_dims[i + 1]->n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)left_dims[i + 1]->n_states[k] *
                                        m / left_dims[i + 1]->n_states_total) +
                                   0.1);
                    left_dims[i + 1]->n_states[k] = (ubond_t)min(
                        new_n_states, (uint32_t)numeric_limits<ubond_t>::max());
                    new_total += left_dims[i + 1]->n_states[k];
                }
                left_dims[i + 1]->n_states_total = new_total;
            }
        right_dims[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims[i] =
                make_shared<StateInfo<S>>(right_dims_fci[i]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            if (right_dims[i]->n_states_total > m) {
                int new_total = 0;
                for (int k = 0; k < right_dims[i]->n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)right_dims[i]->n_states[k] * m /
                                        right_dims[i]->n_states_total) +
                                   0.1);
                    right_dims[i]->n_states[k] = (ubond_t)min(
                        new_n_states, (uint32_t)numeric_limits<ubond_t>::max());
                    new_total += right_dims[i]->n_states[k];
                }
                right_dims[i]->n_states_total = new_total;
            }
        for (int i = -1; i < n_sites - 1; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *left_dims[i + 1], *basis[i + 1], *left_dims_fci[i + 2]);
            int new_total = 0;
            for (int k = 0; k < left_dims[i + 2]->n; k++) {
                int tk = t.find_state(left_dims[i + 2]->quanta[k]);
                if (tk == -1)
                    left_dims[i + 2]->n_states[k] = 0;
                else if (left_dims[i + 2]->n_states[k] > t.n_states[tk])
                    left_dims[i + 2]->n_states[k] = t.n_states[tk];
                new_total += left_dims[i + 2]->n_states[k];
            }
            left_dims[i + 2]->n_states_total = new_total;
            t.deallocate();
        }
        for (int i = n_sites; i > 0; i--) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *basis[i - 1], *right_dims[i], *right_dims_fci[i - 1]);
            int new_total = 0;
            for (int k = 0; k < right_dims[i - 1]->n; k++) {
                int tk = t.find_state(right_dims[i - 1]->quanta[k]);
                if (tk == -1)
                    right_dims[i - 1]->n_states[k] = 0;
                else if (right_dims[i - 1]->n_states[k] > t.n_states[tk])
                    right_dims[i - 1]->n_states[k] = t.n_states[tk];
                new_total += right_dims[i - 1]->n_states[k];
            }
            right_dims[i - 1]->n_states_total = new_total;
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
        if (frame->prefix_can_write)
            for (int i = 0; i < n_sites + 1; i++) {
                left_dims[i]->save_data(get_filename(true, i));
                right_dims[i]->save_data(get_filename(false, i));
            }
    }
    void load_mutable_left() const {
        for (int i = 0; i <= n_sites; i++)
            left_dims[i]->load_data(get_filename(true, i));
    }
    void load_mutable_right() const {
        for (int i = n_sites; i >= 0; i--)
            right_dims[i]->load_data(get_filename(false, i));
    }
    void load_mutable() const {
        load_mutable_left();
        load_mutable_right();
    }
    void deallocate_mutable() {
        for (int i = 0; i <= n_sites; i++)
            right_dims[i]->deallocate();
        for (int i = n_sites; i >= 0; i--)
            left_dims[i]->deallocate();
    }
    void save_left_dims(int i) const {
        if (frame->prefix_can_write)
            left_dims[i]->save_data(get_filename(true, i));
    }
    void save_right_dims(int i) const {
        if (frame->prefix_can_write)
            right_dims[i]->save_data(get_filename(false, i));
    }
    void load_left_dims(int i) {
        left_dims[i]->load_data(get_filename(true, i));
    }
    void load_right_dims(int i) {
        right_dims[i]->load_data(get_filename(false, i));
    }
    void deallocate_left() {
        for (int i = n_sites; i >= 0; i--)
            left_dims_fci[i]->deallocate();
    }
    void deallocate_right() {
        for (int i = 0; i <= n_sites; i++)
            right_dims_fci[i]->deallocate();
    }
    void deallocate() {
        deallocate_right();
        deallocate_left();
    }
};

// Quantum number infomation in a MPS
// Used for warm-up sweep
template <typename S> struct DynamicMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::target;
    using MPSInfo<S>::bond_dim;
    using MPSInfo<S>::basis;
    using MPSInfo<S>::left_dims_fci;
    using MPSInfo<S>::right_dims_fci;
    using MPSInfo<S>::left_dims;
    using MPSInfo<S>::right_dims;
    vector<uint8_t> iocc;
    uint16_t n_local = 0; // number of nearset sites using FCI quantum numbers
    DynamicMPSInfo(int n_sites, S vacuum, S target,
                   const vector<shared_ptr<StateInfo<S>>> &basis,
                   const vector<uint8_t> &iocc, bool init_fci = true)
        : iocc(iocc), MPSInfo<S>(n_sites, vacuum, target, basis, init_fci) {}
    WarmUpTypes get_warm_up_type() const override { return WarmUpTypes::Local; }
    void set_bond_dimension(ubond_t m) override {
        bond_dim = m;
        left_dims[0] = make_shared<StateInfo<S>>(vacuum);
        right_dims[n_sites] = make_shared<StateInfo<S>>(vacuum);
    }
    void set_left_bond_dimension_local(uint16_t i, bool match_prev = false) {
        left_dims[0] = make_shared<StateInfo<S>>(vacuum);
        int j = max(0, i - n_local + 1);
        for (int k = 0; k < j; k++)
            left_dims[k + 1] = make_shared<StateInfo<S>>(
                left_dims[k]->quanta[0] + basis[k]->quanta[iocc[k]]);
        for (int k = j; k <= i; k++) {
            StateInfo<S> x = StateInfo<S>::tensor_product(
                *left_dims[k], *basis[k], *left_dims_fci[k + 1]);
            x.reduce_n_states(bond_dim);
            if (match_prev) {
                this->load_left_dims(k + 1);
                for (int l = 0, il; l < left_dims[k + 1]->n; l++) {
                    assert((il = x.find_state(left_dims[k + 1]->quanta[l])) !=
                           -1);
                    x.n_states[il] =
                        max(x.n_states[il], left_dims[k + 1]->n_states[l]);
                }
                left_dims[k + 1]->deallocate();
            }
            left_dims[k + 1] = make_shared<StateInfo<S>>(x);
        }
        for (int k = i + 1; k < n_sites; k++)
            left_dims[k + 1]->n = 0;
    }
    void set_right_bond_dimension_local(uint16_t i, bool match_prev = false) {
        right_dims[n_sites] = make_shared<StateInfo<S>>(vacuum);
        int j = min(n_sites - 1, i + n_local - 1);
        for (int k = n_sites - 1; k > j; k--)
            right_dims[k] = make_shared<StateInfo<S>>(
                basis[k]->quanta[iocc[k]] + right_dims[k + 1]->quanta[0]);
        for (int k = j; k >= i; k--) {
            StateInfo<S> x = StateInfo<S>::tensor_product(
                *basis[k], *right_dims[k + 1], *right_dims_fci[k]);
            x.reduce_n_states(bond_dim);
            if (match_prev) {
                this->load_right_dims(k);
                for (int l = 0, il; l < right_dims[k]->n; l++) {
                    assert((il = x.find_state(right_dims[k]->quanta[l])) != -1);
                    x.n_states[il] =
                        max(x.n_states[il], right_dims[k]->n_states[l]);
                }
                right_dims[k]->deallocate();
            }
            right_dims[k] = make_shared<StateInfo<S>>(x);
        }
        for (int k = i - 1; k >= 0; k--)
            right_dims[k]->n = 0;
    }
};

enum struct ActiveTypes : uint8_t { Empty, Active, Frozen };

// MPSInfo for CASCI calculation
template <typename S> struct CASCIMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::target;
    using MPSInfo<S>::bond_dim;
    using MPSInfo<S>::basis;
    using MPSInfo<S>::left_dims_fci;
    using MPSInfo<S>::right_dims_fci;
    vector<ActiveTypes> casci_mask;
    static vector<ActiveTypes> active_space(int n_sites, S target,
                                            int n_active_sites,
                                            int n_active_electrons) {
        vector<ActiveTypes> casci_mask(n_sites, ActiveTypes::Empty);
        assert(!((target.n() - n_active_electrons) & 1));
        int n_frozen = (target.n() - n_active_electrons) >> 1;
        assert(n_frozen + n_active_sites <= n_sites);
        for (size_t i = 0; i < n_frozen; i++)
            casci_mask[i] = ActiveTypes::Frozen;
        for (size_t i = n_frozen; i < n_frozen + n_active_sites; i++)
            casci_mask[i] = ActiveTypes::Active;
        return casci_mask;
    }
    CASCIMPSInfo(int n_sites, S vacuum, S target,
                 const vector<shared_ptr<StateInfo<S>>> &basis,
                 const vector<ActiveTypes> &casci_mask, bool init_fci = true)
        : casci_mask(casci_mask), MPSInfo<S>(n_sites, vacuum, target, basis,
                                             false) {
        if (init_fci)
            set_bond_dimension_fci();
    }
    CASCIMPSInfo(int n_sites, S vacuum, S target,
                 const vector<shared_ptr<StateInfo<S>>> &basis,
                 int n_active_sites, int n_active_electrons,
                 bool init_fci = true)
        : casci_mask(active_space(n_sites, target, n_active_sites,
                                  n_active_electrons)),
          MPSInfo<S>(n_sites, vacuum, target, basis, false) {
        if (init_fci)
            set_bond_dimension_fci();
    }
    void set_bond_dimension_fci() override {
        assert(casci_mask.size() == n_sites);
        StateInfo<S> empty = StateInfo<S>(vacuum);
        S frozen_state;
        // currently only works with symmetrized basis
        for (int i = 0; i < basis[0]->n; i++)
            if (basis[0]->quanta[i].n() == 2) {
                frozen_state = basis[0]->quanta[i];
                break;
            }
        StateInfo<S> frozen = StateInfo<S>(frozen_state);
        left_dims_fci[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            switch (casci_mask[i]) {
            case ActiveTypes::Active:
                left_dims_fci[i + 1] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        *left_dims_fci[i], *basis[i], target));
                break;
            case ActiveTypes::Frozen:
                left_dims_fci[i + 1] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        *left_dims_fci[i], frozen, target));
                break;
            case ActiveTypes::Empty:
                left_dims_fci[i + 1] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        *left_dims_fci[i], empty, target));
                break;
            default:
                assert(false);
                break;
            }
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            switch (casci_mask[i]) {
            case ActiveTypes::Active:
                right_dims_fci[i] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        *basis[i], *right_dims_fci[i + 1], target));
                break;
            case ActiveTypes::Frozen:
                right_dims_fci[i] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        frozen, *right_dims_fci[i + 1], target));
                break;
            case ActiveTypes::Empty:
                right_dims_fci[i] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        empty, *right_dims_fci[i + 1], target));
                break;
            default:
                assert(false);
                break;
            }
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<S>::filter(*left_dims_fci[i], *right_dims_fci[i], target);
            StateInfo<S>::filter(*right_dims_fci[i], *left_dims_fci[i], target);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i]->collect();
    }
};

/** Restrict quantum numbers to describe an uncontracted MRCI wavefunction.
 *
 * The last *right* n_ext sites are restricted to have only up to ci_order
 * electrons. I.e., ci_order = 2 gives MR-CISD.
 * @author: Henrik R. Larsson <larsson@caltech.edu>
 */
template <typename S> struct MRCIMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::left_dims_fci; // Resolve names of template base class
    using MPSInfo<S>::right_dims_fci;
    using MPSInfo<S>::left_dims;
    using MPSInfo<S>::right_dims;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::target;
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::basis;
    int n_ext;    //!> Number of external orbitals: CI space
    int ci_order; //!> Up to how many electrons are allowed in ext. orbitals: 2
                  //! gives MR-CISD
    MRCIMPSInfo(int n_sites, int n_ext, int ci_order, S vacuum, S target,
                const vector<shared_ptr<StateInfo<S>>> &basis,
                bool init_fci = true)
        : MPSInfo<S>{n_sites, vacuum, target, basis, false}, n_ext{n_ext},
          ci_order{ci_order} {
        assert(n_ext < n_sites);
        if (init_fci)
            set_bond_dimension_fci();
    }
    void set_bond_dimension(ubond_t m) override {
        MPSInfo<S>::set_bond_dimension(m); // call base class method
        // zero states may occur
        for (int i = 0; i <= n_sites; i++)
            left_dims[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims[i]->collect();
    }
    void set_bond_dimension_fci() override {
        // Same as in the base class: Create left/right fci dims w/o
        // restrictions
        left_dims_fci[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *left_dims_fci[i], *basis[i], target));
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *basis[i], *right_dims_fci[i + 1], target));
        // Now, restrict right_dims_fci
        for (int i = n_sites - n_ext; i < n_sites; ++i) {
            auto &state_info = right_dims_fci[i];
            for (int q = 0; q < state_info->n; ++q) {
                if (state_info->quanta[q].n() > ci_order) {
                    state_info->n_states[q] = 0;
                }
            }
        }
        // vv same as in base class: Take intersection of left/right fci dims
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<S>::filter(*left_dims_fci[i], *right_dims_fci[i], target);
            StateInfo<S>::filter(*right_dims_fci[i], *left_dims_fci[i], target);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i]->collect();
    }
};

// Adding tensors for ancilla sites to a MPS
// n_sites = 2 * n_physical_sites
template <typename S> struct AncillaMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::target;
    using MPSInfo<S>::bond_dim;
    using MPSInfo<S>::basis;
    using MPSInfo<S>::left_dims;
    using MPSInfo<S>::right_dims;
    using MPSInfo<S>::left_dims_fci;
    using MPSInfo<S>::right_dims_fci;
    int n_physical_sites;
    static vector<shared_ptr<StateInfo<S>>>
    trans_basis(const vector<shared_ptr<StateInfo<S>>> &a, int n_sites) {
        vector<shared_ptr<StateInfo<S>>> b(n_sites << 1, nullptr);
        for (int i = 0, j = 0; i < n_sites; i++, j += 2)
            b[j] = b[j + 1] = a[i];
        return b;
    }
    AncillaMPSInfo(int n_sites, S vacuum, S target,
                   const vector<shared_ptr<StateInfo<S>>> &basis,
                   bool init_fci = true)
        : n_physical_sites(n_sites), MPSInfo<S>(n_sites << 1, vacuum, target,
                                                trans_basis(basis, n_sites),
                                                init_fci) {}
    AncillaTypes get_ancilla_type() const override {
        return AncillaTypes::Ancilla;
    }
    void set_thermal_limit() {
        left_dims[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            if (i & 1) {
                S q = left_dims[i]->quanta[left_dims[i]->n - 1] +
                      basis[i]->quanta[0];
                assert(q.count() == 1);
                left_dims[i + 1] = make_shared<StateInfo<S>>(q);
            } else
                left_dims[i + 1] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        *left_dims[i], *basis[i], target));
        right_dims[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            if (i & 1)
                right_dims[i] =
                    make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                        *basis[i], *right_dims[i + 1], target));
            else {
                S q = basis[i]->quanta[0] +
                      right_dims[i + 1]->quanta[right_dims[i + 1]->n - 1];
                assert(q.count() == 1);
                right_dims[i] = make_shared<StateInfo<S>>(q);
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
        if (center >= 0 && center < n_sites)
            for (int i = center; i < center + dot; i++)
                canonical_form[i] = 'C';
        for (int i = center + dot; i < n_sites; i++)
            canonical_form[i] = 'R';
    }
    virtual ~MPS() = default;
    void initialize_left(const shared_ptr<MPSInfo<S>> &info, int i_right) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        this->info = info;
        vector<shared_ptr<SparseMatrixInfo<S>>> mat_infos;
        mat_infos.resize(n_sites);
        tensors.resize(n_sites);
        for (int i = 0; i <= i_right; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *info->left_dims[i], *info->basis[i],
                *info->left_dims_fci[i + 1]);
            mat_infos[i] = make_shared<SparseMatrixInfo<S>>(i_alloc);
            mat_infos[i]->initialize(t, *info->left_dims[i + 1], info->vacuum,
                                     false);
            tensors[i] = make_shared<SparseMatrix<S>>(d_alloc);
            tensors[i]->allocate(mat_infos[i]);
        }
    }
    void initialize_right(const shared_ptr<MPSInfo<S>> &info, int i_left) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        this->info = info;
        vector<shared_ptr<SparseMatrixInfo<S>>> mat_infos;
        mat_infos.resize(n_sites);
        tensors.resize(n_sites);
        for (int i = i_left; i < n_sites; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *info->basis[i], *info->right_dims[i + 1],
                *info->right_dims_fci[i]);
            mat_infos[i] = make_shared<SparseMatrixInfo<S>>(i_alloc);
            mat_infos[i]->initialize(*info->right_dims[i], t, info->vacuum,
                                     false);
            tensors[i] = make_shared<SparseMatrix<S>>(d_alloc);
            tensors[i]->allocate(mat_infos[i]);
        }
    }
    virtual void initialize(const shared_ptr<MPSInfo<S>> &info,
                            bool init_left = true, bool init_right = true) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        this->info = info;
        shared_ptr<SparseMatrixInfo<S>> mat_info;
        tensors.resize(n_sites);
        if (init_left)
            initialize_left(info, center - 1);
        if (center >= 0 && center < n_sites && (init_left || init_right)) {
            mat_info = make_shared<SparseMatrixInfo<S>>(i_alloc);
            if (dot == 1) {
                StateInfo<S> t = StateInfo<S>::tensor_product(
                    *info->left_dims[center], *info->basis[center],
                    *info->left_dims_fci[center + dot]);
                mat_info->initialize(t, *info->right_dims[center + dot],
                                     info->target, false, true);
                canonical_form[center] = 'K';
            } else {
                StateInfo<S> tl = StateInfo<S>::tensor_product(
                    *info->left_dims[center], *info->basis[center],
                    *info->left_dims_fci[center + 1]);
                StateInfo<S> tr = StateInfo<S>::tensor_product(
                    *info->basis[center + 1], *info->right_dims[center + dot],
                    *info->right_dims_fci[center + 1]);
                mat_info->initialize(tl, tr, info->target, false, true);
            }
            tensors[center] = make_shared<SparseMatrix<S>>(d_alloc);
            tensors[center]->allocate(mat_info);
        }
        if (init_right)
            initialize_right(info, center + dot);
    }
    void fill_thermal_limit() {
        assert(info->get_ancilla_type() == AncillaTypes::Ancilla);
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                if (i < center || i > center || (i == center && dot == 1)) {
                    int n = info->basis[i]->n;
                    assert(tensors[i]->total_memory == n);
                    if (i & 1)
                        for (int j = 0; j < n; j++)
                            tensors[i]->data[j] = 1;
                    else {
                        double norm = 0;
                        for (int j = 0; j < n; j++)
                            norm += info->basis[i]->quanta[j].multiplicity();
                        norm = sqrt(norm);
                        for (int j = 0; j < n; j++)
                            tensors[i]->data[j] =
                                sqrt(info->basis[i]->quanta[j].multiplicity()) /
                                norm;
                    }
                } else {
                    assert(!(i & 1));
                    assert(info->basis[i]->n == tensors[i]->info->n);
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
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        for (int i = 0; i < center; i++) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> tmat =
                make_shared<SparseMatrix<S>>(d_alloc);
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            tmat_info->initialize(*info->left_dims[i + 1],
                                  *info->left_dims[i + 1], info->vacuum, false);
            tmat->allocate(tmat_info);
            tensors[i]->left_canonicalize(tmat);
            StateInfo<S> l = *info->left_dims[i + 1], m = *info->basis[i + 1];
            StateInfo<S> lm = StateInfo<S>::tensor_product(
                             l, m, *info->left_dims_fci[i + 2]),
                         r;
            StateInfo<S> lmc = StateInfo<S>::get_connection_info(l, m, lm);
            if (i + 1 == center && dot == 1)
                r = *info->right_dims[center + dot];
            else if (i + 1 == center && dot == 2)
                r = StateInfo<S>::tensor_product(
                    *info->basis[center + 1], *info->right_dims[center + dot],
                    *info->right_dims_fci[center + 1]);
            else
                r = *info->left_dims[i + 2];
            assert(tensors[i + 1] != nullptr);
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
            shared_ptr<SparseMatrix<S>> tmat =
                make_shared<SparseMatrix<S>>(d_alloc);
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            tmat_info->initialize(*info->right_dims[i], *info->right_dims[i],
                                  info->vacuum, false);
            tmat->allocate(tmat_info);
            tensors[i]->right_canonicalize(tmat);
            if (dot == 1 && i - 1 == center) {
                shared_ptr<SparseMatrix<S>> tmp =
                    make_shared<SparseMatrix<S>>(d_alloc);
                tmp->allocate(tensors[i - 1]->info);
                tmp->copy_data_from(tensors[i - 1]);
                assert(tensors[i - 1] != nullptr);
                tensors[i - 1]->contract(tmp, tmat);
                tmp->deallocate();
            } else {
                StateInfo<S> m = *info->basis[i - 1], r = *info->right_dims[i];
                StateInfo<S> mr = StateInfo<S>::tensor_product(
                    m, r, *info->right_dims_fci[i - 1]);
                StateInfo<S> mrc = StateInfo<S>::get_connection_info(m, r, mr);
                StateInfo<S> l;
                if (i - 1 == center + 1 && dot == 2) {
                    l = StateInfo<S>::tensor_product(
                        *info->left_dims[center], *info->basis[center],
                        *info->left_dims_fci[center + 1]);
                    assert(tensors[i - 2] != nullptr);
                    tensors[i - 2]->right_multiply(tmat, l, m, r, mr, mrc);
                } else {
                    l = *info->right_dims[i - 1];
                    assert(tensors[i - 1] != nullptr);
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
    void random_canonicalize_tensor(int i) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        if (tensors[i] != nullptr) {
            shared_ptr<SparseMatrix<S>> tmat =
                make_shared<SparseMatrix<S>>(d_alloc);
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            tensors[i]->randomize();
            if (i < center) {
                tmat_info->initialize(*info->left_dims[i + 1],
                                      *info->left_dims[i + 1], info->vacuum,
                                      false);
                tmat->allocate(tmat_info);
                tensors[i]->left_canonicalize(tmat);
            } else if (i > center) {
                tmat_info->initialize(*info->right_dims[i],
                                      *info->right_dims[i], info->vacuum,
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
    virtual void random_canonicalize() {
        for (int i = 0; i < n_sites; i++)
            random_canonicalize_tensor(i);
    }
    virtual string get_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MPS." << info->tag
           << "." << Parsing::to_string(i);
        return ss.str();
    }
    void load_data_from(ifstream &ifs) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        ifs.read((char *)&n_sites, sizeof(n_sites));
        ifs.read((char *)&center, sizeof(center));
        ifs.read((char *)&dot, sizeof(dot));
        canonical_form = string(n_sites, ' ');
        ifs.read((char *)&canonical_form[0], sizeof(char) * n_sites);
        vector<uint8_t> bs(n_sites);
        ifs.read((char *)&bs[0], sizeof(uint8_t) * n_sites);
        tensors.resize(n_sites, nullptr);
        for (int i = 0; i < n_sites; i++)
            if (bs[i])
                tensors[i] = make_shared<SparseMatrix<S>>(d_alloc);
    }
    virtual void load_data() {
        ifstream ifs(get_filename(-1).c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPS::load_data on '" + get_filename(-1) +
                                "' failed.");
        load_data_from(ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPS::load_data on '" + get_filename(-1) +
                                "' failed.");
        ifs.close();
    }
    void save_data_to(ofstream &ofs) const {
        ofs.write((char *)&n_sites, sizeof(n_sites));
        ofs.write((char *)&center, sizeof(center));
        ofs.write((char *)&dot, sizeof(dot));
        ofs.write((char *)&canonical_form[0], sizeof(char) * n_sites);
        vector<uint8_t> bs(n_sites);
        for (int i = 0; i < n_sites; i++)
            bs[i] = uint8_t(tensors[i] != nullptr);
        ofs.write((char *)&bs[0], sizeof(uint8_t) * n_sites);
    }
    virtual void save_data() const {
        if (frame->prefix_can_write) {
            ofstream ofs(get_filename(-1).c_str(), ios::binary);
            if (!ofs.good())
                throw runtime_error("MPS::save_data on '" + get_filename(-1) +
                                    "' failed.");
            save_data_to(ofs);
            if (!ofs.good())
                throw runtime_error("MPS::save_data on '" + get_filename(-1) +
                                    "' failed.");
            ofs.close();
        }
    }
    void load_mutable_left() const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        for (int i = 0; i < center; i++)
            if (tensors[i] != nullptr) {
                tensors[i]->alloc = d_alloc;
                tensors[i]->load_data(get_filename(i), true, i_alloc);
            }
    }
    void load_mutable_right() const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        for (int i = center + dot; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                tensors[i]->alloc = d_alloc;
                tensors[i]->load_data(get_filename(i), true, i_alloc);
            }
    }
    virtual void load_mutable() const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                tensors[i]->alloc = d_alloc;
                tensors[i]->load_data(get_filename(i), true, i_alloc);
            }
    }
    virtual void save_mutable() const {
        if (frame->prefix_can_write)
            for (int i = 0; i < n_sites; i++)
                if (tensors[i] != nullptr)
                    tensors[i]->save_data(get_filename(i), true);
    }
    virtual void save_tensor(int i) const {
        if (frame->prefix_can_write) {
            assert(tensors[i] != nullptr);
            tensors[i]->save_data(get_filename(i), true);
        }
    }
    virtual void load_tensor(int i) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(tensors[i] != nullptr);
        tensors[i]->alloc = d_alloc;
        tensors[i]->load_data(get_filename(i), true, i_alloc);
    }
    virtual void unload_tensor(int i) {
        assert(tensors[i] != nullptr);
        tensors[i]->info->deallocate();
        tensors[i]->deallocate();
    }
    virtual void deallocate() {
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->deallocate();
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->info->deallocate();
    }
};

} // namespace block2
