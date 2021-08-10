
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

#include "../core/parallel_rule.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/state_info.hpp"
#include "../core/utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

/** Indicating whether a MPS/MPO includes ancilla sites. */
enum struct AncillaTypes : uint8_t {
    None,   //!< No ancilla sites.
    Ancilla //!< With ancilla sites.
};

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

enum struct MPSTypes : uint8_t { None = 0, MultiWfn = 1, MultiCenter = 2 };

inline bool operator&(MPSTypes a, MPSTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline MPSTypes operator|(MPSTypes a, MPSTypes b) {
    return MPSTypes((uint8_t)a | (uint8_t)b);
}

inline MPSTypes operator^(MPSTypes a, MPSTypes b) {
    return MPSTypes((uint8_t)a ^ (uint8_t)b);
}

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
    MPSInfo(int n_sites) : n_sites(n_sites), bond_dim(0) {}
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
    virtual MPSTypes get_type() const { return MPSTypes::None; }
    virtual void load_data(istream &ifs) {
        ifs.read((char *)&n_sites, sizeof(n_sites));
        ifs.read((char *)&vacuum, sizeof(vacuum));
        ifs.read((char *)&target, sizeof(target));
        ifs.read((char *)&bond_dim, sizeof(bond_dim));
        int ltag = 0;
        ifs.read((char *)&ltag, sizeof(ltag));
        tag = string(ltag, ' ');
        ifs.read((char *)&tag[0], sizeof(char) * ltag);
        basis.resize(n_sites);
        left_dims_fci.resize(n_sites + 1);
        right_dims_fci.resize(n_sites + 1);
        left_dims.resize(n_sites + 1);
        right_dims.resize(n_sites + 1);
        for (int i = 0; i < n_sites; i++) {
            basis[i] = make_shared<StateInfo<S>>();
            basis[i]->load_data(ifs);
        }
        vector<vector<shared_ptr<StateInfo<S>>> *> arrs = {&left_dims_fci,
                                                           &right_dims_fci};
        for (auto *arr : arrs)
            for (int i = 0; i <= n_sites; i++) {
                (*arr)[i] = make_shared<StateInfo<S>>();
                (*arr)[i]->load_data(ifs);
            }
        for (int i = 0; i <= n_sites; i++)
            left_dims[i] = make_shared<StateInfo<S>>();
        for (int i = n_sites; i >= 0; i--)
            right_dims[i] = make_shared<StateInfo<S>>();
    }
    void load_data(const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPSInfo:load_data on '" + filename +
                                "' failed.");
        load_data(ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPSInfo:load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    virtual void save_data(ostream &ofs) const {
        ofs.write((char *)&n_sites, sizeof(n_sites));
        ofs.write((char *)&vacuum, sizeof(vacuum));
        ofs.write((char *)&target, sizeof(target));
        ofs.write((char *)&bond_dim, sizeof(bond_dim));
        int ltag = (int)tag.size();
        ofs.write((char *)&ltag, sizeof(ltag));
        ofs.write((char *)&tag[0], sizeof(char) * ltag);
        assert((int)basis.size() == n_sites);
        for (int i = 0; i < n_sites; i++) {
            assert(basis[i] != nullptr);
            basis[i]->save_data(ofs);
        }
        vector<const vector<shared_ptr<StateInfo<S>>> *> arrs = {
            &left_dims_fci, &right_dims_fci};
        for (const auto *arr : arrs) {
            assert((int)arr->size() == n_sites + 1);
            for (int i = 0; i <= n_sites; i++) {
                assert((*arr)[i] != nullptr);
                (*arr)[i]->save_data(ofs);
            }
        }
    }
    void save_data(const string &filename) const {
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPSInfo:save_data on '" + filename +
                                "' failed.");
        save_data(ofs);
        if (!ofs.good())
            throw runtime_error("MPSInfo:save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    virtual vector<S> get_complementary(S q) const {
        return vector<S>{target - q};
    }
    // normal case is left_vacuum == right_vacuum == vacuum
    // only for singlet embedding left_vacuum is not vacuum
    virtual void set_bond_dimension_full_fci(S left_vacuum = S(S::invalid),
                                             S right_vacuum = S(S::invalid)) {
        left_dims_fci[0] = make_shared<StateInfo<S>>(
            left_vacuum == S(S::invalid) ? vacuum : left_vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *left_dims_fci[i], *basis[i], target));
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(
            right_vacuum == S(S::invalid) ? vacuum : right_vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *basis[i], *right_dims_fci[i + 1], target));
    }
    virtual void set_bond_dimension_fci(S left_vacuum = S(S::invalid),
                                        S right_vacuum = S(S::invalid)) {
        set_bond_dimension_full_fci(left_vacuum, right_vacuum);
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
        vector<vector<vector<double>>> site_prefs(n_sites);
        assert(occ.size() == n_sites || occ.size() == n_sites * 2 ||
               occ.size() == n_sites * 4);
        for (int i = 0; i < n_sites; i++) {
            site_probs[i] = make_shared<StateProbability<S>>();
            site_probs[i]->allocate(basis[i]->n);
            if (occ.size() == n_sites || occ.size() == n_sites * 2) {
                double alpha_occ, beta_occ;
                if (occ.size() == n_sites) {
                    alpha_occ = occ[i];
                    if (bias != 1.0) {
                        if (alpha_occ > 1)
                            alpha_occ = 1 + pow(alpha_occ - 1, bias);
                        else if (alpha_occ < 1)
                            alpha_occ = 1 - pow(1 - alpha_occ, bias);
                    }
                    alpha_occ /= 2;
                    beta_occ = alpha_occ;
                } else {
                    alpha_occ = occ[2 * i] * 2, beta_occ = occ[2 * i + 1] * 2;
                    if (bias != 1.0) {
                        if (alpha_occ > 1)
                            alpha_occ = 1 + pow(alpha_occ - 1, bias);
                        else if (alpha_occ < 1)
                            alpha_occ = 1 - pow(1 - alpha_occ, bias);
                        if (beta_occ > 1)
                            beta_occ = 1 + pow(beta_occ - 1, bias);
                        else if (beta_occ < 1)
                            beta_occ = 1 - pow(1 - beta_occ, bias);
                    }
                    alpha_occ /= 2;
                    beta_occ /= 2;
                }
                assert(0 <= alpha_occ && alpha_occ <= 1);
                assert(0 <= beta_occ && beta_occ <= 1);
                for (int j = 0; j < basis[i]->n; j++) {
                    site_probs[i]->quanta[j] = basis[i]->quanta[j];
                    if (basis[i]->quanta[j].n() == 0)
                        site_probs[i]->probs[j] =
                            (1 - alpha_occ) * (1 - beta_occ);
                    else if (basis[i]->quanta[j].n() == 2)
                        site_probs[i]->probs[j] = alpha_occ * beta_occ;
                    else if (basis[i]->quanta[j].n() == 1 &&
                             basis[i]->quanta[j].twos() == 1)
                        site_probs[i]->probs[j] = alpha_occ * (1 - beta_occ);
                    else if (basis[i]->quanta[j].n() == 1 &&
                             basis[i]->quanta[j].twos() == -1)
                        site_probs[i]->probs[j] = beta_occ * (1 - alpha_occ);
                    else
                        assert(false);
                }
            } else {
                if (basis[i]->n == 4)
                    for (int j = 0; j < basis[i]->n; j++) {
                        site_probs[i]->quanta[j] = basis[i]->quanta[j];
                        if (basis[i]->quanta[j].n() == 0)
                            site_probs[i]->probs[j] = occ[4 * i + 0];
                        else if (basis[i]->quanta[j].n() == 2)
                            site_probs[i]->probs[j] = occ[4 * i + 3];
                        else if (basis[i]->quanta[j].n() == 1 &&
                                 basis[i]->quanta[j].twos() == 1)
                            site_probs[i]->probs[j] = occ[4 * i + 1];
                        else if (basis[i]->quanta[j].n() == 1 &&
                                 basis[i]->quanta[j].twos() == -1)
                            site_probs[i]->probs[j] = occ[4 * i + 2];
                        else
                            assert(false);
                    }
                else if (basis[i]->n == 3) {
                    for (int j = 0; j < basis[i]->n; j++) {
                        site_probs[i]->quanta[j] = basis[i]->quanta[j];
                        if (basis[i]->quanta[j].n() == 0)
                            site_probs[i]->probs[j] = occ[4 * i + 0];
                        else if (basis[i]->quanta[j].n() == 2)
                            site_probs[i]->probs[j] = occ[4 * i + 3];
                        else if (basis[i]->quanta[j].n() == 1 &&
                                 basis[i]->quanta[j].twos() == 1)
                            site_probs[i]->probs[j] =
                                occ[4 * i + 1] + occ[4 * i + 2];
                        else
                            assert(false);
                    }
                    site_prefs[i].resize(3);
                    site_prefs[i][1] = vector<double>{1};
                    const double st = occ[4 * i + 1] + occ[4 * i + 2];
                    const double sl =
                        abs(st) > TINY ? occ[4 * i + 2] / st : 0.5;
                    const double sh =
                        abs(st) > TINY ? occ[4 * i + 1] / st : 0.5;
                    site_prefs[i][2] = vector<double>{sl, sh};
                } else
                    assert(false);
            }
        }
        // left and right block probabilities
        vector<shared_ptr<StateProbability<S>>> left_probs(n_sites + 1);
        vector<shared_ptr<StateProbability<S>>> right_probs(n_sites + 1);
        left_probs[0] =
            make_shared<StateProbability<S>>(left_dims_fci[0]->quanta[0]);
        for (int i = 0; i < n_sites; i++)
            left_probs[i + 1] = make_shared<StateProbability<S>>(
                StateProbability<S>::tensor_product_no_collect(
                    *left_probs[i], *site_probs[i], *left_dims_fci[i + 1],
                    site_prefs[i]));
        right_probs[n_sites] = make_shared<StateProbability<S>>(
            right_dims_fci[n_sites]->quanta[0]);
        for (int i = n_sites - 1; i >= 0; i--)
            right_probs[i] = make_shared<StateProbability<S>>(
                StateProbability<S>::tensor_product_no_collect(
                    *site_probs[i], *right_probs[i + 1], *right_dims_fci[i],
                    site_prefs[i]));
        // conditional probabilities
        for (int i = 0; i <= n_sites; i++) {
            vector<double> lprobs(left_probs[i]->n), rprobs(right_probs[i]->n);
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
        left_dims[0] = make_shared<StateInfo<S>>(left_dims_fci[0]->deep_copy());
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
        right_dims[n_sites] =
            make_shared<StateInfo<S>>(right_dims_fci[n_sites]->deep_copy());
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
        left_dims[0] = make_shared<StateInfo<S>>(left_dims_fci[0]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            left_dims[i + 1] =
                make_shared<StateInfo<S>>(left_dims_fci[i + 1]->deep_copy());
        for (int i = 0; i < n_sites; i++)
            if (left_dims[i + 1]->n_states_total > m) {
                total_bond_t new_total = 0;
                for (int k = 0; k < left_dims[i + 1]->n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)left_dims[i + 1]->n_states[k] *
                                        m / left_dims[i + 1]->n_states_total) +
                                   0.1);
                    assert(new_n_states != 0);
                    left_dims[i + 1]->n_states[k] = (ubond_t)min(
                        new_n_states, (uint32_t)numeric_limits<ubond_t>::max());
                    new_total += left_dims[i + 1]->n_states[k];
                }
                left_dims[i + 1]->n_states_total = new_total;
            }
        right_dims[n_sites] =
            make_shared<StateInfo<S>>(right_dims_fci[n_sites]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims[i] =
                make_shared<StateInfo<S>>(right_dims_fci[i]->deep_copy());
        for (int i = n_sites - 1; i >= 0; i--)
            if (right_dims[i]->n_states_total > m) {
                total_bond_t new_total = 0;
                for (int k = 0; k < right_dims[i]->n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)right_dims[i]->n_states[k] * m /
                                        right_dims[i]->n_states_total) +
                                   0.1);
                    assert(new_n_states != 0);
                    right_dims[i]->n_states[k] = (ubond_t)min(
                        new_n_states, (uint32_t)numeric_limits<ubond_t>::max());
                    new_total += right_dims[i]->n_states[k];
                }
                right_dims[i]->n_states_total = new_total;
            }
        check_bond_dimensions();
    }
    ubond_t get_max_bond_dimension() const {
        total_bond_t max_bdim = 0;
        for (int i = 0; i <= n_sites; i++)
            max_bdim = max(left_dims[i]->n_states_total, max_bdim);
        for (int i = n_sites; i >= 0; i--)
            max_bdim = max(right_dims[i]->n_states_total, max_bdim);
        return (ubond_t)min(max_bdim,
                            (total_bond_t)numeric_limits<ubond_t>::max());
    }
    // remove unavailable bond dimensions
    void check_bond_dimensions() {
        for (int i = -1; i < n_sites - 1; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *left_dims[i + 1], *basis[i + 1], *left_dims_fci[i + 2]);
            total_bond_t new_total = 0;
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
            total_bond_t new_total = 0;
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
        for (int i = 0; i <= n_sites; i++)
            left_dims[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims[i]->collect();
    }
    shared_ptr<SparseMatrix<S>>
    swap_wfn_to_fused_left(int i, const shared_ptr<SparseMatrix<S>> &old_wfn,
                           const shared_ptr<CG<S>> &cg) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        shared_ptr<SparseMatrixInfo<S>> wfn_info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>(d_alloc);
        load_left_dims(i);
        load_right_dims(i + 1);
        l = *left_dims[i], m = *basis[i], r = *right_dims[i + 1];
        lm = StateInfo<S>::tensor_product(l, m, *left_dims_fci[i + 1]);
        lmc = StateInfo<S>::get_connection_info(l, m, lm);
        mr = StateInfo<S>::tensor_product(m, r, *right_dims_fci[i]);
        mrc = StateInfo<S>::get_connection_info(m, r, mr);
        shared_ptr<SparseMatrixInfo<S>> owinfo = old_wfn->info;
        wfn_info->initialize(lm, r, owinfo->delta_quantum, owinfo->is_fermion,
                             owinfo->is_wavefunction);
        wfn->allocate(wfn_info);
        wfn->swap_to_fused_left(old_wfn, l, m, r, mr, mrc, lm, lmc, cg);
        mrc.deallocate(), mr.deallocate(), lmc.deallocate();
        lm.deallocate(), r.deallocate(), l.deallocate();
        return wfn;
    }
    shared_ptr<SparseMatrix<S>>
    swap_wfn_to_fused_right(int i, const shared_ptr<SparseMatrix<S>> &old_wfn,
                            const shared_ptr<CG<S>> &cg) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        shared_ptr<SparseMatrixInfo<S>> wfn_info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>(d_alloc);
        load_left_dims(i);
        load_right_dims(i + 1);
        l = *left_dims[i], m = *basis[i], r = *right_dims[i + 1];
        lm = StateInfo<S>::tensor_product(l, m, *left_dims_fci[i + 1]);
        lmc = StateInfo<S>::get_connection_info(l, m, lm);
        mr = StateInfo<S>::tensor_product(m, r, *right_dims_fci[i]);
        mrc = StateInfo<S>::get_connection_info(m, r, mr);
        shared_ptr<SparseMatrixInfo<S>> owinfo = old_wfn->info;
        wfn_info->initialize(l, mr, owinfo->delta_quantum, owinfo->is_fermion,
                             owinfo->is_wavefunction);
        wfn->allocate(wfn_info);
        wfn->swap_to_fused_right(old_wfn, l, m, r, lm, lmc, mr, mrc, cg);
        mrc.deallocate(), mr.deallocate(), lmc.deallocate();
        lm.deallocate(), r.deallocate(), l.deallocate();
        return wfn;
    }
    string get_filename(bool left, int i, const string &dir = "") const {
        stringstream ss;
        ss << (dir == "" ? frame->mps_dir : dir) << "/" << frame->prefix
           << ".MPS.INFO." << tag << (left ? ".LEFT." : ".RIGHT.")
           << Parsing::to_string(i);
        return ss.str();
    }
    void shallow_copy_to(const shared_ptr<MPSInfo<S>> &info) const {
        if (frame->prefix_can_write)
            for (int i = 0; i < n_sites + 1; i++) {
                Parsing::link_file(get_filename(true, i),
                                   info->get_filename(true, i));
                Parsing::link_file(get_filename(false, i),
                                   info->get_filename(false, i));
            }
    }
    virtual shared_ptr<MPSInfo<S>> shallow_copy(const string &new_tag) const {
        shared_ptr<MPSInfo<S>> info = make_shared<MPSInfo<S>>(*this);
        info->tag = new_tag;
        shallow_copy_to(info);
        return info;
    }
    virtual shared_ptr<MPSInfo<S>> deep_copy() const {
        stringstream ss;
        save_data(ss);
        shared_ptr<MPSInfo<S>> info = make_shared<MPSInfo<S>>(0);
        info->load_data(ss);
        return info;
    }
    void copy_mutable(const string &dir) const {
        if (frame->prefix_can_write) {
            for (int i = 0; i < n_sites + 1; i++) {
                Parsing::copy_file(get_filename(true, i),
                                   get_filename(true, i, dir));
                Parsing::copy_file(get_filename(false, i),
                                   get_filename(false, i, dir));
            }
            save_data(dir + "/mps_info.bin");
        }
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

template <typename S1, typename S2, typename = void, typename = void>
struct TransMPSInfo {
    static shared_ptr<MPSInfo<S2>> forward(const shared_ptr<MPSInfo<S1>> &si,
                                           S2 target) {
        return TransMPSInfo<S2, S1>::backward(si, target);
    }
    static shared_ptr<MPSInfo<S1>> backward(const shared_ptr<MPSInfo<S2>> &si,
                                            S1 target) {
        return TransMPSInfo<S2, S1>::forward(si, target);
    }
};

// Translation between SU2 and SZ MPSInfo
template <typename S1, typename S2>
struct TransMPSInfo<S1, S2, typename S1::is_sz_t, typename S2::is_su2_t> {
    template <typename SX, typename SY>
    static shared_ptr<MPSInfo<SY>> transform(const shared_ptr<MPSInfo<SX>> &si,
                                             SY target) {
        int n_sites = si->n_sites;
        SY vacuum(si->vacuum.n(), abs(si->vacuum.twos()), si->vacuum.pg());
        vector<shared_ptr<StateInfo<SY>>> basis(n_sites);
        for (int i = 0; i < n_sites; i++)
            basis[i] = TransStateInfo<SX, SY>::forward(si->basis[i]);
        shared_ptr<MPSInfo<SY>> so =
            make_shared<MPSInfo<SY>>(n_sites, vacuum, target, basis);
        // handle the singlet embedding case
        so->left_dims_fci[0] =
            TransStateInfo<SX, SY>::forward(si->left_dims_fci[0]);
        for (int i = 0; i < n_sites; i++)
            so->left_dims_fci[i + 1] =
                make_shared<StateInfo<SY>>(StateInfo<SY>::tensor_product(
                    *so->left_dims_fci[i], *basis[i], target));
        so->right_dims_fci[n_sites] =
            TransStateInfo<SX, SY>::forward(si->right_dims_fci[n_sites]);
        for (int i = n_sites - 1; i >= 0; i--)
            so->right_dims_fci[i] =
                make_shared<StateInfo<SY>>(StateInfo<SY>::tensor_product(
                    *basis[i], *so->right_dims_fci[i + 1], target));
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<SY>::filter(*so->left_dims_fci[i], *so->right_dims_fci[i],
                                  target);
            StateInfo<SY>::filter(*so->right_dims_fci[i], *so->left_dims_fci[i],
                                  target);
        }
        for (int i = 0; i <= n_sites; i++)
            so->left_dims_fci[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            so->right_dims_fci[i]->collect();
        for (int i = 0; i <= n_sites; i++)
            so->left_dims[i] =
                TransStateInfo<SX, SY>::forward(si->left_dims[i]);
        for (int i = n_sites; i >= 0; i--)
            so->right_dims[i] =
                TransStateInfo<SX, SY>::forward(si->right_dims[i]);
        so->check_bond_dimensions();
        so->bond_dim = so->get_max_bond_dimension();
        so->tag = si->tag;
        return so;
    }
    static shared_ptr<MPSInfo<S2>> forward(const shared_ptr<MPSInfo<S1>> &si,
                                           S2 target) {
        return transform<S1, S2>(si, target);
    }
    static shared_ptr<MPSInfo<S1>> backward(const shared_ptr<MPSInfo<S2>> &si,
                                            S1 target) {
        return transform<S2, S1>(si, target);
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
    using MPSInfo<S>::shallow_copy_to;
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
                for (int l = 0; l < left_dims[k + 1]->n; l++) {
                    int il = x.find_state(left_dims[k + 1]->quanta[l]);
                    assert(il != -1);
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
                for (int l = 0; l < right_dims[k]->n; l++) {
                    int il = x.find_state(right_dims[k]->quanta[l]);
                    assert(il != -1);
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
    shared_ptr<MPSInfo<S>> shallow_copy(const string &new_tag) const override {
        shared_ptr<MPSInfo<S>> info = make_shared<DynamicMPSInfo<S>>(*this);
        info->tag = new_tag;
        shallow_copy_to(info);
        return info;
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
    using MPSInfo<S>::shallow_copy_to;
    using MPSInfo<S>::set_bond_dimension_fci;
    vector<ActiveTypes> casci_mask;
    // only works for normal sites
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
    // works for normal and big sites
    static vector<ActiveTypes> active_space(int n_sites, int n_inactive_sites,
                                            int n_active_sites,
                                            int n_virtual_sites) {
        vector<ActiveTypes> casci_mask(n_sites, ActiveTypes::Empty);
        assert(n_sites == n_inactive_sites + n_active_sites + n_virtual_sites);
        for (size_t i = 0; i < n_inactive_sites; i++)
            casci_mask[i] = ActiveTypes::Frozen;
        for (size_t i = n_inactive_sites; i < n_inactive_sites + n_active_sites;
             i++)
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
    CASCIMPSInfo(int n_sites, S vacuum, S target,
                 const vector<shared_ptr<StateInfo<S>>> &basis,
                 int n_inactive_sites, int n_active_sites, int n_virtual_sites,
                 bool init_fci = true)
        : casci_mask(active_space(n_sites, n_inactive_sites, n_active_sites,
                                  n_virtual_sites)),
          MPSInfo<S>(n_sites, vacuum, target, basis, false) {
        if (init_fci)
            set_bond_dimension_fci();
    }
    void set_bond_dimension_full_fci(S left_vacuum = S(S::invalid),
                                     S right_vacuum = S(S::invalid)) override {
        assert(casci_mask.size() == n_sites);
        vector<shared_ptr<StateInfo<S>>> adj_basis(basis);
        for (int i = 0; i < n_sites; i++) {
            if (casci_mask[i] == ActiveTypes::Frozen) {
                adj_basis[i] = make_shared<StateInfo<S>>(vacuum);
                for (int j = 0; j < basis[i]->n; j++)
                    if (basis[i]->quanta[j].n() > adj_basis[i]->quanta[0].n())
                        adj_basis[i]->quanta[0] = basis[i]->quanta[j];
            } else if (casci_mask[i] == ActiveTypes::Empty)
                adj_basis[i] = make_shared<StateInfo<S>>(vacuum);
        }
        left_dims_fci[0] = make_shared<StateInfo<S>>(
            left_vacuum == S(S::invalid) ? vacuum : left_vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *left_dims_fci[i], *adj_basis[i], target));
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(
            right_vacuum == S(S::invalid) ? vacuum : right_vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *adj_basis[i], *right_dims_fci[i + 1], target));
    }
    shared_ptr<MPSInfo<S>> shallow_copy(const string &new_tag) const override {
        shared_ptr<MPSInfo<S>> info = make_shared<CASCIMPSInfo<S>>(*this);
        info->tag = new_tag;
        shallow_copy_to(info);
        return info;
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
    using MPSInfo<S>::shallow_copy_to;
    using MPSInfo<S>::set_bond_dimension_fci;
    using MPSInfo<S>::set_bond_dimension_full_fci;
    int n_inactive; //!> Number of inactive orbitals: CI space
    int n_external; //!> Number of external orbitals: CI space
    int ci_order; //!> Up to how many electrons are allowed in ext. orbitals: 2
                  //! gives MR-CISD
    MRCIMPSInfo(int n_sites, int n_external, int ci_order, S vacuum, S target,
                const vector<shared_ptr<StateInfo<S>>> &basis,
                bool init_fci = true)
        : MRCIMPSInfo(n_sites, 0, n_external, ci_order, vacuum, target, basis,
                      init_fci) {}
    MRCIMPSInfo(int n_sites, int n_inactive, int n_external, int ci_order,
                S vacuum, S target,
                const vector<shared_ptr<StateInfo<S>>> &basis,
                bool init_fci = true)
        : MPSInfo<S>(n_sites, vacuum, target, basis, false),
          n_inactive(n_inactive), n_external(n_external), ci_order(ci_order) {
        assert(n_external + n_inactive <= n_sites);
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
    void set_bond_dimension_full_fci(S left_vacuum = S(S::invalid),
                                     S right_vacuum = S(S::invalid)) override {
        // Same as in the base class: Create left/right fci dims w/o
        // restrictions
        MPSInfo<S>::set_bond_dimension_full_fci(left_vacuum, right_vacuum);
        // Restrict left_dims_fci
        for (int i = 0; i < n_inactive; ++i) {
            auto &state_info = left_dims_fci[i];
            int max_n = 0;
            for (int q = 0; q < state_info->n; ++q)
                if (state_info->quanta[q].n() > max_n)
                    max_n = state_info->quanta[q].n();
            for (int q = 0; q < state_info->n; ++q)
                if (state_info->quanta[q].n() < max_n - ci_order ||
                    state_info->quanta[q].twos() > ci_order)
                    state_info->n_states[q] = 0;
        }
        // Restrict right_dims_fci
        for (int i = n_sites - n_external; i < n_sites; ++i) {
            auto &state_info = right_dims_fci[i];
            for (int q = 0; q < state_info->n; ++q)
                if (state_info->quanta[q].n() > ci_order)
                    state_info->n_states[q] = 0;
        }
    }
    shared_ptr<MPSInfo<S>> shallow_copy(const string &new_tag) const override {
        shared_ptr<MPSInfo<S>> info = make_shared<MRCIMPSInfo<S>>(*this);
        info->tag = new_tag;
        shallow_copy_to(info);
        return info;
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
    using MPSInfo<S>::shallow_copy_to;
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
    shared_ptr<MPSInfo<S>> shallow_copy(const string &new_tag) const override {
        shared_ptr<MPSInfo<S>> info = make_shared<AncillaMPSInfo<S>>(*this);
        info->tag = new_tag;
        shallow_copy_to(info);
        return info;
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
    virtual MPSTypes get_type() const { return MPSTypes::None; }
    // in bytes; 0 = peak memory, 1 = total disk storage
    // only count lower bound of doubles
    virtual vector<size_t>
    estimate_storage(shared_ptr<MPSInfo<S>> info = nullptr) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        size_t peak = 0, total = 0;
        shared_ptr<SparseMatrixInfo<S>> mat_info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        if (info == nullptr)
            info = this->info;
        assert(info != nullptr);
        vector<size_t> left_total(1, 0), right_total(1, 0);
        for (int i = 0; i < n_sites; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *info->left_dims[i], *info->basis[i],
                *info->left_dims_fci[i + 1]);
            mat_info->initialize(t, *info->left_dims[i + 1], info->vacuum,
                                 false);
            left_total.push_back(left_total.back() +
                                 mat_info->get_total_memory());
            mat_info->deallocate();
        }
        for (int i = n_sites - 1; i >= 0; i--) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                *info->basis[i], *info->right_dims[i + 1],
                *info->right_dims_fci[i]);
            mat_info->initialize(*info->right_dims[i], t, info->vacuum, false);
            right_total.push_back(right_total.back() +
                                  mat_info->get_total_memory());
            mat_info->deallocate();
        }
        if (dot == 2) {
            for (int i = 0; i < n_sites - 1; i++) {
                StateInfo<S> tl = StateInfo<S>::tensor_product(
                    *info->left_dims[i], *info->basis[i],
                    *info->left_dims_fci[i + 1]);
                StateInfo<S> tr = StateInfo<S>::tensor_product(
                    *info->basis[i + 1], *info->right_dims[i + 2],
                    *info->right_dims_fci[i + 1]);
                mat_info->initialize(tl, tr, info->target, false, true);
                peak = max(peak, (size_t)mat_info->get_total_memory());
                total = max(total, left_total[i] +
                                       right_total[n_sites + 1 - (i + 2)] +
                                       mat_info->get_total_memory());
                mat_info->deallocate();
            }
        } else
            for (int i = 1; i <= n_sites; i++) {
                peak = max(peak, (size_t)(left_total[i] - left_total[i - 1]));
                total = max(total, left_total[i] + right_total[n_sites - i]);
            }
        return vector<size_t>{peak * 8, total * 8};
    }
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
    void
    flip_fused_form(int center, const shared_ptr<CG<S>> &cg,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        if (para_rule == nullptr || para_rule->is_root()) {
            load_tensor(center);
            if (canonical_form[center] == 'S')
                tensors[center] =
                    info->swap_wfn_to_fused_left(center, tensors[center], cg);
            else if (canonical_form[center] == 'K')
                tensors[center] =
                    info->swap_wfn_to_fused_right(center, tensors[center], cg);
            else
                assert(false);
            save_tensor(center);
            unload_tensor(center);
        }
        if (canonical_form[center] == 'S')
            canonical_form[center] = 'K';
        else if (canonical_form[center] == 'K')
            canonical_form[center] = 'S';
        else
            assert(false);
        if (para_rule != nullptr)
            para_rule->comm->barrier();
    }
    void from_singlet_embedding_wfn(
        const shared_ptr<CG<S>> &cg,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(center == 0);
        char orig_canonical_form = canonical_form[center];
        if (canonical_form[center] == 'K')
            flip_fused_form(center, cg, para_rule);
        assert(canonical_form[center] == 'S');
        load_tensor(center);
        S dqse = tensors[center]->info->delta_quantum;
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        if (para_rule == nullptr || para_rule->is_root()) {
            assert(tensors[center]->info->n == 1);
            S lq = tensors[center]->info->quanta[0].get_bra(dqse);
            S dq(dqse.n() - lq.twos(), lq.twos(), dqse.pg());
            assert(tensors[center]->info->is_wavefunction);
            shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                make_shared<VectorAllocator<uint32_t>>();
            shared_ptr<VectorAllocator<double>> d_alloc =
                make_shared<VectorAllocator<double>>();
            shared_ptr<SparseMatrixInfo<S>> wfn_info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            shared_ptr<SparseMatrix<S>> wfn =
                make_shared<SparseMatrix<S>>(d_alloc);
            StateInfo<S> lsi(info->vacuum), rsi(dq);
            rsi.n_states[0] = tensors[center]->info->n_states_ket[0];
            wfn_info->initialize(lsi, rsi, dq, false, true);
            wfn->allocate(wfn_info);
            rsi.deallocate();
            lsi.deallocate();
            assert(wfn->total_memory == tensors[center]->total_memory);
            memcpy(wfn->data, tensors[center]->data,
                   sizeof(double) * wfn->total_memory);
            unload_tensor(center);
            tensors[center] = wfn;
            save_tensor(center);
            assert(info->target == dqse);
            info->target = dq;
            info->set_bond_dimension_fci();
            info->load_left_dims(center);
            info->left_dims[center]->quanta[0] = info->vacuum;
            info->save_left_dims(center);
        } else {
            S lq = tensors[center]->info->quanta[0].get_bra(dqse);
            S dq(dqse.n() - lq.twos(), lq.twos(), dqse.pg());
            assert(info->target == dqse);
            info->target = dq;
            info->set_bond_dimension_fci();
        }
        unload_tensor(center);
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        if (canonical_form[center] != orig_canonical_form)
            flip_fused_form(center, cg, para_rule);
    }
    void to_singlet_embedding_wfn(
        const shared_ptr<CG<S>> &cg,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(center == 0);
        char orig_canonical_form = canonical_form[center];
        if (canonical_form[center] == 'K')
            flip_fused_form(center, cg, para_rule);
        assert(canonical_form[center] == 'S');
        load_tensor(center);
        S dq = tensors[center]->info->delta_quantum;
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        if (para_rule == nullptr || para_rule->is_root()) {
            assert(tensors[center]->info->n == 1 &&
                   tensors[center]->info->quanta[0].get_bra(dq) ==
                       info->vacuum);
            assert(tensors[center]->info->is_wavefunction);
            shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                make_shared<VectorAllocator<uint32_t>>();
            shared_ptr<VectorAllocator<double>> d_alloc =
                make_shared<VectorAllocator<double>>();
            shared_ptr<SparseMatrixInfo<S>> wfn_info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            shared_ptr<SparseMatrix<S>> wfn =
                make_shared<SparseMatrix<S>>(d_alloc);
            S dqse(dq.n() + dq.twos(), 0, dq.pg());
            S lq(dq.twos(), dq.twos(), 0);
            StateInfo<S> lsi(lq), rsi(dq);
            rsi.n_states[0] = tensors[center]->info->n_states_ket[0];
            wfn_info->initialize(lsi, rsi, dqse, false, true);
            wfn->allocate(wfn_info);
            rsi.deallocate();
            lsi.deallocate();
            assert(wfn->total_memory == tensors[center]->total_memory);
            memcpy(wfn->data, tensors[center]->data,
                   sizeof(double) * wfn->total_memory);
            unload_tensor(center);
            tensors[center] = wfn;
            save_tensor(center);
            assert(info->target == dq);
            info->target = dqse;
            info->set_bond_dimension_fci(lq);
            info->load_left_dims(center);
            info->left_dims[center]->quanta[0] = lq;
            info->save_left_dims(center);
        } else {
            S dqse(dq.n() + dq.twos(), 0, dq.pg());
            S lq(dq.twos(), dq.twos(), 0);
            assert(info->target == dq);
            info->target = dqse;
            info->set_bond_dimension_fci(lq);
        }
        unload_tensor(center);
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        if (canonical_form[center] != orig_canonical_form)
            flip_fused_form(center, cg, para_rule);
    }
    // CC -> KR or K -> S or LS -> KR or LK -> KR
    void move_left(const shared_ptr<CG<S>> &cg,
                   const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        if (canonical_form[center] == 'C') {
            assert(center + 1 < n_sites);
            assert(dot == 2 && tensors[center + 1] == nullptr);
            if (para_rule == nullptr || para_rule->is_root()) {
                shared_ptr<SparseMatrix<S>> left, right;
                load_tensor(center);
                tensors[center]->right_split(left, right, info->bond_dim);
                info->right_dims[center + 1] =
                    right->info->extract_state_info(false);
                info->save_right_dims(center + 1);
                unload_tensor(center);
                tensors[center] = left;
                tensors[center + 1] = right;
                save_tensor(center);
                save_tensor(center + 1);
                unload_tensor(center + 1);
                unload_tensor(center);
            } else {
                tensors[center] = make_shared<SparseMatrix<S>>();
                tensors[center + 1] = make_shared<SparseMatrix<S>>();
            }
            if (para_rule != nullptr)
                para_rule->comm->barrier();
            canonical_form[center] = 'K';
            canonical_form[center + 1] = 'R';
        } else if (canonical_form[center] == 'S' ||
                   canonical_form[center] == 'K') {
            if (para_rule == nullptr || para_rule->is_root()) {
                load_tensor(center);
                if (canonical_form[center] == 'K') {
                    tensors[center] = info->swap_wfn_to_fused_right(
                        center, tensors[center], cg);
                    if (center == 0) {
                        canonical_form[center] = 'S';
                        save_tensor(center);
                        unload_tensor(center);
                        if (para_rule != nullptr)
                            para_rule->comm->barrier();
                        return;
                    }
                }
                assert(canonical_form[center - 1] == 'L');
                shared_ptr<SparseMatrix<S>> left, right;
                tensors[center]->right_split(left, right, info->bond_dim);
                info->right_dims[center] =
                    right->info->extract_state_info(false);
                info->save_right_dims(center);
                unload_tensor(center);
                tensors[center] = right;
                save_tensor(center);
                assert(tensors[center]->info->n != 0);
                load_tensor(center - 1);
                assert(tensors[center - 1]->info->n != 0);
                shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                    make_shared<VectorAllocator<uint32_t>>();
                shared_ptr<VectorAllocator<double>> d_alloc =
                    make_shared<VectorAllocator<double>>();
                shared_ptr<SparseMatrix<S>> wfn =
                    make_shared<SparseMatrix<S>>(d_alloc);
                shared_ptr<SparseMatrixInfo<S>> winfo =
                    make_shared<SparseMatrixInfo<S>>(i_alloc);
                winfo->initialize_contract(tensors[center - 1]->info,
                                           left->info);
                wfn->allocate(winfo);
                wfn->contract(tensors[center - 1], left);
                tensors[center - 1] = wfn;
                save_tensor(center - 1);
                assert(tensors[center - 1]->info->n != 0);
            } else {
                if (canonical_form[center] == 'K') {
                    if (center == 0) {
                        canonical_form[center] = 'S';
                        if (para_rule != nullptr)
                            para_rule->comm->barrier();
                        return;
                    }
                }
            }
            canonical_form[center] = 'R';
            canonical_form[center - 1] = 'K';
            if (para_rule != nullptr)
                para_rule->comm->barrier();
            center--;
        } else
            assert(false);
    }
    // CC -> LS or S -> K or KR -> LS or SR -> LS
    void move_right(const shared_ptr<CG<S>> &cg,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        if (canonical_form[center] == 'C') {
            assert(center + 1 < n_sites);
            assert(dot == 2 && tensors[center + 1] == nullptr);
            if (para_rule == nullptr || para_rule->is_root()) {
                shared_ptr<SparseMatrix<S>> left, right;
                load_tensor(center);
                tensors[center]->left_split(left, right, info->bond_dim);
                info->left_dims[center + 1] =
                    left->info->extract_state_info(true);
                info->save_left_dims(center + 1);
                unload_tensor(center);
                tensors[center] = left;
                tensors[center + 1] = right;
                save_tensor(center);
                save_tensor(center + 1);
                unload_tensor(center + 1);
                unload_tensor(center);
            } else {
                tensors[center] = make_shared<SparseMatrix<S>>();
                tensors[center + 1] = make_shared<SparseMatrix<S>>();
            }
            if (para_rule != nullptr)
                para_rule->comm->barrier();
            canonical_form[center] = 'L';
            canonical_form[center + 1] = 'S';
            center++;
        } else if (canonical_form[center] == 'S' ||
                   canonical_form[center] == 'K') {
            if (para_rule == nullptr || para_rule->is_root()) {
                load_tensor(center);
                assert(tensors[center]->info->n != 0);
                if (canonical_form[center] == 'S') {
                    tensors[center] = info->swap_wfn_to_fused_left(
                        center, tensors[center], cg);
                    if (center == n_sites - 1) {
                        canonical_form[center] = 'K';
                        save_tensor(center);
                        unload_tensor(center);
                        if (para_rule != nullptr)
                            para_rule->comm->barrier();
                        return;
                    }
                }
                assert(tensors[center]->info->n != 0);
                assert(canonical_form[center + 1] == 'R');
                shared_ptr<SparseMatrix<S>> left, right;
                tensors[center]->left_split(left, right, info->bond_dim);
                info->left_dims[center + 1] =
                    left->info->extract_state_info(true);
                info->save_left_dims(center + 1);
                unload_tensor(center);
                tensors[center] = left;
                save_tensor(center);
                load_tensor(center + 1);
                shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                    make_shared<VectorAllocator<uint32_t>>();
                shared_ptr<VectorAllocator<double>> d_alloc =
                    make_shared<VectorAllocator<double>>();
                shared_ptr<SparseMatrix<S>> wfn =
                    make_shared<SparseMatrix<S>>(d_alloc);
                shared_ptr<SparseMatrixInfo<S>> winfo =
                    make_shared<SparseMatrixInfo<S>>(i_alloc);
                winfo->initialize_contract(right->info,
                                           tensors[center + 1]->info);
                wfn->allocate(winfo);
                wfn->contract(right, tensors[center + 1]);
                tensors[center + 1] = wfn;
                save_tensor(center + 1);
            } else {
                if (canonical_form[center] == 'S') {
                    if (center == n_sites - 1) {
                        canonical_form[center] = 'K';
                        if (para_rule != nullptr)
                            para_rule->comm->barrier();
                        return;
                    }
                }
            }
            canonical_form[center] = 'L';
            canonical_form[center + 1] = 'S';
            if (para_rule != nullptr)
                para_rule->comm->barrier();
            center++;
        } else
            assert(false);
    }
    // can reduce bond dims
    void dynamic_canonicalize() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        for (int i = 0; i < center; i++) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> left, right;
            tensors[i]->left_split(left, right, 0);
            tensors[i] = left;
            shared_ptr<StateInfo<S>> nl = left->info->extract_state_info(true);
            StateInfo<S> l = *info->left_dims[i + 1], m = *info->basis[i + 1];
            StateInfo<S> lm = StateInfo<S>::tensor_product(
                             l, m, *info->left_dims_fci[i + 2]),
                         r;
            StateInfo<S> nlm = StateInfo<S>::tensor_product(
                *nl, m, *info->left_dims_fci[i + 2]);
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
            tensors[i + 1] =
                tensors[i + 1]->left_multiply(right, l, m, r, lm, lmc, nlm);
            if (i + 1 == center && dot == 2)
                r.deallocate();
            lmc.deallocate();
            nlm.deallocate();
            lm.deallocate();
            info->left_dims[i + 1] = nl;
            info->save_left_dims(i + 1);
            right->info->deallocate();
            right->deallocate();
        }
        for (int i = n_sites - 1; i >= center + dot; i--) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> left, right;
            tensors[i]->right_split(left, right, 0);
            tensors[i] = right;
            shared_ptr<StateInfo<S>> nr =
                right->info->extract_state_info(false);
            if (dot == 1 && i - 1 == center) {
                shared_ptr<SparseMatrix<S>> wfn =
                    make_shared<SparseMatrix<S>>(d_alloc);
                shared_ptr<SparseMatrixInfo<S>> winfo =
                    make_shared<SparseMatrixInfo<S>>(i_alloc);
                winfo->initialize_contract(tensors[i - 1]->info, left->info);
                wfn->allocate(winfo);
                wfn->contract(tensors[i - 1], left);
                assert(tensors[i - 1] != nullptr);
                tensors[i - 1] = wfn;
            } else {
                StateInfo<S> m = *info->basis[i - 1], r = *info->right_dims[i];
                StateInfo<S> mr = StateInfo<S>::tensor_product(
                    m, r, *info->right_dims_fci[i - 1]);
                StateInfo<S> nmr = StateInfo<S>::tensor_product(
                    m, *nr, *info->right_dims_fci[i - 1]);
                StateInfo<S> mrc = StateInfo<S>::get_connection_info(m, r, mr);
                StateInfo<S> l;
                if (i - 1 == center + 1 && dot == 2) {
                    l = StateInfo<S>::tensor_product(
                        *info->left_dims[center], *info->basis[center],
                        *info->left_dims_fci[center + 1]);
                    assert(tensors[i - 2] != nullptr);
                    tensors[i - 2] = tensors[i - 2]->right_multiply(
                        left, l, m, r, mr, mrc, nmr);
                } else {
                    l = *info->right_dims[i - 1];
                    assert(tensors[i - 1] != nullptr);
                    tensors[i - 1] = tensors[i - 1]->right_multiply(
                        left, l, m, r, mr, mrc, nmr);
                }
                if (i - 1 == center + 1 && dot == 2)
                    l.deallocate();
                mrc.deallocate();
                nmr.deallocate();
                mr.deallocate();
            }
            info->right_dims[i] = nr;
            info->save_right_dims(i);
            left->info->deallocate();
            left->deallocate();
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
            tensors[i + 1]->left_multiply_inplace(tmat, l, m, r, lm, lmc);
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
                    tensors[i - 2]->right_multiply_inplace(tmat, l, m, r, mr,
                                                           mrc);
                } else {
                    l = *info->right_dims[i - 1];
                    assert(tensors[i - 1] != nullptr);
                    tensors[i - 1]->right_multiply_inplace(tmat, l, m, r, mr,
                                                           mrc);
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
    virtual string get_filename(int i, const string &dir = "") const {
        stringstream ss;
        ss << (dir == "" ? frame->mps_dir : dir) << "/" << frame->prefix
           << ".MPS." << info->tag << "." << Parsing::to_string(i);
        return ss.str();
    }
    void shallow_copy_to(const shared_ptr<MPS<S>> &mps) const {
        if (frame->prefix_can_write)
            for (int i = 0; i < n_sites; i++)
                Parsing::link_file(get_filename(i), mps->get_filename(i));
    }
    virtual shared_ptr<MPS<S>> shallow_copy(const string &new_tag) const {
        shared_ptr<MPSInfo<S>> new_info = info->shallow_copy(new_tag);
        shared_ptr<MPS<S>> mps = make_shared<MPS<S>>(*this);
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        for (int i = 0; i < mps->n_sites; i++)
            if (mps->tensors[i] != nullptr)
                mps->tensors[i] = make_shared<SparseMatrix<S>>(d_alloc);
        mps->info = new_info;
        shallow_copy_to(mps);
        return mps;
    }
    virtual shared_ptr<MPS<S>> deep_copy(const string &xtag) const {
        shared_ptr<MPSInfo<S>> xinfo = info->deep_copy();
        xinfo->load_mutable();
        shared_ptr<MPS<S>> xmps = make_shared<MPS<S>>(xinfo);
        xmps->load_data();
        xmps->load_mutable();
        xinfo->tag = xtag;
        xinfo->save_mutable();
        xmps->save_mutable();
        xmps->save_data();
        xmps->deallocate();
        xinfo->deallocate_mutable();
        return xmps;
    }
    virtual void copy_data(const string &dir) const {
        if (frame->prefix_can_write) {
            for (int i = 0; i < n_sites; i++)
                if (tensors[i] != nullptr)
                    Parsing::copy_file(get_filename(i), get_filename(i, dir));
            Parsing::copy_file(get_filename(-1), get_filename(-1, dir));
        }
    }
    void load_data_from(istream &ifs) {
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
    void save_data_to(ostream &ofs) const {
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
            string filename = get_filename(-1);
            if (Parsing::link_exists(filename))
                Parsing::remove_file(filename);
            ofstream ofs(filename.c_str(), ios::binary);
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
