
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

#include "mps.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace std;

namespace block2 {

// MPSInfo for multiple targets
template <typename S> struct MultiMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::left_dims_fci;
    using MPSInfo<S>::right_dims_fci;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::basis;
    vector<S> targets;
    MultiMPSInfo(int n_sites, S vacuum, const vector<S> &targets,
                 const vector<shared_ptr<StateInfo<S>>> &basis,
                 bool init_fci = true)
        : targets(targets), MPSInfo<S>(n_sites, vacuum, vacuum, basis, false) {
        if (init_fci)
            set_bond_dimension_fci();
    }
    MultiTypes get_multi_type() const override { return MultiTypes::Multi; }
    vector<S> get_complementary(S q) const override {
        vector<S> r;
        for (auto target : targets) {
            S qs = target - q;
            for (int i = 0; i < qs.count(); i++)
                r.push_back(qs[i]);
        }
        sort(r.begin(), r.end());
        r.resize(distance(r.begin(), unique(r.begin(), r.end())));
        return r;
    }
    void set_bond_dimension_fci() override {
        S max_target = *max_element(targets.begin(), targets.end());
        left_dims_fci[0] = make_shared<StateInfo<S>>(vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *left_dims_fci[i], *basis[i], max_target));
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *basis[i], *right_dims_fci[i + 1], max_target));
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<S>::multi_target_filter(*left_dims_fci[i],
                                              *right_dims_fci[i], targets);
            StateInfo<S>::multi_target_filter(*right_dims_fci[i],
                                              *left_dims_fci[i], targets);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i]->collect();
    }
};

// Matrix Product State for multiple targets and multiple wavefunctions
template <typename S> struct MultiMPS : MPS<S> {
    using MPS<S>::n_sites;
    using MPS<S>::center;
    using MPS<S>::dot;
    using MPS<S>::info;
    using MPS<S>::tensors;
    using MPS<S>::canonical_form;
    // numebr of wavefunctions
    int nroots;
    // wavefunctions
    vector<shared_ptr<SparseMatrixGroup<S>>> wfns;
    // weights of wavefunctions
    vector<double> weights;
    MultiMPS(const shared_ptr<MultiMPSInfo<S>> &info) : MPS<S>(info) {}
    MultiMPS(int n_sites, int center, int dot, int nroots)
        : MPS<S>(n_sites, center, dot), nroots(nroots) {
        if (center >= 0 && center < n_sites)
            for (int i = center; i < center + dot; i++)
                canonical_form[i] = 'M';
        weights = vector<double>(nroots, 1.0 / nroots);
    }
    void initialize(const shared_ptr<MPSInfo<S>> &info, bool init_left = true,
                    bool init_right = true) override {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        this->info = info;
        assert(info->get_multi_type() == MultiTypes::Multi);
        shared_ptr<MultiMPSInfo<S>> minfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(info);
        vector<shared_ptr<SparseMatrixInfo<S>>> wfn_infos;
        wfn_infos.resize(minfo->targets.size());
        tensors.resize(n_sites);
        wfns.resize(nroots);
        if (init_left)
            MPS<S>::initialize_left(info, center - 1);
        if (center >= 0 && center < n_sites && (init_left || init_right)) {
            for (size_t i = 0; i < minfo->targets.size(); i++)
                wfn_infos[i] = make_shared<SparseMatrixInfo<S>>(i_alloc);
            if (dot == 1) {
                StateInfo<S> t = StateInfo<S>::tensor_product(
                    *info->left_dims[center], *info->basis[center],
                    *info->left_dims_fci[center + dot]);
                for (size_t i = 0; i < minfo->targets.size(); i++)
                    wfn_infos[i]->initialize(t, *info->right_dims[center + dot],
                                             minfo->targets[i], false, true);
                canonical_form[center] = 'J';
            } else {
                StateInfo<S> tl = StateInfo<S>::tensor_product(
                    *info->left_dims[center], *info->basis[center],
                    *info->left_dims_fci[center + 1]);
                StateInfo<S> tr = StateInfo<S>::tensor_product(
                    *info->basis[center + 1], *info->right_dims[center + dot],
                    *info->right_dims_fci[center + 1]);
                for (size_t i = 0; i < minfo->targets.size(); i++)
                    wfn_infos[i]->initialize(tl, tr, minfo->targets[i], false,
                                             true);
            }
            for (int j = 0; j < nroots; j++) {
                wfns[j] = make_shared<SparseMatrixGroup<S>>(d_alloc);
                wfns[j]->allocate(wfn_infos);
            }
        }
        if (init_right)
            MPS<S>::initialize_right(info, center + dot);
    }
    void random_canonicalize() override {
        for (int i = 0; i < n_sites; i++)
            MPS<S>::random_canonicalize_tensor(i);
        for (int j = 0; j < nroots; j++)
            wfns[j]->randomize();
    }
    string get_filename(int i) const override {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MMPS." << info->tag
           << "." << Parsing::to_string(i);
        return ss.str();
    }
    string get_wfn_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MMPS-WFN."
           << info->tag << "." << Parsing::to_string(i);
        return ss.str();
    }
    void load_data() override {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        ifstream ifs(get_filename(-1).c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MultiMPS::load_data on '" + get_filename(-1) +
                                "' failed.");
        MPS<S>::load_data_from(ifs);
        ifs.read((char *)&nroots, sizeof(nroots));
        weights.resize(nroots);
        wfns.resize(nroots);
        ifs.read((char *)&weights[0], sizeof(double) * nroots);
        for (int i = 0; i < nroots; i++)
            wfns[i] = make_shared<SparseMatrixGroup<S>>(d_alloc);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MultiMPS::load_data on '" + get_filename(-1) +
                                "' failed.");
        ifs.close();
    }
    void save_data() const override {
        if (frame->prefix_can_write) {
            ofstream ofs(get_filename(-1).c_str(), ios::binary);
            if (!ofs.good())
                throw runtime_error("MultiMPS::save_data on '" +
                                    get_filename(-1) + "' failed.");
            MPS<S>::save_data_to(ofs);
            ofs.write((char *)&nroots, sizeof(nroots));
            assert(weights.size() == nroots);
            ofs.write((char *)&weights[0], sizeof(double) * nroots);
            if (!ofs.good())
                throw runtime_error("MultiMPS::save_data on '" +
                                    get_filename(-1) + "' failed.");
            ofs.close();
        }
    }
    void load_mutable() const override {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr)
                tensors[i]->load_data(get_filename(i), true, i_alloc);
            else if (i == center)
                for (int j = 0; j < nroots; j++) {
                    wfns[j]->load_data(get_wfn_filename(j), j == 0, i_alloc);
                    wfns[j]->infos = wfns[0]->infos;
                }
    }
    void save_mutable() const override {
        if (frame->prefix_can_write) {
            for (int i = 0; i < n_sites; i++)
                if (tensors[i] != nullptr)
                    tensors[i]->save_data(get_filename(i), true);
                else if (i == center)
                    for (int j = 0; j < nroots; j++)
                        wfns[j]->save_data(get_wfn_filename(j), j == 0);
        }
    }
    void save_wavefunction(int i) const {
        if (frame->prefix_can_write) {
            assert(tensors[i] == nullptr);
            for (int j = 0; j < nroots; j++)
                wfns[j]->save_data(get_wfn_filename(j), j == 0);
        }
    }
    void load_wavefunction(int i) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        assert(tensors[i] == nullptr);
        for (int j = 0; j < nroots; j++) {
            wfns[j]->load_data(get_wfn_filename(j), j == 0, i_alloc);
            wfns[j]->infos = wfns[0]->infos;
        }
    }
    void unload_wavefunction(int i) {
        assert(tensors[i] == nullptr);
        for (int j = nroots - 1; j >= 0; j--)
            wfns[j]->deallocate();
        if (nroots != 0)
            wfns[0]->deallocate_infos();
    }
    void save_tensor(int i) const override {
        if (frame->prefix_can_write) {
            assert(tensors[i] != nullptr || i == center);
            if (tensors[i] != nullptr)
                tensors[i]->save_data(get_filename(i), true);
            else
                for (int j = 0; j < nroots; j++)
                    wfns[j]->save_data(get_wfn_filename(j), j == 0);
        }
    }
    void load_tensor(int i) override {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        assert(tensors[i] != nullptr || i == center);
        if (tensors[i] != nullptr)
            tensors[i]->load_data(get_filename(i), true, i_alloc);
        else
            for (int j = 0; j < nroots; j++) {
                wfns[j]->load_data(get_wfn_filename(j), j == 0, i_alloc);
                wfns[j]->infos = wfns[0]->infos;
            }
    }
    void unload_tensor(int i) override {
        assert(tensors[i] != nullptr || i == center);
        if (tensors[i] != nullptr) {
            tensors[i]->deallocate();
            tensors[i]->info->deallocate();
        } else {
            for (int j = nroots - 1; j >= 0; j--)
                wfns[j]->deallocate();
            if (nroots != 0)
                wfns[0]->deallocate_infos();
        }
    }
    void deallocate() override {
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->deallocate();
            else if (i == center)
                for (int j = nroots - 1; j >= 0; j--)
                    wfns[j]->deallocate();
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->info->deallocate();
            else if (i == center && nroots != 0)
                wfns[0]->deallocate_infos();
    }
};

} // namespace block2
