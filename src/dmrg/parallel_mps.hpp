
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

#include "mps.hpp"
#include "../core/parallel_rule.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

// Matrix Product State with multi canonical centers
template <typename S> struct ParallelMPS : MPS<S> {
    using MPS<S>::n_sites;
    using MPS<S>::center;
    using MPS<S>::dot;
    using MPS<S>::info;
    using MPS<S>::tensors;
    using MPS<S>::canonical_form;
    using MPS<S>::move_left;
    using MPS<S>::move_right;
    using MPS<S>::load_tensor;
    using MPS<S>::save_tensor;
    using MPS<S>::unload_tensor;
    using MPS<S>::get_filename;
    vector<int> conn_centers;
    vector<shared_ptr<SparseMatrix<S>>> conn_matrices;
    shared_ptr<ParallelRule<S>> rule;
    int ncenter = 0;
    string ref_canonical_form;
    double svd_eps = 1E-4;
    double svd_cutoff = 1E-12;
    ParallelMPS(const shared_ptr<MPSInfo<S>> &info,
                const shared_ptr<ParallelRule<S>> &rule = nullptr)
        : MPS<S>(info), rule(rule) {
        n_sites = info->n_sites;
        init_para_mps();
    }
    ParallelMPS(int n_sites, int center, int dot,
                const shared_ptr<ParallelRule<S>> &rule = nullptr)
        : MPS<S>(n_sites, center, dot), rule(rule) {
        init_para_mps();
    }
    // need to manually disable MPS writing in multi procs
    ParallelMPS(const shared_ptr<MPS<S>> &mps,
                const shared_ptr<ParallelRule<S>> &rule = nullptr)
        : MPS<S>(*mps), rule(rule) {
        init_para_mps();
    }
    MPSTypes get_type() const override { return MPSTypes::MultiCenter; }
    void init_para_mps() {
        disable_parallel_writing();
        if (rule != nullptr) {
            assert(rule->comm->size % rule->comm->gsize == 0);
            conn_centers.clear();
            for (int i = 1; i < rule->comm->ngroup; i++) {
                int j = i * n_sites / rule->comm->ngroup;
                if (j < 2 || j > n_sites - 2 ||
                    (conn_centers.size() != 0 && j - conn_centers.back() < 2))
                    continue;
                conn_centers.push_back(j);
            }
        }
        if (conn_centers.size() == 0 && n_sites / 2 >= 2 &&
            n_sites / 2 <= n_sites - 2)
            conn_centers.push_back(n_sites / 2);
    }
    void set_ref_canonical_form() {
        if (rule == nullptr)
            return;
        ref_canonical_form = canonical_form;
        rule->comm->barrier();
    }
    void sync_canonical_form() {
        if (rule == nullptr)
            return;
        vector<char> canonical_form_change(canonical_form.length(), 0);
        if (rule == nullptr || rule->comm->grank == rule->comm->root)
            for (size_t i = 0; i < canonical_form.length(); i++)
                if (canonical_form[i] != ref_canonical_form[i])
                    canonical_form_change[i] = canonical_form[i];
        rule->comm->allreduce_xor(canonical_form_change.data(),
                                  canonical_form_change.size());
        for (size_t i = 0; i < canonical_form.length(); i++)
            if (canonical_form_change[i] != 0)
                canonical_form[i] = canonical_form_change[i];
        ref_canonical_form = canonical_form;
    }
    void enable_parallel_writing() const {
        if (rule != nullptr) {
            frame->prefix_can_write = rule->comm->grank == rule->comm->root;
            if (rule->comm->grank == rule->comm->root)
                cout.clear();
            else
                cout.setstate(ios::failbit);
        }
    }
    void disable_parallel_writing() const {
        if (rule != nullptr) {
            frame->prefix_can_write = rule->comm->rank == rule->comm->root;
            if (rule->comm->rank == rule->comm->root)
                cout.clear();
            else
                cout.setstate(ios::failbit);
        }
    }
    // K|S -> L|S
    void para_merge(int pidx,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int center = conn_centers[pidx];
        assert(canonical_form[center] == 'S');
        assert(canonical_form[center - 1] == 'K');
        if (para_rule == nullptr || para_rule->is_root()) {
            load_tensor(center - 1);
            load_conn_matrix(pidx);
            shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                make_shared<VectorAllocator<uint32_t>>();
            shared_ptr<VectorAllocator<double>> d_alloc =
                make_shared<VectorAllocator<double>>();
            shared_ptr<SparseMatrix<S>> rot =
                make_shared<SparseMatrix<S>>(d_alloc);
            shared_ptr<SparseMatrixInfo<S>> rinfo =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            rinfo->initialize_trans_contract(tensors[center - 1]->info,
                                             conn_matrices[pidx]->info,
                                             info->vacuum, true);
            assert(rinfo->n != 0);
            rot->allocate(rinfo);
            rot->contract(tensors[center - 1], conn_matrices[pidx], true);
            tensors[center - 1] = rot;
            save_tensor(center - 1);
        }
        canonical_form[center - 1] = 'L';
        if (para_rule != nullptr)
            para_rule->comm->barrier();
    }
    // L|S -> K|S
    shared_ptr<SparseMatrix<S>>
    para_split(int pidx,
               const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int center = conn_centers[pidx];
        assert(canonical_form[center] == 'S');
        assert(canonical_form[center - 1] == 'L');
        shared_ptr<SparseMatrix<S>> rmat;
        if (para_rule == nullptr || para_rule->is_root()) {
            load_tensor(center);
            shared_ptr<SparseMatrix<S>> left, right;
            tensors[center]->right_split(left, right, info->bond_dim);
            conn_matrices[pidx] =
                left->pseudo_inverse(info->bond_dim, svd_eps, svd_cutoff);
            save_conn_matrix(pidx);
            info->right_dims[center] = right->info->extract_state_info(false);
            info->save_right_dims(center);
            rmat = tensors[center];
            tensors[center] = right;
            save_tensor(center);
            canonical_form[center - 1] = 'K';
            load_tensor(center - 1);
            shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                make_shared<VectorAllocator<uint32_t>>();
            shared_ptr<VectorAllocator<double>> d_alloc =
                make_shared<VectorAllocator<double>>();
            shared_ptr<SparseMatrix<S>> wfn =
                make_shared<SparseMatrix<S>>(d_alloc);
            shared_ptr<SparseMatrixInfo<S>> winfo =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            assert(!tensors[center - 1]->info->is_wavefunction);
            assert(left->info->is_wavefunction);
            winfo->initialize_contract(tensors[center - 1]->info, left->info);
            wfn->allocate(winfo);
            wfn->contract(tensors[center - 1], left);
            tensors[center - 1] = wfn;
            save_tensor(center - 1);
        } else {
            conn_matrices[pidx] = make_shared<SparseMatrix<S>>();
            canonical_form[center - 1] = 'K';
            rmat = make_shared<SparseMatrix<S>>();
        }
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        return rmat;
    }
    void load_data() override {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        ifstream ifs(get_filename(-1).c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("ParallelMPS::load_data on '" +
                                get_filename(-1) + "' failed.");
        MPS<S>::load_data_from(ifs);
        ifs.read((char *)&ncenter, sizeof(ncenter));
        conn_centers.resize(ncenter);
        ifs.read((char *)&conn_centers[0], sizeof(int) * ncenter);
        uint8_t has_conn;
        ifs.read((char *)&has_conn, sizeof(has_conn));
        if (has_conn != 0) {
            conn_matrices.resize(ncenter);
            for (int i = 0; i < ncenter; i++)
                conn_matrices[i] = make_shared<SparseMatrix<S>>(d_alloc);
        }
        if (ifs.fail() || ifs.bad())
            throw runtime_error("ParallelMPS::load_data on '" +
                                get_filename(-1) + "' failed.");
        ifs.close();
    }
    void save_data() const override {
        if (frame->prefix_can_write) {
            string filename = get_filename(-1);
            if (Parsing::link_exists(filename))
                Parsing::remove_file(filename);
            ofstream ofs(filename.c_str(), ios::binary);
            if (!ofs.good())
                throw runtime_error("ParallelMPS::save_data on '" +
                                    get_filename(-1) + "' failed.");
            MPS<S>::save_data_to(ofs);
            ofs.write((char *)&ncenter, sizeof(ncenter));
            ofs.write((char *)&conn_centers[0], sizeof(int) * ncenter);
            uint8_t has_conn = conn_matrices.size() != 0;
            ofs.write((char *)&has_conn, sizeof(has_conn));
            if (!ofs.good())
                throw runtime_error("ParallelMPS::save_data on '" +
                                    get_filename(-1) + "' failed.");
            ofs.close();
        }
    }
    string get_conn_filename(int i, const string &dir = "") const {
        stringstream ss;
        ss << (dir == "" ? frame->mps_dir : dir) << "/" << frame->prefix
           << ".MPS-CONN." << info->tag << "." << Parsing::to_string(i);
        return ss.str();
    }
    virtual void save_conn_matrix(int i) const {
        if (frame->prefix_can_write) {
            assert(conn_matrices[i] != nullptr);
            conn_matrices[i]->save_data(get_conn_filename(i), true);
        }
    }
    virtual void load_conn_matrix(int i) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(conn_matrices[i] != nullptr);
        conn_matrices[i]->alloc = d_alloc;
        conn_matrices[i]->load_data(get_conn_filename(i), true, i_alloc);
    }
};

} // namespace block2
