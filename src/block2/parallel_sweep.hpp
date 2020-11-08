
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

#include "moving_environment.hpp"
#include "mps.hpp"
#include "sweep_algorithm.hpp"
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
    using MPS<S>::flip_fused_form;
    vector<int> conn_centers;
    vector<shared_ptr<SparseMatrix<S>>> conn_matrices;
    int ncenter = 0;
    ParallelMPS(const shared_ptr<MPSInfo<S>> &info) : MPS<S>(info) {}
    ParallelMPS(int n_sites, int center, int dot)
        : MPS<S>(n_sites, center, dot) {}
    void parallelize(const vector<int> &conn_centers,
                     const shared_ptr<MovingEnvironment<S>> &me) {
        shared_ptr<CG<S>> cg = me->mpo->tf->opf->cg;
        this->conn_centers = conn_centers;
        ncenter = conn_centers.size();
        assert(conn_matrices.size() == 0);
        conn_matrices.resize(ncenter);
        while (center != 0) {
            move_left(cg);
            if (me != nullptr)
                me->right_contract_rotate_unordered(center - dot + 1);
        }
        assert(center == 0);
        for (int i = 0; i < ncenter; i++) {
            while (center != conn_centers[i]) {
                move_right(cg);
                if (me != nullptr)
                    me->left_contract_rotate_unordered(center);
            }
            auto rmat = para_split(i);
            if (me != nullptr)
                me->right_contract_rotate_unordered(center - dot);
            tensors[center] = rmat;
            save_tensor(center);
        }
        while (center != n_sites - 1) {
            move_right(cg);
            if (me != nullptr)
                me->left_contract_rotate_unordered(center);
        }
        move_right(cg);
        for (int i = 0; i < ncenter; i += 2) {
            center = i != ncenter - 1 ? conn_centers[i + 1] - 1 : n_sites - 1;
            while (center != conn_centers[i])
                move_left(cg);
        }
        for (int i = 0; dot == 2 && i < ncenter + 1; i += 2) {
            center = i != ncenter ? conn_centers[i] - 1 : n_sites - 1;
            flip_fused_form(center, cg);
        }
        center = conn_centers[0] - 1;
    }
    void serialize(const shared_ptr<MovingEnvironment<S>> &me) {
        shared_ptr<CG<S>> cg = me->mpo->tf->opf->cg;
        assert(conn_matrices.size() != 0);
        if (canonical_form[n_sites - 1] == 'C')
            canonical_form[n_sites - 1] = 'S';
        else if (canonical_form[n_sites - 1] == 'K')
            flip_fused_form(n_sites - 1, cg);
        if (canonical_form[0] == 'C')
            canonical_form[0] = 'K';
        else if (canonical_form[0] == 'S')
            flip_fused_form(0, cg);
        for (int i = 0; i <= ncenter; i++) {
            center = i == 0 ? 0 : conn_centers[i - 1];
            int j = i == ncenter ? n_sites - 1 : conn_centers[i] - 1;
            if (canonical_form[center] == 'K')
                while (center != j) {
                    move_right(cg);
                    if (me != nullptr)
                        me->left_contract_rotate_unordered(center);
                }
        }
        center = n_sites - 1;
        for (int i = ncenter - 1; i >= 0; i--) {
            while (center != conn_centers[i]) {
                move_left(cg);
                if (me != nullptr)
                    me->right_contract_rotate_unordered(center - dot + 1);
            }
            flip_fused_form(center - 1, cg);
            flip_fused_form(center, cg);
            para_merge(i);
        }
        while (center != 0) {
            move_left(cg);
            if (me != nullptr)
                me->right_contract_rotate_unordered(center - dot + 1);
        }
        me->center = center;
        conn_matrices.clear();
    }
    // K|S -> L|S
    void para_merge(int pidx) {
        int center = conn_centers[pidx];
        assert(canonical_form[center] == 'S');
        assert(canonical_form[center - 1] == 'K');
        load_tensor(center - 1);
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> rot = make_shared<SparseMatrix<S>>(d_alloc);
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        rinfo->initialize_trans_contract(tensors[center - 1]->info,
                                         conn_matrices[pidx]->info,
                                         info->vacuum, true);
        rot->allocate(rinfo);
        rot->contract(tensors[center - 1], conn_matrices[pidx], true);
        tensors[center - 1] = rot;
        save_tensor(center - 1);
        canonical_form[center - 1] = 'L';
    }
    // L|S -> K|S
    shared_ptr<SparseMatrix<S>> para_split(int pidx) {
        int center = conn_centers[pidx];
        assert(canonical_form[center] == 'S');
        assert(canonical_form[center - 1] == 'L');
        load_tensor(center);
        shared_ptr<SparseMatrix<S>> left, middle, right;
        tensors[center]->left_inverse(left, middle, right);
        conn_matrices[pidx] = middle;
        info->right_dims[center] = right->info->extract_state_info(false);
        info->save_right_dims(center);
        shared_ptr<SparseMatrix<S>> rmat = tensors[center];
        tensors[center] = right;
        save_tensor(center);
        canonical_form[center - 1] = 'K';
        load_tensor(center - 1);
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>(d_alloc);
        shared_ptr<SparseMatrixInfo<S>> winfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        winfo->initialize_contract(tensors[center - 1]->info, left->info);
        wfn->allocate(winfo);
        wfn->contract(tensors[center - 1], left);
        tensors[center - 1] = wfn;
        save_tensor(center - 1);
        return rmat;
    }
};

// Site-Parallel Density Matrix Renormalization Group
template <typename S> struct ParallelDMRG : DMRG<S> {
    using DMRG<S>::iprint;
    using DMRG<S>::me;
    using DMRG<S>::sweep_cumulative_nflop;
    using DMRG<S>::sweep_energies;
    using DMRG<S>::sweep_discarded_weights;
    using DMRG<S>::sweep_quanta;
    using DMRG<S>::blocking;
    using typename DMRG<S>::Iteration;
    shared_ptr<ParallelMPS<S>> para_mps;
    ParallelDMRG(const shared_ptr<MovingEnvironment<S>> &me,
                 const vector<ubond_t> &bond_dims, const vector<double> &noises)
        : DMRG<S>(me, bond_dims, noises) {
        para_mps = dynamic_pointer_cast<ParallelMPS<S>>(me->ket);
    }
    void partial_sweep(int ip, bool forward, bool connect, ubond_t bond_dim,
                       double noise, double davidson_conv_thrd) {
        int a = ip == 0 ? 0 : para_mps->conn_centers[ip - 1];
        int b =
            ip == para_mps->ncenter ? me->n_sites : para_mps->conn_centers[ip];
        if (connect) {
            a = para_mps->conn_centers[ip] - 1;
            b = a + me->dot;
        } else
            forward ^= ip & 1;
        if (para_mps->canonical_form[a] == 'C' ||
            para_mps->canonical_form[a] == 'K')
            me->center = a;
        else if (para_mps->canonical_form[b - 1] == 'C' ||
                 para_mps->canonical_form[b - 1] == 'S')
            me->center = b - me->dot;
        else if (para_mps->canonical_form[b - 2] == 'C' ||
                 para_mps->canonical_form[b - 2] == 'K')
            me->center = b - me->dot;
        else
            assert(false);
        me->partial_prepare(a, b);
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < b - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= a; it--)
                sweep_range.push_back(it);
        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                cout << " " << (connect ? "CON" : "PAR") << setw(4) << ip;
                cout << " " << (forward ? "-->" : "<--");
                if (me->dot == 2)
                    cout << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r =
                blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
            sweep_cumulative_nflop += r.nflop;
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            sweep_energies[i] = r.energies;
            sweep_discarded_weights[i] = r.error;
            sweep_quanta[i] = r.quanta;
        }
        if (me->dot == 2 && !connect) {
            if (forward)
                me->left_contract_rotate_unordered(me->center + 1);
            else
                me->right_contract_rotate_unordered(me->center - 1);
        }
    }
    void connection_sweep(int ip, ubond_t bond_dim, double noise,
                          double davidson_conv_thrd) {
        me->center = para_mps->conn_centers[ip] - 1;
        if (para_mps->canonical_form[me->center] == 'C' &&
            para_mps->canonical_form[me->center + 1] == 'C')
            para_mps->canonical_form[me->center] = 'K',
            para_mps->canonical_form[me->center + 1] = 'S';
        else if (para_mps->canonical_form[me->center] == 'S' &&
                 para_mps->canonical_form[me->center + 1] == 'K') {
            para_mps->flip_fused_form(me->center, me->mpo->tf->opf->cg);
            para_mps->flip_fused_form(me->center + 1, me->mpo->tf->opf->cg);
        }
        if (para_mps->canonical_form[me->center] == 'K' &&
            para_mps->canonical_form[me->center + 1] == 'S') {
            para_mps->para_merge(ip);
            partial_sweep(ip, true, true, bond_dim, noise,
                          davidson_conv_thrd); // LK
            me->left_contract_rotate_unordered(me->center + 1);
            para_mps->canonical_form[me->center + 1] = 'K';
            para_mps->flip_fused_form(me->center + 1,
                                      me->mpo->tf->opf->cg); // LS
            auto rmat = para_mps->para_split(ip);            // KR
            me->right_contract_rotate_unordered(me->center - 1);
            para_mps->tensors[me->center + 1] = rmat;
            para_mps->save_tensor(me->center + 1); // KS
            para_mps->flip_fused_form(me->center, me->mpo->tf->opf->cg);
            para_mps->flip_fused_form(me->center + 1,
                                      me->mpo->tf->opf->cg); // SK
        }
    }
    tuple<vector<double>, double, vector<vector<pair<S, double>>>>
    sweep(bool forward, ubond_t bond_dim, double noise,
          double davidson_conv_thrd) override {
        sweep_energies.clear();
        sweep_discarded_weights.clear();
        sweep_quanta.clear();
        sweep_cumulative_nflop = 0;
        frame->reset_peak_used_memory();
        sweep_energies.resize(me->n_sites - me->dot + 1, vector<double>{1E9});
        sweep_discarded_weights.resize(me->n_sites - me->dot + 1);
        sweep_quanta.resize(me->n_sites - me->dot + 1);
        for (int ip = 0; ip < para_mps->ncenter; ip++)
            connection_sweep(ip, bond_dim, noise, davidson_conv_thrd);
        for (int ip = 0; ip <= para_mps->ncenter; ip++)
            partial_sweep(ip, forward, false, bond_dim, noise,
                          davidson_conv_thrd);
        for (int ip = 0; ip < para_mps->ncenter; ip++)
            connection_sweep(ip, bond_dim, noise, davidson_conv_thrd);
        size_t idx =
            min_element(sweep_energies.begin(), sweep_energies.end(),
                        [](const vector<double> &x, const vector<double> &y) {
                            return x.back() < y.back();
                        }) -
            sweep_energies.begin();
        return make_tuple(sweep_energies[idx], sweep_discarded_weights[idx],
                          sweep_quanta[idx]);
    }
};

} // namespace block2
