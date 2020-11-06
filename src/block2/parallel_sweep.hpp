
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

// Matrix Product State
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
    vector<int> para_centers;
    vector<shared_ptr<SparseMatrix<S>>> para_matrices;
    int ncenter = 0;
    ParallelMPS(const shared_ptr<MPSInfo<S>> &info) : MPS<S>(info) {}
    ParallelMPS(int n_sites, int center, int dot)
        : MPS<S>(n_sites, center, dot) {}
    void prepare(const vector<int> &para_centers,
                 const shared_ptr<MovingEnvironment<S>> &me = nullptr) {
        shared_ptr<CG<S>> cg = me->mpo->tf->opf->cg;
        this->para_centers = para_centers;
        ncenter = para_centers.size();
        para_matrices.resize(ncenter);
        while (center != 0) {
            move_left(cg);
            if (me != nullptr && center - dot + 1 >= 0)
                me->right_contract_rotate(center - dot + 1);
        }
        assert(center == 0);
        for (int i = 0; i < ncenter; i++) {
            while (center != para_centers[i]) {
                move_right(cg);
                if (me != nullptr && center - 1 >= 0)
                    me->left_contract_rotate(center);
            }
            para_split();
        }
        while (center != n_sites - 1)
            move_right(cg);
        move_right(cg);
        for (int i = 0; i < ncenter; i += 2) {
            center = i != ncenter - 1 ? para_centers[i + 1] - 1 : n_sites - 1;
            while (center != para_centers[i])
                move_left(cg);
        }
        for (int i = 0; dot == 2 && i < ncenter + 1; i += 2) {
            center = i != ncenter ? para_centers[i] - 1 : n_sites - 1;
            move_left(cg);
        }
    }
    // LL|S -> LK|S
    void para_split() {
        int pidx = -1;
        for (int i = 0; i < (int)para_centers.size(); i++)
            if (para_centers[i] == center)
                pidx = i;
        assert(pidx != -1);
        assert(canonical_form[center] == 'S');
        assert(canonical_form[center - 1] == 'L');
        load_tensor(center);
        shared_ptr<SparseMatrix<S>> left, right;
        tensors[center]->left_inverse(left, right);
        para_matrices[pidx] = right;
        // info->right_dims[center] = left->info->extract_state_info(true);
        // info->save_right_dims(center);
        unload_tensor(center);
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
    void partial_sweep(int ip, bool forward, ubond_t bond_dim, double noise,
                       double davidson_conv_thrd) {
        int a = ip == 0 ? 0 : para_mps->para_centers[ip - 1];
        int b = ip == para_mps->ncenter ? me->n_sites : para_mps->para_centers[ip];
        me->center = (forward ^ (ip & 1)) ? a : b - me->dot;
        me->partial_prepare(a, b);
        vector<int> sweep_range;
        if (forward ^ (ip & 1))
            for (int it = me->center; it < b - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= a; it--)
                sweep_range.push_back(it);
        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << " .. ";
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
    }
    tuple<vector<double>, double, vector<vector<pair<S, double>>>>
    sweep(bool forward, ubond_t bond_dim, double noise,
          double davidson_conv_thrd) override {
        sweep_energies.clear();
        sweep_discarded_weights.clear();
        sweep_quanta.clear();
        sweep_cumulative_nflop = 0;
        frame->reset_peak_used_memory();
        sweep_energies.resize(me->n_sites - me->dot + 1);
        sweep_discarded_weights.resize(me->n_sites - me->dot + 1);
        sweep_quanta.resize(me->n_sites - me->dot + 1);
        for (int ip = 1; ip <= para_mps->ncenter; ip++) {
            cout << "PARA = " << setw(4) << ip << endl;
            partial_sweep(ip, forward, bond_dim, noise, davidson_conv_thrd);
        }
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
