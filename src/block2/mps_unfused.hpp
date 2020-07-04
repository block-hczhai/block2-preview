
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

#include "matrix.hpp"
#include "mps.hpp"
#include "sparse_matrix.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

// General rank-n dense tensor
struct Tensor {
    vector<int> shape;
    vector<double> data;
    Tensor(int m, int k, int n) : shape{m, k, n} { data.resize(m * k * n); }
    Tensor(const vector<int> &shape) : shape(shape) {
        data.resize(
            accumulate(shape.begin(), shape.end(), 1, multiplies<double>()));
    }
    MatrixRef ref() {
        if (shape.size() == 3 && shape[1] == 1)
            return MatrixRef(&data[0], shape[0], shape[2]);
        else if (shape.size() == 2)
            return MatrixRef(&data[0], shape[0], shape[1]);
        else if (shape.size() == 1)
            return MatrixRef(&data[0], shape[0], 1);
        else {
            assert(false);
            return MatrixRef(&data[0], 0, 1);
        }
    }
    friend ostream &operator<<(ostream &os, const Tensor &ts) {
        os << "TENSOR ( ";
        for (auto sh : ts.shape)
            os << sh << " ";
        os << ")" << endl;
        os << "   DATA [";
        for (auto x : ts.data)
            os << fixed << setw(20) << setprecision(14) << x << " ";
        os << "]" << endl;
        return os;
    }
};

// block-sparse three-index tensor
template <typename S> struct SparseTensor {
    vector<vector<pair<pair<S, S>, shared_ptr<Tensor>>>> data;
    SparseTensor() {}
    SparseTensor(
        const vector<vector<pair<pair<S, S>, shared_ptr<Tensor>>>> &data)
        : data(data) {}
    friend ostream &operator<<(ostream &os, const SparseTensor &spt) {
        int ip = 0;
        for (int ip = 0; ip < (int)spt.data.size(); ip++)
            for (auto &r : spt.data[ip]) {
                os << setw(2) << ip << " LQ=" << setw(20) << r.first.first
                   << " RQ=" << setw(20) << r.first.second << " ";
                os << *r.second;
            }
        return os;
    }
};

template <typename, typename = void> struct UnfusedMPS;

// MPS represented in three-index tensor
template <typename S> struct UnfusedMPS<S, typename S::is_sz_t> {
    shared_ptr<MPSInfo<S>> info;
    vector<shared_ptr<SparseTensor<S>>> tensors;
    UnfusedMPS() {}
    UnfusedMPS(const shared_ptr<MPS<S>> &mps) { this->initialize(mps); }
    static shared_ptr<SparseTensor<S>>
    transform_left_fused(int i, const shared_ptr<MPS<S>> &mps, bool wfn) {
        shared_ptr<SparseTensor<S>> ts = make_shared<SparseTensor<S>>();
        StateInfo<S> m = mps->info->get_basis(i);
        ts->data.resize(m.n);
        mps->info->load_left_dims(i);
        StateInfo<S> l = mps->info->left_dims[i];
        StateInfo<S> lm =
            StateInfo<S>::tensor_product(l, m, mps->info->left_dims_fci[i + 1]);
        StateInfo<S> clm = StateInfo<S>::get_connection_info(l, m, lm);
        shared_ptr<SparseMatrix<S>> mat = mps->tensors[i];
        assert(wfn == mat->info->is_wavefunction);
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = mat->info->quanta[i].get_ket();
            if (wfn)
                ket = -ket;
            int ib = lm.find_state(bra);
            int bbed = ib == lm.n - 1 ? clm.n : clm.n_states[ib + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int bb = clm.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = clm.quanta[bb].data >> 16,
                         ibbb = clm.quanta[bb].data & (0xFFFFU);
                uint32_t lp = (uint32_t)l.n_states[ibba] * m.n_states[ibbb] *
                              mat->info->n_states_ket[i];
                ts->data[ibbb].push_back(make_pair(
                    make_pair(l.quanta[ibba],
                              wfn ? mps->info->target - ket : ket),
                    make_shared<Tensor>(l.n_states[ibba], m.n_states[ibbb],
                                        mat->info->n_states_ket[i])));
                memcpy(&ts->data[ibbb].back().second->data[0], mat->data + p,
                       lp * sizeof(double));
                p += lp;
            }
            assert(p == (i != mat->info->n - 1
                             ? mat->info->n_states_total[i + 1]
                             : mat->total_memory));
        }
        clm.deallocate();
        lm.deallocate();
        l.deallocate();
        return ts;
    }
    static shared_ptr<SparseTensor<S>>
    transform_right_fused(int i, const shared_ptr<MPS<S>> &mps, bool wfn) {
        shared_ptr<SparseTensor<S>> ts = make_shared<SparseTensor<S>>();
        StateInfo<S> m = mps->info->get_basis(i);
        ts->data.resize(m.n);
        mps->info->load_right_dims(i + 1);
        StateInfo<S> r = mps->info->right_dims[i + 1];
        StateInfo<S> mr =
            StateInfo<S>::tensor_product(m, r, mps->info->right_dims_fci[i]);
        StateInfo<S> cmr = StateInfo<S>::get_connection_info(m, r, mr);
        shared_ptr<SparseMatrix<S>> mat = mps->tensors[i];
        assert(wfn == mat->info->is_wavefunction);
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = mat->info->quanta[i].get_ket();
            if (wfn)
                ket = -ket;
            int ik = mr.find_state(ket);
            int kked = ik == mr.n - 1 ? cmr.n : cmr.n_states[ik + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int kk = cmr.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = cmr.quanta[kk].data >> 16,
                         ikkb = cmr.quanta[kk].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                ts->data[ikka].push_back(make_pair(
                    make_pair(wfn ? bra : mps->info->target - bra,
                              mps->info->target - r.quanta[ikkb]),
                    make_shared<Tensor>(mat->info->n_states_bra[i],
                                        m.n_states[ikka], r.n_states[ikkb])));
                for (int ip = 0; ip < mat->info->n_states_bra[i]; ip++)
                    memcpy(&ts->data[ikka].back().second->data[ip * lp],
                           mat->data + p + ip * mat->info->n_states_ket[i],
                           lp * sizeof(double));
                p += lp;
            }
            assert(p - mat->info->n_states_total[i] ==
                   mat->info->n_states_ket[i]);
        }
        cmr.deallocate();
        mr.deallocate();
        r.deallocate();
        return ts;
    }
    static shared_ptr<SparseTensor<S>>
    transform_mps_tensor(int i, const shared_ptr<MPS<S>> &mps) {
        assert(mps->tensors[i] != nullptr);
        mps->load_tensor(i);
        shared_ptr<SparseTensor<S>> ts;
        if (mps->canonical_form[i] == 'L' ||
            (i == 0 && mps->canonical_form[i] == 'C')) {
            ts = transform_left_fused(i, mps, mps->canonical_form[i] == 'C');
        } else if (mps->canonical_form[i] == 'R' ||
                   (i == mps->n_sites - 1 && mps->canonical_form[i] == 'C'))
            ts = transform_right_fused(i, mps, mps->canonical_form[i] == 'C');
        else
            assert(false);
        mps->unload_tensor(i);
        return ts;
    }
    void initialize(const shared_ptr<MPS<S>> &mps) {
        this->info = mps->info;
        tensors.resize(mps->n_sites);
        for (int i = 0; i < mps->n_sites; i++)
            tensors[i] = transform_mps_tensor(i, mps);
    }
};

} // namespace block2
