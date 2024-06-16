
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

#include "../core/matrix.hpp"
#include "../core/sparse_matrix.hpp"
#include "mps.hpp"
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

// block-sparse three-index tensor
template <typename S, typename FL> struct SparseTensor {
    vector<vector<pair<pair<S, S>, shared_ptr<GTensor<FL>>>>> data;
    SparseTensor() {}
    SparseTensor(
        const vector<vector<pair<pair<S, S>, shared_ptr<GTensor<FL>>>>> &data)
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

template <typename S1, typename S2, typename FL, typename = void,
          typename = void>
struct TransSparseTensor;

// Translation between SU2 and SZ SparseTensor
// only works for normal nstate = 1 basis
template <typename S1, typename S2, typename FL>
struct TransSparseTensor<S1, S2, FL, typename S1::is_su2_t,
                         typename S2::is_sz_t> {
    static shared_ptr<SparseTensor<S2, FL>>
    forward(const shared_ptr<SparseTensor<S1, FL>> &spt,
            const shared_ptr<StateInfo<S1>> &basis,
            const shared_ptr<StateInfo<S1>> &left_dim,
            const shared_ptr<StateInfo<S1>> &right_dim,
            const shared_ptr<CG<S1>> &cg, bool left, S2 ref) {
        assert(basis->n == (int)spt->data.size());
        shared_ptr<StateInfo<S2>> tr_basis =
            TransStateInfo<S2, S1>::backward(basis);
        shared_ptr<StateInfo<S2>> tr_left_dim =
            TransStateInfo<S2, S1>::backward(left_dim);
        shared_ptr<StateInfo<S2>> tr_right_dim =
            TransStateInfo<S2, S1>::backward(right_dim);
        shared_ptr<StateInfo<S1>> conn_left_dim =
            TransStateInfo<S2, S1>::backward_connection(left_dim, tr_left_dim);
        shared_ptr<StateInfo<S1>> conn_right_dim =
            TransStateInfo<S2, S1>::backward_connection(right_dim,
                                                        tr_right_dim);
        vector<map<pair<S2, S2>, shared_ptr<GTensor<FL>>>> mp(tr_basis->n);
        for (int ip = 0; ip < basis->n; ip++) {
            S1 mq = basis->quanta[ip];
            for (auto &r : spt->data[ip]) {
                S1 lq = r.first.first, rq = r.first.second;
                for (int imz = -mq.twos(); imz <= mq.twos(); imz += 2)
                    for (int ilz = -lq.twos(); ilz <= lq.twos(); ilz += 2)
                        for (int irz = -rq.twos(); irz <= rq.twos(); irz += 2) {
                            S2 mqz(mq.n(), imz, mq.pg());
                            S2 lqz(lq.n(), ilz, lq.pg());
                            S2 rqz(rq.n(), irz, rq.pg());
                            double factor =
                                left ? cg->cg(lq.twos(), mq.twos(), rq.twos(),
                                              ilz, imz, irz)
                                     : cg->cg(mq.twos(), rq.twos(), lq.twos(),
                                              imz, irz, ilz);
                            if (abs(factor) < TINY)
                                continue;
                            int imqz = tr_basis->find_state(mqz);
                            if (!mp[imqz].count(make_pair(lqz, rqz))) {
                                MKL_INT m =
                                    (MKL_INT)tr_left_dim
                                        ->n_states[tr_left_dim->find_state(
                                            lqz)];
                                MKL_INT k =
                                    (MKL_INT)tr_basis
                                        ->n_states[tr_basis->find_state(mqz)];
                                MKL_INT n =
                                    (MKL_INT)tr_right_dim
                                        ->n_states[tr_right_dim->find_state(
                                            rqz)];
                                mp[imqz][make_pair(lqz, rqz)] =
                                    make_shared<GTensor<FL>>(m, k, n);
                            }
                            shared_ptr<GTensor<FL>> x =
                                mp[imqz].at(make_pair(lqz, rqz));
                            int il = tr_left_dim->find_state(lqz);
                            int ir = tr_right_dim->find_state(rqz);
                            int klst = conn_left_dim->n_states[il];
                            int krst = conn_right_dim->n_states[ir];
                            int kled = il == tr_left_dim->n - 1
                                           ? conn_left_dim->n
                                           : conn_left_dim->n_states[il + 1];
                            int kred = ir == tr_right_dim->n - 1
                                           ? conn_right_dim->n
                                           : conn_right_dim->n_states[ir + 1];
                            MKL_INT lsh = 0, rsh = 0;
                            for (int ilp = klst;
                                 ilp < kled && conn_left_dim->quanta[ilp] != lq;
                                 ilp++)
                                lsh += left_dim->n_states[left_dim->find_state(
                                    conn_left_dim->quanta[ilp])];
                            for (int irp = krst;
                                 irp < kred &&
                                 conn_right_dim->quanta[irp] != rq;
                                 irp++)
                                rsh +=
                                    right_dim->n_states[right_dim->find_state(
                                        conn_right_dim->quanta[irp])];
                            assert(
                                tr_basis->n_states[tr_basis->find_state(mqz)] ==
                                1);
                            for (MKL_INT i = 0; i < r.second->shape[0]; i++)
                                for (MKL_INT j = 0; j < r.second->shape[2]; j++)
                                    (*x)({i + lsh, 0, j + rsh}) =
                                        (FL)factor * (*r.second)({i, 0, j});
                        }
            }
        }
        shared_ptr<SparseTensor<S2, FL>> rst =
            make_shared<SparseTensor<S2, FL>>();
        rst->data.resize(tr_basis->n);
        for (int i = 0; i < tr_basis->n; i++)
            rst->data[i] = vector<pair<pair<S2, S2>, shared_ptr<GTensor<FL>>>>(
                mp[i].cbegin(), mp[i].cend());
        return rst;
    }
};

// Translation between SZ and SGF SparseTensor
// only works for normal nstate = 1 basis
template <typename S1, typename S2, typename FL>
struct TransSparseTensor<S1, S2, FL, typename S1::is_sz_t,
                         typename S2::is_sg_t> {
    static shared_ptr<SparseTensor<S2, FL>>
    forward(const shared_ptr<SparseTensor<S1, FL>> &spt,
            const shared_ptr<StateInfo<S1>> &basis,
            const shared_ptr<StateInfo<S1>> &left_dim,
            const shared_ptr<StateInfo<S1>> &right_dim,
            const shared_ptr<CG<S1>> &cg, bool left, S2 ref) {
        assert(basis->n == (int)spt->data.size());
        shared_ptr<StateInfo<S2>> tr_basis =
            TransStateInfo<S2, S1>::backward(basis);
        shared_ptr<StateInfo<S2>> tr_left_dim =
            TransStateInfo<S2, S1>::backward(left_dim);
        shared_ptr<StateInfo<S2>> tr_right_dim =
            TransStateInfo<S2, S1>::backward(right_dim);
        shared_ptr<StateInfo<S1>> conn_left_dim =
            TransStateInfo<S2, S1>::backward_connection(left_dim, tr_left_dim);
        shared_ptr<StateInfo<S1>> conn_right_dim =
            TransStateInfo<S2, S1>::backward_connection(right_dim,
                                                        tr_right_dim);
        vector<map<pair<S2, S2>, shared_ptr<GTensor<FL>>>> mp(tr_basis->n);
        for (int ip = 0; ip < basis->n; ip++) {
            S1 mq = basis->quanta[ip];
            for (auto &r : spt->data[ip]) {
                S1 lq = r.first.first, rq = r.first.second;
                S2 mqz(mq.n(), mq.pg());
                S2 lqz(lq.n(), lq.pg());
                S2 rqz(rq.n(), rq.pg());
                int imqz = tr_basis->find_state(mqz);
                if (!mp[imqz].count(make_pair(lqz, rqz))) {
                    MKL_INT m =
                        (MKL_INT)
                            tr_left_dim->n_states[tr_left_dim->find_state(lqz)];
                    MKL_INT k =
                        (MKL_INT)tr_basis->n_states[tr_basis->find_state(mqz)];
                    MKL_INT n = (MKL_INT)tr_right_dim
                                    ->n_states[tr_right_dim->find_state(rqz)];
                    mp[imqz][make_pair(lqz, rqz)] =
                        make_shared<GTensor<FL>>(m, k, n);
                }
                shared_ptr<GTensor<FL>> x = mp[imqz].at(make_pair(lqz, rqz));
                int il = tr_left_dim->find_state(lqz);
                int ir = tr_right_dim->find_state(rqz);
                int klst = conn_left_dim->n_states[il];
                int krst = conn_right_dim->n_states[ir];
                int kled = il == tr_left_dim->n - 1
                               ? conn_left_dim->n
                               : conn_left_dim->n_states[il + 1];
                int kred = ir == tr_right_dim->n - 1
                               ? conn_right_dim->n
                               : conn_right_dim->n_states[ir + 1];
                MKL_INT lsh = 0, rsh = 0;
                for (int ilp = klst;
                     ilp < kled && conn_left_dim->quanta[ilp] != lq; ilp++)
                    lsh += left_dim->n_states[left_dim->find_state(
                        conn_left_dim->quanta[ilp])];
                for (int irp = krst;
                     irp < kred && conn_right_dim->quanta[irp] != rq; irp++)
                    rsh += right_dim->n_states[right_dim->find_state(
                        conn_right_dim->quanta[irp])];
                assert(tr_basis->n_states[tr_basis->find_state(mqz)] == 1);
                for (MKL_INT i = 0; i < r.second->shape[0]; i++)
                    for (MKL_INT j = 0; j < r.second->shape[2]; j++)
                        (*x)({i + lsh, 0, j + rsh}) = (*r.second)({i, 0, j});
            }
        }
        shared_ptr<SparseTensor<S2, FL>> rst =
            make_shared<SparseTensor<S2, FL>>();
        rst->data.resize(tr_basis->n);
        for (int i = 0; i < tr_basis->n; i++)
            rst->data[i] = vector<pair<pair<S2, S2>, shared_ptr<GTensor<FL>>>>(
                mp[i].cbegin(), mp[i].cend());
        return rst;
    }
};

// Translation between SAny SparseTensor
template <typename S, typename FL>
struct TransSparseTensor<S, S, FL, typename S::is_sany_t,
                         typename S::is_sany_t> {
    static shared_ptr<SparseTensor<S, FL>>
    forward(const shared_ptr<SparseTensor<S, FL>> &spt,
            const shared_ptr<StateInfo<S>> &basis,
            const shared_ptr<StateInfo<S>> &left_dim,
            const shared_ptr<StateInfo<S>> &right_dim,
            const shared_ptr<CG<S>> &cg, bool left, S ref) {
        assert(basis->n == (int)spt->data.size());
        shared_ptr<StateInfo<S>> tr_basis =
            TransStateInfo<S, S>::backward(basis, ref);
        shared_ptr<StateInfo<S>> tr_left_dim =
            TransStateInfo<S, S>::backward(left_dim, ref);
        shared_ptr<StateInfo<S>> tr_right_dim =
            TransStateInfo<S, S>::backward(right_dim, ref);
        shared_ptr<StateInfo<S>> conn_basis =
            TransStateInfo<S, S>::backward_connection(basis, tr_basis);
        shared_ptr<StateInfo<S>> conn_left_dim =
            TransStateInfo<S, S>::backward_connection(left_dim, tr_left_dim);
        shared_ptr<StateInfo<S>> conn_right_dim =
            TransStateInfo<S, S>::backward_connection(right_dim, tr_right_dim);
        vector<map<pair<S, S>, shared_ptr<GTensor<FL>>>> mp(tr_basis->n);
        for (int ip = 0; ip < basis->n; ip++) {
            S mq = basis->quanta[ip];
            for (auto &r : spt->data[ip]) {
                S lq = r.first.first, rq = r.first.second;
                shared_ptr<StateInfo<S>> mqzs = TransStateInfo<S, S>::forward(
                    make_shared<StateInfo<S>>(mq), ref);
                shared_ptr<StateInfo<S>> lqzs = TransStateInfo<S, S>::forward(
                    make_shared<StateInfo<S>>(lq), ref);
                shared_ptr<StateInfo<S>> rqzs = TransStateInfo<S, S>::forward(
                    make_shared<StateInfo<S>>(rq), ref);
                for (int imz = 0; imz < mqzs->n; imz++)
                    for (int ilz = 0; ilz < lqzs->n; ilz++)
                        for (int irz = 0; irz < rqzs->n; irz++) {
                            S mqz = mqzs->quanta[imz], lqz = lqzs->quanta[ilz],
                              rqz = rqzs->quanta[irz];
                            double factor =
                                left ? cg->cg(lq, mq, rq, lqz, mqz, rqz)
                                     : cg->cg(mq, rq, lq, mqz, rqz, lqz);
                            if (abs(factor) < TINY)
                                continue;
                            int imqz = tr_basis->find_state(mqz);
                            if (!mp[imqz].count(make_pair(lqz, rqz))) {
                                MKL_INT m =
                                    (MKL_INT)tr_left_dim
                                        ->n_states[tr_left_dim->find_state(
                                            lqz)];
                                MKL_INT k =
                                    (MKL_INT)tr_basis
                                        ->n_states[tr_basis->find_state(mqz)];
                                MKL_INT n =
                                    (MKL_INT)tr_right_dim
                                        ->n_states[tr_right_dim->find_state(
                                            rqz)];
                                mp[imqz][make_pair(lqz, rqz)] =
                                    make_shared<GTensor<FL>>(m, k, n);
                            }
                            shared_ptr<GTensor<FL>> x =
                                mp[imqz].at(make_pair(lqz, rqz));
                            int il = tr_left_dim->find_state(lqz);
                            int ir = tr_right_dim->find_state(rqz);
                            int kmst = conn_basis->n_states[imqz];
                            int klst = conn_left_dim->n_states[il];
                            int krst = conn_right_dim->n_states[ir];
                            int kmed = imqz == tr_basis->n - 1
                                           ? conn_basis->n
                                           : conn_basis->n_states[imqz + 1];
                            int kled = il == tr_left_dim->n - 1
                                           ? conn_left_dim->n
                                           : conn_left_dim->n_states[il + 1];
                            int kred = ir == tr_right_dim->n - 1
                                           ? conn_right_dim->n
                                           : conn_right_dim->n_states[ir + 1];
                            MKL_INT msh = 0, lsh = 0, rsh = 0;
                            for (int imp = kmst;
                                 imp < kmed && conn_basis->quanta[imp] != mq;
                                 imp++)
                                msh += basis->n_states[basis->find_state(
                                    conn_basis->quanta[imp])];
                            for (int ilp = klst;
                                 ilp < kled && conn_left_dim->quanta[ilp] != lq;
                                 ilp++)
                                lsh += left_dim->n_states[left_dim->find_state(
                                    conn_left_dim->quanta[ilp])];
                            for (int irp = krst;
                                 irp < kred &&
                                 conn_right_dim->quanta[irp] != rq;
                                 irp++)
                                rsh +=
                                    right_dim->n_states[right_dim->find_state(
                                        conn_right_dim->quanta[irp])];
                            for (MKL_INT i = 0; i < r.second->shape[0]; i++)
                                for (MKL_INT k = 0; k < r.second->shape[1]; k++)
                                    for (MKL_INT j = 0; j < r.second->shape[2];
                                         j++)
                                        (*x)({i + lsh, k + msh, j + rsh}) =
                                            (FL)factor * (*r.second)({i, k, j});
                        }
            }
        }
        shared_ptr<SparseTensor<S, FL>> rst =
            make_shared<SparseTensor<S, FL>>();
        rst->data.resize(tr_basis->n);
        for (int i = 0; i < tr_basis->n; i++)
            rst->data[i] = vector<pair<pair<S, S>, shared_ptr<GTensor<FL>>>>(
                mp[i].cbegin(), mp[i].cend());
        return rst;
    }
};

// MPS represented in three-index tensor
template <typename S, typename FL> struct UnfusedMPS {
    shared_ptr<MPSInfo<S>> info;
    vector<shared_ptr<SparseTensor<S, FL>>> tensors;
    string canonical_form;
    int center, n_sites, dot;
    UnfusedMPS() {}
    UnfusedMPS(const shared_ptr<MPS<S, FL>> &mps) { this->initialize(mps); }
    static shared_ptr<SparseTensor<S, FL>>
    forward_left_fused(int ii, const shared_ptr<MPS<S, FL>> &mps, bool wfn) {
        shared_ptr<SparseTensor<S, FL>> ts = make_shared<SparseTensor<S, FL>>();
        StateInfo<S> m = *mps->info->basis[ii];
        ts->data.resize(m.n);
        mps->info->load_left_dims(ii);
        StateInfo<S> l = *mps->info->left_dims[ii];
        StateInfo<S> lm = StateInfo<S>::tensor_product(
            l, m, *mps->info->left_dims_fci[ii + 1]);
        shared_ptr<typename StateInfo<S>::ConnectionInfo> clm =
            StateInfo<S>::get_connection_info(l, m, lm);
        shared_ptr<SparseMatrix<S, FL>> mat = mps->tensors[ii];
        assert(wfn == mat->info->is_wavefunction);
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = mat->info->quanta[i].get_ket();
            if (wfn)
                ket = -ket;
            int ib = lm.find_state(bra);
            int bbed = clm->acc_n_states[ib + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int bb = clm->acc_n_states[ib]; bb < bbed; bb++) {
                uint32_t ibba = clm->ij_indices[bb].first,
                         ibbb = clm->ij_indices[bb].second;
                uint32_t lp = (uint32_t)l.n_states[ibba] * m.n_states[ibbb] *
                              mat->info->n_states_ket[i];
                ts->data[ibbb].push_back(make_pair(
                    make_pair(l.quanta[ibba],
                              wfn ? mps->info->target - ket : ket),
                    make_shared<GTensor<FL>>(l.n_states[ibba], m.n_states[ibbb],
                                             mat->info->n_states_ket[i])));
                memcpy(ts->data[ibbb].back().second->data->data(),
                       mat->data + p, lp * sizeof(FL));
                p += lp;
            }
            assert(p == (i != mat->info->n - 1
                             ? mat->info->n_states_total[i + 1]
                             : mat->total_memory));
        }
        lm.deallocate();
        l.deallocate();
        return ts;
    }
    static shared_ptr<SparseTensor<S, FL>>
    forward_right_fused(int ii, const shared_ptr<MPS<S, FL>> &mps, bool wfn) {
        shared_ptr<SparseTensor<S, FL>> ts = make_shared<SparseTensor<S, FL>>();
        StateInfo<S> m = *mps->info->basis[ii];
        ts->data.resize(m.n);
        mps->info->load_right_dims(ii + 1);
        StateInfo<S> r = *mps->info->right_dims[ii + 1];
        StateInfo<S> mr =
            StateInfo<S>::tensor_product(m, r, *mps->info->right_dims_fci[ii]);
        shared_ptr<typename StateInfo<S>::ConnectionInfo> cmr =
            StateInfo<S>::get_connection_info(m, r, mr);
        shared_ptr<SparseMatrix<S, FL>> mat = mps->tensors[ii];
        assert(wfn == mat->info->is_wavefunction);
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = mat->info->quanta[i].get_ket();
            if (wfn)
                ket = -ket;
            int ik = mr.find_state(ket);
            int kked = cmr->acc_n_states[ik + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int kk = cmr->acc_n_states[ik]; kk < kked; kk++) {
                uint32_t ikka = cmr->ij_indices[kk].first,
                         ikkb = cmr->ij_indices[kk].second;
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                ts->data[ikka].push_back(make_pair(
                    make_pair(wfn ? bra : mps->info->target - bra,
                              mps->info->target - r.quanta[ikkb]),
                    make_shared<GTensor<FL>>(mat->info->n_states_bra[i],
                                             m.n_states[ikka],
                                             r.n_states[ikkb])));
                for (int ip = 0; ip < (int)mat->info->n_states_bra[i]; ip++)
                    memcpy(&(*ts->data[ikka].back().second->data)[ip * lp],
                           mat->data + p + ip * mat->info->n_states_ket[i],
                           lp * sizeof(FL));
                p += lp;
            }
            assert(p - mat->info->n_states_total[i] ==
                   mat->info->n_states_ket[i]);
        }
        mr.deallocate();
        r.deallocate();
        return ts;
    }
    static shared_ptr<SparseTensor<S, FL>>
    forward_mps_tensor(int i, const shared_ptr<MPS<S, FL>> &mps) {
        assert(mps->tensors[i] != nullptr);
        mps->load_tensor(i);
        shared_ptr<SparseTensor<S, FL>> ts;
        if (mps->canonical_form[i] == 'L' || mps->canonical_form[i] == 'K' ||
            (i == 0 && mps->canonical_form[i] == 'C')) {
            ts = forward_left_fused(i, mps,
                                    mps->canonical_form[i] == 'C' ||
                                        mps->canonical_form[i] == 'K');
        } else if (mps->canonical_form[i] == 'R' ||
                   mps->canonical_form[i] == 'S' ||
                   (i == mps->n_sites - 1 && mps->canonical_form[i] == 'C'))
            ts = forward_right_fused(i, mps,
                                     mps->canonical_form[i] == 'C' ||
                                         mps->canonical_form[i] == 'S');
        else
            assert(false);
        mps->unload_tensor(i);
        return ts;
    }
    static shared_ptr<SparseMatrix<S, FL>>
    backward_left_fused(int ii, const shared_ptr<MPS<S, FL>> &mps,
                        const shared_ptr<SparseTensor<S, FL>> &spt, bool wfn) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<typename GMatrix<FL>::FP>> d_alloc =
            make_shared<VectorAllocator<typename GMatrix<FL>::FP>>();
        StateInfo<S> m = *mps->info->basis[ii];
        StateInfo<S> l = *mps->info->left_dims[ii];
        StateInfo<S> lm = StateInfo<S>::tensor_product(
            l, m, *mps->info->left_dims_fci[ii + 1]);
        shared_ptr<typename StateInfo<S>::ConnectionInfo> clm =
            StateInfo<S>::get_connection_info(l, m, lm);
        shared_ptr<SparseMatrixInfo<S>> minfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        if (wfn)
            minfo->initialize(lm, *mps->info->right_dims[ii + 1],
                              mps->info->target, false, true);
        else
            minfo->initialize(lm, *mps->info->left_dims[ii + 1],
                              mps->info->vacuum, false);
        shared_ptr<SparseMatrix<S, FL>> mat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        mat->allocate(minfo);
        assert(wfn == mat->info->is_wavefunction);
        vector<map<pair<S, S>, shared_ptr<GTensor<FL>>>> mp(spt->data.size());
        for (size_t i = 0; i < spt->data.size(); i++)
            mp[i] = map<pair<S, S>, shared_ptr<GTensor<FL>>>(
                spt->data[i].cbegin(), spt->data[i].cend());
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = mat->info->quanta[i].get_ket();
            if (wfn)
                ket = -ket;
            int ib = lm.find_state(bra);
            int bbed = clm->acc_n_states[ib + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int bb = clm->acc_n_states[ib]; bb < bbed; bb++) {
                uint32_t ibba = clm->ij_indices[bb].first,
                         ibbb = clm->ij_indices[bb].second;
                uint32_t lp = (uint32_t)l.n_states[ibba] * m.n_states[ibbb] *
                              mat->info->n_states_ket[i];
                pair<S, S> qq = make_pair(l.quanta[ibba],
                                          wfn ? mps->info->target - ket : ket);
                if (mp[ibbb].count(qq)) {
                    shared_ptr<GTensor<FL>> ts = mp[ibbb].at(qq);
                    assert(ts->shape[0] == l.n_states[ibba]);
                    assert(ts->shape[1] == m.n_states[ibbb]);
                    assert(ts->shape[2] == mat->info->n_states_ket[i]);
                    memcpy(mat->data + p, ts->data->data(), lp * sizeof(FL));
                }
                p += lp;
            }
            assert(p == (i != mat->info->n - 1
                             ? mat->info->n_states_total[i + 1]
                             : mat->total_memory));
        }
        lm.deallocate();
        return mat;
    }
    static shared_ptr<SparseMatrix<S, FL>>
    backward_right_fused(int ii, const shared_ptr<MPS<S, FL>> &mps,
                         const shared_ptr<SparseTensor<S, FL>> &spt, bool wfn) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<typename GMatrix<FL>::FP>> d_alloc =
            make_shared<VectorAllocator<typename GMatrix<FL>::FP>>();
        StateInfo<S> m = *mps->info->basis[ii];
        StateInfo<S> r = *mps->info->right_dims[ii + 1];
        StateInfo<S> mr =
            StateInfo<S>::tensor_product(m, r, *mps->info->right_dims_fci[ii]);
        shared_ptr<typename StateInfo<S>::ConnectionInfo> cmr =
            StateInfo<S>::get_connection_info(m, r, mr);
        shared_ptr<SparseMatrixInfo<S>> minfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        if (wfn)
            minfo->initialize(*mps->info->left_dims[ii], mr, mps->info->target,
                              false, true);
        else
            minfo->initialize(*mps->info->right_dims[ii], mr, mps->info->vacuum,
                              false);
        shared_ptr<SparseMatrix<S, FL>> mat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        mat->allocate(minfo);
        assert(wfn == mat->info->is_wavefunction);
        vector<map<pair<S, S>, shared_ptr<GTensor<FL>>>> mp(spt->data.size());
        for (size_t i = 0; i < spt->data.size(); i++)
            mp[i] = map<pair<S, S>, shared_ptr<GTensor<FL>>>(
                spt->data[i].cbegin(), spt->data[i].cend());
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = mat->info->quanta[i].get_ket();
            if (wfn)
                ket = -ket;
            int ik = mr.find_state(ket);
            int kked = cmr->acc_n_states[ik + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int kk = cmr->acc_n_states[ik]; kk < kked; kk++) {
                uint32_t ikka = cmr->ij_indices[kk].first,
                         ikkb = cmr->ij_indices[kk].second;
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                pair<S, S> qq = make_pair(wfn ? bra : mps->info->target - bra,
                                          mps->info->target - r.quanta[ikkb]);
                if (mp[ikka].count(qq)) {
                    shared_ptr<GTensor<FL>> ts = mp[ikka].at(qq);
                    assert(ts->shape[0] == mat->info->n_states_bra[i]);
                    assert(ts->shape[1] == m.n_states[ikka]);
                    assert(ts->shape[2] == r.n_states[ikkb]);
                    for (int ip = 0; ip < (int)mat->info->n_states_bra[i]; ip++)
                        memcpy(mat->data + p + ip * mat->info->n_states_ket[i],
                               &(*ts->data)[ip * lp], lp * sizeof(FL));
                }
                p += lp;
            }
            assert(p - mat->info->n_states_total[i] ==
                   mat->info->n_states_ket[i]);
        }
        mr.deallocate();
        return mat;
    }
    static shared_ptr<SparseMatrix<S, FL>>
    backward_mps_tensor(int i, const shared_ptr<MPS<S, FL>> &mps,
                        const shared_ptr<SparseTensor<S, FL>> &spt) {
        shared_ptr<SparseMatrix<S, FL>> mat;
        if (mps->canonical_form[i] == 'L' || mps->canonical_form[i] == 'K' ||
            (i == 0 && mps->canonical_form[i] == 'C')) {
            mat = backward_left_fused(i, mps, spt,
                                      mps->canonical_form[i] == 'C' ||
                                          mps->canonical_form[i] == 'K');
        } else if (mps->canonical_form[i] == 'R' ||
                   mps->canonical_form[i] == 'S' ||
                   (i == mps->n_sites - 1 && mps->canonical_form[i] == 'C'))
            mat = backward_right_fused(i, mps, spt,
                                       mps->canonical_form[i] == 'C' ||
                                           mps->canonical_form[i] == 'S');
        else
            assert(false);
        return mat;
    }
    void initialize(const shared_ptr<MPS<S, FL>> &mps) {
        this->info = mps->info;
        canonical_form = mps->canonical_form;
        center = mps->center;
        n_sites = mps->n_sites;
        dot = mps->dot;
        tensors.resize(mps->n_sites);
        for (int i = 0; i < mps->n_sites; i++)
            tensors[i] = forward_mps_tensor(i, mps);
    }
    // Transform from Unfused MPS to normal MPS
    shared_ptr<MPS<S, FL>>
    finalize(const shared_ptr<ParallelRule<S>> &para_rule = nullptr) const {
        info->load_mutable();
        shared_ptr<MPS<S, FL>> xmps = make_shared<MPS<S, FL>>(info);
        xmps->canonical_form = canonical_form;
        xmps->center = center;
        xmps->n_sites = n_sites;
        xmps->dot = dot;
        xmps->tensors.resize(n_sites);
        for (int i = 0; i < xmps->n_sites; i++)
            xmps->tensors[i] = backward_mps_tensor(i, xmps, tensors[i]);
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        if (para_rule == nullptr || para_rule->is_root()) {
            info->save_mutable();
            xmps->save_mutable();
            xmps->save_data();
        }
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        xmps->deallocate();
        info->deallocate_mutable();
        return xmps;
    }
    // select one sz component from SZ Unfused MPS transformed from SU2 MPS
    void resolve_singlet_embedding(int twosz) {
        assert(info->target.twos() == 0);
        vector<S> lqs;
        int lidx = -1;
        for (int i = 0; i < info->left_dims_fci[0]->n; i++) {
            lqs.push_back(info->left_dims_fci[0]->quanta[i]);
            if (-twosz == lqs.back().twos()) {
                assert(lidx == -1);
                lidx = i;
            }
        }
        assert(lidx != -1);
        S lq = lqs[lidx];
        info->set_bond_dimension_full_fci(lq, info->vacuum);
        info->load_mutable();
        info->left_dims[0] = make_shared<StateInfo<S>>(lq);
        info->target = info->target - lq;
        info->set_bond_dimension_fci();
        for (int i = 0; i <= n_sites; i++) {
            for (int j = 0; j < info->left_dims[i]->n; j++)
                info->left_dims[i]->quanta[j] =
                    info->left_dims[i]->quanta[j] - lq;
            info->left_dims[i]->sort_states();
        }
        info->save_mutable();
        info->deallocate_mutable();
        shared_ptr<SparseTensor<S, FL>> rst =
            make_shared<SparseTensor<S, FL>>();
        rst->data.resize(info->basis[0]->n);
        for (int i = 0; i < info->basis[0]->n; i++) {
            for (auto &x : tensors[0]->data[i])
                if (x.first.first == lq)
                    rst->data[i].push_back(x);
        }
        tensors[0] = rst;
        for (int i = 0; i < n_sites; i++)
            for (size_t j = 0; j < tensors[i]->data.size(); j++)
                for (auto &x : tensors[i]->data[j]) {
                    x.first.first = x.first.first - lq;
                    x.first.second = x.first.second - lq;
                }
    }
};

// Translation between SU2 and SZ / SZ and SGF unfused MPS
// only works for normal nstate = 1 basis
template <typename S1, typename S2, typename FL, typename = void,
          typename = void>
struct TransUnfusedMPS {
    static shared_ptr<UnfusedMPS<S2, FL>>
    forward(const shared_ptr<UnfusedMPS<S1, FL>> &umps, const string &xtag,
            const shared_ptr<CG<S1>> &cg, S2 target) {
        shared_ptr<UnfusedMPS<S2, FL>> fmps = make_shared<UnfusedMPS<S2, FL>>();
        umps->info->load_mutable();
        fmps->info = TransMPSInfo<S1, S2>::forward(umps->info, target);
        fmps->info->tag = xtag;
        fmps->info->save_mutable();
        fmps->tensors.resize(umps->tensors.size());
        fmps->canonical_form = umps->canonical_form;
        fmps->center = umps->center;
        fmps->n_sites = umps->n_sites;
        fmps->dot = umps->dot;
        umps->info->load_mutable();
        for (int i = 0; i < umps->n_sites; i++)
            if (umps->canonical_form[i] == 'L')
                fmps->tensors[i] = TransSparseTensor<S1, S2, FL>::forward(
                    umps->tensors[i], umps->info->basis[i],
                    umps->info->left_dims[i], umps->info->left_dims[i + 1], cg,
                    true, target);
            else if (umps->canonical_form[i] == 'R') {
                shared_ptr<StateInfo<S1>> ri =
                    make_shared<StateInfo<S1>>(StateInfo<S1>::complementary(
                        *umps->info->right_dims[i], umps->info->target));
                shared_ptr<StateInfo<S1>> rj =
                    make_shared<StateInfo<S1>>(StateInfo<S1>::complementary(
                        *umps->info->right_dims[i + 1], umps->info->target));
                fmps->tensors[i] = TransSparseTensor<S1, S2, FL>::forward(
                    umps->tensors[i], umps->info->basis[i], ri, rj, cg, true,
                    target);
            } else {
                shared_ptr<StateInfo<S1>> ri =
                    make_shared<StateInfo<S1>>(StateInfo<S1>::complementary(
                        *umps->info->right_dims[i + 1], umps->info->target));
                fmps->tensors[i] = TransSparseTensor<S1, S2, FL>::forward(
                    umps->tensors[i], umps->info->basis[i],
                    umps->info->left_dims[i], ri, cg, true, target);
            }
        umps->info->deallocate_mutable();
        return fmps;
    }
};

} // namespace block2
