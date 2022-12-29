
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

#include "batch_gemm.hpp"
#include "cg.hpp"
#include "matrix_functions.hpp"
#include "sparse_matrix.hpp"
#include <array>
#include <cassert>
#include <memory>

using namespace std;

namespace block2 {

enum struct NoiseTypes : uint16_t {
    None = 0,
    Wavefunction = 1,
    DensityMatrix = 2,
    Perturbative = 4,
    Reduced = 8,
    Unscaled = 16,
    Collected = 32,
    LowMem = 64,
    ReducedPerturbative = 4 | 8,
    PerturbativeUnscaled = 4 | 16,
    ReducedPerturbativeUnscaled = 4 | 8 | 16,
    PerturbativeCollected = 4 | 32,
    PerturbativeUnscaledCollected = 4 | 16 | 32,
    ReducedPerturbativeCollected = 4 | 8 | 32,
    ReducedPerturbativeUnscaledCollected = 4 | 8 | 16 | 32,
    ReducedPerturbativeLowMem = 4 | 8 | 64,
    ReducedPerturbativeUnscaledLowMem = 4 | 8 | 16 | 64,
    ReducedPerturbativeCollectedLowMem = 4 | 8 | 32 | 64,
    ReducedPerturbativeUnscaledCollectedLowMem = 4 | 8 | 16 | 32 | 64
};

enum struct TraceTypes : uint8_t { None = 0, Left = 1, Right = 2 };

inline bool operator&(NoiseTypes a, NoiseTypes b) {
    return ((uint16_t)a & (uint16_t)b) != 0;
}

inline NoiseTypes operator|(NoiseTypes a, NoiseTypes b) {
    return NoiseTypes((uint16_t)a | (uint16_t)b);
}

// SparseMatrix operations
template <typename S, typename FL> struct OperatorFunctions {
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<CG<S>> cg;
    shared_ptr<BatchGEMMSeq<FL>> seq = nullptr;
    OperatorFunctions(const shared_ptr<CG<S>> &cg) : cg(cg) {
        seq = make_shared<BatchGEMMSeq<FL>>(0, threading->seq_type);
    }
    virtual ~OperatorFunctions() = default;
    virtual SparseMatrixTypes get_type() const {
        return SparseMatrixTypes::Normal;
    }
    virtual shared_ptr<OperatorFunctions> copy() const {
        shared_ptr<OperatorFunctions> opf =
            make_shared<OperatorFunctions>(this->cg);
        opf->seq = this->seq->copy();
        return opf;
    }
    virtual void
    parallel_reduce(const vector<shared_ptr<SparseMatrix<S, FL>>> &mats, int i,
                    int j) const {
        assert(j > i);
        if (j - i == 1)
            return;
        assert(mats[i]->get_type() == SparseMatrixTypes::Normal);
        int m = (i + j) >> 1;
#ifdef _MSC_VER
        parallel_reduce(mats, i, m);
        parallel_reduce(mats, m, j);
#else
#pragma omp task
        parallel_reduce(mats, i, m);
#pragma omp task
        parallel_reduce(mats, m, j);
#pragma omp taskwait
#endif
        GMatrixFunctions<FL>::iadd(
            GMatrix<FL>(mats[i]->data, 1, (MKL_INT)mats[i]->total_memory),
            GMatrix<FL>(mats[m]->data, 1, (MKL_INT)mats[m]->total_memory), 1.0);
    }
    virtual void
    parallel_reduce(const vector<shared_ptr<SparseMatrixGroup<S, FL>>> &mats,
                    int i, int j) const {
        assert(j > i);
        if (j - i == 1)
            return;
        int m = (i + j) >> 1;
#ifdef _MSC_VER
        parallel_reduce(mats, i, m);
        parallel_reduce(mats, m, j);
#else
#pragma omp task
        parallel_reduce(mats, i, m);
#pragma omp task
        parallel_reduce(mats, m, j);
#pragma omp taskwait
#endif
        // avoid possible int32 overflow
        for (int j = 0; j < mats[i]->n; j++)
            GMatrixFunctions<FL>::iadd(
                GMatrix<FL>((*mats[i])[j]->data, 1,
                            (MKL_INT)(*mats[i])[j]->total_memory),
                GMatrix<FL>((*mats[m])[j]->data, 1,
                            (MKL_INT)(*mats[m])[j]->total_memory),
                1.0);
    }
    // a += b * scale
    virtual void iadd(const shared_ptr<SparseMatrix<S, FL>> &a,
                      const shared_ptr<SparseMatrix<S, FL>> &b, FL scale = 1.0,
                      bool conj = false) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal);
        if (a->info == b->info && !conj) {
            if (seq->mode != SeqTypes::None && seq->mode != SeqTypes::Tasked)
                seq->iadd(GMatrix<FL>(a->data, 1, (MKL_INT)a->total_memory),
                          GMatrix<FL>(b->data, 1, (MKL_INT)b->total_memory),
                          scale * b->factor, false, a->factor);
            else
                GMatrixFunctions<FL>::iadd(
                    GMatrix<FL>(a->data, 1, (MKL_INT)a->total_memory),
                    GMatrix<FL>(b->data, 1, (MKL_INT)b->total_memory),
                    scale * b->factor, false, a->factor);
        } else {
            S bdq = b->info->delta_quantum;
            for (int ia = 0, ib; ia < a->info->n; ia++) {
                S bra = a->info->quanta[ia].get_bra(a->info->delta_quantum);
                S ket = a->info->quanta[ia].get_ket();
                S bq = conj ? bdq.combine(ket, bra) : bdq.combine(bra, ket);
                if (bq != S(S::invalid) &&
                    ((ib = b->info->find_state(bq)) != -1)) {
                    FL factor =
                        scale * (conj ? xconj<FL>(b->factor) : b->factor);
                    if (conj)
                        factor *= cg->transpose_cg(bdq, bra, ket);
                    if (seq->mode != SeqTypes::None &&
                        seq->mode != SeqTypes::Tasked)
                        seq->iadd((*a)[ia], (*b)[ib], factor, conj, a->factor);
                    else
                        GMatrixFunctions<FL>::iadd((*a)[ia], (*b)[ib], factor,
                                                   conj, a->factor);
                }
            }
        }
        a->factor = 1.0;
    }
    virtual void tensor_rotate(const shared_ptr<SparseMatrix<S, FL>> &a,
                               const shared_ptr<SparseMatrix<S, FL>> &c,
                               const shared_ptr<SparseMatrix<S, FL>> &rot_bra,
                               const shared_ptr<SparseMatrix<S, FL>> &rot_ket,
                               bool trans, FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal &&
               rot_bra->get_type() == SparseMatrixTypes::Normal &&
               rot_ket->get_type() == SparseMatrixTypes::Normal);
        scale =
            scale * a->factor * xconj<FL>(rot_bra->factor) * rot_ket->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, cdq = c->info->delta_quantum;
        // possible assert failure here due to the same bra/ket tag
        assert(adq == cdq && a->info->n >= c->info->n);
        for (int ic = 0, ia = 0; ic < c->info->n; ia++, ic++) {
            while (a->info->quanta[ia] != c->info->quanta[ic])
                ia++;
            S cq = c->info->quanta[ic].get_bra(cdq);
            S cqprime = c->info->quanta[ic].get_ket();
            int ibra = rot_bra->info->find_state(cq);
            int iket = rot_ket->info->find_state(cqprime);
            if (seq->mode != SeqTypes::None && seq->mode != SeqTypes::Tasked)
                seq->rotate((*a)[ia], (*c)[ic], (*rot_bra)[ibra], !trans | 2,
                            (*rot_ket)[iket], trans, scale);
            else
                GMatrixFunctions<FL>::rotate((*a)[ia], (*c)[ic],
                                             (*rot_bra)[ibra], !trans | 2,
                                             (*rot_ket)[iket], trans, scale);
        }
        if (seq->mode & SeqTypes::Simple)
            seq->simple_perform();
    }
    virtual void
    tensor_product_diagonal(uint8_t conj,
                            const shared_ptr<SparseMatrix<S, FL>> &a,
                            const shared_ptr<SparseMatrix<S, FL>> &b,
                            const shared_ptr<SparseMatrix<S, FL>> &c, S opdq,
                            FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal);
        scale = scale * a->factor * b->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->tensor_product_diagonal(conj, (*a)[ia], (*b)[ib], (*c)[ic],
                                             scale * (FP)factor);
            else
                GMatrixFunctions<FL>::tensor_product_diagonal(
                    conj, (*a)[ia], (*b)[ib], (*c)[ic], scale * (FP)factor);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    virtual void three_tensor_product_diagonal(
        uint8_t conj, const shared_ptr<SparseMatrix<S, FL>> &a,
        const shared_ptr<SparseMatrix<S, FL>> &b,
        const shared_ptr<SparseMatrix<S, FL>> &c, uint8_t dconj,
        const shared_ptr<SparseMatrix<S, FL>> &da,
        const shared_ptr<SparseMatrix<S, FL>> &db, bool dleft, S opdq,
        FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal &&
               da->get_type() == SparseMatrixTypes::Normal &&
               db->get_type() == SparseMatrixTypes::Normal);
        scale = scale * ((conj & 1) ? xconj<FL>(a->factor) : a->factor) *
                ((conj & 2) ? xconj<FL>(b->factor) : b->factor) *
                (((dconj & 1) ^ (dleft ? (conj & 1) : ((conj & 2) >> 1)))
                     ? xconj<FL>(da->factor)
                     : da->factor) *
                ((((dconj & 2) >> 1) ^ (dleft ? (conj & 1) : ((conj & 2) >> 1)))
                     ? xconj<FL>(db->factor)
                     : db->factor);
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        const shared_ptr<SparseMatrix<S, FL>> dc = (dleft ? a : b);
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        S dadq = da->info->delta_quantum, dbdq = db->info->delta_quantum,
          dcdq = dc->info->delta_quantum;
        S dabdq = dcdq.combine((dconj & 1) ? -dadq : dadq,
                               (dconj & 2) ? dbdq : -dbdq);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>
            cinfo = c->info->cinfo,
            dinfo = dc->info->cinfo;
        assert(cinfo != nullptr && dinfo != nullptr);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        int idk =
            (int)(lower_bound(dinfo->quanta + dinfo->n[dconj],
                              dinfo->quanta + dinfo->n[dconj + 1], dabdq) -
                  dinfo->quanta);
        assert(idk < dinfo->n[dconj + 1]);
        int idxa = dinfo->idx[idk];
        int idxb = idk == dinfo->n[4] - 1 ? dinfo->nc : dinfo->idx[idk + 1];
        for (int idp = 0; idp < idxb; idp++) {
            bool found = false;
            for (int il = ixa; il < ixb; il++) {
                int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
                double factor = cinfo->factor[il];
                int idc = dleft ? ia : ib;
                int idl =
                    (int)(lower_bound(dinfo->ic + idxa, dinfo->ic + idxb, idc) -
                          dinfo->ic + idp);
                for (; idl < idxb && dinfo->ic[idl] == idc; idl++) {
                    found = true;
                    int ida = dinfo->ia[idl], idb = dinfo->ib[idl];
                    uint32_t stride = dinfo->stride[idl];
                    double dfactor = dinfo->factor[idl];
                    if (seq->mode != SeqTypes::None) {
                        seq->three_tensor_product_diagonal(
                            conj, (*a)[ia], (*b)[ib], (*c)[ic], (*da)[ida],
                            dconj & 1, (*db)[idb], (dconj & 2) >> 1, dleft,
                            scale * (FP)(factor * dfactor), stride);
                        if (seq->mode == SeqTypes::Simple)
                            break;
                    } else
                        GMatrixFunctions<FL>::three_tensor_product_diagonal(
                            conj, (*a)[ia], (*b)[ib], (*c)[ic], (*da)[ida],
                            dconj & 1, (*db)[idb], (dconj & 2) >> 1, dleft,
                            scale * (FP)(factor * dfactor), stride);
                }
            }
            if (seq->mode == SeqTypes::Simple)
                seq->simple_perform();
            if (!found || seq->mode != SeqTypes::Simple)
                break;
        }
    }
    template <typename T, typename X>
    static void simple_sort(vector<T> &arr, X extract) {
        if (arr.size() == 0)
            return;
        sort(arr.begin(), arr.end(), [&extract](const T &x, const T &y) {
            return extract(x) < extract(y);
        });
        vector<T> sorted = arr;
        vector<int> len(1, 1);
        vector<size_t> start(1, 0);
        auto prev = extract(arr[0]);
        for (size_t k = 1; k < arr.size(); k++) {
            auto cur = extract(arr[k]);
            if (cur == prev)
                len.back()++;
            else
                len.push_back(1), prev = cur, start.push_back(k);
        }
        for (size_t j = 0; j < arr.size();)
            for (size_t k = 0; k < len.size(); k++)
                if (len[k] != 0)
                    sorted[j++] = arr[start[k]], start[k]++, len[k]--;
        arr = sorted;
    }
    // b = < v | a | c >
    virtual void
    tensor_left_partial_expectation(uint8_t conj,
                                    const shared_ptr<SparseMatrix<S, FL>> &a,
                                    const shared_ptr<SparseMatrix<S, FL>> &b,
                                    const shared_ptr<SparseMatrix<S, FL>> &c,
                                    const shared_ptr<SparseMatrix<S, FL>> &v,
                                    S opdq, FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal &&
               v->get_type() == SparseMatrixTypes::Normal);
        scale = scale * ((conj & 1) ? xconj<FL>(a->factor) : a->factor) *
                xconj<FL>(v->factor) * c->factor;
        if (conj & 2)
            scale = xconj<FL>(scale);
        assert(b->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        vector<pair<array<int, 4>, double>> abcv(ixb - ixa);
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = (int)cinfo->stride[il];
            double factor = cinfo->factor[il];
            abcv[il - ixa] = make_pair(array<int, 4>{ia, ib, ic, iv}, factor);
        }
        simple_sort(abcv, [](const pair<array<int, 4>, double> &x) {
            return x.first[1];
        });
        for (int il = 0; il < (int)abcv.size(); il++) {
            int ia = abcv[il].first[0], ib = abcv[il].first[1],
                ic = abcv[il].first[2], iv = abcv[il].first[3];
            if (seq->mode == SeqTypes::Simple && il != 0 &&
                ib <= abcv[il - 1].first[1])
                seq->simple_perform();
            double factor = abcv[il].second;
            if (seq->mode != SeqTypes::None)
                seq->left_partial_rotate((*a)[ia], conj & 1, (*b)[ib],
                                         (conj & 2) >> 1, (*v)[iv], (*c)[ic],
                                         scale * (FP)factor);
            else
                seq->cumulative_nflop +=
                    GMatrixFunctions<FL>::left_partial_rotate(
                        (*a)[ia], conj & 1, (*b)[ib], (conj & 2) >> 1, (*v)[iv],
                        (*c)[ic], scale * (FP)factor);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    // a = < v | b | c >
    virtual void
    tensor_right_partial_expectation(uint8_t conj,
                                     const shared_ptr<SparseMatrix<S, FL>> &a,
                                     const shared_ptr<SparseMatrix<S, FL>> &b,
                                     const shared_ptr<SparseMatrix<S, FL>> &c,
                                     const shared_ptr<SparseMatrix<S, FL>> &v,
                                     S opdq, FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal &&
               v->get_type() == SparseMatrixTypes::Normal);
        scale = scale * ((conj & 2) ? xconj<FL>(b->factor) : b->factor) *
                xconj<FL>(v->factor) * c->factor;
        if (conj & 1)
            scale = xconj<FL>(scale);
        assert(a->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        vector<pair<array<int, 4>, double>> abcv(ixb - ixa);
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = (int)cinfo->stride[il];
            double factor = cinfo->factor[il];
            abcv[il - ixa] = make_pair(array<int, 4>{ia, ib, ic, iv}, factor);
        }
        simple_sort(abcv, [](const pair<array<int, 4>, double> &x) {
            return x.first[0];
        });
        for (int il = 0; il < (int)abcv.size(); il++) {
            int ia = abcv[il].first[0], ib = abcv[il].first[1],
                ic = abcv[il].first[2], iv = abcv[il].first[3];
            if (seq->mode == SeqTypes::Simple && il != 0 &&
                ia <= abcv[il - 1].first[0])
                seq->simple_perform();
            double factor = abcv[il].second;
            if (seq->mode != SeqTypes::None)
                seq->right_partial_rotate((*b)[ib], (conj & 2) >> 1, (*a)[ia],
                                          conj & 1, (*v)[iv], (*c)[ic],
                                          scale * (FP)factor);
            else
                seq->cumulative_nflop +=
                    GMatrixFunctions<FL>::right_partial_rotate(
                        (*b)[ib], (conj & 2) >> 1, (*a)[ia], conj & 1, (*v)[iv],
                        (*c)[ic], scale * (FP)factor);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    // v = (a x b) @ c
    virtual void tensor_product_multiply(
        uint8_t conj, const shared_ptr<SparseMatrix<S, FL>> &a,
        const shared_ptr<SparseMatrix<S, FL>> &b,
        const shared_ptr<SparseMatrix<S, FL>> &c,
        const shared_ptr<SparseMatrix<S, FL>> &v, S opdq, FL scale = 1.0,
        TraceTypes tt = TraceTypes::None) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal &&
               v->get_type() == SparseMatrixTypes::Normal);
        scale = scale * ((conj & 1) ? xconj<FL>(a->factor) : a->factor) *
                ((conj & 2) ? xconj<FL>(b->factor) : b->factor) * c->factor;
        assert(v->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = (int)cinfo->stride[il];
            if (seq->mode == SeqTypes::Simple && il != ixa &&
                iv <= (int)cinfo->stride[il - 1])
                seq->simple_perform();
            double factor = cinfo->factor[il];
            switch (tt) {
            case TraceTypes::None:
                if (seq->mode != SeqTypes::None)
                    seq->rotate((*c)[ic], (*v)[iv], (*a)[ia],
                                (conj & 1) ? 3 : 0, (*b)[ib],
                                (conj & 2) ? 2 : 1, scale * (FP)factor);
                else
                    seq->cumulative_nflop += GMatrixFunctions<FL>::rotate(
                        (*c)[ic], (*v)[iv], (*a)[ia], (conj & 1) ? 3 : 0,
                        (*b)[ib], (conj & 2) ? 2 : 1, scale * (FP)factor);
                break;
            case TraceTypes::Left:
                if (seq->mode != SeqTypes::None)
                    seq->multiply((*c)[ic], false, (*b)[ib], (conj & 2) ? 2 : 1,
                                  (*v)[iv], scale * (FP)factor, 1.0);
                else
                    GMatrixFunctions<FL>::multiply((*c)[ic], false, (*b)[ib],
                                                   (conj & 2) ? 2 : 1, (*v)[iv],
                                                   scale * (FP)factor, 1.0);
                break;
            case TraceTypes::Right:
                if (seq->mode != SeqTypes::None)
                    seq->multiply((*a)[ia], (conj & 1) ? 3 : 0, (*c)[ic], false,
                                  (*v)[iv], scale * (FP)factor, 1.0);
                else
                    GMatrixFunctions<FL>::multiply((*a)[ia], (conj & 1) ? 3 : 0,
                                                   (*c)[ic], false, (*v)[iv],
                                                   scale * (FP)factor, 1.0);
                break;
            default:
                assert(false);
            }
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    virtual void three_tensor_product_multiply(
        uint8_t conj, const shared_ptr<SparseMatrix<S, FL>> &a,
        const shared_ptr<SparseMatrix<S, FL>> &b,
        const shared_ptr<SparseMatrix<S, FL>> &c,
        const shared_ptr<SparseMatrix<S, FL>> &v, uint8_t dconj,
        const shared_ptr<SparseMatrix<S, FL>> &da,
        const shared_ptr<SparseMatrix<S, FL>> &db, bool dleft, S opdq,
        FL scale = 1.0, TraceTypes tt = TraceTypes::None) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal &&
               v->get_type() == SparseMatrixTypes::Normal &&
               da->get_type() == SparseMatrixTypes::Normal &&
               db->get_type() == SparseMatrixTypes::Normal);
        scale = scale * ((conj & 1) ? xconj<FL>(a->factor) : a->factor) *
                ((conj & 2) ? xconj<FL>(b->factor) : b->factor) * c->factor *
                (((dconj & 1) ^ (dleft ? (conj & 1) : ((conj & 2) >> 1)))
                     ? xconj<FL>(da->factor)
                     : da->factor) *
                ((((dconj & 2) >> 1) ^ (dleft ? (conj & 1) : ((conj & 2) >> 1)))
                     ? xconj<FL>(db->factor)
                     : db->factor);
        assert(v->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        const shared_ptr<SparseMatrix<S, FL>> dc = (dleft ? a : b);
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        S dadq = da->info->delta_quantum, dbdq = db->info->delta_quantum,
          dcdq = dc->info->delta_quantum;
        S dabdq = dcdq.combine((dconj & 1) ? -dadq : dadq,
                               (dconj & 2) ? dbdq : -dbdq);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>
            cinfo = c->info->cinfo,
            dinfo = dc->info->cinfo;
        assert(cinfo != nullptr && dinfo != nullptr);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        int idk =
            (int)(lower_bound(dinfo->quanta + dinfo->n[dconj],
                              dinfo->quanta + dinfo->n[dconj + 1], dabdq) -
                  dinfo->quanta);
        assert(idk < dinfo->n[dconj + 1]);
        int idxa = dinfo->idx[idk];
        int idxb = idk == dinfo->n[4] - 1 ? dinfo->nc : dinfo->idx[idk + 1];
        for (int idp = 0; idp < idxb; idp++) {
            bool found = false;
            for (int il = ixa; il < ixb; il++) {
                int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                    iv = (int)cinfo->stride[il];
                if (seq->mode == SeqTypes::Simple && il != ixa &&
                    iv <= (int)cinfo->stride[il - 1])
                    seq->simple_perform();
                double factor = cinfo->factor[il];
                int idc = dleft ? ia : ib;
                int idl =
                    (int)(lower_bound(dinfo->ic + idxa, dinfo->ic + idxb, idc) -
                          dinfo->ic + idp);
                for (; idl < idxb && dinfo->ic[idl] == idc; idl++) {
                    found = true;
                    int ida = dinfo->ia[idl], idb = dinfo->ib[idl];
                    uint32_t stride = dinfo->stride[idl];
                    double dfactor = dinfo->factor[idl];
                    switch (tt) {
                    case TraceTypes::None:
                        if (seq->mode != SeqTypes::None)
                            seq->three_rotate(
                                (*c)[ic], (*v)[iv], (*a)[ia], conj & 1,
                                (*b)[ib], !(conj & 2), (*da)[ida], dconj & 1,
                                (*db)[idb], (dconj & 2) >> 1, dleft,
                                scale * (FP)(factor * dfactor), stride);
                        else
                            seq->cumulative_nflop +=
                                GMatrixFunctions<FL>::three_rotate(
                                    (*c)[ic], (*v)[iv], (*a)[ia], conj & 1,
                                    (*b)[ib], !(conj & 2), (*da)[ida],
                                    dconj & 1, (*db)[idb], (dconj & 2) >> 1,
                                    dleft, scale * (FP)(factor * dfactor),
                                    stride);
                        break;
                    case TraceTypes::Left:
                        if (seq->mode != SeqTypes::None)
                            seq->three_rotate_tr_left(
                                (*c)[ic], (*v)[iv], (*a)[ia], conj & 1,
                                (*b)[ib], !(conj & 2), (*da)[ida], dconj & 1,
                                (*db)[idb], (dconj & 2) >> 1, dleft,
                                scale * (FP)(factor * dfactor), stride);
                        else
                            seq->cumulative_nflop +=
                                GMatrixFunctions<FL>::three_rotate_tr_left(
                                    (*c)[ic], (*v)[iv], (*a)[ia], conj & 1,
                                    (*b)[ib], !(conj & 2), (*da)[ida],
                                    dconj & 1, (*db)[idb], (dconj & 2) >> 1,
                                    dleft, scale * (FP)(factor * dfactor),
                                    stride);
                        break;
                    case TraceTypes::Right:
                        if (seq->mode != SeqTypes::None)
                            seq->three_rotate_tr_right(
                                (*c)[ic], (*v)[iv], (*a)[ia], conj & 1,
                                (*b)[ib], !(conj & 2), (*da)[ida], dconj & 1,
                                (*db)[idb], (dconj & 2) >> 1, dleft,
                                scale * (FP)(factor * dfactor), stride);
                        else
                            seq->cumulative_nflop +=
                                GMatrixFunctions<FL>::three_rotate_tr_right(
                                    (*c)[ic], (*v)[iv], (*a)[ia], conj & 1,
                                    (*b)[ib], !(conj & 2), (*da)[ida],
                                    dconj & 1, (*db)[idb], (dconj & 2) >> 1,
                                    dleft, scale * (FP)(factor * dfactor),
                                    stride);
                        break;
                    default:
                        assert(false);
                    }
                    if (seq->mode == SeqTypes::Simple)
                        break;
                }
            }
            if (seq->mode == SeqTypes::Simple)
                seq->simple_perform();
            if (!found || seq->mode != SeqTypes::Simple)
                break;
        }
    }
    virtual void tensor_product(uint8_t conj,
                                const shared_ptr<SparseMatrix<S, FL>> &a,
                                const shared_ptr<SparseMatrix<S, FL>> &b,
                                const shared_ptr<SparseMatrix<S, FL>> &c,
                                FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal);
        scale = scale * a->factor * b->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum,
          cdq = c->info->delta_quantum;
        assert(c->info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c->info->cinfo;
        S abdq = cdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = (int)(lower_bound(cinfo->quanta + cinfo->n[conj],
                                   cinfo->quanta + cinfo->n[conj + 1], abdq) -
                       cinfo->quanta);
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            uint32_t stride = cinfo->stride[il];
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None && seq->mode != SeqTypes::Tasked)
                seq->tensor_product((*a)[ia], conj & 1, (*b)[ib],
                                    (conj & 2) >> 1, (*c)[ic],
                                    scale * (FP)factor, stride);
            else
                GMatrixFunctions<FL>::tensor_product(
                    (*a)[ia], conj & 1, (*b)[ib], (conj & 2) >> 1, (*c)[ic],
                    scale * (FP)factor, stride);
        }
        if (seq->mode & SeqTypes::Simple)
            seq->simple_perform();
    }
    // c = a * b * scale
    void product(uint8_t conj, const shared_ptr<SparseMatrix<S, FL>> &a,
                 const shared_ptr<SparseMatrix<S, FL>> &b,
                 const shared_ptr<SparseMatrix<S, FL>> &c,
                 FL scale = 1.0) const {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal &&
               c->get_type() == SparseMatrixTypes::Normal);
        scale = scale * a->factor * b->factor;
        assert(c->factor == (FP)1.0);
        if (abs(scale) < TINY)
            return;
        bool cja = conj & 1, cjb = (conj & 2) >> 1;
        S adq = a->info->delta_quantum, bdq = b->info->delta_quantum,
          cdq = c->info->delta_quantum;
        S sadq = cja ? -a->info->delta_quantum : a->info->delta_quantum;
        S sbdq = cjb ? -b->info->delta_quantum : b->info->delta_quantum;
        for (int ic = 0; ic < c->info->n; ic++) {
            S cq = c->info->quanta[ic].get_bra(c->info->delta_quantum);
            S cqprime = c->info->quanta[ic].get_ket();
            S aps = cq - sadq;
            for (int k = 0; k < aps.count(); k++) {
                S aqprime = aps[k];
                S al = cja ? (-sadq).combine(aps[k], cq)
                           : sadq.combine(cq, aps[k]);
                int ia = a->info->find_state(al);
                if (ia != -1) {
                    S bl = cjb ? (-sbdq).combine(cqprime, aqprime)
                               : sbdq.combine(aqprime, cqprime);
                    if (bl != S(S::invalid)) {
                        int ib = b->info->find_state(bl);
                        if (ib != -1) {
                            double factor =
                                cg->racah(cqprime, bdq, cq, adq, aqprime, cdq);
                            factor *= sqrt(cdq.multiplicity() *
                                           aqprime.multiplicity()) *
                                      cg->phase(adq, bdq, cdq);
                            if (cja)
                                factor *= cg->transpose_cg(-sadq, cq, aqprime);
                            if (cjb)
                                factor *=
                                    cg->transpose_cg(-sbdq, aqprime, cqprime);
                            GMatrixFunctions<FL>::multiply(
                                (*a)[ia], cja, (*b)[ib], cjb, (*c)[ic],
                                scale * (FP)factor, 1.0);
                        }
                    }
                }
            }
        }
    }
    // dot product with no complex conj
    virtual FL dot_product(const shared_ptr<SparseMatrix<S, FL>> &a,
                           const shared_ptr<SparseMatrix<S, FL>> &b,
                           FL scale = 1.0) {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal);
        assert(a->total_memory == b->total_memory);
        GMatrix<FL> amat(a->data, (MKL_INT)a->total_memory, 1);
        GMatrix<FL> bmat(b->data, (MKL_INT)b->total_memory, 1);
        seq->cumulative_nflop += a->total_memory;
        return GMatrixFunctions<FL>::dot(amat, bmat) * scale;
    }
    // Product with conj transposed tensor: [a] x [b]^H or [a]^H x [b]
    static void
    trans_product(const shared_ptr<SparseMatrix<S, FL>> &a,
                  const shared_ptr<SparseMatrix<S, FL>> &b, bool trace_right,
                  FL noise = 0.0,
                  NoiseTypes noise_type = NoiseTypes::DensityMatrix) {
        assert(a->get_type() == SparseMatrixTypes::Normal &&
               b->get_type() == SparseMatrixTypes::Normal);
        FL scale = a->factor * a->factor, noise_scale = 0;
        assert(b->factor == (FP)1.0);
        if (abs(scale) < TINY &&
            (noise == (FP)0.0 || (!(noise_type & NoiseTypes::Wavefunction) &&
                                  !(noise_type & NoiseTypes::DensityMatrix))))
            return;
        SparseMatrix<S, FL> tmp;
        if (noise != (FP)0.0 && (noise_type & NoiseTypes::Wavefunction)) {
            tmp.allocate(a->info);
            tmp.randomize(-0.5, 0.5);
            noise_scale = noise / tmp.norm();
            noise_scale *= noise_scale;
        } else if (noise != (FP)0.0 &&
                   (noise_type & NoiseTypes::DensityMatrix)) {
            tmp.allocate(b->info);
            tmp.randomize(0.0, 1.0);
            noise_scale = noise * noise / tmp.norm();
        }
        if (trace_right)
            for (int ia = 0; ia < a->info->n; ia++) {
                S qb = a->info->quanta[ia].get_bra(a->info->delta_quantum);
                int ib = b->info->find_state(qb);
                if (ib == -1)
                    continue;
                GMatrixFunctions<FL>::multiply((*a)[ia], false, (*a)[ia], 3,
                                               (*b)[ib], scale, 1.0);
                if (noise_scale != (FP)0.0 &&
                    (noise_type & NoiseTypes::Wavefunction))
                    GMatrixFunctions<FL>::multiply(tmp[ia], false, tmp[ia], 3,
                                                   (*b)[ib], noise_scale, 1.0);
                else if (noise_scale != (FP)0.0 &&
                         (noise_type & NoiseTypes::DensityMatrix))
                    GMatrixFunctions<FL>::iadd((*b)[ib], tmp[ib], noise_scale);
            }
        else
            for (int ia = 0; ia < a->info->n; ia++) {
                S qb = -a->info->quanta[ia].get_ket();
                int ib = b->info->find_state(qb);
                if (ib == -1)
                    continue;
                GMatrixFunctions<FL>::multiply((*a)[ia], 3, (*a)[ia], false,
                                               (*b)[ib], scale, 1.0);
                if (noise_scale != (FP)0.0 &&
                    (noise_type & NoiseTypes::Wavefunction))
                    GMatrixFunctions<FL>::multiply(tmp[ia], 3, tmp[ia], false,
                                                   (*b)[ib], noise_scale, 1.0);
                else if (noise_scale != (FP)0.0 &&
                         (noise_type & NoiseTypes::DensityMatrix))
                    GMatrixFunctions<FL>::iadd((*b)[ib], tmp[ib], noise_scale);
            }
        if (noise != (FP)0.0)
            tmp.deallocate();
    }
};

} // namespace block2
