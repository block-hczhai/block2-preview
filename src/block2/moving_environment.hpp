
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

#include "determinant.hpp"
#include "mpo.hpp"
#include "mps.hpp"
#include "parallel_mpo.hpp"
#include "parallel_rule.hpp"
#include "partition.hpp"
#include "state_averaged.hpp"
#include "tensor_functions.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

enum FuseTypes : uint8_t {
    NoFuseL = 4,
    NoFuseR = 8,
    FuseL = 1,
    FuseR = 2,
    FuseLR = 3
};

template <typename S, typename = MPS<S>> struct EffectiveHamiltonian;

// Effective Hamiltonian
template <typename S> struct EffectiveHamiltonian<S, MPS<S>> {
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
        right_op_infos;
    // Symbolic expression of effective H
    shared_ptr<DelayedOperatorTensor<S>> op;
    shared_ptr<SparseMatrix<S>> bra, ket, diag, cmat, vmat;
    shared_ptr<TensorFunctions<S>> tf;
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    // Delta quantum of effective H
    S opdq;
    // Whether diagonal element of effective H should be computed
    bool compute_diag;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S>> &op,
        const shared_ptr<SparseMatrix<S>> &bra,
        const shared_ptr<SparseMatrix<S>> &ket,
        const shared_ptr<OpElement<S>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat,
        const shared_ptr<TensorFunctions<S>> &tf, bool compute_diag = true)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(tf), hop_mat(hop_mat),
          compute_diag(compute_diag) {
        // wavefunction
        if (compute_diag) {
            assert(bra == ket);
            diag = make_shared<SparseMatrix<S>>();
            diag->allocate(ket->info);
        }
        // unique sub labels
        S cdq = ket->info->delta_quantum;
        S vdq = bra->info->delta_quantum;
        opdq = hop->q_label;
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S>::get_uniq_sub_labels(op->mat, hop_mat, msl);
        // tensor prodcut diagonal
        if (compute_diag) {
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> diag_info =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            diag_info->initialize_diag(cdq, opdq, msubsl[0], left_op_infos,
                                       right_op_infos, diag->info, tf->opf->cg);
            diag->info->cinfo = diag_info;
            tf->tensor_product_diagonal(op->mat->data[0], op->lops, op->rops,
                                        diag, opdq);
            if (tf->opf->seq->mode == SeqTypes::Auto)
                tf->opf->seq->auto_perform();
            diag_info->deallocate();
        }
        // temp wavefunction
        cmat = make_shared<SparseMatrix<S>>();
        vmat = make_shared<SparseMatrix<S>>();
        *cmat = *ket;
        *vmat = *bra;
        // temp wavefunction info
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> wfn_info =
            make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
        wfn_info->initialize_wfn(cdq, vdq, opdq, msubsl[0], left_op_infos,
                                 right_op_infos, ket->info, bra->info,
                                 tf->opf->cg);
        cmat->info->cinfo = wfn_info;
        // prepare batch gemm
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            cmat->data = vmat->data = (double *)0;
            tf->tensor_product_multiply(op->mat->data[0], op->lops, op->rops,
                                        cmat, vmat, opdq, false);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        }
    }
    shared_ptr<SparseMatrixGroup<S>>
    perturbative_noise(bool trace_right, int iL, int iR, FuseTypes ftype,
                       const shared_ptr<MPSInfo<S>> &mps_info,
                       const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        assert(msl.size() == 1 && msl[0] == opdq);
        vector<pair<uint8_t, S>> psubsl = Partition<S>::get_uniq_sub_labels(
            op->mat, hop_mat, msl, true, trace_right)[0];
        vector<S> perturb_ket_labels;
        S ket_label = ket->info->delta_quantum;
        for (size_t j = 0; j < psubsl.size(); j++) {
            S pks = ket_label + psubsl[j].second;
            for (int k = 0; k < pks.count(); k++)
                perturb_ket_labels.push_back(pks[k]);
        }
        sort(perturb_ket_labels.begin(), perturb_ket_labels.end());
        perturb_ket_labels.resize(distance(
            perturb_ket_labels.begin(),
            unique(perturb_ket_labels.begin(), perturb_ket_labels.end())));
        if (para_rule != nullptr) {
            para_rule->comm->allreduce_sum(perturb_ket_labels);
            sort(perturb_ket_labels.begin(), perturb_ket_labels.end());
            perturb_ket_labels.resize(distance(
                perturb_ket_labels.begin(),
                unique(perturb_ket_labels.begin(), perturb_ket_labels.end())));
        }
        // perturbed wavefunctions infos
        mps_info->load_left_dims(iL);
        mps_info->load_right_dims(iR + 1);
        StateInfo<S> l = *mps_info->left_dims[iL], ml = *mps_info->basis[iL],
                     mr = *mps_info->basis[iR],
                     r = *mps_info->right_dims[iR + 1];
        StateInfo<S> ll = (ftype & FuseTypes::FuseL)
                              ? StateInfo<S>::tensor_product(
                                    l, ml, *mps_info->left_dims_fci[iL + 1])
                              : l;
        StateInfo<S> rr = (ftype & FuseTypes::FuseR)
                              ? StateInfo<S>::tensor_product(
                                    mr, r, *mps_info->right_dims_fci[iR])
                              : r;
        vector<shared_ptr<SparseMatrixInfo<S>>> infos;
        infos.reserve(perturb_ket_labels.size());
        for (size_t j = 0; j < perturb_ket_labels.size(); j++) {
            shared_ptr<SparseMatrixInfo<S>> info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            info->initialize(ll, rr, perturb_ket_labels[j], false, true);
            infos.push_back(info);
        }
        if (ftype & FuseTypes::FuseR)
            rr.deallocate();
        if (ftype & FuseTypes::FuseL)
            ll.deallocate();
        r.deallocate();
        l.deallocate();
        // perturbed wavefunctions
        shared_ptr<SparseMatrixGroup<S>> perturb_ket =
            make_shared<SparseMatrixGroup<S>>(d_alloc);
        perturb_ket->allocate(infos);
        // connection infos
        frame->activate(0);
        vector<vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
            cinfos;
        cinfos.resize(psubsl.size());
        S idq = S(0);
        for (size_t j = 0; j < psubsl.size(); j++) {
            S pks = ket_label + psubsl[j].second;
            cinfos[j].resize(pks.count());
            for (int k = 0; k < pks.count(); k++) {
                cinfos[j][k] =
                    make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
                int ib = lower_bound(perturb_ket_labels.begin(),
                                     perturb_ket_labels.end(), pks[k]) -
                         perturb_ket_labels.begin();
                S opdq = psubsl[j].second;
                vector<pair<uint8_t, S>> subdq = {
                    trace_right
                        ? make_pair(psubsl[j].first, opdq.combine(opdq, -idq))
                        : make_pair((uint8_t)(psubsl[j].first << 1),
                                    opdq.combine(idq, -opdq))};
                cinfos[j][k]->initialize_wfn(
                    ket_label, pks[k], psubsl[j].second, subdq, left_op_infos,
                    right_op_infos, ket->info, infos[ib], tf->opf->cg);
                assert(cinfos[j][k]->n[4] == 1);
            }
        }
        // perform multiplication
        tf->tensor_product_partial_multiply(
            op->mat->data[0], op->lops, op->rops, trace_right, cmat, psubsl,
            cinfos, perturb_ket_labels, perturb_ket);
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            tf->opf->seq->auto_perform();
            if (para_rule != nullptr)
                para_rule->comm->reduce_sum(perturb_ket, para_rule->comm->root);
        }
        for (int j = (int)cinfos.size() - 1; j >= 0; j--)
            for (int k = (int)cinfos[j].size() - 1; k >= 0; k--)
                cinfos[j][k]->deallocate();
        return perturb_ket;
    }
    int get_mpo_bond_dimension() const {
        if (op->mat->data.size() == 0)
            return 0;
        else if (op->mat->data[0]->get_type() == OpTypes::Zero)
            return 0;
        else if (op->mat->data[0]->get_type() == OpTypes::Sum) {
            int r = 0;
            for (auto &opx :
                 dynamic_pointer_cast<OpSum<S>>(op->mat->data[0])->strings) {
                if (opx->get_type() == OpTypes::Prod ||
                    opx->get_type() == OpTypes::Elem)
                    r++;
                else if (opx->get_type() == OpTypes::SumProd)
                    r += (int)dynamic_pointer_cast<OpSumProd<S>>(opx)
                             ->ops.size();
            }
            return r;
        } else if (op->mat->data[0]->get_type() == OpTypes::SumProd)
            return (int)dynamic_pointer_cast<OpSumProd<S>>(op->mat->data[0])
                ->ops.size();
        else
            return 1;
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const MatrixRef &b, const MatrixRef &c, int idx = 0,
                    double factor = 1.0, bool all_reduce = true) {
        assert(b.m * b.n == cmat->total_memory);
        assert(c.m * c.n == vmat->total_memory);
        cmat->data = b.data;
        vmat->data = c.data;
        cmat->factor = factor;
        tf->tensor_product_multiply(op->mat->data[idx], op->lops, op->rops,
                                    cmat, vmat, opdq, all_reduce);
    }
    // Find eigenvalues and eigenvectors of [H_eff]
    // energy, ndav, nflop, tdav
    tuple<double, int, size_t, double>
    eigs(bool iprint = false, double conv_thrd = 5E-6, int max_iter = 5000,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int ndav = 0;
        assert(compute_diag);
        DiagonalMatrix aa(diag->data, diag->total_memory);
        vector<MatrixRef> bs =
            vector<MatrixRef>{MatrixRef(ket->data, ket->total_memory, 1)};
        frame->activate(0);
        Timer t;
        t.get_time();
        vector<double> eners =
            tf->opf->seq->mode == SeqTypes::Auto
                ? MatrixFunctions::davidson(
                      *tf->opf->seq, aa, bs, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter)
                : MatrixFunctions::davidson(
                      *this, aa, bs, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter);
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners[0], ndav, (size_t)nflop, t.get_time());
    }
    // [bra] = [H_eff] x [ket]
    // norm, nflop, tdav
    tuple<double, size_t, double>
    multiply(const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        bra->clear();
        Timer t;
        t.get_time();
        if (tf->opf->seq->mode == SeqTypes::Auto)
            (*tf->opf->seq)(MatrixRef(ket->data, ket->total_memory, 1),
                            MatrixRef(bra->data, bra->total_memory, 1));
        else
            (*this)(MatrixRef(ket->data, ket->total_memory, 1),
                    MatrixRef(bra->data, bra->total_memory, 1));
        double norm =
            MatrixFunctions::norm(MatrixRef(bra->data, bra->total_memory, 1));
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(norm, (size_t)nflop, t.get_time());
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tmult
    tuple<vector<pair<shared_ptr<OpExpr<S>>, double>>, size_t, double>
    expect(const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        Timer t;
        t.get_time();
        MatrixRef ktmp(ket->data, ket->total_memory, 1);
        MatrixRef rtmp(bra->data, bra->total_memory, 1);
        MatrixRef btmp(nullptr, bra->total_memory, 1);
        btmp.allocate();
        assert(tf->opf->seq->mode != SeqTypes::Auto);
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations;
        expectations.reserve(op->mat->data.size());
        vector<double> results;
        vector<size_t> results_idx;
        results.reserve(op->mat->data.size());
        results_idx.reserve(op->mat->data.size());
        for (size_t i = 0; i < op->mat->data.size(); i++) {
            if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->name ==
                OpNames::Zero)
                continue;
            else if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->q_label !=
                     opdq)
                expectations.push_back(make_pair(op->ops[i], 0.0));
            else {
                double r = 0.0;
                if (para_rule == nullptr || !para_rule->number(op->ops[i])) {
                    btmp.clear();
                    (*this)(ktmp, btmp, i, 1.0, true);
                    r = MatrixFunctions::dot(btmp, rtmp);
                } else {
                    if (para_rule->own(op->ops[i])) {
                        btmp.clear();
                        (*this)(ktmp, btmp, i, 1.0, false);
                        r = MatrixFunctions::dot(btmp, rtmp);
                    }
                    results.push_back(r);
                    results_idx.push_back(expectations.size());
                }
                expectations.push_back(make_pair(op->ops[i], r));
            }
        }
        btmp.deallocate();
        if (results.size() != 0) {
            assert(para_rule != nullptr);
            para_rule->comm->allreduce_sum(results.data(), results.size());
            for (size_t i = 0; i < results.size(); i++)
                expectations[results_idx[i]].second = results[i];
        }
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, (size_t)nflop, t.get_time());
    }
    // [ket] = exp( [H_eff] ) | [ket] > (RK4 approximation)
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<MatrixRef>, tuple<double, double, int, size_t, double>>
    rk4_apply(double beta, double const_e, bool eval_energy = false,
              const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        MatrixRef v(ket->data, ket->total_memory, 1);
        vector<MatrixRef> k, r;
        Timer t;
        t.get_time();
        frame->activate(1);
        for (int i = 0; i < 3; i++) {
            r.push_back(MatrixRef(nullptr, ket->total_memory, 1));
            r[i].allocate();
        }
        frame->activate(0);
        for (int i = 0; i < 4; i++) {
            k.push_back(MatrixRef(nullptr, ket->total_memory, 1));
            k[i].allocate(), k[i].clear();
        }
        assert(tf->opf->seq->mode != SeqTypes::Auto);
        const vector<double> ks = vector<double>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<double>> cs = vector<vector<double>>{
            vector<double>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0,
                           -5.0 / 162.0},
            vector<double>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<double>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        // k0 ~ k3
        for (int i = 0; i < 4; i++) {
            if (i == 0)
                (*this)(v, k[i], 0, beta);
            else {
                MatrixFunctions::copy(r[0], v);
                tf->opf->seq->iadd(r[0], k[i - 1], ks[i], 1.0);
                if (tf->opf->seq->mode != SeqTypes::None)
                    tf->opf->seq->simple_perform();
                (*this)(r[0], k[i], 0, beta);
            }
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            MatrixFunctions::copy(r[i], v);
            double factor = exp(beta * (i + 1) / 3 * const_e);
            for (size_t j = 0; j < 4; j++) {
                tf->opf->seq->iadd(r[i], k[j], cs[i][j] * factor, factor);
                if (tf->opf->seq->mode != SeqTypes::None)
                    tf->opf->seq->simple_perform();
            }
        }
        double norm = MatrixFunctions::norm(r[2]);
        double energy = -const_e;
        if (eval_energy) {
            k[0].clear();
            (*this)(r[2], k[0]);
            energy = MatrixFunctions::dot(r[2], k[0]) / (norm * norm);
        }
        for (int i = 3; i >= 0; i--)
            k[i].deallocate();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(r, make_tuple(energy, norm, 4 + eval_energy,
                                       (size_t)nflop, t.get_time()));
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    tuple<double, double, int, size_t, double>
    expo_apply(double beta, double const_e, bool iprint = false,
               const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(compute_diag);
        double anorm =
            MatrixFunctions::norm(MatrixRef(diag->data, diag->total_memory, 1));
        MatrixRef v(ket->data, ket->total_memory, 1);
        Timer t;
        t.get_time();
        int nexpo = tf->opf->seq->mode == SeqTypes::Auto
                        ? MatrixFunctions::expo_apply(
                              *tf->opf->seq, beta, anorm, v, const_e, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm)
                        : MatrixFunctions::expo_apply(
                              *this, beta, anorm, v, const_e, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm);
        double norm = MatrixFunctions::norm(v);
        MatrixRef tmp(nullptr, ket->total_memory, 1);
        tmp.allocate();
        tmp.clear();
        if (tf->opf->seq->mode == SeqTypes::Auto)
            (*tf->opf->seq)(v, tmp);
        else
            (*this)(v, tmp);
        double energy = MatrixFunctions::dot(v, tmp) / (norm * norm);
        tmp.deallocate();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, (size_t)nflop, t.get_time());
    }
    void deallocate() {
        frame->activate(0);
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            tf->opf->seq->deallocate();
            tf->opf->seq->clear();
        }
        cmat->info->cinfo->deallocate();
        if (compute_diag)
            diag->deallocate();
        op->deallocate();
        for (int i = right_op_infos.size() - 1; i >= 0; i--) {
            if (right_op_infos[i].second->cinfo != nullptr)
                right_op_infos[i].second->cinfo->deallocate();
            right_op_infos[i].second->deallocate();
        }
        for (int i = left_op_infos.size() - 1; i >= 0; i--) {
            if (left_op_infos[i].second->cinfo != nullptr)
                left_op_infos[i].second->cinfo->deallocate();
            left_op_infos[i].second->deallocate();
        }
    }
};

// Effective Hamiltonian for MultiMPS
template <typename S> struct EffectiveHamiltonian<S, MultiMPS<S>> {
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
        right_op_infos;
    // Symbolic expression of effective H
    shared_ptr<DelayedOperatorTensor<S>> op;
    shared_ptr<SparseMatrixGroup<S>> diag;
    vector<shared_ptr<SparseMatrixGroup<S>>> bra, ket;
    shared_ptr<SparseMatrixGroup<S>> cmat, vmat;
    shared_ptr<TensorFunctions<S>> tf;
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    // Delta quantum of effective H
    S opdq;
    // Whether diagonal element of effective H should be computed
    bool compute_diag;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S>> &op,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &bra,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &ket,
        const shared_ptr<OpElement<S>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat,
        const shared_ptr<TensorFunctions<S>> &tf, bool compute_diag = true)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(tf), hop_mat(hop_mat),
          compute_diag(compute_diag) {
        // wavefunction
        if (compute_diag) {
            assert(bra == ket);
            diag = make_shared<SparseMatrixGroup<S>>();
            diag->allocate(ket[0]->infos);
        }
        // unique sub labels
        opdq = hop->q_label;
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S>::get_uniq_sub_labels(op->mat, hop_mat, msl);
        // tensor prodcut diagonal
        if (compute_diag) {
            for (int i = 0; i < diag->n; i++) {
                shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>
                    diag_info = make_shared<
                        typename SparseMatrixInfo<S>::ConnectionInfo>();
                diag_info->initialize_diag(
                    ket[0]->infos[i]->delta_quantum, opdq, msubsl[0],
                    left_op_infos, right_op_infos, diag->infos[i], tf->opf->cg);
                diag->infos[i]->cinfo = diag_info;
                shared_ptr<SparseMatrix<S>> xdiag = (*diag)[i];
                tf->tensor_product_diagonal(op->mat->data[0], op->lops,
                                            op->rops, xdiag, opdq);
                if (tf->opf->seq->mode == SeqTypes::Auto)
                    tf->opf->seq->auto_perform();
                diag_info->deallocate();
            }
        }
        // temp wavefunction
        cmat = make_shared<SparseMatrixGroup<S>>();
        vmat = make_shared<SparseMatrixGroup<S>>();
        *cmat = *ket[0];
        *vmat = *bra[0];
        // temp wavefunction info
        for (int i = 0; i < cmat->n; i++) {
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> wfn_info =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            wfn_info->initialize_wfn(
                cmat->infos[i]->delta_quantum, vmat->infos[i]->delta_quantum,
                opdq, msubsl[0], left_op_infos, right_op_infos, cmat->infos[i],
                vmat->infos[i], tf->opf->cg);
            cmat->infos[i]->cinfo = wfn_info;
        }
        // prepare batch gemm
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            cmat->data = vmat->data = (double *)0;
            tf->tensor_product_multi_multiply(
                op->mat->data[0], op->lops, op->rops, cmat, vmat, opdq, false);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        }
    }
    int get_mpo_bond_dimension() const {
        if (op->mat->data.size() == 0)
            return 0;
        else if (op->mat->data[0]->get_type() == OpTypes::Zero)
            return 0;
        else if (op->mat->data[0]->get_type() == OpTypes::Sum) {
            int r = 0;
            for (auto &opx :
                 dynamic_pointer_cast<OpSum<S>>(op->mat->data[0])->strings) {
                if (opx->get_type() == OpTypes::Prod ||
                    opx->get_type() == OpTypes::Elem)
                    r++;
                else if (opx->get_type() == OpTypes::SumProd)
                    r += (int)dynamic_pointer_cast<OpSumProd<S>>(opx)
                             ->ops.size();
            }
            return r;
        } else if (op->mat->data[0]->get_type() == OpTypes::SumProd)
            return (int)dynamic_pointer_cast<OpSumProd<S>>(op->mat->data[0])
                ->ops.size();
        else
            return 1;
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const MatrixRef &b, const MatrixRef &c, int idx = 0,
                    bool all_reduce = true) {
        assert(b.m * b.n == cmat->total_memory);
        assert(c.m * c.n == vmat->total_memory);
        cmat->data = b.data;
        vmat->data = c.data;
        tf->tensor_product_multi_multiply(op->mat->data[idx], op->lops,
                                          op->rops, cmat, vmat, opdq,
                                          all_reduce);
    }
    // Find eigenvalues and eigenvectors of [H_eff]
    // energies, ndav, nflop, tdav
    tuple<vector<double>, int, size_t, double>
    eigs(bool iprint = false, double conv_thrd = 5E-6, int max_iter = 5000,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int ndav = 0;
        assert(compute_diag);
        DiagonalMatrix aa(diag->data, diag->total_memory);
        vector<MatrixRef> bs;
        for (int i = 0; i < min((int)ket.size(), aa.n); i++)
            bs.push_back(MatrixRef(ket[i]->data, ket[i]->total_memory, 1));
        frame->activate(0);
        Timer t;
        t.get_time();
        vector<double> eners =
            tf->opf->seq->mode == SeqTypes::Auto
                ? MatrixFunctions::davidson(
                      *tf->opf->seq, aa, bs, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter)
                : MatrixFunctions::davidson(
                      *this, aa, bs, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter);
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners, ndav, (size_t)nflop, t.get_time());
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tmult
    tuple<vector<pair<shared_ptr<OpExpr<S>>, vector<double>>>, size_t, double>
    expect(const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        Timer t;
        t.get_time();
        MatrixRef ktmp(nullptr, ket[0]->total_memory, 1);
        MatrixRef rtmp(nullptr, bra[0]->total_memory, 1);
        MatrixRef btmp(nullptr, bra[0]->total_memory, 1);
        btmp.allocate();
        assert(tf->opf->seq->mode != SeqTypes::Auto);
        vector<pair<shared_ptr<OpExpr<S>>, vector<double>>> expectations;
        expectations.reserve(op->mat->data.size());
        vector<double> results;
        vector<size_t> results_idx;
        results.reserve(op->mat->data.size() * ket.size());
        results_idx.reserve(op->mat->data.size());
        for (size_t i = 0; i < op->mat->data.size(); i++) {
            vector<double> rr(ket.size(), 0);
            if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->name ==
                OpNames::Zero)
                continue;
            else if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->q_label !=
                     opdq)
                expectations.push_back(make_pair(op->ops[i], rr));
            else {
                if (para_rule == nullptr || !para_rule->number(op->ops[i])) {
                    for (int j = 0; j < (int)ket.size(); j++) {
                        ktmp.data = ket[j]->data;
                        rtmp.data = bra[j]->data;
                        btmp.clear();
                        (*this)(ktmp, btmp, i, true);
                        rr[j] = MatrixFunctions::dot(btmp, rtmp);
                    }
                } else {
                    if (para_rule->own(op->ops[i])) {
                        for (int j = 0; j < (int)ket.size(); j++) {
                            ktmp.data = ket[j]->data;
                            rtmp.data = bra[j]->data;
                            btmp.clear();
                            (*this)(ktmp, btmp, i, false);
                            rr[j] = MatrixFunctions::dot(btmp, rtmp);
                        }
                    }
                    results.insert(results.end(), rr.begin(), rr.end());
                    results_idx.push_back(expectations.size());
                }
                expectations.push_back(make_pair(op->ops[i], rr));
            }
        }
        btmp.deallocate();
        if (results.size() != 0) {
            assert(para_rule != nullptr);
            para_rule->comm->allreduce_sum(results.data(), results.size());
            for (size_t i = 0; i < results.size(); i += ket.size())
                memcpy(expectations[results_idx[i]].second.data(),
                       results.data() + i, sizeof(double) * ket.size());
        }
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, (size_t)nflop, t.get_time());
    }
    void deallocate() {
        frame->activate(0);
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            tf->opf->seq->deallocate();
            tf->opf->seq->clear();
        }
        for (int i = cmat->n - 1; i >= 0; i--)
            cmat->infos[i]->cinfo->deallocate();
        if (compute_diag)
            diag->deallocate();
        op->deallocate();
        for (int i = right_op_infos.size() - 1; i >= 0; i--) {
            if (right_op_infos[i].second->cinfo != nullptr)
                right_op_infos[i].second->cinfo->deallocate();
            right_op_infos[i].second->deallocate();
        }
        for (int i = left_op_infos.size() - 1; i >= 0; i--) {
            if (left_op_infos[i].second->cinfo != nullptr)
                left_op_infos[i].second->cinfo->deallocate();
            left_op_infos[i].second->deallocate();
        }
    }
};

enum struct TruncationTypes : uint8_t {
    Physical = 0,
    Reduced = 1,
    ReducedInversed = 2
};

// A tensor network < bra | mpo | ket >
template <typename S> struct MovingEnvironment {
    int n_sites, center, dot;
    shared_ptr<MPO<S>> mpo;
    shared_ptr<MPS<S>> bra, ket;
    // Represent the environments contracted around different center sites
    vector<shared_ptr<Partition<S>>> envs;
    // Symbol of the whole-block operator the MPO represents
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    // Tag is used to generate filename for disk storage
    string tag;
    // Paralell execution control
    shared_ptr<ParallelRule<S>> para_rule;
    bool iprint = false;
    MovingEnvironment(const shared_ptr<MPO<S>> &mpo,
                      const shared_ptr<MPS<S>> &bra,
                      const shared_ptr<MPS<S>> &ket, const string &tag = "DMRG")
        : n_sites(ket->n_sites), center(ket->center), dot(ket->dot), mpo(mpo),
          bra(bra), ket(ket), tag(tag), para_rule(nullptr) {
        assert(bra->n_sites == ket->n_sites && mpo->n_sites == ket->n_sites);
        assert(bra->center == ket->center && bra->dot == ket->dot);
        hop_mat = make_shared<SymbolicColumnVector<S>>(1);
        (*hop_mat)[0] = mpo->op;
        if (mpo->get_parallel_type() == ParallelTypes::Distributed) {
            para_rule = dynamic_pointer_cast<ParallelMPO<S>>(mpo)->rule;
            para_rule->comm->barrier();
        }
    }
    // Contract and renormalize left block by one site
    // new site = i - 1
    void left_contract_rotate(int i) {
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos_notrunc;
        vector<shared_ptr<Symbolic<S>>> mats = {
            mpo->left_operator_names[i - 1]};
        if (mpo->schemer != nullptr && i - 1 == mpo->schemer->left_trans_site)
            mats.push_back(mpo->schemer->left_new_operator_names);
        vector<S> sl = Partition<S>::get_uniq_labels(mats);
        shared_ptr<Symbolic<S>> exprs =
            envs[i - 1]->left == nullptr
                ? nullptr
                : (mpo->left_operator_exprs.size() != 0
                       ? mpo->left_operator_exprs[i - 1]
                       : envs[i - 1]->left->lmat *
                             envs[i - 1]->middle.front()->lmat);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S>::get_uniq_sub_labels(
                exprs, mpo->left_operator_names[i - 1], sl);
        Partition<S>::init_left_op_infos_notrunc(
            i - 1, bra->info, ket->info, sl, subsl, envs[i - 1]->left_op_infos,
            mpo->site_op_infos[i - 1], left_op_infos_notrunc, mpo->tf->opf->cg);
        frame->activate(0);
        shared_ptr<OperatorTensor<S>> new_left = Partition<S>::build_left(
            {mpo->left_operator_names[i - 1]}, left_op_infos_notrunc,
            mpo->sparse_form[i - 1] == 'S');
        mpo->tf->left_contract(envs[i - 1]->left, envs[i - 1]->middle.front(),
                               new_left,
                               mpo->left_operator_exprs.size() != 0
                                   ? mpo->left_operator_exprs[i - 1]
                                   : nullptr);
        bra->load_tensor(i - 1);
        if (bra != ket)
            ket->load_tensor(i - 1);
        frame->reset(1);
        Partition<S>::init_left_op_infos(i - 1, bra->info, ket->info, sl,
                                         envs[i]->left_op_infos);
        frame->activate(1);
        envs[i]->left = Partition<S>::build_left(mats, envs[i]->left_op_infos);
        mpo->tf->left_rotate(new_left, bra->tensors[i - 1], ket->tensors[i - 1],
                             envs[i]->left);
        if (mpo->schemer != nullptr && i - 1 == mpo->schemer->left_trans_site)
            mpo->tf->numerical_transform(envs[i]->left, mats[1],
                                         mpo->schemer->left_new_operator_exprs);
        frame->activate(0);
        if (bra != ket)
            ket->unload_tensor(i - 1);
        bra->unload_tensor(i - 1);
        new_left->deallocate();
        Partition<S>::deallocate_op_infos_notrunc(left_op_infos_notrunc);
        frame->save_data(1, get_left_partition_filename(i));
    }
    // Contract and renormalize right block by one site
    // new site = i + dot
    void right_contract_rotate(int i) {
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> right_op_infos_notrunc;
        vector<shared_ptr<Symbolic<S>>> mats = {
            mpo->right_operator_names[i + dot]};
        if (mpo->schemer != nullptr &&
            i + dot == mpo->schemer->right_trans_site)
            mats.push_back(mpo->schemer->right_new_operator_names);
        vector<S> sl = Partition<S>::get_uniq_labels(mats);
        shared_ptr<Symbolic<S>> exprs =
            envs[i + 1]->right == nullptr
                ? nullptr
                : (mpo->right_operator_exprs.size() != 0
                       ? mpo->right_operator_exprs[i + dot]
                       : envs[i + 1]->middle.back()->rmat *
                             envs[i + 1]->right->rmat);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S>::get_uniq_sub_labels(
                exprs, mpo->right_operator_names[i + dot], sl);
        Partition<S>::init_right_op_infos_notrunc(
            i + dot, bra->info, ket->info, sl, subsl,
            envs[i + 1]->right_op_infos, mpo->site_op_infos[i + dot],
            right_op_infos_notrunc, mpo->tf->opf->cg);
        frame->activate(0);
        shared_ptr<OperatorTensor<S>> new_right = Partition<S>::build_right(
            {mpo->right_operator_names[i + dot]}, right_op_infos_notrunc,
            mpo->sparse_form[i + dot] == 'S');
        mpo->tf->right_contract(envs[i + 1]->right, envs[i + 1]->middle.back(),
                                new_right,
                                mpo->right_operator_exprs.size() != 0
                                    ? mpo->right_operator_exprs[i + dot]
                                    : nullptr);
        bra->load_tensor(i + dot);
        if (bra != ket)
            ket->load_tensor(i + dot);
        frame->reset(1);
        Partition<S>::init_right_op_infos(i + dot, bra->info, ket->info, sl,
                                          envs[i]->right_op_infos);
        frame->activate(1);
        envs[i]->right =
            Partition<S>::build_right(mats, envs[i]->right_op_infos);
        mpo->tf->right_rotate(new_right, bra->tensors[i + dot],
                              ket->tensors[i + dot], envs[i]->right);
        if (mpo->schemer != nullptr &&
            i + dot == mpo->schemer->right_trans_site)
            mpo->tf->numerical_transform(
                envs[i]->right, mats[1],
                mpo->schemer->right_new_operator_exprs);
        frame->activate(0);
        if (bra != ket)
            ket->unload_tensor(i + dot);
        bra->unload_tensor(i + dot);
        new_right->deallocate();
        Partition<S>::deallocate_op_infos_notrunc(right_op_infos_notrunc);
        frame->save_data(1, get_right_partition_filename(i));
    }
    string get_left_partition_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix_distri << ".PART." << tag
           << ".LEFT." << Parsing::to_string(i);
        return ss.str();
    }
    string get_right_partition_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix_distri << ".PART." << tag
           << ".RIGHT." << Parsing::to_string(i);
        return ss.str();
    }
    // Generate contracted environment blocks for all center sites
    virtual void init_environments(bool iprint = false) {
        this->iprint = iprint;
        envs.clear();
        envs.resize(n_sites);
        for (int i = 0; i < n_sites; i++) {
            envs[i] =
                make_shared<Partition<S>>(nullptr, nullptr, mpo->tensors[i]);
            if (i != n_sites - 1 && dot == 2)
                envs[i]->middle.push_back(mpo->tensors[i + 1]);
        }
        if (bra->info->get_warm_up_type() == WarmUpTypes::None &&
            ket->info->get_warm_up_type() == WarmUpTypes::None) {
            for (int i = 1; i <= center; i++) {
                check_signal_()();
                if (iprint)
                    cout << "init .. L = " << i << endl;
                left_contract_rotate(i);
            }
            for (int i = n_sites - dot - 1; i >= center; i--) {
                check_signal_()();
                if (iprint)
                    cout << "init .. R = " << i << endl;
                right_contract_rotate(i);
            }
        }
        frame->reset(1);
    }
    // Remove old environment for starting a new sweep
    void prepare() {
        if (dot == 2 && envs[0]->middle.size() == 1)
            throw runtime_error("switching from one-site algorithm to two-site "
                                "algorithm is not allowed.");
        // two-site to one-site transition
        if (dot == 1 && envs[0]->middle.size() == 2) {
            if (center == n_sites - 2 &&
                (ket->canonical_form[n_sites - 1] == 'C' ||
                 ket->canonical_form[n_sites - 1] == 'M')) {
                center = n_sites - 1;
                frame->reset(1);
                if (envs[center - 1]->left != nullptr)
                    frame->load_data(1,
                                     get_left_partition_filename(center - 1));
                left_contract_rotate(center);
            }
            for (int i = n_sites - 1; i >= center; i--)
                if (envs[i]->right != nullptr)
                    frame->rename_data(get_right_partition_filename(i),
                                       get_right_partition_filename(i + 1));
            for (int i = n_sites - 1; i >= 0; i--) {
                envs[i]->middle.resize(1);
                if (i > 0) {
                    envs[i]->right_op_infos = envs[i - 1]->right_op_infos;
                    envs[i]->right = envs[i - 1]->right;
                } else if (center == 0) {
                    frame->reset(1);
                    if (envs[center + 1]->right != nullptr)
                        frame->load_data(
                            1, get_right_partition_filename(center + 1));
                    envs[center]->right_op_infos.clear();
                    envs[center]->right = nullptr;
                    right_contract_rotate(center);
                }
            }
        }
        for (int i = n_sites - 1; i > center; i--) {
            envs[i]->left_op_infos.clear();
            envs[i]->left = nullptr;
        }
        for (int i = 0; i < center; i++) {
            envs[i]->right_op_infos.clear();
            envs[i]->right = nullptr;
        }
    }
    // Move the center site by one
    virtual void move_to(int i) {
        string new_data_name = "";
        if (i > center) {
            if (envs[center]->left != nullptr)
                frame->load_data(1, get_left_partition_filename(center));
            left_contract_rotate(++center);
            if (envs[center]->left != nullptr)
                new_data_name = get_left_partition_filename(center);
        } else if (i < center) {
            if (envs[center]->right != nullptr)
                frame->load_data(1, get_right_partition_filename(center));
            right_contract_rotate(--center);
            if (envs[center]->right != nullptr)
                new_data_name = get_right_partition_filename(center);
        }
        bra->center = ket->center = center;
        // dynamic environment generation for warmup sweep
        if (i != n_sites - dot && envs[i]->right == nullptr) {
            frame->reset(1);
            frame->activate(0);
            vector<shared_ptr<MPS<S>>> mpss =
                bra == ket ? vector<shared_ptr<MPS<S>>>{bra}
                           : vector<shared_ptr<MPS<S>>>{bra, ket};
            for (auto &mps : mpss) {
                if (mps->info->get_warm_up_type() == WarmUpTypes::Local) {
                    mps->info->load_mutable_left();
                    shared_ptr<DynamicMPSInfo<S>> mps_info =
                        dynamic_pointer_cast<DynamicMPSInfo<S>>(mps->info);
                    if (mps->tensors[i + 1] == nullptr || dot == 1) {
                        mps_info->set_right_bond_dimension_local(i + dot);
                        mps->load_mutable_left();
                        mps->initialize(mps_info, false, true);
                        for (int j = i; j < n_sites; j++)
                            mps->random_canonicalize_tensor(j);
                    } else {
                        mps_info->set_right_bond_dimension_local(i + dot, true);
                        mps_info->load_right_dims(i + 1);
                        mps->load_mutable_left();
                        mps->load_tensor(i);
                        mps->initialize_right(mps_info, i + 1);
                        for (int j = i + 1; j < n_sites; j++)
                            mps->random_canonicalize_tensor(j);
                    }
                } else if (mps->info->get_warm_up_type() ==
                           WarmUpTypes::Determinant) {
                    shared_ptr<DeterminantMPSInfo<S>> mps_info =
                        dynamic_pointer_cast<DeterminantMPSInfo<S>>(mps->info);
                    bool ctrd_two_dot =
                        mps->tensors[i + 1] == nullptr || dot == 1;
                    StateInfo<S> st = mps_info->get_complementary_right_dims(
                        i + dot, i - 1, !ctrd_two_dot);
                    vector<vector<vector<uint8_t>>> dets =
                        mps_info->get_determinants(st, i + dot,
                                                   mps->info->n_sites);
                    mps->info->load_mutable_left();
                    if (ctrd_two_dot) {
                        mps_info->set_right_bond_dimension(i + dot, dets);
                        mps->load_mutable_left();
                        mps->initialize(mps_info, false, true);
                        for (int j = i; j < n_sites; j++)
                            mps->random_canonicalize_tensor(j);
                    } else {
                        mps_info->set_right_bond_dimension(i + dot, dets);
                        mps_info->load_right_dims(i + 1);
                        mps->load_mutable_left();
                        mps->load_tensor(i);
                        mps->initialize_right(mps_info, i + 1);
                        for (int j = i + 1; j < n_sites; j++)
                            mps->random_canonicalize_tensor(j);
                    }
                }
                mps->save_mutable();
                mps->deallocate();
                mps->info->save_mutable();
                mps->info->deallocate_mutable();
            }
            for (int j = n_sites - dot - 1; j >= i; j--) {
                check_signal_()();
                if (iprint)
                    cout << "warm up init .. R = " << j << endl;
                right_contract_rotate(j);
            }
            for (int j = n_sites - dot - 1; j > i; j--) {
                envs[j]->right_op_infos.clear();
                envs[j]->right = nullptr;
            }
            frame->reset(1);
            if (new_data_name != "")
                frame->load_data(1, new_data_name);
        }
        if (i != 0 && envs[i]->left == nullptr) {
            frame->reset(1);
            frame->activate(0);
            vector<shared_ptr<MPS<S>>> mpss =
                bra == ket ? vector<shared_ptr<MPS<S>>>{bra}
                           : vector<shared_ptr<MPS<S>>>{bra, ket};
            for (auto &mps : mpss) {
                if (mps->info->get_warm_up_type() == WarmUpTypes::Local) {
                    shared_ptr<DynamicMPSInfo<S>> mps_info =
                        dynamic_pointer_cast<DynamicMPSInfo<S>>(mps->info);
                    if (mps->tensors[i + 1] != nullptr && dot == 2) {
                        mps_info->set_left_bond_dimension_local(i - 1, true);
                        mps_info->load_left_dims(i);
                    } else
                        mps_info->set_left_bond_dimension_local(i - 1);
                    mps->info->load_mutable_right();
                    if (mps->tensors[i + 1] == nullptr || dot == 1) {
                        mps->initialize(mps_info, true, false);
                    } else {
                        mps->initialize_left(mps_info, i);
                        mps->load_tensor(i + 1);
                    }
                    mps->load_mutable_right();
                    for (int j = 0; j <= i; j++)
                        mps->random_canonicalize_tensor(j);
                } else if (mps->info->get_warm_up_type() ==
                           WarmUpTypes::Determinant) {
                    shared_ptr<DeterminantMPSInfo<S>> mps_info =
                        dynamic_pointer_cast<DeterminantMPSInfo<S>>(mps->info);
                    bool ctrd_two_dot =
                        mps->tensors[i + 1] == nullptr || dot == 1;
                    StateInfo<S> st = mps_info->get_complementary_left_dims(
                        i - 1, i + dot, !ctrd_two_dot);
                    vector<vector<vector<uint8_t>>> dets =
                        mps_info->get_determinants(st, 0, i - 1);
                    if (ctrd_two_dot) {
                        mps_info->set_left_bond_dimension(i - 1, dets);
                        mps->info->load_mutable_left();
                        mps->load_mutable_right();
                        mps->initialize(mps_info, true, false);
                    } else {
                        mps_info->set_right_bond_dimension(i - 1, dets);
                        mps_info->load_left_dims(i);
                        mps->load_tensor(i + 1);
                        mps->initialize_left(mps_info, i);
                    }
                    mps->load_mutable_right();
                    for (int j = 0; j <= i; j++)
                        mps->random_canonicalize_tensor(j);
                }
                mps->save_mutable();
                mps->deallocate();
                mps->info->save_mutable();
                mps->info->deallocate_mutable();
            }
            for (int j = 1; j <= i; j++) {
                check_signal_()();
                if (iprint)
                    cout << "warm up init .. L = " << j << endl;
                left_contract_rotate(j);
            }
            for (int j = 1; j < i; j++) {
                envs[j]->left_op_infos.clear();
                envs[j]->left = nullptr;
            }
            frame->reset(1);
            if (new_data_name != "")
                frame->load_data(1, new_data_name);
        }
    }
    // Contract left block for constructing effective Hamiltonian
    // site iL is the new site
    void left_contract(
        int iL, vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        shared_ptr<OperatorTensor<S>> &new_left) {
        // left contract infos
        vector<shared_ptr<Symbolic<S>>> lmats = {mpo->left_operator_names[iL]};
        if (mpo->schemer != nullptr && iL == mpo->schemer->left_trans_site &&
            mpo->schemer->right_trans_site - mpo->schemer->left_trans_site <= 1)
            lmats.push_back(mpo->schemer->left_new_operator_names);
        vector<S> lsl = Partition<S>::get_uniq_labels(lmats);
        shared_ptr<Symbolic<S>> lexprs =
            envs[iL]->left == nullptr
                ? nullptr
                : (mpo->left_operator_exprs.size() != 0
                       ? mpo->left_operator_exprs[iL]
                       : envs[iL]->left->lmat * envs[iL]->middle.front()->lmat);
        vector<vector<pair<uint8_t, S>>> lsubsl =
            Partition<S>::get_uniq_sub_labels(
                lexprs, mpo->left_operator_names[iL], lsl);
        if (envs[iL]->left != nullptr)
            frame->load_data(1, get_left_partition_filename(iL));
        Partition<S>::init_left_op_infos_notrunc(
            iL, bra->info, ket->info, lsl, lsubsl, envs[iL]->left_op_infos,
            mpo->site_op_infos[iL], left_op_infos, mpo->tf->opf->cg);
        // left contract
        frame->activate(0);
        new_left = Partition<S>::build_left(lmats, left_op_infos,
                                            mpo->sparse_form[iL] == 'S');
        mpo->tf->left_contract(
            envs[iL]->left, envs[iL]->middle.front(), new_left,
            mpo->left_operator_exprs.size() != 0 ? mpo->left_operator_exprs[iL]
                                                 : nullptr);
        if (mpo->schemer != nullptr && iL == mpo->schemer->left_trans_site &&
            mpo->schemer->right_trans_site - mpo->schemer->left_trans_site <= 1)
            mpo->tf->numerical_transform(new_left, lmats[1],
                                         mpo->schemer->left_new_operator_exprs);
    }
    // Contract right block for constructing effective Hamiltonian
    // site iR is the new site
    void right_contract(
        int iR,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        shared_ptr<OperatorTensor<S>> &new_right) {
        // right contract infos
        vector<shared_ptr<Symbolic<S>>> rmats = {mpo->right_operator_names[iR]};
        vector<S> rsl = Partition<S>::get_uniq_labels(rmats);
        shared_ptr<Symbolic<S>> rexprs =
            envs[iR - dot + 1]->right == nullptr
                ? nullptr
                : (mpo->right_operator_exprs.size() != 0
                       ? mpo->right_operator_exprs[iR]
                       : envs[iR - dot + 1]->middle.back()->rmat *
                             envs[iR - dot + 1]->right->rmat);
        vector<vector<pair<uint8_t, S>>> rsubsl =
            Partition<S>::get_uniq_sub_labels(
                rexprs, mpo->right_operator_names[iR], rsl);
        if (envs[iR - dot + 1]->right != nullptr)
            frame->load_data(1, get_right_partition_filename(iR - dot + 1));
        Partition<S>::init_right_op_infos_notrunc(
            iR, bra->info, ket->info, rsl, rsubsl,
            envs[iR - dot + 1]->right_op_infos, mpo->site_op_infos[iR],
            right_op_infos, mpo->tf->opf->cg);
        // right contract
        frame->activate(0);
        new_right = Partition<S>::build_right(rmats, right_op_infos,
                                              mpo->sparse_form[iR] == 'S');
        mpo->tf->right_contract(envs[iR - dot + 1]->right,
                                envs[iR - dot + 1]->middle.back(), new_right,
                                mpo->right_operator_exprs.size() != 0
                                    ? mpo->right_operator_exprs[iR]
                                    : nullptr);
    }
    // Copy left-most left block for constructing effective Hamiltonian
    // block to the left of site iL is copied
    void
    left_copy(int iL,
              vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
              shared_ptr<OperatorTensor<S>> &new_left) {
        assert(envs[iL]->left != nullptr);
        frame->load_data(1, get_left_partition_filename(iL));
        frame->activate(0);
        Partition<S>::copy_op_infos(envs[iL]->left_op_infos, left_op_infos);
        new_left = envs[iL]->left->deep_copy();
        for (auto &p : new_left->ops)
            p.second->info = Partition<S>::find_op_info(
                left_op_infos, p.second->info->delta_quantum);
    }
    // Copy right-most right block for constructing effective Hamiltonian
    // block to the right of site iR is copied
    void
    right_copy(int iR,
               vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
               shared_ptr<OperatorTensor<S>> &new_right) {
        assert(envs[iR - dot + 1]->right != nullptr);
        frame->load_data(1, get_right_partition_filename(iR - dot + 1));
        frame->activate(0);
        Partition<S>::copy_op_infos(envs[iR - dot + 1]->right_op_infos,
                                    right_op_infos);
        new_right = envs[iR - dot + 1]->right->deep_copy();
        for (auto &p : new_right->ops)
            p.second->info = Partition<S>::find_op_info(
                right_op_infos, p.second->info->delta_quantum);
    }
    // Generate effective hamiltonian at current center site
    shared_ptr<EffectiveHamiltonian<S>>
    eff_ham(FuseTypes fuse_type, bool compute_diag,
            const shared_ptr<SparseMatrix<S>> &bra_wfn,
            const shared_ptr<SparseMatrix<S>> &ket_wfn) {
        assert(bra->info->get_multi_type() == MultiTypes::None);
        assert(ket->info->get_multi_type() == MultiTypes::None);
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
            right_op_infos;
        shared_ptr<OperatorTensor<S>> new_left, new_right;
        int iL = -1, iR = -1, iM = -1;
        if (dot == 2) {
            if (fuse_type == FuseTypes::FuseLR)
                iL = center, iR = center + 1, iM = center;
            else if (fuse_type == FuseTypes::FuseR)
                iL = center, iR = center, iM = center - 1;
            else if (fuse_type == FuseTypes::FuseL)
                iL = center + 1, iR = center + 1, iM = center + 1;
            else
                assert(false);
        } else if (dot == 1) {
            if (fuse_type == FuseTypes::FuseR)
                iL = center, iR = center, iM = center - 1;
            else if (fuse_type == FuseTypes::FuseL)
                iL = center, iR = center, iM = center;
            else if (fuse_type == FuseTypes::NoFuseL)
                iL = center, iR = center - 1, iM = center - 1;
            else if (fuse_type == FuseTypes::NoFuseR)
                iL = center + 1, iR = center, iM = center;
            else
                assert(false);
        } else
            assert(false);
        if (fuse_type & FuseTypes::FuseL)
            left_contract(iL, left_op_infos, new_left);
        else
            left_copy(iL, left_op_infos, new_left);
        if (fuse_type & FuseTypes::FuseR)
            right_contract(iR, right_op_infos, new_right);
        else
            right_copy(iR, right_op_infos, new_right);
        // delayed left-right contract
        shared_ptr<DelayedOperatorTensor<S>> op =
            mpo->middle_operator_exprs.size() != 0
                ? mpo->tf->delayed_contract(new_left, new_right,
                                            mpo->middle_operator_names[iM],
                                            mpo->middle_operator_exprs[iM])
                : mpo->tf->delayed_contract(new_left, new_right, mpo->op);
        frame->activate(0);
        frame->reset(1);
        shared_ptr<SymbolicColumnVector<S>> hops =
            mpo->middle_operator_exprs.size() != 0
                ? dynamic_pointer_cast<SymbolicColumnVector<S>>(
                      mpo->middle_operator_names[iM])
                : hop_mat;
        shared_ptr<EffectiveHamiltonian<S>> efh =
            make_shared<EffectiveHamiltonian<S>>(left_op_infos, right_op_infos,
                                                 op, bra_wfn, ket_wfn, mpo->op,
                                                 hops, mpo->tf, compute_diag);
        return efh;
    }
    // Generate effective hamiltonian at current center site
    // for MultiMPS case
    shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>>
    multi_eff_ham(FuseTypes fuse_type, bool compute_diag) {
        assert(bra->info->get_multi_type() == MultiTypes::Multi);
        assert(ket->info->get_multi_type() == MultiTypes::Multi);
        shared_ptr<MultiMPS<S>> mbra = dynamic_pointer_cast<MultiMPS<S>>(bra);
        shared_ptr<MultiMPS<S>> mket = dynamic_pointer_cast<MultiMPS<S>>(ket);
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
            right_op_infos;
        shared_ptr<OperatorTensor<S>> new_left, new_right;
        int iL = -1, iR = -1, iM = -1;
        if (dot == 2) {
            if (fuse_type == FuseTypes::FuseLR)
                iL = center, iR = center + 1, iM = center;
            else if (fuse_type == FuseTypes::FuseR)
                iL = center, iR = center, iM = center - 1;
            else if (fuse_type == FuseTypes::FuseL)
                iL = center + 1, iR = center + 1, iM = center + 1;
            else
                assert(false);
        } else if (dot == 1) {
            if (fuse_type == FuseTypes::FuseR)
                iL = center, iR = center, iM = center - 1;
            else if (fuse_type == FuseTypes::FuseL)
                iL = center, iR = center, iM = center;
            else
                assert(false);
        } else
            assert(false);
        if (fuse_type & FuseTypes::FuseL)
            left_contract(iL, left_op_infos, new_left);
        else
            left_copy(iL, left_op_infos, new_left);
        if (fuse_type & FuseTypes::FuseR)
            right_contract(iR, right_op_infos, new_right);
        else
            right_copy(iR, right_op_infos, new_right);
        // delayed left-right contract
        shared_ptr<DelayedOperatorTensor<S>> op =
            mpo->middle_operator_exprs.size() != 0
                ? mpo->tf->delayed_contract(new_left, new_right,
                                            mpo->middle_operator_names[iM],
                                            mpo->middle_operator_exprs[iM])
                : mpo->tf->delayed_contract(new_left, new_right, mpo->op);
        frame->activate(0);
        frame->reset(1);
        shared_ptr<SymbolicColumnVector<S>> hops =
            mpo->middle_operator_exprs.size() != 0
                ? dynamic_pointer_cast<SymbolicColumnVector<S>>(
                      mpo->middle_operator_names[iM])
                : hop_mat;
        shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>> efh =
            make_shared<EffectiveHamiltonian<S, MultiMPS<S>>>(
                left_op_infos, right_op_infos, op, mbra->wfns, mket->wfns,
                mpo->op, hops, mpo->tf, compute_diag);
        return efh;
    }
    // Absorb wfn matrix into adjacent MPS tensor in one-site algorithm
    static void contract_one_dot(int i, const shared_ptr<SparseMatrix<S>> &wfn,
                                 const shared_ptr<MPS<S>> &mps, bool forward,
                                 bool reduced = false) {
        shared_ptr<SparseMatrix<S>> old_wfn = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrixInfo<S>> old_wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        frame->activate(1);
        mps->load_tensor(i);
        frame->activate(0);
        if (reduced) {
            if (forward)
                old_wfn_info->initialize_contract(wfn->info,
                                                  mps->tensors[i]->info);
            else
                old_wfn_info->initialize_contract(mps->tensors[i]->info,
                                                  wfn->info);
        } else {
            frame->activate(1);
            mps->info->load_left_dims(i);
            mps->info->load_right_dims(i + 1);
            StateInfo<S> l = *mps->info->left_dims[i], m = *mps->info->basis[i],
                         r = *mps->info->right_dims[i + 1];
            StateInfo<S> ll = forward
                                  ? l
                                  : StateInfo<S>::tensor_product(
                                        l, m, *mps->info->left_dims_fci[i + 1]);
            StateInfo<S> rr = !forward
                                  ? r
                                  : StateInfo<S>::tensor_product(
                                        m, r, *mps->info->right_dims_fci[i]);
            frame->activate(0);
            old_wfn_info->initialize(ll, rr, mps->info->target, false, true);
            frame->activate(1);
            if (forward)
                rr.deallocate();
            else
                ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame->activate(0);
        }
        frame->activate(0);
        old_wfn->allocate(old_wfn_info);
        if (forward)
            old_wfn->contract(wfn, mps->tensors[i]);
        else
            old_wfn->contract(mps->tensors[i], wfn);
        frame->activate(1);
        mps->unload_tensor(i);
        frame->activate(0);
        mps->tensors[i] = old_wfn;
    }
    // Contract two adjcent MPS tensors to one two-site MPS tensor
    static void contract_two_dot(int i, const shared_ptr<MPS<S>> &mps,
                                 bool reduced = false) {
        shared_ptr<SparseMatrix<S>> old_wfn = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrixInfo<S>> old_wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        frame->activate(1);
        mps->load_tensor(i);
        mps->load_tensor(i + 1);
        frame->activate(0);
        if (reduced)
            old_wfn_info->initialize_contract(mps->tensors[i]->info,
                                              mps->tensors[i + 1]->info);
        else {
            frame->activate(1);
            mps->info->load_left_dims(i);
            mps->info->load_right_dims(i + 2);
            StateInfo<S> l = *mps->info->left_dims[i],
                         ml = *mps->info->basis[i],
                         mr = *mps->info->basis[i + 1],
                         r = *mps->info->right_dims[i + 2];
            StateInfo<S> ll = StateInfo<S>::tensor_product(
                l, ml, *mps->info->left_dims_fci[i + 1]);
            StateInfo<S> rr = StateInfo<S>::tensor_product(
                mr, r, *mps->info->right_dims_fci[i + 1]);
            frame->activate(0);
            old_wfn_info->initialize(ll, rr, mps->info->target, false, true);
            frame->activate(1);
            rr.deallocate();
            ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame->activate(0);
        }
        frame->activate(0);
        old_wfn->allocate(old_wfn_info);
        old_wfn->contract(mps->tensors[i], mps->tensors[i + 1]);
        frame->activate(1);
        mps->unload_tensor(i + 1);
        mps->unload_tensor(i);
        frame->activate(0);
        mps->tensors[i] = old_wfn;
        mps->tensors[i + 1] = nullptr;
    }
    // Absorb wfn matrices into adjacent MultiMPS tensor in one-site algorithm
    static void
    contract_multi_one_dot(int i,
                           const vector<shared_ptr<SparseMatrixGroup<S>>> &wfns,
                           const shared_ptr<MultiMPS<S>> &mps, bool forward,
                           bool reduced = false) {
        vector<shared_ptr<SparseMatrixGroup<S>>> old_wfns;
        vector<shared_ptr<SparseMatrixInfo<S>>> old_wfn_infos;
        frame->activate(1);
        mps->load_tensor(i);
        assert(mps->tensors[i] != nullptr);
        frame->activate(0);
        old_wfns.resize(wfns.size());
        old_wfn_infos.resize(wfns[0]->n);
        if (reduced) {
            for (int j = 0; j < wfns[0]->n; j++) {
                old_wfn_infos[j] = make_shared<SparseMatrixInfo<S>>();
                if (forward)
                    old_wfn_infos[j]->initialize_contract(
                        wfns[0]->infos[j], mps->tensors[i]->info);
                else
                    old_wfn_infos[j]->initialize_contract(mps->tensors[i]->info,
                                                          wfns[0]->infos[j]);
            }
        } else {
            frame->activate(1);
            mps->info->load_left_dims(i);
            mps->info->load_right_dims(i + 1);
            StateInfo<S> l = *mps->info->left_dims[i], m = *mps->info->basis[i],
                         r = *mps->info->right_dims[i + 1];
            StateInfo<S> ll = forward
                                  ? l
                                  : StateInfo<S>::tensor_product(
                                        l, m, *mps->info->left_dims_fci[i + 1]);
            StateInfo<S> rr = !forward
                                  ? r
                                  : StateInfo<S>::tensor_product(
                                        m, r, *mps->info->right_dims_fci[i]);
            frame->activate(0);
            for (int j = 0; j < wfns[0]->n; j++) {
                old_wfn_infos[j] = make_shared<SparseMatrixInfo<S>>();
                old_wfn_infos[j]->initialize(
                    ll, rr, wfns[0]->infos[j]->delta_quantum, false, true);
            }
            frame->activate(1);
            if (forward)
                rr.deallocate();
            else
                ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame->activate(0);
        }
        frame->activate(0);
        for (int k = 0; k < mps->nroots; k++) {
            old_wfns[k] = make_shared<SparseMatrixGroup<S>>();
            old_wfns[k]->allocate(old_wfn_infos);
            if (forward)
                for (int j = 0; j < old_wfns[k]->n; j++)
                    (*old_wfns[k])[j]->contract((*wfns[k])[j], mps->tensors[i]);
            else
                for (int j = 0; j < old_wfns[k]->n; j++)
                    (*old_wfns[k])[j]->contract(mps->tensors[i], (*wfns[k])[j]);
        }
        frame->activate(1);
        mps->unload_tensor(i);
        frame->activate(0);
        mps->tensors[i] = nullptr;
        mps->wfns = old_wfns;
    }
    // Contract two adjcent MultiMPS tensors to one two-site MultiMPS tensor
    static void contract_multi_two_dot(int i,
                                       const shared_ptr<MultiMPS<S>> &mps,
                                       bool reduced = false) {
        vector<shared_ptr<SparseMatrixGroup<S>>> old_wfns;
        vector<shared_ptr<SparseMatrixInfo<S>>> old_wfn_infos;
        frame->activate(1);
        assert(mps->tensors[i] == nullptr || mps->tensors[i + 1] == nullptr);
        bool left_wfn = mps->tensors[i] == nullptr;
        if (left_wfn) {
            mps->load_wavefunction(i);
            mps->load_tensor(i + 1);
        } else {
            mps->load_tensor(i);
            mps->load_wavefunction(i + 1);
        }
        frame->activate(0);
        old_wfns.resize(mps->nroots);
        old_wfn_infos.resize(mps->wfns[0]->n);
        if (reduced) {
            for (int j = 0; j < mps->wfns[0]->n; j++) {
                old_wfn_infos[j] = make_shared<SparseMatrixInfo<S>>();
                if (left_wfn)
                    old_wfn_infos[j]->initialize_contract(
                        mps->wfns[0]->infos[j], mps->tensors[i + 1]->info);
                else
                    old_wfn_infos[j]->initialize_contract(
                        mps->tensors[i]->info, mps->wfns[0]->infos[j]);
            }
        } else {
            frame->activate(1);
            mps->info->load_left_dims(i);
            mps->info->load_right_dims(i + 2);
            StateInfo<S> l = *mps->info->left_dims[i],
                         ml = *mps->info->basis[i],
                         mr = *mps->info->basis[i + 1],
                         r = *mps->info->right_dims[i + 2];
            StateInfo<S> ll = StateInfo<S>::tensor_product(
                l, ml, *mps->info->left_dims_fci[i + 1]);
            StateInfo<S> rr = StateInfo<S>::tensor_product(
                mr, r, *mps->info->right_dims_fci[i + 1]);
            frame->activate(0);
            for (int j = 0; j < mps->wfns[0]->n; j++) {
                old_wfn_infos[j] = make_shared<SparseMatrixInfo<S>>();
                old_wfn_infos[j]->initialize(
                    ll, rr, mps->wfns[0]->infos[j]->delta_quantum, false, true);
            }
            frame->activate(1);
            rr.deallocate();
            ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame->activate(0);
        }
        frame->activate(0);
        for (int k = 0; k < mps->nroots; k++) {
            old_wfns[k] = make_shared<SparseMatrixGroup<S>>();
            old_wfns[k]->allocate(old_wfn_infos);
            if (left_wfn)
                for (int j = 0; j < old_wfns[k]->n; j++)
                    (*old_wfns[k])[j]->contract((*mps->wfns[k])[j],
                                                mps->tensors[i + 1]);
            else
                for (int j = 0; j < old_wfns[k]->n; j++)
                    (*old_wfns[k])[j]->contract(mps->tensors[i],
                                                (*mps->wfns[k])[j]);
        }
        frame->activate(1);
        if (left_wfn) {
            mps->unload_tensor(i + 1);
            mps->unload_wavefunction(i);
        } else {
            mps->unload_wavefunction(i + 1);
            mps->unload_tensor(i);
        }
        frame->activate(0);
        mps->tensors[i] = mps->tensors[i + 1] = nullptr;
        mps->wfns = old_wfns;
    }
    // Density matrix of a MPS tensor
    static shared_ptr<SparseMatrix<S>>
    density_matrix(S opdq, const shared_ptr<SparseMatrix<S>> &psi,
                   bool trace_right, double noise, NoiseTypes noise_type) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(
            vector<shared_ptr<SparseMatrixInfo<S>>>{psi->info}, opdq,
            trace_right);
        shared_ptr<SparseMatrix<S>> dm = make_shared<SparseMatrix<S>>();
        dm->allocate(dm_info);
        OperatorFunctions<S>::trans_product(psi, dm, trace_right, noise,
                                            noise_type);
        return dm;
    }
    // Direct add noise to wavefunction (before svd)
    static void wavefunction_add_noise(const shared_ptr<SparseMatrix<S>> &psi,
                                       double noise) {
        assert(psi->factor == 1.0);
        if (abs(noise) < TINY && noise == 0.0)
            return;
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
        tmp->allocate(psi->info);
        tmp->randomize(-0.5, 0.5);
        double noise_scale = noise / tmp->norm();
        MatrixFunctions::iadd(MatrixRef(psi->data, psi->total_memory, 1),
                              MatrixRef(tmp->data, tmp->total_memory, 1),
                              noise_scale);
        tmp->deallocate();
    }
    // Density matrix of a MultiMPS tensor
    static shared_ptr<SparseMatrix<S>> density_matrix_with_multi_target(
        S opdq, const vector<shared_ptr<SparseMatrixGroup<S>>> &psi,
        const vector<double> weights, bool trace_right, double noise,
        NoiseTypes noise_type) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(psi[0]->infos, opdq, trace_right);
        shared_ptr<SparseMatrix<S>> dm = make_shared<SparseMatrix<S>>();
        dm->allocate(dm_info);
        assert(weights.size() == psi.size());
        for (size_t i = 0; i < psi.size(); i++)
            for (int j = 0; j < psi[i]->n; j++) {
                shared_ptr<SparseMatrix<S>> wfn = (*psi[i])[j];
                wfn->factor = weights[i];
                OperatorFunctions<S>::trans_product(wfn, dm, trace_right, noise,
                                                    noise_type);
            }
        return dm;
    }
    // Density matrix with perturbed wavefunctions as noise
    static shared_ptr<SparseMatrix<S>> density_matrix_with_perturbative_noise(
        S opdq, const shared_ptr<SparseMatrix<S>> &psi, bool trace_right,
        double noise, const shared_ptr<SparseMatrixGroup<S>> &mats) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(
            vector<shared_ptr<SparseMatrixInfo<S>>>{psi->info}, opdq,
            trace_right);
        shared_ptr<SparseMatrix<S>> dm = make_shared<SparseMatrix<S>>();
        dm->allocate(dm_info);
        for (int i = 1; i < mats->n; i++)
            OperatorFunctions<S>::trans_product((*mats)[i], dm, trace_right,
                                                0.0, NoiseTypes::None);
        double norm = dm->norm();
        dm->iscale(noise / norm);
        OperatorFunctions<S>::trans_product(psi, dm, trace_right, 0.0,
                                            NoiseTypes::None);
        return dm;
    }
    // Density matrix of several MPS tensors summed with weights
    static shared_ptr<SparseMatrix<S>> density_matrix_with_weights(
        S opdq, const shared_ptr<SparseMatrix<S>> &psi, bool trace_right,
        double noise, const vector<MatrixRef> &mats,
        const vector<double> &weights, NoiseTypes noise_type) {
        double *ptr = psi->data;
        assert(psi->factor == 1.0);
        assert(mats.size() == weights.size() - 1);
        psi->factor = sqrt(weights[0]);
        shared_ptr<SparseMatrix<S>> dm =
            density_matrix(opdq, psi, trace_right, noise, noise_type);
        for (size_t i = 1; i < weights.size(); i++) {
            psi->data = mats[i - 1].data;
            psi->factor = sqrt(weights[i]);
            OperatorFunctions<S>::trans_product(psi, dm, trace_right, 0.0);
        }
        psi->data = ptr, psi->factor = 1.0;
        return dm;
    }
    // Diagonalize density matrix and truncate to k eigenvalues
    static double truncate_density_matrix(const shared_ptr<SparseMatrix<S>> &dm,
                                          vector<pair<int, int>> &ss, int k,
                                          double cutoff,
                                          TruncationTypes trunc_type) {
        vector<DiagonalMatrix> eigen_values;
        vector<MatrixRef> eigen_values_reduced;
        int k_total = 0;
        for (int i = 0; i < dm->info->n; i++) {
            DiagonalMatrix w(nullptr, dm->info->n_states_bra[i]);
            w.allocate();
            MatrixFunctions::eigs((*dm)[i], w);
            MatrixRef wr(nullptr, w.n, 1);
            wr.allocate();
            MatrixFunctions::copy(wr, MatrixRef(w.data, w.n, 1));
            if (trunc_type == TruncationTypes::Reduced)
                MatrixFunctions::iscale(
                    wr, 1.0 / dm->info->quanta[i].multiplicity());
            else if (trunc_type == TruncationTypes::ReducedInversed)
                MatrixFunctions::iscale(wr, dm->info->quanta[i].multiplicity());
            eigen_values.push_back(w);
            eigen_values_reduced.push_back(wr);
            k_total += w.n;
        }
        double error = 0.0;
        ss.reserve(k_total);
        for (int i = 0; i < (int)eigen_values.size(); i++)
            for (int j = 0; j < eigen_values[i].n; j++)
                ss.push_back(make_pair(i, j));
        if (k != -1) {
            sort(ss.begin(), ss.end(),
                 [&eigen_values_reduced](const pair<int, int> &a,
                                         const pair<int, int> &b) {
                     return eigen_values_reduced[a.first].data[a.second] >
                            eigen_values_reduced[b.first].data[b.second];
                 });
            for (int i = k; i < k_total; i++) {
                double x = eigen_values[ss[i].first].data[ss[i].second];
                if (x > 0)
                    error += x;
            }
            for (k = min(k, (int)ss.size());
                 k > 1 &&
                 eigen_values_reduced[ss[k - 1].first].data[ss[k - 1].second] <
                     cutoff;
                 k--) {
                double x = eigen_values[ss[k - 1].first].data[ss[k - 1].second];
                if (x > 0)
                    error += x;
            }
            if (k < (int)ss.size())
                ss.resize(k);
            sort(ss.begin(), ss.end(),
                 [](const pair<int, int> &a, const pair<int, int> &b) {
                     return a.first != b.first ? a.first < b.first
                                               : a.second < b.second;
                 });
        }
        for (int i = dm->info->n - 1; i >= 0; i--) {
            eigen_values_reduced[i].deallocate();
            eigen_values[i].deallocate();
        }
        return error;
    }
    // Truncate and keep k singular values
    static double truncate_singular_values(const vector<S> &qs,
                                           const vector<shared_ptr<Tensor>> &s,
                                           vector<pair<int, int>> &ss, int k,
                                           double cutoff,
                                           TruncationTypes trunc_type) {
        vector<shared_ptr<Tensor>> s_reduced;
        cutoff = sqrt(cutoff);
        int k_total = 0;
        for (int i = 0; i < (int)s.size(); i++) {
            shared_ptr<Tensor> wr = make_shared<Tensor>(s[i]->shape);
            MatrixFunctions::copy(wr->ref(), s[i]->ref());
            if (trunc_type == TruncationTypes::Reduced)
                MatrixFunctions::iscale(wr->ref(),
                                        sqrt(1.0 / qs[i].multiplicity()));
            else if (trunc_type == TruncationTypes::ReducedInversed)
                MatrixFunctions::iscale(wr->ref(), sqrt(qs[i].multiplicity()));
            s_reduced.push_back(wr);
            k_total += wr->shape[0];
        }
        double error = 0.0;
        ss.reserve(k_total);
        for (int i = 0; i < (int)s.size(); i++)
            for (int j = 0; j < s[i]->shape[0]; j++)
                ss.push_back(make_pair(i, j));
        if (k != -1) {
            sort(
                ss.begin(), ss.end(),
                [&s_reduced](const pair<int, int> &a, const pair<int, int> &b) {
                    return s_reduced[a.first]->data[a.second] >
                           s_reduced[b.first]->data[b.second];
                });
            for (int i = k; i < k_total; i++) {
                double x = s[ss[i].first]->data[ss[i].second];
                if (x > 0)
                    error += x * x;
            }
            for (k = min(k, (int)ss.size());
                 k > 1 &&
                 s_reduced[ss[k - 1].first]->data[ss[k - 1].second] < cutoff;
                 k--) {
                double x = s[ss[k - 1].first]->data[ss[k - 1].second];
                if (x > 0)
                    error += x * x;
            }
            if (k < (int)ss.size())
                ss.resize(k);
            sort(ss.begin(), ss.end(),
                 [](const pair<int, int> &a, const pair<int, int> &b) {
                     return a.first != b.first ? a.first < b.first
                                               : a.second < b.second;
                 });
        }
        return error;
    }
    // Get rotation matrix info from svd info
    static shared_ptr<SparseMatrixInfo<S>>
    rotation_matrix_info_from_svd(S opdq, const vector<S> &qs,
                                  const vector<shared_ptr<Tensor>> &ts,
                                  bool trace_right, const vector<uint16_t> &ilr,
                                  const vector<uint16_t> &im) {
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>();
        rinfo->is_fermion = false;
        rinfo->is_wavefunction = false;
        rinfo->delta_quantum = opdq;
        int kk = ilr.size();
        rinfo->allocate(kk);
        for (int i = 0; i < kk; i++) {
            rinfo->quanta[i] = qs[ilr[i]];
            rinfo->n_states_bra[i] = trace_right ? ts[ilr[i]]->shape[0] : im[i];
            rinfo->n_states_ket[i] = trace_right ? im[i] : ts[ilr[i]]->shape[1];
        }
        rinfo->n_states_total[0] = 0;
        for (int i = 0; i < kk - 1; i++)
            rinfo->n_states_total[i + 1] =
                rinfo->n_states_total[i] +
                (uint32_t)rinfo->n_states_bra[i] * rinfo->n_states_ket[i];
        return rinfo;
    }
    // Get wavefunction matrix info from svd info
    static shared_ptr<SparseMatrixInfo<S>> wavefunction_info_from_svd(
        const vector<S> &qs, const shared_ptr<SparseMatrixInfo<S>> &wfninfo,
        bool trace_right, const vector<uint16_t> &ilr,
        const vector<uint16_t> &im, vector<vector<uint16_t>> &idx_dm_to_wfn) {
        shared_ptr<SparseMatrixInfo<S>> winfo =
            make_shared<SparseMatrixInfo<S>>();
        winfo->is_fermion = false;
        winfo->is_wavefunction = true;
        winfo->delta_quantum = wfninfo->delta_quantum;
        idx_dm_to_wfn.resize(qs.size());
        if (trace_right)
            for (int i = 0; i < wfninfo->n; i++) {
                S pb = wfninfo->quanta[i].get_bra(wfninfo->delta_quantum);
                int iq = lower_bound(qs.begin(), qs.end(), pb) - qs.begin();
                idx_dm_to_wfn[iq].push_back(i);
            }
        else
            for (int i = 0; i < wfninfo->n; i++) {
                S pk = -wfninfo->quanta[i].get_ket();
                int iq = lower_bound(qs.begin(), qs.end(), pk) - qs.begin();
                idx_dm_to_wfn[iq].push_back(i);
            }
        int kkw = 0, kk = (int)ilr.size();
        for (int i = 0; i < kk; i++)
            kkw += idx_dm_to_wfn[ilr[i]].size();
        winfo->allocate(kkw);
        for (int i = 0, j = 0; i < kk; i++) {
            for (int iw = 0; iw < (int)idx_dm_to_wfn[ilr[i]].size(); iw++) {
                winfo->quanta[j + iw] =
                    wfninfo->quanta[idx_dm_to_wfn[ilr[i]][iw]];
                winfo->n_states_bra[j + iw] =
                    trace_right
                        ? im[i]
                        : wfninfo->n_states_bra[idx_dm_to_wfn[ilr[i]][iw]];
                winfo->n_states_ket[j + iw] =
                    trace_right
                        ? wfninfo->n_states_ket[idx_dm_to_wfn[ilr[i]][iw]]
                        : im[i];
            }
            j += (int)idx_dm_to_wfn[ilr[i]].size();
        }
        winfo->sort_states();
        return winfo;
    }
    // Get rotation matrix info from density matrix info
    static shared_ptr<SparseMatrixInfo<S>>
    rotation_matrix_info_from_density_matrix(
        const shared_ptr<SparseMatrixInfo<S>> &dminfo, bool trace_right,
        const vector<uint16_t> &ilr, const vector<uint16_t> &im) {
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>();
        rinfo->is_fermion = false;
        rinfo->is_wavefunction = false;
        rinfo->delta_quantum = dminfo->delta_quantum;
        int kk = ilr.size();
        rinfo->allocate(kk);
        for (int i = 0; i < kk; i++) {
            rinfo->quanta[i] = dminfo->quanta[ilr[i]];
            rinfo->n_states_bra[i] =
                trace_right ? dminfo->n_states_bra[ilr[i]] : im[i];
            rinfo->n_states_ket[i] =
                trace_right ? im[i] : dminfo->n_states_ket[ilr[i]];
        }
        rinfo->n_states_total[0] = 0;
        for (int i = 0; i < kk - 1; i++)
            rinfo->n_states_total[i + 1] =
                rinfo->n_states_total[i] +
                (uint32_t)rinfo->n_states_bra[i] * rinfo->n_states_ket[i];
        return rinfo;
    }
    // Get wavefunction matrix info from density matrix info
    static shared_ptr<SparseMatrixInfo<S>>
    wavefunction_info_from_density_matrix(
        const shared_ptr<SparseMatrixInfo<S>> &dminfo,
        const shared_ptr<SparseMatrixInfo<S>> &wfninfo, bool trace_right,
        const vector<uint16_t> &ilr, const vector<uint16_t> &im,
        vector<vector<uint16_t>> &idx_dm_to_wfn) {
        shared_ptr<SparseMatrixInfo<S>> winfo =
            make_shared<SparseMatrixInfo<S>>();
        winfo->is_fermion = false;
        winfo->is_wavefunction = true;
        winfo->delta_quantum = wfninfo->delta_quantum;
        idx_dm_to_wfn.resize(dminfo->n);
        if (trace_right)
            for (int i = 0; i < wfninfo->n; i++) {
                S pb = wfninfo->quanta[i].get_bra(wfninfo->delta_quantum);
                idx_dm_to_wfn[dminfo->find_state(pb)].push_back(i);
            }
        else
            for (int i = 0; i < wfninfo->n; i++) {
                S pk = -wfninfo->quanta[i].get_ket();
                idx_dm_to_wfn[dminfo->find_state(pk)].push_back(i);
            }
        int kkw = 0, kk = (int)ilr.size();
        for (int i = 0; i < kk; i++)
            kkw += idx_dm_to_wfn[ilr[i]].size();
        winfo->allocate(kkw);
        for (int i = 0, j = 0; i < kk; i++) {
            for (int iw = 0; iw < (int)idx_dm_to_wfn[ilr[i]].size(); iw++) {
                winfo->quanta[j + iw] =
                    wfninfo->quanta[idx_dm_to_wfn[ilr[i]][iw]];
                winfo->n_states_bra[j + iw] =
                    trace_right
                        ? im[i]
                        : wfninfo->n_states_bra[idx_dm_to_wfn[ilr[i]][iw]];
                winfo->n_states_ket[j + iw] =
                    trace_right
                        ? wfninfo->n_states_ket[idx_dm_to_wfn[ilr[i]][iw]]
                        : im[i];
            }
            j += (int)idx_dm_to_wfn[ilr[i]].size();
        }
        winfo->sort_states();
        return winfo;
    }
    // Split wavefunction to two MPS tensors using svd
    static double split_wavefunction_svd(
        S opdq, const shared_ptr<SparseMatrix<S>> &wfn, int k, bool trace_right,
        bool normalize, shared_ptr<SparseMatrix<S>> &left,
        shared_ptr<SparseMatrix<S>> &right, double cutoff,
        TruncationTypes trunc_type = TruncationTypes::Physical) {
        vector<shared_ptr<Tensor>> l, s, r;
        vector<S> qs;
        if (trace_right)
            wfn->right_svd(qs, l, s, r);
        else
            wfn->left_svd(qs, l, s, r);
        // ss: pair<quantum index in dm, reduced matrix index in dm>
        vector<pair<int, int>> ss;
        double error = MovingEnvironment<S>::truncate_singular_values(
            qs, s, ss, k, cutoff, trunc_type);
        // ilr: row index in singular values list
        // im: number of states
        vector<uint16_t> ilr, im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (k != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        shared_ptr<SparseMatrixInfo<S>> linfo, rinfo;
        vector<vector<uint16_t>> idx_dm_to_wfn;
        if (trace_right) {
            linfo = MovingEnvironment<S>::rotation_matrix_info_from_svd(
                opdq, qs, l, true, ilr, im);
            rinfo = MovingEnvironment<S>::wavefunction_info_from_svd(
                qs, wfn->info, true, ilr, im, idx_dm_to_wfn);
        } else {
            linfo = MovingEnvironment<S>::wavefunction_info_from_svd(
                qs, wfn->info, false, ilr, im, idx_dm_to_wfn);
            rinfo = MovingEnvironment<S>::rotation_matrix_info_from_svd(
                opdq, qs, r, false, ilr, im);
        }
        int kk = ilr.size();
        left = make_shared<SparseMatrix<S>>();
        right = make_shared<SparseMatrix<S>>();
        left->allocate(linfo);
        right->allocate(rinfo);
        int iss = 0;
        if (trace_right) {
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(left->data + linfo->n_states_total[i] + j,
                                  linfo->n_states_bra[i], 1),
                        MatrixRef(
                            &l[ss[iss + j].first]->ref()(0, ss[iss + j].second),
                            linfo->n_states_bra[i], 1),
                        linfo->n_states_ket[i], l[ss[iss + j].first]->shape[1]);
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int ir = rinfo->find_state(wfn->info->quanta[iw]);
                    assert(ir != -1);
                    for (int j = 0; j < im[i]; j++) {
                        MatrixFunctions::copy(
                            MatrixRef(right->data + rinfo->n_states_total[ir] +
                                          j * r[iw]->shape[1],
                                      1, r[iw]->shape[1]),
                            MatrixRef(&r[iw]->ref()(ss[iss + j].second, 0), 1,
                                      r[iw]->shape[1]));
                        MatrixFunctions::iscale(
                            MatrixRef(right->data + rinfo->n_states_total[ir] +
                                          j * r[iw]->shape[1],
                                      1, r[iw]->shape[1]),
                            s[ss[iss + j].first]->data[ss[iss + j].second]);
                    }
                }
                iss += im[i];
            }
            if (normalize)
                right->normalize();
        } else {
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(right->data + rinfo->n_states_total[i] +
                                      j * r[ss[iss + j].first]->shape[1],
                                  1, r[ss[iss + j].first]->shape[1]),
                        MatrixRef(
                            &r[ss[iss + j].first]->ref()(ss[iss + j].second, 0),
                            1, r[ss[iss + j].first]->shape[1]));
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int il = linfo->find_state(wfn->info->quanta[iw]);
                    assert(il != -1);
                    for (int j = 0; j < im[i]; j++) {
                        MatrixFunctions::copy(
                            MatrixRef(left->data + linfo->n_states_total[il] +
                                          j,
                                      linfo->n_states_bra[il], 1),
                            MatrixRef(&l[iw]->ref()(0, ss[iss + j].second),
                                      linfo->n_states_bra[il], 1),
                            linfo->n_states_ket[il], l[iw]->shape[1]);
                        MatrixFunctions::iscale(
                            MatrixRef(left->data + linfo->n_states_total[il] +
                                          j,
                                      linfo->n_states_bra[il], 1),
                            s[ss[iss + j].first]->data[ss[iss + j].second],
                            linfo->n_states_ket[il]);
                    }
                }
                iss += im[i];
            }
            if (normalize)
                left->normalize();
        }
        assert(iss == ss.size());
        return error;
    }
    // Split wavefunction to two MPS tensors by solving eigenvalue problem
    static double split_density_matrix(
        const shared_ptr<SparseMatrix<S>> &dm,
        const shared_ptr<SparseMatrix<S>> &wfn, int k, bool trace_right,
        bool normalize, shared_ptr<SparseMatrix<S>> &left,
        shared_ptr<SparseMatrix<S>> &right, double cutoff,
        TruncationTypes trunc_type = TruncationTypes::Physical) {
        // ss: pair<quantum index in dm, reduced matrix index in dm>
        vector<pair<int, int>> ss;
        double error = MovingEnvironment<S>::truncate_density_matrix(
            dm, ss, k, cutoff, trunc_type);
        // ilr: row index in dm
        // im: number of states
        vector<uint16_t> ilr, im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (k != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        shared_ptr<SparseMatrixInfo<S>> linfo, rinfo;
        vector<vector<uint16_t>> idx_dm_to_wfn;
        if (trace_right) {
            linfo =
                MovingEnvironment<S>::rotation_matrix_info_from_density_matrix(
                    dm->info, true, ilr, im);
            rinfo = MovingEnvironment<S>::wavefunction_info_from_density_matrix(
                dm->info, wfn->info, true, ilr, im, idx_dm_to_wfn);
        } else {
            linfo = MovingEnvironment<S>::wavefunction_info_from_density_matrix(
                dm->info, wfn->info, false, ilr, im, idx_dm_to_wfn);
            rinfo =
                MovingEnvironment<S>::rotation_matrix_info_from_density_matrix(
                    dm->info, false, ilr, im);
        }
        int kk = ilr.size();
        left = make_shared<SparseMatrix<S>>();
        right = make_shared<SparseMatrix<S>>();
        left->allocate(linfo);
        right->allocate(rinfo);
        int iss = 0;
        if (trace_right) {
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(left->data + linfo->n_states_total[i] + j,
                                  linfo->n_states_bra[i], 1),
                        MatrixRef(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0),
                            linfo->n_states_bra[i], 1),
                        linfo->n_states_ket[i], 1);
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int ir = right->info->find_state(wfn->info->quanta[iw]);
                    assert(ir != -1);
                    MatrixFunctions::multiply((*left)[i], true, (*wfn)[iw],
                                              false, (*right)[ir], 1.0, 0.0);
                }
                iss += im[i];
            }
            if (normalize)
                right->normalize();
        } else {
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(right->data + rinfo->n_states_total[i] +
                                      j * (*right)[i].n,
                                  1, (*right)[i].n),
                        MatrixRef(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0), 1,
                            (*right)[i].n));
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int il = left->info->find_state(wfn->info->quanta[iw]);
                    assert(il != -1);
                    MatrixFunctions::multiply((*wfn)[iw], false, (*right)[i],
                                              true, (*left)[il], 1.0, 0.0);
                }
                iss += im[i];
            }
            if (normalize)
                left->normalize();
        }
        assert(iss == ss.size());
        return error;
    }
    // Split density matrix to two MultiMPS tensors by solving eigenvalue
    // problem
    static double multi_split_density_matrix(
        const shared_ptr<SparseMatrix<S>> &dm,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &wfns, int k,
        bool trace_right, bool normalize,
        vector<shared_ptr<SparseMatrixGroup<S>>> &new_wfns,
        shared_ptr<SparseMatrix<S>> &rot_mat, double cutoff,
        TruncationTypes trunc_type = TruncationTypes::Physical) {
        // ss: pair<quantum index in dm, reduced matrix index in dm>
        vector<pair<int, int>> ss;
        double error = MovingEnvironment<S>::truncate_density_matrix(
            dm, ss, k, cutoff, trunc_type);
        // ilr: row index in dm
        // im: number of states
        vector<uint16_t> ilr, im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (k != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        shared_ptr<SparseMatrixInfo<S>> rinfo;
        vector<shared_ptr<SparseMatrixInfo<S>>> winfos;
        vector<vector<vector<uint16_t>>> idx_dm_to_wfns;
        idx_dm_to_wfns.resize(wfns[0]->n);
        if (trace_right)
            rinfo =
                MovingEnvironment<S>::rotation_matrix_info_from_density_matrix(
                    dm->info, trace_right, ilr, im);
        winfos.resize(wfns[0]->n);
        for (size_t j = 0; j < wfns[0]->n; j++) {
            winfos[j] =
                MovingEnvironment<S>::wavefunction_info_from_density_matrix(
                    dm->info, wfns[0]->infos[j], trace_right, ilr, im,
                    idx_dm_to_wfns[j]);
        }
        if (!trace_right)
            rinfo =
                MovingEnvironment<S>::rotation_matrix_info_from_density_matrix(
                    dm->info, trace_right, ilr, im);
        int kk = ilr.size();
        rot_mat = make_shared<SparseMatrix<S>>();
        new_wfns =
            vector<shared_ptr<SparseMatrixGroup<S>>>(wfns.size(), nullptr);
        if (trace_right)
            rot_mat->allocate(rinfo);
        for (size_t k = 0; k < wfns.size(); k++) {
            new_wfns[k] = make_shared<SparseMatrixGroup<S>>();
            new_wfns[k]->allocate(winfos);
        }
        if (!trace_right)
            rot_mat->allocate(rinfo);
        int iss = 0;
        if (trace_right) {
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(rot_mat->data + rinfo->n_states_total[i] + j,
                                  rinfo->n_states_bra[i], 1),
                        MatrixRef(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0),
                            rinfo->n_states_bra[i], 1),
                        rinfo->n_states_ket[i], 1);
                for (size_t k = 0; k < wfns.size(); k++)
                    for (int j = 0; j < wfns[k]->n; j++)
                        for (int iww = 0;
                             iww < (int)idx_dm_to_wfns[j][ss[iss].first].size();
                             iww++) {
                            int iw = idx_dm_to_wfns[j][ss[iss].first][iww];
                            int ir = winfos[j]->find_state(
                                wfns[k]->infos[j]->quanta[iw]);
                            assert(ir != -1);
                            MatrixFunctions::multiply(
                                (*rot_mat)[i], true, (*(*wfns[k])[j])[iw],
                                false, (*(*new_wfns[k])[j])[ir], 1.0, 0.0);
                        }
                iss += im[i];
            }
            if (normalize)
                for (size_t k = 0; k < new_wfns.size(); k++)
                    new_wfns[k]->normalize();
        } else {
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(rot_mat->data + rinfo->n_states_total[i] +
                                      j * (*rot_mat)[i].n,
                                  1, (*rot_mat)[i].n),
                        MatrixRef(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0), 1,
                            (*rot_mat)[i].n));
                for (size_t k = 0; k < wfns.size(); k++)
                    for (int j = 0; j < wfns[k]->n; j++)
                        for (int iww = 0;
                             iww < (int)idx_dm_to_wfns[j][ss[iss].first].size();
                             iww++) {
                            int iw = idx_dm_to_wfns[j][ss[iss].first][iww];
                            int il = winfos[j]->find_state(
                                wfns[k]->infos[j]->quanta[iw]);
                            assert(il != -1);
                            MatrixFunctions::multiply(
                                (*(*wfns[k])[j])[iw], false, (*rot_mat)[i],
                                true, (*(*new_wfns[k])[j])[il], 1.0, 0.0);
                        }
                iss += im[i];
            }
            if (normalize)
                for (size_t k = 0; k < new_wfns.size(); k++)
                    new_wfns[k]->normalize();
        }
        assert(iss == ss.size());
        return error;
    }
    static shared_ptr<SparseMatrix<S>>
    swap_wfn_to_fused_left(int i, const shared_ptr<MPSInfo<S>> &mps_info,
                           const shared_ptr<SparseMatrix<S>> &old_wfn,
                           const shared_ptr<CG<S>> &cg) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        shared_ptr<SparseMatrixInfo<S>> wfn_info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>(d_alloc);
        mps_info->load_left_dims(i);
        mps_info->load_right_dims(i + 1);
        l = *mps_info->left_dims[i], m = *mps_info->basis[i],
        r = *mps_info->right_dims[i + 1];
        lm =
            StateInfo<S>::tensor_product(l, m, *mps_info->left_dims_fci[i + 1]);
        lmc = StateInfo<S>::get_connection_info(l, m, lm);
        mr = StateInfo<S>::tensor_product(m, r, *mps_info->right_dims_fci[i]);
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
    static shared_ptr<SparseMatrix<S>>
    swap_wfn_to_fused_right(int i, const shared_ptr<MPSInfo<S>> &mps_info,
                            const shared_ptr<SparseMatrix<S>> &old_wfn,
                            const shared_ptr<CG<S>> &cg) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        shared_ptr<SparseMatrixInfo<S>> wfn_info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>(d_alloc);
        mps_info->load_left_dims(i);
        mps_info->load_right_dims(i + 1);
        l = *mps_info->left_dims[i], m = *mps_info->basis[i],
        r = *mps_info->right_dims[i + 1];
        lm =
            StateInfo<S>::tensor_product(l, m, *mps_info->left_dims_fci[i + 1]);
        lmc = StateInfo<S>::get_connection_info(l, m, lm);
        mr = StateInfo<S>::tensor_product(m, r, *mps_info->right_dims_fci[i]);
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
    static vector<shared_ptr<SparseMatrixGroup<S>>>
    swap_multi_wfn_to_fused_left(
        int i, const shared_ptr<MPSInfo<S>> &mps_info,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &old_wfns,
        const shared_ptr<CG<S>> &cg) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        vector<shared_ptr<SparseMatrixInfo<S>>> wfn_infos;
        vector<shared_ptr<SparseMatrixGroup<S>>> wfns;
        mps_info->load_left_dims(i);
        mps_info->load_right_dims(i + 1);
        l = *mps_info->left_dims[i], m = *mps_info->basis[i],
        r = *mps_info->right_dims[i + 1];
        lm =
            StateInfo<S>::tensor_product(l, m, *mps_info->left_dims_fci[i + 1]);
        lmc = StateInfo<S>::get_connection_info(l, m, lm);
        mr = StateInfo<S>::tensor_product(m, r, *mps_info->right_dims_fci[i]);
        mrc = StateInfo<S>::get_connection_info(m, r, mr);
        vector<shared_ptr<SparseMatrixInfo<S>>> owinfos = old_wfns[0]->infos;
        wfn_infos.resize(old_wfns[0]->n);
        for (int j = 0; j < old_wfns[0]->n; j++) {
            wfn_infos[j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
            wfn_infos[j]->initialize(lm, r, owinfos[j]->delta_quantum,
                                     owinfos[j]->is_fermion,
                                     owinfos[j]->is_wavefunction);
        }
        wfns.resize(old_wfns.size());
        for (int k = 0; k < (int)old_wfns.size(); k++) {
            wfns[k] = make_shared<SparseMatrixGroup<S>>(d_alloc);
            wfns[k]->allocate(wfn_infos);
        }
        for (int k = 0; k < (int)old_wfns.size(); k++)
            for (int j = 0; j < old_wfns[k]->n; j++)
                (*wfns[k])[j]->swap_to_fused_left((*old_wfns[k])[j], l, m, r,
                                                  mr, mrc, lm, lmc, cg);
        mrc.deallocate(), mr.deallocate(), lmc.deallocate();
        lm.deallocate(), r.deallocate(), l.deallocate();
        return wfns;
    }
    static vector<shared_ptr<SparseMatrixGroup<S>>>
    swap_multi_wfn_to_fused_right(
        int i, const shared_ptr<MPSInfo<S>> &mps_info,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &old_wfns,
        const shared_ptr<CG<S>> &cg) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        vector<shared_ptr<SparseMatrixInfo<S>>> wfn_infos;
        vector<shared_ptr<SparseMatrixGroup<S>>> wfns;
        mps_info->load_left_dims(i);
        mps_info->load_right_dims(i + 1);
        l = *mps_info->left_dims[i], m = *mps_info->basis[i],
        r = *mps_info->right_dims[i + 1];
        lm =
            StateInfo<S>::tensor_product(l, m, *mps_info->left_dims_fci[i + 1]);
        lmc = StateInfo<S>::get_connection_info(l, m, lm);
        mr = StateInfo<S>::tensor_product(m, r, *mps_info->right_dims_fci[i]);
        mrc = StateInfo<S>::get_connection_info(m, r, mr);
        vector<shared_ptr<SparseMatrixInfo<S>>> owinfos = old_wfns[0]->infos;
        wfn_infos.resize(old_wfns[0]->n);
        for (int j = 0; j < old_wfns[0]->n; j++) {
            wfn_infos[j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
            wfn_infos[j]->initialize(l, mr, owinfos[j]->delta_quantum,
                                     owinfos[j]->is_fermion,
                                     owinfos[j]->is_wavefunction);
        }
        wfns.resize(old_wfns.size());
        for (int k = 0; k < (int)old_wfns.size(); k++) {
            wfns[k] = make_shared<SparseMatrixGroup<S>>(d_alloc);
            wfns[k]->allocate(wfn_infos);
        }
        for (int k = 0; k < (int)old_wfns.size(); k++)
            for (int j = 0; j < old_wfns[k]->n; j++)
                (*wfns[k])[j]->swap_to_fused_right((*old_wfns[k])[j], l, m, r,
                                                   lm, lmc, mr, mrc, cg);
        mrc.deallocate(), mr.deallocate(), lmc.deallocate();
        lm.deallocate(), r.deallocate(), l.deallocate();
        return wfns;
    }
    // Change the fusing type of MPS tensor so that it can be used in next sweep
    // iteration
    static void propagate_wfn(int i, int n_sites, const shared_ptr<MPS<S>> &mps,
                              bool forward, const shared_ptr<CG<S>> &cg) {
        if (forward) {
            if (i + 1 != n_sites - 1) {
                mps->load_tensor(i + 1);
                shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i + 1];
                mps->tensors[i + 1] =
                    swap_wfn_to_fused_left(i + 1, mps->info, old_wfn, cg);
                mps->save_tensor(i + 1);
                mps->unload_tensor(i + 1);
                old_wfn->info->deallocate();
                old_wfn->deallocate();
            }
        } else {
            if (i != 0) {
                mps->load_tensor(i);
                shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                mps->tensors[i] =
                    swap_wfn_to_fused_right(i, mps->info, old_wfn, cg);
                mps->save_tensor(i);
                mps->unload_tensor(i);
                old_wfn->info->deallocate();
                old_wfn->deallocate();
            }
        }
    }
    // Change the fusing type of MultiMPS tensor so that it can be used in next
    // sweep iteration
    static void propagate_multi_wfn(int i, int n_sites,
                                    const shared_ptr<MultiMPS<S>> &mps,
                                    bool forward, const shared_ptr<CG<S>> &cg) {
        if (forward) {
            if (i + 1 != n_sites - 1) {
                mps->load_wavefunction(i + 1);
                vector<shared_ptr<SparseMatrixGroup<S>>> old_wfns = mps->wfns;
                mps->wfns = swap_multi_wfn_to_fused_left(i + 1, mps->info,
                                                         old_wfns, cg);
                mps->save_wavefunction(i + 1);
                mps->unload_wavefunction(i + 1);
                for (int j = (int)old_wfns.size() - 1; j >= 0; j--)
                    old_wfns[j]->deallocate();
                if (old_wfns.size() != 0)
                    old_wfns[0]->deallocate_infos();
            }
        } else {
            if (i != 0) {
                mps->load_wavefunction(i);
                vector<shared_ptr<SparseMatrixGroup<S>>> old_wfns = mps->wfns;
                mps->wfns =
                    swap_multi_wfn_to_fused_right(i, mps->info, old_wfns, cg);
                mps->save_wavefunction(i);
                mps->unload_wavefunction(i);
                for (int j = (int)old_wfns.size() - 1; j >= 0; j--)
                    old_wfns[j]->deallocate();
                if (old_wfns.size() != 0)
                    old_wfns[0]->deallocate_infos();
            }
        }
    }
};

} // namespace block2
