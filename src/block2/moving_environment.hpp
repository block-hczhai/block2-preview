
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

#include "mpo.hpp"
#include "mps.hpp"
#include "partition.hpp"
#include "tensor_functions.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

// Effective Hamiltonian
template <typename S> struct EffectiveHamiltonian {
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
        right_op_infos;
    // Symbolic expression of effective H
    shared_ptr<DelayedOperatorTensor<S>> op;
    shared_ptr<SparseMatrix<S>> bra, ket, diag, cmat, vmat;
    shared_ptr<TensorFunctions<S>> tf;
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
          bra(bra), ket(ket), tf(tf), compute_diag(compute_diag) {
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
                                        cmat, vmat, opdq);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        }
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const MatrixRef &b, const MatrixRef &c, int idx = 0,
                    double factor = 1.0) {
        assert(b.m * b.n == cmat->total_memory);
        assert(c.m * c.n == vmat->total_memory);
        cmat->data = b.data;
        vmat->data = c.data;
        cmat->factor = factor;
        tf->tensor_product_multiply(op->mat->data[idx], op->lops, op->rops,
                                    cmat, vmat, opdq);
    }
    // Find eigenvalues and eigenvectors of [H_eff]
    // energy, ndav, nflop, tdav
    tuple<double, int, size_t, double> eigs(bool iprint = false) {
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
                ? MatrixFunctions::davidson(*tf->opf->seq, aa, bs, ndav, iprint)
                : MatrixFunctions::davidson(*this, aa, bs, ndav, iprint);
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners[0], ndav, nflop, t.get_time());
    }
    // [bra] = [H_eff] x [ket]
    // norm, nflop, tdav
    tuple<double, size_t, double> multiply() {
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
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(norm, nflop, t.get_time());
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tdav
    tuple<vector<pair<shared_ptr<OpExpr<S>>, double>>, size_t, double>
    expect() {
        Timer t;
        t.get_time();
        MatrixRef ktmp(ket->data, ket->total_memory, 1);
        MatrixRef rtmp(bra->data, bra->total_memory, 1);
        MatrixRef btmp(nullptr, bra->total_memory, 1);
        btmp.allocate();
        assert(tf->opf->seq->mode != SeqTypes::Auto);
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations;
        expectations.reserve(op->mat->data.size());
        for (size_t i = 0; i < op->mat->data.size(); i++) {
            if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->name ==
                OpNames::Zero)
                continue;
            else if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->q_label !=
                     opdq)
                expectations.push_back(make_pair(op->ops[i], 0.0));
            else {
                btmp.clear();
                (*this)(ktmp, btmp, i);
                double r = MatrixFunctions::dot(btmp, rtmp);
                expectations.push_back(make_pair(op->ops[i], r));
            }
        }
        btmp.deallocate();
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, nflop, t.get_time());
    }
    // [ket] = exp( [H_eff] ) | [ket] > (RK4 approximation)
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<MatrixRef>, tuple<double, double, int, size_t, double>>
    rk4_apply(double beta, double const_e, bool eval_energy = false) {
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
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(
            r, make_tuple(energy, norm, 4 + eval_energy, nflop, t.get_time()));
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    tuple<double, double, int, size_t, double>
    expo_apply(double beta, double const_e, bool iprint = false) {
        assert(compute_diag);
        double anorm =
            MatrixFunctions::norm(MatrixRef(diag->data, diag->total_memory, 1));
        MatrixRef v(ket->data, ket->total_memory, 1);
        Timer t;
        t.get_time();
        int nexpo = tf->opf->seq->mode == SeqTypes::Auto
                        ? MatrixFunctions::expo_apply(*tf->opf->seq, beta,
                                                      anorm, v, const_e, iprint)
                        : MatrixFunctions::expo_apply(*this, beta, anorm, v,
                                                      const_e, iprint);
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
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, nflop, t.get_time());
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

enum FuseTypes : uint8_t { NoFuse = 0, FuseL = 1, FuseR = 2, FuseLR = 3 };

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
    MovingEnvironment(const shared_ptr<MPO<S>> &mpo,
                      const shared_ptr<MPS<S>> &bra,
                      const shared_ptr<MPS<S>> &ket, const string &tag = "DMRG")
        : n_sites(ket->n_sites), center(ket->center), dot(ket->dot), mpo(mpo),
          bra(bra), ket(ket), tag(tag) {
        assert(bra->n_sites == ket->n_sites && mpo->n_sites == ket->n_sites);
        assert(bra->center == ket->center && bra->dot == ket->dot);
        hop_mat = make_shared<SymbolicColumnVector<S>>(1);
        (*hop_mat)[0] = mpo->op;
    }
    // Contract and renormalize left block by one site
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
            mpo->site_op_infos[bra->info->orbsym[i - 1]], left_op_infos_notrunc,
            mpo->tf->opf->cg);
        frame->activate(0);
        shared_ptr<OperatorTensor<S>> new_left = Partition<S>::build_left(
            {mpo->left_operator_names[i - 1]}, left_op_infos_notrunc);
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
            envs[i + 1]->right_op_infos,
            mpo->site_op_infos[bra->info->orbsym[i + dot]],
            right_op_infos_notrunc, mpo->tf->opf->cg);
        frame->activate(0);
        shared_ptr<OperatorTensor<S>> new_right = Partition<S>::build_right(
            {mpo->right_operator_names[i + dot]}, right_op_infos_notrunc);
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
        ss << frame->save_dir << "/" << frame->prefix << ".PART." << tag
           << ".LEFT." << Parsing::to_string(i);
        return ss.str();
    }
    string get_right_partition_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".PART." << tag
           << ".RIGHT." << Parsing::to_string(i);
        return ss.str();
    }
    // Generate contracted environment blocks for all center sites
    void init_environments(bool iprint = false) {
        envs.clear();
        envs.resize(n_sites);
        for (int i = 0; i < n_sites; i++) {
            envs[i] =
                make_shared<Partition<S>>(nullptr, nullptr, mpo->tensors[i]);
            if (i != n_sites - 1 && dot == 2)
                envs[i]->middle.push_back(mpo->tensors[i + 1]);
        }
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
        frame->reset(1);
    }
    // Remove old environment for starting a new sweep
    void prepare() {
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
    void move_to(int i) {
        if (i > center) {
            frame->load_data(1, get_left_partition_filename(center));
            left_contract_rotate(++center);
        } else if (i < center) {
            frame->load_data(1, get_right_partition_filename(center));
            right_contract_rotate(--center);
        }
        bra->center = ket->center = center;
    }
    // Generate effective hamiltonian at current center site
    shared_ptr<EffectiveHamiltonian<S>> eff_ham(FuseTypes fuse_type,
                                                bool compute_diag) {
        if (dot == 2) {
            vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
                right_op_infos;
            shared_ptr<OperatorTensor<S>> new_left, new_right;
            int iL = -1, iR = -1, iM = -1;
            if (fuse_type == FuseTypes::FuseLR)
                iL = center, iR = center + 1, iM = center;
            else if (fuse_type == FuseTypes::FuseR)
                iL = center, iR = center, iM = center - 1;
            else if (fuse_type == FuseTypes::FuseL)
                iL = center + 1, iR = center + 1, iM = center + 1;
            else
                assert(false);
            if (fuse_type & FuseTypes::FuseL) {
                // left contract infos
                vector<shared_ptr<Symbolic<S>>> lmats = {
                    mpo->left_operator_names[iL]};
                if (mpo->schemer != nullptr &&
                    iL == mpo->schemer->left_trans_site &&
                    mpo->schemer->right_trans_site -
                            mpo->schemer->left_trans_site <=
                        1)
                    lmats.push_back(mpo->schemer->left_new_operator_names);
                vector<S> lsl = Partition<S>::get_uniq_labels(lmats);
                shared_ptr<Symbolic<S>> lexprs =
                    envs[iL]->left == nullptr
                        ? nullptr
                        : (mpo->left_operator_exprs.size() != 0
                               ? mpo->left_operator_exprs[iL]
                               : envs[iL]->left->lmat *
                                     envs[iL]->middle.front()->lmat);
                vector<vector<pair<uint8_t, S>>> lsubsl =
                    Partition<S>::get_uniq_sub_labels(
                        lexprs, mpo->left_operator_names[iL], lsl);
                if (envs[iL]->left != nullptr)
                    frame->load_data(1, get_left_partition_filename(iL));
                Partition<S>::init_left_op_infos_notrunc(
                    iL, bra->info, ket->info, lsl, lsubsl,
                    envs[iL]->left_op_infos,
                    mpo->site_op_infos[bra->info->orbsym[iL]], left_op_infos,
                    mpo->tf->opf->cg);
                // left contract
                frame->activate(0);
                new_left = Partition<S>::build_left(lmats, left_op_infos);
                mpo->tf->left_contract(envs[iL]->left, envs[iL]->middle.front(),
                                       new_left,
                                       mpo->left_operator_exprs.size() != 0
                                           ? mpo->left_operator_exprs[iL]
                                           : nullptr);
                if (mpo->schemer != nullptr &&
                    iL == mpo->schemer->left_trans_site &&
                    mpo->schemer->right_trans_site -
                            mpo->schemer->left_trans_site <=
                        1)
                    mpo->tf->numerical_transform(
                        new_left, lmats[1],
                        mpo->schemer->left_new_operator_exprs);
            } else {
                assert(envs[iL]->left != nullptr);
                frame->load_data(1, get_left_partition_filename(iL));
                frame->activate(0);
                Partition<S>::copy_op_infos(envs[iL]->left_op_infos,
                                            left_op_infos);
                new_left = envs[iL]->left->deep_copy();
                for (auto &p : new_left->ops)
                    p.second->info = Partition<S>::find_op_info(
                        left_op_infos, p.second->info->delta_quantum);
            }
            if (fuse_type & FuseTypes::FuseR) {
                // right contract infos
                vector<shared_ptr<Symbolic<S>>> rmats = {
                    mpo->right_operator_names[iR]};
                vector<S> rsl = Partition<S>::get_uniq_labels(rmats);
                shared_ptr<Symbolic<S>> rexprs =
                    envs[iR - 1]->right == nullptr
                        ? nullptr
                        : (mpo->right_operator_exprs.size() != 0
                               ? mpo->right_operator_exprs[iR]
                               : envs[iR - 1]->middle.back()->rmat *
                                     envs[iR - 1]->right->rmat);
                vector<vector<pair<uint8_t, S>>> rsubsl =
                    Partition<S>::get_uniq_sub_labels(
                        rexprs, mpo->right_operator_names[iR], rsl);
                if (envs[iR - 1]->right != nullptr)
                    frame->load_data(1, get_right_partition_filename(iR - 1));
                Partition<S>::init_right_op_infos_notrunc(
                    iR, bra->info, ket->info, rsl, rsubsl,
                    envs[iR - 1]->right_op_infos,
                    mpo->site_op_infos[bra->info->orbsym[iR]], right_op_infos,
                    mpo->tf->opf->cg);
                // right contract
                frame->activate(0);
                new_right = Partition<S>::build_right(rmats, right_op_infos);
                mpo->tf->right_contract(envs[iR - 1]->right,
                                        envs[iR - 1]->middle.back(), new_right,
                                        mpo->right_operator_exprs.size() != 0
                                            ? mpo->right_operator_exprs[iR]
                                            : nullptr);
            } else {
                assert(envs[iR - 1]->right != nullptr);
                frame->load_data(1, get_right_partition_filename(iR - 1));
                frame->activate(0);
                Partition<S>::copy_op_infos(envs[iR - 1]->right_op_infos,
                                            right_op_infos);
                new_right = envs[iR - 1]->right->deep_copy();
                for (auto &p : new_right->ops)
                    p.second->info = Partition<S>::find_op_info(
                        right_op_infos, p.second->info->delta_quantum);
            }
            // delayed left-right contract
            shared_ptr<DelayedOperatorTensor<S>> op =
                mpo->middle_operator_exprs.size() != 0
                    ? TensorFunctions<S>::delayed_contract(
                          new_left, new_right, mpo->middle_operator_names[iM],
                          mpo->middle_operator_exprs[iM])
                    : TensorFunctions<S>::delayed_contract(new_left, new_right,
                                                           mpo->op);
            frame->activate(0);
            frame->reset(1);
            shared_ptr<SymbolicColumnVector<S>> hops =
                mpo->middle_operator_exprs.size() != 0
                    ? dynamic_pointer_cast<SymbolicColumnVector<S>>(
                          mpo->middle_operator_names[iM])
                    : hop_mat;
            shared_ptr<EffectiveHamiltonian<S>> efh =
                make_shared<EffectiveHamiltonian<S>>(
                    left_op_infos, right_op_infos, op, bra->tensors[iL],
                    ket->tensors[iL], mpo->op, hops, mpo->tf, compute_diag);
            return efh;
        } else
            return nullptr;
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
            StateInfo<S> l = mps->info->left_dims[i],
                         ml = mps->info->basis[mps->info->orbsym[i]],
                         mr = mps->info->basis[mps->info->orbsym[i + 1]],
                         r = mps->info->right_dims[i + 2];
            StateInfo<S> ll = StateInfo<S>::tensor_product(
                l, ml, mps->info->left_dims_fci[i + 1]);
            StateInfo<S> rr = StateInfo<S>::tensor_product(
                mr, r, mps->info->right_dims_fci[i + 1]);
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
    // Density matrix of a MPS tensor
    static shared_ptr<SparseMatrix<S>>
    density_matrix(S opdq, const shared_ptr<SparseMatrix<S>> &psi,
                   bool trace_right, double noise) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(psi->info, opdq, trace_right);
        shared_ptr<SparseMatrix<S>> dm = make_shared<SparseMatrix<S>>();
        dm->allocate(dm_info);
        OperatorFunctions<S>::trans_product(*psi, *dm, trace_right, noise);
        return dm;
    }
    // Density matrix of several MPS tensors summed with weights
    static shared_ptr<SparseMatrix<S>>
    density_matrix_with_weights(S opdq, const shared_ptr<SparseMatrix<S>> &psi,
                                bool trace_right, double noise,
                                const vector<MatrixRef> &mats,
                                const vector<double> &weights) {
        double *ptr = psi->data;
        assert(psi->factor == 1.0);
        assert(mats.size() == weights.size() - 1);
        psi->factor = sqrt(weights[0]);
        shared_ptr<SparseMatrix<S>> dm =
            density_matrix(opdq, psi, trace_right, noise);
        for (size_t i = 1; i < weights.size(); i++) {
            psi->data = mats[i - 1].data;
            psi->factor = sqrt(weights[i]);
            OperatorFunctions<S>::trans_product(*psi, *dm, trace_right, 0.0);
        }
        psi->data = ptr, psi->factor = 1.0;
        return dm;
    }
    // Split density matrix to two MPS tensors by solving eigenvalue problem
    static double split_density_matrix(const shared_ptr<SparseMatrix<S>> &dm,
                                       const shared_ptr<SparseMatrix<S>> &wfn,
                                       int k, bool trace_right, bool normalize,
                                       shared_ptr<SparseMatrix<S>> &left,
                                       shared_ptr<SparseMatrix<S>> &right) {
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
            MatrixFunctions::iscale(wr,
                                    1.0 / dm->info->quanta[i].multiplicity());
            eigen_values.push_back(w);
            eigen_values_reduced.push_back(wr);
            k_total += w.n;
        }
        shared_ptr<SparseMatrixInfo<S>> linfo =
            make_shared<SparseMatrixInfo<S>>();
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>();
        double error = 0.0;
        vector<pair<int, int>> ss;
        ss.reserve(k_total);
        for (int i = 0; i < (int)eigen_values.size(); i++)
            for (int j = 0; j < eigen_values[i].n; j++)
                ss.push_back(make_pair(i, j));
        if (k != -1 && k_total > k) {
            sort(ss.begin(), ss.end(),
                 [&eigen_values_reduced](const pair<int, int> &a,
                                         const pair<int, int> &b) {
                     return eigen_values_reduced[a.first].data[a.second] >
                            eigen_values_reduced[b.first].data[b.second];
                 });
            for (int i = k; i < k_total; i++)
                error += eigen_values[ss[i].first].data[ss[i].second];
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
        int kk = ilr.size();
        linfo->is_fermion = rinfo->is_fermion = false;
        linfo->is_wavefunction = !trace_right;
        rinfo->is_wavefunction = trace_right;
        linfo->delta_quantum =
            trace_right ? dm->info->delta_quantum : wfn->info->delta_quantum;
        rinfo->delta_quantum =
            trace_right ? wfn->info->delta_quantum : dm->info->delta_quantum;
        linfo->allocate(kk);
        rinfo->allocate(kk);
        uint16_t idx_dm_to_wfn[dm->info->n];
        if (trace_right) {
            for (int i = 0; i < wfn->info->n; i++) {
                S pb = wfn->info->quanta[i].get_bra(wfn->info->delta_quantum);
                idx_dm_to_wfn[dm->info->find_state(pb)] = i;
            }
            for (int i = 0; i < kk; i++) {
                linfo->quanta[i] = dm->info->quanta[ilr[i]];
                rinfo->quanta[i] = wfn->info->quanta[idx_dm_to_wfn[ilr[i]]];
                linfo->n_states_bra[i] = dm->info->n_states_bra[ilr[i]];
                linfo->n_states_ket[i] = im[i];
                rinfo->n_states_bra[i] = im[i];
                rinfo->n_states_ket[i] =
                    wfn->info->n_states_ket[idx_dm_to_wfn[ilr[i]]];
            }
            linfo->n_states_total[0] = 0;
            for (int i = 0; i < kk - 1; i++)
                linfo->n_states_total[i + 1] =
                    linfo->n_states_total[i] +
                    (uint32_t)linfo->n_states_bra[i] * linfo->n_states_ket[i];
            rinfo->sort_states();
        } else {
            for (int i = 0; i < wfn->info->n; i++) {
                S pk = -wfn->info->quanta[i].get_ket();
                idx_dm_to_wfn[dm->info->find_state(pk)] = i;
            }
            for (int i = 0; i < kk; i++) {
                linfo->quanta[i] = wfn->info->quanta[idx_dm_to_wfn[ilr[i]]];
                rinfo->quanta[i] = dm->info->quanta[ilr[i]];
                linfo->n_states_bra[i] =
                    wfn->info->n_states_bra[idx_dm_to_wfn[ilr[i]]];
                linfo->n_states_ket[i] = im[i];
                rinfo->n_states_bra[i] = im[i];
                rinfo->n_states_ket[i] = dm->info->n_states_ket[ilr[i]];
            }
            linfo->sort_states();
            rinfo->n_states_total[0] = 0;
            for (int i = 0; i < kk - 1; i++)
                rinfo->n_states_total[i + 1] =
                    rinfo->n_states_total[i] +
                    (uint32_t)rinfo->n_states_bra[i] * rinfo->n_states_ket[i];
        }
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
                int iw = idx_dm_to_wfn[ss[iss].first];
                int ir = right->info->find_state(wfn->info->quanta[iw]);
                assert(ir != -1);
                MatrixFunctions::multiply((*left)[i], true, (*wfn)[iw], false,
                                          (*right)[ir], 1.0, 0.0);
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
                int iw = idx_dm_to_wfn[ss[iss].first];
                int il = left->info->find_state(wfn->info->quanta[iw]);
                assert(il != -1);
                MatrixFunctions::multiply((*wfn)[iw], false, (*right)[i], true,
                                          (*left)[il], 1.0, 0.0);
                iss += im[i];
            }
            if (normalize)
                left->normalize();
        }
        assert(iss == ss.size());
        return error;
    }
    // Change the fusing type of MPS tensor so that it can be used in next sweep iteration
    static void propagate_wfn(int i, int n_sites, const shared_ptr<MPS<S>> &mps,
                              bool forward) {
        shared_ptr<MPSInfo<S>> mps_info = mps->info;
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        shared_ptr<SparseMatrixInfo<S>> wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>();
        bool swapped = false;
        if (forward) {
            if ((swapped = i + 1 != n_sites - 1)) {
                mps_info->load_left_dims(i + 1);
                mps_info->load_right_dims(i + 2);
                l = mps_info->left_dims[i + 1],
                m = mps_info->basis[mps_info->orbsym[i + 1]],
                r = mps_info->right_dims[i + 2];
                lm = StateInfo<S>::tensor_product(
                    l, m, mps_info->left_dims_fci[i + 2]);
                lmc = StateInfo<S>::get_connection_info(l, m, lm);
                mr = StateInfo<S>::tensor_product(
                    m, r, mps_info->right_dims_fci[i + 1]);
                mrc = StateInfo<S>::get_connection_info(m, r, mr);
                shared_ptr<SparseMatrixInfo<S>> owinfo =
                    mps->tensors[i + 1]->info;
                wfn_info->initialize(lm, r, owinfo->delta_quantum,
                                     owinfo->is_fermion,
                                     owinfo->is_wavefunction);
                wfn->allocate(wfn_info);
                mps->load_tensor(i + 1);
                wfn->swap_to_fused_left(mps->tensors[i + 1], l, m, r, mr, mrc,
                                        lm, lmc);
                mps->unload_tensor(i + 1);
                mps->tensors[i + 1] = wfn;
                mps->save_tensor(i + 1);
            }
        } else {
            if ((swapped = i != 0)) {
                mps_info->load_left_dims(i);
                mps_info->load_right_dims(i + 1);
                l = mps_info->left_dims[i],
                m = mps_info->basis[mps_info->orbsym[i]],
                r = mps_info->right_dims[i + 1];
                lm = StateInfo<S>::tensor_product(
                    l, m, mps_info->left_dims_fci[i + 1]);
                lmc = StateInfo<S>::get_connection_info(l, m, lm);
                mr = StateInfo<S>::tensor_product(m, r,
                                                  mps_info->right_dims_fci[i]);
                mrc = StateInfo<S>::get_connection_info(m, r, mr);
                shared_ptr<SparseMatrixInfo<S>> owinfo = mps->tensors[i]->info;
                wfn_info->initialize(l, mr, owinfo->delta_quantum,
                                     owinfo->is_fermion,
                                     owinfo->is_wavefunction);
                wfn->allocate(wfn_info);
                mps->load_tensor(i);
                wfn->swap_to_fused_right(mps->tensors[i], l, m, r, lm, lmc, mr,
                                         mrc);
                mps->unload_tensor(i);
                mps->tensors[i] = wfn;
                mps->save_tensor(i);
            }
        }
        if (swapped) {
            wfn->deallocate();
            wfn_info->deallocate();
            mrc.deallocate();
            mr.deallocate();
            lmc.deallocate();
            lm.deallocate();
            r.deallocate();
            l.deallocate();
        }
    }
};

} // namespace block2
