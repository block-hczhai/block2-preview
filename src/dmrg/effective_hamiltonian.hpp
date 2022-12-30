
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

#include "../core/complex_matrix_functions.hpp"
#include "../core/iterative_matrix_functions.hpp"
#include "../core/tensor_functions.hpp"
#include "mpo.hpp"
#include "mps.hpp"
#include "partition.hpp"
#include "state_averaged.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
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

enum struct ExpectationAlgorithmTypes : uint16_t {
    Automatic = 1,
    Normal = 2,
    Fast = 4,
    SymbolFree = 8,
    Compressed = 16,
    LowMem = 32
};

inline ExpectationAlgorithmTypes operator|(ExpectationAlgorithmTypes a,
                                           ExpectationAlgorithmTypes b) {
    return ExpectationAlgorithmTypes((uint16_t)a | (uint16_t)b);
}

inline uint16_t operator&(ExpectationAlgorithmTypes a,
                          ExpectationAlgorithmTypes b) {
    return (uint16_t)a & (uint16_t)b;
}

enum struct ExpectationTypes : uint8_t { Real, Complex };

enum struct LinearSolverTypes : uint8_t {
    Automatic,
    CG,
    MinRes,
    GCROT,
    IDRS,
    LSQR,
    Cheby
};

template <typename S, typename FL, typename = MPS<S, FL>>
struct EffectiveHamiltonian;

// Effective Hamiltonian
template <typename S, typename FL>
struct EffectiveHamiltonian<S, FL, MPS<S, FL>> {
    typedef S ST;
    typedef FL FLT;
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FC FC;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
        right_op_infos;
    // Symbolic expression of effective H
    shared_ptr<DelayedOperatorTensor<S, FL>> op;
    shared_ptr<SparseMatrix<S, FL>> bra, ket, diag, cmat, vmat;
    shared_ptr<TensorFunctions<S, FL>> tf;
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    // Delta quantum of effective H
    S opdq;
    // Left vacuum of MPO
    S hop_left_vacuum;
    // Whether diagonal element of effective H should be computed
    bool compute_diag;
    vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>> wfn_infos;
    vector<S> operator_quanta;
    shared_ptr<NPDMScheme> npdm_scheme = nullptr;
    string npdm_fragment_filename = "";
    int npdm_n_sites = 0, npdm_center = -1, npdm_parallel_center = -1;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S, FL>> &op,
        const shared_ptr<SparseMatrix<S, FL>> &bra,
        const shared_ptr<SparseMatrix<S, FL>> &ket,
        const shared_ptr<OpElement<S, FL>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat, S hop_left_vacuum,
        const shared_ptr<TensorFunctions<S, FL>> &ptf, bool compute_diag = true,
        const shared_ptr<NPDMScheme> &npdm_scheme = nullptr)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(ptf->copy()), hop_mat(hop_mat),
          hop_left_vacuum(hop_left_vacuum), compute_diag(compute_diag),
          npdm_scheme(npdm_scheme) {
        // wavefunction
        if (compute_diag) {
            // for non-hermitian hamiltonian, bra and ket may share the same
            // info but they are different objects
            assert(bra->info->n == ket->info->n);
            diag = make_shared<SparseMatrix<S, FL>>();
            diag->allocate(ket->info);
        }
        // unique sub labels
        S cdq = ket->info->delta_quantum;
        S vdq = bra->info->delta_quantum;
        opdq = hop->q_label;
        vector<S> msl = Partition<S, FL>::get_uniq_labels({hop_mat});
        operator_quanta = msl;
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S, FL>::get_uniq_sub_labels(op->mat, hop_mat, msl,
                                                  hop_left_vacuum);
        // symbol-free npdm case
        if (npdm_scheme != nullptr && op->mat->data.size() == 1 &&
            dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->name ==
                OpNames::XPDM &&
            dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->site_index ==
                SiteIndex()) {
            for (int i = 0; i < (int)msl.size(); i++) {
                set<S> set_subsl;
                for (auto &pl : left_op_infos)
                    for (auto &pr : right_op_infos) {
                        S p = msl[i].combine(pl.first, -pr.first);
                        if (p != S(S::invalid))
                            msubsl[i].push_back(make_pair(0, p));
                    }
                sort(msubsl[i].begin(), msubsl[i].end());
                msubsl[i].resize(
                    distance(msubsl[i].begin(),
                             unique(msubsl[i].begin(), msubsl[i].end())));
            }
        }
        // tensor product diagonal
        if (compute_diag) {
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> diag_info =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            diag_info->initialize_diag(cdq, opdq, msubsl[0], left_op_infos,
                                       right_op_infos, diag->info, tf->opf->cg);
            diag->info->cinfo = diag_info;
            tf->tensor_product_diagonal(op->mat->data[0], op->lopt, op->ropt,
                                        diag, opdq);
            diag_info->deallocate();
        }
        // temp wavefunction
        cmat = make_shared<SparseMatrix<S, FL>>();
        vmat = make_shared<SparseMatrix<S, FL>>();
        *cmat = *ket;
        *vmat = *bra;
        // temp wavefunction info
        wfn_infos.resize(msl.size(), nullptr);
        for (int i = 0; i < (int)msl.size(); i++)
            if (msl[i].combine(vdq, cdq) != S(S::invalid)) {
                wfn_infos[i] =
                    make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
                wfn_infos[i]->initialize_wfn(cdq, vdq, msl[i], msubsl[i],
                                             left_op_infos, right_op_infos,
                                             ket->info, bra->info, tf->opf->cg);
            }
        cmat->info->cinfo = nullptr;
        for (int i = 0; i < (int)msl.size(); i++)
            if (wfn_infos[i] != nullptr) {
                cmat->info->cinfo = wfn_infos[i];
                break;
            }
    }
    // prepare batch gemm
    void precompute() const {
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            cmat->data = vmat->data = (FL *)0;
            cmat->factor = 1.0;
            tf->tensor_product_multiply(op->mat->data[0], op->lopt, op->ropt,
                                        cmat, vmat, opdq, false);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        } else if (tf->opf->seq->mode & SeqTypes::Tasked) {
            cmat->data = vmat->data = (FL *)0;
            cmat->factor = 1.0;
            tf->tensor_product_multiply(op->mat->data[0], op->lopt, op->ropt,
                                        cmat, vmat, opdq, false);
        }
    }
    void post_precompute() const {
        if (tf->opf->seq->mode == SeqTypes::Auto ||
            (tf->opf->seq->mode & SeqTypes::Tasked)) {
            tf->opf->seq->deallocate();
            tf->opf->seq->clear();
        }
    }
    shared_ptr<SparseMatrixGroup<S, FL>>
    perturbative_noise(bool trace_right, int iL, int iR, FuseTypes ftype,
                       const shared_ptr<MPSInfo<S>> &mps_info,
                       const NoiseTypes noise_type,
                       const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<S> msl = Partition<S, FL>::get_uniq_labels({hop_mat});
        assert(msl.size() == 1 && msl[0] == opdq);
        shared_ptr<OpExpr<S>> pexpr = op->mat->data[0];
        shared_ptr<Symbolic<S>> pmat = make_shared<SymbolicColumnVector<S>>(
            1, vector<shared_ptr<OpExpr<S>>>{pexpr});
        vector<pair<uint8_t, S>> psubsl = Partition<S, FL>::get_uniq_sub_labels(
            pmat, hop_mat, msl, hop_left_vacuum, true, trace_right, false)[0];
        vector<S> perturb_ket_labels, all_perturb_ket_labels;
        S ket_label = ket->info->delta_quantum;
        for (size_t j = 0; j < psubsl.size(); j++) {
            S pks = ket_label + psubsl[j].second;
            for (int k = 0; k < pks.count(); k++)
                perturb_ket_labels.push_back(pks[k]);
        }
        sort(psubsl.begin(), psubsl.end());
        psubsl.resize(
            distance(psubsl.begin(), unique(psubsl.begin(), psubsl.end())));
        all_perturb_ket_labels = perturb_ket_labels;
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
        shared_ptr<SparseMatrixGroup<S, FL>> perturb_ket =
            make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
        assert(noise_type & NoiseTypes::Perturbative);
        bool do_reduce = !(noise_type & NoiseTypes::Collected);
        bool reduced = noise_type & NoiseTypes::Reduced;
        bool low_mem = noise_type & NoiseTypes::LowMem;
        if (reduced)
            perturb_ket->allocate(infos);
        else {
            vector<shared_ptr<SparseMatrixInfo<S>>> all_infos;
            all_infos.reserve(all_perturb_ket_labels.size());
            for (S q : all_perturb_ket_labels) {
                size_t ib = lower_bound(perturb_ket_labels.begin(),
                                        perturb_ket_labels.end(), q) -
                            perturb_ket_labels.begin();
                all_infos.push_back(infos[ib]);
            }
            perturb_ket->allocate(all_infos);
        }
        // connection infos
        frame_<FP>()->activate(0);
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
                size_t ib = lower_bound(perturb_ket_labels.begin(),
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
        int vidx = reduced ? -1 : 0;
        // perform multiplication
        tf->tensor_product_partial_multiply(pexpr, op->lopt, op->ropt,
                                            trace_right, ket, psubsl, cinfos,
                                            perturb_ket_labels, perturb_ket,
                                            vidx, low_mem ? -2 : -1, do_reduce);
        if (!reduced)
            assert(vidx == perturb_ket->n);
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            tf->opf->seq->auto_perform();
            if (para_rule != nullptr && do_reduce)
                para_rule->comm->reduce_sum(perturb_ket, para_rule->comm->root);
        } else if (tf->opf->seq->mode & SeqTypes::Tasked) {
            if (!low_mem) {
                assert(perturb_ket->total_memory <=
                       (size_t)numeric_limits<decltype(GMatrix<FL>::n)>::max());
                tf->opf->seq->auto_perform(GMatrix<FL>(
                    perturb_ket->data, (MKL_INT)perturb_ket->total_memory, 1));
            } else {
                vector<GMatrix<FL>> pmats(perturb_ket->n,
                                          GMatrix<FL>(nullptr, 0, 0));
                for (int j = 0; j < perturb_ket->n; j++)
                    pmats[j] = GMatrix<FL>(
                        (*perturb_ket)[j]->data,
                        (MKL_INT)(*perturb_ket)[j]->total_memory, 1);
                tf->opf->seq->auto_perform(pmats);
            }
            if (para_rule != nullptr && do_reduce)
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
                 dynamic_pointer_cast<OpSum<S, FL>>(op->mat->data[0])
                     ->strings) {
                if (opx->get_type() == OpTypes::Prod ||
                    opx->get_type() == OpTypes::Elem)
                    r++;
                else if (opx->get_type() == OpTypes::SumProd)
                    r += (int)dynamic_pointer_cast<OpSumProd<S, FL>>(opx)
                             ->ops.size();
            }
            return r;
        } else if (op->mat->data[0]->get_type() == OpTypes::SumProd)
            return (int)dynamic_pointer_cast<OpSumProd<S, FL>>(op->mat->data[0])
                ->ops.size();
        else
            return 1;
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const GMatrix<FL> &b, const GMatrix<FL> &c, int idx = 0,
                    FL factor = 1.0, bool all_reduce = true) {
        assert(b.m * b.n == cmat->total_memory);
        assert(c.m * c.n == vmat->total_memory);
        cmat->data = b.data;
        vmat->data = c.data;
        cmat->factor = factor;
        S idx_opdq =
            dynamic_pointer_cast<OpElement<S, FL>>(op->dops[idx])->q_label;
        size_t ic = lower_bound(operator_quanta.begin(), operator_quanta.end(),
                                idx_opdq) -
                    operator_quanta.begin();
        assert(ic < operator_quanta.size() && wfn_infos[ic] != nullptr);
        cmat->info->cinfo = wfn_infos[ic];
        tf->tensor_product_multiply(op->mat->data[idx], op->lopt, op->ropt,
                                    cmat, vmat, idx_opdq, all_reduce);
    }
    // Find eigenvalues and eigenvectors of [H_eff]
    // energy, ndav, nflop, tdav
    tuple<typename const_fl_type<FP>::FL, int, size_t, double>
    eigs(bool iprint = false, FP conv_thrd = 5E-6, int max_iter = 5000,
         int soft_max_iter = -1,
         DavidsonTypes davidson_type = DavidsonTypes::Normal, FP shift = 0,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr,
         const vector<shared_ptr<SparseMatrix<S, FL>>> &ortho_bra =
             vector<shared_ptr<SparseMatrix<S, FL>>>(),
         const vector<FP> &projection_weights = vector<FP>()) {
        int ndav = 0;
        assert(compute_diag);
        GDiagonalMatrix<FL> aa(diag->data, (MKL_INT)diag->total_memory);
        vector<GMatrix<FL>> bs = vector<GMatrix<FL>>{
            GMatrix<FL>(ket->data, (MKL_INT)ket->total_memory, 1)};
        if (davidson_type & DavidsonTypes::LeftEigen)
            bs.push_back(GMatrix<FL>(bra->data, (MKL_INT)bra->total_memory, 1));
        vector<GMatrix<FL>> ors =
            vector<GMatrix<FL>>(ortho_bra.size(), GMatrix<FL>(nullptr, 0, 0));
        for (size_t i = 0; i < ortho_bra.size(); i++)
            ors[i] = GMatrix<FL>(ortho_bra[i]->data,
                                 (MKL_INT)ortho_bra[i]->total_memory, 1);
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        vector<FP> eners =
            (tf->opf->seq->mode == SeqTypes::Auto ||
             (tf->opf->seq->mode & SeqTypes::Tasked))
                ? IterativeMatrixFunctions<FL>::harmonic_davidson(
                      *tf, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, 2, 50, ors,
                      projection_weights)
                : IterativeMatrixFunctions<FL>::harmonic_davidson(
                      *this, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, 2, 50, ors,
                      projection_weights);
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple((typename const_fl_type<FP>::FL)eners[0], ndav,
                          (size_t)nflop, t.get_time());
    }
    // [bra] = [H_eff]^(-1) x [ket]
    // energy, nmult, nflop, tmult
    tuple<FL, pair<int, int>, size_t, double> inverse_multiply(
        typename const_fl_type<FL>::FL const_e, LinearSolverTypes solver_type,
        pair<int, int> linear_solver_params, bool iprint = false,
        FP conv_thrd = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        if (solver_type == LinearSolverTypes::Automatic)
            solver_type = LinearSolverTypes::MinRes;
        int nmult = 0, niter = 0;
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        GMatrix<FL> mket(ket->data, (MKL_INT)ket->total_memory, 1);
        GMatrix<FL> mbra(bra->data, (MKL_INT)bra->total_memory, 1);
        tf->opf->seq->cumulative_nflop = 0;
        GDiagonalMatrix<FL> aa(nullptr, 0);
        if (compute_diag && solver_type != LinearSolverTypes::MinRes) {
            aa = GDiagonalMatrix<FL>(nullptr, (MKL_INT)diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++)
                aa.data[i] = diag->data[i] + (FL)const_e;
        }
        precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &)> &f =
            [this](const GMatrix<FL> &a, const GMatrix<FL> &b) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b);
                else
                    return (*this)(a, b);
            };
        FL r =
            solver_type == LinearSolverTypes::CG
                ? IterativeMatrixFunctions<FL>::conjugate_gradient(
                      f, aa, mbra, mket, nmult, (FL)const_e, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter)
                : (solver_type == LinearSolverTypes::MinRes
                       ? IterativeMatrixFunctions<FL>::minres(
                             f, mbra, mket, nmult, (FL)const_e, iprint,
                             para_rule == nullptr ? nullptr : para_rule->comm,
                             conv_thrd, max_iter, soft_max_iter)
                       : IterativeMatrixFunctions<FL>::gcrotmk(
                             f, aa, mbra, mket, nmult, niter,
                             linear_solver_params.first,
                             linear_solver_params.second, (FL)const_e, iprint,
                             para_rule == nullptr ? nullptr : para_rule->comm,
                             conv_thrd, max_iter, soft_max_iter));
        if (compute_diag && solver_type != LinearSolverTypes::MinRes)
            aa.deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(r, make_pair(nmult, niter), (size_t)nflop,
                          t.get_time());
    }
    shared_ptr<OpExpr<S>>
    add_const_term(typename const_fl_type<FL>::FL const_e,
                   const shared_ptr<ParallelRule<S>> &para_rule) {
        shared_ptr<OpExpr<S>> expr = op->mat->data[0];
        if ((FL)const_e != (FP)0.0) {
            // q_label does not matter
            shared_ptr<OpExpr<S>> iop = make_shared<OpElement<S, FL>>(
                OpNames::I, SiteIndex(),
                dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->q_label);
            if (hop_left_vacuum !=
                dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->q_label)
                throw runtime_error(
                    "non-singlet MPO cannot have constant term!");
            if (para_rule == nullptr || para_rule->is_root()) {
                if (op->lopt->get_type() == OperatorTensorTypes::Delayed ||
                    op->ropt->get_type() == OperatorTensorTypes::Delayed) {
                    bool dleft =
                        op->lopt->get_type() == OperatorTensorTypes::Delayed;
                    shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                        dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(
                            dleft ? op->lopt : op->ropt);
                    shared_ptr<OpElement<S, FL>> xiop =
                        dynamic_pointer_cast<OpElement<S, FL>>(iop);
                    if (dopt->lopt->ops.count(iop) != 0 &&
                        dopt->ropt->ops.count(iop) != 0)
                        op->mat->data[0] =
                            expr +
                            (shared_ptr<OpExpr<S>>)
                                make_shared<OpSumProd<S, FL>>(
                                    xiop, xiop,
                                    vector<shared_ptr<OpElement<S, FL>>>{xiop,
                                                                         xiop},
                                    vector<bool>{false, false}, (FL)const_e, 0);
                    else
                        op->mat->data[0] = expr + (FL)const_e * (iop * iop);
                } else
                    op->mat->data[0] = expr + (FL)const_e * (iop * iop);
            }
        }
        return expr;
    }
    // [bra] = [H_eff] x [ket]
    // norm, nmult, nflop, tmult
    tuple<FP, int, size_t, double>
    multiply(typename const_fl_type<FL>::FL const_e,
             const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        bra->clear();
        shared_ptr<OpExpr<S>> expr = add_const_term(const_e, para_rule);
        Timer t;
        t.get_time();
        // Auto mode cannot add const_e term
        SeqTypes mode = tf->opf->seq->mode;
        tf->opf->seq->mode = tf->opf->seq->mode & SeqTypes::Simple
                                 ? SeqTypes::Simple
                                 : SeqTypes::None;
        tf->opf->seq->cumulative_nflop = 0;
        (*this)(GMatrix<FL>(ket->data, (MKL_INT)ket->total_memory, 1),
                GMatrix<FL>(bra->data, (MKL_INT)bra->total_memory, 1));
        op->mat->data[0] = expr;
        FP norm = GMatrixFunctions<FL>::norm(
            GMatrix<FL>(bra->data, (MKL_INT)bra->total_memory, 1));
        tf->opf->seq->mode = mode;
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(norm, 1, (size_t)nflop, t.get_time());
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tmult
    // fuse_left: 1 : must fuse left, 0: must fuse right, -1: arbitrary
    tuple<vector<pair<shared_ptr<OpExpr<S>>, FL>>, size_t, double>
    expect(typename const_fl_type<FL>::FL const_e,
           ExpectationAlgorithmTypes algo_type, ExpectationTypes ex_type,
           const shared_ptr<ParallelRule<S>> &para_rule = nullptr,
           uint8_t fuse_left = -1) {
        shared_ptr<OpExpr<S>> expr = nullptr;
        if ((FL)const_e != (FL)0.0 && op->mat->data.size() > 0)
            expr = add_const_term(const_e, para_rule);
        assert(ex_type == ExpectationTypes::Real || is_complex<FL>::value);
        if (algo_type == ExpectationAlgorithmTypes::Automatic) {
            algo_type = op->mat->data.size() > 1
                            ? ExpectationAlgorithmTypes::Fast
                            : ExpectationAlgorithmTypes::Normal;
            if (npdm_scheme != nullptr && op->mat->data.size() == 1 &&
                dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->name ==
                    OpNames::XPDM &&
                dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])
                        ->site_index == SiteIndex())
                algo_type = ExpectationAlgorithmTypes::SymbolFree |
                            ExpectationAlgorithmTypes::Compressed;
        }
        SeqTypes mode = tf->opf->seq->mode;
        tf->opf->seq->mode = tf->opf->seq->mode & SeqTypes::Simple
                                 ? SeqTypes::Simple
                                 : SeqTypes::None;
        tf->opf->seq->cumulative_nflop = 0;
        Timer t;
        t.get_time();
        vector<pair<shared_ptr<OpExpr<S>>, FL>> expectations;
        // may happen for NPDM with ancilla
        if (op->mat->data.size() == 1 &&
            dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->name ==
                OpNames::Zero)
            ;
        else if (algo_type == ExpectationAlgorithmTypes::Normal) {
            GMatrix<FL> ktmp(ket->data, (MKL_INT)ket->total_memory, 1);
            GMatrix<FL> rtmp(bra->data, (MKL_INT)bra->total_memory, 1);
            GMatrix<FL> btmp(nullptr, (MKL_INT)bra->total_memory, 1);
            btmp.allocate();
            expectations.reserve(op->mat->data.size());
            vector<FL> results;
            vector<size_t> results_idx;
            if (para_rule != nullptr && npdm_scheme == nullptr) {
                results.reserve(op->mat->data.size());
                results_idx.reserve(op->mat->data.size());
                para_rule->set_partition(ParallelRulePartitionTypes::Middle);
            }
            for (size_t i = 0; i < op->mat->data.size(); i++) {
                using OESF = OpElement<S, FL>;
                assert(dynamic_pointer_cast<OESF>(op->dops[i])->name !=
                       OpNames::Zero);
                S idx_opdq = dynamic_pointer_cast<OpElement<S, FL>>(op->dops[i])
                                 ->q_label;
                S ket_dq = ket->info->delta_quantum;
                S bra_dq = bra->info->delta_quantum;
                if (idx_opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                    expectations.push_back(make_pair(op->dops[i], 0.0));
                else {
                    FL r = 0.0;
                    if (para_rule == nullptr || npdm_scheme != nullptr ||
                        !dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)
                             ->number(op->dops[i])) {
                        btmp.clear();
                        (*this)(ktmp, btmp, (int)i, 1.0, npdm_scheme == nullptr);
                        r = GMatrixFunctions<FL>::complex_dot(rtmp, btmp);
                    } else {
                        if (dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)
                                ->own(op->dops[i])) {
                            btmp.clear();
                            (*this)(ktmp, btmp, (int)i, 1.0, false);
                            r = GMatrixFunctions<FL>::complex_dot(rtmp, btmp);
                        }
                        results.push_back(r);
                        results_idx.push_back(expectations.size());
                    }
                    expectations.push_back(make_pair(op->dops[i], r));
                }
            }
            btmp.deallocate();
            if (results.size() != 0) {
                assert(para_rule != nullptr);
                para_rule->comm->allreduce_sum(results.data(), results.size());
                for (size_t i = 0; i < results.size(); i++)
                    expectations[results_idx[i]].second = results[i];
            }
        } else if (algo_type == ExpectationAlgorithmTypes::Fast) {
            expectations = tf->tensor_product_expectation(
                op->dops, op->mat->data, op->lopt, op->ropt, ket, bra,
                npdm_scheme == nullptr);
        } else if (algo_type & ExpectationAlgorithmTypes::SymbolFree) {
            if (npdm_scheme == nullptr)
                throw runtime_error("ExpectationAlgorithmTypes::SymbolFree "
                                    "only works with general NPDM MPO.");
            expectations = tf->tensor_product_npdm_fragment(
                npdm_scheme, opdq, npdm_fragment_filename, npdm_n_sites,
                npdm_center, npdm_parallel_center, op->lopt, op->ropt, ket, bra,
                fuse_left == -1 ? op->lopt->ops.size() < op->ropt->ops.size()
                                : fuse_left,
                algo_type & ExpectationAlgorithmTypes::Compressed,
                algo_type & ExpectationAlgorithmTypes::LowMem);
        }
        if ((FL)const_e != (FL)0.0 && op->mat->data.size() > 0)
            op->mat->data[0] = expr;
        tf->opf->seq->mode = mode;
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, (size_t)nflop, t.get_time());
    }
    // return |ket> and beta [H_eff] |ket>
    pair<vector<shared_ptr<SparseMatrix<S, FL>>>, tuple<int, size_t, double>>
    first_rk4_apply(FL beta, typename const_fl_type<FL>::FL const_e,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<shared_ptr<SparseMatrix<S, FL>>> r(2);
        for (int i = 0; i < 2; i++) {
            r[i] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            r[i]->allocate(bra->info);
        }
        GMatrix<FL> kk(ket->data, (MKL_INT)ket->total_memory, 1);
        GMatrix<FL> r0(r[0]->data, (MKL_INT)bra->total_memory, 1);
        GMatrix<FL> r1(r[1]->data, (MKL_INT)bra->total_memory, 1);
        Timer t;
        t.get_time();
        assert(op->mat->data.size() > 0);
        precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &, FL)> &f =
            [this](const GMatrix<FL> &a, const GMatrix<FL> &b, FL scale) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b, scale);
                else
                    return (*this)(a, b, 0, scale);
            };
        tf->opf->seq->cumulative_nflop = 0;
        f(kk, r1, beta);
        shared_ptr<OpExpr<S>> expr = op->mat->data[0];
        op->mat->data[0] = make_shared<OpExpr<S>>();
        add_const_term(1.0, para_rule);
        f(kk, r0, 1.0);
        op->mat->data[0] = expr;
        // if (const_e != (FL)0.0)
        //     MatrixFunctions::iadd(r1, r0, beta * const_e);
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(r, make_tuple(1, (size_t)nflop, t.get_time()));
    }
    pair<vector<shared_ptr<SparseMatrix<S, FL>>>,
         tuple<FL, FP, int, size_t, double>>
    second_rk4_apply(FL beta, typename const_fl_type<FL>::FL const_e,
                     const shared_ptr<SparseMatrix<S, FL>> &hket,
                     bool eval_energy = false,
                     const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<shared_ptr<SparseMatrix<S, FL>>> rr(3), kk(4);
        kk[0] = hket;
        for (int i = 0; i < 3; i++) {
            rr[i] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            rr[i]->allocate(ket->info);
        }
        for (int i = 0; i < 3; i++) {
            kk[i + 1] = make_shared<SparseMatrix<S, FL>>(d_alloc);
            kk[i + 1]->allocate(ket->info);
        }
        GMatrix<FL> v(ket->data, (MKL_INT)ket->total_memory, 1);
        vector<GMatrix<FL>> k(4, v), r(3, v);
        Timer t;
        t.get_time();
        for (int i = 0; i < 3; i++)
            r[i] = GMatrix<FL>(rr[i]->data, (MKL_INT)ket->total_memory, 1);
        for (int i = 0; i < 4; i++)
            k[i] = GMatrix<FL>(kk[i]->data, (MKL_INT)ket->total_memory, 1);
        tf->opf->seq->cumulative_nflop = 0;
        const vector<FP> ks = vector<FP>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<FP>> cs = vector<vector<FP>>{
            vector<FP>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0, -5.0 / 162.0},
            vector<FP>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<FP>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &, FL)> &f =
            [this](const GMatrix<FL> &a, const GMatrix<FL> &b, FL scale) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b, scale);
                else
                    return (*this)(a, b, 0, scale);
            };
        // k1 ~ k3
        for (int i = 1; i < 4; i++) {
            GMatrixFunctions<FL>::copy(r[0], v);
            GMatrixFunctions<FL>::iadd(r[0], k[i - 1], ks[i]);
            f(r[0], k[i], beta);
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            FL factor = exp(beta * (FL)(i + 1.0) / (FL)3.0 * (FL)const_e);
            GMatrixFunctions<FL>::copy(r[i], v);
            GMatrixFunctions<FL>::iscale(r[i], factor);
            for (size_t j = 0; j < 4; j++)
                GMatrixFunctions<FL>::iadd(r[i], k[j], cs[i][j] * factor);
        }
        FP norm = GMatrixFunctions<FL>::norm(r[2]);
        FL energy = -(FL)const_e;
        if (eval_energy) {
            k[0].clear();
            f(r[2], k[0], 1.0);
            energy =
                GMatrixFunctions<FL>::complex_dot(r[2], k[0]) / (norm * norm);
        }
        for (int i = 3; i >= 1; i--)
            kk[i]->deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(rr, make_tuple(energy, norm, 3 + eval_energy,
                                        (size_t)nflop, t.get_time()));
    }
    // [ket] = exp( [H_eff] ) | [ket] > (RK4 approximation)
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<GMatrix<FL>>, tuple<FL, FP, int, size_t, double>>
    rk4_apply(FL beta, typename const_fl_type<FL>::FL const_e,
              bool eval_energy = false,
              const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        GMatrix<FL> v(ket->data, (MKL_INT)ket->total_memory, 1);
        vector<GMatrix<FL>> k, r;
        Timer t;
        t.get_time();
        frame_<FP>()->activate(1);
        for (int i = 0; i < 3; i++) {
            r.push_back(GMatrix<FL>(nullptr, (MKL_INT)ket->total_memory, 1));
            r[i].allocate();
        }
        frame_<FP>()->activate(0);
        for (int i = 0; i < 4; i++) {
            k.push_back(GMatrix<FL>(nullptr, (MKL_INT)ket->total_memory, 1));
            k[i].allocate(), k[i].clear();
        }
        tf->opf->seq->cumulative_nflop = 0;
        const vector<FP> ks = vector<FP>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<FP>> cs = vector<vector<FP>>{
            vector<FP>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0, -5.0 / 162.0},
            vector<FP>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<FP>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &, FL)> &f =
            [this](const GMatrix<FL> &a, const GMatrix<FL> &b, FL scale) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b, scale);
                else
                    return (*this)(a, b, 0, scale);
            };
        // k0 ~ k3
        for (int i = 0; i < 4; i++) {
            if (i == 0)
                f(v, k[i], beta);
            else {
                GMatrixFunctions<FL>::copy(r[0], v);
                GMatrixFunctions<FL>::iadd(r[0], k[i - 1], ks[i]);
                f(r[0], k[i], beta);
            }
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            FL factor = exp(beta * (FL)(i + 1.0) / (FL)3.0 * (FL)const_e);
            GMatrixFunctions<FL>::copy(r[i], v);
            GMatrixFunctions<FL>::iscale(r[i], factor);
            for (size_t j = 0; j < 4; j++)
                GMatrixFunctions<FL>::iadd(r[i], k[j], cs[i][j] * factor);
        }
        FP norm = GMatrixFunctions<FL>::norm(r[2]);
        FL energy = -(FL)const_e;
        if (eval_energy) {
            k[0].clear();
            f(r[2], k[0], 1.0);
            energy =
                GMatrixFunctions<FL>::complex_dot(r[2], k[0]) / (norm * norm);
        }
        for (int i = 3; i >= 0; i--)
            k[i].deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(r, make_tuple(energy, norm, 4 + eval_energy,
                                       (size_t)nflop, t.get_time()));
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    tuple<FL, FP, int, size_t, double>
    expo_apply(FL beta, typename const_fl_type<FL>::FL const_e, bool symmetric,
               bool iprint = false,
               const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(compute_diag);
        FP anorm = GMatrixFunctions<FL>::norm(
            GMatrix<FL>(diag->data, (MKL_INT)diag->total_memory, 1));
        GMatrix<FL> v(ket->data, (MKL_INT)ket->total_memory, 1);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        int nexpo =
            (tf->opf->seq->mode == SeqTypes::Auto ||
             (tf->opf->seq->mode & SeqTypes::Tasked))
                ? IterativeMatrixFunctions<FL>::expo_apply(
                      *tf, beta, anorm, v, (FL)const_e, symmetric, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm)
                : IterativeMatrixFunctions<FL>::expo_apply(
                      *this, beta, anorm, v, (FL)const_e, symmetric, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm);
        FP norm = GMatrixFunctions<FL>::norm(v);
        GMatrix<FL> tmp(nullptr, (MKL_INT)ket->total_memory, 1);
        tmp.allocate();
        tmp.clear();
        if (tf->opf->seq->mode == SeqTypes::Auto ||
            (tf->opf->seq->mode & SeqTypes::Tasked))
            (*tf)(v, tmp);
        else
            (*this)(v, tmp);
        FL energy = GMatrixFunctions<FL>::complex_dot(v, tmp) / (norm * norm);
        tmp.deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, (size_t)nflop, t.get_time());
    }
    void deallocate() {
        frame_<FP>()->activate(0);
        for (int i = (int)wfn_infos.size() - 1; i >= 0; i--)
            if (wfn_infos[i] != nullptr)
                wfn_infos[i]->deallocate();
        if (compute_diag)
            diag->deallocate();
        op->deallocate();
        vector<pair<S *, shared_ptr<SparseMatrixInfo<S>>>> mp;
        mp.reserve(left_op_infos.size() + right_op_infos.size());
        for (int i = (int)right_op_infos.size() - 1; i >= 0; i--)
            mp.emplace_back(right_op_infos[i].second->quanta,
                            right_op_infos[i].second);
        for (int i = (int)left_op_infos.size() - 1; i >= 0; i--)
            mp.emplace_back(left_op_infos[i].second->quanta,
                            left_op_infos[i].second);
        sort(mp.begin(), mp.end(),
             [](const pair<S *, shared_ptr<SparseMatrixInfo<S>>> &a,
                const pair<S *, shared_ptr<SparseMatrixInfo<S>>> &b) {
                 return a.first > b.first;
             });
        for (const auto &t : mp) {
            if (t.second->cinfo != nullptr)
                t.second->cinfo->deallocate();
            t.second->deallocate();
        }
    }
};

// Linear combination of Effective Hamiltonians
template <typename S, typename FL> struct LinearEffectiveHamiltonian {
    typedef S ST;
    typedef FL FLT;
    typedef typename GMatrix<FL>::FP FP;
    vector<shared_ptr<EffectiveHamiltonian<S, FL>>> h_effs;
    vector<FL> coeffs;
    S opdq;
    LinearEffectiveHamiltonian(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff)
        : h_effs{h_eff}, coeffs{1} {}
    LinearEffectiveHamiltonian(
        const vector<shared_ptr<EffectiveHamiltonian<S, FL>>> &h_effs,
        const vector<FL> &coeffs)
        : h_effs(h_effs), coeffs(coeffs) {}
    static shared_ptr<LinearEffectiveHamiltonian<S, FL>>
    linearize(const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &x) {
        return x;
    }
    static shared_ptr<LinearEffectiveHamiltonian<S, FL>>
    linearize(const shared_ptr<EffectiveHamiltonian<S, FL>> &x) {
        return make_shared<LinearEffectiveHamiltonian<S, FL>>(x);
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const GMatrix<FL> &b, const GMatrix<FL> &c) {
        for (size_t ih = 0; ih < h_effs.size(); ih++)
            if (h_effs[ih]->tf->opf->seq->mode == SeqTypes::Auto ||
                (h_effs[ih]->tf->opf->seq->mode & SeqTypes::Tasked))
                h_effs[ih]->tf->operator()(b, c, coeffs[ih]);
            else
                h_effs[ih]->operator()(b, c, 0, coeffs[ih]);
    }
    size_t get_op_total_memory() const {
        size_t r = 0;
        for (size_t ih = 0; ih < h_effs.size(); ih++)
            r += h_effs[ih]->op->get_total_memory();
        return r;
    }
    // Find eigenvalues and eigenvectors of [H_eff]
    // energy, ndav, nflop, tdav
    tuple<typename const_fl_type<FP>::FL, int, size_t, double>
    eigs(bool iprint = false, FP conv_thrd = 5E-6, int max_iter = 5000,
         int soft_max_iter = -1,
         DavidsonTypes davidson_type = DavidsonTypes::Normal, FP shift = 0,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int ndav = 0;
        assert(h_effs.size() != 0);
        const shared_ptr<TensorFunctions<S, FL>> &tf = h_effs[0]->tf;
        GDiagonalMatrix<FL> aa(nullptr, (MKL_INT)h_effs[0]->diag->total_memory);
        aa.allocate();
        aa.clear();
        for (size_t ih = 0; ih < h_effs.size(); ih++) {
            assert(h_effs[ih]->compute_diag);
            GMatrixFunctions<FL>::iadd(
                GMatrix<FL>(aa.data, (MKL_INT)aa.size(), 1),
                GMatrix<FL>(h_effs[ih]->diag->data,
                            (MKL_INT)h_effs[ih]->diag->total_memory, 1),
                coeffs[ih]);
            h_effs[ih]->precompute();
        }
        vector<GMatrix<FL>> bs = vector<GMatrix<FL>>{GMatrix<FL>(
            h_effs[0]->ket->data, (MKL_INT)h_effs[0]->ket->total_memory, 1)};
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        vector<FP> eners = IterativeMatrixFunctions<FL>::harmonic_davidson(
            *this, aa, bs, shift, davidson_type, ndav, iprint,
            para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
            max_iter, soft_max_iter);
        for (size_t ih = 0; ih < h_effs.size(); ih++)
            h_effs[ih]->post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        aa.deallocate();
        return make_tuple((typename const_fl_type<FP>::FL)eners[0], ndav,
                          (size_t)nflop, t.get_time());
    }
    void deallocate() {}
};

template <typename T>
inline shared_ptr<LinearEffectiveHamiltonian<typename T::ST, typename T::FLT>>
operator*(typename T::FLT d, const shared_ptr<T> &x) {
    shared_ptr<LinearEffectiveHamiltonian<typename T::ST, typename T::FLT>> xx =
        LinearEffectiveHamiltonian<typename T::ST, typename T::FLT>::linearize(
            x);
    vector<typename T::FLT> new_coeffs;
    for (auto &c : xx->coeffs)
        new_coeffs.push_back(c * d);
    return make_shared<
        LinearEffectiveHamiltonian<typename T::ST, typename T::FLT>>(
        xx->h_effs, new_coeffs);
}

template <typename T>
inline shared_ptr<LinearEffectiveHamiltonian<typename T::ST, typename T::FLT>>
operator*(const shared_ptr<T> &x, typename T::FLT d) {
    return d * x;
}

template <typename T>
inline shared_ptr<LinearEffectiveHamiltonian<typename T::ST, typename T::FLT>>
operator-(const shared_ptr<T> &x) {
    return (-1.0) * x;
}

template <typename T1, typename T2>
inline shared_ptr<LinearEffectiveHamiltonian<typename T1::ST, typename T1::FLT>>
operator+(const shared_ptr<T1> &x, const shared_ptr<T2> &y) {
    shared_ptr<LinearEffectiveHamiltonian<typename T1::ST, typename T1::FLT>>
        xx = LinearEffectiveHamiltonian<typename T1::ST,
                                        typename T1::FLT>::linearize(x);
    shared_ptr<LinearEffectiveHamiltonian<typename T1::ST, typename T1::FLT>>
        yy = LinearEffectiveHamiltonian<typename T1::ST,
                                        typename T1::FLT>::linearize(y);
    vector<shared_ptr<EffectiveHamiltonian<typename T1::ST, typename T1::FLT>>>
        h_effs = xx->h_effs;
    vector<typename T1::FLT> coeffs = xx->coeffs;
    h_effs.insert(h_effs.end(), yy->h_effs.begin(), yy->h_effs.end());
    coeffs.insert(coeffs.end(), yy->coeffs.begin(), yy->coeffs.end());
    return make_shared<
        LinearEffectiveHamiltonian<typename T1::ST, typename T1::FLT>>(h_effs,
                                                                       coeffs);
}

template <typename T1, typename T2>
inline shared_ptr<LinearEffectiveHamiltonian<typename T1::ST, typename T1::FLT>>
operator-(const shared_ptr<T1> &x, const shared_ptr<T2> &y) {
    return x + (-1.0) * y;
}

// Effective Hamiltonian for MultiMPS
template <typename S, typename FL>
struct EffectiveHamiltonian<S, FL, MultiMPS<S, FL>> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FC FC;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
        right_op_infos;
    // Symbolic expression of effective H
    shared_ptr<DelayedOperatorTensor<S, FL>> op;
    shared_ptr<SparseMatrixGroup<S, FL>> diag;
    vector<shared_ptr<SparseMatrixGroup<S, FL>>> bra, ket;
    shared_ptr<SparseMatrixGroup<S, FL>> cmat, vmat;
    shared_ptr<TensorFunctions<S, FL>> tf;
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    // Delta quantum of effective H
    S opdq;
    // Left vacuum of MPO
    S hop_left_vacuum;
    // Whether diagonal element of effective H should be computed
    bool compute_diag;
    vector<unordered_map<
        S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
        wfn_infos;
    vector<S> operator_quanta;
    shared_ptr<NPDMScheme> npdm_scheme = nullptr;
    string npdm_fragment_filename = "";
    int npdm_n_sites = 0, npdm_center = -1, npdm_parallel_center = -1;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S, FL>> &op,
        const vector<shared_ptr<SparseMatrixGroup<S, FL>>> &bra,
        const vector<shared_ptr<SparseMatrixGroup<S, FL>>> &ket,
        const shared_ptr<OpElement<S, FL>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat, S hop_left_vacuum,
        const shared_ptr<TensorFunctions<S, FL>> &ptf, bool compute_diag = true,
        const shared_ptr<NPDMScheme> &npdm_scheme = nullptr)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(ptf->copy()), hop_mat(hop_mat),
          hop_left_vacuum(hop_left_vacuum), compute_diag(compute_diag),
          npdm_scheme(npdm_scheme) {
        // wavefunction
        if (compute_diag) {
            // for non-hermitian hamiltonian, bra and ket may share the same
            // info but they are different objects
            assert(bra.size() == ket.size());
            for (size_t i = 0; i < bra.size(); i++) {
                assert(bra[i]->infos.size() == ket[i]->infos.size());
                for (size_t j = 0; j < bra[i]->infos.size(); j++)
                    assert(bra[i]->infos[j]->n == ket[i]->infos[j]->n);
            }
            diag = make_shared<SparseMatrixGroup<S, FL>>();
            diag->allocate(ket[0]->infos);
        }
        // unique sub labels
        opdq = hop->q_label;
        vector<S> msl = Partition<S, FL>::get_uniq_labels({hop_mat});
        operator_quanta = msl;
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S, FL>::get_uniq_sub_labels(op->mat, hop_mat, msl,
                                                  hop_left_vacuum);
        // tensor product diagonal
        if (compute_diag) {
            for (int i = 0; i < diag->n; i++) {
                shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>
                    diag_info = make_shared<
                        typename SparseMatrixInfo<S>::ConnectionInfo>();
                diag_info->initialize_diag(
                    ket[0]->infos[i]->delta_quantum, opdq, msubsl[0],
                    left_op_infos, right_op_infos, diag->infos[i], tf->opf->cg);
                diag->infos[i]->cinfo = diag_info;
                shared_ptr<SparseMatrix<S, FL>> xdiag = (*diag)[i];
                tf->tensor_product_diagonal(op->mat->data[0], op->lopt,
                                            op->ropt, xdiag, opdq);
                diag_info->deallocate();
            }
        }
        // temp wavefunction
        cmat = make_shared<SparseMatrixGroup<S, FL>>();
        vmat = make_shared<SparseMatrixGroup<S, FL>>();
        *cmat = *ket[0];
        *vmat = *bra[0];
        // temp wavefunction info
        wfn_infos.resize(msl.size());
        for (int i = 0; i < (int)msl.size(); i++)
            for (int ic = 0; ic < cmat->n; ic++)
                for (int iv = 0; iv < vmat->n; iv++) {
                    S cdq = cmat->infos[ic]->delta_quantum;
                    S vdq = vmat->infos[iv]->delta_quantum;
                    S cvdq = msl[i].combine(vdq, cdq);
                    if (cvdq == S(S::invalid) || wfn_infos[i].count(cvdq))
                        continue;
                    shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>
                        wfn_info = make_shared<
                            typename SparseMatrixInfo<S>::ConnectionInfo>();
                    wfn_info->initialize_wfn(cdq, vdq, msl[i], msubsl[i],
                                             left_op_infos, right_op_infos,
                                             cmat->infos[ic], vmat->infos[iv],
                                             tf->opf->cg);
                    wfn_infos[i][cvdq] = wfn_info;
                }
        for (int i = 0; i < cmat->n; i++) {
            S cdq = cmat->infos[i]->delta_quantum;
            S vdq = vmat->infos[i]->delta_quantum;
            S cvdq = opdq.combine(vdq, cdq);
            if (cvdq != S(S::invalid))
                cmat->infos[i]->cinfo = wfn_infos[0][cvdq];
        }
    }
    // prepare batch gemm
    void precompute() const {
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            cmat->data = vmat->data = (FL *)0;
            tf->tensor_product_multi_multiply(op->mat->data[0], op->lopt,
                                              op->ropt, cmat, vmat,
                                              wfn_infos[0], opdq, 1.0, false);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        } else if (tf->opf->seq->mode & SeqTypes::Tasked) {
            cmat->data = vmat->data = (FL *)0;
            tf->tensor_product_multi_multiply(op->mat->data[0], op->lopt,
                                              op->ropt, cmat, vmat,
                                              wfn_infos[0], opdq, 1.0, false);
        }
    }
    void post_precompute() const {
        if (tf->opf->seq->mode == SeqTypes::Auto ||
            (tf->opf->seq->mode & SeqTypes::Tasked)) {
            tf->opf->seq->deallocate();
            tf->opf->seq->clear();
        }
    }
    shared_ptr<SparseMatrixGroup<S, FL>>
    perturbative_noise(bool trace_right, int iL, int iR, FuseTypes ftype,
                       const shared_ptr<MPSInfo<S>> &mps_info,
                       const vector<FP> &weights, const NoiseTypes noise_type,
                       const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(mps_info->get_type() & MPSTypes::MultiWfn);
        shared_ptr<MultiMPSInfo<S>> minfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(mps_info);
        vector<S> msl = Partition<S, FL>::get_uniq_labels({hop_mat});
        assert(msl.size() == 1 && msl[0] == opdq);
        shared_ptr<OpExpr<S>> pexpr = op->mat->data[0];
        shared_ptr<Symbolic<S>> pmat = make_shared<SymbolicColumnVector<S>>(
            1, vector<shared_ptr<OpExpr<S>>>{pexpr});
        vector<pair<uint8_t, S>> psubsl = Partition<S, FL>::get_uniq_sub_labels(
            pmat, hop_mat, msl, hop_left_vacuum, true, trace_right, false)[0];
        vector<S> perturb_ket_labels, all_perturb_ket_labels;
        for (int i = 0; i < ket[0]->n; i++) {
            S ket_label = ket[0]->infos[i]->delta_quantum;
            for (size_t j = 0; j < psubsl.size(); j++) {
                S pks = ket_label + psubsl[j].second;
                for (int k = 0; k < pks.count(); k++)
                    perturb_ket_labels.push_back(pks[k]);
            }
        }
        sort(psubsl.begin(), psubsl.end());
        psubsl.resize(
            distance(psubsl.begin(), unique(psubsl.begin(), psubsl.end())));
        all_perturb_ket_labels = perturb_ket_labels;
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
        minfo->load_left_dims(iL);
        minfo->load_right_dims(iR + 1);
        StateInfo<S> l = *minfo->left_dims[iL], ml = *minfo->basis[iL],
                     mr = *minfo->basis[iR], r = *minfo->right_dims[iR + 1];
        StateInfo<S> ll = (ftype & FuseTypes::FuseL)
                              ? StateInfo<S>::tensor_product(
                                    l, ml, *minfo->left_dims_fci[iL + 1])
                              : l;
        StateInfo<S> rr = (ftype & FuseTypes::FuseR)
                              ? StateInfo<S>::tensor_product(
                                    mr, r, *minfo->right_dims_fci[iR])
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
        shared_ptr<SparseMatrixGroup<S, FL>> perturb_ket =
            make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
        assert(noise_type & NoiseTypes::Perturbative);
        bool do_reduce = !(noise_type & NoiseTypes::Collected);
        bool reduced = noise_type & NoiseTypes::Reduced;
        bool low_mem = noise_type & NoiseTypes::LowMem;
        if (reduced)
            perturb_ket->allocate(infos);
        else {
            vector<shared_ptr<SparseMatrixInfo<S>>> all_infos;
            all_infos.reserve(all_perturb_ket_labels.size());
            for (S q : all_perturb_ket_labels) {
                size_t ib = lower_bound(perturb_ket_labels.begin(),
                                        perturb_ket_labels.end(), q) -
                            perturb_ket_labels.begin();
                all_infos.push_back(infos[ib]);
            }
            perturb_ket->allocate(all_infos);
        }
        // connection infos
        frame_<FP>()->activate(0);
        vector<vector<
            vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>>
            cinfos;
        cinfos.resize(ket[0]->n);
        int vidx = reduced ? -1 : 0;
        for (int i = 0; i < ket[0]->n; i++) {
            cinfos[i].resize(psubsl.size());
            S idq = S(0);
            S ket_label = ket[0]->infos[i]->delta_quantum;
            for (size_t j = 0; j < psubsl.size(); j++) {
                S pks = ket_label + psubsl[j].second;
                cinfos[i][j].resize(pks.count());
                for (int k = 0; k < pks.count(); k++) {
                    cinfos[i][j][k] = make_shared<
                        typename SparseMatrixInfo<S>::ConnectionInfo>();
                    size_t ib = lower_bound(perturb_ket_labels.begin(),
                                            perturb_ket_labels.end(), pks[k]) -
                                perturb_ket_labels.begin();
                    S opdq = psubsl[j].second;
                    vector<pair<uint8_t, S>> subdq = {
                        trace_right ? make_pair(psubsl[j].first,
                                                opdq.combine(opdq, -idq))
                                    : make_pair((uint8_t)(psubsl[j].first << 1),
                                                opdq.combine(idq, -opdq))};
                    cinfos[i][j][k]->initialize_wfn(
                        ket_label, pks[k], psubsl[j].second, subdq,
                        left_op_infos, right_op_infos, ket[0]->infos[i],
                        infos[ib], tf->opf->cg);
                    assert(cinfos[i][j][k]->n[4] == 1);
                }
            }
            // perform multiplication
            for (int ii = 0, pvidx = vidx; ii < (int)ket.size(); ii++) {
                vidx = pvidx;
                FP ket_norm = (*ket[ii])[i]->norm();
                if (abs(ket_norm) > TINY)
                    tf->tensor_product_partial_multiply(
                        (weights[ii] / ket_norm) * pexpr, op->lopt, op->ropt,
                        trace_right, (*ket[ii])[i], psubsl, cinfos[i],
                        perturb_ket_labels, perturb_ket, vidx,
                        low_mem ? -2 : -1, do_reduce);
            }
        }
        if (!reduced)
            assert(vidx == perturb_ket->n);
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            tf->opf->seq->auto_perform();
            if (para_rule != nullptr && do_reduce)
                para_rule->comm->reduce_sum(perturb_ket, para_rule->comm->root);
        } else if (tf->opf->seq->mode & SeqTypes::Tasked) {
            if (!low_mem) {
                assert(perturb_ket->total_memory <=
                       (size_t)numeric_limits<decltype(GMatrix<FL>::n)>::max());
                tf->opf->seq->auto_perform(GMatrix<FL>(
                    perturb_ket->data, (MKL_INT)perturb_ket->total_memory, 1));
            } else {
                vector<GMatrix<FL>> pmats(perturb_ket->n,
                                          GMatrix<FL>(nullptr, 0, 0));
                for (int j = 0; j < perturb_ket->n; j++)
                    pmats[j] = GMatrix<FL>(
                        (*perturb_ket)[j]->data,
                        (MKL_INT)(*perturb_ket)[j]->total_memory, 1);
                tf->opf->seq->auto_perform(pmats);
            }
            if (para_rule != nullptr && do_reduce)
                para_rule->comm->reduce_sum(perturb_ket, para_rule->comm->root);
        }
        for (int i = (int)cinfos.size() - 1; i >= 0; i--)
            for (int j = (int)cinfos[i].size() - 1; j >= 0; j--)
                for (int k = (int)cinfos[i][j].size() - 1; k >= 0; k--)
                    cinfos[i][j][k]->deallocate();
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
                 dynamic_pointer_cast<OpSum<S, FL>>(op->mat->data[0])
                     ->strings) {
                if (opx->get_type() == OpTypes::Prod ||
                    opx->get_type() == OpTypes::Elem)
                    r++;
                else if (opx->get_type() == OpTypes::SumProd)
                    r += (int)dynamic_pointer_cast<OpSumProd<S, FL>>(opx)
                             ->ops.size();
            }
            return r;
        } else if (op->mat->data[0]->get_type() == OpTypes::SumProd)
            return (int)dynamic_pointer_cast<OpSumProd<S, FL>>(op->mat->data[0])
                ->ops.size();
        else
            return 1;
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const GMatrix<FL> &b, const GMatrix<FL> &c, int idx = 0,
                    FL factor = 1.0, bool all_reduce = true) {
        assert(b.m * b.n == cmat->total_memory);
        assert(c.m * c.n == vmat->total_memory);
        cmat->data = b.data;
        vmat->data = c.data;
        S idx_opdq =
            dynamic_pointer_cast<OpElement<S, FL>>(op->dops[idx])->q_label;
        size_t ic = lower_bound(operator_quanta.begin(), operator_quanta.end(),
                                idx_opdq) -
                    operator_quanta.begin();
        assert(ic < operator_quanta.size());
        tf->tensor_product_multi_multiply(op->mat->data[idx], op->lopt,
                                          op->ropt, cmat, vmat, wfn_infos[ic],
                                          idx_opdq, factor, all_reduce);
    }
    // Find eigenvalues and eigenvectors of [H_eff]
    // energies, ndav, nflop, tdav
    tuple<vector<typename const_fl_type<FP>::FL>, int, size_t, double>
    eigs(bool iprint = false, FP conv_thrd = 5E-6, int max_iter = 5000,
         int soft_max_iter = -1,
         DavidsonTypes davidson_type = DavidsonTypes::Normal, FP shift = 0,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr,
         const vector<shared_ptr<SparseMatrix<S, FL>>> &ortho_bra =
             vector<shared_ptr<SparseMatrix<S, FL>>>(),
         const vector<FP> &projection_weights = vector<FP>()) {
        int ndav = 0;
        assert(compute_diag);
        GDiagonalMatrix<FL> aa(diag->data, (MKL_INT)diag->total_memory);
        vector<GMatrix<FL>> bs;
        for (int i = 0; i < (int)min((MKL_INT)ket.size(), (MKL_INT)aa.n); i++)
            bs.push_back(
                GMatrix<FL>(ket[i]->data, (MKL_INT)ket[i]->total_memory, 1));
        if (davidson_type & DavidsonTypes::LeftEigen)
            for (int i = 0; i < (int)min((MKL_INT)bra.size(), (MKL_INT)aa.n);
                 i++)
                bs.push_back(GMatrix<FL>(bra[i]->data,
                                         (MKL_INT)bra[i]->total_memory, 1));
        vector<GMatrix<FL>> ors =
            vector<GMatrix<FL>>(ortho_bra.size(), GMatrix<FL>(nullptr, 0, 0));
        for (size_t i = 0; i < ortho_bra.size(); i++)
            ors[i] = GMatrix<FL>(ortho_bra[i]->data,
                                 (MKL_INT)ortho_bra[i]->total_memory, 1);
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        vector<FP> xeners =
            (tf->opf->seq->mode == SeqTypes::Auto ||
             (tf->opf->seq->mode & SeqTypes::Tasked))
                ? IterativeMatrixFunctions<FL>::harmonic_davidson(
                      *tf, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, 2, 50, ors,
                      projection_weights)
                : IterativeMatrixFunctions<FL>::harmonic_davidson(
                      *this, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, 2, 50, ors,
                      projection_weights);
        vector<typename const_fl_type<FP>::FL> eners(xeners.size());
        for (size_t i = 0; i < xeners.size(); i++)
            eners[i] = (typename const_fl_type<FP>::FL)xeners[i];
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners, ndav, (size_t)nflop, t.get_time());
    }
    shared_ptr<OpExpr<S>>
    add_const_term(typename const_fl_type<FL>::FL const_e,
                   const shared_ptr<ParallelRule<S>> &para_rule) {
        shared_ptr<OpExpr<S>> expr = op->mat->data[0];
        if ((FL)const_e != (FL)0.0) {
            // q_label does not matter
            shared_ptr<OpExpr<S>> iop = make_shared<OpElement<S, FL>>(
                OpNames::I, SiteIndex(),
                dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->q_label);
            if (hop_left_vacuum !=
                dynamic_pointer_cast<OpElement<S, FL>>(op->dops[0])->q_label)
                throw runtime_error(
                    "non-singlet MPO cannot have constant term!");
            if (para_rule == nullptr || para_rule->is_root()) {
                if (op->lopt->get_type() == OperatorTensorTypes::Delayed ||
                    op->ropt->get_type() == OperatorTensorTypes::Delayed) {
                    bool dleft =
                        op->lopt->get_type() == OperatorTensorTypes::Delayed;
                    shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                        dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(
                            dleft ? op->lopt : op->ropt);
                    shared_ptr<OpElement<S, FL>> xiop =
                        dynamic_pointer_cast<OpElement<S, FL>>(iop);
                    if (dopt->lopt->ops.count(iop) != 0 &&
                        dopt->ropt->ops.count(iop) != 0)
                        op->mat->data[0] =
                            expr +
                            (shared_ptr<OpExpr<S>>)
                                make_shared<OpSumProd<S, FL>>(
                                    xiop, xiop,
                                    vector<shared_ptr<OpElement<S, FL>>>{xiop,
                                                                         xiop},
                                    vector<bool>{false, false}, (FL)const_e, 0);
                    else
                        op->mat->data[0] = expr + (FL)const_e * (iop * iop);
                } else
                    op->mat->data[0] = expr + (FL)const_e * (iop * iop);
            }
        }
        return expr;
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tmult
    tuple<vector<pair<shared_ptr<OpExpr<S>>, vector<FL>>>, size_t, double>
    expect(typename const_fl_type<FL>::FL const_e,
           ExpectationAlgorithmTypes algo_type, ExpectationTypes ex_type,
           const shared_ptr<ParallelRule<S>> &para_rule = nullptr,
           uint8_t fuse_left = -1) {
        shared_ptr<OpExpr<S>> expr = nullptr;
        if ((FL)const_e != (FL)0.0 && op->mat->data.size() > 0)
            expr = add_const_term(const_e, para_rule);
        Timer t;
        t.get_time();
        GMatrix<FL> ktmp(nullptr, (MKL_INT)ket[0]->total_memory, 1);
        GMatrix<FL> rtmp(nullptr, (MKL_INT)bra[0]->total_memory, 1);
        GMatrix<FL> btmp(nullptr, (MKL_INT)bra[0]->total_memory, 1);
        btmp.allocate();
        SeqTypes mode = tf->opf->seq->mode;
        tf->opf->seq->mode = tf->opf->seq->mode & SeqTypes::Simple
                                 ? SeqTypes::Simple
                                 : SeqTypes::None;
        tf->opf->seq->cumulative_nflop = 0;
        vector<pair<shared_ptr<OpExpr<S>>, vector<FL>>> expectations;
        expectations.reserve(op->mat->data.size());
        vector<FL> results;
        vector<size_t> results_idx;
        results.reserve(op->mat->data.size() * ket.size());
        results_idx.reserve(op->mat->data.size());
        if (para_rule != nullptr)
            para_rule->set_partition(ParallelRulePartitionTypes::Middle);
        for (size_t i = 0; i < op->mat->data.size(); i++) {
            vector<FL> rr(ket.size(), 0);
            if (dynamic_pointer_cast<OpElement<S, FL>>(op->dops[i])->name ==
                OpNames::Zero)
                continue;
            else if (ex_type == ExpectationTypes::Real) {
                if (para_rule == nullptr ||
                    !dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)
                         ->number(op->dops[i])) {
                    for (int j = 0; j < (int)ket.size(); j++) {
                        ktmp.data = ket[j]->data;
                        rtmp.data = bra[j]->data;
                        btmp.clear();
                        (*this)(ktmp, btmp, (int)i, 1.0, true);
                        rr[j] = GMatrixFunctions<FL>::complex_dot(rtmp, btmp);
                    }
                } else {
                    if (dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)
                            ->own(op->dops[i])) {
                        for (int j = 0; j < (int)ket.size(); j++) {
                            ktmp.data = ket[j]->data;
                            rtmp.data = bra[j]->data;
                            btmp.clear();
                            (*this)(ktmp, btmp, (int)i, 1.0, false);
                            rr[j] =
                                GMatrixFunctions<FL>::complex_dot(rtmp, btmp);
                        }
                    }
                    results.insert(results.end(), rr.begin(), rr.end());
                    results_idx.push_back(expectations.size());
                }
                expectations.push_back(make_pair(op->dops[i], rr));
            } else if (ex_type == ExpectationTypes::Complex) {
                assert(ket.size() == 2 && bra.size() == 2);
                assert(ket[0]->total_memory == ket[1]->total_memory);
                assert(bra[0]->total_memory == bra[1]->total_memory);
                GMatrix<FL> itmp(nullptr, (MKL_INT)bra[1]->total_memory, 1);
                if (para_rule == nullptr ||
                    !dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)
                         ->number(op->dops[i]) ||
                    dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)->own(
                        op->dops[i])) {
                    rtmp.data = bra[0]->data;
                    itmp.data = bra[1]->data;
                    ktmp.data = ket[0]->data;
                    btmp.clear();
                    (*this)(ktmp, btmp, (int)i, 1.0, true);
                    rr[0] = GMatrixFunctions<FL>::complex_dot(rtmp, btmp);
                    rr[1] = -GMatrixFunctions<FL>::complex_dot(itmp, btmp);
                    ktmp.data = ket[1]->data;
                    btmp.clear();
                    (*this)(ktmp, btmp, (int)i, 1.0, true);
                    rr[1] += GMatrixFunctions<FL>::complex_dot(rtmp, btmp);
                    rr[0] += GMatrixFunctions<FL>::complex_dot(itmp, btmp);
                }
                if (para_rule != nullptr &&
                    dynamic_pointer_cast<ParallelRule<S, FL>>(para_rule)
                        ->number(op->dops[i])) {
                    results.insert(results.end(), rr.begin(), rr.end());
                    results_idx.push_back(expectations.size());
                }
                expectations.push_back(make_pair(op->dops[i], rr));
            } else
                assert(false);
        }
        btmp.deallocate();
        if ((FL)const_e != (FL)0.0 && op->mat->data.size() > 0)
            op->mat->data[0] = expr;
        if (results.size() != 0) {
            assert(para_rule != nullptr);
            para_rule->comm->allreduce_sum(results.data(), results.size());
            for (size_t i = 0; i < results.size(); i += ket.size())
                memcpy(expectations[results_idx[i]].second.data(),
                       results.data() + i, sizeof(FL) * ket.size());
        }
        tf->opf->seq->mode = mode;
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, (size_t)nflop, t.get_time());
    }
    // [ket] = exp( [H_eff] ) | [ket] > (RK4 approximation)
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<GMatrix<FL>>, tuple<FL, FP, int, size_t, double>>
    rk4_apply(FC beta, typename const_fl_type<FL>::FL const_e,
              bool eval_energy = false,
              const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(ket.size() == 2);
        GMatrix<FL> vr(ket[0]->data, (MKL_INT)ket[0]->total_memory, 1);
        GMatrix<FL> vi(ket[1]->data, (MKL_INT)ket[1]->total_memory, 1);
        vector<GMatrix<FL>> k, r;
        Timer t;
        t.get_time();
        frame_<FP>()->activate(1);
        for (int i = 0; i < 3; i++) {
            r.push_back(GMatrix<FL>(nullptr, (MKL_INT)ket[0]->total_memory, 1));
            r[i + i].allocate();
            r.push_back(GMatrix<FL>(nullptr, (MKL_INT)ket[1]->total_memory, 1));
            r[i + i + 1].allocate();
        }
        frame_<FP>()->activate(0);
        for (int i = 0; i < 4; i++) {
            k.push_back(GMatrix<FL>(nullptr, (MKL_INT)ket[0]->total_memory, 1));
            k[i + i].allocate(), k[i + i].clear();
            k.push_back(GMatrix<FL>(nullptr, (MKL_INT)ket[1]->total_memory, 1));
            k[i + i + 1].allocate(), k[i + i + 1].clear();
        }
        tf->opf->seq->cumulative_nflop = 0;
        const vector<FP> ks = vector<FP>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<FP>> cs = vector<vector<FP>>{
            vector<FP>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0, -5.0 / 162.0},
            vector<FP>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<FP>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &,
                            const GMatrix<FL> &, const GMatrix<FL> &, FC)> &f =
            [this](const GMatrix<FL> &are, const GMatrix<FL> &aim,
                   const GMatrix<FL> &bre, const GMatrix<FL> &bim, FC scale) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked)) {
                    if (scale.real() != 0) {
                        this->tf->operator()(are, bre, scale.real());
                        this->tf->operator()(aim, bim, scale.real());
                    }
                    if (scale.imag() != 0) {
                        this->tf->operator()(are, bim, scale.imag());
                        this->tf->operator()(aim, bre, -scale.imag());
                    }
                } else {
                    if (scale.real() != 0) {
                        (*this)(are, bre, 0, scale.real());
                        (*this)(aim, bim, 0, scale.real());
                    }
                    if (scale.imag() != 0) {
                        (*this)(are, bim, 0, scale.imag());
                        (*this)(aim, bre, 0, -scale.imag());
                    }
                }
            };
        // k0 ~ k3
        for (int i = 0; i < 4; i++) {
            if (i == 0)
                f(vr, vi, k[i + i], k[i + i + 1], beta);
            else {
                GMatrixFunctions<FL>::copy(r[0], vr);
                GMatrixFunctions<FL>::copy(r[1], vi);
                GMatrixFunctions<FL>::iadd(r[0], k[i + i - 2], ks[i]);
                GMatrixFunctions<FL>::iadd(r[1], k[i + i - 1], ks[i]);
                f(r[0], r[1], k[i + i], k[i + i + 1], beta);
            }
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            FC factor = exp(beta * (FL)((i + 1) / 3) * (FL)const_e);
            r[i + i].clear(), r[i + i + 1].clear();
            if (factor.real() != 0) {
                GMatrixFunctions<FL>::iadd(r[i + i], vr, factor.real());
                GMatrixFunctions<FL>::iadd(r[i + i + 1], vi, factor.real());
            }
            if (factor.imag() != 0) {
                GMatrixFunctions<FL>::iadd(r[i + i], vi, factor.imag());
                GMatrixFunctions<FL>::iadd(r[i + i + 1], vr, -factor.imag());
            }
            for (size_t j = 0; j < 4; j++) {
                if (factor.real() != 0) {
                    GMatrixFunctions<FL>::iadd(r[i + i], k[j + j],
                                               cs[i][j] * factor.real());
                    GMatrixFunctions<FL>::iadd(r[i + i + 1], k[j + j + 1],
                                               cs[i][j] * factor.real());
                }
                if (factor.imag() != 0) {
                    GMatrixFunctions<FL>::iadd(r[i + i], k[j + j + 1],
                                               cs[i][j] * factor.imag());
                    GMatrixFunctions<FL>::iadd(r[i + i + 1], k[j + j],
                                               -cs[i][j] * factor.imag());
                }
            }
        }
        FP norm_re = GMatrixFunctions<FL>::norm(r[2 + 2]);
        FP norm_im = GMatrixFunctions<FL>::norm(r[2 + 2 + 1]);
        FP norm = sqrt(norm_re * norm_re + norm_im * norm_im);
        FL energy = -(FL)const_e;
        if (eval_energy) {
            k[0].clear();
            k[1].clear();
            f(r[2 + 2], r[2 + 2 + 1], k[0], k[1], 1.0);
            energy = (GMatrixFunctions<FL>::complex_dot(r[2 + 2], k[0]) +
                      GMatrixFunctions<FL>::complex_dot(r[2 + 2 + 1], k[1])) /
                     (norm * norm);
        }
        for (int i = 3; i >= 0; i--)
            k[i + i + 1].deallocate(), k[i + i].deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(r, make_tuple(energy, norm, 4 + eval_energy,
                                       (size_t)nflop, t.get_time()));
    }
    void deallocate() {
        frame_<FP>()->activate(0);
        for (int i = (int)wfn_infos.size() - 1; i >= 0; i--) {
            vector<pair<
                S *, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
                mp;
            mp.reserve(wfn_infos[i].size());
            for (auto it = wfn_infos[i].cbegin(); it != wfn_infos[i].cend();
                 it++)
                mp.emplace_back(it->second->quanta, it->second);
            sort(mp.begin(), mp.end(),
                 [](const pair<S *, shared_ptr<typename SparseMatrixInfo<
                                        S>::ConnectionInfo>> &a,
                    const pair<S *, shared_ptr<typename SparseMatrixInfo<
                                        S>::ConnectionInfo>> &b) {
                     return a.first > b.first;
                 });
            for (const auto &t : mp)
                t.second->deallocate();
        }
        if (compute_diag)
            diag->deallocate();
        op->deallocate();
        vector<pair<S *, shared_ptr<SparseMatrixInfo<S>>>> mp;
        mp.reserve(left_op_infos.size() + right_op_infos.size());
        for (int i = (int)right_op_infos.size() - 1; i >= 0; i--)
            mp.emplace_back(right_op_infos[i].second->quanta,
                            right_op_infos[i].second);
        for (int i = (int)left_op_infos.size() - 1; i >= 0; i--)
            mp.emplace_back(left_op_infos[i].second->quanta,
                            left_op_infos[i].second);
        sort(mp.begin(), mp.end(),
             [](const pair<S *, shared_ptr<SparseMatrixInfo<S>>> &a,
                const pair<S *, shared_ptr<SparseMatrixInfo<S>>> &b) {
                 return a.first > b.first;
             });
        for (const auto &t : mp) {
            if (t.second->cinfo != nullptr)
                t.second->cinfo->deallocate();
            t.second->deallocate();
        }
    }
};

} // namespace block2
