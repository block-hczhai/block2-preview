
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

enum struct ExpectationAlgorithmTypes : uint8_t { Automatic, Normal, Fast };

enum struct ExpectationTypes : uint8_t { Real, Complex };

enum struct LinearSolverTypes : uint8_t { CG, MinRes, GCROT };

template <typename S, typename = MPS<S>> struct EffectiveHamiltonian;

// Effective Hamiltonian
template <typename S> struct EffectiveHamiltonian<S, MPS<S>> {
    typedef S s_type;
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
    vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>> wfn_infos;
    vector<S> operator_quanta;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S>> &op,
        const shared_ptr<SparseMatrix<S>> &bra,
        const shared_ptr<SparseMatrix<S>> &ket,
        const shared_ptr<OpElement<S>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat,
        const shared_ptr<TensorFunctions<S>> &ptf, bool compute_diag = true)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(ptf->copy()), hop_mat(hop_mat),
          compute_diag(compute_diag) {
        // wavefunction
        if (compute_diag) {
            assert(bra->info == ket->info);
            diag = make_shared<SparseMatrix<S>>();
            diag->allocate(ket->info);
        }
        // unique sub labels
        S cdq = ket->info->delta_quantum;
        S vdq = bra->info->delta_quantum;
        opdq = hop->q_label;
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        operator_quanta = msl;
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S>::get_uniq_sub_labels(op->mat, hop_mat, msl);
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
        cmat = make_shared<SparseMatrix<S>>();
        vmat = make_shared<SparseMatrix<S>>();
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
            cmat->data = vmat->data = (double *)0;
            cmat->factor = 1.0;
            tf->tensor_product_multiply(op->mat->data[0], op->lopt, op->ropt,
                                        cmat, vmat, opdq, false);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        } else if (tf->opf->seq->mode & SeqTypes::Tasked) {
            cmat->data = vmat->data = (double *)0;
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
    shared_ptr<SparseMatrixGroup<S>>
    perturbative_noise(bool trace_right, int iL, int iR, FuseTypes ftype,
                       const shared_ptr<MPSInfo<S>> &mps_info,
                       const NoiseTypes noise_type,
                       const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        assert(msl.size() == 1 && msl[0] == opdq);
        shared_ptr<OpExpr<S>> pexpr = op->mat->data[0];
        shared_ptr<Symbolic<S>> pmat = make_shared<SymbolicColumnVector<S>>(
            1, vector<shared_ptr<OpExpr<S>>>{pexpr});
        vector<pair<uint8_t, S>> psubsl = Partition<S>::get_uniq_sub_labels(
            pmat, hop_mat, msl, true, trace_right, false)[0];
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
        shared_ptr<SparseMatrixGroup<S>> perturb_ket =
            make_shared<SparseMatrixGroup<S>>(d_alloc);
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
                       (size_t)numeric_limits<decltype(MatrixRef::n)>::max());
                tf->opf->seq->auto_perform(MatrixRef(
                    perturb_ket->data, (MKL_INT)perturb_ket->total_memory, 1));
            } else {
                vector<MatrixRef> pmats(perturb_ket->n,
                                        MatrixRef(nullptr, 0, 0));
                for (int j = 0; j < perturb_ket->n; j++)
                    pmats[j] =
                        MatrixRef((*perturb_ket)[j]->data,
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
        S idx_opdq = dynamic_pointer_cast<OpElement<S>>(op->dops[idx])->q_label;
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
    tuple<double, int, size_t, double>
    eigs(bool iprint = false, double conv_thrd = 5E-6, int max_iter = 5000,
         int soft_max_iter = -1,
         DavidsonTypes davidson_type = DavidsonTypes::Normal, double shift = 0,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr,
         const vector<shared_ptr<SparseMatrix<S>>> &ortho_bra =
             vector<shared_ptr<SparseMatrix<S>>>()) {
        int ndav = 0;
        assert(compute_diag);
        DiagonalMatrix aa(diag->data, (MKL_INT)diag->total_memory);
        vector<MatrixRef> bs = vector<MatrixRef>{
            MatrixRef(ket->data, (MKL_INT)ket->total_memory, 1)};
        vector<MatrixRef> ors =
            vector<MatrixRef>(ortho_bra.size(), MatrixRef(nullptr, 0, 0));
        for (size_t i = 0; i < ortho_bra.size(); i++)
            ors[i] = MatrixRef(ortho_bra[i]->data,
                               (MKL_INT)ortho_bra[i]->total_memory, 1);
        frame->activate(0);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        vector<double> eners =
            (tf->opf->seq->mode == SeqTypes::Auto ||
             (tf->opf->seq->mode & SeqTypes::Tasked))
                ? MatrixFunctions::harmonic_davidson(
                      *tf, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, 2, 50, ors)
                : MatrixFunctions::harmonic_davidson(
                      *this, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, 2, 50, ors);
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners[0], ndav, (size_t)nflop, t.get_time());
    }
    // [bra] = ([H_eff] + omega + i eta)^(-1) x [ket]
    // (real gf, imag gf), (nmult, niter), nflop, tmult
    tuple<pair<double, double>, pair<int, int>, size_t, double>
    greens_function(double const_e, double omega, double eta,
                    const shared_ptr<SparseMatrix<S>> &real_bra,
                    pair<int, int> gcrotmk_size, bool iprint = false,
                    double conv_thrd = 5E-6, int max_iter = 5000,
                    int soft_max_iter = -1,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int nmult = 0, nmultx = 0, niter = 0;
        frame->activate(0);
        Timer t;
        t.get_time();
        MatrixRef mket(ket->data, (MKL_INT)ket->total_memory, 1);
        MatrixRef ibra(bra->data, (MKL_INT)bra->total_memory, 1);
        MatrixRef rbra(real_bra->data, (MKL_INT)real_bra->total_memory, 1);
        MatrixRef bre(nullptr, (MKL_INT)ket->total_memory, 1);
        MatrixRef cre(nullptr, (MKL_INT)ket->total_memory, 1);
        ComplexMatrixRef cbra(nullptr, (MKL_INT)bra->total_memory, 1);
        ComplexMatrixRef cket(nullptr, (MKL_INT)bra->total_memory, 1);
        bre.allocate();
        cre.allocate();
        cbra.allocate();
        cket.allocate();
        ComplexDiagonalMatrix aa(nullptr, 0);
        if (compute_diag) {
            aa = ComplexDiagonalMatrix(nullptr, (MKL_INT)diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++)
                aa.data[i] =
                    complex<double>(diag->data[i] + const_e + omega, eta);
        }
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &)> &f =
            [this](const MatrixRef &a, const MatrixRef &b) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b);
                else
                    return (*this)(a, b);
            };
        auto op = [omega, eta, const_e, &f, &bre, &cre,
                   &nmult](const ComplexMatrixRef &b,
                           const ComplexMatrixRef &c) -> void {
            ComplexMatrixFunctions::extract_complex(
                b, bre, MatrixRef(nullptr, bre.m, bre.n));
            cre.clear();
            f(bre, cre);
            ComplexMatrixFunctions::fill_complex(
                c, cre, MatrixRef(nullptr, cre.m, cre.n));
            ComplexMatrixFunctions::extract_complex(
                b, MatrixRef(nullptr, bre.m, bre.n), bre);
            cre.clear();
            f(bre, cre);
            ComplexMatrixFunctions::fill_complex(
                c, MatrixRef(nullptr, cre.m, cre.n), cre);
            ComplexMatrixFunctions::iadd(c, b,
                                         complex<double>(const_e + omega, eta));
            nmult += 2;
        };
        tf->opf->seq->cumulative_nflop = 0;
        rbra.clear();
        f(ibra, rbra);
        MatrixFunctions::iadd(rbra, ibra, const_e + omega);
        MatrixFunctions::iscale(rbra, -1 / eta);
        ComplexMatrixFunctions::fill_complex(cbra, rbra, ibra);
        cket.clear();
        ComplexMatrixFunctions::fill_complex(
            cket, mket, MatrixRef(nullptr, mket.m, mket.n));
        // solve bra
        complex<double> gf = ComplexMatrixFunctions::gcrotmk(
            op, aa, cbra, cket, nmultx, niter, gcrotmk_size.first,
            gcrotmk_size.second, iprint,
            para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
            max_iter, soft_max_iter);
        gf = conj(gf);
        ComplexMatrixFunctions::extract_complex(cbra, rbra, ibra);
        if (compute_diag)
            aa.deallocate();
        cket.deallocate();
        cbra.deallocate();
        cre.deallocate();
        bre.deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(make_pair(real(gf), imag(gf)),
                          make_pair(nmult, niter), (size_t)nflop, t.get_time());
    }
    // [ibra] = (([H_eff] + omega)^2 + eta^2)^(-1) x (-eta [ket])
    // [rbra] = -([H_eff] + omega) (1/eta) [bra]
    // (real gf, imag gf), (nmult, numltp), nflop, tmult
    tuple<pair<double, double>, pair<int, int>, size_t, double>
    greens_function_squared(
        double const_e, double omega, double eta,
        const shared_ptr<SparseMatrix<S>> &real_bra,
        int n_harmonic_projection = 0, bool iprint = false,
        double conv_thrd = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int nmult = 0, nmultx = 0;
        frame->activate(0);
        Timer t;
        t.get_time();
        MatrixRef mket(ket->data, (MKL_INT)ket->total_memory, 1);
        MatrixRef ibra(bra->data, (MKL_INT)bra->total_memory, 1);
        MatrixRef ktmp(nullptr, (MKL_INT)ket->total_memory, 1);
        ktmp.allocate();
        MatrixRef btmp(nullptr, (MKL_INT)bra->total_memory, 1);
        btmp.allocate();
        ktmp.clear();
        MatrixFunctions::iadd(ktmp, mket, -eta);
        DiagonalMatrix aa(nullptr, 0);
        if (compute_diag) {
            aa = DiagonalMatrix(nullptr, (MKL_INT)diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++) {
                aa.data[i] = diag->data[i] + const_e + omega;
                aa.data[i] = aa.data[i] * aa.data[i] + eta * eta;
            }
        }
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &)> &f =
            [this](const MatrixRef &a, const MatrixRef &b) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b);
                else
                    return (*this)(a, b);
            };
        auto op = [omega, eta, const_e, &f, &btmp,
                   &nmult](const MatrixRef &b, const MatrixRef &c) -> void {
            btmp.clear();
            f(b, btmp);
            MatrixFunctions::iadd(btmp, b, const_e + omega);
            f(btmp, c);
            MatrixFunctions::iadd(c, btmp, const_e + omega);
            MatrixFunctions::iadd(c, b, eta * eta);
            nmult += 2;
        };
        tf->opf->seq->cumulative_nflop = 0;
        // solve imag part -> ibra
        double igf = 0;
        int nmultp = 0;
        if (n_harmonic_projection == 0)
            igf = MatrixFunctions::conjugate_gradient(
                      op, aa, ibra, ktmp, nmultx, 0.0, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter) /
                  (-eta);
        else if (n_harmonic_projection < 0) {
            int ndav = 0, ncg = 0;
            int kk = -n_harmonic_projection;
            igf =
                MatrixFunctions::davidson_projected_deflated_conjugate_gradient(
                    op, aa, ibra, ktmp, kk, ncg, ndav, 0.0, iprint,
                    para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
                    conv_thrd, max_iter * kk, soft_max_iter * kk) /
                (-eta);
            nmult = ncg * 2;
            nmultp = ndav * 2;
        } else {
            vector<MatrixRef> bs = vector<MatrixRef>(
                n_harmonic_projection,
                MatrixRef(nullptr, (MKL_INT)ket->total_memory, 1));
            for (int ih = 0; ih < n_harmonic_projection; ih++) {
                bs[ih].allocate();
                if (ih == 0)
                    MatrixFunctions::copy(bs[ih], ibra);
                else
                    Random::fill_rand_double(bs[ih].data, bs[ih].size());
            }
            MatrixFunctions::harmonic_davidson(
                op, aa, bs, 0.0,
                DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond,
                nmultx, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, 1E-4,
                max_iter, soft_max_iter, 2, 50);
            nmultp = nmult;
            nmult = 0;
            igf = MatrixFunctions::deflated_conjugate_gradient(
                      op, aa, ibra, ktmp, nmultx, 0.0, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, bs) /
                  (-eta);
            for (int ih = n_harmonic_projection - 1; ih >= 0; ih--)
                bs[ih].deallocate();
        }
        if (compute_diag)
            aa.deallocate();
        btmp.deallocate();
        ktmp.deallocate();
        // compute real part -> rbra
        MatrixRef rbra(real_bra->data, (MKL_INT)real_bra->total_memory, 1);
        rbra.clear();
        f(ibra, rbra);
        MatrixFunctions::iadd(rbra, ibra, const_e + omega);
        MatrixFunctions::iscale(rbra, -1 / eta);
        // compute real part green's function
        double rgf = MatrixFunctions::dot(rbra, mket);
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(make_pair(rgf, igf), make_pair(nmult + 1, nmultp),
                          (size_t)nflop, t.get_time());
    }
    // [bra] = [H_eff]^(-1) x [ket]
    // energy, nmult, nflop, tmult
    tuple<double, pair<int, int>, size_t, double>
    inverse_multiply(double const_e, LinearSolverTypes solver_type,
                     pair<int, int> gcrotmk_size, bool iprint = false,
                     double conv_thrd = 5E-6, int max_iter = 5000,
                     int soft_max_iter = -1,
                     const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int nmult = 0, niter = 0;
        frame->activate(0);
        Timer t;
        t.get_time();
        MatrixRef mket(ket->data, (MKL_INT)ket->total_memory, 1);
        MatrixRef mbra(bra->data, (MKL_INT)bra->total_memory, 1);
        tf->opf->seq->cumulative_nflop = 0;
        DiagonalMatrix aa(nullptr, 0);
        if (compute_diag && solver_type != LinearSolverTypes::MinRes) {
            aa = DiagonalMatrix(nullptr, (MKL_INT)diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++)
                aa.data[i] = diag->data[i] + const_e;
        }
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &)> &f =
            [this](const MatrixRef &a, const MatrixRef &b) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b);
                else
                    return (*this)(a, b);
            };
        double r =
            solver_type == LinearSolverTypes::CG
                ? MatrixFunctions::conjugate_gradient(
                      f, aa, mbra, mket, nmult, const_e, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter)
                : (solver_type == LinearSolverTypes::MinRes
                       ? MatrixFunctions::minres(
                             f, mbra, mket, nmult, const_e, iprint,
                             para_rule == nullptr ? nullptr : para_rule->comm,
                             conv_thrd, max_iter, soft_max_iter)
                       : MatrixFunctions::gcrotmk(
                             f, aa, mbra, mket, nmult, niter,
                             gcrotmk_size.first, gcrotmk_size.second, const_e,
                             iprint,
                             para_rule == nullptr ? nullptr : para_rule->comm,
                             conv_thrd, max_iter, soft_max_iter));
        if (compute_diag && solver_type != LinearSolverTypes::MinRes)
            aa.deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(r, make_pair(nmult, niter), (size_t)nflop,
                          t.get_time());
    }
    shared_ptr<OpExpr<S>>
    add_const_term(double const_e,
                   const shared_ptr<ParallelRule<S>> &para_rule) {
        shared_ptr<OpExpr<S>> expr = op->mat->data[0];
        if (const_e != 0) {
            // q_label does not matter
            shared_ptr<OpExpr<S>> iop = make_shared<OpElement<S>>(
                OpNames::I, SiteIndex(),
                dynamic_pointer_cast<OpElement<S>>(op->dops[0])->q_label);
            if (para_rule == nullptr || para_rule->is_root()) {
                if (op->lopt->get_type() == OperatorTensorTypes::Delayed ||
                    op->ropt->get_type() == OperatorTensorTypes::Delayed) {
                    bool dleft =
                        op->lopt->get_type() == OperatorTensorTypes::Delayed;
                    shared_ptr<DelayedOperatorTensor<S>> dopt =
                        dynamic_pointer_cast<DelayedOperatorTensor<S>>(
                            dleft ? op->lopt : op->ropt);
                    shared_ptr<OpElement<S>> xiop =
                        dynamic_pointer_cast<OpElement<S>>(iop);
                    if (dopt->lopt->ops.count(iop) != 0 &&
                        dopt->ropt->ops.count(iop) != 0)
                        op->mat->data[0] =
                            expr +
                            (shared_ptr<OpExpr<S>>)make_shared<OpSumProd<S>>(
                                xiop, xiop,
                                vector<shared_ptr<OpElement<S>>>{xiop, xiop},
                                vector<bool>{false, false}, const_e, 0);
                    else
                        op->mat->data[0] = expr + const_e * (iop * iop);
                } else
                    op->mat->data[0] = expr + const_e * (iop * iop);
            }
        }
        return expr;
    }
    // [bra] = [H_eff] x [ket]
    // norm, nmult, nflop, tmult
    tuple<double, int, size_t, double>
    multiply(double const_e,
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
        (*this)(MatrixRef(ket->data, (MKL_INT)ket->total_memory, 1),
                MatrixRef(bra->data, (MKL_INT)bra->total_memory, 1));
        op->mat->data[0] = expr;
        double norm = MatrixFunctions::norm(
            MatrixRef(bra->data, (MKL_INT)bra->total_memory, 1));
        tf->opf->seq->mode = mode;
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(norm, 1, (size_t)nflop, t.get_time());
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tmult
    tuple<vector<pair<shared_ptr<OpExpr<S>>, double>>, size_t, double>
    expect(double const_e, ExpectationAlgorithmTypes algo_type,
           ExpectationTypes ex_type,
           const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<OpExpr<S>> expr = nullptr;
        if (const_e != 0 && op->mat->data.size() > 0)
            expr = add_const_term(const_e, para_rule);
        assert(ex_type == ExpectationTypes::Real);
        if (algo_type == ExpectationAlgorithmTypes::Automatic)
            algo_type = op->mat->data.size() > 1
                            ? ExpectationAlgorithmTypes::Fast
                            : ExpectationAlgorithmTypes::Normal;
        SeqTypes mode = tf->opf->seq->mode;
        tf->opf->seq->mode = tf->opf->seq->mode & SeqTypes::Simple
                                 ? SeqTypes::Simple
                                 : SeqTypes::None;
        tf->opf->seq->cumulative_nflop = 0;
        Timer t;
        t.get_time();
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations;
        // may happen for NPDM with ancilla
        if (op->mat->data.size() == 1 &&
            dynamic_pointer_cast<OpElement<S>>(op->dops[0])->name ==
                OpNames::Zero)
            ;
        else if (algo_type == ExpectationAlgorithmTypes::Normal) {
            MatrixRef ktmp(ket->data, (MKL_INT)ket->total_memory, 1);
            MatrixRef rtmp(bra->data, (MKL_INT)bra->total_memory, 1);
            MatrixRef btmp(nullptr, (MKL_INT)bra->total_memory, 1);
            btmp.allocate();
            expectations.reserve(op->mat->data.size());
            vector<double> results;
            vector<size_t> results_idx;
            results.reserve(op->mat->data.size());
            results_idx.reserve(op->mat->data.size());
            if (para_rule != nullptr)
                para_rule->set_partition(ParallelRulePartitionTypes::Middle);
            for (size_t i = 0; i < op->mat->data.size(); i++) {
                assert(dynamic_pointer_cast<OpElement<S>>(op->dops[i])->name !=
                       OpNames::Zero);
                S idx_opdq =
                    dynamic_pointer_cast<OpElement<S>>(op->dops[i])->q_label;
                S ket_dq = ket->info->delta_quantum;
                S bra_dq = bra->info->delta_quantum;
                if (idx_opdq.combine(bra_dq, ket_dq) == S(S::invalid))
                    expectations.push_back(make_pair(op->dops[i], 0.0));
                else {
                    double r = 0.0;
                    if (para_rule == nullptr ||
                        !para_rule->number(op->dops[i])) {
                        btmp.clear();
                        (*this)(ktmp, btmp, (int)i, 1.0, true);
                        r = MatrixFunctions::dot(btmp, rtmp);
                    } else {
                        if (para_rule->own(op->dops[i])) {
                            btmp.clear();
                            (*this)(ktmp, btmp, (int)i, 1.0, false);
                            r = MatrixFunctions::dot(btmp, rtmp);
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
        } else
            expectations = tf->tensor_product_expectation(
                op->dops, op->mat->data, op->lopt, op->ropt, ket, bra);
        if (const_e != 0 && op->mat->data.size() > 0)
            op->mat->data[0] = expr;
        tf->opf->seq->mode = mode;
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, (size_t)nflop, t.get_time());
    }
    // return |ket> and beta [H_eff] |ket>
    pair<vector<shared_ptr<SparseMatrix<S>>>, tuple<int, size_t, double>>
    first_rk4_apply(double beta, double const_e,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        vector<shared_ptr<SparseMatrix<S>>> r(2);
        for (int i = 0; i < 2; i++) {
            r[i] = make_shared<SparseMatrix<S>>(d_alloc);
            r[i]->allocate(bra->info);
        }
        MatrixRef kk(ket->data, (MKL_INT)ket->total_memory, 1);
        MatrixRef r0(r[0]->data, (MKL_INT)bra->total_memory, 1);
        MatrixRef r1(r[1]->data, (MKL_INT)bra->total_memory, 1);
        Timer t;
        t.get_time();
        assert(op->mat->data.size() > 0);
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &, double)> &f =
            [this](const MatrixRef &a, const MatrixRef &b, double scale) {
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
        // if (const_e != 0)
        //     MatrixFunctions::iadd(r1, r0, beta * const_e);
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(r, make_tuple(1, (size_t)nflop, t.get_time()));
    }
    pair<vector<shared_ptr<SparseMatrix<S>>>,
         tuple<double, double, int, size_t, double>>
    second_rk4_apply(double beta, double const_e,
                     const shared_ptr<SparseMatrix<S>> &hket,
                     bool eval_energy = false,
                     const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        vector<shared_ptr<SparseMatrix<S>>> rr(3), kk(4);
        kk[0] = hket;
        for (int i = 0; i < 3; i++) {
            rr[i] = make_shared<SparseMatrix<S>>(d_alloc);
            rr[i]->allocate(ket->info);
        }
        for (int i = 0; i < 3; i++) {
            kk[i + 1] = make_shared<SparseMatrix<S>>(d_alloc);
            kk[i + 1]->allocate(ket->info);
        }
        MatrixRef v(ket->data, (MKL_INT)ket->total_memory, 1);
        vector<MatrixRef> k(4, v), r(3, v);
        Timer t;
        t.get_time();
        for (int i = 0; i < 3; i++)
            r[i] = MatrixRef(rr[i]->data, (MKL_INT)ket->total_memory, 1);
        for (int i = 0; i < 4; i++)
            k[i] = MatrixRef(kk[i]->data, (MKL_INT)ket->total_memory, 1);
        tf->opf->seq->cumulative_nflop = 0;
        const vector<double> ks = vector<double>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<double>> cs = vector<vector<double>>{
            vector<double>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0,
                           -5.0 / 162.0},
            vector<double>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<double>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &, double)> &f =
            [this](const MatrixRef &a, const MatrixRef &b, double scale) {
                if (this->tf->opf->seq->mode == SeqTypes::Auto ||
                    (this->tf->opf->seq->mode & SeqTypes::Tasked))
                    return this->tf->operator()(a, b, scale);
                else
                    return (*this)(a, b, 0, scale);
            };
        // k1 ~ k3
        for (int i = 1; i < 4; i++) {
            MatrixFunctions::copy(r[0], v);
            MatrixFunctions::iadd(r[0], k[i - 1], ks[i]);
            f(r[0], k[i], beta);
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            double factor = exp(beta * (i + 1) / 3 * const_e);
            MatrixFunctions::copy(r[i], v);
            MatrixFunctions::iscale(r[i], factor);
            for (size_t j = 0; j < 4; j++)
                MatrixFunctions::iadd(r[i], k[j], cs[i][j] * factor);
        }
        double norm = MatrixFunctions::norm(r[2]);
        double energy = -const_e;
        if (eval_energy) {
            k[0].clear();
            f(r[2], k[0], 1.0);
            energy = MatrixFunctions::dot(r[2], k[0]) / (norm * norm);
        }
        for (int i = 3; i >= 1; i--)
            kk[i]->deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(rr, make_tuple(energy, norm, 3 + eval_energy,
                                        (size_t)nflop, t.get_time()));
    }
    // [ket] = exp( [H_eff] ) | [ket] > (RK4 approximation)
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<MatrixRef>, tuple<double, double, int, size_t, double>>
    rk4_apply(double beta, double const_e, bool eval_energy = false,
              const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        MatrixRef v(ket->data, (MKL_INT)ket->total_memory, 1);
        vector<MatrixRef> k, r;
        Timer t;
        t.get_time();
        frame->activate(1);
        for (int i = 0; i < 3; i++) {
            r.push_back(MatrixRef(nullptr, (MKL_INT)ket->total_memory, 1));
            r[i].allocate();
        }
        frame->activate(0);
        for (int i = 0; i < 4; i++) {
            k.push_back(MatrixRef(nullptr, (MKL_INT)ket->total_memory, 1));
            k[i].allocate(), k[i].clear();
        }
        tf->opf->seq->cumulative_nflop = 0;
        const vector<double> ks = vector<double>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<double>> cs = vector<vector<double>>{
            vector<double>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0,
                           -5.0 / 162.0},
            vector<double>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<double>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &, double)> &f =
            [this](const MatrixRef &a, const MatrixRef &b, double scale) {
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
                MatrixFunctions::copy(r[0], v);
                MatrixFunctions::iadd(r[0], k[i - 1], ks[i]);
                f(r[0], k[i], beta);
            }
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            double factor = exp(beta * (i + 1) / 3 * const_e);
            MatrixFunctions::copy(r[i], v);
            MatrixFunctions::iscale(r[i], factor);
            for (size_t j = 0; j < 4; j++)
                MatrixFunctions::iadd(r[i], k[j], cs[i][j] * factor);
        }
        double norm = MatrixFunctions::norm(r[2]);
        double energy = -const_e;
        if (eval_energy) {
            k[0].clear();
            f(r[2], k[0], 1.0);
            energy = MatrixFunctions::dot(r[2], k[0]) / (norm * norm);
        }
        for (int i = 3; i >= 0; i--)
            k[i].deallocate();
        post_precompute();
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
    expo_apply(double beta, double const_e, bool symmetric, bool iprint = false,
               const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(compute_diag);
        double anorm = MatrixFunctions::norm(
            MatrixRef(diag->data, (MKL_INT)diag->total_memory, 1));
        MatrixRef v(ket->data, (MKL_INT)ket->total_memory, 1);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        int nexpo = (tf->opf->seq->mode == SeqTypes::Auto ||
                     (tf->opf->seq->mode & SeqTypes::Tasked))
                        ? MatrixFunctions::expo_apply(
                              *tf, beta, anorm, v, const_e, symmetric, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm)
                        : MatrixFunctions::expo_apply(
                              *this, beta, anorm, v, const_e, symmetric, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm);
        double norm = MatrixFunctions::norm(v);
        MatrixRef tmp(nullptr, (MKL_INT)ket->total_memory, 1);
        tmp.allocate();
        tmp.clear();
        if (tf->opf->seq->mode == SeqTypes::Auto ||
            (tf->opf->seq->mode & SeqTypes::Tasked))
            (*tf)(v, tmp);
        else
            (*this)(v, tmp);
        double energy = MatrixFunctions::dot(v, tmp) / (norm * norm);
        tmp.deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, (size_t)nflop, t.get_time());
    }
    void deallocate() {
        frame->activate(0);
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
template <typename S> struct LinearEffectiveHamiltonian {
    typedef S s_type;
    vector<shared_ptr<EffectiveHamiltonian<S>>> h_effs;
    vector<double> coeffs;
    S opdq;
    LinearEffectiveHamiltonian(const shared_ptr<EffectiveHamiltonian<S>> &h_eff)
        : h_effs{h_eff}, coeffs{1} {}
    LinearEffectiveHamiltonian(
        const vector<shared_ptr<EffectiveHamiltonian<S>>> &h_effs,
        const vector<double> &coeffs)
        : h_effs(h_effs), coeffs(coeffs) {}
    static shared_ptr<LinearEffectiveHamiltonian<S>>
    linearize(const shared_ptr<LinearEffectiveHamiltonian<S>> &x) {
        return x;
    }
    static shared_ptr<LinearEffectiveHamiltonian<S>>
    linearize(const shared_ptr<EffectiveHamiltonian<S>> &x) {
        return make_shared<LinearEffectiveHamiltonian<S>>(x);
    }
    // [c] = [H_eff[idx]] x [b]
    void operator()(const MatrixRef &b, const MatrixRef &c) {
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
    tuple<double, int, size_t, double>
    eigs(bool iprint = false, double conv_thrd = 5E-6, int max_iter = 5000,
         int soft_max_iter = -1,
         DavidsonTypes davidson_type = DavidsonTypes::Normal, double shift = 0,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int ndav = 0;
        assert(h_effs.size() != 0);
        const shared_ptr<TensorFunctions<S>> &tf = h_effs[0]->tf;
        DiagonalMatrix aa(nullptr, (MKL_INT)h_effs[0]->diag->total_memory);
        aa.allocate();
        aa.clear();
        for (size_t ih = 0; ih < h_effs.size(); ih++) {
            assert(h_effs[ih]->compute_diag);
            MatrixFunctions::iadd(
                MatrixRef(aa.data, (MKL_INT)aa.size(), 1),
                MatrixRef(h_effs[ih]->diag->data,
                          (MKL_INT)h_effs[ih]->diag->total_memory, 1),
                coeffs[ih]);
            h_effs[ih]->precompute();
        }
        vector<MatrixRef> bs = vector<MatrixRef>{MatrixRef(
            h_effs[0]->ket->data, (MKL_INT)h_effs[0]->ket->total_memory, 1)};
        frame->activate(0);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        vector<double> eners = MatrixFunctions::harmonic_davidson(
            *this, aa, bs, shift, davidson_type, ndav, iprint,
            para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
            max_iter, soft_max_iter);
        for (size_t ih = 0; ih < h_effs.size(); ih++)
            h_effs[ih]->post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        aa.deallocate();
        return make_tuple(eners[0], ndav, (size_t)nflop, t.get_time());
    }
    void deallocate() {}
};

template <typename T>
inline shared_ptr<LinearEffectiveHamiltonian<typename T::s_type>>
operator*(double d, const shared_ptr<T> &x) {
    shared_ptr<LinearEffectiveHamiltonian<typename T::s_type>> xx =
        LinearEffectiveHamiltonian<typename T::s_type>::linearize(x);
    vector<double> new_coeffs;
    for (auto &c : xx->coeffs)
        new_coeffs.push_back(c * d);
    return make_shared<LinearEffectiveHamiltonian<typename T::s_type>>(
        xx->h_effs, new_coeffs);
}

template <typename T>
inline shared_ptr<LinearEffectiveHamiltonian<typename T::s_type>>
operator*(const shared_ptr<T> &x, double d) {
    return d * x;
}

template <typename T>
inline shared_ptr<LinearEffectiveHamiltonian<typename T::s_type>>
operator-(const shared_ptr<T> &x) {
    return (-1.0) * x;
}

template <typename T1, typename T2>
inline shared_ptr<LinearEffectiveHamiltonian<typename T1::s_type>>
operator+(const shared_ptr<T1> &x, const shared_ptr<T2> &y) {
    shared_ptr<LinearEffectiveHamiltonian<typename T1::s_type>> xx =
        LinearEffectiveHamiltonian<typename T1::s_type>::linearize(x);
    shared_ptr<LinearEffectiveHamiltonian<typename T1::s_type>> yy =
        LinearEffectiveHamiltonian<typename T1::s_type>::linearize(y);
    vector<shared_ptr<EffectiveHamiltonian<typename T1::s_type>>> h_effs =
        xx->h_effs;
    vector<double> coeffs = xx->coeffs;
    h_effs.insert(h_effs.end(), yy->h_effs.begin(), yy->h_effs.end());
    coeffs.insert(coeffs.end(), yy->coeffs.begin(), yy->coeffs.end());
    return make_shared<LinearEffectiveHamiltonian<typename T1::s_type>>(h_effs,
                                                                        coeffs);
}

template <typename T1, typename T2>
inline shared_ptr<LinearEffectiveHamiltonian<typename T1::s_type>>
operator-(const shared_ptr<T1> &x, const shared_ptr<T2> &y) {
    return x + (-1.0) * y;
}

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
    vector<unordered_map<
        S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
        wfn_infos;
    vector<S> operator_quanta;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S>> &op,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &bra,
        const vector<shared_ptr<SparseMatrixGroup<S>>> &ket,
        const shared_ptr<OpElement<S>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat,
        const shared_ptr<TensorFunctions<S>> &ptf, bool compute_diag = true)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(ptf->copy()), hop_mat(hop_mat),
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
        operator_quanta = msl;
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S>::get_uniq_sub_labels(op->mat, hop_mat, msl);
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
                shared_ptr<SparseMatrix<S>> xdiag = (*diag)[i];
                tf->tensor_product_diagonal(op->mat->data[0], op->lopt,
                                            op->ropt, xdiag, opdq);
                diag_info->deallocate();
            }
        }
        // temp wavefunction
        cmat = make_shared<SparseMatrixGroup<S>>();
        vmat = make_shared<SparseMatrixGroup<S>>();
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
            cmat->data = vmat->data = (double *)0;
            tf->tensor_product_multi_multiply(op->mat->data[0], op->lopt,
                                              op->ropt, cmat, vmat,
                                              wfn_infos[0], opdq, 1.0, false);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        } else if (tf->opf->seq->mode & SeqTypes::Tasked) {
            cmat->data = vmat->data = (double *)0;
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
    shared_ptr<SparseMatrixGroup<S>>
    perturbative_noise(bool trace_right, int iL, int iR, FuseTypes ftype,
                       const shared_ptr<MPSInfo<S>> &mps_info,
                       const vector<double> &weights,
                       const NoiseTypes noise_type,
                       const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(mps_info->get_type() & MPSTypes::MultiWfn);
        shared_ptr<MultiMPSInfo<S>> minfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(mps_info);
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        assert(msl.size() == 1 && msl[0] == opdq);
        shared_ptr<OpExpr<S>> pexpr = op->mat->data[0];
        shared_ptr<Symbolic<S>> pmat = make_shared<SymbolicColumnVector<S>>(
            1, vector<shared_ptr<OpExpr<S>>>{pexpr});
        vector<pair<uint8_t, S>> psubsl = Partition<S>::get_uniq_sub_labels(
            pmat, hop_mat, msl, true, trace_right, false)[0];
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
        shared_ptr<SparseMatrixGroup<S>> perturb_ket =
            make_shared<SparseMatrixGroup<S>>(d_alloc);
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
        frame->activate(0);
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
                double ket_norm = (*ket[ii])[i]->norm();
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
                       (size_t)numeric_limits<decltype(MatrixRef::n)>::max());
                tf->opf->seq->auto_perform(MatrixRef(
                    perturb_ket->data, (MKL_INT)perturb_ket->total_memory, 1));
            } else {
                vector<MatrixRef> pmats(perturb_ket->n,
                                        MatrixRef(nullptr, 0, 0));
                for (int j = 0; j < perturb_ket->n; j++)
                    pmats[j] =
                        MatrixRef((*perturb_ket)[j]->data,
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
        S idx_opdq = dynamic_pointer_cast<OpElement<S>>(op->dops[idx])->q_label;
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
    tuple<vector<double>, int, size_t, double>
    eigs(bool iprint = false, double conv_thrd = 5E-6, int max_iter = 5000,
         DavidsonTypes davidson_type = DavidsonTypes::Normal, double shift = 0,
         const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int ndav = 0;
        assert(compute_diag);
        DiagonalMatrix aa(diag->data, (MKL_INT)diag->total_memory);
        vector<MatrixRef> bs;
        for (int i = 0; i < (int)min((MKL_INT)ket.size(), (MKL_INT)aa.n); i++)
            bs.push_back(
                MatrixRef(ket[i]->data, (MKL_INT)ket[i]->total_memory, 1));
        frame->activate(0);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        vector<double> eners =
            (tf->opf->seq->mode == SeqTypes::Auto ||
             (tf->opf->seq->mode & SeqTypes::Tasked))
                ? MatrixFunctions::harmonic_davidson(
                      *tf, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter)
                : MatrixFunctions::harmonic_davidson(
                      *this, aa, bs, shift, davidson_type, ndav, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter);
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners, ndav, (size_t)nflop, t.get_time());
    }
    shared_ptr<OpExpr<S>>
    add_const_term(double const_e,
                   const shared_ptr<ParallelRule<S>> &para_rule) {
        shared_ptr<OpExpr<S>> expr = op->mat->data[0];
        if (const_e != 0) {
            // q_label does not matter
            shared_ptr<OpExpr<S>> iop = make_shared<OpElement<S>>(
                OpNames::I, SiteIndex(),
                dynamic_pointer_cast<OpElement<S>>(op->dops[0])->q_label);
            if (para_rule == nullptr || para_rule->is_root()) {
                if (op->lopt->get_type() == OperatorTensorTypes::Delayed ||
                    op->ropt->get_type() == OperatorTensorTypes::Delayed) {
                    bool dleft =
                        op->lopt->get_type() == OperatorTensorTypes::Delayed;
                    shared_ptr<DelayedOperatorTensor<S>> dopt =
                        dynamic_pointer_cast<DelayedOperatorTensor<S>>(
                            dleft ? op->lopt : op->ropt);
                    shared_ptr<OpElement<S>> xiop =
                        dynamic_pointer_cast<OpElement<S>>(iop);
                    if (dopt->lopt->ops.count(iop) != 0 &&
                        dopt->ropt->ops.count(iop) != 0)
                        op->mat->data[0] =
                            expr +
                            (shared_ptr<OpExpr<S>>)make_shared<OpSumProd<S>>(
                                xiop, xiop,
                                vector<shared_ptr<OpElement<S>>>{xiop, xiop},
                                vector<bool>{false, false}, const_e, 0);
                    else
                        op->mat->data[0] = expr + const_e * (iop * iop);
                } else
                    op->mat->data[0] = expr + const_e * (iop * iop);
            }
        }
        return expr;
    }
    // X = < [bra] | [H_eff] | [ket] >
    // expectations, nflop, tmult
    tuple<vector<pair<shared_ptr<OpExpr<S>>, vector<double>>>, size_t, double>
    expect(double const_e, ExpectationAlgorithmTypes algo_type,
           ExpectationTypes ex_type,
           const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        shared_ptr<OpExpr<S>> expr = nullptr;
        if (const_e != 0 && op->mat->data.size() > 0)
            expr = add_const_term(const_e, para_rule);
        Timer t;
        t.get_time();
        MatrixRef ktmp(nullptr, (MKL_INT)ket[0]->total_memory, 1);
        MatrixRef rtmp(nullptr, (MKL_INT)bra[0]->total_memory, 1);
        MatrixRef btmp(nullptr, (MKL_INT)bra[0]->total_memory, 1);
        btmp.allocate();
        SeqTypes mode = tf->opf->seq->mode;
        tf->opf->seq->mode = tf->opf->seq->mode & SeqTypes::Simple
                                 ? SeqTypes::Simple
                                 : SeqTypes::None;
        tf->opf->seq->cumulative_nflop = 0;
        vector<pair<shared_ptr<OpExpr<S>>, vector<double>>> expectations;
        expectations.reserve(op->mat->data.size());
        vector<double> results;
        vector<size_t> results_idx;
        results.reserve(op->mat->data.size() * ket.size());
        results_idx.reserve(op->mat->data.size());
        if (para_rule != nullptr)
            para_rule->set_partition(ParallelRulePartitionTypes::Middle);
        for (size_t i = 0; i < op->mat->data.size(); i++) {
            vector<double> rr(ket.size(), 0);
            if (dynamic_pointer_cast<OpElement<S>>(op->dops[i])->name ==
                OpNames::Zero)
                continue;
            else if (ex_type == ExpectationTypes::Real) {
                if (para_rule == nullptr || !para_rule->number(op->dops[i])) {
                    for (int j = 0; j < (int)ket.size(); j++) {
                        ktmp.data = ket[j]->data;
                        rtmp.data = bra[j]->data;
                        btmp.clear();
                        (*this)(ktmp, btmp, (int)i, 1.0, true);
                        rr[j] = MatrixFunctions::dot(btmp, rtmp);
                    }
                } else {
                    if (para_rule->own(op->dops[i])) {
                        for (int j = 0; j < (int)ket.size(); j++) {
                            ktmp.data = ket[j]->data;
                            rtmp.data = bra[j]->data;
                            btmp.clear();
                            (*this)(ktmp, btmp, (int)i, 1.0, false);
                            rr[j] = MatrixFunctions::dot(btmp, rtmp);
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
                MatrixRef itmp(nullptr, (MKL_INT)bra[1]->total_memory, 1);
                if (para_rule == nullptr || !para_rule->number(op->dops[i]) ||
                    para_rule->own(op->dops[i])) {
                    rtmp.data = bra[0]->data;
                    itmp.data = bra[1]->data;
                    ktmp.data = ket[0]->data;
                    btmp.clear();
                    (*this)(ktmp, btmp, (int)i, 1.0, true);
                    rr[0] = MatrixFunctions::dot(btmp, rtmp);
                    rr[1] = -MatrixFunctions::dot(btmp, itmp);
                    ktmp.data = ket[1]->data;
                    btmp.clear();
                    (*this)(ktmp, btmp, (int)i, 1.0, true);
                    rr[1] += MatrixFunctions::dot(btmp, rtmp);
                    rr[0] += MatrixFunctions::dot(btmp, itmp);
                }
                if (para_rule != nullptr && para_rule->number(op->dops[i])) {
                    results.insert(results.end(), rr.begin(), rr.end());
                    results_idx.push_back(expectations.size());
                }
                expectations.push_back(make_pair(op->dops[i], rr));
            } else
                assert(false);
        }
        btmp.deallocate();
        if (const_e != 0 && op->mat->data.size() > 0)
            op->mat->data[0] = expr;
        if (results.size() != 0) {
            assert(para_rule != nullptr);
            para_rule->comm->allreduce_sum(results.data(), results.size());
            for (size_t i = 0; i < results.size(); i += ket.size())
                memcpy(expectations[results_idx[i]].second.data(),
                       results.data() + i, sizeof(double) * ket.size());
        }
        tf->opf->seq->mode = mode;
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, (size_t)nflop, t.get_time());
    }
    // [ket] = exp( [H_eff] ) | [ket] > (RK4 approximation)
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<MatrixRef>, tuple<double, double, int, size_t, double>>
    rk4_apply(complex<double> beta, double const_e, bool eval_energy = false,
              const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(ket.size() == 2);
        MatrixRef vr(ket[0]->data, (MKL_INT)ket[0]->total_memory, 1);
        MatrixRef vi(ket[1]->data, (MKL_INT)ket[1]->total_memory, 1);
        vector<MatrixRef> k, r;
        Timer t;
        t.get_time();
        frame->activate(1);
        for (int i = 0; i < 3; i++) {
            r.push_back(MatrixRef(nullptr, (MKL_INT)ket[0]->total_memory, 1));
            r[i + i].allocate();
            r.push_back(MatrixRef(nullptr, (MKL_INT)ket[1]->total_memory, 1));
            r[i + i + 1].allocate();
        }
        frame->activate(0);
        for (int i = 0; i < 4; i++) {
            k.push_back(MatrixRef(nullptr, (MKL_INT)ket[0]->total_memory, 1));
            k[i + i].allocate(), k[i + i].clear();
            k.push_back(MatrixRef(nullptr, (MKL_INT)ket[1]->total_memory, 1));
            k[i + i + 1].allocate(), k[i + i + 1].clear();
        }
        tf->opf->seq->cumulative_nflop = 0;
        const vector<double> ks = vector<double>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<double>> cs = vector<vector<double>>{
            vector<double>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0,
                           -5.0 / 162.0},
            vector<double>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<double>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        precompute();
        const function<void(const MatrixRef &, const MatrixRef &,
                            const MatrixRef &, const MatrixRef &,
                            complex<double>)> &f =
            [this](const MatrixRef &are, const MatrixRef &aim,
                   const MatrixRef &bre, const MatrixRef &bim,
                   complex<double> scale) {
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
                MatrixFunctions::copy(r[0], vr);
                MatrixFunctions::copy(r[1], vi);
                MatrixFunctions::iadd(r[0], k[i + i - 2], ks[i]);
                MatrixFunctions::iadd(r[1], k[i + i - 1], ks[i]);
                f(r[0], r[1], k[i + i], k[i + i + 1], beta);
            }
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            complex<double> factor =
                exp(beta * (double)((i + 1) / 3) * const_e);
            r[i + i].clear(), r[i + i + 1].clear();
            if (factor.real() != 0) {
                MatrixFunctions::iadd(r[i + i], vr, factor.real());
                MatrixFunctions::iadd(r[i + i + 1], vi, factor.real());
            }
            if (factor.imag() != 0) {
                MatrixFunctions::iadd(r[i + i], vi, factor.imag());
                MatrixFunctions::iadd(r[i + i + 1], vr, -factor.imag());
            }
            for (size_t j = 0; j < 4; j++) {
                if (factor.real() != 0) {
                    MatrixFunctions::iadd(r[i + i], k[j + j],
                                          cs[i][j] * factor.real());
                    MatrixFunctions::iadd(r[i + i + 1], k[j + j + 1],
                                          cs[i][j] * factor.real());
                }
                if (factor.imag() != 0) {
                    MatrixFunctions::iadd(r[i + i], k[j + j + 1],
                                          cs[i][j] * factor.imag());
                    MatrixFunctions::iadd(r[i + i + 1], k[j + j],
                                          -cs[i][j] * factor.imag());
                }
            }
        }
        double norm_re = MatrixFunctions::norm(r[2 + 2]);
        double norm_im = MatrixFunctions::norm(r[2 + 2 + 1]);
        double norm = sqrt(norm_re * norm_re + norm_im * norm_im);
        double energy = -const_e;
        if (eval_energy) {
            k[0].clear();
            k[1].clear();
            f(r[2 + 2], r[2 + 2 + 1], k[0], k[1], 1.0);
            energy = (MatrixFunctions::dot(r[2 + 2], k[0]) +
                      MatrixFunctions::dot(r[2 + 2 + 1], k[1])) /
                     (norm * norm);
        }
        for (int i = 3; i >= 0; i--)
            k[i + i + 1].deallocate(), k[i + i].deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(r, make_tuple(energy, norm, 4 + eval_energy,
                                       (size_t)nflop, t.get_time()));
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    // nexpo is number of complex matrix multiplications
    tuple<double, double, int, size_t, double>
    expo_apply(complex<double> beta, double const_e, bool iprint = false,
               const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(compute_diag);
        assert(ket.size() == 2);
        double anorm = MatrixFunctions::norm(
            MatrixRef(diag->data, (MKL_INT)diag->total_memory, 1));
        MatrixRef vr(ket[0]->data, (MKL_INT)ket[0]->total_memory, 1);
        MatrixRef vi(ket[1]->data, (MKL_INT)ket[1]->total_memory, 1);
        Timer t;
        t.get_time();
        tf->opf->seq->cumulative_nflop = 0;
        precompute();
        int nexpo = (tf->opf->seq->mode == SeqTypes::Auto ||
                     (tf->opf->seq->mode & SeqTypes::Tasked))
                        ? ComplexMatrixFunctions::expo_apply(
                              *tf, beta, anorm, vr, vi, const_e, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm)
                        : ComplexMatrixFunctions::expo_apply(
                              *this, beta, anorm, vr, vi, const_e, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm);
        double norm_re = MatrixFunctions::norm(vr);
        double norm_im = MatrixFunctions::norm(vi);
        double norm = sqrt(norm_re * norm_re + norm_im * norm_im);
        MatrixRef tmp_re(nullptr, (MKL_INT)ket[0]->total_memory, 1);
        MatrixRef tmp_im(nullptr, (MKL_INT)ket[1]->total_memory, 1);
        tmp_re.allocate();
        tmp_im.allocate();
        tmp_re.clear();
        tmp_im.clear();
        if (tf->opf->seq->mode == SeqTypes::Auto ||
            (tf->opf->seq->mode & SeqTypes::Tasked))
            (*tf)(vr, tmp_re), (*tf)(vi, tmp_im);
        else
            (*this)(vr, tmp_re), (*this)(vi, tmp_im);
        double energy = (MatrixFunctions::dot(vr, tmp_re) +
                         MatrixFunctions::dot(vi, tmp_im)) /
                        (norm * norm);
        tmp_im.deallocate();
        tmp_re.deallocate();
        post_precompute();
        uint64_t nflop = tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, (size_t)nflop, t.get_time());
    }
    void deallocate() {
        frame->activate(0);
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
