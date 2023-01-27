
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

#include "../core/archived_tensor_functions.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/tensor_functions.hpp"
#include "determinant.hpp"
#include "effective_hamiltonian.hpp"
#include "mpo.hpp"
#include "mps.hpp"
#include "parallel_mpo.hpp"
#include "parallel_mps.hpp"
#include "partition.hpp"
#include "state_averaged.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

enum struct DecompositionTypes : uint8_t {
    SVD = 0,
    PureSVD = 1,
    DensityMatrix = 2
};

enum struct TruncationTypes : ubond_t {
    Physical = 0,
    Reduced = 1,
    ReducedInversed = 2,
    RealDensityMatrix = 4,
    SpectraWithMultiplicity = 8,
    KeepOne = 16,
};

enum struct OpCachingTypes : ubond_t {
    None = 0,
    Left = 1,
    Right = 2,
    LeftCopy = 3,
    RightCopy = 4
};

inline TruncationTypes operator*(TruncationTypes a, ubond_t b) {
    return TruncationTypes((ubond_t)a * b);
}

inline TruncationTypes operator|(TruncationTypes a, TruncationTypes b) {
    return TruncationTypes((ubond_t)a | (ubond_t)b);
}

inline ubond_t operator&(TruncationTypes a, TruncationTypes b) {
    return (ubond_t)a & (ubond_t)b;
}

template <typename, typename, typename> struct ComplexMixture;

template <typename S, typename FL> struct ComplexMixture<S, FL, FL> {
    static shared_ptr<SparseMatrix<S, FL>>
    forward(shared_ptr<SparseMatrix<S, FL>> mat) {
        return mat;
    }
    static shared_ptr<SparseMatrixGroup<S, FL>>
    forward(shared_ptr<SparseMatrixGroup<S, FL>> wfn) {
        return wfn;
    }
    static vector<shared_ptr<SparseMatrixGroup<S, FL>>>
    forward(const vector<shared_ptr<SparseMatrixGroup<S, FL>>> &wfns) {
        return wfns;
    }
};

template <typename S, typename FP>
struct ComplexMixture<S, complex<FP>, FP> : ComplexMixture<S, FP, FP> {
    typedef complex<FP> FL;
    static shared_ptr<SparseMatrix<S, FL>>
    forward(shared_ptr<SparseMatrix<S, FP>> mat) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(mat->get_type() == SparseMatrixTypes::Normal);
        shared_ptr<SparseMatrix<S, FL>> cmat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        cmat->allocate(mat->info);
        cmat->factor = mat->factor;
        GMatrixFunctions<FL>::fill_complex(
            GMatrix<FL>(cmat->data, cmat->total_memory, 1),
            GMatrix<FP>(mat->data, mat->total_memory, 1),
            GMatrix<FP>(nullptr, mat->total_memory, 1));
        return cmat;
    }
    static shared_ptr<SparseMatrixGroup<S, FL>>
    forward(shared_ptr<SparseMatrixGroup<S, FP>> wfn) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrixGroup<S, FL>> cwfn =
            make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
        cwfn->allocate(wfn->infos);
        GMatrixFunctions<FL>::fill_complex(
            GMatrix<FL>(cwfn->data, cwfn->total_memory, 1),
            GMatrix<FP>(wfn->data, wfn->total_memory, 1),
            GMatrix<FP>(nullptr, wfn->total_memory, 1));
        return cwfn;
    }
    static vector<shared_ptr<SparseMatrixGroup<S, FL>>>
    forward(const vector<shared_ptr<SparseMatrixGroup<S, FP>>> wfns) {
        vector<shared_ptr<SparseMatrixGroup<S, FL>>> cwfns(wfns.size() / 2,
                                                           nullptr);
        for (size_t i = 0; i < cwfns.size(); i++) {
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            cwfns[i] = make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
            cwfns[i]->allocate(wfns[i + i]->infos);
            GMatrixFunctions<FL>::fill_complex(
                GMatrix<FL>(cwfns[i]->data, cwfns[i]->total_memory, 1),
                GMatrix<FP>(wfns[i + i]->data, wfns[i + i]->total_memory, 1),
                GMatrix<FP>(wfns[i + i + 1]->data,
                            wfns[i + i + 1]->total_memory, 1));
        }
        return cwfns;
    }
};

// A tensor network < bra | mpo | ket >
template <typename S, typename FL, typename FLS> struct MovingEnvironment {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FLS>::FP FPS;
    typedef typename const_fl_type<FLS>::FL FLLS;
    typedef typename const_fl_type<typename GMatrix<FLS>::FP>::FL FPLS;
    typedef typename GMatrix<FLS>::FC FCS;
    int n_sites, center, dot;
    shared_ptr<MPO<S, FL>> mpo;
    shared_ptr<MPS<S, FLS>> bra, ket;
    // Represent the environments contracted around different center sites
    vector<shared_ptr<Partition<S, FL>>> envs;
    // Symbol of the whole-block operator the MPO represents
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    // Tag is used to generate filename for disk storage
    string tag;
    // Parallel execution control
    shared_ptr<ParallelRule<S>> para_rule;
    // cached contracted opt for reuse in rotation
    shared_ptr<OperatorTensor<S, FL>> cached_opt = nullptr;
    // info for cached opt
    pair<OpCachingTypes, int> cached_info = make_pair(OpCachingTypes::None, -1);
    // whether caching contracted opt (only available when
    // !frame_<FP>()->use_main_stack)
    bool cached_contraction = false;
    // whether contraction and rotation should be done within one-step, without
    // using large memory for blocking (only saving memory when no explicit
    // left/right_contact is invoked, which is the case for zero-dot expt)
    // fused_contraction_rotation = T conflicts with cached_contraction = T
    bool fused_contraction_rotation = false;
    double tctr = 0, trot = 0, tint = 0, tmid = 0, tdiag = 0, tdctr = 0,
           tinfo = 0;
    Timer _t, _t2, _t3;
    bool iprint = false;
    bool save_partition_info = false;
    OpNamesSet delayed_contraction = OpNamesSet();
    int fuse_center;
    MovingEnvironment(const shared_ptr<MPO<S, FL>> &mpo,
                      const shared_ptr<MPS<S, FLS>> &bra,
                      const shared_ptr<MPS<S, FLS>> &ket,
                      const string &tag = "DMRG")
        : n_sites(ket->n_sites), center(ket->center), dot(ket->dot), mpo(mpo),
          bra(bra), ket(ket), tag(tag), para_rule(nullptr) {
        assert(bra->n_sites == ket->n_sites && mpo->n_sites == ket->n_sites);
        assert(bra->center == ket->center && bra->dot == ket->dot);
        hop_mat = make_shared<SymbolicColumnVector<S>>(1);
        (*hop_mat)[0] = mpo->op;
        fuse_center = mpo->schemer == nullptr ? n_sites - 2
                                              : mpo->schemer->left_trans_site;
        if (mpo->get_parallel_type() & ParallelTypes::Distributed) {
            if (mpo->get_parallel_type() & ParallelTypes::NewScheme)
                para_rule = dynamic_pointer_cast<ParallelMPO<S, FL>>(mpo)->rule;
            else
                para_rule =
                    dynamic_pointer_cast<ClassicParallelMPO<S, FL>>(mpo)->rule;
            para_rule->comm->barrier();
        }
        if (ket->get_type() & MPSTypes::MultiCenter) {
            save_partition_info = true;
            if (dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket)->rule != nullptr)
                dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket)
                    ->rule->comm->barrier();
        }
    }
    virtual ~MovingEnvironment() = default;
    // Contract and renormalize left block by one site
    // new site = i - 1
    // return <intmed memory, rotated renormalized op memory>
    pair<size_t, size_t> left_contract_rotate(int i,
                                              bool preserve_data = false) {
        mpo->load_left_operators(i - 1);
        mpo->load_tensor(i - 1);
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos_notrunc;
        _t.get_time();
        vector<shared_ptr<Symbolic<S>>> mats = {
            mpo->left_operator_names[i - 1]};
        if (mpo->schemer != nullptr && i - 1 == mpo->schemer->left_trans_site) {
            mpo->load_schemer();
            mats.push_back(mpo->schemer->left_new_operator_names);
        }
        vector<S> sl = Partition<S, FL>::get_uniq_labels(mats);
        shared_ptr<Symbolic<S>> exprs =
            envs[i - 1]->left == nullptr
                ? nullptr
                : (mpo->left_operator_exprs.size() != 0
                       ? mpo->left_operator_exprs[i - 1]
                       : envs[i - 1]->left->lmat * mpo->tensors[i - 1]->lmat);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S, FL>::get_uniq_sub_labels(
                exprs, mpo->left_operator_names[i - 1], sl, mpo->left_vacuum);
        Partition<S, FL>::init_left_op_infos_notrunc(
            i - 1, bra->info, ket->info, sl, subsl, envs[i - 1]->left_op_infos,
            mpo->site_op_infos[i - 1], left_op_infos_notrunc, mpo->tf->opf->cg);
        frame_<FP>()->activate(0);
        shared_ptr<OperatorTensor<S, FL>> new_left;
        if (cached_info.first == OpCachingTypes::Left &&
            cached_info.second == i - 1) {
            new_left = cached_opt;
            for (auto &p : new_left->ops)
                p.second->info = Partition<S, FL>::find_op_info(
                    left_op_infos_notrunc, p.second->info->delta_quantum);
        } else
            new_left = Partition<S, FL>::build_left(
                {mpo->left_operator_names[i - 1]}, left_op_infos_notrunc,
                mpo->sparse_form[i - 1] == 'S');
        tinfo += _t.get_time();
        if (mpo->tf->get_type() == TensorFunctionsTypes::Archived) {
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->filename = get_middle_archive_filename();
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->offset = 0;
        }
        shared_ptr<OperatorTensor<S, FL>> copied_left = nullptr;
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> copied_infos;
        size_t copied_mem = 0;
        if (fused_contraction_rotation) {
            if (envs[i - 1]->left != nullptr) {
                left_copy(i - 1, copied_infos, copied_left, false);
                copied_mem = copied_left->get_total_memory();
            }
        } else {
            // cached_opt might be partially delayed,
            // so further contraction is still needed
            mpo->tf->left_contract(envs[i - 1]->left, mpo->tensors[i - 1],
                                   new_left,
                                   mpo->left_operator_exprs.size() != 0
                                       ? mpo->left_operator_exprs[i - 1]
                                       : nullptr);
            mpo->unload_tensor(i - 1);
            mpo->unload_left_operators(i - 1);
        }
        tctr += _t.get_time();
        bra->load_tensor(i - 1);
        if (bra != ket)
            ket->load_tensor(i - 1);
        frame_<FP>()->reset(1);
        Partition<S, FL>::init_left_op_infos(i - 1, bra->info, ket->info, sl,
                                             envs[i]->left_op_infos);
        frame_<FP>()->activate(1);
        envs[i]->left =
            Partition<S, FL>::build_left(mats, envs[i]->left_op_infos);
        tinfo += _t.get_time();
        if (mpo->tf->get_type() == TensorFunctionsTypes::Archived) {
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->filename = get_left_archive_filename(i);
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->offset = 0;
        }
        shared_ptr<SparseMatrix<S, FL>> fbt =
            ComplexMixture<S, FL, FLS>::forward(bra->tensors[i - 1]);
        shared_ptr<SparseMatrix<S, FL>> fkt =
            bra == ket
                ? fbt
                : ComplexMixture<S, FL, FLS>::forward(ket->tensors[i - 1]);
        if (!fused_contraction_rotation)
            mpo->tf->left_rotate(new_left, fbt, fkt, envs[i]->left);
        else {
            mpo->tf->left_contract_rotate(copied_left, mpo->tensors[i - 1], fbt,
                                          fkt, new_left, envs[i]->left,
                                          mpo->left_operator_exprs.size() != 0
                                              ? mpo->left_operator_exprs[i - 1]
                                              : nullptr);
            mpo->unload_tensor(i - 1);
            mpo->unload_left_operators(i - 1);
            copied_left = nullptr;
        }
        size_t blocking_mem = new_left->get_total_memory() + copied_mem;
        size_t renormal_mem = envs[i]->left->get_total_memory();
        if (!frame_<FP>()->use_main_stack)
            new_left->deallocate();
        trot += _t.get_time();
        if (mpo->schemer != nullptr && i - 1 == mpo->schemer->left_trans_site) {
            mpo->tf->numerical_transform(envs[i]->left, mats[1],
                                         mpo->schemer->left_new_operator_exprs);
            mpo->unload_schemer();
            frame_<FP>()->update_peak_used_memory();
            // when using conventional mpo transform scheme, dot = 1/2 only
            // compatible if we keep both N/C for dot = 1
            // also in tdvp (1-dot after 2-dot) we need to keep data
            if (dot == 2 && !preserve_data)
                mpo->tf->post_numerical_transform(envs[i]->left, mats[0],
                                                  mats[1]);
        }
        tmid += _t.get_time();
        if (i < mpo->left_operator_exprs.size()) {
            mpo->load_left_operators(i);
            mpo->tf->intermediates(mpo->left_operator_names[i],
                                   mpo->left_operator_exprs[i], envs[i]->left,
                                   true);
            mpo->unload_left_operators(i);
        }
        tint += _t.get_time();
        frame_<FP>()->activate(0);
        if (bra != ket)
            ket->unload_tensor(i - 1);
        bra->unload_tensor(i - 1);
        if (frame_<FP>()->use_main_stack)
            new_left->deallocate();
        Partition<S, FL>::deallocate_op_infos_notrunc(left_op_infos_notrunc);
        frame_<FP>()->save_data(1, get_left_partition_filename(i));
        if (save_partition_info) {
            frame_<FP>()->activate(1);
            envs[i]->save_data(true, get_left_partition_filename(i, true));
            frame_<FP>()->activate(0);
        }
        return make_pair(blocking_mem, renormal_mem);
    }
    // Contract and renormalize right block by one site
    // new site = i + dot
    // return <intmed memory, rotated renormalized op memory>
    pair<size_t, size_t> right_contract_rotate(int i,
                                               bool preserve_data = false) {
        mpo->load_right_operators(i + dot);
        mpo->load_tensor(i + dot);
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> right_op_infos_notrunc;
        _t.get_time();
        vector<shared_ptr<Symbolic<S>>> mats = {
            mpo->right_operator_names[i + dot]};
        if (mpo->schemer != nullptr &&
            i + dot == mpo->schemer->right_trans_site) {
            mpo->load_schemer();
            mats.push_back(mpo->schemer->right_new_operator_names);
        }
        vector<S> sl = Partition<S, FL>::get_uniq_labels(mats);
        shared_ptr<Symbolic<S>> exprs =
            envs[i + 1]->right == nullptr
                ? nullptr
                : (mpo->right_operator_exprs.size() != 0
                       ? mpo->right_operator_exprs[i + dot]
                       : mpo->tensors[i + dot]->rmat *
                             envs[i + 1]->right->rmat);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S, FL>::get_uniq_sub_labels(
                exprs, mpo->right_operator_names[i + dot], sl,
                ket->info->vacuum);
        Partition<S, FL>::init_right_op_infos_notrunc(
            i + dot, bra->info, ket->info, sl, subsl,
            envs[i + 1]->right_op_infos, mpo->site_op_infos[i + dot],
            right_op_infos_notrunc, mpo->tf->opf->cg);
        frame_<FP>()->activate(0);
        shared_ptr<OperatorTensor<S, FL>> new_right;
        if (cached_info.first == OpCachingTypes::Right &&
            cached_info.second == i + dot) {
            new_right = cached_opt;
            for (auto &p : new_right->ops)
                p.second->info = Partition<S, FL>::find_op_info(
                    right_op_infos_notrunc, p.second->info->delta_quantum);
        } else
            new_right = Partition<S, FL>::build_right(
                {mpo->right_operator_names[i + dot]}, right_op_infos_notrunc,
                mpo->sparse_form[i + dot] == 'S');
        tinfo += _t.get_time();
        if (mpo->tf->get_type() == TensorFunctionsTypes::Archived) {
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->filename = get_middle_archive_filename();
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->offset = 0;
        }
        shared_ptr<OperatorTensor<S, FL>> copied_right = nullptr;
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> copied_infos;
        size_t copied_mem = 0;
        if (fused_contraction_rotation) {
            if (envs[i + 1]->right != nullptr) {
                right_copy(i + dot, copied_infos, copied_right, false);
                copied_mem = copied_right->get_total_memory();
            }
        } else {
            // cached_opt might be partially delayed,
            // so further contraction is still needed
            mpo->tf->right_contract(envs[i + 1]->right, mpo->tensors[i + dot],
                                    new_right,
                                    mpo->right_operator_exprs.size() != 0
                                        ? mpo->right_operator_exprs[i + dot]
                                        : nullptr);
            mpo->unload_tensor(i + dot);
            mpo->unload_right_operators(i + dot);
        }
        tctr += _t.get_time();
        bra->load_tensor(i + dot);
        if (bra != ket)
            ket->load_tensor(i + dot);
        frame_<FP>()->reset(1);
        Partition<S, FL>::init_right_op_infos(i + dot, bra->info, ket->info, sl,
                                              envs[i]->right_op_infos);
        frame_<FP>()->activate(1);
        envs[i]->right =
            Partition<S, FL>::build_right(mats, envs[i]->right_op_infos);
        tinfo += _t.get_time();
        if (mpo->tf->get_type() == TensorFunctionsTypes::Archived) {
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->filename = get_right_archive_filename(i);
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->offset = 0;
        }
        shared_ptr<SparseMatrix<S, FL>> fbt =
            ComplexMixture<S, FL, FLS>::forward(bra->tensors[i + dot]);
        shared_ptr<SparseMatrix<S, FL>> fkt =
            bra == ket
                ? fbt
                : ComplexMixture<S, FL, FLS>::forward(ket->tensors[i + dot]);
        if (!fused_contraction_rotation)
            mpo->tf->right_rotate(new_right, fbt, fkt, envs[i]->right);
        else {
            mpo->tf->right_contract_rotate(
                copied_right, mpo->tensors[i + dot], fbt, fkt, new_right,
                envs[i]->right,
                mpo->right_operator_exprs.size() != 0
                    ? mpo->right_operator_exprs[i + dot]
                    : nullptr);
            mpo->unload_tensor(i + dot);
            mpo->unload_right_operators(i + dot);
            copied_right = nullptr;
        }
        size_t blocking_mem = new_right->get_total_memory() + copied_mem;
        size_t renormal_mem = envs[i]->right->get_total_memory();
        if (!frame_<FP>()->use_main_stack)
            new_right->deallocate();
        trot += _t.get_time();
        if (mpo->schemer != nullptr &&
            i + dot == mpo->schemer->right_trans_site) {
            mpo->tf->numerical_transform(
                envs[i]->right, mats[1],
                mpo->schemer->right_new_operator_exprs);
            mpo->unload_schemer();
            frame_<FP>()->update_peak_used_memory();
            // when using conventional mpo transform scheme, dot = 1/2 only
            // compatible if we keep both N/C for dot = 1
            // also in tdvp (1-dot after 2-dot) we need to keep data
            if (dot == 2 && !preserve_data)
                mpo->tf->post_numerical_transform(envs[i]->right, mats[0],
                                                  mats[1]);
        }
        tmid += _t.get_time();
        if (i + dot - 1 >= 0 &&
            i + dot - 1 < mpo->right_operator_exprs.size()) {
            mpo->load_right_operators(i + dot - 1);
            mpo->tf->intermediates(mpo->right_operator_names[i + dot - 1],
                                   mpo->right_operator_exprs[i + dot - 1],
                                   envs[i]->right, false);
            mpo->unload_right_operators(i + dot - 1);
        }
        tint += _t.get_time();
        frame_<FP>()->activate(0);
        if (bra != ket)
            ket->unload_tensor(i + dot);
        bra->unload_tensor(i + dot);
        if (frame_<FP>()->use_main_stack)
            new_right->deallocate();
        Partition<S, FL>::deallocate_op_infos_notrunc(right_op_infos_notrunc);
        frame_<FP>()->save_data(1, get_right_partition_filename(i));
        if (save_partition_info) {
            frame_<FP>()->activate(1);
            envs[i]->save_data(false, get_right_partition_filename(i, true));
            frame_<FP>()->activate(0);
        }
        return make_pair(blocking_mem, renormal_mem);
    }
    void left_contract_rotate_unordered(
        int i, const shared_ptr<ParallelRule<S>> &rule = nullptr) {
        if (i == 0)
            return;
        if (rule == nullptr || rule->is_root()) {
            envs[i]->left_op_infos.clear();
            envs[i]->left = nullptr;
            frame_<FP>()->activate(1);
            if (i - 1 != 0) {
                envs[i - 1]->load_data(
                    true, get_left_partition_filename(i - 1, true));
                if (envs[i - 1]->left != nullptr)
                    frame_<FP>()->load_data(1,
                                            get_left_partition_filename(i - 1));
            }
            left_contract_rotate(i);
        }
        if (rule != nullptr)
            rule->comm->barrier();
        if (rule != nullptr && !rule->is_root()) {
            frame_<FP>()->activate(1);
            envs[i]->load_data(true, get_left_partition_filename(i, true));
            frame_<FP>()->activate(0);
        }
    }
    void right_contract_rotate_unordered(
        int i, const shared_ptr<ParallelRule<S>> &rule = nullptr) {
        if (!(i >= 0 && i + 1 < n_sites))
            return;
        if (rule == nullptr || rule->is_root()) {
            envs[i]->right_op_infos.clear();
            envs[i]->right = nullptr;
            frame_<FP>()->activate(1);
            envs[i + 1]->load_data(false,
                                   get_right_partition_filename(i + 1, true));
            if (envs[i + 1]->right != nullptr)
                frame_<FP>()->load_data(1, get_right_partition_filename(i + 1));
            right_contract_rotate(i);
        }
        if (rule != nullptr)
            rule->comm->barrier();
        if (rule != nullptr && !rule->is_root()) {
            frame_<FP>()->activate(1);
            envs[i]->load_data(false, get_right_partition_filename(i, true));
            frame_<FP>()->activate(0);
        }
    }
    // change from standard single-center MPS to multi-center MPS
    void parallelize_mps() {
        assert(ket->get_type() & MPSTypes::MultiCenter);
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket);
        shared_ptr<CG<S>> cg = mpo->tf->opf->cg;
        if (para_mps->ncenter != 0)
            return;
        assert(para_mps->conn_centers.size() != 0);
        para_mps->ncenter = (int)para_mps->conn_centers.size();
        assert(para_mps->conn_matrices.size() == 0);
        para_mps->conn_matrices.resize(para_mps->ncenter);
        if (para_mps->rule != nullptr)
            para_mps->rule->comm->barrier();
        while (para_mps->center != 0) {
            para_mps->move_left(cg, para_mps->rule);
            right_contract_rotate_unordered(
                para_mps->center - para_mps->dot + 1, para_mps->rule);
        }
        assert(para_mps->center == 0);
        for (int i = 0; i < para_mps->ncenter; i++) {
            while (para_mps->center != para_mps->conn_centers[i]) {
                para_mps->move_right(cg, para_mps->rule);
                left_contract_rotate_unordered(para_mps->center,
                                               para_mps->rule);
            }
            auto rmat = para_mps->para_split(i, para_mps->rule);
            right_contract_rotate_unordered(para_mps->center - para_mps->dot,
                                            para_mps->rule);
            if (para_mps->rule == nullptr || para_mps->rule->is_root()) {
                para_mps->tensors[para_mps->center] = rmat;
                para_mps->save_tensor(para_mps->center);
            }
            if (para_mps->rule != nullptr)
                para_mps->rule->comm->barrier();
        }
        while (para_mps->center != para_mps->n_sites - 1) {
            para_mps->move_right(cg, para_mps->rule);
            left_contract_rotate_unordered(para_mps->center, para_mps->rule);
        }
        para_mps->move_right(cg, para_mps->rule);
        for (int i = 0; i < para_mps->ncenter; i += 2) {
            para_mps->center = i != para_mps->ncenter - 1
                                   ? para_mps->conn_centers[i + 1] - 1
                                   : para_mps->n_sites - 1;
            while (para_mps->center != para_mps->conn_centers[i]) {
                para_mps->move_left(cg, para_mps->rule);
                right_contract_rotate_unordered(
                    para_mps->center - para_mps->dot + 1, para_mps->rule);
            }
        }
        for (int i = 0; para_mps->dot == 2 && i < para_mps->ncenter + 1;
             i += 2) {
            para_mps->center = i != para_mps->ncenter
                                   ? para_mps->conn_centers[i] - 1
                                   : para_mps->n_sites - 1;
            para_mps->flip_fused_form(para_mps->center, cg, para_mps->rule);
        }
        para_mps->center = para_mps->conn_centers[0] - 1;
    }
    // change from multi-center MPS to standard single-center MPS
    void serialize_mps() {
        assert(ket->get_type() & MPSTypes::MultiCenter);
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket);
        shared_ptr<CG<S>> cg = mpo->tf->opf->cg;
        assert(para_mps->conn_matrices.size() != 0);
        if (para_mps->rule != nullptr)
            para_mps->rule->comm->barrier();
        if (para_mps->canonical_form[para_mps->n_sites - 1] == 'C')
            para_mps->canonical_form[para_mps->n_sites - 1] = 'S';
        else if (para_mps->canonical_form[para_mps->n_sites - 1] == 'K')
            para_mps->flip_fused_form(para_mps->n_sites - 1, cg,
                                      para_mps->rule);
        if (para_mps->canonical_form[0] == 'C')
            para_mps->canonical_form[0] = 'K';
        else if (para_mps->canonical_form[0] == 'S')
            para_mps->flip_fused_form(0, cg, para_mps->rule);
        for (int i = 0; i <= para_mps->ncenter; i++) {
            para_mps->center = i == 0 ? 0 : para_mps->conn_centers[i - 1];
            int j = i == para_mps->ncenter ? para_mps->n_sites - 1
                                           : para_mps->conn_centers[i] - 1;
            if (para_mps->canonical_form[para_mps->center] == 'K')
                while (para_mps->center != j) {
                    para_mps->move_right(cg, para_mps->rule);
                    left_contract_rotate_unordered(para_mps->center,
                                                   para_mps->rule);
                }
        }
        para_mps->center = para_mps->n_sites - 1;
        for (int i = para_mps->ncenter - 1; i >= 0; i--) {
            while (para_mps->center != para_mps->conn_centers[i]) {
                para_mps->move_left(cg, para_mps->rule);
                right_contract_rotate_unordered(
                    para_mps->center - para_mps->dot + 1, para_mps->rule);
            }
            para_mps->flip_fused_form(para_mps->center - 1, cg, para_mps->rule);
            para_mps->flip_fused_form(para_mps->center, cg, para_mps->rule);
            para_mps->para_merge(i, para_mps->rule);
        }
        while (para_mps->center != 0) {
            para_mps->move_left(cg, para_mps->rule);
            right_contract_rotate_unordered(
                para_mps->center - para_mps->dot + 1, para_mps->rule);
        }
        center = para_mps->center;
        para_mps->conn_matrices.clear();
        para_mps->ncenter = 0;
    }
    string get_left_archive_filename(int i) const {
        stringstream ss;
        ss << frame_<FP>()->save_dir << "/" << frame_<FP>()->prefix_distri
           << ".AR." << tag << ".LEFT." << Parsing::to_string(i);
        return ss.str();
    }
    string get_middle_archive_filename() const {
        stringstream ss;
        ss << frame_<FP>()->save_dir << "/" << frame_<FP>()->prefix_distri
           << ".AR." << tag << ".MIDDLE." << Parsing::to_string(0);
        return ss.str();
    }
    string get_right_archive_filename(int i) const {
        stringstream ss;
        ss << frame_<FP>()->save_dir << "/" << frame_<FP>()->prefix_distri
           << ".AR." << tag << ".RIGHT." << Parsing::to_string(i);
        return ss.str();
    }
    string get_left_partition_filename(int i, bool info = false) const {
        stringstream ss;
        ss << frame_<FP>()->save_dir << "/" << frame_<FP>()->prefix_distri
           << ".PART." << (info ? "INFO." : "") << tag << ".LEFT."
           << Parsing::to_string(i);
        return ss.str();
    }
    string get_right_partition_filename(int i, bool info = false) const {
        stringstream ss;
        ss << frame_<FP>()->save_dir << "/" << frame_<FP>()->prefix_distri
           << ".PART." << (info ? "INFO." : "") << tag << ".RIGHT."
           << Parsing::to_string(i);
        return ss.str();
    }
    string get_npdm_fragment_filename(int i) const {
        stringstream ss;
        ss << frame_<FP>()->save_dir << "/" << frame_<FP>()->prefix_distri
           << ".PART." << tag << ".NPDM.FRAG." << Parsing::to_string(i);
        return ss.str();
    }
    void shallow_copy_to(const shared_ptr<MovingEnvironment> &me) const {
        for (int i = 0; i < n_sites; i++) {
            me->envs[i] = make_shared<Partition<S, FL>>(*envs[i]);
            me->envs[i]->left_op_infos = envs[i]->left_op_infos;
            me->envs[i]->right_op_infos = envs[i]->right_op_infos;
            if (envs[i]->left != nullptr)
                Parsing::link_file(get_left_partition_filename(i),
                                   me->get_left_partition_filename(i));
            if (envs[i]->right != nullptr)
                Parsing::link_file(get_right_partition_filename(i),
                                   me->get_right_partition_filename(i));
        }
    }
    virtual shared_ptr<MovingEnvironment>
    shallow_copy(const string &new_tag) const {
        shared_ptr<MovingEnvironment> me =
            make_shared<MovingEnvironment>(*this);
        me->tag = new_tag;
        shallow_copy_to(me);
        return me;
    }
    virtual void finalize_environments(bool renormalize_ops = true) {
        if (!(ket->get_type() & MPSTypes::MultiCenter))
            return;
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket);
        shared_ptr<CG<S>> cg = mpo->tf->opf->cg;
        assert(para_mps->conn_matrices.size() != 0);
        para_mps->enable_parallel_writing();
        if (para_mps->rule != nullptr)
            para_mps->rule->comm->barrier();
        if (para_mps->canonical_form[para_mps->n_sites - 1] == 'C')
            para_mps->canonical_form[para_mps->n_sites - 1] = 'S';
        else if (para_mps->canonical_form[para_mps->n_sites - 1] == 'K') {
            if (para_mps->rule == nullptr ||
                para_mps->rule->comm->group ==
                    para_mps->ncenter % para_mps->rule->comm->ngroup)
                para_mps->flip_fused_form(para_mps->n_sites - 1, cg, para_rule);
            para_mps->canonical_form[para_mps->n_sites - 1] = 'S';
        }
        if (para_mps->canonical_form[0] == 'C')
            para_mps->canonical_form[0] = 'K';
        else if (para_mps->canonical_form[0] == 'S') {
            if (para_mps->rule == nullptr ||
                para_mps->rule->comm->group == 0 % para_mps->rule->comm->ngroup)
                para_mps->flip_fused_form(0, cg, para_rule);
            para_mps->canonical_form[0] = 'K';
        }
        vector<int> conn_idxs(para_mps->ncenter);
        for (int i = 0; i < para_mps->ncenter; i++)
            conn_idxs[i] = i;
        while (conn_idxs.size() != 0) {
            if (para_mps->rule != nullptr)
                para_mps->rule->comm->barrier();
            vector<int> new_conn_idxs;
            bool l_form = false;
            bool last_rev =
                para_mps->canonical_form[para_mps->n_sites - 1] == 'S';
            if (para_mps->canonical_form[0] == 'K') {
                int ip = conn_idxs[0];
                l_form = !l_form;
                if (para_mps->rule == nullptr ||
                    para_mps->rule->comm->group ==
                        0 % para_mps->rule->comm->ngroup) {
                    para_mps->center = 0;
                    while (para_mps->center != para_mps->conn_centers[ip] - 1) {
                        para_mps->move_right(cg, para_rule);
                        check_signal_()();
                        if (renormalize_ops)
                            left_contract_rotate_unordered(para_mps->center);
                    }
                }
                for (int i = 0; i < para_mps->conn_centers[ip] - 1; i++)
                    para_mps->canonical_form[i] = 'L';
                para_mps->canonical_form[para_mps->conn_centers[ip] - 1] = 'S';
            }
            for (int ipx = 0; ipx < (int)conn_idxs.size(); ipx++) {
                int ip = conn_idxs[ipx];
                if (para_mps->canonical_form[para_mps->conn_centers[ip]] !=
                        'L' &&
                    para_mps->canonical_form[para_mps->conn_centers[ip]] !=
                        'R') {
                    l_form = !l_form;
                    int pj =
                        ipx == (int)conn_idxs.size() - 1
                            ? n_sites - 1
                            : para_mps->conn_centers[conn_idxs[ipx + 1]] - 1;
                    int pi = ipx == 0
                                 ? 0
                                 : para_mps->conn_centers[conn_idxs[ipx - 1]];
                    if (para_mps->rule == nullptr ||
                        para_mps->rule->comm->group ==
                            (ip + 1) % para_mps->rule->comm->ngroup) {
                        center = para_mps->conn_centers[ip] - 1;
                        if (para_mps->canonical_form[center] == 'C' &&
                            para_mps->canonical_form[center + 1] == 'C')
                            para_mps->canonical_form[center] = 'K',
                            para_mps->canonical_form[center + 1] = 'S';
                        else if (para_mps->canonical_form[center] == 'S' &&
                                 para_mps->canonical_form[center + 1] == 'K') {
                            para_mps->flip_fused_form(center, cg, para_rule);
                            para_mps->flip_fused_form(center + 1, cg,
                                                      para_rule);
                        }
                        assert(para_mps->canonical_form[center] == 'K' &&
                               para_mps->canonical_form[center + 1] == 'S');
                        para_mps->para_merge(ip, para_rule); // LS
                        para_mps->center = para_mps->conn_centers[ip];
                        if (l_form) {
                            para_mps->move_left(cg, para_rule);
                            para_mps->move_right(cg, para_rule);
                            check_signal_()();
                            if (renormalize_ops)
                                left_contract_rotate_unordered(
                                    para_mps->center);
                            while (para_mps->center != pj) {
                                para_mps->move_right(cg, para_rule);
                                check_signal_()();
                                if (renormalize_ops)
                                    left_contract_rotate_unordered(
                                        para_mps->center);
                            }
                        } else
                            while (para_mps->center != pi) {
                                para_mps->move_left(cg, para_rule);
                                check_signal_()();
                                if (renormalize_ops)
                                    right_contract_rotate_unordered(
                                        para_mps->center - para_mps->dot + 1);
                            }
                    }
                    for (int i = pi; i <= pj; i++)
                        para_mps->canonical_form[i] = l_form ? 'L' : 'R';
                    if (l_form)
                        para_mps->canonical_form[pj] = 'S';
                    else
                        para_mps->canonical_form[pi] = 'K';
                } else
                    new_conn_idxs.push_back(ip);
            }
            if (last_rev && l_form) {
                int ip = conn_idxs[conn_idxs.size() - 1];
                l_form = !l_form;
                if (para_mps->rule == nullptr ||
                    para_mps->rule->comm->group ==
                        para_mps->ncenter % para_mps->rule->comm->ngroup) {
                    para_mps->center = para_mps->n_sites - 1;
                    while (para_mps->center != para_mps->conn_centers[ip]) {
                        para_mps->move_left(cg, para_rule);
                        check_signal_()();
                        if (renormalize_ops)
                            right_contract_rotate_unordered(para_mps->center -
                                                            para_mps->dot + 1);
                    }
                }
                for (int i = para_mps->conn_centers[ip] + 1; i < n_sites; i++)
                    para_mps->canonical_form[i] = 'R';
                para_mps->canonical_form[para_mps->conn_centers[ip]] = 'K';
            }
            conn_idxs = new_conn_idxs;
        }
        para_mps->conn_matrices.clear();
        para_mps->conn_centers.clear();
        para_mps->ncenter = 0;
        // for two-site
        para_mps->center = para_mps->n_sites - 2;
        center = para_mps->center;
        if (para_mps->rule != nullptr)
            para_mps->rule->comm->barrier();
        if (para_mps->rule == nullptr || para_mps->rule->comm->group == 0)
            para_mps->save_data();
        if (renormalize_ops) {
            frame_<FP>()->activate(1);
            for (int i = 0; i < n_sites; i++) {
                if (i != 0)
                    envs[i]->load_data(true,
                                       get_left_partition_filename(i, true));
                envs[i]->load_data(false,
                                   get_right_partition_filename(i, true));
            }
            frame_<FP>()->activate(0);
        }
        // outside code may have cout
        para_mps->disable_parallel_writing();
    }
    virtual void init_parallel_environments(
        int pi, int pj,
        const shared_ptr<ParallelCommunicator<S>> &pcomm = nullptr,
        bool init = false) {
        assert(pj >= pi + 2 && pi % 2 == 0);
        assert(ket->get_type() & MPSTypes::MultiCenter);
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket);
        shared_ptr<CG<S>> cg = mpo->tf->opf->cg;
        int pm = (pi + pj) / 2;
        if (pm % 2 != 0 && !(pj == pi + 2))
            pm++;
        if (para_mps->center ==
            (pi == 0 ? 0 : para_mps->conn_centers[pi - 1])) {
            // SRRR -> LLSR
            if (para_mps->rule == nullptr ||
                para_mps->rule->comm->group ==
                    pi % para_mps->rule->comm->ngroup) {
                while (para_mps->center != para_mps->conn_centers[pm - 1]) {
                    para_mps->move_right(cg, para_rule);
                    check_signal_()();
                    if (iprint)
                        cout << "init .. L = " << para_mps->center << endl;
                    left_contract_rotate_unordered(para_mps->center);
                }
            }
            if ((para_mps->rule == nullptr ||
                 para_mps->rule->comm->group ==
                     pm % para_mps->rule->comm->ngroup) &&
                init) {
                for (int i = n_sites - dot - 1;
                     i > para_mps->conn_centers[pm - 1] - para_mps->dot; i--) {
                    check_signal_()();
                    if (iprint)
                        cout << "init .. R = " << i << endl;
                    right_contract_rotate_unordered(i);
                }
            }
        } else {
            // LLLK -> LLKR -> LLSR
            if ((para_mps->rule == nullptr ||
                 para_mps->rule->comm->group ==
                     pi % para_mps->rule->comm->ngroup) &&
                init) {
                for (int i = 1; i <= para_mps->conn_centers[pm - 1]; i++) {
                    check_signal_()();
                    if (iprint)
                        cout << "init .. L = " << para_mps->center << endl;
                    left_contract_rotate_unordered(i);
                }
            }
            if (para_mps->rule == nullptr ||
                para_mps->rule->comm->group ==
                    pm % para_mps->rule->comm->ngroup) {
                while (para_mps->center != para_mps->conn_centers[pm - 1]) {
                    para_mps->move_left(cg, para_rule);
                    check_signal_()();
                    if (iprint)
                        cout << "init .. R = "
                             << para_mps->center - para_mps->dot + 1 << endl;
                    right_contract_rotate_unordered(para_mps->center -
                                                    para_mps->dot + 1);
                }
                para_mps->flip_fused_form(para_mps->center, cg, para_rule);
            }
        }
        for (int i = (pi == 0 ? 0 : para_mps->conn_centers[pi - 1]);
             i < (pj == para_mps->ncenter + 1 ? n_sites
                                              : para_mps->conn_centers[pj - 1]);
             i++) {
            if (para_mps->tensors[i] == nullptr)
                para_mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
            if (i == para_mps->conn_centers[pm - 1])
                para_mps->canonical_form[i] = 'S';
            else if (i < para_mps->conn_centers[pm - 1])
                para_mps->canonical_form[i] = 'L';
            else
                para_mps->canonical_form[i] = 'R';
        }
        if (pcomm != nullptr)
            pcomm->barrier();
        // LLSR -> LKSR
        if (para_mps->rule == nullptr ||
            para_mps->rule->comm->group == pi % para_mps->rule->comm->ngroup) {
            para_mps->center = para_mps->conn_centers[pm - 1];
            auto rmat = para_mps->para_split(pm - 1, para_rule);
            check_signal_()();
            if (iprint)
                cout << "init .. R = " << para_mps->center - para_mps->dot
                     << endl;
            right_contract_rotate_unordered(para_mps->center - para_mps->dot);
            if (para_rule != nullptr)
                para_rule->comm->barrier();
            if (para_rule == nullptr || para_rule->is_root()) {
                para_mps->tensors[para_mps->center] = rmat;
                para_mps->save_tensor(para_mps->center);
            }
            if (para_rule != nullptr)
                para_rule->comm->barrier();
        }
        para_mps->canonical_form[para_mps->conn_centers[pm - 1] - 1] = 'K';
        shared_ptr<ParallelCommunicator<S>> lpcomm = nullptr, rpcomm = nullptr;
        if (pcomm != nullptr) {
            if (pj - pi > para_mps->rule->comm->ngroup)
                lpcomm = pcomm, rpcomm = pcomm;
            else {
                int second = 0;
                for (int px = pm; px < pj; px++)
                    if (para_mps->rule->comm->group ==
                        px % para_mps->rule->comm->ngroup)
                        second = 1;
                shared_ptr<ParallelCommunicator<S>> ppcomm =
                    pcomm->split(second, pcomm->rank);
                lpcomm = second == 0 ? ppcomm : nullptr;
                rpcomm = second == 1 ? ppcomm : nullptr;
            }
        }
        if (pm > pi + 1) {
            para_mps->center = para_mps->conn_centers[pm - 1] - 1;
            init_parallel_environments(pi, pm, lpcomm);
        }
        if (pj > pm + 1) {
            para_mps->center = para_mps->conn_centers[pm - 1];
            init_parallel_environments(pm, pj, rpcomm);
        } else if (pm % 2 == 0) {
            para_mps->center = para_mps->conn_centers[pm - 1];
            int j = pj == para_mps->ncenter + 1
                        ? n_sites - 1
                        : para_mps->conn_centers[pj - 1] - 1;
            // SRRR -> LLLS
            if (para_mps->rule == nullptr ||
                para_mps->rule->comm->group ==
                    pm % para_mps->rule->comm->ngroup) {
                while (para_mps->center != j) {
                    para_mps->move_right(cg, para_rule);
                    check_signal_()();
                    if (iprint)
                        cout << "init .. L = " << para_mps->center << endl;
                    left_contract_rotate_unordered(para_mps->center);
                }
            }
            for (int i = para_mps->conn_centers[pm - 1]; i < j; i++)
                para_mps->canonical_form[i] = 'L';
            para_mps->canonical_form[j] = 'S';
        }
    }
    virtual bool check_singlet_embedding() const {
        if (bra->info->vacuum != bra->info->left_dims_fci[0]->quanta[0] ||
            ket->info->vacuum != ket->info->left_dims_fci[0]->quanta[0] ||
            mpo->left_vacuum != bra->info->vacuum ||
            mpo->left_vacuum != ket->info->vacuum) {
            S dq = mpo->left_vacuum,
              bq = bra->info->left_dims_fci[0]->quanta[0],
              kq = ket->info->left_dims_fci[0]->quanta[0];
            dq.set_n(bq.n() - kq.n());
            return dq.combine(bq, kq) != S(S::invalid);
        } else
            return true;
    }
    // Generate contracted environment blocks for all center sites
    virtual void init_environments(bool iprint = false) {
        this->iprint = iprint;
        envs.clear();
        envs.resize(n_sites);
        frame_<FPS>()->twrite = frame_<FPS>()->tread = frame_<FPS>()->tasync =
            0;
        frame_<FPS>()->fpwrite = frame_<FPS>()->fpread = 0;
        if (frame_<FPS>()->fp_codec != nullptr)
            frame_<FPS>()->fp_codec->ndata = frame_<FPS>()->fp_codec->ncpsd = 0;
        if (iprint)
            cout << "Environment initialization | Nsites = " << setw(5)
                 << n_sites << " | Center = " << setw(5) << center << endl;
        for (int i = 0; i < n_sites; i++) {
            envs[i] = make_shared<Partition<S, FL>>(nullptr, nullptr,
                                                    mpo->tensors[i]);
            if (i != n_sites - 1 && dot == 2)
                envs[i]->middle.push_back(mpo->tensors[i + 1]);
        }
        // singlet embedding
        if (bra->info->vacuum != bra->info->left_dims_fci[0]->quanta[0] ||
            ket->info->vacuum != ket->info->left_dims_fci[0]->quanta[0] ||
            mpo->left_vacuum != bra->info->vacuum ||
            mpo->left_vacuum != ket->info->vacuum) {
            envs[0]->left = make_shared<OperatorTensor<S, FL>>();
            shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                make_shared<VectorAllocator<uint32_t>>();
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            shared_ptr<SparseMatrixInfo<S>> xinfo =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            S dq = mpo->left_vacuum,
              bq = bra->info->left_dims_fci[0]->quanta[0],
              kq = ket->info->left_dims_fci[0]->quanta[0];
            if (dq.combine(bq, kq) == S(S::invalid)) {
                cout << "bra q = " << bq << "mpo q = " << dq << "ket q = " << kq
                     << endl;
                throw runtime_error(
                    "singlet embedding constraint cannot be satisfied!");
            }
            xinfo->initialize(*bra->info->left_dims_fci[0],
                              *ket->info->left_dims_fci[0], dq, false, false);
            shared_ptr<SparseMatrix<S, FL>> xmat =
                make_shared<SparseMatrix<S, FL>>(d_alloc);
            xmat->allocate(xinfo);
            xmat->data[0] = 1.0;
            envs[0]->left->ops[make_shared<OpExpr<S>>()] = xmat;
            envs[0]->left_op_infos.push_back(make_pair(dq, xinfo));
        }
        if (ket->get_type() & MPSTypes::MultiCenter) {
            shared_ptr<ParallelMPS<S, FLS>> para_mps =
                dynamic_pointer_cast<ParallelMPS<S, FLS>>(ket);
            para_mps->enable_parallel_writing();
            if (para_mps->rule == nullptr || para_mps->rule->comm->group == 0) {
                frame_<FP>()->activate(1);
                for (int i = 0; i < n_sites; i++) {
                    envs[i]->save_data(true,
                                       get_left_partition_filename(i, true));
                    envs[i]->save_data(false,
                                       get_right_partition_filename(i, true));
                }
                frame_<FP>()->activate(0);
            }
            if (para_mps->rule != nullptr)
                para_mps->rule->comm->barrier();
            shared_ptr<CG<S>> cg = mpo->tf->opf->cg;
            while (para_mps->center != 0) {
                if (iprint && (para_mps->rule == nullptr ||
                               0 % para_mps->rule->comm->ngroup ==
                                   para_mps->rule->comm->group))
                    cout << "pre init .. " << para_mps->center << " : "
                         << para_mps->canonical_form << endl;
                para_mps->move_left(cg, para_mps->rule);
            }
            assert(para_mps->conn_centers.size() != 0);
            para_mps->ncenter = (int)para_mps->conn_centers.size();
            para_mps->conn_matrices.resize(para_mps->ncenter);
            for (int i = 0; i < para_mps->ncenter; i++)
                para_mps->conn_matrices[i] =
                    make_shared<SparseMatrix<S, FLS>>();
            init_parallel_environments(
                0, para_mps->ncenter + 1,
                para_mps->rule == nullptr ? nullptr : para_mps->rule->comm,
                true);
            para_mps->center = para_mps->conn_centers[0];
            if (para_mps->rule != nullptr)
                para_mps->rule->comm->barrier();
            if (para_mps->rule == nullptr || para_mps->rule->comm->group == 0)
                para_mps->save_data();
            frame_<FP>()->activate(1);
            for (int i = 0; i < n_sites; i++) {
                if (i != 0)
                    envs[i]->load_data(true,
                                       get_left_partition_filename(i, true));
                envs[i]->load_data(false,
                                   get_right_partition_filename(i, true));
            }
            frame_<FP>()->activate(0);
            // outside code may have cout
            para_mps->disable_parallel_writing();
        } else if (bra->info->get_warm_up_type() == WarmUpTypes::None &&
                   ket->info->get_warm_up_type() == WarmUpTypes::None) {
            _t3.get_time();
            pair<size_t, size_t> max_pbr = make_pair(0, 0);
            for (int i = 1; i <= center; i++) {
                check_signal_()();
                if (iprint) {
                    cout << " INIT-L --> Site = " << setw(4) << i << " .. ";
                    cout.flush();
                }
                _t2.get_time();
                pair<size_t, size_t> pbr = left_contract_rotate(i);
                if (max_pbr.first + max_pbr.second <= pbr.first + pbr.second)
                    max_pbr.first = pbr.first, max_pbr.second = pbr.second;
                if (iprint) {
                    cout << " Bmem = " << setw(7)
                         << Parsing::to_size_string(pbr.first * sizeof(FL));
                    cout << " Rmem = " << setw(7)
                         << Parsing::to_size_string(pbr.second * sizeof(FL));
                    cout << " T = " << setw(4) << fixed << setprecision(2)
                         << _t2.get_time() << endl;
                }
            }
            for (int i = n_sites - dot - 1; i >= center; i--) {
                check_signal_()();
                if (iprint) {
                    cout << " INIT-R <-- Site = " << setw(4) << i << " .. ";
                    cout.flush();
                }
                _t2.get_time();
                pair<size_t, size_t> pbr = right_contract_rotate(i);
                if (max_pbr.first + max_pbr.second <= pbr.first + pbr.second)
                    max_pbr.first = pbr.first, max_pbr.second = pbr.second;
                if (iprint) {
                    cout << " Bmem = " << setw(7)
                         << Parsing::to_size_string(pbr.first * sizeof(FL));
                    cout << " Rmem = " << setw(7)
                         << Parsing::to_size_string(pbr.second * sizeof(FL));
                    cout << " T = " << setw(4) << fixed << setprecision(2)
                         << _t2.get_time() << endl;
                }
            }
            if (iprint) {
                cout << fixed << setprecision(3);
                cout << "Time init sweep = " << setw(12) << _t3.get_time();
                cout << " | MaxBmem = " << setw(7)
                     << Parsing::to_size_string(max_pbr.first * sizeof(FL));
                cout << " | MaxRmem = " << setw(7)
                     << Parsing::to_size_string(max_pbr.second * sizeof(FL))
                     << endl;
                cout << " | Tread = " << frame_<FPS>()->tread
                     << " | Twrite = " << frame_<FPS>()->twrite
                     << " | Tfpread = " << frame_<FPS>()->fpread
                     << " | Tfpwrite = " << frame_<FPS>()->fpwrite;
                if (frame_<FPS>()->fp_codec != nullptr)
                    cout << " | data = "
                         << Parsing::to_size_string(
                                frame_<FPS>()->fp_codec->ndata * sizeof(FPS))
                         << " | cpsd = "
                         << Parsing::to_size_string(
                                frame_<FPS>()->fp_codec->ncpsd * sizeof(FPS));
                cout << " | Tasync = " << frame_<FPS>()->tasync << endl << endl;
            }
        }
        frame_<FP>()->reset(1);
    }
    void partial_prepare(int a, int b) {
        assert(a >= 0 && b <= n_sites);
        tctr = trot = tmid = tint = tdctr = tdiag = tinfo = 0;
        // in unordered sweep, same-site contraction may not be identical
        cached_info = make_pair(OpCachingTypes::None, -1);
        frame_<FP>()->activate(1);
        // when conn center can change dynamically,
        // partition info in the middle also needs reloading
        for (int i = a; i <= b - dot; i++) {
            if (i != 0)
                envs[i]->load_data(true, get_left_partition_filename(i, true));
            envs[i]->load_data(false, get_right_partition_filename(i, true));
        }
        frame_<FP>()->activate(0);
        for (int i = b - dot; i > center; i--) {
            envs[i]->left_op_infos.clear();
            envs[i]->left = nullptr;
        }
        for (int i = a; i < center; i++) {
            envs[i]->right_op_infos.clear();
            envs[i]->right = nullptr;
        }
    }
    // Remove old environment for starting a new sweep
    void prepare(int start_site = 0, int end_site = -1) {
        tctr = trot = tmid = tint = tdctr = tdiag = tinfo = 0;
        if (dot == 2 && envs[0]->middle.size() == 1)
            throw runtime_error("switching from one-site algorithm to two-site "
                                "algorithm is not allowed.");
        if (end_site == -1)
            end_site = n_sites;
        // two-site to one-site transition
        if (dot == 1 && envs[0]->middle.size() == 2) {
            frame_<FP>()->reset_buffer(1);
            if (center == end_site - 2 &&
                (ket->canonical_form[end_site - 1] == 'C' ||
                 ket->canonical_form[end_site - 1] == 'M')) {
                center = end_site - 1;
                ket->canonical_form[center] =
                    ket->canonical_form[center] == 'C' ? 'S' : 'T';
                if (bra != ket && (bra->canonical_form[center] == 'C' ||
                                   bra->canonical_form[center] == 'M'))
                    bra->canonical_form[center] =
                        bra->canonical_form[center] == 'C' ? 'S' : 'T';
                fuse_center = mpo->schemer == nullptr
                                  ? end_site - 2
                                  : min((uint16_t)(end_site - 2),
                                        mpo->schemer->right_trans_site);
                frame_<FP>()->reset(1);
                if (envs[center - 1]->left != nullptr && center - 1 != 0)
                    frame_<FP>()->load_data(
                        1, get_left_partition_filename(center - 1));
                left_contract_rotate(center);
            }
            for (int i = n_sites - 1; i >= center; i--)
                if (envs[i]->right != nullptr)
                    frame_<FP>()->rename_data(
                        get_right_partition_filename(i),
                        get_right_partition_filename(i + 1));
            for (int i = n_sites - 1; i >= 0; i--) {
                envs[i]->middle.resize(1);
                if (i > start_site) {
                    envs[i]->right_op_infos = envs[i - 1]->right_op_infos;
                    envs[i]->right = envs[i - 1]->right;
                } else if (center == start_site) {
                    if (ket->canonical_form[center] == 'C')
                        ket->canonical_form[center] = 'K';
                    else if (ket->canonical_form[center] == 'M')
                        ket->canonical_form[center] = 'J';
                    if (bra != ket) {
                        if (bra->canonical_form[center] == 'C')
                            bra->canonical_form[center] = 'K';
                        else if (bra->canonical_form[center] == 'M')
                            bra->canonical_form[center] = 'J';
                    }
                    frame_<FP>()->reset(1);
                    if (envs[center + 1]->right != nullptr)
                        frame_<FP>()->load_data(
                            1, get_right_partition_filename(center + 1));
                    envs[center]->right_op_infos.clear();
                    envs[center]->right = nullptr;
                    right_contract_rotate(center);
                } else if (center == end_site - 1 && end_site < n_sites) {
                    frame_<FP>()->reset(1);
                    if (envs[center + 1]->right != nullptr)
                        frame_<FP>()->load_data(
                            1, get_right_partition_filename(center + 1));
                    envs[center]->right_op_infos.clear();
                    envs[center]->right = nullptr;
                    right_contract_rotate(center);
                }
            }
        }
        for (int i = end_site - 1; i > center; i--) {
            envs[i]->left_op_infos.clear();
            envs[i]->left = nullptr;
        }
        for (int i = start_site; i < center; i++) {
            envs[i]->right_op_infos.clear();
            envs[i]->right = nullptr;
        }
    }
    virtual void remove_partition_files() const {
        for (int i = 0; i < n_sites; i++)
            for (int info = 0; info < 2; info++) {
                string left_data_name = get_left_partition_filename(i, info);
                if (Parsing::file_exists(left_data_name))
                    Parsing::remove_file(left_data_name);
                string right_data_name = get_right_partition_filename(i, info);
                if (Parsing::file_exists(right_data_name))
                    Parsing::remove_file(right_data_name);
            }
    }
    // Move the center site by one
    virtual pair<size_t, size_t> move_to(int i, bool preserve_data = false) {
        string new_data_name = "";
        pair<size_t, size_t> pbr;
        // here the ialloc part is still needed even if we have cached
        // but consider two cases (for why it can be skipped):
        // 1. when center is delayed, then it is already in ialloc
        //    whether skipping loading makes no difference
        // 2. when it is not delayed, SpMatInfo is reconstructed
        //    but it is based on MPSInfo
        //    only SpMatInfo.cinfo needs prev_op_infos
        //    but since no contraction will be performed,
        //    incorrect cinfo does not have effects
        if (i > center) {
            if (envs[center]->left != nullptr &&
                !(cached_info.first == OpCachingTypes::Left &&
                  cached_info.second == center) &&
                center != 0)
                frame_<FP>()->load_data(1, get_left_partition_filename(center));
            // this will create left partition ++center (new_data_name)
            pbr = left_contract_rotate(++center, preserve_data);
            if (envs[center]->left != nullptr)
                new_data_name = get_left_partition_filename(center);
            if (frame_<FP>()->minimal_disk_usage && !preserve_data &&
                envs[center - 1]->right != nullptr) {
                string old_data_name = get_right_partition_filename(center - 1);
                if (Parsing::file_exists(old_data_name))
                    Parsing::remove_file(old_data_name);
            }
        } else if (i < center) {
            if (envs[center]->right != nullptr &&
                !(cached_info.first == OpCachingTypes::Right &&
                  cached_info.second == center + dot - 1))
                frame_<FP>()->load_data(1,
                                        get_right_partition_filename(center));
            // this will create right partition --center (new_data_name)
            pbr = right_contract_rotate(--center, preserve_data);
            if (envs[center]->right != nullptr)
                new_data_name = get_right_partition_filename(center);
            if (frame_<FP>()->minimal_disk_usage && !preserve_data &&
                envs[center + 1]->left != nullptr) {
                string old_data_name = get_left_partition_filename(center + 1);
                if (Parsing::file_exists(old_data_name))
                    Parsing::remove_file(old_data_name);
            }
        }
        if (para_rule != nullptr)
            para_rule->comm->barrier();
        bra->center = ket->center = center;
        // dynamic environment generation for warmup sweep
        if (i != n_sites - dot && envs[i]->right == nullptr &&
            ket->info->get_warm_up_type() != WarmUpTypes::None) {
            frame_<FP>()->reset(1);
            frame_<FP>()->activate(0);
            vector<shared_ptr<MPS<S, FLS>>> mpss =
                bra == ket ? vector<shared_ptr<MPS<S, FLS>>>{bra}
                           : vector<shared_ptr<MPS<S, FLS>>>{bra, ket};
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
                    shared_ptr<DeterminantMPSInfo<S, FL>> mps_info =
                        dynamic_pointer_cast<DeterminantMPSInfo<S, FL>>(
                            mps->info);
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
            frame_<FP>()->reset(1);
            if (new_data_name != "")
                frame_<FP>()->load_data(1, new_data_name);
        }
        if (i != 0 && envs[i]->left == nullptr &&
            ket->info->get_warm_up_type() != WarmUpTypes::None) {
            frame_<FP>()->reset(1);
            frame_<FP>()->activate(0);
            vector<shared_ptr<MPS<S, FLS>>> mpss =
                bra == ket ? vector<shared_ptr<MPS<S, FLS>>>{bra}
                           : vector<shared_ptr<MPS<S, FLS>>>{bra, ket};
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
                    shared_ptr<DeterminantMPSInfo<S, FL>> mps_info =
                        dynamic_pointer_cast<DeterminantMPSInfo<S, FL>>(
                            mps->info);
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
            frame_<FP>()->reset(1);
            if (new_data_name != "")
                frame_<FP>()->load_data(1, new_data_name);
        }
        return pbr;
    }
    // Contract left block for constructing effective Hamiltonian
    // site iL is the new site
    void left_contract(
        int iL, vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        shared_ptr<OperatorTensor<S, FL>> &new_left, bool delayed) {
        mpo->load_left_operators(iL);
        mpo->load_tensor(iL);
        // left contract infos
        vector<shared_ptr<Symbolic<S>>> lmats = {mpo->left_operator_names[iL]};
        if (mpo->schemer != nullptr && iL == mpo->schemer->left_trans_site &&
            mpo->schemer->right_trans_site - mpo->schemer->left_trans_site <=
                1) {
            mpo->load_schemer();
            lmats.push_back(mpo->schemer->left_new_operator_names);
        }
        vector<S> lsl = Partition<S, FL>::get_uniq_labels(lmats);
        shared_ptr<Symbolic<S>> lexprs =
            envs[iL]->left == nullptr
                ? nullptr
                : (mpo->left_operator_exprs.size() != 0
                       ? mpo->left_operator_exprs[iL]
                       : envs[iL]->left->lmat * mpo->tensors[iL]->lmat);
        vector<vector<pair<uint8_t, S>>> lsubsl =
            Partition<S, FL>::get_uniq_sub_labels(
                lexprs, mpo->left_operator_names[iL], lsl, mpo->left_vacuum);
        if (envs[iL]->left != nullptr && iL != 0)
            frame_<FP>()->load_data(1, get_left_partition_filename(iL));
        Partition<S, FL>::init_left_op_infos_notrunc(
            iL, bra->info, ket->info, lsl, lsubsl, envs[iL]->left_op_infos,
            mpo->site_op_infos[iL], left_op_infos, mpo->tf->opf->cg);
        // left contract
        frame_<FP>()->activate(0);
        if (cached_info.first == OpCachingTypes::Left &&
            cached_info.second == iL) {
            new_left = cached_opt;
            for (auto &p : new_left->ops)
                p.second->info = Partition<S, FL>::find_op_info(
                    left_op_infos, p.second->info->delta_quantum);
        } else {
            new_left = Partition<S, FL>::build_left(
                lmats, left_op_infos, mpo->sparse_form[iL] == 'S');
            mpo->tf->left_contract(envs[iL]->left, mpo->tensors[iL], new_left,
                                   mpo->left_operator_exprs.size() != 0
                                       ? mpo->left_operator_exprs[iL]
                                       : nullptr,
                                   delayed ? delayed_contraction
                                           : OpNamesSet());
            // for conventional scheme this will not be the case
            if (mpo->schemer != nullptr &&
                iL == mpo->schemer->left_trans_site &&
                mpo->schemer->right_trans_site -
                        mpo->schemer->left_trans_site <=
                    1) {
                mpo->tf->numerical_transform(
                    new_left, lmats[1], mpo->schemer->left_new_operator_exprs);
                mpo->unload_schemer();
            }
        }
        mpo->unload_tensor(iL);
        mpo->unload_left_operators(iL);
    }
    void delayed_left_contract(int iL,
                               shared_ptr<OperatorTensor<S, FL>> &new_left) {
        if (envs[iL]->left != nullptr && iL != 0)
            frame_<FP>()->load_data(1, get_left_partition_filename(iL));
        frame_<FP>()->activate(0);
        mpo->load_left_operators(iL);
        mpo->load_tensor(iL);
        mpo->tf->delayed_left_contract(
            envs[iL]->left, mpo->tensors[iL], new_left,
            mpo->left_operator_exprs.size() != 0 ? mpo->left_operator_exprs[iL]
                                                 : nullptr);
        mpo->unload_tensor(iL);
        mpo->unload_left_operators(iL);
    }
    // Contract right block for constructing effective Hamiltonian
    // site iR is the new site
    void right_contract(
        int iR,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        shared_ptr<OperatorTensor<S, FL>> &new_right, bool delayed) {
        mpo->load_right_operators(iR);
        mpo->load_tensor(iR);
        // right contract infos
        vector<shared_ptr<Symbolic<S>>> rmats = {mpo->right_operator_names[iR]};
        vector<S> rsl = Partition<S, FL>::get_uniq_labels(rmats);
        shared_ptr<Symbolic<S>> rexprs =
            envs[iR - dot + 1]->right == nullptr
                ? nullptr
                : (mpo->right_operator_exprs.size() != 0
                       ? mpo->right_operator_exprs[iR]
                       : mpo->tensors[iR]->rmat *
                             envs[iR - dot + 1]->right->rmat);
        vector<vector<pair<uint8_t, S>>> rsubsl =
            Partition<S, FL>::get_uniq_sub_labels(
                rexprs, mpo->right_operator_names[iR], rsl, ket->info->vacuum);
        if (envs[iR - dot + 1]->right != nullptr)
            frame_<FP>()->load_data(1,
                                    get_right_partition_filename(iR - dot + 1));
        Partition<S, FL>::init_right_op_infos_notrunc(
            iR, bra->info, ket->info, rsl, rsubsl,
            envs[iR - dot + 1]->right_op_infos, mpo->site_op_infos[iR],
            right_op_infos, mpo->tf->opf->cg);
        // right contract
        frame_<FP>()->activate(0);
        if (cached_info.first == OpCachingTypes::Right &&
            cached_info.second == iR) {
            new_right = cached_opt;
            for (auto &p : new_right->ops)
                p.second->info = Partition<S, FL>::find_op_info(
                    right_op_infos, p.second->info->delta_quantum);
        } else {
            new_right = Partition<S, FL>::build_right(
                rmats, right_op_infos, mpo->sparse_form[iR] == 'S');
            mpo->tf->right_contract(
                envs[iR - dot + 1]->right, mpo->tensors[iR], new_right,
                mpo->right_operator_exprs.size() != 0
                    ? mpo->right_operator_exprs[iR]
                    : nullptr,
                delayed ? delayed_contraction : OpNamesSet());
        }
        mpo->unload_tensor(iR);
        mpo->unload_right_operators(iR);
    }
    void delayed_right_contract(int iR,
                                shared_ptr<OperatorTensor<S, FL>> &new_right) {
        if (envs[iR - dot + 1]->right != nullptr)
            frame_<FP>()->load_data(1,
                                    get_right_partition_filename(iR - dot + 1));
        frame_<FP>()->activate(0);
        mpo->load_right_operators(iR);
        mpo->load_tensor(iR);
        mpo->tf->delayed_right_contract(envs[iR - dot + 1]->right,
                                        mpo->tensors[iR], new_right,
                                        mpo->right_operator_exprs.size() != 0
                                            ? mpo->right_operator_exprs[iR]
                                            : nullptr);
        mpo->unload_tensor(iR);
        mpo->unload_right_operators(iR);
    }
    // Copy left-most left block for constructing effective Hamiltonian
    // block to the left of site iL is copied
    void left_copy(
        int iL, vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        shared_ptr<OperatorTensor<S, FL>> &new_left, bool need_load = true) {
        assert(envs[iL]->left != nullptr);
        if (iL != 0 && need_load)
            frame_<FP>()->load_data(1, get_left_partition_filename(iL));
        shared_ptr<Allocator<FP>> d_alloc =
            make_shared<TemporaryAllocator<FP>>(frame_<FP>()->dallocs[1]->used);
        frame_<FP>()->activate(0);
        Partition<S, FL>::copy_op_infos(envs[iL]->left_op_infos, left_op_infos);
        if (cached_info.first == OpCachingTypes::LeftCopy &&
            cached_info.second == iL)
            new_left = cached_opt;
        else
            new_left = envs[iL]->left->deep_copy(
                frame_<FP>()->use_main_stack ? nullptr : d_alloc,
                frame_<FP>()->dallocs[1]);
        for (auto &p : new_left->ops)
            p.second->info = Partition<S, FL>::find_op_info(
                left_op_infos, p.second->info->delta_quantum);
    }
    // Copy right-most right block for constructing effective Hamiltonian
    // block to the right of site iR is copied
    void
    right_copy(int iR,
               vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
               shared_ptr<OperatorTensor<S, FL>> &new_right,
               bool need_load = true) {
        assert(envs[iR - dot + 1]->right != nullptr);
        if (need_load)
            frame_<FP>()->load_data(1,
                                    get_right_partition_filename(iR - dot + 1));
        shared_ptr<Allocator<FP>> d_alloc =
            make_shared<TemporaryAllocator<FP>>(frame_<FP>()->dallocs[1]->used);
        frame_<FP>()->activate(0);
        Partition<S, FL>::copy_op_infos(envs[iR - dot + 1]->right_op_infos,
                                        right_op_infos);
        if (cached_info.first == OpCachingTypes::RightCopy &&
            cached_info.second == iR)
            new_right = cached_opt;
        else
            new_right = envs[iR - dot + 1]->right->deep_copy(
                frame_<FP>()->use_main_stack ? nullptr : d_alloc,
                frame_<FP>()->dallocs[1]);
        for (auto &p : new_right->ops)
            p.second->info = Partition<S, FL>::find_op_info(
                right_op_infos, p.second->info->delta_quantum);
    }
    // Generate effective hamiltonian at current center site
    shared_ptr<EffectiveHamiltonian<S, FL>>
    eff_ham(FuseTypes fuse_type, bool forward, bool compute_diag,
            const shared_ptr<SparseMatrix<S, FLS>> &bra_wfn,
            const shared_ptr<SparseMatrix<S, FLS>> &ket_wfn) {
        // for level shift projection, we can have mixed multibra + single ket
        // assert(!(bra->get_type() & MPSTypes::MultiWfn));
        assert(!(ket->get_type() & MPSTypes::MultiWfn));
        const bool delay_left = center <= fuse_center;
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
            right_op_infos;
        shared_ptr<OperatorTensor<S, FL>> new_left, new_right;
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
        if (mpo->tf->get_type() == TensorFunctionsTypes::Archived) {
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->filename = get_middle_archive_filename();
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->offset = 0;
        }
        // avoid keeping too many large objects in memory simultaneously
        if (cached_opt != nullptr && cached_info.second != iL &&
            cached_info.second != iR)
            cached_opt->deallocate();
        if (cached_contraction)
            if (frame_<FP>()->use_main_stack)
                throw runtime_error("Cached contraction only works when "
                                    "use_main_stack is false.");
        // avoid swapping stack memory in forward/backward sweeps
        if (forward) {
            if (fuse_type & FuseTypes::FuseL)
                left_contract(iL, left_op_infos, new_left,
                              delay_left && iL != 0);
            else
                left_copy(iL, left_op_infos, new_left);
            if (fuse_type & FuseTypes::FuseR)
                right_contract(iR, right_op_infos, new_right,
                               !delay_left && iR != n_sites - 1);
            else
                right_copy(iR, right_op_infos, new_right);
            if (cached_contraction) {
                cached_opt = new_left;
                cached_info = make_pair((fuse_type & FuseTypes::FuseL)
                                            ? OpCachingTypes::Left
                                            : OpCachingTypes::LeftCopy,
                                        iL);
            }
        } else {
            if (fuse_type & FuseTypes::FuseR)
                right_contract(iR, right_op_infos, new_right,
                               !delay_left && iR != n_sites - 1);
            else
                right_copy(iR, right_op_infos, new_right);
            if (fuse_type & FuseTypes::FuseL)
                left_contract(iL, left_op_infos, new_left,
                              delay_left && iL != 0);
            else
                left_copy(iL, left_op_infos, new_left);
            if (cached_contraction) {
                cached_opt = new_right;
                cached_info = make_pair((fuse_type & FuseTypes::FuseR)
                                            ? OpCachingTypes::Right
                                            : OpCachingTypes::RightCopy,
                                        iR);
            }
        }
        _t2.get_time();
        // make sure that the previous block is still in memory
        if (!delayed_contraction.empty()) {
            if ((fuse_type & FuseTypes::FuseL) && delay_left && iL != 0)
                delayed_left_contract(iL, new_left);
            else if ((fuse_type & FuseTypes::FuseR) && !delay_left &&
                     iR != n_sites - 1)
                delayed_right_contract(iR, new_right);
        }
        mpo->load_middle_operators(iM);
        // delayed left-right contract
        shared_ptr<DelayedOperatorTensor<S, FL>> op =
            mpo->middle_operator_exprs.size() != 0
                ? mpo->tf->delayed_contract(
                      new_left, new_right, mpo->middle_operator_names[iM],
                      mpo->middle_operator_exprs[iM], delayed_contraction)
                : mpo->tf->delayed_contract(new_left, new_right, mpo->op,
                                            delayed_contraction);
        tdctr += _t2.get_time();
        frame_<FP>()->activate(0);
        shared_ptr<SymbolicColumnVector<S>> hops =
            mpo->middle_operator_exprs.size() != 0
                ? dynamic_pointer_cast<SymbolicColumnVector<S>>(
                      mpo->middle_operator_names[iM])
                : hop_mat;
        mpo->unload_middle_operators(iM);
        shared_ptr<SparseMatrix<S, FL>> fbw =
            ComplexMixture<S, FL, FLS>::forward(bra_wfn);
        shared_ptr<SparseMatrix<S, FL>> fkw =
            bra_wfn == ket_wfn ? fbw
                               : ComplexMixture<S, FL, FLS>::forward(ket_wfn);
        shared_ptr<EffectiveHamiltonian<S, FL>> efh =
            make_shared<EffectiveHamiltonian<S, FL>>(
                left_op_infos, right_op_infos, op, fbw, fkw, mpo->op, hops,
                mpo->left_vacuum, mpo->tf, compute_diag, mpo->npdm_scheme);
        efh->npdm_fragment_filename = get_npdm_fragment_filename(iM);
        efh->npdm_n_sites = n_sites;
        efh->npdm_center = iM;
        efh->npdm_parallel_center = mpo->npdm_parallel_center;
        tdiag += _t2.get_time();
        frame_<FP>()->update_peak_used_memory();
        return efh;
    }
    // Generate effective hamiltonian at current center site
    // for MultiMPS case
    shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>>
    multi_eff_ham(FuseTypes fuse_type, bool forward, bool compute_diag) {
        assert(bra->get_type() & MPSTypes::MultiWfn);
        assert(ket->get_type() & MPSTypes::MultiWfn);
        const bool delay_left = center <= fuse_center;
        shared_ptr<MultiMPS<S, FLS>> mbra =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(bra);
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(ket);
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
            right_op_infos;
        shared_ptr<OperatorTensor<S, FL>> new_left, new_right;
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
        if (mpo->tf->get_type() == TensorFunctionsTypes::Archived) {
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->filename = get_middle_archive_filename();
            dynamic_pointer_cast<ArchivedTensorFunctions<S, FL>>(mpo->tf)
                ->offset = 0;
        }
        // avoid keeping too many large objects in memory simultaneously
        if (cached_opt != nullptr && cached_info.second != iL &&
            cached_info.second != iR)
            cached_opt->deallocate();
        if (cached_contraction)
            if (frame_<FP>()->use_main_stack)
                throw runtime_error("Cached contraction only works when "
                                    "use_main_stack is false.");
        // avoid swapping stack memory in forward/backward sweeps
        if (forward) {
            if (fuse_type & FuseTypes::FuseL)
                left_contract(iL, left_op_infos, new_left,
                              delay_left && iL != 0);
            else
                left_copy(iL, left_op_infos, new_left);
            if (fuse_type & FuseTypes::FuseR)
                right_contract(iR, right_op_infos, new_right,
                               !delay_left && iR != n_sites - 1);
            else
                right_copy(iR, right_op_infos, new_right);
            if (cached_contraction) {
                cached_opt = new_left;
                cached_info = make_pair((fuse_type & FuseTypes::FuseL)
                                            ? OpCachingTypes::Left
                                            : OpCachingTypes::LeftCopy,
                                        iL);
            }
        } else {
            if (fuse_type & FuseTypes::FuseR)
                right_contract(iR, right_op_infos, new_right,
                               !delay_left && iR != n_sites - 1);
            else
                right_copy(iR, right_op_infos, new_right);
            if (fuse_type & FuseTypes::FuseL)
                left_contract(iL, left_op_infos, new_left,
                              delay_left && iL != 0);
            else
                left_copy(iL, left_op_infos, new_left);
            if (cached_contraction) {
                cached_opt = new_right;
                cached_info = make_pair((fuse_type & FuseTypes::FuseR)
                                            ? OpCachingTypes::Right
                                            : OpCachingTypes::RightCopy,
                                        iR);
            }
        }
        _t2.get_time();
        // make sure that the previous block is still in memory
        if (!delayed_contraction.empty()) {
            if ((fuse_type & FuseTypes::FuseL) && delay_left && iL != 0)
                delayed_left_contract(iL, new_left);
            else if ((fuse_type & FuseTypes::FuseR) && !delay_left &&
                     iR != n_sites - 1)
                delayed_right_contract(iR, new_right);
        }
        mpo->load_middle_operators(iM);
        // delayed left-right contract
        shared_ptr<DelayedOperatorTensor<S, FL>> op =
            mpo->middle_operator_exprs.size() != 0
                ? mpo->tf->delayed_contract(
                      new_left, new_right, mpo->middle_operator_names[iM],
                      mpo->middle_operator_exprs[iM], delayed_contraction)
                : mpo->tf->delayed_contract(new_left, new_right, mpo->op,
                                            delayed_contraction);
        tdctr += _t2.get_time();
        frame_<FP>()->activate(0);
        shared_ptr<SymbolicColumnVector<S>> hops =
            mpo->middle_operator_exprs.size() != 0
                ? dynamic_pointer_cast<SymbolicColumnVector<S>>(
                      mpo->middle_operator_names[iM])
                : hop_mat;
        mpo->unload_middle_operators(iM);
        vector<shared_ptr<SparseMatrixGroup<S, FL>>> fbw =
            ComplexMixture<S, FL, FLS>::forward(mbra->wfns);
        vector<shared_ptr<SparseMatrixGroup<S, FL>>> fkw =
            mbra == mket ? fbw
                         : ComplexMixture<S, FL, FLS>::forward(mket->wfns);
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> efh =
            make_shared<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>>(
                left_op_infos, right_op_infos, op, fbw, fkw, mpo->op, hops,
                mpo->left_vacuum, mpo->tf, compute_diag, mpo->npdm_scheme);
        efh->npdm_fragment_filename = get_npdm_fragment_filename(iM);
        efh->npdm_n_sites = n_sites;
        efh->npdm_center = iM;
        efh->npdm_parallel_center = mpo->npdm_parallel_center;
        tdiag += _t2.get_time();
        frame_<FP>()->update_peak_used_memory();
        return efh;
    }
    // Absorb wfn matrix into adjacent MPS tensor in one-site algorithm
    static void contract_one_dot(int i,
                                 const shared_ptr<SparseMatrix<S, FLS>> &wfn,
                                 const shared_ptr<MPS<S, FLS>> &mps,
                                 bool forward, bool reduced = false) {
        shared_ptr<SparseMatrix<S, FLS>> old_wfn =
            make_shared<SparseMatrix<S, FLS>>();
        shared_ptr<SparseMatrixInfo<S>> old_wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        frame_<FP>()->activate(1);
        mps->load_tensor(i);
        frame_<FP>()->activate(0);
        if (reduced) {
            if (forward)
                old_wfn_info->initialize_contract(wfn->info,
                                                  mps->tensors[i]->info);
            else
                old_wfn_info->initialize_contract(mps->tensors[i]->info,
                                                  wfn->info);
        } else {
            frame_<FP>()->activate(1);
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
            frame_<FP>()->activate(0);
            old_wfn_info->initialize(ll, rr, mps->info->target, false, true);
            frame_<FP>()->activate(1);
            if (forward)
                rr.deallocate();
            else
                ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame_<FP>()->activate(0);
        }
        frame_<FP>()->activate(0);
        old_wfn->allocate(old_wfn_info);
        if (forward)
            old_wfn->contract(wfn, mps->tensors[i]);
        else
            old_wfn->contract(mps->tensors[i], wfn);
        frame_<FP>()->activate(1);
        mps->unload_tensor(i);
        frame_<FP>()->activate(0);
        mps->tensors[i] = old_wfn;
    }
    // Contract two adjcent MPS tensors to one two-site MPS tensor
    static void contract_two_dot(int i, const shared_ptr<MPS<S, FLS>> &mps,
                                 bool reduced = false) {
        shared_ptr<SparseMatrix<S, FLS>> old_wfn =
            make_shared<SparseMatrix<S, FLS>>();
        shared_ptr<SparseMatrixInfo<S>> old_wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        frame_<FP>()->activate(1);
        mps->load_tensor(i);
        mps->load_tensor(i + 1);
        frame_<FP>()->activate(0);
        if (reduced)
            old_wfn_info->initialize_contract(mps->tensors[i]->info,
                                              mps->tensors[i + 1]->info);
        else {
            frame_<FP>()->activate(1);
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
            frame_<FP>()->activate(0);
            old_wfn_info->initialize(ll, rr, mps->info->target, false, true);
            frame_<FP>()->activate(1);
            rr.deallocate();
            ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame_<FP>()->activate(0);
        }
        frame_<FP>()->activate(0);
        old_wfn->allocate(old_wfn_info);
        old_wfn->contract(mps->tensors[i], mps->tensors[i + 1]);
        frame_<FP>()->activate(1);
        mps->unload_tensor(i + 1);
        mps->unload_tensor(i);
        frame_<FP>()->activate(0);
        mps->tensors[i] = old_wfn;
        mps->tensors[i + 1] = nullptr;
    }
    // Absorb wfn matrices into adjacent MultiMPS tensor in one-site algorithm
    static void contract_multi_one_dot(
        int i, const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &wfns,
        const shared_ptr<MultiMPS<S, FLS>> &mps, bool forward,
        bool reduced = false) {
        vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfns;
        vector<shared_ptr<SparseMatrixInfo<S>>> old_wfn_infos;
        frame_<FP>()->activate(1);
        mps->load_tensor(i);
        assert(mps->tensors[i] != nullptr);
        frame_<FP>()->activate(0);
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
            frame_<FP>()->activate(1);
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
            frame_<FP>()->activate(0);
            for (int j = 0; j < wfns[0]->n; j++) {
                old_wfn_infos[j] = make_shared<SparseMatrixInfo<S>>();
                old_wfn_infos[j]->initialize(
                    ll, rr, wfns[0]->infos[j]->delta_quantum, false, true);
            }
            frame_<FP>()->activate(1);
            if (forward)
                rr.deallocate();
            else
                ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame_<FP>()->activate(0);
        }
        frame_<FP>()->activate(0);
        for (int k = 0; k < mps->nroots; k++) {
            old_wfns[k] = make_shared<SparseMatrixGroup<S, FLS>>();
            old_wfns[k]->allocate(old_wfn_infos);
            if (forward)
                for (int j = 0; j < old_wfns[k]->n; j++)
                    (*old_wfns[k])[j]->contract((*wfns[k])[j], mps->tensors[i]);
            else
                for (int j = 0; j < old_wfns[k]->n; j++)
                    (*old_wfns[k])[j]->contract(mps->tensors[i], (*wfns[k])[j]);
        }
        frame_<FP>()->activate(1);
        mps->unload_tensor(i);
        frame_<FP>()->activate(0);
        mps->tensors[i] = nullptr;
        mps->wfns = old_wfns;
    }
    // Contract two adjcent MultiMPS tensors to one two-site MultiMPS tensor
    static void contract_multi_two_dot(int i,
                                       const shared_ptr<MultiMPS<S, FLS>> &mps,
                                       bool reduced = false) {
        vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfns;
        vector<shared_ptr<SparseMatrixInfo<S>>> old_wfn_infos;
        frame_<FP>()->activate(1);
        assert(mps->tensors[i] == nullptr || mps->tensors[i + 1] == nullptr);
        bool left_wfn = mps->tensors[i] == nullptr;
        if (left_wfn) {
            mps->load_wavefunction(i);
            mps->load_tensor(i + 1);
        } else {
            mps->load_tensor(i);
            mps->load_wavefunction(i + 1);
        }
        frame_<FP>()->activate(0);
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
            frame_<FP>()->activate(1);
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
            frame_<FP>()->activate(0);
            for (int j = 0; j < mps->wfns[0]->n; j++) {
                old_wfn_infos[j] = make_shared<SparseMatrixInfo<S>>();
                old_wfn_infos[j]->initialize(
                    ll, rr, mps->wfns[0]->infos[j]->delta_quantum, false, true);
            }
            frame_<FP>()->activate(1);
            rr.deallocate();
            ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame_<FP>()->activate(0);
        }
        frame_<FP>()->activate(0);
        for (int k = 0; k < mps->nroots; k++) {
            old_wfns[k] = make_shared<SparseMatrixGroup<S, FLS>>();
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
        frame_<FP>()->activate(1);
        if (left_wfn) {
            mps->unload_tensor(i + 1);
            mps->unload_wavefunction(i);
        } else {
            mps->unload_wavefunction(i + 1);
            mps->unload_tensor(i);
        }
        frame_<FP>()->activate(0);
        mps->tensors[i] = mps->tensors[i + 1] = nullptr;
        mps->wfns = old_wfns;
    }
    // Density matrix of a MPS tensor
    static shared_ptr<SparseMatrix<S, FLS>> density_matrix(
        S vacuum, const shared_ptr<SparseMatrix<S, FLS>> &psi, bool trace_right,
        FPS noise, NoiseTypes noise_type, FPS scale = 1.0,
        const shared_ptr<SparseMatrixGroup<S, FLS>> &pkets = nullptr) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(
            vector<shared_ptr<SparseMatrixInfo<S>>>{psi->info}, vacuum,
            trace_right);
        shared_ptr<SparseMatrix<S, FLS>> dm =
            make_shared<SparseMatrix<S, FLS>>();
        dm->allocate(dm_info);
        assert(psi->factor == (FPS)1.0);
        psi->factor = sqrt(scale);
        OperatorFunctions<S, FLS>::trans_product(psi, dm, trace_right,
                                                 sqrt(noise), noise_type);
        psi->factor = 1;
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0) {
            assert(pkets != nullptr);
            scale_perturbative_noise(noise, noise_type, pkets);
            for (int i = 1; i < pkets->n; i++)
                OperatorFunctions<S, FLS>::trans_product(
                    (*pkets)[i], dm, trace_right, 0.0, NoiseTypes::None);
        }
        return dm;
    }
    // Density matrix of a MultiMPS tensor
    // noise will be added several times (for each wfn)
    static shared_ptr<SparseMatrix<S, FLS>> density_matrix_with_multi_target(
        S vacuum, const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &psi,
        const vector<FPS> weights, bool trace_right, FPS noise,
        NoiseTypes noise_type, FP scale = 1.0,
        const shared_ptr<SparseMatrixGroup<S, FLS>> &pkets = nullptr) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(psi[0]->infos, vacuum, trace_right);
        shared_ptr<SparseMatrix<S, FLS>> dm =
            make_shared<SparseMatrix<S, FLS>>();
        dm->allocate(dm_info);
        assert(weights.size() == psi.size());
        for (size_t i = 0; i < psi.size(); i++)
            for (int j = 0; j < psi[i]->n; j++) {
                shared_ptr<SparseMatrix<S, FLS>> wfn = (*psi[i])[j];
                wfn->factor = sqrt(weights[i] * scale);
                OperatorFunctions<S, FLS>::trans_product(
                    wfn, dm, trace_right, sqrt(noise), noise_type);
            }
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0) {
            assert(pkets != nullptr);
            scale_perturbative_noise(noise, noise_type, pkets);
            for (int i = 1; i < pkets->n; i++)
                OperatorFunctions<S, FLS>::trans_product(
                    (*pkets)[i], dm, trace_right, 0.0, NoiseTypes::None);
        }
        return dm;
    }
    // Add wavefunction to density matrix
    static void
    density_matrix_add_wfn(const shared_ptr<SparseMatrix<S, FLS>> &dm,
                           const shared_ptr<SparseMatrix<S, FLS>> &psi,
                           bool trace_right, FPS scale = 1.0) {
        assert(psi->factor == (FLS)1.0);
        psi->factor = sqrt(scale);
        OperatorFunctions<S, FLS>::trans_product(psi, dm, trace_right, 0.0,
                                                 NoiseTypes::None);
        psi->factor = 1;
    }
    // Add wavefunction group to density matrix
    static void density_matrix_add_wfn_groups(
        const shared_ptr<SparseMatrix<S, FLS>> &dm,
        const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &psi,
        const vector<FPS> weights, bool trace_right, FPS scale = 1.0) {
        assert(weights.size() == psi.size());
        for (size_t i = 0; i < psi.size(); i++)
            for (int j = 0; j < psi[i]->n; j++) {
                shared_ptr<SparseMatrix<S, FLS>> wfn = (*psi[i])[j];
                wfn->factor = sqrt(weights[i] * scale);
                OperatorFunctions<S, FLS>::trans_product(wfn, dm, trace_right,
                                                         0.0, NoiseTypes::None);
            }
    }
    // Density matrix with perturbed wavefunctions as noise
    static void density_matrix_add_perturbative_noise(
        const shared_ptr<SparseMatrix<S, FLS>> &dm, bool trace_right, FPS noise,
        NoiseTypes noise_type,
        const shared_ptr<SparseMatrixGroup<S, FLS>> &mats) {
        scale_perturbative_noise(noise, noise_type, mats);
        for (int i = 1; i < mats->n; i++)
            OperatorFunctions<S, FLS>::trans_product(
                (*mats)[i], dm, trace_right, 0.0, NoiseTypes::None);
    }
    // Density matrix of several MPS tensors summed with weights
    static void density_matrix_add_matrices(
        const shared_ptr<SparseMatrix<S, FLS>> &dm,
        const shared_ptr<SparseMatrix<S, FLS>> &psi, bool trace_right,
        const vector<GMatrix<FLS>> &mats, const vector<FPS> &weights) {
        FLS *ptr = psi->data;
        assert(psi->factor == (FLS)1.0);
        assert(mats.size() == weights.size() - 1);
        for (size_t i = 1; i < weights.size(); i++) {
            psi->data = mats[i - 1].data;
            psi->factor = sqrt(weights[i]);
            OperatorFunctions<S, FLS>::trans_product(psi, dm, trace_right, 0.0);
        }
        psi->data = ptr, psi->factor = 1.0;
    }
    // Density matrix of several MPS tensors summed with weights
    static void density_matrix_add_matrix_groups(
        const shared_ptr<SparseMatrix<S, FLS>> &dm,
        const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &psi,
        bool trace_right, const vector<GMatrix<FLS>> &mats,
        const vector<FPS> &weights) {
        int p = 0, np = psi.size() * psi[0]->n;
        assert(mats.size() == (weights.size() - 1) * np);
        for (size_t k = 1; k < weights.size(); k++)
            for (size_t i = 0; i < psi.size(); i++)
                for (int j = 0; j < psi[i]->n; j++) {
                    shared_ptr<SparseMatrix<S, FLS>> wfn = (*psi[i])[j];
                    wfn->data = mats[p++].data;
                    wfn->factor = sqrt(weights[k] / np);
                    OperatorFunctions<S, FLS>::trans_product(wfn, dm,
                                                             trace_right, 0.0);
                }
        assert((size_t)p == mats.size());
    }
    // Direct add noise to wavefunction (before svd)
    static void
    wavefunction_add_noise(const shared_ptr<SparseMatrix<S, FLS>> &psi,
                           FPS noise) {
        assert(psi->factor == (FPS)1.0);
        if (abs(noise) < TINY && noise == (FPS)0.0)
            return;
        shared_ptr<SparseMatrix<S, FLS>> tmp =
            make_shared<SparseMatrix<S, FLS>>();
        tmp->allocate(psi->info);
        tmp->randomize(-0.5, 0.5);
        FPS noise_scale = sqrt(noise) / tmp->norm();
        GMatrixFunctions<FLS>::iadd(
            GMatrix<FLS>(psi->data, (MKL_INT)psi->total_memory, 1),
            GMatrix<FLS>(tmp->data, (MKL_INT)tmp->total_memory, 1),
            noise_scale);
        tmp->deallocate();
    }
    // Scale perturbative noise (before svd)
    static void scale_perturbative_noise(
        FPS noise, NoiseTypes noise_type,
        const shared_ptr<SparseMatrixGroup<S, FLS>> &mats) {
        if (abs(noise) < TINY && noise == (FPS)0.0)
            return;
        if (!(noise_type & NoiseTypes::Unscaled)) {
            for (int i = 0; i < mats->n; i++) {
                FPS mat_norm = (*mats)[i]->norm();
                if (abs(mat_norm) > TINY)
                    (*mats)[i]->iscale(1 / mat_norm);
            }
        }
        FPS norm = mats->norm();
        if (abs(norm) > TINY)
            mats->iscale(sqrt(noise) / norm);
    }
    // Diagonalize density matrix and truncate to k eigenvalues
    static FPS
    truncate_density_matrix(const shared_ptr<SparseMatrix<S, FLS>> &dm,
                            vector<pair<int, int>> &ss, int k, FPS cutoff,
                            bool store_wfn_spectra, vector<FPS> &wfn_spectra,
                            TruncationTypes trunc_type) {
        vector<shared_ptr<VectorAllocator<FPS>>> d_allocs(dm->info->n);
        vector<GDiagonalMatrix<FPS>> eigen_values(
            dm->info->n, GDiagonalMatrix<FPS>(nullptr, 0));
        vector<GMatrix<FPS>> eigen_values_reduced(dm->info->n,
                                                  GMatrix<FPS>(nullptr, 0, 0));
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int i = 0; i < dm->info->n; i++) {
            d_allocs[i] = make_shared<VectorAllocator<FPS>>();
            GDiagonalMatrix<FPS> w(nullptr, dm->info->n_states_bra[i]);
            w.allocate(d_allocs[i]);
            if (trunc_type & TruncationTypes::RealDensityMatrix)
                GMatrixFunctions<FLS>::keep_real((*dm)[i]);
            GMatrixFunctions<FLS>::eigs((*dm)[i], w);
            GMatrix<FPS> wr(nullptr, w.n, 1);
            wr.allocate(d_allocs[i]);
            GMatrixFunctions<FPS>::copy(wr, GMatrix<FPS>(w.data, w.n, 1));
            if (trunc_type & TruncationTypes::Reduced)
                GMatrixFunctions<FPS>::iscale(
                    wr, 1.0 / dm->info->quanta[i].multiplicity());
            else if (trunc_type & TruncationTypes::ReducedInversed)
                GMatrixFunctions<FPS>::iscale(
                    wr, dm->info->quanta[i].multiplicity());
            eigen_values[i] = w;
            eigen_values_reduced[i] = wr;
        }
        threading->activate_normal();
        int k_total = 0, k_total_multi = 0;
        for (int i = 0; i < dm->info->n; i++) {
            k_total += eigen_values[i].n;
            k_total_multi +=
                eigen_values[i].n * dm->info->quanta[i].multiplicity();
        }
        if (store_wfn_spectra) {
            bool with_multi =
                trunc_type & TruncationTypes::SpectraWithMultiplicity;
            wfn_spectra.clear();
            wfn_spectra.reserve(with_multi ? k_total_multi : k_total);
            for (int i = 0; i < dm->info->n; i++)
                if (with_multi) {
                    for (int p = 0; p < dm->info->quanta[i].multiplicity(); p++)
                        for (int j = 0; j < eigen_values[i].n; j++)
                            wfn_spectra.push_back(
                                sqrt(max((FPS)0, eigen_values[i].data[j])));
                } else {
                    for (int j = 0; j < eigen_values[i].n; j++)
                        wfn_spectra.push_back(
                            sqrt(max((FPS)0, eigen_values[i].data[j])));
                }
        }
        FPS error = 0.0;
        ss.reserve(k_total);
        for (int i = 0; i < (int)eigen_values.size(); i++)
            for (int j = 0; j < eigen_values[i].n; j++)
                ss.push_back(make_pair(i, j));
        assert(k_total == (int)ss.size());
        if (k != -1) {
            sort(ss.begin(), ss.end(),
                 [&eigen_values_reduced](const pair<int, int> &a,
                                         const pair<int, int> &b) {
                     return eigen_values_reduced[a.first].data[a.second] >
                            eigen_values_reduced[b.first].data[b.second];
                 });
            if (((ubond_t)trunc_type / (ubond_t)TruncationTypes::KeepOne) ==
                0) {
                for (int i = k; i < k_total; i++) {
                    FPS x = eigen_values[ss[i].first].data[ss[i].second];
                    if (x > 0)
                        error += x;
                }
                for (k = min(k, k_total);
                     k > 1 && eigen_values_reduced[ss[k - 1].first]
                                      .data[ss[k - 1].second] < cutoff;
                     k--) {
                    FPS x =
                        eigen_values[ss[k - 1].first].data[ss[k - 1].second];
                    if (x > 0)
                        error += x;
                }
                if (k < k_total)
                    ss.resize(k);
            } else {
                ubond_t keep =
                    (ubond_t)trunc_type / (ubond_t)TruncationTypes::KeepOne;
                vector<int> mask(eigen_values.size(), 0), smask(k_total, 0);
                for (int i = 0; i < k_total; i++) {
                    mask[ss[i].first]++;
                    smask[i] = mask[ss[i].first] > (int)keep;
                }
                for (int i = k; i < k_total; i++) {
                    FPS x = eigen_values[ss[i].first].data[ss[i].second];
                    if (x > 0 && smask[i])
                        error += x;
                }
                for (k = min(k, k_total);
                     k > 1 && eigen_values_reduced[ss[k - 1].first]
                                      .data[ss[k - 1].second] < cutoff;
                     k--) {
                    FPS x =
                        eigen_values[ss[k - 1].first].data[ss[k - 1].second];
                    if (x > 0 && smask[k - 1])
                        error += x;
                }
                vector<pair<int, int>> rss(
                    ss.begin(), k < k_total ? ss.begin() + k : ss.end());
                for (int i = k; i < k_total; i++)
                    if (!smask[i])
                        rss.push_back(ss[i]);
                ss = rss;
                assert(ss.size() != 0);
            }
            sort(ss.begin(), ss.end(),
                 [](const pair<int, int> &a, const pair<int, int> &b) {
                     return a.first != b.first ? a.first < b.first
                                               : a.second < b.second;
                 });
        }
        for (int i = dm->info->n - 1; i >= 0; i--) {
            eigen_values_reduced[i].deallocate(d_allocs[i]);
            eigen_values[i].deallocate(d_allocs[i]);
        }
        return error;
    }
    // Truncate and keep k singular values
    static FPS truncate_singular_values(
        const vector<S> &qs, const vector<shared_ptr<GTensor<FPS>>> &s,
        vector<pair<int, int>> &ss, int k, FPS cutoff, bool store_wfn_spectra,
        vector<FPS> &wfn_spectra, TruncationTypes trunc_type) {
        vector<shared_ptr<GTensor<FPS>>> s_reduced;
        cutoff = sqrt(cutoff);
        int k_total = 0, k_total_multi = 0;
        for (int i = 0; i < (int)s.size(); i++) {
            shared_ptr<GTensor<FPS>> wr =
                make_shared<GTensor<FPS>>(s[i]->shape);
            GMatrixFunctions<FPS>::copy(wr->ref(), s[i]->ref());
            if (trunc_type & TruncationTypes::Reduced)
                GMatrixFunctions<FPS>::iscale(wr->ref(),
                                              sqrt(1.0 / qs[i].multiplicity()));
            else if (trunc_type & TruncationTypes::ReducedInversed)
                GMatrixFunctions<FPS>::iscale(wr->ref(),
                                              sqrt(qs[i].multiplicity()));
            s_reduced.push_back(wr);
            k_total += wr->shape[0];
            k_total_multi += wr->shape[0] * qs[i].multiplicity();
        }
        if (store_wfn_spectra) {
            bool with_multi =
                trunc_type & TruncationTypes::SpectraWithMultiplicity;
            wfn_spectra.clear();
            wfn_spectra.reserve(with_multi ? k_total_multi : k_total);
            for (int i = 0; i < (int)s.size(); i++)
                if (with_multi)
                    for (int p = 0; p < qs[i].multiplicity(); p++)
                        wfn_spectra.insert(wfn_spectra.end(),
                                           s[i]->data->begin(),
                                           s[i]->data->end());
                else
                    wfn_spectra.insert(wfn_spectra.end(), s[i]->data->begin(),
                                       s[i]->data->end());
        }
        FPS error = 0.0;
        ss.reserve(k_total);
        for (int i = 0; i < (int)s.size(); i++)
            for (int j = 0; j < s[i]->shape[0]; j++)
                ss.push_back(make_pair(i, j));
        assert(k_total == (int)ss.size());
        if (k != -1) {
            sort(
                ss.begin(), ss.end(),
                [&s_reduced](const pair<int, int> &a, const pair<int, int> &b) {
                    return (*s_reduced[a.first]->data)[a.second] >
                           (*s_reduced[b.first]->data)[b.second];
                });
            if (((ubond_t)trunc_type / (ubond_t)TruncationTypes::KeepOne) ==
                0) {
                for (int i = k; i < k_total; i++) {
                    FPS x = (*s[ss[i].first]->data)[ss[i].second];
                    if (x > 0)
                        error += x * x;
                }
                for (k = min(k, k_total);
                     k > 1 &&
                     (*s_reduced[ss[k - 1].first]->data)[ss[k - 1].second] <
                         cutoff;
                     k--) {
                    FPS x = (*s[ss[k - 1].first]->data)[ss[k - 1].second];
                    if (x > 0)
                        error += x * x;
                }
                if (k < k_total)
                    ss.resize(k);
            } else {
                ubond_t keep =
                    (ubond_t)trunc_type / (ubond_t)TruncationTypes::KeepOne;
                vector<int> mask(s.size(), 0), smask(k_total, 0);
                for (int i = 0; i < k_total; i++) {
                    mask[ss[i].first]++;
                    smask[i] = mask[ss[i].first] > (int)keep;
                }
                for (int i = k; i < k_total; i++) {
                    FPS x = (*s[ss[i].first]->data)[ss[i].second];
                    if (x > 0 && smask[i])
                        error += x * x;
                }
                for (k = min(k, k_total);
                     k > 1 &&
                     (*s_reduced[ss[k - 1].first]->data)[ss[k - 1].second] <
                         cutoff;
                     k--) {
                    FPS x = (*s[ss[k - 1].first]->data)[ss[k - 1].second];
                    if (x > 0 && smask[k - 1])
                        error += x * x;
                }
                vector<pair<int, int>> rss(
                    ss.begin(), k < k_total ? ss.begin() + k : ss.end());
                for (int i = k; i < k_total; i++)
                    if (!smask[i])
                        rss.push_back(ss[i]);
                ss = rss;
                assert(ss.size() != 0);
            }
            sort(ss.begin(), ss.end(),
                 [](const pair<int, int> &a, const pair<int, int> &b) {
                     return a.first != b.first ? a.first < b.first
                                               : a.second < b.second;
                 });
        }
        return error;
    }
    // Get rotation matrix info from svd info
    static shared_ptr<SparseMatrixInfo<S>> rotation_matrix_info_from_svd(
        S opdq, const vector<S> &qs, const vector<shared_ptr<GTensor<FLS>>> &ts,
        bool trace_right, const vector<int> &ilr, const vector<ubond_t> &im) {
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>();
        rinfo->is_fermion = false;
        rinfo->is_wavefunction = false;
        rinfo->delta_quantum = opdq;
        int kk = (int)ilr.size();
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
        bool trace_right, const vector<int> &ilr, const vector<ubond_t> &im,
        vector<vector<int>> &idx_dm_to_wfn) {
        shared_ptr<SparseMatrixInfo<S>> winfo =
            make_shared<SparseMatrixInfo<S>>();
        winfo->is_fermion = false;
        winfo->is_wavefunction = true;
        winfo->delta_quantum = wfninfo->delta_quantum;
        idx_dm_to_wfn.resize(qs.size());
        if (trace_right)
            for (int i = 0; i < wfninfo->n; i++) {
                S pb = wfninfo->quanta[i].get_bra(wfninfo->delta_quantum);
                size_t iq = lower_bound(qs.begin(), qs.end(), pb) - qs.begin();
                idx_dm_to_wfn[iq].push_back(i);
            }
        else
            for (int i = 0; i < wfninfo->n; i++) {
                S pk = -wfninfo->quanta[i].get_ket();
                size_t iq = lower_bound(qs.begin(), qs.end(), pk) - qs.begin();
                idx_dm_to_wfn[iq].push_back(i);
            }
        int kkw = 0, kk = (int)ilr.size();
        for (int i = 0; i < kk; i++)
            kkw += (int)idx_dm_to_wfn[ilr[i]].size();
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
        const vector<int> &ilr, const vector<ubond_t> &im) {
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>();
        rinfo->is_fermion = false;
        rinfo->is_wavefunction = false;
        rinfo->delta_quantum = dminfo->delta_quantum;
        int kk = (int)ilr.size();
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
        const vector<int> &ilr, const vector<ubond_t> &im,
        vector<vector<int>> &idx_dm_to_wfn) {
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
            kkw += (int)idx_dm_to_wfn[ilr[i]].size();
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
    static FPS split_wavefunction_svd(
        S opdq, const shared_ptr<SparseMatrix<S, FLS>> &wfn, int k,
        bool trace_right, bool normalize,
        shared_ptr<SparseMatrix<S, FLS>> &left,
        shared_ptr<SparseMatrix<S, FLS>> &right, FPS cutoff,
        bool store_wfn_spectra, vector<FPS> &wfn_spectra,
        TruncationTypes trunc_type = TruncationTypes::Physical,
        DecompositionTypes decomp_type = DecompositionTypes::SVD,
        const shared_ptr<SparseMatrixGroup<S, FLS>> &mwfn = nullptr,
        const vector<shared_ptr<SparseMatrix<S, FLS>>> &xwfns =
            vector<shared_ptr<SparseMatrix<S, FLS>>>(),
        const vector<FPS> &weights = vector<FPS>()) {
        vector<shared_ptr<GTensor<FLS>>> l, r;
        vector<shared_ptr<GTensor<FPS>>> s;
        vector<S> qs;
        // for perturbative SVD
        if (mwfn != nullptr) {
            vector<vector<shared_ptr<GTensor<FLS>>>> xlr;
            vector<shared_ptr<SparseMatrix<S, FLS>>> xxwfns = {wfn};
            if (xwfns.size() != 0)
                xxwfns.insert(xxwfns.end(), xwfns.begin(), xwfns.end());
            if (trace_right) {
                mwfn->right_svd(qs, l, s, xlr, xxwfns, weights);
                r = xlr.back();
            } else {
                mwfn->left_svd(qs, xlr, s, r, xxwfns, weights);
                l = xlr.back();
            }
        } else if (xwfns.size() != 0) {
            vector<vector<shared_ptr<GTensor<FLS>>>> xlr;
            shared_ptr<SparseMatrixGroup<S, FLS>> xmwfn =
                make_shared<SparseMatrixGroup<S, FLS>>();
            xmwfn->allocate(vector<shared_ptr<SparseMatrixInfo<S>>>());
            vector<shared_ptr<SparseMatrix<S, FLS>>> xxwfns = {wfn};
            xxwfns.insert(xxwfns.end(), xwfns.begin(), xwfns.end());
            if (trace_right) {
                xmwfn->right_svd(qs, l, s, xlr, xxwfns, weights);
                r = xlr.back();
            } else {
                xmwfn->left_svd(qs, xlr, s, r, xxwfns, weights);
                l = xlr.back();
            }
        } else {
            if (trace_right)
                wfn->right_svd(qs, l, s, r);
            else
                wfn->left_svd(qs, l, s, r);
        }
        // ss: pair<quantum index in dm, reduced matrix index in dm>
        vector<pair<int, int>> ss;
        FPS error = truncate_singular_values(
            qs, s, ss, k, cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
        // ilr: row index in singular values list
        // im: number of states
        vector<int> ilr;
        vector<ubond_t> im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (ss.size() != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        shared_ptr<SparseMatrixInfo<S>> linfo, rinfo;
        vector<vector<int>> idx_dm_to_wfn;
        if (trace_right) {
            linfo = rotation_matrix_info_from_svd(opdq, qs, l, true, ilr, im);
            rinfo = wavefunction_info_from_svd(qs, wfn->info, true, ilr, im,
                                               idx_dm_to_wfn);
        } else {
            linfo = wavefunction_info_from_svd(qs, wfn->info, false, ilr, im,
                                               idx_dm_to_wfn);
            rinfo = rotation_matrix_info_from_svd(opdq, qs, r, false, ilr, im);
        }
        int kk = (int)ilr.size();
        left = make_shared<SparseMatrix<S, FLS>>();
        right = make_shared<SparseMatrix<S, FLS>>();
        left->allocate(linfo);
        right->allocate(rinfo);
        int iss = 0;
        if (trace_right) {
            for (int i = 0; i < kk; i++) {
                for (ubond_t j = 0; j < im[i]; j++)
                    GMatrixFunctions<FLS>::copy(
                        GMatrix<FLS>(left->data + linfo->n_states_total[i] + j,
                                     linfo->n_states_bra[i], 1),
                        GMatrix<FLS>(
                            &l[ss[iss + j].first]->ref()(0, ss[iss + j].second),
                            linfo->n_states_bra[i], 1),
                        linfo->n_states_ket[i], l[ss[iss + j].first]->shape[1]);
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int ir = rinfo->find_state(wfn->info->quanta[iw]);
                    assert(ir != -1);
                    if (decomp_type == DecompositionTypes::PureSVD) {
                        for (ubond_t j = 0; j < im[i]; j++) {
                            GMatrixFunctions<FLS>::copy(
                                GMatrix<FLS>(right->data +
                                                 rinfo->n_states_total[ir] +
                                                 j * r[iw]->shape[1],
                                             1, r[iw]->shape[1]),
                                GMatrix<FLS>(
                                    &r[iw]->ref()(ss[iss + j].second, 0), 1,
                                    r[iw]->shape[1]));
                            GMatrixFunctions<FLS>::iscale(
                                GMatrix<FLS>(right->data +
                                                 rinfo->n_states_total[ir] +
                                                 j * r[iw]->shape[1],
                                             1, r[iw]->shape[1]),
                                (*s[ss[iss + j].first]
                                      ->data)[ss[iss + j].second]);
                        }
                    } else
                        GMatrixFunctions<FLS>::multiply((*left)[i], 3,
                                                        (*wfn)[iw], false,
                                                        (*right)[ir], 1.0, 0.0);
                }
                iss += im[i];
            }
            if (normalize)
                right->normalize();
        } else {
            for (int i = 0; i < kk; i++) {
                for (ubond_t j = 0; j < im[i]; j++)
                    GMatrixFunctions<FLS>::copy(
                        GMatrix<FLS>(right->data + rinfo->n_states_total[i] +
                                         j * r[ss[iss + j].first]->shape[1],
                                     1, r[ss[iss + j].first]->shape[1]),
                        GMatrix<FLS>(
                            &r[ss[iss + j].first]->ref()(ss[iss + j].second, 0),
                            1, r[ss[iss + j].first]->shape[1]));
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int il = linfo->find_state(wfn->info->quanta[iw]);
                    assert(il != -1);
                    if (decomp_type == DecompositionTypes::PureSVD) {
                        for (ubond_t j = 0; j < im[i]; j++) {
                            GMatrixFunctions<FLS>::copy(
                                GMatrix<FLS>(left->data +
                                                 linfo->n_states_total[il] + j,
                                             linfo->n_states_bra[il], 1),
                                GMatrix<FLS>(
                                    &l[iw]->ref()(0, ss[iss + j].second),
                                    linfo->n_states_bra[il], 1),
                                linfo->n_states_ket[il], l[iw]->shape[1]);
                            GMatrixFunctions<FLS>::iscale(
                                GMatrix<FLS>(left->data +
                                                 linfo->n_states_total[il] + j,
                                             linfo->n_states_bra[il], 1),
                                (*s[ss[iss + j].first]
                                      ->data)[ss[iss + j].second],
                                linfo->n_states_ket[il]);
                        }
                    } else
                        GMatrixFunctions<FLS>::multiply((*wfn)[iw], false,
                                                        (*right)[i], 3,
                                                        (*left)[il], 1.0, 0.0);
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
    static FPS split_density_matrix(
        const shared_ptr<SparseMatrix<S, FLS>> &dm,
        const shared_ptr<SparseMatrix<S, FLS>> &wfn, int k, bool trace_right,
        bool normalize, shared_ptr<SparseMatrix<S, FLS>> &left,
        shared_ptr<SparseMatrix<S, FLS>> &right, FPS cutoff,
        bool store_wfn_spectra, vector<FPS> &wfn_spectra,
        TruncationTypes trunc_type = TruncationTypes::Physical) {
        // ss: pair<quantum index in dm, reduced matrix index in dm>
        vector<pair<int, int>> ss;
        FPS error = truncate_density_matrix(
            dm, ss, k, cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
        // ilr: row index in dm
        // im: number of states
        vector<int> ilr;
        vector<ubond_t> im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (ss.size() != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        shared_ptr<SparseMatrixInfo<S>> linfo, rinfo;
        vector<vector<int>> idx_dm_to_wfn;
        if (trace_right) {
            linfo = rotation_matrix_info_from_density_matrix(dm->info, true,
                                                             ilr, im);
            rinfo = wavefunction_info_from_density_matrix(
                dm->info, wfn->info, true, ilr, im, idx_dm_to_wfn);
        } else {
            linfo = wavefunction_info_from_density_matrix(
                dm->info, wfn->info, false, ilr, im, idx_dm_to_wfn);
            rinfo = rotation_matrix_info_from_density_matrix(dm->info, false,
                                                             ilr, im);
        }
        int kk = (int)ilr.size();
        left = make_shared<SparseMatrix<S, FLS>>();
        right = make_shared<SparseMatrix<S, FLS>>();
        left->allocate(linfo);
        right->allocate(rinfo);
        int iss = 0;
        if (trace_right) {
            for (int i = 0; i < kk; i++) {
                for (ubond_t j = 0; j < im[i]; j++)
                    GMatrixFunctions<FLS>::copy(
                        GMatrix<FLS>(left->data + linfo->n_states_total[i] + j,
                                     linfo->n_states_bra[i], 1),
                        GMatrix<FLS>(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0),
                            linfo->n_states_bra[i], 1),
                        linfo->n_states_ket[i], 1);
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int ir = right->info->find_state(wfn->info->quanta[iw]);
                    assert(ir != -1);
                    GMatrixFunctions<FLS>::multiply((*left)[i], 3, (*wfn)[iw],
                                                    false, (*right)[ir], 1.0,
                                                    0.0);
                }
                iss += im[i];
            }
            if (normalize)
                right->normalize();
        } else {
            for (int i = 0; i < dm->info->n; i++)
                GMatrixFunctions<FLS>::conjugate((*dm)[i]);
            for (int i = 0; i < kk; i++) {
                for (ubond_t j = 0; j < im[i]; j++)
                    GMatrixFunctions<FLS>::copy(
                        GMatrix<FLS>(right->data + rinfo->n_states_total[i] +
                                         j * (*right)[i].n,
                                     1, (*right)[i].n),
                        GMatrix<FLS>(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0), 1,
                            (*right)[i].n));
                for (int iww = 0;
                     iww < (int)idx_dm_to_wfn[ss[iss].first].size(); iww++) {
                    int iw = idx_dm_to_wfn[ss[iss].first][iww];
                    int il = left->info->find_state(wfn->info->quanta[iw]);
                    assert(il != -1);
                    GMatrixFunctions<FLS>::multiply((*wfn)[iw], false,
                                                    (*right)[i], 3, (*left)[il],
                                                    1.0, 0.0);
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
    static FPS multi_split_density_matrix(
        const shared_ptr<SparseMatrix<S, FLS>> &dm,
        const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &wfns, int k,
        bool trace_right, bool normalize,
        vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &new_wfns,
        shared_ptr<SparseMatrix<S, FLS>> &rot_mat, FPS cutoff,
        bool store_wfn_spectra, vector<FPS> &wfn_spectra,
        TruncationTypes trunc_type = TruncationTypes::Physical) {
        // ss: pair<quantum index in dm, reduced matrix index in dm>
        vector<pair<int, int>> ss;
        FPS error = truncate_density_matrix(
            dm, ss, k, cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
        // ilr: row index in dm
        // im: number of states
        vector<int> ilr;
        vector<ubond_t> im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (ss.size() != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        shared_ptr<SparseMatrixInfo<S>> rinfo;
        vector<shared_ptr<SparseMatrixInfo<S>>> winfos;
        vector<vector<vector<int>>> idx_dm_to_wfns;
        idx_dm_to_wfns.resize(wfns[0]->n);
        if (trace_right)
            rinfo = rotation_matrix_info_from_density_matrix(
                dm->info, trace_right, ilr, im);
        winfos.resize(wfns[0]->n);
        for (size_t j = 0; j < wfns[0]->n; j++) {
            winfos[j] = wavefunction_info_from_density_matrix(
                dm->info, wfns[0]->infos[j], trace_right, ilr, im,
                idx_dm_to_wfns[j]);
        }
        if (!trace_right)
            rinfo = rotation_matrix_info_from_density_matrix(
                dm->info, trace_right, ilr, im);
        int kk = (int)ilr.size();
        rot_mat = make_shared<SparseMatrix<S, FLS>>();
        new_wfns =
            vector<shared_ptr<SparseMatrixGroup<S, FLS>>>(wfns.size(), nullptr);
        if (trace_right)
            rot_mat->allocate(rinfo);
        for (size_t k = 0; k < wfns.size(); k++) {
            new_wfns[k] = make_shared<SparseMatrixGroup<S, FLS>>();
            new_wfns[k]->allocate(winfos);
        }
        if (!trace_right)
            rot_mat->allocate(rinfo);
        int iss = 0;
        if (trace_right) {
            for (int i = 0; i < kk; i++) {
                for (ubond_t j = 0; j < im[i]; j++)
                    GMatrixFunctions<FLS>::copy(
                        GMatrix<FLS>(rot_mat->data + rinfo->n_states_total[i] +
                                         j,
                                     rinfo->n_states_bra[i], 1),
                        GMatrix<FLS>(
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
                            GMatrixFunctions<FLS>::multiply(
                                (*rot_mat)[i], 3, (*(*wfns[k])[j])[iw], false,
                                (*(*new_wfns[k])[j])[ir], 1.0, 0.0);
                        }
                iss += im[i];
            }
            if (normalize)
                for (size_t k = 0; k < new_wfns.size(); k++)
                    new_wfns[k]->normalize();
        } else {
            for (int i = 0; i < kk; i++) {
                for (ubond_t j = 0; j < im[i]; j++)
                    GMatrixFunctions<FLS>::copy(
                        GMatrix<FLS>(rot_mat->data + rinfo->n_states_total[i] +
                                         j * (*rot_mat)[i].n,
                                     1, (*rot_mat)[i].n),
                        GMatrix<FLS>(
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
                            GMatrixFunctions<FLS>::multiply(
                                (*(*wfns[k])[j])[iw], false, (*rot_mat)[i], 3,
                                (*(*new_wfns[k])[j])[il], 1.0, 0.0);
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
    static shared_ptr<SparseMatrix<S, FLS>>
    swap_wfn_to_fused_left(int i, const shared_ptr<MPSInfo<S>> &mps_info,
                           const shared_ptr<SparseMatrix<S, FLS>> &old_wfn,
                           const shared_ptr<CG<S>> &cg) {
        return mps_info->swap_wfn_to_fused_left(i, old_wfn, cg);
    }
    static shared_ptr<SparseMatrix<S, FLS>>
    swap_wfn_to_fused_right(int i, const shared_ptr<MPSInfo<S>> &mps_info,
                            const shared_ptr<SparseMatrix<S, FLS>> &old_wfn,
                            const shared_ptr<CG<S>> &cg) {
        return mps_info->swap_wfn_to_fused_right(i, old_wfn, cg);
    }
    static vector<shared_ptr<SparseMatrixGroup<S, FLS>>>
    swap_multi_wfn_to_fused_left(
        int i, const shared_ptr<MPSInfo<S>> &mps_info,
        const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &old_wfns,
        const shared_ptr<CG<S>> &cg) {
        return mps_info->swap_multi_wfn_to_fused_left(i, old_wfns, cg);
    }
    static vector<shared_ptr<SparseMatrixGroup<S, FLS>>>
    swap_multi_wfn_to_fused_right(
        int i, const shared_ptr<MPSInfo<S>> &mps_info,
        const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &old_wfns,
        const shared_ptr<CG<S>> &cg) {
        return mps_info->swap_multi_wfn_to_fused_right(i, old_wfns, cg);
    }
    // Change the fusing type of MPS tensor so that it can be used in next sweep
    // iteration
    static void propagate_wfn(int i, int start_site, int end_site,
                              const shared_ptr<MPS<S, FLS>> &mps, bool forward,
                              const shared_ptr<CG<S>> &cg) {
        if (forward) {
            if (i + 1 != end_site - 1) {
                mps->load_tensor(i + 1);
                shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i + 1];
                mps->tensors[i + 1] =
                    swap_wfn_to_fused_left(i + 1, mps->info, old_wfn, cg);
                mps->save_tensor(i + 1);
                mps->unload_tensor(i + 1);
                old_wfn->info->deallocate();
                old_wfn->deallocate();
            }
        } else {
            if (i != start_site) {
                mps->load_tensor(i);
                shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i];
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
    static void propagate_multi_wfn(int i, int start_site, int end_site,
                                    const shared_ptr<MultiMPS<S, FLS>> &mps,
                                    bool forward, const shared_ptr<CG<S>> &cg) {
        if (forward) {
            if (i + 1 != end_site - 1) {
                mps->load_wavefunction(i + 1);
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfns =
                    mps->wfns;
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
            if (i != start_site) {
                mps->load_wavefunction(i);
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfns =
                    mps->wfns;
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
