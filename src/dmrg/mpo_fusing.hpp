
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

#include "../core/allocator.hpp"
#include "mpo.hpp"
#include "partition.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#define TINY (1E-20)

using namespace std;

namespace block2 {

// Fuse adjacent mpo sites to one site
// MPO must be unsimplified
// Currently only edge sites are allowed to be fused
template <typename S, typename FL> struct FusedMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::tensors;
    using MPO<S, FL>::site_op_infos;
    using MPO<S, FL>::left_operator_names;
    using MPO<S, FL>::right_operator_names;
    AncillaTypes ancilla_type;
    FusedMPO(const shared_ptr<MPO<S, FL>> &mpo,
             const vector<shared_ptr<StateInfo<S>>> &basis, uint16_t a,
             uint16_t b, const shared_ptr<StateInfo<S>> &ref = nullptr,
             const string &tag = "")
        : MPO<S, FL>(mpo->n_sites - 1, tag == "" ? mpo->tag : tag) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(b == a + 1);
        assert(mpo->n_sites == basis.size());
        assert(mpo->left_operator_exprs.size() == 0);
        assert(mpo->right_operator_exprs.size() == 0);
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::op = mpo->op;
        if (mpo->schemer == nullptr)
            MPO<S, FL>::schemer = nullptr;
        else {
            mpo->load_schemer();
            MPO<S, FL>::schemer =
                frame->minimal_memory_usage
                    ? make_shared<MPOSchemer<S>>(*mpo->schemer)
                    : mpo->schemer->copy();
            mpo->unload_schemer();
        }
        MPO<S, FL>::tf = mpo->tf;
        ancilla_type = mpo->get_ancilla_type();
        char fused_sparse_form =
            mpo->sparse_form[a] == 'N' && mpo->sparse_form[b] == 'N' ? 'N'
                                                                     : 'S';
        mpo->load_tensor(a, true);
        mpo->load_tensor(b, true);
        assert(mpo->tensors[a]->lmat == mpo->tensors[a]->rmat);
        assert(mpo->tensors[b]->lmat == mpo->tensors[b]->rmat);
        shared_ptr<Symbolic<S>> fused_mat =
            mpo->tensors[a]->lmat * mpo->tensors[b]->lmat;
        mpo->unload_tensor(b);
        mpo->unload_tensor(a);
        assert(fused_mat->m == 1 || fused_mat->n == 1);
        shared_ptr<StateInfo<S>> fused_basis = nullptr;
        if (ref == nullptr)
            fused_basis = make_shared<StateInfo<S>>(
                StateInfo<S>::tensor_product(*basis[a], *basis[b], S::invalid));
        else
            fused_basis = make_shared<StateInfo<S>>(
                StateInfo<S>::tensor_product(*basis[a], *basis[b], *ref));
        shared_ptr<StateInfo<S>> fused_cinfo =
            make_shared<StateInfo<S>>(StateInfo<S>::get_connection_info(
                *basis[a], *basis[b], *fused_basis));
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> fused_op_infos;
        shared_ptr<OperatorTensor<S, FL>> opt =
            make_shared<OperatorTensor<S, FL>>();
        vector<shared_ptr<Symbolic<S>>> mats(1);
        if (fused_mat->m == 1) {
            // left contract infos
            mpo->load_left_operators(b);
            mats[0] = mpo->left_operator_names[b];
            mpo->unload_left_operators(b);
            assert(mats[0] != nullptr);
            assert(mats[0]->get_type() == SymTypes::RVec);
            opt->lmat = make_shared<SymbolicRowVector<S>>(
                *dynamic_pointer_cast<SymbolicRowVector<S>>(mats[0]));
        } else {
            // right contract infos
            mpo->load_right_operators(a);
            mats[0] = mpo->right_operator_names[a];
            mpo->unload_right_operators(a);
            assert(mats[0] != nullptr);
            assert(mats[0]->get_type() == SymTypes::CVec);
            opt->lmat = make_shared<SymbolicColumnVector<S>>(
                *dynamic_pointer_cast<SymbolicColumnVector<S>>(mats[0]));
        }
        vector<S> sl = Partition<S, FL>::get_uniq_labels(mats);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S, FL>::get_uniq_sub_labels(fused_mat, mats[0], sl);
        // site info
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> op_notrunc =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            fused_op_infos.push_back(make_pair(sl[i], op_notrunc));
            op_notrunc->initialize(*fused_basis, *fused_basis, sl[i],
                                   sl[i].is_fermion());
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            cinfo->initialize_tp(sl[i], subsl[i], *fused_basis, *fused_basis,
                                 *basis[a], *basis[b], *basis[a], *basis[b],
                                 *fused_cinfo, *fused_cinfo,
                                 mpo->site_op_infos[a], mpo->site_op_infos[b],
                                 op_notrunc, mpo->tf->opf->cg);
            op_notrunc->cinfo = cinfo;
        }
        // build
        opt->rmat = opt->lmat;
        for (auto &mat : mats) {
            for (size_t i = 0; i < mat->data.size(); i++)
                if (mat->data[i]->get_type() != OpTypes::Zero) {
                    shared_ptr<OpExpr<S>> op = abs_value(mat->data[i]);
                    opt->ops[op] = fused_sparse_form == 'N'
                                       ? make_shared<SparseMatrix<S, FL>>()
                                       : make_shared<CSRSparseMatrix<S, FL>>();
                }
        }
        // here main stack is not used
        // but when frame->use_main_stack == false:
        // tf->left/right_contract will skip allocated matrices if alloc !=
        // nullptr
        for (auto &p : opt->ops) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            if (frame->use_main_stack)
                p.second->alloc = d_alloc;
            p.second->info =
                Partition<S, FL>::find_op_info(fused_op_infos, op->q_label);
        }
        mpo->load_tensor(a);
        mpo->load_tensor(b);
        // contract
        if (fused_mat->m == 1)
            mpo->tf->left_contract(mpo->tensors[a], mpo->tensors[b], opt,
                                   nullptr);
        else
            mpo->tf->right_contract(mpo->tensors[b], mpo->tensors[a], opt,
                                    nullptr);
        mpo->unload_tensor(b);
        mpo->unload_tensor(a);
        for (int i = (int)fused_op_infos.size() - 1; i >= 0; i--)
            if (fused_op_infos[i].second->cinfo != nullptr)
                fused_op_infos[i].second->cinfo->deallocate();
        this->sparse_form = "";
        for (uint16_t m = 0; m < mpo->n_sites; m++) {
            if (m == a) {
                site_op_infos.push_back(fused_op_infos);
                tensors.push_back(opt);
                this->save_tensor((int)tensors.size() - 1);
                this->unload_tensor((int)tensors.size() - 1);
                this->basis.push_back(fused_basis);
                mpo->load_right_operators(m);
                right_operator_names.push_back(mpo->right_operator_names[m]);
                mpo->unload_right_operators(m);
                this->sparse_form.push_back(fused_sparse_form);
                this->save_right_operators((int)right_operator_names.size() -
                                           1);
                this->unload_right_operators((int)right_operator_names.size() -
                                             1);
            } else if (m != b) {
                site_op_infos.push_back(mpo->site_op_infos[m]);
                mpo->load_tensor(m);
                tensors.push_back(mpo->tensors[m]);
                mpo->unload_tensor(m);
                this->save_tensor((int)tensors.size() - 1);
                this->unload_tensor((int)tensors.size() - 1);
                this->basis.push_back(basis[m]);
                mpo->load_left_operators(m);
                left_operator_names.push_back(mpo->left_operator_names[m]);
                mpo->unload_left_operators(m);
                this->save_left_operators((int)left_operator_names.size() - 1);
                this->unload_left_operators((int)left_operator_names.size() -
                                            1);
                mpo->load_right_operators(m);
                right_operator_names.push_back(mpo->right_operator_names[m]);
                mpo->unload_right_operators(m);
                this->save_right_operators((int)right_operator_names.size() -
                                           1);
                this->unload_right_operators((int)right_operator_names.size() -
                                             1);
                this->sparse_form.push_back(mpo->sparse_form[m]);
            } else {
                mpo->load_left_operators(m);
                left_operator_names.push_back(mpo->left_operator_names[m]);
                mpo->unload_left_operators(m);
                this->save_left_operators((int)left_operator_names.size() - 1);
                this->unload_left_operators((int)left_operator_names.size() -
                                            1);
            }
        }
        if (this->schemer != nullptr && this->schemer->left_trans_site >= b)
            this->schemer->left_trans_site--;
        if (this->schemer != nullptr && this->schemer->right_trans_site >= b)
            this->schemer->right_trans_site--;
        this->save_schemer();
        this->unload_schemer();
    }
    AncillaTypes get_ancilla_type() const override { return ancilla_type; }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            if (this->tensors[m] != nullptr)
                this->tensors[m]->deallocate();
    }
};

} // namespace block2
