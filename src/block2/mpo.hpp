
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

#include "ancilla.hpp"
#include "operator_tensor.hpp"
#include "symbolic.hpp"
#include "tensor_functions.hpp"
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace block2 {

// Information for middle site numerical tranform
// from normal operators to complementary operators
template <typename S> struct MPOSchemer {
    uint16_t left_trans_site, right_trans_site;
    shared_ptr<SymbolicRowVector<S>> left_new_operator_names;
    shared_ptr<SymbolicColumnVector<S>> right_new_operator_names;
    shared_ptr<SymbolicRowVector<S>> left_new_operator_exprs;
    shared_ptr<SymbolicColumnVector<S>> right_new_operator_exprs;
    MPOSchemer(uint16_t left_trans_site, uint16_t right_trans_site)
        : left_trans_site(left_trans_site), right_trans_site(right_trans_site) {
    }
    shared_ptr<MPOSchemer> copy() const {
        shared_ptr<MPOSchemer> r =
            make_shared<MPOSchemer>(left_trans_site, right_trans_site);
        r->left_new_operator_names = left_new_operator_names;
        r->right_new_operator_names = right_new_operator_names;
        r->left_new_operator_exprs = left_new_operator_exprs;
        r->right_new_operator_exprs = right_new_operator_exprs;
        return r;
    }
    string get_transform_formulas() const {
        stringstream ss;
        ss << "LEFT  TRANSFORM :: SITE = " << (int)left_trans_site << endl;
        for (int j = 0; j < left_new_operator_names->data.size(); j++) {
            if (left_new_operator_names->data[j]->get_type() != OpTypes::Zero)
                ss << "[" << setw(4) << j << "] " << setw(15)
                   << left_new_operator_names->data[j]
                   << " := " << left_new_operator_exprs->data[j] << endl;
            else
                ss << "[" << setw(4) << j << "] "
                   << left_new_operator_names->data[j] << endl;
        }
        ss << endl;
        ss << "RIGHT TRANSFORM :: SITE = " << (int)right_trans_site << endl;
        for (int j = 0; j < right_new_operator_names->data.size(); j++) {
            if (right_new_operator_names->data[j]->get_type() != OpTypes::Zero)
                ss << "[" << setw(4) << j << "] " << setw(15)
                   << right_new_operator_names->data[j]
                   << " := " << right_new_operator_exprs->data[j] << endl;
            else
                ss << "[" << setw(4) << j << "] "
                   << right_new_operator_names->data[j] << endl;
        }
        ss << endl;
        return ss.str();
    }
};

// Symbolic Matrix Product Operator
template <typename S> struct MPO {
    vector<shared_ptr<OperatorTensor<S>>> tensors;
    vector<shared_ptr<Symbolic<S>>> left_operator_names;
    vector<shared_ptr<Symbolic<S>>> right_operator_names;
    vector<shared_ptr<Symbolic<S>>> middle_operator_names;
    vector<shared_ptr<Symbolic<S>>> left_operator_exprs;
    vector<shared_ptr<Symbolic<S>>> right_operator_exprs;
    vector<shared_ptr<Symbolic<S>>> middle_operator_exprs;
    shared_ptr<OpElement<S>> op;
    shared_ptr<MPOSchemer<S>> schemer;
    // Number of sites
    int n_sites;
    // Const energy term
    double const_e;
    shared_ptr<TensorFunctions<S>> tf;
    vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>> site_op_infos;
    MPO(int n_sites)
        : n_sites(n_sites), const_e(0.0), op(nullptr), schemer(nullptr),
          tf(nullptr) {}
    virtual AncillaTypes get_ancilla_type() const { return AncillaTypes::None; }
    virtual void deallocate() {}
    string get_blocking_formulas() const {
        stringstream ss;
        for (int i = 0; i < n_sites; i++) {
            ss << "LEFT BLOCKING :: SITE = " << i << endl;
            for (int j = 0; j < left_operator_names[i]->data.size(); j++) {
                if (left_operator_exprs.size() != 0)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << left_operator_names[i]->data[j]
                       << " := " << left_operator_exprs[i]->data[j] << endl;
                else
                    ss << "[" << setw(4) << j << "] "
                       << left_operator_names[i]->data[j] << endl;
            }
            ss << endl;
        }
        for (int i = n_sites - 1; i >= 0; i--) {
            ss << "RIGHT BLOCKING :: SITE = " << i << endl;
            for (int j = 0; j < right_operator_names[i]->data.size(); j++) {
                if (right_operator_exprs.size() != 0)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << right_operator_names[i]->data[j]
                       << " := " << right_operator_exprs[i]->data[j] << endl;
                else
                    ss << "[" << setw(4) << j << "] "
                       << right_operator_names[i]->data[j] << endl;
            }
            ss << endl;
        }
        if (middle_operator_names.size() != 0) {
            for (int i = 0; i < n_sites - 1; i++) {
                ss << "HAMIL PARTITION :: SITE = " << i << endl;
                for (int j = 0; j < middle_operator_names[i]->data.size(); j++)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << middle_operator_names[i]->data[j]
                       << " := " << middle_operator_exprs[i]->data[j] << endl;
                ss << endl;
            }
        }
        if (schemer != nullptr)
            ss << schemer->get_transform_formulas() << endl;
        return ss.str();
    }
};

// Adding ancilla (identity) sites to a MPO
// n_sites = 2 * n_physical_sites
template <typename S> struct AncillaMPO : MPO<S> {
    int n_physical_sites;
    shared_ptr<MPO<S>> prim_mpo;
    AncillaMPO(const shared_ptr<MPO<S>> &mpo, bool npdm = false)
        : n_physical_sites(mpo->n_sites),
          prim_mpo(mpo), MPO<S>(mpo->n_sites << 1) {
        const auto n_sites = MPO<S>::n_sites;
        const shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), S());
        MPO<S>::const_e = mpo->const_e;
        MPO<S>::op = mpo->op;
        MPO<S>::tf = mpo->tf;
        MPO<S>::site_op_infos = vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(n_sites);
        for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
            MPO<S>::site_op_infos[j] = mpo->site_op_infos[i];
            MPO<S>::site_op_infos[j + 1] = mpo->site_op_infos[i];
        }
        // operator names
        MPO<S>::left_operator_names.resize(n_sites, nullptr);
        MPO<S>::right_operator_names.resize(n_sites, nullptr);
        for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
            MPO<S>::left_operator_names[j] = mpo->left_operator_names[i];
            MPO<S>::left_operator_names[j + 1] =
                MPO<S>::left_operator_names[j]->copy();
            MPO<S>::right_operator_names[j] = mpo->right_operator_names[i];
            if (j - 1 >= 0)
                MPO<S>::right_operator_names[j - 1] =
                    MPO<S>::right_operator_names[j]->copy();
        }
        MPO<S>::right_operator_names[n_sites - 1] =
            make_shared<SymbolicColumnVector<S>>(1);
        MPO<S>::right_operator_names[n_sites - 1]->data[0] = i_op;
        // middle operators
        if (mpo->middle_operator_names.size() != 0) {
            assert(mpo->schemer == nullptr);
            MPO<S>::middle_operator_names.resize(n_sites - 1);
            MPO<S>::middle_operator_exprs.resize(n_sites - 1);
            shared_ptr<SymbolicColumnVector<S>> zero_mat =
                make_shared<SymbolicColumnVector<S>>(1);
            (*zero_mat)[0] =
                make_shared<OpElement<S>>(OpNames::Zero, SiteIndex(), S());
            shared_ptr<SymbolicColumnVector<S>> zero_expr =
                make_shared<SymbolicColumnVector<S>>(1);
            (*zero_expr)[0] = make_shared<OpExpr<S>>();
            for (int i = 0, j = 0; i < n_physical_sites - 1; i++, j += 2) {
                MPO<S>::middle_operator_names[j] =
                    mpo->middle_operator_names[i];
                MPO<S>::middle_operator_exprs[j] =
                    mpo->middle_operator_exprs[i];
                if (!npdm) {
                    MPO<S>::middle_operator_names[j + 1] =
                        mpo->middle_operator_names[i];
                    MPO<S>::middle_operator_exprs[j + 1] =
                        mpo->middle_operator_exprs[i];
                } else {
                    MPO<S>::middle_operator_names[j + 1] = zero_mat;
                    MPO<S>::middle_operator_exprs[j + 1] = zero_expr;
                }
            }
            if (mpo->op != nullptr && mpo->op->name != OpNames::Zero) {
                shared_ptr<SymbolicColumnVector<S>> hop_mat =
                    make_shared<SymbolicColumnVector<S>>(1);
                (*hop_mat)[0] = mpo->op;
                shared_ptr<SymbolicColumnVector<S>> hop_expr =
                    make_shared<SymbolicColumnVector<S>>(1);
                (*hop_expr)[0] = (shared_ptr<OpExpr<S>>)mpo->op * i_op;
                MPO<S>::middle_operator_names[n_sites - 2] = hop_mat;
                MPO<S>::middle_operator_exprs[n_sites - 2] = hop_expr;
            } else {
                MPO<S>::middle_operator_names[n_sites - 2] = zero_mat;
                MPO<S>::middle_operator_exprs[n_sites - 2] = zero_expr;
            }
        }
        // operator tensors
        MPO<S>::tensors.resize(n_sites, nullptr);
        for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
            MPO<S>::tensors[j + 1] = make_shared<OperatorTensor<S>>();
            if (j + 1 != n_sites - 1) {
                MPO<S>::tensors[j] = mpo->tensors[i];
                int rshape = MPO<S>::tensors[j]->lmat->n;
                MPO<S>::tensors[j + 1]->lmat = MPO<S>::tensors[j + 1]->rmat =
                    make_shared<SymbolicMatrix<S>>(rshape, rshape);
                for (int k = 0; k < rshape; k++)
                    (*MPO<S>::tensors[j + 1]->lmat)[{k, k}] = i_op;
                if (mpo->tensors[i]->lmat != mpo->tensors[i]->rmat &&
                    !(mpo->schemer != nullptr &&
                      mpo->schemer->right_trans_site -
                              mpo->schemer->left_trans_site ==
                          2)) {
                    int lshape = mpo->tensors[i + 1]->rmat->m;
                    MPO<S>::tensors[j + 1]->rmat =
                        make_shared<SymbolicMatrix<S>>(lshape, lshape);
                    for (int k = 0; k < lshape; k++)
                        (*MPO<S>::tensors[j + 1]->rmat)[{k, k}] = i_op;
                }
            } else {
                int lshape = mpo->tensors[i]->lmat->m;
                MPO<S>::tensors[j] = make_shared<OperatorTensor<S>>();
                MPO<S>::tensors[j]->lmat = MPO<S>::tensors[j]->rmat =
                    make_shared<SymbolicMatrix<S>>(lshape, 1);
                for (int k = 0; k < lshape; k++)
                    (*MPO<S>::tensors[j]->lmat)[{k, 0}] =
                        mpo->tensors[i]->lmat->data[k];
                if (mpo->tensors[i]->lmat != mpo->tensors[i]->rmat) {
                    lshape = mpo->tensors[i]->rmat->m;
                    MPO<S>::tensors[j]->rmat =
                        make_shared<SymbolicMatrix<S>>(lshape, 1);
                    for (int k = 0; k < lshape; k++)
                        (*MPO<S>::tensors[j]->rmat)[{k, 0}] =
                            mpo->tensors[i]->rmat->data[k];
                }
                MPO<S>::tensors[j]->ops = mpo->tensors[i]->ops;
                MPO<S>::tensors[j + 1]->lmat = MPO<S>::tensors[j + 1]->rmat =
                    make_shared<SymbolicColumnVector<S>>(1);
                MPO<S>::tensors[j + 1]->lmat->data[0] = i_op;
            }
            MPO<S>::tensors[j + 1]->ops[i_op] =
                MPO<S>::tensors[j]->ops.at(i_op);
        }
        // numerical transform
        if (mpo->schemer != nullptr &&
            mpo->schemer->right_trans_site - mpo->schemer->left_trans_site ==
                2) {
            MPO<S>::schemer = mpo->schemer->copy();
            if (n_physical_sites & 1) {
                MPO<S>::schemer->left_trans_site = n_physical_sites - 2;
                MPO<S>::schemer->right_trans_site = n_physical_sites;
            } else {
                MPO<S>::schemer->left_trans_site = n_physical_sites - 1;
                MPO<S>::schemer->right_trans_site = n_physical_sites + 1;
            }
        } else if (mpo->schemer != nullptr)
            assert(false);
        else
            MPO<S>::schemer = nullptr;
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

} // namespace block2
