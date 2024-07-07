
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

#include "../core/clebsch_gordan.hpp"
#include "../core/csr_sparse_matrix.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/sparse_matrix.hpp"
#include "mps.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

// Represent a specific partition of left block and right block
template <typename S, typename FL> struct Partition {
    typedef typename GMatrix<FL>::FP FP;
    // Operator tensor formed by contraction of left block MPO tensors
    shared_ptr<OperatorTensor<S, FL>> left;
    // Operator tensor formed by contraction of right block MPO tensors
    shared_ptr<OperatorTensor<S, FL>> right;
    // MPO tensors in dot block(s)
    vector<shared_ptr<OperatorTensor<S, FL>>> middle;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> right_op_infos;
    Partition(const shared_ptr<OperatorTensor<S, FL>> &left,
              const shared_ptr<OperatorTensor<S, FL>> &right,
              const shared_ptr<OperatorTensor<S, FL>> &dot)
        : left(left), right(right), middle{dot} {}
    Partition(const shared_ptr<OperatorTensor<S, FL>> &left,
              const shared_ptr<OperatorTensor<S, FL>> &right,
              const shared_ptr<OperatorTensor<S, FL>> &ldot,
              const shared_ptr<OperatorTensor<S, FL>> &rdot)
        : left(left), right(right), middle{ldot, rdot} {}
    Partition(const Partition &other)
        : left(other.left), right(other.right), middle(other.middle) {}
    void load_data(istream &ifs, bool left_part) {
        if (left_part) {
            uint8_t has_left;
            ifs.read((char *)&has_left, sizeof(has_left));
            if (has_left) {
                left = make_shared<OperatorTensor<S, FL>>();
                left->load_data(ifs, true);
            } else
                left = nullptr;
            left_op_infos = Partition::load_op_infos(ifs);
        } else {
            uint8_t has_right;
            ifs.read((char *)&has_right, sizeof(has_right));
            if (has_right) {
                right = make_shared<OperatorTensor<S, FL>>();
                right->load_data(ifs, true);
            } else
                right = nullptr;
            right_op_infos = Partition::load_op_infos(ifs);
        }
    }
    void load_data(bool left_part, const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("Partition:load_data on '" + filename +
                                "' failed.");
        load_data(ifs, left_part);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("Partition:load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    void save_data(ostream &ofs, bool left_part) const {
        if (left_part) {
            uint8_t has_left = left != nullptr;
            ofs.write((char *)&has_left, sizeof(has_left));
            if (has_left)
                left->save_data(ofs, true);
            Partition::save_op_infos(left_op_infos, ofs);
        } else {
            uint8_t has_right = right != nullptr;
            ofs.write((char *)&has_right, sizeof(has_right));
            if (has_right)
                right->save_data(ofs, true);
            Partition::save_op_infos(right_op_infos, ofs);
        }
    }
    void save_data(bool left_part, const string &filename) const {
        if (!frame_<FP>()->partition_can_write)
            return;
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("Partition:save_data on '" + filename +
                                "' failed.");
        save_data(ofs, left_part);
        if (!ofs.good())
            throw runtime_error("Partition:save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    static vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
    load_op_infos(istream &ifs) {
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> op_infos(sz);
        for (int i = 0; i < sz; i++) {
            ifs.read((char *)&op_infos[i].first, sizeof(op_infos[i].first));
            op_infos[i].second = make_shared<SparseMatrixInfo<S>>(ialloc);
            op_infos[i].second->load_data(ifs, true);
        }
        return op_infos;
    }
    static void save_op_infos(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &op_infos,
        ostream &ofs) {
        int sz = (int)op_infos.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (auto &op_info : op_infos) {
            ofs.write((char *)&op_info.first, sizeof(op_info.first));
            op_info.second->save_data(ofs, true);
        }
    }
    static shared_ptr<SparseMatrixInfo<S>> find_op_info(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &op_infos, S q) {
        auto p = lower_bound(op_infos.begin(), op_infos.end(), q,
                             SparseMatrixInfo<S>::cmp_op_info);
        if (p == op_infos.end() || p->first != q)
            return nullptr;
        else
            return p->second;
    }
    // Build the shell of contracted left block operators
    static shared_ptr<OperatorTensor<S, FL>> build_left(
        const vector<shared_ptr<Symbolic<S>>> &mats,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        bool csr = false, shared_ptr<Symbolic<S>> stacked_mat = nullptr) {
        shared_ptr<OperatorTensor<S, FL>> opt =
            make_shared<OperatorTensor<S, FL>>();
        assert(mats[0] != nullptr);
        assert(mats[0]->get_type() == SymTypes::RVec);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        if (stacked_mat == nullptr)
            opt->lmat = make_shared<SymbolicRowVector<S>>(
                *dynamic_pointer_cast<SymbolicRowVector<S>>(mats[0]));
        else {
            opt->lmat =
                make_shared<SymbolicRowVector<S>>(mats[0]->n * stacked_mat->n);
            for (size_t j = 0; j < stacked_mat->data.size(); j++) {
                shared_ptr<OpElement<S, FL>> opb =
                    stacked_mat->data[j]->get_type() != OpTypes::Zero
                        ? dynamic_pointer_cast<OpElement<S, FL>>(
                              abs_value(stacked_mat->data[j]))
                        : nullptr;
                for (size_t i = 0; i < mats[0]->data.size(); i++) {
                    shared_ptr<OpElement<S, FL>> opa =
                        mats[0]->data[i]->get_type() != OpTypes::Zero
                            ? dynamic_pointer_cast<OpElement<S, FL>>(
                                  abs_value(mats[0]->data[i]))
                            : nullptr;
                    opt->lmat->data[j * mats[0]->data.size() + i] =
                        opa == nullptr || opb == nullptr
                            ? zero
                            : make_shared<OpProduct<S, FL>>(opa, opb, (FL)1.0);
                }
            }
        }
        for (auto &mat : mats)
            if (stacked_mat == nullptr) {
                for (size_t i = 0; i < mat->data.size(); i++)
                    if (mat->data[i]->get_type() != OpTypes::Zero) {
                        shared_ptr<OpExpr<S>> op = abs_value(mat->data[i]);
                        opt->ops[op] =
                            csr ? make_shared<CSRSparseMatrix<S, FL>>()
                                : make_shared<SparseMatrix<S, FL>>();
                    }
            } else {
                for (size_t j = 0; j < stacked_mat->data.size(); j++)
                    if (stacked_mat->data[j]->get_type() != OpTypes::Zero) {
                        shared_ptr<OpElement<S, FL>> opb =
                            dynamic_pointer_cast<OpElement<S, FL>>(
                                abs_value(stacked_mat->data[j]));
                        for (size_t i = 0; i < mat->data.size(); i++)
                            if (mat->data[i]->get_type() != OpTypes::Zero) {
                                shared_ptr<OpElement<S, FL>> opa =
                                    dynamic_pointer_cast<OpElement<S, FL>>(
                                        abs_value(mat->data[i]));
                                shared_ptr<OpExpr<S>> op =
                                    make_shared<OpProduct<S, FL>>(opa, opb,
                                                                  (FL)1.0);
                                opt->ops[op] =
                                    csr ? make_shared<CSRSparseMatrix<S, FL>>()
                                        : make_shared<SparseMatrix<S, FL>>();
                            }
                    }
            }
        if (stacked_mat == nullptr) {
            for (auto &p : opt->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                p.second->info = find_op_info(left_op_infos, op->q_label);
            }
        } else {
            for (auto &p : opt->ops) {
                shared_ptr<OpProduct<S, FL>> op =
                    dynamic_pointer_cast<OpProduct<S, FL>>(p.first);
                p.second->info = find_op_info(
                    left_op_infos, (op->a->q_label + op->b->q_label)[0]);
                assert(p.second->info != nullptr);
            }
        }
        return opt;
    }
    // Build the shell of contracted right block operators
    static shared_ptr<OperatorTensor<S, FL>> build_right(
        const vector<shared_ptr<Symbolic<S>>> &mats,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        bool csr = false, shared_ptr<Symbolic<S>> stacked_mat = nullptr) {
        shared_ptr<OperatorTensor<S, FL>> opt =
            make_shared<OperatorTensor<S, FL>>();
        assert(mats[0] != nullptr);
        assert(mats[0]->get_type() == SymTypes::CVec);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        if (stacked_mat == nullptr)
            opt->rmat = make_shared<SymbolicColumnVector<S>>(
                *dynamic_pointer_cast<SymbolicColumnVector<S>>(mats[0]));
        else {
            opt->rmat = make_shared<SymbolicColumnVector<S>>(mats[0]->m *
                                                             stacked_mat->m);
            for (size_t j = 0; j < stacked_mat->data.size(); j++) {
                shared_ptr<OpElement<S, FL>> opb =
                    stacked_mat->data[j]->get_type() != OpTypes::Zero
                        ? dynamic_pointer_cast<OpElement<S, FL>>(
                              abs_value(stacked_mat->data[j]))
                        : nullptr;
                for (size_t i = 0; i < mats[0]->data.size(); i++) {
                    shared_ptr<OpElement<S, FL>> opa =
                        mats[0]->data[i]->get_type() != OpTypes::Zero
                            ? dynamic_pointer_cast<OpElement<S, FL>>(
                                  abs_value(mats[0]->data[i]))
                            : nullptr;
                    opt->rmat->data[j * mats[0]->data.size() + i] =
                        opa == nullptr || opb == nullptr
                            ? zero
                            : make_shared<OpProduct<S, FL>>(opa, opb, (FL)1.0);
                }
            }
        }
        for (auto &mat : mats)
            if (stacked_mat == nullptr) {
                for (size_t i = 0; i < mat->data.size(); i++)
                    if (mat->data[i]->get_type() != OpTypes::Zero) {
                        shared_ptr<OpExpr<S>> op = abs_value(mat->data[i]);
                        opt->ops[op] =
                            csr ? make_shared<CSRSparseMatrix<S, FL>>()
                                : make_shared<SparseMatrix<S, FL>>();
                    }
            } else {
                for (size_t j = 0; j < stacked_mat->data.size(); j++)
                    if (stacked_mat->data[j]->get_type() != OpTypes::Zero) {
                        shared_ptr<OpElement<S, FL>> opb =
                            dynamic_pointer_cast<OpElement<S, FL>>(
                                abs_value(stacked_mat->data[j]));
                        for (size_t i = 0; i < mat->data.size(); i++)
                            if (mat->data[i]->get_type() != OpTypes::Zero) {
                                shared_ptr<OpElement<S, FL>> opa =
                                    dynamic_pointer_cast<OpElement<S, FL>>(
                                        abs_value(mat->data[i]));
                                shared_ptr<OpExpr<S>> op =
                                    make_shared<OpProduct<S, FL>>(opa, opb,
                                                                  (FL)1.0);
                                opt->ops[op] =
                                    csr ? make_shared<CSRSparseMatrix<S, FL>>()
                                        : make_shared<SparseMatrix<S, FL>>();
                            }
                    }
            }
        if (stacked_mat == nullptr) {
            for (auto &p : opt->ops) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                p.second->info = find_op_info(right_op_infos, op->q_label);
            }
        } else {
            for (auto &p : opt->ops) {
                shared_ptr<OpProduct<S, FL>> op =
                    dynamic_pointer_cast<OpProduct<S, FL>>(p.first);
                p.second->info = find_op_info(
                    right_op_infos, (op->a->q_label + op->b->q_label)[0]);
                assert(p.second->info != nullptr);
            }
        }
        return opt;
    }
    // Get all possible delta quantum numbers from the symbolic matrix of
    // operators
    static vector<S>
    get_uniq_labels(const vector<shared_ptr<Symbolic<S>>> &mats) {
        vector<S> sl;
        for (auto &mat : mats) {
            assert(mat != nullptr);
            assert(mat->get_type() == SymTypes::RVec ||
                   mat->get_type() == SymTypes::CVec);
            sl.reserve(sl.size() + mat->data.size());
            for (size_t i = 0; i < mat->data.size(); i++) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(mat->data[i]);
                sl.push_back(op->q_label);
            }
        }
        sort(sl.begin(), sl.end());
        sl.resize(distance(sl.begin(), unique(sl.begin(), sl.end())));
        return sl;
    }
    // Get all possible combination of delta quantum numbers
    // from the matrix of symbolic expressions of operators
    // op_left_vacuum should be true vacuum when this is for right contract
    static vector<vector<pair<uint8_t, S>>>
    get_uniq_sub_labels(const shared_ptr<Symbolic<S>> &exprs,
                        const shared_ptr<Symbolic<S>> &mat, const vector<S> &sl,
                        S op_left_vacuum, bool partial = false,
                        bool left_only = true, bool uniq_sorted = true) {
        vector<vector<pair<uint8_t, S>>> subsl(sl.size());
        if (exprs == nullptr)
            return subsl;
        assert(mat->data.size() == exprs->data.size());
        for (size_t i = 0; i < mat->data.size(); i++) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(mat->data[i]);
            S l = op->q_label;
            size_t idx = lower_bound(sl.begin(), sl.end(), l) - sl.begin();
            assert(idx != sl.size());
            shared_ptr<OpExpr<S>> opx =
                exprs->data[i]->get_type() == OpTypes::ExprRef
                    ? dynamic_pointer_cast<OpExprRef<S>>(exprs->data[i])->op
                    : exprs->data[i];
            switch (opx->get_type()) {
            case OpTypes::Zero:
                break;
            case OpTypes::Elem: {
                // only needed for singlet embedding
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(opx);
                S p = l.combine(op_left_vacuum, -op->q_label);
                assert(p != S(S::invalid));
                subsl[idx].push_back(make_pair(0, p));
            } break;
            case OpTypes::Prod: {
                shared_ptr<OpProduct<S, FL>> op =
                    dynamic_pointer_cast<OpProduct<S, FL>>(opx);
                assert(op->b != nullptr);
                S bra = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                S ket = (op->conj & 2) ? op->b->q_label : -op->b->q_label;
                if (!partial) {
                    S p = l.combine(bra, ket);
                    assert(p != S(S::invalid));
                    subsl[idx].push_back(make_pair(op->conj, p));
                } else if (left_only)
                    subsl[idx].push_back(make_pair(op->conj & 1, bra));
                else
                    subsl[idx].push_back(make_pair(!!(op->conj & 2), -ket));
            } break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S, FL>> sop =
                    dynamic_pointer_cast<OpSum<S, FL>>(opx);
                for (auto &op : sop->strings) {
                    S bra, ket;
                    if (op->a != nullptr && op->b != nullptr) {
                        bra = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                        ket = (op->conj & 2) ? op->b->q_label : -op->b->q_label;
                    } else if (op->b == nullptr &&
                               op->get_type() == OpTypes::Prod) {
                        // only needed for singlet embedding
                        assert(op->conj == 0);
                        bra = op_left_vacuum, ket = -op->get_op()->q_label;
                    } else {
                        assert(op->get_type() == OpTypes::SumProd);
                        shared_ptr<OpSumProd<S, FL>> spop =
                            dynamic_pointer_cast<OpSumProd<S, FL>>(op);
                        assert(spop->ops.size() != 0);
                        if (spop->a != nullptr) {
                            bra = (op->conj & 1) ? -op->a->q_label
                                                 : op->a->q_label;
                            ket = (op->conj & 2) ? spop->ops[0]->q_label
                                                 : -spop->ops[0]->q_label;
                        } else if (spop->b != nullptr) {
                            bra = (op->conj & 1) ? -spop->ops[0]->q_label
                                                 : spop->ops[0]->q_label;
                            ket = (op->conj & 2) ? op->b->q_label
                                                 : -op->b->q_label;
                        } else
                            assert(false);
                    }
                    if (!partial) {
                        S p = l.combine(bra, ket);
                        // here possible error can be due to non-zero (small)
                        // integral element violating point group symmetry
                        assert(p != S(S::invalid));
                        subsl[idx].push_back(make_pair(op->conj, p));
                    } else if (left_only)
                        subsl[idx].push_back(make_pair(op->conj & 1, bra));
                    else
                        subsl[idx].push_back(make_pair(!!(op->conj & 2), -ket));
                }
            } break;
            default:
                assert(false);
            }
            if (uniq_sorted) {
                // needed for iop x iop in me delayed contraction with MPI
                S p0 = l.combine(op_left_vacuum, (l - l)[0]);
                if (p0 != S(S::invalid))
                    subsl[idx].push_back(make_pair((uint8_t)0, p0));
            }
        }
        if (uniq_sorted) {
            for (size_t i = 0; i < subsl.size(); i++) {
                sort(subsl[i].begin(), subsl[i].end());
                subsl[i].resize(
                    distance(subsl[i].begin(),
                             unique(subsl[i].begin(), subsl[i].end())));
            }
        }
        return subsl;
    }
    static vector<S> get_stacked_uniq_labels(const vector<S> &ra,
                                             const vector<S> &rb) {
        vector<S> r;
        for (const auto &xa : ra)
            for (const auto &xb : rb) {
                const S m = xa + xb;
                for (int i = 0; i < m.count(); i++)
                    r.push_back(m[i]);
            }
        sort(r.begin(), r.end());
        r.resize(distance(r.begin(), unique(r.begin(), r.end())));
        return r;
    }
    static vector<vector<pair<uint8_t, S>>> get_stacked_uniq_sub_labels(
        const vector<S> &sla, const vector<S> &slb, const vector<S> &sl,
        const vector<vector<pair<uint8_t, S>>> &subsla,
        const vector<vector<pair<uint8_t, S>>> &subslb, bool partial = false,
        bool left_only = true, bool uniq_sorted = true) {
        vector<vector<pair<uint8_t, S>>> subsl(sl.size());
        assert(subsla.size() == sla.size());
        assert(subslb.size() == slb.size());
        for (size_t ib = 0; ib < subslb.size(); ib++)
            for (size_t ia = 0; ia < subsla.size(); ia++) {
                S la = sla[ia], lb = slb[ib];
                S l = (la + lb)[0];
                size_t idx = lower_bound(sl.begin(), sl.end(), l) - sl.begin();
                assert(idx != sl.size());
                for (const auto &ma : subsla[ia])
                    for (const auto &mb : subslb[ib])
                        if (!partial) {
                            S bra_a = ma.second.get_bra(la),
                              ket_a = ma.second.get_ket();
                            S bra_b = mb.second.get_bra(lb),
                              ket_b = mb.second.get_ket();
                            S bra = (bra_a + bra_b)[0],
                              ket = (ket_a + ket_b)[0];
                            S p = l.combine(bra, ket);
                            assert(p != S(S::invalid));
                            subsl[idx].push_back(make_pair(ma.first, p));
                        } else if (left_only) {
                            S bra_a = ma.second, bra_b = mb.second;
                            S bra = (bra_a + bra_b)[0];
                            subsl[idx].push_back(make_pair(ma.first, bra));
                        } else {
                            S mket_a = ma.second, mket_b = mb.second;
                            S mket = (mket_a + mket_b)[0];
                            subsl[idx].push_back(make_pair(ma.first, mket));
                        }
            }
        if (uniq_sorted) {
            for (size_t i = 0; i < subsl.size(); i++) {
                sort(subsl[i].begin(), subsl[i].end());
                subsl[i].resize(
                    distance(subsl[i].begin(),
                             unique(subsl[i].begin(), subsl[i].end())));
            }
        }
        return subsl;
    }
    static void deallocate_op_infos_notrunc(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &op_infos_notrunc) {
        for (int i = (int)op_infos_notrunc.size() - 1; i >= 0; i--) {
            op_infos_notrunc[i].second->cinfo->deallocate();
            op_infos_notrunc[i].second->deallocate();
        }
    }
    static void copy_op_infos(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &from_op_infos,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &to_op_infos) {
        assert(to_op_infos.size() == 0);
        to_op_infos.reserve(from_op_infos.size());
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        for (size_t i = 0; i < from_op_infos.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> info =
                make_shared<SparseMatrixInfo<S>>(
                    from_op_infos[i].second->deep_copy(i_alloc));
            to_op_infos.push_back(make_pair(from_op_infos[i].first, info));
        }
    }
    static map<S, shared_ptr<SparseMatrixInfo<S>>> get_stacked_op_info(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &infoa,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &infob,
        shared_ptr<StateInfo<S>> basis) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        map<S, shared_ptr<SparseMatrixInfo<S>>> info;
        for (const auto &xb : infob)
            for (const auto &xa : infoa) {
                S q = xa.first + xb.first;
                for (int iq = 0; iq < q.count(); iq++)
                    info[q[iq]] = nullptr;
            }
        for (auto &p : info) {
            p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
            p.second->initialize(*basis, *basis, p.first, p.first.is_fermion());
        }
        return info;
    }
    // Generate SparseMatrixInfo for contracted left block operators
    // after renormalization (or rotation)
    static void init_left_op_infos(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos) {
        frame_<FP>()->activate(0);
        bra_info->load_left_dims(m + 1);
        StateInfo<S> ibra = *bra_info->left_dims[m + 1], iket = ibra;
        if (bra_info != ket_info) {
            ket_info->load_left_dims(m + 1);
            iket = *ket_info->left_dims[m + 1];
        }
        frame_<FP>()->activate(1);
        assert(left_op_infos.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> lop =
                make_shared<SparseMatrixInfo<S>>();
            left_op_infos.push_back(make_pair(sl[i], lop));
            lop->initialize(ibra, iket, sl[i], sl[i].is_fermion());
        }
        frame_<FP>()->activate(0);
        if (bra_info != ket_info)
            iket.deallocate();
        ibra.deallocate();
    }
    // Generate SparseMatrixInfo for contracted left block operators
    // before renormalization (or rotation)
    static void init_left_op_infos_notrunc(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        const vector<vector<pair<uint8_t, S>>> &subsl,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &prev_left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &site_op_infos,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos_notrunc,
        const shared_ptr<CG<S>> &cg) {
        frame_<FP>()->activate(1);
        bra_info->load_left_dims(m);
        StateInfo<S> ibra_prev = *bra_info->left_dims[m], iket_prev = ibra_prev;
        StateInfo<S> ibra_notrunc = StateInfo<S>::tensor_product(
                         ibra_prev, *bra_info->basis[m],
                         *bra_info->left_dims_fci[m + 1]),
                     iket_notrunc = ibra_notrunc;
        shared_ptr<typename StateInfo<S>::ConnectionInfo>
            ibra_cinfo = StateInfo<S>::get_connection_info(
                ibra_prev, *bra_info->basis[m], ibra_notrunc),
            iket_cinfo = ibra_cinfo;
        if (bra_info != ket_info) {
            ket_info->load_left_dims(m);
            iket_prev = *ket_info->left_dims[m];
            iket_notrunc =
                StateInfo<S>::tensor_product(iket_prev, *ket_info->basis[m],
                                             *ket_info->left_dims_fci[m + 1]);
            iket_cinfo = StateInfo<S>::get_connection_info(
                iket_prev, *ket_info->basis[m], iket_notrunc);
        }
        frame_<FP>()->activate(0);
        assert(left_op_infos_notrunc.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> lop_notrunc =
                make_shared<SparseMatrixInfo<S>>();
            left_op_infos_notrunc.push_back(make_pair(sl[i], lop_notrunc));
            lop_notrunc->initialize(ibra_notrunc, iket_notrunc, sl[i],
                                    sl[i].is_fermion());
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            cinfo->initialize_tp(
                sl[i], subsl[i], ibra_notrunc, iket_notrunc, ibra_prev,
                *bra_info->basis[m], iket_prev, *ket_info->basis[m], ibra_cinfo,
                iket_cinfo, prev_left_op_infos, site_op_infos, lop_notrunc, cg);
            lop_notrunc->cinfo = cinfo;
        }
        frame_<FP>()->activate(1);
        if (bra_info != ket_info) {
            iket_notrunc.deallocate();
            iket_prev.deallocate();
        }
        ibra_notrunc.deallocate();
        ibra_prev.deallocate();
    }
    // Generate SparseMatrixInfo for contracted right block operators
    // after renormalization (or rotation)
    static void init_right_op_infos(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos) {
        frame_<FP>()->activate(0);
        bra_info->load_right_dims(m);
        StateInfo<S> ibra = *bra_info->right_dims[m], iket = ibra;
        if (bra_info != ket_info) {
            ket_info->load_right_dims(m);
            iket = *ket_info->right_dims[m];
        }
        frame_<FP>()->activate(1);
        assert(right_op_infos.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> rop =
                make_shared<SparseMatrixInfo<S>>();
            right_op_infos.push_back(make_pair(sl[i], rop));
            rop->initialize(ibra, iket, sl[i], sl[i].is_fermion());
        }
        frame_<FP>()->activate(0);
        if (bra_info != ket_info)
            iket.deallocate();
        ibra.deallocate();
    }
    // Generate SparseMatrixInfo for contracted right block operators
    // before renormalization (or rotation)
    static void init_right_op_infos_notrunc(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        const vector<vector<pair<uint8_t, S>>> &subsl,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &prev_right_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &site_op_infos,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &right_op_infos_notrunc,
        const shared_ptr<CG<S>> &cg) {
        frame_<FP>()->activate(1);
        bra_info->load_right_dims(m + 1);
        StateInfo<S> ibra_prev = *bra_info->right_dims[m + 1],
                     iket_prev = ibra_prev;
        StateInfo<S> ibra_notrunc = StateInfo<S>::tensor_product(
                         *bra_info->basis[m], ibra_prev,
                         *bra_info->right_dims_fci[m]),
                     iket_notrunc = ibra_notrunc;
        shared_ptr<typename StateInfo<S>::ConnectionInfo>
            ibra_cinfo = StateInfo<S>::get_connection_info(
                *bra_info->basis[m], ibra_prev, ibra_notrunc),
            iket_cinfo = ibra_cinfo;
        if (bra_info != ket_info) {
            ket_info->load_right_dims(m + 1);
            iket_prev = *ket_info->right_dims[m + 1];
            iket_notrunc = StateInfo<S>::tensor_product(
                *ket_info->basis[m], iket_prev, *ket_info->right_dims_fci[m]);
            iket_cinfo = StateInfo<S>::get_connection_info(
                *ket_info->basis[m], iket_prev, iket_notrunc);
        }
        frame_<FP>()->activate(0);
        assert(right_op_infos_notrunc.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> rop_notrunc =
                make_shared<SparseMatrixInfo<S>>();
            right_op_infos_notrunc.push_back(make_pair(sl[i], rop_notrunc));
            rop_notrunc->initialize(ibra_notrunc, iket_notrunc, sl[i],
                                    sl[i].is_fermion());
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            cinfo->initialize_tp(sl[i], subsl[i], ibra_notrunc, iket_notrunc,
                                 *bra_info->basis[m], ibra_prev,
                                 *ket_info->basis[m], iket_prev, ibra_cinfo,
                                 iket_cinfo, site_op_infos, prev_right_op_infos,
                                 rop_notrunc, cg);
            rop_notrunc->cinfo = cinfo;
        }
        frame_<FP>()->activate(1);
        if (bra_info != ket_info) {
            iket_notrunc.deallocate();
            iket_prev.deallocate();
        }
        ibra_notrunc.deallocate();
        ibra_prev.deallocate();
    }
};

} // namespace block2
