
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

/** Operations for operator tensors (renormalized operators, etc.)
 * with internal data stored in disk file.
 * Note: this is inefficient and may not be compatitble with many other
 * features.
 */

#pragma once

#include "archived_sparse_matrix.hpp"
#include "tensor_functions.hpp"

using namespace std;

namespace block2 {

/** Operations for operator tensors with internal data stored in disk file.
 * @tparam S Quantum label type.
 * @tparam FL float point type.
 */
template <typename S, typename FL>
struct ArchivedTensorFunctions : TensorFunctions<S, FL> {
    using TensorFunctions<S, FL>::opf;
    string filename = ""; //!< The name of the associated disk file.
    mutable int64_t offset =
        0; //!< Byte offset in the file (where to read/write the content).
    /** Constructor.
     * @param opf Sparse matrix algebra driver.
     */
    ArchivedTensorFunctions(const shared_ptr<OperatorFunctions<S, FL>> &opf)
        : TensorFunctions<S, FL>(opf) {}
    /** Get the type of this driver for tensor functions.
     * @return Type of this driver for tensor functions.
     */
    TensorFunctionsTypes get_type() const override {
        return TensorFunctionsTypes::Archived;
    }
    /** Save the content of an operator tensor into disk,
     * transforming its internal representation to sparse matrices with internal
     * data stored in disk file, and deallocating its memory data.
     * @param a Operator tensor with ordinary memory storage.
     */
    void archive_tensor(const shared_ptr<OperatorTensor<S, FL>> &a) const {
        map<FL *, vector<shared_ptr<SparseMatrix<S, FL>>>> mp;
        for (auto &op : a->ops) {
            shared_ptr<SparseMatrix<S, FL>> mat = op.second;
            shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                make_shared<ArchivedSparseMatrix<S, FL>>(filename, offset);
            arc->save_archive(mat);
            mp[op.second->data].push_back(op.second);
            op.second = arc;
            offset += arc->total_memory;
        }
        for (auto it = mp.crbegin(); it != mp.crend(); it++)
            for (const auto &t : it->second)
                t->deallocate();
    }
    /** Left assignment (copy) operation: c = a. This is the edge case for the
     * left blocking step. Left assignment means that the operator tensor is a
     * row vector of symbols.
     * @param a Operator a (input).
     * @param c Operator c (output).
     */
    void left_assign(const shared_ptr<OperatorTensor<S, FL>> &a,
                     shared_ptr<OperatorTensor<S, FL>> &c) const override {
        assert(a->lmat != nullptr);
        assert(a->lmat->get_type() == SymTypes::RVec);
        assert(c->lmat != nullptr);
        assert(c->lmat->get_type() == SymTypes::RVec);
        assert(a->lmat->data.size() == c->lmat->data.size());
        for (size_t i = 0; i < a->lmat->data.size(); i++) {
            if (a->lmat->data[i]->get_type() == OpTypes::Zero)
                c->lmat->data[i] = a->lmat->data[i];
            else {
                assert(a->lmat->data[i] == c->lmat->data[i]);
                shared_ptr<OpExpr<S>> pa = abs_value(a->lmat->data[i]),
                                      pc = abs_value(c->lmat->data[i]);
                shared_ptr<SparseMatrix<S, FL>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        a->ops[pa])
                        ->load_archive();
                shared_ptr<SparseMatrix<S, FL>> matc = c->ops.at(pc);
                matc->allocate(matc->info);
                if (matc->info->n == mata->info->n)
                    matc->copy_data_from(mata, true);
                else
                    matc->selective_copy_from(mata, true);
                matc->factor = mata->factor;
                shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                    make_shared<ArchivedSparseMatrix<S, FL>>(filename, offset);
                arc->save_archive(matc);
                matc->deallocate();
                c->ops.at(pc) = arc;
                offset += arc->total_memory;
                mata->deallocate();
            }
        }
    }
    /** Right assignment (copy) operation: c = a. This is the edge case for the
     * right blocking step. Right assignment means that the operator tensor is a
     * column vector of symbols.
     * @param a Operator a (input).
     * @param c Operator c (output).
     */
    void right_assign(const shared_ptr<OperatorTensor<S, FL>> &a,
                      shared_ptr<OperatorTensor<S, FL>> &c) const override {
        assert(a->rmat != nullptr);
        assert(a->rmat->get_type() == SymTypes::CVec);
        assert(c->rmat != nullptr);
        assert(c->rmat->get_type() == SymTypes::CVec);
        assert(a->rmat->data.size() == c->rmat->data.size());
        for (size_t i = 0; i < a->rmat->data.size(); i++) {
            if (a->rmat->data[i]->get_type() == OpTypes::Zero)
                c->rmat->data[i] = a->rmat->data[i];
            else {
                assert(a->rmat->data[i] == c->rmat->data[i]);
                shared_ptr<OpExpr<S>> pa = abs_value(a->rmat->data[i]),
                                      pc = abs_value(c->rmat->data[i]);
                shared_ptr<SparseMatrix<S, FL>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        a->ops[pa])
                        ->load_archive();
                shared_ptr<SparseMatrix<S, FL>> matc = c->ops.at(pc);
                matc->allocate(matc->info);
                if (matc->info->n == mata->info->n)
                    matc->copy_data_from(mata, true);
                else
                    matc->selective_copy_from(mata, true);
                matc->factor = mata->factor;
                shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                    make_shared<ArchivedSparseMatrix<S, FL>>(filename, offset);
                arc->save_archive(matc);
                matc->deallocate();
                c->ops.at(pc) = arc;
                offset += arc->total_memory;
                mata->deallocate();
            }
        }
    }
    /**
     * Partial tensor product multiplication operation: vmat = expr[L part | R
     * part] x cmat. This is used only for perturbative noise, where only left
     * or right block part of the Hamiltonian expression is multiplied by the
     * wavefunction cmat, to get a perurbed wavefunction vmat.
     * @param expr Symbolic expression in form of sum of tensor products.
     * @param lopt Symbol lookup table for left operands in the tensor products.
     * @param ropt Symbol lookup table for right operands in the tensor
     * products.
     * @param trace_right If true, the left operands in the tensor products are
     * used. The right operands are treated as identity. Otherwise, the right
     * operands in the tensor products are used.
     * @param cmat Input "vector" operand (wavefunction).
     * @param psubsl Vector of transpose pattern and delta quantum of the
     * "matrix" operand (in the same order as cinfos).
     * @param cinfos Vector of sparse matrix connection info (in the same order
     * as cinfos).
     * @param vdqs Vector of quantum number of each vmat (for lookup).
     * @param vmats Vector of output "vectors" (perurbed wavefunctions).
     * @param vidx If -1, there is only one perurbed wavefunction for each
     * target quantum number (used in NoiseTypes::ReducedPerturbative).
     * Otherwise, one vmat is created for each tensor product (used in
     * NoiseTypes::Perturbative), and vidx is used as an incremental index in
     * vmats.
     * @param tvidx If -1, vmats is copied to every thread, which is the high
     * memory mode but may be more load-balanced, the multi-thread
     * parallelization is over different terms in expr for this case. If -2,
     * every thread works on a single vmat, which is the low memory mode (used
     * in NoiseTypes:: NoiseTypes::LowMem), the multi-thread parallelization is
     * over different vmats for this case. Otherwise, if >= 0, only the
     * specified vmat is handled, which is used only internally.
     * @param do_reduce If true, the output vmats are accumulated to the root
     * processor in the distributed parallel case.
     */
    void tensor_product_partial_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S, FL>> &lopt,
        const shared_ptr<OperatorTensor<S, FL>> &ropt, bool trace_right,
        const shared_ptr<SparseMatrix<S, FL>> &cmat,
        const vector<pair<uint8_t, S>> &psubsl,
        const vector<
            vector<shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>>
            &cinfos,
        const vector<S> &vdqs,
        const shared_ptr<SparseMatrixGroup<S, FL>> &vmats, int &vidx, int tvidx,
        bool do_reduce) const override {
        const shared_ptr<OpElement<S, FL>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), S());
        if ((!trace_right && lopt->ops.count(i_op) == 0) ||
            (trace_right && ropt->ops.count(i_op) == 0))
            return;
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->b != nullptr);
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> old_cinfo =
                cmat->info->cinfo;
            if (trace_right) {
                assert(lopt->ops.count(op->a) != 0 &&
                       ropt->ops.count(i_op) != 0);
                shared_ptr<SparseMatrix<S, FL>> lmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        lopt->ops.at(op->a))
                        ->load_archive();
                shared_ptr<SparseMatrix<S, FL>> rmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        ropt->ops.at(i_op))
                        ->load_archive();
                S opdq = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                S pks = cmat->info->delta_quantum + opdq;
                int ij = (int)(lower_bound(
                                   psubsl.begin(), psubsl.end(),
                                   make_pair((uint8_t)(op->conj & 1), opdq)) -
                               psubsl.begin());
                for (int k = 0; k < pks.count(); k++) {
                    S vdq = pks[k];
                    int iv = (int)(lower_bound(vdqs.begin(), vdqs.end(), vdq) -
                                   vdqs.begin());
                    shared_ptr<SparseMatrix<S, FL>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    opf->tensor_product_multiply(op->conj & 1, lmat, rmat, cmat,
                                                 vmat, opdq, op->factor,
                                                 TraceTypes::Right);
                }
                rmat->deallocate();
                lmat->deallocate();
            } else {
                assert(lopt->ops.count(i_op) != 0 &&
                       ropt->ops.count(op->b) != 0);
                shared_ptr<SparseMatrix<S, FL>> lmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        lopt->ops.at(i_op))
                        ->load_archive();
                shared_ptr<SparseMatrix<S, FL>> rmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        ropt->ops.at(op->b))
                        ->load_archive();
                S opdq = (op->conj & 2) ? -op->b->q_label : op->b->q_label;
                S pks = cmat->info->delta_quantum + opdq;
                int ij =
                    (int)(lower_bound(
                              psubsl.begin(), psubsl.end(),
                              make_pair((uint8_t)(!!(op->conj & 2)), opdq)) -
                          psubsl.begin());
                for (int k = 0; k < pks.count(); k++) {
                    S vdq = pks[k];
                    int iv = (int)(lower_bound(vdqs.begin(), vdqs.end(), vdq) -
                                   vdqs.begin());
                    shared_ptr<SparseMatrix<S, FL>> vmat =
                        vidx == -1 ? (*vmats)[iv] : (*vmats)[vidx++];
                    cmat->info->cinfo = cinfos[ij][k];
                    opf->tensor_product_multiply(op->conj & 2, lmat, rmat, cmat,
                                                 vmat, opdq, op->factor,
                                                 TraceTypes::Left);
                }
                rmat->deallocate();
                lmat->deallocate();
            }
            cmat->info->cinfo = old_cinfo;
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            for (auto &x : op->strings)
                tensor_product_partial_multiply(x, lopt, ropt, trace_right,
                                                cmat, psubsl, cinfos, vdqs,
                                                vmats, vidx, tvidx, false);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    /** Tensor product multiplication operation (multi-root case): vmats = expr
     * x cmats. Both cmats and vmats are wavefunctions with multi-target
     * components.
     * @param expr Symbolic expression in form of sum of tensor products.
     * @param lopt Symbol lookup table for left operands in the tensor products.
     * @param ropt Symbol lookup table for right operands in the tensor
     * products.
     * @param cmats Input "vector" operand (multi-target wavefunction).
     * @param vmats Output "vector" result (multi-target wavefunction).
     * @param cinfos Lookup table of sparse matrix connection info, where the
     * key is the combined quantum number of the vmat and cmat.
     * @param opdq The delta quantum number of expr.
     * @param factor Sacling factor applied to the results.
     * @param all_reduce If true, the output result is accumulated and
     * broadcast to all processors.
     */
    void tensor_product_multi_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const shared_ptr<OperatorTensor<S, FL>> &lopt,
        const shared_ptr<OperatorTensor<S, FL>> &ropt,
        const shared_ptr<SparseMatrixGroup<S, FL>> &cmats,
        const shared_ptr<SparseMatrixGroup<S, FL>> &vmats,
        const unordered_map<
            S, shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>
            &cinfos,
        S opdq, FL factor, bool all_reduce) const override {
        unordered_map<S, int> vdqs;
        vdqs.reserve(vmats->n);
        for (int iv = 0; iv < vmats->n; iv++)
            vdqs[vmats->infos[iv]->delta_quantum] = iv;
        for (int i = 0; i < cmats->n; i++) {
            shared_ptr<SparseMatrix<S, FL>> pcmat = (*cmats)[i];
            pcmat->factor = factor;
            shared_ptr<SparseMatrixInfo<S>> pcmat_info =
                make_shared<SparseMatrixInfo<S>>(*pcmat->info);
            pcmat->info = pcmat_info;
            S cdq = pcmat->info->delta_quantum;
            S vdq = opdq + cdq;
            for (int iv = 0; iv < vdq.count(); iv++)
                if (vdqs.count(vdq[iv])) {
                    pcmat->info->cinfo = cinfos.at(opdq.combine(vdq[iv], cdq));
                    tensor_product_multiply(expr, lopt, ropt, pcmat,
                                            (*vmats)[vdqs[vdq[iv]]], opdq,
                                            false);
                }
        }
    }
    /** Tensor product multiplication operation (single-root case): vmat = expr
     * x cmat.
     * @param expr Symbolic expression in form of sum of tensor products.
     * @param lopt Symbol lookup table for left operands in the tensor products.
     * @param ropt Symbol lookup table for right operands in the tensor
     * products.
     * @param cmat Input "vector" operand (wavefunction), assuming the cinfo is
     * already attached.
     * @param vmat Output "vector" result (wavefunction).
     * @param opdq The delta quantum number of expr.
     * @param all_reduce If true, the output result is accumulated and broadcast
     * to all processors.
     */
    void tensor_product_multiply(const shared_ptr<OpExpr<S>> &expr,
                                 const shared_ptr<OperatorTensor<S, FL>> &lopt,
                                 const shared_ptr<OperatorTensor<S, FL>> &ropt,
                                 const shared_ptr<SparseMatrix<S, FL>> &cmat,
                                 const shared_ptr<SparseMatrix<S, FL>> &vmat,
                                 S opdq, bool all_reduce) const override {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->b != nullptr);
            assert(
                !(lopt->ops.count(op->a) == 0 || ropt->ops.count(op->b) == 0));
            shared_ptr<SparseMatrix<S, FL>> lmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                    lopt->ops.at(op->a))
                    ->load_archive();
            shared_ptr<SparseMatrix<S, FL>> rmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                    ropt->ops.at(op->b))
                    ->load_archive();
            opf->tensor_product_multiply(op->conj, lmat, rmat, cmat, vmat, opdq,
                                         op->factor);
            rmat->deallocate();
            lmat->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            for (auto &x : op->strings)
                tensor_product_multiply(x, lopt, ropt, cmat, vmat, opdq, false);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    /** Extraction of diagonal of a tensor product expression: mat = diag(expr).
     * @param expr Symbolic expression in form of sum of tensor products.
     * @param lopt Symbol lookup table for left operands in the tensor products.
     * @param ropt Symbol lookup table for right operands in the tensor
     * products.
     * @param mat Output "vector" result (diagonal part).
     * @param opdq The delta quantum number of expr.
     */
    void tensor_product_diagonal(const shared_ptr<OpExpr<S>> &expr,
                                 const shared_ptr<OperatorTensor<S, FL>> &lopt,
                                 const shared_ptr<OperatorTensor<S, FL>> &ropt,
                                 const shared_ptr<SparseMatrix<S, FL>> &mat,
                                 S opdq) const override {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->b != nullptr);
            assert(
                !(lopt->ops.count(op->a) == 0 || ropt->ops.count(op->b) == 0));
            shared_ptr<SparseMatrix<S, FL>> lmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                    lopt->ops.at(op->a))
                    ->load_archive();
            shared_ptr<SparseMatrix<S, FL>> rmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                    ropt->ops.at(op->b))
                    ->load_archive();
            opf->tensor_product_diagonal(op->conj, lmat, rmat, mat, opdq,
                                         op->factor);
            rmat->deallocate();
            lmat->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            for (auto &x : op->strings)
                tensor_product_diagonal(x, lopt, ropt, mat, opdq);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    /** Direct evaluation of a tensor product expression: mat = eval(expr).
     * @param expr Symbolic expression in form of sum of tensor products.
     * @param lop Symbol lookup table for left operands in the tensor products.
     * @param rop Symbol lookup table for right operands in the tensor products.
     * @param mat Output "vector" result (the sparse matrix value of the
     * expression).
     */
    void
    tensor_product(const shared_ptr<OpExpr<S>> &expr,
                   const unordered_map<shared_ptr<OpExpr<S>>,
                                       shared_ptr<SparseMatrix<S, FL>>> &lop,
                   const unordered_map<shared_ptr<OpExpr<S>>,
                                       shared_ptr<SparseMatrix<S, FL>>> &rop,
                   shared_ptr<SparseMatrix<S, FL>> &mat) const override {
        shared_ptr<ArchivedSparseMatrix<S, FL>> aromat = nullptr;
        shared_ptr<SparseMatrix<S, FL>> omat;
        if (mat->get_type() == SparseMatrixTypes::Archived) {
            aromat = dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(mat);
            omat = aromat->load_archive();
        } else
            omat = mat;
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpProduct<S, FL>> op =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            assert(op->b != nullptr);
            assert(lop.count(op->a) != 0 && rop.count(op->b) != 0);
            shared_ptr<SparseMatrix<S, FL>> lmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(lop.at(op->a))
                    ->load_archive();
            shared_ptr<SparseMatrix<S, FL>> rmat =
                dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(rop.at(op->b))
                    ->load_archive();
            opf->tensor_product(op->conj, lmat, rmat, omat, op->factor);
            rmat->deallocate();
            lmat->deallocate();
        } break;
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S, FL>> op =
                dynamic_pointer_cast<OpSumProd<S, FL>>(expr);
            assert((op->a == nullptr) ^ (op->b == nullptr));
            assert(op->ops.size() != 0);
            shared_ptr<SparseMatrix<S, FL>> tmp =
                make_shared<SparseMatrix<S, FL>>();
            if (op->c != nullptr && ((op->b == nullptr && rop.count(op->c)) ||
                                     (op->a == nullptr && lop.count(op->c)))) {
                if (op->b == nullptr && rop.count(op->c))
                    tmp = dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                              rop.at(op->c))
                              ->load_archive();
                else
                    tmp = dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                              lop.at(op->c))
                              ->load_archive();
            } else if (op->b == nullptr) {
                shared_ptr<OpExpr<S>> opb =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(op->a) != 0 && rop.count(opb) != 0);
                tmp->allocate(rop.at(opb)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    shared_ptr<SparseMatrix<S, FL>> rmat =
                        dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                            rop.at(
                                abs_value((shared_ptr<OpExpr<S>>)op->ops[i])))
                            ->load_archive();
                    opf->iadd(tmp, rmat, op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode == SeqTypes::Simple)
                        opf->seq->simple_perform();
                    rmat->deallocate();
                }
            } else {
                shared_ptr<OpExpr<S>> opa =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(opa) != 0 && rop.count(op->b) != 0);
                tmp->allocate(lop.at(opa)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    shared_ptr<SparseMatrix<S, FL>> lmat =
                        dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                            lop.at(
                                abs_value((shared_ptr<OpExpr<S>>)op->ops[i])))
                            ->load_archive();
                    opf->iadd(tmp, lmat, op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode == SeqTypes::Simple)
                        opf->seq->simple_perform();
                    lmat->deallocate();
                }
            }
            if (op->b == nullptr) {
                shared_ptr<SparseMatrix<S, FL>> lmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        lop.at(op->a))
                        ->load_archive();
                opf->tensor_product(op->conj, lmat, tmp, omat, op->factor);
                lmat->deallocate();
            } else {
                shared_ptr<SparseMatrix<S, FL>> rmat =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        rop.at(op->b))
                        ->load_archive();
                opf->tensor_product(op->conj, tmp, rmat, omat, op->factor);
                rmat->deallocate();
            }
            tmp->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S, FL>> op =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            for (auto &x : op->strings)
                tensor_product(x, lop, rop, omat);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
        if (omat != mat) {
            aromat->save_archive(omat);
            omat->deallocate();
        }
    }
    /** Rotation (renormalization) of a left-block operator tensor: c =
     * mpst_bra.T x a x mpst_ket. In the above expression, [x] means
     * multiplication. Note that the row and column of mpst_bra and mpst_ket are
     * for system and environment indices, respectively. Left rotation means
     * that system indices are contracted.
     * @param a Input operator tensor a (as a row vector of symbols).
     * @param mpst_bra Rotation matrix (bra MPS tensor) for row of sparse
     * matrices in a.
     * @param mpst_ket Rotation matrix (ket MPS tensor) for column of sparse
     * matrices in a.
     * @param c Output operator tensor c (as a row vector of symbols).
     */
    void left_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                     const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                     const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                     shared_ptr<OperatorTensor<S, FL>> &c) const override {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
            shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                make_shared<ArchivedSparseMatrix<S, FL>>(filename, offset);
            arc->save_archive(c->ops.at(op));
            c->ops.at(op)->deallocate();
            c->ops.at(op) = arc;
            offset += arc->total_memory;
        }
        for (size_t i = 0; i < a->lmat->data.size(); i++)
            if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->lmat->data[i]);
                shared_ptr<SparseMatrix<S, FL>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        a->ops.at(pa))
                        ->load_archive();
                shared_ptr<ArchivedSparseMatrix<S, FL>> armatc =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        c->ops.at(pa));
                shared_ptr<SparseMatrix<S, FL>> matc = armatc->load_archive();
                opf->tensor_rotate(mata, matc, mpst_bra, mpst_ket, false);
                armatc->save_archive(matc);
                matc->deallocate();
                mata->deallocate();
            }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    /** Rotation (renormalization) of a right-block operator tensor: c =
     * mpst_bra x a x mpst_ket.T. In the above expression, [x] means
     * multiplication. Note that the row and column of mpst_bra and mpst_ket are
     * for system and environment indices, respectively. Right rotation means
     * that environment indices are contracted.
     * @param a Input operator tensor a (as a column vector of symbols).
     * @param mpst_bra Rotation matrix (bra MPS tensor) for row of sparse
     * matrices in a.
     * @param mpst_ket Rotation matrix (ket MPS tensor) for column of sparse
     * matrices in a.
     * @param c Output operator tensor c (as a column vector of symbols).
     */
    void right_rotate(const shared_ptr<OperatorTensor<S, FL>> &a,
                      const shared_ptr<SparseMatrix<S, FL>> &mpst_bra,
                      const shared_ptr<SparseMatrix<S, FL>> &mpst_ket,
                      shared_ptr<OperatorTensor<S, FL>> &c) const override {
        for (auto &p : c->ops) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            c->ops.at(op)->allocate(c->ops.at(op)->info);
            shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                make_shared<ArchivedSparseMatrix<S, FL>>(filename, offset);
            arc->save_archive(c->ops.at(op));
            c->ops.at(op)->deallocate();
            c->ops.at(op) = arc;
            offset += arc->total_memory;
        }
        for (size_t i = 0; i < a->rmat->data.size(); i++)
            if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->rmat->data[i]);
                shared_ptr<SparseMatrix<S, FL>> mata =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        a->ops.at(pa))
                        ->load_archive();
                shared_ptr<ArchivedSparseMatrix<S, FL>> armatc =
                    dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                        c->ops.at(pa));
                shared_ptr<SparseMatrix<S, FL>> matc = armatc->load_archive();
                opf->tensor_rotate(mata, matc, mpst_bra, mpst_ket, true);
                armatc->save_archive(matc);
                matc->deallocate();
                mata->deallocate();
            }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    /** Compute the intermediates to speed up the tensor product operations in
     * the next blocking step. Intermediates are formed by collecting terms
     * sharing the same left or right operands during the MPO simplification
     * step.
     * @param names Operator names (symbols, only used in distributed
     * parallelization).
     * @param exprs Tensor product expressions (expressions of symbols).
     * @param a Symbol lookup table for symbols in the tensor product
     * expressions (updated).
     * @param left Whether this is for left-blocking or right-blocking.
     */
    void intermediates(const shared_ptr<Symbolic<S>> &names,
                       const shared_ptr<Symbolic<S>> &exprs,
                       const shared_ptr<OperatorTensor<S, FL>> &a,
                       bool left) const override {
        for (size_t i = 0; i < exprs->data.size(); i++)
            if (exprs->data[i]->get_type() == OpTypes::Sum) {
                shared_ptr<OpSum<S, FL>> expr =
                    dynamic_pointer_cast<OpSum<S, FL>>(exprs->data[i]);
                for (size_t j = 0; j < expr->strings.size(); j++)
                    if (expr->strings[j]->get_type() == OpTypes::SumProd) {
                        shared_ptr<OpSumProd<S, FL>> ex =
                            dynamic_pointer_cast<OpSumProd<S, FL>>(
                                expr->strings[j]);
                        if ((left && ex->b == nullptr) ||
                            (!left && ex->a == nullptr) || ex->c == nullptr)
                            continue;
                        if (a->ops.count(ex->c) != 0)
                            continue;
                        shared_ptr<SparseMatrix<S, FL>> tmp =
                            make_shared<SparseMatrix<S, FL>>();
                        shared_ptr<OpExpr<S>> opb =
                            abs_value((shared_ptr<OpExpr<S>>)ex->ops[0]);
                        assert(a->ops.count(opb) != 0);
                        tmp->allocate(a->ops.at(opb)->info);
                        for (size_t k = 0; k < ex->ops.size(); k++) {
                            shared_ptr<SparseMatrix<S, FL>> xmat =
                                dynamic_pointer_cast<
                                    ArchivedSparseMatrix<S, FL>>(
                                    a->ops.at(abs_value(
                                        (shared_ptr<OpExpr<S>>)ex->ops[k])))
                                    ->load_archive();
                            opf->iadd(tmp, xmat, ex->ops[k]->factor,
                                      ex->conjs[k]);
                            if (opf->seq->mode == SeqTypes::Simple)
                                opf->seq->simple_perform();
                            xmat->deallocate();
                        }
                        shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                            make_shared<ArchivedSparseMatrix<S, FL>>(filename,
                                                                     offset);
                        arc->save_archive(tmp);
                        tmp->deallocate();
                        a->ops[ex->c] = arc;
                        offset += arc->total_memory;
                    }
            }
    }
    /** Numerical transform from normal operators to complementary operators
     * near the middle site.
     * @param a Symbol lookup table for symbols in the tensor product
     * expressions (updated).
     * @param names List of complementary operator names (symbols).
     * @param exprs List of symbolic expression of complementary operators as
     * linear combination of normal operators.
     */
    void
    numerical_transform(const shared_ptr<OperatorTensor<S, FL>> &a,
                        const shared_ptr<Symbolic<S>> &names,
                        const shared_ptr<Symbolic<S>> &exprs) const override {
        assert(names->data.size() == exprs->data.size());
        assert((a->lmat == nullptr) ^ (a->rmat == nullptr));
        if (a->lmat == nullptr)
            a->rmat = names;
        else
            a->lmat = names;
        for (size_t i = 0; i < a->ops.size(); i++) {
            bool found = false;
            for (size_t k = 0; k < names->data.size(); k++) {
                if (exprs->data[k]->get_type() == OpTypes::Zero)
                    continue;
                shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
                shared_ptr<OpExpr<S>> expr =
                    exprs->data[k] *
                    ((FL)1.0 /
                     dynamic_pointer_cast<OpElement<S, FL>>(names->data[k])
                         ->factor);
                assert(a->ops.count(nop) != 0);
                switch (expr->get_type()) {
                case OpTypes::Sum: {
                    shared_ptr<OpSum<S, FL>> op =
                        dynamic_pointer_cast<OpSum<S, FL>>(expr);
                    found |= i < op->strings.size();
                    if (i < op->strings.size()) {
                        shared_ptr<OpElement<S, FL>> nexpr =
                            op->strings[i]->get_op();
                        assert(a->ops.count(nexpr) != 0);
                        shared_ptr<SparseMatrix<S, FL>> imat =
                            dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                                a->ops.at(nexpr))
                                ->load_archive();
                        shared_ptr<ArchivedSparseMatrix<S, FL>> aromat =
                            dynamic_pointer_cast<ArchivedSparseMatrix<S, FL>>(
                                a->ops.at(nop));
                        shared_ptr<SparseMatrix<S, FL>> omat =
                            aromat->load_archive();
                        opf->iadd(omat, imat, op->strings[i]->factor,
                                  op->strings[i]->conj != 0);
                        if (opf->seq->mode == SeqTypes::Simple)
                            opf->seq->simple_perform();
                        aromat->save_archive(omat);
                        omat->deallocate();
                        imat->deallocate();
                    }
                } break;
                case OpTypes::Zero:
                    break;
                default:
                    assert(false);
                    break;
                }
            }
            if (!found)
                break;
        }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    /** Tensor product operation in left blocking: c = a x b.
     * @param a Operator a (left block tensor).
     * @param b Operator b (dot block single-site tensor).
     * @param c Operator c (enlarged left block tensor).
     * @param cexprs Symbolic expression for the tensor product operation. If
     * nullptr, this is automatically contructed from expressions in a and b.
     * @param delayed The set of operator names for which the tensor product
     * operation should be delayed. The delayed tensor product will not be
     * performed here. Instead, it will be evaluated as three-tensor operations
     * later.
     */
    void left_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                       const shared_ptr<OperatorTensor<S, FL>> &b,
                       shared_ptr<OperatorTensor<S, FL>> &c,
                       const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                       OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr)
            left_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == c->lmat->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpElement<S, FL>> cop =
                    dynamic_pointer_cast<OpElement<S, FL>>(c->lmat->data[i]);
                shared_ptr<OpExpr<S>> op = abs_value(c->lmat->data[i]);
                shared_ptr<OpExpr<S>> expr =
                    exprs->data[i] * ((FL)1.0 / cop->factor);
                if (!delayed(cop->name)) {
                    c->ops.at(op)->allocate(c->ops.at(op)->info);
                    tensor_product(expr, a->ops, b->ops, c->ops.at(op));
                    shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                        make_shared<ArchivedSparseMatrix<S, FL>>(filename,
                                                                 offset);
                    arc->save_archive(c->ops.at(op));
                    c->ops.at(op)->deallocate();
                    c->ops.at(op) = arc;
                    offset += arc->total_memory;
                }
            }
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
    /** Tensor product operation in right blocking: c = b x a.
     * @param a Operator a (right block tensor).
     * @param b Operator b (dot block single-site tensor).
     * @param c Operator c (enlarged right block tensor).
     * @param cexprs Symbolic expression for the tensor product operation. If
     * nullptr, this is automatically contructed from expressions in b and a.
     * @param delayed The set of operator names for which the tensor product
     * operation should be delayed. The delayed tensor product will not be
     * performed here. Instead, it will be evaluated as three-tensor operations
     * later.
     */
    void right_contract(const shared_ptr<OperatorTensor<S, FL>> &a,
                        const shared_ptr<OperatorTensor<S, FL>> &b,
                        shared_ptr<OperatorTensor<S, FL>> &c,
                        const shared_ptr<Symbolic<S>> &cexprs = nullptr,
                        OpNamesSet delayed = OpNamesSet()) const override {
        if (a == nullptr)
            right_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == c->rmat->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpElement<S, FL>> cop =
                    dynamic_pointer_cast<OpElement<S, FL>>(c->rmat->data[i]);
                shared_ptr<OpExpr<S>> op = abs_value(c->rmat->data[i]);
                shared_ptr<OpExpr<S>> expr =
                    exprs->data[i] * ((FL)1.0 / cop->factor);
                if (!delayed(cop->name)) {
                    c->ops.at(op)->allocate(c->ops.at(op)->info);
                    tensor_product(expr, b->ops, a->ops, c->ops.at(op));
                    shared_ptr<ArchivedSparseMatrix<S, FL>> arc =
                        make_shared<ArchivedSparseMatrix<S, FL>>(filename,
                                                                 offset);
                    arc->save_archive(c->ops.at(op));
                    c->ops.at(op)->deallocate();
                    c->ops.at(op) = arc;
                    offset += arc->total_memory;
                }
            }
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
};

} // namespace block2
