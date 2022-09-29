
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

#include "../core/delayed_tensor_functions.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/symbolic.hpp"
#include "../core/tensor_functions.hpp"
#include "../core/threading.hpp"
#include "mpo.hpp"
#include "qc_hamiltonian.hpp"
#include <cassert>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

// MPO of identity operator
template <typename S, typename FL> struct IdentityMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::site_op_infos;
    // identity between differect pg
    IdentityMPO(const vector<shared_ptr<StateInfo<S>>> &bra_basis,
                const vector<shared_ptr<StateInfo<S>>> &ket_basis, S vacuum,
                S delta_quantum,
                const shared_ptr<OperatorFunctions<S, FL>> &opf,
                const vector<typename S::pg_t> &bra_orb_sym =
                    vector<typename S::pg_t>(),
                const vector<typename S::pg_t> &ket_orb_sym =
                    vector<typename S::pg_t>(),
                const string &tag = "ID")
        : MPO<S, FL>((int)bra_basis.size(), tag) {
        shared_ptr<OpElement<S, FL>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), vacuum);
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(bra_basis.size() == ket_basis.size());
        assert(bra_orb_sym.size() == ket_orb_sym.size());
        unordered_map<typename S::pg_t, set<typename S::pg_t>> ket_to_bra_map;
        vector<pair<typename S::pg_t, typename S::pg_t>> map_que(
            ket_orb_sym.size());
        for (size_t ik = 0; ik < ket_orb_sym.size(); ik++)
            map_que[ik] = make_pair(ket_orb_sym[ik], bra_orb_sym[ik]);
        size_t imq = 0;
        while (imq != map_que.size()) {
            if (!ket_to_bra_map.count(map_que[imq].first) ||
                !ket_to_bra_map.at(map_que[imq].first)
                     .count(map_que[imq].second)) {
                ket_to_bra_map[map_que[imq].first].insert(map_que[imq].second);
                for (auto &mm : ket_to_bra_map)
                    if (!ket_to_bra_map.count(
                            S::pg_mul(map_que[imq].first, mm.first)))
                        for (auto &mms : mm.second)
                            map_que.push_back(make_pair(
                                S::pg_mul(map_que[imq].first, mm.first),
                                S::pg_mul(map_que[imq].second, mms)));
            }
            imq++;
        }
        MPO<S, FL>::op = make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(),
                                                       delta_quantum);
        MPO<S, FL>::left_vacuum = vacuum;
        MPO<S, FL>::const_e = (typename const_fl_type<FL>::FL)0.0;
        // site operator infos
        site_op_infos.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++) {
            shared_ptr<SparseMatrixInfo<S>> info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            set<S> dqs;
            if (ket_to_bra_map.size() == 0) {
                // for CSR case, we cannot determine irrep mapping automatically
                // a mapping list must be given
                assert(bra_basis[m]->n == ket_basis[m]->n);
                for (size_t i = 0; i < bra_basis[m]->n; i++)
                    dqs.insert(
                        (bra_basis[m]->quanta[i] - ket_basis[m]->quanta[i])[0]);
            } else {
                for (size_t i = 0; i < bra_basis[m]->n; i++)
                    for (size_t j = 0; j < ket_basis[m]->n; j++) {
                        S qb = bra_basis[m]->quanta[i],
                          qk = ket_basis[m]->quanta[j];
                        S qbp = qk;
                        qbp.set_pg(qb.pg());
                        if (ket_to_bra_map.at((typename S::pg_t)qk.pg())
                                .count((typename S::pg_t)qb.pg()) &&
                            qbp == qb)
                            dqs.insert((qb - qk)[0]);
                    }
            }
            for (auto dq : dqs) {
                info = make_shared<SparseMatrixInfo<S>>(i_alloc);
                info->initialize(*bra_basis[m], *ket_basis[m], dq, false);
                site_op_infos[m].push_back(make_pair(dq, info));
            }
        }
        vector<set<S>> left_dqs(n_sites + 1), right_dqs(n_sites + 1);
        right_dqs[n_sites].insert(vacuum);
        left_dqs[0].insert(vacuum);
        for (int m = 0; m < n_sites; m++) {
            for (auto pdq : left_dqs[m])
                for (auto &soi : site_op_infos[m])
                    left_dqs[m + 1].insert(pdq + soi.first);
        }
        for (int m = n_sites - 1; m >= 0; m--) {
            for (auto pdq : right_dqs[m + 1])
                for (auto &soi : site_op_infos[m])
                    right_dqs[m].insert(soi.first + pdq);
        }
        vector<vector<S>> vldqs(n_sites + 1), vrdqs(n_sites + 1);
        for (int m = 0; m <= n_sites; m++) {
            vector<S> new_left_dqs, new_right_dqs;
            for (auto &dq : left_dqs[m])
                if (right_dqs[m].count(delta_quantum - dq)) {
                    vldqs[m].push_back(dq);
                    vrdqs[m].push_back(delta_quantum - dq);
                }
        }
        bool has_sparse = false;
        for (uint16_t m = 0; m < n_sites; m++) {
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            assert(vldqs[m + 1].size() != 0 && vldqs[m].size() != 0);
            if (m == 0)
                pmat =
                    make_shared<SymbolicRowVector<S>>((int)vldqs[m + 1].size());
            else if (m == n_sites - 1)
                pmat =
                    make_shared<SymbolicColumnVector<S>>((int)vldqs[m].size());
            else
                pmat = make_shared<SymbolicMatrix<S>>((int)vldqs[m].size(),
                                                      (int)vldqs[m + 1].size());
            for (int xi = 0; xi < vldqs[m].size(); xi++)
                for (int xj = 0; xj < vldqs[m + 1].size(); xj++) {
                    S dq = vldqs[m + 1][xj] - vldqs[m][xi];
                    for (int xk = 0; xk < site_op_infos[m].size(); xk++)
                        if (site_op_infos[m][xk].first == dq) {
                            if (dq.pg() == 0)
                                (*pmat)[{xi, xj}] = i_op;
                            else
                                (*pmat)[{xi, xj}] =
                                    make_shared<OpElement<S, FL>>(
                                        OpNames::I,
                                        SiteIndex((uint16_t)dq.pg()), dq);
                        }
                }
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>((int)vldqs[m + 1].size());
            for (int xj = 0; xj < vldqs[m + 1].size(); xj++) {
                S dq = vldqs[m + 1][xj];
                if (dq.pg() == 0)
                    (*plop)[xj] = i_op;
                else
                    (*plop)[xj] = make_shared<OpElement<S, FL>>(
                        OpNames::I, SiteIndex((uint16_t)dq.pg()), dq);
            }
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>((int)vrdqs[m].size());
            for (int xi = 0; xi < vrdqs[m].size(); xi++) {
                S dq = vrdqs[m][xi];
                if (dq.pg() == 0)
                    (*prop)[xi] = i_op;
                else
                    (*prop)[xi] = make_shared<OpElement<S, FL>>(
                        OpNames::I, SiteIndex((uint16_t)dq.pg()), dq);
            }
            this->right_operator_names.push_back(prop);
            bool no_sparse = bra_basis[m]->n == bra_basis[m]->n_states_total &&
                             ket_basis[m]->n == ket_basis[m]->n_states_total;
            map<S, MKL_INT> qbra_shift, qket_shift;
            // site operators
            for (int xk = 0; xk < site_op_infos[m].size(); xk++) {
                S dq = site_op_infos[m][xk].first;
                shared_ptr<SparseMatrixInfo<S>> info =
                    site_op_infos[m][xk].second;
                S mat_dq = info->delta_quantum;
                shared_ptr<OpElement<S, FL>> xop =
                    dq.pg() == 0
                        ? i_op
                        : make_shared<OpElement<S, FL>>(
                              OpNames::I, SiteIndex((uint16_t)dq.pg()), dq);
                assert(xop->q_label == info->delta_quantum);
                if (no_sparse) {
                    shared_ptr<SparseMatrix<S, FL>> mat =
                        make_shared<SparseMatrix<S, FL>>(d_alloc);
                    opt->ops[xop] = mat;
                    mat->allocate(info);
                    if (ket_to_bra_map.size() == 0) {
                        for (MKL_INT i = 0; i < mat->total_memory; i++)
                            mat->data[i] = 1;
                    } else {
                        for (int i = 0; i < info->n; i++)
                            if (ket_to_bra_map
                                    .at((typename S::pg_t)info->quanta[i]
                                            .get_ket()
                                            .pg())
                                    .count((typename S::pg_t)info->quanta[i]
                                               .get_bra(mat_dq)
                                               .pg()))
                                (*mat)[i].data[0] = 1;
                            else
                                (*mat)[i].data[0] = 0;
                    }
                } else {
                    has_sparse = true;
                    MPO<S, FL>::sparse_form[m] = 'S';
                    shared_ptr<CSRSparseMatrix<S, FL>> mat =
                        make_shared<CSRSparseMatrix<S, FL>>(d_alloc);
                    opt->ops[xop] = mat;
                    mat->initialize(info);
                    for (int i = 0; i < info->n; i++) {
                        shared_ptr<GCSRMatrix<FL>> cmat = mat->csr_data[i];
                        cmat->nnz = min(cmat->m, cmat->n);
                        cmat->allocate();
                        if (ket_to_bra_map.size() == 0 ||
                            ket_to_bra_map
                                .at((typename S::pg_t)info->quanta[i]
                                        .get_ket()
                                        .pg())
                                .count((typename S::pg_t)info->quanta[i]
                                           .get_bra(mat_dq)
                                           .pg())) {
                            for (MKL_INT j = 0; j < cmat->nnz; j++)
                                cmat->data[j] = 1;
                            if (cmat->nnz != cmat->size()) {
                                MKL_INT sh =
                                    qbra_shift[info->quanta[i].get_bra(mat_dq)];
                                MKL_INT shk =
                                    qket_shift[info->quanta[i].get_ket()];
                                assert(sh + cmat->nnz <= cmat->m);
                                assert(shk + cmat->nnz <= cmat->n);
                                for (MKL_INT j = 0; j < sh; j++)
                                    cmat->rows[j] = 0;
                                for (MKL_INT j = 0; j < cmat->nnz; j++)
                                    cmat->rows[j + sh] = j,
                                                   cmat->cols[j] = j + shk;
                                for (MKL_INT j = cmat->nnz; j <= cmat->m - sh;
                                     j++)
                                    cmat->rows[j + sh] = cmat->nnz;
                                if (cmat->m > cmat->n)
                                    qbra_shift[info->quanta[i].get_bra(
                                        mat_dq)] += cmat->nnz;
                                else if (cmat->n > cmat->m)
                                    qket_shift[info->quanta[i].get_ket()] +=
                                        cmat->nnz;
                            }
                        } else {
                            for (MKL_INT j = 0; j < cmat->nnz; j++)
                                cmat->data[j] = 0;
                            if (cmat->nnz != cmat->size()) {
                                for (MKL_INT j = 0; j < cmat->nnz; j++)
                                    cmat->rows[j] = j, cmat->cols[j] = j;
                                for (MKL_INT j = cmat->nnz; j <= cmat->m; j++)
                                    cmat->rows[j] = cmat->nnz;
                            }
                        }
                    }
                }
            }
            this->tensors.push_back(opt);
            this->save_tensor(m);
            this->save_left_operators(m);
            this->save_right_operators(m);
            this->unload_tensor(m);
            this->unload_left_operators(m);
            this->unload_right_operators(m);
        }
        if (has_sparse) {
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(
                make_shared<CSROperatorFunctions<S, FL>>(opf->cg));
            MPO<S, FL>::tf->opf->seq = opf->seq;
        } else
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(opf);
    }
    IdentityMPO(const vector<shared_ptr<StateInfo<S>>> &bra_basis,
                const vector<shared_ptr<StateInfo<S>>> &ket_basis, S vacuum,
                const shared_ptr<OperatorFunctions<S, FL>> &opf,
                const string &tag = "ID")
        : MPO<S, FL>((int)bra_basis.size(), tag) {
        shared_ptr<OpElement<S, FL>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), vacuum);
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(bra_basis.size() == ket_basis.size());
        MPO<S, FL>::op = i_op;
        MPO<S, FL>::left_vacuum = vacuum;
        MPO<S, FL>::const_e = (typename const_fl_type<FL>::FL)0.0;
        // site operator infos
        site_op_infos.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++) {
            shared_ptr<SparseMatrixInfo<S>> info =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            info->initialize(*bra_basis[m], *ket_basis[m], vacuum, false);
            site_op_infos[m].push_back(make_pair(vacuum, info));
        }
        bool has_sparse = false;
        for (uint16_t m = 0; m < n_sites; m++) {
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(1);
            else if (m == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(1);
            else
                pmat = make_shared<SymbolicMatrix<S>>(1, 1);
            (*pmat)[{0, 0}] = i_op;
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(1);
            (*plop)[0] = i_op;
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(1);
            (*prop)[0] = i_op;
            this->right_operator_names.push_back(prop);
            bool no_sparse = bra_basis[m]->n == bra_basis[m]->n_states_total &&
                             ket_basis[m]->n == ket_basis[m]->n_states_total;
            // site operators
            shared_ptr<SparseMatrixInfo<S>> info = site_op_infos[m][0].second;
            if (no_sparse) {
                shared_ptr<SparseMatrix<S, FL>> mat =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
                opt->ops[i_op] = mat;
                mat->allocate(info);
                for (MKL_INT i = 0; i < mat->total_memory; i++)
                    mat->data[i] = 1;
            } else {
                has_sparse = true;
                MPO<S, FL>::sparse_form[m] = 'S';
                shared_ptr<CSRSparseMatrix<S, FL>> mat =
                    make_shared<CSRSparseMatrix<S, FL>>(d_alloc);
                opt->ops[i_op] = mat;
                mat->initialize(info);
                for (int i = 0; i < info->n; i++) {
                    shared_ptr<GCSRMatrix<FL>> cmat = mat->csr_data[i];
                    cmat->nnz = min(cmat->m, cmat->n);
                    cmat->allocate();
                    for (MKL_INT j = 0; j < cmat->nnz; j++)
                        cmat->data[j] = 1;
                    if (cmat->nnz != cmat->size()) {
                        for (MKL_INT j = 0; j < cmat->nnz; j++)
                            cmat->rows[j] = j, cmat->cols[j] = j;
                        for (MKL_INT j = cmat->nnz; j <= cmat->m; j++)
                            cmat->rows[j] = cmat->nnz;
                    }
                }
            }
            this->tensors.push_back(opt);
            this->save_tensor(m);
            this->save_left_operators(m);
            this->save_right_operators(m);
            this->unload_tensor(m);
            this->unload_left_operators(m);
            this->unload_right_operators(m);
        }
        if (has_sparse) {
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(
                make_shared<CSROperatorFunctions<S, FL>>(opf->cg));
            MPO<S, FL>::tf->opf->seq = opf->seq;
        } else
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(opf);
    }
    IdentityMPO(const shared_ptr<Hamiltonian<S, FL>> &hamil,
                const string &tag = "ID")
        : MPO<S, FL>(hamil->n_sites, tag) {
        shared_ptr<OpElement<S, FL>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::op = i_op;
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        MPO<S, FL>::const_e = (typename const_fl_type<FL>::FL)0.0;
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        MPO<S, FL>::basis = hamil->basis;
        for (uint16_t m = 0; m < n_sites; m++) {
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(1);
            else if (m == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(1);
            else
                pmat = make_shared<SymbolicMatrix<S>>(1, 1);
            (*pmat)[{0, 0}] = i_op;
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(1);
            (*plop)[0] = i_op;
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(1);
            (*prop)[0] = i_op;
            this->right_operator_names.push_back(prop);
            // site operators
            hamil->filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
            this->save_tensor(m);
            this->save_left_operators(m);
            this->save_right_operators(m);
            this->unload_tensor(m);
            this->unload_left_operators(m);
            this->unload_right_operators(m);
        }
    }
};

// MPO of single site operator
template <typename S, typename FL> struct SiteMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::site_op_infos;
    // build site operator op at site k
    SiteMPO(const shared_ptr<Hamiltonian<S, FL>> &hamil,
            const shared_ptr<OpElement<S, FL>> &op, int k = -1,
            const string &tag = "SITE")
        : MPO<S, FL>(hamil->n_sites, tag) {
        shared_ptr<OpElement<S, FL>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::op = op;
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        MPO<S, FL>::const_e = (typename const_fl_type<FL>::FL)0.0;
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        if (k == -1) {
            assert(op->site_index.size() >= 1);
            k = op->site_index[0];
        }
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        for (uint16_t pm = 0; pm < n_sites; pm++) {
            uint16_t m = pm + n_orbs_big_left - 1;
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (pm == 0)
                pmat = make_shared<SymbolicRowVector<S>>(1);
            else if (pm == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(1);
            else
                pmat = make_shared<SymbolicMatrix<S>>(1, 1);
            (*pmat)[{0, 0}] = m == k ? op : i_op;
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(1);
            (*plop)[0] = m >= k ? op : i_op;
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(1);
            (*prop)[0] = m <= k ? op : i_op;
            this->right_operator_names.push_back(prop);
            // site operators
            hamil->filter_site_ops(pm, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
            this->save_tensor(pm);
            this->save_left_operators(pm);
            this->save_right_operators(pm);
            this->unload_tensor(pm);
            this->unload_left_operators(pm);
            this->unload_right_operators(pm);
        }
    }
};

// sum of MPO of single site operators
template <typename S, typename FL> struct LocalMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::site_op_infos;
    LocalMPO(const shared_ptr<Hamiltonian<S, FL>> &hamil,
             const vector<shared_ptr<OpElement<S, FL>>> &ops,
             const string &tag = "LOCAL")
        : MPO<S, FL>(hamil->n_sites, tag) {
        shared_ptr<OpElement<S, FL>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        shared_ptr<OpElement<S, FL>> h_op = make_shared<OpElement<S, FL>>(
            ops[0]->name, SiteIndex(), ops[0]->q_label);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::op = h_op;
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        assert((uint16_t)ops.size() == n_sites);
        for (auto op : ops)
            assert(op->q_label == ops[0]->q_label);
        MPO<S, FL>::const_e = (typename const_fl_type<FL>::FL)0.0;
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        // FIXME: need special name for the sum operators
        shared_ptr<OpExpr<S>> f_op =
            n_orbs_big_left == 1
                ? ops[0]
                : sum(vector<shared_ptr<OpExpr<S>>>(
                      ops.begin(), ops.begin() + n_orbs_big_left));
        shared_ptr<OpExpr<S>> l_op =
            n_orbs_big_right == 1
                ? ops[n_sites - 1]
                : sum(vector<shared_ptr<OpExpr<S>>>(
                      ops.end() - n_orbs_big_right, ops.end()));
        for (uint16_t pm = 0; pm < n_sites; pm++) {
            uint16_t m = pm + n_orbs_big_left - 1;
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (pm == 0) {
                pmat = make_shared<SymbolicRowVector<S>>(2);
                (*pmat)[{0, 0}] = f_op;
                (*pmat)[{0, 1}] = i_op;
            } else if (pm == n_sites - 1) {
                pmat = make_shared<SymbolicColumnVector<S>>(2);
                (*pmat)[{0, 0}] = i_op;
                (*pmat)[{1, 0}] = l_op;
            } else {
                pmat = make_shared<SymbolicMatrix<S>>(2, 2);
                (*pmat)[{0, 0}] = i_op;
                (*pmat)[{1, 0}] = ops[m];
                (*pmat)[{1, 1}] = i_op;
            }
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop;
            if (pm == n_sites - 1) {
                plop = make_shared<SymbolicRowVector<S>>(1);
                (*plop)[0] = pm == 0 ? f_op : h_op;
            } else {
                plop = make_shared<SymbolicRowVector<S>>(2);
                (*plop)[0] = pm == 0 ? f_op : h_op;
                (*plop)[1] = i_op;
            }
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop;
            if (pm == 0) {
                prop = make_shared<SymbolicColumnVector<S>>(1);
                (*prop)[0] = pm == n_sites - 1 ? l_op : h_op;
            } else {
                prop = make_shared<SymbolicColumnVector<S>>(2);
                (*prop)[0] = i_op;
                (*prop)[1] = pm == n_sites - 1 ? l_op : h_op;
            }
            this->right_operator_names.push_back(prop);
            // site operators
            hamil->filter_site_ops(pm, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
            this->save_tensor(pm);
            this->save_left_operators(pm);
            this->save_right_operators(pm);
            this->unload_tensor(pm);
            this->unload_left_operators(pm);
            this->unload_right_operators(pm);
        }
    }
};

// Quantum Chemistry MPO schemes
// NC: Normal (left block) / Complementary (right block) scheme
// CN: Complementary (left block) / Normal (right block) scheme
// Conventional: Use NC scheme before middle site
//               and CN scheme after middle site
enum QCTypes : uint8_t { NC = 1, CN = 2, Conventional = 4 };

template <typename, typename, typename = void> struct MPOQC;

// Quantum chemistry MPO (non-spin-adapted)
// trans_delta: the number of sites in trans center
template <typename S, typename FL>
struct MPOQC<S, FL, typename S::is_sz_t> : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    QCTypes mode;
    const bool symmetrized_p = true;
    MPOQC(const shared_ptr<HamiltonianQC<S, FL>> &hamil,
          QCTypes mode = QCTypes::NC, const string &tag = "HQC",
          int trans_center = -1, int trans_delta = 1, bool symmetrized_p = true)
        : MPO<S, FL>(hamil->n_sites, tag), mode(mode),
          symmetrized_p(symmetrized_p) {
        shared_ptr<OpExpr<S>> h_op = make_shared<OpElement<S, FL>>(
            OpNames::H, SiteIndex(), hamil->vacuum);
        shared_ptr<OpExpr<S>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> c_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            d_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> mc_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            md_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> rd_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            r_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> mrd_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            mr_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> a_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> ad_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> b_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> p_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> pd_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> q_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(4)));
#else
        shared_ptr<OpExpr<S>> c_op[n_orbs][2], d_op[n_orbs][2];
        shared_ptr<OpExpr<S>> mc_op[n_orbs][2], md_op[n_orbs][2];
        shared_ptr<OpExpr<S>> rd_op[n_orbs][2], r_op[n_orbs][2];
        shared_ptr<OpExpr<S>> mrd_op[n_orbs][2], mr_op[n_orbs][2];
        shared_ptr<OpExpr<S>> a_op[n_orbs][n_orbs][4];
        shared_ptr<OpExpr<S>> ad_op[n_orbs][n_orbs][4];
        shared_ptr<OpExpr<S>> b_op[n_orbs][n_orbs][4];
        shared_ptr<OpExpr<S>> p_op[n_orbs][n_orbs][4];
        shared_ptr<OpExpr<S>> pd_op[n_orbs][n_orbs][4];
        shared_ptr<OpExpr<S>> q_op[n_orbs][n_orbs][4];
#endif
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::op = dynamic_pointer_cast<OpElement<S, FL>>(h_op);
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        MPO<S, FL>::const_e = hamil->e();
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        uint16_t trans_l = -1, trans_r = n_sites;
        if (trans_center == -1)
            trans_center = n_sites >> 1;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN))
            trans_l = trans_center - 1, trans_r = trans_center;
        else if (mode == QCTypes::Conventional)
            trans_l = trans_center - 1, trans_r = trans_center + trans_delta;
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint16_t m = 0; m < n_orbs; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::C, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil->orb_sym[m]));
                d_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], S::pg_inv(hamil->orb_sym[m])));
                mc_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::C, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil->orb_sym[m]), -1.0);
                md_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], S::pg_inv(hamil->orb_sym[m])), -1.0);
                rd_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::RD, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil->orb_sym[m]));
                r_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::R, SiteIndex({m}, {s}),
                    S(-1, -sz[s], S::pg_inv(hamil->orb_sym[m])));
                mrd_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::RD, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil->orb_sym[m]), -1.0);
                mr_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::R, SiteIndex({m}, {s}),
                    S(-1, -sz[s], S::pg_inv(hamil->orb_sym[m])), -1.0);
            }
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    a_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::A, sidx,
                        S(2, sz_plus[s],
                          S::pg_mul(hamil->orb_sym[i], hamil->orb_sym[j])));
                    ad_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::AD, sidx,
                        S(-2, -sz_plus[s],
                          S::pg_mul(S::pg_inv(hamil->orb_sym[i]),
                                    S::pg_inv(hamil->orb_sym[j]))));
                    b_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s],
                          S::pg_mul(hamil->orb_sym[i],
                                    S::pg_inv(hamil->orb_sym[j]))));
                    p_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::P, sidx,
                        S(-2, -sz_plus[s],
                          S::pg_mul(S::pg_inv(hamil->orb_sym[i]),
                                    S::pg_inv(hamil->orb_sym[j]))));
                    pd_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::PD, sidx,
                        S(2, sz_plus[s],
                          S::pg_mul(hamil->orb_sym[i], hamil->orb_sym[j])));
                    q_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::Q, sidx,
                        S(0, -sz_minus[s],
                          S::pg_inv(S::pg_mul(hamil->orb_sym[i],
                                              S::pg_inv(hamil->orb_sym[j])))));
                }
        uint16_t n_repeat_m =
            mode == QCTypes::Conventional && trans_l + 1 >= 0 &&
                    trans_r - 1 < n_sites && trans_l + 1 <= trans_r - 1
                ? 1 + (trans_r - 1) - (trans_l + 1)
                : 0;
        this->left_operator_names.resize(n_sites, nullptr);
        this->right_operator_names.resize(n_sites, nullptr);
        this->tensors.resize(n_sites, nullptr);
        for (uint16_t m = 0; m < n_sites; m++)
            this->tensors[m] = make_shared<OperatorTensor<S, FL>>();
        int ntg = threading->activate_global();
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int xxm = 0; xxm < (int)(n_sites + n_repeat_m); xxm++) {
            uint16_t xm = (uint16_t)xxm;
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (uint16_t xm = 0; xm < n_sites + n_repeat_m; xm++) {
#endif
            uint16_t pm = xm;
            int p;
            bool repeat_m = false;
            if (n_repeat_m && xm > trans_r - 1) {
                pm = xm - n_repeat_m;
                if (pm <= trans_r - 1)
                    repeat_m = true;
            }
            uint16_t m = pm + n_orbs_big_left - 1;
            shared_ptr<Symbolic<S>> pmat;
            int lshape, rshape;
            QCTypes effective_mode;
            if (mode == QCTypes::NC ||
                ((mode & QCTypes::NC) && pm <= trans_l) ||
                (mode == QCTypes::Conventional && pm <= trans_r - 1 &&
                 !repeat_m))
                effective_mode = QCTypes::NC;
            else if (mode == QCTypes::CN ||
                     ((mode & QCTypes::CN) && pm >= trans_r) ||
                     (mode == QCTypes::Conventional && pm >= trans_l + 1))
                effective_mode = QCTypes::CN;
            else
                assert(false);
            switch (effective_mode) {
            case QCTypes::NC:
                lshape = 2 + 4 * n_orbs + 12 * m * m;
                rshape = 2 + 4 * n_orbs + 12 * (m + 1) * (m + 1);
                break;
            case QCTypes::CN:
                lshape = 2 + 4 * n_orbs + 12 * (n_orbs - m) * (n_orbs - m);
                rshape =
                    2 + 4 * n_orbs + 12 * (n_orbs - m - 1) * (n_orbs - m - 1);
                break;
            default:
                assert(false);
            }
            if (pm == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (pm == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (pm == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        mat[{0, p + j}] = c_op[j][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        mat[{0, p + j}] = d_op[j][s];
                    p += m + 1;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        mat[{0, p + j - m - 1}] = rd_op[j][s];
                    p += n_orbs - (m + 1);
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        mat[{0, p + j - m - 1}] = mr_op[j][s];
                    p += n_orbs - (m + 1);
                }
            } else if (pm == n_sites - 1) {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = r_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = mrd_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_orbs; j++)
                        mat[{p + j - m, 0}] = d_op[j][s];
                    p += n_orbs - m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint16_t j = m; j < n_orbs; j++)
                        mat[{p + j - m, 0}] = c_op[j][s];
                    p += n_orbs - m;
                }
            }
            switch (effective_mode) {
            case QCTypes::NC:
                if (pm == 0) {
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                mat[{0, p + k}] = a_op[j][k][s];
                            p += m + 1;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                mat[{0, p + k}] = ad_op[j][k][s];
                            p += m + 1;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                mat[{0, p + k}] = b_op[j][k][s];
                            p += m + 1;
                        }
                    assert(p == mat.n);
                } else {
                    if (pm != n_sites - 1) {
                        mat[{0, 0}] = i_op;
                        mat[{1, 0}] = h_op;
                        p = 2;
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint16_t j = 0; j < m; j++)
                                mat[{p + j, 0}] = r_op[j][s];
                            p += m;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint16_t j = 0; j < m; j++)
                                mat[{p + j, 0}] = mrd_op[j][s];
                            p += m;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{p, 0}] = d_op[m][s];
                            p += n_orbs - m;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{p, 0}] = c_op[m][s];
                            p += n_orbs - m;
                        }
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = (FL)0.5 * p_op[j][k][s];
                            p += m;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = (FL)0.5 * pd_op[j][k][s];
                            p += m;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = q_op[j][k][s];
                            p += m;
                        }
                    assert(p == mat.m);
                }
                if (pm != 0 && pm != n_sites - 1) {
                    mat[{1, 1}] = i_op;
                    p = 2;
                    // pointers
                    int pi = 1;
                    int pc[2] = {2, 2 + m};
                    int pd[2] = {2 + m * 2, 2 + m * 3};
                    int prd[2] = {2 + m * 4 - m, 2 + m * 3 + n_orbs - m};
                    int pr[2] = {2 + m * 2 + n_orbs * 2 - m,
                                 2 + m + n_orbs * 3 - m};
                    int pa[4] = {
                        2 + n_orbs * 4 + m * m * 0, 2 + n_orbs * 4 + m * m * 1,
                        2 + n_orbs * 4 + m * m * 2, 2 + n_orbs * 4 + m * m * 3};
                    int pad[4] = {
                        2 + n_orbs * 4 + m * m * 4, 2 + n_orbs * 4 + m * m * 5,
                        2 + n_orbs * 4 + m * m * 6, 2 + n_orbs * 4 + m * m * 7};
                    int pb[4] = {2 + n_orbs * 4 + m * m * 8,
                                 2 + n_orbs * 4 + m * m * 9,
                                 2 + n_orbs * 4 + m * m * 10,
                                 2 + n_orbs * 4 + m * m * 11};
                    // C
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pc[s] + j, p + j}] = i_op;
                        mat[{pi, p + m}] = c_op[m][s];
                        p += m + 1;
                    }
                    // D
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pd[s] + j, p + j}] = i_op;
                        mat[{pi, p + m}] = d_op[m][s];
                        p += m + 1;
                    }
                    // RD
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{prd[s] + i, p + i - (m + 1)}] = i_op;
                            mat[{pi, p + i - (m + 1)}] = rd_op[i][s];
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = 0; k < m; k++) {
                                    mat[{pd[sp] + k, p + i - (m + 1)}] =
                                        (FL)-1.0 * pd_op[k][i][sp | (s << 1)];
                                    mat[{pc[sp] + k, p + i - (m + 1)}] =
                                        q_op[k][i][sp | (s << 1)];
                                }
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = 0; j < m; j++)
                                        for (uint16_t l = 0; l < m; l++) {
                                            FL f = hamil->v(s, sp, j, i, l, m);
                                            mat[{pa[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] =
                                                f * d_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = 0; j < m; j++)
                                        for (uint16_t l = 0; l < m; l++) {
                                            FL f0 = (FL)0.5 *
                                                    hamil->v(s, sp, j, i, l, m),
                                               f1 = (FL)-0.5 *
                                                    hamil->v(s, sp, l, i, j, m);
                                            mat[{pa[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f0 * d_op[m][sp];
                                            mat[{pa[sp | (s << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f1 * d_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = 0; k < m; k++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        FL f = hamil->v(s, sp, m, i, l, k);
                                        mat[{pb[sp | (sp << 1)] + l * m + k,
                                             p + i - (m + 1)}] = f * c_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t k = 0; k < m; k++) {
                                        FL f = (FL)-1.0 *
                                               hamil->v(s, sp, j, i, m, k);
                                        mat[{pb[s | (sp << 1)] + j * m + k,
                                             p + i - (m + 1)}] +=
                                            f * c_op[m][sp];
                                    }
                        }
                        p += n_orbs - (m + 1);
                    }
                    // R
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{pr[s] + i, p + i - (m + 1)}] = i_op;
                            mat[{pi, p + i - (m + 1)}] = mr_op[i][s];
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = 0; k < m; k++) {
                                    mat[{pc[sp] + k, p + i - (m + 1)}] =
                                        p_op[k][i][sp | (s << 1)];
                                    mat[{pd[sp] + k, p + i - (m + 1)}] =
                                        (FL)-1.0 * q_op[i][k][s | (sp << 1)];
                                }
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = 0; j < m; j++)
                                        for (uint16_t l = 0; l < m; l++) {
                                            FL f = (FL)-1.0 *
                                                   hamil->v(s, sp, i, j, m, l);
                                            mat[{pad[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] =
                                                f * c_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = 0; j < m; j++)
                                        for (uint16_t l = 0; l < m; l++) {
                                            FL f0 = (FL)-0.5 *
                                                    hamil->v(s, sp, i, j, m, l),
                                               f1 = (FL)0.5 *
                                                    hamil->v(s, sp, i, l, m, j);
                                            mat[{pad[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f0 * c_op[m][sp];
                                            mat[{pad[sp | (s << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f1 * c_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = 0; k < m; k++)
                                    for (uint16_t l = 0; l < m; l++) {
                                        FL f = (FL)-1.0 *
                                               hamil->v(s, sp, i, m, k, l);
                                        mat[{pb[sp | (sp << 1)] + k * m + l,
                                             p + i - (m + 1)}] = f * d_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = 0; j < m; j++)
                                    for (uint16_t k = 0; k < m; k++) {
                                        FL f = (FL)(-1.0) * (FL)(-1.0) *
                                               hamil->v(s, sp, i, j, k, m);
                                        mat[{pb[sp | (s << 1)] + k * m + j,
                                             p + i - (m + 1)}] =
                                            f * d_op[m][sp];
                                    }
                        }
                        p += n_orbs - (m + 1);
                    }
                    // A
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint16_t i = 0; i < m; i++)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pa[s] + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{pc[s & 1] + i, p + i * (m + 1) + m}] =
                                c_op[m][s >> 1];
                            mat[{pc[s >> 1] + i, p + m * (m + 1) + i}] =
                                mc_op[m][s & 1];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = a_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // AD
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint16_t i = 0; i < m; i++)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pad[s] + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{pd[s & 1] + i, p + i * (m + 1) + m}] =
                                md_op[m][s >> 1];
                            mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] =
                                d_op[m][s & 1];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // B
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint16_t i = 0; i < m; i++)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pb[s] + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{pc[s & 1] + i, p + i * (m + 1) + m}] =
                                d_op[m][s >> 1];
                            mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] =
                                mc_op[m][s & 1];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = b_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    assert(p == mat.n);
                }
                break;
            case QCTypes::CN:
                if (pm == n_sites - 1) {
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                mat[{p + k - m, 0}] = a_op[j][k][s];
                            p += n_orbs - m;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                mat[{p + k - m, 0}] = ad_op[j][k][s];
                            p += n_orbs - m;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                mat[{p + k - m, 0}] = b_op[j][k][s];
                            p += n_orbs - m;
                        }
                    assert(p == mat.m);
                } else {
                    if (pm != 0) {
                        mat[{1, 0}] = h_op;
                        mat[{1, 1}] = i_op;
                        p = 2;
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{1, p + m}] = c_op[m][s];
                            p += m + 1;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{1, p + m}] = d_op[m][s];
                            p += m + 1;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{1, p + j - m - 1}] = rd_op[j][s];
                            p += n_orbs - m - 1;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{1, p + j - m - 1}] = mr_op[j][s];
                            p += n_orbs - m - 1;
                        }
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                mat[{!!pm, p + k - m - 1}] =
                                    (FL)0.5 * p_op[j][k][s];
                            p += n_orbs - m - 1;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                mat[{!!pm, p + k - m - 1}] =
                                    (FL)0.5 * pd_op[j][k][s];
                            p += n_orbs - m - 1;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                mat[{!!pm, p + k - m - 1}] = q_op[j][k][s];
                            p += n_orbs - m - 1;
                        }
                    assert(p == mat.n);
                }
                if (pm != 0 && pm != n_sites - 1) {
                    mat[{0, 0}] = i_op;
                    p = 2;
                    // pointers
                    int mm = n_orbs - m - 1;
                    int ppm = n_orbs - m;
                    int pi = 0;
                    int pr[2] = {2, 2 + m + 1};
                    int prd[2] = {2 + (m + 1) * 2, 2 + (m + 1) * 3};
                    int pd[2] = {2 + (m + 1) * 4 - m - 1,
                                 2 + (m + 1) * 3 + n_orbs - m - 1};
                    int pc[2] = {2 + (m + 1) * 2 + n_orbs * 2 - m - 1,
                                 2 + (m + 1) + n_orbs * 3 - m - 1};
                    int pa[4] = {2 + n_orbs * 4 + mm * mm * 0,
                                 2 + n_orbs * 4 + mm * mm * 1,
                                 2 + n_orbs * 4 + mm * mm * 2,
                                 2 + n_orbs * 4 + mm * mm * 3};
                    int pad[4] = {2 + n_orbs * 4 + mm * mm * 4,
                                  2 + n_orbs * 4 + mm * mm * 5,
                                  2 + n_orbs * 4 + mm * mm * 6,
                                  2 + n_orbs * 4 + mm * mm * 7};
                    int pb[4] = {2 + n_orbs * 4 + mm * mm * 8,
                                 2 + n_orbs * 4 + mm * mm * 9,
                                 2 + n_orbs * 4 + mm * mm * 10,
                                 2 + n_orbs * 4 + mm * mm * 11};
                    // R
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{p + i, pi}] = r_op[i][s];
                            mat[{p + i, pr[s] + i}] = i_op;
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = m + 1; j < n_orbs; j++)
                                        for (uint16_t l = m + 1; l < n_orbs;
                                             l++) {
                                            FL f = hamil->v(s, sp, i, j, m, l);
                                            mat[{p + i, pad[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] =
                                                f * c_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = m + 1; j < n_orbs; j++)
                                        for (uint16_t l = m + 1; l < n_orbs;
                                             l++) {
                                            FL f0 = (FL)0.5 *
                                                    hamil->v(s, sp, i, j, m, l);
                                            FL f1 = (FL)-0.5 *
                                                    hamil->v(s, sp, i, l, m, j);
                                            mat[{p + i, pad[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f0 * c_op[m][sp];
                                            mat[{p + i, pad[sp | (s << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f1 * c_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    for (uint16_t l = m + 1; l < n_orbs; l++) {
                                        FL f = hamil->v(s, sp, i, m, k, l);
                                        mat[{p + i, pb[sp | (sp << 1)] +
                                                        (k - m - 1) * mm + l -
                                                        m - 1}] =
                                            f * d_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = m + 1; j < n_orbs; j++)
                                    for (uint16_t k = m + 1; k < n_orbs; k++) {
                                        FL f = (FL)(-1.0) *
                                               hamil->v(s, sp, i, j, k, m);
                                        mat[{p + i, pb[sp | (s << 1)] +
                                                        (k - m - 1) * mm + j -
                                                        m - 1}] =
                                            f * d_op[m][sp];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = m + 1; k < n_orbs; k++) {
                                    mat[{p + i, pc[sp] + k}] =
                                        p_op[i][k][s | (sp << 1)];
                                    mat[{p + i, pd[sp] + k}] =
                                        q_op[i][k][s | (sp << 1)];
                                }
                        }
                        p += m;
                    }
                    // RD
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{p + i, pi}] = mrd_op[i][s];
                            mat[{p + i, prd[s] + i}] = i_op;
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = m + 1; j < n_orbs; j++)
                                        for (uint16_t l = m + 1; l < n_orbs;
                                             l++) {
                                            FL f = (FL)-1.0 *
                                                   hamil->v(s, sp, j, i, l, m);
                                            mat[{p + i, pa[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] =
                                                f * d_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint16_t j = m + 1; j < n_orbs; j++)
                                        for (uint16_t l = m + 1; l < n_orbs;
                                             l++) {
                                            FL f0 = (FL)-0.5 *
                                                    hamil->v(s, sp, j, i, l, m);
                                            FL f1 = (FL)0.5 *
                                                    hamil->v(s, sp, l, i, j, m);
                                            mat[{p + i, pa[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f0 * d_op[m][sp];
                                            mat[{p + i, pa[sp | (s << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f1 * d_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    for (uint16_t l = m + 1; l < n_orbs; l++) {
                                        FL f = (FL)-1.0 *
                                               hamil->v(s, sp, m, i, l, k);
                                        mat[{p + i, pb[sp | (sp << 1)] +
                                                        (l - m - 1) * mm + k -
                                                        m - 1}] =
                                            f * c_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t j = m + 1; j < n_orbs; j++)
                                    for (uint16_t k = m + 1; k < n_orbs; k++) {
                                        FL f = hamil->v(s, sp, j, i, m, k);
                                        mat[{p + i, pb[s | (sp << 1)] +
                                                        (j - m - 1) * mm + k -
                                                        m - 1}] =
                                            f * c_op[m][sp];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint16_t k = m + 1; k < n_orbs; k++) {
                                    mat[{p + i, pd[sp] + k}] =
                                        (FL)-1.0 * pd_op[i][k][s | (sp << 1)];
                                    mat[{p + i, pc[sp] + k}] =
                                        (FL)-1.0 * q_op[k][i][sp | (s << 1)];
                                }
                        }
                        p += m;
                    }
                    // D
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p + m - m, pi}] = d_op[m][s];
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{p + j - m, pd[s] + j}] = i_op;
                        p += n_orbs - m;
                    }
                    // C
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p + m - m, pi}] = c_op[m][s];
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{p + j - m, pc[s] + j}] = i_op;
                        p += n_orbs - m;
                    }
                    // A
                    for (uint8_t s = 0; s < 4; s++) {
                        mat[{p + (m - m) * ppm + m - m, pi}] = a_op[m][m][s];
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{p + (m - m) * ppm + i - m, pc[s >> 1] + i}] =
                                c_op[m][s & 1];
                            mat[{p + (i - m) * ppm + m - m, pc[s & 1] + i}] =
                                mc_op[m][s >> 1];
                        }
                        for (uint16_t i = m + 1; i < n_orbs; i++)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{p + (i - m) * ppm + j - m,
                                     pa[s] + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += ppm * ppm;
                    }
                    // AD
                    for (uint8_t s = 0; s < 4; s++) {
                        mat[{p + (m - m) * ppm + m - m, pi}] = ad_op[m][m][s];
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{p + (m - m) * ppm + i - m, pd[s >> 1] + i}] =
                                md_op[m][s & 1];
                            mat[{p + (i - m) * ppm + m - m, pd[s & 1] + i}] =
                                d_op[m][s >> 1];
                        }
                        for (uint16_t i = m + 1; i < n_orbs; i++)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{p + (i - m) * ppm + j - m,
                                     pad[s] + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += ppm * ppm;
                    }
                    // B
                    for (uint8_t s = 0; s < 4; s++) {
                        mat[{p + (m - m) * ppm + m - m, pi}] = b_op[m][m][s];
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{p + (m - m) * ppm + i - m, pd[s >> 1] + i}] =
                                c_op[m][s & 1];
                            mat[{p + (i - m) * ppm + m - m, pc[s & 1] + i}] =
                                md_op[m][s >> 1];
                        }
                        for (uint16_t i = m + 1; i < n_orbs; i++)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{p + (i - m) * ppm + j - m,
                                     pb[s] + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += ppm * ppm;
                    }
                    assert(p == mat.m);
                }
                break;
            default:
                assert(false);
                break;
            }
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[pm];
            if (mode != QCTypes::Conventional || pm < trans_l + 1 ||
                pm > trans_r - 1)
                opt->lmat = opt->rmat = pmat;
            else if (!repeat_m)
                opt->rmat = pmat;
            else
                opt->lmat = pmat;
            // operator names
            if (opt->lmat == pmat) {
                shared_ptr<SymbolicRowVector<S>> plop;
                if (pm == n_sites - 1)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (pm != n_sites - 1) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            lop[p + j] = c_op[j][s];
                        p += m + 1;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            lop[p + j] = d_op[j][s];
                        p += m + 1;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            lop[p + j - (m + 1)] = rd_op[j][s];
                        p += n_orbs - (m + 1);
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            lop[p + j - (m + 1)] = mr_op[j][s];
                        p += n_orbs - (m + 1);
                    }
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = 0; j < m + 1; j++) {
                                for (uint16_t k = 0; k < m + 1; k++)
                                    lop[p + k] = a_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = 0; j < m + 1; j++) {
                                for (uint16_t k = 0; k < m + 1; k++)
                                    lop[p + k] = ad_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = 0; j < m + 1; j++) {
                                for (uint16_t k = 0; k < m + 1; k++)
                                    lop[p + k] = b_op[j][k][s];
                                p += m + 1;
                            }
                        break;
                    case QCTypes::CN:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    lop[p + k - m - 1] =
                                        (FL)0.5 * p_op[j][k][s];
                                p += n_orbs - m - 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    lop[p + k - m - 1] =
                                        (FL)0.5 * pd_op[j][k][s];
                                p += n_orbs - m - 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    lop[p + k - m - 1] = q_op[j][k][s];
                                p += n_orbs - m - 1;
                            }
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    assert(p == rshape);
                }
                this->left_operator_names[pm] = plop;
            }
            if (opt->rmat == pmat) {
                shared_ptr<SymbolicColumnVector<S>> prop;
                if (pm == 0)
                    prop = make_shared<SymbolicColumnVector<S>>(1);
                else
                    prop = make_shared<SymbolicColumnVector<S>>(lshape);
                SymbolicColumnVector<S> &rop = *prop;
                if (pm == 0)
                    rop[0] = h_op;
                else {
                    rop[0] = i_op;
                    rop[1] = h_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            rop[p + j] = r_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            rop[p + j] = mrd_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m; j < n_orbs; j++)
                            rop[p + j - m] = d_op[j][s];
                        p += n_orbs - m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = m; j < n_orbs; j++)
                            rop[p + j - m] = c_op[j][s];
                        p += n_orbs - m;
                    }
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = 0; j < m; j++) {
                                for (uint16_t k = 0; k < m; k++)
                                    rop[p + k] = (FL)0.5 * p_op[j][k][s];
                                p += m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = 0; j < m; j++) {
                                for (uint16_t k = 0; k < m; k++)
                                    rop[p + k] = (FL)0.5 * pd_op[j][k][s];
                                p += m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = 0; j < m; j++) {
                                for (uint16_t k = 0; k < m; k++)
                                    rop[p + k] = q_op[j][k][s];
                                p += m;
                            }
                        break;
                    case QCTypes::CN:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = m; j < n_orbs; j++) {
                                for (uint16_t k = m; k < n_orbs; k++)
                                    rop[p + k - m] = a_op[j][k][s];
                                p += n_orbs - m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = m; j < n_orbs; j++) {
                                for (uint16_t k = m; k < n_orbs; k++)
                                    rop[p + k - m] = ad_op[j][k][s];
                                p += n_orbs - m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint16_t j = m; j < n_orbs; j++) {
                                for (uint16_t k = m; k < n_orbs; k++)
                                    rop[p + k - m] = b_op[j][k][s];
                                p += n_orbs - m;
                            }
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    assert(p == lshape);
                }
                this->right_operator_names[pm] = prop;
            }
            if (mode != QCTypes::Conventional || pm < trans_l + 1 ||
                pm > trans_r - 1) {
                this->save_tensor(pm);
                this->save_left_operators(pm);
                this->save_right_operators(pm);
                this->unload_tensor(pm);
                this->unload_left_operators(pm);
                this->unload_right_operators(pm);
            }
        }
        if (mode == QCTypes::Conventional)
            for (uint16_t pm = trans_l + 1; pm <= trans_r - 1; pm++) {
                this->save_tensor(pm);
                this->save_left_operators(pm);
                this->unload_tensor(pm);
                this->save_right_operators(pm);
                this->unload_left_operators(pm);
                this->unload_right_operators(pm);
            }
        SeqTypes seqt = hamil->opf->seq->mode;
        hamil->opf->seq->mode = SeqTypes::None;
        const uint16_t m_start = hamil->get_n_orbs_left() > 0 ? 1 : 0;
        const uint16_t m_end =
            hamil->get_n_orbs_right() > 0 ? n_sites - 1 : n_sites;
        int ntgb = frame_<FP>()->minimal_memory_usage ? 1 : ntg;
#pragma omp parallel for schedule(dynamic) num_threads(ntgb)
#ifdef _MSC_VER
        for (int m = (int)m_start; m < (int)m_end; m++) {
#else
        for (uint16_t m = m_start; m < m_end; m++) {
#endif
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO build ... " << setw(4) << m << " / " << setw(4)
                     << n_sites << endl;
            this->load_tensor(m);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[m];
            hamil->filter_site_ops((uint16_t)m, {opt->lmat, opt->rmat},
                                   opt->ops);
            this->save_tensor(m);
            this->unload_tensor(m);
        }
        if (hamil->get_n_orbs_left() > 0 && n_sites > 0) {
            this->load_tensor(0);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[0];
            hamil->filter_site_ops(0, {opt->lmat, opt->rmat}, opt->ops);
            this->save_tensor(0);
            this->unload_tensor(0);
        }
        if (hamil->get_n_orbs_right() > 0 && n_sites > 0) {
            this->load_tensor(n_sites - 1);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[n_sites - 1];
            hamil->filter_site_ops(n_sites - 1, {opt->lmat, opt->rmat},
                                   opt->ops);
            this->save_tensor(n_sites - 1);
            this->unload_tensor(n_sites - 1);
        }
        hamil->opf->seq->mode = seqt;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN) ||
            mode == QCTypes::Conventional) {
            uint16_t m, pm;
            MPO<S, FL>::schemer = make_shared<MPOSchemer<S>>(trans_l, trans_r);
            if (frame_<FP>()->minimal_memory_usage)
                cout << "left middle transform ... " << endl;
            // left transform
            pm = trans_l;
            m = pm + n_orbs_big_left - 1;
            int new_rshape =
                2 + 4 * n_orbs + 12 * (n_orbs - m - 1) * (n_orbs - m - 1);
            MPO<S, FL>::schemer->left_new_operator_names =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            MPO<S, FL>::schemer->left_new_operator_exprs =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            SymbolicRowVector<S> &lop =
                *MPO<S, FL>::schemer->left_new_operator_names;
            SymbolicRowVector<S> &lexpr =
                *MPO<S, FL>::schemer->left_new_operator_exprs;
            this->load_left_operators(pm);
            for (int i = 0; i < 2 + 4 * n_orbs; i++)
                lop[i] = this->left_operator_names[pm]->data[i];
            this->unload_left_operators(pm);
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
            for (int sj = 0; sj < (int)(4 * (n_orbs - (m + 1))); sj++) {
                uint8_t s = (uint8_t)(sj / (n_orbs - (m + 1)));
                uint16_t j = (uint16_t)(sj % (n_orbs - (m + 1)) + m + 1);
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg) collapse(2)
            for (uint8_t s = 0; s < 4; s++)
                for (uint16_t j = m + 1; j < n_orbs; j++) {
#endif
                vector<shared_ptr<OpExpr<S>>> exprs;
                exprs.reserve((m + 1) * (m + 1));
                for (uint16_t k = m + 1; k < n_orbs; k++) {
                    int p = (k - m - 1) + (j - m - 1) * (n_orbs - m - 1) +
                            s * (n_orbs - m - 1) * (n_orbs - m - 1);
                    exprs.clear();
                    p += 2 + 4 * n_orbs;
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(s & 1, s >> 1, j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 *
                                     hamil->v(s & 1, s >> 1, j, g, k, h)) *
                                    ad_op[g][h][s]);
                    lop[p] = (FL)0.5 * p_op[j][k][s];
                    lexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 4 * (n_orbs - m - 1) * (n_orbs - m - 1);
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(s & 1, s >> 1, j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 *
                                     hamil->v(s & 1, s >> 1, j, g, k, h)) *
                                    a_op[g][h][s]);
                    lop[p] = (FL)0.5 * pd_op[j][k][s];
                    lexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 4 * (n_orbs - m - 1) * (n_orbs - m - 1);
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++) {
                            if (abs(hamil->v(s & 1, s >> 1, j, h, g, k)) > TINY)
                                exprs.push_back(
                                    -hamil->v(s & 1, s >> 1, j, h, g, k) *
                                    b_op[g][h][((s & 1) << 1) | (s >> 1)]);
                            if ((s & 1) == (s >> 1))
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    if (abs(hamil->v(s & 1, sp, j, k, g, h)) >
                                        TINY)
                                        exprs.push_back(
                                            hamil->v(s & 1, sp, j, k, g, h) *
                                            b_op[g][h][(sp << 1) | sp]);
                        }
                    lop[p] = q_op[j][k][s];
                    lexpr[p] = sum(exprs);
                }
            }
            if (frame_<FP>()->minimal_memory_usage)
                cout << "right middle transform ... " << endl;
            // right transform
            pm = trans_r - 1;
            m = pm + n_orbs_big_left - 1;
            int new_lshape = 2 + 4 * n_orbs + 12 * (m + 1) * (m + 1);
            MPO<S, FL>::schemer->right_new_operator_names =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            MPO<S, FL>::schemer->right_new_operator_exprs =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            SymbolicColumnVector<S> &rop =
                *MPO<S, FL>::schemer->right_new_operator_names;
            SymbolicColumnVector<S> &rexpr =
                *MPO<S, FL>::schemer->right_new_operator_exprs;
            this->load_right_operators(pm + 1);
            for (int i = 0; i < 2 + 4 * n_orbs; i++)
                rop[i] = this->right_operator_names[pm + 1]->data[i];
            this->unload_right_operators(pm + 1);
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
            for (int sj = 0; sj < (int)(4 * (m + 1)); sj++) {
                uint8_t s = (uint8_t)(sj / (m + 1));
                uint16_t j = (uint16_t)(sj % (m + 1));
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg) collapse(2)
            for (uint8_t s = 0; s < 4; s++)
                for (uint16_t j = 0; j < m + 1; j++) {
#endif
                vector<shared_ptr<OpExpr<S>>> exprs;
                exprs.reserve((n_orbs - m - 1) * (n_orbs - m - 1));
                for (uint16_t k = 0; k < m + 1; k++) {
                    int p = k + j * (m + 1) + s * (m + 1) * (m + 1);
                    exprs.clear();
                    p += 2 + 4 * n_orbs;
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(s & 1, s >> 1, j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 *
                                     hamil->v(s & 1, s >> 1, j, g, k, h)) *
                                    ad_op[g][h][s]);
                    rop[p] = (FL)0.5 * p_op[j][k][s];
                    rexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 4 * (m + 1) * (m + 1);
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(s & 1, s >> 1, j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 *
                                     hamil->v(s & 1, s >> 1, j, g, k, h)) *
                                    a_op[g][h][s]);
                    rop[p] = (FL)0.5 * pd_op[j][k][s];
                    rexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 4 * (m + 1) * (m + 1);
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++) {
                            if (abs(hamil->v(s & 1, s >> 1, j, h, g, k)) > TINY)
                                exprs.push_back(
                                    -hamil->v(s & 1, s >> 1, j, h, g, k) *
                                    b_op[g][h][((s & 1) << 1) | (s >> 1)]);
                            if ((s & 1) == (s >> 1))
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    if (abs(hamil->v(s & 1, sp, j, k, g, h)) >
                                        TINY)
                                        exprs.push_back(
                                            hamil->v(s & 1, sp, j, k, g, h) *
                                            b_op[g][h][(sp << 1) | sp]);
                        }
                    rop[p] = q_op[j][k][s];
                    rexpr[p] = sum(exprs);
                }
            }
            this->save_schemer();
            this->unload_schemer();
        }
        threading->activate_normal();
    }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            if (this->tensors[m] != nullptr)
                this->tensors[m]->deallocate();
    }
};

// Quantum chemistry MPO (spin-adapted)
template <typename S, typename FL>
struct MPOQC<S, FL, typename S::is_su2_t> : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    QCTypes mode;
    MPOQC(const shared_ptr<HamiltonianQC<S, FL>> &hamil,
          QCTypes mode = QCTypes::NC, const string &tag = "HQC",
          int trans_center = -1, int trans_delta = 1)
        : MPO<S, FL>(hamil->n_sites, tag), mode(mode) {
        shared_ptr<OpExpr<S>> h_op = make_shared<OpElement<S, FL>>(
            OpNames::H, SiteIndex(), hamil->vacuum);
        shared_ptr<OpExpr<S>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
#ifdef _MSC_VER
        vector<shared_ptr<OpExpr<S>>> c_op(n_orbs), d_op(n_orbs);
        vector<shared_ptr<OpExpr<S>>> mc_op(n_orbs), md_op(n_orbs);
        vector<shared_ptr<OpExpr<S>>> trd_op(n_orbs), tr_op(n_orbs);
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> a_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(2)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> ad_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(2)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> b_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(2)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> p_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(2)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> pd_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(2)));
        vector<vector<vector<shared_ptr<OpExpr<S>>>>> q_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(
                        n_orbs, vector<shared_ptr<OpExpr<S>>>(2)));
#else
        shared_ptr<OpExpr<S>> c_op[n_orbs], d_op[n_orbs];
        shared_ptr<OpExpr<S>> mc_op[n_orbs], md_op[n_orbs];
        shared_ptr<OpExpr<S>> trd_op[n_orbs], tr_op[n_orbs];
        shared_ptr<OpExpr<S>> a_op[n_orbs][n_orbs][2];
        shared_ptr<OpExpr<S>> ad_op[n_orbs][n_orbs][2];
        shared_ptr<OpExpr<S>> b_op[n_orbs][n_orbs][2];
        shared_ptr<OpExpr<S>> p_op[n_orbs][n_orbs][2];
        shared_ptr<OpExpr<S>> pd_op[n_orbs][n_orbs][2];
        shared_ptr<OpExpr<S>> q_op[n_orbs][n_orbs][2];
#endif
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::op = dynamic_pointer_cast<OpElement<S, FL>>(h_op);
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        MPO<S, FL>::const_e = hamil->e();
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        uint16_t trans_l = -1, trans_r = n_sites;
        if (trans_center == -1)
            trans_center = n_sites >> 1;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN))
            trans_l = trans_center - 1, trans_r = trans_center;
        else if (mode == QCTypes::Conventional)
            trans_l = trans_center - 1, trans_r = trans_center + trans_delta;
        for (uint16_t m = 0; m < n_orbs; m++) {
            c_op[m] = make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex(m),
                                                    S(1, 1, hamil->orb_sym[m]));
            d_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::D, SiteIndex(m),
                S(-1, 1, S::pg_inv(hamil->orb_sym[m])));
            mc_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::C, SiteIndex(m), S(1, 1, hamil->orb_sym[m]), -1.0);
            md_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::D, SiteIndex(m),
                S(-1, 1, S::pg_inv(hamil->orb_sym[m])), -1.0);
            trd_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::RD, SiteIndex(m), S(1, 1, hamil->orb_sym[m]), 2.0);
            tr_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::R, SiteIndex(m),
                S(-1, 1, S::pg_inv(hamil->orb_sym[m])), 2.0);
        }
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++)
                for (uint8_t s = 0; s < 2; s++) {
                    a_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::A, SiteIndex(i, j, s),
                        S(2, s * 2,
                          S::pg_mul(hamil->orb_sym[i], hamil->orb_sym[j])));
                    ad_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::AD, SiteIndex(i, j, s),
                        S(-2, s * 2,
                          S::pg_mul(S::pg_inv(hamil->orb_sym[i]),
                                    S::pg_inv(hamil->orb_sym[j]))));
                    b_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::B, SiteIndex(i, j, s),
                        S(0, s * 2,
                          S::pg_mul(hamil->orb_sym[i],
                                    S::pg_inv(hamil->orb_sym[j]))));
                    p_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::P, SiteIndex(i, j, s),
                        S(-2, s * 2,
                          S::pg_mul(S::pg_inv(hamil->orb_sym[i]),
                                    S::pg_inv(hamil->orb_sym[j]))));
                    pd_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::PD, SiteIndex(i, j, s),
                        S(2, s * 2,
                          S::pg_mul(hamil->orb_sym[i], hamil->orb_sym[j])));
                    q_op[i][j][s] = make_shared<OpElement<S, FL>>(
                        OpNames::Q, SiteIndex(i, j, s),
                        S(0, s * 2,
                          S::pg_inv(S::pg_mul(hamil->orb_sym[i],
                                              S::pg_inv(hamil->orb_sym[j])))));
                }
        uint16_t n_repeat_m =
            mode == QCTypes::Conventional && trans_l + 1 >= 0 &&
                    trans_r - 1 < n_sites && trans_l + 1 <= trans_r - 1
                ? 1 + (trans_r - 1) - (trans_l + 1)
                : 0;
        this->left_operator_names.resize(n_sites, nullptr);
        this->right_operator_names.resize(n_sites, nullptr);
        this->tensors.resize(n_sites, nullptr);
        for (uint16_t m = 0; m < n_sites; m++)
            this->tensors[m] = make_shared<OperatorTensor<S, FL>>();
        int ntg = threading->activate_global();
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int xxm = 0; xxm < (int)(n_sites + n_repeat_m); xxm++) {
            uint16_t xm = (uint16_t)xxm;
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (uint16_t xm = 0; xm < n_sites + n_repeat_m; xm++) {
#endif
            uint16_t pm = xm;
            int p;
            bool repeat_m = false;
            if (n_repeat_m && xm > trans_r - 1) {
                pm = xm - n_repeat_m;
                if (pm <= trans_r - 1)
                    repeat_m = true;
            }
            uint16_t m = pm + n_orbs_big_left - 1;
            shared_ptr<Symbolic<S>> pmat;
            int lshape, rshape;
            QCTypes effective_mode;
            if (mode == QCTypes::NC ||
                ((mode & QCTypes::NC) && pm <= trans_l) ||
                (mode == QCTypes::Conventional && pm <= trans_r - 1 &&
                 !repeat_m))
                effective_mode = QCTypes::NC;
            else if (mode == QCTypes::CN ||
                     ((mode & QCTypes::CN) && pm >= trans_r) ||
                     (mode == QCTypes::Conventional && pm >= trans_l + 1))
                effective_mode = QCTypes::CN;
            else
                assert(false);
            switch (effective_mode) {
            case QCTypes::NC:
                lshape = 2 + 2 * n_orbs + 6 * m * m;
                rshape = 2 + 2 * n_orbs + 6 * (m + 1) * (m + 1);
                break;
            case QCTypes::CN:
                lshape = 2 + 2 * n_orbs + 6 * (n_orbs - m) * (n_orbs - m);
                rshape =
                    2 + 2 * n_orbs + 6 * (n_orbs - m - 1) * (n_orbs - m - 1);
                break;
            default:
                assert(false);
            }
            if (pm == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (pm == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (pm == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                p = 2;
                for (uint16_t j = 0; j < m + 1; j++)
                    mat[{0, p + j}] = c_op[j];
                p += m + 1;
                for (uint16_t j = 0; j < m + 1; j++)
                    mat[{0, p + j}] = d_op[j];
                p += m + 1;
                for (uint16_t j = m + 1; j < n_orbs; j++)
                    mat[{0, p + j - m - 1}] = trd_op[j];
                p += n_orbs - (m + 1);
                for (uint16_t j = m + 1; j < n_orbs; j++)
                    mat[{0, p + j - m - 1}] = tr_op[j];
                p += n_orbs - (m + 1);
            } else if (pm == n_sites - 1) {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint16_t j = 0; j < m; j++)
                    mat[{p + j, 0}] = tr_op[j];
                p += m;
                for (uint16_t j = 0; j < m; j++)
                    mat[{p + j, 0}] = trd_op[j];
                p += m;
                for (uint16_t j = m; j < n_orbs; j++)
                    mat[{p + j - m, 0}] = d_op[j];
                p += n_orbs - m;
                for (uint16_t j = m; j < n_orbs; j++)
                    mat[{p + j - m, 0}] = c_op[j];
                p += n_orbs - m;
            }
            switch (effective_mode) {
            case QCTypes::NC:
                if (pm == 0) {
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                mat[{0, p + k}] = a_op[j][k][s];
                            p += m + 1;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                mat[{0, p + k}] = ad_op[j][k][s];
                            p += m + 1;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                mat[{0, p + k}] = b_op[j][k][s];
                            p += m + 1;
                        }
                    assert(p == mat.n);
                } else {
                    if (pm != n_sites - 1) {
                        mat[{0, 0}] = i_op;
                        mat[{1, 0}] = h_op;
                        p = 2;
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = tr_op[j];
                        p += m;
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = trd_op[j];
                        p += m;
                        mat[{p, 0}] = d_op[m];
                        p += n_orbs - m;
                        mat[{p, 0}] = c_op[m];
                        p += n_orbs - m;
                    }
                    vector<FL> su2_factor{(FL)-0.5, (FL)(-0.5 * sqrt(3))};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = su2_factor[s] * p_op[j][k][s];
                            p += m;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                mat[{p + k, 0}] =
                                    su2_factor[s] * pd_op[j][k][s];
                            p += m;
                        }
                    su2_factor = {(FL)1.0, (FL)sqrt(3)};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = su2_factor[s] * q_op[j][k][s];
                            p += m;
                        }
                    assert(p == mat.m);
                }
                if (pm != 0 && pm != n_sites - 1) {
                    mat[{1, 1}] = i_op;
                    p = 2;
                    // pointers
                    int pi = 1, pc = 2, pd = 2 + m;
                    int prd = 2 + m + m - m, pr = 2 + m + n_orbs - m;
                    int pa0 = 2 + (n_orbs << 1) + m * m * 0;
                    int pa1 = 2 + (n_orbs << 1) + m * m * 1;
                    int pad0 = 2 + (n_orbs << 1) + m * m * 2;
                    int pad1 = 2 + (n_orbs << 1) + m * m * 3;
                    int pb0 = 2 + (n_orbs << 1) + m * m * 4;
                    int pb1 = 2 + (n_orbs << 1) + m * m * 5;
                    // C
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pc + j, p + j}] = i_op;
                    mat[{pi, p + m}] = c_op[m];
                    p += m + 1;
                    // D
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pd + j, p + j}] = i_op;
                    mat[{pi, p + m}] = d_op[m];
                    p += m + 1;
                    // RD
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{prd + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = trd_op[i];
                        for (uint16_t k = 0; k < m; k++) {
                            mat[{pd + k, p + i - (m + 1)}] =
                                (FL)2.0 *
                                ((FL)(-0.5) * pd_op[k][i][0] +
                                 (FL)(0.5 * sqrt(3)) * pd_op[k][i][1]);
                            mat[{pc + k, p + i - (m + 1)}] =
                                (FL)2.0 *
                                ((FL)(0.5) * q_op[k][i][0] +
                                 (FL)(-0.5 * sqrt(3)) * q_op[k][i][1]);
                        }
                        for (uint16_t j = 0; j < m; j++)
                            for (uint16_t l = 0; l < m; l++) {
                                FL f0 =
                                    hamil->v(j, i, l, m) + hamil->v(l, i, j, m);
                                FL f1 =
                                    hamil->v(j, i, l, m) - hamil->v(l, i, j, m);
                                mat[{pa0 + j * m + l, p + i - (m + 1)}] =
                                    f0 * (FL)(-0.5) * d_op[m];
                                mat[{pa1 + j * m + l, p + i - (m + 1)}] =
                                    f1 * (FL)(0.5 * sqrt(3)) * d_op[m];
                            }
                        for (uint16_t k = 0; k < m; k++)
                            for (uint16_t l = 0; l < m; l++) {
                                FL f = (FL)2.0 * hamil->v(m, i, l, k) -
                                       hamil->v(l, i, m, k);
                                mat[{pb0 + l * m + k, p + i - (m + 1)}] =
                                    f * c_op[m];
                            }
                        for (uint16_t j = 0; j < m; j++)
                            for (uint16_t k = 0; k < m; k++) {
                                FL f = hamil->v(j, i, m, k) * (FL)sqrt(3);
                                mat[{pb1 + j * m + k, p + i - (m + 1)}] =
                                    f * c_op[m];
                            }
                    }
                    p += n_orbs - (m + 1);
                    // R
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{pr + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = tr_op[i];
                        for (uint16_t k = 0; k < m; k++) {
                            mat[{pc + k, p + i - (m + 1)}] =
                                (FL)2.0 *
                                ((FL)(-0.5) * p_op[k][i][0] +
                                 (FL)(-0.5 * sqrt(3)) * p_op[k][i][1]);
                            mat[{pd + k, p + i - (m + 1)}] =
                                (FL)2.0 * ((FL)(0.5) * q_op[i][k][0] +
                                           (FL)(0.5 * sqrt(3)) * q_op[i][k][1]);
                        }
                        for (uint16_t j = 0; j < m; j++)
                            for (uint16_t l = 0; l < m; l++) {
                                FL f0 =
                                    hamil->v(i, j, m, l) + hamil->v(i, l, m, j);
                                FL f1 =
                                    hamil->v(i, j, m, l) - hamil->v(i, l, m, j);
                                mat[{pad0 + j * m + l, p + i - (m + 1)}] =
                                    f0 * (FL)(-0.5) * c_op[m];
                                mat[{pad1 + j * m + l, p + i - (m + 1)}] =
                                    f1 * (FL)(-0.5 * sqrt(3)) * c_op[m];
                            }
                        for (uint16_t k = 0; k < m; k++)
                            for (uint16_t l = 0; l < m; l++) {
                                FL f = (FL)2.0 * hamil->v(i, m, k, l) -
                                       hamil->v(i, l, k, m);
                                mat[{pb0 + k * m + l, p + i - (m + 1)}] =
                                    f * d_op[m];
                            }
                        for (uint16_t j = 0; j < m; j++)
                            for (uint16_t k = 0; k < m; k++) {
                                FL f = (FL)(-1.0) * hamil->v(i, j, k, m) *
                                       (FL)sqrt(3);
                                mat[{pb1 + k * m + j, p + i - (m + 1)}] =
                                    f * d_op[m];
                            }
                    }
                    p += n_orbs - (m + 1);
                    // A
                    for (uint8_t s = 0; s < 2; s++) {
                        int pa = s ? pa1 : pa0;
                        for (uint16_t i = 0; i < m; i++)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pa + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{pc + i, p + i * (m + 1) + m}] = c_op[m];
                            mat[{pc + i, p + m * (m + 1) + i}] =
                                s ? mc_op[m] : c_op[m];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = a_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // AD
                    for (uint8_t s = 0; s < 2; s++) {
                        int pad = s ? pad1 : pad0;
                        for (uint16_t i = 0; i < m; i++)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pad + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{pd + i, p + i * (m + 1) + m}] =
                                s ? md_op[m] : d_op[m];
                            mat[{pd + i, p + m * (m + 1) + i}] = d_op[m];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // B
                    for (uint8_t s = 0; s < 2; s++) {
                        int pb = s ? pb1 : pb0;
                        for (uint16_t i = 0; i < m; i++)
                            for (uint16_t j = 0; j < m; j++)
                                mat[{pb + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint16_t i = 0; i < m; i++) {
                            mat[{pc + i, p + i * (m + 1) + m}] = d_op[m];
                            mat[{pd + i, p + m * (m + 1) + i}] =
                                s ? mc_op[m] : c_op[m];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = b_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    assert(p == mat.n);
                }
                break;
            case QCTypes::CN:
                if (pm == n_sites - 1) {
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                mat[{p + k - m, 0}] = a_op[j][k][s];
                            p += n_orbs - m;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                mat[{p + k - m, 0}] = ad_op[j][k][s];
                            p += n_orbs - m;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                mat[{p + k - m, 0}] = b_op[j][k][s];
                            p += n_orbs - m;
                        }
                    assert(p == mat.m);
                } else {
                    if (pm != 0) {
                        mat[{1, 0}] = h_op;
                        mat[{1, 1}] = i_op;
                        p = 2;
                        mat[{1, p + m}] = c_op[m];
                        p += m + 1;
                        mat[{1, p + m}] = d_op[m];
                        p += m + 1;
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{1, p + j - m - 1}] = trd_op[j];
                        p += n_orbs - m - 1;
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{1, p + j - m - 1}] = tr_op[j];
                        p += n_orbs - m - 1;
                    }
                    vector<FL> su2_factor{(FL)-0.5, (FL)(-0.5 * sqrt(3))};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                mat[{!!pm, p + k - m - 1}] =
                                    su2_factor[s] * p_op[j][k][s];
                            p += n_orbs - m - 1;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                mat[{!!pm, p + k - m - 1}] =
                                    su2_factor[s] * pd_op[j][k][s];
                            p += n_orbs - m - 1;
                        }
                    su2_factor = {(FL)1.0, (FL)sqrt(3)};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                mat[{!!pm, p + k - m - 1}] =
                                    su2_factor[s] * q_op[j][k][s];
                            p += n_orbs - m - 1;
                        }
                    assert(p == mat.n);
                }
                if (pm != 0 && pm != n_sites - 1) {
                    mat[{0, 0}] = i_op;
                    p = 2;
                    // pointers
                    int mm = n_orbs - m - 1;
                    int ppm = n_orbs - m;
                    int pi = 0, pr = 2, prd = 2 + m + 1;
                    int pd = 2 + m + m + 2 - m - 1,
                        pc = 2 + m + 1 + n_orbs - m - 1;
                    int pa0 = 2 + (n_orbs << 1) + mm * mm * 0;
                    int pa1 = 2 + (n_orbs << 1) + mm * mm * 1;
                    int pad0 = 2 + (n_orbs << 1) + mm * mm * 2;
                    int pad1 = 2 + (n_orbs << 1) + mm * mm * 3;
                    int pb0 = 2 + (n_orbs << 1) + mm * mm * 4;
                    int pb1 = 2 + (n_orbs << 1) + mm * mm * 5;
                    // R
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{p + i, pi}] = tr_op[i];
                        mat[{p + i, pr + i}] = i_op;
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            for (uint16_t l = m + 1; l < n_orbs; l++) {
                                FL f0 =
                                    hamil->v(i, j, m, l) + hamil->v(i, l, m, j);
                                FL f1 =
                                    hamil->v(i, j, m, l) - hamil->v(i, l, m, j);
                                mat[{p + i, pad0 + (j - m - 1) * mm + l - m -
                                                1}] = f0 * (FL)(-0.5) * c_op[m];
                                mat[{p + i,
                                     pad1 + (j - m - 1) * mm + l - m - 1}] =
                                    f1 * (FL)(0.5 * sqrt(3)) * c_op[m];
                            }
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            for (uint16_t l = m + 1; l < n_orbs; l++) {
                                FL f = (FL)2.0 * hamil->v(i, m, k, l) -
                                       hamil->v(i, l, k, m);
                                mat[{p + i, pb0 + (k - m - 1) * mm + l - m -
                                                1}] = f * d_op[m];
                            }
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            for (uint16_t k = m + 1; k < n_orbs; k++) {
                                FL f = hamil->v(i, j, k, m) * (FL)sqrt(3);
                                mat[{p + i, pb1 + (k - m - 1) * mm + j - m -
                                                1}] = f * d_op[m];
                            }
                        for (uint16_t k = m + 1; k < n_orbs; k++) {
                            mat[{p + i, pc + k}] =
                                (FL)2.0 *
                                ((FL)(-0.5) * p_op[i][k][0] +
                                 (FL)(-0.5 * sqrt(3)) * p_op[i][k][1]);
                            mat[{p + i, pd + k}] =
                                (FL)2.0 *
                                ((FL)(0.5) * q_op[i][k][0] +
                                 (FL)(-0.5 * sqrt(3)) * q_op[i][k][1]);
                        }
                    }
                    p += m;
                    // RD
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{p + i, pi}] = trd_op[i];
                        mat[{p + i, prd + i}] = i_op;
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            for (uint16_t l = m + 1; l < n_orbs; l++) {
                                FL f0 =
                                    hamil->v(j, i, l, m) + hamil->v(l, i, j, m);
                                FL f1 =
                                    hamil->v(j, i, l, m) - hamil->v(l, i, j, m);
                                mat[{p + i, pa0 + (j - m - 1) * mm + l - m -
                                                1}] = f0 * (FL)(-0.5) * d_op[m];
                                mat[{p + i,
                                     pa1 + (j - m - 1) * mm + l - m - 1}] =
                                    f1 * (FL)(-0.5 * sqrt(3)) * d_op[m];
                            }
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            for (uint16_t l = m + 1; l < n_orbs; l++) {
                                FL f = (FL)2.0 * hamil->v(m, i, l, k) -
                                       hamil->v(l, i, m, k);
                                mat[{p + i, pb0 + (l - m - 1) * mm + k - m -
                                                1}] = f * c_op[m];
                            }
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            for (uint16_t k = m + 1; k < n_orbs; k++) {
                                FL f = (FL)(-1.0) * hamil->v(j, i, m, k) *
                                       (FL)sqrt(3);
                                mat[{p + i, pb1 + (j - m - 1) * mm + k - m -
                                                1}] = f * c_op[m];
                            }
                        for (uint16_t k = m + 1; k < n_orbs; k++) {
                            mat[{p + i, pd + k}] =
                                (FL)2.0 *
                                ((FL)(-0.5) * pd_op[i][k][0] +
                                 (FL)(0.5 * sqrt(3)) * pd_op[i][k][1]);
                            mat[{p + i, pc + k}] =
                                (FL)2.0 * ((FL)(0.5) * q_op[k][i][0] +
                                           (FL)(0.5 * sqrt(3)) * q_op[k][i][1]);
                        }
                    }
                    p += m;
                    // D
                    mat[{p + m - m, pi}] = d_op[m];
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        mat[{p + j - m, pd + j}] = i_op;
                    p += n_orbs - m;
                    // C
                    mat[{p + m - m, pi}] = c_op[m];
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        mat[{p + j - m, pc + j}] = i_op;
                    p += n_orbs - m;
                    // A
                    for (uint8_t s = 0; s < 2; s++) {
                        int pa = s ? pa1 : pa0;
                        mat[{p + (m - m) * ppm + m - m, pi}] = a_op[m][m][s];
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{p + (m - m) * ppm + i - m, pc + i}] = c_op[m];
                            mat[{p + (i - m) * ppm + m - m, pc + i}] =
                                s ? mc_op[m] : c_op[m];
                        }
                        for (uint16_t i = m + 1; i < n_orbs; i++)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{p + (i - m) * ppm + j - m,
                                     pa + (i - m - 1) * mm + j - m - 1}] = i_op;
                        p += ppm * ppm;
                    }
                    // AD
                    for (uint8_t s = 0; s < 2; s++) {
                        int pad = s ? pad1 : pad0;
                        mat[{p + (m - m) * ppm + m - m, pi}] = ad_op[m][m][s];
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{p + (m - m) * ppm + i - m, pd + i}] =
                                s ? md_op[m] : d_op[m];
                            mat[{p + (i - m) * ppm + m - m, pd + i}] = d_op[m];
                        }
                        for (uint16_t i = m + 1; i < n_orbs; i++)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{p + (i - m) * ppm + j - m,
                                     pad + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += ppm * ppm;
                    }
                    // B
                    for (uint8_t s = 0; s < 2; s++) {
                        int pb = s ? pb1 : pb0;
                        mat[{p + (m - m) * ppm + m - m, pi}] = b_op[m][m][s];
                        for (uint16_t i = m + 1; i < n_orbs; i++) {
                            mat[{p + (m - m) * ppm + i - m, pd + i}] = c_op[m];
                            mat[{p + (i - m) * ppm + m - m, pc + i}] =
                                s ? md_op[m] : d_op[m];
                        }
                        for (uint16_t i = m + 1; i < n_orbs; i++)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                mat[{p + (i - m) * ppm + j - m,
                                     pb + (i - m - 1) * mm + j - m - 1}] = i_op;
                        p += ppm * ppm;
                    }
                    assert(p == mat.m);
                }
                break;
            default:
                assert(false);
                break;
            }
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[pm];
            if (mode != QCTypes::Conventional || pm < trans_l + 1 ||
                pm > trans_r - 1)
                opt->lmat = opt->rmat = pmat;
            else if (!repeat_m)
                opt->rmat = pmat;
            else
                opt->lmat = pmat;
            // operator names
            if (opt->lmat == pmat) {
                shared_ptr<SymbolicRowVector<S>> plop;
                if (pm == n_sites - 1)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (pm != n_sites - 1) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = c_op[j];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = d_op[j];
                    p += m + 1;
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        lop[p + j - (m + 1)] = trd_op[j];
                    p += n_orbs - (m + 1);
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        lop[p + j - (m + 1)] = tr_op[j];
                    p += n_orbs - (m + 1);
                    vector<FL> su2_factor;
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = 0; j < m + 1; j++) {
                                for (uint16_t k = 0; k < m + 1; k++)
                                    lop[p + k] = a_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = 0; j < m + 1; j++) {
                                for (uint16_t k = 0; k < m + 1; k++)
                                    lop[p + k] = ad_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = 0; j < m + 1; j++) {
                                for (uint16_t k = 0; k < m + 1; k++)
                                    lop[p + k] = b_op[j][k][s];
                                p += m + 1;
                            }
                        break;
                    case QCTypes::CN:
                        su2_factor = {(FL)-0.5, (FL)(-0.5 * sqrt(3))};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    lop[p + k - m - 1] =
                                        su2_factor[s] * p_op[j][k][s];
                                p += n_orbs - m - 1;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    lop[p + k - m - 1] =
                                        su2_factor[s] * pd_op[j][k][s];
                                p += n_orbs - m - 1;
                            }
                        su2_factor = {(FL)1.0, (FL)sqrt(3)};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = m + 1; j < n_orbs; j++) {
                                for (uint16_t k = m + 1; k < n_orbs; k++)
                                    lop[p + k - m - 1] =
                                        su2_factor[s] * q_op[j][k][s];
                                p += n_orbs - m - 1;
                            }
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    assert(p == rshape);
                }
                this->left_operator_names[pm] = plop;
            }
            if (opt->rmat == pmat) {
                shared_ptr<SymbolicColumnVector<S>> prop;
                if (pm == 0)
                    prop = make_shared<SymbolicColumnVector<S>>(1);
                else
                    prop = make_shared<SymbolicColumnVector<S>>(lshape);
                SymbolicColumnVector<S> &rop = *prop;
                if (pm == 0)
                    rop[0] = h_op;
                else {
                    rop[0] = i_op;
                    rop[1] = h_op;
                    p = 2;
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = tr_op[j];
                    p += m;
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = trd_op[j];
                    p += m;
                    for (uint16_t j = m; j < n_orbs; j++)
                        rop[p + j - m] = d_op[j];
                    p += n_orbs - m;
                    for (uint16_t j = m; j < n_orbs; j++)
                        rop[p + j - m] = c_op[j];
                    p += n_orbs - m;
                    vector<FL> su2_factor;
                    switch (effective_mode) {
                    case QCTypes::NC:
                        su2_factor = {(FL)-0.5, (FL)(-0.5 * sqrt(3))};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = 0; j < m; j++) {
                                for (uint16_t k = 0; k < m; k++)
                                    rop[p + k] = su2_factor[s] * p_op[j][k][s];
                                p += m;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = 0; j < m; j++) {
                                for (uint16_t k = 0; k < m; k++)
                                    rop[p + k] = su2_factor[s] * pd_op[j][k][s];
                                p += m;
                            }
                        su2_factor = {(FL)1.0, (FL)sqrt(3)};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = 0; j < m; j++) {
                                for (uint16_t k = 0; k < m; k++)
                                    rop[p + k] = su2_factor[s] * q_op[j][k][s];
                                p += m;
                            }
                        break;
                    case QCTypes::CN:
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = m; j < n_orbs; j++) {
                                for (uint16_t k = m; k < n_orbs; k++)
                                    rop[p + k - m] = a_op[j][k][s];
                                p += n_orbs - m;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = m; j < n_orbs; j++) {
                                for (uint16_t k = m; k < n_orbs; k++)
                                    rop[p + k - m] = ad_op[j][k][s];
                                p += n_orbs - m;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint16_t j = m; j < n_orbs; j++) {
                                for (uint16_t k = m; k < n_orbs; k++)
                                    rop[p + k - m] = b_op[j][k][s];
                                p += n_orbs - m;
                            }
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    assert(p == lshape);
                }
                this->right_operator_names[pm] = prop;
            }
            if (mode != QCTypes::Conventional || pm < trans_l + 1 ||
                pm > trans_r - 1) {
                this->save_tensor(pm);
                this->save_left_operators(pm);
                this->save_right_operators(pm);
                this->unload_tensor(pm);
                this->unload_left_operators(pm);
                this->unload_right_operators(pm);
            }
        }
        if (mode == QCTypes::Conventional)
            for (uint16_t pm = trans_l + 1; pm <= trans_r - 1; pm++) {
                this->save_tensor(pm);
                this->save_left_operators(pm);
                this->unload_tensor(pm);
                this->save_right_operators(pm);
                this->unload_left_operators(pm);
                this->unload_right_operators(pm);
            }
        SeqTypes seqt = hamil->opf->seq->mode;
        hamil->opf->seq->mode = SeqTypes::None;
        const uint16_t m_start = hamil->get_n_orbs_left() > 0 ? 1 : 0;
        const uint16_t m_end =
            hamil->get_n_orbs_right() > 0 ? n_sites - 1 : n_sites;
        int ntgb = frame_<FP>()->minimal_memory_usage ? 1 : ntg;
#pragma omp parallel for schedule(dynamic) num_threads(ntgb)
#ifdef _MSC_VER
        for (int m = (int)m_start; m < (int)m_end; m++) {
#else
        for (uint16_t m = m_start; m < m_end; m++) {
#endif
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO build ... " << setw(4) << m << " / " << setw(4)
                     << n_sites << endl;
            this->load_tensor(m);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[m];
            hamil->filter_site_ops((uint16_t)m, {opt->lmat, opt->rmat},
                                   opt->ops);
            this->save_tensor(m);
            this->unload_tensor(m);
        }
        if (hamil->get_n_orbs_left() > 0 && n_sites > 0) {
            this->load_tensor(0);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[0];
            hamil->filter_site_ops(0, {opt->lmat, opt->rmat}, opt->ops);
            this->save_tensor(0);
            this->unload_tensor(0);
        }
        if (hamil->get_n_orbs_right() > 0 && n_sites > 0) {
            this->load_tensor(n_sites - 1);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[n_sites - 1];
            hamil->filter_site_ops(n_sites - 1, {opt->lmat, opt->rmat},
                                   opt->ops);
            this->save_tensor(n_sites - 1);
            this->unload_tensor(n_sites - 1);
        }
        hamil->opf->seq->mode = seqt;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN) ||
            mode == QCTypes::Conventional) {
            uint16_t m, pm;
            MPO<S, FL>::schemer = make_shared<MPOSchemer<S>>(trans_l, trans_r);
            if (frame_<FP>()->minimal_memory_usage)
                cout << "left middle transform ... " << endl;
            // left transform
            pm = trans_l;
            m = pm + n_orbs_big_left - 1;
            int new_rshape =
                2 + 2 * n_orbs + 6 * (n_orbs - m - 1) * (n_orbs - m - 1);
            MPO<S, FL>::schemer->left_new_operator_names =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            MPO<S, FL>::schemer->left_new_operator_exprs =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            SymbolicRowVector<S> &lop =
                *MPO<S, FL>::schemer->left_new_operator_names;
            SymbolicRowVector<S> &lexpr =
                *MPO<S, FL>::schemer->left_new_operator_exprs;
            this->load_left_operators(pm);
            for (int i = 0; i < 2 + 2 * n_orbs; i++)
                lop[i] = this->left_operator_names[pm]->data[i];
            this->unload_left_operators(pm);
            const vector<FL> su2_factor_p = {(FL)-0.5, (FL)(-0.5 * sqrt(3))};
            const vector<FL> su2_factor_q = {(FL)1.0, (FL)sqrt(3)};
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
            for (int sj = 0; sj < (int)(2 * (n_orbs - (m + 1))); sj++) {
                uint8_t s = (uint8_t)(sj / (n_orbs - (m + 1)));
                uint16_t j = (uint16_t)(sj % (n_orbs - (m + 1)) + m + 1);
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg) collapse(2)
            for (uint8_t s = 0; s < 2; s++)
                for (uint16_t j = m + 1; j < n_orbs; j++) {
#endif
                vector<shared_ptr<OpExpr<S>>> exprs;
                exprs.reserve((m + 1) * (m + 1));
                for (uint16_t k = m + 1; k < n_orbs; k++) {
                    int p = (k - m - 1) + (j - m - 1) * (n_orbs - m - 1) +
                            s * (n_orbs - m - 1) * (n_orbs - m - 1);
                    exprs.clear();
                    p += 2 + 2 * n_orbs;
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    (su2_factor_p[s] * hamil->v(j, g, k, h) *
                                     (FL)(s ? -1.0 : 1.0)) *
                                    ad_op[g][h][s]);
                    lop[p] = su2_factor_p[s] * p_op[j][k][s];
                    lexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 2 * (n_orbs - m - 1) * (n_orbs - m - 1);
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    (su2_factor_p[s] * hamil->v(j, g, k, h) *
                                     (FL)(s ? -1.0 : 1.0)) *
                                    a_op[g][h][s]);
                    lop[p] = su2_factor_p[s] * pd_op[j][k][s];
                    lexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 2 * (n_orbs - m - 1) * (n_orbs - m - 1);
                    if (s == 0) {
                        for (uint16_t g = 0; g < m + 1; g++)
                            for (uint16_t h = 0; h < m + 1; h++)
                                if (abs((FL)2.0 * hamil->v(j, k, g, h) -
                                        hamil->v(j, h, g, k)) > TINY)
                                    exprs.push_back(
                                        (su2_factor_q[0] *
                                         ((FL)2.0 * hamil->v(j, k, g, h) -
                                          hamil->v(j, h, g, k))) *
                                        b_op[g][h][0]);
                    } else {
                        for (uint16_t g = 0; g < m + 1; g++)
                            for (uint16_t h = 0; h < m + 1; h++)
                                if (abs(hamil->v(j, h, g, k)) > TINY)
                                    exprs.push_back((su2_factor_q[1] *
                                                     hamil->v(j, h, g, k)) *
                                                    b_op[g][h][1]);
                    }
                    lop[p] = su2_factor_q[s] * q_op[j][k][s];
                    lexpr[p] = sum(exprs);
                }
            }
            if (frame_<FP>()->minimal_memory_usage)
                cout << "right middle transform ... " << endl;
            // right transform
            pm = trans_r - 1;
            m = pm + n_orbs_big_left - 1;
            int new_lshape = 2 + 2 * n_orbs + 6 * (m + 1) * (m + 1);
            MPO<S, FL>::schemer->right_new_operator_names =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            MPO<S, FL>::schemer->right_new_operator_exprs =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            SymbolicColumnVector<S> &rop =
                *MPO<S, FL>::schemer->right_new_operator_names;
            SymbolicColumnVector<S> &rexpr =
                *MPO<S, FL>::schemer->right_new_operator_exprs;
            this->load_right_operators(pm + 1);
            for (int i = 0; i < 2 + 2 * n_orbs; i++)
                rop[i] = this->right_operator_names[pm + 1]->data[i];
            this->unload_right_operators(pm + 1);
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
            for (int sj = 0; sj < (int)(2 * (m + 1)); sj++) {
                uint8_t s = (uint8_t)(sj / (m + 1));
                uint16_t j = (uint16_t)(sj % (m + 1));
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg) collapse(2)
            for (uint8_t s = 0; s < 2; s++)
                for (uint16_t j = 0; j < m + 1; j++) {
#endif
                vector<shared_ptr<OpExpr<S>>> exprs;
                exprs.reserve((n_orbs - m - 1) * (n_orbs - m - 1));
                for (uint16_t k = 0; k < m + 1; k++) {
                    int p = k + j * (m + 1) + s * (m + 1) * (m + 1);
                    exprs.clear();
                    p += 2 + 2 * n_orbs;
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    (su2_factor_p[s] * hamil->v(j, g, k, h) *
                                     (FL)(s ? -1.0 : 1.0)) *
                                    ad_op[g][h][s]);
                    rop[p] = su2_factor_p[s] * p_op[j][k][s];
                    rexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 2 * (m + 1) * (m + 1);
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    (su2_factor_p[s] * hamil->v(j, g, k, h) *
                                     (FL)(s ? -1.0 : 1.0)) *
                                    a_op[g][h][s]);
                    rop[p] = su2_factor_p[s] * pd_op[j][k][s];
                    rexpr[p] = sum(exprs);
                    exprs.clear();
                    p += 2 * (m + 1) * (m + 1);
                    if (s == 0) {
                        for (uint16_t g = m + 1; g < n_orbs; g++)
                            for (uint16_t h = m + 1; h < n_orbs; h++)
                                if (abs((FL)2.0 * hamil->v(j, k, g, h) -
                                        hamil->v(j, h, g, k)) > TINY)
                                    exprs.push_back(
                                        (su2_factor_q[0] *
                                         ((FL)2.0 * hamil->v(j, k, g, h) -
                                          hamil->v(j, h, g, k))) *
                                        b_op[g][h][0]);
                    } else {
                        for (uint16_t g = m + 1; g < n_orbs; g++)
                            for (uint16_t h = m + 1; h < n_orbs; h++)
                                if (abs(hamil->v(j, h, g, k)) > TINY)
                                    exprs.push_back((su2_factor_q[1] *
                                                     hamil->v(j, h, g, k)) *
                                                    b_op[g][h][1]);
                    }
                    rop[p] = su2_factor_q[s] * q_op[j][k][s];
                    rexpr[p] = sum(exprs);
                }
            }
            this->save_schemer();
            this->unload_schemer();
        }
        threading->activate_normal();
    }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            if (this->tensors[m] != nullptr)
                this->tensors[m]->deallocate();
    }
};

// Quantum chemistry MPO (general spin)
template <typename S, typename FL>
struct MPOQC<S, FL, typename S::is_sg_t> : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    QCTypes mode;
    const bool symmetrized_p = true;
    MPOQC(const shared_ptr<HamiltonianQC<S, FL>> &hamil,
          QCTypes mode = QCTypes::NC, const string &tag = "HQC",
          int trans_center = -1, int trans_delta = 1, bool symmetrized_p = true)
        : MPO<S, FL>(hamil->n_sites, tag), mode(mode),
          symmetrized_p(symmetrized_p) {
        // fermionic exchange factor
        const FL exf = S::GIF ? -1.0 : 1.0;
        shared_ptr<OpExpr<S>> h_op = make_shared<OpElement<S, FL>>(
            OpNames::H, SiteIndex(), hamil->vacuum);
        shared_ptr<OpExpr<S>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
#ifdef _MSC_VER
        vector<shared_ptr<OpExpr<S>>> c_op(n_orbs), d_op(n_orbs);
        vector<shared_ptr<OpExpr<S>>> rd_op(n_orbs), r_op(n_orbs);
        vector<shared_ptr<OpExpr<S>>> mrd_op(n_orbs), mr_op(n_orbs);
        vector<vector<shared_ptr<OpExpr<S>>>> a_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(n_orbs));
        vector<vector<shared_ptr<OpExpr<S>>>> ad_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(n_orbs));
        vector<vector<shared_ptr<OpExpr<S>>>> b_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(n_orbs));
        vector<vector<shared_ptr<OpExpr<S>>>> p_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(n_orbs));
        vector<vector<shared_ptr<OpExpr<S>>>> pd_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(n_orbs));
        vector<vector<shared_ptr<OpExpr<S>>>> q_op(
            n_orbs, vector<vector<shared_ptr<OpExpr<S>>>>(n_orbs));
#else
        shared_ptr<OpExpr<S>> c_op[n_orbs], d_op[n_orbs];
        shared_ptr<OpExpr<S>> rd_op[n_orbs], r_op[n_orbs];
        shared_ptr<OpExpr<S>> mrd_op[n_orbs], mr_op[n_orbs];
        shared_ptr<OpExpr<S>> a_op[n_orbs][n_orbs];
        shared_ptr<OpExpr<S>> ad_op[n_orbs][n_orbs];
        shared_ptr<OpExpr<S>> b_op[n_orbs][n_orbs];
        shared_ptr<OpExpr<S>> p_op[n_orbs][n_orbs];
        shared_ptr<OpExpr<S>> pd_op[n_orbs][n_orbs];
        shared_ptr<OpExpr<S>> q_op[n_orbs][n_orbs];
#endif
        MPO<S, FL>::hamil = hamil;
        MPO<S, FL>::op = dynamic_pointer_cast<OpElement<S, FL>>(h_op);
        MPO<S, FL>::left_vacuum = hamil->vacuum;
        MPO<S, FL>::const_e = hamil->e();
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        uint16_t trans_l = -1, trans_r = n_sites;
        if (trans_center == -1)
            trans_center = n_sites >> 1;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN))
            trans_l = trans_center - 1, trans_r = trans_center;
        else if (mode == QCTypes::Conventional)
            trans_l = trans_center - 1, trans_r = trans_center + trans_delta;
        for (uint16_t m = 0; m < n_orbs; m++) {
            c_op[m] = make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex(m),
                                                    S(1, hamil->orb_sym[m]));
            d_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::D, SiteIndex(m), S(-1, S::pg_inv(hamil->orb_sym[m])));
            rd_op[m] = make_shared<OpElement<S, FL>>(OpNames::RD, SiteIndex(m),
                                                     S(1, hamil->orb_sym[m]));
            r_op[m] = make_shared<OpElement<S, FL>>(
                OpNames::R, SiteIndex(m), S(-1, S::pg_inv(hamil->orb_sym[m])));
            mrd_op[m] = exf * rd_op[m];
            mr_op[m] = exf * r_op[m];
        }
        for (uint16_t i = 0; i < n_orbs; i++)
            for (uint16_t j = 0; j < n_orbs; j++) {
                a_op[i][j] = make_shared<OpElement<S, FL>>(
                    OpNames::A, SiteIndex(i, j),
                    S(2, S::pg_mul(hamil->orb_sym[i], hamil->orb_sym[j])));
                ad_op[i][j] = make_shared<OpElement<S, FL>>(
                    OpNames::AD, SiteIndex(i, j),
                    S(-2, S::pg_mul(S::pg_inv(hamil->orb_sym[i]),
                                    S::pg_inv(hamil->orb_sym[j]))));
                b_op[i][j] = make_shared<OpElement<S, FL>>(
                    OpNames::B, SiteIndex(i, j),
                    S(0, S::pg_mul(hamil->orb_sym[i],
                                   S::pg_inv(hamil->orb_sym[j]))));
                p_op[i][j] = make_shared<OpElement<S, FL>>(
                    OpNames::P, SiteIndex(i, j),
                    S(-2, S::pg_mul(S::pg_inv(hamil->orb_sym[i]),
                                    S::pg_inv(hamil->orb_sym[j]))));
                pd_op[i][j] = make_shared<OpElement<S, FL>>(
                    OpNames::PD, SiteIndex(i, j),
                    S(2, S::pg_mul(hamil->orb_sym[i], hamil->orb_sym[j])));
                q_op[i][j] = make_shared<OpElement<S, FL>>(
                    OpNames::Q, SiteIndex(i, j),
                    S(0, S::pg_inv(S::pg_mul(hamil->orb_sym[i],
                                             S::pg_inv(hamil->orb_sym[j])))));
            }
        uint16_t n_repeat_m =
            mode == QCTypes::Conventional && trans_l + 1 >= 0 &&
                    trans_r - 1 < n_sites && trans_l + 1 <= trans_r - 1
                ? 1 + (trans_r - 1) - (trans_l + 1)
                : 0;
        this->left_operator_names.resize(n_sites, nullptr);
        this->right_operator_names.resize(n_sites, nullptr);
        this->tensors.resize(n_sites, nullptr);
        for (uint16_t m = 0; m < n_sites; m++)
            this->tensors[m] = make_shared<OperatorTensor<S, FL>>();
        int ntg = threading->activate_global();
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int xxm = 0; xxm < (int)(n_sites + n_repeat_m); xxm++) {
            uint16_t xm = (uint16_t)xxm;
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (uint16_t xm = 0; xm < n_sites + n_repeat_m; xm++) {
#endif
            uint16_t pm = xm;
            int p;
            bool repeat_m = false;
            if (n_repeat_m && xm > trans_r - 1) {
                pm = xm - n_repeat_m;
                if (pm <= trans_r - 1)
                    repeat_m = true;
            }
            uint16_t m = pm + n_orbs_big_left - 1;
            shared_ptr<Symbolic<S>> pmat;
            int lshape, rshape;
            QCTypes effective_mode;
            if (mode == QCTypes::NC ||
                ((mode & QCTypes::NC) && pm <= trans_l) ||
                (mode == QCTypes::Conventional && pm <= trans_r - 1 &&
                 !repeat_m))
                effective_mode = QCTypes::NC;
            else if (mode == QCTypes::CN ||
                     ((mode & QCTypes::CN) && pm >= trans_r) ||
                     (mode == QCTypes::Conventional && pm >= trans_l + 1))
                effective_mode = QCTypes::CN;
            else
                assert(false);
            switch (effective_mode) {
            case QCTypes::NC:
                lshape = 2 + 2 * n_orbs + 3 * m * m;
                rshape = 2 + 2 * n_orbs + 3 * (m + 1) * (m + 1);
                break;
            case QCTypes::CN:
                lshape = 2 + 2 * n_orbs + 3 * (n_orbs - m) * (n_orbs - m);
                rshape =
                    2 + 2 * n_orbs + 3 * (n_orbs - m - 1) * (n_orbs - m - 1);
                break;
            default:
                assert(false);
            }
            if (pm == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (pm == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (pm == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                p = 2;
                for (uint16_t j = 0; j < m + 1; j++)
                    mat[{0, p + j}] = c_op[j];
                p += m + 1;
                for (uint16_t j = 0; j < m + 1; j++)
                    mat[{0, p + j}] = d_op[j];
                p += m + 1;
                for (uint16_t j = m + 1; j < n_orbs; j++)
                    mat[{0, p + j - m - 1}] = rd_op[j];
                p += n_orbs - (m + 1);
                for (uint16_t j = m + 1; j < n_orbs; j++)
                    mat[{0, p + j - m - 1}] = mr_op[j];
                p += n_orbs - (m + 1);
            } else if (pm == n_sites - 1) {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint16_t j = 0; j < m; j++)
                    mat[{p + j, 0}] = r_op[j];
                p += m;
                for (uint16_t j = 0; j < m; j++)
                    mat[{p + j, 0}] = mrd_op[j];
                p += m;
                for (uint16_t j = m; j < n_orbs; j++)
                    mat[{p + j - m, 0}] = d_op[j];
                p += n_orbs - m;
                for (uint16_t j = m; j < n_orbs; j++)
                    mat[{p + j - m, 0}] = c_op[j];
                p += n_orbs - m;
            }
            switch (effective_mode) {
            case QCTypes::NC:
                if (pm == 0) {
                    for (uint16_t j = 0; j < m + 1; j++) {
                        for (uint16_t k = 0; k < m + 1; k++)
                            mat[{0, p + k}] = a_op[j][k];
                        p += m + 1;
                    }
                    for (uint16_t j = 0; j < m + 1; j++) {
                        for (uint16_t k = 0; k < m + 1; k++)
                            mat[{0, p + k}] = ad_op[j][k];
                        p += m + 1;
                    }
                    for (uint16_t j = 0; j < m + 1; j++) {
                        for (uint16_t k = 0; k < m + 1; k++)
                            mat[{0, p + k}] = b_op[j][k];
                        p += m + 1;
                    }
                    assert(p == mat.n);
                } else {
                    if (pm != n_sites - 1) {
                        mat[{0, 0}] = i_op;
                        mat[{1, 0}] = h_op;
                        p = 2;
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = r_op[j];
                        p += m;
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = mrd_op[j];
                        p += m;
                        mat[{p, 0}] = d_op[m];
                        p += n_orbs - m;
                        mat[{p, 0}] = c_op[m];
                        p += n_orbs - m;
                    }
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p + k, 0}] = (FL)0.5 * p_op[j][k];
                        p += m;
                    }
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p + k, 0}] = (FL)0.5 * pd_op[j][k];
                        p += m;
                    }
                    for (uint16_t j = 0; j < m; j++) {
                        for (uint16_t k = 0; k < m; k++)
                            mat[{p + k, 0}] = q_op[j][k];
                        p += m;
                    }
                    assert(p == mat.m);
                }
                if (pm != 0 && pm != n_sites - 1) {
                    mat[{1, 1}] = i_op;
                    p = 2;
                    // pointers
                    int pi = 1, pc = 2, pd = 2 + m;
                    int prd = 2 + m + m - m, pr = 2 + m + n_orbs - m;
                    int pa = 2 + (n_orbs << 1) + m * m * 0;
                    int pad = 2 + (n_orbs << 1) + m * m * 1;
                    int pb = 2 + (n_orbs << 1) + m * m * 2;
                    // C
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pc + j, p + j}] = i_op;
                    mat[{pi, p + m}] = c_op[m];
                    p += m + 1;
                    // D
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pd + j, p + j}] = i_op;
                    mat[{pi, p + m}] = d_op[m];
                    p += m + 1;
                    // RD
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{prd + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = rd_op[i];
                        for (uint16_t k = 0; k < m; k++) {
                            mat[{pd + k, p + i - (m + 1)}] = pd_op[i][k];
                            mat[{pc + k, p + i - (m + 1)}] = q_op[k][i];
                        }
                        if (!symmetrized_p)
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t l = 0; l < m; l++)
                                    mat[{pa + j * m + l, p + i - (m + 1)}] =
                                        hamil->v(j, i, l, m) * d_op[m];
                        else
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t l = 0; l < m; l++)
                                    mat[{pa + j * m + l, p + i - (m + 1)}] =
                                        ((FL)0.5 *
                                         (hamil->v(j, i, l, m) +
                                          exf * hamil->v(l, i, j, m))) *
                                        d_op[m];
                        for (uint16_t k = 0; k < m; k++)
                            for (uint16_t l = 0; l < m; l++)
                                mat[{pb + l * m + k, p + i - (m + 1)}] =
                                    (hamil->v(m, i, l, k) -
                                     hamil->v(l, i, m, k)) *
                                    c_op[m];
                    }
                    p += n_orbs - (m + 1);
                    // R
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{pr + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = mr_op[i];
                        for (uint16_t k = 0; k < m; k++) {
                            mat[{pc + k, p + i - (m + 1)}] = exf * p_op[i][k];
                            mat[{pd + k, p + i - (m + 1)}] = exf * q_op[i][k];
                        }
                        if (!symmetrized_p)
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t l = 0; l < m; l++)
                                    mat[{pad + j * m + l, p + i - (m + 1)}] =
                                        (exf * hamil->v(i, j, m, l)) * c_op[m];
                        else
                            for (uint16_t j = 0; j < m; j++)
                                for (uint16_t l = 0; l < m; l++)
                                    mat[{pad + j * m + l, p + i - (m + 1)}] =
                                        (exf * (FL)0.5 *
                                         (hamil->v(i, j, m, l) +
                                          exf * hamil->v(i, l, m, j))) *
                                        c_op[m];
                        for (uint16_t k = 0; k < m; k++)
                            for (uint16_t l = 0; l < m; l++)
                                mat[{pb + k * m + l, p + i - (m + 1)}] =
                                    (exf * (hamil->v(i, m, k, l) -
                                            hamil->v(i, l, k, m))) *
                                    d_op[m];
                    }
                    p += n_orbs - (m + 1);
                    // A
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pa + i * m + j, p + i * (m + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pc + i, p + i * (m + 1) + m}] = c_op[m];
                        mat[{pc + i, p + m * (m + 1) + i}] = exf * c_op[m];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = a_op[m][m];
                    p += (m + 1) * (m + 1);
                    // AD
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pad + i * m + j, p + i * (m + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pd + i, p + i * (m + 1) + m}] = exf * d_op[m];
                        mat[{pd + i, p + m * (m + 1) + i}] = d_op[m];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m];
                    p += (m + 1) * (m + 1);
                    // B
                    for (uint16_t i = 0; i < m; i++)
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pb + i * m + j, p + i * (m + 1) + j}] = i_op;
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{pc + i, p + i * (m + 1) + m}] = d_op[m];
                        mat[{pd + i, p + m * (m + 1) + i}] = exf * c_op[m];
                    }
                    mat[{pi, p + m * (m + 1) + m}] = b_op[m][m];
                    p += (m + 1) * (m + 1);
                    assert(p == mat.n);
                }
                break;
            case QCTypes::CN:
                if (pm == n_sites - 1) {
                    for (uint16_t j = m; j < n_orbs; j++) {
                        for (uint16_t k = m; k < n_orbs; k++)
                            mat[{p + k - m, 0}] = a_op[j][k];
                        p += n_orbs - m;
                    }
                    for (uint16_t j = m; j < n_orbs; j++) {
                        for (uint16_t k = m; k < n_orbs; k++)
                            mat[{p + k - m, 0}] = ad_op[j][k];
                        p += n_orbs - m;
                    }
                    for (uint16_t j = m; j < n_orbs; j++) {
                        for (uint16_t k = m; k < n_orbs; k++)
                            mat[{p + k - m, 0}] = b_op[j][k];
                        p += n_orbs - m;
                    }
                    assert(p == mat.m);
                } else {
                    if (pm != 0) {
                        mat[{1, 0}] = h_op;
                        mat[{1, 1}] = i_op;
                        p = 2;
                        mat[{1, p + m}] = c_op[m];
                        p += m + 1;
                        mat[{1, p + m}] = d_op[m];
                        p += m + 1;
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{1, p + j - m - 1}] = rd_op[j];
                        p += n_orbs - m - 1;
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{1, p + j - m - 1}] = mr_op[j];
                        p += n_orbs - m - 1;
                    }
                    for (uint16_t j = m + 1; j < n_orbs; j++) {
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            mat[{!!pm, p + k - m - 1}] = (FL)0.5 * p_op[j][k];
                        p += n_orbs - m - 1;
                    }
                    for (uint16_t j = m + 1; j < n_orbs; j++) {
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            mat[{!!pm, p + k - m - 1}] = (FL)0.5 * pd_op[j][k];
                        p += n_orbs - m - 1;
                    }
                    for (uint16_t j = m + 1; j < n_orbs; j++) {
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            mat[{!!pm, p + k - m - 1}] = q_op[j][k];
                        p += n_orbs - m - 1;
                    }
                    assert(p == mat.n);
                }
                if (pm != 0 && pm != n_sites - 1) {
                    mat[{0, 0}] = i_op;
                    p = 2;
                    // pointers
                    int mm = n_orbs - m - 1;
                    int ppm = n_orbs - m;
                    int pi = 0, pr = 2, prd = 2 + m + 1;
                    int pd = 2 + m + m + 2 - m - 1,
                        pc = 2 + m + 1 + n_orbs - m - 1;
                    int pa = 2 + (n_orbs << 1) + mm * mm * 0;
                    int pad = 2 + (n_orbs << 1) + mm * mm * 1;
                    int pb = 2 + (n_orbs << 1) + mm * mm * 2;
                    // R
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{p + i, pi}] = r_op[i];
                        mat[{p + i, pr + i}] = i_op;
                        if (!symmetrized_p)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                for (uint16_t l = m + 1; l < n_orbs; l++)
                                    mat[{p + i,
                                         pad + (j - m - 1) * mm + l - m - 1}] =
                                        hamil->v(i, j, m, l) * c_op[m];
                        else
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                for (uint16_t l = m + 1; l < n_orbs; l++)
                                    mat[{p + i,
                                         pad + (j - m - 1) * mm + l - m - 1}] =
                                        ((FL)0.5 *
                                         (hamil->v(i, j, m, l) +
                                          exf * hamil->v(i, l, m, j))) *
                                        c_op[m];
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            for (uint16_t l = m + 1; l < n_orbs; l++)
                                mat[{p + i, pb + (k - m - 1) * mm + l - m -
                                                1}] = (hamil->v(i, m, k, l) -
                                                       hamil->v(i, l, k, m)) *
                                                      d_op[m];
                        for (uint16_t k = m + 1; k < n_orbs; k++) {
                            mat[{p + i, pc + k}] = p_op[i][k];
                            mat[{p + i, pd + k}] = q_op[i][k];
                        }
                    }
                    p += m;
                    // RD
                    for (uint16_t i = 0; i < m; i++) {
                        mat[{p + i, pi}] = mrd_op[i];
                        mat[{p + i, prd + i}] = i_op;
                        if (!symmetrized_p)
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                for (uint16_t l = m + 1; l < n_orbs; l++)
                                    mat[{p + i,
                                         pa + (j - m - 1) * mm + l - m - 1}] =
                                        (exf * hamil->v(j, i, l, m)) * d_op[m];
                        else
                            for (uint16_t j = m + 1; j < n_orbs; j++)
                                for (uint16_t l = m + 1; l < n_orbs; l++)
                                    mat[{p + i,
                                         pa + (j - m - 1) * mm + l - m - 1}] =
                                        (exf * (FL)0.5 *
                                         (hamil->v(j, i, l, m) +
                                          exf * hamil->v(l, i, j, m))) *
                                        d_op[m];
                        for (uint16_t k = m + 1; k < n_orbs; k++)
                            for (uint16_t l = m + 1; l < n_orbs; l++)
                                mat[{p + i,
                                     pb + (l - m - 1) * mm + k - m - 1}] =
                                    (exf * (hamil->v(m, i, l, k) -
                                            hamil->v(l, i, m, k))) *
                                    c_op[m];
                        for (uint16_t k = m + 1; k < n_orbs; k++) {
                            mat[{p + i, pd + k}] = exf * pd_op[i][k];
                            mat[{p + i, pc + k}] = exf * q_op[k][i];
                        }
                    }
                    p += m;
                    // D
                    mat[{p + m - m, pi}] = d_op[m];
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        mat[{p + j - m, pd + j}] = i_op;
                    p += n_orbs - m;
                    // C
                    mat[{p + m - m, pi}] = c_op[m];
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        mat[{p + j - m, pc + j}] = i_op;
                    p += n_orbs - m;
                    // A
                    mat[{p + (m - m) * ppm + m - m, pi}] = a_op[m][m];
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{p + (m - m) * ppm + i - m, pc + i}] = c_op[m];
                        mat[{p + (i - m) * ppm + m - m, pc + i}] =
                            exf * c_op[m];
                    }
                    for (uint16_t i = m + 1; i < n_orbs; i++)
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{p + (i - m) * ppm + j - m,
                                 pa + (i - m - 1) * mm + j - m - 1}] = i_op;
                    p += ppm * ppm;
                    // AD
                    mat[{p + (m - m) * ppm + m - m, pi}] = ad_op[m][m];
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{p + (m - m) * ppm + i - m, pd + i}] =
                            exf * d_op[m];
                        mat[{p + (i - m) * ppm + m - m, pd + i}] = d_op[m];
                    }
                    for (uint16_t i = m + 1; i < n_orbs; i++)
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{p + (i - m) * ppm + j - m,
                                 pad + (i - m - 1) * mm + j - m - 1}] = i_op;
                    p += ppm * ppm;
                    // B
                    mat[{p + (m - m) * ppm + m - m, pi}] = b_op[m][m];
                    for (uint16_t i = m + 1; i < n_orbs; i++) {
                        mat[{p + (m - m) * ppm + i - m, pd + i}] = c_op[m];
                        mat[{p + (i - m) * ppm + m - m, pc + i}] =
                            exf * d_op[m];
                    }
                    for (uint16_t i = m + 1; i < n_orbs; i++)
                        for (uint16_t j = m + 1; j < n_orbs; j++)
                            mat[{p + (i - m) * ppm + j - m,
                                 pb + (i - m - 1) * mm + j - m - 1}] = i_op;
                    p += ppm * ppm;
                    assert(p == mat.m);
                }
                break;
            default:
                assert(false);
                break;
            }
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[pm];
            if (mode != QCTypes::Conventional || pm < trans_l + 1 ||
                pm > trans_r - 1)
                opt->lmat = opt->rmat = pmat;
            else if (!repeat_m)
                opt->rmat = pmat;
            else
                opt->lmat = pmat;
            // operator names
            if (opt->lmat == pmat) {
                shared_ptr<SymbolicRowVector<S>> plop;
                if (pm == n_sites - 1)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (pm != n_sites - 1) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = c_op[j];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = d_op[j];
                    p += m + 1;
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        lop[p + j - (m + 1)] = rd_op[j];
                    p += n_orbs - (m + 1);
                    for (uint16_t j = m + 1; j < n_orbs; j++)
                        lop[p + j - (m + 1)] = mr_op[j];
                    p += n_orbs - (m + 1);
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                lop[p + k] = a_op[j][k];
                            p += m + 1;
                        }
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                lop[p + k] = ad_op[j][k];
                            p += m + 1;
                        }
                        for (uint16_t j = 0; j < m + 1; j++) {
                            for (uint16_t k = 0; k < m + 1; k++)
                                lop[p + k] = b_op[j][k];
                            p += m + 1;
                        }
                        break;
                    case QCTypes::CN:
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                lop[p + k - m - 1] = (FL)0.5 * p_op[j][k];
                            p += n_orbs - m - 1;
                        }
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                lop[p + k - m - 1] = (FL)0.5 * pd_op[j][k];
                            p += n_orbs - m - 1;
                        }
                        for (uint16_t j = m + 1; j < n_orbs; j++) {
                            for (uint16_t k = m + 1; k < n_orbs; k++)
                                lop[p + k - m - 1] = q_op[j][k];
                            p += n_orbs - m - 1;
                        }
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    assert(p == rshape);
                }
                this->left_operator_names[pm] = plop;
            }
            if (opt->rmat == pmat) {
                shared_ptr<SymbolicColumnVector<S>> prop;
                if (pm == 0)
                    prop = make_shared<SymbolicColumnVector<S>>(1);
                else
                    prop = make_shared<SymbolicColumnVector<S>>(lshape);
                SymbolicColumnVector<S> &rop = *prop;
                if (pm == 0)
                    rop[0] = h_op;
                else {
                    rop[0] = i_op;
                    rop[1] = h_op;
                    p = 2;
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = r_op[j];
                    p += m;
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = mrd_op[j];
                    p += m;
                    for (uint16_t j = m; j < n_orbs; j++)
                        rop[p + j - m] = d_op[j];
                    p += n_orbs - m;
                    for (uint16_t j = m; j < n_orbs; j++)
                        rop[p + j - m] = c_op[j];
                    p += n_orbs - m;
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                rop[p + k] = (FL)0.5 * p_op[j][k];
                            p += m;
                        }
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                rop[p + k] = (FL)0.5 * pd_op[j][k];
                            p += m;
                        }
                        for (uint16_t j = 0; j < m; j++) {
                            for (uint16_t k = 0; k < m; k++)
                                rop[p + k] = q_op[j][k];
                            p += m;
                        }
                        break;
                    case QCTypes::CN:
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                rop[p + k - m] = a_op[j][k];
                            p += n_orbs - m;
                        }
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                rop[p + k - m] = ad_op[j][k];
                            p += n_orbs - m;
                        }
                        for (uint16_t j = m; j < n_orbs; j++) {
                            for (uint16_t k = m; k < n_orbs; k++)
                                rop[p + k - m] = b_op[j][k];
                            p += n_orbs - m;
                        }
                        break;
                    default:
                        assert(false);
                        break;
                    }
                    assert(p == lshape);
                }
                this->right_operator_names[pm] = prop;
            }
            if (mode != QCTypes::Conventional || pm < trans_l + 1 ||
                pm > trans_r - 1) {
                this->save_tensor(pm);
                this->save_left_operators(pm);
                this->save_right_operators(pm);
                this->unload_tensor(pm);
                this->unload_left_operators(pm);
                this->unload_right_operators(pm);
            }
        }
        if (mode == QCTypes::Conventional)
            for (uint16_t pm = trans_l + 1; pm <= trans_r - 1; pm++) {
                this->save_tensor(pm);
                this->save_left_operators(pm);
                this->save_right_operators(pm);
                this->unload_tensor(pm);
                this->unload_left_operators(pm);
                this->unload_right_operators(pm);
            }
        SeqTypes seqt = hamil->opf->seq->mode;
        hamil->opf->seq->mode = SeqTypes::None;
        const uint16_t m_start = hamil->get_n_orbs_left() > 0 ? 1 : 0;
        const uint16_t m_end =
            hamil->get_n_orbs_right() > 0 ? n_sites - 1 : n_sites;
        int ntgb = frame_<FP>()->minimal_memory_usage ? 1 : ntg;
#pragma omp parallel for schedule(dynamic) num_threads(ntgb)
#ifdef _MSC_VER
        for (int m = (int)m_start; m < (int)m_end; m++) {
#else
        for (uint16_t m = m_start; m < m_end; m++) {
#endif
            if (frame_<FP>()->minimal_memory_usage)
                cout << "MPO build ... " << setw(4) << m << " / " << setw(4)
                     << n_sites << endl;
            this->load_tensor(m);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[m];
            hamil->filter_site_ops((uint16_t)m, {opt->lmat, opt->rmat},
                                   opt->ops);
            this->save_tensor(m);
            this->unload_tensor(m);
        }
        if (hamil->get_n_orbs_left() > 0 && n_sites > 0) {
            this->load_tensor(0);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[0];
            hamil->filter_site_ops(0, {opt->lmat, opt->rmat}, opt->ops);
            this->save_tensor(0);
            this->unload_tensor(0);
        }
        if (hamil->get_n_orbs_right() > 0 && n_sites > 0) {
            this->load_tensor(n_sites - 1);
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[n_sites - 1];
            hamil->filter_site_ops(n_sites - 1, {opt->lmat, opt->rmat},
                                   opt->ops);
            this->save_tensor(n_sites - 1);
            this->unload_tensor(n_sites - 1);
        }
        hamil->opf->seq->mode = seqt;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN) ||
            mode == QCTypes::Conventional) {
            uint16_t m, pm;
            MPO<S, FL>::schemer = make_shared<MPOSchemer<S>>(trans_l, trans_r);
            if (frame_<FP>()->minimal_memory_usage)
                cout << "left middle transform ... " << endl;
            // left transform
            pm = trans_l;
            m = pm + n_orbs_big_left - 1;
            int new_rshape =
                2 + 2 * n_orbs + 3 * (n_orbs - m - 1) * (n_orbs - m - 1);
            MPO<S, FL>::schemer->left_new_operator_names =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            MPO<S, FL>::schemer->left_new_operator_exprs =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            SymbolicRowVector<S> &lop =
                *MPO<S, FL>::schemer->left_new_operator_names;
            SymbolicRowVector<S> &lexpr =
                *MPO<S, FL>::schemer->left_new_operator_exprs;
            this->load_left_operators(pm);
            for (int i = 0; i < 2 + 2 * n_orbs; i++)
                lop[i] = this->left_operator_names[pm]->data[i];
            this->unload_left_operators(pm);
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
#ifdef _MSC_VER
            for (int j = (int)m + 1; j < (int)n_orbs; j++) {
#else
            for (uint16_t j = m + 1; j < n_orbs; j++) {
#endif
                vector<shared_ptr<OpExpr<S>>> exprs;
                exprs.reserve((m + 1) * (m + 1));
                for (uint16_t k = m + 1; k < n_orbs; k++) {
                    int p = (k - m - 1) + (j - m - 1) * (n_orbs - m - 1);
                    exprs.clear();
                    p += 2 + 2 * n_orbs;
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 * hamil->v(j, g, k, h)) *
                                    ad_op[g][h]);
                    lop[p] = (FL)0.5 * p_op[j][k];
                    lexpr[p] = sum(exprs);
                    exprs.clear();
                    p += (n_orbs - m - 1) * (n_orbs - m - 1);
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 * hamil->v(j, g, k, h)) *
                                    a_op[g][h]);
                    lop[p] = (FL)0.5 * pd_op[j][k];
                    lexpr[p] = sum(exprs);
                    exprs.clear();
                    p += (n_orbs - m - 1) * (n_orbs - m - 1);
                    for (uint16_t g = 0; g < m + 1; g++)
                        for (uint16_t h = 0; h < m + 1; h++)
                            if (abs(hamil->v(j, k, g, h) -
                                    hamil->v(j, h, g, k)) > TINY)
                                exprs.push_back((hamil->v(j, k, g, h) -
                                                 hamil->v(j, h, g, k)) *
                                                b_op[g][h]);
                    lop[p] = q_op[j][k];
                    lexpr[p] = sum(exprs);
                }
            }
            if (frame_<FP>()->minimal_memory_usage)
                cout << "right middle transform ... " << endl;
            // right transform
            pm = trans_r - 1;
            m = pm + n_orbs_big_left - 1;
            int new_lshape = 2 + 2 * n_orbs + 3 * (m + 1) * (m + 1);
            MPO<S, FL>::schemer->right_new_operator_names =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            MPO<S, FL>::schemer->right_new_operator_exprs =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            SymbolicColumnVector<S> &rop =
                *MPO<S, FL>::schemer->right_new_operator_names;
            SymbolicColumnVector<S> &rexpr =
                *MPO<S, FL>::schemer->right_new_operator_exprs;
            this->load_right_operators(pm + 1);
            for (int i = 0; i < 2 + 2 * n_orbs; i++)
                rop[i] = this->right_operator_names[pm + 1]->data[i];
            this->unload_right_operators(pm + 1);
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
            for (int j = 0; j < (int)m + 1; j++) {
#else
            for (uint16_t j = 0; j < m + 1; j++) {
#endif
                vector<shared_ptr<OpExpr<S>>> exprs;
                exprs.reserve((n_orbs - m - 1) * (n_orbs - m - 1));
                for (uint16_t k = 0; k < m + 1; k++) {
                    int p = k + j * (m + 1);
                    exprs.clear();
                    p += 2 + 2 * n_orbs;
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 * hamil->v(j, g, k, h)) *
                                    ad_op[g][h]);
                    rop[p] = (FL)0.5 * p_op[j][k];
                    rexpr[p] = sum(exprs);
                    exprs.clear();
                    p += (m + 1) * (m + 1);
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(j, g, k, h)) > TINY)
                                exprs.push_back(
                                    ((FL)0.5 * hamil->v(j, g, k, h)) *
                                    a_op[g][h]);
                    rop[p] = (FL)0.5 * pd_op[j][k];
                    rexpr[p] = sum(exprs);
                    exprs.clear();
                    p += (m + 1) * (m + 1);
                    for (uint16_t g = m + 1; g < n_orbs; g++)
                        for (uint16_t h = m + 1; h < n_orbs; h++)
                            if (abs(hamil->v(j, k, g, h) -
                                    hamil->v(j, h, g, k)) > TINY)
                                exprs.push_back((hamil->v(j, k, g, h) -
                                                 hamil->v(j, h, g, k)) *
                                                b_op[g][h]);
                    rop[p] = q_op[j][k];
                    rexpr[p] = sum(exprs);
                }
            }
            this->save_schemer();
            this->unload_schemer();
        }
        threading->activate_normal();
    }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            if (this->tensors[m] != nullptr)
                this->tensors[m]->deallocate();
    }
};

} // namespace block2
