
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

// Stack two MPOs vertically, not working for SU(2)
template <typename S, typename FL> struct StackedMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::tensors;
    using MPO<S, FL>::site_op_infos;
    using MPO<S, FL>::left_operator_names;
    using MPO<S, FL>::right_operator_names;
    using MPO<S, FL>::middle_operator_names;
    using MPO<S, FL>::middle_operator_exprs;
    using MPO<S, FL>::basis;
    AncillaTypes ancilla_type;
    StackedMPO(const shared_ptr<MPO<S, FL>> &mpoa,
               const shared_ptr<MPO<S, FL>> &mpob, int iprint = 1,
               const string &tag = "")
        : MPO<S, FL>(mpoa->n_sites,
                     tag == "" ? mpoa->tag + "+" + mpob->tag : tag) {
        assert(mpoa->n_sites == mpob->n_sites);
        assert(mpoa->left_operator_exprs.size() == 0);
        assert(mpoa->right_operator_exprs.size() == 0);
        assert(mpob->left_operator_exprs.size() == 0);
        assert(mpob->right_operator_exprs.size() == 0);
        assert(mpoa->const_e == (typename const_fl_type<FL>::FL)0.0 ||
               mpob->const_e == (typename const_fl_type<FL>::FL)0.0);
        assert(mpoa->schemer == nullptr && mpob->schemer == nullptr);
        MPO<S, FL>::const_e = mpoa->const_e + mpob->const_e;
        MPO<S, FL>::op = mpoa->op;
        MPO<S, FL>::left_vacuum = (mpoa->left_vacuum + mpob->left_vacuum)[0];
        MPO<S, FL>::tf = mpoa->tf;
        assert(mpoa->get_ancilla_type() == mpob->get_ancilla_type());
        assert(mpoa->sparse_form == mpob->sparse_form);
        ancilla_type = mpoa->get_ancilla_type();
        MPO<S, FL>::sparse_form = mpoa->sparse_form;
        basis = mpoa->basis;
        if (mpoa->hamil == nullptr)
            MPO<S, FL>::hamil = mpob->hamil;
        else if (mpob->hamil == nullptr)
            MPO<S, FL>::hamil = mpoa->hamil;
        else if (mpoa->hamil == mpob->hamil)
            MPO<S, FL>::hamil = mpoa->hamil;
        else
            assert(false);
        tensors = vector<shared_ptr<OperatorTensor<S, FL>>>(n_sites);
        site_op_infos =
            vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(n_sites);
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<map<S, shared_ptr<SparseMatrixInfo<S>>>> site_op_infos_mp =
            vector<map<S, shared_ptr<SparseMatrixInfo<S>>>>(n_sites);
        for (uint16_t m = 0; m < n_sites; m++) {
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> infoa =
                mpoa->site_op_infos[m];
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> infob =
                mpob->site_op_infos[m];
            map<S, shared_ptr<SparseMatrixInfo<S>>> &info = site_op_infos_mp[m];
            for (const auto &xa : infoa)
                for (const auto &xb : infob) {
                    S q = xa.first + xb.first;
                    for (int iq = 0; iq < q.count(); iq++)
                        info[q[iq]] = nullptr;
                }
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        left_operator_names.resize(n_sites);
        right_operator_names.resize(n_sites);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        const shared_ptr<OpElement<S, FL>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(),
            (MPO<S, FL>::left_vacuum - MPO<S, FL>::left_vacuum)[0]);
        SeqTypes seqt = MPO<S, FL>::tf->opf->seq->mode;
        MPO<S, FL>::tf->opf->seq->mode = SeqTypes::None;
        if (iprint) {
            cout << endl;
            cout << "Stack MPO | Nsites = " << setw(5) << n_sites << endl;
        }
        int ntg = threading->activate_global();
        Timer _t;
        double ttotal = 0;
        int bond_max = 0;
        for (uint16_t m = 0; m < n_sites; m++) {
            _t.get_time();
            if (iprint) {
                cout << " Site = " << setw(5) << m << " / " << setw(5)
                     << n_sites << " .. ";
                cout.flush();
            }
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            mpoa->load_tensor(m);
            mpoa->load_left_operators(m);
            mpoa->load_right_operators(m);
            mpob->load_tensor(m);
            mpob->load_left_operators(m);
            mpob->load_right_operators(m);
            const int xm =
                mpoa->tensors[m]->lmat->m * mpob->tensors[m]->lmat->m;
            const int xn =
                mpoa->tensors[m]->lmat->n * mpob->tensors[m]->lmat->n;
            left_operator_names[m] = make_shared<SymbolicRowVector<S>>(xn);
            right_operator_names[m] = make_shared<SymbolicColumnVector<S>>(xm);
            for (int ima = 0; ima < mpoa->left_operator_names[m]->n; ima++)
                for (int imb = 0; imb < mpob->left_operator_names[m]->n;
                     imb++) {
                    const int imx = ima * mpob->left_operator_names[m]->n + imb;
                    const auto opxa = dynamic_pointer_cast<OpElement<S, FL>>(
                        mpoa->left_operator_names[m]->data[ima]);
                    const auto opxb = dynamic_pointer_cast<OpElement<S, FL>>(
                        mpob->left_operator_names[m]->data[imb]);
                    left_operator_names[m]->data[imx] =
                        make_shared<OpElement<S, FL>>(
                            OpNames::XL,
                            SiteIndex({(uint16_t)(imx / 1000 / 1000),
                                       (uint16_t)(imx / 1000 % 1000),
                                       (uint16_t)(imx % 1000)},
                                      {}),
                            opxa->q_label + opxb->q_label,
                            opxa->factor * opxb->factor);
                }
            for (int ima = 0; ima < mpoa->right_operator_names[m]->m; ima++)
                for (int imb = 0; imb < mpob->right_operator_names[m]->m;
                     imb++) {
                    const int imx =
                        ima * mpob->right_operator_names[m]->m + imb;
                    const auto opxa = dynamic_pointer_cast<OpElement<S, FL>>(
                        mpoa->right_operator_names[m]->data[ima]);
                    const auto opxb = dynamic_pointer_cast<OpElement<S, FL>>(
                        mpob->right_operator_names[m]->data[imb]);
                    right_operator_names[m]->data[imx] =
                        make_shared<OpElement<S, FL>>(
                            OpNames::XR,
                            SiteIndex({(uint16_t)(imx / 1000 / 1000),
                                       (uint16_t)(imx / 1000 % 1000),
                                       (uint16_t)(imx % 1000)},
                                      {}),
                            opxa->q_label + opxb->q_label,
                            opxa->factor * opxb->factor);
                }
            if (m == 0) {
                shared_ptr<SymbolicRowVector<S>> pmata =
                    dynamic_pointer_cast<SymbolicRowVector<S>>(
                        mpoa->tensors[m]->lmat);
                shared_ptr<SymbolicRowVector<S>> pmatb =
                    dynamic_pointer_cast<SymbolicRowVector<S>>(
                        mpob->tensors[m]->lmat);
                shared_ptr<SymbolicRowVector<S>> pmat =
                    make_shared<SymbolicRowVector<S>>(xn);
                for (int ima = 0; ima < pmata->n; ima++)
                    for (int imb = 0; imb < pmatb->n; imb++) {
                        const int imx = ima * pmatb->n + imb;
                        if (pmata->data[ima]->get_type() == OpTypes::Zero)
                            pmat->data[imx] = zero;
                        else if (pmatb->data[imb]->get_type() == OpTypes::Zero)
                            pmat->data[imx] = zero;
                        else {
                            shared_ptr<OpElement<S, FL>> opel =
                                make_shared<OpElement<S, FL>>(
                                    OpNames::XL,
                                    SiteIndex({(uint16_t)(imx / 1000 / 1000),
                                               (uint16_t)(imx / 1000 % 1000),
                                               (uint16_t)(imx % 1000)},
                                              {}),
                                    dynamic_pointer_cast<OpElement<S, FL>>(
                                        pmata->data[ima])
                                            ->q_label +
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            pmatb->data[imb])
                                            ->q_label);
                            shared_ptr<SparseMatrix<S, FL>> mata =
                                mpoa->tensors[m]->ops.at(
                                    abs_value(pmata->data[ima]));
                            shared_ptr<SparseMatrix<S, FL>> matb =
                                mpob->tensors[m]->ops.at(
                                    abs_value(pmatb->data[imb]));
                            shared_ptr<SparseMatrix<S, FL>> xmat =
                                make_shared<SparseMatrix<S, FL>>(d_alloc);
                            const S q = (mata->info->delta_quantum +
                                         matb->info->delta_quantum)[0];
                            const FL phase =
                                mata->info->delta_quantum.is_fermion() &&
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            mpob->right_operator_names[m]
                                                ->data[0])
                                            ->q_label.is_fermion()
                                    ? -1
                                    : 1;
                            xmat->allocate(site_op_infos_mp[m].at(q));
                            MPO<S, FL>::tf->opf->product(0, mata, matb, xmat,
                                                         phase);
                            pmat->data[imx] = opel->scalar_multiply(
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    pmata->data[ima])
                                    ->factor *
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    pmatb->data[imb])
                                    ->factor);
                            opt->ops[opel] = xmat;
                        }
                    }
                opt->lmat = opt->rmat = pmat;
            } else if (m == n_sites - 1) {
                shared_ptr<SymbolicColumnVector<S>> pmata =
                    dynamic_pointer_cast<SymbolicColumnVector<S>>(
                        mpoa->tensors[m]->lmat);
                shared_ptr<SymbolicColumnVector<S>> pmatb =
                    dynamic_pointer_cast<SymbolicColumnVector<S>>(
                        mpob->tensors[m]->lmat);
                shared_ptr<SymbolicColumnVector<S>> pmat =
                    make_shared<SymbolicColumnVector<S>>(xm);
                for (int ima = 0; ima < pmata->m; ima++)
                    for (int imb = 0; imb < pmatb->m; imb++) {
                        const int imx = ima * pmatb->m + imb;
                        if (pmata->data[ima]->get_type() == OpTypes::Zero)
                            pmat->data[imx] = zero;
                        else if (pmatb->data[imb]->get_type() == OpTypes::Zero)
                            pmat->data[imx] = zero;
                        else {
                            shared_ptr<OpElement<S, FL>> opel =
                                make_shared<OpElement<S, FL>>(
                                    OpNames::XR,
                                    SiteIndex({(uint16_t)(imx / 1000 / 1000),
                                               (uint16_t)(imx / 1000 % 1000),
                                               (uint16_t)(imx % 1000)},
                                              {}),
                                    dynamic_pointer_cast<OpElement<S, FL>>(
                                        pmata->data[ima])
                                            ->q_label +
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            pmatb->data[imb])
                                            ->q_label);
                            shared_ptr<SparseMatrix<S, FL>> mata =
                                mpoa->tensors[m]->ops.at(
                                    abs_value(pmata->data[ima]));
                            shared_ptr<SparseMatrix<S, FL>> matb =
                                mpob->tensors[m]->ops.at(
                                    abs_value(pmatb->data[imb]));
                            shared_ptr<SparseMatrix<S, FL>> xmat =
                                make_shared<SparseMatrix<S, FL>>(d_alloc);
                            const S q = (mata->info->delta_quantum +
                                         matb->info->delta_quantum)[0];
                            const FL phase =
                                mata->info->delta_quantum.is_fermion() &&
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            mpob->right_operator_names[m]
                                                ->data[imb])
                                            ->q_label.is_fermion()
                                    ? -1
                                    : 1;
                            xmat->allocate(site_op_infos_mp[m].at(q));
                            MPO<S, FL>::tf->opf->product(0, mata, matb, xmat,
                                                         phase);
                            pmat->data[imx] = opel->scalar_multiply(
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    pmata->data[ima])
                                    ->factor *
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    pmatb->data[imb])
                                    ->factor);
                            opt->ops[opel] = xmat;
                        }
                    }
                opt->lmat = opt->rmat = pmat;
            } else {
                shared_ptr<SymbolicMatrix<S>> pmata =
                    dynamic_pointer_cast<SymbolicMatrix<S>>(
                        mpoa->tensors[m]->lmat);
                shared_ptr<SymbolicMatrix<S>> pmatb =
                    dynamic_pointer_cast<SymbolicMatrix<S>>(
                        mpob->tensors[m]->lmat);
                shared_ptr<SymbolicMatrix<S>> pmat =
                    make_shared<SymbolicMatrix<S>>(xm, xn);
                pmat->indices.resize(pmata->indices.size() *
                                     pmatb->indices.size());
                pmat->data.resize(pmata->indices.size() *
                                  pmatb->indices.size());
                vector<shared_ptr<SparseMatrix<S, FL>>> mats(
                    pmata->indices.size() * pmatb->indices.size());
                const size_t nimx = pmat->data.size(),
                             nimx_pt = (nimx + ntg - 1) / ntg;
#pragma omp parallel num_threads(ntg)
                {
                    shared_ptr<VectorAllocator<FP>> xd_alloc =
                        make_shared<VectorAllocator<FP>>();
                    const int tid = threading->get_thread_id();
                    const size_t imx_st = tid * nimx_pt,
                                 imx_ed = min(nimx, (tid + 1) * nimx_pt);
                    for (size_t imx = imx_st; imx < imx_ed; imx++) {
                        const size_t ima = imx / pmatb->indices.size(),
                                     imb = imx % pmatb->indices.size();
                        pmat->indices[imx] =
                            make_pair(pmata->indices[ima].first * pmatb->m +
                                          pmatb->indices[imb].first,
                                      pmata->indices[ima].second * pmatb->n +
                                          pmatb->indices[imb].second);
                        if (pmata->data[ima]->get_type() == OpTypes::Zero ||
                            pmatb->data[imb]->get_type() == OpTypes::Zero) {
                            pmat->data[imx] = zero;
                            continue;
                        }
                        vector<shared_ptr<OpElement<S, FL>>> ppas;
                        vector<shared_ptr<OpElement<S, FL>>> ppbs;
                        if (pmata->data[ima]->get_type() == OpTypes::Elem)
                            ppas.push_back(
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    pmata->data[ima]));
                        else if (pmata->data[ima]->get_type() == OpTypes::Sum) {
                            const auto p = dynamic_pointer_cast<OpSum<S, FL>>(
                                pmata->data[ima]);
                            for (size_t j = 0; j < p->strings.size(); j++)
                                ppas.push_back(
                                    dynamic_pointer_cast<OpElement<S, FL>>(
                                        p->strings[j]
                                            ->get_op()
                                            ->scalar_multiply(
                                                p->strings[j]->factor)));
                        } else
                            assert(false);
                        if (pmatb->data[imb]->get_type() == OpTypes::Elem)
                            ppbs.push_back(
                                dynamic_pointer_cast<OpElement<S, FL>>(
                                    pmatb->data[imb]));
                        else if (pmatb->data[imb]->get_type() == OpTypes::Sum) {
                            const auto p = dynamic_pointer_cast<OpSum<S, FL>>(
                                pmatb->data[imb]);
                            for (size_t j = 0; j < p->strings.size(); j++)
                                ppbs.push_back(
                                    dynamic_pointer_cast<OpElement<S, FL>>(
                                        p->strings[j]
                                            ->get_op()
                                            ->scalar_multiply(
                                                p->strings[j]->factor)));
                        } else
                            assert(false);
                        shared_ptr<OpElement<S, FL>> opel =
                            make_shared<OpElement<S, FL>>(
                                OpNames::X,
                                SiteIndex({(uint16_t)(imx / 1000 / 1000 / 1000),
                                           (uint16_t)(imx / 1000 / 1000 % 1000),
                                           (uint16_t)(imx / 1000 % 1000),
                                           (uint16_t)(imx % 1000)},
                                          {}),
                                ppas[0]->q_label + ppbs[0]->q_label);
                        shared_ptr<SparseMatrix<S, FL>> xmat =
                            make_shared<SparseMatrix<S, FL>>(xd_alloc);
                        const S q = (ppas[0]->q_label + ppbs[0]->q_label)[0];
                        xmat->allocate(site_op_infos_mp[m].at(q));
                        for (const auto xpa : ppas)
                            for (const auto xpb : ppbs) {
                                shared_ptr<SparseMatrix<S, FL>> mata =
                                    mpoa->tensors[m]->ops.at(
                                        abs_value((shared_ptr<OpExpr<S>>)xpa));
                                shared_ptr<SparseMatrix<S, FL>> matb =
                                    mpob->tensors[m]->ops.at(
                                        abs_value((shared_ptr<OpExpr<S>>)xpb));
                                const FL phase =
                                    mata->info->delta_quantum.is_fermion() &&
                                            dynamic_pointer_cast<
                                                OpElement<S, FL>>(
                                                mpob->right_operator_names[m]
                                                    ->data[pmatb->indices[imb]
                                                               .first])
                                                ->q_label.is_fermion()
                                        ? -1
                                        : 1;
                                MPO<S, FL>::tf->opf->product(
                                    0, mata, matb, xmat,
                                    phase * xpa->factor * xpb->factor);
                            }
                        pmat->data[imx] = opel;
                        mats[imx] = xmat;
                    }
                }
                opt->ops.reserve(nimx);
                for (size_t imx = 0; imx < nimx; imx++)
                    opt->ops[abs_value(pmat->data[imx])] = mats[imx];
                opt->lmat = opt->rmat = pmat;
            }
            if (mpoa->tensors[m]->ops.count(i_op) &&
                mpob->tensors[m]->ops.count(i_op)) {
                shared_ptr<SparseMatrix<S, FL>> mata =
                    mpoa->tensors[m]->ops.at(i_op);
                shared_ptr<SparseMatrix<S, FL>> matb =
                    mpob->tensors[m]->ops.at(i_op);
                shared_ptr<SparseMatrix<S, FL>> xmat =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
                const S q =
                    (mata->info->delta_quantum + matb->info->delta_quantum)[0];
                xmat->allocate(site_op_infos_mp[m].at(q));
                MPO<S, FL>::tf->opf->product(0, mata, matb, xmat, (FL)1.0);
                opt->ops[i_op] = xmat;
            }
            tensors[m] = opt;
            double tsite = _t.get_time();
            bond_max = max(xn, bond_max);
            ttotal += tsite;
            if (iprint)
                cout << "Mmpo = " << setw(10) << xn << " T = " << fixed
                     << setprecision(3) << tsite << endl;
            mpoa->unload_tensor(m);
            mpoa->unload_left_operators(m);
            mpoa->unload_right_operators(m);
            mpob->unload_tensor(m);
            mpob->unload_left_operators(m);
            mpob->unload_right_operators(m);
            this->save_tensor(m);
            this->unload_tensor(m);
            this->save_left_operators(m);
            this->unload_left_operators(m);
            this->save_right_operators(m);
            this->unload_right_operators(m);
        }
        if (iprint) {
            cout << "Ttotal = " << fixed << setprecision(3) << setw(10)
                 << ttotal;
            cout << " MPO bond dimension = " << setw(10) << bond_max;
            cout << endl << endl;
        }
        MPO<S, FL>::tf->opf->seq->mode = seqt;
        threading->activate_normal();
    }
    AncillaTypes get_ancilla_type() const override { return ancilla_type; }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            if (this->tensors[m] != nullptr)
                this->tensors[m]->deallocate();
    }
};

// Merge every two adjacent sites into one site
// Not working for SU(2)
template <typename S, typename FL> struct CondensedMPO : MPO<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    using MPO<S, FL>::n_sites;
    using MPO<S, FL>::tensors;
    using MPO<S, FL>::site_op_infos;
    using MPO<S, FL>::left_operator_names;
    using MPO<S, FL>::right_operator_names;
    using MPO<S, FL>::middle_operator_names;
    using MPO<S, FL>::middle_operator_exprs;
    using MPO<S, FL>::basis;
    AncillaTypes ancilla_type;
    CondensedMPO(const shared_ptr<MPO<S, FL>> &mpo,
                 const vector<shared_ptr<StateInfo<S>>> &orig_basis,
                 bool is_pdm = false, const string &tag = "")
        : MPO<S, FL>(mpo->n_sites / 2, tag == "" ? mpo->tag : tag) {
        assert(mpo->n_sites % 2 == 0);
        assert(mpo->n_sites == orig_basis.size());
        assert(mpo->left_operator_exprs.size() == 0);
        assert(mpo->right_operator_exprs.size() == 0);
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::left_vacuum = mpo->left_vacuum;
        MPO<S, FL>::npdm_scheme = mpo->npdm_scheme;
        MPO<S, FL>::npdm_parallel_center = mpo->npdm_parallel_center;
        shared_ptr<OpExpr<S>> zero_op = make_shared<OpExpr<S>>();
        if (mpo->schemer == nullptr)
            MPO<S, FL>::schemer = nullptr;
        else {
            mpo->load_schemer();
            MPO<S, FL>::schemer =
                frame_<FP>()->minimal_memory_usage
                    ? make_shared<MPOSchemer<S>>(*mpo->schemer)
                    : mpo->schemer->copy();
            mpo->unload_schemer();
        }
        MPO<S, FL>::tf = mpo->tf;
        ancilla_type = mpo->get_ancilla_type();
        stringstream new_sparse_form;
        for (uint16_t m = 0; m < n_sites; m++)
            new_sparse_form << (mpo->sparse_form[m + m] == 'N' &&
                                        mpo->sparse_form[m + m + 1] == 'N'
                                    ? 'N'
                                    : 'S');
        MPO<S, FL>::sparse_form = new_sparse_form.str();
        basis = MPSInfo<S>::condense_basis(orig_basis);
        tensors = vector<shared_ptr<OperatorTensor<S, FL>>>(n_sites);
        site_op_infos =
            vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(n_sites);
        left_operator_names.resize(n_sites);
        right_operator_names.resize(n_sites);
        if (is_pdm) {
            assert(mpo->middle_operator_names.size() != 0);
            middle_operator_names.resize(n_sites - 1);
            middle_operator_exprs.resize(n_sites - 1);
        }
        int prevm = 0, lastm = 0;
        for (uint16_t m = 0; m < n_sites; m++) {
            shared_ptr<OperatorTensor<S, FL>> opt =
                make_shared<OperatorTensor<S, FL>>();
            mpo->load_tensor(m + m);
            mpo->load_tensor(m + m + 1);
            mpo->load_left_operators(m + m + 1);
            mpo->load_right_operators(m + m);
            bool gl = mpo->tensors[m + m]->lmat == mpo->tensors[m + m]->rmat;
            bool gr =
                mpo->tensors[m + m + 1]->lmat == mpo->tensors[m + m + 1]->rmat;
            assert(gl == gr);
            left_operator_names[m] = mpo->left_operator_names[m + m + 1];
            right_operator_names[m] = mpo->right_operator_names[m + m];
            opt->lmat =
                mpo->tensors[m + m]->lmat * mpo->tensors[m + m + 1]->lmat;
            int ixx = 0;
            shared_ptr<OpExpr<S>> iop = make_shared<OpExpr<S>>();
            map<S, shared_ptr<SparseMatrixInfo<S>>> mik;
            unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<OpExpr<S>>> midop;
            size_t nx_mat_size = opt->lmat->data.size();
            if (is_pdm) {
                mpo->load_left_operators(m + m);
                unordered_map<shared_ptr<OpExpr<S>>, size_t> rmmap;
                vector<int> xrm(mpo->left_operator_names[m + m]->data.size(),
                                -1);
                int g = (int)mpo->left_operator_names[m + m + 1]->data.size();
                for (size_t i = 0;
                     i < mpo->left_operator_names[m + m + 1]->data.size(); i++)
                    rmmap[mpo->left_operator_names[m + m + 1]->data[i]] = i;
                left_operator_names[m] = left_operator_names[m]->copy();
                for (size_t i = 0;
                     i < mpo->left_operator_names[m + m]->data.size(); i++)
                    if (!rmmap.count(
                            mpo->left_operator_names[m + m]->data[i])) {
                        xrm[i] = g++;
                        left_operator_names[m]->data.push_back(
                            mpo->left_operator_names[m + m]->data[i]);
                    }
                iop = mpo->left_operator_names[m + m]->data[0];
                assert((dynamic_pointer_cast<OpElement<S, FL>>(iop))->name ==
                       OpNames::I);
                if (opt->lmat->get_type() != SymTypes::CVec)
                    opt->lmat->n = g;
                left_operator_names[m]->n = g;
                nx_mat_size = opt->lmat->data.size();
                if (mpo->tensors[m + m]->lmat->get_type() == SymTypes::Mat) {
                    shared_ptr<SymbolicMatrix<S>> xlmat =
                        dynamic_pointer_cast<SymbolicMatrix<S>>(
                            mpo->tensors[m + m]->lmat);
                    if (opt->lmat->get_type() == SymTypes::Mat) {
                        for (size_t i = 0; i < xlmat->data.size(); i++)
                            if (xrm[xlmat->indices[i].second] != -1) {
                                opt->lmat->data.push_back(xlmat->data[i] * iop);
                                if (xlmat->indices[i].first == 0)
                                    midop[left_operator_names[m]->data
                                              [xrm[xlmat->indices[i].second]]] =
                                        xlmat->data[i];
                                dynamic_pointer_cast<SymbolicMatrix<S>>(
                                    opt->lmat)
                                    ->indices.push_back(make_pair(
                                        xlmat->indices[i].first,
                                        xrm[xlmat->indices[i].second]));
                            }
                    } else {
                        for (size_t i = 0; i < xlmat->data.size(); i++)
                            if (xrm[xlmat->indices[i].second] != -1)
                                if (xlmat->indices[i].first == 0)
                                    midop[left_operator_names[m]->data
                                              [xrm[xlmat->indices[i].second]]] =
                                        xlmat->data[i];
                    }
                } else if (mpo->tensors[m + m]->lmat->get_type() ==
                           SymTypes::RVec) {
                    for (size_t i = 0;
                         i < mpo->tensors[m + m]->lmat->data.size(); i++)
                        if (xrm[i] != -1)
                            opt->lmat->data.push_back(
                                mpo->tensors[m + m]->lmat->data[i] * iop);
                } else
                    assert(false);
                mpo->unload_left_operators(m + m);
                if (m != n_sites - 1) {
                    mpo->load_middle_operators(m + m);
                    mpo->load_middle_operators(m + m + 1);
                    middle_operator_names[m] =
                        mpo->middle_operator_names[m + m]->copy();
                    middle_operator_names[m]->data.insert(
                        middle_operator_names[m]->data.end(),
                        mpo->middle_operator_names[m + m + 1]->data.begin(),
                        mpo->middle_operator_names[m + m + 1]->data.end());
                    middle_operator_exprs[m] =
                        mpo->middle_operator_exprs[m + m]->copy();
                    middle_operator_exprs[m]->data.insert(
                        middle_operator_exprs[m]->data.end(),
                        mpo->middle_operator_exprs[m + m + 1]->data.begin(),
                        mpo->middle_operator_exprs[m + m + 1]->data.end());
                    middle_operator_names[m]->m =
                        (int)middle_operator_names[m]->data.size();
                    middle_operator_exprs[m]->m =
                        (int)middle_operator_exprs[m]->data.size();
                    for (size_t i = 0;
                         i < mpo->middle_operator_exprs[m + m]->data.size();
                         i++)
                        if (middle_operator_exprs[m]->data[i]->get_type() ==
                            OpTypes::Prod) {
                            shared_ptr<OpProduct<S, FL>> xop =
                                dynamic_pointer_cast<OpProduct<S, FL>>(
                                    middle_operator_exprs[m]->data[i]);
                            if (xop->b->name != OpNames::I &&
                                mpo->tensors[m + m + 1]->ops.count(xop->b)) {
                                if (!mpo->tensors[m + m]->ops.count(xop->a))
                                    xop->a =
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            midop.at(xop->a));
                                opt->lmat->data.push_back(xop);
                                if (opt->lmat->get_type() == SymTypes::Mat)
                                    dynamic_pointer_cast<SymbolicMatrix<S>>(
                                        opt->lmat)
                                        ->indices.push_back(make_pair(0, g++));
                                shared_ptr<OpElement<S, FL>> kk =
                                    infer_expr(xop);
                                assert(kk != nullptr);
                                left_operator_names[m]->data.push_back(kk);
                                middle_operator_exprs[m]->data[i] =
                                    (shared_ptr<OpExpr<S>>)kk * iop;
                            }
                        }
                    opt->lmat->n = g;
                    left_operator_names[m]->n = g;
                    mpo->unload_middle_operators(m + m + 1);
                    mpo->unload_middle_operators(m + m);
                    this->save_middle_operators(m);
                    this->unload_middle_operators(m);
                } else
                    left_operator_names[m]->data.resize(1);
                opt->lmat->m = max(opt->lmat->m, prevm);
                if (opt->lmat->get_type() == SymTypes::CVec)
                    opt->lmat->data.resize(opt->lmat->m, zero_op);
                prevm = opt->lmat->n;
            }
            shared_ptr<OpExpr<S>> iiop = iop * iop;
            for (size_t i = 0; i < opt->lmat->data.size(); i++) {
                shared_ptr<SparseMatrix<S, FL>> xmat = evaluate_expr(
                    opt->lmat->data[i], mpo->tensors[m + m]->ops,
                    mpo->tensors[m + m + 1]->ops, basis[m], orig_basis[m + m],
                    orig_basis[m + m + 1], mpo->tf->opf);
                if (xmat != nullptr) {
                    shared_ptr<OpElement<S, FL>> kk;
                    if (m == 0) {
                        kk = dynamic_pointer_cast<OpElement<S, FL>>(
                            left_operator_names[m]->data[i]);
                        xmat->iscale((FL)1.0 / kk->factor);
                    } else if (i > nx_mat_size) {
                        kk = dynamic_pointer_cast<OpElement<S, FL>>(
                            left_operator_names[m]
                                ->data[dynamic_pointer_cast<SymbolicMatrix<S>>(
                                           opt->lmat)
                                           ->indices[i]
                                           .second]);
                        xmat->iscale((FL)1.0 / kk->factor);
                    } else if (m == n_sites - 1) {
                        kk = dynamic_pointer_cast<OpElement<S, FL>>(
                            right_operator_names[m]->data[i]);
                        xmat->iscale((FL)1.0 / kk->factor);
                    } else if ((kk = infer_expr(opt->lmat->data[i])) !=
                               nullptr) {
                        xmat->iscale((FL)1.0 / kk->factor);
                    } else {
                        kk = make_shared<OpElement<S, FL>>(
                            OpNames::X,
                            SiteIndex({(uint16_t)(ixx / 1000),
                                       (uint16_t)(ixx % 1000),
                                       (uint16_t)n_sites},
                                      {}),
                            xmat->info->delta_quantum);
                        ixx++;
                    }
                    opt->ops[abs_value((shared_ptr<OpExpr<S>>)kk)] = xmat;
                    opt->lmat->data[i] = kk;
                    mik[xmat->info->delta_quantum] = xmat->info;
                }
            }
            if (gl)
                opt->rmat = opt->lmat;
            else {
                opt->rmat =
                    mpo->tensors[m + m]->rmat * mpo->tensors[m + m + 1]->rmat;
                if (is_pdm && m == n_sites - 1) {
                    mpo->load_right_operators(m + m + 1);
                    unordered_map<shared_ptr<OpExpr<S>>, size_t> rmmap;
                    int g = (int)mpo->right_operator_names[m + m]->data.size();
                    for (size_t i = 0;
                         i < mpo->right_operator_names[m + m]->data.size(); i++)
                        rmmap[mpo->right_operator_names[m + m]->data[i]] = i;
                    right_operator_names[m] = right_operator_names[m]->copy();
                    for (size_t i = 0;
                         i < mpo->right_operator_names[m + m + 1]->data.size();
                         i++)
                        if (!rmmap.count(mpo->right_operator_names[m + m + 1]
                                             ->data[i])) {
                            right_operator_names[m]->data.push_back(
                                mpo->right_operator_names[m + m + 1]->data[i]);
                            opt->rmat->data.push_back(
                                iop * mpo->tensors[m + m + 1]->rmat->data[i]);
                            g++;
                        }
                    mpo->unload_right_operators(m + m + 1);
                    mpo->load_middle_operators(m + m);
                    this->load_middle_operators(m - 1);
                    for (size_t i = 0;
                         i < mpo->middle_operator_names[m + m]->data.size();
                         i++) {
                        middle_operator_names[m - 1]->data.push_back(
                            mpo->middle_operator_names[m + m]->data[i]);
                        middle_operator_exprs[m - 1]->data.push_back(
                            mpo->middle_operator_exprs[m + m]->data[i]);
                        if (middle_operator_exprs[m - 1]
                                ->data.back()
                                ->get_type() == OpTypes::Prod) {
                            shared_ptr<OpProduct<S, FL>> xop =
                                dynamic_pointer_cast<OpProduct<S, FL>>(
                                    middle_operator_exprs[m - 1]->data.back());
                            if (xop->a->name != OpNames::I) {
                                if (!mpo->tensors[m + m]->ops.count(xop->a))
                                    xop->a =
                                        dynamic_pointer_cast<OpElement<S, FL>>(
                                            midop.at(xop->a));
                                shared_ptr<OpElement<S, FL>> kk =
                                    infer_expr(xop);
                                assert(kk != nullptr);
                                right_operator_names[m]->data.push_back(kk);
                                opt->rmat->data.push_back(xop);
                                g++;
                                middle_operator_exprs[m - 1]->data.back() =
                                    iop * (shared_ptr<OpExpr<S>>)kk;
                            }
                        }
                    }
                    mpo->unload_middle_operators(m + m);
                    opt->rmat->m = g;
                    right_operator_names[m]->m = g;
                    lastm = g;
                    middle_operator_names[m - 1]->m =
                        (int)middle_operator_names[m - 1]->data.size();
                    middle_operator_exprs[m - 1]->m =
                        (int)middle_operator_exprs[m - 1]->data.size();
                    this->save_middle_operators(m - 1);
                    this->unload_middle_operators(m - 1);
                }
                for (size_t i = 0; i < opt->rmat->data.size(); i++) {
                    shared_ptr<SparseMatrix<S, FL>> xmat = evaluate_expr(
                        opt->rmat->data[i], mpo->tensors[m + m]->ops,
                        mpo->tensors[m + m + 1]->ops, basis[m],
                        orig_basis[m + m], orig_basis[m + m + 1], mpo->tf->opf);
                    if (xmat != nullptr) {
                        shared_ptr<OpElement<S, FL>> kk;
                        if (m == 0) {
                            kk = dynamic_pointer_cast<OpElement<S, FL>>(
                                left_operator_names[m]->data[i]);
                            xmat->iscale((FL)1.0 / kk->factor);
                        } else if (m == n_sites - 1) {
                            kk = dynamic_pointer_cast<OpElement<S, FL>>(
                                right_operator_names[m]->data[i]);
                            xmat->iscale((FL)1.0 / kk->factor);
                        } else if ((kk = infer_expr(opt->rmat->data[i])) !=
                                   nullptr) {
                            xmat->iscale((FL)1.0 / kk->factor);
                        } else {
                            kk = make_shared<OpElement<S, FL>>(
                                OpNames::X,
                                SiteIndex({(uint16_t)(ixx / 1000),
                                           (uint16_t)(ixx % 1000),
                                           (uint16_t)n_sites},
                                          {}),
                                xmat->info->delta_quantum);
                            ixx++;
                        }
                        opt->ops[abs_value((shared_ptr<OpExpr<S>>)kk)] = xmat;
                        opt->rmat->data[i] = kk;
                        mik[xmat->info->delta_quantum] = xmat->info;
                    }
                }
            }
            tensors[m] = opt;
            vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> xop_infos;
            for (auto &mkk : mik)
                xop_infos.push_back(make_pair(mkk.first, mkk.second));
            site_op_infos[m] = xop_infos;
            mpo->unload_left_operators(m + m + 1);
            mpo->unload_right_operators(m + m);
            mpo->unload_tensor(m + m + 1);
            mpo->unload_tensor(m + m);
            this->save_tensor(m);
            this->unload_tensor(m);
            this->save_left_operators(m);
            this->unload_left_operators(m);
            this->save_right_operators(m);
            this->unload_right_operators(m);
        }
        if (is_pdm) {
            this->load_tensor(n_sites - 2);
            tensors[n_sites - 2]->rmat->n = lastm;
            this->save_tensor(n_sites - 2);
            this->unload_tensor(n_sites - 2);
        }
        if (this->schemer != nullptr) {
            assert(this->schemer->left_trans_site % 2 == 1);
            assert(this->schemer->right_trans_site % 2 == 0);
            this->schemer->left_trans_site /= 2;
            this->schemer->right_trans_site /= 2;
        }
        this->save_schemer();
        this->unload_schemer();
    }
    static shared_ptr<OpElement<S, FL>>
    infer_expr(const shared_ptr<OpExpr<S>> &expr) {
        if (expr->get_type() == OpTypes::Zero)
            return nullptr;
        else if (expr->get_type() == OpTypes::Prod) {
            shared_ptr<OpProduct<S, FL>> xop =
                dynamic_pointer_cast<OpProduct<S, FL>>(expr);
            FL f = xop->factor;
            if (xop->a->name == OpNames::I && xop->b->name == OpNames::I)
                return dynamic_pointer_cast<OpElement<S, FL>>(
                    xop->a->scalar_multiply(f));
            else if (xop->a->name == OpNames::I)
                return dynamic_pointer_cast<OpElement<S, FL>>(
                    xop->b->scalar_multiply(f));
            else if (xop->b->name == OpNames::I)
                return dynamic_pointer_cast<OpElement<S, FL>>(
                    xop->a->scalar_multiply(f));
            else if (xop->a->name == OpNames::C && xop->b->name == OpNames::D)
                return make_shared<OpElement<S, FL>>(
                    OpNames::B,
                    SiteIndex(
                        {xop->a->site_index[0], xop->b->site_index[0]},
                        {xop->a->site_index.s(0), xop->b->site_index.s(0)}),
                    xop->a->q_label + xop->b->q_label, f);
            else if (xop->a->name == OpNames::D && xop->b->name == OpNames::C)
                return make_shared<OpElement<S, FL>>(
                    OpNames::BD,
                    SiteIndex(
                        {xop->a->site_index[0], xop->b->site_index[0]},
                        {xop->a->site_index.s(0), xop->b->site_index.s(0)}),
                    xop->a->q_label + xop->b->q_label, f);
            else if (xop->a->name == OpNames::C && xop->b->name == OpNames::C)
                return make_shared<OpElement<S, FL>>(
                    OpNames::A,
                    SiteIndex(
                        {xop->a->site_index[0], xop->b->site_index[0]},
                        {xop->a->site_index.s(0), xop->b->site_index.s(0)}),
                    xop->a->q_label + xop->b->q_label, f);
            else if (xop->a->name == OpNames::D && xop->b->name == OpNames::D)
                return make_shared<OpElement<S, FL>>(
                    OpNames::AD,
                    SiteIndex(
                        {xop->b->site_index[0], xop->a->site_index[0]},
                        {xop->b->site_index.s(0), xop->a->site_index.s(0)}),
                    xop->a->q_label + xop->b->q_label, f);
            else
                return nullptr;
        } else if (expr->get_type() == OpTypes::Sum) {
            shared_ptr<OpSum<S, FL>> xop =
                dynamic_pointer_cast<OpSum<S, FL>>(expr);
            shared_ptr<OpElement<S, FL>> xr = nullptr;
            for (auto &xxop : xop->strings)
                if (xxop->a->name == OpNames::I)
                    xr = dynamic_pointer_cast<OpElement<S, FL>>(
                        xxop->b->scalar_multiply(xxop->factor));
                else if (xxop->b->name == OpNames::I)
                    xr = dynamic_pointer_cast<OpElement<S, FL>>(
                        xxop->a->scalar_multiply(xxop->factor));
            if (xr != nullptr &&
                (xr->name == OpNames::R || xr->name == OpNames::RD ||
                 xr->name == OpNames::H || xr->name == OpNames::Q))
                return xr;
            else
                return nullptr;
        }
        return nullptr;
    }
    static shared_ptr<SparseMatrix<S, FL>>
    evaluate_expr(const shared_ptr<OpExpr<S>> &expr,
                  const unordered_map<shared_ptr<OpExpr<S>>,
                                      shared_ptr<SparseMatrix<S, FL>>> &opl,
                  const unordered_map<shared_ptr<OpExpr<S>>,
                                      shared_ptr<SparseMatrix<S, FL>>> &opr,
                  const shared_ptr<StateInfo<S>> &basis,
                  const shared_ptr<StateInfo<S>> &basis_a,
                  const shared_ptr<StateInfo<S>> &basis_b,
                  const shared_ptr<OperatorFunctions<S, FL>> &opf) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<OpSum<S, FL>> xexpr;
        if (expr->get_type() == OpTypes::Zero)
            return nullptr;
        else if (expr->get_type() == OpTypes::Prod)
            xexpr =
                make_shared<OpSum<S, FL>>(vector<shared_ptr<OpProduct<S, FL>>>{
                    dynamic_pointer_cast<OpProduct<S, FL>>(expr)});
        else if (expr->get_type() == OpTypes::Sum)
            xexpr = dynamic_pointer_cast<OpSum<S, FL>>(expr);
        else
            assert(false);
        assert(xexpr->strings[0]->conj == 0);
        S dq = xexpr->strings[0]->a->q_label + xexpr->strings[0]->b->q_label;
        vector<pair<uint8_t, S>> subsl;
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> ainfos;
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> binfos;
        map<S, shared_ptr<OpElement<S, FL>>> lqs, rqs;
        for (auto &op : xexpr->strings) {
            S bra, ket;
            if (op->a != nullptr && op->b != nullptr) {
                bra = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                ket = (op->conj & 2) ? op->b->q_label : -op->b->q_label;
            } else
                assert(false);
            S p = dq.combine(bra, ket);
            // here possible error can be due to non-zero (small)
            // integral element violating point group symmetry
            assert(p != S(S::invalid));
            subsl.push_back(make_pair(op->conj, p));
            lqs[op->a->q_label] = op->a;
            rqs[op->b->q_label] = op->b;
        }
        sort(subsl.begin(), subsl.end());
        subsl.resize(
            distance(subsl.begin(), unique(subsl.begin(), subsl.end())));
        for (auto &q : lqs)
            ainfos.push_back(make_pair(q.first, opl.at(q.second)->info));
        for (auto &q : rqs)
            binfos.push_back(make_pair(q.first, opr.at(q.second)->info));
        shared_ptr<SparseMatrixInfo<S>> info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        info->initialize(*basis, *basis, dq, dq.is_fermion());
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
        shared_ptr<typename StateInfo<S>::ConnectionInfo> bas_cinfo =
            StateInfo<S>::get_connection_info(*basis_a, *basis_b, *basis);
        cinfo->initialize_tp(dq, subsl, *basis, *basis, *basis_a, *basis_b,
                             *basis_a, *basis_b, bas_cinfo, bas_cinfo, ainfos,
                             binfos, info, opf->cg);
        info->cinfo = cinfo;
        shared_ptr<SparseMatrix<S, FL>> spmat =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        spmat->allocate(info);
        SeqTypes mode_bak = opf->seq->mode;
        opf->seq->mode = SeqTypes::None;
        for (auto &op : xexpr->strings)
            opf->tensor_product(op->conj, opl.at(op->a), opr.at(op->b), spmat,
                                op->factor);
        cinfo->deallocate();
        opf->seq->mode = mode_bak;
        return spmat;
    }
    AncillaTypes get_ancilla_type() const override { return ancilla_type; }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            if (this->tensors[m] != nullptr)
                this->tensors[m]->deallocate();
    }
};

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
        MPO<S, FL>::left_vacuum = mpo->left_vacuum;
        MPO<S, FL>::npdm_scheme = mpo->npdm_scheme;
        MPO<S, FL>::npdm_parallel_center = mpo->npdm_parallel_center;
        if (mpo->schemer == nullptr)
            MPO<S, FL>::schemer = nullptr;
        else {
            mpo->load_schemer();
            MPO<S, FL>::schemer =
                frame_<FP>()->minimal_memory_usage
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
            fused_basis =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *basis[a], *basis[b], S(S::invalid)));
        else
            fused_basis = make_shared<StateInfo<S>>(
                StateInfo<S>::tensor_product(*basis[a], *basis[b], *ref));
        shared_ptr<typename StateInfo<S>::ConnectionInfo> fused_cinfo =
            StateInfo<S>::get_connection_info(*basis[a], *basis[b],
                                              *fused_basis);
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
            Partition<S, FL>::get_uniq_sub_labels(
                fused_mat, mats[0], sl,
                fused_mat->m == 1 ? mpo->left_vacuum
                                  : (mpo->left_vacuum - mpo->left_vacuum)[0]);
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
                                 fused_cinfo, fused_cinfo,
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
        // but when frame_<FP>()->use_main_stack == false:
        // tf->left/right_contract will skip allocated matrices if alloc !=
        // nullptr
        for (auto &p : opt->ops) {
            shared_ptr<OpElement<S, FL>> op =
                dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            if (frame_<FP>()->use_main_stack)
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
