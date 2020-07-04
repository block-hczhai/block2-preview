
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

#include "expr.hpp"
#include "hamiltonian.hpp"
#include "integral.hpp"
#include "sparse_matrix.hpp"
#include <map>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct HamiltonianQC;

// Quantum chemistry Hamiltonian (non-spin-adapted)
template <typename S>
struct HamiltonianQC<S, typename S::is_sz_t> : Hamiltonian<S> {
    map<OpNames, shared_ptr<SparseMatrix<S>>> op_prims[6];
    shared_ptr<FCIDUMP> fcidump;
    double mu = 0;
    HamiltonianQC(S vacuum, int n_sites, const vector<uint8_t> &orb_sym,
                  const shared_ptr<FCIDUMP> &fcidump)
        : Hamiltonian<S>(vacuum, n_sites, orb_sym), fcidump(fcidump) {
        for (int i = 0; i < this->n_syms; i++) {
            this->basis[i].allocate(4);
            this->basis[i].quanta[0] = this->vacuum;
            this->basis[i].quanta[1] = S(1, 1, i);
            this->basis[i].quanta[2] = S(1, -1, i);
            this->basis[i].quanta[3] = S(2, 0, 0);
            this->basis[i].n_states[0] = this->basis[i].n_states[1] =
                this->basis[i].n_states[2] = this->basis[i].n_states[3] = 1;
            this->basis[i].sort_states();
        }
        init_site_ops();
    }
    void init_site_ops() {
        // site operator infos
        for (int i = 0; i < this->n_syms; i++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vacuum] = nullptr;
            for (int n = -1; n <= 1; n += 2)
                for (int s = -1; s <= 1; s += 2)
                    info[S(n, s, i)] = nullptr;
            for (int n = -2; n <= 2; n += 2)
                for (int s = -2; s <= 2; s += 2)
                    info[S(n, s, 0)] = nullptr;
            this->site_op_infos[i] =
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                 info.end());
            for (auto &p : this->site_op_infos[i]) {
                p.second = make_shared<SparseMatrixInfo<S>>();
                p.second->initialize(this->basis[i], this->basis[i], p.first,
                                     p.first.is_fermion());
            }
        }
        op_prims[0][OpNames::I] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::I]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::I])[S(0, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, -1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(2, 0, 0)](0, 0) = 1.0;
        const int sz[2] = {1, -1};
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s][OpNames::N] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::N]->allocate(
                this->find_site_op_info(S(0, 0, 0), 0));
            (*op_prims[s][OpNames::N])[S(0, 0, 0)](0, 0) = 0.0;
            (*op_prims[s][OpNames::N])[S(1, -1, 0)](0, 0) = s;
            (*op_prims[s][OpNames::N])[S(1, 1, 0)](0, 0) = 1 - s;
            (*op_prims[s][OpNames::N])[S(2, 0, 0)](0, 0) = 1.0;
            op_prims[s][OpNames::C] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::C]->allocate(
                this->find_site_op_info(S(1, sz[s], 0), 0));
            (*op_prims[s][OpNames::C])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims[s][OpNames::C])[S(1, -sz[s], 0)](0, 0) = s ? -1.0 : 1.0;
            op_prims[s][OpNames::D] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::D]->allocate(
                this->find_site_op_info(S(-1, -sz[s], 0), 0));
            (*op_prims[s][OpNames::D])[S(1, sz[s], 0)](0, 0) = 1.0;
            (*op_prims[s][OpNames::D])[S(2, 0, 0)](0, 0) = s ? -1.0 : 1.0;
        }
        // low (&1): left index, high (>>1): right index
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t s = 0; s < 4; s++) {
            op_prims[s][OpNames::NN] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::NN]->allocate(
                this->find_site_op_info(S(0, 0, 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::N],
                               *op_prims[s >> 1][OpNames::N],
                               *op_prims[s][OpNames::NN]);
            op_prims[s][OpNames::A] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::A]->allocate(
                this->find_site_op_info(S(2, sz_plus[s], 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::C],
                               *op_prims[s >> 1][OpNames::C],
                               *op_prims[s][OpNames::A]);
            op_prims[s][OpNames::AD] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::AD]->allocate(
                this->find_site_op_info(S(-2, -sz_plus[s], 0), 0));
            this->opf->product(*op_prims[s >> 1][OpNames::D],
                               *op_prims[s & 1][OpNames::D],
                               *op_prims[s][OpNames::AD]);
            op_prims[s][OpNames::B] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::B]->allocate(
                this->find_site_op_info(S(0, sz_minus[s], 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::C],
                               *op_prims[s >> 1][OpNames::D],
                               *op_prims[s][OpNames::B]);
        }
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s + 4][OpNames::NN] = make_shared<SparseMatrix<S>>();
            op_prims[s + 4][OpNames::NN]->allocate(
                this->find_site_op_info(S(0, 0, 0), 0));
            this->opf->product(*op_prims[s | ((!s) << 1)][OpNames::B],
                               *op_prims[(!s) | (s << 1)][OpNames::B],
                               *op_prims[s + 4][OpNames::NN]);
        }
        // low (&1): R index, high (>>1): B index
        for (uint8_t s = 0; s < 4; s++) {
            op_prims[s][OpNames::R] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::R]->allocate(
                this->find_site_op_info(S(-1, -sz[s & 1], 0), 0));
            this->opf->product(*op_prims[(s >> 1) | (s & 2)][OpNames::B],
                               *op_prims[s & 1][OpNames::D],
                               *op_prims[s][OpNames::R]);
            op_prims[s][OpNames::RD] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::RD]->allocate(
                this->find_site_op_info(S(1, sz[s & 1], 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::C],
                               *op_prims[(s >> 1) | (s & 2)][OpNames::B],
                               *op_prims[s][OpNames::RD]);
        }
        // site norm operators
        map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
            ops[this->n_syms];
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), this->vacuum);
        const shared_ptr<OpElement<S>> n_op[2] = {
            make_shared<OpElement<S>>(OpNames::N, SiteIndex({}, {0}),
                                      this->vacuum),
            make_shared<OpElement<S>>(OpNames::N, SiteIndex({}, {1}),
                                      this->vacuum)};
        for (uint8_t i = 0; i < this->n_syms; i++) {
            ops[i][i_op] = nullptr;
            for (uint8_t s = 0; s < 2; s++)
                ops[i][n_op[s]] = nullptr;
        }
        for (uint8_t m = 0; m < this->n_sites; m++) {
            for (uint8_t s = 0; s < 2; s++) {
                ops[this->orb_sym[m]]
                   [make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], this->orb_sym[m]))] =
                       nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], this->orb_sym[m]))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::NN,
                    SiteIndex({m, m},
                              {(uint8_t)(s & 1), (uint8_t)0, (uint8_t)1}),
                    S(0, 0, 0))] = nullptr;
            }
            for (uint8_t s = 0; s < 4; s++) {
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::A,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(2, sz_plus[s], 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::AD,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(-2, -sz_plus[s], 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::B,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, sz_minus[s], 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::NN,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, 0, 0))] = nullptr;
            }
        }
        for (uint8_t i = 0; i < this->n_syms; i++) {
            this->site_norm_ops[i] = vector<
                pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>(
                ops[i].begin(), ops[i].end());
            for (auto &p : this->site_norm_ops[i]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(this->find_site_op_info(op.q_label, i),
                                   op_prims[op.site_index.ss()][op.name]->data);
            }
        }
    }
    void get_site_ops(uint8_t m,
                      map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                          op_expr_less<S>> &ops) const override {
        uint8_t i, j, k, s;
        shared_ptr<SparseMatrix<S>> zero = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            switch (op.name) {
            case OpNames::I:
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
                p.second = this->find_site_norm_op(p.first, this->orb_sym[m]);
                break;
            case OpNames::H:
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(
                    this->find_site_op_info(op.q_label, this->orb_sym[m]));
                (*p.second)[S(0, 0, 0)](0, 0) = 0.0;
                (*p.second)[S(1, -1, this->orb_sym[m])](0, 0) = t(1, m, m);
                (*p.second)[S(1, 1, this->orb_sym[m])](0, 0) = t(0, m, m);
                (*p.second)[S(2, 0, 0)](0, 0) =
                    t(0, m, m) + t(1, m, m) +
                    0.5 * (v(0, 1, m, m, m, m) + v(1, 0, m, m, m, m));
                break;
            case OpNames::R:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[s].at(OpNames::D));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            *op_prims[s + (sp << 1)].at(OpNames::R));
                        tmp->factor = v(s, sp, i, m, m, m);
                        this->opf->iadd(*p.second, *tmp);
                        if (this->opf->seq->mode != SeqTypes::None)
                            this->opf->seq->simple_perform();
                    }
                    tmp->deallocate();
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[s].at(OpNames::C));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            *op_prims[s + (sp << 1)].at(OpNames::RD));
                        tmp->factor = v(s, sp, i, m, m, m);
                        this->opf->iadd(*p.second, *tmp);
                        if (this->opf->seq->mode != SeqTypes::None)
                            this->opf->seq->simple_perform();
                    }
                    tmp->deallocate();
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::AD)->data);
                    p.second->factor *= v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::A)->data);
                    p.second->factor *= v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::Q:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.ss();
                switch (s) {
                case 0U:
                case 3U:
                    if (abs(v(s & 1, s >> 1, i, m, m, j)) < TINY &&
                        abs(v(s & 1, 0, i, j, m, m)) < TINY &&
                        abs(v(s & 1, 1, i, j, m, m)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                            op.q_label, this->orb_sym[m]));
                        p.second->copy_data_from(
                            *op_prims[(s >> 1) | ((s & 1) << 1)].at(
                                OpNames::B));
                        p.second->factor *= -v(s & 1, s >> 1, i, m, m, j);
                        tmp->allocate(this->find_site_op_info(
                            op.q_label, this->orb_sym[m]));
                        for (uint8_t sp = 0; sp < 2; sp++) {
                            tmp->copy_data_from(
                                *op_prims[sp | (sp << 1)].at(OpNames::B));
                            tmp->factor = v(s & 1, sp, i, j, m, m);
                            this->opf->iadd(*p.second, *tmp);
                            if (this->opf->seq->mode != SeqTypes::None)
                                this->opf->seq->simple_perform();
                        }
                        tmp->deallocate();
                    }
                    break;
                case 1U:
                case 2U:
                    if (abs(v(s & 1, s >> 1, i, m, m, j)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                                               op.q_label, this->orb_sym[m]),
                                           op_prims[(s >> 1) | ((s & 1) << 1)]
                                               .at(OpNames::B)
                                               ->data);
                        p.second->factor *= -v(s & 1, s >> 1, i, m, m, j);
                    }
                    break;
                }
                break;
            default:
                assert(false);
            }
        }
    }
    void deallocate() override {
        for (int8_t s = 3; s >= 0; s--)
            for (auto name : vector<OpNames>{OpNames::RD, OpNames::R})
                op_prims[s][name]->deallocate();
        for (int8_t s = 5; s >= 4; s--)
            for (auto name : vector<OpNames>{OpNames::NN})
                op_prims[s][name]->deallocate();
        for (int8_t s = 3; s >= 0; s--)
            for (auto name : vector<OpNames>{OpNames::B, OpNames::AD,
                                             OpNames::A, OpNames::NN})
                op_prims[s][name]->deallocate();
        for (int8_t s = 1; s >= 0; s--)
            for (auto name :
                 vector<OpNames>{OpNames::D, OpNames::C, OpNames::N})
                op_prims[s][name]->deallocate();
        op_prims[0][OpNames::I]->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            for (int j = this->site_op_infos[i].size() - 1; j >= 0; j--)
                this->site_op_infos[i][j].second->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            this->basis[i].deallocate();
        Hamiltonian<S>::deallocate();
    }
    double v(uint8_t sl, uint8_t sr, uint8_t i, uint8_t j, uint8_t k,
             uint8_t l) const {
        return fcidump->v(sl, sr, i, j, k, l);
    }
    double t(uint8_t s, uint8_t i, uint8_t j) const {
        return i == j ? fcidump->t(s, i, i) - mu : fcidump->t(s, i, j);
    }
    double e() const { return fcidump->e; }
};

// Quantum chemistry Hamiltonian (spin-adapted)
template <typename S>
struct HamiltonianQC<S, typename S::is_su2_t> : Hamiltonian<S> {
    map<OpNames, shared_ptr<SparseMatrix<S>>> op_prims[2];
    shared_ptr<FCIDUMP> fcidump;
    double mu = 0;
    HamiltonianQC(S vacuum, int n_sites, const vector<uint8_t> &orb_sym,
                  const shared_ptr<FCIDUMP> &fcidump)
        : Hamiltonian<S>(vacuum, n_sites, orb_sym), fcidump(fcidump) {
        assert(!fcidump->uhf);
        for (int i = 0; i < this->n_syms; i++) {
            this->basis[i].allocate(3);
            this->basis[i].quanta[0] = vacuum;
            this->basis[i].quanta[1] = S(1, 1, i);
            this->basis[i].quanta[2] = S(2, 0, 0);
            this->basis[i].n_states[0] = this->basis[i].n_states[1] =
                this->basis[i].n_states[2] = 1;
            this->basis[i].sort_states();
        }
        init_site_ops();
    }
    void init_site_ops() {
        // site operator infos
        for (int i = 0; i < this->n_syms; i++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vacuum] = nullptr;
            info[S(1, 1, i)] = nullptr;
            info[S(-1, 1, i)] = nullptr;
            for (int n = -2; n <= 2; n += 2)
                for (int s = 0; s <= 2; s += 2)
                    info[S(n, s, 0)] = nullptr;
            this->site_op_infos[i] =
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                 info.end());
            for (auto &p : this->site_op_infos[i]) {
                p.second = make_shared<SparseMatrixInfo<S>>();
                p.second->initialize(this->basis[i], this->basis[i], p.first,
                                     p.first.is_fermion());
            }
        }
        op_prims[0][OpNames::I] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::I]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::I])[S(0, 0, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, 1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(2, 0, 0, 0)](0, 0) = 1.0;
        op_prims[0][OpNames::N] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::N]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::N])[S(0, 0, 0, 0)](0, 0) = 0.0;
        (*op_prims[0][OpNames::N])[S(1, 1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::N])[S(2, 0, 0, 0)](0, 0) = 2.0;
        // NN[0] = (sum_{sigma} ad_{p,sigma} a_{p,sigma}) ^ 2
        op_prims[0][OpNames::NN] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::NN]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::NN])[S(0, 0, 0, 0)](0, 0) = 0.0;
        (*op_prims[0][OpNames::NN])[S(1, 1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::NN])[S(2, 0, 0, 0)](0, 0) = 4.0;
        op_prims[0][OpNames::C] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::C]->allocate(
            this->find_site_op_info(S(1, 1, 0), 0));
        (*op_prims[0][OpNames::C])[S(0, 1, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::C])[S(1, 0, 1, 0)](0, 0) = -sqrt(2);
        op_prims[0][OpNames::D] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::D]->allocate(
            this->find_site_op_info(S(-1, 1, 0), 0));
        (*op_prims[0][OpNames::D])[S(1, 0, 1, 0)](0, 0) = sqrt(2);
        (*op_prims[0][OpNames::D])[S(2, 1, 0, 0)](0, 0) = 1.0;
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s][OpNames::A] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::A]->allocate(
                this->find_site_op_info(S(2, s * 2, 0), 0));
            this->opf->product(*op_prims[0][OpNames::C],
                               *op_prims[0][OpNames::C],
                               *op_prims[s][OpNames::A]);
            op_prims[s][OpNames::AD] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::AD]->allocate(
                this->find_site_op_info(S(-2, s * 2, 0), 0));
            this->opf->product(*op_prims[0][OpNames::D],
                               *op_prims[0][OpNames::D],
                               *op_prims[s][OpNames::AD]);
            op_prims[s][OpNames::B] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::B]->allocate(
                this->find_site_op_info(S(0, s * 2, 0), 0));
            this->opf->product(*op_prims[0][OpNames::C],
                               *op_prims[0][OpNames::D],
                               *op_prims[s][OpNames::B]);
        }
        // NN[1] = sum_{sigma,tau} ad_{p,sigma} a_{p,tau} ad_{p,tau} a_{p,sigma}
        // = -sqrt(3) B1 x B1 + B0 x B0 where B0 x B0 == 0.5 NN
        op_prims[1][OpNames::NN] = make_shared<SparseMatrix<S>>();
        op_prims[1][OpNames::NN]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        this->opf->product(*op_prims[1][OpNames::B], *op_prims[1][OpNames::B],
                           *op_prims[1][OpNames::NN], -sqrt(3.0));
        this->opf->iadd(*op_prims[1][OpNames::NN], *op_prims[0][OpNames::NN],
                        0.5);
        if (this->opf->seq->mode != SeqTypes::None)
            this->opf->seq->simple_perform();
        op_prims[0][OpNames::R] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::R]->allocate(
            this->find_site_op_info(S(-1, 1, 0), 0));
        this->opf->product(*op_prims[0][OpNames::B], *op_prims[0][OpNames::D],
                           *op_prims[0][OpNames::R]);
        op_prims[0][OpNames::RD] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::RD]->allocate(
            this->find_site_op_info(S(1, 1, 0), 0));
        this->opf->product(*op_prims[0][OpNames::C], *op_prims[0][OpNames::B],
                           *op_prims[0][OpNames::RD]);
        // site norm operators
        map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
            ops[this->n_syms];
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), this->vacuum);
        const shared_ptr<OpElement<S>> n_op =
            make_shared<OpElement<S>>(OpNames::N, SiteIndex(), this->vacuum);
        for (uint8_t i = 0; i < this->n_syms; i++) {
            ops[i][i_op] = nullptr;
            ops[i][n_op] = nullptr;
        }
        for (uint8_t m = 0; m < this->n_sites; m++) {
            ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                OpNames::C, SiteIndex(m), S(1, 1, this->orb_sym[m]))] = nullptr;
            ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                OpNames::D, SiteIndex(m), S(-1, 1, this->orb_sym[m]))] =
                nullptr;
            for (uint8_t s = 0; s < 2; s++) {
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::A, SiteIndex(m, m, s), S(2, s * 2, 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::AD, SiteIndex(m, m, s), S(-2, s * 2, 0))] =
                    nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(m, m, s), S(0, s * 2, 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::NN, SiteIndex(m, m, s), this->vacuum)] = nullptr;
            }
        }
        for (uint8_t i = 0; i < this->n_syms; i++) {
            this->site_norm_ops[i] = vector<
                pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>(
                ops[i].begin(), ops[i].end());
            for (auto &p : this->site_norm_ops[i]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(this->find_site_op_info(op.q_label, i),
                                   op_prims[op.site_index.ss()][op.name]->data);
            }
        }
    }
    void get_site_ops(uint8_t m,
                      map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                          op_expr_less<S>> &ops) const override {
        uint8_t i, j, k, s;
        shared_ptr<SparseMatrix<S>> zero = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            switch (op.name) {
            case OpNames::I:
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
                p.second = this->find_site_norm_op(p.first, this->orb_sym[m]);
                break;
            case OpNames::H:
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(
                    this->find_site_op_info(op.q_label, this->orb_sym[m]));
                (*p.second)[S(0, 0, 0, 0)](0, 0) = 0.0;
                (*p.second)[S(1, 1, 1, this->orb_sym[m])](0, 0) = t(m, m);
                (*p.second)[S(2, 0, 0, 0)](0, 0) = t(m, m) * 2 + v(m, m, m, m);
                break;
            case OpNames::R:
                i = op.site_index[0];
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(i, m)) < TINY && abs(v(i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[0].at(OpNames::D));
                    p.second->factor *= t(i, m) * sqrt(2) / 4;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    tmp->copy_data_from(*op_prims[0].at(OpNames::R));
                    tmp->factor = v(i, m, m, m);
                    this->opf->iadd(*p.second, *tmp);
                    if (this->opf->seq->mode != SeqTypes::None)
                        this->opf->seq->simple_perform();
                    tmp->deallocate();
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(i, m)) < TINY && abs(v(i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[0].at(OpNames::C));
                    p.second->factor *= t(i, m) * sqrt(2) / 4;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    tmp->copy_data_from(*op_prims[0].at(OpNames::RD));
                    tmp->factor = v(i, m, m, m);
                    this->opf->iadd(*p.second, *tmp);
                    if (this->opf->seq->mode != SeqTypes::None)
                        this->opf->seq->simple_perform();
                    tmp->deallocate();
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.s();
                if (abs(v(i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::AD)->data);
                    p.second->factor *= v(i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.s();
                if (abs(v(i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::A)->data);
                    p.second->factor *= v(i, m, k, m);
                }
                break;
            case OpNames::Q:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.s();
                switch (s) {
                case 0U:
                    if (abs(2 * v(i, j, m, m) - v(i, m, m, j)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                                               op.q_label, this->orb_sym[m]),
                                           op_prims[0].at(OpNames::B)->data);
                        p.second->factor *= 2 * v(i, j, m, m) - v(i, m, m, j);
                    }
                    break;
                case 1U:
                    if (abs(v(i, m, m, j)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                                               op.q_label, this->orb_sym[m]),
                                           op_prims[1].at(OpNames::B)->data);
                        p.second->factor *= v(i, m, m, j);
                    }
                    break;
                }
                break;
            default:
                assert(false);
            }
        }
    }
    void deallocate() override {
        for (auto name : vector<OpNames>{OpNames::RD, OpNames::R})
            op_prims[0][name]->deallocate();
        for (auto name :
             vector<OpNames>{OpNames::NN, OpNames::B, OpNames::AD, OpNames::A})
            op_prims[1][name]->deallocate();
        for (auto name :
             vector<OpNames>{OpNames::B, OpNames::AD, OpNames::A, OpNames::D,
                             OpNames::C, OpNames::NN, OpNames::N, OpNames::I})
            op_prims[0][name]->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            for (int j = this->site_op_infos[i].size() - 1; j >= 0; j--)
                this->site_op_infos[i][j].second->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            this->basis[i].deallocate();
        Hamiltonian<S>::deallocate();
    }
    double v(uint8_t i, uint8_t j, uint8_t k, uint8_t l) const {
        return fcidump->v(i, j, k, l);
    }
    double t(uint8_t i, uint8_t j) const {
        return i == j ? fcidump->t(i, i) - mu : fcidump->t(i, j);
    }
    double e() const { return fcidump->e; }
};

} // namespace block2
