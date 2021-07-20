
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

#include "../core/expr.hpp"
#include "../core/hamiltonian.hpp"
#include "../core/integral.hpp"
#include "../core/sparse_matrix.hpp"
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct HamiltonianQC;

// Quantum chemistry Hamiltonian (non-spin-adapted)
template <typename S>
struct HamiltonianQC<S, typename S::is_sz_t> : Hamiltonian<S> {
    using Hamiltonian<S>::vacuum;
    using Hamiltonian<S>::n_sites;
    using Hamiltonian<S>::n_syms;
    using Hamiltonian<S>::basis;
    using Hamiltonian<S>::site_op_infos;
    using Hamiltonian<S>::orb_sym;
    using Hamiltonian<S>::find_site_op_info;
    using Hamiltonian<S>::opf;
    using Hamiltonian<S>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    vector<unordered_map<OpNames, shared_ptr<SparseMatrix<S>>>> op_prims;
    // For storage of one-electron and two-electron integrals
    shared_ptr<FCIDUMP> fcidump;
    // Chemical potenital parameter in Hamiltonian
    double mu = 0;
    HamiltonianQC()
        : Hamiltonian<S>(S(), 0, vector<uint8_t>()), fcidump(nullptr) {}
    HamiltonianQC(S vacuum, int n_sites, const vector<uint8_t> &orb_sym,
                  const shared_ptr<FCIDUMP> &fcidump)
        : Hamiltonian<S>(vacuum, n_sites, orb_sym), fcidump(fcidump) {
        // SZ does not need CG factors
        opf = make_shared<OperatorFunctions<S>>(make_shared<CG<S>>());
        opf->cg->initialize();
        basis.resize(n_sites);
        op_prims.resize(6);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual void set_mu(double mu) { this->mu = mu; }
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(4);
        b->quanta[0] = vacuum;
        b->quanta[1] = S(1, 1, orb_sym[m]);
        b->quanta[2] = S(1, -1, orb_sym[m]);
        b->quanta[3] = S(2, 0, 0);
        b->n_states[0] = b->n_states[1] = b->n_states[2] = b->n_states[3] = 1;
        b->sort_states();
        return b;
    }
    void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vacuum] = nullptr;
            for (int n = -1; n <= 1; n += 2)
                for (int s = -3; s <= 3; s += 2)
                    info[S(n, s, orb_sym[m])] = nullptr;
            for (int n = -2; n <= 2; n += 2)
                for (int s = -4; s <= 4; s += 2)
                    info[S(n, s, 0)] = nullptr;
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        const uint8_t ipg = orb_sym[0];
        op_prims[0][OpNames::I] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::I]->allocate(find_site_op_info(0, S(0, 0, 0)));
        (*op_prims[0][OpNames::I])[S(0, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, -1, ipg)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, 1, ipg)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(2, 0, 0)](0, 0) = 1.0;
        const int sz[2] = {1, -1};
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s][OpNames::N] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::N]->allocate(find_site_op_info(0, S(0, 0, 0)));
            (*op_prims[s][OpNames::N])[S(0, 0, 0)](0, 0) = 0.0;
            (*op_prims[s][OpNames::N])[S(1, -1, ipg)](0, 0) = s;
            (*op_prims[s][OpNames::N])[S(1, 1, ipg)](0, 0) = 1 - s;
            (*op_prims[s][OpNames::N])[S(2, 0, 0)](0, 0) = 1.0;
            op_prims[s][OpNames::C] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::C]->allocate(
                find_site_op_info(0, S(1, sz[s], ipg)));
            (*op_prims[s][OpNames::C])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims[s][OpNames::C])[S(1, -sz[s], ipg)](0, 0) =
                s ? -1.0 : 1.0;
            op_prims[s][OpNames::D] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::D]->allocate(
                find_site_op_info(0, S(-1, -sz[s], ipg)));
            (*op_prims[s][OpNames::D])[S(1, sz[s], ipg)](0, 0) = 1.0;
            (*op_prims[s][OpNames::D])[S(2, 0, 0)](0, 0) = s ? -1.0 : 1.0;
        }
        // low (&1): left index, high (>>1): right index
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t s = 0; s < 4; s++) {
            op_prims[s][OpNames::NN] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::NN]->allocate(
                find_site_op_info(0, S(0, 0, 0)));
            opf->product(0, op_prims[s & 1][OpNames::N],
                         op_prims[s >> 1][OpNames::N],
                         op_prims[s][OpNames::NN]);
            op_prims[s][OpNames::A] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::A]->allocate(
                find_site_op_info(0, S(2, sz_plus[s], 0)));
            opf->product(0, op_prims[s & 1][OpNames::C],
                         op_prims[s >> 1][OpNames::C], op_prims[s][OpNames::A]);
            op_prims[s][OpNames::AD] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::AD]->allocate(
                find_site_op_info(0, S(-2, -sz_plus[s], 0)));
            opf->product(0, op_prims[s >> 1][OpNames::D],
                         op_prims[s & 1][OpNames::D], op_prims[s][OpNames::AD]);
            op_prims[s][OpNames::B] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::B]->allocate(
                find_site_op_info(0, S(0, sz_minus[s], 0)));
            opf->product(0, op_prims[s & 1][OpNames::C],
                         op_prims[s >> 1][OpNames::D], op_prims[s][OpNames::B]);
            op_prims[s][OpNames::BD] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::BD]->allocate(
                find_site_op_info(0, S(0, -sz_minus[s], 0)));
            opf->product(0, op_prims[s & 1][OpNames::D],
                         op_prims[s >> 1][OpNames::C],
                         op_prims[s][OpNames::BD]);
        }
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s + 4][OpNames::NN] =
                make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s + 4][OpNames::NN]->allocate(
                find_site_op_info(0, S(0, 0, 0)));
            opf->product(0, op_prims[s | ((!s) << 1)][OpNames::B],
                         op_prims[(!s) | (s << 1)][OpNames::B],
                         op_prims[s + 4][OpNames::NN]);
        }
        // low (&1): R index, high (>>1): B index
        for (uint8_t s = 0; s < 4; s++) {
            // C (s & 2) D (s & 2) D (s & 1)
            op_prims[s][OpNames::R] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::R]->allocate(
                find_site_op_info(0, S(-1, -sz[s & 1], ipg)));
            opf->product(0, op_prims[(s >> 1) | (s & 2)][OpNames::B],
                         op_prims[s & 1][OpNames::D], op_prims[s][OpNames::R]);
            // C (s & 1) C (s & 2) D (s & 2)
            op_prims[s][OpNames::RD] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::RD]->allocate(
                find_site_op_info(0, S(1, sz[s & 1], ipg)));
            opf->product(0, op_prims[s & 1][OpNames::C],
                         op_prims[(s >> 1) | (s & 2)][OpNames::B],
                         op_prims[s][OpNames::RD]);
        }
        // site norm operators
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), vacuum);
        const shared_ptr<OpElement<S>> n_op[2] = {
            make_shared<OpElement<S>>(OpNames::N, SiteIndex({}, {0}), vacuum),
            make_shared<OpElement<S>>(OpNames::N, SiteIndex({}, {1}), vacuum)};
        for (uint16_t m = 0; m < n_sites; m++) {
            site_norm_ops[m][i_op] = nullptr;
            for (uint8_t s = 0; s < 2; s++) {
                site_norm_ops[m][n_op[s]] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::C, SiteIndex({m}, {s}), S(1, sz[s], orb_sym[m]))] =
                    nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], orb_sym[m]))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::NN,
                    SiteIndex({m, m},
                              {(uint8_t)(s & 1), (uint8_t)0, (uint8_t)1}),
                    S(0, 0, 0))] = nullptr;
            }
            for (uint8_t s = 0; s < 4; s++) {
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::A,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(2, sz_plus[s], 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::AD,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(-2, -sz_plus[s], 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::B,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, sz_minus[s], 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::BD,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, -sz_minus[s], 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::NN,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, 0, 0))] = nullptr;
            }
        }
        for (uint16_t m = 0; m < n_sites; m++)
            for (auto &p : site_norm_ops[m]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                // no memory allocated by allocator
                p.second = make_shared<SparseMatrix<S>>(nullptr);
                p.second->allocate(find_site_op_info(m, op.q_label),
                                   op_prims[op.site_index.ss()][op.name]->data);
            }
    }
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        uint16_t i, j, k;
        uint8_t s;
        shared_ptr<SparseMatrix<S>> zero =
            make_shared<SparseMatrix<S>>(nullptr);
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>(d_alloc);
        shared_ptr<Hamiltonian<S>> ph = nullptr;
        if (delayed != DelayedOpNames::None) {
            ph = make_shared<HamiltonianQC>(*this);
            ph->delayed = DelayedOpNames::None;
            assert(delayed != DelayedOpNames::None);
        }
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            shared_ptr<SparseMatrixInfo<S>> info =
                find_site_op_info(m, op.q_label);
            switch (op.name) {
            case OpNames::I:
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
            case OpNames::BD:
                if (!(delayed & DelayedOpNames::Normal))
                    p.second = site_norm_ops[m].at(p.first);
                break;
            case OpNames::CCDD:
                if (!(delayed & DelayedOpNames::CCDD)) {
                    s = op.site_index.ss();
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    // q_label is not used for comparison
                    opf->product(
                        0,
                        site_norm_ops[m].at(make_shared<OpElement<S>>(
                            OpNames::A,
                            SiteIndex({m, m}, {(uint8_t)(s & 1),
                                               (uint8_t)((s & 2) >> 1)}),
                            vacuum)),
                        site_norm_ops[m].at(make_shared<OpElement<S>>(
                            OpNames::AD,
                            SiteIndex({m, m}, {(uint8_t)((s & 8) >> 3),
                                               (uint8_t)((s & 4) >> 2)}),
                            vacuum)),
                        p.second);
                }
                break;
            case OpNames::CCD:
                if (!(delayed & DelayedOpNames::CCD)) {
                    s = op.site_index.ss();
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    // q_label is not used for comparison
                    opf->product(
                        0,
                        site_norm_ops[m].at(make_shared<OpElement<S>>(
                            OpNames::A,
                            SiteIndex({m, m}, {(uint8_t)(s & 1),
                                               (uint8_t)((s & 2) >> 1)}),
                            vacuum)),
                        site_norm_ops[m].at(make_shared<OpElement<S>>(
                            OpNames::D, SiteIndex({m}, {(uint8_t)(s >> 2)}),
                            vacuum)),
                        p.second);
                }
                break;
            case OpNames::CDD:
                if (!(delayed & DelayedOpNames::CDD)) {
                    s = op.site_index.ss();
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    // q_label is not used for comparison
                    opf->product(
                        0,
                        site_norm_ops[m].at(make_shared<OpElement<S>>(
                            OpNames::B,
                            SiteIndex({m, m}, {(uint8_t)(s & 1),
                                               (uint8_t)((s & 2) >> 1)}),
                            vacuum)),
                        site_norm_ops[m].at(make_shared<OpElement<S>>(
                            OpNames::D, SiteIndex({m}, {(uint8_t)(s >> 2)}),
                            vacuum)),
                        p.second);
                }
                break;
            case OpNames::H:
                if (!(delayed & DelayedOpNames::H)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    (*p.second)[S(0, 0, 0)](0, 0) = 0.0;
                    (*p.second)[S(1, -1, orb_sym[m])](0, 0) = t(1, m, m);
                    (*p.second)[S(1, 1, orb_sym[m])](0, 0) = t(0, m, m);
                    (*p.second)[S(2, 0, 0)](0, 0) =
                        t(0, m, m) + t(1, m, m) +
                        0.5 * (v(0, 1, m, m, m, m) + v(1, 0, m, m, m, m));
                }
                break;
            case OpNames::R:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::R)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    p.second->copy_data_from(op_prims[s].at(OpNames::D));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->alloc = d_alloc;
                    tmp->allocate(info);
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            op_prims[s + (sp << 1)].at(OpNames::R));
                        tmp->factor = v(s, sp, i, m, m, m);
                        opf->iadd(p.second, tmp);
                        if (opf->seq->mode != SeqTypes::None)
                            opf->seq->simple_perform();
                    }
                    tmp->deallocate();
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::RD)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    p.second->copy_data_from(op_prims[s].at(OpNames::C));
                    p.second->factor *= t(s, m, i) * 0.5;
                    tmp->alloc = d_alloc;
                    tmp->allocate(info);
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            op_prims[s + (sp << 1)].at(OpNames::RD));
                        tmp->factor = v(s, sp, i, m, m, m);
                        opf->iadd(p.second, tmp);
                        if (opf->seq->mode != SeqTypes::None)
                            opf->seq->simple_perform();
                    }
                    tmp->deallocate();
                }
                break;
            case OpNames::TR:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY &&
                     abs(v(0, s, m, m, i, m)) < TINY &&
                     abs(v(1, s, m, m, i, m)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::TR)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    p.second->copy_data_from(op_prims[s].at(OpNames::D));
                    p.second->iscale(-2.0 * t(s, i, m));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        // q_label is not used for comparison
                        opf->product(
                            0,
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::B, SiteIndex({m, m}, {sp, sp}),
                                vacuum)),
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::D, SiteIndex({m}, {s}), vacuum)),
                            p.second, -v(s, sp, i, m, m, m));
                        opf->product(
                            0,
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::B, SiteIndex({m, m}, {sp, s}),
                                vacuum)),
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::D, SiteIndex({m}, {sp}), vacuum)),
                            p.second, v(sp, s, m, m, i, m));
                    }
                }
                break;
            case OpNames::TS:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(v(s, 0, m, i, m, m)) < TINY &&
                     abs(v(s, 1, m, i, m, m)) < TINY &&
                     abs(v(0, s, m, m, m, i)) < TINY &&
                     abs(v(1, s, m, m, m, i)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::TS)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        // q_label is not used for comparison
                        opf->product(
                            0,
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::C, SiteIndex({m}, {s}), vacuum)),
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::B, SiteIndex({m, m}, {sp, sp}),
                                vacuum)),
                            p.second, v(s, sp, m, i, m, m));
                        opf->product(
                            0,
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::C, SiteIndex({m}, {sp}), vacuum)),
                            site_norm_ops[m].at(make_shared<OpElement<S>>(
                                OpNames::B, SiteIndex({m, m}, {s, sp}),
                                vacuum)),
                            p.second, -v(sp, s, m, m, m, i));
                    }
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::P)) {
                    p.second = make_shared<SparseMatrix<S>>(nullptr);
                    p.second->allocate(info, op_prims[s].at(OpNames::AD)->data);
                    p.second->factor *= v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::PD)) {
                    p.second = make_shared<SparseMatrix<S>>(nullptr);
                    p.second->allocate(info, op_prims[s].at(OpNames::A)->data);
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
                    else if (!(delayed & DelayedOpNames::Q)) {
                        p.second = make_shared<SparseMatrix<S>>(d_alloc);
                        p.second->allocate(info);
                        p.second->copy_data_from(
                            op_prims[(s >> 1) | ((s & 1) << 1)].at(OpNames::B));
                        p.second->factor *= -v(s & 1, s >> 1, i, m, m, j);
                        tmp->alloc = d_alloc;
                        tmp->allocate(info);
                        for (uint8_t sp = 0; sp < 2; sp++) {
                            tmp->copy_data_from(
                                op_prims[sp | (sp << 1)].at(OpNames::B));
                            tmp->factor = v(s & 1, sp, i, j, m, m);
                            opf->iadd(p.second, tmp);
                            if (opf->seq->mode != SeqTypes::None)
                                opf->seq->simple_perform();
                        }
                        tmp->deallocate();
                    }
                    break;
                case 1U:
                case 2U:
                    if (abs(v(s & 1, s >> 1, i, m, m, j)) < TINY)
                        p.second = zero;
                    else if (!(delayed & DelayedOpNames::Q)) {
                        p.second = make_shared<SparseMatrix<S>>(nullptr);
                        p.second->allocate(info,
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
            if (p.second == nullptr)
                p.second = make_shared<DelayedSparseMatrix<S, Hamiltonian<S>>>(
                    ph, m, p.first, info);
        }
    }
    void deallocate() override {
        for (auto &ops_map : op_prims)
            for (auto &p : ops_map)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        opf->cg->deallocate();
        Hamiltonian<S>::deallocate();
    }
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const {
        return fcidump->v(sl, sr, i, j, k, l);
    }
    double t(uint8_t s, uint16_t i, uint16_t j) const {
        return i == j ? fcidump->t(s, i, i) - mu : fcidump->t(s, i, j);
    }
    double e() const { return fcidump->e(); }
};

// Quantum chemistry Hamiltonian (spin-adapted)
template <typename S>
struct HamiltonianQC<S, typename S::is_su2_t> : Hamiltonian<S> {
    using Hamiltonian<S>::vacuum;
    using Hamiltonian<S>::n_sites;
    using Hamiltonian<S>::n_syms;
    using Hamiltonian<S>::basis;
    using Hamiltonian<S>::site_op_infos;
    using Hamiltonian<S>::orb_sym;
    using Hamiltonian<S>::find_site_op_info;
    using Hamiltonian<S>::opf;
    using Hamiltonian<S>::delayed;
    // Sparse matrix representation for normal site operators
    vector<unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>
        site_norm_ops;
    // Primitives for sparse matrix representation for normal site operators
    vector<unordered_map<OpNames, shared_ptr<SparseMatrix<S>>>> op_prims;
    // For storage of one-electron and two-electron integrals
    shared_ptr<FCIDUMP> fcidump;
    // Chemical potenital parameter in Hamiltonian
    double mu = 0;
    HamiltonianQC()
        : Hamiltonian<S>(S(), 0, vector<uint8_t>()), fcidump(nullptr) {}
    HamiltonianQC(S vacuum, int n_sites, const vector<uint8_t> &orb_sym,
                  const shared_ptr<FCIDUMP> &fcidump)
        : Hamiltonian<S>(vacuum, n_sites, orb_sym), fcidump(fcidump) {
        // SU2 does not support UHF orbitals
        assert(!fcidump->uhf);
        opf = make_shared<OperatorFunctions<S>>(make_shared<CG<S>>(100));
        opf->cg->initialize();
        basis.resize(n_sites);
        op_prims.resize(2);
        site_op_infos.resize(n_sites);
        site_norm_ops.resize(n_sites);
        for (uint16_t m = 0; m < n_sites; m++)
            basis[m] = get_site_basis(m);
        init_site_ops();
    }
    virtual void set_mu(double mu) { this->mu = mu; }
    virtual shared_ptr<StateInfo<S>> get_site_basis(uint16_t m) const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(3);
        b->quanta[0] = vacuum;
        b->quanta[1] = S(1, 1, orb_sym[m]);
        b->quanta[2] = S(2, 0, 0);
        b->n_states[0] = b->n_states[1] = b->n_states[2] = 1;
        b->sort_states();
        return b;
    }
    void init_site_ops() {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        // site operator infos
        for (uint16_t m = 0; m < n_sites; m++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[vacuum] = nullptr;
            info[S(1, 1, orb_sym[m])] = nullptr;
            info[S(-1, 1, orb_sym[m])] = nullptr;
            for (int n = -2; n <= 2; n += 2)
                for (int s = 0; s <= 2; s += 2)
                    info[S(n, s, 0)] = nullptr;
            for (auto &p : info) {
                p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
                p.second->initialize(*basis[m], *basis[m], p.first,
                                     p.first.is_fermion());
            }
            site_op_infos[m] = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
                info.begin(), info.end());
        }
        const uint8_t ipg = orb_sym[0];
        op_prims[0][OpNames::I] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::I]->allocate(find_site_op_info(0, S(0, 0, 0)));
        (*op_prims[0][OpNames::I])[S(0, 0, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, 1, 1, ipg)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(2, 0, 0, 0)](0, 0) = 1.0;
        op_prims[0][OpNames::N] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::N]->allocate(find_site_op_info(0, S(0, 0, 0)));
        (*op_prims[0][OpNames::N])[S(0, 0, 0, 0)](0, 0) = 0.0;
        (*op_prims[0][OpNames::N])[S(1, 1, 1, ipg)](0, 0) = 1.0;
        (*op_prims[0][OpNames::N])[S(2, 0, 0, 0)](0, 0) = 2.0;
        // NN[0] = (sum_{sigma} ad_{p,sigma} a_{p,sigma}) ^ 2
        op_prims[0][OpNames::NN] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::NN]->allocate(find_site_op_info(0, S(0, 0, 0)));
        (*op_prims[0][OpNames::NN])[S(0, 0, 0, 0)](0, 0) = 0.0;
        (*op_prims[0][OpNames::NN])[S(1, 1, 1, ipg)](0, 0) = 1.0;
        (*op_prims[0][OpNames::NN])[S(2, 0, 0, 0)](0, 0) = 4.0;
        op_prims[0][OpNames::C] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::C]->allocate(find_site_op_info(0, S(1, 1, ipg)));
        (*op_prims[0][OpNames::C])[S(0, 1, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::C])[S(1, 0, 1, ipg)](0, 0) = -sqrt(2);
        op_prims[0][OpNames::D] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::D]->allocate(find_site_op_info(0, S(-1, 1, ipg)));
        (*op_prims[0][OpNames::D])[S(1, 0, 1, ipg)](0, 0) = sqrt(2);
        (*op_prims[0][OpNames::D])[S(2, 1, 0, 0)](0, 0) = 1.0;
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s][OpNames::A] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::A]->allocate(
                find_site_op_info(0, S(2, s * 2, 0)));
            opf->product(0, op_prims[0][OpNames::C], op_prims[0][OpNames::C],
                         op_prims[s][OpNames::A]);
            op_prims[s][OpNames::AD] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::AD]->allocate(
                find_site_op_info(0, S(-2, s * 2, 0)));
            opf->product(0, op_prims[0][OpNames::D], op_prims[0][OpNames::D],
                         op_prims[s][OpNames::AD]);
            op_prims[s][OpNames::B] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::B]->allocate(
                find_site_op_info(0, S(0, s * 2, 0)));
            opf->product(0, op_prims[0][OpNames::C], op_prims[0][OpNames::D],
                         op_prims[s][OpNames::B]);
            op_prims[s][OpNames::BD] = make_shared<SparseMatrix<S>>(d_alloc);
            op_prims[s][OpNames::BD]->allocate(
                find_site_op_info(0, S(0, s * 2, 0)));
            opf->product(0, op_prims[0][OpNames::D], op_prims[0][OpNames::C],
                         op_prims[s][OpNames::BD]);
        }
        // NN[1] = sum_{sigma,tau} ad_{p,sigma} a_{p,tau} ad_{p,tau} a_{p,sigma}
        // = -sqrt(3) B1 x B1 + B0 x B0 where B0 x B0 == 0.5 NN
        op_prims[1][OpNames::NN] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[1][OpNames::NN]->allocate(find_site_op_info(0, S(0, 0, 0)));
        opf->product(0, op_prims[1][OpNames::B], op_prims[1][OpNames::B],
                     op_prims[1][OpNames::NN], -sqrt(3.0));
        opf->iadd(op_prims[1][OpNames::NN], op_prims[0][OpNames::NN], 0.5);
        if (opf->seq->mode != SeqTypes::None)
            opf->seq->simple_perform();
        op_prims[0][OpNames::R] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::R]->allocate(find_site_op_info(0, S(-1, 1, ipg)));
        opf->product(0, op_prims[0][OpNames::B], op_prims[0][OpNames::D],
                     op_prims[0][OpNames::R]);
        op_prims[0][OpNames::RD] = make_shared<SparseMatrix<S>>(d_alloc);
        op_prims[0][OpNames::RD]->allocate(find_site_op_info(0, S(1, 1, ipg)));
        opf->product(0, op_prims[0][OpNames::C], op_prims[0][OpNames::B],
                     op_prims[0][OpNames::RD]);
        // site norm operators
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), vacuum);
        const shared_ptr<OpElement<S>> n_op =
            make_shared<OpElement<S>>(OpNames::N, SiteIndex(), vacuum);
        for (uint16_t m = 0; m < n_sites; m++) {
            site_norm_ops[m][i_op] = nullptr;
            site_norm_ops[m][n_op] = nullptr;
            site_norm_ops[m][make_shared<OpElement<S>>(
                OpNames::C, SiteIndex(m), S(1, 1, orb_sym[m]))] = nullptr;
            site_norm_ops[m][make_shared<OpElement<S>>(
                OpNames::D, SiteIndex(m), S(-1, 1, orb_sym[m]))] = nullptr;
            for (uint8_t s = 0; s < 2; s++) {
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::A, SiteIndex(m, m, s), S(2, s * 2, 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::AD, SiteIndex(m, m, s), S(-2, s * 2, 0))] =
                    nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(m, m, s), S(0, s * 2, 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::BD, SiteIndex(m, m, s), S(0, s * 2, 0))] = nullptr;
                site_norm_ops[m][make_shared<OpElement<S>>(
                    OpNames::NN, SiteIndex(m, m, s), vacuum)] = nullptr;
            }
        }
        for (uint16_t m = 0; m < n_sites; m++) {
            for (auto &p : site_norm_ops[m]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                // no memory allocated by allocator
                p.second = make_shared<SparseMatrix<S>>(nullptr);
                p.second->allocate(find_site_op_info(m, op.q_label),
                                   op_prims[op.site_index.ss()][op.name]->data);
            }
        }
    }
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        uint16_t i, j, k;
        uint8_t s;
        shared_ptr<SparseMatrix<S>> zero =
            make_shared<SparseMatrix<S>>(nullptr);
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>(d_alloc);
        shared_ptr<Hamiltonian<S>> ph = nullptr;
        if (delayed != DelayedOpNames::None) {
            ph = make_shared<HamiltonianQC>(*this);
            ph->delayed = DelayedOpNames::None;
            assert(delayed != DelayedOpNames::None);
        }
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            shared_ptr<SparseMatrixInfo<S>> info =
                find_site_op_info(m, op.q_label);
            switch (op.name) {
            case OpNames::I:
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
            case OpNames::BD:
                if (!(delayed & DelayedOpNames::Normal))
                    p.second = site_norm_ops[m].at(p.first);
                break;
            case OpNames::CCDD:
                if (!(delayed & DelayedOpNames::CCDD)) {
                    s = op.site_index.ss();
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    // q_label is not used for comparison
                    opf->product(0,
                                 site_norm_ops[m].at(make_shared<OpElement<S>>(
                                     OpNames::A, SiteIndex(m, m, s), vacuum)),
                                 site_norm_ops[m].at(make_shared<OpElement<S>>(
                                     OpNames::AD, SiteIndex(m, m, s), vacuum)),
                                 p.second);
                }
                break;
            case OpNames::CCD:
                if (!(delayed & DelayedOpNames::CCD)) {
                    s = op.site_index.ss();
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    // q_label is not used for comparison
                    opf->product(0,
                                 site_norm_ops[m].at(make_shared<OpElement<S>>(
                                     OpNames::C, SiteIndex(m), vacuum)),
                                 site_norm_ops[m].at(make_shared<OpElement<S>>(
                                     OpNames::B, SiteIndex(m, m, s), vacuum)),
                                 p.second);
                }
                break;
            case OpNames::CDD:
                if (!(delayed & DelayedOpNames::CDD)) {
                    s = op.site_index.ss();
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    // q_label is not used for comparison
                    opf->product(0,
                                 site_norm_ops[m].at(make_shared<OpElement<S>>(
                                     OpNames::C, SiteIndex(m), vacuum)),
                                 site_norm_ops[m].at(make_shared<OpElement<S>>(
                                     OpNames::AD, SiteIndex(m, m, s), vacuum)),
                                 p.second);
                }
                break;
            case OpNames::H:
                if (!(delayed & DelayedOpNames::H)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    (*p.second)[S(0, 0, 0, 0)](0, 0) = 0.0;
                    (*p.second)[S(1, 1, 1, orb_sym[m])](0, 0) = t(m, m);
                    (*p.second)[S(2, 0, 0, 0)](0, 0) =
                        t(m, m) * 2 + v(m, m, m, m);
                }
                break;
            case OpNames::R:
                i = op.site_index[0];
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(t(i, m)) < TINY && abs(v(i, m, m, m)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::R)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    p.second->copy_data_from(op_prims[0].at(OpNames::D));
                    p.second->factor *= t(i, m) * sqrt(2) / 4;
                    tmp->alloc = d_alloc;
                    tmp->allocate(info);
                    tmp->copy_data_from(op_prims[0].at(OpNames::R));
                    tmp->factor = v(i, m, m, m);
                    opf->iadd(p.second, tmp);
                    if (opf->seq->mode != SeqTypes::None)
                        opf->seq->simple_perform();
                    tmp->deallocate();
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(t(m, i)) < TINY && abs(v(i, m, m, m)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::RD)) {
                    p.second = make_shared<SparseMatrix<S>>(d_alloc);
                    p.second->allocate(info);
                    p.second->copy_data_from(op_prims[0].at(OpNames::C));
                    p.second->factor *= t(m, i) * sqrt(2) / 4;
                    tmp->alloc = d_alloc;
                    tmp->allocate(info);
                    tmp->copy_data_from(op_prims[0].at(OpNames::RD));
                    tmp->factor = v(i, m, m, m);
                    opf->iadd(p.second, tmp);
                    if (opf->seq->mode != SeqTypes::None)
                        opf->seq->simple_perform();
                    tmp->deallocate();
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.s();
                if (abs(v(i, m, k, m)) < TINY)
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::P)) {
                    p.second = make_shared<SparseMatrix<S>>(nullptr);
                    p.second->allocate(info, op_prims[s].at(OpNames::AD)->data);
                    p.second->factor *= v(i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.s();
                if (abs(v(i, m, k, m)) < TINY)
                    p.second = zero;
                else if (!(delayed & DelayedOpNames::PD)) {
                    p.second = make_shared<SparseMatrix<S>>(nullptr);
                    p.second->allocate(info, op_prims[s].at(OpNames::A)->data);
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
                    else if (!(delayed & DelayedOpNames::Q)) {
                        p.second = make_shared<SparseMatrix<S>>(nullptr);
                        p.second->allocate(info,
                                           op_prims[0].at(OpNames::B)->data);
                        p.second->factor *= 2 * v(i, j, m, m) - v(i, m, m, j);
                    }
                    break;
                case 1U:
                    if (abs(v(i, m, m, j)) < TINY)
                        p.second = zero;
                    else if (!(delayed & DelayedOpNames::Q)) {
                        p.second = make_shared<SparseMatrix<S>>(nullptr);
                        p.second->allocate(info,
                                           op_prims[1].at(OpNames::B)->data);
                        p.second->factor *= v(i, m, m, j);
                    }
                    break;
                }
                break;
            default:
                assert(false);
            }
            if (p.second == nullptr)
                p.second = make_shared<DelayedSparseMatrix<S, Hamiltonian<S>>>(
                    ph, m, p.first, info);
        }
    }
    void deallocate() override {
        for (auto &ops_map : op_prims)
            for (auto &p : ops_map)
                p.second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            for (int j = (int)site_op_infos[m].size() - 1; j >= 0; j--)
                site_op_infos[m][j].second->deallocate();
        for (int16_t m = n_sites - 1; m >= 0; m--)
            basis[m]->deallocate();
        opf->cg->deallocate();
        Hamiltonian<S>::deallocate();
    }
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return fcidump->v(i, j, k, l);
    }
    double t(uint16_t i, uint16_t j) const {
        return i == j ? fcidump->t(i, i) - mu : fcidump->t(i, j);
    }
    double e() const { return fcidump->e(); }
};

} // namespace block2
