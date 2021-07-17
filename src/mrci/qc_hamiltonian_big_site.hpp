
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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
#include "../core/integral.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/rule.hpp"
#include "../core/sparse_matrix.hpp"
#include "../dmrg/qc_hamiltonian.hpp"
#include "qc_big_site.hpp"
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

/** MRCI Hamiltonian */

using namespace std;

namespace block2 {

template <typename, typename = void> struct HamiltonianQCBigSite;

/** Non-Spin-Adapted bigsite Hamiltonian.
 * The last site is the big site.
 * @tparam S Spin
 */
template <typename S>
struct HamiltonianQCBigSite<S, typename S::is_sz_t> : HamiltonianQC<S> {
    using HamiltonianQC<S>::n_syms;
    using HamiltonianQC<S>::n_sites;
    using HamiltonianQC<S>::vacuum;
    using HamiltonianQC<S>::basis;
    using HamiltonianQC<S>::site_op_infos;
    using HamiltonianQC<S>::find_site_op_info;
    using HamiltonianQC<S>::opf;
    using HamiltonianQC<S>::orb_sym;
    using HamiltonianQC<S>::delayed;
    shared_ptr<BigSiteQC<S>> big_left,
        big_right; //!< Wrapper classes for the physical operators/determinants
                   //!< on the big site
    int n_orbs_left;  //!> Number of spatial orbitals for left SCI space (small
                      //! nOrb)
    int n_orbs_right; //!> Number of spatial orbitals for right SCI space (large
                      //! nOrb)
    int n_orbs_cas;   //!> Number of spatial orbitals in CAS (handled by normal
                      //! MPS)
    bool big_site_finalize = true;
    shared_ptr<Rule<S>> rule = nullptr;
    shared_ptr<ParallelRule<S>> parallel_rule = nullptr;
    shared_ptr<HamiltonianQC<S>> full_hamil;
    HamiltonianQCBigSite(S vacuum, int n_orbs_total,
                         const vector<uint8_t> &orb_sym,
                         const shared_ptr<FCIDUMP> &fcidump,
                         const shared_ptr<BigSiteQC<S>> &big_left = nullptr,
                         const shared_ptr<BigSiteQC<S>> &big_right = nullptr)
        : HamiltonianQC<S>(), big_left(big_left), big_right(big_right),
          n_orbs_left(big_left == nullptr ? 0 : big_left->n_orbs_this),
          n_orbs_right(big_right == nullptr ? 0 : big_right->n_orbs_this),
          n_orbs_cas(n_orbs_total - n_orbs_left - n_orbs_right) {
        full_hamil = make_shared<HamiltonianQC<S>>(vacuum, n_orbs_total,
                                                   orb_sym, fcidump);
        if (big_left != nullptr || big_right != nullptr)
            opf = make_shared<CSROperatorFunctions<S>>(full_hamil->opf->cg);
        else
            opf = full_hamil->opf;
        vacuum = full_hamil->vacuum;
        n_sites = n_orbs_cas + !!n_orbs_left + !!n_orbs_right;
        this->orb_sym = orb_sym;
        n_syms = orb_sym.size() == 0
                     ? 0
                     : *max_element(orb_sym.begin(), orb_sym.end()) + 1;
        assert((int)orb_sym.size() == n_orbs_left + n_orbs_cas + n_orbs_right);
        basis = vector<shared_ptr<StateInfo<S>>>(
            full_hamil->basis.begin() + n_orbs_left,
            full_hamil->basis.end() - n_orbs_right);
        site_op_infos =
            vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
                full_hamil->site_op_infos.begin() + n_orbs_left,
                full_hamil->site_op_infos.end() - n_orbs_right);
        if (big_left != nullptr) {
            basis.insert(basis.begin(), get_big_site_basis(big_left));
            site_op_infos.insert(site_op_infos.begin(),
                                 get_big_site_op_infos(0));
        }
        if (big_right != nullptr) {
            basis.push_back(get_big_site_basis(big_right));
            site_op_infos.push_back(get_big_site_op_infos(n_sites - 1));
        }
    }
    void set_mu(double mu) override {
        this->mu = mu;
        full_hamil->set_mu(mu);
    }
    shared_ptr<StateInfo<S>>
    get_big_site_basis(shared_ptr<BigSiteQC<S>> big) const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate((int)big->qs.size());
        memcpy(b->quanta, big->qs.data(), b->n * sizeof(S));
        for (int iq = 0; iq < (int)big->qs.size(); iq++)
            b->n_states[iq] = big->offsets[iq].second - big->offsets[iq].first;
        b->sort_states();
        return b;
    }
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
    get_big_site_op_infos(uint16_t m) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        map<S, shared_ptr<SparseMatrixInfo<S>>> info;
        info[vacuum] = nullptr;
        for (auto ipg : orb_sym) {
            for (int n = -1; n <= 1; n += 2)
                for (int s = -3; s <= 3; s += 2)
                    info[S(n, s, ipg)] = nullptr;
            for (auto jpg : orb_sym)
                for (int n = -2; n <= 2; n += 2)
                    for (int s = -4; s <= 4; s += 2)
                        info[S(n, s, ipg ^ jpg)] = nullptr;
        }
        for (auto &p : info) {
            p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
            p.second->initialize(*basis[m], *basis[m], p.first,
                                 p.first.is_fermion());
        }
        return vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                info.end());
    }
    void deallocate() override {
        if (big_left != nullptr) {
            for (int j = (int)site_op_infos[0].size() - 1; j >= 0; j--)
                site_op_infos[0][j].second->deallocate();
            basis[0]->deallocate();
        }
        if (big_right != nullptr) {
            for (int j = (int)site_op_infos.back().size() - 1; j >= 0; j--)
                site_op_infos.back()[j].second->deallocate();
            basis.back()->deallocate();
        }
        full_hamil->deallocate();
    }
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<HamiltonianQCBigSite<S>> ph = nullptr;
        if (delayed != DelayedOpNames::None) {
            ph = make_shared<HamiltonianQCBigSite<S>>(*this);
            ph->delayed = DelayedOpNames::None;
            ph->big_site_finalize = false;
        }
        if (m == n_sites - 1 && big_right != nullptr) {
            if (!(delayed & DelayedOpNames::RightBig))
                get_big_site_ops(big_right, m, ops);
            else {
                for (auto &p : ops) {
                    OpElement<S> &op =
                        *dynamic_pointer_cast<OpElement<S>>(p.first);
                    p.second =
                        make_shared<DelayedSparseMatrix<S, Hamiltonian<S>>>(
                            ph, m, p.first, find_site_op_info(m, op.q_label));
                }
            }
        } else if (m == 0 && big_left != nullptr) {
            if (!(delayed & DelayedOpNames::LeftBig))
                get_big_site_ops(big_left, m, ops);
            else {
                for (auto &p : ops) {
                    OpElement<S> &op =
                        *dynamic_pointer_cast<OpElement<S>>(p.first);
                    p.second =
                        make_shared<DelayedSparseMatrix<S, Hamiltonian<S>>>(
                            ph, m, p.first, find_site_op_info(m, op.q_label));
                }
            }
        } else
            full_hamil->get_site_ops(m + (n_orbs_left ? n_orbs_left - 1 : 0),
                                     ops);
    }
    void get_big_site_ops(
        const shared_ptr<BigSiteQC<S>> &big, uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const {
        int ii, jj; // spin orbital indices
        // For optimization purposes (parallelization + orbital loop elsewhere
        // I want to collect operators of same name and quantum numbers
        // A,AD,B would be for a small site => I assume that it is not big and
        // have not yet optimized it Same for C and D, which is fast
        unordered_map<S, vector<pair<pair<int, int>, CSRSparseMatrix<S>>>>
            ops_q, ops_p, ops_pd;
        unordered_map<S, vector<pair<int, CSRSparseMatrix<S>>>> ops_r, ops_rd;
        for (auto &p : ops) {
            shared_ptr<OpElement<S>> pop =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            OpElement<S> &op = *pop;
            bool skip =
                parallel_rule != nullptr && !parallel_rule->available(pop);
            skip = skip || (rule != nullptr && (*rule)(pop) != nullptr);
            if (skip) {
                p.second = make_shared<DelayedSparseMatrix<S, OpExpr<S>>>(
                    m, p.first, find_site_op_info(m, op.q_label));
                continue;
            }
            auto pmat = make_shared<CSRSparseMatrix<S>>();
            auto &mat = *pmat;
            p.second = pmat;
            // ATTENTION vv if you change allocation, you need to change the
            //                      deallocation routine in MPOQCSCI
            // Also, the CSR stuff is more complicated and I will do the actual
            // allocation
            //      of the individual matrices in the fillOp* routines.
            //      So here, the CSRMatrices are only initialized (i.e., their
            //      sizes are set)
            mat.initialize(find_site_op_info(m, op.q_label));
            // get orbital indices
            ii = -1;
            jj = -1;
            switch (op.name) {
            case OpNames::C:
            case OpNames::D:
            case OpNames::R:
            case OpNames::RD:
                ii = 2 * op.site_index[0] + op.site_index.s(0);
                break;
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
            case OpNames::P:
            case OpNames::PD:
            case OpNames::Q:
                ii = 2 * op.site_index[0] + op.site_index.s(0);
                jj = 2 * op.site_index[1] + op.site_index.s(1);
                break;
            default:
                break;
            }
            switch (op.name) {
            case OpNames::I:
                big->fill_op_i(mat);
                break;
            case OpNames::N:
                big->fill_op_n(mat);
                break;
            case OpNames::NN:
                big->fill_op_nn(mat);
                break;
            case OpNames::H:
                big->fill_op_h(mat);
                break;
            case OpNames::C:
                big->fill_op_c(mat, ii);
                break;
            case OpNames::D:
                big->fill_op_d(mat, ii);
                break;
            case OpNames::R:
                ops_r[op.q_label].emplace_back(ii, mat);
                break;
            case OpNames::RD:
                ops_rd[op.q_label].emplace_back(ii, mat);
                break;
            case OpNames::A:
                big->fill_op_a(mat, ii, jj);
                break;
            case OpNames::AD:
                big->fill_op_ad(mat, ii, jj);
                break;
            case OpNames::B:
                big->fill_op_b(mat, ii, jj);
                break;
            case OpNames::P:
                ops_p[op.q_label].emplace_back(make_pair(ii, jj), mat);
                break;
            case OpNames::PD:
                ops_pd[op.q_label].emplace_back(make_pair(ii, jj), mat);
                break;
            case OpNames::Q:
                ops_q[op.q_label].emplace_back(make_pair(ii, jj), mat);
                break;
            default:
                assert(false);
            }
        }
        for (auto &pairs : ops_r) {
            big->fill_op_r(pairs.second);
            pairs.second.clear();
            pairs.second.shrink_to_fit();
        }
        for (auto &pairs : ops_rd) {
            big->fill_op_rd(pairs.second);
            pairs.second.clear();
            pairs.second.shrink_to_fit();
        }
        for (auto &pairs : ops_p) {
            big->fill_op_p(pairs.second);
            pairs.second.clear();
            pairs.second.shrink_to_fit();
        }
        for (auto &pairs : ops_pd) {
            big->fill_op_pd(pairs.second);
            pairs.second.clear();
            pairs.second.shrink_to_fit();
        }
        for (auto &pairs : ops_q) {
            big->fill_op_q(pairs.second);
            pairs.second.clear();
            pairs.second.shrink_to_fit();
        }
        if (big_site_finalize)
            big->finalize();
        if (rule != nullptr)
            // Take care operators are fully "symmetric" for simplification.
            // E.g. if P[i,j] is 0, PD[j,i] should be 0 as well
            for (auto &p : ops) {
                auto pop = dynamic_pointer_cast<OpElement<S>>(p.first);
                if (parallel_rule != nullptr && !parallel_rule->available(pop))
                    continue;
                if (p.second->get_type() == SparseMatrixTypes::Delayed) {
                    auto ref_op = (*rule)(pop)->op;
                    if (ops.count(ref_op) && (ops.at(ref_op)->factor == 0.0 ||
                                              ops.at(ref_op)->norm() < TINY))
                        p.second->factor = 0.0;
                }
            }
        // Take care of zeros in MPI...
        if (parallel_rule != nullptr) {
            vector<char> is_zero(
                ops.size()); // bool is not fully mpi compatible -.-
            int ii = 0;
            for (auto &p : ops)
                is_zero[ii++] = p.second->factor == 0.0;
            parallel_rule->comm->allreduce_logical_or(is_zero.data(),
                                                      is_zero.size());
            ii = 0;
            for (auto &p : ops)
                if (is_zero[ii++])
                    p.second->factor = 0.0;
        }
    }
};

} // namespace block2
