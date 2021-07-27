
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

#include "../core/csr_operator_functions.hpp"
#include "../core/expr.hpp"
#include "../core/integral.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/rule.hpp"
#include "../core/sparse_matrix.hpp"
#include "../dmrg/qc_hamiltonian.hpp"
#include "big_site.hpp"
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

/** Big-site Hamiltonian. */

using namespace std;

namespace block2 {

/** HamiltonianQC with left and/or right big sites. */
template <typename S> struct HamiltonianQCBigSite : HamiltonianQC<S> {
    using HamiltonianQC<S>::n_syms;
    using HamiltonianQC<S>::n_sites;
    using HamiltonianQC<S>::vacuum;
    using HamiltonianQC<S>::basis;
    using HamiltonianQC<S>::site_op_infos;
    using HamiltonianQC<S>::find_site_op_info;
    using HamiltonianQC<S>::opf;
    using HamiltonianQC<S>::delayed;
    shared_ptr<BigSite<S>> big_left,
        big_right; //!< Wrapper classes for the physical operators/determinants
                   //!< on the big site
    int n_orbs_left;  //!> Number of spatial orbitals for the left big site
    int n_orbs_right; //!> Number of spatial orbitals for the right big site
    int n_orbs_cas;   //!> Number of spatial orbitals in CAS (handled by normal
                      //! MPS)
    int n_orbs;
    shared_ptr<Rule<S>> rule = nullptr;
    shared_ptr<ParallelRule<S>> parallel_rule = nullptr;
    shared_ptr<HamiltonianQC<S>> full_hamil;
    HamiltonianQCBigSite(S vacuum, int n_orbs_total,
                         const vector<uint8_t> &orb_sym,
                         const shared_ptr<FCIDUMP> &fcidump,
                         const shared_ptr<BigSite<S>> &big_left = nullptr,
                         const shared_ptr<BigSite<S>> &big_right = nullptr)
        : HamiltonianQC<S>(), big_left(big_left), big_right(big_right),
          n_orbs(n_orbs_total),
          n_orbs_left(big_left == nullptr ? 0 : big_left->n_orbs),
          n_orbs_right(big_right == nullptr ? 0 : big_right->n_orbs),
          n_orbs_cas(n_orbs_total - n_orbs_left - n_orbs_right) {
        full_hamil = make_shared<HamiltonianQC<S>>(vacuum, n_orbs_total,
                                                   orb_sym, fcidump);
        if (big_left != nullptr || big_right != nullptr)
            opf = make_shared<CSROperatorFunctions<S>>(full_hamil->opf->cg);
        else
            opf = full_hamil->opf;
        n_sites = n_orbs_cas + !!n_orbs_left + !!n_orbs_right;
        this->vacuum = full_hamil->vacuum;
        this->orb_sym = full_hamil->orb_sym;
        this->fcidump = full_hamil->fcidump;
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
            basis.insert(basis.begin(), big_left->basis);
            site_op_infos.insert(site_op_infos.begin(), big_left->op_infos);
        }
        if (big_right != nullptr) {
            basis.push_back(big_right->basis);
            site_op_infos.push_back(big_right->op_infos);
        }
    }
    int get_n_orbs_left() const override { return n_orbs_left; }
    int get_n_orbs_right() const override { return n_orbs_right; }
    void set_mu(double mu) override {
        this->mu = mu;
        full_hamil->set_mu(mu);
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
        }
        if (m == n_sites - 1 && big_right != nullptr) {
            if (!(delayed & DelayedOpNames::RightBig))
                big_right->get_site_ops(m, ops);
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
                big_left->get_site_ops(m, ops);
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
};

} // namespace block2
