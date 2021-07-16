
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
#include "../core/sparse_matrix.hpp"
#include "abstract_sci_wrapper.hpp"
#include "hamiltonian_sci.hpp"
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <utility>
#include <vector>
/** SCI Hamiltonian wrapper.
 */

using namespace std;


namespace block2 {
template <typename, typename = void> struct HamiltonianQCSCI;

/** Non-Spin-Adapted MR-SCI Hamiltonian.
 *
 * The last site is the MR-SCI site
 *
 * @ATTENTION: The API is still in progress!
 *
 * @tparam S Spin
 */
template <typename S>
struct HamiltonianQCSCI<S, typename S::is_sz_t> : HamiltonianSCI<S> {
private:
    static int getNOrbCas(const int nOrbTot, const std::shared_ptr<sci::AbstractSciWrapper<S>> &sciLeft,
                          const std::shared_ptr<sci::AbstractSciWrapper<S>> &sciRight){
        auto nCas = nOrbTot;
        nCas -= sciLeft==nullptr?0:sciLeft->nOrbThis;
        nCas -= sciRight==nullptr?0:sciRight->nOrbThis;
        return nCas;
    }
    static int getNSites(const int nOrbTot, const std::shared_ptr<sci::AbstractSciWrapper<S>> &sciLeft,
                         const std::shared_ptr<sci::AbstractSciWrapper<S>> &sciRight){
        auto nCas = getNOrbCas(nOrbTot, sciLeft, sciRight);
        if(sciLeft == nullptr){
            assert(sciRight != nullptr);
            return 1 + nCas;
        }else if(sciRight == nullptr){
            return 1 + nCas;
        }else{
            return 2 + nCas;
        }
    }
public:
    using HamiltonianSCI<S>::n_syms;
    using HamiltonianSCI<S>::n_sites;
    using HamiltonianSCI<S>::vacuum;
    using HamiltonianSCI<S>::basis;
    using HamiltonianSCI<S>::site_op_infos;
    using HamiltonianSCI<S>::find_site_op_info;
    using HamiltonianSCI<S>::find_site_norm_op;
    using HamiltonianSCI<S>::site_norm_ops;
    using HamiltonianSCI<S>::opf;
    using HamiltonianSCI<S>::orb_sym;
    using HamiltonianSCI<S>::delayed;

    map<OpNames, shared_ptr<SparseMatrix<S>>>
        op_prims_normal[4]; //!< Primitive operators for normal spatial orbitals
    //!! This is only used in get_site_ops
    shared_ptr<FCIDUMP> fcidump;
    double mu = 0; //!> Chemical potential
    std::shared_ptr<sci::AbstractSciWrapper<S>>
        sciWrapperLeft, sciWrapperRight; //!< Wrapper classes for the
                                         //!!  physical operators/determinants on the big site
    int nOrbLeft; //!> Number of spatial orbitals for left SCI space (small nOrb)
    int nOrbRight; //!> Number of spatial orbitals for right SCI space  (large nOrb)
    int nOrbCas; //!> Number of spatial orbitals in CAS (handled by normal MPS)
    bool sci_finalize = true;
    bool useRuleQC = true; // Use RuleQC for big sites
    mutable std::shared_ptr<Rule<S>> ruleQC = nullptr; // Will be changed in const method
    std::shared_ptr<ParallelRuleQC<S>> parallelRule = nullptr; // Enable it for big site in useRuleQC
    HamiltonianQCSCI(const S vacuum, const int nOrbTot, const vector<uint8_t> &orb_sym,
        const shared_ptr<FCIDUMP> &fcidump,
        const std::shared_ptr<sci::AbstractSciWrapper<S>> &sciWrapperLeft = nullptr,
        const std::shared_ptr<sci::AbstractSciWrapper<S>> &sciWrapperRight = nullptr)
            : HamiltonianSCI<S>(vacuum, getNSites(nOrbTot, sciWrapperLeft,sciWrapperRight), orb_sym),
              fcidump(fcidump),
              sciWrapperLeft(sciWrapperLeft), sciWrapperRight(sciWrapperRight),
              nOrbLeft{sciWrapperLeft==nullptr?0:sciWrapperLeft->nOrbThis},
              nOrbRight{sciWrapperRight==nullptr?0:sciWrapperRight->nOrbThis},
              nOrbCas{getNOrbCas(nOrbTot, sciWrapperLeft,sciWrapperRight)}
        {
        assert(sciWrapperLeft!= nullptr or sciWrapperRight != nullptr);
        const auto verbose = sciWrapperLeft != nullptr ? sciWrapperLeft->verbose : sciWrapperRight->verbose;
        if (verbose) {
            cout << " Hamiltonian: n_sites = " << (int) n_sites
                 << ", nOrbs = [" << nOrbLeft << ", " << nOrbCas << ", " << nOrbRight << "]" << endl;
        }
        if(orb_sym.size() != nOrbLeft + nOrbCas + nOrbRight){
            cout << "Error! orb_sym.size()=" << orb_sym.size() << "!= nOrbTot=" <<
            nOrbLeft + nOrbCas + nOrbRight << endl;
            throw std::runtime_error("Wrong orb_sym size");
        }
        if(nOrbLeft > nOrbRight and verbose){
            cout << "Warning: Left big site should have fewer orbitals than right one" << endl;
        }
        if(nOrbLeft + nOrbCas > nOrbRight and verbose){
            cout << "Warning: There should be more orbitals on the right big site "
                    "than on the other sites (NC scheme)" << endl;
        }
        iStart = sciWrapperLeft != nullptr ? 1 : 0;
        iEnd = sciWrapperRight != nullptr ? n_sites-1 : n_sites;
        if(sciWrapperLeft != nullptr) {
           initSciBasis(sciWrapperLeft, 0);
        }
        // CAS sites
        for (int iSite = iStart; iSite < iEnd; ++iSite) {
            basis[iSite] = make_shared<StateInfo<S>>();
            auto &bas = *basis[iSite];
            auto iSym = orb_sym[nOrbLeft == 0 ? iSite : iSite + nOrbLeft - 1];
            bas.allocate(4);
            bas.quanta[0] = vacuum;
            bas.quanta[1] = S(1, -1, iSym);
            bas.quanta[2] = S(1, 1, iSym);
            bas.quanta[3] = S(2, 0, 0);
            bas.n_states[0] = bas.n_states[1] = bas.n_states[2] =
                bas.n_states[3] = 1;
            bas.sort_states();
        }
        if (std::abs(mu) > 1e-12) {
            throw std::runtime_error("mu needs to be 0 right now");
        }
        if(sciWrapperRight != nullptr) {
            initSciBasis(sciWrapperRight, n_sites-1);
        }
        init_site_ops();
    }

    void deallocate() override {
        for (int8_t s = 3; s >= 0; s--)
            for (auto name : vector<OpNames>{OpNames::RD, OpNames::R})
                op_prims_normal[s][name]->deallocate();
        for (int8_t s = 3; s >= 0; s--)
            for (auto name : vector<OpNames>{OpNames::B, OpNames::AD,
                                             OpNames::A, OpNames::NN})
                op_prims_normal[s][name]->deallocate();
        for (int8_t s = 1; s >= 0; s--)
            for (auto name :
                 vector<OpNames>{OpNames::D, OpNames::C, OpNames::N})
                op_prims_normal[s][name]->deallocate();
        op_prims_normal[0][OpNames::I]->deallocate();
        for (int iSite = n_sites - 1; iSite >= 0; --iSite) {
            auto &info = site_op_infos[iSite];
            for (int j = info.size() - 1; j >= 0; j--)
                info[j].second->deallocate();
        }
        for (int iSite = iEnd; iSite >= iStart; --iSite) {
            basis[iSite]->deallocate();
        }
        HamiltonianSCI<S>::deallocate();
    }
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const {
        return fcidump->v(sl, sr, i, j, k, l);
    }
    double t(uint8_t s, uint16_t i, uint16_t j) const {
        return i == j ? fcidump->t(s, i, i) - mu : fcidump->t(s, i, j);
    }

    double e() const { return fcidump->e(); }

  protected:
    void get_site_ops(uint16_t m,
                      unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops) const override {
        shared_ptr<HamiltonianQCSCI> ph = nullptr;
        if (delayed != DelayedSCIOpNames::None) {
            ph = make_shared<HamiltonianQCSCI>(*this);
            ph->delayed = DelayedSCIOpNames::None;
            ph->sci_finalize = false;
        }
        uint16_t pm = m;
        if (m == n_sites - 1 and sciWrapperRight != nullptr) {
            if (!(delayed & DelayedSCIOpNames::RightBig))
                get_site_ops_big_site(sciWrapperRight, ops, m);
            else {
                for (auto &p : ops) {
                    OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                    p.second = make_shared<DelayedSparseMatrix<S, HamiltonianSCI<S>>>(
                        ph, pm, p.first, find_site_op_info(op.q_label, pm));
                }
            }
            return;
        }else if(m == 0 and sciWrapperLeft != nullptr){
            if (!(delayed & DelayedSCIOpNames::LeftBig))
                get_site_ops_big_site(sciWrapperLeft, ops, m);
            else {
                for (auto &p : ops) {
                    OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                    p.second = make_shared<DelayedSparseMatrix<S, HamiltonianSCI<S>>>(
                        ph, pm, p.first, find_site_op_info(op.q_label, pm));
                }
            }
            return;
        } else m = nOrbLeft == 0 ? m : m + nOrbLeft - 1;
        uint16_t i, j, k;
        uint8_t s;
        shared_ptr<SparseMatrix<S>> zero = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            switch (op.name) {
            case OpNames::I: // hrl: these are just normal operators so we can
                             // apply find_site_norm_op
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
                if (!(delayed & DelayedSCIOpNames::Normal))
                    p.second = find_site_norm_op(p.first, pm);
                break;
            case OpNames::H: // hrl: Here we need to take integrals into account
                if (!(delayed & DelayedSCIOpNames::H)) {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(find_site_op_info(op.q_label, pm));
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
                // hrl: this is S + R or S + .5 R; see webpage
                if (orb_sym[i] != orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else if (!(delayed & DelayedSCIOpNames::R)) {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(find_site_op_info(op.q_label, pm));
                    p.second->copy_data_from(op_prims_normal[s].at(OpNames::D));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->allocate(find_site_op_info(op.q_label, pm));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            op_prims_normal[s + (sp << 1)].at(OpNames::R));
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
                else if (!(delayed & DelayedSCIOpNames::RD)) {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(find_site_op_info(op.q_label, pm));
                    p.second->copy_data_from(op_prims_normal[s].at(OpNames::C));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->allocate(find_site_op_info(op.q_label, pm));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            op_prims_normal[s + (sp << 1)].at(OpNames::RD));
                        tmp->factor = v(s, sp, i, m, m, m);
                        opf->iadd(p.second, tmp);
                        if (opf->seq->mode != SeqTypes::None)
                            opf->seq->simple_perform();
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
                else if (!(delayed & DelayedSCIOpNames::P)) {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        find_site_op_info(op.q_label, pm),
                        op_prims_normal[s].at(OpNames::AD)->data);
                    p.second->factor *= v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else if (!(delayed & DelayedSCIOpNames::PD)) {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(find_site_op_info(op.q_label, pm),
                                       op_prims_normal[s].at(OpNames::A)->data);
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
                    else if (!(delayed & DelayedSCIOpNames::Q)) {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(find_site_op_info(op.q_label, pm));
                        p.second->copy_data_from(
                            op_prims_normal[(s >> 1) | ((s & 1) << 1)].at(
                                OpNames::B));
                        p.second->factor *= -v(s & 1, s >> 1, i, m, m, j);
                        tmp->allocate(find_site_op_info(op.q_label, pm));
                        for (uint8_t sp = 0; sp < 2; sp++) {
                            tmp->copy_data_from(
                                op_prims_normal[sp | (sp << 1)].at(OpNames::B));
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
                    else if (!(delayed & DelayedSCIOpNames::Q)) {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(
                            find_site_op_info(op.q_label, pm),
                            op_prims_normal[(s >> 1) | ((s & 1) << 1)]
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
                p.second = make_shared<DelayedSparseMatrix<S, HamiltonianSCI<S>>>(
                    ph, pm, p.first, find_site_op_info(op.q_label, pm));
        }
    }

  private:
    uint16_t iStart, iEnd; // Loop indices for normal sites
    void initSciBasis(const std::shared_ptr<sci::AbstractSciWrapper<S>>& sciWrapper, const int iSite){
        if (vacuum.n() != 0 or vacuum.twos() != 0) {
            // TODO why is vacuum an input?
            throw std::runtime_error("Weird vacuum; not implemented for sciwrapper");
        }
        basis[iSite] = make_shared<StateInfo<S>>();
        auto &bas = *basis[iSite];
        const auto qSize = static_cast<int>(sciWrapper->quantumNumbers.size());
        bas.allocate(qSize);
        for (int iQ = 0; iQ < qSize; ++iQ) {
            auto o1 = sciWrapper->offsets[iQ].first;
            auto o2 = sciWrapper->offsets[iQ].second;
            bas.quanta[iQ] = sciWrapper->quantumNumbers[iQ];
            bas.n_states[iQ] = o2 - o1;
        }
        bas.sort_states();
        for (int iQ = 0; iQ < qSize; ++iQ) {
            if (bas.quanta[iQ] != sciWrapper->quantumNumbers[iQ]) {
                // This should not happen as the quantum numbers are now always sorted accordingly
                cout << "Error for iSite = " << iSite << endl;
                throw std::runtime_error(
                        "HamiltonianQCSCI: sciWrapper states were not sorted according to StateInfo sort");
            }
        }
    }


    void init_site_ops() {
        init_site_op_infos();

        // pg symmetry for reference orbital
        // op_prims_normal need normal site as reference => use first normal orbital
        const uint8_t ipg = orb_sym[nOrbLeft];
        op_prims_normal[0][OpNames::I] = make_shared<SparseMatrix<S>>();
        op_prims_normal[0][OpNames::I]->allocate(
            find_site_op_info(S(0, 0, 0), iStart));
        (*op_prims_normal[0][OpNames::I])[S(0, 0, 0)](0, 0) = 1.0; //  |00>
        (*op_prims_normal[0][OpNames::I])[S(1, -1, ipg)](0, 0) =
            1.0; // |10> (alpha)
        (*op_prims_normal[0][OpNames::I])[S(1, 1, ipg)](0, 0) =
            1.0; //  |01> (beta)
        (*op_prims_normal[0][OpNames::I])[S(2, 0, 0)](0, 0) = 1.0; //  |11>
        const int sz[2] = {1, -1};
        for (uint8_t s = 0; s < 2; s++) {
            op_prims_normal[s][OpNames::N] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::N]->allocate(
                find_site_op_info(S(0, 0, 0), iStart));
            (*op_prims_normal[s][OpNames::N])[S(0, 0, 0)](0, 0) = 0.0;
            (*op_prims_normal[s][OpNames::N])[S(1, -1, ipg)](0, 0) = s;
            (*op_prims_normal[s][OpNames::N])[S(1, 1, ipg)](0, 0) = 1 - s;
            (*op_prims_normal[s][OpNames::N])[S(2, 0, 0)](0, 0) = 1.0;
            op_prims_normal[s][OpNames::C] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::C]->allocate(
                find_site_op_info(S(1, sz[s], ipg), iStart));
            (*op_prims_normal[s][OpNames::C])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims_normal[s][OpNames::C])[S(1, -sz[s], ipg)](0, 0) =
                s ? -1.0 : 1.0;
            op_prims_normal[s][OpNames::D] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::D]->allocate(
                find_site_op_info(S(-1, -sz[s], ipg), iStart));
            (*op_prims_normal[s][OpNames::D])[S(1, sz[s], ipg)](0, 0) = 1.0;
            (*op_prims_normal[s][OpNames::D])[S(2, 0, 0)](0, 0) =
                s ? -1.0 : 1.0;
        }
        // low (&1): left index, high (>>1): right index
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t s = 0; s < 4; s++) {
            op_prims_normal[s][OpNames::NN] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::NN]->allocate(
                find_site_op_info(S(0, 0, 0), iStart));
            // hrl: s & 1 : 0 => 0; 1 => 1; 2 => 0; 3 => 1 (first index)
            // hrl: s >> 1: 0 => 0; 1 => 0; 2 => 1; 3 => 1  (second index)
            opf->product(0, op_prims_normal[s & 1][OpNames::N],
                         op_prims_normal[s >> 1][OpNames::N],
                         op_prims_normal[s][OpNames::NN]);
            op_prims_normal[s][OpNames::A] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::A]->allocate(
                find_site_op_info(S(2, sz_plus[s], 0), iStart));
            opf->product(0, op_prims_normal[s & 1][OpNames::C],
                         op_prims_normal[s >> 1][OpNames::C],
                         op_prims_normal[s][OpNames::A]);
            op_prims_normal[s][OpNames::AD] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::AD]->allocate(
                find_site_op_info(S(-2, -sz_plus[s], 0), iStart));
            opf->product(0, op_prims_normal[s >> 1][OpNames::D],
                         op_prims_normal[s & 1][OpNames::D],
                         op_prims_normal[s][OpNames::AD]);
            op_prims_normal[s][OpNames::B] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::B]->allocate(
                find_site_op_info(S(0, sz_minus[s], 0), iStart));
            opf->product(0, op_prims_normal[s & 1][OpNames::C],
                         op_prims_normal[s >> 1][OpNames::D],
                         op_prims_normal[s][OpNames::B]);
        }
        // low (&1): R index, high (>>1): B index
        for (uint8_t s = 0; s < 4; s++) {
            op_prims_normal[s][OpNames::R] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::R]->allocate(
                find_site_op_info(S(-1, -sz[s & 1], ipg), iStart));
            opf->product(0, op_prims_normal[(s >> 1) | (s & 2)][OpNames::B],
                         op_prims_normal[s & 1][OpNames::D],
                         op_prims_normal[s][OpNames::R]);
            op_prims_normal[s][OpNames::RD] = make_shared<SparseMatrix<S>>();
            op_prims_normal[s][OpNames::RD]->allocate(
                find_site_op_info(S(1, sz[s & 1], ipg), iStart));
            opf->product(0, op_prims_normal[s & 1][OpNames::C],
                         op_prims_normal[(s >> 1) | (s & 2)][OpNames::B],
                         op_prims_normal[s][OpNames::RD]);
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
        const shared_ptr<OpElement<S>> nn_op[4] = {
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {0, 0}),
                                      this->vacuum),
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {1, 0}),
                                      this->vacuum),
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {0, 1}),
                                      this->vacuum),
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {1, 1}),
                                      this->vacuum)};
        for (uint8_t i = 0; i < n_syms; i++) {
            ops[i][i_op] = nullptr;
            for (uint8_t s = 0; s < 2; s++)
                ops[i][n_op[s]] = nullptr;
            for (uint8_t s = 0; s < 4; s++)
                ops[i][nn_op[s]] = nullptr;
        }
        // vv only for normal sites
        //                vvv needs to include all left and cas orbitals for NC scheme
        for (uint16_t m = nOrbLeft; m < nOrbLeft + nOrbCas; m++) {
            for (uint8_t s = 0; s < 2; s++) {
                ops[orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::C, SiteIndex({m}, {s}), S(1, sz[s], orb_sym[m]))] =
                    nullptr;
                ops[orb_sym[m]]
                   [make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], orb_sym[m]))] =
                       nullptr;
            }
            for (uint8_t s = 0; s < 4; s++) {
                ops[orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::A,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(2, sz_plus[s], 0))] = nullptr;
                ops[orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::AD,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(-2, -sz_plus[s], 0))] = nullptr;
                ops[orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::B,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, sz_minus[s], 0))] = nullptr;
            }
        }
        // vv only for normal sites
        for (uint16_t iSite = iStart; iSite < iEnd; ++iSite) {
            const auto iSym = orb_sym[nOrbLeft == 0 ? iSite : iSite + nOrbLeft - 1];
            site_norm_ops[iSite] = vector<
                pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>(
                ops[iSym].begin(), ops[iSym].end());
            for (auto &p : this->site_norm_ops[iSite]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(
                    find_site_op_info(op.q_label, iSite),
                    op_prims_normal[op.site_index.ss()][op.name]->data);
            }
        }
    }

    void init_site_op_infos() {
        // hrl this should be ok for big site as this is the delta quantum
        // number
        //  compare with qLabels in MPOQCSCI
        for (int iSite = 0; iSite < n_sites; ++iSite) {
            const auto iSym = orb_sym[nOrbLeft == 0 ? iSite : iSite + nOrbLeft - 1];
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vacuum] = nullptr;
            for (auto n : {-1, 1})
                for (auto s : {-1, 1})
                    info[S(n, s, iSym)] = nullptr;
            for (auto n : {-2, 0, 2})
                for (auto s : {-2, 0, 2})
                    info[S(n, s, 0)] = nullptr;
            if( (iSite == n_sites-1 and sciWrapperRight != nullptr) or
                (iSite == 0 and sciWrapperLeft != nullptr)){ // extremal site... pg
                // TODO which do I really need? Starting xSite from iSite does not work
                for(int xSite = 0; xSite < orb_sym.size(); ++xSite) {
                    for (auto n : {-1, 1})
                        for (auto s : {-1, 1})
                            info[S(n, s, orb_sym[xSite])] = nullptr;
                    for(int ySite = 0; ySite < orb_sym.size(); ++ySite) {
                        auto sym = orb_sym[xSite] ^ orb_sym[ySite];
                        for (auto n : {-2, 0, 2})
                            for (auto s : {-2, 0, 2})
                                info[S(n, s, sym)] = nullptr;
                    }
                }
            }
            site_op_infos[iSite] =
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                 info.end());
            for (auto &p : site_op_infos[iSite]) {
                p.second = make_shared<SparseMatrixInfo<S>>();
                auto &bas = *basis[iSite];
                p.second->initialize(bas, bas, p.first, p.first.is_fermion());
            }
        }
    }
    void get_site_ops_big_site(const std::shared_ptr<sci::AbstractSciWrapper<S>>& sciWrapper,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>
            &ops, int iSite) const {
        int ii, jj; // spin orbital indices
        // For optimization purposes (parallelization + orbital loop elsewhere
        // I want to collect operators of same name and quantum numbers
        using fillTuple2 = typename sci::AbstractSciWrapper<S>::entryTuple2;
        using fillTuple1 = typename sci::AbstractSciWrapper<S>::entryTuple1;
        // A,AD,B would be for a small site => I assume that it is not big and have not yet optimized it
        // Same for C and D, which is fast
        unordered_map< S, std::vector<fillTuple2>, sci::SHasher<S> > opsQ, opsP, opsPD;
        unordered_map< S, std::vector<fillTuple1>, sci::SHasher<S> > opsR, opsRD;
        if(not useRuleQC and ruleQC != nullptr){
            throw std::invalid_argument("HamiltonianQCSCI: ruleQC is set but useRuleQC is false");
        }
        if(useRuleQC and ruleQC == nullptr){
            ruleQC = std::make_shared<RuleQC<S>>();
        }
        for (auto &p : ops) {
            shared_ptr<OpElement<S>> pop = dynamic_pointer_cast<OpElement<S>>(p.first);
            OpElement<S> &op = *pop;
            auto canBeSkipped = parallelRule != nullptr and not parallelRule->available(pop);
            canBeSkipped = canBeSkipped or (useRuleQC and (*ruleQC)(pop) != nullptr);
            if(canBeSkipped){
                p.second = make_shared<DelayedSparseMatrix < S, OpExpr < S>>>(
                        iSite, p.first, find_site_op_info(op.q_label, iSite));
                continue;
            }
            auto pmat = make_shared<CSRSparseMatrix<S>>();
            auto &mat = *pmat;
            p.second = pmat;
            // ATTENTION vv if you change allocation, you need to change the
            //                      deallocation routine in MPOQCSCI
            // Also, the CSR stuff is more complicated and I will do the actual allocation
            //      of the individual matrices in the fillOp* routines.
            //      So here, the CSRMatrices are only initialized (i.e., their sizes are set)
            mat.initialize(find_site_op_info(op.q_label, iSite));
            const auto& delta_qn = op.q_label;
            if (false and op.name == OpNames::R) { // DEBUG
                cout << "m == " << iSite << "allocate" << op.name
                     << "s" << (int)op.site_index[0] << ","
                     << (int)op.site_index[1] << "ss" << (int)op.site_index.s(0)
                     << (int)op.site_index.s(1) << endl;
                cout << "q_label:" << op.q_label << endl;
            }
            // get orbital indices
            ii = -1;
            jj = -1; // debug
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
                sciWrapper->fillOp_I(mat);
                break;
            case OpNames::N:
                sciWrapper->fillOp_N(mat);
                break;
            case OpNames::NN:
                sciWrapper->fillOp_NN(mat);
                break;
            case OpNames::H:
                sciWrapper->fillOp_H(mat);
                break;
            case OpNames::C:
                sciWrapper->fillOp_C(delta_qn, mat, ii);
                break;
            case OpNames::D:
                sciWrapper->fillOp_D(delta_qn, mat, ii);
                break;
            case OpNames::R:
                opsR[delta_qn].emplace_back(mat,ii);
                break;
            case OpNames::RD:
                opsRD[delta_qn].emplace_back(mat,ii);
                break;
            case OpNames::A:
                sciWrapper->fillOp_A(delta_qn, mat, ii, jj);
                break;
            case OpNames::AD:
                sciWrapper->fillOp_AD(delta_qn, mat, ii, jj);
                break;
            case OpNames::B:
                sciWrapper->fillOp_B(delta_qn, mat, ii, jj);
                break;
            case OpNames::P:
                opsP[delta_qn].emplace_back(mat,ii,jj);
                break;
            case OpNames::PD:
                opsPD[delta_qn].emplace_back(mat,ii,jj);
                break;
            case OpNames::Q:
                opsQ[delta_qn].emplace_back(mat,ii,jj);
                break;
            default:
                assert(false);
            }
        }
        for(auto& pairs: opsR){
            sciWrapper->fillOp_R(pairs.first, pairs.second);
            pairs.second.clear(); pairs.second.shrink_to_fit();
        }
        for(auto& pairs: opsRD){
            sciWrapper->fillOp_RD(pairs.first, pairs.second);
            pairs.second.clear(); pairs.second.shrink_to_fit();
        }
        for(auto& pairs: opsP){
            sciWrapper->fillOp_P(pairs.first, pairs.second);
            pairs.second.clear(); pairs.second.shrink_to_fit();
        }
        for(auto& pairs: opsPD){
            sciWrapper->fillOp_PD(pairs.first, pairs.second);
            pairs.second.clear(); pairs.second.shrink_to_fit();
        }
        for(auto& pairs: opsQ){
            sciWrapper->fillOp_Q(pairs.first, pairs.second);
            pairs.second.clear(); pairs.second.shrink_to_fit();
        }
        if (sci_finalize)
            sciWrapper->finalize();

        if (useRuleQC)
            // Take care operators are fully "symmetric" for simplification.
            // E.g. if P[i,j] is 0, PD[j,i] should be 0 as well
            for (auto &p : ops) {
                auto pop = dynamic_pointer_cast<OpElement<S>>(p.first);
                if(parallelRule != nullptr and not parallelRule->available(pop)){
                    continue;
                }
                if (p.second->get_type() == SparseMatrixTypes::Delayed) {
                    auto ref_op = (*ruleQC)(pop)->op;
                    if (ops.count(ref_op) && (ops.at(ref_op)->factor == 0.0 ||
                                              ops.at(ref_op)->norm() < TINY))
                        p.second->factor = 0.0;
                }
            }
#ifdef _HAS_MPI
        // Take care of zeros in MPI...
        if(parallelRule != nullptr){
            vector<char> isZero(ops.size()); // bool is not fully mpi compatible -.-
            int ii = 0;
            for (auto &p : ops) {
                isZero[ii++] = p.second->factor == 0.0;
            }
            int ierr = MPI_Allreduce(MPI_IN_PLACE, isZero.data(), isZero.size(),
                                     MPI_CHAR, MPI_LOR,
                                     MPI_COMM_WORLD);
            assert(ierr == 0);
            ii = 0;
            for (auto &p : ops) {
                if(isZero[ii++]){
                        p.second->factor = 0.0;
                }
            }
        }
#endif
    }
};

} // namespace block2
