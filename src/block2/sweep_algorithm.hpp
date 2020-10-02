
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
#include "matrix.hpp"
#include "moving_environment.hpp"
#include "sparse_matrix.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

// Density Matrix Renormalization Group
template <typename S> struct DMRG {
    shared_ptr<MovingEnvironment<S>> me;
    vector<ubond_t> bond_dims;
    vector<double> noises;
    vector<vector<double>> energies;
    vector<double> discarded_weights;
    vector<vector<vector<pair<S, double>>>> mps_quanta;
    vector<double> davidson_conv_thrds;
    int davidson_max_iter = 5000;
    int davidson_soft_max_iter = -1;
    bool forward;
    uint8_t iprint = 2;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    double cutoff = 1E-14;
    double quanta_cutoff = 1E-3;
    bool decomp_last_site = true;
    DMRG(const shared_ptr<MovingEnvironment<S>> &me,
         const vector<ubond_t> &bond_dims, const vector<double> &noises)
        : me(me), bond_dims(bond_dims), noises(noises), forward(false) {}
    virtual ~DMRG() = default;
    struct Iteration {
        vector<double> energies;
        vector<vector<pair<S, double>>> quanta;
        double error, tdav;
        int ndav, mmps;
        size_t nflop;
        Iteration(const vector<double> &energies, double error, int mmps,
                  int ndav, size_t nflop = 0, double tdav = 1.0)
            : energies(energies), error(error), mmps(mmps), ndav(ndav),
              nflop(nflop), tdav(tdav) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Mmps =" << setw(5) << r.mmps;
            os << " Ndav =" << setw(4) << r.ndav;
            if (r.energies.size() == 1)
                os << " E = " << setw(17) << setprecision(10) << r.energies[0];
            else if (r.quanta.size() == 0) {
                os << " E = ";
                for (auto x : r.energies)
                    os << setw(17) << setprecision(10) << x;
            }
            os << " Error = " << scientific << setw(8) << setprecision(2)
               << r.error << " FLOPS = " << scientific << setw(8)
               << setprecision(2) << (double)r.nflop / r.tdav
               << " Tdav = " << fixed << setprecision(2) << r.tdav;
            if (r.energies.size() != 1 && r.quanta.size() != 0) {
                for (size_t i = 0; i < r.energies.size(); i++) {
                    os << endl;
                    os << setw(15) << " .. E[" << setw(3) << i << "] = ";
                    os << setw(15) << setprecision(8) << r.energies[i];
                    for (size_t j = 0; j < r.quanta[i].size(); j++)
                        os << " " << setw(20) << r.quanta[i][j].first << " ("
                           << setw(8) << setprecision(6)
                           << r.quanta[i][j].second << ")";
                }
            }
            return os;
        }
    };
    // one-site single-state dmrg algorithm
    // canonical form for wavefunction: K = left-fused, S = right-fused
    Iteration update_one_dot(int i, bool forward, ubond_t bond_dim,
                             double noise, double davidson_conv_thrd) {
        frame->activate(0);
        bool fuse_left = i <= me->n_sites / 2;
        if (me->ket->canonical_form[i] == 'C') {
            if (i == 0)
                me->ket->canonical_form[i] = 'K';
            else if (i == me->n_sites - 1)
                me->ket->canonical_form[i] = 'S';
            else
                assert(false);
        }
        // guess wavefunction
        // change to fused form for super-block hamiltonian
        // note that this switch exactly matches two-site conventional mpo
        // middle-site switch, so two-site conventional mpo can work
        me->ket->load_tensor(i);
        if ((fuse_left && me->ket->canonical_form[i] == 'S') ||
            (!fuse_left && me->ket->canonical_form[i] == 'K')) {
            shared_ptr<SparseMatrix<S>> prev_wfn = me->ket->tensors[i];
            if (fuse_left && me->ket->canonical_form[i] == 'S')
                me->ket->tensors[i] =
                    MovingEnvironment<S>::swap_wfn_to_fused_left(
                        i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
            else if (!fuse_left && me->ket->canonical_form[i] == 'K')
                me->ket->tensors[i] =
                    MovingEnvironment<S>::swap_wfn_to_fused_right(
                        i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
            prev_wfn->info->deallocate();
            prev_wfn->deallocate();
        }
        S opdq = me->mpo->op->q_label;
        int mmps = 0;
        double error = 0.0;
        tuple<double, int, size_t, double> pdi;
        shared_ptr<SparseMatrixGroup<S>> pket = nullptr;
        // effective hamiltonian
        if (davidson_soft_max_iter != 0 || noise != 0) {
            shared_ptr<EffectiveHamiltonian<S>> h_eff =
                me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                            true, me->bra->tensors[i], me->ket->tensors[i]);
            pdi =
                h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                            davidson_soft_max_iter, me->para_rule);
            if (noise_type == NoiseTypes::Perturbative && noise != 0)
                pket = h_eff->perturbative_noise(forward, i, i,
                                                 fuse_left ? FuseTypes::FuseL
                                                           : FuseTypes::FuseR,
                                                 me->ket->info, me->para_rule);
            h_eff->deallocate();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (!decomp_last_site &&
                ((forward && i == me->n_sites - 1 && !fuse_left) ||
                 (!forward && i == 0 && fuse_left))) {
                me->ket->save_tensor(i);
                me->ket->unload_tensor(i);
                me->ket->canonical_form[i] = forward ? 'S' : 'K';
            } else {
                // change to fused form for splitting
                if (fuse_left != forward) {
                    shared_ptr<SparseMatrix<S>> prev_wfn = me->ket->tensors[i];
                    if (!fuse_left && forward)
                        me->ket->tensors[i] =
                            MovingEnvironment<S>::swap_wfn_to_fused_left(
                                i, me->ket->info, prev_wfn,
                                me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        me->ket->tensors[i] =
                            MovingEnvironment<S>::swap_wfn_to_fused_right(
                                i, me->ket->info, prev_wfn,
                                me->mpo->tf->opf->cg);
                    prev_wfn->info->deallocate();
                    prev_wfn->deallocate();
                    if (pket != nullptr) {
                        vector<shared_ptr<SparseMatrixGroup<S>>> prev_pkets = {
                            pket};
                        if (!fuse_left && forward)
                            pket = MovingEnvironment<S>::
                                swap_multi_wfn_to_fused_left(
                                    i, me->ket->info, prev_pkets,
                                    me->mpo->tf->opf->cg)[0];
                        else if (fuse_left && !forward)
                            pket = MovingEnvironment<S>::
                                swap_multi_wfn_to_fused_right(
                                    i, me->ket->info, prev_pkets,
                                    me->mpo->tf->opf->cg)[0];
                        prev_pkets[0]->deallocate_infos();
                        prev_pkets[0]->deallocate();
                    }
                }
                // splitting of wavefunction
                shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
                shared_ptr<SparseMatrix<S>> dm, left, right;
                if (decomp_type == DecompositionTypes::DensityMatrix) {
                    if (noise_type == NoiseTypes::Perturbative && noise != 0) {
                        dm = MovingEnvironment<S>::
                            density_matrix_with_perturbative_noise(
                                opdq, me->ket->tensors[i], forward, noise,
                                pket);
                        pket->deallocate_infos();
                        pket->deallocate();
                    } else
                        dm = MovingEnvironment<S>::density_matrix(
                            opdq, me->ket->tensors[i], forward, noise,
                            noise_type);
                    error = MovingEnvironment<S>::split_density_matrix(
                        dm, me->ket->tensors[i], (int)bond_dim, forward, true,
                        left, right, cutoff, trunc_type);
                } else if (decomp_type == DecompositionTypes::SVD ||
                           decomp_type == DecompositionTypes::PureSVD) {
                    assert(noise_type == NoiseTypes::None ||
                           noise_type == NoiseTypes::Perturbative ||
                           noise_type == NoiseTypes::Wavefunction);
                    if (noise != 0) {
                        if (noise_type == NoiseTypes::Wavefunction)
                            MovingEnvironment<S>::wavefunction_add_noise(
                                me->ket->tensors[i], noise);
                        else if (noise_type == NoiseTypes::Perturbative)
                            MovingEnvironment<S>::sacle_perturbative_noise(
                                noise, pket);
                    }
                    error = MovingEnvironment<S>::split_wavefunction_svd(
                        opdq, me->ket->tensors[i], (int)bond_dim, forward, true,
                        left, right, cutoff, trunc_type, decomp_type, pket);
                } else
                    assert(false);
                shared_ptr<StateInfo<S>> info = nullptr;
                // propagation
                if (forward) {
                    me->ket->tensors[i] = left;
                    me->ket->save_tensor(i);
                    info = left->info->extract_state_info(forward);
                    mmps = info->n_states_total;
                    me->ket->info->left_dims[i + 1] = info;
                    me->ket->info->save_left_dims(i + 1);
                    info->deallocate();
                    if (i != me->n_sites - 1) {
                        MovingEnvironment<S>::contract_one_dot(
                            i + 1, right, me->ket, forward);
                        me->ket->save_tensor(i + 1);
                        me->ket->unload_tensor(i + 1);
                        me->ket->canonical_form[i] = 'L';
                        me->ket->canonical_form[i + 1] = 'S';
                    } else {
                        me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                        MovingEnvironment<S>::contract_one_dot(
                            i, right, me->ket, !forward);
                        me->ket->save_tensor(i);
                        me->ket->unload_tensor(i);
                        me->ket->canonical_form[i] = 'K';
                    }
                } else {
                    me->ket->tensors[i] = right;
                    me->ket->save_tensor(i);
                    info = right->info->extract_state_info(forward);
                    mmps = info->n_states_total;
                    me->ket->info->right_dims[i] = info;
                    me->ket->info->save_right_dims(i);
                    info->deallocate();
                    if (i > 0) {
                        MovingEnvironment<S>::contract_one_dot(
                            i - 1, left, me->ket, forward);
                        me->ket->save_tensor(i - 1);
                        me->ket->unload_tensor(i - 1);
                        me->ket->canonical_form[i - 1] = 'K';
                        me->ket->canonical_form[i] = 'R';
                    } else {
                        me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                        MovingEnvironment<S>::contract_one_dot(i, left, me->ket,
                                                               !forward);
                        me->ket->save_tensor(i);
                        me->ket->unload_tensor(i);
                        me->ket->canonical_form[i] = 'S';
                    }
                }
                right->info->deallocate();
                right->deallocate();
                left->info->deallocate();
                left->deallocate();
                if (dm != nullptr) {
                    dm->info->deallocate();
                    dm->deallocate();
                }
                old_wfn->info->deallocate();
                old_wfn->deallocate();
            }
        } else {
            if (pket != nullptr) {
                pket->deallocate();
                pket->deallocate_infos();
            }
            me->ket->unload_tensor(i);
            if (!decomp_last_site &&
                ((forward && i == me->n_sites - 1 && !fuse_left) ||
                 (!forward && i == 0 && fuse_left)))
                me->ket->canonical_form[i] = forward ? 'S' : 'K';
            else {
                if (forward) {
                    if (i != me->n_sites - 1) {
                        me->ket->canonical_form[i] = 'L';
                        me->ket->canonical_form[i + 1] = 'S';
                    } else
                        me->ket->canonical_form[i] = 'K';
                } else {
                    if (i > 0) {
                        me->ket->canonical_form[i - 1] = 'K';
                        me->ket->canonical_form[i] = 'R';
                    } else
                        me->ket->canonical_form[i] = 'S';
                }
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(vector<double>{get<0>(pdi) + me->mpo->const_e}, error,
                         mmps, get<1>(pdi), get<2>(pdi), get<3>(pdi));
    }
    // two-site single-state dmrg algorithm
    // canonical form for wavefunction: C = center
    Iteration update_two_dot(int i, bool forward, ubond_t bond_dim,
                             double noise, double davidson_conv_thrd) {
        frame->activate(0);
        if (me->ket->tensors[i] != nullptr &&
            me->ket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_two_dot(i, me->ket);
        else {
            me->ket->load_tensor(i);
            me->ket->tensors[i + 1] = nullptr;
        }
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        S opdq = me->mpo->op->q_label;
        int mmps = 0;
        double error = 0.0;
        tuple<double, int, size_t, double> pdi;
        shared_ptr<SparseMatrixGroup<S>> pket = nullptr;
        // effective hamiltonian
        if (davidson_soft_max_iter != 0 || noise != 0) {
            shared_ptr<EffectiveHamiltonian<S>> h_eff =
                me->eff_ham(FuseTypes::FuseLR, true, me->bra->tensors[i],
                            me->ket->tensors[i]);
            pdi =
                h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                            davidson_soft_max_iter, me->para_rule);
            if (noise_type == NoiseTypes::Perturbative && noise != 0)
                pket = h_eff->perturbative_noise(forward, i, i + 1,
                                                 FuseTypes::FuseLR,
                                                 me->ket->info, me->para_rule);
            h_eff->deallocate();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<SparseMatrix<S>> dm;
            if (decomp_type == DecompositionTypes::DensityMatrix) {
                if (noise_type == NoiseTypes::Perturbative && noise != 0) {
                    dm = MovingEnvironment<
                        S>::density_matrix_with_perturbative_noise(opdq,
                                                                   old_wfn,
                                                                   forward,
                                                                   noise, pket);
                    pket->deallocate();
                    pket->deallocate_infos();
                } else
                    dm = MovingEnvironment<S>::density_matrix(
                        opdq, old_wfn, forward, noise, noise_type);
                error = MovingEnvironment<S>::split_density_matrix(
                    dm, old_wfn, (int)bond_dim, forward, true,
                    me->ket->tensors[i], me->ket->tensors[i + 1], cutoff,
                    trunc_type);
            } else if (decomp_type == DecompositionTypes::SVD ||
                       decomp_type == DecompositionTypes::PureSVD) {
                assert(noise_type == NoiseTypes::None ||
                       noise_type == NoiseTypes::Perturbative ||
                       noise_type == NoiseTypes::Wavefunction);
                if (noise != 0) {
                    if (noise_type == NoiseTypes::Wavefunction)
                        MovingEnvironment<S>::wavefunction_add_noise(old_wfn,
                                                                     noise);
                    else if (noise_type == NoiseTypes::Perturbative)
                        MovingEnvironment<S>::sacle_perturbative_noise(noise,
                                                                       pket);
                }
                error = MovingEnvironment<S>::split_wavefunction_svd(
                    opdq, old_wfn, (int)bond_dim, forward, true,
                    me->ket->tensors[i], me->ket->tensors[i + 1], cutoff,
                    trunc_type, decomp_type, pket);
            } else
                assert(false);
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                info = me->ket->tensors[i]->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'C';
            } else {
                info =
                    me->ket->tensors[i + 1]->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->right_dims[i + 1] = info;
                me->ket->info->save_right_dims(i + 1);
                me->ket->canonical_form[i] = 'C';
                me->ket->canonical_form[i + 1] = 'R';
            }
            info->deallocate();
            me->ket->save_tensor(i + 1);
            me->ket->save_tensor(i);
            me->ket->unload_tensor(i + 1);
            me->ket->unload_tensor(i);
            if (dm != nullptr) {
                dm->info->deallocate();
                dm->deallocate();
            }
            old_wfn->info->deallocate();
            old_wfn->deallocate();
            MovingEnvironment<S>::propagate_wfn(i, me->n_sites, me->ket,
                                                forward, me->mpo->tf->opf->cg);
        } else {
            if (pket != nullptr) {
                pket->deallocate();
                pket->deallocate_infos();
            }
            old_wfn->info->deallocate();
            old_wfn->deallocate();
            me->ket->tensors[i + 1] = make_shared<SparseMatrix<S>>();
            if (forward) {
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'C';
            } else {
                me->ket->canonical_form[i] = 'C';
                me->ket->canonical_form[i + 1] = 'R';
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(vector<double>{get<0>(pdi) + me->mpo->const_e}, error,
                         mmps, get<1>(pdi), get<2>(pdi), get<3>(pdi));
    }
    // State-averaged one-site algorithm
    // canonical form for wavefunction: J = left-fused, T = right-fused
    Iteration update_multi_one_dot(int i, bool forward, ubond_t bond_dim,
                                   double noise, double davidson_conv_thrd) {
        shared_ptr<MultiMPS<S>> mket =
            dynamic_pointer_cast<MultiMPS<S>>(me->ket);
        frame->activate(0);
        bool fuse_left = i <= me->n_sites / 2;
        if (mket->canonical_form[i] == 'M') {
            if (i == 0)
                mket->canonical_form[i] = 'J';
            else if (i == me->n_sites - 1)
                mket->canonical_form[i] = 'T';
            else
                assert(false);
        }
        // guess wavefunction
        // change to fused form for super-block hamiltonian
        // note that this switch exactly matches two-site conventional mpo
        // middle-site switch, so two-site conventional mpo can work
        mket->load_tensor(i);
        if ((fuse_left && mket->canonical_form[i] == 'T') ||
            (!fuse_left && mket->canonical_form[i] == 'J')) {
            vector<shared_ptr<SparseMatrixGroup<S>>> prev_wfns = mket->wfns;
            if (fuse_left && mket->canonical_form[i] == 'T')
                mket->wfns = MovingEnvironment<S>::swap_multi_wfn_to_fused_left(
                    i, mket->info, prev_wfns, me->mpo->tf->opf->cg);
            else if (!fuse_left && mket->canonical_form[i] == 'J')
                mket->wfns =
                    MovingEnvironment<S>::swap_multi_wfn_to_fused_right(
                        i, mket->info, prev_wfns, me->mpo->tf->opf->cg);
            for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                prev_wfns[j]->deallocate();
            if (prev_wfns.size() != 0)
                prev_wfns[0]->deallocate_infos();
        }
        S opdq = me->mpo->op->q_label;
        int mmps = 0;
        double error = 0.0;
        tuple<vector<double>, int, size_t, double> pdi;
        vector<vector<pair<S, double>>> mps_quanta(mket->nroots);
        // effective hamiltonian
        if (davidson_soft_max_iter != 0 || noise != 0) {
            shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>> h_eff =
                me->multi_eff_ham(
                    fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, true);
            pdi = h_eff->eigs(iprint >= 3, davidson_conv_thrd,
                              davidson_max_iter, me->para_rule);
            for (int i = 0; i < mket->nroots; i++) {
                mps_quanta[i] = h_eff->ket[i]->delta_quanta();
                mps_quanta[i].erase(
                    remove_if(mps_quanta[i].begin(), mps_quanta[i].end(),
                              [this](const pair<S, double> &p) {
                                  return p.second < this->quanta_cutoff;
                              }),
                    mps_quanta[i].end());
            }
            h_eff->deallocate();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(noise_type != NoiseTypes::Perturbative);
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            // change to fused form for splitting
            if (fuse_left != forward) {
                vector<shared_ptr<SparseMatrixGroup<S>>> prev_wfns = mket->wfns;
                if (!fuse_left && forward)
                    mket->wfns =
                        MovingEnvironment<S>::swap_multi_wfn_to_fused_left(
                            i, mket->info, prev_wfns, me->mpo->tf->opf->cg);
                else if (fuse_left && !forward)
                    mket->wfns =
                        MovingEnvironment<S>::swap_multi_wfn_to_fused_right(
                            i, mket->info, prev_wfns, me->mpo->tf->opf->cg);
                for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                    prev_wfns[j]->deallocate();
                if (prev_wfns.size() != 0)
                    prev_wfns[0]->deallocate_infos();
            }
            // splitting of wavefunction
            vector<shared_ptr<SparseMatrixGroup<S>>> old_wfns = mket->wfns,
                                                     new_wfns;
            shared_ptr<SparseMatrix<S>> dm, rot;
            dm = MovingEnvironment<S>::density_matrix_with_multi_target(
                opdq, mket->wfns, mket->weights, forward, noise, noise_type);
            error = MovingEnvironment<S>::multi_split_density_matrix(
                dm, mket->wfns, (int)bond_dim, forward, true, new_wfns, rot,
                cutoff, trunc_type);
            shared_ptr<StateInfo<S>> info = nullptr;
            // propagation
            if (forward) {
                mket->tensors[i] = rot;
                mket->save_tensor(i);
                info = rot->info->extract_state_info(forward);
                mmps = info->n_states_total;
                mket->info->left_dims[i + 1] = info;
                mket->info->save_left_dims(i + 1);
                info->deallocate();
                if (i != me->n_sites - 1) {
                    MovingEnvironment<S>::contract_multi_one_dot(
                        i + 1, new_wfns, mket, forward);
                    mket->save_wavefunction(i + 1);
                    mket->unload_wavefunction(i + 1);
                    mket->canonical_form[i] = 'L';
                    mket->canonical_form[i + 1] = 'T';
                } else {
                    mket->tensors[i] = make_shared<SparseMatrix<S>>();
                    MovingEnvironment<S>::contract_multi_one_dot(
                        i, new_wfns, mket, !forward);
                    mket->save_wavefunction(i);
                    mket->unload_wavefunction(i);
                    mket->canonical_form[i] = 'J';
                }
            } else {
                mket->tensors[i] = rot;
                mket->save_tensor(i);
                info = rot->info->extract_state_info(forward);
                mmps = info->n_states_total;
                mket->info->right_dims[i] = info;
                mket->info->save_right_dims(i);
                info->deallocate();
                if (i > 0) {
                    MovingEnvironment<S>::contract_multi_one_dot(
                        i - 1, new_wfns, mket, forward);
                    mket->save_wavefunction(i - 1);
                    mket->unload_wavefunction(i - 1);
                    mket->canonical_form[i - 1] = 'J';
                    mket->canonical_form[i] = 'R';
                } else {
                    mket->tensors[i] = make_shared<SparseMatrix<S>>();
                    MovingEnvironment<S>::contract_multi_one_dot(
                        i, new_wfns, mket, !forward);
                    mket->save_wavefunction(i);
                    mket->unload_wavefunction(i);
                    mket->canonical_form[i] = 'T';
                }
            }
            if (forward) {
                for (int j = (int)new_wfns.size() - 1; j >= 0; j--)
                    new_wfns[j]->deallocate();
                if (new_wfns.size() != 0)
                    new_wfns[0]->deallocate_infos();
                rot->info->deallocate();
                rot->deallocate();
            } else {
                rot->info->deallocate();
                rot->deallocate();
                for (int j = (int)new_wfns.size() - 1; j >= 0; j--)
                    new_wfns[j]->deallocate();
                if (new_wfns.size() != 0)
                    new_wfns[0]->deallocate_infos();
            }
            dm->info->deallocate();
            dm->deallocate();

            for (int j = (int)old_wfns.size() - 1; j >= 0; j--)
                old_wfns[j]->deallocate();
            if (old_wfns.size() != 0)
                old_wfns[0]->deallocate_infos();
        } else {
            mket->unload_tensor(i);
            if (forward) {
                if (i != me->n_sites - 1) {
                    mket->tensors[i] = make_shared<SparseMatrix<S>>();
                    mket->tensors[i + 1] = nullptr;
                    mket->canonical_form[i] = 'L';
                    mket->canonical_form[i + 1] = 'T';
                } else
                    mket->canonical_form[i] = 'J';
            } else {
                if (i > 0) {
                    mket->tensors[i - 1] = nullptr;
                    mket->tensors[i] = make_shared<SparseMatrix<S>>();
                    mket->canonical_form[i - 1] = 'J';
                    mket->canonical_form[i] = 'R';
                } else
                    mket->canonical_form[i] = 'T';
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        for (auto &x : get<0>(pdi))
            x += me->mpo->const_e;
        Iteration r = Iteration(get<0>(pdi), error, mmps, get<1>(pdi),
                                get<2>(pdi), get<3>(pdi));
        r.quanta = mps_quanta;
        return r;
    }
    // State-averaged two-site algorithm
    // canonical form for wavefunction: M = multi center
    Iteration update_multi_two_dot(int i, bool forward, ubond_t bond_dim,
                                   double noise, double davidson_conv_thrd) {
        shared_ptr<MultiMPS<S>> mket =
            dynamic_pointer_cast<MultiMPS<S>>(me->ket);
        frame->activate(0);
        if (mket->tensors[i] != nullptr || mket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_multi_two_dot(i, mket);
        else
            mket->load_tensor(i);
        mket->tensors[i] = mket->tensors[i + 1] = nullptr;
        vector<shared_ptr<SparseMatrixGroup<S>>> old_wfns = mket->wfns;
        S opdq = me->mpo->op->q_label;
        // effective hamiltonian
        int mmps = 0;
        double error = 0.0;
        tuple<vector<double>, int, size_t, double> pdi;
        vector<vector<pair<S, double>>> mps_quanta(mket->nroots);
        if (davidson_soft_max_iter != 0 || noise != 0) {
            shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>> h_eff =
                me->multi_eff_ham(FuseTypes::FuseLR, true);
            pdi = h_eff->eigs(iprint >= 3, davidson_conv_thrd,
                              davidson_max_iter, me->para_rule);
            for (int i = 0; i < mket->nroots; i++) {
                mps_quanta[i] = h_eff->ket[i]->delta_quanta();
                mps_quanta[i].erase(
                    remove_if(mps_quanta[i].begin(), mps_quanta[i].end(),
                              [this](const pair<S, double> &p) {
                                  return p.second < this->quanta_cutoff;
                              }),
                    mps_quanta[i].end());
            }
            h_eff->deallocate();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(noise_type != NoiseTypes::Perturbative);
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            shared_ptr<SparseMatrix<S>> dm;
            dm = MovingEnvironment<S>::density_matrix_with_multi_target(
                opdq, old_wfns, mket->weights, forward, noise, noise_type);
            error = MovingEnvironment<S>::multi_split_density_matrix(
                dm, old_wfns, (int)bond_dim, forward, true, mket->wfns,
                forward ? mket->tensors[i] : mket->tensors[i + 1], cutoff,
                trunc_type);
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                info = me->ket->tensors[i]->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'M';
            } else {
                info =
                    me->ket->tensors[i + 1]->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->right_dims[i + 1] = info;
                me->ket->info->save_right_dims(i + 1);
                me->ket->canonical_form[i] = 'M';
                me->ket->canonical_form[i + 1] = 'R';
            }
            info->deallocate();
            if (forward) {
                mket->save_wavefunction(i + 1);
                mket->save_tensor(i);
                mket->unload_wavefunction(i + 1);
                mket->unload_tensor(i);
            } else {
                mket->save_tensor(i + 1);
                mket->save_wavefunction(i);
                mket->unload_tensor(i + 1);
                mket->unload_wavefunction(i);
            }
            dm->info->deallocate();
            dm->deallocate();
            for (int k = mket->nroots - 1; k >= 0; k--)
                old_wfns[k]->deallocate();
            old_wfns[0]->deallocate_infos();
            MovingEnvironment<S>::propagate_multi_wfn(
                i, me->n_sites, mket, forward, me->mpo->tf->opf->cg);
        } else {
            for (int k = mket->nroots - 1; k >= 0; k--)
                old_wfns[k]->deallocate();
            old_wfns[0]->deallocate_infos();
            if (forward) {
                me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                me->ket->tensors[i + 1] = nullptr;
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'M';
            } else {
                me->ket->tensors[i] = nullptr;
                me->ket->tensors[i + 1] = make_shared<SparseMatrix<S>>();
                me->ket->canonical_form[i] = 'M';
                me->ket->canonical_form[i + 1] = 'R';
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        for (auto &x : get<0>(pdi))
            x += me->mpo->const_e;
        Iteration r = Iteration(get<0>(pdi), error, mmps, get<1>(pdi),
                                get<2>(pdi), get<3>(pdi));
        r.quanta = mps_quanta;
        return r;
    }
    virtual Iteration blocking(int i, bool forward, ubond_t bond_dim,
                               double noise, double davidson_conv_thrd) {
        me->move_to(i);
        assert(me->dot == 1 || me->dot == 2);
        if (me->dot == 2) {
            if (me->ket->canonical_form[i] == 'M' ||
                me->ket->canonical_form[i + 1] == 'M')
                return update_multi_two_dot(i, forward, bond_dim, noise,
                                            davidson_conv_thrd);
            else
                return update_two_dot(i, forward, bond_dim, noise,
                                      davidson_conv_thrd);
        } else {
            if (me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T')
                return update_multi_one_dot(i, forward, bond_dim, noise,
                                            davidson_conv_thrd);
            else
                return update_one_dot(i, forward, bond_dim, noise,
                                      davidson_conv_thrd);
        }
    }
    tuple<vector<double>, double, vector<vector<pair<S, double>>>>
    sweep(bool forward, ubond_t bond_dim, double noise,
          double davidson_conv_thrd) {
        me->prepare();
        vector<vector<double>> energies;
        vector<double> discarded_weights;
        vector<vector<vector<pair<S, double>>>> quanta;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r =
                blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            energies.push_back(r.energies);
            discarded_weights.push_back(r.error);
            quanta.push_back(r.quanta);
        }
        size_t idx =
            min_element(energies.begin(), energies.end(),
                        [](const vector<double> &x, const vector<double> &y) {
                            return x[0] < y[0];
                        }) -
            energies.begin();
        return make_tuple(energies[idx], discarded_weights[idx], quanta[idx]);
    }
    double solve(int n_sweeps, bool forward = true, double tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        if (davidson_conv_thrds.size() < n_sweeps)
            for (size_t i = davidson_conv_thrds.size(); i < noises.size(); i++)
                davidson_conv_thrds.push_back(
                    (noises[i] == 0 ? (tol == 0 ? 1E-9 : tol) : noises[i]) *
                    0.1);
        Timer start, current;
        start.get_time();
        energies.clear();
        discarded_weights.clear();
        mps_quanta.clear();
        bool converged;
        double energy_difference;
        for (int iw = 0; iw < n_sweeps; iw++) {
            if (iprint >= 1)
                cout << "Sweep = " << setw(4) << iw
                     << " | Direction = " << setw(8)
                     << (forward ? "forward" : "backward")
                     << " | Bond dimension = " << setw(4)
                     << (uint32_t)bond_dims[iw] << " | Noise = " << scientific
                     << setw(9) << setprecision(2) << noises[iw]
                     << " | Dav threshold = " << scientific << setw(9)
                     << setprecision(2) << davidson_conv_thrds[iw] << endl;
            auto sweep_results = sweep(forward, bond_dims[iw], noises[iw],
                                       davidson_conv_thrds[iw]);
            energies.push_back(get<0>(sweep_results));
            discarded_weights.push_back(get<1>(sweep_results));
            mps_quanta.push_back(get<2>(sweep_results));
            if (energies.size() >= 2)
                energy_difference = energies[energies.size() - 1].back() -
                                    energies[energies.size() - 2].back();
            converged = energies.size() >= 2 && tol > 0 &&
                        abs(energy_difference) < tol &&
                        noises[iw] == noises.back() &&
                        bond_dims[iw] == bond_dims.back();
            forward = !forward;
            current.get_time();
            if (iprint >= 1) {
                cout << "Time elapsed = " << setw(10) << setprecision(3)
                     << current.current - start.current;
                cout << fixed << setprecision(8);
                if (get<0>(sweep_results).size() == 1)
                    cout << " | E = " << setw(15) << get<0>(sweep_results)[0];
                else {
                    cout << " | E[" << setw(3) << get<0>(sweep_results).size()
                         << "] = ";
                    for (double x : get<0>(sweep_results))
                        cout << setw(15) << x;
                }
                if (energies.size() >= 2)
                    cout << " | DE = " << setw(6) << setprecision(2)
                         << scientific << energy_difference;
                cout << endl;
            }

            if (converged)
                break;
        }
        this->forward = forward;
        if (!converged && iprint > 0)
            cout << "ATTENTION: DMRG is not converged to desired tolerance of "
                 << scientific << tol << endl;
        return energies.back()[0];
    }
};

enum struct TETypes : uint8_t { TangentSpace, RK4 };

enum struct TruncPatternTypes : uint8_t { None, TruncAfterOdd, TruncAfterEven };

// Imaginary Time Evolution
template <typename S> struct ImaginaryTE {
    shared_ptr<MovingEnvironment<S>> me;
    vector<ubond_t> bond_dims;
    vector<double> noises;
    vector<double> errors;
    vector<double> energies;
    vector<double> normsqs;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    TruncPatternTypes trunc_pattern = TruncPatternTypes::None;
    bool forward;
    TETypes mode;
    int n_sub_sweeps;
    vector<double> weights = {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0};
    uint8_t iprint = 2;
    double cutoff = 1E-14;
    ImaginaryTE(const shared_ptr<MovingEnvironment<S>> &me,
                const vector<ubond_t> &bond_dims,
                TETypes mode = TETypes::TangentSpace, int n_sub_sweeps = 1)
        : me(me), bond_dims(bond_dims), noises(vector<double>{0.0}),
          forward(false), mode(mode), n_sub_sweeps(n_sub_sweeps) {}
    struct Iteration {
        double energy, normsq, error;
        int nexpo, nexpok, mmps;
        double texpo;
        size_t nflop;
        Iteration(double energy, double normsq, double error, int mmps,
                  int nexpo, int nexpok, size_t nflop = 0, double texpo = 1.0)
            : energy(energy), normsq(normsq), error(error), mmps(mmps),
              nexpo(nexpo), nexpok(nexpok), nflop(nflop), texpo(texpo) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Mmps =" << setw(5) << r.mmps;
            os << " Nexpo = " << setw(4) << r.nexpo << "/" << setw(4)
               << r.nexpok << " E = " << setw(17) << setprecision(10)
               << r.energy << " Error = " << scientific << setw(8)
               << setprecision(2) << r.error << " FLOPS = " << scientific
               << setw(8) << setprecision(2) << (double)r.nflop / r.texpo
               << " Texpo = " << fixed << setprecision(2) << r.texpo;
            return os;
        }
    };
    // one-site algorithm
    Iteration update_one_dot(int i, bool forward, bool advance, double beta,
                             ubond_t bond_dim, double noise) {
        frame->activate(0);
        bool fuse_left = i <= me->n_sites / 2;
        if (me->ket->canonical_form[i] == 'C') {
            if (i == 0)
                me->ket->canonical_form[i] = 'K';
            else if (i == me->n_sites - 1)
                me->ket->canonical_form[i] = 'S';
            else
                assert(false);
        }
        // guess wavefunction
        // change to fused form for super-block hamiltonian
        // note that this switch exactly matches two-site conventional mpo
        // middle-site switch, so two-site conventional mpo can work
        me->ket->load_tensor(i);
        if ((fuse_left && me->ket->canonical_form[i] == 'S') ||
            (!fuse_left && me->ket->canonical_form[i] == 'K')) {
            shared_ptr<SparseMatrix<S>> prev_wfn = me->ket->tensors[i];
            if (fuse_left && me->ket->canonical_form[i] == 'S')
                me->ket->tensors[i] =
                    MovingEnvironment<S>::swap_wfn_to_fused_left(
                        i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
            else if (!fuse_left && me->ket->canonical_form[i] == 'K')
                me->ket->tensors[i] =
                    MovingEnvironment<S>::swap_wfn_to_fused_right(
                        i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
            prev_wfn->info->deallocate();
            prev_wfn->deallocate();
        }
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, true,
                        me->bra->tensors[i], me->ket->tensors[i]);
        tuple<double, double, int, size_t, double> pdi;
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<MatrixRef> pdpf;
        if (!advance &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            // TangentSpace method does not allow multiple sweeps for one time
            // step
            assert(mode == TETypes::RK4);
            MatrixRef tmp(nullptr, h_eff->ket->total_memory, 1);
            tmp.allocate();
            memcpy(tmp.data, h_eff->ket->data,
                   h_eff->ket->total_memory * sizeof(double));
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, iprint >= 3,
                                    me->para_rule);
            memcpy(h_eff->ket->data, tmp.data,
                   h_eff->ket->total_memory * sizeof(double));
            tmp.deallocate();
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
        } else if (effective_mode == TETypes::TangentSpace)
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, iprint >= 3,
                                    me->para_rule);
        else if (effective_mode == TETypes::RK4) {
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
            pdi = pdp.second;
        }
        h_eff->deallocate();
        int bdim = bond_dim, mmps = 0, expok = 0;
        double error = 0.0;
        shared_ptr<SparseMatrix<S>> dm = nullptr;
        shared_ptr<SparseMatrix<S>> old_wfn;
        shared_ptr<SparseMatrix<S>> left = nullptr, right = nullptr;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                shared_ptr<SparseMatrix<S>> prev_wfn = me->ket->tensors[i];
                if (!fuse_left && forward)
                    me->ket->tensors[i] =
                        MovingEnvironment<S>::swap_wfn_to_fused_left(
                            i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
                else if (fuse_left && !forward)
                    me->ket->tensors[i] =
                        MovingEnvironment<S>::swap_wfn_to_fused_right(
                            i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
                if (pdpf.size() != 0) {
                    shared_ptr<SparseMatrix<S>> tmp_wfn;
                    for (size_t ip = 0; ip < pdpf.size(); ip++) {
                        assert(pdpf[ip].size() == prev_wfn->total_memory);
                        memcpy(prev_wfn->data, pdpf[ip].data,
                               pdpf[ip].size() * sizeof(double));
                        if (!fuse_left && forward)
                            tmp_wfn =
                                MovingEnvironment<S>::swap_wfn_to_fused_left(
                                    i, me->ket->info, prev_wfn,
                                    me->mpo->tf->opf->cg);
                        else if (fuse_left && !forward)
                            tmp_wfn =
                                MovingEnvironment<S>::swap_wfn_to_fused_right(
                                    i, me->ket->info, prev_wfn,
                                    me->mpo->tf->opf->cg);
                        assert(pdpf[ip].size() == tmp_wfn->total_memory);
                        memcpy(pdpf[ip].data, tmp_wfn->data,
                               pdpf[ip].size() * sizeof(double));
                        tmp_wfn->info->deallocate();
                        tmp_wfn->deallocate();
                    }
                }
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
            }
            if (pdpf.size() != 0) {
                dm = MovingEnvironment<S>::density_matrix_with_weights(
                    h_eff->opdq, me->ket->tensors[i], forward, noise, pdpf,
                    weights, noise_type);
                frame->activate(1);
                for (int i = pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame->activate(0);
            } else
                dm = MovingEnvironment<S>::density_matrix(
                    h_eff->opdq, me->ket->tensors[i], forward, noise,
                    noise_type);
            // splitting of wavefunction
            old_wfn = me->ket->tensors[i];
            if ((this->trunc_pattern == TruncPatternTypes::TruncAfterOdd &&
                 i % 2 == 0) ||
                (this->trunc_pattern == TruncPatternTypes::TruncAfterEven &&
                 i % 2 == 1))
                bdim = -1;
            error = MovingEnvironment<S>::split_density_matrix(
                dm, me->ket->tensors[i], bdim, forward, false, left, right,
                cutoff, trunc_type);
        } else {
            if (pdpf.size() != 0) {
                frame->activate(1);
                for (int i = pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame->activate(0);
            }
            old_wfn = me->ket->tensors[i];
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                if (mode == TETypes::RK4 && (i != me->n_sites - 1 || !advance))
                    right->normalize();
                me->ket->tensors[i] = left;
                me->ket->save_tensor(i);
                info = left->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
            } else {
                if (mode == TETypes::RK4 && (i != 0 || !advance))
                    left->normalize();
                me->ket->tensors[i] = right;
                me->ket->save_tensor(i);
                info = right->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->right_dims[i] = info;
                me->ket->info->save_right_dims(i);
            }
            info->deallocate();
        }
        if (mode == TETypes::TangentSpace &&
            ((forward && i != me->n_sites - 1) || (!forward && i != 0))) {
            if (me->para_rule != nullptr) {
                if (me->para_rule->is_root()) {
                    if (forward)
                        right->save_data(me->ket->get_filename(-2), true);
                    else
                        left->save_data(me->ket->get_filename(-2), true);
                }
                me->para_rule->comm->barrier();
                if (!me->para_rule->is_root()) {
                    if (forward) {
                        right = make_shared<SparseMatrix<S>>();
                        right->load_data(me->ket->get_filename(-2), true);
                    } else {
                        left = make_shared<SparseMatrix<S>>();
                        left->load_data(me->ket->get_filename(-2), true);
                    }
                }
            }
            if (forward) {
                me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                me->move_to(i + 1);
                shared_ptr<EffectiveHamiltonian<S>> k_eff =
                    me->eff_ham(FuseTypes::NoFuseL, true, right, right);
                auto pdk = k_eff->expo_apply(beta, me->mpo->const_e,
                                             iprint >= 3, me->para_rule);
                k_eff->deallocate();
                if (me->para_rule == nullptr || me->para_rule->is_root()) {
                    right->normalize();
                    get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
                    expok = get<2>(pdk);
                    MovingEnvironment<S>::contract_one_dot(i + 1, right,
                                                           me->ket, forward);
                    me->ket->save_tensor(i + 1);
                    me->ket->unload_tensor(i + 1);
                }
            } else {
                me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                me->move_to(i - 1);
                shared_ptr<EffectiveHamiltonian<S>> k_eff =
                    me->eff_ham(FuseTypes::NoFuseR, true, left, left);
                auto pdk = k_eff->expo_apply(beta, me->mpo->const_e,
                                             iprint >= 3, me->para_rule);
                k_eff->deallocate();
                if (me->para_rule == nullptr || me->para_rule->is_root()) {
                    left->normalize();
                    get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
                    expok = get<2>(pdk);
                    MovingEnvironment<S>::contract_one_dot(i - 1, left, me->ket,
                                                           forward);
                    me->ket->save_tensor(i - 1);
                    me->ket->unload_tensor(i - 1);
                }
            }
        } else {
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                // propagation
                if (forward) {
                    if (i != me->n_sites - 1) {
                        MovingEnvironment<S>::contract_one_dot(
                            i + 1, right, me->ket, forward);
                        me->ket->save_tensor(i + 1);
                        me->ket->unload_tensor(i + 1);
                    } else {
                        me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                        MovingEnvironment<S>::contract_one_dot(
                            i, right, me->ket, !forward);
                        me->ket->save_tensor(i);
                        me->ket->unload_tensor(i);
                    }
                } else {
                    if (i > 0) {
                        MovingEnvironment<S>::contract_one_dot(
                            i - 1, left, me->ket, forward);
                        me->ket->save_tensor(i - 1);
                        me->ket->unload_tensor(i - 1);
                    } else {
                        me->ket->tensors[i] = make_shared<SparseMatrix<S>>();
                        MovingEnvironment<S>::contract_one_dot(i, left, me->ket,
                                                               !forward);
                        me->ket->save_tensor(i);
                        me->ket->unload_tensor(i);
                    }
                }
            }
        }
        if (right != nullptr) {
            right->info->deallocate();
            right->deallocate();
        }
        if (left != nullptr) {
            left->info->deallocate();
            left->deallocate();
        }
        if (dm != nullptr) {
            dm->info->deallocate();
            dm->deallocate();
        }
        old_wfn->info->deallocate();
        old_wfn->deallocate();
        if (forward) {
            if (i != me->n_sites - 1) {
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'S';
            } else
                me->ket->canonical_form[i] = 'K';
        } else {
            if (i > 0) {
                me->ket->canonical_form[i - 1] = 'K';
                me->ket->canonical_form[i] = 'R';
            } else
                me->ket->canonical_form[i] = 'S';
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, mmps, get<2>(pdi),
                         expok, get<3>(pdi), get<4>(pdi));
    }
    // two-site algorithm
    Iteration update_two_dot(int i, bool forward, bool advance, double beta,
                             ubond_t bond_dim, double noise) {
        frame->activate(0);
        if (me->ket->tensors[i] != nullptr &&
            me->ket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_two_dot(i, me->ket);
        else {
            me->ket->load_tensor(i);
            me->ket->tensors[i + 1] = nullptr;
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff = me->eff_ham(
            FuseTypes::FuseLR, true, me->bra->tensors[i], me->ket->tensors[i]);
        tuple<double, double, int, size_t, double> pdi;
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<MatrixRef> pdpf;
        if (!advance &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            // TangentSpace method does not allow multiple sweeps for one time
            // step
            assert(mode == TETypes::RK4);
            MatrixRef tmp(nullptr, h_eff->ket->total_memory, 1);
            tmp.allocate();
            memcpy(tmp.data, h_eff->ket->data,
                   h_eff->ket->total_memory * sizeof(double));
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, iprint >= 3,
                                    me->para_rule);
            memcpy(h_eff->ket->data, tmp.data,
                   h_eff->ket->total_memory * sizeof(double));
            tmp.deallocate();
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
        } else if (effective_mode == TETypes::TangentSpace)
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, iprint >= 3,
                                    me->para_rule);
        else if (effective_mode == TETypes::RK4) {
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
            pdi = pdp.second;
        }
        h_eff->deallocate();
        int bdim = bond_dim, mmps = 0;
        double error = 0.0;
        shared_ptr<SparseMatrix<S>> dm;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (pdpf.size() != 0) {
                dm = MovingEnvironment<S>::density_matrix_with_weights(
                    h_eff->opdq, h_eff->ket, forward, noise, pdpf, weights,
                    noise_type);
                frame->activate(1);
                for (int i = pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame->activate(0);
            } else
                dm = MovingEnvironment<S>::density_matrix(
                    h_eff->opdq, h_eff->ket, forward, noise, noise_type);
            if ((this->trunc_pattern == TruncPatternTypes::TruncAfterOdd &&
                 i % 2 == 0) ||
                (this->trunc_pattern == TruncPatternTypes::TruncAfterEven &&
                 i % 2 == 1))
                bdim = -1;
            error = MovingEnvironment<S>::split_density_matrix(
                dm, h_eff->ket, bdim, forward, false, me->ket->tensors[i],
                me->ket->tensors[i + 1], cutoff, trunc_type);
        } else {
            if (pdpf.size() != 0) {
                frame->activate(1);
                for (int i = pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame->activate(0);
            }
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                if (mode == TETypes::RK4 &&
                    (i + 1 != me->n_sites - 1 || !advance))
                    me->ket->tensors[i + 1]->normalize();
                info = me->ket->tensors[i]->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'C';
            } else {
                if (mode == TETypes::RK4 && (i != 0 || !advance))
                    me->ket->tensors[i]->normalize();
                info =
                    me->ket->tensors[i + 1]->info->extract_state_info(forward);
                mmps = info->n_states_total;
                me->ket->info->right_dims[i + 1] = info;
                me->ket->info->save_right_dims(i + 1);
                me->ket->canonical_form[i] = 'C';
                me->ket->canonical_form[i + 1] = 'R';
            }
            info->deallocate();
            me->ket->save_tensor(i + 1);
            me->ket->save_tensor(i);
            me->ket->unload_tensor(i + 1);
            me->ket->unload_tensor(i);
            dm->info->deallocate();
            dm->deallocate();
        } else {
            me->ket->tensors[i + 1] = make_shared<SparseMatrix<S>>();
            if (forward) {
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'C';
            } else {
                me->ket->canonical_form[i] = 'C';
                me->ket->canonical_form[i + 1] = 'R';
            }
        }
        old_wfn->info->deallocate();
        old_wfn->deallocate();
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        int expok = 0;
        if (mode == TETypes::TangentSpace && forward &&
            i + 1 != me->n_sites - 1) {
            me->move_to(i + 1);
            me->ket->load_tensor(i + 1);
            shared_ptr<EffectiveHamiltonian<S>> k_eff =
                me->eff_ham(FuseTypes::FuseR, true, me->bra->tensors[i + 1],
                            me->ket->tensors[i + 1]);
            auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, iprint >= 3,
                                         me->para_rule);
            k_eff->deallocate();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                me->ket->tensors[i + 1]->normalize();
                me->ket->save_tensor(i + 1);
            }
            me->ket->unload_tensor(i + 1);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        } else if (mode == TETypes::TangentSpace && !forward && i != 0) {
            me->move_to(i - 1);
            me->ket->load_tensor(i);
            shared_ptr<EffectiveHamiltonian<S>> k_eff =
                me->eff_ham(FuseTypes::FuseL, true, me->bra->tensors[i],
                            me->ket->tensors[i]);
            auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, iprint >= 3,
                                         me->para_rule);
            k_eff->deallocate();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                me->ket->tensors[i]->normalize();
                me->ket->save_tensor(i);
            }
            me->ket->unload_tensor(i);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        }
        if (me->para_rule == nullptr || me->para_rule->is_root())
            MovingEnvironment<S>::propagate_wfn(i, me->n_sites, me->ket,
                                                forward, me->mpo->tf->opf->cg);
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, mmps, get<2>(pdi),
                         expok, get<3>(pdi), get<4>(pdi));
    }
    Iteration blocking(int i, bool forward, bool advance, double beta,
                       ubond_t bond_dim, double noise) {
        me->move_to(i);
        assert(me->dot == 2 || me->dot == 1);
        if (me->dot == 2)
            return update_two_dot(i, forward, advance, beta, bond_dim, noise);
        else
            return update_one_dot(i, forward, advance, beta, bond_dim, noise);
    }
    tuple<double, double, double> sweep(bool forward, bool advance, double beta,
                                        ubond_t bond_dim, double noise) {
        me->prepare();
        vector<double> energies, normsqs;
        vector<int> sweep_range;
        double largest_error = 0.0;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r = blocking(i, forward, advance, beta, bond_dim, noise);
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            energies.push_back(r.energy);
            normsqs.push_back(r.normsq);
            largest_error = max(largest_error, r.error);
        }
        return make_tuple(energies.back(), normsqs.back(), largest_error);
    }
    void normalize() {
        size_t center = me->ket->canonical_form.find('C');
        if (center == string::npos)
            center = me->ket->canonical_form.find('K');
        if (center == string::npos)
            center = me->ket->canonical_form.find('S');
        assert(center != string::npos);
        me->ket->load_tensor(center);
        me->ket->tensors[center]->normalize();
        me->ket->save_tensor(center);
        me->ket->unload_tensor(center);
    }
    double solve(int n_sweeps, double beta, bool forward = true,
                 double tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        energies.clear();
        normsqs.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            for (int isw = 0; isw < n_sub_sweeps; isw++) {
                if (iprint >= 1) {
                    cout << "Sweep = " << setw(4) << iw;
                    if (n_sub_sweeps != 1)
                        cout << " (" << setw(2) << isw << "/" << setw(2)
                             << (int)n_sub_sweeps << ")";
                    cout << " | Direction = " << setw(8)
                         << (forward ? "forward" : "backward")
                         << " | Beta = " << fixed << setw(10) << setprecision(5)
                         << beta << " | Bond dimension = " << setw(4)
                         << (uint32_t)bond_dims[iw]
                         << " | Noise = " << scientific << setw(9)
                         << setprecision(2) << noises[iw] << endl;
                }
                auto r = sweep(forward, isw == n_sub_sweeps - 1, beta,
                               bond_dims[iw], noises[iw]);
                forward = !forward;
                current.get_time();
                if (iprint == 1) {
                    cout << fixed << setprecision(8);
                    cout << " .. Energy = " << setw(15) << get<0>(r)
                         << " Norm = " << setw(15) << sqrt(get<1>(r))
                         << " MaxError = " << setw(15) << setprecision(12)
                         << get<2>(r) << " ";
                }
                if (iprint >= 1)
                    cout << "Time elapsed = " << setw(10) << setprecision(3)
                         << current.current - start.current << endl;
                if (isw == n_sub_sweeps - 1) {
                    energies.push_back(get<0>(r));
                    normsqs.push_back(get<1>(r));
                }
            }
            normalize();
        }
        this->forward = forward;
        return energies.back();
    }
};

// Compression
template <typename S> struct Compress {
    shared_ptr<MovingEnvironment<S>> me;
    vector<ubond_t> bra_bond_dims, ket_bond_dims;
    vector<double> noises;
    vector<double> norms;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    bool forward;
    uint8_t iprint = 2;
    double cutoff = 1E-14;
    bool decomp_last_site = true;
    Compress(const shared_ptr<MovingEnvironment<S>> &me,
             const vector<ubond_t> &bra_bond_dims,
             const vector<ubond_t> &ket_bond_dims,
             const vector<double> &noises = vector<double>())
        : me(me), bra_bond_dims(bra_bond_dims), ket_bond_dims(ket_bond_dims),
          noises(noises), forward(false) {}
    struct Iteration {
        int mmps;
        double norm, error;
        double tmult;
        size_t nflop;
        Iteration(double norm, double error, int mmps, size_t nflop = 0,
                  double tmult = 1.0)
            : norm(norm), error(error), mmps(mmps), nflop(nflop), tmult(tmult) {
        }
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Mmps =" << setw(5) << r.mmps;
            os << " Norm = " << setw(15) << r.norm << " Error = " << setw(15)
               << setprecision(12) << r.error << " FLOPS = " << scientific
               << setw(8) << setprecision(2) << (double)r.nflop / r.tmult
               << " Tmult = " << fixed << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_one_dot(int i, bool forward, ubond_t bra_bond_dim,
                             ubond_t ket_bond_dim, double noise) {
        assert(me->bra != me->ket);
        frame->activate(0);
        bool fuse_left = i <= me->n_sites / 2;
        for (auto &mps : {me->bra, me->ket}) {
            if (mps->canonical_form[i] == 'C') {
                if (i == 0)
                    mps->canonical_form[i] = 'K';
                else if (i == me->n_sites - 1)
                    mps->canonical_form[i] = 'S';
                else
                    assert(false);
            }
            // guess wavefunction
            // change to fused form for super-block hamiltonian
            // note that this switch exactly matches two-site conventional mpo
            // middle-site switch, so two-site conventional mpo can work
            mps->load_tensor(i);
            if ((fuse_left && mps->canonical_form[i] == 'S') ||
                (!fuse_left && mps->canonical_form[i] == 'K')) {
                shared_ptr<SparseMatrix<S>> prev_wfn = mps->tensors[i];
                if (fuse_left && mps->canonical_form[i] == 'S')
                    mps->tensors[i] =
                        MovingEnvironment<S>::swap_wfn_to_fused_left(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'K')
                    mps->tensors[i] =
                        MovingEnvironment<S>::swap_wfn_to_fused_right(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
            }
        }
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, false,
                        me->bra->tensors[i], me->ket->tensors[i]);
        auto pdi = h_eff->multiply(me->para_rule);
        h_eff->deallocate();
        double bra_error = 0.0;
        int bra_mmps = 0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (!decomp_last_site &&
                ((forward && i == me->n_sites - 1 && !fuse_left) ||
                 (!forward && i == 0 && fuse_left))) {
                for (auto &mps : {me->bra, me->ket}) {
                    mps->save_tensor(i);
                    mps->unload_tensor(i);
                    mps->canonical_form[i] = forward ? 'S' : 'K';
                }
            } else {
                // change to fused form for splitting
                if (fuse_left != forward) {
                    for (auto &mps : {me->bra, me->ket}) {
                        shared_ptr<SparseMatrix<S>> prev_wfn = mps->tensors[i];
                        if (!fuse_left && forward)
                            mps->tensors[i] =
                                MovingEnvironment<S>::swap_wfn_to_fused_left(
                                    i, mps->info, prev_wfn,
                                    me->mpo->tf->opf->cg);
                        else if (fuse_left && !forward)
                            mps->tensors[i] =
                                MovingEnvironment<S>::swap_wfn_to_fused_right(
                                    i, mps->info, prev_wfn,
                                    me->mpo->tf->opf->cg);
                        prev_wfn->info->deallocate();
                        prev_wfn->deallocate();
                    }
                }
                shared_ptr<SparseMatrix<S>> old_bra = me->bra->tensors[i];
                shared_ptr<SparseMatrix<S>> old_ket = me->ket->tensors[i];
                for (auto &mps : {me->bra, me->ket}) {
                    // splitting of wavefunction
                    shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                    shared_ptr<SparseMatrix<S>> left, right;
                    shared_ptr<SparseMatrix<S>> dm = nullptr;
                    int bond_dim = mps == me->bra ? (int)bra_bond_dim
                                                  : (int)ket_bond_dim,
                        error;
                    if (decomp_type == DecompositionTypes::DensityMatrix) {
                        dm = MovingEnvironment<S>::density_matrix(
                            h_eff->opdq, old_wfn, forward,
                            mps == me->bra ? noise : 0.0,
                            mps == me->bra && noise != 0 ? noise_type
                                                         : NoiseTypes::None);
                        error = MovingEnvironment<S>::split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false, left, right,
                            cutoff, trunc_type);
                    } else if (decomp_type == DecompositionTypes::SVD ||
                               decomp_type == DecompositionTypes::PureSVD) {
                        if (noise != 0)
                            MovingEnvironment<S>::wavefunction_add_noise(
                                old_wfn, mps == me->bra ? noise : 0.0);
                        error = MovingEnvironment<S>::split_wavefunction_svd(
                            h_eff->opdq, old_wfn, bond_dim, forward, false,
                            left, right, cutoff, trunc_type, decomp_type);
                    } else
                        assert(false);
                    if (mps == me->bra)
                        bra_error = error;
                    shared_ptr<StateInfo<S>> info = nullptr;
                    // propagation
                    if (forward) {
                        mps->tensors[i] = left;
                        mps->save_tensor(i);
                        info = left->info->extract_state_info(forward);
                        if (mps == me->bra)
                            bra_mmps = info->n_states_total;
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        info->deallocate();
                        if (i != me->n_sites - 1) {
                            MovingEnvironment<S>::contract_one_dot(
                                i + 1, right, mps, forward);
                            mps->save_tensor(i + 1);
                            mps->unload_tensor(i + 1);
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            MovingEnvironment<S>::contract_one_dot(
                                i, right, mps, !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                            mps->canonical_form[i] = 'K';
                        }
                    } else {
                        mps->tensors[i] = right;
                        mps->save_tensor(i);
                        info = right->info->extract_state_info(forward);
                        if (mps == me->bra)
                            bra_mmps = info->n_states_total;
                        mps->info->right_dims[i] = info;
                        mps->info->save_right_dims(i);
                        info->deallocate();
                        if (i > 0) {
                            MovingEnvironment<S>::contract_one_dot(
                                i - 1, left, mps, forward);
                            mps->save_tensor(i - 1);
                            mps->unload_tensor(i - 1);
                            mps->canonical_form[i - 1] = 'K';
                            mps->canonical_form[i] = 'R';
                        } else {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            MovingEnvironment<S>::contract_one_dot(i, left, mps,
                                                                   !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                            mps->canonical_form[i] = 'S';
                        }
                    }
                    right->info->deallocate();
                    right->deallocate();
                    left->info->deallocate();
                    left->deallocate();
                    if (dm != nullptr) {
                        dm->info->deallocate();
                        dm->deallocate();
                    }
                }
                for (auto &old_wfn : {old_ket, old_bra}) {
                    old_wfn->info->deallocate();
                    old_wfn->deallocate();
                }
            }
        } else {
            if (!decomp_last_site &&
                ((forward && i == me->n_sites - 1 && !fuse_left) ||
                 (!forward && i == 0 && fuse_left)))
                for (auto &mps : {me->bra, me->ket})
                    mps->canonical_form[i] = forward ? 'S' : 'K';
            else
                for (auto &mps : {me->bra, me->ket}) {
                    if (forward) {
                        if (i != me->n_sites - 1) {
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else
                            mps->canonical_form[i] = 'K';
                    } else {
                        if (i > 0) {
                            mps->canonical_form[i - 1] = 'K';
                            mps->canonical_form[i] = 'R';
                        } else
                            mps->canonical_form[i] = 'S';
                    }
                }
            me->ket->unload_tensor(i);
            me->bra->unload_tensor(i);
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi), bra_error, bra_mmps, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_two_dot(int i, bool forward, ubond_t bra_bond_dim,
                             ubond_t ket_bond_dim, double noise) {
        assert(me->bra != me->ket);
        frame->activate(0);
        for (auto &mps : {me->bra, me->ket}) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S>::contract_two_dot(i, mps, mps == me->ket);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff = me->eff_ham(
            FuseTypes::FuseLR, false, me->bra->tensors[i], me->ket->tensors[i]);
        auto pdi = h_eff->multiply(me->para_rule);
        h_eff->deallocate();
        shared_ptr<SparseMatrix<S>> old_bra = me->bra->tensors[i];
        shared_ptr<SparseMatrix<S>> old_ket = me->ket->tensors[i];
        double bra_error = 0.0;
        int bra_mmps = 0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (auto &mps : {me->bra, me->ket}) {
                shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                shared_ptr<SparseMatrix<S>> dm = nullptr;
                int bond_dim =
                        mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim,
                    error;
                if (decomp_type == DecompositionTypes::DensityMatrix) {
                    dm = MovingEnvironment<S>::density_matrix(
                        h_eff->opdq, old_wfn, forward,
                        mps == me->bra ? noise : 0.0,
                        mps == me->bra && noise != 0 ? noise_type
                                                     : NoiseTypes::None);
                    error = MovingEnvironment<S>::split_density_matrix(
                        dm, old_wfn, bond_dim, forward, false, mps->tensors[i],
                        mps->tensors[i + 1], cutoff, trunc_type);
                } else if (decomp_type == DecompositionTypes::SVD ||
                           decomp_type == DecompositionTypes::PureSVD) {
                    if (noise != 0)
                        MovingEnvironment<S>::wavefunction_add_noise(
                            old_wfn, mps == me->bra ? noise : 0.0);
                    error = MovingEnvironment<S>::split_wavefunction_svd(
                        h_eff->opdq, old_wfn, bond_dim, forward, false,
                        mps->tensors[i], mps->tensors[i + 1], cutoff,
                        trunc_type, decomp_type);
                } else
                    assert(false);
                if (mps == me->bra)
                    bra_error = error;
                shared_ptr<StateInfo<S>> info = nullptr;
                if (forward) {
                    info = mps->tensors[i]->info->extract_state_info(forward);
                    mps->info->left_dims[i + 1] = info;
                    mps->info->save_left_dims(i + 1);
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'C';
                } else {
                    info =
                        mps->tensors[i + 1]->info->extract_state_info(forward);
                    mps->info->right_dims[i + 1] = info;
                    mps->info->save_right_dims(i + 1);
                    mps->canonical_form[i] = 'C';
                    mps->canonical_form[i + 1] = 'R';
                }
                if (mps == me->bra)
                    bra_mmps = info->n_states_total;
                info->deallocate();
                mps->save_tensor(i + 1);
                mps->save_tensor(i);
                mps->unload_tensor(i + 1);
                mps->unload_tensor(i);
                if (dm != nullptr) {
                    dm->info->deallocate();
                    dm->deallocate();
                }
                MovingEnvironment<S>::propagate_wfn(
                    i, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
            }
        } else {
            for (auto &mps : {me->bra, me->ket}) {
                mps->tensors[i + 1] = make_shared<SparseMatrix<S>>();
                if (forward) {
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'C';
                } else {
                    mps->canonical_form[i] = 'C';
                    mps->canonical_form[i + 1] = 'R';
                }
            }
        }
        for (auto &old_wfn : {old_ket, old_bra}) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi), bra_error, bra_mmps, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, ubond_t bra_bond_dim,
                       ubond_t ket_bond_dim, double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, bra_bond_dim, ket_bond_dim,
                                  noise);
        else
            return update_one_dot(i, forward, bra_bond_dim, ket_bond_dim,
                                  noise);
    }
    double sweep(bool forward, ubond_t bra_bond_dim, ubond_t ket_bond_dim,
                 double noise) {
        me->prepare();
        vector<double> norms;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r =
                blocking(i, forward, bra_bond_dim, ket_bond_dim, noise);
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            norms.push_back(r.norm);
        }
        return norms.back();
    }
    double solve(int n_sweeps, bool forward = true, double tol = 1E-6) {
        if (bra_bond_dims.size() < n_sweeps)
            bra_bond_dims.resize(n_sweeps, bra_bond_dims.back());
        if (ket_bond_dims.size() < n_sweeps)
            ket_bond_dims.resize(n_sweeps, ket_bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.size() == 0 ? 0.0 : noises.back());
        Timer start, current;
        start.get_time();
        norms.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            if (iprint >= 1)
                cout << "Sweep = " << setw(4) << iw
                     << " | Direction = " << setw(8)
                     << (forward ? "forward" : "backward")
                     << " | BRA bond dimension = " << setw(4)
                     << (uint32_t)bra_bond_dims[iw]
                     << " | Noise = " << scientific << setw(9)
                     << setprecision(2) << noises[iw] << endl;
            double norm = sweep(forward, bra_bond_dims[iw], ket_bond_dims[iw],
                                noises[iw]);
            norms.push_back(norm);
            bool converged =
                norms.size() >= 2 && tol > 0 &&
                abs(norms[norms.size() - 1] - norms[norms.size() - 2]) < tol &&
                noises[iw] == noises.back() &&
                bra_bond_dims[iw] == bra_bond_dims.back();
            forward = !forward;
            current.get_time();
            if (iprint == 1) {
                cout << fixed << setprecision(8);
                cout << " .. Norm = " << setw(15) << norm << " ";
            }
            if (iprint >= 1)
                cout << "Time elapsed = " << setw(10) << setprecision(3)
                     << current.current - start.current << endl;
            if (converged)
                break;
        }
        this->forward = forward;
        return norms.back();
    }
};

inline vector<long double>
get_partition_weights(double beta, const vector<double> &energies,
                      const vector<int> &multiplicities) {
    vector<long double> partition_weights(energies.size());
    for (size_t i = 0; i < energies.size(); i++)
        partition_weights[i] =
            multiplicities[i] *
            expl(-(long double)beta *
                 ((long double)energies[i] - (long double)energies[0]));
    long double psum =
        accumulate(partition_weights.begin(), partition_weights.end(), 0.0L);
    for (size_t i = 0; i < energies.size(); i++)
        partition_weights[i] /= psum;
    return partition_weights;
}

// Expectation value
template <typename S> struct Expect {
    shared_ptr<MovingEnvironment<S>> me;
    ubond_t bra_bond_dim, ket_bond_dim;
    vector<vector<pair<shared_ptr<OpExpr<S>>, double>>> expectations;
    bool forward;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    uint8_t iprint = 2;
    double cutoff = 0.0;
    double beta = 0.0;
    // partition function (for thermal-averaged MultiMPS)
    vector<long double> partition_weights;
    Expect(const shared_ptr<MovingEnvironment<S>> &me, ubond_t bra_bond_dim,
           ubond_t ket_bond_dim)
        : me(me), bra_bond_dim(bra_bond_dim), ket_bond_dim(ket_bond_dim),
          forward(false) {
        expectations.resize(me->n_sites - me->dot + 1);
        partition_weights = vector<long double>{1.0L};
    }
    Expect(const shared_ptr<MovingEnvironment<S>> &me, ubond_t bra_bond_dim,
           ubond_t ket_bond_dim, double beta, const vector<double> &energies,
           const vector<int> &multiplicities)
        : Expect(me, bra_bond_dim, ket_bond_dim) {
        this->beta = beta;
        this->partition_weights =
            get_partition_weights(beta, energies, multiplicities);
    }
    struct Iteration {
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations;
        double bra_error, ket_error;
        double tmult;
        size_t nflop;
        Iteration(
            const vector<pair<shared_ptr<OpExpr<S>>, double>> &expectations,
            double bra_error, double ket_error, size_t nflop = 0,
            double tmult = 1.0)
            : expectations(expectations), bra_error(bra_error),
              ket_error(ket_error), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            if (r.expectations.size() == 1)
                os << " " << setw(14) << r.expectations[0].second;
            else
                os << " Nterms = " << setw(6) << r.expectations.size();
            os << " Error = " << setw(15) << setprecision(12) << r.bra_error
               << "/" << setw(15) << setprecision(12) << r.ket_error
               << " FLOPS = " << scientific << setw(8) << setprecision(2)
               << (double)r.nflop / r.tmult << " Tmult = " << fixed
               << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_one_dot(int i, bool forward, bool propagate,
                             ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        frame->activate(0);
        vector<shared_ptr<MPS<S>>> mpss =
            me->bra == me->ket ? vector<shared_ptr<MPS<S>>>{me->bra}
                               : vector<shared_ptr<MPS<S>>>{me->bra, me->ket};
        bool fuse_left = i <= me->n_sites / 2;
        for (auto &mps : mpss) {
            if (mps->canonical_form[i] == 'C') {
                if (i == 0)
                    mps->canonical_form[i] = 'K';
                else if (i == me->n_sites - 1)
                    mps->canonical_form[i] = 'S';
                else
                    assert(false);
            }
            // guess wavefunction
            // change to fused form for super-block hamiltonian
            // note that this switch exactly matches two-site conventional mpo
            // middle-site switch, so two-site conventional mpo can work
            mps->load_tensor(i);
            if ((fuse_left && mps->canonical_form[i] == 'S') ||
                (!fuse_left && mps->canonical_form[i] == 'K')) {
                shared_ptr<SparseMatrix<S>> prev_wfn = mps->tensors[i];
                if (fuse_left && mps->canonical_form[i] == 'S')
                    mps->tensors[i] =
                        MovingEnvironment<S>::swap_wfn_to_fused_left(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'K')
                    mps->tensors[i] =
                        MovingEnvironment<S>::swap_wfn_to_fused_right(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
            }
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, false,
                        me->bra->tensors[i], me->ket->tensors[i]);
        auto pdi = h_eff->expect(me->para_rule);
        h_eff->deallocate();
        double bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                for (auto &mps : mpss) {
                    shared_ptr<SparseMatrix<S>> prev_wfn = mps->tensors[i];
                    if (!fuse_left && forward)
                        mps->tensors[i] =
                            MovingEnvironment<S>::swap_wfn_to_fused_left(
                                i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        mps->tensors[i] =
                            MovingEnvironment<S>::swap_wfn_to_fused_right(
                                i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                    prev_wfn->info->deallocate();
                    prev_wfn->deallocate();
                }
            }
            vector<shared_ptr<SparseMatrix<S>>> old_wfns =
                me->bra == me->ket
                    ? vector<shared_ptr<SparseMatrix<S>>>{me->bra->tensors[i]}
                    : vector<shared_ptr<SparseMatrix<S>>>{me->ket->tensors[i],
                                                          me->bra->tensors[i]};
            if (propagate) {
                for (auto &mps : mpss) {
                    shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                    shared_ptr<SparseMatrix<S>> left, right;
                    shared_ptr<SparseMatrix<S>> dm =
                        MovingEnvironment<S>::density_matrix(
                            h_eff->opdq, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    int bond_dim =
                        mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    double error = MovingEnvironment<S>::split_density_matrix(
                        dm, old_wfn, bond_dim, forward, false, left, right,
                        cutoff, trunc_type);
                    if (mps == me->bra)
                        bra_error = error;
                    else
                        ket_error = error;
                    shared_ptr<StateInfo<S>> info = nullptr;
                    // propagation
                    if (forward) {
                        mps->tensors[i] = left;
                        mps->save_tensor(i);
                        info = left->info->extract_state_info(forward);
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        info->deallocate();
                        if (i != me->n_sites - 1) {
                            MovingEnvironment<S>::contract_one_dot(
                                i + 1, right, mps, forward);
                            mps->save_tensor(i + 1);
                            mps->unload_tensor(i + 1);
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            MovingEnvironment<S>::contract_one_dot(
                                i, right, mps, !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                            mps->canonical_form[i] = 'K';
                        }
                    } else {
                        mps->tensors[i] = right;
                        mps->save_tensor(i);
                        info = right->info->extract_state_info(forward);
                        mps->info->right_dims[i] = info;
                        mps->info->save_right_dims(i);
                        info->deallocate();
                        if (i > 0) {
                            MovingEnvironment<S>::contract_one_dot(
                                i - 1, left, mps, forward);
                            mps->save_tensor(i - 1);
                            mps->unload_tensor(i - 1);
                            mps->canonical_form[i - 1] = 'K';
                            mps->canonical_form[i] = 'R';
                        } else {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            MovingEnvironment<S>::contract_one_dot(i, left, mps,
                                                                   !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                            mps->canonical_form[i] = 'S';
                        }
                    }
                    right->info->deallocate();
                    right->deallocate();
                    left->info->deallocate();
                    left->deallocate();
                    dm->info->deallocate();
                    dm->deallocate();
                }
            }
            for (auto &old_wfn : old_wfns) {
                old_wfn->info->deallocate();
                old_wfn->deallocate();
            }
        } else {
            if (propagate) {
                for (auto &mps : mpss) {
                    if (forward) {
                        if (i != me->n_sites - 1) {
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else
                            mps->canonical_form[i] = 'K';
                    } else {
                        if (i > 0) {
                            mps->canonical_form[i - 1] = 'K';
                            mps->canonical_form[i] = 'R';
                        } else
                            mps->canonical_form[i] = 'S';
                    }
                }
            }
            me->ket->unload_tensor(i);
            if (me->bra != me->ket)
                me->bra->unload_tensor(i);
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi), bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_two_dot(int i, bool forward, bool propagate,
                             ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        frame->activate(0);
        vector<shared_ptr<MPS<S>>> mpss =
            me->bra == me->ket ? vector<shared_ptr<MPS<S>>>{me->bra}
                               : vector<shared_ptr<MPS<S>>>{me->bra, me->ket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S>::contract_two_dot(i, mps, mps == me->ket);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff = me->eff_ham(
            FuseTypes::FuseLR, false, me->bra->tensors[i], me->ket->tensors[i]);
        auto pdi = h_eff->expect(me->para_rule);
        h_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S>>> old_wfns =
            me->bra == me->ket
                ? vector<shared_ptr<SparseMatrix<S>>>{me->bra->tensors[i]}
                : vector<shared_ptr<SparseMatrix<S>>>{me->ket->tensors[i],
                                                      me->bra->tensors[i]};
        double bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (propagate) {
                for (auto &mps : mpss) {
                    shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                    shared_ptr<SparseMatrix<S>> dm =
                        MovingEnvironment<S>::density_matrix(
                            h_eff->opdq, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    int bond_dim =
                        mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    double error = MovingEnvironment<S>::split_density_matrix(
                        dm, old_wfn, bond_dim, forward, false, mps->tensors[i],
                        mps->tensors[i + 1], cutoff, trunc_type);
                    if (mps == me->bra)
                        bra_error = error;
                    else
                        ket_error = error;
                    shared_ptr<StateInfo<S>> info = nullptr;
                    if (forward) {
                        info =
                            mps->tensors[i]->info->extract_state_info(forward);
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'C';
                    } else {
                        info = mps->tensors[i + 1]->info->extract_state_info(
                            forward);
                        mps->info->right_dims[i + 1] = info;
                        mps->info->save_right_dims(i + 1);
                        mps->canonical_form[i] = 'C';
                        mps->canonical_form[i + 1] = 'R';
                    }
                    info->deallocate();
                    mps->save_tensor(i + 1);
                    mps->save_tensor(i);
                    mps->unload_tensor(i + 1);
                    mps->unload_tensor(i);
                    dm->info->deallocate();
                    dm->deallocate();
                    MovingEnvironment<S>::propagate_wfn(
                        i, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
                }
            } else
                for (auto &mps : mpss)
                    mps->save_tensor(i);
        } else {
            if (propagate) {
                for (auto &mps : mpss) {
                    mps->tensors[i + 1] = make_shared<SparseMatrix<S>>();
                    if (forward) {
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'C';
                    } else {
                        mps->canonical_form[i] = 'C';
                        mps->canonical_form[i + 1] = 'R';
                    }
                }
            }
        }
        for (auto &old_wfn : old_wfns) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi), bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_multi_one_dot(int i, bool forward, bool propagate,
                                   ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        shared_ptr<MultiMPS<S>> mket =
                                    dynamic_pointer_cast<MultiMPS<S>>(me->ket),
                                mbra =
                                    dynamic_pointer_cast<MultiMPS<S>>(me->bra);
        if (me->bra == me->ket)
            assert(mbra == mket);
        frame->activate(0);
        vector<shared_ptr<MultiMPS<S>>> mpss =
            me->bra == me->ket ? vector<shared_ptr<MultiMPS<S>>>{mbra}
                               : vector<shared_ptr<MultiMPS<S>>>{mbra, mket};
        bool fuse_left = i <= me->n_sites / 2;
        for (auto &mps : mpss) {
            if (mps->canonical_form[i] == 'M') {
                if (i == 0)
                    mps->canonical_form[i] = 'J';
                else if (i == me->n_sites - 1)
                    mps->canonical_form[i] = 'T';
                else
                    assert(false);
            }
            // guess wavefunction
            // change to fused form for super-block hamiltonian
            // note that this switch exactly matches two-site conventional mpo
            // middle-site switch, so two-site conventional mpo can work
            mps->load_tensor(i);
            if ((fuse_left && mps->canonical_form[i] == 'T') ||
                (!fuse_left && mps->canonical_form[i] == 'J')) {
                vector<shared_ptr<SparseMatrixGroup<S>>> prev_wfns = mps->wfns;
                if (fuse_left && mps->canonical_form[i] == 'T')
                    mps->wfns =
                        MovingEnvironment<S>::swap_multi_wfn_to_fused_left(
                            i, mps->info, prev_wfns, me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'J')
                    mps->wfns =
                        MovingEnvironment<S>::swap_multi_wfn_to_fused_right(
                            i, mps->info, prev_wfns, me->mpo->tf->opf->cg);
                for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                    prev_wfns[j]->deallocate();
                if (prev_wfns.size() != 0)
                    prev_wfns[0]->deallocate_infos();
            }
        }
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>> h_eff =
            me->multi_eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                              true);
        auto pdi = h_eff->expect(me->para_rule);
        h_eff->deallocate();
        double bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                for (auto &mps : mpss) {
                    vector<shared_ptr<SparseMatrixGroup<S>>> prev_wfns =
                        mps->wfns;
                    if (!fuse_left && forward)
                        mps->wfns =
                            MovingEnvironment<S>::swap_multi_wfn_to_fused_left(
                                i, mps->info, prev_wfns, me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        mps->wfns =
                            MovingEnvironment<S>::swap_multi_wfn_to_fused_right(
                                i, mps->info, prev_wfns, me->mpo->tf->opf->cg);
                    for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                        prev_wfns[j]->deallocate();
                    if (prev_wfns.size() != 0)
                        prev_wfns[0]->deallocate_infos();
                }
            }
            // splitting of wavefunction
            vector<vector<shared_ptr<SparseMatrixGroup<S>>>> old_wfnss =
                me->bra == me->ket
                    ? vector<
                          vector<shared_ptr<SparseMatrixGroup<S>>>>{mbra->wfns}
                    : vector<vector<shared_ptr<SparseMatrixGroup<S>>>>{
                          mket->wfns, mbra->wfns};
            if (propagate) {
                for (auto &mps : mpss) {
                    vector<shared_ptr<SparseMatrixGroup<S>>> old_wfn =
                                                                 mps->wfns,
                                                             new_wfns;
                    shared_ptr<SparseMatrix<S>> rot;
                    shared_ptr<SparseMatrix<S>> dm =
                        MovingEnvironment<S>::density_matrix_with_multi_target(
                            h_eff->opdq, old_wfn, mps->weights, forward, 0.0,
                            NoiseTypes::None);
                    int bond_dim =
                        mps == mbra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    double error =
                        MovingEnvironment<S>::multi_split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false, new_wfns,
                            rot, cutoff, trunc_type);
                    if (mps == mbra)
                        bra_error = error;
                    else
                        ket_error = error;
                    shared_ptr<StateInfo<S>> info = nullptr;
                    // propagation
                    if (forward) {
                        mps->tensors[i] = rot;
                        mps->save_tensor(i);
                        info = rot->info->extract_state_info(forward);
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        info->deallocate();
                        if (i != me->n_sites - 1) {
                            MovingEnvironment<S>::contract_multi_one_dot(
                                i + 1, new_wfns, mps, forward);
                            mps->save_wavefunction(i + 1);
                            mps->unload_wavefunction(i + 1);
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'T';
                        } else {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            MovingEnvironment<S>::contract_multi_one_dot(
                                i, new_wfns, mps, !forward);
                            mps->save_wavefunction(i);
                            mps->unload_wavefunction(i);
                            mps->canonical_form[i] = 'J';
                        }
                    } else {
                        mps->tensors[i] = rot;
                        mps->save_tensor(i);
                        info = rot->info->extract_state_info(forward);
                        mps->info->right_dims[i] = info;
                        mps->info->save_right_dims(i);
                        info->deallocate();
                        if (i > 0) {
                            MovingEnvironment<S>::contract_multi_one_dot(
                                i - 1, new_wfns, mps, forward);
                            mps->save_wavefunction(i - 1);
                            mps->unload_wavefunction(i - 1);
                            mps->canonical_form[i - 1] = 'J';
                            mps->canonical_form[i] = 'R';
                        } else {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            MovingEnvironment<S>::contract_multi_one_dot(
                                i, new_wfns, mps, !forward);
                            mps->save_wavefunction(i);
                            mps->unload_wavefunction(i);
                            mps->canonical_form[i] = 'T';
                        }
                    }
                    if (forward) {
                        for (int j = (int)new_wfns.size() - 1; j >= 0; j--)
                            new_wfns[j]->deallocate();
                        if (new_wfns.size() != 0)
                            new_wfns[0]->deallocate_infos();
                        rot->info->deallocate();
                        rot->deallocate();
                    } else {
                        rot->info->deallocate();
                        rot->deallocate();
                        for (int j = (int)new_wfns.size() - 1; j >= 0; j--)
                            new_wfns[j]->deallocate();
                        if (new_wfns.size() != 0)
                            new_wfns[0]->deallocate_infos();
                    }
                    dm->info->deallocate();
                    dm->deallocate();
                }
            }
            // if not propagate, the wfns are changed but not saved, so no need
            // to save
            for (auto &old_wfns : old_wfnss) {
                for (int k = mket->nroots - 1; k >= 0; k--)
                    old_wfns[k]->deallocate();
                old_wfns[0]->deallocate_infos();
            }
        } else {
            vector<vector<shared_ptr<SparseMatrixGroup<S>>>> old_wfnss =
                me->bra == me->ket
                    ? vector<
                          vector<shared_ptr<SparseMatrixGroup<S>>>>{mbra->wfns}
                    : vector<vector<shared_ptr<SparseMatrixGroup<S>>>>{
                          mket->wfns, mbra->wfns};
            for (auto &old_wfns : old_wfnss) {
                for (int k = mket->nroots - 1; k >= 0; k--)
                    old_wfns[k]->deallocate();
                old_wfns[0]->deallocate_infos();
            }
            if (propagate) {
                for (auto &mps : mpss) {
                    if (forward) {
                        if (i != me->n_sites - 1) {
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            mps->tensors[i + 1] = nullptr;
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'T';
                        } else
                            mps->canonical_form[i] = 'J';
                    } else {
                        if (i > 0) {
                            mps->tensors[i - 1] = nullptr;
                            mps->tensors[i] = make_shared<SparseMatrix<S>>();
                            mps->canonical_form[i - 1] = 'J';
                            mps->canonical_form[i] = 'R';
                        } else
                            mps->canonical_form[i] = 'T';
                    }
                }
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++) {
            long double x = 0.0;
            for (size_t l = 0; l < partition_weights.size(); l++)
                x += partition_weights[l] * get<0>(pdi)[k].second[l];
            expectations[k] = make_pair(get<0>(pdi)[k].first, (double)x);
        }
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_multi_two_dot(int i, bool forward, bool propagate,
                                   ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        shared_ptr<MultiMPS<S>> mket =
                                    dynamic_pointer_cast<MultiMPS<S>>(me->ket),
                                mbra =
                                    dynamic_pointer_cast<MultiMPS<S>>(me->bra);
        if (me->bra == me->ket)
            assert(mbra == mket);
        frame->activate(0);
        vector<shared_ptr<MultiMPS<S>>> mpss =
            me->bra == me->ket ? vector<shared_ptr<MultiMPS<S>>>{mbra}
                               : vector<shared_ptr<MultiMPS<S>>>{mbra, mket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr || mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S>::contract_multi_two_dot(i, mps,
                                                             mps == mket);
            else
                mps->load_tensor(i);
            mps->tensors[i] = mps->tensors[i + 1] = nullptr;
        }
        shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>> h_eff =
            me->multi_eff_ham(FuseTypes::FuseLR, false);
        auto pdi = h_eff->expect(me->para_rule);
        h_eff->deallocate();
        vector<vector<shared_ptr<SparseMatrixGroup<S>>>> old_wfnss =
            me->bra == me->ket
                ? vector<vector<shared_ptr<SparseMatrixGroup<S>>>>{mbra->wfns}
                : vector<vector<shared_ptr<SparseMatrixGroup<S>>>>{mket->wfns,
                                                                   mbra->wfns};
        double bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (propagate) {
                for (auto &mps : mpss) {
                    vector<shared_ptr<SparseMatrixGroup<S>>> old_wfn =
                        mps->wfns;
                    shared_ptr<SparseMatrix<S>> dm =
                        MovingEnvironment<S>::density_matrix_with_multi_target(
                            h_eff->opdq, old_wfn, mps->weights, forward, 0.0,
                            NoiseTypes::None);
                    int bond_dim =
                        mps == mbra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    double error =
                        MovingEnvironment<S>::multi_split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false, mps->wfns,
                            forward ? mps->tensors[i] : mps->tensors[i + 1],
                            cutoff, trunc_type);
                    if (mps == mbra)
                        bra_error = error;
                    else
                        ket_error = error;
                    shared_ptr<StateInfo<S>> info = nullptr;
                    if (forward) {
                        info =
                            mps->tensors[i]->info->extract_state_info(forward);
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'M';
                    } else {
                        info = mps->tensors[i + 1]->info->extract_state_info(
                            forward);
                        mps->info->right_dims[i + 1] = info;
                        mps->info->save_right_dims(i + 1);
                        mps->canonical_form[i] = 'M';
                        mps->canonical_form[i + 1] = 'R';
                    }
                    info->deallocate();
                    if (forward) {
                        mps->save_wavefunction(i + 1);
                        mps->save_tensor(i);
                        mps->unload_wavefunction(i + 1);
                        mps->unload_tensor(i);
                    } else {
                        mps->save_tensor(i + 1);
                        mps->save_wavefunction(i);
                        mps->unload_tensor(i + 1);
                        mps->unload_wavefunction(i);
                    }
                    dm->info->deallocate();
                    dm->deallocate();
                    MovingEnvironment<S>::propagate_multi_wfn(
                        i, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
                }
            } else {
                for (auto &mps : mpss)
                    mps->save_tensor(i);
            }
        } else {
            if (propagate) {
                for (auto &mps : mpss) {
                    if (forward) {
                        mps->tensors[i] = make_shared<SparseMatrix<S>>();
                        mps->tensors[i + 1] = nullptr;
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'M';
                    } else {
                        mps->tensors[i] = nullptr;
                        mps->tensors[i + 1] = make_shared<SparseMatrix<S>>();
                        mps->canonical_form[i] = 'M';
                        mps->canonical_form[i + 1] = 'R';
                    }
                }
            }
        }
        for (auto &old_wfns : old_wfnss) {
            for (int k = mket->nroots - 1; k >= 0; k--)
                old_wfns[k]->deallocate();
            old_wfns[0]->deallocate_infos();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++) {
            long double x = 0.0;
            for (size_t l = 0; l < partition_weights.size(); l++)
                x += partition_weights[l] * get<0>(pdi)[k].second[l];
            expectations[k] = make_pair(get<0>(pdi)[k].first, (double)x);
        }
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, bool propagate,
                       ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        me->move_to(i);
        assert(me->dot == 1 || me->dot == 2);
        if (me->dot == 2) {
            if (me->ket->canonical_form[i] == 'M' ||
                me->ket->canonical_form[i + 1] == 'M')
                return update_multi_two_dot(i, forward, propagate, bra_bond_dim,
                                            ket_bond_dim);
            else
                return update_two_dot(i, forward, propagate, bra_bond_dim,
                                      ket_bond_dim);
        } else {
            if (me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T')
                return update_multi_one_dot(i, forward, propagate, bra_bond_dim,
                                            ket_bond_dim);
            else
                return update_one_dot(i, forward, propagate, bra_bond_dim,
                                      ket_bond_dim);
        }
    }
    void sweep(bool forward, ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        me->prepare();
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r =
                blocking(i, forward, true, bra_bond_dim, ket_bond_dim);
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            expectations[i] = r.expectations;
        }
    }
    double solve(bool propagate, bool forward = true) {
        Timer start, current;
        start.get_time();
        for (auto &x : expectations)
            x.clear();
        if (propagate) {
            if (iprint >= 1) {
                cout << "Expectation | Direction = " << setw(8)
                     << (forward ? "forward" : "backward")
                     << " | BRA bond dimension = " << setw(4)
                     << (uint32_t)bra_bond_dim
                     << " | KET bond dimension = " << setw(4)
                     << (uint32_t)ket_bond_dim;
                if (beta != 0.0)
                    cout << " | 1/T = " << fixed << setw(10) << setprecision(5)
                         << beta;
                cout << endl;
            }
            sweep(forward, bra_bond_dim, ket_bond_dim);
            forward = !forward;
            current.get_time();
            if (iprint >= 1)
                cout << "Time elapsed = " << setw(10) << setprecision(3)
                     << current.current - start.current << endl;
            this->forward = forward;
            return 0.0;
        } else {
            Iteration r = blocking(me->center, forward, false, bra_bond_dim,
                                   ket_bond_dim);
            assert(r.expectations.size() != 0);
            return r.expectations[0].second;
        }
    }
    // only works for SU2
    MatrixRef get_1pdm_spatial(uint16_t n_physical_sites = 0U) {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        MatrixRef r(nullptr, n_physical_sites, n_physical_sites);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                r(op->site_index[0], op->site_index[1]) = x.second;
            }
        return r;
    }
    // only works for SZ
    MatrixRef get_1pdm(uint16_t n_physical_sites = 0U) {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        MatrixRef r(nullptr, n_physical_sites * 2, n_physical_sites * 2);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                r(2 * op->site_index[0] + op->site_index.s(0),
                  2 * op->site_index[1] + op->site_index.s(1)) = x.second;
            }
        return r;
    }
    // only works for SZ
    shared_ptr<Tensor> get_2pdm(uint16_t n_physical_sites = 0U) {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        shared_ptr<Tensor> r = make_shared<Tensor>(
            vector<int>{n_physical_sites * 2, n_physical_sites * 2,
                        n_physical_sites * 2, n_physical_sites * 2});
        r->clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM2);
                (*r)({op->site_index[0] * 2 + op->site_index.s(0),
                      op->site_index[1] * 2 + op->site_index.s(1),
                      op->site_index[2] * 2 + op->site_index.s(2),
                      op->site_index[3] * 2 + op->site_index.s(3)}) = x.second;
            }
        return r;
    }
    // only works for SU2
    // number of particle correlation
    // s == 0: pure spin; s == 1: mixed spin
    MatrixRef get_1npc_spatial(uint8_t s, uint16_t n_physical_sites = 0U) {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        MatrixRef r(nullptr, n_physical_sites, n_physical_sites);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                assert(op->site_index.ss() < 2);
                if (s == op->site_index.ss())
                    r(op->site_index[0], op->site_index[1]) = x.second;
            }
        return r;
    }
    // only works for SZ
    // number of particle correlation
    // s == 0: pure spin; s == 1: mixed spin
    MatrixRef get_1npc(uint8_t s, uint16_t n_physical_sites = 0U) {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        MatrixRef r(nullptr, n_physical_sites * 2, n_physical_sites * 2);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                if (s == 0 && op->site_index.s(2) == 0)
                    r(2 * op->site_index[0] + op->site_index.s(0),
                      2 * op->site_index[1] + op->site_index.s(1)) = x.second;
                else if (s == 1 && op->site_index.s(2) == 1)
                    r(2 * op->site_index[0] + op->site_index.s(0),
                      2 * op->site_index[1] + !op->site_index.s(0)) = x.second;
            }
        return r;
    }
};

} // namespace block2
