
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
#include "../core/matrix.hpp"
#include "../core/sparse_matrix.hpp"
#include "effective_functions.hpp"
#include "moving_environment.hpp"
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

enum struct TETypes : uint8_t {
    ImagTE = 1,
    RealTE = 2,
    TangentSpace = 4,
    RK4 = 8,
    RK4PP = 16,
};

inline bool operator&(TETypes a, TETypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline TETypes operator|(TETypes a, TETypes b) {
    return TETypes((uint8_t)a | (uint8_t)b);
}

enum struct TruncPatternTypes : uint8_t { None, TruncAfterOdd, TruncAfterEven };

// Imaginary/Real Time Evolution (td-DMRG++/RK4)
template <typename S, typename FL, typename FLS> struct TDDMRG {
    typedef typename MovingEnvironment<S, FL, FLS>::FPS FPS;
    typedef typename MovingEnvironment<S, FL, FLS>::FLLS FLLS;
    shared_ptr<MovingEnvironment<S, FL, FLS>> me;
    shared_ptr<MovingEnvironment<S, FL, FLS>> lme, rme;
    vector<ubond_t> bond_dims;
    vector<FPS> noises;
    vector<FLLS> energies;
    vector<FPS> normsqs;
    vector<FPS> discarded_weights;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    bool forward;
    TETypes mode = TETypes::ImagTE | TETypes::RK4PP;
    int n_sub_sweeps = 1;
    vector<FPS> weights = {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0};
    uint8_t iprint = 2;
    FPS cutoff = 1E-14;
    bool decomp_last_site = true;
    bool hermitian = true; //!< Whether the Hamiltonian is Hermitian (symmetric)
    size_t sweep_cumulative_nflop = 0;
    // store all wfn singular values (for analysis) at each site
    bool store_wfn_spectra = false;
    vector<vector<FPS>> sweep_wfn_spectra;
    vector<FPS> wfn_spectra;
    TDDMRG(const shared_ptr<MovingEnvironment<S, FL, FLS>> &me,
           const vector<ubond_t> &bond_dims,
           const vector<FPS> &noises = vector<FPS>())
        : me(me), bond_dims(bond_dims), noises(noises), forward(false) {}
    struct Iteration {
        FLLS energy;
        FPS normsq, error;
        int nmult, mmps;
        double tmult;
        size_t nflop;
        Iteration(FLLS energy, FPS normsq, FPS error, int mmps, int nmult,
                  size_t nflop = 0, double tmult = 1.0)
            : energy(energy), normsq(normsq), error(error), mmps(mmps),
              nmult(nmult), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Mmps =" << setw(5) << r.mmps;
            os << " Nmult = " << setw(4) << r.nmult << " E = " << setw(17)
               << setprecision(10) << r.energy << " Error = " << scientific
               << setw(8) << setprecision(2) << r.error
               << " FLOPS = " << scientific << setw(8) << setprecision(2)
               << (double)r.nflop / r.tmult << " Tmult = " << fixed
               << setprecision(2) << r.tmult;
            return os;
        }
    };
    // one-site algorithm
    Iteration update_one_dot(int i, bool forward, bool advance, FPS beta,
                             ubond_t bond_dim, FPS noise) {
        assert(rme->bra != rme->ket);
        frame_<FPS>()->activate(0);
        bool fuse_left = i <= rme->fuse_center;
        vector<shared_ptr<MPS<S, FLS>>> mpss = {rme->bra, rme->ket};
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
                shared_ptr<SparseMatrix<S, FLS>> prev_wfn = mps->tensors[i];
                if (fuse_left && mps->canonical_form[i] == 'S')
                    mps->tensors[i] =
                        MovingEnvironment<S, FL, FLS>::swap_wfn_to_fused_left(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'K')
                    mps->tensors[i] =
                        MovingEnvironment<S, FL, FLS>::swap_wfn_to_fused_right(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
            }
        }
        shared_ptr<SparseMatrixGroup<S, FLS>> pbra = nullptr;
        tuple<FLS, FPS, int, size_t, double> pdi;
        bool last_site =
            (forward && i == me->n_sites - 1) || (!forward && i == 0);
        vector<shared_ptr<SparseMatrix<S, FLS>>> mrk4;
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S, FL>> r_eff = rme->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, false,
            rme->bra->tensors[i], rme->ket->tensors[i]);
        auto rvmt =
            r_eff->first_rk4_apply(-beta, me->mpo->const_e, rme->para_rule);
        r_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S, FLS>>> hkets = rvmt.first;
        memcpy(lme->bra->tensors[i]->data, hkets[0]->data,
               hkets[0]->total_memory * sizeof(FLS));
        shared_ptr<EffectiveHamiltonian<S, FL>> l_eff = lme->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
            lme->bra->tensors[i], lme->ket->tensors[i]);
        if ((mode & TETypes::RK4PP) && !last_site) {
            auto lvmt = l_eff->second_rk4_apply(-beta, me->mpo->const_e,
                                                hkets[1], false, me->para_rule);
            memcpy(lme->bra->tensors[i]->data, lvmt.first.back()->data,
                   hkets[0]->total_memory * sizeof(FLS));
            pdi = lvmt.second;
        } else
            pdi = l_eff->expo_apply(-beta, me->mpo->const_e, hermitian,
                                    iprint >= 3, me->para_rule);
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pbra = l_eff->perturbative_noise(
                forward, i, i, fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                lme->bra->info, noise_type, me->para_rule);
        l_eff->deallocate();
        get<2>(pdi) += get<0>(rvmt.second);
        get<3>(pdi) += get<1>(rvmt.second);
        get<4>(pdi) += get<2>(rvmt.second);
        FPS bra_error = 0.0;
        int bra_mmps = 0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (!decomp_last_site &&
                ((forward && i == me->n_sites - 1 && !fuse_left) ||
                 (!forward && i == 0 && fuse_left))) {
                for (auto &mps : mpss) {
                    mps->save_tensor(i);
                    mps->unload_tensor(i);
                    mps->canonical_form[i] = forward ? 'S' : 'K';
                }
            } else {
                // change to fused form for splitting
                if (fuse_left != forward) {
                    for (auto &mps : mpss) {
                        shared_ptr<SparseMatrix<S, FLS>> prev_wfn =
                            mps->tensors[i];
                        if (!fuse_left && forward)
                            mps->tensors[i] = MovingEnvironment<S, FL, FLS>::
                                swap_wfn_to_fused_left(i, mps->info, prev_wfn,
                                                       me->mpo->tf->opf->cg);
                        else if (fuse_left && !forward)
                            mps->tensors[i] = MovingEnvironment<S, FL, FLS>::
                                swap_wfn_to_fused_right(i, mps->info, prev_wfn,
                                                        me->mpo->tf->opf->cg);
                        prev_wfn->info->deallocate();
                        prev_wfn->deallocate();
                    }
                    if (pbra != nullptr) {
                        vector<shared_ptr<SparseMatrixGroup<S, FLS>>>
                            prev_pbras = {pbra};
                        if (!fuse_left && forward)
                            pbra = MovingEnvironment<S, FL, FLS>::
                                swap_multi_wfn_to_fused_left(
                                    i, rme->bra->info, prev_pbras,
                                    me->mpo->tf->opf->cg)[0];
                        else if (fuse_left && !forward)
                            pbra = MovingEnvironment<S, FL, FLS>::
                                swap_multi_wfn_to_fused_right(
                                    i, rme->bra->info, prev_pbras,
                                    me->mpo->tf->opf->cg)[0];
                        prev_pbras[0]->deallocate_infos();
                        prev_pbras[0]->deallocate();
                    }
                    if (mrk4.size() != 0)
                        for (size_t ip = 0; ip < mrk4.size(); ip++) {
                            shared_ptr<SparseMatrix<S, FLS>> prev_wfn =
                                mrk4[ip];
                            if (!fuse_left && forward)
                                mrk4[ip] = MovingEnvironment<S, FL, FLS>::
                                    swap_wfn_to_fused_left(
                                        i, rme->bra->info, prev_wfn,
                                        me->mpo->tf->opf->cg);
                            else if (fuse_left && !forward)
                                mrk4[ip] = MovingEnvironment<S, FL, FLS>::
                                    swap_wfn_to_fused_right(
                                        i, rme->bra->info, prev_wfn,
                                        me->mpo->tf->opf->cg);
                            prev_wfn->deallocate();
                        }
                }
                vector<shared_ptr<SparseMatrix<S, FLS>>> old_wfns;
                for (auto &mps : mpss)
                    old_wfns.push_back(mps->tensors[i]);
                if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
                    assert(pbra != nullptr);
                for (auto &mps : mpss) {
                    // splitting of wavefunction
                    shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i];
                    shared_ptr<SparseMatrix<S, FLS>> left, right;
                    shared_ptr<SparseMatrix<S, FLS>> dm = nullptr;
                    FPS error;
                    if (decomp_type == DecompositionTypes::DensityMatrix) {
                        if (mps != rme->bra) {
                            dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                                mps->info->vacuum, old_wfn, forward, 0.0,
                                NoiseTypes::None);
                        } else if (mrk4.size() == 0) {
                            dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                                mps->info->vacuum, old_wfn, forward, noise,
                                noise_type, 1.0, pbra);
                        } else {
                            dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                                mps->info->vacuum, old_wfn, forward, noise,
                                noise_type, weights[0], pbra);
                            assert(mrk4.size() == 3);
                            for (int i = 0; i < 3; i++)
                                MovingEnvironment<S, FL, FLS>::
                                    density_matrix_add_wfn(dm, mrk4[i], forward,
                                                           weights[i + 1]);
                        }
                        error =
                            MovingEnvironment<S, FL, FLS>::split_density_matrix(
                                dm, old_wfn, bond_dim, forward, false, left,
                                right, cutoff,
                                store_wfn_spectra && mps == rme->bra,
                                wfn_spectra, trunc_type);
                    } else if (decomp_type == DecompositionTypes::SVD ||
                               decomp_type == DecompositionTypes::PureSVD) {
                        if (mps != rme->bra) {
                            error = MovingEnvironment<S, FL, FLS>::
                                split_wavefunction_svd(
                                    mps->info->vacuum, old_wfn, bond_dim,
                                    forward, false, left, right, cutoff,
                                    store_wfn_spectra && mps == rme->bra,
                                    wfn_spectra, trunc_type, decomp_type,
                                    nullptr);
                        } else {
                            if (noise != 0 && mps == rme->bra) {
                                if (noise_type & NoiseTypes::Wavefunction)
                                    MovingEnvironment<S, FL, FLS>::
                                        wavefunction_add_noise(old_wfn, noise);
                                else if (noise_type & NoiseTypes::Perturbative)
                                    MovingEnvironment<S, FL, FLS>::
                                        scale_perturbative_noise(
                                            noise, noise_type, pbra);
                            }
                            vector<FPS> xweights = {1};
                            vector<shared_ptr<SparseMatrix<S, FLS>>> xwfns = {};
                            if (mrk4.size() != 0) {
                                assert(mrk4.size() == 3);
                                xweights = weights;
                                xwfns = mrk4;
                            }
                            error = MovingEnvironment<S, FL, FLS>::
                                split_wavefunction_svd(
                                    mps->info->vacuum, old_wfn, bond_dim,
                                    forward, false, left, right, cutoff,
                                    store_wfn_spectra && mps == rme->bra,
                                    wfn_spectra, trunc_type, decomp_type, pbra,
                                    xwfns, xweights);
                        }
                    } else
                        assert(false);
                    if (mps == rme->bra)
                        bra_error = error;
                    shared_ptr<StateInfo<S>> info = nullptr;
                    // propagation
                    if (forward) {
                        mps->tensors[i] = left;
                        mps->save_tensor(i);
                        info = left->info->extract_state_info(forward);
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        if (i != me->n_sites - 1) {
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i + 1, right, mps, forward);
                            mps->save_tensor(i + 1);
                            mps->unload_tensor(i + 1);
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else {
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
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
                        if (i > 0) {
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i - 1, left, mps, forward);
                            mps->save_tensor(i - 1);
                            mps->unload_tensor(i - 1);
                            mps->canonical_form[i - 1] = 'K';
                            mps->canonical_form[i] = 'R';
                        } else {
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i, left, mps, !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                            mps->canonical_form[i] = 'S';
                        }
                    }
                    if (mps == rme->bra) {
                        bra_mmps = (int)info->n_states_total;
                        mps->info->bond_dim =
                            max(mps->info->bond_dim, (ubond_t)bra_mmps);
                    }
                    info->deallocate();
                    right->info->deallocate();
                    right->deallocate();
                    left->info->deallocate();
                    left->deallocate();
                    if (dm != nullptr) {
                        dm->info->deallocate();
                        dm->deallocate();
                    }
                }
                for (auto &old_wfn : vector<shared_ptr<SparseMatrix<S, FLS>>>(
                         old_wfns.rbegin(), old_wfns.rend())) {
                    old_wfn->info->deallocate();
                    old_wfn->deallocate();
                }
            }
            for (auto &mps : mpss)
                mps->save_data();
        } else {
            if (!decomp_last_site &&
                ((forward && i == me->n_sites - 1 && !fuse_left) ||
                 (!forward && i == 0 && fuse_left)))
                for (auto &mps : mpss)
                    mps->canonical_form[i] = forward ? 'S' : 'K';
            else
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
            for (auto &mps :
                 vector<shared_ptr<MPS<S, FLS>>>(mpss.rbegin(), mpss.rend()))
                mps->unload_tensor(i);
        }
        if (pbra != nullptr) {
            pbra->deallocate();
            pbra->deallocate_infos();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration((FLLS)get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), bra_error, bra_mmps,
                         get<2>(pdi), get<3>(pdi), get<4>(pdi));
    }
    // two-site algorithm
    Iteration update_two_dot(int i, bool forward, bool advance, FPS beta,
                             ubond_t bond_dim, FPS noise) {
        assert(rme->bra != rme->ket);
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MPS<S, FLS>>> mpss = {rme->bra, rme->ket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_two_dot(i, mps);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<SparseMatrixGroup<S, FLS>> pbra = nullptr;
        tuple<FLS, FPS, int, size_t, double> pdi;
        bool last_site =
            (forward && i + 1 == me->n_sites - 1) || (!forward && i == 0);
        vector<shared_ptr<SparseMatrix<S, FLS>>> mrk4;
        shared_ptr<EffectiveHamiltonian<S, FL>> r_eff =
            rme->eff_ham(FuseTypes::FuseLR, forward, false,
                         rme->bra->tensors[i], rme->ket->tensors[i]);
        auto rvmt =
            r_eff->first_rk4_apply(-beta, me->mpo->const_e, rme->para_rule);
        r_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S, FLS>>> hkets = rvmt.first;
        memcpy(lme->bra->tensors[i]->data, hkets[0]->data,
               hkets[0]->total_memory * sizeof(FLS));
        shared_ptr<EffectiveHamiltonian<S, FL>> l_eff =
            lme->eff_ham(FuseTypes::FuseLR, forward, true, lme->bra->tensors[i],
                         lme->ket->tensors[i]);
        if ((mode & TETypes::RK4PP) && !last_site) {
            auto lvmt = l_eff->second_rk4_apply(-beta, me->mpo->const_e,
                                                hkets[1], false, me->para_rule);
            memcpy(lme->bra->tensors[i]->data, lvmt.first.back()->data,
                   hkets[0]->total_memory * sizeof(FLS));
            pdi = lvmt.second;
        } else
            pdi = l_eff->expo_apply(-beta, me->mpo->const_e, hermitian,
                                    iprint >= 3, me->para_rule);
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pbra = l_eff->perturbative_noise(forward, i, i + 1,
                                             FuseTypes::FuseLR, lme->bra->info,
                                             noise_type, me->para_rule);
        l_eff->deallocate();
        get<2>(pdi) += get<0>(rvmt.second);
        get<3>(pdi) += get<1>(rvmt.second);
        get<4>(pdi) += get<2>(rvmt.second);
        vector<shared_ptr<SparseMatrix<S, FLS>>> old_wfns;
        for (auto &mps : mpss)
            old_wfns.push_back(mps->tensors[i]);
        FPS bra_error = 0.0;
        int bra_mmps = 0;
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            assert(pbra != nullptr);
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (auto &mps : mpss) {
                shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i];
                shared_ptr<SparseMatrix<S, FLS>> dm = nullptr;
                FPS error;
                if (decomp_type == DecompositionTypes::DensityMatrix) {
                    if (mps != rme->bra) {
                        dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    } else if (mrk4.size() == 0) {
                        dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, noise,
                            noise_type, 1.0, pbra);
                    } else {
                        dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, noise,
                            noise_type, weights[0], pbra);
                        assert(mrk4.size() == 3);
                        for (int i = 0; i < 3; i++)
                            MovingEnvironment<S, FL, FLS>::
                                density_matrix_add_wfn(dm, mrk4[i], forward,
                                                       weights[i + 1]);
                    }
                    error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                        dm, old_wfn, bond_dim, forward, false, mps->tensors[i],
                        mps->tensors[i + 1], cutoff,
                        store_wfn_spectra && mps == rme->bra, wfn_spectra,
                        trunc_type);
                } else if (decomp_type == DecompositionTypes::SVD ||
                           decomp_type == DecompositionTypes::PureSVD) {
                    if (mps != rme->bra) {
                        error = MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                mps->info->vacuum, old_wfn, bond_dim, forward,
                                false, mps->tensors[i], mps->tensors[i + 1],
                                cutoff, store_wfn_spectra && mps == rme->bra,
                                wfn_spectra, trunc_type, decomp_type, nullptr);
                    } else {
                        if (noise != 0 && mps == rme->bra) {
                            if (noise_type & NoiseTypes::Wavefunction)
                                MovingEnvironment<
                                    S, FL, FLS>::wavefunction_add_noise(old_wfn,
                                                                        noise);
                            else if (noise_type & NoiseTypes::Perturbative)
                                MovingEnvironment<S, FL, FLS>::
                                    scale_perturbative_noise(noise, noise_type,
                                                             pbra);
                        }
                        vector<FPS> xweights = {1};
                        vector<shared_ptr<SparseMatrix<S, FLS>>> xwfns = {};
                        if (mrk4.size() != 0) {
                            assert(mrk4.size() == 3);
                            xweights = weights;
                            xwfns = mrk4;
                        }
                        error = MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                mps->info->vacuum, old_wfn, bond_dim, forward,
                                false, mps->tensors[i], mps->tensors[i + 1],
                                cutoff, store_wfn_spectra && mps == rme->bra,
                                wfn_spectra, trunc_type, decomp_type, pbra,
                                xwfns, xweights);
                    }
                } else
                    assert(false);
                if (mps == rme->bra)
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
                if (mps == rme->bra) {
                    bra_mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)bra_mmps);
                }
                info->deallocate();
                mps->save_tensor(i + 1);
                mps->save_tensor(i);
                mps->unload_tensor(i + 1);
                mps->unload_tensor(i);
                if (dm != nullptr) {
                    dm->info->deallocate();
                    dm->deallocate();
                }
            }
        } else {
            for (auto &mps : mpss) {
                mps->tensors[i + 1] = make_shared<SparseMatrix<S, FLS>>();
                if (forward) {
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'C';
                } else {
                    mps->canonical_form[i] = 'C';
                    mps->canonical_form[i + 1] = 'R';
                }
            }
        }
        if (pbra != nullptr) {
            pbra->deallocate();
            pbra->deallocate_infos();
        }
        for (auto mrk : mrk4)
            mrk->deallocate();
        for (auto hket : hkets)
            hket->deallocate();
        for (auto &old_wfn : vector<shared_ptr<SparseMatrix<S, FLS>>>(
                 old_wfns.rbegin(), old_wfns.rend())) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (auto &mps : mpss) {
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, 0, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
                mps->save_data();
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration((FLLS)get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), bra_error, bra_mmps,
                         get<2>(pdi), get<3>(pdi), get<4>(pdi));
    }
    Iteration blocking(int i, bool forward, bool advance, FPS beta,
                       ubond_t bond_dim, FPS noise) {
        lme->move_to(i);
        rme->move_to(i);
        assert(rme->dot == 2 || rme->dot == 1);
        Iteration it(0, 0, 0, 0, 0);
        if (rme->dot == 2)
            it = update_two_dot(i, forward, advance, beta, bond_dim, noise);
        else
            it = update_one_dot(i, forward, advance, beta, bond_dim, noise);
        if (store_wfn_spectra) {
            const int bond_update_idx =
                rme->dot == 1 && !forward && i > 0 ? i - 1 : i;
            if (sweep_wfn_spectra.size() <= bond_update_idx)
                sweep_wfn_spectra.resize(bond_update_idx + 1);
            sweep_wfn_spectra[bond_update_idx] = wfn_spectra;
        }
        return it;
    }
    tuple<FLLS, FPS, FPS> sweep(bool forward, bool advance, FPS beta,
                                ubond_t bond_dim, FPS noise) {
        lme->prepare();
        rme->prepare();
        vector<FLLS> energies;
        vector<FPS> normsqs;
        sweep_cumulative_nflop = 0;
        frame_<FPS>()->reset_peak_used_memory();
        vector<int> sweep_range;
        FPS largest_error = 0.0;
        if (forward)
            for (int it = rme->center; it < rme->n_sites - rme->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = rme->center; it >= 0; it--)
                sweep_range.push_back(it);
        Timer t;
        for (auto i : sweep_range) {
            check_signal_()();
            if (iprint >= 2) {
                if (rme->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << "-" << setw(4)
                         << i + 1 << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                         << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r = blocking(i, forward, advance, beta, bond_dim,
                                   advance ? 0 : noise);
            sweep_cumulative_nflop += r.nflop;
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
        me->ket->load_tensor((int)center);
        me->ket->tensors[center]->normalize();
        me->ket->save_tensor((int)center);
        me->ket->unload_tensor((int)center);
    }
    void init_moving_environments() {
        const string me_tag = Parsing::split(me->tag, "@", true)[0];
        const string mps_tag = Parsing::split(me->ket->info->tag, "@", true)[0];
        vector<string> tags = {me_tag, me_tag + "@L", me_tag + "@R"};
        vector<string> mps_tags = {mps_tag, mps_tag + "@X"};
        vector<string> avail_tags, avail_mps_tags;
        string avail_mps_tag = "";
        avail_tags.reserve(2);
        for (auto tag : tags)
            if (tag != me->tag)
                avail_tags.push_back(tag);
        for (auto tag : mps_tags)
            if (tag != me->ket->info->tag)
                avail_mps_tag = tag;
        assert(avail_tags.size() == 2 && avail_mps_tag != "");
        lme = me->shallow_copy(avail_tags[0]);
        rme = me->shallow_copy(avail_tags[1]);
        rme->bra = me->ket->shallow_copy(avail_mps_tag);
        lme->bra = lme->ket = rme->bra;
    }
    FLLS solve(int n_sweeps, FPS beta, bool forward = true, FPS tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.size() == 0 ? 0.0 : noises.back());
        Timer start, current;
        start.get_time();
        current.get_time();
        energies.clear();
        normsqs.clear();
        discarded_weights.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            init_moving_environments();
            for (int isw = 0; isw < n_sub_sweeps; isw++) {
                if (iprint >= 1) {
                    cout << "Sweep = " << setw(4) << iw;
                    if (n_sub_sweeps != 1)
                        cout << " (" << setw(2) << isw << "/" << setw(2)
                             << (int)n_sub_sweeps << ")";
                    cout << " | Direction = " << setw(8)
                         << (forward ? "forward" : "backward")
                         << ((mode & TETypes::ImagTE) ? " | Beta = "
                                                      : " | Tau = ")
                         << fixed << setw(10) << setprecision(5) << beta
                         << " | Bond dimension = " << setw(4)
                         << (uint32_t)bond_dims[iw]
                         << " | Noise = " << scientific << setw(9)
                         << setprecision(2) << noises[iw] << endl;
                }
                auto sweep_results = sweep(forward, isw == n_sub_sweeps - 1,
                                           beta, bond_dims[iw], noises[iw]);
                forward = !forward;
                double tswp = current.get_time();
                if (iprint >= 1) {
                    cout << "Time elapsed = " << fixed << setw(10)
                         << setprecision(3) << current.current - start.current;
                    cout << fixed << setprecision(8);
                    cout << " | E = " << setw(15) << get<0>(sweep_results);
                    cout << " | Norm^2 = " << setw(15)
                         << sqrt(get<1>(sweep_results));
                    cout << " | DW = " << scientific << setw(12)
                         << setprecision(5) << get<2>(sweep_results);
                    if (iprint >= 2) {
                        size_t dmain = frame_<FPS>()->peak_used_memory[0];
                        size_t dseco = frame_<FPS>()->peak_used_memory[1];
                        size_t imain = frame_<FPS>()->peak_used_memory[2];
                        size_t iseco = frame_<FPS>()->peak_used_memory[3];
                        cout << " | DMEM = "
                             << Parsing::to_size_string(dmain + dseco) << " ("
                             << (dmain * 100 / (dmain + dseco)) << "%)";
                        cout << " | IMEM = "
                             << Parsing::to_size_string(imain + iseco) << " ("
                             << (imain * 100 / (imain + iseco)) << "%)";
                        cout << " | "
                             << Parsing::to_size_string(sweep_cumulative_nflop,
                                                        "FLOP/SWP");
                        cout << endl << fixed << setw(10) << setprecision(3);
                        cout << "Time sweep = " << tswp;
                        if (lme != nullptr)
                            cout << " | Trot = " << lme->trot
                                 << " | Tctr = " << lme->tctr
                                 << " | Tint = " << lme->tint
                                 << " | Tmid = " << lme->tmid;
                    }
                    cout << endl;
                }
                if (isw == n_sub_sweeps - 1) {
                    energies.push_back(get<0>(sweep_results));
                    normsqs.push_back(get<1>(sweep_results));
                    discarded_weights.push_back(get<2>(sweep_results));
                }
            }
            me = lme;
            normalize();
        }
        this->forward = forward;
        return energies.back();
    }
};

// Imaginary/Real Time Evolution
template <typename S, typename FL, typename FLS> struct TimeEvolution {
    typedef typename MovingEnvironment<S, FL, FLS>::FPS FPS;
    typedef typename MovingEnvironment<S, FL, FLS>::FCS FCS;
    typedef typename MovingEnvironment<S, FL, FLS>::FLLS FLLS;
    shared_ptr<MovingEnvironment<S, FL, FLS>> me;
    vector<ubond_t> bond_dims;
    vector<FPS> noises;
    vector<FLLS> energies;
    vector<FPS> normsqs;
    vector<FPS> discarded_weights;
    vector<shared_ptr<MovingEnvironment<S, FL, FLS>>> ext_mes;
    vector<shared_ptr<MPS<S, FLS>>> ext_mpss;
    shared_ptr<EffectiveKernel<FLS>> eff_kernel = nullptr;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    TruncPatternTypes trunc_pattern = TruncPatternTypes::None;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    bool forward;
    TETypes mode;
    int n_sub_sweeps;
    vector<FPS> weights = {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0};
    uint8_t iprint = 2;
    FPS cutoff = 1E-14;
    bool normalize_mps = true;
    bool hermitian = true; //!< Whether the Hamiltonian is Hermitian (symmetric)
    size_t sweep_cumulative_nflop = 0;
    // store all wfn singular values (for analysis) at each site
    bool store_wfn_spectra = false;
    vector<vector<FPS>> sweep_wfn_spectra;
    vector<FPS> wfn_spectra;
    FPS krylov_conv_thrd = 5E-6;
    int krylov_subspace_size = 20;
    TimeEvolution(const shared_ptr<MovingEnvironment<S, FL, FLS>> &me,
                  const vector<ubond_t> &bond_dims,
                  TETypes mode = TETypes::TangentSpace, int n_sub_sweeps = 1)
        : me(me), bond_dims(bond_dims), noises(vector<FPS>{0.0}),
          forward(false), mode(mode), n_sub_sweeps(n_sub_sweeps) {}
    struct Iteration {
        FLLS energy;
        FPS normsq, error;
        int nexpo, nexpok, mmps;
        double texpo;
        size_t nflop;
        Iteration(FLLS energy, FPS normsq, FPS error, int mmps, int nexpo,
                  int nexpok, size_t nflop = 0, double texpo = 1.0)
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
    // one-site algorithm - real MPS - imag time
    Iteration update_one_dot(int i, bool forward, bool advance, FLS beta,
                             ubond_t bond_dim, FPS noise) {
        frame_<FPS>()->activate(0);
        bool fuse_left = i <= me->fuse_center;
        vector<shared_ptr<MPS<S, FLS>>> mpss = {me->ket};
        if (ext_mpss.size() != 0)
            mpss.insert(mpss.end(), ext_mpss.begin(), ext_mpss.end());
        for (auto &mps : mpss) {
            if (mps->canonical_form[i] == 'C') {
                if (i == 0)
                    mps->canonical_form[i] = 'K';
                else if (i == me->n_sites - 1)
                    mps->canonical_form[i] = 'S';
                else if (forward)
                    mps->canonical_form[i] = 'K';
                else
                    mps->canonical_form[i] = 'S';
            }
            // guess wavefunction
            // change to fused form for super-block hamiltonian
            // note that this switch exactly matches two-site conventional mpo
            // middle-site switch, so two-site conventional mpo can work
            mps->load_tensor(i);
            if ((fuse_left && mps->canonical_form[i] == 'S') ||
                (!fuse_left && mps->canonical_form[i] == 'K')) {
                shared_ptr<SparseMatrix<S, FLS>> prev_wfn = mps->tensors[i];
                if (fuse_left && mps->canonical_form[i] == 'S')
                    mps->tensors[i] =
                        MovingEnvironment<S, FL, FLS>::swap_wfn_to_fused_left(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'K')
                    mps->tensors[i] =
                        MovingEnvironment<S, FL, FLS>::swap_wfn_to_fused_right(
                            i, mps->info, prev_wfn, me->mpo->tf->opf->cg);
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
            }
        }
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<GMatrix<FL>> pdpf;
        tuple<FLS, FPS, int, size_t, double> pdi;
        vector<shared_ptr<SparseMatrix<S, FLS>>> ortho_bra;
        if (ext_mpss.size() != 0) {
            shared_ptr<VectorAllocator<FPS>> d_alloc =
                make_shared<VectorAllocator<FPS>>();
            ortho_bra.resize(ext_mpss.size());
            assert(ext_mes.size() == ext_mpss.size());
            for (size_t ist = 0; ist < ext_mes.size(); ist++) {
                ortho_bra[ist] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
                ortho_bra[ist]->allocate(me->bra->tensors[i]->info);
                shared_ptr<EffectiveHamiltonian<S, FL>> i_eff =
                    ext_mes[ist]->eff_ham(fuse_left ? FuseTypes::FuseL
                                                    : FuseTypes::FuseR,
                                          forward, false, ortho_bra[ist],
                                          ext_mpss[ist]->tensors[i]);
                auto ipdi = i_eff->multiply(ext_mes[ist]->mpo->const_e,
                                            ext_mes[ist]->para_rule);
                i_eff->deallocate();
            }
        }
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff = me->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
            me->bra->tensors[i], me->ket->tensors[i]);
        h_eff->eff_kernel = eff_kernel;
        if (!advance &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            // TangentSpace method does not allow multiple sweeps for one time
            // step
            assert(mode == TETypes::RK4);
            GMatrix<FL> tmp(nullptr, (MKL_INT)h_eff->ket->total_memory, 1);
            tmp.allocate();
            memcpy(tmp.data, h_eff->ket->data,
                   h_eff->ket->total_memory * sizeof(FLS));
            pdi = h_eff->expo_apply(
                -beta, me->mpo->const_e, hermitian, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size, ortho_bra);
            memcpy(h_eff->ket->data, tmp.data,
                   h_eff->ket->total_memory * sizeof(FLS));
            tmp.deallocate();
            auto pdp = h_eff->rk4_apply(-beta, me->mpo->const_e, false,
                                        me->para_rule, ortho_bra);
            pdpf = pdp.first;
        } else if (effective_mode == TETypes::TangentSpace)
            pdi = h_eff->expo_apply(
                -beta, me->mpo->const_e, hermitian, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size, ortho_bra);
        else if (effective_mode == TETypes::RK4) {
            auto pdp = h_eff->rk4_apply(-beta, me->mpo->const_e, false,
                                        me->para_rule, ortho_bra);
            pdpf = pdp.first;
            pdi = pdp.second;
        }
        h_eff->deallocate();
        int bdim = bond_dim, mmps = 0, expok = 0;
        FPS error = 0.0;
        shared_ptr<SparseMatrix<S, FLS>> dm = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> old_wfn;
        shared_ptr<SparseMatrix<S, FLS>> left = nullptr, right = nullptr;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                shared_ptr<SparseMatrix<S, FLS>> prev_wfn = me->ket->tensors[i];
                if (!fuse_left && forward)
                    me->ket->tensors[i] =
                        MovingEnvironment<S, FL, FLS>::swap_wfn_to_fused_left(
                            i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
                else if (fuse_left && !forward)
                    me->ket->tensors[i] =
                        MovingEnvironment<S, FL, FLS>::swap_wfn_to_fused_right(
                            i, me->ket->info, prev_wfn, me->mpo->tf->opf->cg);
                if (pdpf.size() != 0) {
                    shared_ptr<SparseMatrix<S, FLS>> tmp_wfn;
                    for (size_t ip = 0; ip < pdpf.size(); ip++) {
                        assert(pdpf[ip].size() == prev_wfn->total_memory);
                        memcpy(prev_wfn->data, pdpf[ip].data,
                               pdpf[ip].size() * sizeof(FLS));
                        if (!fuse_left && forward)
                            tmp_wfn = MovingEnvironment<S, FL, FLS>::
                                swap_wfn_to_fused_left(i, me->ket->info,
                                                       prev_wfn,
                                                       me->mpo->tf->opf->cg);
                        else if (fuse_left && !forward)
                            tmp_wfn = MovingEnvironment<S, FL, FLS>::
                                swap_wfn_to_fused_right(i, me->ket->info,
                                                        prev_wfn,
                                                        me->mpo->tf->opf->cg);
                        assert(pdpf[ip].size() == tmp_wfn->total_memory);
                        memcpy(pdpf[ip].data, tmp_wfn->data,
                               pdpf[ip].size() * sizeof(FLS));
                        tmp_wfn->info->deallocate();
                        tmp_wfn->deallocate();
                    }
                }
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
            }
        }
        for (auto &mps : ext_mpss) {
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                if (fuse_left != forward) {
                    // change to fused form for splitting
                    shared_ptr<SparseMatrix<S, FLS>> prev_wfn = mps->tensors[i];
                    if (!fuse_left && forward)
                        mps->tensors[i] = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_left(i, mps->info, prev_wfn,
                                                   me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        mps->tensors[i] = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_right(i, mps->info, prev_wfn,
                                                    me->mpo->tf->opf->cg);
                    prev_wfn->info->deallocate();
                    prev_wfn->deallocate();
                }
            }
            mps->save_tensor(i);
            mps->unload_tensor(i);
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
            if (mode != TETypes::TangentSpace) {
                mps->canonical_form[i] = forward ? 'K' : 'S';
                if (forward && i != me->n_sites - 1)
                    mps->move_right(me->mpo->tf->opf->cg, me->para_rule);
                else if (!forward && i != 0)
                    mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
            } else
                throw runtime_error(
                    "TDVP 1-dot with ext_mpss not implemented!");
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            if (pdpf.size() != 0) {
                dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                    me->ket->info->vacuum, me->ket->tensors[i], forward, noise,
                    noise_type, weights[0]);
                MovingEnvironment<S, FL, FLS>::density_matrix_add_matrices(
                    dm, me->ket->tensors[i], forward, pdpf, weights);
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            } else
                dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                    me->ket->info->vacuum, me->ket->tensors[i], forward, noise,
                    noise_type);
            // splitting of wavefunction
            old_wfn = me->ket->tensors[i];
            if ((this->trunc_pattern == TruncPatternTypes::TruncAfterOdd &&
                 i % 2 == 0) ||
                (this->trunc_pattern == TruncPatternTypes::TruncAfterEven &&
                 i % 2 == 1))
                bdim = -1;
            error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                dm, me->ket->tensors[i], bdim, forward, false, left, right,
                cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
        } else {
            if (pdpf.size() != 0) {
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            }
            old_wfn = me->ket->tensors[i];
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i != me->n_sites - 1 || !advance))
                    right->normalize();
                me->ket->tensors[i] = left;
                me->ket->save_tensor(i);
                info = left->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
            } else {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i != 0 || !advance))
                    left->normalize();
                me->ket->tensors[i] = right;
                me->ket->save_tensor(i);
                info = right->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
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
                        right = make_shared<SparseMatrix<S, FLS>>();
                        right->load_data(me->ket->get_filename(-2), true);
                    } else {
                        left = make_shared<SparseMatrix<S, FLS>>();
                        left->load_data(me->ket->get_filename(-2), true);
                    }
                }
            }
            if (forward) {
                me->ket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                me->move_to(i + 1, true);
                for (auto &xme : ext_mes)
                    xme->move_to(i + 1, true);
                shared_ptr<EffectiveHamiltonian<S, FL>> k_eff = me->eff_ham(
                    FuseTypes::NoFuseL, forward, true, right, right);
                k_eff->eff_kernel = eff_kernel;
                auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, hermitian,
                                             iprint >= 3, me->para_rule,
                                             krylov_conv_thrd,
                                             krylov_subspace_size, ortho_bra);
                k_eff->deallocate();
                if (me->para_rule == nullptr || me->para_rule->is_root()) {
                    if (normalize_mps)
                        right->normalize();
                    get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
                    expok = get<2>(pdk);
                    MovingEnvironment<S, FL, FLS>::contract_one_dot(
                        i + 1, right, me->ket, forward);
                    me->ket->save_tensor(i + 1);
                    me->ket->unload_tensor(i + 1);
                }
            } else {
                me->ket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                me->move_to(i - 1, true);
                for (auto &xme : ext_mes)
                    xme->move_to(i - 1, true);
                shared_ptr<EffectiveHamiltonian<S, FL>> k_eff =
                    me->eff_ham(FuseTypes::NoFuseR, forward, true, left, left);
                k_eff->eff_kernel = eff_kernel;
                auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, hermitian,
                                             iprint >= 3, me->para_rule,
                                             krylov_conv_thrd,
                                             krylov_subspace_size, ortho_bra);
                k_eff->deallocate();
                if (me->para_rule == nullptr || me->para_rule->is_root()) {
                    if (normalize_mps)
                        left->normalize();
                    get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
                    expok = get<2>(pdk);
                    MovingEnvironment<S, FL, FLS>::contract_one_dot(
                        i - 1, left, me->ket, forward);
                    me->ket->save_tensor(i - 1);
                    me->ket->unload_tensor(i - 1);
                }
            }
        } else {
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                // propagation
                if (forward) {
                    if (i != me->n_sites - 1) {
                        MovingEnvironment<S, FL, FLS>::contract_one_dot(
                            i + 1, right, me->ket, forward);
                        me->ket->save_tensor(i + 1);
                        me->ket->unload_tensor(i + 1);
                    } else {
                        me->ket->tensors[i] =
                            make_shared<SparseMatrix<S, FLS>>();
                        MovingEnvironment<S, FL, FLS>::contract_one_dot(
                            i, right, me->ket, !forward);
                        me->ket->save_tensor(i);
                        me->ket->unload_tensor(i);
                    }
                } else {
                    if (i > 0) {
                        MovingEnvironment<S, FL, FLS>::contract_one_dot(
                            i - 1, left, me->ket, forward);
                        me->ket->save_tensor(i - 1);
                        me->ket->unload_tensor(i - 1);
                    } else {
                        me->ket->tensors[i] =
                            make_shared<SparseMatrix<S, FLS>>();
                        MovingEnvironment<S, FL, FLS>::contract_one_dot(
                            i, left, me->ket, !forward);
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
        me->ket->save_data();
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration((FLLS)get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, mmps, get<2>(pdi),
                         expok, get<3>(pdi), get<4>(pdi));
    }
    // two-site algorithm - real MPS - imag time
    Iteration update_two_dot(int i, bool forward, bool advance, FLS beta,
                             ubond_t bond_dim, FPS noise) {
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MPS<S, FLS>>> mpss = {me->ket};
        if (ext_mpss.size() != 0)
            mpss.insert(mpss.end(), ext_mpss.begin(), ext_mpss.end());
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_two_dot(i, mps);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        tuple<FLS, FPS, int, size_t, double> pdi;
        shared_ptr<SparseMatrix<S, FLS>> old_wfn = me->ket->tensors[i];
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<GMatrix<FL>> pdpf;
        vector<shared_ptr<SparseMatrix<S, FLS>>> ortho_bra;
        if (ext_mpss.size() != 0) {
            shared_ptr<VectorAllocator<FPS>> d_alloc =
                make_shared<VectorAllocator<FPS>>();
            ortho_bra.resize(ext_mpss.size());
            assert(ext_mes.size() == ext_mpss.size());
            for (size_t ist = 0; ist < ext_mes.size(); ist++) {
                ortho_bra[ist] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
                ortho_bra[ist]->allocate(me->bra->tensors[i]->info);
                shared_ptr<EffectiveHamiltonian<S, FL>> i_eff =
                    ext_mes[ist]->eff_ham(FuseTypes::FuseLR, forward, false,
                                          ortho_bra[ist],
                                          ext_mpss[ist]->tensors[i]);
                auto ipdi = i_eff->multiply(ext_mes[ist]->mpo->const_e,
                                            ext_mes[ist]->para_rule);
                i_eff->deallocate();
            }
        }
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, forward, true, me->bra->tensors[i],
                        me->ket->tensors[i]);
        h_eff->eff_kernel = eff_kernel;
        if (!advance &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            // TangentSpace method does not allow multiple sweeps for one time
            // step
            assert(mode == TETypes::RK4);
            GMatrix<FL> tmp(nullptr, (MKL_INT)h_eff->ket->total_memory, 1);
            tmp.allocate();
            memcpy(tmp.data, h_eff->ket->data,
                   h_eff->ket->total_memory * sizeof(FLS));
            pdi = h_eff->expo_apply(
                -beta, me->mpo->const_e, hermitian, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size, ortho_bra);
            memcpy(h_eff->ket->data, tmp.data,
                   h_eff->ket->total_memory * sizeof(FLS));
            tmp.deallocate();
            auto pdp = h_eff->rk4_apply(-beta, me->mpo->const_e, false,
                                        me->para_rule, ortho_bra);
            pdpf = pdp.first;
        } else if (effective_mode == TETypes::TangentSpace)
            pdi = h_eff->expo_apply(
                -beta, me->mpo->const_e, hermitian, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size, ortho_bra);
        else if (effective_mode == TETypes::RK4) {
            auto pdp = h_eff->rk4_apply(-beta, me->mpo->const_e, false,
                                        me->para_rule, ortho_bra);
            pdpf = pdp.first;
            pdi = pdp.second;
        }
        h_eff->deallocate();
        vector<shared_ptr<MPS<S, FLS>>> rev_ext_mpss(ext_mpss.rbegin(),
                                                     ext_mpss.rend());
        for (auto &mps : rev_ext_mpss) {
            mps->save_tensor(i);
            mps->unload_tensor(i);
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
            if (forward) {
                mps->canonical_form[i] = 'C';
                mps->move_right(me->mpo->tf->opf->cg, me->para_rule);
                mps->canonical_form[i + 1] = 'C';
                if (mps->center == mps->n_sites - 1)
                    mps->center = mps->n_sites - 2;
            } else {
                mps->canonical_form[i] = 'C';
                mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
                mps->canonical_form[i] = 'C';
            }
        }
        int bdim = bond_dim, mmps = 0;
        FPS error = 0.0;
        shared_ptr<SparseMatrix<S, FLS>> dm;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            if (pdpf.size() != 0) {
                dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                    me->ket->info->vacuum, h_eff->ket, forward, noise,
                    noise_type, weights[0]);
                MovingEnvironment<S, FL, FLS>::density_matrix_add_matrices(
                    dm, h_eff->ket, forward, pdpf, weights);
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            } else
                dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                    me->ket->info->vacuum, h_eff->ket, forward, noise,
                    noise_type);
            if ((this->trunc_pattern == TruncPatternTypes::TruncAfterOdd &&
                 i % 2 == 0) ||
                (this->trunc_pattern == TruncPatternTypes::TruncAfterEven &&
                 i % 2 == 1))
                bdim = -1;
            error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                dm, h_eff->ket, bdim, forward, false, me->ket->tensors[i],
                me->ket->tensors[i + 1], cutoff, store_wfn_spectra, wfn_spectra,
                trunc_type);
        } else {
            if (pdpf.size() != 0) {
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            }
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i + 1 != me->n_sites - 1 || !advance))
                    me->ket->tensors[i + 1]->normalize();
                info = me->ket->tensors[i]->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'C';
            } else {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i != 0 || !advance))
                    me->ket->tensors[i]->normalize();
                info =
                    me->ket->tensors[i + 1]->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
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
            me->ket->tensors[i + 1] = make_shared<SparseMatrix<S, FLS>>();
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
            me->move_to(i + 1, true);
            for (auto &xme : ext_mes)
                xme->move_to(i + 1, true);
            me->ket->load_tensor(i + 1);
            for (auto &mps : ext_mpss)
                mps->load_tensor(i + 1);
            if (ext_mpss.size() != 0) {
                shared_ptr<VectorAllocator<FPS>> d_alloc =
                    make_shared<VectorAllocator<FPS>>();
                for (size_t ist = 0; ist < ext_mes.size(); ist++) {
                    ortho_bra[ist] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
                    ortho_bra[ist]->allocate(me->bra->tensors[i + 1]->info);
                    shared_ptr<EffectiveHamiltonian<S, FL>> i_eff =
                        ext_mes[ist]->eff_ham(FuseTypes::FuseR, forward, true,
                                              ortho_bra[ist],
                                              ext_mpss[ist]->tensors[i + 1]);
                    auto ipdi = i_eff->multiply(ext_mes[ist]->mpo->const_e,
                                                ext_mes[ist]->para_rule);
                    i_eff->deallocate();
                }
            }
            shared_ptr<EffectiveHamiltonian<S, FL>> k_eff =
                me->eff_ham(FuseTypes::FuseR, forward, true,
                            me->bra->tensors[i + 1], me->ket->tensors[i + 1]);
            k_eff->eff_kernel = eff_kernel;
            auto pdk = k_eff->expo_apply(
                beta, me->mpo->const_e, hermitian, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size, ortho_bra);
            k_eff->deallocate();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                if (normalize_mps)
                    me->ket->tensors[i + 1]->normalize();
                me->ket->save_tensor(i + 1);
                for (auto &mps : ext_mpss)
                    mps->save_tensor(i + 1);
            }
            for (auto &mps : rev_ext_mpss)
                mps->unload_tensor(i + 1);
            me->ket->unload_tensor(i + 1);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        } else if (mode == TETypes::TangentSpace && !forward && i != 0) {
            me->move_to(i - 1, true);
            for (auto &xme : ext_mes)
                xme->move_to(i - 1, true);
            me->ket->load_tensor(i);
            for (auto &mps : ext_mpss)
                mps->load_tensor(i);
            if (ext_mpss.size() != 0) {
                shared_ptr<VectorAllocator<FPS>> d_alloc =
                    make_shared<VectorAllocator<FPS>>();
                for (size_t ist = 0; ist < ext_mes.size(); ist++) {
                    ortho_bra[ist] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
                    ortho_bra[ist]->allocate(me->bra->tensors[i]->info);
                    shared_ptr<EffectiveHamiltonian<S, FL>> i_eff =
                        ext_mes[ist]->eff_ham(FuseTypes::FuseL, forward, true,
                                              ortho_bra[ist],
                                              ext_mpss[ist]->tensors[i]);
                    auto ipdi = i_eff->multiply(ext_mes[ist]->mpo->const_e,
                                                ext_mes[ist]->para_rule);
                    i_eff->deallocate();
                }
            }
            shared_ptr<EffectiveHamiltonian<S, FL>> k_eff =
                me->eff_ham(FuseTypes::FuseL, forward, true,
                            me->bra->tensors[i], me->ket->tensors[i]);
            k_eff->eff_kernel = eff_kernel;
            auto pdk = k_eff->expo_apply(
                beta, me->mpo->const_e, hermitian, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size, ortho_bra);
            k_eff->deallocate();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                if (normalize_mps)
                    me->ket->tensors[i]->normalize();
                me->ket->save_tensor(i);
                for (auto &mps : ext_mpss)
                    mps->save_tensor(i);
            }
            for (auto &mps : rev_ext_mpss)
                mps->unload_tensor(i);
            me->ket->unload_tensor(i);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        }
        if (me->para_rule == nullptr || me->para_rule->is_root())
            for (auto &mps : mpss)
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, 0, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
        me->ket->save_data();
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration((FLLS)get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, mmps, get<2>(pdi),
                         expok, get<3>(pdi), get<4>(pdi));
    }
    // one-site algorithm - complex MPS - complex time
    // canonical form for wavefunction: J = left-fused, T = right-fused
    Iteration update_multi_one_dot(int i, bool forward, bool advance, FCS beta,
                                   ubond_t bond_dim, FPS noise) {
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        frame_<FPS>()->activate(0);
        bool fuse_left = i <= me->fuse_center;
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
            vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_wfns =
                mket->wfns;
            if (fuse_left && mket->canonical_form[i] == 'T')
                mket->wfns =
                    MovingEnvironment<S, FL, FLS>::swap_multi_wfn_to_fused_left(
                        i, mket->info, prev_wfns, me->mpo->tf->opf->cg);
            else if (!fuse_left && mket->canonical_form[i] == 'J')
                mket->wfns = MovingEnvironment<S, FL, FLS>::
                    swap_multi_wfn_to_fused_right(i, mket->info, prev_wfns,
                                                  me->mpo->tf->opf->cg);
            for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                prev_wfns[j]->deallocate();
            if (prev_wfns.size() != 0)
                prev_wfns[0]->deallocate_infos();
        }
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<GMatrix<FL>> pdpf;
        tuple<FLS, FPS, int, size_t, double> pdi;
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> h_eff =
            me->multi_eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                              forward, true);
        if (!advance &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            // TangentSpace method does not allow multiple sweeps for one time
            // step
            assert(mode == TETypes::RK4);
            GMatrix<FLS> tmp_re(nullptr, (MKL_INT)h_eff->ket[0]->total_memory,
                                1);
            GMatrix<FLS> tmp_im(nullptr, (MKL_INT)h_eff->ket[1]->total_memory,
                                1);
            tmp_re.allocate();
            tmp_im.allocate();
            memcpy(tmp_re.data, h_eff->ket[0]->data,
                   h_eff->ket[0]->total_memory * sizeof(FLS));
            memcpy(tmp_im.data, h_eff->ket[1]->data,
                   h_eff->ket[1]->total_memory * sizeof(FLS));
            pdi = EffectiveFunctions<S, FL>::expo_apply(
                h_eff, -beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size);
            memcpy(h_eff->ket[0]->data, tmp_re.data,
                   h_eff->ket[0]->total_memory * sizeof(FLS));
            memcpy(h_eff->ket[1]->data, tmp_im.data,
                   h_eff->ket[1]->total_memory * sizeof(FLS));
            tmp_im.deallocate();
            tmp_re.deallocate();
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
        } else if (effective_mode == TETypes::TangentSpace)
            pdi = EffectiveFunctions<S, FL>::expo_apply(
                h_eff, -beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size);
        else if (effective_mode == TETypes::RK4) {
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
            pdi = pdp.second;
        }
        h_eff->deallocate();
        int bdim = bond_dim, mmps = 0, expok = 0;
        FPS error = 0.0;
        vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfns, new_wfns;
        shared_ptr<SparseMatrix<S, FLS>> dm, rot;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_wfns =
                    mket->wfns;
                if (!fuse_left && forward)
                    mket->wfns = MovingEnvironment<S, FL, FLS>::
                        swap_multi_wfn_to_fused_left(i, mket->info, prev_wfns,
                                                     me->mpo->tf->opf->cg);
                else if (fuse_left && !forward)
                    mket->wfns = MovingEnvironment<S, FL, FLS>::
                        swap_multi_wfn_to_fused_right(i, mket->info, prev_wfns,
                                                      me->mpo->tf->opf->cg);
                if (pdpf.size() != 0) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> tmp_wfns;
                    size_t np = prev_wfns.size() * prev_wfns[0]->n;
                    for (size_t ip = 0; ip < pdpf.size(); ip += np) {
                        for (size_t ipx = 0, ipk = 0; ipx < prev_wfns.size();
                             ipx++)
                            for (int ipy = 0; ipy < prev_wfns[ipx]->n;
                                 ipy++, ipk++) {
                                assert(pdpf[ip + ipk].size() ==
                                       (*prev_wfns[ipx])[ipy]->total_memory);
                                memcpy((*prev_wfns[ipx])[ipy]->data,
                                       pdpf[ip + ipk].data,
                                       pdpf[ip + ipk].size() * sizeof(FLS));
                            }
                        if (!fuse_left && forward)
                            tmp_wfns = MovingEnvironment<S, FL, FLS>::
                                swap_multi_wfn_to_fused_left(
                                    i, mket->info, prev_wfns,
                                    me->mpo->tf->opf->cg);
                        else if (fuse_left && !forward)
                            tmp_wfns = MovingEnvironment<S, FL, FLS>::
                                swap_multi_wfn_to_fused_right(
                                    i, mket->info, prev_wfns,
                                    me->mpo->tf->opf->cg);
                        for (size_t ipx = 0, ipk = 0; ipx < tmp_wfns.size();
                             ipx++)
                            for (int ipy = 0; ipy < tmp_wfns[ipx]->n;
                                 ipy++, ipk++) {
                                assert(pdpf[ip + ipk].size() ==
                                       (*tmp_wfns[ipx])[ipy]->total_memory);
                                memcpy(pdpf[ip + ipk].data,
                                       (*tmp_wfns[ipx])[ipy]->data,
                                       pdpf[ip + ipk].size() * sizeof(FLS));
                            }
                        for (int j = (int)tmp_wfns.size() - 1; j >= 0; j--)
                            tmp_wfns[j]->deallocate();
                        if (tmp_wfns.size() != 0)
                            tmp_wfns[0]->deallocate_infos();
                    }
                }
                for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                    prev_wfns[j]->deallocate();
                if (prev_wfns.size() != 0)
                    prev_wfns[0]->deallocate_infos();
            }
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            old_wfns = mket->wfns;
            if (pdpf.size() != 0) {
                dm = MovingEnvironment<S, FL, FLS>::
                    density_matrix_with_multi_target(
                        mket->info->vacuum, mket->wfns, mket->weights, forward,
                        noise, noise_type, weights[0]);
                MovingEnvironment<S, FL, FLS>::density_matrix_add_matrix_groups(
                    dm, mket->wfns, forward, pdpf, weights);
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            } else
                dm = MovingEnvironment<S, FL, FLS>::
                    density_matrix_with_multi_target(
                        me->ket->info->vacuum, mket->wfns, mket->weights,
                        forward, noise, noise_type);
            // splitting of wavefunction
            if ((this->trunc_pattern == TruncPatternTypes::TruncAfterOdd &&
                 i % 2 == 0) ||
                (this->trunc_pattern == TruncPatternTypes::TruncAfterEven &&
                 i % 2 == 1))
                bdim = -1;
            error = MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                dm, mket->wfns, bdim, forward, false, new_wfns, rot, cutoff,
                store_wfn_spectra, wfn_spectra, trunc_type);
        } else {
            if (pdpf.size() != 0) {
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            }
            old_wfns = mket->wfns;
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<StateInfo<S>> info = nullptr;
            // propagation
            if (forward) {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i != me->n_sites - 1 || !advance))
                    SparseMatrixGroup<S, FLS>::normalize_all(new_wfns);
                mket->tensors[i] = rot;
                mket->save_tensor(i);
                info = rot->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                mket->info->bond_dim = max(mket->info->bond_dim, (ubond_t)mmps);
                mket->info->left_dims[i + 1] = info;
                mket->info->save_left_dims(i + 1);
            } else {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i != 0 || !advance))
                    SparseMatrixGroup<S, FLS>::normalize_all(new_wfns);
                mket->tensors[i] = rot;
                mket->save_tensor(i);
                info = rot->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                mket->info->bond_dim = max(mket->info->bond_dim, (ubond_t)mmps);
                mket->info->right_dims[i] = info;
                mket->info->save_right_dims(i);
            }
            info->deallocate();
        }
        if (mode == TETypes::TangentSpace &&
            ((forward && i != me->n_sites - 1) || (!forward && i != 0))) {
            shared_ptr<VectorAllocator<FPS>> d_alloc =
                make_shared<VectorAllocator<FPS>>();
            if (me->para_rule != nullptr) {
                if (me->para_rule->is_root())
                    for (size_t j = 0; j < new_wfns.size(); j++)
                        new_wfns[j]->save_data(
                            mket->get_wfn_filename((int)j -
                                                   (int)new_wfns.size()),
                            j == 0);
                me->para_rule->comm->barrier();
                if (!me->para_rule->is_root()) {
                    new_wfns.resize(old_wfns.size());
                    shared_ptr<VectorAllocator<uint32_t>> i_alloc =
                        make_shared<VectorAllocator<uint32_t>>();
                    for (size_t j = 0; j < new_wfns.size(); j++) {
                        new_wfns[j] =
                            make_shared<SparseMatrixGroup<S, FLS>>(d_alloc);
                        new_wfns[j]->load_data(
                            mket->get_wfn_filename((int)j -
                                                   (int)new_wfns.size()),
                            j == 0, i_alloc);
                        new_wfns[j]->infos = new_wfns[0]->infos;
                    }
                }
            }
            for (size_t j = 0; j < mket->wfns.size(); j++)
                mket->wfns[j] = make_shared<SparseMatrixGroup<S, FLS>>(d_alloc);
            mket->tensors[i] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
            if (forward) {
                me->move_to(i + 1, true);
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> kwfns =
                    mket->wfns;
                mket->wfns = new_wfns;
                shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> k_eff =
                    me->multi_eff_ham(FuseTypes::NoFuseL, forward, true);
                auto pdk = EffectiveFunctions<S, FL>::expo_apply(
                    k_eff, beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                    krylov_conv_thrd, krylov_subspace_size);
                k_eff->deallocate();
                mket->wfns = kwfns;
                if (me->para_rule == nullptr || me->para_rule->is_root()) {
                    if (normalize_mps)
                        SparseMatrixGroup<S, FLS>::normalize_all(new_wfns);
                    get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
                    expok = get<2>(pdk);
                    MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                        i + 1, new_wfns, mket, forward);
                    mket->save_wavefunction(i + 1);
                    mket->unload_wavefunction(i + 1);
                }
            } else {
                me->move_to(i - 1, true);
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> kwfns =
                    mket->wfns;
                mket->wfns = new_wfns;
                shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> k_eff =
                    me->multi_eff_ham(FuseTypes::NoFuseR, forward, true);
                auto pdk = EffectiveFunctions<S, FL>::expo_apply(
                    k_eff, beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                    krylov_conv_thrd, krylov_subspace_size);
                k_eff->deallocate();
                mket->wfns = kwfns;
                if (me->para_rule == nullptr || me->para_rule->is_root()) {
                    if (normalize_mps)
                        SparseMatrixGroup<S, FLS>::normalize_all(new_wfns);
                    get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
                    expok = get<2>(pdk);
                    MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                        i - 1, new_wfns, mket, forward);
                    mket->save_wavefunction(i - 1);
                    mket->unload_wavefunction(i - 1);
                }
            }
        } else {
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                // propagation
                if (forward) {
                    if (i != me->n_sites - 1) {
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i + 1, new_wfns, mket, forward);
                        mket->save_wavefunction(i + 1);
                        mket->unload_wavefunction(i + 1);
                    } else {
                        mket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i, new_wfns, mket, !forward);
                        mket->save_wavefunction(i);
                        mket->unload_wavefunction(i);
                    }
                } else {
                    if (i > 0) {
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i - 1, new_wfns, mket, forward);
                        mket->save_wavefunction(i - 1);
                        mket->unload_wavefunction(i - 1);
                    } else {
                        mket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i, new_wfns, mket, !forward);
                        mket->save_wavefunction(i);
                        mket->unload_wavefunction(i);
                    }
                }
            }
        }
        if (forward) {
            for (int j = (int)new_wfns.size() - 1; j >= 0; j--)
                new_wfns[j]->deallocate();
            if (new_wfns.size() != 0)
                new_wfns[0]->deallocate_infos();
            if (rot != nullptr) {
                rot->info->deallocate();
                rot->deallocate();
            }
        } else {
            if (rot != nullptr) {
                rot->info->deallocate();
                rot->deallocate();
            }
            for (int j = (int)new_wfns.size() - 1; j >= 0; j--)
                new_wfns[j]->deallocate();
            if (new_wfns.size() != 0)
                new_wfns[0]->deallocate_infos();
        }
        if (dm != nullptr) {
            dm->info->deallocate();
            dm->deallocate();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (int j = (int)old_wfns.size() - 1; j >= 0; j--)
                old_wfns[j]->deallocate();
            if (old_wfns.size() != 0)
                old_wfns[0]->deallocate_infos();
            if (forward) {
                if (i != me->n_sites - 1) {
                    mket->canonical_form[i] = 'L';
                    mket->canonical_form[i + 1] = 'T';
                } else
                    mket->canonical_form[i] = 'J';
            } else {
                if (i > 0) {
                    mket->canonical_form[i - 1] = 'J';
                    mket->canonical_form[i] = 'R';
                } else
                    mket->canonical_form[i] = 'T';
            }
        } else {
            if (forward) {
                if (i != me->n_sites - 1) {
                    mket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                    mket->tensors[i + 1] = nullptr;
                    mket->canonical_form[i] = 'L';
                    mket->canonical_form[i + 1] = 'T';
                } else
                    mket->canonical_form[i] = 'J';
            } else {
                if (i > 0) {
                    mket->tensors[i - 1] = nullptr;
                    mket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                    mket->canonical_form[i - 1] = 'J';
                    mket->canonical_form[i] = 'R';
                } else
                    mket->canonical_form[i] = 'T';
            }
        }
        mket->save_data();
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration((FLLS)get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, mmps, get<2>(pdi),
                         expok, get<3>(pdi), get<4>(pdi));
    }
    // two-site algorithm - complex MPS - complex time
    // canonical form for wavefunction: M = multi center
    Iteration update_multi_two_dot(int i, bool forward, bool advance, FCS beta,
                                   ubond_t bond_dim, FPS noise) {
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        frame_<FPS>()->activate(0);
        if (mket->tensors[i] != nullptr || mket->tensors[i + 1] != nullptr)
            MovingEnvironment<S, FL, FLS>::contract_multi_two_dot(i, mket);
        else {
            mket->load_tensor(i);
            mket->tensors[i] = mket->tensors[i + 1] = nullptr;
        }
        tuple<FLS, FPS, int, size_t, double> pdi;
        vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfns = mket->wfns;
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<GMatrix<FLS>> pdpf;
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> h_eff =
            me->multi_eff_ham(FuseTypes::FuseLR, forward, true);
        if (!advance &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            // TangentSpace method does not allow multiple sweeps for one time
            // step
            assert(mode == TETypes::RK4);
            GMatrix<FLS> tmp_re(nullptr, (MKL_INT)h_eff->ket[0]->total_memory,
                                1);
            GMatrix<FLS> tmp_im(nullptr, (MKL_INT)h_eff->ket[1]->total_memory,
                                1);
            tmp_re.allocate();
            tmp_im.allocate();
            memcpy(tmp_re.data, h_eff->ket[0]->data,
                   h_eff->ket[0]->total_memory * sizeof(FLS));
            memcpy(tmp_im.data, h_eff->ket[1]->data,
                   h_eff->ket[1]->total_memory * sizeof(FLS));
            pdi = EffectiveFunctions<S, FL>::expo_apply(
                h_eff, -beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size);
            memcpy(h_eff->ket[0]->data, tmp_re.data,
                   h_eff->ket[0]->total_memory * sizeof(FLS));
            memcpy(h_eff->ket[1]->data, tmp_im.data,
                   h_eff->ket[1]->total_memory * sizeof(FLS));
            tmp_im.deallocate();
            tmp_re.deallocate();
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
        } else if (effective_mode == TETypes::TangentSpace)
            pdi = EffectiveFunctions<S, FL>::expo_apply(
                h_eff, -beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size);
        else if (effective_mode == TETypes::RK4) {
            auto pdp =
                h_eff->rk4_apply(-beta, me->mpo->const_e, false, me->para_rule);
            pdpf = pdp.first;
            pdi = pdp.second;
        }
        h_eff->deallocate();
        int bdim = bond_dim, mmps = 0;
        FPS error = 0.0;
        shared_ptr<SparseMatrix<S, FLS>> dm;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            if (pdpf.size() != 0) {
                dm = MovingEnvironment<S, FL, FLS>::
                    density_matrix_with_multi_target(
                        mket->info->vacuum, old_wfns, mket->weights, forward,
                        noise, noise_type, weights[0]);
                MovingEnvironment<S, FL, FLS>::density_matrix_add_matrix_groups(
                    dm, old_wfns, forward, pdpf, weights);
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            } else
                dm = MovingEnvironment<S, FL, FLS>::
                    density_matrix_with_multi_target(
                        mket->info->vacuum, old_wfns, mket->weights, forward,
                        noise, noise_type);
            if ((this->trunc_pattern == TruncPatternTypes::TruncAfterOdd &&
                 i % 2 == 0) ||
                (this->trunc_pattern == TruncPatternTypes::TruncAfterEven &&
                 i % 2 == 1))
                bdim = -1;
            error = MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                dm, old_wfns, bdim, forward, false, mket->wfns,
                forward ? mket->tensors[i] : mket->tensors[i + 1], cutoff,
                store_wfn_spectra, wfn_spectra, trunc_type);
        } else {
            if (pdpf.size() != 0) {
                frame_<FPS>()->activate(1);
                for (int i = (int)pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame_<FPS>()->activate(0);
            }
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i + 1 != me->n_sites - 1 || !advance))
                    SparseMatrixGroup<S, FLS>::normalize_all(mket->wfns);
                info = me->ket->tensors[i]->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'M';
            } else {
                if (normalize_mps && mode == TETypes::RK4 &&
                    (i != 0 || !advance))
                    SparseMatrixGroup<S, FLS>::normalize_all(mket->wfns);
                info =
                    me->ket->tensors[i + 1]->info->extract_state_info(forward);
                mmps = (int)info->n_states_total;
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
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
        } else {
            if (forward) {
                me->ket->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                me->ket->tensors[i + 1] = nullptr;
                me->ket->canonical_form[i] = 'L';
                me->ket->canonical_form[i + 1] = 'M';
            } else {
                me->ket->tensors[i] = nullptr;
                me->ket->tensors[i + 1] = make_shared<SparseMatrix<S, FLS>>();
                me->ket->canonical_form[i] = 'M';
                me->ket->canonical_form[i + 1] = 'R';
            }
        }
        for (int k = mket->nroots - 1; k >= 0; k--)
            old_wfns[k]->deallocate();
        old_wfns[0]->deallocate_infos();
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        int expok = 0;
        if (mode == TETypes::TangentSpace && forward &&
            i + 1 != me->n_sites - 1) {
            me->move_to(i + 1, true);
            mket->load_wavefunction(i + 1);
            shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> k_eff =
                me->multi_eff_ham(FuseTypes::FuseR, forward, true);
            auto pdk = EffectiveFunctions<S, FL>::expo_apply(
                k_eff, beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size);
            k_eff->deallocate();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                if (normalize_mps)
                    SparseMatrixGroup<S, FLS>::normalize_all(mket->wfns);
                mket->save_wavefunction(i + 1);
            }
            mket->unload_wavefunction(i + 1);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        } else if (mode == TETypes::TangentSpace && !forward && i != 0) {
            me->move_to(i - 1, true);
            mket->load_wavefunction(i);
            shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> k_eff =
                me->multi_eff_ham(FuseTypes::FuseL, forward, true);
            auto pdk = EffectiveFunctions<S, FL>::expo_apply(
                k_eff, beta, me->mpo->const_e, iprint >= 3, me->para_rule,
                krylov_conv_thrd, krylov_subspace_size);
            k_eff->deallocate();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                if (normalize_mps)
                    SparseMatrixGroup<S, FLS>::normalize_all(mket->wfns);
                mket->save_wavefunction(i);
            }
            mket->unload_wavefunction(i);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        }
        if (me->para_rule == nullptr || me->para_rule->is_root())
            MovingEnvironment<S, FL, FLS>::propagate_multi_wfn(
                i, 0, me->n_sites, mket, forward, me->mpo->tf->opf->cg);
        mket->save_data();
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration((FLLS)get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, mmps, get<2>(pdi),
                         expok, get<3>(pdi), get<4>(pdi));
    }
    Iteration blocking(int i, bool forward, bool advance, FCS beta,
                       ubond_t bond_dim, FPS noise) {
        me->move_to(i);
        for (auto &xme : ext_mes)
            xme->move_to(i);
        assert(me->dot == 2 || me->dot == 1);
        Iteration it(0, 0, 0, 0, 0, 0);
        if (me->dot == 2) {
            if (me->ket->canonical_form[i] == 'M' ||
                me->ket->canonical_form[i + 1] == 'M' ||
                me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T')
                it = update_multi_two_dot(i, forward, advance, beta, bond_dim,
                                          noise);
            else if (beta.imag() == 0)
                it = update_two_dot(i, forward, advance, beta.real(), bond_dim,
                                    noise);
            else if (is_same<FLS, FCS>::value)
                it = update_two_dot(i, forward, advance, (FLS &)beta, bond_dim,
                                    noise);
            else
                throw runtime_error("Cannot do real TE for real MPS!");
        } else {
            if (me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T' ||
                me->ket->canonical_form[i] == 'M')
                it = update_multi_one_dot(i, forward, advance, beta, bond_dim,
                                          noise);
            else if (beta.imag() == 0)
                it = update_one_dot(i, forward, advance, beta.real(), bond_dim,
                                    noise);
            else if (is_same<FLS, FCS>::value)
                it = update_one_dot(i, forward, advance, (FLS &)beta, bond_dim,
                                    noise);
            else
                throw runtime_error("Cannot do real TE for real MPS!");
        }
        if (store_wfn_spectra) {
            const int bond_update_idx =
                me->dot == 1 && !forward && i > 0 ? i - 1 : i;
            if (sweep_wfn_spectra.size() <= bond_update_idx)
                sweep_wfn_spectra.resize(bond_update_idx + 1);
            sweep_wfn_spectra[bond_update_idx] = wfn_spectra;
        }
        return it;
    }
    tuple<FLLS, FPS, FPS> sweep(bool forward, bool advance, FCS beta,
                                ubond_t bond_dim, FPS noise) {
        frame_<FPS>()->twrite = frame_<FPS>()->tread = frame_<FPS>()->tasync =
            0;
        frame_<FPS>()->fpwrite = frame_<FPS>()->fpread = 0;
        if (frame_<FPS>()->fp_codec != nullptr)
            frame_<FPS>()->fp_codec->ndata = frame_<FPS>()->fp_codec->ncpsd = 0;
        me->prepare();
        for (auto &xme : ext_mes)
            xme->prepare();
        vector<FLLS> energies;
        vector<FPS> normsqs;
        sweep_cumulative_nflop = 0;
        vector<int> sweep_range;
        FPS largest_error = 0.0;
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
            sweep_cumulative_nflop += r.nflop;
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
        if (me->ket->get_type() & MPSTypes::MultiWfn) {
            shared_ptr<MultiMPS<S, FLS>> mket =
                dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
            size_t center = mket->canonical_form.find('M');
            if (center == string::npos)
                center = mket->canonical_form.find('J');
            if (center == string::npos)
                center = mket->canonical_form.find('T');
            assert(center != string::npos);
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                mket->load_wavefunction((int)center);
                SparseMatrixGroup<S, FLS>::normalize_all(mket->wfns);
                mket->save_wavefunction((int)center);
                mket->unload_wavefunction((int)center);
            }
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
        } else {
            size_t center = me->ket->canonical_form.find('C');
            if (center == string::npos)
                center = me->ket->canonical_form.find('K');
            if (center == string::npos)
                center = me->ket->canonical_form.find('S');
            assert(center != string::npos);
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                me->ket->load_tensor((int)center);
                me->ket->tensors[center]->normalize();
                me->ket->save_tensor((int)center);
                me->ket->unload_tensor((int)center);
            }
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
        }
    }
    FLLS solve(int n_sweeps, FCS beta, bool forward = true, FPS tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        current.get_time();
        energies.clear();
        normsqs.clear();
        discarded_weights.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            for (int isw = 0; isw < n_sub_sweeps; isw++) {
                if (iprint >= 1) {
                    cout << "Sweep = " << setw(4) << iw;
                    if (n_sub_sweeps != 1)
                        cout << " (" << setw(2) << isw << "/" << setw(2)
                             << (int)n_sub_sweeps << ")";
                    cout << " | Direction = " << setw(8)
                         << (forward ? "forward" : "backward")
                         << " | Beta = " << fixed << setw(15) << setprecision(5)
                         << beta << " | Bond dimension = " << setw(4)
                         << (uint32_t)bond_dims[iw]
                         << " | Noise = " << scientific << setw(9)
                         << setprecision(2) << noises[iw] << endl;
                }
                auto r = sweep(forward, isw == n_sub_sweeps - 1, beta,
                               bond_dims[iw], noises[iw]);
                forward = !forward;
                FPS tswp = current.get_time();
                if (iprint >= 1) {
                    cout << "Time elapsed = " << fixed << setw(10)
                         << setprecision(3) << current.current - start.current;
                    cout << fixed << setprecision(10);
                    cout << " | E = " << setw(18) << get<0>(r);
                    cout << " | Norm^2 = " << setw(18) << get<1>(r);
                    cout << " | DW = " << setw(9) << setprecision(5)
                         << scientific << get<2>(r) << endl;
                }
                if (iprint >= 2) {
                    cout << fixed << setprecision(3);
                    cout << "Time sweep = " << setw(12) << tswp;
                    cout << " | "
                         << Parsing::to_size_string(sweep_cumulative_nflop,
                                                    "FLOP/SWP")
                         << endl;
                    cout << " | Tread = " << frame_<FPS>()->tread
                         << " | Twrite = " << frame_<FPS>()->twrite
                         << " | Tfpread = " << frame_<FPS>()->fpread
                         << " | Tfpwrite = " << frame_<FPS>()->fpwrite;
                    if (frame_<FPS>()->fp_codec != nullptr)
                        cout
                            << " | data = "
                            << Parsing::to_size_string(
                                   frame_<FPS>()->fp_codec->ndata * sizeof(FPS))
                            << " | cpsd = "
                            << Parsing::to_size_string(
                                   frame_<FPS>()->fp_codec->ncpsd *
                                   sizeof(FPS));
                    cout << " | Tasync = " << frame_<FPS>()->tasync << endl;
                }
                if (isw == n_sub_sweeps - 1) {
                    energies.push_back(get<0>(r));
                    normsqs.push_back(get<1>(r));
                    discarded_weights.push_back(get<2>(r));
                }
            }
            if (normalize_mps)
                normalize();
        }
        this->forward = forward;
        return energies.back();
    }
};

} // namespace block2
