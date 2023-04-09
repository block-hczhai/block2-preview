
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
#include "../core/spin_permutation.hpp"
#include "effective_functions.hpp"
#include "moving_environment.hpp"
#include "parallel_mps.hpp"
#include "qc_ncorr.hpp"
#include "qc_pdm1.hpp"
#include "qc_pdm2.hpp"
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

//  hrl: Checks whether there is any file
//  named "BLOCK_STOP_CALCULATION" in the work dir
//  if that file contains "STOP", the sweep will be
//  aborted gracefully
inline bool has_abort_file() {
    const string filename = "BLOCK_STOP_CALCULATION";
    bool stop = false;
    if (Parsing::file_exists(filename)) {
        ifstream ifs(filename.c_str());
        if (ifs.good()) {
            vector<string> lines = Parsing::readlines(&ifs);
            if (lines.size() >= 1 && lines[0] == "STOP") {
                cout << "ATTENTION: Found abort file! Aborting sweep." << endl;
                stop = true;
            }
        }
        ifs.close();
    }
    return stop;
}

// Density Matrix Renormalization Group
template <typename S, typename FL, typename FLS> struct DMRG {
    typedef typename MovingEnvironment<S, FL, FLS>::FP FP;
    typedef typename MovingEnvironment<S, FL, FLS>::FPS FPS;
    typedef typename MovingEnvironment<S, FL, FLS>::FLLS FLLS;
    typedef typename MovingEnvironment<S, FL, FLS>::FPLS FPLS;
    typedef typename GMatrix<FL>::FC FC;
    shared_ptr<MovingEnvironment<S, FL, FLS>> me;
    vector<shared_ptr<MovingEnvironment<S, FL, FLS>>> ext_mes;
    shared_ptr<MovingEnvironment<S, FC, FLS>> cpx_me;
    vector<shared_ptr<MPS<S, FLS>>> ext_mpss;
    vector<ubond_t> bond_dims;
    vector<vector<ubond_t>> site_dependent_bond_dims;
    vector<FPS> noises;
    vector<vector<FPLS>> energies;
    vector<FPS> discarded_weights;
    vector<vector<vector<pair<S, FPS>>>> mps_quanta;
    vector<vector<FPLS>> sweep_energies;
    vector<double> sweep_time;
    vector<FPS> sweep_discarded_weights;
    vector<vector<vector<pair<S, FPS>>>> sweep_quanta;
    vector<FPS> davidson_conv_thrds;
    int isweep = 0;
    int davidson_max_iter = 5000;
    int davidson_soft_max_iter = -1;
    FPS davidson_shift = 0.0;
    DavidsonTypes davidson_type = DavidsonTypes::Normal;
    int conn_adjust_step = 2;
    bool forward;
    uint8_t iprint = 2;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    FPS cutoff = 1E-14;
    FPS quanta_cutoff = 1E-3;
    bool decomp_last_site = true;
    bool state_specific = false;
    vector<FPS> projection_weights;
    size_t sweep_cumulative_nflop = 0;
    size_t sweep_max_pket_size = 0;
    size_t sweep_max_eff_ham_size = 0;
    double tprt = 0, teig = 0, teff = 0, tmve = 0, tblk = 0, tdm = 0, tsplt = 0,
           tsvd = 0, torth = 0;
    bool print_connection_time = false;
    // store all wfn singular values (for analysis) at each site
    bool store_wfn_spectra = false;
    vector<vector<FPS>> sweep_wfn_spectra;
    vector<FPS> wfn_spectra;
    int sweep_start_site = 0;
    int sweep_end_site = -1;
    Timer _t, _t2;
    DMRG(const shared_ptr<MovingEnvironment<S, FL, FLS>> &me,
         const vector<ubond_t> &bond_dims, const vector<FPS> &noises)
        : me(me), bond_dims(bond_dims), noises(noises), forward(false) {
        sweep_end_site = me->n_sites;
    }
    virtual ~DMRG() = default;
    struct Iteration {
        vector<FPLS> energies;
        vector<vector<pair<S, FPS>>> quanta;
        FPS error;
        double tdav;
        int ndav, mmps;
        size_t nflop;
        Iteration(const vector<FPLS> &energies, FPS error, int mmps, int ndav,
                  size_t nflop = 0, double tdav = 1.0)
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
               << setprecision(2)
               << (r.tdav == 0.0 ? 0 : (double)r.nflop / r.tdav)
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
    Iteration update_one_dot(int i, bool forward, ubond_t bond_dim, FPS noise,
                             FPS davidson_conv_thrd) {
        frame_<FPS>()->activate(0);
        bool fuse_left = i <= me->fuse_center;
        vector<shared_ptr<MPS<S, FLS>>> mpss = {me->ket};
        // non-hermitian hamiltonian
        if (me->bra != me->ket)
            mpss.push_back(me->bra);
        // state specific
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
        int mmps = 0;
        FPS error = 0.0;
        tuple<FPS, int, size_t, double> pdi;
        shared_ptr<SparseMatrixGroup<S, FLS>> pket = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> pdm = nullptr;
        bool skip_decomp =
            !decomp_last_site &&
            ((forward && i == sweep_end_site - 1 && !fuse_left) ||
             (!forward && i == sweep_start_site && fuse_left));
        bool build_pdm = noise != 0 && (noise_type & NoiseTypes::Collected);
        // effective hamiltonian
        if (davidson_soft_max_iter != 0 || noise != 0)
            pdi = one_dot_eigs_and_perturb(forward, fuse_left, i,
                                           davidson_conv_thrd, noise, pket);
        else if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (pket != nullptr)
            sweep_max_pket_size = max(sweep_max_pket_size, pket->total_memory);
        if ((build_pdm || me->para_rule == nullptr ||
             me->para_rule->is_root()) &&
            !skip_decomp) {
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
                prev_wfn->info->deallocate();
                prev_wfn->deallocate();
                if (me->bra != me->ket) {
                    prev_wfn = me->bra->tensors[i];
                    if (!fuse_left && forward)
                        me->bra->tensors[i] = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_left(i, me->bra->info, prev_wfn,
                                                   me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        me->bra->tensors[i] = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_right(i, me->bra->info, prev_wfn,
                                                    me->mpo->tf->opf->cg);
                    prev_wfn->info->deallocate();
                    prev_wfn->deallocate();
                }
                if (pket != nullptr) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_pkets = {
                        pket};
                    if (!fuse_left && forward)
                        pket = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_left(
                                i, me->ket->info, prev_pkets,
                                me->mpo->tf->opf->cg)[0];
                    else if (fuse_left && !forward)
                        pket = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_right(
                                i, me->ket->info, prev_pkets,
                                me->mpo->tf->opf->cg)[0];
                    prev_pkets[0]->deallocate_infos();
                    prev_pkets[0]->deallocate();
                }
            }
        }
        // state specific
        for (auto &mps : ext_mpss) {
            if (mps->info->bond_dim < bond_dim)
                mps->info->bond_dim = bond_dim;
            if ((me->para_rule == nullptr || me->para_rule->is_root()) &&
                !skip_decomp) {
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
            if (skip_decomp)
                mps->canonical_form[i] = forward ? 'S' : 'K';
            else {
                mps->canonical_form[i] = forward ? 'K' : 'S';
                if (forward && i != sweep_end_site - 1)
                    mps->move_right(me->mpo->tf->opf->cg, me->para_rule);
                else if (!forward && i != sweep_start_site)
                    mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
            }
        }
        if (build_pdm && !skip_decomp) {
            _t.get_time();
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            pdm = MovingEnvironment<S, FL, FLS>::density_matrix(
                me->ket->info->vacuum, me->ket->tensors[i], forward,
                me->para_rule != nullptr ? noise / me->para_rule->comm->size
                                         : noise,
                noise_type, 0.0, pket);
            if (me->para_rule != nullptr)
                me->para_rule->comm->reduce_sum(pdm, me->para_rule->comm->root);
            tdm += _t.get_time();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (skip_decomp) {
                me->ket->save_tensor(i);
                me->ket->unload_tensor(i);
                me->ket->canonical_form[i] = forward ? 'S' : 'K';
                if (me->bra != me->ket) {
                    me->bra->save_tensor(i);
                    me->bra->unload_tensor(i);
                    me->bra->canonical_form[i] = forward ? 'S' : 'K';
                }
            } else {
                // splitting of wavefunction
                shared_ptr<SparseMatrix<S, FLS>> old_ket = me->ket->tensors[i],
                                                 old_bra = nullptr;
                if (me->bra != me->ket)
                    old_bra = me->bra->tensors[i];
                shared_ptr<SparseMatrix<S, FLS>> dm, dm_b, left_k, right_k,
                    left_b = nullptr, right_b = nullptr;
                if (decomp_type == DecompositionTypes::DensityMatrix) {
                    _t.get_time();
                    const FPS factor = me->bra != me->ket ? (FPS)0.5 : (FPS)1.0;
                    dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                        me->ket->info->vacuum, me->ket->tensors[i], forward,
                        build_pdm ? 0.0 : noise, noise_type, factor, pket);
                    if (me->bra != me->ket)
                        MovingEnvironment<S, FL, FLS>::density_matrix_add_wfn(
                            dm, me->bra->tensors[i], forward, 0.5);
                    if (build_pdm)
                        GMatrixFunctions<FLS>::iadd(
                            GMatrix<FLS>(dm->data, (MKL_INT)dm->total_memory,
                                         1),
                            GMatrix<FLS>(pdm->data, (MKL_INT)pdm->total_memory,
                                         1),
                            1.0);
                    tdm += _t.get_time();
                    if (me->bra != me->ket)
                        dm_b =
                            dm->deep_copy(make_shared<VectorAllocator<FPS>>());
                    error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                        dm, me->ket->tensors[i], (int)bond_dim, forward, true,
                        left_k, right_k, cutoff, store_wfn_spectra, wfn_spectra,
                        trunc_type);
                    // TODO: this may have some problem if small numerical
                    // instability happens when truncating same dm twice
                    if (me->bra != me->ket) {
                        error +=
                            MovingEnvironment<S, FL, FLS>::split_density_matrix(
                                dm_b, me->bra->tensors[i], (int)bond_dim,
                                forward, true, left_b, right_b, cutoff, false,
                                wfn_spectra, trunc_type);
                        error *= (FPS)0.5;
                        dm_b->deallocate();
                    }
                    tsplt += _t.get_time();
                } else if (decomp_type == DecompositionTypes::SVD ||
                           decomp_type == DecompositionTypes::PureSVD) {
                    assert(noise_type == NoiseTypes::None ||
                           (noise_type & NoiseTypes::Perturbative) ||
                           (noise_type & NoiseTypes::Wavefunction));
                    if (noise != 0) {
                        if (noise_type & NoiseTypes::Wavefunction)
                            MovingEnvironment<S, FL, FLS>::
                                wavefunction_add_noise(me->ket->tensors[i],
                                                       noise);
                        else if (noise_type & NoiseTypes::Perturbative)
                            MovingEnvironment<S, FL, FLS>::
                                scale_perturbative_noise(noise, noise_type,
                                                         pket);
                    }
                    _t.get_time();
                    if (me->bra == me->ket)
                        error = MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                me->ket->info->vacuum, me->ket->tensors[i],
                                (int)bond_dim, forward, true, left_k, right_k,
                                cutoff, store_wfn_spectra, wfn_spectra,
                                trunc_type, decomp_type, pket);
                    else {
                        vector<FPS> weights = {(FPS)0.5, (FPS)0.5};
                        vector<shared_ptr<SparseMatrix<S, FLS>>> xwfns = {
                            me->bra->tensors[i]};
                        error = MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                me->ket->info->vacuum, me->ket->tensors[i],
                                (int)bond_dim, forward, true, left_k, right_k,
                                cutoff, store_wfn_spectra, wfn_spectra,
                                trunc_type, decomp_type, pket, xwfns, weights);
                        xwfns[0] = me->ket->tensors[i];
                        error += MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                me->bra->info->vacuum, me->bra->tensors[i],
                                (int)bond_dim, forward, true, left_b, right_b,
                                cutoff, false, wfn_spectra, trunc_type,
                                decomp_type, pket, xwfns, weights);
                        error *= (FPS)0.5;
                    }
                    tsvd += _t.get_time();
                } else
                    assert(false);
                shared_ptr<StateInfo<S>> info = nullptr;
                // propagation
                mpss = {me->ket};
                if (me->bra != me->ket)
                    mpss.insert(mpss.begin(), me->bra);
                for (auto &mps : mpss) {
                    shared_ptr<SparseMatrix<S, FLS>> left, right;
                    if (mps == me->ket)
                        left = left_k, right = right_k;
                    else
                        left = left_b, right = right_b;
                    if (forward) {
                        mps->tensors[i] = left;
                        mps->save_tensor(i);
                        info = left->info->extract_state_info(forward);
                        mmps = (int)info->n_states_total;
                        mps->info->bond_dim =
                            max(mps->info->bond_dim, (ubond_t)mmps);
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        info->deallocate();
                        if (i != sweep_end_site - 1) {
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
                        mmps = (int)info->n_states_total;
                        mps->info->bond_dim =
                            max(mps->info->bond_dim, (ubond_t)mmps);
                        mps->info->right_dims[i] = info;
                        mps->info->save_right_dims(i);
                        info->deallocate();
                        if (i > sweep_start_site) {
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
                }
                if (right_b != nullptr) {
                    right_b->info->deallocate();
                    right_b->deallocate();
                    left_b->info->deallocate();
                    left_b->deallocate();
                }
                right_k->info->deallocate();
                right_k->deallocate();
                left_k->info->deallocate();
                left_k->deallocate();
                if (dm != nullptr) {
                    dm->info->deallocate();
                    dm->deallocate();
                }
                if (pdm != nullptr) {
                    pdm->info->deallocate();
                    pdm->deallocate();
                }
                if (old_bra != nullptr) {
                    old_bra->info->deallocate();
                    old_bra->deallocate();
                }
                old_ket->info->deallocate();
                old_ket->deallocate();
            }
            me->ket->save_data();
            if (me->bra != me->ket)
                me->bra->save_data();
        } else {
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            mpss = {me->ket};
            if (me->bra != me->ket)
                mpss.insert(mpss.begin(), me->bra);
            for (auto &mps : mpss) {
                mps->unload_tensor(i);
                if (skip_decomp)
                    mps->canonical_form[i] = forward ? 'S' : 'K';
                else {
                    if (forward) {
                        if (i != sweep_end_site - 1) {
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else
                            mps->canonical_form[i] = 'K';
                    } else {
                        if (i > sweep_start_site) {
                            mps->canonical_form[i - 1] = 'K';
                            mps->canonical_form[i] = 'R';
                        } else
                            mps->canonical_form[i] = 'S';
                    }
                }
            }
        }
        if (pket != nullptr) {
            pket->deallocate();
            pket->deallocate_infos();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        get<0>(pdi) = get<0>(pdi) + xreal<FLLS>(me->mpo->const_e);
        if (me->para_rule != nullptr)
            me->para_rule->comm->broadcast(&get<0>(pdi), 1,
                                           me->para_rule->comm->root);
        return Iteration(vector<FPLS>{get<0>(pdi)}, error, mmps, get<1>(pdi),
                         get<2>(pdi), get<3>(pdi));
    }
    virtual tuple<FPLS, int, size_t, double>
    one_dot_eigs_and_perturb(const bool forward, const bool fuse_left,
                             const int i, const FPS davidson_conv_thrd,
                             const FPS noise,
                             shared_ptr<SparseMatrixGroup<S, FLS>> &pket) {
        tuple<FPLS, int, size_t, double> pdi;
        vector<shared_ptr<SparseMatrix<S, FLS>>> ortho_bra;
        _t.get_time();
        if (state_specific || projection_weights.size() != 0) {
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
        torth += _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff = me->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
            me->bra->tensors[i], me->ket->tensors[i]);
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        teff += _t.get_time();
        pdi = h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                          davidson_soft_max_iter, davidson_type,
                          davidson_shift - xreal<FL>((FL)me->mpo->const_e),
                          me->para_rule, ortho_bra, projection_weights);
        teig += _t.get_time();
        if (state_specific || projection_weights.size() != 0)
            for (auto &wfn : ortho_bra)
                wfn->deallocate();
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pket = h_eff->perturbative_noise(
                forward, i, i, fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                me->ket->info, noise_type, me->para_rule);
        tprt += _t.get_time();
        h_eff->deallocate();
        return pdi;
    }
    // two-site single-state dmrg algorithm
    // canonical form for wavefunction: C = center
    Iteration update_two_dot(int i, bool forward, ubond_t bond_dim, FPS noise,
                             FPS davidson_conv_thrd) {
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MPS<S, FLS>>> mpss = {me->ket};
        // non-hermitian hamiltonian
        if (me->bra != me->ket)
            mpss.push_back(me->bra);
        // state specific
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
        shared_ptr<SparseMatrix<S, FLS>> old_ket = me->ket->tensors[i],
                                         old_bra = nullptr;
        if (me->bra != me->ket)
            old_bra = me->bra->tensors[i];
        int mmps = 0;
        FPS error = 0.0;
        tuple<FPS, int, size_t, double> pdi;
        shared_ptr<SparseMatrixGroup<S, FLS>> pket = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> pdm = nullptr;
        bool build_pdm = noise != 0 && (noise_type & NoiseTypes::Collected);
        // effective hamiltonian
        if (davidson_soft_max_iter != 0 || noise != 0)
            pdi = two_dot_eigs_and_perturb(forward, i, davidson_conv_thrd,
                                           noise, pket);
        else if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (pket != nullptr)
            sweep_max_pket_size = max(sweep_max_pket_size, pket->total_memory);
        // state specific
        vector<shared_ptr<MPS<S, FLS>>> rev_ext_mpss(ext_mpss.rbegin(),
                                                     ext_mpss.rend());
        for (auto &mps : rev_ext_mpss) {
            mps->save_tensor(i);
            mps->unload_tensor(i);
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
            if (mps->info->bond_dim < bond_dim)
                mps->info->bond_dim = bond_dim;
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
            if (me->para_rule == nullptr || me->para_rule->is_root())
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, sweep_start_site, sweep_end_site, mps, forward,
                    me->mpo->tf->opf->cg);
        }
        if (build_pdm) {
            _t.get_time();
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            pdm = MovingEnvironment<S, FL, FLS>::density_matrix(
                me->ket->info->vacuum, old_ket, forward,
                me->para_rule != nullptr ? noise / me->para_rule->comm->size
                                         : noise,
                noise_type, 0.0, pket);
            if (me->para_rule != nullptr)
                me->para_rule->comm->reduce_sum(pdm, me->para_rule->comm->root);
            tdm += _t.get_time();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // splitting of wavefunction
            shared_ptr<SparseMatrix<S, FLS>> dm, dm_b;
            if (decomp_type == DecompositionTypes::DensityMatrix) {
                _t.get_time();
                const FPS factor = me->bra != me->ket ? (FPS)0.5 : (FPS)1.0;
                dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                    me->ket->info->vacuum, old_ket, forward,
                    build_pdm ? 0.0 : noise, noise_type, factor, pket);
                if (me->bra != me->ket)
                    MovingEnvironment<S, FL, FLS>::density_matrix_add_wfn(
                        dm, old_bra, forward, 0.5);
                if (build_pdm)
                    GMatrixFunctions<FLS>::iadd(
                        GMatrix<FLS>(dm->data, (MKL_INT)dm->total_memory, 1),
                        GMatrix<FLS>(pdm->data, (MKL_INT)pdm->total_memory, 1),
                        1.0);
                tdm += _t.get_time();
                if (me->bra != me->ket)
                    dm_b = dm->deep_copy(make_shared<VectorAllocator<FPS>>());
                error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                    dm, old_ket, (int)bond_dim, forward, true,
                    me->ket->tensors[i], me->ket->tensors[i + 1], cutoff,
                    store_wfn_spectra, wfn_spectra, trunc_type);
                // TODO: this may have some problem if small numerical
                // instability happens when truncating same dm twice
                if (me->bra != me->ket) {
                    error +=
                        MovingEnvironment<S, FL, FLS>::split_density_matrix(
                            dm_b, old_bra, (int)bond_dim, forward, true,
                            me->bra->tensors[i], me->bra->tensors[i + 1],
                            cutoff, false, wfn_spectra, trunc_type);
                    error *= (FPS)0.5;
                    dm_b->deallocate();
                }
                tsplt += _t.get_time();
            } else if (decomp_type == DecompositionTypes::SVD ||
                       decomp_type == DecompositionTypes::PureSVD) {
                assert(noise_type == NoiseTypes::None ||
                       (noise_type & NoiseTypes::Perturbative) ||
                       (noise_type & NoiseTypes::Wavefunction));
                if (noise != 0) {
                    if (noise_type & NoiseTypes::Wavefunction)
                        MovingEnvironment<S, FL, FLS>::wavefunction_add_noise(
                            old_ket, noise);
                    else if (noise_type & NoiseTypes::Perturbative)
                        MovingEnvironment<S, FL, FLS>::scale_perturbative_noise(
                            noise, noise_type, pket);
                }
                _t.get_time();
                if (me->bra == me->ket)
                    error =
                        MovingEnvironment<S, FL, FLS>::split_wavefunction_svd(
                            me->ket->info->vacuum, old_ket, (int)bond_dim,
                            forward, true, me->ket->tensors[i],
                            me->ket->tensors[i + 1], cutoff, store_wfn_spectra,
                            wfn_spectra, trunc_type, decomp_type, pket);
                else {
                    vector<FPS> weights = {(FPS)0.5, (FPS)0.5};
                    vector<shared_ptr<SparseMatrix<S, FLS>>> xwfns = {old_bra};
                    error =
                        MovingEnvironment<S, FL, FLS>::split_wavefunction_svd(
                            me->ket->info->vacuum, old_ket, (int)bond_dim,
                            forward, true, me->ket->tensors[i],
                            me->ket->tensors[i + 1], cutoff, store_wfn_spectra,
                            wfn_spectra, trunc_type, decomp_type, pket, xwfns,
                            weights);
                    xwfns[0] = old_ket;
                    error +=
                        MovingEnvironment<S, FL, FLS>::split_wavefunction_svd(
                            me->bra->info->vacuum, old_bra, (int)bond_dim,
                            forward, true, me->bra->tensors[i],
                            me->bra->tensors[i + 1], cutoff, false, wfn_spectra,
                            trunc_type, decomp_type, pket, xwfns, weights);
                    error *= (FPS)0.5;
                }
                tsvd += _t.get_time();
            } else
                assert(false);
            shared_ptr<StateInfo<S>> info = nullptr;
            // propagation
            mpss = {me->ket};
            if (me->bra != me->ket)
                mpss.insert(mpss.begin(), me->bra);
            for (auto &mps : mpss) {
                if (forward) {
                    info = mps->tensors[i]->info->extract_state_info(forward);
                    mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)mmps);
                    mps->info->left_dims[i + 1] = info;
                    mps->info->save_left_dims(i + 1);
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'C';
                } else {
                    info =
                        mps->tensors[i + 1]->info->extract_state_info(forward);
                    mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)mmps);
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
            }
            if (dm != nullptr) {
                dm->info->deallocate();
                dm->deallocate();
            }
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            if (old_bra != nullptr) {
                old_bra->info->deallocate();
                old_bra->deallocate();
            }
            old_ket->info->deallocate();
            old_ket->deallocate();
            MovingEnvironment<S, FL, FLS>::propagate_wfn(
                i, sweep_start_site, sweep_end_site, me->ket, forward,
                me->mpo->tf->opf->cg);
            me->ket->save_data();
            if (me->bra != me->ket) {
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, sweep_start_site, sweep_end_site, me->bra, forward,
                    me->mpo->tf->opf->cg);
                me->bra->save_data();
            }
        } else {
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            if (old_bra != nullptr) {
                old_bra->info->deallocate();
                old_bra->deallocate();
            }
            old_ket->info->deallocate();
            old_ket->deallocate();
            mpss = {me->ket};
            if (me->bra != me->ket)
                mpss.insert(mpss.begin(), me->bra);
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
        if (pket != nullptr) {
            pket->deallocate();
            pket->deallocate_infos();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        get<0>(pdi) = get<0>(pdi) + xreal<FLLS>(me->mpo->const_e);
        if (me->para_rule != nullptr)
            me->para_rule->comm->broadcast(&get<0>(pdi), 1,
                                           me->para_rule->comm->root);
        return Iteration(vector<FPLS>{get<0>(pdi)}, error, mmps, get<1>(pdi),
                         get<2>(pdi), get<3>(pdi));
    }
    virtual tuple<FPLS, int, size_t, double>
    two_dot_eigs_and_perturb(const bool forward, const int i,
                             const FPS davidson_conv_thrd, const FPS noise,
                             shared_ptr<SparseMatrixGroup<S, FLS>> &pket) {
        tuple<FPLS, int, size_t, double> pdi;
        vector<shared_ptr<SparseMatrix<S, FLS>>> ortho_bra;
        _t.get_time();
        if (state_specific || projection_weights.size() != 0) {
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
        torth += _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, forward, true, me->bra->tensors[i],
                        me->ket->tensors[i]);
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        teff += _t.get_time();
        pdi = h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                          davidson_soft_max_iter, davidson_type,
                          davidson_shift - xreal<FL>((FL)me->mpo->const_e),
                          me->para_rule, ortho_bra, projection_weights);
        teig += _t.get_time();
        if (state_specific || projection_weights.size() != 0)
            for (auto &wfn : ortho_bra)
                wfn->deallocate();
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pket = h_eff->perturbative_noise(forward, i, i + 1,
                                             FuseTypes::FuseLR, me->ket->info,
                                             noise_type, me->para_rule);
        tprt += _t.get_time();
        h_eff->deallocate();
        return pdi;
    }
    // State-averaged one-site algorithm
    // canonical form for wavefunction: J = left-fused, T = right-fused
    Iteration update_multi_one_dot(int i, bool forward, ubond_t bond_dim,
                                   FPS noise, FPS davidson_conv_thrd) {
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        shared_ptr<MultiMPS<S, FLS>> mbra =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->bra);
        if (me->bra == me->ket)
            assert(mbra == mket);
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MultiMPS<S, FLS>>> mpss =
            mbra == mket ? vector<shared_ptr<MultiMPS<S, FLS>>>{mket}
                         : vector<shared_ptr<MultiMPS<S, FLS>>>{mket, mbra};
        bool fuse_left = i <= me->fuse_center;
        for (auto &mps : mpss) {
            if (mps->canonical_form[i] == 'M') {
                if (i == 0)
                    mps->canonical_form[i] = 'J';
                else if (i == me->n_sites - 1)
                    mps->canonical_form[i] = 'T';
                else if (forward)
                    mps->canonical_form[i] = 'J';
                else
                    mps->canonical_form[i] = 'T';
            }
            // guess wavefunction
            // change to fused form for super-block hamiltonian
            // note that this switch exactly matches two-site conventional mpo
            // middle-site switch, so two-site conventional mpo can work
            mps->load_tensor(i);
            if ((fuse_left && mps->canonical_form[i] == 'T') ||
                (!fuse_left && mps->canonical_form[i] == 'J')) {
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_wfns =
                    mps->wfns;
                if (fuse_left && mps->canonical_form[i] == 'T')
                    mps->wfns = MovingEnvironment<S, FL, FLS>::
                        swap_multi_wfn_to_fused_left(i, mps->info, prev_wfns,
                                                     me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'J')
                    mps->wfns = MovingEnvironment<S, FL, FLS>::
                        swap_multi_wfn_to_fused_right(i, mps->info, prev_wfns,
                                                      me->mpo->tf->opf->cg);
                for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                    prev_wfns[j]->deallocate();
                if (prev_wfns.size() != 0)
                    prev_wfns[0]->deallocate_infos();
            }
        }
        // state specific
        for (auto &mps : ext_mpss) {
            // state-averaged mps projection is not yet supported
            assert(mps->canonical_form[i] != 'M');
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
        int mmps = 0;
        FPS error = 0.0;
        tuple<vector<FPLS>, int, size_t, double> pdi;
        shared_ptr<SparseMatrixGroup<S, FLS>> pket = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> pdm = nullptr;
        bool build_pdm = noise != 0 && (noise_type & NoiseTypes::Collected);
        vector<vector<pair<S, FPS>>> mps_quanta(mket->nroots);
        // effective hamiltonian
        if (davidson_soft_max_iter != 0 || noise != 0)
            pdi = multi_one_dot_eigs_and_perturb(forward, fuse_left, i,
                                                 davidson_conv_thrd, noise,
                                                 pket, mps_quanta);
        else if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (pket != nullptr)
            sweep_max_pket_size = max(sweep_max_pket_size, pket->total_memory);
        if (build_pdm || me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(decomp_type == DecompositionTypes::DensityMatrix);
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
                for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                    prev_wfns[j]->deallocate();
                if (prev_wfns.size() != 0)
                    prev_wfns[0]->deallocate_infos();
                if (me->bra != me->ket) {
                    prev_wfns = mbra->wfns;
                    if (!fuse_left && forward)
                        mbra->wfns = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_left(
                                i, mbra->info, prev_wfns, me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        mbra->wfns = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_right(
                                i, mbra->info, prev_wfns, me->mpo->tf->opf->cg);
                    for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                        prev_wfns[j]->deallocate();
                    if (prev_wfns.size() != 0)
                        prev_wfns[0]->deallocate_infos();
                }
                if (pket != nullptr) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_pkets = {
                        pket};
                    if (!fuse_left && forward)
                        pket = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_left(
                                i, mket->info, prev_pkets,
                                me->mpo->tf->opf->cg)[0];
                    else if (fuse_left && !forward)
                        pket = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_right(
                                i, mket->info, prev_pkets,
                                me->mpo->tf->opf->cg)[0];
                    prev_pkets[0]->deallocate_infos();
                    prev_pkets[0]->deallocate();
                }
            }
        }
        // state specific
        for (auto &mps : ext_mpss) {
            if (mps->info->bond_dim < bond_dim)
                mps->info->bond_dim = bond_dim;
            if ((me->para_rule == nullptr || me->para_rule->is_root())) {
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
            mps->canonical_form[i] = forward ? 'K' : 'S';
            if (forward && i != sweep_end_site - 1)
                mps->move_right(me->mpo->tf->opf->cg, me->para_rule);
            else if (!forward && i != sweep_start_site)
                mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
        }
        if (build_pdm) {
            _t.get_time();
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            pdm =
                MovingEnvironment<S, FL, FLS>::density_matrix_with_multi_target(
                    mket->info->vacuum, mket->wfns, mket->weights, forward,
                    me->para_rule != nullptr ? noise / me->para_rule->comm->size
                                             : noise,
                    noise_type, 0.0, pket);
            if (me->para_rule != nullptr)
                me->para_rule->comm->reduce_sum(pdm, me->para_rule->comm->root);
            tdm += _t.get_time();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // splitting of wavefunction
            vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_kets = mket->wfns,
                                                          new_kets, old_bras,
                                                          new_bras;
            if (mbra != mket)
                old_bras = mbra->wfns;
            shared_ptr<SparseMatrix<S, FLS>> dm, dm_b, rot_k, rot_b;
            _t.get_time();
            const FPS factor = mbra != mket ? (FPS)0.5 : (FPS)1.0;
            dm =
                MovingEnvironment<S, FL, FLS>::density_matrix_with_multi_target(
                    mket->info->vacuum, mket->wfns, mket->weights, forward,
                    build_pdm ? 0.0 : noise, noise_type, factor, pket);
            if (mbra != mket)
                MovingEnvironment<S, FL, FLS>::density_matrix_add_wfn_groups(
                    dm, mbra->wfns, mbra->weights, forward, 0.5);
            if (build_pdm)
                GMatrixFunctions<FLS>::iadd(
                    GMatrix<FLS>(dm->data, (MKL_INT)dm->total_memory, 1),
                    GMatrix<FLS>(pdm->data, (MKL_INT)pdm->total_memory, 1),
                    1.0);
            tdm += _t.get_time();
            if (mbra != mket)
                dm_b = dm->deep_copy(make_shared<VectorAllocator<FPS>>());
            error = MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                dm, mket->wfns, (int)bond_dim, forward, cpx_me == nullptr,
                new_kets, rot_k, cutoff, store_wfn_spectra, wfn_spectra,
                trunc_type);
            // TODO: this may have some problem if small numerical
            // instability happens when truncating same dm twice
            // rot_k and rot_b should be the same
            if (mbra != mket) {
                error +=
                    MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                        dm_b, mbra->wfns, (int)bond_dim, forward,
                        cpx_me == nullptr, new_bras, rot_b, cutoff, false,
                        wfn_spectra, trunc_type);
                error *= (FPS)0.5;
                dm_b->deallocate();
            }
            if (cpx_me != nullptr) {
                for (size_t k = 0; k < new_kets.size(); k += 2) {
                    FP nra = new_kets[k]->norm();
                    FP nrb = new_kets[k + 1]->norm();
                    FP nr = sqrt(abs(nra * nra + nrb * nrb));
                    new_kets[k]->iscale(1 / nr);
                    new_kets[k + 1]->iscale(1 / nr);
                }
                for (size_t k = 0; k < new_bras.size(); k += 2) {
                    FP nra = new_bras[k]->norm();
                    FP nrb = new_bras[k + 1]->norm();
                    FP nr = sqrt(abs(nra * nra + nrb * nrb));
                    new_bras[k]->iscale(1 / nr);
                    new_bras[k + 1]->iscale(1 / nr);
                }
            }
            tsplt += _t.get_time();
            shared_ptr<StateInfo<S>> info = nullptr;
            // propagation
            mpss = {mket};
            if (mbra != mket)
                mpss.insert(mpss.begin(), mbra);
            for (auto &mps : mpss) {
                shared_ptr<SparseMatrix<S, FLS>> rot =
                    mps == mket ? rot_k : rot_b;
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> new_wfns =
                    mps == mket ? new_kets : new_bras;
                if (forward) {
                    mps->tensors[i] = rot;
                    mps->save_tensor(i);
                    info = rot->info->extract_state_info(forward);
                    mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)mmps);
                    mps->info->left_dims[i + 1] = info;
                    mps->info->save_left_dims(i + 1);
                    info->deallocate();
                    if (i != sweep_end_site - 1) {
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i + 1, new_wfns, mps, forward);
                        mps->save_wavefunction(i + 1);
                        mps->unload_wavefunction(i + 1);
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'T';
                    } else {
                        mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i, new_wfns, mps, !forward);
                        mps->save_wavefunction(i);
                        mps->unload_wavefunction(i);
                        mps->canonical_form[i] = 'J';
                    }
                } else {
                    mps->tensors[i] = rot;
                    mps->save_tensor(i);
                    info = rot->info->extract_state_info(forward);
                    mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)mmps);
                    mps->info->right_dims[i] = info;
                    mps->info->save_right_dims(i);
                    info->deallocate();
                    if (i > sweep_start_site) {
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
                            i - 1, new_wfns, mps, forward);
                        mps->save_wavefunction(i - 1);
                        mps->unload_wavefunction(i - 1);
                        mps->canonical_form[i - 1] = 'J';
                        mps->canonical_form[i] = 'R';
                    } else {
                        mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        MovingEnvironment<S, FL, FLS>::contract_multi_one_dot(
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
            }
            dm->info->deallocate();
            dm->deallocate();
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            for (int j = (int)old_bras.size() - 1; j >= 0; j--)
                old_bras[j]->deallocate();
            if (old_bras.size() != 0)
                old_bras[0]->deallocate_infos();
            for (int j = (int)old_kets.size() - 1; j >= 0; j--)
                old_kets[j]->deallocate();
            if (old_kets.size() != 0)
                old_kets[0]->deallocate_infos();
            mket->save_data();
            if (mbra != mket)
                mbra->save_data();
        } else {
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            mpss = {mket};
            if (mbra != mket)
                mpss.insert(mpss.begin(), mbra);
            for (auto &mps : mpss) {
                mps->unload_tensor(i);
                if (forward) {
                    if (i != sweep_end_site - 1) {
                        mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        mps->tensors[i + 1] = nullptr;
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'T';
                    } else
                        mps->canonical_form[i] = 'J';
                } else {
                    if (i > sweep_start_site) {
                        mps->tensors[i - 1] = nullptr;
                        mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        mps->canonical_form[i - 1] = 'J';
                        mps->canonical_form[i] = 'R';
                    } else
                        mps->canonical_form[i] = 'T';
                }
            }
        }
        if (pket != nullptr) {
            pket->deallocate();
            pket->deallocate_infos();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        for (auto &x : get<0>(pdi))
            x += xreal<FLLS>(me->mpo->const_e);
        if (me->para_rule != nullptr)
            me->para_rule->comm->broadcast(get<0>(pdi).data(),
                                           get<0>(pdi).size(),
                                           me->para_rule->comm->root);
        Iteration r = Iteration(get<0>(pdi), error, mmps, get<1>(pdi),
                                get<2>(pdi), get<3>(pdi));
        r.quanta = mps_quanta;
        return r;
    }
    virtual tuple<vector<FPLS>, int, size_t, double>
    multi_one_dot_eigs_and_perturb(const bool forward, const bool fuse_left,
                                   const int i, const FPS davidson_conv_thrd,
                                   const FPS noise,
                                   shared_ptr<SparseMatrixGroup<S, FLS>> &pket,
                                   vector<vector<pair<S, FPS>>> &mps_quanta) {
        tuple<vector<FPLS>, int, size_t, double> pdi;
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        vector<shared_ptr<SparseMatrix<S, FLS>>> ortho_bra;
        _t.get_time();
        if (state_specific || projection_weights.size() != 0) {
            shared_ptr<VectorAllocator<FPS>> d_alloc =
                make_shared<VectorAllocator<FPS>>();
            ortho_bra.resize(ext_mpss.size());
            assert(ext_mes.size() == ext_mpss.size());
            shared_ptr<MultiMPS<S, FLS>> mbra =
                dynamic_pointer_cast<MultiMPS<S, FLS>>(me->bra);
            assert(mbra->wfns[0]->infos.size() == 1);
            for (size_t ist = 0; ist < ext_mes.size(); ist++) {
                ortho_bra[ist] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
                ortho_bra[ist]->allocate(mbra->wfns[0]->infos[0]);
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
        torth += _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> h_eff =
            nullptr;
        shared_ptr<EffectiveHamiltonian<S, FC, MultiMPS<S, FC>>> x_eff =
            nullptr;
        // complex correction me
        if (cpx_me != nullptr) {
            x_eff = cpx_me->multi_eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true);
            // create h_eff last to allow delayed contraction
            h_eff = me->multi_eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true);
            sweep_max_eff_ham_size =
                max(sweep_max_eff_ham_size, h_eff->op->get_total_memory() +
                                                x_eff->op->get_total_memory());
        } else {
            h_eff = me->multi_eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true);
            sweep_max_eff_ham_size =
                max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        }
        teff += _t.get_time();
        if (x_eff != nullptr)
            pdi = EffectiveFunctions<S, FL>::eigs_mixed(
                h_eff, x_eff, iprint >= 3, davidson_conv_thrd,
                davidson_max_iter, davidson_soft_max_iter, davidson_type,
                davidson_shift - xreal<FL>((FL)me->mpo->const_e),
                me->para_rule);
        else
            pdi =
                h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                            davidson_soft_max_iter, davidson_type,
                            davidson_shift - xreal<FL>((FL)me->mpo->const_e),
                            me->para_rule, ortho_bra, projection_weights);
        for (int i = 0; i < mket->nroots; i++) {
            mps_quanta[i] = h_eff->ket[i]->delta_quanta();
            mps_quanta[i].erase(
                remove_if(mps_quanta[i].begin(), mps_quanta[i].end(),
                          [this](const pair<S, FPS> &p) {
                              return p.second < this->quanta_cutoff;
                          }),
                mps_quanta[i].end());
        }
        if (x_eff != nullptr)
            for (int i = 0; i < mket->nroots / 2; i++)
                mps_quanta[i] = SparseMatrixGroup<S, FLS>::merge_delta_quanta(
                    mps_quanta[i + i], mps_quanta[i + i + 1]);
        teig += _t.get_time();
        if (state_specific || projection_weights.size() != 0)
            for (auto &wfn : ortho_bra)
                wfn->deallocate();
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pket = h_eff->perturbative_noise(
                forward, i, i, fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                mket->info, mket->weights, noise_type, me->para_rule);
        tprt += _t.get_time();
        h_eff->deallocate();
        if (x_eff != nullptr)
            x_eff->deallocate();
        return pdi;
    }
    // State-averaged two-site algorithm
    // canonical form for wavefunction: M = multi center
    Iteration update_multi_two_dot(int i, bool forward, ubond_t bond_dim,
                                   FPS noise, FPS davidson_conv_thrd) {
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        shared_ptr<MultiMPS<S, FLS>> mbra =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->bra);
        if (me->bra == me->ket)
            assert(mbra == mket);
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MultiMPS<S, FLS>>> mpss =
            mbra == mket ? vector<shared_ptr<MultiMPS<S, FLS>>>{mket}
                         : vector<shared_ptr<MultiMPS<S, FLS>>>{mket, mbra};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr || mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_multi_two_dot(i, mps);
            else
                mps->load_tensor(i);
            mps->tensors[i] = mps->tensors[i + 1] = nullptr;
        }
        // state specific
        for (auto &mps : ext_mpss) {
            assert(!(mps->get_type() & MPSTypes::MultiWfn));
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_two_dot(i, mps);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_kets = mket->wfns,
                                                      old_bras;
        if (mbra != mket)
            old_bras = mbra->wfns;
        // effective hamiltonian
        int mmps = 0;
        FPS error = 0.0;
        tuple<vector<FPLS>, int, size_t, double> pdi;
        shared_ptr<SparseMatrixGroup<S, FLS>> pket = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> pdm = nullptr;
        bool build_pdm = noise != 0 && (noise_type & NoiseTypes::Collected);
        vector<vector<pair<S, FPS>>> mps_quanta(mket->nroots);
        if (davidson_soft_max_iter != 0 || noise != 0)
            pdi = multi_two_dot_eigs_and_perturb(forward, i, davidson_conv_thrd,
                                                 noise, pket, mps_quanta);
        else if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (pket != nullptr)
            sweep_max_pket_size = max(sweep_max_pket_size, pket->total_memory);
        // state specific
        vector<shared_ptr<MPS<S, FLS>>> rev_ext_mpss(ext_mpss.rbegin(),
                                                     ext_mpss.rend());
        for (auto &mps : rev_ext_mpss) {
            mps->save_tensor(i);
            mps->unload_tensor(i);
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
            if (mps->info->bond_dim < bond_dim)
                mps->info->bond_dim = bond_dim;
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
            if (me->para_rule == nullptr || me->para_rule->is_root())
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, sweep_start_site, sweep_end_site, mps, forward,
                    me->mpo->tf->opf->cg);
        }
        if (build_pdm) {
            _t.get_time();
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            pdm =
                MovingEnvironment<S, FL, FLS>::density_matrix_with_multi_target(
                    mket->info->vacuum, old_kets, mket->weights, forward,
                    me->para_rule != nullptr ? noise / me->para_rule->comm->size
                                             : noise,
                    noise_type, 0.0, pket);
            if (me->para_rule != nullptr)
                me->para_rule->comm->reduce_sum(pdm, me->para_rule->comm->root);
            tdm += _t.get_time();
        }
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            shared_ptr<SparseMatrix<S, FLS>> dm, dm_b;
            _t.get_time();
            const FPS factor = mbra != mket ? (FPS)0.5 : (FPS)1.0;
            dm =
                MovingEnvironment<S, FL, FLS>::density_matrix_with_multi_target(
                    mket->info->vacuum, old_kets, mket->weights, forward,
                    build_pdm ? 0.0 : noise, noise_type, factor, pket);
            if (mbra != mket)
                MovingEnvironment<S, FL, FLS>::density_matrix_add_wfn_groups(
                    dm, old_bras, mbra->weights, forward, 0.5);
            if (build_pdm)
                GMatrixFunctions<FLS>::iadd(
                    GMatrix<FLS>(dm->data, (MKL_INT)dm->total_memory, 1),
                    GMatrix<FLS>(pdm->data, (MKL_INT)pdm->total_memory, 1),
                    1.0);
            tdm += _t.get_time();
            if (mbra != mket)
                dm_b = dm->deep_copy(make_shared<VectorAllocator<FPS>>());
            error = MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                dm, old_kets, (int)bond_dim, forward, cpx_me == nullptr,
                mket->wfns, forward ? mket->tensors[i] : mket->tensors[i + 1],
                cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
            if (mbra != mket) {
                error +=
                    MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                        dm_b, old_bras, (int)bond_dim, forward,
                        cpx_me == nullptr, mbra->wfns,
                        forward ? mbra->tensors[i] : mbra->tensors[i + 1],
                        cutoff, false, wfn_spectra, trunc_type);
                error *= (FPS)0.5;
                dm_b->deallocate();
            }
            if (cpx_me != nullptr) {
                for (size_t k = 0; k < mket->wfns.size(); k += 2) {
                    FP nra = mket->wfns[k]->norm();
                    FP nrb = mket->wfns[k + 1]->norm();
                    FP nr = sqrt(abs(nra * nra + nrb * nrb));
                    mket->wfns[k]->iscale(1 / nr);
                    mket->wfns[k + 1]->iscale(1 / nr);
                }
                for (size_t k = 0; k < mbra->wfns.size(); k += 2) {
                    FP nra = mbra->wfns[k]->norm();
                    FP nrb = mbra->wfns[k + 1]->norm();
                    FP nr = sqrt(abs(nra * nra + nrb * nrb));
                    mbra->wfns[k]->iscale(1 / nr);
                    mbra->wfns[k + 1]->iscale(1 / nr);
                }
            }
            tsplt += _t.get_time();
            shared_ptr<StateInfo<S>> info = nullptr;
            mpss = {mket};
            if (mbra != mket)
                mpss.insert(mpss.begin(), mbra);
            for (auto &mps : mpss) {
                if (forward) {
                    info = mps->tensors[i]->info->extract_state_info(forward);
                    mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)mmps);
                    mps->info->left_dims[i + 1] = info;
                    mps->info->save_left_dims(i + 1);
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'M';
                } else {
                    info =
                        mps->tensors[i + 1]->info->extract_state_info(forward);
                    mmps = (int)info->n_states_total;
                    mps->info->bond_dim =
                        max(mps->info->bond_dim, (ubond_t)mmps);
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
            }
            dm->info->deallocate();
            dm->deallocate();
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            for (int k = (int)old_bras.size() - 1; k >= 0; k--)
                old_bras[k]->deallocate();
            if (old_bras.size() != 0)
                old_bras[0]->deallocate_infos();
            for (int k = (int)old_kets.size() - 1; k >= 0; k--)
                old_kets[k]->deallocate();
            if (old_kets.size() != 0)
                old_kets[0]->deallocate_infos();
            MovingEnvironment<S, FL, FLS>::propagate_multi_wfn(
                i, sweep_start_site, sweep_end_site, mket, forward,
                me->mpo->tf->opf->cg);
            mket->save_data();
            if (mbra != mket) {
                MovingEnvironment<S, FL, FLS>::propagate_multi_wfn(
                    i, sweep_start_site, sweep_end_site, mbra, forward,
                    me->mpo->tf->opf->cg);
                mbra->save_data();
            }
        } else {
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            for (int k = (int)old_bras.size() - 1; k >= 0; k--)
                old_bras[k]->deallocate();
            if (old_bras.size() != 0)
                old_bras[0]->deallocate_infos();
            for (int k = (int)old_kets.size() - 1; k >= 0; k--)
                old_kets[k]->deallocate();
            if (old_kets.size() != 0)
                old_kets[0]->deallocate_infos();
            mpss = {mket};
            if (mbra != mket)
                mpss.insert(mpss.begin(), mbra);
            for (auto &mps : mpss) {
                if (forward) {
                    mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                    mps->tensors[i + 1] = nullptr;
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'M';
                } else {
                    mps->tensors[i] = nullptr;
                    mps->tensors[i + 1] = make_shared<SparseMatrix<S, FLS>>();
                    mps->canonical_form[i] = 'M';
                    mps->canonical_form[i + 1] = 'R';
                }
            }
        }
        if (pket != nullptr) {
            pket->deallocate();
            pket->deallocate_infos();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        for (auto &x : get<0>(pdi))
            x += xreal<FLLS>(me->mpo->const_e);
        if (me->para_rule != nullptr)
            me->para_rule->comm->broadcast(get<0>(pdi).data(),
                                           get<0>(pdi).size(),
                                           me->para_rule->comm->root);
        Iteration r = Iteration(get<0>(pdi), error, mmps, get<1>(pdi),
                                get<2>(pdi), get<3>(pdi));
        r.quanta = mps_quanta;
        return r;
    }
    virtual tuple<vector<FPLS>, int, size_t, double>
    multi_two_dot_eigs_and_perturb(const bool forward, const int i,
                                   const FPS davidson_conv_thrd,
                                   const FPS noise,
                                   shared_ptr<SparseMatrixGroup<S, FLS>> &pket,
                                   vector<vector<pair<S, FPS>>> &mps_quanta) {
        tuple<vector<FPLS>, int, size_t, double> pdi;
        vector<shared_ptr<SparseMatrix<S, FLS>>> ortho_bra;
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        _t.get_time();
        if (state_specific || projection_weights.size() != 0) {
            shared_ptr<VectorAllocator<FPS>> d_alloc =
                make_shared<VectorAllocator<FPS>>();
            ortho_bra.resize(ext_mpss.size());
            assert(ext_mes.size() == ext_mpss.size());
            shared_ptr<MultiMPS<S, FLS>> mbra =
                dynamic_pointer_cast<MultiMPS<S, FLS>>(me->bra);
            assert(mbra->wfns[0]->infos.size() == 1);
            for (size_t ist = 0; ist < ext_mes.size(); ist++) {
                ortho_bra[ist] = make_shared<SparseMatrix<S, FLS>>(d_alloc);
                ortho_bra[ist]->allocate(mbra->wfns[0]->infos[0]);
                shared_ptr<EffectiveHamiltonian<S, FL>> i_eff =
                    ext_mes[ist]->eff_ham(FuseTypes::FuseLR, forward, false,
                                          ortho_bra[ist],
                                          ext_mpss[ist]->tensors[i]);
                auto ipdi = i_eff->multiply(ext_mes[ist]->mpo->const_e,
                                            ext_mes[ist]->para_rule);
                i_eff->deallocate();
            }
        }
        torth += _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> h_eff =
            nullptr;
        shared_ptr<EffectiveHamiltonian<S, FC, MultiMPS<S, FC>>> x_eff =
            nullptr;
        // complex correction me
        if (cpx_me != nullptr) {
            x_eff = cpx_me->multi_eff_ham(FuseTypes::FuseLR, forward, true);
            h_eff = me->multi_eff_ham(FuseTypes::FuseLR, forward, true);
            sweep_max_eff_ham_size =
                max(sweep_max_eff_ham_size, h_eff->op->get_total_memory() +
                                                x_eff->op->get_total_memory());
        } else {
            h_eff = me->multi_eff_ham(FuseTypes::FuseLR, forward, true);
            sweep_max_eff_ham_size =
                max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        }
        teff += _t.get_time();
        if (x_eff != nullptr)
            pdi = EffectiveFunctions<S, FL>::eigs_mixed(
                h_eff, x_eff, iprint >= 3, davidson_conv_thrd,
                davidson_max_iter, davidson_soft_max_iter, davidson_type,
                davidson_shift - xreal<FL>((FL)me->mpo->const_e),
                me->para_rule);
        else
            pdi =
                h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                            davidson_soft_max_iter, davidson_type,
                            davidson_shift - xreal<FL>((FL)me->mpo->const_e),
                            me->para_rule, ortho_bra, projection_weights);
        for (int i = 0; i < mket->nroots; i++) {
            mps_quanta[i] = h_eff->ket[i]->delta_quanta();
            mps_quanta[i].erase(
                remove_if(mps_quanta[i].begin(), mps_quanta[i].end(),
                          [this](const pair<S, FPS> &p) {
                              return p.second < this->quanta_cutoff;
                          }),
                mps_quanta[i].end());
        }
        if (x_eff != nullptr)
            for (int i = 0; i < mket->nroots / 2; i++)
                mps_quanta[i] = SparseMatrixGroup<S, FLS>::merge_delta_quanta(
                    mps_quanta[i + i], mps_quanta[i + i + 1]);
        teig += _t.get_time();
        if (state_specific || projection_weights.size() != 0)
            for (auto &wfn : ortho_bra)
                wfn->deallocate();
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pket = h_eff->perturbative_noise(
                forward, i, i + 1, FuseTypes::FuseLR, mket->info, mket->weights,
                noise_type, me->para_rule);
        tprt += _t.get_time();
        h_eff->deallocate();
        if (x_eff != nullptr)
            x_eff->deallocate();
        return pdi;
    }
    virtual Iteration blocking(int i, bool forward, ubond_t bond_dim, FPS noise,
                               FPS davidson_conv_thrd) {
        _t2.get_time();
        me->move_to(i);
        for (auto &xme : ext_mes)
            xme->move_to(i);
        if (cpx_me != nullptr)
            cpx_me->move_to(i);
        tmve += _t2.get_time();
        assert(me->dot == 1 || me->dot == 2);
        Iteration it(vector<FPLS>(), 0, 0, 0);
        // use site dependent bond dims
        if (site_dependent_bond_dims.size() > 0) {
            const int bond_update_idx =
                me->dot == 1 && !forward && i > 0 ? i - 1 : i;
            const int i_update_sweep =
                min(isweep, (int)site_dependent_bond_dims.size() - 1);
            if (site_dependent_bond_dims[i_update_sweep].size() != 0)
                bond_dim = site_dependent_bond_dims[i_update_sweep].at(
                    bond_update_idx);
        }
        if (me->dot == 2) {
            if (me->ket->canonical_form[i] == 'M' ||
                me->ket->canonical_form[i + 1] == 'M' ||
                me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T' ||
                me->ket->canonical_form[i + 1] == 'J' ||
                me->ket->canonical_form[i + 1] == 'T')
                it = update_multi_two_dot(i, forward, bond_dim, noise,
                                          davidson_conv_thrd);
            else
                it = update_two_dot(i, forward, bond_dim, noise,
                                    davidson_conv_thrd);
        } else {
            if (me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T' ||
                me->ket->canonical_form[i] == 'M')
                it = update_multi_one_dot(i, forward, bond_dim, noise,
                                          davidson_conv_thrd);
            else
                it = update_one_dot(i, forward, bond_dim, noise,
                                    davidson_conv_thrd);
        }
        if (store_wfn_spectra) {
            const int bond_update_idx = me->dot == 1 && !forward ? i - 1 : i;
            if (bond_update_idx >= 0) {
                if (sweep_wfn_spectra.size() <= bond_update_idx)
                    sweep_wfn_spectra.resize(bond_update_idx + 1);
                sweep_wfn_spectra[bond_update_idx] = wfn_spectra;
            }
        }
        tblk += _t2.get_time();
        return it;
    }
    // one standard DMRG sweep
    virtual tuple<vector<FPLS>, FPS, vector<vector<pair<S, FPS>>>>
    sweep(bool forward, ubond_t bond_dim, FPS noise, FPS davidson_conv_thrd) {
        teff = teig = tprt = tblk = tmve = tdm = tsplt = tsvd = torth = 0;
        me->mpo->tread = me->mpo->twrite = 0;
        frame_<FPS>()->twrite = frame_<FPS>()->tread = frame_<FPS>()->tasync =
            0;
        frame_<FPS>()->fpwrite = frame_<FPS>()->fpread = 0;
        if (frame_<FPS>()->fp_codec != nullptr)
            frame_<FPS>()->fp_codec->ndata = frame_<FPS>()->fp_codec->ncpsd = 0;
        if (me->para_rule != nullptr && iprint >= 2) {
            me->para_rule->comm->tcomm = 0;
            me->para_rule->comm->tidle = 0;
            me->para_rule->comm->twait = 0;
        }
        me->prepare(sweep_start_site, sweep_end_site);
        for (auto &xme : ext_mes)
            xme->prepare(sweep_start_site, sweep_end_site);
        if (cpx_me != nullptr)
            cpx_me->prepare(sweep_start_site, sweep_end_site);
        sweep_energies.clear();
        sweep_discarded_weights.clear();
        sweep_quanta.clear();
        sweep_cumulative_nflop = 0;
        sweep_max_pket_size = 0;
        sweep_max_eff_ham_size = 0;
        frame_<FPS>()->reset_peak_used_memory();
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < sweep_end_site - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= sweep_start_site; it--)
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
            sweep_cumulative_nflop += r.nflop;
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            sweep_energies.push_back(r.energies);
            sweep_discarded_weights.push_back(r.error);
            sweep_quanta.push_back(r.quanta);
            if (frame_<FPS>()->restart_dir_optimal_mps != "" ||
                frame_<FPS>()->restart_dir_optimal_mps_per_sweep != "") {
                size_t midx =
                    min_element(
                        sweep_energies.begin(), sweep_energies.end(),
                        [](const vector<FPLS> &x, const vector<FPLS> &y) {
                            return x.back() < y.back();
                        }) -
                    sweep_energies.begin();
                if (midx == sweep_energies.size() - 1) {
                    if (me->para_rule == nullptr || me->para_rule->is_root()) {
                        if (frame_<FPS>()->restart_dir_optimal_mps != "") {
                            string rdoe =
                                frame_<FPS>()->restart_dir_optimal_mps;
                            if (!Parsing::path_exists(rdoe))
                                Parsing::mkdir(rdoe);
                            me->ket->info->copy_mutable(rdoe);
                            me->ket->copy_data(rdoe);
                            if (me->bra != me->ket) {
                                me->ket->info->copy_mutable(rdoe);
                                me->ket->copy_data(rdoe);
                            }
                        }
                        if (frame_<FPS>()->restart_dir_optimal_mps_per_sweep !=
                            "") {
                            string rdps =
                                frame_<FPS>()
                                    ->restart_dir_optimal_mps_per_sweep +
                                "." + Parsing::to_string((int)energies.size());
                            if (!Parsing::path_exists(rdps))
                                Parsing::mkdir(rdps);
                            me->ket->info->copy_mutable(rdps);
                            me->ket->copy_data(rdps);
                            if (me->bra != me->ket) {
                                me->ket->info->copy_mutable(rdps);
                                me->ket->copy_data(rdps);
                            }
                        }
                    }
                    if (me->para_rule != nullptr)
                        me->para_rule->comm->barrier();
                }
            }
        }
        size_t idx =
            min_element(sweep_energies.begin(), sweep_energies.end(),
                        [](const vector<FPLS> &x, const vector<FPLS> &y) {
                            return x.back() < y.back();
                        }) -
            sweep_energies.begin();
        if (frame_<FPS>()->restart_dir != "" &&
            (me->para_rule == nullptr || me->para_rule->is_root())) {
            if (!Parsing::path_exists(frame_<FPS>()->restart_dir))
                Parsing::mkdir(frame_<FPS>()->restart_dir);
            me->ket->info->copy_mutable(frame_<FPS>()->restart_dir);
            me->ket->copy_data(frame_<FPS>()->restart_dir);
            if (me->bra != me->ket) {
                me->bra->info->copy_mutable(frame_<FPS>()->restart_dir);
                me->bra->copy_data(frame_<FPS>()->restart_dir);
            }
        }
        if (frame_<FPS>()->restart_dir_per_sweep != "" &&
            (me->para_rule == nullptr || me->para_rule->is_root())) {
            string rdps = frame_<FPS>()->restart_dir_per_sweep + "." +
                          Parsing::to_string((int)energies.size());
            if (!Parsing::path_exists(rdps))
                Parsing::mkdir(rdps);
            me->ket->info->copy_mutable(rdps);
            me->ket->copy_data(rdps);
            if (me->bra != me->ket) {
                me->bra->info->copy_mutable(rdps);
                me->bra->copy_data(rdps);
            }
        }
        FPS max_dw = *max_element(sweep_discarded_weights.begin(),
                                  sweep_discarded_weights.end());
        return make_tuple(sweep_energies[idx], max_dw, sweep_quanta[idx]);
    }
    // one DMRG sweep over a range of sites in multi-center MPS
    void partial_sweep(int ip, bool forward, bool connect, ubond_t bond_dim,
                       FPS noise, FPS davidson_conv_thrd) {
        assert(me->ket->get_type() == MPSTypes::MultiCenter);
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(me->ket);
        int a = ip == 0 ? 0 : para_mps->conn_centers[ip - 1];
        int b =
            ip == para_mps->ncenter ? me->n_sites : para_mps->conn_centers[ip];
        if (connect) {
            a = para_mps->conn_centers[ip] - 1;
            b = a + me->dot;
        } else
            forward ^= ip & 1;
        if (para_mps->canonical_form[a] == 'C' ||
            para_mps->canonical_form[a] == 'K')
            me->center = a;
        else if (para_mps->canonical_form[b - 1] == 'C' ||
                 para_mps->canonical_form[b - 1] == 'S')
            me->center = b - me->dot;
        else if (para_mps->canonical_form[b - 2] == 'C' ||
                 para_mps->canonical_form[b - 2] == 'K')
            me->center = b - me->dot;
        else
            assert(false);
        me->partial_prepare(a, b);
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < b - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= a; it--)
                sweep_range.push_back(it);
        Timer t;
        for (auto i : sweep_range) {
            stringstream sout;
            check_signal_()();
            sout << " " << (connect ? "CON" : "PAR") << setw(4) << ip;
            sout << " " << (forward ? "-->" : "<--");
            if (me->dot == 2)
                sout << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                     << " .. ";
            else
                sout << " Site = " << setw(4) << i << " .. ";
            t.get_time();
            Iteration r =
                blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
            sweep_cumulative_nflop += r.nflop;
            sweep_time[i] = t.get_time();
            sout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << sweep_time[i] << endl;
            if (iprint >= 2)
                cout << sout.rdbuf();
            sweep_energies[i] = r.energies;
            sweep_discarded_weights[i] = r.error;
            sweep_quanta[i] = r.quanta;
        }
        if (me->dot == 2 && !connect) {
            if (forward)
                me->left_contract_rotate_unordered(me->center + 1);
            else
                me->right_contract_rotate_unordered(me->center - 1);
        }
    }
    // update one connection site in multi-center MPS
    void connection_sweep(int ip, ubond_t bond_dim, FPS noise,
                          FPS davidson_conv_thrd, int new_conn_center) {
        assert(me->ket->get_type() == MPSTypes::MultiCenter);
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(me->ket);
        Timer t;
        double tflip = 0, tmerge = 0, tsplit = 0, trot = 0, tmove = 0,
               tsweep = 0;
        me->center = para_mps->conn_centers[ip] - 1;
        t.get_time();
        if (para_mps->canonical_form[me->center] == 'C' &&
            para_mps->canonical_form[me->center + 1] == 'C')
            para_mps->canonical_form[me->center] = 'K',
            para_mps->canonical_form[me->center + 1] = 'S';
        else if (para_mps->canonical_form[me->center] == 'S' &&
                 para_mps->canonical_form[me->center + 1] == 'K') {
            para_mps->flip_fused_form(me->center, me->mpo->tf->opf->cg,
                                      me->para_rule);
            para_mps->flip_fused_form(me->center + 1, me->mpo->tf->opf->cg,
                                      me->para_rule);
        }
        tflip += t.get_time();
        if (para_mps->canonical_form[me->center] == 'K' &&
            para_mps->canonical_form[me->center + 1] == 'S') {
            t.get_time();
            para_mps->para_merge(ip, me->para_rule);
            tmerge += t.get_time();
            partial_sweep(ip, true, true, bond_dim, noise,
                          davidson_conv_thrd); // LK
            tsweep += t.get_time();
            me->left_contract_rotate_unordered(me->center + 1);
            trot += t.get_time();
            para_mps->canonical_form[me->center + 1] = 'K';
            para_mps->center = me->center + 1;
            while (new_conn_center < para_mps->conn_centers[ip]) {
                para_mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
                me->right_contract_rotate_unordered(para_mps->center -
                                                    para_mps->dot + 1);
                para_mps->conn_centers[ip]--;
                me->center--;
            }
            tmove += t.get_time();
            para_mps->flip_fused_form(me->center + 1, me->mpo->tf->opf->cg,
                                      me->para_rule); // LS
            tflip += t.get_time();
            para_mps->center = me->center + 1;
            while (new_conn_center > para_mps->conn_centers[ip]) {
                para_mps->move_right(me->mpo->tf->opf->cg, me->para_rule);
                me->left_contract_rotate_unordered(para_mps->center);
                para_mps->conn_centers[ip]++;
                me->center++;
            }
            tmove += t.get_time();
            auto rmat = para_mps->para_split(ip, me->para_rule); // KR
            me->right_contract_rotate_unordered(me->center - 1);
            trot += t.get_time();
            // if root proc saves tensor too early,
            // right_contract_rotate in other proc will have problems
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                para_mps->tensors[me->center + 1] = rmat;
                para_mps->save_tensor(me->center + 1); // KS
            }
            if (me->para_rule != nullptr)
                me->para_rule->comm->barrier();
            t.get_time();
            para_mps->flip_fused_form(me->center, me->mpo->tf->opf->cg,
                                      me->para_rule);
            para_mps->flip_fused_form(me->center + 1, me->mpo->tf->opf->cg,
                                      me->para_rule); // SK
            tflip += t.get_time();
        }
        if (iprint >= 2 && print_connection_time) {
            stringstream sout;
            sout << "Time connection = [" << ip << "] " << fixed
                 << setprecision(3)
                 << tflip + tmerge + tsplit + trot + tmove + tsweep;
            sout << " | Tflip = " << tflip << " | Tmerge = " << tmerge
                 << " | tsplit = " << tsplit << " | Trot = " << trot
                 << " | Tmove = " << tmove << endl;
            cout << sout.rdbuf();
        }
    }
    // one unordered DMRG sweep (multi-center MPS required)
    tuple<vector<FPLS>, FPS, vector<vector<pair<S, FPS>>>>
    unordered_sweep(bool forward, ubond_t bond_dim, FPS noise,
                    FPS davidson_conv_thrd) {
        assert(me->ket->get_type() == MPSTypes::MultiCenter);
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            dynamic_pointer_cast<ParallelMPS<S, FLS>>(me->ket);
        teff = teig = tprt = tblk = tmve = tdm = tsplt = tsvd = torth = 0;
        me->mpo->tread = me->mpo->twrite = 0;
        frame_<FPS>()->twrite = frame_<FPS>()->tread = frame_<FPS>()->tasync =
            0;
        frame_<FPS>()->fpwrite = frame_<FPS>()->fpread = 0;
        if (frame_<FPS>()->fp_codec != nullptr)
            frame_<FPS>()->fp_codec->ndata = frame_<FPS>()->fp_codec->ncpsd = 0;
        if (para_mps->rule != nullptr && iprint >= 2) {
            para_mps->rule->comm->tcomm = 0;
            para_mps->rule->comm->tidle = 0;
            para_mps->rule->comm->twait = 0;
        }
        if (me->para_rule != nullptr && iprint >= 2) {
            me->para_rule->comm->tcomm = 0;
            me->para_rule->comm->tidle = 0;
            me->para_rule->comm->twait = 0;
        }
        sweep_energies.clear();
        sweep_time.clear();
        sweep_discarded_weights.clear();
        sweep_quanta.clear();
        sweep_cumulative_nflop = 0;
        sweep_max_pket_size = 0;
        sweep_max_eff_ham_size = 0;
        frame_<FPS>()->reset_peak_used_memory();
        sweep_energies.resize(me->n_sites - me->dot + 1, vector<FPLS>{1E9});
        sweep_time.resize(me->n_sites - me->dot + 1, 0);
        sweep_discarded_weights.resize(me->n_sites - me->dot + 1);
        sweep_quanta.resize(me->n_sites - me->dot + 1);
        para_mps->enable_parallel_writing();
        para_mps->set_ref_canonical_form();
        for (int ip = 0; ip < para_mps->ncenter; ip++)
            if (para_mps->rule == nullptr ||
                ip % para_mps->rule->comm->ngroup ==
                    para_mps->rule->comm->group)
                connection_sweep(ip, bond_dim, noise, davidson_conv_thrd,
                                 para_mps->conn_centers[ip]);
        para_mps->sync_canonical_form();
        for (int ip = 0; ip <= para_mps->ncenter; ip++)
            if (para_mps->rule == nullptr ||
                ip % para_mps->rule->comm->ngroup ==
                    para_mps->rule->comm->group)
                partial_sweep(ip, forward, false, bond_dim, noise,
                              davidson_conv_thrd);
        para_mps->sync_canonical_form();
        if (para_mps->rule != nullptr)
            para_mps->rule->comm->allreduce_max(sweep_time);
        vector<double> partition_time(para_mps->ncenter + 1);
        for (int ip = 0; ip <= para_mps->ncenter; ip++) {
            int pi = ip == 0 ? 0 : para_mps->conn_centers[ip - 1];
            int pj = ip == para_mps->ncenter ? me->n_sites
                                             : para_mps->conn_centers[ip];
            double tx = 0;
            for (int ipp = pi; ipp < pj - 1; ipp++)
                tx += sweep_time[ipp];
            partition_time[ip] = tx;
        }
        vector<int> new_conn_centers = para_mps->conn_centers;
        vector<int> old_conn_centers = para_mps->conn_centers;
        for (int ip = 0; ip < para_mps->ncenter; ip++) {
            me->center = para_mps->conn_centers[ip] - 1;
            if (para_mps->canonical_form[me->center] == 'L' ||
                para_mps->canonical_form[me->center] == 'R')
                continue;
            int cc = para_mps->conn_centers[ip];
            int lcc = (ip == 0 ? 0 : para_mps->conn_centers[ip - 1]) + 2;
            int hcc =
                (ip == para_mps->ncenter - 1 ? me->n_sites
                                             : para_mps->conn_centers[ip + 1]) -
                2;
            double tdiff = abs(partition_time[ip] - partition_time[ip + 1]);
            if (partition_time[ip + 1] > partition_time[ip])
                for (int i = 1; i <= conn_adjust_step; i++) {
                    if (cc + 1 <= hcc && 2 * sweep_time[cc] <= tdiff)
                        tdiff -= 2 * sweep_time[cc], cc++;
                    else if (cc + 1 <= hcc &&
                             2 * sweep_time[cc] - tdiff < tdiff) {
                        tdiff = 2 * sweep_time[cc] - tdiff, cc++;
                        break;
                    } else
                        break;
                }
            else if (partition_time[ip + 1] < partition_time[ip])
                for (int i = 1; i <= conn_adjust_step; i++) {
                    if (cc - 1 >= lcc && 2 * sweep_time[cc - 2] <= tdiff)
                        tdiff -= 2 * sweep_time[cc - 2], cc--;
                    else if (cc - 1 >= lcc &&
                             2 * sweep_time[cc - 2] - tdiff < tdiff) {
                        tdiff = 2 * sweep_time[cc - 2] - tdiff, cc--;
                        break;
                    } else
                        break;
                }
            new_conn_centers[ip] = cc;
        }
        if (iprint >= 2) {
            stringstream sout;
            sout << fixed << setprecision(3);
            if (para_mps->rule != nullptr)
                sout << " SW-Group = " << para_mps->rule->comm->group;
            sout << " | Trot = " << me->trot << " | Tctr = " << me->tctr
                 << " | Tint = " << me->tint << " | Tmid = " << me->tmid
                 << " | Tdctr = " << me->tdctr << " | Tdiag = " << me->tdiag
                 << " | Tinfo = " << me->tinfo << endl;
            sout << " | Teff = " << teff << " | Tprt = " << tprt
                 << " | Teig = " << teig << " | Tblk = " << tblk
                 << " | Tmve = " << tmve << " | Tdm = " << tdm
                 << " | Tsplt = " << tsplt << " | Tsvd = " << tsvd
                 << " | Torth = " << torth;
            sout << endl;
            cout << sout.rdbuf();
        }
        for (int ip = 0; ip < para_mps->ncenter; ip++)
            if (para_mps->rule == nullptr ||
                ip % para_mps->rule->comm->ngroup ==
                    para_mps->rule->comm->group)
                connection_sweep(ip, bond_dim, noise, davidson_conv_thrd,
                                 new_conn_centers[ip]);
        para_mps->sync_canonical_form();
        if (para_mps->rule != nullptr) {
            para_mps->rule->comm->allreduce_min(sweep_energies);
            para_mps->rule->comm->allreduce_max(sweep_discarded_weights);
        }
        para_mps->disable_parallel_writing();
        size_t idx =
            min_element(sweep_energies.begin(), sweep_energies.end(),
                        [](const vector<FPLS> &x, const vector<FPLS> &y) {
                            return x.back() < y.back();
                        }) -
            sweep_energies.begin();
        if (iprint >= 2) {
            cout << "Time unordered " << fixed << setprecision(3);
            for (int ip = 0; ip <= para_mps->ncenter; ip++) {
                int pi = ip == 0 ? 0 : old_conn_centers[ip - 1];
                int pj = ip == para_mps->ncenter ? me->n_sites
                                                 : old_conn_centers[ip];
                int npi = ip == 0 ? 0 : new_conn_centers[ip - 1];
                int npj = ip == para_mps->ncenter ? me->n_sites
                                                  : new_conn_centers[ip];
                cout << "| [" << ip << "] " << pi << "~" << pj - 1 << " ("
                     << pj - pi;
                if (npj - npi > pj - pi)
                    cout << "+" << (npj - npi) - (pj - pi);
                else if (npj - npi < pj - pi)
                    cout << (npj - npi) - (pj - pi);
                cout << ") = " << partition_time[ip] << " ";
            }
            cout << endl;
        }
        para_mps->conn_centers = new_conn_centers;
        if (frame_<FPS>()->restart_dir != "" &&
            (para_mps->rule == nullptr || para_mps->rule->comm->group == 0) &&
            (me->para_rule == nullptr || me->para_rule->is_root())) {
            para_mps->save_data();
            if (!Parsing::path_exists(frame_<FPS>()->restart_dir))
                Parsing::mkdir(frame_<FPS>()->restart_dir);
            para_mps->info->copy_mutable(frame_<FPS>()->restart_dir);
            para_mps->copy_data(frame_<FPS>()->restart_dir);
        }
        if (frame_<FPS>()->restart_dir_per_sweep != "" &&
            (para_mps->rule == nullptr || para_mps->rule->comm->group == 0) &&
            (me->para_rule == nullptr || me->para_rule->is_root())) {
            para_mps->save_data();
            string rdps = frame_<FPS>()->restart_dir_per_sweep + "." +
                          Parsing::to_string((int)energies.size());
            if (!Parsing::path_exists(rdps))
                Parsing::mkdir(rdps);
            para_mps->info->copy_mutable(rdps);
            para_mps->copy_data(rdps);
        }
        FPS max_dw = *max_element(sweep_discarded_weights.begin(),
                                  sweep_discarded_weights.end());
        return make_tuple(sweep_energies[idx], max_dw, sweep_quanta[idx]);
    }
    // energy optimization using multiple DMRG sweeps
    FPLS solve(int n_sweeps, bool forward = true, FPS tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.size() == 0 ? 0.0 : noises.back());
        if (davidson_conv_thrds.size() < n_sweeps)
            for (size_t i = davidson_conv_thrds.size(); i < noises.size();
                 i++) {
                davidson_conv_thrds.push_back(
                    (noises[i] == 0 ? (tol == 0 ? 1E-9 : tol) : noises[i]) *
                    0.1);
                if (is_same<FPS, float>::value)
                    davidson_conv_thrds.back() =
                        max(davidson_conv_thrds.back(), (FPS)5E-6);
            }
        shared_ptr<ParallelMPS<S, FLS>> para_mps =
            me->ket->get_type() == MPSTypes::MultiCenter
                ? dynamic_pointer_cast<ParallelMPS<S, FLS>>(me->ket)
                : nullptr;
        if (me->bra != me->ket && para_mps != nullptr)
            throw runtime_error(
                "Parallel MPS and different bra and ket is not yet supported!");
        if ((me->bra != me->ket &&
             !((davidson_type & DavidsonTypes::NonHermitian) &&
               (davidson_type & DavidsonTypes::LeftEigen))) ||
            (me->bra == me->ket &&
             ((davidson_type & DavidsonTypes::NonHermitian) &&
              (davidson_type & DavidsonTypes::LeftEigen))))
            throw runtime_error(
                "Different BRA and KET must be used together "
                "with non-hermitian Hamiltonian and left eigen vector!");
        Timer start, current;
        start.get_time();
        current.get_time();
        energies.clear();
        discarded_weights.clear();
        mps_quanta.clear();
        bool converged;
        FPS energy_difference;
        if (iprint >= 1)
            cout << endl;
        for (int iw = 0; iw < n_sweeps; iw++) {
            isweep = iw;
            if (iprint >= 1)
                cout << "Sweep = " << setw(4) << iw
                     << " | Direction = " << setw(8)
                     << (forward ? "forward" : "backward")
                     << " | Bond dimension = " << setw(4)
                     << (uint32_t)bond_dims[iw] << " | Noise = " << scientific
                     << setw(9) << setprecision(2) << noises[iw]
                     << " | Dav threshold = " << scientific << setw(9)
                     << setprecision(2) << davidson_conv_thrds[iw] << endl;
            auto sweep_results =
                para_mps != nullptr
                    ? unordered_sweep(forward, bond_dims[iw], noises[iw],
                                      davidson_conv_thrds[iw])
                    : sweep(forward, bond_dims[iw], noises[iw],
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
            double tswp = current.get_time();
            if (iprint >= 1) {
                cout << "Time elapsed = " << fixed << setw(10)
                     << setprecision(3) << current.current - start.current;
                cout << fixed << setprecision(10);
                if (get<0>(sweep_results).size() == 1)
                    cout << " | E = " << setw(18) << get<0>(sweep_results)[0];
                else {
                    cout << " | E[" << setw(3) << get<0>(sweep_results).size()
                         << "] = ";
                    for (FPS x : get<0>(sweep_results))
                        cout << setw(18) << x;
                }
                if (energies.size() >= 2)
                    cout << " | DE = " << setw(6) << setprecision(2)
                         << scientific << energy_difference;
                cout << " | DW = " << setw(6) << setprecision(2) << scientific
                     << get<1>(sweep_results) << endl;
                if (iprint >= 2) {
                    cout << fixed << setprecision(3);
                    cout << "Time sweep = " << setw(12) << tswp;
                    cout << " | "
                         << Parsing::to_size_string(sweep_cumulative_nflop,
                                                    "FLOP/SWP")
                         << endl;
                    if (para_mps != nullptr && para_mps->rule != nullptr) {
                        shared_ptr<ParallelCommunicator<S>> comm =
                            para_mps->rule->comm;
                        double tt[2] = {comm->tcomm, comm->tidle};
                        comm->reduce_sum_optional(&tt[0], 2, comm->root);
                        comm->reduce_sum_optional(
                            (uint64_t *)&sweep_cumulative_nflop, 1, comm->root);
                        cout << " | GTcomm = " << tt[0] / comm->size
                             << " | GTidle = " << tt[1] / comm->size << endl;
                    }
                    if (para_mps != nullptr && para_mps->rule != nullptr) {
                        para_mps->enable_parallel_writing();
                        para_mps->rule->comm->barrier();
                    }
                }
                if (iprint >= 2) {
                    stringstream sout;
                    sout << fixed << setprecision(3);
                    if (para_mps != nullptr && para_mps->rule != nullptr)
                        sout << " Group = " << para_mps->rule->comm->group;
                    if (me->para_rule != nullptr) {
                        shared_ptr<ParallelCommunicator<S>> comm =
                            me->para_rule->comm;
                        double tt[3] = {comm->tcomm, comm->tidle, comm->twait};
                        comm->reduce_sum_optional(&tt[0], 3, comm->root);
                        sout << " | Tcomm = " << tt[0] / comm->size
                             << " | Tidle = " << tt[1] / comm->size
                             << " | Twait = " << tt[2] / comm->size;
                    }
                    size_t dmain = frame_<FPS>()->peak_used_memory[0];
                    size_t dseco = frame_<FPS>()->peak_used_memory[1];
                    size_t imain = frame_<FPS>()->peak_used_memory[2];
                    size_t iseco = frame_<FPS>()->peak_used_memory[3];
                    sout << " | Dmem = "
                         << Parsing::to_size_string(dmain + dseco) << " ("
                         << (dmain * 100 / (dmain + dseco)) << "%)";
                    sout << " | Imem = "
                         << Parsing::to_size_string(imain + iseco) << " ("
                         << (imain * 100 / (imain + iseco)) << "%)";
                    sout << " | Hmem = "
                         << Parsing::to_size_string(sweep_max_eff_ham_size *
                                                    sizeof(FL));
                    sout << " | Pmem = "
                         << Parsing::to_size_string(sweep_max_pket_size *
                                                    sizeof(FLS))
                         << endl;
                    sout << " | Tread = " << frame_<FPS>()->tread
                         << " | Twrite = " << frame_<FPS>()->twrite
                         << " | Tfpread = " << frame_<FPS>()->fpread
                         << " | Tfpwrite = " << frame_<FPS>()->fpwrite
                         << " | Tmporead = " << me->mpo->tread
                         << " | Tasync = " << frame_<FPS>()->tasync << endl;
                    if (frame_<FPS>()->fp_codec != nullptr)
                        sout
                            << " | data = "
                            << Parsing::to_size_string(
                                   frame_<FPS>()->fp_codec->ndata * sizeof(FPS))
                            << " | cpsd = "
                            << Parsing::to_size_string(
                                   frame_<FPS>()->fp_codec->ncpsd * sizeof(FPS))
                            << endl;
                    sout << " | Trot = " << me->trot << " | Tctr = " << me->tctr
                         << " | Tint = " << me->tint << " | Tmid = " << me->tmid
                         << " | Tdctr = " << me->tdctr
                         << " | Tdiag = " << me->tdiag
                         << " | Tinfo = " << me->tinfo << endl;
                    sout << " | Teff = " << teff << " | Tprt = " << tprt
                         << " | Teig = " << teig << " | Tblk = " << tblk
                         << " | Tmve = " << tmve << " | Tdm = " << tdm
                         << " | Tsplt = " << tsplt << " | Tsvd = " << tsvd
                         << " | Torth = " << torth;
                    sout << endl;
                    cout << sout.rdbuf();
                    if (para_mps != nullptr && para_mps->rule != nullptr) {
                        para_mps->disable_parallel_writing();
                        para_mps->rule->comm->barrier();
                    }
                }
                cout << endl;
            }
            if (converged || has_abort_file())
                break;
        }
        this->forward = forward;
        if (!converged && iprint > 0 && tol != 0)
            cout << "ATTENTION: DMRG is not converged to desired tolerance of "
                 << scientific << tol << endl;
        return energies.back()[0];
    }
};

enum struct EquationTypes : uint8_t {
    Normal,
    PerturbativeCompression,
    GreensFunction,
    GreensFunctionSquared,
    FitAddition
};

enum struct ConvergenceTypes : uint8_t {
    LastMinimal,
    LastMaximal,
    FirstMinimal,
    FirstMaximal,
    MiddleSite
};

// Solve |x> in Linear Equation LHS|x> = RHS|r>
// where |r> is a constant MPS
// target quantity is calculated in tme
// if lme == nullptr, LHS = 1 (compression)
// if tme == lme == nullptr, target is sqrt(<x|x>)
// if tme == nullptr, target is <x|RHS|r>
// when lme != nullptr, eq_type == PerturbativeCompression
//    then lme is only used to do perturbative noise
//    the equation in this case is 1 |x> = RHS |r> (compression)
// when eq_type == FitAddition
//    This is 1 |x> = RHS|r> + THS|t>
//    rme = <x|RHS|r>, tme = <x|THS|t>,
//      (optionally: lme = <x|H|x> for perturbative noise)
//    prefactor in addtion can be introduced by
//    doing scalar multiplication on MPO in RHS/THS
template <typename S, typename FL, typename FLS> struct Linear {
    typedef typename MovingEnvironment<S, FL, FLS>::FPS FPS;
    typedef typename MovingEnvironment<S, FL, FLS>::FCS FCS;
    // lme = LHS ME, rme = RHS ME, tme = Target ME
    shared_ptr<MovingEnvironment<S, FL, FLS>> lme, rme, tme;
    // ext mes for gf off-diagonals
    vector<shared_ptr<MovingEnvironment<S, FL, FLS>>> ext_tmes;
    vector<shared_ptr<MPS<S, FLS>>> ext_mpss;
    vector<vector<FLS>> ext_targets;
    int ext_target_at_site = -1;
    vector<ubond_t> bra_bond_dims, ket_bond_dims;
    vector<FPS> noises;
    vector<vector<FLS>> targets;
    vector<FPS> discarded_weights;
    vector<vector<FLS>> sweep_targets;
    vector<FPS> sweep_discarded_weights;
    vector<FPS> linear_conv_thrds;
    int linear_max_iter = 5000;
    int linear_soft_max_iter = -1;
    int conv_required_sweeps = 3;
    ConvergenceTypes conv_type = ConvergenceTypes::LastMinimal;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    EquationTypes eq_type = EquationTypes::Normal;
    ExpectationAlgorithmTypes algo_type = ExpectationAlgorithmTypes::Normal;
    ExpectationTypes ex_type = ExpectationTypes::Real;
    LinearSolverTypes solver_type = LinearSolverTypes::Automatic;
    bool forward;
    uint8_t iprint = 2;
    FPS cutoff = 1E-14;
    bool decomp_last_site = true;
    size_t sweep_cumulative_nflop = 0;
    size_t sweep_max_pket_size = 0;
    size_t sweep_max_eff_ham_size = 0;
    double tprt = 0, tmult = 0, teff = 0, tmve = 0, tblk = 0, tdm = 0,
           tsplt = 0, tsvd = 0;
    Timer _t, _t2;
    bool linear_use_precondition = true;
    // number of eigenvalues solved using harmonic Davidson
    // for deflated CG; 0 means normal CG
    int cg_n_harmonic_projection = 0;
    // First entry also used for "S" in IDR(S). Second entry will then be
    // ignored
    pair<int, int> linear_solver_params = make_pair(40, -1);
    // weight for mixing rhs wavefunction in density matrix/svd
    FPS right_weight = 0.0;
    // only useful when target contains some other
    // constant MPS not appeared in the equation
    int target_bra_bond_dim = -1;
    int target_ket_bond_dim = -1;
    // weights for real and imag parts
    vector<FPS> complex_weights = {0.5, 0.5};
    // Green's function parameters
    FLS gf_omega = 0, gf_eta = 0;
    // extra frequencies calculated only at the given site
    vector<FLS> gf_extra_omegas;
    // calculated GF for extra frequencies
    vector<vector<FLS>> gf_extra_targets;
    // which site to calculate extra frequencies
    int gf_extra_omegas_at_site = -1;
    // if not zero, use this eta for extra frequencies
    FLS gf_extra_eta = 0;
    // calculated GF for extra frequencies and ext_mpss
    vector<vector<vector<FLS>>> gf_extra_ext_targets;
    // store all wfn singular values (for analysis) at each site
    bool store_bra_spectra = false, store_ket_spectra = false;
    vector<vector<FPS>> sweep_wfn_spectra;
    vector<FPS> wfn_spectra;
    int sweep_start_site = 0;
    int sweep_end_site = -1;
    Linear(const shared_ptr<MovingEnvironment<S, FL, FLS>> &lme,
           const shared_ptr<MovingEnvironment<S, FL, FLS>> &rme,
           const shared_ptr<MovingEnvironment<S, FL, FLS>> &tme,
           const vector<ubond_t> &bra_bond_dims,
           const vector<ubond_t> &ket_bond_dims,
           const vector<FPS> &noises = vector<FPS>())
        : lme(lme), rme(rme), tme(tme), bra_bond_dims(bra_bond_dims),
          ket_bond_dims(ket_bond_dims), noises(noises), forward(false) {
        if (lme != nullptr) {
            assert(lme->bra == lme->ket && lme->bra == rme->bra);
            assert(lme->tag != rme->tag);
        }
        if (tme != nullptr) {
            assert(tme->tag != rme->tag);
            if (lme != nullptr)
                assert(tme->tag != lme->tag);
        }
        sweep_end_site = rme->n_sites;
    }
    Linear(const shared_ptr<MovingEnvironment<S, FL, FLS>> &rme,
           const vector<ubond_t> &bra_bond_dims,
           const vector<ubond_t> &ket_bond_dims,
           const vector<FPS> &noises = vector<FPS>())
        : Linear(nullptr, rme, nullptr, bra_bond_dims, ket_bond_dims, noises) {}
    Linear(const shared_ptr<MovingEnvironment<S, FL, FLS>> &lme,
           const shared_ptr<MovingEnvironment<S, FL, FLS>> &rme,
           const vector<ubond_t> &bra_bond_dims,
           const vector<ubond_t> &ket_bond_dims,
           const vector<FPS> &noises = vector<FPS>())
        : Linear(lme, rme, nullptr, bra_bond_dims, ket_bond_dims, noises) {}
    virtual ~Linear() = default;
    struct Iteration {
        vector<FLS> targets;
        FPS error;
        double tmult;
        int nmult, nmultp, mmps;
        size_t nflop;
        Iteration(const vector<FLS> &targets, FPS error, int mmps, int nmult,
                  int nmultp, size_t nflop = 0, double tmult = 1.0)
            : targets(targets), error(error), mmps(mmps), nmult(nmult),
              nmultp(nmultp), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Mmps =" << setw(5) << r.mmps;
            os << " Nmult = " << setw(4) << r.nmult;
            if (r.nmultp != 0)
                os << "/" << setw(4) << r.nmultp;
            if (r.targets.size() == 1) {
                os << (abs(r.targets[0]) > 1E-3 ? fixed : scientific);
                os << (abs(r.targets[0]) > 1E-3 ? setprecision(10)
                                                : setprecision(7));
                os << " F = " << setw(17) << r.targets[0];
            } else {
                os << " F = ";
                for (auto x : r.targets) {
                    os << (abs(x) > 1E-3 ? fixed : scientific);
                    os << (abs(x) > 1E-3 ? setprecision(10) : setprecision(7));
                    os << setw(17) << x;
                }
            }
            os << " Error = " << scientific << setw(8) << setprecision(2)
               << r.error << " FLOPS = " << scientific << setw(8)
               << setprecision(2)
               << (r.tmult == 0.0 ? 0 : (double)r.nflop / r.tmult)
               << " Tmult = " << fixed << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_one_dot(int i, bool forward, ubond_t bra_bond_dim,
                             ubond_t ket_bond_dim, FPS noise,
                             FPS linear_conv_thrd) {
        const shared_ptr<MovingEnvironment<S, FL, FLS>> &me = rme;
        assert(me->bra != me->ket);
        frame_<FPS>()->activate(0);
        bool fuse_left = i <= me->fuse_center;
        vector<shared_ptr<MPS<S, FLS>>> mpss = {me->bra, me->ket};
        if (tme != nullptr) {
            if (tme->bra != me->bra && tme->bra != me->ket)
                mpss.push_back(tme->bra);
            if (tme->ket != me->bra && tme->ket != me->ket &&
                tme->ket != tme->bra)
                mpss.push_back(tme->ket);
        }
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
        shared_ptr<SparseMatrix<S, FLS>> right_bra = me->bra->tensors[i],
                                         real_bra = nullptr;
        shared_ptr<SparseMatrixGroup<S, FLS>> pbra = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> pdm = nullptr;
        bool skip_decomp =
            !decomp_last_site &&
            ((forward && i == sweep_end_site - 1 && !fuse_left) ||
             (!forward && i == sweep_start_site && fuse_left));
        bool build_pdm = noise != 0 && (noise_type & NoiseTypes::Collected);
        if ((lme != nullptr &&
             eq_type != EquationTypes::PerturbativeCompression) ||
            eq_type == EquationTypes::FitAddition) {
            right_bra = make_shared<SparseMatrix<S, FLS>>();
            right_bra->allocate(me->bra->tensors[i]->info);
            if ((eq_type == EquationTypes::GreensFunction ||
                 eq_type == EquationTypes::GreensFunctionSquared) &&
                !is_same<FLS, FCS>::value) {
                real_bra = make_shared<SparseMatrix<S, FLS>>();
                real_bra->allocate(me->bra->tensors[i]->info);
            }
        }
        _t.get_time();
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff =
            me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                        forward, false, right_bra, me->ket->tensors[i]);
        teff += _t.get_time();
        tuple<FLS, pair<int, int>, size_t, double> pdi;
        auto mpdi = h_eff->multiply(me->mpo->const_e, me->para_rule);
        get<0>(pdi) = get<0>(mpdi);
        get<1>(pdi).first = get<1>(mpdi);
        get<2>(pdi) = get<2>(mpdi);
        get<3>(pdi) = get<3>(mpdi);
        tmult += _t.get_time();
        vector<FLS> targets = {get<0>(pdi)};
        vector<FLS> extra_bras;
        h_eff->deallocate();
        if (eq_type == EquationTypes::FitAddition ||
            eq_type == EquationTypes::PerturbativeCompression) {
            if (eq_type == EquationTypes::FitAddition) {
                shared_ptr<EffectiveHamiltonian<S, FL>> t_eff = tme->eff_ham(
                    fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward,
                    false, tme->bra->tensors[i], tme->ket->tensors[i]);
                teff += _t.get_time();
                auto tpdi = t_eff->multiply(tme->mpo->const_e, tme->para_rule);
                GMatrix<FLS> mbra(me->bra->tensors[i]->data,
                                  (MKL_INT)me->bra->tensors[i]->total_memory,
                                  1);
                GMatrix<FLS> sbra(right_bra->data,
                                  (MKL_INT)right_bra->total_memory, 1);
                GMatrixFunctions<FLS>::iadd(mbra, sbra, 1);
                tmult += _t.get_time();
                targets[0] = GMatrixFunctions<FLS>::norm(mbra);
                get<1>(pdi).first += get<1>(tpdi);
                get<2>(pdi) += get<2>(tpdi);
                get<3>(pdi) += get<3>(tpdi);
                t_eff->deallocate();
            }
            if (lme != nullptr && noise != 0) {
                shared_ptr<EffectiveHamiltonian<S, FL>> l_eff = lme->eff_ham(
                    fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward,
                    false, lme->bra->tensors[i], lme->ket->tensors[i]);
                teff += _t.get_time();
                if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
                    pbra = l_eff->perturbative_noise(
                        forward, i, i,
                        fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                        me->bra->info, noise_type, me->para_rule);
                tprt += _t.get_time();
                l_eff->deallocate();
            }
        } else if (lme != nullptr) {
            shared_ptr<EffectiveHamiltonian<S, FL>> l_eff = lme->eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward,
                linear_use_precondition, me->bra->tensors[i], right_bra);
            sweep_max_eff_ham_size =
                max(sweep_max_eff_ham_size, l_eff->op->get_total_memory());
            teff += _t.get_time();
            if (eq_type == EquationTypes::Normal) {
                tuple<FLS, pair<int, int>, size_t, double> lpdi;
                lpdi = l_eff->inverse_multiply(
                    lme->mpo->const_e, solver_type, linear_solver_params,
                    iprint >= 3, linear_conv_thrd, linear_max_iter,
                    linear_soft_max_iter, me->para_rule);
                targets[0] = get<0>(lpdi);
                get<1>(pdi).first += get<1>(lpdi).first;
                get<1>(pdi).second += get<1>(lpdi).second;
                get<2>(pdi) += get<2>(lpdi), get<3>(pdi) += get<3>(lpdi);
            } else if (eq_type == EquationTypes::GreensFunction ||
                       eq_type == EquationTypes::GreensFunctionSquared) {
                tuple<FCS, pair<int, int>, size_t, double> lpdi;
                if (gf_extra_omegas_at_site == i &&
                    gf_extra_omegas.size() != 0) {
                    gf_extra_targets.resize(gf_extra_omegas.size());
                    GMatrix<FLS> tmp(nullptr, (MKL_INT)l_eff->bra->total_memory,
                                     1);
                    tmp.allocate();
                    memcpy(tmp.data, l_eff->bra->data,
                           l_eff->bra->total_memory * sizeof(FLS));
                    if (tme != nullptr || ext_tmes.size() != 0)
                        extra_bras.reserve(l_eff->bra->total_memory *
                                           gf_extra_omegas.size() * 2);
                    for (size_t j = 0; j < gf_extra_omegas.size(); j++) {
                        if (eq_type == EquationTypes::GreensFunctionSquared)
                            lpdi = EffectiveFunctions<S, FL>::
                                greens_function_squared(
                                    l_eff, lme->mpo->const_e,
                                    gf_extra_omegas[j],
                                    gf_extra_eta == (FLS)0.0 ? gf_eta
                                                             : gf_extra_eta,
                                    real_bra, cg_n_harmonic_projection,
                                    iprint >= 3, linear_conv_thrd,
                                    linear_max_iter, linear_soft_max_iter,
                                    me->para_rule);
                        else
                            lpdi = EffectiveFunctions<S, FL>::greens_function(
                                l_eff, lme->mpo->const_e, solver_type,
                                gf_extra_omegas[j],
                                gf_extra_eta == (FLS)0.0 ? gf_eta
                                                         : gf_extra_eta,
                                real_bra, linear_solver_params, iprint >= 3,
                                linear_conv_thrd, linear_max_iter,
                                linear_soft_max_iter, me->para_rule);
                        if (tme != nullptr || ext_tmes.size() != 0) {
                            memcpy(extra_bras.data() +
                                       j * 2 * l_eff->bra->total_memory,
                                   l_eff->bra->data,
                                   l_eff->bra->total_memory * sizeof(FLS));
                            if (real_bra != nullptr)
                                memcpy(extra_bras.data() +
                                           (j * 2 + 1) *
                                               l_eff->bra->total_memory,
                                       real_bra->data,
                                       real_bra->total_memory * sizeof(FLS));
                        }
                        gf_extra_targets[j] =
                            is_same<FLS, FCS>::value
                                ? vector<FLS>{(FLS &)get<0>(lpdi)}
                                : vector<FLS>{xreal(get<0>(lpdi)),
                                              ximag(get<0>(lpdi))};
                        get<1>(pdi).first += get<1>(lpdi).first;
                        get<1>(pdi).second += get<1>(lpdi).second;
                        get<2>(pdi) += get<2>(lpdi),
                            get<3>(pdi) += get<3>(lpdi);
                    }
                    memcpy(l_eff->bra->data, tmp.data,
                           l_eff->bra->total_memory * sizeof(FLS));
                    tmp.deallocate();
                }
                if (eq_type == EquationTypes::GreensFunctionSquared)
                    lpdi = EffectiveFunctions<S, FL>::greens_function_squared(
                        l_eff, lme->mpo->const_e, gf_omega, gf_eta, real_bra,
                        cg_n_harmonic_projection, iprint >= 3, linear_conv_thrd,
                        linear_max_iter, linear_soft_max_iter, me->para_rule);
                else
                    lpdi = EffectiveFunctions<S, FL>::greens_function(
                        l_eff, lme->mpo->const_e, solver_type, gf_omega, gf_eta,
                        real_bra, linear_solver_params, iprint >= 3,
                        linear_conv_thrd, linear_max_iter, linear_soft_max_iter,
                        me->para_rule);
                targets =
                    is_same<FLS, FCS>::value
                        ? vector<FLS>{(FLS &)get<0>(lpdi)}
                        : vector<FLS>{xreal(get<0>(lpdi)), ximag(get<0>(lpdi))};
                get<1>(pdi).first += get<1>(lpdi).first;
                get<1>(pdi).second += get<1>(lpdi).second;
                get<2>(pdi) += get<2>(lpdi), get<3>(pdi) += get<3>(lpdi);
            } else
                assert(false);
            tmult += _t.get_time();
            if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
                pbra = l_eff->perturbative_noise(
                    forward, i, i,
                    fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                    me->bra->info, noise_type, me->para_rule);
            tprt += _t.get_time();
            l_eff->deallocate();
        }
        if (pbra != nullptr)
            sweep_max_pket_size = max(sweep_max_pket_size, pbra->total_memory);
        if (tme != nullptr && eq_type != EquationTypes::FitAddition) {
            shared_ptr<EffectiveHamiltonian<S, FL>> t_eff = tme->eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, false,
                tme->bra->tensors[i], tme->ket->tensors[i]);
            teff += _t.get_time();
            auto tpdi = t_eff->expect(tme->mpo->const_e, algo_type, ex_type,
                                      tme->para_rule);
            targets.clear();
            get<1>(pdi).first++;
            get<2>(pdi) += get<1>(tpdi);
            get<3>(pdi) += get<2>(tpdi);
            targets.push_back(get<0>(tpdi)[0].second);
            if (real_bra != nullptr) {
                if (tme->bra->tensors[i] == me->bra->tensors[i])
                    t_eff->bra = real_bra;
                if (tme->ket->tensors[i] == me->bra->tensors[i])
                    t_eff->ket = real_bra;
                tpdi = t_eff->expect(tme->mpo->const_e, algo_type, ex_type,
                                     tme->para_rule);
                targets.insert(targets.begin(), get<0>(tpdi)[0].second);
                get<1>(pdi).first++;
                get<2>(pdi) += get<1>(tpdi);
                get<3>(pdi) += get<2>(tpdi);
            }
            if (gf_extra_omegas_at_site == i && gf_extra_omegas.size() != 0)
                for (size_t j = 0; j < gf_extra_targets.size(); j++) {
                    FLS *tbra_bak = t_eff->bra->data;
                    FLS *tket_bak = t_eff->ket->data;
                    if (tme->bra->tensors[i] == me->bra->tensors[i])
                        t_eff->bra->data = extra_bras.data() +
                                           j * 2 * t_eff->bra->total_memory;
                    if (tme->ket->tensors[i] == me->bra->tensors[i])
                        t_eff->ket->data = extra_bras.data() +
                                           j * 2 * t_eff->ket->total_memory;
                    tpdi = t_eff->expect(tme->mpo->const_e, algo_type, ex_type,
                                         tme->para_rule);
                    get<1>(pdi).first++;
                    get<2>(pdi) += get<1>(tpdi);
                    get<3>(pdi) += get<2>(tpdi);
                    if (is_same<FLS, FCS>::value)
                        gf_extra_targets[j][0] = get<0>(tpdi)[0].second;
                    else {
                        gf_extra_targets[j][1] = get<0>(tpdi)[0].second;
                        if (tme->bra->tensors[i] == me->bra->tensors[i])
                            t_eff->bra->data =
                                extra_bras.data() +
                                (j * 2 + 1) * t_eff->bra->total_memory;
                        if (tme->ket->tensors[i] == me->bra->tensors[i])
                            t_eff->ket->data =
                                extra_bras.data() +
                                (j * 2 + 1) * t_eff->ket->total_memory;
                        tpdi = t_eff->expect(tme->mpo->const_e, algo_type,
                                             ex_type, tme->para_rule);
                        gf_extra_targets[j][0] = get<0>(tpdi)[0].second;
                        get<1>(pdi).first++;
                        get<2>(pdi) += get<1>(tpdi);
                        get<3>(pdi) += get<2>(tpdi);
                    }
                    t_eff->bra->data = tbra_bak;
                    t_eff->ket->data = tket_bak;
                }
            tmult += _t.get_time();
            t_eff->deallocate();
        }
        for (auto &mps : ext_mpss) {
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
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (ext_target_at_site == i && ext_tmes.size() != 0) {
            ext_targets.resize(ext_tmes.size());
            if (gf_extra_omegas_at_site == i && gf_extra_omegas.size() != 0)
                gf_extra_ext_targets.resize(ext_tmes.size());
            for (size_t k = 0; k < ext_tmes.size(); k++) {
                shared_ptr<MovingEnvironment<S, FL, FLS>> xme = ext_tmes[k];
                shared_ptr<EffectiveHamiltonian<S, FL>> t_eff = xme->eff_ham(
                    fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward,
                    false, xme->bra->tensors[i], xme->ket->tensors[i]);
                auto tpdi = t_eff->expect(xme->mpo->const_e, algo_type, ex_type,
                                          xme->para_rule);
                ext_targets[k].resize(2);
                ext_targets[k][1] = get<0>(tpdi)[0].second;
                get<1>(pdi).first++;
                get<2>(pdi) += get<1>(tpdi);
                get<3>(pdi) += get<2>(tpdi);
                if (real_bra != nullptr) {
                    if (xme->bra->tensors[i] == me->bra->tensors[i])
                        t_eff->bra = real_bra;
                    if (xme->ket->tensors[i] == me->bra->tensors[i])
                        t_eff->ket = real_bra;
                }
                tpdi = t_eff->expect(xme->mpo->const_e, algo_type, ex_type,
                                     xme->para_rule);
                ext_targets[k][0] = get<0>(tpdi)[0].second;
                get<1>(pdi).first++;
                get<2>(pdi) += get<1>(tpdi);
                get<3>(pdi) += get<2>(tpdi);
                if (gf_extra_omegas_at_site == i &&
                    gf_extra_omegas.size() != 0) {
                    gf_extra_ext_targets[k].resize(gf_extra_omegas.size());
                    for (size_t j = 0; j < gf_extra_omegas.size(); j++) {
                        FLS *tbra_bak = t_eff->bra->data;
                        FLS *tket_bak = t_eff->ket->data;
                        if (xme->bra->tensors[i] == me->bra->tensors[i])
                            t_eff->bra->data = extra_bras.data() +
                                               j * 2 * t_eff->bra->total_memory;
                        if (xme->ket->tensors[i] == me->bra->tensors[i])
                            t_eff->ket->data = extra_bras.data() +
                                               j * 2 * t_eff->ket->total_memory;
                        tpdi = t_eff->expect(xme->mpo->const_e, algo_type,
                                             ex_type, xme->para_rule);
                        get<1>(pdi).first++;
                        get<2>(pdi) += get<1>(tpdi);
                        get<3>(pdi) += get<2>(tpdi);
                        if (is_same<FLS, FCS>::value) {
                            gf_extra_ext_targets[k][j].resize(1);
                            gf_extra_ext_targets[k][j][0] =
                                get<0>(tpdi)[0].second;
                        } else {
                            gf_extra_ext_targets[k][j].resize(2);
                            gf_extra_ext_targets[k][j][1] =
                                get<0>(tpdi)[0].second;
                            if (xme->bra->tensors[i] == me->bra->tensors[i])
                                t_eff->bra->data =
                                    extra_bras.data() +
                                    (j * 2 + 1) * t_eff->bra->total_memory;
                            if (xme->ket->tensors[i] == me->bra->tensors[i])
                                t_eff->ket->data =
                                    extra_bras.data() +
                                    (j * 2 + 1) * t_eff->ket->total_memory;
                            tpdi = t_eff->expect(xme->mpo->const_e, algo_type,
                                                 ex_type, xme->para_rule);
                            gf_extra_ext_targets[k][j][0] =
                                get<0>(tpdi)[0].second;
                            get<1>(pdi).first++;
                            get<2>(pdi) += get<1>(tpdi);
                            get<3>(pdi) += get<2>(tpdi);
                        }
                        t_eff->bra->data = tbra_bak;
                        t_eff->ket->data = tket_bak;
                    }
                }
                t_eff->deallocate();
            }
        }
        for (auto &mps : ext_mpss) {
            if ((me->para_rule == nullptr || me->para_rule->is_root()) &&
                !skip_decomp) {
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
            if (skip_decomp)
                mps->canonical_form[i] = forward ? 'S' : 'K';
            else {
                mps->canonical_form[i] = forward ? 'K' : 'S';
                if (forward && i != sweep_end_site - 1)
                    mps->move_right(me->mpo->tf->opf->cg, me->para_rule);
                else if (!forward && i != sweep_start_site)
                    mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
            }
        }
        if ((build_pdm || me->para_rule == nullptr ||
             me->para_rule->is_root()) &&
            !skip_decomp) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                if (real_bra != nullptr) {
                    shared_ptr<SparseMatrix<S, FLS>> prev_wfn = real_bra;
                    if (!fuse_left && forward)
                        real_bra = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_left(i, me->bra->info, prev_wfn,
                                                   me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        real_bra = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_right(i, me->bra->info, prev_wfn,
                                                    me->mpo->tf->opf->cg);
                    prev_wfn->deallocate();
                }
                if (right_weight != 0 && right_bra != me->bra->tensors[i]) {
                    shared_ptr<SparseMatrix<S, FLS>> prev_wfn = right_bra;
                    if (!fuse_left && forward)
                        right_bra = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_left(i, me->bra->info, prev_wfn,
                                                   me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        right_bra = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_right(i, me->bra->info, prev_wfn,
                                                    me->mpo->tf->opf->cg);
                    prev_wfn->deallocate();
                }
                for (auto &mps : mpss) {
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
                if (pbra != nullptr) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_pbras = {
                        pbra};
                    if (!fuse_left && forward)
                        pbra = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_left(
                                i, me->bra->info, prev_pbras,
                                me->mpo->tf->opf->cg)[0];
                    else if (fuse_left && !forward)
                        pbra = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_right(
                                i, me->bra->info, prev_pbras,
                                me->mpo->tf->opf->cg)[0];
                    prev_pbras[0]->deallocate_infos();
                    prev_pbras[0]->deallocate();
                }
            }
        }
        if (build_pdm && !skip_decomp) {
            _t.get_time();
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            pdm = MovingEnvironment<S, FL, FLS>::density_matrix(
                me->bra->info->vacuum, me->bra->tensors[i], forward,
                me->para_rule != nullptr ? noise / me->para_rule->comm->size
                                         : noise,
                noise_type, 0.0, pbra);
            if (me->para_rule != nullptr)
                me->para_rule->comm->reduce_sum(pdm, me->para_rule->comm->root);
            tdm += _t.get_time();
        }
        FPS bra_error = 0.0;
        int bra_mmps = 0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (skip_decomp) {
                for (auto &mps : mpss) {
                    mps->save_tensor(i);
                    mps->unload_tensor(i);
                    mps->canonical_form[i] = forward ? 'S' : 'K';
                }
            } else {
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
                    bool store_wfn_spectra =
                        mps == me->bra ? store_bra_spectra : store_ket_spectra;
                    int bond_dim = -1;
                    FPS error;
                    if (mps == me->bra)
                        bond_dim = (int)bra_bond_dim;
                    else if (mps == me->ket)
                        bond_dim = (int)ket_bond_dim;
                    else if (tme != nullptr && mps == tme->bra)
                        bond_dim = target_bra_bond_dim;
                    else if (tme != nullptr && mps == tme->ket)
                        bond_dim = target_ket_bond_dim;
                    else
                        assert(false);
                    assert(right_weight >= 0 && right_weight <= 1);
                    if (decomp_type == DecompositionTypes::DensityMatrix) {
                        _t.get_time();
                        if (mps != me->bra) {
                            dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                                mps->info->vacuum, old_wfn, forward, 0.0,
                                NoiseTypes::None);
                        } else {
                            FPS weight = 1 - right_weight;
                            if (real_bra != nullptr)
                                weight *= complex_weights[1];
                            dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                                mps->info->vacuum, old_wfn, forward,
                                build_pdm ? 0.0 : noise, noise_type, weight,
                                pbra);
                            if (build_pdm)
                                GMatrixFunctions<FLS>::iadd(
                                    GMatrix<FLS>(dm->data,
                                                 (MKL_INT)dm->total_memory, 1),
                                    GMatrix<FLS>(pdm->data,
                                                 (MKL_INT)pdm->total_memory, 1),
                                    1.0);
                            if (real_bra != nullptr) {
                                weight =
                                    complex_weights[0] * (1 - right_weight);
                                MovingEnvironment<S, FL, FLS>::
                                    density_matrix_add_wfn(dm, real_bra,
                                                           forward, weight);
                            }
                            if (right_weight != 0)
                                MovingEnvironment<S, FL, FLS>::
                                    density_matrix_add_wfn(
                                        dm, right_bra, forward, right_weight);
                        }
                        tdm += _t.get_time();
                        error =
                            MovingEnvironment<S, FL, FLS>::split_density_matrix(
                                dm, old_wfn, bond_dim, forward, false, left,
                                right, cutoff, store_wfn_spectra, wfn_spectra,
                                trunc_type);
                        tsplt += _t.get_time();
                    } else if (decomp_type == DecompositionTypes::SVD ||
                               decomp_type == DecompositionTypes::PureSVD) {
                        if (mps != me->bra) {
                            error = MovingEnvironment<S, FL, FLS>::
                                split_wavefunction_svd(
                                    mps->info->vacuum, old_wfn, bond_dim,
                                    forward, false, left, right, cutoff,
                                    store_wfn_spectra, wfn_spectra, trunc_type,
                                    decomp_type, nullptr);
                        } else {
                            if (noise != 0 && mps == me->bra) {
                                if (noise_type & NoiseTypes::Wavefunction)
                                    MovingEnvironment<S, FL, FLS>::
                                        wavefunction_add_noise(old_wfn, noise);
                                else if (noise_type & NoiseTypes::Perturbative)
                                    MovingEnvironment<S, FL, FLS>::
                                        scale_perturbative_noise(
                                            noise, noise_type, pbra);
                            }
                            vector<FPS> weights = {1};
                            vector<shared_ptr<SparseMatrix<S, FLS>>> xwfns = {};
                            if (real_bra != nullptr) {
                                weights = vector<FPS>{sqrt(complex_weights[1]),
                                                      sqrt(complex_weights[0])};
                                xwfns.push_back(real_bra);
                            }
                            if (right_weight != 0) {
                                for (auto w : weights)
                                    w = sqrt(w * w * (1 - right_weight));
                                weights.push_back(sqrt(right_weight));
                                xwfns.push_back(right_bra);
                            }
                            _t.get_time();
                            error = MovingEnvironment<S, FL, FLS>::
                                split_wavefunction_svd(
                                    mps->info->vacuum, old_wfn, bond_dim,
                                    forward, false, left, right, cutoff,
                                    store_wfn_spectra, wfn_spectra, trunc_type,
                                    decomp_type, pbra, xwfns, weights);
                            tsvd += _t.get_time();
                        }
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
                        if (mps == me->bra) {
                            bra_mmps = (int)info->n_states_total;
                            mps->info->bond_dim =
                                max(mps->info->bond_dim, (ubond_t)bra_mmps);
                        }
                        mps->info->left_dims[i + 1] = info;
                        mps->info->save_left_dims(i + 1);
                        info->deallocate();
                        if (i != sweep_end_site - 1) {
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
                        if (mps == me->bra) {
                            bra_mmps = (int)info->n_states_total;
                            mps->info->bond_dim =
                                max(mps->info->bond_dim, (ubond_t)bra_mmps);
                        }
                        mps->info->right_dims[i] = info;
                        mps->info->save_right_dims(i);
                        info->deallocate();
                        if (i > sweep_start_site) {
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
                    right->info->deallocate();
                    right->deallocate();
                    left->info->deallocate();
                    left->deallocate();
                    if (dm != nullptr) {
                        dm->info->deallocate();
                        dm->deallocate();
                    }
                }
                if (pdm != nullptr) {
                    pdm->info->deallocate();
                    pdm->deallocate();
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
            if (pdm != nullptr) {
                pdm->info->deallocate();
                pdm->deallocate();
            }
            if (skip_decomp)
                for (auto &mps : mpss)
                    mps->canonical_form[i] = forward ? 'S' : 'K';
            else
                for (auto &mps : mpss) {
                    if (forward) {
                        if (i != sweep_end_site - 1) {
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'S';
                        } else
                            mps->canonical_form[i] = 'K';
                    } else {
                        if (i > sweep_start_site) {
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
        if ((lme != nullptr &&
             eq_type != EquationTypes::PerturbativeCompression) ||
            eq_type == EquationTypes::FitAddition) {
            if (real_bra != nullptr)
                real_bra->deallocate();
            right_bra->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(targets, bra_error, bra_mmps, get<1>(pdi).first,
                         get<1>(pdi).second, get<2>(pdi), get<3>(pdi));
    }
    Iteration update_two_dot(int i, bool forward, ubond_t bra_bond_dim,
                             ubond_t ket_bond_dim, FPS noise,
                             FPS linear_conv_thrd) {
        const shared_ptr<MovingEnvironment<S, FL, FLS>> &me = rme;
        assert(me->bra != me->ket);
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MPS<S, FLS>>> mpss = {me->bra, me->ket};
        if (tme != nullptr) {
            if (tme->bra != me->bra && tme->bra != me->ket)
                mpss.push_back(tme->bra);
            if (tme->ket != me->bra && tme->ket != me->ket &&
                tme->ket != tme->bra)
                mpss.push_back(tme->ket);
        }
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_two_dot(i, mps);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<SparseMatrix<S, FLS>> right_bra = me->bra->tensors[i],
                                         real_bra = nullptr;
        shared_ptr<SparseMatrixGroup<S, FLS>> pbra = nullptr;
        shared_ptr<SparseMatrix<S, FLS>> pdm = nullptr;
        bool build_pdm = noise != 0 && (noise_type & NoiseTypes::Collected);
        if ((lme != nullptr &&
             eq_type != EquationTypes::PerturbativeCompression) ||
            eq_type == EquationTypes::FitAddition) {
            right_bra = make_shared<SparseMatrix<S, FLS>>();
            right_bra->allocate(me->bra->tensors[i]->info);
            if ((eq_type == EquationTypes::GreensFunction ||
                 eq_type == EquationTypes::GreensFunctionSquared) &&
                !is_same<FLS, FCS>::value) {
                real_bra = make_shared<SparseMatrix<S, FLS>>();
                real_bra->allocate(me->bra->tensors[i]->info);
            }
        }
        _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff = me->eff_ham(
            FuseTypes::FuseLR, forward, false, right_bra, me->ket->tensors[i]);
        teff += _t.get_time();
        tuple<FLS, pair<int, int>, size_t, double> pdi;
        auto mpdi = h_eff->multiply(me->mpo->const_e, me->para_rule);
        get<0>(pdi) = get<0>(mpdi);
        get<1>(pdi).first = get<1>(mpdi);
        get<2>(pdi) = get<2>(mpdi);
        get<3>(pdi) = get<3>(mpdi);
        tmult += _t.get_time();
        vector<FLS> targets = {get<0>(pdi)};
        vector<FLS> extra_bras;
        h_eff->deallocate();
        if (eq_type == EquationTypes::FitAddition ||
            eq_type == EquationTypes::PerturbativeCompression) {
            if (eq_type == EquationTypes::FitAddition) {
                shared_ptr<EffectiveHamiltonian<S, FL>> t_eff =
                    tme->eff_ham(FuseTypes::FuseLR, forward, false,
                                 tme->bra->tensors[i], tme->ket->tensors[i]);
                teff += _t.get_time();
                auto tpdi = t_eff->multiply(tme->mpo->const_e, tme->para_rule);
                GMatrix<FLS> mbra(me->bra->tensors[i]->data,
                                  (MKL_INT)me->bra->tensors[i]->total_memory,
                                  1);
                GMatrix<FLS> sbra(right_bra->data,
                                  (MKL_INT)right_bra->total_memory, 1);
                GMatrixFunctions<FLS>::iadd(mbra, sbra, 1);
                tmult += _t.get_time();
                targets[0] = GMatrixFunctions<FLS>::norm(mbra);
                get<1>(pdi).first += get<1>(tpdi);
                get<2>(pdi) += get<2>(tpdi);
                get<3>(pdi) += get<3>(tpdi);
                t_eff->deallocate();
            }
            if (lme != nullptr && noise != 0) {
                shared_ptr<EffectiveHamiltonian<S, FL>> l_eff =
                    lme->eff_ham(FuseTypes::FuseLR, forward, false,
                                 lme->bra->tensors[i], lme->ket->tensors[i]);
                teff += _t.get_time();
                if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
                    pbra = l_eff->perturbative_noise(
                        forward, i, i + 1, FuseTypes::FuseLR, me->bra->info,
                        noise_type, me->para_rule);
                tprt += _t.get_time();
                l_eff->deallocate();
            }
        } else if (lme != nullptr) {
            shared_ptr<EffectiveHamiltonian<S, FL>> l_eff = lme->eff_ham(
                FuseTypes::FuseLR, forward, linear_use_precondition,
                me->bra->tensors[i], right_bra);
            sweep_max_eff_ham_size =
                max(sweep_max_eff_ham_size, l_eff->op->get_total_memory());
            teff += _t.get_time();
            if (eq_type == EquationTypes::Normal) {
                tuple<FLS, pair<int, int>, size_t, double> lpdi;
                lpdi = l_eff->inverse_multiply(
                    lme->mpo->const_e, solver_type, linear_solver_params,
                    iprint >= 3, linear_conv_thrd, linear_max_iter,
                    linear_soft_max_iter, me->para_rule);
                targets[0] = get<0>(lpdi);
                get<1>(pdi).first += get<1>(lpdi).first;
                get<1>(pdi).second += get<1>(lpdi).second;
                get<2>(pdi) += get<2>(lpdi), get<3>(pdi) += get<3>(lpdi);
            } else if (eq_type == EquationTypes::GreensFunction ||
                       eq_type == EquationTypes::GreensFunctionSquared) {
                tuple<FCS, pair<int, int>, size_t, double> lpdi;
                if (gf_extra_omegas_at_site == i &&
                    gf_extra_omegas.size() != 0) {
                    gf_extra_targets.resize(gf_extra_omegas.size());
                    GMatrix<FLS> tmp(nullptr, (MKL_INT)l_eff->bra->total_memory,
                                     1);
                    tmp.allocate();
                    memcpy(tmp.data, l_eff->bra->data,
                           l_eff->bra->total_memory * sizeof(FLS));
                    if (tme != nullptr || ext_tmes.size() != 0)
                        extra_bras.reserve(l_eff->bra->total_memory *
                                           gf_extra_omegas.size() * 2);
                    for (size_t j = 0; j < gf_extra_omegas.size(); j++) {
                        if (eq_type == EquationTypes::GreensFunctionSquared)
                            lpdi = EffectiveFunctions<S, FL>::
                                greens_function_squared(
                                    l_eff, lme->mpo->const_e,
                                    gf_extra_omegas[j],
                                    gf_extra_eta == (FLS)0.0 ? gf_eta
                                                             : gf_extra_eta,
                                    real_bra, cg_n_harmonic_projection,
                                    iprint >= 3, linear_conv_thrd,
                                    linear_max_iter, linear_soft_max_iter,
                                    me->para_rule);
                        else
                            lpdi = EffectiveFunctions<S, FL>::greens_function(
                                l_eff, lme->mpo->const_e, solver_type,
                                gf_extra_omegas[j],
                                gf_extra_eta == (FLS)0.0 ? gf_eta
                                                         : gf_extra_eta,
                                real_bra, linear_solver_params, iprint >= 3,
                                linear_conv_thrd, linear_max_iter,
                                linear_soft_max_iter, me->para_rule);
                        if (tme != nullptr || ext_tmes.size() != 0) {
                            memcpy(extra_bras.data() +
                                       j * 2 * l_eff->bra->total_memory,
                                   l_eff->bra->data,
                                   l_eff->bra->total_memory * sizeof(FLS));
                            if (real_bra != nullptr)
                                memcpy(extra_bras.data() +
                                           (j * 2 + 1) *
                                               l_eff->bra->total_memory,
                                       real_bra->data,
                                       real_bra->total_memory * sizeof(FLS));
                        }
                        gf_extra_targets[j] =
                            is_same<FLS, FCS>::value
                                ? vector<FLS>{(FLS &)get<0>(lpdi)}
                                : vector<FLS>{xreal(get<0>(lpdi)),
                                              ximag(get<0>(lpdi))};
                        get<1>(pdi).first += get<1>(lpdi).first;
                        get<1>(pdi).second += get<1>(lpdi).second;
                        get<2>(pdi) += get<2>(lpdi),
                            get<3>(pdi) += get<3>(lpdi);
                    }
                    memcpy(l_eff->bra->data, tmp.data,
                           l_eff->bra->total_memory * sizeof(FLS));
                    tmp.deallocate();
                }
                if (eq_type == EquationTypes::GreensFunctionSquared)
                    lpdi = EffectiveFunctions<S, FL>::greens_function_squared(
                        l_eff, lme->mpo->const_e, gf_omega, gf_eta, real_bra,
                        cg_n_harmonic_projection, iprint >= 3, linear_conv_thrd,
                        linear_max_iter, linear_soft_max_iter, me->para_rule);
                else
                    lpdi = EffectiveFunctions<S, FL>::greens_function(
                        l_eff, lme->mpo->const_e, solver_type, gf_omega, gf_eta,
                        real_bra, linear_solver_params, iprint >= 3,
                        linear_conv_thrd, linear_max_iter, linear_soft_max_iter,
                        me->para_rule);
                targets =
                    is_same<FLS, FCS>::value
                        ? vector<FLS>{(FLS &)get<0>(lpdi)}
                        : vector<FLS>{xreal(get<0>(lpdi)), ximag(get<0>(lpdi))};
                get<1>(pdi).first += get<1>(lpdi).first;
                get<1>(pdi).second += get<1>(lpdi).second;
                get<2>(pdi) += get<2>(lpdi), get<3>(pdi) += get<3>(lpdi);
            } else
                assert(false);
            tmult += _t.get_time();
            if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
                pbra = l_eff->perturbative_noise(
                    forward, i, i + 1, FuseTypes::FuseLR, me->bra->info,
                    noise_type, me->para_rule);
            tprt += _t.get_time();
            l_eff->deallocate();
        }
        if (pbra != nullptr)
            sweep_max_pket_size = max(sweep_max_pket_size, pbra->total_memory);
        if (tme != nullptr && eq_type != EquationTypes::FitAddition) {
            shared_ptr<EffectiveHamiltonian<S, FL>> t_eff =
                tme->eff_ham(FuseTypes::FuseLR, forward, false,
                             tme->bra->tensors[i], tme->ket->tensors[i]);
            teff += _t.get_time();
            auto tpdi = t_eff->expect(tme->mpo->const_e, algo_type, ex_type,
                                      tme->para_rule);
            targets.clear();
            get<1>(pdi).first++;
            get<2>(pdi) += get<1>(tpdi);
            get<3>(pdi) += get<2>(tpdi);
            targets.push_back(get<0>(tpdi)[0].second);
            if (real_bra != nullptr) {
                if (tme->bra->tensors[i] == me->bra->tensors[i])
                    t_eff->bra = real_bra;
                if (tme->ket->tensors[i] == me->bra->tensors[i])
                    t_eff->ket = real_bra;
                tpdi = t_eff->expect(tme->mpo->const_e, algo_type, ex_type,
                                     tme->para_rule);
                targets.insert(targets.begin(), get<0>(tpdi)[0].second);
                get<1>(pdi).first++;
                get<2>(pdi) += get<1>(tpdi);
                get<3>(pdi) += get<2>(tpdi);
            }
            if (gf_extra_omegas_at_site == i && gf_extra_omegas.size() != 0)
                for (size_t j = 0; j < gf_extra_targets.size(); j++) {
                    FLS *tbra_bak = t_eff->bra->data;
                    FLS *tket_bak = t_eff->ket->data;
                    if (tme->bra->tensors[i] == me->bra->tensors[i])
                        t_eff->bra->data = extra_bras.data() +
                                           j * 2 * t_eff->bra->total_memory;
                    if (tme->ket->tensors[i] == me->bra->tensors[i])
                        t_eff->ket->data = extra_bras.data() +
                                           j * 2 * t_eff->ket->total_memory;
                    tpdi = t_eff->expect(tme->mpo->const_e, algo_type, ex_type,
                                         tme->para_rule);
                    get<1>(pdi).first++;
                    get<2>(pdi) += get<1>(tpdi);
                    get<3>(pdi) += get<2>(tpdi);
                    if (is_same<FLS, FCS>::value)
                        gf_extra_targets[j][0] = get<0>(tpdi)[0].second;
                    else {
                        gf_extra_targets[j][1] = get<0>(tpdi)[0].second;
                        if (tme->bra->tensors[i] == me->bra->tensors[i])
                            t_eff->bra->data =
                                extra_bras.data() +
                                (j * 2 + 1) * t_eff->bra->total_memory;
                        if (tme->ket->tensors[i] == me->bra->tensors[i])
                            t_eff->ket->data =
                                extra_bras.data() +
                                (j * 2 + 1) * t_eff->ket->total_memory;
                        tpdi = t_eff->expect(tme->mpo->const_e, algo_type,
                                             ex_type, tme->para_rule);
                        gf_extra_targets[j][0] = get<0>(tpdi)[0].second;
                        get<1>(pdi).first++;
                        get<2>(pdi) += get<1>(tpdi);
                        get<3>(pdi) += get<2>(tpdi);
                    }
                    t_eff->bra->data = tbra_bak;
                    t_eff->ket->data = tket_bak;
                }
            tmult += _t.get_time();
            t_eff->deallocate();
        }
        for (auto &mps : ext_mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_two_dot(i, mps);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (ext_target_at_site == i && ext_tmes.size() != 0) {
            ext_targets.resize(ext_tmes.size());
            if (gf_extra_omegas_at_site == i && gf_extra_omegas.size() != 0)
                gf_extra_ext_targets.resize(ext_tmes.size());
            for (size_t k = 0; k < ext_tmes.size(); k++) {
                shared_ptr<MovingEnvironment<S, FL, FLS>> xme = ext_tmes[k];
                shared_ptr<EffectiveHamiltonian<S, FL>> t_eff =
                    xme->eff_ham(FuseTypes::FuseLR, forward, false,
                                 xme->bra->tensors[i], xme->ket->tensors[i]);
                auto tpdi = t_eff->expect(xme->mpo->const_e, algo_type, ex_type,
                                          xme->para_rule);
                ext_targets[k].resize(2);
                ext_targets[k][1] = get<0>(tpdi)[0].second;
                get<1>(pdi).first++;
                get<2>(pdi) += get<1>(tpdi);
                get<3>(pdi) += get<2>(tpdi);
                if (real_bra != nullptr) {
                    if (xme->bra->tensors[i] == me->bra->tensors[i])
                        t_eff->bra = real_bra;
                    if (xme->ket->tensors[i] == me->bra->tensors[i])
                        t_eff->ket = real_bra;
                }
                tpdi = t_eff->expect(xme->mpo->const_e, algo_type, ex_type,
                                     xme->para_rule);
                ext_targets[k][0] = get<0>(tpdi)[0].second;
                get<1>(pdi).first++;
                get<2>(pdi) += get<1>(tpdi);
                get<3>(pdi) += get<2>(tpdi);
                if (gf_extra_omegas_at_site == i &&
                    gf_extra_omegas.size() != 0) {
                    gf_extra_ext_targets[k].resize(gf_extra_omegas.size());
                    for (size_t j = 0; j < gf_extra_omegas.size(); j++) {
                        FLS *tbra_bak = t_eff->bra->data;
                        FLS *tket_bak = t_eff->ket->data;
                        if (xme->bra->tensors[i] == me->bra->tensors[i])
                            t_eff->bra->data = extra_bras.data() +
                                               j * 2 * t_eff->bra->total_memory;
                        if (xme->ket->tensors[i] == me->bra->tensors[i])
                            t_eff->ket->data = extra_bras.data() +
                                               j * 2 * t_eff->ket->total_memory;
                        tpdi = t_eff->expect(xme->mpo->const_e, algo_type,
                                             ex_type, xme->para_rule);
                        get<1>(pdi).first++;
                        get<2>(pdi) += get<1>(tpdi);
                        get<3>(pdi) += get<2>(tpdi);
                        if (is_same<FLS, FCS>::value) {
                            gf_extra_ext_targets[k][j].resize(1);
                            gf_extra_ext_targets[k][j][0] =
                                get<0>(tpdi)[0].second;
                        } else {
                            gf_extra_ext_targets[k][j].resize(2);
                            gf_extra_ext_targets[k][j][1] =
                                get<0>(tpdi)[0].second;
                            if (xme->bra->tensors[i] == me->bra->tensors[i])
                                t_eff->bra->data =
                                    extra_bras.data() +
                                    (j * 2 + 1) * t_eff->bra->total_memory;
                            if (xme->ket->tensors[i] == me->bra->tensors[i])
                                t_eff->ket->data =
                                    extra_bras.data() +
                                    (j * 2 + 1) * t_eff->ket->total_memory;
                            tpdi = t_eff->expect(xme->mpo->const_e, algo_type,
                                                 ex_type, xme->para_rule);
                            gf_extra_ext_targets[k][j][0] =
                                get<0>(tpdi)[0].second;
                            get<1>(pdi).first++;
                            get<2>(pdi) += get<1>(tpdi);
                            get<3>(pdi) += get<2>(tpdi);
                        }
                        t_eff->bra->data = tbra_bak;
                        t_eff->ket->data = tket_bak;
                    }
                }
                tmult += _t.get_time();
                t_eff->deallocate();
            }
        }
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
                if (mps->center == sweep_end_site - 1)
                    mps->center = sweep_end_site - 2;
            } else {
                mps->canonical_form[i] = 'C';
                mps->move_left(me->mpo->tf->opf->cg, me->para_rule);
                mps->canonical_form[i] = 'C';
            }
            if (me->para_rule == nullptr || me->para_rule->is_root())
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, sweep_start_site, sweep_end_site, mps, forward,
                    me->mpo->tf->opf->cg);
        }
        if (build_pdm) {
            _t.get_time();
            assert(decomp_type == DecompositionTypes::DensityMatrix);
            pdm = MovingEnvironment<S, FL, FLS>::density_matrix(
                me->bra->info->vacuum, me->bra->tensors[i], forward,
                me->para_rule != nullptr ? noise / me->para_rule->comm->size
                                         : noise,
                noise_type, 0.0, pbra);
            if (me->para_rule != nullptr)
                me->para_rule->comm->reduce_sum(pdm, me->para_rule->comm->root);
            tdm += _t.get_time();
        }
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
                bool store_wfn_spectra =
                    mps == me->bra ? store_bra_spectra : store_ket_spectra;
                int bond_dim = -1;
                FPS error;
                if (mps == me->bra)
                    bond_dim = (int)bra_bond_dim;
                else if (mps == me->ket)
                    bond_dim = (int)ket_bond_dim;
                else if (tme != nullptr && mps == tme->bra)
                    bond_dim = target_bra_bond_dim;
                else if (tme != nullptr && mps == tme->ket)
                    bond_dim = target_ket_bond_dim;
                else
                    assert(false);
                assert(right_weight >= 0 && right_weight <= 1);
                if (decomp_type == DecompositionTypes::DensityMatrix) {
                    _t.get_time();
                    if (mps != me->bra) {
                        dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    } else {
                        FPS weight = 1 - right_weight;
                        if (real_bra != nullptr)
                            weight *= complex_weights[1];
                        dm = MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward,
                            build_pdm ? 0.0 : noise, noise_type, weight, pbra);
                        if (build_pdm)
                            GMatrixFunctions<FLS>::iadd(
                                GMatrix<FLS>(dm->data,
                                             (MKL_INT)dm->total_memory, 1),
                                GMatrix<FLS>(pdm->data,
                                             (MKL_INT)pdm->total_memory, 1),
                                1.0);
                        if (real_bra != nullptr) {
                            weight = complex_weights[0] * (1 - right_weight);
                            MovingEnvironment<
                                S, FL, FLS>::density_matrix_add_wfn(dm,
                                                                    real_bra,
                                                                    forward,
                                                                    weight);
                        }
                        if (right_weight != 0)
                            MovingEnvironment<S, FL, FLS>::
                                density_matrix_add_wfn(dm, right_bra, forward,
                                                       right_weight);
                    }
                    tdm += _t.get_time();
                    error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                        dm, old_wfn, bond_dim, forward, false, mps->tensors[i],
                        mps->tensors[i + 1], cutoff, store_wfn_spectra,
                        wfn_spectra, trunc_type);
                    tsplt += _t.get_time();
                } else if (decomp_type == DecompositionTypes::SVD ||
                           decomp_type == DecompositionTypes::PureSVD) {
                    if (mps != me->bra) {
                        error = MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                mps->info->vacuum, old_wfn, bond_dim, forward,
                                false, mps->tensors[i], mps->tensors[i + 1],
                                cutoff, store_wfn_spectra, wfn_spectra,
                                trunc_type, decomp_type, nullptr);
                    } else {
                        if (noise != 0 && mps == me->bra) {
                            if (noise_type & NoiseTypes::Wavefunction)
                                MovingEnvironment<
                                    S, FL, FLS>::wavefunction_add_noise(old_wfn,
                                                                        noise);
                            else if (noise_type & NoiseTypes::Perturbative)
                                MovingEnvironment<S, FL, FLS>::
                                    scale_perturbative_noise(noise, noise_type,
                                                             pbra);
                        }
                        vector<FPS> weights = {1};
                        vector<shared_ptr<SparseMatrix<S, FLS>>> xwfns = {};
                        if (real_bra != nullptr) {
                            weights = vector<FPS>{sqrt(complex_weights[1]),
                                                  sqrt(complex_weights[0])};
                            xwfns.push_back(real_bra);
                        }
                        if (right_weight != 0) {
                            for (auto w : weights)
                                w = sqrt(w * w * (1 - right_weight));
                            weights.push_back(sqrt(right_weight));
                            xwfns.push_back(right_bra);
                        }
                        _t.get_time();
                        error = MovingEnvironment<S, FL, FLS>::
                            split_wavefunction_svd(
                                mps->info->vacuum, old_wfn, bond_dim, forward,
                                false, mps->tensors[i], mps->tensors[i + 1],
                                cutoff, store_wfn_spectra, wfn_spectra,
                                trunc_type, decomp_type, pbra, xwfns, weights);
                        tsvd += _t.get_time();
                    }
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
                if (mps == me->bra) {
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
                MovingEnvironment<S, FL, FLS>::propagate_wfn(
                    i, sweep_start_site, sweep_end_site, mps, forward,
                    me->mpo->tf->opf->cg);
            }
            for (auto &mps : mpss)
                mps->save_data();
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
        if (pdm != nullptr) {
            pdm->info->deallocate();
            pdm->deallocate();
        }
        if (pbra != nullptr) {
            pbra->deallocate();
            pbra->deallocate_infos();
        }
        if ((lme != nullptr &&
             eq_type != EquationTypes::PerturbativeCompression) ||
            eq_type == EquationTypes::FitAddition) {
            if (real_bra != nullptr)
                real_bra->deallocate();
            right_bra->deallocate();
        }
        for (auto &old_wfn : vector<shared_ptr<SparseMatrix<S, FLS>>>(
                 old_wfns.rbegin(), old_wfns.rend())) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(targets, bra_error, bra_mmps, get<1>(pdi).first,
                         get<1>(pdi).second, get<2>(pdi), get<3>(pdi));
    }
    virtual Iteration blocking(int i, bool forward, ubond_t bra_bond_dim,
                               ubond_t ket_bond_dim, FPS noise,
                               FPS linear_conv_thrd) {
        _t2.get_time();
        rme->move_to(i);
        if (lme != nullptr)
            lme->move_to(i);
        if (tme != nullptr)
            tme->move_to(i);
        for (auto &xme : ext_tmes)
            xme->move_to(i);
        tmve += _t2.get_time();
        Iteration it(vector<FLS>(), 0, 0, 0, 0);
        if (rme->dot == 2)
            it = update_two_dot(i, forward, bra_bond_dim, ket_bond_dim, noise,
                                linear_conv_thrd);
        else
            it = update_one_dot(i, forward, bra_bond_dim, ket_bond_dim, noise,
                                linear_conv_thrd);
        if (store_bra_spectra || store_ket_spectra) {
            const int bond_update_idx = rme->dot == 1 && !forward ? i - 1 : i;
            if (bond_update_idx >= 0) {
                if (sweep_wfn_spectra.size() <= bond_update_idx)
                    sweep_wfn_spectra.resize(bond_update_idx + 1);
                sweep_wfn_spectra[bond_update_idx] = wfn_spectra;
            }
        }
        tblk += _t2.get_time();
        return it;
    }
    tuple<vector<FLS>, FPS> sweep(bool forward, ubond_t bra_bond_dim,
                                  ubond_t ket_bond_dim, FPS noise,
                                  FPS linear_conv_thrd) {
        teff = tmult = tprt = tblk = tmve = tdm = tsplt = tsvd = 0;
        frame_<FPS>()->twrite = frame_<FPS>()->tread = frame_<FPS>()->tasync =
            0;
        frame_<FPS>()->fpwrite = frame_<FPS>()->fpread = 0;
        if (frame_<FPS>()->fp_codec != nullptr)
            frame_<FPS>()->fp_codec->ndata = frame_<FPS>()->fp_codec->ncpsd = 0;
        if (lme != nullptr && lme->para_rule != nullptr) {
            lme->para_rule->comm->tcomm = 0;
            lme->para_rule->comm->tidle = 0;
            lme->para_rule->comm->twait = 0;
        }
        rme->prepare(sweep_start_site, sweep_end_site);
        if (lme != nullptr)
            lme->prepare(sweep_start_site, sweep_end_site);
        if (tme != nullptr)
            tme->prepare(sweep_start_site, sweep_end_site);
        for (auto &xme : ext_tmes)
            xme->prepare(sweep_start_site, sweep_end_site);
        sweep_targets.clear();
        sweep_discarded_weights.clear();
        sweep_cumulative_nflop = 0;
        sweep_max_pket_size = 0;
        sweep_max_eff_ham_size = 0;
        frame_<FPS>()->reset_peak_used_memory();
        vector<int> sweep_range;
        if (forward)
            for (int it = rme->center; it < sweep_end_site - rme->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = rme->center; it >= sweep_start_site; it--)
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
            Iteration r = blocking(i, forward, bra_bond_dim, ket_bond_dim,
                                   noise, linear_conv_thrd);
            sweep_cumulative_nflop += r.nflop;
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            sweep_targets.push_back(r.targets);
            sweep_discarded_weights.push_back(r.error);
            if (frame_<FPS>()->restart_dir_optimal_mps != "" ||
                frame_<FPS>()->restart_dir_optimal_mps_per_sweep != "") {
                size_t midx = -1;
                switch (conv_type) {
                case ConvergenceTypes::MiddleSite:
                    midx = sweep_targets.size() / 2;
                    break;
                case ConvergenceTypes::LastMinimal:
                    midx = min_element(
                               sweep_targets.begin(), sweep_targets.end(),
                               [](const vector<FLS> &x, const vector<FLS> &y) {
                                   return ximag<FLS>(x.back()) !=
                                                  ximag<FLS>(y.back())
                                              ? ximag<FLS>(x.back()) <
                                                    ximag<FLS>(y.back())
                                              : xreal<FLS>(x.back()) <
                                                    xreal<FLS>(y.back());
                               }) -
                           sweep_targets.begin();
                    break;
                case ConvergenceTypes::LastMaximal:
                    midx = min_element(
                               sweep_targets.begin(), sweep_targets.end(),
                               [](const vector<FLS> &x, const vector<FLS> &y) {
                                   return ximag<FLS>(x.back()) !=
                                                  ximag<FLS>(y.back())
                                              ? ximag<FLS>(x.back()) >
                                                    ximag<FLS>(y.back())
                                              : xreal<FLS>(x.back()) >
                                                    xreal<FLS>(y.back());
                               }) -
                           sweep_targets.begin();
                    break;
                case ConvergenceTypes::FirstMinimal:
                    midx =
                        min_element(
                            sweep_targets.begin(), sweep_targets.end(),
                            [](const vector<FLS> &x, const vector<FLS> &y) {
                                return xreal<FLS>(x[0]) != xreal<FLS>(y[0])
                                           ? xreal<FLS>(x[0]) < xreal<FLS>(y[0])
                                           : ximag<FLS>(x[0]) <
                                                 ximag<FLS>(y[0]);
                            }) -
                        sweep_targets.begin();
                    break;
                case ConvergenceTypes::FirstMaximal:
                    midx =
                        min_element(
                            sweep_targets.begin(), sweep_targets.end(),
                            [](const vector<FLS> &x, const vector<FLS> &y) {
                                return xreal<FLS>(x[0]) != xreal<FLS>(y[0])
                                           ? xreal<FLS>(x[0]) > xreal<FLS>(y[0])
                                           : ximag<FLS>(x[0]) >
                                                 ximag<FLS>(y[0]);
                            }) -
                        sweep_targets.begin();
                    break;
                default:
                    assert(false);
                }
                if (midx == sweep_targets.size() - 1) {
                    if (rme->para_rule == nullptr ||
                        rme->para_rule->is_root()) {
                        if (frame_<FPS>()->restart_dir_optimal_mps != "") {
                            string rdoe =
                                frame_<FPS>()->restart_dir_optimal_mps;
                            if (!Parsing::path_exists(rdoe))
                                Parsing::mkdir(rdoe);
                            rme->bra->info->copy_mutable(rdoe);
                            rme->bra->copy_data(rdoe);
                        }
                        if (frame_<FPS>()->restart_dir_optimal_mps_per_sweep !=
                            "") {
                            string rdps =
                                frame_<FPS>()
                                    ->restart_dir_optimal_mps_per_sweep +
                                "." + Parsing::to_string((int)targets.size());
                            if (!Parsing::path_exists(rdps))
                                Parsing::mkdir(rdps);
                            rme->bra->info->copy_mutable(rdps);
                            rme->bra->copy_data(rdps);
                        }
                    }
                    if (rme->para_rule != nullptr)
                        rme->para_rule->comm->barrier();
                }
            }
        }
        size_t idx = -1;
        switch (conv_type) {
        case ConvergenceTypes::MiddleSite:
            idx = sweep_targets.size() / 2;
            break;
        case ConvergenceTypes::LastMinimal:
            idx = min_element(sweep_targets.begin(), sweep_targets.end(),
                              [](const vector<FLS> &x, const vector<FLS> &y) {
                                  return ximag<FLS>(x.back()) !=
                                                 ximag<FLS>(y.back())
                                             ? ximag<FLS>(x.back()) <
                                                   ximag<FLS>(y.back())
                                             : xreal<FLS>(x.back()) <
                                                   xreal<FLS>(y.back());
                              }) -
                  sweep_targets.begin();
            break;
        case ConvergenceTypes::LastMaximal:
            idx = min_element(sweep_targets.begin(), sweep_targets.end(),
                              [](const vector<FLS> &x, const vector<FLS> &y) {
                                  return ximag<FLS>(x.back()) !=
                                                 ximag<FLS>(y.back())
                                             ? ximag<FLS>(x.back()) >
                                                   ximag<FLS>(y.back())
                                             : xreal<FLS>(x.back()) >
                                                   xreal<FLS>(y.back());
                              }) -
                  sweep_targets.begin();
            break;
        case ConvergenceTypes::FirstMinimal:
            idx = min_element(
                      sweep_targets.begin(), sweep_targets.end(),
                      [](const vector<FLS> &x, const vector<FLS> &y) {
                          return xreal<FLS>(x[0]) != xreal<FLS>(y[0])
                                     ? xreal<FLS>(x[0]) < xreal<FLS>(y[0])
                                     : ximag<FLS>(x[0]) < ximag<FLS>(y[0]);
                      }) -
                  sweep_targets.begin();
            break;
        case ConvergenceTypes::FirstMaximal:
            idx = min_element(
                      sweep_targets.begin(), sweep_targets.end(),
                      [](const vector<FLS> &x, const vector<FLS> &y) {
                          return xreal<FLS>(x[0]) != xreal<FLS>(y[0])
                                     ? xreal<FLS>(x[0]) > xreal<FLS>(y[0])
                                     : ximag<FLS>(x[0]) > ximag<FLS>(y[0]);
                      }) -
                  sweep_targets.begin();
            break;
        default:
            assert(false);
        }
        if (frame_<FPS>()->restart_dir != "" &&
            (rme->para_rule == nullptr || rme->para_rule->is_root())) {
            if (!Parsing::path_exists(frame_<FPS>()->restart_dir))
                Parsing::mkdir(frame_<FPS>()->restart_dir);
            rme->bra->info->copy_mutable(frame_<FPS>()->restart_dir);
            rme->bra->copy_data(frame_<FPS>()->restart_dir);
        }
        if (frame_<FPS>()->restart_dir_per_sweep != "" &&
            (rme->para_rule == nullptr || rme->para_rule->is_root())) {
            string rdps = frame_<FPS>()->restart_dir_per_sweep + "." +
                          Parsing::to_string((int)targets.size());
            if (!Parsing::path_exists(rdps))
                Parsing::mkdir(rdps);
            rme->bra->info->copy_mutable(rdps);
            rme->bra->copy_data(rdps);
        }
        FPS max_dw = *max_element(sweep_discarded_weights.begin(),
                                  sweep_discarded_weights.end());
        return make_tuple(sweep_targets[idx], max_dw);
    }
    FLS solve(int n_sweeps, bool forward = true, FPS tol = 1E-6) {
        if (bra_bond_dims.size() < n_sweeps)
            bra_bond_dims.resize(n_sweeps, bra_bond_dims.back());
        if (ket_bond_dims.size() < n_sweeps)
            ket_bond_dims.resize(n_sweeps, ket_bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.size() == 0 ? 0.0 : noises.back());
        if (linear_conv_thrds.size() < n_sweeps)
            for (size_t i = linear_conv_thrds.size(); i < noises.size(); i++)
                linear_conv_thrds.push_back(
                    (noises[i] == 0 ? (tol == 0 ? 1E-9 : tol) : noises[i]) *
                    0.1);
        Timer start, current;
        start.get_time();
        current.get_time();
        targets.clear();
        discarded_weights.clear();
        bool converged;
        FLS target_difference;
        if (iprint >= 1)
            cout << endl;
        for (int iw = 0; iw < n_sweeps; iw++) {
            if (iprint >= 1) {
                cout << "Sweep = " << setw(4) << iw
                     << " | Direction = " << setw(8)
                     << (forward ? "forward" : "backward")
                     << " | BRA bond dimension = " << setw(4)
                     << (uint32_t)bra_bond_dims[iw]
                     << " | Noise = " << scientific << setw(9)
                     << setprecision(2) << noises[iw];
                if (lme != nullptr)
                    cout << " | Linear threshold = " << scientific << setw(9)
                         << setprecision(2) << linear_conv_thrds[iw];
                cout << endl;
            }
            auto sweep_results =
                sweep(forward, bra_bond_dims[iw], ket_bond_dims[iw], noises[iw],
                      linear_conv_thrds[iw]);
            targets.push_back(get<0>(sweep_results));
            discarded_weights.push_back(get<1>(sweep_results));
            if (targets.size() >= 2)
                target_difference = targets[targets.size() - 1].back() -
                                    targets[targets.size() - 2].back();
            converged = targets.size() >= 2 && tol > 0 &&
                        abs(target_difference) < tol &&
                        noises[iw] == noises.back() &&
                        bra_bond_dims[iw] == bra_bond_dims.back();
            for (int iconv = 1; iconv < conv_required_sweeps && converged;
                 iconv++)
                converged =
                    converged && (int)targets.size() >= 2 + iconv &&
                    abs(targets[targets.size() - 1 - iconv].back() -
                        targets[targets.size() - 2 - iconv].back()) < tol;
            forward = !forward;
            double tswp = current.get_time();
            if (iprint >= 1) {
                cout << "Time elapsed = " << fixed << setw(10)
                     << setprecision(3) << current.current - start.current;
                if (get<0>(sweep_results).size() == 1) {
                    cout << (abs(get<0>(sweep_results)[0]) > 1E-3 ? fixed
                                                                  : scientific);
                    cout << (abs(get<0>(sweep_results)[0]) > 1E-3
                                 ? setprecision(10)
                                 : setprecision(7));
                    cout << " | F = " << setw(15) << get<0>(sweep_results)[0];
                } else {
                    cout << " | F[" << setw(3) << get<0>(sweep_results).size()
                         << "] = ";
                    for (FLS x : get<0>(sweep_results)) {
                        cout << (abs(x) > 1E-3 ? fixed : scientific);
                        cout << (abs(x) > 1E-3 ? setprecision(10)
                                               : setprecision(7));
                        cout << setw(15) << x;
                    }
                }
                if (targets.size() >= 2)
                    cout << " | DF = " << setw(6) << setprecision(2)
                         << scientific << target_difference;
                cout << " | DW = " << setw(6) << setprecision(2) << scientific
                     << get<1>(sweep_results) << endl;
                if (iprint >= 2) {
                    cout << fixed << setprecision(3);
                    cout << "Time sweep = " << setw(12) << tswp;
                    cout << " | "
                         << Parsing::to_size_string(sweep_cumulative_nflop,
                                                    "FLOP/SWP")
                         << endl;
                    if (lme != nullptr && lme->para_rule != nullptr) {
                        double tt[3] = {lme->para_rule->comm->tcomm,
                                        lme->para_rule->comm->tidle,
                                        lme->para_rule->comm->twait};
                        lme->para_rule->comm->reduce_sum_optional(
                            &tt[0], 3, lme->para_rule->comm->root);
                        tt[0] /= lme->para_rule->comm->size;
                        tt[1] /= lme->para_rule->comm->size;
                        tt[2] /= lme->para_rule->comm->size;
                        cout << " | Tcomm = " << tt[0] << " | Tidle = " << tt[1]
                             << " | Twait = " << tt[2];
                        cout << endl;
                    }
                    size_t dmain = frame_<FPS>()->peak_used_memory[0];
                    size_t dseco = frame_<FPS>()->peak_used_memory[1];
                    size_t imain = frame_<FPS>()->peak_used_memory[2];
                    size_t iseco = frame_<FPS>()->peak_used_memory[3];
                    cout << " | Dmem = "
                         << Parsing::to_size_string(dmain + dseco) << " ("
                         << (dmain * 100 / (dmain + dseco)) << "%)";
                    cout << " | Imem = "
                         << Parsing::to_size_string(imain + iseco) << " ("
                         << (imain * 100 / (imain + iseco)) << "%)";
                    cout << " | Hmem = "
                         << Parsing::to_size_string(sweep_max_eff_ham_size *
                                                    sizeof(FL));
                    cout << " | Pmem = "
                         << Parsing::to_size_string(sweep_max_pket_size *
                                                    sizeof(FLS))
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
                    if (lme != nullptr)
                        cout << " | Trot = " << lme->trot
                             << " | Tctr = " << lme->tctr
                             << " | Tint = " << lme->tint
                             << " | Tmid = " << lme->tmid
                             << " | Tdctr = " << lme->tdctr
                             << " | Tdiag = " << lme->tdiag
                             << " | Tinfo = " << lme->tinfo << endl;
                    cout << " | Teff = " << teff << " | Tprt = " << tprt
                         << " | Tmult = " << tmult << " | Tblk = " << tblk
                         << " | Tmve = " << tmve << " | Tdm = " << tdm
                         << " | Tsplt = " << tsplt << " | Tsvd = " << tsvd
                         << endl;
                }
                cout << endl;
            }
            if (converged || has_abort_file())
                break;
        }
        this->forward = forward;
        if (!converged && iprint > 0 && tol != 0)
            cout << "ATTENTION: Linear is not converged to desired tolerance "
                    "of "
                 << scientific << tol << endl;
        return targets.back()[0];
    }
};

template <typename FL, typename = void> struct PartitionWeights;

template <typename FL>
struct PartitionWeights<
    FL, typename enable_if<is_floating_point<FL>::value>::type> {
    typedef typename GMatrix<FL>::FL LFL;
    typedef vector<LFL> type;
    inline static vector<LFL> get_partition_weights() {
        return vector<LFL>{(LFL)1.0};
    }
    inline static vector<LFL>
    get_partition_weights(FL beta, const vector<FL> &energies,
                          const vector<int> &multiplicities) {
        vector<LFL> partition_weights(energies.size());
        for (size_t i = 0; i < energies.size(); i++)
            partition_weights[i] =
                multiplicities[i] *
                exp(-(LFL)beta * ((LFL)energies[i] - (LFL)energies[0]));
        LFL psum = accumulate(partition_weights.begin(),
                              partition_weights.end(), (LFL)0.0);
        for (size_t i = 0; i < energies.size(); i++)
            partition_weights[i] /= psum;
        return partition_weights;
    }
    inline static ExpectationTypes get_type() { return ExpectationTypes::Real; }
};

template <typename FL>
struct PartitionWeights<FL, typename enable_if<is_complex<FL>::value>::type> {
    typedef typename GMatrix<FL>::FP FP;
    typedef vector<FL> type;
    inline static vector<FL> get_partition_weights() {
        return vector<FL>{FL(1, 0), FL(0, 1)};
    }
    inline static vector<FL>
    get_partition_weights(FP beta, const vector<FP> &energies,
                          const vector<int> &multiplicities) {
        throw runtime_error("Not implemented!");
        return vector<FL>{FL(1, 0), FL(0, 1)};
    }
    inline static ExpectationTypes get_type() {
        return ExpectationTypes::Complex;
    }
};

// Expectation value
template <typename S, typename FL, typename FLS, typename FLX = double>
struct Expect {
    typedef typename MovingEnvironment<S, FL, FLS>::FPS FPS;
    shared_ptr<MovingEnvironment<S, FL, FLS>> me;
    ubond_t bra_bond_dim, ket_bond_dim;
    vector<vector<pair<shared_ptr<OpExpr<S>>, FLX>>> expectations;
    bool forward;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    ExpectationAlgorithmTypes algo_type = ExpectationAlgorithmTypes::Automatic;
    // expeditious zero-dot algorithm for fast one-dot pdm expectation
    // only works for non-multi-mps and dot = 1
    bool zero_dot_algo = false;
    ExpectationTypes ex_type = PartitionWeights<FLX>::get_type();
    uint8_t iprint = 2;
    FPS cutoff = 0.0;
    FPS beta = 0.0;
    // partition function (for thermal-averaged MultiMPS)
    typename PartitionWeights<FLX>::type partition_weights;
    // store all wfn singular values (for analysis) at each site
    bool store_bra_spectra = false, store_ket_spectra = false;
    vector<vector<FPS>> sweep_wfn_spectra;
    vector<FPS> wfn_spectra;
    size_t sweep_cumulative_nflop = 0;
    size_t sweep_max_eff_ham_size = 0;
    pair<size_t, size_t> max_move_env_mem;
    double tex = 0, teff = 0, tmve = 0, tblk = 0;
    Timer _t, _t2;
    Expect(const shared_ptr<MovingEnvironment<S, FL, FLS>> &me,
           ubond_t bra_bond_dim, ubond_t ket_bond_dim)
        : me(me), bra_bond_dim(bra_bond_dim), ket_bond_dim(ket_bond_dim),
          forward(false) {
        expectations.resize(me->n_sites - me->dot + 1);
        // for final step of serial execution of parallel npdm
        if (expectations.size() != 0 && me->mpo->npdm_scheme != nullptr)
            expectations[0].push_back(
                make_pair(make_shared<OpCounter<S>>(0), (FLX)0.0));
        partition_weights = PartitionWeights<FLX>::get_partition_weights();
    }
    Expect(const shared_ptr<MovingEnvironment<S, FL, FLS>> &me,
           ubond_t bra_bond_dim, ubond_t ket_bond_dim, FPS beta,
           const vector<FPS> &energies, const vector<int> &multiplicities)
        : Expect(me, bra_bond_dim, ket_bond_dim) {
        this->beta = beta;
        partition_weights = PartitionWeights<FLX>::get_partition_weights(
            beta, energies, multiplicities);
    }
    struct Iteration {
        vector<pair<shared_ptr<OpExpr<S>>, FLX>> expectations;
        FPS bra_error, ket_error;
        double tmult;
        size_t nflop;
        Iteration(const vector<pair<shared_ptr<OpExpr<S>>, FLX>> &expectations,
                  FPS bra_error, FPS ket_error, size_t nflop = 0,
                  double tmult = 1.0)
            : expectations(expectations), bra_error(bra_error),
              ket_error(ket_error), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            if (r.expectations.size() == 1 &&
                r.expectations[0].first->get_type() == OpTypes::Counter)
                os << "Nterms = " << setw(12)
                   << dynamic_pointer_cast<OpCounter<S>>(
                          r.expectations[0].first)
                          ->data;
            else if (r.expectations.size() == 1)
                os << setw(14) << r.expectations[0].second;
            else
                os << "Nterms = " << setw(12) << r.expectations.size();
            os << " Error = " << setw(15) << setprecision(12) << r.bra_error
               << "/" << setw(15) << setprecision(12) << r.ket_error
               << " FLOPS = " << scientific << setw(8) << setprecision(2)
               << (r.tmult == 0.0 ? 0 : (double)r.nflop / r.tmult)
               << " Tmult = " << fixed << setprecision(2) << r.tmult;
            return os;
        }
    };
    // expeditious zero-dot algorithm for fast one-dot pdm expectation
    Iteration update_zero_dot(int i, bool forward, bool propagate,
                              ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        frame_<FPS>()->activate(0);
        assert(propagate);
        vector<shared_ptr<MPS<S, FLS>>> mpss =
            me->bra == me->ket
                ? vector<shared_ptr<MPS<S, FLS>>>{me->bra}
                : vector<shared_ptr<MPS<S, FLS>>>{me->bra, me->ket};
        for (auto &mps : mpss) {
            if (mps->canonical_form[i] == 'C') {
                if (i == 0)
                    mps->canonical_form[i] = 'K';
                else if (i == me->n_sites - 1)
                    mps->canonical_form[i] = 'S';
                else
                    assert(false);
            }
            mps->load_tensor(i);
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        tuple<vector<pair<shared_ptr<OpExpr<S>>, FL>>, size_t, double> pdi;
        FPS bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            for (auto &mps : mpss) {
                bool mps_fuse_left = mps->canonical_form[i] == 'K';
                if (mps_fuse_left != forward) {
                    shared_ptr<SparseMatrix<S, FLS>> prev_wfn = mps->tensors[i];
                    if (!mps_fuse_left && forward)
                        mps->tensors[i] = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_left(i, mps->info, prev_wfn,
                                                   me->mpo->tf->opf->cg);
                    else if (mps_fuse_left && !forward)
                        mps->tensors[i] = MovingEnvironment<S, FL, FLS>::
                            swap_wfn_to_fused_right(i, mps->info, prev_wfn,
                                                    me->mpo->tf->opf->cg);
                    prev_wfn->info->deallocate();
                    prev_wfn->deallocate();
                }
            }
        }
        vector<shared_ptr<SparseMatrix<S, FLS>>> old_wfns =
            me->bra == me->ket
                ? vector<shared_ptr<SparseMatrix<S, FLS>>>{me->bra->tensors[i]}
                : vector<shared_ptr<SparseMatrix<S, FLS>>>{me->ket->tensors[i],
                                                           me->bra->tensors[i]};
        vector<shared_ptr<SparseMatrix<S, FLS>>> lefts, rights, dms;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (auto &mps : mpss) {
                shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i];
                shared_ptr<SparseMatrix<S, FLS>> left, right;
                shared_ptr<SparseMatrix<S, FLS>> dm =
                    MovingEnvironment<S, FL, FLS>::density_matrix(
                        mps->info->vacuum, old_wfn, forward, 0.0,
                        NoiseTypes::None);
                int bond_dim =
                    mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                bool store_wfn_spectra =
                    mps == me->bra ? store_bra_spectra : store_ket_spectra;
                FPS error = MovingEnvironment<S, FL, FLS>::split_density_matrix(
                    dm, old_wfn, bond_dim, forward, false, left, right, cutoff,
                    store_wfn_spectra, wfn_spectra, trunc_type);
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
                } else {
                    mps->tensors[i] = right;
                    mps->save_tensor(i);
                    info = right->info->extract_state_info(forward);
                    mps->info->right_dims[i] = info;
                    mps->info->save_right_dims(i);
                }
                info->deallocate();
                lefts.push_back(left);
                rights.push_back(right);
                dms.push_back(dm);
            }
        }
        if ((forward && i != me->n_sites - 1) || (!forward && i != 0)) {
            if (me->para_rule != nullptr) {
                for (int ip = 0; ip < (int)mpss.size(); ip++) {
                    auto &mps = mpss[ip];
                    if (me->para_rule->is_root()) {
                        auto &left = lefts[ip];
                        auto &right = rights[ip];
                        if (forward)
                            right->save_data(mps->get_filename(-2), true);
                        else
                            left->save_data(mps->get_filename(-2), true);
                    }
                    me->para_rule->comm->barrier();
                    if (!me->para_rule->is_root()) {
                        if (forward) {
                            shared_ptr<SparseMatrix<S, FLS>> right =
                                make_shared<SparseMatrix<S, FLS>>();
                            right->load_data(mps->get_filename(-2), true);
                            rights.push_back(right);
                        } else {
                            shared_ptr<SparseMatrix<S, FLS>> left =
                                make_shared<SparseMatrix<S, FLS>>();
                            left->load_data(mps->get_filename(-2), true);
                            lefts.push_back(left);
                        }
                    }
                }
                me->para_rule->comm->barrier();
            }
            pair<size_t, size_t> pbr;
            if (forward) {
                for (auto &mps : mpss)
                    mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                pbr = me->move_to(i + 1, true);
                _t.get_time();
                shared_ptr<EffectiveHamiltonian<S, FL>> k_eff =
                    me->eff_ham(FuseTypes::NoFuseL, forward, false,
                                rights.front(), rights.back());
                teff += _t.get_time();
                sweep_max_eff_ham_size =
                    max(sweep_max_eff_ham_size, k_eff->op->get_total_memory());
                pdi = k_eff->expect(me->mpo->const_e, algo_type, ex_type,
                                    me->para_rule);
                tex += _t.get_time();
                k_eff->deallocate();
                if (me->para_rule == nullptr || me->para_rule->is_root())
                    for (int ip = 0; ip < (int)mpss.size(); ip++) {
                        auto &mps = mpss[ip];
                        auto &right = rights[ip];
                        MovingEnvironment<S, FL, FLS>::contract_one_dot(
                            i + 1, right, mps, forward);
                        mps->save_tensor(i + 1);
                        mps->unload_tensor(i + 1);
                    }
            } else {
                for (auto &mps : mpss)
                    mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                pbr = me->move_to(i - 1, true);
                _t.get_time();
                shared_ptr<EffectiveHamiltonian<S, FL>> k_eff =
                    me->eff_ham(FuseTypes::NoFuseR, forward, false,
                                lefts.front(), lefts.back());
                teff += _t.get_time();
                sweep_max_eff_ham_size =
                    max(sweep_max_eff_ham_size, k_eff->op->get_total_memory());
                pdi = k_eff->expect(me->mpo->const_e, algo_type, ex_type,
                                    me->para_rule);
                tex += _t.get_time();
                k_eff->deallocate();
                if (me->para_rule == nullptr || me->para_rule->is_root())
                    for (int ip = 0; ip < (int)mpss.size(); ip++) {
                        auto &mps = mpss[ip];
                        auto &left = lefts[ip];
                        MovingEnvironment<S, FL, FLS>::contract_one_dot(
                            i - 1, left, mps, forward);
                        mps->save_tensor(i - 1);
                        mps->unload_tensor(i - 1);
                    }
            }
            if (max_move_env_mem.first + max_move_env_mem.second <=
                pbr.first + pbr.second)
                max_move_env_mem.first = pbr.first,
                max_move_env_mem.second = pbr.second;
        } else {
            if (me->para_rule == nullptr || me->para_rule->is_root()) {
                for (int ip = 0; ip < (int)mpss.size(); ip++) {
                    auto &mps = mpss[ip];
                    auto &left = lefts[ip];
                    auto &right = rights[ip];
                    // propagation
                    if (forward) {
                        if (i != me->n_sites - 1) {
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i + 1, right, mps, forward);
                            mps->save_tensor(i + 1);
                            mps->unload_tensor(i + 1);
                        } else {
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i, right, mps, !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                        }
                    } else {
                        if (i > 0) {
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i - 1, left, mps, forward);
                            mps->save_tensor(i - 1);
                            mps->unload_tensor(i - 1);
                        } else {
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            MovingEnvironment<S, FL, FLS>::contract_one_dot(
                                i, left, mps, !forward);
                            mps->save_tensor(i);
                            mps->unload_tensor(i);
                        }
                    }
                }
            }
            get<1>(pdi) = 0;
            get<2>(pdi) = 1E-24;
        }
        for (int ip = (int)mpss.size() - 1; ip >= 0; ip--) {
            if (rights.size() != 0) {
                rights[ip]->info->deallocate();
                rights[ip]->deallocate();
            }
            if (lefts.size() != 0) {
                lefts[ip]->info->deallocate();
                lefts[ip]->deallocate();
            }
            if (dms.size() != 0) {
                dms[ip]->info->deallocate();
                dms[ip]->deallocate();
            }
        }
        for (auto &old_wfn : old_wfns) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
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
            mps->save_data();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        vector<pair<shared_ptr<OpExpr<S>>, FLX>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++)
            expectations[k] =
                make_pair(get<0>(pdi)[k].first, (FLX)get<0>(pdi)[k].second);
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_one_dot(int i, bool forward, bool propagate,
                             ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MPS<S, FLS>>> mpss =
            me->bra == me->ket
                ? vector<shared_ptr<MPS<S, FLS>>>{me->bra}
                : vector<shared_ptr<MPS<S, FLS>>>{me->bra, me->ket};
        bool fuse_left = i <= me->fuse_center;
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
        _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff = me->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, false,
            me->bra->tensors[i], me->ket->tensors[i]);
        teff += _t.get_time();
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        auto pdi = h_eff->expect(me->mpo->const_e, algo_type, ex_type,
                                 me->para_rule, fuse_left);
        tex += _t.get_time();
        h_eff->deallocate();
        FPS bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                for (auto &mps : mpss) {
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
            vector<shared_ptr<SparseMatrix<S, FLS>>> old_wfns =
                me->bra == me->ket
                    ? vector<shared_ptr<SparseMatrix<S, FLS>>>{me->bra
                                                                   ->tensors[i]}
                    : vector<shared_ptr<SparseMatrix<S, FLS>>>{
                          me->ket->tensors[i], me->bra->tensors[i]};
            if (propagate) {
                for (auto &mps : mpss) {
                    shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i];
                    shared_ptr<SparseMatrix<S, FLS>> left, right;
                    shared_ptr<SparseMatrix<S, FLS>> dm =
                        MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    int bond_dim =
                        mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    bool store_wfn_spectra =
                        mps == me->bra ? store_bra_spectra : store_ket_spectra;
                    FPS error =
                        MovingEnvironment<S, FL, FLS>::split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false, left, right,
                            cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
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
                        info->deallocate();
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
            for (auto &mps : mpss)
                mps->save_data();
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
        vector<pair<shared_ptr<OpExpr<S>>, FLX>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++)
            expectations[k] =
                make_pair(get<0>(pdi)[k].first, (FLX)get<0>(pdi)[k].second);
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_two_dot(int i, bool forward, bool propagate,
                             ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MPS<S, FLS>>> mpss =
            me->bra == me->ket
                ? vector<shared_ptr<MPS<S, FLS>>>{me->bra}
                : vector<shared_ptr<MPS<S, FLS>>>{me->bra, me->ket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_two_dot(i, mps, true);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, forward, false, me->bra->tensors[i],
                        me->ket->tensors[i]);
        teff += _t.get_time();
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        auto pdi =
            h_eff->expect(me->mpo->const_e, algo_type, ex_type, me->para_rule);
        tex += _t.get_time();
        h_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S, FLS>>> old_wfns =
            me->bra == me->ket
                ? vector<shared_ptr<SparseMatrix<S, FLS>>>{me->bra->tensors[i]}
                : vector<shared_ptr<SparseMatrix<S, FLS>>>{me->ket->tensors[i],
                                                           me->bra->tensors[i]};
        FPS bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (propagate) {
                for (auto &mps : mpss) {
                    shared_ptr<SparseMatrix<S, FLS>> old_wfn = mps->tensors[i];
                    shared_ptr<SparseMatrix<S, FLS>> dm =
                        MovingEnvironment<S, FL, FLS>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    int bond_dim =
                        mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    bool store_wfn_spectra =
                        mps == me->bra ? store_bra_spectra : store_ket_spectra;
                    FPS error =
                        MovingEnvironment<S, FL, FLS>::split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false,
                            mps->tensors[i], mps->tensors[i + 1], cutoff,
                            store_wfn_spectra, wfn_spectra, trunc_type);
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
                    MovingEnvironment<S, FL, FLS>::propagate_wfn(
                        i, 0, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
                }
            } else
                for (auto &mps : mpss)
                    mps->save_tensor(i);
            for (auto &mps : mpss)
                mps->save_data();
        } else {
            if (propagate) {
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
        }
        for (auto &old_wfn : old_wfns) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        vector<pair<shared_ptr<OpExpr<S>>, FLX>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++)
            expectations[k] =
                make_pair(get<0>(pdi)[k].first, (FLX)get<0>(pdi)[k].second);
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_multi_one_dot(int i, bool forward, bool propagate,
                                   ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        shared_ptr<MultiMPS<S, FLS>> mket =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->ket);
        shared_ptr<MultiMPS<S, FLS>> mbra =
            dynamic_pointer_cast<MultiMPS<S, FLS>>(me->bra);
        if (me->bra == me->ket)
            assert(mbra == mket);
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MultiMPS<S, FLS>>> mpss =
            me->bra == me->ket
                ? vector<shared_ptr<MultiMPS<S, FLS>>>{mbra}
                : vector<shared_ptr<MultiMPS<S, FLS>>>{mbra, mket};
        bool fuse_left = i <= me->fuse_center;
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
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_wfns =
                    mps->wfns;
                if (fuse_left && mps->canonical_form[i] == 'T')
                    mps->wfns = MovingEnvironment<S, FL, FLS>::
                        swap_multi_wfn_to_fused_left(i, mps->info, prev_wfns,
                                                     me->mpo->tf->opf->cg);
                else if (!fuse_left && mps->canonical_form[i] == 'J')
                    mps->wfns = MovingEnvironment<S, FL, FLS>::
                        swap_multi_wfn_to_fused_right(i, mps->info, prev_wfns,
                                                      me->mpo->tf->opf->cg);
                for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                    prev_wfns[j]->deallocate();
                if (prev_wfns.size() != 0)
                    prev_wfns[0]->deallocate_infos();
            }
        }
        _t.get_time();
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> h_eff =
            me->multi_eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                              forward, false);
        teff += _t.get_time();
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        auto pdi =
            h_eff->expect(me->mpo->const_e, algo_type, ex_type, me->para_rule);
        tex += _t.get_time();
        h_eff->deallocate();
        FPS bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            // change to fused form for splitting
            if (fuse_left != forward) {
                for (auto &mps : mpss) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> prev_wfns =
                        mps->wfns;
                    if (!fuse_left && forward)
                        mps->wfns = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_left(
                                i, mps->info, prev_wfns, me->mpo->tf->opf->cg);
                    else if (fuse_left && !forward)
                        mps->wfns = MovingEnvironment<S, FL, FLS>::
                            swap_multi_wfn_to_fused_right(
                                i, mps->info, prev_wfns, me->mpo->tf->opf->cg);
                    for (int j = (int)prev_wfns.size() - 1; j >= 0; j--)
                        prev_wfns[j]->deallocate();
                    if (prev_wfns.size() != 0)
                        prev_wfns[0]->deallocate_infos();
                }
            }
            // splitting of wavefunction
            vector<vector<shared_ptr<SparseMatrixGroup<S, FLS>>>> old_wfnss =
                me->bra == me->ket
                    ? vector<vector<
                          shared_ptr<SparseMatrixGroup<S, FLS>>>>{mbra->wfns}
                    : vector<vector<shared_ptr<SparseMatrixGroup<S, FLS>>>>{
                          mket->wfns, mbra->wfns};
            if (propagate) {
                for (auto &mps : mpss) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfn =
                                                                      mps->wfns,
                                                                  new_wfns;
                    shared_ptr<SparseMatrix<S, FLS>> rot;
                    shared_ptr<SparseMatrix<S, FLS>> dm =
                        MovingEnvironment<S, FL, FLS>::
                            density_matrix_with_multi_target(
                                mps->info->vacuum, old_wfn, mps->weights,
                                forward, 0.0, NoiseTypes::None);
                    int bond_dim =
                        mps == mbra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    bool store_wfn_spectra =
                        mps == me->bra ? store_bra_spectra : store_ket_spectra;
                    FPS error = MovingEnvironment<S, FL, FLS>::
                        multi_split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false, new_wfns,
                            rot, cutoff, store_wfn_spectra, wfn_spectra,
                            trunc_type);
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
                            MovingEnvironment<
                                S, FL, FLS>::contract_multi_one_dot(i + 1,
                                                                    new_wfns,
                                                                    mps,
                                                                    forward);
                            mps->save_wavefunction(i + 1);
                            mps->unload_wavefunction(i + 1);
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'T';
                        } else {
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            MovingEnvironment<
                                S, FL, FLS>::contract_multi_one_dot(i, new_wfns,
                                                                    mps,
                                                                    !forward);
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
                            MovingEnvironment<
                                S, FL, FLS>::contract_multi_one_dot(i - 1,
                                                                    new_wfns,
                                                                    mps,
                                                                    forward);
                            mps->save_wavefunction(i - 1);
                            mps->unload_wavefunction(i - 1);
                            mps->canonical_form[i - 1] = 'J';
                            mps->canonical_form[i] = 'R';
                        } else {
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            MovingEnvironment<
                                S, FL, FLS>::contract_multi_one_dot(i, new_wfns,
                                                                    mps,
                                                                    !forward);
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
            for (auto &mps : mpss)
                mps->save_data();
        } else {
            vector<vector<shared_ptr<SparseMatrixGroup<S, FLS>>>> old_wfnss =
                me->bra == me->ket
                    ? vector<vector<
                          shared_ptr<SparseMatrixGroup<S, FLS>>>>{mbra->wfns}
                    : vector<vector<shared_ptr<SparseMatrixGroup<S, FLS>>>>{
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
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
                            mps->tensors[i + 1] = nullptr;
                            mps->canonical_form[i] = 'L';
                            mps->canonical_form[i + 1] = 'T';
                        } else
                            mps->canonical_form[i] = 'J';
                    } else {
                        if (i > 0) {
                            mps->tensors[i - 1] = nullptr;
                            mps->tensors[i] =
                                make_shared<SparseMatrix<S, FLS>>();
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
        vector<pair<shared_ptr<OpExpr<S>>, FLX>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++) {
            typename PartitionWeights<FLX>::type::value_type x = 0.0;
            for (size_t l = 0; l < partition_weights.size(); l++)
                x += partition_weights[l] * get<0>(pdi)[k].second[l];
            expectations[k] = make_pair(get<0>(pdi)[k].first, (FLX)x);
        }
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration update_multi_two_dot(int i, bool forward, bool propagate,
                                   ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        shared_ptr<MultiMPS<S, FLS>> mket =
                                         dynamic_pointer_cast<MultiMPS<S, FLS>>(
                                             me->ket),
                                     mbra =
                                         dynamic_pointer_cast<MultiMPS<S, FLS>>(
                                             me->bra);
        if (me->bra == me->ket)
            assert(mbra == mket);
        frame_<FPS>()->activate(0);
        vector<shared_ptr<MultiMPS<S, FLS>>> mpss =
            me->bra == me->ket
                ? vector<shared_ptr<MultiMPS<S, FLS>>>{mbra}
                : vector<shared_ptr<MultiMPS<S, FLS>>>{mbra, mket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr || mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S, FL, FLS>::contract_multi_two_dot(
                    i, mps, mps == mket);
            else
                mps->load_tensor(i);
            mps->tensors[i] = mps->tensors[i + 1] = nullptr;
        }
        _t.get_time();
        shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> h_eff =
            me->multi_eff_ham(FuseTypes::FuseLR, forward, false);
        teff += _t.get_time();
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, h_eff->op->get_total_memory());
        auto pdi =
            h_eff->expect(me->mpo->const_e, algo_type, ex_type, me->para_rule);
        tex += _t.get_time();
        h_eff->deallocate();
        vector<vector<shared_ptr<SparseMatrixGroup<S, FLS>>>> old_wfnss =
            me->bra == me->ket
                ? vector<
                      vector<shared_ptr<SparseMatrixGroup<S, FLS>>>>{mbra->wfns}
                : vector<vector<shared_ptr<SparseMatrixGroup<S, FLS>>>>{
                      mket->wfns, mbra->wfns};
        FPS bra_error = 0.0, ket_error = 0.0;
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            if (propagate) {
                for (auto &mps : mpss) {
                    vector<shared_ptr<SparseMatrixGroup<S, FLS>>> old_wfn =
                        mps->wfns;
                    shared_ptr<SparseMatrix<S, FLS>> dm =
                        MovingEnvironment<S, FL, FLS>::
                            density_matrix_with_multi_target(
                                mps->info->vacuum, old_wfn, mps->weights,
                                forward, 0.0, NoiseTypes::None);
                    int bond_dim =
                        mps == mbra ? (int)bra_bond_dim : (int)ket_bond_dim;
                    bool store_wfn_spectra =
                        mps == me->bra ? store_bra_spectra : store_ket_spectra;
                    FPS error = MovingEnvironment<S, FL, FLS>::
                        multi_split_density_matrix(
                            dm, old_wfn, bond_dim, forward, false, mps->wfns,
                            forward ? mps->tensors[i] : mps->tensors[i + 1],
                            cutoff, store_wfn_spectra, wfn_spectra, trunc_type);
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
                    MovingEnvironment<S, FL, FLS>::propagate_multi_wfn(
                        i, 0, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
                }
            } else {
                for (auto &mps : mpss)
                    mps->save_tensor(i);
            }
            for (auto &mps : mpss)
                mps->save_data();
        } else {
            if (propagate) {
                for (auto &mps : mpss) {
                    if (forward) {
                        mps->tensors[i] = make_shared<SparseMatrix<S, FLS>>();
                        mps->tensors[i + 1] = nullptr;
                        mps->canonical_form[i] = 'L';
                        mps->canonical_form[i + 1] = 'M';
                    } else {
                        mps->tensors[i] = nullptr;
                        mps->tensors[i + 1] =
                            make_shared<SparseMatrix<S, FLS>>();
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
        vector<pair<shared_ptr<OpExpr<S>>, FLX>> expectations(
            get<0>(pdi).size());
        for (size_t k = 0; k < get<0>(pdi).size(); k++) {
            typename PartitionWeights<FLX>::type::value_type x = 0.0;
            for (size_t l = 0; l < partition_weights.size(); l++)
                x += partition_weights[l] * get<0>(pdi)[k].second[l];
            expectations[k] = make_pair(get<0>(pdi)[k].first, (FLX)x);
        }
        return Iteration(expectations, bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, bool propagate,
                       ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        _t2.get_time();
        pair<size_t, size_t> pbr = me->move_to(i);
        if (max_move_env_mem.first + max_move_env_mem.second <=
            pbr.first + pbr.second)
            max_move_env_mem.first = pbr.first,
            max_move_env_mem.second = pbr.second;
        tmve += _t2.get_time();
        assert(me->dot == 1 || me->dot == 2);
        Iteration it(vector<pair<shared_ptr<OpExpr<S>>, FLX>>(), 0, 0, 0, 0);
        if (me->dot == 2) {
            if (me->ket->canonical_form[i] == 'M' ||
                me->ket->canonical_form[i + 1] == 'M' ||
                me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T' ||
                me->ket->canonical_form[i + 1] == 'J' ||
                me->ket->canonical_form[i + 1] == 'T')
                it = update_multi_two_dot(i, forward, propagate, bra_bond_dim,
                                          ket_bond_dim);
            else
                it = update_two_dot(i, forward, propagate, bra_bond_dim,
                                    ket_bond_dim);
        } else {
            if (me->ket->canonical_form[i] == 'J' ||
                me->ket->canonical_form[i] == 'T' ||
                me->ket->canonical_form[i] == 'M')
                it = update_multi_one_dot(i, forward, propagate, bra_bond_dim,
                                          ket_bond_dim);
            else if (!zero_dot_algo)
                it = update_one_dot(i, forward, propagate, bra_bond_dim,
                                    ket_bond_dim);
            else
                it = update_zero_dot(i, forward, propagate, bra_bond_dim,
                                     ket_bond_dim);
        }
        if (store_bra_spectra || store_ket_spectra) {
            const int bond_update_idx =
                me->dot == 1 && !forward && i > 0 ? i - 1 : i;
            if (sweep_wfn_spectra.size() <= bond_update_idx)
                sweep_wfn_spectra.resize(bond_update_idx + 1);
            sweep_wfn_spectra[bond_update_idx] = wfn_spectra;
        }
        tblk += _t2.get_time();
        return it;
    }
    void sweep(bool forward, ubond_t bra_bond_dim, ubond_t ket_bond_dim) {
        teff = tex = tblk = tmve = 0;
        max_move_env_mem.first = max_move_env_mem.second = 0;
        frame_<FPS>()->twrite = frame_<FPS>()->tread = frame_<FPS>()->tasync =
            0;
        frame_<FPS>()->fpwrite = frame_<FPS>()->fpread = 0;
        if (frame_<FPS>()->fp_codec != nullptr)
            frame_<FPS>()->fp_codec->ndata = frame_<FPS>()->fp_codec->ncpsd = 0;
        if (me->para_rule != nullptr) {
            me->para_rule->comm->tcomm = 0;
            me->para_rule->comm->tidle = 0;
            me->para_rule->comm->twait = 0;
        }
        me->prepare();
        sweep_cumulative_nflop = 0;
        sweep_max_eff_ham_size = 0;
        frame_<FPS>()->reset_peak_used_memory();
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
            sweep_cumulative_nflop += r.nflop;
            if (iprint >= 2)
                cout << r << " T = " << setw(4) << fixed << setprecision(2)
                     << t.get_time() << endl;
            expectations[i] = r.expectations;
        }
    }
    FLX solve(bool propagate, bool forward = true) {
        Timer start, current;
        start.get_time();
        current.get_time();
        for (auto &x : expectations)
            x.clear();
        if (propagate) {
            if (iprint >= 1) {
                cout << endl
                     << "Expectation | Direction = " << setw(8)
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
            double tswp = current.get_time();
            if (iprint >= 1)
                cout << "Time elapsed = " << fixed << setw(10)
                     << setprecision(3) << current.current - start.current
                     << endl;
            if (iprint >= 2) {
                cout << fixed << setprecision(3);
                cout << "Time sweep = " << setw(12) << tswp;
                cout << " | "
                     << Parsing::to_size_string(sweep_cumulative_nflop,
                                                "FLOP/SWP")
                     << endl;
                if (me->para_rule != nullptr) {
                    double tt[3] = {me->para_rule->comm->tcomm,
                                    me->para_rule->comm->tidle,
                                    me->para_rule->comm->twait};
                    me->para_rule->comm->reduce_sum_optional(
                        &tt[0], 3, me->para_rule->comm->root);
                    tt[0] /= me->para_rule->comm->size;
                    tt[1] /= me->para_rule->comm->size;
                    tt[2] /= me->para_rule->comm->size;
                    cout << " | Tcomm = " << tt[0] << " | Tidle = " << tt[1]
                         << " | Twait = " << tt[2];
                    cout << endl;
                }
                size_t dmain = frame_<FPS>()->peak_used_memory[0];
                size_t dseco = frame_<FPS>()->peak_used_memory[1];
                size_t imain = frame_<FPS>()->peak_used_memory[2];
                size_t iseco = frame_<FPS>()->peak_used_memory[3];
                cout << " | Dmem = " << Parsing::to_size_string(dmain + dseco)
                     << " (" << (dmain * 100 / (dmain + dseco)) << "%)";
                cout << " | Imem = " << Parsing::to_size_string(imain + iseco)
                     << " (" << (imain * 100 / (imain + iseco)) << "%)";
                cout << " | Hmem = "
                     << Parsing::to_size_string(sweep_max_eff_ham_size *
                                                sizeof(FL));
                cout << " | MaxBmem = "
                     << Parsing::to_size_string(max_move_env_mem.first *
                                                sizeof(FL));
                cout << " | MaxRmem = "
                     << Parsing::to_size_string(max_move_env_mem.second *
                                                sizeof(FL));
                cout << endl;
                cout << " | Tread = " << frame_<FPS>()->tread
                     << " | Twrite = " << frame_<FPS>()->twrite
                     << " | Tfpread = " << frame_<FPS>()->fpread
                     << " | Tfpwrite = " << frame_<FPS>()->fpwrite;
                if (frame_<FPS>()->fp_codec != nullptr)
                    cout << " | data = "
                         << Parsing::to_size_string(
                                frame_<FPS>()->fp_codec->ndata * sizeof(FPS))
                         << " | cpsd = "
                         << Parsing::to_size_string(
                                frame_<FPS>()->fp_codec->ncpsd * sizeof(FPS));
                cout << " | Tasync = " << frame_<FPS>()->tasync << endl;
                if (me != nullptr)
                    cout << " | Trot = " << me->trot << " | Tctr = " << me->tctr
                         << " | Tint = " << me->tint << " | Tmid = " << me->tmid
                         << " | Tdctr = " << me->tdctr
                         << " | Tdiag = " << me->tdiag
                         << " | Tinfo = " << me->tinfo << endl;
                cout << " | Teff = " << teff << " | Texpt = " << tex
                     << " | Tblk = " << tblk << " | Tmve = " << tmve << endl
                     << endl;
            }
            this->forward = forward;
            if (expectations.size() != 0 && expectations[0].size() == 1)
                return expectations[0][0].second;
            else if (expectations.size() != 0 &&
                     expectations.back().size() == 1)
                return expectations.back()[0].second;
            else
                return 0.0;
        } else {
            Iteration r = blocking(me->center, forward, false, bra_bond_dim,
                                   ket_bond_dim);
            assert(r.expectations.size() != 0);
            return r.expectations[0].second;
        }
    }
    GMatrix<FLX> get_1pdm_spatial(uint16_t n_physical_sites = 0U) const {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        return PDM1MPOQC<S, FLX>::get_matrix_spatial(expectations,
                                                     n_physical_sites);
    }
    GMatrix<FLX> get_1pdm(uint16_t n_physical_sites = 0U) const {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        return PDM1MPOQC<S, FLX>::get_matrix(expectations, n_physical_sites);
    }
    shared_ptr<GTensor<FLX>>
    get_2pdm_spatial(uint16_t n_physical_sites = 0U) const {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        return PDM2MPOQC<S, FLX>::get_matrix_spatial(expectations,
                                                     n_physical_sites);
    }
    shared_ptr<GTensor<FLX>> get_2pdm(uint16_t n_physical_sites = 0U) const {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        return PDM2MPOQC<S, FLX>::get_matrix(expectations, n_physical_sites);
    }
    // number of particle correlation
    // s == 0: pure spin; s == 1: mixed spin
    GMatrix<FLX> get_1npc_spatial(uint8_t s,
                                  uint16_t n_physical_sites = 0U) const {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        return NPC1MPOQC<S, FLX>::get_matrix_spatial(s, expectations,
                                                     n_physical_sites);
    }
    // number of particle correlation
    // s == 0: pure spin; s == 1: mixed spin
    GMatrix<FLX> get_1npc(uint8_t s, uint16_t n_physical_sites = 0U) const {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        return NPC1MPOQC<S, FLX>::get_matrix(s, expectations, n_physical_sites);
    }
    vector<shared_ptr<GTensor<FLX>>> get_npdm(uint16_t n_physical_sites = 0U) {
        if (me->mpo->npdm_scheme == nullptr)
            throw runtime_error(
                "Expect::get_npdm only works with general NPDM MPO.");
        shared_ptr<NPDMScheme> scheme = me->mpo->npdm_scheme;
        vector<shared_ptr<GTensor<FLX>>> r(scheme->perms.size());
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        size_t total_mem = 0;
        for (int i = 0; i < (int)scheme->perms.size(); i++) {
            int n_op = (int)scheme->perms[i]->index_patterns[0].size();
            if (scheme->perms[i]->mask.size() != 0) {
                n_op = 1;
                for (int k = 1; k < scheme->perms[i]->mask.size(); k++)
                    n_op += scheme->perms[i]->mask[k] !=
                            scheme->perms[i]->mask[k - 1];
            }
            vector<MKL_INT> shape(n_op, n_physical_sites);
            r[i] = make_shared<GTensor<FLX>>(shape);
            r[i]->clear();
            total_mem += r[i]->size();
        }
        bool symbol_free = false;
        // for zero-dot backward, expectations[0] may be empty
        for (auto &v : expectations)
            if (v.size() == 1 && v[0].first->get_type() == OpTypes::Counter)
                symbol_free = true;
        if (iprint) {
            cout << "NPDM Sorting | Nsites = " << setw(5) << me->n_sites
                 << " | Nmaxops = " << setw(2) << scheme->n_max_ops
                 << " | DotSite = "
                 << (me->dot == 2 ? 2 : (zero_dot_algo ? 0 : 1));
            cout << " | SymbolFree = " << (symbol_free ? "T" : "F");
            if (symbol_free) {
                cout << " | Compressed = "
                     << ((algo_type & ExpectationAlgorithmTypes::Compressed) ||
                                 (algo_type &
                                  ExpectationAlgorithmTypes::Automatic)
                             ? "T"
                             : "F");
                cout << " | LowMem = "
                     << ((algo_type & ExpectationAlgorithmTypes::LowMem) ? "T"
                                                                         : "F");
            } else {
                cout << " | Fast = "
                     << ((algo_type & ExpectationAlgorithmTypes::Fast) ||
                                 (algo_type &
                                  ExpectationAlgorithmTypes::Automatic)
                             ? "T"
                             : "F");
            }
            cout << " | Mem = "
                 << Parsing::to_size_string(total_mem * sizeof(FLX)) << endl;
        }
        Timer current;
        current.get_time();
        double tsite, tsite_total = 0;
        // for zero-dot backward, last expectations may contain data
        // for one-dot, last expectations may contain repeated data
        int ixed = symbol_free || !(zero_dot_algo && me->dot == 1)
                       ? me->n_sites - 1
                       : me->n_sites;
        for (int ix = 0; ix < ixed; ix++) {
            vector<pair<shared_ptr<OpExpr<S>>, FLX>> &v = expectations[ix];
            if (iprint >= 2) {
                cout << " Site = " << setw(5) << ix << " .. ";
                cout.flush();
            }
            if (symbol_free)
                me->mpo->tf->template npdm_sort<FLX>(
                    scheme, r, me->get_npdm_fragment_filename(ix), me->n_sites,
                    ix,
                    (algo_type & ExpectationAlgorithmTypes::Compressed) ||
                        (algo_type & ExpectationAlgorithmTypes::Automatic));
            else
                for (size_t i = 0; i < (size_t)v.size(); i++) {
                    shared_ptr<OpElement<S, FL>> op =
                        dynamic_pointer_cast<OpElement<S, FL>>(v[i].first);
                    assert(op->name == OpNames::XPDM);
                    int ii = op->site_index.ss();
                    size_t kk = ((size_t)op->site_index[0] << 36) |
                                ((size_t)op->site_index[1] << 24) |
                                ((size_t)op->site_index[2] << 12) |
                                ((size_t)op->site_index[3]);
                    (*r[ii]->data)[kk] += v[i].second;
                }
            if (iprint >= 2) {
                tsite = current.get_time();
                cout << " T = " << fixed << setprecision(3) << tsite;
                cout << endl;
                tsite_total += tsite;
            }
        }
        if (!symbol_free && me->para_rule != nullptr) {
            current.get_time();
            for (int i = 0; i < (int)scheme->perms.size(); i++)
                me->para_rule->comm->allreduce_sum(r[i]->data->data(),
                                                   r[i]->data->size());
            tsite = current.get_time();
            cout << "Tcomm = " << fixed << setprecision(3) << tsite << " ";
        }
        if (iprint)
            cout << "Ttotal = " << fixed << setprecision(3) << setw(10)
                 << tsite_total << endl
                 << endl;
        return r;
    }
};

} // namespace block2
