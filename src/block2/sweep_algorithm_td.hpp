
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

enum struct TruncPatternTypes : uint8_t { None, TruncAfterOdd, TruncAfterEven };

// Imaginary/Real Time Evolution (td-DMRG++/RK4)
template <typename S> struct TDDMRG {
    shared_ptr<MovingEnvironment<S>> me;
    shared_ptr<MovingEnvironment<S>> lme, rme;
    vector<ubond_t> bond_dims;
    vector<double> noises;
    vector<double> errors;
    vector<double> energies;
    vector<double> normsqs;
    vector<double> discarded_weights;
    NoiseTypes noise_type = NoiseTypes::DensityMatrix;
    TruncationTypes trunc_type = TruncationTypes::Physical;
    DecompositionTypes decomp_type = DecompositionTypes::DensityMatrix;
    bool forward;
    TETypes mode = TETypes::ImagTE;
    int n_sub_sweeps = 1;
    vector<double> weights = {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0};
    uint8_t iprint = 2;
    double cutoff = 1E-14;
    bool decomp_last_site = true;
    size_t sweep_cumulative_nflop = 0;
    TDDMRG(const shared_ptr<MovingEnvironment<S>> &me,
           const vector<ubond_t> &bond_dims,
           const vector<double> &noises = vector<double>())
        : me(me), bond_dims(bond_dims), noises(noises), forward(false) {}
    struct Iteration {
        double energy, normsq, error;
        int nmult, mmps;
        double tmult;
        size_t nflop;
        Iteration(double energy, double normsq, double error, int mmps,
                  int nmult, size_t nflop = 0, double tmult = 1.0)
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
    // two-site algorithm
    Iteration update_two_dot(int i, bool forward, bool advance, double beta,
                             ubond_t bond_dim, double noise) {
        assert(rme->bra != rme->ket);
        frame->activate(0);
        vector<shared_ptr<MPS<S>>> mpss = {rme->bra, rme->ket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S>::contract_two_dot(i, mps);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<SparseMatrixGroup<S>> pbra = nullptr;
        tuple<double, double, int, size_t, double> pdi;
        bool last_site =
            (forward && i + 1 == me->n_sites - 1) || (!forward && i == 0);
        vector<shared_ptr<SparseMatrix<S>>> mrk4;
        shared_ptr<EffectiveHamiltonian<S>> r_eff =
            rme->eff_ham(FuseTypes::FuseLR, false, rme->bra->tensors[i],
                         rme->ket->tensors[i]);
        auto rvmt =
            r_eff->first_rk4_apply(-beta, me->mpo->const_e, rme->para_rule);
        r_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S>>> hkets = rvmt.first;
        memcpy(lme->bra->tensors[i]->data, hkets[0]->data,
               hkets[0]->total_memory * sizeof(double));
        shared_ptr<EffectiveHamiltonian<S>> l_eff =
            lme->eff_ham(FuseTypes::FuseLR, false, lme->bra->tensors[i],
                         lme->ket->tensors[i]);
        if (!advance && last_site) {
            pdi = l_eff->expo_apply(-beta, me->mpo->const_e, iprint >= 3,
                                    me->para_rule);
            memcpy(lme->bra->tensors[i]->data, hkets[0]->data,
                   hkets[0]->total_memory * sizeof(double));
            auto lvmt = l_eff->second_rk4_apply(-beta, me->mpo->const_e,
                                                hkets[1], false, me->para_rule);
            mrk4 = lvmt.first;
            get<2>(pdi) += get<2>(lvmt.second);
            get<3>(pdi) += get<3>(lvmt.second);
            get<4>(pdi) += get<4>(lvmt.second);
        } else if (last_site)
            pdi = l_eff->expo_apply(-beta, me->mpo->const_e, iprint >= 3,
                                    me->para_rule);
        else {
            auto lvmt = l_eff->second_rk4_apply(-beta, me->mpo->const_e,
                                                hkets[1], false, me->para_rule);
            mrk4 = lvmt.first;
            pdi = lvmt.second;
        }
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pbra = l_eff->perturbative_noise(forward, i, i + 1,
                                             FuseTypes::FuseLR, lme->bra->info,
                                             noise_type, me->para_rule);
        l_eff->deallocate();
        get<2>(pdi) += get<0>(rvmt.second);
        get<3>(pdi) += get<1>(rvmt.second);
        get<4>(pdi) += get<2>(rvmt.second);
        vector<shared_ptr<SparseMatrix<S>>> old_wfns;
        for (auto &mps : mpss)
            old_wfns.push_back(mps->tensors[i]);
        double bra_error = 0.0;
        int bra_mmps = 0;
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            assert(pbra != nullptr);
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (auto &mps : mpss) {
                shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                shared_ptr<SparseMatrix<S>> dm = nullptr;
                double error;
                if (decomp_type == DecompositionTypes::DensityMatrix) {
                    if (mps != rme->bra) {
                        dm = MovingEnvironment<S>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, 0.0,
                            NoiseTypes::None);
                    } else if (mrk4.size() == 0) {
                        dm = MovingEnvironment<S>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, noise,
                            noise_type, 1.0, pbra);
                    } else {
                        dm = MovingEnvironment<S>::density_matrix(
                            mps->info->vacuum, old_wfn, forward, noise,
                            noise_type, weights[0], pbra);
                        assert(mrk4.size() == 3);
                        for (int i = 0; i < 3; i++)
                            MovingEnvironment<S>::density_matrix_add_wfn(
                                dm, mrk4[i], forward, weights[i + 1]);
                    }
                    error = MovingEnvironment<S>::split_density_matrix(
                        dm, old_wfn, bond_dim, forward, false, mps->tensors[i],
                        mps->tensors[i + 1], cutoff, trunc_type);
                } else if (decomp_type == DecompositionTypes::SVD ||
                           decomp_type == DecompositionTypes::PureSVD) {
                    if (mps != me->bra) {
                        error = MovingEnvironment<S>::split_wavefunction_svd(
                            mps->info->vacuum, old_wfn, bond_dim, forward,
                            false, mps->tensors[i], mps->tensors[i + 1], cutoff,
                            trunc_type, decomp_type, nullptr);
                    } else {
                        if (noise != 0 && mps == me->bra) {
                            if (noise_type == NoiseTypes::Wavefunction)
                                MovingEnvironment<S>::wavefunction_add_noise(
                                    old_wfn, noise);
                            else if (noise_type & NoiseTypes::Perturbative)
                                MovingEnvironment<S>::scale_perturbative_noise(
                                    noise, noise_type, pbra);
                        }
                        vector<double> xweights = {1};
                        vector<shared_ptr<SparseMatrix<S>>> xwfns = {};
                        if (mrk4.size() != 0) {
                            assert(mrk4.size() == 3);
                            xweights = weights;
                            xwfns = mrk4;
                        }
                        error = MovingEnvironment<S>::split_wavefunction_svd(
                            mps->info->vacuum, old_wfn, bond_dim, forward,
                            false, mps->tensors[i], mps->tensors[i + 1], cutoff,
                            trunc_type, decomp_type, pbra, xwfns, xweights);
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
                    if (!(last_site && advance)) {
                        if (forward)
                            mps->tensors[i + 1]->normalize();
                        else
                            mps->tensors[i]->normalize();
                    }
                    bra_mmps = info->n_states_total;
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
        if (pbra != nullptr) {
            pbra->deallocate();
            pbra->deallocate_infos();
        }
        for (auto mrk : mrk4)
            mrk->deallocate();
        for (auto hket : hkets)
            hket->deallocate();
        for (auto &old_wfn : vector<shared_ptr<SparseMatrix<S>>>(
                 old_wfns.rbegin(), old_wfns.rend())) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        if (me->para_rule == nullptr || me->para_rule->is_root()) {
            for (auto &mps : mpss) {
                MovingEnvironment<S>::propagate_wfn(
                    i, me->n_sites, mps, forward, me->mpo->tf->opf->cg);
                mps->save_data();
            }
        }
        if (me->para_rule != nullptr)
            me->para_rule->comm->barrier();
        return Iteration(get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), bra_error, bra_mmps,
                         get<2>(pdi), get<3>(pdi), get<4>(pdi));
    }
    Iteration blocking(int i, bool forward, bool advance, double beta,
                       ubond_t bond_dim, double noise) {
        lme->move_to(i);
        rme->move_to(i);
        assert(rme->dot == 2 || rme->dot == 1);
        if (rme->dot == 2)
            return update_two_dot(i, forward, advance, beta, bond_dim, noise);
        else
            assert(false);
        // else
        //     return update_one_dot(i, forward, advance, beta, bond_dim,
        //     noise);
    }
    tuple<double, double, double> sweep(bool forward, bool advance, double beta,
                                        ubond_t bond_dim, double noise) {
        lme->prepare();
        rme->prepare();
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
    void init_moving_environments() {
        const string base_tag = Parsing::split(me->tag, "@", true)[0];
        const string base_mps_tag = Parsing::split(me->ket->tag, "@", true)[0];
        vector<string> tags = {base_tag, base_tag + "@L", base_tag + "@R"};
        vector<string> mps_tags = {base_mps_tag, base_mps_tag + "@X"};
        vector<string> avail_tags, avail_mps_tags;
        string avail_mps_tag = "";
        avail_tags.reserve(2);
        for (auto tag : tags)
            if (tag != me->tag)
                avail_tags.push_back(tag);
        for (auto tag : mps_tags)
            if (tag != me->ket->tag)
                avail_mps_tag = tag;
        assert(avail_tags.size() == 2 && avail_mps_tag != "");
        lme = me->shallow_copy(avail_tags[0]);
        rme = me->shallow_copy(avail_tags[1]);
        rme->bra = me->ket->shallow_copy(avail_mps_tag);
        lme->bra = lme->ket = rme->bra;
    }
    double solve(int n_sweeps, double beta, bool forward = true,
                 double tol = 1E-6) {
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
                    cout << " | Norm = " << setw(15)
                         << sqrt(get<1>(sweep_results));
                    cout << " | MaxError = " << setw(15)
                         << get<2>(sweep_results);
                    if (iprint >= 2) {
                        size_t dmain = frame->peak_used_memory[0];
                        size_t dseco = frame->peak_used_memory[1];
                        size_t imain = frame->peak_used_memory[2];
                        size_t iseco = frame->peak_used_memory[3];
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
        bool fuse_left = i <= me->fuse_center;
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
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<MatrixRef> pdpf;
        tuple<double, double, int, size_t, double> pdi;
        // effective hamiltonian
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, true,
                        me->bra->tensors[i], me->ket->tensors[i]);
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
                dm = MovingEnvironment<S>::density_matrix(
                    me->ket->info->vacuum, me->ket->tensors[i], forward, noise,
                    noise_type, weights[0]);
                MovingEnvironment<S>::density_matrix_add_matrices(
                    dm, me->ket->tensors[i], forward, pdpf, weights);
                frame->activate(1);
                for (int i = pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame->activate(0);
            } else
                dm = MovingEnvironment<S>::density_matrix(
                    me->ket->info->vacuum, me->ket->tensors[i], forward, noise,
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
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
                me->ket->info->left_dims[i + 1] = info;
                me->ket->info->save_left_dims(i + 1);
            } else {
                if (mode == TETypes::RK4 && (i != 0 || !advance))
                    left->normalize();
                me->ket->tensors[i] = right;
                me->ket->save_tensor(i);
                info = right->info->extract_state_info(forward);
                mmps = info->n_states_total;
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
        me->ket->save_data();
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
        tuple<double, double, int, size_t, double> pdi;
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        vector<MatrixRef> pdpf;
        shared_ptr<EffectiveHamiltonian<S>> h_eff = me->eff_ham(
            FuseTypes::FuseLR, true, me->bra->tensors[i], me->ket->tensors[i]);
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
                dm = MovingEnvironment<S>::density_matrix(
                    me->ket->info->vacuum, h_eff->ket, forward, noise,
                    noise_type, weights[0]);
                MovingEnvironment<S>::density_matrix_add_matrices(
                    dm, h_eff->ket, forward, pdpf, weights);
                frame->activate(1);
                for (int i = pdpf.size() - 1; i >= 0; i--)
                    pdpf[i].deallocate();
                frame->activate(0);
            } else
                dm = MovingEnvironment<S>::density_matrix(me->ket->info->vacuum,
                                                          h_eff->ket, forward,
                                                          noise, noise_type);
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
                me->ket->info->bond_dim =
                    max(me->ket->info->bond_dim, (ubond_t)mmps);
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
        me->ket->save_data();
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
                    cout << "Time elapsed = " << fixed << setw(10)
                         << setprecision(3) << current.current - start.current
                         << endl;
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

} // namespace block2
