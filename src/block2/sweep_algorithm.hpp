
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
    vector<uint16_t> bond_dims;
    vector<double> noises;
    vector<double> energies;
    bool forward;
    uint8_t iprint = 2;
    DMRG(const shared_ptr<MovingEnvironment<S>> &me,
         const vector<uint16_t> &bond_dims, const vector<double> &noises)
        : me(me), bond_dims(bond_dims), noises(noises), forward(false) {}
    struct Iteration {
        double energy, error;
        int ndav;
        double tdav;
        size_t nflop;
        Iteration(double energy, double error, int ndav, size_t nflop = 0,
                  double tdav = 1.0)
            : energy(energy), error(error), ndav(ndav), nflop(nflop),
              tdav(tdav) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Ndav = " << setw(4) << r.ndav << " E = " << setw(15)
               << r.energy << " Error = " << setw(15) << setprecision(12)
               << r.error << " FLOPS = " << scientific << setw(8)
               << setprecision(2) << (double)r.nflop / r.tdav
               << " Tdav = " << fixed << setprecision(2) << r.tdav;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, uint16_t bond_dim,
                             double noise) {
        frame->activate(0);
        if (me->ket->tensors[i] != nullptr &&
            me->ket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_two_dot(i, me->ket);
        else {
            me->ket->load_tensor(i);
            me->ket->tensors[i + 1] = nullptr;
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, true);
        auto pdi = h_eff->eigs(false);
        h_eff->deallocate();
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        shared_ptr<SparseMatrix<S>> dm = MovingEnvironment<S>::density_matrix(
            h_eff->opdq, h_eff->ket, forward, noise);
        double error = MovingEnvironment<S>::split_density_matrix(
            dm, h_eff->ket, (int)bond_dim, forward, me->ket->tensors[i],
            me->ket->tensors[i + 1]);
        shared_ptr<StateInfo<S>> info = nullptr;
        if (forward) {
            info = me->ket->tensors[i]->info->extract_state_info(forward);
            me->ket->info->left_dims[i + 1] = *info;
            me->ket->info->save_left_dims(i + 1);
            me->ket->canonical_form[i] = 'L';
            me->ket->canonical_form[i + 1] = 'C';
        } else {
            info = me->ket->tensors[i + 1]->info->extract_state_info(forward);
            me->ket->info->right_dims[i + 1] = *info;
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
        old_wfn->info->deallocate();
        old_wfn->deallocate();
        MovingEnvironment<S>::propagate_wfn(i, me->n_sites, me->ket, forward);
        return Iteration(get<0>(pdi) + me->mpo->const_e, error, get<1>(pdi),
                         get<2>(pdi), get<3>(pdi));
    }
    Iteration blocking(int i, bool forward, uint16_t bond_dim, double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, bond_dim, noise);
        else
            assert(false);
    }
    double sweep(bool forward, uint16_t bond_dim, double noise) {
        me->prepare();
        vector<double> energies;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                        << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r = blocking(i, forward, bond_dim, noise);
            cout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << t.get_time() << endl;
            energies.push_back(r.energy);
        }
        return *min_element(energies.begin(), energies.end());
    }
    double solve(int n_sweeps, bool forward = true, double tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        energies.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            if (iprint >= 1)
                cout << "Sweep = " << setw(4) << iw << " | Direction = " << setw(8)
                    << (forward ? "forward" : "backward")
                    << " | Bond dimension = " << setw(4) << bond_dims[iw]
                    << " | Noise = " << scientific << setw(9) << setprecision(2)
                    << noises[iw] << endl;
            double energy = sweep(forward, bond_dims[iw], noises[iw]);
            energies.push_back(energy);
            bool converged = energies.size() >= 2 && tol > 0 &&
                             abs(energies[energies.size() - 1] -
                                 energies[energies.size() - 2]) < tol &&
                             noises[iw] == noises.back() &&
                             bond_dims[iw] == bond_dims.back();
            forward = !forward;
            current.get_time();
            if (iprint == 1) {
                cout << fixed << setprecision(8);
                cout << "Energy = " << setw(15) << energy << " ";
            }
            if (iprint >= 1)
                cout << "Time elapsed = " << setw(10) << setprecision(2)
                    << current.current - start.current << endl;
            if (converged)
                break;
        }
        this->forward = forward;
        return energies.back();
    }
};

enum struct TETypes : uint8_t { TangentSpace, RK4 };

// Imaginary Time Evolution
template <typename S> struct ImaginaryTE {
    shared_ptr<MovingEnvironment<S>> me;
    vector<uint16_t> bond_dims;
    vector<double> noises;
    vector<double> energies;
    vector<double> normsqs;
    bool forward;
    TETypes mode;
    int n_sub_sweeps;
    uint8_t iprint = 2;
    ImaginaryTE(const shared_ptr<MovingEnvironment<S>> &me,
                const vector<uint16_t> &bond_dims,
                TETypes mode = TETypes::TangentSpace, int n_sub_sweeps = 1)
        : me(me), bond_dims(bond_dims), noises(vector<double>{0.0}),
          forward(false), mode(mode), n_sub_sweeps(n_sub_sweeps) {}
    struct Iteration {
        double energy, normsq, error;
        int nexpo, nexpok;
        double texpo;
        size_t nflop;
        Iteration(double energy, double normsq, double error, int nexpo,
                  int nexpok, size_t nflop = 0, double texpo = 1.0)
            : energy(energy), normsq(normsq), error(error), nexpo(nexpo),
              nexpok(nexpok), nflop(nflop), texpo(texpo) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Nexpo = " << setw(4) << r.nexpo << "/" << setw(4) << r.nexpok
               << " E = " << setw(15) << r.energy << " Error = " << setw(15)
               << setprecision(12) << r.error << " FLOPS = " << scientific
               << setw(8) << setprecision(2) << (double)r.nflop / r.texpo
               << " Texpo = " << fixed << setprecision(2) << r.texpo;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, bool advance, double beta,
                             uint16_t bond_dim, double noise) {
        frame->activate(0);
        if (me->ket->tensors[i] != nullptr &&
            me->ket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_two_dot(i, me->ket);
        else {
            me->ket->load_tensor(i);
            me->ket->tensors[i + 1] = nullptr;
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, true);
        tuple<double, double, int, size_t, double> pdi;
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        shared_ptr<SparseMatrix<S>> dm;
        if (!advance &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0))) {
            assert(effective_mode == TETypes::TangentSpace);
            MatrixRef tmp(nullptr, h_eff->ket->total_memory, 1);
            tmp.allocate();
            memcpy(tmp.data, h_eff->ket->data,
                   h_eff->ket->total_memory * sizeof(double));
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, false);
            memcpy(h_eff->ket->data, tmp.data, 
                   h_eff->ket->total_memory * sizeof(double));
            tmp.deallocate();
            h_eff->deallocate();
            dm = MovingEnvironment<S>::density_matrix(h_eff->opdq, h_eff->ket,
                                                      forward, noise);
        } else if (effective_mode == TETypes::TangentSpace) {
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, false);
            h_eff->deallocate();
            dm = MovingEnvironment<S>::density_matrix(h_eff->opdq, h_eff->ket,
                                                      forward, noise);
        } else if (effective_mode == TETypes::RK4) {
            const vector<double> weights = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0,
                                            1.0 / 6.0};
            auto pdp = h_eff->rk4_apply(-beta, me->mpo->const_e, false);
            pdi = pdp.second;
            h_eff->deallocate();
            dm = MovingEnvironment<S>::density_matrix_with_weights(
                h_eff->opdq, h_eff->ket, forward, noise, pdp.first, weights);
            frame->activate(1);
            for (int i = pdp.first.size() - 1; i >= 0; i--)
                pdp.first[i].deallocate();
            frame->activate(0);
        }
        double error = MovingEnvironment<S>::split_density_matrix(
            dm, h_eff->ket, (int)bond_dim, forward, me->ket->tensors[i],
            me->ket->tensors[i + 1]);
        shared_ptr<StateInfo<S>> info = nullptr;
        if (forward) {
            info = me->ket->tensors[i]->info->extract_state_info(forward);
            me->ket->info->left_dims[i + 1] = *info;
            me->ket->info->save_left_dims(i + 1);
            me->ket->canonical_form[i] = 'L';
            me->ket->canonical_form[i + 1] = 'C';
        } else {
            info = me->ket->tensors[i + 1]->info->extract_state_info(forward);
            me->ket->info->right_dims[i + 1] = *info;
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
        old_wfn->info->deallocate();
        old_wfn->deallocate();
        int expok = 0;
        if (mode == TETypes::TangentSpace && forward &&
            i + 1 != me->n_sites - 1) {
            me->move_to(i + 1);
            me->ket->load_tensor(i + 1);
            shared_ptr<EffectiveHamiltonian<S>> k_eff =
                me->eff_ham(FuseTypes::FuseR, true);
            auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, false);
            k_eff->deallocate();
            me->ket->save_tensor(i + 1);
            me->ket->unload_tensor(i + 1);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        } else if (mode == TETypes::TangentSpace && !forward && i != 0) {
            me->move_to(i - 1);
            me->ket->load_tensor(i);
            shared_ptr<EffectiveHamiltonian<S>> k_eff =
                me->eff_ham(FuseTypes::FuseL, true);
            auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, false);
            k_eff->deallocate();
            me->ket->save_tensor(i);
            me->ket->unload_tensor(i);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        }
        MovingEnvironment<S>::propagate_wfn(i, me->n_sites, me->ket, forward);
        return Iteration(get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, get<2>(pdi), expok,
                         get<3>(pdi), get<4>(pdi));
    }
    Iteration blocking(int i, bool forward, bool advance, double beta,
                       uint16_t bond_dim, double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, advance, beta, bond_dim, noise);
        else
            assert(false);
    }
    pair<double, double> sweep(bool forward, bool advance, double beta,
                               uint16_t bond_dim, double noise) {
        me->prepare();
        vector<double> energies, normsqs;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                        << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r = blocking(i, forward, advance, beta, bond_dim, noise);
            cout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << t.get_time() << endl;
            energies.push_back(r.energy);
            normsqs.push_back(r.normsq);
        }
        return make_pair(energies.back(), normsqs.back());
    }
    void normalize() {
        if (normsqs.back() != 1.0) {
            size_t center = me->ket->canonical_form.find('C');
            assert(center != string::npos);
            me->ket->load_tensor(center);
            me->ket->tensors[center]->factor /= sqrt(normsqs.back());
            me->ket->save_tensor(center);
            me->ket->unload_tensor(center);
        }
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
        for (int iw = 0; iw < n_sweeps; iw++)
            for (int isw = 0; isw < n_sub_sweeps; isw++) {
                if (iprint >= 1) {
                    cout << "Sweep = " << setw(4) << iw;
                    if (n_sub_sweeps != 1)
                        cout << " (" << setw(2) << isw << "/" << setw(2)
                            << (int)n_sub_sweeps << ")";
                    cout << " | Direction = " << setw(8)
                        << (forward ? "forward" : "backward")
                        << " | Beta = " << setw(10) << setprecision(5)
                        << beta * (iw + 1) << " | Bond dimension = " << setw(4)
                        << bond_dims[iw] << " | Noise = " << scientific << setw(9)
                        << setprecision(2) << noises[iw] << endl;
                }
                auto r = sweep(forward, isw == n_sub_sweeps - 1, beta,
                               bond_dims[iw], noises[iw]);
                energies.push_back(r.first);
                normsqs.push_back(r.second);
                normalize();
                forward = !forward;
                current.get_time();
                if (iprint == 1) {
                    cout << fixed << setprecision(8);
                    cout << "Energy = " << setw(15) << r.first
                        << " Norm = " << setw(15) << sqrt(r.second) << " ";
                }
                if (iprint >= 1)
                    cout << "Time elapsed = " << setw(10) << setprecision(2)
                        << current.current - start.current << endl;
            }
        this->forward = forward;
        return energies.back();
    }
};

// Compression
template <typename S> struct Compress {
    shared_ptr<MovingEnvironment<S>> me;
    vector<uint16_t> bra_bond_dims, ket_bond_dims;
    vector<double> noises;
    vector<double> norms;
    bool forward;
    uint8_t iprint = 2;
    Compress(const shared_ptr<MovingEnvironment<S>> &me,
             const vector<uint16_t> &bra_bond_dims,
             const vector<uint16_t> &ket_bond_dims,
             const vector<double> &noises)
        : me(me), bra_bond_dims(bra_bond_dims), ket_bond_dims(ket_bond_dims),
          noises(noises), forward(false) {}
    struct Iteration {
        double norm, error;
        double tmult;
        size_t nflop;
        Iteration(double norm, double error, size_t nflop = 0,
                  double tmult = 1.0)
            : norm(norm), error(error), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << " Norm = " << setw(15) << r.norm << " Error = " << setw(15)
               << setprecision(12) << r.error << " FLOPS = " << scientific
               << setw(8) << setprecision(2) << (double)r.nflop / r.tmult
               << " Tmult = " << fixed << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, uint16_t bra_bond_dim,
                             uint16_t ket_bond_dim, double noise) {
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
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, false);
        auto pdi = h_eff->multiply();
        h_eff->deallocate();
        shared_ptr<SparseMatrix<S>> old_bra = me->bra->tensors[i];
        shared_ptr<SparseMatrix<S>> old_ket = me->ket->tensors[i];
        double bra_error = 0.0;
        for (auto &mps : {me->bra, me->ket}) {
            shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
            shared_ptr<SparseMatrix<S>> dm =
                MovingEnvironment<S>::density_matrix(
                    h_eff->opdq, old_wfn, forward,
                    mps == me->bra ? noise : 0.0);
            int bond_dim =
                mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
            double error = MovingEnvironment<S>::split_density_matrix(
                dm, old_wfn, bond_dim, forward, mps->tensors[i],
                mps->tensors[i + 1]);
            if (mps == me->bra)
                bra_error = error;
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                info = mps->tensors[i]->info->extract_state_info(forward);
                mps->info->left_dims[i + 1] = *info;
                mps->info->save_left_dims(i + 1);
                mps->canonical_form[i] = 'L';
                mps->canonical_form[i + 1] = 'C';
            } else {
                info = mps->tensors[i + 1]->info->extract_state_info(forward);
                mps->info->right_dims[i + 1] = *info;
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
            MovingEnvironment<S>::propagate_wfn(i, me->n_sites, mps, forward);
        }
        for (auto &old_wfn : {old_ket, old_bra}) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        return Iteration(get<0>(pdi), bra_error, get<1>(pdi), get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, uint16_t bra_bond_dim,
                       uint16_t ket_bond_dim, double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, bra_bond_dim, ket_bond_dim,
                                  noise);
        else
            assert(false);
    }
    double sweep(bool forward, uint16_t bra_bond_dim, uint16_t ket_bond_dim,
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
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                        << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r =
                blocking(i, forward, bra_bond_dim, ket_bond_dim, noise);
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
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        norms.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            if (iprint >= 1)
                cout << "Sweep = " << setw(4) << iw << " | Direction = " << setw(8)
                    << (forward ? "forward" : "backward")
                    << " | BRA bond dimension = " << setw(4) << bra_bond_dims[iw]
                    << " | Noise = " << scientific << setw(9) << setprecision(2)
                    << noises[iw] << endl;
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
                cout << "Norm = " << setw(15) << norm << " ";
            }
            if (iprint >= 1)
                cout << "Time elapsed = " << setw(10) << setprecision(2)
                    << current.current - start.current << endl;
            if (converged)
                break;
        }
        this->forward = forward;
        return norms.back();
    }
};

// Expectation value
template <typename S> struct Expect {
    shared_ptr<MovingEnvironment<S>> me;
    uint16_t bra_bond_dim, ket_bond_dim;
    vector<vector<pair<shared_ptr<OpExpr<S>>, double>>> expectations;
    bool forward;
    uint8_t iprint = 2;
    Expect(const shared_ptr<MovingEnvironment<S>> &me, uint16_t bra_bond_dim,
           uint16_t ket_bond_dim)
        : me(me), bra_bond_dim(bra_bond_dim), ket_bond_dim(ket_bond_dim),
          forward(false) {
        expectations.resize(me->n_sites - me->dot + 1);
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
                os << " Nterms = " << setw(5) << r.expectations.size();
            os << " Error = " << setw(15) << setprecision(12) << r.bra_error
               << "/" << setw(15) << setprecision(12) << r.ket_error
               << " FLOPS = " << scientific << setw(8) << setprecision(2)
               << (double)r.nflop / r.tmult << " Tmult = " << fixed
               << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, bool propagate,
                             uint16_t bra_bond_dim, uint16_t ket_bond_dim) {
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
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, false);
        auto pdi = h_eff->expect();
        h_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S>>> old_wfns =
            me->bra == me->ket
                ? vector<shared_ptr<SparseMatrix<S>>>{me->bra->tensors[i]}
                : vector<shared_ptr<SparseMatrix<S>>>{me->ket->tensors[i],
                                                      me->bra->tensors[i]};
        double bra_error = 0.0, ket_error = 0.0;
        if (propagate) {
            for (auto &mps : mpss) {
                shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                shared_ptr<SparseMatrix<S>> dm =
                    MovingEnvironment<S>::density_matrix(h_eff->opdq, old_wfn,
                                                         forward, 0.0);
                int bond_dim =
                    mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                double error = MovingEnvironment<S>::split_density_matrix(
                    dm, old_wfn, bond_dim, forward, mps->tensors[i],
                    mps->tensors[i + 1]);
                if (mps == me->bra)
                    bra_error = error;
                else
                    ket_error = error;
                shared_ptr<StateInfo<S>> info = nullptr;
                if (forward) {
                    info = mps->tensors[i]->info->extract_state_info(forward);
                    mps->info->left_dims[i + 1] = *info;
                    mps->info->save_left_dims(i + 1);
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'C';
                } else {
                    info =
                        mps->tensors[i + 1]->info->extract_state_info(forward);
                    mps->info->right_dims[i + 1] = *info;
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
                MovingEnvironment<S>::propagate_wfn(i, me->n_sites, mps,
                                                    forward);
            }
        }
        for (auto &old_wfn : old_wfns) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        return Iteration(get<0>(pdi), bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, bool propagate,
                       uint16_t bra_bond_dim, uint16_t ket_bond_dim) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, propagate, bra_bond_dim,
                                  ket_bond_dim);
        else
            assert(false);
    }
    void sweep(bool forward, uint16_t bra_bond_dim, uint16_t ket_bond_dim) {
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
            if (iprint >= 2) {
                if (me->dot == 2)
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                        << " .. ";
                else
                    cout << " " << (forward ? "-->" : "<--")
                        << " Site = " << setw(4) << i << " .. ";
                cout.flush();
            }
            t.get_time();
            Iteration r =
                blocking(i, forward, true, bra_bond_dim, ket_bond_dim);
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
            if (iprint >= 1)
                cout << "Expectation | Direction = " << setw(8)
                    << (forward ? "forward" : "backward")
                    << " | BRA bond dimension = " << setw(4) << bra_bond_dim
                    << " | KET bond dimension = " << setw(4) << ket_bond_dim
                    << endl;
            sweep(forward, bra_bond_dim, ket_bond_dim);
            forward = !forward;
            current.get_time();
            if (iprint >= 1)
                cout << "Time elapsed = " << setw(10) << setprecision(2)
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
};

} // namespace block2
