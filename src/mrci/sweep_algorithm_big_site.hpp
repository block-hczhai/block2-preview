
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

#include "../dmrg/sweep_algorithm.hpp"
#include <limits>

using namespace std;

namespace block2 {

// Density Matrix Renormalization Group with big site
template <typename S> struct DMRGBigSite : DMRG<S> {
    using DMRG<S>::iprint;
    using DMRG<S>::me;
    using DMRG<S>::ext_mes;
    using DMRG<S>::davidson_soft_max_iter;
    using DMRG<S>::noise_type;
    using DMRG<S>::decomp_type;
    using typename DMRG<S>::Iteration;
    bool last_site_svd = false;
    bool last_site_1site = false; // ATTENTION: only use in two site algorithm
    DMRGBigSite(const shared_ptr<MovingEnvironment<S>> &me,
                const vector<ubond_t> &bond_dims, const vector<double> &noises)
        : DMRG<S>(me, bond_dims, noises) {}
    Iteration blocking(int i, bool forward, ubond_t bond_dim, double noise,
                       double davidson_conv_thrd) override {
        const int dsmi =
            davidson_soft_max_iter; // Save it as it may be changed here
        const NoiseTypes nt = noise_type;
        const DecompositionTypes dt = decomp_type;
        if (last_site_1site && (i == 0 || i == me->n_sites - 1) && me->dot == 1)
            throw std::runtime_error("DMRGSCI: last_site_1site should only be "
                                     "used in two site algorithm.");
        const auto last_site_1_and_forward =
            last_site_1site && forward && i == me->n_sites - 2;
        const auto last_site_1_and_backward =
            last_site_1site && !forward && i == me->n_sites - 2;
        if (last_site_1_and_forward) {
            assert(me->dot = 2);
            me->dot = 1;
            for (auto &xme : ext_mes)
                xme->dot = 1;
            me->ket->canonical_form[i] = 'K';
            davidson_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            DMRG<S>::blocking(i, forward, bond_dim, 0, davidson_conv_thrd);
            davidson_soft_max_iter = dsmi;
            i++;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        } else if (last_site_1_and_backward) {
            me->dot = 1;
            for (auto &xme : ext_mes)
                xme->dot = 1;
            i = me->n_sites - 1;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        }
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            davidson_soft_max_iter = 0;
            if (noise_type & NoiseTypes::DensityMatrix)
                noise_type = NoiseTypes::Wavefunction;
            else if (noise_type & NoiseTypes::Collected)
                noise_type = NoiseTypes((uint8_t)noise_type ^
                                        (uint8_t)NoiseTypes::Collected);
            decomp_type = DecompositionTypes::SVD;
        }
        Iteration r =
            DMRG<S>::blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            r.energies[0] = 0;
            davidson_soft_max_iter = dsmi;
            noise_type = nt;
            decomp_type = dt;
        }
        if (last_site_1site && forward && i == me->n_sites - 1) {
            me->dot = 2;
            me->center = me->n_sites - 2;
            for (auto &xme : ext_mes) {
                xme->dot = 2;
                xme->center = me->n_sites - 2;
            }
        } else if (last_site_1site && !forward && i == me->n_sites - 1) {
            assert(me->dot = 1);
            davidson_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            DMRG<S>::blocking(i - 1, forward, bond_dim, 0, davidson_conv_thrd);
            davidson_soft_max_iter = dsmi;
            me->envs[i - 1]->right_op_infos.clear();
            me->envs[i - 1]->right = nullptr;
            me->dot = 2;
            for (auto &xme : ext_mes) {
                xme->envs[i - 1]->right_op_infos.clear();
                xme->envs[i - 1]->right = nullptr;
                xme->dot = 2;
            }
            me->ket->canonical_form[i - 2] = 'C';
        }
        return r;
    }
};

// Linear equation with big site
template <typename S> struct LinearBigSite : Linear<S> {
    using Linear<S>::iprint;
    using Linear<S>::lme;
    using Linear<S>::rme;
    using Linear<S>::tme;
    using Linear<S>::minres_soft_max_iter;
    using Linear<S>::noise_type;
    using Linear<S>::decomp_type;
    using Linear<S>::targets;
    using typename Linear<S>::Iteration;
    bool last_site_svd = false;
    bool last_site_1site = false; // ATTENTION: only use in two site algorithm
    LinearBigSite(const shared_ptr<MovingEnvironment<S>> &rme,
                  const vector<ubond_t> &bra_bond_dims,
                  const vector<ubond_t> &ket_bond_dims,
                  const vector<double> &noises = vector<double>())
        : Linear<S>(rme, bra_bond_dims, ket_bond_dims, noises) {}
    LinearBigSite(const shared_ptr<MovingEnvironment<S>> &lme,
                  const shared_ptr<MovingEnvironment<S>> &rme,
                  const vector<ubond_t> &bra_bond_dims,
                  const vector<ubond_t> &ket_bond_dims,
                  const vector<double> &noises = vector<double>())
        : Linear<S>(lme, rme, bra_bond_dims, ket_bond_dims, noises) {}
    LinearBigSite(const shared_ptr<MovingEnvironment<S>> &lme,
                  const shared_ptr<MovingEnvironment<S>> &rme,
                  const shared_ptr<MovingEnvironment<S>> &tme,
                  const vector<ubond_t> &bra_bond_dims,
                  const vector<ubond_t> &ket_bond_dims,
                  const vector<double> &noises = vector<double>())
        : Linear<S>(lme, rme, tme, bra_bond_dims, ket_bond_dims, noises) {}
    Iteration blocking(int i, bool forward, ubond_t bra_bond_dim,
                       ubond_t ket_bond_dim, double noise,
                       double minres_conv_thrd) override {
        const shared_ptr<MovingEnvironment<S>> &me = rme;
        const int dsmi =
            minres_soft_max_iter; // Save it as it may be changed here
        const NoiseTypes nt = noise_type;
        const DecompositionTypes dt = decomp_type;
        if (last_site_1site && (i == 0 || i == me->n_sites - 1) &&
            me->dot == 1) {
            throw std::runtime_error(
                "LinearBigSite: last_site_1site should only be "
                "used in two site algorithm.");
        }
        const auto last_site_1_and_forward =
            last_site_1site && forward && i == me->n_sites - 2;
        const auto last_site_1_and_backward =
            last_site_1site && !forward && i == me->n_sites - 2;
        if (last_site_1_and_forward) {
            assert(me->dot = 2);
            me->dot = 1;
            me->move_to(i);
            if (me->ket->canonical_form[i] == 'C') {
                me->ket->canonical_form[i] = 'K';
                me->ket->move_right(me->mpo->tf->opf->cg, me->para_rule);
            }
            if (lme != nullptr) {
                lme->dot = 1;
                lme->move_to(i);
                if (lme->ket->canonical_form[i] == 'C') {
                    lme->ket->canonical_form[i] = 'K';
                    lme->ket->move_right(lme->mpo->tf->opf->cg, lme->para_rule);
                }
            }
            if (tme != nullptr) {
                tme->dot = 1;
                tme->move_to(i);
                if (tme->ket->canonical_form[i] == 'C') {
                    tme->ket->canonical_form[i] = 'K';
                    tme->ket->move_right(tme->mpo->tf->opf->cg, tme->para_rule);
                }
                if (tme->bra->canonical_form[i] == 'C') {
                    tme->bra->canonical_form[i] = 'K';
                    tme->bra->move_right(tme->mpo->tf->opf->cg, tme->para_rule);
                }
            }
            i++;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        } else if (last_site_1_and_backward) {
            me->dot = 1;
            if (lme != nullptr)
                lme->dot = 1;
            if (tme != nullptr)
                tme->dot = 1;
            i = me->n_sites - 1;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        }
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            minres_soft_max_iter = 0;
            if (noise_type & NoiseTypes::DensityMatrix)
                noise_type = NoiseTypes::Wavefunction;
            else if (noise_type & NoiseTypes::Collected)
                noise_type = NoiseTypes((uint8_t)noise_type ^
                                        (uint8_t)NoiseTypes::Collected);
            decomp_type = DecompositionTypes::SVD;
        }
        Iteration r = Linear<S>::blocking(
            i, forward, bra_bond_dim, ket_bond_dim, noise, minres_conv_thrd);
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            if (targets.size() != 0)
                r.targets = targets.back();
            minres_soft_max_iter = dsmi;
            noise_type = nt;
            decomp_type = dt;
        }
        if (last_site_1site && forward && i == me->n_sites - 1) {
            me->dot = 2;
            me->center = me->n_sites - 2;
            if (lme != nullptr) {
                lme->dot = 2;
                lme->center = me->n_sites - 2;
            }
            if (tme != nullptr) {
                tme->dot = 2;
                tme->center = me->n_sites - 2;
            }
        } else if (last_site_1site && !forward && i == me->n_sites - 1) {
            assert(me->dot = 1);
            minres_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            Linear<S>::blocking(i - 1, forward, bra_bond_dim, ket_bond_dim, 0,
                                minres_conv_thrd);
            minres_soft_max_iter = dsmi;
            me->envs[i - 1]->right_op_infos.clear();
            me->envs[i - 1]->right = nullptr;
            me->dot = 2;
            me->ket->canonical_form[i - 2] = 'C';
            if (lme != nullptr) {
                lme->envs[i - 1]->right_op_infos.clear();
                lme->envs[i - 1]->right = nullptr;
                lme->dot = 2;
                lme->ket->canonical_form[i - 2] = 'C';
            }
            if (tme != nullptr) {
                tme->envs[i - 1]->right_op_infos.clear();
                tme->envs[i - 1]->right = nullptr;
                tme->dot = 2;
                tme->ket->canonical_form[i - 2] = 'C';
            }
        }
        return r;
    }
};

template <typename S> struct DMRGBigSiteAQCC : DMRGBigSite<S> {
    using DMRGBigSite<S>::iprint;
    using DMRGBigSite<S>::me;
    using DMRGBigSite<S>::ext_mes;
    using DMRGBigSite<S>::davidson_soft_max_iter;
    using DMRGBigSite<S>::davidson_max_iter;
    using DMRGBigSite<S>::noise_type;
    using DMRGBigSite<S>::decomp_type;
    using DMRGBigSite<S>::energies;
    using DMRGBigSite<S>::sweep_energies;
    using DMRGBigSite<S>::last_site_svd;
    using DMRGBigSite<S>::last_site_1site;
    using DMRGBigSite<S>::_t;
    using DMRGBigSite<S>::teff;
    using DMRGBigSite<S>::teig;
    using DMRGBigSite<S>::tprt;
    using DMRGBigSite<S>::sweep_max_eff_ham_size;

    double g_factor = 1.0;  // G in +Q formula
    double g_factor2 = 0.0; // G2 in ACPF2
    bool ACPF2_mode = false;
    bool RAS_mode = false;   // 2 sites at both ends: Thawed orbitals
    double ref_energy = 1.0; // Typically CAS-SCF/Reference energy of CAS
    double delta_e =
        0.0; // energy - ref_energy => will be modified during the sweep
    int max_aqcc_iter = 5; // Max iter spent on last site. Convergence depends
                           // on davidson conv. Note that this does not need to
                           // be fully converged as we do sweeps anyways.
    double smallest_energy =
        numeric_limits<double>::max(); // Smallest energy during sweep

    /** Frozen/CAS mode: Only one big site at the end
     * => ME + S * SME  **/
    DMRGBigSiteAQCC(const shared_ptr<MovingEnvironment<S>> &me, double g_factor,
                    const shared_ptr<MovingEnvironment<S>> &sme,
                    const vector<ubond_t> &bond_dims,
                    const vector<double> &noises, double ref_energy)
        : DMRGBigSite<S>(me, bond_dims, noises),
          // vv weird compile error -> cannot find member types -.-
          //     last_site_svd{true}, last_site_1site{true},
          g_factor{g_factor}, ref_energy{ref_energy} {
        last_site_svd = true;
        last_site_1site = me->dot == 2;

        ext_mes.push_back(sme);
    }

    /** Frozen/CAS mode ACPF2: Only one big site at the end
     * => ME + S * SME + S2 * SME2 **/
    DMRGBigSiteAQCC(const shared_ptr<MovingEnvironment<S>> &me, double g_factor,
                    const shared_ptr<MovingEnvironment<S>> &sme,
                    double g_factor2,
                    const shared_ptr<MovingEnvironment<S>> &sme2,
                    const vector<ubond_t> &bond_dims,
                    const vector<double> &noises, double ref_energy)
        : DMRGBigSite<S>(me, bond_dims, noises),
          // vv weird compile error -> cannot find member types -.-
          //     last_site_svd{true}, last_site_1site{true},
          g_factor{g_factor}, g_factor2{g_factor2}, ACPF2_mode{true},
          ref_energy{ref_energy} {
        last_site_svd = true;
        last_site_1site = me->dot == 2;

        ext_mes.push_back(sme);
        ext_mes.push_back(sme2);
    }

    /** RAS mode: Big sites on both ends
     * => ME + S * (SME1 - SME2)  **/
    DMRGBigSiteAQCC(const shared_ptr<MovingEnvironment<S>> &me, double g_factor,
                    const shared_ptr<MovingEnvironment<S>> &sme1,
                    const shared_ptr<MovingEnvironment<S>> &sme2,
                    const vector<ubond_t> &bond_dims,
                    const vector<double> &noises, double ref_energy)
        : DMRGBigSite<S>(me, bond_dims, noises),
          // vv weird compile error -> cannot find member types -.-
          //     last_site_svd{true}, last_site_1site{true},
          g_factor{g_factor}, ref_energy{ref_energy}, RAS_mode{true} {
        last_site_svd = true;
        last_site_1site = me->dot == 2;

        ext_mes.push_back(sme1);
        ext_mes.push_back(sme2);
    }

    /** RAS ACPF2 mode: Big sites on both ends
     * => ME + S * (SME1 - SME2) + S2 * (SME3-SME4)  **/
    DMRGBigSiteAQCC(const shared_ptr<MovingEnvironment<S>> &me, double g_factor,
                    const shared_ptr<MovingEnvironment<S>> &sme1,
                    const shared_ptr<MovingEnvironment<S>> &sme2,
                    double g_factor2,
                    const shared_ptr<MovingEnvironment<S>> &sme3,
                    const shared_ptr<MovingEnvironment<S>> &sme4,
                    const vector<ubond_t> &bond_dims,
                    const vector<double> &noises, double ref_energy)
        : DMRGBigSite<S>(me, bond_dims, noises),
          // vv weird compile error -> cannot find member types -.-
          //     last_site_svd{true}, last_site_1site{true},
          g_factor{g_factor}, g_factor2{g_factor2}, ACPF2_mode{true},
          ref_energy{ref_energy}, RAS_mode{true} {
        last_site_svd = true;
        last_site_1site = me->dot == 2;

        ext_mes.push_back(sme1);
        ext_mes.push_back(sme2);
        ext_mes.push_back(sme3);
        ext_mes.push_back(sme4);
    }

    shared_ptr<LinearEffectiveHamiltonian<S>>
    get_aqcc_eff(shared_ptr<EffectiveHamiltonian<S>> h_eff,
                 shared_ptr<EffectiveHamiltonian<S>> d_eff1,
                 shared_ptr<EffectiveHamiltonian<S>> d_eff2,
                 shared_ptr<EffectiveHamiltonian<S>> d_eff3,
                 shared_ptr<EffectiveHamiltonian<S>> d_eff4) {
        const auto shift = (1. - g_factor) * delta_e;
        const auto shift2 = (1. - g_factor2) * delta_e;
        shared_ptr<LinearEffectiveHamiltonian<S>> aqcc_eff;
        if (not RAS_mode) {
            aqcc_eff = h_eff + shift * d_eff1;
            if (ACPF2_mode) {
                aqcc_eff = h_eff + shift * d_eff1 + shift2 * d_eff2;
            } else {
                aqcc_eff = h_eff + shift * d_eff1;
            }
        } else {
            if (ACPF2_mode) {
                aqcc_eff = h_eff + shift * (d_eff1 - d_eff2) +
                           shift2 * (d_eff3 - d_eff4);
            } else {
                aqcc_eff = h_eff + shift * (d_eff1 - d_eff2);
            }
        }
        sweep_max_eff_ham_size =
            max(sweep_max_eff_ham_size, aqcc_eff->get_op_total_memory());
        return aqcc_eff;
    }

    tuple<double, int, size_t, double> two_dot_eigs_and_perturb(
        const bool forward, const int i, const double davidson_conv_thrd,
        const double noise, shared_ptr<SparseMatrixGroup<S>> &pket) override {
        tuple<double, int, size_t, double> pdi;
        _t.get_time();
        shared_ptr<EffectiveHamiltonian<S>> d_eff1, d_eff2, d_eff3, d_eff4;
        d_eff1 =
            ext_mes.at(0)->eff_ham(FuseTypes::FuseLR, forward, true,
                                   me->bra->tensors[i], me->ket->tensors[i]);
        if (RAS_mode or ACPF2_mode) {
            d_eff2 = ext_mes.at(1)->eff_ham(FuseTypes::FuseLR, forward, true,
                                            me->bra->tensors[i],
                                            me->ket->tensors[i]);
        }
        if (RAS_mode and ACPF2_mode) {
            d_eff3 = ext_mes.at(2)->eff_ham(FuseTypes::FuseLR, forward, true,
                                            me->bra->tensors[i],
                                            me->ket->tensors[i]);
            d_eff4 = ext_mes.at(3)->eff_ham(FuseTypes::FuseLR, forward, true,
                                            me->bra->tensors[i],
                                            me->ket->tensors[i]);
        }
        // h_eff needs to be done *last* for 3idx stuff
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, forward, true, me->bra->tensors[i],
                        me->ket->tensors[i]);
        auto aqcc_eff = get_aqcc_eff(h_eff, d_eff1, d_eff2, d_eff3, d_eff4);
        teff += _t.get_time();
        // TODO: For RAS mode, it might be good to do several iterations
        //       for the first site as well.
        pdi = aqcc_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                             davidson_soft_max_iter, DavidsonTypes::Normal, 0.0,
                             me->para_rule);
        teig += _t.get_time();
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0)
            pket = h_eff->perturbative_noise(forward, i, i + 1,
                                             FuseTypes::FuseLR, me->ket->info,
                                             noise_type, me->para_rule);
        tprt += _t.get_time();
        h_eff->deallocate();
        for (auto &d_eff : {d_eff4, d_eff3, d_eff2, d_eff1}) {
            if (d_eff != nullptr) {
                d_eff->deallocate();
            }
        }
        const auto energy = std::get<0>(pdi) + me->mpo->const_e;
        smallest_energy = min(energy, smallest_energy);
        delta_e = smallest_energy - ref_energy;
        return pdi;
    }
    tuple<double, int, size_t, double>
    one_dot_eigs_and_perturb(const bool forward, const bool fuse_left,
                             const int i_site, const double davidson_conv_thrd,
                             const double noise,
                             shared_ptr<SparseMatrixGroup<S>> &pket) override {
        tuple<double, int, size_t, double> pdi{0., 0, 0,
                                               0.}; // energy, ndav, nflop, tdav
        _t.get_time();
        const auto doAQCC = (i_site == me->n_sites - 1 or i_site == 0) and
                            abs(davidson_soft_max_iter) > 0;
        shared_ptr<EffectiveHamiltonian<S>> d_eff1, d_eff2, d_eff3, d_eff4;
        d_eff1 = ext_mes.at(0)->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
            me->bra->tensors[i_site], me->ket->tensors[i_site]);
        if (RAS_mode or ACPF2_mode) {
            d_eff2 = ext_mes.at(1)->eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
                me->bra->tensors[i_site], me->ket->tensors[i_site]);
        }
        if (RAS_mode and ACPF2_mode) {
            d_eff3 = ext_mes.at(2)->eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
                me->bra->tensors[i_site], me->ket->tensors[i_site]);
            d_eff4 = ext_mes.at(3)->eff_ham(
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
                me->bra->tensors[i_site], me->ket->tensors[i_site]);
        }
        // h_eff needs to be done *last* for 3idx stuff
        shared_ptr<EffectiveHamiltonian<S>> h_eff = me->eff_ham(
            fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, forward, true,
            me->bra->tensors[i_site], me->ket->tensors[i_site]);
        teff += _t.get_time();
        if (doAQCC) {
            // AQCC
            // Here, we actually do several iterations
            // as the last site with virt. space should be most important.
            if (sweep_energies.size() > 0) {
                // vv taken from DRMG::sweep
                size_t idx =
                    min_element(
                        sweep_energies.begin(), sweep_energies.end(),
                        [](const vector<double> &x, const vector<double> &y) {
                            return x[0] < y[0];
                        }) -
                    sweep_energies.begin();
                smallest_energy =
                    min(sweep_energies[idx].at(0), smallest_energy);
                delta_e = smallest_energy - ref_energy;
            }
            double last_delta_e = delta_e;
            if (iprint >= 2) {
                cout << endl;
            }
            for (int itAQCC = 0; itAQCC < max_aqcc_iter; ++itAQCC) {
                //
                // Shift non-reference ops
                //
                auto aqcc_eff =
                    get_aqcc_eff(h_eff, d_eff1, d_eff2, d_eff3, d_eff4);
                //
                // EIG and conv check
                //
                // TODO The best would be to do the adaption of the diagonal
                // directly in eigs
                const auto pdi2 =
                    aqcc_eff->eigs(iprint >= 3, davidson_conv_thrd,
                                   davidson_max_iter, davidson_soft_max_iter,
                                   DavidsonTypes::Normal, 0.0, me->para_rule);
                const auto energy = std::get<0>(pdi2) + me->mpo->const_e;
                const auto ndav = std::get<1>(pdi2);
                std::get<0>(pdi) = std::get<0>(pdi2);
                std::get<1>(pdi) += std::get<1>(pdi2); // ndav
                std::get<2>(pdi) += std::get<2>(pdi2); // nflop
                std::get<3>(pdi) += std::get<3>(pdi2); // tdav
                auto converged = smallest_energy < energy;
                if (not converged) {
                    smallest_energy = energy;
                    // convergence can be loosely defined here
                    converged = abs(delta_e - last_delta_e) / abs(delta_e) <
                                1.1 * max(davidson_conv_thrd, noise);
                }
                delta_e = smallest_energy - ref_energy;
                if (iprint >= 2) {
                    cout << "\tAQCC: " << setw(2) << itAQCC << " E=" << fixed
                         << setw(17) << setprecision(10) << energy
                         << " Delta=" << fixed << setw(17) << setprecision(10)
                         << delta_e << " nDav=" << setw(3) << ndav
                         << " conv=" << (converged ? "T" : "F");
                    if (itAQCC == 0) {
                        cout << "; init Delta=" << fixed << setw(17)
                             << setprecision(10) << last_delta_e << endl;
                    } else {
                        cout << endl;
                    }
                }
                last_delta_e = delta_e;
                if (converged) {
                    break;
                }
            }
            // vv restore printing
            if (iprint >= 2) {
                if (last_site_1site) {
                    cout << (forward ? " -->" : " <--") << " Site = " << setw(4)
                         << i_site << " LAST .. ";
                } else {
                    cout << (forward ? " -->" : " <--") << " Site = " << setw(4)
                         << i_site << " .. ";
                }
                cout.flush();
            }
        } else {
            auto aqcc_eff = get_aqcc_eff(h_eff, d_eff1, d_eff2, d_eff3, d_eff4);
            pdi = aqcc_eff->eigs(iprint >= 3, davidson_conv_thrd,
                                 davidson_max_iter, davidson_soft_max_iter,
                                 DavidsonTypes::Normal, 0.0, me->para_rule);
            const auto energy = std::get<0>(pdi) + me->mpo->const_e;
            smallest_energy = min(energy, smallest_energy);
            delta_e = smallest_energy - ref_energy;
        }
        teig += _t.get_time();
        if ((noise_type & NoiseTypes::Perturbative) && noise != 0) {
            pket = h_eff->perturbative_noise(
                forward, i_site, i_site,
                fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR, me->ket->info,
                noise_type, me->para_rule);
        }
        tprt += _t.get_time();
        h_eff->deallocate();
        for (auto &d_eff : {d_eff4, d_eff3, d_eff2, d_eff1}) {
            if (d_eff != nullptr) {
                d_eff->deallocate();
            }
        }
        return pdi;
    }
};

} // namespace block2
