
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

#include "../core/allocator.hpp"
#include "../core/matrix.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/symmetry.hpp"
#include "general_mpo.hpp"
#include "general_npdm.hpp"
#include "moving_environment.hpp"
#include "mpo.hpp"
#include "mpo_simplification.hpp"
#include "mps.hpp"
#include "sweep_algorithm.hpp"
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace block2 {

// Simple interface for DMRG calculations
template <typename S, typename FL> struct DMRGDriver {
    typedef typename GMatrix<FL>::FP FP;
    bool clean_scratch;
    S vacuum, target, left_vacuum;
    shared_ptr<GeneralHamiltonian<S, FL>> ghamil;
    string pg; // point group name
    shared_ptr<ParallelRule<S, FL>> prule = nullptr;
    DMRGDriver(size_t stack_mem, string scratch, string restart_dir = "",
               int n_threads = -1, bool clean_scratch = true,
               double stack_mem_ratio = 0.4)
        : clean_scratch(clean_scratch) {
        if (frame_<FP>() == nullptr)
            frame_<FP>() = make_shared<DataFrame<FP>>(
                (size_t)(0.1 * stack_mem), (size_t)(0.9 * stack_mem), scratch,
                stack_mem_ratio, stack_mem_ratio);
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        frame_<FP>()->fp_codec = make_shared<FPCodec<FP>>(1E-16, 1024);
        if (restart_dir != "") {
            if (!Parsing::path_exists(restart_dir))
                Parsing::mkdir(restart_dir);
            frame_<FP>()->restart_dir = restart_dir;
        }
    }
    ~DMRGDriver() { frame_<FP>() = nullptr; }
    void initialize_system(
        int n_sites, int n_elec, int spin, int pg_irrep = -1,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int heis_twos = -1, int heis_twosz = 0, bool singlet_embedding = true) {
        vacuum = S(0, 0, 0);
        if (heis_twos != -1 && is_same<S, SU2>::value && n_elec == 0)
            n_elec = n_sites * heis_twos;
        else if (heis_twos == -1 && is_same<S, SGB>::value && n_elec != 0)
            n_elec = 2 * n_elec - n_sites;
        if (pg_irrep == -1)
            pg_irrep = 0;
        if (!is_same<S, SU2>::value || heis_twos != -1)
            singlet_embedding = false;
        if (singlet_embedding) {
            if (heis_twosz != 0)
                throw runtime_error(
                    "Singlet embedding only works for heis_twosz == 0!");
            target = S(n_elec + spin % 2, 0, pg_irrep);
            left_vacuum = S(spin % 2, spin, 0);
        } else {
            target = S(heis_twosz == 0 ? n_elec : heis_twosz, spin, pg_irrep);
            left_vacuum = vacuum;
        }
        ghamil = make_shared<GeneralHamiltonian<S, FL>>(
            vacuum, n_sites,
            orb_sym.size() == 0 ? vector<typename S::pg_t>(n_sites, 0)
                                : orb_sym,
            heis_twos);
    }
    shared_ptr<GeneralFCIDUMP<FL>> expr_builder() const {
        shared_ptr<GeneralFCIDUMP<FL>> b = make_shared<GeneralFCIDUMP<FL>>();
        if (is_same<S, SU2>::value)
            b->elem_type = ElemOpTypes::SU2;
        else if (is_same<S, SZ>::value)
            b->elem_type = ElemOpTypes::SZ;
        else if (is_same<S, SGF>::value)
            b->elem_type = ElemOpTypes::SGF;
        else if (is_same<S, SGB>::value)
            b->elem_type = ElemOpTypes::SGB;
        else
            throw runtime_error("Unknown symmetry type!");
        b->const_e = (FL)0.0;
        return b;
    }
    shared_ptr<MPO<S, FL>> get_spin_square_mpo(int iprint = 1) const {
        shared_ptr<GeneralFCIDUMP<FL>> b = expr_builder();
        size_t n = (size_t)ghamil->n_sites;
        if (is_same<S, SU2>::value) {
            b->exprs.push_back("((C+D)2+(C+D)2)0");
            b->indices.push_back(vector<uint16_t>(n * n * 4));
            b->data.push_back(vector<FL>(n * n));
            for (uint16_t i = 0; i < (uint16_t)n; i++)
                for (uint16_t j = 0; j < (uint16_t)n; j++) {
                    b->data.back()[i * n + j] = (FL)(-sqrt(3.0)) / (FL)2.0;
                    b->indices.back()[(i * n + j) * 4 + 0] = i;
                    b->indices.back()[(i * n + j) * 4 + 1] = i;
                    b->indices.back()[(i * n + j) * 4 + 2] = j;
                    b->indices.back()[(i * n + j) * 4 + 3] = j;
                }
        } else if (is_same<S, SZ>::value) {
            const vector<string> cds = vector<string>{"cd", "CD"};
            for (size_t icd = 0; icd < cds.size(); icd++) {
                b->exprs.push_back(cds[icd]);
                b->indices.push_back(vector<uint16_t>(n * 2));
                b->data.push_back(vector<FL>(n));
                for (uint16_t i = 0; i < (uint16_t)n; i++) {
                    b->data.back()[i] = (FL)0.75;
                    b->indices.back()[i + i + 0] = i;
                    b->indices.back()[i + i + 1] = i;
                }
            }
            const vector<string> xcds =
                vector<string>{"ccdd", "cCDd", "CcdD", "CCDD", "cCDd", "CcdD"};
            const vector<FL> vals = vector<FL>{(FL)0.25, (FL)-0.25, (FL)-0.25,
                                               (FL)0.25, (FL)-0.5,  (FL)-0.5};
            const vector<uint8_t> idxs =
                vector<uint8_t>{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                                0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};
            for (size_t icd = 0; icd < xcds.size(); icd++) {
                b->exprs.push_back(xcds[icd]);
                b->indices.push_back(vector<uint16_t>(n * n * 4));
                b->data.push_back(vector<FL>(n * n));
                array<uint16_t, 2> ij;
                for (ij[0] = 0; ij[0] < (uint16_t)n; ij[0]++)
                    for (ij[1] = 0; ij[1] < (uint16_t)n; ij[1]++) {
                        b->data.back()[ij[0] * n + ij[1]] = vals[icd];
                        b->indices.back()[(ij[0] * n + ij[1]) * 4 + 0] =
                            ij[idxs[icd * 4 + 0]];
                        b->indices.back()[(ij[0] * n + ij[1]) * 4 + 1] =
                            ij[idxs[icd * 4 + 1]];
                        b->indices.back()[(ij[0] * n + ij[1]) * 4 + 2] =
                            ij[idxs[icd * 4 + 2]];
                        b->indices.back()[(ij[0] * n + ij[1]) * 4 + 3] =
                            ij[idxs[icd * 4 + 3]];
                    }
            }
        }
        return get_mpo(b->adjust_order(), iprint);
    }
    shared_ptr<MPO<S, FL>> get_mpo(
        shared_ptr<GeneralFCIDUMP<FL>> expr, int iprint, FP cutoff = (FP)1E-14,
        MPOAlgorithmTypes algo_type = MPOAlgorithmTypes::FastBipartite) const {
        shared_ptr<MPO<S, FL>> mpo = make_shared<GeneralMPO<S, FL>>(
            ghamil, expr, algo_type, cutoff, -1, max(0, iprint - 1));
        mpo->build();
        mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<Rule<S, FL>>(),
                                                false, false);
        mpo = make_shared<IdentityAddedMPO<S, FL>>(mpo);
        return mpo;
    }
    shared_ptr<MPS<S, FL>>
    get_random_mps(const string &tag, ubond_t bond_dim = 500, int center = 0,
                   int dot = 2, S target = S(S::invalid), int nroots = 1,
                   const vector<double> &occs = vector<double>(),
                   bool full_fci = true, S left_vacuum = S(S::invalid)) const {
        if (target == S(S::invalid))
            target = this->target;
        if (left_vacuum == S(S::invalid))
            left_vacuum = this->left_vacuum;
        shared_ptr<MPS<S, FL>> mps = nullptr;
        if (nroots == 1) {
            mps = make_shared<MPS<S, FL>>(ghamil->n_sites, center, dot);
            mps->info = make_shared<MPSInfo<S>>(ghamil->n_sites, ghamil->vacuum,
                                                target, ghamil->basis);
        } else {
            vector<S> targets = vector<S>(1, target);
            mps = make_shared<MultiMPS<S, FL>>(ghamil->n_sites, center, dot,
                                               nroots);
            mps->info = make_shared<MultiMPSInfo<S>>(
                ghamil->n_sites, ghamil->vacuum, targets, ghamil->basis);
        }
        mps->info->tag = tag;
        if (full_fci)
            mps->info->set_bond_dimension_full_fci(left_vacuum, vacuum);
        else
            mps->info->set_bond_dimension_fci(left_vacuum, vacuum);
        if (occs.size() != 0)
            mps->info->set_bond_dimension_using_occ(bond_dim, occs);
        else
            mps->info->set_bond_dimension(bond_dim);
        mps->info->bond_dim = bond_dim;
        mps->initialize(mps->info);
        mps->random_canonicalize();
        if (nroots == 1)
            mps->tensors[mps->center]->normalize();
        else
            for (auto &xwfn : dynamic_pointer_cast<MultiMPS<S, FL>>(mps)->wfns)
                xwfn->normalize();
        mps->save_mutable();
        mps->info->save_mutable();
        mps->save_data();
        mps->info->save_data(frame_<FP>()->save_dir + "/" + tag +
                             "-mps_info.bin");
        return mps;
    }
    shared_ptr<MPS<S, FL>> split_mps(shared_ptr<MPS<S, FL>> ket, int iroot,
                                     const string &tag) const {
        if (prule != nullptr)
            prule->comm->barrier();
        if (ket->get_type() != MPSTypes::MultiWfn)
            throw runtime_error("Can only split State-averaged MPS!");
        shared_ptr<MultiMPS<S, FL>> mket =
            dynamic_pointer_cast<MultiMPS<S, FL>>(ket);
        shared_ptr<MultiMPSInfo<S>> minfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(mket->info);
        shared_ptr<MultiMPS<S, FL>> iket = mket->extract(
            iroot, minfo->targets.size() == 1 ? tag + "@TMP" : tag);
        if (prule != nullptr)
            prule->comm->barrier();
        shared_ptr<MPS<S, FL>> jket;
        if (minfo->targets.size() == 1)
            jket = iket->make_single(tag);
        else
            jket = iket;
        if (prule != nullptr)
            prule->comm->barrier();
        jket = adjust_mps(jket).first;
        jket->info->save_data(frame_<FP>()->save_dir + "/" + tag +
                              "-mps_info.bin");
        return jket;
    }
    pair<shared_ptr<MPS<S, FL>>, bool> adjust_mps(shared_ptr<MPS<S, FL>> ket,
                                                  int dot = -1) const {
        if (dot == -1)
            dot = ket->dot;
        if (ket->center == 0 && dot == 2) {
            if (prule != nullptr)
                prule->comm->barrier();
            if (string("ST").find(ket->canonical_form[ket->center]) !=
                string::npos)
                ket->flip_fused_form(ket->center, ghamil->opf->cg, prule);
            ket->save_data();
            if (prule != nullptr)
                prule->comm->barrier();
            ket->load_mutable();
            ket->info->load_mutable();
            if (prule != nullptr)
                prule->comm->barrier();
        }
        ket->dot = dot;
        bool forward = ket->center == 0;
        if (ket->canonical_form[ket->center] == 'L' &&
            ket->center != ket->n_sites - ket->dot)
            ket->center += 1, forward = true;
        else if ((ket->canonical_form[ket->center] == 'C' ||
                  ket->canonical_form[ket->center] == 'M') &&
                 ket->center != 0)
            ket->center -= 1, forward = false;
        if (ket->canonical_form[ket->center] == 'M' &&
            ket->get_type() != MPSTypes::MultiWfn)
            ket->canonical_form[ket->center] = 'C';
        if (ket->canonical_form[ket->n_sites - 1] == 'M' &&
            ket->get_type() != MPSTypes::MultiWfn)
            ket->canonical_form[ket->n_sites - 1] = 'C';
        if (dot == 1) {
            if (ket->canonical_form[0] == 'C' && ket->canonical_form[1] == 'R')
                ket->canonical_form[0] = 'K';
            else if (ket->canonical_form[ket->n_sites - 1] == 'C' &&
                     ket->canonical_form[ket->n_sites - 2] == 'L') {
                ket->canonical_form[ket->n_sites - 1] = 'S';
                ket->center = ket->n_sites - 1;
            } else if (ket->center == ket->n_sites - 2 &&
                       ket->canonical_form[ket->n_sites - 2] == 'L')
                ket->center = ket->n_sites - 1;
            if (ket->canonical_form[0] == 'M' && ket->canonical_form[1] == 'R')
                ket->canonical_form[0] = 'J';
            else if (ket->canonical_form[ket->n_sites - 1] == 'M' &&
                     ket->canonical_form[ket->n_sites - 2] == 'L') {
                ket->canonical_form[ket->n_sites - 1] = 'T';
                ket->center = ket->n_sites - 1;
            } else if (ket->center == ket->n_sites - 2 &&
                       ket->canonical_form[ket->n_sites - 2] == 'L')
                ket->center = ket->n_sites - 1;
        }
        ket->save_data();
        if (prule != nullptr)
            prule->comm->barrier();
        return make_pair(ket, forward);
    }
    void fix_restarting_mps(shared_ptr<MPS<S, FL>> mps) const {
        shared_ptr<CG<S>> cg = ghamil->opf->cg;
        if (mps->canonical_form[mps->center] == 'L' &&
            mps->center != mps->n_sites - mps->dot) {
            mps->center += 1;
            if (string("ST").find(mps->canonical_form[mps->center]) !=
                    string::npos &&
                mps->dot == 2) {
                if (prule != nullptr)
                    prule->comm->barrier();
                mps->flip_fused_form(mps->center, cg, prule);
                mps->save_data();
                if (prule != nullptr)
                    prule->comm->barrier();
                mps->load_mutable();
                mps->info->load_mutable();
                if (prule != nullptr)
                    prule->comm->barrier();
            }
        } else if (string("CMKJST").find(mps->canonical_form[mps->center]) !=
                       string::npos &&
                   mps->center != 0) {
            if (string("KJ").find(mps->canonical_form[mps->center]) !=
                    string::npos &&
                mps->dot == 2) {
                if (prule != nullptr)
                    prule->comm->barrier();
                mps->flip_fused_form(mps->center, cg, prule);
                mps->save_data();
                if (prule != nullptr)
                    prule->comm->barrier();
                mps->load_mutable();
                mps->info->load_mutable();
                if (prule != nullptr)
                    prule->comm->barrier();
            }
            if (mps->canonical_form.substr(mps->center, 2) != "CC" &&
                mps->dot == 2)
                mps->center -= 1;
        } else if (mps->center == mps->n_sites - 1 && mps->dot == 2) {
            if (prule != nullptr)
                prule->comm->barrier();
            if (string("KJ").find(mps->canonical_form[mps->center]) !=
                string::npos)
                mps->flip_fused_form(mps->center, cg, prule);
            mps->center = mps->n_sites - 2;
            mps->save_data();
            if (prule != nullptr)
                prule->comm->barrier();
            mps->load_mutable();
            mps->info->load_mutable();
            if (prule != nullptr)
                prule->comm->barrier();
        } else if (mps->center == 0 && mps->dot == 2) {
            if (prule != nullptr)
                prule->comm->barrier();
            if (string("ST").find(mps->canonical_form[mps->center]) !=
                string::npos)
                mps->flip_fused_form(mps->center, cg, prule);
            mps->save_data();
            if (prule != nullptr)
                prule->comm->barrier();
            mps->load_mutable();
            mps->info->load_mutable();
            if (prule != nullptr)
                prule->comm->barrier();
        }
    }
    shared_ptr<MPS<S, FL>> load_mps(const string &tag, int nroots = 1) const {
        shared_ptr<MPSInfo<S>> mps_info;
        if (nroots == 1)
            mps_info = make_shared<MPSInfo<S>>(0);
        else
            mps_info = make_shared<MultiMPSInfo<S>>(0);
        if (Parsing::file_exists(frame_<FP>()->save_dir + "/" + tag +
                                 "-mps_info.bin"))
            mps_info->load_data(frame_<FP>()->save_dir + "/" + tag +
                                "-mps_info.bin");
        else
            mps_info->load_data(frame_<FP>()->save_dir + "/mps_info.bin");
        if (mps_info->tag != tag)
            throw runtime_error("Loading MPS with tag " + tag +
                                ", but only found MPS with tag " +
                                mps_info->tag + ".");
        mps_info->load_mutable();
        mps_info->bond_dim =
            max(mps_info->bond_dim, mps_info->get_max_bond_dimension());
        shared_ptr<MPS<S, FL>> mps;
        if (nroots == 1)
            mps = make_shared<MPS<S, FL>>(mps_info);
        else
            mps = make_shared<MultiMPS<S, FL>>(
                dynamic_pointer_cast<MultiMPSInfo<S>>(mps_info));
        mps->load_data();
        mps->load_mutable();
        fix_restarting_mps(mps);
        return mps;
    }
    vector<FL> dmrg(shared_ptr<MPO<S, FL>> mpo, shared_ptr<MPS<S, FL>> ket,
                    int n_sweeps = 10, FP tol = (FP)1E-8,
                    vector<ubond_t> bond_dims = vector<ubond_t>(),
                    vector<FP> noises = vector<FP>(),
                    vector<FP> thrds = vector<FP>(), int iprint = 0,
                    FP cutoff = (FP)1E-20, int dav_max_iter = 4000) const {
        if (bond_dims.size() == 0)
            bond_dims.push_back(ket->info->bond_dim);
        if (noises.size() == 0) {
            noises = vector<FP>(5, (FP)1E-5);
            noises.push_back(0.0);
        }
        if (thrds.size() == 0) {
            if (!is_same<FP, float>::value) {
                thrds = vector<FP>(4, (FP)1E-6);
                thrds.push_back((FP)1E-7);
            } else {
                thrds = vector<FP>(4, (FP)1E-5);
                thrds.push_back((FP)5E-6);
            }
        }
        shared_ptr<MPS<S, FL>> bra = ket;
        shared_ptr<MovingEnvironment<S, FL, FL>> me =
            make_shared<MovingEnvironment<S, FL, FL>>(mpo, bra, ket);
        me->delayed_contraction = OpNamesSet::normal_ops();
        me->cached_contraction = true;
        me->init_environments(iprint >= 2);
        shared_ptr<DMRG<S, FL, FL>> dx =
            make_shared<DMRG<S, FL, FL>>(me, bond_dims, noises);
        dx->noise_type = NoiseTypes::ReducedPerturbative;
        dx->davidson_conv_thrds = thrds;
        dx->davidson_max_iter = dav_max_iter + 100;
        dx->davidson_soft_max_iter = dav_max_iter;
        dx->iprint = iprint;
        dx->cutoff = cutoff;
        dx->trunc_type = dx->trunc_type | TruncationTypes::RealDensityMatrix;
        dx->solve(n_sweeps, ket->center == 0, tol);

        if (clean_scratch) {
            dx->me->remove_partition_files();
            for (auto &xme : dx->ext_mes)
                xme->remove_partition_files();
        }

        ket->info->bond_dim = max(ket->info->bond_dim, bond_dims.back());
        vector<FL> energies(dx->energies.back().size());
        for (size_t i = 0; i < dx->energies.back().size(); i++)
            energies[i] = (FL)dx->energies.back()[i];
        return energies;
    }
    FL expectation(shared_ptr<MPS<S, FL>> bra, shared_ptr<MPO<S, FL>> mpo,
                   shared_ptr<MPS<S, FL>> ket, bool iprint = 0) const {
        shared_ptr<MPS<S, FL>> mbra = bra->deep_copy("EXPE-BRA@TMP"),
                               mket = mbra;
        if (bra != ket)
            mket = ket->deep_copy("EXPE-KET@TMP");
        ubond_t bond_dim = max(mbra->info->bond_dim, mket->info->bond_dim);
        align_mps_center(mbra, mket);
        shared_ptr<MovingEnvironment<S, FL, FL>> me =
            make_shared<MovingEnvironment<S, FL, FL>>(mpo, mbra, mket, "EXPT");
        me->delayed_contraction = OpNamesSet::normal_ops();
        me->cached_contraction = true;
        me->init_environments(iprint >= 2);
        shared_ptr<Expect<S, FL, FL, FL>> dx =
            make_shared<Expect<S, FL, FL, FL>>(me, bond_dim, bond_dim);
        dx->iprint = iprint;
        FL ex = dx->solve(false, mket->center != 0);
        if (clean_scratch)
            me->remove_partition_files();
        if (prule != nullptr)
            prule->comm->barrier();
        return ex;
    }
    void align_mps_center(shared_ptr<MPS<S, FL>> ket,
                          shared_ptr<MPS<S, FL>> ref) const {
        ket->info->bond_dim =
            max(ket->info->bond_dim, ket->info->get_max_bond_dimension());
        if (ket->center != ref->center) {
            if (ref->center == 0) {
                if (ket->dot == 2) {
                    ket->center++;
                    ket->canonical_form[ket->n_sites - 1] = 'S';
                }
                while (ket->center != 0)
                    ket->move_left(ghamil->opf->cg, nullptr);
            } else {
                ket->canonical_form[0] = 'K';
                while (ket->center != ket->n_sites - 1)
                    ket->move_right(ghamil->opf->cg, nullptr);
                if (ket->dot == 2)
                    ket->center--;
            }
            ket->save_data();
            ket->info->save_data(frame_<FP>()->save_dir + "/" + ket->info->tag +
                                 "-mps_info.bin");
        }
    }
    vector<shared_ptr<GTensor<FL>>>
    get_npdm(const vector<string> &exprs, shared_ptr<MPS<S, FL>> ket,
             shared_ptr<MPS<S, FL>> bra, int site_type = 0,
             ExpectationAlgorithmTypes algo_type =
                 ExpectationAlgorithmTypes::SymbolFree |
                 ExpectationAlgorithmTypes::Compressed,
             int iprint = 0, FP cutoff = (FP)1E-24,
             bool fused_contraction_rotation = true) const {
        shared_ptr<MPS<S, FL>> mket = ket->deep_copy("PDM-KET@TMP"), mbra;
        vector<shared_ptr<MPS<S, FL>>> mpss =
            vector<shared_ptr<MPS<S, FL>>>(1, mket);
        if (bra != nullptr && bra != ket) {
            mbra = bra->deep_copy("PDM-BRA@TMP");
            mpss.push_back(mbra);
        } else
            mbra = mket;
        for (auto &mps : mpss) {
            if (mps->dot == 2 && site_type != 2) {
                mps->dot = 1;
                if (mps->center == mps->n_sites - 2) {
                    mps->center = mps->n_sites - 1;
                    mps->canonical_form[mps->n_sites - 1] = 'S';
                } else if (mps->center == 0)
                    mps->canonical_form[0] = 'K';
                else
                    assert(false);
                mps->save_data();
            }
            mps->load_mutable();
            mps->info->bond_dim =
                max(mps->info->bond_dim, mps->info->get_max_bond_dimension());
        }
        align_mps_center(mbra, mket);

        if (iprint >= 1)
            cout << endl;

        if (iprint >= 2) {
            cout << "BRA = " << mbra->canonical_form << " CT = " << mbra->center
                 << " DOT = " << mbra->dot << " Q = " << mbra->info->target
                 << endl;
            cout << "KET = " << mket->canonical_form << " CT = " << mket->center
                 << " DOT = " << mket->dot << " Q = " << mket->info->target
                 << endl;
        }

        vector<shared_ptr<SpinPermScheme>> perms;
        for (const string &op_str : exprs) {
            int n_cds = SpinPermRecoupling::count_cds(op_str);
            if (is_same<S, SU2>::value)
                perms.push_back(make_shared<SpinPermScheme>(
                    SpinPermScheme::initialize_su2(n_cds, op_str, true)));
            else
                perms.push_back(make_shared<SpinPermScheme>(
                    SpinPermScheme::initialize_sz(n_cds, op_str, true)));
            if (iprint >= 1)
                cout << "npdm string = " << op_str << endl;
        }

        if (iprint >= 3)
            for (auto &perm : perms)
                cout << perm->to_str() << endl;

        shared_ptr<NPDMScheme> scheme = make_shared<NPDMScheme>(perms);
        shared_ptr<GeneralNPDMMPO<S, FL>> ppmpo =
            make_shared<GeneralNPDMMPO<S, FL>>(
                ghamil, scheme,
                algo_type & ExpectationAlgorithmTypes::SymbolFree);
        ppmpo->iprint = iprint >= 4 ? 2 : (iprint >= 2 ? 1 : 0);
        ppmpo->build();

        shared_ptr<MPO<S, FL>> pmpo = make_shared<SimplifiedMPO<S, FL>>(
            ppmpo, make_shared<Rule<S, FL>>(), false, false);

        shared_ptr<MovingEnvironment<S, FL, FL>> pme =
            make_shared<MovingEnvironment<S, FL, FL>>(pmpo, mbra, mket, "NPDM");
        if (fused_contraction_rotation) {
            pme->cached_contraction = false;
            pme->fused_contraction_rotation = true;
        } else {
            pme->cached_contraction = true;
            pme->fused_contraction_rotation = false;
        }
        pme->init_environments(iprint >= 2);
        shared_ptr<Expect<S, FL, FL, FL>> dx =
            make_shared<Expect<S, FL, FL, FL>>(pme, mbra->info->bond_dim,
                                               mket->info->bond_dim);
        if (site_type == 0)
            dx->zero_dot_algo = true;
        dx->algo_type = algo_type;
        dx->iprint = iprint;
        dx->cutoff = cutoff;
        dx->solve(true, mket->center == 0);

        if (clean_scratch)
            pme->remove_partition_files();

        vector<shared_ptr<GTensor<FL>>> npdms = dx->get_npdm();

        if (is_same<S, SU2>::value)
            for (size_t i = 0; i < exprs.size(); i++) {
                int n_cds = SpinPermRecoupling::count_cds(exprs[i]);
                FL factor = (FL)1.0;
                for (int j = 0; j < n_cds; j++)
                    factor *= (FL)sqrt(sqrt((FL)2.0));
                for (size_t j = 0; j < npdms[i]->size(); j++)
                    (*npdms[i]->data)[j] *= factor;
            }

        return npdms;
    }
};

} // namespace block2
