
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2021 Seunghoon Lee <seunghoonlee89@gmail.com>
 * Copyright (C) 2021 Huanchen Zhai <hczhai@caltech.edu>
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

#include "../core/integral.hpp"
#include "../core/threading.hpp"
#include "../core/utils.hpp"
#include "../dmrg/mps.hpp"
#include "../dmrg/mps_unfused.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <stack>
#include <tuple>
#include <vector>

/** Stochastic perturbative DMRG.
 *
 * Author: Seunghoon Lee, 2021
 * Revised: Huanchen Zhai, Aug 13, 2021
 *    improved serial efficiency;
 *    added threading;
 *    added spin-adapted (Aug 20, 2021).
 */

using namespace std;

namespace block2 {

template <typename, typename, typename = void> struct StochasticPDMRG;

// stochastic perturbative DMRG
// JCP 148 21104 (2018), doi: 10.1063/1.5031140
template <typename S, typename FL>
struct StochasticPDMRG<S, FL, typename S::is_sz_t> {
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<SparseMatrix<S, FL>> left_psi0, left_qvpsi0;
    vector<shared_ptr<SparseTensor<S, FL>>> tensors_psi0, tensors_qvpsi0;
    FP norm_qvpsi0;
    vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos_psi0, pinfos_qvpsi0;
    int n_sites;
    uint8_t phys_dim;
    StochasticPDMRG() {}
    StochasticPDMRG(const shared_ptr<UnfusedMPS<S, FL>> &mps_psi0,
                    const shared_ptr<UnfusedMPS<S, FL>> &mps_qvpsi0,
                    const FP norm) {
        this->initialize(mps_psi0, mps_qvpsi0, norm);
    }
    // initialize
    void initialize(const shared_ptr<UnfusedMPS<S, FL>> &mps_psi0,
                    const shared_ptr<UnfusedMPS<S, FL>> &mps_qvpsi0,
                    const FP norm) {
        Random::rand_seed(0);
        n_sites = mps_psi0->n_sites;
        phys_dim = 4;

        tensors_psi0.resize(n_sites);
        tensors_qvpsi0.resize(n_sites);
        for (int i = 0; i < n_sites; i++) {
            tensors_psi0[i] = mps_psi0->tensors[i];
            tensors_qvpsi0[i] = mps_qvpsi0->tensors[i];
        }
        pinfos_psi0.resize(n_sites);
        pinfos_qvpsi0.resize(n_sites);
        gen_si_map(pinfos_psi0, mps_psi0);
        gen_si_map(pinfos_qvpsi0, mps_qvpsi0);

        norm_qvpsi0 = norm;
    }
    // generate stateinfo map
    void gen_si_map(vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos,
                    const shared_ptr<UnfusedMPS<S, FL>> &mps) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();

        pinfos.resize(n_sites + 1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(*mps->info->left_dims_fci[0],
                                 *mps->info->left_dims_fci[0],
                                 mps->info->vacuum, false);
        for (int d = 0; d < n_sites; d++) {
            pinfos[d + 1].resize(4);
            for (int j = 0; j < pinfos[d + 1].size(); j++) {
                map<S, MKL_INT> qkets;
                for (auto &m : mps->tensors[d]->data[j]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (!qkets.count(ket))
                        qkets[ket] = m.second->shape[2];
                }
                StateInfo<S> ibra, iket;
                ibra.allocate((int)qkets.size());
                iket.allocate((int)qkets.size());
                int k = 0;
                for (auto &qm : qkets) {
                    ibra.quanta[k] = iket.quanta[k] = qm.first;
                    ibra.n_states[k] = 1;
                    iket.n_states[k] = (ubond_t)qm.second;
                    k++;
                }
                pinfos[d + 1][j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                pinfos[d + 1][j]->initialize(ibra, iket, mps->info->vacuum,
                                             false);
            }
        }
    }
    // ityp == 0: sampling a determinant for C term
    // ityp == 1: sampling a determinant for A,B term
    FP sampling(int ityp, vector<uint8_t> &det_string) const {
        const vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos =
            ityp == 0 ? pinfos_psi0 : pinfos_qvpsi0;
        const vector<shared_ptr<SparseTensor<S, FL>>> &tensors =
            ityp == 0 ? tensors_psi0 : tensors_qvpsi0;

        shared_ptr<SparseMatrix<S, FL>> ptrs;

        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> initp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        initp->allocate(pinfos[0][0]);
        for (size_t j = 0; j < initp->total_memory; j++)
            initp->data[j] = 1.0;

        ptrs = initp;

        det_string.resize(2 * n_sites);
        vector<FP> rand(n_sites);
        Random::fill<FP>((FP *)rand.data(), n_sites);
        vector<FP> cp(phys_dim), accp(phys_dim + 1, 0);
        vector<shared_ptr<SparseMatrix<S, FL>>> ptrs_save(phys_dim);
        FP rnorm = 0;
        for (int i_site = 0; i_site < n_sites; i_site++) {
            for (uint8_t d = 0; d < phys_dim; d++) {
                shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
                shared_ptr<SparseMatrix<S, FL>> cmp =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
                cmp->allocate(pinfos[i_site + 1][d]);
                for (auto &m : tensors[i_site]->data[d]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                                   m.second->ref(), false,
                                                   (*cmp)[ket], 1.0, 1.0);
                }
                FP tmp = cmp->norm();
                cp[d] = tmp * tmp;
                ptrs_save[d] = cmp;
            }
            for (uint8_t d = 0; d < phys_dim; d++)
                accp[d + 1] = accp[d] + cp[d];
            for (uint8_t d = 0; d < phys_dim; d++) {
                accp[d + 1] /= accp[phys_dim];
                if (rand[i_site] < accp[d + 1]) {
                    ptrs = ptrs_save[d];
                    rnorm = cp[d];
                    det_string[2 * i_site] = d & 1;
                    det_string[2 * i_site + 1] = (d & 2) >> 1;
                    break;
                }
            }
        }
        return sqrt(rnorm);
    }
    // ityp == 0: <Psi^(0)|VQ|D>
    // ityp == 1: <Psi^(0)|D>
    FP overlap(int ityp, const vector<uint8_t> &det_string) const {
        const vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos =
            ityp == 1 ? pinfos_psi0 : pinfos_qvpsi0;
        const vector<shared_ptr<SparseTensor<S, FL>>> &tensors =
            ityp == 1 ? tensors_psi0 : tensors_qvpsi0;

        shared_ptr<SparseMatrix<S, FL>> ptrs;
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> initp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        initp->allocate(pinfos[0][0]);
        for (size_t j = 0; j < initp->total_memory; j++)
            initp->data[j] = 1.0;

        ptrs = initp;

        for (int i_site = 0; i_site < n_sites; i_site++) {
            int d = det_string[2 * i_site] + (det_string[2 * i_site + 1] << 1);
            shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
            shared_ptr<SparseMatrix<S, FL>> cmp =
                make_shared<SparseMatrix<S, FL>>(d_alloc);
            cmp->allocate(pinfos[i_site + 1][d]);
            for (auto &m : tensors[i_site]->data[d]) {
                S bra = m.first.first, ket = m.first.second;
                if (pmp->info->find_state(bra) == -1)
                    continue;
                GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                               m.second->ref(), false,
                                               (*cmp)[ket], 1.0, 1.0);
            }
            ptrs = cmp;
            if (i_site == n_sites - 1)
                return cmp->norm();
        }
        return 0;
    }
    void
    gen_tmp_mats(const vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos,
                 const vector<shared_ptr<SparseTensor<S, FL>>> &tensors,
                 vector<vector<shared_ptr<SparseMatrix<S, FL>>>> &pmats) const {
        pmats.resize(n_sites + 1);
        shared_ptr<VectorAllocator<FP>> dd_alloc =
            make_shared<VectorAllocator<FP>>();
        for (int i_site = 0; i_site < n_sites + 1; i_site++) {
            pmats[i_site].resize(pinfos[i_site].size());
            for (uint8_t d = 0; d < (uint8_t)pinfos[i_site].size(); d++) {
                pmats[i_site][d] = make_shared<SparseMatrix<S, FL>>(dd_alloc);
                pmats[i_site][d]->allocate(pinfos[i_site][d]);
            }
        }
        for (size_t j = 0; j < pmats[0][0]->total_memory; j++)
            pmats[0][0]->data[j] = 1.0;
    }
    // parallelized sampling using openmp
    // ityp == 0: sampling a determinant for C term
    //      return H00, H00sq
    // ityp == 1: sampling a determinant for A,B term
    //      return H11, H11sq, H10, H10sq
    template <typename FLI>
    vector<FP>
    parallel_sampling(int n_sample, int ityp,
                      const shared_ptr<FCIDUMP<FLI>> &fcidump) const {
        vector<FP> r(ityp == 0 ? 2 : 4, 0);
        int ntg = threading->activate_global();
        unsigned rand_sd =
            (unsigned)Random::rand_int(0, numeric_limits<int>::max());
        vector<vector<FP>> prr(ntg, r);
#pragma omp parallel num_threads(ntg)
        {
            vector<vector<vector<shared_ptr<SparseMatrix<S, FL>>>>> pmats;
            vector<const vector<shared_ptr<SparseTensor<S, FL>>> *> tensors;
            int tid = threading->get_thread_id();
            RandomMT rand_mt((unsigned)(rand_sd + tid));
            if (ityp == 0) {
                pmats.resize(1);
                tensors.push_back(&tensors_psi0);
                gen_tmp_mats(pinfos_psi0, tensors_psi0, pmats[0]);
            } else {
                pmats.resize(2);
                tensors.push_back(&tensors_qvpsi0);
                tensors.push_back(&tensors_psi0);
                gen_tmp_mats(pinfos_qvpsi0, tensors_qvpsi0, pmats[0]);
                gen_tmp_mats(pinfos_psi0, tensors_psi0, pmats[1]);
            }
            vector<FP> rand(n_sites);
            vector<uint8_t> det_string(n_sites);
            vector<FP> cp(phys_dim), accp(phys_dim + 1, 0);
            vector<shared_ptr<SparseMatrix<S, FL>>> ptrs_save(phys_dim);
            shared_ptr<SparseMatrix<S, FL>> ptrs;
            vector<FP> &rr = prr[tid];
#pragma omp for schedule(static)
            for (int i_sample = 0; i_sample < n_sample; i_sample++) {
                rand_mt.fill<FP>((FP *)rand.data(), n_sites);
                FP det_ener = 0, rnormsq = 0, snorm = 0;
                ptrs = pmats[0][0][0];
                // sample psi0 / qvpsi0
                for (int i_site = 0; i_site < n_sites; i_site++) {
                    for (uint8_t d = 0; d < phys_dim; d++) {
                        shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
                        shared_ptr<SparseMatrix<S, FL>> cmp =
                            pmats[0][i_site + 1][d];
                        cmp->clear();
                        for (auto &m : (*tensors[0])[i_site]->data[d]) {
                            S bra = m.first.first, ket = m.first.second;
                            if (pmp->info->find_state(bra) == -1)
                                continue;
                            GMatrixFunctions<FL>::multiply(
                                (*pmp)[bra], false, m.second->ref(), false,
                                (*cmp)[ket], 1.0, 1.0);
                        }
                        FP tmp = cmp->norm();
                        cp[d] = tmp * tmp;
                        ptrs_save[d] = cmp;
                    }
                    for (uint8_t d = 0; d < phys_dim; d++)
                        accp[d + 1] = accp[d] + cp[d];
                    for (uint8_t d = 0; d < phys_dim; d++) {
                        accp[d + 1] /= accp[phys_dim];
                        if (rand[i_site] < accp[d + 1]) {
                            ptrs = ptrs_save[d];
                            rnormsq = cp[d];
                            det_string[i_site] = d;
                            break;
                        }
                    }
                }
                for (uint16_t i = 0; i < n_sites; i++)
                    for (uint8_t si = 0; si < 2; si++)
                        if (det_string[i] & (si + 1)) {
                            det_ener += (FP)fcidump->t(si, i, i);
                            for (uint16_t j = 0; j < n_sites; j++)
                                for (uint8_t sj = 0; sj < 2; sj++)
                                    if (det_string[j] & (sj + 1)) {
                                        det_ener +=
                                            0.5 *
                                            (FP)fcidump->v(si, sj, i, i, j, j);
                                        if (si == sj)
                                            det_ener -=
                                                0.5 * (FP)fcidump->v(si, sj, i,
                                                                     j, j, i);
                                    }
                        }
                det_ener += (FP)fcidump->const_e;
                if (ityp == 0) {
                    rr[0] += 1 / det_ener;
                    rr[1] += 1 / (det_ener * det_ener);
                } else {
                    rr[0] += norm_qvpsi0 / det_ener;
                    rr[1] += norm_qvpsi0 * norm_qvpsi0 / (det_ener * det_ener);
                    ptrs = pmats[1][0][0];
                    // overlap psi0
                    for (int i_site = 0; i_site < n_sites; i_site++) {
                        const uint8_t d = det_string[i_site];
                        shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
                        shared_ptr<SparseMatrix<S, FL>> cmp =
                            pmats[1][i_site + 1][d];
                        cmp->clear();
                        for (auto &m : (*tensors[1])[i_site]->data[d]) {
                            S bra = m.first.first, ket = m.first.second;
                            if (pmp->info->find_state(bra) == -1)
                                continue;
                            GMatrixFunctions<FL>::multiply(
                                (*pmp)[bra], false, m.second->ref(), false,
                                (*cmp)[ket], 1.0, 1.0);
                        }
                        ptrs = cmp;
                        if (i_site == n_sites - 1)
                            snorm = cmp->norm();
                    }
                    const FP tmp =
                        norm_qvpsi0 * snorm / (sqrt(rnormsq) * det_ener);
                    rr[2] += tmp;
                    rr[3] += tmp * tmp;
                }
            }
        }
        for (int j = 0; j < (int)r.size(); j++) {
            for (int ip = 0; ip < ntg; ip++)
                r[j] += prr[ip][j];
            if (n_sample != 0)
                r[j] /= n_sample;
        }
        threading->activate_normal();
        return r;
    }
    template <typename FLI>
    FP energy_zeroth(const shared_ptr<FCIDUMP<FLI>> &fcidump,
                     GMatrix<FLI> e_pqqp, GMatrix<FLI> e_pqpq,
                     GMatrix<FLI> pdm1) {
        FLI ener = 0.0;
        assert(e_pqqp.size() == e_pqpq.size());

        for (uint16_t p = 0; p < fcidump->n_sites(); p++) {
            for (uint16_t q = 0; q < p; q++) {
                ener += 0.5 * e_pqqp(p, q) * fcidump->v(p, p, q, q);
                ener += 0.5 * e_pqqp(q, p) * fcidump->v(q, q, p, p);
                ener += 0.5 * e_pqpq(p, q) * fcidump->v(p, q, q, p);
                ener += 0.5 * e_pqpq(q, p) * fcidump->v(q, p, p, q);
            }
            ener += 0.5 * e_pqqp(p, p) * fcidump->v(p, p, p, p);
            ener += pdm1(p, p) * fcidump->t(p, p);
        }
        return (FP)ener;
    }
};

template <typename S, typename FL>
struct StochasticPDMRG<S, FL, typename S::is_su2_t> {
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<SparseMatrix<S, FL>> left_psi0, left_qvpsi0;
    vector<shared_ptr<SparseTensor<S, FL>>> tensors_psi0, tensors_qvpsi0;
    FP norm_qvpsi0;
    vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos_psi0, pinfos_qvpsi0;
    int n_sites;
    uint8_t phys_dim;
    StochasticPDMRG() {}
    StochasticPDMRG(const shared_ptr<UnfusedMPS<S, FL>> &mps_psi0,
                    const shared_ptr<UnfusedMPS<S, FL>> &mps_qvpsi0,
                    const FP norm) {
        this->initialize(mps_psi0, mps_qvpsi0, norm);
    }
    void initialize(const shared_ptr<UnfusedMPS<S, FL>> &mps_psi0,
                    const shared_ptr<UnfusedMPS<S, FL>> &mps_qvpsi0,
                    const FP norm) {
        Random::rand_seed(0);
        n_sites = mps_psi0->n_sites;
        phys_dim = 4;

        tensors_psi0.resize(n_sites);
        tensors_qvpsi0.resize(n_sites);
        for (int i = 0; i < n_sites; i++) {
            tensors_psi0[i] = mps_psi0->tensors[i];
            tensors_qvpsi0[i] = mps_qvpsi0->tensors[i];
        }
        pinfos_psi0.resize(n_sites);
        pinfos_qvpsi0.resize(n_sites);
        gen_si_map(pinfos_psi0, mps_psi0);
        gen_si_map(pinfos_qvpsi0, mps_qvpsi0);

        norm_qvpsi0 = norm;
    }
    void gen_si_map(vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos,
                    const shared_ptr<UnfusedMPS<S, FL>> &mps) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();

        pinfos.resize(n_sites + 1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(*mps->info->left_dims_fci[0],
                                 *mps->info->left_dims_fci[0],
                                 mps->info->vacuum, false);
        for (int d = 0; d < n_sites; d++) {
            pinfos[d + 1].resize(4);
            for (int j = 0; j < pinfos[d + 1].size(); j++) {
                int jd = j >= 2 ? j - 1 : j;
                map<S, MKL_INT> qkets;
                for (auto &m : mps->tensors[d]->data[jd]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (jd == 1 && !((j == 1 && ket.twos() > bra.twos()) ||
                                     (j == 2 && ket.twos() < bra.twos())))
                        continue;
                    if (!qkets.count(ket))
                        qkets[ket] = m.second->shape[2];
                }
                StateInfo<S> ibra, iket;
                ibra.allocate((int)qkets.size());
                iket.allocate((int)qkets.size());
                int k = 0;
                for (auto &qm : qkets) {
                    ibra.quanta[k] = iket.quanta[k] = qm.first;
                    ibra.n_states[k] = 1;
                    iket.n_states[k] = (ubond_t)qm.second;
                    k++;
                }
                pinfos[d + 1][j] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                pinfos[d + 1][j]->initialize(ibra, iket, mps->info->vacuum,
                                             false);
            }
        }
    }
    // ityp == 0: sampling a determinant for C term
    // ityp == 1: sampling a determinant for A,B term
    FP sampling(int ityp, vector<uint8_t> &det_string) const {
        const vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos =
            ityp == 0 ? pinfos_psi0 : pinfos_qvpsi0;
        const vector<shared_ptr<SparseTensor<S, FL>>> &tensors =
            ityp == 0 ? tensors_psi0 : tensors_qvpsi0;

        shared_ptr<SparseMatrix<S, FL>> ptrs;

        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> initp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        initp->allocate(pinfos[0][0]);
        for (size_t j = 0; j < initp->total_memory; j++)
            initp->data[j] = 1.0;

        ptrs = initp;

        det_string.resize(2 * n_sites);
        vector<FP> rand(n_sites);
        Random::fill<FP>((FP *)rand.data(), n_sites);
        vector<FP> cp(phys_dim), accp(phys_dim + 1, 0);
        vector<shared_ptr<SparseMatrix<S, FL>>> ptrs_save(phys_dim);
        FP rnorm = 0;
        for (int i_site = 0; i_site < n_sites; i_site++) {
            for (uint8_t d = 0; d < phys_dim; d++) {
                int dd = d >= 2 ? d - 1 : d;
                shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
                shared_ptr<SparseMatrix<S, FL>> cmp =
                    make_shared<SparseMatrix<S, FL>>(d_alloc);
                cmp->allocate(pinfos[i_site + 1][d]);
                for (auto &m : tensors[i_site]->data[dd]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (dd == 1 && !((d == 1 && ket.twos() > bra.twos()) ||
                                     (d == 2 && ket.twos() < bra.twos())))
                        continue;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                                   m.second->ref(), false,
                                                   (*cmp)[ket], 1.0, 1.0);
                }
                FP tmp = cmp->norm();
                cp[d] = tmp * tmp;
                ptrs_save[d] = cmp;
            }
            for (uint8_t d = 0; d < phys_dim; d++)
                accp[d + 1] = accp[d] + cp[d];
            for (uint8_t d = 0; d < phys_dim; d++) {
                accp[d + 1] /= accp[phys_dim];
                if (rand[i_site] < accp[d + 1]) {
                    ptrs = ptrs_save[d];
                    rnorm = cp[d];
                    det_string[2 * i_site] = d & 1;
                    det_string[2 * i_site + 1] = (d & 2) >> 1;
                    break;
                }
            }
        }
        return sqrt(rnorm);
    }
    // ityp == 0: <Psi^(0)|VQ|D>
    // ityp == 1: <Psi^(0)|D>
    FP overlap(int ityp, const vector<uint8_t> &det_string) const {
        const vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos =
            ityp == 1 ? pinfos_psi0 : pinfos_qvpsi0;
        const vector<shared_ptr<SparseTensor<S, FL>>> &tensors =
            ityp == 1 ? tensors_psi0 : tensors_qvpsi0;

        shared_ptr<SparseMatrix<S, FL>> ptrs;
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrix<S, FL>> initp =
            make_shared<SparseMatrix<S, FL>>(d_alloc);
        initp->allocate(pinfos[0][0]);
        for (size_t j = 0; j < initp->total_memory; j++)
            initp->data[j] = 1.0;

        ptrs = initp;

        for (int i_site = 0; i_site < n_sites; i_site++) {
            int d = det_string[2 * i_site] + (det_string[2 * i_site + 1] << 1);
            int dd = d >= 2 ? d - 1 : d;
            shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
            shared_ptr<SparseMatrix<S, FL>> cmp =
                make_shared<SparseMatrix<S, FL>>(d_alloc);
            cmp->allocate(pinfos[i_site + 1][d]);
            for (auto &m : tensors[i_site]->data[dd]) {
                S bra = m.first.first, ket = m.first.second;
                if (dd == 1 && !((d == 1 && ket.twos() > bra.twos()) ||
                                 (d == 2 && ket.twos() < bra.twos())))
                    continue;
                if (pmp->info->find_state(bra) == -1)
                    continue;
                GMatrixFunctions<FL>::multiply((*pmp)[bra], false,
                                               m.second->ref(), false,
                                               (*cmp)[ket], 1.0, 1.0);
            }
            ptrs = cmp;
            if (i_site == n_sites - 1)
                return cmp->norm();
        }
        return 0;
    }
    void
    gen_tmp_mats(const vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos,
                 const vector<shared_ptr<SparseTensor<S, FL>>> &tensors,
                 vector<vector<shared_ptr<SparseMatrix<S, FL>>>> &pmats) const {
        pmats.resize(n_sites + 1);
        shared_ptr<VectorAllocator<FP>> dd_alloc =
            make_shared<VectorAllocator<FP>>();
        for (int i_site = 0; i_site < n_sites + 1; i_site++) {
            pmats[i_site].resize(pinfos[i_site].size());
            for (uint8_t d = 0; d < (uint8_t)pinfos[i_site].size(); d++) {
                pmats[i_site][d] = make_shared<SparseMatrix<S, FL>>(dd_alloc);
                pmats[i_site][d]->allocate(pinfos[i_site][d]);
            }
        }
        for (size_t j = 0; j < pmats[0][0]->total_memory; j++)
            pmats[0][0]->data[j] = 1.0;
    }
    // parallelized sampling using openmp
    // ityp == 0: sampling a determinant for C term
    //      return H00, H00sq
    // ityp == 1: sampling a determinant for A,B term
    //      return H11, H11sq, H10, H10sq
    template <typename FLI>
    vector<FP>
    parallel_sampling(int n_sample, int ityp,
                      const shared_ptr<FCIDUMP<FLI>> &fcidump) const {
        vector<FP> r(ityp == 0 ? 2 : 4, 0);
        int ntg = threading->activate_global();
        unsigned rand_sd =
            (unsigned)Random::rand_int(0, numeric_limits<int>::max());
        vector<vector<FP>> prr(ntg, r);
#pragma omp parallel num_threads(ntg)
        {
            vector<vector<vector<shared_ptr<SparseMatrix<S, FL>>>>> pmats;
            vector<const vector<shared_ptr<SparseTensor<S, FL>>> *> tensors;
            int tid = threading->get_thread_id();
            RandomMT rand_mt((unsigned)(rand_sd + tid));
            if (ityp == 0) {
                pmats.resize(1);
                tensors.push_back(&tensors_psi0);
                gen_tmp_mats(pinfos_psi0, tensors_psi0, pmats[0]);
            } else {
                pmats.resize(2);
                tensors.push_back(&tensors_qvpsi0);
                tensors.push_back(&tensors_psi0);
                gen_tmp_mats(pinfos_qvpsi0, tensors_qvpsi0, pmats[0]);
                gen_tmp_mats(pinfos_psi0, tensors_psi0, pmats[1]);
            }
            vector<FP> rand(n_sites);
            vector<uint8_t> det_string(n_sites);
            vector<FP> cp(phys_dim), accp(phys_dim + 1, 0);
            vector<shared_ptr<SparseMatrix<S, FL>>> ptrs_save(phys_dim);
            shared_ptr<SparseMatrix<S, FL>> ptrs;
            vector<FP> &rr = prr[tid];
#pragma omp for schedule(static)
            for (int i_sample = 0; i_sample < n_sample; i_sample++) {
                rand_mt.fill<FP>((FP *)rand.data(), n_sites);
                FP det_ener = 0, rnormsq = 0, snorm = 0;
                ptrs = pmats[0][0][0];
                // sample psi0 / qvpsi0
                for (int i_site = 0; i_site < n_sites; i_site++) {
                    for (uint8_t d = 0; d < phys_dim; d++) {
                        int dd = d >= 2 ? d - 1 : d;
                        shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
                        shared_ptr<SparseMatrix<S, FL>> cmp =
                            pmats[0][i_site + 1][d];
                        cmp->clear();
                        for (auto &m : (*tensors[0])[i_site]->data[dd]) {
                            S bra = m.first.first, ket = m.first.second;
                            if (dd == 1 &&
                                !((d == 1 && ket.twos() > bra.twos()) ||
                                  (d == 2 && ket.twos() < bra.twos())))
                                continue;
                            if (pmp->info->find_state(bra) == -1)
                                continue;
                            GMatrixFunctions<FL>::multiply(
                                (*pmp)[bra], false, m.second->ref(), false,
                                (*cmp)[ket], 1.0, 1.0);
                        }
                        FP tmp = cmp->norm();
                        cp[d] = tmp * tmp;
                        ptrs_save[d] = cmp;
                    }
                    for (uint8_t d = 0; d < phys_dim; d++)
                        accp[d + 1] = accp[d] + cp[d];
                    for (uint8_t d = 0; d < phys_dim; d++) {
                        accp[d + 1] /= accp[phys_dim];
                        if (rand[i_site] < accp[d + 1]) {
                            ptrs = ptrs_save[d];
                            rnormsq = cp[d];
                            det_string[i_site] = d;
                            break;
                        }
                    }
                }
                for (uint16_t i = 0; i < n_sites; i++)
                    for (uint8_t si = 0; si < 2; si++)
                        if (det_string[i] & (si + 1)) {
                            det_ener += (FP)fcidump->t(si, i, i);
                            for (uint16_t j = 0; j < n_sites; j++)
                                for (uint8_t sj = 0; sj < 2; sj++)
                                    if (det_string[j] & (sj + 1)) {
                                        det_ener +=
                                            0.5 *
                                            (FP)fcidump->v(si, sj, i, i, j, j);
                                        if (si == sj)
                                            det_ener -=
                                                0.5 * (FP)fcidump->v(si, sj, i,
                                                                     j, j, i);
                                    }
                        }
                det_ener += (FP)fcidump->const_e;
                if (ityp == 0) {
                    rr[0] += 1 / det_ener;
                    rr[1] += 1 / (det_ener * det_ener);
                } else {
                    rr[0] += norm_qvpsi0 / det_ener;
                    rr[1] += norm_qvpsi0 * norm_qvpsi0 / (det_ener * det_ener);
                    ptrs = pmats[1][0][0];
                    // overlap psi0
                    for (int i_site = 0; i_site < n_sites; i_site++) {
                        const uint8_t d = det_string[i_site];
                        int dd = d >= 2 ? d - 1 : d;
                        shared_ptr<SparseMatrix<S, FL>> pmp = ptrs;
                        shared_ptr<SparseMatrix<S, FL>> cmp =
                            pmats[1][i_site + 1][d];
                        cmp->clear();
                        for (auto &m : (*tensors[1])[i_site]->data[dd]) {
                            S bra = m.first.first, ket = m.first.second;
                            if (dd == 1 &&
                                !((d == 1 && ket.twos() > bra.twos()) ||
                                  (d == 2 && ket.twos() < bra.twos())))
                                continue;
                            if (pmp->info->find_state(bra) == -1)
                                continue;
                            GMatrixFunctions<FL>::multiply(
                                (*pmp)[bra], false, m.second->ref(), false,
                                (*cmp)[ket], 1.0, 1.0);
                        }
                        ptrs = cmp;
                        if (i_site == n_sites - 1)
                            snorm = cmp->norm();
                    }
                    const FP tmp =
                        norm_qvpsi0 * snorm / (sqrt(rnormsq) * det_ener);
                    rr[2] += tmp;
                    rr[3] += tmp * tmp;
                }
            }
        }
        for (int j = 0; j < (int)r.size(); j++) {
            for (int ip = 0; ip < ntg; ip++)
                r[j] += prr[ip][j];
            if (n_sample != 0)
                r[j] /= n_sample;
        }
        threading->activate_normal();
        return r;
    }
    template <typename FLI>
    FP energy_zeroth(const shared_ptr<FCIDUMP<FLI>> &fcidump,
                     GMatrix<FLI> e_pqqp, GMatrix<FLI> e_pqpq,
                     GMatrix<FLI> pdm1) {
        FLI ener = 0.0;
        assert(e_pqqp.size() == e_pqpq.size());

        for (uint16_t p = 0; p < fcidump->n_sites(); p++) {
            for (uint16_t q = 0; q < p; q++) {
                ener += 0.5 * e_pqqp(p, q) * fcidump->v(p, p, q, q);
                ener += 0.5 * e_pqqp(q, p) * fcidump->v(q, q, p, p);
                ener += 0.5 * e_pqpq(p, q) * fcidump->v(p, q, q, p);
                ener += 0.5 * e_pqpq(q, p) * fcidump->v(q, p, p, q);
            }
            ener += 0.5 * e_pqqp(p, p) * fcidump->v(p, p, p, p);
            ener += pdm1(p, p) * fcidump->t(p, p);
        }
        return (FP)ener;
    }
};

} // namespace block2
