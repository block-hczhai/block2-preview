
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2021 Seunghoon Lee <seunghoonlee89@gmail.com>
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
#include "../dmrg/mps.hpp"
#include "../dmrg/mps_unfused.hpp"
#include <algorithm>
#include <array>
#include <set>
#include <stack>
#include <tuple>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct StochasticPDMRG;

template <typename S> struct StochasticPDMRG <S, typename S::is_sz_t> {
    shared_ptr<SparseMatrix<S>> left_psi0, left_qvpsi0;
    vector<shared_ptr<SparseTensor<S>>> tensors_psi0, tensors_qvpsi0;
    string canonical_form;
    vector<uint8_t> det_string;
    double norm_qvpsi0;
    vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos_psi0, pinfos_qvpsi0;
    int center, n_sites, dot;
    int phys_dim;
    StochasticPDMRG() {}
    StochasticPDMRG(const shared_ptr<UnfusedMPS<S>> &mps_psi0, const shared_ptr<UnfusedMPS<S>> &mps_qvpsi0, const double norm) { this->initialize(mps_psi0, mps_qvpsi0, norm); }
    // initialize
    void initialize(const shared_ptr<UnfusedMPS<S>> &mps_psi0, const shared_ptr<UnfusedMPS<S>> &mps_qvpsi0, const double norm) {
        Random::rand_seed(0);
        // add assertions for canonical_form CR..R, onedot, n_sites
        canonical_form = mps_psi0->canonical_form;
        center = mps_psi0->center;
        n_sites = mps_psi0->n_sites;
        dot = mps_psi0->dot;
        phys_dim = 4;

        det_string.resize(2*n_sites);
        tensors_psi0.resize(n_sites);
        tensors_qvpsi0.resize(n_sites);
        for (int i = 0; i < n_sites; i++){
            tensors_psi0[i]   = mps_psi0->tensors[i];
            tensors_qvpsi0[i] = mps_qvpsi0->tensors[i];
        }
        pinfos_psi0.resize(n_sites);
        pinfos_qvpsi0.resize(n_sites);
        gen_si_map(pinfos_psi0, mps_psi0);
        gen_si_map(pinfos_qvpsi0, mps_qvpsi0);

        norm_qvpsi0 = norm;
    }
    // empty det_string 
    void clear() {
        det_string.clear();
        det_string.resize(2*n_sites);
    }
    // generate stateinfo map 
    void gen_si_map(vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos, const shared_ptr<UnfusedMPS<S>> &mps) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();

        pinfos.resize(n_sites+1);
        pinfos[0].resize(1);
        pinfos[0][0] = make_shared<SparseMatrixInfo<S>>(i_alloc);
        pinfos[0][0]->initialize(StateInfo<S>(mps->info->vacuum),
                                 StateInfo<S>(mps->info->vacuum),
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
    // conditional probability for C term 
    //TODO: combine sampling_c and sampling_ab
    void sampling_c() 
    {
        shared_ptr<SparseMatrix<S>> ptrs;

        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> initp = 
            make_shared<SparseMatrix<S>>(d_alloc);
        initp->allocate(pinfos_psi0[0][0]);
        initp->data[0] = 1.0;

        ptrs = initp;

        vector<double> rand;
        rand.resize(n_sites);
        Random::fill_rand_double((double *)rand.data(), n_sites);

        //double p_norm = 1.0;
        for (int i_site=0; i_site < n_sites; i_site++)
        {
            //test
            //cout << "i_site" << i_site << endl;

            vector<double> cp;
            cp.resize(phys_dim);
            vector<shared_ptr<SparseMatrix<S>>> ptrs_save;
            ptrs_save.resize(phys_dim);
            for (int d = 0; d < phys_dim; d++)
            {
                shared_ptr<VectorAllocator<double>> dc_alloc =
                    make_shared<VectorAllocator<double>>();
                shared_ptr<SparseMatrix<S>> pmp = ptrs;
                shared_ptr<SparseMatrix<S>> cmp = 
                    make_shared<SparseMatrix<S>>(dc_alloc);
                cmp->allocate(pinfos_psi0[i_site + 1][d]);
                for (auto &m : tensors_psi0[i_site]->data[d]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    MatrixFunctions::multiply((*pmp)[bra], false,
                                              m.second->ref(), false,
                                              (*cmp)[ket], 1.0, 1.0);
                }
                double tmp = cmp->norm();
                cp[d] = tmp*tmp;
                ptrs_save[d] = cmp;
                //test
                //cout << cp[d] << endl;
            }

            vector<double> accp;
            accp.resize(phys_dim+1);
            accp[0] = 0.0;
            for (int d = 0; d < phys_dim; d++)
                accp[d+1] = accp[d] + cp[d]; 
            //normalize
            for (int d = 0; d < phys_dim+1; d++)
                accp[d] /= accp[phys_dim]; 

            for (int d = 0; d < phys_dim; d++)
                if ( rand[i_site] > accp[d] && rand[i_site] < accp[d+1] )
                {
                    //p_norm *= cp[d];
                    ptrs = ptrs_save[d];
                    if (d==0 || d==2)
                        det_string[2*i_site] = (uint8_t) 0;
                    else 
                        det_string[2*i_site] = (uint8_t) 1;

                    if (d==0 || d==1)
                        det_string[2*i_site+1] = (uint8_t) 0;
                    else 
                        det_string[2*i_site+1] = (uint8_t) 1;
                }

            //test
            //cout << "accum: " << accp[phys_dim] << " rand:" << rand[i_site] << endl;

            cp.clear(); 
            ptrs_save.clear(); 
        }
    }

    void sampling_ab() 
    {
        shared_ptr<SparseMatrix<S>> ptrs;

        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> initp = 
            make_shared<SparseMatrix<S>>(d_alloc);
        initp->allocate(pinfos_qvpsi0[0][0]);
        initp->data[0] = 1.0;

        ptrs = initp;

        vector<double> rand;
        rand.resize(n_sites);
        Random::fill_rand_double((double *)rand.data(), n_sites);

        //double p_norm = norm_qvpsi0;
        //double p_norm_tmp;
        for (int i_site=0; i_site < n_sites; i_site++)
        {
            vector<double> cp;
            cp.resize(phys_dim);
            vector<shared_ptr<SparseMatrix<S>>> ptrs_save;
            ptrs_save.resize(phys_dim);
            for (int d = 0; d < phys_dim; d++)
            {
                shared_ptr<VectorAllocator<double>> dc_alloc =
                    make_shared<VectorAllocator<double>>();
                shared_ptr<SparseMatrix<S>> pmp = ptrs;
                shared_ptr<SparseMatrix<S>> cmp = 
                    make_shared<SparseMatrix<S>>(dc_alloc);
                cmp->allocate(pinfos_qvpsi0[i_site + 1][d]);
                for (auto &m : tensors_qvpsi0[i_site]->data[d]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    MatrixFunctions::multiply((*pmp)[bra], false,
                                              m.second->ref(), false,
                                              (*cmp)[ket], 1.0, 1.0);
                }
                double tmp = cmp->norm();
                //cp[d] = tmp*tmp/p_norm;
                cp[d] = tmp*tmp;
                ptrs_save[d] = cmp;
            }

            vector<double> accp;
            accp.resize(phys_dim+1);
            accp[0] = 0.0;
            for (int d = 0; d < phys_dim; d++)
                accp[d+1] = accp[d] + cp[d]; 
            //normalize (instead of p_norm)
            for (int d = 0; d < phys_dim+1; d++)
                accp[d] /= accp[phys_dim]; 

            for (int d = 0; d < phys_dim; d++)
                if ( rand[i_site] > accp[d] && rand[i_site] < accp[d+1] )
                {
                    //p_norm *= cp[d];
                    ptrs = ptrs_save[d];
                    if (d==0 || d==2)
                        det_string[2*i_site] = (uint8_t) 0;
                    else 
                        det_string[2*i_site] = (uint8_t) 1;

                    if (d==0 || d==1)
                        det_string[2*i_site+1] = (uint8_t) 0;
                    else 
                        det_string[2*i_site+1] = (uint8_t) 1;
                }

            cp.clear(); 
            ptrs_save.clear(); 
        }

        //return p_norm; 
    }

    double overlap_c() 
    {
        double overlap=0.0;
        shared_ptr<SparseMatrix<S>> ptrs;

        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> pmp = 
            make_shared<SparseMatrix<S>>(d_alloc);
        pmp->allocate(pinfos_qvpsi0[0][0]);
        pmp->data[0] = 1.0;

        ptrs = pmp;

        for (int i_site=0; i_site < n_sites; i_site++)
        {
            int d = det_string[2*i_site] + 2*det_string[2*i_site+1];
            {
                shared_ptr<VectorAllocator<double>> dc_alloc =
                    make_shared<VectorAllocator<double>>();
                shared_ptr<SparseMatrix<S>> pmp = ptrs;
                shared_ptr<SparseMatrix<S>> cmp = 
                    make_shared<SparseMatrix<S>>(dc_alloc);
                cmp->allocate(pinfos_qvpsi0[i_site + 1][d]);
                for (auto &m : tensors_qvpsi0[i_site]->data[d]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    MatrixFunctions::multiply((*pmp)[bra], false,
                                              m.second->ref(), false,
                                              (*cmp)[ket], 1.0, 1.0);
                }
                ptrs = cmp;
                if (i_site == n_sites-1)
                    overlap = cmp->norm();
            }
        }
        return overlap;
    }

    double overlap_ab() 
    {
        double overlap=0.0;
        shared_ptr<SparseMatrix<S>> ptrs;

        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<SparseMatrix<S>> initp= 
            make_shared<SparseMatrix<S>>(d_alloc);
        initp->allocate(pinfos_psi0[0][0]);
        initp->data[0] = 1.0;

        ptrs = initp;

        for (int i_site=0; i_site < n_sites; i_site++)
        {
            int d = det_string[2*i_site] + 2*det_string[2*i_site+1];
            {
                shared_ptr<VectorAllocator<double>> dc_alloc =
                    make_shared<VectorAllocator<double>>();
                shared_ptr<SparseMatrix<S>> pmp = ptrs;
                shared_ptr<SparseMatrix<S>> cmp = 
                    make_shared<SparseMatrix<S>>(dc_alloc);
                cmp->allocate(pinfos_psi0[i_site + 1][d]);
                for (auto &m : tensors_psi0[i_site]->data[d]) {
                    S bra = m.first.first, ket = m.first.second;
                    if (pmp->info->find_state(bra) == -1)
                        continue;
                    MatrixFunctions::multiply((*pmp)[bra], false,
                                              m.second->ref(), false,
                                              (*cmp)[ket], 1.0, 1.0);
                }
                ptrs = cmp;
                if (i_site == n_sites-1)
                    overlap = cmp->norm();
            }
        }
        return overlap;
    }
    double E0(const shared_ptr<FCIDUMP> &fcidump, MatrixRef e_pqqp, MatrixRef e_pqpq, MatrixRef pdm1) {
        double E = 0.0;
        assert(e_pqqp.size() == e_pqpq.size());

        for (uint16_t p = 0; p < fcidump->n_sites(); p++) {
            for (uint16_t q = 0; q < p; q++) {
                E += 0.5 * e_pqqp(p, q) * fcidump->v(p, p, q, q);
                E += 0.5 * e_pqqp(q, p) * fcidump->v(q, q, p, p);
                E += 0.5 * e_pqpq(p, q) * fcidump->v(p, q, q, p);
                E += 0.5 * e_pqpq(q, p) * fcidump->v(q, p, p, q);
            }
            E += 0.5 * e_pqqp(p, p) * fcidump->v(p, p, p, p);
            E += pdm1(p, p) * fcidump->t(p, p);
        }
        return E;
    }
};

template <typename S> struct StochasticPDMRG <S, typename S::is_su2_t> {
    shared_ptr<SparseMatrix<S>> left_psi0, left_qvpsi0;
    vector<shared_ptr<SparseTensor<S>>> tensors_psi0, tensors_qvpsi0;
    string canonical_form;
    vector<uint8_t> det_string;
    double norm_qvpsi0;
    vector<vector<shared_ptr<SparseMatrixInfo<S>>>> pinfos_psi0, pinfos_qvpsi0;
    int center, n_sites, dot;
    int phys_dim;
    StochasticPDMRG() {}
    StochasticPDMRG(const shared_ptr<UnfusedMPS<S>> &mps_psi0, const shared_ptr<UnfusedMPS<S>> &mps_qvpsi0, const double norm) { this->initialize(mps_psi0, mps_qvpsi0, norm); }
    void initialize(const shared_ptr<UnfusedMPS<S>> &mps_psi0, const shared_ptr<UnfusedMPS<S>> &mps_qvpsi0, const double norm) {
        Random::rand_seed(0);
    }
    void clear() {
        det_string.clear();
        det_string.resize(2*n_sites);
    }
    void gen_si_map(vector<vector<shared_ptr<SparseMatrixInfo<S>>>> &pinfos, const shared_ptr<UnfusedMPS<S>> &mps) {}
    void sampling_c(){}
    void sampling_ab() {}
    double overlap_c() { return 0.0; }
    double overlap_ab() { return 0.0; } 
    double E0() { return 0.0; }
};

} // namespace block2
