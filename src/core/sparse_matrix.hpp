
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

#include "clebsch_gordan.hpp"
#include "complex_matrix_functions.hpp"
#include "matrix.hpp"
#include "matrix_functions.hpp"
#include "state_info.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define TINY (1E-20)

using namespace std;

namespace block2 {

template <typename, typename = void> struct SparseMatrixInfo;

// Quantum label information for block-sparse matrix
template <typename S>
struct SparseMatrixInfo<
    S, typename enable_if<integral_constant<
           bool, sizeof(S) % sizeof(uint32_t) == 0>::value>::type> {
    shared_ptr<Allocator<uint32_t>> alloc;
    // Composite quantum number for row and column quanta
    S *quanta;
    ubond_t *n_states_bra, *n_states_ket;
    uint32_t *n_states_total;
    S delta_quantum;
    bool is_fermion;
    bool is_wavefunction;
    // Number of non-zero blocks
    int n;
    static bool cmp_op_info(const pair<S, shared_ptr<SparseMatrixInfo>> &p,
                            S q) {
        return p.first < q;
    }
    // A series of non-zero-block indices
    // for performing SparseMatrix operations
    // These non-zero-block indices can be generated once for each type of
    // combination of delta quantum of operators
    // So quantum numbers will not be checked repeatedly for individual
    // SparseMatrix
    struct ConnectionInfo {
        S *quanta;
        uint32_t *idx;
        uint64_t *stride;
        double *factor;
        uint32_t *ia, *ib, *ic;
        int n[5], nc;
        ConnectionInfo() : nc(-1) { memset(n, -1, sizeof(n)); }
        // Compute non-zero-block indices for 'tensor_product_diagonal'
        void initialize_diag(
            S cdq, S opdq, const vector<pair<uint8_t, S>> &subdq,
            const vector<pair<S, shared_ptr<SparseMatrixInfo>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo>>> &binfos,
            const shared_ptr<SparseMatrixInfo> &cinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size());
            vector<uint32_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = (int)k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = (uint32_t)vic.size();
                S adq = cja ? -subdq[k].second.get_bra(opdq)
                            : subdq[k].second.get_bra(opdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                if ((adq + bdq)[0] != (adq - adq)[0])
                    continue;
                shared_ptr<SparseMatrixInfo> ainfo =
                    lower_bound(ainfos.begin(), ainfos.end(), adq, cmp_op_info)
                        ->second;
                shared_ptr<SparseMatrixInfo> binfo =
                    lower_bound(binfos.begin(), binfos.end(), bdq, cmp_op_info)
                        ->second;
                assert(ainfo->delta_quantum == adq);
                assert(binfo->delta_quantum == bdq);
                for (int ic = 0; ic < cinfo->n; ic++) {
                    S aq = cinfo->quanta[ic].get_bra(cdq);
                    S bq = -cinfo->quanta[ic].get_ket();
                    int ia = ainfo->find_state(aq), ib = binfo->find_state(bq);
                    if (ia != -1 && ib != -1 && aq == aq.get_bra(adq) &&
                        bq == bq.get_bra(bdq)) {
                        double factor =
                            sqrt(cdq.multiplicity() * opdq.multiplicity() *
                                 aq.multiplicity() * bq.multiplicity()) *
                            cg->wigner_9j(aq, bq, cdq, adq, bdq, opdq, aq, bq,
                                          cdq);
                        if (cja)
                            factor *= cg->transpose_cg(adq, aq, aq);
                        if (cjb)
                            factor *= cg->transpose_cg(bdq, bq, bq);
                        factor *=
                            (binfo->is_fermion && aq.is_fermion()) ? -1 : 1;
                        if (abs(factor) >= TINY) {
                            via.push_back(ia);
                            vib.push_back(ib);
                            vic.push_back(ic);
                            vf.push_back(factor);
                        }
                    }
                }
            }
            n[4] = (int)vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = (int)vic.size();
            uint32_t *ptr = ialloc->allocate(n[4] * (sizeof(S) >> 2) + n[4]);
            uint32_t *cptr = ialloc->allocate(nc * 7);
            quanta = (S *)ptr;
            idx = ptr + n[4] * (sizeof(S) >> 2);
            stride = (uint64_t *)cptr;
            factor = (double *)(cptr + nc * 2);
            ia = (uint32_t *)(cptr + nc * 4), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, vidx.data(), n[4] * sizeof(uint32_t));
            memset(stride, 0, nc * sizeof(uint64_t));
            memcpy(factor, vf.data(), nc * sizeof(double));
            memcpy(ia, via.data(), nc * sizeof(uint32_t));
            memcpy(ib, vib.data(), nc * sizeof(uint32_t));
            memcpy(ic, vic.data(), nc * sizeof(uint32_t));
        }
        // Compute non-zero-block indices for 'tensor_product_multiply'
        void initialize_wfn(
            S cdq, S vdq, S opdq, const vector<pair<uint8_t, S>> &subdq,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &binfos,
            const shared_ptr<SparseMatrixInfo<S>> &cinfo,
            const shared_ptr<SparseMatrixInfo<S>> &vinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size());
            vector<uint64_t> viv;
            vector<uint32_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = (int)k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = (uint32_t)viv.size();
                vector<vector<
                    tuple<double, uint64_t, uint32_t, uint32_t, uint32_t>>>
                    pv;
                size_t ip = 0;
                S adq = cja ? -subdq[k].second.get_bra(opdq)
                            : subdq[k].second.get_bra(opdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                shared_ptr<SparseMatrixInfo> ainfo =
                    lower_bound(ainfos.begin(), ainfos.end(), adq, cmp_op_info)
                        ->second;
                shared_ptr<SparseMatrixInfo> binfo =
                    lower_bound(binfos.begin(), binfos.end(), bdq, cmp_op_info)
                        ->second;
                assert(ainfo->delta_quantum == adq);
                assert(binfo->delta_quantum == bdq);
                for (int iv = 0; iv < vinfo->n; iv++) {
                    ip = 0;
                    S lq = vinfo->quanta[iv].get_bra(vdq);
                    S rq = -vinfo->quanta[iv].get_ket();
                    S rqprimes = cjb ? rq + bdq : rq - bdq;
                    for (int r = 0; r < rqprimes.count(); r++) {
                        S rqprime = rqprimes[r];
                        int ib =
                            binfo->find_state(cjb ? bdq.combine(rqprime, rq)
                                                  : bdq.combine(rq, rqprime));
                        if (ib != -1) {
                            S lqprimes = cdq - rqprime;
                            for (int l = 0; l < lqprimes.count(); l++) {
                                S lqprime = lqprimes[l];
                                int ia = ainfo->find_state(
                                    cja ? adq.combine(lqprime, lq)
                                        : adq.combine(lq, lqprime));
                                int ic = cinfo->find_state(
                                    cdq.combine(lqprime, -rqprime));
                                if (ia != -1 && ic != -1) {
                                    double factor =
                                        sqrt(cdq.multiplicity() *
                                             opdq.multiplicity() *
                                             lq.multiplicity() *
                                             rq.multiplicity()) *
                                        cg->wigner_9j(lqprime, rqprime, cdq,
                                                      adq, bdq, opdq, lq, rq,
                                                      vdq);
                                    factor *= (binfo->is_fermion &&
                                               lqprime.is_fermion())
                                                  ? -1
                                                  : 1;
                                    if (cja)
                                        factor *=
                                            cg->transpose_cg(adq, lq, lqprime);
                                    if (cjb)
                                        factor *=
                                            cg->transpose_cg(bdq, rq, rqprime);
                                    if (abs(factor) >= TINY) {
                                        if (pv.size() <= ip)
                                            pv.push_back(
                                                vector<tuple<double, uint64_t,
                                                             uint32_t, uint32_t,
                                                             uint32_t>>());
                                        pv[ip].push_back(
                                            make_tuple(factor, iv, ia, ib, ic));
                                        ip++;
                                    }
                                }
                            }
                        }
                    }
                }
                size_t np = 0;
                for (auto &r : pv)
                    np += r.size();
                vf.reserve(vf.size() + np);
                viv.reserve(viv.size() + np);
                via.reserve(via.size() + np);
                vib.reserve(vib.size() + np);
                vic.reserve(vic.size() + np);
                for (ip = 0; ip < pv.size(); ip++) {
                    for (auto &r : pv[ip]) {
                        vf.push_back(get<0>(r));
                        viv.push_back(get<1>(r));
                        via.push_back(get<2>(r));
                        vib.push_back(get<3>(r));
                        vic.push_back(get<4>(r));
                    }
                }
            }
            n[4] = (int)vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = (int)viv.size();
            uint32_t *ptr = ialloc->allocate(n[4] * (sizeof(S) >> 2) + n[4]);
            uint32_t *cptr = ialloc->allocate(nc * 7);
            quanta = (S *)ptr;
            idx = ptr + n[4] * (sizeof(S) >> 2);
            stride = (uint64_t *)cptr;
            factor = (double *)(cptr + nc * 2);
            ia = (uint32_t *)(cptr + nc * 4), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, vidx.data(), n[4] * sizeof(uint32_t));
            memcpy(stride, viv.data(), nc * sizeof(uint64_t));
            memcpy(factor, vf.data(), nc * sizeof(double));
            memcpy(ia, via.data(), nc * sizeof(uint32_t));
            memcpy(ib, vib.data(), nc * sizeof(uint32_t));
            memcpy(ic, vic.data(), nc * sizeof(uint32_t));
        }
        // Compute non-zero-block indices for 'tensor_product'
        void initialize_tp(
            S cdq, const vector<pair<uint8_t, S>> &subdq,
            const StateInfo<S> &bra, const StateInfo<S> &ket,
            const StateInfo<S> &bra_a, const StateInfo<S> &bra_b,
            const StateInfo<S> &ket_a, const StateInfo<S> &ket_b,
            const shared_ptr<typename StateInfo<S>::ConnectionInfo> &bra_cinfo,
            const shared_ptr<typename StateInfo<S>::ConnectionInfo> &ket_cinfo,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &binfos,
            const shared_ptr<SparseMatrixInfo<S>> &cinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size());
            vector<uint64_t> vstride;
            vector<uint32_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = (int)k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = (uint32_t)vstride.size();
                S adq = cja ? -subdq[k].second.get_bra(cdq)
                            : subdq[k].second.get_bra(cdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                shared_ptr<SparseMatrixInfo<S>> ainfo =
                    lower_bound(ainfos.begin(), ainfos.end(), adq, cmp_op_info)
                        ->second;
                shared_ptr<SparseMatrixInfo<S>> binfo =
                    lower_bound(binfos.begin(), binfos.end(), bdq, cmp_op_info)
                        ->second;
                assert(ainfo->delta_quantum == adq);
                assert(binfo->delta_quantum == bdq);
                for (int ic = 0; ic < cinfo->n; ic++) {
                    int ib = bra.find_state(cinfo->quanta[ic].get_bra(cdq));
                    int ik = ket.find_state(cinfo->quanta[ic].get_ket());
                    int kbed = bra_cinfo->acc_n_states[ib + 1];
                    int kked = ket_cinfo->acc_n_states[ik + 1];
                    uint64_t bra_stride = 0, ket_stride = 0;
                    for (int kb = bra_cinfo->acc_n_states[ib]; kb < kbed;
                         kb++) {
                        uint32_t jba = bra_cinfo->ij_indices[kb].first,
                                 jbb = bra_cinfo->ij_indices[kb].second;
                        ket_stride = 0;
                        for (int kk = ket_cinfo->acc_n_states[ik]; kk < kked;
                             kk++) {
                            uint32_t jka = ket_cinfo->ij_indices[kk].first,
                                     jkb = ket_cinfo->ij_indices[kk].second;
                            S qa = cja ? adq.combine(ket_a.quanta[jka],
                                                     bra_a.quanta[jba])
                                       : adq.combine(bra_a.quanta[jba],
                                                     ket_a.quanta[jka]),
                              qb = cjb ? bdq.combine(ket_b.quanta[jkb],
                                                     bra_b.quanta[jbb])
                                       : bdq.combine(bra_b.quanta[jbb],
                                                     ket_b.quanta[jkb]);
                            if (qa != S(S::invalid) && qb != S(S::invalid)) {
                                int ia = ainfo->find_state(qa),
                                    ib = binfo->find_state(qb);
                                if (ia != -1 && ib != -1) {
                                    S aq = bra_a.quanta[jba];
                                    S aqprime = ket_a.quanta[jka];
                                    S bq = bra_b.quanta[jbb];
                                    S bqprime = ket_b.quanta[jkb];
                                    S cq = cinfo->quanta[ic].get_bra(cdq);
                                    S cqprime = cinfo->quanta[ic].get_ket();
                                    double factor =
                                        sqrt(cqprime.multiplicity() *
                                             cdq.multiplicity() *
                                             aq.multiplicity() *
                                             bq.multiplicity()) *
                                        cg->wigner_9j(aqprime, bqprime, cqprime,
                                                      adq, bdq, cdq, aq, bq,
                                                      cq);
                                    factor *= (binfo->is_fermion &&
                                               aqprime.is_fermion())
                                                  ? -1
                                                  : 1;
                                    if (cja)
                                        factor *=
                                            cg->transpose_cg(adq, aq, aqprime);
                                    if (cjb)
                                        factor *=
                                            cg->transpose_cg(bdq, bq, bqprime);
                                    if (abs(factor) >= TINY) {
                                        via.push_back(ia);
                                        vib.push_back(ib);
                                        vic.push_back(ic);
                                        vstride.push_back(
                                            bra_stride *
                                                cinfo->n_states_ket[ic] +
                                            ket_stride);
                                        vf.push_back(factor);
                                    }
                                }
                            }
                            ket_stride += (uint64_t)ket_a.n_states[jka] *
                                          ket_b.n_states[jkb];
                        }
                        bra_stride +=
                            (uint64_t)bra_a.n_states[jba] * bra_b.n_states[jbb];
                    }
                }
            }
            n[4] = (int)vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = (int)vstride.size();
            uint32_t *ptr = ialloc->allocate(n[4] * (sizeof(S) >> 2) + n[4]);
            uint32_t *cptr = ialloc->allocate(nc * 7);
            quanta = (S *)ptr;
            idx = ptr + n[4] * (sizeof(S) >> 2);
            stride = (uint64_t *)cptr;
            factor = (double *)(cptr + nc * 2);
            ia = (uint32_t *)(cptr + nc * 4), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, vidx.data(), n[4] * sizeof(uint32_t));
            memcpy(stride, vstride.data(), nc * sizeof(uint64_t));
            memcpy(factor, vf.data(), nc * sizeof(double));
            memcpy(ia, via.data(), nc * sizeof(uint32_t));
            memcpy(ib, vib.data(), nc * sizeof(uint32_t));
            memcpy(ic, vic.data(), nc * sizeof(uint32_t));
        }
        void reallocate(bool clean) {
            size_t length = n[4] * (sizeof(S) >> 2) + n[4] + nc * 7;
            uint32_t *ptr = ialloc->reallocate((uint32_t *)quanta, length,
                                               clean ? 0 : length);
            if (ptr != (uint32_t *)quanta) {
                memmove(ptr, quanta, length * sizeof(uint32_t));
                quanta = (S *)ptr;
                idx = ptr + n[4] * (sizeof(S) >> 2);
                stride = (uint64_t *)(ptr + n[4] * (sizeof(S) >> 2) + n[4]);
                factor =
                    (double *)(ptr + n[4] * (sizeof(S) >> 2) + n[4] + nc * 2);
                ia = (uint32_t *)((uint32_t *)stride + nc * 4), ib = ia + nc,
                ic = ib + nc;
            }
            if (clean) {
                quanta = nullptr;
                idx = nullptr;
                stride = nullptr;
                factor = nullptr;
                ia = ib = ic = nullptr;
                nc = -1;
                memset(n, -1, sizeof(n));
            }
        }
        void deallocate() {
            assert(n[4] != -1);
            if (n[4] != 0 || nc != 0)
                ialloc->deallocate((uint32_t *)quanta,
                                   n[4] * (sizeof(S) >> 2) + n[4] + nc * 7);
            quanta = nullptr;
            idx = nullptr;
            stride = nullptr;
            factor = nullptr;
            ia = ib = ic = nullptr;
            nc = -1;
            memset(n, -1, sizeof(n));
        }
        friend ostream &operator<<(ostream &os, const ConnectionInfo &ci) {
            os << "CI N=" << ci.n[4] << " NC=" << ci.nc << endl;
            for (int i = 0; i < 4; i++)
                os << "CJ=" << i << " : " << ci.n[i] << "~" << ci.n[i + 1]
                   << " ; ";
            os << endl;
            for (int i = 0; i < ci.n[4]; i++)
                os << "(BRA) "
                   << ci.quanta[i].get_bra((ci.quanta[i] - ci.quanta[i])[0])
                   << " KET " << -ci.quanta[i].get_ket() << " [ "
                   << (int)ci.idx[i] << "~"
                   << (int)(i != ci.n[4] - 1 ? ci.idx[i + 1] : ci.nc) << " ]"
                   << endl;
            for (int i = 0; i < ci.nc; i++)
                os << setw(4) << i << " IA=" << ci.ia[i] << " IB=" << ci.ib[i]
                   << " IC=" << ci.ic[i] << " STR=" << ci.stride[i]
                   << " factor=" << ci.factor[i] << endl;
            return os;
        }
    };
    shared_ptr<ConnectionInfo> cinfo;
    SparseMatrixInfo(const shared_ptr<Allocator<uint32_t>> &alloc = nullptr)
        : n(-1), cinfo(nullptr), alloc(alloc) {}
    SparseMatrixInfo
    deep_copy(const shared_ptr<Allocator<uint32_t>> &alloc = nullptr) const {
        SparseMatrixInfo other;
        if (alloc == nullptr)
            other.alloc = this->alloc->copy();
        else
            other.alloc = alloc;
        other.allocate(n);
        copy_data_to(other);
        other.delta_quantum = delta_quantum;
        other.is_fermion = is_fermion;
        other.is_wavefunction = is_wavefunction;
        return other;
    }
    void copy_data_to(SparseMatrixInfo &other) const {
        assert(other.n == n);
        memcpy(other.quanta, quanta,
               (n * (sizeof(S) >> 2) + n + _DBL_MEM_SIZE(n)) *
                   sizeof(uint32_t));
    }
    void load_data(const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("SparseMatrixInfo::load_data on '" + filename +
                                "' failed.");
        load_data(ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("SparseMatrixInfo::load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    void load_data(istream &ifs, bool pointer_only = false) {
        ifs.read((char *)&delta_quantum, sizeof(delta_quantum));
        ifs.read((char *)&n, sizeof(n));
        if (alloc == nullptr)
            alloc = ialloc;
        uint32_t *ptr;
        if (pointer_only) {
            size_t psz;
            ifs.read((char *)&psz, sizeof(psz));
            assert(alloc == ialloc || alloc == nullptr);
            ptr = ialloc->data + psz;
        } else {
            ptr = alloc->allocate(n * (sizeof(S) >> 2) + n + _DBL_MEM_SIZE(n));
            ifs.read((char *)ptr, sizeof(uint32_t) * (n * (sizeof(S) >> 2) + n +
                                                      _DBL_MEM_SIZE(n)));
        }
        ifs.read((char *)&is_fermion, sizeof(is_fermion));
        ifs.read((char *)&is_wavefunction, sizeof(is_wavefunction));
        quanta = (S *)ptr;
        n_states_bra = (ubond_t *)(ptr + n * (sizeof(S) >> 2));
        n_states_ket = (ubond_t *)(ptr + n * (sizeof(S) >> 2)) + n;
        n_states_total = ptr + n * (sizeof(S) >> 2) + _DBL_MEM_SIZE(n);
        cinfo = nullptr;
    }
    void save_data(const string &filename) const {
        if (Parsing::link_exists(filename))
            Parsing::remove_file(filename);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("SparseMatrixInfo::save_data on '" + filename +
                                "' failed.");
        save_data(ofs);
        if (!ofs.good())
            throw runtime_error("SparseMatrixInfo::save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    void save_data(ostream &ofs, bool pointer_only = false) const {
        ofs.write((char *)&delta_quantum, sizeof(delta_quantum));
        assert(n != -1);
        ofs.write((char *)&n, sizeof(n));
        if (pointer_only) {
            // for 1-site case with middle site transition
            // one can skip the post-middle-transition then
            // there can be some terms not allocated in stack memory
            // assert(alloc == ialloc);
            size_t psz = (uint32_t *)quanta - ialloc->data;
            ofs.write((char *)&psz, sizeof(psz));
        } else
            ofs.write((char *)quanta,
                      sizeof(uint32_t) *
                          (n * (sizeof(S) >> 2) + n + _DBL_MEM_SIZE(n)));
        ofs.write((char *)&is_fermion, sizeof(is_fermion));
        ofs.write((char *)&is_wavefunction, sizeof(is_wavefunction));
    }
    // L (wfn) x R (wfn)^T = rot
    void initialize_trans_contract(const shared_ptr<SparseMatrixInfo> &linfo,
                                   const shared_ptr<SparseMatrixInfo> &rinfo,
                                   S dq, bool trace_right) {
        this->is_fermion = false;
        this->is_wavefunction = false;
        delta_quantum = dq;
        map<S, pair<int, int>> mqs;
        vector<S> qs;
        if (trace_right) {
            for (int i = 0; i < linfo->n; i++) {
                S q = linfo->quanta[i].get_bra(linfo->delta_quantum);
                mqs[q].first = linfo->n_states_bra[i];
            }
            for (int i = 0; i < rinfo->n; i++) {
                S q = rinfo->quanta[i].get_bra(rinfo->delta_quantum);
                mqs[q].second = rinfo->n_states_bra[i];
            }
        } else {
            for (int i = 0; i < linfo->n; i++) {
                S q = -linfo->quanta[i].get_ket();
                mqs[q].first = linfo->n_states_ket[i];
            }
            for (int i = 0; i < rinfo->n; i++) {
                S q = -rinfo->quanta[i].get_ket();
                mqs[q].second = rinfo->n_states_ket[i];
            }
        }
        for (auto &q : mqs)
            if (q.second.first != 0 && q.second.second != 0)
                qs.push_back(q.first);
        n = (int)qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, qs.data(), n * sizeof(S));
            for (int i = 0; i < n; i++)
                n_states_bra[i] = mqs[qs[i]].first,
                n_states_ket[i] = mqs[qs[i]].second;
            sort_states();
        }
    }
    // Generate minimal SparseMatrixInfo from contracting two SparseMatrix
    void initialize_contract(const shared_ptr<SparseMatrixInfo> &linfo,
                             const shared_ptr<SparseMatrixInfo> &rinfo) {
        assert(linfo->is_wavefunction ^ rinfo->is_wavefunction);
        this->is_fermion = false;
        this->is_wavefunction = true;
        shared_ptr<SparseMatrixInfo> winfo =
            linfo->is_wavefunction ? linfo : rinfo;
        delta_quantum = winfo->delta_quantum;
        vector<S> qs;
        qs.reserve(winfo->n);
        if (rinfo->is_wavefunction)
            for (int i = 0; i < rinfo->n; i++) {
                S bra = rinfo->quanta[i].get_bra(delta_quantum);
                if (linfo->find_state(bra) != -1)
                    qs.push_back(rinfo->quanta[i]);
            }
        else
            for (int i = 0; i < linfo->n; i++) {
                S ket = -linfo->quanta[i].get_ket();
                if (rinfo->find_state(ket) != -1)
                    qs.push_back(linfo->quanta[i]);
            }
        n = (int)qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, qs.data(), n * sizeof(S));
            if (rinfo->is_wavefunction)
                for (int i = 0; i < n; i++) {
                    S bra = quanta[i].get_bra(delta_quantum);
                    n_states_bra[i] =
                        linfo->n_states_bra[linfo->find_state(bra)];
                    n_states_ket[i] =
                        rinfo->n_states_ket[rinfo->find_state(quanta[i])];
                }
            else
                for (int i = 0; i < n; i++) {
                    S ket = -quanta[i].get_ket();
                    n_states_bra[i] =
                        linfo->n_states_bra[linfo->find_state(quanta[i])];
                    n_states_ket[i] =
                        rinfo->n_states_ket[rinfo->find_state(ket)];
                }
            n_states_total[0] = 0;
            for (int i = 0; i < n - 1; i++)
                n_states_total[i + 1] =
                    n_states_total[i] +
                    (uint32_t)n_states_bra[i] * n_states_ket[i];
        }
    }
    // Generate SparseMatrixInfo for density matrix
    void initialize_dm(const vector<shared_ptr<SparseMatrixInfo>> &wfn_infos,
                       S dq, bool trace_right) {
        this->is_fermion = false;
        this->is_wavefunction = false;
        delta_quantum = dq;
        vector<S> qs;
        assert(wfn_infos.size() >= 1);
        qs.reserve(wfn_infos[0]->n);
        for (size_t iw = 0; iw < wfn_infos.size(); iw++) {
            shared_ptr<SparseMatrixInfo> wfn_info = wfn_infos[iw];
            assert(wfn_info->is_wavefunction);
            if (trace_right)
                for (int i = 0; i < wfn_info->n; i++)
                    qs.push_back(
                        wfn_info->quanta[i].get_bra(wfn_info->delta_quantum));
            else
                for (int i = 0; i < wfn_info->n; i++)
                    qs.push_back(-wfn_info->quanta[i].get_ket());
        }
        sort(qs.begin(), qs.end());
        qs.resize(distance(qs.begin(), unique(qs.begin(), qs.end())));
        n = (int)qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, qs.data(), n * sizeof(S));
            if (trace_right)
                for (size_t iw = 0; iw < wfn_infos.size(); iw++) {
                    shared_ptr<SparseMatrixInfo> wfn_info = wfn_infos[iw];
                    for (int i = 0; i < wfn_info->n; i++) {
                        S q = wfn_info->quanta[i].get_bra(
                            wfn_info->delta_quantum);
                        int ii = find_state(q);
                        n_states_bra[ii] = n_states_ket[ii] =
                            wfn_info->n_states_bra[i];
                    }
                }
            else
                for (size_t iw = 0; iw < wfn_infos.size(); iw++) {
                    shared_ptr<SparseMatrixInfo> wfn_info = wfn_infos[iw];
                    for (int i = 0; i < wfn_info->n; i++) {
                        S q = -wfn_info->quanta[i].get_ket();
                        int ii = find_state(q);
                        n_states_bra[ii] = n_states_ket[ii] =
                            wfn_info->n_states_ket[i];
                    }
                }
            n_states_total[0] = 0;
            for (int i = 0; i < n - 1; i++)
                n_states_total[i + 1] =
                    n_states_total[i] +
                    (uint32_t)n_states_bra[i] * n_states_ket[i];
        }
    }
    // Generate SparseMatrixInfo from bra and ket StateInfo and
    // delta quantum of the operator
    void initialize(const StateInfo<S> &bra, const StateInfo<S> &ket, S dq,
                    bool is_fermion, bool wfn = false) {
        this->is_fermion = is_fermion;
        this->is_wavefunction = wfn;
        delta_quantum = dq;
        vector<S> qs;
        qs.reserve(ket.n);
        for (int i = 0; i < ket.n; i++) {
            S q = wfn ? -ket.quanta[i] : ket.quanta[i];
            S bs = dq + q;
            for (int k = 0; k < bs.count(); k++)
                if (bra.find_state(bs[k]) != -1)
                    qs.push_back(dq.combine(bs[k], q));
        }
        n = (int)qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, qs.data(), n * sizeof(S));
            sort(quanta, quanta + n);
            for (int i = 0; i < n; i++) {
                // possible assertion failure when the quantum number
                // exceeds the symmetry type limit
                assert(ket.find_state(wfn ? -quanta[i].get_ket()
                                          : quanta[i].get_ket()) != -1);
                n_states_ket[i] = ket.n_states[ket.find_state(
                    wfn ? -quanta[i].get_ket() : quanta[i].get_ket())];
                n_states_bra[i] =
                    bra.n_states[bra.find_state(quanta[i].get_bra(dq))];
            }
            n_states_total[0] = 0;
            for (int i = 0; i < n - 1; i++)
                n_states_total[i + 1] =
                    n_states_total[i] +
                    (uint32_t)n_states_bra[i] * n_states_ket[i];
        }
    }
    // Extract row or column StateInfo from SparseMatrixInfo
    shared_ptr<StateInfo<S>> extract_state_info(bool right) {
        shared_ptr<StateInfo<S>> info = make_shared<StateInfo<S>>();
        if (delta_quantum == (delta_quantum - delta_quantum)[0] &&
            !is_wavefunction) {
            info->allocate(n);
            memcpy(info->quanta, quanta, n * sizeof(S));
            memcpy(info->n_states, right ? n_states_ket : n_states_bra,
                   n * sizeof(ubond_t));
        } else {
            map<S, ubond_t> qs;
            for (int i = 0; i < n; i++)
                qs[right ? (is_wavefunction ? -quanta[i].get_ket()
                                            : quanta[i].get_ket())
                         : quanta[i].get_bra(delta_quantum)] =
                    right ? n_states_ket[i] : n_states_bra[i];
            info->allocate((int)qs.size());
            int i = 0;
            for (auto &mq : qs)
                info->quanta[i] = mq.first, info->n_states[i++] = mq.second;
        }
        info->n_states_total =
            accumulate(info->n_states, info->n_states + info->n, 0);
        return info;
    }
    int find_state(S q, int start = 0) const {
        auto p = lower_bound(quanta + start, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return (int)(p - quanta);
    }
    void sort_states() {
        vector<int> idx(n);
        vector<S> q(quanta, quanta + n);
        vector<ubond_t> nqb(n_states_bra, n_states_bra + n);
        vector<ubond_t> nqk(n_states_ket, n_states_ket + n);
        for (int i = 0; i < n; i++)
            idx[i] = i;
        sort(idx.begin(), idx.end(),
             [&q](int i, int j) { return q[i] < q[j]; });
        for (int i = 0; i < n; i++)
            quanta[i] = q[idx[i]], n_states_bra[i] = nqb[idx[i]],
            n_states_ket[i] = nqk[idx[i]];
        n_states_total[0] = 0;
        for (int i = 0; i < n - 1; i++) {
            n_states_total[i + 1] =
                n_states_total[i] + (uint32_t)n_states_bra[i] * n_states_ket[i];
            assert(n_states_total[i + 1] >= n_states_total[i]);
        }
    }
    uint32_t get_total_memory() const {
        if (n == 0)
            return 0;
        else {
            uint32_t tmem = n_states_total[n - 1] +
                            (uint32_t)n_states_bra[n - 1] * n_states_ket[n - 1];
            assert(tmem >= n_states_total[n - 1]);
            return tmem;
        }
    }
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0) {
            if (alloc == nullptr)
                alloc = ialloc;
            ptr = alloc->allocate(length * (sizeof(S) >> 2) + length +
                                  _DBL_MEM_SIZE(length));
        }
        quanta = (S *)ptr;
        n_states_bra = (ubond_t *)(ptr + length * (sizeof(S) >> 2));
        n_states_ket = (ubond_t *)(ptr + length * (sizeof(S) >> 2)) + length;
        n_states_total =
            ptr + length * (sizeof(S) >> 2) + _DBL_MEM_SIZE(length);
        n = length;
    }
    void deallocate() {
        assert(n != -1);
        alloc->deallocate((uint32_t *)quanta,
                          n * (sizeof(S) >> 2) + n + _DBL_MEM_SIZE(n));
        alloc = nullptr;
        quanta = nullptr;
        n_states_bra = nullptr;
        n_states_ket = nullptr;
        n_states_total = nullptr;
        n = -1;
    }
    void reallocate(int length) {
        uint32_t *ptr = alloc->reallocate(
            (uint32_t *)quanta, n * (sizeof(S) >> 2) + n + _DBL_MEM_SIZE(n),
            length * (sizeof(S) >> 2) + length + _DBL_MEM_SIZE(length));
        if (ptr == (uint32_t *)quanta)
            memmove(ptr + length * (sizeof(S) >> 2), (uint32_t *)n_states_bra,
                    (length + _DBL_MEM_SIZE(length)) * sizeof(uint32_t));
        else {
            memmove(
                ptr, (uint32_t *)quanta,
                (length * (sizeof(S) >> 2) + length + _DBL_MEM_SIZE(length)) *
                    sizeof(uint32_t));
            quanta = (S *)ptr;
        }
        n_states_bra = (ubond_t *)(ptr + length * (sizeof(S) >> 2));
        n_states_ket = (ubond_t *)(ptr + length * (sizeof(S) >> 2)) + length;
        n_states_total =
            ptr + length * (sizeof(S) >> 2) + _DBL_MEM_SIZE(length);
        n = length;
    }
    friend ostream &operator<<(ostream &os, const SparseMatrixInfo<S> &c) {
        os << "DQ=" << c.delta_quantum << " N=" << c.n
           << " SIZE=" << c.get_total_memory() << endl;
        for (int i = 0; i < c.n; i++)
            os << "BRA " << c.quanta[i].get_bra(c.delta_quantum) << " KET "
               << c.quanta[i].get_ket() << " [ " << (int)c.n_states_bra[i]
               << "x" << (int)c.n_states_ket[i] << " ]" << endl;
        return os;
    }
};

enum struct SparseMatrixTypes : uint8_t {
    Normal = 0,
    CSR = 1,
    Archived = 2,
    Delayed = 3
};

// Block-sparse Matrix
// Representing operator, wavefunction, density matrix and MPS tensors
template <typename S, typename FL> struct SparseMatrix {
    typedef typename GMatrix<FL>::FP FP;
    static const int cpx_sz = sizeof(FL) / sizeof(FP);
    shared_ptr<Allocator<FP>> alloc;
    shared_ptr<SparseMatrixInfo<S>> info;
    FL *data;
    FL factor;
    size_t total_memory;
    struct pair_uint32_t_hasher {
        size_t operator()(const pair<uint32_t, uint32_t> &p) const {
            return (((size_t)p.first) << 32) | p.second;
        }
    };
    SparseMatrix(const shared_ptr<Allocator<FP>> &alloc = nullptr)
        : info(nullptr), data(nullptr), factor(1.0), total_memory(0),
          alloc(alloc) {}
    virtual ~SparseMatrix() = default;
    virtual SparseMatrixTypes get_type() const {
        return SparseMatrixTypes::Normal;
    }
    virtual void load_data(istream &ifs, bool pointer_only = false) {
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&total_memory, sizeof(total_memory));
        if (pointer_only && total_memory != 0) {
            size_t psz;
            ifs.read((char *)&psz, sizeof(psz));
            assert(alloc == dalloc_<FP>() || alloc == nullptr);
            data = (FL *)(dalloc_<FP>()->data + psz);
        } else {
            data = (FL *)alloc->allocate(total_memory * cpx_sz);
            ifs.read((char *)data, sizeof(FL) * total_memory);
        }
    }
    void load_data(const string &filename, bool load_info = false,
                   const shared_ptr<Allocator<uint32_t>> &i_alloc = nullptr) {
        if (alloc == nullptr)
            alloc = dalloc_<FP>();
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("SparseMatrix:load_data on '" + filename +
                                "' failed.");
        if (load_info) {
            info = make_shared<SparseMatrixInfo<S>>(i_alloc);
            info->load_data(ifs);
        } else
            info = nullptr;
        load_data(ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("SparseMatrix:load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    virtual void save_data(ostream &ofs, bool pointer_only = false) const {
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&total_memory, sizeof(total_memory));
        if (pointer_only && total_memory != 0) {
            // for 1-site case with middle site transition
            // one can skip the post-middle-transition then
            // there can be some terms not allocated in stack memory
            // assert(alloc == dalloc_<FP>());
            size_t psz = (FP *)data - dalloc_<FP>()->data;
            ofs.write((char *)&psz, sizeof(psz));
        } else
            ofs.write((char *)data, sizeof(FL) * total_memory);
    }
    void save_data(const string &filename, bool save_info = false) const {
        if (Parsing::link_exists(filename))
            Parsing::remove_file(filename);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("SparseMatrix:save_data on '" + filename +
                                "' failed.");
        if (save_info)
            info->save_data(ofs);
        save_data(ofs);
        if (!ofs.good())
            throw runtime_error("SparseMatrix:save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    virtual shared_ptr<SparseMatrix>
    deep_copy(const shared_ptr<Allocator<FP>> &alloc = nullptr) const {
        shared_ptr<SparseMatrix> r = make_shared<SparseMatrix>(alloc);
        *r = *this;
        r->alloc = alloc == nullptr ? dalloc_<FP>() : alloc;
        r->data = (FL *)alloc->allocate(total_memory * cpx_sz);
        memcpy(r->data, data, sizeof(FL) * total_memory);
        return r;
    }
    virtual void copy_data_from(const shared_ptr<SparseMatrix> &other,
                                bool ref = false) {
        assert(total_memory == other->total_memory);
        memcpy(data, other->data, sizeof(FL) * total_memory);
    }
    virtual void selective_copy_from(const shared_ptr<SparseMatrix> &other,
                                     bool ref = false) {
        for (int i = 0, k; i < other->info->n; i++)
            if ((k = info->find_state(other->info->quanta[i])) != -1)
                memcpy(data + info->n_states_total[k],
                       other->data + other->info->n_states_total[i],
                       sizeof(FL) * ((size_t)info->n_states_bra[k] *
                                     info->n_states_ket[k]));
    }
    virtual void clear() { memset(data, 0, sizeof(FL) * total_memory); }
    virtual void allocate_like(const shared_ptr<SparseMatrix> &mat) {
        allocate(mat->info);
    }
    virtual void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                          FL *ptr = 0) {
        this->info = info;
        total_memory = info->get_total_memory();
        if (total_memory == 0)
            return;
        if (ptr == 0) {
            if (alloc == nullptr)
                alloc = dalloc_<FP>();
            data = (FL *)alloc->allocate(total_memory * cpx_sz);
            memset(data, 0, sizeof(FL) * total_memory);
        } else
            data = ptr;
    }
    virtual void deallocate() {
        if (alloc == nullptr)
            // this is the case when this sparse matrix data pointer
            // is an external pointer, shared by many matrices
            return;
        if (total_memory == 0) {
            assert(data == nullptr);
            return;
        }
        alloc->deallocate(data, total_memory * cpx_sz);
        alloc = nullptr;
        total_memory = 0;
        data = nullptr;
    }
    void reallocate(size_t length) {
        assert(alloc != nullptr);
        FL *ptr = (FL *)alloc->reallocate((FP *)data, total_memory * cpx_sz,
                                          length * cpx_sz);
        if (ptr != data && length != 0)
            memmove(ptr, data, length * sizeof(FL));
        total_memory = length;
        data = length == 0 ? nullptr : ptr;
    }
    void reallocate(shared_ptr<Allocator<FP>> new_alloc) {
        assert(new_alloc != nullptr && new_alloc != alloc);
        FL *ptr = (FL *)new_alloc->allocate(total_memory * cpx_sz);
        memcpy(ptr, data, total_memory * sizeof(FL));
        alloc->deallocate(data, total_memory * cpx_sz);
        alloc = new_alloc;
        data = total_memory == 0 ? nullptr : ptr;
    }
    GMatrix<FL> operator[](S q) const { return (*this)[info->find_state(q)]; }
    GMatrix<FL> operator[](int idx) const {
        assert(idx != -1);
        return GMatrix<FL>(data + info->n_states_total[idx],
                           (int)info->n_states_bra[idx],
                           (int)info->n_states_ket[idx]);
    }
    FL trace() const {
        FL r = 0;
        for (int i = 0; i < info->n; i++)
            r += this->operator[](i).trace();
        return r;
    }
    virtual FP norm() const {
        assert(total_memory <= (size_t)numeric_limits<MKL_INT>::max());
        return GMatrixFunctions<FL>::norm(
            GMatrix<FL>(data, (MKL_INT)total_memory, 1));
    }
    // ratio of zero elements to total size
    virtual FP sparsity() const {
        size_t nnz = 0;
        for (size_t i = 0; i < total_memory; i++)
            nnz += abs(this->data[i]) > TINY;
        return 1.0 - (FP)nnz / total_memory;
    }
    void iscale(FL d) const {
        assert(factor == (FP)1.0);
        assert(total_memory <= (size_t)numeric_limits<MKL_INT>::max());
        GMatrixFunctions<FL>::iscale(
            GMatrix<FL>(data, (MKL_INT)total_memory, 1), d);
    }
    void normalize() const { iscale(1 / norm()); }
    // K = L(l)C or C = L(l)S
    void left_split(shared_ptr<SparseMatrix> &left,
                    shared_ptr<SparseMatrix> &right, ubond_t bond_dim) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<shared_ptr<GTensor<FL>>> l, r;
        vector<shared_ptr<GTensor<FP>>> s;
        vector<S> qs;
        right_svd(qs, l, s, r, bond_dim);
        shared_ptr<SparseMatrixInfo<S>> winfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<SparseMatrixInfo<S>> linfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        winfo->is_fermion = info->is_fermion;
        winfo->is_wavefunction = info->is_wavefunction;
        winfo->delta_quantum = info->delta_quantum;
        winfo->allocate(info->n);
        for (int i = 0; i < winfo->n; i++) {
            winfo->quanta[i] = info->quanta[i];
            winfo->n_states_bra[i] = r[i]->shape[0];
            winfo->n_states_ket[i] = r[i]->shape[1];
        }
        winfo->sort_states();
        linfo->is_fermion = false;
        linfo->is_wavefunction = false;
        linfo->delta_quantum = (info->delta_quantum - info->delta_quantum)[0];
        linfo->allocate((int)l.size());
        for (int i = 0; i < linfo->n; i++) {
            linfo->quanta[i] = qs[i];
            linfo->n_states_bra[i] = l[i]->shape[0];
            linfo->n_states_ket[i] = l[i]->shape[1];
            // this is only necessary for compatibility with StackBlock
            if (l[i]->data->size() == 1 && xreal<FL>((*l[i]->data)[0]) < 0) {
                GMatrixFunctions<FL>::iscale(l[i]->ref(), -1);
                GMatrixFunctions<FP>::iscale(s[i]->ref(), -1);
            }
        }
        linfo->sort_states();
        left = make_shared<SparseMatrix>(d_alloc);
        right = make_shared<SparseMatrix>(d_alloc);
        left->allocate(linfo);
        right->allocate(winfo);
        for (int i = 0; i < winfo->n; i++) {
            GMatrix<FL> mm = (*right)[info->quanta[i]];
            GMatrixFunctions<FL>::copy(mm, r[i]->ref());
            int k =
                linfo->find_state(info->quanta[i].get_bra(info->delta_quantum));
            assert(s[k]->shape[0] == r[i]->shape[0]);
            for (int j = 0; j < r[i]->shape[0]; j++)
                GMatrixFunctions<FL>::iscale(GMatrix<FL>(&mm(j, 0), 1, mm.n),
                                             (*s[k]->data)[j], 1);
        }
        for (int i = 0; i < linfo->n; i++)
            GMatrixFunctions<FL>::copy((*left)[qs[i]], l[i]->ref());
    }
    // S = C(r)R or C = K(r)R
    void right_split(shared_ptr<SparseMatrix> &left,
                     shared_ptr<SparseMatrix> &right, ubond_t bond_dim) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<shared_ptr<GTensor<FL>>> l, r;
        vector<shared_ptr<GTensor<FP>>> s;
        vector<S> qs;
        left_svd(qs, l, s, r, bond_dim);
        shared_ptr<SparseMatrixInfo<S>> winfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        winfo->is_fermion = info->is_fermion;
        winfo->is_wavefunction = info->is_wavefunction;
        winfo->delta_quantum = info->delta_quantum;
        winfo->allocate(info->n);
        for (int i = 0; i < winfo->n; i++) {
            winfo->quanta[i] = info->quanta[i];
            winfo->n_states_bra[i] = l[i]->shape[0];
            winfo->n_states_ket[i] = l[i]->shape[1];
        }
        winfo->sort_states();
        rinfo->is_fermion = false;
        rinfo->is_wavefunction = false;
        rinfo->delta_quantum = (info->delta_quantum - info->delta_quantum)[0];
        rinfo->allocate((int)r.size());
        for (int i = 0; i < rinfo->n; i++) {
            rinfo->quanta[i] = qs[i];
            rinfo->n_states_bra[i] = r[i]->shape[0];
            rinfo->n_states_ket[i] = r[i]->shape[1];
            // this is only necessary for compatibility with StackBlock
            if (r[i]->data->size() == 1 && xreal<FL>((*r[i]->data)[0]) < 0) {
                GMatrixFunctions<FL>::iscale(r[i]->ref(), -1);
                GMatrixFunctions<FP>::iscale(s[i]->ref(), -1);
            }
        }
        rinfo->sort_states();
        left = make_shared<SparseMatrix>(d_alloc);
        right = make_shared<SparseMatrix>(d_alloc);
        left->allocate(winfo);
        right->allocate(rinfo);
        for (int i = 0; i < winfo->n; i++) {
            GMatrix<FL> mm = (*left)[info->quanta[i]];
            GMatrixFunctions<FL>::copy(mm, l[i]->ref());
            int k = rinfo->find_state(info->is_wavefunction
                                          ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket());
            assert(s[k]->shape[0] == l[i]->shape[1]);
            for (int j = 0; j < l[i]->shape[1]; j++)
                GMatrixFunctions<FL>::iscale(GMatrix<FL>(&mm(0, j), mm.m, 1),
                                             (*s[k]->data)[j], mm.n);
        }
        for (int i = 0; i < rinfo->n; i++)
            GMatrixFunctions<FL>::copy((*right)[qs[i]], r[i]->ref());
    }
    // return p = pinv(mat).T
    // only right pseudo: mat @ p.T = I
    shared_ptr<SparseMatrix> pseudo_inverse(ubond_t bond_dim, FP svd_eps = 1E-4,
                                            FP svd_cutoff = 1E-12) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        vector<shared_ptr<GTensor<FL>>> l, r;
        vector<shared_ptr<GTensor<FP>>> s;
        vector<S> qs;
        right_svd(qs, l, s, r, bond_dim, svd_eps);
        shared_ptr<SparseMatrix> pinv = make_shared<SparseMatrix>(d_alloc);
        pinv->allocate(
            make_shared<SparseMatrixInfo<S>>(info->deep_copy(i_alloc)));
        map<S, int> qs_mp;
        for (int i = 0; i < (int)qs.size(); i++)
            qs_mp[qs[i]] = i;
        for (int i = 0; i < info->n; i++) {
            GMatrix<FL> mm = (*pinv)[info->quanta[i]];
            S ql = info->quanta[i].get_bra(info->delta_quantum);
            int k = qs_mp.at(ql);
            GMatrix<FL> ll = l[k]->ref(), rr = r[i]->ref();
            GMatrix<FP> ss = s[k]->ref();
            for (MKL_INT j = 0; j < r[i]->shape[0]; j++)
                if (abs((*s[k]->data)[j]) > svd_cutoff)
                    GMatrixFunctions<FL>::multiply(
                        GMatrix<FL>(&ll(0, j), ll.m, ll.n), false,
                        GMatrix<FL>(&rr(j, 0), 1, rr.n), false, mm,
                        1.0 / (*s[k]->data)[j], 1.0);
        }
        return pinv;
    }
    // l will have the same number of non-zero blocks as this matrix
    // s will be labelled by right q labels
    void left_svd(vector<S> &rqs, vector<shared_ptr<GTensor<FL>>> &l,
                  vector<shared_ptr<GTensor<FP>>> &s,
                  vector<shared_ptr<GTensor<FL>>> &r, ubond_t bond_dim = 0,
                  FP svd_eps = 0) const {
        map<S, MKL_INT> qs_mp;
        for (int i = 0; i < info->n; i++) {
            S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                        : info->quanta[i].get_ket();
            qs_mp[q] += (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
        }
        int nr = (int)qs_mp.size(), k = 0;
        rqs.resize(nr);
        vector<MKL_INT> tmp(nr + 1, 0);
        for (auto &mp : qs_mp)
            rqs[k] = mp.first, tmp[k + 1] = mp.second, k++;
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        FL *dt = (FL *)dalloc_<FP>()->allocate(tmp[nr] * cpx_sz);
        vector<MKL_INT> it(nr, 0), sz(nr, 0);
        for (int i = 0; i < info->n; i++) {
            S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                        : info->quanta[i].get_ket();
            size_t ir = lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
            MKL_INT n_states =
                (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(dt + (tmp[ir] + it[ir]), data + info->n_states_total[i],
                   n_states * sizeof(FL));
            sz[ir] = info->n_states_ket[i];
            it[ir] += n_states;
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        vector<shared_ptr<GTensor<FL>>> merged_l(nr);
        r.resize(nr);
        s.resize(nr);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int ir = 0; ir < nr; ir++) {
            MKL_INT nxr = sz[ir], nxl = (tmp[ir + 1] - tmp[ir]) / nxr;
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0);
            MKL_INT nxk = min(nxl, nxr);
            shared_ptr<GTensor<FL>> tsl =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxl, nxk});
            shared_ptr<GTensor<FP>> tss =
                make_shared<GTensor<FP>>(vector<MKL_INT>{nxk});
            shared_ptr<GTensor<FL>> tsr =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxk, nxr});
            if (svd_eps != 0)
                GMatrixFunctions<FL>::accurate_svd(
                    GMatrix<FL>(dt + tmp[ir], nxl, nxr), tsl->ref(),
                    tss->ref().flip_dims(), tsr->ref(), svd_eps);
            else
                GMatrixFunctions<FL>::svd(GMatrix<FL>(dt + tmp[ir], nxl, nxr),
                                          tsl->ref(), tss->ref().flip_dims(),
                                          tsr->ref());
            merged_l[ir] = tsl;
            s[ir] = tss;
            r[ir] = tsr;
        }
        threading->activate_normal();
        vector<FP> svals;
        for (int ir = 0; ir < nr; ir++)
            svals.insert(svals.end(), s[ir]->data->begin(), s[ir]->data->end());
        if (bond_dim != 0 && svals.size() > bond_dim) {
            sort(svals.begin(), svals.end());
            FP small = svals[svals.size() - bond_dim - 1];
            for (int ir = 0; ir < nr; ir++)
                for (MKL_INT j = 1; j < (MKL_INT)s[ir]->data->size(); j++)
                    if ((*s[ir]->data)[j] <= small) {
                        merged_l[ir]->truncate_right(j);
                        s[ir]->truncate(j);
                        r[ir]->truncate_left(j);
                        break;
                    }
        }
        memset(it.data(), 0, sizeof(MKL_INT) * nr);
        for (int i = 0; i < info->n; i++) {
            S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                        : info->quanta[i].get_ket();
            size_t ir = lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
            shared_ptr<GTensor<FL>> tsl =
                make_shared<GTensor<FL>>(vector<MKL_INT>{
                    (MKL_INT)info->n_states_bra[i], merged_l[ir]->shape[1]});
            memcpy(tsl->data->data(), merged_l[ir]->data->data() + it[ir],
                   tsl->size() * sizeof(FL));
            it[ir] += (MKL_INT)tsl->size();
            l.push_back(tsl);
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == merged_l[ir]->size());
        dalloc_<FP>()->deallocate(dt, tmp[nr] * cpx_sz);
    }
    // r will have the same number of non-zero blocks as this matrix
    // s will be labelled by left q labels
    void right_svd(vector<S> &lqs, vector<shared_ptr<GTensor<FL>>> &l,
                   vector<shared_ptr<GTensor<FP>>> &s,
                   vector<shared_ptr<GTensor<FL>>> &r, ubond_t bond_dim = 0,
                   FP svd_eps = 0) const {
        map<S, MKL_INT> qs_mp;
        for (int i = 0; i < info->n; i++) {
            S q = info->quanta[i].get_bra(info->delta_quantum);
            qs_mp[q] += (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
        }
        int nl = (int)qs_mp.size(), p = 0;
        lqs.resize(nl);
        vector<MKL_INT> tmp(nl + 1, 0);
        for (auto &mp : qs_mp)
            lqs[p] = mp.first, tmp[p + 1] = mp.second, p++;
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        FL *dt = (FL *)dalloc_<FP>()->allocate(tmp[nl] * cpx_sz);
        vector<MKL_INT> it(nl, 0), sz(nl, 0);
        for (int i = 0; i < info->n; i++) {
            S q = info->quanta[i].get_bra(info->delta_quantum);
            size_t il = lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
            MKL_INT nxl = info->n_states_bra[i],
                    nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0);
            MKL_INT inr = info->n_states_ket[i];
            for (MKL_INT k = 0; k < nxl; k++)
                memcpy(dt + (tmp[il] + it[il] + k * nxr),
                       data + (info->n_states_total[i] + k * inr),
                       inr * sizeof(FL));
            sz[il] = nxl;
            it[il] += inr;
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == (tmp[il + 1] - tmp[il]) / sz[il]);
        vector<shared_ptr<GTensor<FL>>> merged_r(nl);
        l.resize(nl);
        s.resize(nl);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int il = 0; il < nl; il++) {
            MKL_INT nxl = sz[il], nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0);
            MKL_INT nxk = min(nxl, nxr);
            shared_ptr<GTensor<FL>> tsl =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxl, nxk});
            shared_ptr<GTensor<FP>> tss =
                make_shared<GTensor<FP>>(vector<MKL_INT>{nxk});
            shared_ptr<GTensor<FL>> tsr =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxk, nxr});
            if (svd_eps != 0)
                GMatrixFunctions<FL>::accurate_svd(
                    GMatrix<FL>(dt + tmp[il], nxl, nxr), tsl->ref(),
                    tss->ref().flip_dims(), tsr->ref(), svd_eps);
            else
                GMatrixFunctions<FL>::svd(GMatrix<FL>(dt + tmp[il], nxl, nxr),
                                          tsl->ref(), tss->ref().flip_dims(),
                                          tsr->ref());
            l[il] = tsl;
            s[il] = tss;
            merged_r[il] = tsr;
        }
        threading->activate_normal();
        vector<FP> svals;
        for (int il = 0; il < nl; il++)
            svals.insert(svals.end(), s[il]->data->begin(), s[il]->data->end());
        if (bond_dim != 0 && svals.size() > bond_dim) {
            sort(svals.begin(), svals.end());
            FP small = svals[svals.size() - bond_dim - 1];
            for (int il = 0; il < nl; il++)
                for (MKL_INT j = 1; j < (MKL_INT)s[il]->data->size(); j++)
                    if ((*s[il]->data)[j] <= small) {
                        l[il]->truncate_right(j);
                        s[il]->truncate(j);
                        merged_r[il]->truncate_left(j);
                        break;
                    }
        }
        memset(it.data(), 0, sizeof(MKL_INT) * nl);
        for (int i = 0; i < info->n; i++) {
            S q = info->quanta[i].get_bra(info->delta_quantum);
            size_t il = lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
            shared_ptr<GTensor<FL>> tsr =
                make_shared<GTensor<FL>>(vector<MKL_INT>{
                    merged_r[il]->shape[0], (MKL_INT)info->n_states_ket[i]});
            MKL_INT inr = info->n_states_ket[i], ixr = merged_r[il]->shape[1];
            MKL_INT inl = merged_r[il]->shape[0];
            for (MKL_INT k = 0; k < inl; k++)
                memcpy(tsr->data->data() + k * inr,
                       merged_r[il]->data->data() + (it[il] + k * ixr),
                       inr * sizeof(FL));
            it[il] += inr;
            r.push_back(tsr);
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == merged_r[il]->shape[1]);
        dalloc_<FP>()->deallocate(dt, tmp[nl] * cpx_sz);
    }
    void left_canonicalize(const shared_ptr<SparseMatrix> &rmat) {
        int nr = rmat->info->n, n = info->n;
        vector<MKL_INT> tmp(nr + 1, 0);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            assert(ir != -1);
            tmp[ir + 1] +=
                (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
        }
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        FL *dt = (FL *)dalloc_<FP>()->allocate(tmp[nr] * cpx_sz);
        vector<MKL_INT> it(nr, 0);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            MKL_INT n_states =
                (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(dt + (tmp[ir] + it[ir]), data + info->n_states_total[i],
                   n_states * sizeof(FL));
            it[ir] += n_states;
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        for (int ir = 0; ir < nr; ir++) {
            MKL_INT nxr = rmat->info->n_states_ket[ir],
                    nxl = (tmp[ir + 1] - tmp[ir]) / nxr;
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0 && nxl >= nxr);
            GMatrixFunctions<FL>::qr(GMatrix<FL>(dt + tmp[ir], nxl, nxr),
                                     GMatrix<FL>(dt + tmp[ir], nxl, nxr),
                                     (*rmat)[ir]);
        }
        memset(it.data(), 0, sizeof(MKL_INT) * nr);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            MKL_INT n_states =
                (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(data + info->n_states_total[i], dt + (tmp[ir] + it[ir]),
                   n_states * sizeof(FL));
            it[ir] += n_states;
        }
        dalloc_<FP>()->deallocate(dt, tmp[nr] * cpx_sz);
    }
    void right_canonicalize(const shared_ptr<SparseMatrix> &lmat) {
        int nl = lmat->info->n, n = info->n;
        vector<MKL_INT> tmp(nl + 1, 0);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            assert(il != -1);
            tmp[il + 1] +=
                (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
        }
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        FL *dt = (FL *)dalloc_<FP>()->allocate(tmp[nl] * cpx_sz);
        vector<MKL_INT> it(nl, 0);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            MKL_INT nxl = info->n_states_bra[i],
                    nxr = (tmp[il + 1] - tmp[il]) / nxl;
            MKL_INT inr = info->n_states_ket[i];
            for (MKL_INT k = 0; k < nxl; k++)
                memcpy(dt + (tmp[il] + it[il] + k * nxr),
                       data + info->n_states_total[i] + k * inr,
                       inr * sizeof(FL));
            it[il] += inr * nxl;
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == tmp[il + 1] - tmp[il]);
        for (int il = 0; il < nl; il++) {
            MKL_INT nxl = lmat->info->n_states_bra[il],
                    nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0 && nxr >= nxl);
            GMatrixFunctions<FL>::lq(GMatrix<FL>(dt + tmp[il], nxl, nxr),
                                     (*lmat)[il],
                                     GMatrix<FL>(dt + tmp[il], nxl, nxr));
        }
        memset(it.data(), 0, sizeof(MKL_INT) * nl);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            MKL_INT nxl = info->n_states_bra[i],
                    nxr = (tmp[il + 1] - tmp[il]) / nxl;
            MKL_INT inr = info->n_states_ket[i];
            for (MKL_INT k = 0; k < nxl; k++)
                memcpy(data + info->n_states_total[i] + k * inr,
                       dt + (tmp[il] + it[il] + k * nxr), inr * sizeof(FL));
            it[il] += inr * nxl;
        }
        dalloc_<FP>()->deallocate(dt, tmp[nl] * cpx_sz);
    }
    shared_ptr<SparseMatrix>
    left_multiply(const shared_ptr<SparseMatrix> &lmat, const StateInfo<S> &l,
                  const StateInfo<S> &m, const StateInfo<S> &r,
                  const StateInfo<S> &old_fused,
                  const shared_ptr<typename StateInfo<S>::ConnectionInfo>
                      &old_fused_cinfo,
                  const StateInfo<S> &new_fused) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrixInfo<S>> xinfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<StateInfo<S>> rsi = info->extract_state_info(true);
        xinfo->initialize(new_fused, *rsi, info->delta_quantum,
                          info->is_fermion, info->is_wavefunction);
        shared_ptr<SparseMatrix> xmat = make_shared<SparseMatrix>(d_alloc);
        xmat->allocate(xinfo);
        assert(xinfo->n <= info->n);
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ix = xinfo->find_state(info->quanta[i]);
            if (ix == -1)
                continue;
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = old_fused_cinfo->acc_n_states[ib + 1];
            MKL_INT p = info->n_states_total[i], xp = xinfo->n_states_total[i];
            for (int bb = old_fused_cinfo->acc_n_states[ib]; bb < bbed; bb++) {
                uint32_t ibba = old_fused_cinfo->ij_indices[bb].first,
                         ibbb = old_fused_cinfo->ij_indices[bb].second;
                int il = lmat->info->find_state(l.quanta[ibba]);
                MKL_INT lp = (MKL_INT)m.n_states[ibbb] * r.n_states[ik];
                if (il != -1) {
                    assert(lmat->info->n_states_ket[il] == l.n_states[ibba]);
                    GMatrixFunctions<FL>::multiply(
                        (*lmat)[il], false,
                        GMatrix<FL>(data + p, l.n_states[ibba], lp), false,
                        GMatrix<FL>(xmat->data + xp,
                                    lmat->info->n_states_bra[il], lp),
                        lmat->factor, 0.0);
                    xp += lmat->info->n_states_bra[il] * lp;
                }
                p += l.n_states[ibba] * lp;
            }
            // here possible error because dot == 2, dynamic canonicalize
            // assumes there is a two-site tensor
            // then the inferred bond_dims can be wrong
            assert(p == (i != info->n - 1 ? info->n_states_total[i + 1]
                                          : total_memory));
            assert(xp == (ix != xinfo->n - 1
                              ? xmat->info->n_states_total[ix + 1]
                              : xmat->total_memory));
        }
        return xmat;
    }
    shared_ptr<SparseMatrix>
    right_multiply(const shared_ptr<SparseMatrix> &rmat, const StateInfo<S> &l,
                   const StateInfo<S> &m, const StateInfo<S> &r,
                   const StateInfo<S> &old_fused,
                   const shared_ptr<typename StateInfo<S>::ConnectionInfo>
                       &old_fused_cinfo,
                   const StateInfo<S> &new_fused) const {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<SparseMatrixInfo<S>> xinfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        shared_ptr<StateInfo<S>> lsi = info->extract_state_info(false);
        xinfo->initialize(*lsi, new_fused, info->delta_quantum,
                          info->is_fermion, info->is_wavefunction);
        shared_ptr<SparseMatrix> xmat = make_shared<SparseMatrix>(d_alloc);
        xmat->allocate(xinfo);
        assert(xinfo->n <= info->n);
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ix = xinfo->find_state(info->quanta[i]);
            if (ix == -1)
                continue;
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int ikn = new_fused.find_state(ket);
            int kked = old_fused_cinfo->acc_n_states[ik + 1];
            MKL_INT p = info->n_states_total[i], xp = xinfo->n_states_total[ix];
            for (int kk = old_fused_cinfo->acc_n_states[ik]; kk < kked; kk++) {
                uint32_t ikka = old_fused_cinfo->ij_indices[kk].first,
                         ikkb = old_fused_cinfo->ij_indices[kk].second;
                int ir = rmat->info->find_state(r.quanta[ikkb]);
                MKL_INT lp = (MKL_INT)m.n_states[ikka] * r.n_states[ikkb];
                if (ir != -1) {
                    MKL_INT lpx = (MKL_INT)m.n_states[ikka] *
                                  rmat->info->n_states_ket[ir];
                    assert(rmat->info->n_states_bra[ir] == r.n_states[ikkb]);
                    for (ubond_t j = 0; j < l.n_states[ib]; j++)
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>(data + p + j * old_fused.n_states[ik],
                                        m.n_states[ikka], r.n_states[ikkb]),
                            false, (*rmat)[ir], false,
                            GMatrix<FL>(
                                xmat->data + xp + j * new_fused.n_states[ikn],
                                m.n_states[ikka], rmat->info->n_states_ket[ir]),
                            rmat->factor, 0.0);
                    xp += lpx;
                }
                p += lp;
            }
        }
        return xmat;
    }
    // lmat must be square-block diagonal
    void left_multiply_inplace(
        const shared_ptr<SparseMatrix> &lmat, const StateInfo<S> &l,
        const StateInfo<S> &m, const StateInfo<S> &r,
        const StateInfo<S> &old_fused,
        const shared_ptr<typename StateInfo<S>::ConnectionInfo>
            &old_fused_cinfo) const {
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = old_fused_cinfo->acc_n_states[ib + 1];
            MKL_INT p = info->n_states_total[i];
            for (int bb = old_fused_cinfo->acc_n_states[ib]; bb < bbed; bb++) {
                uint32_t ibba = old_fused_cinfo->ij_indices[bb].first,
                         ibbb = old_fused_cinfo->ij_indices[bb].second;
                int il = lmat->info->find_state(l.quanta[ibba]);
                MKL_INT lp = (MKL_INT)m.n_states[ibbb] * r.n_states[ik];
                if (il != -1) {
                    assert(lmat->info->n_states_bra[il] ==
                           lmat->info->n_states_ket[il]);
                    assert(lmat->info->n_states_bra[il] == l.n_states[ibba]);
                    GMatrix<FL> tmp(nullptr, l.n_states[ibba], lp);
                    tmp.allocate();
                    GMatrixFunctions<FL>::multiply(
                        (*lmat)[il], false,
                        GMatrix<FL>(data + p, l.n_states[ibba], lp), false, tmp,
                        lmat->factor, 0.0);
                    memcpy(data + p, tmp.data, sizeof(FL) * tmp.size());
                    tmp.deallocate();
                }
                p += l.n_states[ibba] * lp;
            }
            assert(p == (i != info->n - 1 ? info->n_states_total[i + 1]
                                          : total_memory));
        }
    }
    // rmat must be square-block diagonal
    void right_multiply_inplace(
        const shared_ptr<SparseMatrix> &rmat, const StateInfo<S> &l,
        const StateInfo<S> &m, const StateInfo<S> &r,
        const StateInfo<S> &old_fused,
        const shared_ptr<typename StateInfo<S>::ConnectionInfo>
            &old_fused_cinfo) const {
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int kked = old_fused_cinfo->acc_n_states[ik + 1];
            MKL_INT p = info->n_states_total[i];
            for (int kk = old_fused_cinfo->acc_n_states[ik]; kk < kked; kk++) {
                uint32_t ikka = old_fused_cinfo->ij_indices[kk].first,
                         ikkb = old_fused_cinfo->ij_indices[kk].second;
                int ir = rmat->info->find_state(r.quanta[ikkb]);
                MKL_INT lp = (MKL_INT)m.n_states[ikka] * r.n_states[ikkb];
                if (ir != -1) {
                    assert(rmat->info->n_states_bra[ir] ==
                           rmat->info->n_states_ket[ir]);
                    assert(rmat->info->n_states_bra[ir] == r.n_states[ikkb]);
                    GMatrix<FL> tmp(nullptr, m.n_states[ikka],
                                    r.n_states[ikkb]);
                    tmp.allocate();
                    for (ubond_t j = 0; j < l.n_states[ib]; j++) {
                        GMatrixFunctions<FL>::multiply(
                            GMatrix<FL>(data + p + j * old_fused.n_states[ik],
                                        m.n_states[ikka], r.n_states[ikkb]),
                            false, (*rmat)[ir], false, tmp, rmat->factor, 0.0);
                        memcpy(data + p + j * old_fused.n_states[ik], tmp.data,
                               sizeof(FL) * tmp.size());
                    }
                    tmp.deallocate();
                }
                p += lp;
            }
        }
    }
    void randomize(FP a = 0.0, FP b = 1.0) const {
        Random::fill<FP>((FP *)data, total_memory * cpx_sz, a, b);
    }
    // Contract two SparseMatrix
    void contract(const shared_ptr<SparseMatrix> &lmat,
                  const shared_ptr<SparseMatrix> &rmat,
                  bool trace_right = false) {
        if (info->is_wavefunction) {
            // wfn = wfn x rot
            if (lmat->info->is_wavefunction)
                for (int i = 0; i < info->n; i++) {
                    int il = lmat->info->find_state(info->quanta[i]);
                    int ir = rmat->info->find_state(-info->quanta[i].get_ket());
                    if (il != -1 && ir != -1)
                        GMatrixFunctions<FL>::multiply(
                            (*lmat)[il], false, (*rmat)[ir], false, (*this)[i],
                            lmat->factor * rmat->factor, 0.0);
                }
            // wfn = rot x wfn
            else
                for (int i = 0; i < info->n; i++) {
                    int il = lmat->info->find_state(
                        info->quanta[i].get_bra(info->delta_quantum));
                    int ir = rmat->info->find_state(info->quanta[i]);
                    if (il != -1 && ir != -1)
                        GMatrixFunctions<FL>::multiply(
                            (*lmat)[il], false, (*rmat)[ir], false, (*this)[i],
                            lmat->factor * rmat->factor, 0.0);
                }
        } else {
            // rot = trace_right ? wfn x wfn.H : wfn.H x wfn
            assert(lmat->info->is_wavefunction && rmat->info->is_wavefunction);
            clear();
            for (int il = 0; il < lmat->info->n; il++) {
                int ir = rmat->info->find_state(lmat->info->quanta[il]);
                int i = info->find_state(
                    trace_right ? lmat->info->quanta[il].get_bra(
                                      lmat->info->delta_quantum)
                                : -lmat->info->quanta[il].get_ket());
                if (ir != -1 && i != -1)
                    GMatrixFunctions<FL>::multiply(
                        (*lmat)[il], !trace_right ? 3 : 0, (*rmat)[ir],
                        trace_right ? 3 : 0, (*this)[i],
                        trace_right ? lmat->factor * xconj<FL>(rmat->factor)
                                    : xconj<FL>(lmat->factor) * rmat->factor,
                        1.0);
            }
        }
    }
    // Change from [l x (fused m and r)] to [(fused l and m) x r]
    void
    swap_to_fused_left(const shared_ptr<SparseMatrix> &mat,
                       const StateInfo<S> &l, const StateInfo<S> &m,
                       const StateInfo<S> &r, const StateInfo<S> &old_fused,
                       const shared_ptr<typename StateInfo<S>::ConnectionInfo>
                           &old_fused_cinfo,
                       const StateInfo<S> &new_fused,
                       const shared_ptr<typename StateInfo<S>::ConnectionInfo>
                           &new_fused_cinfo,
                       const shared_ptr<CG<S>> &cg) {
        assert(mat->info->is_wavefunction);
        factor = mat->factor;
        // for SU2 with target 2S != 0, for each l m r there can be multiple mr
        // mp is the three-index wavefunction
        unordered_map<pair<uint32_t, uint32_t>,
                      map<uint16_t, vector<pair<MKL_INT, int>>>,
                      pair_uint32_t_hasher>
            mp;
        mp.reserve(mat->info->n);
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = -mat->info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int kked = old_fused_cinfo->acc_n_states[ik + 1];
            MKL_INT p = mat->info->n_states_total[i];
            for (int kk = old_fused_cinfo->acc_n_states[ik]; kk < kked; kk++) {
                uint32_t ikka = old_fused_cinfo->ij_indices[kk].first,
                         ikkb = old_fused_cinfo->ij_indices[kk].second;
                MKL_INT lp = (MKL_INT)m.n_states[ikka] * r.n_states[ikkb];
                mp[make_pair(ib, ikka)][ikkb].push_back(make_pair(p, ik));
                p += lp;
            }
        }
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = -info->quanta[i].get_ket();
            int ib = new_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = new_fused_cinfo->acc_n_states[ib + 1];
            FL *ptr = data + info->n_states_total[i];
            for (int bb = new_fused_cinfo->acc_n_states[ib]; bb < bbed; bb++) {
                uint32_t ibba = new_fused_cinfo->ij_indices[bb].first,
                         ibbb = new_fused_cinfo->ij_indices[bb].second;
                MKL_INT lp = (MKL_INT)m.n_states[ibbb] * r.n_states[ik];
                S bra_l = l.quanta[ibba], bra_m = m.quanta[ibbb];
                if (mp.count(new_fused_cinfo->ij_indices[bb]) &&
                    mp[new_fused_cinfo->ij_indices[bb]].count(ik))
                    for (pair<MKL_INT, int> &t :
                         mp.at(new_fused_cinfo->ij_indices[bb]).at(ik)) {
                        S ket_mr = old_fused.quanta[t.second];
                        FP factor =
                            (FP)(cg->racah(bra_l, bra_m, info->delta_quantum,
                                           ket, bra, ket_mr) *
                                 sqrt(1.0 * bra.multiplicity() *
                                      ket_mr.multiplicity()));
                        for (ubond_t j = 0; j < l.n_states[ibba]; j++)
                            GMatrixFunctions<FL>::iadd(
                                GMatrix<FL>(ptr + j * lp, lp, 1),
                                GMatrix<FL>(
                                    mat->data + t.first +
                                        j * old_fused.n_states[t.second],
                                    lp, 1),
                                factor);
                    }
                ptr += (size_t)l.n_states[ibba] * lp;
            }
            assert(ptr - data == (i != info->n - 1 ? info->n_states_total[i + 1]
                                                   : total_memory));
        }
    }
    // Change from [(fused l and m) x r] to [l x (fused m and r)]
    void
    swap_to_fused_right(const shared_ptr<SparseMatrix> &mat,
                        const StateInfo<S> &l, const StateInfo<S> &m,
                        const StateInfo<S> &r, const StateInfo<S> &old_fused,
                        const shared_ptr<typename StateInfo<S>::ConnectionInfo>
                            &old_fused_cinfo,
                        const StateInfo<S> &new_fused,
                        const shared_ptr<typename StateInfo<S>::ConnectionInfo>
                            &new_fused_cinfo,
                        const shared_ptr<CG<S>> &cg) {
        assert(mat->info->is_wavefunction);
        factor = mat->factor;
        unordered_map<pair<uint32_t, uint32_t>,
                      map<uint16_t, vector<tuple<MKL_INT, MKL_INT, int>>>,
                      pair_uint32_t_hasher>
            mp;
        mp.reserve(mat->info->n);
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = -mat->info->quanta[i].get_ket();
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = old_fused_cinfo->acc_n_states[ib + 1];
            MKL_INT p = mat->info->n_states_total[i];
            for (int bb = old_fused_cinfo->acc_n_states[ib]; bb < bbed; bb++) {
                uint32_t ibba = old_fused_cinfo->ij_indices[bb].first,
                         ibbb = old_fused_cinfo->ij_indices[bb].second;
                MKL_INT lp = (MKL_INT)m.n_states[ibbb] * r.n_states[ik];
                mp[make_pair(ibbb, ik)][ibba].push_back(make_tuple(p, lp, ib));
                p += l.n_states[ibba] * lp;
            }
            assert(p == (i != mat->info->n - 1
                             ? mat->info->n_states_total[i + 1]
                             : mat->total_memory));
        }
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = -info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = new_fused.find_state(ket);
            int kked = new_fused_cinfo->acc_n_states[ik + 1];
            FL *ptr = data + info->n_states_total[i];
            MKL_INT lp = new_fused.n_states[ik];
            for (int kk = new_fused_cinfo->acc_n_states[ik]; kk < kked; kk++) {
                uint32_t ikka = new_fused_cinfo->ij_indices[kk].first,
                         ikkb = new_fused_cinfo->ij_indices[kk].second;
                S ket_m = m.quanta[ikka], ket_r = r.quanta[ikkb];
                if (mp.count(new_fused_cinfo->ij_indices[kk]) &&
                    mp[new_fused_cinfo->ij_indices[kk]].count(ib))
                    for (tuple<MKL_INT, MKL_INT, int> &t :
                         mp.at(new_fused_cinfo->ij_indices[kk]).at(ib)) {
                        S bra_lm = old_fused.quanta[get<2>(t)];
                        FP factor =
                            (FP)(cg->racah(ket_r, ket_m, info->delta_quantum,
                                           bra, ket, bra_lm) *
                                 sqrt(1.0 * ket.multiplicity() *
                                      bra_lm.multiplicity()));
                        for (ubond_t j = 0; j < l.n_states[ib]; j++) {
                            GMatrixFunctions<FL>::iadd(
                                GMatrix<FL>(ptr + j * lp, (MKL_INT)get<1>(t),
                                            1),
                                GMatrix<FL>(mat->data + get<0>(t) +
                                                j * get<1>(t),
                                            (MKL_INT)get<1>(t), 1),
                                factor);
                        }
                    }
                ptr += (size_t)m.n_states[ikka] * r.n_states[ikkb];
            }
        }
    }
    friend ostream &operator<<(ostream &os, const SparseMatrix &c) {
        os << "DATA = [ ";
        for (size_t i = 0; i < c.total_memory; i++)
            os << setw(20) << setprecision(14) << c.data[i] << " ";
        os << "]"
           << " FACTOR = ";
        os << setw(20) << setprecision(14) << c.factor << endl;
        return os;
    }
};

template <typename S, typename FL> struct SparseMatrixGroup {
    typedef typename GMatrix<FL>::FP FP;
    static const int cpx_sz = sizeof(FL) / sizeof(FP);
    shared_ptr<Allocator<FP>> alloc;
    vector<shared_ptr<SparseMatrixInfo<S>>> infos;
    vector<size_t> offsets;
    FL *data;
    size_t total_memory;
    int n;
    SparseMatrixGroup(const shared_ptr<Allocator<FP>> &alloc = nullptr)
        : infos(), offsets(), data(nullptr), total_memory(), alloc(alloc) {}
    void load_data(const string &filename, bool load_info = false,
                   const shared_ptr<Allocator<uint32_t>> &i_alloc = nullptr) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("SparseMatrixGroup::load_data on '" + filename +
                                "' failed.");
        ifs.read((char *)&n, sizeof(n));
        infos.resize(n);
        offsets.resize(n);
        ifs.read((char *)&offsets[0], sizeof(size_t) * n);
        if (load_info)
            for (int i = 0; i < n; i++) {
                infos[i] = make_shared<SparseMatrixInfo<S>>(i_alloc);
                infos[i]->load_data(ifs);
            }
        ifs.read((char *)&total_memory, sizeof(total_memory));
        if (alloc == nullptr)
            alloc = dalloc_<FP>();
        data = (FL *)alloc->allocate(total_memory * cpx_sz);
        ifs.read((char *)data, sizeof(FL) * total_memory);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("SparseMatrixGroup::load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    void save_data(const string &filename, bool save_info = false) const {
        if (Parsing::link_exists(filename))
            Parsing::remove_file(filename);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("SparseMatrixGroup::save_data on '" + filename +
                                "' failed.");
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)&offsets[0], sizeof(size_t) * n);
        if (save_info)
            for (int i = 0; i < n; i++)
                infos[i]->save_data(ofs);
        ofs.write((char *)&total_memory, sizeof(total_memory));
        ofs.write((char *)data, sizeof(FL) * total_memory);
        if (!ofs.good())
            throw runtime_error("SparseMatrixGroup::save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    void allocate_like(const shared_ptr<SparseMatrixGroup> &mat) {
        allocate(mat->infos);
    }
    void allocate(const vector<shared_ptr<SparseMatrixInfo<S>>> &infos,
                  FL *ptr = 0) {
        this->infos = infos;
        n = (int)infos.size();
        offsets.resize(n);
        if (n != 0) {
            offsets[0] = 0;
            for (size_t i = 0; i < n - 1; i++)
                offsets[i + 1] = offsets[i] + infos[i]->get_total_memory();
            total_memory = offsets[n - 1] + infos[n - 1]->get_total_memory();
        } else {
            total_memory = 0;
            data = nullptr;
            return;
        }
        if (ptr == 0) {
            if (alloc == nullptr)
                alloc = dalloc_<FP>();
            data = (FL *)alloc->allocate(total_memory * cpx_sz);
            memset(data, 0, sizeof(FL) * total_memory);
        } else
            data = ptr;
    }
    void deallocate() {
        if (alloc == nullptr)
            // this is the case when this sparse matrix data pointer
            // is an external pointer, shared by many matrices
            return;
        if (total_memory == 0) {
            assert(data == nullptr);
            return;
        }
        alloc->deallocate(data, total_memory * cpx_sz);
        alloc = nullptr;
        total_memory = 0;
        data = nullptr;
    }
    void deallocate_infos() {
        for (int i = n - 1; i >= 0; i--)
            if (infos[i]->n != -1)
                infos[i]->deallocate();
    }
    void randomize(FP a = 0.0, FP b = 1.0) const {
        Random::fill<FP>((FP *)data, total_memory * cpx_sz, a, b);
    }
    vector<pair<S, FP>> delta_quanta() const {
        vector<pair<S, FP>> r(n);
        for (int i = 0; i < n; i++)
            r[i] = make_pair(infos[i]->delta_quantum, (*this)[i]->norm());
        sort(r.begin(), r.end(),
             [](const pair<S, FP> &a, const pair<S, FP> &b) {
                 return a.second > b.second;
             });
        return r;
    }
    static vector<pair<S, FP>>
    merge_delta_quanta(const vector<pair<S, FP>> &a,
                       const vector<pair<S, FP>> &b) {
        vector<pair<S, FP>> r(a);
        r.insert(r.end(), b.begin(), b.end());
        sort(r.begin(), r.end(),
             [](const pair<S, FP> &a, const pair<S, FP> &b) {
                 return a.second > b.second;
             });
        int j = 0;
        for (int i = 1; i < (int)r.size(); i++)
            if (r[i].first == r[j].first)
                r[j].second = sqrt(
                    abs(r[j].second * r[j].second + r[i].second * r[i].second));
            else
                r[j++] = r[i];
        r.resize(r.size() == 0 ? 0 : j + 1);
        return r;
    }
    FP norm() const {
        if (total_memory <= (size_t)numeric_limits<MKL_INT>::max())
            return GMatrixFunctions<FL>::norm(
                GMatrix<FL>(data, (MKL_INT)total_memory, 1));
        else {
            FP normsq = 0.0;
            for (int i = 0; i < n; i++) {
                assert((*this)[i]->total_memory <=
                       (size_t)numeric_limits<MKL_INT>::max());
                FP norm = (*this)[i]->norm();
                normsq += norm * norm;
            }
            return sqrt(abs(normsq));
        }
    }
    void iscale(FL d) const {
        if (total_memory <= (size_t)numeric_limits<MKL_INT>::max())
            GMatrixFunctions<FL>::iscale(
                GMatrix<FL>(data, (MKL_INT)total_memory, 1), d);
        else
            for (int i = 0; i < n; i++)
                (*this)[i]->iscale(d);
    }
    void normalize() const { iscale(1 / norm()); }
    static void
    normalize_all(const vector<shared_ptr<SparseMatrixGroup>> &mats) {
        FP normsq = 0;
        for (auto &mat : mats)
            normsq += pow(mat->norm(), 2);
        for (auto &mat : mats)
            mat->iscale(1 / sqrt(normsq));
    }
    shared_ptr<SparseMatrix<S, FL>> operator[](int idx) const {
        assert(idx >= 0 && idx < n);
        shared_ptr<SparseMatrix<S, FL>> r = make_shared<SparseMatrix<S, FL>>();
        r->data = data + offsets[idx];
        r->info = infos[idx];
        r->total_memory = infos[idx]->get_total_memory();
        return r;
    }
    // l will have the same number of non-zero blocks as this matrix group
    // s will be labelled by right q labels
    void left_svd(vector<S> &rqs, vector<vector<shared_ptr<GTensor<FL>>>> &l,
                  vector<shared_ptr<GTensor<FP>>> &s,
                  vector<shared_ptr<GTensor<FL>>> &r,
                  const vector<shared_ptr<SparseMatrix<S, FL>>> &xmats =
                      vector<shared_ptr<SparseMatrix<S, FL>>>(),
                  const vector<FP> &scales = vector<FP>()) {
        map<S, size_t> qs_mp;
        vector<shared_ptr<SparseMatrixInfo<S>>> xinfos = infos;
        vector<size_t> xoffsets = offsets;
        vector<FP> xscales(infos.size(), 1.0);
        for (auto &xmat : xmats) {
            xinfos.push_back(xmat->info);
            xoffsets.push_back(xmat->data - data);
            xscales.push_back(1.0);
        }
        if (scales.size() != 0) {
            xscales.resize(infos.size());
            assert(scales.size() == xmats.size());
            for (auto &sc : scales)
                xscales.push_back(sc);
        }
        for (const auto &info : xinfos)
            for (int i = 0; i < info->n; i++) {
                S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                            : info->quanta[i].get_ket();
                qs_mp[q] +=
                    (size_t)info->n_states_bra[i] * info->n_states_ket[i];
            }
        int nr = (int)qs_mp.size(), k = 0;
        rqs.resize(nr);
        vector<size_t> tmp(nr + 1, 0);
        for (auto &mp : qs_mp)
            rqs[k] = mp.first, tmp[k + 1] = mp.second, k++;
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        FL *dt = (FL *)dalloc_<FP>()->allocate(tmp[nr] * cpx_sz);
        memset(dt, 0, sizeof(FL) * tmp[nr]);
        vector<size_t> it(nr, 0), sz(nr, 0);
        for (int ii = 0; ii < (int)xinfos.size(); ii++) {
            const auto &info = xinfos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                            : info->quanta[i].get_ket();
                size_t ir =
                    lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
                MKL_INT n_states =
                    (MKL_INT)info->n_states_bra[i] * info->n_states_ket[i];
                if (abs(xscales[ii]) > 1E-12)
                    GMatrixFunctions<FL>::iadd(
                        GMatrix<FL>(dt + (tmp[ir] + it[ir]), n_states, 1),
                        GMatrix<FL>(data + xoffsets[ii] +
                                        info->n_states_total[i],
                                    n_states, 1),
                        xscales[ii]);
                sz[ir] = info->n_states_ket[i];
                it[ir] += n_states;
            }
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        vector<shared_ptr<GTensor<FL>>> merged_l(nr);
        r.resize(nr);
        s.resize(nr);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int ir = 0; ir < nr; ir++) {
            MKL_INT nxr = (MKL_INT)sz[ir],
                    nxl = (MKL_INT)((tmp[ir + 1] - tmp[ir]) / nxr);
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0);
            MKL_INT nxk = min(nxl, nxr);
            shared_ptr<GTensor<FL>> tsl =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxl, nxk});
            shared_ptr<GTensor<FP>> tss =
                make_shared<GTensor<FP>>(vector<MKL_INT>{nxk});
            shared_ptr<GTensor<FL>> tsr =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxk, nxr});
            GMatrixFunctions<FL>::svd(GMatrix<FL>(dt + tmp[ir], nxl, nxr),
                                      tsl->ref(), tss->ref().flip_dims(),
                                      tsr->ref());
            merged_l[ir] = tsl;
            s[ir] = tss;
            r[ir] = tsr;
        }
        threading->activate_normal();
        memset(it.data(), 0, sizeof(size_t) * nr);
        l.resize(xinfos.size());
        for (int ii = 0; ii < (int)xinfos.size(); ii++) {
            const auto &info = xinfos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                            : info->quanta[i].get_ket();
                size_t ir =
                    lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
                shared_ptr<GTensor<FL>> tsl = make_shared<GTensor<FL>>(
                    vector<MKL_INT>{(MKL_INT)info->n_states_bra[i],
                                    merged_l[ir]->shape[1]});
                if (abs(xscales[ii]) > 1E-12)
                    GMatrixFunctions<FL>::iadd(
                        GMatrix<FL>(tsl->data->data(), (MKL_INT)tsl->size(), 1),
                        GMatrix<FL>(merged_l[ir]->data->data() + it[ir],
                                    (MKL_INT)tsl->size(), 1),
                        1.0 / xscales[ii]);
                it[ir] += tsl->size();
                l[ii].push_back(tsl);
            }
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == merged_l[ir]->size());
        dalloc_<FP>()->deallocate(dt, tmp[nr] * cpx_sz);
    }
    // r will have the same number of non-zero blocks as this matrix group
    // s will be labelled by left q labels
    void right_svd(vector<S> &lqs, vector<shared_ptr<GTensor<FL>>> &l,
                   vector<shared_ptr<GTensor<FP>>> &s,
                   vector<vector<shared_ptr<GTensor<FL>>>> &r,
                   const vector<shared_ptr<SparseMatrix<S, FL>>> &xmats =
                       vector<shared_ptr<SparseMatrix<S, FL>>>(),
                   const vector<FP> &scales = vector<FP>()) {
        map<S, size_t> qs_mp;
        vector<shared_ptr<SparseMatrixInfo<S>>> xinfos = infos;
        vector<size_t> xoffsets = offsets;
        vector<FP> xscales(infos.size(), 1.0);
        for (auto &xmat : xmats) {
            xinfos.push_back(xmat->info);
            xoffsets.push_back(xmat->data - data);
            xscales.push_back(1.0);
        }
        if (scales.size() != 0) {
            xscales.resize(infos.size());
            assert(scales.size() == xmats.size());
            for (auto &sc : scales)
                xscales.push_back(sc);
        }
        for (const auto &info : xinfos)
            for (int i = 0; i < info->n; i++) {
                S q = info->quanta[i].get_bra(info->delta_quantum);
                qs_mp[q] +=
                    (size_t)info->n_states_bra[i] * info->n_states_ket[i];
            }
        int nl = (int)qs_mp.size(), p = 0;
        lqs.resize(nl);
        vector<size_t> tmp(nl + 1, 0);
        for (auto &mp : qs_mp)
            lqs[p] = mp.first, tmp[p + 1] = mp.second, p++;
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        FL *dt = (FL *)dalloc_<FP>()->allocate(tmp[nl] * cpx_sz);
        memset(dt, 0, sizeof(FL) * tmp[nl]);
        vector<size_t> it(nl, 0), sz(nl, 0);
        for (int ii = 0; ii < (int)xinfos.size(); ii++) {
            const auto &info = xinfos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->quanta[i].get_bra(info->delta_quantum);
                size_t il =
                    lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
                MKL_INT nxl = info->n_states_bra[i],
                        nxr = (MKL_INT)((tmp[il + 1] - tmp[il]) / nxl);
                assert((tmp[il + 1] - tmp[il]) % nxl == 0);
                MKL_INT inr = info->n_states_ket[i];
                if (abs(xscales[ii]) > 1E-12) {
                    for (MKL_INT k = 0; k < nxl; k++)
                        GMatrixFunctions<FL>::iadd(
                            GMatrix<FL>(dt + (tmp[il] + it[il] + k * nxr), inr,
                                        1),
                            GMatrix<FL>(data + xoffsets[ii] +
                                            (info->n_states_total[i] + k * inr),
                                        inr, 1),
                            xscales[ii]);
                }
                sz[il] = nxl;
                it[il] += inr;
            }
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == (tmp[il + 1] - tmp[il]) / sz[il]);
        vector<shared_ptr<GTensor<FL>>> merged_r(nl);
        l.resize(nl);
        s.resize(nl);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int il = 0; il < nl; il++) {
            MKL_INT nxl = (MKL_INT)sz[il],
                    nxr = (MKL_INT)((tmp[il + 1] - tmp[il]) / nxl);
            assert((tmp[il + 1] - tmp[il]) % nxl == 0);
            MKL_INT nxk = min(nxl, nxr);
            shared_ptr<GTensor<FL>> tsl =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxl, nxk});
            shared_ptr<GTensor<FP>> tss =
                make_shared<GTensor<FP>>(vector<MKL_INT>{nxk});
            shared_ptr<GTensor<FL>> tsr =
                make_shared<GTensor<FL>>(vector<MKL_INT>{nxk, nxr});
            GMatrixFunctions<FL>::svd(GMatrix<FL>(dt + tmp[il], nxl, nxr),
                                      tsl->ref(), tss->ref().flip_dims(),
                                      tsr->ref());
            l[il] = tsl;
            s[il] = tss;
            merged_r[il] = tsr;
        }
        threading->activate_normal();
        memset(it.data(), 0, sizeof(size_t) * nl);
        r.resize(xinfos.size());
        for (int ii = 0; ii < (int)xinfos.size(); ii++) {
            const auto &info = xinfos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->quanta[i].get_bra(info->delta_quantum);
                size_t il =
                    lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
                shared_ptr<GTensor<FL>> tsr = make_shared<GTensor<FL>>(
                    vector<MKL_INT>{merged_r[il]->shape[0],
                                    (MKL_INT)info->n_states_ket[i]});
                MKL_INT inr = info->n_states_ket[i],
                        ixr = merged_r[il]->shape[1];
                MKL_INT inl = merged_r[il]->shape[0];
                if (abs(xscales[ii]) > 1E-12) {
                    for (MKL_INT k = 0; k < inl; k++)
                        GMatrixFunctions<FL>::iadd(
                            GMatrix<FL>(tsr->data->data() + k * inr,
                                        (MKL_INT)inr, 1),
                            GMatrix<FL>(merged_r[il]->data->data() +
                                            (it[il] + k * ixr),
                                        (MKL_INT)inr, 1),
                            1.0 / xscales[ii]);
                }
                it[il] += inr;
                r[ii].push_back(tsr);
            }
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == merged_r[il]->shape[1]);
        dalloc_<FP>()->deallocate(dt, tmp[nl] * cpx_sz);
    }
    friend ostream &operator<<(ostream &os, const SparseMatrixGroup &c) {
        os << "DATA = [ ";
        for (int i = 0; i < c.total_memory; i++)
            os << setw(20) << setprecision(14) << c.data[i] << " ";
        os << "]" << endl;
        return os;
    }
};

// Translation between SparseMatrix with different precision
template <typename S, typename FL1, typename FL2, typename = void>
struct TransSparseMatrix;

template <typename S, typename FL1, typename FL2>
struct TransSparseMatrix<
    S, FL1, FL2,
    typename enable_if<
        (is_floating_point<FL1>::value && is_floating_point<FL2>::value) ||
        (is_complex<FL1>::value && is_complex<FL2>::value) ||
        (is_floating_point<FL1>::value && is_complex<FL2>::value)>::type> {
    static shared_ptr<SparseMatrix<S, FL2>>
    forward(const shared_ptr<SparseMatrix<S, FL1>> &mat) {
        shared_ptr<VectorAllocator<typename GMatrix<FL2>::FP>> d_alloc =
            make_shared<VectorAllocator<typename GMatrix<FL2>::FP>>();
        shared_ptr<SparseMatrix<S, FL2>> xmat =
            make_shared<SparseMatrix<S, FL2>>(d_alloc);
        xmat->allocate(mat->info);
        xmat->factor = (FL2)mat->factor;
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (size_t i = 0; i < xmat->total_memory; i++)
            xmat->data[i] = (FL2)mat->data[i];
        threading->activate_normal();
        return xmat;
    }
};

template <typename S, typename FL1, typename FL2>
struct TransSparseMatrix<
    S, FL1, FL2,
    typename enable_if<is_complex<FL1>::value &&
                       is_floating_point<FL2>::value>::type> {
    static shared_ptr<SparseMatrix<S, FL2>>
    forward(const shared_ptr<SparseMatrix<S, FL1>> &mat) {
        shared_ptr<VectorAllocator<typename GMatrix<FL2>::FP>> d_alloc =
            make_shared<VectorAllocator<typename GMatrix<FL2>::FP>>();
        shared_ptr<SparseMatrix<S, FL2>> xmat =
            make_shared<SparseMatrix<S, FL2>>(d_alloc);
        xmat->allocate(mat->info);
        xmat->factor = (FL2)xreal<FL1>(mat->factor);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (size_t i = 0; i < xmat->total_memory; i++)
            xmat->data[i] = (FL2)xreal<FL1>(mat->data[i]);
        threading->activate_normal();
        return xmat;
    }
};

} // namespace block2
