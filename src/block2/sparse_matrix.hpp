
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

#include "cg.hpp"
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
#include <vector>

#define TINY (1E-20)

using namespace std;

namespace block2 {

template <typename, typename = void> struct SparseMatrixInfo;

// Symmetry label information for block-sparse matrix
template <typename S>
struct SparseMatrixInfo<
    S, typename enable_if<integral_constant<
           bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
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
        uint32_t *stride;
        double *factor;
        uint16_t *ia, *ib, *ic;
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
            vector<uint16_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = vic.size();
                S adq = cja ? -subdq[k].second.get_bra(opdq)
                            : subdq[k].second.get_bra(opdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                if ((adq + bdq)[0].data != 0)
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
                            cg->wigner_9j(aq.twos(), bq.twos(), cdq.twos(),
                                          adq.twos(), bdq.twos(), opdq.twos(),
                                          aq.twos(), bq.twos(), cdq.twos());
                        if (cja)
                            factor *= cg->transpose_cg(adq.twos(), aq.twos(),
                                                       aq.twos());
                        if (cjb)
                            factor *= cg->transpose_cg(bdq.twos(), bq.twos(),
                                                       bq.twos());
                        factor *= (binfo->is_fermion && (aq.n() & 1)) ? -1 : 1;
                        if (abs(factor) >= TINY) {
                            via.push_back(ia);
                            vib.push_back(ib);
                            vic.push_back(ic);
                            vf.push_back(factor);
                        }
                    }
                }
            }
            n[4] = vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = vic.size();
            uint32_t *ptr = ialloc->allocate((n[4] << 1));
            uint32_t *cptr = ialloc->allocate((nc << 2) + nc - (nc >> 1));
            quanta = (S *)ptr;
            idx = ptr + n[4];
            stride = cptr;
            factor = (double *)(cptr + nc);
            ia = (uint16_t *)(cptr + nc + nc + nc), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, &vidx[0], n[4] * sizeof(uint32_t));
            memset(stride, 0, nc * sizeof(uint32_t));
            memcpy(factor, &vf[0], nc * sizeof(double));
            memcpy(ia, &via[0], nc * sizeof(uint16_t));
            memcpy(ib, &vib[0], nc * sizeof(uint16_t));
            memcpy(ic, &vic[0], nc * sizeof(uint16_t));
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
            vector<uint32_t> vidx(subdq.size()), viv;
            vector<uint16_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = viv.size();
                vector<vector<
                    tuple<double, uint32_t, uint16_t, uint16_t, uint16_t>>>
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
                                        cg->wigner_9j(
                                            lqprime.twos(), rqprime.twos(),
                                            cdq.twos(), adq.twos(), bdq.twos(),
                                            opdq.twos(), lq.twos(), rq.twos(),
                                            vdq.twos());
                                    factor *=
                                        (binfo->is_fermion && (lqprime.n() & 1))
                                            ? -1
                                            : 1;
                                    if (cja)
                                        factor *= cg->transpose_cg(
                                            adq.twos(), lq.twos(),
                                            lqprime.twos());
                                    if (cjb)
                                        factor *= cg->transpose_cg(
                                            bdq.twos(), rq.twos(),
                                            rqprime.twos());
                                    if (abs(factor) >= TINY) {
                                        if (pv.size() <= ip)
                                            pv.push_back(
                                                vector<tuple<double, uint32_t,
                                                             uint16_t, uint16_t,
                                                             uint16_t>>());
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
            n[4] = vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = viv.size();
            uint32_t *ptr = ialloc->allocate((n[4] << 1));
            uint32_t *cptr = ialloc->allocate((nc << 2) + nc - (nc >> 1));
            quanta = (S *)ptr;
            idx = ptr + n[4];
            stride = cptr;
            factor = (double *)(cptr + nc);
            ia = (uint16_t *)(cptr + nc + nc + nc), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, &vidx[0], n[4] * sizeof(uint32_t));
            memcpy(stride, &viv[0], nc * sizeof(uint32_t));
            memcpy(factor, &vf[0], nc * sizeof(double));
            memcpy(ia, &via[0], nc * sizeof(uint16_t));
            memcpy(ib, &vib[0], nc * sizeof(uint16_t));
            memcpy(ic, &vic[0], nc * sizeof(uint16_t));
        }
        // Compute non-zero-block indices for 'tensor_product'
        void initialize_tp(
            S cdq, const vector<pair<uint8_t, S>> &subdq,
            const StateInfo<S> &bra, const StateInfo<S> &ket,
            const StateInfo<S> &bra_a, const StateInfo<S> &bra_b,
            const StateInfo<S> &ket_a, const StateInfo<S> &ket_b,
            const StateInfo<S> &bra_cinfo, const StateInfo<S> &ket_cinfo,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &binfos,
            const shared_ptr<SparseMatrixInfo<S>> &cinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size()), vstride;
            vector<uint16_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = vstride.size();
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
                    int kbed = ib == bra.n - 1 ? bra_cinfo.n
                                               : bra_cinfo.n_states[ib + 1];
                    int kked = ik == ket.n - 1 ? ket_cinfo.n
                                               : ket_cinfo.n_states[ik + 1];
                    uint32_t bra_stride = 0, ket_stride = 0;
                    for (int kb = bra_cinfo.n_states[ib]; kb < kbed; kb++) {
                        uint16_t jba = bra_cinfo.quanta[kb].data >> 16,
                                 jbb = bra_cinfo.quanta[kb].data & (0xFFFFU);
                        ket_stride = 0;
                        for (int kk = ket_cinfo.n_states[ik]; kk < kked; kk++) {
                            uint16_t jka = ket_cinfo.quanta[kk].data >> 16,
                                     jkb =
                                         ket_cinfo.quanta[kk].data & (0xFFFFU);
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
                                        cg->wigner_9j(
                                            aqprime.twos(), bqprime.twos(),
                                            cqprime.twos(), adq.twos(),
                                            bdq.twos(), cdq.twos(), aq.twos(),
                                            bq.twos(), cq.twos());
                                    factor *=
                                        (binfo->is_fermion && (aqprime.n() & 1))
                                            ? -1
                                            : 1;
                                    if (cja)
                                        factor *= cg->transpose_cg(
                                            adq.twos(), aq.twos(),
                                            aqprime.twos());
                                    if (cjb)
                                        factor *= cg->transpose_cg(
                                            bdq.twos(), bq.twos(),
                                            bqprime.twos());
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
                            ket_stride += (uint32_t)ket_a.n_states[jka] *
                                          ket_b.n_states[jkb];
                        }
                        bra_stride +=
                            (uint32_t)bra_a.n_states[jba] * bra_b.n_states[jbb];
                    }
                }
            }
            n[4] = vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = vstride.size();
            uint32_t *ptr = ialloc->allocate((n[4] << 1));
            uint32_t *cptr = ialloc->allocate((nc << 2) + nc - (nc >> 1));
            quanta = (S *)ptr;
            idx = ptr + n[4];
            stride = cptr;
            factor = (double *)(cptr + nc);
            ia = (uint16_t *)(cptr + nc + nc + nc), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, &vidx[0], n[4] * sizeof(uint32_t));
            memcpy(stride, &vstride[0], nc * sizeof(uint32_t));
            memcpy(factor, &vf[0], nc * sizeof(double));
            memcpy(ia, &via[0], nc * sizeof(uint16_t));
            memcpy(ib, &vib[0], nc * sizeof(uint16_t));
            memcpy(ic, &vic[0], nc * sizeof(uint16_t));
        }
        void reallocate(bool clean) {
            size_t length = (n[4] << 1) + (nc << 2) + nc - (nc >> 1);
            uint32_t *ptr = ialloc->reallocate((uint32_t *)quanta, length,
                                               clean ? 0 : length);
            if (ptr != (uint32_t *)quanta) {
                memmove(ptr, quanta, length * sizeof(uint32_t));
                quanta = (S *)ptr;
                idx = ptr + n[4];
                stride = ptr + (n[4] << 1);
                factor = (double *)(ptr + (n[4] << 1) + nc);
                ia = (uint16_t *)(stride + nc + nc + nc), ib = ia + nc,
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
                                   (n[4] << 1) + (nc << 2) + nc - (nc >> 1));
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
                os << "(BRA) " << ci.quanta[i].get_bra(S(0)) << " KET "
                   << -ci.quanta[i].get_ket() << " [ " << (int)ci.idx[i] << "~"
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
               ((n << 1) + _DBL_MEM_SIZE(n)) * sizeof(uint32_t));
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
    void load_data(ifstream &ifs) {
        ifs.read((char *)&delta_quantum, sizeof(delta_quantum));
        ifs.read((char *)&n, sizeof(n));
        if (alloc == nullptr)
            alloc = ialloc;
        uint32_t *ptr = alloc->allocate((n << 1) + _DBL_MEM_SIZE(n));
        ifs.read((char *)ptr, sizeof(uint32_t) * ((n << 1) + _DBL_MEM_SIZE(n)));
        ifs.read((char *)&is_fermion, sizeof(is_fermion));
        ifs.read((char *)&is_wavefunction, sizeof(is_wavefunction));
        quanta = (S *)ptr;
        n_states_bra = (ubond_t *)(ptr + n);
        n_states_ket = (ubond_t *)(ptr + n) + n;
        n_states_total = ptr + n + _DBL_MEM_SIZE(n);
        cinfo = nullptr;
    }
    void save_data(const string &filename) const {
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
    void save_data(ofstream &ofs) const {
        ofs.write((char *)&delta_quantum, sizeof(delta_quantum));
        assert(n != -1);
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)quanta,
                  sizeof(uint32_t) * ((n << 1) + _DBL_MEM_SIZE(n)));
        ofs.write((char *)&is_fermion, sizeof(is_fermion));
        ofs.write((char *)&is_wavefunction, sizeof(is_wavefunction));
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
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(S));
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
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(S));
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
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(S));
            sort(quanta, quanta + n);
            for (int i = 0; i < n; i++) {
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
        assert(delta_quantum.data == 0);
        info->allocate(n);
        memcpy(info->quanta, quanta, n * sizeof(S));
        memcpy(info->n_states, right ? n_states_ket : n_states_bra,
               n * sizeof(ubond_t));
        info->n_states_total =
            accumulate(info->n_states, info->n_states + n, 0);
        return info;
    }
    int find_state(S q, int start = 0) const {
        auto p = lower_bound(quanta + start, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return p - quanta;
    }
    void sort_states() {
        int idx[n];
        S q[n];
        ubond_t nqb[n], nqk[n];
        memcpy(q, quanta, n * sizeof(S));
        memcpy(nqb, n_states_bra, n * sizeof(ubond_t));
        memcpy(nqk, n_states_ket, n * sizeof(ubond_t));
        for (int i = 0; i < n; i++)
            idx[i] = i;
        sort(idx, idx + n, [&q](int i, int j) { return q[i] < q[j]; });
        for (int i = 0; i < n; i++)
            quanta[i] = q[idx[i]], n_states_bra[i] = nqb[idx[i]],
            n_states_ket[i] = nqk[idx[i]];
        n_states_total[0] = 0;
        for (int i = 0; i < n - 1; i++)
            n_states_total[i + 1] =
                n_states_total[i] + (uint32_t)n_states_bra[i] * n_states_ket[i];
    }
    uint32_t get_total_memory() const {
        return n == 0 ? 0
                      : n_states_total[n - 1] +
                            (uint32_t)n_states_bra[n - 1] * n_states_ket[n - 1];
    }
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0) {
            if (alloc == nullptr)
                alloc = ialloc;
            ptr = alloc->allocate((length << 1) + _DBL_MEM_SIZE(length));
        }
        quanta = (S *)ptr;
        n_states_bra = (ubond_t *)(ptr + length);
        n_states_ket = (ubond_t *)(ptr + length) + length;
        n_states_total = ptr + length + _DBL_MEM_SIZE(length);
        n = length;
    }
    void deallocate() {
        assert(n != -1);
        alloc->deallocate((uint32_t *)quanta, (n << 1) + _DBL_MEM_SIZE(n));
        alloc = nullptr;
        quanta = nullptr;
        n_states_bra = nullptr;
        n_states_ket = nullptr;
        n_states_total = nullptr;
        n = -1;
    }
    void reallocate(int length) {
        uint32_t *ptr =
            alloc->reallocate((uint32_t *)quanta, (n << 1) + _DBL_MEM_SIZE(n),
                              (length << 1) + _DBL_MEM_SIZE(length));
        if (ptr == (uint32_t *)quanta)
            memmove(ptr + length, n_states_bra,
                    (length + _DBL_MEM_SIZE(length)) * sizeof(uint32_t));
        else {
            memmove(ptr, quanta,
                    ((length << 1) + _DBL_MEM_SIZE(length)) * sizeof(uint32_t));
            quanta = (S *)ptr;
        }
        n_states_bra = (ubond_t *)(ptr + length);
        n_states_ket = (ubond_t *)(ptr + length) + length;
        n_states_total = ptr + length + _DBL_MEM_SIZE(length);
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

enum struct SparseMatrixTypes : uint8_t { Normal = 0, CSR = 1 };

// Block-sparse Matrix
// Representing operator, wavefunction, density matrix and MPS tensors
template <typename S> struct SparseMatrix {
    shared_ptr<Allocator<double>> alloc;
    shared_ptr<SparseMatrixInfo<S>> info;
    double *data;
    double factor;
    size_t total_memory;
    SparseMatrix(const shared_ptr<Allocator<double>> &alloc = nullptr)
        : info(nullptr), data(nullptr), factor(1.0), total_memory(0),
          alloc(alloc) {}
    virtual ~SparseMatrix() = default;
    virtual const SparseMatrixTypes get_type() const {
        return SparseMatrixTypes::Normal;
    }
    virtual void load_data(ifstream &ifs) {
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&total_memory, sizeof(total_memory));
        data = alloc->allocate(total_memory);
        ifs.read((char *)data, sizeof(double) * total_memory);
    }
    void load_data(const string &filename, bool load_info = false,
                   const shared_ptr<Allocator<uint32_t>> &i_alloc = nullptr) {
        if (alloc == nullptr)
            alloc = dalloc;
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
    virtual void save_data(ofstream &ofs) const {
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&total_memory, sizeof(total_memory));
        ofs.write((char *)data, sizeof(double) * total_memory);
    }
    void save_data(const string &filename, bool save_info = false) const {
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
    virtual void copy_data_from(const shared_ptr<SparseMatrix> &other) {
        assert(total_memory == other->total_memory);
        memcpy(data, other->data, sizeof(double) * total_memory);
    }
    virtual void selective_copy_from(const shared_ptr<SparseMatrix> &other) {
        for (int i = 0, k; i < other->info->n; i++)
            if ((k = info->find_state(other->info->quanta[i])) != -1)
                memcpy(data + info->n_states_total[k],
                       other->data + other->info->n_states_total[i],
                       sizeof(double) * ((size_t)info->n_states_bra[k] *
                                         info->n_states_ket[k]));
    }
    virtual void clear() { memset(data, 0, sizeof(double) * total_memory); }
    virtual void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                          double *ptr = 0) {
        this->info = info;
        total_memory = info->get_total_memory();
        if (total_memory == 0)
            return;
        if (ptr == 0) {
            if (alloc == nullptr)
                alloc = dalloc;
            data = alloc->allocate(total_memory);
            memset(data, 0, sizeof(double) * total_memory);
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
        alloc->deallocate(data, total_memory);
        alloc = nullptr;
        total_memory = 0;
        data = nullptr;
    }
    void reallocate(int length) {
        assert(alloc != nullptr);
        double *ptr = alloc->reallocate(data, total_memory, length);
        if (ptr != data && length != 0)
            memmove(ptr, data, length * sizeof(double));
        total_memory = length;
        data = length == 0 ? nullptr : ptr;
    }
    MatrixRef operator[](S q) const { return (*this)[info->find_state(q)]; }
    MatrixRef operator[](int idx) const {
        assert(idx != -1);
        return MatrixRef(data + info->n_states_total[idx],
                         (int)info->n_states_bra[idx],
                         (int)info->n_states_ket[idx]);
    }
    double trace() const {
        double r = 0;
        for (int i = 0; i < info->n; i++)
            r += this->operator[](i).trace();
        return r;
    }
    virtual double norm() const {
        return MatrixFunctions::norm(MatrixRef(data, total_memory, 1));
    }
    // ratio of zero elements to total size
    virtual double sparsity() const {
        size_t nnz = 0;
        for (size_t i = 0; i < total_memory; i++)
            nnz += abs(this->data[i]) > TINY;
        return 1.0 - (double)nnz / total_memory;
    }
    void iscale(double d) const {
        assert(factor == 1.0);
        MatrixFunctions::iscale(MatrixRef(data, total_memory, 1), d);
    }
    void normalize() const { iscale(1 / norm()); }
    // l will have the same number of non-zero blocks as this matrix
    // s will be labelled by right q labels
    void left_svd(vector<S> &rqs, vector<shared_ptr<Tensor>> &l,
                  vector<shared_ptr<Tensor>> &s,
                  vector<shared_ptr<Tensor>> &r) {
        map<S, uint32_t> qs_mp;
        for (int i = 0; i < info->n; i++) {
            S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                        : info->quanta[i].get_ket();
            qs_mp[q] += (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
        }
        int nr = (int)qs_mp.size(), k = 0;
        rqs.resize(nr);
        uint32_t tmp[nr + 1];
        memset(tmp, 0, sizeof(uint32_t) * (nr + 1));
        for (auto &mp : qs_mp)
            rqs[k] = mp.first, tmp[k + 1] = mp.second, k++;
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        double *dt = dalloc->allocate(tmp[nr]);
        uint32_t it[nr], sz[nr];
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int i = 0; i < info->n; i++) {
            S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                        : info->quanta[i].get_ket();
            int ir = lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
            uint32_t n_states =
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(dt + (tmp[ir] + it[ir]), data + info->n_states_total[i],
                   n_states * sizeof(double));
            sz[ir] = info->n_states_ket[i];
            it[ir] += n_states;
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        vector<shared_ptr<Tensor>> merged_l;
        r.clear();
        s.clear();
        for (int ir = 0; ir < nr; ir++) {
            int nxr = sz[ir], nxl = (tmp[ir + 1] - tmp[ir]) / nxr;
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0);
            int nxk = min(nxl, nxr);
            shared_ptr<Tensor> tsl = make_shared<Tensor>(vector<int>{nxl, nxk});
            shared_ptr<Tensor> tss = make_shared<Tensor>(vector<int>{nxk});
            shared_ptr<Tensor> tsr = make_shared<Tensor>(vector<int>{nxk, nxr});
            MatrixFunctions::svd(MatrixRef(dt + tmp[ir], nxl, nxr), tsl->ref(),
                                 tss->ref().flip_dims(), tsr->ref());
            merged_l.push_back(tsl);
            s.push_back(tss);
            r.push_back(tsr);
        }
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int i = 0; i < info->n; i++) {
            S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                        : info->quanta[i].get_ket();
            int ir = lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
            shared_ptr<Tensor> tsl = make_shared<Tensor>(vector<int>{
                (int)info->n_states_bra[i], merged_l[ir]->shape[1]});
            memcpy(tsl->data.data(), merged_l[ir]->data.data() + it[ir],
                   tsl->size() * sizeof(double));
            it[ir] += tsl->size();
            l.push_back(tsl);
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == merged_l[ir]->size());
        dalloc->deallocate(dt, tmp[nr]);
    }
    // r will have the same number of non-zero blocks as this matrix
    // s will be labelled by left q labels
    void right_svd(vector<S> &lqs, vector<shared_ptr<Tensor>> &l,
                   vector<shared_ptr<Tensor>> &s,
                   vector<shared_ptr<Tensor>> &r) {
        map<S, uint32_t> qs_mp;
        for (int i = 0; i < info->n; i++) {
            S q = info->quanta[i].get_bra(info->delta_quantum);
            qs_mp[q] += (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
        }
        int nl = (int)qs_mp.size(), p = 0;
        lqs.resize(nl);
        uint32_t tmp[nl + 1];
        memset(tmp, 0, sizeof(uint32_t) * (nl + 1));
        for (auto &mp : qs_mp)
            lqs[p] = mp.first, tmp[p + 1] = mp.second, p++;
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        double *dt = dalloc->allocate(tmp[nl]);
        uint32_t it[nl], sz[nl];
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int i = 0; i < info->n; i++) {
            S q = info->quanta[i].get_bra(info->delta_quantum);
            int il = lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
            uint32_t nxl = info->n_states_bra[i],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0);
            uint32_t inr = info->n_states_ket[i];
            for (uint32_t k = 0; k < nxl; k++)
                memcpy(dt + (tmp[il] + it[il] + k * nxr),
                       data + (info->n_states_total[i] + k * inr),
                       inr * sizeof(double));
            sz[il] = nxl;
            it[il] += inr;
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == (tmp[il + 1] - tmp[il]) / sz[il]);
        vector<shared_ptr<Tensor>> merged_r;
        l.clear();
        s.clear();
        for (int il = 0; il < nl; il++) {
            int nxl = sz[il], nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0);
            int nxk = min(nxl, nxr);
            shared_ptr<Tensor> tsl = make_shared<Tensor>(vector<int>{nxl, nxk});
            shared_ptr<Tensor> tss = make_shared<Tensor>(vector<int>{nxk});
            shared_ptr<Tensor> tsr = make_shared<Tensor>(vector<int>{nxk, nxr});
            MatrixFunctions::svd(MatrixRef(dt + tmp[il], nxl, nxr), tsl->ref(),
                                 tss->ref().flip_dims(), tsr->ref());
            l.push_back(tsl);
            s.push_back(tss);
            merged_r.push_back(tsr);
        }
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int i = 0; i < info->n; i++) {
            S q = info->quanta[i].get_bra(info->delta_quantum);
            int il = lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
            shared_ptr<Tensor> tsr = make_shared<Tensor>(vector<int>{
                merged_r[il]->shape[0], (int)info->n_states_ket[i]});
            int inr = info->n_states_ket[i], ixr = merged_r[il]->shape[1];
            int inl = merged_r[il]->shape[0];
            for (uint32_t k = 0; k < inl; k++)
                memcpy(tsr->data.data() + k * inr,
                       merged_r[il]->data.data() + (it[il] + k * ixr),
                       inr * sizeof(double));
            it[il] += inr;
            r.push_back(tsr);
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == merged_r[il]->shape[1]);
        dalloc->deallocate(dt, tmp[nl]);
    }
    void left_canonicalize(const shared_ptr<SparseMatrix<S>> &rmat) {
        int nr = rmat->info->n, n = info->n;
        uint32_t tmp[nr + 1];
        memset(tmp, 0, sizeof(uint32_t) * (nr + 1));
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            assert(ir != -1);
            tmp[ir + 1] +=
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
        }
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        double *dt = dalloc->allocate(tmp[nr]);
        uint32_t it[nr];
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            uint32_t n_states =
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(dt + (tmp[ir] + it[ir]), data + info->n_states_total[i],
                   n_states * sizeof(double));
            it[ir] += n_states;
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        for (int ir = 0; ir < nr; ir++) {
            uint32_t nxr = rmat->info->n_states_ket[ir],
                     nxl = (tmp[ir + 1] - tmp[ir]) / nxr;
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0 && nxl >= nxr);
            MatrixFunctions::qr(MatrixRef(dt + tmp[ir], nxl, nxr),
                                MatrixRef(dt + tmp[ir], nxl, nxr), (*rmat)[ir]);
        }
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            uint32_t n_states =
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(data + info->n_states_total[i], dt + (tmp[ir] + it[ir]),
                   n_states * sizeof(double));
            it[ir] += n_states;
        }
        dalloc->deallocate(dt, tmp[nr]);
    }
    void right_canonicalize(const shared_ptr<SparseMatrix<S>> &lmat) {
        int nl = lmat->info->n, n = info->n;
        uint32_t tmp[nl + 1];
        memset(tmp, 0, sizeof(uint32_t) * (nl + 1));
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            assert(il != -1);
            tmp[il + 1] +=
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
        }
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        double *dt = dalloc->allocate(tmp[nl]);
        uint32_t it[nl];
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            uint32_t nxl = info->n_states_bra[i],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            uint32_t inr = info->n_states_ket[i];
            for (uint32_t k = 0; k < nxl; k++)
                memcpy(dt + (tmp[il] + it[il] + k * nxr),
                       data + info->n_states_total[i] + k * inr,
                       inr * sizeof(double));
            it[il] += inr * nxl;
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == tmp[il + 1] - tmp[il]);
        for (int il = 0; il < nl; il++) {
            uint32_t nxl = lmat->info->n_states_bra[il],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0 && nxr >= nxl);
            MatrixFunctions::lq(MatrixRef(dt + tmp[il], nxl, nxr), (*lmat)[il],
                                MatrixRef(dt + tmp[il], nxl, nxr));
        }
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            uint32_t nxl = info->n_states_bra[i],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            uint32_t inr = info->n_states_ket[i];
            for (uint32_t k = 0; k < nxl; k++)
                memcpy(data + info->n_states_total[i] + k * inr,
                       dt + (tmp[il] + it[il] + k * nxr), inr * sizeof(double));
            it[il] += inr * nxl;
        }
        dalloc->deallocate(dt, tmp[nl]);
    }
    void left_multiply(const shared_ptr<SparseMatrix<S>> &lmat,
                       const StateInfo<S> &l, const StateInfo<S> &m,
                       const StateInfo<S> &r, const StateInfo<S> &old_fused,
                       const StateInfo<S> &old_fused_cinfo) const {
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = ib == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ib + 1];
            uint32_t p = info->n_states_total[i];
            for (int bb = old_fused_cinfo.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = old_fused_cinfo.quanta[bb].data >> 16,
                         ibbb = old_fused_cinfo.quanta[bb].data & 0xFFFFU;
                int il = lmat->info->find_state(l.quanta[ibba]);
                uint32_t lp = (uint32_t)m.n_states[ibbb] * r.n_states[ik];
                if (il != -1) {
                    assert(lmat->info->n_states_bra[il] ==
                           lmat->info->n_states_ket[il]);
                    assert(lmat->info->n_states_bra[il] == l.n_states[ibba]);
                    MatrixRef tmp(nullptr, l.n_states[ibba], lp);
                    tmp.allocate();
                    MatrixFunctions::multiply(
                        (*lmat)[il], false,
                        MatrixRef(data + p, l.n_states[ibba], lp), false, tmp,
                        lmat->factor, 0.0);
                    memcpy(data + p, tmp.data, sizeof(double) * tmp.size());
                    tmp.deallocate();
                }
                p += l.n_states[ibba] * lp;
            }
            assert(p == (i != info->n - 1 ? info->n_states_total[i + 1]
                                          : total_memory));
        }
    }
    void right_multiply(const shared_ptr<SparseMatrix<S>> &rmat,
                        const StateInfo<S> &l, const StateInfo<S> &m,
                        const StateInfo<S> &r, const StateInfo<S> &old_fused,
                        const StateInfo<S> &old_fused_cinfo) const {
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int kked = ik == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ik + 1];
            uint32_t p = info->n_states_total[i];
            for (int kk = old_fused_cinfo.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = old_fused_cinfo.quanta[kk].data >> 16,
                         ikkb = old_fused_cinfo.quanta[kk].data & 0xFFFFU;
                int ir = rmat->info->find_state(r.quanta[ikkb]);
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                if (ir != -1) {
                    assert(rmat->info->n_states_bra[ir] ==
                           rmat->info->n_states_ket[ir]);
                    assert(rmat->info->n_states_bra[ir] == r.n_states[ikkb]);
                    MatrixRef tmp(nullptr, m.n_states[ikka], r.n_states[ikkb]);
                    tmp.allocate();
                    for (int j = 0; j < l.n_states[ib]; j++) {
                        MatrixFunctions::multiply(
                            MatrixRef(data + p + j * old_fused.n_states[ik],
                                      m.n_states[ikka], r.n_states[ikkb]),
                            false, (*rmat)[ir], false, tmp, rmat->factor, 0.0);
                        memcpy(data + p + j * old_fused.n_states[ik], tmp.data,
                               sizeof(double) * tmp.size());
                    }
                    tmp.deallocate();
                }
                p += lp;
            }
        }
    }
    void randomize(double a = 0.0, double b = 1.0) const {
        Random::fill_rand_double(data, total_memory, a, b);
    }
    // Contract two SparseMatrix
    void contract(const shared_ptr<SparseMatrix> &lmat,
                  const shared_ptr<SparseMatrix> &rmat) {
        assert(info->is_wavefunction);
        if (lmat->info->is_wavefunction)
            for (int i = 0; i < info->n; i++) {
                int il = lmat->info->find_state(info->quanta[i]);
                int ir = rmat->info->find_state(-info->quanta[i].get_ket());
                if (il != -1 && ir != -1)
                    MatrixFunctions::multiply((*lmat)[il], false, (*rmat)[ir],
                                              false, (*this)[i],
                                              lmat->factor * rmat->factor, 0.0);
            }
        else
            for (int i = 0; i < info->n; i++) {
                int il = lmat->info->find_state(
                    info->quanta[i].get_bra(info->delta_quantum));
                int ir = rmat->info->find_state(info->quanta[i]);
                if (il != -1 && ir != -1)
                    MatrixFunctions::multiply((*lmat)[il], false, (*rmat)[ir],
                                              false, (*this)[i],
                                              lmat->factor * rmat->factor, 0.0);
            }
    }
    // Change from [l x (fused m and r)] to [(fused l and m) x r]
    void swap_to_fused_left(const shared_ptr<SparseMatrix<S>> &mat,
                            const StateInfo<S> &l, const StateInfo<S> &m,
                            const StateInfo<S> &r,
                            const StateInfo<S> &old_fused,
                            const StateInfo<S> &old_fused_cinfo,
                            const StateInfo<S> &new_fused,
                            const StateInfo<S> &new_fused_cinfo,
                            const shared_ptr<CG<S>> &cg) {
        assert(mat->info->is_wavefunction);
        factor = mat->factor;
        // for SU2 with target 2S != 0, for each l m r there can be multiple mr
        // mp is the three-index wavefunction
        map<uint32_t, map<uint16_t, vector<pair<uint32_t, int>>>> mp;
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = -mat->info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int kked = ik == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ik + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int kk = old_fused_cinfo.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = old_fused_cinfo.quanta[kk].data >> 16,
                         ikkb = old_fused_cinfo.quanta[kk].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                mp[(ib << 16) + ikka][ikkb].push_back(make_pair(p, ik));
                p += lp;
            }
        }
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = -info->quanta[i].get_ket();
            int ib = new_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = ib == new_fused.n - 1 ? new_fused_cinfo.n
                                             : new_fused_cinfo.n_states[ib + 1];
            double *ptr = data + info->n_states_total[i];
            for (int bb = new_fused_cinfo.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = new_fused_cinfo.quanta[bb].data >> 16,
                         ibbb = new_fused_cinfo.quanta[bb].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ibbb] * r.n_states[ik];
                S bra_l = l.quanta[ibba], bra_m = m.quanta[ibbb];
                if (mp.count(new_fused_cinfo.quanta[bb].data) &&
                    mp[new_fused_cinfo.quanta[bb].data].count(ik))
                    for (pair<uint32_t, int> &t :
                         mp.at(new_fused_cinfo.quanta[bb].data).at(ik)) {
                        S ket_mr = old_fused.quanta[t.second];
                        double factor =
                            cg->racah(bra_l.twos(), bra_m.twos(),
                                      info->delta_quantum.twos(), ket.twos(),
                                      bra.twos(), ket_mr.twos()) *
                            sqrt(1.0 * bra.multiplicity() *
                                 ket_mr.multiplicity());
                        for (int j = 0; j < l.n_states[ibba]; j++)
                            MatrixFunctions::iadd(
                                MatrixRef(ptr + j * lp, lp, 1),
                                MatrixRef(mat->data + t.first +
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
    void swap_to_fused_right(const shared_ptr<SparseMatrix<S>> &mat,
                             const StateInfo<S> &l, const StateInfo<S> &m,
                             const StateInfo<S> &r,
                             const StateInfo<S> &old_fused,
                             const StateInfo<S> &old_fused_cinfo,
                             const StateInfo<S> &new_fused,
                             const StateInfo<S> &new_fused_cinfo,
                             const shared_ptr<CG<S>> &cg) {
        assert(mat->info->is_wavefunction);
        factor = mat->factor;
        map<uint32_t, map<uint16_t, vector<tuple<uint32_t, uint32_t, int>>>> mp;
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = -mat->info->quanta[i].get_ket();
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = ib == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ib + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int bb = old_fused_cinfo.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = old_fused_cinfo.quanta[bb].data >> 16,
                         ibbb = old_fused_cinfo.quanta[bb].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ibbb] * r.n_states[ik];
                mp[(ibbb << 16) + ik][ibba].push_back(make_tuple(p, lp, ib));
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
            int kked = ik == new_fused.n - 1 ? new_fused_cinfo.n
                                             : new_fused_cinfo.n_states[ik + 1];
            double *ptr = data + info->n_states_total[i];
            uint32_t lp = new_fused.n_states[ik];
            for (int kk = new_fused_cinfo.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = new_fused_cinfo.quanta[kk].data >> 16,
                         ikkb = new_fused_cinfo.quanta[kk].data & (0xFFFFU);
                S ket_m = m.quanta[ikka], ket_r = r.quanta[ikkb];
                if (mp.count(new_fused_cinfo.quanta[kk].data) &&
                    mp[new_fused_cinfo.quanta[kk].data].count(ib))
                    for (tuple<uint32_t, uint32_t, int> &t :
                         mp.at(new_fused_cinfo.quanta[kk].data).at(ib)) {
                        S bra_lm = old_fused.quanta[get<2>(t)];
                        double factor =
                            cg->racah(ket_r.twos(), ket_m.twos(),
                                      info->delta_quantum.twos(), bra.twos(),
                                      ket.twos(), bra_lm.twos()) *
                            sqrt(1.0 * ket.multiplicity() *
                                 bra_lm.multiplicity());
                        for (int j = 0; j < l.n_states[ib]; j++) {
                            MatrixFunctions::iadd(
                                MatrixRef(ptr + j * lp, (int)get<1>(t), 1),
                                MatrixRef(mat->data + get<0>(t) + j * get<1>(t),
                                          (int)get<1>(t), 1),
                                factor);
                        }
                    }
                ptr += (size_t)m.n_states[ikka] * r.n_states[ikkb];
            }
        }
    }
    friend ostream &operator<<(ostream &os, const SparseMatrix<S> &c) {
        os << "DATA = [ ";
        for (int i = 0; i < c.total_memory; i++)
            os << setw(20) << setprecision(14) << c.data[i] << " ";
        os << "]"
           << " FACTOR = ";
        os << setw(20) << setprecision(14) << c.factor << endl;
        return os;
    }
};

template <typename S> struct SparseMatrixGroup {
    shared_ptr<Allocator<double>> alloc;
    vector<shared_ptr<SparseMatrixInfo<S>>> infos;
    vector<size_t> offsets;
    double *data;
    size_t total_memory;
    int n;
    SparseMatrixGroup(const shared_ptr<Allocator<double>> &alloc = nullptr)
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
            alloc = dalloc;
        data = alloc->allocate(total_memory);
        ifs.read((char *)data, sizeof(double) * total_memory);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("SparseMatrixGroup::load_data on '" + filename +
                                "' failed.");
        ifs.close();
    }
    void save_data(const string &filename, bool save_info = false) const {
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
        ofs.write((char *)data, sizeof(double) * total_memory);
        if (!ofs.good())
            throw runtime_error("SparseMatrixGroup::save_data on '" + filename +
                                "' failed.");
        ofs.close();
    }
    void allocate(const vector<shared_ptr<SparseMatrixInfo<S>>> &infos,
                  double *ptr = 0) {
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
                alloc = dalloc;
            data = alloc->allocate(total_memory);
            memset(data, 0, sizeof(double) * total_memory);
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
        alloc->deallocate(data, total_memory);
        alloc = nullptr;
        total_memory = 0;
        data = nullptr;
    }
    void deallocate_infos() {
        for (int i = n - 1; i >= 0; i--)
            infos[i]->deallocate();
    }
    void randomize(double a = 0.0, double b = 1.0) const {
        Random::fill_rand_double(data, total_memory, a, b);
    }
    vector<pair<S, double>> delta_quanta() const {
        vector<pair<S, double>> r(n);
        for (int i = 0; i < n; i++)
            r[i] = make_pair(infos[i]->delta_quantum, (*this)[i]->norm());
        sort(r.begin(), r.end(),
             [](const pair<S, double> &a, const pair<S, double> &b) {
                 return a.second > b.second;
             });
        return r;
    }
    double norm() const {
        return MatrixFunctions::norm(MatrixRef(data, total_memory, 1));
    }
    void iscale(double d) const {
        MatrixFunctions::iscale(MatrixRef(data, total_memory, 1), d);
    }
    void normalize() const { iscale(1 / norm()); }
    shared_ptr<SparseMatrix<S>> operator[](int idx) const {
        assert(idx >= 0 && idx < n);
        shared_ptr<SparseMatrix<S>> r = make_shared<SparseMatrix<S>>();
        r->data = data + offsets[idx];
        r->info = infos[idx];
        r->total_memory = infos[idx]->get_total_memory();
        return r;
    }
    // l will have the same number of non-zero blocks as this matrix group
    // s will be labelled by right q labels
    void left_svd(vector<S> &rqs, vector<vector<shared_ptr<Tensor>>> &l,
                  vector<shared_ptr<Tensor>> &s,
                  vector<shared_ptr<Tensor>> &r) {
        map<S, uint32_t> qs_mp;
        for (const auto &info : infos)
            for (int i = 0; i < info->n; i++) {
                S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                            : info->quanta[i].get_ket();
                qs_mp[q] +=
                    (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            }
        int nr = (int)qs_mp.size(), k = 0;
        rqs.resize(nr);
        uint32_t tmp[nr + 1];
        memset(tmp, 0, sizeof(uint32_t) * (nr + 1));
        for (auto &mp : qs_mp)
            rqs[k] = mp.first, tmp[k + 1] = mp.second, k++;
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        double *dt = dalloc->allocate(tmp[nr]);
        uint32_t it[nr], sz[nr];
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int ii = 0; ii < n; ii++) {
            const auto &info = infos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                            : info->quanta[i].get_ket();
                int ir = lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
                uint32_t n_states =
                    (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
                memcpy(dt + (tmp[ir] + it[ir]),
                       data + offsets[ii] + info->n_states_total[i],
                       n_states * sizeof(double));
                sz[ir] = info->n_states_ket[i];
                it[ir] += n_states;
            }
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        vector<shared_ptr<Tensor>> merged_l;
        r.clear();
        s.clear();
        for (int ir = 0; ir < nr; ir++) {
            int nxr = sz[ir], nxl = (tmp[ir + 1] - tmp[ir]) / nxr;
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0);
            int nxk = min(nxl, nxr);
            shared_ptr<Tensor> tsl = make_shared<Tensor>(vector<int>{nxl, nxk});
            shared_ptr<Tensor> tss = make_shared<Tensor>(vector<int>{nxk});
            shared_ptr<Tensor> tsr = make_shared<Tensor>(vector<int>{nxk, nxr});
            MatrixFunctions::svd(MatrixRef(dt + tmp[ir], nxl, nxr), tsl->ref(),
                                 tss->ref().flip_dims(), tsr->ref());
            merged_l.push_back(tsl);
            s.push_back(tss);
            r.push_back(tsr);
        }
        memset(it, 0, sizeof(uint32_t) * nr);
        l.resize(n);
        for (int ii = 0; ii < n; ii++) {
            const auto &info = infos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->is_wavefunction ? -info->quanta[i].get_ket()
                                            : info->quanta[i].get_ket();
                int ir = lower_bound(rqs.begin(), rqs.end(), q) - rqs.begin();
                shared_ptr<Tensor> tsl = make_shared<Tensor>(vector<int>{
                    (int)info->n_states_bra[i], merged_l[ir]->shape[1]});
                memcpy(tsl->data.data(), merged_l[ir]->data.data() + it[ir],
                       tsl->size() * sizeof(double));
                it[ir] += tsl->size();
                l[ii].push_back(tsl);
            }
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == merged_l[ir]->size());
        dalloc->deallocate(dt, tmp[nr]);
    }
    // r will have the same number of non-zero blocks as this matrix group
    // s will be labelled by left q labels
    void right_svd(vector<S> &lqs, vector<shared_ptr<Tensor>> &l,
                   vector<shared_ptr<Tensor>> &s,
                   vector<vector<shared_ptr<Tensor>>> &r) {
        map<S, uint32_t> qs_mp;
        for (const auto &info : infos)
            for (int i = 0; i < info->n; i++) {
                S q = info->quanta[i].get_bra(info->delta_quantum);
                qs_mp[q] +=
                    (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            }
        int nl = (int)qs_mp.size(), p = 0;
        lqs.resize(nl);
        uint32_t tmp[nl + 1];
        memset(tmp, 0, sizeof(uint32_t) * (nl + 1));
        for (auto &mp : qs_mp)
            lqs[p] = mp.first, tmp[p + 1] = mp.second, p++;
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        double *dt = dalloc->allocate(tmp[nl]);
        uint32_t it[nl], sz[nl];
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int ii = 0; ii < n; ii++) {
            const auto &info = infos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->quanta[i].get_bra(info->delta_quantum);
                int il = lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
                uint32_t nxl = info->n_states_bra[i],
                         nxr = (tmp[il + 1] - tmp[il]) / nxl;
                assert((tmp[il + 1] - tmp[il]) % nxl == 0);
                uint32_t inr = info->n_states_ket[i];
                for (uint32_t k = 0; k < nxl; k++)
                    memcpy(dt + (tmp[il] + it[il] + k * nxr),
                           data + offsets[ii] +
                               (info->n_states_total[i] + k * inr),
                           inr * sizeof(double));
                sz[il] = nxl;
                it[il] += inr;
            }
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == (tmp[il + 1] - tmp[il]) / sz[il]);
        vector<shared_ptr<Tensor>> merged_r;
        l.clear();
        s.clear();
        for (int il = 0; il < nl; il++) {
            int nxl = sz[il], nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0);
            int nxk = min(nxl, nxr);
            shared_ptr<Tensor> tsl = make_shared<Tensor>(vector<int>{nxl, nxk});
            shared_ptr<Tensor> tss = make_shared<Tensor>(vector<int>{nxk});
            shared_ptr<Tensor> tsr = make_shared<Tensor>(vector<int>{nxk, nxr});
            MatrixFunctions::svd(MatrixRef(dt + tmp[il], nxl, nxr), tsl->ref(),
                                 tss->ref().flip_dims(), tsr->ref());
            l.push_back(tsl);
            s.push_back(tss);
            merged_r.push_back(tsr);
        }
        memset(it, 0, sizeof(uint32_t) * nl);
        r.resize(n);
        for (int ii = 0; ii < n; ii++) {
            const auto &info = infos[ii];
            for (int i = 0; i < info->n; i++) {
                S q = info->quanta[i].get_bra(info->delta_quantum);
                int il = lower_bound(lqs.begin(), lqs.end(), q) - lqs.begin();
                shared_ptr<Tensor> tsr = make_shared<Tensor>(vector<int>{
                    merged_r[il]->shape[0], (int)info->n_states_ket[i]});
                int inr = info->n_states_ket[i], ixr = merged_r[il]->shape[1];
                int inl = merged_r[il]->shape[0];
                for (uint32_t k = 0; k < inl; k++)
                    memcpy(tsr->data.data() + k * inr,
                           merged_r[il]->data.data() + (it[il] + k * ixr),
                           inr * sizeof(double));
                it[il] += inr;
                r[ii].push_back(tsr);
            }
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == merged_r[il]->shape[1]);
        dalloc->deallocate(dt, tmp[nl]);
    }
    friend ostream &operator<<(ostream &os, const SparseMatrixGroup<S> &c) {
        os << "DATA = [ ";
        for (int i = 0; i < c.total_memory; i++)
            os << setw(20) << setprecision(14) << c.data[i] << " ";
        os << "]" << endl;
        return os;
    }
};

} // namespace block2
