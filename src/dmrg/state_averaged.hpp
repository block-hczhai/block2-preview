
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

#include "mps.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <vector>

using namespace std;

namespace block2 {

// MPSInfo for multiple targets
template <typename S> struct MultiMPSInfo : MPSInfo<S> {
    using MPSInfo<S>::left_dims_fci;
    using MPSInfo<S>::right_dims_fci;
    using MPSInfo<S>::vacuum;
    using MPSInfo<S>::n_sites;
    using MPSInfo<S>::basis;
    using MPSInfo<S>::shallow_copy_to;
    vector<S> targets;
    MultiMPSInfo(int n_sites) : MPSInfo<S>(n_sites) {}
    MultiMPSInfo(int n_sites, S vacuum, const vector<S> &targets,
                 const vector<shared_ptr<StateInfo<S>>> &basis,
                 bool init_fci = true)
        : targets(targets), MPSInfo<S>(n_sites, vacuum, vacuum, basis, false) {
        if (init_fci)
            set_bond_dimension_fci();
    }
    // translate MPSInfo/MultiMPSInfo to MultiMPSInfo
    // fci part only
    static shared_ptr<MultiMPSInfo<S>>
    from_mps_info(const shared_ptr<MPSInfo<S>> &info) {
        shared_ptr<MultiMPSInfo<S>> minfo =
            info->get_type() & MPSTypes::MultiWfn
                ? make_shared<MultiMPSInfo<S>>(
                      info->n_sites, info->vacuum,
                      dynamic_pointer_cast<MultiMPSInfo<S>>(info)->targets,
                      info->basis, false)
                : make_shared<MultiMPSInfo<S>>(info->n_sites, info->vacuum,
                                               vector<S>{info->target},
                                               info->basis, false);
        for (int i = 0; i <= minfo->n_sites; i++)
            minfo->left_dims_fci[i] =
                make_shared<StateInfo<S>>(info->left_dims_fci[i]->deep_copy());
        for (int i = minfo->n_sites; i >= 0; i--)
            minfo->right_dims_fci[i] =
                make_shared<StateInfo<S>>(info->right_dims_fci[i]->deep_copy());
        minfo->bond_dim = info->bond_dim;
        minfo->tag = info->tag;
        return minfo;
    }
    shared_ptr<MPSInfo<S>> make_single(int itarget) const {
        shared_ptr<MPSInfo<S>> minfo = make_shared<MPSInfo<S>>(
            n_sites, vacuum, targets[itarget], basis, false);
        for (int i = 0; i <= minfo->n_sites; i++)
            minfo->left_dims_fci[i] =
                make_shared<StateInfo<S>>(left_dims_fci[i]->deep_copy());
        for (int i = minfo->n_sites; i >= 0; i--)
            minfo->right_dims_fci[i] =
                make_shared<StateInfo<S>>(right_dims_fci[i]->deep_copy());
        minfo->bond_dim = MPSInfo<S>::bond_dim;
        minfo->tag = MPSInfo<S>::tag;
        return minfo;
    }
    MPSTypes get_type() const override { return MPSTypes::MultiWfn; }
    vector<S> get_complementary(S q) const override {
        vector<S> r;
        for (auto target : targets) {
            S qs = target - q;
            for (int i = 0; i < qs.count(); i++)
                r.push_back(qs[i]);
        }
        sort(r.begin(), r.end());
        r.resize(distance(r.begin(), unique(r.begin(), r.end())));
        return r;
    }
    void set_bond_dimension_full_fci(S left_vacuum = S(S::invalid),
                                     S right_vacuum = S(S::invalid)) override {
        left_dims_fci[0] = make_shared<StateInfo<S>>(
            left_vacuum == S(S::invalid) ? vacuum : left_vacuum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *left_dims_fci[i], *basis[i], S(S::invalid)));
        right_dims_fci[n_sites] = make_shared<StateInfo<S>>(
            right_vacuum == S(S::invalid) ? vacuum : right_vacuum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] =
                make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                    *basis[i], *right_dims_fci[i + 1], S(S::invalid)));
    }
    void set_bond_dimension_fci(S left_vacuum = S(S::invalid),
                                S right_vacuum = S(S::invalid)) override {
        set_bond_dimension_full_fci(left_vacuum, right_vacuum);
        for (int i = 0; i <= n_sites; i++) {
            StateInfo<S>::multi_target_filter(*left_dims_fci[i],
                                              *right_dims_fci[i], targets);
            StateInfo<S>::multi_target_filter(*right_dims_fci[i],
                                              *left_dims_fci[i], targets);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i]->collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i]->collect();
    }
    void load_data(istream &ifs) override {
        MPSInfo<S>::load_data(ifs);
        int n_targets = 0;
        ifs.read((char *)&n_targets, sizeof(n_targets));
        targets.resize(n_targets);
        ifs.read((char *)targets.data(), sizeof(S) * n_targets);
    }
    void save_data(ostream &ofs) const override {
        MPSInfo<S>::save_data(ofs);
        int n_targets = (int)targets.size();
        ofs.write((char *)&n_targets, sizeof(n_targets));
        ofs.write((char *)targets.data(), sizeof(S) * n_targets);
    }
    shared_ptr<MPSInfo<S>> shallow_copy(const string &new_tag) const override {
        shared_ptr<MPSInfo<S>> info = make_shared<MultiMPSInfo<S>>(*this);
        info->tag = new_tag;
        shallow_copy_to(info);
        return info;
    }
    shared_ptr<MPSInfo<S>> deep_copy() const override {
        stringstream ss;
        save_data(ss);
        shared_ptr<MultiMPSInfo<S>> info = make_shared<MultiMPSInfo<S>>(0);
        info->load_data(ss);
        return info;
    }
};

// Matrix Product State for multiple targets and multiple wavefunctions
template <typename S, typename FL> struct MultiMPS : MPS<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    static const int cpx_sz = sizeof(FL) / sizeof(FP);
    typedef typename GMatrix<FL>::FC FC;
    using MPS<S, FL>::n_sites;
    using MPS<S, FL>::center;
    using MPS<S, FL>::dot;
    using MPS<S, FL>::info;
    using MPS<S, FL>::tensors;
    using MPS<S, FL>::canonical_form;
    using MPS<S, FL>::shallow_copy_to;
    // numebr of wavefunctions
    int nroots;
    // wavefunctions
    vector<shared_ptr<SparseMatrixGroup<S, FL>>> wfns;
    // weights of wavefunctions
    vector<FP> weights;
    MultiMPS(const shared_ptr<MultiMPSInfo<S>> &info) : MPS<S, FL>(info) {}
    MultiMPS(int n_sites, int center, int dot, int nroots)
        : MPS<S, FL>(n_sites, center, dot), nroots(nroots) {
        if (center >= 0 && center < n_sites)
            for (int i = center; i < center + dot; i++)
                canonical_form[i] = 'M';
        weights = vector<FP>(nroots, 1.0 / nroots);
    }
    MPSTypes get_type() const override { return MPSTypes::MultiWfn; }
    void initialize(const shared_ptr<MPSInfo<S>> &info, bool init_left = true,
                    bool init_right = true) override {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        this->info = info;
        assert(info->get_type() == MPSTypes::MultiWfn);
        shared_ptr<MultiMPSInfo<S>> minfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(info);
        vector<shared_ptr<SparseMatrixInfo<S>>> wfn_infos;
        wfn_infos.resize(minfo->targets.size());
        tensors.resize(n_sites);
        wfns.resize(nroots);
        if (init_left)
            MPS<S, FL>::initialize_left(info, center - 1);
        if (center >= 0 && center < n_sites && (init_left || init_right)) {
            for (size_t i = 0; i < minfo->targets.size(); i++)
                wfn_infos[i] = make_shared<SparseMatrixInfo<S>>(i_alloc);
            if (dot == 1) {
                StateInfo<S> t = StateInfo<S>::tensor_product(
                    *info->left_dims[center], *info->basis[center],
                    *info->left_dims_fci[center + dot]);
                for (size_t i = 0; i < minfo->targets.size(); i++)
                    wfn_infos[i]->initialize(t, *info->right_dims[center + dot],
                                             minfo->targets[i], false, true);
                canonical_form[center] = 'J';
            } else {
                StateInfo<S> tl = StateInfo<S>::tensor_product(
                    *info->left_dims[center], *info->basis[center],
                    *info->left_dims_fci[center + 1]);
                StateInfo<S> tr = StateInfo<S>::tensor_product(
                    *info->basis[center + 1], *info->right_dims[center + dot],
                    *info->right_dims_fci[center + 1]);
                for (size_t i = 0; i < minfo->targets.size(); i++)
                    wfn_infos[i]->initialize(tl, tr, minfo->targets[i], false,
                                             true);
            }
            for (int j = 0; j < nroots; j++) {
                wfns[j] = make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
                wfns[j]->allocate(wfn_infos);
            }
        }
        if (init_right)
            MPS<S, FL>::initialize_right(info, center + dot);
    }
    void flip_fused_form(
        int center, const shared_ptr<CG<S>> &cg,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) override {
        if (para_rule == nullptr || para_rule->is_root()) {
            load_wavefunction(center);
            if (canonical_form[center] == 'T')
                wfns = info->swap_multi_wfn_to_fused_left(center, wfns, cg);
            else if (canonical_form[center] == 'J')
                wfns = info->swap_multi_wfn_to_fused_right(center, wfns, cg);
            else
                assert(false);
            save_wavefunction(center);
            unload_wavefunction(center);
        }
        if (canonical_form[center] == 'T')
            canonical_form[center] = 'J';
        else if (canonical_form[center] == 'J')
            canonical_form[center] = 'T';
        else
            assert(false);
        if (para_rule != nullptr)
            para_rule->comm->barrier();
    }
    void random_canonicalize() override {
        for (int i = 0; i < n_sites; i++)
            MPS<S, FL>::random_canonicalize_tensor(i);
        for (int j = 0; j < nroots; j++)
            wfns[j]->randomize();
    }
    string get_filename(int i, const string &dir = "") const override {
        stringstream ss;
        ss << (dir == "" ? frame_<FP>()->mps_dir : dir) << "/"
           << frame_<FP>()->prefix << ".MMPS." << info->tag << "."
           << Parsing::to_string(i);
        return ss.str();
    }
    string get_wfn_filename(int i, const string &dir = "") const {
        stringstream ss;
        ss << (dir == "" ? frame_<FP>()->mps_dir : dir) << "/"
           << frame_<FP>()->prefix << ".MMPS-WFN." << info->tag << "."
           << Parsing::to_string(i);
        return ss.str();
    }
    shared_ptr<MPS<S, FL>> make_single(const string &xtag) {
        shared_ptr<MultiMPSInfo<S>> minfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(info);
        assert(nroots == 1);
        shared_ptr<MPSInfo<S>> xinfo = minfo->make_single(0);
        xinfo->load_mutable();
        shared_ptr<MPS<S, FL>> xmps = make_shared<MPS<S, FL>>(xinfo);
        load_data();
        load_mutable();
        int iinfo = 0;
        FP norm = (*wfns[0])[0]->norm(), normx;
        for (int i = 0; i < wfns[0]->n; i++)
            if ((normx = (*wfns[0])[i]->norm()) > norm)
                norm = normx, iinfo = i;
        xinfo->target = minfo->targets[iinfo];
        *xmps = *(MPS<S, FL> *)this;
        xmps->info = xinfo;
        xinfo->tag = xtag;
        xinfo->save_mutable();
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        int ctr = xmps->center;
        if (xmps->tensors[ctr] != nullptr)
            for (ctr = 0; ctr < xmps->n_sites && xmps->tensors[ctr] != nullptr;)
                ctr++;
        assert(xmps->tensors[ctr] == nullptr);
        xmps->tensors[ctr] = make_shared<SparseMatrix<S, FL>>(d_alloc);
        xmps->tensors[ctr]->allocate(wfns[0]->infos[iinfo]);
        xmps->tensors[ctr]->copy_data_from((*wfns[0])[iinfo]);
        const string rp = "CKS", og = "MJT";
        for (int i = 0; i < xmps->n_sites; i++)
            for (size_t j = 0; j < og.length(); j++)
                if (xmps->canonical_form[i] == og[j])
                    xmps->canonical_form[i] = rp[j];
        xmps->save_mutable();
        xmps->save_data();
        xmps->deallocate();
        xinfo->deallocate_mutable();
        return xmps;
    }
    // translate real MPS to complex MPS
    static shared_ptr<MultiMPS<S, FL>>
    make_complex(const shared_ptr<MPS<S, FL>> &mps, const string &xtag) {
        const int nroots = 2;
        shared_ptr<MultiMPSInfo<S>> xinfo =
            MultiMPSInfo<S>::from_mps_info(mps->info);
        xinfo->load_mutable();
        shared_ptr<MultiMPS<S, FL>> xmps = make_shared<MultiMPS<S, FL>>(xinfo);
        mps->load_data();
        mps->load_mutable();
        *(dynamic_pointer_cast<MPS<S, FL>>(xmps)) = *mps;
        xmps->info = xinfo;
        xinfo->tag = xtag;
        xinfo->save_mutable();
        xmps->nroots = nroots;
        xmps->wfns.resize(nroots);
        xmps->weights = vector<FP>(nroots, 1.0);
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        int ctr = xmps->center;
        if (!xmps->tensors[ctr]->info->is_wavefunction)
            for (ctr = 0; ctr < xmps->n_sites &&
                          (xmps->tensors[ctr] == nullptr ||
                           !xmps->tensors[ctr]->info->is_wavefunction);
                 ctr++)
                ;
        assert(xmps->tensors[ctr]->info->is_wavefunction);
        for (int i = 0; i < nroots; i++) {
            xmps->wfns[i] = make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
            xmps->wfns[i]->allocate(vector<shared_ptr<SparseMatrixInfo<S>>>{
                xmps->tensors[ctr]->info});
        }
        (*xmps->wfns[0])[0]->copy_data_from(xmps->tensors[ctr]);
        xmps->tensors[ctr] = nullptr;
        const string og = "CKS", rp = "MJT";
        for (int i = 0; i < xmps->n_sites; i++)
            for (size_t j = 0; j < og.length(); j++)
                if (xmps->canonical_form[i] == og[j])
                    xmps->canonical_form[i] = rp[j];
        xmps->save_mutable();
        xmps->save_data();
        xmps->deallocate();
        xinfo->deallocate_mutable();
        return xmps;
    }
    void iscale(const FC &d) {
        FL dre = xreal(d), dim = ximag(d);
        assert(nroots == 2);
        load_wavefunction(center);
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> tre(nullptr, (MKL_INT)wfns[0]->total_memory, 1);
        GMatrix<FL> tim(nullptr, (MKL_INT)wfns[1]->total_memory, 1);
        tre.data = (FL *)d_alloc->allocate(wfns[0]->total_memory * cpx_sz);
        tim.data = (FL *)d_alloc->allocate(wfns[1]->total_memory * cpx_sz);
        GMatrix<FL> wre(wfns[0]->data, (MKL_INT)wfns[0]->total_memory, 1);
        GMatrix<FL> wim(wfns[1]->data, (MKL_INT)wfns[1]->total_memory, 1);
        GMatrixFunctions<FL>::copy(tre, wre);
        GMatrixFunctions<FL>::copy(tim, wim);
        GMatrixFunctions<FL>::iscale(wre, dre);
        GMatrixFunctions<FL>::iscale(wim, dre);
        GMatrixFunctions<FL>::iadd(wre, tim, -dim);
        GMatrixFunctions<FL>::iadd(wim, tre, dim);
        d_alloc->deallocate(tim.data, wfns[1]->total_memory * cpx_sz);
        d_alloc->deallocate(tre.data, wfns[0]->total_memory * cpx_sz);
        save_wavefunction(center);
    }
    shared_ptr<MultiMPS<S, FL>> extract(int iroot, const string xtag) const {
        shared_ptr<MultiMPSInfo<S>> xinfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(info->deep_copy());
        xinfo->load_mutable();
        shared_ptr<MultiMPS<S, FL>> xmps = make_shared<MultiMPS<S, FL>>(xinfo);
        xmps->load_data();
        xmps->load_mutable();
        xinfo->tag = xtag;
        xinfo->save_mutable();
        xmps->nroots = 1;
        xmps->wfns[0] = xmps->wfns[iroot];
        xmps->wfns.resize(1);
        xmps->weights[0] = 1.0;
        xmps->weights.resize(1);
        xmps->save_mutable();
        xmps->save_data();
        xmps->deallocate();
        xinfo->deallocate_mutable();
        return xmps;
    }
    void shallow_copy_wfn_to(const shared_ptr<MultiMPS<S, FL>> &mps) const {
        if (frame_<FP>()->prefix_can_write) {
            for (int j = 0; j < nroots; j++)
                Parsing::link_file(get_wfn_filename(j),
                                   mps->get_wfn_filename(j));
        }
    }
    shared_ptr<MPS<S, FL>> shallow_copy(const string &new_tag) const override {
        shared_ptr<MPSInfo<S>> new_info = info->shallow_copy(new_tag);
        shared_ptr<MultiMPS<S, FL>> mps = make_shared<MultiMPS<S, FL>>(*this);
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        for (int i = 0; i < mps->n_sites; i++)
            if (mps->tensors[i] != nullptr)
                mps->tensors[i] = make_shared<SparseMatrix<S, FL>>(d_alloc);
        for (int j = 0; j < nroots; j++)
            mps->wfns[j] = make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
        mps->info = new_info;
        shallow_copy_to(mps);
        shallow_copy_wfn_to(mps);
        return mps;
    }
    shared_ptr<MPS<S, FL>> deep_copy(const string &xtag) const override {
        assert(info->get_type() == MPSTypes::MultiWfn);
        shared_ptr<MultiMPSInfo<S>> xinfo =
            dynamic_pointer_cast<MultiMPSInfo<S>>(info->deep_copy());
        xinfo->load_mutable();
        shared_ptr<MPS<S, FL>> xmps = make_shared<MultiMPS<S, FL>>(xinfo);
        xmps->load_data();
        xmps->load_mutable();
        xinfo->tag = xtag;
        xinfo->save_mutable();
        xmps->save_mutable();
        xmps->save_data();
        xmps->deallocate();
        xinfo->deallocate_mutable();
        return xmps;
    }
    void copy_data(const string &dir) const override {
        if (frame_<FP>()->prefix_can_write) {
            for (int i = 0; i < n_sites; i++)
                if (tensors[i] != nullptr)
                    Parsing::copy_file(get_filename(i), get_filename(i, dir));
                else if (i == center || i == center + 1)
                    for (int j = 0; j < nroots; j++)
                        Parsing::copy_file(get_wfn_filename(j),
                                           get_wfn_filename(j, dir));
            Parsing::copy_file(get_filename(-1), get_filename(-1, dir));
        }
    }
    void load_data() override {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        ifstream ifs(get_filename(-1).c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MultiMPS::load_data on '" + get_filename(-1) +
                                "' failed.");
        MPS<S, FL>::load_data_from(ifs);
        ifs.read((char *)&nroots, sizeof(nroots));
        weights.resize(nroots);
        wfns.resize(nroots);
        ifs.read((char *)&weights[0], sizeof(FP) * nroots);
        for (int i = 0; i < nroots; i++)
            wfns[i] = make_shared<SparseMatrixGroup<S, FL>>(d_alloc);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MultiMPS::load_data on '" + get_filename(-1) +
                                "' failed.");
        ifs.close();
    }
    void save_data() const override {
        if (frame_<FP>()->prefix_can_write) {
            string filename = get_filename(-1);
            if (Parsing::link_exists(filename))
                Parsing::remove_file(filename);
            ofstream ofs(filename.c_str(), ios::binary);
            if (!ofs.good())
                throw runtime_error("MultiMPS::save_data on '" +
                                    get_filename(-1) + "' failed.");
            MPS<S, FL>::save_data_to(ofs);
            ofs.write((char *)&nroots, sizeof(nroots));
            assert(weights.size() == nroots);
            ofs.write((char *)&weights[0], sizeof(FP) * nroots);
            if (!ofs.good())
                throw runtime_error("MultiMPS::save_data on '" +
                                    get_filename(-1) + "' failed.");
            ofs.close();
        }
    }
    void load_mutable() const override {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr)
                tensors[i]->load_data(get_filename(i), true, i_alloc);
        for (int j = 0; j < nroots; j++) {
            wfns[j]->load_data(get_wfn_filename(j), j == 0, i_alloc);
            wfns[j]->infos = wfns[0]->infos;
        }
    }
    void save_mutable() const override {
        if (frame_<FP>()->prefix_can_write) {
            for (int i = 0; i < n_sites; i++)
                if (tensors[i] != nullptr)
                    tensors[i]->save_data(get_filename(i), true);
            for (int j = 0; j < nroots; j++)
                wfns[j]->save_data(get_wfn_filename(j), j == 0);
        }
    }
    void save_wavefunction(int i) const {
        if (frame_<FP>()->prefix_can_write) {
            assert(tensors[i] == nullptr);
            for (int j = 0; j < nroots; j++)
                wfns[j]->save_data(get_wfn_filename(j), j == 0);
        }
    }
    void load_wavefunction(int i) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        assert(tensors[i] == nullptr);
        for (int j = 0; j < nroots; j++) {
            wfns[j]->load_data(get_wfn_filename(j), j == 0, i_alloc);
            wfns[j]->infos = wfns[0]->infos;
        }
    }
    void unload_wavefunction(int i) {
        assert(tensors[i] == nullptr);
        for (int j = nroots - 1; j >= 0; j--)
            wfns[j]->deallocate();
        if (nroots != 0)
            wfns[0]->deallocate_infos();
    }
    void save_tensor(int i) const override {
        if (frame_<FP>()->prefix_can_write) {
            assert(tensors[i] != nullptr || i == center);
            if (tensors[i] != nullptr)
                tensors[i]->save_data(get_filename(i), true);
            else
                for (int j = 0; j < nroots; j++)
                    wfns[j]->save_data(get_wfn_filename(j), j == 0);
        }
    }
    void load_tensor(int i) override {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        assert(tensors[i] != nullptr || i == center);
        if (tensors[i] != nullptr)
            tensors[i]->load_data(get_filename(i), true, i_alloc);
        else
            for (int j = 0; j < nroots; j++) {
                wfns[j]->load_data(get_wfn_filename(j), j == 0, i_alloc);
                wfns[j]->infos = wfns[0]->infos;
            }
    }
    void unload_tensor(int i) override {
        assert(tensors[i] != nullptr || i == center);
        if (tensors[i] != nullptr) {
            tensors[i]->deallocate();
            tensors[i]->info->deallocate();
        } else {
            for (int j = nroots - 1; j >= 0; j--)
                wfns[j]->deallocate();
            if (nroots != 0)
                wfns[0]->deallocate_infos();
        }
    }
    void deallocate() override {
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->deallocate();
        for (int j = nroots - 1; j >= 0; j--)
            wfns[j]->deallocate();
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->info->deallocate();
        if (nroots != 0)
            wfns[0]->deallocate_infos();
    }
};

} // namespace block2
