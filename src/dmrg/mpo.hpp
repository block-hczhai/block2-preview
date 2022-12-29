
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

#include "../core/csr_operator_functions.hpp"
#include "../core/hamiltonian.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/rule.hpp"
#include "../core/spin_permutation.hpp"
#include "../core/symbolic.hpp"
#include "../core/tensor_functions.hpp"
#include "mps.hpp"
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace block2 {

// Information for middle site numerical tranform
// from normal operators to complementary operators
template <typename S> struct MPOSchemer {
    uint16_t left_trans_site, right_trans_site;
    shared_ptr<SymbolicRowVector<S>> left_new_operator_names;
    shared_ptr<SymbolicColumnVector<S>> right_new_operator_names;
    shared_ptr<SymbolicRowVector<S>> left_new_operator_exprs;
    shared_ptr<SymbolicColumnVector<S>> right_new_operator_exprs;
    MPOSchemer(uint16_t left_trans_site, uint16_t right_trans_site)
        : left_trans_site(left_trans_site), right_trans_site(right_trans_site) {
    }
    shared_ptr<MPOSchemer> copy() const {
        shared_ptr<MPOSchemer> r =
            make_shared<MPOSchemer>(left_trans_site, right_trans_site);
        r->left_new_operator_names = dynamic_pointer_cast<SymbolicRowVector<S>>(
            left_new_operator_names->copy());
        r->right_new_operator_names =
            dynamic_pointer_cast<SymbolicColumnVector<S>>(
                right_new_operator_names->copy());
        r->left_new_operator_exprs = dynamic_pointer_cast<SymbolicRowVector<S>>(
            left_new_operator_exprs->copy());
        r->right_new_operator_exprs =
            dynamic_pointer_cast<SymbolicColumnVector<S>>(
                right_new_operator_exprs->copy());
        return r;
    }
    template <typename FL> void load_data(istream &ifs, bool minimal = false) {
        ifs.read((char *)&left_trans_site, sizeof(left_trans_site));
        ifs.read((char *)&right_trans_site, sizeof(right_trans_site));
        left_new_operator_names = dynamic_pointer_cast<SymbolicRowVector<S>>(
            load_symbolic<S, FL>(ifs));
        right_new_operator_names =
            dynamic_pointer_cast<SymbolicColumnVector<S>>(
                load_symbolic<S, FL>(ifs));
        left_new_operator_exprs = dynamic_pointer_cast<SymbolicRowVector<S>>(
            load_symbolic<S, FL>(ifs));
        right_new_operator_exprs =
            dynamic_pointer_cast<SymbolicColumnVector<S>>(
                load_symbolic<S, FL>(ifs));
        if (minimal)
            unload_data();
    }
    void unload_data() {
        left_new_operator_names = nullptr;
        right_new_operator_names = nullptr;
        left_new_operator_exprs = nullptr;
        right_new_operator_exprs = nullptr;
    }
    void save_data(ostream &ofs) const {
        ofs.write((char *)&left_trans_site, sizeof(left_trans_site));
        ofs.write((char *)&right_trans_site, sizeof(right_trans_site));
        save_symbolic<S>(left_new_operator_names, ofs);
        save_symbolic<S>(right_new_operator_names, ofs);
        save_symbolic<S>(left_new_operator_exprs, ofs);
        save_symbolic<S>(right_new_operator_exprs, ofs);
    }
    string get_transform_formulas() const {
        stringstream ss;
        ss << "LEFT  TRANSFORM :: SITE = " << (int)left_trans_site << endl;
        for (int j = 0; j < left_new_operator_names->data.size(); j++) {
            if (left_new_operator_names->data[j]->get_type() != OpTypes::Zero)
                ss << "[" << setw(4) << j << "] " << setw(15)
                   << left_new_operator_names->data[j]
                   << " := " << left_new_operator_exprs->data[j] << endl;
            else
                ss << "[" << setw(4) << j << "] "
                   << left_new_operator_names->data[j] << endl;
        }
        ss << endl;
        ss << "RIGHT TRANSFORM :: SITE = " << (int)right_trans_site << endl;
        for (int j = 0; j < right_new_operator_names->data.size(); j++) {
            if (right_new_operator_names->data[j]->get_type() != OpTypes::Zero)
                ss << "[" << setw(4) << j << "] " << setw(15)
                   << right_new_operator_names->data[j]
                   << " := " << right_new_operator_exprs->data[j] << endl;
            else
                ss << "[" << setw(4) << j << "] "
                   << right_new_operator_names->data[j] << endl;
        }
        ss << endl;
        return ss.str();
    }
};

// Symbolic Matrix Product Operator
template <typename S, typename FL> struct MPO {
    typedef typename GMatrix<FL>::FP FP;
    vector<shared_ptr<OperatorTensor<S, FL>>> tensors;
    vector<shared_ptr<Symbolic<S>>> left_operator_names;
    vector<shared_ptr<Symbolic<S>>> right_operator_names;
    vector<shared_ptr<Symbolic<S>>> middle_operator_names;
    vector<shared_ptr<Symbolic<S>>> left_operator_exprs;
    vector<shared_ptr<Symbolic<S>>> right_operator_exprs;
    vector<shared_ptr<Symbolic<S>>> middle_operator_exprs;
    shared_ptr<OpElement<S, FL>> op;
    shared_ptr<MPOSchemer<S>> schemer;
    // Number of sites
    int n_sites;
    // Const energy term
    typename const_fl_type<FL>::FL const_e;
    shared_ptr<TensorFunctions<S, FL>> tf;
    vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>> site_op_infos;
    vector<shared_ptr<StateInfo<S>>> basis; // only for fused mpo
    // N = Normal, S = CSR
    string sparse_form;
    // Marks for dynamically load data
    vector<vector<size_t>> archive_marks;
    size_t archive_schemer_mark;
    string archive_filename = "";
    string tag = "H";
    mutable double tread = 0;  //!< IO Time cost for reading scratch files.
    mutable double twrite = 0; //!< IO Time cost for writing scratch files.
    shared_ptr<Hamiltonian<S, FL>> hamil =
        nullptr;   //!< Optional field, to prevent early release of Hamiltonian
                   //!< data causing MPO data invalid
    S left_vacuum; //!< to support singlet embedding for non-singlet MPO. For
                   //!< normal MPO this is normal vacuum
    shared_ptr<NPDMScheme> npdm_scheme =
        nullptr; //!< Optional field for constructing npdm expectation values
                 //!< without symbols
    int npdm_parallel_center = -1; //!< Optional field for npdm parallelization
    MPO(int n_sites, const string &tag = "H")
        : n_sites(n_sites), tag(tag), sparse_form(n_sites, 'N'),
          const_e((typename const_fl_type<FL>::FL)0.0), op(nullptr),
          schemer(nullptr), tf(nullptr) {}
    virtual ~MPO() = default;
    virtual void build() {}
    virtual AncillaTypes get_ancilla_type() const { return AncillaTypes::None; }
    virtual ParallelTypes get_parallel_type() const {
        return ParallelTypes::Serial;
    }
    // in bytes; 0 = peak term, 1 = peak memory, 2 = total disk storage
    // only count lower bound of floating point numbers
    virtual vector<size_t> estimate_storage(shared_ptr<MPSInfo<S>> info,
                                            int dot) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        size_t peak = 0, total = 0, psz = 0;
        shared_ptr<SparseMatrixInfo<S>> mat_info =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        vector<size_t> left_total(1, 0), right_total(1, 0);
        for (int i = 0; i < n_sites; i++) {
            size_t sz = 0;
            map<S, size_t> mpsz;
            load_left_operators(i);
            for (auto xop : left_operator_names[i]->data) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(xop);
                if (!mpsz.count(op->q_label)) {
                    mat_info->initialize(*info->left_dims[i + 1],
                                         *info->left_dims[i + 1], op->q_label,
                                         op->q_label.is_fermion());
                    mpsz[op->q_label] = mat_info->get_total_memory();
                    mat_info->deallocate();
                }
                sz += mpsz.at(op->q_label);
            }
            unload_left_operators(i);
            left_total.push_back(left_total.back() + sz);
        }
        for (int i = n_sites - 1; i >= 0; i--) {
            size_t sz = 0;
            map<S, size_t> mpsz;
            load_right_operators(i);
            for (auto xop : right_operator_names[i]->data) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(xop);
                if (!mpsz.count(op->q_label)) {
                    mat_info->initialize(*info->right_dims[i],
                                         *info->right_dims[i], op->q_label,
                                         op->q_label.is_fermion());
                    mpsz[op->q_label] = mat_info->get_total_memory();
                    mat_info->deallocate();
                }
                sz += mpsz.at(op->q_label);
            }
            unload_right_operators(i);
            right_total.push_back(right_total.back() + sz);
        }
        for (int i = 0; i < n_sites; i++) {
            if (dot == 2 && i == n_sites - 1)
                break;
            StateInfo<S> tl, tr;
            int iL = -1, iR = -1;
            if (dot == 2) {
                tl = StateInfo<S>::tensor_product(*info->left_dims[i],
                                                  *info->basis[i],
                                                  *info->left_dims_fci[i + 1]);
                tr = StateInfo<S>::tensor_product(*info->basis[i + 1],
                                                  *info->right_dims[i + 2],
                                                  *info->right_dims_fci[i + 1]);
                iL = i, iR = i + 1;
            } else {
                bool fuse_left = schemer == nullptr
                                     ? (i <= n_sites / 2)
                                     : (i < schemer->left_trans_site);
                iL = i, iR = i;
                if (fuse_left) {
                    tl = StateInfo<S>::tensor_product(
                        *info->left_dims[i], *info->basis[i],
                        *info->left_dims_fci[i + 1]);
                    tr = *info->right_dims[i + 1];

                } else {
                    tl = *info->left_dims[i];
                    tr = StateInfo<S>::tensor_product(*info->basis[i],
                                                      *info->right_dims[i + 1],
                                                      *info->right_dims_fci[i]);
                }
            }
            size_t sz = 0;
            map<S, size_t> mpszl, mpszr;
            load_left_operators(iL);
            for (auto xop : left_operator_names[iL]->data) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(xop);
                if (!mpszl.count(op->q_label)) {
                    mat_info->initialize(tl, tl, op->q_label,
                                         op->q_label.is_fermion());
                    mpszl[op->q_label] = mat_info->get_total_memory();
                    mat_info->deallocate();
                }
                sz += mpszl.at(op->q_label);
                psz = max(psz, mpszl.at(op->q_label));
            }
            unload_left_operators(iL);
            load_right_operators(iR);
            for (auto xop : right_operator_names[iR]->data) {
                shared_ptr<OpElement<S, FL>> op =
                    dynamic_pointer_cast<OpElement<S, FL>>(xop);
                if (!mpszr.count(op->q_label)) {
                    mat_info->initialize(tr, tr, op->q_label,
                                         op->q_label.is_fermion());
                    mpszr[op->q_label] = mat_info->get_total_memory();
                    mat_info->deallocate();
                }
                sz += mpszr.at(op->q_label);
                psz = max(psz, mpszr.at(op->q_label));
            }
            unload_right_operators(iR);
            peak = max(peak, sz);
        }
        total = left_total.back() + right_total.back();
        return vector<size_t>{psz * 8, peak * 8, total * 8};
    }
    virtual void deallocate() {
        for (int16_t m = n_sites - 1; m >= 0; m--)
            if (tensors[m] != nullptr)
                tensors[m]->deallocate();
    }
    // nnz, size, bond dimension
    tuple<size_t, size_t, int> get_summary() const {
        size_t lnnz = 0, lsz = 0, rnnz = 0, rsz = 0;
        int bdim = 0;
        for (int ii = 0; ii < n_sites; ii++) {
            shared_ptr<OperatorTensor<S, FL>> opt = tensors[ii];
            lnnz += opt->lmat->nnz(), rnnz += opt->rmat->nnz();
            lsz += opt->lmat->size(), rsz += opt->rmat->size();
            bdim = max(max(bdim, (int)opt->lmat->n), (int)opt->lmat->m);
        }
        return make_tuple(max(lnnz, rnnz), max(lsz, rsz), bdim);
    }
    string get_filename(int i, int ixtag, const string &dir = "") const {
        const static string xtag[] = {"TENSOR", "LEFT.OP", "RIGHT.OP",
                                      "MIDDLE.OP"};
        stringstream ss;
        ss << (dir == "" ? frame_<FP>()->mpo_dir : dir) << "/"
           << frame_<FP>()->prefix_distri << ".MPO." << tag << "."
           << xtag[ixtag] << "." << Parsing::to_string(i);
        return ss.str();
    }
    void load_tensor(int i, bool no_ops = false) {
        if (archive_filename == "" && !frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = frame_<FP>()->minimal_memory_usage
                              ? get_filename(i, 0)
                              : archive_filename;
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPO:load_tensor on '" + filename +
                                "' failed.");
        if (!frame_<FP>()->minimal_memory_usage) {
            ifs.clear();
            ifs.seekg(archive_marks[i][0]);
        }
        tensors[i] = make_shared<OperatorTensor<S, FL>>();
        tensors[i]->load_data(ifs, false, no_ops);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPO:load_tensor on '" + filename +
                                "' failed.");
        ifs.close();
#pragma omp atomic
        tread += _t.get_time();
    }
    void save_tensor(int i) {
        if (!frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = get_filename(i, 0);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPO:save_tensor on '" + filename +
                                "' failed.");
        tensors[i]->save_data(ofs);
        if (!ofs.good())
            throw runtime_error("MPO:save_tensor on '" + filename +
                                "' failed.");
        ofs.close();
#pragma omp atomic
        twrite += _t.get_time();
    }
    void unload_tensor(int i) {
        assert(i < n_sites);
        if (archive_filename != "" || frame_<FP>()->minimal_memory_usage)
            tensors[i] = nullptr;
    }
    void load_schemer() {
        if (archive_filename == "" && !frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        string filename = frame_<FP>()->minimal_memory_usage
                              ? get_filename(-1, 0)
                              : archive_filename;
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPO:load_schemer on '" + filename +
                                "' failed.");
        if (!frame_<FP>()->minimal_memory_usage) {
            ifs.clear();
            ifs.seekg(archive_schemer_mark);
        }
        schemer->template load_data<FL>(ifs, false);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPO:load_schemer on '" + filename +
                                "' failed.");
        ifs.close();
#pragma omp atomic
        tread += _t.get_time();
    }
    void save_schemer() {
        if (!frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        string filename = get_filename(-1, 0);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPO:save_schemer on '" + filename +
                                "' failed.");
        schemer->save_data(ofs);
        if (!ofs.good())
            throw runtime_error("MPO:save_schemer on '" + filename +
                                "' failed.");
        ofs.close();
#pragma omp atomic
        twrite += _t.get_time();
    }
    void unload_schemer() {
        if (archive_filename != "" || frame_<FP>()->minimal_memory_usage)
            schemer->unload_data();
    }
    void load_left_operators(int i) {
        if (archive_filename == "" && !frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = frame_<FP>()->minimal_memory_usage
                              ? get_filename(i, 1)
                              : archive_filename;
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPO:load_left_operators on '" + filename +
                                "' failed.");
        if (!frame_<FP>()->minimal_memory_usage) {
            ifs.clear();
            ifs.seekg(archive_marks[i][1]);
            if (archive_marks[i][1] != 0)
                left_operator_names[i] = load_symbolic<S, FL>(ifs);
            ifs.clear();
            ifs.seekg(archive_marks[i][4]);
            if (archive_marks[i][4] != 0)
                left_operator_exprs[i] = load_symbolic<S, FL>(ifs);
        } else {
            int iex = 0;
            ifs.read((char *)&iex, sizeof(iex));
            if (iex & 1)
                left_operator_names[i] = load_symbolic<S, FL>(ifs);
            if (iex & 2)
                left_operator_exprs[i] = load_symbolic<S, FL>(ifs);
        }
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPO:load_left_operators on '" + filename +
                                "' failed.");
        ifs.close();
#pragma omp atomic
        tread += _t.get_time();
    }
    void save_left_operators(int i) {
        if (!frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = get_filename(i, 1);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPO:save_left_operators on '" + filename +
                                "' failed.");
        int iex = (i < left_operator_names.size() &&
                   left_operator_names[i] != nullptr) |
                  ((i < left_operator_exprs.size() &&
                    left_operator_exprs[i] != nullptr)
                   << 1);
        ofs.write((char *)&iex, sizeof(iex));
        if (iex & 1)
            save_symbolic<S>(left_operator_names[i], ofs);
        if (iex & 2)
            save_symbolic<S>(left_operator_exprs[i], ofs);
        if (!ofs.good())
            throw runtime_error("MPO:save_left_operators on '" + filename +
                                "' failed.");
        ofs.close();
#pragma omp atomic
        twrite += _t.get_time();
    }
    void unload_left_operators(int i) {
        if (archive_filename != "" || frame_<FP>()->minimal_memory_usage) {
            assert(i < n_sites);
            left_operator_names[i] = nullptr;
            if (left_operator_exprs.size() != 0)
                left_operator_exprs[i] = nullptr;
        }
    }
    void load_right_operators(int i) {
        if (archive_filename == "" && !frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = frame_<FP>()->minimal_memory_usage
                              ? get_filename(i, 2)
                              : archive_filename;
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPO:load_right_operators on '" + filename +
                                "' failed.");
        if (!frame_<FP>()->minimal_memory_usage) {
            ifs.clear();
            ifs.seekg(archive_marks[i][2]);
            if (archive_marks[i][2] != 0)
                right_operator_names[i] = load_symbolic<S, FL>(ifs);
            ifs.clear();
            ifs.seekg(archive_marks[i][5]);
            if (archive_marks[i][5] != 0)
                right_operator_exprs[i] = load_symbolic<S, FL>(ifs);
        } else {
            int iex = 0;
            ifs.read((char *)&iex, sizeof(iex));
            if (iex & 1)
                right_operator_names[i] = load_symbolic<S, FL>(ifs);
            if (iex & 2)
                right_operator_exprs[i] = load_symbolic<S, FL>(ifs);
        }
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPO:load_right_operators on '" + filename +
                                "' failed.");
        ifs.close();
#pragma omp atomic
        tread += _t.get_time();
    }
    void save_right_operators(int i) {
        if (!frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = get_filename(i, 2);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPO:save_right_operators on '" + filename +
                                "' failed.");
        int iex = (i < right_operator_names.size() &&
                   right_operator_names[i] != nullptr) |
                  ((i < right_operator_exprs.size() &&
                    right_operator_exprs[i] != nullptr)
                   << 1);
        ofs.write((char *)&iex, sizeof(iex));
        if (iex & 1)
            save_symbolic<S>(right_operator_names[i], ofs);
        if (iex & 2)
            save_symbolic<S>(right_operator_exprs[i], ofs);
        if (!ofs.good())
            throw runtime_error("MPO:save_right_operators on '" + filename +
                                "' failed.");
        ofs.close();
#pragma omp atomic
        twrite += _t.get_time();
    }
    void unload_right_operators(int i) {
        if (archive_filename != "" || frame_<FP>()->minimal_memory_usage) {
            assert(i < n_sites);
            right_operator_names[i] = nullptr;
            if (right_operator_exprs.size() != 0)
                right_operator_exprs[i] = nullptr;
        }
    }
    void load_middle_operators(int i) {
        if (archive_filename == "" && !frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = frame_<FP>()->minimal_memory_usage
                              ? get_filename(i, 3)
                              : archive_filename;
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPO:load_middle_operators on '" + filename +
                                "' failed.");
        if (!frame_<FP>()->minimal_memory_usage) {
            ifs.clear();
            ifs.seekg(archive_marks[i][3]);
            if (archive_marks[i][3] != 0)
                middle_operator_names[i] = load_symbolic<S, FL>(ifs);
            ifs.clear();
            ifs.seekg(archive_marks[i][6]);
            if (archive_marks[i][6] != 0)
                middle_operator_exprs[i] = load_symbolic<S, FL>(ifs);
        } else {
            int iex = 0;
            ifs.read((char *)&iex, sizeof(iex));
            if (iex & 1)
                middle_operator_names[i] = load_symbolic<S, FL>(ifs);
            if (iex & 2)
                middle_operator_exprs[i] = load_symbolic<S, FL>(ifs);
        }
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPO:load_middle_operators on '" + filename +
                                "' failed.");
        ifs.close();
#pragma omp atomic
        tread += _t.get_time();
    }
    void save_middle_operators(int i) {
        if (!frame_<FP>()->minimal_memory_usage)
            return;
        Timer _t;
        _t.get_time();
        assert(i < n_sites);
        string filename = get_filename(i, 3);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPO:save_middle_operators on '" + filename +
                                "' failed.");
        int iex = (i < middle_operator_names.size() &&
                   middle_operator_names[i] != nullptr) |
                  ((i < middle_operator_exprs.size() &&
                    middle_operator_exprs[i] != nullptr)
                   << 1);
        ofs.write((char *)&iex, sizeof(iex));
        if (iex & 1)
            save_symbolic<S>(middle_operator_names[i], ofs);
        if (iex & 2)
            save_symbolic<S>(middle_operator_exprs[i], ofs);
        if (!ofs.good())
            throw runtime_error("MPO:save_middle_operators on '" + filename +
                                "' failed.");
        ofs.close();
#pragma omp atomic
        twrite += _t.get_time();
    }
    void unload_middle_operators(int i) {
        if (archive_filename != "" || frame_<FP>()->minimal_memory_usage) {
            assert(i < n_sites);
            middle_operator_names[i] = nullptr;
            middle_operator_exprs[i] = nullptr;
        }
    }
    virtual void load_data(istream &ifs, bool minimal = false,
                           bool save = false) {
        ifs.read((char *)&n_sites, sizeof(n_sites));
        int ltag = 0;
        ifs.read((char *)&ltag, sizeof(ltag));
        tag = string(ltag, ' ');
        ifs.read((char *)&tag[0], sizeof(char) * ltag);
        ifs.read((char *)&const_e, sizeof(const_e));
        ifs.read((char *)&left_vacuum, sizeof(left_vacuum));
        sparse_form = string(n_sites, 'N');
        ifs.read((char *)&sparse_form[0], sizeof(char) * n_sites);
        shared_ptr<CG<S>> cg = make_shared<CG<S>>();
        if (sparse_form.find('S') == string::npos)
            tf = make_shared<TensorFunctions<S, FL>>(
                make_shared<OperatorFunctions<S, FL>>(cg));
        else
            tf = make_shared<TensorFunctions<S, FL>>(
                make_shared<CSROperatorFunctions<S, FL>>(cg));
        bool has_op, has_schemer;
        ifs.read((char *)&has_op, sizeof(has_op));
        ifs.read((char *)&has_schemer, sizeof(has_schemer));
        if (has_op)
            op = dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs));
        if (has_schemer) {
            schemer = make_shared<MPOSchemer<S>>(0, 0);
            if (minimal)
                archive_schemer_mark = (size_t)ifs.tellg();
            schemer->template load_data<FL>(ifs, minimal);
            if (minimal && save)
                save_schemer();
        }
        int sz, sub_sz;
        ifs.read((char *)&sz, sizeof(sz));
        site_op_infos.resize(sz);
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        for (int i = 0; i < sz; i++) {
            ifs.read((char *)&sub_sz, sizeof(sub_sz));
            site_op_infos[i].resize(sub_sz);
            for (int j = 0; j < sub_sz; j++) {
                ifs.read((char *)&site_op_infos[i][j].first,
                         sizeof(site_op_infos[i][j].first));
                site_op_infos[i][j].second =
                    make_shared<SparseMatrixInfo<S>>(i_alloc);
                site_op_infos[i][j].second->load_data(ifs);
            }
        }
        archive_marks.resize(n_sites + 1);
        for (int i = 0; i <= n_sites; i++)
            archive_marks[i].resize(7);
        ifs.read((char *)&sz, sizeof(sz));
        tensors.resize(sz);
        for (int i = 0; i < sz; i++) {
            tensors[i] = make_shared<OperatorTensor<S, FL>>();
            if (minimal)
                archive_marks[i][0] = (size_t)ifs.tellg();
            tensors[i]->load_data(ifs);
            if (minimal && save)
                save_tensor(i);
            if (minimal)
                tensors[i] = nullptr;
        }
        ifs.read((char *)&sz, sizeof(sz));
        basis.resize(sz);
        for (int i = 0; i < sz; i++) {
            basis[i] = make_shared<StateInfo<S>>();
            basis[i]->load_data(ifs);
        }
        ifs.read((char *)&sz, sizeof(sz));
        left_operator_names.resize(sz);
        for (int i = 0; i < sz; i++) {
            if (minimal)
                archive_marks[i][1] = (size_t)ifs.tellg();
            left_operator_names[i] = load_symbolic<S, FL>(ifs);
            if (minimal && save)
                save_left_operators(i);
            if (minimal)
                left_operator_names[i] = nullptr;
        }
        ifs.read((char *)&sz, sizeof(sz));
        right_operator_names.resize(sz);
        for (int i = 0; i < sz; i++) {
            if (minimal)
                archive_marks[i][2] = (size_t)ifs.tellg();
            right_operator_names[i] = load_symbolic<S, FL>(ifs);
            if (minimal && save)
                save_right_operators(i);
            if (minimal)
                right_operator_names[i] = nullptr;
        }
        ifs.read((char *)&sz, sizeof(sz));
        middle_operator_names.resize(sz);
        for (int i = 0; i < sz; i++) {
            if (minimal)
                archive_marks[i][3] = (size_t)ifs.tellg();
            middle_operator_names[i] = load_symbolic<S, FL>(ifs);
            if (minimal && save)
                save_middle_operators(i);
            if (minimal)
                middle_operator_names[i] = nullptr;
        }
        ifs.read((char *)&sz, sizeof(sz));
        left_operator_exprs.resize(sz);
        for (int i = 0; i < sz; i++) {
            if (minimal)
                archive_marks[i][4] = (size_t)ifs.tellg();
            left_operator_exprs[i] = load_symbolic<S, FL>(ifs);
            if (minimal && save) {
                load_left_operators(i);
                save_left_operators(i);
                unload_left_operators(i);
            }
            if (minimal)
                left_operator_exprs[i] = nullptr;
        }
        ifs.read((char *)&sz, sizeof(sz));
        right_operator_exprs.resize(sz);
        for (int i = 0; i < sz; i++) {
            if (minimal)
                archive_marks[i][5] = (size_t)ifs.tellg();
            right_operator_exprs[i] = load_symbolic<S, FL>(ifs);
            if (minimal && save) {
                load_right_operators(i);
                save_right_operators(i);
                unload_right_operators(i);
            }
            if (minimal)
                right_operator_exprs[i] = nullptr;
        }
        ifs.read((char *)&sz, sizeof(sz));
        middle_operator_exprs.resize(sz);
        for (int i = 0; i < sz; i++) {
            if (minimal)
                archive_marks[i][6] = (size_t)ifs.tellg();
            middle_operator_exprs[i] = load_symbolic<S, FL>(ifs);
            if (minimal && save) {
                load_middle_operators(i);
                save_middle_operators(i);
                unload_middle_operators(i);
            }
            if (minimal)
                middle_operator_exprs[i] = nullptr;
        }
    }
    void load_data(const string &filename, bool minimal = false) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("MPO:load_data on '" + filename + "' failed.");
        load_data(ifs, minimal);
        if (minimal)
            archive_filename = filename;
        if (ifs.fail() || ifs.bad())
            throw runtime_error("MPO:load_data on '" + filename + "' failed.");
        ifs.close();
    }
    virtual void save_data(ostream &ofs) {
        assert(archive_filename == "");
        ofs.write((char *)&n_sites, sizeof(n_sites));
        int ltag = (int)tag.size();
        ofs.write((char *)&ltag, sizeof(ltag));
        ofs.write((char *)&tag[0], sizeof(char) * ltag);
        ofs.write((char *)&const_e, sizeof(const_e));
        ofs.write((char *)&left_vacuum, sizeof(left_vacuum));
        ofs.write((char *)&sparse_form[0], sizeof(char) * n_sites);
        bool has_op = op != nullptr, has_schemer = schemer != nullptr;
        ofs.write((char *)&has_op, sizeof(has_op));
        ofs.write((char *)&has_schemer, sizeof(has_schemer));
        if (has_op)
            save_expr<S>(op, ofs);
        if (has_schemer) {
            load_schemer();
            schemer->save_data(ofs);
            unload_schemer();
        }
        int sz = (int)site_op_infos.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            int sub_sz = (int)site_op_infos[i].size();
            ofs.write((char *)&sub_sz, sizeof(sub_sz));
            for (int j = 0; j < sub_sz; j++) {
                ofs.write((char *)&site_op_infos[i][j].first,
                          sizeof(site_op_infos[i][j].first));
                assert(site_op_infos[i][j].second != nullptr);
                site_op_infos[i][j].second->save_data(ofs);
            }
        }
        sz = (int)tensors.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_tensor(i);
            tensors[i]->save_data(ofs);
            unload_tensor(i);
        }
        sz = (int)basis.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++)
            basis[i]->save_data(ofs);
        sz = (int)left_operator_names.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_left_operators(i);
            save_symbolic<S>(left_operator_names[i], ofs);
            unload_left_operators(i);
        }
        sz = (int)right_operator_names.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_right_operators(i);
            save_symbolic<S>(right_operator_names[i], ofs);
            unload_right_operators(i);
        }
        sz = (int)middle_operator_names.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_middle_operators(i);
            save_symbolic<S>(middle_operator_names[i], ofs);
            unload_middle_operators(i);
        }
        sz = (int)left_operator_exprs.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_left_operators(i);
            save_symbolic<S>(left_operator_exprs[i], ofs);
            unload_left_operators(i);
        }
        sz = (int)right_operator_exprs.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_right_operators(i);
            save_symbolic<S>(right_operator_exprs[i], ofs);
            unload_right_operators(i);
        }
        sz = (int)middle_operator_exprs.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++) {
            load_middle_operators(i);
            save_symbolic<S>(middle_operator_exprs[i], ofs);
            unload_middle_operators(i);
        }
    }
    void save_data(const string &filename) {
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("MPO:save_data on '" + filename + "' failed.");
        save_data(ofs);
        if (!ofs.good())
            throw runtime_error("MPO:save_data on '" + filename + "' failed.");
        ofs.close();
    }
    // For simplified MPO, the tensor symbols can be deleted
    // to save memory and storage
    void reduce_data() const {
        assert(left_operator_exprs.size() != 0);
        for (int i = 1; i < n_sites - 1; i++)
            tensors[i]->lmat = tensors[i]->rmat = 0;
    }
    virtual shared_ptr<MPO> deep_copy(const string &xtag = "") {
        stringstream ss;
        save_data(ss);
        shared_ptr<MPO> mpo = make_shared<MPO>(0, xtag == "" ? tag : xtag);
        mpo->load_data(ss, frame_<FP>()->minimal_memory_usage, true);
        mpo->tf = this->tf;
        return mpo;
    }
    string get_blocking_formulas() {
        stringstream ss;
        for (int i = 0; i < n_sites; i++) {
            ss << "LEFT BLOCKING :: SITE = " << i << endl;
            load_left_operators(i);
            for (int j = 0; j < left_operator_names[i]->data.size(); j++) {
                if (left_operator_exprs.size() != 0)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << left_operator_names[i]->data[j]
                       << " := " << left_operator_exprs[i]->data[j] << endl;
                else
                    ss << "[" << setw(4) << j << "] "
                       << left_operator_names[i]->data[j] << endl;
            }
            ss << endl;
            unload_left_operators(i);
        }
        for (int i = n_sites - 1; i >= 0; i--) {
            ss << "RIGHT BLOCKING :: SITE = " << i << endl;
            load_right_operators(i);
            for (int j = 0; j < right_operator_names[i]->data.size(); j++) {
                if (right_operator_exprs.size() != 0)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << right_operator_names[i]->data[j]
                       << " := " << right_operator_exprs[i]->data[j] << endl;
                else
                    ss << "[" << setw(4) << j << "] "
                       << right_operator_names[i]->data[j] << endl;
            }
            ss << endl;
            unload_right_operators(i);
        }
        if (middle_operator_names.size() != 0) {
            for (int i = 0; i < n_sites - 1; i++) {
                ss << "HAMIL PARTITION :: SITE = " << i << endl;
                load_middle_operators(i);
                for (int j = 0; j < middle_operator_names[i]->data.size(); j++)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << middle_operator_names[i]->data[j]
                       << " := " << middle_operator_exprs[i]->data[j] << endl;
                ss << endl;
                unload_middle_operators(i);
            }
        }
        if (schemer != nullptr) {
            load_schemer();
            ss << schemer->get_transform_formulas() << endl;
            unload_schemer();
        }
        return ss.str();
    }
    virtual shared_ptr<MPO> scalar_multiply(FL d) {
        string new_tag = ximag(d) == 0 ? Parsing::to_string(xreal(d))
                                       : Parsing::to_string(xreal(d)) + "+" +
                                             Parsing::to_string(ximag(d)) + "I";
        shared_ptr<MPO> rmpo = frame_<FP>()->minimal_memory_usage
                                   ? this->deep_copy(tag + "@" + new_tag)
                                   : make_shared<MPO>(*this);
        assert(rmpo->middle_operator_exprs.size() != 0);
        for (size_t ix = 0; ix < rmpo->middle_operator_exprs.size(); ix++) {
            rmpo->load_middle_operators(ix);
            auto &x = rmpo->middle_operator_exprs[ix];
            x = x->copy();
            for (size_t j = 0; j < x->data.size(); j++)
                x->data[j] = d * x->data[j];
            rmpo->save_middle_operators(ix);
            rmpo->unload_middle_operators(ix);
        }
        rmpo->const_e = (typename const_fl_type<FL>::FL)d * rmpo->const_e;
        return rmpo;
    }
};

template <typename S, typename FL>
inline shared_ptr<MPO<S, FL>> operator*(FL d,
                                        const shared_ptr<MPO<S, FL>> &mpo) {
    return mpo->scalar_multiply(d);
}

template <typename S, typename FL>
inline shared_ptr<MPO<S, FL>> operator*(const shared_ptr<MPO<S, FL>> &mpo,
                                        FL d) {
    return d * mpo;
}

template <typename S, typename FL>
inline shared_ptr<MPO<S, FL>> operator-(const shared_ptr<MPO<S, FL>> &mpo) {
    return (FL)(-1.0) * mpo;
}

// Diagonal part of MPO (will copy the diagonal elements)
// MPO must be unsimplified
template <typename S, typename FL> struct DiagonalMPO : MPO<S, FL> {
    using MPO<S, FL>::n_sites;
    using typename MPO<S, FL>::FP;
    DiagonalMPO(const shared_ptr<MPO<S, FL>> &mpo,
                const shared_ptr<Rule<S, FL>> &rule = nullptr,
                const string &tag = "")
        : MPO<S, FL>(mpo->n_sites, tag == "" ? mpo->tag + "@DIAG" : tag) {
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::left_vacuum = mpo->left_vacuum;
        MPO<S, FL>::npdm_scheme = mpo->npdm_scheme;
        MPO<S, FL>::npdm_parallel_center = mpo->npdm_parallel_center;
        MPO<S, FL>::tf = mpo->tf;
        MPO<S, FL>::basis = mpo->basis;
        MPO<S, FL>::site_op_infos = mpo->site_op_infos;
        MPO<S, FL>::sparse_form = mpo->sparse_form;
        mpo->load_schemer();
        MPO<S, FL>::schemer =
            mpo->schemer == nullptr ? nullptr : mpo->schemer->copy();
        mpo->unload_schemer();
        MPO<S, FL>::save_schemer();
        MPO<S, FL>::unload_schemer();
        MPO<S, FL>::left_operator_names = mpo->left_operator_names;
        for (size_t i = 0; i < MPO<S, FL>::left_operator_names.size(); i++) {
            mpo->load_left_operators(i);
            MPO<S, FL>::left_operator_names[i] = mpo->left_operator_names[i];
            mpo->unload_left_operators(i);
            MPO<S, FL>::save_left_operators(i);
            MPO<S, FL>::unload_left_operators(i);
        }
        MPO<S, FL>::right_operator_names = mpo->right_operator_names;
        for (size_t i = 0; i < MPO<S, FL>::right_operator_names.size(); i++) {
            mpo->load_right_operators(i);
            MPO<S, FL>::right_operator_names[i] = mpo->right_operator_names[i];
            mpo->unload_right_operators(i);
            MPO<S, FL>::save_right_operators(i);
            MPO<S, FL>::unload_right_operators(i);
        }
        assert(mpo->left_operator_exprs.size() == 0);
        assert(mpo->right_operator_exprs.size() == 0);
        shared_ptr<SparseMatrix<S, FL>> zmat =
            make_shared<SparseMatrix<S, FL>>();
        zmat->factor = 0;
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        MPO<S, FL>::tensors.resize(n_sites, nullptr);
        for (int m = 0; m < n_sites; m++) {
            shared_ptr<OperatorTensor<S, FL>> r =
                make_shared<OperatorTensor<S, FL>>();
            mpo->load_tensor(m);
            r->lmat = mpo->tensors[m]->lmat->copy();
            r->rmat = mpo->tensors[m]->rmat->copy();
            r->ops = mpo->tensors[m]->ops;
            mpo->unload_tensor(m);
            MPO<S, FL>::tensors[m] = r;
            for (auto &p : r->ops) {
                OpElement<S, FL> &op =
                    *dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                if (op.q_label != mpo->op->q_label)
                    p.second = zmat;
                else if (p.second->get_type() == SparseMatrixTypes::Normal) {
                    shared_ptr<VectorAllocator<FP>> d_alloc =
                        make_shared<VectorAllocator<FP>>();
                    shared_ptr<SparseMatrix<S, FL>> mat =
                        make_shared<SparseMatrix<S, FL>>(d_alloc);
                    mat->allocate(p.second->info);
                    mat->factor = p.second->factor;
                    if (p.second->info->n == p.second->total_memory) {
                        GMatrix<FL> mmat(mat->data, (MKL_INT)mat->total_memory,
                                         1);
                        GMatrix<FL> pmat(p.second->data,
                                         (MKL_INT)p.second->total_memory, 1);
                        GMatrixFunctions<FL>::copy(mmat, pmat);
                    } else {
                        for (int i = 0; i < mat->info->n; i++) {
                            GMatrix<FL> mmat = (*mat)[i], pmat = (*p.second)[i];
                            mmat.n = pmat.n = 1;
                            GMatrixFunctions<FL>::copy(mmat, pmat, mmat.m + 1,
                                                       pmat.m + 1);
                        }
                    }
                    p.second = mat;
                } else if (p.second->get_type() == SparseMatrixTypes::CSR) {
                    shared_ptr<CSRSparseMatrix<S, FL>> pmat =
                        dynamic_pointer_cast<CSRSparseMatrix<S, FL>>(p.second);
                    shared_ptr<VectorAllocator<FP>> d_alloc =
                        make_shared<VectorAllocator<FP>>();
                    shared_ptr<CSRSparseMatrix<S, FL>> mat =
                        make_shared<CSRSparseMatrix<S, FL>>(d_alloc);
                    mat->initialize(p.second->info);
                    for (int i = 0; i < mat->info->n; i++) {
                        shared_ptr<GCSRMatrix<FL>> cmat = mat->csr_data[i];
                        assert(cmat->m == cmat->n);
                        cmat->nnz = cmat->m;
                        cmat->allocate();
                        GMatrix<FL> dmat(cmat->data, cmat->m, 1);
                        pmat->csr_data[i]->diag(dmat);
                        if (cmat->nnz != cmat->size()) {
                            for (MKL_INT j = 0; j < cmat->m; j++)
                                cmat->rows[j] = j, cmat->cols[j] = j;
                            cmat->rows[cmat->m] = cmat->nnz;
                        }
                    }
                    p.second = mat;
                } else if (p.second->get_type() == SparseMatrixTypes::Delayed)
                    p.second = dynamic_pointer_cast<DelayedSparseMatrix<S, FL>>(
                                   p.second)
                                   ->copy();
                else
                    assert(false);
            }
            if (rule != nullptr) {
                for (auto &p : r->ops) {
                    auto pop = dynamic_pointer_cast<OpElement<S, FL>>(p.first);
                    if (p.second->get_type() == SparseMatrixTypes::Delayed) {
                        auto rop = (*rule)(pop);
                        if (rop != nullptr) {
                            auto ref_op = rop->op;
                            if (r->ops.count(ref_op) &&
                                (r->ops.at(ref_op)->factor == (FL)0.0 ||
                                 r->ops.at(ref_op)->info->n == 0 ||
                                 r->ops.at(ref_op)->norm() < TINY))
                                p.second = zmat;
                        }
                    }
                }
            }
            vector<shared_ptr<Symbolic<S>>> pmats = {r->lmat, r->rmat};
            size_t kk;
            shared_ptr<OpSum<S, FL>> px;
            for (auto pmat : pmats)
                for (auto &x : pmat->data) {
                    shared_ptr<OpExpr<S>> xx;
                    switch (x->get_type()) {
                    case OpTypes::Zero:
                        break;
                    case OpTypes::Elem:
                        xx = abs_value(x);
                        if (r->ops[xx]->factor == (FL)0.0 ||
                            r->ops[xx]->info->n == 0 ||
                            r->ops[xx]->norm() < TINY)
                            x = zero;
                        break;
                    case OpTypes::Sum:
                        kk = 0;
                        px = make_shared<OpSum<S, FL>>(
                            dynamic_pointer_cast<OpSum<S, FL>>(x)->strings);
                        x = px;
                        for (size_t i = 0; i < px->strings.size(); i++) {
                            xx = abs_value((shared_ptr<OpExpr<S>>)px->strings[i]
                                               ->get_op());
                            shared_ptr<SparseMatrix<S, FL>> &mat = r->ops[xx];
                            if (!(mat->factor == (FL)0.0 || mat->info->n == 0 ||
                                  mat->norm() < TINY)) {
                                if (i != kk)
                                    px->strings[kk] = px->strings[i];
                                kk++;
                            }
                        }
                        if (kk == 0)
                            x = zero;
                        else if (kk != px->strings.size())
                            px->strings.resize(kk);
                        break;
                    default:
                        assert(false);
                    }
                }
            for (auto pmat : pmats)
                if (pmat->get_type() == SymTypes::Mat) {
                    shared_ptr<SymbolicMatrix<S>> smat =
                        dynamic_pointer_cast<SymbolicMatrix<S>>(pmat);
                    size_t j = 0;
                    for (size_t i = 0; i < smat->indices.size(); i++)
                        if (smat->data[i]->get_type() != OpTypes::Zero) {
                            if (i != j)
                                smat->data[j] = smat->data[i],
                                smat->indices[j] = smat->indices[i];
                            j++;
                        }
                    smat->data.resize(j);
                    smat->indices.resize(j);
                }
            for (auto it = r->ops.cbegin(); it != r->ops.cend();) {
                if (it->second->factor == (FL)0.0 || it->second->info->n == 0)
                    r->ops.erase(it++);
                else
                    it++;
            }
            MPO<S, FL>::save_tensor(m);
            MPO<S, FL>::unload_tensor(m);
        }
    }
};

// Adding ancilla (identity) sites to a MPO
// n_sites = 2 * n_physical_sites
template <typename S, typename FL> struct AncillaMPO : MPO<S, FL> {
    using typename MPO<S, FL>::FP;
    int n_physical_sites;
    shared_ptr<MPO<S, FL>> prim_mpo;
    AncillaMPO(const shared_ptr<MPO<S, FL>> &mpo, bool npdm = false,
               bool trace_right = true, const string &tag = "")
        : n_physical_sites(mpo->n_sites),
          prim_mpo(mpo), MPO<S, FL>(mpo->n_sites << 1,
                                    tag == "" ? mpo->tag + "@ANC" : tag) {
        const auto n_sites = MPO<S, FL>::n_sites;
        const shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), S());
        MPO<S, FL>::hamil = mpo->hamil;
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::left_vacuum = mpo->left_vacuum;
        MPO<S, FL>::npdm_scheme = mpo->npdm_scheme;
        MPO<S, FL>::npdm_parallel_center = mpo->npdm_parallel_center;
        MPO<S, FL>::tf = mpo->tf;
        MPO<S, FL>::site_op_infos =
            vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(n_sites);
        MPO<S, FL>::sparse_form = string(n_sites, 'N');
        for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
            MPO<S, FL>::site_op_infos[j] = mpo->site_op_infos[i];
            MPO<S, FL>::site_op_infos[j + 1] = mpo->site_op_infos[i];
            MPO<S, FL>::sparse_form[trace_right ? j : j + 1] =
                MPO<S, FL>::sparse_form[i];
        }
        // operator names
        MPO<S, FL>::left_operator_names.resize(n_sites, nullptr);
        MPO<S, FL>::right_operator_names.resize(n_sites, nullptr);
        if (trace_right) {
            for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
                mpo->load_left_operators(i);
                MPO<S, FL>::left_operator_names[j] =
                    mpo->left_operator_names[i];
                mpo->unload_left_operators(i);
                MPO<S, FL>::save_left_operators(j);
                MPO<S, FL>::left_operator_names[j + 1] =
                    MPO<S, FL>::left_operator_names[j]->copy();
                MPO<S, FL>::save_left_operators(j + 1);
                MPO<S, FL>::unload_left_operators(j + 1);
                MPO<S, FL>::unload_left_operators(j);
                mpo->load_right_operators(i);
                MPO<S, FL>::right_operator_names[j] =
                    mpo->right_operator_names[i];
                mpo->unload_right_operators(i);
                MPO<S, FL>::save_right_operators(j);
                if (j - 1 >= 0) {
                    MPO<S, FL>::right_operator_names[j - 1] =
                        MPO<S, FL>::right_operator_names[j]->copy();
                    MPO<S, FL>::save_right_operators(j - 1);
                    MPO<S, FL>::unload_right_operators(j - 1);
                }
                MPO<S, FL>::unload_right_operators(j);
            }
            MPO<S, FL>::right_operator_names[n_sites - 1] =
                make_shared<SymbolicColumnVector<S>>(1);
            MPO<S, FL>::right_operator_names[n_sites - 1]->data[0] = i_op;
            MPO<S, FL>::save_right_operators(n_sites - 1);
            MPO<S, FL>::unload_right_operators(n_sites - 1);
        } else {
            for (int i = n_physical_sites - 1, j = n_sites - 2; i >= 0;
                 i--, j -= 2) {
                mpo->load_right_operators(i);
                MPO<S, FL>::right_operator_names[j + 1] =
                    mpo->right_operator_names[i];
                mpo->unload_right_operators(i);
                MPO<S, FL>::save_right_operators(j + 1);
                MPO<S, FL>::right_operator_names[j] =
                    MPO<S, FL>::right_operator_names[j + 1]->copy();
                MPO<S, FL>::save_right_operators(j);
                MPO<S, FL>::unload_right_operators(j);
                MPO<S, FL>::unload_right_operators(j + 1);
                mpo->load_left_operators(i);
                MPO<S, FL>::left_operator_names[j + 1] =
                    mpo->left_operator_names[i];
                mpo->unload_left_operators(i);
                MPO<S, FL>::save_left_operators(j + 1);
                if (j + 2 < n_sites) {
                    MPO<S, FL>::left_operator_names[j + 2] =
                        MPO<S, FL>::left_operator_names[j + 1]->copy();
                    MPO<S, FL>::save_left_operators(j + 2);
                    MPO<S, FL>::unload_left_operators(j + 2);
                }
                MPO<S, FL>::unload_left_operators(j + 1);
            }
            MPO<S, FL>::left_operator_names[0] =
                make_shared<SymbolicRowVector<S>>(1);
            MPO<S, FL>::left_operator_names[0]->data[0] = i_op;
            MPO<S, FL>::save_left_operators(0);
            MPO<S, FL>::unload_left_operators(0);
        }
        // middle operators
        if (mpo->middle_operator_names.size() != 0) {
            assert(mpo->schemer == nullptr);
            MPO<S, FL>::middle_operator_names.resize(n_sites - 1);
            MPO<S, FL>::middle_operator_exprs.resize(n_sites - 1);
            shared_ptr<SymbolicColumnVector<S>> zero_mat =
                make_shared<SymbolicColumnVector<S>>(1);
            (*zero_mat)[0] =
                make_shared<OpElement<S, FL>>(OpNames::Zero, SiteIndex(), S());
            shared_ptr<SymbolicColumnVector<S>> zero_expr =
                make_shared<SymbolicColumnVector<S>>(1);
            (*zero_expr)[0] = make_shared<OpExpr<S>>();
            for (int i = 0, j = trace_right ? 0 : 1; i < n_physical_sites - 1;
                 i++, j += 2) {
                mpo->load_middle_operators(i);
                MPO<S, FL>::middle_operator_names[j] =
                    mpo->middle_operator_names[i];
                MPO<S, FL>::middle_operator_exprs[j] =
                    mpo->middle_operator_exprs[i];
                if (!npdm) {
                    MPO<S, FL>::middle_operator_names[j + 1] =
                        mpo->middle_operator_names[i];
                    MPO<S, FL>::middle_operator_exprs[j + 1] =
                        mpo->middle_operator_exprs[i];
                } else {
                    MPO<S, FL>::middle_operator_names[j + 1] = zero_mat;
                    MPO<S, FL>::middle_operator_exprs[j + 1] = zero_expr;
                }
                mpo->unload_middle_operators(i);
                MPO<S, FL>::save_middle_operators(j);
                MPO<S, FL>::unload_middle_operators(j);
                MPO<S, FL>::save_middle_operators(j + 1);
                MPO<S, FL>::unload_middle_operators(j + 1);
            }
            if (trace_right) {
                if (mpo->op != nullptr && mpo->op->name != OpNames::Zero) {
                    shared_ptr<SymbolicColumnVector<S>> hop_mat =
                        make_shared<SymbolicColumnVector<S>>(1);
                    (*hop_mat)[0] = mpo->op;
                    shared_ptr<SymbolicColumnVector<S>> hop_expr =
                        make_shared<SymbolicColumnVector<S>>(1);
                    (*hop_expr)[0] = (shared_ptr<OpExpr<S>>)mpo->op * i_op;
                    MPO<S, FL>::middle_operator_names[n_sites - 2] = hop_mat;
                    MPO<S, FL>::middle_operator_exprs[n_sites - 2] = hop_expr;
                } else {
                    MPO<S, FL>::middle_operator_names[n_sites - 2] = zero_mat;
                    MPO<S, FL>::middle_operator_exprs[n_sites - 2] = zero_expr;
                }
                MPO<S, FL>::save_middle_operators(n_sites - 2);
                MPO<S, FL>::unload_middle_operators(n_sites - 2);
            } else {
                if (mpo->op != nullptr && mpo->op->name != OpNames::Zero) {
                    shared_ptr<SymbolicRowVector<S>> hop_mat =
                        make_shared<SymbolicRowVector<S>>(1);
                    (*hop_mat)[0] = mpo->op;
                    shared_ptr<SymbolicRowVector<S>> hop_expr =
                        make_shared<SymbolicRowVector<S>>(1);
                    (*hop_expr)[0] = i_op * (shared_ptr<OpExpr<S>>)mpo->op;
                    MPO<S, FL>::middle_operator_names[0] = hop_mat;
                    MPO<S, FL>::middle_operator_exprs[0] = hop_expr;
                } else {
                    MPO<S, FL>::middle_operator_names[0] = zero_mat;
                    MPO<S, FL>::middle_operator_exprs[0] = zero_expr;
                }
                MPO<S, FL>::save_middle_operators(0);
                MPO<S, FL>::unload_middle_operators(0);
            }
        }
        // operator tensors
        MPO<S, FL>::tensors.resize(n_sites, nullptr);
        for (int i = 0, j = trace_right ? 0 : 1; i < n_physical_sites;
             i++, j += 2) {
            mpo->load_tensor(i);
            if (j + 1 < n_sites - 1) {
                MPO<S, FL>::tensors[j] = mpo->tensors[i];
                int rshape = MPO<S, FL>::tensors[j]->lmat->n;
                MPO<S, FL>::tensors[j + 1] =
                    make_shared<OperatorTensor<S, FL>>();
                MPO<S, FL>::tensors[j + 1]->lmat =
                    MPO<S, FL>::tensors[j + 1]->rmat =
                        make_shared<SymbolicMatrix<S>>(rshape, rshape);
                for (int k = 0; k < rshape; k++)
                    (*MPO<S, FL>::tensors[j + 1]->lmat)[{k, k}] = i_op;
                if (mpo->tensors[i]->lmat != mpo->tensors[i]->rmat &&
                    !(mpo->schemer != nullptr &&
                      mpo->schemer->right_trans_site -
                              mpo->schemer->left_trans_site ==
                          2)) {
                    int lshape = mpo->tensors[i + 1]->rmat->m;
                    MPO<S, FL>::tensors[j + 1]->rmat =
                        make_shared<SymbolicMatrix<S>>(lshape, lshape);
                    for (int k = 0; k < lshape; k++)
                        (*MPO<S, FL>::tensors[j + 1]->rmat)[{k, k}] = i_op;
                }
                MPO<S, FL>::save_tensor(j + 1);
                MPO<S, FL>::unload_tensor(j + 1);
            } else if (j == n_sites - 2) {
                int lshape = mpo->tensors[i]->lmat->m;
                MPO<S, FL>::tensors[j] = make_shared<OperatorTensor<S, FL>>();
                MPO<S, FL>::tensors[j]->lmat = MPO<S, FL>::tensors[j]->rmat =
                    make_shared<SymbolicMatrix<S>>(lshape, 1);
                for (int k = 0; k < lshape; k++)
                    (*MPO<S, FL>::tensors[j]->lmat)[{k, 0}] =
                        mpo->tensors[i]->lmat->data[k];
                if (mpo->tensors[i]->lmat != mpo->tensors[i]->rmat) {
                    lshape = mpo->tensors[i]->rmat->m;
                    MPO<S, FL>::tensors[j]->rmat =
                        make_shared<SymbolicMatrix<S>>(lshape, 1);
                    for (int k = 0; k < lshape; k++)
                        (*MPO<S, FL>::tensors[j]->rmat)[{k, 0}] =
                            mpo->tensors[i]->rmat->data[k];
                }
                MPO<S, FL>::tensors[j]->ops = mpo->tensors[i]->ops;
                MPO<S, FL>::tensors[j + 1] =
                    make_shared<OperatorTensor<S, FL>>();
                MPO<S, FL>::tensors[j + 1]->lmat =
                    MPO<S, FL>::tensors[j + 1]->rmat =
                        make_shared<SymbolicColumnVector<S>>(1);
                MPO<S, FL>::tensors[j + 1]->lmat->data[0] = i_op;
                MPO<S, FL>::save_tensor(j + 1);
                MPO<S, FL>::unload_tensor(j + 1);
            } else {
                MPO<S, FL>::tensors[j] = mpo->tensors[i];
                mpo->load_tensor(0);
                MPO<S, FL>::tensors[0] = make_shared<OperatorTensor<S, FL>>();
                MPO<S, FL>::tensors[0]->lmat = MPO<S, FL>::tensors[0]->rmat =
                    make_shared<SymbolicRowVector<S>>(1);
                MPO<S, FL>::tensors[0]->lmat->data[0] = i_op;
                MPO<S, FL>::tensors[0]->ops[i_op] =
                    MPO<S, FL>::tensors[1]->ops.at(i_op);
                int rshape = mpo->tensors[0]->lmat->n;
                MPO<S, FL>::tensors[1] = make_shared<OperatorTensor<S, FL>>();
                MPO<S, FL>::tensors[1]->lmat = MPO<S, FL>::tensors[1]->rmat =
                    make_shared<SymbolicMatrix<S>>(1, rshape);
                for (int k = 0; k < rshape; k++)
                    (*MPO<S, FL>::tensors[1]->lmat)[{0, k}] =
                        mpo->tensors[0]->lmat->data[k];
                if (mpo->tensors[0]->lmat != mpo->tensors[0]->rmat) {
                    rshape = mpo->tensors[0]->rmat->n;
                    MPO<S, FL>::tensors[1]->rmat =
                        make_shared<SymbolicMatrix<S>>(1, rshape);
                    for (int k = 0; k < rshape; k++)
                        (*MPO<S, FL>::tensors[1]->rmat)[{0, k}] =
                            mpo->tensors[0]->rmat->data[k];
                }
                MPO<S, FL>::tensors[1]->ops = mpo->tensors[0]->ops;
                mpo->unload_tensor(0);
                MPO<S, FL>::save_tensor(0);
                MPO<S, FL>::unload_tensor(0);
                MPO<S, FL>::save_tensor(1);
                MPO<S, FL>::unload_tensor(1);
            }
            if (trace_right) {
                MPO<S, FL>::load_tensor(j + 1);
                MPO<S, FL>::tensors[j + 1]->ops[i_op] =
                    MPO<S, FL>::tensors[j]->ops.at(i_op);
                MPO<S, FL>::save_tensor(j + 1);
                MPO<S, FL>::unload_tensor(j + 1);
            } else if (j - 1 != 0) {
                MPO<S, FL>::load_tensor(j - 1);
                MPO<S, FL>::tensors[j - 1]->ops[i_op] =
                    MPO<S, FL>::tensors[j]->ops.at(i_op);
                MPO<S, FL>::save_tensor(j - 1);
                MPO<S, FL>::unload_tensor(j - 1);
            }
            MPO<S, FL>::save_tensor(j);
            MPO<S, FL>::unload_tensor(j);
            mpo->unload_tensor(i);
        }
        // numerical transform
        if (mpo->schemer != nullptr &&
            mpo->schemer->right_trans_site - mpo->schemer->left_trans_site ==
                2) {
            mpo->load_schemer();
            MPO<S, FL>::schemer =
                frame_<FP>()->minimal_memory_usage
                    ? make_shared<MPOSchemer<S>>(*mpo->schemer)
                    : mpo->schemer->copy();
            mpo->unload_schemer();
            if (trace_right) {
                if (n_physical_sites & 1) {
                    MPO<S, FL>::schemer->left_trans_site = n_physical_sites - 2;
                    MPO<S, FL>::schemer->right_trans_site = n_physical_sites;
                } else {
                    MPO<S, FL>::schemer->left_trans_site = n_physical_sites - 1;
                    MPO<S, FL>::schemer->right_trans_site =
                        n_physical_sites + 1;
                }
            } else {
                if (n_physical_sites & 1) {
                    MPO<S, FL>::schemer->left_trans_site = n_physical_sites - 1;
                    MPO<S, FL>::schemer->right_trans_site =
                        n_physical_sites + 1;
                } else {
                    MPO<S, FL>::schemer->left_trans_site = n_physical_sites;
                    MPO<S, FL>::schemer->right_trans_site =
                        n_physical_sites + 2;
                }
            }
        } else if (mpo->schemer != nullptr)
            assert(false);
        else
            MPO<S, FL>::schemer = nullptr;
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

// Add identity operator to MPO (will not change expression of MPO)
// MPO must be simplified
template <typename S, typename FL> struct IdentityAddedMPO : MPO<S, FL> {
    using typename MPO<S, FL>::FP;
    using MPO<S, FL>::n_sites;
    IdentityAddedMPO(const shared_ptr<MPO<S, FL>> &mpo, const string &tag = "")
        : MPO<S, FL>(mpo->n_sites, tag == "" ? mpo->tag : tag) {
        MPO<S, FL>::hamil = mpo->hamil;
        MPO<S, FL>::const_e = mpo->const_e;
        MPO<S, FL>::op = mpo->op;
        MPO<S, FL>::left_vacuum = mpo->left_vacuum;
        MPO<S, FL>::npdm_scheme = mpo->npdm_scheme;
        MPO<S, FL>::npdm_parallel_center = mpo->npdm_parallel_center;
        MPO<S, FL>::tf = mpo->tf;
        MPO<S, FL>::basis = mpo->basis;
        MPO<S, FL>::site_op_infos = mpo->site_op_infos;
        MPO<S, FL>::sparse_form = mpo->sparse_form;
        if (mpo->schemer == nullptr)
            MPO<S, FL>::schemer = nullptr;
        else {
            mpo->load_schemer();
            MPO<S, FL>::schemer =
                frame_<FP>()->minimal_memory_usage
                    ? make_shared<MPOSchemer<S>>(*mpo->schemer)
                    : mpo->schemer->copy();
            mpo->unload_schemer();
        }
        MPO<S, FL>::left_operator_names = mpo->left_operator_names;
        MPO<S, FL>::right_operator_names = mpo->right_operator_names;
        MPO<S, FL>::middle_operator_names = mpo->middle_operator_names;
        MPO<S, FL>::left_operator_exprs = mpo->left_operator_exprs;
        MPO<S, FL>::right_operator_exprs = mpo->right_operator_exprs;
        MPO<S, FL>::middle_operator_exprs = mpo->middle_operator_exprs;
        assert(mpo->left_operator_exprs.size() != 0);
        assert(mpo->right_operator_exprs.size() != 0);
        MPO<S, FL>::tensors = mpo->tensors;
        if (MPO<S, FL>::tag != mpo->tag) {
            for (size_t i = 0; i < MPO<S, FL>::middle_operator_names.size();
                 i++) {
                mpo->load_middle_operators(i);
                MPO<S, FL>::middle_operator_names[i] =
                    mpo->middle_operator_names[i];
                MPO<S, FL>::middle_operator_exprs[i] =
                    mpo->middle_operator_exprs[i];
                mpo->unload_middle_operators(i);
                MPO<S, FL>::save_middle_operators(i);
                MPO<S, FL>::unload_middle_operators(i);
            }
            for (int i = 0; i < MPO<S, FL>::n_sites; i++) {
                mpo->load_tensor(i);
                MPO<S, FL>::tensors[i] = mpo->tensors[i];
                mpo->unload_tensor(i);
                MPO<S, FL>::save_tensor(i);
                MPO<S, FL>::unload_tensor(i);
            }
        }
        const shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S, FL>>(OpNames::I, SiteIndex(), S());
        const shared_ptr<OpExpr<S>> i_op_lv = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), mpo->left_vacuum);
        for (size_t m = 0; m < MPO<S, FL>::left_operator_names.size(); m++) {
            bool found = false;
            auto &x = MPO<S, FL>::left_operator_names[m];
            auto &y = MPO<S, FL>::left_operator_exprs[m];
            mpo->load_left_operators(m);
            x = mpo->left_operator_names[m]->copy();
            y = mpo->left_operator_exprs[m]->copy();
            mpo->unload_left_operators(m);
            if (y->get_type() != x->get_type()) {
                y = make_shared<SymbolicRowVector<S>>(x->n);
                y->data = MPO<S, FL>::left_operator_exprs[m]->data;
            }
            for (size_t j = 0; j < x->data.size(); j++)
                if (x->data[j] == i_op) {
                    found = true;
                    break;
                }
            if (!found) {
                x->data.push_back(i_op_lv);
                y->data.push_back(m == 0 ? i_op : i_op_lv * i_op);
                assert(x->get_type() == SymTypes::RVec);
                x->n = y->n = (int)x->data.size();
                if (m == 0) {
                    auto &z = MPO<S, FL>::tensors[m]->lmat;
                    mpo->load_tensor(m);
                    z = mpo->tensors[m]->lmat->copy();
                    mpo->unload_tensor(m);
                    z->data.push_back(i_op);
                    z->n = (int)z->data.size();
                }
            }
            MPO<S, FL>::save_left_operators(m);
            MPO<S, FL>::unload_left_operators(m);
        }
        for (size_t m = 0; m < MPO<S, FL>::right_operator_names.size(); m++) {
            bool found = false;
            auto &x = MPO<S, FL>::right_operator_names[m];
            auto &y = MPO<S, FL>::right_operator_exprs[m];
            mpo->load_right_operators(m);
            x = mpo->right_operator_names[m]->copy();
            y = mpo->right_operator_exprs[m]->copy();
            mpo->unload_right_operators(m);
            if (y->get_type() != x->get_type()) {
                y = make_shared<SymbolicColumnVector<S>>(x->m);
                y->data = MPO<S, FL>::right_operator_exprs[m]->data;
            }
            for (size_t j = 0; j < x->data.size(); j++)
                if (x->data[j] == i_op) {
                    found = true;
                    break;
                }
            if (!found) {
                x->data.push_back(i_op);
                y->data.push_back(m == MPO<S, FL>::right_operator_names.size() -
                                              1
                                      ? i_op
                                      : i_op * i_op);
                assert(x->get_type() == SymTypes::CVec);
                x->m = y->m = (int)x->data.size();
                if (m == mpo->n_sites - 1) {
                    auto &z = MPO<S, FL>::tensors[m]->rmat;
                    mpo->load_tensor(m);
                    z = mpo->tensors[m]->rmat->copy();
                    mpo->unload_tensor(m);
                    z->data.push_back(i_op);
                    z->m = (int)z->data.size();
                }
            }
            MPO<S, FL>::save_right_operators(m);
            MPO<S, FL>::unload_right_operators(m);
        }
    }
};

} // namespace block2
