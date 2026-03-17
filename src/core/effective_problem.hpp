#pragma once

#include "operator_tensor.hpp"
#include "symmetry.hpp"
#include <array>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace block2 {

template <typename S, typename FL> struct EffectiveProblem {
    enum struct SymmetryTag : uint8_t {
        SU2 = 1,
        SZ = 2,
        SU2K = 3,
        SZK = 4,
        SGF = 5,
        SGB = 6,
        SAny = 7
    };
    enum struct TensorTag : uint8_t { Normal = 0, Delayed = 1 };

    static const std::array<char, 8> &magic() {
        static const std::array<char, 8> x = {
            {'E', 'F', 'F', 'P', 'R', 'O', 'B', '1'}};
        return x;
    }

    static uint32_t version() { return 2; }

    static bool supports_symmetry() {
        return std::is_same<S, SU2Short>::value ||
               std::is_same<S, SU2Long>::value ||
               std::is_same<S, SU2LongLong>::value ||
               std::is_same<S, SU2KLong>::value ||
               std::is_same<S, SZShort>::value ||
               std::is_same<S, SZLong>::value ||
               std::is_same<S, SZLongLong>::value ||
               std::is_same<S, SZKLong>::value ||
               std::is_same<S, SGF>::value || std::is_same<S, SGB>::value ||
               std::is_same<S, SAny>::value;
    }

    static SymmetryTag symmetry_tag() {
        if (std::is_same<S, SU2Short>::value || std::is_same<S, SU2Long>::value ||
            std::is_same<S, SU2LongLong>::value)
            return SymmetryTag::SU2;
        else if (std::is_same<S, SZShort>::value ||
                 std::is_same<S, SZLong>::value ||
                 std::is_same<S, SZLongLong>::value)
            return SymmetryTag::SZ;
        else if (std::is_same<S, SU2KLong>::value)
            return SymmetryTag::SU2K;
        else if (std::is_same<S, SZKLong>::value)
            return SymmetryTag::SZK;
        else if (std::is_same<S, SGF>::value)
            return SymmetryTag::SGF;
        else if (std::is_same<S, SGB>::value)
            return SymmetryTag::SGB;
        else if (std::is_same<S, SAny>::value)
            return SymmetryTag::SAny;
        else
            throw std::runtime_error("EFF-PROB: unsupported symmetry.");
    }

    template <typename T> static void write_value(std::ostream &ofs,
                                                  const T &value) {
        ofs.write((const char *)&value, sizeof(value));
    }

    static void save_problem_header(std::ostream &ofs, int isite,
                                    bool forward) {
        if (!supports_symmetry())
            throw std::runtime_error("EFF-PROB: unsupported symmetry.");
        ofs.write(magic().data(), magic().size());
        write_value(ofs, version());
        SymmetryTag symm = symmetry_tag();
        write_value(ofs, symm);
        const uint8_t q_size = (uint8_t)sizeof(S);
        const uint8_t fl_size = (uint8_t)sizeof(FL);
        const uint8_t fp_size = (uint8_t)sizeof(typename GMatrix<FL>::FP);
        const uint8_t cpx_sz = (uint8_t)SparseMatrix<S, FL>::cpx_sz;
        const uint8_t ubond_size = (uint8_t)sizeof(ubond_t);
        const uint8_t align_type = (uint8_t)threading->align_type;
        write_value(ofs, q_size);
        write_value(ofs, fl_size);
        write_value(ofs, fp_size);
        write_value(ofs, cpx_sz);
        write_value(ofs, ubond_size);
        write_value(ofs, align_type);
        write_value(ofs, isite);
        write_value(ofs, forward);
    }

    static void save_connection_info(
        const std::shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>
            &cinfo,
        std::ostream &ofs) {
        assert(cinfo != nullptr);
        ofs.write((const char *)cinfo->n, sizeof(cinfo->n));
        write_value(ofs, cinfo->nc);
        if (cinfo->n[4] != 0) {
            ofs.write((const char *)cinfo->quanta, sizeof(S) * cinfo->n[4]);
            ofs.write((const char *)cinfo->idx,
                      sizeof(uint32_t) * cinfo->n[4]);
        }
        if (cinfo->nc != 0) {
            ofs.write((const char *)cinfo->stride,
                      sizeof(uint64_t) * cinfo->nc);
            ofs.write((const char *)cinfo->factor,
                      sizeof(double) * cinfo->nc);
            ofs.write((const char *)cinfo->ia, sizeof(uint32_t) * cinfo->nc);
            ofs.write((const char *)cinfo->ib, sizeof(uint32_t) * cinfo->nc);
            ofs.write((const char *)cinfo->ic, sizeof(uint32_t) * cinfo->nc);
        }
    }

    static void save_sparse_matrix_metadata(
        const std::shared_ptr<SparseMatrix<S, FL>> &mat, std::ostream &ofs) {
        const bool present = mat != nullptr;
        write_value(ofs, present);
        if (!present)
            return;
        SparseMatrixTypes type = mat->get_type();
        write_value(ofs, type);
        mat->info->save_data(ofs);
        write_value(ofs, mat->factor);
        write_value(ofs, mat->total_memory);
    }

    static void save_operator_matrix(
        const std::shared_ptr<SparseMatrix<S, FL>> &mat, std::ostream &ofs) {
        assert(mat != nullptr && mat->info != nullptr);
        SparseMatrixTypes type = mat->get_type();
        write_value(ofs, type);
        mat->info->save_data(ofs);
        const bool has_cinfo = mat->info->cinfo != nullptr;
        write_value(ofs, has_cinfo);
        if (has_cinfo)
            save_connection_info(mat->info->cinfo, ofs);
        if (type == SparseMatrixTypes::Normal)
            write_value(ofs, mat->total_memory);
        else
            mat->save_data(ofs);
    }

    static void save_operator_tensor_payload(
        const std::shared_ptr<OperatorTensor<S, FL>> &opt,
        std::ostream &ofs) {
        uint8_t lr =
            opt->lmat == opt->rmat
                ? (opt->lmat == nullptr ? 4 : 1)
                : (opt->rmat == nullptr ? 2 : (opt->lmat == nullptr ? 3 : 0));
        write_value(ofs, lr);
        if (lr == 1 || lr == 2)
            save_symbolic(opt->lmat, ofs);
        else if (lr == 3)
            save_symbolic(opt->rmat, ofs);
        else if (lr == 0) {
            save_symbolic(opt->lmat, ofs);
            save_symbolic(opt->rmat, ofs);
        }
        int sz = (int)opt->ops.size();
        write_value(ofs, sz);
        for (auto &op : opt->ops) {
            save_expr(op.first, ofs);
            save_operator_matrix(op.second, ofs);
        }
    }

    static void save_operator_tensor(
        const std::shared_ptr<OperatorTensor<S, FL>> &opt,
        std::ostream &ofs) {
        const bool present = opt != nullptr;
        write_value(ofs, present);
        if (!present)
            return;
        TensorTag type = opt->get_type() == OperatorTensorTypes::Delayed
                             ? TensorTag::Delayed
                             : TensorTag::Normal;
        write_value(ofs, type);
        save_operator_tensor_payload(opt, ofs);
        if (type == TensorTag::Delayed) {
            std::shared_ptr<DelayedOperatorTensor<S, FL>> dopt =
                std::dynamic_pointer_cast<DelayedOperatorTensor<S, FL>>(opt);
            assert(dopt != nullptr);
            int ndops = (int)dopt->dops.size();
            write_value(ofs, ndops);
            for (int i = 0; i < ndops; i++)
                save_expr<S>(dopt->dops[i], ofs);
            const bool has_mat = dopt->mat != nullptr;
            const bool has_stacked = dopt->stacked_mat != nullptr;
            write_value(ofs, has_mat);
            if (has_mat)
                save_symbolic(dopt->mat, ofs);
            write_value(ofs, has_stacked);
            if (has_stacked)
                save_symbolic(dopt->stacked_mat, ofs);
            save_operator_tensor(dopt->lopt, ofs);
            save_operator_tensor(dopt->ropt, ofs);
            int nexprs = (int)dopt->exprs.size();
            write_value(ofs, nexprs);
            for (auto &expr : dopt->exprs) {
                save_expr<S>(expr.first, ofs);
                write_value(ofs, expr.second.first);
                write_value(ofs, expr.second.second);
            }
        }
    }

    static void save_problem_file(
        const std::string &filename,
        const std::shared_ptr<DelayedOperatorTensor<S, FL>> &op,
        const std::shared_ptr<SparseMatrix<S, FL>> &cmat,
        const std::shared_ptr<SparseMatrix<S, FL>> &vmat, S dq0,
        const std::vector<S> &operator_quanta,
        const std::vector<
            std::shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>
            &wfn_infos,
        int isite, bool forward) {
        if (operator_quanta.size() != wfn_infos.size())
            throw std::runtime_error(
                "EFF-PROB: operator_quanta/wfn_infos mismatch.");
        std::ofstream ofs(filename.c_str(), std::ios::binary);
        if (!ofs.good())
            throw std::runtime_error("EFF-PROB save failed on '" + filename +
                                     "'.");
        save_problem_header(ofs, isite, forward);
        save_operator_tensor(op, ofs);
        save_sparse_matrix_metadata(cmat, ofs);
        save_sparse_matrix_metadata(vmat, ofs);
        write_value(ofs, dq0);
        int nterms = (int)operator_quanta.size();
        write_value(ofs, nterms);
        for (int i = 0; i < nterms; i++) {
            write_value(ofs, operator_quanta[i]);
            const bool has_wfn = wfn_infos[i] != nullptr;
            write_value(ofs, has_wfn);
            if (has_wfn)
                save_connection_info(wfn_infos[i], ofs);
        }
        if (!ofs.good())
            throw std::runtime_error("EFF-PROB save failed on '" + filename +
                                     "'.");
    }
};

} // namespace block2
