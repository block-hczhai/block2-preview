
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

#include "allocator.hpp"
#include "complex_matrix_functions.hpp"
#include "csr_matrix.hpp"
#include "matrix.hpp"
#include "matrix_functions.hpp"
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <typeinfo>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

#ifdef _HAS_INTEL_MKL

template <typename FL>
inline sparse_status_t
mkl_sparse_x_create_csr(sparse_matrix_t *A, const sparse_index_base_t indexing,
                        const MKL_INT rows, const MKL_INT cols,
                        MKL_INT *rows_start, MKL_INT *rows_end,
                        MKL_INT *col_indx, FL *values);

template <>
inline sparse_status_t mkl_sparse_x_create_csr<double>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, double *values) {
    return mkl_sparse_d_create_csr(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_create_csr<complex<double>>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, complex<double> *values) {
    return mkl_sparse_z_create_csr(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_create_csr<float>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, float *values) {
    return mkl_sparse_s_create_csr(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_create_csr<complex<float>>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, complex<float> *values) {
    return mkl_sparse_c_create_csr(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <typename FL>
inline sparse_status_t
mkl_sparse_x_create_csc(sparse_matrix_t *A, const sparse_index_base_t indexing,
                        const MKL_INT rows, const MKL_INT cols,
                        MKL_INT *rows_start, MKL_INT *rows_end,
                        MKL_INT *col_indx, FL *values);

template <>
inline sparse_status_t mkl_sparse_x_create_csc<double>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, double *values) {
    return mkl_sparse_d_create_csc(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_create_csc<complex<double>>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, complex<double> *values) {
    return mkl_sparse_z_create_csc(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_create_csc<float>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, float *values) {
    return mkl_sparse_s_create_csc(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_create_csc<complex<float>>(
    sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows,
    const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,
    MKL_INT *col_indx, complex<float> *values) {
    return mkl_sparse_c_create_csc(A, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <typename FL>
inline sparse_status_t
mkl_sparse_x_export_csr(const sparse_matrix_t source,
                        sparse_index_base_t *indexing, MKL_INT *rows,
                        MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end,
                        MKL_INT **col_indx, FL **values);

template <>
inline sparse_status_t mkl_sparse_x_export_csr<double>(
    const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows,
    MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end, MKL_INT **col_indx,
    double **values) {
    return mkl_sparse_d_export_csr(source, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_export_csr<complex<double>>(
    const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows,
    MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end, MKL_INT **col_indx,
    complex<double> **values) {
    return mkl_sparse_z_export_csr(source, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_export_csr<float>(
    const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows,
    MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end, MKL_INT **col_indx,
    float **values) {
    return mkl_sparse_s_export_csr(source, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <>
inline sparse_status_t mkl_sparse_x_export_csr<complex<float>>(
    const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows,
    MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end, MKL_INT **col_indx,
    complex<float> **values) {
    return mkl_sparse_c_export_csr(source, indexing, rows, cols, rows_start,
                                   rows_end, col_indx, values);
}

template <typename FL>
inline sparse_status_t
mkl_sparse_x_add(const sparse_operation_t operation, const sparse_matrix_t A,
                 const FL alpha, const sparse_matrix_t B, sparse_matrix_t *C);

template <>
inline sparse_status_t
mkl_sparse_x_add<double>(const sparse_operation_t operation,
                         const sparse_matrix_t A, const double alpha,
                         const sparse_matrix_t B, sparse_matrix_t *C) {
    return mkl_sparse_d_add(operation, A, alpha, B, C);
}

template <>
inline sparse_status_t mkl_sparse_x_add<complex<double>>(
    const sparse_operation_t operation, const sparse_matrix_t A,
    const complex<double> alpha, const sparse_matrix_t B, sparse_matrix_t *C) {
    return mkl_sparse_z_add(operation, A, alpha, B, C);
}

template <>
inline sparse_status_t
mkl_sparse_x_add<float>(const sparse_operation_t operation,
                        const sparse_matrix_t A, const float alpha,
                        const sparse_matrix_t B, sparse_matrix_t *C) {
    return mkl_sparse_s_add(operation, A, alpha, B, C);
}

template <>
inline sparse_status_t mkl_sparse_x_add<complex<float>>(
    const sparse_operation_t operation, const sparse_matrix_t A,
    const complex<float> alpha, const sparse_matrix_t B, sparse_matrix_t *C) {
    return mkl_sparse_c_add(operation, A, alpha, B, C);
}

template <typename FL>
inline sparse_status_t
mkl_sparse_x_mm(const sparse_operation_t operation, const FL alpha,
                const sparse_matrix_t A, const struct matrix_descr descr,
                const sparse_layout_t layout, const FL *x,
                const MKL_INT columns, const MKL_INT ldx, const FL beta, FL *y,
                const MKL_INT ldy);

template <>
inline sparse_status_t mkl_sparse_x_mm<double>(
    const sparse_operation_t operation, const double alpha,
    const sparse_matrix_t A, const struct matrix_descr descr,
    const sparse_layout_t layout, const double *x, const MKL_INT columns,
    const MKL_INT ldx, const double beta, double *y, const MKL_INT ldy) {
    return mkl_sparse_d_mm(operation, alpha, A, descr, layout, x, columns, ldx,
                           beta, y, ldy);
}

template <>
inline sparse_status_t mkl_sparse_x_mm<complex<double>>(
    const sparse_operation_t operation, const complex<double> alpha,
    const sparse_matrix_t A, const struct matrix_descr descr,
    const sparse_layout_t layout, const complex<double> *x,
    const MKL_INT columns, const MKL_INT ldx, const complex<double> beta,
    complex<double> *y, const MKL_INT ldy) {
    return mkl_sparse_z_mm(operation, alpha, A, descr, layout, x, columns, ldx,
                           beta, y, ldy);
}

template <>
inline sparse_status_t
mkl_sparse_x_mm<float>(const sparse_operation_t operation, const float alpha,
                       const sparse_matrix_t A, const struct matrix_descr descr,
                       const sparse_layout_t layout, const float *x,
                       const MKL_INT columns, const MKL_INT ldx,
                       const float beta, float *y, const MKL_INT ldy) {
    return mkl_sparse_s_mm(operation, alpha, A, descr, layout, x, columns, ldx,
                           beta, y, ldy);
}

template <>
inline sparse_status_t mkl_sparse_x_mm<complex<float>>(
    const sparse_operation_t operation, const complex<float> alpha,
    const sparse_matrix_t A, const struct matrix_descr descr,
    const sparse_layout_t layout, const complex<float> *x,
    const MKL_INT columns, const MKL_INT ldx, const complex<float> beta,
    complex<float> *y, const MKL_INT ldy) {
    return mkl_sparse_c_mm(operation, alpha, A, descr, layout, x, columns, ldx,
                           beta, y, ldy);
}

// Memory management for MKL sparse matrix
template <typename FL>
struct MKLSparseAllocator : Allocator<typename GMatrix<FL>::FP> {
    shared_ptr<sparse_matrix_t> mat;
    MKLSparseAllocator(const shared_ptr<sparse_matrix_t> &mat) : mat(mat) {}
    void deallocate(void *ptr, size_t n) override { mat = nullptr; }
    struct Deleter {
        void operator()(sparse_matrix_t *p) {
            mkl_sparse_destroy(*p);
            delete p;
        }
    };
    static shared_ptr<sparse_matrix_t>
    to_mkl_sparse_matrix(const GCSRMatrix<FL> &mat, bool conj = false) {
        if (mat.alloc != nullptr) {
            auto &r = *mat.alloc.get();
            if (typeid(r).hash_code() == typeid(MKLSparseAllocator).hash_code())
                return dynamic_pointer_cast<MKLSparseAllocator>(mat.alloc)->mat;
        }
        shared_ptr<sparse_matrix_t> spa =
            shared_ptr<sparse_matrix_t>(new sparse_matrix_t, Deleter());
        sparse_status_t st =
            !conj
                ? mkl_sparse_x_create_csr<FL>(spa.get(), SPARSE_INDEX_BASE_ZERO,
                                              mat.m, mat.n, mat.rows,
                                              mat.rows + 1, mat.cols, mat.data)
                : mkl_sparse_x_create_csc<FL>(spa.get(), SPARSE_INDEX_BASE_ZERO,
                                              mat.n, mat.m, mat.rows,
                                              mat.rows + 1, mat.cols, mat.data);
        assert(st == SPARSE_STATUS_SUCCESS);
        if (conj) {
            shared_ptr<sparse_matrix_t> spx =
                shared_ptr<sparse_matrix_t>(new sparse_matrix_t, Deleter());
            st = mkl_sparse_convert_csr(*spa, SPARSE_OPERATION_NON_TRANSPOSE,
                                        spx.get());
            assert(st == SPARSE_STATUS_SUCCESS);
            spa = spx;
        }
        return spa;
    }
    static GCSRMatrix<FL>
    from_mkl_sparse_matrix(const shared_ptr<sparse_matrix_t> &spa) {
        GCSRMatrix<FL> mat;
        sparse_index_base_t ibt;
        MKL_INT *rows_end;
        sparse_status_t st =
            mkl_sparse_x_export_csr<FL>(*spa, &ibt, &mat.m, &mat.n, &mat.rows,
                                        &rows_end, &mat.cols, &mat.data);
        assert(ibt == SPARSE_INDEX_BASE_ZERO);
        assert(st == SPARSE_STATUS_SUCCESS);
        mat.nnz = rows_end[mat.m - 1];
        mat.alloc = make_shared<MKLSparseAllocator>(spa);
        return mat;
    }
};

#endif

// CSR matrix operations
template <typename FL> struct GCSRMatrixFunctions {
    typedef typename GMatrix<FL>::FP FP;
    static const int cpx_sz = sizeof(FL) / sizeof(FP);
    // a = b
    static void copy(const GCSRMatrix<FL> &a, const GCSRMatrix<FL> &b) {
        const MKL_INT na = a.memory_size(), nb = b.memory_size(), inc = 1;
        assert(na == nb);
        xcopy<FL>(&na, b.data, &inc, a.data, &inc);
    }
    static void iscale(const GCSRMatrix<FL> &a, FL scale) {
        const MKL_INT inc = 1;
        xscal<FL>(&a.nnz, &scale, a.data, &inc);
    }
    static FP norm(const GCSRMatrix<FL> &a) {
        const MKL_INT inc = 1;
        return xnrm2<FL>(&a.nnz, a.data, &inc);
    }
    // normal dot with no conj
    static FL dot(const GCSRMatrix<FL> &a, const GCSRMatrix<FL> &b) {
        assert(a.m == b.m && a.n == b.n && a.nnz == b.nnz);
        const MKL_INT inc = 1;
        return xdot<FL>(&a.nnz, a.data, &inc, b.data, &inc);
    }
    static FL sparse_dot(const GCSRMatrix<FL> &a, const GCSRMatrix<FL> &b) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        FL r = 0;
        if (a.nnz == a.size()) {
            if (b.nnz == b.size())
                r = GMatrixFunctions<FL>::dot(a.dense_ref(), b.dense_ref());
            else {
                GMatrix<FL> bd((FL *)d_alloc->allocate(b.size() * cpx_sz), b.m,
                               b.n);
                b.to_dense(bd);
                r = GMatrixFunctions<FL>::dot(a.dense_ref(), bd);
                bd.deallocate(d_alloc);
            }
            return r;
        } else if (b.nnz == b.size()) {
            GMatrix<FL> ad((FL *)d_alloc->allocate(a.size() * cpx_sz), a.m,
                           a.n);
            a.to_dense(ad);
            r = GMatrixFunctions<FL>::dot(ad, b.dense_ref());
            ad.deallocate(d_alloc);
            return r;
        }
        const MKL_INT *arows = a.rows, *acols = a.cols;
        const MKL_INT *brows = b.rows, *bcols = b.cols;
        FL *adata = a.data, *bdata = b.data;
        const MKL_INT am = a.m, an = a.n, bm = b.m, bn = b.n;
        assert(am == bm && an == bn);
        MKL_INT k = 0;
        for (MKL_INT i = 0; i < am; i++) {
            MKL_INT ja = arows[i], jar = arows[i + 1];
            MKL_INT jb = brows[i], jbr = brows[i + 1];
            for (; ja < jar && jb < jbr; k++) {
                if (acols[ja] == bcols[jb]) {
                    r += adata[ja] * bdata[jb];
                    ja++, jb++;
                } else if (acols[ja] > bcols[jb])
                    jb++;
                else
                    ja++;
            }
        }
        return r;
    }
    // a = a + scale * op(b)
    static void iadd(GCSRMatrix<FL> &a, const GCSRMatrix<FL> &b, FL scale,
                     bool conj = false) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        if (a.nnz == a.size()) {
            if (b.nnz == b.size())
                GMatrixFunctions<FL>::iadd(a.dense_ref(), b.dense_ref(), scale,
                                           conj);
            else {
                GMatrix<FL> bd((FL *)d_alloc->allocate(b.size() * cpx_sz), b.m,
                               b.n);
                b.to_dense(bd);
                GMatrixFunctions<FL>::iadd(a.dense_ref(), bd, scale, conj);
                bd.deallocate(d_alloc);
            }
            return;
        } else if (b.nnz == b.size()) {
            GMatrix<FL> ad((FL *)d_alloc->allocate(a.size() * cpx_sz), a.m,
                           a.n);
            a.to_dense(ad);
            GMatrixFunctions<FL>::iadd(ad, b.dense_ref(), scale, conj);
            a.deallocate();
            a.from_dense(ad);
            ad.deallocate(d_alloc);
            return;
        } else if (b.nnz == 0)
            return;
#ifdef _HAS_INTEL_MKL
        shared_ptr<sparse_matrix_t> spa =
            MKLSparseAllocator<FL>::to_mkl_sparse_matrix(a);
        shared_ptr<sparse_matrix_t> spb =
            MKLSparseAllocator<FL>::to_mkl_sparse_matrix(b);
        shared_ptr<sparse_matrix_t> spc = shared_ptr<sparse_matrix_t>(
            new sparse_matrix_t, typename MKLSparseAllocator<FL>::Deleter());
        sparse_status_t st =
            mkl_sparse_x_add<FL>(conj ? SPARSE_OPERATION_CONJUGATE_TRANSPOSE
                                      : SPARSE_OPERATION_NON_TRANSPOSE,
                                 *spb, scale, *spa, spc.get());
        assert(st == SPARSE_STATUS_SUCCESS);
        a.deallocate();
        a = MKLSparseAllocator<FL>::from_mkl_sparse_matrix(spc);
#else
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        GCSRMatrix<FL> tmp;
        MKL_INT *arows = a.rows, *acols = a.cols, *brows = b.rows,
                *bcols = b.cols;
        FL *adata = a.data, *bdata = b.data;
        const MKL_INT am = a.m, an = a.n;
        const MKL_INT bm = conj ? b.n : b.m, bn = conj ? b.m : b.n;
        assert(am == bm && an == bn);
        if (conj)
            tmp = b.transpose(d_alloc), brows = tmp.rows, bcols = tmp.cols,
            bdata = tmp.data;
        MKL_INT *rrows = (MKL_INT *)i_alloc->allocate((a.m + 1) * _MINTSZ);
        MKL_INT *rcols =
            (MKL_INT *)i_alloc->allocate((a.nnz + b.nnz) * _MINTSZ);
        FL *rdata = (FL *)d_alloc->allocate((a.nnz + b.nnz) * cpx_sz);
        MKL_INT k = 0;
        for (MKL_INT i = 0; i < am; i++) {
            rrows[i] = k;
            MKL_INT ja = arows[i], jar = arows[i + 1];
            MKL_INT jb = brows[i], jbr = brows[i + 1];
            for (; ja < jar || jb < jbr; k++) {
                if (ja >= jar)
                    rcols[k] = bcols[jb],
                    rdata[k] =
                        (conj ? xconj<FL>(bdata[jb]) : bdata[jb]) * scale,
                    jb++;
                else if (jb >= jbr)
                    rcols[k] = acols[ja], rdata[k] = adata[ja], ja++;
                else if (acols[ja] == bcols[jb]) {
                    rcols[k] = acols[ja];
                    rdata[k] =
                        adata[ja] +
                        (conj ? xconj<FL>(bdata[jb]) : bdata[jb]) * scale;
                    if (abs(rdata[k]) < TINY)
                        k--;
                    ja++, jb++;
                } else if (acols[ja] > bcols[jb])
                    rcols[k] = bcols[jb],
                    rdata[k] =
                        (conj ? xconj<FL>(bdata[jb]) : bdata[jb]) * scale,
                    jb++;
                else
                    rcols[k] = acols[ja], rdata[k] = adata[ja], ja++;
            }
        }
        rrows[am] = k;
        GCSRMatrix<FL> r(am, an, k, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        if (r.nnz != r.size()) {
            memcpy(r.rows, rrows, (r.m + 1) * sizeof(MKL_INT));
            memcpy(r.cols, rcols, r.nnz * sizeof(MKL_INT));
        }
        memcpy(r.data, rdata, r.nnz * sizeof(FL));
        d_alloc->deallocate(rdata, (a.nnz + b.nnz) * cpx_sz);
        i_alloc->deallocate(rcols, (a.nnz + b.nnz) * _MINTSZ);
        i_alloc->deallocate(rrows, (a.m + 1) * _MINTSZ);
        a.deallocate();
        a = r;
        if (conj)
            tmp.deallocate();
#endif
    }
    static void multiply(const GCSRMatrix<FL> &a, uint8_t conja,
                         const GCSRMatrix<FL> &b, uint8_t conjb,
                         GCSRMatrix<FL> &c, FL scale, FL cfactor) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        if (a.nnz == a.size() || b.nnz == b.size()) {
            if (c.nnz == c.size()) {
                if (a.nnz == a.size() && b.nnz == b.size())
                    GMatrixFunctions<FL>::multiply(
                        a.dense_ref(), conja, b.dense_ref(), conjb,
                        c.dense_ref(), scale, cfactor);
                else if (a.nnz == a.size())
                    multiply(a.dense_ref(), conja, b, conjb, c.dense_ref(),
                             scale, cfactor);
                else
                    multiply(a, conja, b.dense_ref(), conjb, c.dense_ref(),
                             scale, cfactor);
            } else {
                GMatrix<FL> cd((FL *)d_alloc->allocate(c.size() * cpx_sz), c.m,
                               c.n);
                c.to_dense(cd);
                if (a.nnz == a.size() && b.nnz == b.size())
                    GMatrixFunctions<FL>::multiply(a.dense_ref(), conja,
                                                   b.dense_ref(), conjb, cd,
                                                   scale, cfactor);
                else if (a.nnz == a.size())
                    multiply(a.dense_ref(), conja, b, conjb, cd, scale,
                             cfactor);
                else
                    multiply(a, conja, b.dense_ref(), conjb, cd, scale,
                             cfactor);
                c.deallocate();
                c.from_dense(cd);
                cd.deallocate(d_alloc);
            }
            return;
        } else if (c.nnz == c.size()) {
            GMatrix<FL> bd((FL *)d_alloc->allocate(b.size() * cpx_sz), b.m,
                           b.n);
            b.to_dense(bd);
            multiply(a, conja, bd, conjb, c.dense_ref(), scale, cfactor);
            bd.deallocate(d_alloc);
            return;
        }
        vector<GCSRMatrix<FL>> tmps;
        MKL_INT *arows = a.rows, *acols = a.cols, *brows = b.rows,
                *bcols = b.cols;
        FL *adata = a.data, *bdata = b.data;
        const MKL_INT am = (conja & 1) ? a.n : a.m,
                      an = (conja & 1) ? a.m : a.n;
        const MKL_INT bm = (conjb & 1) ? b.n : b.m,
                      bn = (conjb & 1) ? b.m : b.n;
        assert(am == c.m && bn == c.n && an == bm);
        if (conja & 1)
            tmps.push_back(a.transpose(d_alloc)), arows = tmps.back().rows,
                                                  acols = tmps.back().cols,
                                                  adata = tmps.back().data;
        if (conjb & 1)
            tmps.push_back(b.transpose(d_alloc)), brows = tmps.back().rows,
                                                  bcols = tmps.back().cols,
                                                  bdata = tmps.back().data;
        if (conja == 2 || conja == 1)
            GMatrixFunctions<FL>::conjugate(GMatrix<FL>(adata, a.nnz, 1));
        if (conjb == 2 || conjb == 1)
            GMatrixFunctions<FL>::conjugate(GMatrix<FL>(bdata, b.nnz, 1));
        MKL_INT *r_idx = (MKL_INT *)i_alloc->allocate((c.m + 1) * _MINTSZ);
        vector<MKL_INT> tcols, rcols;
        vector<FL> tdata, rdata;
        for (MKL_INT i = 0, jp, jr, inc = 1; i < am; i++) {
            r_idx[i] = (MKL_INT)rcols.size();
            jp = c.rows[i], jr = i == c.m - 1 ? c.nnz : c.rows[i + 1];
            jr = jr - jp;
            tcols.clear(), tdata.clear();
            if (jr != 0 && cfactor != (FL)0.0) {
                tcols.resize(jr), tdata.resize(jr);
                memcpy(tcols.data(), c.cols + jp, jr * sizeof(MKL_INT));
                memcpy(tdata.data(), c.data + jp, jr * sizeof(FL));
                xscal<FL>(&jr, &cfactor, tdata.data(), &inc);
            }
            jp = arows[i], jr = i == am - 1 ? a.nnz : arows[i + 1];
            for (MKL_INT j = jp; j < jr; j++) {
                MKL_INT kp = brows[acols[j]],
                        kr = acols[j] == bm - 1 ? b.nnz : brows[acols[j] + 1];
                for (MKL_INT k = kp; k < kr; k++)
                    tcols.push_back(bcols[k]),
                        tdata.push_back(adata[j] * bdata[k] * scale);
            }
            if (tcols.size() != 0) {
                MKL_INT *idx =
                    (MKL_INT *)i_alloc->allocate(tcols.size() * _MINTSZ);
                for (MKL_INT l = 0; l < (MKL_INT)tcols.size(); l++)
                    idx[l] = l;
                sort(idx, idx + tcols.size(), [&tcols](MKL_INT ti, MKL_INT tj) {
                    return tcols[ti] < tcols[tj];
                });
                rcols.push_back(tcols[idx[0]]), rdata.push_back(tdata[idx[0]]);
                for (MKL_INT l = 1; l < (MKL_INT)tcols.size(); l++)
                    if (rcols.back() == tcols[idx[l]])
                        rdata.back() += tdata[idx[l]];
                    else if (abs(rdata.back()) < TINY)
                        rcols.back() = tcols[idx[l]],
                        rdata.back() = tdata[idx[l]];
                    else
                        rcols.push_back(tcols[idx[l]]),
                            rdata.push_back(tdata[idx[l]]);
                i_alloc->deallocate(idx, tcols.size() * _MINTSZ);
            }
        }
        r_idx[am] = (MKL_INT)rcols.size();
        GCSRMatrix<FL> r(c.m, c.n, r_idx[am], nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        if (r.nnz != r.size()) {
            memcpy(r.rows, r_idx, (r.m + 1) * sizeof(MKL_INT));
            memcpy(r.cols, rcols.data(), r.nnz * sizeof(MKL_INT));
        }
        memcpy(r.data, rdata.data(), r.nnz * sizeof(FL));
        c.deallocate();
        c = r;
        i_alloc->deallocate(r_idx, (c.m + 1) * _MINTSZ);
        for (MKL_INT it = conja + conjb - 1; it >= 0; it--)
            tmps[it].deallocate();
    }
    static void multiply(const GMatrix<FL> &a, uint8_t conja,
                         const GCSRMatrix<FL> &b, uint8_t conjb,
                         const GMatrix<FL> &c, FL scale, FL cfactor) {
        if (b.nnz == b.size())
            return GMatrixFunctions<FL>::multiply(a, conja, b.dense_ref(),
                                                  conjb, c, scale, cfactor);
#ifdef _HAS_INTEL_MKL
        struct matrix_descr mt;
        mt.type = SPARSE_MATRIX_TYPE_GENERAL;
        assert(((conja & 1) ? a.n : a.m) == c.m);
        assert(((conjb & 1) ? b.m : b.n) == c.n);
        assert(((conja & 1) ? a.m : a.n) == ((conjb & 1) ? b.n : b.m));
        shared_ptr<sparse_matrix_t> spb =
            MKLSparseAllocator<FL>::to_mkl_sparse_matrix(b);
        // TODO: CSR conj not resolved
        assert(conjb != 3);
        if (!conja) {
            sparse_status_t st = mkl_sparse_x_mm<FL>(
                conjb == 2 ? SPARSE_OPERATION_CONJUGATE_TRANSPOSE
                           : (conjb == 0 ? SPARSE_OPERATION_TRANSPOSE
                                         : SPARSE_OPERATION_NON_TRANSPOSE),
                scale, *spb, mt, SPARSE_LAYOUT_COLUMN_MAJOR, a.data, a.m, a.n,
                cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
        } else {
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            GMatrix<FL> at(nullptr, (conja & 1) ? a.n : a.m,
                           (conja & 1) ? a.m : a.n);
            at.allocate(d_alloc);
            if (conja == 3)
                GMatrixFunctions<FL>::iadd(at, a, 1.0, true, 0.0);
            else if (conja == 1)
                GMatrixFunctions<FL>::transpose(at, a, 1.0, 0.0);
            else {
                GMatrixFunctions<FL>::copy(at, a);
                GMatrixFunctions<FL>::conjugate(at);
            }
            sparse_status_t st = mkl_sparse_x_mm<FL>(
                conjb == 2 ? SPARSE_OPERATION_CONJUGATE_TRANSPOSE
                           : (conjb == 0 ? SPARSE_OPERATION_TRANSPOSE
                                         : SPARSE_OPERATION_NON_TRANSPOSE),
                scale, *spb, mt, SPARSE_LAYOUT_COLUMN_MAJOR, at.data, at.m,
                at.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
            at.deallocate(d_alloc);
        }
#else
        const MKL_INT am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        const MKL_INT bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        assert(am == c.m && bn == c.n && an == bm);
        if (cfactor != (FL)1.0)
            GMatrixFunctions<FL>::iscale(c, cfactor);
        if (!conja && !conjb) {
            for (MKL_INT ib = 0; ib < b.m; ib++) {
                const MKL_INT jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (MKL_INT jb = jbp; jb < jbr; jb++) {
                    const FL factor = scale * b.data[jb];
                    xaxpy<FL>(&a.m, &factor, &a(0, ib), &a.n, &c(0, b.cols[jb]),
                              &c.n);
                }
            }
        } else if (conja && !conjb) {
            const MKL_INT inc = 1;
            for (MKL_INT ib = 0; ib < b.m; ib++) {
                const MKL_INT jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (MKL_INT jb = jbp; jb < jbr; jb++) {
                    const FL factor = scale * b.data[jb];
                    xaxpy<FL>(&a.n, &factor, &a(ib, 0), &inc, &c(0, b.cols[jb]),
                              &c.n);
                }
            }
        } else if (!conja && conjb) {
            for (MKL_INT ib = 0; ib < b.m; ib++) {
                const MKL_INT jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (MKL_INT jb = jbp; jb < jbr; jb++) {
                    const FL factor = scale * b.data[jb];
                    xaxpy<FL>(&a.m, &factor, &a(0, b.cols[jb]), &a.n, &c(0, ib),
                              &c.n);
                }
            }
        } else {
            const MKL_INT inc = 1;
            for (MKL_INT ib = 0; ib < b.m; ib++) {
                const MKL_INT jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (MKL_INT jb = jbp; jb < jbr; jb++) {
                    const FL factor = scale * b.data[jb];
                    xaxpy<FL>(&a.n, &factor, &a(b.cols[jb], 0), &inc, &c(0, ib),
                              &c.n);
                }
            }
        }
#endif
    }
    static void multiply(const GCSRMatrix<FL> &a, uint8_t conja,
                         const GMatrix<FL> &b, uint8_t conjb,
                         const GMatrix<FL> &c, FL scale, FL cfactor) {
        if (a.nnz == a.size())
            return GMatrixFunctions<FL>::multiply(a.dense_ref(), conja, b,
                                                  conjb, c, scale, cfactor);
#ifdef _HAS_INTEL_MKL
        const struct matrix_descr mt {
            SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER,
                SPARSE_DIAG_NON_UNIT
        };
        shared_ptr<sparse_matrix_t> spa =
            MKLSparseAllocator<FL>::to_mkl_sparse_matrix(a);
        if (!conjb) {
            sparse_status_t st =
                mkl_sparse_x_mm<FL>(conja ? SPARSE_OPERATION_CONJUGATE_TRANSPOSE
                                          : SPARSE_OPERATION_NON_TRANSPOSE,
                                    scale, *spa, mt, SPARSE_LAYOUT_ROW_MAJOR,
                                    b.data, b.n, b.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
        } else {
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            GMatrix<FL> bt(nullptr, b.n, b.m);
            bt.allocate(d_alloc);
            if (conjb == 3)
                GMatrixFunctions<FL>::iadd(bt, b, 1.0, true, 0.0);
            else if (conjb == 1)
                GMatrixFunctions<FL>::transpose(bt, b, 1.0, 0.0);
            else {
                GMatrixFunctions<FL>::copy(bt, b);
                GMatrixFunctions<FL>::conjugate(bt);
            }
            sparse_status_t st =
                mkl_sparse_x_mm<FL>(conja ? SPARSE_OPERATION_CONJUGATE_TRANSPOSE
                                          : SPARSE_OPERATION_NON_TRANSPOSE,
                                    scale, *spa, mt, SPARSE_LAYOUT_ROW_MAJOR,
                                    bt.data, bt.n, bt.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
            bt.deallocate(d_alloc);
        }
#else
        const MKL_INT am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        const MKL_INT bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        const MKL_INT inc = 1;
        assert(am == c.m && bn == c.n && an == bm);
        if (cfactor != (FL)1.0)
            GMatrixFunctions<FL>::iscale(c, cfactor);
        if (!conja && !conjb) {
            for (MKL_INT ia = 0; ia < a.m; ia++) {
                const MKL_INT jap = a.rows[ia], jar = a.rows[ia + 1];
                for (MKL_INT ja = jap; ja < jar; ja++) {
                    const FL factor = scale * a.data[ja];
                    xaxpy<FL>(&b.n, &factor, &b(a.cols[ja], 0), &inc, &c(ia, 0),
                              &inc);
                }
            }
        } else if (conja && !conjb) {
            for (MKL_INT ia = 0; ia < a.m; ia++) {
                const MKL_INT jap = a.rows[ia], jar = a.rows[ia + 1];
                for (MKL_INT ja = jap; ja < jar; ja++) {
                    const FL factor = scale * a.data[ja];
                    xaxpy<FL>(&b.n, &factor, &b(ia, 0), &inc, &c(a.cols[ja], 0),
                              &inc);
                }
            }
        } else if (!conja && conjb) {
            for (MKL_INT ia = 0; ia < a.m; ia++) {
                const MKL_INT jap = a.rows[ia], jar = a.rows[ia + 1];
                for (MKL_INT ja = jap; ja < jar; ja++) {
                    const FL factor = scale * a.data[ja];
                    xaxpy<FL>(&b.m, &factor, &b(0, a.cols[ja]), &b.n, &c(ia, 0),
                              &inc);
                }
            }
        } else {
            for (MKL_INT ia = 0; ia < a.m; ia++) {
                const MKL_INT jap = a.rows[ia], jar = a.rows[ia + 1];
                for (MKL_INT ja = jap; ja < jar; ja++) {
                    const FL factor = scale * a.data[ja];
                    xaxpy<FL>(&b.m, &factor, &b(0, ia), &b.n, &c(a.cols[ja], 0),
                              &inc);
                }
            }
        }
#endif
    }
    // c = bra * a * ket(.T) for tensor product multiplication
    static void rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                       const GCSRMatrix<FL> &bra, uint8_t conj_bra,
                       const GCSRMatrix<FL> &ket, uint8_t conj_ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate(d_alloc);
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate(d_alloc);
    }
    // c = bra * a * ket(.T) for tensor product multiplication
    static void rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                       const GCSRMatrix<FL> &bra, uint8_t conj_bra,
                       const GMatrix<FL> &ket, uint8_t conj_ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate(d_alloc);
        GMatrixFunctions<FL>::multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate(d_alloc);
    }
    // c = bra * a * ket(.T) for tensor product multiplication
    static void rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                       const GMatrix<FL> &bra, uint8_t conj_bra,
                       const GCSRMatrix<FL> &ket, uint8_t conj_ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate(d_alloc);
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        GMatrixFunctions<FL>::multiply(bra, conj_bra, work, false, c, scale,
                                       1.0);
        work.deallocate(d_alloc);
    }
    // c = bra * a * ket(.T) for operator rotation
    static void rotate(const GCSRMatrix<FL> &a, const GMatrix<FL> &c,
                       const GMatrix<FL> &bra, uint8_t conj_bra,
                       const GMatrix<FL> &ket, uint8_t conj_ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate(d_alloc);
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        GMatrixFunctions<FL>::multiply(bra, conj_bra, work, false, c, scale,
                                       1.0);
        work.deallocate(d_alloc);
    }
    // c(.T) = bra.T * a(.T) * ket for partial expectation
    static void left_partial_rotate(const GCSRMatrix<FL> &a, bool conj_a,
                                    const GMatrix<FL> &c, bool conj_c,
                                    const GMatrix<FL> &bra,
                                    const GMatrix<FL> &ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, conj_a ? a.n : a.m, ket.n);
        work.allocate(d_alloc);
        multiply(a, conj_a, ket, false, work, 1.0, 0.0);
        if (!conj_c)
            GMatrixFunctions<FL>::multiply(bra, true, work, false, c, scale,
                                           1.0);
        else
            GMatrixFunctions<FL>::multiply(work, true, bra, false, c, scale,
                                           1.0);
        work.deallocate(d_alloc);
    }
    // c(.T) = bra.c * a(.T) * ket.t for partial expectation
    static void right_partial_rotate(const GCSRMatrix<FL> &a, bool conj_a,
                                     const GMatrix<FL> &c, bool conj_c,
                                     const GMatrix<FL> &bra,
                                     const GMatrix<FL> &ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, conj_a ? a.m : a.n, bra.m);
        work.allocate(d_alloc);
        multiply(a, !conj_a, bra, true, work, 1.0, 0.0);
        if (!conj_c)
            GMatrixFunctions<FL>::multiply(work, true, ket, true, c, scale,
                                           1.0);
        else
            GMatrixFunctions<FL>::multiply(ket, false, work, false, c, scale,
                                           1.0);
        work.deallocate(d_alloc);
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(uint8_t abconj, const GCSRMatrix<FL> &a,
                                        const GMatrix<FL> &b,
                                        const GMatrix<FL> &c, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const FL cfactor = 1.0;
        const MKL_INT k = 1, lda = 1, ldb = b.n + 1;
        GMatrix<FL> ad(nullptr, a.m, 1);
        ad.allocate(d_alloc);
        a.diag(ad);
        xgemm<FL>("t", "n", &b.n, &a.n, &k, &scale, b.data, &ldb, ad.data, &lda,
                  &cfactor, c.data, &c.n);
        ad.deallocate(d_alloc);
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(uint8_t abconj, const GMatrix<FL> &a,
                                        const GCSRMatrix<FL> &b,
                                        const GMatrix<FL> &c, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const FL cfactor = 1.0;
        const MKL_INT k = 1, lda = a.n + 1, ldb = 1;
        GMatrix<FL> bd(nullptr, b.m, 1);
        bd.allocate(d_alloc);
        b.diag(bd);
        xgemm<FL>("t", "n", &b.n, &a.n, &k, &scale, bd.data, &ldb, a.data, &lda,
                  &cfactor, c.data, &c.n);
        bd.deallocate(d_alloc);
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(uint8_t abconj, const GCSRMatrix<FL> &a,
                                        const GCSRMatrix<FL> &b,
                                        const GMatrix<FL> &c, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const FL cfactor = 1.0;
        const MKL_INT k = 1, lda = 1, ldb = 1;
        GMatrix<FL> ad(nullptr, a.m, 1), bd(nullptr, b.m, 1);
        ad.allocate(d_alloc), bd.allocate(d_alloc);
        a.diag(ad), b.diag(bd);
        xgemm<FL>("t", "n", &b.n, &a.n, &k, &scale, bd.data, &ldb, ad.data,
                  &lda, &cfactor, c.data, &c.n);
        bd.deallocate(d_alloc), ad.deallocate(d_alloc);
    }
    static void tensor_product(const GCSRMatrix<FL> &a, bool conja,
                               const GCSRMatrix<FL> &b, bool conjb,
                               GCSRMatrix<FL> &c, FL scale, uint32_t stride) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        if (a.nnz == a.size() || b.nnz == b.size()) {
            if (a.nnz == a.size())
                tensor_product(a.dense_ref(), conja, b, conjb, c, scale,
                               stride);
            else
                tensor_product(a, conja, b.dense_ref(), conjb, c, scale,
                               stride);
            return;
        } else if (c.nnz == c.size()) {
            GMatrix<FL> ad = GMatrix<FL>(
                (FL *)d_alloc->allocate(a.size() * cpx_sz), a.m, a.n);
            a.to_dense(ad);
            GMatrix<FL> bd = GMatrix<FL>(
                (FL *)d_alloc->allocate(b.size() * cpx_sz), b.m, b.n);
            b.to_dense(bd);
            GMatrixFunctions<FL>::tensor_product(ad, conja, bd, conjb,
                                                 c.dense_ref(), scale, stride);
            ad.deallocate(d_alloc);
            bd.deallocate(d_alloc);
            return;
        }
        GCSRMatrix<FL> r(c.m, c.n, a.nnz * b.nnz, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        uint32_t m_length = (conja ? a.n : a.m) * (conjb ? b.n : b.m);
        memset(r.rows, 0, (r.m + 1) * sizeof(MKL_INT));
        vector<GCSRMatrix<FL>> tmps;
        MKL_INT *arows = a.rows, *acols = a.cols, *brows = b.rows,
                *bcols = b.cols;
        FL *adata = a.data, *bdata = b.data;
        MKL_INT am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        MKL_INT bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        if (conja)
            tmps.push_back(a.transpose(d_alloc)), arows = tmps.back().rows,
                                                  acols = tmps.back().cols,
                                                  adata = tmps.back().data;
        if (conjb)
            tmps.push_back(b.transpose(d_alloc)), brows = tmps.back().rows,
                                                  bcols = tmps.back().cols,
                                                  bdata = tmps.back().data;
        for (MKL_INT ir = m_stride, ix = 0, ia = 0; ia < am; ia++) {
            MKL_INT jap = arows[ia], jar = ia == am - 1 ? a.nnz : arows[ia + 1];
            for (MKL_INT ib = 0; ib < bm; ib++, ir++) {
                MKL_INT jbp = brows[ib],
                        jbr = ib == bm - 1 ? b.nnz : brows[ib + 1];
                r.rows[ir + 1] = r.rows[ir] + (jar - jap) * (jbr - jbp);
                for (MKL_INT ja = jap; ja < jar; ja++)
                    for (MKL_INT jb = jbp; jb < jbr; jb++, ix++) {
                        r.cols[ix] = bn * acols[ja] + bcols[jb] + n_stride;
                        r.data[ix] = scale * adata[ja] * bdata[jb];
                    }
            }
        }
        for (MKL_INT it = conja + conjb - 1; it >= 0; it--)
            tmps[it].deallocate();
        assert(r.rows[m_stride + m_length] == r.nnz);
        for (MKL_INT ir = m_stride + m_length; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0) {
            c.deallocate();
            c = r;
        } else {
            iadd(c, r, 1.0);
            r.deallocate();
        }
    }
    static void tensor_product(const GCSRMatrix<FL> &a, bool conja,
                               const GMatrix<FL> &b, bool conjb,
                               GCSRMatrix<FL> &c, FL scale, uint32_t stride) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        if (a.nnz == a.size() || c.nnz == c.size()) {
            GMatrix<FL> ad =
                a.nnz == a.size()
                    ? a.dense_ref()
                    : GMatrix<FL>((FL *)d_alloc->allocate(a.size() * cpx_sz),
                                  a.m, a.n);
            if (a.nnz != a.size())
                a.to_dense(ad);
            GMatrix<FL> cd =
                c.nnz == c.size()
                    ? c.dense_ref()
                    : GMatrix<FL>((FL *)d_alloc->allocate(c.size() * cpx_sz),
                                  c.m, c.n);
            if (c.nnz != c.size())
                c.to_dense(cd);
            GMatrixFunctions<FL>::tensor_product(ad, conja, b, conjb, cd, scale,
                                                 stride);
            if (a.nnz != a.size())
                ad.deallocate(d_alloc);
            if (c.nnz != c.size()) {
                c.deallocate();
                c.from_dense(cd);
                cd.deallocate(d_alloc);
            }
            return;
        }
        GCSRMatrix<FL> r(c.m, c.n, a.nnz * b.m * b.n, nullptr, nullptr,
                         nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        uint32_t m_length = (conja ? a.n : a.m) * (conjb ? b.n : b.m);
        memset(r.rows, 0, (r.m + 1) * sizeof(MKL_INT));
        GCSRMatrix<FL> tmp;
        MKL_INT *arows = a.rows, *acols = a.cols;
        FL *adata = a.data;
        MKL_INT am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        MKL_INT bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        if (conja)
            tmp = a.transpose(d_alloc), arows = tmp.rows, acols = tmp.cols,
            adata = tmp.data;
        for (MKL_INT ir = m_stride, ix = 0, ia = 0; ia < am; ia++) {
            MKL_INT jap = arows[ia], jar = ia == am - 1 ? a.nnz : arows[ia + 1];
            for (MKL_INT ib = 0; ib < bm; ib++, ir++) {
                r.rows[ir + 1] = r.rows[ir] + (jar - jap) * bn;
                if (conjb)
                    for (MKL_INT ja = jap; ja < jar; ja++)
                        for (MKL_INT jb = 0; jb < bn; jb++, ix++) {
                            r.cols[ix] = bn * acols[ja] + jb + n_stride;
                            r.data[ix] =
                                scale * adata[ja] * b.data[jb * bm + ib];
                        }
                else
                    for (MKL_INT ja = jap; ja < jar; ja++)
                        for (MKL_INT jb = 0; jb < bn; jb++, ix++) {
                            r.cols[ix] = bn * acols[ja] + jb + n_stride;
                            r.data[ix] =
                                scale * adata[ja] * b.data[ib * bn + jb];
                        }
            }
        }
        if (conja)
            tmp.deallocate();
        assert(r.rows[m_stride + m_length] == r.nnz);
        for (MKL_INT ir = m_stride + m_length; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0) {
            c.deallocate();
            c = r;
        } else {
            iadd(c, r, 1.0);
            r.deallocate();
        }
    }
    static void tensor_product(const GMatrix<FL> &a, bool conja,
                               const GCSRMatrix<FL> &b, bool conjb,
                               GCSRMatrix<FL> &c, FL scale, uint32_t stride) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        if (b.nnz == b.size() || c.nnz == c.size()) {
            GMatrix<FL> bd =
                b.nnz == b.size()
                    ? b.dense_ref()
                    : GMatrix<FL>((FL *)d_alloc->allocate(b.size() * cpx_sz),
                                  b.m, b.n);
            if (b.nnz != b.size())
                b.to_dense(bd);
            GMatrix<FL> cd =
                c.nnz == c.size()
                    ? c.dense_ref()
                    : GMatrix<FL>((FL *)d_alloc->allocate(c.size() * cpx_sz),
                                  c.m, c.n);
            if (c.nnz != c.size())
                c.to_dense(cd);
            GMatrixFunctions<FL>::tensor_product(a, conja, bd, conjb, cd, scale,
                                                 stride);
            if (b.nnz != b.size())
                bd.deallocate(d_alloc);
            if (c.nnz != c.size()) {
                c.deallocate();
                c.from_dense(cd);
                cd.deallocate(d_alloc);
            }
            return;
        }
        GCSRMatrix<FL> r(c.m, c.n, a.m * a.n * b.nnz, nullptr, nullptr,
                         nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        uint32_t m_length = (conja ? a.n : a.m) * (conjb ? b.n : b.m);
        memset(r.rows, 0, (r.m + 1) * sizeof(MKL_INT));
        GCSRMatrix<FL> tmp;
        MKL_INT *brows = b.rows, *bcols = b.cols;
        FL *bdata = b.data;
        MKL_INT am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        MKL_INT bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        if (conjb)
            tmp = b.transpose(d_alloc), brows = tmp.rows, bcols = tmp.cols,
            bdata = tmp.data;
        for (MKL_INT ir = m_stride, ix = 0, ia = 0; ia < am; ia++) {
            for (MKL_INT ib = 0; ib < bm; ib++, ir++) {
                MKL_INT jbp = brows[ib],
                        jbr = ib == bm - 1 ? b.nnz : brows[ib + 1];
                r.rows[ir + 1] = r.rows[ir] + an * (jbr - jbp);
                if (conja)
                    for (MKL_INT ja = 0; ja < an; ja++)
                        for (MKL_INT jb = jbp; jb < jbr; jb++, ix++) {
                            r.cols[ix] = bn * ja + bcols[jb] + n_stride;
                            r.data[ix] =
                                scale * a.data[ja * am + ia] * bdata[jb];
                        }
                else
                    for (MKL_INT ja = 0; ja < an; ja++)
                        for (MKL_INT jb = jbp; jb < jbr; jb++, ix++) {
                            r.cols[ix] = bn * ja + bcols[jb] + n_stride;
                            r.data[ix] =
                                scale * a.data[ia * an + ja] * bdata[jb];
                        }
            }
        }
        if (conjb)
            tmp.deallocate();
        assert(r.rows[m_stride + m_length] == r.nnz);
        for (MKL_INT ir = m_stride + m_length; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0) {
            c.deallocate();
            c = r;
        } else {
            iadd(c, r, 1.0);
            r.deallocate();
        }
    }
};

} // namespace block2
