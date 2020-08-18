
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

#include "allocator.hpp"
#include "csr_matrix.hpp"
#include "matrix.hpp"
#include "matrix_functions.hpp"
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
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

// Memory management for MKL sparse matrix
struct MKLSparseAllocator : Allocator<double> {
    shared_ptr<sparse_matrix_t> mat;
    MKLSparseAllocator(const shared_ptr<sparse_matrix_t> &mat) : mat(mat) {}
    void deallocate(void *ptr, size_t n) override { mkl_sparse_destroy(*mat); }
    struct Deleter {
        void operator()(sparse_matrix_t *p) {
            mkl_sparse_destroy(*p);
            delete p;
        }
    };
    static shared_ptr<sparse_matrix_t>
    to_mkl_sparse_matrix(const CSRMatrixRef &mat, bool conj = false) {
        if (typeid(*mat.alloc).hash_code() ==
            typeid(MKLSparseAllocator).hash_code())
            return dynamic_pointer_cast<MKLSparseAllocator>(mat.alloc)->mat;
        shared_ptr<sparse_matrix_t> spa =
            shared_ptr<sparse_matrix_t>(new sparse_matrix_t, Deleter());
        sparse_status_t st =
            !conj ? mkl_sparse_d_create_csr(spa.get(), SPARSE_INDEX_BASE_ZERO,
                                            mat.m, mat.n, mat.rows,
                                            mat.rows + 1, mat.cols, mat.data)
                  : mkl_sparse_d_create_csc(spa.get(), SPARSE_INDEX_BASE_ZERO,
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
    static CSRMatrixRef
    from_mkl_sparse_matrix(const shared_ptr<sparse_matrix_t> &spa) {
        CSRMatrixRef mat;
        sparse_index_base_t ibt;
        int *rows_end;
        sparse_status_t st =
            mkl_sparse_d_export_csr(*spa, &ibt, &mat.m, &mat.n, &mat.rows,
                                    &rows_end, &mat.cols, &mat.data);
        assert(ibt == SPARSE_INDEX_BASE_ZERO);
        assert(st == SPARSE_STATUS_SUCCESS);
        mat.nnz = rows_end[mat.m - 1];
        mat.alloc = make_shared<MKLSparseAllocator>(spa);
        return mat;
    }
};

// CSR matrix operations
struct CSRMatrixFunctions {
    // a = b
    static void copy(const CSRMatrixRef &a, const CSRMatrixRef &b) {
        const int na = a.memory_size(), nb = b.memory_size(), inc = 1;
        assert(na == nb);
        dcopy(&na, b.data, &inc, a.data, &inc);
    }
    static void iscale(const CSRMatrixRef &a, double scale) {
        const int inc = 1;
        dscal(&a.nnz, &scale, a.data, &inc);
    }
    // a = a + scale * op(b)
    static void iadd(CSRMatrixRef &a, const CSRMatrixRef &b, double scale,
                     bool conj = false) {
        shared_ptr<sparse_matrix_t> spa =
            MKLSparseAllocator::to_mkl_sparse_matrix(a);
        shared_ptr<sparse_matrix_t> spb =
            MKLSparseAllocator::to_mkl_sparse_matrix(b);
        shared_ptr<sparse_matrix_t> spc = make_shared<sparse_matrix_t>();
        sparse_status_t st = mkl_sparse_d_add(
            conj ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
            *spb, scale, *spa, spc.get());
        assert(st == SPARSE_STATUS_SUCCESS);
        a = MKLSparseAllocator::from_mkl_sparse_matrix(spc);
    }
    static double norm(const CSRMatrixRef &a) {
        const int inc = 1;
        return dnrm2(&a.nnz, a.data, &inc);
    }
    static double dot(const CSRMatrixRef &a, const CSRMatrixRef &b) {
        assert(a.m == b.m && a.n == b.n && a.nnz == b.nnz);
        const int inc = 1;
        return ddot(&a.nnz, a.data, &inc, b.data, &inc);
    }
    static void multiply(const CSRMatrixRef &a, bool conja,
                         const CSRMatrixRef &b, bool conjb, CSRMatrixRef &c,
                         double scale, double cfactor) {
        shared_ptr<sparse_matrix_t> spa =
            MKLSparseAllocator::to_mkl_sparse_matrix(a);
        shared_ptr<sparse_matrix_t> spb =
            MKLSparseAllocator::to_mkl_sparse_matrix(b, conjb);
        shared_ptr<sparse_matrix_t> spc = make_shared<sparse_matrix_t>();
        sparse_status_t st = mkl_sparse_spmm(
            conja ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
            *spa, *spb, spc.get());
        assert(st == SPARSE_STATUS_SUCCESS);
        CSRMatrixRef r = MKLSparseAllocator::from_mkl_sparse_matrix(spc);
        if (scale != 1)
            iscale(r, scale);
        if (cfactor != 0)
            iadd(r, c, cfactor, false);
        c = r;
    }
    static void multiply(const MatrixRef &a, bool conja, const CSRMatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        struct matrix_descr mt;
        mt.type = SPARSE_MATRIX_TYPE_GENERAL;
        assert((conja ? a.n : a.m) == c.m);
        assert((conjb ? b.m : b.n) == c.n);
        assert((conja ? a.m : a.n) == (conjb ? b.n : b.m));
        shared_ptr<sparse_matrix_t> spb =
            MKLSparseAllocator::to_mkl_sparse_matrix(b);
        if (!conja) {
            sparse_status_t st =
                mkl_sparse_d_mm(!conjb ? SPARSE_OPERATION_TRANSPOSE
                                       : SPARSE_OPERATION_NON_TRANSPOSE,
                                scale, *spb, mt, SPARSE_LAYOUT_COLUMN_MAJOR,
                                a.data, a.m, a.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
        } else {
            MatrixRef at(nullptr, a.n, a.m);
            at.allocate();
            for (int i = 0, inc = 1; i < at.m; i++)
                dcopy(&at.n, a.data + i, &at.m, at.data + i * at.n, &inc);
            sparse_status_t st =
                mkl_sparse_d_mm(!conjb ? SPARSE_OPERATION_TRANSPOSE
                                       : SPARSE_OPERATION_NON_TRANSPOSE,
                                scale, *spb, mt, SPARSE_LAYOUT_COLUMN_MAJOR,
                                at.data, at.m, at.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
            at.deallocate();
        }
    }
    static void multiply(const CSRMatrixRef &a, bool conja, const MatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        const struct matrix_descr mt {
            SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER,
                SPARSE_DIAG_NON_UNIT
        };
        shared_ptr<sparse_matrix_t> spa =
            MKLSparseAllocator::to_mkl_sparse_matrix(a);
        if (!conjb) {
            sparse_status_t st =
                mkl_sparse_d_mm(conja ? SPARSE_OPERATION_TRANSPOSE
                                      : SPARSE_OPERATION_NON_TRANSPOSE,
                                scale, *spa, mt, SPARSE_LAYOUT_ROW_MAJOR,
                                b.data, b.n, b.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
        } else {
            MatrixRef bt(nullptr, b.n, b.m);
            bt.allocate();
            for (int i = 0, inc = 1; i < bt.m; i++)
                dcopy(&bt.n, b.data + i, &bt.m, bt.data + i * bt.n, &inc);
            sparse_status_t st =
                mkl_sparse_d_mm(conja ? SPARSE_OPERATION_TRANSPOSE
                                      : SPARSE_OPERATION_NON_TRANSPOSE,
                                scale, *spa, mt, SPARSE_LAYOUT_ROW_MAJOR,
                                bt.data, bt.n, bt.n, cfactor, c.data, c.n);
            assert(st == SPARSE_STATUS_SUCCESS);
            bt.deallocate();
        }
    }
    // c = bra * a * ket(.T) for tensor product multiplication
    static void rotate(const MatrixRef &a, const MatrixRef &c,
                       const CSRMatrixRef &bra, bool conj_bra,
                       const CSRMatrixRef &ket, bool conj_ket, double scale) {
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate();
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate();
    }
    // c = bra * a * ket(.T) for operator rotation
    static void rotate(const CSRMatrixRef &a, const MatrixRef &c,
                       const MatrixRef &bra, bool conj_bra,
                       const MatrixRef &ket, bool conj_ket, double scale) {
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate();
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        MatrixFunctions::multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate();
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(const CSRMatrixRef &a,
                                        const MatrixRef &b, const MatrixRef &c,
                                        double scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const int k = 1, lda = 1, ldb = b.n + 1;
        MatrixRef ad(nullptr, a.m, 1);
        ad.allocate();
        a.diag(ad);
        dgemm("t", "n", &b.n, &a.n, &k, &scale, b.data, &ldb, ad.data, &lda,
              &cfactor, c.data, &c.n);
        ad.deallocate();
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(const MatrixRef &a,
                                        const CSRMatrixRef &b,
                                        const MatrixRef &c, double scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const int k = 1, lda = a.n + 1, ldb = 1;
        MatrixRef bd(nullptr, b.m, 1);
        bd.allocate();
        b.diag(bd);
        dgemm("t", "n", &b.n, &a.n, &k, &scale, bd.data, &ldb, a.data, &lda,
              &cfactor, c.data, &c.n);
        bd.deallocate();
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(const CSRMatrixRef &a,
                                        const CSRMatrixRef &b,
                                        const MatrixRef &c, double scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const int k = 1, lda = 1, ldb = 1;
        MatrixRef ad(nullptr, a.m, 1), bd(nullptr, b.m, 1);
        ad.allocate(), bd.allocate();
        a.diag(ad), b.diag(bd);
        dgemm("t", "n", &b.n, &a.n, &k, &scale, bd.data, &ldb, ad.data, &lda,
              &cfactor, c.data, &c.n);
        bd.deallocate(), ad.deallocate();
    }
    static void tensor_product(const CSRMatrixRef &a, bool conja,
                               const CSRMatrixRef &b, bool conjb,
                               CSRMatrixRef &c, double scale, uint32_t stride) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        CSRMatrixRef r(c.m, c.n, a.nnz * b.nnz, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        memset(r.rows, 0, (r.m + 1) * sizeof(int));
        for (int ir = 0; ir <= m_stride; ir++)
            r.rows[ir] = 0;
        for (int ir = m_stride, ix = 0, ia = 0; ia < a.m; ia++) {
            int jap = a.rows[ia], jar = ia == a.m - 1 ? a.nnz : a.rows[ia + 1];
            for (int ib = 0; ib < b.m; ib++, ir++) {
                int jbp = b.rows[ib],
                    jbr = ib == b.m - 1 ? b.nnz : b.rows[ib + 1];
                r.rows[ir + 1] = r.rows[ir] + (jar - jap) * (jbr - jbp);
                for (int ja = jap; ja < jar; ja++)
                    for (int jb = jbp; jb < jbr; jb++, ix++) {
                        r.cols[ix] = b.n * a.cols[ja] + b.cols[jb] + n_stride;
                        r.data[ix] = scale * a.data[ja] * b.data[jb];
                    }
            }
        }
        assert(r.rows[m_stride + a.m * b.m] == r.nnz);
        for (int ir = m_stride + a.m * b.m; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0)
            c = r;
        else
            iadd(c, r, 1.0, false);
    }
};

#endif

} // namespace block2
