
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

// Memory management for MKL sparse matrix
struct MKLSparseAllocator : Allocator<double> {
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
    to_mkl_sparse_matrix(const CSRMatrixRef &mat, bool conj = false) {
        if (mat.alloc != nullptr) {
            auto &r = *mat.alloc.get();
            if (typeid(r).hash_code() == typeid(MKLSparseAllocator).hash_code())
                return dynamic_pointer_cast<MKLSparseAllocator>(mat.alloc)->mat;
        }
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

#endif

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
    static double norm(const CSRMatrixRef &a) {
        const int inc = 1;
        return dnrm2(&a.nnz, a.data, &inc);
    }
    static double dot(const CSRMatrixRef &a, const CSRMatrixRef &b) {
        assert(a.m == b.m && a.n == b.n && a.nnz == b.nnz);
        const int inc = 1;
        return ddot(&a.nnz, a.data, &inc, b.data, &inc);
    }
    // a = a + scale * op(b)
    static void iadd(CSRMatrixRef &a, const CSRMatrixRef &b, double scale,
                     bool conj = false) {
        if (a.nnz == a.size()) {
            if (b.nnz == b.size())
                MatrixFunctions::iadd(a.dense_ref(), b.dense_ref(), scale,
                                      conj);
            else {
                MatrixRef bd(dalloc->allocate(b.size()), b.m, b.n);
                b.to_dense(bd);
                MatrixFunctions::iadd(a.dense_ref(), bd, scale, conj);
                bd.deallocate();
            }
            return;
        } else if (b.nnz == b.size()) {
            MatrixRef ad(dalloc->allocate(a.size()), a.m, a.n);
            a.to_dense(ad);
            MatrixFunctions::iadd(ad, b.dense_ref(), scale, conj);
            a.deallocate();
            a.from_dense(ad);
            ad.deallocate();
            return;
        }
#ifdef _HAS_INTEL_MKL
        shared_ptr<sparse_matrix_t> spa =
            MKLSparseAllocator::to_mkl_sparse_matrix(a);
        shared_ptr<sparse_matrix_t> spb =
            MKLSparseAllocator::to_mkl_sparse_matrix(b);
        shared_ptr<sparse_matrix_t> spc = shared_ptr<sparse_matrix_t>(
            new sparse_matrix_t, MKLSparseAllocator::Deleter());
        sparse_status_t st = mkl_sparse_d_add(
            conj ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
            *spb, scale, *spa, spc.get());
        assert(st == SPARSE_STATUS_SUCCESS);
        a.deallocate();
        a = MKLSparseAllocator::from_mkl_sparse_matrix(spc);
#else
        CSRMatrixRef tmp;
        int *arows = a.rows, *acols = a.cols, *brows = b.rows, *bcols = b.cols;
        double *adata = a.data, *bdata = b.data;
        const int am = a.m, an = a.n;
        const int bm = conj ? b.n : b.m, bn = conj ? b.m : b.n;
        assert(am == bm && an == bn);
        if (conj)
            tmp = b.transpose(dalloc), brows = tmp.rows, bcols = tmp.cols,
            bdata = tmp.data;
        int *rrows = (int *)ialloc->allocate(a.m + 1);
        int *rcols = (int *)ialloc->allocate(a.nnz + b.nnz);
        double *rdata = dalloc->allocate(a.nnz + b.nnz);
        int k = 0;
        for (int i = 0; i < am; i++) {
            rrows[i] = k;
            int ja = arows[i], jar = arows[i + 1];
            int jb = brows[i], jbr = brows[i + 1];
            for (; ja < jar || jb < jbr; k++) {
                if (ja >= jar)
                    rcols[k] = bcols[jb], rdata[k] = bdata[jb] * scale, jb++;
                else if (jb >= jbr)
                    rcols[k] = acols[ja], rdata[k] = adata[ja], ja++;
                else if (acols[ja] == bcols[jb]) {
                    rcols[k] = acols[ja];
                    rdata[k] = adata[ja] + bdata[jb] * scale;
                    if (abs(rdata[k]) < TINY)
                        k--;
                    ja++, jb++;
                } else if (acols[ja] > bcols[jb])
                    rcols[k] = bcols[jb], rdata[k] = bdata[jb] * scale, jb++;
                else
                    rcols[k] = acols[ja], rdata[k] = adata[ja], ja++;
            }
        }
        rrows[am] = k;
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        CSRMatrixRef r(am, an, k, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        if (r.nnz != r.size()) {
            memcpy(r.rows, rrows, (r.m + 1) * sizeof(int));
            memcpy(r.cols, rcols, r.nnz * sizeof(int));
        }
        memcpy(r.data, rdata, r.nnz * sizeof(double));
        dalloc->deallocate(rdata, a.nnz + b.nnz);
        ialloc->deallocate(rcols, a.nnz + b.nnz);
        ialloc->deallocate(rrows, a.m + 1);
        a.deallocate();
        a = r;
        if (conj)
            tmp.deallocate();
#endif
    }
    static void multiply(const CSRMatrixRef &a, bool conja,
                         const CSRMatrixRef &b, bool conjb, CSRMatrixRef &c,
                         double scale, double cfactor) {
        if (a.nnz == a.size() || b.nnz == b.size()) {
            if (c.nnz == c.size()) {
                if (a.nnz == a.size() && b.nnz == b.size())
                    MatrixFunctions::multiply(a.dense_ref(), conja,
                                              b.dense_ref(), conjb,
                                              c.dense_ref(), scale, cfactor);
                else if (a.nnz == a.size())
                    multiply(a.dense_ref(), conja, b, conjb, c.dense_ref(),
                             scale, cfactor);
                else
                    multiply(a, conja, b.dense_ref(), conjb, c.dense_ref(),
                             scale, cfactor);
            } else {
                MatrixRef cd(dalloc->allocate(c.size()), c.m, c.n);
                c.to_dense(cd);
                if (a.nnz == a.size() && b.nnz == b.size())
                    MatrixFunctions::multiply(a.dense_ref(), conja,
                                              b.dense_ref(), conjb, cd, scale,
                                              cfactor);
                else if (a.nnz == a.size())
                    multiply(a.dense_ref(), conja, b, conjb, cd, scale,
                             cfactor);
                else
                    multiply(a, conja, b.dense_ref(), conjb, cd, scale,
                             cfactor);
                c.deallocate();
                c.from_dense(cd);
                cd.deallocate();
            }
            return;
        } else if (c.nnz == c.size()) {
            MatrixRef bd(dalloc->allocate(b.size()), b.m, b.n);
            b.to_dense(bd);
            multiply(a, conja, bd, conjb, c.dense_ref(), scale, cfactor);
            bd.deallocate();
            return;
        }
        vector<CSRMatrixRef> tmps;
        int *arows = a.rows, *acols = a.cols, *brows = b.rows, *bcols = b.cols;
        double *adata = a.data, *bdata = b.data;
        const int am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        const int bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        assert(am == c.m && bn == c.n && an == bm);
        if (conja)
            tmps.push_back(a.transpose(dalloc)), arows = tmps.back().rows,
                                                 acols = tmps.back().cols,
                                                 adata = tmps.back().data;
        if (conjb)
            tmps.push_back(b.transpose(dalloc)), brows = tmps.back().rows,
                                                 bcols = tmps.back().cols,
                                                 bdata = tmps.back().data;
        int *r_idx = (int *)ialloc->allocate(c.m + 1);
        vector<int> tcols, rcols;
        vector<double> tdata, rdata;
        for (int i = 0, jp, jr, inc = 1; i < am; i++) {
            r_idx[i] = (int)rcols.size();
            jp = c.rows[i], jr = i == c.m - 1 ? c.nnz : c.rows[i + 1];
            jr = jr - jp;
            tcols.clear(), tdata.clear();
            if (jr != 0 && cfactor != 0) {
                tcols.resize(jr), tdata.resize(jr);
                memcpy(tcols.data(), c.cols + jp, jr * sizeof(int));
                memcpy(tdata.data(), c.data + jp, jr * sizeof(double));
                dscal(&jr, &cfactor, tdata.data(), &inc);
            }
            jp = arows[i], jr = i == am - 1 ? a.nnz : arows[i + 1];
            for (int j = jp; j < jr; j++) {
                int kp = brows[acols[j]],
                    kr = acols[j] == bm - 1 ? b.nnz : brows[acols[j] + 1];
                for (int k = kp; k < kr; k++)
                    tcols.push_back(bcols[k]),
                        tdata.push_back(adata[j] * bdata[k] * scale);
            }
            if (tcols.size() != 0) {
                int *idx = (int *)ialloc->allocate(tcols.size());
                for (int l = 0; l < (int)tcols.size(); l++)
                    idx[l] = l;
                sort(idx, idx + tcols.size(), [&tcols](int ti, int tj) {
                    return tcols[ti] < tcols[tj];
                });
                rcols.push_back(tcols[idx[0]]), rdata.push_back(tdata[idx[0]]);
                for (int l = 1; l < (int)tcols.size(); l++)
                    if (rcols.back() == tcols[idx[l]])
                        rdata.back() += tdata[idx[l]];
                    else if (abs(rdata.back()) < TINY)
                        rcols.back() = tcols[idx[l]],
                        rdata.back() = tdata[idx[l]];
                    else
                        rcols.push_back(tcols[idx[l]]),
                            rdata.push_back(tdata[idx[l]]);
                ialloc->deallocate(idx, tcols.size());
            }
        }
        r_idx[am] = (int)rcols.size();
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        CSRMatrixRef r(c.m, c.n, r_idx[am], nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        if (r.nnz != r.size()) {
            memcpy(r.rows, r_idx, (r.m + 1) * sizeof(int));
            memcpy(r.cols, rcols.data(), r.nnz * sizeof(int));
        }
        memcpy(r.data, rdata.data(), r.nnz * sizeof(double));
        c.deallocate();
        c = r;
        ialloc->deallocate(r_idx, c.m + 1);
        for (int it = conja + conjb - 1; it >= 0; it--)
            tmps[it].deallocate();
    }
    static void multiply(const MatrixRef &a, bool conja, const CSRMatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        if (b.nnz == b.size())
            return MatrixFunctions::multiply(a, conja, b.dense_ref(), conjb, c,
                                             scale, cfactor);
#ifdef _HAS_INTEL_MKL
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
#else
        const int am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        const int bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        assert(am == c.m && bn == c.n && an == bm);
        if (cfactor != 1)
            MatrixFunctions::iscale(c, cfactor);
        if (!conja && !conjb) {
            for (int ib = 0; ib < b.m; ib++) {
                const int jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (int jb = jbp; jb < jbr; jb++) {
                    const double factor = scale * b.data[jb];
                    daxpy(&a.m, &factor, &a(0, ib), &a.n, &c(0, b.cols[jb]),
                          &c.n);
                }
            }
        } else if (conja && !conjb) {
            const int inc = 1;
            for (int ib = 0; ib < b.m; ib++) {
                const int jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (int jb = jbp; jb < jbr; jb++) {
                    const double factor = scale * b.data[jb];
                    daxpy(&a.n, &factor, &a(ib, 0), &inc, &c(0, b.cols[jb]),
                          &c.n);
                }
            }
        } else if (!conja && conjb) {
            for (int ib = 0; ib < b.m; ib++) {
                const int jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (int jb = jbp; jb < jbr; jb++) {
                    const double factor = scale * b.data[jb];
                    daxpy(&a.m, &factor, &a(0, b.cols[jb]), &a.n, &c(0, ib),
                          &c.n);
                }
            }
        } else {
            const int inc = 1;
            for (int ib = 0; ib < b.m; ib++) {
                const int jbp = b.rows[ib], jbr = b.rows[ib + 1];
                for (int jb = jbp; jb < jbr; jb++) {
                    const double factor = scale * b.data[jb];
                    daxpy(&a.n, &factor, &a(b.cols[jb], 0), &inc, &c(0, ib),
                          &c.n);
                }
            }
        }
#endif
    }
    static void multiply(const CSRMatrixRef &a, bool conja, const MatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        if (a.nnz == a.size())
            return MatrixFunctions::multiply(a.dense_ref(), conja, b, conjb, c,
                                             scale, cfactor);
#ifdef _HAS_INTEL_MKL
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
#else
        const int am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        const int bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        const int inc = 1;
        assert(am == c.m && bn == c.n && an == bm);
        if (cfactor != 1)
            MatrixFunctions::iscale(c, cfactor);
        if (!conja && !conjb) {
            for (int ia = 0; ia < a.m; ia++) {
                const int jap = a.rows[ia], jar = a.rows[ia + 1];
                for (int ja = jap; ja < jar; ja++) {
                    const double factor = scale * a.data[ja];
                    daxpy(&b.n, &factor, &b(a.cols[ja], 0), &inc, &c(ia, 0),
                          &inc);
                }
            }
        } else if (conja && !conjb) {
            for (int ia = 0; ia < a.m; ia++) {
                const int jap = a.rows[ia], jar = a.rows[ia + 1];
                for (int ja = jap; ja < jar; ja++) {
                    const double factor = scale * a.data[ja];
                    daxpy(&b.n, &factor, &b(ia, 0), &inc, &c(a.cols[ja], 0),
                          &inc);
                }
            }
        } else if (!conja && conjb) {
            for (int ia = 0; ia < a.m; ia++) {
                const int jap = a.rows[ia], jar = a.rows[ia + 1];
                for (int ja = jap; ja < jar; ja++) {
                    const double factor = scale * a.data[ja];
                    daxpy(&b.m, &factor, &b(0, a.cols[ja]), &b.n, &c(ia, 0),
                          &inc);
                }
            }
        } else {
            for (int ia = 0; ia < a.m; ia++) {
                const int jap = a.rows[ia], jar = a.rows[ia + 1];
                for (int ja = jap; ja < jar; ja++) {
                    const double factor = scale * a.data[ja];
                    daxpy(&b.m, &factor, &b(0, ia), &b.n, &c(a.cols[ja], 0),
                          &inc);
                }
            }
        }
#endif
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
    // c = bra * a * ket(.T) for tensor product multiplication
    static void rotate(const MatrixRef &a, const MatrixRef &c,
                       const CSRMatrixRef &bra, bool conj_bra,
                       const MatrixRef &ket, bool conj_ket, double scale) {
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate();
        MatrixFunctions::multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate();
    }
    // c = bra * a * ket(.T) for tensor product multiplication
    static void rotate(const MatrixRef &a, const MatrixRef &c,
                       const MatrixRef &bra, bool conj_bra,
                       const CSRMatrixRef &ket, bool conj_ket, double scale) {
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate();
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        MatrixFunctions::multiply(bra, conj_bra, work, false, c, scale, 1.0);
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
        if (a.nnz == a.size() || b.nnz == b.size()) {
            if (a.nnz == a.size())
                tensor_product(a.dense_ref(), conja, b, conjb, c, scale,
                               stride);
            else
                tensor_product(a, conja, b.dense_ref(), conjb, c, scale,
                               stride);
            return;
        } else if (c.nnz == c.size()) {
            MatrixRef ad = MatrixRef(dalloc->allocate(a.size()), a.m, a.n);
            a.to_dense(ad);
            MatrixRef bd = MatrixRef(dalloc->allocate(b.size()), b.m, b.n);
            b.to_dense(bd);
            MatrixFunctions::tensor_product(ad, conja, bd, conjb, c.dense_ref(),
                                            scale, stride);
            ad.deallocate();
            bd.deallocate();
            return;
        }
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        CSRMatrixRef r(c.m, c.n, a.nnz * b.nnz, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        uint32_t m_length = (conja ? a.n : a.m) * (conjb ? b.n : b.m);
        memset(r.rows, 0, (r.m + 1) * sizeof(int));
        vector<CSRMatrixRef> tmps;
        int *arows = a.rows, *acols = a.cols, *brows = b.rows, *bcols = b.cols;
        double *adata = a.data, *bdata = b.data;
        int am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        int bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        if (conja)
            tmps.push_back(a.transpose(dalloc)), arows = tmps.back().rows,
                                                 acols = tmps.back().cols,
                                                 adata = tmps.back().data;
        if (conjb)
            tmps.push_back(b.transpose(dalloc)), brows = tmps.back().rows,
                                                 bcols = tmps.back().cols,
                                                 bdata = tmps.back().data;
        for (int ir = m_stride, ix = 0, ia = 0; ia < am; ia++) {
            int jap = arows[ia], jar = ia == am - 1 ? a.nnz : arows[ia + 1];
            for (int ib = 0; ib < bm; ib++, ir++) {
                int jbp = brows[ib], jbr = ib == bm - 1 ? b.nnz : brows[ib + 1];
                r.rows[ir + 1] = r.rows[ir] + (jar - jap) * (jbr - jbp);
                for (int ja = jap; ja < jar; ja++)
                    for (int jb = jbp; jb < jbr; jb++, ix++) {
                        r.cols[ix] = bn * acols[ja] + bcols[jb] + n_stride;
                        r.data[ix] = scale * adata[ja] * bdata[jb];
                    }
            }
        }
        for (int it = conja + conjb - 1; it >= 0; it--)
            tmps[it].deallocate();
        assert(r.rows[m_stride + m_length] == r.nnz);
        for (int ir = m_stride + m_length; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0) {
            c.deallocate();
            c = r;
        } else {
            iadd(c, r, 1.0, false);
            r.deallocate();
        }
    }
    static void tensor_product(const CSRMatrixRef &a, bool conja,
                               const MatrixRef &b, bool conjb, CSRMatrixRef &c,
                               double scale, uint32_t stride) {
        if (a.nnz == a.size() || c.nnz == c.size()) {
            MatrixRef ad =
                a.nnz == a.size()
                    ? a.dense_ref()
                    : MatrixRef(dalloc->allocate(a.size()), a.m, a.n);
            if (a.nnz != a.size())
                a.to_dense(ad);
            MatrixRef cd =
                c.nnz == c.size()
                    ? c.dense_ref()
                    : MatrixRef(dalloc->allocate(c.size()), c.m, c.n);
            if (c.nnz != c.size())
                c.to_dense(cd);
            MatrixFunctions::tensor_product(ad, conja, b, conjb, cd, scale,
                                            stride);
            if (a.nnz != a.size())
                ad.deallocate();
            if (c.nnz != c.size()) {
                c.deallocate();
                c.from_dense(cd);
                cd.deallocate();
            }
            return;
        }
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        CSRMatrixRef r(c.m, c.n, a.nnz * b.m * b.n, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        uint32_t m_length = (conja ? a.n : a.m) * (conjb ? b.n : b.m);
        memset(r.rows, 0, (r.m + 1) * sizeof(int));
        CSRMatrixRef tmp;
        int *arows = a.rows, *acols = a.cols;
        double *adata = a.data;
        int am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        int bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        if (conja)
            tmp = a.transpose(dalloc), arows = tmp.rows, acols = tmp.cols,
            adata = tmp.data;
        for (int ir = m_stride, ix = 0, ia = 0; ia < am; ia++) {
            int jap = arows[ia], jar = ia == am - 1 ? a.nnz : arows[ia + 1];
            for (int ib = 0; ib < bm; ib++, ir++) {
                r.rows[ir + 1] = r.rows[ir] + (jar - jap) * bn;
                if (conjb)
                    for (int ja = jap; ja < jar; ja++)
                        for (int jb = 0; jb < bn; jb++, ix++) {
                            r.cols[ix] = bn * acols[ja] + jb + n_stride;
                            r.data[ix] =
                                scale * adata[ja] * b.data[jb * bm + ib];
                        }
                else
                    for (int ja = jap; ja < jar; ja++)
                        for (int jb = 0; jb < bn; jb++, ix++) {
                            r.cols[ix] = bn * acols[ja] + jb + n_stride;
                            r.data[ix] =
                                scale * adata[ja] * b.data[ib * bn + jb];
                        }
            }
        }
        if (conja)
            tmp.deallocate();
        assert(r.rows[m_stride + m_length] == r.nnz);
        for (int ir = m_stride + m_length; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0) {
            c.deallocate();
            c = r;
        } else {
            iadd(c, r, 1.0, false);
            r.deallocate();
        }
    }
    static void tensor_product(const MatrixRef &a, bool conja,
                               const CSRMatrixRef &b, bool conjb,
                               CSRMatrixRef &c, double scale, uint32_t stride) {
        if (b.nnz == b.size() || c.nnz == c.size()) {
            MatrixRef bd =
                b.nnz == b.size()
                    ? b.dense_ref()
                    : MatrixRef(dalloc->allocate(b.size()), b.m, b.n);
            if (b.nnz != b.size())
                b.to_dense(bd);
            MatrixRef cd =
                c.nnz == c.size()
                    ? c.dense_ref()
                    : MatrixRef(dalloc->allocate(c.size()), c.m, c.n);
            if (c.nnz != c.size())
                c.to_dense(cd);
            MatrixFunctions::tensor_product(a, conja, bd, conjb, cd, scale,
                                            stride);
            if (b.nnz != b.size())
                bd.deallocate();
            if (c.nnz != c.size()) {
                c.deallocate();
                c.from_dense(cd);
                cd.deallocate();
            }
            return;
        }
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        CSRMatrixRef r(c.m, c.n, a.m * a.n * b.nnz, nullptr, nullptr, nullptr);
        r.alloc = d_alloc;
        r.allocate();
        uint32_t m_stride = stride / c.n, n_stride = stride % c.n;
        uint32_t m_length = (conja ? a.n : a.m) * (conjb ? b.n : b.m);
        memset(r.rows, 0, (r.m + 1) * sizeof(int));
        CSRMatrixRef tmp;
        int *brows = b.rows, *bcols = b.cols;
        double *bdata = b.data;
        int am = conja ? a.n : a.m, an = conja ? a.m : a.n;
        int bm = conjb ? b.n : b.m, bn = conjb ? b.m : b.n;
        if (conjb)
            tmp = b.transpose(dalloc), brows = tmp.rows, bcols = tmp.cols,
            bdata = tmp.data;
        for (int ir = m_stride, ix = 0, ia = 0; ia < am; ia++) {
            for (int ib = 0; ib < bm; ib++, ir++) {
                int jbp = brows[ib], jbr = ib == bm - 1 ? b.nnz : brows[ib + 1];
                r.rows[ir + 1] = r.rows[ir] + an * (jbr - jbp);
                if (conja)
                    for (int ja = 0; ja < an; ja++)
                        for (int jb = jbp; jb < jbr; jb++, ix++) {
                            r.cols[ix] = bn * ja + bcols[jb] + n_stride;
                            r.data[ix] =
                                scale * a.data[ja * am + ia] * bdata[jb];
                        }
                else
                    for (int ja = 0; ja < an; ja++)
                        for (int jb = jbp; jb < jbr; jb++, ix++) {
                            r.cols[ix] = bn * ja + bcols[jb] + n_stride;
                            r.data[ix] =
                                scale * a.data[ia * an + ja] * bdata[jb];
                        }
            }
        }
        if (conjb)
            tmp.deallocate();
        assert(r.rows[m_stride + m_length] == r.nnz);
        for (int ir = m_stride + m_length; ir < r.m; ir++)
            r.rows[ir + 1] = r.nnz;
        if (c.nnz == 0) {
            c.deallocate();
            c = r;
        } else {
            iadd(c, r, 1.0, false);
            r.deallocate();
        }
    }
};

} // namespace block2
