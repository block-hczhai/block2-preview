
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
#include "matrix.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

extern "C" {

#ifndef _HAS_INTEL_MKL

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void dscal(const MKL_INT *n, const double *sa, double *sx,
                  const MKL_INT *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void dcopy(const MKL_INT *n, const double *dx, const MKL_INT *incx,
                  double *dy, const MKL_INT *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + double [sa] * vector [sx]
extern void daxpy(const MKL_INT *n, const double *sa, const double *sx,
                  const MKL_INT *incx, double *sy,
                  const MKL_INT *incy) noexcept;

// vector dot product
extern double ddot(const MKL_INT *n, const double *dx, const MKL_INT *incx,
                   const double *dy, const MKL_INT *incy) noexcept;

// Euclidean norm of a vector
extern double dnrm2(const MKL_INT *n, const double *x,
                    const MKL_INT *incx) noexcept;

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void dgemm(const char *transa, const char *transb, const MKL_INT *m,
                  const MKL_INT *n, const MKL_INT *k, const double *alpha,
                  const double *a, const MKL_INT *lda, const double *b,
                  const MKL_INT *ldb, const double *beta, double *c,
                  const MKL_INT *ldc) noexcept;

// matrix-vector multiplication
// vec [y] = double [alpha] * mat [a] * vec [x] + double [beta] * vec [y]
extern void dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const double *alpha, const double *a, const MKL_INT *lda,
                  const double *x, const MKL_INT *incx, const double *beta,
                  double *y, const MKL_INT *incy) noexcept;

// linear system a * x = b
extern void dgesv(const MKL_INT *n, const MKL_INT *nrhs, double *a,
                  const MKL_INT *lda, MKL_INT *ipiv, double *b,
                  const MKL_INT *ldb, MKL_INT *info);

// QR factorization
extern void dgeqrf(const MKL_INT *m, const MKL_INT *n, double *a,
                   const MKL_INT *lda, double *tau, double *work,
                   const MKL_INT *lwork, MKL_INT *info);
extern void dorgqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   double *a, const MKL_INT *lda, const double *tau,
                   double *work, const MKL_INT *lwork, MKL_INT *info);

// LQ factorization
extern void dgelqf(const MKL_INT *m, const MKL_INT *n, double *a,
                   const MKL_INT *lda, double *tau, double *work,
                   const MKL_INT *lwork, MKL_INT *info);
extern void dorglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   double *a, const MKL_INT *lda, const double *tau,
                   double *work, const MKL_INT *lwork, MKL_INT *info);

// eigenvalue problem
extern void dsyev(const char *jobz, const char *uplo, const MKL_INT *n,
                  double *a, const MKL_INT *lda, double *w, double *work,
                  const MKL_INT *lwork, MKL_INT *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void dgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, double *a, const MKL_INT *lda, double *s,
                   double *u, const MKL_INT *ldu, double *vt,
                   const MKL_INT *ldvt, double *work, const MKL_INT *lwork,
                   MKL_INT *info);

#endif
}

// Dense matrix operations
struct MatrixFunctions {
    // a = b
    static void copy(const MatrixRef &a, const MatrixRef &b,
                     const MKL_INT inca = 1, const MKL_INT incb = 1) {
        assert(a.m == b.m && a.n == b.n);
        const MKL_INT n = a.m * a.n;
        dcopy(&n, b.data, &incb, a.data, &inca);
    }
    static void iscale(const MatrixRef &a, double scale,
                       const MKL_INT inc = 1) {
        MKL_INT n = a.m * a.n;
        dscal(&n, &scale, a.data, &inc);
    }
    // a = a + scale * op(b)
    static void iadd(const MatrixRef &a, const MatrixRef &b, double scale,
                     bool conj = false) {
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            MKL_INT n = a.m * a.n, inc = 1;
            daxpy(&n, &scale, b.data, &inc, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n == b.m);
            for (MKL_INT i = 0, inc = 1; i < a.m; i++)
                daxpy(&a.n, &scale, b.data + i, &a.m, a.data + i * a.n, &inc);
        }
    }
    static double norm(const MatrixRef &a) {
        MKL_INT n = a.m * a.n, inc = 1;
        return dnrm2(&n, a.data, &inc);
    }
    static double dot(const MatrixRef &a, const MatrixRef &b) {
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        return ddot(&n, a.data, &inc, b.data, &inc);
    }
    template <typename T1, typename T2>
    static bool all_close(const T1 &a, const T2 &b, double atol = 1E-8,
                          double rtol = 1E-5, double scale = 1.0) {
        assert(a.m == b.m && a.n == b.n);
        for (MKL_INT i = 0; i < a.m; i++)
            for (MKL_INT j = 0; j < a.n; j++)
                if (abs(a(i, j) - scale * b(i, j)) > atol + rtol * abs(b(i, j)))
                    return false;
        return true;
    }
    // solve a^T x[i, :] = b[i, :] => output in b; a will be overwritten
    static void linear(const MatrixRef &a, const MatrixRef &b) {
        assert(a.m == a.n && a.m == b.n);
        MKL_INT *work = (MKL_INT *)ialloc->allocate(a.n * _MINTSZ), info = -1;
        dgesv(&a.m, &b.m, a.data, &a.n, work, b.data, &a.n, &info);
        assert(info == 0);
        ialloc->deallocate(work, a.n * _MINTSZ);
    }
    // c.n is used for ldc; a.n is used for lda
    static void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        if (!conja && !conjb) {
            assert(a.n >= b.m && c.m == a.m && c.n >= b.n);
            dgemm("n", "n", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (!conja && conjb) {
            assert(a.n >= b.n && c.m == a.m && c.n >= b.m);
            dgemm("t", "n", &b.m, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (conja && !conjb) {
            assert(a.m == b.m && c.m <= a.n && c.n >= b.n);
            dgemm("n", "t", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else {
            assert(a.m == b.n && c.m <= a.n && c.n >= b.m);
            dgemm("t", "t", &b.m, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        }
    }
    // c = bra * a * ket(.T)
    // return nflop
    static size_t rotate(const MatrixRef &a, const MatrixRef &c,
                         const MatrixRef &bra, bool conj_bra,
                         const MatrixRef &ket, bool conj_ket, double scale) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate(d_alloc);
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate(d_alloc);
        return (size_t)ket.m * ket.n * work.m + (size_t)work.m * work.n * c.m;
    }
    // dleft == true : c = bra (= da x db) * a * ket
    // dleft == false: c = bra * a * ket (= da x db)
    // return nflop
    static size_t three_rotate(const MatrixRef &a, const MatrixRef &c,
                               const MatrixRef &bra, bool conj_bra,
                               const MatrixRef &ket, bool conj_ket,
                               const MatrixRef &da, bool dconja,
                               const MatrixRef &db, bool dconjb, bool dleft,
                               double scale, uint32_t stride) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            MatrixRef work(nullptr, am, conj_ket ? ket.m : ket.n);
            work.allocate(d_alloc);
            // work = a * ket
            multiply(MatrixRef(&a(ast, 0), am, a.n), false, ket, conj_ket, work,
                     1.0, 0.0);
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * work
                multiply(db, dconjb, work, false,
                         MatrixRef(&c(cst, 0), cm, c.n), scale * *da.data, 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * work
                multiply(da, dconja, work, false,
                         MatrixRef(&c(cst, 0), cm, c.n), scale * *db.data, 1.0);
            else
                assert(false);
            work.deallocate(d_alloc);
            return (size_t)ket.m * ket.n * work.m +
                   (size_t)work.m * work.n * cm;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            MatrixRef work(nullptr, a.m, kn);
            work.allocate(d_alloc);
            if (da.m == 1 && da.n == 1)
                // work = a * (1 x db)
                multiply(MatrixRef(&a(0, ast), a.m, a.n), false, db, dconjb,
                         work, *da.data * scale, 0.0);
            else if (db.m == 1 && db.n == 1)
                // work = a * (da x 1)
                multiply(MatrixRef(&a(0, ast), a.m, a.n), false, da, dconja,
                         work, *db.data * scale, 0.0);
            else
                assert(false);
            // c = bra * work
            multiply(bra, conj_bra, work, false,
                     MatrixRef(&c(0, cst), c.m, c.n), 1.0, 1.0);
            work.deallocate(d_alloc);
            return (size_t)km * kn * work.m + (size_t)work.m * work.n * c.m;
        }
    }
    // dleft == true : c = a * ket
    // dleft == false: c = a * ket (= da x db)
    // return nflop
    static size_t three_rotate_tr_left(const MatrixRef &a, const MatrixRef &c,
                                       const MatrixRef &bra, bool conj_bra,
                                       const MatrixRef &ket, bool conj_ket,
                                       const MatrixRef &da, bool dconja,
                                       const MatrixRef &db, bool dconjb,
                                       bool dleft, double scale,
                                       uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            multiply(MatrixRef(&a(ast, 0), am, a.n), false, ket, conj_ket,
                     MatrixRef(&c(cst, 0), cm, c.n), scale, 1.0);
            return (size_t)ket.m * ket.n * am;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            if (da.m == 1 && da.n == 1)
                // c = a * (1 x db)
                multiply(MatrixRef(&a(0, ast), a.m, a.n), false, db, dconjb,
                         MatrixRef(&c(0, cst), c.m, c.n), *da.data * scale,
                         1.0);
            else if (db.m == 1 && db.n == 1)
                // c = a * (da x 1)
                multiply(MatrixRef(&a(0, ast), a.m, a.n), false, da, dconja,
                         MatrixRef(&c(0, cst), c.m, c.n), *db.data * scale,
                         1.0);
            else
                assert(false);
            return (size_t)km * kn * c.m;
        }
    }
    // dleft == true : c = bra (= da x db) * a
    // dleft == false: c = bra * a
    // return nflop
    static size_t three_rotate_tr_right(const MatrixRef &a, const MatrixRef &c,
                                        const MatrixRef &bra, bool conj_bra,
                                        const MatrixRef &ket, bool conj_ket,
                                        const MatrixRef &da, bool dconja,
                                        const MatrixRef &db, bool dconjb,
                                        bool dleft, double scale,
                                        uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * a
                multiply(db, dconjb, MatrixRef(&a(ast, 0), am, a.n), false,
                         MatrixRef(&c(cst, 0), cm, c.n), scale * *da.data, 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * a
                multiply(da, dconja, MatrixRef(&a(ast, 0), am, a.n), false,
                         MatrixRef(&c(cst, 0), cm, c.n), scale * *db.data, 1.0);
            else
                assert(false);
            return (size_t)am * a.n * cm;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            const double cfactor = 1.0;
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            dgemm("n", conj_bra ? "t" : "n", &kn, &c.m, &a.m, &scale,
                  &a(0, ast), &a.n, bra.data, &bra.n, &cfactor, &c(0, cst),
                  &c.n);
            return (size_t)a.m * a.n * c.m;
        }
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                        const MatrixRef &c, double scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const MKL_INT k = 1, lda = a.n + 1, ldb = b.n + 1;
        dgemm("t", "n", &b.n, &a.n, &k, &scale, b.data, &ldb, a.data, &lda,
              &cfactor, c.data, &c.n);
    }
    // diagonal element of three-matrix tensor product
    static void
    three_tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                  const MatrixRef &c, const MatrixRef &da,
                                  bool dconja, const MatrixRef &db, bool dconjb,
                                  bool dleft, double scale, uint32_t stride) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const MKL_INT dstrm = (MKL_INT)stride / (dleft ? a.m : b.m);
        const MKL_INT dstrn = (MKL_INT)stride % (dleft ? a.m : b.m);
        if (dstrn != dstrm)
            return;
        assert(da.m == da.n && db.m == db.n);
        const MKL_INT ddstr = 0;
        const MKL_INT k = 1, lda = a.n + 1, ldb = b.n + 1;
        const MKL_INT ldda = da.n + 1, lddb = db.n + 1;
        if (da.m == 1 && da.n == 1) {
            scale *= *da.data;
            const MKL_INT dn = db.n - abs(ddstr);
            const double *bdata =
                dconjb ? &db(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &db(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft)
                    // (1 x db) x b
                    dgemm("t", "n", &b.n, &dn, &k, &scale, b.data, &ldb, bdata,
                          &lddb, &cfactor, &c(max(dstrn, dstrm), (MKL_INT)0),
                          &c.n);
                else
                    // a x (1 x db)
                    dgemm("t", "n", &dn, &a.n, &k, &scale, bdata, &lddb, a.data,
                          &lda, &cfactor, &c(0, max(dstrn, dstrm)), &c.n);
            }
        } else if (db.m == 1 && db.n == 1) {
            scale *= *db.data;
            const MKL_INT dn = da.n - abs(ddstr);
            const double *adata =
                dconja ? &da(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &da(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft)
                    // (da x 1) x b
                    dgemm("t", "n", &b.n, &dn, &k, &scale, b.data, &ldb, adata,
                          &ldda, &cfactor, &c(max(dstrn, dstrm), (MKL_INT)0),
                          &c.n);
                else
                    // a x (da x 1)
                    dgemm("t", "n", &dn, &a.n, &k, &scale, adata, &ldda, a.data,
                          &lda, &cfactor, &c(0, max(dstrn, dstrm)), &c.n);
            }
        } else
            assert(false);
    }
    static void tensor_product(const MatrixRef &a, bool conja,
                               const MatrixRef &b, bool conjb,
                               const MatrixRef &c, double scale,
                               uint32_t stride) {
        const double cfactor = 1.0;
        switch (conja | (conjb << 1)) {
        case 0:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const MKL_INT n = b.m * b.n;
                    dgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (MKL_INT k = 0; k < b.m; k++)
                        dgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const MKL_INT n = a.m * a.n;
                    dgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (MKL_INT k = 0; k < a.m; k++)
                        dgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++) {
                        const double factor = scale * a(i, j);
                        for (MKL_INT k = 0; k < b.m; k++)
                            daxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 1:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const MKL_INT n = b.m * b.n;
                    dgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (MKL_INT k = 0; k < b.m; k++)
                        dgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                assert(a.m <= c.n);
                for (MKL_INT k = 0; k < a.n; k++)
                    dgemm("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const double factor = scale * a(j, i);
                        for (MKL_INT k = 0; k < b.m; k++)
                            daxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 2:
            if (a.m == 1 && a.n == 1) {
                assert(b.m <= c.n);
                for (MKL_INT k = 0; k < b.n; k++)
                    dgemm("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const MKL_INT n = a.m * a.n;
                    dgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (MKL_INT k = 0; k < a.m; k++)
                        dgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (MKL_INT i = 0, incb = b.n, inc = 1; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++) {
                        const double factor = scale * a(i, j);
                        for (MKL_INT k = 0; k < b.n; k++)
                            daxpy(&b.m, &factor, &b(0, k), &incb,
                                  &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        case 1 | 2:
            if (a.m == 1 && a.n == 1) {
                for (MKL_INT k = 0; k < b.n; k++)
                    dgemm("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                for (MKL_INT k = 0; k < a.n; k++)
                    dgemm("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (MKL_INT i = 0, incb = b.n, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const double factor = scale * a(j, i);
                        for (MKL_INT k = 0; k < b.n; k++)
                            daxpy(&b.m, &factor, &b(0, k), &incb,
                                  &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        default:
            assert(false);
        }
    }
    // SVD; original matrix will be destroyed
    static void svd(const MatrixRef &a, const MatrixRef &l, const MatrixRef &s,
                    const MatrixRef &r) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MKL_INT k = min(a.m, a.n), info = 0, lwork = 34 * max(a.m, a.n);
        // double work[lwork];
        double *work = d_alloc->allocate(lwork);
        assert(a.m == l.m && a.n == r.n && l.n == k && r.m == k && s.n == k);
        dgesvd("S", "S", &a.n, &a.m, a.data, &a.n, s.data, r.data, &a.n, l.data,
               &k, work, &lwork, &info);
        assert(info == 0);
        d_alloc->deallocate(work, lwork);
    }
    // LQ factorization
    static void lq(const MatrixRef &a, const MatrixRef &l, const MatrixRef &q) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.m;
        // double work[lwork], tau[k], t[a.m * a.n];
        double *work = d_alloc->allocate(lwork);
        double *tau = d_alloc->allocate(k);
        double *t = d_alloc->allocate(a.m * a.n);
        assert(a.m == l.m && a.n == q.n && l.n == k && q.m == k);
        memcpy(t, a.data, sizeof(double) * a.m * a.n);
        dgeqrf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(l.data, 0, sizeof(double) * k * a.m);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(l.data + j * k, t + j * a.n, sizeof(double) * min(j + 1, k));
        dorgqr(&a.n, &k, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memcpy(q.data, t, sizeof(double) * k * a.n);
        d_alloc->deallocate(t, a.m * a.n);
        d_alloc->deallocate(tau, k);
        d_alloc->deallocate(work, lwork);
    }
    // QR factorization
    static void qr(const MatrixRef &a, const MatrixRef &q, const MatrixRef &r) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.n;
        // double work[lwork], tau[k], t[a.m * a.n];
        double *work = d_alloc->allocate(lwork);
        double *tau = d_alloc->allocate(k);
        double *t = d_alloc->allocate(a.m * a.n);
        assert(a.m == q.m && a.n == r.n && q.n == k && r.m == k);
        memcpy(t, a.data, sizeof(double) * a.m * a.n);
        dgelqf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(r.data, 0, sizeof(double) * k * a.n);
        for (MKL_INT j = 0; j < k; j++)
            memcpy(r.data + j * a.n + j, t + j * a.n + j,
                   sizeof(double) * (a.n - j));
        dorglq(&k, &a.m, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(q.data + j * k, t + j * a.n, sizeof(double) * k);
        d_alloc->deallocate(t, a.m * a.n);
        d_alloc->deallocate(tau, k);
        d_alloc->deallocate(work, lwork);
    }
    // eigenvectors are row vectors
    static void eigs(const MatrixRef &a, const DiagonalMatrix &w) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(a.m == a.n && w.n == a.n);
        MKL_INT lwork = 34 * a.n, info;
        // double work[lwork];
        double *work = d_alloc->allocate(lwork);
        dsyev("V", "U", &a.n, a.data, &a.n, w.data, work, &lwork, &info);
        assert(info == 0);
        d_alloc->deallocate(work, lwork);
    }
    // z = r / aa
    static void precondition(const MatrixRef &z, const MatrixRef &r,
                             const DiagonalMatrix &aa) {
        copy(z, r);
        if (aa.size() != 0) {
            assert(aa.size() == r.size() && r.size() == z.size());
            for (MKL_INT i = 0; i < aa.n; i++)
                if (abs(aa.data[i]) > 1E-12)
                    z.data[i] /= aa.data[i];
        }
    }
    static void olsen_precondition(const MatrixRef &q, const MatrixRef &c,
                                   double ld, const DiagonalMatrix &aa) {
        assert(aa.size() == c.size());
        MatrixRef t(nullptr, c.m, c.n);
        t.allocate();
        copy(t, c);
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                t.data[i] /= ld - aa.data[i];
        iadd(q, c, -dot(t, q) / dot(c, t));
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                q.data[i] /= ld - aa.data[i];
        t.deallocate();
    }
    // Davidson algorithm
    // aa: diag elements of a (for precondition)
    // bs: input/output vector
    template <typename MatMul, typename PComm>
    static vector<double>
    davidson(MatMul &op, const DiagonalMatrix &aa, vector<MatrixRef> &vs,
             int &ndav, bool iprint = false, const PComm &pcomm = nullptr,
             double conv_thrd = 5E-6, int max_iter = 5000,
             int soft_max_iter = -1, int deflation_min_size = 2,
             int deflation_max_size = 50) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        int k = (int)vs.size();
        if (deflation_min_size < k)
            deflation_min_size = k;
        if (deflation_max_size < k + k / 2)
            deflation_max_size = k + k / 2;
        MatrixRef pbs(nullptr, deflation_max_size * vs[0].size(), 1);
        MatrixRef pss(nullptr, deflation_max_size * vs[0].size(), 1);
        pbs.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        pss.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        vector<MatrixRef> bs(deflation_max_size,
                             MatrixRef(nullptr, vs[0].m, vs[0].n));
        vector<MatrixRef> sigmas(deflation_max_size,
                                 MatrixRef(nullptr, vs[0].m, vs[0].n));
        for (int i = 0; i < deflation_max_size; i++) {
            bs[i].data = pbs.data + bs[i].size() * i;
            sigmas[i].data = pss.data + sigmas[i].size() * i;
        }
        for (int i = 0; i < k; i++)
            copy(bs[i], vs[i]);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++)
                iadd(bs[i], bs[j], -dot(bs[j], bs[i]));
            iscale(bs[i], 1.0 / sqrt(dot(bs[i], bs[i])));
        }
        vector<double> eigvals(k);
        MatrixRef q(nullptr, bs[0].m, bs[0].n);
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.allocate();
        int ck = 0, msig = 0, m = k, xiter = 0;
        double qq;
        if (iprint)
            cout << endl;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (pcomm != nullptr && xiter != 1)
                pcomm->broadcast(pbs.data + bs[0].size() * msig,
                                 bs[0].size() * (m - msig), pcomm->root);
            for (int i = msig; i < m; i++, msig++) {
                sigmas[i].clear();
                op(bs[i], sigmas[i]);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                DiagonalMatrix ld(nullptr, m);
                MatrixRef alpha(nullptr, m, m);
                ld.allocate();
                alpha.allocate();
                vector<MatrixRef> tmp(m, MatrixRef(nullptr, bs[0].m, bs[0].n));
                for (int i = 0; i < m; i++)
                    tmp[i].allocate();
                int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
                {
#pragma omp for schedule(dynamic) collapse(2)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++)
                            if (j <= i)
                                alpha(i, j) = dot(bs[i], sigmas[j]);
#pragma omp single
                    eigs(alpha, ld);
                    // note alpha row/column is diff from python
                    // b[1:m] = np.dot(b[:], alpha[:, 1:m])
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], bs[j]);
                        iscale(bs[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(bs[j], tmp[i], alpha(j, i));
                    // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], sigmas[j]);
                        iscale(sigmas[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(sigmas[j], tmp[i], alpha(j, i));
                }
                threading->activate_normal();
                for (int i = m - 1; i >= 0; i--)
                    tmp[i].deallocate();
                alpha.deallocate();
                for (int i = 0; i < ck; i++) {
                    copy(q, sigmas[i]);
                    iadd(q, bs[i], -ld(i, i));
                    if (dot(q, q) >= conv_thrd) {
                        ck = i;
                        break;
                    }
                }
                copy(q, sigmas[ck]);
                iadd(q, bs[ck], -ld(ck, ck));
                qq = dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8) << ld.data[ck]
                         << scientific << setw(13) << setprecision(2) << qq
                         << endl;
                olsen_precondition(q, bs[ck], ld.data[ck], aa);
                eigvals.resize(ck + 1);
                if (ck + 1 != 0)
                    memcpy(eigvals.data(), ld.data, (ck + 1) * sizeof(double));
                ld.deallocate();
            }
            if (pcomm != nullptr) {
                pcomm->broadcast(&qq, 1, pcomm->root);
                pcomm->broadcast(&ck, 1, pcomm->root);
            }
            if (qq < conv_thrd) {
                ck++;
                if (ck == k)
                    break;
            } else {
                if (m >= deflation_max_size)
                    m = msig = deflation_min_size;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    for (int j = 0; j < m; j++)
                        iadd(q, bs[j], -dot(bs[j], q));
                    iscale(q, 1.0 / sqrt(dot(q, q)));
                    copy(bs[m], q);
                }
                m++;
            }
            if (xiter == soft_max_iter)
                break;
        }
        if (xiter == soft_max_iter)
            eigvals.resize(k, 0);
        if (xiter == max_iter) {
            cout << "Error : only " << ck << " converged!" << endl;
            assert(false);
        }
        if (pcomm != nullptr) {
            pcomm->broadcast(eigvals.data(), eigvals.size(), pcomm->root);
            pcomm->broadcast(pbs.data, bs[0].size() * k, pcomm->root);
        }
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.deallocate();
        for (int i = 0; i < k; i++)
            copy(vs[i], bs[i]);
        d_alloc->deallocate(pss.data, deflation_max_size * vs[0].size());
        d_alloc->deallocate(pbs.data, deflation_max_size * vs[0].size());
        ndav = xiter;
        return eigvals;
    }
    // Computes exp(t*H), the matrix exponential of a general matrix in
    // full, using the irreducible rational Pade approximation
    // Adapted from expokit fortran code dgpadm.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = 4 * m * m + ideg + 1
    // exp(tH) is located at work[ret:ret+m*m]
    static pair<MKL_INT, MKL_INT> expo_pade(MKL_INT ideg, MKL_INT m,
                                            const double *h, MKL_INT ldh,
                                            double t, double *work) {
        static const double zero = 0.0, one = 1.0, mone = -1.0, two = 2.0;
        static const MKL_INT inc = 1;
        // check restrictions on input parameters
        MKL_INT mm = m * m;
        MKL_INT iflag = 0;
        assert(ldh >= m);
        // initialize pointers
        MKL_INT icoef = 0, ih2 = icoef + (ideg + 1), ip = ih2 + mm,
                iq = ip + mm, ifree = iq + mm;
        // scaling: seek ns such that ||t*H/2^ns|| < 1/2;
        // and set scale = t/2^ns ...
        memset(work, 0, sizeof(double) * m);
        for (MKL_INT j = 0; j < m; j++)
            for (MKL_INT i = 0; i < m; i++)
                work[i] += abs(h[j * m + i]);
        double hnorm = 0.0;
        for (MKL_INT i = 0; i < m; i++)
            hnorm = max(hnorm, work[i]);
        hnorm = abs(t * hnorm);
        if (hnorm == 0.0) {
            cerr << "Error - null H in expo pade" << endl;
            abort();
        }
        MKL_INT ns = max((MKL_INT)0, (MKL_INT)(log(hnorm) / log(2.0)) + 2);
        double scale = t / (double)(1LL << ns);
        double scale2 = scale * scale;
        // compute Pade coefficients
        MKL_INT i = ideg + 1, j = 2 * ideg + 1;
        work[icoef] = 1.0;
        for (MKL_INT k = 1; k <= ideg; k++)
            work[icoef + k] =
                work[icoef + k - 1] * (double)(i - k) / double(k * (j - k));
        // H2 = scale2*H*H ...
        dgemm("n", "n", &m, &m, &m, &scale2, h, &ldh, h, &ldh, &zero,
              work + ih2, &m);
        // initialize p (numerator) and q (denominator)
        memset(work + ip, 0, sizeof(double) * mm * 2);
        double cp = work[icoef + ideg - 1];
        double cq = work[icoef + ideg];
        for (MKL_INT j = 0; j < m; j++)
            work[ip + j * (m + 1)] = cp, work[iq + j * (m + 1)] = cq;
        // Apply Horner rule
        MKL_INT iodd = 1;
        for (MKL_INT k = ideg - 1; k > 0; k--) {
            MKL_INT iused = iodd * iq + (1 - iodd) * ip;
            dgemm("n", "n", &m, &m, &m, &one, work + iused, &m, work + ih2, &m,
                  &zero, work + ifree, &m);
            for (MKL_INT j = 0; j < m; j++)
                work[ifree + j * (m + 1)] += work[icoef + k - 1];
            ip = (1 - iodd) * ifree + iodd * ip;
            iq = iodd * ifree + (1 - iodd) * iq;
            ifree = iused;
            iodd = 1 - iodd;
        }
        // Obtain (+/-)(I + 2*(p\q))
        MKL_INT *iqp = iodd ? &iq : &ip;
        dgemm("n", "n", &m, &m, &m, &scale, work + *iqp, &m, h, &ldh, &zero,
              work + ifree, &m);
        *iqp = ifree;
        daxpy(&mm, &mone, work + ip, &inc, work + iq, &inc);
        dgesv(&m, &m, work + iq, &m, (MKL_INT *)work + ih2, work + ip, &m,
              &iflag);
        if (iflag != 0) {
            cerr << "Problem in DGESV in expo pade" << endl;
            abort();
        }
        dscal(&mm, &two, work + ip, &inc);
        for (MKL_INT j = 0; j < m; j++)
            work[ip + j * (m + 1)]++;
        MKL_INT iput = ip;
        if (ns == 0 && iodd) {
            dscal(&mm, &mone, work + ip, &inc);
        } else {
            // squaring : exp(t*H) = (exp(t*H))^(2^ns)
            iodd = 1;
            for (MKL_INT k = 0; k < ns; k++) {
                MKL_INT iget = iodd * ip + (1 - iodd) * iq;
                iput = (1 - iodd) * ip + iodd * iq;
                dgemm("n", "n", &m, &m, &m, &one, work + iget, &m, work + iget,
                      &m, &zero, work + iput, &m);
                iodd = 1 - iodd;
            }
        }
        return make_pair(iput, ns);
    }
    // Computes w = exp(t*A)*v - for a (sparse) symmetric matrix A.
    // Adapted from expokit fortran code dsexpv.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = n*(m+1)+n+(m+2)^2+4*(m+2)^2+ideg+1
    template <typename MatMul, typename PComm>
    static MKL_INT expo_krylov(MatMul &op, MKL_INT n, MKL_INT m, double t,
                               double *v, double *w, double &tol, double anorm,
                               double *work, MKL_INT lwork, bool iprint,
                               const PComm &pcomm = nullptr) {
        const MKL_INT inc = 1;
        const double sqr1 = sqrt(0.1), zero = 0.0;
        const MKL_INT mxstep = 500, mxreject = 0, ideg = 6;
        const double delta = 1.2, gamma = 0.9;
        MKL_INT iflag = 0;
        if (lwork < n * (m + 2) + 5 * (m + 2) * (m + 2) + ideg + 1)
            iflag = -1;
        if (m >= n || m <= 0)
            iflag = -3;
        if (iflag != 0) {
            cerr << "bad sizes (in input of expo krylov)" << endl;
            abort();
        }
        // initializations
        MKL_INT k1 = 2, mh = m + 2, iv = 0, ih = iv + n * (m + 1) + n;
        MKL_INT ifree = ih + mh * mh, lfree = lwork - ifree, iexph;
        MKL_INT ibrkflag = 0, mbrkdwn = m, nmult = 0, mx;
        MKL_INT nreject = 0, nexph = 0, nscale = 0, ns = 0;
        double t_out = abs(t), tbrkdwn = 0.0, t_now = 0.0, t_new = 0.0;
        double step_min = t_out, step_max = 0.0, s_error = 0.0, x_error = 0.0;
        double err_loc;
        MKL_INT nstep = 0;
        // machine precision
        double eps = 0.0;
        for (double p1 = 4.0 / 3.0, p2, p3; eps == 0.0;)
            p2 = p1 - 1.0, p3 = p2 + p2 + p2, eps = abs(p3 - 1.0);
        if (tol <= eps)
            tol = sqrt(eps);
        double rndoff = eps * anorm, break_tol = 1E-7;
        double sgn = t >= 0 ? 1.0 : -1.0;
        dcopy(&n, v, &inc, w, &inc);
        double beta = dnrm2(&n, w, &inc), vnorm = beta, hump = beta, avnorm;
        // obtain the very first stepsize
        double xm = 1.0 / (double)m, p1;
        p1 = tol * pow((m + 1) / 2.72, m + 1) * sqrt(2.0 * 3.14 * (m + 1));
        t_new = (1.0 / anorm) * pow(p1 / (4.0 * beta * anorm), xm);
        p1 = pow(10.0, round(log10(t_new) - sqr1) - 1);
        t_new = floor(t_new / p1 + 0.55) * p1;
        // step-by-step integration
        for (; t_now < t_out;) {
            nstep++;
            double t_step = min(t_out - t_now, t_new);
            p1 = 1.0 / beta;
            for (MKL_INT i = 0; i < n; i++)
                work[iv + i] = p1 * w[i];
            if (pcomm == nullptr || pcomm->root == pcomm->rank)
                memset(work + ih, 0, sizeof(double) * mh * mh);
            // Lanczos loop
            MKL_INT j1v = iv + n;
            double hj1j = 0.0;
            for (MKL_INT j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (j != 0) {
                        p1 = -work[ih + j * mh + j - 1];
                        daxpy(&n, &p1, work + j1v - n - n, &inc, work + j1v,
                              &inc);
                    }
                    double hjj =
                        -ddot(&n, work + j1v - n, &inc, work + j1v, &inc);
                    work[ih + j * (mh + 1)] = -hjj;
                    daxpy(&n, &hjj, work + j1v - n, &inc, work + j1v, &inc);
                    hj1j = dnrm2(&n, work + j1v, &inc);
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(&hj1j, 1, pcomm->root);
                // if "happy breakdown" go straightforward at the end
                if (hj1j <= break_tol) {
                    if (iprint)
                        cout << "happy breakdown: mbrkdwn =" << j + 1
                             << " h = " << hj1j << endl;
                    k1 = 0, ibrkflag = 1;
                    mbrkdwn = j + 1, tbrkdwn = t_now;
                    t_step = t_out - t_now;
                    break;
                }
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    work[ih + j * mh + j + 1] = hj1j;
                    work[ih + (j + 1) * mh + j] = hj1j;
                    hj1j = 1.0 / hj1j;
                    dscal(&n, &hj1j, work + j1v, &inc);
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(work + j1v, n, pcomm->root);
                j1v += n;
            }
            if (k1 != 0) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank)
                    avnorm = dnrm2(&n, work + j1v, &inc);
            }
            MKL_INT ireject = 0;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                // set 1 for the 2-corrected scheme
                work[ih + m * mh + m - 1] = 0.0;
                work[ih + m * mh + m + 1] = 1.0;
                // loop while ireject<mxreject until the tolerance is reached
                for (ireject = 0;;) {
                    // compute w = beta*V*exp(t_step*H)*e1
                    nexph++;
                    mx = mbrkdwn + k1;
                    // irreducible rational Pade approximation
                    auto xp = expo_pade(ideg, mx, work + ih, mh, sgn * t_step,
                                        work + ifree);
                    iexph = xp.first + ifree, ns = xp.second;
                    nscale += ns;
                    // error estimate
                    if (k1 == 0)
                        err_loc = tol;
                    else {
                        double p1 = abs(work[iexph + m]) * beta;
                        double p2 = abs(work[iexph + m + 1]) * beta * avnorm;
                        if (p1 > 10.0 * p2)
                            err_loc = p2, xm = 1.0 / (double)m;
                        else if (p1 > p2)
                            err_loc = p1 * p2 / (p1 - p2), xm = 1.0 / (double)m;
                        else
                            err_loc = p1, xm = 1.0 / (double)(m - 1);
                    }
                    // reject the step-size if the error is not acceptable
                    if (k1 != 0 && err_loc > delta * t_step * tol &&
                        (mxreject == 0 || ireject < mxreject)) {
                        double t_old = t_step;
                        t_step =
                            gamma * t_step * pow(t_step * tol / err_loc, xm);
                        p1 = pow(10.0, round(log10(t_step) - sqr1) - 1);
                        t_step = floor(t_step / p1 + 0.55) * p1;
                        if (iprint)
                            cout << "t_step = " << t_old
                                 << " err_loc = " << err_loc
                                 << " err_required = " << delta * t_old * tol
                                 << endl
                                 << "  stepsize rejected, stepping down to:"
                                 << t_step << endl;
                        ireject++;
                        nreject++;
                        break;
                    } else
                        break;
                }
            }
            if (mxreject != 0 && pcomm != nullptr)
                pcomm->broadcast(&ireject, 1, pcomm->root);
            if (mxreject != 0 && ireject > mxreject) {
                cerr << "failure in expo krylov: ---"
                     << " The requested tolerance is too high. Rerun "
                        "with a smaller value.";
                abort();
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                // now update w = beta*V*exp(t_step*H)*e1 and the hump
                mx = mbrkdwn + max((MKL_INT)0, k1 - 1);
                dgemv("n", &n, &mx, &beta, work + iv, &n, work + iexph, &inc,
                      &zero, w, &inc);
                beta = dnrm2(&n, w, &inc);
                hump = max(hump, beta);
                // suggested value for the next stepsize
                t_new = gamma * t_step * pow(t_step * tol / err_loc, xm);
                p1 = pow(10.0, round(log10(t_new) - sqr1) - 1);
                t_new = floor(t_new / p1 + 0.55) * p1;
                err_loc = max(err_loc, rndoff);
                // update the time covered
                t_now += t_step;
                // display and keep some information
                if (iprint)
                    cout << "integration " << nstep << " scale-square =" << ns
                         << " step_size = " << t_step
                         << " err_loc = " << err_loc << " next_step = " << t_new
                         << endl;
                step_min = min(step_min, t_step);
                step_max = max(step_max, t_step);
                s_error += err_loc;
                x_error = max(x_error, err_loc);
            }
            if (pcomm != nullptr) {
                double tmp[3] = {beta, t_new, t_now};
                pcomm->broadcast(tmp, 3, pcomm->root);
                pcomm->broadcast(w, n, pcomm->root);
                beta = tmp[0], t_new = tmp[1], t_now = tmp[2];
            }
            if (mxstep != 0 && nstep >= mxstep) {
                iflag = 1;
                break;
            }
        }
        return nmult;
    }
    // apply exponential of a matrix to a vector
    // v: input/output vector
    template <typename MatMul, typename PComm>
    static int expo_apply(MatMul &op, double t, double anorm, MatrixRef &v,
                          double consta = 0.0, bool iprint = false,
                          const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        MKL_INT vm = v.m, vn = v.n, n = vm * vn;
        if (n < 4) {
            const MKL_INT lwork = 4 * n * n + 7;
            double te[n], h[n * n], work[lwork];
            MatrixRef e = MatrixRef(&te[0], vm, vn);
            memset(e.data, 0, sizeof(double) * n);
            memset(h, 0, sizeof(double) * n * n);
            for (MKL_INT i = 0; i < n; i++) {
                e.data[i] = 1.0;
                op(e, MatrixRef(h + i * n, vm, vn));
                h[i * (n + 1)] += consta;
                e.data[i] = 0.0;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                MKL_INT iptr = expo_pade(6, n, h, n, t, work).first;
                MatrixFunctions::multiply(MatrixRef(work + iptr, n, n), true, v,
                                          false, e, 1.0, 0.0);
                memcpy(v.data, e.data, sizeof(double) * n);
            }
            if (pcomm != nullptr)
                pcomm->broadcast(v.data, n, pcomm->root);
            return n;
        }
        auto lop = [&op, consta, n, vm, vn](double *a, double *b) -> void {
            static MKL_INT inc = 1;
            memset(b, 0, sizeof(double) * n);
            op(MatrixRef(a, vm, vn), MatrixRef(b, vm, vn));
            daxpy(&n, &consta, a, &inc, b, &inc);
        };
        MKL_INT m = min((MKL_INT)deflation_max_size, n - 1);
        MKL_INT lwork = n * (m + 2) + 5 * (m + 2) * (m + 2) + 7;
        vector<double> w(n), work(lwork);
        if (anorm < 1E-10)
            anorm = 1.0;
        MKL_INT nmult = MatrixFunctions::expo_krylov(
            lop, n, m, t, v.data, w.data(), conv_thrd, anorm, work.data(),
            lwork, iprint, (PComm)pcomm);
        memcpy(v.data, w.data(), sizeof(double) * n);
        return (int)nmult;
    }
    // Solve x in linear equation H x = b
    // by applying linear CG method
    // where H is symmetric and positive-definite
    // H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static double
    conjugate_gradient(MatMul &op, const DiagonalMatrix &aa, MatrixRef x,
                       MatrixRef b, int &nmult, double consta = 0.0,
                       bool iprint = false, const PComm &pcomm = nullptr,
                       double conv_thrd = 5E-6, int max_iter = 5000,
                       int soft_max_iter = -1) {
        MatrixRef p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        double ff[2];
        double &error = ff[0], &func = ff[1];
        double old_error = 0.0;
        r.allocate();
        p.allocate();
        r.clear();
        p.clear();
        op(x, r);
        if (consta != 0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            precondition(p, r, aa);
            error = dot(p, r);
        }
        if (iprint)
            cout << endl;
        if (pcomm != nullptr)
            pcomm->broadcast(&error, 1, pcomm->root);
        if (error < conv_thrd) {
            if (pcomm == nullptr || pcomm->root == pcomm->rank)
                func = dot(x, b);
            if (pcomm != nullptr)
                pcomm->broadcast(&func, 1, pcomm->root);
            if (iprint)
                cout << setw(6) << 0 << fixed << setw(15) << setprecision(8)
                     << func << scientific << setw(13) << setprecision(2)
                     << error << endl;
            p.deallocate();
            r.deallocate();
            nmult = 1;
            return func;
        }
        old_error = error;
        if (pcomm != nullptr)
            pcomm->broadcast(p.data, p.size(), pcomm->root);
        MatrixRef hp(nullptr, x.m, x.n), z(nullptr, x.m, x.n);
        hp.allocate();
        z.allocate();
        int xiter = 0;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            hp.clear();
            op(p, hp);
            if (consta != 0)
                iadd(hp, p, consta);

            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                double alpha = old_error / dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                precondition(z, r, aa);
                error = dot(z, r);
                func = dot(x, b);
                if (iprint)
                    cout << setw(6) << xiter << fixed << setw(15)
                         << setprecision(8) << func << scientific << setw(13)
                         << setprecision(2) << error << endl;
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&error, 2, pcomm->root);
            if (error < conv_thrd)
                break;
            else {
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    double beta = error / old_error;
                    old_error = error;
                    iadd(p, z, 1 / beta);
                    iscale(p, beta);
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(p.data, p.size(), pcomm->root);
            }
        }
        if (xiter == max_iter && error >= conv_thrd) {
            cout << "Error : linear solver (cg) not converged!" << endl;
            assert(false);
        }
        nmult = xiter + 1;
        z.deallocate();
        hp.deallocate();
        p.deallocate();
        r.deallocate();
        if (pcomm != nullptr)
            pcomm->broadcast(x.data, x.size(), pcomm->root);
        return func;
    }
    // Solve x in linear equation H x = b where H^T = H
    // by applying linear CG method to equation (H H) x = H b
    // where H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static double minres(MatMul &op, MatrixRef x, MatrixRef b, int &nmult,
                         double consta = 0.0, bool iprint = false,
                         const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
                         int max_iter = 5000, int soft_max_iter = -1) {
        MatrixRef p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        double ff[2];
        double &error = ff[0], &func = ff[1];
        r.allocate();
        r.clear();
        op(x, r);
        if (consta != 0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iadd(r, b, -1);
            iscale(r, -1);
            p.allocate();
            copy(p, r);
            error = dot(r, r);
        }
        if (iprint)
            cout << endl;
        if (pcomm != nullptr)
            pcomm->broadcast(&error, 1, pcomm->root);
        if (error < conv_thrd) {
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                func = dot(x, b);
                p.deallocate();
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&func, 1, pcomm->root);
            if (iprint)
                cout << setw(6) << 0 << fixed << setw(15) << setprecision(8)
                     << func << scientific << setw(13) << setprecision(2)
                     << error << endl;
            r.deallocate();
            nmult = 1;
            return func;
        }
        if (pcomm != nullptr)
            pcomm->broadcast(r.data, r.size(), pcomm->root);
        double beta = 0, prev_beta = 0;
        MatrixRef hp(nullptr, x.m, x.n), hr(nullptr, x.m, x.n);
        hr.allocate();
        hr.clear();
        op(r, hr);
        if (consta != 0)
            iadd(hr, r, consta);
        prev_beta = dot(r, hr);
        int xiter = 0;

        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            hp.allocate();
            copy(hp, hr);
        }

        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                double alpha = dot(r, hr) / dot(hp, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                error = dot(r, r);
                func = dot(x, b);
                if (iprint)
                    cout << setw(6) << xiter << fixed << setw(15)
                         << setprecision(8) << func << scientific << setw(13)
                         << setprecision(2) << error << endl;
            }
            if (pcomm != nullptr) {
                pcomm->broadcast(&error, 2, pcomm->root);
                pcomm->broadcast(r.data, r.size(), pcomm->root);
            }
            if (error < conv_thrd)
                break;
            else {
                hr.clear();
                op(r, hr);
                if (consta != 0)
                    iadd(hr, r, consta);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    beta = dot(r, hr);
                    iadd(p, r, prev_beta / beta);
                    iscale(p, beta / prev_beta);
                    iadd(hp, hr, prev_beta / beta);
                    iscale(hp, beta / prev_beta);
                    prev_beta = beta;
                }
            }
        }
        if (xiter == max_iter && error >= conv_thrd) {
            cout << "Error : linear solver (minres) not converged!" << endl;
            assert(false);
        }
        nmult = xiter + 1;
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            hp.deallocate();
        hr.deallocate();
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            p.deallocate();
        r.deallocate();
        if (pcomm != nullptr)
            pcomm->broadcast(x.data, x.size(), pcomm->root);
        return func;
    }
};

} // namespace block2
