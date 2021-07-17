
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

// LU factorization
extern void dgetrf(const MKL_INT *m, const MKL_INT *n, double *a,
                   const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info);

// matrix inverse
extern void dgetri(const MKL_INT *m, double *a, const MKL_INT *lda,
                   MKL_INT *ipiv, double *work, const MKL_INT *lwork,
                   MKL_INT *info);

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

// least squares problem a * x = b
extern void dgels(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const MKL_INT *nrhs, double *a, const MKL_INT *lda, double *b,
                  const MKL_INT *ldb, double *work, const MKL_INT *lwork,
                  MKL_INT *info);

#endif
}

enum struct DavidsonTypes : uint8_t {
    Normal = 0,
    GreaterThan = 1,
    LessThan = 2,
    CloseTo = 4,
    Harmonic = 16,
    HarmonicGreaterThan = 16 | 1,
    HarmonicLessThan = 16 | 2,
    HarmonicCloseTo = 16 | 4,
    DavidsonPrecond = 32,
    NoPrecond = 64
};

inline bool operator&(DavidsonTypes a, DavidsonTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline DavidsonTypes operator|(DavidsonTypes a, DavidsonTypes b) {
    return DavidsonTypes((uint8_t)a | (uint8_t)b);
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
                     bool conj = false, double cfactor = 1.0) {
        static const double x = 1.0;
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            MKL_INT n = a.m * a.n, inc = 1;
            if (cfactor == 1.0)
                daxpy(&n, &scale, b.data, &inc, a.data, &inc);
            else
                dgemm("N", "N", &inc, &n, &inc, &scale, &x, &inc, b.data, &inc,
                      &cfactor, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n == b.m);
            assert(cfactor == 1.0);
            for (MKL_INT i = 0, inc = 1; i < a.m; i++)
                daxpy(&a.n, &scale, b.data + i, &a.m, a.data + i * a.n, &inc);
        }
    }
    static double norm(const MatrixRef &a) {
        MKL_INT n = a.m * a.n, inc = 1;
        return dnrm2(&n, a.data, &inc);
    }
    // determinant
    static double det(const MatrixRef &a) {
        assert(a.m == a.n);
        vector<double> aa;
        vector<MKL_INT> ipiv;
        aa.reserve(a.m * a.n);
        ipiv.reserve(a.m);
        memcpy(aa.data(), a.data, sizeof(double) * a.m * a.n);
        MKL_INT info = -1;
        dgetrf(&a.m, &a.n, aa.data(), &a.m, ipiv.data(), &info);
        assert(info == 0);
        double det = 1.0;
        for (int i = 0; i < a.m; i++)
            det *= ipiv[i] != i + 1 ? -aa[i * a.m + i] : aa[i * a.m + i];
        return det;
    }
    // matrix inverse
    static void inverse(const MatrixRef &a) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(a.m == a.n);
        vector<MKL_INT> ipiv;
        ipiv.reserve(a.m);
        MKL_INT info = -1, lwork = 34 * a.m;
        dgetrf(&a.m, &a.n, a.data, &a.m, ipiv.data(), &info);
        assert(info == 0);
        double *work = d_alloc->allocate(lwork);
        dgetri(&a.m, a.data, &a.m, ipiv.data(), work, &lwork, &info);
        assert(info == 0);
        d_alloc->deallocate(work, lwork);
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
    // least squares problem a x = b
    // return the residual (norm, not squared)
    // a.n is used as lda
    static double least_squares(const MatrixRef &a, const MatrixRef &b,
                                const MatrixRef &x) {
        assert(a.m == b.m && a.n >= x.m && b.n == 1 && x.n == 1);
        vector<double> work, atr, xtr;
        MKL_INT lwork = 34 * min(a.m, x.m), info = -1, nrhs = 1,
                mn = max(a.m, x.m), nr = a.m - x.m;
        work.reserve(lwork);
        atr.reserve(a.size());
        xtr.reserve(mn);
        dcopy(&a.m, b.data, &nrhs, xtr.data(), &nrhs);
        for (MKL_INT i = 0; i < x.m; i++)
            dcopy(&a.m, a.data + i, &a.n, atr.data() + i * a.m, &nrhs);
        dgels("N", &a.m, &x.m, &nrhs, atr.data(), &a.m, xtr.data(), &mn,
              work.data(), &lwork, &info);
        assert(info == 0);
        dcopy(&x.m, xtr.data(), &nrhs, x.data, &nrhs);
        return nr > 0 ? dnrm2(&nr, xtr.data() + x.m, &nrhs) : 0;
    }
    // c.n is used for ldc; a.n is used for lda
    static void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        // if assertion failes here, check whether it is the case
        // where different bra and ket are used with the transpose rule
        // use no-transpose-rule to fix it
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
    // c = bra(.T) * a * ket(.T)
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
    // c(.T) = bra.T * a(.T) * ket
    // return nflop
    static size_t rotate(const MatrixRef &a, bool conj_a, const MatrixRef &c,
                         bool conj_c, const MatrixRef &bra,
                         const MatrixRef &ket, double scale) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MatrixRef work(nullptr, conj_a ? a.n : a.m, ket.n);
        work.allocate(d_alloc);
        multiply(a, conj_a, ket, false, work, 1.0, 0.0);
        if (!conj_c)
            multiply(bra, true, work, false, c, scale, 1.0);
        else
            multiply(work, true, bra, false, c, scale, 1.0);
        work.deallocate(d_alloc);
        return (size_t)a.m * a.n * work.n + (size_t)work.m * work.n * bra.n;
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
        switch ((uint8_t)conja | (conjb << 1)) {
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
    // SVD for parallelism over sites; PRB 87, 155137 (2013)
    static void accurate_svd(const MatrixRef &a, const MatrixRef &l,
                             const MatrixRef &s, const MatrixRef &r,
                             double eps = 1E-4) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MatrixRef aa(nullptr, a.m, a.n);
        aa.data = d_alloc->allocate(aa.size());
        copy(aa, a);
        svd(aa, l, s, r);
        MKL_INT k = min(a.m, a.n);
        MKL_INT p = -1;
        for (MKL_INT ip = 0; ip < k; ip++)
            if (s.data[ip] < eps * s.data[0]) {
                p = ip;
                break;
            }
        if (p != -1) {
            MatrixRef xa(nullptr, k - p, k - p), xl(nullptr, k - p, k - p),
                xr(nullptr, k - p, k - p);
            xa.data = d_alloc->allocate(xa.size());
            xl.data = d_alloc->allocate(xl.size());
            xr.data = d_alloc->allocate(xr.size());
            rotate(a, xa, MatrixRef(l.data + p, l.m, l.n), true,
                   MatrixRef(r.data + p * r.n, r.m - p, r.n), true, 1.0);
            accurate_svd(xa, xl, MatrixRef(s.data + p, 1, k - p), xr, eps);
            MatrixRef bl(nullptr, l.m, l.n), br(nullptr, r.m, r.n);
            bl.data = d_alloc->allocate(bl.size());
            br.data = d_alloc->allocate(br.size());
            copy(bl, l);
            copy(br, r);
            multiply(MatrixRef(bl.data + p, bl.m, bl.n), false, xl, false,
                     MatrixRef(l.data + p, l.m, l.n), 1.0, 0.0);
            multiply(xr, false, MatrixRef(br.data + p * br.n, br.m - p, br.n),
                     false, MatrixRef(r.data + p * r.n, r.m - p, r.n), 1.0,
                     0.0);
            d_alloc->deallocate(br.data, br.size());
            d_alloc->deallocate(bl.data, bl.size());
            d_alloc->deallocate(xr.data, xr.size());
            d_alloc->deallocate(xl.data, xl.size());
            d_alloc->deallocate(xa.data, xa.size());
        }
        d_alloc->deallocate(aa.data, aa.size());
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
    // b = a.T
    static void transpose(const MatrixRef &a, const MatrixRef &b) {
        b.clear();
        iadd(b, a, 1.0, true);
    }
    // diagonalization for each symmetry block
    static void block_eigs(const MatrixRef &a, const DiagonalMatrix &w,
                           const vector<uint8_t> &x) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        uint8_t maxx = *max_element(x.begin(), x.end()) + 1;
        vector<vector<MKL_INT>> mp(maxx);
        assert(a.m == a.n && w.n == a.n && (MKL_INT)x.size() == a.n);
        for (MKL_INT i = 0; i < a.n; i++)
            mp[x[i]].push_back(i);
        for (uint8_t i = 0; i < maxx; i++)
            if (mp[i].size() != 0) {
                double *work = d_alloc->allocate(mp[i].size() * mp[i].size());
                double *wwork = d_alloc->allocate(mp[i].size());
                for (size_t j = 0; j < mp[i].size(); j++)
                    for (size_t k = 0; k < mp[i].size(); k++)
                        work[j * mp[i].size() + k] = a(mp[i][j], mp[i][k]);
                eigs(MatrixRef(work, (MKL_INT)mp[i].size(),
                               (MKL_INT)mp[i].size()),
                     DiagonalMatrix(wwork, (MKL_INT)mp[i].size()));
                for (size_t j = 0; j < mp[i].size(); j++)
                    for (MKL_INT k = 0; k < a.n; k++)
                        a(mp[i][j], k) = 0.0, a(k, mp[i][j]) = 0.0;
                for (size_t j = 0; j < mp[i].size(); j++)
                    for (size_t k = 0; k < mp[i].size(); k++)
                        a(mp[i][j], mp[i][k]) = work[j * mp[i].size() + k];
                for (size_t j = 0; j < mp[i].size(); j++)
                    w(mp[i][j], mp[i][j]) = wwork[j];
                d_alloc->deallocate(wwork, mp[i].size());
                d_alloc->deallocate(work, mp[i].size() * mp[i].size());
            }
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
    static void cg_precondition(const MatrixRef &z, const MatrixRef &r,
                                const DiagonalMatrix &aa) {
        copy(z, r);
        if (aa.size() != 0) {
            assert(aa.size() == r.size() && r.size() == z.size());
            for (MKL_INT i = 0; i < aa.n; i++)
                if (abs(aa.data[i]) > 1E-12)
                    z.data[i] /= aa.data[i];
        }
    }
    // ER, Davidson. "The iterative calculation of a few of the lowest
    // eigenvalues and corresponding eigenvectors of large real-symmetric
    // matrices." Journal of Computational Physics 17 (1975): 87-94.
    // Section III. D
    static void davidson_precondition(const MatrixRef &q, double ld,
                                      const DiagonalMatrix &aa) {
        assert(aa.size() == q.size());
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                q.data[i] /= ld - aa.data[i];
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
    // ors: orthogonal states to be projected out
    template <typename MatMul, typename PComm>
    static vector<double>
    davidson(MatMul &op, const DiagonalMatrix &aa, vector<MatrixRef> &vs,
             double shift, DavidsonTypes davidson_type, int &ndav,
             bool iprint = false, const PComm &pcomm = nullptr,
             double conv_thrd = 5E-6, int max_iter = 5000,
             int soft_max_iter = -1, int deflation_min_size = 2,
             int deflation_max_size = 50,
             const vector<MatrixRef> &ors = vector<MatrixRef>()) {
        assert(!(davidson_type & DavidsonTypes::Harmonic));
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        int k = (int)vs.size(), nor = (int)ors.size();
        if (deflation_min_size < k)
            deflation_min_size = k;
        if (deflation_max_size < k + k / 2)
            deflation_max_size = k + k / 2;
        MatrixRef pbs(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()), 1);
        MatrixRef pss(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()), 1);
        pbs.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        pss.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        vector<MatrixRef> bs(deflation_max_size,
                             MatrixRef(nullptr, vs[0].m, vs[0].n));
        vector<MatrixRef> sigmas(deflation_max_size,
                                 MatrixRef(nullptr, vs[0].m, vs[0].n));
        vector<double> or_normsqs(nor);
        for (int i = 0; i < nor; i++) {
            for (int j = 0; j < i; j++)
                if (or_normsqs[j] > 1E-14)
                    iadd(ors[i], ors[j], -dot(ors[j], ors[i]) / or_normsqs[j]);
            or_normsqs[i] = dot(ors[i], ors[i]);
        }
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
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < nor; j++)
                if (or_normsqs[j] > 1E-14)
                    iadd(bs[i], ors[j], -dot(ors[j], bs[i]) / or_normsqs[j]);
            double normsq = dot(bs[i], bs[i]);
            if (normsq < 1E-14) {
                cout << "Cannot generate initial guess " << i
                     << " for Davidson orthogonal to all given states!" << endl;
                assert(false);
            }
            iscale(bs[i], 1.0 / sqrt(normsq));
        }
        vector<double> eigvals(k);
        vector<int> eigval_idxs(deflation_max_size);
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
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
                    for (int ij = 0; ij < m * m; ij++) {
                        int i = ij / m, j = ij % m;
#else
#pragma omp for schedule(dynamic) collapse(2)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++) {
#endif
                        if (j <= i)
                            alpha(i, j) = dot(bs[i], sigmas[j]);
                    }
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
                for (int i = 0; i < m; i++)
                    eigval_idxs[i] = i;
                if (davidson_type & DavidsonTypes::CloseTo)
                    sort(eigval_idxs.begin(), eigval_idxs.begin() + m,
                         [&ld, shift](int i, int j) {
                             return abs(ld.data[i] - shift) <
                                    abs(ld.data[j] - shift);
                         });
                else if (davidson_type & DavidsonTypes::LessThan)
                    sort(eigval_idxs.begin(), eigval_idxs.begin() + m,
                         [&ld, shift](int i, int j) {
                             if ((shift >= ld.data[i]) != (shift >= ld.data[j]))
                                 return shift >= ld.data[i];
                             else if (shift >= ld.data[i])
                                 return shift - ld.data[i] < shift - ld.data[j];
                             else
                                 return ld.data[i] - shift >=
                                        ld.data[j] - shift;
                         });
                else if (davidson_type & DavidsonTypes::GreaterThan)
                    sort(eigval_idxs.begin(), eigval_idxs.begin() + m,
                         [&ld, shift](int i, int j) {
                             if ((shift > ld.data[i]) != (shift > ld.data[j]))
                                 return shift > ld.data[j];
                             else if (shift > ld.data[i])
                                 return shift - ld.data[i] > shift - ld.data[j];
                             else
                                 return ld.data[i] - shift <=
                                        ld.data[j] - shift;
                         });
                for (int i = 0; i < ck; i++) {
                    int ii = eigval_idxs[i];
                    copy(q, sigmas[ii]);
                    iadd(q, bs[ii], -ld(ii, ii));
                    if (dot(q, q) >= conv_thrd) {
                        ck = i;
                        break;
                    }
                }
                int ick = eigval_idxs[ck];
                copy(q, sigmas[ick]);
                iadd(q, bs[ick], -ld(ick, ick));
                for (int j = 0; j < nor; j++)
                    if (or_normsqs[j] > 1E-14)
                        iadd(q, ors[j], -dot(ors[j], q) / or_normsqs[j]);
                qq = dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8) << ld.data[ick]
                         << scientific << setw(13) << setprecision(2) << qq
                         << endl;
                if (davidson_type & DavidsonTypes::DavidsonPrecond)
                    davidson_precondition(q, ld.data[ick], aa);
                else if (!(davidson_type & DavidsonTypes::NoPrecond))
                    olsen_precondition(q, bs[ick], ld.data[ick], aa);
                eigvals.resize(ck + 1);
                if (ck + 1 != 0)
                    for (int i = 0; i <= ck; i++)
                        eigvals[i] = ld.data[eigval_idxs[i]];
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
                bool do_deflation = false;
                if (m >= deflation_max_size) {
                    m = msig = deflation_min_size;
                    do_deflation =
                        (davidson_type & DavidsonTypes::LessThan) ||
                        (davidson_type & DavidsonTypes::GreaterThan) ||
                        (davidson_type & DavidsonTypes::CloseTo);
                }
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (do_deflation) {
                        vector<MatrixRef> tmp(
                            m, MatrixRef(nullptr, bs[0].m, bs[0].n));
                        for (int i = 0; i < m; i++)
                            tmp[i].allocate();
                        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
                        {
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(tmp[j], bs[eigval_idxs[j]]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(bs[j], tmp[j]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(tmp[j], sigmas[eigval_idxs[j]]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(sigmas[j], tmp[j]);
                        }
                        threading->activate_normal();
                        for (int i = m - 1; i >= 0; i--)
                            tmp[i].deallocate();
                    }
                    for (int j = 0; j < m; j++)
                        iadd(q, bs[j], -dot(bs[j], q));
                    for (int j = 0; j < nor; j++)
                        if (or_normsqs[j] > 1E-14)
                            iadd(q, ors[j], -dot(ors[j], q) / or_normsqs[j]);
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
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            for (int i = 0; i < k; i++)
                copy(vs[i], bs[eigval_idxs[i]]);
        if (pcomm != nullptr) {
            pcomm->broadcast(eigvals.data(), eigvals.size(), pcomm->root);
            for (int j = 0; j < k; j++)
                pcomm->broadcast(vs[j].data, vs[j].size(), pcomm->root);
        }
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.deallocate();
        d_alloc->deallocate(pss.data, deflation_max_size * vs[0].size());
        d_alloc->deallocate(pbs.data, deflation_max_size * vs[0].size());
        ndav = xiter;
        return eigvals;
    }
    // Harmonic Davidson algorithm
    // aa: diag elements of a (for precondition)
    // bs: input/output vector
    // shift: solve for eigenvalues near this value
    // davidson_type: whether eigenvalues should be above/below/near shift
    // ors: orthogonal states to be projected out
    template <typename MatMul, typename PComm>
    static vector<double> harmonic_davidson(
        MatMul &op, const DiagonalMatrix &aa, vector<MatrixRef> &vs,
        double shift, DavidsonTypes davidson_type, int &ndav,
        bool iprint = false, const PComm &pcomm = nullptr,
        double conv_thrd = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        int deflation_min_size = 2, int deflation_max_size = 50,
        const vector<MatrixRef> &ors = vector<MatrixRef>()) {
        if (!(davidson_type & DavidsonTypes::Harmonic))
            return davidson(op, aa, vs, shift, davidson_type, ndav, iprint,
                            pcomm, conv_thrd, max_iter, soft_max_iter,
                            deflation_min_size, deflation_max_size, ors);
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        int k = (int)vs.size(), nor = (int)ors.size();
        if (deflation_min_size < k)
            deflation_min_size = k;
        if (deflation_max_size < k + k / 2)
            deflation_max_size = k + k / 2;
        MatrixRef pbs(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()), 1);
        MatrixRef pss(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()), 1);
        pbs.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        pss.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        vector<MatrixRef> bs(deflation_max_size,
                             MatrixRef(nullptr, vs[0].m, vs[0].n));
        vector<MatrixRef> sigmas(deflation_max_size,
                                 MatrixRef(nullptr, vs[0].m, vs[0].n));
        vector<double> or_normsqs(nor);
        for (int i = 0; i < nor; i++) {
            for (int j = 0; j < i; j++)
                if (or_normsqs[j] > 1E-14)
                    iadd(ors[i], ors[j], -dot(ors[j], ors[i]) / or_normsqs[j]);
            or_normsqs[i] = dot(ors[i], ors[i]);
        }
        for (int i = 0; i < deflation_max_size; i++) {
            bs[i].data = pbs.data + bs[i].size() * i;
            sigmas[i].data = pss.data + sigmas[i].size() * i;
        }
        for (int i = 0; i < k; i++)
            copy(bs[i], vs[i]);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < nor; j++)
                if (or_normsqs[j] > 1E-14)
                    iadd(bs[i], ors[j], -dot(ors[j], bs[i]) / or_normsqs[j]);
            double normsq = dot(bs[i], bs[i]);
            if (normsq < 1E-14) {
                cout << "Cannot generate initial guess " << i
                     << " for Davidson orthogonal to all given states!" << endl;
                assert(false);
            }
            iscale(bs[i], 1.0 / sqrt(normsq));
        }
        int num_matmul = 0;
        for (int i = 0; i < k; i++) {
            sigmas[i].clear();
            op(bs[i], sigmas[i]);
            if (shift != 0.0)
                iadd(sigmas[i], bs[i], -shift);
            num_matmul++;
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++) {
                iadd(bs[i], bs[j], -dot(sigmas[j], sigmas[i]));
                iadd(sigmas[i], sigmas[j], -dot(sigmas[j], sigmas[i]));
            }
            iscale(bs[i], 1.0 / sqrt(dot(sigmas[i], sigmas[i])));
            iscale(sigmas[i], 1.0 / sqrt(dot(sigmas[i], sigmas[i])));
        }
        vector<double> eigvals(k);
        vector<int> eigval_idxs(deflation_max_size);
        MatrixRef q(nullptr, bs[0].m, bs[0].n);
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.allocate();
        int ck = 0, m = k, xiter = 0;
        double qq;
        if (iprint)
            cout << endl;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
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
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
                    for (int ij = 0; ij < m * m; ij++) {
                        int i = ij / m, j = ij % m;
#else
#pragma omp for schedule(dynamic) collapse(2)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++) {
#endif
                        if (j <= i)
                            alpha(i, j) = dot(bs[i], sigmas[j]);
                    }
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
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        ld(j, j) = dot(bs[j], sigmas[j]) / dot(bs[j], bs[j]);
                }
                threading->activate_normal();
                for (int i = m - 1; i >= 0; i--)
                    tmp[i].deallocate();
                alpha.deallocate();
                if (davidson_type & DavidsonTypes::LessThan)
                    for (int i = 0; i < m; i++)
                        eigval_idxs[i] = i;
                else
                    for (int i = 0; i < m; i++)
                        eigval_idxs[i] = m - 1 - i;
                if (davidson_type & DavidsonTypes::CloseTo)
                    sort(eigval_idxs.begin(), eigval_idxs.begin() + m,
                         [&ld](int i, int j) {
                             return abs(ld(i, i)) < abs(ld(j, j));
                         });
                for (int i = 0; i < ck; i++) {
                    int ii = eigval_idxs[i];
                    copy(q, sigmas[ii]);
                    iadd(q, bs[ii], -ld(ii, ii));
                    if (dot(q, q) >= conv_thrd) {
                        ck = i;
                        break;
                    }
                }
                int ick = eigval_idxs[ck];
                copy(q, sigmas[ick]);
                iadd(q, bs[ick], -ld(ick, ick));
                for (int j = 0; j < nor; j++)
                    if (or_normsqs[j] > 1E-14)
                        iadd(q, ors[j], -dot(ors[j], q) / or_normsqs[j]);
                qq = dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8)
                         << ld.data[ick] + shift << scientific << setw(13)
                         << setprecision(2) << qq << endl;
                if (davidson_type & DavidsonTypes::DavidsonPrecond)
                    davidson_precondition(q, ld.data[ick] + shift, aa);
                else if (!(davidson_type & DavidsonTypes::NoPrecond))
                    olsen_precondition(q, bs[ick], ld.data[ick] + shift, aa);
                eigvals.resize(ck + 1);
                if (ck + 1 != 0)
                    for (int i = 0; i <= ck; i++)
                        eigvals[i] = ld.data[eigval_idxs[i]] + shift;
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
                bool do_deflation = false;
                if (m >= deflation_max_size) {
                    m = deflation_min_size;
                    do_deflation = !(davidson_type & DavidsonTypes::LessThan);
                }
                copy(bs[m], q);
                if (pcomm != nullptr)
                    pcomm->broadcast(bs[m].data, bs[m].size(), pcomm->root);
                sigmas[m].clear();
                op(bs[m], sigmas[m]);
                if (shift != 0.0)
                    iadd(sigmas[m], bs[m], -shift);
                num_matmul++;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (do_deflation) {
                        vector<MatrixRef> tmp(
                            m, MatrixRef(nullptr, bs[0].m, bs[0].n));
                        for (int i = 0; i < m; i++)
                            tmp[i].allocate();
                        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
                        {
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(tmp[j], bs[eigval_idxs[j]]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(bs[j], tmp[j]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(tmp[j], sigmas[eigval_idxs[j]]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(sigmas[j], tmp[j]);
                        }
                        threading->activate_normal();
                        for (int i = m - 1; i >= 0; i--)
                            tmp[i].deallocate();
                    }
                    for (int i = 0; i < m + 1; i++) {
                        for (int j = 0; j < i; j++) {
                            iadd(bs[i], bs[j], -dot(sigmas[j], sigmas[i]));
                            iadd(sigmas[i], sigmas[j],
                                 -dot(sigmas[j], sigmas[i]));
                        }
                        iscale(bs[i], 1.0 / sqrt(dot(sigmas[i], sigmas[i])));
                        iscale(sigmas[i],
                               1.0 / sqrt(dot(sigmas[i], sigmas[i])));
                    }
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
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            for (int i = 0; i < k; i++)
                copy(vs[i], sigmas[eigval_idxs[i]]);
        if (pcomm != nullptr) {
            pcomm->broadcast(eigvals.data(), eigvals.size(), pcomm->root);
            for (int j = 0; j < k; j++)
                pcomm->broadcast(vs[j].data, vs[j].size(), pcomm->root);
        }
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.deallocate();
        d_alloc->deallocate(pss.data, deflation_max_size * vs[0].size());
        d_alloc->deallocate(pbs.data, deflation_max_size * vs[0].size());
        ndav = num_matmul;
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
    // Computes w = exp(t*A)*v - for a (sparse) symmetric / general matrix A.
    // Adapted from expokit fortran code dsexpv.f/dgexpy.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = n*(m+1)+n+(m+2)^2+4*(m+2)^2+ideg+1
    template <typename MatMul, typename PComm>
    static MKL_INT expo_krylov(MatMul &op, MKL_INT n, MKL_INT m, double t,
                               double *v, double *w, double &tol, double anorm,
                               double *work, MKL_INT lwork, bool symmetric,
                               bool iprint, const PComm &pcomm = nullptr) {
        const MKL_INT inc = 1;
        const double sqr1 = sqrt(0.1), zero = 0.0;
        const MKL_INT mxstep = symmetric ? 500 : 1000, mxreject = 0, ideg = 6;
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
            // Lanczos loop / Arnoldi loop
            MKL_INT j1v = iv + n;
            double hj1j = 0.0;
            for (MKL_INT j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (symmetric) {
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
                    } else {
                        for (MKL_INT i = 0; i <= j; i++) {
                            double hij = -ddot(&n, work + iv + i * n, &inc,
                                               work + j1v, &inc);
                            daxpy(&n, &hij, work + iv + i * n, &inc, work + j1v,
                                  &inc);
                            work[ih + j * mh + i] = -hij;
                        }
                        hj1j = dnrm2(&n, work + j1v, &inc);
                    }
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
                    if (symmetric)
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
                if (symmetric)
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
                          double consta, bool symmetric, bool iprint = false,
                          const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        MKL_INT vm = v.m, vn = v.n, n = vm * vn;
        if (n < 4) {
            const MKL_INT lwork = 4 * n * n + 7;
            vector<double> te(n), h(n * n), work(lwork);
            MatrixRef e = MatrixRef(te.data(), vm, vn);
            memset(e.data, 0, sizeof(double) * n);
            for (MKL_INT i = 0; i < n; i++) {
                e.data[i] = 1.0;
                op(e, MatrixRef(h.data() + i * n, vm, vn));
                h[i * (n + 1)] += consta;
                e.data[i] = 0.0;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                MKL_INT iptr =
                    expo_pade(6, n, h.data(), n, t, work.data()).first;
                MatrixFunctions::multiply(MatrixRef(work.data() + iptr, n, n),
                                          true, v, false, e, 1.0, 0.0);
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
        anorm += abs(consta) * n;
        if (anorm < 1E-10)
            anorm = 1.0;
        MKL_INT nmult = MatrixFunctions::expo_krylov(
            lop, n, m, t, v.data, w.data(), conv_thrd, anorm, work.data(),
            lwork, symmetric, iprint, (PComm)pcomm);
        memcpy(v.data, w.data(), sizeof(double) * n);
        return (int)nmult;
    }
    // Solve x in linear equation H x = b
    // by applying deflated CG method
    // where H is symmetric and positive-definite
    // H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static double harmonic_projected_deflated_conjugate_gradient(
        MatMul &op, const DiagonalMatrix &aa, MatrixRef x, MatrixRef b,
        int &nmult, int &ndav, double consta = 0.0, bool iprint = false,
        const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
        double conv_thrd_p = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        int deflation_min_size = 2, int deflation_max_size = 50) {
        int k = 5;
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        if (deflation_max_size > (int)x.size())
            deflation_max_size = (int)x.size();
        if (deflation_min_size > deflation_max_size)
            deflation_min_size = deflation_max_size;
        if (deflation_min_size < k)
            deflation_min_size = k;
        MatrixRef pbs(nullptr, (MKL_INT)(deflation_max_size * x.size()), 1);
        MatrixRef pss(nullptr, (MKL_INT)(deflation_max_size * x.size()), 1);
        pbs.data = d_alloc->allocate(deflation_max_size * x.size());
        pss.data = d_alloc->allocate(deflation_max_size * x.size());
        vector<MatrixRef> ws(deflation_max_size, MatrixRef(nullptr, x.m, x.n));
        vector<MatrixRef> aws(deflation_max_size, MatrixRef(nullptr, x.m, x.n));
        for (int i = 0; i < deflation_max_size; i++) {
            ws[i].data = pbs.data + ws[i].size() * i;
            aws[i].data = pss.data + aws[i].size() * i;
        }
        copy(ws[0], x);
        int num_matmul = 0;
        aws[0].clear();
        op(ws[0], aws[0]);
        if (consta != 0)
            iadd(aws[0], ws[0], consta);
        iscale(ws[0], 1.0 / sqrt(dot(aws[0], aws[0])));
        iscale(aws[0], 1.0 / sqrt(dot(aws[0], aws[0])));
        num_matmul++;
        vector<int> eigval_idxs(deflation_max_size);
        MatrixRef q(nullptr, x.m, x.n);
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.allocate();
        int ck = 0, msig = 0, m = 1, xiter = 0;
        double qq;
        if (iprint)
            cout << endl;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                DiagonalMatrix ld(nullptr, m);
                MatrixRef alpha(nullptr, m, m);
                ld.allocate();
                alpha.allocate();
                vector<MatrixRef> tmp(m, MatrixRef(nullptr, x.m, x.n));
                for (int i = 0; i < m; i++)
                    tmp[i].allocate();
                int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
                {
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
                    for (int ij = 0; ij < m * m; ij++) {
                        int i = ij / m, j = ij % m;
#else
#pragma omp for schedule(dynamic) collapse(2)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++) {
#endif
                        if (j <= i)
                            alpha(i, j) = dot(ws[i], aws[j]);
                    }
#pragma omp single
                    eigs(alpha, ld);
                    // note alpha row/column is diff from python
                    // b[1:m] = np.dot(b[:], alpha[:, 1:m])
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], ws[j]);
                        iscale(ws[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(ws[j], tmp[i], alpha(j, i));
                                // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], aws[j]);
                        iscale(aws[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(aws[j], tmp[i], alpha(j, i));
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        ld(j, j) = dot(ws[j], aws[j]) / dot(ws[j], ws[j]);
                }
                threading->activate_normal();
                for (int i = m - 1; i >= 0; i--)
                    tmp[i].deallocate();
                alpha.deallocate();
                for (int i = 0; i < m; i++)
                    eigval_idxs[i] = m - 1 - i;
                for (int i = 0; i < ck; i++) {
                    int ii = eigval_idxs[i];
                    copy(q, aws[ii]);
                    iadd(q, ws[ii], -ld(ii, ii));
                    if (dot(q, q) >= conv_thrd_p) {
                        ck = i;
                        break;
                    }
                }
                int ick = eigval_idxs[ck];
                copy(q, aws[ick]);
                iadd(q, ws[ick], -ld(ick, ick));
                qq = dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8) << ld.data[ick]
                         << scientific << setw(13) << setprecision(2) << qq
                         << endl;
                olsen_precondition(q, ws[ick], ld.data[ick], aa);
                ld.deallocate();
            }
            if (pcomm != nullptr) {
                pcomm->broadcast(&qq, 1, pcomm->root);
                pcomm->broadcast(&ck, 1, pcomm->root);
            }
            if (m == deflation_max_size && ck >= k - 1 && qq < conv_thrd_p) {
                ck = k;
                break;
            }
            if (m == deflation_max_size &&
                xiter + (deflation_max_size - deflation_min_size) >=
                    soft_max_iter)
                break;
            if (qq < conv_thrd_p && ck < k - 1)
                ck++;
            else {
                bool do_deflation = false;
                if (m >= deflation_max_size) {
                    m = msig = deflation_min_size;
                    do_deflation = true;
                }
                copy(ws[m], q);
                if (pcomm != nullptr)
                    pcomm->broadcast(ws[m].data, ws[m].size(), pcomm->root);
                aws[m].clear();
                op(ws[m], aws[m]);
                if (consta != 0)
                    iadd(aws[m], ws[m], consta);
                num_matmul++;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (do_deflation) {
                        vector<MatrixRef> tmp(m, MatrixRef(nullptr, x.m, x.n));
                        for (int i = 0; i < m; i++)
                            tmp[i].allocate();
                        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
                        {
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(tmp[j], ws[eigval_idxs[j]]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(ws[j], tmp[j]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(tmp[j], aws[eigval_idxs[j]]);
#pragma omp for schedule(static)
                            for (int j = 0; j < m; j++)
                                copy(aws[j], tmp[j]);
                        }
                        threading->activate_normal();
                        for (int i = m - 1; i >= 0; i--)
                            tmp[i].deallocate();
                    }
                    for (int i = 0; i < m + 1; i++) {
                        for (int j = 0; j < i; j++) {
                            iadd(ws[i], ws[j], -dot(aws[j], aws[i]));
                            iadd(aws[i], aws[j], -dot(aws[j], aws[i]));
                        }
                        iscale(ws[i], 1.0 / sqrt(dot(aws[i], aws[i])));
                        iscale(aws[i], 1.0 / sqrt(dot(aws[i], aws[i])));
                    }
                }
                m++;
            }
        }
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.deallocate();
        assert(m == deflation_max_size);
        ndav = num_matmul;
        MatrixRef p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        MatrixRef hp(nullptr, x.m, x.n);
        double ff[2];
        double &error = ff[0], &func = ff[1];
        double beta = 0.0, old_beta = 0.0;
        r.allocate();
        p.allocate();
        hp.allocate();
        int nw = (int)ws.size();
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < i; j++) {
                double ww = dot(ws[j], ws[i]);
                iadd(ws[i], ws[j], -ww);
                iadd(aws[i], aws[j], -ww);
            }
            double w_normsq = sqrt(dot(ws[i], ws[i]));
            assert(w_normsq > 1E-14);
            iscale(ws[i], 1.0 / w_normsq);
            iscale(aws[i], 1.0 / w_normsq);
        }
        MatrixRef winv(nullptr, nw, nw);
        MatrixRef mu(nullptr, nw, 1);
        winv.allocate();
        mu.allocate();
        r.clear();
        p.clear();
        hp.clear();
        op(x, r);
        if (consta != 0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            if (nw != 0) {
                for (int i = 0; i < nw; i++)
                    for (int j = 0; j <= i; j++) {
                        winv(i, j) = dot(aws[i], ws[j]);
                        winv(j, i) = winv(i, j);
                    }
                inverse(winv);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += dot(r, ws[j]) * winv(i, j);
                    iadd(x, ws[i], mu.data[i]);
                    iadd(r, aws[i], -mu.data[i]);
                }
            }
            cg_precondition(p, r, aa);
            if (nw != 0) {
                op(p, hp);
                if (consta != 0)
                    iadd(hp, p, consta);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += dot(hp, ws[j]) * winv(i, j);
                    iadd(p, ws[i], -mu.data[i]);
                    iadd(hp, aws[i], -mu.data[i]);
                }
            }
            beta = dot(p, r);
            error = dot(r, r);
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
            mu.deallocate();
            winv.deallocate();
            hp.deallocate();
            p.deallocate();
            r.deallocate();
            nmult = 1;
            d_alloc->deallocate(pss.data, deflation_max_size * x.size());
            d_alloc->deallocate(pbs.data, deflation_max_size * x.size());
            return func;
        }
        old_beta = beta;
        if (pcomm != nullptr)
            pcomm->broadcast(p.data, p.size(), pcomm->root);
        MatrixRef z(nullptr, x.m, x.n), az(nullptr, x.m, x.n);
        z.allocate();
        az.allocate();
        xiter = 0;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (nw == 0) {
                hp.clear();
                op(p, hp);
                if (consta != 0)
                    iadd(hp, p, consta);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                double alpha = old_beta / dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                cg_precondition(z, r, aa);
                error = dot(r, r);
                beta = dot(z, r);
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
                    double gamma = beta / old_beta;
                    old_beta = beta;
                    iscale(p, gamma);
                    iadd(p, z, 1.0);
                    if (nw != 0) {
                        az.clear();
                        op(z, az);
                        if (consta != 0)
                            iadd(az, z, consta);
                        iscale(hp, gamma);
                        iadd(hp, az, 1.0);
                        mu.clear();
                        for (int i = 0; i < nw; i++) {
                            for (int j = 0; j < nw; j++)
                                mu.data[i] += dot(az, ws[j]) * winv(i, j);
                            iadd(p, ws[i], -mu.data[i]);
                            iadd(hp, aws[i], -mu.data[i]);
                        }
                    }
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
        az.deallocate();
        z.deallocate();
        mu.deallocate();
        winv.deallocate();
        hp.deallocate();
        p.deallocate();
        r.deallocate();
        if (pcomm != nullptr)
            pcomm->broadcast(x.data, x.size(), pcomm->root);
        d_alloc->deallocate(pss.data, deflation_max_size * x.size());
        d_alloc->deallocate(pbs.data, deflation_max_size * x.size());
        return func;
    }
    // Solve x in linear equation H x = b
    // by applying deflated CG method
    // where H is symmetric and positive-definite
    // H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static double davidson_projected_deflated_conjugate_gradient(
        MatMul &op, const DiagonalMatrix &aa, MatrixRef x, MatrixRef b, int k,
        int &nmult, int &ndav, double consta = 0.0, bool iprint = false,
        const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
        double conv_thrd_p = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        int deflation_min_size = 2, int deflation_max_size = 50) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        if (deflation_max_size > (int)x.size())
            deflation_max_size = (int)x.size();
        if (deflation_min_size > deflation_max_size)
            deflation_min_size = deflation_max_size;
        if (deflation_min_size < k)
            deflation_min_size = k;
        MatrixRef pbs(nullptr, (MKL_INT)(deflation_max_size * x.size()), 1);
        MatrixRef pss(nullptr, (MKL_INT)(deflation_max_size * x.size()), 1);
        pbs.data = d_alloc->allocate(deflation_max_size * x.size());
        pss.data = d_alloc->allocate(deflation_max_size * x.size());
        vector<MatrixRef> ws(deflation_max_size, MatrixRef(nullptr, x.m, x.n));
        vector<MatrixRef> aws(deflation_max_size, MatrixRef(nullptr, x.m, x.n));
        for (int i = 0; i < deflation_max_size; i++) {
            ws[i].data = pbs.data + ws[i].size() * i;
            aws[i].data = pss.data + aws[i].size() * i;
        }
        copy(ws[0], x);
        iscale(ws[0], 1.0 / sqrt(dot(ws[0], ws[0])));
        MatrixRef q(nullptr, x.m, x.n);
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.allocate();
        int ck = 0, msig = 0, m = 1, xiter = 0;
        double qq;
        if (iprint)
            cout << endl;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (pcomm != nullptr && xiter != 1)
                pcomm->broadcast(pbs.data + x.size() * msig,
                                 x.size() * (m - msig), pcomm->root);
            for (int i = msig; i < m; i++, msig++) {
                aws[i].clear();
                op(ws[i], aws[i]);
                if (consta != 0)
                    iadd(aws[i], ws[i], consta);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                DiagonalMatrix ld(nullptr, m);
                MatrixRef alpha(nullptr, m, m);
                ld.allocate();
                alpha.allocate();
                vector<MatrixRef> tmp(m, MatrixRef(nullptr, x.m, x.n));
                for (int i = 0; i < m; i++)
                    tmp[i].allocate();
                int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
                {
#ifdef _MSC_VER
#pragma omp for schedule(dynamic)
                    for (int ij = 0; ij < m * m; ij++) {
                        int i = ij / m, j = ij % m;
#else
#pragma omp for schedule(dynamic) collapse(2)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++) {
#endif
                        if (j <= i)
                            alpha(i, j) = dot(ws[i], aws[j]);
                    }
#pragma omp single
                    eigs(alpha, ld);
                    // note alpha row/column is diff from python
                    // b[1:m] = np.dot(b[:], alpha[:, 1:m])
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], ws[j]);
                        iscale(ws[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(ws[j], tmp[i], alpha(j, i));
                                // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], aws[j]);
                        iscale(aws[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(aws[j], tmp[i], alpha(j, i));
                }
                threading->activate_normal();
                for (int i = m - 1; i >= 0; i--)
                    tmp[i].deallocate();
                alpha.deallocate();
                for (int i = 0; i < ck; i++) {
                    copy(q, aws[i]);
                    iadd(q, ws[i], -ld(i, i));
                    if (dot(q, q) >= conv_thrd_p) {
                        ck = i;
                        break;
                    }
                }
                copy(q, aws[ck]);
                iadd(q, ws[ck], -ld(ck, ck));
                qq = dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8) << ld.data[ck]
                         << scientific << setw(13) << setprecision(2) << qq
                         << endl;
                olsen_precondition(q, ws[ck], ld.data[ck], aa);
                ld.deallocate();
            }
            if (pcomm != nullptr) {
                pcomm->broadcast(&qq, 1, pcomm->root);
                pcomm->broadcast(&ck, 1, pcomm->root);
            }
            if (m == deflation_max_size && ck >= k - 1 && qq < conv_thrd_p) {
                ck = k;
                break;
            }
            if (m == deflation_max_size &&
                xiter + (deflation_max_size - deflation_min_size) >=
                    soft_max_iter)
                break;
            if (qq < conv_thrd_p && ck < k - 1)
                ck++;
            else {
                if (m >= deflation_max_size)
                    m = msig = deflation_min_size;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    for (int j = 0; j < m; j++)
                        iadd(q, ws[j], -dot(ws[j], q));
                    iscale(q, 1.0 / sqrt(dot(q, q)));
                    copy(ws[m], q);
                }
                m++;
            }
        }
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.deallocate();
        assert(m == deflation_max_size);
        ndav = xiter;
        MatrixRef p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        MatrixRef hp(nullptr, x.m, x.n);
        double ff[2];
        double &error = ff[0], &func = ff[1];
        double beta = 0.0, old_beta = 0.0;
        r.allocate();
        p.allocate();
        hp.allocate();
        int nw = (int)ws.size();
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < i; j++) {
                double ww = dot(ws[j], ws[i]);
                iadd(ws[i], ws[j], -ww);
                iadd(aws[i], aws[j], -ww);
            }
            double w_normsq = sqrt(dot(ws[i], ws[i]));
            assert(w_normsq > 1E-14);
            iscale(ws[i], 1.0 / w_normsq);
            iscale(aws[i], 1.0 / w_normsq);
        }
        MatrixRef winv(nullptr, nw, nw);
        MatrixRef mu(nullptr, nw, 1);
        winv.allocate();
        mu.allocate();
        r.clear();
        p.clear();
        hp.clear();
        op(x, r);
        if (consta != 0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            if (nw != 0) {
                for (int i = 0; i < nw; i++)
                    for (int j = 0; j <= i; j++) {
                        winv(i, j) = dot(aws[i], ws[j]);
                        winv(j, i) = winv(i, j);
                    }
                inverse(winv);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += dot(r, ws[j]) * winv(i, j);
                    iadd(x, ws[i], mu.data[i]);
                    iadd(r, aws[i], -mu.data[i]);
                }
            }
            cg_precondition(p, r, aa);
            if (nw != 0) {
                op(p, hp);
                if (consta != 0)
                    iadd(hp, p, consta);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += dot(hp, ws[j]) * winv(i, j);
                    iadd(p, ws[i], -mu.data[i]);
                    iadd(hp, aws[i], -mu.data[i]);
                }
            }
            beta = dot(p, r);
            error = dot(r, r);
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
            mu.deallocate();
            winv.deallocate();
            hp.deallocate();
            p.deallocate();
            r.deallocate();
            nmult = 1;
            d_alloc->deallocate(pss.data, deflation_max_size * x.size());
            d_alloc->deallocate(pbs.data, deflation_max_size * x.size());
            return func;
        }
        old_beta = beta;
        if (pcomm != nullptr)
            pcomm->broadcast(p.data, p.size(), pcomm->root);
        MatrixRef z(nullptr, x.m, x.n), az(nullptr, x.m, x.n);
        z.allocate();
        az.allocate();
        xiter = 0;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (nw == 0) {
                hp.clear();
                op(p, hp);
                if (consta != 0)
                    iadd(hp, p, consta);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                double alpha = old_beta / dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                cg_precondition(z, r, aa);
                error = dot(r, r);
                beta = dot(z, r);
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
                    double gamma = beta / old_beta;
                    old_beta = beta;
                    iscale(p, gamma);
                    iadd(p, z, 1.0);
                    if (nw != 0) {
                        az.clear();
                        op(z, az);
                        if (consta != 0)
                            iadd(az, z, consta);
                        iscale(hp, gamma);
                        iadd(hp, az, 1.0);
                        mu.clear();
                        for (int i = 0; i < nw; i++) {
                            for (int j = 0; j < nw; j++)
                                mu.data[i] += dot(az, ws[j]) * winv(i, j);
                            iadd(p, ws[i], -mu.data[i]);
                            iadd(hp, aws[i], -mu.data[i]);
                        }
                    }
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
        az.deallocate();
        z.deallocate();
        mu.deallocate();
        winv.deallocate();
        hp.deallocate();
        p.deallocate();
        r.deallocate();
        if (pcomm != nullptr)
            pcomm->broadcast(x.data, x.size(), pcomm->root);
        d_alloc->deallocate(pss.data, deflation_max_size * x.size());
        d_alloc->deallocate(pbs.data, deflation_max_size * x.size());
        return func;
    }
    // Solve x in linear equation H x = b
    // by applying deflated CG method
    // where H is symmetric and positive-definite
    // H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static double deflated_conjugate_gradient(
        MatMul &op, const DiagonalMatrix &aa, MatrixRef x, MatrixRef b,
        int &nmult, double consta = 0.0, bool iprint = false,
        const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
        int max_iter = 5000, int soft_max_iter = -1,
        const vector<MatrixRef> &ws = vector<MatrixRef>()) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MatrixRef p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        MatrixRef hp(nullptr, x.m, x.n);
        double ff[2];
        double &error = ff[0], &func = ff[1];
        double beta = 0.0, old_beta = 0.0;
        r.allocate();
        p.allocate();
        hp.allocate();
        int nw = (int)ws.size();
        vector<MatrixRef> aws(nw, MatrixRef(nullptr, x.m, x.n));
        MatrixRef paws(nullptr, (MKL_INT)(nw * x.size()), 1);
        paws.data = d_alloc->allocate(nw * x.size());
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < i; j++)
                iadd(ws[i], ws[j], -dot(ws[j], ws[i]));
            double w_normsq = sqrt(dot(ws[i], ws[i]));
            assert(w_normsq > 1E-14);
            iscale(ws[i], 1.0 / w_normsq);
            aws[i].data = paws.data + ws[i].size() * i;
            aws[i].clear();
            op(ws[i], aws[i]);
            if (consta != 0)
                iadd(aws[i], ws[i], consta);
        }
        MatrixRef winv(nullptr, nw, nw);
        MatrixRef mu(nullptr, nw, 1);
        winv.allocate();
        mu.allocate();
        r.clear();
        p.clear();
        hp.clear();
        op(x, r);
        if (consta != 0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            if (nw != 0) {
                for (int i = 0; i < nw; i++)
                    for (int j = 0; j <= i; j++) {
                        winv(i, j) = dot(aws[i], ws[j]);
                        winv(j, i) = winv(i, j);
                    }
                inverse(winv);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += dot(r, ws[j]) * winv(i, j);
                    iadd(x, ws[i], mu.data[i]);
                    iadd(r, aws[i], -mu.data[i]);
                }
            }
            cg_precondition(p, r, aa);
            if (nw != 0) {
                op(p, hp);
                if (consta != 0)
                    iadd(hp, p, consta);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += dot(hp, ws[j]) * winv(i, j);
                    iadd(p, ws[i], -mu.data[i]);
                    iadd(hp, aws[i], -mu.data[i]);
                }
            }
            beta = dot(p, r);
            error = dot(r, r);
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
            mu.deallocate();
            winv.deallocate();
            hp.deallocate();
            p.deallocate();
            r.deallocate();
            nmult = 1;
            d_alloc->deallocate(paws.data, nw * x.size());
            return func;
        }
        old_beta = beta;
        if (pcomm != nullptr)
            pcomm->broadcast(p.data, p.size(), pcomm->root);
        MatrixRef z(nullptr, x.m, x.n), az(nullptr, x.m, x.n);
        z.allocate();
        az.allocate();
        int xiter = 0;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (nw == 0) {
                hp.clear();
                op(p, hp);
                if (consta != 0)
                    iadd(hp, p, consta);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                double alpha = old_beta / dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                cg_precondition(z, r, aa);
                error = dot(r, r);
                beta = dot(z, r);
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
                    double gamma = beta / old_beta;
                    old_beta = beta;
                    iscale(p, gamma);
                    iadd(p, z, 1.0);
                    if (nw != 0) {
                        az.clear();
                        op(z, az);
                        if (consta != 0)
                            iadd(az, z, consta);
                        iscale(hp, gamma);
                        iadd(hp, az, 1.0);
                        mu.clear();
                        for (int i = 0; i < nw; i++) {
                            for (int j = 0; j < nw; j++)
                                mu.data[i] += dot(az, ws[j]) * winv(i, j);
                            iadd(p, ws[i], -mu.data[i]);
                            iadd(hp, aws[i], -mu.data[i]);
                        }
                    }
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
        az.deallocate();
        z.deallocate();
        mu.deallocate();
        winv.deallocate();
        hp.deallocate();
        p.deallocate();
        r.deallocate();
        if (pcomm != nullptr)
            pcomm->broadcast(x.data, x.size(), pcomm->root);
        d_alloc->deallocate(paws.data, nw * x.size());
        return func;
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
        double beta = 0.0, old_beta = 0.0;
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
            cg_precondition(p, r, aa);
            beta = dot(p, r);
            error = dot(r, r);
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
        old_beta = beta;
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
                double alpha = old_beta / dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                cg_precondition(z, r, aa);
                error = dot(r, r);
                beta = dot(z, r);
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
                    double gamma = beta / old_beta;
                    old_beta = beta;
                    iadd(p, z, 1 / gamma);
                    iscale(p, gamma);
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
    // GCROT(m, k) method for solving x in linear equation H x = b
    template <typename MatMul, typename PComm>
    static double gcrotmk(MatMul &op, const DiagonalMatrix &aa, MatrixRef x,
                          MatrixRef b, int &nmult, int &niter, int m = 20,
                          int k = -1, double consta = 0.0, bool iprint = false,
                          const PComm &pcomm = nullptr, double conv_thrd = 5E-6,
                          int max_iter = 5000, int soft_max_iter = -1) {
        MatrixRef r(nullptr, x.m, x.n), w(nullptr, x.m, x.n);
        double ff[4];
        double &beta = ff[0], &rr = ff[1];
        double &func = ff[2];
        r.allocate();
        w.allocate();
        r.clear();
        op(x, r);
        if (consta != 0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            func = dot(x, b);
            beta = norm(r);
        }
        if (pcomm != nullptr)
            pcomm->broadcast(&beta, 4, pcomm->root);
        if (iprint)
            cout << endl;
        if (k == -1)
            k = m;
        int xiter = 0, jiter = 1, nn = k + m + 2;
        vector<MatrixRef> cvs(nn, MatrixRef(nullptr, x.m, x.n));
        vector<MatrixRef> uzs(nn, MatrixRef(nullptr, x.m, x.n));
        vector<double> pcus;
        pcus.reserve(x.size() * 2 * nn);
        for (int i = 0; i < nn; i++) {
            cvs[i].data = pcus.data() + cvs[i].size() * i;
            uzs[i].data = pcus.data() + uzs[i].size() * (i + nn);
        }
        int ncs = 0, icu = 0;
        MatrixRef bmat(nullptr, k, k + m);
        MatrixRef hmat(nullptr, k + m + 1, k + m);
        MatrixRef ys(nullptr, k + m, 1);
        MatrixRef bys(nullptr, k, 1);
        MatrixRef hys(nullptr, k + m + 1, 1);
        bmat.allocate();
        hmat.allocate();
        ys.allocate();
        bys.allocate();
        hys.allocate();
        while (jiter < max_iter &&
               (soft_max_iter == -1 || jiter < soft_max_iter)) {
            xiter++;
            if (iprint)
                cout << setw(6) << xiter << setw(6) << jiter << fixed
                     << setw(15) << setprecision(8) << func << scientific
                     << setw(13) << setprecision(2) << beta * beta << endl;
            if (beta * beta < conv_thrd)
                break;
            int ml = m + max(k - ncs, 0), ivz = icu + ncs + 1, nz = 0;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                iadd(cvs[ivz % nn], r, 1 / beta, 0.0);
                hmat.clear();
                hys.clear();
                hys.data[0] = beta;
            }
            for (int j = 0; j < ml; j++) {
                jiter++;
                MatrixRef z(uzs[(ivz + j) % nn].data, x.m, x.n);
                if (pcomm == nullptr || pcomm->root == pcomm->rank)
                    cg_precondition(z, cvs[(ivz + j) % nn], aa);
                if (pcomm != nullptr)
                    pcomm->broadcast(z.data, z.size(), pcomm->root);
                w.clear();
                op(z, w);
                if (consta != 0)
                    iadd(w, z, consta);
                nz = j + 1;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    for (int i = 0; i < ncs; i++) {
                        bmat(i, j) = dot(cvs[(icu + i) % nn], w);
                        iadd(w, cvs[(icu + i) % nn], -bmat(i, j));
                    }
                    for (int i = 0; i < nz; i++) {
                        hmat(i, j) = dot(cvs[(ivz + i) % nn], w);
                        iadd(w, cvs[(ivz + i) % nn], -hmat(i, j));
                    }
                    hmat(j + 1, j) = norm(w);
                    iadd(cvs[(ivz + nz) % nn], w, 1.0 / hmat(j + 1, j), false,
                         0.0);
                    rr = least_squares(MatrixRef(hmat.data, j + 2, hmat.n),
                                       MatrixRef(hys.data, j + 2, 1),
                                       MatrixRef(ys.data, j + 1, 1));
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(&rr, 1, pcomm->root);
                if (rr * rr < conv_thrd)
                    break;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                multiply(MatrixRef(bmat.data, ncs, bmat.n), false,
                         MatrixRef(ys.data, nz, 1), false,
                         MatrixRef(bys.data, ncs, 1), 1.0, 0.0);
                multiply(MatrixRef(hmat.data, nz + 1, hmat.n), false,
                         MatrixRef(ys.data, nz, 1), false,
                         MatrixRef(hys.data, nz + 1, 1), 1.0, 0.0);
                for (int i = 0; i < nz; i++)
                    iadd(uzs[(icu + ncs) % nn], uzs[(ivz + i) % nn], ys(i, 0),
                         false, !!i);
                for (int i = 0; i < ncs; i++)
                    iadd(uzs[(icu + ncs) % nn], uzs[(icu + i) % nn], -bys(i, 0),
                         false);
                for (int i = 0; i < nz + 1; i++)
                    iadd(cvs[(icu + ncs) % nn], cvs[(ivz + i) % nn], hys(i, 0),
                         false, !!i);
                double alpha = norm(cvs[(icu + ncs) % nn]);
                iscale(cvs[(icu + ncs) % nn], 1 / alpha);
                iscale(uzs[(icu + ncs) % nn], 1 / alpha);
                double gamma = dot(cvs[(icu + ncs) % nn], r);
                iadd(r, cvs[(icu + ncs) % nn], -gamma);
                iadd(x, uzs[(icu + ncs) % nn], gamma);
                func = dot(x, b);
                beta = norm(r);
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&beta, 4, pcomm->root);
            if (ncs == k)
                icu = (icu + 1) % nn;
            else
                ncs++;
        }
        if (jiter >= max_iter && beta * beta >= conv_thrd) {
            cout << "Error : linear solver GCROT(m, k) not converged!" << endl;
            assert(false);
        }
        nmult = jiter;
        niter = xiter + 1;
        hys.deallocate();
        bys.deallocate();
        ys.deallocate();
        hmat.deallocate();
        bmat.deallocate();
        w.deallocate();
        r.deallocate();
        if (pcomm != nullptr)
            pcomm->broadcast(x.data, x.size(), pcomm->root);
        return func;
    }
};

} // namespace block2
