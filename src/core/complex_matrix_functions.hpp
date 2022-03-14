
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

#include "matrix_functions.hpp"
#include "utils.hpp"
#include <complex>

using namespace std;

namespace block2 {

extern "C" {

#ifndef _HAS_INTEL_MKL

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void zdscal(const MKL_INT *n, const double *sa, complex<double> *sx,
                   const MKL_INT *incx) noexcept;

// vector [sx] = complex [sa] * vector [sx]
extern void zscal(const MKL_INT *n, const complex<double> *sa,
                  complex<double> *sx, const MKL_INT *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void zcopy(const MKL_INT *n, const complex<double> *dx,
                  const MKL_INT *incx, complex<double> *dy,
                  const MKL_INT *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + complex [sa] * vector [sx]
extern void zaxpy(const MKL_INT *n, const complex<double> *sa,
                  const complex<double> *sx, const MKL_INT *incx,
                  complex<double> *sy, const MKL_INT *incy) noexcept;

// vector dot product
// extern void zdotc(complex<double> *pres, const MKL_INT *n,
//                   const complex<double> *zx, const MKL_INT *incx,
//                   const complex<double> *zy, const MKL_INT *incy) noexcept;

// Euclidean norm of a vector
extern double dznrm2(const MKL_INT *n, const complex<double> *x,
                     const MKL_INT *incx) noexcept;

// matrix multiplication
// mat [c] = complex [alpha] * mat [a] * mat [b] + complex [beta] * mat [c]
extern void zgemm(const char *transa, const char *transb, const MKL_INT *m,
                  const MKL_INT *n, const MKL_INT *k,
                  const complex<double> *alpha, const complex<double> *a,
                  const MKL_INT *lda, const complex<double> *b,
                  const MKL_INT *ldb, const complex<double> *beta,
                  complex<double> *c, const MKL_INT *ldc) noexcept;

// LU factorization
extern void zgetrf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info);

// matrix inverse
extern void zgetri(const MKL_INT *n, complex<double> *a, const MKL_INT *lda,
                   MKL_INT *ipiv, complex<double> *work, const MKL_INT *lwork,
                   MKL_INT *info);

// eigenvalue problem
extern void zgeev(const char *jobvl, const char *jobvr, const MKL_INT *n,
                  complex<double> *a, const MKL_INT *lda, complex<double> *w,
                  complex<double> *vl, const MKL_INT *ldvl, complex<double> *vr,
                  const MKL_INT *ldvr, complex<double> *work,
                  const MKL_INT *lwork, double *rwork, MKL_INT *info);

// matrix-vector multiplication
// vec [y] = complex [alpha] * mat [a] * vec [x] + complex [beta] * vec [y]
extern void zgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const complex<double> *alpha, const complex<double> *a,
                  const MKL_INT *lda, const complex<double> *x,
                  const MKL_INT *incx, const complex<double> *beta,
                  complex<double> *y, const MKL_INT *incy) noexcept;

// linear system a * x = b
extern void zgesv(const MKL_INT *n, const MKL_INT *nrhs, complex<double> *a,
                  const MKL_INT *lda, MKL_INT *ipiv, complex<double> *b,
                  const MKL_INT *ldb, MKL_INT *info);

// least squares problem a * x = b
extern void zgels(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const MKL_INT *nrhs, complex<double> *a, const MKL_INT *lda,
                  complex<double> *b, const MKL_INT *ldb, complex<double> *work,
                  const MKL_INT *lwork, MKL_INT *info);

// matrix copy
// mat [b] = mat [a]
extern void zlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const complex<double> *a, const MKL_INT *lda,
                   complex<double> *b, const MKL_INT *ldb);

// QR factorization
extern void zgeqrf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, complex<double> *tau,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info);
extern void zungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<double> *a, const MKL_INT *lda,
                   const complex<double> *tau, complex<double> *work,
                   const MKL_INT *lwork, MKL_INT *info);

// LQ factorization
extern void zgelqf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, complex<double> *tau,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info);
extern void zunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<double> *a, const MKL_INT *lda,
                   const complex<double> *tau, complex<double> *work,
                   const MKL_INT *lwork, MKL_INT *info);

// eigenvalue problem
extern void zheev(const char *jobz, const char *uplo, const MKL_INT *n,
                  complex<double> *a, const MKL_INT *lda, double *w,
                  complex<double> *work, const MKL_INT *lwork, double *rwork,
                  MKL_INT *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void zgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, complex<double> *a, const MKL_INT *lda,
                   double *s, complex<double> *u, const MKL_INT *ldu,
                   complex<double> *vt, const MKL_INT *ldvt,
                   complex<double> *work, const MKL_INT *lwork, double *rwork,
                   MKL_INT *info);

#endif
}

template <typename FL>
inline void xgemm(const char *transa, const char *transb, const MKL_INT *m,
                  const MKL_INT *n, const MKL_INT *k, const FL *alpha,
                  const FL *a, const MKL_INT *lda, const FL *b,
                  const MKL_INT *ldb, const FL *beta, FL *c,
                  const MKL_INT *ldc) noexcept;

template <>
inline void xgemm<double>(const char *transa, const char *transb,
                          const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          const double *alpha, const double *a,
                          const MKL_INT *lda, const double *b,
                          const MKL_INT *ldb, const double *beta, double *c,
                          const MKL_INT *ldc) noexcept {
    return dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void
xgemm<complex<double>>(const char *transa, const char *transb, const MKL_INT *m,
                       const MKL_INT *n, const MKL_INT *k,
                       const complex<double> *alpha, const complex<double> *a,
                       const MKL_INT *lda, const complex<double> *b,
                       const MKL_INT *ldb, const complex<double> *beta,
                       complex<double> *c, const MKL_INT *ldc) noexcept {
    return zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename FL>
inline void xscal(const MKL_INT *n, const FL *sa, FL *sx,
                  const MKL_INT *incx) noexcept;

template <>
inline void xscal<double>(const MKL_INT *n, const double *sa, double *sx,
                          const MKL_INT *incx) noexcept {
    dscal(n, sa, sx, incx);
}

template <>
inline void xscal<complex<double>>(const MKL_INT *n, const complex<double> *sa,
                                   complex<double> *sx,
                                   const MKL_INT *incx) noexcept {
    zscal(n, sa, sx, incx);
}

template <typename FL>
inline void xdscal(const MKL_INT *n, const double *sa, FL *sx,
                   const MKL_INT *incx) noexcept;

template <>
inline void xdscal<double>(const MKL_INT *n, const double *sa, double *sx,
                           const MKL_INT *incx) noexcept {
    dscal(n, sa, sx, incx);
}

template <>
inline void xdscal<complex<double>>(const MKL_INT *n, const double *sa,
                                    complex<double> *sx,
                                    const MKL_INT *incx) noexcept {
    zdscal(n, sa, sx, incx);
}

template <typename FL>
inline double xnrm2(const MKL_INT *n, const FL *x,
                    const MKL_INT *incx) noexcept;

template <>
inline double xnrm2<double>(const MKL_INT *n, const double *x,
                            const MKL_INT *incx) noexcept {
    return dnrm2(n, x, incx);
}

template <>
inline double xnrm2<complex<double>>(const MKL_INT *n, const complex<double> *x,
                                     const MKL_INT *incx) noexcept {
    return dznrm2(n, x, incx);
}

template <typename FL>
inline void xcopy(const MKL_INT *n, const FL *dx, const MKL_INT *incx, FL *dy,
                  const MKL_INT *incy) noexcept;

template <>
inline void xcopy<double>(const MKL_INT *n, const double *dx,
                          const MKL_INT *incx, double *dy,
                          const MKL_INT *incy) noexcept {
    dcopy(n, dx, incx, dy, incy);
}

template <>
inline void xcopy<complex<double>>(const MKL_INT *n, const complex<double> *dx,
                                   const MKL_INT *incx, complex<double> *dy,
                                   const MKL_INT *incy) noexcept {
    zcopy(n, dx, incx, dy, incy);
}

template <typename FL>
inline FL xdot(const MKL_INT *n, const FL *dx, const MKL_INT *incx,
               const FL *dy, const MKL_INT *incy) noexcept;

template <>
inline double xdot<double>(const MKL_INT *n, const double *dx,
                           const MKL_INT *incx, const double *dy,
                           const MKL_INT *incy) noexcept {
    return ddot(n, dx, incx, dy, incy);
}

template <>
inline complex<double>
xdot<complex<double>>(const MKL_INT *n, const complex<double> *dx,
                      const MKL_INT *incx, const complex<double> *dy,
                      const MKL_INT *incy) noexcept {
    static const complex<double> x = 1.0, zz = 0.0;
    MKL_INT inc = 1;
    complex<double> r;
    zgemm("n", "t", &inc, &inc, n, &x, dy, incy, dx, incx, &zz, &r, &inc);
    return r;
}

template <typename FL>
inline void xaxpy(const MKL_INT *n, const FL *sa, const FL *sx,
                  const MKL_INT *incx, FL *sy, const MKL_INT *incy) noexcept;

template <>
inline void xaxpy<double>(const MKL_INT *n, const double *sa, const double *sx,
                          const MKL_INT *incx, double *sy,
                          const MKL_INT *incy) noexcept {
    daxpy(n, sa, sx, incx, sy, incy);
}

template <>
inline void xaxpy<complex<double>>(const MKL_INT *n, const complex<double> *sa,
                                   const complex<double> *sx,
                                   const MKL_INT *incx, complex<double> *sy,
                                   const MKL_INT *incy) noexcept {
    zaxpy(n, sa, sx, incx, sy, incy);
}

template <typename FL>
inline void xlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const FL *a, const MKL_INT *lda, FL *b, const MKL_INT *ldb);

template <>
inline void xlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const double *a, const MKL_INT *lda, double *b,
                   const MKL_INT *ldb) {
    dlacpy(uplo, m, n, a, lda, b, ldb);
}
template <>
inline void xlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const complex<double> *a, const MKL_INT *lda,
                   complex<double> *b, const MKL_INT *ldb) {
    zlacpy(uplo, m, n, a, lda, b, ldb);
}

template <typename FL>
inline void xgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const FL *alpha, const FL *a, const MKL_INT *lda, const FL *x,
                  const MKL_INT *incx, const FL *beta, FL *y,
                  const MKL_INT *incy);

template <>
inline void xgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const double *alpha, const double *a, const MKL_INT *lda,
                  const double *x, const MKL_INT *incx, const double *beta,
                  double *y, const MKL_INT *incy) {
    dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void xgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const complex<double> *alpha, const complex<double> *a,
                  const MKL_INT *lda, const complex<double> *x,
                  const MKL_INT *incx, const complex<double> *beta,
                  complex<double> *y, const MKL_INT *incy) {
    zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <typename FL>
inline void xgeqrf(const MKL_INT *m, const MKL_INT *n, FL *a,
                   const MKL_INT *lda, FL *tau, FL *work, const MKL_INT *lwork,
                   MKL_INT *info);
template <>
inline void xgeqrf(const MKL_INT *m, const MKL_INT *n, double *a,
                   const MKL_INT *lda, double *tau, double *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    dgeqrf(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgeqrf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, complex<double> *tau,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info) {
    zgeqrf(m, n, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, FL *a,
                   const MKL_INT *lda, const FL *tau, FL *work,
                   const MKL_INT *lwork, MKL_INT *info);
template <>
inline void xungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   double *a, const MKL_INT *lda, const double *tau,
                   double *work, const MKL_INT *lwork, MKL_INT *info) {
    dorgqr(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<double> *a, const MKL_INT *lda,
                   const complex<double> *tau, complex<double> *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    zungqr(m, n, k, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xgelqf(const MKL_INT *m, const MKL_INT *n, FL *a,
                   const MKL_INT *lda, FL *tau, FL *work, const MKL_INT *lwork,
                   MKL_INT *info);
template <>
inline void xgelqf(const MKL_INT *m, const MKL_INT *n, double *a,
                   const MKL_INT *lda, double *tau, double *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    dgelqf(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgelqf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, complex<double> *tau,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info) {
    zgelqf(m, n, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, FL *a,
                   const MKL_INT *lda, const FL *tau, FL *work,
                   const MKL_INT *lwork, MKL_INT *info);
template <>
inline void xunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   double *a, const MKL_INT *lda, const double *tau,
                   double *work, const MKL_INT *lwork, MKL_INT *info) {
    dorglq(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<double> *a, const MKL_INT *lda,
                   const complex<double> *tau, complex<double> *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    zunglq(m, n, k, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, FL *a, const MKL_INT *lda, double *s,
                   FL *u, const MKL_INT *ldu, FL *vt, const MKL_INT *ldvt,
                   FL *work, const MKL_INT *lwork, MKL_INT *info);
template <>
inline void xgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, double *a, const MKL_INT *lda, double *s,
                   double *u, const MKL_INT *ldu, double *vt,
                   const MKL_INT *ldvt, double *work, const MKL_INT *lwork,
                   MKL_INT *info) {
    dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}
template <>
inline void xgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, complex<double> *a, const MKL_INT *lda,
                   double *s, complex<double> *u, const MKL_INT *ldu,
                   complex<double> *vt, const MKL_INT *ldvt,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info) {
    vector<double> rwork;
    rwork.reserve(5 * min(*m, *n));
    zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork,
           rwork.data(), info);
}

// General matrix operations
template <typename FL> struct GMatrixFunctions;

// Dense complex number matrix operations
template <> struct GMatrixFunctions<complex<double>> {
    // a = re + im i
    static void fill_complex(const ComplexMatrixRef &a, const MatrixRef &re,
                             const MatrixRef &im) {
        if (re.data != nullptr)
            MatrixFunctions::copy(MatrixRef((double *)a.data, a.m, a.n), re, 2,
                                  1);
        if (im.data != nullptr)
            MatrixFunctions::copy(MatrixRef((double *)a.data + 1, a.m, a.n), im,
                                  2, 1);
    }
    // re + im i = a
    static void extract_complex(const ComplexMatrixRef &a, const MatrixRef &re,
                                const MatrixRef &im) {
        if (re.data != nullptr)
            MatrixFunctions::copy(re, MatrixRef((double *)a.data, a.m, a.n), 1,
                                  2);
        if (im.data != nullptr)
            MatrixFunctions::copy(im, MatrixRef((double *)a.data + 1, a.m, a.n),
                                  1, 2);
    }
    // a = b
    static void copy(const ComplexMatrixRef &a, const ComplexMatrixRef &b,
                     const MKL_INT inca = 1, const MKL_INT incb = 1) {
        assert(a.m == b.m && a.n == b.n);
        const MKL_INT n = a.m * a.n;
        zcopy(&n, b.data, &incb, a.data, &inca);
    }
    static void iscale(const ComplexMatrixRef &a, complex<double> scale,
                       const MKL_INT inc = 1) {
        MKL_INT n = a.m * a.n;
        zscal(&n, &scale, a.data, &inc);
    }
    static void keep_real(const ComplexMatrixRef &a) {
        const MKL_INT incx = 2;
        const double scale = 0.0;
        MKL_INT n = a.m * a.n;
        dscal(&n, &scale, (double *)a.data + 1, &incx);
    }
    static void conjugate(const ComplexMatrixRef &a) {
        const MKL_INT incx = 2;
        const double scale = -1.0;
        MKL_INT n = a.m * a.n;
        dscal(&n, &scale, (double *)a.data + 1, &incx);
    }
    // a = a + scale * op(b)
    // conj means conj trans
    static void iadd(const ComplexMatrixRef &a, const ComplexMatrixRef &b,
                     complex<double> scale, bool conj = false,
                     complex<double> cfactor = 1.0) {
        static const complex<double> x = 1.0;
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            MKL_INT n = a.m * a.n, inc = 1;
            if (cfactor == 1.0)
                zaxpy(&n, &scale, b.data, &inc, a.data, &inc);
            else
                zgemm("n", "n", &inc, &n, &inc, &scale, &x, &inc, b.data, &inc,
                      &cfactor, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n == b.m);
            const complex<double> one = 1.0;
            for (MKL_INT k = 0, inc = 1; k < b.n; k++)
                zgemm("c", "n", &b.m, &inc, &inc, &scale, &b(0, k), &b.n, &one,
                      &inc, &cfactor, &a(k, 0), &a.n);
        }
    }
    static double norm(const ComplexMatrixRef &a) {
        MKL_INT n = a.m * a.n, inc = 1;
        return dznrm2(&n, a.data, &inc);
    }
    // dot product (a ^ H, b)
    static complex<double> complex_dot(const ComplexMatrixRef &a,
                                       const ComplexMatrixRef &b) {
        static const complex<double> x = 1.0, zz = 0.0;
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        complex<double> r;
        // zdotc can sometimes return zero
        // zdotc(&r, &n, a.data, &inc, b.data, &inc);
        zgemm("c", "n", &inc, &inc, &n, &x, a.data, &n, b.data, &n, &zz, &r,
              &inc);
        return r;
    }
    // Computes norm more accurately
    static double norm_accurate(const ComplexMatrixRef &a) {
        MKL_INT n = a.m * a.n;
        // do re and im separately, as in numpy
        long double out_real = 0.0;
        long double out_imag = 0.0;
        long double compensate_real = 0.0;
        long double compensate_imag = 0.0;
        for (MKL_INT ii = 0; ii < n; ++ii) {
            long double &&xre = (long double)real(a.data[ii]);
            long double &&xim = (long double)imag(a.data[ii]);
            long double sumi_real = xre * xre;
            long double sumi_imag = xim * xim;
            // Kahan summation
            auto y_real = sumi_real - compensate_real;
            auto y_imag = sumi_imag - compensate_imag;
            const volatile long double t_real = out_real + y_real;
            const volatile long double t_imag = out_imag + y_imag;
            const volatile long double z_real = t_real - out_real;
            const volatile long double z_imag = t_imag - out_imag;
            compensate_real = z_real - y_real;
            compensate_imag = z_imag - y_imag;
            out_real = t_real;
            out_imag = t_imag;
        }
        long double out = sqrt(out_real + out_imag);
        return static_cast<double>(out);
    }
    template <typename T1, typename T2>
    static bool all_close(const T1 &a, const T2 &b, double atol = 1E-8,
                          double rtol = 1E-5, complex<double> scale = 1.0) {
        assert(a.m == b.m && a.n == b.n);
        for (MKL_INT i = 0; i < a.m; i++)
            for (MKL_INT j = 0; j < a.n; j++)
                if (abs(a(i, j) - scale * b(i, j)) > atol + rtol * abs(b(i, j)))
                    return false;
        return true;
    }
    // dot product (a ^ T, b)
    static complex<double> dot(const ComplexMatrixRef &a,
                               const ComplexMatrixRef &b) {
        static const complex<double> x = 1.0, zz = 0.0;
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        complex<double> r;
        zgemm("t", "n", &inc, &inc, &n, &x, a.data, &n, b.data, &n, &zz, &r,
              &inc);
        return r;
    }
    // matrix inverse
    static void inverse(const ComplexMatrixRef &a) {
        assert(a.m == a.n);
        vector<MKL_INT> ipiv;
        vector<complex<double>> work;
        ipiv.reserve(a.m);
        MKL_INT lwork = 34 * a.n, info = -1;
        work.reserve(lwork);
        zgetrf(&a.m, &a.n, a.data, &a.m, ipiv.data(), &info);
        assert(info == 0);
        zgetri(&a.n, a.data, &a.m, ipiv.data(), work.data(), &lwork, &info);
        assert(info == 0);
    }
    // least squares problem a x = b
    // return the residual (norm, not squared)
    // a.n is used as lda
    static double least_squares(const ComplexMatrixRef &a,
                                const ComplexMatrixRef &b,
                                const ComplexMatrixRef &x) {
        assert(a.m == b.m && a.n >= x.m && b.n == 1 && x.n == 1);
        vector<complex<double>> work, atr, xtr;
        MKL_INT lwork = 34 * min(a.m, x.m), info = -1, nrhs = 1,
                mn = max(a.m, x.m), nr = a.m - x.m;
        work.reserve(lwork);
        atr.reserve(a.size());
        xtr.reserve(mn);
        zcopy(&a.m, b.data, &nrhs, xtr.data(), &nrhs);
        for (MKL_INT i = 0; i < x.m; i++)
            zcopy(&a.m, a.data + i, &a.n, atr.data() + i * a.m, &nrhs);
        zgels("N", &a.m, &x.m, &nrhs, atr.data(), &a.m, xtr.data(), &mn,
              work.data(), &lwork, &info);
        assert(info == 0);
        zcopy(&x.m, xtr.data(), &nrhs, x.data, &nrhs);
        return nr > 0 ? dznrm2(&nr, xtr.data() + x.m, &nrhs) : 0;
    }
    // eigenvectors are row right-vectors: A u(j) = lambda(j) u(j)
    static void eig(const ComplexMatrixRef &a, const ComplexDiagonalMatrix &w) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(a.m == a.n && w.n == a.n);
        MKL_INT lwork = 34 * a.n, info;
        complex<double> *work = d_alloc->complex_allocate(lwork);
        double *rwork = d_alloc->allocate(a.m * 2);
        complex<double> *vl = d_alloc->complex_allocate(a.m * a.n);
        zgeev("V", "N", &a.n, a.data, &a.n, w.data, vl, &a.n, nullptr, &a.n,
              work, &lwork, rwork, &info);
        assert(info == 0);
        for (size_t k = 0; k < a.m * a.n; k++)
            a.data[k] = conj(vl[k]);
        d_alloc->complex_deallocate(vl, a.m * a.n);
        d_alloc->deallocate(rwork, a.m * 2);
        d_alloc->complex_deallocate(work, lwork);
    }
    // matrix logarithm using diagonalization
    static void logarithm(const ComplexMatrixRef &a) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(a.m == a.n);
        ComplexDiagonalMatrix w(nullptr, a.m);
        w.data = (complex<double> *)d_alloc->allocate(a.m * 2);
        ComplexMatrixRef wa(nullptr, a.m, a.n);
        wa.data = (complex<double> *)d_alloc->allocate(a.m * a.n * 2);
        ComplexMatrixRef ua(nullptr, a.m, a.n);
        ua.data = (complex<double> *)d_alloc->allocate(a.m * a.n * 2);
        memcpy(ua.data, a.data, sizeof(complex<double>) * a.size());
        eig(ua, w);
        for (MKL_INT i = 0; i < a.m; i++)
            for (MKL_INT j = 0; j < a.n; j++)
                wa(i, j) = ua(i, j) * log(w(i, i));
        inverse(ua);
        multiply(wa, true, ua, true, a, 1.0, 0.0);
        d_alloc->deallocate((double *)ua.data, a.m * a.n * 2);
        d_alloc->deallocate((double *)wa.data, a.m * a.n * 2);
        d_alloc->deallocate((double *)w.data, a.m * 2);
    }
    // solve a^T x[i, :] = b[i, :] => output in b; a will be overwritten
    static void linear(const ComplexMatrixRef &a, const ComplexMatrixRef &b) {
        assert(a.m == a.n && a.m == b.n);
        MKL_INT *work = (MKL_INT *)ialloc->allocate(a.n * _MINTSZ), info = -1;
        zgesv(&a.m, &b.m, a.data, &a.n, work, b.data, &a.n, &info);
        assert(info == 0);
        ialloc->deallocate(work, a.n * _MINTSZ);
    }
    // c.n is used for ldc; a.n is used for lda
    // conj can be 0 (no conj no trans), 1 (trans), 3 (conj trans)
    static void multiply(const ComplexMatrixRef &a, uint8_t conja,
                         const ComplexMatrixRef &b, uint8_t conjb,
                         const ComplexMatrixRef &c, complex<double> scale,
                         complex<double> cfactor) {
        static const char ntxc[5] = "ntxc";
        // if assertion failes here, check whether it is the case
        // where different bra and ket are used with the transpose rule
        // use no-transpose-rule to fix it
        if (!conja && !conjb) {
            assert(a.n >= b.m && c.m == a.m && c.n >= b.n);
            zgemm("n", "n", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (!conja && conjb != 2) {
            assert(a.n >= b.n && c.m == a.m && c.n >= b.m);
            zgemm(ntxc + conjb, "n", &b.m, &c.m, &b.n, &scale, b.data, &b.n,
                  a.data, &a.n, &cfactor, c.data, &c.n);
        } else if (conja != 2 && !conjb) {
            assert(a.m == b.m && c.m <= a.n && c.n >= b.n);
            zgemm("n", ntxc + conja, &b.n, &c.m, &b.m, &scale, b.data, &b.n,
                  a.data, &a.n, &cfactor, c.data, &c.n);
        } else if (conja != 2 && conjb != 2) {
            assert(a.m == b.n && c.m <= a.n && c.n >= b.m);
            zgemm(ntxc + conjb, ntxc + conja, &b.m, &c.m, &b.n, &scale, b.data,
                  &b.n, a.data, &a.n, &cfactor, c.data, &c.n);
        } else if (conja == 2 && conjb != 2) {
            const MKL_INT one = 1;
            for (MKL_INT k = 0; k < c.m; k++)
                zgemm(ntxc + conjb, "c", (conjb & 1) ? &b.m : &b.n, &one,
                      (conjb & 1) ? &b.n : &b.m, &scale, b.data, &b.n, &a(k, 0),
                      &one, &cfactor, &c(k, 0), &c.n);
        } else if (conja != 3 && conjb == 2) {
            const MKL_INT one = 1;
            for (MKL_INT k = 0; k < c.m; k++)
                zgemm(ntxc + (conja ^ 1), "c", &one, &b.n, &b.m, &scale,
                      (conja & 1) ? &a(0, k) : &a(k, 0), &a.n, b.data, &b.n,
                      &cfactor, &c(k, 0), &one);
        } else
            assert(false);
    }
    // c = bra(.T) * a * ket(.T)
    // return nflop
    // conj can be 0 (no conj no trans), 1 (trans), 2 (conj), 3 (conj trans)
    static size_t rotate(const ComplexMatrixRef &a, const ComplexMatrixRef &c,
                         const ComplexMatrixRef &bra, uint8_t conj_bra,
                         const ComplexMatrixRef &ket, uint8_t conj_ket,
                         complex<double> scale) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        if (conj_bra != 2 && conj_ket != 2) {
            ComplexMatrixRef work(nullptr, a.m, (conj_ket & 1) ? ket.m : ket.n);
            work.allocate(d_alloc);
            multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
            multiply(bra, conj_bra, work, false, c, scale, 1.0);
            work.deallocate(d_alloc);
            return (size_t)ket.m * ket.n * work.m +
                   (size_t)work.m * work.n * c.m;
        } else if (conj_bra != 2) {
            ComplexMatrixRef work(nullptr, ket.n, a.m);
            work.allocate(d_alloc);
            multiply(ket, 3, a, true, work, 1.0, 0.0);
            multiply(bra, conj_bra, work, true, c, scale, 1.0);
            work.deallocate(d_alloc);
            return (size_t)ket.m * ket.n * work.n +
                   (size_t)work.m * work.n * c.m;
        } else if (conj_ket != 2) {
            ComplexMatrixRef work(nullptr, a.n, bra.m);
            work.allocate(d_alloc);
            multiply(a, true, bra, 3, work, 1.0, 0.0);
            multiply(work, true, ket, conj_ket, c, scale, 1.0);
            work.deallocate(d_alloc);
            return (size_t)bra.m * bra.n * work.n +
                   (size_t)work.m * work.n * ((conj_ket & 1) ? ket.m : ket.n);
        } else {
            ComplexMatrixRef work(nullptr, ket.n, a.m);
            ComplexMatrixRef work2(nullptr, work.m, bra.m);
            work.allocate(d_alloc);
            work2.allocate(d_alloc);
            multiply(ket, 3, a, true, work, 1.0, 0.0);
            multiply(work, false, bra, 3, work2, 1.0, 0.0);
            transpose(c, work2, scale);
            work2.deallocate(d_alloc);
            work.deallocate(d_alloc);
            return (size_t)ket.m * ket.n * work.n +
                   (size_t)work.m * work.n * c.m + (size_t)work2.m * work2.n;
        }
        return 0;
    }
    // c(.T) = bra.T * a(.T) * ket
    // return nflop. (.T) is always transpose conjugate
    static size_t rotate(const ComplexMatrixRef &a, bool conj_a,
                         const ComplexMatrixRef &c, bool conj_c,
                         const ComplexMatrixRef &bra,
                         const ComplexMatrixRef &ket, complex<double> scale) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        ComplexMatrixRef work(nullptr, conj_a ? a.n : a.m, ket.n);
        work.allocate(d_alloc);
        multiply(a, conj_a ? 3 : 0, ket, false, work, 1.0, 0.0);
        if (!conj_c)
            multiply(bra, 3, work, false, c, scale, 1.0);
        else
            multiply(work, 3, bra, false, c, conj(scale), 1.0);
        work.deallocate(d_alloc);
        return (size_t)a.m * a.n * work.n + (size_t)work.m * work.n * bra.n;
    }
    // dleft == true : c = bra (= da x db) * a * ket
    // dleft == false: c = bra * a * ket (= da x db)
    // return nflop. conj means conj and trans
    // conj means conj and trans / none for bra, trans / conj for ket
    static size_t three_rotate(const ComplexMatrixRef &a,
                               const ComplexMatrixRef &c,
                               const ComplexMatrixRef &bra, bool conj_bra,
                               const ComplexMatrixRef &ket, bool conj_ket,
                               const ComplexMatrixRef &da, bool dconja,
                               const ComplexMatrixRef &db, bool dconjb,
                               bool dleft, complex<double> scale,
                               uint32_t stride) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            ComplexMatrixRef work(nullptr, am, conj_ket ? ket.m : ket.n);
            work.allocate(d_alloc);
            // work = a * ket
            multiply(ComplexMatrixRef(&a(ast, 0), am, a.n), false, ket,
                     conj_ket ? 1 : 2, work, 1.0, 0.0);
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * work
                multiply(db, dconjb ? 3 : 0, work, false,
                         ComplexMatrixRef(&c(cst, 0), cm, c.n),
                         scale * (dconja ? conj(*da.data) : *da.data), 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * work
                multiply(da, dconja ? 3 : 0, work, false,
                         ComplexMatrixRef(&c(cst, 0), cm, c.n),
                         scale * (dconjb ? conj(*db.data) : *db.data), 1.0);
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
            ComplexMatrixRef work(nullptr, a.m, kn);
            work.allocate(d_alloc);
            if (da.m == 1 && da.n == 1)
                // work = a * (1 x db)
                multiply(ComplexMatrixRef(&a(0, ast), a.m, a.n), false, db,
                         dconjb ? 1 : 2, work,
                         (!dconja ? conj(*da.data) : *da.data) * scale, 0.0);
            else if (db.m == 1 && db.n == 1)
                // work = a * (da x 1)
                multiply(ComplexMatrixRef(&a(0, ast), a.m, a.n), false, da,
                         dconja ? 1 : 2, work,
                         (!dconjb ? conj(*db.data) : *db.data) * scale, 0.0);
            else
                assert(false);
            // c = bra * work
            multiply(bra, conj_bra ? 3 : 0, work, false,
                     ComplexMatrixRef(&c(0, cst), c.m, c.n), 1.0, 1.0);
            work.deallocate(d_alloc);
            return (size_t)km * kn * work.m + (size_t)work.m * work.n * c.m;
        }
    }
    // dleft == true : c = a * ket
    // dleft == false: c = a * ket (= da x db)
    // return nflop
    static size_t
    three_rotate_tr_left(const ComplexMatrixRef &a, const ComplexMatrixRef &c,
                         const ComplexMatrixRef &bra, bool conj_bra,
                         const ComplexMatrixRef &ket, bool conj_ket,
                         const ComplexMatrixRef &da, bool dconja,
                         const ComplexMatrixRef &db, bool dconjb, bool dleft,
                         complex<double> scale, uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            multiply(ComplexMatrixRef(&a(ast, 0), am, a.n), false, ket,
                     conj_ket ? 1 : 2, ComplexMatrixRef(&c(cst, 0), cm, c.n),
                     scale, 1.0);
            return (size_t)ket.m * ket.n * am;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            if (da.m == 1 && da.n == 1)
                // c = a * (1 x db)
                multiply(ComplexMatrixRef(&a(0, ast), a.m, a.n), false, db,
                         dconjb ? 1 : 2, ComplexMatrixRef(&c(0, cst), c.m, c.n),
                         (!dconja ? conj(*da.data) : *da.data) * scale, 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = a * (da x 1)
                multiply(ComplexMatrixRef(&a(0, ast), a.m, a.n), false, da,
                         dconja ? 1 : 2, ComplexMatrixRef(&c(0, cst), c.m, c.n),
                         (!dconjb ? conj(*db.data) : *db.data) * scale, 1.0);
            else
                assert(false);
            return (size_t)km * kn * c.m;
        }
    }
    // dleft == true : c = bra (= da x db) * a
    // dleft == false: c = bra * a
    // return nflop
    static size_t
    three_rotate_tr_right(const ComplexMatrixRef &a, const ComplexMatrixRef &c,
                          const ComplexMatrixRef &bra, bool conj_bra,
                          const ComplexMatrixRef &ket, bool conj_ket,
                          const ComplexMatrixRef &da, bool dconja,
                          const ComplexMatrixRef &db, bool dconjb, bool dleft,
                          complex<double> scale, uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * a
                multiply(db, dconjb ? 3 : 0,
                         ComplexMatrixRef(&a(ast, 0), am, a.n), false,
                         ComplexMatrixRef(&c(cst, 0), cm, c.n),
                         scale * (dconja ? conj(*da.data) : *da.data), 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * a
                multiply(da, dconja ? 3 : 0,
                         ComplexMatrixRef(&a(ast, 0), am, a.n), false,
                         ComplexMatrixRef(&c(cst, 0), cm, c.n),
                         scale * (dconjb ? conj(*db.data) : *db.data), 1.0);
            else
                assert(false);
            return (size_t)am * a.n * cm;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            const complex<double> cfactor = 1.0;
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            zgemm("n", conj_bra ? "c" : "n", &kn, &c.m, &a.m, &scale,
                  &a(0, ast), &a.n, bra.data, &bra.n, &cfactor, &c(0, cst),
                  &c.n);
            return (size_t)a.m * a.n * c.m;
        }
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(uint8_t abconj,
                                        const ComplexMatrixRef &a,
                                        const ComplexMatrixRef &b,
                                        const ComplexMatrixRef &c,
                                        complex<double> scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const complex<double> cfactor = 1.0;
        const MKL_INT k = 1, lda = a.n + 1, ldb = b.n + 1;
        if (!(abconj & 1))
            zgemm(abconj & 2 ? "c" : "t", "n", &b.n, &a.n, &k, &scale, b.data,
                  &ldb, a.data, &lda, &cfactor, c.data, &c.n);
        else
            for (MKL_INT i = 0; i < a.n; i++)
                zgemm(abconj & 2 ? "c" : "t", "c", &b.n, &k, &k, &scale, b.data,
                      &ldb, a.data + i * lda, &k, &cfactor, c.data + i * c.n,
                      &c.n);
    }
    // diagonal element of three-matrix tensor product
    static void three_tensor_product_diagonal(
        uint8_t abconj, const ComplexMatrixRef &a, const ComplexMatrixRef &b,
        const ComplexMatrixRef &c, const ComplexMatrixRef &da, bool dconja,
        const ComplexMatrixRef &db, bool dconjb, bool dleft,
        complex<double> scale, uint32_t stride) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const complex<double> cfactor = 1.0;
        const MKL_INT dstrm = (MKL_INT)stride / (dleft ? a.m : b.m);
        const MKL_INT dstrn = (MKL_INT)stride % (dleft ? a.m : b.m);
        if (dstrn != dstrm)
            return;
        assert(da.m == da.n && db.m == db.n);
        const MKL_INT ddstr = 0;
        const MKL_INT k = 1, lda = a.n + 1, ldb = b.n + 1;
        const MKL_INT ldda = da.n + 1, lddb = db.n + 1;
        const bool ddconja =
            dconja ^ (dleft ? (abconj & 1) : ((abconj & 2) >> 1));
        const bool ddconjb =
            dconjb ^ (dleft ? (abconj & 1) : ((abconj & 2) >> 1));
        if (da.m == 1 && da.n == 1) {
            scale *= ddconja ? conj(*da.data) : *da.data;
            const MKL_INT dn = db.n - abs(ddstr);
            const complex<double> *bdata =
                dconjb ? &db(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &db(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft) {
                    // (1 x db) x b
                    if (!ddconjb)
                        zgemm(abconj & 2 ? "c" : "t", "n", &b.n, &dn, &k,
                              &scale, b.data, &ldb, bdata, &lddb, &cfactor,
                              &c(max(dstrn, dstrm), (MKL_INT)0), &c.n);
                    else
                        for (MKL_INT i = 0; i < dn; i++)
                            zgemm(abconj & 2 ? "c" : "t", "c", &b.n, &k, &k,
                                  &scale, b.data, &ldb, bdata + i * lddb, &k,
                                  &cfactor,
                                  &c(max(dstrn, dstrm) + i, (MKL_INT)0), &c.n);
                } else {
                    // a x (1 x db)
                    if (!(abconj & 1))
                        zgemm(ddconjb ? "c" : "t", "n", &dn, &a.n, &k, &scale,
                              bdata, &lddb, a.data, &lda, &cfactor,
                              &c(0, max(dstrn, dstrm)), &c.n);
                    else
                        for (MKL_INT i = 0; i < a.n; i++)
                            zgemm(ddconjb ? "c" : "t", "c", &dn, &k, &k, &scale,
                                  bdata, &lddb, a.data + i * lda, &k, &cfactor,
                                  &c(i, max(dstrn, dstrm)), &c.n);
                }
            }
        } else if (db.m == 1 && db.n == 1) {
            scale *= ddconjb ? conj(*db.data) : *db.data;
            const MKL_INT dn = da.n - abs(ddstr);
            const complex<double> *adata =
                dconja ? &da(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &da(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft) {
                    // (da x 1) x b
                    if (!ddconja)
                        zgemm(abconj & 2 ? "c" : "t", "n", &b.n, &dn, &k,
                              &scale, b.data, &ldb, adata, &ldda, &cfactor,
                              &c(max(dstrn, dstrm), (MKL_INT)0), &c.n);
                    else
                        for (MKL_INT i = 0; i < dn; i++)
                            zgemm(abconj & 2 ? "c" : "t", "c", &b.n, &k, &k,
                                  &scale, b.data, &ldb, adata + i * ldda, &k,
                                  &cfactor,
                                  &c(max(dstrn, dstrm) + i, (MKL_INT)0), &c.n);
                } else {
                    // a x (da x 1)
                    if (!(abconj & 1))
                        zgemm(ddconja ? "c" : "t", "n", &dn, &a.n, &k, &scale,
                              adata, &ldda, a.data, &lda, &cfactor,
                              &c(0, max(dstrn, dstrm)), &c.n);
                    else
                        for (MKL_INT i = 0; i < a.n; i++)
                            zgemm(ddconja ? "c" : "t", "c", &dn, &k, &k, &scale,
                                  adata, &ldda, a.data + i * lda, &k, &cfactor,
                                  &c(i, max(dstrn, dstrm)), &c.n);
                }
            }
        } else
            assert(false);
    }
    static void tensor_product(const ComplexMatrixRef &a, bool conja,
                               const ComplexMatrixRef &b, bool conjb,
                               const ComplexMatrixRef &c, complex<double> scale,
                               uint32_t stride) {
        const complex<double> cfactor = 1.0;
        switch ((uint8_t)conja | (conjb << 1)) {
        case 0:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const MKL_INT n = b.m * b.n;
                    zgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (MKL_INT k = 0; k < b.m; k++)
                        zgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const MKL_INT n = a.m * a.n;
                    zgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (MKL_INT k = 0; k < a.m; k++)
                        zgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++) {
                        const complex<double> factor = scale * a(i, j);
                        for (MKL_INT k = 0; k < b.m; k++)
                            zaxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 1:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const MKL_INT n = b.m * b.n;
                    zgemm("n", "c", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (MKL_INT k = 0; k < b.m; k++)
                        zgemm("n", "c", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                assert(a.m <= c.n);
                for (MKL_INT k = 0; k < a.n; k++)
                    zgemm("c", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const complex<double> factor = scale * conj(a(j, i));
                        for (MKL_INT k = 0; k < b.m; k++)
                            zaxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 2:
            if (a.m == 1 && a.n == 1) {
                assert(b.m <= c.n);
                for (MKL_INT k = 0; k < b.n; k++)
                    zgemm("c", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const MKL_INT n = a.m * a.n;
                    zgemm("n", "c", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (MKL_INT k = 0; k < a.m; k++)
                        zgemm("n", "c", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (MKL_INT i = 0, inca = 1, inc = b.m; i < b.n; i++)
                    for (MKL_INT j = 0; j < b.m; j++) {
                        const complex<double> factor = scale * conj(b(j, i));
                        for (MKL_INT k = 0; k < a.m; k++)
                            zaxpy(&a.n, &factor, &a(k, 0), &inca,
                                  &c(k * b.n + i, j + stride), &inc);
                    }
            }
            break;
        case 1 | 2:
            if (a.m == 1 && a.n == 1) {
                for (MKL_INT k = 0; k < b.n; k++)
                    zgemm("c", "c", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                for (MKL_INT k = 0; k < a.n; k++)
                    zgemm("c", "c", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (MKL_INT i = 0, incb = b.n, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const complex<double> factor = scale * conj(a(j, i));
                        for (MKL_INT k = 0; k < b.n; k++)
                            for (MKL_INT l = 0; l < b.m; l++)
                                c(i * b.n + k, j * b.m + l + stride) +=
                                    factor * conj(b(l, k));
                    }
            }
            break;
        default:
            assert(false);
        }
    }
    // SVD; original matrix will be destroyed
    static void svd(const ComplexMatrixRef &a, const ComplexMatrixRef &l,
                    const MatrixRef &s, const ComplexMatrixRef &r) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MKL_INT k = min(a.m, a.n), info = 0, lwork = 34 * max(a.m, a.n),
                lrwork = 5 * min(a.m, a.n);
        complex<double> *work = d_alloc->complex_allocate(lwork);
        double *rwork = d_alloc->allocate(lrwork);
        assert(a.m == l.m && a.n == r.n && l.n == k && r.m == k && s.n == k);
        zgesvd("S", "S", &a.n, &a.m, a.data, &a.n, s.data, r.data, &a.n, l.data,
               &k, work, &lwork, rwork, &info);
        assert(info == 0);
        d_alloc->deallocate(rwork, lrwork);
        d_alloc->complex_deallocate(work, lwork);
    }
    // SVD for parallelism over sites; PRB 87, 155137 (2013)
    static void accurate_svd(const ComplexMatrixRef &a,
                             const ComplexMatrixRef &l, const MatrixRef &s,
                             const ComplexMatrixRef &r, double eps = 1E-4) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        ComplexMatrixRef aa(nullptr, a.m, a.n);
        aa.data = d_alloc->complex_allocate(aa.size());
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
            ComplexMatrixRef xa(nullptr, k - p, k - p),
                xl(nullptr, k - p, k - p), xr(nullptr, k - p, k - p);
            xa.data = d_alloc->complex_allocate(xa.size());
            xl.data = d_alloc->complex_allocate(xl.size());
            xr.data = d_alloc->complex_allocate(xr.size());
            rotate(a, xa, ComplexMatrixRef(l.data + p, l.m, l.n), 3,
                   ComplexMatrixRef(r.data + p * r.n, r.m - p, r.n), 3, 1.0);
            accurate_svd(xa, xl, MatrixRef(s.data + p, 1, k - p), xr, eps);
            ComplexMatrixRef bl(nullptr, l.m, l.n), br(nullptr, r.m, r.n);
            bl.data = d_alloc->complex_allocate(bl.size());
            br.data = d_alloc->complex_allocate(br.size());
            copy(bl, l);
            copy(br, r);
            multiply(ComplexMatrixRef(bl.data + p, bl.m, bl.n), false, xl,
                     false, ComplexMatrixRef(l.data + p, l.m, l.n), 1.0, 0.0);
            multiply(xr, false,
                     ComplexMatrixRef(br.data + p * br.n, br.m - p, br.n),
                     false, ComplexMatrixRef(r.data + p * r.n, r.m - p, r.n),
                     1.0, 0.0);
            d_alloc->complex_deallocate(br.data, br.size());
            d_alloc->complex_deallocate(bl.data, bl.size());
            d_alloc->complex_deallocate(xr.data, xr.size());
            d_alloc->complex_deallocate(xl.data, xl.size());
            d_alloc->complex_deallocate(xa.data, xa.size());
        }
        d_alloc->complex_deallocate(aa.data, aa.size());
    }
    // LQ factorization
    static void lq(const ComplexMatrixRef &a, const ComplexMatrixRef &l,
                   const ComplexMatrixRef &q) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.m;
        complex<double> *work = d_alloc->complex_allocate(lwork);
        complex<double> *tau = d_alloc->complex_allocate(k);
        complex<double> *t = d_alloc->complex_allocate(a.m * a.n);
        assert(a.m == l.m && a.n == q.n && l.n == k && q.m == k);
        memcpy(t, a.data, sizeof(complex<double>) * a.m * a.n);
        zgeqrf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(l.data, 0, sizeof(complex<double>) * k * a.m);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(l.data + j * k, t + j * a.n,
                   sizeof(complex<double>) * min(j + 1, k));
        zungqr(&a.n, &k, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memcpy(q.data, t, sizeof(complex<double>) * k * a.n);
        d_alloc->complex_deallocate(t, a.m * a.n);
        d_alloc->complex_deallocate(tau, k);
        d_alloc->complex_deallocate(work, lwork);
    }
    // QR factorization
    static void qr(const ComplexMatrixRef &a, const ComplexMatrixRef &q,
                   const ComplexMatrixRef &r) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.n;
        complex<double> *work = d_alloc->complex_allocate(lwork);
        complex<double> *tau = d_alloc->complex_allocate(k);
        complex<double> *t = d_alloc->complex_allocate(a.m * a.n);
        assert(a.m == q.m && a.n == r.n && q.n == k && r.m == k);
        memcpy(t, a.data, sizeof(complex<double>) * a.m * a.n);
        zgelqf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(r.data, 0, sizeof(complex<double>) * k * a.n);
        for (MKL_INT j = 0; j < k; j++)
            memcpy(r.data + j * a.n + j, t + j * a.n + j,
                   sizeof(complex<double>) * (a.n - j));
        zunglq(&k, &a.m, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(q.data + j * k, t + j * a.n, sizeof(complex<double>) * k);
        d_alloc->complex_deallocate(t, a.m * a.n);
        d_alloc->complex_deallocate(tau, k);
        d_alloc->complex_deallocate(work, lwork);
    }
    // a += b.T
    static void transpose(const ComplexMatrixRef &a, const ComplexMatrixRef &b,
                          complex<double> scale = 1.0,
                          complex<double> cfactor = 1.0) {
        assert(a.m == b.n && a.n == b.m);
        const complex<double> one = 1.0;
        for (MKL_INT k = 0, inc = 1; k < b.n; k++)
            zgemm("t", "n", &b.m, &inc, &inc, &scale, &b(0, k), &b.n, &one,
                  &inc, &cfactor, &a(k, 0), &a.n);
    }
    // diagonalization for each symmetry block
    static void block_eigs(const ComplexMatrixRef &a, const DiagonalMatrix &w,
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
                complex<double> *work =
                    d_alloc->complex_allocate(mp[i].size() * mp[i].size());
                double *wwork = d_alloc->allocate(mp[i].size());
                for (size_t j = 0; j < mp[i].size(); j++)
                    for (size_t k = 0; k < mp[i].size(); k++)
                        work[j * mp[i].size() + k] = a(mp[i][j], mp[i][k]);
                eigs(ComplexMatrixRef(work, (MKL_INT)mp[i].size(),
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
                d_alloc->complex_deallocate(work, mp[i].size() * mp[i].size());
            }
    }
    // eigenvectors are row right vectors
    // U A^T = W U
    static void eigs(const ComplexMatrixRef &a, const DiagonalMatrix &w) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(a.m == a.n && w.n == a.n);
        const double scale = -1.0;
        MKL_INT lwork = 34 * a.n, n = a.m * a.n, incx = 2, info;
        complex<double> *work = d_alloc->complex_allocate(lwork);
        double *rwork = d_alloc->allocate(max((MKL_INT)1, 3 * a.n - 2));
        zheev("V", "U", &a.n, a.data, &a.n, w.data, work, &lwork, rwork, &info);
        assert((size_t)a.m * a.n == n);
        dscal(&n, &scale, (double *)a.data + 1, &incx);
        assert(info == 0);
        d_alloc->deallocate(rwork, max((MKL_INT)1, 3 * a.n - 2));
        d_alloc->complex_deallocate(work, lwork);
    }
    // z = r / aa
    static void cg_precondition(const ComplexMatrixRef &z,
                                const ComplexMatrixRef &r,
                                const ComplexDiagonalMatrix &aa) {
        copy(z, r);
        if (aa.size() != 0) {
            assert(aa.size() == r.size() && r.size() == z.size());
            for (MKL_INT i = 0; i < aa.n; i++)
                if (abs(aa.data[i]) > 1E-12)
                    z.data[i] /= aa.data[i];
        }
    }
    // Computes exp(t*H), the matrix exponential of a general complex
    // matrix in full, using the irreducible rational Pade approximation
    // Adapted from expokit fortran code zgpadm.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = 4 * m * m + ideg + 1
    // exp(tH) is located at work[ret:ret+m*m]
    static pair<MKL_INT, MKL_INT> expo_pade(MKL_INT ideg, MKL_INT m,
                                            const complex<double> *h,
                                            MKL_INT ldh, double t,
                                            complex<double> *work) {
        static const complex<double> zero = 0.0, one = 1.0, mone = -1.0;
        static const double dtwo = 2.0, dmone = -1.0;
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
        memset(work, 0, sizeof(complex<double>) * m);
        for (MKL_INT j = 0; j < m; j++)
            for (MKL_INT i = 0; i < m; i++)
                work[i] += abs(h[j * m + i]);
        double hnorm = 0.0;
        for (MKL_INT i = 0; i < m; i++)
            hnorm = max(hnorm, work[i].real());
        hnorm = abs(t * hnorm);
        if (hnorm == 0.0) {
            cerr << "Error - null H in expo pade" << endl;
            abort();
        }
        MKL_INT ns = max((MKL_INT)0, (MKL_INT)(log(hnorm) / log(2.0)) + 2);
        complex<double> scale = t / (double)(1LL << ns);
        complex<double> scale2 = scale * scale;
        // compute Pade coefficients
        MKL_INT i = ideg + 1, j = 2 * ideg + 1;
        work[icoef] = 1.0;
        for (MKL_INT k = 1; k <= ideg; k++)
            work[icoef + k] =
                work[icoef + k - 1] * (double)(i - k) / double(k * (j - k));
        // H2 = scale2*H*H ...
        zgemm("n", "n", &m, &m, &m, &scale2, h, &ldh, h, &ldh, &zero,
              work + ih2, &m);
        // initialize p (numerator) and q (denominator)
        memset(work + ip, 0, sizeof(complex<double>) * mm * 2);
        complex<double> cp = work[icoef + ideg - 1];
        complex<double> cq = work[icoef + ideg];
        for (MKL_INT j = 0; j < m; j++)
            work[ip + j * (m + 1)] = cp, work[iq + j * (m + 1)] = cq;
        // Apply Horner rule
        MKL_INT iodd = 1;
        for (MKL_INT k = ideg - 1; k > 0; k--) {
            MKL_INT iused = iodd * iq + (1 - iodd) * ip;
            zgemm("n", "n", &m, &m, &m, &one, work + iused, &m, work + ih2, &m,
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
        zgemm("n", "n", &m, &m, &m, &scale, work + *iqp, &m, h, &ldh, &zero,
              work + ifree, &m);
        *iqp = ifree;
        zaxpy(&mm, &mone, work + ip, &inc, work + iq, &inc);
        zgesv(&m, &m, work + iq, &m, (MKL_INT *)work + ih2, work + ip, &m,
              &iflag);
        if (iflag != 0) {
            cerr << "Problem in DGESV in expo pade" << endl;
            abort();
        }
        zdscal(&mm, &dtwo, work + ip, &inc);
        for (MKL_INT j = 0; j < m; j++)
            work[ip + j * (m + 1)] = work[ip + j * (m + 1)] + one;
        MKL_INT iput = ip;
        if (ns == 0 && iodd) {
            zdscal(&mm, &dmone, work + ip, &inc);
        } else {
            // squaring : exp(t*H) = (exp(t*H))^(2^ns)
            iodd = 1;
            for (MKL_INT k = 0; k < ns; k++) {
                MKL_INT iget = iodd * ip + (1 - iodd) * iq;
                iput = (1 - iodd) * ip + iodd * iq;
                zgemm("n", "n", &m, &m, &m, &one, work + iget, &m, work + iget,
                      &m, &zero, work + iput, &m);
                iodd = 1 - iodd;
            }
        }
        return make_pair(iput, ns);
    }
    // Computes w = exp(t*A)*v - for a (sparse) general matrix A.
    // Adapted from expokit fortran code zgexpv.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = n*(m+1)+n+(m+2)^2+4*(m+2)^2+ideg+1
    template <typename MatMul, typename PComm>
    static MKL_INT expo_krylov(MatMul &op, MKL_INT n, MKL_INT m, double t,
                               complex<double> *v, complex<double> *w,
                               double &tol, double anorm, complex<double> *work,
                               MKL_INT lwork, bool iprint,
                               const PComm &pcomm = nullptr) {
        const MKL_INT inc = 1;
        const double sqr1 = sqrt(0.1);
        const complex<double> zero = 0.0;
        const MKL_INT mxstep = 500, mxreject = 0, ideg = 6;
        const double delta = 1.2, gamma = 0.9;
        MKL_INT iflag = 0;
        // check restrictions on input parameters
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
        zcopy(&n, v, &inc, w, &inc);
        double beta = dznrm2(&n, w, &inc), vnorm = beta, hump = beta, avnorm;
        // obtain the very first stepsize
        double xm = 1.0 / (double)m, p1;
        p1 = tol * pow((m + 1) / 2.72, m + 1) * sqrt(2.0 * 3.14 * (m + 1));
        t_new = (1.0 / anorm) * pow(p1 / (4.0 * beta * anorm), xm);
        p1 = pow(10.0, round(log10(t_new) - sqr1) - 1);
        t_new = floor(t_new / p1 + 0.55) * p1;
        complex<double> hij;
        // step-by-step integration
        for (; t_now < t_out;) {
            nstep++;
            double t_step = min(t_out - t_now, t_new);
            p1 = 1.0 / beta;
            for (MKL_INT i = 0; i < n; i++)
                work[iv + i] = p1 * w[i];
            if (pcomm == nullptr || pcomm->root == pcomm->rank)
                memset(work + ih, 0, sizeof(complex<double>) * mh * mh);
            // Arnoldi loop
            MKL_INT j1v = iv + n;
            double hj1j = 0.0;
            for (MKL_INT j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    for (MKL_INT i = 0; i <= j; i++) {
                        hij = -complex_dot(
                            ComplexMatrixRef(work + iv + i * n, n, 1),
                            ComplexMatrixRef(work + j1v, n, 1));
                        zaxpy(&n, &hij, work + iv + i * n, &inc, work + j1v,
                              &inc);
                        work[ih + j * mh + i] = -hij;
                    }
                    hj1j = dznrm2(&n, work + j1v, &inc);
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
                    work[ih + j * mh + j + 1] = (complex<double>)hj1j;
                    hj1j = 1.0 / hj1j;
                    zdscal(&n, &hj1j, work + j1v, &inc);
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(work + j1v, n, pcomm->root);
                j1v += n;
            }
            if (k1 != 0) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank)
                    avnorm = dznrm2(&n, work + j1v, &inc);
            }
            MKL_INT ireject = 0;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                // set 1 for the 2-corrected scheme
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
                complex<double> hjj = (complex<double>)beta;
                zgemv("n", &n, &mx, &hjj, work + iv, &n, work + iexph, &inc,
                      &zero, w, &inc);
                beta = dznrm2(&n, w, &inc);
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
    // apply exponential of a real matrix to a vector
    // vr/vi: real/imag part of input/output vector
    template <typename MatMul, typename PComm>
    static int expo_apply(MatMul &op, complex<double> t, double anorm,
                          MatrixRef &vr, MatrixRef &vi, double consta = 0.0,
                          bool iprint = false, const PComm &pcomm = nullptr,
                          double conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        const MKL_INT vm = vr.m, vn = vr.n, n = vm * vn;
        assert(vi.m == vr.m && vi.n == vr.n);
        auto cop = [&op, vm, vn, n](const ComplexMatrixRef &a,
                                    const ComplexMatrixRef &b) -> void {
            vector<double> dar(n), dai(n), dbr(n, 0), dbi(n, 0);
            extract_complex(a, MatrixRef(dar.data(), vm, vn),
                            MatrixRef(dai.data(), vm, vn));
            op(MatrixRef(dar.data(), vm, vn), MatrixRef(dbr.data(), vm, vn));
            op(MatrixRef(dai.data(), vm, vn), MatrixRef(dbi.data(), vm, vn));
            fill_complex(b, MatrixRef(dbr.data(), vm, vn),
                         MatrixRef(dbi.data(), vm, vn));
        };
        vector<complex<double>> v(n);
        ComplexMatrixRef cv(v.data(), vm, vn);
        fill_complex(cv, vr, vi);
        MKL_INT nmult =
            expo_apply_complex_op(cop, t, anorm, cv, consta, iprint,
                                  (PComm)pcomm, conv_thrd, deflation_max_size);
        extract_complex(cv, vr, vi);
        return nmult;
    }
    // apply exponential of a matrix to a vector
    // vr/vi: real/imag part of input/output vector
    template <typename MatMul, typename PComm>
    static int expo_apply_complex_op(MatMul &op, complex<double> t,
                                     double anorm, ComplexMatrixRef &v,
                                     double consta = 0.0, bool iprint = false,
                                     const PComm &pcomm = nullptr,
                                     double conv_thrd = 5E-6,
                                     int deflation_max_size = 20) {
        MKL_INT vm = v.m, vn = v.n, n = vm * vn;
        double abst = abs(t);
        assert(abst != 0);
        complex<double> tt = t / abst;
        if (n < 4) {
            const MKL_INT lwork = 4 * n * n + 7;
            vector<complex<double>> h(n * n), work(lwork);
            vector<complex<double>> te(n), to(n);
            ComplexMatrixRef e = ComplexMatrixRef(te.data(), vm, vn);
            ComplexMatrixRef o = ComplexMatrixRef(to.data(), vm, vn);
            memset(e.data, 0, sizeof(complex<double>) * n);
            for (MKL_INT i = 0; i < n; i++) {
                e.data[i] = 1.0;
                memset(o.data, 0, sizeof(complex<double>) * n);
                op(e, o);
                for (MKL_INT j = 0; j < n; j++)
                    h[i * n + j] = tt * o.data[j];
                h[i * (n + 1)] += tt * consta;
                e.data[i] = 0.0;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                MKL_INT iptr =
                    expo_pade(6, n, h.data(), n, abst, work.data()).first;
                vector<complex<double>> w(n, 0);
                ComplexMatrixRef mvin = ComplexMatrixRef(v.data, v.m, v.n);
                ComplexMatrixRef mvout = ComplexMatrixRef(w.data(), v.m, v.n);
                multiply(ComplexMatrixRef(work.data() + iptr, n, n), true, mvin,
                         false, mvout, 1.0, 0.0);
                memcpy(v.data, w.data(), sizeof(complex<double>) * w.size());
            }
            if (pcomm != nullptr)
                pcomm->broadcast(v.data, n, pcomm->root);
            return n;
        }
        auto lop = [&op, consta, n, vm, vn, tt](complex<double> *a,
                                                complex<double> *b) -> void {
            static MKL_INT inc = 1;
            static complex<double> x = 1.0;
            op(ComplexMatrixRef(a, vm, vn), ComplexMatrixRef(b, vm, vn));
            const complex<double> cconsta = consta * tt;
            zgemm("n", "n", &inc, &n, &inc, &x, &cconsta, &inc, a, &inc, &tt, b,
                  &inc);
        };
        MKL_INT m = min((MKL_INT)deflation_max_size, n - 1);
        MKL_INT lwork = n * (m + 2) + 5 * (m + 2) * (m + 2) + 7;
        vector<complex<double>> w(n), work(lwork);
        anorm = (anorm + abs(consta) * n) * abs(tt);
        if (anorm < 1E-10)
            anorm = 1.0;
        MKL_INT nmult =
            expo_krylov(lop, n, m, abst, v.data, w.data(), conv_thrd, anorm,
                        work.data(), lwork, iprint, (PComm)pcomm);
        memcpy(v.data, w.data(), sizeof(complex<double>) * w.size());
        return (int)nmult;
    }
};

typedef GMatrixFunctions<complex<double>> ComplexMatrixFunctions;

} // namespace block2
