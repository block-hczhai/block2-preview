
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
// vector [sx] = float [sa] * vector [sx]
extern void FNAME(csscal)(const MKL_INT *n, const float *sa, complex<float> *sx,
                          const MKL_INT *incx) noexcept;

// vector [sx] = complex [sa] * vector [sx]
extern void FNAME(cscal)(const MKL_INT *n, const complex<float> *sa,
                         complex<float> *sx, const MKL_INT *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void FNAME(ccopy)(const MKL_INT *n, const complex<float> *dx,
                         const MKL_INT *incx, complex<float> *dy,
                         const MKL_INT *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + complex [sa] * vector [sx]
extern void FNAME(caxpy)(const MKL_INT *n, const complex<float> *sa,
                         const complex<float> *sx, const MKL_INT *incx,
                         complex<float> *sy, const MKL_INT *incy) noexcept;

// vector dot product
// extern void FNAME(cdotc)(complex<float> *pres, const MKL_INT *n,
//                   const complex<float> *zx, const MKL_INT *incx,
//                   const complex<float> *zy, const MKL_INT *incy) noexcept;

// Euclidean norm of a vector
extern float FNAME(scnrm2)(const MKL_INT *n, const complex<float> *x,
                           const MKL_INT *incx) noexcept;

// matrix multiplication
// mat [c] = complex [alpha] * mat [a] * mat [b] + complex [beta] * mat [c]
extern void FNAME(cgemm)(const char *transa, const char *transb,
                         const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                         const complex<float> *alpha, const complex<float> *a,
                         const MKL_INT *lda, const complex<float> *b,
                         const MKL_INT *ldb, const complex<float> *beta,
                         complex<float> *c, const MKL_INT *ldc) noexcept;

// LU factorization
extern void FNAME(cgetrf)(const MKL_INT *m, const MKL_INT *n, complex<float> *a,
                          const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info);

// matrix inverse
extern void FNAME(cgetri)(const MKL_INT *n, complex<float> *a,
                          const MKL_INT *lda, MKL_INT *ipiv,
                          complex<float> *work, const MKL_INT *lwork,
                          MKL_INT *info);

// eigenvalue problem
extern void FNAME(cgeev)(const char *jobvl, const char *jobvr, const MKL_INT *n,
                         complex<float> *a, const MKL_INT *lda,
                         complex<float> *w, complex<float> *vl,
                         const MKL_INT *ldvl, complex<float> *vr,
                         const MKL_INT *ldvr, complex<float> *work,
                         const MKL_INT *lwork, float *rwork, MKL_INT *info);

// matrix-vector multiplication
// vec [y] = complex [alpha] * mat [a] * vec [x] + complex [beta] * vec [y]
extern void FNAME(cgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const complex<float> *alpha, const complex<float> *a,
                         const MKL_INT *lda, const complex<float> *x,
                         const MKL_INT *incx, const complex<float> *beta,
                         complex<float> *y, const MKL_INT *incy) noexcept;

// linear system a * x = b
extern void FNAME(cgesv)(const MKL_INT *n, const MKL_INT *nrhs,
                         complex<float> *a, const MKL_INT *lda, MKL_INT *ipiv,
                         complex<float> *b, const MKL_INT *ldb, MKL_INT *info);

// least squares problem a * x = b
extern void FNAME(cgels)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const MKL_INT *nrhs, complex<float> *a,
                         const MKL_INT *lda, complex<float> *b,
                         const MKL_INT *ldb, complex<float> *work,
                         const MKL_INT *lwork, MKL_INT *info);

// matrix copy
// mat [b] = mat [a]
extern void FNAME(clacpy)(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                          const complex<float> *a, const MKL_INT *lda,
                          complex<float> *b, const MKL_INT *ldb);

// QR factorization
extern void FNAME(cgeqrf)(const MKL_INT *m, const MKL_INT *n, complex<float> *a,
                          const MKL_INT *lda, complex<float> *tau,
                          complex<float> *work, const MKL_INT *lwork,
                          MKL_INT *info);
extern void FNAME(cungqr)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          complex<float> *a, const MKL_INT *lda,
                          const complex<float> *tau, complex<float> *work,
                          const MKL_INT *lwork, MKL_INT *info);

// LQ factorization
extern void FNAME(cgelqf)(const MKL_INT *m, const MKL_INT *n, complex<float> *a,
                          const MKL_INT *lda, complex<float> *tau,
                          complex<float> *work, const MKL_INT *lwork,
                          MKL_INT *info);
extern void FNAME(cunglq)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          complex<float> *a, const MKL_INT *lda,
                          const complex<float> *tau, complex<float> *work,
                          const MKL_INT *lwork, MKL_INT *info);

// eigenvalue problem
extern void FNAME(cheev)(const char *jobz, const char *uplo, const MKL_INT *n,
                         complex<float> *a, const MKL_INT *lda, float *w,
                         complex<float> *work, const MKL_INT *lwork,
                         float *rwork, MKL_INT *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void FNAME(cgesvd)(const char *jobu, const char *jobvt, const MKL_INT *m,
                          const MKL_INT *n, complex<float> *a,
                          const MKL_INT *lda, float *s, complex<float> *u,
                          const MKL_INT *ldu, complex<float> *vt,
                          const MKL_INT *ldvt, complex<float> *work,
                          const MKL_INT *lwork, float *rwork, MKL_INT *info);

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void FNAME(zdscal)(const MKL_INT *n, const double *sa,
                          complex<double> *sx, const MKL_INT *incx) noexcept;

// vector [sx] = complex [sa] * vector [sx]
extern void FNAME(zscal)(const MKL_INT *n, const complex<double> *sa,
                         complex<double> *sx, const MKL_INT *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void FNAME(zcopy)(const MKL_INT *n, const complex<double> *dx,
                         const MKL_INT *incx, complex<double> *dy,
                         const MKL_INT *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + complex [sa] * vector [sx]
extern void FNAME(zaxpy)(const MKL_INT *n, const complex<double> *sa,
                         const complex<double> *sx, const MKL_INT *incx,
                         complex<double> *sy, const MKL_INT *incy) noexcept;

// vector dot product
// extern void FNAME(zdotc)(complex<double> *pres, const MKL_INT *n,
//                   const complex<double> *zx, const MKL_INT *incx,
//                   const complex<double> *zy, const MKL_INT *incy) noexcept;

// Euclidean norm of a vector
extern double FNAME(dznrm2)(const MKL_INT *n, const complex<double> *x,
                            const MKL_INT *incx) noexcept;

// matrix multiplication
// mat [c] = complex [alpha] * mat [a] * mat [b] + complex [beta] * mat [c]
extern void FNAME(zgemm)(const char *transa, const char *transb,
                         const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                         const complex<double> *alpha, const complex<double> *a,
                         const MKL_INT *lda, const complex<double> *b,
                         const MKL_INT *ldb, const complex<double> *beta,
                         complex<double> *c, const MKL_INT *ldc) noexcept;

// LU factorization
extern void FNAME(zgetrf)(const MKL_INT *m, const MKL_INT *n,
                          complex<double> *a, const MKL_INT *lda, MKL_INT *ipiv,
                          MKL_INT *info);

// matrix inverse
extern void FNAME(zgetri)(const MKL_INT *n, complex<double> *a,
                          const MKL_INT *lda, MKL_INT *ipiv,
                          complex<double> *work, const MKL_INT *lwork,
                          MKL_INT *info);

// eigenvalue problem
extern void FNAME(zgeev)(const char *jobvl, const char *jobvr, const MKL_INT *n,
                         complex<double> *a, const MKL_INT *lda,
                         complex<double> *w, complex<double> *vl,
                         const MKL_INT *ldvl, complex<double> *vr,
                         const MKL_INT *ldvr, complex<double> *work,
                         const MKL_INT *lwork, double *rwork, MKL_INT *info);

// matrix-vector multiplication
// vec [y] = complex [alpha] * mat [a] * vec [x] + complex [beta] * vec [y]
extern void FNAME(zgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const complex<double> *alpha, const complex<double> *a,
                         const MKL_INT *lda, const complex<double> *x,
                         const MKL_INT *incx, const complex<double> *beta,
                         complex<double> *y, const MKL_INT *incy) noexcept;

// linear system a * x = b
extern void FNAME(zgesv)(const MKL_INT *n, const MKL_INT *nrhs,
                         complex<double> *a, const MKL_INT *lda, MKL_INT *ipiv,
                         complex<double> *b, const MKL_INT *ldb, MKL_INT *info);

// least squares problem a * x = b
extern void FNAME(zgels)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const MKL_INT *nrhs, complex<double> *a,
                         const MKL_INT *lda, complex<double> *b,
                         const MKL_INT *ldb, complex<double> *work,
                         const MKL_INT *lwork, MKL_INT *info);

// matrix copy
// mat [b] = mat [a]
extern void FNAME(zlacpy)(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                          const complex<double> *a, const MKL_INT *lda,
                          complex<double> *b, const MKL_INT *ldb);

// QR factorization
extern void FNAME(zgeqrf)(const MKL_INT *m, const MKL_INT *n,
                          complex<double> *a, const MKL_INT *lda,
                          complex<double> *tau, complex<double> *work,
                          const MKL_INT *lwork, MKL_INT *info);
extern void FNAME(zungqr)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          complex<double> *a, const MKL_INT *lda,
                          const complex<double> *tau, complex<double> *work,
                          const MKL_INT *lwork, MKL_INT *info);

// LQ factorization
extern void FNAME(zgelqf)(const MKL_INT *m, const MKL_INT *n,
                          complex<double> *a, const MKL_INT *lda,
                          complex<double> *tau, complex<double> *work,
                          const MKL_INT *lwork, MKL_INT *info);
extern void FNAME(zunglq)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          complex<double> *a, const MKL_INT *lda,
                          const complex<double> *tau, complex<double> *work,
                          const MKL_INT *lwork, MKL_INT *info);

// eigenvalue problem
extern void FNAME(zheev)(const char *jobz, const char *uplo, const MKL_INT *n,
                         complex<double> *a, const MKL_INT *lda, double *w,
                         complex<double> *work, const MKL_INT *lwork,
                         double *rwork, MKL_INT *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void FNAME(zgesvd)(const char *jobu, const char *jobvt, const MKL_INT *m,
                          const MKL_INT *n, complex<double> *a,
                          const MKL_INT *lda, double *s, complex<double> *u,
                          const MKL_INT *ldu, complex<double> *vt,
                          const MKL_INT *ldvt, complex<double> *work,
                          const MKL_INT *lwork, double *rwork, MKL_INT *info);

#endif
}

template <>
inline void xgemm<complex<float>>(const char *transa, const char *transb,
                                  const MKL_INT *m, const MKL_INT *n,
                                  const MKL_INT *k, const complex<float> *alpha,
                                  const complex<float> *a, const MKL_INT *lda,
                                  const complex<float> *b, const MKL_INT *ldb,
                                  const complex<float> *beta, complex<float> *c,
                                  const MKL_INT *ldc) noexcept {
    return FNAME(cgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}
template <>
inline void
xgemm<complex<double>>(const char *transa, const char *transb, const MKL_INT *m,
                       const MKL_INT *n, const MKL_INT *k,
                       const complex<double> *alpha, const complex<double> *a,
                       const MKL_INT *lda, const complex<double> *b,
                       const MKL_INT *ldb, const complex<double> *beta,
                       complex<double> *c, const MKL_INT *ldc) noexcept {
    return FNAME(zgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <>
inline void xscal<complex<float>>(const MKL_INT *n, const complex<float> *sa,
                                  complex<float> *sx,
                                  const MKL_INT *incx) noexcept {
    FNAME(cscal)(n, sa, sx, incx);
}
template <>
inline void xscal<complex<double>>(const MKL_INT *n, const complex<double> *sa,
                                   complex<double> *sx,
                                   const MKL_INT *incx) noexcept {
    FNAME(zscal)(n, sa, sx, incx);
}

template <>
inline void xdscal<complex<float>>(const MKL_INT *n, const float *sa,
                                   complex<float> *sx,
                                   const MKL_INT *incx) noexcept {
    FNAME(csscal)(n, sa, sx, incx);
}
template <>
inline void xdscal<complex<double>>(const MKL_INT *n, const double *sa,
                                    complex<double> *sx,
                                    const MKL_INT *incx) noexcept {
    FNAME(zdscal)(n, sa, sx, incx);
}

template <>
inline float xnrm2<complex<float>>(const MKL_INT *n, const complex<float> *x,
                                   const MKL_INT *incx) noexcept {
    return FNAME(scnrm2)(n, x, incx);
}
template <>
inline double xnrm2<complex<double>>(const MKL_INT *n, const complex<double> *x,
                                     const MKL_INT *incx) noexcept {
    return FNAME(dznrm2)(n, x, incx);
}

template <>
inline void xcopy<complex<float>>(const MKL_INT *n, const complex<float> *dx,
                                  const MKL_INT *incx, complex<float> *dy,
                                  const MKL_INT *incy) noexcept {
    FNAME(ccopy)(n, dx, incx, dy, incy);
}
template <>
inline void xcopy<complex<double>>(const MKL_INT *n, const complex<double> *dx,
                                   const MKL_INT *incx, complex<double> *dy,
                                   const MKL_INT *incy) noexcept {
    FNAME(zcopy)(n, dx, incx, dy, incy);
}

template <>
inline complex<float>
xdot<complex<float>>(const MKL_INT *n, const complex<float> *dx,
                     const MKL_INT *incx, const complex<float> *dy,
                     const MKL_INT *incy) noexcept {
    static const complex<float> x = 1.0, zz = 0.0;
    MKL_INT inc = 1;
    complex<float> r;
    FNAME(cgemm)
    ("n", "t", &inc, &inc, n, &x, dy, incy, dx, incx, &zz, &r, &inc);
    return r;
}
template <>
inline complex<double>
xdot<complex<double>>(const MKL_INT *n, const complex<double> *dx,
                      const MKL_INT *incx, const complex<double> *dy,
                      const MKL_INT *incy) noexcept {
    static const complex<double> x = 1.0, zz = 0.0;
    MKL_INT inc = 1;
    complex<double> r;
    FNAME(zgemm)
    ("n", "t", &inc, &inc, n, &x, dy, incy, dx, incx, &zz, &r, &inc);
    return r;
}

template <>
inline void xaxpy<complex<float>>(const MKL_INT *n, const complex<float> *sa,
                                  const complex<float> *sx, const MKL_INT *incx,
                                  complex<float> *sy,
                                  const MKL_INT *incy) noexcept {
    FNAME(caxpy)(n, sa, sx, incx, sy, incy);
}
template <>
inline void xaxpy<complex<double>>(const MKL_INT *n, const complex<double> *sa,
                                   const complex<double> *sx,
                                   const MKL_INT *incx, complex<double> *sy,
                                   const MKL_INT *incy) noexcept {
    FNAME(zaxpy)(n, sa, sx, incx, sy, incy);
}

template <>
inline void xlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const complex<float> *a, const MKL_INT *lda,
                   complex<float> *b, const MKL_INT *ldb) {
    FNAME(clacpy)(uplo, m, n, a, lda, b, ldb);
}
template <>
inline void xlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const complex<double> *a, const MKL_INT *lda,
                   complex<double> *b, const MKL_INT *ldb) {
    FNAME(zlacpy)(uplo, m, n, a, lda, b, ldb);
}

template <>
inline void xgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const complex<float> *alpha, const complex<float> *a,
                  const MKL_INT *lda, const complex<float> *x,
                  const MKL_INT *incx, const complex<float> *beta,
                  complex<float> *y, const MKL_INT *incy) {
    FNAME(cgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
template <>
inline void xgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const complex<double> *alpha, const complex<double> *a,
                  const MKL_INT *lda, const complex<double> *x,
                  const MKL_INT *incx, const complex<double> *beta,
                  complex<double> *y, const MKL_INT *incy) {
    FNAME(zgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void xgeqrf(const MKL_INT *m, const MKL_INT *n, complex<float> *a,
                   const MKL_INT *lda, complex<float> *tau,
                   complex<float> *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(cgeqrf)(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgeqrf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, complex<double> *tau,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(zgeqrf)(m, n, a, lda, tau, work, lwork, info);
}

template <>
inline void xgetrf(const MKL_INT *m, const MKL_INT *n, complex<float> *a,
                   const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info) {
    FNAME(cgetrf)(m, n, a, lda, ipiv, info);
}

template <>
inline void xgetrf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info) {
    FNAME(zgetrf)(m, n, a, lda, ipiv, info);
}

template <>
inline void xgetri(const MKL_INT *m, complex<float> *a, const MKL_INT *lda,
                   MKL_INT *ipiv, complex<float> *work, const MKL_INT *lwork,
                   MKL_INT *info) {
    FNAME(cgetri)(m, a, lda, ipiv, work, lwork, info);
}

template <>
inline void xgetri(const MKL_INT *m, complex<double> *a, const MKL_INT *lda,
                   MKL_INT *ipiv, complex<double> *work, const MKL_INT *lwork,
                   MKL_INT *info) {
    FNAME(zgetri)(m, a, lda, ipiv, work, lwork, info);
}

template <>
inline void xungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<float> *a, const MKL_INT *lda,
                   const complex<float> *tau, complex<float> *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    FNAME(cungqr)(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<double> *a, const MKL_INT *lda,
                   const complex<double> *tau, complex<double> *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    FNAME(zungqr)(m, n, k, a, lda, tau, work, lwork, info);
}

template <>
inline void xgelqf(const MKL_INT *m, const MKL_INT *n, complex<float> *a,
                   const MKL_INT *lda, complex<float> *tau,
                   complex<float> *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(cgelqf)(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgelqf(const MKL_INT *m, const MKL_INT *n, complex<double> *a,
                   const MKL_INT *lda, complex<double> *tau,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(zgelqf)(m, n, a, lda, tau, work, lwork, info);
}

template <>
inline void xgels(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const MKL_INT *nrhs, complex<float> *a, const MKL_INT *lda,
                  complex<float> *b, const MKL_INT *ldb, complex<float> *work,
                  const MKL_INT *lwork, MKL_INT *info) {
    FNAME(cgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

template <>
inline void xgels(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const MKL_INT *nrhs, complex<double> *a, const MKL_INT *lda,
                  complex<double> *b, const MKL_INT *ldb, complex<double> *work,
                  const MKL_INT *lwork, MKL_INT *info) {
    FNAME(zgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

template <>
inline void xunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<float> *a, const MKL_INT *lda,
                   const complex<float> *tau, complex<float> *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    FNAME(cunglq)(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   complex<double> *a, const MKL_INT *lda,
                   const complex<double> *tau, complex<double> *work,
                   const MKL_INT *lwork, MKL_INT *info) {
    FNAME(zunglq)(m, n, k, a, lda, tau, work, lwork, info);
}

template <>
inline void xgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, complex<float> *a, const MKL_INT *lda,
                   float *s, complex<float> *u, const MKL_INT *ldu,
                   complex<float> *vt, const MKL_INT *ldvt,
                   complex<float> *work, const MKL_INT *lwork, MKL_INT *info) {
    vector<float> rwork;
    rwork.reserve(5 * min(*m, *n));
    FNAME(cgesvd)
    (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork.data(),
     info);
}
template <>
inline void xgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, complex<double> *a, const MKL_INT *lda,
                   double *s, complex<double> *u, const MKL_INT *ldu,
                   complex<double> *vt, const MKL_INT *ldvt,
                   complex<double> *work, const MKL_INT *lwork, MKL_INT *info) {
    vector<double> rwork;
    rwork.reserve(5 * min(*m, *n));
    FNAME(zgesvd)
    (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork.data(),
     info);
}

template <>
inline void xgesv(const MKL_INT *n, const MKL_INT *nrhs, complex<float> *a,
                  const MKL_INT *lda, MKL_INT *ipiv, complex<float> *b,
                  const MKL_INT *ldb, MKL_INT *info) {
    FNAME(cgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
}

template <>
inline void xgesv(const MKL_INT *n, const MKL_INT *nrhs, complex<double> *a,
                  const MKL_INT *lda, MKL_INT *ipiv, complex<double> *b,
                  const MKL_INT *ldb, MKL_INT *info) {
    FNAME(zgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
}

template <>
inline void xheev(const char *jobz, const char *uplo, const MKL_INT *n,
                  complex<float> *a, const MKL_INT *lda, float *w,
                  complex<float> *work, const MKL_INT *lwork, float *rwork,
                  MKL_INT *info) {
    FNAME(cheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}
template <>
inline void xheev(const char *jobz, const char *uplo, const MKL_INT *n,
                  complex<double> *a, const MKL_INT *lda, double *w,
                  complex<double> *work, const MKL_INT *lwork, double *rwork,
                  MKL_INT *info) {
    FNAME(zheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}

template <>
inline void xgeev(const char *jobvl, const char *jobvr, const MKL_INT *n,
                  complex<float> *a, const MKL_INT *lda, complex<float> *w,
                  complex<float> *vl, const MKL_INT *ldvl, complex<float> *vr,
                  const MKL_INT *ldvr, complex<float> *work,
                  const MKL_INT *lwork, float *rwork, MKL_INT *info) {
    FNAME(cgeev)
    (jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}
template <>
inline void xgeev(const char *jobvl, const char *jobvr, const MKL_INT *n,
                  complex<double> *a, const MKL_INT *lda, complex<double> *w,
                  complex<double> *vl, const MKL_INT *ldvl, complex<double> *vr,
                  const MKL_INT *ldvr, complex<double> *work,
                  const MKL_INT *lwork, double *rwork, MKL_INT *info) {
    FNAME(zgeev)
    (jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

// General matrix operations
template <typename FL, typename> struct GMatrixFunctions;

// Dense complex number matrix operations
template <typename FL>
struct GMatrixFunctions<FL, typename enable_if<is_complex<FL>::value>::type> {
    typedef typename GMatrix<FL>::FP FP;
    // a = re + im i
    static void fill_complex(const GMatrix<FL> &a, const GMatrix<FP> &re,
                             const GMatrix<FP> &im) {
        if (re.data != nullptr)
            GMatrixFunctions<FP>::copy(GMatrix<FP>((FP *)a.data, a.m, a.n), re,
                                       2, 1);
        if (im.data != nullptr)
            GMatrixFunctions<FP>::copy(GMatrix<FP>((FP *)a.data + 1, a.m, a.n),
                                       im, 2, 1);
    }
    // re + im i = a
    static void extract_complex(const GMatrix<FL> &a, const GMatrix<FP> &re,
                                const GMatrix<FP> &im) {
        if (re.data != nullptr)
            GMatrixFunctions<FP>::copy(re, GMatrix<FP>((FP *)a.data, a.m, a.n),
                                       1, 2);
        if (im.data != nullptr)
            GMatrixFunctions<FP>::copy(
                im, GMatrix<FP>((FP *)a.data + 1, a.m, a.n), 1, 2);
    }
    // a = b
    static void copy(const GMatrix<FL> &a, const GMatrix<FL> &b,
                     const MKL_INT inca = 1, const MKL_INT incb = 1) {
        assert(a.m == b.m && a.n == b.n);
        const MKL_INT n = a.m * a.n;
        xcopy<FL>(&n, b.data, &incb, a.data, &inca);
    }
    static void iscale(const GMatrix<FL> &a, FL scale, const MKL_INT inc = 1) {
        MKL_INT n = a.m * a.n;
        xscal<FL>(&n, &scale, a.data, &inc);
    }
    static void keep_real(const GMatrix<FL> &a) {
        const MKL_INT incx = 2;
        const FP scale = 0.0;
        MKL_INT n = a.m * a.n;
        xscal<FP>(&n, &scale, (FP *)a.data + 1, &incx);
    }
    static void conjugate(const GMatrix<FL> &a) {
        const MKL_INT incx = 2;
        const FP scale = -1.0;
        MKL_INT n = a.m * a.n;
        xscal<FP>(&n, &scale, (FP *)a.data + 1, &incx);
    }
    // a = a + scale * op(b)
    // conj means conj trans
    static void iadd(const GMatrix<FL> &a, const GMatrix<FL> &b, FL scale,
                     bool conj = false, FL cfactor = 1.0) {
        static const FL x = 1.0;
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            MKL_INT n = a.m * a.n, inc = 1;
            if (cfactor == (FP)1.0)
                xaxpy<FL>(&n, &scale, b.data, &inc, a.data, &inc);
            else
                xgemm<FL>("n", "n", &inc, &n, &inc, &scale, &x, &inc, b.data,
                          &inc, &cfactor, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n == b.m);
            const FL one = 1.0;
            for (MKL_INT k = 0, inc = 1; k < b.n; k++)
                xgemm<FL>("c", "n", &b.m, &inc, &inc, &scale, &b(0, k), &b.n,
                          &one, &inc, &cfactor, &a(k, 0), &a.n);
        }
    }
    static FP norm(const GMatrix<FL> &a) {
        MKL_INT n = a.m * a.n, inc = 1;
        return xnrm2<FL>(&n, a.data, &inc);
    }
    // dot product (a ^ H, b)
    static FL complex_dot(const GMatrix<FL> &a, const GMatrix<FL> &b) {
        static const FL x = 1.0, zz = 0.0;
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        FL r;
        // zdotc can sometimes return zero
        // zdotc(&r, &n, a.data, &inc, b.data, &inc);
        xgemm<FL>("c", "n", &inc, &inc, &n, &x, a.data, &n, b.data, &n, &zz, &r,
                  &inc);
        return r;
    }
    // Computes norm more accurately
    static FP norm_accurate(const GMatrix<FL> &a) {
        MKL_INT n = a.m * a.n;
        // do re and im separately, as in numpy
        typename GMatrix<FP>::FL out_real = 0.0;
        typename GMatrix<FP>::FL out_imag = 0.0;
        typename GMatrix<FP>::FL compensate_real = 0.0;
        typename GMatrix<FP>::FL compensate_imag = 0.0;
        for (MKL_INT ii = 0; ii < n; ++ii) {
            typename GMatrix<FP>::FL &&xre =
                (typename GMatrix<FP>::FL)real(a.data[ii]);
            typename GMatrix<FP>::FL &&xim =
                (typename GMatrix<FP>::FL)imag(a.data[ii]);
            typename GMatrix<FP>::FL sumi_real = xre * xre;
            typename GMatrix<FP>::FL sumi_imag = xim * xim;
            // Kahan summation
            auto y_real = sumi_real - compensate_real;
            auto y_imag = sumi_imag - compensate_imag;
            const volatile typename GMatrix<FP>::FL t_real = out_real + y_real;
            const volatile typename GMatrix<FP>::FL t_imag = out_imag + y_imag;
            const volatile typename GMatrix<FP>::FL z_real = t_real - out_real;
            const volatile typename GMatrix<FP>::FL z_imag = t_imag - out_imag;
            compensate_real = z_real - y_real;
            compensate_imag = z_imag - y_imag;
            out_real = t_real;
            out_imag = t_imag;
        }
        typename GMatrix<FP>::FL out = sqrt(out_real + out_imag);
        return static_cast<FP>(out);
    }
    template <typename T1, typename T2>
    static bool all_close(const T1 &a, const T2 &b, FP atol = 1E-8,
                          FP rtol = 1E-5, FL scale = 1.0) {
        assert(a.m == b.m && a.n == b.n);
        for (MKL_INT i = 0; i < a.m; i++)
            for (MKL_INT j = 0; j < a.n; j++)
                if (abs((FL)a(i, j) - scale * (FL)b(i, j)) >
                    atol + rtol * abs((FL)b(i, j)))
                    return false;
        return true;
    }
    // dot product (a ^ T, b)
    static FL dot(const GMatrix<FL> &a, const GMatrix<FL> &b) {
        static const FL x = 1.0, zz = 0.0;
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        FL r;
        xgemm<FL>("t", "n", &inc, &inc, &n, &x, a.data, &n, b.data, &n, &zz, &r,
                  &inc);
        return r;
    }
    // matrix inverse
    static void inverse(const GMatrix<FL> &a) {
        assert(a.m == a.n);
        vector<MKL_INT> ipiv;
        vector<FL> work;
        ipiv.reserve(a.m);
        MKL_INT lwork = 34 * a.n, info = -1;
        work.reserve(lwork);
        xgetrf<FL>(&a.m, &a.n, a.data, &a.m, ipiv.data(), &info);
        assert(info == 0);
        xgetri<FL>(&a.n, a.data, &a.m, ipiv.data(), work.data(), &lwork, &info);
        assert(info == 0);
    }
    // least squares problem a x = b
    // return the residual (norm, not squared)
    // a.n is used as lda
    static FP least_squares(const GMatrix<FL> &a, const GMatrix<FL> &b,
                            const GMatrix<FL> &x) {
        assert(a.m == b.m && a.n >= x.m && b.n == 1 && x.n == 1);
        vector<FL> work, atr, xtr;
        MKL_INT lwork = 34 * min(a.m, x.m), info = -1, nrhs = 1,
                mn = max(a.m, x.m), nr = a.m - x.m;
        work.reserve(lwork);
        atr.reserve(a.size());
        xtr.reserve(mn);
        xcopy<FL>(&a.m, b.data, &nrhs, xtr.data(), &nrhs);
        for (MKL_INT i = 0; i < x.m; i++)
            xcopy<FL>(&a.m, a.data + i, &a.n, atr.data() + i * a.m, &nrhs);
        xgels<FL>("N", &a.m, &x.m, &nrhs, atr.data(), &a.m, xtr.data(), &mn,
                  work.data(), &lwork, &info);
        assert(info == 0);
        xcopy<FL>(&x.m, xtr.data(), &nrhs, x.data, &nrhs);
        return nr > 0 ? xnrm2<FL>(&nr, xtr.data() + x.m, &nrhs) : 0;
    }
    // eigenvalue for non-hermitian matrix, A is overwritten
    // row right-vectors: A u(j) = lambda(j) u(j) (stored in alpha)
    // row left-vectors: u(j)**H A = lambda(j) u(j)**H (optional)
    static void eig(const GMatrix<FL> &a, const GDiagonalMatrix<FL> &w,
                    const GMatrix<FL> &lv) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(a.m == a.n && w.n == a.n);
        MKL_INT lwork = -1, info;
        FL twork;
        FP *rwork = d_alloc->allocate(a.m * 2);
        xgeev<FL>("V", lv.data == 0 ? "N" : "V", &a.n, a.data, &a.n, w.data,
                  nullptr, &a.n, lv.data, &a.n, &twork, &lwork, rwork, &info);
        assert(info == 0);
        lwork = (MKL_INT)xreal<FL>(twork);
        FL *work = d_alloc->complex_allocate(lwork);
        FL *vr = d_alloc->complex_allocate(a.m * a.n);
        xgeev<FL>("V", lv.data == 0 ? "N" : "V", &a.n, a.data, &a.n, w.data, vr,
                  &a.n, lv.data, &a.n, work, &lwork, rwork, &info);
        assert(info == 0);
        for (size_t k = 0; k < a.m * a.n; k++)
            a.data[k] = conj(vr[k]);
        if (lv.data != 0)
            conjugate(lv);
        d_alloc->complex_deallocate(vr, a.m * a.n);
        d_alloc->complex_deallocate(work, lwork);
        d_alloc->deallocate(rwork, a.m * 2);
    }
    static void eig(const GMatrix<FL> &a, const GDiagonalMatrix<FP> &wr,
                    const GDiagonalMatrix<FP> &wi, const GMatrix<FL> &lv) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GDiagonalMatrix<FL> wx(nullptr, wr.m);
        wx.allocate();
        eig(a, wx, lv);
        extract_complex(GMatrix<FL>(wx.data, wx.m, 1),
                        GMatrix<FP>(wr.data, wr.m, 1),
                        GMatrix<FP>(wi.data, wi.m, 1));
        wx.deallocate();
    }
    // matrix logarithm using diagonalization
    static void logarithm(const GMatrix<FL> &a) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(a.m == a.n);
        GDiagonalMatrix<FL> w(nullptr, a.m);
        w.data = (FL *)d_alloc->allocate(a.m * 2);
        GMatrix<FL> wa(nullptr, a.m, a.n);
        wa.data = (FL *)d_alloc->allocate(a.m * a.n * 2);
        GMatrix<FL> ua(nullptr, a.m, a.n);
        ua.data = (FL *)d_alloc->allocate(a.m * a.n * 2);
        memcpy(ua.data, a.data, sizeof(FL) * a.size());
        eig(ua, w, GMatrix<FL>(nullptr, a.m, a.n));
        for (MKL_INT i = 0; i < a.m; i++)
            for (MKL_INT j = 0; j < a.n; j++)
                wa(i, j) = ua(i, j) * log(w(i, i));
        inverse(ua);
        multiply(wa, true, ua, true, a, 1.0, 0.0);
        d_alloc->deallocate((FP *)ua.data, a.m * a.n * 2);
        d_alloc->deallocate((FP *)wa.data, a.m * a.n * 2);
        d_alloc->deallocate((FP *)w.data, a.m * 2);
    }
    // solve a^T x[i, :] = b[i, :] => output in b; a will be overwritten
    static void linear(const GMatrix<FL> &a, const GMatrix<FL> &b) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        assert(a.m == a.n && a.m == b.n);
        MKL_INT *work = (MKL_INT *)i_alloc->allocate(a.n * _MINTSZ), info = -1;
        xgesv<FL>(&a.m, &b.m, a.data, &a.n, work, b.data, &a.n, &info);
        assert(info == 0);
        i_alloc->deallocate(work, a.n * _MINTSZ);
    }
    // c.n is used for ldc; a.n is used for lda
    // conj can be 0 (no conj no trans), 1 (trans), 3 (conj trans)
    static void multiply(const GMatrix<FL> &a, uint8_t conja,
                         const GMatrix<FL> &b, uint8_t conjb,
                         const GMatrix<FL> &c, FL scale, FL cfactor) {
        static const char ntxc[5] = "ntxc";
        // if assertion failes here, check whether it is the case
        // where different bra and ket are used with the transpose rule
        // use no-transpose-rule to fix it
        if (!conja && !conjb) {
            assert(a.n >= b.m && c.m == a.m && c.n >= b.n);
            xgemm<FL>("n", "n", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                      &a.n, &cfactor, c.data, &c.n);
        } else if (!conja && conjb != 2) {
            assert(a.n >= b.n && c.m == a.m && c.n >= b.m);
            xgemm<FL>(ntxc + conjb, "n", &b.m, &c.m, &b.n, &scale, b.data, &b.n,
                      a.data, &a.n, &cfactor, c.data, &c.n);
        } else if (conja != 2 && !conjb) {
            assert(a.m == b.m && c.m <= a.n && c.n >= b.n);
            xgemm<FL>("n", ntxc + conja, &b.n, &c.m, &b.m, &scale, b.data, &b.n,
                      a.data, &a.n, &cfactor, c.data, &c.n);
        } else if (conja != 2 && conjb != 2) {
            assert(a.m == b.n && c.m <= a.n && c.n >= b.m);
            xgemm<FL>(ntxc + conjb, ntxc + conja, &b.m, &c.m, &b.n, &scale,
                      b.data, &b.n, a.data, &a.n, &cfactor, c.data, &c.n);
        } else if (conja == 2 && conjb != 2) {
            const MKL_INT one = 1;
            for (MKL_INT k = 0; k < c.m; k++)
                xgemm<FL>(ntxc + conjb, "c", (conjb & 1) ? &b.m : &b.n, &one,
                          (conjb & 1) ? &b.n : &b.m, &scale, b.data, &b.n,
                          &a(k, 0), &one, &cfactor, &c(k, 0), &c.n);
        } else if (conja != 3 && conjb == 2) {
            const MKL_INT one = 1;
            for (MKL_INT k = 0; k < c.m; k++)
                xgemm<FL>(ntxc + (conja ^ 1), "c", &one, &b.n, &b.m, &scale,
                          (conja & 1) ? &a(0, k) : &a(k, 0), &a.n, b.data, &b.n,
                          &cfactor, &c(k, 0), &one);
        } else
            assert(false);
    }
    // c = bra(.T) * a * ket(.T)
    // return nflop
    // conj can be 0 (no conj no trans), 1 (trans), 2 (conj), 3 (conj trans)
    static size_t rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                         const GMatrix<FL> &bra, uint8_t conj_bra,
                         const GMatrix<FL> &ket, uint8_t conj_ket, FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        if (conj_bra != 2 && conj_ket != 2) {
            GMatrix<FL> work(nullptr, a.m, (conj_ket & 1) ? ket.m : ket.n);
            work.allocate(d_alloc);
            multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
            multiply(bra, conj_bra, work, false, c, scale, 1.0);
            work.deallocate(d_alloc);
            return (size_t)ket.m * ket.n * work.m +
                   (size_t)work.m * work.n * c.m;
        } else if (conj_bra != 2) {
            GMatrix<FL> work(nullptr, ket.n, a.m);
            work.allocate(d_alloc);
            multiply(ket, 3, a, true, work, 1.0, 0.0);
            multiply(bra, conj_bra, work, true, c, scale, 1.0);
            work.deallocate(d_alloc);
            return (size_t)ket.m * ket.n * work.n +
                   (size_t)work.m * work.n * c.m;
        } else if (conj_ket != 2) {
            GMatrix<FL> work(nullptr, a.n, bra.m);
            work.allocate(d_alloc);
            multiply(a, true, bra, 3, work, 1.0, 0.0);
            multiply(work, true, ket, conj_ket, c, scale, 1.0);
            work.deallocate(d_alloc);
            return (size_t)bra.m * bra.n * work.n +
                   (size_t)work.m * work.n * ((conj_ket & 1) ? ket.m : ket.n);
        } else {
            GMatrix<FL> work(nullptr, ket.n, a.m);
            GMatrix<FL> work2(nullptr, work.m, bra.m);
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
    static size_t left_partial_rotate(const GMatrix<FL> &a, bool conj_a,
                         const GMatrix<FL> &c, bool conj_c,
                         const GMatrix<FL> &bra, const GMatrix<FL> &ket,
                         FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, conj_a ? a.n : a.m, ket.n);
        work.allocate(d_alloc);
        multiply(a, conj_a ? 3 : 0, ket, false, work, 1.0, 0.0);
        if (!conj_c)
            multiply(bra, 3, work, false, c, scale, 1.0);
        else
            multiply(work, 3, bra, false, c, conj(scale), 1.0);
        work.deallocate(d_alloc);
        return (size_t)a.m * a.n * work.n + (size_t)work.m * work.n * bra.n;
    }
    // c(.T) = bra.c * a(.T) * ket.t = (a(~.T) * bra.t).T * ket.t
    // return nflop. (.T) is always transpose conjugate
    static size_t right_partial_rotate(const GMatrix<FL> &a, bool conj_a,
                         const GMatrix<FL> &c, bool conj_c,
                         const GMatrix<FL> &bra, const GMatrix<FL> &ket,
                         FL scale) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> work(nullptr, conj_a ? a.m : a.n, bra.m);
        work.allocate(d_alloc);
        multiply(a, conj_a ? 0 : 3, bra, 1, work, 1.0, 0.0);
        if (!conj_c)
            multiply(work, 3, ket, 1, c, scale, 1.0);
        else
            multiply(ket, 2, work, false, c, conj(scale), 1.0);
        work.deallocate(d_alloc);
        return (size_t)a.m * a.n * work.n + (size_t)work.m * work.n * ket.m;
    }
    // dleft == true : c = bra (= da x db) * a * ket
    // dleft == false: c = bra * a * ket (= da x db)
    // return nflop. conj means conj and trans
    // conj means conj and trans / none for bra, trans / conj for ket
    static size_t three_rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                               const GMatrix<FL> &bra, bool conj_bra,
                               const GMatrix<FL> &ket, bool conj_ket,
                               const GMatrix<FL> &da, bool dconja,
                               const GMatrix<FL> &db, bool dconjb, bool dleft,
                               FL scale, uint32_t stride) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            GMatrix<FL> work(nullptr, am, conj_ket ? ket.m : ket.n);
            work.allocate(d_alloc);
            // work = a * ket
            multiply(GMatrix<FL>(&a(ast, 0), am, a.n), false, ket,
                     conj_ket ? 1 : 2, work, 1.0, 0.0);
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * work
                multiply(db, dconjb ? 3 : 0, work, false,
                         GMatrix<FL>(&c(cst, 0), cm, c.n),
                         scale * (dconja ? conj(*da.data) : *da.data), 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * work
                multiply(da, dconja ? 3 : 0, work, false,
                         GMatrix<FL>(&c(cst, 0), cm, c.n),
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
            GMatrix<FL> work(nullptr, a.m, kn);
            work.allocate(d_alloc);
            if (da.m == 1 && da.n == 1)
                // work = a * (1 x db)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, db,
                         dconjb ? 1 : 2, work,
                         (!dconja ? conj(*da.data) : *da.data) * scale, 0.0);
            else if (db.m == 1 && db.n == 1)
                // work = a * (da x 1)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, da,
                         dconja ? 1 : 2, work,
                         (!dconjb ? conj(*db.data) : *db.data) * scale, 0.0);
            else
                assert(false);
            // c = bra * work
            multiply(bra, conj_bra ? 3 : 0, work, false,
                     GMatrix<FL>(&c(0, cst), c.m, c.n), 1.0, 1.0);
            work.deallocate(d_alloc);
            return (size_t)km * kn * work.m + (size_t)work.m * work.n * c.m;
        }
    }
    // dleft == true : c = a * ket
    // dleft == false: c = a * ket (= da x db)
    // return nflop
    static size_t three_rotate_tr_left(const GMatrix<FL> &a,
                                       const GMatrix<FL> &c,
                                       const GMatrix<FL> &bra, bool conj_bra,
                                       const GMatrix<FL> &ket, bool conj_ket,
                                       const GMatrix<FL> &da, bool dconja,
                                       const GMatrix<FL> &db, bool dconjb,
                                       bool dleft, FL scale, uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            multiply(GMatrix<FL>(&a(ast, 0), am, a.n), false, ket,
                     conj_ket ? 1 : 2, GMatrix<FL>(&c(cst, 0), cm, c.n), scale,
                     1.0);
            return (size_t)ket.m * ket.n * am;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            if (da.m == 1 && da.n == 1)
                // c = a * (1 x db)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, db,
                         dconjb ? 1 : 2, GMatrix<FL>(&c(0, cst), c.m, c.n),
                         (!dconja ? conj(*da.data) : *da.data) * scale, 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = a * (da x 1)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, da,
                         dconja ? 1 : 2, GMatrix<FL>(&c(0, cst), c.m, c.n),
                         (!dconjb ? conj(*db.data) : *db.data) * scale, 1.0);
            else
                assert(false);
            return (size_t)km * kn * c.m;
        }
    }
    // dleft == true : c = bra (= da x db) * a
    // dleft == false: c = bra * a
    // return nflop
    static size_t three_rotate_tr_right(const GMatrix<FL> &a,
                                        const GMatrix<FL> &c,
                                        const GMatrix<FL> &bra, bool conj_bra,
                                        const GMatrix<FL> &ket, bool conj_ket,
                                        const GMatrix<FL> &da, bool dconja,
                                        const GMatrix<FL> &db, bool dconjb,
                                        bool dleft, FL scale, uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * a
                multiply(db, dconjb ? 3 : 0, GMatrix<FL>(&a(ast, 0), am, a.n),
                         false, GMatrix<FL>(&c(cst, 0), cm, c.n),
                         scale * (dconja ? conj(*da.data) : *da.data), 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * a
                multiply(da, dconja ? 3 : 0, GMatrix<FL>(&a(ast, 0), am, a.n),
                         false, GMatrix<FL>(&c(cst, 0), cm, c.n),
                         scale * (dconjb ? conj(*db.data) : *db.data), 1.0);
            else
                assert(false);
            return (size_t)am * a.n * cm;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            const FL cfactor = 1.0;
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            xgemm<FL>("n", conj_bra ? "c" : "n", &kn, &c.m, &a.m, &scale,
                      &a(0, ast), &a.n, bra.data, &bra.n, &cfactor, &c(0, cst),
                      &c.n);
            return (size_t)a.m * a.n * c.m;
        }
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(uint8_t abconj, const GMatrix<FL> &a,
                                        const GMatrix<FL> &b,
                                        const GMatrix<FL> &c, FL scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const FL cfactor = 1.0;
        const MKL_INT k = 1, lda = a.n + 1, ldb = b.n + 1;
        if (!(abconj & 1))
            xgemm<FL>(abconj & 2 ? "c" : "t", "n", &b.n, &a.n, &k, &scale,
                      b.data, &ldb, a.data, &lda, &cfactor, c.data, &c.n);
        else
            for (MKL_INT i = 0; i < a.n; i++)
                xgemm<FL>(abconj & 2 ? "c" : "t", "c", &b.n, &k, &k, &scale,
                          b.data, &ldb, a.data + i * lda, &k, &cfactor,
                          c.data + i * c.n, &c.n);
    }
    // diagonal element of three-matrix tensor product
    static void
    three_tensor_product_diagonal(uint8_t abconj, const GMatrix<FL> &a,
                                  const GMatrix<FL> &b, const GMatrix<FL> &c,
                                  const GMatrix<FL> &da, bool dconja,
                                  const GMatrix<FL> &db, bool dconjb,
                                  bool dleft, FL scale, uint32_t stride) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const FL cfactor = 1.0;
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
            const FL *bdata =
                dconjb ? &db(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &db(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft) {
                    // (1 x db) x b
                    if (!ddconjb)
                        xgemm<FL>(abconj & 2 ? "c" : "t", "n", &b.n, &dn, &k,
                                  &scale, b.data, &ldb, bdata, &lddb, &cfactor,
                                  &c(max(dstrn, dstrm), (MKL_INT)0), &c.n);
                    else
                        for (MKL_INT i = 0; i < dn; i++)
                            xgemm<FL>(abconj & 2 ? "c" : "t", "c", &b.n, &k, &k,
                                      &scale, b.data, &ldb, bdata + i * lddb,
                                      &k, &cfactor,
                                      &c(max(dstrn, dstrm) + i, (MKL_INT)0),
                                      &c.n);
                } else {
                    // a x (1 x db)
                    if (!(abconj & 1))
                        xgemm<FL>(ddconjb ? "c" : "t", "n", &dn, &a.n, &k,
                                  &scale, bdata, &lddb, a.data, &lda, &cfactor,
                                  &c(0, max(dstrn, dstrm)), &c.n);
                    else
                        for (MKL_INT i = 0; i < a.n; i++)
                            xgemm<FL>(ddconjb ? "c" : "t", "c", &dn, &k, &k,
                                      &scale, bdata, &lddb, a.data + i * lda,
                                      &k, &cfactor, &c(i, max(dstrn, dstrm)),
                                      &c.n);
                }
            }
        } else if (db.m == 1 && db.n == 1) {
            scale *= ddconjb ? conj(*db.data) : *db.data;
            const MKL_INT dn = da.n - abs(ddstr);
            const FL *adata =
                dconja ? &da(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &da(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft) {
                    // (da x 1) x b
                    if (!ddconja)
                        xgemm<FL>(abconj & 2 ? "c" : "t", "n", &b.n, &dn, &k,
                                  &scale, b.data, &ldb, adata, &ldda, &cfactor,
                                  &c(max(dstrn, dstrm), (MKL_INT)0), &c.n);
                    else
                        for (MKL_INT i = 0; i < dn; i++)
                            xgemm<FL>(abconj & 2 ? "c" : "t", "c", &b.n, &k, &k,
                                      &scale, b.data, &ldb, adata + i * ldda,
                                      &k, &cfactor,
                                      &c(max(dstrn, dstrm) + i, (MKL_INT)0),
                                      &c.n);
                } else {
                    // a x (da x 1)
                    if (!(abconj & 1))
                        xgemm<FL>(ddconja ? "c" : "t", "n", &dn, &a.n, &k,
                                  &scale, adata, &ldda, a.data, &lda, &cfactor,
                                  &c(0, max(dstrn, dstrm)), &c.n);
                    else
                        for (MKL_INT i = 0; i < a.n; i++)
                            xgemm<FL>(ddconja ? "c" : "t", "c", &dn, &k, &k,
                                      &scale, adata, &ldda, a.data + i * lda,
                                      &k, &cfactor, &c(i, max(dstrn, dstrm)),
                                      &c.n);
                }
            }
        } else
            assert(false);
    }
    static void tensor_product(const GMatrix<FL> &a, bool conja,
                               const GMatrix<FL> &b, bool conjb,
                               const GMatrix<FL> &c, FL scale,
                               uint32_t stride) {
        const FL cfactor = 1.0;
        switch ((uint8_t)conja | (conjb << 1)) {
        case 0:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const MKL_INT n = b.m * b.n;
                    xgemm<FL>("n", "n", &n, &a.n, &a.n, &scale, b.data, &n,
                              a.data, &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (MKL_INT k = 0; k < b.m; k++)
                        xgemm<FL>("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                                  &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                                  &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const MKL_INT n = a.m * a.n;
                    xgemm<FL>("n", "n", &n, &b.n, &b.n, &scale, a.data, &n,
                              b.data, &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (MKL_INT k = 0; k < a.m; k++)
                        xgemm<FL>("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                                  &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                                  &c.n);
                }
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++) {
                        const FL factor = scale * a(i, j);
                        for (MKL_INT k = 0; k < b.m; k++)
                            xaxpy<FL>(&b.n, &factor, &b(k, 0), &inc,
                                      &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 1:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const MKL_INT n = b.m * b.n;
                    xgemm<FL>("n", "c", &n, &a.n, &a.n, &scale, b.data, &n,
                              a.data, &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (MKL_INT k = 0; k < b.m; k++)
                        xgemm<FL>("n", "c", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                                  &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                                  &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                assert(a.m <= c.n);
                for (MKL_INT k = 0; k < a.n; k++)
                    xgemm<FL>("c", "n", &a.m, &b.n, &b.n, &scale, &a(0, k),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const FL factor = scale * conj(a(j, i));
                        for (MKL_INT k = 0; k < b.m; k++)
                            xaxpy<FL>(&b.n, &factor, &b(k, 0), &inc,
                                      &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 2:
            if (a.m == 1 && a.n == 1) {
                assert(b.m <= c.n);
                for (MKL_INT k = 0; k < b.n; k++)
                    xgemm<FL>("c", "n", &b.m, &a.n, &a.n, &scale, &b(0, k),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const MKL_INT n = a.m * a.n;
                    xgemm<FL>("n", "c", &n, &b.n, &b.n, &scale, a.data, &n,
                              b.data, &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (MKL_INT k = 0; k < a.m; k++)
                        xgemm<FL>("n", "c", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                                  &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                                  &c.n);
                }
            } else {
                for (MKL_INT i = 0, inca = 1, inc = b.m; i < b.n; i++)
                    for (MKL_INT j = 0; j < b.m; j++) {
                        const FL factor = scale * conj(b(j, i));
                        for (MKL_INT k = 0; k < a.m; k++)
                            xaxpy<FL>(&a.n, &factor, &a(k, 0), &inca,
                                      &c(k * b.n + i, j + stride), &inc);
                    }
            }
            break;
        case 1 | 2:
            if (a.m == 1 && a.n == 1) {
                for (MKL_INT k = 0; k < b.n; k++)
                    xgemm<FL>("c", "c", &b.m, &a.n, &a.n, &scale, &b(0, k),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
            } else if (b.m == 1 && b.n == 1) {
                for (MKL_INT k = 0; k < a.n; k++)
                    xgemm<FL>("c", "c", &a.m, &b.n, &b.n, &scale, &a(0, k),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
            } else {
                for (MKL_INT i = 0, incb = b.n, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const FL factor = scale * conj(a(j, i));
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
    static void svd(const GMatrix<FL> &a, const GMatrix<FL> &l,
                    const GMatrix<FP> &s, const GMatrix<FL> &r) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        MKL_INT k = min(a.m, a.n), info = 0, lwork = -1;
        FL twork;
        assert(a.m == l.m && a.n <= r.n && l.n >= k && r.m == k && s.n == k);
        xgesvd<FL>("S", "S", &a.n, &a.m, a.data, &a.n, s.data, r.data, &r.n,
                   l.data, &l.n, &twork, &lwork, &info);
        assert(info == 0);
        lwork = (MKL_INT)xreal<FL>(twork);
        FL *work = d_alloc->complex_allocate(lwork);
        xgesvd<FL>("S", "S", &a.n, &a.m, a.data, &a.n, s.data, r.data, &r.n,
                   l.data, &l.n, work, &lwork, &info);
        assert(info == 0);
        d_alloc->complex_deallocate(work, lwork);
    }
    // SVD for parallelism over sites; PRB 87, 155137 (2013)
    static void accurate_svd(const GMatrix<FL> &a, const GMatrix<FL> &l,
                             const GMatrix<FP> &s, const GMatrix<FL> &r,
                             FP eps = 1E-4) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        GMatrix<FL> aa(nullptr, a.m, a.n);
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
            GMatrix<FL> xa(nullptr, k - p, k - p), xl(nullptr, k - p, k - p),
                xr(nullptr, k - p, k - p);
            xa.data = d_alloc->complex_allocate(xa.size());
            xl.data = d_alloc->complex_allocate(xl.size());
            xr.data = d_alloc->complex_allocate(xr.size());
            rotate(a, xa, GMatrix<FL>(l.data + p, l.m, l.n), 3,
                   GMatrix<FL>(r.data + p * r.n, r.m - p, r.n), 3, 1.0);
            accurate_svd(xa, xl, GMatrix<FP>(s.data + p, 1, k - p), xr, eps);
            GMatrix<FL> bl(nullptr, l.m, l.n), br(nullptr, r.m, r.n);
            bl.data = d_alloc->complex_allocate(bl.size());
            br.data = d_alloc->complex_allocate(br.size());
            copy(bl, l);
            copy(br, r);
            multiply(GMatrix<FL>(bl.data + p, bl.m, bl.n), false, xl, false,
                     GMatrix<FL>(l.data + p, l.m, l.n), 1.0, 0.0);
            multiply(xr, false, GMatrix<FL>(br.data + p * br.n, br.m - p, br.n),
                     false, GMatrix<FL>(r.data + p * r.n, r.m - p, r.n), 1.0,
                     0.0);
            d_alloc->complex_deallocate(br.data, br.size());
            d_alloc->complex_deallocate(bl.data, bl.size());
            d_alloc->complex_deallocate(xr.data, xr.size());
            d_alloc->complex_deallocate(xl.data, xl.size());
            d_alloc->complex_deallocate(xa.data, xa.size());
        }
        d_alloc->complex_deallocate(aa.data, aa.size());
    }
    // LQ factorization
    static void lq(const GMatrix<FL> &a, const GMatrix<FL> &l,
                   const GMatrix<FL> &q) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.m;
        FL *work = d_alloc->complex_allocate(lwork);
        FL *tau = d_alloc->complex_allocate(k);
        FL *t = d_alloc->complex_allocate(a.m * a.n);
        assert(a.m == l.m && a.n == q.n && l.n == k && q.m == k);
        memcpy(t, a.data, sizeof(FL) * a.m * a.n);
        xgeqrf<FL>(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(l.data, 0, sizeof(FL) * k * a.m);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(l.data + j * k, t + j * a.n, sizeof(FL) * min(j + 1, k));
        xungqr<FL>(&a.n, &k, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memcpy(q.data, t, sizeof(FL) * k * a.n);
        d_alloc->complex_deallocate(t, a.m * a.n);
        d_alloc->complex_deallocate(tau, k);
        d_alloc->complex_deallocate(work, lwork);
    }
    // QR factorization
    static void qr(const GMatrix<FL> &a, const GMatrix<FL> &q,
                   const GMatrix<FL> &r) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.n;
        FL *work = d_alloc->complex_allocate(lwork);
        FL *tau = d_alloc->complex_allocate(k);
        FL *t = d_alloc->complex_allocate(a.m * a.n);
        assert(a.m == q.m && a.n == r.n && q.n == k && r.m == k);
        memcpy(t, a.data, sizeof(FL) * a.m * a.n);
        xgelqf<FL>(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(r.data, 0, sizeof(FL) * k * a.n);
        for (MKL_INT j = 0; j < k; j++)
            memcpy(r.data + j * a.n + j, t + j * a.n + j,
                   sizeof(FL) * (a.n - j));
        xunglq<FL>(&k, &a.m, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(q.data + j * k, t + j * a.n, sizeof(FL) * k);
        d_alloc->complex_deallocate(t, a.m * a.n);
        d_alloc->complex_deallocate(tau, k);
        d_alloc->complex_deallocate(work, lwork);
    }
    // a += b.T
    static void transpose(const GMatrix<FL> &a, const GMatrix<FL> &b,
                          FL scale = 1.0, FL cfactor = 1.0) {
        assert(a.m == b.n && a.n >= b.m);
        const FL one = 1.0;
        for (MKL_INT k = 0, inc = 1; k < b.n; k++)
            xgemm<FL>("t", "n", &b.m, &inc, &inc, &scale, &b(0, k), &b.n, &one,
                      &inc, &cfactor, &a(k, 0), &a.n);
    }
    // diagonalization for each symmetry block
    static void block_eigs(const GMatrix<FL> &a, const GDiagonalMatrix<FP> &w,
                           const vector<uint8_t> &x) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        uint8_t maxx = *max_element(x.begin(), x.end()) + 1;
        vector<vector<MKL_INT>> mp(maxx);
        assert(a.m == a.n && w.n == a.n && (MKL_INT)x.size() == a.n);
        for (MKL_INT i = 0; i < a.n; i++)
            mp[x[i]].push_back(i);
        for (uint8_t i = 0; i < maxx; i++)
            if (mp[i].size() != 0) {
                FL *work =
                    d_alloc->complex_allocate(mp[i].size() * mp[i].size());
                FP *wwork = d_alloc->allocate(mp[i].size());
                for (size_t j = 0; j < mp[i].size(); j++)
                    for (size_t k = 0; k < mp[i].size(); k++)
                        work[j * mp[i].size() + k] = a(mp[i][j], mp[i][k]);
                eigs(GMatrix<FL>(work, (MKL_INT)mp[i].size(),
                                 (MKL_INT)mp[i].size()),
                     GDiagonalMatrix<FP>(wwork, (MKL_INT)mp[i].size()));
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
    static void eigs(const GMatrix<FL> &a, const GDiagonalMatrix<FP> &w) {
        shared_ptr<VectorAllocator<FP>> d_alloc =
            make_shared<VectorAllocator<FP>>();
        assert(a.m == a.n && w.n == a.n);
        const FP scale = -1.0;
        MKL_INT lwork = -1, n = a.m * a.n, incx = 2, info;
        FL twork;
        FP *rwork = d_alloc->allocate(max((MKL_INT)1, 3 * a.n - 2));
        xheev<FL>("V", "U", &a.n, a.data, &a.n, w.data, &twork, &lwork, rwork,
                  &info);
        assert(info == 0);
        lwork = (MKL_INT)xreal<FL>(twork);
        FL *work = d_alloc->complex_allocate(lwork);
        xheev<FL>("V", "U", &a.n, a.data, &a.n, w.data, work, &lwork, rwork,
                  &info);
        assert((size_t)a.m * a.n == n);
        xscal<FP>(&n, &scale, (FP *)a.data + 1, &incx);
        assert(info == 0);
        d_alloc->complex_deallocate(work, lwork);
        d_alloc->deallocate(rwork, max((MKL_INT)1, 3 * a.n - 2));
    }
    // z = r / aa
    static void cg_precondition(const GMatrix<FL> &z, const GMatrix<FL> &r,
                                const GDiagonalMatrix<FL> &aa) {
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
                                            const FL *h, MKL_INT ldh, FP t,
                                            FL *work) {
        static const FL zero = 0.0, one = 1.0, mone = -1.0;
        static const FP dtwo = 2.0, dmone = -1.0;
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
        memset(work, 0, sizeof(FL) * m);
        for (MKL_INT j = 0; j < m; j++)
            for (MKL_INT i = 0; i < m; i++)
                work[i] += abs(h[j * m + i]);
        FP hnorm = 0.0;
        for (MKL_INT i = 0; i < m; i++)
            hnorm = max(hnorm, work[i].real());
        hnorm = abs(t * hnorm);
        if (hnorm == 0.0) {
            cerr << "Error - null H in expo pade" << endl;
            abort();
        }
        MKL_INT ns = max((MKL_INT)0, (MKL_INT)(log(hnorm) / log(2.0)) + 2);
        FL scale = t / (FP)(1LL << ns);
        FL scale2 = scale * scale;
        // compute Pade coefficients
        MKL_INT i = ideg + 1, j = 2 * ideg + 1;
        work[icoef] = 1.0;
        for (MKL_INT k = 1; k <= ideg; k++)
            work[icoef + k] =
                work[icoef + k - 1] * (FP)(i - k) / FP(k * (j - k));
        // H2 = scale2*H*H ...
        xgemm<FL>("n", "n", &m, &m, &m, &scale2, h, &ldh, h, &ldh, &zero,
                  work + ih2, &m);
        // initialize p (numerator) and q (denominator)
        memset(work + ip, 0, sizeof(FL) * mm * 2);
        FL cp = work[icoef + ideg - 1];
        FL cq = work[icoef + ideg];
        for (MKL_INT j = 0; j < m; j++)
            work[ip + j * (m + 1)] = cp, work[iq + j * (m + 1)] = cq;
        // Apply Horner rule
        MKL_INT iodd = 1;
        for (MKL_INT k = ideg - 1; k > 0; k--) {
            MKL_INT iused = iodd * iq + (1 - iodd) * ip;
            xgemm<FL>("n", "n", &m, &m, &m, &one, work + iused, &m, work + ih2,
                      &m, &zero, work + ifree, &m);
            for (MKL_INT j = 0; j < m; j++)
                work[ifree + j * (m + 1)] += work[icoef + k - 1];
            ip = (1 - iodd) * ifree + iodd * ip;
            iq = iodd * ifree + (1 - iodd) * iq;
            ifree = iused;
            iodd = 1 - iodd;
        }
        // Obtain (+/-)(I + 2*(p\q))
        MKL_INT *iqp = iodd ? &iq : &ip;
        xgemm<FL>("n", "n", &m, &m, &m, &scale, work + *iqp, &m, h, &ldh, &zero,
                  work + ifree, &m);
        *iqp = ifree;
        xaxpy<FL>(&mm, &mone, work + ip, &inc, work + iq, &inc);
        xgesv<FL>(&m, &m, work + iq, &m, (MKL_INT *)work + ih2, work + ip, &m,
                  &iflag);
        if (iflag != 0) {
            cerr << "Problem in DGESV in expo pade" << endl;
            abort();
        }
        xdscal<FL>(&mm, &dtwo, work + ip, &inc);
        for (MKL_INT j = 0; j < m; j++)
            work[ip + j * (m + 1)] = work[ip + j * (m + 1)] + one;
        MKL_INT iput = ip;
        if (ns == 0 && iodd) {
            xdscal<FL>(&mm, &dmone, work + ip, &inc);
        } else {
            // squaring : exp(t*H) = (exp(t*H))^(2^ns)
            iodd = 1;
            for (MKL_INT k = 0; k < ns; k++) {
                MKL_INT iget = iodd * ip + (1 - iodd) * iq;
                iput = (1 - iodd) * ip + iodd * iq;
                xgemm<FL>("n", "n", &m, &m, &m, &one, work + iget, &m,
                          work + iget, &m, &zero, work + iput, &m);
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
    static MKL_INT expo_krylov(MatMul &op, MKL_INT n, MKL_INT m, FP t, FL *v,
                               FL *w, FP &tol, FP anorm, FL *work,
                               MKL_INT lwork, bool iprint,
                               const PComm &pcomm = nullptr) {
        const MKL_INT inc = 1;
        const FP sqr1 = sqrt(0.1);
        const FL zero = 0.0;
        const MKL_INT mxstep = 500, mxreject = 0, ideg = 6;
        const FP delta = 1.2, gamma = 0.9;
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
        FP t_out = abs(t), tbrkdwn = 0.0, t_now = 0.0, t_new = 0.0;
        FP step_min = t_out, step_max = 0.0, s_error = 0.0, x_error = 0.0;
        FP err_loc;
        MKL_INT nstep = 0;
        // machine precision
        FP eps = 0.0;
        for (FP p1 = 4.0 / 3.0, p2, p3; eps == 0.0;)
            p2 = p1 - 1.0, p3 = p2 + p2 + p2, eps = abs(p3 - 1.0);
        if (tol <= eps)
            tol = sqrt(eps);
        FP rndoff = eps * anorm, break_tol = 1E-7;
        FP sgn = t >= 0 ? 1.0 : -1.0;
        xcopy<FL>(&n, v, &inc, w, &inc);
        FP beta = xnrm2<FL>(&n, w, &inc), vnorm = beta, hump = beta, avnorm;
        // obtain the very first stepsize
        FP xm = 1.0 / (FP)m, p1;
        p1 = tol * pow((m + 1) / 2.72, m + 1) * sqrt(2.0 * 3.14 * (m + 1));
        t_new = (1.0 / anorm) * pow(p1 / (4.0 * beta * anorm), xm);
        p1 = pow(10.0, round(log10(t_new) - sqr1) - 1);
        t_new = floor(t_new / p1 + 0.55) * p1;
        FL hij;
        // step-by-step integration
        for (; t_now < t_out;) {
            nstep++;
            FP t_step = min(t_out - t_now, t_new);
            p1 = 1.0 / beta;
            for (MKL_INT i = 0; i < n; i++)
                work[iv + i] = p1 * w[i];
            if (pcomm == nullptr || pcomm->root == pcomm->rank)
                memset(work + ih, 0, sizeof(FL) * mh * mh);
            // Arnoldi loop
            MKL_INT j1v = iv + n;
            FP hj1j = 0.0;
            for (MKL_INT j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    for (MKL_INT i = 0; i <= j; i++) {
                        hij = -complex_dot(GMatrix<FL>(work + iv + i * n, n, 1),
                                           GMatrix<FL>(work + j1v, n, 1));
                        xaxpy<FL>(&n, &hij, work + iv + i * n, &inc, work + j1v,
                                  &inc);
                        work[ih + j * mh + i] = -hij;
                    }
                    hj1j = xnrm2<FL>(&n, work + j1v, &inc);
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
                    work[ih + j * mh + j + 1] = (FL)hj1j;
                    hj1j = 1.0 / hj1j;
                    xdscal<FL>(&n, &hj1j, work + j1v, &inc);
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(work + j1v, n, pcomm->root);
                j1v += n;
            }
            if (k1 != 0) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank)
                    avnorm = xnrm2<FL>(&n, work + j1v, &inc);
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
                        FP p1 = abs(work[iexph + m]) * beta;
                        FP p2 = abs(work[iexph + m + 1]) * beta * avnorm;
                        if (p1 > 10.0 * p2)
                            err_loc = p2, xm = 1.0 / (FP)m;
                        else if (p1 > p2)
                            err_loc = p1 * p2 / (p1 - p2), xm = 1.0 / (FP)m;
                        else
                            err_loc = p1, xm = 1.0 / (FP)(m - 1);
                    }
                    // reject the step-size if the error is not acceptable
                    if (k1 != 0 && err_loc > delta * t_step * tol &&
                        (mxreject == 0 || ireject < mxreject)) {
                        FP t_old = t_step;
                        t_step =
                            gamma * t_step * pow(t_step * tol / err_loc, xm);
                        p1 = pow((FP)10.0, round(log10(t_step) - sqr1) - 1);
                        t_step = floor(t_step / p1 + (FP)0.55) * p1;
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
                FL hjj = (FL)beta;
                xgemv<FL>("n", &n, &mx, &hjj, work + iv, &n, work + iexph, &inc,
                          &zero, w, &inc);
                beta = xnrm2<FL>(&n, w, &inc);
                hump = max(hump, beta);
                // suggested value for the next stepsize
                t_new = gamma * t_step * pow(t_step * tol / err_loc, xm);
                p1 = pow((FP)10.0, round(log10(t_new) - sqr1) - 1);
                t_new = floor(t_new / p1 + (FP)0.55) * p1;
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
                FP tmp[3] = {beta, t_new, t_now};
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
    static int expo_apply(MatMul &op, FL t, FP anorm, GMatrix<FP> &vr,
                          GMatrix<FP> &vi, FP consta = 0.0, bool iprint = false,
                          const PComm &pcomm = nullptr, FP conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        const MKL_INT vm = vr.m, vn = vr.n, n = vm * vn;
        assert(vi.m == vr.m && vi.n == vr.n);
        auto cop = [&op, vm, vn, n](const GMatrix<FL> &a,
                                    const GMatrix<FL> &b) -> void {
            vector<FP> dar(n), dai(n), dbr(n, 0), dbi(n, 0);
            extract_complex(a, GMatrix<FP>(dar.data(), vm, vn),
                            GMatrix<FP>(dai.data(), vm, vn));
            op(GMatrix<FP>(dar.data(), vm, vn),
               GMatrix<FP>(dbr.data(), vm, vn));
            op(GMatrix<FP>(dai.data(), vm, vn),
               GMatrix<FP>(dbi.data(), vm, vn));
            fill_complex(b, GMatrix<FP>(dbr.data(), vm, vn),
                         GMatrix<FP>(dbi.data(), vm, vn));
        };
        vector<FL> v(n);
        GMatrix<FL> cv(v.data(), vm, vn);
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
    static int expo_apply_complex_op(MatMul &op, FL t, FP anorm, GMatrix<FL> &v,
                                     FP consta = 0.0, bool iprint = false,
                                     const PComm &pcomm = nullptr,
                                     FP conv_thrd = 5E-6,
                                     int deflation_max_size = 20) {
        MKL_INT vm = v.m, vn = v.n, n = vm * vn;
        FP abst = abs(t);
        assert(abst != 0);
        FL tt = t / abst;
        if (n < 4) {
            const MKL_INT lwork = 4 * n * n + 7;
            vector<FL> h(n * n), work(lwork);
            vector<FL> te(n), to(n);
            GMatrix<FL> e = GMatrix<FL>(te.data(), vm, vn);
            GMatrix<FL> o = GMatrix<FL>(to.data(), vm, vn);
            memset(e.data, 0, sizeof(FL) * n);
            for (MKL_INT i = 0; i < n; i++) {
                e.data[i] = 1.0;
                memset(o.data, 0, sizeof(FL) * n);
                op(e, o);
                for (MKL_INT j = 0; j < n; j++)
                    h[i * n + j] = tt * o.data[j];
                h[i * (n + 1)] += tt * consta;
                e.data[i] = 0.0;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                MKL_INT iptr =
                    expo_pade(6, n, h.data(), n, abst, work.data()).first;
                vector<FL> w(n, 0);
                GMatrix<FL> mvin = GMatrix<FL>(v.data, v.m, v.n);
                GMatrix<FL> mvout = GMatrix<FL>(w.data(), v.m, v.n);
                multiply(GMatrix<FL>(work.data() + iptr, n, n), true, mvin,
                         false, mvout, 1.0, 0.0);
                memcpy(v.data, w.data(), sizeof(FL) * w.size());
            }
            if (pcomm != nullptr)
                pcomm->broadcast(v.data, n, pcomm->root);
            return n;
        }
        auto lop = [&op, consta, n, vm, vn, tt](FL *a, FL *b) -> void {
            static MKL_INT inc = 1;
            static FL x = 1.0;
            op(GMatrix<FL>(a, vm, vn), GMatrix<FL>(b, vm, vn));
            const FL cconsta = consta * tt;
            xgemm<FL>("n", "n", &inc, &n, &inc, &x, &cconsta, &inc, a, &inc,
                      &tt, b, &inc);
        };
        MKL_INT m = min((MKL_INT)deflation_max_size, n - 1);
        MKL_INT lwork = n * (m + 2) + 5 * (m + 2) * (m + 2) + 7;
        vector<FL> w(n), work(lwork);
        anorm = (anorm + abs(consta) * n) * abs(tt);
        if (anorm < (FP)1E-10)
            anorm = 1.0;
        MKL_INT nmult =
            expo_krylov(lop, n, m, abst, v.data, w.data(), conv_thrd, anorm,
                        work.data(), lwork, iprint, (PComm)pcomm);
        memcpy(v.data, w.data(), sizeof(FL) * w.size());
        return (int)nmult;
    }
};

typedef GMatrixFunctions<complex<double>> ComplexMatrixFunctions;

} // namespace block2
