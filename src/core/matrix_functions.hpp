
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
// vector [sx] = float [sa] * vector [sx]
extern void FNAME(sscal)(const MKL_INT *n, const float *sa, float *sx,
                         const MKL_INT *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void FNAME(scopy)(const MKL_INT *n, const float *dx, const MKL_INT *incx,
                         float *dy, const MKL_INT *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + float [sa] * vector [sx]
extern void FNAME(saxpy)(const MKL_INT *n, const float *sa, const float *sx,
                         const MKL_INT *incx, float *sy,
                         const MKL_INT *incy) noexcept;

// vector dot product
extern float FNAME(sdot)(const MKL_INT *n, const float *dx, const MKL_INT *incx,
                         const float *dy, const MKL_INT *incy) noexcept;

// Euclidean norm of a vector
extern float FNAME(snrm2)(const MKL_INT *n, const float *x,
                          const MKL_INT *incx) noexcept;

// matrix multiplication
// mat [c] = float [alpha] * mat [a] * mat [b] + float [beta] * mat [c]
extern void FNAME(sgemm)(const char *transa, const char *transb,
                         const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                         const float *alpha, const float *a, const MKL_INT *lda,
                         const float *b, const MKL_INT *ldb, const float *beta,
                         float *c, const MKL_INT *ldc) noexcept;

// matrix-vector multiplication
// vec [y] = float [alpha] * mat [a] * vec [x] + float [beta] * vec [y]
extern void FNAME(sgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const float *alpha, const float *a, const MKL_INT *lda,
                         const float *x, const MKL_INT *incx, const float *beta,
                         float *y, const MKL_INT *incy) noexcept;

// linear system a * x = b
extern void FNAME(sgesv)(const MKL_INT *n, const MKL_INT *nrhs, float *a,
                         const MKL_INT *lda, MKL_INT *ipiv, float *b,
                         const MKL_INT *ldb, MKL_INT *info);

// QR factorization
extern void FNAME(sgeqrf)(const MKL_INT *m, const MKL_INT *n, float *a,
                          const MKL_INT *lda, float *tau, float *work,
                          const MKL_INT *lwork, MKL_INT *info);
extern void FNAME(sorgqr)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          float *a, const MKL_INT *lda, const float *tau,
                          float *work, const MKL_INT *lwork, MKL_INT *info);

// LQ factorization
extern void FNAME(sgelqf)(const MKL_INT *m, const MKL_INT *n, float *a,
                          const MKL_INT *lda, float *tau, float *work,
                          const MKL_INT *lwork, MKL_INT *info);
extern void FNAME(sorglq)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          float *a, const MKL_INT *lda, const float *tau,
                          float *work, const MKL_INT *lwork, MKL_INT *info);

// LU factorization
extern void FNAME(sgetrf)(const MKL_INT *m, const MKL_INT *n, float *a,
                          const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info);

// matrix inverse
extern void FNAME(sgetri)(const MKL_INT *m, float *a, const MKL_INT *lda,
                          MKL_INT *ipiv, float *work, const MKL_INT *lwork,
                          MKL_INT *info);

// eigenvalue problem
extern void FNAME(ssyev)(const char *jobz, const char *uplo, const MKL_INT *n,
                         float *a, const MKL_INT *lda, float *w, float *work,
                         const MKL_INT *lwork, MKL_INT *info);

extern void FNAME(sgeev)(const char *jobvl, const char *jobvr, const MKL_INT *n,
                         float *a, const MKL_INT *lda, float *wr, float *wi,
                         float *vl, const MKL_INT *ldvl, float *vr,
                         const MKL_INT *ldvr, float *work, const MKL_INT *lwork,
                         MKL_INT *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void FNAME(sgesvd)(const char *jobu, const char *jobvt, const MKL_INT *m,
                          const MKL_INT *n, float *a, const MKL_INT *lda,
                          float *s, float *u, const MKL_INT *ldu, float *vt,
                          const MKL_INT *ldvt, float *work,
                          const MKL_INT *lwork, MKL_INT *info);

// least squares problem a * x = b
extern void FNAME(sgels)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const MKL_INT *nrhs, float *a, const MKL_INT *lda,
                         float *b, const MKL_INT *ldb, float *work,
                         const MKL_INT *lwork, MKL_INT *info);

// matrix copy
// mat [b] = mat [a]
extern void FNAME(slacpy)(const char *uplo, const int *m, const int *n,
                          const float *a, const int *lda, float *b,
                          const int *ldb);

// vector scale
// vector [sx] = float [sa] * vector [sx]
extern void FNAME(dscal)(const MKL_INT *n, const double *sa, double *sx,
                         const MKL_INT *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void FNAME(dcopy)(const MKL_INT *n, const double *dx,
                         const MKL_INT *incx, double *dy,
                         const MKL_INT *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + double [sa] * vector [sx]
extern void FNAME(daxpy)(const MKL_INT *n, const double *sa, const double *sx,
                         const MKL_INT *incx, double *sy,
                         const MKL_INT *incy) noexcept;

// vector dot product
extern double FNAME(ddot)(const MKL_INT *n, const double *dx,
                          const MKL_INT *incx, const double *dy,
                          const MKL_INT *incy) noexcept;

// Euclidean norm of a vector
extern double FNAME(dnrm2)(const MKL_INT *n, const double *x,
                           const MKL_INT *incx) noexcept;

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void FNAME(dgemm)(const char *transa, const char *transb,
                         const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                         const double *alpha, const double *a,
                         const MKL_INT *lda, const double *b,
                         const MKL_INT *ldb, const double *beta, double *c,
                         const MKL_INT *ldc) noexcept;

// matrix-vector multiplication
// vec [y] = double [alpha] * mat [a] * vec [x] + double [beta] * vec [y]
extern void FNAME(dgemv)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const double *alpha, const double *a,
                         const MKL_INT *lda, const double *x,
                         const MKL_INT *incx, const double *beta, double *y,
                         const MKL_INT *incy) noexcept;

// linear system a * x = b
extern void FNAME(dgesv)(const MKL_INT *n, const MKL_INT *nrhs, double *a,
                         const MKL_INT *lda, MKL_INT *ipiv, double *b,
                         const MKL_INT *ldb, MKL_INT *info);

// QR factorization
extern void FNAME(dgeqrf)(const MKL_INT *m, const MKL_INT *n, double *a,
                          const MKL_INT *lda, double *tau, double *work,
                          const MKL_INT *lwork, MKL_INT *info);
extern void FNAME(dorgqr)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          double *a, const MKL_INT *lda, const double *tau,
                          double *work, const MKL_INT *lwork, MKL_INT *info);

// LQ factorization
extern void FNAME(dgelqf)(const MKL_INT *m, const MKL_INT *n, double *a,
                          const MKL_INT *lda, double *tau, double *work,
                          const MKL_INT *lwork, MKL_INT *info);
extern void FNAME(dorglq)(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          double *a, const MKL_INT *lda, const double *tau,
                          double *work, const MKL_INT *lwork, MKL_INT *info);

// LU factorization
extern void FNAME(dgetrf)(const MKL_INT *m, const MKL_INT *n, double *a,
                          const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info);

// matrix inverse
extern void FNAME(dgetri)(const MKL_INT *m, double *a, const MKL_INT *lda,
                          MKL_INT *ipiv, double *work, const MKL_INT *lwork,
                          MKL_INT *info);

// eigenvalue problem
extern void FNAME(dsyev)(const char *jobz, const char *uplo, const MKL_INT *n,
                         double *a, const MKL_INT *lda, double *w, double *work,
                         const MKL_INT *lwork, MKL_INT *info);

extern void FNAME(dgeev)(const char *jobvl, const char *jobvr, const MKL_INT *n,
                         double *a, const MKL_INT *lda, double *wr, double *wi,
                         double *vl, const MKL_INT *ldvl, double *vr,
                         const MKL_INT *ldvr, double *work,
                         const MKL_INT *lwork, MKL_INT *info);

// SVD
// mat [a] = mat [u] * vector [sigma] * mat [vt]
extern void FNAME(dgesvd)(const char *jobu, const char *jobvt, const MKL_INT *m,
                          const MKL_INT *n, double *a, const MKL_INT *lda,
                          double *s, double *u, const MKL_INT *ldu, double *vt,
                          const MKL_INT *ldvt, double *work,
                          const MKL_INT *lwork, MKL_INT *info);

// least squares problem a * x = b
extern void FNAME(dgels)(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const MKL_INT *nrhs, double *a, const MKL_INT *lda,
                         double *b, const MKL_INT *ldb, double *work,
                         const MKL_INT *lwork, MKL_INT *info);

// matrix copy
// mat [b] = mat [a]
extern void FNAME(dlacpy)(const char *uplo, const int *m, const int *n,
                          const double *a, const int *lda, double *b,
                          const int *ldb);

#endif
}

enum struct DavidsonTypes : uint16_t {
    Normal = 0,
    GreaterThan = 1,
    LessThan = 2,
    CloseTo = 4,
    Harmonic = 16,
    HarmonicGreaterThan = 16 | 1,
    HarmonicLessThan = 16 | 2,
    HarmonicCloseTo = 16 | 4,
    DavidsonPrecond = 32,
    NoPrecond = 64,
    NonHermitian = 128,
    Exact = 256,
    LeftEigen = 512,
    ExactNonHermitian = 128 | 256,
    ExactNonHermitianLeftEigen = 128 | 256 | 512,
    NonHermitianDavidsonPrecond = 128 | 32,
    NonHermitianDavidsonPrecondLeftEigen = 128 | 32 | 512,
    NonHermitianLeftEigen = 128 | 512
};

inline bool operator&(DavidsonTypes a, DavidsonTypes b) {
    return ((uint16_t)a & (uint16_t)b) != 0;
}

inline DavidsonTypes operator|(DavidsonTypes a, DavidsonTypes b) {
    return DavidsonTypes((uint16_t)a | (uint16_t)b);
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
    return FNAME(dgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <>
inline void xgemm<float>(const char *transa, const char *transb,
                         const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                         const float *alpha, const float *a, const MKL_INT *lda,
                         const float *b, const MKL_INT *ldb, const float *beta,
                         float *c, const MKL_INT *ldc) noexcept {
    return FNAME(sgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <typename FL>
inline void xscal(const MKL_INT *n, const FL *sa, FL *sx,
                  const MKL_INT *incx) noexcept;

template <>
inline void xscal<double>(const MKL_INT *n, const double *sa, double *sx,
                          const MKL_INT *incx) noexcept {
    FNAME(dscal)(n, sa, sx, incx);
}

template <>
inline void xscal<float>(const MKL_INT *n, const float *sa, float *sx,
                         const MKL_INT *incx) noexcept {
    FNAME(sscal)(n, sa, sx, incx);
}

template <typename FL>
inline void xdscal(const MKL_INT *n, const typename GMatrix<FL>::FP *sa, FL *sx,
                   const MKL_INT *incx) noexcept;

template <>
inline void xdscal<double>(const MKL_INT *n, const double *sa, double *sx,
                           const MKL_INT *incx) noexcept {
    FNAME(dscal)(n, sa, sx, incx);
}

template <>
inline void xdscal<float>(const MKL_INT *n, const float *sa, float *sx,
                          const MKL_INT *incx) noexcept {
    FNAME(sscal)(n, sa, sx, incx);
}

template <typename FL>
inline typename GMatrix<FL>::FP xnrm2(const MKL_INT *n, const FL *x,
                                      const MKL_INT *incx) noexcept;

template <>
inline double xnrm2<double>(const MKL_INT *n, const double *x,
                            const MKL_INT *incx) noexcept {
    return FNAME(dnrm2)(n, x, incx);
}

template <>
inline float xnrm2<float>(const MKL_INT *n, const float *x,
                          const MKL_INT *incx) noexcept {
    return FNAME(snrm2)(n, x, incx);
}

template <typename FL>
inline void xcopy(const MKL_INT *n, const FL *dx, const MKL_INT *incx, FL *dy,
                  const MKL_INT *incy) noexcept;

template <>
inline void xcopy<double>(const MKL_INT *n, const double *dx,
                          const MKL_INT *incx, double *dy,
                          const MKL_INT *incy) noexcept {
    FNAME(dcopy)(n, dx, incx, dy, incy);
}

template <>
inline void xcopy<float>(const MKL_INT *n, const float *dx, const MKL_INT *incx,
                         float *dy, const MKL_INT *incy) noexcept {
    FNAME(scopy)(n, dx, incx, dy, incy);
}

template <typename FL>
inline FL xdot(const MKL_INT *n, const FL *dx, const MKL_INT *incx,
               const FL *dy, const MKL_INT *incy) noexcept;

template <>
inline double xdot<double>(const MKL_INT *n, const double *dx,
                           const MKL_INT *incx, const double *dy,
                           const MKL_INT *incy) noexcept {
    return FNAME(ddot)(n, dx, incx, dy, incy);
}

template <>
inline float xdot<float>(const MKL_INT *n, const float *dx, const MKL_INT *incx,
                         const float *dy, const MKL_INT *incy) noexcept {
    return FNAME(sdot)(n, dx, incx, dy, incy);
}

template <typename FL>
inline void xaxpy(const MKL_INT *n, const FL *sa, const FL *sx,
                  const MKL_INT *incx, FL *sy, const MKL_INT *incy) noexcept;

template <>
inline void xaxpy<double>(const MKL_INT *n, const double *sa, const double *sx,
                          const MKL_INT *incx, double *sy,
                          const MKL_INT *incy) noexcept {
    FNAME(daxpy)(n, sa, sx, incx, sy, incy);
}

template <>
inline void xaxpy<float>(const MKL_INT *n, const float *sa, const float *sx,
                         const MKL_INT *incx, float *sy,
                         const MKL_INT *incy) noexcept {
    FNAME(saxpy)(n, sa, sx, incx, sy, incy);
}

template <typename FL>
inline void xlacpy(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                   const FL *a, const MKL_INT *lda, FL *b, const MKL_INT *ldb);

template <>
inline void xlacpy<double>(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                           const double *a, const MKL_INT *lda, double *b,
                           const MKL_INT *ldb) {
    FNAME(dlacpy)(uplo, m, n, a, lda, b, ldb);
}
template <>
inline void xlacpy<float>(const char *uplo, const MKL_INT *m, const MKL_INT *n,
                          const float *a, const MKL_INT *lda, float *b,
                          const MKL_INT *ldb) {
    FNAME(slacpy)(uplo, m, n, a, lda, b, ldb);
}

template <typename FL>
inline void xgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const FL *alpha, const FL *a, const MKL_INT *lda, const FL *x,
                  const MKL_INT *incx, const FL *beta, FL *y,
                  const MKL_INT *incy);

template <>
inline void xgemv<double>(const char *trans, const MKL_INT *m, const MKL_INT *n,
                          const double *alpha, const double *a,
                          const MKL_INT *lda, const double *x,
                          const MKL_INT *incx, const double *beta, double *y,
                          const MKL_INT *incy) {
    FNAME(dgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void xgemv<float>(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const float *alpha, const float *a, const MKL_INT *lda,
                         const float *x, const MKL_INT *incx, const float *beta,
                         float *y, const MKL_INT *incy) {
    FNAME(sgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <typename FL>
inline void xgesv(const MKL_INT *n, const MKL_INT *nrhs, FL *a,
                  const MKL_INT *lda, MKL_INT *ipiv, FL *b, const MKL_INT *ldb,
                  MKL_INT *info);

template <>
inline void xgesv<double>(const MKL_INT *n, const MKL_INT *nrhs, double *a,
                          const MKL_INT *lda, MKL_INT *ipiv, double *b,
                          const MKL_INT *ldb, MKL_INT *info) {
    FNAME(dgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
}

template <>
inline void xgesv<float>(const MKL_INT *n, const MKL_INT *nrhs, float *a,
                         const MKL_INT *lda, MKL_INT *ipiv, float *b,
                         const MKL_INT *ldb, MKL_INT *info) {
    FNAME(sgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
}

template <typename FL>
inline void xgeqrf(const MKL_INT *m, const MKL_INT *n, FL *a,
                   const MKL_INT *lda, FL *tau, FL *work, const MKL_INT *lwork,
                   MKL_INT *info);
template <>
inline void xgeqrf<double>(const MKL_INT *m, const MKL_INT *n, double *a,
                           const MKL_INT *lda, double *tau, double *work,
                           const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dgeqrf)(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgeqrf<float>(const MKL_INT *m, const MKL_INT *n, float *a,
                          const MKL_INT *lda, float *tau, float *work,
                          const MKL_INT *lwork, MKL_INT *info) {
    FNAME(sgeqrf)(m, n, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xungqr(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, FL *a,
                   const MKL_INT *lda, const FL *tau, FL *work,
                   const MKL_INT *lwork, MKL_INT *info);
template <>
inline void xungqr<double>(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                           double *a, const MKL_INT *lda, const double *tau,
                           double *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dorgqr)(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xungqr<float>(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          float *a, const MKL_INT *lda, const float *tau,
                          float *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(sorgqr)(m, n, k, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xgelqf(const MKL_INT *m, const MKL_INT *n, FL *a,
                   const MKL_INT *lda, FL *tau, FL *work, const MKL_INT *lwork,
                   MKL_INT *info);
template <>
inline void xgelqf<double>(const MKL_INT *m, const MKL_INT *n, double *a,
                           const MKL_INT *lda, double *tau, double *work,
                           const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dgelqf)(m, n, a, lda, tau, work, lwork, info);
}
template <>
inline void xgelqf<float>(const MKL_INT *m, const MKL_INT *n, float *a,
                          const MKL_INT *lda, float *tau, float *work,
                          const MKL_INT *lwork, MKL_INT *info) {
    FNAME(sgelqf)(m, n, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xunglq(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, FL *a,
                   const MKL_INT *lda, const FL *tau, FL *work,
                   const MKL_INT *lwork, MKL_INT *info);
template <>
inline void xunglq<double>(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                           double *a, const MKL_INT *lda, const double *tau,
                           double *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dorglq)(m, n, k, a, lda, tau, work, lwork, info);
}
template <>
inline void xunglq<float>(const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          float *a, const MKL_INT *lda, const float *tau,
                          float *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(sorglq)(m, n, k, a, lda, tau, work, lwork, info);
}

template <typename FL>
inline void xgetrf(const MKL_INT *m, const MKL_INT *n, FL *a,
                   const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info);

template <>
inline void xgetrf<double>(const MKL_INT *m, const MKL_INT *n, double *a,
                           const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info) {
    FNAME(dgetrf)(m, n, a, lda, ipiv, info);
}

template <>
inline void xgetrf<float>(const MKL_INT *m, const MKL_INT *n, float *a,
                          const MKL_INT *lda, MKL_INT *ipiv, MKL_INT *info) {
    FNAME(sgetrf)(m, n, a, lda, ipiv, info);
}

template <typename FL>
inline void xgetri(const MKL_INT *m, FL *a, const MKL_INT *lda, MKL_INT *ipiv,
                   FL *work, const MKL_INT *lwork, MKL_INT *info);

template <>
inline void xgetri<double>(const MKL_INT *m, double *a, const MKL_INT *lda,
                           MKL_INT *ipiv, double *work, const MKL_INT *lwork,
                           MKL_INT *info) {
    FNAME(dgetri)(m, a, lda, ipiv, work, lwork, info);
}

template <>
inline void xgetri<float>(const MKL_INT *m, float *a, const MKL_INT *lda,
                          MKL_INT *ipiv, float *work, const MKL_INT *lwork,
                          MKL_INT *info) {
    FNAME(sgetri)(m, a, lda, ipiv, work, lwork, info);
}

template <typename FL>
inline void xgesvd(const char *jobu, const char *jobvt, const MKL_INT *m,
                   const MKL_INT *n, FL *a, const MKL_INT *lda,
                   typename GMatrix<FL>::FP *s, FL *u, const MKL_INT *ldu,
                   FL *vt, const MKL_INT *ldvt, FL *work, const MKL_INT *lwork,
                   MKL_INT *info);
template <>
inline void xgesvd<double>(const char *jobu, const char *jobvt,
                           const MKL_INT *m, const MKL_INT *n, double *a,
                           const MKL_INT *lda, double *s, double *u,
                           const MKL_INT *ldu, double *vt, const MKL_INT *ldvt,
                           double *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dgesvd)
    (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}
template <>
inline void xgesvd<float>(const char *jobu, const char *jobvt, const MKL_INT *m,
                          const MKL_INT *n, float *a, const MKL_INT *lda,
                          float *s, float *u, const MKL_INT *ldu, float *vt,
                          const MKL_INT *ldvt, float *work,
                          const MKL_INT *lwork, MKL_INT *info) {
    FNAME(sgesvd)
    (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

template <typename FL>
inline void xsyev(const char *jobz, const char *uplo, const MKL_INT *n, FL *a,
                  const MKL_INT *lda, FL *w, FL *work, const MKL_INT *lwork,
                  MKL_INT *info);

template <>
inline void xsyev<double>(const char *jobz, const char *uplo, const MKL_INT *n,
                          double *a, const MKL_INT *lda, double *w,
                          double *work, const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dsyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
}

template <>
inline void xsyev<float>(const char *jobz, const char *uplo, const MKL_INT *n,
                         float *a, const MKL_INT *lda, float *w, float *work,
                         const MKL_INT *lwork, MKL_INT *info) {
    FNAME(ssyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
}

template <typename FL>
inline void xgels(const char *trans, const MKL_INT *m, const MKL_INT *n,
                  const MKL_INT *nrhs, FL *a, const MKL_INT *lda, FL *b,
                  const MKL_INT *ldb, FL *work, const MKL_INT *lwork,
                  MKL_INT *info);

template <>
inline void xgels<double>(const char *trans, const MKL_INT *m, const MKL_INT *n,
                          const MKL_INT *nrhs, double *a, const MKL_INT *lda,
                          double *b, const MKL_INT *ldb, double *work,
                          const MKL_INT *lwork, MKL_INT *info) {
    FNAME(dgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

template <>
inline void xgels<float>(const char *trans, const MKL_INT *m, const MKL_INT *n,
                         const MKL_INT *nrhs, float *a, const MKL_INT *lda,
                         float *b, const MKL_INT *ldb, float *work,
                         const MKL_INT *lwork, MKL_INT *info) {
    FNAME(sgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

template <typename FL>
inline void xheev(const char *jobz, const char *uplo, const MKL_INT *n, FL *a,
                  const MKL_INT *lda, typename GMatrix<FL>::FP *w, FL *work,
                  const MKL_INT *lwork, typename GMatrix<FL>::FP *rwork,
                  MKL_INT *info);

template <typename FL>
inline void xgeev(const char *jobvl, const char *jobvr, const MKL_INT *n, FL *a,
                  const MKL_INT *lda, FL *w, FL *vl, const MKL_INT *ldvl,
                  FL *vr, const MKL_INT *ldvr, FL *work, const MKL_INT *lwork,
                  typename GMatrix<FL>::FP *rwork, MKL_INT *info);

// w_imag will be in rwork
template <>
inline void xgeev<double>(const char *jobvl, const char *jobvr,
                          const MKL_INT *n, double *a, const MKL_INT *lda,
                          double *w, double *vl, const MKL_INT *ldvl,
                          double *vr, const MKL_INT *ldvr, double *work,
                          const MKL_INT *lwork, double *rwork, MKL_INT *info) {
    FNAME(dgeev)
    (jobvl, jobvr, n, a, lda, w, rwork, vl, ldvl, vr, ldvr, work, lwork, info);
}

// w_imag will be in rwork
template <>
inline void xgeev<float>(const char *jobvl, const char *jobvr, const MKL_INT *n,
                         float *a, const MKL_INT *lda, float *w, float *vl,
                         const MKL_INT *ldvl, float *vr, const MKL_INT *ldvr,
                         float *work, const MKL_INT *lwork, float *rwork,
                         MKL_INT *info) {
    FNAME(sgeev)
    (jobvl, jobvr, n, a, lda, w, rwork, vl, ldvl, vr, ldvr, work, lwork, info);
}

// General matrix operations
template <typename FL, typename = void> struct GMatrixFunctions;

// Dense matrix operations
template <typename FL>
struct GMatrixFunctions<
    FL, typename enable_if<is_floating_point<FL>::value>::type> {
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
    static void keep_real(const GMatrix<FL> &a) {}
    static void conjugate(const GMatrix<FL> &a) {}
    // a = a + scale * op(b)
    static void iadd(const GMatrix<FL> &a, const GMatrix<FL> &b, FL scale,
                     bool conj = false, FL cfactor = 1.0) {
        const FL one = 1.0;
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            MKL_INT n = a.m * a.n, inc = 1;
            if (cfactor == 1.0)
                xaxpy<FL>(&n, &scale, b.data, &inc, a.data, &inc);
            else
                xgemm<FL>("n", "n", &inc, &n, &inc, &scale, &one, &inc, b.data,
                          &inc, &cfactor, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n >= b.m);
            for (MKL_INT k = 0, inc = 1; k < b.n; k++)
                xgemm<FL>("t", "n", &b.m, &inc, &inc, &scale, &b(0, k), &b.n,
                          &one, &inc, &cfactor, &a(k, 0), &a.n);
        }
    }
    static FL norm(const GMatrix<FL> &a) {
        MKL_INT n = a.m * a.n, inc = 1;
        return xnrm2(&n, a.data, &inc);
    }
    // Computes norm more accurately
    static FL norm_accurate(const GMatrix<FL> &a) {
        MKL_INT n = a.m * a.n;
        typename GMatrix<FL>::FL out = 0.0;
        typename GMatrix<FL>::FL compensate = 0.0;
        for (MKL_INT ii = 0; ii < n; ++ii) {
            typename GMatrix<FL>::FL sumi = a.data[ii];
            sumi *= a.data[ii];
            // Kahan summation
            auto y = sumi - compensate;
            const volatile typename GMatrix<FL>::FL t = out + y;
            const volatile typename GMatrix<FL>::FL z = t - out;
            compensate = z - y;
            out = t;
        }
        out = sqrt(out);
        volatile typename GMatrix<FL>::FL outd = real(out);
        return static_cast<FL>(outd);
    }
    // determinant
    static FL det(const GMatrix<FL> &a) {
        assert(a.m == a.n);
        vector<FL> aa;
        vector<MKL_INT> ipiv;
        aa.reserve(a.m * a.n);
        ipiv.reserve(a.m);
        memcpy(aa.data(), a.data, sizeof(FL) * a.m * a.n);
        MKL_INT info = -1;
        xgetrf<FL>(&a.m, &a.n, aa.data(), &a.m, ipiv.data(), &info);
        assert(info == 0);
        FL det = 1.0;
        for (int i = 0; i < a.m; i++)
            det *= ipiv[i] != i + 1 ? -aa[i * a.m + i] : aa[i * a.m + i];
        return det;
    }
    // matrix inverse
    static void inverse(const GMatrix<FL> &a) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        assert(a.m == a.n);
        vector<MKL_INT> ipiv;
        ipiv.reserve(a.m);
        MKL_INT info = -1, lwork = 34 * a.m;
        xgetrf<FL>(&a.m, &a.n, a.data, &a.m, ipiv.data(), &info);
        assert(info == 0);
        FL *work = d_alloc->allocate(lwork);
        xgetri<FL>(&a.m, a.data, &a.m, ipiv.data(), work, &lwork, &info);
        assert(info == 0);
        d_alloc->deallocate(work, lwork);
    }
    static FL dot(const GMatrix<FL> &a, const GMatrix<FL> &b) {
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        return xdot<FL>(&n, a.data, &inc, b.data, &inc);
    }
    static FL complex_dot(const GMatrix<FL> &a, const GMatrix<FL> &b) {
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        return xdot<FL>(&n, a.data, &inc, b.data, &inc);
    }
    template <typename T1, typename T2>
    static bool all_close(const T1 &a, const T2 &b, FL atol = 1E-8,
                          FL rtol = 1E-5, FL scale = 1.0) {
        assert(a.m == b.m && a.n == b.n);
        for (MKL_INT i = 0; i < a.m; i++)
            for (MKL_INT j = 0; j < a.n; j++)
                if (abs(a(i, j) - scale * b(i, j)) > atol + rtol * abs(b(i, j)))
                    return false;
        return true;
    }
    // solve a^T x[i, :] = b[i, :] => output in b; a will be overwritten
    static void linear(const GMatrix<FL> &a, const GMatrix<FL> &b) {
        assert(a.m == a.n && a.m == b.n);
        MKL_INT *work = (MKL_INT *)ialloc->allocate(a.n * _MINTSZ), info = -1;
        xgesv<FL>(&a.m, &b.m, a.data, &a.n, work, b.data, &a.n, &info);
        assert(info == 0);
        ialloc->deallocate(work, a.n * _MINTSZ);
    }
    // least squares problem a x = b
    // return the residual (norm, not squared)
    // a.n is used as lda
    static FL least_squares(const GMatrix<FL> &a, const GMatrix<FL> &b,
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
    // c.n is used for ldc; a.n is used for lda
    static void multiply(const GMatrix<FL> &a, uint8_t conja,
                         const GMatrix<FL> &b, uint8_t conjb,
                         const GMatrix<FL> &c, FL scale, FL cfactor) {
        // if assertion fails here, check whether it is the case
        // where different bra and ket are used with the transpose rule
        // use no-transpose-rule to fix it
        if (!(conja & 1) && !(conjb & 1)) {
            assert(a.n >= b.m && c.m == a.m && c.n >= b.n);
            xgemm<FL>("n", "n", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                      &a.n, &cfactor, c.data, &c.n);
        } else if (!(conja & 1) && (conjb & 1)) {
            assert(a.n >= b.n && c.m == a.m && c.n >= b.m);
            xgemm<FL>("t", "n", &b.m, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                      &a.n, &cfactor, c.data, &c.n);
        } else if ((conja & 1) && !(conjb & 1)) {
            assert(a.m == b.m && c.m <= a.n && c.n >= b.n);
            xgemm<FL>("n", "t", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                      &a.n, &cfactor, c.data, &c.n);
        } else {
            assert(a.m == b.n && c.m <= a.n && c.n >= b.m);
            xgemm<FL>("t", "t", &b.m, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                      &a.n, &cfactor, c.data, &c.n);
        }
    }
    // c = bra(.T) * a * ket(.T)
    // return nflop
    // conj can be 0 (no conj no trans), 1 (trans), 2 (conj), 3 (conj trans)
    // for real numbers we just need (& 1) to exclude conj
    static size_t rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                         const GMatrix<FL> &bra, uint8_t conj_bra,
                         const GMatrix<FL> &ket, uint8_t conj_ket, FL scale) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        GMatrix<FL> work(nullptr, a.m, (conj_ket & 1) ? ket.m : ket.n);
        work.allocate(d_alloc);
        multiply(a, false, ket, conj_ket & 1, work, 1.0, 0.0);
        multiply(bra, conj_bra & 1, work, false, c, scale, 1.0);
        work.deallocate(d_alloc);
        return (size_t)ket.m * ket.n * work.m + (size_t)work.m * work.n * c.m;
    }
    // c(.T) = bra.T * a(.T) * ket
    // return nflop. (.T) is always transpose conjugate
    static size_t left_partial_rotate(const GMatrix<FL> &a, bool conj_a,
                         const GMatrix<FL> &c, bool conj_c,
                         const GMatrix<FL> &bra, const GMatrix<FL> &ket,
                         FL scale) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        GMatrix<FL> work(nullptr, conj_a ? a.n : a.m, ket.n);
        work.allocate(d_alloc);
        multiply(a, conj_a, ket, false, work, 1.0, 0.0);
        if (!conj_c)
            multiply(bra, true, work, false, c, scale, 1.0);
        else
            multiply(work, true, bra, false, c, scale, 1.0);
        work.deallocate(d_alloc);
        return (size_t)a.m * a.n * work.n + (size_t)work.m * work.n * bra.n;
    }
    // c(.T) = bra.c * a(.T) * ket.t = (a(~.T) * bra.t).T * ket.t
    // return nflop. (.T) is always transpose conjugate
    static size_t right_partial_rotate(const GMatrix<FL> &a, bool conj_a,
                         const GMatrix<FL> &c, bool conj_c,
                         const GMatrix<FL> &bra, const GMatrix<FL> &ket,
                         FL scale) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        GMatrix<FL> work(nullptr, conj_a ? a.m : a.n, bra.m);
        work.allocate(d_alloc);
        multiply(a, !conj_a, bra, true, work, 1.0, 0.0);
        if (!conj_c)
            multiply(work, true, ket, true, c, scale, 1.0);
        else
            multiply(ket, false, work, false, c, scale, 1.0);
        work.deallocate(d_alloc);
        return (size_t)a.m * a.n * work.n + (size_t)work.m * work.n * ket.m;
    }
    // dleft == true : c = bra (= da x db) * a * ket
    // dleft == false: c = bra * a * ket (= da x db)
    // return nflop.
    // conj means conj and trans / none for bra, trans / conj for ket
    static size_t three_rotate(const GMatrix<FL> &a, const GMatrix<FL> &c,
                               const GMatrix<FL> &bra, bool conj_bra,
                               const GMatrix<FL> &ket, bool conj_ket,
                               const GMatrix<FL> &da, bool dconja,
                               const GMatrix<FL> &db, bool dconjb, bool dleft,
                               FL scale, uint32_t stride) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            GMatrix<FL> work(nullptr, am, conj_ket ? ket.m : ket.n);
            work.allocate(d_alloc);
            // work = a * ket
            multiply(GMatrix<FL>(&a(ast, 0), am, a.n), false, ket, conj_ket,
                     work, 1.0, 0.0);
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * work
                multiply(db, dconjb, work, false,
                         GMatrix<FL>(&c(cst, 0), cm, c.n), scale * *da.data,
                         1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * work
                multiply(da, dconja, work, false,
                         GMatrix<FL>(&c(cst, 0), cm, c.n), scale * *db.data,
                         1.0);
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
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, db, dconjb,
                         work, *da.data * scale, 0.0);
            else if (db.m == 1 && db.n == 1)
                // work = a * (da x 1)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, da, dconja,
                         work, *db.data * scale, 0.0);
            else
                assert(false);
            // c = bra * work
            multiply(bra, conj_bra, work, false,
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
            multiply(GMatrix<FL>(&a(ast, 0), am, a.n), false, ket, conj_ket,
                     GMatrix<FL>(&c(cst, 0), cm, c.n), scale, 1.0);
            return (size_t)ket.m * ket.n * am;
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            if (da.m == 1 && da.n == 1)
                // c = a * (1 x db)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, db, dconjb,
                         GMatrix<FL>(&c(0, cst), c.m, c.n), *da.data * scale,
                         1.0);
            else if (db.m == 1 && db.n == 1)
                // c = a * (da x 1)
                multiply(GMatrix<FL>(&a(0, ast), a.m, a.n), false, da, dconja,
                         GMatrix<FL>(&c(0, cst), c.m, c.n), *db.data * scale,
                         1.0);
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
                multiply(db, dconjb, GMatrix<FL>(&a(ast, 0), am, a.n), false,
                         GMatrix<FL>(&c(cst, 0), cm, c.n), scale * *da.data,
                         1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * a
                multiply(da, dconja, GMatrix<FL>(&a(ast, 0), am, a.n), false,
                         GMatrix<FL>(&c(cst, 0), cm, c.n), scale * *db.data,
                         1.0);
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
            xgemm<FL>("n", conj_bra ? "t" : "n", &kn, &c.m, &a.m, &scale,
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
        xgemm<FL>("t", "n", &b.n, &a.n, &k, &scale, b.data, &ldb, a.data, &lda,
                  &cfactor, c.data, &c.n);
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
        if (da.m == 1 && da.n == 1) {
            scale *= *da.data;
            const MKL_INT dn = db.n - abs(ddstr);
            const FL *bdata =
                dconjb ? &db(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &db(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft)
                    // (1 x db) x b
                    xgemm<FL>("t", "n", &b.n, &dn, &k, &scale, b.data, &ldb,
                              bdata, &lddb, &cfactor,
                              &c(max(dstrn, dstrm), (MKL_INT)0), &c.n);
                else
                    // a x (1 x db)
                    xgemm<FL>("t", "n", &dn, &a.n, &k, &scale, bdata, &lddb,
                              a.data, &lda, &cfactor, &c(0, max(dstrn, dstrm)),
                              &c.n);
            }
        } else if (db.m == 1 && db.n == 1) {
            scale *= *db.data;
            const MKL_INT dn = da.n - abs(ddstr);
            const FL *adata =
                dconja ? &da(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &da(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (dn > 0) {
                if (dleft)
                    // (da x 1) x b
                    xgemm<FL>("t", "n", &b.n, &dn, &k, &scale, b.data, &ldb,
                              adata, &ldda, &cfactor,
                              &c(max(dstrn, dstrm), (MKL_INT)0), &c.n);
                else
                    // a x (da x 1)
                    xgemm<FL>("t", "n", &dn, &a.n, &k, &scale, adata, &ldda,
                              a.data, &lda, &cfactor, &c(0, max(dstrn, dstrm)),
                              &c.n);
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
                assert(a.m <= c.n);
                for (MKL_INT k = 0; k < a.n; k++)
                    xgemm<FL>("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
            } else {
                for (MKL_INT i = 0, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const FL factor = scale * a(j, i);
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
                    xgemm<FL>("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
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
                for (MKL_INT i = 0, incb = b.n, inc = 1; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++) {
                        const FL factor = scale * a(i, j);
                        for (MKL_INT k = 0; k < b.n; k++)
                            xaxpy<FL>(&b.m, &factor, &b(0, k), &incb,
                                      &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        case 1 | 2:
            if (a.m == 1 && a.n == 1) {
                for (MKL_INT k = 0; k < b.n; k++)
                    xgemm<FL>("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
            } else if (b.m == 1 && b.n == 1) {
                for (MKL_INT k = 0; k < a.n; k++)
                    xgemm<FL>("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
            } else {
                for (MKL_INT i = 0, incb = b.n, inc = 1; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++) {
                        const FL factor = scale * a(j, i);
                        for (MKL_INT k = 0; k < b.n; k++)
                            xaxpy<FL>(&b.m, &factor, &b(0, k), &incb,
                                      &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        default:
            assert(false);
        }
    }
    // SVD; original matrix will be destroyed
    static void svd(const GMatrix<FL> &a, const GMatrix<FL> &l,
                    const GMatrix<FL> &s, const GMatrix<FL> &r) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        MKL_INT k = min(a.m, a.n), info = 0, lwork = -1;
        FL twork;
        assert(a.m == l.m && a.n == r.n && l.n >= k && r.m == k && s.n == k);
        xgesvd<FL>("S", "S", &a.n, &a.m, a.data, &a.n, s.data, r.data, &a.n,
                   l.data, &l.n, &twork, &lwork, &info);
        assert(info == 0);
        lwork = (MKL_INT)twork;
        // FL work[lwork];
        FL *work = d_alloc->allocate(lwork);
        xgesvd<FL>("S", "S", &a.n, &a.m, a.data, &a.n, s.data, r.data, &a.n,
                   l.data, &l.n, work, &lwork, &info);
        assert(info == 0);
        d_alloc->deallocate(work, lwork);
    }
    // SVD for parallelism over sites; PRB 87, 155137 (2013)
    static void accurate_svd(const GMatrix<FL> &a, const GMatrix<FL> &l,
                             const GMatrix<FL> &s, const GMatrix<FL> &r,
                             FL eps = 1E-4) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        GMatrix<FL> aa(nullptr, a.m, a.n);
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
            GMatrix<FL> xa(nullptr, k - p, k - p), xl(nullptr, k - p, k - p),
                xr(nullptr, k - p, k - p);
            xa.data = d_alloc->allocate(xa.size());
            xl.data = d_alloc->allocate(xl.size());
            xr.data = d_alloc->allocate(xr.size());
            rotate(a, xa, GMatrix<FL>(l.data + p, l.m, l.n), true,
                   GMatrix<FL>(r.data + p * r.n, r.m - p, r.n), true, 1.0);
            accurate_svd(xa, xl, GMatrix<FL>(s.data + p, 1, k - p), xr, eps);
            GMatrix<FL> bl(nullptr, l.m, l.n), br(nullptr, r.m, r.n);
            bl.data = d_alloc->allocate(bl.size());
            br.data = d_alloc->allocate(br.size());
            copy(bl, l);
            copy(br, r);
            multiply(GMatrix<FL>(bl.data + p, bl.m, bl.n), false, xl, false,
                     GMatrix<FL>(l.data + p, l.m, l.n), 1.0, 0.0);
            multiply(xr, false, GMatrix<FL>(br.data + p * br.n, br.m - p, br.n),
                     false, GMatrix<FL>(r.data + p * r.n, r.m - p, r.n), 1.0,
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
    static void lq(const GMatrix<FL> &a, const GMatrix<FL> &l,
                   const GMatrix<FL> &q) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.m;
        // FL work[lwork], tau[k], t[a.m * a.n];
        FL *work = d_alloc->allocate(lwork);
        FL *tau = d_alloc->allocate(k);
        FL *t = d_alloc->allocate(a.m * a.n);
        assert(a.m == l.m && a.n == q.n && l.n == k && q.m == k);
        memcpy(t, a.data, sizeof(FL) * a.m * a.n);
        xgeqrf<FL>(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(l.data, 0, sizeof(FL) * k * a.m);
        for (MKL_INT j = 0; j < a.m; j++)
            memcpy(l.data + j * k, t + j * a.n, sizeof(FL) * min(j + 1, k));
        xungqr(&a.n, &k, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memcpy(q.data, t, sizeof(FL) * k * a.n);
        d_alloc->deallocate(t, a.m * a.n);
        d_alloc->deallocate(tau, k);
        d_alloc->deallocate(work, lwork);
    }
    // QR factorization
    static void qr(const GMatrix<FL> &a, const GMatrix<FL> &q,
                   const GMatrix<FL> &r) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        MKL_INT k = min(a.m, a.n), info, lwork = 34 * a.n;
        // FL work[lwork], tau[k], t[a.m * a.n];
        FL *work = d_alloc->allocate(lwork);
        FL *tau = d_alloc->allocate(k);
        FL *t = d_alloc->allocate(a.m * a.n);
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
        d_alloc->deallocate(t, a.m * a.n);
        d_alloc->deallocate(tau, k);
        d_alloc->deallocate(work, lwork);
    }
    // a += b.T
    static void transpose(const GMatrix<FL> &a, const GMatrix<FL> &b,
                          FL scale = 1.0, FL cfactor = 1.0) {
        iadd(a, b, scale, true, cfactor);
    }
    // diagonalization for each symmetry block
    static void block_eigs(const GMatrix<FL> &a, const GDiagonalMatrix<FL> &w,
                           const vector<uint8_t> &x) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        uint8_t maxx = *max_element(x.begin(), x.end()) + 1;
        vector<vector<MKL_INT>> mp(maxx);
        assert(a.m == a.n && w.n == a.n && (MKL_INT)x.size() == a.n);
        for (MKL_INT i = 0; i < a.n; i++)
            mp[x[i]].push_back(i);
        for (uint8_t i = 0; i < maxx; i++)
            if (mp[i].size() != 0) {
                FL *work = d_alloc->allocate(mp[i].size() * mp[i].size());
                FL *wwork = d_alloc->allocate(mp[i].size());
                for (size_t j = 0; j < mp[i].size(); j++)
                    for (size_t k = 0; k < mp[i].size(); k++)
                        work[j * mp[i].size() + k] = a(mp[i][j], mp[i][k]);
                eigs(GMatrix<FL>(work, (MKL_INT)mp[i].size(),
                                 (MKL_INT)mp[i].size()),
                     GDiagonalMatrix<FL>(wwork, (MKL_INT)mp[i].size()));
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
    static void eigs(const GMatrix<FL> &a, const GDiagonalMatrix<FL> &w) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        assert(a.m == a.n && w.n == a.n);
        MKL_INT lwork = -1, info;
        FL twork;
        xsyev<FL>("V", "U", &a.n, a.data, &a.n, w.data, &twork, &lwork, &info);
        assert(info == 0);
        lwork = (MKL_INT)twork;
        // FL work[lwork];
        FL *work = d_alloc->allocate(lwork);
        xsyev<FL>("V", "U", &a.n, a.data, &a.n, w.data, work, &lwork, &info);
        if (info != 0)
            cout << "ATTENTION: xsyev info = " << info << endl;
        // assert(info == 0);
        d_alloc->deallocate(work, lwork);
    }
    // eigenvectors for non-symmetric matrices
    // if any eigenvalue is complex, eigenvectors are stored in separate real
    // and imag part form
    static void eig(const GMatrix<FL> &a, const GDiagonalMatrix<FL> &wr,
                    const GDiagonalMatrix<FL> &wi, const GMatrix<FL> &lv) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        assert(a.m == a.n && wr.n == a.n && wi.n == a.n);
        MKL_INT lwork = -1, info;
        FL twork;
        xgeev<FL>("V", lv.data == nullptr ? "N" : "V", &a.n, a.data, &a.n,
                  wr.data, nullptr, &a.n, lv.data, &a.n, &twork, &lwork,
                  wi.data, &info);
        assert(info == 0);
        lwork = (MKL_INT)twork;
        FL *work = d_alloc->allocate(lwork);
        FL *vr = d_alloc->allocate(a.m * a.n);
        xgeev<FL>("V", lv.data == nullptr ? "N" : "V", &a.n, a.data, &a.n,
                  wr.data, vr, &a.n, lv.data, &a.n, work, &lwork, wi.data,
                  &info);
        assert(info == 0);
        uint8_t tag = 0;
        copy(a, GMatrix<FL>(vr, a.m, a.n));
        for (MKL_INT k = 0; k < a.m; k++)
            if (wi(k, k) != (FL)0.0) {
                k++;
                for (MKL_INT j = 0; j < a.n; j++)
                    a(k, j) = -a(k, j);
                if (lv.data != nullptr)
                    for (MKL_INT j = 0; j < a.n; j++)
                        lv(k, j) = -lv(k, j);
            }
        d_alloc->deallocate(vr, a.m * a.n);
        d_alloc->deallocate(work, lwork);
    }
};

typedef GMatrixFunctions<double> MatrixFunctions;

} // namespace block2
