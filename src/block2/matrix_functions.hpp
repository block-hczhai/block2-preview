
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

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void dscal(const int *n, const double *sa, double *sx,
                  const int *incx) noexcept;

// vector copy
// vector [dy] = [dx]
extern void dcopy(const int *n, const double *dx, const int *incx, double *dy,
                  const int *incy) noexcept;

// vector addition
// vector [sy] = vector [sy] + double [sa] * vector [sx]
extern void daxpy(const int *n, const double *sa, const double *sx,
                  const int *incx, double *sy, const int *incy) noexcept;

// vector dot product
extern double ddot(const int *n, const double *dx, const int *incx,
                   const double *dy, const int *incy) noexcept;

// Euclidean norm of a vector
extern double dnrm2(const int *n, const double *x, const int *incx) noexcept;

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void dgemm(const char *transa, const char *transb, const int *n,
                  const int *m, const int *k, const double *alpha,
                  const double *a, const int *lda, const double *b,
                  const int *ldb, const double *beta, double *c,
                  const int *ldc) noexcept;

// matrix-vector multiplication
// vec [y] = double [alpha] * mat [a] * vec [x] + double [beta] * vec [y]
extern void dgemv(const char *trans, const int *m, const int *n,
                  const double *alpha, const double *a, const int *lda,
                  const double *x, const int *incx, const double *beta,
                  double *y, const int *incy) noexcept;

// linear system a * x = b
extern void dgesv(const int *n, const int *nrhs, double *a, const int *lda,
                  int *ipiv, double *b, const int *ldb, int *info);

// QR factorization
extern void dgeqrf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void dorgqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);

// LQ factorization
extern void dgelqf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void dorglq(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);

// eigenvalue problem
extern void dsyev(const char *jobz, const char *uplo, const int *n, double *a,
                  const int *lda, double *w, double *work, const int *lwork,
                  int *info);
}

// Dense matrix operations
struct MatrixFunctions {
    // a = b
    static void copy(const MatrixRef &a, const MatrixRef &b, const int inca = 1,
                     const int incb = 1) {
        assert(a.m == b.m && a.n == b.n);
        const int n = a.m * a.n;
        dcopy(&n, b.data, &incb, a.data, &inca);
    }
    static void iscale(const MatrixRef &a, double scale) {
        int n = a.m * a.n, inc = 1;
        dscal(&n, &scale, a.data, &inc);
    }
    static void iadd(const MatrixRef &a, const MatrixRef &b, double scale,
                     bool conj = false) {
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            int n = a.m * a.n, inc = 1;
            daxpy(&n, &scale, b.data, &inc, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n == b.m);
            for (int i = 0, inc = 1; i < a.m; i++)
                daxpy(&a.n, &scale, b.data + i, &a.m, a.data, &inc);
        }
    }
    static double norm(const MatrixRef &a) {
        int n = a.m * a.n, inc = 1;
        return dnrm2(&n, a.data, &inc);
    }
    static double dot(const MatrixRef &a, const MatrixRef &b) {
        assert(a.m == b.m && a.n == b.n);
        int n = a.m * a.n, inc = 1;
        return ddot(&n, a.data, &inc, b.data, &inc);
    }
    template <typename T1, typename T2>
    static bool all_close(const T1 &a, const T2 &b, double atol = 1E-8,
                          double rtol = 1E-5, double scale = 1.0) {
        assert(a.m == b.m && a.n == b.n);
        for (int i = 0; i < a.m; i++)
            for (int j = 0; j < a.n; j++)
                if (abs(a(i, j) - scale * b(i, j)) > atol + rtol * abs(b(i, j)))
                    return false;
        return true;
    }
    static void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        if (!conja && !conjb) {
            assert(a.n == b.m && c.m == a.m && c.n == b.n);
            dgemm("n", "n", &c.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (!conja && conjb) {
            assert(a.n == b.n && c.m == a.m && c.n == b.m);
            dgemm("t", "n", &c.n, &c.m, &a.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (conja && !conjb) {
            assert(a.m == b.m && c.m == a.n && c.n == b.n);
            dgemm("n", "t", &c.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else {
            assert(a.m == b.n && c.m == a.n && c.n == b.m);
            dgemm("t", "t", &c.n, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        }
    }
    // c = bra * a * ket.T
    static void rotate(const MatrixRef &a, const MatrixRef &c,
                       const MatrixRef &bra, bool conj_bra,
                       const MatrixRef &ket, bool conj_ket, double scale) {
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate();
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate();
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                        const MatrixRef &c, double scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const int k = 1, lda = a.n + 1, ldb = b.n + 1;
        dgemm("t", "n", &b.n, &a.n, &k, &scale, b.data, &ldb, a.data, &lda,
              &cfactor, c.data, &c.n);
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
                    const int n = b.m * b.n;
                    dgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (int k = 0; k < b.m; k++)
                        dgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const int n = a.m * a.n;
                    dgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (int k = 0; k < a.m; k++)
                        dgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (int i = 0, inc = 1; i < a.m; i++)
                    for (int j = 0; j < a.n; j++) {
                        const double factor = scale * a(i, j);
                        for (int k = 0; k < b.m; k++)
                            daxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 1:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const int n = b.m * b.n;
                    dgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (int k = 0; k < b.m; k++)
                        dgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                assert(a.m <= c.n);
                for (int k = 0; k < a.n; k++)
                    dgemm("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (int i = 0, inc = 1; i < a.n; i++)
                    for (int j = 0; j < a.m; j++) {
                        const double factor = scale * a(j, i);
                        for (int k = 0; k < b.m; k++)
                            daxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 2:
            if (a.m == 1 && a.n == 1) {
                assert(b.m <= c.n);
                for (int k = 0; k < b.n; k++)
                    dgemm("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const int n = a.m * a.n;
                    dgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (int k = 0; k < a.m; k++)
                        dgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (int i = 0, incb = b.n, inc = 1; i < a.m; i++)
                    for (int j = 0; j < a.n; j++) {
                        const double factor = scale * a(i, j);
                        for (int k = 0; k < b.n; k++)
                            daxpy(&b.m, &factor, &b(0, k), &incb,
                                  &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        case 1 | 2:
            if (a.m == 1 && a.n == 1) {
                for (int k = 0; k < b.n; k++)
                    dgemm("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                for (int k = 0; k < a.n; k++)
                    dgemm("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (int i = 0, incb = b.n, inc = 1; i < a.n; i++)
                    for (int j = 0; j < a.m; j++) {
                        const double factor = scale * a(j, i);
                        for (int k = 0; k < b.n; k++)
                            daxpy(&b.m, &factor, &b(0, k), &incb,
                                  &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        default:
            assert(false);
        }
    }
    // LQ factorization
    static void lq(const MatrixRef &a, const MatrixRef &l, const MatrixRef &q) {
        int k = min(a.m, a.n), info, lwork = 34 * a.m;
        double work[lwork], tau[k], t[a.m * a.n];
        assert(a.m == l.m && a.n == q.n && l.n == k && q.m == k);
        memcpy(t, a.data, sizeof(t));
        dgeqrf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(l.data, 0, sizeof(double) * k * a.m);
        for (int j = 0; j < a.m; j++)
            memcpy(l.data + j * k, t + j * a.n, sizeof(double) * (j + 1));
        dorgqr(&a.n, &k, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memcpy(q.data, t, sizeof(double) * k * a.n);
    }
    // QR factorization
    static void qr(const MatrixRef &a, const MatrixRef &q, const MatrixRef &r) {
        int k = min(a.m, a.n), info, lwork = 34 * a.n;
        double work[lwork], tau[k], t[a.m * a.n];
        assert(a.m == q.m && a.n == r.n && q.n == k && r.m == k);
        memcpy(t, a.data, sizeof(t));
        dgelqf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(r.data, 0, sizeof(double) * k * a.n);
        for (int j = 0; j < k; j++)
            memcpy(r.data + j * a.n + j, t + j * a.n + j,
                   sizeof(double) * (a.n - j));
        dorglq(&k, &a.m, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        for (int j = 0; j < a.m; j++)
            memcpy(q.data + j * k, t + j * a.n, sizeof(double) * k);
    }
    // eigenvectors are row vectors
    static void eigs(const MatrixRef &a, const DiagonalMatrix &w) {
        assert(a.m == a.n && w.n == a.n);
        int lwork = 34 * a.n, info;
        double work[lwork];
        dsyev("V", "U", &a.n, a.data, &a.n, w.data, work, &lwork, &info);
        assert(info == 0);
    }
    static void olsen_precondition(const MatrixRef &q, const MatrixRef &c,
                                   double ld, const DiagonalMatrix &aa) {
        assert(aa.size() == c.size());
        MatrixRef t(nullptr, c.m, c.n);
        t.allocate();
        copy(t, c);
        for (int i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                t.data[i] /= ld - aa.data[i];
        iadd(q, c, -dot(t, q) / dot(c, t));
        for (int i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                q.data[i] /= ld - aa.data[i];
        t.deallocate();
    }
    // Davidson algorithm
    // aa: diag elements of a (for precondition)
    // bs: input/output vector
    template <typename MatMul>
    static vector<double>
    davidson(MatMul op, const DiagonalMatrix &aa, vector<MatrixRef> &bs,
             int &ndav, bool iprint = false, double conv_thrd = 5E-6,
             int max_iter = 5000, int deflation_min_size = 2,
             int deflation_max_size = 50) {
        int k = (int)bs.size();
        if (deflation_min_size < k)
            deflation_min_size = k;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++)
                iadd(bs[i], bs[j], -dot(bs[j], bs[i]));
            iscale(bs[i], 1.0 / sqrt(dot(bs[i], bs[i])));
        }
        vector<double> eigvals;
        vector<MatrixRef> sigmas;
        sigmas.reserve(k);
        for (int i = 0; i < k; i++) {
            sigmas.push_back(MatrixRef(nullptr, bs[i].m, bs[i].n));
            sigmas[i].allocate();
        }
        MatrixRef q(nullptr, bs[0].m, bs[0].n);
        q.allocate();
        q.clear();
        int l = k, ck = 0, msig = 0, m = k, xiter = 0;
        if (iprint)
            cout << endl;
        while (xiter < max_iter) {
            xiter++;
            for (int i = msig; i < m; i++, msig++) {
                sigmas[i].clear();
                op(bs[i], sigmas[i]);
            }
            DiagonalMatrix ld(nullptr, m);
            MatrixRef alpha(nullptr, m, m);
            ld.allocate();
            alpha.allocate();
            for (int i = 0; i < m; i++)
                for (int j = 0; j <= i; j++)
                    alpha(i, j) = dot(bs[i], sigmas[j]);
            eigs(alpha, ld);
            vector<MatrixRef> tmp(m, MatrixRef(nullptr, bs[0].m, bs[0].n));
            for (int i = 0; i < m; i++) {
                tmp[i].allocate();
                copy(tmp[i], bs[i]);
            }
            // note alpha row/column is diff from python
            // b[1:m] = np.dot(b[:], alpha[:, 1:m])
            for (int j = 0; j < m; j++)
                iscale(bs[j], alpha(j, j));
            for (int j = 0; j < m; j++)
                for (int i = 0; i < m; i++)
                    if (i != j)
                        iadd(bs[j], tmp[i], alpha(j, i));
            // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
            for (int j = 0; j < m; j++) {
                copy(tmp[j], sigmas[j]);
                iscale(sigmas[j], alpha(j, j));
            }
            for (int j = 0; j < m; j++)
                for (int i = 0; i < m; i++)
                    if (i != j)
                        iadd(sigmas[j], tmp[i], alpha(j, i));
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
            double qq = dot(q, q);
            if (iprint)
                cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                     << fixed << setw(15) << setprecision(8) << ld.data[ck]
                     << scientific << setw(13) << setprecision(2) << qq << endl;
            olsen_precondition(q, bs[ck], ld.data[ck], aa);
            eigvals.resize(ck + 1);
            if (ck + 1 != 0)
                memcpy(&eigvals[0], ld.data, (ck + 1) * sizeof(double));
            ld.deallocate();
            if (qq < conv_thrd) {
                ck++;
                if (ck == k)
                    break;
            } else {
                if (m >= deflation_max_size)
                    m = msig = deflation_min_size;
                for (int j = 0; j < m; j++)
                    iadd(q, bs[j], -dot(bs[j], q));
                iscale(q, 1.0 / sqrt(dot(q, q)));
                if (m >= (int)bs.size()) {
                    bs.push_back(MatrixRef(nullptr, bs[0].m, bs[0].n));
                    bs[m].allocate();
                    sigmas.push_back(MatrixRef(nullptr, bs[0].m, bs[0].n));
                    sigmas[m].allocate();
                    sigmas[m].clear();
                }
                copy(bs[m], q);
                m++;
            }
            if (xiter == max_iter) {
                cout << "Error : only " << ck << " converged!" << endl;
                assert(false);
            }
        }
        for (int i = (int)bs.size() - 1; i >= k; i--)
            sigmas[i].deallocate(), bs[i].deallocate();
        q.deallocate();
        for (int i = k - 1; i >= 0; i--)
            sigmas[i].deallocate();
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
    static pair<int, int> expo_pade(int ideg, int m, const double *h, int ldh,
                                    double t, double *work) {
        static const double zero = 0.0, one = 1.0, mone = -1.0, two = 2.0;
        static const int inc = 1;
        // check restrictions on input parameters
        int mm = m * m;
        int iflag = 0;
        assert(ldh >= m);
        // initialize pointers
        int icoef = 0, ih2 = icoef + (ideg + 1), ip = ih2 + mm, iq = ip + mm,
            ifree = iq + mm;
        // scaling: seek ns such that ||t*H/2^ns|| < 1/2;
        // and set scale = t/2^ns ...
        memset(work, 0, sizeof(double) * m);
        for (int j = 0; j < m; j++)
            for (int i = 0; i < m; i++)
                work[i] += abs(h[j * m + i]);
        double hnorm = 0.0;
        for (int i = 0; i < m; i++)
            hnorm = max(hnorm, work[i]);
        hnorm = abs(t * hnorm);
        if (hnorm == 0.0) {
            cerr << "Error - null H in expo pade" << endl;
            abort();
        }
        int ns = max(0, (int)(log(hnorm) / log(2.0)) + 2);
        double scale = t / (double)(1LL << ns);
        double scale2 = scale * scale;
        // compute Pade coefficients
        int i = ideg + 1, j = 2 * ideg + 1;
        work[icoef] = 1.0;
        for (int k = 1; k <= ideg; k++)
            work[icoef + k] =
                work[icoef + k - 1] * (double)(i - k) / double(k * (j - k));
        // H2 = scale2*H*H ...
        dgemm("n", "n", &m, &m, &m, &scale2, h, &ldh, h, &ldh, &zero,
              work + ih2, &m);
        // initialize p (numerator) and q (denominator)
        memset(work + ip, 0, sizeof(double) * mm * 2);
        double cp = work[icoef + ideg - 1];
        double cq = work[icoef + ideg];
        for (int j = 0; j < m; j++)
            work[ip + j * (m + 1)] = cp, work[iq + j * (m + 1)] = cq;
        // Apply Horner rule
        int iodd = 1;
        for (int k = ideg - 1; k > 0; k--) {
            int iused = iodd * iq + (1 - iodd) * ip;
            dgemm("n", "n", &m, &m, &m, &one, work + iused, &m, work + ih2, &m,
                  &zero, work + ifree, &m);
            for (int j = 0; j < m; j++)
                work[ifree + j * (m + 1)] += work[icoef + k - 1];
            ip = (1 - iodd) * ifree + iodd * ip;
            iq = iodd * ifree + (1 - iodd) * iq;
            ifree = iused;
            iodd = 1 - iodd;
        }
        // Obtain (+/-)(I + 2*(p\q))
        int *iqp = iodd ? &iq : &ip;
        dgemm("n", "n", &m, &m, &m, &scale, work + *iqp, &m, h, &ldh, &zero,
              work + ifree, &m);
        *iqp = ifree;
        daxpy(&mm, &mone, work + ip, &inc, work + iq, &inc);
        dgesv(&m, &m, work + iq, &m, (int *)work + ih2, work + ip, &m, &iflag);
        if (iflag != 0) {
            cerr << "Problem in DGESV in expo pade" << endl;
            abort();
        }
        dscal(&mm, &two, work + ip, &inc);
        for (int j = 0; j < m; j++)
            work[ip + j * (m + 1)]++;
        int iput = ip;
        if (ns == 0 && iodd) {
            dscal(&mm, &mone, work + ip, &inc);
        } else {
            // squaring : exp(t*H) = (exp(t*H))^(2^ns)
            iodd = 1;
            for (int k = 0; k < ns; k++) {
                int iget = iodd * ip + (1 - iodd) * iq;
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
    template <typename MatMul>
    static int expo_krylov(MatMul op, int n, int m, double t, double *v,
                           double *w, double &tol, double anorm, double *work,
                           int lwork, bool iprint) {
        const int inc = 1;
        const double sqr1 = sqrt(0.1), zero = 0.0;
        const int mxstep = 500, mxreject = 0, ideg = 6;
        const double delta = 1.2, gamma = 0.9;
        int iflag = 0;
        if (lwork < n * (m + 2) + 5 * (m + 2) * (m + 2) + ideg + 1)
            iflag = -1;
        if (m >= n || m <= 0)
            iflag = -3;
        if (iflag != 0) {
            cerr << "bad sizes (in input of expo krylov)" << endl;
            abort();
        }
        // initializations
        int k1 = 2, mh = m + 2, iv = 0, ih = iv + n * (m + 1) + n;
        int ifree = ih + mh * mh, lfree = lwork - ifree, iexph;
        int ibrkflag = 0, mbrkdwn = m, nmult = 0, mx;
        int nreject = 0, nexph = 0, nscale = 0, ns = 0;
        double t_out = abs(t), tbrkdwn = 0.0, t_now = 0.0, t_new = 0.0;
        double step_min = t_out, step_max = 0.0, s_error = 0.0, x_error = 0.0;
        double err_loc;
        int nstep = 0;
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
            for (int i = 0; i < n; i++)
                work[iv + i] = p1 * w[i];
            memset(work + ih, 0, sizeof(double) * mh * mh);
            // Lanczos loop
            int j1v = iv + n;
            for (int j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (j != 0) {
                    p1 = -work[ih + j * mh + j - 1];
                    daxpy(&n, &p1, work + j1v - n - n, &inc, work + j1v, &inc);
                }
                double hjj = -ddot(&n, work + j1v - n, &inc, work + j1v, &inc);
                daxpy(&n, &hjj, work + j1v - n, &inc, work + j1v, &inc);
                double hj1j = dnrm2(&n, work + j1v, &inc);
                work[ih + j * (mh + 1)] = -hjj;
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
                work[ih + j * mh + j + 1] = hj1j;
                work[ih + (j + 1) * mh + j] = hj1j;
                hj1j = 1.0 / hj1j;
                dscal(&n, &hj1j, work + j1v, &inc);
                j1v += n;
            }
            if (k1 != 0) {
                nmult++;
                op(work + j1v - n, work + j1v);
                avnorm = dnrm2(&n, work + j1v, &inc);
            }
            // set 1 for the 2-corrected scheme
            work[ih + m * mh + m - 1] = 0.0;
            work[ih + m * mh + m + 1] = 1.0;
            // loop while ireject<mxreject until the tolerance is reached
            for (int ireject = 0;;) {
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
                    t_step = gamma * t_step * pow(t_step * tol / err_loc, xm);
                    p1 = pow(10.0, round(log10(t_step) - sqr1) - 1);
                    t_step = floor(t_step / p1 + 0.55) * p1;
                    if (iprint)
                        cout << "t_step = " << t_old << " err_loc = " << err_loc
                             << " err_required = " << delta * t_old * tol
                             << endl
                             << "  stepsize rejected, stepping down to:"
                             << t_step << endl;
                    ireject++;
                    nreject++;
                    if (mxreject != 0 && ireject > mxreject) {
                        cerr << "failure in expo krylov: ---"
                             << " The requested tolerance is too high. Rerun "
                                "with a smaller value.";
                        abort();
                    }
                } else
                    break;
            }
            // now update w = beta*V*exp(t_step*H)*e1 and the hump
            mx = mbrkdwn + max(0, k1 - 1);
            dgemv("n", &n, &mx, &beta, work + iv, &n, work + iexph, &inc, &zero,
                  w, &inc);
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
                     << " step_size = " << t_step << " err_loc = " << err_loc
                     << " next_step = " << t_new << endl;
            step_min = min(step_min, t_step);
            step_max = max(step_max, t_step);
            s_error += err_loc;
            x_error = max(x_error, err_loc);
            if (mxstep != 0 && nstep >= mxstep) {
                iflag = 1;
                break;
            }
        }
        return nmult;
    }
    // apply exponential of a matrix to a vector
    // v: input/output vector
    template <typename MatMul>
    static int expo_apply(MatMul op, double t, double anorm, MatrixRef &v,
                          double consta = 0.0, bool iprint = false,
                          double conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        int vm = v.m, vn = v.n, n = vm * vn;
        if (n < 4) {
            const int lwork = 4 * n * n + 7;
            MatrixRef e = MatrixRef(dalloc->allocate(n), vm, vn);
            double *h = dalloc->allocate(n * n);
            double *work = dalloc->allocate(lwork);
            memset(e.data, 0, sizeof(double) * n);
            memset(h, 0, sizeof(double) * n * n);
            for (int i = 0; i < n; i++) {
                e.data[i] = 1.0;
                op(e, MatrixRef(h + i * n, vm, vn));
                h[i * (n + 1)] += consta;
                e.data[i] = 0.0;
            }
            int iptr = expo_pade(6, n, h, n, t, work).first;
            MatrixFunctions::multiply(MatrixRef(work + iptr, n, n), true, v,
                                      false, e, 1.0, 0.0);
            memcpy(v.data, e.data, sizeof(double) * n);
            dalloc->deallocate(work, lwork);
            dalloc->deallocate(h, n * n);
            e.deallocate();
            return n;
        }
        auto lop = [&op, consta, n, vm, vn](double *a, double *b) -> void {
            static int inc = 1;
            memset(b, 0, sizeof(double) * n);
            op(MatrixRef(a, vm, vn), MatrixRef(b, vm, vn));
            daxpy(&n, &consta, a, &inc, b, &inc);
        };
        int m = min(deflation_max_size, n - 1);
        int lwork = n * (m + 2) + 5 * (m + 2) * (m + 2) + 7;
        double *w = dalloc->allocate(n);
        double *work = dalloc->allocate(lwork);
        if (anorm < 1E-10)
            anorm = 1.0;
        int nmult = MatrixFunctions::expo_krylov(
            lop, n, m, t, v.data, w, conv_thrd, anorm, work, lwork, iprint);
        memcpy(v.data, w, sizeof(double) * n);
        dalloc->deallocate(work, lwork);
        dalloc->deallocate(w, n);
        return nmult;
    }
};

} // namespace block2
