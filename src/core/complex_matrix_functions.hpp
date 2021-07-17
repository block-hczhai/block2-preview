
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

#endif
}

// Dense complex number matrix operations
struct ComplexMatrixFunctions {
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
    // a = a + scale * op(b)
    static void iadd(const ComplexMatrixRef &a, const ComplexMatrixRef &b,
                     complex<double> scale, complex<double> cfactor = 1.0) {
        static const complex<double> x = 1.0;
        assert(a.m == b.m && a.n == b.n);
        MKL_INT n = a.m * a.n, inc = 1;
        if (cfactor == 1.0)
            zaxpy(&n, &scale, b.data, &inc, a.data, &inc);
        else
            zgemm("N", "N", &inc, &n, &inc, &scale, &x, &inc, b.data, &inc,
                  &cfactor, a.data, &inc);
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
        zgemm("C", "N", &inc, &inc, &n, &x, a.data, &n, b.data, &n, &zz, &r,
              &inc);
        return r;
    }
    static double norm(const ComplexMatrixRef &a) {
        MKL_INT n = a.m * a.n, inc = 1;
        return dznrm2(&n, a.data, &inc);
    }
    // eigenvectors are row right-vectors: A u(j) = lambda(j) u(j)
    static void eig(const ComplexMatrixRef &a, const ComplexDiagonalMatrix &w) {
        shared_ptr<VectorAllocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        assert(a.m == a.n && w.n == a.n);
        MKL_INT lwork = 34 * a.n, info;
        complex<double> *work = (complex<double> *)d_alloc->allocate(lwork * 2);
        double *rwork = d_alloc->allocate(a.m * 2);
        complex<double> *vl =
            (complex<double> *)d_alloc->allocate(a.m * a.n * 2);
        zgeev("V", "N", &a.n, a.data, &a.n, w.data, vl, &a.n, nullptr, &a.n,
              work, &lwork, rwork, &info);
        assert(info == 0);
        for (size_t k = 0; k < a.m * a.n; k++)
            a.data[k] = conj(vl[k]);
        d_alloc->deallocate((double *)vl, a.m * a.n * 2);
        d_alloc->deallocate(rwork, a.m * 2);
        d_alloc->deallocate((double *)work, lwork * 2);
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
    static void multiply(const ComplexMatrixRef &a, bool conja,
                         const ComplexMatrixRef &b, bool conjb,
                         const ComplexMatrixRef &c, complex<double> scale,
                         complex<double> cfactor) {
        // if assertion failes here, check whether it is the case
        // where different bra and ket are used with the transpose rule
        // use no-transpose-rule to fix it
        if (!conja && !conjb) {
            assert(a.n >= b.m && c.m == a.m && c.n >= b.n);
            zgemm("n", "n", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (!conja && conjb) {
            assert(a.n >= b.n && c.m == a.m && c.n >= b.m);
            zgemm("t", "n", &b.m, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (conja && !conjb) {
            assert(a.m == b.m && c.m <= a.n && c.n >= b.n);
            zgemm("n", "t", &b.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else {
            assert(a.m == b.n && c.m <= a.n && c.n >= b.m);
            zgemm("t", "t", &b.m, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
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
        MKL_INT nmult = ComplexMatrixFunctions::expo_krylov(
            lop, n, m, abst, v.data, w.data(), conv_thrd, anorm, work.data(),
            lwork, iprint, (PComm)pcomm);
        memcpy(v.data, w.data(), sizeof(complex<double>) * w.size());
        return (int)nmult;
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
    // GCROT(m, k) method for solving x in linear equation H x = b
    template <typename MatMul, typename PComm>
    static complex<double>
    gcrotmk(MatMul &op, const ComplexDiagonalMatrix &aa, ComplexMatrixRef x,
            ComplexMatrixRef b, int &nmult, int &niter, int m = 20, int k = -1,
            bool iprint = false, const PComm &pcomm = nullptr,
            double conv_thrd = 5E-6, int max_iter = 5000,
            int soft_max_iter = -1) {
        ComplexMatrixRef r(nullptr, x.m, x.n), w(nullptr, x.m, x.n);
        double ff[4];
        double &beta = ff[0], &rr = ff[1];
        complex<double> &func = (complex<double> &)ff[2];
        r.allocate();
        w.allocate();
        r.clear();
        op(x, r);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            func = complex_dot(x, b);
            beta = norm(r);
        }
        if (pcomm != nullptr)
            pcomm->broadcast(&beta, 4, pcomm->root);
        if (iprint)
            cout << endl;
        if (k == -1)
            k = m;
        int xiter = 0, jiter = 1, nn = k + m + 2;
        vector<ComplexMatrixRef> cvs(nn, ComplexMatrixRef(nullptr, x.m, x.n));
        vector<ComplexMatrixRef> uzs(nn, ComplexMatrixRef(nullptr, x.m, x.n));
        vector<complex<double>> pcus;
        pcus.reserve(x.size() * 2 * nn);
        for (int i = 0; i < nn; i++) {
            cvs[i].data = pcus.data() + cvs[i].size() * i;
            uzs[i].data = pcus.data() + uzs[i].size() * (i + nn);
        }
        int ncs = 0, icu = 0;
        ComplexMatrixRef bmat(nullptr, k, k + m);
        ComplexMatrixRef hmat(nullptr, k + m + 1, k + m);
        ComplexMatrixRef ys(nullptr, k + m, 1);
        ComplexMatrixRef bys(nullptr, k, 1);
        ComplexMatrixRef hys(nullptr, k + m + 1, 1);
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
                     << setw(15) << setprecision(8) << real(func) << "+"
                     << setw(15) << setprecision(8) << imag(func) << "i"
                     << scientific << setw(13) << setprecision(2) << beta * beta
                     << endl;
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
                ComplexMatrixRef z(uzs[(ivz + j) % nn].data, x.m, x.n);
                if (pcomm == nullptr || pcomm->root == pcomm->rank)
                    cg_precondition(z, cvs[(ivz + j) % nn], aa);
                if (pcomm != nullptr)
                    pcomm->broadcast(z.data, z.size(), pcomm->root);
                w.clear();
                op(z, w);
                nz = j + 1;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    for (int i = 0; i < ncs; i++) {
                        bmat(i, j) = complex_dot(cvs[(icu + i) % nn], w);
                        iadd(w, cvs[(icu + i) % nn], -bmat(i, j));
                    }
                    for (int i = 0; i < nz; i++) {
                        hmat(i, j) = complex_dot(cvs[(ivz + i) % nn], w);
                        iadd(w, cvs[(ivz + i) % nn], -hmat(i, j));
                    }
                    hmat(j + 1, j) = norm(w);
                    iadd(cvs[(ivz + nz) % nn], w, 1.0 / hmat(j + 1, j), 0.0);
                    rr = least_squares(
                        ComplexMatrixRef(hmat.data, j + 2, hmat.n),
                        ComplexMatrixRef(hys.data, j + 2, 1),
                        ComplexMatrixRef(ys.data, j + 1, 1));
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(&rr, 1, pcomm->root);
                if (rr * rr < conv_thrd)
                    break;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                multiply(ComplexMatrixRef(bmat.data, ncs, bmat.n), false,
                         ComplexMatrixRef(ys.data, nz, 1), false,
                         ComplexMatrixRef(bys.data, ncs, 1), 1.0, 0.0);
                multiply(ComplexMatrixRef(hmat.data, nz + 1, hmat.n), false,
                         ComplexMatrixRef(ys.data, nz, 1), false,
                         ComplexMatrixRef(hys.data, nz + 1, 1), 1.0, 0.0);
                for (int i = 0; i < nz; i++)
                    iadd(uzs[(icu + ncs) % nn], uzs[(ivz + i) % nn], ys(i, 0),
                         !!i);
                for (int i = 0; i < ncs; i++)
                    iadd(uzs[(icu + ncs) % nn], uzs[(icu + i) % nn],
                         -bys(i, 0));
                for (int i = 0; i < nz + 1; i++)
                    iadd(cvs[(icu + ncs) % nn], cvs[(ivz + i) % nn], hys(i, 0),
                         !!i);
                double alpha = norm(cvs[(icu + ncs) % nn]);
                iscale(cvs[(icu + ncs) % nn], 1 / alpha);
                iscale(uzs[(icu + ncs) % nn], 1 / alpha);
                complex<double> gamma = complex_dot(cvs[(icu + ncs) % nn], r);
                iadd(r, cvs[(icu + ncs) % nn], -gamma);
                iadd(x, uzs[(icu + ncs) % nn], gamma);
                func = complex_dot(x, b);
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
