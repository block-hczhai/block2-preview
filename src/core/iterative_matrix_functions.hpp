
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

#include "complex_matrix_functions.hpp"
#include "matrix_functions.hpp"
#include "threading.hpp"
#include <algorithm>

using namespace std;

namespace block2 {

template <typename FL> struct IterativeMatrixFunctions : GMatrixFunctions<FL> {
    using GMatrixFunctions<FL>::copy;
    using GMatrixFunctions<FL>::iadd;
    using GMatrixFunctions<FL>::complex_dot;
    using GMatrixFunctions<FL>::dot;
    using GMatrixFunctions<FL>::norm;
    using GMatrixFunctions<FL>::norm_accurate;
    using GMatrixFunctions<FL>::iscale;
    using GMatrixFunctions<FL>::eigs;
    using GMatrixFunctions<FL>::linear;
    using GMatrixFunctions<FL>::multiply;
    using GMatrixFunctions<FL>::inverse;
    using GMatrixFunctions<FL>::least_squares;
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FC FC;
    static const int cpx_sz = sizeof(FL) / sizeof(FP);
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
    // E.R. Davidson. "The iterative calculation of a few of the lowest
    // eigenvalues and corresponding eigenvectors of large real-symmetric
    // matrices." Journal of Computational Physics 17 (1975): 87-94.
    // Section III. D
    static void davidson_precondition(const GMatrix<FL> &q, FP ld,
                                      const GDiagonalMatrix<FL> &aa) {
        assert(aa.size() == q.size());
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                q.data[i] /= ld - aa.data[i];
    }
    // q = Kinv q - (Kinv c, q) / (c, Kinv c) Kinv c
    static void olsen_precondition_old(const GMatrix<FL> &q,
                                       const GMatrix<FL> &c, FP ld,
                                       const GDiagonalMatrix<FL> &aa) {
        assert(aa.size() == c.size());
        GMatrix<FL> t(nullptr, c.m, c.n);
        t.allocate();
        copy(t, c);
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                t.data[i] /= ld - aa.data[i];
        iadd(q, c, -complex_dot(t, q) / complex_dot(c, t));
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                q.data[i] /= ld - aa.data[i];
        t.deallocate();
    }
    // q = Kinv q - (c, Kinv q) / (c, Kinv c) Kinv c
    static void olsen_precondition(const GMatrix<FL> &q, const GMatrix<FL> &c,
                                   FP ld, const GDiagonalMatrix<FL> &aa) {
        assert(aa.size() == c.size());
        GMatrix<FL> t(nullptr, c.m, c.n);
        t.allocate();
        copy(t, c);
        for (MKL_INT i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12) {
                t.data[i] /= ld - aa.data[i];
                q.data[i] /= ld - aa.data[i];
            }
        iadd(q, t, -complex_dot(c, q) / complex_dot(c, t));
        t.deallocate();
    }
    // Davidson algorithm
    // E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    // aa: diag elements of a (for precondition)
    // bs: input/output vector
    // ors: orthogonal states to be projected out
    template <typename MatMul, typename PComm>
    static vector<FP>
    davidson(MatMul &op, const GDiagonalMatrix<FL> &aa, vector<GMatrix<FL>> &vs,
             FP shift, DavidsonTypes davidson_type, int &ndav,
             bool iprint = false, const PComm &pcomm = nullptr,
             FP conv_thrd = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
             int deflation_min_size = 2, int deflation_max_size = 50,
             const vector<GMatrix<FL>> &ors = vector<GMatrix<FL>>(),
             const vector<FP> &proj_weights = vector<FP>()) {
        assert(!(davidson_type & DavidsonTypes::Harmonic));
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        int k = (int)vs.size(), nor = (int)ors.size(), nwg = 0;
        // if proj_weights is empty, then projection is done by (1 - |v><v|)
        // if proj_weights is not empty, projection is done by change H to (H +
        // w |v><v|)
        if (proj_weights.size() != 0) {
            assert(proj_weights.size() == ors.size());
            nwg = (int)ors.size(), nor = 0;
        }
        if (deflation_min_size < k)
            deflation_min_size = k;
        if (deflation_max_size < k + k / 2)
            deflation_max_size = k + k / 2;
        GMatrix<FL> pbs(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()),
                        1);
        GMatrix<FL> pss(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()),
                        1);
        pbs.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        pss.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        vector<GMatrix<FL>> bs(deflation_max_size,
                               GMatrix<FL>(nullptr, vs[0].m, vs[0].n));
        vector<GMatrix<FL>> sigmas(deflation_max_size,
                                   GMatrix<FL>(nullptr, vs[0].m, vs[0].n));
        vector<FL> or_normsqs(nor);
        for (int i = 0; i < nor; i++) {
            for (int j = 0; j < i; j++)
                if (abs(or_normsqs[j]) > 1E-14)
                    iadd(ors[i], ors[j],
                         -complex_dot(ors[j], ors[i]) / or_normsqs[j]);
            or_normsqs[i] = complex_dot(ors[i], ors[i]);
        }
        for (int i = 0; i < deflation_max_size; i++) {
            bs[i].data = pbs.data + bs[i].size() * i;
            sigmas[i].data = pss.data + sigmas[i].size() * i;
        }
        for (int i = 0; i < k; i++)
            copy(bs[i], vs[i]);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++)
                iadd(bs[i], bs[j], -complex_dot(bs[j], bs[i]));
            iscale(bs[i], 1.0 / norm(bs[i]));
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < nor; j++)
                if (abs(or_normsqs[j]) > 1E-14)
                    iadd(bs[i], ors[j],
                         -complex_dot(ors[j], bs[i]) / or_normsqs[j]);
            FL normx = norm(bs[i]);
            if (abs(normx * normx) < 1E-14) {
                cout << "Cannot generate initial guess " << i
                     << " for Davidson unitary to all given states!" << endl;
                assert(false);
            }
            iscale(bs[i], 1.0 / normx);
        }
        vector<FP> eigvals(k);
        vector<int> eigval_idxs(deflation_max_size);
        GMatrix<FL> q(nullptr, bs[0].m, bs[0].n);
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.allocate();
        int ck = 0, msig = 0, m = k, xiter = 0;
        FL qq;
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
                for (int j = 0; j < nwg; j++)
                    iadd(sigmas[i], ors[j],
                         complex_dot(ors[j], bs[i]) * proj_weights[j]);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                GDiagonalMatrix<FP> ld(nullptr, m);
                GMatrix<FL> alpha(nullptr, m, m);
                ld.allocate();
                alpha.allocate();
                vector<GMatrix<FL>> tmp(m,
                                        GMatrix<FL>(nullptr, bs[0].m, bs[0].n));
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
                            alpha(i, j) = complex_dot(sigmas[i], bs[j]);
                    }
#pragma omp single
                    eigs(alpha, ld);
                    // note alpha row/column is diff from python
#pragma omp for schedule(static)
                    // b[1:m] = np.dot(b[:], alpha[:, 1:m])
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], bs[j]);
                        iscale(bs[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(bs[j], tmp[i], alpha(j, i));
#pragma omp for schedule(static)
                    // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
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
                                 return ld.data[i] - shift > ld.data[j] - shift;
                         });
                else if (davidson_type & DavidsonTypes::GreaterThan)
                    sort(eigval_idxs.begin(), eigval_idxs.begin() + m,
                         [&ld, shift](int i, int j) {
                             if ((shift > ld.data[i]) != (shift > ld.data[j]))
                                 return shift > ld.data[j];
                             else if (shift > ld.data[i])
                                 return shift - ld.data[i] > shift - ld.data[j];
                             else
                                 return ld.data[i] - shift < ld.data[j] - shift;
                         });
                for (int i = 0; i < ck; i++) {
                    int ii = eigval_idxs[i];
                    copy(q, sigmas[ii]);
                    iadd(q, bs[ii], -ld(ii, ii));
                    if (abs(complex_dot(q, q)) >= conv_thrd) {
                        ck = i;
                        break;
                    }
                }
                int ick = eigval_idxs[ck];
                copy(q, sigmas[ick]);
                iadd(q, bs[ick], -ld(ick, ick));
                for (int j = 0; j < nor; j++)
                    if (abs(or_normsqs[j]) > 1E-14)
                        iadd(q, ors[j],
                             -complex_dot(ors[j], q) / or_normsqs[j]);
                qq = complex_dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8) << ld.data[ick]
                         << scientific << setw(13) << setprecision(2) << abs(qq)
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
            if (abs(qq) < conv_thrd) {
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
                        vector<GMatrix<FL>> tmp(
                            m, GMatrix<FL>(nullptr, bs[0].m, bs[0].n));
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
                        iadd(q, bs[j], -complex_dot(bs[j], q));
                    for (int j = 0; j < nor; j++)
                        if (abs(or_normsqs[j]) > 1E-14)
                            iadd(q, ors[j],
                                 -complex_dot(ors[j], q) / or_normsqs[j]);
                    iscale(q, 1.0 / norm(q));
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
    static vector<FP> harmonic_davidson(
        MatMul &op, const GDiagonalMatrix<FL> &aa, vector<GMatrix<FL>> &vs,
        FP shift, DavidsonTypes davidson_type, int &ndav, bool iprint = false,
        const PComm &pcomm = nullptr, FP conv_thrd = 5E-6, int max_iter = 5000,
        int soft_max_iter = -1, int deflation_min_size = 2,
        int deflation_max_size = 50,
        const vector<GMatrix<FL>> &ors = vector<GMatrix<FL>>(),
        const vector<FP> &proj_weights = vector<FP>()) {
        if (!(davidson_type & DavidsonTypes::Harmonic))
            return davidson(op, aa, vs, shift, davidson_type, ndav, iprint,
                            pcomm, conv_thrd, max_iter, soft_max_iter,
                            deflation_min_size, deflation_max_size, ors,
                            proj_weights);
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        int k = (int)vs.size(), nor = (int)ors.size(), nwg = 0;
        // if proj_weights is empty, then projection is done by (1 - |v><v|)
        // if proj_weights is not empty, projection is done by change H to (H +
        // w |v><v|)
        if (proj_weights.size() != 0) {
            assert(proj_weights.size() == ors.size());
            nwg = (int)ors.size(), nor = 0;
        }
        if (deflation_min_size < k)
            deflation_min_size = k;
        if (deflation_max_size < k + k / 2)
            deflation_max_size = k + k / 2;
        GMatrix<FL> pbs(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()),
                        1);
        GMatrix<FL> pss(nullptr, (MKL_INT)(deflation_max_size * vs[0].size()),
                        1);
        pbs.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        pss.data = d_alloc->allocate(deflation_max_size * vs[0].size());
        vector<GMatrix<FL>> bs(deflation_max_size,
                               GMatrix<FL>(nullptr, vs[0].m, vs[0].n));
        vector<GMatrix<FL>> sigmas(deflation_max_size,
                                   GMatrix<FL>(nullptr, vs[0].m, vs[0].n));
        vector<FL> or_normsqs(nor);
        for (int i = 0; i < nor; i++) {
            for (int j = 0; j < i; j++)
                if (abs(or_normsqs[j]) > 1E-14)
                    iadd(ors[i], ors[j],
                         -complex_dot(ors[j], ors[i]) / or_normsqs[j]);
            or_normsqs[i] = complex_dot(ors[i], ors[i]);
        }
        for (int i = 0; i < deflation_max_size; i++) {
            bs[i].data = pbs.data + bs[i].size() * i;
            sigmas[i].data = pss.data + sigmas[i].size() * i;
        }
        for (int i = 0; i < k; i++)
            copy(bs[i], vs[i]);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < nor; j++)
                if (abs(or_normsqs[j]) > 1E-14)
                    iadd(bs[i], ors[j],
                         -complex_dot(ors[j], bs[i]) / or_normsqs[j]);
            FL normsq = complex_dot(bs[i], bs[i]);
            if (abs(normsq) < 1E-14) {
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
            for (int j = 0; j < nwg; j++)
                iadd(sigmas[i], ors[j],
                     complex_dot(ors[j], bs[i]) * proj_weights[j]);
            if (shift != 0.0)
                iadd(sigmas[i], bs[i], -shift);
            num_matmul++;
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++) {
                iadd(bs[i], bs[j], -complex_dot(sigmas[j], sigmas[i]));
                iadd(sigmas[i], sigmas[j], -complex_dot(sigmas[j], sigmas[i]));
            }
            iscale(bs[i], 1.0 / sqrt(complex_dot(sigmas[i], sigmas[i])));
            iscale(sigmas[i], 1.0 / sqrt(complex_dot(sigmas[i], sigmas[i])));
        }
        vector<FP> eigvals(k);
        vector<int> eigval_idxs(deflation_max_size);
        GMatrix<FL> q(nullptr, bs[0].m, bs[0].n);
        if (pcomm == nullptr || pcomm->root == pcomm->rank)
            q.allocate();
        int ck = 0, m = k, xiter = 0;
        FL qq;
        if (iprint)
            cout << endl;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                GDiagonalMatrix<FP> ld(nullptr, m);
                GMatrix<FL> alpha(nullptr, m, m);
                ld.allocate();
                alpha.allocate();
                vector<GMatrix<FL>> tmp(m,
                                        GMatrix<FL>(nullptr, bs[0].m, bs[0].n));
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
                            alpha(i, j) = complex_dot(sigmas[i], bs[j]);
                    }
#pragma omp single
                    eigs(alpha, ld);
                    // note alpha row/column is diff from python
#pragma omp for schedule(static)
                    // b[1:m] = np.dot(b[:], alpha[:, 1:m])
                    for (int j = 0; j < m; j++) {
                        copy(tmp[j], bs[j]);
                        iscale(bs[j], alpha(j, j));
                    }
#pragma omp for schedule(static)
                    for (int j = 0; j < m; j++)
                        for (int i = 0; i < m; i++)
                            if (i != j)
                                iadd(bs[j], tmp[i], alpha(j, i));
#pragma omp for schedule(static)
                    // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
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
                        ld(j, j) = xreal(complex_dot(bs[j], sigmas[j]) /
                                         complex_dot(bs[j], bs[j]));
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
                    if (abs(complex_dot(q, q)) >= conv_thrd) {
                        ck = i;
                        break;
                    }
                }
                int ick = eigval_idxs[ck];
                copy(q, sigmas[ick]);
                iadd(q, bs[ick], -ld(ick, ick));
                for (int j = 0; j < nor; j++)
                    if (abs(or_normsqs[j]) > 1E-14)
                        iadd(q, ors[j],
                             -complex_dot(ors[j], q) / or_normsqs[j]);
                qq = complex_dot(q, q);
                if (iprint)
                    cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                         << fixed << setw(15) << setprecision(8)
                         << ld.data[ick] + shift << scientific << setw(13)
                         << setprecision(2) << abs(qq) << endl;
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
            if (abs(qq) < conv_thrd) {
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
                for (int j = 0; j < nwg; j++)
                    iadd(sigmas[m], ors[j],
                         complex_dot(ors[j], bs[m]) * proj_weights[j]);
                if (shift != 0.0)
                    iadd(sigmas[m], bs[m], -shift);
                num_matmul++;
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (do_deflation) {
                        vector<GMatrix<FL>> tmp(
                            m, GMatrix<FL>(nullptr, bs[0].m, bs[0].n));
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
                            iadd(bs[i], bs[j],
                                 -complex_dot(sigmas[j], sigmas[i]));
                            iadd(sigmas[i], sigmas[j],
                                 -complex_dot(sigmas[j], sigmas[i]));
                        }
                        iscale(bs[i],
                               1.0 / sqrt(complex_dot(sigmas[i], sigmas[i])));
                        iscale(sigmas[i],
                               1.0 / sqrt(complex_dot(sigmas[i], sigmas[i])));
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
    // Computes exp(t*H), the matrix exponential of a general complex
    // matrix in full, using the irreducible rational Pade approximation
    // Adapted from expokit fortran code dgpadm.f/zgpadm.f:
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
            hnorm = max(hnorm, xreal(work[i]));
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
                work[icoef + k - 1] * (FP)(i - k) / (FP)(k * (j - k));
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
        linear(GMatrix<FL>(work + iq, m, m), GMatrix<FL>(work + ip, m, m));
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
    // Computes w = exp(t*A)*v - for a (sparse) symmetric / hermitian / general
    // matrix A. Adapted from expokit fortran code dsexpv.f/dgexpy.f/zgexpv.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = n*(m+1)+n+(m+2)^2+4*(m+2)^2+ideg+1
    template <typename MatMul, typename PComm>
    static MKL_INT expo_krylov(MatMul &op, MKL_INT n, MKL_INT m, FP t, FL *v,
                               FL *w, FP &tol, FP anorm, FL *work,
                               MKL_INT lwork, bool symmetric, bool iprint,
                               const PComm &pcomm = nullptr) {
        const MKL_INT inc = 1;
        const FP sqr1 = sqrt(0.1);
        const FL zero = 0.0;
        const MKL_INT mxstep = symmetric ? 500 : 1000, mxreject = 0, ideg = 6;
        const FP delta = 1.2, gamma = 0.9;
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
        FP t_out = abs(t), tbrkdwn = 0.0, t_now = 0.0, t_new = 0.0;
        FP step_min = t_out, step_max = 0.0, s_error = 0.0, x_error = 0.0;
        FP err_loc;
        MKL_INT nstep = 0;
        // machine precision
        FP eps = 0.0;
        for (FP p1 = (FP)4.0 / (FP)3.0, p2, p3; eps == (FP)0.0;)
            p2 = p1 - (FP)1.0, p3 = p2 + p2 + p2, eps = abs(p3 - (FP)1.0);
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
            // Lanczos loop / Arnoldi loop
            MKL_INT j1v = iv + n;
            FP hj1j = 0.0;
            for (MKL_INT j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    if (symmetric) {
                        if (j != 0) {
                            FL pp = -work[ih + j * mh + j - 1];
                            xaxpy<FL>(&n, &pp, work + j1v - n - n, &inc,
                                      work + j1v, &inc);
                        }
                        FL hjj = -complex_dot(GMatrix<FL>(work + j1v - n, n, 1),
                                              GMatrix<FL>(work + j1v, n, 1));
                        work[ih + j * (mh + 1)] = -hjj;
                        xaxpy<FL>(&n, &hjj, work + j1v - n, &inc, work + j1v,
                                  &inc);
                        hj1j = xnrm2<FL>(&n, work + j1v, &inc);
                    } else {
                        for (MKL_INT i = 0; i <= j; i++) {
                            hij = -complex_dot(
                                GMatrix<FL>(work + iv + i * n, n, 1),
                                GMatrix<FL>(work + j1v, n, 1));
                            xaxpy<FL>(&n, &hij, work + iv + i * n, &inc,
                                      work + j1v, &inc);
                            work[ih + j * mh + i] = -hij;
                        }
                        hj1j = xnrm2<FL>(&n, work + j1v, &inc);
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
                    work[ih + j * mh + j + 1] = (FL)hj1j;
                    if (symmetric)
                        work[ih + (j + 1) * mh + j] = (FL)hj1j;
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
                FL hjj = (FL)beta;
                xgemv<FL>("n", &n, &mx, &hjj, work + iv, &n, work + iexph, &inc,
                          &zero, w, &inc);
                beta = xnrm2<FL>(&n, w, &inc);
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
    // apply exponential of a matrix to a vector
    // v: input/output vector
    template <typename MatMul, typename PComm>
    static int expo_apply(MatMul &op, FL t, FP anorm, GMatrix<FL> &v, FL consta,
                          bool symmetric, bool iprint = false,
                          const PComm &pcomm = nullptr, FP conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        MKL_INT vm = v.m, vn = v.n, n = vm * vn;
        FP abst = abs(t);
        assert(abst != 0);
        FL tt = t / abst;
        if (n < 4) {
            const MKL_INT lwork = 4 * n * n + 7;
            vector<FL> te(n), to(n), h(n * n), work(lwork);
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
            static const MKL_INT inc = 1;
            static const FL x = 1.0;
            memset(b, 0, sizeof(FL) * n);
            op(GMatrix<FL>(a, vm, vn), GMatrix<FL>(b, vm, vn));
            const FL cconsta = consta * tt;
            xgemm<FL>("n", "n", &inc, &n, &inc, &x, &cconsta, &inc, a, &inc,
                      &tt, b, &inc);
        };
        MKL_INT m = min((MKL_INT)deflation_max_size, n - 1);
        MKL_INT lwork = n * (m + 2) + 5 * (m + 2) * (m + 2) + 7;
        vector<FL> w(n), work(lwork);
        anorm = (anorm + abs(consta) * n) * abs(tt);
        if (anorm < 1E-10)
            anorm = 1.0;
        MKL_INT nmult =
            expo_krylov(lop, n, m, abst, v.data, w.data(), conv_thrd, anorm,
                        work.data(), lwork, symmetric, iprint, (PComm)pcomm);
        memcpy(v.data, w.data(), sizeof(FL) * n);
        return (int)nmult;
    }
    // Solve x in linear equation H x = b
    // by applying linear CG method
    // where H is Hermitian and positive-definite
    // H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static FL conjugate_gradient(MatMul &op, const GDiagonalMatrix<FL> &aa,
                                 GMatrix<FL> x, GMatrix<FL> b, int &nmult,
                                 FL consta = 0.0, bool iprint = false,
                                 const PComm &pcomm = nullptr,
                                 FP conv_thrd = 5E-6, int max_iter = 5000,
                                 int soft_max_iter = -1) {
        GMatrix<FL> p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        FL ff[2];
        FL &error = ff[0], &func = ff[1];
        FL beta = 0.0, old_beta = 0.0;
        r.allocate();
        p.allocate();
        r.clear();
        p.clear();
        op(x, r);
        if (consta != 0.0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            cg_precondition(p, r, aa);
            beta = complex_dot(r, p);
            error = complex_dot(r, r);
        }
        if (iprint)
            cout << endl;
        if (pcomm != nullptr)
            pcomm->broadcast(&error, 1, pcomm->root);
        if (abs(error) < conv_thrd) {
            if (pcomm == nullptr || pcomm->root == pcomm->rank)
                func = complex_dot(x, b);
            if (pcomm != nullptr)
                pcomm->broadcast(&func, 1, pcomm->root);
            if (iprint)
                cout << setw(6) << 0 << fixed << setw(24) << setprecision(8)
                     << func << scientific << setw(13) << setprecision(2)
                     << abs(error) << endl;
            p.deallocate();
            r.deallocate();
            nmult = 1;
            return func;
        }
        old_beta = beta;
        if (pcomm != nullptr)
            pcomm->broadcast(p.data, p.size(), pcomm->root);
        GMatrix<FL> hp(nullptr, x.m, x.n), z(nullptr, x.m, x.n);
        hp.allocate();
        z.allocate();
        int xiter = 0;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            hp.clear();
            op(p, hp);
            if (consta != 0.0)
                iadd(hp, p, consta);

            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                FL alpha = old_beta / complex_dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                cg_precondition(z, r, aa);
                error = complex_dot(r, r);
                beta = complex_dot(r, z);
                func = complex_dot(x, b);
                if (iprint)
                    cout << setw(6) << xiter << fixed << setw(24)
                         << setprecision(8) << func << scientific << setw(13)
                         << setprecision(2) << abs(error) << endl;
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&error, 2, pcomm->root);
            if (abs(error) < conv_thrd)
                break;
            else {
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    FL gamma = beta / old_beta;
                    old_beta = beta;
                    iadd(p, z, 1.0 / gamma);
                    iscale(p, gamma);
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(p.data, p.size(), pcomm->root);
            }
        }
        if (xiter == max_iter && abs(error) >= conv_thrd) {
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
    // Solve x in linear equation H x = b
    // by applying deflated CG method
    // where H is symmetric and positive-definite
    // H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static FL deflated_conjugate_gradient(
        MatMul &op, const GDiagonalMatrix<FL> &aa, GMatrix<FL> x, GMatrix<FL> b,
        int &nmult, FL consta = 0.0, bool iprint = false,
        const PComm &pcomm = nullptr, FP conv_thrd = 5E-6, int max_iter = 5000,
        int soft_max_iter = -1,
        const vector<GMatrix<FL>> &ws = vector<GMatrix<FL>>()) {
        shared_ptr<VectorAllocator<FL>> d_alloc =
            make_shared<VectorAllocator<FL>>();
        GMatrix<FL> p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        GMatrix<FL> hp(nullptr, x.m, x.n);
        FL ff[2];
        FL &error = ff[0], &func = ff[1];
        FL beta = 0.0, old_beta = 0.0;
        r.allocate();
        p.allocate();
        hp.allocate();
        int nw = (int)ws.size();
        vector<GMatrix<FL>> aws(nw, GMatrix<FL>(nullptr, x.m, x.n));
        GMatrix<FL> paws(nullptr, (MKL_INT)(nw * x.size()), 1);
        paws.data = d_alloc->allocate(nw * x.size());
        for (int i = 0; i < nw; i++) {
            for (int j = 0; j < i; j++)
                iadd(ws[i], ws[j], -complex_dot(ws[j], ws[i]));
            FL w_normsq = sqrt(complex_dot(ws[i], ws[i]));
            assert(abs(w_normsq) > 1E-14);
            iscale(ws[i], 1.0 / w_normsq);
            aws[i].data = paws.data + ws[i].size() * i;
            aws[i].clear();
            op(ws[i], aws[i]);
            if (consta != 0.0)
                iadd(aws[i], ws[i], consta);
        }
        GMatrix<FL> winv(nullptr, nw, nw);
        GMatrix<FL> mu(nullptr, nw, 1);
        winv.allocate();
        mu.allocate();
        r.clear();
        p.clear();
        hp.clear();
        op(x, r);
        if (consta != 0.0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            if (nw != 0) {
                for (int i = 0; i < nw; i++)
                    for (int j = 0; j <= i; j++) {
                        winv(i, j) = complex_dot(aws[i], ws[j]);
                        winv(j, i) = xconj<FL>(winv(i, j));
                    }
                inverse(winv);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += complex_dot(ws[j], r) * winv(i, j);
                    iadd(x, ws[i], mu.data[i]);
                    iadd(r, aws[i], -mu.data[i]);
                }
            }
            cg_precondition(p, r, aa);
            if (nw != 0) {
                op(p, hp);
                if (consta != 0.0)
                    iadd(hp, p, consta);
                mu.clear();
                for (int i = 0; i < nw; i++) {
                    for (int j = 0; j < nw; j++)
                        mu.data[i] += complex_dot(ws[j], hp) * winv(i, j);
                    iadd(p, ws[i], -mu.data[i]);
                    iadd(hp, aws[i], -mu.data[i]);
                }
            }
            beta = complex_dot(r, p);
            error = complex_dot(r, r);
        }
        if (iprint)
            cout << endl;
        if (pcomm != nullptr)
            pcomm->broadcast(&error, 1, pcomm->root);
        if (abs(error) < conv_thrd) {
            if (pcomm == nullptr || pcomm->root == pcomm->rank)
                func = complex_dot(x, b);
            if (pcomm != nullptr)
                pcomm->broadcast(&func, 1, pcomm->root);
            if (iprint)
                cout << setw(6) << 0 << fixed << setw(24) << setprecision(8)
                     << func << scientific << setw(13) << setprecision(2)
                     << abs(error) << endl;
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
        GMatrix<FL> z(nullptr, x.m, x.n), az(nullptr, x.m, x.n);
        z.allocate();
        az.allocate();
        int xiter = 0;
        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (nw == 0) {
                hp.clear();
                op(p, hp);
                if (consta != 0.0)
                    iadd(hp, p, consta);
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                FL alpha = old_beta / complex_dot(p, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                cg_precondition(z, r, aa);
                error = complex_dot(r, r);
                beta = complex_dot(r, z);
                func = complex_dot(x, b);
                if (iprint)
                    cout << setw(6) << xiter << fixed << setw(24)
                         << setprecision(8) << func << scientific << setw(13)
                         << setprecision(2) << abs(error) << endl;
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&error, 2, pcomm->root);
            if (abs(error) < conv_thrd)
                break;
            else {
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    FL gamma = beta / old_beta;
                    old_beta = beta;
                    iscale(p, gamma);
                    iadd(p, z, 1.0);
                    if (nw != 0) {
                        az.clear();
                        op(z, az);
                        if (consta != 0.0)
                            iadd(az, z, consta);
                        iscale(hp, gamma);
                        iadd(hp, az, 1.0);
                        mu.clear();
                        for (int i = 0; i < nw; i++) {
                            for (int j = 0; j < nw; j++)
                                mu.data[i] +=
                                    complex_dot(ws[j], az) * winv(i, j);
                            iadd(p, ws[i], -mu.data[i]);
                            iadd(hp, aws[i], -mu.data[i]);
                        }
                    }
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(p.data, p.size(), pcomm->root);
            }
        }
        if (xiter == max_iter && abs(error) >= conv_thrd) {
            cout << "Error : linear solver (dcg) not converged!" << endl;
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
    // Solve x in linear equation H x = b where H is Hermitian and pd
    // where H x := op(x) + consta * x
    template <typename MatMul, typename PComm>
    static FL minres(MatMul &op, GMatrix<FL> x, GMatrix<FL> b, int &nmult,
                     FL consta = 0.0, bool iprint = false,
                     const PComm &pcomm = nullptr, FP conv_thrd = 5E-6,
                     int max_iter = 5000, int soft_max_iter = -1) {
        GMatrix<FL> p(nullptr, x.m, x.n), r(nullptr, x.m, x.n);
        FL ff[2];
        FL &error = ff[0], &func = ff[1];
        r.allocate();
        r.clear();
        op(x, r);
        if (consta != 0.0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iadd(r, b, -1);
            iscale(r, -1);
            p.allocate();
            copy(p, r);
            error = complex_dot(r, r);
        }
        if (iprint)
            cout << endl;
        if (pcomm != nullptr)
            pcomm->broadcast(&error, 1, pcomm->root);
        if (abs(error) < conv_thrd) {
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                func = complex_dot(x, b);
                p.deallocate();
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&func, 1, pcomm->root);
            if (iprint)
                cout << setw(6) << 0 << fixed << setw(24) << setprecision(8)
                     << func << scientific << setw(13) << setprecision(2)
                     << abs(error) << endl;
            r.deallocate();
            nmult = 1;
            return func;
        }
        if (pcomm != nullptr)
            pcomm->broadcast(r.data, r.size(), pcomm->root);
        FL beta = 0.0, prev_beta = 0.0;
        GMatrix<FL> hp(nullptr, x.m, x.n), hr(nullptr, x.m, x.n);
        hr.allocate();
        hr.clear();
        op(r, hr);
        if (consta != 0.0)
            iadd(hr, r, consta);
        prev_beta = complex_dot(r, hr);
        int xiter = 0;

        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            hp.allocate();
            copy(hp, hr);
        }

        while (xiter < max_iter &&
               (soft_max_iter == -1 || xiter < soft_max_iter)) {
            xiter++;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                FL alpha = complex_dot(r, hr) / complex_dot(hp, hp);
                iadd(x, p, alpha);
                iadd(r, hp, -alpha);
                error = complex_dot(r, r);
                func = complex_dot(x, b);
                if (iprint)
                    cout << setw(6) << xiter << fixed << setw(24)
                         << setprecision(8) << func << scientific << setw(13)
                         << setprecision(2) << abs(error) << endl;
            }
            if (pcomm != nullptr) {
                pcomm->broadcast(&error, 2, pcomm->root);
                pcomm->broadcast(r.data, r.size(), pcomm->root);
            }
            if (abs(error) < conv_thrd)
                break;
            else {
                hr.clear();
                op(r, hr);
                if (consta != 0.0)
                    iadd(hr, r, consta);
                if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                    beta = complex_dot(r, hr);
                    iadd(p, r, prev_beta / beta);
                    iscale(p, beta / prev_beta);
                    iadd(hp, hr, prev_beta / beta);
                    iscale(hp, beta / prev_beta);
                    prev_beta = beta;
                }
            }
        }
        if (xiter == max_iter && abs(error) >= conv_thrd) {
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
    static FL gcrotmk(MatMul &op, const GDiagonalMatrix<FL> &aa, GMatrix<FL> x,
                      GMatrix<FL> b, int &nmult, int &niter, int m = 20,
                      int k = -1, FL consta = 0.0, bool iprint = false,
                      const PComm &pcomm = nullptr, FP conv_thrd = 5E-6,
                      int max_iter = 5000, int soft_max_iter = -1) {
        GMatrix<FL> r(nullptr, x.m, x.n), w(nullptr, x.m, x.n);
        FL ff[3];
        FL &beta = ff[0], &rr = ff[1], &func = ff[2];
        r.allocate();
        w.allocate();
        r.clear();
        op(x, r);
        if (consta != 0.0)
            iadd(r, x, consta);
        if (pcomm == nullptr || pcomm->root == pcomm->rank) {
            iscale(r, -1);
            iadd(r, b, 1); // r = b - Ax
            func = complex_dot(x, b);
            beta = norm(r);
        }
        if (pcomm != nullptr)
            pcomm->broadcast(&beta, 3, pcomm->root);
        if (iprint)
            cout << endl;
        if (k == -1)
            k = m;
        int xiter = 0, jiter = 1, nn = k + m + 2;
        vector<GMatrix<FL>> cvs(nn, GMatrix<FL>(nullptr, x.m, x.n));
        vector<GMatrix<FL>> uzs(nn, GMatrix<FL>(nullptr, x.m, x.n));
        vector<FL> pcus;
        pcus.reserve(x.size() * 2 * nn);
        for (int i = 0; i < nn; i++) {
            cvs[i].data = pcus.data() + cvs[i].size() * i;
            uzs[i].data = pcus.data() + uzs[i].size() * (i + nn);
        }
        int ncs = 0, icu = 0;
        GMatrix<FL> bmat(nullptr, k, k + m);
        GMatrix<FL> hmat(nullptr, k + m + 1, k + m);
        GMatrix<FL> ys(nullptr, k + m, 1);
        GMatrix<FL> bys(nullptr, k, 1);
        GMatrix<FL> hys(nullptr, k + m + 1, 1);
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
                     << setw(24) << setprecision(8) << func << scientific
                     << setw(13) << setprecision(2) << abs(beta * beta) << endl;
            if (abs(beta * beta) < conv_thrd)
                break;
            int ml = m + max(k - ncs, 0), ivz = icu + ncs + 1, nz = 0;
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                iadd(cvs[ivz % nn], r, 1.0 / beta, false, 0.0);
                hmat.clear();
                hys.clear();
                hys.data[0] = beta;
            }
            for (int j = 0; j < ml; j++) {
                jiter++;
                GMatrix<FL> z(uzs[(ivz + j) % nn].data, x.m, x.n);
                if (pcomm == nullptr || pcomm->root == pcomm->rank)
                    cg_precondition(z, cvs[(ivz + j) % nn], aa);
                if (pcomm != nullptr)
                    pcomm->broadcast(z.data, z.size(), pcomm->root);
                w.clear();
                op(z, w);
                if (consta != 0.0)
                    iadd(w, z, consta);
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
                    iadd(cvs[(ivz + nz) % nn], w, 1.0 / hmat(j + 1, j), false,
                         0.0);
                    rr = least_squares(GMatrix<FL>(hmat.data, j + 2, hmat.n),
                                       GMatrix<FL>(hys.data, j + 2, 1),
                                       GMatrix<FL>(ys.data, j + 1, 1));
                }
                if (pcomm != nullptr)
                    pcomm->broadcast(&rr, 1, pcomm->root);
                if (abs(rr * rr) < conv_thrd)
                    break;
            }
            if (pcomm == nullptr || pcomm->root == pcomm->rank) {
                multiply(GMatrix<FL>(bmat.data, ncs, bmat.n), false,
                         GMatrix<FL>(ys.data, nz, 1), false,
                         GMatrix<FL>(bys.data, ncs, 1), 1.0, 0.0);
                multiply(GMatrix<FL>(hmat.data, nz + 1, hmat.n), false,
                         GMatrix<FL>(ys.data, nz, 1), false,
                         GMatrix<FL>(hys.data, nz + 1, 1), 1.0, 0.0);
                for (int i = 0; i < nz; i++)
                    iadd(uzs[(icu + ncs) % nn], uzs[(ivz + i) % nn], ys(i, 0),
                         false, !!i);
                for (int i = 0; i < ncs; i++)
                    iadd(uzs[(icu + ncs) % nn], uzs[(icu + i) % nn], -bys(i, 0),
                         false);
                for (int i = 0; i < nz + 1; i++)
                    iadd(cvs[(icu + ncs) % nn], cvs[(ivz + i) % nn], hys(i, 0),
                         false, !!i);
                FP alpha = norm(cvs[(icu + ncs) % nn]);
                iscale(cvs[(icu + ncs) % nn], 1.0 / alpha);
                iscale(uzs[(icu + ncs) % nn], 1.0 / alpha);
                FL gamma = complex_dot(cvs[(icu + ncs) % nn], r);
                iadd(r, cvs[(icu + ncs) % nn], -gamma);
                iadd(x, uzs[(icu + ncs) % nn], gamma);
                func = complex_dot(x, b);
                beta = norm(r);
            }
            if (pcomm != nullptr)
                pcomm->broadcast(&beta, 3, pcomm->root);
            if (ncs == k)
                icu = (icu + 1) % nn;
            else
                ncs++;
        }
        if (jiter >= max_iter && abs(beta * beta) >= conv_thrd) {
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
    /** Leja ordering of x.
     *
     * Not that this only works for nondegenerate x and the ordering is not
     * unique
     *
     * @see L, Reichel, The application of Leja points to Richardson iteration
     * and polynomial preconditioning, Linear Algebra and its Applications, 154,
     * 389 (1991) https://doi.org/10.1016/0024-3795(91)90386-B.
     *
     * @param x Input/Output vector (leja ordered)
     * @param permutation Permutation order
     */
    static void leja_order(vector<FL> &x, vector<int> &permutation) {
        const auto n = x.size();
        permutation.resize(n);
        iota(permutation.begin(), permutation.end(), 0);
        int argmax = 0;
        auto m = x[0];
        for (int i = 1; i < n; ++i) {
            if (abs(x[i]) > real(m)) {
                argmax = i;
                m = abs(x[i]);
            }
        }
        swap(x[0], x[argmax]);
        swap(permutation[0], permutation[argmax]);

        vector<FL> p(n, 1); // product vector
        for (int k = 1; k < n - 1; ++k) {
            for (int i = k; i < n; ++i) {
                p[i] *= x[i] - x[k - 1];
            }
            argmax = k;
            m = p[k];
            for (int i = k + 1; i < n; ++i) {
                if (abs(p[i]) > real(m)) {
                    argmax = i;
                    m = p[i];
                }
            }
            swap(x[k], x[argmax]);
            swap(p[k], p[argmax]);
            swap(permutation[k], permutation[argmax]);
        }
    }
    /** Use Induced Dimension Reduction method [IDR(s)] to solve A x = b
     *  IDR(1) is identical to BI-CGSTAB.
     *
     *  Based on https://github.com/astudillor/idrs, which itself is based on
     * the IDR(s)'authors matlab version from
     * http://homepage.tudelft.nl/1w5b5/idrs-software.html
     *
     * See (1) Van Gijzen, M. B.; Sonneveld, P.
     *      Algorithm 913: An Elegant IDR(s) Variant That Efficiently Exploits
     * Biorthogonality Properties. ACM Trans. Math. Softw. 2011, 38 (1), 1–19.
     * https://doi.org/10.1145/2049662.2049667. (Fig. 2)
     *
     * Note: There is also a STAB(L) variant, which may be faster. See, e.g.,
     *          1.  Gerard L. G. Sleijpen and Martin B. van Gijzen,
     *              Exploiting BiCGstab(l) Strategies to Induce Dimension
     * Reduction, SIAM J. Sci. Comput., 32(5), 2687–2709.
     *              https://doi.org/10.1137/090752341
     *          2. Aihara, K., Abe, K., & Ishiwata, E. (2014). A variant of
     * IDRstab with reliable update strategies for solving sparse linear
     * systems. Journal of Computational and Applied Mathematics, 259, 244-258.
     *                     doi:10.1016/j.cam.2013.08.028
     *          3. Aihara, K., Abe, K., & Ishiwata, E. (2015). Preconditioned
     *                   IDRSTABL Algorithms for Solving Nonsymmetric Linear
     * Systems. International Journal of Applied Mathematics, 45(3).
     *
     * @author: Henrik R. Larsson, based on versions by Reinaldo Astudillo and
     * Martin B. van Gijzen
     *
     * @param op Computes op(x) = A x
     * @param a_diagonal Diagonal of A; used for preconditioning.
     *                           Can point to nullptr if it should not be used
     * @param x Input guess/ output solution
     * @param b Right-hand side
     * @param nmult Used number of matrix-vector products (same as niter)
     * @param niter Used total number of iterations
     * @param S Shadow space; similar to Krylov space in GMRES
     *              Typically, s being around 10 or even 4 is enough.
     *              Only very badly conditioned problems require s ~ 100; see
     * (1). S=1 is identical to BI-CGSTAB.
     * @param iprint Whether to print output during the iterations
     * @param pcomm MPI communicator
     * @param precond_reg Preconditioning regularizer. Fix the inverse of
     * a_diagonal to be at max. the inverse of this.
     * @param tol Convergence tolerance: ||Ax - b|| <=  max(tol*||b||, atol)
     * @param atol Convergence tolerance: ||Ax - b|| <=  max(tol*||b||, atol)
     * @param max_iter Maximum number of iterations. Throws error afterward.
     * @param soft_max_iter Maximum number of iterations, without throwing error
     * @param init_basis_in Optional initial basis for the search direction.
     *      Defaults to zero
     * @param omega_used Optional values of used direction magnitudes.
     *      Defaults to GMRES strategy. If given, Leja ordering is useful
     *      (+ permuting init_basis accordingly)
     * @param orthogonalize_P Orthogonalize the random space P matrix of size
     *      (N x S). May be good for numerical stability.
     * @param random_seed Random seed for setting up P. Defaults to
     *      day-time-convolution
     * @param use_leja_ordering Use Leja ordering for omega_used and init_basis
     *          Attention: This will modify omega_used and init_basis
     * @return <x,b>
     */
    template <typename MatMul, typename PComm>
    static FL
    idrs(MatMul &op, const GDiagonalMatrix<FL> &a_diagonal, GMatrix<FL> x,
         GMatrix<FL> b, int &nmult, int &niter, MKL_INT S = 8,
         const bool iprint = false, const PComm &pcomm = nullptr,
         const FP precond_reg = 1E-8, const FP tol = 1E-3, const FP atol = 0.0,
         const int max_iter = 5000, const int soft_max_iter = -1,
         const vector<GMatrix<FL>> &init_basis_in = {},
         const vector<FL> &omega_used_in = {},
         const bool orthogonalize_P = true, const int random_seed = -1,
         const bool use_leja_ordering = false) {
        assert(b.m == x.m);
        assert(b.n == x.n);
        vector<GMatrix<FL>> init_basis;
        init_basis.reserve(init_basis_in.size());
        auto omega_used = omega_used_in;
        if (use_leja_ordering) {
            assert(omega_used.size() > 0);
            vector<int> permutation;
            leja_order(omega_used, permutation);
            if (init_basis_in.size() > 0) {
                assert(init_basis_in.size() == omega_used.size());
                assert(init_basis_in.size() == permutation.size());
                for (int i = 0; i < init_basis_in.size(); ++i) {
                    const auto &b = init_basis_in[permutation[i]];
                    init_basis.emplace_back(GMatrix<FL>(b.data, b.m, b.n));
                }
            }
        } else { // Copy init_basis references
            for (int i = 0; i < init_basis_in.size(); ++i) {
                const auto &ba = init_basis_in[i];
                init_basis.emplace_back(GMatrix<FL>(ba.data, ba.m, ba.n));
            }
        }
        const auto N = b.m; // vector size
        S = min(S, N);      // Gracefully change S to sth reasonable.
                            // This should only affect tiny linear problems.
        assert(b.n == 1 &&
               "IDRS currently is only implemented for rhs being a vector.");
        // Allocations
        GMatrix<FL> r(nullptr, N, 1); // Residual
        GMatrix<FL> P(nullptr, S,
                      N); // Shadow-space matrix; S will be left null space of P
        GMatrix<FL> f(nullptr, S, 1);        // P r
        GMatrix<FL> cStorage(nullptr, S, 1); // Mc = f
        GMatrix<FL> v(nullptr, N, 1);        // r - G c
        GMatrix<FL> tmp(nullptr, N, 1);
        //                          vvv changed from py implementation for
        //                          row-majorness
        GMatrix<FL> G(nullptr, S, N);  // Subspace matrix; for updating residual
        GMatrix<FL> U(nullptr, S, N);  // For updating x.
        GMatrix<FL> M(nullptr, S, S);  // M = P' G
        GMatrix<FL> MM(nullptr, S, S); // copy of M
        r.allocate();
        r.clear();
        P.allocate();
        P.clear();
        f.allocate();
        f.clear();
        cStorage.allocate();
        cStorage.clear();
        v.allocate();
        v.clear();
        tmp.allocate();
        tmp.clear();
        G.allocate();
        G.clear();
        U.allocate();
        U.clear();
        M.allocate();
        M.clear();
        MM.allocate();
        MM.clear();
        // Initialization
        const FP norm_b = norm(b);
        const auto used_tol = max(tol * norm_b, atol);
        // compute residual: r = b - Ax
        if (norm(x) > 1E-20) {
            op(x, tmp);
            copy(r, b);
            iadd(r, tmp, -1.0);
        } else {
            copy(r, b);
        }
        FP norm_r = norm(r);
        if (pcomm != nullptr)
            pcomm->broadcast(&norm_r, 1, pcomm->root);
        // Fill P with random numbers. Let's hope it is of full rank.
        {
            Random rgen;
            rgen.rand_seed(random_seed);
            rgen.fill<FP>((FP *)P.data, P.size() * cpx_sz, -1.0, 1.0);
            if (orthogonalize_P) {
                // Make P orthogonal. This is not required but may improve
                // numerical stability. The original code IDR(S) does it.
                MKL_INT k = min(P.m, P.n);
                vector<FL> tau(k);
                vector<FL> work(P.m);
                MKL_INT info = 0;
                xgeqrf<FL>(&P.n, &P.m, P.data, &P.n, tau.data(), work.data(),
                           &P.m, &info);
                assert(info == 0 &&
                       "IDR(S): Fail in QR decomposition of random matrix P");
                xungqr<FL>(&P.n, &P.m, &k, P.data, &P.n, tau.data(),
                           work.data(), &P.m, &info);
                assert(info == 0 && "IDR(S): Fail in Q build up of QR "
                                    "decomposition of random matrix P");
            }
        }
        // M = 1
        for (size_t i = 0; i < S; ++i) {
            M(i, i) = 1.0;
        }
        const FP angle = 0.7071067811865476; // To avoid too small residuals
                                             //  see (1) on page 4; same as
                                             //  Bi-CGSTAB; sqrt(2)/2
        FL omega = 1.0;
        int iOmega = 0;
        if (omega_used.size() > 0) {
            omega = omega_used[0];
        }
        // do it
        const auto doContinue = [max_iter, soft_max_iter, used_tol](int iter,
                                                                    FP rnorm) {
            if (rnorm <= used_tol)
                return false;
            return iter < max_iter &&
                   (soft_max_iter == -1 || iter < soft_max_iter);
        };
        const auto precondition = [&a_diagonal, precond_reg,
                                   N](GMatrix<FL> in) {
            if (a_diagonal.data == nullptr)
                return;
            for (size_t i = 0; i < N; ++i) {
                if (abs(a_diagonal(i, i)) > precond_reg) {
                    in(i, 0) /= a_diagonal(i, i);
                } else {
                    in(i, 0) /= precond_reg;
                }
            }
        };
        niter = 0;
        if (iprint) {
            cout << endl << "Start IDR(" << S << ")" << endl;
            cout << "tol= " << scientific << tol << " atol= " << scientific
                 << atol << " used tol= " << scientific << used_tol << endl;
            cout << "maxiter: " << max_iter << "; " << soft_max_iter << endl;
            auto xdb = complex_dot(x, b);
            cout << "Initially:         " << fixed << setw(15)
                 << setprecision(8) << real(xdb) << "+" << setw(15)
                 << setprecision(8) << imag(xdb) << "i" << scientific
                 << setw(13) << setprecision(2) << norm_r << endl;
        }
        int outeriter = 0;
        while (doContinue(niter, norm_r)) {
            // vvv I need P.conj() @ r ...; on the other hand, P is random
            // anyways. so it should not matter?
            // multiply(P,false, r,false, f, cmplx(1.0,0.0), cmplx(0.0,0.0));
            for (size_t i = 0; i < S; ++i) {
                f(i, 0) = complex_dot(GMatrix<FL>(&P(i, 0), N, 1), r);
            }
            for (size_t k = 0; k < S; ++k) { // Krylov space setup
                // solve Mc = f
                const auto size = S - k;
                GMatrix<FL> uk(&U(k, 0), N, 1);
                GMatrix<FL> gk(&G(k, 0), N, 1);
                if (outeriter > 0) {
                    GMatrix<FL> c(cStorage.data, size, 1);
                    {
                        // c = la.solve(M[k:S, k:S], f[k:S])
                        // TODO: avoid copy&paste; would it work by changing
                        // lda?
                        GMatrix<FL> M2(MM.data, size, size);
                        GMatrix<FL> ff(&f(k, 0), size, 1);
                        for (size_t i = k; i < S; ++i) {
                            for (size_t j = k; j < S; ++j) {
                                M2(i - k, j - k) = M(i, j);
                            }
                        }
                        least_squares(M2, ff,
                                      c); // zgels may be a bit overkill but I
                                          // guess it can't hurt and S is small
                    }
                    // v = r - G[k:S,:].T @ c
                    copy(v, r);
                    //       vv this should work as I assume that G is row-major
                    //                          ATTENTION:      vvv is
                    //                          transpose, not adjoint (I do
                    //                          want transpose here)
                    multiply(GMatrix<FL>(&G(k, 0), size, N), true, c, false, v,
                             -1.0, 1.0);

                    precondition(v);
                    // Compute new U[:,k] and G[:,k]; G[:,k] is in space G_j
                    // ATTENTION: vv Need to be N x 1 and not 1 x N.
                    //  Otherwise an assertion explodes somewhere deep in the
                    //  code when cllaed op
                    //       uk = U c + omega v
                    // tmp = U[k:S,:].T @ cc
                    copy(tmp, v);
                    multiply(GMatrix<FL>(&U(k, 0), size, N), true, c, false,
                             tmp, 1.0, omega);
                    copy(uk, tmp);
                } else if (k < init_basis.size()) {
                    assert(init_basis[k].m == N && init_basis[k].n == 1);
                    copy(uk, init_basis[k]);
                } else {
                    copy(uk, r);
                    precondition(uk);
                }
                // G = A @ U[:,k]
                op(uk, gk);
                // Bi-Orthogonalize the new basis vectors
                for (size_t i = 0; i < k; ++i) {
                    FL alpha = complex_dot(GMatrix<FL>(&P(i, 0), N, 1), gk);
                    alpha /= M(i, i);
                    iadd(gk, GMatrix<FL>(&G(i, 0), N, 1), -alpha);
                    iadd(uk, GMatrix<FL>(&U(i, 0), N, 1), -alpha);
                }
                // M = P' G (first k-1 entries are zero)
                for (size_t i = k; i < S; ++i) {
                    M(i, k) = complex_dot(GMatrix<FL>(&P(i, 0), N, 1), gk);
                }
                if (abs(M(k, k)) < 1e-20) { // oops!
                    if (iprint)
                        cout << "ATTENTION! |M(k,k)| < 1e-20" << endl;
                    break;
                }
                // Make r orthogonal to g_i, i = 1..k
                const FL beta = f(k, 0) / M(k, k);
                iadd(r, gk, -beta);
                iadd(x, uk, +beta);
                norm_r = norm(r);
                if (pcomm != nullptr) {
                    pcomm->broadcast(x.data, x.size(), pcomm->root);
                    pcomm->broadcast(r.data, r.size(), pcomm->root);
                    pcomm->broadcast(&norm_r, 1, pcomm->root);
                }
                ++niter;
                if (iprint) {
                    FL xdb = complex_dot(x, b);
                    cout << setw(6) << niter << " inner " << setw(6) << k
                         << fixed << setw(17) << setprecision(8) << real(xdb)
                         << "+" << setw(17) << setprecision(8) << imag(xdb)
                         << "i" << scientific << setw(13) << setprecision(2)
                         << norm_r << endl;
                }
                if (not doContinue(niter, norm_r)) {
                    break;
                }
                // New f = P'*r (first k components are zero)
                if (k < S - 1) {
                    for (size_t j = k + 1; j < S; ++j) {
                        f(j, 0) -= beta * M(j, k);
                    }
                }
            } // Krylov space setup
            ++outeriter;

            if (not doContinue(niter, norm_r))
                break;
            // Precondition v = Minv r; TODO avoid copy&paste
            copy(v, r);
            precondition(v);

            op(v, tmp); // tmp = A v
            if (omega_used.size() == 0) {
                const auto norm_Av = norm(tmp);
                const auto tr = complex_dot(tmp, r);
                omega = tr / complex_dot(tmp, tmp);
                auto abs_rho = abs(tr / (norm_Av * norm_r));
                if (pcomm != nullptr) {
                    pcomm->broadcast(&abs_rho, 1, pcomm->root);
                    pcomm->broadcast(&omega, 1, pcomm->root);
                }
                if (abs_rho < angle) {
                    omega *= angle / abs_rho;
                }
            } else {
                omega = omega_used[iOmega++];
                iOmega = iOmega > omega_used.size() ? 0 : iOmega;
            }
            // r -= omega t; x += omega v
            iadd(r, tmp, -omega);
            iadd(x, v, +omega);
            if (pcomm != nullptr) {
                pcomm->broadcast(x.data, x.size(), pcomm->root);
                pcomm->broadcast(r.data, r.size(), pcomm->root);
            }
            norm_r = norm(r);
            if (pcomm != nullptr)
                pcomm->broadcast(&norm_r, 1, pcomm->root);
            ++niter;
            if (iprint) {
                auto xdb = complex_dot(x, b);
                cout << setw(6) << niter << " outer " << outeriter << fixed
                     << setw(17) << setprecision(8) << real(xdb) << "+"
                     << setw(17) << setprecision(8) << imag(xdb) << "i"
                     << scientific << setw(13) << setprecision(2) << norm_r
                     << endl;
            }
            if (norm_r <= used_tol) {
                break;
            }
        }

        // Deallocations
        MM.deallocate();
        M.deallocate();
        U.deallocate();
        G.deallocate();
        tmp.deallocate();
        v.deallocate();
        cStorage.deallocate();
        f.deallocate();
        P.deallocate();
        r.deallocate();

        if (niter >= max_iter && norm_r > used_tol) {
            cerr << "Error: linear solver IDR(S) not converged!" << endl;
            cerr << "\t total number of iterations used:" << niter << endl;
            cerr << "\t S=:" << S << endl;
            cerr << "\t ||Ax-b||=" << norm_r << endl;
            throw runtime_error("Linear solver IDR(S) not converged.");
        }
        nmult = niter;
        auto out = complex_dot(x, b);
        if (pcomm != nullptr) {
            pcomm->broadcast(x.data, x.size(), pcomm->root);
            pcomm->broadcast(&out, 1, pcomm->root);
        }
        return out;
    }
    //////////////////////////
    // LSQR stuff
    // Closely following scipy's implementation
    // Henrik R. Larsson
    // Original licence text in scipy:
    /*
    The original Fortran code was written by C. C. Paige and M. A. Saunders as
    described in
    C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
    equations and sparse least squares, TOMS 8(1), 43--71 (1982).
    C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
    equations and least-squares problems, TOMS 8(2), 195--209 (1982).
    It is licensed under the following BSD license:
            Copyright (c) 2006, Systems Optimization Laboratory
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions
    are met:
            * Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
            disclaimer in the documentation and/or other materials provided
            with the distribution.
    * Neither the name of Stanford University nor the names of its
            contributors may be used to endorse or promote products derived
            from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
            LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
    OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. The Fortran code was translated
    to Python for use in CVXOPT by Jeffery Kline with contributions by Mridul
    Aanjaneya and Bob Myhill. Adapted for SciPy by Stefan van der Walt.
     */
    //////////////////////////
    /**     Stable implementation of Givens rotation.
     * References
     * ----------
     * .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
     *    and Least-Squares Problems", Dissertation,
     *    http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
     */
    static tuple<FP, FP, FP> sym_ortho(const FP a, const FP b) {
        const auto sign = [](FP v) -> FP {
            return (FP)(int)((0.0 < v) - (v < 0.0));
        };
        // const auto sign = [&signR](cmplx v){ return abs(real(v)) < 1e-30 ?
        // signR(real(v)) : signR(imag(v)); };
        if (b == 0.0) {
            return make_tuple<FP, FP, FP>(sign(a), 0.0, abs(a));
        } else if (a == 0.0) {
            return make_tuple<FP, FP, FP>(0.0, sign(b), abs(b));
        } else if (abs(b) > abs(a)) {
            const FP tau = a / b;
            const FP s = sign(b) / sqrt(1. + tau * tau);
            const FP c = s * tau;
            const FP r = b / s;
            return make_tuple(c, s, r);
        } else {
            const FP tau = b / a;
            const FP c = sign(a) / sqrt(1. + tau * tau);
            const FP s = c * tau;
            const FP r = a / c;
            return make_tuple(c, s, r);
        }
    }

    /** LSQR implementation. See scipy.sparse.linalg.lsqr
     *
     *  I removed the tamp parameter
     * @author  Henrik R. Larsson, based on scipy's implementation
     * @param op Computes op(x) = A x
     * @param rop Computes rop(x) = A' x
     * @param a_diagonal Diagonal of A; used for preconditioning. Can point to
     * nullptr if it should not be used Here, preconditioning solves [A inv(M)]
     * [M x] = b
     * @param x Input guess/ output solution
     * @param b Right-hand side
     * @param nmult Used number of matrix-vector products (same as niter)
     * @param niter Used total number of iterations
     * @param iprint Whether to print output during the iterations
     * @param pcomm MPI communicator
     * @param precond_reg Preconditioning regularizer. Fix the inverse of
     * a_diagonal to be at max. the inverse of this.
     * @param btol, atol Stopping tolerances. If both are 1.0e-9 (say),
     *          the final residual norm should be accurate to about 9 digits.
     *         (The final x will usually have fewer correct digits, depending on
     * cond(A)) atol (btol) defines relative error estimate in A (b) The
     * stopping criteria are:
     * 1: ||Ax - b || <= btol ||b|| + atol ||A|| ||x||
     * 2: ||A (A x- b)'|| / (||A|| ||Ax - b|| + eps) <= atol
     * @param max_iter Maximum number of iterations. Throws error afterward.
     * @param soft_max_iter Maximum number of iterations, without throwing error
     * @return <x,b>
     */
    template <typename MatMul, typename MatMul2, typename PComm>
    static FL lsqr(MatMul &op, MatMul2 &rop,
                   const GDiagonalMatrix<FL> &a_diagonal, GMatrix<FL> x,
                   GMatrix<FL> b, int &nmult, int &niter,
                   const bool iprint = false, const PComm &pcomm = nullptr,
                   const FP precond_reg = 1E-8, const FP btol = 1E-3,
                   const FP atol = 1E-3, const int max_iter = 5000,
                   const int soft_max_iter = -1) {
        assert(b.m == x.m);
        assert(b.n == x.n);
        constexpr FP one = 1.0;
        const auto N = b.m; // vector size
        const auto precondition = [&a_diagonal, precond_reg,
                                   N](const GMatrix<FL> &in,
                                      const GMatrix<FL> &out) {
            assert(a_diagonal.data != nullptr);
            for (size_t i = 0; i < N; ++i) {
                if (abs(a_diagonal(i, i)) > precond_reg) {
                    out(i, 0) = in(i, 0) / a_diagonal(i, i);
                } else {
                    out(i, 0) = in(i, 0) / precond_reg;
                }
            }
        };
        GMatrix<FL> tmpP(nullptr, N, 1);
        if (a_diagonal.data != nullptr) {
            tmpP.allocate();
        }
        const auto opM = [&op, &a_diagonal, &tmpP, precond_reg,
                          N](const GMatrix<FL> &in, const GMatrix<FL> &out) {
            if (a_diagonal.data == nullptr) {
                op(in, out);
                return;
            }
            // out = A M in
            for (size_t i = 0; i < N; ++i) {
                if (abs(a_diagonal(i, i)) > precond_reg) {
                    tmpP(i, 0) = in(i, 0) / a_diagonal(i, i);
                } else {
                    tmpP(i, 0) = in(i, 0) / precond_reg;
                }
            }
            op(tmpP, out);
        };
        const auto ropM = [&rop, &a_diagonal, precond_reg,
                           N](const GMatrix<FL> &in, const GMatrix<FL> &out) {
            rop(in, out);
            if (a_diagonal.data == nullptr) {
                return;
            }
            // out = M' A' in
            for (size_t i = 0; i < N; ++i) {
                if (abs(a_diagonal(i, i)) > precond_reg) {
                    out(i, 0) /= xconj<FL>(a_diagonal(i, i));
                } else {
                    out(i, 0) /= precond_reg;
                }
            }
        };

        GMatrix<FL> u(nullptr, N, 1);
        GMatrix<FL> tmp(nullptr, N, 1);
        GMatrix<FL> v(nullptr, N, 1);
        GMatrix<FL> w(nullptr, N, 1);
        u.allocate();
        tmp.allocate();
        v.allocate();
        w.allocate();

        niter = 0;
        FP beta, alpha;
        int istop = 0;
        FP anorm = 0.;
        FP acond = 0.;
        FP ddnorm = 0.;
        FP res1 = 0.;
        FP res2 = 0.;
        FP xnorm = 0.;
        FP xxnorm = 0.;
        FP z = 0.;
        FP cs2 = -1.;
        FP sn2 = 0.;
        FP test1, test2, test3, rtol;
        constexpr FP eps = std::numeric_limits<FP>::epsilon();
        // Set up the first vectors u and v for the bidiagonalization.
        //        These satisfy  beta*u = b - A*x,  alpha*v = A'*u.
        copy(u, b);
        auto bnorm = norm(b);
        xnorm = norm(x);
        if (pcomm != nullptr) {
            pcomm->broadcast(&bnorm, 1, pcomm->root);
            pcomm->broadcast(&xnorm, 1, pcomm->root);
        }
        if (xnorm < 1E-20) {
            iscale(x, 0.0);
            beta = bnorm;
        } else {
            opM(x, v);
            ++nmult;
            iadd(u, v, -1.);
            beta = norm_accurate(u);
        }
        if (beta > 0.0) {
            // iscale(u, 1.0 / beta); // vv is more accurate
            for (size_t i = 0; i < N; ++i) {
                u(i, 0) /= beta;
            }
            ropM(u, v);
            ++nmult;
            alpha = norm_accurate(v);
            if (pcomm != nullptr) {
                pcomm->broadcast(&alpha, 1, pcomm->root);
            }
            // iscale(v, 1.0 / alpha); // vv is more accurate
            for (size_t i = 0; i < N; ++i) {
                v(i, 0) /= alpha;
            }
        } else {
            alpha = 0.0;
            copy(v, x);
        }
        copy(w, v);

        auto rhobar = alpha;
        auto phibar = beta;
        auto rnorm = beta;
        auto r1norm = rnorm;
        auto arnorm = alpha * beta;
        vector<string> msg{
            "The exact solution is  x = 0                              ",
            "Ax - b is small enough, given atol, btol                  ",
            "The least-squares solution is good enough, given atol     ",
            "The estimate of cond(Abar) has exceeded conlim            ",
            "Ax - b is small enough for this machine                   ",
            "The least-squares solution is good enough for this machine",
            "Cond(Abar) seems to be too large for this machine         ",
            "The iteration limit has been reached                      "};
        assert(arnorm != 0 && "The exact solution is x = 0");
        test1 = one;
        test2 = alpha / beta;
        if (iprint) {
            cout << endl
                 << "   Itn    <x|b>                             r1norm     "
                 << "      Compatible       LS               Norm A           "
                    "Cond A"
                 << endl;
            auto out = complex_dot(x, b);
            cout << setw(6) << niter << scientific << setw(17)
                 << setprecision(8) << real(out) << "+" << scientific
                 << setw(17) << setprecision(8) << imag(out) << "i  "
                 << scientific << setw(9) << setprecision(8) << r1norm << "   "
                 << scientific << setw(9) << setprecision(8) << test1 << "   "
                 << scientific << setw(9) << setprecision(8) << test2 << endl;
        }
        // Main iteration loop
        while (niter < max_iter &&
               (soft_max_iter == -1 || niter < soft_max_iter)) {
            ++niter;
            /*
             *  Perform the next step of the bidiagonalization to obtain the
             *  next  beta, u, alfa, v.  These satisfy the relations
             *   beta*u  =  a*v   -  alpha*u,
             *     alpha*v  =  A'*u  -  beta*v.
             */
            // u = A @ v - alpha u
            opM(v, tmp);
            ++nmult;
            iscale(u, -alpha);
            iadd(u, tmp, 1.);
            beta = norm_accurate(u);
            if (pcomm != nullptr) {
                pcomm->broadcast(&beta, 1, pcomm->root);
            }

            if (beta > 0.0) {
                // iscale(u, 1./beta); // vv is more accurate
                for (size_t i = 0; i < N; ++i) {
                    u(i, 0) /= beta;
                }
                anorm = sqrt(anorm * anorm + alpha * alpha + beta * beta);
                // v = A' @ u - beta * v
                ropM(u, tmp);
                ++nmult;
                iscale(v, -beta);
                iadd(v, tmp, 1.0);
                alpha = norm_accurate(v);
                if (pcomm != nullptr) {
                    pcomm->broadcast(&alpha, 1, pcomm->root);
                }
                if (alpha > 0.0) {
                    // iscale(v, 1./alpha);
                    for (size_t i = 0; i < N; ++i) {
                        v(i, 0) /= alpha;
                    }
                }
            }

            // Use a plane rotation to eliminate the damping parameter.
            // This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
            auto rhobar1 = sqrt(rhobar * rhobar);
            auto cs1 = rhobar / rhobar1;
            auto sn1 = 1. / rhobar1;
            auto psi = sn1 * phibar;
            phibar = cs1 * phibar;

            // Use a plane rotation to eliminate the subdiagonal element (beta)
            // of the lower-bidiagonal matrix, giving an upper-bidiagonal
            // matrix.
            // auto [cs, sn, rho] = sym_ortho(rhobar1, beta); //Sigh
            auto tupl = sym_ortho(rhobar1, beta); // Sigh; C++17
            auto cs = get<0>(tupl);
            auto sn = get<1>(tupl);
            auto rho = get<2>(tupl);

            auto theta = sn * alpha;
            rhobar = -cs * alpha;
            auto phi = cs * phibar;
            phibar = sn * phibar;
            auto tau = sn * phi;

            // Update x and w.
            auto t1 = phi / rho;
            auto t2 = -theta / rho;
            if (pcomm != nullptr) {
                pcomm->broadcast(&rho, 1, pcomm->root);
                pcomm->broadcast(&t1, 1, pcomm->root);
                pcomm->broadcast(&t2, 1, pcomm->root);
            }
            copy(tmp, w);
            iscale(tmp, 1. / rho);

            iadd(x, w, t1); // x = x + t1 * w
            iscale(w, t2);  // w = v + t2 * w
            iadd(w, v, 1.);

            auto normdk = norm_accurate(tmp);
            if (pcomm != nullptr) {
                pcomm->broadcast(&normdk, 1, pcomm->root);
            }
            ddnorm = ddnorm + normdk * normdk;

            // Use a plane rotation on the right to eliminate the
            // super-diagonal element (theta) of the upper-bidiagonal matrix.
            // Then use the result to estimate norm(x).
            auto delta = sn2 * rho;
            auto gambar = -cs2 * rho;
            auto rhs = phi - delta * z;
            auto zbar = rhs / gambar;
            xnorm = sqrt(xxnorm + zbar * zbar);
            auto gamma = sqrt(gambar * gambar + theta * theta);
            cs2 = gambar / gamma;
            sn2 = theta / gamma;
            z = rhs / gamma;
            xxnorm = xxnorm + z * z;

            // Test for convergence.
            // First, estimate the condition of the matrix  Abar,
            // and the norms of  rbar  and  Abar'rbar.
            acond = anorm * sqrt(ddnorm);
            res1 = phibar * phibar;
            res2 = res2 + psi * psi;
            rnorm = sqrt(res1 + res2);
            arnorm = alpha * abs(tau);

            auto r1sq = rnorm * rnorm;
            r1norm = sqrt(abs(r1sq));
            if (r1sq < 0)
                r1norm *= -1;

            // Now use these norms to estimate certain other quantities,
            // some of which will be small near a solution.
            test1 = rnorm / bnorm;
            test2 = arnorm / (anorm * rnorm + eps);
            test3 = one / (acond + eps);
            t1 = test1 / (one + anorm * xnorm / bnorm);
            rtol = btol + atol * anorm * xnorm / bnorm;

            // The following tests guard against extremely small values of
            // atol, btol  or   (The user may have set any or all of
            // the parameters  atol, btol, conlim  to 0.)
            // The effect is equivalent to the normal tests using
            // atol = eps, btol = eps, conlim = 1/eps.
            if (one + test3 <= one) {
                istop = 6;
            }
            if (one + test2 <= one) {
                istop = 5;
            }
            if (one + t1 <= one) {
                istop = 4;
            }
            // Allow for tolerances set by the user.
            if (test2 <= atol) {
                istop = 2;
            }
            if (test1 <= rtol) {
                istop = 1;
            }
            if (!(niter < max_iter &&
                  (soft_max_iter == -1 || niter < soft_max_iter))) {
                istop = 7;
            }

            if (iprint) {
                auto out = complex_dot(x, b);
                cout << setw(6) << niter << scientific << setw(17)
                     << setprecision(8) << real(out) << "+" << scientific
                     << setw(17) << setprecision(8) << imag(out) << "i  "
                     << scientific << setw(9) << setprecision(8) << r1norm
                     << "   " << scientific << scientific << setw(9)
                     << setprecision(8) << test1 << "   " << scientific
                     << setw(9) << setprecision(8) << test2 << "   "
                     << scientific << setw(9) << setprecision(8) << anorm
                     << "   " << scientific << setw(9) << setprecision(8)
                     << acond << endl;
            }
            if (istop != 0) {
                break;
            }
        }
        if (iprint) {
            cout << "istop = " << istop << endl;
            cout << "msg = " << msg.at(istop) << endl;
        }
        // hrl: istop == 5 should be fine
        if (niter >= max_iter || (istop > 2 && istop != 7 && istop != 5)) {
            cerr << "Error: linear solver LSQR not converged!" << endl;
            cerr << "\t total number of iterations used:" << niter << endl;
            cout << "msg = " << msg.at(istop) << endl;
            throw runtime_error("Linear solver LSQR not converged.");
        }

        w.deallocate();
        v.deallocate();
        tmp.deallocate();
        u.deallocate();
        if (a_diagonal.data != nullptr) {
            tmpP.deallocate();
            // M x = z
            for (size_t i = 0; i < N; ++i) {
                if (abs(a_diagonal(i, i)) > precond_reg) {
                    x(i, 0) /= a_diagonal(i, i);
                } else {
                    x(i, 0) /= precond_reg;
                }
            }
        }
        nmult = niter;
        auto out = complex_dot(x, b);
        if (pcomm != nullptr) {
            pcomm->broadcast(x.data, x.size(), pcomm->root);
            pcomm->broadcast(&out, 1, pcomm->root);
        }
        return out;
    }

    /** Chebychev implementation
     *
     * @author  Henrik R. Larsson
     * @param op Computes op(x) = A x
     * @param x Input guess/ output solution
     * @param b Right-hand side
     * @param iprint Whether to print output during the iterations
     * @param pcomm MPI communicator
     * @param max_iter Maximum number of iterations, without throwing error
     * @return <x,b>
     */
    // TODO allow for preconditioner
    // TODO allow for zero eta, using numerical expansion
    template <typename MatMul, typename PComm>
    static FC cheby(MatMul &op, GMatrix<FC> x, // ATTENTION: Assume x and shift
                                               // is complex but op is real
                    const GMatrix<FL> b, const FC evalShift, const FP tol,
                    const int max_iter, FP eMin, FP eMax, const FP maxInterval,
                    const int damping, // 0 1 2
                    const bool iprint = false, const PComm &pcomm = nullptr) {
        assert(maxInterval <= 1 && maxInterval > 0);
        assert(eMin < eMax);
        const auto scale = 2 * maxInterval / (eMax - eMin); // 1/a = deltaH
        const auto Hbar = eMin + maxInterval / scale; //  (eMax + eMin) / 2
        const auto Ashift = -(scale * eMin + maxInterval);
        assert(b.m == x.m);
        assert(b.n == x.n);
        //
        // Compute max cheby expansion from numerical coefficient
        //
        const auto chebCoeffNum = [scale, Hbar, evalShift](int j,
                                                           int polOrder) {
            FC c{0., 0.};
            const auto pi = acos(-1.);
            for (int k = 0; k < polOrder; ++k) {
                auto pix = cos(pi * (k + .5) / polOrder) / scale + Hbar;
                auto fct =
                    1. / (pix + evalShift); // Function f(pix) to approximate
                c += fct * cos(pi * j * (k + .5) / polOrder);
            }
            c *= 2. / polOrder;
            return c;
        };
        const auto eta = imag(evalShift);
        assert(eta >= 0.);
        int nCheby = min(static_cast<int>(ceil(1.1 / (scale * eta))),
                         max_iter); // just an estimate
        if (abs(chebCoeffNum(nCheby - 1, nCheby)) < tol) {
            for (; nCheby >= 3; --nCheby) {
                if (abs(chebCoeffNum(nCheby - 1, nCheby)) > tol) {
                    ++nCheby;
                    break;
                }
            }
        } else {
            for (; nCheby <= max_iter; ++nCheby) {
                if (abs(chebCoeffNum(nCheby - 1, nCheby)) < tol)
                    break;
            }
        }

        //
        // Init
        //
        const auto N = b.m; // vector size
        // Compute chebychev expansion on the fly
        GMatrix<FL> phi(nullptr, N, 1);
        GMatrix<FL> phiMinus(nullptr, N, 1);
        GMatrix<FL> phiMinusMinus(nullptr, N, 1);
        phi.allocate();
        phiMinus.allocate();
        phiMinusMinus.allocate();

        const auto AshiftOp = [&op, scale, Ashift, N](const GMatrix<FL> &in,
                                                      GMatrix<FL> &out) {
            op(in, out);
            for (size_t i = 0; i < N; ++i)
                out(i, 0) = scale * out(i, 0) + Ashift * in(i, 0);
        };
        //
        // Series
        //
        complex<long double> zs = evalShift;
        constexpr complex<long double> zone{1., 0.};
        const auto cast = [](const double in) {
            return static_cast<long double>(in);
        };
        //                  vv original formula was for (-A + w); so need to
        //                  change sign here
        zs = cast(scale) * -zs + cast(Ashift);
        const auto zs2 = zs * zs;
        vector<complex<long double>> xOut(N, {0., 0.});
        for (int iCheb = 0; iCheb < nCheby; ++iCheb) {
            if (iCheb == 0) {
                copy(phi, b); // phi0 = b
            } else if (iCheb == 1) {
                AshiftOp(phiMinus, phi); // phi1 = H phi0
            } else {
                AshiftOp(phiMinus, phi); // phin = 2 H phi_n-1 - phi_n-2
                for (size_t i = 0; i < N; ++i) {
                    phi(i, 0) = 2. * phi(i, 0) - phiMinusMinus(i, 0);
                }
            }
            // add
            auto fac = iCheb == 0 ? zone : zone + zone;
            auto damp = zone;             // TODO add damping option
            auto fa = pow(zs, iCheb + 1); // TODO compute iteratively.
            auto fu = pow(zone + sqrt(zs2) * sqrt(zs2 - zone) / zs2, -iCheb);
            auto prec = fa * sqrt(zone - zone / zs2);
            // alternative vv; less accurate but seems to be more stable
            // prec = -static_cast<complex<long double>>(chebCoeffNum(iCheb,
            // nCheby));
            if (prec != zone and not isnan(real(fu / prec)) and
                not isnan(imag(fu / prec))) {
                prec = damp * fac * fu / prec;
                if (iprint)
                    cout << iCheb << " " << prec << ", " << dot(phi, phi)
                         << endl;
                for (size_t i = 0; i < N; ++i)
                    //      vv original formula was for (-A + w); so need to
                    //      change sign here
                    xOut[i] -= cast(scale) * prec * cast(phi(i, 0));
            } else {
                break; // Only gets worse!
                // Can I abort expansion?; With the tol criterium, this should
                // not occur, though
            }
            // next
            // vv could also be done much cheaper using pointers
            copy(phiMinusMinus, phiMinus);
            copy(phiMinus, phi);
        }
        phiMinusMinus.deallocate();
        phiMinus.deallocate();
        phi.deallocate();

        FC out{0., 0.};
        for (size_t i = 0; i < N; ++i) {
            x(i, 0) = xOut[i];
            out += conj(x(i, 0)) * b(i, 0);
        }
        if (pcomm != nullptr) {
            pcomm->broadcast(x.data, x.size(), pcomm->root);
            pcomm->broadcast(&out, 1, pcomm->root);
        }
        return out;
    }
};

} // namespace block2
