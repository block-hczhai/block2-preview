
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

#include "matrix.hpp"
#include "matrix_functions.hpp"
#include "threading.hpp"
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace std;

namespace block2 {

#ifndef _HAS_INTEL_MKL

extern "C" {

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void dgemm(const char *transa, const char *transb, const MKL_INT *m,
                  const MKL_INT *n, const MKL_INT *k, const double *alpha,
                  const double *a, const MKL_INT *lda, const double *b,
                  const MKL_INT *ldb, const double *beta, double *c,
                  const MKL_INT *ldc) noexcept;
}

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

inline void cblas_dgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array,
    const MKL_INT *N_Array, const MKL_INT *K_Array, const double *alpha_Array,
    const double **A_Array, const MKL_INT *lda_Array, const double **B_Array,
    const MKL_INT *ldb_Array, const double *beta_Array, double **C_Array,
    const MKL_INT *ldc_Array, const MKL_INT group_count,
    const MKL_INT *group_size) {
    assert(Layout == CblasRowMajor);
    for (MKL_INT ig = 0, i = 0; ig < group_count; ig++) {
        const char *tra = TransA_Array[ig] == CblasNoTrans ? "n" : "t";
        const char *trb = TransB_Array[ig] == CblasNoTrans ? "n" : "t";
        const MKL_INT m = M_Array[ig], n = N_Array[ig], k = K_Array[ig];
        const double alpha = alpha_Array[ig], beta = beta_Array[ig];
        const MKL_INT lda = lda_Array[ig], ldb = ldb_Array[ig],
                      ldc = ldc_Array[ig];
        const MKL_INT gsize = group_size[ig];
        for (MKL_INT j = 0; j < gsize; j++, i++)
            dgemm(trb, tra, &n, &m, &k, &alpha, B_Array[i], &ldb, A_Array[i],
                  &lda, &beta, C_Array[i], &ldc);
    }
}

#endif

inline void threaded_dgemm_batch(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array,
    const MKL_INT *N_Array, const MKL_INT *K_Array, const double *alpha_Array,
    const double **A_Array, const MKL_INT *lda_Array, const double **B_Array,
    const MKL_INT *ldb_Array, const double *beta_Array, double **C_Array,
    const MKL_INT *ldc_Array, const MKL_INT group_count,
    const MKL_INT *group_size) {
    assert(Layout == CblasRowMajor);
    vector<MKL_INT> gidxs;
    gidxs.reserve(group_count);
    for (MKL_INT ig = 0, i = 0; ig < group_count; ig++) {
        const MKL_INT gsize = group_size[ig];
        gidxs.insert(gidxs.end(), gsize, ig);
        i += gsize;
    }
    int ntq = threading->activate_quanta();
#pragma omp parallel for schedule(dynamic) num_threads(ntq)
    for (MKL_INT i = 0; i < (int)gidxs.size(); i++) {
        const MKL_INT ig = gidxs[i];
        const char *tra = TransA_Array[ig] == CblasNoTrans ? "n" : "t";
        const char *trb = TransB_Array[ig] == CblasNoTrans ? "n" : "t";
        const MKL_INT m = M_Array[ig], n = N_Array[ig], k = K_Array[ig];
        const double alpha = alpha_Array[ig], beta = beta_Array[ig];
        const MKL_INT lda = lda_Array[ig], ldb = ldb_Array[ig],
                      ldc = ldc_Array[ig];
        const MKL_INT gsize = group_size[ig];
        dgemm(trb, tra, &n, &m, &k, &alpha, B_Array[i], &ldb, A_Array[i], &lda,
              &beta, C_Array[i], &ldc);
    }
}

inline void single_dgemm(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
    const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array,
    const MKL_INT *N_Array, const MKL_INT *K_Array, const double *alpha_Array,
    const double *A, const MKL_INT *lda_Array, const double *B,
    const MKL_INT *ldb_Array, const double *beta_Array, double *C,
    const MKL_INT *ldc_Array, const MKL_INT *group_size, double scale = 1.0) {
    const MKL_INT ig = 0, i = 0;
    const char *tra = TransA_Array[ig] == CblasNoTrans ? "n" : "t";
    const char *trb = TransB_Array[ig] == CblasNoTrans ? "n" : "t";
    const MKL_INT m = M_Array[ig], n = N_Array[ig], k = K_Array[ig];
    const double alpha = alpha_Array[ig] * scale, beta = beta_Array[ig];
    const MKL_INT lda = lda_Array[ig], ldb = ldb_Array[ig], ldc = ldc_Array[ig];
    const MKL_INT gsize = group_size[ig];
    dgemm(trb, tra, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

// The parameters for a series of DGEMM operations
struct BatchGEMM {
    const CBLAS_LAYOUT layout = CblasRowMajor;
    vector<CBLAS_TRANSPOSE> ta, tb;
    vector<MKL_INT> n, m, k, gp, lda, ldb, ldc;
    vector<double> alpha, beta;
    vector<const double *> a, b;
    vector<double *> c;
    size_t work, nflop;
    BatchGEMM() : work(0), nflop(0) {}
    void resize(size_t nn) {
        ta.resize(nn), tb.resize(nn);
        n.resize(nn), m.resize(nn), k.resize(nn);
        gp.resize(nn), alpha.resize(nn), beta.resize(nn);
        lda.resize(nn), ldb.resize(nn), ldc.resize(nn);
        a.resize(nn), b.resize(nn), c.resize(nn);
    }
    void copy_from(size_t ii, const shared_ptr<BatchGEMM> &batch) {
        memcpy(ta.data() + ii, batch->ta.data(),
               sizeof(CBLAS_TRANSPOSE) * batch->ta.size());
        memcpy(tb.data() + ii, batch->tb.data(),
               sizeof(CBLAS_TRANSPOSE) * batch->tb.size());
        memcpy(n.data() + ii, batch->n.data(),
               sizeof(MKL_INT) * batch->n.size());
        memcpy(m.data() + ii, batch->m.data(),
               sizeof(MKL_INT) * batch->m.size());
        memcpy(k.data() + ii, batch->k.data(),
               sizeof(MKL_INT) * batch->k.size());
        memcpy(gp.data() + ii, batch->gp.data(),
               sizeof(MKL_INT) * batch->gp.size());
        memcpy(lda.data() + ii, batch->lda.data(),
               sizeof(MKL_INT) * batch->lda.size());
        memcpy(ldb.data() + ii, batch->ldb.data(),
               sizeof(MKL_INT) * batch->ldb.size());
        memcpy(ldc.data() + ii, batch->ldc.data(),
               sizeof(MKL_INT) * batch->ldc.size());
        memcpy(alpha.data() + ii, batch->alpha.data(),
               sizeof(double) * batch->alpha.size());
        memcpy(beta.data() + ii, batch->beta.data(),
               sizeof(double) * batch->beta.size());
        memcpy(a.data() + ii, batch->a.data(),
               sizeof(const double *) * batch->a.size());
        memcpy(b.data() + ii, batch->b.data(),
               sizeof(const double *) * batch->b.size());
        memcpy(c.data() + ii, batch->c.data(),
               sizeof(double *) * batch->c.size());
    }
    void dgemm_group(bool conja, bool conjb, MKL_INT m, MKL_INT n, MKL_INT k,
                     double alpha, MKL_INT lda, MKL_INT ldb, double beta,
                     MKL_INT ldc, MKL_INT gc) {
        ta.push_back(conja ? CblasTrans : CblasNoTrans);
        tb.push_back(conjb ? CblasTrans : CblasNoTrans);
        this->m.push_back(m), this->n.push_back(n), this->k.push_back(k);
        this->alpha.push_back(alpha), this->beta.push_back(beta);
        this->lda.push_back(lda), this->ldb.push_back(ldb),
            this->ldc.push_back(ldc);
        this->gp.push_back(gc);
        nflop += (size_t)m * n * k * gc;
    }
    void dgemm_array(const double *a, const double *b, double *c) {
        this->a.push_back(a), this->b.push_back(b), this->c.push_back(c);
    }
    void dgemm(bool conja, bool conjb, MKL_INT m, MKL_INT n, MKL_INT k,
               double alpha, const double *a, MKL_INT lda, const double *b,
               MKL_INT ldb, double beta, double *c, MKL_INT ldc) {
        dgemm_group(conja, conjb, m, n, k, alpha, lda, ldb, beta, ldc, 1);
        dgemm_array(a, b, c);
    }
    // [a] += scale * [b]
    void iadd(double *a, const double *b, MKL_INT n, double scale = 1.0,
              double cfactor = 1.0) {
        static double x = 1.0;
        this->dgemm(false, false, n, 1, 1, scale, b, 1, &x, 1, cfactor, a, 1);
    }
    // [a] = scale * [a]
    void iscale(double *a, MKL_INT n, double scale = 1.0) {
        static double x = 1.0;
        this->dgemm(false, false, n, 1, 1, 0.0, a, 1, &x, 1, scale, a, 1);
    }
    // [c] = [a] * (scalar b) or [c] = (scalar a) * [b] or [c] = [a] \otimes [b]
    void tensor_product(const MatrixRef &a, bool conja, const MatrixRef &b,
                        bool conjb, const MatrixRef &c, double scale,
                        uint32_t stride, double cfactor = 1.0) {
        if (a.m == 1 && a.n == 1) {
            if (!conjb && b.n == c.n)
                this->dgemm(false, false, b.m * b.n, 1, 1, scale, b.data, 1,
                            a.data, 1, cfactor, &c(0, stride), 1);
            else if (!conjb) {
                this->dgemm_group(false, false, b.n, 1, 1, scale, 1, 1, cfactor,
                                  1, b.m);
                for (MKL_INT k = 0; k < b.m; k++)
                    this->dgemm_array(&b(k, 0), a.data, &c(k, stride));
            } else {
                this->dgemm_group(false, false, b.m, 1, 1, scale, b.n, 1,
                                  cfactor, 1, b.n);
                for (MKL_INT k = 0; k < b.n; k++)
                    this->dgemm_array(&b(0, k), a.data, &c(k, stride));
            }
        } else if (b.m == 1 && b.n == 1) {
            if (!conja && a.n == c.n)
                this->dgemm(false, false, a.m * a.n, 1, 1, scale, a.data, 1,
                            b.data, 1, cfactor, &c(0, stride), 1);
            else if (!conja) {
                this->dgemm_group(false, false, a.n, 1, 1, scale, 1, 1, cfactor,
                                  1, a.m);
                for (MKL_INT k = 0; k < a.m; k++)
                    this->dgemm_array(&a(k, 0), b.data, &c(k, stride));
            } else {
                this->dgemm_group(false, false, a.m, 1, 1, scale, a.n, 1,
                                  cfactor, 1, a.n);
                for (MKL_INT k = 0; k < a.n; k++)
                    this->dgemm_array(&a(0, k), b.data, &c(k, stride));
            }
        } else {
            if (!conja && !conjb) {
                this->dgemm_group(false, false, b.n, 1, 1, scale, 1, 1, cfactor,
                                  1, b.m * a.m * a.n);
                for (MKL_INT i = 0; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++)
                        for (MKL_INT k = 0; k < b.m; k++)
                            this->dgemm_array(
                                &b(k, 0), &a(i, j),
                                &c(i * b.m + k, j * b.n + stride));
            } else if (conja && !conjb) {
                this->dgemm_group(false, false, b.n, 1, 1, scale, 1, 1, cfactor,
                                  1, b.m * a.m * a.n);
                for (MKL_INT i = 0; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++)
                        for (MKL_INT k = 0; k < b.m; k++)
                            this->dgemm_array(
                                &b(k, 0), &a(j, i),
                                &c(i * b.m + k, j * b.n + stride));
            } else if (!conja && conjb) {
                this->dgemm_group(false, false, b.m, 1, 1, scale, b.n, 1,
                                  cfactor, 1, b.n * a.m * a.n);
                for (MKL_INT i = 0; i < a.m; i++)
                    for (MKL_INT j = 0; j < a.n; j++)
                        for (MKL_INT k = 0; k < b.n; k++)
                            this->dgemm_array(
                                &b(0, k), &a(i, j),
                                &c(i * b.n + k, j * b.m + stride));
            } else {
                this->dgemm_group(false, false, b.m, 1, 1, scale, b.n, 1,
                                  cfactor, 1, b.n * a.m * a.n);
                for (MKL_INT i = 0; i < a.n; i++)
                    for (MKL_INT j = 0; j < a.m; j++)
                        for (MKL_INT k = 0; k < b.n; k++)
                            this->dgemm_array(
                                &b(0, k), &a(j, i),
                                &c(i * b.n + k, j * b.m + stride));
            }
        }
    }
    // [c] = [a] x [b]
    void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                  bool conjb, const MatrixRef &c, double scale,
                  double cfactor) {
        this->dgemm(conja, conjb, c.m, conjb ? b.m : b.n, conjb ? b.n : b.m,
                    scale, a.data, a.n, b.data, b.n, cfactor, c.data, c.n);
    }
    // [c] = diag(a) (out product) diag(b)
    void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                 const MatrixRef &c, double scale) {
        this->dgemm(false, true, a.n, b.n, 1, scale, a.data, a.n + 1, b.data,
                    b.n + 1, 1.0, c.data, c.n);
    }
    //  dleft: [c] = scale * diag([a] = da x db) x diag(b)
    // !dleft: [c] = scale * diag(a) x diag([b] = da x db)
    void three_tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                       const MatrixRef &c, const MatrixRef &da,
                                       bool dconja, const MatrixRef &db,
                                       bool dconjb, bool dleft, double scale,
                                       uint32_t stride) {
        const MKL_INT dstrm = (MKL_INT)stride / (dleft ? a.m : b.m);
        const MKL_INT dstrn = (MKL_INT)stride % (dleft ? a.m : b.m);
        if (dstrn != dstrm)
            return;
        assert(da.m == da.n && db.m == db.n);
        const MKL_INT ddstr = 0;
        if (da.m == 1 && da.n == 1) {
            const double *bdata =
                dconjb ? &db(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &db(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (db.n > abs(ddstr)) {
                if (dleft)
                    // (1 x db) x b
                    this->dgemm(false, true, db.n - abs(ddstr), b.n, 1,
                                scale * *da.data, bdata, db.n + 1, b.data,
                                b.n + 1, 1.0, &c(max(dstrn, dstrm), (MKL_INT)0),
                                c.n);
                else
                    // a x (1 x db)
                    this->dgemm(false, true, a.n, db.n - abs(ddstr), 1,
                                scale * *da.data, a.data, a.n + 1, bdata,
                                db.n + 1, 1.0, &c(0, max(dstrn, dstrm)), c.n);
            }
        } else if (db.m == 1 && db.n == 1) {
            const double *adata =
                dconja ? &da(max(-ddstr, (MKL_INT)0), max(ddstr, (MKL_INT)0))
                       : &da(max(ddstr, (MKL_INT)0), max(-ddstr, (MKL_INT)0));
            if (da.n > abs(ddstr)) {
                if (dleft)
                    // (da x 1) x b
                    this->dgemm(false, true, da.n - abs(ddstr), b.n, 1,
                                scale * *db.data, adata, da.n + 1, b.data,
                                b.n + 1, 1.0, &c(max(dstrn, dstrm), (MKL_INT)0),
                                c.n);
                else
                    // a x (da x 1)
                    this->dgemm(false, true, a.n, da.n - abs(ddstr), 1,
                                scale * *db.data, a.data, a.n + 1, adata,
                                da.n + 1, 1.0, &c(0, max(dstrn, dstrm)), c.n);
            }
        } else
            assert(false);
    }
    // Execute DGEMM operation groups from index ii to ii + nn
    void perform(MKL_INT ii = 0, MKL_INT kk = 0, MKL_INT nn = 0) {
        if (nn != 0 || gp.size() != 0) {
            if (threading->type & ThreadingTypes::Quanta)
                threaded_dgemm_batch(
                    layout, &ta[ii], &tb[ii], &m[ii], &n[ii], &k[ii],
                    &alpha[ii], &a[kk], &lda[ii], &b[kk], &ldb[ii], &beta[ii],
                    &c[kk], &ldc[ii], nn == 0 ? (MKL_INT)gp.size() : nn,
                    &gp[ii]);
            else
                cblas_dgemm_batch(layout, &ta[ii], &tb[ii], &m[ii], &n[ii],
                                  &k[ii], &alpha[ii], &a[kk], &lda[ii], &b[kk],
                                  &ldb[ii], &beta[ii], &c[kk], &ldc[ii],
                                  nn == 0 ? (MKL_INT)gp.size() : nn, &gp[ii]);
        }
    }
    inline void perform_single(MKL_INT ii, const double *a, const double *b,
                               double *c, double scale) {
        single_dgemm(layout, &ta[ii], &tb[ii], &m[ii], &n[ii], &k[ii],
                     &alpha[ii], a, &lda[ii], b, &ldb[ii], &beta[ii], c,
                     &ldc[ii], &gp[ii], scale);
    }
    inline void perform_single(MKL_INT ii, const double *a, const double *b,
                               double *c) {
        single_dgemm(layout, &ta[ii], &tb[ii], &m[ii], &n[ii], &k[ii],
                     &alpha[ii], a, &lda[ii], b, &ldb[ii], &beta[ii], c,
                     &ldc[ii], &gp[ii]);
    }
    void clear() {
        ta.clear(), tb.clear();
        n.clear(), m.clear(), k.clear(), gp.clear();
        lda.clear(), ldb.clear(), ldc.clear();
        alpha.clear(), beta.clear();
        a.clear(), b.clear(), c.clear();
        work = nflop = 0;
    }
    friend ostream &operator<<(ostream &os, const BatchGEMM &c) {
        for (size_t i = 0, k = 0; i < c.gp.size(); k += c.gp[i], i++) {
            os << "[" << setw(3) << i << "] :: GC=" << c.gp[i]
               << " TA=" << (c.ta[i] == CblasTrans ? "T" : "N")
               << " TB=" << (c.tb[i] == CblasTrans ? "T" : "N")
               << " M=" << c.m[i] << " N=" << c.n[i] << " K=" << c.k[i]
               << " ALPHA=" << c.alpha[i] << " BETA=" << c.beta[i]
               << " LDA=" << c.lda[i] << " LDB=" << c.ldb[i]
               << " LDC=" << c.ldc[i] << endl;
            for (size_t j = 0; j < c.gp[i]; j++)
                os << setw(9) << ">" << setw(3) << j << hex
                   << " :: A=" << c.a[k + j] << " B=" << c.b[k + j]
                   << " C=" << c.c[k + j] << dec << endl;
        }
        return os;
    }
};

// A ref to part of BatchGEMM
struct BatchGEMMRef {
    shared_ptr<BatchGEMM> batch;
    MKL_INT i, k, n, nk;
    size_t nflop, work, rwork = 0;
    size_t ipost = 0;
    BatchGEMMRef(const shared_ptr<BatchGEMM> &batch, size_t nflop, size_t work,
                 MKL_INT i, MKL_INT k, MKL_INT n, MKL_INT nk)
        : batch(batch), nflop(nflop), work(work), i(i), k(k), n(n), nk(nk) {}
    void perform() {
        if (n != 0)
            batch->perform(i, k, n);
    }
};

// Batched DGEMM analyzer
struct BatchGEMMSeq {
    shared_ptr<vector<double>> vdata;
    vector<shared_ptr<BatchGEMM>> batch;
    vector<shared_ptr<BatchGEMM>> post_batch;
    // conj_c for change work array position
    // for "rotate" for partial expectation
    vector<bool> cjc;
    vector<BatchGEMMRef> refs;
    size_t cumulative_nflop = 0;
    size_t max_batch_flops = 1LU << 30;
    size_t max_work, max_rwork;
    double *work, *rwork;
    SeqTypes mode;
    bool no_check = true;
    BatchGEMMSeq(size_t max_batch_flops = 1LU << 30,
                 SeqTypes mode = SeqTypes::None)
        : max_batch_flops(max_batch_flops), mode(mode), vdata(nullptr) {
        batch.push_back(make_shared<BatchGEMM>());
        batch.push_back(make_shared<BatchGEMM>());
    }
    // clear copy
    shared_ptr<BatchGEMMSeq> copy() const {
        shared_ptr<BatchGEMMSeq> seq = make_shared<BatchGEMMSeq>(*this);
        assert(seq->vdata == nullptr);
        seq->batch.clear();
        seq->batch.push_back(make_shared<BatchGEMM>());
        seq->batch.push_back(make_shared<BatchGEMM>());
        seq->cjc.clear();
        return seq;
    }
    // [a] = cfactor * [a] + scale * [b]
    void iadd(const MatrixRef &a, const MatrixRef &b, double scale = 1.0,
              double cfactor = 1.0, bool conj = false) {
        static double x = 1;
        if (!conj)
            batch[1]->iadd(a.data, b.data, a.m * a.n, scale, cfactor);
        else
            batch[1]->tensor_product(b, conj, MatrixRef(&x, 1, 1), false, a,
                                     scale, 0, cfactor);
    }
    // [c] = scale * [a] x [b] + cfactor * [c]
    void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                  bool conjb, const MatrixRef &c, double scale,
                  double cfactor) {
        batch[1]->multiply(a, conja, b, conjb, c, scale, cfactor);
    }
    // [c] = scale * [bra](.T) x [a] x [ket](.T)
    void rotate(const MatrixRef &a, const MatrixRef &c, const MatrixRef &bra,
                bool conj_bra, const MatrixRef &ket, bool conj_ket,
                double scale) {
        MatrixRef work((double *)0 + batch[0]->work, a.m,
                       conj_ket ? ket.m : ket.n);
        batch[0]->multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        batch[1]->multiply(bra, conj_bra, work, false, c, scale, 1.0);
        if (mode & SeqTypes::Tasked)
            max_work = max(max_work, work.size());
        batch[0]->work += work.size();
        batch[1]->work += work.size();
    }
    // [c](.T) = scale * [bra].T x [a](.T) x [ket]
    void rotate(const MatrixRef &a, bool conj_a, const MatrixRef &c,
                bool conj_c, const MatrixRef &bra, const MatrixRef &ket,
                double scale) {
        MatrixRef work((double *)0 + batch[0]->work, conj_a ? a.n : a.m, ket.n);
        batch[0]->multiply(a, conj_a, ket, false, work, 1.0, 0.0);
        if (!conj_c)
            batch[1]->multiply(bra, true, work, false, c, scale, 1.0);
        else
            batch[1]->multiply(work, true, bra, false, c, scale, 1.0);
        cjc.push_back(conj_c);
        if (mode & SeqTypes::Tasked)
            max_work = max(max_work, work.size());
        batch[0]->work += work.size();
        batch[1]->work += work.size();
    }
    //  dleft: [c] = scale * [bra] (= [da] x [db]) * [a] * [ket]
    // !dleft: [c] = scale * [bra] * [a] * [ket] (= [da] x [db])
    void three_rotate(const MatrixRef &a, const MatrixRef &c,
                      const MatrixRef &bra, bool conj_bra, const MatrixRef &ket,
                      bool conj_ket, const MatrixRef &da, bool dconja,
                      const MatrixRef &db, bool dconjb, bool dleft,
                      double scale, uint32_t stride) {
        MatrixRef work(nullptr, 0, 0);
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            work = MatrixRef((double *)0 + batch[0]->work, am,
                             conj_ket ? ket.m : ket.n);
            // work = a * ket
            batch[0]->multiply(MatrixRef(&a(ast, 0), am, a.n), false, ket,
                               conj_ket, work, scale, 0.0);
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * work
                batch[1]->multiply(db, dconjb, work, false,
                                   MatrixRef(&c(cst, 0), cm, c.n), *da.data,
                                   1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * work
                batch[1]->multiply(da, dconja, work, false,
                                   MatrixRef(&c(cst, 0), cm, c.n), *db.data,
                                   1.0);
            else
                assert(false);
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            work = MatrixRef((double *)0 + batch[0]->work, a.m, kn);
            if (da.m == 1 && da.n == 1)
                // work = a * (1 x db)
                batch[0]->multiply(MatrixRef(&a(0, ast), a.m, a.n), false, db,
                                   dconjb, work, *da.data, 0.0);
            else if (db.m == 1 && db.n == 1)
                // work = a * (da x 1)
                batch[0]->multiply(MatrixRef(&a(0, ast), a.m, a.n), false, da,
                                   dconja, work, *db.data, 0.0);
            else
                assert(false);
            // c = bra * work
            batch[1]->multiply(bra, conj_bra, work, false,
                               MatrixRef(&c(0, cst), c.m, c.n), scale, 1.0);
        }
        if (mode & SeqTypes::Tasked)
            max_work = max(max_work, work.size());
        batch[0]->work += work.size();
        batch[1]->work += work.size();
    }
    //  dleft: [c] = scale * [a] * [ket]
    // !dleft: [c] = scale * [a] * [ket] (= [da] x [db])
    void three_rotate_tr_left(const MatrixRef &a, const MatrixRef &c,
                              const MatrixRef &bra, bool conj_bra,
                              const MatrixRef &ket, bool conj_ket,
                              const MatrixRef &da, bool dconja,
                              const MatrixRef &db, bool dconjb, bool dleft,
                              double scale, uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            batch[1]->multiply(MatrixRef(&a(ast, 0), am, a.n), false, ket,
                               conj_ket, MatrixRef(&c(cst, 0), cm, c.n), scale,
                               1.0);
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            if (da.m == 1 && da.n == 1)
                // c = a * (1 x db)
                batch[1]->multiply(MatrixRef(&a(0, ast), a.m, a.n), false, db,
                                   dconjb, MatrixRef(&c(0, cst), c.m, c.n),
                                   *da.data * scale, 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = a * (da x 1)
                batch[1]->multiply(MatrixRef(&a(0, ast), a.m, a.n), false, da,
                                   dconja, MatrixRef(&c(0, cst), c.m, c.n),
                                   *db.data * scale, 1.0);
            else
                assert(false);
        }
    }
    //  dleft: [c] = scale * [bra] (= [da] x [db]) * [a]
    // !dleft: [c] = scale * [bra] * [a]
    void three_rotate_tr_right(const MatrixRef &a, const MatrixRef &c,
                               const MatrixRef &bra, bool conj_bra,
                               const MatrixRef &ket, bool conj_ket,
                               const MatrixRef &da, bool dconja,
                               const MatrixRef &db, bool dconjb, bool dleft,
                               double scale, uint32_t stride) {
        if (dleft) {
            dconja ^= conj_bra, dconjb ^= conj_bra;
            MKL_INT am = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT cm = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_bra ? stride / bra.n : stride % bra.n;
            uint32_t cst = conj_bra ? stride % bra.n : stride / bra.n;
            if (da.m == 1 && da.n == 1)
                // c = (1 x db) * a
                batch[1]->multiply(db, dconjb, MatrixRef(&a(ast, 0), am, a.n),
                                   false, MatrixRef(&c(cst, 0), cm, c.n),
                                   *da.data * scale, 1.0);
            else if (db.m == 1 && db.n == 1)
                // c = (da x 1) * a
                batch[1]->multiply(da, dconja, MatrixRef(&a(ast, 0), am, a.n),
                                   false, MatrixRef(&c(cst, 0), cm, c.n),
                                   *db.data * scale, 1.0);
            else
                assert(false);
        } else {
            dconja ^= conj_ket, dconjb ^= conj_ket;
            MKL_INT kn = (dconja ? da.m : da.n) * (dconjb ? db.m : db.n);
            MKL_INT km = (dconja ? da.n : da.m) * (dconjb ? db.n : db.m);
            uint32_t ast = conj_ket ? stride % ket.n : stride / ket.n;
            uint32_t cst = conj_ket ? stride / ket.n : stride % ket.n;
            // c = bra * a
            batch[1]->dgemm(conj_bra, false, c.m, kn, a.m, scale, bra.data,
                            bra.n, &a(0, ast), a.n, 1.0, &c(0, cst), c.n);
        }
    }
    // [c] = scale * diag(a) (out product) diag(b)
    void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                 const MatrixRef &c, double scale) {
        batch[1]->tensor_product_diagonal(a, b, c, scale);
    }
    //  dleft: [c] = scale * diag([a] = da x db) x diag(b)
    // !dleft: [c] = scale * diag(a) x diag([b] = da x db)
    void three_tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                       const MatrixRef &c, const MatrixRef &da,
                                       bool dconja, const MatrixRef &db,
                                       bool dconjb, bool dleft, double scale,
                                       uint32_t stride) {
        batch[1]->three_tensor_product_diagonal(a, b, c, da, dconja, db, dconjb,
                                                dleft, scale, stride);
    }
    // [c + stride] = [a] * (scalar b) or [c] = (scalar a) * [b]
    void tensor_product(const MatrixRef &a, bool conja, const MatrixRef &b,
                        bool conjb, const MatrixRef &c, double scale,
                        uint32_t stride) {
        batch[1]->tensor_product(a, conja, b, conjb, c, scale, stride);
    }
    // Divide batch to several batches
    // so that nflop of each batch is roughly max_batch_flops
    void divide_batch() {
        // no need to divide
        if (max_batch_flops == 0) {
            if (batch[0]->gp.size() != 0)
                refs.push_back(BatchGEMMRef(batch[0], batch[0]->nflop,
                                            batch[0]->work, 0, 0,
                                            (MKL_INT)batch[0]->gp.size(), 0));
            refs.push_back(BatchGEMMRef(batch[1], batch[1]->nflop,
                                        batch[1]->work, 0, 0,
                                        (MKL_INT)batch[1]->gp.size(), 0));
            return;
        }
        size_t cur = 0, cur0 = 0, cwork = 0, pwork = 0;
        MKL_INT ip = 0, kp = 0;
        for (MKL_INT i = 0, k = 0; i < batch[1]->gp.size();
             k += batch[1]->gp[i++]) {
            cur += (size_t)batch[1]->m[i] * batch[1]->n[i] * batch[1]->k[i] *
                   batch[1]->gp[i];
            if (batch[0]->gp.size() != 0) {
                cur0 += (size_t)batch[0]->m[i] * batch[0]->n[i] *
                        batch[0]->k[i] * batch[0]->gp[i];
                cwork += (size_t)batch[0]->m[i] * batch[0]->n[i];
            }
            if (max_batch_flops != 0 && cur >= max_batch_flops) {
                if (batch[0]->gp.size() != 0)
                    refs.push_back(BatchGEMMRef(batch[0], cur0, cwork - pwork,
                                                ip, kp, i + 1 - ip,
                                                k + batch[0]->gp[i] - kp));
                refs.push_back(BatchGEMMRef(batch[1], cur, cwork - pwork, ip,
                                            kp, i + 1 - ip,
                                            k + batch[1]->gp[i] - kp));
                if (pwork != 0) {
                    for (size_t kk = kp; kk < k + batch[1]->gp[i]; kk++)
                        batch[0]->c[kk] -= pwork;
                    for (size_t kk = kp; kk < k + batch[1]->gp[i]; kk++)
                        batch[1]->b[kk] -= pwork;
                }
                cur = 0, cur0 = 0, ip = i + 1, kp = k + batch[1]->gp[i];
                pwork = cwork;
            }
        }
        if (cur != 0) {
            if (batch[0]->gp.size() != 0)
                refs.push_back(BatchGEMMRef(batch[0], cur0, cwork - pwork, ip,
                                            kp, (int)batch[1]->gp.size() - ip,
                                            (int)batch[0]->c.size() - kp));
            refs.push_back(BatchGEMMRef(batch[1], cur, cwork - pwork, ip, kp,
                                        (int)batch[1]->gp.size() - ip,
                                        (int)batch[1]->b.size() - kp));
            if (pwork != 0) {
                for (size_t kk = kp; kk < batch[0]->c.size(); kk++)
                    batch[0]->c[kk] -= pwork;
                for (size_t kk = kp; kk < batch[1]->b.size(); kk++)
                    batch[1]->b[kk] -= pwork;
            }
        }
    }
    // Check whether there are conflicts in output arrays
    bool check() {
        MKL_INT max_nk = 0, db = batch[0]->gp.size() == 0 ? 1 : 2;
        for (MKL_INT ib = !!batch[0]->gp.size(); ib < refs.size(); ib += db)
            max_nk = max(max_nk, refs[ib].nk);
        vector<double *> ptr(max_nk);
        vector<MKL_INT> len(max_nk), idx(max_nk);
        for (MKL_INT ib = !!batch[0]->gp.size(),
                     db = batch[0]->gp.size() == 0 ? 1 : 2;
             ib < refs.size(); ib += db) {
            shared_ptr<BatchGEMM> b = refs[ib].batch;
            if (refs[ib].nk == 0)
                continue;
            MKL_INT xi = refs[ib].i, xk = refs[ib].k;
            for (MKL_INT i = 0, k = 0; i < refs[ib].n; k += b->gp[xi + i++]) {
                for (MKL_INT kk = k; kk < k + b->gp[xi + i]; kk++)
                    ptr[kk] = b->c[xk + kk],
                    len[kk] = b->m[xi + i] * b->n[xi + i];
            }
            for (MKL_INT kk = 0; kk < refs[ib].nk; kk++)
                idx[kk] = kk;
            sort(idx.begin(), idx.begin() + refs[ib].nk,
                 [&ptr](MKL_INT a, MKL_INT b) { return ptr[a] < ptr[b]; });
            for (MKL_INT kk = 1; kk < refs[ib].nk; kk++)
                if (!(ptr[idx[kk]] >= ptr[idx[kk - 1]] + len[idx[kk - 1]]))
                    return false;
        }
        return true;
    }
    // Automatically solve conflicts in output arrays
    // by introducing temporary work arrays
    void prepare() {
        divide_batch();
        MKL_INT max_nk = 0, db = batch[0]->gp.size() == 0 ? 1 : 2;
        for (MKL_INT ib = !!batch[0]->gp.size(); ib < refs.size(); ib += db)
            max_nk = max(max_nk, refs[ib].nk);
        vector<double *> ptr(max_nk);
        vector<MKL_INT> len(max_nk), pos(max_nk), idx(max_nk);
        vector<double *> ptrs;
        vector<MKL_INT> lens;
        vector<map<pair<MKL_INT, MKL_INT>, vector<MKL_INT>>> shifts;
        vector<size_t> pwork;
        for (MKL_INT ib = !!batch[0]->gp.size(); ib < refs.size(); ib += db) {
            shared_ptr<BatchGEMM> b = refs[ib].batch;
            MKL_INT xi = refs[ib].i, xk = refs[ib].k;
            for (MKL_INT i = 0, k = 0; i < refs[ib].n; k += b->gp[xi + i++]) {
                for (MKL_INT kk = k; kk < k + b->gp[xi + i]; kk++)
                    ptr[kk] = b->c[xk + kk],
                    len[kk] = b->m[xi + i] * b->n[xi + i], pos[kk] = xk + kk;
            }
            for (MKL_INT kk = 0; kk < refs[ib].nk; kk++)
                idx[kk] = kk;
            sort(idx.begin(), idx.begin() + refs[ib].nk,
                 [&ptr](MKL_INT a, MKL_INT b) { return ptr[a] < ptr[b]; });
            ptrs.clear(), lens.clear(), shifts.clear();
            for (MKL_INT kk = 0; kk < refs[ib].nk; kk++) {
                if (ptrs.size() == 0) {
                    ptrs.push_back(ptr[idx[kk]]);
                    lens.push_back(len[idx[kk]]);
                    shifts.push_back(
                        map<pair<MKL_INT, MKL_INT>, vector<MKL_INT>>());
                    shifts.back()[make_pair(0, len[idx[kk]])].push_back(
                        pos[idx[kk]]);
                } else if (ptr[idx[kk]] >= ptrs.back() &&
                           ptr[idx[kk]] < ptrs.back() + lens.back()) {
                    shifts
                        .back()[make_pair((MKL_INT)(ptr[idx[kk]] - ptrs.back()),
                                          len[idx[kk]])]
                        .push_back(pos[idx[kk]]);
                    if (ptr[idx[kk]] + len[idx[kk]] > ptrs.back() + lens.back())
                        lens.back() = (MKL_INT)(ptr[idx[kk]] + len[idx[kk]] -
                                                ptrs.back());
                } else if (ptr[idx[kk]] == ptrs.back() + lens.back()) {
                    lens.back() += len[idx[kk]];
                    shifts
                        .back()[make_pair((MKL_INT)(ptr[idx[kk]] - ptrs.back()),
                                          len[idx[kk]])]
                        .push_back(pos[idx[kk]]);
                } else {
                    ptrs.push_back(ptr[idx[kk]]);
                    lens.push_back(len[idx[kk]]);
                    shifts.push_back(
                        map<pair<MKL_INT, MKL_INT>, vector<MKL_INT>>());
                    shifts.back()[make_pair(0, len[idx[kk]])].push_back(
                        pos[idx[kk]]);
                }
            }
            pwork.clear();
            pwork.reserve(ptrs.size());
            vector<vector<pair<MKL_INT, vector<MKL_INT>>>> rshifts;
            for (size_t p = 0; p < ptrs.size(); p++) {
                pwork.push_back(0);
                rshifts.push_back(vector<pair<MKL_INT, vector<MKL_INT>>>());
                MKL_INT sh = 0, le = 0;
                for (auto &r : shifts[p]) {
                    if (r.first.first > sh || le == 0)
                        sh = r.first.first, le = r.first.second;
                    if (r.first.first == sh && r.first.second == le)
                        rshifts.back().push_back(make_pair(sh, r.second));
                }
                size_t q = 0;
                for (auto &r : shifts[p]) {
                    if (r.first.first != rshifts.back()[q].first) {
                        assert(r.first.first == rshifts.back()[q - 1].first);
                        rshifts.back()[q - 1].second.insert(
                            rshifts.back()[q - 1].second.end(),
                            r.second.begin(), r.second.end());
                        for (size_t qq = q; qq < rshifts.back().size(); qq++)
                            if (rshifts.back()[qq].first > r.first.first &&
                                rshifts.back()[qq].first <
                                    r.first.first + r.first.second)
                                for (size_t u = 0; u < r.second.size(); u++)
                                    rshifts.back()[qq].second.push_back(-1);
                    } else
                        q++;
                }
                for (auto &r : rshifts[p])
                    if (r.second.size() > pwork.back())
                        pwork.back() = r.second.size();
            }
            refs[ib].rwork = 0;
            for (size_t p = 0; p < ptrs.size(); p++)
                refs[ib].rwork += pwork[p] * lens[p];
            double *rr = 0;
            for (size_t p = 0; p < ptrs.size(); p++) {
                for (auto &r : rshifts[p]) {
                    for (size_t q = 0; q < r.second.size(); q++)
                        if (r.second[q] != -1)
                            b->c[r.second[q]] = rr + q * lens[p] + r.first;
                }
                rr += pwork[p] * lens[p];
            }
            size_t max_pwork = *max_element(pwork.begin(), pwork.end());
            size_t ppost = post_batch.size(), ipost = 0;
            while (max_pwork > ((size_t)1 << ipost))
                ipost++;
            refs[ib].ipost = ipost + 1;
            for (size_t ip = 0; ip < ipost + 1; ip++)
                post_batch.push_back(make_shared<BatchGEMM>());
            rr = 0;
            for (size_t p = 0; p < ptrs.size(); p++) {
                for (size_t ip = 0, ipx = 1, ipy = 2; ip < ipost;
                     ip++, ipx <<= 1, ipy <<= 1)
                    for (size_t q = 0; q + ipx < pwork[p]; q += ipy)
                        post_batch[ppost + ip]->iadd(rr + q * lens[p],
                                                     rr + (q + ipx) * lens[p],
                                                     lens[p]);
                post_batch[ppost + ipost]->iadd(ptrs[p], rr, lens[p]);
                rr += pwork[p] * lens[p];
            }
        }
    }
    // Allocate work arrays
    void allocate() {
        max_work = max_rwork = 0;
        for (MKL_INT ib = 0; ib < refs.size(); ib++) {
            max_work = max(max_work, refs[ib].work);
            max_rwork = max(max_rwork, refs[ib].rwork);
        }
        vdata = make_shared<vector<double>>(max_work + max_rwork);
        if (max_work != 0) {
            work = vdata->data();
            size_t shift = work - (double *)0;
            for (size_t i = 0; i < batch[0]->c.size(); i++)
                batch[0]->c[i] += shift;
            if (cjc.size() == 0)
                for (size_t i = 0; i < batch[1]->b.size(); i++)
                    batch[1]->b[i] += shift;
            else {
                for (size_t i = 0; i < batch[1]->b.size(); i++)
                    if (!cjc[i])
                        batch[1]->b[i] += shift;
                    else
                        batch[1]->a[i] += shift;
            }
        }
        if (max_rwork != 0) {
            rwork = vdata->data() + max_work;
            size_t shift = rwork - (double *)0;
            size_t ipost = 0;
            for (size_t i = 0; i < batch[1]->c.size(); i++)
                batch[1]->c[i] += shift;
            for (MKL_INT ib = !!batch[0]->gp.size(),
                         db = batch[0]->gp.size() == 0 ? 1 : 2;
                 ib < refs.size(); ib += db) {
                for (size_t k = ipost; k < ipost + refs[ib].ipost - 1; k++)
                    for (size_t i = 0; i < post_batch[k]->a.size(); i++) {
                        post_batch[k]->a[i] += shift;
                        post_batch[k]->c[i] += shift;
                    }
                for (size_t i = 0, p = ipost + refs[ib].ipost - 1;
                     i < post_batch[p]->a.size(); i++)
                    post_batch[p]->a[i] += shift;
                ipost += refs[ib].ipost;
            }
        }
    }
    // Deallocate work arrays
    void deallocate() { vdata = nullptr; }
    // Perform non-confliciting batched DGEMM
    void simple_perform() {
        divide_batch();
        if (!no_check)
            assert(check());
        allocate();
        perform();
        deallocate();
        clear();
    }
    // SeqTypes::Auto:
    //   Perform possibly confliciting batched DGEMM
    //   An analysis is performed to automatically resolve conflicts
    // SeqTypes::Tasked:
    //   Each thread write to thread-copied outputs
    void auto_perform(const MatrixRef &v = MatrixRef(nullptr, 0, 0)) {
        if (mode == SeqTypes::Auto) {
            prepare();
            allocate();
            perform();
            deallocate();
            clear();
        } else if (mode & SeqTypes::Tasked) {
            int ntop = threading->activate_operator();
            vector<MatrixRef> vts(ntop, v);
            assert(batch[0]->c.size() == 0);
#pragma omp parallel num_threads(ntop)
            {
                int tid = threading->get_thread_id();
                shared_ptr<VectorAllocator<double>> d_alloc =
                    make_shared<VectorAllocator<double>>();
                if (tid != 0)
                    vts[tid].allocate(d_alloc);
                size_t t_vshift = vts[tid].data - v.data;
#pragma omp for schedule(static)
                for (int i = 0; i < (int)batch[1]->c.size(); i++)
                    batch[1]->perform_single(i, batch[1]->a[i], batch[1]->b[i],
                                             batch[1]->c[i] + t_vshift);
#pragma omp single
                parallel_reduce(vts, 0, ntop);
                if (tid != 0)
                    vts[tid].deallocate(d_alloc);
            }
            threading->activate_normal();
            cumulative_nflop += batch[1]->nflop;
            clear();
        }
    }
    // low mem mode for noise
    void auto_perform(const vector<MatrixRef> &vs) {
        assert(mode & SeqTypes::Tasked);
        int ntop = threading->activate_operator();
        assert(batch[0]->c.size() == 0);
#pragma omp parallel num_threads(ntop)
        {
            int tid = threading->get_thread_id();
            for (int ig = 0; tid + ig * ntop < vs.size(); ig++)
                for (size_t i = 0; i < batch[1]->c.size(); i++) {
                    const MatrixRef &v = vs[tid + ig * ntop];
                    if (batch[1]->c[i] >= v.data &&
                        batch[1]->c[i] < v.data + v.size())
                        batch[1]->perform_single((int)i, batch[1]->a[i],
                                                 batch[1]->b[i],
                                                 batch[1]->c[i]);
                }
        }
        threading->activate_normal();
        cumulative_nflop += batch[1]->nflop;
        clear();
    }
    // Directly perform batched DGEMM
    void perform() {
        size_t ipost = 0;
        for (auto b : refs) {
            if (b.rwork != 0)
                memset(rwork, 0, sizeof(double) * b.rwork);
            cumulative_nflop += b.nflop;
            b.perform();
            for (size_t ib = ipost; ib < ipost + b.ipost; ib++)
                post_batch[ib]->perform();
            ipost += b.ipost;
        }
        assert(ipost == post_batch.size());
    }
    void parallel_reduce(const vector<MatrixRef> &mats, int i, int j) const {
        assert(j > i);
        if (j - i == 1)
            return;
        int m = (i + j) >> 1;
#ifdef _MSC_VER
        parallel_reduce(mats, i, m);
        parallel_reduce(mats, m, j);
#else
#pragma omp task
        parallel_reduce(mats, i, m);
#pragma omp task
        parallel_reduce(mats, m, j);
#pragma omp taskwait
#endif
        MatrixFunctions::iadd(mats[i], mats[m], 1.0);
    }
    // Matrix multiply vector (c) => vector (v)
    // (in automatic mode)
    void operator()(const MatrixRef &c, const MatrixRef &v,
                    double scale = 1.0) {
        size_t cshift = c.data - (double *)0;
        size_t vshift = v.data - (double *)0;
        if (mode == SeqTypes::Auto) {
            assert(scale == 1.0);
            for (size_t i = 0; i < batch[0]->a.size(); i++)
                batch[0]->a[i] += cshift;
            size_t ipost = 0;
            for (auto b : refs) {
                if (b.ipost != 0)
                    for (size_t i = 0;
                         i < post_batch[ipost + b.ipost - 1]->c.size(); i++)
                        post_batch[ipost + b.ipost - 1]->c[i] += vshift;
                ipost += b.ipost;
            }
            perform();
            for (size_t i = 0; i < batch[0]->a.size(); i++)
                batch[0]->a[i] -= cshift;
            ipost = 0;
            for (auto b : refs) {
                if (b.ipost != 0)
                    for (size_t i = 0;
                         i < post_batch[ipost + b.ipost - 1]->c.size(); i++)
                        post_batch[ipost + b.ipost - 1]->c[i] -= vshift;
                ipost += b.ipost;
            }
        } else if (mode & SeqTypes::Tasked) {
            int ntop = threading->activate_operator();
            vector<MatrixRef> vts(ntop, v);
            vector<MatrixRef> works(ntop,
                                    MatrixRef(nullptr, (MKL_INT)max_work, 1));
            if (batch[0]->c.size() == 0 && batch[1]->c.size() == 0)
                return;
            assert(max_rwork == 0 && max_work != 0);
#pragma omp parallel num_threads(ntop)
            {
                int tid = threading->get_thread_id();
                shared_ptr<VectorAllocator<double>> d_alloc =
                    make_shared<VectorAllocator<double>>();
                if (tid != 0)
                    vts[tid].allocate(d_alloc);
                works[tid].allocate(d_alloc);
                size_t t_vshift = vts[tid].data - (double *)0;
#pragma omp for schedule(static)
                for (int i = 0; i < (int)batch[0]->c.size(); i++) {
                    batch[0]->perform_single(i, batch[0]->a[i] + cshift,
                                             batch[0]->b[i], works[tid].data);
                    batch[1]->perform_single(i, batch[1]->a[i], works[tid].data,
                                             batch[1]->c[i] + t_vshift, scale);
                }
#pragma omp single
                parallel_reduce(vts, 0, ntop);
                works[tid].deallocate(d_alloc);
                if (tid != 0)
                    vts[tid].deallocate(d_alloc);
            }
            threading->activate_normal();
            cumulative_nflop += batch[0]->nflop;
            cumulative_nflop += batch[1]->nflop;
        } else
            assert(false);
    }
    // Clear all DGEMM parameters
    void clear() {
        for (auto b : batch)
            b->clear();
        post_batch.clear();
        refs.clear();
        cjc.clear();
        max_rwork = max_work = 0;
    }
    friend ostream &operator<<(ostream &os, const BatchGEMMSeq &c) {
        os << endl;
        os << "[0] SIZE = " << c.batch[0]->gp.size()
           << " WORK = " << c.batch[0]->work << endl;
        os << "[1] SIZE = " << c.batch[1]->gp.size()
           << " WORK = " << c.batch[1]->work << endl;
        return os;
    }
};

} // namespace block2
