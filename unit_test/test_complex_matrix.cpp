
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestComplexMatrix : public ::testing::Test {
  protected:
    static const int n_tests = 100;
    struct MatMul {
        ComplexMatrixRef a;
        MatMul(const ComplexMatrixRef &a) : a(a) {}
        void operator()(const ComplexMatrixRef &b, const ComplexMatrixRef &c) {
            ComplexMatrixFunctions::multiply(a, false, b, false, c, 1.0, 0.0);
        }
    };
    size_t isize = 1L << 24;
    size_t dsize = 1L << 28;
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

TEST_F(TestComplexMatrix, TestIadd) {
    shared_ptr<BatchGEMMSeq<complex<double>>> seq =
        make_shared<BatchGEMMSeq<complex<double>>>(0, SeqTypes::None);
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200), n = Random::rand_int(1, 200);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * n), m, n);
        ComplexMatrixRef c(dalloc_()->complex_allocate(m * n), m, n);
        ComplexMatrixRef b(dalloc_()->complex_allocate(m * n), m, n);
        uint8_t conjb = Random::rand_int(0, 3);
        bool ii = Random::rand_int(0, 2), jj = Random::rand_int(0, 2);
        complex<double> scale, cfactor;
        Random::complex_fill<double>(&scale, 1);
        Random::complex_fill<double>(&cfactor, 1);
        if (ii)
            scale = 1.0;
        if (jj)
            cfactor = 1.0;
        if (conjb == 2)
            cfactor = 1;
        ComplexMatrixRef tb = b;
        if (conjb == 1) {
            tb = ComplexMatrixRef(dalloc_()->complex_allocate(n * m), n, m);
            for (MKL_INT ik = 0; ik < m; ik++)
                for (MKL_INT jk = 0; jk < n; jk++)
                    tb(jk, ik) = conj(b(ik, jk));
        } else if (conjb == 2) {
            tb = ComplexMatrixRef(dalloc_()->complex_allocate(n * m), n, m);
            for (MKL_INT ik = 0; ik < m; ik++)
                for (MKL_INT jk = 0; jk < n; jk++)
                    tb(jk, ik) = b(ik, jk);
        }
        ComplexMatrixFunctions::copy(c, a);
        if (conjb == 2)
            ComplexMatrixFunctions::transpose(c, tb, scale);
        else
            ComplexMatrixFunctions::iadd(c, tb, scale, conjb, cfactor);
        for (MKL_INT ik = 0; ik < m; ik++)
            for (MKL_INT jk = 0; jk < n; jk++)
                ASSERT_LT(
                    abs(cfactor * a(ik, jk) + scale * b(ik, jk) - c(ik, jk)),
                    1E-12);
        if (conjb != 2) {
            ComplexMatrixFunctions::copy(c, a);
            seq->iadd(c, tb, scale, conjb, cfactor);
            seq->simple_perform();
            for (MKL_INT ik = 0; ik < m; ik++)
                for (MKL_INT jk = 0; jk < n; jk++)
                    ASSERT_LT(abs(cfactor * a(ik, jk) + scale * b(ik, jk) -
                                  c(ik, jk)),
                              1E-12);
        }
        if (conjb)
            tb.deallocate();
        b.deallocate();
        c.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestMultiply) {
    shared_ptr<BatchGEMMSeq<complex<double>>> seq =
        make_shared<BatchGEMMSeq<complex<double>>>(0, SeqTypes::None);
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200), n = Random::rand_int(1, 200),
                k = Random::rand_int(1, 200);
        uint8_t conja = Random::rand_int(0, 4), conjb = Random::rand_int(0, 4);
        while (conja == 3 && conjb == 2) {
            conja = Random::rand_int(0, 4);
            conjb = Random::rand_int(0, 4);
        }
        bool exa = Random::rand_int(0, 2), exc = Random::rand_int(0, 2);
        MKL_INT lda = (conja & 1) ? m : k, ldc = n;
        if (exa)
            lda = Random::rand_int(lda, lda + 200);
        if (exc)
            ldc = Random::rand_int(ldc, ldc + 200);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * k), m, k);
        ComplexMatrixRef b(dalloc_()->complex_allocate(k * n), k, n);
        ComplexMatrixRef c(dalloc_()->complex_allocate(m * ldc), m, ldc);
        ComplexMatrixRef cc(dalloc_()->complex_allocate(m * ldc), m, ldc);
        ComplexMatrixRef ta(
            dalloc_()->complex_allocate((conja & 1) ? k * lda : m * lda),
            (conja & 1) ? k : m, lda);
        ComplexMatrixRef tb(dalloc_()->complex_allocate(k * n),
                            (conjb & 1) ? n : k, (conjb & 1) ? k : n);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(b.data, b.size());
        Random::complex_fill<double>(c.data, c.size());
        Random::complex_fill<double>(ta.data, ta.size());
        Random::complex_fill<double>(tb.data, tb.size());
        if (conja & 1)
            for (MKL_INT ik = 0; ik < a.m; ik++)
                for (MKL_INT jk = 0; jk < a.n; jk++)
                    ta(jk, ik) = a(ik, jk);
        else
            for (MKL_INT ik = 0; ik < a.m; ik++)
                for (MKL_INT jk = 0; jk < a.n; jk++)
                    ta(ik, jk) = a(ik, jk);
        if (conja & 2)
            for (MKL_INT ik = 0; ik < ta.m; ik++)
                for (MKL_INT jk = 0; jk < ta.n; jk++)
                    ta(ik, jk) = conj(ta(ik, jk));
        if (conjb & 1)
            for (MKL_INT ik = 0; ik < b.m; ik++)
                for (MKL_INT jk = 0; jk < b.n; jk++)
                    tb(jk, ik) = b(ik, jk);
        else
            for (MKL_INT ik = 0; ik < b.m; ik++)
                for (MKL_INT jk = 0; jk < b.n; jk++)
                    tb(ik, jk) = b(ik, jk);
        if (conjb & 2)
            for (MKL_INT ik = 0; ik < tb.m; ik++)
                for (MKL_INT jk = 0; jk < tb.n; jk++)
                    tb(ik, jk) = conj(tb(ik, jk));
        bool ii = Random::rand_int(0, 2), jj = Random::rand_int(0, 2);
        complex<double> scale, cfactor;
        Random::complex_fill<double>(&scale, 1);
        Random::complex_fill<double>(&cfactor, 1);
        if (ii)
            scale = 1.0;
        if (jj)
            cfactor = 1.0;
        ComplexMatrixFunctions::copy(cc, c);
        ComplexMatrixFunctions::multiply(ta, conja, tb, conjb, cc, scale,
                                         cfactor);
        for (MKL_INT ik = 0; ik < m; ik++)
            for (MKL_INT jk = 0; jk < n; jk++) {
                complex<double> x = cfactor * c(ik, jk);
                for (MKL_INT kk = 0; kk < k; kk++)
                    x += scale * a(ik, kk) * b(kk, jk);
                ASSERT_LT(abs(x - cc(ik, jk)), 1E-8);
            }
        ComplexMatrixFunctions::copy(cc, c);
        seq->multiply(ta, conja, tb, conjb, cc, scale, cfactor);
        seq->simple_perform();
        for (MKL_INT ik = 0; ik < m; ik++)
            for (MKL_INT jk = 0; jk < n; jk++) {
                complex<double> x = cfactor * c(ik, jk);
                for (MKL_INT kk = 0; kk < k; kk++)
                    x += scale * a(ik, kk) * b(kk, jk);
                ASSERT_LT(abs(x - cc(ik, jk)), 1E-8);
            }
        tb.deallocate();
        ta.deallocate();
        cc.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestRotate) {
    shared_ptr<BatchGEMMSeq<complex<double>>> seq =
        make_shared<BatchGEMMSeq<complex<double>>>(0, SeqTypes::None);
    for (int i = 0; i < n_tests; i++) {
        MKL_INT mk = Random::rand_int(1, 200), nk = Random::rand_int(1, 200);
        MKL_INT mb = Random::rand_int(1, 200), nb = Random::rand_int(1, 200);
        ComplexMatrixRef k(dalloc_()->complex_allocate(mk * nk), mk, nk);
        ComplexMatrixRef b(dalloc_()->complex_allocate(mb * nb), mb, nb);
        ComplexMatrixRef a(dalloc_()->complex_allocate(nb * mk), nb, mk);
        ComplexMatrixRef c(dalloc_()->complex_allocate(mb * nk), mb, nk);
        ComplexMatrixRef ba(dalloc_()->complex_allocate(mb * mk), mb, mk);
        Random::complex_fill<double>(k.data, k.size());
        Random::complex_fill<double>(b.data, b.size());
        Random::complex_fill<double>(a.data, a.size());
        uint8_t conjk = Random::rand_int(0, 4);
        uint8_t conjb = Random::rand_int(0, 4);
        ComplexMatrixRef tk = k, tb = b;
        if (conjk == 1) {
            tk = ComplexMatrixRef(dalloc_()->complex_allocate(mk * nk), nk, mk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(jk, ik) = k(ik, jk);
        } else if (conjk == 2) {
            tk = ComplexMatrixRef(dalloc_()->complex_allocate(mk * nk), mk, nk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(ik, jk) = conj(k(ik, jk));
        } else if (conjk == 3) {
            tk = ComplexMatrixRef(dalloc_()->complex_allocate(mk * nk), nk, mk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(jk, ik) = conj(k(ik, jk));
        }
        if (conjb == 1) {
            tb = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        } else if (conjb == 2) {
            tb = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nb), mb, nb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(ib, jb) = conj(b(ib, jb));
        } else if (conjb == 3) {
            tb = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = conj(b(ib, jb));
        }
        c.clear();
        ComplexMatrixFunctions::rotate(a, c, tb, conjb, tk, conjk,
                                       complex<double>(2.0, 1.0));
        ba.clear();
        for (MKL_INT jb = 0; jb < nb; jb++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    ba(ib, ja) +=
                        b(ib, jb) * a(jb, ja) * complex<double>(2.0, 1.0);
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                complex<double> x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), 1E-8);
            }
        // batch gemm does not support both conjb/conjk case
        if (conjb != 2 || conjk != 2) {
            c.clear();
            seq->rotate(a, c, tb, conjb, tk, conjk, complex<double>(2.0, 1.0));
            seq->simple_perform();
            ba.clear();
            for (MKL_INT jb = 0; jb < nb; jb++)
                for (MKL_INT ib = 0; ib < mb; ib++)
                    for (MKL_INT ja = 0; ja < mk; ja++)
                        ba(ib, ja) +=
                            b(ib, jb) * a(jb, ja) * complex<double>(2.0, 1.0);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jk = 0; jk < nk; jk++) {
                    complex<double> x = 0;
                    for (MKL_INT ik = 0; ik < mk; ik++)
                        x += ba(ib, ik) * k(ik, jk);
                    ASSERT_LT(abs(x - c(ib, jk)), 1E-8);
                }
        }
        if (conjb)
            tb.deallocate();
        if (conjk)
            tk.deallocate();
        bool conja = Random::rand_int(0, 2);
        bool conjc = Random::rand_int(0, 2);
        ComplexMatrixRef ta = a, tc = c;
        tb = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nb), nb, mb);
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jb = 0; jb < nb; jb++)
                tb(jb, ib) = conj(b(ib, jb));
        if (conja) {
            ta = ComplexMatrixRef(dalloc_()->complex_allocate(mk * nb), mk, nb);
            for (MKL_INT ia = 0; ia < nb; ia++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    ta(ja, ia) = conj(a(ia, ja));
        }
        c.clear();
        if (conjc) {
            tc = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nk), nk, mb);
            tc.clear();
        }
        ComplexMatrixFunctions::rotate(ta, conja, tc, conjc, tb, k,
                                       complex<double>(2.0, 1.0));
        if (conjc) {
            for (MKL_INT ic = 0; ic < mb; ic++)
                for (MKL_INT jc = 0; jc < nk; jc++)
                    c(ic, jc) = conj(tc(jc, ic));
        }
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                complex<double> x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), 1E-8);
            }
        if (conjc)
            tc.deallocate();
        c.clear();
        if (conjc) {
            tc = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nk), nk, mb);
            tc.clear();
        }
        seq->rotate(ta, conja, tc, conjc, tb, k, complex<double>(2.0, 1.0));
        seq->simple_perform();
        if (conjc) {
            for (MKL_INT ic = 0; ic < mb; ic++)
                for (MKL_INT jc = 0; jc < nk; jc++)
                    c(ic, jc) = conj(tc(jc, ic));
        }
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                complex<double> x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), 1E-8);
            }
        if (conjc)
            tc.deallocate();
        if (conja)
            ta.deallocate();
        tb.deallocate();
        ba.deallocate();
        c.deallocate();
        a.deallocate();
        b.deallocate();
        k.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestTensorProductDiagonal) {
    shared_ptr<BatchGEMMSeq<complex<double>>> seq =
        make_shared<BatchGEMMSeq<complex<double>>>(0, SeqTypes::None);
    for (int i = 0; i < n_tests; i++) {
        MKL_INT ma = Random::rand_int(1, 200), na = ma;
        MKL_INT mb = Random::rand_int(1, 200), nb = mb;
        ComplexMatrixRef a(dalloc_()->complex_allocate(ma * na), ma, na);
        ComplexMatrixRef b(dalloc_()->complex_allocate(mb * nb), mb, nb);
        ComplexMatrixRef c(dalloc_()->complex_allocate(ma * nb), ma, nb);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(b.data, b.size());
        c.clear();
        uint8_t conj = Random::rand_int(0, 4);
        ComplexMatrixFunctions::tensor_product_diagonal(
            conj, a, b, c, complex<double>(2.0, 1.0));
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                ASSERT_LE(abs(complex<double>(2.0, 1.0) *
                                  ((conj & 1) ? xconj(a(ia, ia)) : a(ia, ia)) *
                                  ((conj & 2) ? xconj(b(ib, ib)) : b(ib, ib)) -
                              c(ia, ib)),
                          1E-14);
        c.clear();
        seq->tensor_product_diagonal(conj, a, b, c, complex<double>(2.0, 1.0));
        seq->simple_perform();
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                ASSERT_LE(abs(complex<double>(2.0, 1.0) *
                                  ((conj & 1) ? xconj(a(ia, ia)) : a(ia, ia)) *
                                  ((conj & 2) ? xconj(b(ib, ib)) : b(ib, ib)) -
                              c(ia, ib)),
                          1E-14);
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestThreeRotate) {
    shared_ptr<BatchGEMMSeq<complex<double>>> seq =
        make_shared<BatchGEMMSeq<complex<double>>>(0, SeqTypes::None);
    const int sz = 200;
    for (int i = 0; i < n_tests; i++) {
        MKL_INT ii = Random::rand_int(0, 2), jj = Random::rand_int(0, 2);
        bool ll = Random::rand_int(0, 2);
        MKL_INT mda = Random::rand_int(1, sz), nda = Random::rand_int(1, sz);
        MKL_INT mdb = Random::rand_int(1, sz), ndb = Random::rand_int(1, sz);
        MKL_INT mx = Random::rand_int(1, sz), nx = Random::rand_int(1, sz);
        if (ii == 0)
            mda = nda = 1;
        else if (ii == 1)
            mdb = ndb = 1;
        MKL_INT mdc = mda * mdb * (jj + 1), ndc = nda * ndb * (jj + 1);
        bool dconja = Random::rand_int(0, 2), dconjb = Random::rand_int(0, 2);
        bool conjk = Random::rand_int(0, 2), conjb = Random::rand_int(0, 2);
        ComplexMatrixRef da(dalloc_()->complex_allocate(mda * nda), mda, nda);
        ComplexMatrixRef db(dalloc_()->complex_allocate(mdb * ndb), mdb, ndb);
        ComplexMatrixRef dc(dalloc_()->complex_allocate(mdc * ndc), mdc, ndc);
        ComplexMatrixRef x(dalloc_()->complex_allocate(mx * nx), mx, nx);
        MKL_INT mb = ll ? mdc : mx, nb = ll ? ndc : nx;
        MKL_INT
        mk = ll ? mx : mdc, nk = ll ? nx : ndc;
        ComplexMatrixRef a(dalloc_()->complex_allocate(nb * mk), nb, mk);
        ComplexMatrixRef c(dalloc_()->complex_allocate(mb * nk), mb, nk);
        ComplexMatrixRef cc(dalloc_()->complex_allocate(mb * nk), mb, nk);
        Random::complex_fill<double>(da.data, da.size());
        Random::complex_fill<double>(db.data, db.size());
        Random::complex_fill<double>(a.data, a.size());
        MKL_INT dcm_stride = Random::rand_int(0, mdc - mda * mdb + 1);
        MKL_INT dcn_stride = Random::rand_int(0, ndc - nda * ndb + 1);
        if (conjb)
            ll ? (dc = dc.flip_dims()) : (x = x.flip_dims());
        if (conjk)
            !ll ? (dc = dc.flip_dims()) : (x = x.flip_dims());
        if (ll ? conjb : conjk)
            swap(dcm_stride, dcn_stride);
        if (dconja ^ (ll ? conjb : conjk))
            da = da.flip_dims();
        if (dconjb ^ (ll ? conjb : conjk))
            db = db.flip_dims();
        MKL_INT dc_stride = dcm_stride * dc.n + dcn_stride;
        c.clear();
        ComplexMatrixFunctions::three_rotate(
            a, c, ll ? dc : x, conjb, ll ? x : dc, conjk, da, dconja, db,
            dconjb, ll, complex<double>(2.0, 1.0), dc_stride);
        dc.clear(), cc.clear();
        ComplexMatrixFunctions::tensor_product(da, dconja, db, dconjb, dc, 1.0,
                                               dc_stride);
        ComplexMatrixFunctions::rotate(a, cc, ll ? dc : x, conjb ? 3 : 0,
                                       ll ? x : dc, conjk ? 1 : 2,
                                       complex<double>(2.0, 1.0));
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-10, 1E-10));
        c.clear();
        seq->three_rotate(a, c, ll ? dc : x, conjb, ll ? x : dc, conjk, da,
                          dconja, db, dconjb, ll, complex<double>(2.0, 1.0),
                          dc_stride);
        seq->simple_perform();
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-10, 1E-10));
        cc.deallocate();
        c.deallocate();
        a.deallocate();
        x.deallocate();
        dc.deallocate();
        db.deallocate();
        da.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestThreeTensorProductDiagonal) {
    shared_ptr<BatchGEMM<complex<double>>> batch =
        make_shared<BatchGEMM<complex<double>>>();
    for (int i = 0; i < n_tests; i++) {
        MKL_INT ii = Random::rand_int(0, 2), jj = Random::rand_int(0, 2);
        MKL_INT ll = Random::rand_int(0, 2);
        MKL_INT mda = Random::rand_int(1, 400), nda = mda;
        MKL_INT mdb = Random::rand_int(1, 400), ndb = mdb;
        MKL_INT mx = Random::rand_int(1, 400), nx = mx;
        if (ii == 0)
            mda = nda = 1;
        else if (ii == 1)
            mdb = ndb = 1;
        MKL_INT mdc = mda * mdb * (jj + 1), ndc = nda * ndb * (jj + 1);
        bool dconja = Random::rand_int(0, 2), dconjb = Random::rand_int(0, 2);
        ComplexMatrixRef da(dalloc_()->complex_allocate(mda * nda), mda, nda);
        ComplexMatrixRef db(dalloc_()->complex_allocate(mdb * ndb), mdb, ndb);
        ComplexMatrixRef dc(dalloc_()->complex_allocate(mdc * ndc), mdc, ndc);
        ComplexMatrixRef x(dalloc_()->complex_allocate(mx * nx), mx, nx);
        ComplexMatrixRef c(dalloc_()->complex_allocate(mdc * mx), mdc, mx);
        ComplexMatrixRef cc(dalloc_()->complex_allocate(mdc * mx), mdc, mx);
        if (!ll)
            c = c.flip_dims(), cc = cc.flip_dims();
        Random::complex_fill<double>(da.data, da.size());
        Random::complex_fill<double>(db.data, db.size());
        MKL_INT dcm_stride = Random::rand_int(0, mdc - mda * mdb + 1);
        MKL_INT dcn_stride = Random::rand_int(0, ndc - nda * ndb + 1);
        dcn_stride = dcm_stride;
        MKL_INT dc_stride = dcm_stride * dc.n + dcn_stride;
        c.clear();
        uint8_t conj = Random::rand_int(0, 4);
        ComplexMatrixFunctions::three_tensor_product_diagonal(
            conj, ll ? dc : x, ll ? x : dc, c, da, dconja, db, dconjb, ll,
            complex<double>(2.0, 1.0), dc_stride);
        dc.clear(), cc.clear();
        ComplexMatrixFunctions::tensor_product(da, dconja, db, dconjb, dc, 1.0,
                                               dc_stride);
        ComplexMatrixFunctions::tensor_product_diagonal(
            conj, ll ? dc : x, ll ? x : dc, cc, complex<double>(2.0, 1.0));
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-8, 1E-8));
        c.clear();
        AdvancedGEMM<complex<double>>::three_tensor_product_diagonal(
            batch, conj, ll ? dc : x, ll ? x : dc, c, da, dconja, db, dconjb,
            ll, complex<double>(2.0, 1.0), dc_stride);
        batch->perform();
        batch->clear();
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-8, 1E-8));
        cc.deallocate();
        c.deallocate();
        x.deallocate();
        dc.deallocate();
        db.deallocate();
        da.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestTensorProduct) {
    shared_ptr<BatchGEMM<complex<double>>> batch =
        make_shared<BatchGEMM<complex<double>>>();
    for (int i = 0; i < n_tests; i++) {
        MKL_INT ii = Random::rand_int(0, 4), jj = Random::rand_int(0, 2);
        MKL_INT ma = Random::rand_int(1, 700), na = Random::rand_int(1, 700);
        MKL_INT mb = Random::rand_int(1, 700), nb = Random::rand_int(1, 700);
        if (ii == 0)
            ma = na = 1;
        else if (ii == 1)
            mb = nb = 1;
        else if (ii == 2) {
            ma = Random::rand_int(1, 20), na = Random::rand_int(1, 20);
            mb = Random::rand_int(1, 7), nb = Random::rand_int(1, 7);
        } else {
            ma = Random::rand_int(1, 7), na = Random::rand_int(1, 7);
            mb = Random::rand_int(1, 20), nb = Random::rand_int(1, 20);
        }
        MKL_INT mc = ma * mb * (jj + 1), nc = na * nb * (jj + 1);
        ComplexMatrixRef a(dalloc_()->complex_allocate(ma * na), ma, na);
        ComplexMatrixRef b(dalloc_()->complex_allocate(mb * nb), mb, nb);
        ComplexMatrixRef c(dalloc_()->complex_allocate(mc * nc), mc, nc);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(b.data, b.size());
        c.clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        ComplexMatrixRef ta = a, tb = b;
        if (conja) {
            ta = ComplexMatrixRef(dalloc_()->complex_allocate(ma * na), na, ma);
            for (MKL_INT ia = 0; ia < ma; ia++)
                for (MKL_INT ja = 0; ja < na; ja++)
                    ta(ja, ia) = conj(a(ia, ja));
        }
        if (conjb) {
            tb = ComplexMatrixRef(dalloc_()->complex_allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = conj(b(ib, jb));
        }
        MKL_INT cm_stride = Random::rand_int(0, mc - ma * mb + 1);
        MKL_INT cn_stride = Random::rand_int(0, nc - na * nb + 1);
        MKL_INT c_stride = cm_stride * c.n + cn_stride;
        if (Random::rand_int(0, 2))
            ComplexMatrixFunctions::tensor_product(
                ta, conja, tb, conjb, c, complex<double>(2.0, 1.0), c_stride);
        else {
            AdvancedGEMM<complex<double>>::tensor_product(
                batch, ta, conja, tb, conjb, c, complex<double>(2.0, 1.0),
                c_stride);
            batch->perform();
            batch->clear();
        }
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ja = 0; ja < na; ja++)
                for (MKL_INT ib = 0; ib < mb; ib++)
                    for (MKL_INT jb = 0; jb < nb; jb++)
                        ASSERT_LE(abs(complex<double>(2.0, 1.0) * a(ia, ja) *
                                          b(ib, jb) -
                                      c(ia * mb + ib + cm_stride,
                                        ja * nb + jb + cn_stride)),
                                  1E-14);
        if (conjb)
            tb.deallocate();
        if (conja)
            ta.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestExponentialPade) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(1, 200);
        int ideg = 6;
        double t = Random::rand_double(-0.1, 0.1);
        ComplexMatrixRef a(dalloc_()->complex_allocate(n * n), n, n);
        ComplexMatrixRef v(dalloc_()->complex_allocate(n), n, 1);
        ComplexMatrixRef u(dalloc_()->complex_allocate(n), n, 1);
        ComplexMatrixRef w(dalloc_()->complex_allocate(n), n, 1);
        ComplexMatrixRef work(dalloc_()->complex_allocate(4 * n * n + ideg + 1),
                              4 * n * n + ideg + 1, 1);
        Random::complex_fill<double>(a.data, a.size());
        // note that eigs can only work on hermitian matrix
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = conj(a(ki, kj));
            a(ki, ki) = real(a(ki, ki));
        }
        Random::complex_fill<double>(v.data, v.size());
        for (MKL_INT ki = 0; ki < n; ki++)
            w(ki, 0) = v(ki, 0);
        auto x = IterativeMatrixFunctions<complex<double>>::expo_pade(
            ideg, n, a.data, n, t, work.data);
        ComplexMatrixFunctions::multiply(
            ComplexMatrixRef(work.data + x.first, n, n), false, v, false, w,
            1.0, 0.0);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        ComplexMatrixFunctions::eigs(a, ww);
        ComplexMatrixFunctions::multiply(v, true, a, 3, u.flip_dims(), 1.0,
                                         0.0);
        for (MKL_INT i = 0; i < n; i++)
            v(i, 0) = exp(t * ww(i, i)) * u(i, 0);
        ComplexMatrixFunctions::multiply(v, true, a, false, u.flip_dims(), 1.0,
                                         0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(u, w, 1E-6, 0.0));
        ww.deallocate();
        work.deallocate();
        w.deallocate();
        u.deallocate();
        v.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestExponential) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(1, 300);
        bool symm = Random::rand_int(0, 2);
        complex<double> t(Random::rand_double(-0.1, 0.1),
                          symm ? 0.0 : Random::rand_double(-0.1, 0.1));
        double consta = Random::rand_double(-2.0, 2.0);
        ComplexMatrixRef a(dalloc_()->complex_allocate(n * n), n, n);
        ComplexMatrixRef aa(dalloc_()->complex_allocate(n), n, 1);
        ComplexMatrixRef v(dalloc_()->complex_allocate(n), n, 1);
        ComplexMatrixRef u(dalloc_()->complex_allocate(n), n, 1);
        ComplexMatrixRef w(dalloc_()->complex_allocate(n), n, 1);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(v.data, v.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = conj(a(kj, ki));
            a(ki, ki) = real(a(ki, ki));
            aa(ki, 0) = a(ki, ki);
            w(ki, 0) = v(ki, 0);
        }
        double anorm = ComplexMatrixFunctions::norm(a);
        MatMul mop(a);
        int nmult = IterativeMatrixFunctions<complex<double>>::expo_apply(
            mop, t, anorm, w, consta, symm, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        ComplexMatrixFunctions::eigs(a, ww);
        ComplexMatrixFunctions::multiply(v, true, a, 3, u.flip_dims(), 1.0,
                                         0.0);
        for (MKL_INT i = 0; i < n; i++)
            v(i, 0) = exp(t * (ww(i, i) + consta)) * u(i, 0);
        ComplexMatrixFunctions::multiply(v, true, a, false, u.flip_dims(), 1.0,
                                         0.0);
        EXPECT_TRUE(MatrixFunctions::all_close(u, w, 1E-6, 1E-6));
        ww.deallocate();
        w.deallocate();
        u.deallocate();
        v.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestHarmonicDavidson) {
    // this test is not very stable
    Random::rand_seed(1234);
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(3, 50);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 5));
        if (k > n / 2)
            k = n / 2;
        int ndav = 0;
        ComplexMatrixRef a(dalloc_()->complex_allocate(n * n), n, n);
        ComplexDiagonalMatrix aa(dalloc_()->complex_allocate(n), n);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        vector<ComplexMatrixRef> bs(k, ComplexMatrixRef(nullptr, n, 1));
        Random::complex_fill<double>(a.data, a.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = conj(a(ki, kj));
            a(ki, ki) = real(a(ki, ki));
            aa(ki, ki) = a(ki, ki);
        }
        for (int i = 0; i < k; i++) {
            bs[i].allocate();
            bs[i].clear();
            bs[i].data[i] = 1;
        }
        MatMul mop(a);
        double shift = 0.1;
        DavidsonTypes davidson_type =
            Random::rand_int(0, 2)
                ? DavidsonTypes::HarmonicLessThan | DavidsonTypes::NoPrecond
                : DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond;
        vector<double> vw =
            IterativeMatrixFunctions<complex<double>>::harmonic_davidson(
                mop, aa, bs, shift, davidson_type, ndav, false,
                (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-9,
                n * k * 500, -1, 2, 35);
        ASSERT_EQ((int)vw.size(), k);
        DiagonalMatrix w(&vw[0], k);
        ComplexMatrixFunctions::eigs(a, ww);
        vector<int> eigval_idxs(ww.size());
        for (int i = 0; i < (int)ww.size(); i++)
            eigval_idxs[i] = i;
        if (davidson_type & DavidsonTypes::CloseTo)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     return abs(ww.data[i] - shift) < abs(ww.data[j] - shift);
                 });
        else if (davidson_type & DavidsonTypes::LessThan)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     if ((shift >= ww.data[i]) != (shift >= ww.data[j]))
                         return shift >= ww.data[i];
                     else if (shift >= ww.data[i])
                         return shift - ww.data[i] < shift - ww.data[j];
                     else
                         return ww.data[i] - shift > ww.data[j] - shift;
                 });
        else if (davidson_type & DavidsonTypes::GreaterThan)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     if ((shift > ww.data[i]) != (shift > ww.data[j]))
                         return shift > ww.data[j];
                     else if (shift > ww.data[i])
                         return shift - ww.data[i] > shift - ww.data[j];
                     else
                         return ww.data[i] - shift < ww.data[j] - shift;
                 });
        // last root may be inaccurate (rare)
        for (int i = 0; i < k - 1; i++)
            ASSERT_LT(abs(ww.data[eigval_idxs[i]] - vw[i]), 1E-6);
        for (int i = 0; i < k - 1; i++) {
            complex<double> factor = 0.0;
            for (int j = 0; j < a.n; j++)
                if (abs(bs[i].data[j]) > 1E-10)
                    factor += a.data[a.n * eigval_idxs[i] + j] / bs[i].data[j];
            factor = factor / abs(factor);
            ASSERT_LE(abs(factor) - 1.0, 1E-3);
            ASSERT_TRUE(ComplexMatrixFunctions::all_close(
                ComplexMatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1), bs[i],
                1E-3, 0, factor));
        }
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
    Random::rand_seed(0);
}

TEST_F(TestComplexMatrix, TestDavidson) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 10));
        int ndav = 0;
        ComplexMatrixRef a(dalloc_()->complex_allocate(n * n), n, n);
        ComplexDiagonalMatrix aa(dalloc_()->complex_allocate(n), n);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        vector<ComplexMatrixRef> bs(k, ComplexMatrixRef(nullptr, n, 1));
        Random::complex_fill<double>(a.data, a.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = conj(a(ki, kj));
            a(ki, ki) = real(a(ki, ki));
            aa(ki, ki) = a(ki, ki);
        }
        for (int i = 0; i < k; i++) {
            bs[i].allocate();
            bs[i].clear();
            bs[i].data[i] = 1;
        }
        MatMul mop(a);
        vector<double> vw = IterativeMatrixFunctions<complex<double>>::davidson(
            mop, aa, bs, 0, DavidsonTypes::Normal, ndav, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8, n * k * 2, -1,
            k * 2, max((MKL_INT)5, k + 10));
        ASSERT_EQ((int)vw.size(), k);
        DiagonalMatrix w(&vw[0], k);
        ComplexMatrixFunctions::eigs(a, ww);
        DiagonalMatrix w2(ww.data, k);
        ASSERT_TRUE(MatrixFunctions::all_close(w, w2, 1E-6, 1E-6));
        for (int i = 0; i < k; i++) {
            complex<double> factor = 0.0;
            for (int j = 0; j < a.n; j++)
                if (abs(bs[i].data[j]) > 1E-10)
                    factor += a.data[a.n * i + j] / bs[i].data[j];
            factor = factor / abs(factor);
            ASSERT_LE(abs(factor) - 1.0, 1E-3);
            ASSERT_TRUE(ComplexMatrixFunctions::all_close(
                ComplexMatrixRef(a.data + a.n * i, a.n, 1), bs[i], 1E-3, 0,
                factor));
        }
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestLinear) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef bg(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), n, m);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(b.data, b.size());
        ComplexMatrixFunctions::copy(af, a);
        ComplexMatrixFunctions::copy(x, b);
        ComplexMatrixFunctions::linear(af, x);
        // note that linear solves the H^T problem
        ComplexMatrixFunctions::multiply(x, false, a, false, bg, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(bg, b, 1E-9, 1E-8));
        x.deallocate();
        bg.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestCG) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0;
        complex<double> eta = 0.05;
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexDiagonalMatrix aa(dalloc_()->complex_allocate(m), m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), n, m);
        Random::complex_fill<double>(af.data, af.size());
        Random::complex_fill<double>(b.data, b.size());
        Random::complex_fill<double>(xg.data, xg.size());
        ComplexMatrixFunctions::multiply(af, false, af, 3, a, 1.0, 0.0);
        for (MKL_INT ki = 0; ki < m; ki++)
            a(ki, ki) += eta, aa(ki, ki) = a(ki, ki);
        af.clear();
        ComplexMatrixFunctions::transpose(af, a, 1.0);
        ComplexMatrixFunctions::copy(x, b);
        // note that linear solves the H^T problem
        ComplexMatrixFunctions::linear(af, x);
        MatMul mop(a);
        complex<double> func =
            IterativeMatrixFunctions<complex<double>>::conjugate_gradient(
                mop, aa, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
                (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-14, 5000);
        ASSERT_TRUE(MatrixFunctions::all_close(xg, x, 1E-3, 1E-3));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        aa.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestMinRes) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0;
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), n, m);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(b.data, b.size());
        Random::complex_fill<double>(xg.data, xg.size());
        for (MKL_INT ki = 0; ki < m; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = conj(a(kj, ki));
            a(ki, ki) = real(a(ki, ki));
        }
        af.clear();
        ComplexMatrixFunctions::transpose(af, a, 1.0);
        ComplexMatrixFunctions::copy(x, b);
        // note that linear solves the H^T problem
        ComplexMatrixFunctions::linear(af, x);
        MatMul mop(a);
        IterativeMatrixFunctions<complex<double>>::minres(
            mop, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-14, 5000);
        ASSERT_TRUE(MatrixFunctions::all_close(xg, x, 1E-3, 0.0));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestInverse) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ap(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ag(dalloc_()->complex_allocate(m * m), m, m);
        Random::complex_fill<double>(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++)
            for (MKL_INT kj = 0; kj < m; kj++)
                ap(ki, kj) = a(ki, kj);
        ComplexMatrixFunctions::inverse(a);
        ComplexMatrixFunctions::multiply(a, false, ap, false, ag, 1.0, 0.0);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                EXPECT_LT(abs(ag(j, k) - (k == j ? 1.0 : 0.0)), 1E-9);
        ag.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestDeflatedCG) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0;
        double eta = 0.05;
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ag(dalloc_()->complex_allocate(m * m), m, m);
        ComplexDiagonalMatrix aa(dalloc_()->complex_allocate(m), m);
        DiagonalMatrix ww(dalloc_()->allocate(m), m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), n, m);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), n, m);
        Random::complex_fill<double>(af.data, af.size());
        Random::complex_fill<double>(b.data, b.size());
        Random::complex_fill<double>(xg.data, xg.size());
        ComplexMatrixFunctions::multiply(af, false, af, 3, a, 1.0, 0.0);
        for (MKL_INT ki = 0; ki < m; ki++)
            a(ki, ki) += eta;
        ComplexMatrixFunctions::copy(ag, a);
        ComplexMatrixFunctions::eigs(ag, ww);
        ComplexMatrixRef w(ag.data, m, 1);
        af.clear();
        ComplexMatrixFunctions::transpose(af, a, 1.0);
        ComplexMatrixFunctions::copy(x, b);
        ComplexMatrixFunctions::linear(af, x);
        MatMul mop(a);
        for (MKL_INT ki = 0; ki < m; ki++)
            aa(ki, ki) = a(ki, ki);
        complex<double> func = IterativeMatrixFunctions<complex<double>>::
            deflated_conjugate_gradient(
                mop, aa, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
                (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-14, 5000, -1,
                vector<ComplexMatrixRef>{w});
        ASSERT_TRUE(MatrixFunctions::all_close(xg, x, 1E-4, 1E-4));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        ww.deallocate();
        aa.deallocate();
        ag.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestEigs) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ap(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ag(dalloc_()->complex_allocate(m * m), m, m);
        DiagonalMatrix w(dalloc_()->allocate(m), m);
        Random::complex_fill<double>(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = conj(a(kj, ki));
            a(ki, ki) = real(a(ki, ki));
        }
        ComplexMatrixFunctions::copy(ap, a);
        ComplexMatrixFunctions::eigs(a, w);
        ComplexMatrixFunctions::multiply(a, false, ap, true, ag, 1.0, 0.0);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                ag(k, j) /= w(k, k);
        ASSERT_TRUE(MatrixFunctions::all_close(ag, a, 1E-9, 1E-8));
        w.deallocate();
        ag.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestEig) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 150);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ap(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef aq(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef ag(dalloc_()->complex_allocate(m * m), m, m);
        ComplexDiagonalMatrix w(dalloc_()->complex_allocate(m), m);
        Random::complex_fill<double>(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++)
            for (MKL_INT kj = 0; kj < m; kj++)
                ap(ki, kj) = a(ki, kj);
        ComplexMatrixFunctions::eig(a, w);
        ComplexMatrixFunctions::multiply(a, false, ap, true, ag, 1.0, 0.0);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                ag(k, j) /= w(k, k);
        EXPECT_TRUE(MatrixFunctions::all_close(ag, a, 1E-9, 0.0));
        // a[i, j] u[k, j] = w[k, k] u[k, i]
        // a[i, j] = uinv[j, k] w[k, k] u[k, i] = uinv[j, k] a[i, jp] u[k, jp]
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                ag(k, j) = a(k, j) * w(k, k);
        ComplexMatrixFunctions::inverse(a);
        ComplexMatrixFunctions::multiply(ag, true, a, true, aq, 1.0, 0.0);
        EXPECT_TRUE(MatrixFunctions::all_close(ap, aq, 1E-9, 0.0));
        w.deallocate();
        ag.deallocate();
        aq.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestSVD) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(m, n);
        shared_ptr<ComplexTensor> a =
            make_shared<ComplexTensor>(vector<MKL_INT>{m, n});
        shared_ptr<ComplexTensor> l =
            make_shared<ComplexTensor>(vector<MKL_INT>{m, k});
        shared_ptr<Tensor> s = make_shared<Tensor>(vector<MKL_INT>{k});
        shared_ptr<ComplexTensor> r =
            make_shared<ComplexTensor>(vector<MKL_INT>{k, n});
        shared_ptr<ComplexTensor> aa =
            make_shared<ComplexTensor>(vector<MKL_INT>{m, n});
        shared_ptr<ComplexTensor> kk =
            make_shared<ComplexTensor>(vector<MKL_INT>{k, k});
        shared_ptr<ComplexTensor> ll =
            make_shared<ComplexTensor>(vector<MKL_INT>{m, m});
        shared_ptr<ComplexTensor> rr =
            make_shared<ComplexTensor>(vector<MKL_INT>{n, n});
        Random::complex_fill<double>(a->data.data(), a->size());
        ComplexMatrixFunctions::copy(aa->ref(), a->ref());
        if (Random::rand_int(0, 2))
            ComplexMatrixFunctions::accurate_svd(
                a->ref(), l->ref(), s->ref().flip_dims(), r->ref(), 1E-1);
        else
            ComplexMatrixFunctions::svd(a->ref(), l->ref(),
                                        s->ref().flip_dims(), r->ref());
        ComplexMatrixFunctions::multiply(l->ref(), 3, l->ref(), false,
                                         kk->ref(), 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(kk->ref(), IdentityMatrix(k),
                                               1E-12, 0.0));
        ComplexMatrixFunctions::multiply(r->ref(), false, r->ref(), 3,
                                         kk->ref(), 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(kk->ref(), IdentityMatrix(k),
                                               1E-12, 0.0));
        if (m <= n) {
            ComplexMatrixFunctions::multiply(l->ref(), false, l->ref(), 3,
                                             ll->ref(), 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(ll->ref(), IdentityMatrix(m),
                                                   1E-12, 0.0));
        }
        if (n <= m) {
            ComplexMatrixFunctions::multiply(r->ref(), 3, r->ref(), false,
                                             rr->ref(), 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(rr->ref(), IdentityMatrix(n),
                                                   1E-12, 0.0));
        }
        ComplexMatrixRef x(r->data.data(), 1, n);
        for (MKL_INT i = 0; i < k; i++) {
            ASSERT_GE((*s)({i}), 0.0);
            ComplexMatrixFunctions::iscale(x.shift_ptr(i * n), (*s)({i}));
        }
        ComplexMatrixFunctions::multiply(l->ref(), false, r->ref(), false,
                                         a->ref(), 1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(aa->ref(), a->ref(), 1E-12, 0.0));
    }
}

TEST_F(TestComplexMatrix, TestQR) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(m, n);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * n), m, n);
        ComplexMatrixRef qr(dalloc_()->complex_allocate(m * n), m, n);
        Random::complex_fill<double>(a.data, a.size());
        ComplexMatrixRef q(dalloc_()->complex_allocate(m * k), m, k);
        ComplexMatrixRef qq(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef qqk(dalloc_()->complex_allocate(k * k), k, k);
        ComplexMatrixRef r(dalloc_()->complex_allocate(n * k), k, n);
        ComplexMatrixFunctions::qr(a, q, r);
        ComplexMatrixFunctions::multiply(q, false, r, false, qr, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(a, qr, 1E-12, 0.0));
        ComplexMatrixFunctions::multiply(q, 3, q, false, qqk, 1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(qqk, IdentityMatrix(k), 1E-12, 0.0));
        if (m <= n) {
            ComplexMatrixFunctions::multiply(q, false, q, 3, qq, 1.0, 0.0);
            ASSERT_TRUE(
                MatrixFunctions::all_close(qq, IdentityMatrix(m), 1E-12, 0.0));
        }
        r.deallocate();
        qqk.deallocate();
        qq.deallocate();
        q.deallocate();
        qr.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestLQ) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(m, n);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * n), m, n);
        ComplexMatrixRef lq(dalloc_()->complex_allocate(m * n), m, n);
        Random::complex_fill<double>(a.data, a.size());
        ComplexMatrixRef l(dalloc_()->complex_allocate(m * k), m, k);
        ComplexMatrixRef q(dalloc_()->complex_allocate(n * k), k, n);
        ComplexMatrixRef qq(dalloc_()->complex_allocate(n * n), n, n);
        ComplexMatrixRef qqk(dalloc_()->complex_allocate(k * k), k, k);
        ComplexMatrixFunctions::lq(a, l, q);
        ComplexMatrixFunctions::multiply(l, false, q, false, lq, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(a, lq, 1E-12, 0.0));
        ComplexMatrixFunctions::multiply(q, false, q, 3, qqk, 1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(qqk, IdentityMatrix(k), 1E-12, 0.0));
        if (m >= n) {
            ComplexMatrixFunctions::multiply(q, 3, q, false, qq, 1.0, 0.0);
            ASSERT_TRUE(
                MatrixFunctions::all_close(qq, IdentityMatrix(n), 1E-12, 0.0));
        }
        qqk.deallocate();
        qq.deallocate();
        q.deallocate();
        l.deallocate();
        lq.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestLeastSquares) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, m + 1);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * n), m, n);
        ComplexMatrixRef b(dalloc_()->complex_allocate(m), m, 1);
        ComplexMatrixRef br(dalloc_()->complex_allocate(m), m, 1);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n), n, 1);
        Random::complex_fill<double>(a.data, a.size());
        Random::complex_fill<double>(b.data, b.size());
        double res = ComplexMatrixFunctions::least_squares(a, b, x);
        ComplexMatrixFunctions::multiply(a, false, x, false, br, 1.0, 0.0);
        ComplexMatrixFunctions::iadd(br, b, -1);
        double cres = ComplexMatrixFunctions::norm(br);
        EXPECT_LT(abs(res - cres), 1E-9);
        x.deallocate();
        br.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestGCROT) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 300);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        double eta = 0.05;
        MatrixRef ra(dalloc_()->allocate(m * m), m, m);
        MatrixRef rax(dalloc_()->allocate(m * m), m, m);
        MatrixRef rb(dalloc_()->allocate(n * m), m, n);
        MatrixRef rbg(dalloc_()->allocate(n * m), m, n);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), m, n);
        Random::fill<double>(ra.data, ra.size());
        Random::fill<double>(rax.data, rax.size());
        Random::fill<double>(rb.data, rb.size());
        a.clear();
        b.clear();
        MatrixFunctions::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        ComplexMatrixFunctions::fill_complex(a, ra, MatrixRef(nullptr, m, m));
        for (MKL_INT k = 0; k < m; k++)
            a(k, k) += complex<double>(0, eta);
        ComplexMatrixFunctions::fill_complex(b, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, MatrixRef(nullptr, m, n), rb);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        MatMul mop(a);
        complex<double> func =
            IterativeMatrixFunctions<complex<double>>::gcrotmk(
                mop, ComplexDiagonalMatrix(nullptr, 0), x, b, nmult, niter, 20,
                -1, 0.0, false, (shared_ptr<ParallelCommunicator<SZ>>)nullptr,
                1E-14, 10000);
        ComplexMatrixFunctions::copy(xg, b);
        ComplexMatrixFunctions::linear(af, xg.flip_dims());
        ComplexMatrixFunctions::extract_complex(xg, rbg,
                                                MatrixRef(nullptr, m, n));
        ComplexMatrixFunctions::extract_complex(x, rb,
                                                MatrixRef(nullptr, m, n));
        EXPECT_TRUE(MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3));
        ComplexMatrixFunctions::extract_complex(xg, MatrixRef(nullptr, m, n),
                                                rbg);
        ComplexMatrixFunctions::extract_complex(x, MatrixRef(nullptr, m, n),
                                                rb);
        EXPECT_TRUE(MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
        rbg.deallocate();
        rb.deallocate();
        rax.deallocate();
        ra.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestIDRS) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 300);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        double eta = 0.05;
        MatrixRef ra(dalloc_()->allocate(m * m), m, m);
        MatrixRef rax(dalloc_()->allocate(m * m), m, m);
        MatrixRef rb(dalloc_()->allocate(n * m), m, n);
        MatrixRef rbg(dalloc_()->allocate(n * m), m, n);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), m, n);
        Random::fill<double>(ra.data, ra.size());
        Random::fill<double>(rax.data, rax.size());
        Random::fill<double>(rb.data, rb.size());
        a.clear();
        b.clear();
        MatrixFunctions::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        ComplexMatrixFunctions::fill_complex(a, ra, MatrixRef(nullptr, m, m));
        for (MKL_INT k = 0; k < m; k++)
            a(k, k) += complex<double>(0, eta);
        ComplexMatrixFunctions::fill_complex(b, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, MatrixRef(nullptr, m, n), rb);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        MatMul mop(a);
        complex<double> func = IterativeMatrixFunctions<complex<double>>::idrs(
            mop, ComplexDiagonalMatrix(nullptr, 0), x, b, nmult, niter, 8,
            false, (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8, 0.0,
            1E-7, 10000);
        ComplexMatrixFunctions::copy(xg, b);
        ComplexMatrixFunctions::linear(af, xg.flip_dims());
        ComplexMatrixFunctions::extract_complex(xg, rbg,
                                                MatrixRef(nullptr, m, n));
        ComplexMatrixFunctions::extract_complex(x, rb,
                                                MatrixRef(nullptr, m, n));
        EXPECT_TRUE(MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3));
        ComplexMatrixFunctions::extract_complex(xg, MatrixRef(nullptr, m, n),
                                                rbg);
        ComplexMatrixFunctions::extract_complex(x, MatrixRef(nullptr, m, n),
                                                rb);
        EXPECT_TRUE(MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
        rbg.deallocate();
        rb.deallocate();
        rax.deallocate();
        ra.deallocate();
    }
}

TEST_F(TestComplexMatrix, TestLSQR) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        double eta = 0.05;
        MatrixRef ra(dalloc_()->allocate(m * m), m, m);
        MatrixRef rax(dalloc_()->allocate(m * m), m, m);
        MatrixRef rb(dalloc_()->allocate(n * m), m, n);
        MatrixRef rbg(dalloc_()->allocate(n * m), m, n);
        ComplexMatrixRef a(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef af(dalloc_()->complex_allocate(m * m), m, m);
        ComplexMatrixRef b(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef x(dalloc_()->complex_allocate(n * m), m, n);
        ComplexMatrixRef xg(dalloc_()->complex_allocate(n * m), m, n);
        Random::fill<double>(ra.data, ra.size());
        Random::fill<double>(rax.data, rax.size());
        Random::fill<double>(rb.data, rb.size());
        a.clear();
        b.clear();
        MatrixFunctions::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        ComplexMatrixFunctions::fill_complex(a, ra, MatrixRef(nullptr, m, m));
        for (MKL_INT k = 0; k < m; k++)
            a(k, k) += complex<double>(0, eta);
        ComplexMatrixFunctions::fill_complex(b, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, rb, MatrixRef(nullptr, m, n));
        Random::fill<double>(rb.data, rb.size());
        ComplexMatrixFunctions::fill_complex(x, MatrixRef(nullptr, m, n), rb);
        ComplexMatrixFunctions::copy(af, a);
        ComplexMatrixFunctions::conjugate(af);
        MatMul mop(a), rop(af);
        // hrl: Note: The input matrix can be highly illconditioned (cond~10^5)
        //      which causes problems for lsqr.
        //      It is important to have long maxiters and small atol.
        //      It may still fail in extreme situations,
        //      in particular when m ~ 300.
        complex<double> func = IterativeMatrixFunctions<complex<double>>::lsqr(
            mop, rop, ComplexDiagonalMatrix(nullptr, 0), x, b, nmult, niter,
            false, (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8,
            /*rtol*/ 1E-7, /*atol*/ 0., 10000);
        ComplexMatrixFunctions::copy(xg, b);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        ComplexMatrixFunctions::linear(af, xg.flip_dims());
        ComplexMatrixFunctions::extract_complex(xg, rbg,
                                                MatrixRef(nullptr, m, n));
        ComplexMatrixFunctions::extract_complex(x, rb,
                                                MatrixRef(nullptr, m, n));
        EXPECT_TRUE(MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3));
        ComplexMatrixFunctions::extract_complex(xg, MatrixRef(nullptr, m, n),
                                                rbg);
        ComplexMatrixFunctions::extract_complex(x, MatrixRef(nullptr, m, n),
                                                rb);
        EXPECT_TRUE(MatrixFunctions::all_close(rbg, rb, 1E-3, 1E-3));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
        rbg.deallocate();
        rb.deallocate();
        rax.deallocate();
        ra.deallocate();
    }
}
