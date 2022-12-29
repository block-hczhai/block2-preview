
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestComplexMatrix : public ::testing::Test {
  protected:
    typedef typename GMatrix<FL>::FP FP;
    static const int n_tests = 100;
    struct MatMul {
        GMatrix<FL> a;
        MatMul(const GMatrix<FL> &a) : a(a) {}
        void operator()(const GMatrix<FL> &b, const GMatrix<FL> &c) {
            GMatrixFunctions<FL>::multiply(a, false, b, false, c, 1.0, 0.0);
        }
    };
    size_t isize = 1L << 24;
    size_t dsize = 1L << 28;
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

#ifdef _USE_SINGLE_PREC
typedef ::testing::Types<complex<float>, complex<double>> TestFL;
#else
typedef ::testing::Types<complex<double>> TestFL;
#endif

TYPED_TEST_CASE(TestComplexMatrix, TestFL);

TYPED_TEST(TestComplexMatrix, TestIadd) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-12 : 1E-6;
    shared_ptr<BatchGEMMSeq<FL>> seq =
        make_shared<BatchGEMMSeq<FL>>(0, SeqTypes::None);
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200), n = Random::rand_int(1, 200);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * n), m, n);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(m * n), m, n);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(m * n), m, n);
        uint8_t conjb = Random::rand_int(0, 3);
        bool ii = Random::rand_int(0, 2), jj = Random::rand_int(0, 2);
        FL scale, cfactor;
        Random::complex_fill<FP>(&scale, 1);
        Random::complex_fill<FP>(&cfactor, 1);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        if (ii)
            scale = 1.0;
        if (jj)
            cfactor = 1.0;
        if (conjb == 2)
            cfactor = 1;
        GMatrix<FL> tb = b;
        if (conjb == 1) {
            tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(n * m), n, m);
            for (MKL_INT ik = 0; ik < m; ik++)
                for (MKL_INT jk = 0; jk < n; jk++)
                    tb(jk, ik) = conj(b(ik, jk));
        } else if (conjb == 2) {
            tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(n * m), n, m);
            for (MKL_INT ik = 0; ik < m; ik++)
                for (MKL_INT jk = 0; jk < n; jk++)
                    tb(jk, ik) = b(ik, jk);
        }
        GMatrixFunctions<FL>::copy(c, a);
        if (conjb == 2)
            GMatrixFunctions<FL>::transpose(c, tb, scale);
        else
            GMatrixFunctions<FL>::iadd(c, tb, scale, conjb, cfactor);
        for (MKL_INT ik = 0; ik < m; ik++)
            for (MKL_INT jk = 0; jk < n; jk++)
                ASSERT_LT(
                    abs(cfactor * a(ik, jk) + scale * b(ik, jk) - c(ik, jk)),
                    thrd + thrd * abs(c(ik, jk)));
        if (conjb != 2) {
            GMatrixFunctions<FL>::copy(c, a);
            seq->iadd(c, tb, scale, conjb, cfactor);
            seq->simple_perform();
            for (MKL_INT ik = 0; ik < m; ik++)
                for (MKL_INT jk = 0; jk < n; jk++)
                    ASSERT_LT(abs(cfactor * a(ik, jk) + scale * b(ik, jk) -
                                  c(ik, jk)),
                              thrd + thrd * abs(c(ik, jk)));
        }
        if (conjb)
            tb.deallocate();
        b.deallocate();
        c.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestMultiply) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-8 : 1E-2;
    shared_ptr<BatchGEMMSeq<FL>> seq =
        make_shared<BatchGEMMSeq<FL>>(0, SeqTypes::None);
    for (int i = 0; i < this->n_tests; i++) {
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
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * k), m, k);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(k * n), k, n);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(m * ldc), m, ldc);
        GMatrix<FL> cc(dalloc_<FP>()->complex_allocate(m * ldc), m, ldc);
        GMatrix<FL> ta(
            dalloc_<FP>()->complex_allocate((conja & 1) ? k * lda : m * lda),
            (conja & 1) ? k : m, lda);
        GMatrix<FL> tb(dalloc_<FP>()->complex_allocate(k * n),
                       (conjb & 1) ? n : k, (conjb & 1) ? k : n);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        Random::complex_fill<FP>(c.data, c.size());
        Random::complex_fill<FP>(ta.data, ta.size());
        Random::complex_fill<FP>(tb.data, tb.size());
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
        FL scale, cfactor;
        Random::complex_fill<FP>(&scale, 1);
        Random::complex_fill<FP>(&cfactor, 1);
        if (ii)
            scale = 1.0;
        if (jj)
            cfactor = 1.0;
        GMatrixFunctions<FL>::copy(cc, c);
        GMatrixFunctions<FL>::multiply(ta, conja, tb, conjb, cc, scale,
                                       cfactor);
        for (MKL_INT ik = 0; ik < m; ik++)
            for (MKL_INT jk = 0; jk < n; jk++) {
                FL x = cfactor * c(ik, jk);
                for (MKL_INT kk = 0; kk < k; kk++)
                    x += scale * a(ik, kk) * b(kk, jk);
                ASSERT_LT(abs(x - cc(ik, jk)), thrd);
            }
        GMatrixFunctions<FL>::copy(cc, c);
        seq->multiply(ta, conja, tb, conjb, cc, scale, cfactor);
        seq->simple_perform();
        for (MKL_INT ik = 0; ik < m; ik++)
            for (MKL_INT jk = 0; jk < n; jk++) {
                FL x = cfactor * c(ik, jk);
                for (MKL_INT kk = 0; kk < k; kk++)
                    x += scale * a(ik, kk) * b(kk, jk);
                ASSERT_LT(abs(x - cc(ik, jk)), thrd);
            }
        tb.deallocate();
        ta.deallocate();
        cc.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestRotate) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 100;
    const FP thrd = is_same<FP, double>::value ? 1E-8 : 1E-2;
    shared_ptr<BatchGEMMSeq<FL>> seq =
        make_shared<BatchGEMMSeq<FL>>(0, SeqTypes::None);
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT mk = Random::rand_int(1, sz), nk = Random::rand_int(1, sz);
        MKL_INT mb = Random::rand_int(1, sz), nb = Random::rand_int(1, sz);
        GMatrix<FL> k(dalloc_<FP>()->complex_allocate(mk * nk), mk, nk);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(mb * nb), mb, nb);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(nb * mk), nb, mk);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mb * nk), mb, nk);
        GMatrix<FL> ba(dalloc_<FP>()->complex_allocate(mb * mk), mb, mk);
        GMatrix<FL> bc(dalloc_<FP>()->complex_allocate(nb * nk), nb, nk);
        Random::complex_fill<FP>(k.data, k.size());
        Random::complex_fill<FP>(b.data, b.size());
        Random::complex_fill<FP>(a.data, a.size());
        uint8_t conjk = Random::rand_int(0, 4);
        uint8_t conjb = Random::rand_int(0, 4);
        GMatrix<FL> tk = k, tb = b;
        if (conjk == 1) {
            tk = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mk * nk), nk, mk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(jk, ik) = k(ik, jk);
        } else if (conjk == 2) {
            tk = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mk * nk), mk, nk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(ik, jk) = conj(k(ik, jk));
        } else if (conjk == 3) {
            tk = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mk * nk), nk, mk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(jk, ik) = conj(k(ik, jk));
        }
        if (conjb == 1) {
            tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        } else if (conjb == 2) {
            tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nb), mb, nb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(ib, jb) = conj(b(ib, jb));
        } else if (conjb == 3) {
            tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = conj(b(ib, jb));
        }
        c.clear();
        GMatrixFunctions<FL>::rotate(a, c, tb, conjb, tk, conjk, FL(2.0, 1.0));
        ba.clear();
        for (MKL_INT jb = 0; jb < nb; jb++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    ba(ib, ja) += b(ib, jb) * a(jb, ja) * FL(2.0, 1.0);
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                FL x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), thrd);
            }
        // batch gemm does not support both conjb/conjk case
        if (conjb != 2 || conjk != 2) {
            c.clear();
            seq->rotate(a, c, tb, conjb, tk, conjk, FL(2.0, 1.0));
            seq->simple_perform();
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jk = 0; jk < nk; jk++) {
                    FL x = 0;
                    for (MKL_INT ik = 0; ik < mk; ik++)
                        x += ba(ib, ik) * k(ik, jk);
                    ASSERT_LT(abs(x - c(ib, jk)), thrd);
                }
        }
        if (conjb)
            tb.deallocate();
        if (conjk)
            tk.deallocate();
        bool conja = Random::rand_int(0, 2);
        bool conjc = Random::rand_int(0, 2);
        GMatrix<FL> ta = a, tc = c;
        tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nb), nb, mb);
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jb = 0; jb < nb; jb++)
                tb(jb, ib) = conj(b(ib, jb));
        if (conja) {
            ta = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mk * nb), mk, nb);
            for (MKL_INT ia = 0; ia < nb; ia++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    ta(ja, ia) = conj(a(ia, ja));
        }
        c.clear();
        if (conjc) {
            tc = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nk), nk, mb);
            tc.clear();
        }
        GMatrixFunctions<FL>::left_partial_rotate(ta, conja, tc, conjc, tb, k,
                                                  FL(2.0, 1.0));
        if (conjc) {
            for (MKL_INT ic = 0; ic < mb; ic++)
                for (MKL_INT jc = 0; jc < nk; jc++)
                    c(ic, jc) = conj(tc(jc, ic));
        }
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                FL x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), thrd);
            }
        if (conjc)
            tc.deallocate();
        c.clear();
        if (conjc) {
            tc = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nk), nk, mb);
            tc.clear();
        }
        seq->left_partial_rotate(ta, conja, tc, conjc, tb, k, FL(2.0, 1.0));
        seq->simple_perform();
        if (conjc) {
            for (MKL_INT ic = 0; ic < mb; ic++)
                for (MKL_INT jc = 0; jc < nk; jc++)
                    c(ic, jc) = conj(tc(jc, ic));
        }
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                FL x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), thrd);
            }
        if (conjc)
            tc.deallocate();
        if (conja)
            ta.deallocate();
        Random::complex_fill<FP>(c.data, c.size());
        bc.clear();
        for (MKL_INT jb = 0; jb < nb; jb++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jc = 0; jc < nk; jc++)
                    bc(jb, jc) += b(ib, jb) * c(ib, jc) * FL(2.0, 1.0);
        if (conjc) {
            tc = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nk), nk, mb);
            for (MKL_INT ic = 0; ic < nk; ic++)
                for (MKL_INT jc = 0; jc < mb; jc++)
                    tc(ic, jc) = conj(c(jc, ic));
        }
        a.clear();
        if (conja) {
            ta = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mk * nb), mk, nb);
            ta.clear();
        }
        GMatrixFunctions<FL>::right_partial_rotate(tc, conjc, ta, conja, tb, k,
                                                   FL(2.0, 1.0));
        if (conja) {
            for (MKL_INT ia = 0; ia < nb; ia++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    a(ia, ja) = conj(ta(ja, ia));
        }
        for (MKL_INT ib = 0; ib < nb; ib++)
            for (MKL_INT jk = 0; jk < mk; jk++) {
                FL x = 0;
                for (MKL_INT ik = 0; ik < nk; ik++)
                    x += bc(ib, ik) * k(jk, ik);
                ASSERT_LT(abs(x - a(ib, jk)), thrd);
            }
        if (conja)
            ta.deallocate();
        a.clear();
        if (conja) {
            ta = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mk * nb), mk, nb);
            ta.clear();
        }
        seq->right_partial_rotate(tc, conjc, ta, conja, tb, k, FL(2.0, 1.0));
        seq->simple_perform();
        if (conja) {
            for (MKL_INT ia = 0; ia < nb; ia++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    a(ia, ja) = conj(ta(ja, ia));
        }
        for (MKL_INT ib = 0; ib < nb; ib++)
            for (MKL_INT jk = 0; jk < mk; jk++) {
                FL x = 0;
                for (MKL_INT ik = 0; ik < nk; ik++)
                    x += bc(ib, ik) * k(jk, ik);
                ASSERT_LT(abs(x - a(ib, jk)), thrd);
            }
        if (conja)
            ta.deallocate();
        if (conjc)
            tc.deallocate();
        tb.deallocate();
        bc.deallocate();
        ba.deallocate();
        c.deallocate();
        a.deallocate();
        b.deallocate();
        k.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestTensorProductDiagonal) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-14 : 5E-6;
    shared_ptr<BatchGEMMSeq<FL>> seq =
        make_shared<BatchGEMMSeq<FL>>(0, SeqTypes::None);
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT ma = Random::rand_int(1, 200), na = ma;
        MKL_INT mb = Random::rand_int(1, 200), nb = mb;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(ma * na), ma, na);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(mb * nb), mb, nb);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(ma * nb), ma, nb);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        c.clear();
        uint8_t conj = Random::rand_int(0, 4);
        GMatrixFunctions<FL>::tensor_product_diagonal(conj, a, b, c,
                                                      FL(2.0, 1.0));
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                ASSERT_LE(abs(FL(2.0, 1.0) *
                                  ((conj & 1) ? xconj(a(ia, ia)) : a(ia, ia)) *
                                  ((conj & 2) ? xconj(b(ib, ib)) : b(ib, ib)) -
                              c(ia, ib)),
                          thrd);
        c.clear();
        seq->tensor_product_diagonal(conj, a, b, c, FL(2.0, 1.0));
        seq->simple_perform();
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                ASSERT_LE(abs(FL(2.0, 1.0) *
                                  ((conj & 1) ? xconj(a(ia, ia)) : a(ia, ia)) *
                                  ((conj & 2) ? xconj(b(ib, ib)) : b(ib, ib)) -
                              c(ia, ib)),
                          thrd);
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestThreeRotate) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-10 : 1E-3;
    shared_ptr<BatchGEMMSeq<FL>> seq =
        make_shared<BatchGEMMSeq<FL>>(0, SeqTypes::None);
    const int sz = 200;
    for (int i = 0; i < this->n_tests; i++) {
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
        GMatrix<FL> da(dalloc_<FP>()->complex_allocate(mda * nda), mda, nda);
        GMatrix<FL> db(dalloc_<FP>()->complex_allocate(mdb * ndb), mdb, ndb);
        GMatrix<FL> dc(dalloc_<FP>()->complex_allocate(mdc * ndc), mdc, ndc);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(mx * nx), mx, nx);
        MKL_INT mb = ll ? mdc : mx, nb = ll ? ndc : nx;
        MKL_INT
        mk = ll ? mx : mdc, nk = ll ? nx : ndc;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(nb * mk), nb, mk);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mb * nk), mb, nk);
        GMatrix<FL> cc(dalloc_<FP>()->complex_allocate(mb * nk), mb, nk);
        Random::complex_fill<FP>(da.data, da.size());
        Random::complex_fill<FP>(db.data, db.size());
        Random::complex_fill<FP>(a.data, a.size());
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
        GMatrixFunctions<FL>::three_rotate(a, c, ll ? dc : x, conjb,
                                           ll ? x : dc, conjk, da, dconja, db,
                                           dconjb, ll, FL(2.0, 1.0), dc_stride);
        dc.clear(), cc.clear();
        GMatrixFunctions<FL>::tensor_product(da, dconja, db, dconjb, dc, 1.0,
                                             dc_stride);
        GMatrixFunctions<FL>::rotate(a, cc, ll ? dc : x, conjb ? 3 : 0,
                                     ll ? x : dc, conjk ? 1 : 2, FL(2.0, 1.0));
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(c, cc, thrd, thrd));
        c.clear();
        seq->three_rotate(a, c, ll ? dc : x, conjb, ll ? x : dc, conjk, da,
                          dconja, db, dconjb, ll, FL(2.0, 1.0), dc_stride);
        seq->simple_perform();
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(c, cc, thrd, thrd));
        cc.deallocate();
        c.deallocate();
        a.deallocate();
        x.deallocate();
        dc.deallocate();
        db.deallocate();
        da.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestThreeTensorProductDiagonal) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-8 : 1E-3;
    shared_ptr<BatchGEMM<FL>> batch = make_shared<BatchGEMM<FL>>();
    for (int i = 0; i < this->n_tests; i++) {
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
        GMatrix<FL> da(dalloc_<FP>()->complex_allocate(mda * nda), mda, nda);
        GMatrix<FL> db(dalloc_<FP>()->complex_allocate(mdb * ndb), mdb, ndb);
        GMatrix<FL> dc(dalloc_<FP>()->complex_allocate(mdc * ndc), mdc, ndc);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(mx * nx), mx, nx);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mdc * mx), mdc, mx);
        GMatrix<FL> cc(dalloc_<FP>()->complex_allocate(mdc * mx), mdc, mx);
        if (!ll)
            c = c.flip_dims(), cc = cc.flip_dims();
        Random::complex_fill<FP>(da.data, da.size());
        Random::complex_fill<FP>(db.data, db.size());
        MKL_INT dcm_stride = Random::rand_int(0, mdc - mda * mdb + 1);
        MKL_INT dcn_stride = Random::rand_int(0, ndc - nda * ndb + 1);
        dcn_stride = dcm_stride;
        MKL_INT dc_stride = dcm_stride * dc.n + dcn_stride;
        c.clear();
        uint8_t conj = Random::rand_int(0, 4);
        GMatrixFunctions<FL>::three_tensor_product_diagonal(
            conj, ll ? dc : x, ll ? x : dc, c, da, dconja, db, dconjb, ll,
            FL(2.0, 1.0), dc_stride);
        dc.clear(), cc.clear();
        GMatrixFunctions<FL>::tensor_product(da, dconja, db, dconjb, dc, 1.0,
                                             dc_stride);
        GMatrixFunctions<FL>::tensor_product_diagonal(
            conj, ll ? dc : x, ll ? x : dc, cc, FL(2.0, 1.0));
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(c, cc, thrd, thrd));
        c.clear();
        AdvancedGEMM<FL>::three_tensor_product_diagonal(
            batch, conj, ll ? dc : x, ll ? x : dc, c, da, dconja, db, dconjb,
            ll, FL(2.0, 1.0), dc_stride);
        batch->perform();
        batch->clear();
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(c, cc, thrd, thrd));
        cc.deallocate();
        c.deallocate();
        x.deallocate();
        dc.deallocate();
        db.deallocate();
        da.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestTensorProduct) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-14 : 5E-6;
    shared_ptr<BatchGEMM<FL>> batch = make_shared<BatchGEMM<FL>>();
    for (int i = 0; i < this->n_tests; i++) {
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
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(ma * na), ma, na);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(mb * nb), mb, nb);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mc * nc), mc, nc);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        c.clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        GMatrix<FL> ta = a, tb = b;
        if (conja) {
            ta = GMatrix<FL>(dalloc_<FP>()->complex_allocate(ma * na), na, ma);
            for (MKL_INT ia = 0; ia < ma; ia++)
                for (MKL_INT ja = 0; ja < na; ja++)
                    ta(ja, ia) = conj(a(ia, ja));
        }
        if (conjb) {
            tb = GMatrix<FL>(dalloc_<FP>()->complex_allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = conj(b(ib, jb));
        }
        MKL_INT cm_stride = Random::rand_int(0, mc - ma * mb + 1);
        MKL_INT cn_stride = Random::rand_int(0, nc - na * nb + 1);
        MKL_INT c_stride = cm_stride * c.n + cn_stride;
        if (Random::rand_int(0, 2))
            GMatrixFunctions<FL>::tensor_product(ta, conja, tb, conjb, c,
                                                 FL(2.0, 1.0), c_stride);
        else {
            AdvancedGEMM<FL>::tensor_product(batch, ta, conja, tb, conjb, c,
                                             FL(2.0, 1.0), c_stride);
            batch->perform();
            batch->clear();
        }
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ja = 0; ja < na; ja++)
                for (MKL_INT ib = 0; ib < mb; ib++)
                    for (MKL_INT jb = 0; jb < nb; jb++)
                        ASSERT_LE(abs(FL(2.0, 1.0) * a(ia, ja) * b(ib, jb) -
                                      c(ia * mb + ib + cm_stride,
                                        ja * nb + jb + cn_stride)),
                                  thrd);
        if (conjb)
            tb.deallocate();
        if (conja)
            ta.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestExponentialPade) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-6 : 1E-3;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT n = Random::rand_int(1, sz);
        int ideg = 6;
        FP t = (FP)Random::rand_double(-0.1, 0.1);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(n * n), n, n);
        GMatrix<FL> v(dalloc_<FP>()->complex_allocate(n), n, 1);
        GMatrix<FL> u(dalloc_<FP>()->complex_allocate(n), n, 1);
        GMatrix<FL> w(dalloc_<FP>()->complex_allocate(n), n, 1);
        GMatrix<FL> work(dalloc_<FP>()->complex_allocate(4 * n * n + ideg + 1),
                         4 * n * n + ideg + 1, 1);
        Random::complex_fill<FP>(a.data, a.size());
        // note that eigs can only work on hermitian matrix
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = conj(a(ki, kj));
            a(ki, ki) = real(a(ki, ki));
        }
        Random::complex_fill<FP>(v.data, v.size());
        for (MKL_INT ki = 0; ki < n; ki++)
            w(ki, 0) = v(ki, 0);
        auto x = IterativeMatrixFunctions<FL>::expo_pade(ideg, n, a.data, n, t,
                                                         work.data);
        GMatrixFunctions<FL>::multiply(GMatrix<FL>(work.data + x.first, n, n),
                                       false, v, false, w, 1.0, 0.0);
        GDiagonalMatrix<FP> ww(dalloc_<FP>()->allocate(n), n);
        GMatrixFunctions<FL>::eigs(a, ww);
        GMatrixFunctions<FL>::multiply(v, true, a, 3, u.flip_dims(), 1.0, 0.0);
        for (MKL_INT i = 0; i < n; i++)
            v(i, 0) = exp(t * ww(i, i)) * u(i, 0);
        GMatrixFunctions<FL>::multiply(v, true, a, false, u.flip_dims(), 1.0,
                                       0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(u, w, thrd, thrd));
        ww.deallocate();
        work.deallocate();
        w.deallocate();
        u.deallocate();
        v.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestExponential) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 300 : 150;
    const FP conv = is_same<FP, double>::value ? 1E-8 : 1E-4;
    const FP thrd = is_same<FP, double>::value ? 1E-6 : 1E-3;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT n = Random::rand_int(1, sz);
        bool symm = Random::rand_int(0, 2);
        FL t((FP)Random::rand_double(-0.1, 0.1),
             symm ? (FP)0.0 : (FP)Random::rand_double(-0.1, 0.1));
        FP consta = (FP)Random::rand_double(-2.0, 2.0);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(n * n), n, n);
        GMatrix<FL> aa(dalloc_<FP>()->complex_allocate(n), n, 1);
        GMatrix<FL> v(dalloc_<FP>()->complex_allocate(n), n, 1);
        GMatrix<FL> u(dalloc_<FP>()->complex_allocate(n), n, 1);
        GMatrix<FL> w(dalloc_<FP>()->complex_allocate(n), n, 1);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(v.data, v.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = conj(a(kj, ki));
            a(ki, ki) = real(a(ki, ki));
            aa(ki, 0) = a(ki, ki);
            w(ki, 0) = v(ki, 0);
        }
        FP anorm = GMatrixFunctions<FL>::norm(a);
        MatMul mop(a);
        int nmult = IterativeMatrixFunctions<FL>::expo_apply(
            mop, t, anorm, w, consta, symm, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv);
        GDiagonalMatrix<FP> ww(dalloc_<FP>()->allocate(n), n);
        GMatrixFunctions<FL>::eigs(a, ww);
        GMatrixFunctions<FL>::multiply(v, true, a, 3, u.flip_dims(), 1.0, 0.0);
        for (MKL_INT i = 0; i < n; i++)
            v(i, 0) = exp(t * (ww(i, i) + consta)) * u(i, 0);
        GMatrixFunctions<FL>::multiply(v, true, a, false, u.flip_dims(), 1.0,
                                       0.0);
        EXPECT_TRUE(GMatrixFunctions<FL>::all_close(u, w, thrd, thrd));
        ww.deallocate();
        w.deallocate();
        u.deallocate();
        v.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestHarmonicDavidson) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 50 : 25;
    const int sz2 = is_same<FP, double>::value ? 35 : 10;
    const FP conv = is_same<FP, double>::value ? 1E-9 : 1E-3;
    const FP thrd = is_same<FP, double>::value ? 1E-6 : 1E+1;
    const FP thrd2 = is_same<FP, double>::value ? 1E-3 : 1E+1;
    const FP thrd3 = is_same<FP, double>::value ? 1E-10 : 1E-5;
    // this test is not very stable
    Random::rand_seed(1234);
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT n = Random::rand_int(3, sz);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 5));
        if (k > n / 2)
            k = n / 2;
        int ndav = 0;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(n * n), n, n);
        GDiagonalMatrix<FL> aa(dalloc_<FP>()->complex_allocate(n), n);
        GDiagonalMatrix<FP> ww(dalloc_<FP>()->allocate(n), n);
        vector<GMatrix<FL>> bs(k, GMatrix<FL>(nullptr, n, 1));
        Random::complex_fill<FP>(a.data, a.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = conj(a(ki, kj));
            a(ki, ki) = real(a(ki, ki));
            a(ki, ki) += 2.0;
            aa(ki, ki) = a(ki, ki);
        }
        for (int i = 0; i < k; i++) {
            bs[i].allocate();
            bs[i].clear();
            bs[i].data[i] = 1;
        }
        MatMul mop(a);
        FP shift = (FP)0.1;
        DavidsonTypes davidson_type =
            Random::rand_int(0, 2)
                ? DavidsonTypes::HarmonicLessThan | DavidsonTypes::NoPrecond
                : DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond;
        vector<FP> vw = IterativeMatrixFunctions<FL>::harmonic_davidson(
            mop, aa, bs, shift, davidson_type, ndav, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv, n * k * 500,
            n * k * 400, 2, sz2);
        ASSERT_EQ((int)vw.size(), k);
        GDiagonalMatrix<FP> w(&vw[0], k);
        GMatrixFunctions<FL>::eigs(a, ww);
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
            ASSERT_LT(abs(ww.data[eigval_idxs[i]] - vw[i]),
                      thrd + abs(vw[i]) * thrd);
        for (int i = 0; i < k - 1; i++) {
            FL factor = 0.0;
            for (int j = 0; j < a.n; j++)
                if (abs(bs[i].data[j]) > thrd3)
                    factor += a.data[a.n * eigval_idxs[i] + j] / bs[i].data[j];
            factor = factor / abs(factor);
            ASSERT_LE(abs(factor) - 1.0, thrd2);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                GMatrix<FL>(a.data + a.n * eigval_idxs[i], a.n, 1), bs[i],
                thrd2, thrd2, factor));
        }
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
    Random::rand_seed(0);
}

TYPED_TEST(TestComplexMatrix, TestDavidson) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 120;
    const FP conv = is_same<FP, double>::value ? 1E-8 : 1E-6;
    const FP thrd = is_same<FP, double>::value ? 1E-6 : 5E-3;
    const FP thrd2 = is_same<FP, double>::value ? 1E-3 : 1E-1;
    const FP thrd3 = is_same<FP, double>::value ? 1E-10 : 1E-5;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT n = Random::rand_int(1, sz);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 10));
        int ndav = 0;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(n * n), n, n);
        GDiagonalMatrix<FL> aa(dalloc_<FP>()->complex_allocate(n), n);
        GDiagonalMatrix<FP> ww(dalloc_<FP>()->allocate(n), n);
        vector<GMatrix<FL>> bs(k, GMatrix<FL>(nullptr, n, 1));
        Random::complex_fill<FP>(a.data, a.size());
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
        vector<FP> vw = IterativeMatrixFunctions<FL>::davidson(
            mop, aa, bs, 0, DavidsonTypes::Normal, ndav, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv, n * k * 5,
            n * k * 4, k * 2, max((MKL_INT)5, k + 10));
        ASSERT_EQ((int)vw.size(), k);
        GDiagonalMatrix<FP> w(&vw[0], k);
        GMatrixFunctions<FL>::eigs(a, ww);
        GDiagonalMatrix<FP> w2(ww.data, k);
        ASSERT_TRUE(GMatrixFunctions<FP>::all_close(w, w2, thrd, thrd));
        for (int i = 0; i < k; i++) {
            FL factor = 0.0;
            for (int j = 0; j < a.n; j++)
                if (abs(bs[i].data[j]) > thrd3)
                    factor += a.data[a.n * i + j] / bs[i].data[j];
            factor = factor / abs(factor);
            ASSERT_LE(abs(factor) - 1.0, thrd2);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                GMatrix<FL>(a.data + a.n * i, a.n, 1), bs[i], thrd2, thrd2,
                factor));
        }
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestLinear) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-7 : 5E-2;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = Random::rand_int(1, sz);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> bg(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), n, m);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        GMatrixFunctions<FL>::copy(af, a);
        GMatrixFunctions<FL>::copy(x, b);
        GMatrixFunctions<FL>::linear(af, x);
        // note that linear solves the H^T problem
        GMatrixFunctions<FL>::multiply(x, false, a, false, bg, 1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(bg, b, thrd, thrd));
        x.deallocate();
        bg.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestCG) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP conv = is_same<FP, double>::value ? 1E-14 : 1E-7;
    const FP thrd = is_same<FP, double>::value ? 1E-3 : 5E-2;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = 1;
        int nmult = 0;
        FL eta = 0.05;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GDiagonalMatrix<FL> aa(dalloc_<FP>()->complex_allocate(m), m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> xg(dalloc_<FP>()->complex_allocate(n * m), n, m);
        Random::complex_fill<FP>(af.data, af.size());
        Random::complex_fill<FP>(b.data, b.size());
        Random::complex_fill<FP>(xg.data, xg.size());
        GMatrixFunctions<FL>::multiply(af, false, af, 3, a, 1.0, 0.0);
        for (MKL_INT ki = 0; ki < m; ki++)
            a(ki, ki) += eta, aa(ki, ki) = a(ki, ki);
        af.clear();
        GMatrixFunctions<FL>::transpose(af, a, 1.0);
        GMatrixFunctions<FL>::copy(x, b);
        // note that linear solves the H^T problem
        GMatrixFunctions<FL>::linear(af, x);
        MatMul mop(a);
        FL func = IterativeMatrixFunctions<FL>::conjugate_gradient(
            mop, aa, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv, 5000, 4000);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(xg, x, thrd, thrd));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        aa.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestMinRes) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP conv = is_same<FP, double>::value ? 1E-14 : 1E-7;
    const FP thrd = is_same<FP, double>::value ? 1E-3 : 1E+1;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = 1;
        int nmult = 0;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> xg(dalloc_<FP>()->complex_allocate(n * m), n, m);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        Random::complex_fill<FP>(xg.data, xg.size());
        for (MKL_INT ki = 0; ki < m; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = conj(a(kj, ki));
            a(ki, ki) = real(a(ki, ki));
        }
        af.clear();
        GMatrixFunctions<FL>::transpose(af, a, 1.0);
        GMatrixFunctions<FL>::copy(x, b);
        // note that linear solves the H^T problem
        GMatrixFunctions<FL>::linear(af, x);
        MatMul mop(a);
        IterativeMatrixFunctions<FL>::minres(
            mop, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv, 5000, 4000);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(xg, x, thrd, thrd));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestInverse) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-7 : 1E+1;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ap(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ag(dalloc_<FP>()->complex_allocate(m * m), m, m);
        Random::complex_fill<FP>(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++)
            for (MKL_INT kj = 0; kj < m; kj++)
                ap(ki, kj) = a(ki, kj);
        GMatrixFunctions<FL>::inverse(a);
        GMatrixFunctions<FL>::multiply(a, false, ap, false, ag, 1.0, 0.0);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                EXPECT_LT(abs(ag(j, k) - (FP)(k == j ? 1.0 : 0.0)), thrd);
        ag.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestDeflatedCG) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 50;
    const FP conv = is_same<FP, double>::value ? 1E-14 : 1E-6;
    const FP thrd = is_same<FP, double>::value ? 1E-4 : 1E+0;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = 1;
        int nmult = 0;
        FP eta = 0.05;
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ag(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GDiagonalMatrix<FL> aa(dalloc_<FP>()->complex_allocate(m), m);
        GDiagonalMatrix<FP> ww(dalloc_<FP>()->allocate(m), m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), n, m);
        GMatrix<FL> xg(dalloc_<FP>()->complex_allocate(n * m), n, m);
        Random::complex_fill<FP>(af.data, af.size());
        Random::complex_fill<FP>(b.data, b.size());
        Random::complex_fill<FP>(xg.data, xg.size());
        GMatrixFunctions<FL>::multiply(af, false, af, 3, a, 1.0, 0.0);
        for (MKL_INT ki = 0; ki < m; ki++)
            a(ki, ki) += eta;
        GMatrixFunctions<FL>::copy(ag, a);
        GMatrixFunctions<FL>::eigs(ag, ww);
        GMatrix<FL> w(ag.data, m, 1);
        af.clear();
        GMatrixFunctions<FL>::transpose(af, a, 1.0);
        GMatrixFunctions<FL>::copy(x, b);
        GMatrixFunctions<FL>::linear(af, x);
        MatMul mop(a);
        for (MKL_INT ki = 0; ki < m; ki++)
            aa(ki, ki) = a(ki, ki);
        FL func = IterativeMatrixFunctions<FL>::deflated_conjugate_gradient(
            mop, aa, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv, 5000, 4000,
            vector<GMatrix<FL>>{w});
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(xg, x, thrd, thrd));
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

TYPED_TEST(TestComplexMatrix, TestEigs) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-7 : 1E+0;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ap(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ag(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GDiagonalMatrix<FP> w(dalloc_<FP>()->allocate(m), m);
        Random::complex_fill<FP>(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = conj(a(kj, ki));
            a(ki, ki) = real(a(ki, ki));
        }
        GMatrixFunctions<FL>::copy(ap, a);
        GMatrixFunctions<FL>::eigs(a, w);
        GMatrixFunctions<FL>::multiply(a, false, ap, true, ag, 1.0, 0.0);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                ag(k, j) /= w(k, k);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(ag, a, thrd, thrd));
        w.deallocate();
        ag.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestEig) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 150 : 50;
    const FP thrd = is_same<FP, double>::value ? 1E-9 : 1E-2;
    const FP thrd2 = is_same<FP, double>::value ? 0 : 1E-2;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ap(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> aq(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> ag(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GDiagonalMatrix<FL> w(dalloc_<FP>()->complex_allocate(m), m);
        Random::complex_fill<FP>(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++)
            for (MKL_INT kj = 0; kj < m; kj++)
                ap(ki, kj) = a(ki, kj);
        GMatrixFunctions<FL>::eig(a, w, GMatrix<FL>(nullptr, a.m, a.n));
        GMatrixFunctions<FL>::multiply(a, false, ap, true, ag, 1.0, 0.0);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                ag(k, j) /= w(k, k);
        EXPECT_TRUE(GMatrixFunctions<FL>::all_close(ag, a, thrd, thrd2));
        // a[i, j] u[k, j] = w[k, k] u[k, i]
        // a[i, j] = uinv[j, k] w[k, k] u[k, i] = uinv[j, k] a[i, jp] u[k, jp]
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                ag(k, j) = a(k, j) * w(k, k);
        GMatrixFunctions<FL>::inverse(a);
        GMatrixFunctions<FL>::multiply(ag, true, a, true, aq, 1.0, 0.0);
        EXPECT_TRUE(GMatrixFunctions<FL>::all_close(ap, aq, thrd, thrd2));
        w.deallocate();
        ag.deallocate();
        aq.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestSVD) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-12 : 1E-3;
    const FP thrd2 = is_same<FP, double>::value ? 0 : 1E-3;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = Random::rand_int(1, sz);
        MKL_INT k = min(m, n);
        shared_ptr<GTensor<FL>> a =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, n});
        shared_ptr<GTensor<FL>> l =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, k});
        shared_ptr<GTensor<FP>> s =
            make_shared<GTensor<FP>>(vector<MKL_INT>{k});
        shared_ptr<GTensor<FL>> r =
            make_shared<GTensor<FL>>(vector<MKL_INT>{k, n});
        shared_ptr<GTensor<FL>> aa =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, n});
        shared_ptr<GTensor<FL>> kk =
            make_shared<GTensor<FL>>(vector<MKL_INT>{k, k});
        shared_ptr<GTensor<FL>> ll =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, m});
        shared_ptr<GTensor<FL>> rr =
            make_shared<GTensor<FL>>(vector<MKL_INT>{n, n});
        Random::complex_fill<FP>(a->data->data(), a->size());
        GMatrixFunctions<FL>::copy(aa->ref(), a->ref());
        if (Random::rand_int(0, 2))
            GMatrixFunctions<FL>::accurate_svd(
                a->ref(), l->ref(), s->ref().flip_dims(), r->ref(), 1E-1);
        else
            GMatrixFunctions<FL>::svd(a->ref(), l->ref(), s->ref().flip_dims(),
                                      r->ref());
        GMatrixFunctions<FL>::multiply(l->ref(), 3, l->ref(), false, kk->ref(),
                                       1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
            kk->ref(), IdentityMatrix(k), thrd, thrd2));
        GMatrixFunctions<FL>::multiply(r->ref(), false, r->ref(), 3, kk->ref(),
                                       1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
            kk->ref(), IdentityMatrix(k), thrd, thrd2));
        if (m <= n) {
            GMatrixFunctions<FL>::multiply(l->ref(), false, l->ref(), 3,
                                           ll->ref(), 1.0, 0.0);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                ll->ref(), IdentityMatrix(m), thrd, thrd2));
        }
        if (n <= m) {
            GMatrixFunctions<FL>::multiply(r->ref(), 3, r->ref(), false,
                                           rr->ref(), 1.0, 0.0);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                rr->ref(), IdentityMatrix(n), thrd, thrd2));
        }
        GMatrix<FL> x(r->data->data(), 1, n);
        for (MKL_INT i = 0; i < k; i++) {
            ASSERT_GE((*s)({i}), 0.0);
            GMatrixFunctions<FL>::iscale(x.shift_ptr(i * n), (*s)({i}));
        }
        GMatrixFunctions<FL>::multiply(l->ref(), false, r->ref(), false,
                                       a->ref(), 1.0, 0.0);
        ASSERT_TRUE(
            GMatrixFunctions<FP>::all_close(aa->ref(), a->ref(), thrd, thrd2));
    }
}

TYPED_TEST(TestComplexMatrix, TestDisjointSVD) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-10 : 1E-3;
    const FP thrd2 = is_same<FP, double>::value ? 1E-10 : 1E-3;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = Random::rand_int(1, sz);
        MKL_INT k = min(m, n);
        size_t nnz = (size_t)(0.07 * m * n);
        shared_ptr<GTensor<FL>> a =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, n});
        shared_ptr<GTensor<FL>> l =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, k});
        shared_ptr<GTensor<FP>> s =
            make_shared<GTensor<FP>>(vector<MKL_INT>{k});
        shared_ptr<GTensor<FL>> r =
            make_shared<GTensor<FL>>(vector<MKL_INT>{k, n});
        shared_ptr<GTensor<FL>> aa =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, n});
        shared_ptr<GTensor<FL>> kk =
            make_shared<GTensor<FL>>(vector<MKL_INT>{k, k});
        shared_ptr<GTensor<FL>> ll =
            make_shared<GTensor<FL>>(vector<MKL_INT>{m, m});
        shared_ptr<GTensor<FL>> rr =
            make_shared<GTensor<FL>>(vector<MKL_INT>{n, n});
        a->clear();
        for (size_t i = 0; i < nnz; i++)
            Random::complex_fill<FP>(
                &(*a->data)[Random::rand_int(0, (int)a->size())], 1);
        GMatrixFunctions<FL>::copy(aa->ref(), a->ref());
        vector<FP> levels{0.6, 0.3};
        levels.resize(Random::rand_int(0, (int)levels.size() + 1));
        IterativeMatrixFunctions<FL>::disjoint_svd(
            a->ref(), l->ref(), s->ref().flip_dims(), r->ref(), levels);
        if (levels.size() == 0 && is_same<FP, double>::value) {
            GMatrixFunctions<FL>::multiply(l->ref(), 3, l->ref(), false,
                                           kk->ref(), 1.0, 0.0);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                kk->ref(), IdentityMatrix(k), thrd, thrd2));
            GMatrixFunctions<FL>::multiply(r->ref(), false, r->ref(), 3,
                                           kk->ref(), 1.0, 0.0);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                kk->ref(), IdentityMatrix(k), thrd, thrd2));
            if (m <= n) {
                GMatrixFunctions<FL>::multiply(l->ref(), false, l->ref(), 3,
                                               ll->ref(), 1.0, 0.0);
                ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                    ll->ref(), IdentityMatrix(m), thrd, thrd2));
            }
            if (n <= m) {
                GMatrixFunctions<FL>::multiply(r->ref(), 3, r->ref(), false,
                                               rr->ref(), 1.0, 0.0);
                ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                    rr->ref(), IdentityMatrix(n), thrd, thrd2));
            }
        }
        GMatrix<FL> x(r->data->data(), 1, n);
        for (MKL_INT i = 0; i < k; i++) {
            ASSERT_GE((*s)({i}), 0.0);
            GMatrixFunctions<FL>::iscale(x.shift_ptr(i * n), (*s)({i}));
        }
        GMatrixFunctions<FL>::multiply(l->ref(), false, r->ref(), false,
                                       a->ref(), 1.0, 0.0);
        ASSERT_TRUE(
            GMatrixFunctions<FP>::all_close(aa->ref(), a->ref(), thrd, thrd2));
    }
}

TYPED_TEST(TestComplexMatrix, TestQR) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 120;
    const FP thrd = is_same<FP, double>::value ? 1E-12 : 1E-3;
    const FP thrd2 = is_same<FP, double>::value ? 0 : 1E-3;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = Random::rand_int(1, sz);
        MKL_INT k = min(m, n);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * n), m, n);
        GMatrix<FL> qr(dalloc_<FP>()->complex_allocate(m * n), m, n);
        Random::complex_fill<FP>(a.data, a.size());
        GMatrix<FL> q(dalloc_<FP>()->complex_allocate(m * k), m, k);
        GMatrix<FL> qq(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> qqk(dalloc_<FP>()->complex_allocate(k * k), k, k);
        GMatrix<FL> r(dalloc_<FP>()->complex_allocate(n * k), k, n);
        GMatrixFunctions<FL>::qr(a, q, r);
        GMatrixFunctions<FL>::multiply(q, false, r, false, qr, 1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(a, qr, thrd, thrd2));
        GMatrixFunctions<FL>::multiply(q, 3, q, false, qqk, 1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(qqk, IdentityMatrix(k),
                                                    thrd, thrd2));
        if (m <= n) {
            GMatrixFunctions<FL>::multiply(q, false, q, 3, qq, 1.0, 0.0);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(qq, IdentityMatrix(m),
                                                        thrd, thrd2));
        }
        r.deallocate();
        qqk.deallocate();
        qq.deallocate();
        q.deallocate();
        qr.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestLQ) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 120;
    const FP thrd = is_same<FP, double>::value ? 1E-12 : 1E-3;
    const FP thrd2 = is_same<FP, double>::value ? 0 : 1E-3;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = Random::rand_int(1, sz);
        MKL_INT k = min(m, n);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * n), m, n);
        GMatrix<FL> lq(dalloc_<FP>()->complex_allocate(m * n), m, n);
        Random::complex_fill<FP>(a.data, a.size());
        GMatrix<FL> l(dalloc_<FP>()->complex_allocate(m * k), m, k);
        GMatrix<FL> q(dalloc_<FP>()->complex_allocate(n * k), k, n);
        GMatrix<FL> qq(dalloc_<FP>()->complex_allocate(n * n), n, n);
        GMatrix<FL> qqk(dalloc_<FP>()->complex_allocate(k * k), k, k);
        GMatrixFunctions<FL>::lq(a, l, q);
        GMatrixFunctions<FL>::multiply(l, false, q, false, lq, 1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(a, lq, thrd, thrd2));
        GMatrixFunctions<FL>::multiply(q, false, q, 3, qqk, 1.0, 0.0);
        ASSERT_TRUE(GMatrixFunctions<FL>::all_close(qqk, GIdentityMatrix<FP>(k),
                                                    thrd, thrd2));
        if (m >= n) {
            GMatrixFunctions<FL>::multiply(q, 3, q, false, qq, 1.0, 0.0);
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                qq, GIdentityMatrix<FP>(n), thrd, thrd2));
        }
        qqk.deallocate();
        qq.deallocate();
        q.deallocate();
        l.deallocate();
        lq.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestLeastSquares) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP thrd = is_same<FP, double>::value ? 1E-9 : 5E-2;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = Random::rand_int(1, m + 1);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * n), m, n);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(m), m, 1);
        GMatrix<FL> br(dalloc_<FP>()->complex_allocate(m), m, 1);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n), n, 1);
        Random::complex_fill<FP>(a.data, a.size());
        Random::complex_fill<FP>(b.data, b.size());
        FP res = GMatrixFunctions<FL>::least_squares(a, b, x);
        GMatrixFunctions<FL>::multiply(a, false, x, false, br, 1.0, 0.0);
        GMatrixFunctions<FL>::iadd(br, b, -1);
        FP cres = GMatrixFunctions<FL>::norm(br);
        EXPECT_LT(abs(res - cres), thrd);
        x.deallocate();
        br.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TYPED_TEST(TestComplexMatrix, TestGCROT) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 75;
    const FP conv = is_same<FP, double>::value ? 1E-14 : 1E-7;
    const FP thrd = is_same<FP, double>::value ? 1E-3 : 1E+0;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        FP eta = 0.05;
        GMatrix<FP> ra(dalloc_<FP>()->allocate(m * m), m, m);
        GMatrix<FP> rax(dalloc_<FP>()->allocate(m * m), m, m);
        GMatrix<FP> rb(dalloc_<FP>()->allocate(n * m), m, n);
        GMatrix<FP> rbg(dalloc_<FP>()->allocate(n * m), m, n);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), m, n);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), m, n);
        GMatrix<FL> xg(dalloc_<FP>()->complex_allocate(n * m), m, n);
        Random::fill<FP>(ra.data, ra.size());
        Random::fill<FP>(rax.data, rax.size());
        Random::fill<FP>(rb.data, rb.size());
        a.clear();
        b.clear();
        GMatrixFunctions<FP>::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        GMatrixFunctions<FL>::fill_complex(a, ra, GMatrix<FP>(nullptr, m, m));
        for (MKL_INT k = 0; k < m; k++)
            a(k, k) += FL(0, eta);
        GMatrixFunctions<FL>::fill_complex(b, rb, GMatrix<FP>(nullptr, m, n));
        Random::fill<FP>(rb.data, rb.size());
        GMatrixFunctions<FL>::fill_complex(x, rb, GMatrix<FP>(nullptr, m, n));
        Random::fill<FP>(rb.data, rb.size());
        GMatrixFunctions<FL>::fill_complex(x, GMatrix<FP>(nullptr, m, n), rb);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        MatMul mop(a);
        FL func = IterativeMatrixFunctions<FL>::gcrotmk(
            mop, GDiagonalMatrix<FL>(nullptr, 0), x, b, nmult, niter, 20, -1,
            0.0, false, (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv,
            10000);
        GMatrixFunctions<FL>::copy(xg, b);
        GMatrixFunctions<FL>::linear(af, xg.flip_dims());
        GMatrixFunctions<FL>::extract_complex(xg, rbg,
                                              GMatrix<FP>(nullptr, m, n));
        GMatrixFunctions<FL>::extract_complex(x, rb,
                                              GMatrix<FP>(nullptr, m, n));
        EXPECT_TRUE(GMatrixFunctions<FP>::all_close(rbg, rb, thrd, thrd));
        GMatrixFunctions<FL>::extract_complex(xg, GMatrix<FP>(nullptr, m, n),
                                              rbg);
        GMatrixFunctions<FL>::extract_complex(x, GMatrix<FP>(nullptr, m, n),
                                              rb);
        EXPECT_TRUE(GMatrixFunctions<FP>::all_close(rbg, rb, thrd, thrd));
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

TYPED_TEST(TestComplexMatrix, TestIDRS) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 50;
    const FP conv = is_same<FP, double>::value ? 1E-8 : 1E-6;
    const FP conv2 = is_same<FP, double>::value ? 1E-7 : 1E-5;
    const FP thrd = is_same<FP, double>::value ? 1E-3 : 2E+0;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        FP eta = 0.05;
        GMatrix<FP> ra(dalloc_<FP>()->allocate(m * m), m, m);
        GMatrix<FP> rax(dalloc_<FP>()->allocate(m * m), m, m);
        GMatrix<FP> rb(dalloc_<FP>()->allocate(n * m), m, n);
        GMatrix<FP> rbg(dalloc_<FP>()->allocate(n * m), m, n);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), m, n);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), m, n);
        GMatrix<FL> xg(dalloc_<FP>()->complex_allocate(n * m), m, n);
        Random::fill<FP>(ra.data, ra.size());
        Random::fill<FP>(rax.data, rax.size());
        Random::fill<FP>(rb.data, rb.size());
        a.clear();
        b.clear();
        GMatrixFunctions<FP>::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        GMatrixFunctions<FL>::fill_complex(a, ra, GMatrix<FP>(nullptr, m, m));
        for (MKL_INT k = 0; k < m; k++)
            a(k, k) += FL(0, eta);
        GMatrixFunctions<FL>::fill_complex(b, rb, GMatrix<FP>(nullptr, m, n));
        Random::fill<FP>(rb.data, rb.size());
        GMatrixFunctions<FL>::fill_complex(x, rb, GMatrix<FP>(nullptr, m, n));
        Random::fill<FP>(rb.data, rb.size());
        GMatrixFunctions<FL>::fill_complex(x, GMatrix<FP>(nullptr, m, n), rb);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        MatMul mop(a);
        FL func = IterativeMatrixFunctions<FL>::idrs(
            mop, GDiagonalMatrix<FL>(nullptr, 0), x, b, nmult, niter, 8, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv, 0.0, conv2,
            10000);
        GMatrixFunctions<FL>::copy(xg, b);
        GMatrixFunctions<FL>::linear(af, xg.flip_dims());
        GMatrixFunctions<FL>::extract_complex(xg, rbg,
                                              GMatrix<FP>(nullptr, m, n));
        GMatrixFunctions<FL>::extract_complex(x, rb,
                                              GMatrix<FP>(nullptr, m, n));
        EXPECT_TRUE(GMatrixFunctions<FP>::all_close(rbg, rb, thrd, thrd));
        GMatrixFunctions<FL>::extract_complex(xg, GMatrix<FP>(nullptr, m, n),
                                              rbg);
        GMatrixFunctions<FL>::extract_complex(x, GMatrix<FP>(nullptr, m, n),
                                              rb);
        EXPECT_TRUE(GMatrixFunctions<FP>::all_close(rbg, rb, thrd, thrd));
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

TYPED_TEST(TestComplexMatrix, TestLSQR) {
    using FL = TypeParam;
    using MatMul = typename TestComplexMatrix<FL>::MatMul;
    typedef typename GMatrix<FL>::FP FP;
    const int sz = is_same<FP, double>::value ? 200 : 50;
    const FP conv = is_same<FP, double>::value ? 1E-8 : 1E-6;
    const FP conv2 = is_same<FP, double>::value ? 1E-7 : 1E-5;
    const FP thrd = is_same<FP, double>::value ? 1E-3 : 1E0;
    for (int i = 0; i < this->n_tests; i++) {
        MKL_INT m = Random::rand_int(1, sz);
        MKL_INT n = 1;
        int nmult = 0, niter = 0;
        FP eta = 0.05;
        GMatrix<FP> ra(dalloc_<FP>()->allocate(m * m), m, m);
        GMatrix<FP> rax(dalloc_<FP>()->allocate(m * m), m, m);
        GMatrix<FP> rb(dalloc_<FP>()->allocate(n * m), m, n);
        GMatrix<FP> rbg(dalloc_<FP>()->allocate(n * m), m, n);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> af(dalloc_<FP>()->complex_allocate(m * m), m, m);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(n * m), m, n);
        GMatrix<FL> x(dalloc_<FP>()->complex_allocate(n * m), m, n);
        GMatrix<FL> xg(dalloc_<FP>()->complex_allocate(n * m), m, n);
        Random::fill<FP>(ra.data, ra.size());
        Random::fill<FP>(rax.data, rax.size());
        Random::fill<FP>(rb.data, rb.size());
        a.clear();
        b.clear();
        GMatrixFunctions<FP>::multiply(rax, false, rax, true, ra, 1.0, 0.0);
        GMatrixFunctions<FL>::fill_complex(a, ra, GMatrix<FP>(nullptr, m, m));
        for (MKL_INT k = 0; k < m; k++)
            a(k, k) += FL(0, eta);
        GMatrixFunctions<FL>::fill_complex(b, rb, GMatrix<FP>(nullptr, m, n));
        Random::fill<FP>(rb.data, rb.size());
        GMatrixFunctions<FL>::fill_complex(x, rb, GMatrix<FP>(nullptr, m, n));
        Random::fill<FP>(rb.data, rb.size());
        GMatrixFunctions<FL>::fill_complex(x, GMatrix<FP>(nullptr, m, n), rb);
        GMatrixFunctions<FL>::copy(af, a);
        GMatrixFunctions<FL>::conjugate(af);
        MatMul mop(a), rop(af);
        // hrl: Note: The input matrix can be highly illconditioned (cond~10^5)
        //      which causes problems for lsqr.
        //      It is important to have long maxiters and small atol.
        //      It may still fail in extreme situations,
        //      in particular when m ~ 300.
        FL func = IterativeMatrixFunctions<FL>::lsqr(
            mop, rop, GDiagonalMatrix<FL>(nullptr, 0), x, b, nmult, niter,
            false, (shared_ptr<ParallelCommunicator<SZ>>)nullptr, conv,
            /*rtol*/ conv2, /*atol*/ 0., 10000);
        GMatrixFunctions<FL>::copy(xg, b);
        for (MKL_INT k = 0; k < m; k++)
            for (MKL_INT j = 0; j < m; j++)
                af(k, j) = a(j, k);
        GMatrixFunctions<FL>::linear(af, xg.flip_dims());
        GMatrixFunctions<FL>::extract_complex(xg, rbg,
                                              GMatrix<FP>(nullptr, m, n));
        GMatrixFunctions<FL>::extract_complex(x, rb,
                                              GMatrix<FP>(nullptr, m, n));
        EXPECT_TRUE(GMatrixFunctions<FP>::all_close(rbg, rb, thrd, thrd));
        GMatrixFunctions<FL>::extract_complex(xg, GMatrix<FP>(nullptr, m, n),
                                              rbg);
        GMatrixFunctions<FL>::extract_complex(x, GMatrix<FP>(nullptr, m, n),
                                              rb);
        EXPECT_TRUE(GMatrixFunctions<FP>::all_close(rbg, rb, thrd, thrd));
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
