
#include "quantum.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestMatrix : public ::testing::Test {
  protected:
    size_t isize = 1E7;
    size_t dsize = 1E7;
    static const int n_tests = 100;
    struct MatMul {
        MatrixRef a;
        MatMul(const MatrixRef &a) : a(a) {}
        void operator()(const MatrixRef &b, const MatrixRef &c) {
            MatrixFunctions::multiply(a, false, b, false, c, 1.0, 0.0);
        }
    };
    void SetUp() override {
        Random::rand_seed(0);
        ialloc = new StackAllocator<uint32_t>(new uint32_t[isize], isize);
        dalloc = new StackAllocator<double>(new double[dsize], dsize);
    }
    void TearDown() override {
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete[] ialloc->data;
        delete[] dalloc->data;
    }
};

TEST_F(TestMatrix, TestTensorProductDiagonal) {
    for (int i = 0; i < n_tests; i++) {
        int ma = Random::rand_int(1, 200), na = ma;
        int mb = Random::rand_int(1, 200), nb = mb;
        MatrixRef a(dalloc->allocate(ma * na), ma, na);
        MatrixRef b(dalloc->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc->allocate(ma * nb), ma, nb);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(b.data, b.size());
        c.clear();
        MatrixFunctions::tensor_product_diagonal(a, b, c, 2.0);
        for (int ia = 0; ia < ma; ia++)
            for (int ib = 0; ib < mb; ib++)
                ASSERT_EQ(2.0 * a(ia, ia) * b(ib, ib), c(ia, ib));
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestTensorProduct) {
    shared_ptr<BatchGEMM> batch = make_shared<BatchGEMM>();
    for (int i = 0; i < n_tests; i++) {
        int ii = Random::rand_int(0, 2), jj = Random::rand_int(0, 2);
        int ma = Random::rand_int(1, 700), na = Random::rand_int(1, 700);
        int mb = Random::rand_int(1, 700), nb = Random::rand_int(1, 700);
        if (ii == 0)
            ma = na = 1;
        else if (ii == 1)
            mb = nb = 1;
        else {
            ma = Random::rand_int(1, 30), na = Random::rand_int(1, 30);
            mb = Random::rand_int(1, 30), nb = Random::rand_int(1, 30);
        }
        int mc = ma * mb * (jj + 1), nc = na * nb * (jj + 1);
        MatrixRef a(dalloc->allocate(ma * na), ma, na);
        MatrixRef b(dalloc->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc->allocate(mc * nc), mc, nc);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(b.data, b.size());
        c.clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        MatrixRef ta = a, tb = b;
        if (conja) {
            ta = MatrixRef(dalloc->allocate(ma * na), na, ma);
            for (int ia = 0; ia < ma; ia++)
                for (int ja = 0; ja < na; ja++)
                    ta(ja, ia) = a(ia, ja);
        }
        if (conjb) {
            tb = MatrixRef(dalloc->allocate(mb * nb), nb, mb);
            for (int ib = 0; ib < mb; ib++)
                    for (int jb = 0; jb < nb; jb++)
                        tb(jb, ib) = b(ib, jb);
        }
        if (Random::rand_int(0, 2) || ii == 2)
            MatrixFunctions::tensor_product(ta, conja, tb, conjb, c, 2.0, 0);
        else {
            batch->tensor_product(ta, conja, tb, conjb, c, 2.0, 0);
            batch->perform();
            batch->clear();
        }
        for (int ia = 0; ia < ma; ia++)
            for (int ja = 0; ja < na; ja++)
                for (int ib = 0; ib < mb; ib++)
                    for (int jb = 0; jb < nb; jb++)
                        ASSERT_EQ(2.0 * a(ia, ja) * b(ib, jb),
                                  c(ia * mb + ib, ja * nb + jb));
        if (conjb)
            tb.deallocate();
        if (conja)
            ta.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestExponential) {
    for (int i = 0; i < n_tests; i++) {
        int n = Random::rand_int(1, 300);
        double t = Random::rand_double(-0.1, 0.1);
        double consta = Random::rand_double(-2.0, 2.0);
        MatrixRef a(dalloc->allocate(n * n), n, n);
        MatrixRef aa(dalloc->allocate(n), n, 1);
        MatrixRef v(dalloc->allocate(n), n, 1);
        MatrixRef u(dalloc->allocate(n), n, 1);
        MatrixRef w(dalloc->allocate(n), n, 1);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(v.data, v.size());
        for (int ki = 0; ki < n; ki++) {
            for (int kj = 0; kj < ki; kj++)
                a(kj, ki) = a(ki, kj);
            w(ki, 0) = v(ki, 0);
            aa(ki, 0) = a(ki, ki);
        }
        double anorm = MatrixFunctions::norm(aa);
        MatMul mop(a);
        int nmult = MatrixFunctions::expo_apply(
            mop, t, anorm, w, consta, false, 1E-8);
        DiagonalMatrix ww(dalloc->allocate(n), n);
        MatrixFunctions::eigs(a, ww);
        MatrixFunctions::multiply(a, false, v, false, u, 1.0, 0.0);
        for (int i = 0; i < n; i++)
            v(i, 0) = exp(t * (ww(i, i) + consta)) * u(i, 0);
        MatrixFunctions::multiply(a, true, v, false, u, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(u, w, 1E-6, 0.0));
        ww.deallocate();
        w.deallocate();
        u.deallocate();
        v.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestDavidson) {
    for (int i = 0; i < n_tests; i++) {
        int n = Random::rand_int(1, 200);
        int k = min(n, Random::rand_int(1, 10));
        int ndav = 0;
        MatrixRef a(dalloc->allocate(n * n), n, n);
        DiagonalMatrix aa(dalloc->allocate(n), n);
        DiagonalMatrix ww(dalloc->allocate(n), n);
        vector<MatrixRef> bs(k, MatrixRef(nullptr, n, 1));
        Random::fill_rand_double(a.data, a.size());
        for (int ki = 0; ki < n; ki++) {
            for (int kj = 0; kj < ki; kj++)
                a(kj, ki) = a(ki, kj);
            aa(ki, ki) = a(ki, ki);
        }
        for (int i = 0; i < k; i++) {
            bs[i].allocate();
            bs[i].clear();
            bs[i].data[i] = 1;
        }
        MatMul mop(a);
        vector<double> vw = MatrixFunctions::davidson(
            mop, aa, bs, ndav, false, 1E-8, n * k * 2, k * 2, max(5, k + 10));
        ASSERT_EQ((int)vw.size(), k);
        DiagonalMatrix w(&vw[0], k);
        MatrixFunctions::eigs(a, ww);
        DiagonalMatrix w2(ww.data, k);
        ASSERT_TRUE(MatrixFunctions::all_close(w, w2, 1E-6, 0.0));
        for (int i = 0; i < k; i++)
            ASSERT_TRUE(
                MatrixFunctions::all_close(
                    bs[i], MatrixRef(a.data + a.n * i, a.n, 1), 1E-3, 0.0) ||
                MatrixFunctions::all_close(bs[i],
                                           MatrixRef(a.data + a.n * i, a.n, 1),
                                           1E-3, 0.0, -1.0));
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestEigs) {
    for (int i = 0; i < n_tests; i++) {
        int m = Random::rand_int(1, 200);
        MatrixRef a(dalloc->allocate(m * m), m, m);
        MatrixRef ap(dalloc->allocate(m * m), m, m);
        MatrixRef ag(dalloc->allocate(m * m), m, m);
        DiagonalMatrix w(dalloc->allocate(m), m);
        Random::fill_rand_double(a.data, a.size());
        for (int ki = 0; ki < m; ki++)
            for (int kj = 0; kj <= ki; kj++)
                ap(ki, kj) = ap(kj, ki) = a(ki, kj);
        MatrixFunctions::eigs(a, w);
        MatrixFunctions::multiply(a, false, ap, true, ag, 1.0, 0.0);
        for (int k = 0; k < m; k++)
            for (int j = 0; j < m; j++)
                ag(k, j) /= w(k, k);
        ASSERT_TRUE(MatrixFunctions::all_close(ag, a, 1E-9, 0.0));
        w.deallocate();
        ag.deallocate();
        ap.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestQR) {
    for (int i = 0; i < n_tests; i++) {
        int m = Random::rand_int(1, 200);
        int n = Random::rand_int(1, 200);
        int k = min(m, n);
        MatrixRef a(dalloc->allocate(m * n), m, n);
        MatrixRef qr(dalloc->allocate(m * n), m, n);
        Random::fill_rand_double(a.data, a.size());
        MatrixRef q(dalloc->allocate(m * k), m, k);
        MatrixRef qq(dalloc->allocate(m * m), m, m);
        MatrixRef qqk(dalloc->allocate(k * k), k, k);
        MatrixRef r(dalloc->allocate(n * k), k, n);
        MatrixFunctions::qr(a, q, r);
        MatrixFunctions::multiply(q, false, r, false, qr, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(a, qr, 1E-12, 0.0));
        MatrixFunctions::multiply(q, true, q, false, qqk, 1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(qqk, IdentityMatrix(k), 1E-12, 0.0));
        if (m <= n) {
            MatrixFunctions::multiply(q, false, q, true, qq, 1.0, 0.0);
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

TEST_F(TestMatrix, TestLQ) {
    for (int i = 0; i < n_tests; i++) {
        int m = Random::rand_int(1, 200);
        int n = Random::rand_int(1, 200);
        int k = min(m, n);
        MatrixRef a(dalloc->allocate(m * n), m, n);
        MatrixRef lq(dalloc->allocate(m * n), m, n);
        Random::fill_rand_double(a.data, a.size());
        MatrixRef l(dalloc->allocate(m * k), m, k);
        MatrixRef q(dalloc->allocate(n * k), k, n);
        MatrixRef qq(dalloc->allocate(n * n), n, n);
        MatrixRef qqk(dalloc->allocate(k * k), k, k);
        MatrixFunctions::lq(a, l, q);
        MatrixFunctions::multiply(l, false, q, false, lq, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(a, lq, 1E-12, 0.0));
        MatrixFunctions::multiply(q, false, q, true, qqk, 1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(qqk, IdentityMatrix(k), 1E-12, 0.0));
        if (m >= n) {
            MatrixFunctions::multiply(q, true, q, false, qq, 1.0, 0.0);
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
