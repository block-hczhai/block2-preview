
#include "quantum.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestMatrix : public ::testing::Test {
  protected:
    size_t isize = 1E7;
    size_t dsize = 1E7;
    static const int n_tests = 100;
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
