
#include "block2.hpp"
#include "gtest/gtest.h"

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
    size_t isize = 1L << 20;
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

TEST_F(TestComplexMatrix, TestExponential) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(1, 300);
        complex<double> t(Random::rand_double(-0.1, 0.1),
                          Random::rand_double(-0.1, 0.1));
        double consta = Random::rand_double(-2.0, 2.0);
        MatrixRef a(dalloc_()->allocate(n * n), n, n);
        ComplexMatrixRef ca((complex<double> *)dalloc_()->allocate(n * n * 2),
                            n, n);
        ComplexMatrixRef aa((complex<double> *)dalloc_()->allocate(n * 2), n,
                            1);
        ComplexMatrixRef v((complex<double> *)dalloc_()->allocate(n * 2), n, 1);
        ComplexMatrixRef u((complex<double> *)dalloc_()->allocate(n * 2), n, 1);
        ComplexMatrixRef w((complex<double> *)dalloc_()->allocate(n * 2), n, 1);
        Random::fill_rand_double((double *)a.data, a.size());
        Random::fill_rand_double((double *)v.data, v.size() * 2);
        for (MKL_INT ki = 0; ki < n; ki++) {
            ca(ki, ki) = a(ki, ki);
            for (MKL_INT kj = 0; kj < ki; kj++)
                ca(ki, kj) = ca(kj, ki) = a(kj, ki) = a(ki, kj);
            w(ki, 0) = v(ki, 0);
            aa(ki, 0) = a(ki, ki);
        }
        double canorm = ComplexMatrixFunctions::norm(ca);
        double anorm = MatrixFunctions::norm(a);
        ASSERT_LT(abs(canorm - anorm), 1E-10);
        MatMul mop(ca);
        int nmult = ComplexMatrixFunctions::expo_apply_complex_op(
            mop, t, anorm, w, consta, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        MatrixFunctions::eigs(a, ww);
        ca.clear();
        ComplexMatrixFunctions::fill_complex(ca, a, MatrixRef(nullptr, n, n));
        ComplexMatrixFunctions::multiply(ca, false, v, false, u, 1.0, 0.0);
        for (MKL_INT i = 0; i < n; i++)
            v(i, 0) = exp(t * (ww(i, i) + consta)) * u(i, 0);
        ComplexMatrixFunctions::multiply(ca, true, v, false, u, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(u, w, 1E-6, 0.0));
        ww.deallocate();
        w.deallocate();
        u.deallocate();
        v.deallocate();
        aa.deallocate();
        ca.deallocate();
        a.deallocate();
    }
}
