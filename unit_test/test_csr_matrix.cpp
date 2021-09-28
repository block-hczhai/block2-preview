
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestCSRMatrix : public ::testing::Test {
  protected:
    static const int n_tests = 200;
    size_t isize = 1L << 24;
    size_t dsize = 1L << 28;
    double sparsity = 0.5;
    void fill_sparse_double(double *data, size_t n) {
        Random::fill_rand_double(data, n);
        if (Random::rand_double() > 0.2) {
            for (size_t i = 0; i < n; i++)
                if (Random::rand_double() < sparsity)
                    data[i] = 0;
        }
    }
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

TEST_F(TestCSRMatrix, TestIadd) {
    Timer t;
    double dst = 0.0, spt = 0.0;
    for (int i = 0; i < n_tests; i++) {
        int ma = Random::rand_int(1, 200), na = Random::rand_int(1, 200);
        MatrixRef a(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef b(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef stda(dalloc_()->allocate(ma * na), ma, na);
        fill_sparse_double(a.data, a.size());
        fill_sparse_double(b.data, b.size());
        fill_sparse_double(stda.data, stda.size());
        bool conj = Random::rand_int(0, 2);
        MatrixRef tb = b;
        if (conj) {
            tb = MatrixRef(dalloc_()->allocate(ma * na), na, ma);
            for (int ib = 0; ib < ma; ib++)
                for (int jb = 0; jb < na; jb++)
                    tb(jb, ib) = b(ib, jb);
        }
        double alpha = Random::rand_double();
        MatrixFunctions::copy(a, stda);
        t.get_time();
        MatrixFunctions::iadd(a, tb, alpha, conj);
        dst += t.get_time();
        CSRMatrixRef ca, cb;
        ca.from_dense(stda);
        cb.from_dense(tb);
        memset(dalloc_()->data + dalloc_()->used, 0, a.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::iadd(ca, cb, alpha, conj);
        spt += t.get_time();
        ca.to_dense(stda);
        ASSERT_TRUE(MatrixFunctions::all_close(stda, a, 1E-10, 0.0));
        if (conj)
            tb.deallocate();
        stda.deallocate();
        b.deallocate();
        a.deallocate();
    }
    cout << "IADD dense T = " << dst << " csr T = " << spt << endl;
}

TEST_F(TestCSRMatrix, TestMultiply) {
    Timer t;
    double dst = 0.0, spt = 0.0;
    for (int i = 0; i < n_tests; i++) {
        int ma = Random::rand_int(1, 200), na = Random::rand_int(1, 200);
        int mb = na, nb = Random::rand_int(1, 200);
        MatrixRef a(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc_()->allocate(ma * nb), ma, nb);
        MatrixRef stdc(dalloc_()->allocate(ma * nb), ma, nb);
        fill_sparse_double(a.data, a.size());
        fill_sparse_double(b.data, b.size());
        fill_sparse_double(stdc.data, stdc.size());
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        MatrixRef ta = a, tb = b;
        if (conja) {
            ta = MatrixRef(dalloc_()->allocate(ma * na), na, ma);
            for (int ia = 0; ia < ma; ia++)
                for (int ja = 0; ja < na; ja++)
                    ta(ja, ia) = a(ia, ja);
        }
        if (conjb) {
            tb = MatrixRef(dalloc_()->allocate(mb * nb), nb, mb);
            for (int ib = 0; ib < mb; ib++)
                for (int jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        }
        double alpha = Random::rand_double();
        double cfactor = Random::rand_double();
        MatrixFunctions::copy(c, stdc);
        t.get_time();
        MatrixFunctions::multiply(ta, conja, tb, conjb, c, alpha, cfactor);
        dst += t.get_time();
        CSRMatrixRef ca, cb, cc;
        ca.from_dense(ta);
        cb.from_dense(tb);
        cc.from_dense(stdc);
        memset(dalloc_()->data + dalloc_()->used, 0, c.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::multiply(ca, conja, cb, conjb, cc, alpha, cfactor);
        spt += t.get_time();
        cc.to_dense(stdc);
        ASSERT_TRUE(MatrixFunctions::all_close(stdc, c, 1E-10, 0.0));
        if (conjb)
            tb.deallocate();
        if (conja)
            ta.deallocate();
        stdc.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
    cout << "MULTI dense T = " << dst << " csr T = " << spt << endl;
}

TEST_F(TestCSRMatrix, TestRotate) {
    Timer t;
    double dst = 0.0, spt = 0.0;
    for (int i = 0; i < n_tests; i++) {
        int mk = Random::rand_int(1, 300), nk = Random::rand_int(1, 300);
        int mb = Random::rand_int(1, 300), nb = Random::rand_int(1, 300);
        MatrixRef k(dalloc_()->allocate(mk * nk), mk, nk);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef a(dalloc_()->allocate(nb * mk), nb, mk);
        MatrixRef c(dalloc_()->allocate(mb * nk), mb, nk);
        MatrixRef cc(dalloc_()->allocate(mb * nk), mb, nk);
        fill_sparse_double(k.data, k.size());
        fill_sparse_double(b.data, b.size());
        fill_sparse_double(a.data, a.size());
        bool conjk = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        MatrixRef tk = k, tb = b;
        if (conjk) {
            tk = MatrixRef(dalloc_()->allocate(mk * nk), nk, mk);
            for (int ik = 0; ik < mk; ik++)
                for (int jk = 0; jk < nk; jk++)
                    tk(jk, ik) = k(ik, jk);
        }
        if (conjb) {
            tb = MatrixRef(dalloc_()->allocate(mb * nb), nb, mb);
            for (int ib = 0; ib < mb; ib++)
                for (int jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        }
        double alpha = Random::rand_double();
        c.clear();
        t.get_time();
        MatrixFunctions::rotate(a, c, tb, conjb, tk, conjk, alpha);
        dst += t.get_time();
        CSRMatrixRef ca, cb, ck;
        ca.from_dense(a);
        cb.from_dense(tb);
        ck.from_dense(tk);
        // spket * a * spbra
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::rotate(a, cc, cb, conjb, ck, conjk, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-10, 0.0));
        // spket * a * bra
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::rotate(a, cc, tb, conjb, ck, conjk, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-10, 0.0));
        // ket * a * spbra
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::rotate(a, cc, cb, conjb, tk, conjk, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-10, 0.0));
        // ket * spa * bra
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::rotate(ca, cc, tb, conjb, tk, conjk, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-10, 0.0));
        if (conjb)
            tb.deallocate();
        if (conjk)
            tk.deallocate();
        cc.deallocate();
        c.deallocate();
        a.deallocate();
        b.deallocate();
        k.deallocate();
    }
    cout << "ROTATE dense T = " << dst << " csr T = " << spt / 4 << endl;
}

TEST_F(TestCSRMatrix, TestTensorProductDiagonal) {
    Timer t;
    double dst = 0.0, spt = 0.0;
    for (int i = 0; i < n_tests; i++) {
        int ma = Random::rand_int(1, 200), na = ma;
        int mb = Random::rand_int(1, 200), nb = mb;
        MatrixRef a(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc_()->allocate(ma * nb), ma, nb);
        MatrixRef cc(dalloc_()->allocate(ma * nb), ma, nb);
        fill_sparse_double(a.data, a.size());
        fill_sparse_double(b.data, b.size());
        c.clear();
        double alpha = Random::rand_double();
        t.get_time();
        MatrixFunctions::tensor_product_diagonal(a, b, c, alpha);
        dst += t.get_time();
        CSRMatrixRef ca, cb;
        ca.from_dense(a);
        cb.from_dense(b);
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::tensor_product_diagonal(ca, b, cc, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-15, 0.0));
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::tensor_product_diagonal(a, cb, cc, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-15, 0.0));
        cc.clear();
        memset(dalloc_()->data + dalloc_()->used, 0,
               cc.size() * sizeof(double));
        t.get_time();
        CSRMatrixFunctions::tensor_product_diagonal(ca, cb, cc, alpha);
        spt += t.get_time();
        ASSERT_TRUE(MatrixFunctions::all_close(cc, c, 1E-15, 0.0));
        cc.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
    cout << "TP-DIAG dense T = " << dst << " csr T = " << spt / 3 << endl;
}

TEST_F(TestCSRMatrix, TestTensorProduct) {
    Timer t;
    double dst = 0.0, spt = 0.0;
    for (int i = 0; i < n_tests; i++) {
        int ii = Random::rand_int(0, 3), jj = Random::rand_int(0, 2);
        int ma = Random::rand_int(1, 700), na = Random::rand_int(1, 700);
        int mb = Random::rand_int(1, 700), nb = Random::rand_int(1, 700);
        if (ii == 0)
            ma = na = 1;
        else if (ii == 1)
            mb = nb = 1;
        else {
            ma = Random::rand_int(1, 50), na = Random::rand_int(1, 50);
            mb = Random::rand_int(1, 50), nb = Random::rand_int(1, 50);
        }
        int mc = ma * mb * (jj + 1), nc = na * nb * (jj + 1);
        MatrixRef a(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc_()->allocate(mc * nc), mc, nc);
        MatrixRef xc(dalloc_()->allocate(mc * nc), mc, nc);
        fill_sparse_double(a.data, a.size());
        fill_sparse_double(b.data, b.size());
        c.clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        MatrixRef ta = a, tb = b;
        if (conja) {
            ta = MatrixRef(dalloc_()->allocate(ma * na), na, ma);
            for (int ia = 0; ia < ma; ia++)
                for (int ja = 0; ja < na; ja++)
                    ta(ja, ia) = a(ia, ja);
        }
        if (conjb) {
            tb = MatrixRef(dalloc_()->allocate(mb * nb), nb, mb);
            for (int ib = 0; ib < mb; ib++)
                for (int jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        }
        int cm_stride = Random::rand_int(0, mc - ma * mb + 1);
        int cn_stride = Random::rand_int(0, nc - na * nb + 1);
        int c_stride = cm_stride * c.n + cn_stride;
        double alpha = Random::rand_double() * 0 + 1;
        t.get_time();
        MatrixFunctions::tensor_product(ta, conja, tb, conjb, c, alpha,
                                        c_stride);
        dst += t.get_time();
        CSRMatrixRef ca, cb, cc(c.m, c.n);
        ca.from_dense(ta);
        cb.from_dense(tb);
        // sp x sp
        t.get_time();
        CSRMatrixFunctions::tensor_product(ca, conja, cb, conjb, cc, alpha,
                                           c_stride);
        spt += t.get_time();
        cc.to_dense(xc);
        cc.deallocate();
        ASSERT_TRUE(MatrixFunctions::all_close(xc, c, 1E-15, 0.0));
        // sp x ds
        cc = CSRMatrixRef(c.m, c.n);
        t.get_time();
        CSRMatrixFunctions::tensor_product(ca, conja, tb, conjb, cc, alpha,
                                           c_stride);
        spt += t.get_time();
        cc.to_dense(xc);
        cc.deallocate();
        ASSERT_TRUE(MatrixFunctions::all_close(xc, c, 1E-15, 0.0));
        // ds x sp
        cc = CSRMatrixRef(c.m, c.n);
        t.get_time();
        CSRMatrixFunctions::tensor_product(ta, conja, cb, conjb, cc, alpha,
                                           c_stride);
        spt += t.get_time();
        cc.to_dense(xc);
        cc.deallocate();
        ASSERT_TRUE(MatrixFunctions::all_close(xc, c, 1E-15, 0.0));
        if (conjb)
            tb.deallocate();
        if (conja)
            ta.deallocate();
        xc.deallocate();
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
    cout << "TP dense T = " << dst << " csr T = " << spt / 3 << endl;
}
