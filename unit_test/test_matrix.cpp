
#include "block2_core.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestMatrix : public ::testing::Test {
  protected:
    static const int n_tests = 100;
    struct MatMul {
        MatrixRef a;
        MatMul(const MatrixRef &a) : a(a) {}
        void operator()(const MatrixRef &b, const MatrixRef &c) {
            MatrixFunctions::multiply(a, false, b, false, c, 1.0, 0.0);
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

TEST_F(TestMatrix, TestRotate) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT mk = Random::rand_int(1, 200), nk = Random::rand_int(1, 200);
        MKL_INT mb = Random::rand_int(1, 200), nb = Random::rand_int(1, 200);
        MatrixRef k(dalloc_()->allocate(mk * nk), mk, nk);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef a(dalloc_()->allocate(nb * mk), nb, mk);
        MatrixRef c(dalloc_()->allocate(mb * nk), mb, nk);
        MatrixRef ba(dalloc_()->allocate(mb * mk), mb, mk);
        Random::fill_rand_double(k.data, k.size());
        Random::fill_rand_double(b.data, b.size());
        Random::fill_rand_double(a.data, a.size());
        bool conjk = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        MatrixRef tk = k, tb = b;
        if (conjk) {
            tk = MatrixRef(dalloc_()->allocate(mk * nk), nk, mk);
            for (MKL_INT ik = 0; ik < mk; ik++)
                for (MKL_INT jk = 0; jk < nk; jk++)
                    tk(jk, ik) = k(ik, jk);
        }
        if (conjb) {
            tb = MatrixRef(dalloc_()->allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        }
        c.clear();
        MatrixFunctions::rotate(a, c, tb, conjb, tk, conjk, 2.0);
        ba.clear();
        for (MKL_INT jb = 0; jb < nb; jb++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT ja = 0; ja < mk; ja++)
                    ba(ib, ja) += b(ib, jb) * a(jb, ja) * 2.0;
        for (MKL_INT ib = 0; ib < mb; ib++)
            for (MKL_INT jk = 0; jk < nk; jk++) {
                double x = 0;
                for (MKL_INT ik = 0; ik < mk; ik++)
                    x += ba(ib, ik) * k(ik, jk);
                ASSERT_LT(abs(x - c(ib, jk)), 1E-10);
            }
        if (conjb)
            tb.deallocate();
        if (conjk)
            tk.deallocate();
        ba.deallocate();
        c.deallocate();
        a.deallocate();
        b.deallocate();
        k.deallocate();
    }
}

TEST_F(TestMatrix, TestTensorProductDiagonal) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT ma = Random::rand_int(1, 200), na = ma;
        MKL_INT mb = Random::rand_int(1, 200), nb = mb;
        MatrixRef a(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc_()->allocate(ma * nb), ma, nb);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(b.data, b.size());
        c.clear();
        MatrixFunctions::tensor_product_diagonal(a, b, c, 2.0);
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ib = 0; ib < mb; ib++)
                ASSERT_EQ(2.0 * a(ia, ia) * b(ib, ib), c(ia, ib));
        c.deallocate();
        b.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestThreeRotate) {
    shared_ptr<BatchGEMMSeq> seq = make_shared<BatchGEMMSeq>(0, SeqTypes::None);
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
        MatrixRef da(dalloc_()->allocate(mda * nda), mda, nda);
        MatrixRef db(dalloc_()->allocate(mdb * ndb), mdb, ndb);
        MatrixRef dc(dalloc_()->allocate(mdc * ndc), mdc, ndc);
        MatrixRef x(dalloc_()->allocate(mx * nx), mx, nx);
        MKL_INT mb = ll ? mdc : mx, nb = ll ? ndc : nx;
        MKL_INT mk = ll ? mx : mdc, nk = ll ? nx : ndc;
        MatrixRef a(dalloc_()->allocate(nb * mk), nb, mk);
        MatrixRef c(dalloc_()->allocate(mb * nk), mb, nk);
        MatrixRef cc(dalloc_()->allocate(mb * nk), mb, nk);
        Random::fill_rand_double(da.data, da.size());
        Random::fill_rand_double(db.data, db.size());
        Random::fill_rand_double(a.data, a.size());
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
        MatrixFunctions::three_rotate(a, c, ll ? dc : x, conjb, ll ? x : dc,
                                      conjk, da, dconja, db, dconjb, ll, 2.0,
                                      dc_stride);
        dc.clear(), cc.clear();
        MatrixFunctions::tensor_product(da, dconja, db, dconjb, dc, 1.0,
                                        dc_stride);
        MatrixFunctions::rotate(a, cc, ll ? dc : x, conjb, ll ? x : dc, conjk,
                                2.0);
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-10, 1E-10));
        c.clear();
        seq->three_rotate(a, c, ll ? dc : x, conjb, ll ? x : dc, conjk, da,
                          dconja, db, dconjb, ll, 2.0, dc_stride);
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

TEST_F(TestMatrix, TestThreeTensorProductDiagonal) {
    shared_ptr<BatchGEMM> batch = make_shared<BatchGEMM>();
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
        MatrixRef da(dalloc_()->allocate(mda * nda), mda, nda);
        MatrixRef db(dalloc_()->allocate(mdb * ndb), mdb, ndb);
        MatrixRef dc(dalloc_()->allocate(mdc * ndc), mdc, ndc);
        MatrixRef x(dalloc_()->allocate(mx * nx), mx, nx);
        MatrixRef c(dalloc_()->allocate(mdc * mx), mdc, mx);
        MatrixRef cc(dalloc_()->allocate(mdc * mx), mdc, mx);
        if (!ll)
            c = c.flip_dims(), cc = cc.flip_dims();
        Random::fill_rand_double(da.data, da.size());
        Random::fill_rand_double(db.data, db.size());
        MKL_INT dcm_stride = Random::rand_int(0, mdc - mda * mdb + 1);
        MKL_INT dcn_stride = Random::rand_int(0, ndc - nda * ndb + 1);
        dcn_stride = dcm_stride;
        MKL_INT dc_stride = dcm_stride * dc.n + dcn_stride;
        c.clear();
        MatrixFunctions::three_tensor_product_diagonal(
            ll ? dc : x, ll ? x : dc, c, da, dconja, db, dconjb, ll, 2.0,
            dc_stride);
        dc.clear(), cc.clear();
        MatrixFunctions::tensor_product(da, dconja, db, dconjb, dc, 1.0,
                                        dc_stride);
        MatrixFunctions::tensor_product_diagonal(ll ? dc : x, ll ? x : dc, cc,
                                                 2.0);
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-10, 0.0));
        c.clear();
        batch->three_tensor_product_diagonal(ll ? dc : x, ll ? x : dc, c, da,
                                             dconja, db, dconjb, ll, 2.0,
                                             dc_stride);
        batch->perform();
        batch->clear();
        ASSERT_TRUE(MatrixFunctions::all_close(c, cc, 1E-10, 0.0));
        cc.deallocate();
        c.deallocate();
        x.deallocate();
        dc.deallocate();
        db.deallocate();
        da.deallocate();
    }
}

TEST_F(TestMatrix, TestTensorProduct) {
    shared_ptr<BatchGEMM> batch = make_shared<BatchGEMM>();
    for (int i = 0; i < n_tests; i++) {
        MKL_INT ii = Random::rand_int(0, 3), jj = Random::rand_int(0, 2);
        MKL_INT ma = Random::rand_int(1, 700), na = Random::rand_int(1, 700);
        MKL_INT mb = Random::rand_int(1, 700), nb = Random::rand_int(1, 700);
        if (ii == 0)
            ma = na = 1;
        else if (ii == 1)
            mb = nb = 1;
        else {
            ma = Random::rand_int(1, 30), na = Random::rand_int(1, 30);
            mb = Random::rand_int(1, 30), nb = Random::rand_int(1, 30);
        }
        MKL_INT mc = ma * mb * (jj + 1), nc = na * nb * (jj + 1);
        MatrixRef a(dalloc_()->allocate(ma * na), ma, na);
        MatrixRef b(dalloc_()->allocate(mb * nb), mb, nb);
        MatrixRef c(dalloc_()->allocate(mc * nc), mc, nc);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(b.data, b.size());
        c.clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        MatrixRef ta = a, tb = b;
        if (conja) {
            ta = MatrixRef(dalloc_()->allocate(ma * na), na, ma);
            for (MKL_INT ia = 0; ia < ma; ia++)
                for (MKL_INT ja = 0; ja < na; ja++)
                    ta(ja, ia) = a(ia, ja);
        }
        if (conjb) {
            tb = MatrixRef(dalloc_()->allocate(mb * nb), nb, mb);
            for (MKL_INT ib = 0; ib < mb; ib++)
                for (MKL_INT jb = 0; jb < nb; jb++)
                    tb(jb, ib) = b(ib, jb);
        }
        MKL_INT cm_stride = Random::rand_int(0, mc - ma * mb + 1);
        MKL_INT cn_stride = Random::rand_int(0, nc - na * nb + 1);
        MKL_INT c_stride = cm_stride * c.n + cn_stride;
        if (Random::rand_int(0, 2))
            MatrixFunctions::tensor_product(ta, conja, tb, conjb, c, 2.0,
                                            c_stride);
        else {
            batch->tensor_product(ta, conja, tb, conjb, c, 2.0, c_stride);
            batch->perform();
            batch->clear();
        }
        for (MKL_INT ia = 0; ia < ma; ia++)
            for (MKL_INT ja = 0; ja < na; ja++)
                for (MKL_INT ib = 0; ib < mb; ib++)
                    for (MKL_INT jb = 0; jb < nb; jb++)
                        ASSERT_EQ(2.0 * a(ia, ja) * b(ib, jb),
                                  c(ia * mb + ib + cm_stride,
                                    ja * nb + jb + cn_stride));
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
        MKL_INT n = Random::rand_int(1, 300);
        double t = Random::rand_double(-0.1, 0.1);
        double consta = Random::rand_double(-2.0, 2.0);
        MatrixRef a(dalloc_()->allocate(n * n), n, n);
        MatrixRef aa(dalloc_()->allocate(n), n, 1);
        MatrixRef v(dalloc_()->allocate(n), n, 1);
        MatrixRef u(dalloc_()->allocate(n), n, 1);
        MatrixRef w(dalloc_()->allocate(n), n, 1);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(v.data, v.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = a(ki, kj);
            w(ki, 0) = v(ki, 0);
            aa(ki, 0) = a(ki, ki);
        }
        double anorm = MatrixFunctions::norm(aa);
        MatMul mop(a);
        int nmult = MatrixFunctions::expo_apply(
            mop, t, anorm, w, consta, true, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        MatrixFunctions::eigs(a, ww);
        MatrixFunctions::multiply(a, false, v, false, u, 1.0, 0.0);
        for (MKL_INT i = 0; i < n; i++)
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

TEST_F(TestMatrix, TestHarmonicDavidson) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(3, 50);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 5));
        if (k > n / 2)
            k = n / 2;
        int ndav = 0;
        MatrixRef a(dalloc_()->allocate(n * n), n, n);
        DiagonalMatrix aa(dalloc_()->allocate(n), n);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        vector<MatrixRef> bs(k, MatrixRef(nullptr, n, 1));
        Random::fill_rand_double(a.data, a.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(kj, ki) = a(ki, kj);
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
        vector<double> vw = MatrixFunctions::harmonic_davidson(
            mop, aa, bs, shift, davidson_type, ndav, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8, n * k * 100,
            -1, 2, 30);
        ASSERT_EQ((int)vw.size(), k);
        DiagonalMatrix w(&vw[0], k);
        MatrixFunctions::eigs(a, ww);
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
                         return ww.data[i] - shift >= ww.data[j] - shift;
                 });
        else if (davidson_type & DavidsonTypes::GreaterThan)
            sort(eigval_idxs.begin(), eigval_idxs.end(),
                 [&ww, shift](int i, int j) {
                     if ((shift > ww.data[i]) != (shift > ww.data[j]))
                         return shift > ww.data[j];
                     else if (shift > ww.data[i])
                         return shift - ww.data[i] > shift - ww.data[j];
                     else
                         return ww.data[i] - shift <= ww.data[j] - shift;
                 });
        // last root may be inaccurate (rare)
        for (int i = 0; i < k - 1; i++)
            ASSERT_LT(abs(ww.data[eigval_idxs[i]] - vw[i]), 1E-6);
        for (int i = 0; i < k - 1; i++)
            ASSERT_TRUE(
                MatrixFunctions::all_close(
                    bs[i], MatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1),
                    1E-3, 1E-3) ||
                MatrixFunctions::all_close(
                    bs[i], MatrixRef(a.data + a.n * eigval_idxs[i], a.n, 1),
                    1E-3, 1E-3, -1.0));
        for (int i = k - 1; i >= 0; i--)
            bs[i].deallocate();
        ww.deallocate();
        aa.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestDavidson) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(n, (MKL_INT)Random::rand_int(1, 10));
        int ndav = 0;
        MatrixRef a(dalloc_()->allocate(n * n), n, n);
        DiagonalMatrix aa(dalloc_()->allocate(n), n);
        DiagonalMatrix ww(dalloc_()->allocate(n), n);
        vector<MatrixRef> bs(k, MatrixRef(nullptr, n, 1));
        Random::fill_rand_double(a.data, a.size());
        for (MKL_INT ki = 0; ki < n; ki++) {
            for (MKL_INT kj = 0; kj < ki; kj++)
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
            mop, aa, bs, 0, DavidsonTypes::Normal, ndav, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-8, n * k * 2, -1,
            k * 2, max((MKL_INT)5, k + 10));
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

TEST_F(TestMatrix, TestMinRes) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0;
        MatrixRef a(dalloc_()->allocate(m * m), m, m);
        MatrixRef af(dalloc_()->allocate(m * m), m, m);
        MatrixRef b(dalloc_()->allocate(n * m), n, m);
        MatrixRef x(dalloc_()->allocate(n * m), n, m);
        MatrixRef xg(dalloc_()->allocate(n * m), n, m);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(b.data, b.size());
        Random::fill_rand_double(xg.data, xg.size());
        for (MKL_INT ki = 0; ki < m; ki++)
            for (MKL_INT kj = 0; kj < ki; kj++)
                a(ki, kj) = a(kj, ki);
        MatrixFunctions::copy(af, a);
        MatrixFunctions::copy(x, b);
        MatrixFunctions::linear(af, x);
        MatMul mop(a);
        double func = MatrixFunctions::minres(
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

TEST_F(TestMatrix, TestLinear) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MatrixRef a(dalloc_()->allocate(m * m), m, m);
        MatrixRef af(dalloc_()->allocate(m * m), m, m);
        MatrixRef b(dalloc_()->allocate(n * m), n, m);
        MatrixRef bg(dalloc_()->allocate(n * m), n, m);
        MatrixRef x(dalloc_()->allocate(n * m), n, m);
        Random::fill_rand_double(a.data, a.size());
        Random::fill_rand_double(b.data, b.size());
        MatrixFunctions::copy(af, a);
        MatrixFunctions::copy(x, b);
        MatrixFunctions::linear(af, x);
        MatrixFunctions::multiply(x, false, a, false, bg, 1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(bg, b, 1E-9, 1E-8));
        x.deallocate();
        bg.deallocate();
        b.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestInverse) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MatrixRef a(dalloc_()->allocate(m * m), m, m);
        MatrixRef af(dalloc_()->allocate(m * m), m, m);
        MatrixRef idt(dalloc_()->allocate(m * m), m, m);
        Random::fill_rand_double(a.data, a.size());
        MatrixFunctions::copy(af, a);
        MatrixFunctions::inverse(a);
        MatrixFunctions::multiply(a, false, af, false, idt, 1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(idt, IdentityMatrix(m), 1E-7, 0.0));
        idt.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestCG) {
    for (int i = 0; i < n_tests * 10; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0;
        MatrixRef a(dalloc_()->allocate(m * m), m, m);
        MatrixRef af(dalloc_()->allocate(m * m), m, m);
        DiagonalMatrix aa(dalloc_()->allocate(m), m);
        MatrixRef b(dalloc_()->allocate(n * m), n, m);
        MatrixRef x(dalloc_()->allocate(n * m), n, m);
        MatrixRef xg(dalloc_()->allocate(n * m), n, m);
        Random::fill_rand_double(af.data, af.size());
        Random::fill_rand_double(b.data, b.size());
        Random::fill_rand_double(xg.data, xg.size());
        MatrixFunctions::multiply(af, false, af, true, a, 1.0, 0.0);
        for (MKL_INT ki = 0; ki < m; ki++)
            aa(ki, ki) = a(ki, ki);
        MatrixFunctions::copy(af, a);
        MatrixFunctions::copy(x, b);
        MatrixFunctions::linear(af, x);
        MatMul mop(a);
        double func = MatrixFunctions::conjugate_gradient(
            mop, aa, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-14, 5000);
        ASSERT_TRUE(MatrixFunctions::all_close(xg, x, 0, 1E-3));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        aa.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestDeflatedCG) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = 1;
        int nmult = 0;
        MatrixRef a(dalloc_()->allocate(m * m), m, m);
        MatrixRef af(dalloc_()->allocate(m * m), m, m);
        MatrixRef ag(dalloc_()->allocate(m * m), m, m);
        DiagonalMatrix aa(dalloc_()->allocate(m), m);
        MatrixRef b(dalloc_()->allocate(n * m), n, m);
        MatrixRef x(dalloc_()->allocate(n * m), n, m);
        MatrixRef xg(dalloc_()->allocate(n * m), n, m);
        Random::fill_rand_double(af.data, af.size());
        Random::fill_rand_double(b.data, b.size());
        Random::fill_rand_double(xg.data, xg.size());
        MatrixFunctions::multiply(af, false, af, true, a, 1.0, 0.0);
        MatrixFunctions::copy(ag, a);
        MatrixFunctions::eigs(ag, aa);
        MatrixRef w(ag.data, m, 1);
        MatrixFunctions::copy(af, a);
        MatrixFunctions::copy(x, b);
        MatrixFunctions::linear(af, x);
        MatMul mop(a);
        for (MKL_INT ki = 0; ki < m; ki++)
            aa(ki, ki) = a(ki, ki);
        double func = MatrixFunctions::deflated_conjugate_gradient(
            mop, aa, xg.flip_dims(), b.flip_dims(), nmult, 0.0, false,
            (shared_ptr<ParallelCommunicator<SZ>>)nullptr, 1E-14, 5000, -1,
            vector<MatrixRef>{w});
        ASSERT_TRUE(MatrixFunctions::all_close(xg, x, 0, 1E-4));
        xg.deallocate();
        x.deallocate();
        b.deallocate();
        aa.deallocate();
        ag.deallocate();
        af.deallocate();
        a.deallocate();
    }
}

TEST_F(TestMatrix, TestEigs) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MatrixRef a(dalloc_()->allocate(m * m), m, m);
        MatrixRef ap(dalloc_()->allocate(m * m), m, m);
        MatrixRef ag(dalloc_()->allocate(m * m), m, m);
        DiagonalMatrix w(dalloc_()->allocate(m), m);
        Random::fill_rand_double(a.data, a.size());
        for (MKL_INT ki = 0; ki < m; ki++)
            for (MKL_INT kj = 0; kj <= ki; kj++)
                ap(ki, kj) = ap(kj, ki) = a(ki, kj);
        MatrixFunctions::eigs(a, w);
        MatrixFunctions::multiply(a, false, ap, true, ag, 1.0, 0.0);
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

TEST_F(TestMatrix, TestSVD) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(m, n);
        shared_ptr<Tensor> a = make_shared<Tensor>(vector<MKL_INT>{m, n});
        shared_ptr<Tensor> l = make_shared<Tensor>(vector<MKL_INT>{m, k});
        shared_ptr<Tensor> s = make_shared<Tensor>(vector<MKL_INT>{k});
        shared_ptr<Tensor> r = make_shared<Tensor>(vector<MKL_INT>{k, n});
        shared_ptr<Tensor> aa = make_shared<Tensor>(vector<MKL_INT>{m, n});
        shared_ptr<Tensor> kk = make_shared<Tensor>(vector<MKL_INT>{k, k});
        shared_ptr<Tensor> ll = make_shared<Tensor>(vector<MKL_INT>{m, m});
        shared_ptr<Tensor> rr = make_shared<Tensor>(vector<MKL_INT>{n, n});
        Random::fill_rand_double(a->data.data(), a->size());
        MatrixFunctions::copy(aa->ref(), a->ref());
        if (Random::rand_int(0, 2))
            MatrixFunctions::accurate_svd(a->ref(), l->ref(),
                                          s->ref().flip_dims(), r->ref(), 1E-1);
        else
            MatrixFunctions::svd(a->ref(), l->ref(), s->ref().flip_dims(),
                                 r->ref());
        MatrixFunctions::multiply(l->ref(), true, l->ref(), false, kk->ref(),
                                  1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(kk->ref(), IdentityMatrix(k),
                                               1E-12, 0.0));
        MatrixFunctions::multiply(r->ref(), false, r->ref(), true, kk->ref(),
                                  1.0, 0.0);
        ASSERT_TRUE(MatrixFunctions::all_close(kk->ref(), IdentityMatrix(k),
                                               1E-12, 0.0));
        if (m <= n) {
            MatrixFunctions::multiply(l->ref(), false, l->ref(), true,
                                      ll->ref(), 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(ll->ref(), IdentityMatrix(m),
                                                   1E-12, 0.0));
        }
        if (n <= m) {
            MatrixFunctions::multiply(r->ref(), true, r->ref(), false,
                                      rr->ref(), 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(rr->ref(), IdentityMatrix(n),
                                                   1E-12, 0.0));
        }
        MatrixRef x(r->data.data(), 1, n);
        for (MKL_INT i = 0; i < k; i++) {
            ASSERT_GE((*s)({i}), 0.0);
            MatrixFunctions::iscale(x.shift_ptr(i * n), (*s)({i}));
        }
        MatrixFunctions::multiply(l->ref(), false, r->ref(), false, a->ref(),
                                  1.0, 0.0);
        ASSERT_TRUE(
            MatrixFunctions::all_close(aa->ref(), a->ref(), 1E-12, 0.0));
    }
}

TEST_F(TestMatrix, TestQR) {
    for (int i = 0; i < n_tests; i++) {
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(m, n);
        MatrixRef a(dalloc_()->allocate(m * n), m, n);
        MatrixRef qr(dalloc_()->allocate(m * n), m, n);
        Random::fill_rand_double(a.data, a.size());
        MatrixRef q(dalloc_()->allocate(m * k), m, k);
        MatrixRef qq(dalloc_()->allocate(m * m), m, m);
        MatrixRef qqk(dalloc_()->allocate(k * k), k, k);
        MatrixRef r(dalloc_()->allocate(n * k), k, n);
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
        MKL_INT m = Random::rand_int(1, 200);
        MKL_INT n = Random::rand_int(1, 200);
        MKL_INT k = min(m, n);
        MatrixRef a(dalloc_()->allocate(m * n), m, n);
        MatrixRef lq(dalloc_()->allocate(m * n), m, n);
        Random::fill_rand_double(a.data, a.size());
        MatrixRef l(dalloc_()->allocate(m * k), m, k);
        MatrixRef q(dalloc_()->allocate(n * k), k, n);
        MatrixRef qq(dalloc_()->allocate(n * n), n, n);
        MatrixRef qqk(dalloc_()->allocate(k * k), k, k);
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
