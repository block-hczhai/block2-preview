
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestBatchGEMM : public ::testing::Test {
  protected:
    typedef typename GMatrix<FL>::FP FP;
    size_t isize = 1LL << 20;
    size_t dsize = 1LL << 24;
    static const int n_tests = 200;
    void SetUp() override {
        Random::rand_seed(1969);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 4, 4,
            4);
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

TYPED_TEST_CASE(TestBatchGEMM, TestFL);

TYPED_TEST(TestBatchGEMM, TestRotate) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-10 : 1E-5;
    shared_ptr<BatchGEMMSeq<FP>> seq = make_shared<BatchGEMMSeq<FP>>(1 << 24);
    seq->mode = SeqTypes::Auto;
    for (int i = 0; i < this->n_tests; i++) {
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mc = Random::rand_int(1, 100), nc = Random::rand_int(1, 100);
        int ncbatch = Random::rand_int(1, 30);
        int nbatch = Random::rand_int(1, 30);
        GMatrix<FP> a(dalloc_<FP>()->allocate(ma * na * nbatch), ma, na);
        GMatrix<FP> c(dalloc_<FP>()->allocate(mc * nc * ncbatch), mc, nc);
        GMatrix<FP> d(dalloc_<FP>()->allocate(ncbatch), ncbatch, 1);
        GMatrix<FP> l(dalloc_<FP>()->allocate(ma * mc), mc, ma);
        GMatrix<FP> r(dalloc_<FP>()->allocate(na * nc), na, nc);
        Random::fill<FP>(l.data, l.size());
        Random::fill<FP>(r.data, r.size());
        Random::fill<FP>(a.data, a.size() * nbatch);
        Random::fill<FP>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        bool conjl = Random::rand_int(0, 2);
        bool conjr = Random::rand_int(0, 2);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FP> xa = a.shift_ptr(ma * na * ii);
                GMatrix<FP> xc = GMatrix<FP>(c.data + mc * nc * ic, mc, nc);
                seq->rotate(xa, xc, conjl ? l.flip_dims() : l, conjl,
                            conjr ? r.flip_dims() : r, conjr, d(ic, 0));
            }
        seq->auto_perform();
        GMatrix<FP> cstd(dalloc_<FP>()->allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FP> xa = a.shift_ptr(ma * na * ii);
                GMatrixFunctions<FP>::rotate(
                    xa, cstd, conjl ? l.flip_dims() : l, conjl,
                    conjr ? r.flip_dims() : r, conjr, d(ic, 0));
            }
            ASSERT_TRUE(GMatrixFunctions<FP>::all_close(
                c.shift_ptr(mc * nc * ic), cstd, thrd, thrd));
        }
        cstd.deallocate();
        r.deallocate();
        l.deallocate();
        d.deallocate();
        dalloc_<FP>()->deallocate(c.data, mc * nc * ncbatch);
        dalloc_<FP>()->deallocate(a.data, ma * na * nbatch);
    }
}

TYPED_TEST(TestBatchGEMM, TestRotateTasked) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-10 : 1E-5;
    shared_ptr<BatchGEMMSeq<FP>> seq = make_shared<BatchGEMMSeq<FP>>(1 << 24);
    seq->mode = SeqTypes::Tasked;
    for (int i = 0; i < this->n_tests; i++) {
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mc = Random::rand_int(1, 100), nc = Random::rand_int(1, 100);
        int ncbatch = Random::rand_int(1, 30);
        int nbatch = Random::rand_int(1, 30);
        GMatrix<FP> a(dalloc_<FP>()->allocate(ma * na * nbatch), ma, na);
        GMatrix<FP> c(dalloc_<FP>()->allocate(mc * nc * ncbatch), mc, nc);
        GMatrix<FP> xxa(nullptr, ma, na);
        GMatrix<FP> xxc(nullptr, mc, nc);
        GMatrix<FP> d(dalloc_<FP>()->allocate(ncbatch), ncbatch, 1);
        GMatrix<FP> l(dalloc_<FP>()->allocate(ma * mc), mc, ma);
        GMatrix<FP> r(dalloc_<FP>()->allocate(na * nc), na, nc);
        Random::fill<FP>(l.data, l.size());
        Random::fill<FP>(r.data, r.size());
        Random::fill<FP>(a.data, a.size() * nbatch);
        Random::fill<FP>(d.data, d.size());
        for (int ic = 0; ic < ncbatch; ic++)
            c.shift_ptr(mc * nc * ic).clear();
        bool conjl = Random::rand_int(0, 2);
        bool conjr = Random::rand_int(0, 2);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FP> xa = xxa.shift_ptr(ma * na * ii);
                GMatrix<FP> xc = GMatrix<FP>(xxc.data + mc * nc * ic, mc, nc);
                seq->rotate(xa, xc, conjl ? l.flip_dims() : l, conjl,
                            conjr ? r.flip_dims() : r, conjr, d(ic, 0));
            }
        seq->operator()(a, GMatrix<FP>(c.data, mc * ncbatch, nc));
        seq->deallocate();
        seq->clear();
        GMatrix<FP> cstd(dalloc_<FP>()->allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FP> xa = a.shift_ptr(ma * na * ii);
                GMatrixFunctions<FP>::rotate(
                    xa, cstd, conjl ? l.flip_dims() : l, conjl,
                    conjr ? r.flip_dims() : r, conjr, d(ic, 0));
            }
            ASSERT_TRUE(GMatrixFunctions<FP>::all_close(
                c.shift_ptr(mc * nc * ic), cstd, thrd, thrd));
        }
        cstd.deallocate();
        r.deallocate();
        l.deallocate();
        d.deallocate();
        dalloc_<FP>()->deallocate(c.data, mc * nc * ncbatch);
        dalloc_<FP>()->deallocate(a.data, ma * na * nbatch);
    }
}

TYPED_TEST(TestBatchGEMM, TestTensorProduct) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    const FP thrd = is_same<FP, double>::value ? 1E-12 : 1E-6;
    const FP thrd2 = is_same<FP, double>::value ? 0 : 1E-6;
    shared_ptr<BatchGEMMSeq<FP>> seq = make_shared<BatchGEMMSeq<FP>>();
    seq->mode = SeqTypes::Auto;
    for (int i = 0; i < this->n_tests; i++) {
        int ii = Random::rand_int(0, 4), jj = Random::rand_int(0, 2);
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mb = Random::rand_int(1, 100), nb = Random::rand_int(1, 100);
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
        int mc = ma * mb * (jj + 1), nc = na * nb * (jj + 1);
        int ncbatch = Random::rand_int(1, 20);
        int nbatch = Random::rand_int(1, 20);
        GMatrix<FP> a(dalloc_<FP>()->allocate(ma * na * nbatch), ma, na);
        GMatrix<FP> b(dalloc_<FP>()->allocate(mb * nb * nbatch), mb, nb);
        GMatrix<FP> c(dalloc_<FP>()->allocate(mc * nc * ncbatch), mc, nc);
        GMatrix<FP> d(dalloc_<FP>()->allocate(ncbatch), ncbatch, 1);
        Random::fill<FP>(a.data, a.size() * nbatch);
        Random::fill<FP>(b.data, b.size() * nbatch);
        Random::fill<FP>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FP> xa = a.shift_ptr(ma * na * ii);
                GMatrix<FP> xb = b.shift_ptr(mb * nb * ii);
                GMatrix<FP> xc = c.shift_ptr(mc * nc * ic);
                for (int j = 0; j < jj + 1; j++)
                    for (int k = 0; k < jj + 1; k++)
                        seq->tensor_product(conja ? xa.flip_dims() : xa, conja,
                                            conjb ? xb.flip_dims() : xb, conjb,
                                            xc, d(ic, 0),
                                            j * nc * ma * mb + k * na * nb);
            }
        seq->auto_perform();
        GMatrix<FP> cstd(dalloc_<FP>()->allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FP> xa = a.shift_ptr(ma * na * ii);
                GMatrix<FP> xb = b.shift_ptr(mb * nb * ii);
                for (int j = 0; j < jj + 1; j++)
                    for (int k = 0; k < jj + 1; k++)
                        GMatrixFunctions<FP>::tensor_product(
                            conja ? xa.flip_dims() : xa, conja,
                            conjb ? xb.flip_dims() : xb, conjb, cstd, d(ic, 0),
                            j * nc * ma * mb + k * na * nb);
            }
            ASSERT_TRUE(GMatrixFunctions<FP>::all_close(
                c.shift_ptr(mc * nc * ic), cstd, thrd, thrd2));
        }
        cstd.deallocate();
        d.deallocate();
        dalloc_<FP>()->deallocate(c.data, mc * nc * ncbatch);
        dalloc_<FP>()->deallocate(b.data, mb * nb * nbatch);
        dalloc_<FP>()->deallocate(a.data, ma * na * nbatch);
    }
}

#ifdef _USE_COMPLEX

TYPED_TEST(TestBatchGEMM, TestComplexRotate) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<BatchGEMMSeq<FL>> seq = make_shared<BatchGEMMSeq<FL>>(1 << 24);
    seq->mode = SeqTypes::Auto;
    const FP thrd = is_same<FP, double>::value ? 1E-10 : 1E-5;
    for (int i = 0; i < this->n_tests; i++) {
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mc = Random::rand_int(1, 100), nc = Random::rand_int(1, 100);
        int ncbatch = Random::rand_int(1, 30);
        int nbatch = Random::rand_int(1, 30);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(ma * na * nbatch), ma,
                      na);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mc * nc * ncbatch), mc,
                      nc);
        GMatrix<FL> d(dalloc_<FP>()->complex_allocate(ncbatch), ncbatch, 1);
        GMatrix<FL> l(dalloc_<FP>()->complex_allocate(ma * mc), mc, ma);
        GMatrix<FL> r(dalloc_<FP>()->complex_allocate(na * nc), na, nc);
        Random::complex_fill<FP>(l.data, l.size());
        Random::complex_fill<FP>(r.data, r.size());
        Random::complex_fill<FP>(a.data, a.size() * nbatch);
        Random::complex_fill<FP>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        uint8_t conjl = Random::rand_int(0, 2);
        uint8_t conjr = Random::rand_int(0, 2);
        while (conjl == 2 && conjr == 2)
            conjl = Random::rand_int(0, 4), conjr = Random::rand_int(0, 4);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FL> xa = a.shift_ptr(ma * na * ii);
                GMatrix<FL> xc = GMatrix<FL>(c.data + mc * nc * ic, mc, nc);
                seq->rotate(xa, xc, (conjl & 1) ? l.flip_dims() : l, conjl,
                            (conjr & 1) ? r.flip_dims() : r, conjr, d(ic, 0));
            }
        seq->auto_perform();
        GMatrix<FL> cstd(dalloc_<FP>()->complex_allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FL> xa = a.shift_ptr(ma * na * ii);
                GMatrixFunctions<FL>::rotate(
                    xa, cstd, (conjl & 1) ? l.flip_dims() : l, conjl,
                    (conjr & 1) ? r.flip_dims() : r, conjr, d(ic, 0));
            }
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                c.shift_ptr(mc * nc * ic), cstd, thrd, thrd));
        }
        cstd.deallocate();
        r.deallocate();
        l.deallocate();
        d.deallocate();
        dalloc_<FP>()->complex_deallocate(c.data, mc * nc * ncbatch);
        dalloc_<FP>()->complex_deallocate(a.data, ma * na * nbatch);
    }
}

TYPED_TEST(TestBatchGEMM, TestComplexRotateTasked) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<BatchGEMMSeq<FL>> seq = make_shared<BatchGEMMSeq<FL>>(1 << 24);
    seq->mode = SeqTypes::Tasked;
    const FP thrd = is_same<FP, double>::value ? 1E-10 : 1E-5;
    for (int i = 0; i < this->n_tests; i++) {
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mc = Random::rand_int(1, 100), nc = Random::rand_int(1, 100);
        int ncbatch = Random::rand_int(1, 30);
        int nbatch = Random::rand_int(1, 30);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(ma * na * nbatch), ma,
                      na);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mc * nc * ncbatch), mc,
                      nc);
        GMatrix<FL> xxa(nullptr, ma, na);
        GMatrix<FL> xxc(nullptr, mc, nc);
        GMatrix<FL> d(dalloc_<FP>()->complex_allocate(ncbatch), ncbatch, 1);
        GMatrix<FL> l(dalloc_<FP>()->complex_allocate(ma * mc), mc, ma);
        GMatrix<FL> r(dalloc_<FP>()->complex_allocate(na * nc), na, nc);
        Random::complex_fill<FP>(l.data, l.size());
        Random::complex_fill<FP>(r.data, r.size());
        Random::complex_fill<FP>(a.data, a.size() * nbatch);
        Random::complex_fill<FP>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        uint8_t conjl = Random::rand_int(0, 4);
        uint8_t conjr = Random::rand_int(0, 4);
        while (conjl == 2 && conjr == 2)
            conjl = Random::rand_int(0, 4), conjr = Random::rand_int(0, 4);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FL> xa = xxa.shift_ptr(ma * na * ii);
                GMatrix<FL> xc = GMatrix<FL>(xxc.data + mc * nc * ic, mc, nc);
                seq->rotate(xa, xc, (conjl & 1) ? l.flip_dims() : l, conjl,
                            (conjr & 1) ? r.flip_dims() : r, conjr, d(ic, 0));
            }
        seq->operator()(a, GMatrix<FL>(c.data, mc * ncbatch, nc));
        seq->deallocate();
        seq->clear();
        GMatrix<FL> cstd(dalloc_<FP>()->complex_allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FL> xa = a.shift_ptr(ma * na * ii);
                GMatrixFunctions<FL>::rotate(
                    xa, cstd, (conjl & 1) ? l.flip_dims() : l, conjl,
                    (conjr & 1) ? r.flip_dims() : r, conjr, d(ic, 0));
            }
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                c.shift_ptr(mc * nc * ic), cstd, thrd, thrd));
        }
        cstd.deallocate();
        r.deallocate();
        l.deallocate();
        d.deallocate();
        dalloc_<FP>()->complex_deallocate(c.data, mc * nc * ncbatch);
        dalloc_<FP>()->complex_deallocate(a.data, ma * na * nbatch);
    }
}

TYPED_TEST(TestBatchGEMM, TestComplexTensorProduct) {
    using FL = TypeParam;
    typedef typename GMatrix<FL>::FP FP;
    shared_ptr<BatchGEMMSeq<FL>> seq = make_shared<BatchGEMMSeq<FL>>();
    seq->mode = SeqTypes::Auto;
    const FP thrd = is_same<FP, double>::value ? 1E-12 : 1E-6;
    const FP thrd2 = is_same<FP, double>::value ? 0 : 1E-6;
    for (int i = 0; i < this->n_tests; i++) {
        int ii = Random::rand_int(0, 4), jj = Random::rand_int(0, 2);
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mb = Random::rand_int(1, 100), nb = Random::rand_int(1, 100);
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
        int mc = ma * mb * (jj + 1), nc = na * nb * (jj + 1);
        int ncbatch = Random::rand_int(1, 20);
        int nbatch = Random::rand_int(1, 20);
        GMatrix<FL> a(dalloc_<FP>()->complex_allocate(ma * na * nbatch), ma,
                      na);
        GMatrix<FL> b(dalloc_<FP>()->complex_allocate(mb * nb * nbatch), mb,
                      nb);
        GMatrix<FL> c(dalloc_<FP>()->complex_allocate(mc * nc * ncbatch), mc,
                      nc);
        GMatrix<FL> d(dalloc_<FP>()->complex_allocate(ncbatch), ncbatch, 1);
        Random::complex_fill<FP>(a.data, a.size() * nbatch);
        Random::complex_fill<FP>(b.data, b.size() * nbatch);
        Random::complex_fill<FP>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FL> xa = a.shift_ptr(ma * na * ii);
                GMatrix<FL> xb = b.shift_ptr(mb * nb * ii);
                GMatrix<FL> xc = c.shift_ptr(mc * nc * ic);
                for (int j = 0; j < jj + 1; j++)
                    for (int k = 0; k < jj + 1; k++)
                        seq->tensor_product(conja ? xa.flip_dims() : xa, conja,
                                            conjb ? xb.flip_dims() : xb, conjb,
                                            xc, d(ic, 0),
                                            j * nc * ma * mb + k * na * nb);
            }
        seq->auto_perform();
        GMatrix<FL> cstd(dalloc_<FP>()->complex_allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                GMatrix<FL> xa = a.shift_ptr(ma * na * ii);
                GMatrix<FL> xb = b.shift_ptr(mb * nb * ii);
                for (int j = 0; j < jj + 1; j++)
                    for (int k = 0; k < jj + 1; k++)
                        GMatrixFunctions<FL>::tensor_product(
                            conja ? xa.flip_dims() : xa, conja,
                            conjb ? xb.flip_dims() : xb, conjb, cstd, d(ic, 0),
                            j * nc * ma * mb + k * na * nb);
            }
            ASSERT_TRUE(GMatrixFunctions<FL>::all_close(
                c.shift_ptr(mc * nc * ic), cstd, thrd, thrd2));
        }
        cstd.deallocate();
        d.deallocate();
        dalloc_<FP>()->complex_deallocate(c.data, mc * nc * ncbatch);
        dalloc_<FP>()->complex_deallocate(b.data, mb * nb * nbatch);
        dalloc_<FP>()->complex_deallocate(a.data, ma * na * nbatch);
    }
}
#endif
