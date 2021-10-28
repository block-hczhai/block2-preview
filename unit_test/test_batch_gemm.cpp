
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestBatchGEMM : public ::testing::Test {
  protected:
    size_t isize = 1L << 20;
    size_t dsize = 1L << 24;
    static const int n_tests = 200;
    void SetUp() override {
        Random::rand_seed(1969);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 1, 1,
            4);
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

TEST_F(TestBatchGEMM, TestRotate) {
    shared_ptr<BatchGEMMSeq<double>> seq =
        make_shared<BatchGEMMSeq<double>>(1 << 24);
    seq->mode = SeqTypes::Auto;
    for (int i = 0; i < n_tests; i++) {
        int ma = Random::rand_int(1, 100), na = Random::rand_int(1, 100);
        int mc = Random::rand_int(1, 100), nc = Random::rand_int(1, 100);
        int ncbatch = Random::rand_int(1, 30);
        int nbatch = Random::rand_int(1, 30);
        MatrixRef a(dalloc_()->allocate(ma * na * nbatch), ma, na);
        MatrixRef c(dalloc_()->allocate(mc * nc * ncbatch), mc, nc);
        MatrixRef d(dalloc_()->allocate(ncbatch), ncbatch, 1);
        MatrixRef l(dalloc_()->allocate(ma * mc), mc, ma);
        MatrixRef r(dalloc_()->allocate(na * nc), na, nc);
        Random::fill<double>(l.data, l.size());
        Random::fill<double>(r.data, r.size());
        Random::fill<double>(a.data, a.size() * nbatch);
        Random::fill<double>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        bool conjl = Random::rand_int(0, 2);
        bool conjr = Random::rand_int(0, 2);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                MatrixRef xa = a.shift_ptr(ma * na * ii);
                MatrixRef xc = MatrixRef(c.data + mc * nc * ic, mc, nc);
                seq->rotate(xa, xc, conjl ? l.flip_dims() : l, conjl,
                            conjr ? r.flip_dims() : r, conjr, d(ic, 0));
            }
        seq->auto_perform();
        MatrixRef cstd(dalloc_()->allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                MatrixRef xa = a.shift_ptr(ma * na * ii);
                MatrixFunctions::rotate(xa, cstd, conjl ? l.flip_dims() : l,
                                        conjl, conjr ? r.flip_dims() : r, conjr,
                                        d(ic, 0));
            }
            ASSERT_TRUE(MatrixFunctions::all_close(c.shift_ptr(mc * nc * ic),
                                                   cstd, 1E-10, 0.0));
        }
        cstd.deallocate();
        r.deallocate();
        l.deallocate();
        d.deallocate();
        dalloc_()->deallocate(c.data, mc * nc * ncbatch);
        dalloc_()->deallocate(a.data, ma * na * nbatch);
    }
}

TEST_F(TestBatchGEMM, TestTensorProduct) {
    shared_ptr<BatchGEMMSeq<double>> seq = make_shared<BatchGEMMSeq<double>>();
    seq->mode = SeqTypes::Auto;
    for (int i = 0; i < n_tests; i++) {
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
        MatrixRef a(dalloc_()->allocate(ma * na * nbatch), ma, na);
        MatrixRef b(dalloc_()->allocate(mb * nb * nbatch), mb, nb);
        MatrixRef c(dalloc_()->allocate(mc * nc * ncbatch), mc, nc);
        MatrixRef d(dalloc_()->allocate(ncbatch), ncbatch, 1);
        Random::fill<double>(a.data, a.size() * nbatch);
        Random::fill<double>(b.data, b.size() * nbatch);
        Random::fill<double>(d.data, d.size());
        for (int ii = 0; ii < ncbatch; ii++)
            c.shift_ptr(mc * nc * ii).clear();
        bool conja = Random::rand_int(0, 2);
        bool conjb = Random::rand_int(0, 2);
        for (int ic = 0; ic < ncbatch; ic++)
            for (int ii = 0; ii < nbatch; ii++) {
                MatrixRef xa = a.shift_ptr(ma * na * ii);
                MatrixRef xb = b.shift_ptr(mb * nb * ii);
                MatrixRef xc = c.shift_ptr(mc * nc * ic);
                for (int j = 0; j < jj + 1; j++)
                    for (int k = 0; k < jj + 1; k++)
                        seq->tensor_product(conja ? xa.flip_dims() : xa, conja,
                                            conjb ? xb.flip_dims() : xb, conjb,
                                            xc, d(ic, 0),
                                            j * nc * ma * mb + k * na * nb);
            }
        seq->auto_perform();
        MatrixRef cstd(dalloc_()->allocate(mc * nc), mc, nc);
        for (int ic = 0; ic < ncbatch; ic++) {
            cstd.clear();
            for (int ii = 0; ii < nbatch; ii++) {
                MatrixRef xa = a.shift_ptr(ma * na * ii);
                MatrixRef xb = b.shift_ptr(mb * nb * ii);
                for (int j = 0; j < jj + 1; j++)
                    for (int k = 0; k < jj + 1; k++)
                        MatrixFunctions::tensor_product(
                            conja ? xa.flip_dims() : xa, conja,
                            conjb ? xb.flip_dims() : xb, conjb, cstd, d(ic, 0),
                            j * nc * ma * mb + k * na * nb);
            }
            ASSERT_TRUE(MatrixFunctions::all_close(c.shift_ptr(mc * nc * ic),
                                                   cstd, 1E-12, 0.0));
        }
        cstd.deallocate();
        d.deallocate();
        dalloc_()->deallocate(c.data, mc * nc * ncbatch);
        dalloc_()->deallocate(b.data, mb * nb * nbatch);
        dalloc_()->deallocate(a.data, ma * na * nbatch);
    }
}
