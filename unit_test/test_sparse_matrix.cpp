
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename S> class TestSparseMatrix : public ::testing::Test {
  protected:
    size_t isize = 1L << 20;
    size_t dsize = 1L << 24;
    typedef double FP;
    static const int n_tests = 200;
    void SetUp() override {
        Random::rand_seed(1969);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 1, 1,
            4);
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
    shared_ptr<StateInfo<S>> random_state_info(int iter, int count,
                                               int max_n_states) {
        shared_ptr<StateInfo<S>> rr = SiteBasis<S>::get(Random::rand_int(0, 8));
        for (int j = 1; j < iter; j++)
            rr = make_shared<StateInfo<S>>(StateInfo<S>::tensor_product(
                *rr, *SiteBasis<S>::get(Random::rand_int(0, 8)),
                S(S::invalid)));
        shared_ptr<StateInfo<S>> r = make_shared<StateInfo<S>>(rr->deep_copy());
        r->n_states_total = 0;
        while (r->n_states_total == 0) {
            r = make_shared<StateInfo<S>>(rr->deep_copy());
            for (int j = 0; j < r->n; j++)
                r->n_states[j] = Random::rand_int(0, max_n_states + 1);
            r->sort_states();
            r->collect();
        }
        vector<int> x(r->n);
        for (int i = 0; i < r->n; i++)
            x[i] = i;
        for (int i = 0; i < r->n; i++)
            swap(x[i], x[Random::rand_int(i, r->n)]);
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(min(count, r->n));
        for (int i = 0; i < min(count, r->n); i++)
            b->quanta[i] = r->quanta[x[i]], b->n_states[i] = r->n_states[x[i]];
        b->sort_states();
        b->collect();
        return b;
    }
};

typedef ::testing::Types<SZ, SU2, SZK, SU2K> TestS;

TYPED_TEST_CASE(TestSparseMatrix, TestS);

TYPED_TEST(TestSparseMatrix, TestInfo) {
    using S = TypeParam;
    shared_ptr<Allocator<uint32_t>> i_alloc =
        make_shared<VectorAllocator<uint32_t>>();
    shared_ptr<Allocator<double>> d_alloc =
        make_shared<VectorAllocator<double>>();
    shared_ptr<OperatorFunctions<S, double>> opf =
        make_shared<OperatorFunctions<S, double>>(make_shared<CG<S>>());
    int iter = 10, nst = 50, nq = 20;
    for (int i = 0; i < this->n_tests; i++) {
        shared_ptr<StateInfo<S>> ksi = this->random_state_info(
            Random::rand_int(2, iter), Random::rand_int(4, nq),
            Random::rand_int(4, nst));
        shared_ptr<StateInfo<S>> ph =
            this->random_state_info(Random::rand_int(2, iter), 4, 1);
        shared_ptr<StateInfo<S>> bsi = make_shared<StateInfo<S>>(
            StateInfo<S>::tensor_product(*ksi, *ph, S(S::invalid)));
        S dq = ph->quanta[Random::rand_int(0, ph->n)];
        shared_ptr<SparseMatrixInfo<S>> minfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        minfo->initialize(*bsi, *ksi, dq, dq.is_fermion(), false);
        shared_ptr<SparseMatrix<S, double>> a =
            make_shared<SparseMatrix<S, double>>(d_alloc);
        shared_ptr<SparseMatrix<S, double>> b =
            make_shared<SparseMatrix<S, double>>(d_alloc);
        shared_ptr<SparseMatrix<S, double>> c =
            make_shared<SparseMatrix<S, double>>(d_alloc);
        a->allocate(minfo);
        a->factor = Random::rand_double(-4.0, 4.0);
        a->randomize();
        bool conjb = Random::rand_int(0, 2);
        double scale = Random::rand_double(-4.0, 4.0);
        if (!conjb) {
            b->allocate(minfo);
        } else {
            shared_ptr<SparseMatrixInfo<S>> minfo2 =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            minfo2->initialize(*ksi, *bsi, -dq, dq.is_fermion(), false);
            b->allocate(minfo2);
        }
        b->factor = Random::rand_double(-4.0, 4.0);
        b->randomize();
        c->allocate(minfo);
        c->factor = a->factor;
        c->copy_data_from(a);
        opf->iadd(c, b, scale, conjb);
        opf->iadd(c, b, -scale / 2, conjb);
        opf->iadd(c, b, -scale / 2, conjb);
        opf->iadd(c, a, -1.0, false);
        EXPECT_LT(c->norm(), 1E-10);
    }
}

TYPED_TEST(TestSparseMatrix, TestSplit) {
    using S = TypeParam;
    shared_ptr<OperatorFunctions<S, double>> opf =
        make_shared<OperatorFunctions<S, double>>(make_shared<CG<S>>());
    int iter = 5, nst = 50, nq = 20;
    for (int i = 0; i < this->n_tests; i++) {
        shared_ptr<Allocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<Allocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<StateInfo<S>> bsi = this->random_state_info(
            Random::rand_int(2, iter), Random::rand_int(4, nq),
            Random::rand_int(4, nst));
        shared_ptr<StateInfo<S>> ksi = this->random_state_info(
            Random::rand_int(2, iter), Random::rand_int(4, nq),
            Random::rand_int(4, nst));
        S target = bsi->quanta[Random::rand_int(0, bsi->n)] +
                   ksi->quanta[Random::rand_int(0, ksi->n)];
        target = target[Random::rand_int(0, target.count())];
        shared_ptr<SparseMatrixInfo<S>> minfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        minfo->initialize(*bsi, *ksi, target, false, true);
        assert(minfo->n > 0);
        shared_ptr<SparseMatrix<S, double>> a =
            make_shared<SparseMatrix<S, double>>(d_alloc);
        shared_ptr<SparseMatrix<S, double>> left =
            make_shared<SparseMatrix<S, double>>(d_alloc);
        shared_ptr<SparseMatrix<S, double>> right =
            make_shared<SparseMatrix<S, double>>(d_alloc);
        a->allocate(minfo);
        a->factor = 1.0;
        a->randomize();
        a->right_split(left, right, nq * nq);
        for (int i = 0; i < right->info->n; i++) {
            MatrixRef r = (*right)[i];
            MatrixRef tt(d_alloc->allocate(r.m * r.m), r.m, r.m);
            MatrixFunctions::multiply(r, false, r, true, tt, 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(tt, IdentityMatrix(r.m),
                                                   1E-10, 0.0));
            d_alloc->deallocate(tt.data, r.m * r.m);
        }
        a->left_split(left, right, nq * nq);
        for (int i = 0; i < left->info->n; i++) {
            MatrixRef l = (*left)[i];
            MatrixRef tt(d_alloc->allocate(l.n * l.n), l.n, l.n);
            MatrixFunctions::multiply(l, true, l, false, tt, 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(tt, IdentityMatrix(l.n),
                                                   1E-10, 0.0));
            d_alloc->deallocate(tt.data, l.n * l.n);
        }
    }
}

TYPED_TEST(TestSparseMatrix, TestComplexInfo) {
    using S = TypeParam;
    shared_ptr<Allocator<uint32_t>> i_alloc =
        make_shared<VectorAllocator<uint32_t>>();
    shared_ptr<Allocator<double>> d_alloc =
        make_shared<VectorAllocator<double>>();
    shared_ptr<OperatorFunctions<S, complex<double>>> opf =
        make_shared<OperatorFunctions<S, complex<double>>>(
            make_shared<CG<S>>());
    int iter = 10, nst = 50, nq = 20;
    for (int i = 0; i < this->n_tests; i++) {
        shared_ptr<StateInfo<S>> ksi = this->random_state_info(
            Random::rand_int(2, iter), Random::rand_int(4, nq),
            Random::rand_int(4, nst));
        shared_ptr<StateInfo<S>> ph =
            this->random_state_info(Random::rand_int(2, iter), 4, 1);
        shared_ptr<StateInfo<S>> bsi = make_shared<StateInfo<S>>(
            StateInfo<S>::tensor_product(*ksi, *ph, S(S::invalid)));
        S dq = ph->quanta[Random::rand_int(0, ph->n)];
        shared_ptr<SparseMatrixInfo<S>> minfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        minfo->initialize(*bsi, *ksi, dq, dq.is_fermion(), false);
        shared_ptr<SparseMatrix<S, complex<double>>> a =
            make_shared<SparseMatrix<S, complex<double>>>(d_alloc);
        shared_ptr<SparseMatrix<S, complex<double>>> b =
            make_shared<SparseMatrix<S, complex<double>>>(d_alloc);
        shared_ptr<SparseMatrix<S, complex<double>>> c =
            make_shared<SparseMatrix<S, complex<double>>>(d_alloc);
        a->allocate(minfo);
        a->factor = complex<double>(Random::rand_double(-4.0, 4.0),
                                    Random::rand_double(-4.0, 4.0));
        a->randomize();
        bool conjb = Random::rand_int(0, 2);
        complex<double> scale = complex<double>(Random::rand_double(-4.0, 4.0),
                                                Random::rand_double(-4.0, 4.0));
        if (!conjb) {
            b->allocate(minfo);
        } else {
            shared_ptr<SparseMatrixInfo<S>> minfo2 =
                make_shared<SparseMatrixInfo<S>>(i_alloc);
            minfo2->initialize(*ksi, *bsi, -dq, dq.is_fermion(), false);
            b->allocate(minfo2);
        }
        b->factor = complex<double>(Random::rand_double(-4.0, 4.0),
                                    Random::rand_double(-4.0, 4.0));
        b->randomize();
        c->allocate(minfo);
        c->factor = a->factor;
        c->copy_data_from(a);
        opf->iadd(c, b, scale, conjb);
        opf->iadd(c, b, -scale / 2.0, conjb);
        opf->iadd(c, b, -scale / 2.0, conjb);
        opf->iadd(c, a, -1.0, false);
        EXPECT_LT(c->norm(), 1E-10);
    }
}

TYPED_TEST(TestSparseMatrix, TestComplexSplit) {
    using S = TypeParam;
    shared_ptr<OperatorFunctions<S, complex<double>>> opf =
        make_shared<OperatorFunctions<S, complex<double>>>(
            make_shared<CG<S>>());
    int iter = 5, nst = 50, nq = 20;
    for (int i = 0; i < this->n_tests; i++) {
        shared_ptr<Allocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        shared_ptr<Allocator<double>> d_alloc =
            make_shared<VectorAllocator<double>>();
        shared_ptr<StateInfo<S>> bsi = this->random_state_info(
            Random::rand_int(2, iter), Random::rand_int(4, nq),
            Random::rand_int(4, nst));
        shared_ptr<StateInfo<S>> ksi = this->random_state_info(
            Random::rand_int(2, iter), Random::rand_int(4, nq),
            Random::rand_int(4, nst));
        S target = bsi->quanta[Random::rand_int(0, bsi->n)] +
                   ksi->quanta[Random::rand_int(0, ksi->n)];
        target = target[Random::rand_int(0, target.count())];
        shared_ptr<SparseMatrixInfo<S>> minfo =
            make_shared<SparseMatrixInfo<S>>(i_alloc);
        minfo->initialize(*bsi, *ksi, target, false, true);
        assert(minfo->n > 0);
        shared_ptr<SparseMatrix<S, complex<double>>> a =
            make_shared<SparseMatrix<S, complex<double>>>(d_alloc);
        shared_ptr<SparseMatrix<S, complex<double>>> left =
            make_shared<SparseMatrix<S, complex<double>>>(d_alloc);
        shared_ptr<SparseMatrix<S, complex<double>>> right =
            make_shared<SparseMatrix<S, complex<double>>>(d_alloc);
        a->allocate(minfo);
        a->factor = 1.0;
        a->randomize();
        a->right_split(left, right, nq * nq);
        for (int i = 0; i < right->info->n; i++) {
            ComplexMatrixRef r = (*right)[i];
            ComplexMatrixRef tt(d_alloc->complex_allocate(r.m * r.m), r.m, r.m);
            ComplexMatrixFunctions::multiply(r, false, r, 3, tt, 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(tt, IdentityMatrix(r.m),
                                                   1E-10, 0.0));
            d_alloc->complex_deallocate(tt.data, r.m * r.m);
        }
        a->left_split(left, right, nq * nq);
        for (int i = 0; i < left->info->n; i++) {
            ComplexMatrixRef l = (*left)[i];
            ComplexMatrixRef tt(d_alloc->complex_allocate(l.n * l.n), l.n, l.n);
            ComplexMatrixFunctions::multiply(l, 3, l, false, tt, 1.0, 0.0);
            ASSERT_TRUE(MatrixFunctions::all_close(tt, IdentityMatrix(l.n),
                                                   1E-10, 0.0));
            d_alloc->complex_deallocate(tt.data, l.n * l.n);
        }
    }
}
