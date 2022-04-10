
#include "block2_core.hpp"
#include "gtest/gtest.h"

using namespace block2;

template <typename FL> class TestOperator : public ::testing::Test {
  protected:
    typedef typename GMatrix<FL>::FP FP;
    vector<shared_ptr<OpExpr<SZ>>> ops;
    vector<string> reprs;
    shared_ptr<OpExpr<SZ>> zero;
    void SetUp() override {
        ops.push_back(make_shared<OpElement<SZ, FL>>(OpNames::H, SiteIndex(),
                                                     SZ(0, 0, 0), 2.0));
        reprs.push_back("(2 H)");
        ops.push_back(make_shared<OpElement<SZ, FL>>(OpNames::I, SiteIndex(),
                                                     SZ(0, 0, 0)));
        reprs.push_back("I");
        ops.push_back(make_shared<OpElement<SZ, FL>>(
            OpNames::C, SiteIndex((uint16_t)0), SZ(1, 1, 0)));
        reprs.push_back("C0");
        ops.push_back(make_shared<OpElement<SZ, FL>>(
            OpNames::Q, SiteIndex(1, 1, 0), SZ(0, 0, 0)));
        reprs.push_back("Q[ 1 1 0 ]");
        ops.push_back(make_shared<OpElement<SZ, FL>>(
            OpNames::P, SiteIndex(1, 2, 1), SZ(2, 0, 0)));
        reprs.push_back("P[ 1 2 1 ]");
        ops.push_back(make_shared<OpElement<SZ, complex<FL>>>(
            OpNames::H, SiteIndex(), SZ(0, 0, 0), complex<FL>(2.0, -1.0)));
        reprs.push_back("((2,-1) H)");
        ops.push_back(make_shared<OpElement<SZ, complex<FL>>>(
            OpNames::I, SiteIndex(), SZ(0, 0, 0)));
        reprs.push_back("I");
        ops.push_back(make_shared<OpElement<SZ, complex<FL>>>(
            OpNames::C, SiteIndex((uint16_t)0), SZ(1, 1, 0)));
        reprs.push_back("C0");
        ops.push_back(make_shared<OpElement<SZ, complex<FL>>>(
            OpNames::Q, SiteIndex(1, 1, 0), SZ(0, 0, 0)));
        reprs.push_back("Q[ 1 1 0 ]");
        ops.push_back(make_shared<OpElement<SZ, complex<FL>>>(
            OpNames::P, SiteIndex(1, 2, 1), SZ(2, 0, 0)));
        reprs.push_back("P[ 1 2 1 ]");
        zero = make_shared<OpExpr<SZ>>();
    }
};

#ifdef _USE_SINGLE_PREC
typedef ::testing::Types<double, float> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestOperator, TestFL);

TYPED_TEST(TestOperator, TestSiteIndex) {
    EXPECT_EQ(SiteIndex().size(), 0);
    EXPECT_EQ(SiteIndex().spin_size(), 0);
    EXPECT_EQ(SiteIndex()[0], 0);
    EXPECT_EQ(SiteIndex((uint16_t)3).size(), 1);
    EXPECT_EQ(SiteIndex((uint16_t)3).spin_size(), 0);
    EXPECT_EQ(SiteIndex((uint16_t)3)[0], 3);
    EXPECT_EQ(SiteIndex(3, 4, 1).size(), 2);
    EXPECT_EQ(SiteIndex(3, 4, 1).spin_size(), 1);
    EXPECT_EQ(SiteIndex(3, 4, 1)[0], 3);
    EXPECT_EQ(SiteIndex(3, 4, 1)[1], 4);
    EXPECT_EQ(SiteIndex(3, 4, 1).s(), 1);
    EXPECT_EQ(SiteIndex(3, 4, 0).s(), 0);
    EXPECT_EQ(SiteIndex({3, 4}, {1, 0}).data,
              SiteIndex({4, 3}, {1, 0}).flip_spatial().data);
    EXPECT_EQ(SiteIndex({3, 4}, {1, 0}).data,
              SiteIndex({4, 3}, {0, 1}).flip().data);
}

TYPED_TEST(TestOperator, TestExpr) {
    const auto &ops = this->ops;
    const auto &reprs = this->reprs;
    const auto &zero = this->zero;
    for (size_t i = 0; i < ops.size(); i++) {
        EXPECT_EQ(ops[i]->to_str(), reprs[i]);
        EXPECT_TRUE(ops[i] * (-5.0) == (-5.0) * ops[i]);
        EXPECT_TRUE(ops[i] * (-5.0) == 2.5 * ops[i] * (-2.0));
        EXPECT_TRUE((-5.0) * ops[i] == 2.5 * ops[i] * (-2.0));
        EXPECT_EQ((ops[i] * 2.0)->to_str(), (2.0 * ops[i])->to_str());
        EXPECT_EQ(((-1.0) * ops[i] * ops[i])->to_str(),
                  (ops[i] * (-1.0) * ops[i])->to_str());
        EXPECT_EQ(((-1.0) * ops[i] * ops[i])->to_str(),
                  (ops[i] * ops[i] * (-1.0))->to_str());
        EXPECT_EQ(hash_value(ops[i] * 2.0), hash_value(2.0 * ops[i]));
        EXPECT_NE(hash_value(ops[i] * 2.0), hash_value(2.01 * ops[i]));
        EXPECT_NE(hash_value(ops[i] * 2.0), hash<double>{}(2.0));
        EXPECT_NE(hash_value(ops[i] * 2.0), hash_value(ops[i]));
        EXPECT_TRUE(ops[i] * 0.0 == zero);
        EXPECT_TRUE(0.0 * ops[i] == zero);
        EXPECT_TRUE(0.0 * ops[i] == ops[i] * 0.0);
        EXPECT_TRUE(ops[i] * zero == zero);
        EXPECT_TRUE(zero * ops[i] == zero);
        EXPECT_TRUE(zero * ops[i] == ops[i] * zero);
        EXPECT_TRUE(ops[i] + zero == ops[i]);
        EXPECT_TRUE(zero + ops[i] == ops[i]);
    }
    for (int ix = 0; ix < 2; ix++) {
        shared_ptr<OpExpr<SZ>> a = ops[0 + ix * 5], b = ops[1 + ix * 5],
                               c = ops[2 + ix * 5];
        shared_ptr<OpExpr<SZ>> d = a * 0.5;
        shared_ptr<OpExpr<SZ>> e = b * 0.2;
        EXPECT_NE((a * b)->to_str(), (b * a)->to_str());
        EXPECT_NE((a + b)->to_str(), (b + a)->to_str());
        EXPECT_TRUE(a * b * 1.0 == a * b);
        EXPECT_TRUE(!(a * b * 1.0 != a * b));
        EXPECT_TRUE(abs_value(a * b * 0.5) ==
                    abs_value(0.4 * a) * abs_value(b));
        EXPECT_TRUE(d * e * 0.5 == 0.5 * (d * e));
        EXPECT_TRUE(d * e * 0.5 != 0.6 * (d * e));
        EXPECT_TRUE(!(d * e * 0.5 == 0.6 * (d * e)));
        EXPECT_TRUE(d * e * zero == zero);
        EXPECT_TRUE(zero * d * e == zero);
        EXPECT_TRUE(zero * (d * e) == zero);
        EXPECT_TRUE(d * e * 0.0 == zero);
        EXPECT_TRUE(0.0 * d * e == zero);
        EXPECT_TRUE(0.0 * (d * e) == zero);
        EXPECT_TRUE(c * d + zero == zero + c * d);
        EXPECT_TRUE(c * d + a * b + e * a == c * d + (a * b + e * a));
        EXPECT_TRUE((c * d + a * b) + e * a == c * d + (a * b + e * a));
        EXPECT_TRUE(a + b + c == (a + b) + c);
        EXPECT_TRUE(a + (b + c) == (a + b) + c);
        EXPECT_FALSE(a + (c + b) == (a + b) + c);
        EXPECT_TRUE(a + b + c + d + e == (a + b + c) + (d + e));
        EXPECT_TRUE((a + b) + c + (d + e) == (a + b + c) + (d + e));
        // EXPECT_TRUE(a * b * c * d * e == (a * b * c) * (d * e));
        // EXPECT_TRUE((a * b) * c * (d * e) == (a * b * c) * (d * e));
        // EXPECT_TRUE(a * b * c == (a * b) * c);
        // EXPECT_TRUE(a * (b * c) == (a * b) * c);
        EXPECT_FALSE(a + b == a + b + c);
        // EXPECT_FALSE(a * b == a * b * c);
        EXPECT_FALSE(a * b == b * a);
        EXPECT_FALSE(a == zero || b == zero || c == zero);
        EXPECT_FALSE(a * b == zero || zero == a * b || a * b == a);
        EXPECT_FALSE(a + b == zero || zero == a + b || a + b == a);
        EXPECT_TRUE(0.5 * (a + b) == 0.5 * a + 0.5 * b);
        EXPECT_EQ((a + b + e)->to_str(), (a + (b + e))->to_str());
        EXPECT_TRUE((a * b) + c * d + zero == zero + ((a * b) + c * d));
        EXPECT_TRUE((a * b) + c * d == zero + ((a * b) + c * d));
        EXPECT_TRUE(a * (c + d) == a * c + a * d);
        EXPECT_TRUE((a + c + e) * d == a * d + c * d + e * d);
        EXPECT_TRUE((a + c + e) * zero == zero * (a + c + e));
        EXPECT_TRUE(zero == zero * (a + c + e));
        EXPECT_TRUE((a + c + e) * 0.0 == 0.0 * (a + c + e));
        EXPECT_TRUE(zero == 0.0 * (a + c + e));
        EXPECT_TRUE((a + c + e) * (-0.5) == (-0.5) * (a + c + e));
    }
}
