
#include "quantum.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestOperator : public ::testing::Test {
  protected:
    vector<shared_ptr<OpExpr>> ops;
    vector<string> reprs;
    shared_ptr<OpExpr> zero;
    void SetUp() override {
        ops.push_back(make_shared<OpElement>(
            OpElement(OpNames::H, vector<uint8_t>{}, SpinLabel(0, 0, 0))));
        reprs.push_back("H");
        ops.push_back(make_shared<OpElement>(
            OpElement(OpNames::I, vector<uint8_t>{}, SpinLabel(0, 0, 0))));
        reprs.push_back("I");
        ops.push_back(make_shared<OpElement>(
            OpElement(OpNames::C, vector<uint8_t>{0}, SpinLabel(1, 1, 0))));
        reprs.push_back("C0");
        ops.push_back(make_shared<OpElement>(
            OpElement(OpNames::Q, vector<uint8_t>{1, 1}, SpinLabel(0, 0, 0))));
        reprs.push_back("Q[ 1 1 ]");
        ops.push_back(make_shared<OpElement>(OpElement(
            OpNames::P, vector<uint8_t>{1, 2, 3}, SpinLabel(2, 0, 0))));
        reprs.push_back("P[ 1 2 3 ]");
        zero = make_shared<OpExpr>();
    }
};

TEST_F(TestOperator, Test) {
    for (size_t i = 0; i < ops.size(); i++) {
        EXPECT_EQ(to_str(ops[i]), reprs[i]);
        EXPECT_TRUE(ops[i] * (-5.0) == (-5.0) * ops[i]);
        EXPECT_TRUE(ops[i] * (-5.0) == 2.5 * ops[i] * (-2.0));
        EXPECT_TRUE((-5.0) * ops[i] == 2.5 * ops[i] * (-2.0));
        EXPECT_EQ(to_str(ops[i] * 2.0), to_str(2.0 * ops[i]));
        EXPECT_EQ(to_str((-1.0) * ops[i] * ops[i]),
                  to_str(ops[i] * (-1.0) * ops[i]));
        EXPECT_EQ(to_str((-1.0) * ops[i] * ops[i]),
                  to_str(ops[i] * ops[i] * (-1.0)));
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
    shared_ptr<OpExpr> a = ops[0], b = ops[1], c = ops[2];
    shared_ptr<OpExpr> d = a * 0.5;
    shared_ptr<OpExpr> e = b * 0.2;
    EXPECT_NE(to_str(a * b), to_str(b * a));
    EXPECT_NE(to_str(a + b), to_str(b + a));
    EXPECT_TRUE(a * b * 1.0 == a * b);
    EXPECT_TRUE(abs_value(a * b * 0.5) == a * b);
    EXPECT_TRUE(d * e * 0.5 == 0.5 * (d * e));
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
    EXPECT_TRUE(a * b * c * d * e == (a * b * c) * (d * e));
    EXPECT_TRUE((a * b) * c * (d * e) == (a * b * c) * (d * e));
    EXPECT_TRUE(a * b * c == (a * b) * c);
    EXPECT_TRUE(a * (b * c) == (a * b) * c);
    EXPECT_FALSE(a + b == a + b + c);
    EXPECT_FALSE(a * b == a * b * c);
    EXPECT_FALSE(a * b == b * a);
    EXPECT_FALSE(a == zero || b == zero || c == zero);
    EXPECT_FALSE(a * b == zero || zero == a * b || a * b == a);
    EXPECT_FALSE(a + b == zero || zero == a + b || a + b == a);
    EXPECT_TRUE(0.5 * (a + b) == 0.5 * a + 0.5 * b);
    EXPECT_EQ(to_str(a + b + e), to_str(a + (b + e)));
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
