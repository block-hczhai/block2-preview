
#include "block2_core.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestCG : public ::testing::Test {
  protected:
    typedef double FP;
    size_t isize = 1L << 20;
    size_t dsize = 1L << 24;
    static const int max_twoj = 300;
    static const int n_tests = 10000;
    SU2CG cg;
    double factorial[max_twoj];
    void SetUp() override {
        Random::rand_seed(0);
        cg = SU2CG(max_twoj);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        factorial[0] = 1;
        for (int i = 1; i < max_twoj; i++)
            factorial[i] = i * factorial[i - 1];
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
    int rand_proj(int tj) { return Random::rand_int(0, tj + 1) * 2 - tj; }
    int rand_triangle(int tj1, int tj2) {
        return Random::rand_int(0, ((tj1 + tj2 - abs(tj1 - tj2)) >> 1) + 1) *
                   2 +
               abs(tj1 - tj2);
    }
};

TEST_F(TestCG, TestCGTranspose) {
    for (int i = 0; i < n_tests; i++) {
        int d = Random::rand_int(0, 20);
        int l = Random::rand_int(0, 20);
        int r = rand_triangle(d, l);
        double actual = cg.transpose_cg(d, l, r);
        double expected = -1;
        for (int lz = -l; lz <= l && expected == -1; lz += 2)
            for (int rz = -r; rz <= r; rz += 2) {
                long double factor = cg.cg(l, d, r, lz, -d, rz);
                if (abs(factor) > TINY) {
                    expected =
                        ((d & 1) ? -1 : 1) * factor / cg.cg(r, d, l, rz, d, lz);
                    break;
                }
            }
        EXPECT_LT(abs(actual - expected), 1E-15);
    }
}

// <j1 m1 j2 m2 |0 0> = \delta_{j1,j2} \delta_{m1,-m2} (-1)^{j1-m1}/\sqrt{2j1+1}
TEST_F(TestCG, TestCGJ0) {
    for (int i = 0; i < n_tests; i++) {
        int tJ = 0, tj1 = Random::rand_int(0, 20), tj2 = tj1;
        int tm1 = rand_proj(tj1),
            tm2 = Random::rand_double() > 0.2 ? -tm1 : rand_proj(tj2);
        double actual = cg.cg(tj1, tj2, tJ, tm1, tm2, 0);
        double expected = (((tj1 - tm1) & 2) ? -1 : 1) / sqrt(tj1 + 1);
        expected *= (tj1 == tj2) * (tm1 == -tm2);
        EXPECT_LT(abs(actual - expected), 1E-15);
    }
}

// <j1 j1 j2 j2 | (j1+j2) (j1+j2)> = 1
TEST_F(TestCG, TestCGMEqJ) {
    for (int i = 0; i < n_tests; i++) {
        int tj1 = Random::rand_int(0, 20);
        int tj2 = Random::rand_int(0, 20);
        int tJ = tj1 + tj2;
        double actual = cg.cg(tj1, tj2, tJ, tj1, tj2, tJ);
        EXPECT_LT(abs(actual - 1), 1E-15);
    }
}

// j1=j2=J/2 and m1 = -m2
TEST_F(TestCG, TestCGEqJ1) {
    for (int i = 0; i < n_tests; i++) {
        int tj1 = Random::rand_int(0, 20), tJ = tj1 * 2;
        int tm1 = rand_proj(tj1), tm2 = -tm1;
        double actual = cg.cg(tj1, tj1, tJ, tm1, tm2, 0);
        double expected =
            factorial[tj1] * factorial[tj1] / sqrt(factorial[tj1 << 1]) /
            (factorial[(tj1 - tm1) >> 1] * factorial[(tj1 + tm1) >> 1]);
        EXPECT_LT(abs(actual - expected), 1E-15);
    }
}

// <j1 (M-1/2) 1/2  1/2|(j1 \pm 1/2) M> = \pm\sqrt{1/2 (1 \pm M/(j1 + 1/2))}
// <j1 (M+1/2) 1/2 -1/2|(j1 \pm 1/2) M> =    \sqrt{1/2 (1 \mp M/(j1 + 1/2))}
TEST_F(TestCG, TestCGHalf) {
    for (int i = 0; i < n_tests; i++) {
        int tj1 = Random::rand_int(0, 20), tj2 = 1;
        int tm2 = rand_proj(tj2);
        for (int ij = 0; ij < 2; ij++) {
            int tJ = tj1 + tj2 - ij * 2;
            if (tJ < 0)
                continue;
            int tM = rand_proj(tJ);
            double pm = ij == 0 ? 1 : -1;
            if (abs(tM - 1) <= tj1) {
                int tm1 = tM - 1, tm2 = 1;
                double actual = cg.cg(tj1, tj2, tJ, tm1, tm2, tM);
                double expected =
                    pm * sqrt(0.5 * (1 + pm * (1.0 * tM / (tj1 + 1))));
                EXPECT_LT(abs(actual - expected), 1E-15);
            } else {
                int tm1 = tM + 1, tm2 = -1;
                double actual = cg.cg(tj1, tj2, tJ, tm1, tm2, tM);
                double expected = sqrt(0.5 * (1 - pm * (1.0 * tM / (tj1 + 1))));
                EXPECT_LT(abs(actual - expected), 1E-15);
            }
        }
    }
}

// m1=m2=m3=0
// if l1+l2+l3 odd, w3j = 0
// if 2p=l1+l2+l3 even, w3j = (-)**p sqrt_delta(l1 l2 l3)
// p!/(p-l1)!(p-l2)!(p-l3)!
TEST_F(TestCG, TestW3jMZero) {
    for (int i = 0; i < n_tests; i++) {
        int tj1 = Random::rand_int(0, 20);
        int tj2 = Random::rand_int(0, 20);
        int tJ = Random::rand_double() > 0.4 ? rand_triangle(tj1, tj2)
                                             : Random::rand_int(0, 20);
        double actual = cg.wigner_3j(tj1, tj2, tJ, 0, 0, 0);
        if ((tj1 + tj2 + tJ) & 2 || tJ < abs(tj1 - tj2) || tJ > (tj1 + tj2) ||
            (tj1 & 1) || (tj2 & 1) || (tJ & 1))
            EXPECT_LT(abs(actual - 0), 1E-15);
        else {
            int p = (tj1 + tj2 + tJ) >> 2;
            double expected =
                ((p & 1) ? -1 : 1) * cg.sqrt_delta(tj1, tj2, tJ) *
                factorial[p] /
                (factorial[p - (tj1 >> 1)] * factorial[p - (tj2 >> 1)] *
                 factorial[p - (tJ >> 1)]);
            EXPECT_LT(abs(actual - expected), 1E-15);
        }
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.37)
// One of the j is null
TEST_F(TestCG, TestW6jJcZero) {
    for (int i = 0; i < n_tests; i++) {
        int tj = Random::rand_int(0, 20);
        int tjp = Random::rand_double() > 0.2 ? tj : Random::rand_int(0, 20);
        int tJ = Random::rand_int(0, 20);
        int tJp = Random::rand_double() > 0.2 ? tJ : Random::rand_int(0, 20);
        int tg = rand_triangle(tj, tJ);
        double actual = cg.wigner_6j(tj, tjp, 0, tJ, tJp, tg);
        double expected = (((tj + tJ + tg) & 2) ? -1 : 1) * (tj == tjp) *
                          (tJ == tJp) / sqrt((tj + 1) * (tJ + 1));
        EXPECT_LT(abs(actual - expected), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.38)
// One of the j is 1/2
TEST_F(TestCG, TestW6jJcHalf) {
    for (int i = 0; i < n_tests; i++) {
        int tj = Random::rand_int(0, 20);
        int tJ = Random::rand_int(0, 20);
        int tg = rand_triangle(tj, tJ);
        double actual1 = cg.wigner_6j(tj, tj + 1, 1, tJ, tJ + 1, tg + 1);
        double actual2 = cg.wigner_6j(tj, tj + 1, 1, tJ + 1, tJ, tg);
        double expected1 = (((tj + tJ + tg) & 2) ? 1 : -1) *
                           sqrt((2 + tg + tj - tJ) * (2 + tg - tj + tJ) >> 2) /
                           sqrt((tj + 1) * (tj + 2) * (tJ + 1) * (tJ + 2));
        double expected2 = (((tj + tJ + tg) & 2) ? 1 : -1) *
                           sqrt((2 - tg + tj + tJ) * (4 + tg + tj + tJ) >> 2) /
                           sqrt((tj + 1) * (tj + 2) * (tJ + 1) * (tJ + 2));
        EXPECT_LT(abs(actual1 - expected1), 1E-15);
        EXPECT_LT(abs(actual2 - expected2), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.35a)
// sum_x (-)^(2x) ( 2x + 1 ) { a b x; a b f } = 1
TEST_F(TestCG, TestW6jRacahElliotA) {
    for (int i = 0; i < n_tests; i++) {
        int ta = Random::rand_int(0, 20);
        int tb = Random::rand_int(0, 20);
        int tf = rand_triangle(ta, tb);
        long double r = 0;
        for (int tx = abs(ta - tb); tx <= ta + tb; tx++)
            r += ((tx & 1) ? -1 : 1) * (tx + 1) *
                 cg.wigner_6j(ta, tb, tx, ta, tb, tf);
        // need long double to get this accuracy
        EXPECT_LT(abs(r - 1), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.35b)
// sum_x (-)^(a+b+x) ( 2x + 1 ) { a b x; b a f } = delta_(f0) sqrt((2a+1)(2b+1))
TEST_F(TestCG, TestW6jRacahElliotB) {
    for (int i = 0; i < n_tests; i++) {
        int ta = Random::rand_int(0, 20);
        int tb = Random::rand_double() > 0.7 ? ta : Random::rand_int(0, 20);
        int tf =
            Random::rand_double() > 0.5 && ta == tb ? 0 : rand_triangle(ta, tb);
        long double r = 0;
        for (int tx = abs(ta - tb); tx <= ta + tb; tx++)
            r += (((ta + tb + tx) & 2) ? -1 : 1) * (tx + 1) *
                 cg.wigner_6j(ta, tb, tx, tb, ta, tf);
        long double expected = (tf == 0) * sqrtl((ta + 1) * (tb + 1));
        // need long double to get this accuracy
        EXPECT_LT(abs(r - expected), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.35c)
// sum_x ( 2x + 1 ) { a b x; c d f } { c d x; a b g } = delta_(fg) / (2f + 1)
TEST_F(TestCG, TestW6jRacahElliotC) {
    for (int i = 0; i < n_tests; i++) {
        int ta = Random::rand_int(0, 20);
        int tb = Random::rand_int(0, 20);
        int tc = Random::rand_int(0, 20);
        int td = Random::rand_int(0, 20);
        int tf = rand_triangle(tc, td);
        int tg = cg.triangle(ta, tb, tf) ? tf : rand_triangle(ta, tb);
        long double r = 0;
        for (int tx = abs(ta - tb); tx <= ta + tb; tx++)
            r += (tx + 1) * cg.wigner_6j(ta, tb, tx, tc, td, tf) *
                 cg.wigner_6j(tc, td, tx, ta, tb, tg);
        long double expected = 1.0L * (tf == tg) / (tf + 1) *
                               cg.triangle(ta, td, tg) *
                               cg.triangle(tc, tb, tg);
        EXPECT_LT(abs(r - expected), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.35d)
// sum_x (-)^(f+g+x) ( 2x + 1 ) { a b x; c d f } { c d x; b a g } = { a d f; b c
// g }
TEST_F(TestCG, TestW6jRacahElliotD) {
    for (int i = 0; i < n_tests; i++) {
        int ta = Random::rand_int(0, 20);
        int tb = Random::rand_int(0, 20);
        int tc = Random::rand_int(0, 20);
        int td = Random::rand_int(0, 20);
        int tf = rand_triangle(tc, td);
        int tg = rand_triangle(ta, tb);
        long double r = 0;
        for (int tx = abs(ta - tb); tx <= ta + tb; tx++)
            r += (((tf + tg + tx) & 2) ? -1 : 1) * (tx + 1) *
                 cg.wigner_6j(ta, tb, tx, tc, td, tf) *
                 cg.wigner_6j(tc, td, tx, tb, ta, tg);
        long double expected = cg.wigner_6j(ta, td, tf, tb, tc, tg);
        EXPECT_LT(abs(r - expected), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.35e)
// sum_x (-)^(a+b+c+d+e+f+g+h+x+j) ( 2x + 1 ) { a b x; c d g } { c d x; e f h }
// { e f x; b a j } = { g h j; e a d } { g h j; f b c }
TEST_F(TestCG, TestW6jRacahElliotE) {
    for (int i = 0; i < n_tests; i++) {
        int ta = Random::rand_int(0, 20);
        int tb = Random::rand_int(0, 20);
        int tc = Random::rand_double() > 0.3 ? rand_triangle(ta, tb)
                                             : Random::rand_int(0, 20);
        int td = Random::rand_double() > 0.3 ? rand_triangle(ta, tb)
                                             : Random::rand_int(0, 20);
        int te = Random::rand_double() > 0.3 ? rand_triangle(tc, td)
                                             : Random::rand_int(0, 20);
        int tf = Random::rand_double() > 0.3 ? rand_triangle(tc, td)
                                             : Random::rand_int(0, 20);
        int tg = rand_triangle(tc, td);
        int th = rand_triangle(te, tf);
        int tj = rand_triangle(ta, tb);
        long double r = 0;
        for (int tx = abs(ta - tb); tx <= ta + tb; tx++)
            r += (((ta + tb + tc + td + te + tf + tg + th + tx + tj) & 2) ? -1
                                                                          : 1) *
                 (tx + 1) * cg.wigner_6j(ta, tb, tx, tc, td, tg) *
                 cg.wigner_6j(tc, td, tx, te, tf, th) *
                 cg.wigner_6j(te, tf, tx, tb, ta, tj);
        long double expected = cg.wigner_6j(tg, th, tj, te, ta, td) *
                               cg.wigner_6j(tg, th, tj, tf, tb, tc);
        EXPECT_LT(abs(r - expected), 1E-15);
    }
}

// Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.42)
TEST_F(TestCG, TestW9jJiZero) {
    for (int i = 0; i < n_tests; i++) {
        int ta = Random::rand_int(0, 20);
        int tb = Random::rand_int(0, 20);
        int td = Random::rand_double() > 0.4 ? rand_triangle(ta, tb)
                                             : Random::rand_int(0, 20);
        int te = Random::rand_double() > 0.4 ? rand_triangle(ta, tb)
                                             : Random::rand_int(0, 20);
        int tf = rand_triangle(ta, tb);
        int tfp = cg.triangle(td, te, tf) ? tf : rand_triangle(td, te);
        int tg = rand_triangle(ta, td);
        int tgp = cg.triangle(tb, te, tg) ? tg : rand_triangle(tb, te);
        double actual = cg.wigner_9j(ta, tb, tf, td, te, tfp, tg, tgp, 0);
        double expected =
            (tf == tfp) * (tg == tgp) * (((tb + td + tf + tg) & 2) ? -1 : 1) /
            sqrt((tf + 1) * (tg + 1)) * cg.wigner_6j(ta, tb, tf, te, td, tg);
        EXPECT_LT(abs(actual - expected), 1E-15);
    }
}