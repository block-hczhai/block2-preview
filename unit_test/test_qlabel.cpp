
#include "block2_core.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestQ : public ::testing::Test {
  protected:
    static const int n_tests = 10000000;
    void SetUp() override { Random::rand_seed(0); }
    void TearDown() override {}
};

template <typename S> struct QZLabel {
    const static int nmin, nmax, tsmin, tsmax, pgmin, pgmax, kmin, kmax;
    int n, twos, pg, k, kmod;
    QZLabel(int kmod = 0) : kmod(kmod) {
        n = Random::rand_int(nmin, nmax + 1);
        twos = Random::rand_int(tsmin, tsmax + 1);
        pg = Random::rand_int(pgmin, pgmax + 1);
        k = kmod == 0 ? Random::rand_int(kmin, kmax + 1)
                      : Random::rand_int(0, kmod);
        if ((n & 1) != (twos & 1))
            twos = (twos & (~1)) | (n & 1);
    }
    QZLabel(int n, int twos, int pg) : n(n), twos(twos), pg(pg) {}
    QZLabel(int n, int twos, int kmod, int k, int pg)
        : n(n), twos(twos), kmod(kmod), k(k), pg(pg) {}
    bool in_range() const {
        return n >= nmin && n <= nmax && twos >= tsmin && twos <= tsmax &&
               pg >= pgmin && pg <= pgmax &&
               ((kmod == 0 && k >= kmin && k <= kmax) ||
                (kmod != 0 && k >= 0 && k < kmod && kmod >= kmin &&
                 kmod <= kmax));
    }
    int multi() const { return 1; }
    int fermion() const { return n & 1; }
    int composite_pg() const {
        return (int)(pg | (k * (pgmax - pgmin + 1)) |
                     (kmod * (pgmax - pgmin + 1) * (kmax - kmin + 1)));
    }
    QZLabel<S> operator-() const {
        return QZLabel<S>(-n, -twos, kmod,
                          k == 0 ? k : (kmod == 0 ? kmax + 1 - k : kmod - k),
                          S::pg_inv(pg));
    }
    QZLabel<S> operator+(QZLabel<S> other) const {
        return QZLabel<S>(n + other.n, twos + other.twos, kmod | other.kmod,
                          (kmod | other.kmod) == 0
                              ? (k + other.k) % (kmax + 1)
                              : (k + other.k) % (kmod | other.kmod),
                          S::pg_mul(pg, other.pg));
    }
    QZLabel<S> operator-(QZLabel<S> other) const { return *this + (-other); }
    static void check() {
        int kmod =
            Random::rand_int(0, 3)
                ? ((Random::rand_int(0, 2) || (kmin == kmax && kmax == 0))
                       ? Random::rand_int(kmin, kmax + 1)
                       : 1)
                : 0;
        QZLabel<S> qq(kmod), qq2(kmod), qq3(kmod);
        if (kmod != 0 && !Random::rand_int(0, 5))
            qq.kmod = qq.k = 0;
        if (kmod != 0 && !Random::rand_int(0, 5))
            qq2.kmod = qq2.k = 0;
        if (kmod != 0 && !Random::rand_int(0, 5))
            qq3.kmod = qq3.k = 0;
        S q(qq.n, qq.twos, qq.composite_pg());
        // getter
        EXPECT_EQ(q.n(), qq.n);
        EXPECT_EQ(q.twos(), qq.twos);
        EXPECT_EQ(q.pg(), qq.composite_pg());
        EXPECT_EQ(q.multiplicity(), qq.multi());
        EXPECT_EQ(q.is_fermion(), qq.fermion());
        // setter
        q.set_n(qq2.n);
        EXPECT_EQ(q.n(), qq2.n);
        if ((qq2.n & 1) == (qq.n & 1))
            EXPECT_EQ(q.twos(), qq.twos);
        EXPECT_EQ(q.pg(), qq.composite_pg());
        q.set_twos(qq2.twos);
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        EXPECT_EQ(q.pg(), qq.composite_pg());
        EXPECT_EQ(q.multiplicity(), qq2.multi());
        EXPECT_EQ(q.is_fermion(), qq2.fermion());
        q.set_pg(qq2.composite_pg());
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        EXPECT_EQ(q.pg(), qq2.composite_pg());
        EXPECT_EQ(q.multiplicity(), qq2.multi());
        EXPECT_EQ(q.is_fermion(), qq2.fermion());
        // setter different order
        q.set_twos(qq3.twos);
        if ((qq3.n & 1) == (qq2.n & 1))
            EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_EQ(q.pg(), qq2.composite_pg());
        q.set_pg(qq.composite_pg());
        if ((qq3.n & 1) == (qq2.n & 1))
            EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_EQ(q.pg(), qq.composite_pg());
        q.set_n(qq3.n);
        EXPECT_EQ(q.n(), qq3.n);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_EQ(q.pg(), qq.composite_pg());
        EXPECT_EQ(q.multiplicity(), qq3.multi());
        EXPECT_EQ(q.is_fermion(), qq3.fermion());
        q.set_pg(qq3.composite_pg());
        // negate
        if ((-qq3).in_range()) {
            EXPECT_EQ((-q).n(), (-qq3).n);
            EXPECT_EQ((-q).twos(), (-qq3).twos);
            EXPECT_EQ((-q).pg(), (-qq3).composite_pg());
        }
        // addition
        S q2(qq2.n, qq2.twos, qq2.composite_pg());
        QZLabel<S> qq4 = qq2 + qq3;
        if (qq4.in_range()) {
            EXPECT_EQ((q + q2).n(), qq4.n);
            EXPECT_EQ((q + q2).twos(), qq4.twos);
            EXPECT_EQ((q + q2).pg(), qq4.composite_pg());
        }
        // subtraction
        QZLabel<S> qq5 = qq3 - qq2;
        if (qq5.in_range()) {
            EXPECT_EQ((q - q2).n(), qq5.n);
            EXPECT_EQ((q - q2).twos(), qq5.twos);
            EXPECT_EQ((q - q2).pg(), qq5.composite_pg());
        }
    }
};

template <typename S> struct QULabel {
    const static int nmin, nmax, tsmin, tsmax, pgmin, pgmax, kmin, kmax;
    int n, twos, twosl, pg, k, kmod;
    QULabel(int kmod = 0) : kmod(kmod) {
        n = Random::rand_int(nmin, nmax + 1);
        twos = Random::rand_int(tsmin, tsmax + 1);
        twosl = Random::rand_int(tsmin, tsmax + 1);
        pg = Random::rand_int(pgmin, pgmax + 1);
        k = kmod == 0 ? Random::rand_int(kmin, kmax + 1)
                      : Random::rand_int(0, kmod);
        if ((n & 1) != (twos & 1))
            twos = (twos & (~1)) | (n & 1);
        if ((n & 1) != (twosl & 1))
            twosl = (twosl & (~1)) | (n & 1);
    }
    QULabel(int n, int twosl, int twos, int pg)
        : n(n), twosl(twosl), twos(twos), pg(pg) {}
    QULabel(int n, int twosl, int twos, int kmod, int k, int pg)
        : n(n), twosl(twosl), twos(twos), kmod(kmod), k(k), pg(pg) {}
    bool in_range() const {
        return n >= nmin && n <= nmax && twos >= tsmin && twos <= tsmax &&
               twosl >= tsmin && twosl <= tsmax && pg >= pgmin && pg <= pgmax &&
               ((kmod == 0 && k >= kmin && k <= kmax) ||
                (kmod != 0 && k >= 0 && k < kmod && kmod >= kmin &&
                 kmod <= kmax));
    }
    int multi() const { return twos + 1; }
    int fermion() const { return n & 1; }
    int composite_pg() const {
        return (int)(pg | (k * (pgmax - pgmin + 1)) |
                     (kmod * (pgmax - pgmin + 1) * (kmax - kmin + 1)));
    }
    static int pg_equal(int a, int b) {
        return a == b ||
               (a % (pgmax - pgmin + 1) == b % (pgmax - pgmin + 1) &&
                (((a / (pgmax - pgmin + 1)) % (kmax - kmin + 1) == 0 &&
                  b / (pgmax - pgmin + 1) == 0) ||
                 ((b / (pgmax - pgmin + 1)) % (kmax - kmin + 1) == 0 &&
                  a / (pgmax - pgmin + 1) == 0)));
    }
    QULabel<S> operator-() const {
        return QULabel<S>(-n, twosl, twos, kmod,
                          k == 0 ? k : (kmod == 0 ? kmax + 1 - k : kmod - k),
                          S::pg_inv(pg));
    }
    QULabel<S> operator+(QULabel<S> other) const {
        return QULabel<S>(n + other.n, abs(twos - other.twos),
                          twos + other.twos, kmod | other.kmod,
                          (kmod | other.kmod) == 0
                              ? (k + other.k) % (kmax + 1)
                              : (k + other.k) % (kmod | other.kmod),
                          S::pg_mul(pg, other.pg));
    }
    QULabel<S> operator-(QULabel<S> other) const { return *this + (-other); }
    QULabel<S> operator[](int i) const {
        return QULabel<S>(n, twosl + i * 2, twosl + i * 2, kmod, k, pg);
    }
    QULabel<S> get_ket() const {
        return QULabel<S>(n, twos, twos, kmod, k, pg);
    }
    QULabel<S> get_bra(QULabel<S> dq) const {
        return QULabel<S>(n + dq.n, twosl, twosl, kmod | dq.kmod,
                          (kmod | dq.kmod) == 0 ? (k + dq.k) % (kmax + 1)
                                                : (k + dq.k) % (kmod | dq.kmod),
                          S::pg_mul(pg, dq.pg));
    }
    QULabel<S> combine(QULabel<S> bra, QULabel<S> ket) const {
        return QULabel<S>(ket.n, bra.twos, ket.twos, ket.kmod, ket.k, ket.pg);
    }
    int count() const { return (twos - twosl) / 2 + 1; }
    static void check() {
        int kmod =
            Random::rand_int(0, 3)
                ? ((Random::rand_int(0, 2) || (kmin == kmax && kmax == 0))
                       ? Random::rand_int(kmin, kmax + 1)
                       : 1)
                : 0;
        QULabel<S> qq(kmod), qq2(kmod), qq3(kmod);
        if (kmod != 0 && !Random::rand_int(0, 5))
            qq.kmod = qq.k = 0;
        if (kmod != 0 && !Random::rand_int(0, 5))
            qq2.kmod = qq2.k = 0;
        if (kmod != 0 && !Random::rand_int(0, 5))
            qq3.kmod = qq3.k = 0;
        S q(qq.n, qq.twosl, qq.twos, qq.composite_pg());
        // getter
        EXPECT_EQ(q.n(), qq.n);
        EXPECT_EQ(q.twos(), qq.twos);
        EXPECT_EQ(q.twos_low(), qq.twosl);
        EXPECT_TRUE(pg_equal(q.pg(), qq.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq.multi());
        EXPECT_EQ(q.is_fermion(), qq.fermion());
        if (qq.twosl <= qq.twos) {
            EXPECT_EQ(q.count(), qq.count());
            int kk = Random::rand_int(0, qq.count());
            EXPECT_EQ(q[kk].n(), qq[kk].n);
            EXPECT_EQ(q[kk].twos(), qq[kk].twos);
            EXPECT_EQ(q[kk].twos_low(), qq[kk].twosl);
            EXPECT_TRUE(pg_equal(q[kk].pg(), qq[kk].composite_pg()));
        }
        // setter
        q.set_n(qq2.n);
        EXPECT_EQ(q.n(), qq2.n);
        if ((qq2.n & 1) == (qq.n & 1)) {
            EXPECT_EQ(q.twos(), qq.twos);
            EXPECT_EQ(q.twos_low(), qq.twosl);
        }
        EXPECT_TRUE(pg_equal(q.pg(), qq.composite_pg()));
        q.set_twos(qq2.twos);
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        if ((qq2.n & 1) == (qq.n & 1)) {
            EXPECT_EQ(q.twos_low(), qq.twosl);
        }
        EXPECT_TRUE(pg_equal(q.pg(), qq.composite_pg()));
        q.set_twos_low(qq2.twosl);
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        EXPECT_EQ(q.twos_low(), qq2.twosl);
        EXPECT_TRUE(pg_equal(q.pg(), qq.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq2.multi());
        EXPECT_EQ(q.is_fermion(), qq2.fermion());
        if (qq2.twosl <= qq2.twos) {
            EXPECT_EQ(q.count(), qq2.count());
            int kk = Random::rand_int(0, qq2.count());
            EXPECT_EQ(q[kk].n(), qq2[kk].n);
            EXPECT_EQ(q[kk].twos(), qq2[kk].twos);
            EXPECT_EQ(q[kk].twos_low(), qq2[kk].twosl);
            EXPECT_TRUE(pg_equal(q[kk].pg(), qq.composite_pg()));
        }
        q.set_pg(qq2.composite_pg());
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        EXPECT_EQ(q.twos_low(), qq2.twosl);
        EXPECT_TRUE(pg_equal(q.pg(), qq2.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq2.multi());
        EXPECT_EQ(q.is_fermion(), qq2.fermion());
        // setter different order
        q.set_twos_low(qq3.twosl);
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        if ((qq3.n & 1) == (qq2.n & 1)) {
            EXPECT_EQ(q.twos(), qq2.twos);
            EXPECT_EQ(q.n(), qq2.n);
        }
        EXPECT_TRUE(pg_equal(q.pg(), qq2.composite_pg()));
        q.set_twos(qq3.twos);
        if ((qq3.n & 1) == (qq2.n & 1)) {
            EXPECT_EQ(q.n(), qq2.n);
        }
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_TRUE(pg_equal(q.pg(), qq2.composite_pg()));
        q.set_pg(qq.composite_pg());
        if ((qq3.n & 1) == (qq2.n & 1))
            EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        EXPECT_TRUE(pg_equal(q.pg(), qq.composite_pg()));
        q.set_n(qq3.n);
        EXPECT_EQ(q.n(), qq3.n);
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_TRUE(pg_equal(q.pg(), qq.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq3.multi());
        EXPECT_EQ(q.is_fermion(), qq3.fermion());
        q.set_pg(qq3.composite_pg());
        // negate
        if ((-qq3).in_range()) {
            EXPECT_EQ((-q).n(), (-qq3).n);
            EXPECT_EQ((-q).twos(), (-qq3).twos);
            EXPECT_EQ((-q).twos_low(), (-qq3).twosl);
            EXPECT_TRUE(pg_equal((-q).pg(), (-qq3).composite_pg()));
            EXPECT_EQ((-q).multiplicity(), (-qq3).multi());
            EXPECT_EQ((-q).is_fermion(), (-qq3).fermion());
            if ((-qq3).twosl <= (-qq3).twos) {
                EXPECT_EQ((-q).count(), (-qq3).count());
                int kk = Random::rand_int(0, (-qq3).count());
                EXPECT_EQ((-q)[kk].n(), (-qq3)[kk].n);
                EXPECT_EQ((-q)[kk].twos(), (-qq3)[kk].twos);
                EXPECT_EQ((-q)[kk].twos_low(), (-qq3)[kk].twosl);
                EXPECT_TRUE(pg_equal((-q)[kk].pg(), (-qq3)[kk].composite_pg()));
            }
        }
        // addition
        q.set_twos_low(qq3.twos);
        S q2(qq2.n, qq2.twos, qq2.twos, qq2.composite_pg());
        QULabel<S> qq4 = qq2 + qq3;
        if (qq4.in_range()) {
            EXPECT_EQ((q + q2).n(), qq4.n);
            EXPECT_EQ((q + q2).twos_low(), qq4.twosl);
            EXPECT_EQ((q + q2).twos(), qq4.twos);
            EXPECT_TRUE(pg_equal((q + q2).pg(), qq4.composite_pg()));
            EXPECT_EQ((q + q2).multiplicity(), qq4.multi());
            EXPECT_EQ((q + q2).is_fermion(), qq4.fermion());
            if (qq4.twosl <= qq4.twos) {
                EXPECT_EQ((q + q2).count(), qq4.count());
                int kk = Random::rand_int(0, qq4.count());
                EXPECT_EQ((q + q2)[kk].n(), qq4[kk].n);
                EXPECT_EQ((q + q2)[kk].twos(), qq4[kk].twos);
                EXPECT_EQ((q + q2)[kk].twos_low(), qq4[kk].twosl);
                EXPECT_TRUE(
                    pg_equal((q + q2)[kk].pg(), qq4[kk].composite_pg()));
            }
        }
        // subtraction
        QULabel<S> qq5 = qq3 - qq2;
        if (qq5.in_range()) {
            EXPECT_EQ((q - q2).n(), qq5.n);
            EXPECT_EQ((q - q2).twos_low(), qq5.twosl);
            EXPECT_EQ((q - q2).twos(), qq5.twos);
            EXPECT_TRUE(pg_equal((q - q2).pg(), qq5.composite_pg()));
            EXPECT_EQ((q - q2).multiplicity(), qq5.multi());
            EXPECT_EQ((q - q2).is_fermion(), qq5.fermion());
            if (qq5.twosl <= qq5.twos) {
                EXPECT_EQ((q - q2).count(), qq5.count());
                int kk = Random::rand_int(0, qq5.count());
                EXPECT_EQ((q - q2)[kk].n(), qq5[kk].n);
                EXPECT_EQ((q - q2)[kk].twos(), qq5[kk].twos);
                EXPECT_EQ((q - q2)[kk].twos_low(), qq5[kk].twosl);
                EXPECT_TRUE(
                    pg_equal((q - q2)[kk].pg(), qq5[kk].composite_pg()));
            }
        }
        // combine
        if (qq5.in_range()) {
            QULabel<S> qc = qq5.combine(qq3, qq2);
            EXPECT_EQ((q - q2).combine(q, q2).n(), qc.n);
            // EXPECT_EQ((q - q2).combine(q, q2).twos_low(), qc.twosl);
            EXPECT_EQ((q - q2).combine(q, q2).twos(), qc.twos);
            EXPECT_TRUE(
                pg_equal((q - q2).combine(q, q2).pg(), qc.composite_pg()));
            EXPECT_EQ((q - q2).combine(q, q2).get_bra(q - q2).n(), qq3.n);
            EXPECT_EQ((q - q2).combine(q, q2).get_bra(q - q2).twos_low(),
                      qq3.twos);
            EXPECT_EQ((q - q2).combine(q, q2).get_bra(q - q2).twos(), qq3.twos);
            EXPECT_TRUE(pg_equal((q - q2).combine(q, q2).get_bra(q - q2).pg(),
                                 qq3.composite_pg()));
            EXPECT_EQ((q - q2).combine(q, q2).get_ket().n(), qq2.n);
            EXPECT_EQ((q - q2).combine(q, q2).get_ket().twos_low(), qq2.twos);
            EXPECT_EQ((q - q2).combine(q, q2).get_ket().twos(), qq2.twos);
            EXPECT_TRUE(pg_equal((q - q2).combine(q, q2).get_ket().pg(),
                                 qq2.composite_pg()));
        }
        if ((-qq5).in_range()) {
            QULabel<S> qc = (-qq5).combine(qq2, qq3);
            EXPECT_EQ((-(q - q2)).combine(q2, q).n(), qc.n);
            // EXPECT_EQ((-(q - q2)).combine(q2, q).twos_low(), qc.twosl);
            EXPECT_EQ((-(q - q2)).combine(q2, q).twos(), qc.twos);
            EXPECT_TRUE(
                pg_equal((-(q - q2)).combine(q2, q).pg(), qc.composite_pg()));
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_bra(q2 - q).n(), qq2.n);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_bra(q2 - q).twos_low(),
                      qq2.twos);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_bra(q2 - q).twos(),
                      qq2.twos);
            EXPECT_TRUE(
                pg_equal((-(q - q2)).combine(q2, q).get_bra(q2 - q).pg(),
                         qq2.composite_pg()));
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_ket().n(), qq3.n);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_ket().twos_low(),
                      qq3.twos);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_ket().twos(), qq3.twos);
            EXPECT_TRUE(pg_equal((-(q - q2)).combine(q2, q).get_ket().pg(),
                                 qq3.composite_pg()));
        }
    }
};

template <> const int QZLabel<SZShort>::nmin = -128;
template <> const int QZLabel<SZShort>::nmax = 127;
template <> const int QZLabel<SZShort>::tsmin = -128;
template <> const int QZLabel<SZShort>::tsmax = 127;
template <> const int QZLabel<SZShort>::pgmin = 0;
template <> const int QZLabel<SZShort>::pgmax = 7;
template <> const int QZLabel<SZShort>::kmin = 0;
template <> const int QZLabel<SZShort>::kmax = 0;

template <> const int QZLabel<SZLong>::nmin = -16384;
template <> const int QZLabel<SZLong>::nmax = 16383;
template <> const int QZLabel<SZLong>::tsmin = -16384;
template <> const int QZLabel<SZLong>::tsmax = 16383;
template <> const int QZLabel<SZLong>::pgmin = 0;
template <> const int QZLabel<SZLong>::pgmax = 7;
template <> const int QZLabel<SZLong>::kmin = 0;
template <> const int QZLabel<SZLong>::kmax = 0;

template <> const int QZLabel<SZLongLong>::nmin = -32768;
template <> const int QZLabel<SZLongLong>::nmax = 32767;
template <> const int QZLabel<SZLongLong>::tsmin = -32768;
template <> const int QZLabel<SZLongLong>::tsmax = 32767;
template <> const int QZLabel<SZLongLong>::pgmin = 0;
template <> const int QZLabel<SZLongLong>::pgmax = 65535;
template <> const int QZLabel<SZLongLong>::kmin = 0;
template <> const int QZLabel<SZLongLong>::kmax = 0;

template <> const int QZLabel<SZKLong>::nmin = -32768;
template <> const int QZLabel<SZKLong>::nmax = 32767;
template <> const int QZLabel<SZKLong>::tsmin = -32768;
template <> const int QZLabel<SZKLong>::tsmax = 32767;
template <> const int QZLabel<SZKLong>::pgmin = 0;
template <> const int QZLabel<SZKLong>::pgmax = 15;
template <> const int QZLabel<SZKLong>::kmin = 0;
template <> const int QZLabel<SZKLong>::kmax = 16383;

template <> const int QZLabel<SZLZ>::nmin = -1024;
template <> const int QZLabel<SZLZ>::nmax = 1023;
template <> const int QZLabel<SZLZ>::tsmin = -1024;
template <> const int QZLabel<SZLZ>::tsmax = 1023;
template <> const int QZLabel<SZLZ>::pgmin = -1024;
template <> const int QZLabel<SZLZ>::pgmax = 1023;
template <> const int QZLabel<SZLZ>::kmin = 0;
template <> const int QZLabel<SZLZ>::kmax = 0;

template <> const int QULabel<SU2Short>::nmin = -128;
template <> const int QULabel<SU2Short>::nmax = 127;
template <> const int QULabel<SU2Short>::tsmin = 0;
template <> const int QULabel<SU2Short>::tsmax = 127;
template <> const int QULabel<SU2Short>::pgmin = 0;
template <> const int QULabel<SU2Short>::pgmax = 7;
template <> const int QULabel<SU2Short>::kmin = 0;
template <> const int QULabel<SU2Short>::kmax = 0;

template <> const int QULabel<SU2Long>::nmin = -1024;
template <> const int QULabel<SU2Long>::nmax = 1023;
template <> const int QULabel<SU2Long>::tsmin = 0;
template <> const int QULabel<SU2Long>::tsmax = 1023;
template <> const int QULabel<SU2Long>::pgmin = 0;
template <> const int QULabel<SU2Long>::pgmax = 7;
template <> const int QULabel<SU2Long>::kmin = 0;
template <> const int QULabel<SU2Long>::kmax = 0;

template <> const int QULabel<SU2LongLong>::nmin = -32768;
template <> const int QULabel<SU2LongLong>::nmax = 32767;
template <> const int QULabel<SU2LongLong>::tsmin = 0;
template <> const int QULabel<SU2LongLong>::tsmax = 65535;
template <> const int QULabel<SU2LongLong>::pgmin = 0;
template <> const int QULabel<SU2LongLong>::pgmax = 65535;
template <> const int QULabel<SU2LongLong>::kmin = 0;
template <> const int QULabel<SU2LongLong>::kmax = 0;

template <> const int QULabel<SU2KLong>::nmin = -2048;
template <> const int QULabel<SU2KLong>::nmax = 2047;
template <> const int QULabel<SU2KLong>::tsmin = 0;
template <> const int QULabel<SU2KLong>::tsmax = 4095;
template <> const int QULabel<SU2KLong>::pgmin = 0;
template <> const int QULabel<SU2KLong>::pgmax = 15;
template <> const int QULabel<SU2KLong>::kmin = 0;
template <> const int QULabel<SU2KLong>::kmax = 4095;

template <> const int QULabel<SU2LZ>::nmin = -256;
template <> const int QULabel<SU2LZ>::nmax = 255;
template <> const int QULabel<SU2LZ>::tsmin = 0;
template <> const int QULabel<SU2LZ>::tsmax = 255;
template <> const int QULabel<SU2LZ>::pgmin = -256;
template <> const int QULabel<SU2LZ>::pgmax = 255;
template <> const int QULabel<SU2LZ>::kmin = 0;
template <> const int QULabel<SU2LZ>::kmax = 0;

TEST_F(TestQ, TestSZShort) {
    for (int i = 0; i < n_tests; i++)
        QZLabel<SZShort>::check();
}

TEST_F(TestQ, TestSZLong) {
    for (int i = 0; i < n_tests; i++)
        QZLabel<SZLong>::check();
}

TEST_F(TestQ, TestSZLongLong) {
    for (int i = 0; i < n_tests; i++)
        QZLabel<SZLongLong>::check();
}

TEST_F(TestQ, TestSZKLong) {
    for (int i = 0; i < n_tests; i++)
        QZLabel<SZKLong>::check();
}

TEST_F(TestQ, TestSZLZ) {
    for (int i = 0; i < n_tests; i++)
        QZLabel<SZLZ>::check();
}

TEST_F(TestQ, TestSU2Short) {
    for (int i = 0; i < n_tests; i++)
        QULabel<SU2Short>::check();
}

TEST_F(TestQ, TestSU2Long) {
    for (int i = 0; i < n_tests; i++)
        QULabel<SU2Long>::check();
}

TEST_F(TestQ, TestSU2LongLong) {
    for (int i = 0; i < n_tests; i++)
        QULabel<SU2LongLong>::check();
}

TEST_F(TestQ, TestSU2KLong) {
    for (int i = 0; i < n_tests; i++)
        QULabel<SU2KLong>::check();
}

TEST_F(TestQ, TestSU2LZ) {
    for (int i = 0; i < n_tests; i++)
        QULabel<SU2LZ>::check();
}

TEST_F(TestQ, TestSAnySU2) {
    typedef SAny S;
    for (int i = 0; i < n_tests; i++) {
        QULabel<SU2> qq(0), qq2(0), qq3(0);
        S q = S::init_su2(qq.n, qq.twosl, qq.twos, qq.composite_pg());
        // getter
        EXPECT_EQ(q.n(), qq.n);
        EXPECT_EQ(q.twos(), qq.twos);
        EXPECT_EQ(q.twos_low(), qq.twosl);
        EXPECT_TRUE(S::pg_equal(q.pg(), qq.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq.multi());
        EXPECT_EQ(q.is_fermion(), qq.fermion());
        if (qq.twosl <= qq.twos) {
            EXPECT_EQ(q.count(), qq.count());
            int kk = Random::rand_int(0, qq.count());
            EXPECT_EQ(q[kk].n(), qq[kk].n);
            EXPECT_EQ(q[kk].twos(), qq[kk].twos);
            EXPECT_EQ(q[kk].twos_low(), qq[kk].twosl);
            EXPECT_TRUE(S::pg_equal(q[kk].pg(), qq[kk].composite_pg()));
        }
        // setter
        q.set_n(qq2.n);
        EXPECT_EQ(q.n(), qq2.n);
        if ((qq2.n & 1) == (qq.n & 1)) {
            EXPECT_EQ(q.twos(), qq.twos);
            EXPECT_EQ(q.twos_low(), qq.twosl);
        }
        EXPECT_TRUE(S::pg_equal(q.pg(), qq.composite_pg()));
        q.set_twos(qq2.twos);
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        if ((qq2.n & 1) == (qq.n & 1)) {
            EXPECT_EQ(q.twos_low(), qq.twosl);
        }
        EXPECT_TRUE(S::pg_equal(q.pg(), qq.composite_pg()));
        q.set_twos_low(qq2.twosl);
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        EXPECT_EQ(q.twos_low(), qq2.twosl);
        EXPECT_TRUE(S::pg_equal(q.pg(), qq.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq2.multi());
        EXPECT_EQ(q.is_fermion(), qq2.fermion());
        if (qq2.twosl <= qq2.twos) {
            EXPECT_EQ(q.count(), qq2.count());
            int kk = Random::rand_int(0, qq2.count());
            EXPECT_EQ(q[kk].n(), qq2[kk].n);
            EXPECT_EQ(q[kk].twos(), qq2[kk].twos);
            EXPECT_EQ(q[kk].twos_low(), qq2[kk].twosl);
            EXPECT_TRUE(S::pg_equal(q[kk].pg(), qq.composite_pg()));
        }
        q.set_pg(qq2.composite_pg());
        EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq2.twos);
        EXPECT_EQ(q.twos_low(), qq2.twosl);
        EXPECT_TRUE(S::pg_equal(q.pg(), qq2.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq2.multi());
        EXPECT_EQ(q.is_fermion(), qq2.fermion());
        // setter different order
        q.set_twos_low(qq3.twosl);
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        if ((qq3.n & 1) == (qq2.n & 1)) {
            EXPECT_EQ(q.twos(), qq2.twos);
            EXPECT_EQ(q.n(), qq2.n);
        }
        EXPECT_TRUE(S::pg_equal(q.pg(), qq2.composite_pg()));
        q.set_twos(qq3.twos);
        if ((qq3.n & 1) == (qq2.n & 1)) {
            EXPECT_EQ(q.n(), qq2.n);
        }
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_TRUE(S::pg_equal(q.pg(), qq2.composite_pg()));
        q.set_pg(qq.composite_pg());
        if ((qq3.n & 1) == (qq2.n & 1))
            EXPECT_EQ(q.n(), qq2.n);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        EXPECT_TRUE(S::pg_equal(q.pg(), qq.composite_pg()));
        q.set_n(qq3.n);
        EXPECT_EQ(q.n(), qq3.n);
        EXPECT_EQ(q.twos_low(), qq3.twosl);
        EXPECT_EQ(q.twos(), qq3.twos);
        EXPECT_TRUE(S::pg_equal(q.pg(), qq.composite_pg()));
        EXPECT_EQ(q.multiplicity(), qq3.multi());
        EXPECT_EQ(q.is_fermion(), qq3.fermion());
        q.set_pg(qq3.composite_pg());
        // negate
        if ((-qq3).in_range()) {
            EXPECT_EQ((-q).n(), (-qq3).n);
            EXPECT_EQ((-q).twos(), (-qq3).twos);
            EXPECT_EQ((-q).twos_low(), (-qq3).twosl);
            EXPECT_TRUE(S::pg_equal((-q).pg(), (-qq3).composite_pg()));
            EXPECT_EQ((-q).multiplicity(), (-qq3).multi());
            EXPECT_EQ((-q).is_fermion(), (-qq3).fermion());
            if ((-qq3).twosl <= (-qq3).twos) {
                EXPECT_EQ((-q).count(), (-qq3).count());
                int kk = Random::rand_int(0, (-qq3).count());
                EXPECT_EQ((-q)[kk].n(), (-qq3)[kk].n);
                EXPECT_EQ((-q)[kk].twos(), (-qq3)[kk].twos);
                EXPECT_EQ((-q)[kk].twos_low(), (-qq3)[kk].twosl);
                EXPECT_TRUE(
                    S::pg_equal((-q)[kk].pg(), (-qq3)[kk].composite_pg()));
            }
        }
        // addition
        q.set_twos_low(qq3.twos);
        S q2 = S::init_su2(qq2.n, qq2.twos, qq2.twos, qq2.composite_pg());
        QULabel<SU2> qq4 = qq2 + qq3;
        if (qq4.in_range()) {
            EXPECT_EQ((q + q2).n(), qq4.n);
            EXPECT_EQ((q + q2).twos_low(), qq4.twosl);
            EXPECT_EQ((q + q2).twos(), qq4.twos);
            EXPECT_TRUE(S::pg_equal((q + q2).pg(), qq4.composite_pg()));
            EXPECT_EQ((q + q2).multiplicity(), qq4.multi());
            EXPECT_EQ((q + q2).is_fermion(), qq4.fermion());
            if (qq4.twosl <= qq4.twos) {
                EXPECT_EQ((q + q2).count(), qq4.count());
                int kk = Random::rand_int(0, qq4.count());
                EXPECT_EQ((q + q2)[kk].n(), qq4[kk].n);
                EXPECT_EQ((q + q2)[kk].twos(), qq4[kk].twos);
                EXPECT_EQ((q + q2)[kk].twos_low(), qq4[kk].twosl);
                EXPECT_TRUE(
                    S::pg_equal((q + q2)[kk].pg(), qq4[kk].composite_pg()));
            }
        }
        // subtraction
        QULabel<SU2> qq5 = qq3 - qq2;
        if (qq5.in_range()) {
            EXPECT_EQ((q - q2).n(), qq5.n);
            EXPECT_EQ((q - q2).twos_low(), qq5.twosl);
            EXPECT_EQ((q - q2).twos(), qq5.twos);
            EXPECT_TRUE(S::pg_equal((q - q2).pg(), qq5.composite_pg()));
            EXPECT_EQ((q - q2).multiplicity(), qq5.multi());
            EXPECT_EQ((q - q2).is_fermion(), qq5.fermion());
            if (qq5.twosl <= qq5.twos) {
                EXPECT_EQ((q - q2).count(), qq5.count());
                int kk = Random::rand_int(0, qq5.count());
                EXPECT_EQ((q - q2)[kk].n(), qq5[kk].n);
                EXPECT_EQ((q - q2)[kk].twos(), qq5[kk].twos);
                EXPECT_EQ((q - q2)[kk].twos_low(), qq5[kk].twosl);
                EXPECT_TRUE(
                    S::pg_equal((q - q2)[kk].pg(), qq5[kk].composite_pg()));
            }
        }
        // combine
        if (qq5.in_range()) {
            QULabel<SU2> qc = qq5.combine(qq3, qq2);
            EXPECT_EQ((q - q2).combine(q, q2).n(), qc.n);
            // EXPECT_EQ((q - q2).combine(q, q2).twos_low(), qc.twosl);
            EXPECT_EQ((q - q2).combine(q, q2).twos(), qc.twos);
            EXPECT_TRUE(
                S::pg_equal((q - q2).combine(q, q2).pg(), qc.composite_pg()));
            EXPECT_EQ((q - q2).combine(q, q2).get_bra(q - q2).n(), qq3.n);
            EXPECT_EQ((q - q2).combine(q, q2).get_bra(q - q2).twos_low(),
                      qq3.twos);
            EXPECT_EQ((q - q2).combine(q, q2).get_bra(q - q2).twos(), qq3.twos);
            EXPECT_TRUE(
                S::pg_equal((q - q2).combine(q, q2).get_bra(q - q2).pg(),
                            qq3.composite_pg()));
            EXPECT_EQ((q - q2).combine(q, q2).get_ket().n(), qq2.n);
            EXPECT_EQ((q - q2).combine(q, q2).get_ket().twos_low(), qq2.twos);
            EXPECT_EQ((q - q2).combine(q, q2).get_ket().twos(), qq2.twos);
            EXPECT_TRUE(S::pg_equal((q - q2).combine(q, q2).get_ket().pg(),
                                    qq2.composite_pg()));
        }
        if ((-qq5).in_range()) {
            QULabel<SU2> qc = (-qq5).combine(qq2, qq3);
            EXPECT_EQ((-(q - q2)).combine(q2, q).n(), qc.n);
            // EXPECT_EQ((-(q - q2)).combine(q2, q).twos_low(), qc.twosl);
            EXPECT_EQ((-(q - q2)).combine(q2, q).twos(), qc.twos);
            EXPECT_TRUE(S::pg_equal((-(q - q2)).combine(q2, q).pg(),
                                    qc.composite_pg()));
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_bra(q2 - q).n(), qq2.n);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_bra(q2 - q).twos_low(),
                      qq2.twos);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_bra(q2 - q).twos(),
                      qq2.twos);
            EXPECT_TRUE(
                S::pg_equal((-(q - q2)).combine(q2, q).get_bra(q2 - q).pg(),
                            qq2.composite_pg()));
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_ket().n(), qq3.n);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_ket().twos_low(),
                      qq3.twos);
            EXPECT_EQ((-(q - q2)).combine(q2, q).get_ket().twos(), qq3.twos);
            EXPECT_TRUE(S::pg_equal((-(q - q2)).combine(q2, q).get_ket().pg(),
                                    qq3.composite_pg()));
        }
    }
}
