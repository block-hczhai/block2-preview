
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

/** Fast Fourier Transform (FFT) and number theory algorithms. */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

/** Number theory algorithms for prime numbers and primitive root. */
struct Prime {
    typedef long long LL;
    int np; //!< Maximal number considered for generation of set of small
            //!< primes.
    vector<int> primes; //!< Set of small primes.
    /** Default constructor. */
    Prime() : np(0) {}
    /** Initialize set of small primes, using sieve of Eratosthenes.
     * @param np Maximal number considered for finding primes.
     */
    void init_primes(int np = 50000) {
        this->np = np;
        vector<bool> p(np + 1, false);
        primes.clear();
        primes.push_back(2);
        for (int i = 3; i <= np; i += 2)
            if (!p[i]) {
                primes.push_back(i);
                for (int j = i, k = i + i; j <= np; j += k)
                    p[j] = true;
            }
    }
    /** Return positive x mod n. */
    inline static int pmod(LL x, LL n) { return (x % n + n) % n; }
    /** Extended Euclidean algorithm.
     * Find integers x and y such that a x + b y = gcd(a, b).
     * @param a Integer a.
     * @param b Integer b.
     * @param d (output) Greatest Common Divisor gcd(a, b).
     * @param x (output) Integer x.
     * @param y (output) Integer y.
     */
    static void exgcd(LL a, LL b, LL &d, LL &x, LL &y) {
        if (!b)
            d = a, x = 1, y = 0;
        else {
            exgcd(b, a % b, d, y, x);
            y -= x * (a / b);
        }
    }
    /** Euclidean algorithm.
     * Find Greatest Common Divisor gcd(a, b).
     * @param a Integer a.
     * @param b Integer b.
     * @return Greatest Common Divisor gcd(a, b).
     */
    static LL gcd(LL a, LL b) { return b ? gcd(b, a % b) : a; }
    /** Find modular multiplicative inverse of a (mod n).
     * Using extended Euclidean algorithm.
     * @param a Integer a.
     * @param n Modulus n.
     * @return a^{-1} (mod n) if it exists, otherwise -1.
     */
    static LL inv(LL a, LL n) {
        LL d, x, y;
        exgcd(a, n, d, x, y);
        return d == 1 ? pmod(x, n) : -1;
    }
    /** Find n to the power of i using binary algorithm.
     * @param n Integer n.
     * @param i Integer i (i >= 0).
     * @return n^i.
     */
    static LL power(LL n, int i) {
        LL r = 1;
        for (; i; i >>= 1, n = n * n)
            (i & 1) && (r *= n);
        return r;
    }
    /** Find the largest number r such that r * r <= x.
     * Using binary algorithm.
     * @param x Integer x.
     * @return floor(sqrt(x)).
     */
    static LL sqrt(LL x) {
        LL a = x, b = x + 2;
        if (x == 0)
            return 0;
        while (b - a > 1)
            a = ((b = a) + x / a) >> 1;
        return a * a > x ? a - 1 : a;
    }
    /** Find (x * y) mod p.
     * Note that the expression x * y % p may overflow if the intermediate
     * x * y > 2^63 - 1. This function will not overflow.
     * @param x Integer x.
     * @param y Integer y.
     * @param p Modulus p.
     * @return (x * y) mod p.
     */
    static LL quick_multiply(LL x, LL y, LL p) {
        static LL T = 0, pp = 0, t = 0;
        if (pp != p) {
            pp = p, T = Prime::sqrt(p), t = T * T - p;
            abs(t) > T && (T++, t = T * T - p);
        }
        LL a = (x %= p) / T, b = x % T, c = (y %= p) / T, d = y % T;
        LL e = a * c / T, f = a * c % T;
        LL v = ((a * d + b * c) % p + e * t) % p, g = v / T, h = v % T;
        LL r = (((f + g) * t % p + b * d) % p + h * T) % p;
        return r < 0 ? r + p : r;
    }
    /** Find (n ^ i) mod p using binary algorithm.
     * @param n Integer n.
     * @param i Integer i (i >= 0).
     * @param p Modulus p.
     * @return (n ^ i) mod p.
     */
    static LL quick_power(LL n, LL i, LL p) {
        LL r = 1;
        for (n %= p; i; i >>= 1, n = quick_multiply(n, n, p))
            (i & 1) && (r = quick_multiply(r, n, p));
        return r;
    }
    /** Miller-Rabin primality test.
     * @param a Base number a.
     * @param n The number to be tested for prime (n >= 3).
     * @return ``true`` if n is likely to be a prime. ``false`` if n is not
     *   a prime.
     * @note If n < 3215031751, it is enough to test a = 2, 3, 5, and 7;
     *   if n < 341550071728321, it is enough to test a = 2, 3, 5, 7, 11, 13,
     *   and 17.
     */
    static bool miller_rabin(LL a, LL n) {
        LL r = 0, s = n - 1;
        while (!(s & 1))
            s >>= 1, r++;
        LL x = quick_power(a, s, n);
        if (x == 1)
            return true;
        for (int j = 0; j < r; j++, x = quick_multiply(x, x, n))
            if (x == n - 1)
                return true;
        return false;
    }
    /** Pollard's rho algorithm.
     * Find a factor of integer n.
     * @param n Integer n.
     * @return A (not necessarily prime) factor of n.
     */
    static LL pollard_rho(LL n) {
        for (LL c = 1, x, y, d;; c++) {
            for (x = y = 2;;) {
                x = quick_multiply(x, x, n), x = (x + c) % n;
                y = quick_multiply(y, y, n), y = (y + c) % n;
                y = quick_multiply(y, y, n), y = (y + c) % n;
                d = Prime::gcd(abs(y - x), n);
                if (d == n)
                    break;
                else if (d > 1)
                    return d;
            }
        }
    }
    /** Integer factorization.
     * @param x Integer x.
     * @param pp (output) Set of prime factors and their occurrence.
     */
    void factors(LL x, vector<pair<LL, int>> &pp) {
        if (np == 0)
            init_primes();
        bool large = (LL)np * np < x;
        for (auto g : primes)
            if (x % g == 0) {
                int cnt = 0;
                while (x % g == 0)
                    x /= g, cnt++;
                pp.push_back(make_pair(g, cnt));
                if (x == 1)
                    break;
            }
        if (x != 1) {
            if (!large || is_prime(x))
                pp.push_back(make_pair(x, 1));
            else {
                LL d = pollard_rho(x);
                vector<pair<LL, int>> px;
                factors(d, px), factors(x / d, px);
                sort(px.begin(), px.end());
                for (auto p : px) {
                    if (!pp.empty() && pp.back().first == p.first)
                        pp.back().second += p.second;
                    else
                        pp.push_back(p);
                }
            }
        }
    }
    /** Euler's totient function.
     * @param n Integer n (not necessarily prime).
     * @return phi(n), which counts the positive integers up to the given
     *   integer n that are relatively prime to n.
     */
    LL euler(LL n) {
        vector<pair<LL, int>> pp;
        factors(n, pp);
        LL r = 1;
        for (auto &p : pp)
            r *= quick_power(p.first, p.second - 1, n) * (p.first - 1);
        return r;
    }
    /** Prime testing.
     * @param n Integer n to be tested (n < 2^63 - 1).
     * @return ``true`` if n is a prime, ``false`` if n is not a prime.
     */
    bool is_prime(LL n) {
        static LL p4 = 3215031751LL, p7 = 341550071728321LL;
        if (np == 0)
            init_primes();
        for (int i = 0; i < 100; i++)
            if (n == primes[i])
                return true;
            else if (n % primes[i] == 0)
                return false;
        int k = n < p4 ? 4 : (n < p7 ? 7 : 12);
        for (int i = 0; i < k; i++)
            if (!miller_rabin(primes[i], n))
                return false;
        return true;
    }
    /** Find one of primitive roots modulo n.
     * A number g is a primitive root modulo n if every number a coprime to n is
     *   congruent to a power of g modulo n.
     * @param p Modulus p.
     * @return a primitive root g.
     */
    int primitive_root(LL p) {
        vector<pair<LL, int>> pp;
        LL phi = p - 1;
        if (p <= 4)
            return p - 1;
        else if (!is_prime(p)) {
            factors((p & 1) ? p : p / 2, pp);
            if (pp.size() != 1 || pp[0].first == 2)
                return -1;
            pp.clear(), phi = euler(p);
        }
        factors(phi, pp);
        for (int g = 2; g < p; g++) {
            bool ok = gcd(g, p) == 1;
            for (auto &x : pp)
                if (quick_power(g, phi / x.first, p) == 1) {
                    ok = false;
                    break;
                }
            if (ok)
                return g;
        }
        return -1;
    }
    /** Find all primitive roots modulo n.
     * @param p Modulus p.
     * @param gg (output) All primitive roots modulo n.
     */
    void primitive_roots(LL p, vector<LL> &gg) {
        LL g = primitive_root(p);
        for (LL i = 1, q = g; i < p; i++, q = quick_multiply(q, g, p))
            if (gcd(p - 1, i) == 1)
                gg.push_back(q);
    }
};

/** Fast Fourier Transform (FFT) with array length of base^k (k >= 0).
 * The complexity is O(n log_base n * base^2).
 * @tparam base The radix (base = 2 is specialized) */
template <int base> struct BasicFFT {
    vector<size_t> r;           //!< The index permutation array.
    vector<complex<double>> wb, //!< The precomputed primitive nth root of 1
                                //!< exp(i2pi k/n) for backward FFT.
        wf; //!< The precomputed primitive nth root of 1 exp(-i2pi k/n) for
            //!< forward FFT.
    array<array<complex<double>, base>, base>
        xw[2]; //!< The precomputed primitive ``base``-th root of 1
               //!< exp(-/+ i2pi jk/base) for forward/backward FFT.
    /** Constructor. */
    BasicFFT() {
        for (int forth = 0; forth < 2; forth++) {
            const complex<double> ipi(0, forth ? acos(-1) : -acos(-1));
            for (size_t ib = 0; ib < base; ib++)
                for (size_t jb = 1; jb < base; jb++)
                    xw[forth][ib][jb] = exp(ipi * (-2.0 * ib * jb / base));
        }
    }
    /** Find smallest number x such that x = base^k >= n.
     * @param n The array length.
     * @return The padded array length.
     */
    static size_t pad(size_t n) {
        size_t x = 1;
        for (; x < n; x *= base)
            ;
        return x;
    }
    /** Precompute for array length n for both forward and backward FFT.
     * @param n The array length, must be a power of ``base``.
     */
    void init(size_t n) {
        const static double pi = acos(-1);
        if (n <= 1)
            return;
        size_t x = pad(n);
        assert(x == n);
        r.resize(n), wf.resize(n), wb.resize(n);
        r[0] = 0;
        for (size_t i = 1, k = n / base; i < n; i++)
            r[i] = r[i / base] / base + i % base * k;
        wf[0] = wb[0] = 1;
        for (size_t i = 0; i < n; i++)
            wf[i] = complex<double>(cos(2 * pi * i / n), -sin(2 * pi * i / n));
        for (size_t i = 0; i < n; i++)
            wb[i] = conj(wf[i]);
    }
    /** Perform inplace FFT.
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array, must be a power of ``base``.
     * @param forth Whether this is forward transform.
     */
    void fft(complex<double> *arr, size_t n, bool forth) {
        if (n <= 1)
            return;
        if (r.size() != n)
            init(n);
        const complex<double> *w = forth ? wf.data() : wb.data();
        for (size_t i = 1; i < n; i++)
            if (i < r[i])
                swap(arr[i], arr[r[i]]);
        array<complex<double>, base> x;
        const array<array<complex<double>, base>, base> &ww = xw[forth];
        for (size_t i = 1; i < n; i *= base)
            for (size_t ii = i * base, ni = n / ii, j = 0, k; j < n; j += ii)
                for (k = 0; k < i; k++) {
                    for (size_t ib = 0; ib < base; ib++)
                        x[ib] = arr[j + k + ib * i] * w[ib * ni * k];
                    for (size_t ib = 0; ib < base; ib++) {
                        arr[j + k + ib * i] = x[0];
                        for (size_t jb = 1; jb < base; jb++)
                            arr[j + k + ib * i] += x[jb] * ww[ib][jb];
                    }
                }
        if (!forth)
            for (size_t i = 0; i < n; i++)
                arr[i] = arr[i] / (double)n;
    }
};

/** Radix-2 Fast Fourier Transform (FFT) with complexity O(nlog n).*/
template <> struct BasicFFT<2> {
    vector<size_t> r;           //!< The index permutation array.
    vector<complex<double>> wb, //!< The precomputed primitive nth root of 1
                                //!< exp(i2pi k/n) for backward FFT.
        wf;                     //!< The precomputed primitive nth root of 1
                                //!< exp(-i2pi k/n) for forward FFT.
    /** Constructor. */
    BasicFFT() {}
    /** Find smallest number x such that x = 2^k >= n.
     * @param n The array length.
     * @return The padded array length.
     */
    static size_t pad(size_t n) {
        size_t x = 1;
        for (; x < n; x <<= 1)
            ;
        return x;
    }
    /** Precompute for array length n for both forward and backward FFT.
     * @param n The array length must be a power of 2.
     */
    void init(size_t n) {
        const static double pi = acos(-1);
        if (n <= 1)
            return;
        size_t m = 0;
        for (size_t x = n; x >>= 1; m++)
            ;
        assert((1 << m) == n);
        r.resize(n), wf.resize(n), wb.resize(n);
        r[0] = 0;
        for (size_t i = 1, k = m - 1; i < n; i++)
            r[i] = (r[i >> 1] >> 1) | ((i & 1) << k);
        wf[0] = wb[0] = 1;
        for (size_t i = 0; i < n; i++)
            wf[i] = complex<double>(cos(2 * pi * i / n), -sin(2 * pi * i / n));
        for (size_t i = 0; i < n; i++)
            wb[i] = conj(wf[i]);
    }
    /** Perform inplace FFT.
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array, must be a power of 2.
     * @param forth Whether this is forward transform.
     */
    void fft(complex<double> *arr, size_t n, bool forth) {
        if (n <= 1)
            return;
        if (r.size() != n)
            init(n);
        const complex<double> *w = forth ? wf.data() : wb.data();
        for (size_t i = 1; i < n; i++)
            if (i < r[i])
                swap(arr[i], arr[r[i]]);
        for (size_t i = 1; i < n; i <<= 1)
            for (size_t ii = i << 1, ni = n / ii, j = 0, k; j < n; j += ii)
                for (k = 0; k < i; k++) {
                    complex<double> x = arr[j + k],
                                    y = arr[j + k + i] * w[ni * k];
                    arr[j + k] = x + y, arr[j + k + i] = x - y;
                }
        if (!forth)
            for (size_t i = 0; i < n; i++)
                arr[i] = arr[i] / (double)n;
    }
};

/** Rader's FFT algorithm for prime array length.
 * This algorithm transforms a FFT of length n to two (forward and backward)
 * FFTs of length m, where m is the padded array length for 2 * n - 3 for the
 * backend FFT.
 * @tparam B The backend FFT for computing the padded FFT.
 */
template <typename B = BasicFFT<2>> struct RaderFFT {
    typedef typename Prime::LL LL;
    vector<LL> wb, //!< Precomputed inverse of the power of primitive root
                   //!< g^{-k} mod n for k = 0, 1, ..., n - 1.
        wf;        //!< Precomputed power of primitive root g^k mod n
                   //!< for k = 0, 1, ..., n - 1.
    vector<complex<double>> cb, //!< FFT transformed exp(i2pi g^{-k}/n).
        cf,                     //!< FFT transformed exp(-i2pi g^{-k}/n).
        arx;                    //!< Working space for padded array.
    size_t nn;                  //!< Padded array length.
    B b;                        //!< The backend FFT instance.
    shared_ptr<Prime> prime;    //!< Instance for prime number algorithms.
    /** Default constructor. */
    RaderFFT() : prime(make_shared<Prime>()) {}
    /** Constructor.
     * @param prime Instance for prime number algorithms.
     */
    RaderFFT(const shared_ptr<Prime> &prime) : prime(prime) {}
    /** Precompute for array length n for both forward and backward FFT.
     * @param n The array length, which must be a prime number.
     */
    void init(size_t n) {
        const static double pi = acos(-1);
        if (n <= 1)
            return;
        nn = B::pad((n << 1) - 3);
        b.init(nn);
        wf.resize(n), wb.resize(n);
        assert(prime->is_prime((LL)n));
        LL g = prime->primitive_root((LL)n);
        LL gg = Prime::quick_power(g, (LL)n - 2, (LL)n);
        wf[0] = wb[0] = 1;
        for (size_t i = 1; i < n; i++) {
            wf[i] = Prime::quick_multiply(wf[i - 1], g, (LL)n);
            wb[i] = Prime::quick_multiply(wb[i - 1], gg, (LL)n);
        }
        cf.resize(nn), cb.resize(nn);
        for (size_t i = 0; i < n - 1; i++)
            cf[i] = complex<double>(cos(2 * pi * wb[i] / (LL)n),
                                    -sin(2 * pi * wb[i] / (LL)n));
        for (size_t i = 0; i < nn - (n - 1); i++)
            cf[i + n - 1] = cf[i];
        for (size_t i = 0; i < nn; i++)
            cb[i] = conj(cf[i]);
        b.fft(cf.data(), nn, true);
        b.fft(cb.data(), nn, true);
        arx.resize(nn);
    }
    /** Perform inplace FFT.
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array, which must be a prime number.
     * @param forth Whether this is forward transform.
     */
    void fft(complex<double> *arr, size_t n, bool forth) {
        const static double pi = acos(-1);
        if (n <= 1)
            return;
        if (wb.size() != n)
            init(n);
        const complex<double> *c = forth ? cf.data() : cb.data();
        const complex<double> x0 = arr[0];
        arx[0] = arr[wf[0]];
        for (size_t i = 1; i < n - 1; i++)
            arx[i + nn + 1 - n] = arr[wf[i]];
        memset(arx.data() + 1, 0, (nn + 1 - n) * sizeof(complex<double>));
        b.fft(arx.data(), nn, true);
        arr[0] = arx[0] + x0;
        for (size_t i = 0; i < nn; i++)
            arx[i] *= c[i];
        b.fft(arx.data(), nn, false);
        for (size_t i = 0; i < n - 1; i++)
            arr[wb[i]] = arx[i] + x0;
        if (!forth)
            for (size_t i = 0; i < n; i++)
                arr[i] = arr[i] / (double)n;
    }
};

/** Bluestein's FFT algorithm for arbitrary array length.
 * This algorithm transforms a FFT of length n to two (forward and backward)
 * FFTs of length m, where m is the padded array length for 2 * n for the
 * backend FFT.
 * @tparam B The backend FFT for computing the padded FFT.
 */
template <typename B = BasicFFT<2>> struct BluesteinFFT {
    typedef typename Prime::LL LL;
    vector<complex<double>> wb, //!< The precomputed primitive nth root of 1
                                //!< exp(i2pi k/n) for backward FFT.
        wf,                     //!< The precomputed primitive nth root of 1
                                //!< exp(-i2pi k/n) for forward FFT.
        cb,  //!< FFT transformed exp(i2pi [((k - n) - (k - n)(k - n)) / 2]/n).
        cf,  //!< FFT transformed exp(-i2pi [((k - n) - (k - n)(k - n)) / 2]/n).
        arx; //!< Working space for padded array.
    size_t nn; //!< Padded array length.
    B b;       //!< The backend FFT instance.
    /** Default constructor. */
    BluesteinFFT() {}
    /** Constructor.
     * @param prime Instance for prime number algorithms (ignored).
     */
    BluesteinFFT(const shared_ptr<Prime> &prime) {}
    /** Precompute for array length n for both forward and backward FFT.
     * @param n The array length.
     */
    void init(size_t n) {
        const static double pi = acos(-1);
        if (n <= 1)
            return;
        nn = B::pad(n << 1);
        b.init(nn);
        wf.resize(n), wb.resize(n);
        wf[0] = wb[0] = 1;
        for (size_t i = 0; i < n; i++)
            wf[i] = complex<double>(cos(2 * pi * i / n), -sin(2 * pi * i / n));
        for (size_t i = 0; i < n; i++)
            wb[i] = conj(wf[i]);
        cf.resize(nn), cb.resize(nn);
        for (size_t i = 0; i < nn; i++) {
            LL j = (LL)i - (LL)n;
            LL jj = ((j - j * j) / 2 % (LL)n + (LL)n) % (LL)n;
            cf[i] = wf[jj], cb[i] = wb[jj];
        }
        b.fft(cf.data(), nn, true);
        b.fft(cb.data(), nn, true);
        arx.resize(nn);
    }
    /** Perform inplace FFT.
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array.
     * @param forth Whether this is forward transform.
     */
    void fft(complex<double> *arr, size_t n, bool forth) {
        if (n <= 1)
            return;
        if (wb.size() != n)
            init(n);
        const complex<double> *w = forth ? wf.data() : wb.data();
        const complex<double> *c = forth ? cf.data() : cb.data();
        for (size_t i = 0; i < n; i++)
            arx[i] = arr[i] * w[(LL)i * ((LL)i + 1) / 2 % (LL)n];
        memset(arx.data() + n, 0, (nn - n) * sizeof(complex<double>));
        b.fft(arx.data(), nn, true);
        for (size_t i = 0; i < nn; i++)
            arx[i] *= c[i];
        b.fft(arx.data(), nn, false);
        for (size_t i = 0; i < n; i++)
            arr[i] = arx[i + n] * w[(LL)i * ((LL)i - 1) / 2 % (LL)n];
        if (!forth)
            for (size_t i = 0; i < n; i++)
                arr[i] = arr[i] / (double)n;
    }
};

/** Naive Discrete Fourier Transform (DFT) algorithm with complexity O(n^2). */
struct DFT {
    /** Default constructor. */
    DFT() {}
    /** Constructor.
     * @param prime Instance for prime number algorithms (ignored).
     */
    DFT(const shared_ptr<Prime> &prime) {}
    /** Precompute for array length n for both forward and backward FFT.
     * For DFT this method does nothing.
     * @param n The array length.
     */
    void init(size_t n) {}
    /** \rst
        Perform inplace DFT.

        **Forward DFT:**
        :math:`X[k] = \sum_{j=0}^{n - 1} x[j] \exp(-2 \pi \mathrm{i} jk/n)`

        **Backward DFT:**
        :math:`X[k] = \frac{1}{n} \sum_{j=0}^{n - 1} x[j] \exp(2 \pi
        \mathrm{i} jk/n)` \endrst
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array.
     * @param forth Whether this is forward transform.
     */
    void fft(complex<double> *arr, size_t n, bool forth) {
        const complex<double> ipi(0, forth ? acos(-1) : -acos(-1));
        const vector<complex<double>> arx(arr, arr + n);
        memset(arr, 0, n * sizeof(complex<double>));
        for (size_t i = 0; i < n; i++)
            for (size_t k = 0; k < n; k++)
                arr[i] += arx[k] * exp(ipi * (-2.0 * i * k / n));
        if (!forth)
            for (size_t i = 0; i < n; i++)
                arr[i] = arr[i] / (double)n;
    }
};

/** FFT algorithm using different radix FFT backends.
 * The array length is first factorized, then different FFT backends
 * will be used and then the results is merged using the Cooley-Tukey FFT
 * algorithm.
 * @tparam F The prime number FFT backend.
 * @tparam P Using Radix-P FFT backend.
 * @tparam Q Using Radix-(Q1, Q2, ...) FFT backend.
 */
template <typename F, int P, int... Q>
struct FactorizedFFT : FactorizedFFT<F, Q...> {
    /** Default constructor. */
    FactorizedFFT() : FactorizedFFT<F, Q...>(P) {}
    /** Constructor.
     * @param max_factor Maximal radix that should be checked for radix based
     * FFT.
     */
    FactorizedFFT(int max_factor)
        : FactorizedFFT<F, Q...>(max(max_factor, P)) {}
    /** Perform independent FFTs for p arrays, each with length q.
     * @param arr A pointer to the array of complex numbers (as a matrix).
     * @param p Number of rows (FFTs).
     * @param q Number of columns (length of each FFT).
     * @param forth Whether this is forward transform.
     * @param b Radix. Zero if radix based FFT should not be used.
     */
    void fft_internal(complex<double> *arr, size_t p, size_t q, bool forth,
                      int b) override {
        switch (b) {
        case P: {
            BasicFFT<P> fft;
            fft.init(q);
            for (int ip = 0; ip < p; ip++)
                fft.fft(arr + ip * q, q, forth);
        } break;
        default:
            FactorizedFFT<F, Q...>::fft_internal(arr, p, q, forth, b);
            break;
        }
    }
};

/** FFT algorithm using different radix FFT backends.
 * The array length is first factorized, then different FFT backends
 * will be used and then the results is merged using the Cooley-Tukey FFT
 * algorithm.
 * @tparam F The prime number FFT backend.
 * @tparam P Using Radix-P FFT backend.
 */
template <typename F, int P> struct FactorizedFFT<F, P> {
    const int max_factor = P; //!< Maximal radix number.
    shared_ptr<Prime> prime;  //!< Instance for prime number algorithms.
    /** Default constructor. */
    FactorizedFFT() : prime(make_shared<Prime>()) {}
    /** Constructor.
     * @param max_factor Maximal radix that should be checked for radix based
     * FFT.
     */
    FactorizedFFT(int max_factor)
        : max_factor(max(max_factor, P)), prime(make_shared<Prime>()) {}
    /** Precompute for array length n for both forward and backward FFT.
     * For FactorizedFFT this method does nothing.
     * @param n The array length.
     */
    void init(size_t n) {}
    /** Perform independent FFTs for p arrays, each with length q.
     * @param arr A pointer to the array of complex numbers (as a matrix).
     * @param p Number of rows (FFTs).
     * @param q Number of columns (length of each FFT).
     * @param forth Whether this is forward transform.
     * @param b Radix. Zero if radix based FFT should not be used.
     */
    virtual void fft_internal(complex<double> *arr, size_t p, size_t q,
                              bool forth, int b) {
        switch (b) {
        case P: {
            BasicFFT<P> fft;
            fft.init(q);
            for (int ip = 0; ip < p; ip++)
                fft.fft(arr + ip * q, q, forth);
        } break;
        default: {
            F fft = F(make_shared<Prime>());
            fft.init(q);
            for (int ip = 0; ip < p; ip++)
                fft.fft(arr + ip * q, q, forth);
        } break;
        }
    }
    /** Cooley-Tukey FFT algorithm for FFT with array length being a composite
     * number.
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array.
     * @param forth Whether this is forward transform.
     * @param pr A pointer to the array of factors in n;
     * @param b A pointer to the array of radices for each factor.
     *   pr should be multiple of b, if b is not zero.
     * @param np Number of factors.
     */
    void cooley_tukey(complex<double> *arr, size_t n, bool forth,
                      const size_t *pr, const int *b, size_t np) {
        const complex<double> ipi(0, forth ? acos(-1) : -acos(-1));
        size_t p = pr[0];
        assert(n % p == 0);
        const size_t q = n / p;
        if (q == 1)
            return fft_internal(arr, q, p, forth, b[0]);
        vector<complex<double>> arx(arr, arr + n);
        for (int ip = 0; ip < p; ip++)
            for (int iq = 0; iq < q; iq++)
                arr[ip * q + iq] = arx[iq * p + ip];
        if (np == 2)
            fft_internal(arr, p, q, forth, b[1]);
        else
            for (int ip = 0; ip < p; ip++)
                cooley_tukey(arr + ip * q, q, forth, pr + 1, b + 1, np - 1);
        for (size_t ip = 0; ip < p; ip++)
            for (int iq = 0; iq < q; iq++)
                arx[iq * p + ip] =
                    arr[ip * q + iq] * exp(ipi * (-2.0 * ip * iq / n));
        fft_internal(arx.data(), q, p, forth, b[0]);
        for (int ip = 0; ip < p; ip++)
            for (int iq = 0; iq < q; iq++)
                arr[ip * q + iq] = arx[iq * p + ip];
    }
    /** \rst
        Perform inplace FFT.

        **Forward FFT:**
        :math:`X[k] = \sum_{j=0}^{n - 1} x[j] \exp(-2 \pi \mathrm{i} jk/n)`

        **Backward FFT:**
        :math:`X[k] = \frac{1}{n} \sum_{j=0}^{n - 1} x[j] \exp(2 \pi
        \mathrm{i} jk/n)` \endrst
     * @param arr A pointer to the array of complex numbers.
     * @param n Number of elements in the array.
     * @param forth Whether this is forward transform.
     */
    void fft(complex<double> *arr, size_t n, bool forth) {
        if (n <= 1)
            return;
        vector<pair<typename Prime::LL, int>> factors;
        prime->factors((Prime::LL)n, factors);
        vector<size_t> pr;
        vector<int> b;
        pr.reserve(factors.size());
        b.reserve(factors.size());
        for (auto &f : factors) {
            if (f.first <= max_factor)
                pr.push_back((size_t)Prime::power(f.first, f.second)),
                    b.push_back(f.first);
            else
                for (int i = 0; i < f.second; i++)
                    pr.push_back((size_t)f.first), b.push_back(0);
        }
        cooley_tukey(arr, n, forth, pr.data(), b.data(), b.size());
    }
    static void fftshift(complex<double> *arr, size_t n, bool forth) {
        vector<complex<double>> arx(arr, arr + n);
        if (forth) {
            memcpy(arr + n / 2, arx.data(),
                   (n - n / 2) * sizeof(complex<double>));
            memcpy(arr, arx.data() + (n - n / 2),
                   (n / 2) * sizeof(complex<double>));
        } else {
            memcpy(arr + (n - n / 2), arx.data(),
                   (n / 2) * sizeof(complex<double>));
            memcpy(arr, arx.data() + n / 2,
                   (n - n / 2) * sizeof(complex<double>));
        }
    }
    static void fftfreq(double *arr, Prime::LL n, double d) {
        d = 1.0 / (n * d);
        for (Prime::LL i = 0; i < n - n / 2; i++)
            arr[i] = i * d;
        for (Prime::LL i = n - n / 2; i < n; i++)
            arr[i] = (i - n) * d;
    }
};

/** FFT with small prime factorization implemeted using Rader's algorithm. */
typedef FactorizedFFT<RaderFFT<>, 2, 3, 5, 7, 11> FFT2;
/** FFT with small prime factorization implemeted using Bluestein's algorithm.
 */
typedef FactorizedFFT<BluesteinFFT<>, 2, 3, 5, 7, 11> FFT;

} // namespace block2
