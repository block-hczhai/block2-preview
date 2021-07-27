
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
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

/** Number theory algorithms and combinatorics. */

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
     * x * y > 2^63 - 1. This function will not overflow (if p <= 2^62 - 1)
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
        if (x == 1 || x == n - 1)
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

struct Combinatorics {
    typedef long long LL;
    static const LL mod_p =
        4611686018427387847LL; //!< Largest prime number <= 2^62 - 1
    vector<LL> fact, inv_fact;
    const int n_max;
    /** Default constructor. */
    Combinatorics(int n_max) : n_max(n_max) {
        fact.resize(n_max + 1);
        inv_fact.resize(n_max + 1);
        fact[0] = 1;
        for (int i = 1; i <= n_max; i++)
            fact[i] = Prime::quick_multiply(fact[i - 1], i, mod_p);
        inv_fact[n_max] = Prime::quick_power(fact[n_max], mod_p - 2, mod_p);
        for (int i = n_max - 1; i >= 0; i--)
            inv_fact[i] = Prime::quick_multiply(inv_fact[i + 1], i + 1, mod_p);
        assert(inv_fact[0] == fact[0]);
    }
    LL combination(int n, int k) const {
        if (k > n)
            return 0;
        assert(n >= 0 && n <= n_max && k >= 0);
        return Prime::quick_multiply(
            Prime::quick_multiply(fact[n], inv_fact[k], mod_p), inv_fact[n - k],
            mod_p);
    }
};

} // namespace block2
