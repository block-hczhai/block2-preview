
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

#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct CG;

// Trivial CG factors for Abelian symmetry
struct TrivialCG {
    TrivialCG() {}
    virtual ~TrivialCG() = default;
    long double wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                          int tjf) const noexcept {
        return 1.0L;
    }
    long double wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf,
                          int tjg, int tjh, int tji) const noexcept {
        return 1.0L;
    }
    long double racah(int ta, int tb, int tc, int td, int te,
                      int tf) const noexcept {
        return 1.0L;
    }
    long double transpose_cg(int td, int tl, int tr) const noexcept {
        return 1.0L;
    }
    long double phase(int ta, int tb, int tc) const { return 1.0L; }
};

// CG factors for SU(2) symmetry
struct SU2CG {
    shared_ptr<vector<double>> vdata;
    long double *sqrt_fact;
    int n_sf;
    SU2CG(int n_sqrt_fact = 200) : n_sf(n_sqrt_fact) {
        vdata = make_shared<vector<double>>(n_sf * 2);
        sqrt_fact = (long double *)vdata->data();
        sqrt_fact[0] = 1;
        for (int i = 1; i < n_sf; i++)
            sqrt_fact[i] = sqrt_fact[i - 1] * sqrtl(i);
    }
    virtual ~SU2CG() = default;
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    long double sqrt_delta(int tja, int tjb, int tjc) const {
        return sqrt_fact[(tja + tjb - tjc) >> 1] *
               sqrt_fact[(tja - tjb + tjc) >> 1] *
               sqrt_fact[(-tja + tjb + tjc) >> 1] /
               sqrt_fact[(tja + tjb + tjc + 2) >> 1];
    }
    long double cg(int tja, int tjb, int tjc, int tma, int tmb, int tmc) const {
        return (1 - ((tmc + tja - tjb) & 2)) * sqrtl(tjc + 1) *
               wigner_3j(tja, tjb, tjc, tma, tmb, -tmc);
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.21)
    // Adapted from Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_3j(int tja, int tjb, int tjc, int tma, int tmb,
                          int tmc) const {
        if (tma + tmb + tmc != 0 || !triangle(tja, tjb, tjc) ||
            ((tja + tma) & 1) || ((tjb + tmb) & 1) || ((tjc + tmc) & 1))
            return 0;
        const int alpha1 = (tjb - tjc - tma) >> 1,
                  alpha2 = (tja - tjc + tmb) >> 1;
        const int beta1 = (tja + tjb - tjc) >> 1, beta2 = (tja - tma) >> 1,
                  beta3 = (tjb + tmb) >> 1;
        const int max_alpha = max(0, max(alpha1, alpha2));
        const int min_beta = min(beta1, min(beta2, beta3));
        if (max_alpha > min_beta)
            return 0;
        long double factor =
            (1 - ((tja - tjb - tmc) & 2)) * ((max_alpha & 1) ? -1 : 1) *
            sqrt_delta(tja, tjb, tjc) * sqrt_fact[(tja + tma) >> 1] *
            sqrt_fact[(tja - tma) >> 1] * sqrt_fact[(tjb + tmb) >> 1] *
            sqrt_fact[(tjb - tmb) >> 1] * sqrt_fact[(tjc + tmc) >> 1] *
            sqrt_fact[(tjc - tmc) >> 1];
        long double r = 0, rst;
        for (int t = max_alpha; t <= min_beta; ++t, factor = -factor) {
            rst = sqrt_fact[t] * sqrt_fact[t - alpha1] * sqrt_fact[t - alpha2] *
                  sqrt_fact[beta1 - t] * sqrt_fact[beta2 - t] *
                  sqrt_fact[beta3 - t];
            r += factor / (rst * rst);
        }
        return r;
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.36)
    // Adapted from Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                          int tjf) const {
        if (!triangle(tja, tjb, tjc) || !triangle(tja, tje, tjf) ||
            !triangle(tjd, tjb, tjf) || !triangle(tjd, tje, tjc))
            return 0;
        const int alpha1 = (tja + tjb + tjc) >> 1,
                  alpha2 = (tja + tje + tjf) >> 1,
                  alpha3 = (tjd + tjb + tjf) >> 1,
                  alpha4 = (tjd + tje + tjc) >> 1;
        const int beta1 = (tja + tjb + tjd + tje) >> 1,
                  beta2 = (tjb + tjc + tje + tjf) >> 1,
                  beta3 = (tja + tjc + tjd + tjf) >> 1;
        const int max_alpha = max(alpha1, max(alpha2, max(alpha3, alpha4)));
        const int min_beta = min(beta1, min(beta2, beta3));
        if (max_alpha > min_beta)
            return 0;
        long double factor =
            ((max_alpha & 1) ? -1 : 1) * sqrt_delta(tja, tjb, tjc) *
            sqrt_delta(tja, tje, tjf) * sqrt_delta(tjd, tjb, tjf) *
            sqrt_delta(tjd, tje, tjc);
        long double r = 0, rst;
        for (int t = max_alpha; t <= min_beta; ++t, factor = -factor) {
            rst = sqrt_fact[t - alpha1] * sqrt_fact[t - alpha2] *
                  sqrt_fact[t - alpha3] * sqrt_fact[t - alpha4] *
                  sqrt_fact[beta1 - t] * sqrt_fact[beta2 - t] *
                  sqrt_fact[beta3 - t];
            r += factor * sqrt_fact[t + 1] * sqrt_fact[t + 1] / (rst * rst);
        }
        return r;
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.41)
    // Adapted from Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf,
                          int tjg, int tjh, int tji) const {
        if (!triangle(tja, tjb, tjc) || !triangle(tjd, tje, tjf) ||
            !triangle(tjg, tjh, tji) || !triangle(tja, tjd, tjg) ||
            !triangle(tjb, tje, tjh) || !triangle(tjc, tjf, tji))
            return 0;
        const int alpha1 = abs(tja - tji), alpha2 = abs(tjd - tjh),
                  alpha3 = abs(tjb - tjf);
        const int beta1 = tja + tji, beta2 = tjd + tjh, beta3 = tjb + tjf;
        const int max_alpha = max(alpha1, max(alpha2, alpha3));
        const int min_beta = min(beta1, min(beta2, beta3));
        long double r = 0;
        for (int tg = max_alpha; tg <= min_beta; tg += 2) {
            r += (tg + 1) * wigner_6j(tja, tjb, tjc, tjf, tji, tg) *
                 wigner_6j(tjd, tje, tjf, tjb, tg, tjh) *
                 wigner_6j(tjg, tjh, tji, tg, tja, tjd);
        }
        return ((max_alpha & 1) ? -1 : 1) * r;
    }
    // D.M. Brink, G.R. Satchler. Angular Momentum. P142
    long double racah(int ta, int tb, int tc, int td, int te, int tf) const {
        return (1 - ((ta + tb + tc + td) & 2)) *
               wigner_6j(ta, tb, te, td, tc, tf);
    }
    // Transpose factor for an operator with delta quantum number 2S = td
    // and row / column quantum number 2S = tl / tr
    long double transpose_cg(int td, int tl, int tr) const {
        return (1 - ((td + tl - tr) & 2)) * sqrtl(tr + 1) / sqrtl(tl + 1);
    }
    long double phase(int ta, int tb, int tc) const {
        return (1 - ((ta + tb - tc) & 2));
    }
    long double wigner_d(int tj, int tmp, int tms, long double beta) const {
        if (((tj + tmp) & 1) || ((tj + tms) & 1) ||
            !(tmp >= -tj && tmp <= tj) || !(tms >= -tj && tms <= tj))
            return 0;
        const int jpp = (tj + tmp) >> 1, jmp = (tj - tmp) >> 1;
        const int jps = (tj + tms) >> 1, jms = (tj - tms) >> 1;
        const int min_s = max(0, jps - jpp), max_s = min(jps, jmp);
        long double rr = -tanl(beta * 0.5) * tanl(beta * 0.5), rst;
        long double r = 0, factor = (1 - ((tmp - tms + min_s + min_s) & 2));
        factor *= powl(cosl(beta * 0.5), jps + jmp - min_s - min_s);
        factor *= powl(sinl(beta * 0.5), jpp - jps + min_s + min_s);
        for (int s = min_s; s <= max_s; ++s, factor *= rr) {
            rst = sqrt_fact[jps - s] * sqrt_fact[s] * sqrt_fact[jpp - jps + s] *
                  sqrt_fact[jmp - s];
            r += factor / (rst * rst);
        }
        return r * sqrt_fact[jpp] * sqrt_fact[jmp] * sqrt_fact[jps] *
               sqrt_fact[jms];
    }
};

// CG factors for SO(3) symmetry in real spherical harmonics
struct SO3RSHCG : SU2CG {
    SO3RSHCG(int n_sqrt_fact = 200) : SU2CG(n_sqrt_fact) {}
    virtual ~SO3RSHCG() = default;
    // RSH to SH : m1 cpx; m2 xzy
    static complex<long double> u_star(int tm1, int tm2) {
        if (tm1 != tm2 && tm1 != -tm2)
            return 0.0;
        else if (tm1 == tm2)
            return tm1 == 0
                       ? (long double)1.0
                       : (tm1 > 0 ? (long double)(1 - (tm1 & 2)) * sqrtl(0.5)
                                  : complex<long double>(0.0, -sqrtl(0.5)));
        else
            return tm1 > 0 ? complex<long double>(
                                 0.0, (long double)(1 - (tm1 & 2)) * sqrtl(0.5))
                           : sqrtl(0.5);
    }
    complex<long double> cg(int tja, int tjb, int tjc, int tma, int tmb,
                            int tmc) const {
        return (1 - ((tmc + tja - tjb) & 2)) * sqrtl(tjc + 1) *
               wigner_3j(tja, tjb, tjc, tma, tmb, -tmc);
    }
    complex<long double> wigner_3j(int tja, int tjb, int tjc, int tma, int tmb,
                                   int tmc) const {
        complex<long double> r = 0.0;
        for (int8_t i = 0; i < 8; i++)
            if (((i & 1) | tma) && ((i & 2) | tmb) && ((i & 4) | tmc))
                r += SU2CG::wigner_3j(tja, tjb, tjc, i & 1 ? tma : -tma,
                                      i & 2 ? tmb : -tmb, i & 4 ? tmc : -tmc) *
                     u_star(i & 1 ? tma : -tma, tma) *
                     u_star(i & 2 ? tmb : -tmb, tmb) *
                     conj(u_star(i & 4 ? -tmc : tmc, -tmc));
        return r;
    }
};

template <typename FL> struct AnyCG {
    typedef decltype(abs((FL)0.0)) FP;
    AnyCG() {}
    virtual ~AnyCG() = default;
    virtual FL cg(int tja, int tjb, int tjc, int tma, int tmb, int tmc) const {
        assert(false);
        return (FL)0.0;
    }
    virtual FL wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                         int tjf) const {
        return wigner_9j(tja, tjb, tjc, 0, tjd, tjd, tja, tjf, tje);
    }
    virtual FL wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf,
                         int tjg, int tjh, int tji) const {
        int tmi = tji % 2;
        FL r = 0.0;
        for (int tma = -tja; tma <= tja; tma += 2)
            for (int tmb = -tjb; tmb <= tjb; tmb += 2)
                for (int tmd = -tjd; tmd <= tjd; tmd += 2)
                    for (int tme = -tje; tme <= tje; tme += 2) {
                        FL ra = 0.0, rb = 0.0;
                        for (int tmc = -tjc; tmc <= tjc; tmc += 2)
                            for (int tmf = -tjf; tmf <= tjf; tmf += 2)
                                ra += cg(tja, tjb, tjc, tma, tmb, tmc) *
                                      cg(tjd, tje, tjf, tmd, tme, tmf) *
                                      cg(tjc, tjf, tji, tmc, tmf, tmi);
                        for (int tmg = -tjg; tmg <= tjg; tmg += 2)
                            for (int tmh = -tjh; tmh <= tjh; tmh += 2)
                                rb += cg(tja, tjd, tjg, tma, tmd, tmg) *
                                      cg(tjb, tje, tjh, tmb, tme, tmh) *
                                      cg(tjg, tjh, tji, tmg, tmh, tmi);
                        r += ra * rb;
                    }
        return r;
    }
    virtual FL racah(int ta, int tb, int tc, int td, int te, int tf) const {
        int tmc = tc % 2;
        FL r = (FL)0.0;
        for (int tma = -ta; tma <= ta; tma += 2)
            for (int tmb = -tb; tmb <= tb; tmb += 2)
                for (int tmd = -td; tmd <= td; tmd += 2)
                    for (int tme = -te; tme <= te; tme += 2)
                        for (int tmf = -tf; tmf <= tf; tmf += 2)
                            r += cg(ta, tf, tc, tma, tmf, tmc) *
                                 cg(tb, td, tf, tmb, tmd, tmf) *
                                 cg(ta, tb, te, tma, tmb, tme) *
                                 cg(te, td, tc, tme, tmd, tmc);
        return r / (FL)(FP)sqrt((te + 1) * (tf + 1));
    }
    virtual FL phase(int ta, int tb, int tc) const {
        return (FL)(1 - ((ta + tb - tc) & 2));
    }
};

template <typename FL> struct AnySU2CG : AnyCG<FL> {
    typedef decltype(abs((FL)0.0)) FP;
    shared_ptr<SU2CG> su2cg = make_shared<SU2CG>(200);
    AnySU2CG() : AnyCG<FL>() {}
    virtual ~AnySU2CG() = default;
    FL cg(int tja, int tjb, int tjc, int tma, int tmb, int tmc) const override {
        return (FL)(FP)su2cg->cg(tja, tjb, tjc, tma, tmb, tmc);
    }
    FL wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                 int tjf) const override {
        return (FL)(FP)su2cg->wigner_6j(tja, tjb, tjc, tjd, tje, tjf);
    }
    FL wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf, int tjg,
                 int tjh, int tji) const override {
        return (FL)(FP)su2cg->wigner_9j(tja, tjb, tjc, tjd, tje, tjf, tjg, tjh,
                                        tji);
    }
    FL racah(int ta, int tb, int tc, int td, int te, int tf) const override {
        return (FL)(FP)su2cg->racah(ta, tb, tc, td, te, tf);
    }
};

template <typename FL> struct AnySO3RSHCG : AnyCG<FL> {
    typedef decltype(abs((FL)0.0)) FP;
    shared_ptr<SO3RSHCG> so3cg = make_shared<SO3RSHCG>(200);
    AnySO3RSHCG() : AnyCG<FL>() {}
    virtual ~AnySO3RSHCG() = default;
    FL cg(int tja, int tjb, int tjc, int tma, int tmb, int tmc) const override {
        complex<FP> r = (complex<FP>)so3cg->cg(tja, tjb, tjc, tma, tmb, tmc);
        if ((tja + tjb + tjc) % 4 != 0)
            r = (FL)0.0;
        assert(abs(imag(r)) < (FP)1E-10);
        return (FL)real(r);
    }
    FL wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                 int tjf) const override {
        complex<FP> r =
            (complex<FP>)so3cg->wigner_6j(tja, tjb, tjc, tjd, tje, tjf);
        assert(abs(imag(r)) < (FP)1E-10);
        return (FL)real(r);
    }
    FL wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf, int tjg,
                 int tjh, int tji) const override {
        complex<FP> r = (complex<FP>)so3cg->wigner_9j(tja, tjb, tjc, tjd, tje,
                                                      tjf, tjg, tjh, tji);
        assert(abs(imag(r)) < (FP)1E-10);
        return (FL)real(r);
    }
    FL racah(int ta, int tb, int tc, int td, int te, int tf) const override {
        complex<FP> r = (complex<FP>)so3cg->racah(ta, tb, tc, td, te, tf);
        assert(abs(imag(r)) < (FP)1E-10);
        return (FL)real(r);
    }
};

template <typename S> struct CG<S, typename S::is_sany_t> : SU2CG {
    CG(int n_sqrt_fact = 200) : SU2CG() {}
    virtual ~CG() = default;
    long double wigner_6j(S a, S b, S c, S d, S e, S f) const {
        long double r = 1.0L;
        for (int k : a.su2_indices())
            r *= SU2CG::wigner_6j(a.values[k], b.values[k], c.values[k],
                                  d.values[k], e.values[k], f.values[k]);
        return r;
    }
    long double wigner_9j(S a, S b, S c, S d, S e, S f, S g, S h, S i) const {
        long double r = 1.0L;
        for (int k : a.su2_indices())
            r *= SU2CG::wigner_9j(a.values[k], b.values[k], c.values[k],
                                  d.values[k], e.values[k], f.values[k],
                                  g.values[k], h.values[k], i.values[k]);
        return r;
    }
    long double racah(S a, S b, S c, S d, S e, S f) const {
        long double r = 1.0L;
        for (int k : a.su2_indices())
            r *= SU2CG::racah(a.values[k], b.values[k], c.values[k],
                              d.values[k], e.values[k], f.values[k]);
        return r;
    }
    long double transpose_cg(S d, S l, S r) const {
        long double x = 1.0L;
        for (int k : d.su2_indices())
            x *= SU2CG::transpose_cg(d.values[k], l.values[k], r.values[k]);
        return x;
    }
    long double phase(S a, S b, S c) const {
        long double r = 1.0L;
        for (int k : a.su2_indices())
            r *= SU2CG::phase(a.values[k], b.values[k], c.values[k]);
        return r;
    }
    long double cg(S a, S b, S c, S ma, S mb, S mc) const {
        const vector<int> aix = a.su2_indices();
        const vector<int> mix = ma.u1_indices();
        const size_t jx = mix.size() - aix.size();
        long double r = 1.0L;
        for (size_t ix = 0; ix < aix.size(); ix++)
            r *= SU2CG::cg(a.values[aix[ix]], b.values[aix[ix]],
                           c.values[aix[ix]], ma.values[mix[ix + jx]],
                           mb.values[mix[ix + jx]], mc.values[mix[ix + jx]]);
        return r;
    }
};

template <typename S> struct CG<S, typename S::is_sz_t> : TrivialCG {
    CG(int n_sqrt_fact = 200) : TrivialCG() {}
    virtual ~CG() = default;
    long double wigner_6j(S a, S b, S c, S d, S e, S f) const { return 1.0L; }
    long double wigner_9j(S a, S b, S c, S d, S e, S f, S g, S h, S i) const {
        return 1.0L;
    }
    long double racah(S a, S b, S c, S d, S e, S f) const { return 1.0L; }
    long double transpose_cg(S d, S l, S r) const { return 1.0L; }
    long double phase(S a, S b, S c) const { return 1.0L; }
};

template <typename S> struct CG<S, typename S::is_sg_t> : TrivialCG {
    CG(int n_sqrt_fact = 200) : TrivialCG() {}
    virtual ~CG() = default;
    long double wigner_6j(S a, S b, S c, S d, S e, S f) const { return 1.0L; }
    long double wigner_9j(S a, S b, S c, S d, S e, S f, S g, S h, S i) const {
        return 1.0L;
    }
    long double racah(S a, S b, S c, S d, S e, S f) const { return 1.0L; }
    long double transpose_cg(S d, S l, S r) const { return 1.0L; }
    long double phase(S a, S b, S c) const { return 1.0L; }
};

template <typename S> struct CG<S, typename S::is_su2_t> : SU2CG {
    CG(int n_sqrt_fact = 200) : SU2CG(n_sqrt_fact) {}
    virtual ~CG() = default;
    long double wigner_6j(S a, S b, S c, S d, S e, S f) const {
        return SU2CG::wigner_6j(a.twos(), b.twos(), c.twos(), d.twos(),
                                e.twos(), f.twos());
    }
    long double wigner_9j(S a, S b, S c, S d, S e, S f, S g, S h, S i) const {
        return SU2CG::wigner_9j(a.twos(), b.twos(), c.twos(), d.twos(),
                                e.twos(), f.twos(), g.twos(), h.twos(),
                                i.twos());
    }
    long double racah(S a, S b, S c, S d, S e, S f) const {
        return SU2CG::racah(a.twos(), b.twos(), c.twos(), d.twos(), e.twos(),
                            f.twos());
    }
    long double transpose_cg(S d, S l, S r) const {
        return SU2CG::transpose_cg(d.twos(), l.twos(), r.twos());
    }
    long double phase(S a, S b, S c) const {
        return SU2CG::phase(a.twos(), b.twos(), c.twos());
    }
};

} // namespace block2
