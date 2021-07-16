
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

#include "../core/integral.hpp"
#include "../core/matrix_functions.hpp"
#include "../core/utils.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

template <typename EvalOp> struct GAOptimization {
    EvalOp evop;
    uint16_t n_sites, n_bits, n_bunit;
    vector<uint16_t> ford;
    int n_configs = 50;
    int n_elite = 1;
    double clone_rate = 0.1;
    double mutate_rate = 0.1;
    vector<double> cumu_probs, probs;
    vector<uint16_t> ords;
    GAOptimization(uint16_t n_sites, const vector<uint16_t> &ford, EvalOp &evop,
                   int n_configs)
        : ford(ford), n_sites(n_sites), evop(evop), n_configs(n_configs) {
        probs.resize(n_configs);
        cumu_probs.resize(n_configs);
        ords.resize(2 * n_sites * n_configs);
        for (n_bunit = 0; (1 << n_bunit) < (int)(sizeof(uint16_t) * 8);
             n_bunit++)
            ;
        n_bits = (n_sites >> n_bunit) + !!(n_sites & ((1 << n_bunit) - 1));
    }
    vector<uint16_t> find_best() {
        size_t i = max_element(probs.begin(), probs.end()) - probs.begin();
        vector<uint16_t> r(n_sites);
        memcpy(r.data(), ords.data() + i * n_sites, n_sites * sizeof(uint16_t));
        return r;
    }
    void evaluate(int ir) {
        uint16_t irr = ir * n_sites * n_configs;
        double ssq_prob = 0, sum_prob = 0, min_prob = 1E99;
        for (int i = 0; i < n_configs; i++) {
            probs[i] = sqrt(abs(evop(ords.data() + irr + i * n_sites)));
            sum_prob += probs[i];
            ssq_prob += probs[i] * probs[i];
            min_prob = min(min_prob, probs[i]);
        }
        double mu = sum_prob / n_configs;
        double sigma = ssq_prob / n_configs - mu * mu;
        if (abs(sigma) < 1E-12)
            sigma = 1.0;
        sum_prob = 0;
        for (int i = 0; i < n_configs; sum_prob += probs[i], i++)
            probs[i] =
                exp(-(probs[i] - min_prob) * (probs[i] - min_prob) / sigma);
        cumu_probs[0] = probs[0] / sum_prob;
        for (int i = 1; i < n_configs; i++)
            cumu_probs[i] = cumu_probs[i - 1] + probs[i] / sum_prob;
        assert(abs(cumu_probs[n_configs - 1] - 1.0) < 1E-12);
        cumu_probs[n_configs - 1] = 1.0;
    }
    void initialize(int ir) {
        uint16_t irr = ir * n_sites * n_configs;
        if (ford.size() != 0)
            memcpy(ords.data() + irr, ford.data(), n_sites * sizeof(uint16_t));
        vector<uint16_t> idx(n_sites);
        for (uint16_t i = 0; i < n_sites; i++)
            idx[i] = i;
        for (int i = ford.size() != 0; i < n_configs; i++) {
            for (uint16_t j = 0; j < n_sites; j++)
                swap(idx[j], idx[Random::rand_int(j, n_sites)]);
            memcpy(ords.data() + irr + i * n_sites, idx.data(),
                   n_sites * sizeof(uint16_t));
        }
    }
    void point_mutate(int ic) {
        int itrial = Random::rand_int(1, 4);
        for (int i = 0; i < itrial; i++) {
            int ja = Random::rand_int(0, n_sites),
                jb = Random::rand_int(0, n_sites);
            swap(ords[ic + ja], ords[ic + jb]);
        }
    }
    void global_mutate(int ic) {
        vector<uint16_t> tmp;
        tmp.reserve(n_sites + 4);
        memcpy(tmp.data(), ords.data() + ic, n_sites * sizeof(uint16_t));
        for (int i = 0; i < 4; i++)
            tmp[n_sites + i] = Random::rand_int(0, n_sites);
        const uint16_t *m = tmp.data() + n_sites;
        sort(tmp.data() + n_sites, tmp.data() + (n_sites + 4));
        memcpy(ords.data() + (ic + m[0]), tmp.data() + m[2],
               (m[3] - m[2]) * sizeof(uint16_t));
        memcpy(ords.data() + (ic + m[0] + m[3] - m[2]), tmp.data() + m[1],
               (m[2] - m[1]) * sizeof(uint16_t));
        memcpy(ords.data() + (ic + m[0] + m[3] - m[1]), tmp.data() + m[0],
               (m[1] - m[0]) * sizeof(uint16_t));
    }
    void cross_over(int ia, int ib, int ic) {
        vector<uint16_t> tmp;
        tmp.reserve(n_sites * 2 + n_bits);
        for (uint16_t i = 0; i < n_sites; i++)
            tmp[ords[ia + i]] = i, tmp[ords[ib + i] + n_sites] = i;
        for (uint16_t i = 0; i < n_bits; i++)
            tmp[n_sites + n_sites + i] =
                (uint16_t)Random::rand_int(0, 1 << n_bits);
        const uint16_t *ma = tmp.data(), *mb = tmp.data() + n_sites;
        const uint16_t *mask = tmp.data() + n_sites + n_sites;
        memcpy(ords.data() + ic, ords.data() + ib, n_sites * sizeof(uint16_t));
        const int n_bmask = (1 << n_bunit) - 1;
        for (uint16_t i = 0, j, k; i < n_sites; i++)
            if ((mask[i >> n_bunit] >> (i & n_bmask)) & 1) {
                k = ma[ords[ib + i]];
                if (!((mask[k >> n_bunit] >> (k & n_bmask)) & 1)) {
                    for (j = mb[ords[ia + i]];
                         (mask[j >> n_bunit] >> (j & n_bmask)) & 1;)
                        j = mb[ords[ia + j]];
                    swap(ords[ic + i], ords[ic + j]);
                }
            }
        for (uint16_t i = 0; i < n_sites; i++)
            if ((mask[i >> n_bunit] >> (i & n_bmask)) & 1)
                ords[ic + i] = ords[ia + i];
    }
    void optimize(int ip) {
        int ir = !ip;
        uint16_t irr = ir * n_sites * n_configs;
        uint16_t ipp = ip * n_sites * n_configs;
        vector<uint16_t> idx(n_configs);
        for (int i = 0; i < n_configs; i++)
            idx[i] = i;
        sort(idx.begin(), idx.end(), [this](uint16_t i, uint16_t j) {
            return this->probs[i] > this->probs[j];
        });
        for (int i = 0; i < n_elite; i++)
            memcpy(ords.data() + irr + i * n_sites,
                   ords.data() + ipp + idx[i] * n_sites,
                   n_sites * sizeof(uint16_t));
        for (int i = n_elite; i < n_configs; i++) {
            if (Random::rand_double() < clone_rate) {
                int j = (int)(lower_bound(cumu_probs.begin(), cumu_probs.end(),
                                          Random::rand_double()) -
                              cumu_probs.begin());
                memcpy(ords.data() + irr + i * n_sites,
                       ords.data() + ipp + j * n_sites,
                       n_sites * sizeof(uint16_t));
            } else {
                int ja = (int)(lower_bound(cumu_probs.begin(), cumu_probs.end(),
                                           Random::rand_double()) -
                               cumu_probs.begin());
                int jb = (int)(lower_bound(cumu_probs.begin(), cumu_probs.end(),
                                           Random::rand_double()) -
                               cumu_probs.begin());
                cross_over(ipp + ja * n_sites, ipp + jb * n_sites,
                           irr + i * n_sites);
            }
            if (Random::rand_double() < mutate_rate)
                point_mutate(irr + i * n_sites);
            if (Random::rand_double() < mutate_rate)
                global_mutate(irr + i * n_sites);
        }
        evaluate(ir);
    }
    vector<uint16_t> solve(int n_generations = 10000) {
        initialize(n_generations & 1);
        evaluate(n_generations & 1);
        for (int i = 0, ip = n_generations & 1; i < n_generations;
             i++, ip = !ip)
            optimize(ip);
        return find_best();
    }
};

struct OrbitalOrdering {
    static vector<double> exp_trans(const vector<double> &mat) {
        vector<double> emat(mat.size());
        for (size_t i = 0; i < mat.size(); i++)
            emat[i] = exp(-mat[i]);
        return emat;
    }
    static double evaluate(uint16_t n_sites, const vector<double> &kmat,
                           const vector<uint16_t> &ord) {
        double r = 0, rsum = 0;
        if (ord.size() == 0)
            for (uint16_t i = 0; i < n_sites; i++)
                for (uint16_t j = i + 1; j < n_sites; j++)
                    r += (double)(j - i) * (j - i) * kmat[i * n_sites + j],
                        rsum += kmat[i * n_sites + j];
        else
            for (uint16_t i = 0; i < n_sites; i++)
                for (uint16_t j = i + 1; j < n_sites; j++)
                    r += (double)(j - i) * (j - i) *
                         kmat[ord[i] * n_sites + ord[j]],
                        rsum += kmat[ord[i] * n_sites + ord[j]];
        return r / rsum;
    }
    static vector<uint16_t> ga_opt(uint16_t n_sites, const vector<double> &kmat,
                                   int n_generations = 10000,
                                   int n_configs = 54, int n_elite = 5,
                                   double clone_rate = 0.1,
                                   double mutate_rate = 0.1) {
        double rsum = 0;
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = i + 1; j < n_sites; j++)
                rsum += kmat[i * n_sites + j];
        auto eval_op = [n_sites, rsum, &kmat](uint16_t *ord) {
            double r = 0;
            for (uint16_t i = 0; i < n_sites; i++)
                for (uint16_t j = i + 1, ii = ord[i]; j < n_sites; j++)
                    r += kmat[ii * n_sites + ord[j]] * (j - i) * (j - i);
            return r / rsum;
        };
        vector<uint16_t> ford = fiedler(n_sites, kmat);
        GAOptimization<decltype(eval_op)> ga(n_sites, ford, eval_op, n_configs);
        ga.n_elite = n_elite;
        ga.clone_rate = clone_rate;
        ga.mutate_rate = mutate_rate;
        return ga.solve(n_generations);
    }
    static vector<uint16_t> fiedler(uint16_t n_sites,
                                    const vector<double> &kmat) {
        assert(kmat.size() == n_sites * n_sites);
        vector<double> lmat(n_sites * n_sites);
        for (uint16_t i = 0; i < n_sites; i++)
            for (uint16_t j = 0; j < n_sites; j++) {
                lmat[i * n_sites + i] += abs(kmat[i * n_sites + j]);
                lmat[i * n_sites + j] -= kmat[i * n_sites + j];
            }
        vector<double> wmat(n_sites);
        MatrixFunctions::eigs(MatrixRef(lmat.data(), n_sites, n_sites),
                              DiagonalMatrix(wmat.data(), n_sites));
        double factor = 1.0;
        for (uint16_t i = 0; i < n_sites; i++)
            if (abs(lmat[n_sites + i]) > 1E-12) {
                factor = lmat[n_sites + i] > 0 ? 1.0 : -1.0;
                break;
            }
        for (uint16_t i = 0; i < n_sites; i++)
            lmat[i] = lmat[n_sites + i] * factor;
        vector<uint16_t> ord(n_sites);
        for (uint16_t i = 0; i < n_sites; i++)
            ord[i] = i;
        stable_sort(ord.begin(), ord.end(), [&lmat](uint16_t i, uint16_t j) {
            return lmat[i] < lmat[j];
        });
        return ord;
    }
};

} // namespace block2
