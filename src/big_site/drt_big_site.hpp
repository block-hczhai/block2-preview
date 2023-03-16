
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2023 Huanchen Zhai <hczhai@caltech.edu>
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

#include "../core/allocator.hpp"
#include "../core/cg.hpp"
#include "../core/state_info.hpp"
#include "../core/threading.hpp"
#include "../dmrg/general_mpo.hpp"
#include "big_site.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

// Distinct Row Table
template <typename S, ElemOpTypes T> struct DRT {
    typedef long long LL;
    vector<array<int16_t, 3>> abc;
    vector<typename S::pg_t> pgs;
    vector<typename S::pg_t> orb_sym;
    vector<array<int, 4>> jds;
    vector<array<LL, 5>> xs;
    int n_sites, n_init_qs;
    DRT() : n_sites(0), n_init_qs(0) {}
    DRT(int16_t a, int16_t b, int16_t c,
        typename S::pg_t ipg = (typename S::pg_t)0,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>())
        : DRT(a + abs(b) + c, vector<S>{S(a + a + b, b, ipg)}, orb_sym) {}
    DRT(int n_sites, S q,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>())
        : DRT(n_sites, vector<S>{q}, orb_sym) {}
    DRT(int n_sites, const vector<S> &init_qs,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>())
        : n_sites(n_sites), orb_sym(orb_sym), n_init_qs((int)init_qs.size()) {
        if (T == ElemOpTypes::SU2 || T == ElemOpTypes::SZ) {
            for (auto &q : init_qs) {
                abc.push_back(array<int16_t, 3>{
                    (int16_t)((q.n() - q.twos()) >> 1), (int16_t)q.twos(),
                    (int16_t)(n_sites - ((q.n() + q.twos()) >> 1))});
                pgs.push_back(q.pg());
            }
        } else
            assert(false);
        if (this->orb_sym.size() == 0)
            this->orb_sym.resize(n_sites, (typename S::pg_t)0);
        initialize();
    }
    virtual ~DRT() = default;
    int n_rows() const { return (int)abc.size(); }
    void initialize() {
        abc.resize(n_init_qs);
        pgs.resize(n_init_qs);
        auto make_abc = [](int16_t a, int16_t b, int16_t c,
                           int16_t d) -> array<int16_t, 3> {
            switch (d) {
            case 0:
                return array<int16_t, 3>{a, b, (int16_t)(c - 1)};
            case 1:
                return array<int16_t, 3>{(int16_t)(a - (b <= 0)),
                                         (int16_t)(b - 1),
                                         (int16_t)(c - (b <= 0))};
            case 2:
                return array<int16_t, 3>{(int16_t)(a - (b >= 0)),
                                         (int16_t)(b + 1),
                                         (int16_t)(c - (b >= 0))};
            case 3:
                return array<int16_t, 3>{(int16_t)(a - 1), b, c};
            default:
                return array<int16_t, 3>{-1, -1, -1};
            }
        };
        auto allow_abc = [](int16_t a, int16_t b, int16_t c,
                            int16_t d) -> bool {
            switch (d) {
            case 0:
                return c;
            case 1:
                return T == ElemOpTypes::SU2 ? b : (b > 0 || a * c);
            case 2:
                return T == ElemOpTypes::SU2 ? a * c : (b < 0 || a * c);
            case 3:
                return a;
            default:
                return false;
            }
        };
        auto make_pg = [](typename S::pg_t g, typename S::pg_t gk, int16_t d) ->
            typename S::pg_t {
                return (d & 1) ^ (d >> 1) ? S::pg_mul(gk, g) : g;
            };
        auto allow_pg = [](int k, typename S::pg_t g, typename S::pg_t gk,
                           int16_t d) -> bool {
            return k != 0 || ((d & 1) ^ (d >> 1) ? S::pg_mul(gk, g) : g) == 0;
        };
        auto compare_abc_pg =
            [](const pair<array<int16_t, 3>, typename S::pg_t> &p,
               const pair<array<int16_t, 3>, typename S::pg_t> &q) {
                return p.first != q.first ? p.first > q.first
                                          : p.second > q.second;
            };
        vector<vector<pair<array<int16_t, 3>, typename S::pg_t>>> pabc(n_sites +
                                                                       1);
        for (size_t i = 0; i < abc.size(); i++)
            pabc[0].push_back(make_pair(abc[i], pgs[i]));
        // construct graph
        for (int k = n_sites - 1, j = 0; k >= 0; k--, j++) {
            vector<pair<array<int16_t, 3>, typename S::pg_t>> &kabc =
                pabc[j + 1];
            for (const auto &abcg : pabc[j]) {
                const array<int16_t, 3> &x = abcg.first;
                const typename S::pg_t &g = abcg.second;
                for (int16_t d = 0; d < 4; d++)
                    if (allow_abc(x[0], x[1], x[2], d) &&
                        allow_pg(k, g, orb_sym[k], d))
                        kabc.push_back(make_pair(make_abc(x[0], x[1], x[2], d),
                                                 make_pg(g, orb_sym[k], d)));
            }
            sort(kabc.begin(), kabc.end(), compare_abc_pg);
            kabc.resize(
                distance(kabc.begin(), unique(kabc.begin(), kabc.end())));
        }
        int n_abc = 1;
        // filter graph
        for (int k = n_sites - 1, j, i; k >= 0; k--, n_abc += j) {
            vector<pair<array<int16_t, 3>, typename S::pg_t>> &kabc = pabc[k];
            const vector<pair<array<int16_t, 3>, typename S::pg_t>> &fabc =
                pabc[k + 1];
            for (i = 0, j = 0; i < kabc.size(); i++) {
                const array<int16_t, 3> &x = kabc[i].first;
                const typename S::pg_t &g = kabc[i].second;
                bool found = false;
                for (int16_t d = 0; d < 4 && !found; d++)
                    found =
                        found ||
                        binary_search(
                            fabc.begin(), fabc.end(),
                            make_pair(make_abc(x[0], x[1], x[2], d),
                                      make_pg(g, orb_sym[n_sites - 1 - k], d)),
                            compare_abc_pg);
                if (found)
                    kabc[j++] = kabc[i];
            }
            kabc.resize(j);
        }
        // construct abc
        abc.clear(), pgs.clear();
        abc.reserve(n_abc), pgs.reserve(n_abc);
        for (auto &kabc : pabc)
            for (auto &abcg : kabc)
                abc.push_back(abcg.first), pgs.push_back(abcg.second);
        // construct jds
        jds.clear();
        jds.reserve(n_abc);
        for (int k = n_sites - 1, j = 0, p = 0; k >= 0; k--, j++) {
            p += pabc[j].size();
            for (auto &abcg : pabc[j]) {
                array<int, 4> jd;
                for (int16_t d = 0; d < 4; d++) {
                    auto v = make_pair(make_abc(abcg.first[0], abcg.first[1],
                                                abcg.first[2], d),
                                       make_pg(abcg.second, orb_sym[k], d));
                    auto it = lower_bound(pabc[j + 1].begin(),
                                          pabc[j + 1].end(), v, compare_abc_pg);
                    jd[d] = it != pabc[j + 1].end() && *it == v
                                ? p + (int)(it - pabc[j + 1].begin())
                                : 0;
                }
                jds.push_back(jd);
            }
        }
        jds.push_back(array<int, 4>{0, 0, 0, 0});
        // construct xs
        xs.clear();
        xs.resize(max(1, n_abc), array<LL, 5>{0, 0, 0, 0, 0});
        xs.back() = array<LL, 5>{0, 0, 0, 0, 1};
        for (int j = n_abc - 2; j >= 0; j--)
            for (int16_t d = 0; d < 4; d++)
                xs[j][d + 1] = xs[j][d] + xs[jds[j][d]][4] * (jds[j][d] != 0);
    }
    string operator[](LL i) const {
        string r(n_sites, ' ');
        for (int j = 0, k = n_sites - 1; k >= 0; k--) {
            uint8_t d = (uint8_t)(upper_bound(xs[j].begin(), xs[j].end(), i) -
                                  1 - xs[j].begin());
            i -= xs[j][d], j = jds[j][d], r[k] = "0+-2"[d];
        }
        return r;
    }
    LL index(const string &x) const {
        LL i = 0;
        for (int j = 0, k = n_sites - 1; k >= 0; k--) {
            uint8_t d = (uint8_t)string("0+-2").find(x[k]);
            i += xs[j][d], j = jds[j][d];
        }
        return i;
    }
    LL size() const { return xs[0].back(); }
    shared_ptr<StateInfo<S>> get_basis() const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(n_init_qs);
        for (int i = 0; i < n_init_qs; i++) {
            b->quanta[i] =
                S(abc[i][0] + abc[i][0] + abc[i][1], abc[i][1], pgs[i]);
            b->n_states[i] = xs[i][4];
        }
        b->sort_states();
        return b;
    }
    string to_str() const {
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "A" << setw(4)
           << "B" << setw(4) << "C" << setw(6) << "PG" << setw(6) << "JD0"
           << setw(6) << "JD1" << setw(6) << "JD2" << setw(6) << "JD3"
           << " " << setw(12) << "X0"
           << " " << setw(12) << "X1"
           << " " << setw(12) << "X2"
           << " " << setw(12) << "X3" << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            k = abc[i][0] + abc[i][1] + abc[i][2];
            ss << setw(4) << (i + 1);
            if (k == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << k;
            ss << setw(4) << abc[i][0] << setw(4) << abc[i][1] << setw(4)
               << abc[i][2];
            ss << setw(6) << (int)pgs[i];
            for (int16_t dk = 0; dk < 4; dk++)
                if (jds[i][dk] == 0)
                    ss << setw(6) << "";
                else
                    ss << setw(6) << jds[i][dk] + 1;
            for (int16_t dk = 0; dk < 4; dk++)
                ss << " " << setw(12) << xs[i][dk + 1];
            ss << endl;
        }
        return ss.str();
    }
};

// Hamiltonian Distinct Row Table
template <typename S, ElemOpTypes T> struct HDRT {
    typedef long long LL;
    vector<array<int16_t, 5>> qs;
    vector<typename S::pg_t> pgs;
    vector<typename S::pg_t> orb_sym;
    vector<int> jds;
    vector<LL> xs;
    int n_sites, n_init_qs, nd;
    map<pair<string, int8_t>, int> d_map;
    vector<array<int16_t, 6>> d_step;
    vector<pair<string, int8_t>> d_expr;
    HDRT() : n_sites(0), n_init_qs(0), nd(1) {}
    HDRT(int n_sites, const vector<pair<S, pair<int16_t, int16_t>>> &init_qs,
         const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>())
        : n_sites(n_sites), n_init_qs((int)init_qs.size()), orb_sym(orb_sym) {
        for (auto &q : init_qs) {
            qs.push_back(array<int16_t, 5>{
                (int16_t)n_sites, (int16_t)q.first.n(), (int16_t)q.first.twos(),
                q.second.first, q.second.second});
            pgs.push_back(q.first.pg());
        }
        if (this->orb_sym.size() == 0)
            this->orb_sym.resize(n_sites, (typename S::pg_t)0);
    }
    int n_rows() const { return (int)qs.size(); }
    void initialize_steps(const vector<shared_ptr<SpinPermScheme>> &schemes) {
        d_map.clear(), d_step.clear();
        d_map[make_pair("", 0)] = 0;
        d_step.push_back(array<int16_t, 6>{1, 0, 0, 0, 0, 0});
        for (const auto &scheme : schemes) {
            for (int i = 0; i < (int)scheme->data.size(); i++) {
                set<string> exprs;
                for (const auto &m : scheme->data[i])
                    for (const auto &p : m.second)
                        exprs.insert(p.second);
                const vector<uint16_t> &pat = scheme->index_patterns[i];
                for (int k = 0, n = (int)pat.size(), l; k < n; k = l) {
                    for (l = k; l < n && pat[k] == pat[l];)
                        l++;
                    for (const auto &expr : exprs) {
                        string x = SpinPermRecoupling::get_sub_expr(expr, k, l);
                        int8_t dq =
                            SpinPermRecoupling::get_target_twos(
                                SpinPermRecoupling::get_sub_expr(expr, 0, l)) -
                            SpinPermRecoupling::get_target_twos(
                                SpinPermRecoupling::get_sub_expr(expr, 0, k));
                        if (!d_map.count(make_pair(x, dq))) {
                            int16_t xc =
                                (int16_t)count(x.begin(), x.end(), 'C');
                            int16_t xd =
                                (int16_t)count(x.begin(), x.end(), 'D');
                            d_map[make_pair(x, dq)] = (int)d_step.size();
                            d_step.push_back(array<int16_t, 6>{
                                1, (int16_t)(xc - xd), dq, 1,
                                (int16_t)(xc + xd),
                                (int16_t)(xc == xd
                                              ? 0
                                              : (xc > xd ? ((xc - xd) & 1)
                                                         : -((xd - xc) & 1)))});
                        }
                    }
                }
            }
        }
        nd = (int)d_map.size();
        d_expr.resize(nd);
        for (auto &dm : d_map)
            d_expr[dm.second] = dm.first;
    }
    void initialize() {
        qs.resize(n_init_qs);
        pgs.resize(n_init_qs);
        auto make_q = [](const array<int16_t, 5> &q,
                         const array<int16_t, 6> &d) -> array<int16_t, 5> {
            return array<int16_t, 5>{
                (int16_t)(q[0] - d[0]), (int16_t)(q[1] - d[1]),
                (int16_t)(q[2] - d[2]), (int16_t)(q[3] - d[3]),
                (int16_t)(q[4] - d[4])};
        };
        auto allow_q = [](const array<int16_t, 5> &q) -> bool {
            return (q[0] > 0 && (T != ElemOpTypes::SU2 || q[2] >= 0) &&
                    q[3] >= 0 && q[4] >= 0) ||
                   (q[0] == 0 && q[1] == 0 && q[2] == 0 && q[3] == 0 &&
                    q[4] == 0);
        };
        auto make_pg = [](typename S::pg_t g, typename S::pg_t gk,
                          const array<int16_t, 6> &d) ->
            typename S::pg_t { return d[5] != 0 ? S::pg_mul(gk, g) : g; };
        auto allow_pg = [](int k, typename S::pg_t g) -> bool {
            return k != 0 || g == 0;
        };
        auto compare_q_pg =
            [](const pair<array<int16_t, 5>, typename S::pg_t> &p,
               const pair<array<int16_t, 5>, typename S::pg_t> &q) {
                return p.first != q.first ? p.first > q.first
                                          : p.second > q.second;
            };
        vector<vector<pair<array<int16_t, 5>, typename S::pg_t>>> pqs(n_sites +
                                                                      1);
        for (size_t i = 0; i < qs.size(); i++)
            pqs[0].push_back(make_pair(qs[i], pgs[i]));
        // construct graph
        for (int k = n_sites - 1, j = 0; k >= 0; k--, j++) {
            vector<pair<array<int16_t, 5>, typename S::pg_t>> &kq = pqs[j + 1];
            for (const auto &qg : pqs[j]) {
                for (int16_t d = 0; d < nd; d++) {
                    const auto &nq = make_q(qg.first, d_step[d]);
                    const auto &ng = make_pg(qg.second, orb_sym[k], d_step[d]);
                    if (allow_q(nq) && allow_pg(k, ng))
                        kq.push_back(make_pair(nq, ng));
                }
            }
            sort(kq.begin(), kq.end(), compare_q_pg);
            kq.resize(distance(kq.begin(), unique(kq.begin(), kq.end())));
        }
        int n_qs = 1;
        // filter graph
        for (int k = n_sites - 1, j, i; k >= 0; k--, n_qs += j) {
            vector<pair<array<int16_t, 5>, typename S::pg_t>> &kq = pqs[k];
            const vector<pair<array<int16_t, 5>, typename S::pg_t>> &fq =
                pqs[k + 1];
            for (i = 0, j = 0; i < kq.size(); i++) {
                bool found = false;
                for (int16_t d = 0; d < nd && !found; d++) {
                    const auto &nq = make_q(kq[i].first, d_step[d]);
                    const auto &ng = make_pg(
                        kq[i].second, orb_sym[n_sites - 1 - k], d_step[d]);
                    found =
                        found || binary_search(fq.begin(), fq.end(),
                                               make_pair(nq, ng), compare_q_pg);
                }
                if (found)
                    kq[j++] = kq[i];
            }
            kq.resize(j);
        }
        // construct qs
        qs.clear(), pgs.clear();
        qs.reserve(n_qs), pgs.reserve(n_qs);
        for (auto &kq : pqs)
            for (auto &qg : kq)
                qs.push_back(qg.first), pgs.push_back(qg.second);
        // construct jds
        jds.clear();
        jds.reserve(n_qs * nd);
        for (int k = n_sites - 1, j = 0, p = 0; k >= 0; k--, j++) {
            p += pqs[j].size();
            for (auto &qg : pqs[j]) {
                for (int16_t d = 0; d < nd; d++) {
                    const auto &nqg =
                        make_pair(make_q(qg.first, d_step[d]),
                                  make_pg(qg.second, orb_sym[k], d_step[d]));
                    auto it = lower_bound(pqs[j + 1].begin(), pqs[j + 1].end(),
                                          nqg, compare_q_pg);
                    int jd = it != pqs[j + 1].end() && *it == nqg
                                 ? p + (int)(it - pqs[j + 1].begin())
                                 : 0;
                    jds.push_back(jd);
                }
            }
        }
        for (int16_t d = 0; d < nd; d++)
            jds.push_back(0);
        // construct xs
        xs.clear();
        xs.resize(max(1, n_qs * (nd + 1)), 0);
        for (int16_t d = 0; d < nd; d++)
            xs[(n_qs - 1) * (nd + 1) + d] = 0;
        xs[(n_qs - 1) * (nd + 1) + nd] = 1;
        for (int j = n_qs - 2; j >= 0; j--)
            for (int16_t d = 0; d < nd; d++)
                xs[j * (nd + 1) + d + 1] =
                    xs[j * (nd + 1) + d] + xs[jds[j * nd + d] * (nd + 1) + nd] *
                                               (jds[j * nd + d] != 0);
    }
    pair<string, vector<uint16_t>> operator[](LL i) const {
        string r = "";
        int rq = 0;
        vector<uint16_t> kidx;
        int j = 0;
        for (; i >= xs[j * (nd + 1) + nd]; j++)
            i -= xs[j * (nd + 1) + nd];
        for (int k = n_sites - 1; k >= 0; k--) {
            int16_t d =
                (int16_t)(upper_bound(xs.begin() + j * (nd + 1),
                                      xs.begin() + (j + 1) * (nd + 1), i) -
                          1 - (xs.begin() + j * (nd + 1)));
            i -= xs[j * (nd + 1) + d], j = jds[j * nd + d];
            pair<string, int8_t> dx = d_expr[d];
            if (dx.first != "") {
                for (size_t l = 0; l < d_step[d][4]; l++)
                    kidx.insert(kidx.begin(), (uint16_t)k);
                if (r == "")
                    r = dx.first, rq = d_step[d][1];
                else {
                    rq += d_step[d][1];
                    stringstream ss;
                    ss << "(" << dx.first << "+" << r << ")" << rq;
                    r = ss.str();
                }
            }
        }
        return make_pair(r, kidx);
    }
    LL index(const string &expr, const vector<uint16_t> &idxs) const {
        vector<int16_t> ds(n_sites, d_map.at(make_pair("", 0)));
        for (int k = 0, n = (int)idxs.size(), l; k < n; k = l) {
            for (l = k; l < n && idxs[k] == idxs[l];)
                l++;
            string x = SpinPermRecoupling::get_sub_expr(expr, k, l);
            int8_t dq = SpinPermRecoupling::get_target_twos(
                            SpinPermRecoupling::get_sub_expr(expr, 0, l)) -
                        SpinPermRecoupling::get_target_twos(
                            SpinPermRecoupling::get_sub_expr(expr, 0, k));
            if (!d_map.count(make_pair(x, dq)))
                throw runtime_error("expr not found : " + x + " dq = " +
                                    string(1, '0' + dq) + " expr = " + expr);
            ds[idxs[k]] = d_map.at(make_pair(x, dq));
        }
        array<int16_t, 5> iq = qs.back();
        typename S::pg_t ipg = pgs.back();
        for (int k = 0; k < n_sites; k++) {
            iq = array<int16_t, 5>{(int16_t)(iq[0] + d_step[ds[k]][0]),
                                   (int16_t)(iq[1] + d_step[ds[k]][1]),
                                   (int16_t)(iq[2] + d_step[ds[k]][2]),
                                   (int16_t)(iq[3] + d_step[ds[k]][3]),
                                   (int16_t)(iq[4] + d_step[ds[k]][4])};
            ipg = d_step[ds[k]][5] != 0 ? S::pg_mul(ipg, orb_sym[k]) : ipg;
        }
        LL i = 0;
        int j = 0;
        for (; j < n_init_qs && (iq != qs[j] || ipg != pgs[j]); j++)
            i += xs[j * (nd + 1) + nd];
        assert(j < n_init_qs);
        for (int k = n_sites - 1; k >= 0; k--)
            i += xs[j * (nd + 1) + ds[k]], j = jds[j * nd + ds[k]];
        return i;
    }
    LL size() const {
        LL r = 0;
        for (int i = 0; i < n_init_qs; i++)
            r += xs[i * (nd + 1) + nd];
        return r;
    }
    template <typename FL>
    shared_ptr<vector<FL>> fill_data(const vector<string> &exprs,
                                     const vector<vector<uint16_t>> &indices,
                                     const vector<vector<FL>> &data) const {
        shared_ptr<vector<FL>> r = make_shared<vector<FL>>(size(), (FL)0.0);
        for (size_t ix = 0; ix < exprs.size(); ix++) {
            const string &expr = exprs[ix];
            const int nn = SpinPermRecoupling::count_cds(expr);
            for (size_t j = 0; j < data[ix].size(); j++)
                (*r)[index(expr, vector<uint16_t>(indices[ix].begin() + j * nn,
                                                  indices[ix].begin() +
                                                      (j + 1) * nn))] +=
                    data[ix][j];
        }
        return r;
    }
    string to_str() const {
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "N" << setw(4)
           << "2S" << setw(4) << "W" << setw(4) << "L" << setw(6) << "PG";
        for (int16_t dk = 0; dk < nd; dk++)
            ss << setw(5) << "JD" << (int)dk;
        for (int16_t dk = 0; dk < nd; dk++)
            ss << setw(5) << "X" << (int)dk;
        ss << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            ss << setw(4) << (i + 1);
            if (qs[i][0] == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << qs[i][0];
            ss << setw(4) << qs[i][1] << setw(4) << qs[i][2] << setw(4)
               << qs[i][3] << setw(4) << qs[i][4];
            ss << setw(6) << (int)pgs[i];
            for (int16_t dk = 0; dk < nd; dk++)
                if (jds[i * nd + dk] == 0)
                    ss << setw(6) << "";
                else
                    ss << setw(6) << jds[i * nd + dk] + 1;
            for (int16_t dk = 0; dk < nd; dk++)
                ss << setw(6) << xs[i * (nd + 1) + dk + 1];
            ss << endl;
        }
        return ss.str();
    }
};

template <typename FL> struct SU2Matrix {
    vector<FL> data;
    vector<pair<int16_t, int16_t>> indices;
    int16_t dq;
    SU2Matrix(int16_t dq, const vector<FL> &data,
              const vector<pair<int16_t, int16_t>> &indices)
        : dq(dq), indices(indices), data(data) {}
    static SU2CG &cg() {
        static SU2CG _cg;
        return _cg;
    }
    static const vector<SU2Matrix<FL>> &op_matrices() {
        static vector<SU2Matrix<FL>> _mats = vector<SU2Matrix<FL>>{
            SU2Matrix<FL>(0, vector<FL>{(FL)1.0, (FL)1.0, (FL)1.0},
                          vector<pair<int16_t, int16_t>>{make_pair(0, 0),
                                                         make_pair(1, 1),
                                                         make_pair(2, 2)}),
            SU2Matrix<FL>(1, vector<FL>{(FL)1.0, (FL)(-sqrtl(2))},
                          vector<pair<int16_t, int16_t>>{make_pair(1, 0),
                                                         make_pair(2, 1)}),
            SU2Matrix<FL>(1, vector<FL>{(FL)sqrtl(2), (FL)1.0},
                          vector<pair<int16_t, int16_t>>{make_pair(0, 1),
                                                         make_pair(1, 2)})};
        return _mats;
    }
    static SU2Matrix<FL> multiply(const SU2Matrix<FL> &a,
                                  const SU2Matrix<FL> &b, int16_t dq) {
        map<pair<int16_t, int16_t>, FL> r;
        for (int i = 0; i < (int)a.data.size(); i++)
            for (int j = 0; j < (int)b.data.size(); j++)
                if (a.indices[i].second == b.indices[j].first)
                    r[make_pair(a.indices[i].first, b.indices[j].second)] +=
                        a.data[i] * b.data[j] *
                        (FL)cg().racah(b.indices[j].second & 1, b.dq,
                                       a.indices[i].first & 1, a.dq,
                                       a.indices[i].second & 1, dq) *
                        (FL)sqrtl((dq + 1) * ((a.indices[i].second & 1) + 1)) *
                        (FL)cg().phase(a.dq, b.dq, dq);
        vector<FL> data;
        vector<pair<int16_t, int16_t>> indices;
        for (auto &x : r)
            if (x.second != (FL)0.0)
                indices.push_back(x.first), data.push_back(x.second);
        return SU2Matrix<FL>(dq, data, indices);
    }
    static SU2Matrix<FL> build_matrix(const string &expr) {
        if (expr == "")
            return op_matrices()[0];
        else if (expr == "C")
            return op_matrices()[1];
        else if (expr == "D")
            return op_matrices()[2];
        int ix = 0, depth = 0;
        for (auto &c : expr) {
            if (c == '(')
                depth++;
            else if (c == ')')
                depth--;
            else if (c == '+' && depth == 1)
                break;
            ix++;
        }
        int dq = 0, iy = 0;
        for (int i = (int)expr.length() - 1, k = 1; i >= 0; i--, k *= 10)
            if (expr[i] >= '0' && expr[i] <= '9')
                dq += (expr[i] - '0') * k;
            else {
                iy = i;
                break;
            }
        SU2Matrix<FL> a = build_matrix(expr.substr(1, ix - 1));
        SU2Matrix<FL> b = build_matrix(expr.substr(ix + 1, iy - ix - 1));
        return multiply(a, b, dq);
    }
    SU2Matrix<FL> expand() const {
        vector<FL> rd;
        vector<pair<int16_t, int16_t>> ri;
        for (int i = 0; i < (int)data.size(); i++) {
            int16_t p = indices[i].first, q = indices[i].second;
            p += (p >> 1), q += (q >> 1);
            if (p == 1 && q == 1) {
                for (int k = 1; k <= 2; k++)
                    for (int l = 1; l <= 2; l++)
                        ri.push_back(make_pair(k, l)), rd.push_back(data[i]);
            } else if (p == 1) {
                for (int k = 1; k <= 2; k++)
                    ri.push_back(make_pair(k, q)), rd.push_back(data[i]);
            } else if (q == 1) {
                for (int k = 1; k <= 2; k++)
                    ri.push_back(make_pair(p, k)), rd.push_back(data[i]);
            } else
                ri.push_back(make_pair(p, q)), rd.push_back(data[i]);
        }
        return SU2Matrix<FL>(dq, rd, ri);
    }
};

template <typename, typename, typename = void> struct DRTBigSite;

template <typename S, typename FL>
struct DRTBigSite<S, FL, typename S::is_su2_t> : BigSite<S, FL> {
    typedef typename GMatrix<FL>::FP FP;
    typedef long long LL;
    using BigSite<S, FL>::n_orbs;
    using BigSite<S, FL>::basis;
    using BigSite<S, FL>::op_infos;
    shared_ptr<GeneralFCIDUMP<FL>> fcidump;
    shared_ptr<DRT<S, ElemOpTypes::SU2>> drt;
    shared_ptr<HDRT<S, ElemOpTypes::SU2>> hdrt;
    shared_ptr<vector<FL>> ints;
    vector<vector<SU2Matrix<FL>>> site_matrices;
    shared_ptr<vector<FL>> factors;
    array<size_t, 7> factor_strides;
    bool is_right;
    int iprint;
    const static int max_n = 10, max_s = 10;
    DRTBigSite(S q, bool is_right, int n_orbs,
               const vector<typename S::pg_t> &orb_sym,
               const shared_ptr<GeneralFCIDUMP<FL>> &fcidump, int iprint = 0)
        : BigSite<S, FL>(n_orbs), is_right(is_right), fcidump(fcidump),
          iprint(iprint) {
        drt = make_shared<DRT<S, ElemOpTypes::SU2>>(n_orbs, q, orb_sym);
        hdrt = make_shared<HDRT<S, ElemOpTypes::SU2>>(
            n_orbs,
            vector<pair<S, pair<int16_t, int16_t>>>{
                make_pair(S(0, 0, 0), make_pair(2, 2)),
                make_pair(S(0, 0, 0), make_pair(1, 2)),
                make_pair(S(0, 0, 0), make_pair(4, 4)),
                make_pair(S(0, 0, 0), make_pair(3, 4)),
                make_pair(S(0, 0, 0), make_pair(2, 4)),
                make_pair(S(0, 0, 0), make_pair(1, 4))},
            orb_sym);
        vector<shared_ptr<SpinPermScheme>> schemes;
        schemes.reserve(fcidump->exprs.size());
        for (size_t ix = 0; ix < fcidump->exprs.size(); ix++)
            schemes.push_back(
                make_shared<SpinPermScheme>(SpinPermScheme::initialize_su2(
                    SpinPermRecoupling::count_cds(fcidump->exprs[ix]),
                    fcidump->exprs[ix], false, true)));
        hdrt->initialize_steps(schemes);
        hdrt->initialize();
        ints = hdrt->fill_data(fcidump->exprs, fcidump->indices, fcidump->data);
        site_matrices.resize(drt->n_sites);
        for (int i = 0; i < drt->n_sites; i++) {
            for (int d = 0; d < hdrt->nd; d++)
                site_matrices[i].push_back(
                    SU2Matrix<FL>::build_matrix(hdrt->d_expr[d].first)
                        .expand());
        }
        basis = drt->get_basis();
        op_infos = get_site_op_infos(orb_sym);
        prepare_factors();
    }
    virtual ~DRTBigSite() = default;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
    get_site_op_infos(const vector<uint8_t> &orb_sym) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        map<S, shared_ptr<SparseMatrixInfo<S>>> info;
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        info[S(0)] = nullptr;
        for (auto ipg : orb_sym) {
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = 1; s <= max_s_odd; s += 2) {
                    info[S(n, s, ipg)] = nullptr;
                    info[S(n, s, S::pg_inv(ipg))] = nullptr;
                }
            for (auto jpg : orb_sym)
                for (int n = -max_n_even; n <= max_n_even; n += 2)
                    for (int s = 0; s <= max_s_even; s += 2) {
                        info[S(n, s, S::pg_mul(ipg, jpg))] = nullptr;
                        info[S(n, s, S::pg_mul(ipg, S::pg_inv(jpg)))] = nullptr;
                        info[S(n, s, S::pg_mul(S::pg_inv(ipg), jpg))] = nullptr;
                        info[S(n, s,
                               S::pg_mul(S::pg_inv(ipg), S::pg_inv(jpg)))] =
                            nullptr;
                    }
        }
        for (auto &p : info) {
            p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
            p.second->initialize(*basis, *basis, p.first, p.first.is_fermion());
        }
        return vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                info.end());
    }
    void prepare_factors() {
        int16_t max_bb = 0, max_bk = 0, max_bh = 0, max_dh = 0;
        for (auto &p : drt->abc)
            max_bb = max(max_bb, p[1]);
        for (auto &p : drt->abc)
            max_bk = max(max_bk, p[1]);
        for (auto &p : hdrt->qs)
            max_bh = max(max_bh, p[2]);
        for (int i = 0; i < drt->n_sites; i++)
            for (int d = 0; d < hdrt->nd; d++)
                max_dh = max(max_dh, site_matrices[i][d].dq);
        array<int, 7> factor_shape = array<int, 7>{
            max_bb + 1, 3, max_bk + 1, 3, max_bh + 1, max_bh + 1, max_dh + 1};
        factor_strides[6] = 1;
        for (int i = 6; i > 0; i--)
            factor_strides[i - 1] = factor_strides[i] * factor_shape[i];
        factors = make_shared<vector<FL>>(factor_strides[0] * factor_shape[0]);
        for (int16_t bb = 0; bb <= max_bb; bb++)
            for (int16_t db = 0; db <= 2; db++)
                for (int16_t bk = 0; bk <= max_bk; bk++)
                    for (int16_t dk = 0; dk <= 2; dk++)
                        for (int16_t fq = 0; fq <= max_bh; fq++)
                            for (int16_t iq = 0; iq <= max_bh; iq++)
                                for (int16_t dq = 0; dq <= max_dh; dq++)
                                    (*factors)[bb * factor_strides[0] +
                                               db * factor_strides[1] +
                                               bk * factor_strides[2] +
                                               dk * factor_strides[3] +
                                               fq * factor_strides[4] +
                                               iq * factor_strides[5] +
                                               dq * factor_strides[6]] =
                                        (FL)SU2Matrix<FL>::cg().wigner_9j(
                                            bk + dk - 1, 1 - (dk & 1), bk, iq,
                                            dq, fq, bb + db - 1, 1 - (db & 1),
                                            bb) *
                                        (FL)sqrtl((bk + 1) * (fq + 1) *
                                                  (bb + db) * (2 - (db & 1))) *
                                        (FL)(1 -
                                             ((((bk + dk - 1) & 1) & (dq & 1))
                                              << 1));
    }
    void fill_csr_matrix(vector<pair<pair<MKL_INT, MKL_INT>, FL>> &data,
                         GCSRMatrix<FL> &mat) const {
        const FP sparse_max_nonzero_ratio = 0.25;
        const size_t n = data.size();
        assert(mat.data == nullptr);
        assert(mat.alloc != nullptr);
        vector<size_t> idx(n), idx2;
        for (size_t i = 0; i < n; i++)
            idx[i] = i;

        sort(idx.begin(), idx.end(), [&data](size_t i, size_t j) {
            return data[i].first < data[j].first;
        });
        for (auto ii : idx)
            if (idx2.empty() || data[ii].first != data[idx2.back()].first)
                idx2.push_back(ii);
            else
                data[idx2.back()].second += data[ii].second;
        mat.nnz = (MKL_INT)idx2.size();
        if ((size_t)mat.nnz != idx2.size())
            throw runtime_error(
                "NNZ " + Parsing::to_string(idx2.size()) +
                " exceeds MKL_INT. Rebuild with -DUSE_MKL64=ON.");
        if (mat.nnz < mat.size() &&
            mat.nnz <= sparse_max_nonzero_ratio * mat.size()) {
            mat.allocate();
            MKL_INT cur_row = -1;
            for (size_t k = 0; k < idx2.size(); k++) {
                while (data[idx2[k]].first.first != cur_row)
                    mat.rows[++cur_row] = k;
                mat.data[k] = data[idx2[k]].second,
                mat.cols[k] = data[idx2[k]].first.second;
            }
            while (mat.m != cur_row)
                mat.rows[++cur_row] = mat.nnz;
        } else if (mat.nnz < mat.size()) {
            mat.nnz = mat.size();
            mat.allocate();
            for (size_t k = 0; k < idx2.size(); k++)
                mat.data[data[idx2[k]].first.second +
                         data[idx2[k]].first.first * mat.n] =
                    data[idx2[k]].second;
        } else {
            mat.allocate();
            for (size_t k = 0; k < idx2.size(); k++)
                mat.data[k] = data[idx2[k]].second;
        }
    }
    void build_hamiltonian_matrix(
        const shared_ptr<CSRSparseMatrix<S, FL>> &mat) const {
        vector<int> jh[2], jbra[2], jket[2];
        vector<LL> ph[2], pbra[2], pket[2];
        vector<FL> hv[2];
        int pi = 0;
        for (int i = 0; i < hdrt->n_init_qs; i++) {
            jh[pi].push_back(i);
            ph[pi].push_back(
                i != 0 ? ph[pi].back() +
                             hdrt->xs[(i - 1) * (hdrt->nd + 1) + hdrt->nd]
                       : 0);
            jbra[pi].push_back(0);
            pbra[pi].push_back(0);
            jket[pi].push_back(0);
            pket[pi].push_back(0);
            hv[pi].push_back((FL)1.0);
        }
        for (int k = drt->n_sites - 1; k >= 0; k--, pi ^= 1) {
            int xd = 0;
            for (int d = 0; d < hdrt->nd; d++)
                xd += (int)site_matrices[k][d].data.size();
            const size_t hsz = hv[pi].size() * xd;
            jh[pi ^ 1].reserve(hsz), jh[pi ^ 1].clear();
            ph[pi ^ 1].reserve(hsz), ph[pi ^ 1].clear();
            jbra[pi ^ 1].reserve(hsz), jbra[pi ^ 1].clear();
            pbra[pi ^ 1].reserve(hsz), pbra[pi ^ 1].clear();
            jket[pi ^ 1].reserve(hsz), jket[pi ^ 1].clear();
            pket[pi ^ 1].reserve(hsz), pket[pi ^ 1].clear();
            hv[pi ^ 1].reserve(hsz), hv[pi ^ 1].clear();
            for (size_t j = 0; j < jh[pi].size(); j++)
                for (int d = 0; d < hdrt->nd; d++) {
                    const int jhv = hdrt->jds[jh[pi][j] * hdrt->nd + d];
                    if (jhv != 0)
                        for (size_t md = 0;
                             md < (int)site_matrices[k][d].data.size(); md++) {
                            const int16_t dbra =
                                site_matrices[k][d].indices[md].first;
                            const int16_t dket =
                                site_matrices[k][d].indices[md].second;
                            const int jbv = drt->jds[jbra[pi][j]][dbra];
                            const int jkv = drt->jds[jket[pi][j]][dket];
                            if (jbv != 0 && jkv != 0) {
                                const int16_t bfq = drt->abc[jbra[pi][j]][1];
                                const int16_t kfq = drt->abc[jket[pi][j]][1];
                                const int16_t biq = drt->abc[jbv][1];
                                const int16_t kiq = drt->abc[jkv][1];
                                const int16_t mdq = site_matrices[k][d].dq;
                                const int16_t mfq = hdrt->qs[jh[pi][j]][2];
                                const int16_t miq = hdrt->qs[jhv][2];
                                const FL f =
                                    (*factors)[bfq * factor_strides[0] +
                                               (biq - bfq + 1) *
                                                   factor_strides[1] +
                                               kfq * factor_strides[2] +
                                               (kiq - kfq + 1) *
                                                   factor_strides[3] +
                                               mfq * factor_strides[4] +
                                               miq * factor_strides[5] +
                                               mdq * factor_strides[6]];
                                if (abs(f) < abs((FL)1E-14))
                                    continue;
                                jbra[pi ^ 1].push_back(jbv);
                                jket[pi ^ 1].push_back(jkv);
                                jh[pi ^ 1].push_back(jhv);
                                pbra[pi ^ 1].push_back(
                                    drt->xs[jbra[pi][j]][dbra] + pbra[pi][j]);
                                pket[pi ^ 1].push_back(
                                    drt->xs[jket[pi][j]][dket] + pket[pi][j]);
                                ph[pi ^ 1].push_back(
                                    hdrt->xs[jh[pi][j] * (hdrt->nd + 1) + d] +
                                    ph[pi][j]);
                                hv[pi ^ 1].push_back(
                                    f * hv[pi][j] *
                                    site_matrices[k][d].data[md]);
                            }
                        }
                }
        }
        jbra[pi] = vector<int>(), jbra[pi ^ 1] = vector<int>();
        jket[pi] = vector<int>(), jket[pi ^ 1] = vector<int>();
        jh[pi] = vector<int>(), jh[pi ^ 1] = vector<int>();
        pbra[pi ^ 1] = vector<LL>(), pket[pi ^ 1] = vector<LL>();
        ph[pi ^ 1] = vector<LL>(), hv[pi ^ 1] = vector<FL>();
        vector<pair<pair<MKL_INT, MKL_INT>, FL>> rv;
        rv.reserve(hv[pi].size());
        for (size_t j = 0; j < hv[pi].size(); j++)
            rv.push_back(
                make_pair(make_pair((MKL_INT)pbra[pi][j], (MKL_INT)pket[pi][j]),
                          hv[pi][j] * (*ints)[ph[pi][j]]));
        assert(mat->info->n == 1);
        fill_csr_matrix(rv, *mat->csr_data[0]);
    }
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>>
            &ops) const override {
        shared_ptr<SparseMatrix<S, FL>> zero =
            make_shared<SparseMatrix<S, FL>>(nullptr);
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S, FL> &op =
                *dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            shared_ptr<CSRSparseMatrix<S, FL>> mat =
                make_shared<CSRSparseMatrix<S, FL>>();
            mat->initialize(BigSite<S, FL>::find_site_op_info(op.q_label));
            for (int l = 0; l < mat->info->n; l++)
                mat->csr_data[l]->alloc = d_alloc;
            p.second = mat;
            switch (op.name) {
            case OpNames::H:
                build_hamiltonian_matrix(mat);
                break;
            default:
                assert(false);
                break;
            }
        }
    }
};

} // namespace block2
