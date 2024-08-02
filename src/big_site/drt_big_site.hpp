
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
#include "../core/clebsch_gordan.hpp"
#include "../core/state_info.hpp"
#include "../core/threading.hpp"
#include "../dmrg/general_hamiltonian.hpp"
#include "big_site.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename S, typename = void> struct ElemT;

template <typename S>
struct ElemT<S, typename S::is_su2_t>
    : integral_constant<ElemOpTypes, ElemOpTypes::SU2> {};

template <typename S>
struct ElemT<S, typename S::is_sz_t>
    : integral_constant<ElemOpTypes, ElemOpTypes::SZ> {};

// Distinct Row Table
template <typename S, ElemOpTypes T = ElemT<S>::value> struct DRT {
    typedef long long LL;
    vector<array<int16_t, 4>> abc;
    vector<typename S::pg_t> pgs;
    vector<typename S::pg_t> orb_sym;
    vector<array<int, 4>> jds;
    vector<array<LL, 5>> xs;
    int n_sites, n_init_qs;
    int n_core, n_virt, n_ex;
    bool single_ref;
    DRT() : n_sites(0), n_init_qs(0), n_core(0), n_virt(0), n_ex(0) {}
    DRT(int16_t a, int16_t b, int16_t c,
        typename S::pg_t ipg = (typename S::pg_t)0,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int n_core = 0, int n_virt = 0, int n_ex = 0)
        : DRT(a + abs(b) + c, vector<S>{S(a + a + abs(b), b, ipg)}, orb_sym,
              n_core, n_virt, n_ex) {}
    DRT(int n_sites, S q,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int n_core = 0, int n_virt = 0, int n_ex = 0)
        : DRT(n_sites, vector<S>{q}, orb_sym, n_core, n_virt, n_ex) {}
    DRT(int n_sites, const vector<S> &init_qs,
        const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>(),
        int n_core = 0, int n_virt = 0, int n_ex = 0, int nc_ref = 0,
        bool single_ref = false)
        : n_sites(n_sites), orb_sym(orb_sym), n_init_qs((int)init_qs.size()),
          n_core(n_core), n_virt(n_virt), n_ex(n_ex), single_ref(single_ref) {
        if (T == ElemOpTypes::SU2 || T == ElemOpTypes::SZ) {
            for (auto &q : init_qs) {
                abc.push_back(array<int16_t, 4>{
                    (int16_t)((q.n() - abs(q.twos())) >> 1), (int16_t)q.twos(),
                    (int16_t)(n_sites - ((q.n() + abs(q.twos())) >> 1)),
                    (int16_t)max(0, nc_ref + nc_ref - q.n())});
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
        const int nc = this->n_core, nv = this->n_virt, nx = this->n_ex;
        const bool sr = this->single_ref;
        const int sp = (int)(n_init_qs > 0 && abc[0][1] > 0);
        abc.resize(n_init_qs);
        pgs.resize(n_init_qs);
        auto make_abc = [](int16_t a, int16_t b, int16_t c, int16_t t,
                           int16_t d) -> array<int16_t, 4> {
            switch (d) {
            case 0:
                return array<int16_t, 4>{a, b, (int16_t)(c - 1), t};
            case 1:
                return array<int16_t, 4>{(int16_t)(a - (b <= 0)),
                                         (int16_t)(b - 1),
                                         (int16_t)(c - (b <= 0)), t};
            case 2:
                return array<int16_t, 4>{(int16_t)(a - (b >= 0)),
                                         (int16_t)(b + 1),
                                         (int16_t)(c - (b >= 0)), t};
            case 3:
                return array<int16_t, 4>{(int16_t)(a - 1), b, c, t};
            default:
                return array<int16_t, 4>{-1, -1, -1, -1};
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
        auto make_abct = [&make_abc, &nc, &nv, &sr,
                          &sp](int k, int16_t a, int16_t b, int16_t c,
                               int16_t t, int16_t d) -> array<int16_t, 4> {
            array<int16_t, 4> r = make_abc(a, b, c, t, d);
            if (sr && k >= nc + nv)
                r[3] = (int16_t)(t + (sp ? d >= 2 : d == 1 || d == 3));
            else
                r[3] =
                    (int16_t)(k < nv || k > nc + nv
                                  ? 0
                                  : (k < nc + nv ? (int)t
                                                 : max(0, nc + nc -
                                                              (a + a + abs(b) -
                                                               (d + 1) / 2))));
            return r;
        };
        auto allow_abct = [&allow_abc, &nc, &nv,
                           &nx](int k, int16_t a, int16_t b, int16_t c,
                                int16_t t, int16_t d) -> bool {
            return allow_abc(a, b, c, d) &&
                   ((k != nv && k != nc + nv) ||
                    (k == nv && a + a + abs(b) - (d + 1) / 2 <= nx - t) ||
                    (k == nc + nv &&
                     a + a + abs(b) - (d + 1) / 2 <= nc + nc + nx &&
                     a + a + abs(b) - (d + 1) / 2 >= nc + nc - nx));
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
            [](const pair<array<int16_t, 4>, typename S::pg_t> &p,
               const pair<array<int16_t, 4>, typename S::pg_t> &q) {
                return p.first != q.first ? p.first > q.first
                                          : p.second > q.second;
            };
        vector<vector<pair<array<int16_t, 4>, typename S::pg_t>>> pabc(n_sites +
                                                                       1);
        for (size_t i = 0; i < abc.size(); i++)
            pabc[0].push_back(make_pair(abc[i], pgs[i]));
        // construct graph
        for (int k = n_sites - 1, j = 0; k >= 0; k--, j++) {
            vector<pair<array<int16_t, 4>, typename S::pg_t>> &kabc =
                pabc[j + 1];
            for (const auto &abcg : pabc[j]) {
                const array<int16_t, 4> &x = abcg.first;
                const typename S::pg_t &g = abcg.second;
                if (n_core == 0 && n_virt == 0) {
                    for (int16_t d = 0; d < 4; d++)
                        if (allow_abc(x[0], x[1], x[2], d) &&
                            allow_pg(k, g, orb_sym[k], d))
                            kabc.push_back(
                                make_pair(make_abc(x[0], x[1], x[2], x[3], d),
                                          make_pg(g, orb_sym[k], d)));
                } else {
                    for (int16_t d = 0; d < 4; d++)
                        if (allow_abct(k, x[0], x[1], x[2], x[3], d) &&
                            allow_pg(k, g, orb_sym[k], d))
                            kabc.push_back(make_pair(
                                make_abct(k, x[0], x[1], x[2], x[3], d),
                                make_pg(g, orb_sym[k], d)));
                }
            }
            sort(kabc.begin(), kabc.end(), compare_abc_pg);
            kabc.resize(
                distance(kabc.begin(), unique(kabc.begin(), kabc.end())));
        }
        int n_abc = 1;
        // filter graph
        for (int k = n_sites - 1, j, i; k >= 0; k--, n_abc += j) {
            vector<pair<array<int16_t, 4>, typename S::pg_t>> &kabc = pabc[k];
            const vector<pair<array<int16_t, 4>, typename S::pg_t>> &fabc =
                pabc[k + 1];
            for (i = 0, j = 0; i < kabc.size(); i++) {
                const array<int16_t, 4> &x = kabc[i].first;
                const typename S::pg_t &g = kabc[i].second;
                bool found = false;
                for (int16_t d = 0; d < 4 && !found; d++)
                    found =
                        found ||
                        binary_search(
                            fabc.begin(), fabc.end(),
                            make_pair(n_core == 0 && n_virt == 0
                                          ? make_abc(x[0], x[1], x[2], x[3], d)
                                          : make_abct(n_sites - 1 - k, x[0],
                                                      x[1], x[2], x[3], d),
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
            p += (int)pabc[j].size();
            for (auto &abcg : pabc[j]) {
                array<int, 4> jd;
                for (int16_t d = 0; d < 4; d++) {
                    auto v = make_pair(
                        n_core == 0 && n_virt == 0
                            ? make_abc(abcg.first[0], abcg.first[1],
                                       abcg.first[2], abcg.first[3], d)
                            : make_abct(k, abcg.first[0], abcg.first[1],
                                        abcg.first[2], abcg.first[3], d),
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
        int j = 0;
        for (; j < n_init_qs && i >= xs[j].back(); j++)
            i -= xs[j].back();
        for (int k = n_sites - 1; k >= 0; k--) {
            uint8_t d = (uint8_t)(upper_bound(xs[j].begin(), xs[j].end(), i) -
                                  1 - xs[j].begin());
            i -= xs[j][d], j = jds[j][d], r[k] = "0+-2"[d];
        }
        return r;
    }
    LL index(const string &x) const {
        LL i = 0;
        int j = 0;
        if (n_init_qs > 1) {
            array<int16_t, 4> iabc = array<int16_t, 4>{0, 0, 0, 0};
            typename S::pg_t ipg = (typename S::pg_t)0;
            for (int k = 0; k < n_sites; k++)
                if (x[k] == '0')
                    iabc = array<int16_t, 4>{iabc[0], iabc[1],
                                             (int16_t)(iabc[2] + 1), 0};
                else if (x[k] == '+')
                    iabc = array<int16_t, 4>{(int16_t)(iabc[0] + (iabc[1] < 0)),
                                             (int16_t)(iabc[1] + 1),
                                             (int16_t)(iabc[2] + (iabc[1] < 0)),
                                             0},
                    ipg = S::pg_mul(ipg, orb_sym[k]);
                else if (x[k] == '-')
                    iabc = array<int16_t, 4>{(int16_t)(iabc[0] + (iabc[1] > 0)),
                                             (int16_t)(iabc[1] - 1),
                                             (int16_t)(iabc[2] + (iabc[1] > 0)),
                                             0},
                    ipg = S::pg_mul(ipg, orb_sym[k]);
                else
                    iabc = array<int16_t, 4>{(int16_t)(iabc[0] + 1), iabc[1],
                                             iabc[2], 0};
            for (; j < n_init_qs && (iabc != abc[j] || ipg != pgs[j]); j++)
                i += xs[j].back();
        }
        for (int k = n_sites - 1; k >= 0; k--) {
            uint8_t d = (uint8_t)string("0+-2").find(x[k]);
            i += xs[j][d], j = jds[j][d];
        }
        return i;
    }
    LL size() const { return xs[0].back(); }
    int q_index(S q) const {
        for (int j = 0; j < n_init_qs; j++)
            if (S(abc[j][0] + abc[j][0] + abs(abc[j][1]), abc[j][1], pgs[j]) ==
                q)
                return j;
        return -1;
    }
    pair<LL, LL> q_range(int i) const {
        LL a = 0, b = 0;
        for (int j = 0; j <= i; j++)
            a = b, b += xs[j].back();
        return make_pair(a, b);
    }
    vector<S> get_init_qs() const {
        vector<S> r(n_init_qs);
        for (int j = 0; j < n_init_qs; j++)
            r[j] = S(abc[j][0] + abc[j][0] + abs(abc[j][1]), abc[j][1], pgs[j]);
        return r;
    }
    shared_ptr<DRT<S>> operator^(int n_ex_new) const {
        return make_shared<DRT<S>>(n_sites, get_init_qs(), orb_sym, n_core,
                                   n_virt, n_ex_new, single_ref);
    }
    vector<LL> operator>>(const shared_ptr<DRT<S>> &other) const {
        vector<vector<int>> pbr(2, vector<int>()), pkr(2, vector<int>());
        vector<vector<LL>> pb(2, vector<LL>());
        size_t max_sz = min(size(), other->size());
        pbr[0].reserve(max_sz), pkr[0].reserve(max_sz), pb[0].reserve(max_sz);
        pbr[1].reserve(max_sz), pkr[1].reserve(max_sz), pb[1].reserve(max_sz);
        LL x = 0;
        for (int i = 0; i < n_init_qs; i++) {
            for (int j = 0; j < other->n_init_qs; j++) {
                pbr[0].push_back(i);
                pkr[0].push_back(j);
                pb[0].push_back(x);
            }
            x += xs[i].back();
        }
        assert(n_sites == other->n_sites);
        int pi = 0, pj = pi ^ 1;
        for (int k = 0; k < n_sites; k++, pi ^= 1, pj ^= 1) {
            pbr[pj].clear(), pkr[pj].clear(), pb[pj].clear();
            for (int j = 0; j < pbr[pi].size(); j++)
                for (int d = 0; d < 4; d++)
                    if (jds[pbr[pi][j]][d] != 0 &&
                        other->jds[pkr[pi][j]][d] != 0) {
                        pbr[pj].push_back(jds[pbr[pi][j]][d]);
                        pkr[pj].push_back(other->jds[pkr[pi][j]][d]);
                        pb[pj].push_back(pb[pi][j] + xs[pbr[pi][j]][d]);
                    }
        }
        return pb[pi];
    }
    shared_ptr<StateInfo<S>> get_basis() const {
        shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
        b->allocate(n_init_qs);
        for (int i = 0; i < n_init_qs; i++) {
            b->quanta[i] =
                S(abc[i][0] + abc[i][0] + abs(abc[i][1]), abc[i][1], pgs[i]);
            b->n_states[i] = (ubond_t)xs[i][4];
        }
        b->sort_states();
        return b;
    }
    string to_str() const {
        stringstream ss;
        ss << setw(4) << "J" << setw(6) << "K" << setw(4) << "A" << setw(4)
           << "B" << setw(4) << "C";
        if (n_core != 0 || n_virt != 0)
            ss << setw(4) << "T";
        ss << setw(6) << "PG" << setw(6) << "JD0" << setw(6) << "JD1" << setw(6)
           << "JD2" << setw(6) << "JD3"
           << " " << setw(12) << "X0"
           << " " << setw(12) << "X1"
           << " " << setw(12) << "X2"
           << " " << setw(12) << "X3" << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            k = abc[i][0] + abs(abc[i][1]) + abc[i][2];
            ss << setw(4) << (i + 1);
            if (k == pk)
                ss << setw(6) << "";
            else
                ss << setw(6) << k;
            ss << setw(4) << abc[i][0] << setw(4) << abc[i][1] << setw(4)
               << abc[i][2];
            if (n_core != 0 || n_virt != 0)
                ss << setw(4) << abc[i][3];
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
template <typename S, ElemOpTypes T = ElemT<S>::value> struct HDRT {
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
    HDRT() : n_sites(0), n_init_qs(0), nd(0) {}
    HDRT(int n_sites, const vector<pair<S, pair<int16_t, int16_t>>> &init_qs,
         const vector<typename S::pg_t> &orb_sym = vector<typename S::pg_t>())
        : n_sites(n_sites), n_init_qs((int)init_qs.size()), orb_sym(orb_sym),
          nd(0) {
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
    static string get_sub_expr(const string &expr, int i, int j) {
        if (T == ElemOpTypes::SU2)
            return SpinPermRecoupling::get_sub_expr(expr, i, j);
        else
            return expr.substr(i, j - i);
    }
    static int get_target_twos(const string &expr) {
        if (T == ElemOpTypes::SU2)
            return SpinPermRecoupling::get_target_twos(expr);
        else {
            int dq = 0;
            for (char x : expr)
                dq += (x == 'c' || x == 'D') - (x == 'C' || x == 'd');
            return dq;
        }
    }
    void initialize_steps(const vector<shared_ptr<SpinPermScheme>> &schemes) {
        d_map.clear(), d_step.clear();
        d_map[make_pair("", 0)] = 0;
        // dk dn d2s dw dl dpg
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
                        string x = get_sub_expr(expr, k, l);
                        int8_t dq = get_target_twos(get_sub_expr(expr, 0, l)) -
                                    get_target_twos(get_sub_expr(expr, 0, k));
                        if (!d_map.count(make_pair(x, dq))) {
                            int16_t xc =
                                (int16_t)count(x.begin(), x.end(), 'C');
                            int16_t xd =
                                (int16_t)count(x.begin(), x.end(), 'D');
                            if (T != ElemOpTypes::SU2) {
                                xc += (int16_t)count(x.begin(), x.end(), 'c');
                                xd += (int16_t)count(x.begin(), x.end(), 'd');
                            }
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
        vector<int16_t> ddq(nd);
        for (int16_t d = 0; d < nd; d++)
            ddq[d] = SpinPermRecoupling::get_target_twos(d_expr[d].first);
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
                    if (allow_q(nq) && allow_pg(k, ng) &&
                        (T != ElemOpTypes::SU2 ||
                         SU2CG::triangle(ddq[d], qg.first[2], nq[2])))
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
                if (found || k == 0)
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
            p += (int)pqs[j].size();
            for (auto &qg : pqs[j]) {
                for (int16_t d = 0; d < nd; d++) {
                    const auto &nqg =
                        make_pair(make_q(qg.first, d_step[d]),
                                  make_pg(qg.second, orb_sym[k], d_step[d]));
                    bool allowed =
                        allow_q(nqg.first) && allow_pg(k, nqg.second) &&
                        (T != ElemOpTypes::SU2 ||
                         SU2CG::triangle(ddq[d], qg.first[2], nqg.first[2]));
                    auto it = lower_bound(pqs[j + 1].begin(), pqs[j + 1].end(),
                                          nqg, compare_q_pg);
                    int jd = allowed && it != pqs[j + 1].end() && *it == nqg
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
        vector<uint16_t> kidx;
        vector<pair<string, int16_t>> cds;
        int j = 0;
        for (; i >= xs[j * (nd + 1) + nd]; j++)
            i -= xs[j * (nd + 1) + nd];
        for (int k = n_sites - 1; k >= 0; k--) {
            int16_t d =
                (int16_t)(upper_bound(xs.begin() + j * (nd + 1),
                                      xs.begin() + (j + 1) * (nd + 1), i) -
                          1 - (xs.begin() + j * (nd + 1)));
            pair<string, int8_t> dx = d_expr[d];
            if (dx.first != "") {
                for (size_t l = 0; l < d_step[d][4]; l++)
                    kidx.insert(kidx.begin(), (uint16_t)k);
                cds.insert(cds.begin(), make_pair(dx.first, qs[j][2]));
            }
            i -= xs[j * (nd + 1) + d], j = jds[j * nd + d];
        }
        string r = "";
        for (auto &cd : cds)
            if (r == "")
                r = cd.first;
            else {
                stringstream ss;
                if (T == ElemOpTypes::SU2)
                    ss << "(" << r << "+" << cd.first << ")" << cd.second;
                else
                    ss << r << cd.first;
                r = ss.str();
            }
        return make_pair(r, kidx);
    }
    LL index(const string &expr, const vector<uint16_t> &idxs) const {
        vector<int16_t> ds(n_sites, d_map.at(make_pair("", 0)));
        for (int k = 0, n = (int)idxs.size(), l; k < n; k = l) {
            for (l = k; l < n && idxs[k] == idxs[l];)
                l++;
            string x = get_sub_expr(expr, k, l);
            int8_t dq = get_target_twos(get_sub_expr(expr, 0, l)) -
                        get_target_twos(get_sub_expr(expr, 0, k));
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
            ss << setw(4 + (dk < 10)) << "JD" << (int)dk;
        for (int16_t dk = 0; dk < nd; dk++)
            ss << setw(4 + (dk < 10)) << "X" << (int)dk;
        ss << endl;
        int n = n_rows();
        int pk = -1;
        for (int i = 0, k; i < n; i++, pk = k) {
            ss << setw(4) << (i + 1);
            k = qs[i][0];
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

template <typename S, typename FL, typename = void> struct ElemMat;

template <typename S, typename FL> struct ElemMat<S, FL, typename S::is_su2_t> {
    vector<FL> data;
    vector<pair<int16_t, int16_t>> indices;
    int16_t dq;
    ElemMat(int16_t dq, const vector<FL> &data,
            const vector<pair<int16_t, int16_t>> &indices)
        : dq(dq), indices(indices), data(data) {}
    static SU2CG &cg() {
        static SU2CG _cg;
        return _cg;
    }
    static const vector<ElemMat<S, FL>> &op_matrices() {
        static vector<ElemMat<S, FL>> _mats = vector<ElemMat<S, FL>>{
            ElemMat<S, FL>(0, vector<FL>{(FL)1.0, (FL)1.0, (FL)1.0},
                           vector<pair<int16_t, int16_t>>{make_pair(0, 0),
                                                          make_pair(1, 1),
                                                          make_pair(2, 2)}),
            ElemMat<S, FL>(1, vector<FL>{(FL)1.0, (FL)(-sqrtl(2))},
                           vector<pair<int16_t, int16_t>>{make_pair(1, 0),
                                                          make_pair(2, 1)}),
            ElemMat<S, FL>(1, vector<FL>{(FL)sqrtl(2), (FL)1.0},
                           vector<pair<int16_t, int16_t>>{make_pair(0, 1),
                                                          make_pair(1, 2)})};
        return _mats;
    }
    static ElemMat<S, FL> multiply(const ElemMat<S, FL> &a,
                                   const ElemMat<S, FL> &b, int16_t dq) {
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
        return ElemMat<S, FL>(dq, data, indices);
    }
    static ElemMat<S, FL> build_matrix(const string &expr) {
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
        ElemMat<S, FL> a = build_matrix(expr.substr(1, ix - 1));
        ElemMat<S, FL> b = build_matrix(expr.substr(ix + 1, iy - ix - 1));
        return multiply(a, b, dq);
    }
    ElemMat<S, FL> expand() const {
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
        return ElemMat<S, FL>(dq, rd, ri);
    }
};

template <typename S, typename FL> struct ElemMat<S, FL, typename S::is_sz_t> {
    vector<FL> data;
    vector<pair<int16_t, int16_t>> indices;
    int16_t dq;
    ElemMat(int16_t dq, const vector<FL> &data,
            const vector<pair<int16_t, int16_t>> &indices)
        : dq(dq), indices(indices), data(data) {}
    static TrivialCG &cg() {
        static TrivialCG _cg;
        return _cg;
    }
    static const vector<ElemMat<S, FL>> &op_matrices() {
        static vector<ElemMat<S, FL>> _mats = vector<ElemMat<S, FL>>{
            ElemMat<S, FL>(0, vector<FL>{(FL)1.0, (FL)1.0, (FL)1.0, (FL)1.0},
                           vector<pair<int16_t, int16_t>>{
                               make_pair(0, 0), make_pair(1, 1),
                               make_pair(2, 2), make_pair(3, 3)}),
            ElemMat<S, FL>(1, vector<FL>{(FL)1.0, (FL)1.0},
                           vector<pair<int16_t, int16_t>>{make_pair(1, 0),
                                                          make_pair(3, 2)}),
            ElemMat<S, FL>(1, vector<FL>{(FL)1.0, (FL)1.0},
                           vector<pair<int16_t, int16_t>>{make_pair(0, 1),
                                                          make_pair(2, 3)}),
            ElemMat<S, FL>(1, vector<FL>{(FL)1.0, (FL)-1.0},
                           vector<pair<int16_t, int16_t>>{make_pair(2, 0),
                                                          make_pair(3, 1)}),
            ElemMat<S, FL>(1, vector<FL>{(FL)1.0, (FL)-1.0},
                           vector<pair<int16_t, int16_t>>{make_pair(0, 2),
                                                          make_pair(1, 3)})};
        return _mats;
    }
    static ElemMat<S, FL> multiply(const ElemMat<S, FL> &a,
                                   const ElemMat<S, FL> &b, int16_t dq) {
        map<pair<int16_t, int16_t>, FL> r;
        for (int i = 0; i < (int)a.data.size(); i++)
            for (int j = 0; j < (int)b.data.size(); j++)
                if (a.indices[i].second == b.indices[j].first)
                    r[make_pair(a.indices[i].first, b.indices[j].second)] +=
                        a.data[i] * b.data[j];
        vector<FL> data;
        vector<pair<int16_t, int16_t>> indices;
        for (auto &x : r)
            if (x.second != (FL)0.0)
                indices.push_back(x.first), data.push_back(x.second);
        return ElemMat<S, FL>(dq, data, indices);
    }
    static ElemMat<S, FL> build_matrix(const string &expr) {
        if (expr == "")
            return op_matrices()[0];
        else if (expr == "c")
            return op_matrices()[1];
        else if (expr == "d")
            return op_matrices()[2];
        else if (expr == "C")
            return op_matrices()[3];
        else if (expr == "D")
            return op_matrices()[4];
        ElemMat<S, FL> a = build_matrix(expr.substr(0, 1));
        ElemMat<S, FL> b = build_matrix(expr.substr(1, expr.length() - 1));
        int dq = 0;
        for (char x : expr)
            dq += (x == 'c' || x == 'D') - (x == 'C' || x == 'd');
        return multiply(a, b, dq);
    }
    ElemMat<S, FL> expand() const { return *this; }
};

template <typename S, typename FL, ElemOpTypes T = ElemT<S>::value>
struct HDRTScheme {
    typedef long long LL;
    shared_ptr<HDRT<S, T>> hdrt;
    vector<shared_ptr<SpinPermScheme>> schemes;
    map<string, map<vector<uint16_t>, int>> expr_mp;
    vector<vector<pair<int, LL>>> hjumps;
    vector<vector<int16_t>> ds;
    vector<map<typename S::pg_t, pair<int, LL>>> jis;
    int n_patterns;
    HDRTScheme(const shared_ptr<HDRT<S, T>> &hdrt,
               const vector<shared_ptr<SpinPermScheme>> &schemes)
        : hdrt(hdrt), schemes(schemes) {
        for (const auto &scheme : schemes)
            for (int i = 0; i < (int)scheme->data.size(); i++)
                for (const auto &d : scheme->data[i])
                    for (const auto &dex : d.second)
                        if (!expr_mp[dex.second].count(
                                scheme->index_patterns[i]))
                            expr_mp[dex.second][scheme->index_patterns[i]] = 0;
        n_patterns = 0;
        for (auto &m : expr_mp)
            for (auto &mm : m.second)
                mm.second = n_patterns++;
        ds = vector<vector<int16_t>>(n_patterns);
        jis = vector<map<typename S::pg_t, pair<int, LL>>>(n_patterns);
        int im;
        for (auto &m : expr_mp)
            for (auto &mm : m.second) {
                im = mm.second;
                for (int k = 0, n = (int)mm.first.size(), l; k < n; k = l) {
                    for (l = k; l < n && mm.first[k] == mm.first[l];)
                        l++;
                    string x = HDRT<S, T>::get_sub_expr(m.first, k, l);
                    int8_t dq = HDRT<S, T>::get_target_twos(
                                    HDRT<S, T>::get_sub_expr(m.first, 0, l)) -
                                HDRT<S, T>::get_target_twos(
                                    HDRT<S, T>::get_sub_expr(m.first, 0, k));
                    if (!hdrt->d_map.count(make_pair(x, dq)))
                        throw runtime_error("expr not found : " + x +
                                            " dq = " + string(1, '0' + dq) +
                                            " expr = " + m.first);
                    ds[im].push_back(hdrt->d_map.at(make_pair(x, dq)));
                }
                array<int16_t, 5> iq = hdrt->qs.back();
                for (int k = 0; k < (int)ds[im].size(); k++)
                    iq = array<int16_t, 5>{
                        (int16_t)(iq[0] + hdrt->d_step[ds[im][k]][0]),
                        (int16_t)(iq[1] + hdrt->d_step[ds[im][k]][1]),
                        (int16_t)(iq[2] + hdrt->d_step[ds[im][k]][2]),
                        (int16_t)(iq[3] + hdrt->d_step[ds[im][k]][3]),
                        (int16_t)(iq[4] + hdrt->d_step[ds[im][k]][4])};
                iq[0] = (int16_t)hdrt->n_sites;
                LL i = 0;
                for (int j = 0; j < hdrt->n_init_qs; j++) {
                    if (iq == hdrt->qs[j])
                        jis[im][hdrt->pgs[j]] = make_pair(j, i);
                    i += hdrt->xs[j * (hdrt->nd + 1) + hdrt->nd];
                }
            }
        hjumps = vector<vector<pair<int, LL>>>(
            hdrt->n_rows(), vector<pair<int, LL>>{make_pair(0, 0)});
        for (int j = hdrt->n_rows() - 1, k; j >= 0; j--) {
            hjumps[j][0].first = j;
            if ((k = hdrt->jds[j * hdrt->nd + 0]) != 0) {
                hjumps[j].insert(hjumps[j].end(), hjumps[k].begin(),
                                 hjumps[k].end());
                const LL x = hdrt->xs[j * (hdrt->nd + 1) + 0];
                for (int l = 1; l < (int)hjumps[j].size(); l++)
                    hjumps[j][l].second += x;
            }
        }
    }
    virtual ~HDRTScheme() = default;
    shared_ptr<vector<FL>>
    sort_integral(const shared_ptr<GeneralFCIDUMP<FL>> &gfd) const {
        int ntg = threading->activate_global();
        shared_ptr<vector<FL>> r =
            make_shared<vector<FL>>(hdrt->size(), (FL)0.0);
        for (size_t ix = 0; ix < gfd->exprs.size(); ix++) {
            const string &expr = gfd->exprs[ix];
            const int nn = SpinPermRecoupling::count_cds(expr);
            const map<vector<uint16_t>, int> &xmp = expr_mp.at(expr);
#ifdef _MSC_VER
#pragma omp parallel for schedule(static, 100) num_threads(ntg)
            for (int ip = 0; ip < (int)gfd->indices[ix].size(); ip += nn)
#else
#pragma omp parallel for schedule(static, 100) num_threads(ntg)
            for (size_t ip = 0; ip < gfd->indices[ix].size(); ip += nn)
#endif
            {
                vector<uint16_t> idx(gfd->indices[ix].begin() + ip,
                                     gfd->indices[ix].begin() + ip + nn);
                vector<uint16_t> idx_mat(nn);
                if (nn >= 1)
                    idx_mat[0] = 0;
                for (int j = 1; j < nn; j++)
                    idx_mat[j] = idx_mat[j - 1] + (idx[j] != idx[j - 1]);
                typename S::pg_t ipg = hdrt->pgs.back();
                for (auto &x : idx)
                    ipg = S::pg_mul(ipg, hdrt->orb_sym[x]);
                int im = xmp.at(idx_mat);
                if (!jis[im].count(ipg)) {
                    throw runtime_error("Small integral elements violating "
                                        "point group symmetry!");
                }
                int j = jis[im].at(ipg).first, k = hdrt->n_sites - 1;
                LL i = jis[im].at(ipg).second;
                const vector<int16_t> &xds = ds[im];
                for (int l = nn - 1, g, m = (int)xds.size() - 1; l >= 0;
                     l = g, m--, k--) {
                    for (g = l; g >= 0 && idx[g] == idx[l];)
                        g--;
                    i += hjumps[j][k - idx[l]].second;
                    j = hjumps[j][k - idx[l]].first;
                    i += hdrt->xs[j * (hdrt->nd + 1) + xds[m]];
                    j = hdrt->jds[j * hdrt->nd + xds[m]];
                    k = idx[l];
                }
                i += hjumps[j][k + 1].second;
                (*r)[i] = gfd->data[ix][ip / nn];
            }
        }
        threading->activate_normal();
        return r;
    }
    shared_ptr<vector<vector<LL>>> sort_npdm() const {
        shared_ptr<vector<vector<LL>>> r =
            make_shared<vector<vector<LL>>>(n_patterns);
        for (const auto &xmp : expr_mp) {
            const string &expr = xmp.first;
            for (const auto &g : xmp.second) {
                const vector<uint16_t> &idx_pat = g.first;
                int nn = (int)idx_pat.size(), im = g.second;
                shared_ptr<NPDMCounter> counter =
                    make_shared<NPDMCounter>(nn, hdrt->n_sites + 1);
                uint32_t cnt =
                    counter->count_left(idx_pat, hdrt->n_sites - 1, false);
                vector<LL> &rr = (*r)[im];
                rr.resize(cnt, -1);
                vector<uint16_t> idx;
                counter->init_left(idx_pat, hdrt->n_sites - 1, false, idx);
                for (uint32_t il = 0; il < cnt; il++) {
                    typename S::pg_t ipg = hdrt->pgs.back();
                    for (auto &x : idx)
                        ipg = S::pg_mul(ipg, hdrt->orb_sym[x]);
                    if (jis[im].count(ipg)) {
                        int j = jis[im].at(ipg).first, k = hdrt->n_sites - 1;
                        LL i = jis[im].at(ipg).second;
                        const vector<int16_t> &xds = ds[im];
                        for (int l = nn - 1, g, m = (int)xds.size() - 1; l >= 0;
                             l = g, m--, k--) {
                            for (g = l; g >= 0 && idx[g] == idx[l];)
                                g--;
                            i += hjumps[j][k - idx[l]].second;
                            j = hjumps[j][k - idx[l]].first;
                            i += hdrt->xs[j * (hdrt->nd + 1) + xds[m]];
                            j = hdrt->jds[j * hdrt->nd + xds[m]];
                            k = idx[l];
                        }
                        i += hjumps[j][k + 1].second;
                        rr[il] = i;
                    }
                    counter->next_left(idx_pat, hdrt->n_sites - 1, idx);
                }
            }
        }
        return r;
    }
};

template <typename S, typename FL> struct DRTBigSiteBase : BigSite<S, FL> {
    typedef integral_constant<ElemOpTypes, ElemT<S>::value> T;
    typedef typename GMatrix<FL>::FP FP;
    typedef long long LL;
    using BigSite<S, FL>::n_orbs;
    using BigSite<S, FL>::basis;
    shared_ptr<FCIDUMP<FL>> fcidump = nullptr;
    shared_ptr<DRT<S>> drt;
    shared_ptr<vector<FL>> factors;
    array<size_t, 7> factor_strides;
    bool is_right;
    int iprint;
    int n_total_orbs;
    const static int max_cg = 10;
    FP cutoff = 1E-14;
    DRTBigSiteBase(const vector<S> &qs, bool is_right, int n_orbs,
                   const vector<typename S::pg_t> &orb_sym,
                   const shared_ptr<FCIDUMP<FL>> &fcidump = nullptr,
                   int iprint = 0)
        : BigSite<S, FL>(n_orbs), is_right(is_right), fcidump(fcidump),
          n_total_orbs(fcidump == nullptr ? 0 : fcidump->n_sites()),
          iprint(iprint) {
        vector<typename S::pg_t> big_orb_sym(n_orbs);
        if (!is_right)
            for (int i = 0; i < n_orbs; i++)
                big_orb_sym[i] = orb_sym[i];
        else
            for (int i = 0; i < n_orbs; i++)
                big_orb_sym[i] = orb_sym[n_orbs - 1 - i];
        drt = make_shared<DRT<S>>(n_orbs, qs, big_orb_sym);
        if (iprint >= 3)
            cout << "DRT ::" << endl << drt->to_str() << endl;
        basis = drt->get_basis();
    }
    virtual ~DRTBigSiteBase() = default;
    void
    fill_csr_matrix_from_coo(const vector<pair<MKL_INT, MKL_INT>> &coo_idxs,
                             vector<FL> &values, GCSRMatrix<FL> &mat) const {
        const FP sparse_max_nonzero_ratio = 0.25;
        assert(mat.data == nullptr);
        assert(mat.alloc != nullptr);
        assert(values.size() == coo_idxs.size());
        const size_t n = values.size();
        vector<size_t> idx(n), idx2;
        for (size_t i = 0; i < n; i++)
            idx[i] = i;

        sort(idx.begin(), idx.end(), [&coo_idxs](size_t i, size_t j) {
            return coo_idxs[i] < coo_idxs[j];
        });
        for (auto ii : idx)
            if (idx2.empty() || coo_idxs[ii] != coo_idxs[idx2.back()])
                idx2.push_back(ii);
            else
                values[idx2.back()] += values[ii];
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
                while (coo_idxs[idx2[k]].first != cur_row)
                    mat.rows[++cur_row] = (MKL_INT)k;
                mat.data[k] = values[idx2[k]],
                mat.cols[k] = coo_idxs[idx2[k]].second;
            }
            while (mat.m != cur_row)
                mat.rows[++cur_row] = mat.nnz;
        } else if (mat.nnz < mat.size()) {
            mat.nnz = (MKL_INT)mat.size();
            mat.allocate();
            for (size_t k = 0; k < idx2.size(); k++)
                mat.data[coo_idxs[idx2[k]].second +
                         coo_idxs[idx2[k]].first * mat.n] = values[idx2[k]];
        } else {
            mat.allocate();
            for (size_t k = 0; k < idx2.size(); k++)
                mat.data[k] = values[idx2[k]];
        }
    }
    void fill_csr_matrix(const vector<vector<MKL_INT>> &col_idxs,
                         const vector<vector<FL>> &values,
                         GCSRMatrix<FL> &mat) const {
        const FP sparse_max_nonzero_ratio = 0.25;
        assert(mat.data == nullptr);
        assert(mat.alloc != nullptr);
        size_t nnz = 0;
        for (auto &xv : values)
            nnz += xv.size();
        mat.nnz = (MKL_INT)nnz;
        if ((size_t)mat.nnz != nnz)
            throw runtime_error(
                "NNZ " + Parsing::to_string(nnz) +
                " exceeds MKL_INT. Rebuild with -DUSE_MKL64=ON.");
        if (mat.nnz < mat.size() &&
            mat.nnz <= sparse_max_nonzero_ratio * mat.size()) {
            mat.allocate();
            for (size_t i = 0, k = 0; i < values.size(); i++) {
                mat.rows[i] = (MKL_INT)k;
                memcpy(&mat.data[k], &values[i][0],
                       sizeof(FL) * values[i].size());
                memcpy(&mat.cols[k], &col_idxs[i][0],
                       sizeof(MKL_INT) * col_idxs[i].size());
                k += values[i].size();
            }
            mat.rows[values.size()] = mat.nnz;
        } else {
            mat.nnz = (MKL_INT)mat.size();
            mat.allocate();
            for (size_t i = 0; i < values.size(); i++)
                for (size_t j = 0; j < values[i].size(); j++)
                    mat.data[col_idxs[i][j] + i * mat.n] = values[i][j];
        }
    }
    void print_hdrt_infos(const set<S> &iqs, const vector<string> &std_exprs,
                          const shared_ptr<HDRT<S>> &hdrt) const {
        if (iprint >= 1) {
            cout << "    HDRT :: QS = [ ";
            for (auto iq : iqs)
                cout << iq << " ";
            cout << "] EXPRS = [ ";
            for (size_t ix = 0; ix < std_exprs.size(); ix++)
                cout << std_exprs[ix]
                     << (ix == std_exprs.size() - 1 ? " " : " + ");
            cout << "] NTERMS = " << hdrt->size() << endl;
            if (iprint >= 3)
                cout << "    HDRT = " << endl << hdrt->to_str() << endl;
            if (iprint >= 4) {
                cout << "    HDRT STEPS = " << endl;
                for (LL ih = 0; ih < (LL)hdrt->d_expr.size(); ih++)
                    cout << "   * " << setw(8) << ih << " = "
                         << hdrt->d_expr[ih].first
                         << " :: " << (hdrt->d_expr[ih].second >= 0 ? "+" : "")
                         << (int)hdrt->d_expr[ih].second << "" << endl;
                cout << "    HDRT TERMS = " << endl;
                for (LL ih = 0; ih < (LL)hdrt->size(); ih++) {
                    auto hterm = (*hdrt)[ih];
                    assert(hdrt->index(hterm.first, hterm.second) == ih);
                    cout << "   * " << setw(8) << ih << " = " << hterm.first
                         << " [";
                    for (auto xh : hterm.second)
                        cout << " " << (int)xh;
                    cout << " ]" << endl;
                }
            }
        }
    }
    vector<shared_ptr<GTensor<FL>>>
    sort_npdm(const vector<shared_ptr<SpinPermScheme>> &schemes,
              const shared_ptr<vector<FL>> &pr,
              const map<string, map<vector<uint16_t>, int>> &expr_mp,
              const vector<size_t> &psum) const {
        vector<shared_ptr<GTensor<FL>>> r(schemes.size());
        for (size_t it = 0; it < schemes.size(); it++) {
            int n_op = (int)schemes[it]->index_patterns[0].size();
            vector<MKL_INT> shape(n_op, n_orbs);
            shared_ptr<NPDMCounter> counter =
                make_shared<NPDMCounter>(n_op, n_orbs + 1);
            r[it] = make_shared<GTensor<FL>>(shape);
            for (size_t j = 0; j < schemes[it]->index_patterns.size(); j++) {
                for (const auto &m : schemes[it]->data[j]) {
                    const vector<uint16_t> &perm = m.first;
                    vector<uint64_t> mx(perm.size());
                    uint64_t mxx = 1;
                    for (int k = (int)perm.size() - 1; k >= 0;
                         k--, mxx *= n_orbs)
                        mx[perm[k]] = mxx;
                    const uint64_t lcnt = counter->count_left(
                        schemes[it]->index_patterns[j], n_orbs - 1, false);
                    vector<uint16_t> idx;
                    counter->init_left(schemes[it]->index_patterns[j],
                                       n_orbs - 1, false, idx);
                    for (uint64_t il = 0; il < lcnt; il++) {
                        for (auto &xr : m.second) {
                            int im = expr_mp.at(xr.second).at(
                                schemes[it]->index_patterns[j]);
                            uint64_t ix = 0;
                            for (int k = 0; k < (int)mx.size(); k++)
                                ix += idx[k] * mx[k];
                            (*r[it]->data)[ix] +=
                                (FL)xr.first * (*pr)[psum[im] + il];
                        }
                        counter->next_left(schemes[it]->index_patterns[j],
                                           n_orbs - 1, idx);
                    }
                }
            }
        }
        return r;
    }
    vector<vector<ElemMat<S, FL>>>
    get_site_matrices(const shared_ptr<HDRT<S>> &hdrt) const {
        vector<vector<ElemMat<S, FL>>> site_matrices(drt->n_sites);
        for (int i = 0; i < drt->n_sites; i++) {
            for (int d = 0; d < hdrt->nd; d++)
                site_matrices[i].push_back(
                    ElemMat<S, FL>::build_matrix(hdrt->d_expr[d].first)
                        .expand());
        }
        return site_matrices;
    }
    void prepare_factors() {
        if (T::value != ElemOpTypes::SU2)
            return;
        int16_t max_bb = 0, max_bk = 0, max_bh = max_cg, max_dh = max_cg;
        for (auto &p : drt->abc)
            max_bb = max(max_bb, p[1]);
        for (auto &p : drt->abc)
            max_bk = max(max_bk, p[1]);
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
                                        (FL)ElemMat<S, FL>::cg().wigner_9j(
                                            bk + dk - 1, 1 - (dk & 1), bk, iq,
                                            dq, fq, bb + db - 1, 1 - (db & 1),
                                            bb) *
                                        (FL)sqrtl((bk + 1) * (fq + 1) *
                                                  (bb + db) * (2 - (db & 1))) *
                                        (FL)(1 -
                                             ((((bk + dk - 1) & 1) & (dq & 1))
                                              << 1));
    }
    void build_npdm_operator_matrices(
        const shared_ptr<HDRT<S>> &hdrt,
        const vector<vector<ElemMat<S, FL>>> &site_matrices,
        const vector<pair<LL, FL>> &mat_idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const {
        if (mats.size() == 0)
            return;
        assert(mat_idxs.size() == mats.size());
        int ntg = threading->activate_global();
        vector<vector<vector<int>>> jbra(ntg, vector<vector<int>>(2));
        vector<vector<vector<int>>> jket(ntg, vector<vector<int>>(2));
        vector<vector<vector<pair<MKL_INT, MKL_INT>>>> pbk(
            ntg, vector<vector<pair<MKL_INT, MKL_INT>>>(2));
        vector<vector<vector<FL>>> hv(ntg, vector<vector<FL>>(2));
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int it = 0; it < (int)mats.size(); it++) {
            for (int im = 0; im < mats[it]->info->n; im++) {
                const int tid = threading->get_thread_id();
                S opdq = mats[it]->info->delta_quantum;
                S qbra = mats[it]->info->quanta[im].get_bra(opdq);
                S qket = mats[it]->info->quanta[im].get_ket();
                // SU2 and fermion factor for exchange:
                //   ket x op -> op x ket when is_right
                FL xf = (FL)1.0;
                if (T::value == ElemOpTypes::SU2 && is_right)
                    xf *= (FL)(1 - ((opdq.twos() & qket.twos() & 1) << 1)) *
                          (FL)ElemMat<S, FL>::cg().phase(
                              opdq.twos(), qket.twos(), qbra.twos());
                int imb = drt->q_index(qbra), imk = drt->q_index(qket);
                assert(mats[it]->info->n_states_bra[im] == drt->xs[imb].back());
                assert(mats[it]->info->n_states_ket[im] == drt->xs[imk].back());
                int pi = 0, pj = pi ^ 1;
                vector<vector<pair<MKL_INT, MKL_INT>>> &xpbk = pbk[tid];
                vector<vector<int>> &xjb = jbra[tid], &xjk = jket[tid];
                vector<vector<FL>> &xhv = hv[tid];
                xpbk[pi].clear(), xjb[pi].clear();
                xjk[pi].clear(), xhv[pi].clear();
                int jh = 0;
                LL ih = mat_idxs[it].first;
                for (; ih >= hdrt->xs[jh * (hdrt->nd + 1) + hdrt->nd]; jh++)
                    ih -= hdrt->xs[jh * (hdrt->nd + 1) + hdrt->nd];
                xpbk[pi].push_back(make_pair(0, 0));
                xjb[pi].push_back(imb), xjk[pi].push_back(imk);
                xhv[pi].push_back(mat_idxs[it].second * xf);
                for (int k = drt->n_sites - 1; k >= 0; k--, pi ^= 1, pj ^= 1) {
                    int16_t dh =
                        (int16_t)(upper_bound(hdrt->xs.begin() +
                                                  jh * (hdrt->nd + 1),
                                              hdrt->xs.begin() +
                                                  (jh + 1) * (hdrt->nd + 1),
                                              ih) -
                                  1 - (hdrt->xs.begin() + jh * (hdrt->nd + 1)));
                    const int jhv = hdrt->jds[jh * hdrt->nd + dh];
                    const ElemMat<S, FL> &smat = site_matrices[k][dh];
                    const size_t hsz = xhv[pi].size() * smat.data.size();
                    xpbk[pj].reserve(hsz), xpbk[pj].clear();
                    xjb[pj].reserve(hsz), xjb[pj].clear();
                    xjk[pj].reserve(hsz), xjk[pj].clear();
                    xhv[pj].reserve(hsz), xhv[pj].clear();
                    for (size_t j = 0; j < xjk[pi].size(); j++)
                        for (size_t md = 0; md < smat.data.size(); md++) {
                            const int16_t dbra = smat.indices[md].first;
                            const int16_t dket = smat.indices[md].second;
                            const int jbv = drt->jds[xjb[pi][j]][dbra];
                            const int jkv = drt->jds[xjk[pi][j]][dket];
                            if (jbv == 0 || jkv == 0)
                                continue;
                            const int16_t bfq = drt->abc[xjb[pi][j]][1];
                            const int16_t kfq = drt->abc[xjk[pi][j]][1];
                            const int16_t biq = drt->abc[jbv][1];
                            const int16_t kiq = drt->abc[jkv][1];
                            const int16_t mdq = smat.dq;
                            const int16_t mfq = hdrt->qs[jh][2];
                            const int16_t miq = hdrt->qs[jhv][2];
                            const FL f =
                                T::value == ElemOpTypes::SU2
                                    ? (*factors)[bfq * factor_strides[0] +
                                                 (biq - bfq + 1) *
                                                     factor_strides[1] +
                                                 kfq * factor_strides[2] +
                                                 (kiq - kfq + 1) *
                                                     factor_strides[3] +
                                                 mfq * factor_strides[4] +
                                                 miq * factor_strides[5] +
                                                 mdq * factor_strides[6]]
                                    : (FL)(1 - (((kiq & 1) & (mdq & 1)) << 1));
                            if (abs(f) < (FP)1E-14)
                                continue;
                            xjb[pj].push_back(jbv);
                            xjk[pj].push_back(jkv);
                            xpbk[pj].push_back(
                                make_pair((MKL_INT)drt->xs[xjb[pi][j]][dbra] +
                                              xpbk[pi][j].first,
                                          (MKL_INT)drt->xs[xjk[pi][j]][dket] +
                                              xpbk[pi][j].second));
                            xhv[pj].push_back(f * xhv[pi][j] * smat.data[md]);
                        }
                    ih -= hdrt->xs[jh * (hdrt->nd + 1) + dh];
                    jh = jhv;
                }
                fill_csr_matrix_from_coo(xpbk[pi], xhv[pi],
                                         *mats[it]->csr_data[im]);
            }
        }
        threading->activate_normal();
    }
    void build_operator_matrices(
        const shared_ptr<HDRT<S>> &hdrt,
        const vector<vector<ElemMat<S, FL>>> &site_matrices,
        const vector<shared_ptr<vector<FL>>> &ints,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const {
        if (mats.size() == 0)
            return;
        assert(ints.size() == mats.size());
        int ntg = threading->activate_global();
        vector<vector<vector<int>>> jh(ntg, vector<vector<int>>(2));
        vector<vector<vector<int>>> jket(ntg, vector<vector<int>>(2));
        vector<vector<vector<LL>>> ph(ntg, vector<vector<LL>>(2));
        vector<vector<vector<LL>>> pket(ntg, vector<vector<LL>>(2));
        vector<vector<vector<FL>>> hv(ntg, vector<vector<FL>>(2));
        map<S, vector<size_t>> dq_mats;
        for (size_t it = 0; it < mats.size(); it++)
            dq_mats[mats[it]->info->delta_quantum].push_back(it);
        for (auto &dqm : dq_mats) {
            const auto &rep_mat = mats[dqm.second[0]];
            for (int im = 0; im < rep_mat->info->n; im++) {
                S opdq = rep_mat->info->delta_quantum;
                S qbra = rep_mat->info->quanta[im].get_bra(opdq);
                S qket = rep_mat->info->quanta[im].get_ket();
                // SU2 and fermion factor for exchange:
                //   ket x op -> op x ket when is_right
                FL xf = (FL)1.0;
                if (T::value == ElemOpTypes::SU2 && is_right)
                    xf = (FL)(1 - ((opdq.twos() & qket.twos() & 1) << 1)) *
                         (FL)ElemMat<S, FL>::cg().phase(
                             opdq.twos(), qket.twos(), qbra.twos());
                int imb = drt->q_index(qbra), imk = drt->q_index(qket);
                assert(rep_mat->info->n_states_bra[im] == drt->xs[imb].back());
                assert(rep_mat->info->n_states_ket[im] == drt->xs[imk].back());
                vector<vector<vector<
                    vector<pair<pair<int16_t, int16_t>, pair<int16_t, FL>>>>>>
                    hm(drt->n_sites,
                       vector<vector<vector<
                           pair<pair<int16_t, int16_t>, pair<int16_t, FL>>>>>(
                           4));
                vector<vector<size_t>> max_d(drt->n_sites,
                                             vector<size_t>(4, 0));
                vector<int> kjis(drt->n_sites);
                for (int k = drt->n_sites - 1, ji = 0, jj; k >= 0;
                     k--, ji = jj) {
                    for (jj = ji; hdrt->qs[jj][0] == k + 1;)
                        jj++;
                    kjis[k] = ji;
                    for (int dbra = 0; dbra < 4; dbra++) {
                        hm[k][dbra].resize(jj - ji);
                        for (int jk = ji; jk < jj; jk++) {
                            for (int d = 0; d < hdrt->nd; d++)
                                if (hdrt->jds[jk * hdrt->nd + d] != 0)
                                    for (size_t md = 0;
                                         md <
                                         (int)site_matrices[k][d].data.size();
                                         md++)
                                        if (site_matrices[k][d]
                                                .indices[md]
                                                .first == dbra)
                                            hm[k][dbra][jk - ji].push_back(
                                                make_pair(
                                                    make_pair(
                                                        site_matrices[k][d].dq,
                                                        site_matrices[k][d]
                                                            .indices[md]
                                                            .second),
                                                    make_pair(
                                                        d, site_matrices[k][d]
                                                               .data[md])));
                            max_d[k][dbra] = max(max_d[k][dbra],
                                                 hm[k][dbra][jk - ji].size());
                        }
                    }
                }
                vector<vector<vector<MKL_INT>>> col_idxs(
                    dqm.second.size(),
                    vector<vector<MKL_INT>>(drt->xs[imb].back()));
                vector<vector<vector<FL>>> values(
                    dqm.second.size(), vector<vector<FL>>(drt->xs[imb].back()));
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
                for (int ibra = 0; ibra < (int)drt->xs[imb].back(); ibra++)
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
                for (LL ibra = 0; ibra < drt->xs[imb].back(); ibra++)
#endif
                {
                    const int tid = threading->get_thread_id();
                    int pi = 0, pj = pi ^ 1, jbra = imb;
                    vector<vector<int>> &xjh = jh[tid], &xjk = jket[tid];
                    vector<vector<LL>> &xph = ph[tid], &xpk = pket[tid];
                    vector<vector<FL>> &xhv = hv[tid];
                    xjh[pi].clear(), xph[pi].clear(), xjk[pi].clear();
                    xpk[pi].clear(), xhv[pi].clear();
                    for (int i = 0; i < hdrt->n_init_qs; i++) {
                        xjh[pi].push_back(i), xjk[pi].push_back(imk);
                        xph[pi].push_back(
                            i != 0 ? xph[pi].back() +
                                         hdrt->xs[(i - 1) * (hdrt->nd + 1) +
                                                  hdrt->nd]
                                   : 0);
                        xpk[pi].push_back(0), xhv[pi].push_back(xf);
                    }
                    LL pbra = ibra;
                    for (int k = drt->n_sites - 1; k >= 0;
                         k--, pi ^= 1, pj ^= 1) {
                        const int16_t dbra =
                            (int16_t)(upper_bound(drt->xs[jbra].begin(),
                                                  drt->xs[jbra].end(), pbra) -
                                      1 - drt->xs[jbra].begin());
                        pbra -= drt->xs[jbra][dbra];
                        const int jbv = drt->jds[jbra][dbra];
                        const size_t hsz = xhv[pi].size() * max_d[k][dbra];
                        xjh[pj].reserve(hsz), xjh[pj].clear();
                        xph[pj].reserve(hsz), xph[pj].clear();
                        xjk[pj].reserve(hsz), xjk[pj].clear();
                        xpk[pj].reserve(hsz), xpk[pj].clear();
                        xhv[pj].reserve(hsz), xhv[pj].clear();
                        for (size_t j = 0; j < xjh[pi].size(); j++)
                            for (const auto &md :
                                 hm[k][dbra][xjh[pi][j] - kjis[k]]) {
                                const int16_t d = md.second.first;
                                const int jhv =
                                    hdrt->jds[xjh[pi][j] * hdrt->nd + d];
                                const int16_t dket = md.first.second;
                                const int jkv = drt->jds[xjk[pi][j]][dket];
                                if (jkv == 0)
                                    continue;
                                const int16_t bfq = drt->abc[jbra][1];
                                const int16_t kfq = drt->abc[xjk[pi][j]][1];
                                const int16_t biq = drt->abc[jbv][1];
                                const int16_t kiq = drt->abc[jkv][1];
                                const int16_t mdq = md.first.first;
                                const int16_t mfq = hdrt->qs[xjh[pi][j]][2];
                                const int16_t miq = hdrt->qs[jhv][2];
                                const FL f =
                                    T::value == ElemOpTypes::SU2
                                        ? (*factors)[bfq * factor_strides[0] +
                                                     (biq - bfq + 1) *
                                                         factor_strides[1] +
                                                     kfq * factor_strides[2] +
                                                     (kiq - kfq + 1) *
                                                         factor_strides[3] +
                                                     mfq * factor_strides[4] +
                                                     miq * factor_strides[5] +
                                                     mdq * factor_strides[6]]
                                        : (FL)(1 -
                                               (((kiq & 1) & (mdq & 1)) << 1));
                                if (abs(f) < (FP)1E-14)
                                    continue;
                                xjk[pj].push_back(jkv);
                                xjh[pj].push_back(jhv);
                                xpk[pj].push_back(drt->xs[xjk[pi][j]][dket] +
                                                  xpk[pi][j]);
                                xph[pj].push_back(
                                    hdrt->xs[xjh[pi][j] * (hdrt->nd + 1) + d] +
                                    xph[pi][j]);
                                xhv[pj].push_back(f * xhv[pi][j] *
                                                  md.second.second);
                            }
                        jbra = jbv;
                    }
                    vector<LL> idxs;
                    idxs.reserve(xhv[pi].size());
                    for (LL i = 0; i < (LL)xhv[pi].size(); i++)
                        idxs.push_back(i);
                    sort(idxs.begin(), idxs.end(), [&xpk, pi](LL a, LL b) {
                        return xpk[pi][a] < xpk[pi][b];
                    });
                    LL xn = idxs.size() > 0;
                    for (LL i = 1; i < (LL)idxs.size(); i++)
                        xn += (xpk[pi][idxs[i]] != xpk[pi][idxs[i - 1]]);
                    for (size_t it = 0; it < dqm.second.size(); it++) {
                        col_idxs[it][ibra].reserve(xn);
                        values[it][ibra].reserve(xn);
                    }
                    for (size_t it = 0; it < dqm.second.size(); it++) {
                        for (LL i = 0; i < (LL)idxs.size(); i++)
                            if (i == 0 ||
                                (xpk[pi][idxs[i]] != xpk[pi][idxs[i - 1]] &&
                                 abs(values[it][ibra].back()) > cutoff)) {
                                col_idxs[it][ibra].push_back(
                                    (int)xpk[pi][idxs[i]]);
                                values[it][ibra].push_back(
                                    xhv[pi][idxs[i]] *
                                    (*ints[dqm.second[it]])[xph[pi][idxs[i]]]);
                            } else {
                                col_idxs[it][ibra].back() =
                                    (int)xpk[pi][idxs[i]];
                                values[it][ibra].back() +=
                                    xhv[pi][idxs[i]] *
                                    (*ints[dqm.second[it]])[xph[pi][idxs[i]]];
                            }
                        assert(col_idxs[it][ibra].size() <= xn &&
                               values[it][ibra].size() <= xn);
                    }
                }
                for (size_t it = 0; it < dqm.second.size(); it++) {
                    fill_csr_matrix(col_idxs[it], values[it],
                                    *mats[dqm.second[it]]->csr_data[im]);
                }
            }
        }
        threading->activate_normal();
    }
    vector<shared_ptr<GTensor<FL>>>
    build_npdm(const string &expr, const FL *bra_ci, const FL *ket_ci) const {
        int16_t op_twos = HDRT<S, T::value>::get_target_twos(expr);
        int16_t xc = (int16_t)(count(expr.begin(), expr.end(), 'c') +
                               count(expr.begin(), expr.end(), 'C'));
        int16_t xd = (int16_t)(count(expr.begin(), expr.end(), 'd') +
                               count(expr.begin(), expr.end(), 'D'));
        S iq(xc - xd, op_twos, 0); // assume ipg is the same
        vector<pair<S, pair<int16_t, int16_t>>> iop_qs;
        for (int16_t i = (int16_t)(xc + xd), j = min(i, (int16_t)1); j <= i;
             j++)
            iop_qs.push_back(make_pair(iq, make_pair(j, i)));
        shared_ptr<HDRT<S>> hdrt =
            make_shared<HDRT<S>>(n_orbs, iop_qs, drt->orb_sym);
        vector<shared_ptr<SpinPermScheme>> schemes =
            vector<shared_ptr<SpinPermScheme>>{make_shared<SpinPermScheme>(
                T::value == ElemOpTypes::SU2
                    ? SpinPermScheme::initialize_su2(xc + xd, expr, false, true)
                    : SpinPermScheme::initialize_sz(xc + xd, expr, true))};
        hdrt->initialize_steps(schemes);
        hdrt->initialize();
        print_hdrt_infos(set<S>{iq}, vector<string>{expr}, hdrt);
        shared_ptr<HDRTScheme<S, FL>> hdrt_scheme =
            make_shared<HDRTScheme<S, FL>>(hdrt, schemes);
        shared_ptr<vector<vector<LL>>> npdm_ord = hdrt_scheme->sort_npdm();
        vector<size_t> psum(npdm_ord->size() + 1, 0);
        for (size_t i = 0; i < npdm_ord->size(); i++)
            psum[i + 1] = psum[i] + (*npdm_ord)[i].size();
        shared_ptr<vector<FL>> r =
            make_shared<vector<FL>>(psum.back(), (FL)0.0);
        int ntg = threading->activate_global();
        vector<vector<vector<int>>> jbra(ntg, vector<vector<int>>(2));
        vector<vector<vector<int>>> jket(ntg, vector<vector<int>>(2));
        vector<vector<vector<pair<MKL_INT, MKL_INT>>>> pbk(
            ntg, vector<vector<pair<MKL_INT, MKL_INT>>>(2));
        vector<vector<vector<FL>>> hv(ntg, vector<vector<FL>>(2));
        vector<vector<ElemMat<S, FL>>> site_matrices = get_site_matrices(hdrt);
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int i = 0; i < (int)r->size(); i++)
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (size_t i = 0; i < r->size(); i++)
#endif
        {
            const size_t jx = upper_bound(psum.begin(), psum.end(), (size_t)i) -
                              psum.begin() - 1;
            const int tid = threading->get_thread_id();
            int pi = 0, pj = pi ^ 1;
            vector<vector<pair<MKL_INT, MKL_INT>>> &xpbk = pbk[tid];
            vector<vector<int>> &xjb = jbra[tid], &xjk = jket[tid];
            vector<vector<FL>> &xhv = hv[tid];
            xpbk[pi].clear(), xjb[pi].clear();
            xjk[pi].clear(), xhv[pi].clear();
            int jh = 0;
            LL ih = (*npdm_ord)[jx][i - psum[jx]];
            if (ih == -1)
                continue;
            for (; ih >= hdrt->xs[jh * (hdrt->nd + 1) + hdrt->nd]; jh++)
                ih -= hdrt->xs[jh * (hdrt->nd + 1) + hdrt->nd];
            for (int imb = 0; imb < drt->n_init_qs; imb++)
                for (int imk = 0; imk < drt->n_init_qs; imk++) {
                    xpbk[pi].push_back(make_pair(0, 0));
                    xjb[pi].push_back(imb), xjk[pi].push_back(imk);
                    xhv[pi].push_back((FL)1.0);
                }
            for (int k = drt->n_sites - 1; k >= 0; k--, pi ^= 1, pj ^= 1) {
                int16_t dh =
                    (int16_t)(upper_bound(
                                  hdrt->xs.begin() + jh * (hdrt->nd + 1),
                                  hdrt->xs.begin() + (jh + 1) * (hdrt->nd + 1),
                                  ih) -
                              1 - (hdrt->xs.begin() + jh * (hdrt->nd + 1)));
                const int jhv = hdrt->jds[jh * hdrt->nd + dh];
                const ElemMat<S, FL> &smat = site_matrices[k][dh];
                const size_t hsz = xhv[pi].size() * smat.data.size();
                xpbk[pj].reserve(hsz), xpbk[pj].clear();
                xjb[pj].reserve(hsz), xjb[pj].clear();
                xjk[pj].reserve(hsz), xjk[pj].clear();
                xhv[pj].reserve(hsz), xhv[pj].clear();
                for (size_t j = 0; j < xjk[pi].size(); j++)
                    for (size_t md = 0; md < smat.data.size(); md++) {
                        const int16_t dbra = smat.indices[md].first;
                        const int16_t dket = smat.indices[md].second;
                        const int jbv = drt->jds[xjb[pi][j]][dbra];
                        const int jkv = drt->jds[xjk[pi][j]][dket];
                        if (jbv == 0 || jkv == 0)
                            continue;
                        const int16_t bfq = drt->abc[xjb[pi][j]][1];
                        const int16_t kfq = drt->abc[xjk[pi][j]][1];
                        const int16_t biq = drt->abc[jbv][1];
                        const int16_t kiq = drt->abc[jkv][1];
                        const int16_t mdq = smat.dq;
                        const int16_t mfq = hdrt->qs[jh][2];
                        const int16_t miq = hdrt->qs[jhv][2];
                        const FL f =
                            T::value == ElemOpTypes::SU2
                                ? (*factors)[bfq * factor_strides[0] +
                                             (biq - bfq + 1) *
                                                 factor_strides[1] +
                                             kfq * factor_strides[2] +
                                             (kiq - kfq + 1) *
                                                 factor_strides[3] +
                                             mfq * factor_strides[4] +
                                             miq * factor_strides[5] +
                                             mdq * factor_strides[6]]
                                : (FL)(1 - (((kiq & 1) & (mdq & 1)) << 1));
                        if (abs(f) < (FP)1E-14)
                            continue;
                        xjb[pj].push_back(jbv);
                        xjk[pj].push_back(jkv);
                        xpbk[pj].push_back(
                            make_pair((MKL_INT)(drt->xs[xjb[pi][j]][dbra] +
                                                xpbk[pi][j].first),
                                      (MKL_INT)(drt->xs[xjk[pi][j]][dket] +
                                                xpbk[pi][j].second)));
                        xhv[pj].push_back(f * xhv[pi][j] * smat.data[md]);
                    }
                ih -= hdrt->xs[jh * (hdrt->nd + 1) + dh];
                jh = jhv;
            }
            FL rv = (FL)0.0;
            for (size_t j = 0; j < xpbk[pi].size(); j++)
                rv += bra_ci[xpbk[pi][j].first] * ket_ci[xpbk[pi][j].second] *
                      xhv[pi][j];
            (*r)[i] = rv;
        }
        threading->activate_normal();
        vector<shared_ptr<GTensor<FL>>> rr =
            sort_npdm(schemes, r, hdrt_scheme->expr_mp, psum);
        return rr;
    }
    virtual void build_normal_site_ops(
        OpNames op_name, int8_t iq, const set<S> &iqs,
        const vector<uint16_t> &idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const {}
    virtual void build_complementary_site_ops(
        OpNames op_name, int8_t iq, const set<S> &iqs,
        const vector<uint16_t> &idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const {}
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>>
            &ops) const override {
        shared_ptr<SparseMatrix<S, FL>> zero =
            make_shared<SparseMatrix<S, FL>>(nullptr);
        zero->factor = 0.0;
        map<pair<OpNames, int8_t>, set<S>> n_op_qs, c_op_qs;
        map<pair<OpNames, int8_t>, vector<uint16_t>> n_op_idxs, c_op_idxs;
        map<pair<OpNames, int8_t>, vector<shared_ptr<CSRSparseMatrix<S, FL>>>>
            n_op_mats, c_op_mats;
        if (iprint >= 1) {
            cout << endl
                 << "DRT Big Site :: NORBS = " << n_orbs << " "
                 << (is_right ? "Right" : "Left");
            cout << " ORB-SYM = [ ";
            for (auto &x : drt->orb_sym)
                cout << (int)x << " ";
            cout << "]" << endl;
        }
        for (auto &p : ops) {
            OpElement<S, FL> &op =
                *dynamic_pointer_cast<OpElement<S, FL>>(p.first);
            shared_ptr<VectorAllocator<FP>> d_alloc =
                make_shared<VectorAllocator<FP>>();
            shared_ptr<CSRSparseMatrix<S, FL>> mat =
                make_shared<CSRSparseMatrix<S, FL>>();
            shared_ptr<SparseMatrixInfo<S>> info =
                BigSite<S, FL>::find_site_op_info(op.q_label);
            // when big site is too small (no available ipg) this may happen
            if (info == nullptr) {
                p.second = zero;
                continue;
            }
            mat->initialize(info);
            for (int l = 0; l < mat->info->n; l++)
                mat->csr_data[l]->alloc = d_alloc;
            p.second = mat;
            int8_t s;
            switch (op.name) {
            case OpNames::I:
                n_op_qs[make_pair(op.name, -1)].insert(op.q_label);
                n_op_idxs[make_pair(op.name, -1)].push_back(0);
                n_op_mats[make_pair(op.name, -1)].push_back(mat);
                break;
            case OpNames::C:
            case OpNames::D:
                s = (int8_t)op.site_index.ss();
                n_op_qs[make_pair(op.name, s)].insert(op.q_label);
                n_op_idxs[make_pair(op.name, s)].push_back(
                    is_right ? n_total_orbs - 1 - op.site_index[0]
                             : op.site_index[0]);
                n_op_mats[make_pair(op.name, s)].push_back(mat);
                break;
            case OpNames::A:
            case OpNames::B:
            case OpNames::BD:
                s = (int8_t)op.site_index.ss();
                n_op_qs[make_pair(op.name, s)].insert(op.q_label);
                n_op_idxs[make_pair(op.name, s)].push_back(
                    is_right ? n_total_orbs - 1 - op.site_index[0]
                             : op.site_index[0]);
                n_op_idxs[make_pair(op.name, s)].push_back(
                    is_right ? n_total_orbs - 1 - op.site_index[1]
                             : op.site_index[1]);
                n_op_mats[make_pair(op.name, s)].push_back(mat);
                break;
            case OpNames::AD:
                // note that ad is defined as ad[i, j] = C[j] * C[i]
                s = (int8_t)op.site_index.ss();
                n_op_qs[make_pair(op.name, s)].insert(op.q_label);
                n_op_idxs[make_pair(op.name, s)].push_back(
                    is_right ? n_total_orbs - 1 - op.site_index[1]
                             : op.site_index[1]);
                n_op_idxs[make_pair(op.name, s)].push_back(
                    is_right ? n_total_orbs - 1 - op.site_index[0]
                             : op.site_index[0]);
                n_op_mats[make_pair(op.name, s)].push_back(mat);
                break;
            case OpNames::P:
            case OpNames::PD:
            case OpNames::Q:
                s = (int8_t)op.site_index.ss();
                c_op_qs[make_pair(op.name, s)].insert(op.q_label);
                c_op_idxs[make_pair(op.name, s)].push_back(op.site_index[0]);
                c_op_idxs[make_pair(op.name, s)].push_back(op.site_index[1]);
                c_op_mats[make_pair(op.name, s)].push_back(mat);
                break;
            case OpNames::R:
            case OpNames::RD:
                s = (int8_t)op.site_index.ss();
                c_op_qs[make_pair(op.name, s)].insert(op.q_label);
                c_op_idxs[make_pair(op.name, s)].push_back(op.site_index[0]);
                c_op_mats[make_pair(op.name, s)].push_back(mat);
                break;
            case OpNames::H:
                c_op_qs[make_pair(op.name, -1)].insert(op.q_label);
                c_op_idxs[make_pair(op.name, -1)].push_back(0);
                c_op_mats[make_pair(op.name, -1)].push_back(mat);
                break;
            default:
                assert(false);
                break;
            }
        }
        Timer _t, _t2;
        _t2.get_time();
        size_t size_all = 0, nnz_all = 0, size_total, nnz_total;
        for (const auto &m : n_op_qs) {
            check_signal_()();
            if (iprint >= 1 && n_op_mats.at(m.first).size() != 0) {
                cout << "  Build normal operator " << m.first.first;
                if (m.first.second != -1)
                    cout << (int)m.first.second;
                cout << ".. NOPS = " << n_op_mats.at(m.first).size() << " .."
                     << endl;
            }
            _t.get_time();
            build_normal_site_ops(m.first.first, m.first.second, m.second,
                                  n_op_idxs.at(m.first), n_op_mats.at(m.first));
            double top = _t.get_time();
            size_total = 0, nnz_total = 0;
            for (const auto &mat : n_op_mats.at(m.first))
                for (int i = 0; i < mat->info->n; i++) {
                    nnz_total += mat->csr_data[i]->nnz;
                    size_total += mat->csr_data[i]->size();
                }
            size_all += size_total, nnz_all += nnz_total;
            if (iprint >= 1 && n_op_mats.at(m.first).size() != 0) {
                cout << "    SIZE = " << setw(8)
                     << Parsing::to_size_string(size_total * sizeof(FL))
                     << " NNZ = " << setw(8)
                     << Parsing::to_size_string(nnz_total * sizeof(FL));
                cout << " SPT = " << fixed << setprecision(4) << setw(6)
                     << (double)(size_total - nnz_total) / size_total;
                cout << " T = " << fixed << setprecision(3) << setw(10) << top
                     << endl;
            }
        }
        for (const auto &m : c_op_qs) {
            check_signal_()();
            if (iprint >= 1 && c_op_mats.at(m.first).size() != 0) {
                cout << "  Build complementary operator " << m.first.first;
                if (m.first.second != -1)
                    cout << (int)m.first.second;
                cout << " .. NOPS = " << c_op_mats.at(m.first).size() << " .."
                     << endl;
            }
            _t.get_time();
            build_complementary_site_ops(m.first.first, m.first.second,
                                         m.second, c_op_idxs.at(m.first),
                                         c_op_mats.at(m.first));
            double top = _t.get_time();
            size_total = 0, nnz_total = 0;
            for (const auto &mat : c_op_mats.at(m.first))
                for (int i = 0; i < mat->info->n; i++) {
                    nnz_total += mat->csr_data[i]->nnz;
                    size_total += mat->csr_data[i]->size();
                }
            size_all += size_total, nnz_all += nnz_total;
            if (iprint >= 1 && c_op_mats.at(m.first).size() != 0) {
                cout << "    SIZE = " << setw(8)
                     << Parsing::to_size_string(size_total * sizeof(FL))
                     << " NNZ = " << setw(8)
                     << Parsing::to_size_string(nnz_total * sizeof(FL));
                cout << " SPT = " << fixed << setprecision(4) << setw(6)
                     << (double)(size_total - nnz_total) / size_total;
                cout << " T = " << fixed << setprecision(3) << setw(10) << top
                     << endl;
            }
        }
        double tall = _t2.get_time();
        if (iprint >= 1) {
            cout << "ALL SIZE = " << setw(8)
                 << Parsing::to_size_string(size_all * sizeof(FL))
                 << " NNZ = " << setw(8)
                 << Parsing::to_size_string(nnz_all * sizeof(FL));
            cout << " SPT = " << fixed << setprecision(4) << setw(6)
                 << (double)(size_all - nnz_all) / size_all;
            cout << " T = " << fixed << setprecision(3) << setw(10) << tall
                 << endl;
        }
    }
};

template <typename, typename, typename = void> struct DRTBigSite;

template <typename S, typename FL>
struct DRTBigSite<S, FL, typename S::is_su2_t> : DRTBigSiteBase<S, FL> {
    typedef integral_constant<ElemOpTypes, ElemT<S>::value> T;
    typedef typename GMatrix<FL>::FP FP;
    typedef long long LL;
    using DRTBigSiteBase<S, FL>::n_orbs;
    using DRTBigSiteBase<S, FL>::basis;
    using DRTBigSiteBase<S, FL>::op_infos;
    using DRTBigSiteBase<S, FL>::fcidump;
    using DRTBigSiteBase<S, FL>::drt;
    using DRTBigSiteBase<S, FL>::is_right;
    using DRTBigSiteBase<S, FL>::iprint;
    using DRTBigSiteBase<S, FL>::n_total_orbs;
    using DRTBigSiteBase<S, FL>::factors;
    using DRTBigSiteBase<S, FL>::factor_strides;
    using DRTBigSiteBase<S, FL>::print_hdrt_infos;
    using DRTBigSiteBase<S, FL>::sort_npdm;
    using DRTBigSiteBase<S, FL>::get_site_matrices;
    using DRTBigSiteBase<S, FL>::prepare_factors;
    using DRTBigSiteBase<S, FL>::build_operator_matrices;
    using DRTBigSiteBase<S, FL>::build_npdm_operator_matrices;
    using DRTBigSiteBase<S, FL>::cutoff;
    shared_ptr<GeneralFCIDUMP<FL>> gfd = nullptr;
    const static int max_n = 10, max_s = 10;
    DRTBigSite(const vector<S> &qs, bool is_right, int n_orbs,
               const vector<typename S::pg_t> &orb_sym,
               const shared_ptr<FCIDUMP<FL>> &fcidump = nullptr, int iprint = 0)
        : DRTBigSiteBase<S, FL>(qs, is_right, n_orbs, orb_sym, fcidump,
                                iprint) {
        op_infos = get_site_op_infos(orb_sym);
        prepare_factors();
    }
    virtual ~DRTBigSite() = default;
    static vector<S> get_target_quanta(bool is_right, int n_orbs,
                                       int n_max_elec,
                                       const vector<typename S::pg_t> &orb_sym,
                                       int nc_ref = 0) {
        S vacuum, target(S::invalid);
        vector<shared_ptr<StateInfo<S>>> site_basis(n_orbs);
        for (int m = 0; m < n_orbs; m++) {
            shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
            b->allocate(3);
            b->quanta[0] = vacuum;
            b->quanta[1] = S(1, 1, orb_sym[m]);
            b->quanta[2] = S(2, 0, 0);
            b->n_states[0] = b->n_states[1] = b->n_states[2] = 1;
            b->sort_states();
            site_basis[m] = b;
        }
        shared_ptr<StateInfo<S>> x = make_shared<StateInfo<S>>(vacuum);
        if (!is_right) {
            for (int i = 0; i < n_orbs; i++)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*x, *site_basis[i], target));
            int max_n = 0;
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > max_n)
                    max_n = x->quanta[q].n();
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() < max_n - n_max_elec ||
                    x->quanta[q].twos() > n_max_elec)
                    x->n_states[q] = 0;
        } else if (nc_ref == 0) {
            for (int i = n_orbs - 1; i >= 0; i--)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*site_basis[i], *x, target));
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > n_max_elec)
                    x->n_states[q] = 0;
        } else {
            shared_ptr<StateInfo<S>> y = make_shared<StateInfo<S>>(vacuum);
            for (int i = 0; i < nc_ref; i++)
                y = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*y, *site_basis[i], target));
            int max_n = 0;
            for (int q = 0; q < y->n; q++)
                if (y->quanta[q].n() > max_n)
                    max_n = y->quanta[q].n();
            for (int q = 0; q < y->n; q++)
                if (y->quanta[q].n() < max_n - n_max_elec ||
                    y->quanta[q].twos() > n_max_elec)
                    y->n_states[q] = 0;
            for (int i = n_orbs - 1; i >= nc_ref; i--)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*site_basis[i], *x, target));
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > n_max_elec)
                    x->n_states[q] = 0;
            x = make_shared<StateInfo<S>>(
                StateInfo<S>::tensor_product(*x, *y, target));
        }
        x->collect();
        return vector<S>(&x->quanta[0], &x->quanta[0] + x->n);
    }
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
    get_site_op_infos(const vector<uint8_t> &orb_sym) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        map<S, shared_ptr<SparseMatrixInfo<S>>> info;
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        info[S(0)] = nullptr;
        set<uint8_t> all_orb_sym(orb_sym.begin(), orb_sym.end());
        for (int i = 0; i < max_n; i++) {
            set<uint8_t> old_orb_sym = all_orb_sym;
            for (auto ipg : old_orb_sym) {
                if (i == 0)
                    all_orb_sym.insert(S::pg_inv(ipg));
                for (auto jpg : orb_sym) {
                    all_orb_sym.insert(S::pg_mul(ipg, jpg));
                    all_orb_sym.insert(S::pg_mul(ipg, S::pg_inv(jpg)));
                }
            }
        }
        for (auto ipg : all_orb_sym) {
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = 1; s <= max_s_odd; s += 2)
                    info[S(n, s, ipg)] = nullptr;
            for (int n = -max_n_even; n <= max_n_even; n += 2)
                for (int s = 0; s <= max_s_even; s += 2)
                    info[S(n, s, ipg)] = nullptr;
        }
        for (auto &p : info) {
            p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
            p.second->initialize(*basis, *basis, p.first, p.first.is_fermion());
        }
        return vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                info.end());
    }
    void build_normal_site_ops(
        OpNames op_name, int8_t iq, const set<S> &iqs,
        const vector<uint16_t> &idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const override {
        if (mats.size() == 0)
            return;
        const map<OpNames, vector<int16_t>> op_map =
            map<OpNames, vector<int16_t>>{{OpNames::I, vector<int16_t>{0}},
                                          {OpNames::C, vector<int16_t>{1}},
                                          {OpNames::D, vector<int16_t>{1}},
                                          {OpNames::A, vector<int16_t>{2}},
                                          {OpNames::AD, vector<int16_t>{2}},
                                          {OpNames::B, vector<int16_t>{2}},
                                          {OpNames::BD, vector<int16_t>{2}}};
        vector<pair<S, pair<int16_t, int16_t>>> iop_qs;
        for (auto &iqx : iqs)
            for (int16_t i : op_map.at(op_name))
                for (int16_t j = min(i, (int16_t)1); j <= i; j++)
                    iop_qs.push_back(make_pair(iqx, make_pair(j, i)));
        shared_ptr<HDRT<S>> hdrt =
            make_shared<HDRT<S>>(n_orbs, iop_qs, drt->orb_sym);
        vector<shared_ptr<SpinPermScheme>> schemes;
        vector<string> std_exprs;
        if (op_name == OpNames::I)
            std_exprs.push_back("");
        else if (op_name == OpNames::C)
            std_exprs.push_back("C");
        else if (op_name == OpNames::D)
            std_exprs.push_back("D");
        else if (op_name == OpNames::A && iq == 0)
            std_exprs.push_back("(C+C)0");
        else if (op_name == OpNames::A && iq != 0)
            std_exprs.push_back("(C+C)2");
        else if (op_name == OpNames::AD && iq == 0)
            std_exprs.push_back("(D+D)0");
        else if (op_name == OpNames::AD && iq != 0)
            std_exprs.push_back("(D+D)2");
        else if (op_name == OpNames::B && iq == 0)
            std_exprs.push_back("(C+D)0");
        else if (op_name == OpNames::B && iq != 0)
            std_exprs.push_back("(C+D)2");
        else if (op_name == OpNames::BD && iq == 0)
            std_exprs.push_back("(D+C)0");
        else if (op_name == OpNames::BD && iq != 0)
            std_exprs.push_back("(D+C)2");
        else
            throw runtime_error("Unsupported operator name!");
        schemes.reserve(std_exprs.size());
        for (size_t ix = 0; ix < std_exprs.size(); ix++)
            schemes.push_back(
                make_shared<SpinPermScheme>(SpinPermScheme::initialize_su2(
                    SpinPermRecoupling::count_cds(std_exprs[ix]), std_exprs[ix],
                    false, true)));
        hdrt->initialize_steps(schemes);
        hdrt->initialize();
        print_hdrt_infos(iqs, std_exprs, hdrt);
        shared_ptr<HDRTScheme<S, FL>> hdrt_scheme =
            make_shared<HDRTScheme<S, FL>>(hdrt, schemes);
        vector<pair<LL, FL>> mat_idxs(mats.size());
        shared_ptr<vector<vector<LL>>> npdm_ord = hdrt_scheme->sort_npdm();
        int nn = SpinPermRecoupling::count_cds(std_exprs[0]);
        map<vector<uint16_t>, int> idx_pattern_mp;
        vector<shared_ptr<NPDMCounter>> counters(
            schemes[0]->index_patterns.size());
        for (int i = 0; i < (int)schemes[0]->index_patterns.size(); i++) {
            idx_pattern_mp[schemes[0]->index_patterns[i]] = i;
            counters[i] = make_shared<NPDMCounter>(nn, hdrt->n_sites + 1);
        }
        const FL xf = (FL)(iq == -1 || iq == 0 || !is_right ? 1.0 : -1.0);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int i = 0; i < (int)mats.size(); i++) {
            vector<uint16_t> idx(idxs.begin() + (size_t)i * nn,
                                 idxs.begin() + (size_t)(i + 1) * nn);
            vector<uint16_t> idx_mat(nn), idx_idx(nn);
            for (int j = 0; j < nn; j++)
                idx_idx[j] = j;
            sort(idx_idx.begin(), idx_idx.begin() + nn,
                 [&idx](uint16_t a, uint16_t b) { return idx[a] < idx[b]; });
            if (nn >= 1)
                idx_mat[0] = 0;
            for (int j = 1; j < nn; j++)
                idx_mat[j] =
                    idx_mat[j - 1] + (idx[idx_idx[j]] != idx[idx_idx[j - 1]]);
            const int ii = idx_pattern_mp.at(idx_mat);
            const auto &xschs = schemes[0]->data[ii].at(idx_idx);
            const auto &counter = counters[ii];
            assert(xschs.size() == 1);
            const pair<double, string> &pds = xschs[0];
            const vector<LL> &xord =
                (*npdm_ord)[hdrt_scheme->expr_mp.at(pds.second).at(idx_mat)];
            for (int j = 0; j < nn; j++)
                idx_mat[j] = idx[idx_idx[j]];
            mat_idxs[i] =
                make_pair(xord[counter->find_left(hdrt->n_sites - 1, idx_mat)],
                          (FL)pds.first * xf);
        }
        threading->activate_normal();
        build_npdm_operator_matrices(hdrt, get_site_matrices(hdrt), mat_idxs,
                                     mats);
    }
    void build_complementary_site_ops(
        OpNames op_name, int8_t iq, const set<S> &iqs,
        const vector<uint16_t> &idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const override {
        if (mats.size() == 0)
            return;
        const map<OpNames, vector<int16_t>> op_map =
            map<OpNames, vector<int16_t>>{{OpNames::H, vector<int16_t>{2, 4}},
                                          {OpNames::R, vector<int16_t>{1, 3}},
                                          {OpNames::RD, vector<int16_t>{1, 3}},
                                          {OpNames::P, vector<int16_t>{2}},
                                          {OpNames::PD, vector<int16_t>{2}},
                                          {OpNames::Q, vector<int16_t>{2}}};
        vector<pair<S, pair<int16_t, int16_t>>> iop_qs;
        for (auto &iqx : iqs)
            for (int16_t i : op_map.at(op_name))
                for (int16_t j = 1; j <= i; j++)
                    iop_qs.push_back(make_pair(iqx, make_pair(j, i)));
        shared_ptr<HDRT<S>> hdrt =
            make_shared<HDRT<S>>(n_orbs, iop_qs, drt->orb_sym);
        vector<shared_ptr<SpinPermScheme>> schemes;
        vector<shared_ptr<GeneralFCIDUMP<FL>>> gfds;
        vector<string> std_exprs;
        if (this->gfd != nullptr)
            gfds.push_back(this->gfd), std_exprs = this->gfd->exprs;
        else if (op_name == OpNames::H) {
            shared_ptr<GeneralFCIDUMP<FL>> gfd =
                make_shared<GeneralFCIDUMP<FL>>(T::value);
            gfd->exprs.push_back("((C+(C+D)0)1+D)0");
            gfd->indices.push_back(vector<uint16_t>());
            gfd->data.push_back(vector<FL>());
            auto *idx = &gfd->indices.back();
            auto *dt = &gfd->data.back();
            array<uint16_t, 4> arr;
            for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                for (arr[1] = 0; arr[1] < n_orbs; arr[1]++)
                    for (arr[2] = 0; arr[2] < n_orbs; arr[2]++)
                        for (arr[3] = 0; arr[3] < n_orbs; arr[3]++) {
                            const FL v =
                                is_right ? fcidump->v(n_total_orbs - 1 - arr[0],
                                                      n_total_orbs - 1 - arr[3],
                                                      n_total_orbs - 1 - arr[1],
                                                      n_total_orbs - 1 - arr[2])
                                         : fcidump->v(arr[0], arr[3], arr[1],
                                                      arr[2]);
                            if (abs(v) > cutoff) {
                                idx->insert(idx->end(), arr.begin(), arr.end());
                                dt->push_back(v);
                            }
                        }
            gfd->exprs.push_back("(C+D)0");
            gfd->indices.push_back(vector<uint16_t>());
            gfd->data.push_back(vector<FL>());
            idx = &gfd->indices.back(), dt = &gfd->data.back();
            for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                    const FL v = is_right
                                     ? fcidump->t(n_total_orbs - 1 - arr[0],
                                                  n_total_orbs - 1 - arr[1])
                                     : fcidump->t(arr[0], arr[1]);
                    if (abs(v) > cutoff) {
                        idx->insert(idx->end(), arr.begin(), arr.begin() + 2);
                        dt->push_back((FL)sqrtl(2) * v);
                    }
                }
            std_exprs = gfd->exprs;
            gfds.push_back(gfd->adjust_order(schemes, true, true));
        } else if (op_name == OpNames::R || op_name == OpNames::RD) {
            for (uint16_t ix : idxs) {
                shared_ptr<GeneralFCIDUMP<FL>> gfd =
                    make_shared<GeneralFCIDUMP<FL>>(T::value);
                gfd->exprs.push_back(op_name == OpNames::R ? "((C+D)0+D)1"
                                                           : "(C+(C+D)0)1");
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 3> arr;
                //  R: arr = k l j : v(i, j, k, l) * ((Ck+Dl)0+Dj)1
                // RD: arr = i k l : v(i, j, k, l) * (Ci+(Ck+Dl)0)1
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                    for (arr[1] = 0; arr[1] < n_orbs; arr[1]++)
                        for (arr[2] = 0; arr[2] < n_orbs; arr[2]++) {
                            const FL v =
                                op_name == OpNames::R
                                    ? (is_right
                                           ? fcidump->v(
                                                 ix, n_total_orbs - 1 - arr[2],
                                                 n_total_orbs - 1 - arr[0],
                                                 n_total_orbs - 1 - arr[1])
                                           : fcidump->v(ix, arr[2], arr[0],
                                                        arr[1]))
                                    : (is_right
                                           ? fcidump->v(
                                                 n_total_orbs - 1 - arr[0], ix,
                                                 n_total_orbs - 1 - arr[1],
                                                 n_total_orbs - 1 - arr[2])
                                           : fcidump->v(arr[0], ix, arr[1],
                                                        arr[2]));
                            if (abs(v) > cutoff) {
                                idx->insert(idx->end(), arr.begin(), arr.end());
                                dt->push_back(v);
                            }
                        }
                gfd->exprs.push_back(op_name == OpNames::R ? "D" : "C");
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                idx = &gfd->indices.back(), dt = &gfd->data.back();
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++) {
                    const FL v = is_right
                                     ? fcidump->t(ix, n_total_orbs - 1 - arr[0])
                                     : fcidump->t(ix, arr[0]);
                    if (abs(v) > cutoff) {
                        idx->push_back(arr[0]);
                        dt->push_back((FL)(sqrtl(2) / 4.0) * v);
                    }
                }
                std_exprs = gfd->exprs;
                gfds.push_back(gfd->adjust_order(schemes, true, true));
            }
        } else if (op_name == OpNames::P || op_name == OpNames::PD) {
            for (int ixx = 0; ixx < (int)idxs.size(); ixx += 2) {
                const uint16_t ix0 = idxs[ixx], ix1 = idxs[ixx + 1];
                shared_ptr<GeneralFCIDUMP<FL>> gfd =
                    make_shared<GeneralFCIDUMP<FL>>(T::value);
                gfd->exprs.push_back(op_name == OpNames::P
                                         ? (iq == 0 ? "(D+D)0" : "(D+D)2")
                                         : (iq == 0 ? "(C+C)0" : "(C+C)2"));
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 2> arr;
                //  P: arr = j l : v(i, j, k, l) * (Dj+Dl)0
                // PD: arr = l j : v(j, i, l, k) * (Cl+Cj)0
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                    for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                        const FL v =
                            op_name == OpNames::P
                                ? (is_right
                                       ? (FL)(iq == 0 ? 1.0 : -1.0) *
                                             fcidump->v(
                                                 ix0, n_total_orbs - 1 - arr[0],
                                                 ix1, n_total_orbs - 1 - arr[1])
                                       : fcidump->v(ix0, arr[0], ix1, arr[1]))
                                : (is_right
                                       ? (FL)(iq == 0 ? 1.0 : -1.0) *
                                             fcidump->v(
                                                 n_total_orbs - 1 - arr[1], ix0,
                                                 n_total_orbs - 1 - arr[0], ix1)
                                       : fcidump->v(arr[1], ix0, arr[0], ix1));
                        if (abs(v) > cutoff) {
                            idx->insert(idx->end(), arr.begin(), arr.end());
                            dt->push_back(v);
                        }
                    }
                std_exprs = gfd->exprs;
                gfds.push_back(gfd->adjust_order(schemes, true, true));
            }
        } else if (op_name == OpNames::Q) {
            for (int ixx = 0; ixx < (int)idxs.size(); ixx += 2) {
                const uint16_t ix0 = idxs[ixx], ix1 = idxs[ixx + 1];
                shared_ptr<GeneralFCIDUMP<FL>> gfd =
                    make_shared<GeneralFCIDUMP<FL>>(T::value);
                gfd->exprs.push_back(iq == 0 ? "(C+D)0" : "(C+D)2");
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 2> arr;
                // Q0: arr = k l : [ 2 v(i,j,k,l) - v(i,l,k,j) ] * (Ck+Dl)0
                // Q2: arr = k l : v(i,l,k,j) * (Ck+Dl)2
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                    for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                        const FL v =
                            iq == 0
                                ? (is_right
                                       ? (FL)2.0 * fcidump->v(ix0, ix1,
                                                              n_total_orbs - 1 -
                                                                  arr[0],
                                                              n_total_orbs - 1 -
                                                                  arr[1]) -
                                             fcidump->v(
                                                 ix0, n_total_orbs - 1 - arr[1],
                                                 n_total_orbs - 1 - arr[0], ix1)
                                       : (FL)2.0 * fcidump->v(ix0, ix1, arr[0],
                                                              arr[1]) -
                                             fcidump->v(ix0, arr[1], arr[0],
                                                        ix1))
                                : (is_right
                                       ? (FL)(-1.0) *
                                             fcidump->v(
                                                 ix0, n_total_orbs - 1 - arr[1],
                                                 n_total_orbs - 1 - arr[0], ix1)
                                       : fcidump->v(ix0, arr[1], arr[0], ix1));
                        if (abs(v) > cutoff) {
                            idx->insert(idx->end(), arr.begin(), arr.end());
                            dt->push_back(v);
                        }
                    }
                std_exprs = gfd->exprs;
                gfds.push_back(gfd->adjust_order(schemes, true, true));
            }
        } else
            throw runtime_error("Unsupported operator name!");
        schemes.reserve(std_exprs.size());
        for (size_t ix = 0; ix < std_exprs.size(); ix++)
            schemes.push_back(
                make_shared<SpinPermScheme>(SpinPermScheme::initialize_su2(
                    SpinPermRecoupling::count_cds(std_exprs[ix]), std_exprs[ix],
                    false, true)));
        hdrt->initialize_steps(schemes);
        hdrt->initialize();
        print_hdrt_infos(iqs, std_exprs, hdrt);
        shared_ptr<HDRTScheme<S, FL>> hdrt_scheme =
            make_shared<HDRTScheme<S, FL>>(hdrt, schemes);
        vector<shared_ptr<vector<FL>>> ints(gfds.size());
        for (size_t i = 0; i < gfds.size(); i++)
            ints[i] = hdrt_scheme->sort_integral(gfds[i]);
        build_operator_matrices(hdrt, get_site_matrices(hdrt), ints, mats);
    }
};

template <typename S, typename FL>
struct DRTBigSite<S, FL, typename S::is_sz_t> : DRTBigSiteBase<S, FL> {
    typedef integral_constant<ElemOpTypes, ElemT<S>::value> T;
    typedef typename GMatrix<FL>::FP FP;
    typedef long long LL;
    using DRTBigSiteBase<S, FL>::n_orbs;
    using DRTBigSiteBase<S, FL>::basis;
    using DRTBigSiteBase<S, FL>::op_infos;
    using DRTBigSiteBase<S, FL>::fcidump;
    using DRTBigSiteBase<S, FL>::drt;
    using DRTBigSiteBase<S, FL>::is_right;
    using DRTBigSiteBase<S, FL>::iprint;
    using DRTBigSiteBase<S, FL>::n_total_orbs;
    using DRTBigSiteBase<S, FL>::print_hdrt_infos;
    using DRTBigSiteBase<S, FL>::sort_npdm;
    using DRTBigSiteBase<S, FL>::get_site_matrices;
    using DRTBigSiteBase<S, FL>::build_operator_matrices;
    using DRTBigSiteBase<S, FL>::build_npdm_operator_matrices;
    using DRTBigSiteBase<S, FL>::cutoff;
    shared_ptr<GeneralFCIDUMP<FL>> gfd = nullptr;
    const static int max_n = 10, max_s = 10;
    DRTBigSite(const vector<S> &qs, bool is_right, int n_orbs,
               const vector<typename S::pg_t> &orb_sym,
               const shared_ptr<FCIDUMP<FL>> &fcidump = nullptr, int iprint = 0)
        : DRTBigSiteBase<S, FL>(qs, is_right, n_orbs, orb_sym, fcidump,
                                iprint) {
        op_infos = get_site_op_infos(orb_sym);
    }
    virtual ~DRTBigSite() = default;
    static vector<S> get_target_quanta(bool is_right, int n_orbs,
                                       int n_max_elec,
                                       const vector<typename S::pg_t> &orb_sym,
                                       int nc_ref = 0) {
        S vacuum, target(S::invalid);
        vector<shared_ptr<StateInfo<S>>> site_basis(n_orbs);
        for (int m = 0; m < n_orbs; m++) {
            shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
            b->allocate(4);
            b->quanta[0] = vacuum;
            b->quanta[1] = S(1, 1, orb_sym[m]);
            b->quanta[2] = S(1, -1, orb_sym[m]);
            b->quanta[3] = S(2, 0, 0);
            b->n_states[0] = b->n_states[1] = b->n_states[2] = b->n_states[3] =
                1;
            b->sort_states();
            site_basis[m] = b;
        }
        shared_ptr<StateInfo<S>> x = make_shared<StateInfo<S>>(vacuum);
        if (!is_right) {
            for (int i = 0; i < n_orbs; i++)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*x, *site_basis[i], target));
            int max_n = 0;
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > max_n)
                    max_n = x->quanta[q].n();
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() < max_n - n_max_elec ||
                    abs(x->quanta[q].twos()) > n_max_elec)
                    x->n_states[q] = 0;
        } else if (nc_ref == 0) {
            for (int i = n_orbs - 1; i >= 0; i--)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*site_basis[i], *x, target));
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > n_max_elec)
                    x->n_states[q] = 0;
        } else {
            shared_ptr<StateInfo<S>> y = make_shared<StateInfo<S>>(vacuum);
            for (int i = 0; i < nc_ref; i++)
                y = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*y, *site_basis[i], target));
            int max_n = 0;
            for (int q = 0; q < y->n; q++)
                if (y->quanta[q].n() > max_n)
                    max_n = y->quanta[q].n();
            for (int q = 0; q < y->n; q++)
                if (y->quanta[q].n() < max_n - n_max_elec ||
                    abs(y->quanta[q].twos()) > n_max_elec)
                    y->n_states[q] = 0;
            for (int i = n_orbs - 1; i >= nc_ref; i--)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*site_basis[i], *x, target));
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > n_max_elec)
                    x->n_states[q] = 0;
            x = make_shared<StateInfo<S>>(
                StateInfo<S>::tensor_product(*x, *y, target));
        }
        x->collect();
        return vector<S>(&x->quanta[0], &x->quanta[0] + x->n);
    }
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
    get_site_op_infos(const vector<uint8_t> &orb_sym) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        map<S, shared_ptr<SparseMatrixInfo<S>>> info;
        const int max_n_odd = max_n | 1, max_s_odd = max_s | 1;
        const int max_n_even = max_n_odd ^ 1, max_s_even = max_s_odd ^ 1;
        info[S(0)] = nullptr;
        set<uint8_t> all_orb_sym(orb_sym.begin(), orb_sym.end());
        for (int i = 0; i < max_n; i++) {
            set<uint8_t> old_orb_sym = all_orb_sym;
            for (auto ipg : old_orb_sym) {
                if (i == 0)
                    all_orb_sym.insert(S::pg_inv(ipg));
                for (auto jpg : orb_sym) {
                    all_orb_sym.insert(S::pg_mul(ipg, jpg));
                    all_orb_sym.insert(S::pg_mul(ipg, S::pg_inv(jpg)));
                }
            }
        }
        for (auto ipg : all_orb_sym) {
            for (int n = -max_n_odd; n <= max_n_odd; n += 2)
                for (int s = -max_s_odd; s <= max_s_odd; s += 2)
                    info[S(n, s, ipg)] = nullptr;
            for (int n = -max_n_even; n <= max_n_even; n += 2)
                for (int s = -max_s_even; s <= max_s_even; s += 2)
                    info[S(n, s, ipg)] = nullptr;
        }
        for (auto &p : info) {
            p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
            p.second->initialize(*basis, *basis, p.first, p.first.is_fermion());
        }
        return vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                info.end());
    }
    void build_normal_site_ops(
        OpNames op_name, int8_t iq, const set<S> &iqs,
        const vector<uint16_t> &idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const override {
        if (mats.size() == 0)
            return;
        const map<OpNames, vector<int16_t>> op_map =
            map<OpNames, vector<int16_t>>{{OpNames::I, vector<int16_t>{0}},
                                          {OpNames::C, vector<int16_t>{1}},
                                          {OpNames::D, vector<int16_t>{1}},
                                          {OpNames::A, vector<int16_t>{2}},
                                          {OpNames::AD, vector<int16_t>{2}},
                                          {OpNames::B, vector<int16_t>{2}},
                                          {OpNames::BD, vector<int16_t>{2}}};
        vector<pair<S, pair<int16_t, int16_t>>> iop_qs;
        for (auto &iqx : iqs)
            for (int16_t i : op_map.at(op_name))
                for (int16_t j = min(i, (int16_t)1); j <= i; j++)
                    iop_qs.push_back(make_pair(iqx, make_pair(j, i)));
        shared_ptr<HDRT<S>> hdrt =
            make_shared<HDRT<S>>(n_orbs, iop_qs, drt->orb_sym);
        vector<shared_ptr<SpinPermScheme>> schemes;
        vector<string> std_exprs;
        if (op_name == OpNames::I)
            std_exprs.push_back("");
        else if (op_name == OpNames::C)
            std_exprs.push_back(iq ? "C" : "c");
        else if (op_name == OpNames::D)
            std_exprs.push_back(iq ? "D" : "d");
        else if (op_name == OpNames::A)
            std_exprs.push_back(string((iq & 1) ? "C" : "c") +
                                string((iq >> 1) ? "C" : "c"));
        else if (op_name == OpNames::AD)
            std_exprs.push_back(string((iq >> 1) ? "D" : "d") +
                                string((iq & 1) ? "D" : "d"));
        else if (op_name == OpNames::B)
            std_exprs.push_back(string((iq & 1) ? "C" : "c") +
                                string((iq >> 1) ? "D" : "d"));
        else if (op_name == OpNames::BD)
            std_exprs.push_back(string((iq & 1) ? "D" : "d") +
                                string((iq >> 1) ? "C" : "c"));
        else
            throw runtime_error("Unsupported operator name!");
        schemes.reserve(std_exprs.size());
        for (size_t ix = 0; ix < std_exprs.size(); ix++)
            schemes.push_back(
                make_shared<SpinPermScheme>(SpinPermScheme::initialize_sz(
                    SpinPermRecoupling::count_cds(std_exprs[ix]), std_exprs[ix],
                    true)));
        hdrt->initialize_steps(schemes);
        hdrt->initialize();
        print_hdrt_infos(iqs, std_exprs, hdrt);
        shared_ptr<HDRTScheme<S, FL>> hdrt_scheme =
            make_shared<HDRTScheme<S, FL>>(hdrt, schemes);
        vector<pair<LL, FL>> mat_idxs(mats.size());
        shared_ptr<vector<vector<LL>>> npdm_ord = hdrt_scheme->sort_npdm();
        int nn = SpinPermRecoupling::count_cds(std_exprs[0]);
        map<vector<uint16_t>, int> idx_pattern_mp;
        vector<shared_ptr<NPDMCounter>> counters(
            schemes[0]->index_patterns.size());
        for (int i = 0; i < (int)schemes[0]->index_patterns.size(); i++) {
            idx_pattern_mp[schemes[0]->index_patterns[i]] = i;
            counters[i] = make_shared<NPDMCounter>(nn, hdrt->n_sites + 1);
        }
        const FL xf = (FL)1.0;
        // const FL xf = (FL)(iq != 2 || !is_right ? 1.0 : -1.0);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int i = 0; i < (int)mats.size(); i++) {
            vector<uint16_t> idx(idxs.begin() + (size_t)i * nn,
                                 idxs.begin() + (size_t)(i + 1) * nn);
            vector<uint16_t> idx_mat(nn), idx_idx(nn);
            for (int j = 0; j < nn; j++)
                idx_idx[j] = j;
            sort(idx_idx.begin(), idx_idx.begin() + nn,
                 [&idx](uint16_t a, uint16_t b) { return idx[a] < idx[b]; });
            if (nn >= 1)
                idx_mat[0] = 0;
            for (int j = 1; j < nn; j++)
                idx_mat[j] =
                    idx_mat[j - 1] + (idx[idx_idx[j]] != idx[idx_idx[j - 1]]);
            const int ii = idx_pattern_mp.at(idx_mat);
            const auto &xschs = schemes[0]->data[ii].at(idx_idx);
            const auto &counter = counters[ii];
            assert(xschs.size() == 1);
            const pair<double, string> &pds = xschs[0];
            const vector<LL> &xord =
                (*npdm_ord)[hdrt_scheme->expr_mp.at(pds.second).at(idx_mat)];
            for (int j = 0; j < nn; j++)
                idx_mat[j] = idx[idx_idx[j]];
            mat_idxs[i] =
                make_pair(xord[counter->find_left(hdrt->n_sites - 1, idx_mat)],
                          (FL)pds.first * xf);
        }
        threading->activate_normal();
        build_npdm_operator_matrices(hdrt, get_site_matrices(hdrt), mat_idxs,
                                     mats);
    }
    void build_complementary_site_ops(
        OpNames op_name, int8_t iq, const set<S> &iqs,
        const vector<uint16_t> &idxs,
        const vector<shared_ptr<CSRSparseMatrix<S, FL>>> &mats) const override {
        if (mats.size() == 0)
            return;
        const map<OpNames, vector<int16_t>> op_map =
            map<OpNames, vector<int16_t>>{{OpNames::H, vector<int16_t>{2, 4}},
                                          {OpNames::R, vector<int16_t>{1, 3}},
                                          {OpNames::RD, vector<int16_t>{1, 3}},
                                          {OpNames::P, vector<int16_t>{2}},
                                          {OpNames::PD, vector<int16_t>{2}},
                                          {OpNames::Q, vector<int16_t>{2}}};
        vector<pair<S, pair<int16_t, int16_t>>> iop_qs;
        for (auto &iqx : iqs)
            for (int16_t i : op_map.at(op_name))
                for (int16_t j = 1; j <= i; j++)
                    iop_qs.push_back(make_pair(iqx, make_pair(j, i)));
        shared_ptr<HDRT<S>> hdrt =
            make_shared<HDRT<S>>(n_orbs, iop_qs, drt->orb_sym);
        vector<shared_ptr<SpinPermScheme>> schemes;
        vector<shared_ptr<GeneralFCIDUMP<FL>>> gfds;
        vector<string> std_exprs;
        if (this->gfd != nullptr)
            gfds.push_back(this->gfd), std_exprs = this->gfd->exprs;
        else if (op_name == OpNames::H) {
            shared_ptr<GeneralFCIDUMP<FL>> gfd =
                make_shared<GeneralFCIDUMP<FL>>(T::value);
            gfd->exprs.push_back("ccdd");
            gfd->exprs.push_back("cCDd");
            gfd->exprs.push_back("CcdD");
            gfd->exprs.push_back("CCDD");
            for (uint8_t si = 0; si < 2; si++)
                for (uint8_t sj = 0; sj < 2; sj++) {
                    gfd->indices.push_back(vector<uint16_t>());
                    gfd->data.push_back(vector<FL>());
                    auto *idx = &gfd->indices.back();
                    auto *dt = &gfd->data.back();
                    array<uint16_t, 4> arr;
                    for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                        for (arr[1] = 0; arr[1] < n_orbs; arr[1]++)
                            for (arr[2] = 0; arr[2] < n_orbs; arr[2]++)
                                for (arr[3] = 0; arr[3] < n_orbs; arr[3]++) {
                                    const FL v =
                                        is_right
                                            ? fcidump->v(
                                                  si, sj,
                                                  n_total_orbs - 1 - arr[0],
                                                  n_total_orbs - 1 - arr[3],
                                                  n_total_orbs - 1 - arr[1],
                                                  n_total_orbs - 1 - arr[2])
                                            : fcidump->v(si, sj, arr[0], arr[3],
                                                         arr[1], arr[2]);
                                    if (abs(v) > cutoff) {
                                        idx->insert(idx->end(), arr.begin(),
                                                    arr.end());
                                        dt->push_back((FL)0.5 * v);
                                    }
                                }
                }
            gfd->exprs.push_back("cd");
            gfd->exprs.push_back("CD");
            for (uint8_t si = 0; si < 2; si++) {
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 2> arr;
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                    for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                        const FL v =
                            is_right ? fcidump->t(si, n_total_orbs - 1 - arr[0],
                                                  n_total_orbs - 1 - arr[1])
                                     : fcidump->t(si, arr[0], arr[1]);
                        if (abs(v) > cutoff) {
                            idx->insert(idx->end(), arr.begin(),
                                        arr.begin() + 2);
                            dt->push_back(v);
                        }
                    }
            }
            std_exprs = gfd->exprs;
            gfds.push_back(gfd->adjust_order(schemes, true, true));
        } else if (op_name == OpNames::R || op_name == OpNames::RD) {
            for (uint16_t ix : idxs) {
                shared_ptr<GeneralFCIDUMP<FL>> gfd =
                    make_shared<GeneralFCIDUMP<FL>>(T::value);
                if (iq == 0) {
                    gfd->exprs.push_back(op_name == OpNames::R ? "cdd" : "ccd");
                    gfd->exprs.push_back(op_name == OpNames::R ? "CDd" : "cCD");
                } else {
                    gfd->exprs.push_back(op_name == OpNames::R ? "cdD" : "Ccd");
                    gfd->exprs.push_back(op_name == OpNames::R ? "CDD" : "CCD");
                }
                //  R: arr = k l j : v(i, j, k, l) * Ck Dl Dj
                // RD: arr = i k l : v(i, j, k, l) * Ci Ck Dl
                for (uint8_t si = 0; si < 2; si++) {
                    gfd->indices.push_back(vector<uint16_t>());
                    gfd->data.push_back(vector<FL>());
                    auto *idx = &gfd->indices.back();
                    auto *dt = &gfd->data.back();
                    array<uint16_t, 3> arr;
                    for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                        for (arr[1] = 0; arr[1] < n_orbs; arr[1]++)
                            for (arr[2] = 0; arr[2] < n_orbs; arr[2]++) {
                                const FL v =
                                    op_name == OpNames::R
                                        ? (is_right
                                               ? fcidump->v(
                                                     iq, si, ix,
                                                     n_total_orbs - 1 - arr[2],
                                                     n_total_orbs - 1 - arr[0],
                                                     n_total_orbs - 1 - arr[1])
                                               : fcidump->v(iq, si, ix, arr[2],
                                                            arr[0], arr[1]))
                                        : (is_right
                                               ? fcidump->v(
                                                     iq, si,
                                                     n_total_orbs - 1 - arr[0],
                                                     ix,
                                                     n_total_orbs - 1 - arr[1],
                                                     n_total_orbs - 1 - arr[2])
                                               : fcidump->v(iq, si, arr[0], ix,
                                                            arr[1], arr[2]));
                                if (abs(v) > cutoff) {
                                    idx->insert(idx->end(), arr.begin(),
                                                arr.end());
                                    dt->push_back(v);
                                }
                            }
                }
                gfd->exprs.push_back(iq ? (op_name == OpNames::R ? "D" : "C")
                                        : (op_name == OpNames::R ? "d" : "c"));
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 1> arr;
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++) {
                    const FL v =
                        is_right ? fcidump->t(iq, ix, n_total_orbs - 1 - arr[0])
                                 : fcidump->t(iq, ix, arr[0]);
                    if (abs(v) > cutoff) {
                        idx->push_back(arr[0]);
                        dt->push_back((FL)0.5 * v);
                    }
                }
                std_exprs = gfd->exprs;
                gfds.push_back(gfd->adjust_order(schemes, true, true));
            }
        } else if (op_name == OpNames::P || op_name == OpNames::PD) {
            for (int ixx = 0; ixx < (int)idxs.size(); ixx += 2) {
                const uint16_t ix0 = idxs[ixx], ix1 = idxs[ixx + 1];
                shared_ptr<GeneralFCIDUMP<FL>> gfd =
                    make_shared<GeneralFCIDUMP<FL>>(T::value);
                gfd->exprs.push_back(op_name == OpNames::P
                                         ? string((iq & 1) ? "D" : "d") +
                                               string((iq >> 1) ? "D" : "d")
                                         : string((iq >> 1) ? "C" : "c") +
                                               string((iq & 1) ? "C" : "c"));
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 2> arr;
                //  P: arr = l j : v(i, j, k, l) * Dl Dj
                // PD: arr = j l : v(j, i, l, k) * Cj Cl
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                    for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                        const FL v =
                            op_name == OpNames::P
                                ? (is_right
                                       ? -fcidump->v(iq & 1, iq >> 1, ix0,
                                                     n_total_orbs - 1 - arr[0],
                                                     ix1,
                                                     n_total_orbs - 1 - arr[1])
                                       : -fcidump->v(iq & 1, iq >> 1, ix0,
                                                     arr[0], ix1, arr[1]))
                                : (is_right
                                       ? -fcidump->v(
                                             iq & 1, iq >> 1,
                                             n_total_orbs - 1 - arr[1], ix0,
                                             n_total_orbs - 1 - arr[0], ix1)
                                       : -fcidump->v(iq & 1, iq >> 1, arr[1],
                                                     ix0, arr[0], ix1));
                        if (abs(v) > cutoff) {
                            idx->insert(idx->end(), arr.begin(), arr.end());
                            dt->push_back(v);
                        }
                    }
                std_exprs = gfd->exprs;
                gfds.push_back(gfd->adjust_order(schemes, true, true));
            }
        } else if (op_name == OpNames::Q) {
            for (int ixx = 0; ixx < (int)idxs.size(); ixx += 2) {
                const uint16_t ix0 = idxs[ixx], ix1 = idxs[ixx + 1];
                shared_ptr<GeneralFCIDUMP<FL>> gfd =
                    make_shared<GeneralFCIDUMP<FL>>(T::value);
                gfd->exprs.push_back(string((iq >> 1) ? "C" : "c") +
                                     string((iq & 1) ? "D" : "d"));
                gfd->indices.push_back(vector<uint16_t>());
                gfd->data.push_back(vector<FL>());
                auto *idx = &gfd->indices.back();
                auto *dt = &gfd->data.back();
                array<uint16_t, 2> arr;
                // Q12: arr  = k l : -v(i,l,k,j) * Ck Dl
                // Q03: arr += k l : +v(i,j,k,l) * Ck Dl
                for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                    for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                        const FL v =
                            is_right
                                ? -fcidump->v(iq & 1, iq >> 1, ix0,
                                              n_total_orbs - 1 - arr[1],
                                              n_total_orbs - 1 - arr[0], ix1)
                                : -fcidump->v(iq & 1, iq >> 1, ix0, arr[1],
                                              arr[0], ix1);
                        if (abs(v) > cutoff) {
                            idx->insert(idx->end(), arr.begin(), arr.end());
                            dt->push_back(v);
                        }
                    }
                if (iq == 0 || iq == 3)
                    for (uint8_t si = 0; si < 2; si++) {
                        gfd->exprs.push_back(si == 0 ? "cd" : "CD");
                        gfd->indices.push_back(vector<uint16_t>());
                        gfd->data.push_back(vector<FL>());
                        idx = &gfd->indices.back(), dt = &gfd->data.back();
                        for (arr[0] = 0; arr[0] < n_orbs; arr[0]++)
                            for (arr[1] = 0; arr[1] < n_orbs; arr[1]++) {
                                const FL v =
                                    is_right
                                        ? fcidump->v(iq & 1, si, ix0, ix1,
                                                     n_total_orbs - 1 - arr[0],
                                                     n_total_orbs - 1 - arr[1])
                                        : fcidump->v(iq & 1, si, ix0, ix1,
                                                     arr[0], arr[1]);
                                if (abs(v) > cutoff) {
                                    idx->insert(idx->end(), arr.begin(),
                                                arr.end());
                                    dt->push_back(v);
                                }
                            }
                    }
                std_exprs = gfd->exprs;
                gfds.push_back(gfd->adjust_order(schemes, true, true));
            }
        } else
            throw runtime_error("Unsupported operator name!");
        schemes.reserve(std_exprs.size());
        for (size_t ix = 0; ix < std_exprs.size(); ix++)
            schemes.push_back(
                make_shared<SpinPermScheme>(SpinPermScheme::initialize_sz(
                    SpinPermRecoupling::count_cds(std_exprs[ix]), std_exprs[ix],
                    true)));
        hdrt->initialize_steps(schemes);
        hdrt->initialize();
        print_hdrt_infos(iqs, std_exprs, hdrt);
        shared_ptr<HDRTScheme<S, FL>> hdrt_scheme =
            make_shared<HDRTScheme<S, FL>>(hdrt, schemes);
        vector<shared_ptr<vector<FL>>> ints(gfds.size());
        for (size_t i = 0; i < gfds.size(); i++)
            ints[i] = hdrt_scheme->sort_integral(gfds[i]);
        build_operator_matrices(hdrt, get_site_matrices(hdrt), ints, mats);
    }
};

} // namespace block2
