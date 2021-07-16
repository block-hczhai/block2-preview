
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

/** Fast Fourier Transform (FFT) and number theory algorithms. */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

/** Kuhn-Munkres algorithm for finding the best matching
 * with lowest cost.
 * Complexity: O(n^3).
 */
struct KuhnMunkres {
    vector<double> cost;  //!< Flattened n x n matrix of cost.
    int n;                //!< Number of rows or columns.
    double inf;           //!< Infinity (constant).
    double eps;           //!< Machine precision (constant).
    vector<double> lx;    //!< Left feasible labeling (working array).
    vector<double> ly;    //!< Right feasible labeling (working array).
    vector<double> slack; //!< Slack working array.
    vector<char> st; //!< Candidate augmenting alternating path (working array).
    /** Constructor.
     * @param cost_matrix Flattened matrix of cost.
     * @param m Number of rows.
     * @param n Number of columns.
     */
    KuhnMunkres(const vector<double> &cost_matrix, int m, int n = 0) {
        if (n == 0)
            n = m;
        this->n = max(m, n);
        cost.resize(this->n * this->n, 0);
        for (int i = 0; i < m; i++)
            memcpy(cost.data() + i * this->n, cost_matrix.data() + i * n,
                   n * sizeof(double));
        n = this->n;
        inf = *max_element(cost.begin(), cost.end()) * n;
        // machine precision
        eps = 0.0;
        for (double p1 = 4.0 / 3.0, p2, p3; eps == 0.0;)
            p2 = p1 - 1.0, p3 = p2 + p2 + p2, eps = abs(p3 - 1.0);
        lx.resize(n), ly.resize(n), slack.resize(n);
        st.resize(n);
    }
    /** Find an augmenting alternating path in the equality subgraph.
     * If an equality subgraph has a perfect matching, the it is a
     * maximum-weight matching in the graph.
     * @param x Current matching.
     * @param u Starting vertex.
     * @return ``true`` if an augmenting alternating path is found.
     */
    bool match(vector<int> &x, int u) {
        st[u] |= 2;
        for (int v = 0; v < n; v++) {
            if (st[v] & 1)
                continue;
            double kk = lx[u] + ly[v] + cost[u * n + v];
            if (abs(kk) < eps) {
                st[v] |= 1;
                if (x[v] == -1 || match(x, x[v])) {
                    x[v] = u;
                    return true;
                }
            } else if (kk < slack[v])
                slack[v] = kk;
        }
        return false;
    }
    /** Find the lowest cost and a matching.
     * @return the lowest cost and an index array x, with x[i] = j meaning
     *   that column index i should be matched to row index j.
     */
    pair<double, vector<int>> solve() {
        vector<int> x(n, -1);
        for (int i = 0; i < n; i++)
            lx[i] =
                -*min_element(cost.data() + i * n, cost.data() + (i + 1) * n);
        memset(ly.data(), 0, sizeof(double) * n);
        for (int i = 0; i < n; i++) {
            fill(slack.begin(), slack.end(), inf);
            while (true) {
                memset(st.data(), 0, sizeof(char) * n);
                if (match(x, i))
                    break;
                else {
                    double d = inf;
                    for (int p = 0; p < n; p++)
                        if (!(st[p] & 1) && slack[p] < d)
                            d = slack[p];
                    for (int j = 0; j < n; j++) {
                        if (st[j] & 2)
                            lx[j] -= d;
                        if (st[j] & 1)
                            ly[j] += d;
                    }
                }
            }
        }
        double f = 0;
        for (int i = 0; i < n; i++)
            f += cost[x[i] * n + i];
        return make_pair(f, x);
    }
};

} // namespace block2
