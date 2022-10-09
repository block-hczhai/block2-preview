
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2022 Huanchen Zhai <hczhai@caltech.edu>
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

/** Graph Theory Algorithms. */

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

// Max flow (dinic)
struct Flow {
    typedef unordered_map<int, int>::const_iterator mci;
    vector<unordered_map<int, int>> resi;
    vector<int> dist;
    vector<mci> rix;
    int n, nfs, inf;
    Flow(int n) : n(n) {
        resi.resize(n + 2);
        dist.resize(n + 2);
        rix.resize(n + 2);
        nfs = 0;
        inf = numeric_limits<int>::max();
    }
    void mvc_dfs(int i, uint8_t *vis) {
        vis[i] = 1;
        for (const auto &ri : resi[i])
            if (ri.second != 0 && !vis[ri.first])
                mvc_dfs(ri.first, vis);
    }
    // Minimum Vertex Cover (bipartite graph)
    void mvc(int xi, int yi, int xn, int yn, vector<int> &vx, vector<int> &vy) {
        dinic();
        vector<uint8_t> vis;
        vis.reserve(n + 2);
        memset(vis.data(), 0, (n + 2) * sizeof(uint8_t));
        mvc_dfs(n, vis.data());
        vx.reserve(xn);
        vy.reserve(yn);
        for (int i = 0; i < xn; i++)
            if (resi[n][xi + i] == 0 && !vis[xi + i])
                vx.push_back(i);
        for (int i = 0; i < yn; i++)
            if (resi[yi + i][n + 1] == 0 && vis[yi + i])
                vy.push_back(i);
    }
    // dinic algorithm (O(n^2m). for capacity=1, O(m sqrt(n)))
    int dinic() {
        for (int i = 0; i < n + 2; i++)
            for (const auto &ri : resi[i])
                if (ri.second != 0)
                    resi[ri.first][i] = 0;
        int rs = 0;
        for (nfs = 0; dbfs(); rs += ddfs(n, inf), nfs++)
            ;
        return rs;
    }
    // dinic dfs
    int ddfs(int x, int flow) {
        if (x == n + 1)
            return flow;
        int used = 0;
        for (mci ri = rix[x]; ri != resi[x].end(); ri++) {
            int i = (rix[x] = ri)->first, fx;
            if (ri->second != 0 && dist[i] == dist[x] + 1) {
                fx = ddfs(i, min(flow - used, ri->second));
                resi[x][i] -= fx;
                resi[i][x] += fx;
                used += fx;
                if (used == flow)
                    return used;
            }
        }
        if (!used)
            dist[x] = -1;
        return used;
    }
    // dinic bfs
    bool dbfs() {
        int nn = n + 2;
        for (int i = 0; i < nn; i++)
            rix[i] = resi[i].begin();
        memset(dist.data(), -1, nn * sizeof(int));
        dist[n] = 0;
        int h = 0, r = 1, x;
        vector<int> q;
        q.reserve(nn);
        q[0] = n;
        while (h < r) {
            if ((x = q[h++]) == n + 1)
                return true;
            for (const auto &ri : resi[x])
                if (ri.second != 0 && dist[ri.first] == -1)
                    dist[ri.first] = dist[x] + 1, q[r++] = ri.first;
        }
        return dist[n + 1] != -1;
    }
};

// Min cost max flow (SAP & SPFA)
struct CostFlow {
    typedef unordered_map<int, pair<int, int>>::const_iterator mci;
    vector<unordered_map<int, pair<int, int>>> resi;
    vector<int> dist;
    vector<int> pre;
    int n, nfs, inf;
    CostFlow(int n) : n(n) {
        resi.resize(n + 2);
        dist.resize(n + 2);
        pre.resize(n + 2);
        nfs = 0;
        inf = numeric_limits<int>::max();
    }

    void mvc_dfs(int i, uint8_t *vis) {
        vis[i] = 1;
        for (const auto &ri : resi[i])
            if (ri.second.first != 0 && !vis[ri.first])
                mvc_dfs(ri.first, vis);
    }
    // Minimum Vertex Cover (bipartite graph)
    int mvc(int xi, int yi, int xn, int yn, vector<int> &vx, vector<int> &vy) {
        sap();
        vector<uint8_t> vis;
        vis.reserve(n + 2);
        memset(vis.data(), 0, (n + 2) * sizeof(uint8_t));
        mvc_dfs(n, vis.data());
        vx.reserve(xn);
        vy.reserve(yn);
        int cost = 0;
        for (int i = 0; i < xn; i++)
            if (resi[n][xi + i].first == 0 && !vis[xi + i])
                vx.push_back(i), cost += resi[n][xi + i].second;
        for (int i = 0; i < yn; i++)
            if (resi[yi + i][n + 1].first == 0 && vis[yi + i])
                vy.push_back(i), cost += resi[yi + i][n + 1].second;
        return cost;
    }

    // Shortest Augmenting Path
    pair<int, int> sap() {
        int flow = 0, cost = 0;
        for (; spfa();) {
            for (int i = n + 1; i != n; i = pre[i]) {
                resi[pre[i]].at(i).first -= pre[n];
                if (resi[i].count(pre[i]))
                    resi[i][pre[i]].first += pre[n];
                else
                    resi[i][pre[i]] =
                        make_pair(pre[n], -resi[pre[i]][i].second);
            }
            flow += pre[n];
            cost += pre[n] * dist[n + 1];
        }
        return make_pair(flow, cost);
    }

    // Shortest Path Faster Algorithm
    bool spfa() {
        int nn = n + 2;
        vector<int> q;
        q.reserve(nn + 1);
        vector<uint8_t> v;
        v.reserve(nn);
        int h = 0, l = 1, x;
        for (int i = 0; i < nn; i++)
            dist[i] = inf;
        memset(pre.data(), -1, nn * sizeof(int));
        memset(v.data(), 0, nn * sizeof(uint8_t));
        dist[n] = 0, pre[n] = -1;
        q[0] = n, v[n] = 1;
        while (l != 0) {
            x = q[h];
            v[q[h]] = 0;
            l--;
            h = (h + 1) % nn;
            for (mci ix = resi[x].begin(); ix != resi[x].end(); ix++) {
                if (ix->second.first == 0)
                    continue;
                int i = ix->first, p = ix->second.second;
                if (dist[x] + p < dist[i]) {
                    dist[i] = dist[x] + p;
                    pre[i] = x;
                    if (!v[i])
                        v[q[(h + l++) % nn] = i] = 1;
                }
            }
        }
        if (pre[n + 1] != -1) {
            int flow = inf;
            for (int i = n + 1; i != n; i = pre[i])
                if (resi[pre[i]][i].first < flow)
                    flow = resi[pre[i]][i].first;
            pre[n] = flow;
        }
        return pre[n + 1] != -1;
    }
};

// Disjoint Set Union
struct DSU {
    int n;
    vector<int> parex, rankx;
    unordered_map<int, vector<int>> roots;
    DSU(int n) : n(n) {
        parex.resize(n), rankx.resize(n);
        for (int i = 0; i < n; i++)
            parex[i] = i, rankx[i] = 0;
    }
    ~DSU() {}
    int findx(int x) {
        if (parex[x] != x)
            parex[x] = findx(parex[x]);
        return parex[x];
    }
    void unionx(int x, int y) {
        x = findx(x);
        y = findx(y);
        if (x == y)
            return;
        else if (rankx[x] < rankx[y])
            parex[x] = y;
        else if (rankx[x] > rankx[y])
            parex[y] = x;
        else
            parex[y] = x, rankx[x]++;
    }
    void post() {
        roots.clear();
        for (int i = 0; i < n; i++)
            roots[findx(i)].push_back(i);
    }
};

} // namespace block2
