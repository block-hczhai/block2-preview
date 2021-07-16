
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

#include "expr.hpp"
#include "threading.hpp"
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

using namespace std;

namespace block2 {

enum struct SymTypes : uint8_t { RVec, CVec, Mat };

// General symbolic tensor
template <typename S> struct Symbolic {
    int m, n; //!< rows, columns
    vector<shared_ptr<OpExpr<S>>> data;
    Symbolic(int m, int n) : m(m), n(n), data(){};
    Symbolic(int m, int n, const vector<shared_ptr<OpExpr<S>>> &data)
        : m(m), n(n), data(data){};
    virtual ~Symbolic() = default;
    virtual SymTypes get_type() const = 0;
    virtual shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) {
        // The purpose of this implementation is to simplify pybind
        assert(false);
        return data[0];
    }
    virtual shared_ptr<Symbolic<S>> copy() const = 0;
};

// Row vector of symbols
template <typename S> struct SymbolicRowVector : Symbolic<S> {
    SymbolicRowVector(int n) : Symbolic<S>(1, n) {
        Symbolic<S>::data =
            vector<shared_ptr<OpExpr<S>>>(n, make_shared<OpExpr<S>>());
    }
    SymbolicRowVector(int n, const vector<shared_ptr<OpExpr<S>>> &data)
        : Symbolic<S>(1, n, data) {}
    SymTypes get_type() const override { return SymTypes::RVec; }
    shared_ptr<OpExpr<S>> &operator[](int i) {
        assert(i < this->data.size());
        return Symbolic<S>::data[i];
    }
    /** Access via {0,i} (treated as matrix) */
    shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) override {
        assert(ix.size() == 2 && "access via {0,i}");
        auto i = ix.begin();
        assert(*i == 0);
        return (*this)[*(++i)];
    }
    shared_ptr<Symbolic<S>> copy() const override {
        shared_ptr<Symbolic<S>> r =
            make_shared<SymbolicRowVector<S>>(Symbolic<S>::n);
        r->data = Symbolic<S>::data;
        return r;
    }
};

// Column vector of symbols
template <typename S> struct SymbolicColumnVector : Symbolic<S> {
    SymbolicColumnVector(int n) : Symbolic<S>(n, 1) {
        Symbolic<S>::data =
            vector<shared_ptr<OpExpr<S>>>(n, make_shared<OpExpr<S>>());
    }
    SymbolicColumnVector(int n, const vector<shared_ptr<OpExpr<S>>> &data)
        : Symbolic<S>(n, 1, data) {}
    SymTypes get_type() const override { return SymTypes::CVec; }
    shared_ptr<OpExpr<S>> &operator[](int i) {
        assert(i < this->data.size());
        return Symbolic<S>::data[i];
    }
    /** Access via {i,0} (treated as matrix) */
    shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) override {
        assert(ix.size() == 2 && "access via {i,0}");
        assert(*(ix.begin() + 1) == 0);
        return (*this)[*ix.begin()];
    }
    shared_ptr<Symbolic<S>> copy() const override {
        shared_ptr<Symbolic<S>> r =
            make_shared<SymbolicColumnVector<S>>(Symbolic<S>::m);
        r->data = Symbolic<S>::data;
        return r;
    }
};

// (Element-wise) sparse matrix of symbols
template <typename S> struct SymbolicMatrix : Symbolic<S> {
    vector<pair<int, int>> indices; // there can be repeated pair of indices
    SymbolicMatrix(int m, int n) : Symbolic<S>(m, n) {}
    SymTypes get_type() const override { return SymTypes::Mat; }
    void add(int i, int j, const shared_ptr<OpExpr<S>> elem) {
        assert(i < this->m);
        assert(j < this->n);
        indices.push_back(make_pair(i, j));
        Symbolic<S>::data.push_back(elem);
    }
    shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) override {
        auto j = ix.begin(), i = j++;
        add(*i, *j, make_shared<OpExpr<S>>());
        return Symbolic<S>::data.back();
    }
    shared_ptr<Symbolic<S>> copy() const override {
        shared_ptr<SymbolicMatrix<S>> r =
            make_shared<SymbolicMatrix<S>>(Symbolic<S>::m, Symbolic<S>::n);
        r->data = Symbolic<S>::data;
        r->indices = indices;
        return r;
    }
};

template <typename S>
inline ostream &operator<<(ostream &os, const shared_ptr<Symbolic<S>> sym) {
    switch (sym->get_type()) {
    case SymTypes::RVec:
        os << "SymRVector [SIZE= " << sym->n << " ]" << endl;
        for (size_t i = 0; i < sym->data.size(); i++)
            os << "[ " << i << " ] = " << sym->data[i] << endl;
        break;
    case SymTypes::CVec:
        os << "SymCVector [SIZE= " << sym->m << " ]" << endl;
        for (size_t i = 0; i < sym->data.size(); i++)
            os << "[ " << i << " ] = " << sym->data[i] << endl;
        break;
    case SymTypes::Mat: {
        vector<pair<int, int>> &indices =
            dynamic_pointer_cast<SymbolicMatrix<S>>(sym)->indices;
        os << "SymMatrix [SIZE= " << sym->m << "x" << sym->n << " ]" << endl;
        for (size_t i = 0; i < sym->data.size(); i++)
            os << "[ " << indices[i].first << "," << indices[i].second
               << " ] = " << sym->data[i] << endl;
        break;
    }
    default:
        assert(false);
        break;
    }
    return os;
}

template <typename S>
inline void save_symbolic(const shared_ptr<Symbolic<S>> &x, ostream &ofs) {
    SymTypes tp = x->get_type();
    ofs.write((char *)&tp, sizeof(tp));
    ofs.write((char *)&x->m, sizeof(x->m));
    ofs.write((char *)&x->n, sizeof(x->n));
    int sz = (int)x->data.size();
    ofs.write((char *)&sz, sizeof(sz));
    for (int i = 0; i < sz; i++) {
        assert(x->data[i] != nullptr);
        save_expr(x->data[i], ofs);
    }
    if (tp == SymTypes::RVec)
        assert(x->m == 1 && sz == x->n);
    else if (tp == SymTypes::CVec)
        assert(x->n == 1 && sz == x->m);
    else if (tp == SymTypes::Mat) {
        shared_ptr<SymbolicMatrix<S>> mat =
            dynamic_pointer_cast<SymbolicMatrix<S>>(x);
        assert((int)mat->indices.size() == sz);
        ofs.write((char *)mat->indices.data(), sizeof(pair<int, int>) * sz);
    }
}

template <typename S>
inline shared_ptr<Symbolic<S>> load_symbolic(istream &ifs) {
    SymTypes tp;
    int m, n, sz;
    ifs.read((char *)&tp, sizeof(tp));
    ifs.read((char *)&m, sizeof(m));
    ifs.read((char *)&n, sizeof(n));
    ifs.read((char *)&sz, sizeof(sz));
    vector<shared_ptr<OpExpr<S>>> data(sz);
    for (int i = 0; i < sz; i++)
        data[i] = load_expr<S>(ifs);
    if (tp == SymTypes::RVec) {
        assert(m == 1 && sz == n);
        return make_shared<SymbolicRowVector<S>>(n, data);
    } else if (tp == SymTypes::CVec) {
        assert(n == 1 && sz == m);
        return make_shared<SymbolicColumnVector<S>>(m, data);
    } else if (tp == SymTypes::Mat) {
        shared_ptr<SymbolicMatrix<S>> mat =
            make_shared<SymbolicMatrix<S>>(m, n);
        mat->data = data;
        mat->indices.resize(sz);
        ifs.read((char *)mat->indices.data(), sizeof(pair<int, int>) * sz);
        return mat;
    } else {
        assert(false);
        return nullptr;
    }
}

// Dot product of symbolic vector/matrix
template <typename S>
inline const shared_ptr<Symbolic<S>>
operator*(const shared_ptr<Symbolic<S>> a, const shared_ptr<Symbolic<S>> b) {
    assert(a->n == b->m);
    if (a->get_type() == SymTypes::Mat && b->get_type() == SymTypes::Mat) {
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        shared_ptr<SymbolicMatrix<S>> r(
            make_shared<SymbolicMatrix<S>>(a->m, b->n));
        vector<pair<int, int>> &aidx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(a)->indices;
        vector<pair<int, int>> &bidx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(b)->indices;
        map<pair<int, int>, shared_ptr<OpExpr<S>>> mp;
        for (size_t j = 0; j < a->data.size(); j++)
            for (size_t k = 0; k < b->data.size(); k++)
                if (aidx[j].second == bidx[k].first) {
                    pair<int, int> p = make_pair(aidx[j].first, bidx[k].second);
                    if (mp.count(p) == 0)
                        mp[p] = zero;
                    mp.at(p) += a->data[j] * b->data[k];
                }
        for (auto &p : mp)
            (*r)[{p.first.first, p.first.second}] = p.second;
        return r;
    } else if (a->get_type() == SymTypes::RVec &&
               b->get_type() == SymTypes::Mat) {
        shared_ptr<SymbolicRowVector<S>> r(
            make_shared<SymbolicRowVector<S>>(b->n));
        int ntg = threading->activate_global();
        vector<pair<int, int>> &idx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(b)->indices;
        vector<int> pidx(idx.size());
        for (int i = 0; i < pidx.size(); i++)
            pidx[i] = i;
        sort(pidx.begin(), pidx.end(),
             [&idx](int i, int j) { return idx[i].second < idx[j].second; });
#pragma omp parallel for schedule(static, 50) num_threads(ntg)
        for (int j = 0; j < b->n; j++) {
            size_t ki = lower_bound(pidx.begin(), pidx.end(), j,
                                    [&idx](int ii, int jj) {
                                        return idx[ii].second < jj;
                                    }) -
                        pidx.begin();
            vector<shared_ptr<OpExpr<S>>> xs;
            xs.reserve(pidx.size() - ki);
            for (size_t k = ki; k < pidx.size() && idx[pidx[k]].second == j;
                 k++)
                xs.push_back(a->data[idx[pidx[k]].first] * b->data[pidx[k]]);
            (*r)[j] = sum(xs);
        }
        threading->activate_normal();
        return r;
    } else if (a->get_type() == SymTypes::Mat &&
               b->get_type() == SymTypes::CVec) {
        shared_ptr<SymbolicColumnVector<S>> r(
            make_shared<SymbolicColumnVector<S>>(a->m));
        int ntg = threading->activate_global();
        vector<pair<int, int>> &idx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(a)->indices;
        vector<int> pidx(idx.size());
        for (int i = 0; i < pidx.size(); i++)
            pidx[i] = i;
        sort(pidx.begin(), pidx.end(),
             [&idx](int i, int j) { return idx[i].first < idx[j].first; });
#pragma omp parallel for schedule(static, 50) num_threads(ntg)
        for (int i = 0; i < a->m; i++) {
            size_t ki = lower_bound(pidx.begin(), pidx.end(), i,
                                    [&idx](int ii, int jj) {
                                        return idx[ii].first < jj;
                                    }) -
                        pidx.begin();
            vector<shared_ptr<OpExpr<S>>> xs;
            xs.reserve(pidx.size() - ki);
            for (size_t k = ki; k < pidx.size() && idx[pidx[k]].first == i;
                 k++)
                xs.push_back(a->data[pidx[k]] * b->data[idx[pidx[k]].second]);
            (*r)[i] = sum(xs);
        }
        threading->activate_normal();
        return r;
    } else if (a->get_type() == SymTypes::RVec &&
               b->get_type() == SymTypes::CVec) {
        shared_ptr<SymbolicColumnVector<S>> r(
            make_shared<SymbolicColumnVector<S>>(1));
        (*r)[0] = dot_product(a->data, b->data);
        return r;
    }
    assert(false);
    return nullptr;
}

} // namespace block2
