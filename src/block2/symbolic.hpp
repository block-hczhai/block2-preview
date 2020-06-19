
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

#pragma once

#include "expr.hpp"
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
    virtual const SymTypes get_type() const = 0;
    virtual shared_ptr<OpExpr<S>> &
    operator[](const initializer_list<int> ix) = 0;
    virtual shared_ptr<Symbolic<S>> copy() const = 0;
};

// Row vector of symbols
template <typename S> struct SymbolicRowVector : Symbolic<S> {
    SymbolicRowVector(int n) : Symbolic<S>(1, n) {
        Symbolic<S>::data =
            vector<shared_ptr<OpExpr<S>>>(n, make_shared<OpExpr<S>>());
    }
    const SymTypes get_type() const override { return SymTypes::RVec; }
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
    const SymTypes get_type() const override { return SymTypes::CVec; }
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
    const SymTypes get_type() const override { return SymTypes::Mat; }
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

// Dot product of symbolic vector/matrix
template <typename S>
inline const shared_ptr<Symbolic<S>>
operator*(const shared_ptr<Symbolic<S>> a, const shared_ptr<Symbolic<S>> b) {
    assert(a->n == b->m);
    if (a->get_type() == SymTypes::RVec && b->get_type() == SymTypes::Mat) {
        shared_ptr<SymbolicRowVector<S>> r(
            make_shared<SymbolicRowVector<S>>(b->n));
        vector<pair<int, int>> &idx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(b)->indices;
        vector<shared_ptr<OpExpr<S>>> xs[b->n];
        for (size_t k = 0; k < b->data.size(); k++) {
            int i = idx[k].first, j = idx[k].second;
            xs[j].push_back(a->data[i] * b->data[k]);
        }
        for (size_t j = 0; j < b->n; j++)
            (*r)[j] = sum(xs[j]);
        return r;
    } else if (a->get_type() == SymTypes::Mat &&
               b->get_type() == SymTypes::CVec) {
        shared_ptr<SymbolicColumnVector<S>> r(
            make_shared<SymbolicColumnVector<S>>(a->m));
        vector<pair<int, int>> &idx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(a)->indices;
        vector<shared_ptr<OpExpr<S>>> xs[a->m];
        for (size_t k = 0; k < a->data.size(); k++) {
            int i = idx[k].first, j = idx[k].second;
            xs[i].push_back(a->data[k] * b->data[j]);
        }
        for (size_t i = 0; i < a->m; i++)
            (*r)[i] = sum(xs[i]);
        return r;
    } else if (a->get_type() == SymTypes::RVec &&
               b->get_type() == SymTypes::CVec) {
        shared_ptr<SymbolicColumnVector<S>> r(
            make_shared<SymbolicColumnVector<S>>(1));
        (*r)[0] = dot_product(a->data, b->data);
        return r;
    }
    assert(false);
}

} // namespace block2
