
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

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace block2 {

// Operator names
enum struct OpNames : uint8_t {
    H,
    I,
    N,
    NN,
    C,
    D,
    R,
    RD,
    A,
    AD,
    P,
    PD,
    B,
    Q,
    Zero,
    PDM1,
    PDM2
};

inline ostream &operator<<(ostream &os, const OpNames c) {
    const static string repr[] = {"H", "I",  "N",    "NN",   "C",   "D",
                                  "R", "RD", "A",    "AD",   "P",   "PD",
                                  "B", "Q",  "Zero", "PDM1", "PDM2"};
    os << repr[(uint8_t)c];
    return os;
}

// Expression types
enum struct OpTypes : uint8_t { Zero, Elem, Prod, Sum, ElemRef, SumProd };

// Expression zero
template <typename S> struct OpExpr {
    virtual const OpTypes get_type() const { return OpTypes::Zero; }
    bool operator==(const OpExpr &other) const { return true; }
};

// Site index in operator symbol
// (support at most 4 site indices (< 4096) and 8 spin indices)
// (spin-index: 8bits) - (orb-index: 4 * 12bits at 8/20/32/44) - (nspin: 4bits)
// - (norb: 4bits)
struct SiteIndex {
    uint64_t data;
    SiteIndex() : data(0) {}
    SiteIndex(uint64_t data) : data(data) {}
    SiteIndex(uint16_t i) : data(1ULL | ((uint64_t)i << 8)) {}
    SiteIndex(uint16_t i, uint16_t j)
        : data(2ULL | ((uint64_t)i << 8) | ((uint64_t)j << 20)) {}
    SiteIndex(uint16_t i, uint16_t j, uint8_t s)
        : data(2ULL | (1ULL << 4) | ((uint64_t)i << 8) | ((uint64_t)j << 20) |
               ((uint64_t)s << 56)) {}
    SiteIndex(const initializer_list<uint16_t> i,
              const initializer_list<uint8_t> s)
        : data(0) {
        data |= i.size() | (s.size() << 4);
        int x = 8;
        for (auto iit = i.begin(); iit != i.end(); iit++, x += 12)
            data |= (uint64_t)(*iit) << x;
        x = 56;
        for (auto sit = s.begin(); sit != s.end(); sit++, x++) {
            assert((*sit) == 0 || (*sit) == 1);
            data |= (uint64_t)(*sit) << x;
        }
    }
    // Number of site indices
    uint8_t size() const noexcept { return (uint8_t)(data & 0xFU); }
    // Number of spin indices
    uint8_t spin_size() const noexcept { return (uint8_t)((data >> 4) & 0xFU); }
    // Composite spin index
    uint8_t ss() const noexcept { return (data >> 56) & 0xFFU; }
    // Spin index
    uint8_t s(uint8_t i = 0) const noexcept {
        return !!(data & (1ULL << (56 + i)));
    }
    // Site index
    uint16_t operator[](uint8_t i) const noexcept {
        return (data >> (8 + i * 12)) & 0xFFFU;
    }
    bool operator==(SiteIndex other) const noexcept {
        return data == other.data;
    }
    bool operator!=(SiteIndex other) const noexcept {
        return data != other.data;
    }
    bool operator<(SiteIndex other) const noexcept { return data < other.data; }
    // Flip first two site indices
    SiteIndex flip_spatial() const noexcept {
        return SiteIndex((uint64_t)((data & 0xFF000000000000FFULL) |
                                    ((uint64_t)(*this)[0] << 20) |
                                    ((uint64_t)(*this)[1] << 8)));
    }
    // Flip first two site indices and associated spin indices
    SiteIndex flip() const noexcept {
        return SiteIndex((uint64_t)(
            (data & 0xFFULL) | ((uint64_t)s(0) << 57) | ((uint64_t)s(1) << 56) |
            ((uint64_t)(*this)[0] << 20) | ((uint64_t)(*this)[1] << 8)));
    }
    size_t hash() const noexcept { return (size_t)data; }
    vector<uint16_t> to_array() const {
        vector<uint16_t> r;
        r.reserve(size() + spin_size());
        for (uint8_t i = 0; i < size(); i++)
            r.push_back((*this)[i]);
        for (uint8_t i = 0; i < spin_size(); i++)
            r.push_back(s(i));
        return r;
    }
    string to_str() const {
        stringstream ss;
        ss << "[ ";
        for (uint8_t i = 0; i < size(); i++)
            ss << (int)(*this)[i] << " ";
        for (uint8_t i = 0; i < spin_size(); i++)
            ss << (int)s(i) << " ";
        ss << "]";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SiteIndex c) {
        os << c.to_str();
        return os;
    }
};

// Single operator symbol: (A)
template <typename S> struct OpElement : OpExpr<S> {
    OpNames name;
    SiteIndex site_index;
    double factor;
    S q_label;
    OpElement(OpNames name, SiteIndex site_index, S q_label,
              double factor = 1.0)
        : name(name), site_index(site_index), factor(factor), q_label(q_label) {
    }
    const OpTypes get_type() const override { return OpTypes::Elem; }
    OpElement abs() const { return OpElement(name, site_index, q_label, 1.0); }
    OpElement operator*(double d) const {
        return OpElement(name, site_index, q_label, factor * d);
    }
    bool operator==(const OpElement &other) const {
        return name == other.name && site_index == other.site_index &&
               factor == other.factor;
    }
    bool operator<(const OpElement &other) const {
        if (name != other.name)
            return name < other.name;
        else if (site_index != other.site_index)
            return site_index < other.site_index;
        else if (factor != other.factor)
            return factor < other.factor;
        else
            return false;
    }
    size_t hash() const noexcept {
        size_t h = (size_t)name;
        h ^= site_index.hash() + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(factor) + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    friend ostream &operator<<(ostream &os, const OpElement<S> &c) {
        if (c.factor != 1.0)
            os << "(" << c.factor << " " << c.abs() << ")";
        else if (c.site_index.data == 0)
            os << c.name;
        else if (c.site_index.size() == 1 && c.site_index.spin_size() == 0)
            os << c.name << (int)c.site_index[0];
        else
            os << c.name << c.site_index;
        return os;
    }
};

// Reference to original or transposed symbol: (A) or (A)^T
template <typename S> struct OpElementRef : OpExpr<S> {
    shared_ptr<OpElement<S>> op;
    int8_t factor;
    int8_t trans;
    OpElementRef(const shared_ptr<OpElement<S>> &op, int8_t trans,
                 int8_t factor)
        : op(op), trans(trans), factor(factor) {}
    const OpTypes get_type() const override { return OpTypes::ElemRef; }
};

// Tensor product of two operator symbols: (A) x (B)
// (conj & 1) indicates whether a is transposed
// (conj & 2) indicates whether b is transposed
template <typename S> struct OpString : OpExpr<S> {
    shared_ptr<OpElement<S>> a, b;
    double factor;
    uint8_t conj;
    OpString(const shared_ptr<OpElement<S>> &op, double factor,
             uint8_t conj = 0)
        : factor(factor * op->factor), a(make_shared<OpElement<S>>(op->abs())),
          b(nullptr), conj(conj) {}
    OpString(const shared_ptr<OpElement<S>> &a,
             const shared_ptr<OpElement<S>> &b, double factor, uint8_t conj = 0)
        : factor(factor * (a == nullptr ? 1.0 : a->factor) *
                 (b == nullptr ? 1.0 : b->factor)),
          a(a == nullptr ? nullptr : make_shared<OpElement<S>>(a->abs())),
          b(b == nullptr ? nullptr : make_shared<OpElement<S>>(b->abs())),
          conj(conj) {}
    const OpTypes get_type() const override { return OpTypes::Prod; }
    OpString abs() const { return OpString(a, b, 1.0, conj); }
    shared_ptr<OpElement<S>> get_op() const {
        assert(b == nullptr);
        return a;
    }
    OpString operator*(double d) const {
        return OpString(a, b, factor * d, conj);
    }
    bool operator==(const OpString &other) const {
        return *a == *other.a &&
               (b == nullptr ? other.b == nullptr
                             : (other.b != nullptr && *b == *other.b)) &&
               factor == other.factor && conj == other.conj;
    }
    friend ostream &operator<<(ostream &os, const OpString<S> &c) {
        if (c.factor != 1.0)
            os << "(" << c.factor << " " << c.abs() << ")";
        else {
            os << *c.a << (c.conj & 1 ? "^T " : " ");
            if (c.b != nullptr)
                os << *c.b << (c.conj & 2 ? "^T " : " ");
        }
        return os;
    }
};

// Tensor product of one symbol and a sum:
// (A) x {(B1) + (B2) + ...} or {(A1) + (A2) + ...} x (B)
template <typename S> struct OpSumProd : OpString<S> {
    vector<shared_ptr<OpElement<S>>> ops;
    vector<bool> conjs;
    OpSumProd(const shared_ptr<OpElement<S>> &lop,
              const vector<shared_ptr<OpElement<S>>> &ops,
              const vector<bool> &conjs, double factor, uint8_t conj = 0)
        : ops(ops), conjs(conjs), OpString<S>(lop, nullptr, factor, conj) {}
    OpSumProd(const vector<shared_ptr<OpElement<S>>> &ops,
              const shared_ptr<OpElement<S>> &rop, const vector<bool> &conjs,
              double factor, uint8_t conj = 0)
        : ops(ops), conjs(conjs), OpString<S>(nullptr, rop, factor, conj) {}
    const OpTypes get_type() const override { return OpTypes::SumProd; }
    OpSumProd operator*(double d) const {
        if (OpString<S>::a == nullptr)
            return OpSumProd(ops, OpString<S>::b, conjs,
                             OpString<S>::factor * d, OpString<S>::conj);
        else if (OpString<S>::b == nullptr)
            return OpSumProd(OpString<S>::a, ops, conjs,
                             OpString<S>::factor * d, OpString<S>::conj);
        else
            assert(false);
    }
    bool operator==(const OpSumProd &other) const {
        if (ops.size() != other.ops.size() ||
            (OpString<S>::a == nullptr) != (other.a == nullptr) ||
            (OpString<S>::b == nullptr) != (other.b == nullptr))
            return false;
        else if (OpString<S>::a == nullptr && !(*OpString<S>::b == *other.b))
            return false;
        else if (OpString<S>::b == nullptr && !(*OpString<S>::a == *other.a))
            return false;
        else if (conjs != other.conjs)
            return false;
        else
            for (size_t i = 0; i < ops.size(); i++)
                if (!(*ops[i] == *other.ops[i]))
                    return false;
        return true;
    }
    friend ostream &operator<<(ostream &os, const OpSumProd<S> &c) {
        if (c.ops.size() != 0) {
            if (c.factor != 1.0)
                os << "(" << c.factor << " ";
            if (c.a != nullptr)
                os << *c.a << (c.conj & 1 ? "^T " : " ");
            os << "{ ";
            for (size_t i = 0; i < c.ops.size() - 1; i++)
                os << *c.ops[i] << (c.conjs[i] ? "^T " : " ") << " + ";
            os << *c.ops.back();
            os << " }" << (c.conj & ((c.a != nullptr) + 1) ? "^T" : "");
            if (c.b != nullptr)
                os << " " << *c.b << (c.conj & 2 ? "^T " : " ");
            if (c.factor != 1.0)
                os << " )";
        }
        return os;
    }
};

// Sum of tensor products:
// (A) + (B) + (C) + ... or
// (A1) x (B1) + (A2) x (B2) + ... or
// (A1) x { (B1) + (B2) + ...} + (A2) x { {C1} + ... } + ...
template <typename S> struct OpSum : OpExpr<S> {
    vector<shared_ptr<OpString<S>>> strings;
    OpSum(const vector<shared_ptr<OpString<S>>> &strings) : strings(strings) {}
    const OpTypes get_type() const override { return OpTypes::Sum; }
    OpSum operator*(double d) const {
        vector<shared_ptr<OpString<S>>> strs;
        strs.reserve(strings.size());
        for (auto &r : strings)
            if (r->get_type() == OpTypes::Prod)
                strs.push_back(make_shared<OpString<S>>(*r * d));
            else
                strs.push_back(make_shared<OpSumProd<S>>(
                    *dynamic_pointer_cast<OpSumProd<S>>(r) * d));
        return OpSum(strs);
    }
    bool operator==(const OpSum &other) const {
        if (strings.size() != other.strings.size())
            return false;
        for (size_t i = 0; i < strings.size(); i++)
            if (!(*strings[i] == *other.strings[i]))
                return false;
        return true;
    }
    friend ostream &operator<<(ostream &os, const OpSum<S> &c) {
        if (c.strings.size() != 0) {
            for (size_t i = 0; i < c.strings.size() - 1; i++)
                if (c.strings[i]->get_type() == OpTypes::Prod)
                    os << *c.strings[i] << " + ";
                else if (c.strings[i]->get_type() == OpTypes::SumProd)
                    os << *dynamic_pointer_cast<OpSumProd<S>>(c.strings[i])
                       << " + ";
            if (c.strings.back()->get_type() == OpTypes::Prod)
                os << *c.strings.back();
            else if (c.strings.back()->get_type() == OpTypes::SumProd)
                os << *dynamic_pointer_cast<OpSumProd<S>>(c.strings.back());
        }
        return os;
    }
};

template <typename S> inline size_t hash_value(const shared_ptr<OpExpr<S>> &x) {
    assert(x->get_type() == OpTypes::Elem);
    return dynamic_pointer_cast<OpElement<S>>(x)->hash();
}

// Absolute value
template <typename S>
inline shared_ptr<OpExpr<S>> abs_value(const shared_ptr<OpExpr<S>> &x) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (x->get_type() == OpTypes::Elem) {
        shared_ptr<OpElement<S>> op = dynamic_pointer_cast<OpElement<S>>(x);
        return op->factor == 1.0 ? x : make_shared<OpElement<S>>(op->abs());
    } else if (x->get_type() == OpTypes::Prod) {
        shared_ptr<OpString<S>> op = dynamic_pointer_cast<OpString<S>>(x);
        return op->factor == 1.0 ? x : make_shared<OpString<S>>(op->abs());
    }
    assert(false);
}

// String representation
template <typename S> inline string to_str(const shared_ptr<OpExpr<S>> &x) {
    stringstream ss;
    if (x->get_type() == OpTypes::Zero)
        ss << 0;
    else if (x->get_type() == OpTypes::Elem)
        ss << *dynamic_pointer_cast<OpElement<S>>(x);
    else if (x->get_type() == OpTypes::Prod)
        ss << *dynamic_pointer_cast<OpString<S>>(x);
    else if (x->get_type() == OpTypes::Sum)
        ss << *dynamic_pointer_cast<OpSum<S>>(x);
    else if (x->get_type() == OpTypes::SumProd)
        ss << *dynamic_pointer_cast<OpSumProd<S>>(x);
    return ss.str();
}

template <typename S>
inline bool operator==(const shared_ptr<OpExpr<S>> &a,
                       const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() != b->get_type())
        return false;
    switch (a->get_type()) {
    case OpTypes::Zero:
        return *a == *b;
    case OpTypes::Elem:
        return *dynamic_pointer_cast<OpElement<S>>(a) ==
               *dynamic_pointer_cast<OpElement<S>>(b);
    case OpTypes::Prod:
        return *dynamic_pointer_cast<OpString<S>>(a) ==
               *dynamic_pointer_cast<OpString<S>>(b);
    case OpTypes::Sum:
        return *dynamic_pointer_cast<OpSum<S>>(a) ==
               *dynamic_pointer_cast<OpSum<S>>(b);
    case OpTypes::SumProd:
        return *dynamic_pointer_cast<OpSumProd<S>>(a) ==
               *dynamic_pointer_cast<OpSumProd<S>>(b);
    default:
        return false;
    }
}

template <typename S> struct op_expr_less {
    bool operator()(const shared_ptr<OpExpr<S>> &a,
                    const shared_ptr<OpExpr<S>> &b) const {
        assert(a->get_type() == OpTypes::Elem &&
               b->get_type() == OpTypes::Elem);
        return *dynamic_pointer_cast<OpElement<S>>(a) <
               *dynamic_pointer_cast<OpElement<S>>(b);
    }
};

// Sum of two symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>> operator+(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() == OpTypes::Zero)
        return b;
    else if (b->get_type() == OpTypes::Zero)
        return a;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a), 1.0));
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(b), 1.0));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size() + 1);
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a), 1.0));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size() + 1);
            strs.push_back(dynamic_pointer_cast<OpString<S>>(a));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(2);
            strs.push_back(dynamic_pointer_cast<OpString<S>>(a));
            strs.push_back(dynamic_pointer_cast<OpString<S>>(b));
            return make_shared<OpSum<S>>(strs);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() + 1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(b), 1.0));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() + 1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.push_back(dynamic_pointer_cast<OpString<S>>(b));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() +
                         dynamic_pointer_cast<OpSum<S>>(b)->strings.size());
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        }
    }
    assert(false);
}

template <typename S>
inline const shared_ptr<OpExpr<S>> operator+=(shared_ptr<OpExpr<S>> &a,
                                              const shared_ptr<OpExpr<S>> &b) {
    return a = a + b;
}

// A symbolic expression multiply a scalar
template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(const shared_ptr<OpExpr<S>> &x,
                                             double d) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (d == 0.0)
        return make_shared<OpExpr<S>>();
    else if (d == 1.0)
        return x;
    else if (x->get_type() == OpTypes::Elem)
        return make_shared<OpElement<S>>(
            *dynamic_pointer_cast<OpElement<S>>(x) * d);
    else if (x->get_type() == OpTypes::Prod)
        return make_shared<OpString<S>>(*dynamic_pointer_cast<OpString<S>>(x) *
                                        d);
    else if (x->get_type() == OpTypes::Sum)
        return make_shared<OpSum<S>>(*dynamic_pointer_cast<OpSum<S>>(x) * d);
    assert(false);
}

// A scalar multiply a symbolic expression
template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(double d,
                                             const shared_ptr<OpExpr<S>> &x) {
    return x * d;
}

// Tensor product of two symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() == OpTypes::Zero)
        return a;
    else if (b->get_type() == OpTypes::Zero)
        return b;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S>>(b)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpString<S>>(
                    dynamic_pointer_cast<OpElement<S>>(a), r->a, r->factor));
            }
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Elem)
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a),
                dynamic_pointer_cast<OpElement<S>>(b), 1.0);
        else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OpString<S>>(b)->b == nullptr);
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a),
                dynamic_pointer_cast<OpString<S>>(b)->a,
                dynamic_pointer_cast<OpString<S>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Elem) {
            assert(dynamic_pointer_cast<OpString<S>>(a)->b == nullptr);
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpString<S>>(a)->a,
                dynamic_pointer_cast<OpElement<S>>(b),
                dynamic_pointer_cast<OpString<S>>(a)->factor);
        } else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OpString<S>>(a)->b == nullptr);
            assert(dynamic_pointer_cast<OpString<S>>(b)->b == nullptr);
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpString<S>>(a)->a,
                dynamic_pointer_cast<OpString<S>>(b)->a,
                dynamic_pointer_cast<OpString<S>>(a)->factor *
                    dynamic_pointer_cast<OpString<S>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S>>(a)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpString<S>>(
                    r->a, dynamic_pointer_cast<OpElement<S>>(b), r->factor));
            }
            return make_shared<OpSum<S>>(strs);
        }
    }
    assert(false);
}

// Sum of several symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>>
sum(const vector<shared_ptr<OpExpr<S>>> &xs) {
    const static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
    vector<shared_ptr<OpString<S>>> strs;
    for (auto &r : xs)
        if (r->get_type() == OpTypes::Prod)
            strs.push_back(dynamic_pointer_cast<OpString<S>>(r));
        else if (r->get_type() == OpTypes::SumProd)
            strs.push_back(dynamic_pointer_cast<OpSumProd<S>>(r));
        else if (r->get_type() == OpTypes::Elem)
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(r), 1.0));
        else if (r->get_type() == OpTypes::Sum) {
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(r)->strings.size() +
                         strs.size());
            for (auto &rr : dynamic_pointer_cast<OpSum<S>>(r)->strings)
                strs.push_back(rr);
        }
    return strs.size() != 0 ? make_shared<OpSum<S>>(strs) : zero;
}

// Dot product of two vectors of symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>>
dot_product(const vector<shared_ptr<OpExpr<S>>> &a,
            const vector<shared_ptr<OpExpr<S>>> &b) {
    vector<shared_ptr<OpExpr<S>>> xs;
    assert(a.size() == b.size());
    for (size_t k = 0; k < a.size(); k++)
        xs.push_back(a[k] * b[k]);
    return sum(xs);
}

template <typename S>
inline ostream &operator<<(ostream &os, const shared_ptr<OpExpr<S>> &c) {
    os << to_str(c);
    return os;
}

} // namespace block2

namespace std {

template <typename S> struct hash<block2::OpElement<S>> {
    size_t operator()(const block2::OpElement<S> &s) const noexcept {
        return s.hash();
    }
};

} // namespace std
