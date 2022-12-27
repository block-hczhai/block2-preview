
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

#include "utils.hpp"
#include <cassert>
#include <complex>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

using namespace std;

namespace block2 {

enum struct ParallelTypes : uint8_t {
    Serial = 0,
    Distributed = 1,
    NewScheme = 2,
    Simple = 4
};

inline bool operator&(ParallelTypes a, ParallelTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline ParallelTypes operator|(ParallelTypes a, ParallelTypes b) {
    return ParallelTypes((uint8_t)a | (uint8_t)b);
}

inline ParallelTypes operator^(ParallelTypes a, ParallelTypes b) {
    return ParallelTypes((uint8_t)a ^ (uint8_t)b);
}

// Operator names
enum struct OpNames : uint8_t {
    Zero,
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
    BD,
    Q,
    TR,
    TS,
    PDM1,
    PDM2,
    CCDD,
    CCD,
    CDC,
    CDD,
    DCC,
    DCD,
    DDC,
    TEMP,
    XL,
    XR,
    X,
    XPDM,
    SP,
    SM,
    SZ
};

inline ostream &operator<<(ostream &os, const OpNames c) {
    const static string repr[] = {
        "Zero", "H",    "I",    "N",   "NN",  "C",   "D",   "R",   "RD",
        "A",    "AD",   "P",    "PD",  "B",   "BD",  "Q",   "TR",  "TS",
        "PDM1", "PDM2", "CCDD", "CCD", "CDC", "CDD", "DCC", "DCD", "DDC",
        "TEMP", "XL",   "XR",   "X",   "SP",  "SM",  "SZ"};
    os << repr[(uint8_t)c];
    return os;
}

// Expression types
enum struct OpTypes : uint8_t {
    Zero,
    Elem,
    Prod,
    Sum,
    ElemRef,
    SumProd,
    ExprRef,
    Counter
};

struct OpNamesSet {
    uint32_t data;
    OpNamesSet() : data(0) {}
    OpNamesSet(uint32_t data) : data(data) {}
    OpNamesSet(const initializer_list<OpNames> names) : data(0) {
        for (auto iit = names.begin(); iit != names.end(); iit++)
            data |= (1 << (uint8_t)*iit);
    }
    OpNamesSet(const vector<OpNames> &names) : data(0) {
        for (auto iit = names.begin(); iit != names.end(); iit++)
            data |= (1 << (uint8_t)*iit);
    }
    static OpNamesSet normal_ops() noexcept {
        return OpNamesSet({OpNames::I, OpNames::N, OpNames::NN, OpNames::C,
                           OpNames::D, OpNames::A, OpNames::AD, OpNames::B,
                           OpNames::BD});
    }
    static OpNamesSet all_ops() noexcept {
        return OpNamesSet({OpNames::H, OpNames::I, OpNames::N, OpNames::NN,
                           OpNames::C, OpNames::D, OpNames::A, OpNames::AD,
                           OpNames::B, OpNames::BD, OpNames::R, OpNames::RD,
                           OpNames::P, OpNames::PD, OpNames::Q, OpNames::TR,
                           OpNames::TS});
    }
    bool operator()(OpNames name) const noexcept {
        return data & (1 << (uint8_t)name);
    }
    bool empty() const noexcept { return data == 0; }
};

// Expression zero
template <typename S> struct OpExpr {
    virtual ~OpExpr() = default;
    virtual OpTypes get_type() const { return OpTypes::Zero; }
    bool operator==(const OpExpr &other) const { return true; }
    virtual string get_name() const { return ""; }
    virtual bool is_normalized() const { return get_type() == OpTypes::Zero; }
    virtual shared_ptr<OpExpr> abs_expr() const {
        assert(false);
        return nullptr;
    }
    virtual shared_ptr<OpExpr> scalar_multiply(float d) const {
        return scalar_multiply((complex<float>)d);
    }
    virtual shared_ptr<OpExpr> scalar_multiply(complex<float> d) const {
        return make_shared<OpExpr>();
    }
    virtual shared_ptr<OpExpr> scalar_multiply(double d) const {
        return scalar_multiply((complex<double>)d);
    }
    virtual shared_ptr<OpExpr> scalar_multiply(complex<double> d) const {
        return make_shared<OpExpr>();
    }
    virtual size_t hash() const noexcept { return 0; }
    virtual bool is_equal_to(const shared_ptr<OpExpr> &other) const {
        if (get_type() != other->get_type())
            return false;
        else if (get_type() == OpTypes::Zero)
            return *this == *other;
        else
            assert(false);
        return false;
    }
    virtual bool is_less_than(const shared_ptr<OpExpr> &other) const {
        assert(false);
        return false;
    }
    virtual shared_ptr<OpExpr>
    sum(const vector<shared_ptr<OpExpr<S>>> &x) const {
        return nullptr;
    }
    virtual shared_ptr<OpExpr> plus(const shared_ptr<OpExpr> &a,
                                    const shared_ptr<OpExpr> &b) const {
        if (a->get_type() == OpTypes::Zero)
            return b;
        else if (b->get_type() == OpTypes::Zero)
            return a;
        else
            return nullptr;
    }
    virtual shared_ptr<OpExpr> multiply(const shared_ptr<OpExpr> &a,
                                        const shared_ptr<OpExpr> &b) const {
        if (a->get_type() == OpTypes::Zero)
            return a;
        else if (b->get_type() == OpTypes::Zero)
            return b;
        else
            return nullptr;
    }
    virtual void save(ostream &ofs) const {
        OpTypes tp = get_type();
        ofs.write((char *)&tp, sizeof(tp));
    }
    static shared_ptr<OpExpr> load(istream &ifs) {
        return make_shared<OpExpr>();
    }
    virtual string to_str() const {
        stringstream ss;
        ss << 0;
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, const OpExpr &c) {
        os << c.to_str();
        return os;
    }
};

template <typename S, typename FL>
inline const shared_ptr<OpExpr<S>> plus_expr(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b);

template <typename S>
inline const shared_ptr<OpExpr<S>>
plus_expr_ref(const shared_ptr<OpExpr<S>> &a, const shared_ptr<OpExpr<S>> &b);

template <typename S, typename FL>
inline const shared_ptr<OpExpr<S>>
multiply_expr(const shared_ptr<OpExpr<S>> &a, const shared_ptr<OpExpr<S>> &b);

template <typename S, typename FL>
inline const shared_ptr<OpExpr<S>>
sum_expr(const vector<shared_ptr<OpExpr<S>>> &x);

template <typename S, typename FL>
inline shared_ptr<OpExpr<S>> load_expr(istream &ifs);

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
        for (auto sit = s.begin(); sit != s.end(); sit++, x++)
            data |= (uint64_t)(*sit) << x;
    }
    SiteIndex(const vector<uint16_t> &i, const vector<uint8_t> &s) : data(0) {
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
        return SiteIndex((uint64_t)((data & 0xFFULL) | ((uint64_t)s(0) << 57) |
                                    ((uint64_t)s(1) << 56) |
                                    ((uint64_t)(*this)[0] << 20) |
                                    ((uint64_t)(*this)[1] << 8)));
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
    string get_name() const {
        stringstream ss;
        for (uint8_t i = 0; i < size(); i++)
            ss << (int)(*this)[i] << (i == size() - 1 ? "" : "-");
        if (spin_size() != 0)
            ss << "~";
        for (uint8_t i = 0; i < spin_size(); i++)
            ss << (int)s(i) << (i == spin_size() - 1 ? "" : "-");
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SiteIndex c) {
        os << c.to_str();
        return os;
    }
};

// Single operator symbol: (A)
template <typename S, typename FL> struct OpElement : OpExpr<S> {
    OpNames name;
    SiteIndex site_index;
    FL factor;
    S q_label;
    OpElement(OpNames name, SiteIndex site_index, S q_label, FL factor = 1.0)
        : name(name), site_index(site_index), factor(factor), q_label(q_label) {
    }
    OpTypes get_type() const override { return OpTypes::Elem; }
    bool is_normalized() const override { return factor == (FL)1; }
    shared_ptr<OpExpr<S>> abs_expr() const override {
        return make_shared<OpElement>(abs());
    }
    OpElement abs() const { return OpElement(name, site_index, q_label, 1.0); }
    shared_ptr<OpExpr<S>> scalar_multiply(FL d) const override {
        return make_shared<OpElement>(*this * d);
    }
    shared_ptr<OpExpr<S>>
    scalar_multiply(typename alt_fl_type<FL>::FL d) const override {
        return make_shared<OpElement>(*this * (FL)d);
    }
    OpElement operator*(FL d) const {
        return OpElement(name, site_index, q_label, factor * d);
    }
    bool is_equal_to(const shared_ptr<OpExpr<S>> &other) const override {
        if (get_type() != other->get_type())
            return false;
        return *this == *dynamic_pointer_cast<OpElement>(other);
    }
    bool is_less_than(const shared_ptr<OpExpr<S>> &other) const override {
        return *this < *dynamic_pointer_cast<OpElement>(other);
    }
    bool operator==(const OpElement &other) const {
        return name == other.name && site_index == other.site_index &&
               std::abs(factor - other.factor) < 1E-12;
    }
    bool operator<(const OpElement &other) const {
        if (name != other.name)
            return name < other.name;
        else if (site_index != other.site_index)
            return site_index < other.site_index;
        else if (std::abs(factor - other.factor) >= 1E-12)
            return xreal<FL>(factor) != xreal<FL>(other.factor)
                       ? xreal<FL>(factor) < xreal<FL>(other.factor)
                       : ximag<FL>(factor) < ximag<FL>(other.factor);
        else
            return false;
    }
    size_t hash() const noexcept override {
        size_t h = (size_t)name;
        h ^= site_index.hash() + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= std::hash<FL>{}(factor) + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    string to_str() const override {
        stringstream ss;
        if (ximag<FL>(factor) != 0.0)
            ss << "(" << factor << " " << abs() << ")";
        else if (xreal<FL>(factor) != 1.0)
            ss << "(" << xreal<FL>(factor) << " " << abs() << ")";
        else if (site_index.data == 0)
            ss << name;
        else if (site_index.size() == 1 && site_index.spin_size() == 0)
            ss << name << (int)site_index[0];
        else
            ss << name << site_index;
        return ss.str();
    }
    string get_name() const override {
        stringstream ss;
        if (factor != (FL)1)
            ss << scientific << setprecision(6) << factor;
        ss << name << site_index.get_name() << endl;
        return ss.str();
    }
    shared_ptr<OpExpr<S>> plus(const shared_ptr<OpExpr<S>> &a,
                               const shared_ptr<OpExpr<S>> &b) const override {
        return plus_expr<S, FL>(a, b);
    }
    shared_ptr<OpExpr<S>>
    sum(const vector<shared_ptr<OpExpr<S>>> &x) const override {
        return sum_expr<S, FL>(x);
    }
    shared_ptr<OpExpr<S>>
    multiply(const shared_ptr<OpExpr<S>> &a,
             const shared_ptr<OpExpr<S>> &b) const override {
        return multiply_expr<S, FL>(a, b);
    }
    void save(ostream &ofs) const override {
        OpExpr<S>::save(ofs);
        ofs.write((char *)&name, sizeof(name));
        ofs.write((char *)&site_index, sizeof(site_index));
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&q_label, sizeof(q_label));
    }
    static shared_ptr<OpElement> load(istream &ifs) {
        OpNames name;
        SiteIndex site_index;
        FL factor;
        S q_label;
        ifs.read((char *)&name, sizeof(name));
        ifs.read((char *)&site_index, sizeof(site_index));
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&q_label, sizeof(q_label));
        return make_shared<OpElement>(name, site_index, q_label, factor);
    }
};

// Reference to original or transposed symbol: (A) or (A)^T
template <typename S, typename FL> struct OpElementRef : OpExpr<S> {
    shared_ptr<OpElement<S, FL>> op;
    int8_t factor;
    int8_t trans;
    OpElementRef(const shared_ptr<OpElement<S, FL>> &op, int8_t trans,
                 int8_t factor)
        : op(op), trans(trans), factor(factor) {}
    OpTypes get_type() const override { return OpTypes::ElemRef; }
    shared_ptr<OpExpr<S>> plus(const shared_ptr<OpExpr<S>> &a,
                               const shared_ptr<OpExpr<S>> &b) const override {
        return plus_expr<S, FL>(a, b);
    }
    shared_ptr<OpExpr<S>>
    sum(const vector<shared_ptr<OpExpr<S>>> &x) const override {
        return sum_expr<S, FL>(x);
    }
    shared_ptr<OpExpr<S>>
    multiply(const shared_ptr<OpExpr<S>> &a,
             const shared_ptr<OpExpr<S>> &b) const override {
        return multiply_expr<S, FL>(a, b);
    }
    void save(ostream &ofs) const override {
        OpExpr<S>::save(ofs);
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&trans, sizeof(trans));
        assert(op != nullptr);
        op->save(ofs);
    }
    static shared_ptr<OpElementRef> load(istream &ifs) {
        int8_t factor, trans;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&trans, sizeof(trans));
        shared_ptr<OpElement<S, FL>> op =
            dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs));
        return make_shared<OpElementRef>(op, trans, factor);
    }
};

// Tensor product of two operator symbols: (A) x (B)
// (conj & 1) indicates whether a is transposed
// (conj & 2) indicates whether b is transposed
template <typename S, typename FL> struct OpProduct : OpExpr<S> {
    shared_ptr<OpElement<S, FL>> a, b;
    FL factor;
    uint8_t conj;
    OpProduct(const shared_ptr<OpElement<S, FL>> &op, FL factor,
              uint8_t conj = 0)
        : factor(factor * op->factor),
          a(make_shared<OpElement<S, FL>>(op->abs())), b(nullptr), conj(conj) {}
    OpProduct(const shared_ptr<OpElement<S, FL>> &a,
              const shared_ptr<OpElement<S, FL>> &b, FL factor,
              uint8_t conj = 0)
        : factor(factor * (a == nullptr ? 1.0 : a->factor) *
                 (b == nullptr ? 1.0 : b->factor)),
          a(a == nullptr ? nullptr : make_shared<OpElement<S, FL>>(a->abs())),
          b(b == nullptr ? nullptr : make_shared<OpElement<S, FL>>(b->abs())),
          conj(conj) {}
    virtual OpTypes get_type() const override { return OpTypes::Prod; }
    bool is_normalized() const override { return factor == (FL)1; }
    shared_ptr<OpExpr<S>> abs_expr() const override {
        return make_shared<OpProduct>(abs());
    }
    OpProduct abs() const { return OpProduct(a, b, 1.0, conj); }
    // return abs value
    shared_ptr<OpElement<S, FL>> get_op() const {
        assert(b == nullptr);
        return a;
    }
    shared_ptr<OpExpr<S>> scalar_multiply(FL d) const override {
        return make_shared<OpProduct>(*this * d);
    }
    shared_ptr<OpExpr<S>>
    scalar_multiply(typename alt_fl_type<FL>::FL d) const override {
        return make_shared<OpProduct>(*this * (FL)d);
    }
    OpProduct operator*(FL d) const {
        return OpProduct(a, b, factor * d, conj);
    }
    bool is_equal_to(const shared_ptr<OpExpr<S>> &other) const override {
        if (get_type() != other->get_type())
            return false;
        return *this == *dynamic_pointer_cast<OpProduct>(other);
    }
    bool operator==(const OpProduct &other) const {
        return *a == *other.a &&
               (b == nullptr ? other.b == nullptr
                             : (other.b != nullptr && *b == *other.b)) &&
               std::abs(factor - other.factor) < 1E-12 && conj == other.conj;
    }
    shared_ptr<OpExpr<S>> plus(const shared_ptr<OpExpr<S>> &a,
                               const shared_ptr<OpExpr<S>> &b) const override {
        return plus_expr<S, FL>(a, b);
    }
    shared_ptr<OpExpr<S>>
    sum(const vector<shared_ptr<OpExpr<S>>> &x) const override {
        return sum_expr<S, FL>(x);
    }
    shared_ptr<OpExpr<S>>
    multiply(const shared_ptr<OpExpr<S>> &a,
             const shared_ptr<OpExpr<S>> &b) const override {
        return multiply_expr<S, FL>(a, b);
    }
    size_t hash() const noexcept override {
        size_t h = std::hash<FL>{}(factor);
        h ^= (a == nullptr ? 0 : a->hash()) + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= (b == nullptr ? 0 : b->hash()) + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= conj + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    string to_str() const override {
        stringstream ss;
        if (ximag<FL>(factor) != 0.0)
            ss << "(" << factor << " " << abs() << ")";
        else if (xreal<FL>(factor) != 1.0)
            ss << "(" << xreal<FL>(factor) << " " << abs() << ")";
        else {
            ss << *a << (conj & 1 ? "^T " : " ");
            if (b != nullptr)
                ss << *b << (conj & 2 ? "^T " : " ");
        }
        return ss.str();
    }
    void save(ostream &ofs) const override {
        OpExpr<S>::save(ofs);
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&conj, sizeof(conj));
        uint8_t has_ab =
            (uint8_t)((uint8_t)(a != nullptr) | ((b != nullptr) << 1));
        ofs.write((char *)&has_ab, sizeof(has_ab));
        if (has_ab & 1)
            a->save(ofs);
        if (has_ab & 2)
            b->save(ofs);
    }
    static shared_ptr<OpProduct> load(istream &ifs) {
        FL factor;
        uint8_t conj, has_ab;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&conj, sizeof(conj));
        ifs.read((char *)&has_ab, sizeof(has_ab));
        shared_ptr<OpElement<S, FL>> a =
            (has_ab & 1)
                ? dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs))
                : nullptr;
        shared_ptr<OpElement<S, FL>> b =
            (has_ab & 2)
                ? dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs))
                : nullptr;
        return make_shared<OpProduct>(a, b, factor, conj);
    }
};

// Tensor product of one symbol and a sum:
// (A) x {(B1) + (B2) + ...} or {(A1) + (A2) + ...} x (B)
// first element in conjs must be 0
// Optionally, c is the intermediate operator name for ops
// if a b != nullptr, this is A * B * C; ops = (A, B) or (B, C)
template <typename S, typename FL> struct OpSumProd : OpProduct<S, FL> {
    using OpProduct<S, FL>::factor;
    using OpProduct<S, FL>::a;
    using OpProduct<S, FL>::b;
    using OpProduct<S, FL>::conj;
    vector<shared_ptr<OpElement<S, FL>>> ops;
    shared_ptr<OpElement<S, FL>> c;
    vector<bool> conjs;
    OpSumProd(const shared_ptr<OpElement<S, FL>> &lop,
              const vector<shared_ptr<OpElement<S, FL>>> &ops,
              const vector<bool> &conjs, FL factor, uint8_t conj = 0,
              const shared_ptr<OpElement<S, FL>> &c = nullptr)
        : ops(ops), conjs(conjs),
          c(c), OpProduct<S, FL>(lop, nullptr, factor, conj) {}
    OpSumProd(const vector<shared_ptr<OpElement<S, FL>>> &ops,
              const shared_ptr<OpElement<S, FL>> &rop,
              const vector<bool> &conjs, FL factor, uint8_t conj = 0,
              const shared_ptr<OpElement<S, FL>> &c = nullptr)
        : ops(ops), conjs(conjs),
          c(c), OpProduct<S, FL>(nullptr, rop, factor, conj) {}
    OpSumProd(const shared_ptr<OpElement<S, FL>> &a,
              const shared_ptr<OpElement<S, FL>> &b,
              const vector<shared_ptr<OpElement<S, FL>>> &ops,
              const vector<bool> &conjs, FL factor, uint8_t conj = 0)
        : ops(ops), conjs(conjs), OpProduct<S, FL>(a, b, factor, conj) {}
    OpTypes get_type() const override { return OpTypes::SumProd; }
    shared_ptr<OpExpr<S>> abs_expr() const override {
        return make_shared<OpSumProd>(abs());
    }
    OpSumProd abs() const {
        if (OpProduct<S, FL>::a == nullptr)
            return OpSumProd(ops, OpProduct<S, FL>::b, conjs, 1.0,
                             OpProduct<S, FL>::conj, c);
        else if (OpProduct<S, FL>::b == nullptr)
            return OpSumProd(OpProduct<S, FL>::a, ops, conjs, 1.0,
                             OpProduct<S, FL>::conj, c);
        else
            return OpSumProd(OpProduct<S, FL>::a, OpProduct<S, FL>::b, ops,
                             conjs, 1.0, OpProduct<S, FL>::conj);
    }
    shared_ptr<OpExpr<S>> scalar_multiply(FL d) const override {
        return make_shared<OpSumProd>(*this * d);
    }
    shared_ptr<OpExpr<S>>
    scalar_multiply(typename alt_fl_type<FL>::FL d) const override {
        return make_shared<OpSumProd>(*this * (FL)d);
    }
    OpSumProd operator*(FL d) const {
        if (OpProduct<S, FL>::a == nullptr)
            return OpSumProd(ops, OpProduct<S, FL>::b, conjs,
                             OpProduct<S, FL>::factor * d,
                             OpProduct<S, FL>::conj, c);
        else if (OpProduct<S, FL>::b == nullptr)
            return OpSumProd(OpProduct<S, FL>::a, ops, conjs,
                             OpProduct<S, FL>::factor * d,
                             OpProduct<S, FL>::conj, c);
        else
            return OpSumProd(OpProduct<S, FL>::a, OpProduct<S, FL>::b, ops,
                             conjs, OpProduct<S, FL>::factor * d,
                             OpProduct<S, FL>::conj);
    }
    bool is_equal_to(const shared_ptr<OpExpr<S>> &other) const override {
        if (get_type() != other->get_type())
            return false;
        return *this == *dynamic_pointer_cast<OpSumProd>(other);
    }
    bool operator==(const OpSumProd &other) const {
        if (ops.size() != other.ops.size() ||
            (OpProduct<S, FL>::a == nullptr) != (other.a == nullptr) ||
            (OpProduct<S, FL>::b == nullptr) != (other.b == nullptr))
            return false;
        else if (OpProduct<S, FL>::a == nullptr &&
                 !(*OpProduct<S, FL>::b == *other.b))
            return false;
        else if (OpProduct<S, FL>::b == nullptr &&
                 !(*OpProduct<S, FL>::a == *other.a))
            return false;
        else if (OpProduct<S, FL>::conj != other.conj)
            return false;
        else if (conjs != other.conjs)
            return false;
        else if (std::abs(OpProduct<S, FL>::factor - other.factor) >= 1E-12)
            return false;
        else
            for (size_t i = 0; i < ops.size(); i++)
                if (!(*ops[i] == *other.ops[i]))
                    return false;
        return true;
    }
    shared_ptr<OpExpr<S>> plus(const shared_ptr<OpExpr<S>> &a,
                               const shared_ptr<OpExpr<S>> &b) const override {
        return plus_expr<S, FL>(a, b);
    }
    shared_ptr<OpExpr<S>>
    sum(const vector<shared_ptr<OpExpr<S>>> &x) const override {
        return sum_expr<S, FL>(x);
    }
    shared_ptr<OpExpr<S>>
    multiply(const shared_ptr<OpExpr<S>> &a,
             const shared_ptr<OpExpr<S>> &b) const override {
        return multiply_expr<S, FL>(a, b);
    }
    string to_str() const override {
        stringstream ss;
        if (ops.size() != 0) {
            if (ximag<FL>(factor) != 0.0)
                ss << "(" << factor << " ";
            else if (xreal<FL>(factor) != 1.0)
                ss << "(" << xreal<FL>(factor) << " ";
            if (a != nullptr)
                ss << *a << (conj & 1 ? "^T " : " ");
            if (c != nullptr)
                ss << "[[~ " << *c << " ]]";
            ss << "{ ";
            for (size_t i = 0; i < ops.size() - 1; i++)
                ss << *ops[i] << (conjs[i] ? "^T " : " ") << " + ";
            ss << *ops.back();
            ss << " }" << (conj & ((a != nullptr) + 1) ? "^T" : "");
            if (b != nullptr)
                ss << " " << *b << (conj & 2 ? "^T " : " ");
            if (ximag<FL>(factor) != 0.0 || xreal<FL>(factor) != 1.0)
                ss << " )";
        }
        return ss.str();
    }
    void save(ostream &ofs) const override {
        OpExpr<S>::save(ofs);
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&conj, sizeof(conj));
        uint8_t has_abc =
            (uint8_t)((uint8_t)(a != nullptr) | ((b != nullptr) << 1) |
                      ((c != nullptr) << 2));
        ofs.write((char *)&has_abc, sizeof(has_abc));
        if (has_abc & 1)
            a->save(ofs);
        if (has_abc & 2)
            b->save(ofs);
        if (has_abc & 4)
            c->save(ofs);
        assert(ops.size() == conjs.size());
        int sz = (int)ops.size();
        assert(ops.size() == (size_t)sz);
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++)
            ops[i]->save(ofs);
        for (int i = 0; i < sz; i++) {
            bool x = conjs[i];
            ofs.write((char *)&x, sizeof(x));
        }
    }
    static shared_ptr<OpSumProd> load(istream &ifs) {
        FL factor;
        uint8_t conj, has_abc;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&conj, sizeof(conj));
        ifs.read((char *)&has_abc, sizeof(has_abc));
        shared_ptr<OpElement<S, FL>> a =
            (has_abc & 1)
                ? dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs))
                : nullptr;
        shared_ptr<OpElement<S, FL>> b =
            (has_abc & 2)
                ? dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs))
                : nullptr;
        shared_ptr<OpElement<S, FL>> c =
            (has_abc & 4)
                ? dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs))
                : nullptr;
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        vector<shared_ptr<OpElement<S, FL>>> ops(sz);
        vector<bool> conjs(sz);
        for (int i = 0; i < sz; i++)
            ops[i] =
                dynamic_pointer_cast<OpElement<S, FL>>(load_expr<S, FL>(ifs));
        for (int i = 0; i < sz; i++) {
            bool x;
            ifs.read((char *)&x, sizeof(x));
            conjs[i] = x;
        }
        assert(a == nullptr || b == nullptr);
        return b == nullptr
                   ? make_shared<OpSumProd>(a, ops, conjs, factor, conj, c)
                   : make_shared<OpSumProd>(ops, b, conjs, factor, conj, c);
    }
};

// Sum of tensor products:
// (A) + (B) + (C) + ... or
// (A1) x (B1) + (A2) x (B2) + ... or
// (A1) x { (B1) + (B2) + ...} + (A2) x { {C1} + ... } + ...
template <typename S, typename FL> struct OpSum : OpExpr<S> {
    vector<shared_ptr<OpProduct<S, FL>>> strings;
    OpSum(const vector<shared_ptr<OpProduct<S, FL>>> &strings)
        : strings(strings) {}
    OpTypes get_type() const override { return OpTypes::Sum; }
    shared_ptr<OpExpr<S>> scalar_multiply(FL d) const override {
        return make_shared<OpSum>(*this * d);
    }
    shared_ptr<OpExpr<S>>
    scalar_multiply(typename alt_fl_type<FL>::FL d) const override {
        return make_shared<OpSum>(*this * (FL)d);
    }
    OpSum operator*(FL d) const {
        vector<shared_ptr<OpProduct<S, FL>>> strs;
        strs.reserve(strings.size());
        for (auto &r : strings)
            if (r->get_type() == OpTypes::Prod)
                strs.push_back(make_shared<OpProduct<S, FL>>(*r * d));
            else
                strs.push_back(make_shared<OpSumProd<S, FL>>(
                    *dynamic_pointer_cast<OpSumProd<S, FL>>(r) * d));
        return OpSum(strs);
    }
    bool is_equal_to(const shared_ptr<OpExpr<S>> &other) const override {
        if (get_type() != other->get_type())
            return false;
        return *this == *dynamic_pointer_cast<OpSum>(other);
    }
    bool operator==(const OpSum &other) const {
        if (strings.size() != other.strings.size())
            return false;
        for (size_t i = 0; i < strings.size(); i++)
            if (!(*strings[i] == *other.strings[i]))
                return false;
        return true;
    }
    shared_ptr<OpExpr<S>> plus(const shared_ptr<OpExpr<S>> &a,
                               const shared_ptr<OpExpr<S>> &b) const override {
        return plus_expr<S, FL>(a, b);
    }
    shared_ptr<OpExpr<S>>
    sum(const vector<shared_ptr<OpExpr<S>>> &x) const override {
        return sum_expr<S, FL>(x);
    }
    shared_ptr<OpExpr<S>>
    multiply(const shared_ptr<OpExpr<S>> &a,
             const shared_ptr<OpExpr<S>> &b) const override {
        return multiply_expr<S, FL>(a, b);
    }
    string to_str() const override {
        stringstream ss;
        if (strings.size() != 0) {
            for (size_t i = 0; i < strings.size() - 1; i++)
                ss << *strings[i] << " + ";
            ss << *strings.back();
        }
        return ss.str();
    }
    void save(ostream &ofs) const override {
        OpExpr<S>::save(ofs);
        int sz = (int)strings.size();
        assert((size_t)sz == strings.size());
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++)
            strings[i]->save(ofs);
    }
    static shared_ptr<OpSum> load(istream &ifs) {
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        vector<shared_ptr<OpProduct<S, FL>>> strings(sz);
        for (int i = 0; i < sz; i++)
            strings[i] =
                dynamic_pointer_cast<OpProduct<S, FL>>(load_expr<S, FL>(ifs));
        return make_shared<OpSum>(strings);
    }
};

// Reference to local or distributed sum
template <typename S> struct OpExprRef : OpExpr<S> {
    bool is_local;
    shared_ptr<OpExpr<S>> op, orig;
    OpExprRef(const shared_ptr<OpExpr<S>> &op, bool is_local,
              const shared_ptr<OpExpr<S>> &orig = nullptr)
        : op(op), is_local(is_local), orig(orig) {}
    OpTypes get_type() const override { return OpTypes::ExprRef; }
    bool is_normalized() const override {
        return op->is_normalized() && orig->is_normalized();
    }
    shared_ptr<OpExpr<S>> abs_expr() const override {
        return make_shared<OpExprRef>(abs());
    }
    OpExprRef abs() const {
        return OpExprRef(op->abs_expr(), is_local, orig->abs_expr());
    }
    shared_ptr<OpExpr<S>> scalar_multiply(float d) const override {
        return make_shared<OpExprRef>(*this * d);
    }
    shared_ptr<OpExpr<S>> scalar_multiply(complex<float> d) const override {
        return make_shared<OpExprRef>(*this * d);
    }
    shared_ptr<OpExpr<S>> scalar_multiply(double d) const override {
        return make_shared<OpExprRef>(*this * d);
    }
    shared_ptr<OpExpr<S>> scalar_multiply(complex<double> d) const override {
        return make_shared<OpExprRef>(*this * d);
    }
    template <typename FL> OpExprRef operator*(FL d) const {
        return OpExprRef(op->scalar_multiply(d), is_local,
                         orig != nullptr ? orig->scalar_multiply(d) : nullptr);
    }
    shared_ptr<OpExpr<S>> plus(const shared_ptr<OpExpr<S>> &a,
                               const shared_ptr<OpExpr<S>> &b) const override {
        return plus_expr_ref<S>(a, b);
    }
    shared_ptr<OpExpr<S>>
    multiply(const shared_ptr<OpExpr<S>> &a,
             const shared_ptr<OpExpr<S>> &b) const override {
        assert(false);
        return nullptr;
    }
    void save(ostream &ofs) const override {
        OpExpr<S>::save(ofs);
        ofs.write((char *)&is_local, sizeof(is_local));
        assert(op != nullptr);
        op->save(ofs);
        uint8_t has_orig = orig != nullptr;
        ofs.write((char *)&has_orig, sizeof(has_orig));
        if (has_orig & 1)
            orig->save(ofs);
    }
    template <typename FL> static shared_ptr<OpExprRef> load(istream &ifs) {
        bool is_local;
        ifs.read((char *)&is_local, sizeof(is_local));
        shared_ptr<OpExpr<S>> op = load_expr<S, FL>(ifs);
        uint8_t has_orig;
        ifs.read((char *)&has_orig, sizeof(has_orig));
        shared_ptr<OpExpr<S>> orig =
            (has_orig & 1) ? load_expr<S, FL>(ifs) : nullptr;
        return make_shared<OpExprRef>(op, is_local, orig);
    }
    string to_str() const override {
        stringstream ss;
        ss << "[" << (is_local ? "T" : "F") << "]" << op->to_str();
        return ss.str();
    }
};

// Counter used in npdm expectation to avoid explicit symbols
template <typename S> struct OpCounter : OpExpr<S> {
    uint64_t data;
    OpCounter(uint64_t data) : data(data) {}
    OpTypes get_type() const override { return OpTypes::Counter; }
};

template <typename S>
inline void save_expr(const shared_ptr<OpExpr<S>> &x, ostream &ofs) {
    x->save(ofs);
}

template <typename S, typename FL>
inline shared_ptr<OpExpr<S>> load_expr(istream &ifs) {
    OpTypes tp;
    ifs.read((char *)&tp, sizeof(tp));
    if (tp == OpTypes::Zero)
        return OpExpr<S>::load(ifs);
    else if (tp == OpTypes::Elem)
        return OpElement<S, FL>::load(ifs);
    else if (tp == OpTypes::Prod)
        return OpProduct<S, FL>::load(ifs);
    else if (tp == OpTypes::Sum)
        return OpSum<S, FL>::load(ifs);
    else if (tp == OpTypes::ElemRef)
        return OpElementRef<S, FL>::load(ifs);
    else if (tp == OpTypes::SumProd)
        return OpSumProd<S, FL>::load(ifs);
    else if (tp == OpTypes::ExprRef)
        return OpExprRef<S>::template load<FL>(ifs);
    else
        assert(false);
    return nullptr;
}

template <typename S> inline size_t hash_value(const shared_ptr<OpExpr<S>> &x) {
    if (x->get_type() == OpTypes::Elem || x->get_type() == OpTypes::Prod ||
        x->get_type() == OpTypes::Zero)
        return x->hash();
    else
        assert(false);
    return 0;
}

// Absolute value
template <typename S>
inline shared_ptr<OpExpr<S>> abs_value(const shared_ptr<OpExpr<S>> &x) {
    return x->is_normalized() ? x : x->abs_expr();
}

template <typename S>
inline bool operator==(const shared_ptr<OpExpr<S>> &a,
                       const shared_ptr<OpExpr<S>> &b) {
    return a->is_equal_to(b);
}

template <typename S>
inline bool operator!=(const shared_ptr<OpExpr<S>> &a,
                       const shared_ptr<OpExpr<S>> &b) {
    return !(a == b);
}

template <typename S> struct op_expr_less {
    bool operator()(const shared_ptr<OpExpr<S>> &a,
                    const shared_ptr<OpExpr<S>> &b) const {
        return a->is_less_than(b);
    }
};

// Sum of two symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>> operator+(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    return a->plus(a, b);
}

template <typename S>
inline const shared_ptr<OpExpr<S>>
plus_expr_ref(const shared_ptr<OpExpr<S>> &a, const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() == OpTypes::Zero)
        return b;
    else if (b->get_type() == OpTypes::Zero)
        return a;
    else if (a->get_type() == OpTypes::ExprRef &&
             b->get_type() == OpTypes::ExprRef) {
        bool is_local = dynamic_pointer_cast<OpExprRef<S>>(a)->is_local &&
                        dynamic_pointer_cast<OpExprRef<S>>(b)->is_local;
        shared_ptr<OpExpr<S>> op = dynamic_pointer_cast<OpExprRef<S>>(a)->op +
                                   dynamic_pointer_cast<OpExprRef<S>>(b)->op;
        shared_ptr<OpExpr<S>> orig =
            dynamic_pointer_cast<OpExprRef<S>>(a)->orig +
            dynamic_pointer_cast<OpExprRef<S>>(b)->orig;
        return make_shared<OpExprRef<S>>(op, is_local, orig);
    } else if (a->get_type() == OpTypes::ExprRef) {
        bool is_local = dynamic_pointer_cast<OpExprRef<S>>(a)->is_local;
        shared_ptr<OpExpr<S>> op =
            dynamic_pointer_cast<OpExprRef<S>>(a)->op + b;
        shared_ptr<OpExpr<S>> orig =
            dynamic_pointer_cast<OpExprRef<S>>(a)->orig + b;
        return make_shared<OpExprRef<S>>(op, is_local, orig);
    } else if (b->get_type() == OpTypes::ExprRef) {
        bool is_local = dynamic_pointer_cast<OpExprRef<S>>(b)->is_local;
        shared_ptr<OpExpr<S>> op =
            a + dynamic_pointer_cast<OpExprRef<S>>(b)->op;
        shared_ptr<OpExpr<S>> orig =
            a + dynamic_pointer_cast<OpExprRef<S>>(b)->orig;
        return make_shared<OpExprRef<S>>(op, is_local, orig);
    }
    assert(false);
    return nullptr;
}

template <typename S, typename FL>
inline const shared_ptr<OpExpr<S>> plus_expr(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() == OpTypes::Zero)
        return b;
    else if (b->get_type() == OpTypes::Zero)
        return a;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.push_back(make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(a), 1.0));
            strs.push_back(make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(b), 1.0));
            return make_shared<OpSum<S, FL>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.size() +
                         1);
            strs.push_back(make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(a), 1.0));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.end());
            return make_shared<OpSum<S, FL>>(strs);
        }
    } else if (a->get_type() == OpTypes::Prod ||
               a->get_type() == OpTypes::SumProd) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.size() +
                         1);
            strs.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(a));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.end());
            return make_shared<OpSum<S, FL>>(strs);
        } else if (b->get_type() == OpTypes::Prod ||
                   b->get_type() == OpTypes::SumProd) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(2);
            strs.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(a));
            strs.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(b));
            return make_shared<OpSum<S, FL>>(strs);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.size() +
                         1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.end());
            strs.push_back(make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(b), 1.0));
            return make_shared<OpSum<S, FL>>(strs);
        } else if (b->get_type() == OpTypes::Prod ||
                   b->get_type() == OpTypes::SumProd) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.size() +
                         1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.end());
            strs.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(b));
            return make_shared<OpSum<S, FL>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.size() +
                         dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.size());
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.end());
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.end());
            return make_shared<OpSum<S, FL>>(strs);
        }
    } else
        return plus_expr_ref<S>(a, b);
    assert(false);
    return nullptr;
}

template <typename S>
inline const shared_ptr<OpExpr<S>> operator+=(shared_ptr<OpExpr<S>> &a,
                                              const shared_ptr<OpExpr<S>> &b) {
    return a = a + b;
}

// A symbolic expression multiply a scalar
template <typename S, typename FL,
          typename = typename enable_if<is_complex<FL>::value ||
                                        is_floating_point<FL>::value>::type>
inline const shared_ptr<OpExpr<S>> operator*(const shared_ptr<OpExpr<S>> &x,
                                             FL d) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (d == (FL)0.0)
        return make_shared<OpExpr<S>>();
    else if (d == (FL)1.0)
        return x;
    else
        return ((*x).*((shared_ptr<OpExpr<S>>(OpExpr<S>::*)(FL) const) &
                       OpExpr<S>::scalar_multiply))(d);
}

// A scalar multiply a symbolic expression
template <typename S, typename FL,
          typename = typename enable_if<is_complex<FL>::value ||
                                        is_floating_point<FL>::value>::type>
inline const shared_ptr<OpExpr<S>> operator*(FL d,
                                             const shared_ptr<OpExpr<S>> &x) {
    return x * d;
}

// Tensor product of two symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    return a->multiply(a, b);
}

template <typename S, typename FL>
inline const shared_ptr<OpExpr<S>>
multiply_expr(const shared_ptr<OpExpr<S>> &a, const shared_ptr<OpExpr<S>> &b) {
    using OPSF = OpProduct<S, FL>;
    if (a->get_type() == OpTypes::Zero)
        return a;
    else if (b->get_type() == OpTypes::Zero)
        return b;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(b)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S, FL>>(b)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpProduct<S, FL>>(
                    dynamic_pointer_cast<OpElement<S, FL>>(a), r->a,
                    r->factor));
            }
            return make_shared<OpSum<S, FL>>(strs);
        } else if (b->get_type() == OpTypes::Elem)
            return make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(a),
                dynamic_pointer_cast<OpElement<S, FL>>(b), 1.0);
        else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OPSF>(b)->b == nullptr);
            return make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(a),
                dynamic_pointer_cast<OpProduct<S, FL>>(b)->a,
                dynamic_pointer_cast<OpProduct<S, FL>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Elem) {
            assert(dynamic_pointer_cast<OPSF>(a)->b == nullptr);
            return make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpProduct<S, FL>>(a)->a,
                dynamic_pointer_cast<OpElement<S, FL>>(b),
                dynamic_pointer_cast<OpProduct<S, FL>>(a)->factor);
        } else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OPSF>(a)->b == nullptr);
            assert(dynamic_pointer_cast<OPSF>(b)->b == nullptr);
            return make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpProduct<S, FL>>(a)->a,
                dynamic_pointer_cast<OpProduct<S, FL>>(b)->a,
                dynamic_pointer_cast<OpProduct<S, FL>>(a)->factor *
                    dynamic_pointer_cast<OpProduct<S, FL>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpProduct<S, FL>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(a)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S, FL>>(a)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpProduct<S, FL>>(
                    r->a, dynamic_pointer_cast<OpElement<S, FL>>(b),
                    r->factor));
            }
            return make_shared<OpSum<S, FL>>(strs);
        }
    }
    assert(false);
    return nullptr;
}

// Sum of several symbolic expressions
template <typename S, typename FL>
inline const shared_ptr<OpExpr<S>>
sum_expr(const vector<shared_ptr<OpExpr<S>>> &xs) {
    const static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
    vector<shared_ptr<OpProduct<S, FL>>> strs;
    for (auto &r : xs)
        if (r->get_type() == OpTypes::Zero)
            ;
        else if (r->get_type() == OpTypes::Prod)
            strs.push_back(dynamic_pointer_cast<OpProduct<S, FL>>(r));
        else if (r->get_type() == OpTypes::SumProd)
            strs.push_back(dynamic_pointer_cast<OpSumProd<S, FL>>(r));
        else if (r->get_type() == OpTypes::Elem)
            strs.push_back(make_shared<OpProduct<S, FL>>(
                dynamic_pointer_cast<OpElement<S, FL>>(r), 1.0));
        else if (r->get_type() == OpTypes::ElemRef)
            assert(false);
        else if (r->get_type() == OpTypes::Sum) {
            strs.reserve(dynamic_pointer_cast<OpSum<S, FL>>(r)->strings.size() +
                         strs.size());
            for (auto &rr : dynamic_pointer_cast<OpSum<S, FL>>(r)->strings)
                strs.push_back(rr);
        } else
            assert(false);
    return strs.size() != 0 ? make_shared<OpSum<S, FL>>(strs) : zero;
}

// Sum of several symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>>
sum(const vector<shared_ptr<OpExpr<S>>> &xs) {
    const static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
    for (auto &x : xs) {
        shared_ptr<OpExpr<S>> r = x->sum(xs);
        if (r != nullptr)
            return r;
    }
    return zero;
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
    return sum<S>(xs);
}

template <typename S>
inline ostream &operator<<(ostream &os, const shared_ptr<OpExpr<S>> &c) {
    os << *c;
    return os;
}

} // namespace block2

namespace std {

template <> struct hash<block2::OpNames> {
    size_t operator()(block2::OpNames s) const noexcept { return (size_t)s; }
};

template <typename FL> struct hash<complex<FL>> {
    size_t operator()(const complex<FL> &x) const noexcept {
        size_t h = hash<FL>{}(real(x));
        h ^= hash<FL>{}(imag(x)) + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
};

template <typename S, typename FL> struct hash<block2::OpElement<S, FL>> {
    size_t operator()(const block2::OpElement<S, FL> &s) const noexcept {
        return s.hash();
    }
};

template <typename S, typename FL> struct hash<block2::OpProduct<S, FL>> {
    size_t operator()(const block2::OpProduct<S, FL> &s) const noexcept {
        return s.hash();
    }
};

template <typename S> struct hash<shared_ptr<block2::OpExpr<S>>> {
    size_t operator()(const shared_ptr<block2::OpExpr<S>> &s) const noexcept {
        return hash_value(s);
    }
};

} // namespace std
