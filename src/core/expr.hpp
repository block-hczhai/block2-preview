
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
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace std;

namespace block2 {

enum struct ParallelTypes : uint8_t {
    Serial = 0,
    Distributed = 1,
    NewScheme = 2
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
    TEMP
};

inline ostream &operator<<(ostream &os, const OpNames c) {
    const static string repr[] = {
        "Zero", "H",   "I",   "N",   "NN",  "C",   "D",   "R",   "RD",   "A",
        "AD",   "P",   "PD",  "B",   "BD",  "Q",   "TR",  "TS",  "PDM1", "PDM2",
        "CCDD", "CCD", "CDC", "CDD", "DCC", "DCD", "DDC", "TEMP"};
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
    ExprRef
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
template <typename S> struct OpElement : OpExpr<S> {
    OpNames name;
    SiteIndex site_index;
    double factor;
    S q_label;
    OpElement(OpNames name, SiteIndex site_index, S q_label,
              double factor = 1.0)
        : name(name), site_index(site_index), factor(factor), q_label(q_label) {
    }
    OpTypes get_type() const override { return OpTypes::Elem; }
    OpElement abs() const { return OpElement(name, site_index, q_label, 1.0); }
    OpElement operator*(double d) const {
        return OpElement(name, site_index, q_label, factor * d);
    }
    bool operator==(const OpElement &other) const {
        return name == other.name && site_index == other.site_index &&
               ::abs(factor - other.factor) < 1E-12;
    }
    bool operator<(const OpElement &other) const {
        if (name != other.name)
            return name < other.name;
        else if (site_index != other.site_index)
            return site_index < other.site_index;
        else if (::abs(factor - other.factor) >= 1E-12)
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
    string get_name() const override {
        stringstream ss;
        if (factor != 1.0)
            ss << scientific << setprecision(6) << factor;
        ss << name << site_index.get_name() << endl;
        return ss.str();
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
    OpTypes get_type() const override { return OpTypes::ElemRef; }
};

// Tensor product of two operator symbols: (A) x (B)
// (conj & 1) indicates whether a is transposed
// (conj & 2) indicates whether b is transposed
template <typename S> struct OpProduct : OpExpr<S> {
    shared_ptr<OpElement<S>> a, b;
    double factor;
    uint8_t conj;
    OpProduct(const shared_ptr<OpElement<S>> &op, double factor,
              uint8_t conj = 0)
        : factor(factor * op->factor), a(make_shared<OpElement<S>>(op->abs())),
          b(nullptr), conj(conj) {}
    OpProduct(const shared_ptr<OpElement<S>> &a,
              const shared_ptr<OpElement<S>> &b, double factor,
              uint8_t conj = 0)
        : factor(factor * (a == nullptr ? 1.0 : a->factor) *
                 (b == nullptr ? 1.0 : b->factor)),
          a(a == nullptr ? nullptr : make_shared<OpElement<S>>(a->abs())),
          b(b == nullptr ? nullptr : make_shared<OpElement<S>>(b->abs())),
          conj(conj) {}
    virtual OpTypes get_type() const override { return OpTypes::Prod; }
    OpProduct abs() const { return OpProduct(a, b, 1.0, conj); }
    shared_ptr<OpElement<S>> get_op() const {
        assert(b == nullptr);
        return a;
    }
    OpProduct operator*(double d) const {
        return OpProduct(a, b, factor * d, conj);
    }
    bool operator==(const OpProduct &other) const {
        return *a == *other.a &&
               (b == nullptr ? other.b == nullptr
                             : (other.b != nullptr && *b == *other.b)) &&
               factor == other.factor && conj == other.conj;
    }
    size_t hash() const noexcept {
        size_t h = std::hash<double>{}(factor);
        h ^= (a == nullptr ? 0 : a->hash()) + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= (b == nullptr ? 0 : b->hash()) + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= conj + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    friend ostream &operator<<(ostream &os, const OpProduct<S> &c) {
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
// first element in conjs must be 0
// Optionally, c is the intermediate operator name for ops
// if a b != nullptr, this is A * B * C; ops = (A, B) or (B, C)
template <typename S> struct OpSumProd : OpProduct<S> {
    vector<shared_ptr<OpElement<S>>> ops;
    shared_ptr<OpElement<S>> c;
    vector<bool> conjs;
    OpSumProd(const shared_ptr<OpElement<S>> &lop,
              const vector<shared_ptr<OpElement<S>>> &ops,
              const vector<bool> &conjs, double factor, uint8_t conj = 0,
              const shared_ptr<OpElement<S>> &c = nullptr)
        : ops(ops), conjs(conjs),
          c(c), OpProduct<S>(lop, nullptr, factor, conj) {}
    OpSumProd(const vector<shared_ptr<OpElement<S>>> &ops,
              const shared_ptr<OpElement<S>> &rop, const vector<bool> &conjs,
              double factor, uint8_t conj = 0,
              const shared_ptr<OpElement<S>> &c = nullptr)
        : ops(ops), conjs(conjs),
          c(c), OpProduct<S>(nullptr, rop, factor, conj) {}
    OpSumProd(const shared_ptr<OpElement<S>> &a,
              const shared_ptr<OpElement<S>> &b,
              const vector<shared_ptr<OpElement<S>>> &ops,
              const vector<bool> &conjs, double factor, uint8_t conj = 0)
        : ops(ops), conjs(conjs), OpProduct<S>(a, b, factor, conj) {}
    OpTypes get_type() const override { return OpTypes::SumProd; }
    OpSumProd abs() const {
        if (OpProduct<S>::a == nullptr)
            return OpSumProd(ops, OpProduct<S>::b, conjs, 1.0,
                             OpProduct<S>::conj, c);
        else if (OpProduct<S>::b == nullptr)
            return OpSumProd(OpProduct<S>::a, ops, conjs, 1.0,
                             OpProduct<S>::conj, c);
        else
            return OpSumProd(OpProduct<S>::a, OpProduct<S>::b, ops, conjs, 1.0,
                             OpProduct<S>::conj);
    }
    OpSumProd operator*(double d) const {
        if (OpProduct<S>::a == nullptr)
            return OpSumProd(ops, OpProduct<S>::b, conjs,
                             OpProduct<S>::factor * d, OpProduct<S>::conj, c);
        else if (OpProduct<S>::b == nullptr)
            return OpSumProd(OpProduct<S>::a, ops, conjs,
                             OpProduct<S>::factor * d, OpProduct<S>::conj, c);
        else
            return OpSumProd(OpProduct<S>::a, OpProduct<S>::b, ops, conjs,
                             OpProduct<S>::factor * d, OpProduct<S>::conj);
    }
    bool operator==(const OpSumProd &other) const {
        if (ops.size() != other.ops.size() ||
            (OpProduct<S>::a == nullptr) != (other.a == nullptr) ||
            (OpProduct<S>::b == nullptr) != (other.b == nullptr))
            return false;
        else if (OpProduct<S>::a == nullptr && !(*OpProduct<S>::b == *other.b))
            return false;
        else if (OpProduct<S>::b == nullptr && !(*OpProduct<S>::a == *other.a))
            return false;
        else if (OpProduct<S>::conj != other.conj)
            return false;
        else if (conjs != other.conjs)
            return false;
        else if (OpProduct<S>::factor != other.factor)
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
            if (c.c != nullptr)
                os << "[[~ " << *c.c << " ]]";
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
    vector<shared_ptr<OpProduct<S>>> strings;
    OpSum(const vector<shared_ptr<OpProduct<S>>> &strings) : strings(strings) {}
    OpTypes get_type() const override { return OpTypes::Sum; }
    OpSum operator*(double d) const {
        vector<shared_ptr<OpProduct<S>>> strs;
        strs.reserve(strings.size());
        for (auto &r : strings)
            if (r->get_type() == OpTypes::Prod)
                strs.push_back(make_shared<OpProduct<S>>(*r * d));
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

// Reference to local or distributed sum
template <typename S> struct OpExprRef : OpExpr<S> {
    bool is_local;
    shared_ptr<OpExpr<S>> op, orig;
    OpExprRef(const shared_ptr<OpExpr<S>> &op, bool is_local,
              const shared_ptr<OpExpr<S>> &orig = nullptr)
        : op(op), is_local(is_local), orig(orig) {}
    OpTypes get_type() const override { return OpTypes::ExprRef; }
};

template <typename S>
inline void save_expr(const shared_ptr<OpExpr<S>> &x, ostream &ofs) {
    OpTypes tp = x->get_type();
    ofs.write((char *)&tp, sizeof(tp));
    if (tp == OpTypes::Zero)
        ;
    else if (tp == OpTypes::Elem) {
        shared_ptr<OpElement<S>> op = dynamic_pointer_cast<OpElement<S>>(x);
        ofs.write((char *)&op->name, sizeof(op->name));
        ofs.write((char *)&op->site_index, sizeof(op->site_index));
        ofs.write((char *)&op->factor, sizeof(op->factor));
        ofs.write((char *)&op->q_label, sizeof(op->q_label));
    } else if (tp == OpTypes::Prod) {
        shared_ptr<OpProduct<S>> op = dynamic_pointer_cast<OpProduct<S>>(x);
        ofs.write((char *)&op->factor, sizeof(op->factor));
        ofs.write((char *)&op->conj, sizeof(op->conj));
        uint8_t has_ab =
            (uint8_t)((uint8_t)(op->a != nullptr) | ((op->b != nullptr) << 1));
        ofs.write((char *)&has_ab, sizeof(has_ab));
        if (has_ab & 1)
            save_expr<S>(op->a, ofs);
        if (has_ab & 2)
            save_expr<S>(op->b, ofs);
    } else if (tp == OpTypes::Sum) {
        shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(x);
        int sz = (int)op->strings.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++)
            save_expr<S>(op->strings[i], ofs);
    } else if (tp == OpTypes::ElemRef) {
        shared_ptr<OpElementRef<S>> op =
            dynamic_pointer_cast<OpElementRef<S>>(x);
        ofs.write((char *)&op->factor, sizeof(op->factor));
        ofs.write((char *)&op->trans, sizeof(op->trans));
        assert(op->op != nullptr);
        save_expr<S>(op->op, ofs);
    } else if (tp == OpTypes::SumProd) {
        shared_ptr<OpSumProd<S>> op = dynamic_pointer_cast<OpSumProd<S>>(x);
        ofs.write((char *)&op->factor, sizeof(op->factor));
        ofs.write((char *)&op->conj, sizeof(op->conj));
        uint8_t has_abc =
            (uint8_t)((uint8_t)(op->a != nullptr) | ((op->b != nullptr) << 1) |
                      ((op->c != nullptr) << 2));
        ofs.write((char *)&has_abc, sizeof(has_abc));
        if (has_abc & 1)
            save_expr<S>(op->a, ofs);
        if (has_abc & 2)
            save_expr<S>(op->b, ofs);
        if (has_abc & 4)
            save_expr<S>(op->c, ofs);
        assert(op->ops.size() == op->conjs.size());
        int sz = (int)op->ops.size();
        ofs.write((char *)&sz, sizeof(sz));
        for (int i = 0; i < sz; i++)
            save_expr<S>(op->ops[i], ofs);
        for (int i = 0; i < sz; i++) {
            bool x = op->conjs[i];
            ofs.write((char *)&x, sizeof(x));
        }
    } else if (tp == OpTypes::ExprRef) {
        shared_ptr<OpExprRef<S>> op = dynamic_pointer_cast<OpExprRef<S>>(x);
        ofs.write((char *)&op->is_local, sizeof(op->is_local));
        assert(op->op != nullptr);
        save_expr<S>(op->op, ofs);
        uint8_t has_orig = op->orig != nullptr;
        ofs.write((char *)&has_orig, sizeof(has_orig));
        if (has_orig & 1)
            save_expr<S>(op->orig, ofs);
    } else
        assert(false);
}

template <typename S> inline shared_ptr<OpExpr<S>> load_expr(istream &ifs) {
    OpTypes tp;
    ifs.read((char *)&tp, sizeof(tp));
    if (tp == OpTypes::Zero)
        return make_shared<OpExpr<S>>();
    else if (tp == OpTypes::Elem) {
        OpNames name;
        SiteIndex site_index;
        double factor;
        S q_label;
        ifs.read((char *)&name, sizeof(name));
        ifs.read((char *)&site_index, sizeof(site_index));
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&q_label, sizeof(q_label));
        return make_shared<OpElement<S>>(name, site_index, q_label, factor);
    } else if (tp == OpTypes::Prod) {
        double factor;
        uint8_t conj, has_ab;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&conj, sizeof(conj));
        ifs.read((char *)&has_ab, sizeof(has_ab));
        shared_ptr<OpElement<S>> a =
            (has_ab & 1) ? dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs))
                         : nullptr;
        shared_ptr<OpElement<S>> b =
            (has_ab & 2) ? dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs))
                         : nullptr;
        return make_shared<OpProduct<S>>(a, b, factor, conj);
    } else if (tp == OpTypes::Sum) {
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        vector<shared_ptr<OpProduct<S>>> strings(sz);
        for (int i = 0; i < sz; i++)
            strings[i] = dynamic_pointer_cast<OpProduct<S>>(load_expr<S>(ifs));
        return make_shared<OpSum<S>>(strings);
    } else if (tp == OpTypes::ElemRef) {
        int8_t factor, trans;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&trans, sizeof(trans));
        shared_ptr<OpElement<S>> op =
            dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs));
        return make_shared<OpElementRef<S>>(op, trans, factor);
    } else if (tp == OpTypes::SumProd) {
        double factor;
        uint8_t conj, has_abc;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&conj, sizeof(conj));
        ifs.read((char *)&has_abc, sizeof(has_abc));
        shared_ptr<OpElement<S>> a =
            (has_abc & 1)
                ? dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs))
                : nullptr;
        shared_ptr<OpElement<S>> b =
            (has_abc & 2)
                ? dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs))
                : nullptr;
        shared_ptr<OpElement<S>> c =
            (has_abc & 4)
                ? dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs))
                : nullptr;
        int sz;
        ifs.read((char *)&sz, sizeof(sz));
        vector<shared_ptr<OpElement<S>>> ops(sz);
        vector<bool> conjs(sz);
        for (int i = 0; i < sz; i++)
            ops[i] = dynamic_pointer_cast<OpElement<S>>(load_expr<S>(ifs));
        for (int i = 0; i < sz; i++) {
            bool x;
            ifs.read((char *)&x, sizeof(x));
            conjs[i] = x;
        }
        assert(a == nullptr || b == nullptr);
        return b == nullptr
                   ? make_shared<OpSumProd<S>>(a, ops, conjs, factor, conj, c)
                   : make_shared<OpSumProd<S>>(ops, b, conjs, factor, conj, c);
    } else if (tp == OpTypes::ExprRef) {
        bool is_local;
        ifs.read((char *)&is_local, sizeof(is_local));
        shared_ptr<OpExpr<S>> op = load_expr<S>(ifs);
        uint8_t has_orig;
        ifs.read((char *)&has_orig, sizeof(has_orig));
        shared_ptr<OpExpr<S>> orig =
            (has_orig & 1) ? load_expr<S>(ifs) : nullptr;
        return make_shared<OpExprRef<S>>(op, is_local, orig);
    } else {
        assert(false);
        return nullptr;
    }
}

template <typename S> inline size_t hash_value(const shared_ptr<OpExpr<S>> &x) {
    if (x->get_type() == OpTypes::Elem)
        return dynamic_pointer_cast<OpElement<S>>(x)->hash();
    else if (x->get_type() == OpTypes::Prod)
        return dynamic_pointer_cast<OpProduct<S>>(x)->hash();
    else if (x->get_type() == OpTypes::Zero)
        return 0;
    else
        assert(false);
    return 0;
}

// Absolute value
template <typename S>
inline shared_ptr<OpExpr<S>> abs_value(const shared_ptr<OpExpr<S>> &x) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (x->get_type() == OpTypes::Elem) {
        shared_ptr<OpElement<S>> op = dynamic_pointer_cast<OpElement<S>>(x);
        return op->factor == 1.0 ? x : make_shared<OpElement<S>>(op->abs());
    } else if (x->get_type() == OpTypes::ExprRef) {
        shared_ptr<OpExprRef<S>> op = dynamic_pointer_cast<OpExprRef<S>>(x);
        return make_shared<OpExprRef<S>>(abs_value(op->op), op->is_local,
                                         abs_value(op->orig));
    } else {
        assert(x->get_type() == OpTypes::Prod);
        shared_ptr<OpProduct<S>> op = dynamic_pointer_cast<OpProduct<S>>(x);
        return op->factor == 1.0 ? x : make_shared<OpProduct<S>>(op->abs());
    }
}

// String representation
template <typename S> inline string to_str(const shared_ptr<OpExpr<S>> &x) {
    stringstream ss;
    if (x->get_type() == OpTypes::Zero)
        ss << 0;
    else if (x->get_type() == OpTypes::Elem)
        ss << *dynamic_pointer_cast<OpElement<S>>(x);
    else if (x->get_type() == OpTypes::Prod)
        ss << *dynamic_pointer_cast<OpProduct<S>>(x);
    else if (x->get_type() == OpTypes::Sum)
        ss << *dynamic_pointer_cast<OpSum<S>>(x);
    else if (x->get_type() == OpTypes::SumProd)
        ss << *dynamic_pointer_cast<OpSumProd<S>>(x);
    else if (x->get_type() == OpTypes::ExprRef)
        ss << "["
           << (dynamic_pointer_cast<OpExprRef<S>>(x)->is_local ? "T" : "F")
           << "]" << to_str(dynamic_pointer_cast<OpExprRef<S>>(x)->op);
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
        return *dynamic_pointer_cast<OpProduct<S>>(a) ==
               *dynamic_pointer_cast<OpProduct<S>>(b);
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

template <typename S>
inline bool operator!=(const shared_ptr<OpExpr<S>> &a,
                       const shared_ptr<OpExpr<S>> &b) {
    return !(a == b);
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
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.push_back(make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpElement<S>>(a), 1.0));
            strs.push_back(make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpElement<S>>(b), 1.0));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size() + 1);
            strs.push_back(make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpElement<S>>(a), 1.0));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        }
    } else if (a->get_type() == OpTypes::Prod ||
               a->get_type() == OpTypes::SumProd) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size() + 1);
            strs.push_back(dynamic_pointer_cast<OpProduct<S>>(a));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Prod ||
                   b->get_type() == OpTypes::SumProd) {
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(2);
            strs.push_back(dynamic_pointer_cast<OpProduct<S>>(a));
            strs.push_back(dynamic_pointer_cast<OpProduct<S>>(b));
            return make_shared<OpSum<S>>(strs);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() + 1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.push_back(make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpElement<S>>(b), 1.0));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Prod ||
                   b->get_type() == OpTypes::SumProd) {
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() + 1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.push_back(dynamic_pointer_cast<OpProduct<S>>(b));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpProduct<S>>> strs;
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
    } else if (a->get_type() == OpTypes::ExprRef &&
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
        return make_shared<OpProduct<S>>(
            *dynamic_pointer_cast<OpProduct<S>>(x) * d);
    else if (x->get_type() == OpTypes::Sum)
        return make_shared<OpSum<S>>(*dynamic_pointer_cast<OpSum<S>>(x) * d);
    else if (x->get_type() == OpTypes::ExprRef)
        return make_shared<OpExprRef<S>>(
            dynamic_pointer_cast<OpExprRef<S>>(x)->op * d,
            dynamic_pointer_cast<OpExprRef<S>>(x)->is_local,
            dynamic_pointer_cast<OpExprRef<S>>(x)->orig != nullptr
                ? dynamic_pointer_cast<OpExprRef<S>>(x)->orig * d
                : nullptr);
    assert(false);
    return nullptr;
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
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S>>(b)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpProduct<S>>(
                    dynamic_pointer_cast<OpElement<S>>(a), r->a, r->factor));
            }
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Elem)
            return make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpElement<S>>(a),
                dynamic_pointer_cast<OpElement<S>>(b), 1.0);
        else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OpProduct<S>>(b)->b == nullptr);
            return make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpElement<S>>(a),
                dynamic_pointer_cast<OpProduct<S>>(b)->a,
                dynamic_pointer_cast<OpProduct<S>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Elem) {
            assert(dynamic_pointer_cast<OpProduct<S>>(a)->b == nullptr);
            return make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpProduct<S>>(a)->a,
                dynamic_pointer_cast<OpElement<S>>(b),
                dynamic_pointer_cast<OpProduct<S>>(a)->factor);
        } else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OpProduct<S>>(a)->b == nullptr);
            assert(dynamic_pointer_cast<OpProduct<S>>(b)->b == nullptr);
            return make_shared<OpProduct<S>>(
                dynamic_pointer_cast<OpProduct<S>>(a)->a,
                dynamic_pointer_cast<OpProduct<S>>(b)->a,
                dynamic_pointer_cast<OpProduct<S>>(a)->factor *
                    dynamic_pointer_cast<OpProduct<S>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpProduct<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S>>(a)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpProduct<S>>(
                    r->a, dynamic_pointer_cast<OpElement<S>>(b), r->factor));
            }
            return make_shared<OpSum<S>>(strs);
        }
    }
    assert(false);
    return nullptr;
}

// Sum of several symbolic expressions
template <typename S>
inline const shared_ptr<OpExpr<S>>
sum(const vector<shared_ptr<OpExpr<S>>> &xs) {
    const static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
    vector<shared_ptr<OpProduct<S>>> strs;
    for (auto &r : xs)
        if (r->get_type() == OpTypes::Prod)
            strs.push_back(dynamic_pointer_cast<OpProduct<S>>(r));
        else if (r->get_type() == OpTypes::SumProd)
            strs.push_back(dynamic_pointer_cast<OpSumProd<S>>(r));
        else if (r->get_type() == OpTypes::Elem)
            strs.push_back(make_shared<OpProduct<S>>(
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

template <> struct hash<block2::OpNames> {
    size_t operator()(block2::OpNames s) const noexcept { return (size_t)s; }
};

template <typename S> struct hash<block2::OpElement<S>> {
    size_t operator()(const block2::OpElement<S> &s) const noexcept {
        return s.hash();
    }
};

template <typename S> struct hash<block2::OpProduct<S>> {
    size_t operator()(const block2::OpProduct<S> &s) const noexcept {
        return s.hash();
    }
};

template <typename S> struct hash<shared_ptr<block2::OpExpr<S>>> {
    size_t operator()(const shared_ptr<block2::OpExpr<S>> &s) const noexcept {
        return hash_value(s);
    }
};

} // namespace std
