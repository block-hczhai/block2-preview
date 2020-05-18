
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

#include <cstdint>
#include <sstream>
#include <string>

using namespace std;

namespace block2 {

// Quantum number with particle number, projected spin
// and point group irreducible representation (non-spin-adapted)
// (N: 8bits) - (0: 8bits) - (2S: 8bits) - (0: 5bits) - (pg: 3bits)
struct SZ {
    typedef void is_sz_t;
    uint32_t data;
    SZ() : data(0) {}
    SZ(uint32_t data) : data(data) {}
    SZ(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | ((uint8_t)twos << 8) | pg)) {}
    int n() const { return (int)(((int32_t)data) >> 24); }
    int twos() const { return (int)(int8_t)((data >> 8) & 0xFFU); }
    int pg() const { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data =
            (data & (~0xFFFF00U)) | ((uint32_t)(((uint8_t)twos & 0xFFU) << 8));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return 1; }
    bool is_fermion() const noexcept { return twos() & 1; }
    bool operator==(SZ other) const noexcept { return data == other.data; }
    bool operator!=(SZ other) const noexcept { return data != other.data; }
    bool operator<(SZ other) const noexcept { return data < other.data; }
    SZ operator-() const noexcept {
        return SZ((data & 0xFFU) | (((~data) + (1 << 8)) & 0xFF00U) |
                  (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SZ operator-(SZ other) const noexcept { return *this + (-other); }
    SZ operator+(SZ other) const noexcept {
        return SZ((((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) &
                   0xFF00FF00U) |
                  ((data & 0xFFU) ^ (other.data & 0xFFU)));
    }
    SZ operator[](int i) const noexcept { return *this; }
    SZ get_ket() const noexcept { return *this; }
    SZ get_bra(SZ dq) const noexcept { return *this + dq; }
    SZ combine(SZ bra, SZ ket) const {
        return ket + *this == bra ? ket : SZ(0xFFFFFFFFU);
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept { return 1; }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " SZ=";
        if (twos() & 1)
            ss << twos() << "/2";
        else
            ss << (twos() >> 1);
        ss << " PG=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SZ c) {
        os << c.to_str();
        return os;
    }
};

// Quantum number with particle number, SU(2) spin
// and point group irreducible representation (spin-adapted)
// (N: 8bits) - (2SL: 8bits) - (2S: 8bits) - (0: 5bits) - (pg: 3bits)
// (A) This can represent a single quantum number (2SL == 2S)
// (B) This can also represent a range of 'SU2' from addition two 'SU2':
//    the results of addition will have the same N and pg,
//    but new 2S = 2SL, 2SL + 2, ... , 2S
//      SU2::operator[] is for obtaining a specific value in the range
// (C) This can also represent two quantum numbers L and R with L = DQ + R
//    note that given DQ and R, the N and pg of L are unique but 2S is not
//    2SL can be used to store 2S of L. (N, 2S, pg) are those of R
//      SU2::get_bra(dq) and SU2::get_ket() are for extracting L and R
//      DQ::combine(L, R) is for compress two quantum number
struct SU2 {
    typedef void is_su2_t;
    uint32_t data;
    SU2() : data(0) {}
    SU2(uint32_t data) : data(data) {}
    SU2(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos << 16) | (twos << 8) | pg)) {}
    SU2(int n, int twos_low, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos_low << 16) | (twos << 8) | pg)) {}
    int n() const noexcept { return (int)(((int32_t)data) >> 24); }
    int twos() const noexcept { return (int)(int16_t)((data >> 8) & 0xFFU); }
    int twos_low() const noexcept {
        return (int)(int16_t)((data >> 16) & 0xFFU);
    }
    int pg() const noexcept { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data = (data & (~0xFFFF00U)) | ((uint32_t)((twos & 0xFFU) << 16)) |
               ((uint32_t)((twos & 0xFFU) << 8));
    }
    void set_twos_low(int twos) {
        data = (data & (~0xFF0000U)) | ((uint32_t)((twos & 0xFFU) << 16));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return twos() + 1; }
    bool is_fermion() const noexcept { return twos() & 1; }
    bool operator==(SU2 other) const noexcept { return data == other.data; }
    bool operator!=(SU2 other) const noexcept { return data != other.data; }
    bool operator<(SU2 other) const noexcept { return data < other.data; }
    SU2 operator-() const noexcept {
        return SU2((data & 0xFFFFFFU) | (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SU2 operator-(SU2 other) const noexcept { return *this + (-other); }
    SU2 operator+(SU2 other) const noexcept {
        uint32_t add_data =
            ((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) |
            ((data & 0xFFU) ^ (other.data & 0xFFU));
        uint32_t sub_data_lr =
            ((data & 0xFF00U) << 8) - (other.data & 0xFF0000U);
        sub_data_lr =
            (((sub_data_lr >> 7) + sub_data_lr) ^ (sub_data_lr >> 7)) &
            0xFF0000U;
        uint32_t sub_data_rl =
            ((other.data & 0xFF00U) << 8) - (data & 0xFF0000U);
        sub_data_rl =
            (((sub_data_rl >> 7) + sub_data_rl) ^ (sub_data_rl >> 7)) &
            0xFF0000U;
        return SU2(add_data | min(sub_data_lr, sub_data_rl));
    }
    SU2 operator[](int i) const noexcept {
        return SU2(((data + (i << 17)) & (~0x00FF00U)) |
                   (((data + (i << 17)) & 0xFF0000U) >> 8));
    }
    SU2 get_ket() const noexcept {
        return SU2((data & 0xFF00FFFFU) | ((data & 0xFF00U) << 8));
    }
    SU2 get_bra(SU2 dq) const noexcept {
        return SU2(((data & 0xFF000000U) + (dq.data & 0xFF000000U)) |
                   ((data & 0xFF0000U) >> 8) | (data & 0xFF0000U) |
                   ((data & 0xFFU) ^ (dq.data & 0xFFU)));
    }
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    SU2 combine(SU2 bra, SU2 ket) const {
        ket.set_twos_low(bra.twos());
        if (ket.get_bra(*this) != bra ||
            !triangle(ket.twos(), this->twos(), bra.twos()))
            return SU2(0xFFFFFFFFU);
        return ket;
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept {
        return (int)(((data >> 9) - (data >> 17)) & 0x7FU) + 1;
    }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " S=";
        if (twos_low() != twos()) {
            if (twos_low() & 1)
                ss << twos_low() << "/2~";
            else
                ss << (twos_low() >> 1) << "~";
        }
        if (twos() & 1)
            ss << twos() << "/2";
        else
            ss << (twos() >> 1);
        ss << " PG=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SU2 c) {
        os << c.to_str();
        return os;
    }
};

} // namespace block2

namespace std {

template <> struct hash<block2::SZ> {
    size_t operator()(const block2::SZ &s) const noexcept { return s.hash(); }
};

template <> struct less<block2::SZ> {
    bool operator()(const block2::SZ &lhs, const block2::SZ &rhs) const
        noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SZ &a, block2::SZ &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::SU2> {
    size_t operator()(const block2::SU2 &s) const noexcept { return s.hash(); }
};

template <> struct less<block2::SU2> {
    bool operator()(const block2::SU2 &lhs, const block2::SU2 &rhs) const
        noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SU2 &a, block2::SU2 &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

} // namespace std
