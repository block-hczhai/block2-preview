
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

#include <cstdint>
#include <sstream>
#include <string>

using namespace std;

namespace block2 {

// Quantum number with particle number, projected spin
// and point group irreducible representation (non-spin-adapted)
// N/2S = -128 ~ 127
// (N: 8bits) - (0: 8bits) - (2S: 8bits) - (0: 5bits) - (pg: 3bits)
struct SZShort {
    typedef void is_sz_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SZShort() : data(0) {}
    SZShort(uint32_t data) : data(data) {}
    SZShort(int n, int twos, int pg)
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
    bool operator==(SZShort other) const noexcept { return data == other.data; }
    bool operator!=(SZShort other) const noexcept { return data != other.data; }
    bool operator<(SZShort other) const noexcept { return data < other.data; }
    SZShort operator-() const noexcept {
        return SZShort((data & 0xFFU) | (((~data) + (1 << 8)) & 0xFF00U) |
                       (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SZShort operator-(SZShort other) const noexcept { return *this + (-other); }
    SZShort operator+(SZShort other) const noexcept {
        return SZShort((((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) &
                        0xFF00FF00U) |
                       ((data & 0xFFU) ^ (other.data & 0xFFU)));
    }
    SZShort operator[](int i) const noexcept { return *this; }
    SZShort get_ket() const noexcept { return *this; }
    SZShort get_bra(SZShort dq) const noexcept { return *this + dq; }
    SZShort combine(SZShort bra, SZShort ket) const {
        return ket + *this == bra ? ket : SZShort(invalid);
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
    friend ostream &operator<<(ostream &os, SZShort c) {
        os << c.to_str();
        return os;
    }
};

// Quantum number with particle number, projected spin
// and point group irreducible representation (non-spin-adapted)
// N and 2S must be of the same odd/even property (not checked)
// N/2S = -16384 ~ 16383
// (N: 14bits) - (2S: 14bits) - (fermion: 1bit) - (pg: 3bits)
struct SZLong {
    typedef void is_sz_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SZLong() : data(0) {}
    SZLong(uint32_t data) : data(data) {}
    SZLong(int n, int twos, int pg)
        : data((((uint32_t)n >> 1) << 18) |
               ((uint32_t)((twos & 0x7FFFU) << 3) | pg)) {}
    int n() const {
        return (int)(((((int32_t)data) >> 18) << 1) | ((data >> 3) & 1));
    }
    int twos() const { return (int)((int16_t)(data >> 2) >> 1); }
    int pg() const { return (int)(data & 0x7U); }
    void set_n(int n) {
        data = (data & 0x3FFF7U) | (((uint32_t)n >> 1) << 18) | ((n & 1) << 3);
    }
    void set_twos(int twos) {
        data = (data & 0xFFFC0007U) | ((uint32_t)((twos & 0x7FFFU) << 3));
    }
    void set_pg(int pg) { data = (data & (~0x7U)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return 1; }
    bool is_fermion() const noexcept { return (data >> 3) & 1; }
    bool operator==(SZLong other) const noexcept { return data == other.data; }
    bool operator!=(SZLong other) const noexcept { return data != other.data; }
    bool operator<(SZLong other) const noexcept { return data < other.data; }
    SZLong operator-() const noexcept {
        return SZLong((data & 0xFU) | (((~data) + (1 << 3)) & 0x3FFF8U) |
                      (((~data) + (((~data) & 0x8U) << 15)) & 0xFFFC0000U));
    }
    SZLong operator-(SZLong other) const noexcept { return *this + (-other); }
    SZLong operator+(SZLong other) const noexcept {
        return SZLong(
            ((data & 0xFFFC0000U) + (other.data & 0xFFFC0000U) +
             (((data & other.data) & 0x8U) << 15)) |
            (((data & 0x3FFF8U) + (other.data & 0x3FFF8U)) & 0x3FFF8U) |
            ((data ^ other.data) & 0xFU));
    }
    SZLong operator[](int i) const noexcept { return *this; }
    SZLong get_ket() const noexcept { return *this; }
    SZLong get_bra(SZLong dq) const noexcept { return *this + dq; }
    SZLong combine(SZLong bra, SZLong ket) const {
        return ket + *this == bra ? ket : SZLong(invalid);
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
    friend ostream &operator<<(ostream &os, SZLong c) {
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
struct SU2Short {
    typedef void is_su2_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SU2Short() : data(0) {}
    SU2Short(uint32_t data) : data(data) {}
    SU2Short(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos << 16) | (twos << 8) | pg)) {}
    SU2Short(int n, int twos_low, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos_low << 16) | (twos << 8) | pg)) {}
    int n() const noexcept { return (int)(((int32_t)data) >> 24); }
    int twos() const noexcept { return (int)(int16_t)((data >> 8) & 0xFFU); }
    int twos_low() const noexcept {
        return (int)(int16_t)((data >> 16) & 0xFFU);
    }
    int pg() const noexcept { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data = (data & (~0x00FF00U)) | ((uint32_t)((twos & 0xFFU) << 8));
    }
    void set_twos_low(int twos) {
        data = (data & (~0xFF0000U)) | ((uint32_t)((twos & 0xFFU) << 16));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return twos() + 1; }
    bool is_fermion() const noexcept { return twos() & 1; }
    bool operator==(SU2Short other) const noexcept {
        return data == other.data;
    }
    bool operator!=(SU2Short other) const noexcept {
        return data != other.data;
    }
    bool operator<(SU2Short other) const noexcept { return data < other.data; }
    SU2Short operator-() const noexcept {
        return SU2Short((data & 0xFFFFFFU) |
                        (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SU2Short operator-(SU2Short other) const noexcept {
        return *this + (-other);
    }
    SU2Short operator+(SU2Short other) const noexcept {
        uint32_t add_data =
            ((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) |
            ((data & 0xFFU) ^ (other.data & 0xFFU));
        uint32_t sub_data_lr =
            ((data & 0xFF00U) << 8) - (other.data & 0xFF0000U);
        uint32_t sub_data_rl =
            ((other.data & 0xFF00U) << 8) - (data & 0xFF0000U);
        return SU2Short(add_data | min(sub_data_lr, sub_data_rl));
    }
    SU2Short operator[](int i) const noexcept {
        return SU2Short(((data + (i << 17)) & (~0x00FF00U)) |
                        (((data + (i << 17)) & 0xFF0000U) >> 8));
    }
    SU2Short get_ket() const noexcept {
        return SU2Short((data & 0xFF00FFFFU) | ((data & 0xFF00U) << 8));
    }
    SU2Short get_bra(SU2Short dq) const noexcept {
        return SU2Short(((data & 0xFF000000U) + (dq.data & 0xFF000000U)) |
                        ((data & 0xFF0000U) >> 8) | (data & 0xFF0000U) |
                        ((data & 0xFFU) ^ (dq.data & 0xFFU)));
    }
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    SU2Short combine(SU2Short bra, SU2Short ket) const {
        ket.set_twos_low(bra.twos());
        if (ket.get_bra(*this) != bra ||
            !triangle(ket.twos(), this->twos(), bra.twos()))
            return SU2Short(invalid);
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
    friend ostream &operator<<(ostream &os, SU2Short c) {
        os << c.to_str();
        return os;
    }
};

// Quantum number with particle number, SU(2) spin
// and point group irreducible representation (spin-adapted)
// N and 2S must be of the same odd/even property (not checked)
// N = -1024 ~ 1023; 2S = 0 ~ 1023
// (2N: 10bits) - (SL: 9bits) - (S: 9bits) - (fermion: 1bit) - (pg: 3bits)
struct SU2Long {
    typedef void is_su2_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SU2Long() : data(0) {}
    SU2Long(uint32_t data) : data(data) {}
    SU2Long(int n, int twos, int pg)
        : data((uint32_t)(((n >> 1) << 22) | ((twos >> 1) << 13) | (twos << 3) |
                          pg)) {}
    SU2Long(int n, int twos_low, int twos, int pg)
        : data((uint32_t)(((n >> 1) << 22) | ((twos_low >> 1) << 13) |
                          (twos << 3) | pg)) {}
    int n() const noexcept {
        return (int)(((((int32_t)data) >> 22) << 1) | ((data >> 3) & 1));
    }
    int twos() const noexcept { return (int)((data >> 3) & 0x3FFU); }
    int twos_low() const noexcept {
        return (int)(((data >> 12) & 0x3FEU) | ((data >> 3) & 1));
    }
    int pg() const noexcept { return (int)(data & 0x7U); }
    void set_n(int n) {
        data = (data & 0x3FFFF7U) | (((uint32_t)n >> 1) << 22) | ((n & 1) << 3);
    }
    void set_twos(int twos) {
        data = (data & 0xFFFFE007U) | ((uint32_t)twos << 3);
    }
    void set_twos_low(int twos) {
        data = (data & 0xFFC01FF7U) | (((uint32_t)twos >> 1) << 13) |
               ((twos & 1) << 3);
    }
    void set_pg(int pg) { data = (data & (~0x7U)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return twos() + 1; }
    bool is_fermion() const noexcept { return (data >> 3) & 1; }
    bool operator==(SU2Long other) const noexcept { return data == other.data; }
    bool operator!=(SU2Long other) const noexcept { return data != other.data; }
    bool operator<(SU2Long other) const noexcept { return data < other.data; }
    SU2Long operator-() const noexcept {
        return SU2Long((data & 0x3FFFFFU) |
                       (((~data) + (((~data) & 0x8U) << 19)) & 0xFFC00000U));
    }
    SU2Long operator-(SU2Long other) const noexcept { return *this + (-other); }
    SU2Long operator+(SU2Long other) const noexcept {
        uint32_t add_data = ((data & 0xFFC01FF8U) + (other.data & 0xFFC01FF8U) +
                             (((data & other.data) & 0x8U) << 19)) |
                            ((data ^ other.data) & 0xFU);
        uint32_t sub_data_lr = ((data & 0x1FF0U) << 9) -
                               (other.data & 0x3FE000U) -
                               ((((~data) & other.data) & 0x8U) << 10);
        uint32_t sub_data_rl = ((other.data & 0x1FF0U) << 9) -
                               (data & 0x3FE000U) -
                               (((data & (~other.data)) & 0x8U) << 10);
        return SU2Long(add_data | min(sub_data_lr, sub_data_rl));
    }
    SU2Long operator[](int i) const noexcept {
        return SU2Long(((data + (i << 13)) & 0xFFFFE00FU) |
                       (((data + (i << 13)) & 0x3FE000U) >> 9));
    }
    SU2Long get_ket() const noexcept {
        return SU2Long((data & 0xFFC01FFFU) | ((data & 0x1FF0U) << 9));
    }
    SU2Long get_bra(SU2Long dq) const noexcept {
        return SU2Long((((data & 0xFFC00000U) + (dq.data & 0xFFC00000U) +
                         (((data & dq.data) & 0x8U) << 19)) |
                        ((data ^ dq.data) & 0xFU)) |
                       ((data & 0x3FE000U) >> 9) | (data & 0x3FE000U));
    }
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    SU2Long combine(SU2Long bra, SU2Long ket) const {
        ket.set_twos_low((bra.twos() & (~1)) | (int)ket.is_fermion());
        if (ket.get_bra(*this) != bra ||
            !triangle(ket.twos(), this->twos(), bra.twos()))
            return SU2Long(invalid);
        return ket;
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept {
        return (int)(((data >> 4) - (data >> 13)) & 0x1FFU) + 1;
    }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " S=";
        // for bra, the odd/even of bra is unknown
        if (twos_low() != twos()) {
            if (twos_low() & 1)
                ss << twos_low() << "/2?~";
            else
                ss << (twos_low() >> 1) << "?~";
        }
        if (twos() & 1)
            ss << twos() << "/2";
        else
            ss << (twos() >> 1);
        ss << " PG=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SU2Long c) {
        os << c.to_str();
        return os;
    }
};

typedef SZLong SZ;
typedef SU2Long SU2;

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
