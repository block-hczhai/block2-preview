
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
    typedef uint8_t pg_t;
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
    static inline int pg_inv(int a) noexcept { return a; }
    static inline int pg_mul(int a, int b) noexcept { return a ^ b; }
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
    typedef uint8_t pg_t;
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
    static inline int pg_inv(int a) noexcept { return a; }
    static inline int pg_mul(int a, int b) noexcept { return a ^ b; }
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

// Quantum number with particle number, projected spin
// and point group irreducible representation (non-spin-adapted)
// N/2S = -32768 ~ 32767; PG = 0 ~ 65535
// (N: 16bits) - (0: 16bits) - (2S: 16bits) - (pg: 16bits)
struct SZLongLong {
    typedef void is_sz_t;
    typedef uint8_t pg_t;
    uint64_t data;
    // S(invalid) must have maximal particle number n
    const static uint64_t invalid = 0x7FFFFFFFFFFFFFFFULL;
    SZLongLong() : data(0) {}
    SZLongLong(uint64_t data) : data(data) {}
    SZLongLong(int n, int twos, int pg)
        : data((uint64_t)(((int64_t)n << 48) |
                          ((uint64_t)(twos & 0xFFFFULL) << 16) |
                          (uint64_t)(pg))) {}
    int n() const { return (int)(((int64_t)data) >> 48); }
    int twos() const { return (int)(int16_t)((data >> 16) & 0xFFFFULL); }
    int pg() const { return (int)(data & 0xFFFFULL); }
    void set_n(int n) {
        data = (data & 0xFFFFFFFFFFFFULL) | ((uint64_t)((int64_t)n << 48));
    }
    void set_twos(int twos) {
        data = (data & (~0xFFFFFFFF0000ULL)) |
               ((uint64_t)((uint64_t)(twos & 0xFFFFULL) << 16));
    }
    void set_pg(int pg) { data = (data & (~0xFFFFULL)) | ((uint64_t)pg); }
    int multiplicity() const noexcept { return 1; }
    bool is_fermion() const noexcept { return twos() & 1; }
    bool operator==(SZLongLong other) const noexcept {
        return data == other.data;
    }
    bool operator!=(SZLongLong other) const noexcept {
        return data != other.data;
    }
    bool operator<(SZLongLong other) const noexcept {
        return data < other.data;
    }
    SZLongLong operator-() const noexcept {
        return SZLongLong((data & 0xFFFFULL) |
                          (((~data) + (1ULL << 16)) & 0xFFFF0000ULL) |
                          (((~data) + (1ULL << 48)) & 0xFFFF000000000000ULL));
    }
    SZLongLong operator-(SZLongLong other) const noexcept {
        return *this + (-other);
    }
    SZLongLong operator+(SZLongLong other) const noexcept {
        return SZLongLong((((data & 0xFFFF0000FFFF0000ULL) +
                            (other.data & 0xFFFF0000FFFF0000ULL)) &
                           0xFFFF0000FFFF0000ULL) |
                          ((data ^ other.data) & 0xFFFFULL));
    }
    SZLongLong operator[](int i) const noexcept { return *this; }
    SZLongLong get_ket() const noexcept { return *this; }
    SZLongLong get_bra(SZLongLong dq) const noexcept { return *this + dq; }
    static inline int pg_inv(int a) noexcept { return a; }
    static inline int pg_mul(int a, int b) noexcept { return a ^ b; }
    SZLongLong combine(SZLongLong bra, SZLongLong ket) const {
        return ket + *this == bra ? ket : SZLongLong(invalid);
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
    friend ostream &operator<<(ostream &os, SZLongLong c) {
        os << c.to_str();
        return os;
    }
};

// Quantum number with particle number, projected spin
// and Lz spatial symmetry as point group symmetryn (non-spin-adapted)
// N and 2S must be of the same odd/even property (not checked)
// N/2S/LZ (PG) = -1024 ~ 1023
// (N: 10bits) - (2S: 10bits) - (fermion: 1bit) - (pg: 11bits)
struct SZLZ {
    typedef void is_sz_t;
    typedef int16_t pg_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SZLZ() : data(0) {}
    SZLZ(uint32_t data) : data(data) {}
    SZLZ(int n, int twos, int pg)
        : data((((uint32_t)n >> 1) << 22) |
               ((uint32_t)((twos & 0x7FFU) << 11) | (pg & 0x7FFU))) {}
    int n() const {
        return (int)(((((int32_t)data) >> 22) << 1) | ((data >> 11) & 1));
    }
    int twos() const { return (int)((int16_t)(data >> 6) >> 5); }
    int pg() const { return (int)((int16_t)(data << 5) >> 5); }
    void set_n(int n) {
        data =
            (data & 0x3FF7FFU) | (((uint32_t)n >> 1) << 22) | ((n & 1) << 11);
    }
    void set_twos(int twos) {
        data = (data & 0xFFC007FFU) | ((uint32_t)((twos & 0x7FFU) << 11));
    }
    void set_pg(int pg) {
        data = (data & (~0x7FFU)) | ((uint32_t)(pg & 0x7FFU));
    }
    int multiplicity() const noexcept { return 1; }
    bool is_fermion() const noexcept { return (data >> 11) & 1; }
    bool operator==(SZLZ other) const noexcept { return data == other.data; }
    bool operator!=(SZLZ other) const noexcept { return data != other.data; }
    bool operator<(SZLZ other) const noexcept { return data < other.data; }
    SZLZ operator-() const noexcept {
        return SZLZ((((~data) + (1 << 11)) & 0x3FF800U) |
                    (((~data) + (((~data) & 0x800U) << 11)) & 0xFFC00000U) |
                    (((~data) + 1) & 0x7FFU));
    }
    SZLZ operator-(SZLZ other) const noexcept { return *this + (-other); }
    SZLZ operator+(SZLZ other) const noexcept {
        return SZLZ(
            ((data & 0xFFC00000U) + (other.data & 0xFFC00000U) +
             (((data & other.data) & 0x800U) << 11)) |
            (((data & 0x3FF800U) + (other.data & 0x3FF800U)) & 0x3FF800U) |
            ((data + other.data) & 0x7FFU));
    }
    SZLZ operator[](int i) const noexcept { return *this; }
    SZLZ get_ket() const noexcept { return *this; }
    SZLZ get_bra(SZLZ dq) const noexcept { return *this + dq; }
    static inline int pg_inv(int a) noexcept { return -a; }
    static inline int pg_mul(int a, int b) noexcept { return a + b; }
    SZLZ combine(SZLZ bra, SZLZ ket) const {
        return ket + *this == bra ? ket : SZLZ(invalid);
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
        ss << " LZ=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SZLZ c) {
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
    typedef uint8_t pg_t;
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
    static inline int pg_inv(int a) noexcept { return a; }
    static inline int pg_mul(int a, int b) noexcept { return a ^ b; }
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
    typedef uint8_t pg_t;
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
    static inline int pg_inv(int a) noexcept { return a; }
    static inline int pg_mul(int a, int b) noexcept { return a ^ b; }
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

// Quantum number with particle number, SU(2) spin
// and point group irreducible representation (spin-adapted)
// N = -32768 ~ 32767; 2S/PG = 0 ~ 65535
// (N: 16bits) - (2S: 16bits) - (2S: 16bits) - (pg: 16bits)
struct SU2LongLong {
    typedef void is_su2_t;
    typedef uint8_t pg_t;
    uint64_t data;
    // S(invalid) must have maximal particle number n
    const static uint64_t invalid = 0x7FFFFFFFFFFFFFFFULL;
    SU2LongLong() : data(0) {}
    SU2LongLong(uint64_t data) : data(data) {}
    SU2LongLong(int n, int twos, int pg)
        : data((uint64_t)(((uint64_t)(int64_t)n << 48) |
                          ((uint64_t)twos << 32) | ((uint64_t)twos << 16) |
                          (uint64_t)(pg))) {}
    SU2LongLong(int n, int twos_low, int twos, int pg)
        : data((uint64_t)(((uint64_t)(int64_t)n << 48) |
                          ((uint64_t)twos_low << 32) | ((uint64_t)twos << 16) |
                          (uint64_t)(pg))) {}
    int n() const noexcept { return (int)(int16_t)(data >> 48); }
    int twos() const noexcept {
        return (int)(((uint64_t)data >> 16) & 0xFFFFULL);
    }
    int twos_low() const noexcept {
        return (int)(((uint64_t)data >> 32) & 0xFFFFULL);
    }
    int pg() const noexcept { return (int)(data & 0xFFFFULL); }
    void set_n(int n) {
        data = (data & 0xFFFFFFFFFFFFULL) | ((uint64_t)(int64_t)n << 48);
    }
    void set_twos(int twos) {
        data = (data & 0xFFFFFFFF0000FFFFULL) | ((uint64_t)twos << 16);
    }
    void set_twos_low(int twos) {
        data = (data & 0xFFFF0000FFFFFFFFULL) | ((uint64_t)twos << 32);
    }
    void set_pg(int pg) { data = (data & (~0xFFFFULL)) | ((uint64_t)pg); }
    int multiplicity() const noexcept { return twos() + 1; }
    bool is_fermion() const noexcept { return (data >> 16) & 1; }
    bool operator==(SU2LongLong other) const noexcept {
        return data == other.data;
    }
    bool operator!=(SU2LongLong other) const noexcept {
        return data != other.data;
    }
    bool operator<(SU2LongLong other) const noexcept {
        return data < other.data;
    }
    SU2LongLong operator-() const noexcept {
        return SU2LongLong(
            (data & 0xFFFFFFFFFFFFULL) |
            (((~data) + 0x1000000000000ULL) & 0xFFFF000000000000ULL));
    }
    SU2LongLong operator-(SU2LongLong other) const noexcept {
        return *this + (-other);
    }
    SU2LongLong operator+(SU2LongLong other) const noexcept {
        uint64_t add_data = (((data & 0xFFFF0000FFFF0000ULL) +
                              (other.data & 0xFFFF0000FFFF0000ULL)) &
                             0xFFFF0000FFFF0000ULL) |
                            ((data ^ other.data) & 0xFFFFULL);
        uint64_t sub_data_lr =
            ((data & 0xFFFF0000ULL) << 16) - (other.data & 0xFFFF00000000ULL);
        uint64_t sub_data_rl =
            ((other.data & 0xFFFF0000ULL) << 16) - (data & 0xFFFF00000000ULL);
        return SU2LongLong(add_data | min(sub_data_lr, sub_data_rl));
    }
    SU2LongLong operator[](int i) const noexcept {
        return SU2LongLong(
            ((data + ((uint64_t)i << 33)) & 0xFFFFFFFF0000FFFFULL) |
            (((data + ((uint64_t)i << 33)) & 0xFFFF00000000ULL) >> 16));
    }
    SU2LongLong get_ket() const noexcept {
        return SU2LongLong((data & 0xFFFF0000FFFFFFFFULL) |
                           ((data & 0xFFFF0000ULL) << 16));
    }
    SU2LongLong get_bra(SU2LongLong dq) const noexcept {
        return SU2LongLong(((data & 0xFFFF000000000000ULL) +
                            (dq.data & 0xFFFF000000000000ULL)) |
                           ((data ^ dq.data) & 0xFFFFULL) |
                           ((data & 0xFFFF00000000ULL) >> 16) |
                           (data & 0xFFFF00000000ULL));
    }
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    static inline int pg_inv(int a) noexcept { return a; }
    static inline int pg_mul(int a, int b) noexcept { return a ^ b; }
    SU2LongLong combine(SU2LongLong bra, SU2LongLong ket) const {
        ket.set_twos_low(bra.twos());
        if (ket.get_bra(*this) != bra ||
            !triangle(ket.twos(), this->twos(), bra.twos()))
            return SU2LongLong(invalid);
        return ket;
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept {
        return (int)(((data >> 17) - (data >> 33)) & 0x7FFFULL) + 1;
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
    friend ostream &operator<<(ostream &os, SU2LongLong c) {
        os << c.to_str();
        return os;
    }
};

// Quantum number with particle number, SU(2) spin
// and Lz spatial symmetry as point group symmetryn (spin-adapted)
// N and 2S must be of the same odd/even property (not checked)
// N = -256 ~ 255; 2S = 0 ~ 255; LZ (PG) = -256 ~ 255
// (2N: 8bits) - (SL: 7bits) - (S: 7bits) - (fermion: 1bit) - (pg: 9bits)
struct SU2LZ {
    typedef void is_su2_t;
    typedef int16_t pg_t;
    uint32_t data;
    // S(invalid) must have maximal particle number n
    const static uint32_t invalid = 0x7FFFFFFFU;
    SU2LZ() : data(0) {}
    SU2LZ(uint32_t data) : data(data) {}
    SU2LZ(int n, int twos, int pg)
        : data((uint32_t)(((n >> 1) << 24) | ((twos >> 1) << 17) | (twos << 9) |
                          (pg & 0x1FFU))) {}
    SU2LZ(int n, int twos_low, int twos, int pg)
        : data((uint32_t)(((n >> 1) << 24) | ((twos_low >> 1) << 17) |
                          (twos << 9) | (pg & 0x1FFU))) {}
    int n() const noexcept {
        return (int)(((((int32_t)data) >> 24) << 1) | ((data >> 9) & 1));
    }
    int twos() const noexcept { return (int)((data >> 9) & 0xFFU); }
    int twos_low() const noexcept {
        return (int)(((data >> 16) & 0xFEU) | ((data >> 9) & 1));
    }
    int pg() const noexcept { return (int)((int16_t)(data << 7) >> 7); }
    void set_n(int n) {
        data = (data & 0xFFFDFFU) | (((uint32_t)n >> 1) << 24) | ((n & 1) << 9);
    }
    void set_twos(int twos) {
        data = (data & 0xFFFE01FFU) | ((uint32_t)twos << 9);
    }
    void set_twos_low(int twos) {
        data = (data & 0xFF01FDFFU) | (((uint32_t)twos >> 1) << 17) |
               ((twos & 1) << 9);
    }
    void set_pg(int pg) {
        data = (data & (~0x1FFU)) | ((uint32_t)(pg & 0x1FFU));
    }
    int multiplicity() const noexcept { return twos() + 1; }
    bool is_fermion() const noexcept { return (data >> 9) & 1; }
    bool operator==(SU2LZ other) const noexcept { return data == other.data; }
    bool operator!=(SU2LZ other) const noexcept { return data != other.data; }
    bool operator<(SU2LZ other) const noexcept { return data < other.data; }
    SU2LZ operator-() const noexcept {
        return SU2LZ((data & 0xFFFE00U) |
                     (((~data) + (((~data) & 0x200U) << 15)) & 0xFF000000U) |
                     (((~data) + 1) & 0x1FFU));
    }
    SU2LZ operator-(SU2LZ other) const noexcept { return *this + (-other); }
    SU2LZ operator+(SU2LZ other) const noexcept {
        uint32_t add_data = ((data & 0xFF01FE00U) + (other.data & 0xFF01FE00U) +
                             (((data & other.data) & 0x200U) << 15)) |
                            ((data + other.data) & 0x1FFU);
        uint32_t sub_data_lr = ((data & 0x1FC00U) << 7) -
                               (other.data & 0xFE0000U) -
                               ((((~data) & other.data) & 0x200U) << 8);
        uint32_t sub_data_rl = ((other.data & 0x1FC00U) << 7) -
                               (data & 0xFE0000U) -
                               (((data & (~other.data)) & 0x200U) << 8);
        return SU2LZ(add_data | min(sub_data_lr, sub_data_rl));
    }
    SU2LZ operator[](int i) const noexcept {
        return SU2LZ(((data + (i << 17)) & 0xFFFE03FFU) |
                     (((data + (i << 17)) & 0xFE0000U) >> 7));
    }
    SU2LZ get_ket() const noexcept {
        return SU2LZ((data & 0xFF01FFFFU) | ((data & 0x1FC00U) << 7));
    }
    SU2LZ get_bra(SU2LZ dq) const noexcept {
        return SU2LZ(
            (((((data & 0xFF0001FFU) + (dq.data & 0xFF0001FFU)) & 0xFF0001FFU) +
              (((data & dq.data) & 0x200U) << 15)) |
             ((data ^ dq.data) & 0x200U)) |
            ((data & 0xFE0000U) >> 7) | (data & 0xFE0000U));
    }
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    static inline int pg_inv(int a) noexcept { return -a; }
    static inline int pg_mul(int a, int b) noexcept { return a + b; }
    SU2LZ combine(SU2LZ bra, SU2LZ ket) const {
        ket.set_twos_low((bra.twos() & (~1)) | (int)ket.is_fermion());
        if (ket.get_bra(*this) != bra ||
            !triangle(ket.twos(), this->twos(), bra.twos()))
            return SU2LZ(invalid);
        return ket;
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept {
        return (int)(((data >> 10) - (data >> 17)) & 0x7FU) + 1;
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
        ss << " LZ=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SU2LZ c) {
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
    bool operator()(const block2::SZ &lhs,
                    const block2::SZ &rhs) const noexcept {
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
    bool operator()(const block2::SU2 &lhs,
                    const block2::SU2 &rhs) const noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SU2 &a, block2::SU2 &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::SZLZ> {
    size_t operator()(const block2::SZLZ &s) const noexcept { return s.hash(); }
};

template <> struct less<block2::SZLZ> {
    bool operator()(const block2::SZLZ &lhs,
                    const block2::SZLZ &rhs) const noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SZLZ &a, block2::SZLZ &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::SU2LZ> {
    size_t operator()(const block2::SU2LZ &s) const noexcept {
        return s.hash();
    }
};

template <> struct less<block2::SU2LZ> {
    bool operator()(const block2::SU2LZ &lhs,
                    const block2::SU2LZ &rhs) const noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SU2LZ &a, block2::SU2LZ &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

} // namespace std
