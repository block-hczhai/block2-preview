
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

using namespace std;

namespace block2 {

enum struct PGTypes : uint8_t { C1, C2, CI, CS, D2, C2V, C2H, D2H };

// PYSCF convention
// https://sunqm.github.io/pyscf/symm.html
struct PointGroup {
    // D2H
    // 0   1   2   3   4   5   6   7   8   (FCIDUMP)
    //     A1g B3u B2u B1g B1u B2g B3g A1u
    // 0   1   2   3   4   5   6   7   (XOR)
    // A1g B1g B2g B3g A1u B1u B2u B3u
    static uint8_t swap_d2h(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 7, 6, 1, 5, 2, 3, 4};
        return arr_swap[isym];
    }
    // C2V
    // 0  1  2  3  4  (FCIDUMP)
    //    A1 B1 B2 A2
    // 0  1  2  3  (XOR)
    // A1 A2 B1 B2
    static uint8_t swap_c2v(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 2, 3, 1};
        return arr_swap[isym];
    }
    // C2H
    // 0  1  2  3  4  (FCIDUMP)
    //    Ag Au Bu Bg
    // 0  1  2  3  (XOR)
    // Ag Bg Au Bu
    static uint8_t swap_c2h(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 2, 3, 1};
        return arr_swap[isym];
    }
    // D2
    // 0  1  2  3  4  (FCIDUMP)
    //    A1 B3 B2 B1
    // 0  1  2  3  (XOR)
    // A1 B1 B2 B3
    static uint8_t swap_d2(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 3, 2, 1};
        return arr_swap[isym];
    }
    // CS
    // 0  1  2   (FCIDUMP)
    //    A' A''
    // 0  1  (XOR)
    // A' A''
    static uint8_t swap_cs(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 1};
        return arr_swap[isym];
    }
    // C2
    // 0  1  2   (FCIDUMP)
    //    A  B
    // 0  1  (XOR)
    // A  B
    static uint8_t swap_c2(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 1};
        return arr_swap[isym];
    }
    // CI
    // 0  1  2   (FCIDUMP)
    //    Ag Au
    // 0  1  (XOR)
    // Ag Au
    static uint8_t swap_ci(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 1};
        return arr_swap[isym];
    }
    // C1
    // 0  1  (FCIDUMP)
    //    A
    // 0  (XOR)
    // A
    static uint8_t swap_c1(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0};
        return arr_swap[isym];
    }
    static auto swap_pg(PGTypes pg) -> uint8_t (*)(uint8_t isym) {
        switch (pg) {
        case PGTypes::C1:
            return swap_c1;
        case PGTypes::C2:
            return swap_c2;
        case PGTypes::CI:
            return swap_ci;
        case PGTypes::CS:
            return swap_cs;
        case PGTypes::D2:
            return swap_d2;
        case PGTypes::C2V:
            return swap_c2v;
        case PGTypes::C2H:
            return swap_c2h;
        case PGTypes::D2H:
            return swap_d2h;
        default:
            return swap_c1;
        }
    }
};

} // namespace block2
