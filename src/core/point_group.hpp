
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

enum struct PGTypes : uint8_t { C1, C2, CI, CS, D2, C2V, C2H, D2H, NOPG };

// PYSCF convention
// https://sunqm.github.io/pyscf/symm.html (deleted)
// https://github.com/pyscf/pyscf/blob/1.7/doc_legacy/source/symm.rst
struct PointGroup {
    // D2H
    // 0   1   2   3   4   5   6   7   8   (FCIDUMP)
    //     A1g B3u B2u B1g B1u B2g B3g A1u
    // 0   1   2   3   4   5   6   7   (XOR)
    // A1g B1g B2g B3g A1u B1u B2u B3u
    static int16_t swap_d2h(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 7, 6, 1, 5, 2, 3, 4};
        return arr_swap[isym];
    }
    // C2V
    // 0  1  2  3  4  (FCIDUMP)
    //    A1 B1 B2 A2
    // 0  1  2  3  (XOR)
    // A1 A2 B1 B2
    static int16_t swap_c2v(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 2, 3, 1, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    // C2H
    // 0  1  2  3  4  (FCIDUMP)
    //    Ag Au Bu Bg
    // 0  1  2  3  (XOR)
    // Ag Bg Au Bu
    static int16_t swap_c2h(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 2, 3, 1, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    // D2
    // 0  1  2  3  4  (FCIDUMP)
    //    A1 B3 B2 B1
    // 0  1  2  3  (XOR)
    // A1 B1 B2 B3
    static int16_t swap_d2(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 3, 2, 1, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    // CS
    // 0  1  2   (FCIDUMP)
    //    A' A''
    // 0  1  (XOR)
    // A' A''
    static int16_t swap_cs(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 1, 8, 8, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    // C2
    // 0  1  2   (FCIDUMP)
    //    A  B
    // 0  1  (XOR)
    // A  B
    static int16_t swap_c2(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 1, 8, 8, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    // CI
    // 0  1  2   (FCIDUMP)
    //    Ag Au
    // 0  1  (XOR)
    // Ag Au
    static int16_t swap_ci(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 1, 8, 8, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    // C1
    // 0  1  (FCIDUMP)
    //    A
    // 0  (XOR)
    // A
    static int16_t swap_c1(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 8, 8, 8, 8, 8, 8, 8};
        return arr_swap[isym];
    }
    static int16_t swap_nopg(int16_t isym) {
        static int16_t arr_swap[] = {8, 0, 0, 0, 0, 0, 0, 0, 0};
        return arr_swap[isym];
    }
    static auto swap_pg(PGTypes pg) -> int16_t (*)(int16_t isym) {
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
        case PGTypes::NOPG:
            return swap_nopg;
        default:
            return swap_c1;
        }
    }
};

} // namespace block2
