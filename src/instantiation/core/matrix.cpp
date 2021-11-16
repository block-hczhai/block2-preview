
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

#include "../block2_core.hpp"

template struct block2::GMatrix<double>;
template struct block2::GDiagonalMatrix<double>;
template struct block2::GIdentityMatrix<double>;
template struct block2::GTensor<double>;

template struct block2::GMatrix<complex<double>>;
template struct block2::GDiagonalMatrix<complex<double>>;
template struct block2::GTensor<complex<double>>;
