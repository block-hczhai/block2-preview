
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

template struct block2::SparseMatrixInfo<block2::SGF>;
template struct block2::SparseMatrix<block2::SGF, double>;
template struct block2::SparseMatrixGroup<block2::SGF, double>;

template struct block2::SparseMatrixInfo<block2::SGB>;
template struct block2::SparseMatrix<block2::SGB, double>;
template struct block2::SparseMatrixGroup<block2::SGB, double>;
