
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

template struct block2::Allocator<uint32_t>;
template struct block2::StackAllocator<uint32_t>;
template struct block2::VectorAllocator<uint32_t>;

template struct block2::Allocator<double>;
template struct block2::StackAllocator<double>;
template struct block2::VectorAllocator<double>;
template struct block2::TemporaryAllocator<double>;
template struct block2::DataFrame<double>;
