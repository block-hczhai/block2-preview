
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

#include "core/allocator.hpp"
#include "core/archived_sparse_matrix.hpp"
#include "core/archived_tensor_functions.hpp"
#include "core/batch_gemm.hpp"
#include "core/cg.hpp"
#include "core/complex_matrix_functions.hpp"
#include "core/csr_matrix.hpp"
#include "core/csr_matrix_functions.hpp"
#include "core/csr_operator_functions.hpp"
#include "core/csr_sparse_matrix.hpp"
#include "core/delayed_sparse_matrix.hpp"
#include "core/delayed_tensor_functions.hpp"
#include "core/expr.hpp"
#include "core/fft.hpp"
#include "core/fp_codec.hpp"
#include "core/hamiltonian.hpp"
#include "core/hubbard.hpp"
#include "core/integral.hpp"
#include "core/integral_compressed.hpp"
#include "core/integral_dyall.hpp"
#include "core/matching.hpp"
#include "core/matrix.hpp"
#include "core/matrix_functions.hpp"
#include "core/operator_functions.hpp"
#include "core/operator_tensor.hpp"
#include "core/parallel_mpi.hpp"
#include "core/parallel_rule.hpp"
#include "core/parallel_tensor_functions.hpp"
#include "core/point_group.hpp"
#include "core/rule.hpp"
#include "core/sparse_matrix.hpp"
#include "core/state_info.hpp"
#include "core/symbolic.hpp"
#include "core/symmetry.hpp"
#include "core/tensor_functions.hpp"
#include "core/threading.hpp"
#include "core/utils.hpp"

#ifdef _EXPLICIT_TEMPLATE
#include "instantiation/block2_core.hpp"
#endif
