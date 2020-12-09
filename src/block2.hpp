
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

#include "block2/allocator.hpp"
#include "block2/ancilla.hpp"
#include "block2/archived_mpo.hpp"
#include "block2/archived_sparse_matrix.hpp"
#include "block2/archived_tensor_functions.hpp"
#include "block2/batch_gemm.hpp"
#include "block2/cg.hpp"
#include "block2/csr_matrix.hpp"
#include "block2/csr_matrix_functions.hpp"
#include "block2/csr_operator_functions.hpp"
#include "block2/csr_sparse_matrix.hpp"
#include "block2/delayed_sparse_matrix.hpp"
#include "block2/delayed_tensor_functions.hpp"
#include "block2/determinant.hpp"
#include "block2/expr.hpp"
#include "block2/hamiltonian.hpp"
#include "block2/hubbard.hpp"
#include "block2/integral.hpp"
#include "block2/matrix.hpp"
#include "block2/matrix_functions.hpp"
#include "block2/moving_environment.hpp"
#include "block2/mpo.hpp"
#include "block2/mpo_fusing.hpp"
#include "block2/mpo_simplification.hpp"
#include "block2/mps.hpp"
#include "block2/mps_unfused.hpp"
#include "block2/operator_functions.hpp"
#include "block2/operator_tensor.hpp"
#include "block2/parallel_mpi.hpp"
#include "block2/parallel_mpo.hpp"
#include "block2/parallel_rule_sum_mpo.hpp"
#include "block2/parallel_rule.hpp"
#include "block2/parallel_sweep.hpp"
#include "block2/parallel_tensor_functions.hpp"
#include "block2/partition.hpp"
#include "block2/point_group.hpp"
#include "block2/qc_hamiltonian.hpp"
#include "block2/qc_mpo.hpp"
#include "block2/qc_ncorr.hpp"
#include "block2/qc_parallel_rule.hpp"
#include "block2/qc_pdm1.hpp"
#include "block2/qc_pdm2.hpp"
#include "block2/qc_rule.hpp"
#include "block2/qc_sum_mpo.hpp"
#include "block2/rule.hpp"
#include "block2/sparse_matrix.hpp"
#include "block2/state_averaged.hpp"
#include "block2/state_info.hpp"
#include "block2/sweep_algorithm_td.hpp"
#include "block2/sweep_algorithm.hpp"
#include "block2/symbolic.hpp"
#include "block2/symmetry.hpp"
#include "block2/tensor_functions.hpp"
#include "block2/threading.hpp"
#include "block2/utils.hpp"

#ifdef _USE_SCI
#include "sci/abstract_sci_wrapper.hpp"
#include "sci/hamiltonian_sci.hpp"
#include "sci/qc_hamiltonian_sci.hpp"
#include "sci/qc_mpo_sci.hpp"
#include "sci/sweep_algorithm_sci.hpp"
#endif

#ifdef _EXPLICIT_TEMPLATE
#include "instantiation/instantiation.hpp"
#endif

#undef ialloc
#undef dalloc
#undef frame
#undef threading
