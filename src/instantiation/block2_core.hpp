
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

#include "../core/allocator.hpp"
#include "../core/archived_sparse_matrix.hpp"
#include "../core/archived_tensor_functions.hpp"
#include "../core/cg.hpp"
#include "../core/csr_operator_functions.hpp"
#include "../core/csr_sparse_matrix.hpp"
#include "../core/delayed_sparse_matrix.hpp"
#include "../core/delayed_tensor_functions.hpp"
#include "../core/expr.hpp"
#include "../core/fft.hpp"
#include "../core/fp_codec.hpp"
#include "../core/hamiltonian.hpp"
#include "../core/operator_functions.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/parallel_mpi.hpp"
#include "../core/parallel_rule.hpp"
#include "../core/parallel_tensor_functions.hpp"
#include "../core/rule.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/state_info.hpp"
#include "../core/symbolic.hpp"
#include "../core/symmetry.hpp"
#include "../core/tensor_functions.hpp"
#include <cstdint>

// allocator.hpp
extern template struct block2::StackAllocator<uint32_t>;
extern template struct block2::StackAllocator<double>;

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZ>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZ>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2>;

// cg.hpp
extern template struct block2::CG<block2::SZ>;
extern template struct block2::CG<block2::SU2>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZ>;
extern template struct block2::CSROperatorFunctions<block2::SU2>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZ>;
extern template struct block2::CSRSparseMatrix<block2::SU2>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZ>;
extern template struct block2::DelayedSparseMatrix<block2::SU2>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, block2::SparseMatrix<block2::SZ>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, block2::SparseMatrix<block2::SU2>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, block2::CSRSparseMatrix<block2::SZ>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, block2::CSRSparseMatrix<block2::SU2>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZ>;
extern template struct block2::DelayedTensorFunctions<block2::SU2>;

// expr.hpp
extern template struct block2::OpExpr<block2::SZ>;
extern template struct block2::OpElement<block2::SZ>;
extern template struct block2::OpElementRef<block2::SZ>;
extern template struct block2::OpProduct<block2::SZ>;
extern template struct block2::OpSumProd<block2::SZ>;
extern template struct block2::OpSum<block2::SZ>;

extern template struct block2::OpExpr<block2::SU2>;
extern template struct block2::OpElement<block2::SU2>;
extern template struct block2::OpElementRef<block2::SU2>;
extern template struct block2::OpProduct<block2::SU2>;
extern template struct block2::OpSumProd<block2::SU2>;
extern template struct block2::OpSum<block2::SU2>;

// fft.hpp
extern template struct block2::FactorizedFFT<block2::RaderFFT<>, 2, 3, 5, 7,
                                             11>;
extern template struct block2::FactorizedFFT<block2::BluesteinFFT<>, 2, 3, 5, 7,
                                             11>;

// fp_codec.hpp
extern template struct block2::FPCodec<double>;
extern template struct block2::CompressedVector<double>;
extern template struct block2::CompressedVectorMT<double>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SZ>;
extern template struct block2::Hamiltonian<block2::SU2>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, block2::Hamiltonian<block2::SZ>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, block2::Hamiltonian<block2::SU2>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZ>;
extern template struct block2::OperatorFunctions<block2::SU2>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZ>;
extern template struct block2::DelayedOperatorTensor<block2::SZ>;

extern template struct block2::OperatorTensor<block2::SU2>;
extern template struct block2::DelayedOperatorTensor<block2::SU2>;

// parallel_mpi.hpp
#ifdef _HAS_MPI
extern template struct block2::MPICommunicator<block2::SZ>;
extern template struct block2::MPICommunicator<block2::SU2>;
#endif

// parallel_rule.hpp
extern template struct block2::ParallelCommunicator<block2::SZ>;
extern template struct block2::ParallelRule<block2::SZ>;

extern template struct block2::ParallelCommunicator<block2::SU2>;
extern template struct block2::ParallelRule<block2::SU2>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZ>;
extern template struct block2::ParallelTensorFunctions<block2::SU2>;

// rule.hpp
extern template struct block2::Rule<block2::SZ>;
extern template struct block2::NoTransposeRule<block2::SZ>;

extern template struct block2::Rule<block2::SU2>;
extern template struct block2::NoTransposeRule<block2::SU2>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrixInfo<block2::SZ>;
extern template struct block2::SparseMatrix<block2::SZ>;
extern template struct block2::SparseMatrixGroup<block2::SZ>;

extern template struct block2::SparseMatrixInfo<block2::SU2>;
extern template struct block2::SparseMatrix<block2::SU2>;
extern template struct block2::SparseMatrixGroup<block2::SU2>;

// state_info.hpp
extern template struct block2::StateInfo<block2::SZ>;
extern template struct block2::StateProbability<block2::SZ>;

extern template struct block2::StateInfo<block2::SU2>;
extern template struct block2::StateProbability<block2::SU2>;

extern template struct block2::TransStateInfo<block2::SZ, block2::SU2>;
extern template struct block2::TransStateInfo<block2::SU2, block2::SZ>;

// symbolic.hpp
extern template struct block2::Symbolic<block2::SZ>;
extern template struct block2::SymbolicRowVector<block2::SZ>;
extern template struct block2::SymbolicColumnVector<block2::SZ>;
extern template struct block2::SymbolicMatrix<block2::SZ>;

extern template struct block2::Symbolic<block2::SU2>;
extern template struct block2::SymbolicRowVector<block2::SU2>;
extern template struct block2::SymbolicColumnVector<block2::SU2>;
extern template struct block2::SymbolicMatrix<block2::SU2>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SZ>;
extern template struct block2::TensorFunctions<block2::SU2>;
