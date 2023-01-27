
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
#include "../core/batch_gemm.hpp"
#include "../core/cg.hpp"
#include "../core/csr_matrix.hpp"
#include "../core/csr_matrix_functions.hpp"
#include "../core/csr_operator_functions.hpp"
#include "../core/csr_sparse_matrix.hpp"
#include "../core/delayed_sparse_matrix.hpp"
#include "../core/delayed_tensor_functions.hpp"
#include "../core/expr.hpp"
#include "../core/fft.hpp"
#include "../core/fp_codec.hpp"
#include "../core/hamiltonian.hpp"
#include "../core/integral.hpp"
#include "../core/iterative_matrix_functions.hpp"
#include "../core/matrix.hpp"
#include "../core/matrix_functions.hpp"
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
extern template struct block2::Allocator<uint32_t>;
extern template struct block2::StackAllocator<uint32_t>;
extern template struct block2::VectorAllocator<uint32_t>;

extern template struct block2::Allocator<double>;
extern template struct block2::StackAllocator<double>;
extern template struct block2::VectorAllocator<double>;
extern template struct block2::TemporaryAllocator<double>;
extern template struct block2::DataFrame<double>;

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZ, double>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2, double>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZ, double>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2, double>;

// batch_gemm.hpp
extern template struct block2::BatchGEMM<double>;
extern template struct block2::BatchGEMMRef<double>;
extern template struct block2::AdvancedGEMM<double>;
extern template struct block2::BatchGEMMSeq<double>;

extern template struct block2::BatchGEMM<complex<double>>;
extern template struct block2::BatchGEMMRef<complex<double>>;
extern template struct block2::AdvancedGEMM<complex<double>>;
extern template struct block2::BatchGEMMSeq<complex<double>>;

// cg.hpp
extern template struct block2::CG<block2::SZ>;
extern template struct block2::CG<block2::SU2>;

// csr_matrix.hpp
extern template struct block2::GCSRMatrix<double>;
extern template struct block2::GCSRMatrix<complex<double>>;

// csr_matrix_functions.hpp
extern template struct block2::GCSRMatrixFunctions<double>;
extern template struct block2::GCSRMatrixFunctions<complex<double>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZ, double>;
extern template struct block2::CSROperatorFunctions<block2::SU2, double>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZ, double>;
extern template struct block2::CSRSparseMatrix<block2::SU2, double>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZ, double>;
extern template struct block2::DelayedSparseMatrix<block2::SU2, double>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, double, block2::SparseMatrix<block2::SZ, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, double, block2::SparseMatrix<block2::SU2, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, double, block2::CSRSparseMatrix<block2::SZ, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, double, block2::CSRSparseMatrix<block2::SU2, double>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZ, double>;
extern template struct block2::DelayedTensorFunctions<block2::SU2, double>;

// expr.hpp
extern template struct block2::OpExpr<block2::SZ>;
extern template struct block2::OpExprRef<block2::SZ>;
extern template struct block2::OpCounter<block2::SZ>;
extern template struct block2::OpElement<block2::SZ, double>;
extern template struct block2::OpElementRef<block2::SZ, double>;
extern template struct block2::OpProduct<block2::SZ, double>;
extern template struct block2::OpSumProd<block2::SZ, double>;
extern template struct block2::OpSum<block2::SZ, double>;

extern template struct block2::OpExpr<block2::SU2>;
extern template struct block2::OpExprRef<block2::SU2>;
extern template struct block2::OpCounter<block2::SU2>;
extern template struct block2::OpElement<block2::SU2, double>;
extern template struct block2::OpElementRef<block2::SU2, double>;
extern template struct block2::OpProduct<block2::SU2, double>;
extern template struct block2::OpSumProd<block2::SU2, double>;
extern template struct block2::OpSum<block2::SU2, double>;

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
extern template struct block2::SiteBasis<block2::SZ>;
extern template struct block2::SiteBasis<block2::SU2>;
extern template struct block2::Hamiltonian<block2::SZ, double>;
extern template struct block2::Hamiltonian<block2::SU2, double>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, double, block2::Hamiltonian<block2::SZ, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, double, block2::Hamiltonian<block2::SU2, double>>;

// integral.hpp
extern template struct block2::FCIDUMP<double>;
extern template struct block2::SpinOrbitalFCIDUMP<double>;
extern template struct block2::MRCISFCIDUMP<double>;

// iterative_matrix_functions.hpp
extern template struct block2::IterativeMatrixFunctions<double>;
extern template struct block2::IterativeMatrixFunctions<complex<double>>;

// matrix.hpp
extern template struct block2::GMatrix<double>;
extern template struct block2::GDiagonalMatrix<double>;
extern template struct block2::GIdentityMatrix<double>;
extern template struct block2::GTensor<double>;

extern template struct block2::GMatrix<complex<double>>;
extern template struct block2::GDiagonalMatrix<complex<double>>;
extern template struct block2::GTensor<complex<double>>;

// matrix_functions.hpp
extern template struct block2::GMatrixFunctions<double>;
extern template struct block2::GMatrixFunctions<complex<double>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZ, double>;
extern template struct block2::OperatorFunctions<block2::SU2, double>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZ, double>;
extern template struct block2::DelayedOperatorTensor<block2::SZ, double>;

extern template struct block2::OperatorTensor<block2::SU2, double>;
extern template struct block2::DelayedOperatorTensor<block2::SU2, double>;

// parallel_mpi.hpp
#ifdef _HAS_MPI
extern template struct block2::MPICommunicator<block2::SZ>;
extern template struct block2::MPICommunicator<block2::SU2>;
#endif

// parallel_rule.hpp
extern template struct block2::ParallelCommunicator<block2::SZ>;
extern template struct block2::ParallelRule<block2::SZ>;
extern template struct block2::ParallelRule<block2::SZ, double>;

extern template struct block2::ParallelCommunicator<block2::SU2>;
extern template struct block2::ParallelRule<block2::SU2>;
extern template struct block2::ParallelRule<block2::SU2, double>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZ, double>;
extern template struct block2::ParallelTensorFunctions<block2::SU2, double>;

// rule.hpp
extern template struct block2::Rule<block2::SZ, double>;
extern template struct block2::NoTransposeRule<block2::SZ, double>;

extern template struct block2::Rule<block2::SU2, double>;
extern template struct block2::NoTransposeRule<block2::SU2, double>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrixInfo<block2::SZ>;
extern template struct block2::SparseMatrix<block2::SZ, double>;
extern template struct block2::SparseMatrixGroup<block2::SZ, double>;

extern template struct block2::SparseMatrixInfo<block2::SU2>;
extern template struct block2::SparseMatrix<block2::SU2, double>;
extern template struct block2::SparseMatrixGroup<block2::SU2, double>;

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
extern template struct block2::TensorFunctions<block2::SZ, double>;
extern template struct block2::TensorFunctions<block2::SU2, double>;

#ifdef _USE_KSYMM

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZK, double>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2K, double>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZK, double>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2K, double>;

// cg.hpp
extern template struct block2::CG<block2::SZK>;
extern template struct block2::CG<block2::SU2K>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZK, double>;
extern template struct block2::CSROperatorFunctions<block2::SU2K, double>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZK, double>;
extern template struct block2::CSRSparseMatrix<block2::SU2K, double>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZK, double>;
extern template struct block2::DelayedSparseMatrix<block2::SU2K, double>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZK, double, block2::SparseMatrix<block2::SZK, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2K, double, block2::SparseMatrix<block2::SU2K, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZK, double, block2::CSRSparseMatrix<block2::SZK, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2K, double, block2::CSRSparseMatrix<block2::SU2K, double>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZK, double>;
extern template struct block2::DelayedTensorFunctions<block2::SU2K, double>;

// expr.hpp
extern template struct block2::OpExpr<block2::SZK>;
extern template struct block2::OpExprRef<block2::SZK>;
extern template struct block2::OpCounter<block2::SZK>;
extern template struct block2::OpElement<block2::SZK, double>;
extern template struct block2::OpElementRef<block2::SZK, double>;
extern template struct block2::OpProduct<block2::SZK, double>;
extern template struct block2::OpSumProd<block2::SZK, double>;
extern template struct block2::OpSum<block2::SZK, double>;

extern template struct block2::OpExpr<block2::SU2K>;
extern template struct block2::OpExprRef<block2::SU2K>;
extern template struct block2::OpCounter<block2::SU2K>;
extern template struct block2::OpElement<block2::SU2K, double>;
extern template struct block2::OpElementRef<block2::SU2K, double>;
extern template struct block2::OpProduct<block2::SU2K, double>;
extern template struct block2::OpSumProd<block2::SU2K, double>;
extern template struct block2::OpSum<block2::SU2K, double>;

// hamiltonian.hpp
extern template struct block2::SiteBasis<block2::SZK>;
extern template struct block2::SiteBasis<block2::SU2K>;
extern template struct block2::Hamiltonian<block2::SZK, double>;
extern template struct block2::Hamiltonian<block2::SU2K, double>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZK, double, block2::Hamiltonian<block2::SZK, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2K, double, block2::Hamiltonian<block2::SU2K, double>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZK, double>;
extern template struct block2::OperatorFunctions<block2::SU2K, double>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZK, double>;
extern template struct block2::DelayedOperatorTensor<block2::SZK, double>;

extern template struct block2::OperatorTensor<block2::SU2K, double>;
extern template struct block2::DelayedOperatorTensor<block2::SU2K, double>;

// parallel_mpi.hpp
#ifdef _HAS_MPI
extern template struct block2::MPICommunicator<block2::SZK>;
extern template struct block2::MPICommunicator<block2::SU2K>;
#endif

// parallel_rule.hpp
extern template struct block2::ParallelCommunicator<block2::SZK>;
extern template struct block2::ParallelRule<block2::SZK>;
extern template struct block2::ParallelRule<block2::SZK, double>;

extern template struct block2::ParallelCommunicator<block2::SU2K>;
extern template struct block2::ParallelRule<block2::SU2K>;
extern template struct block2::ParallelRule<block2::SU2K, double>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZK, double>;
extern template struct block2::ParallelTensorFunctions<block2::SU2K, double>;

// rule.hpp
extern template struct block2::Rule<block2::SZK, double>;
extern template struct block2::NoTransposeRule<block2::SZK, double>;

extern template struct block2::Rule<block2::SU2K, double>;
extern template struct block2::NoTransposeRule<block2::SU2K, double>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrixInfo<block2::SZK>;
extern template struct block2::SparseMatrix<block2::SZK, double>;
extern template struct block2::SparseMatrixGroup<block2::SZK, double>;

extern template struct block2::SparseMatrixInfo<block2::SU2K>;
extern template struct block2::SparseMatrix<block2::SU2K, double>;
extern template struct block2::SparseMatrixGroup<block2::SU2K, double>;

// state_info.hpp
extern template struct block2::StateInfo<block2::SZK>;
extern template struct block2::StateProbability<block2::SZK>;

extern template struct block2::StateInfo<block2::SU2K>;
extern template struct block2::StateProbability<block2::SU2K>;

extern template struct block2::TransStateInfo<block2::SZK, block2::SU2K>;
extern template struct block2::TransStateInfo<block2::SU2K, block2::SZK>;

// symbolic.hpp
extern template struct block2::Symbolic<block2::SZK>;
extern template struct block2::SymbolicRowVector<block2::SZK>;
extern template struct block2::SymbolicColumnVector<block2::SZK>;
extern template struct block2::SymbolicMatrix<block2::SZK>;

extern template struct block2::Symbolic<block2::SU2K>;
extern template struct block2::SymbolicRowVector<block2::SU2K>;
extern template struct block2::SymbolicColumnVector<block2::SU2K>;
extern template struct block2::SymbolicMatrix<block2::SU2K>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SZK, double>;
extern template struct block2::TensorFunctions<block2::SU2K, double>;

#endif

#ifdef _USE_SG

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SGF, double>;
extern template struct block2::ArchivedSparseMatrix<block2::SGB, double>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SGF, double>;
extern template struct block2::ArchivedTensorFunctions<block2::SGB, double>;

// cg.hpp
extern template struct block2::CG<block2::SGF>;
extern template struct block2::CG<block2::SGB>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SGF, double>;
extern template struct block2::CSROperatorFunctions<block2::SGB, double>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SGF, double>;
extern template struct block2::CSRSparseMatrix<block2::SGB, double>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SGF, double>;
extern template struct block2::DelayedSparseMatrix<block2::SGB, double>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, double, block2::SparseMatrix<block2::SGF, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, double, block2::SparseMatrix<block2::SGB, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, double, block2::CSRSparseMatrix<block2::SGF, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, double, block2::CSRSparseMatrix<block2::SGB, double>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SGF, double>;
extern template struct block2::DelayedTensorFunctions<block2::SGB, double>;

// expr.hpp
extern template struct block2::OpExpr<block2::SGF>;
extern template struct block2::OpExprRef<block2::SGF>;
extern template struct block2::OpCounter<block2::SGF>;
extern template struct block2::OpElement<block2::SGF, double>;
extern template struct block2::OpElementRef<block2::SGF, double>;
extern template struct block2::OpProduct<block2::SGF, double>;
extern template struct block2::OpSumProd<block2::SGF, double>;
extern template struct block2::OpSum<block2::SGF, double>;

extern template struct block2::OpExpr<block2::SGB>;
extern template struct block2::OpExprRef<block2::SGB>;
extern template struct block2::OpCounter<block2::SGB>;
extern template struct block2::OpElement<block2::SGB, double>;
extern template struct block2::OpElementRef<block2::SGB, double>;
extern template struct block2::OpProduct<block2::SGB, double>;
extern template struct block2::OpSumProd<block2::SGB, double>;
extern template struct block2::OpSum<block2::SGB, double>;

// hamiltonian.hpp
extern template struct block2::SiteBasis<block2::SGF>;
extern template struct block2::SiteBasis<block2::SGB>;
extern template struct block2::Hamiltonian<block2::SGF, double>;
extern template struct block2::Hamiltonian<block2::SGB, double>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, double, block2::Hamiltonian<block2::SGF, double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, double, block2::Hamiltonian<block2::SGB, double>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SGF, double>;
extern template struct block2::OperatorFunctions<block2::SGB, double>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SGF, double>;
extern template struct block2::DelayedOperatorTensor<block2::SGF, double>;

extern template struct block2::OperatorTensor<block2::SGB, double>;
extern template struct block2::DelayedOperatorTensor<block2::SGB, double>;

// parallel_mpi.hpp
#ifdef _HAS_MPI
extern template struct block2::MPICommunicator<block2::SGF>;
extern template struct block2::MPICommunicator<block2::SGB>;
#endif

// parallel_rule.hpp
extern template struct block2::ParallelCommunicator<block2::SGF>;
extern template struct block2::ParallelRule<block2::SGF>;
extern template struct block2::ParallelRule<block2::SGF, double>;

extern template struct block2::ParallelCommunicator<block2::SGB>;
extern template struct block2::ParallelRule<block2::SGB>;
extern template struct block2::ParallelRule<block2::SGB, double>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SGF, double>;
extern template struct block2::ParallelTensorFunctions<block2::SGB, double>;

// rule.hpp
extern template struct block2::Rule<block2::SGF, double>;
extern template struct block2::NoTransposeRule<block2::SGF, double>;

extern template struct block2::Rule<block2::SGB, double>;
extern template struct block2::NoTransposeRule<block2::SGB, double>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrixInfo<block2::SGF>;
extern template struct block2::SparseMatrix<block2::SGF, double>;
extern template struct block2::SparseMatrixGroup<block2::SGF, double>;

extern template struct block2::SparseMatrixInfo<block2::SGB>;
extern template struct block2::SparseMatrix<block2::SGB, double>;
extern template struct block2::SparseMatrixGroup<block2::SGB, double>;

// state_info.hpp
extern template struct block2::StateInfo<block2::SGF>;
extern template struct block2::StateProbability<block2::SGF>;

extern template struct block2::StateInfo<block2::SGB>;
extern template struct block2::StateProbability<block2::SGB>;

// symbolic.hpp
extern template struct block2::Symbolic<block2::SGF>;
extern template struct block2::SymbolicRowVector<block2::SGF>;
extern template struct block2::SymbolicColumnVector<block2::SGF>;
extern template struct block2::SymbolicMatrix<block2::SGF>;

extern template struct block2::Symbolic<block2::SGB>;
extern template struct block2::SymbolicRowVector<block2::SGB>;
extern template struct block2::SymbolicColumnVector<block2::SGB>;
extern template struct block2::SymbolicMatrix<block2::SGB>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SGF, double>;
extern template struct block2::TensorFunctions<block2::SGB, double>;

#endif

#ifdef _USE_COMPLEX

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZ,
                                                    complex<double>>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2,
                                                    complex<double>>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZ,
                                                       complex<double>>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2,
                                                       complex<double>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZ,
                                                    complex<double>>;
extern template struct block2::CSROperatorFunctions<block2::SU2,
                                                    complex<double>>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZ, complex<double>>;
extern template struct block2::CSRSparseMatrix<block2::SU2, complex<double>>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZ, complex<double>>;
extern template struct block2::DelayedSparseMatrix<block2::SU2,
                                                   complex<double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, complex<double>,
    block2::SparseMatrix<block2::SZ, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, complex<double>,
    block2::SparseMatrix<block2::SU2, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, complex<double>,
    block2::CSRSparseMatrix<block2::SZ, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, complex<double>,
    block2::CSRSparseMatrix<block2::SU2, complex<double>>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZ,
                                                      complex<double>>;
extern template struct block2::DelayedTensorFunctions<block2::SU2,
                                                      complex<double>>;

// expr.hpp
extern template struct block2::OpElement<block2::SZ, complex<double>>;
extern template struct block2::OpElementRef<block2::SZ, complex<double>>;
extern template struct block2::OpProduct<block2::SZ, complex<double>>;
extern template struct block2::OpSumProd<block2::SZ, complex<double>>;
extern template struct block2::OpSum<block2::SZ, complex<double>>;

extern template struct block2::OpElement<block2::SU2, complex<double>>;
extern template struct block2::OpElementRef<block2::SU2, complex<double>>;
extern template struct block2::OpProduct<block2::SU2, complex<double>>;
extern template struct block2::OpSumProd<block2::SU2, complex<double>>;
extern template struct block2::OpSum<block2::SU2, complex<double>>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SZ, complex<double>>;
extern template struct block2::Hamiltonian<block2::SU2, complex<double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, complex<double>,
    block2::Hamiltonian<block2::SZ, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, complex<double>,
    block2::Hamiltonian<block2::SU2, complex<double>>>;

// integral.hpp
extern template struct block2::FCIDUMP<complex<double>>;
extern template struct block2::SpinOrbitalFCIDUMP<complex<double>>;
extern template struct block2::MRCISFCIDUMP<complex<double>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZ, complex<double>>;
extern template struct block2::OperatorFunctions<block2::SU2, complex<double>>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZ, complex<double>>;
extern template struct block2::DelayedOperatorTensor<block2::SZ,
                                                     complex<double>>;

extern template struct block2::OperatorTensor<block2::SU2, complex<double>>;
extern template struct block2::DelayedOperatorTensor<block2::SU2,
                                                     complex<double>>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SZ, complex<double>>;
extern template struct block2::ParallelRule<block2::SU2, complex<double>>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZ,
                                                       complex<double>>;
extern template struct block2::ParallelTensorFunctions<block2::SU2,
                                                       complex<double>>;

// rule.hpp
extern template struct block2::Rule<block2::SZ, complex<double>>;
extern template struct block2::NoTransposeRule<block2::SZ, complex<double>>;

extern template struct block2::Rule<block2::SU2, complex<double>>;
extern template struct block2::NoTransposeRule<block2::SU2, complex<double>>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SZ, complex<double>>;
extern template struct block2::SparseMatrixGroup<block2::SZ, complex<double>>;

extern template struct block2::SparseMatrix<block2::SU2, complex<double>>;
extern template struct block2::SparseMatrixGroup<block2::SU2, complex<double>>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SZ, complex<double>>;
extern template struct block2::TensorFunctions<block2::SU2, complex<double>>;

#ifdef _USE_KSYMM

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZK,
                                                    complex<double>>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2K,
                                                    complex<double>>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZK,
                                                       complex<double>>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2K,
                                                       complex<double>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZK,
                                                    complex<double>>;
extern template struct block2::CSROperatorFunctions<block2::SU2K,
                                                    complex<double>>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZK, complex<double>>;
extern template struct block2::CSRSparseMatrix<block2::SU2K, complex<double>>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZK,
                                                   complex<double>>;
extern template struct block2::DelayedSparseMatrix<block2::SU2K,
                                                   complex<double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZK, complex<double>,
    block2::SparseMatrix<block2::SZK, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2K, complex<double>,
    block2::SparseMatrix<block2::SU2K, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZK, complex<double>,
    block2::CSRSparseMatrix<block2::SZK, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2K, complex<double>,
    block2::CSRSparseMatrix<block2::SU2K, complex<double>>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZK,
                                                      complex<double>>;
extern template struct block2::DelayedTensorFunctions<block2::SU2K,
                                                      complex<double>>;

// expr.hpp
extern template struct block2::OpElement<block2::SZK, complex<double>>;
extern template struct block2::OpElementRef<block2::SZK, complex<double>>;
extern template struct block2::OpProduct<block2::SZK, complex<double>>;
extern template struct block2::OpSumProd<block2::SZK, complex<double>>;
extern template struct block2::OpSum<block2::SZK, complex<double>>;

extern template struct block2::OpElement<block2::SU2K, complex<double>>;
extern template struct block2::OpElementRef<block2::SU2K, complex<double>>;
extern template struct block2::OpProduct<block2::SU2K, complex<double>>;
extern template struct block2::OpSumProd<block2::SU2K, complex<double>>;
extern template struct block2::OpSum<block2::SU2K, complex<double>>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SZK, complex<double>>;
extern template struct block2::Hamiltonian<block2::SU2K, complex<double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZK, complex<double>,
    block2::Hamiltonian<block2::SZK, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2K, complex<double>,
    block2::Hamiltonian<block2::SU2K, complex<double>>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZK, complex<double>>;
extern template struct block2::OperatorFunctions<block2::SU2K, complex<double>>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZK, complex<double>>;
extern template struct block2::DelayedOperatorTensor<block2::SZK,
                                                     complex<double>>;

extern template struct block2::OperatorTensor<block2::SU2K, complex<double>>;
extern template struct block2::DelayedOperatorTensor<block2::SU2K,
                                                     complex<double>>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SZK, complex<double>>;
extern template struct block2::ParallelRule<block2::SU2K, complex<double>>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZK,
                                                       complex<double>>;
extern template struct block2::ParallelTensorFunctions<block2::SU2K,
                                                       complex<double>>;

// rule.hpp
extern template struct block2::Rule<block2::SZK, complex<double>>;
extern template struct block2::NoTransposeRule<block2::SZK, complex<double>>;

extern template struct block2::Rule<block2::SU2K, complex<double>>;
extern template struct block2::NoTransposeRule<block2::SU2K, complex<double>>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SZK, complex<double>>;
extern template struct block2::SparseMatrixGroup<block2::SZK, complex<double>>;

extern template struct block2::SparseMatrix<block2::SU2K, complex<double>>;
extern template struct block2::SparseMatrixGroup<block2::SU2K, complex<double>>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SZK, complex<double>>;
extern template struct block2::TensorFunctions<block2::SU2K, complex<double>>;

#endif

#ifdef _USE_SG

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SGF,
                                                    complex<double>>;
extern template struct block2::ArchivedSparseMatrix<block2::SGB,
                                                    complex<double>>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SGF,
                                                       complex<double>>;
extern template struct block2::ArchivedTensorFunctions<block2::SGB,
                                                       complex<double>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SGF,
                                                    complex<double>>;
extern template struct block2::CSROperatorFunctions<block2::SGB,
                                                    complex<double>>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SGF, complex<double>>;
extern template struct block2::CSRSparseMatrix<block2::SGB, complex<double>>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SGF,
                                                   complex<double>>;
extern template struct block2::DelayedSparseMatrix<block2::SGB,
                                                   complex<double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, complex<double>,
    block2::SparseMatrix<block2::SGF, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, complex<double>,
    block2::SparseMatrix<block2::SGB, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, complex<double>,
    block2::CSRSparseMatrix<block2::SGF, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, complex<double>,
    block2::CSRSparseMatrix<block2::SGB, complex<double>>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SGF,
                                                      complex<double>>;
extern template struct block2::DelayedTensorFunctions<block2::SGB,
                                                      complex<double>>;

// expr.hpp
extern template struct block2::OpElement<block2::SGF, complex<double>>;
extern template struct block2::OpElementRef<block2::SGF, complex<double>>;
extern template struct block2::OpProduct<block2::SGF, complex<double>>;
extern template struct block2::OpSumProd<block2::SGF, complex<double>>;
extern template struct block2::OpSum<block2::SGF, complex<double>>;

extern template struct block2::OpElement<block2::SGB, complex<double>>;
extern template struct block2::OpElementRef<block2::SGB, complex<double>>;
extern template struct block2::OpProduct<block2::SGB, complex<double>>;
extern template struct block2::OpSumProd<block2::SGB, complex<double>>;
extern template struct block2::OpSum<block2::SGB, complex<double>>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SGF, complex<double>>;
extern template struct block2::Hamiltonian<block2::SGB, complex<double>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, complex<double>,
    block2::Hamiltonian<block2::SGF, complex<double>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, complex<double>,
    block2::Hamiltonian<block2::SGB, complex<double>>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SGF, complex<double>>;
extern template struct block2::OperatorFunctions<block2::SGB, complex<double>>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SGF, complex<double>>;
extern template struct block2::DelayedOperatorTensor<block2::SGF,
                                                     complex<double>>;

extern template struct block2::OperatorTensor<block2::SGB, complex<double>>;
extern template struct block2::DelayedOperatorTensor<block2::SGB,
                                                     complex<double>>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SGF, complex<double>>;
extern template struct block2::ParallelRule<block2::SGB, complex<double>>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SGF,
                                                       complex<double>>;
extern template struct block2::ParallelTensorFunctions<block2::SGB,
                                                       complex<double>>;

// rule.hpp
extern template struct block2::Rule<block2::SGF, complex<double>>;
extern template struct block2::NoTransposeRule<block2::SGF, complex<double>>;

extern template struct block2::Rule<block2::SGB, complex<double>>;
extern template struct block2::NoTransposeRule<block2::SGB, complex<double>>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SGF, complex<double>>;
extern template struct block2::SparseMatrixGroup<block2::SGF, complex<double>>;

extern template struct block2::SparseMatrix<block2::SGB, complex<double>>;
extern template struct block2::SparseMatrixGroup<block2::SGB, complex<double>>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SGF, complex<double>>;
extern template struct block2::TensorFunctions<block2::SGB, complex<double>>;

#endif

#endif

#ifdef _USE_SINGLE_PREC

// allocator.hpp
extern template struct block2::Allocator<float>;
extern template struct block2::StackAllocator<float>;
extern template struct block2::VectorAllocator<float>;
extern template struct block2::TemporaryAllocator<float>;
extern template struct block2::DataFrame<float>;

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZ, float>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2, float>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZ, float>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2, float>;

// batch_gemm.hpp
extern template struct block2::BatchGEMM<float>;
extern template struct block2::BatchGEMMRef<float>;
extern template struct block2::AdvancedGEMM<float>;
extern template struct block2::BatchGEMMSeq<float>;

extern template struct block2::BatchGEMM<complex<float>>;
extern template struct block2::BatchGEMMRef<complex<float>>;
extern template struct block2::AdvancedGEMM<complex<float>>;
extern template struct block2::BatchGEMMSeq<complex<float>>;

// csr_matrix.hpp
extern template struct block2::GCSRMatrix<float>;
extern template struct block2::GCSRMatrix<complex<float>>;

// csr_matrix_functions.hpp
extern template struct block2::GCSRMatrixFunctions<float>;
extern template struct block2::GCSRMatrixFunctions<complex<float>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZ, float>;
extern template struct block2::CSROperatorFunctions<block2::SU2, float>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZ, float>;
extern template struct block2::CSRSparseMatrix<block2::SU2, float>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZ, float>;
extern template struct block2::DelayedSparseMatrix<block2::SU2, float>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, float, block2::SparseMatrix<block2::SZ, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, float, block2::SparseMatrix<block2::SU2, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, float, block2::CSRSparseMatrix<block2::SZ, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, float, block2::CSRSparseMatrix<block2::SU2, float>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZ, float>;
extern template struct block2::DelayedTensorFunctions<block2::SU2, float>;

// expr.hpp
extern template struct block2::OpElement<block2::SZ, float>;
extern template struct block2::OpElementRef<block2::SZ, float>;
extern template struct block2::OpProduct<block2::SZ, float>;
extern template struct block2::OpSumProd<block2::SZ, float>;
extern template struct block2::OpSum<block2::SZ, float>;

extern template struct block2::OpElement<block2::SU2, float>;
extern template struct block2::OpElementRef<block2::SU2, float>;
extern template struct block2::OpProduct<block2::SU2, float>;
extern template struct block2::OpSumProd<block2::SU2, float>;
extern template struct block2::OpSum<block2::SU2, float>;

// fp_codec.hpp
extern template struct block2::FPCodec<float>;
extern template struct block2::CompressedVector<float>;
extern template struct block2::CompressedVectorMT<float>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SZ, float>;
extern template struct block2::Hamiltonian<block2::SU2, float>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, float, block2::Hamiltonian<block2::SZ, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, float, block2::Hamiltonian<block2::SU2, float>>;

// integral.hpp
extern template struct block2::FCIDUMP<float>;
extern template struct block2::SpinOrbitalFCIDUMP<float>;
extern template struct block2::MRCISFCIDUMP<float>;

// iterative_matrix_functions.hpp
extern template struct block2::IterativeMatrixFunctions<float>;
extern template struct block2::IterativeMatrixFunctions<complex<float>>;

// matrix.hpp
extern template struct block2::GMatrix<float>;
extern template struct block2::GDiagonalMatrix<float>;
extern template struct block2::GIdentityMatrix<float>;
extern template struct block2::GTensor<float>;

extern template struct block2::GMatrix<complex<float>>;
extern template struct block2::GDiagonalMatrix<complex<float>>;
extern template struct block2::GTensor<complex<float>>;

// matrix_functions.hpp
extern template struct block2::GMatrixFunctions<float>;
extern template struct block2::GMatrixFunctions<complex<float>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZ, float>;
extern template struct block2::OperatorFunctions<block2::SU2, float>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZ, float>;
extern template struct block2::DelayedOperatorTensor<block2::SZ, float>;

extern template struct block2::OperatorTensor<block2::SU2, float>;
extern template struct block2::DelayedOperatorTensor<block2::SU2, float>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SZ, float>;
extern template struct block2::ParallelRule<block2::SU2, float>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZ, float>;
extern template struct block2::ParallelTensorFunctions<block2::SU2, float>;

// rule.hpp
extern template struct block2::Rule<block2::SZ, float>;
extern template struct block2::NoTransposeRule<block2::SZ, float>;

extern template struct block2::Rule<block2::SU2, float>;
extern template struct block2::NoTransposeRule<block2::SU2, float>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SZ, float>;
extern template struct block2::SparseMatrixGroup<block2::SZ, float>;
extern template struct block2::SparseMatrix<block2::SU2, float>;
extern template struct block2::SparseMatrixGroup<block2::SU2, float>;

extern template struct block2::TransSparseMatrix<block2::SZ, float, double>;
extern template struct block2::TransSparseMatrix<block2::SZ, double, float>;
extern template struct block2::TransSparseMatrix<block2::SU2, float, double>;
extern template struct block2::TransSparseMatrix<block2::SU2, double, float>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SZ, float>;
extern template struct block2::TensorFunctions<block2::SU2, float>;

#ifdef _USE_SG

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SGF, float>;
extern template struct block2::ArchivedSparseMatrix<block2::SGB, float>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SGF, float>;
extern template struct block2::ArchivedTensorFunctions<block2::SGB, float>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SGF, float>;
extern template struct block2::CSROperatorFunctions<block2::SGB, float>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SGF, float>;
extern template struct block2::CSRSparseMatrix<block2::SGB, float>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SGF, float>;
extern template struct block2::DelayedSparseMatrix<block2::SGB, float>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, float, block2::SparseMatrix<block2::SGF, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, float, block2::SparseMatrix<block2::SGB, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, float, block2::CSRSparseMatrix<block2::SGF, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, float, block2::CSRSparseMatrix<block2::SGB, float>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SGF, float>;
extern template struct block2::DelayedTensorFunctions<block2::SGB, float>;

// expr.hpp
extern template struct block2::OpElement<block2::SGF, float>;
extern template struct block2::OpElementRef<block2::SGF, float>;
extern template struct block2::OpProduct<block2::SGF, float>;
extern template struct block2::OpSumProd<block2::SGF, float>;
extern template struct block2::OpSum<block2::SGF, float>;

extern template struct block2::OpElement<block2::SGB, float>;
extern template struct block2::OpElementRef<block2::SGB, float>;
extern template struct block2::OpProduct<block2::SGB, float>;
extern template struct block2::OpSumProd<block2::SGB, float>;
extern template struct block2::OpSum<block2::SGB, float>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SGF, float>;
extern template struct block2::Hamiltonian<block2::SGB, float>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, float, block2::Hamiltonian<block2::SGF, float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, float, block2::Hamiltonian<block2::SGB, float>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SGF, float>;
extern template struct block2::OperatorFunctions<block2::SGB, float>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SGF, float>;
extern template struct block2::DelayedOperatorTensor<block2::SGF, float>;

extern template struct block2::OperatorTensor<block2::SGB, float>;
extern template struct block2::DelayedOperatorTensor<block2::SGB, float>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SGF, float>;
extern template struct block2::ParallelRule<block2::SGB, float>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SGF, float>;
extern template struct block2::ParallelTensorFunctions<block2::SGB, float>;

// rule.hpp
extern template struct block2::Rule<block2::SGF, float>;
extern template struct block2::NoTransposeRule<block2::SGF, float>;

extern template struct block2::Rule<block2::SGB, float>;
extern template struct block2::NoTransposeRule<block2::SGB, float>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SGF, float>;
extern template struct block2::SparseMatrixGroup<block2::SGF, float>;
extern template struct block2::SparseMatrix<block2::SGB, float>;
extern template struct block2::SparseMatrixGroup<block2::SGB, float>;

extern template struct block2::TransSparseMatrix<block2::SGF, float, double>;
extern template struct block2::TransSparseMatrix<block2::SGF, double, float>;
extern template struct block2::TransSparseMatrix<block2::SGB, float, double>;
extern template struct block2::TransSparseMatrix<block2::SGB, double, float>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SGF, float>;
extern template struct block2::TensorFunctions<block2::SGB, float>;

#endif

#ifdef _USE_COMPLEX

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SZ, complex<float>>;
extern template struct block2::ArchivedSparseMatrix<block2::SU2,
                                                    complex<float>>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SZ,
                                                       complex<float>>;
extern template struct block2::ArchivedTensorFunctions<block2::SU2,
                                                       complex<float>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SZ, complex<float>>;
extern template struct block2::CSROperatorFunctions<block2::SU2,
                                                    complex<float>>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SZ, complex<float>>;
extern template struct block2::CSRSparseMatrix<block2::SU2, complex<float>>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SZ, complex<float>>;
extern template struct block2::DelayedSparseMatrix<block2::SU2, complex<float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, complex<float>,
    block2::SparseMatrix<block2::SZ, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, complex<float>,
    block2::SparseMatrix<block2::SU2, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, complex<float>,
    block2::CSRSparseMatrix<block2::SZ, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, complex<float>,
    block2::CSRSparseMatrix<block2::SU2, complex<float>>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SZ,
                                                      complex<float>>;
extern template struct block2::DelayedTensorFunctions<block2::SU2,
                                                      complex<float>>;

// expr.hpp
extern template struct block2::OpElement<block2::SZ, complex<float>>;
extern template struct block2::OpElementRef<block2::SZ, complex<float>>;
extern template struct block2::OpProduct<block2::SZ, complex<float>>;
extern template struct block2::OpSumProd<block2::SZ, complex<float>>;
extern template struct block2::OpSum<block2::SZ, complex<float>>;

extern template struct block2::OpElement<block2::SU2, complex<float>>;
extern template struct block2::OpElementRef<block2::SU2, complex<float>>;
extern template struct block2::OpProduct<block2::SU2, complex<float>>;
extern template struct block2::OpSumProd<block2::SU2, complex<float>>;
extern template struct block2::OpSum<block2::SU2, complex<float>>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SZ, complex<float>>;
extern template struct block2::Hamiltonian<block2::SU2, complex<float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, complex<float>,
    block2::Hamiltonian<block2::SZ, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, complex<float>,
    block2::Hamiltonian<block2::SU2, complex<float>>>;

// integral.hpp
extern template struct block2::FCIDUMP<complex<float>>;
extern template struct block2::SpinOrbitalFCIDUMP<complex<float>>;
extern template struct block2::MRCISFCIDUMP<complex<float>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SZ, complex<float>>;
extern template struct block2::OperatorFunctions<block2::SU2, complex<float>>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SZ, complex<float>>;
extern template struct block2::DelayedOperatorTensor<block2::SZ,
                                                     complex<float>>;

extern template struct block2::OperatorTensor<block2::SU2, complex<float>>;
extern template struct block2::DelayedOperatorTensor<block2::SU2,
                                                     complex<float>>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SZ, complex<float>>;
extern template struct block2::ParallelRule<block2::SU2, complex<float>>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZ,
                                                       complex<float>>;
extern template struct block2::ParallelTensorFunctions<block2::SU2,
                                                       complex<float>>;

// rule.hpp
extern template struct block2::Rule<block2::SZ, complex<float>>;
extern template struct block2::NoTransposeRule<block2::SZ, complex<float>>;

extern template struct block2::Rule<block2::SU2, complex<float>>;
extern template struct block2::NoTransposeRule<block2::SU2, complex<float>>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SZ, complex<float>>;
extern template struct block2::SparseMatrixGroup<block2::SZ, complex<float>>;

extern template struct block2::SparseMatrix<block2::SU2, complex<float>>;
extern template struct block2::SparseMatrixGroup<block2::SU2, complex<float>>;

extern template struct block2::TransSparseMatrix<block2::SZ, complex<float>,
                                                 complex<double>>;
extern template struct block2::TransSparseMatrix<block2::SZ, complex<double>,
                                                 complex<float>>;
extern template struct block2::TransSparseMatrix<block2::SU2, complex<float>,
                                                 complex<double>>;
extern template struct block2::TransSparseMatrix<block2::SU2, complex<double>,
                                                 complex<float>>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SZ, complex<float>>;
extern template struct block2::TensorFunctions<block2::SU2, complex<float>>;

#ifdef _USE_SG

// archived_sparse_matrix.hpp
extern template struct block2::ArchivedSparseMatrix<block2::SGF,
                                                    complex<float>>;
extern template struct block2::ArchivedSparseMatrix<block2::SGB,
                                                    complex<float>>;

// archived_tensor_functions.hpp
extern template struct block2::ArchivedTensorFunctions<block2::SGF,
                                                       complex<float>>;
extern template struct block2::ArchivedTensorFunctions<block2::SGB,
                                                       complex<float>>;

// csr_operator_functions.hpp
extern template struct block2::CSROperatorFunctions<block2::SGF,
                                                    complex<float>>;
extern template struct block2::CSROperatorFunctions<block2::SGB,
                                                    complex<float>>;

// csr_sparse_matrix.hpp
extern template struct block2::CSRSparseMatrix<block2::SGF, complex<float>>;
extern template struct block2::CSRSparseMatrix<block2::SGB, complex<float>>;

// delayed_sparse_matrix.hpp
extern template struct block2::DelayedSparseMatrix<block2::SGF, complex<float>>;
extern template struct block2::DelayedSparseMatrix<block2::SGB, complex<float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, complex<float>,
    block2::SparseMatrix<block2::SGF, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, complex<float>,
    block2::SparseMatrix<block2::SGB, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, complex<float>,
    block2::CSRSparseMatrix<block2::SGF, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, complex<float>,
    block2::CSRSparseMatrix<block2::SGB, complex<float>>>;

// delayed_tensor_functions.hpp
extern template struct block2::DelayedTensorFunctions<block2::SGF,
                                                      complex<float>>;
extern template struct block2::DelayedTensorFunctions<block2::SGB,
                                                      complex<float>>;

// expr.hpp
extern template struct block2::OpElement<block2::SGF, complex<float>>;
extern template struct block2::OpElementRef<block2::SGF, complex<float>>;
extern template struct block2::OpProduct<block2::SGF, complex<float>>;
extern template struct block2::OpSumProd<block2::SGF, complex<float>>;
extern template struct block2::OpSum<block2::SGF, complex<float>>;

extern template struct block2::OpElement<block2::SGB, complex<float>>;
extern template struct block2::OpElementRef<block2::SGB, complex<float>>;
extern template struct block2::OpProduct<block2::SGB, complex<float>>;
extern template struct block2::OpSumProd<block2::SGB, complex<float>>;
extern template struct block2::OpSum<block2::SGB, complex<float>>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SGF, complex<float>>;
extern template struct block2::Hamiltonian<block2::SGB, complex<float>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGF, complex<float>,
    block2::Hamiltonian<block2::SGF, complex<float>>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SGB, complex<float>,
    block2::Hamiltonian<block2::SGB, complex<float>>>;

// operator_functions.hpp
extern template struct block2::OperatorFunctions<block2::SGF, complex<float>>;
extern template struct block2::OperatorFunctions<block2::SGB, complex<float>>;

// operator_tensor.hpp
extern template struct block2::OperatorTensor<block2::SGF, complex<float>>;
extern template struct block2::DelayedOperatorTensor<block2::SGF,
                                                     complex<float>>;

extern template struct block2::OperatorTensor<block2::SGB, complex<float>>;
extern template struct block2::DelayedOperatorTensor<block2::SGB,
                                                     complex<float>>;

// parallel_rule.hpp
extern template struct block2::ParallelRule<block2::SGF, complex<float>>;
extern template struct block2::ParallelRule<block2::SGB, complex<float>>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SGF,
                                                       complex<float>>;
extern template struct block2::ParallelTensorFunctions<block2::SGB,
                                                       complex<float>>;

// rule.hpp
extern template struct block2::Rule<block2::SGF, complex<float>>;
extern template struct block2::NoTransposeRule<block2::SGF, complex<float>>;

extern template struct block2::Rule<block2::SGB, complex<float>>;
extern template struct block2::NoTransposeRule<block2::SGB, complex<float>>;

// sparse_matrix.hpp
extern template struct block2::SparseMatrix<block2::SGF, complex<float>>;
extern template struct block2::SparseMatrixGroup<block2::SGF, complex<float>>;

extern template struct block2::SparseMatrix<block2::SGB, complex<float>>;
extern template struct block2::SparseMatrixGroup<block2::SGB, complex<float>>;

extern template struct block2::TransSparseMatrix<block2::SGF, complex<float>,
                                                 complex<double>>;
extern template struct block2::TransSparseMatrix<block2::SGF, complex<double>,
                                                 complex<float>>;
extern template struct block2::TransSparseMatrix<block2::SGB, complex<float>,
                                                 complex<double>>;
extern template struct block2::TransSparseMatrix<block2::SGB, complex<double>,
                                                 complex<float>>;

// tensor_functions.hpp
extern template struct block2::TensorFunctions<block2::SGF, complex<float>>;
extern template struct block2::TensorFunctions<block2::SGB, complex<float>>;

#endif

#endif

#endif
