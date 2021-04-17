
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

#include "../block2/allocator.hpp"
#include "../block2/archived_mpo.hpp"
#include "../block2/archived_sparse_matrix.hpp"
#include "../block2/archived_tensor_functions.hpp"
#include "../block2/cg.hpp"
#include "../block2/csr_operator_functions.hpp"
#include "../block2/csr_sparse_matrix.hpp"
#include "../block2/delayed_sparse_matrix.hpp"
#include "../block2/delayed_tensor_functions.hpp"
#include "../block2/determinant.hpp"
#include "../block2/effective_hamiltonian.hpp"
#include "../block2/expr.hpp"
#include "../block2/fp_codec.hpp"
#include "../block2/hamiltonian.hpp"
#include "../block2/moving_environment.hpp"
#include "../block2/mpo.hpp"
#include "../block2/mpo_fusing.hpp"
#include "../block2/mpo_simplification.hpp"
#include "../block2/mps.hpp"
#include "../block2/mps_unfused.hpp"
#include "../block2/operator_functions.hpp"
#include "../block2/operator_tensor.hpp"
#include "../block2/parallel_mpi.hpp"
#include "../block2/parallel_mpo.hpp"
#include "../block2/parallel_mps.hpp"
#include "../block2/parallel_rule.hpp"
#include "../block2/parallel_rule_sum_mpo.hpp"
#include "../block2/parallel_tensor_functions.hpp"
#include "../block2/partition.hpp"
#include "../block2/qc_hamiltonian.hpp"
#include "../block2/qc_mpo.hpp"
#include "../block2/qc_ncorr.hpp"
#include "../block2/qc_parallel_rule.hpp"
#include "../block2/qc_pdm1.hpp"
#include "../block2/qc_pdm2.hpp"
#include "../block2/qc_rule.hpp"
#include "../block2/qc_sum_mpo.hpp"
#include "../block2/rule.hpp"
#include "../block2/sparse_matrix.hpp"
#include "../block2/state_averaged.hpp"
#include "../block2/state_info.hpp"
#include "../block2/sweep_algorithm.hpp"
#include "../block2/sweep_algorithm_td.hpp"
#include "../block2/symbolic.hpp"
#include "../block2/tensor_functions.hpp"

#include "../block2/symmetry.hpp"
#include <cstdint>

// allocator.hpp
extern template struct block2::StackAllocator<uint32_t>;
extern template struct block2::StackAllocator<double>;

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZ>;
extern template struct block2::ArchivedMPO<block2::SU2>;

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

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZ>;
extern template struct block2::DeterminantQC<block2::SZ>;
extern template struct block2::DeterminantMPSInfo<block2::SZ>;

extern template struct block2::DeterminantQC<block2::SU2>;
extern template struct block2::DeterminantMPSInfo<block2::SU2>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<block2::SZ,
                                                    block2::MPS<block2::SZ>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZ>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, block2::MultiMPS<block2::SZ>>;

extern template struct block2::EffectiveHamiltonian<block2::SU2,
                                                    block2::MPS<block2::SU2>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2, block2::MultiMPS<block2::SU2>>;

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

// fp_codec.hpp
extern template struct block2::FPCodec<double>;

// hamiltonian.hpp
extern template struct block2::Hamiltonian<block2::SZ>;
extern template struct block2::Hamiltonian<block2::SU2>;
extern template struct block2::DelayedSparseMatrix<
    block2::SZ, block2::Hamiltonian<block2::SZ>>;
extern template struct block2::DelayedSparseMatrix<
    block2::SU2, block2::Hamiltonian<block2::SU2>>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZ>;
extern template struct block2::MovingEnvironment<block2::SU2>;

// mpo.hpp
extern template struct block2::MPOSchemer<block2::SZ>;
extern template struct block2::MPO<block2::SZ>;
extern template struct block2::DiagonalMPO<block2::SZ>;
extern template struct block2::AncillaMPO<block2::SZ>;
extern template struct block2::IdentityAddedMPO<block2::SZ>;

extern template struct block2::MPOSchemer<block2::SU2>;
extern template struct block2::MPO<block2::SU2>;
extern template struct block2::DiagonalMPO<block2::SU2>;
extern template struct block2::AncillaMPO<block2::SU2>;
extern template struct block2::IdentityAddedMPO<block2::SU2>;

// mpo_fusing.hpp
extern template struct block2::FusedMPO<block2::SZ>;
extern template struct block2::FusedMPO<block2::SU2>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZ>;
extern template struct block2::SimplifiedMPO<block2::SU2>;

// mps.hpp
extern template struct block2::MPSInfo<block2::SZ>;
extern template struct block2::DynamicMPSInfo<block2::SZ>;
extern template struct block2::CASCIMPSInfo<block2::SZ>;
extern template struct block2::MRCIMPSInfo<block2::SZ>;
extern template struct block2::AncillaMPSInfo<block2::SZ>;
extern template struct block2::MPS<block2::SZ>;

extern template struct block2::MPSInfo<block2::SU2>;
extern template struct block2::DynamicMPSInfo<block2::SU2>;
extern template struct block2::CASCIMPSInfo<block2::SU2>;
extern template struct block2::MRCIMPSInfo<block2::SU2>;
extern template struct block2::AncillaMPSInfo<block2::SU2>;
extern template struct block2::MPS<block2::SU2>;

extern template struct block2::TransMPSInfo<block2::SZ, block2::SU2>;
extern template struct block2::TransMPSInfo<block2::SU2, block2::SZ>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZ>;
extern template struct block2::UnfusedMPS<block2::SZ>;

extern template struct block2::SparseTensor<block2::SU2>;

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

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZ>;
extern template struct block2::ParallelMPO<block2::SZ>;

extern template struct block2::ClassicParallelMPO<block2::SU2>;
extern template struct block2::ParallelMPO<block2::SU2>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZ>;
extern template struct block2::ParallelMPS<block2::SU2>;

// parallel_rule_sum_mpo.hpp
extern template struct block2::ParallelRuleSumMPO<block2::SZ>;
extern template struct block2::SumMPORule<block2::SZ>;
extern template struct block2::ParallelFCIDUMP<block2::SZ>;

extern template struct block2::ParallelRuleSumMPO<block2::SU2>;
extern template struct block2::SumMPORule<block2::SU2>;
extern template struct block2::ParallelFCIDUMP<block2::SU2>;

// parallel_rule.hpp
extern template struct block2::ParallelCommunicator<block2::SZ>;
extern template struct block2::ParallelRule<block2::SZ>;

extern template struct block2::ParallelCommunicator<block2::SU2>;
extern template struct block2::ParallelRule<block2::SU2>;

// parallel_tensor_functions.hpp
extern template struct block2::ParallelTensorFunctions<block2::SZ>;
extern template struct block2::ParallelTensorFunctions<block2::SU2>;

// partition.hpp
extern template struct block2::Partition<block2::SZ>;
extern template struct block2::Partition<block2::SU2>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZ>;
extern template struct block2::HamiltonianQC<block2::SU2>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZ>;
extern template struct block2::SiteMPO<block2::SZ>;
extern template struct block2::MPOQC<block2::SZ>;

extern template struct block2::IdentityMPO<block2::SU2>;
extern template struct block2::SiteMPO<block2::SU2>;
extern template struct block2::MPOQC<block2::SU2>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZ>;
extern template struct block2::NPC1MPOQC<block2::SU2>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZ>;
extern template struct block2::ParallelRulePDM1QC<block2::SZ>;
extern template struct block2::ParallelRulePDM2QC<block2::SZ>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZ>;
extern template struct block2::ParallelRuleSiteQC<block2::SZ>;
extern template struct block2::ParallelRuleIdentity<block2::SZ>;

extern template struct block2::ParallelRuleQC<block2::SU2>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2>;
extern template struct block2::ParallelRuleIdentity<block2::SU2>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZ>;
extern template struct block2::PDM1MPOQC<block2::SU2>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZ>;
extern template struct block2::PDM2MPOQC<block2::SU2>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZ>;
extern template struct block2::RuleQC<block2::SU2>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZ>;

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

// state_averaged.hpp
extern template struct block2::MultiMPSInfo<block2::SZ>;
extern template struct block2::MultiMPS<block2::SZ>;

extern template struct block2::MultiMPSInfo<block2::SU2>;
extern template struct block2::MultiMPS<block2::SU2>;

// state_info.hpp
extern template struct block2::StateInfo<block2::SZ>;
extern template struct block2::StateProbability<block2::SZ>;

extern template struct block2::StateInfo<block2::SU2>;
extern template struct block2::StateProbability<block2::SU2>;

extern template struct block2::TransStateInfo<block2::SZ, block2::SU2>;
extern template struct block2::TransStateInfo<block2::SU2, block2::SZ>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZ>;
extern template struct block2::ImaginaryTE<block2::SZ>;

extern template struct block2::TDDMRG<block2::SU2>;
extern template struct block2::ImaginaryTE<block2::SU2>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZ>;
extern template struct block2::Linear<block2::SZ>;
extern template struct block2::Expect<block2::SZ>;

extern template struct block2::DMRG<block2::SU2>;
extern template struct block2::Linear<block2::SU2>;
extern template struct block2::Expect<block2::SU2>;

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
