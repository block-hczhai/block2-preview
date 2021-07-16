
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

#include "../core/symmetry.hpp"
#include "../dmrg/archived_mpo.hpp"
#include "../dmrg/determinant.hpp"
#include "../dmrg/effective_hamiltonian.hpp"
#include "../dmrg/moving_environment.hpp"
#include "../dmrg/mpo.hpp"
#include "../dmrg/mpo_fusing.hpp"
#include "../dmrg/mpo_simplification.hpp"
#include "../dmrg/mps.hpp"
#include "../dmrg/mps_unfused.hpp"
#include "../dmrg/parallel_mpo.hpp"
#include "../dmrg/parallel_mps.hpp"
#include "../dmrg/parallel_rule_sum_mpo.hpp"
#include "../dmrg/partition.hpp"
#include "../dmrg/qc_hamiltonian.hpp"
#include "../dmrg/qc_mpo.hpp"
#include "../dmrg/qc_ncorr.hpp"
#include "../dmrg/qc_parallel_rule.hpp"
#include "../dmrg/qc_pdm1.hpp"
#include "../dmrg/qc_pdm2.hpp"
#include "../dmrg/qc_rule.hpp"
#include "../dmrg/qc_sum_mpo.hpp"
#include "../dmrg/state_averaged.hpp"
#include "../dmrg/sweep_algorithm.hpp"
#include "../dmrg/sweep_algorithm_td.hpp"
#include "block2_core.hpp"

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZ>;
extern template struct block2::ArchivedMPO<block2::SU2>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZ>;
extern template struct block2::DeterminantQC<block2::SZ>;
extern template struct block2::DeterminantMPSInfo<block2::SZ>;

extern template struct block2::DeterminantTRIE<block2::SU2>;
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
extern template struct block2::UnfusedMPS<block2::SU2>;

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
extern template struct block2::ParallelRuleOneBodyQC<block2::SZ>;
extern template struct block2::ParallelRulePDM1QC<block2::SZ>;
extern template struct block2::ParallelRulePDM2QC<block2::SZ>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZ>;
extern template struct block2::ParallelRuleSiteQC<block2::SZ>;
extern template struct block2::ParallelRuleIdentity<block2::SZ>;

extern template struct block2::ParallelRuleQC<block2::SU2>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2>;
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
extern template struct block2::AntiHermitianRuleQC<block2::SZ>;

extern template struct block2::RuleQC<block2::SU2>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZ>;

// state_averaged.hpp
extern template struct block2::MultiMPSInfo<block2::SZ>;
extern template struct block2::MultiMPS<block2::SZ>;

extern template struct block2::MultiMPSInfo<block2::SU2>;
extern template struct block2::MultiMPS<block2::SU2>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZ>;
extern template struct block2::Linear<block2::SZ>;
extern template struct block2::Expect<block2::SZ>;

extern template struct block2::DMRG<block2::SU2>;
extern template struct block2::Linear<block2::SU2>;
extern template struct block2::Expect<block2::SU2>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZ>;
extern template struct block2::TimeEvolution<block2::SZ>;

extern template struct block2::TDDMRG<block2::SU2>;
extern template struct block2::TimeEvolution<block2::SU2>;
