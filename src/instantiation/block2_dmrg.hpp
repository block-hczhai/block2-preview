
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
#include "../dmrg/dmrg_driver.hpp"
#include "../dmrg/effective_functions.hpp"
#include "../dmrg/effective_hamiltonian.hpp"
#include "../dmrg/general_mpo.hpp"
#include "../dmrg/general_npdm.hpp"
#include "../dmrg/moving_environment.hpp"
#include "../dmrg/mpo.hpp"
#include "../dmrg/mpo_fusing.hpp"
#include "../dmrg/mpo_simplification.hpp"
#include "../dmrg/mps.hpp"
#include "../dmrg/mps_unfused.hpp"
#include "../dmrg/parallel_mpo.hpp"
#include "../dmrg/parallel_mps.hpp"
#include "../dmrg/parallel_simple.hpp"
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
extern template struct block2::ArchivedMPO<block2::SZ, double>;
extern template struct block2::ArchivedMPO<block2::SU2, double>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZ, double>;
extern template struct block2::DeterminantQC<block2::SZ, double>;
extern template struct block2::DeterminantMPSInfo<block2::SZ, double>;

extern template struct block2::DeterminantTRIE<block2::SU2, double>;
extern template struct block2::DeterminantQC<block2::SU2, double>;
extern template struct block2::DeterminantMPSInfo<block2::SU2, double>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SZ, double>;
extern template struct block2::DMRGDriver<block2::SU2, double>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SZ, double>;
extern template struct block2::EffectiveFunctions<block2::SU2, double>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, double, block2::MPS<block2::SZ, double>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZ, double>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, double, block2::MultiMPS<block2::SZ, double>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SU2, double, block2::MPS<block2::SU2, double>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2, double>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2, double, block2::MultiMPS<block2::SU2, double>>;

// general_mpo.hpp
extern template struct block2::GeneralFCIDUMP<double>;

extern template struct block2::GeneralHamiltonian<block2::SZ, double>;
extern template struct block2::GeneralMPO<block2::SZ, double>;

extern template struct block2::GeneralHamiltonian<block2::SU2, double>;
extern template struct block2::GeneralMPO<block2::SU2, double>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SZ, double>;
extern template struct block2::GeneralNPDMMPO<block2::SU2, double>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZ, double, double>;
extern template struct block2::MovingEnvironment<block2::SU2, double, double>;

// mpo.hpp
extern template struct block2::MPOSchemer<block2::SZ>;
extern template struct block2::MPO<block2::SZ, double>;
extern template struct block2::DiagonalMPO<block2::SZ, double>;
extern template struct block2::AncillaMPO<block2::SZ, double>;
extern template struct block2::IdentityAddedMPO<block2::SZ, double>;

extern template struct block2::MPOSchemer<block2::SU2>;
extern template struct block2::MPO<block2::SU2, double>;
extern template struct block2::DiagonalMPO<block2::SU2, double>;
extern template struct block2::AncillaMPO<block2::SU2, double>;
extern template struct block2::IdentityAddedMPO<block2::SU2, double>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SZ, double>;
extern template struct block2::FusedMPO<block2::SZ, double>;

extern template struct block2::CondensedMPO<block2::SU2, double>;
extern template struct block2::FusedMPO<block2::SU2, double>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZ, double>;
extern template struct block2::SimplifiedMPO<block2::SU2, double>;

// mps.hpp
extern template struct block2::MPSInfo<block2::SZ>;
extern template struct block2::DynamicMPSInfo<block2::SZ>;
extern template struct block2::CASCIMPSInfo<block2::SZ>;
extern template struct block2::MRCIMPSInfo<block2::SZ>;
extern template struct block2::NEVPTMPSInfo<block2::SZ>;
extern template struct block2::AncillaMPSInfo<block2::SZ>;
extern template struct block2::MPS<block2::SZ, double>;

extern template struct block2::MPSInfo<block2::SU2>;
extern template struct block2::DynamicMPSInfo<block2::SU2>;
extern template struct block2::CASCIMPSInfo<block2::SU2>;
extern template struct block2::MRCIMPSInfo<block2::SU2>;
extern template struct block2::NEVPTMPSInfo<block2::SU2>;
extern template struct block2::AncillaMPSInfo<block2::SU2>;
extern template struct block2::MPS<block2::SU2, double>;

extern template struct block2::TransMPSInfo<block2::SZ, block2::SU2>;
extern template struct block2::TransMPSInfo<block2::SU2, block2::SZ>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZ, double>;
extern template struct block2::UnfusedMPS<block2::SZ, double>;

extern template struct block2::SparseTensor<block2::SU2, double>;
extern template struct block2::UnfusedMPS<block2::SU2, double>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZ, double>;
extern template struct block2::ParallelMPO<block2::SZ, double>;

extern template struct block2::ClassicParallelMPO<block2::SU2, double>;
extern template struct block2::ParallelMPO<block2::SU2, double>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZ, double>;
extern template struct block2::ParallelMPS<block2::SU2, double>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SZ, double>;
extern template struct block2::SumMPORule<block2::SZ, double>;
extern template struct block2::ParallelFCIDUMP<block2::SZ, double>;

extern template struct block2::ParallelRuleSimple<block2::SU2, double>;
extern template struct block2::SumMPORule<block2::SU2, double>;
extern template struct block2::ParallelFCIDUMP<block2::SU2, double>;

// partition.hpp
extern template struct block2::Partition<block2::SZ, double>;
extern template struct block2::Partition<block2::SU2, double>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZ, double>;
extern template struct block2::HamiltonianQC<block2::SU2, double>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZ, double>;
extern template struct block2::SiteMPO<block2::SZ, double>;
extern template struct block2::MPOQC<block2::SZ, double>;

extern template struct block2::IdentityMPO<block2::SU2, double>;
extern template struct block2::SiteMPO<block2::SU2, double>;
extern template struct block2::MPOQC<block2::SU2, double>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZ, double>;
extern template struct block2::NPC1MPOQC<block2::SU2, double>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZ, double>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SZ, double>;
extern template struct block2::ParallelRulePDM1QC<block2::SZ, double>;
extern template struct block2::ParallelRulePDM2QC<block2::SZ, double>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZ, double>;
extern template struct block2::ParallelRuleSiteQC<block2::SZ, double>;
extern template struct block2::ParallelRuleIdentity<block2::SZ, double>;

extern template struct block2::ParallelRuleQC<block2::SU2, double>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2, double>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2, double>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2, double>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2, double>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2, double>;
extern template struct block2::ParallelRuleIdentity<block2::SU2, double>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZ, double>;
extern template struct block2::PDM1MPOQC<block2::SU2, double>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZ, double>;
extern template struct block2::PDM2MPOQC<block2::SU2, double>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZ, double>;
extern template struct block2::AntiHermitianRuleQC<block2::SZ, double>;

extern template struct block2::RuleQC<block2::SU2, double>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2, double>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZ, double>;

// state_averaged.hpp
extern template struct block2::MultiMPSInfo<block2::SZ>;
extern template struct block2::MultiMPS<block2::SZ, double>;

extern template struct block2::MultiMPSInfo<block2::SU2>;
extern template struct block2::MultiMPS<block2::SU2, double>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZ, double, double>;
extern template struct block2::Linear<block2::SZ, double, double>;
extern template struct block2::Expect<block2::SZ, double, double, double>;
extern template struct block2::Expect<block2::SZ, double, double,
                                      complex<double>>;

extern template struct block2::DMRG<block2::SU2, double, double>;
extern template struct block2::Linear<block2::SU2, double, double>;
extern template struct block2::Expect<block2::SU2, double, double, double>;
extern template struct block2::Expect<block2::SU2, double, double,
                                      complex<double>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZ, double, double>;
extern template struct block2::TimeEvolution<block2::SZ, double, double>;

extern template struct block2::TDDMRG<block2::SU2, double, double>;
extern template struct block2::TimeEvolution<block2::SU2, double, double>;

#ifdef _USE_KSYMM

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZK, double>;
extern template struct block2::ArchivedMPO<block2::SU2K, double>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZK, double>;
extern template struct block2::DeterminantQC<block2::SZK, double>;
extern template struct block2::DeterminantMPSInfo<block2::SZK, double>;

extern template struct block2::DeterminantTRIE<block2::SU2K, double>;
extern template struct block2::DeterminantQC<block2::SU2K, double>;
extern template struct block2::DeterminantMPSInfo<block2::SU2K, double>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SZK, double>;
extern template struct block2::DMRGDriver<block2::SU2K, double>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SZK, double>;
extern template struct block2::EffectiveFunctions<block2::SU2K, double>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SZK, double, block2::MPS<block2::SZK, double>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZK, double>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZK, double, block2::MultiMPS<block2::SZK, double>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SU2K, double, block2::MPS<block2::SU2K, double>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2K, double>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2K, double, block2::MultiMPS<block2::SU2K, double>>;

// general_mpo.hpp
extern template struct block2::GeneralHamiltonian<block2::SZK, double>;
extern template struct block2::GeneralMPO<block2::SZK, double>;

extern template struct block2::GeneralHamiltonian<block2::SU2K, double>;
extern template struct block2::GeneralMPO<block2::SU2K, double>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SZK, double>;
extern template struct block2::GeneralNPDMMPO<block2::SU2K, double>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZK, double, double>;
extern template struct block2::MovingEnvironment<block2::SU2K, double, double>;

// mpo.hpp
extern template struct block2::MPOSchemer<block2::SZK>;
extern template struct block2::MPO<block2::SZK, double>;
extern template struct block2::DiagonalMPO<block2::SZK, double>;
extern template struct block2::AncillaMPO<block2::SZK, double>;
extern template struct block2::IdentityAddedMPO<block2::SZK, double>;

extern template struct block2::MPOSchemer<block2::SU2K>;
extern template struct block2::MPO<block2::SU2K, double>;
extern template struct block2::DiagonalMPO<block2::SU2K, double>;
extern template struct block2::AncillaMPO<block2::SU2K, double>;
extern template struct block2::IdentityAddedMPO<block2::SU2K, double>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SZK, double>;
extern template struct block2::FusedMPO<block2::SZK, double>;

extern template struct block2::CondensedMPO<block2::SU2K, double>;
extern template struct block2::FusedMPO<block2::SU2K, double>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZK, double>;
extern template struct block2::SimplifiedMPO<block2::SU2K, double>;

// mps.hpp
extern template struct block2::MPSInfo<block2::SZK>;
extern template struct block2::DynamicMPSInfo<block2::SZK>;
extern template struct block2::CASCIMPSInfo<block2::SZK>;
extern template struct block2::MRCIMPSInfo<block2::SZK>;
extern template struct block2::NEVPTMPSInfo<block2::SZK>;
extern template struct block2::AncillaMPSInfo<block2::SZK>;
extern template struct block2::MPS<block2::SZK, double>;

extern template struct block2::MPSInfo<block2::SU2K>;
extern template struct block2::DynamicMPSInfo<block2::SU2K>;
extern template struct block2::CASCIMPSInfo<block2::SU2K>;
extern template struct block2::MRCIMPSInfo<block2::SU2K>;
extern template struct block2::NEVPTMPSInfo<block2::SU2K>;
extern template struct block2::AncillaMPSInfo<block2::SU2K>;
extern template struct block2::MPS<block2::SU2K, double>;

extern template struct block2::TransMPSInfo<block2::SZK, block2::SU2K>;
extern template struct block2::TransMPSInfo<block2::SU2K, block2::SZK>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZK, double>;
extern template struct block2::UnfusedMPS<block2::SZK, double>;

extern template struct block2::SparseTensor<block2::SU2K, double>;
extern template struct block2::UnfusedMPS<block2::SU2K, double>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZK, double>;
extern template struct block2::ParallelMPO<block2::SZK, double>;

extern template struct block2::ClassicParallelMPO<block2::SU2K, double>;
extern template struct block2::ParallelMPO<block2::SU2K, double>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZK, double>;
extern template struct block2::ParallelMPS<block2::SU2K, double>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SZK, double>;
extern template struct block2::SumMPORule<block2::SZK, double>;
extern template struct block2::ParallelFCIDUMP<block2::SZK, double>;

extern template struct block2::ParallelRuleSimple<block2::SU2K, double>;
extern template struct block2::SumMPORule<block2::SU2K, double>;
extern template struct block2::ParallelFCIDUMP<block2::SU2K, double>;

// partition.hpp
extern template struct block2::Partition<block2::SZK, double>;
extern template struct block2::Partition<block2::SU2K, double>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZK, double>;
extern template struct block2::HamiltonianQC<block2::SU2K, double>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZK, double>;
extern template struct block2::SiteMPO<block2::SZK, double>;
extern template struct block2::MPOQC<block2::SZK, double>;

extern template struct block2::IdentityMPO<block2::SU2K, double>;
extern template struct block2::SiteMPO<block2::SU2K, double>;
extern template struct block2::MPOQC<block2::SU2K, double>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZK, double>;
extern template struct block2::NPC1MPOQC<block2::SU2K, double>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZK, double>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SZK, double>;
extern template struct block2::ParallelRulePDM1QC<block2::SZK, double>;
extern template struct block2::ParallelRulePDM2QC<block2::SZK, double>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZK, double>;
extern template struct block2::ParallelRuleSiteQC<block2::SZK, double>;
extern template struct block2::ParallelRuleIdentity<block2::SZK, double>;

extern template struct block2::ParallelRuleQC<block2::SU2K, double>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2K, double>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2K, double>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2K, double>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2K, double>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2K, double>;
extern template struct block2::ParallelRuleIdentity<block2::SU2K, double>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZK, double>;
extern template struct block2::PDM1MPOQC<block2::SU2K, double>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZK, double>;
extern template struct block2::PDM2MPOQC<block2::SU2K, double>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZK, double>;
extern template struct block2::AntiHermitianRuleQC<block2::SZK, double>;

extern template struct block2::RuleQC<block2::SU2K, double>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2K, double>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZK, double>;

// state_averaged.hpp
extern template struct block2::MultiMPSInfo<block2::SZK>;
extern template struct block2::MultiMPS<block2::SZK, double>;

extern template struct block2::MultiMPSInfo<block2::SU2K>;
extern template struct block2::MultiMPS<block2::SU2K, double>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZK, double, double>;
extern template struct block2::Linear<block2::SZK, double, double>;
extern template struct block2::Expect<block2::SZK, double, double, double>;
extern template struct block2::Expect<block2::SZK, double, double,
                                      complex<double>>;

extern template struct block2::DMRG<block2::SU2K, double, double>;
extern template struct block2::Linear<block2::SU2K, double, double>;
extern template struct block2::Expect<block2::SU2K, double, double, double>;
extern template struct block2::Expect<block2::SU2K, double, double,
                                      complex<double>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZK, double, double>;
extern template struct block2::TimeEvolution<block2::SZK, double, double>;

extern template struct block2::TDDMRG<block2::SU2K, double, double>;
extern template struct block2::TimeEvolution<block2::SU2K, double, double>;

#endif

#ifdef _USE_SG

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SGF, double>;
extern template struct block2::ArchivedMPO<block2::SGB, double>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SGF, double>;
extern template struct block2::DeterminantTRIE<block2::SGB, double>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SGF, double>;
extern template struct block2::DMRGDriver<block2::SGB, double>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SGF, double>;
extern template struct block2::EffectiveFunctions<block2::SGB, double>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, double, block2::MPS<block2::SGF, double>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGF, double>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, double, block2::MultiMPS<block2::SGF, double>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SGB, double, block2::MPS<block2::SGB, double>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGB, double>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGB, double, block2::MultiMPS<block2::SGB, double>>;

// general_mpo.hpp
extern template struct block2::GeneralHamiltonian<block2::SGF, double>;
extern template struct block2::GeneralMPO<block2::SGF, double>;

extern template struct block2::GeneralHamiltonian<block2::SGB, double>;
extern template struct block2::GeneralMPO<block2::SGB, double>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SGF, double>;
extern template struct block2::GeneralNPDMMPO<block2::SGB, double>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SGF, double, double>;
extern template struct block2::MovingEnvironment<block2::SGB, double, double>;

// mpo.hpp
extern template struct block2::MPOSchemer<block2::SGF>;
extern template struct block2::MPO<block2::SGF, double>;
extern template struct block2::DiagonalMPO<block2::SGF, double>;
extern template struct block2::AncillaMPO<block2::SGF, double>;
extern template struct block2::IdentityAddedMPO<block2::SGF, double>;

extern template struct block2::MPOSchemer<block2::SGB>;
extern template struct block2::MPO<block2::SGB, double>;
extern template struct block2::DiagonalMPO<block2::SGB, double>;
extern template struct block2::AncillaMPO<block2::SGB, double>;
extern template struct block2::IdentityAddedMPO<block2::SGB, double>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SGF, double>;
extern template struct block2::FusedMPO<block2::SGF, double>;

extern template struct block2::CondensedMPO<block2::SGB, double>;
extern template struct block2::FusedMPO<block2::SGB, double>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SGF, double>;
extern template struct block2::SimplifiedMPO<block2::SGB, double>;

// mps.hpp
extern template struct block2::MPSInfo<block2::SGF>;
extern template struct block2::DynamicMPSInfo<block2::SGF>;
extern template struct block2::CASCIMPSInfo<block2::SGF>;
extern template struct block2::MRCIMPSInfo<block2::SGF>;
extern template struct block2::NEVPTMPSInfo<block2::SGF>;
extern template struct block2::AncillaMPSInfo<block2::SGF>;
extern template struct block2::MPS<block2::SGF, double>;

extern template struct block2::MPSInfo<block2::SGB>;
extern template struct block2::DynamicMPSInfo<block2::SGB>;
extern template struct block2::CASCIMPSInfo<block2::SGB>;
extern template struct block2::MRCIMPSInfo<block2::SGB>;
extern template struct block2::NEVPTMPSInfo<block2::SGB>;
extern template struct block2::AncillaMPSInfo<block2::SGB>;
extern template struct block2::MPS<block2::SGB, double>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SGF, double>;
extern template struct block2::UnfusedMPS<block2::SGF, double>;

extern template struct block2::SparseTensor<block2::SGB, double>;
extern template struct block2::UnfusedMPS<block2::SGB, double>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SGF, double>;
extern template struct block2::ParallelMPO<block2::SGF, double>;

extern template struct block2::ClassicParallelMPO<block2::SGB, double>;
extern template struct block2::ParallelMPO<block2::SGB, double>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SGF, double>;
extern template struct block2::ParallelMPS<block2::SGB, double>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SGF, double>;
extern template struct block2::SumMPORule<block2::SGF, double>;
extern template struct block2::ParallelFCIDUMP<block2::SGF, double>;

extern template struct block2::ParallelRuleSimple<block2::SGB, double>;
extern template struct block2::SumMPORule<block2::SGB, double>;
extern template struct block2::ParallelFCIDUMP<block2::SGB, double>;

// partition.hpp
extern template struct block2::Partition<block2::SGF, double>;
extern template struct block2::Partition<block2::SGB, double>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SGF, double>;
extern template struct block2::HamiltonianQC<block2::SGB, double>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SGF, double>;
extern template struct block2::SiteMPO<block2::SGF, double>;
extern template struct block2::MPOQC<block2::SGF, double>;

extern template struct block2::IdentityMPO<block2::SGB, double>;
extern template struct block2::SiteMPO<block2::SGB, double>;
extern template struct block2::MPOQC<block2::SGB, double>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SGF, double>;
extern template struct block2::NPC1MPOQC<block2::SGB, double>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SGF, double>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGF, double>;
extern template struct block2::ParallelRulePDM1QC<block2::SGF, double>;
extern template struct block2::ParallelRulePDM2QC<block2::SGF, double>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGF, double>;
extern template struct block2::ParallelRuleSiteQC<block2::SGF, double>;
extern template struct block2::ParallelRuleIdentity<block2::SGF, double>;

extern template struct block2::ParallelRuleQC<block2::SGB, double>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGB, double>;
extern template struct block2::ParallelRulePDM1QC<block2::SGB, double>;
extern template struct block2::ParallelRulePDM2QC<block2::SGB, double>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGB, double>;
extern template struct block2::ParallelRuleSiteQC<block2::SGB, double>;
extern template struct block2::ParallelRuleIdentity<block2::SGB, double>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SGF, double>;
extern template struct block2::PDM1MPOQC<block2::SGB, double>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SGF, double>;
extern template struct block2::PDM2MPOQC<block2::SGB, double>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SGF, double>;
extern template struct block2::AntiHermitianRuleQC<block2::SGF, double>;

extern template struct block2::RuleQC<block2::SGB, double>;
extern template struct block2::AntiHermitianRuleQC<block2::SGB, double>;

// state_averaged.hpp
extern template struct block2::MultiMPSInfo<block2::SGF>;
extern template struct block2::MultiMPS<block2::SGF, double>;

extern template struct block2::MultiMPSInfo<block2::SGB>;
extern template struct block2::MultiMPS<block2::SGB, double>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SGF, double, double>;
extern template struct block2::Linear<block2::SGF, double, double>;
extern template struct block2::Expect<block2::SGF, double, double, double>;
extern template struct block2::Expect<block2::SGF, double, double,
                                      complex<double>>;

extern template struct block2::DMRG<block2::SGB, double, double>;
extern template struct block2::Linear<block2::SGB, double, double>;
extern template struct block2::Expect<block2::SGB, double, double, double>;
extern template struct block2::Expect<block2::SGB, double, double,
                                      complex<double>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SGF, double, double>;
extern template struct block2::TimeEvolution<block2::SGF, double, double>;

extern template struct block2::TDDMRG<block2::SGB, double, double>;
extern template struct block2::TimeEvolution<block2::SGB, double, double>;

#endif

#ifdef _USE_COMPLEX

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZ, complex<double>>;
extern template struct block2::ArchivedMPO<block2::SU2, complex<double>>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZ, complex<double>>;
extern template struct block2::DeterminantQC<block2::SZ, complex<double>>;
extern template struct block2::DeterminantMPSInfo<block2::SZ, complex<double>>;

extern template struct block2::DeterminantTRIE<block2::SU2, complex<double>>;
extern template struct block2::DeterminantQC<block2::SU2, complex<double>>;
extern template struct block2::DeterminantMPSInfo<block2::SU2, complex<double>>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SZ, complex<double>>;
extern template struct block2::DMRGDriver<block2::SU2, complex<double>>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SZ, complex<double>>;
extern template struct block2::EffectiveFunctions<block2::SU2, complex<double>>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, complex<double>, block2::MPS<block2::SZ, complex<double>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZ,
                                                          complex<double>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, complex<double>, block2::MultiMPS<block2::SZ, complex<double>>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SU2, complex<double>, block2::MPS<block2::SU2, complex<double>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2,
                                                          complex<double>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2, complex<double>,
    block2::MultiMPS<block2::SU2, complex<double>>>;

// general_mpo.hpp
extern template struct block2::GeneralFCIDUMP<complex<double>>;

extern template struct block2::GeneralHamiltonian<block2::SZ, complex<double>>;
extern template struct block2::GeneralMPO<block2::SZ, complex<double>>;

extern template struct block2::GeneralHamiltonian<block2::SU2, complex<double>>;
extern template struct block2::GeneralMPO<block2::SU2, complex<double>>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SZ, complex<double>>;
extern template struct block2::GeneralNPDMMPO<block2::SU2, complex<double>>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZ, complex<double>,
                                                 complex<double>>;
extern template struct block2::MovingEnvironment<block2::SU2, complex<double>,
                                                 complex<double>>;
extern template struct block2::MovingEnvironment<block2::SZ, complex<double>,
                                                 double>;
extern template struct block2::MovingEnvironment<block2::SU2, complex<double>,
                                                 double>;

// mpo.hpp
extern template struct block2::MPO<block2::SZ, complex<double>>;
extern template struct block2::DiagonalMPO<block2::SZ, complex<double>>;
extern template struct block2::AncillaMPO<block2::SZ, complex<double>>;
extern template struct block2::IdentityAddedMPO<block2::SZ, complex<double>>;

extern template struct block2::MPO<block2::SU2, complex<double>>;
extern template struct block2::DiagonalMPO<block2::SU2, complex<double>>;
extern template struct block2::AncillaMPO<block2::SU2, complex<double>>;
extern template struct block2::IdentityAddedMPO<block2::SU2, complex<double>>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SZ, complex<double>>;
extern template struct block2::FusedMPO<block2::SZ, complex<double>>;

extern template struct block2::CondensedMPO<block2::SU2, complex<double>>;
extern template struct block2::FusedMPO<block2::SU2, complex<double>>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZ, complex<double>>;
extern template struct block2::SimplifiedMPO<block2::SU2, complex<double>>;

// mps.hpp
extern template struct block2::MPS<block2::SZ, complex<double>>;
extern template struct block2::MPS<block2::SU2, complex<double>>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZ, complex<double>>;
extern template struct block2::UnfusedMPS<block2::SZ, complex<double>>;

extern template struct block2::SparseTensor<block2::SU2, complex<double>>;
extern template struct block2::UnfusedMPS<block2::SU2, complex<double>>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZ, complex<double>>;
extern template struct block2::ParallelMPO<block2::SZ, complex<double>>;

extern template struct block2::ClassicParallelMPO<block2::SU2, complex<double>>;
extern template struct block2::ParallelMPO<block2::SU2, complex<double>>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZ, complex<double>>;
extern template struct block2::ParallelMPS<block2::SU2, complex<double>>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SZ, complex<double>>;
extern template struct block2::SumMPORule<block2::SZ, complex<double>>;
extern template struct block2::ParallelFCIDUMP<block2::SZ, complex<double>>;

extern template struct block2::ParallelRuleSimple<block2::SU2, complex<double>>;
extern template struct block2::SumMPORule<block2::SU2, complex<double>>;
extern template struct block2::ParallelFCIDUMP<block2::SU2, complex<double>>;

// partition.hpp
extern template struct block2::Partition<block2::SZ, complex<double>>;
extern template struct block2::Partition<block2::SU2, complex<double>>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZ, complex<double>>;
extern template struct block2::HamiltonianQC<block2::SU2, complex<double>>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZ, complex<double>>;
extern template struct block2::SiteMPO<block2::SZ, complex<double>>;
extern template struct block2::MPOQC<block2::SZ, complex<double>>;

extern template struct block2::IdentityMPO<block2::SU2, complex<double>>;
extern template struct block2::SiteMPO<block2::SU2, complex<double>>;
extern template struct block2::MPOQC<block2::SU2, complex<double>>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZ, complex<double>>;
extern template struct block2::NPC1MPOQC<block2::SU2, complex<double>>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZ, complex<double>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SZ,
                                                     complex<double>>;
extern template struct block2::ParallelRulePDM1QC<block2::SZ, complex<double>>;
extern template struct block2::ParallelRulePDM2QC<block2::SZ, complex<double>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZ, complex<double>>;
extern template struct block2::ParallelRuleSiteQC<block2::SZ, complex<double>>;
extern template struct block2::ParallelRuleIdentity<block2::SZ,
                                                    complex<double>>;

extern template struct block2::ParallelRuleQC<block2::SU2, complex<double>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2,
                                                     complex<double>>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2, complex<double>>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2, complex<double>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2, complex<double>>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2, complex<double>>;
extern template struct block2::ParallelRuleIdentity<block2::SU2,
                                                    complex<double>>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZ, complex<double>>;
extern template struct block2::PDM1MPOQC<block2::SU2, complex<double>>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZ, complex<double>>;
extern template struct block2::PDM2MPOQC<block2::SU2, complex<double>>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZ, complex<double>>;
extern template struct block2::AntiHermitianRuleQC<block2::SZ, complex<double>>;

extern template struct block2::RuleQC<block2::SU2, complex<double>>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2,
                                                   complex<double>>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZ, complex<double>>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SZ, complex<double>>;
extern template struct block2::MultiMPS<block2::SU2, complex<double>>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZ, complex<double>,
                                    complex<double>>;
extern template struct block2::Linear<block2::SZ, complex<double>,
                                      complex<double>>;
extern template struct block2::Expect<block2::SZ, complex<double>,
                                      complex<double>, complex<double>>;

extern template struct block2::DMRG<block2::SU2, complex<double>,
                                    complex<double>>;
extern template struct block2::Linear<block2::SU2, complex<double>,
                                      complex<double>>;
extern template struct block2::Expect<block2::SU2, complex<double>,
                                      complex<double>, complex<double>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZ, complex<double>,
                                      complex<double>>;
extern template struct block2::TimeEvolution<block2::SZ, complex<double>,
                                             complex<double>>;

extern template struct block2::TDDMRG<block2::SU2, complex<double>,
                                      complex<double>>;
extern template struct block2::TimeEvolution<block2::SU2, complex<double>,
                                             complex<double>>;

#ifdef _USE_KSYMM

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZK, complex<double>>;
extern template struct block2::ArchivedMPO<block2::SU2K, complex<double>>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZK, complex<double>>;
extern template struct block2::DeterminantQC<block2::SZK, complex<double>>;
extern template struct block2::DeterminantMPSInfo<block2::SZK, complex<double>>;

extern template struct block2::DeterminantTRIE<block2::SU2K, complex<double>>;
extern template struct block2::DeterminantQC<block2::SU2K, complex<double>>;
extern template struct block2::DeterminantMPSInfo<block2::SU2K,
                                                  complex<double>>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SZK, complex<double>>;
extern template struct block2::DMRGDriver<block2::SU2K, complex<double>>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SZK, complex<double>>;
extern template struct block2::EffectiveFunctions<block2::SU2K,
                                                  complex<double>>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SZK, complex<double>, block2::MPS<block2::SZK, complex<double>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZK,
                                                          complex<double>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZK, complex<double>,
    block2::MultiMPS<block2::SZK, complex<double>>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SU2K, complex<double>, block2::MPS<block2::SU2K, complex<double>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2K,
                                                          complex<double>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2K, complex<double>,
    block2::MultiMPS<block2::SU2K, complex<double>>>;

// general_mpo.hpp
extern template struct block2::GeneralHamiltonian<block2::SZK, complex<double>>;
extern template struct block2::GeneralMPO<block2::SZK, complex<double>>;

extern template struct block2::GeneralHamiltonian<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::GeneralMPO<block2::SU2K, complex<double>>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SZK, complex<double>>;
extern template struct block2::GeneralNPDMMPO<block2::SU2K, complex<double>>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZK, complex<double>,
                                                 complex<double>>;
extern template struct block2::MovingEnvironment<block2::SU2K, complex<double>,
                                                 complex<double>>;
extern template struct block2::MovingEnvironment<block2::SZK, complex<double>,
                                                 double>;
extern template struct block2::MovingEnvironment<block2::SU2K, complex<double>,
                                                 double>;

// mpo.hpp
extern template struct block2::MPO<block2::SZK, complex<double>>;
extern template struct block2::DiagonalMPO<block2::SZK, complex<double>>;
extern template struct block2::AncillaMPO<block2::SZK, complex<double>>;
extern template struct block2::IdentityAddedMPO<block2::SZK, complex<double>>;

extern template struct block2::MPO<block2::SU2K, complex<double>>;
extern template struct block2::DiagonalMPO<block2::SU2K, complex<double>>;
extern template struct block2::AncillaMPO<block2::SU2K, complex<double>>;
extern template struct block2::IdentityAddedMPO<block2::SU2K, complex<double>>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SZK, complex<double>>;
extern template struct block2::FusedMPO<block2::SZK, complex<double>>;

extern template struct block2::CondensedMPO<block2::SU2K, complex<double>>;
extern template struct block2::FusedMPO<block2::SU2K, complex<double>>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZK, complex<double>>;
extern template struct block2::SimplifiedMPO<block2::SU2K, complex<double>>;

// mps.hpp
extern template struct block2::MPS<block2::SZK, complex<double>>;
extern template struct block2::MPS<block2::SU2K, complex<double>>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZK, complex<double>>;
extern template struct block2::UnfusedMPS<block2::SZK, complex<double>>;

extern template struct block2::SparseTensor<block2::SU2K, complex<double>>;
extern template struct block2::UnfusedMPS<block2::SU2K, complex<double>>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZK, complex<double>>;
extern template struct block2::ParallelMPO<block2::SZK, complex<double>>;

extern template struct block2::ClassicParallelMPO<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::ParallelMPO<block2::SU2K, complex<double>>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZK, complex<double>>;
extern template struct block2::ParallelMPS<block2::SU2K, complex<double>>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SZK, complex<double>>;
extern template struct block2::SumMPORule<block2::SZK, complex<double>>;
extern template struct block2::ParallelFCIDUMP<block2::SZK, complex<double>>;

extern template struct block2::ParallelRuleSimple<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::SumMPORule<block2::SU2K, complex<double>>;
extern template struct block2::ParallelFCIDUMP<block2::SU2K, complex<double>>;

// partition.hpp
extern template struct block2::Partition<block2::SZK, complex<double>>;
extern template struct block2::Partition<block2::SU2K, complex<double>>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZK, complex<double>>;
extern template struct block2::HamiltonianQC<block2::SU2K, complex<double>>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZK, complex<double>>;
extern template struct block2::SiteMPO<block2::SZK, complex<double>>;
extern template struct block2::MPOQC<block2::SZK, complex<double>>;

extern template struct block2::IdentityMPO<block2::SU2K, complex<double>>;
extern template struct block2::SiteMPO<block2::SU2K, complex<double>>;
extern template struct block2::MPOQC<block2::SU2K, complex<double>>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZK, complex<double>>;
extern template struct block2::NPC1MPOQC<block2::SU2K, complex<double>>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZK, complex<double>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SZK,
                                                     complex<double>>;
extern template struct block2::ParallelRulePDM1QC<block2::SZK, complex<double>>;
extern template struct block2::ParallelRulePDM2QC<block2::SZK, complex<double>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZK, complex<double>>;
extern template struct block2::ParallelRuleSiteQC<block2::SZK, complex<double>>;
extern template struct block2::ParallelRuleIdentity<block2::SZK,
                                                    complex<double>>;

extern template struct block2::ParallelRuleQC<block2::SU2K, complex<double>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2K,
                                                     complex<double>>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2K,
                                                  complex<double>>;
extern template struct block2::ParallelRuleIdentity<block2::SU2K,
                                                    complex<double>>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZK, complex<double>>;
extern template struct block2::PDM1MPOQC<block2::SU2K, complex<double>>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZK, complex<double>>;
extern template struct block2::PDM2MPOQC<block2::SU2K, complex<double>>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZK, complex<double>>;
extern template struct block2::AntiHermitianRuleQC<block2::SZK,
                                                   complex<double>>;

extern template struct block2::RuleQC<block2::SU2K, complex<double>>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2K,
                                                   complex<double>>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZK, complex<double>>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SZK, complex<double>>;
extern template struct block2::MultiMPS<block2::SU2K, complex<double>>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZK, complex<double>,
                                    complex<double>>;
extern template struct block2::Linear<block2::SZK, complex<double>,
                                      complex<double>>;
extern template struct block2::Expect<block2::SZK, complex<double>,
                                      complex<double>, complex<double>>;

extern template struct block2::DMRG<block2::SU2K, complex<double>,
                                    complex<double>>;
extern template struct block2::Linear<block2::SU2K, complex<double>,
                                      complex<double>>;
extern template struct block2::Expect<block2::SU2K, complex<double>,
                                      complex<double>, complex<double>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZK, complex<double>,
                                      complex<double>>;
extern template struct block2::TimeEvolution<block2::SZK, complex<double>,
                                             complex<double>>;

extern template struct block2::TDDMRG<block2::SU2K, complex<double>,
                                      complex<double>>;
extern template struct block2::TimeEvolution<block2::SU2K, complex<double>,
                                             complex<double>>;

#endif

#ifdef _USE_SG

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SGF, complex<double>>;
extern template struct block2::ArchivedMPO<block2::SGB, complex<double>>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SGF, complex<double>>;
extern template struct block2::DeterminantTRIE<block2::SGB, complex<double>>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SGF, complex<double>>;
extern template struct block2::DMRGDriver<block2::SGB, complex<double>>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SGF, complex<double>>;
extern template struct block2::EffectiveFunctions<block2::SGB, complex<double>>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, complex<double>, block2::MPS<block2::SGF, complex<double>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGF,
                                                          complex<double>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, complex<double>,
    block2::MultiMPS<block2::SGF, complex<double>>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SGB, complex<double>, block2::MPS<block2::SGB, complex<double>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGB,
                                                          complex<double>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGB, complex<double>,
    block2::MultiMPS<block2::SGB, complex<double>>>;

// general_mpo.hpp
extern template struct block2::GeneralHamiltonian<block2::SGF, complex<double>>;
extern template struct block2::GeneralMPO<block2::SGF, complex<double>>;

extern template struct block2::GeneralHamiltonian<block2::SGB, complex<double>>;
extern template struct block2::GeneralMPO<block2::SGB, complex<double>>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SGF, complex<double>>;
extern template struct block2::GeneralNPDMMPO<block2::SGB, complex<double>>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SGF, complex<double>,
                                                 complex<double>>;
extern template struct block2::MovingEnvironment<block2::SGB, complex<double>,
                                                 complex<double>>;
extern template struct block2::MovingEnvironment<block2::SGF, complex<double>,
                                                 double>;
extern template struct block2::MovingEnvironment<block2::SGB, complex<double>,
                                                 double>;

// mpo.hpp
extern template struct block2::MPO<block2::SGF, complex<double>>;
extern template struct block2::DiagonalMPO<block2::SGF, complex<double>>;
extern template struct block2::AncillaMPO<block2::SGF, complex<double>>;
extern template struct block2::IdentityAddedMPO<block2::SGF, complex<double>>;

extern template struct block2::MPO<block2::SGB, complex<double>>;
extern template struct block2::DiagonalMPO<block2::SGB, complex<double>>;
extern template struct block2::AncillaMPO<block2::SGB, complex<double>>;
extern template struct block2::IdentityAddedMPO<block2::SGB, complex<double>>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SGF, complex<double>>;
extern template struct block2::FusedMPO<block2::SGF, complex<double>>;

extern template struct block2::CondensedMPO<block2::SGB, complex<double>>;
extern template struct block2::FusedMPO<block2::SGB, complex<double>>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SGF, complex<double>>;
extern template struct block2::SimplifiedMPO<block2::SGB, complex<double>>;

// mps.hpp
extern template struct block2::MPS<block2::SGF, complex<double>>;
extern template struct block2::MPS<block2::SGB, complex<double>>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SGF, complex<double>>;
extern template struct block2::UnfusedMPS<block2::SGF, complex<double>>;

extern template struct block2::SparseTensor<block2::SGB, complex<double>>;
extern template struct block2::UnfusedMPS<block2::SGB, complex<double>>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SGF, complex<double>>;
extern template struct block2::ParallelMPO<block2::SGF, complex<double>>;

extern template struct block2::ClassicParallelMPO<block2::SGB, complex<double>>;
extern template struct block2::ParallelMPO<block2::SGB, complex<double>>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SGF, complex<double>>;
extern template struct block2::ParallelMPS<block2::SGB, complex<double>>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SGF, complex<double>>;
extern template struct block2::SumMPORule<block2::SGF, complex<double>>;
extern template struct block2::ParallelFCIDUMP<block2::SGF, complex<double>>;

extern template struct block2::ParallelRuleSimple<block2::SGB, complex<double>>;
extern template struct block2::SumMPORule<block2::SGB, complex<double>>;
extern template struct block2::ParallelFCIDUMP<block2::SGB, complex<double>>;

// partition.hpp
extern template struct block2::Partition<block2::SGF, complex<double>>;
extern template struct block2::Partition<block2::SGB, complex<double>>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SGF, complex<double>>;
extern template struct block2::HamiltonianQC<block2::SGB, complex<double>>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SGF, complex<double>>;
extern template struct block2::SiteMPO<block2::SGF, complex<double>>;
extern template struct block2::MPOQC<block2::SGF, complex<double>>;

extern template struct block2::IdentityMPO<block2::SGB, complex<double>>;
extern template struct block2::SiteMPO<block2::SGB, complex<double>>;
extern template struct block2::MPOQC<block2::SGB, complex<double>>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SGF, complex<double>>;
extern template struct block2::NPC1MPOQC<block2::SGB, complex<double>>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SGF, complex<double>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGF,
                                                     complex<double>>;
extern template struct block2::ParallelRulePDM1QC<block2::SGF, complex<double>>;
extern template struct block2::ParallelRulePDM2QC<block2::SGF, complex<double>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGF, complex<double>>;
extern template struct block2::ParallelRuleSiteQC<block2::SGF, complex<double>>;
extern template struct block2::ParallelRuleIdentity<block2::SGF,
                                                    complex<double>>;

extern template struct block2::ParallelRuleQC<block2::SGB, complex<double>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGB,
                                                     complex<double>>;
extern template struct block2::ParallelRulePDM1QC<block2::SGB, complex<double>>;
extern template struct block2::ParallelRulePDM2QC<block2::SGB, complex<double>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGB, complex<double>>;
extern template struct block2::ParallelRuleSiteQC<block2::SGB, complex<double>>;
extern template struct block2::ParallelRuleIdentity<block2::SGB,
                                                    complex<double>>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SGF, complex<double>>;
extern template struct block2::PDM1MPOQC<block2::SGB, complex<double>>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SGF, complex<double>>;
extern template struct block2::PDM2MPOQC<block2::SGB, complex<double>>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SGF, complex<double>>;
extern template struct block2::AntiHermitianRuleQC<block2::SGF,
                                                   complex<double>>;

extern template struct block2::RuleQC<block2::SGB, complex<double>>;
extern template struct block2::AntiHermitianRuleQC<block2::SGB,
                                                   complex<double>>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SGF, complex<double>>;
extern template struct block2::MultiMPS<block2::SGB, complex<double>>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SGF, complex<double>,
                                    complex<double>>;
extern template struct block2::Linear<block2::SGF, complex<double>,
                                      complex<double>>;
extern template struct block2::Expect<block2::SGF, complex<double>,
                                      complex<double>, complex<double>>;

extern template struct block2::DMRG<block2::SGB, complex<double>,
                                    complex<double>>;
extern template struct block2::Linear<block2::SGB, complex<double>,
                                      complex<double>>;
extern template struct block2::Expect<block2::SGB, complex<double>,
                                      complex<double>, complex<double>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SGF, complex<double>,
                                      complex<double>>;
extern template struct block2::TimeEvolution<block2::SGF, complex<double>,
                                             complex<double>>;

extern template struct block2::TDDMRG<block2::SGB, complex<double>,
                                      complex<double>>;
extern template struct block2::TimeEvolution<block2::SGB, complex<double>,
                                             complex<double>>;

#endif

#endif

#ifdef _USE_SINGLE_PREC

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZ, float>;
extern template struct block2::ArchivedMPO<block2::SU2, float>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZ, float>;
extern template struct block2::DeterminantQC<block2::SZ, float>;
extern template struct block2::DeterminantMPSInfo<block2::SZ, float>;

extern template struct block2::DeterminantTRIE<block2::SU2, float>;
extern template struct block2::DeterminantQC<block2::SU2, float>;
extern template struct block2::DeterminantMPSInfo<block2::SU2, float>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SZ, float>;
extern template struct block2::DMRGDriver<block2::SU2, float>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SZ, float>;
extern template struct block2::EffectiveFunctions<block2::SU2, float>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, float, block2::MPS<block2::SZ, float>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZ, float>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, float, block2::MultiMPS<block2::SZ, float>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SU2, float, block2::MPS<block2::SU2, float>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2, float>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2, float, block2::MultiMPS<block2::SU2, float>>;

// general_mpo.hpp
extern template struct block2::GeneralFCIDUMP<float>;

extern template struct block2::GeneralHamiltonian<block2::SZ, float>;
extern template struct block2::GeneralMPO<block2::SZ, float>;

extern template struct block2::GeneralHamiltonian<block2::SU2, float>;
extern template struct block2::GeneralMPO<block2::SU2, float>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SZ, float>;
extern template struct block2::GeneralNPDMMPO<block2::SU2, float>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZ, float, float>;
extern template struct block2::MovingEnvironment<block2::SU2, float, float>;

// mpo.hpp
extern template struct block2::MPO<block2::SZ, float>;
extern template struct block2::DiagonalMPO<block2::SZ, float>;
extern template struct block2::AncillaMPO<block2::SZ, float>;
extern template struct block2::IdentityAddedMPO<block2::SZ, float>;

extern template struct block2::MPO<block2::SU2, float>;
extern template struct block2::DiagonalMPO<block2::SU2, float>;
extern template struct block2::AncillaMPO<block2::SU2, float>;
extern template struct block2::IdentityAddedMPO<block2::SU2, float>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SZ, float>;
extern template struct block2::FusedMPO<block2::SZ, float>;

extern template struct block2::CondensedMPO<block2::SU2, float>;
extern template struct block2::FusedMPO<block2::SU2, float>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZ, float>;
extern template struct block2::SimplifiedMPO<block2::SU2, float>;

// mps.hpp
extern template struct block2::MPS<block2::SZ, float>;
extern template struct block2::MPS<block2::SU2, float>;

extern template struct block2::TransMPS<block2::SZ, double, float>;
extern template struct block2::TransMPS<block2::SZ, float, double>;
extern template struct block2::TransMPS<block2::SU2, double, float>;
extern template struct block2::TransMPS<block2::SU2, float, double>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZ, float>;
extern template struct block2::UnfusedMPS<block2::SZ, float>;

extern template struct block2::SparseTensor<block2::SU2, float>;
extern template struct block2::UnfusedMPS<block2::SU2, float>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZ, float>;
extern template struct block2::ParallelMPO<block2::SZ, float>;

extern template struct block2::ClassicParallelMPO<block2::SU2, float>;
extern template struct block2::ParallelMPO<block2::SU2, float>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZ, float>;
extern template struct block2::ParallelMPS<block2::SU2, float>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SZ, float>;
extern template struct block2::SumMPORule<block2::SZ, float>;
extern template struct block2::ParallelFCIDUMP<block2::SZ, float>;

extern template struct block2::ParallelRuleSimple<block2::SU2, float>;
extern template struct block2::SumMPORule<block2::SU2, float>;
extern template struct block2::ParallelFCIDUMP<block2::SU2, float>;

// partition.hpp
extern template struct block2::Partition<block2::SZ, float>;
extern template struct block2::Partition<block2::SU2, float>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZ, float>;
extern template struct block2::HamiltonianQC<block2::SU2, float>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZ, float>;
extern template struct block2::SiteMPO<block2::SZ, float>;
extern template struct block2::MPOQC<block2::SZ, float>;

extern template struct block2::IdentityMPO<block2::SU2, float>;
extern template struct block2::SiteMPO<block2::SU2, float>;
extern template struct block2::MPOQC<block2::SU2, float>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZ, float>;
extern template struct block2::NPC1MPOQC<block2::SU2, float>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZ, float>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SZ, float>;
extern template struct block2::ParallelRulePDM1QC<block2::SZ, float>;
extern template struct block2::ParallelRulePDM2QC<block2::SZ, float>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZ, float>;
extern template struct block2::ParallelRuleSiteQC<block2::SZ, float>;
extern template struct block2::ParallelRuleIdentity<block2::SZ, float>;

extern template struct block2::ParallelRuleQC<block2::SU2, float>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2, float>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2, float>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2, float>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2, float>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2, float>;
extern template struct block2::ParallelRuleIdentity<block2::SU2, float>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZ, float>;
extern template struct block2::PDM1MPOQC<block2::SU2, float>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZ, float>;
extern template struct block2::PDM2MPOQC<block2::SU2, float>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZ, float>;
extern template struct block2::AntiHermitianRuleQC<block2::SZ, float>;

extern template struct block2::RuleQC<block2::SU2, float>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2, float>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZ, float>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SZ, float>;
extern template struct block2::MultiMPS<block2::SU2, float>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZ, float, float>;
extern template struct block2::Linear<block2::SZ, float, float>;
extern template struct block2::Expect<block2::SZ, float, float, float>;
extern template struct block2::Expect<block2::SZ, float, float, complex<float>>;

extern template struct block2::DMRG<block2::SU2, float, float>;
extern template struct block2::Linear<block2::SU2, float, float>;
extern template struct block2::Expect<block2::SU2, float, float, float>;
extern template struct block2::Expect<block2::SU2, float, float,
                                      complex<float>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZ, float, float>;
extern template struct block2::TimeEvolution<block2::SZ, float, float>;

extern template struct block2::TDDMRG<block2::SU2, float, float>;
extern template struct block2::TimeEvolution<block2::SU2, float, float>;

#ifdef _USE_SG

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SGF, float>;
extern template struct block2::ArchivedMPO<block2::SGB, float>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SGF, float>;
extern template struct block2::DeterminantTRIE<block2::SGB, float>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SGF, float>;
extern template struct block2::DMRGDriver<block2::SGB, float>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SGF, float>;
extern template struct block2::EffectiveFunctions<block2::SGB, float>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, float, block2::MPS<block2::SGF, float>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGF, float>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, float, block2::MultiMPS<block2::SGF, float>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SGB, float, block2::MPS<block2::SGB, float>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGB, float>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGB, float, block2::MultiMPS<block2::SGB, float>>;

// general_mpo.hpp
extern template struct block2::GeneralHamiltonian<block2::SGF, float>;
extern template struct block2::GeneralMPO<block2::SGF, float>;

extern template struct block2::GeneralHamiltonian<block2::SGB, float>;
extern template struct block2::GeneralMPO<block2::SGB, float>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SGF, float>;
extern template struct block2::GeneralNPDMMPO<block2::SGB, float>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SGF, float, float>;
extern template struct block2::MovingEnvironment<block2::SGB, float, float>;

// mpo.hpp
extern template struct block2::MPO<block2::SGF, float>;
extern template struct block2::DiagonalMPO<block2::SGF, float>;
extern template struct block2::AncillaMPO<block2::SGF, float>;
extern template struct block2::IdentityAddedMPO<block2::SGF, float>;

extern template struct block2::MPO<block2::SGB, float>;
extern template struct block2::DiagonalMPO<block2::SGB, float>;
extern template struct block2::AncillaMPO<block2::SGB, float>;
extern template struct block2::IdentityAddedMPO<block2::SGB, float>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SGF, float>;
extern template struct block2::FusedMPO<block2::SGF, float>;

extern template struct block2::CondensedMPO<block2::SGB, float>;
extern template struct block2::FusedMPO<block2::SGB, float>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SGF, float>;
extern template struct block2::SimplifiedMPO<block2::SGB, float>;

// mps.hpp
extern template struct block2::MPS<block2::SGF, float>;
extern template struct block2::MPS<block2::SGB, float>;

extern template struct block2::TransMPS<block2::SGF, double, float>;
extern template struct block2::TransMPS<block2::SGF, float, double>;
extern template struct block2::TransMPS<block2::SGB, double, float>;
extern template struct block2::TransMPS<block2::SGB, float, double>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SGF, float>;
extern template struct block2::UnfusedMPS<block2::SGF, float>;

extern template struct block2::SparseTensor<block2::SGB, float>;
extern template struct block2::UnfusedMPS<block2::SGB, float>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SGF, float>;
extern template struct block2::ParallelMPO<block2::SGF, float>;

extern template struct block2::ClassicParallelMPO<block2::SGB, float>;
extern template struct block2::ParallelMPO<block2::SGB, float>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SGF, float>;
extern template struct block2::ParallelMPS<block2::SGB, float>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SGF, float>;
extern template struct block2::SumMPORule<block2::SGF, float>;
extern template struct block2::ParallelFCIDUMP<block2::SGF, float>;

extern template struct block2::ParallelRuleSimple<block2::SGB, float>;
extern template struct block2::SumMPORule<block2::SGB, float>;
extern template struct block2::ParallelFCIDUMP<block2::SGB, float>;

// partition.hpp
extern template struct block2::Partition<block2::SGF, float>;
extern template struct block2::Partition<block2::SGB, float>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SGF, float>;
extern template struct block2::HamiltonianQC<block2::SGB, float>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SGF, float>;
extern template struct block2::SiteMPO<block2::SGF, float>;
extern template struct block2::MPOQC<block2::SGF, float>;

extern template struct block2::IdentityMPO<block2::SGB, float>;
extern template struct block2::SiteMPO<block2::SGB, float>;
extern template struct block2::MPOQC<block2::SGB, float>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SGF, float>;
extern template struct block2::NPC1MPOQC<block2::SGB, float>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SGF, float>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGF, float>;
extern template struct block2::ParallelRulePDM1QC<block2::SGF, float>;
extern template struct block2::ParallelRulePDM2QC<block2::SGF, float>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGF, float>;
extern template struct block2::ParallelRuleSiteQC<block2::SGF, float>;
extern template struct block2::ParallelRuleIdentity<block2::SGF, float>;

extern template struct block2::ParallelRuleQC<block2::SGB, float>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGB, float>;
extern template struct block2::ParallelRulePDM1QC<block2::SGB, float>;
extern template struct block2::ParallelRulePDM2QC<block2::SGB, float>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGB, float>;
extern template struct block2::ParallelRuleSiteQC<block2::SGB, float>;
extern template struct block2::ParallelRuleIdentity<block2::SGB, float>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SGF, float>;
extern template struct block2::PDM1MPOQC<block2::SGB, float>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SGF, float>;
extern template struct block2::PDM2MPOQC<block2::SGB, float>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SGF, float>;
extern template struct block2::AntiHermitianRuleQC<block2::SGF, float>;

extern template struct block2::RuleQC<block2::SGB, float>;
extern template struct block2::AntiHermitianRuleQC<block2::SGB, float>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SGF, float>;
extern template struct block2::MultiMPS<block2::SGB, float>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SGF, float, float>;
extern template struct block2::Linear<block2::SGF, float, float>;
extern template struct block2::Expect<block2::SGF, float, float, float>;
extern template struct block2::Expect<block2::SGF, float, float,
                                      complex<float>>;

extern template struct block2::DMRG<block2::SGB, float, float>;
extern template struct block2::Linear<block2::SGB, float, float>;
extern template struct block2::Expect<block2::SGB, float, float, float>;
extern template struct block2::Expect<block2::SGB, float, float,
                                      complex<float>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SGF, float, float>;
extern template struct block2::TimeEvolution<block2::SGF, float, float>;

extern template struct block2::TDDMRG<block2::SGB, float, float>;
extern template struct block2::TimeEvolution<block2::SGB, float, float>;

#endif

#ifdef _USE_COMPLEX

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SZ, complex<float>>;
extern template struct block2::ArchivedMPO<block2::SU2, complex<float>>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SZ, complex<float>>;
extern template struct block2::DeterminantQC<block2::SZ, complex<float>>;
extern template struct block2::DeterminantMPSInfo<block2::SZ, complex<float>>;

extern template struct block2::DeterminantTRIE<block2::SU2, complex<float>>;
extern template struct block2::DeterminantQC<block2::SU2, complex<float>>;
extern template struct block2::DeterminantMPSInfo<block2::SU2, complex<float>>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SZ, complex<float>>;
extern template struct block2::DMRGDriver<block2::SU2, complex<float>>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SZ, complex<float>>;
extern template struct block2::EffectiveFunctions<block2::SU2, complex<float>>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, complex<float>, block2::MPS<block2::SZ, complex<float>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SZ,
                                                          complex<float>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SZ, complex<float>, block2::MultiMPS<block2::SZ, complex<float>>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SU2, complex<float>, block2::MPS<block2::SU2, complex<float>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SU2,
                                                          complex<float>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SU2, complex<float>, block2::MultiMPS<block2::SU2, complex<float>>>;

// general_mpo.hpp
extern template struct block2::GeneralFCIDUMP<complex<float>>;

extern template struct block2::GeneralHamiltonian<block2::SZ, complex<float>>;
extern template struct block2::GeneralMPO<block2::SZ, complex<float>>;

extern template struct block2::GeneralHamiltonian<block2::SU2, complex<float>>;
extern template struct block2::GeneralMPO<block2::SU2, complex<float>>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SZ, complex<float>>;
extern template struct block2::GeneralNPDMMPO<block2::SU2, complex<float>>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SZ, complex<float>,
                                                 complex<float>>;
extern template struct block2::MovingEnvironment<block2::SU2, complex<float>,
                                                 complex<float>>;
extern template struct block2::MovingEnvironment<block2::SZ, complex<float>,
                                                 float>;
extern template struct block2::MovingEnvironment<block2::SU2, complex<float>,
                                                 float>;

// mpo.hpp
extern template struct block2::MPO<block2::SZ, complex<float>>;
extern template struct block2::DiagonalMPO<block2::SZ, complex<float>>;
extern template struct block2::AncillaMPO<block2::SZ, complex<float>>;
extern template struct block2::IdentityAddedMPO<block2::SZ, complex<float>>;

extern template struct block2::MPO<block2::SU2, complex<float>>;
extern template struct block2::DiagonalMPO<block2::SU2, complex<float>>;
extern template struct block2::AncillaMPO<block2::SU2, complex<float>>;
extern template struct block2::IdentityAddedMPO<block2::SU2, complex<float>>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SZ, complex<float>>;
extern template struct block2::FusedMPO<block2::SZ, complex<float>>;

extern template struct block2::CondensedMPO<block2::SU2, complex<float>>;
extern template struct block2::FusedMPO<block2::SU2, complex<float>>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SZ, complex<float>>;
extern template struct block2::SimplifiedMPO<block2::SU2, complex<float>>;

// mps.hpp
extern template struct block2::MPS<block2::SZ, complex<float>>;
extern template struct block2::MPS<block2::SU2, complex<float>>;

extern template struct block2::TransMPS<block2::SZ, complex<double>,
                                        complex<float>>;
extern template struct block2::TransMPS<block2::SZ, complex<float>,
                                        complex<double>>;
extern template struct block2::TransMPS<block2::SU2, complex<double>,
                                        complex<float>>;
extern template struct block2::TransMPS<block2::SU2, complex<float>,
                                        complex<double>>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SZ, complex<float>>;
extern template struct block2::UnfusedMPS<block2::SZ, complex<float>>;

extern template struct block2::SparseTensor<block2::SU2, complex<float>>;
extern template struct block2::UnfusedMPS<block2::SU2, complex<float>>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SZ, complex<float>>;
extern template struct block2::ParallelMPO<block2::SZ, complex<float>>;

extern template struct block2::ClassicParallelMPO<block2::SU2, complex<float>>;
extern template struct block2::ParallelMPO<block2::SU2, complex<float>>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SZ, complex<float>>;
extern template struct block2::ParallelMPS<block2::SU2, complex<float>>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SZ, complex<float>>;
extern template struct block2::SumMPORule<block2::SZ, complex<float>>;
extern template struct block2::ParallelFCIDUMP<block2::SZ, complex<float>>;

extern template struct block2::ParallelRuleSimple<block2::SU2, complex<float>>;
extern template struct block2::SumMPORule<block2::SU2, complex<float>>;
extern template struct block2::ParallelFCIDUMP<block2::SU2, complex<float>>;

// partition.hpp
extern template struct block2::Partition<block2::SZ, complex<float>>;
extern template struct block2::Partition<block2::SU2, complex<float>>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SZ, complex<float>>;
extern template struct block2::HamiltonianQC<block2::SU2, complex<float>>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SZ, complex<float>>;
extern template struct block2::SiteMPO<block2::SZ, complex<float>>;
extern template struct block2::MPOQC<block2::SZ, complex<float>>;

extern template struct block2::IdentityMPO<block2::SU2, complex<float>>;
extern template struct block2::SiteMPO<block2::SU2, complex<float>>;
extern template struct block2::MPOQC<block2::SU2, complex<float>>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SZ, complex<float>>;
extern template struct block2::NPC1MPOQC<block2::SU2, complex<float>>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SZ, complex<float>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SZ,
                                                     complex<float>>;
extern template struct block2::ParallelRulePDM1QC<block2::SZ, complex<float>>;
extern template struct block2::ParallelRulePDM2QC<block2::SZ, complex<float>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SZ, complex<float>>;
extern template struct block2::ParallelRuleSiteQC<block2::SZ, complex<float>>;
extern template struct block2::ParallelRuleIdentity<block2::SZ, complex<float>>;

extern template struct block2::ParallelRuleQC<block2::SU2, complex<float>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SU2,
                                                     complex<float>>;
extern template struct block2::ParallelRulePDM1QC<block2::SU2, complex<float>>;
extern template struct block2::ParallelRulePDM2QC<block2::SU2, complex<float>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SU2, complex<float>>;
extern template struct block2::ParallelRuleSiteQC<block2::SU2, complex<float>>;
extern template struct block2::ParallelRuleIdentity<block2::SU2,
                                                    complex<float>>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SZ, complex<float>>;
extern template struct block2::PDM1MPOQC<block2::SU2, complex<float>>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SZ, complex<float>>;
extern template struct block2::PDM2MPOQC<block2::SU2, complex<float>>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SZ, complex<float>>;
extern template struct block2::AntiHermitianRuleQC<block2::SZ, complex<float>>;

extern template struct block2::RuleQC<block2::SU2, complex<float>>;
extern template struct block2::AntiHermitianRuleQC<block2::SU2, complex<float>>;

// qc_sum_mpo.hpp
extern template struct block2::SumMPOQC<block2::SZ, complex<float>>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SZ, complex<float>>;
extern template struct block2::MultiMPS<block2::SU2, complex<float>>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SZ, complex<float>, complex<float>>;
extern template struct block2::Linear<block2::SZ, complex<float>,
                                      complex<float>>;
extern template struct block2::Expect<block2::SZ, complex<float>,
                                      complex<float>, complex<float>>;

extern template struct block2::DMRG<block2::SU2, complex<float>,
                                    complex<float>>;
extern template struct block2::Linear<block2::SU2, complex<float>,
                                      complex<float>>;
extern template struct block2::Expect<block2::SU2, complex<float>,
                                      complex<float>, complex<float>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SZ, complex<float>,
                                      complex<float>>;
extern template struct block2::TimeEvolution<block2::SZ, complex<float>,
                                             complex<float>>;

extern template struct block2::TDDMRG<block2::SU2, complex<float>,
                                      complex<float>>;
extern template struct block2::TimeEvolution<block2::SU2, complex<float>,
                                             complex<float>>;

#ifdef _USE_SG

// archived_mpo.hpp
extern template struct block2::ArchivedMPO<block2::SGF, complex<float>>;
extern template struct block2::ArchivedMPO<block2::SGB, complex<float>>;

// determinant.hpp
extern template struct block2::DeterminantTRIE<block2::SGF, complex<float>>;
extern template struct block2::DeterminantTRIE<block2::SGB, complex<float>>;

// dmrg_driver.hpp
extern template struct block2::DMRGDriver<block2::SGF, complex<float>>;
extern template struct block2::DMRGDriver<block2::SGB, complex<float>>;

// effective_functions.hpp
extern template struct block2::EffectiveFunctions<block2::SGF, complex<float>>;
extern template struct block2::EffectiveFunctions<block2::SGB, complex<float>>;

// effective_hamiltonian.hpp
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, complex<float>, block2::MPS<block2::SGF, complex<float>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGF,
                                                          complex<float>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGF, complex<float>, block2::MultiMPS<block2::SGF, complex<float>>>;

extern template struct block2::EffectiveHamiltonian<
    block2::SGB, complex<float>, block2::MPS<block2::SGB, complex<float>>>;
extern template struct block2::LinearEffectiveHamiltonian<block2::SGB,
                                                          complex<float>>;
extern template struct block2::EffectiveHamiltonian<
    block2::SGB, complex<float>, block2::MultiMPS<block2::SGB, complex<float>>>;

// general_mpo.hpp
extern template struct block2::GeneralHamiltonian<block2::SGF, complex<float>>;
extern template struct block2::GeneralMPO<block2::SGF, complex<float>>;

extern template struct block2::GeneralHamiltonian<block2::SGB, complex<float>>;
extern template struct block2::GeneralMPO<block2::SGB, complex<float>>;

// general_npdm.hpp
extern template struct block2::GeneralNPDMMPO<block2::SGF, complex<float>>;
extern template struct block2::GeneralNPDMMPO<block2::SGB, complex<float>>;

// moving_environment.hpp
extern template struct block2::MovingEnvironment<block2::SGF, complex<float>,
                                                 complex<float>>;
extern template struct block2::MovingEnvironment<block2::SGB, complex<float>,
                                                 complex<float>>;
extern template struct block2::MovingEnvironment<block2::SGF, complex<float>,
                                                 float>;
extern template struct block2::MovingEnvironment<block2::SGB, complex<float>,
                                                 float>;

// mpo.hpp
extern template struct block2::MPO<block2::SGF, complex<float>>;
extern template struct block2::DiagonalMPO<block2::SGF, complex<float>>;
extern template struct block2::AncillaMPO<block2::SGF, complex<float>>;
extern template struct block2::IdentityAddedMPO<block2::SGF, complex<float>>;

extern template struct block2::MPO<block2::SGB, complex<float>>;
extern template struct block2::DiagonalMPO<block2::SGB, complex<float>>;
extern template struct block2::AncillaMPO<block2::SGB, complex<float>>;
extern template struct block2::IdentityAddedMPO<block2::SGB, complex<float>>;

// mpo_fusing.hpp
extern template struct block2::CondensedMPO<block2::SGF, complex<float>>;
extern template struct block2::FusedMPO<block2::SGF, complex<float>>;

extern template struct block2::CondensedMPO<block2::SGB, complex<float>>;
extern template struct block2::FusedMPO<block2::SGB, complex<float>>;

// mpo_simplification.hpp
extern template struct block2::SimplifiedMPO<block2::SGF, complex<float>>;
extern template struct block2::SimplifiedMPO<block2::SGB, complex<float>>;

// mps.hpp
extern template struct block2::MPS<block2::SGF, complex<float>>;
extern template struct block2::MPS<block2::SGB, complex<float>>;

extern template struct block2::TransMPS<block2::SGF, complex<double>,
                                        complex<float>>;
extern template struct block2::TransMPS<block2::SGF, complex<float>,
                                        complex<double>>;
extern template struct block2::TransMPS<block2::SGB, complex<double>,
                                        complex<float>>;
extern template struct block2::TransMPS<block2::SGB, complex<float>,
                                        complex<double>>;

// mps_unfused.hpp
extern template struct block2::SparseTensor<block2::SGF, complex<float>>;
extern template struct block2::UnfusedMPS<block2::SGF, complex<float>>;

extern template struct block2::SparseTensor<block2::SGB, complex<float>>;
extern template struct block2::UnfusedMPS<block2::SGB, complex<float>>;

// parallel_mpo.hpp
extern template struct block2::ClassicParallelMPO<block2::SGF, complex<float>>;
extern template struct block2::ParallelMPO<block2::SGF, complex<float>>;

extern template struct block2::ClassicParallelMPO<block2::SGB, complex<float>>;
extern template struct block2::ParallelMPO<block2::SGB, complex<float>>;

// parallel_mps.hpp
extern template struct block2::ParallelMPS<block2::SGF, complex<float>>;
extern template struct block2::ParallelMPS<block2::SGB, complex<float>>;

// parallel_simple.hpp
extern template struct block2::ParallelRuleSimple<block2::SGF, complex<float>>;
extern template struct block2::SumMPORule<block2::SGF, complex<float>>;
extern template struct block2::ParallelFCIDUMP<block2::SGF, complex<float>>;

extern template struct block2::ParallelRuleSimple<block2::SGB, complex<float>>;
extern template struct block2::SumMPORule<block2::SGB, complex<float>>;
extern template struct block2::ParallelFCIDUMP<block2::SGB, complex<float>>;

// partition.hpp
extern template struct block2::Partition<block2::SGF, complex<float>>;
extern template struct block2::Partition<block2::SGB, complex<float>>;

// qc_hamiltonian.hpp
extern template struct block2::HamiltonianQC<block2::SGF, complex<float>>;
extern template struct block2::HamiltonianQC<block2::SGB, complex<float>>;

// qc_mpo.hpp
extern template struct block2::IdentityMPO<block2::SGF, complex<float>>;
extern template struct block2::SiteMPO<block2::SGF, complex<float>>;
extern template struct block2::MPOQC<block2::SGF, complex<float>>;

extern template struct block2::IdentityMPO<block2::SGB, complex<float>>;
extern template struct block2::SiteMPO<block2::SGB, complex<float>>;
extern template struct block2::MPOQC<block2::SGB, complex<float>>;

// qc_ncorr.hpp
extern template struct block2::NPC1MPOQC<block2::SGF, complex<float>>;
extern template struct block2::NPC1MPOQC<block2::SGB, complex<float>>;

// qc_parallel_rule.hpp
extern template struct block2::ParallelRuleQC<block2::SGF, complex<float>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGF,
                                                     complex<float>>;
extern template struct block2::ParallelRulePDM1QC<block2::SGF, complex<float>>;
extern template struct block2::ParallelRulePDM2QC<block2::SGF, complex<float>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGF, complex<float>>;
extern template struct block2::ParallelRuleSiteQC<block2::SGF, complex<float>>;
extern template struct block2::ParallelRuleIdentity<block2::SGF,
                                                    complex<float>>;

extern template struct block2::ParallelRuleQC<block2::SGB, complex<float>>;
extern template struct block2::ParallelRuleOneBodyQC<block2::SGB,
                                                     complex<float>>;
extern template struct block2::ParallelRulePDM1QC<block2::SGB, complex<float>>;
extern template struct block2::ParallelRulePDM2QC<block2::SGB, complex<float>>;
extern template struct block2::ParallelRuleNPDMQC<block2::SGB, complex<float>>;
extern template struct block2::ParallelRuleSiteQC<block2::SGB, complex<float>>;
extern template struct block2::ParallelRuleIdentity<block2::SGB,
                                                    complex<float>>;

// qc_pdm1.hpp
extern template struct block2::PDM1MPOQC<block2::SGF, complex<float>>;
extern template struct block2::PDM1MPOQC<block2::SGB, complex<float>>;

// qc_pdm2.hpp
extern template struct block2::PDM2MPOQC<block2::SGF, complex<float>>;
extern template struct block2::PDM2MPOQC<block2::SGB, complex<float>>;

// qc_rule.hpp
extern template struct block2::RuleQC<block2::SGF, complex<float>>;
extern template struct block2::AntiHermitianRuleQC<block2::SGF, complex<float>>;

extern template struct block2::RuleQC<block2::SGB, complex<float>>;
extern template struct block2::AntiHermitianRuleQC<block2::SGB, complex<float>>;

// state_averaged.hpp
extern template struct block2::MultiMPS<block2::SGF, complex<float>>;
extern template struct block2::MultiMPS<block2::SGB, complex<float>>;

// sweep_algorithm.hpp
extern template struct block2::DMRG<block2::SGF, complex<float>,
                                    complex<float>>;
extern template struct block2::Linear<block2::SGF, complex<float>,
                                      complex<float>>;
extern template struct block2::Expect<block2::SGF, complex<float>,
                                      complex<float>, complex<float>>;

extern template struct block2::DMRG<block2::SGB, complex<float>,
                                    complex<float>>;
extern template struct block2::Linear<block2::SGB, complex<float>,
                                      complex<float>>;
extern template struct block2::Expect<block2::SGB, complex<float>,
                                      complex<float>, complex<float>>;

// sweep_algorithm_td.hpp
extern template struct block2::TDDMRG<block2::SGF, complex<float>,
                                      complex<float>>;
extern template struct block2::TimeEvolution<block2::SGF, complex<float>,
                                             complex<float>>;

extern template struct block2::TDDMRG<block2::SGB, complex<float>,
                                      complex<float>>;
extern template struct block2::TimeEvolution<block2::SGB, complex<float>,
                                             complex<float>>;

#endif

#endif

#endif
