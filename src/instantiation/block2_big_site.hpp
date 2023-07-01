
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2023 Huanchen Zhai <hczhai@caltech.edu>
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

#include "../big_site/big_site.hpp"
#include "../big_site/csf_big_site.hpp"
#include "../big_site/drt_big_site.hpp"
#include "../big_site/drt_mps.hpp"
#include "../big_site/qc_hamiltonian_big_site.hpp"
#include "../big_site/sci_fock_big_site.hpp"
#include "../big_site/sweep_algorithm_big_site.hpp"
#include "../core/symmetry.hpp"
#include "block2_core.hpp"
#include "block2_dmrg.hpp"

#ifdef _USE_SU2SZ

// big_site.hpp
extern template struct block2::BigSite<block2::SZ, double>;
extern template struct block2::SimplifiedBigSite<block2::SZ, double>;
extern template struct block2::ParallelBigSite<block2::SZ, double>;

extern template struct block2::BigSite<block2::SU2, double>;
extern template struct block2::SimplifiedBigSite<block2::SU2, double>;
extern template struct block2::ParallelBigSite<block2::SU2, double>;

// csf_big_site.hpp
extern template struct block2::CSFSpace<block2::SU2, double>;
extern template struct block2::CSFBigSite<block2::SU2, double>;

// drt_big_site.hpp
extern template struct block2::ElemMat<block2::SZ, double>;
extern template struct block2::DRT<block2::SZ, block2::ElemOpTypes::SZ>;
extern template struct block2::HDRT<block2::SZ, block2::ElemOpTypes::SZ>;

extern template struct block2::HDRTScheme<block2::SZ, double,
                                          block2::ElemOpTypes::SZ>;
extern template struct block2::DRTBigSiteBase<block2::SZ, double>;
extern template struct block2::DRTBigSite<block2::SZ, double>;

extern template struct block2::ElemMat<block2::SU2, double>;
extern template struct block2::DRT<block2::SU2, block2::ElemOpTypes::SU2>;
extern template struct block2::HDRT<block2::SU2, block2::ElemOpTypes::SU2>;

extern template struct block2::HDRTScheme<block2::SU2, double,
                                          block2::ElemOpTypes::SU2>;
extern template struct block2::DRTBigSiteBase<block2::SU2, double>;
extern template struct block2::DRTBigSite<block2::SU2, double>;

// drt_mps.hpp
extern template struct block2::DRTMPS<block2::SZ, double,
                                      block2::ElemOpTypes::SZ>;
extern template struct block2::HDRTMPO<block2::SZ, double,
                                       block2::ElemOpTypes::SZ>;

extern template struct block2::DRTMPS<block2::SU2, double,
                                      block2::ElemOpTypes::SU2>;
extern template struct block2::HDRTMPO<block2::SU2, double,
                                       block2::ElemOpTypes::SU2>;

// qc_hamiltonian_big_site.hpp
extern template struct block2::HamiltonianQCBigSite<block2::SZ, double>;
extern template struct block2::HamiltonianQCBigSite<block2::SU2, double>;

// sci_fock_big_site.hpp
extern template struct block2::SCIFockBigSite<block2::SZ, double>;

// sweep_algorithm_big_site.hpp
extern template struct block2::DMRGBigSite<block2::SZ, double, double>;
extern template struct block2::DMRGBigSiteAQCC<block2::SZ, double, double>;
extern template struct block2::DMRGBigSiteAQCCOLD<block2::SZ, double, double>;
extern template struct block2::LinearBigSite<block2::SZ, double, double>;

extern template struct block2::DMRGBigSite<block2::SU2, double, double>;
extern template struct block2::DMRGBigSiteAQCC<block2::SU2, double, double>;
extern template struct block2::DMRGBigSiteAQCCOLD<block2::SU2, double, double>;
extern template struct block2::LinearBigSite<block2::SU2, double, double>;

#endif
