
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

#include "block2_core.hpp"

#ifdef _USE_DMRG
#include "block2_dmrg.hpp"
#endif

#ifdef _USE_BIG_SITE
#include "block2_big_site.hpp"
#endif

#ifdef _USE_SP_DMRG
#include "block2_sp_dmrg.hpp"
#endif

#ifdef _USE_IC
#include "block2_ic.hpp"
#endif

#ifdef _USE_SCI
#include "sci/abstract_sci_wrapper.hpp"
#include "sci/hamiltonian_sci.hpp"
#include "sci/qc_hamiltonian_sci.hpp"
#include "sci/qc_mpo_sci.hpp"
#include "sci/sweep_algorithm_sci.hpp"
#endif

#undef ialloc
#undef dalloc
#undef threading
