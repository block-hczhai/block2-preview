
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

#include "pybind/pybind_core.hpp"

#ifdef _USE_DMRG
#include "pybind/pybind_dmrg.hpp"
#endif

#ifdef _USE_BIG_SITE
#include "pybind/pybind_big_site.hpp"
#endif

#ifdef _USE_SP_DMRG
#include "pybind/pybind_sp_dmrg.hpp"
#endif

#ifdef _USE_SCI
#include "sci/pybind.hpp"
#ifdef _SCI_WRAPPER2
#include "pybind_sci.hpp"
#endif
#endif

PYBIND11_MODULE(block2, m) {

    m.doc() = "python interface for block2.";

    // Handle Ctrl-C from python side
    check_signal_() = []() {
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    };

    bind_data<>(m);
    bind_types<>(m);
    bind_io<>(m);
    bind_matrix<>(m);
    bind_symmetry<>(m);

    py::module m_su2 = m.def_submodule("su2", "Spin-adapted.");
    py::module m_sz = m.def_submodule("sz", "Non-spin-adapted.");
    bind_core<SU2>(m_su2, "SU2");
    bind_core<SZ>(m_sz, "SZ");
    bind_trans_state_info<SU2, SZ>(m_su2, "sz");
    bind_trans_state_info<SZ, SU2>(m_sz, "su2");
    bind_trans_state_info_spin_specific<SU2, SZ>(m_su2, "sz");

#ifdef _USE_DMRG
    bind_dmrg_types<>(m);
    bind_dmrg_io<>(m);
    bind_dmrg<SU2>(m_su2, "SU2");
    bind_dmrg<SZ>(m_sz, "SZ");
    bind_trans_mps<SU2, SZ>(m_su2, "sz");
    bind_trans_mps<SZ, SU2>(m_sz, "su2");
    bind_trans_mps_spin_specific<SU2, SZ>(m_su2, "sz");
#endif

#ifdef _USE_BIG_SITE
    bind_big_site<SU2>(m_su2);
    bind_hamiltonian_big_site<SU2>(m_su2);
    bind_dmrg_big_site<SU2>(m_su2);
    bind_big_site<SZ>(m_sz);
    bind_hamiltonian_big_site<SZ>(m_sz);
    bind_dmrg_big_site<SZ>(m_sz);

    bind_sci_big_site_fock<SZ>(m_sz);

    bind_csf_big_site<SU2>(m_su2);
#endif

#ifdef _USE_SP_DMRG
    bind_sp_dmrg<SZ>(m_sz);
#endif

#ifdef _USE_SCI
    bind_sci_wrapper<SZ>(m_sz);
#ifdef _SCI_WRAPPER2
    bind_sci_wrapper2<SZ>(m_sz);
#endif
    bind_hamiltonian_sci<SZ>(m_sz);
    bind_mpo_sci<SZ>(m_sz);
    bind_types_sci<>(m);
#endif

}
