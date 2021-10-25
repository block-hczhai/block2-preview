
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

#ifdef _USE_IC
#include "pybind/pybind_ic.hpp"
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

    py::module m_su2 = m.def_submodule("su2", "Spin-adapted.");
    py::module m_sz = m.def_submodule("sz", "Non-spin-adapted.");
    bind_data<>(m);

#ifdef _USE_CORE
    bind_types<>(m);
    bind_io<>(m);
    bind_matrix<>(m);
    bind_symmetry<>(m);

    bind_core<SU2>(m_su2, "SU2");
    bind_core<SZ>(m_sz, "SZ");
    bind_trans_state_info<SU2, SZ>(m_su2, "sz");
    bind_trans_state_info<SZ, SU2>(m_sz, "su2");
    bind_trans_state_info_spin_specific<SU2, SZ>(m_su2, "sz");
#ifdef _USE_KSYMM
    py::module m_su2k =
        m.def_submodule("su2k", "Spin-adapted with k symmetry.");
    py::module m_szk =
        m.def_submodule("szk", "Non-spin-adapted with k symmetry.");
    bind_core<SU2K>(m_su2k, "SU2K");
    bind_core<SZK>(m_szk, "SZK");
    bind_trans_state_info<SU2K, SZK>(m_su2k, "szk");
    bind_trans_state_info<SZK, SU2K>(m_szk, "su2k");
    bind_trans_state_info_spin_specific<SU2K, SZK>(m_su2k, "szk");
#endif
#endif

#ifdef _USE_DMRG
    bind_dmrg_types<>(m);
    bind_dmrg_io<>(m);
    bind_dmrg<SU2>(m_su2, "SU2");
    bind_dmrg<SZ>(m_sz, "SZ");
    bind_trans_mps<SU2, SZ>(m_su2, "sz");
    bind_trans_mps<SZ, SU2>(m_sz, "su2");
    bind_trans_mps_spin_specific<SU2, SZ>(m_su2, "sz");
#ifdef _USE_KSYMM
    bind_dmrg<SU2K>(m_su2k, "SU2K");
    bind_dmrg<SZK>(m_szk, "SZK");
    bind_trans_mps<SU2K, SZK>(m_su2k, "szk");
    bind_trans_mps<SZK, SU2K>(m_szk, "su2k");
    bind_trans_mps_spin_specific<SU2K, SZK>(m_su2k, "szk");
#endif
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
    bind_sp_dmrg<SU2>(m_su2);
    bind_sp_dmrg<SZ>(m_sz);
#ifdef _USE_KSYMM
    bind_sp_dmrg<SU2K>(m_su2k);
    bind_sp_dmrg<SZK>(m_szk);
#endif
#endif

#ifdef _USE_IC
    bind_wick<>(m);
    bind_nd_array<>(m);
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
