
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

#ifdef _USE_COMPLEX
    py::module m_cpx = m.def_submodule("cpx", "Complex numbers.");
    py::module m_su2_cpx =
        m_cpx.def_submodule("su2", "Spin-adapted (complex).");
    py::module m_sz_cpx =
        m_cpx.def_submodule("sz", "Non-spin-adapted (complex).");
#endif

#ifdef _USE_SINGLE_PREC

    py::module m_sp = m.def_submodule("sp", "Single precision.");
    py::module m_su2_sp =
        m_sp.def_submodule("su2", "Spin-adapted (single precision).");
    py::module m_sz_sp =
        m_sp.def_submodule("sz", "Non-spin-adapted (single precision).");

#ifdef _USE_COMPLEX
    py::module m_cpx_sp =
        m_sp.def_submodule("cpx", "Complex single precision.");
    py::module m_su2_cpx_sp = m_cpx_sp.def_submodule(
        "su2", "Spin-adapted (complex single precision).");
    py::module m_sz_cpx_sp = m_cpx_sp.def_submodule(
        "sz", "Non-spin-adapted (complex single precision).");
#endif

#endif

#ifdef _USE_CORE
    bind_types<>(m);
    bind_io<>(m);
    bind_fl_io<double>(m, "Double");
    bind_matrix<double>(m);
    bind_symmetry<>(m);
    bind_fl_matrix<double>(m);
    bind_post_matrix<>(m);

    bind_core<SU2, double>(m_su2, "SU2", "Double");
    bind_core<SZ, double>(m_sz, "SZ", "Double");
    bind_trans_state_info<SU2, SZ>(m_su2, "sz");
    bind_trans_state_info<SZ, SU2>(m_sz, "su2");
    bind_trans_state_info_spin_specific<SU2, SZ>(m_su2, "sz");
#ifdef _USE_COMPLEX
    bind_fl_matrix<complex<double>>(m_cpx);
    bind_core<SU2, complex<double>>(m_su2_cpx, "SU2", "Double");
    bind_core<SZ, complex<double>>(m_sz_cpx, "SZ", "Double");
#endif

#ifdef _USE_KSYMM
    py::module m_su2k =
        m.def_submodule("su2k", "Spin-adapted with k symmetry.");
    py::module m_szk =
        m.def_submodule("szk", "Non-spin-adapted with k symmetry.");
    bind_core<SU2K, double>(m_su2k, "SU2K", "Double");
    bind_core<SZK, double>(m_szk, "SZK", "Double");
    bind_trans_state_info<SU2K, SZK>(m_su2k, "szk");
    bind_trans_state_info<SZK, SU2K>(m_szk, "su2k");
    bind_trans_state_info_spin_specific<SU2K, SZK>(m_su2k, "szk");
#ifdef _USE_COMPLEX
    py::module m_su2k_cpx =
        m_cpx.def_submodule("su2k", "Spin-adapted with k symmetry (complex).");
    py::module m_szk_cpx = m_cpx.def_submodule(
        "szk", "Non-spin-adapted with k symmetry (complex).");
    bind_core<SU2K, complex<double>>(m_su2k_cpx, "SU2K", "Double");
    bind_core<SZK, complex<double>>(m_szk_cpx, "SZK", "Double");
#endif
#endif

#ifdef _USE_SG
    py::module m_sgf = m.def_submodule("sgf", "General spin (fermionic).");
    py::module m_sgb = m.def_submodule("sgb", "General spin (bosonic).");
    bind_core<SGF, double>(m_sgf, "SGF", "Double");
    bind_core<SGB, double>(m_sgb, "SGB", "Double");
    bind_trans_state_info<SZ, SGF>(m_sz, "sgf");
    bind_trans_state_info<SGF, SZ>(m_sgf, "sz");
    bind_trans_state_info_spin_specific<SZ, SGF>(m_sz, "sgf");
#ifdef _USE_COMPLEX
    py::module m_sgf_cpx =
        m_cpx.def_submodule("sgf", "General spin (fermionic, complex).");
    py::module m_sgb_cpx =
        m_cpx.def_submodule("sgb", "General spin (bosonic, complex).");
    bind_core<SGF, complex<double>>(m_sgf_cpx, "SGF", "Double");
    bind_core<SGB, complex<double>>(m_sgb_cpx, "SGB", "Double");
#endif
#endif

#ifdef _USE_SINGLE_PREC

    bind_fl_io<float>(m, "Float");
    bind_matrix<float>(m_sp);
    bind_fl_matrix<float>(m_sp);

    bind_core<SU2, float>(m_su2_sp, "SU2", "Float");
    bind_core<SZ, float>(m_sz_sp, "SZ", "Float");

    bind_trans_sparse_matrix<SU2, float, double>(m_su2_sp, "double");
    bind_trans_sparse_matrix<SU2, double, float>(m_su2, "float");
    bind_trans_sparse_matrix<SZ, float, double>(m_sz_sp, "double");
    bind_trans_sparse_matrix<SZ, double, float>(m_sz, "float");
#ifdef _USE_COMPLEX
    bind_fl_matrix<complex<float>>(m_cpx_sp);
    bind_core<SU2, complex<float>>(m_su2_cpx_sp, "SU2", "Float");
    bind_core<SZ, complex<float>>(m_sz_cpx_sp, "SZ", "Float");

    bind_trans_sparse_matrix<SU2, complex<float>, complex<double>>(m_su2_cpx_sp,
                                                                   "double");
    bind_trans_sparse_matrix<SU2, complex<double>, complex<float>>(m_su2_cpx,
                                                                   "float");
    bind_trans_sparse_matrix<SZ, complex<float>, complex<double>>(m_sz_cpx_sp,
                                                                  "double");
    bind_trans_sparse_matrix<SZ, complex<double>, complex<float>>(m_sz_cpx,
                                                                  "float");
#endif

#ifdef _USE_SG
    py::module m_sgf_sp =
        m_sp.def_submodule("sgf", "General spin (fermionic single precision).");
    py::module m_sgb_sp =
        m_sp.def_submodule("sgb", "General spin (bosonic single precision).");
    bind_core<SGF, float>(m_sgf_sp, "SGF", "Float");
    bind_core<SGB, float>(m_sgb_sp, "SGB", "Float");

    bind_trans_sparse_matrix<SGF, float, double>(m_sgf_sp, "double");
    bind_trans_sparse_matrix<SGF, double, float>(m_sgf, "float");
    bind_trans_sparse_matrix<SGB, float, double>(m_sgb_sp, "double");
    bind_trans_sparse_matrix<SGB, double, float>(m_sgb, "float");
#ifdef _USE_COMPLEX
    py::module m_sgf_cpx_sp = m_cpx_sp.def_submodule(
        "sgf", "General spin (fermionic, complex single precision).");
    py::module m_sgb_cpx_sp = m_cpx_sp.def_submodule(
        "sgb", "General spin (bosonic, complex single precision).");
    bind_core<SGF, complex<float>>(m_sgf_cpx_sp, "SGF", "Float");
    bind_core<SGB, complex<float>>(m_sgb_cpx_sp, "SGB", "Float");

    bind_trans_sparse_matrix<SGF, complex<float>, complex<double>>(m_sgf_cpx_sp,
                                                                   "double");
    bind_trans_sparse_matrix<SGF, complex<double>, complex<float>>(m_sgf_cpx,
                                                                   "float");
    bind_trans_sparse_matrix<SGB, complex<float>, complex<double>>(m_sgb_cpx_sp,
                                                                   "double");
    bind_trans_sparse_matrix<SGB, complex<double>, complex<float>>(m_sgb_cpx,
                                                                   "float");
#endif
#endif

#endif

#endif

#ifdef _USE_DMRG
    bind_dmrg_types<>(m);
    bind_dmrg_io<>(m);
    bind_general_fcidump<double>(m);
    bind_dmrg<SU2, double>(m_su2, "SU2");
    bind_dmrg<SZ, double>(m_sz, "SZ");
    bind_trans_mps<SU2, SZ>(m_su2, "sz");
    bind_trans_mps<SZ, SU2>(m_sz, "su2");
    bind_fl_trans_mps_spin_specific<SU2, SZ, double>(m_su2, "sz");
#ifdef _USE_COMPLEX
    bind_general_fcidump<complex<double>>(m_cpx);
    bind_dmrg<SU2, complex<double>>(m_su2_cpx, "SU2");
    bind_dmrg<SZ, complex<double>>(m_sz_cpx, "SZ");
    bind_fl_trans_mps_spin_specific<SU2, SZ, complex<double>>(m_su2_cpx, "sz");
#endif

#ifdef _USE_KSYMM
    bind_dmrg<SU2K, double>(m_su2k, "SU2K");
    bind_dmrg<SZK, double>(m_szk, "SZK");
    bind_trans_mps<SU2K, SZK>(m_su2k, "szk");
    bind_trans_mps<SZK, SU2K>(m_szk, "su2k");
    bind_fl_trans_mps_spin_specific<SU2K, SZK, double>(m_su2k, "szk");
#ifdef _USE_COMPLEX
    bind_dmrg<SU2K, complex<double>>(m_su2k_cpx, "SU2K");
    bind_dmrg<SZK, complex<double>>(m_szk_cpx, "SZK");
    bind_fl_trans_mps_spin_specific<SU2K, SZK, complex<double>>(m_su2k_cpx,
                                                                "szk");
#endif
#endif

#ifdef _USE_SG
    bind_dmrg<SGF, double>(m_sgf, "SGF");
    bind_dmrg<SGB, double>(m_sgb, "SGB");
    bind_trans_mps<SZ, SGF>(m_sz, "sgf");
    bind_trans_mps<SGF, SZ>(m_sgf, "sz");
    bind_fl_trans_mps_spin_specific<SZ, SGF, double>(m_sz, "sgf");
#ifdef _USE_COMPLEX
    bind_dmrg<SGF, complex<double>>(m_sgf_cpx, "SGF");
    bind_dmrg<SGB, complex<double>>(m_sgb_cpx, "SGB");
    bind_fl_trans_mps_spin_specific<SZ, SGF, complex<double>>(m_sz_cpx, "sgf");
#endif
#endif

#ifdef _USE_SINGLE_PREC
    bind_general_fcidump<float>(m_sp);
    bind_dmrg<SU2, float>(m_su2_sp, "SU2");
    bind_dmrg<SZ, float>(m_sz_sp, "SZ");
    bind_fl_trans_mps_spin_specific<SU2, SZ, float>(m_su2_sp, "sz");

    bind_fl_trans_mps<SU2, float, double>(m_su2_sp, "double");
    bind_fl_trans_mps<SU2, double, float>(m_su2, "float");
    bind_fl_trans_mps<SZ, float, double>(m_sz_sp, "double");
    bind_fl_trans_mps<SZ, double, float>(m_sz, "float");
#ifdef _USE_COMPLEX
    bind_general_fcidump<complex<float>>(m_cpx_sp);
    bind_dmrg<SU2, complex<float>>(m_su2_cpx_sp, "SU2");
    bind_dmrg<SZ, complex<float>>(m_sz_cpx_sp, "SZ");
    bind_fl_trans_mps_spin_specific<SU2, SZ, complex<float>>(m_su2_cpx_sp,
                                                             "sz");

    bind_fl_trans_mps<SU2, complex<float>, complex<double>>(m_su2_cpx_sp,
                                                            "double");
    bind_fl_trans_mps<SU2, complex<double>, complex<float>>(m_su2_cpx, "float");
    bind_fl_trans_mps<SZ, complex<float>, complex<double>>(m_sz_cpx_sp,
                                                           "double");
    bind_fl_trans_mps<SZ, complex<double>, complex<float>>(m_sz_cpx, "float");
#endif

#ifdef _USE_SG
    bind_dmrg<SGF, float>(m_sgf_sp, "SGF");
    bind_dmrg<SGB, float>(m_sgb_sp, "SGB");
    bind_fl_trans_mps_spin_specific<SZ, SGF, float>(m_sz_sp, "sgf");

    bind_fl_trans_mps<SGF, float, double>(m_sgf_sp, "double");
    bind_fl_trans_mps<SGF, double, float>(m_sgf, "float");
    bind_fl_trans_mps<SGB, float, double>(m_sgb_sp, "double");
    bind_fl_trans_mps<SGB, double, float>(m_sgb, "float");
#ifdef _USE_COMPLEX
    bind_dmrg<SGF, complex<float>>(m_sgf_cpx_sp, "SGF");
    bind_dmrg<SGB, complex<float>>(m_sgb_cpx_sp, "SGB");
    bind_fl_trans_mps_spin_specific<SZ, SGF, complex<float>>(m_sz_cpx_sp,
                                                             "sgf");

    bind_fl_trans_mps<SGF, complex<float>, complex<double>>(m_sgf_cpx_sp,
                                                            "double");
    bind_fl_trans_mps<SGF, complex<double>, complex<float>>(m_sgf_cpx, "float");
    bind_fl_trans_mps<SGB, complex<float>, complex<double>>(m_sgb_cpx_sp,
                                                            "double");
    bind_fl_trans_mps<SGB, complex<double>, complex<float>>(m_sgb_cpx, "float");
#endif
#endif

#endif

#endif

#ifdef _USE_BIG_SITE
    bind_fl_big_site<SU2, double>(m_su2);
    bind_fl_hamiltonian_big_site<SU2, double>(m_su2);
    bind_fl_dmrg_big_site<SU2, double, double>(m_su2);
    bind_fl_big_site<SZ, double>(m_sz);
    bind_fl_hamiltonian_big_site<SZ, double>(m_sz);
    bind_fl_dmrg_big_site<SZ, double, double>(m_sz);

    bind_fl_sci_big_site_fock<SZ, double>(m_sz);

    bind_fl_csf_big_site<SU2, double>(m_su2);
#endif

#ifdef _USE_SP_DMRG
    bind_fl_sp_dmrg<SU2, double>(m_su2);
    bind_fl_sp_dmrg<SZ, double>(m_sz);
#ifdef _USE_KSYMM
    bind_fl_sp_dmrg<SU2K, double>(m_su2k);
    bind_fl_sp_dmrg<SZK, double>(m_szk);
#endif
#endif

#ifdef _USE_IC
    bind_wick<>(m);
    bind_nd_array<>(m);
    bind_guga<>(m);
    bind_guga<SU2>(m_su2);
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
