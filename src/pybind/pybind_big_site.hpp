
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "../block2_big_site.hpp"
#include "../block2_core.hpp"
#include "../block2_dmrg.hpp"

namespace py = pybind11;
using namespace block2;

template <typename S> void bind_big_site(py::module &m) {

    py::class_<BigSite<S>, shared_ptr<BigSite<S>>>(m, "BigSite")
        .def(py::init<int>())
        .def_readwrite("n_orbs", &BigSite<S>::n_orbs)
        .def_readwrite("basis", &BigSite<S>::basis)
        .def_readwrite("op_infos", &BigSite<S>::op_infos)
        .def("get_site_ops", &BigSite<S>::get_site_ops);

    py::class_<SimplifiedBigSite<S>, shared_ptr<SimplifiedBigSite<S>>,
               BigSite<S>>(m, "SimplifiedBigSite")
        .def(py::init<const shared_ptr<BigSite<S>> &,
                      const shared_ptr<Rule<S>> &>())
        .def_readwrite("big_site", &SimplifiedBigSite<S>::big_site)
        .def_readwrite("rule", &SimplifiedBigSite<S>::rule);

    py::class_<ParallelBigSite<S>, shared_ptr<ParallelBigSite<S>>, BigSite<S>>(
        m, "ParallelBigSite")
        .def(py::init<const shared_ptr<BigSite<S>> &,
                      const shared_ptr<ParallelRule<S>> &>())
        .def_readwrite("big_site", &ParallelBigSite<S>::big_site)
        .def_readwrite("rule", &ParallelBigSite<S>::rule);
}

template <typename S> void bind_sci_big_site_fock(py::module &m) {

    py::class_<SCIFockBigSite<S>, shared_ptr<SCIFockBigSite<S>>, BigSite<S>>(
        m, "SCIFockBigSite")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP> &,
                      const std::vector<uint8_t> &, int, int, int, bool>(),
             py::arg("nOrb"), py::arg("nOrbThis"), py::arg("isRight"),
             py::arg("fcidump"), py::arg("orbsym"), py::arg("nMaxAlphaEl"),
             py::arg("nMaxBetaEl"), py::arg("nMaxEl"),
             py::arg("verbose") = true,
             "Initialization via generated CI space based on nMax*")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP> &,
                      const std::vector<uint8_t> &, const vector<vector<int>> &,
                      bool>(),
             py::arg("nOrb"), py::arg("nOrbThis"), py::arg("isRight"),
             py::arg("fcidump"), py::arg("orbsym"), py::arg("occs"),
             py::arg("verbose") = true,
             "Initialization via externally given determinants in `occs`")
        .def_readwrite("excludeQNs", &SCIFockBigSite<S>::excludeQNs)
        .def_static("ras_space", &SCIFockBigSite<S>::ras_space)
        .def_readonly("quantumNumbers", &SCIFockBigSite<S>::quantumNumbers)
        .def_readonly("nOrbOther", &SCIFockBigSite<S>::nOrbOther)
        .def_readonly("nOrbThis", &SCIFockBigSite<S>::nOrbThis)
        .def_readonly("nOrb", &SCIFockBigSite<S>::nOrb)
        .def_readonly("isRight", &SCIFockBigSite<S>::isRight)
        .def_readonly("nMaxAlphaEl", &SCIFockBigSite<S>::nMaxAlphaEl)
        .def_readonly("nMaxBetaEl", &SCIFockBigSite<S>::nMaxBetaEl)
        .def_readonly("nMaxEl", &SCIFockBigSite<S>::nMaxEl)
        .def_readonly("nDet", &SCIFockBigSite<S>::nDet)
        .def_readwrite("sparsityThresh", &SCIFockBigSite<S>::sparsityThresh,
                       "After > #zeros/#tot the sparse matrix is activated")
        .def_readwrite("sparsityStart", &SCIFockBigSite<S>::sparsityStart,
                       "After which matrix size (nCol * nRow) should sparse "
                       "matrices be activated")
        .def_readwrite("eps", &SCIFockBigSite<S>::eps,
                       "Sparsity value threshold. Everything below eps will be "
                       "set to 0.0")
        .def("setOmpThreads", &SCIFockBigSite<S>::setOmpThreads)
        // vv setter
        .def_property("qnIdxBra", nullptr,
                      (void (SCIFockBigSite<S>::*)(const std::vector<int> &)) &
                          SCIFockBigSite<S>::setQnIdxBra)
        .def_property("qnIdxKet", nullptr,
                      (void (SCIFockBigSite<S>::*)(const std::vector<int> &)) &
                          SCIFockBigSite<S>::setQnIdxKet)
        .def("setQnIdxBra",
             (void (SCIFockBigSite<S>::*)(const std::vector<int> &,
                                          const std::vector<char> &)) &
                 SCIFockBigSite<S>::setQnIdxBra)
        .def("setQnIdxKet",
             (void (SCIFockBigSite<S>::*)(const std::vector<int> &,
                                          const std::vector<char> &)) &
                 SCIFockBigSite<S>::setQnIdxKet)
        .def_readwrite("qnIdxBraH", &SCIFockBigSite<S>::qnIdxBraH)
        .def_readwrite("qnIdxKetH", &SCIFockBigSite<S>::qnIdxKetH)
        .def_readwrite("qnIdxBraQ", &SCIFockBigSite<S>::qnIdxBraQ)
        .def_readwrite("qnIdxKetQ", &SCIFockBigSite<S>::qnIdxKetQ)
        .def_readwrite("qnIdxBraI", &SCIFockBigSite<S>::qnIdxBraI)
        .def_readwrite("qnIdxKetI", &SCIFockBigSite<S>::qnIdxKetI)
        .def_readwrite("qnIdxBraA", &SCIFockBigSite<S>::qnIdxBraA)
        .def_readwrite("qnIdxKetA", &SCIFockBigSite<S>::qnIdxKetA)
        .def_readwrite("qnIdxBraB", &SCIFockBigSite<S>::qnIdxBraB)
        .def_readwrite("qnIdxKetB", &SCIFockBigSite<S>::qnIdxKetB)
        .def_readwrite("qnIdxBraP", &SCIFockBigSite<S>::qnIdxBraP)
        .def_readwrite("qnIdxKetP", &SCIFockBigSite<S>::qnIdxKetP)
        .def_readwrite("qnIdxBraR", &SCIFockBigSite<S>::qnIdxBraR)
        .def_readwrite("qnIdxKetR", &SCIFockBigSite<S>::qnIdxKetR)
        .def_readwrite("qnIdxBraC", &SCIFockBigSite<S>::qnIdxBraC)
        .def_readwrite("qnIdxKetC", &SCIFockBigSite<S>::qnIdxKetC);
}

template <typename S> void bind_csf_big_site(py::module &m) {

    py::class_<CSFSpace<S>, shared_ptr<CSFSpace<S>>>(m, "CSFSpace")
        .def(py::init<int, int, bool, const std::vector<uint8_t> &>())
        .def("get_config", &CSFSpace<S>::get_config)
        .def("index_config", &CSFSpace<S>::index_config)
        .def("to_string", &CSFSpace<S>::to_string)
        .def("cfg_apply_ops", &CSFSpace<S>::cfg_apply_ops)
        .def("csf_apply_ops", &CSFSpace<S>::csf_apply_ops)
        .def("__getitem__", &CSFSpace<S>::operator[], py::arg("idx"))
        .def_readwrite("qs", &CSFSpace<S>::qs)
        .def_readwrite("qs_idxs", &CSFSpace<S>::qs_idxs)
        .def_readwrite("n_unpaired", &CSFSpace<S>::n_unpaired)
        .def_readwrite("n_unpaired_shapes", &CSFSpace<S>::n_unpaired_shapes)
        .def_readwrite("csfs", &CSFSpace<S>::csfs)
        .def_readwrite("csf_idxs", &CSFSpace<S>::csf_idxs)
        .def_readwrite("csf_sub_idxs", &CSFSpace<S>::csf_sub_idxs)
        .def_readwrite("csf_offsets", &CSFSpace<S>::csf_offsets)
        .def_readwrite("combinatorics", &CSFSpace<S>::combinatorics)
        .def_readwrite("basis", &CSFSpace<S>::basis)
        .def_readwrite("cg", &CSFSpace<S>::cg)
        .def_readwrite("n_orbs", &CSFSpace<S>::n_orbs)
        .def_readwrite("n_max_elec", &CSFSpace<S>::n_max_elec)
        .def_readwrite("n_max_unpaired", &CSFSpace<S>::n_max_unpaired)
        .def_readwrite("is_right", &CSFSpace<S>::is_right);

    py::class_<CSFBigSite<S>, shared_ptr<CSFBigSite<S>>, BigSite<S>>(
        m, "CSFBigSite")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP> &,
                      const std::vector<uint8_t> &>())
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP> &,
                      const std::vector<uint8_t> &, int>())
        .def_static("fill_csr_matrix", &CSFBigSite<S>::fill_csr_matrix)
        .def("build_site_op", &CSFBigSite<S>::build_site_op)
        .def_readwrite("fcidump", &CSFBigSite<S>::fcidump)
        .def_readwrite("csf_space", &CSFBigSite<S>::csf_space)
        .def_readwrite("is_right", &CSFBigSite<S>::is_right);
}

template <typename S> void bind_hamiltonian_big_site(py::module &m) {

    py::class_<HamiltonianQCBigSite<S>, shared_ptr<HamiltonianQCBigSite<S>>,
               HamiltonianQC<S>>(m, "HamiltonianQCBigSite")
        .def(py::init<S, int, const vector<uint8_t> &,
                      const shared_ptr<FCIDUMP> &,
                      const shared_ptr<BigSite<S>> &,
                      const shared_ptr<BigSite<S>> &>(),
             py::arg("vacuum"), py::arg("n_orbs_total"), py::arg("orb_sym"),
             py::arg("fcidump"), py::arg("big_left") = nullptr,
             py::arg("big_right") = nullptr)
        .def_readwrite("big_left", &HamiltonianQCBigSite<S>::big_left)
        .def_readwrite("big_right", &HamiltonianQCBigSite<S>::big_right)
        .def_readwrite("n_orbs", &HamiltonianQCBigSite<S>::n_orbs)
        .def_readwrite("n_orbs_left", &HamiltonianQCBigSite<S>::n_orbs_left)
        .def_readwrite("n_orbs_right", &HamiltonianQCBigSite<S>::n_orbs_right)
        .def_readwrite("n_orbs_cas", &HamiltonianQCBigSite<S>::n_orbs_cas)
        .def_readwrite("full_hamil", &HamiltonianQCBigSite<S>::full_hamil);
}

template <typename S> void bind_dmrg_big_site(py::module &m) {

    py::class_<DMRGBigSite<S>, shared_ptr<DMRGBigSite<S>>, DMRG<S>>(
        m, "DMRGBigSite")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &>())
        .def_readwrite("last_site_svd", &DMRGBigSite<S>::last_site_svd)
        .def_readwrite("last_site_1site", &DMRGBigSite<S>::last_site_1site)
        .def("blocking", &DMRGBigSite<S>::blocking);

    py::class_<LinearBigSite<S>, shared_ptr<LinearBigSite<S>>, Linear<S>>(
        m, "LinearBigSite")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<double> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<double> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<double> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def_readwrite("last_site_svd", &LinearBigSite<S>::last_site_svd)
        .def_readwrite("last_site_1site", &LinearBigSite<S>::last_site_1site)
        .def("blocking", &LinearBigSite<S>::blocking);

    py::class_<DMRGBigSiteAQCCOLD<S>, shared_ptr<DMRGBigSiteAQCCOLD<S>>,
               DMRGBigSite<S>>(m, "DMRGBigSiteAQCCOLD")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &, double,
                      double, const std::vector<S> &>())
        .def_readwrite("max_aqcc_iter", &DMRGBigSiteAQCCOLD<S>::max_aqcc_iter)
        .def_readwrite("g_factor", &DMRGBigSiteAQCCOLD<S>::g_factor)
        .def_readwrite("delta_e", &DMRGBigSiteAQCCOLD<S>::delta_e)
        .def_readwrite("ref_energy", &DMRGBigSiteAQCCOLD<S>::ref_energy);

    py::class_<DMRGBigSiteAQCC<S>, shared_ptr<DMRGBigSiteAQCC<S>>,
               DMRGBigSite<S>>(m, "DMRGBigSiteAQCC")
        .def(
            py::init<const shared_ptr<MovingEnvironment<S>> &, double,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const vector<ubond_t> &, const vector<double> &, double>(),
            "Frozen/CAS mode: Only one big site at the end")
        .def(
            py::init<const shared_ptr<MovingEnvironment<S>> &, double,
                     const shared_ptr<MovingEnvironment<S>> &, double,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const vector<ubond_t> &, const vector<double> &, double>(),
            "Frozen/CAS mode ACPF2: Only one big site at the end")
        .def(
            py::init<const shared_ptr<MovingEnvironment<S>> &, double,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const vector<ubond_t> &, const vector<double> &, double>(),
            "RAS mode: Big sites on both ends")
        .def(
            py::init<const shared_ptr<MovingEnvironment<S>> &, double,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const shared_ptr<MovingEnvironment<S>> &, double,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const shared_ptr<MovingEnvironment<S>> &,
                     const vector<ubond_t> &, const vector<double> &, double>(),
            "RAS ACPF2 mode: Big sites on both ends")
        .def_readwrite("smallest_energy", &DMRGBigSiteAQCC<S>::smallest_energy)
        .def_readwrite("max_aqcc_iter", &DMRGBigSiteAQCC<S>::max_aqcc_iter)
        .def_readwrite("g_factor", &DMRGBigSiteAQCC<S>::g_factor)
        .def_readwrite("g_factor2", &DMRGBigSiteAQCC<S>::g_factor2)
        .def_readwrite("ACPF2_mode", &DMRGBigSiteAQCC<S>::ACPF2_mode)
        .def_readwrite("RAS_mode", &DMRGBigSiteAQCC<S>::RAS_mode)
        .def_readwrite("delta_e", &DMRGBigSiteAQCC<S>::delta_e)
        .def_readwrite("ref_energy", &DMRGBigSiteAQCC<S>::ref_energy);
}
