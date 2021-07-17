
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "../block2_core.hpp"
#include "../block2_dmrg.hpp"
#include "../block2_mrci.hpp"

namespace py = pybind11;
using namespace block2;

template <typename S> void bind_big_site_qc(py::module &m) {

    py::class_<BigSiteQC<S>, shared_ptr<BigSiteQC<S>>>(m, "BigSiteQC")
        .def(py::init<>())
        .def_readonly("qs", &BigSiteQC<S>::qs)
        .def_readonly("offsets", &BigSiteQC<S>::offsets)
        .def_readonly("n_orbs_other", &BigSiteQC<S>::n_orbs_other)
        .def_readonly("n_orbs_this", &BigSiteQC<S>::n_orbs_this)
        .def_readonly("n_orbs", &BigSiteQC<S>::n_orbs)
        .def_readonly("is_right", &BigSiteQC<S>::is_right)
        .def_readonly("n_alpha", &BigSiteQC<S>::n_alpha)
        .def_readonly("n_beta", &BigSiteQC<S>::n_beta)
        .def_readonly("n_elec", &BigSiteQC<S>::n_elec)
        .def_readonly("n_det", &BigSiteQC<S>::n_det)
        .def_readwrite("sparsity_thresh", &BigSiteQC<S>::sparsity_thresh,
                       "After > #zeros/#tot the sparse matrix is activated")
        .def_readwrite("sparsity_start", &BigSiteQC<S>::sparsity_start,
                       "After which matrix size (nCol * nRow) should sparse "
                       "matrices be activated")
        .def_readwrite("eps", &BigSiteQC<S>::eps,
                       "Sparsity value threshold. Everything below eps will be "
                       "set to 0.0");
};

template <typename S> void bind_hamiltonian_big_site(py::module &m) {

    py::class_<HamiltonianQCBigSite<S>, shared_ptr<HamiltonianQCBigSite<S>>,
               HamiltonianQC<S>>(m, "HamiltonianQCBigSite")
        .def(py::init<S, int, const vector<uint8_t> &,
                      const shared_ptr<FCIDUMP> &,
                      const shared_ptr<BigSiteQC<S>> &,
                      const shared_ptr<BigSiteQC<S>> &>(),
             py::arg("vacuum"), py::arg("n_orbs_total"), py::arg("orb_sym"),
             py::arg("fcidump"), py::arg("big_left") = nullptr,
             py::arg("big_right") = nullptr)
        .def_readonly("n_orbs_left", &HamiltonianQCBigSite<S>::n_orbs_left)
        .def_readonly("n_orbs_right", &HamiltonianQCBigSite<S>::n_orbs_right)
        .def_readonly("n_orbs_cas", &HamiltonianQCBigSite<S>::n_orbs_cas)
        .def_readonly("full_hamil", &HamiltonianQCBigSite<S>::full_hamil)
        .def_readwrite("rule", &HamiltonianQCBigSite<S>::rule)
        .def_readwrite("parallel_rule",
                       &HamiltonianQCBigSite<S>::parallel_rule);
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
