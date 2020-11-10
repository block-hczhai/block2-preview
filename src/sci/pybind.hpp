
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

#include "../block2.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace block2;

template <typename S> void bind_sci_wrapper(py::module &m) {
    py::class_<sci::AbstractSciWrapper<S>,
               shared_ptr<sci::AbstractSciWrapper<S>>>(m, "AbstractSciWrapper")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP> &,
                      const std::vector<uint8_t> &, int, int, int>(),
             py::arg("nOrb"), py::arg("nOrbThis"), py::arg("isRight"),
             py::arg("fcidump"), py::arg("orbsym"), py::arg("nMaxAlphaEl"),
             py::arg("nMaxBetaEl"), py::arg("nMaxEl"),
             "Initialization via generated CI space based on nMax*")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP> &,
                      const std::vector<uint8_t> &,
                      const vector<vector<int>> &>(),
             py::arg("nOrb"), py::arg("nOrbThis"), py::arg("isRight"),
             py::arg("fcidump"), py::arg("orbsym"), py::arg("occs"),
             "Initialization via externally given determinants in `occs`")
        .def_readonly("quantumNumbers",
                      &sci::AbstractSciWrapper<S>::quantumNumbers)
        .def_readonly("nOrbOther", &sci::AbstractSciWrapper<S>::nOrbOther)
        .def_readonly("nOrbThis", &sci::AbstractSciWrapper<S>::nOrbThis)
        .def_readonly("nOrb", &sci::AbstractSciWrapper<S>::nOrb)
        .def_readonly("isRight", &sci::AbstractSciWrapper<S>::isRight)
        .def_readonly("nMaxAlphaEl", &sci::AbstractSciWrapper<S>::nMaxAlphaEl)
        .def_readonly("nMaxBetaEl", &sci::AbstractSciWrapper<S>::nMaxBetaEl)
        .def_readonly("nMaxEl", &sci::AbstractSciWrapper<S>::nMaxEl)
        .def_readonly("nDet", &sci::AbstractSciWrapper<S>::nDet)
        .def_readwrite("sparsityThresh",
                       &sci::AbstractSciWrapper<S>::sparsityThresh,
                       "After > #zeros/#tot the sparse matrix is activated")
        .def_readwrite("sparsityStart",
                       &sci::AbstractSciWrapper<S>::sparsityStart,
                       "After which matrix size (nCol * nRow) should sparse "
                       "matrices be activated")
        .def_readwrite("eps", &sci::AbstractSciWrapper<S>::eps,
                       "Sparsity value threshold. Everything below eps will be "
                       "set to 0.0");
};

template <typename S> void bind_hamiltonian_sci(py::module &m) {

    py::class_<HamiltonianSCI<S>, shared_ptr<HamiltonianSCI<S>>>(
        m, "HamiltonianSCI")
        .def(py::init<S, int, const vector<uint8_t> &>())
        .def_readwrite("n_syms", &HamiltonianSCI<S>::n_syms)
        .def_readwrite("opf", &HamiltonianSCI<S>::opf)
        .def_readwrite("n_sites", &HamiltonianSCI<S>::n_sites)
        .def_readwrite("orb_sym", &HamiltonianSCI<S>::orb_sym)
        .def_readwrite("vacuum", &HamiltonianSCI<S>::vacuum)
        .def_readwrite("basis", &HamiltonianSCI<S>::basis)
        .def_readwrite("delayed", &HamiltonianSCI<S>::delayed)
        .def("deallocate", &HamiltonianSCI<S>::deallocate);
    // vv hrl: switched off as not really required (now protected)
    /*.def_property_readonly(
            "site_op_infos",
            [](HamiltonianSCI<S> *self) {
                return Array<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
                        self->site_op_infos, self->n_syms);
            })
    .def("get_site_ops", &HamiltonianSCI<S>::get_site_ops)
    .def("filter_site_ops", &HamiltonianSCI<S>::filter_site_ops)
    .def("find_site_op_info", &HamiltonianSCI<S>::find_site_op_info)
    .def("find_site_norm_op", &HamiltonianSCI<S>::find_site_norm_op)
    */

    py::class_<HamiltonianQCSCI<S>, shared_ptr<HamiltonianQCSCI<S>>,
               HamiltonianSCI<S>>(m, "HamiltonianQCSCI")
        .def(py::init<S, int, const vector<uint8_t> &,
                      const shared_ptr<FCIDUMP> &,
                      const std::shared_ptr<sci::AbstractSciWrapper<S>> &,
                      const std::shared_ptr<sci::AbstractSciWrapper<S>> &>(),
             py::arg("vacuum"), py::arg("nOrbTot"), py::arg("orb_Sym"),
             py::arg("fcidump"), py::arg("sciWrapperLeft") = nullptr,
             py::arg("sciWraperRight") = nullptr)
        .def_readonly("nOrbLeft", &HamiltonianQCSCI<S>::nOrbLeft)
        .def_readonly("nOrbRight", &HamiltonianQCSCI<S>::nOrbRight)
        .def_readonly("nOrbCas", &HamiltonianQCSCI<S>::nOrbCas)
        .def_readwrite("useRuleQC", &HamiltonianQCSCI<S>::useRuleQC)
        .def_readwrite("ruleQC", &HamiltonianQCSCI<S>::ruleQC)
        .def_readwrite("parallelRule", &HamiltonianQCSCI<S>::parallelRule)
        .def_readwrite("fcidump", &HamiltonianQCSCI<S>::fcidump)
        .def_readwrite("mu", &HamiltonianQCSCI<S>::mu)
        .def("v", &HamiltonianQCSCI<S>::v)
        .def("t", &HamiltonianQCSCI<S>::t)
        .def("e", &HamiltonianQCSCI<S>::e);
    // vv hrl: switched off as not really required (now protected)
    //.def("init_site_ops", &HamiltonianQCSCI<S>::init_site_ops)
    //.def("get_site_ops", &HamiltonianQCSCI<S>::get_site_ops);
}

template <typename S> void bind_mpo_sci(py::module &m) {

    py::class_<DMRGSCI<S>, shared_ptr<DMRGSCI<S>>, DMRG<S>>(m, "DMRGSCI")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &>())
        .def_readwrite("last_site_svd", &DMRGSCI<S>::last_site_svd)
        .def_readwrite("last_site_1site", &DMRGSCI<S>::last_site_1site)
        .def("blocking", &DMRGSCI<S>::blocking);

    py::class_<LinearSCI<S>, shared_ptr<LinearSCI<S>>, Linear<S>>(m,
                                                                  "LinearSCI")
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
        .def_readwrite("last_site_svd", &LinearSCI<S>::last_site_svd)
        .def_readwrite("last_site_1site", &LinearSCI<S>::last_site_1site)
        .def("blocking", &LinearSCI<S>::blocking);

    py::class_<DMRGSCIAQCCOLD<S>, shared_ptr<DMRGSCIAQCCOLD<S>>, DMRGSCI<S>>(
        m, "DMRGSCIAQCCOLD")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &, double,
                      double, const std::vector<S> &>())
        .def_readwrite("max_aqcc_iter", &DMRGSCIAQCCOLD<S>::max_aqcc_iter)
        .def_readwrite("g_factor", &DMRGSCIAQCCOLD<S>::g_factor)
        .def_readwrite("delta_e", &DMRGSCIAQCCOLD<S>::delta_e)
        .def_readwrite("ref_energy", &DMRGSCIAQCCOLD<S>::ref_energy);

    py::class_<DMRGSCIAQCC<S>, shared_ptr<DMRGSCIAQCC<S>>, DMRGSCI<S>>(
        m, "DMRGSCIAQCC")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      double,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &,
                      double>(), "Frozen/CAS mode: Only one big site at the end")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      double,
                      const shared_ptr<MovingEnvironment<S>> &,
                      double,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &,
                      double>(), "Frozen/CAS mode ACPF2: Only one big site at the end")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      double,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &,
                      double>(), "RAS mode: Big sites on both ends")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      double,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      double,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &,
                      double>(), "RAS ACPF2 mode: Big sites on both ends")
        .def_readwrite("smallest_energy", &DMRGSCIAQCC<S>::smallest_energy)
        .def_readwrite("max_aqcc_iter", &DMRGSCIAQCC<S>::max_aqcc_iter)
        .def_readwrite("g_factor", &DMRGSCIAQCC<S>::g_factor)
        .def_readwrite("g_factor2", &DMRGSCIAQCC<S>::g_factor2)
        .def_readwrite("ACPF2_mode", &DMRGSCIAQCC<S>::ACPF2_mode)
        .def_readwrite("RAS_mode", &DMRGSCIAQCC<S>::RAS_mode)
        .def_readwrite("delta_e", &DMRGSCIAQCC<S>::delta_e)
        .def_readwrite("ref_energy", &DMRGSCIAQCC<S>::ref_energy);

    py::class_<MPOQCSCI<S>, shared_ptr<MPOQCSCI<S>>, MPO<S>>(m, "MPOQCSCI")
        .def_readwrite("mode", &MPOQCSCI<S>::mode)
        .def(py::init<const HamiltonianQCSCI<S> &>())
        .def(py::init<const HamiltonianQCSCI<S> &, QCTypes>());

    py::class_<SiteMPOSCI<S>, shared_ptr<SiteMPOSCI<S>>, MPO<S>>(m, "SiteMPOSCI")
            .def(py::init<const HamiltonianQCSCI<S> &,
                    const shared_ptr<OpElement<S>> &>())
            .def(py::init<const HamiltonianQCSCI<S> &, const shared_ptr<OpElement<S>> &,
                    int>());

    py::class_<IdentityMPOSCI<S>, shared_ptr<IdentityMPOSCI<S>>, MPO<S>>(m, "IdentityMPOSCI")
            .def(py::init<const HamiltonianQCSCI<S> &>());
}

template <typename S = void> void bind_types_sci(py::module &m) {
    py::enum_<DelayedSCIOpNames>(m, "DelayedSCIOpNames", py::arithmetic())
        .value("Nothing", DelayedSCIOpNames::None)
        .value("H", DelayedSCIOpNames::H)
        .value("Normal", DelayedSCIOpNames::Normal)
        .value("R", DelayedSCIOpNames::R)
        .value("RD", DelayedSCIOpNames::RD)
        .value("P", DelayedSCIOpNames::P)
        .value("PD", DelayedSCIOpNames::PD)
        .value("Q", DelayedSCIOpNames::Q)
        .value("LeftBig", DelayedSCIOpNames::LeftBig)
        .value("RightBig", DelayedSCIOpNames::RightBig)
        .def(py::self & py::self)
        .def(py::self | py::self);
}
