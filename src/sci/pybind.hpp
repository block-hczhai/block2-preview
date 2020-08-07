
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

template <typename S> void bind_sci_wrapper(py::module &m){
    py::class_<sci::AbstractSciWrapper<S>, shared_ptr<sci::AbstractSciWrapper<S>>>
                                                        (m, "AbstractSciWrapper")
        .def(py::init<int, int, const shared_ptr<FCIDUMP> &,
                                   int, int, int>(),
                           py::arg("nOrbCas"), py::arg("nOrbExt"), py::arg("fcidump"),
                           py::arg("nMaxAlphaEl"), py::arg("nMaxBetaEl"), py::arg("nMaxEl"),
                           "Initialization via generated CI space based on nMax*")
        .def(py::init<int, int, const shared_ptr<FCIDUMP> &,
                     const vector<vector<int>>&>(),
             py::arg("nOrbCas"), py::arg("nOrbExt"), py::arg("fcidump"),
             py::arg("occs"),
             "Initialization via externally given determinants in `occs`")
        .def_readonly("nOrbCas", &sci::AbstractSciWrapper<S>::nOrbCas)
        .def_readonly("nOrbExt", &sci::AbstractSciWrapper<S>::nOrbExt)
        .def_readonly("nOrb", &sci::AbstractSciWrapper<S>::nOrb)
        .def_readonly("nMaxAlphaEl", &sci::AbstractSciWrapper<S>::nMaxAlphaEl)
        .def_readonly("nMaxBetaEl", &sci::AbstractSciWrapper<S>::nMaxBetaEl)
        .def_readonly("nMaxEl", &sci::AbstractSciWrapper<S>::nMaxEl)
        .def_readonly("nDet", &sci::AbstractSciWrapper<S>::nDet)
        .def_readonly("eps", &sci::AbstractSciWrapper<S>::eps);
};

template <typename S> void bind_hamiltonian_sci(py::module &m) {

    py::class_<HamiltonianSCI<S>, shared_ptr<HamiltonianSCI<S>>>(m, "HamiltonianSCI")
            .def(py::init<S, int, const vector<uint8_t> &>())
            .def_readwrite("n_syms", &HamiltonianSCI<S>::n_syms)
            .def_readwrite("opf", &HamiltonianSCI<S>::opf)
            .def_readwrite("n_sites", &HamiltonianSCI<S>::n_sites)
            .def_readwrite("orb_sym", &HamiltonianSCI<S>::orb_sym)
            .def_readwrite("vacuum", &HamiltonianSCI<S>::vacuum)
            .def_property_readonly("basis",
                                   [](HamiltonianSCI<S> *self) {
                                       return Array<StateInfo<S>>(self->basis,
                                                                  self->n_syms);
                                   })
            .def("deallocate", &HamiltonianSCI<S>::deallocate);
           //vv hrl: switched off as not really required (now protected)
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

    py::class_<HamiltonianQCSCI<S>, shared_ptr<HamiltonianQCSCI<S>>, HamiltonianSCI<S>>(
            m, "HamiltonianQCSCI")
            .def(py::init<S, int, int, const vector<uint8_t> &,
                    const shared_ptr<FCIDUMP> &,
                    const std::shared_ptr<sci::AbstractSciWrapper<S>> &>(),
                    py::arg("vacuum"), py::arg("nOrbCas"), py::arg("nOrbExt"),
                    py::arg("orb_Sym"), py::arg("fcidump"), py::arg("sciWrapper"))
            .def_readwrite("fcidump", &HamiltonianQCSCI<S>::fcidump)
            .def_readwrite("mu", &HamiltonianQCSCI<S>::mu)
            .def("v", &HamiltonianQCSCI<S>::v)
            .def("t", &HamiltonianQCSCI<S>::t)
            .def("e", &HamiltonianQCSCI<S>::e);
            //vv hrl: switched off as not really required (now protected)
            //.def("init_site_ops", &HamiltonianQCSCI<S>::init_site_ops)
            //.def("get_site_ops", &HamiltonianQCSCI<S>::get_site_ops);
}

template <typename S> void bind_mpo_sci(py::module &m) {

    py::class_<MPOQCSCI<S>, shared_ptr<MPOQCSCI<S>>, MPO<S>>(m, "MPOQCSCI")
            .def_readwrite("mode", &MPOQCSCI<S>::mode)
            .def(py::init<const HamiltonianQCSCI<S> &>())
            .def(py::init<const HamiltonianQCSCI<S> &, QCTypes>());

}
