
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2021 Seunghoon Lee <seunghoonlee89@gmail.com>
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
#include "../block2_sp_dmrg.hpp"

namespace py = pybind11;
using namespace block2;

template <typename S> void bind_sp_dmrg(py::module &m) {
    py::class_<StochasticPDMRG<S>, shared_ptr<StochasticPDMRG<S>>>(
        m, "StochasticPDMRG")
        .def(py::init<>())
        .def(py::init<const shared_ptr<UnfusedMPS<S>> &, const shared_ptr<UnfusedMPS<S>> &, double>())
        .def_readwrite("phys_dim", &StochasticPDMRG<S>::phys_dim)
        .def_readwrite("tensors_psi0", &StochasticPDMRG<S>::tensors_psi0)
        .def_readwrite("tensors_qvpsi0", &StochasticPDMRG<S>::tensors_qvpsi0)
        .def_readwrite("pinfos_psi0", &StochasticPDMRG<S>::pinfos_psi0)
        .def_readwrite("pinfos_qvpsi0", &StochasticPDMRG<S>::pinfos_qvpsi0)
        .def_readwrite("norm_qvpsi0", &StochasticPDMRG<S>::norm_qvpsi0)
        .def_readwrite("n_sites", &StochasticPDMRG<S>::n_sites)
        .def_readwrite("center", &StochasticPDMRG<S>::center)
        .def_readwrite("dot", &StochasticPDMRG<S>::dot)
        .def_readwrite("canonical_form", &StochasticPDMRG<S>::canonical_form)
        .def_readwrite("det_string", &StochasticPDMRG<S>::det_string)
        .def("E0", 
             [](StochasticPDMRG<S> *self, const shared_ptr<FCIDUMP> &fcidump,
                py::array_t<double> &e_pqqp, py::array_t<double> &e_pqpq,
                py::array_t<double> &one_pdm) {
                 assert(e_pqqp.ndim() == 2);
                 assert(e_pqpq.ndim() == 2);
                 assert(one_pdm.ndim() == 2);
                 assert(e_pqqp.strides()[1] == sizeof(double));
                 assert(e_pqpq.strides()[1] == sizeof(double));
                 assert(one_pdm.strides()[1] == sizeof(double));
                 MatrixRef dm_pqqp(e_pqqp.mutable_data(), e_pqqp.shape()[0], e_pqqp.shape()[1]);
                 MatrixRef dm_pqpq(e_pqpq.mutable_data(), e_pqpq.shape()[0], e_pqpq.shape()[1]);
                 MatrixRef dm_one(one_pdm.mutable_data(), one_pdm.shape()[0], one_pdm.shape()[1]);
                 double E0 = self->E0(fcidump, dm_pqqp, dm_pqpq, dm_one);
                 return E0;
             })
        .def("sampling_ab", &StochasticPDMRG<S>::sampling_ab)
        .def("sampling_c", &StochasticPDMRG<S>::sampling_c)
        .def("overlap_ab", &StochasticPDMRG<S>::overlap_ab)
        .def("overlap_c", &StochasticPDMRG<S>::overlap_c)
        .def("clear", &StochasticPDMRG<S>::clear);
}
