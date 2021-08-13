
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
        .def(py::init<const shared_ptr<UnfusedMPS<S>> &,
                      const shared_ptr<UnfusedMPS<S>> &, double>())
        .def_readwrite("phys_dim", &StochasticPDMRG<S>::phys_dim)
        .def_readwrite("tensors_psi0", &StochasticPDMRG<S>::tensors_psi0)
        .def_readwrite("tensors_qvpsi0", &StochasticPDMRG<S>::tensors_qvpsi0)
        .def_readwrite("pinfos_psi0", &StochasticPDMRG<S>::pinfos_psi0)
        .def_readwrite("pinfos_qvpsi0", &StochasticPDMRG<S>::pinfos_qvpsi0)
        .def_readwrite("norm_qvpsi0", &StochasticPDMRG<S>::norm_qvpsi0)
        .def_readwrite("n_sites", &StochasticPDMRG<S>::n_sites)
        .def("energy_zeroth",
             [](StochasticPDMRG<S> *self, const shared_ptr<FCIDUMP> &fcidump,
                py::array_t<double> &e_pqqp, py::array_t<double> &e_pqpq,
                py::array_t<double> &one_pdm) {
                 assert(e_pqqp.ndim() == 2);
                 assert(e_pqpq.ndim() == 2);
                 assert(one_pdm.ndim() == 2);
                 assert(e_pqqp.strides()[1] == sizeof(double));
                 assert(e_pqpq.strides()[1] == sizeof(double));
                 assert(one_pdm.strides()[1] == sizeof(double));
                 MatrixRef dm_pqqp(e_pqqp.mutable_data(), e_pqqp.shape()[0],
                                   e_pqqp.shape()[1]);
                 MatrixRef dm_pqpq(e_pqpq.mutable_data(), e_pqpq.shape()[0],
                                   e_pqpq.shape()[1]);
                 MatrixRef dm_one(one_pdm.mutable_data(), one_pdm.shape()[0],
                                  one_pdm.shape()[1]);
                 double ener =
                     self->energy_zeroth(fcidump, dm_pqqp, dm_pqpq, dm_one);
                 return ener;
             })
        .def("sampling", &StochasticPDMRG<S>::sampling)
        .def("overlap", &StochasticPDMRG<S>::overlap)
        .def("parallel_sampling", &StochasticPDMRG<S>::parallel_sampling);
}
