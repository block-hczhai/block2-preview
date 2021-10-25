
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

#include "../block2_ic.hpp"

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<WickIndex>);
PYBIND11_MAKE_OPAQUE(vector<WickPermutation>);
PYBIND11_MAKE_OPAQUE(vector<WickTensor>);
PYBIND11_MAKE_OPAQUE(vector<WickString>);

template <typename S = void> void bind_nd_array(py::module &m) {
    py::class_<NDArray, shared_ptr<NDArray>>(m, "NDArray",
                                             py::buffer_protocol())
        .def_readwrite("shape", &NDArray::shape)
        .def_readwrite("strides", &NDArray::strides)
        .def_property_readonly("ndim", &NDArray::ndim)
        .def_property_readonly("size", &NDArray::size)
        .def(py::init(
                 [](py::array_t<double> &arr, bool copy = true) { // will copy
                     vector<MKL_INT> shape(arr.ndim());
                     vector<ssize_t> strides(arr.ndim());
                     for (int i = 0; i < arr.ndim(); i++)
                         shape[i] = arr.shape()[i],
                         strides[i] = arr.strides()[i] / sizeof(double);
                     if (copy) {
                         NDArray r(shape, strides, nullptr);
                         r.vdata = make_shared<vector<double>>(
                             arr.data(), arr.data() + r.max_size());
                         r.data = r.vdata->data();
                         return r;
                     } else
                         return NDArray(shape, strides, arr.mutable_data());
                 }),
             py::arg("arr"), py::arg("copy") = true)
        .def_static("zeros",
                    [](const py::tuple &t) {
                        vector<MKL_INT> shapes(t.size());
                        for (int i = 0; i < (int)t.size(); i++)
                            shapes[i] = t[i].cast<MKL_INT>();
                        return NDArray(shapes);
                    })
        .def_static("zeros",
                    [](MKL_INT t) {
                        vector<MKL_INT> shapes = {t};
                        return NDArray(shapes);
                    })
        .def_static("random",
                    [](const py::tuple &t) {
                        vector<MKL_INT> shapes(t.size());
                        for (int i = 0; i < (int)t.size(); i++)
                            shapes[i] = t[i].cast<MKL_INT>();
                        return NDArray::random(shapes);
                    })
        .def_static("random",
                    [](MKL_INT t) {
                        vector<MKL_INT> shapes = {t};
                        return NDArray::random(shapes);
                    })
        .def_static("einsum",
                    [](const string &script, py::args &args) {
                        vector<NDArray> xarrs;
                        for (auto &x : args)
                            xarrs.push_back(x.cast<NDArray>());
                        return NDArray::einsum(script, xarrs);
                    })
        .def("transpose",
             [](NDArray *self, const py::tuple &t) {
                 vector<int> perm(t.size());
                 for (int i = 0; i < (int)t.size(); i++)
                     perm[i] = t[i].cast<int>();
                 return self->transpose(perm);
             })
        .def("diag",
             [](NDArray *self, const py::tuple &t) {
                 vector<int> perm(t.size());
                 for (int i = 0; i < (int)t.size(); i++)
                     perm[i] = t[i].cast<int>();
                 return self->diag(perm);
             })
        .def("is_c_order", &NDArray::is_c_order)
        .def("copy",
             [](NDArray *self) {
                 NDArray r(self->shape);
                 NDArray::transpose(*self, r);
                 return r;
             })
        .def("__repr__",
             [](NDArray *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("__getitem__",
             [](NDArray *self, const py::object &t) -> py::object {
                 vector<NDArraySlice> idxs;
                 int n_int = 0;
                 if (t.is_none())
                     idxs.push_back(NDArraySlice(0, 0, 0));
                 else if (py::isinstance<py::slice>(t)) {
                     py::slice tx = t.cast<py::slice>();
                     ssize_t st, ed, sp, sl;
                     tx.compute(self->shape[0], &st, &ed, &sp, &sl);
                     idxs.push_back(NDArraySlice((int)st, (int)ed, (int)sp));
                 } else if (py::isinstance<py::int_>(t)) {
                     idxs.push_back(NDArraySlice(t.cast<int>(), -1, 0));
                     n_int++;
                 } else if (py::isinstance<py::tuple>(t)) {
                     int j = 0;
                     py::tuple tx = t.cast<py::tuple>();
                     for (int i = 0; i < (int)tx.size(); i++)
                         if (tx[i].is_none())
                             idxs.push_back(NDArraySlice(0, 0, 0));
                         else if (py::isinstance<py::slice>(tx[i])) {
                             py::slice txx = tx[i].cast<py::slice>();
                             ssize_t st, ed, sp, sl;
                             assert(j < self->ndim());
                             txx.compute(self->shape[j++], &st, &ed, &sp, &sl);
                             idxs.push_back(
                                 NDArraySlice((int)st, (int)ed, (int)sp));
                         } else if (py::isinstance<py::int_>(tx[i])) {
                             idxs.push_back(
                                 NDArraySlice(tx[i].cast<int>(), -1, 0));
                             j++;
                             n_int++;
                         }
                 }
                 if (n_int == self->ndim()) {
                     vector<MKL_INT> iidx;
                     for (auto &x : idxs)
                         iidx.push_back(x.start);
                     return py::cast((*self)[iidx]);
                 } else
                     return py::cast(self->slice(idxs));
             })
        .def(py::self + py::self)
        .def(-py::self)
        .def_buffer([](NDArray *self) -> py::buffer_info {
            vector<ssize_t> shape(self->ndim()), strides(self->ndim());
            for (int i = 0; i < self->ndim(); i++)
                shape[i] = self->shape[i],
                strides[i] = self->strides[i] * sizeof(double);
            return py::buffer_info(self->data, sizeof(double),
                                   py::format_descriptor<double>::format(),
                                   self->ndim(), shape, strides);
        });
}

template <typename S = void> void bind_wick(py::module &m) {

    py::enum_<WickIndexTypes>(m, "WickIndexTypes", py::arithmetic())
        .value("Nothing", WickIndexTypes::None)
        .value("Inactive", WickIndexTypes::Inactive)
        .value("Active", WickIndexTypes::Active)
        .value("External", WickIndexTypes::External);

    py::enum_<WickTensorTypes>(m, "WickTensorTypes", py::arithmetic())
        .value("CreationOperator", WickTensorTypes::CreationOperator)
        .value("DestroyOperator", WickTensorTypes::DestroyOperator)
        .value("SpinFreeOperator", WickTensorTypes::SpinFreeOperator)
        .value("KroneckerDelta", WickTensorTypes::KroneckerDelta)
        .value("Tensor", WickTensorTypes::Tensor);

    py::class_<WickIndex, shared_ptr<WickIndex>>(m, "WickIndex")
        .def(py::init<>())
        .def(py::init<const string &>());

    py::bind_vector<vector<WickIndex>>(m, "VectorWickIndex");

    py::class_<WickPermutation, shared_ptr<WickPermutation>>(m,
                                                             "WickPermutation")
        .def(py::init<>())
        .def(py::init<const vector<int16_t> &>())
        .def(py::init<const vector<int16_t> &, bool>());

    py::class_<WickTensor, shared_ptr<WickTensor>>(m, "WickTensor")
        .def(py::init<>())
        .def(py::init<const string &, const vector<WickIndex> &>())
        .def_static("kronecker_delta", &WickTensor::kronecker_delta)
        .def_static("spin_free", &WickTensor::spin_free)
        .def_static("cre", (WickTensor(*)(const WickIndex &, const string &)) &
                               WickTensor::cre)
        .def_static("cre",
                    (WickTensor(*)(const WickIndex &,
                                   const map<WickIndexTypes, set<WickIndex>> &,
                                   const string &)) &
                        WickTensor::cre)
        .def_static("des", (WickTensor(*)(const WickIndex &, const string &)) &
                               WickTensor::des)
        .def_static("des",
                    (WickTensor(*)(const WickIndex &,
                                   const map<WickIndexTypes, set<WickIndex>> &,
                                   const string &)) &
                        WickTensor::des);

    py::class_<WickString, shared_ptr<WickString>>(m, "WickString")
        .def(py::init<>())
        .def(py::init<const vector<WickTensor> &>())
        .def(py::init<const vector<WickTensor> &, const set<WickIndex> &>());

    py::class_<WickExpr, shared_ptr<WickExpr>>(m, "WickExpr")
        .def(py::init<>())
        .def(py::init<const WickString &>())
        .def(py::init<const vector<WickString> &>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def("expand", &WickExpr::expand, py::arg("full_ctr") = false,
             py::arg("no_ctr") = false)
        .def("simple_sort", &WickExpr::simple_sort)
        .def("simplify_delta", &WickExpr::simplify_delta)
        .def("simplify_zero", &WickExpr::simplify_zero)
        .def("simplify_merge", &WickExpr::simplify_merge)
        .def("simplify", &WickExpr::simplify)
        .def("__repr__", [](WickExpr *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<WickCCSD, shared_ptr<WickCCSD>>(m, "WickCCSD")
        .def(py::init<>())
        .def_readwrite("h1", &WickCCSD::h1)
        .def_readwrite("h2", &WickCCSD::h2)
        .def_readwrite("h", &WickCCSD::h)
        .def_readwrite("t1", &WickCCSD::t1)
        .def_readwrite("t2", &WickCCSD::t2)
        .def_readwrite("t", &WickCCSD::t)
        .def_readwrite("ex1", &WickCCSD::ex1)
        .def_readwrite("ex2", &WickCCSD::ex2)
        .def("t1_equations", &WickCCSD::t1_equations)
        .def("t2_equations", &WickCCSD::t2_equations);
}
