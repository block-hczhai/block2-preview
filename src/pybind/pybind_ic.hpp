
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2022 Huanchen Zhai <hczhai@caltech.edu>
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

namespace pybind11 {
template <typename Set, typename holder_type = std::unique_ptr<Set>,
          typename... Args>
class_<Set, holder_type> bind_set_block2(handle scope, std::string const &name,
                                         Args &&...args) {
    using Class_ = class_<Set, holder_type>;

    // If the value_type is unregistered (e.g. a converting type) or is itself
    // registered module-local then make the set binding module-local as well:
    using T = typename Set::value_type;
    using ItType = typename Set::iterator;

    auto vtype_info = detail::get_type_info(typeid(T));
    bool local = !vtype_info || vtype_info->module_local;

    Class_ cl(scope, name.c_str(), pybind11::module_local(local),
              std::forward<Args>(args)...);
    cl.def(init<>());
    cl.def(init<const Set &>(), "Copy constructor");
    cl.def(init([](iterable it) {
        auto s = std::unique_ptr<Set>(new Set());
        for (handle h : it)
            s->insert(h.cast<T>());
        return s.release();
    }));
    cl.def(self == self);
    cl.def(self != self);
    cl.def(
        "remove",
        [](Set &s, const T &x) {
            auto p = s.find(x);
            if (p != s.end())
                s.erase(p);
            else
                throw value_error();
        },
        arg("x"),
        "Remove the item from the set whose value is x. "
        "It is an error if there is no such item.");
    cl.def(
        "__contains__",
        [](const Set &s, const T &x) { return s.find(x) != s.end(); }, arg("x"),
        "Return true if the container contains ``x``.");
    cl.def(
        "add", [](Set &s, const T &value) { s.insert(value); }, arg("x"),
        "Add an item to the set.");
    cl.def(
        "clear", [](Set &s) { s.clear(); }, "Clear the contents.");
    cl.def(
        "__iter__",
        [](Set &s) {
            return make_iterator<return_value_policy::copy, ItType, ItType, T>(
                s.begin(), s.end());
        },
        keep_alive<0, 1>());
    cl.def(
        "__repr__",
        [name](Set &s) {
            std::ostringstream os;
            os << name << '{';
            for (auto it = s.begin(); it != s.end(); ++it) {
                if (it != s.begin())
                    os << ", ";
                os << *it;
            }
            os << '}';
            return os.str();
        },
        "Return the canonical string representation of this set.");
    cl.def(
        "__bool__", [](const Set &s) -> bool { return !s.empty(); },
        "Check whether the set is nonempty");
    cl.def("__len__", &Set::size);

    return cl;
}
} // namespace pybind11

#include "../block2_ic.hpp"

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<WickIndex>);
PYBIND11_MAKE_OPAQUE(std::set<WickIndex>);
PYBIND11_MAKE_OPAQUE(vector<WickPermutation>);
PYBIND11_MAKE_OPAQUE(vector<WickTensor>);
PYBIND11_MAKE_OPAQUE(vector<WickString>);
PYBIND11_MAKE_OPAQUE(vector<WickExpr>);
PYBIND11_MAKE_OPAQUE(map<WickIndexTypes, std::set<WickIndex>>);
PYBIND11_MAKE_OPAQUE(map<string, pair<WickTensor, vector<WickString>>>);
PYBIND11_MAKE_OPAQUE(map<pair<string, int>, vector<WickPermutation>>);
PYBIND11_MAKE_OPAQUE(map<string, pair<WickTensor, WickExpr>>);

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
        .value("External", WickIndexTypes::External)
        .value("Alpha", WickIndexTypes::Alpha)
        .value("Beta", WickIndexTypes::Beta)
        .value("AlphaBeta", WickIndexTypes::AlphaBeta)
        .value("InactiveAlpha", WickIndexTypes::InactiveAlpha)
        .value("ActiveAlpha", WickIndexTypes::ActiveAlpha)
        .value("ExternalAlpha", WickIndexTypes::ExternalAlpha)
        .value("InactiveBeta", WickIndexTypes::InactiveBeta)
        .value("ActiveBeta", WickIndexTypes::ActiveBeta)
        .value("ExternalBeta", WickIndexTypes::ExternalBeta)
        .def(py::self & py::self)
        .def(py::self | py::self)
        .def(~py::self);

    py::enum_<WickTensorTypes>(m, "WickTensorTypes", py::arithmetic())
        .value("CreationOperator", WickTensorTypes::CreationOperator)
        .value("DestroyOperator", WickTensorTypes::DestroyOperator)
        .value("SpinFreeOperator", WickTensorTypes::SpinFreeOperator)
        .value("KroneckerDelta", WickTensorTypes::KroneckerDelta)
        .value("Tensor", WickTensorTypes::Tensor);

    py::class_<WickIndex, shared_ptr<WickIndex>>(m, "WickIndex")
        .def(py::init<>())
        .def(py::init<const string &>())
        .def(py::init<const string &, WickIndexTypes>())
        .def_readwrite("name", &WickIndex::name)
        .def_readwrite("types", &WickIndex::types)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__hash__", &WickIndex::hash)
        .def("__repr__",
             [](WickIndex *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("has_types", &WickIndex::has_types)
        .def("is_short", &WickIndex::is_short)
        .def("with_no_types", &WickIndex::with_no_types)
        .def_static("parse", &WickIndex::parse)
        .def_static("add_types", &WickIndex::add_types)
        .def_static("parse_with_types", &WickIndex::parse_with_types)
        .def_static("parse_set", &WickIndex::parse_set)
        .def_static("parse_set_with_types", &WickIndex::parse_set_with_types);

    py::bind_vector<vector<WickIndex>>(m, "VectorWickIndex");
    py::bind_set_block2<std::set<WickIndex>>(m, "SetWickIndex");
    py::bind_map<map<WickIndexTypes, set<WickIndex>>>(m,
                                                      "MapWickIndexTypesSet");

    py::class_<WickPermutation, shared_ptr<WickPermutation>>(m,
                                                             "WickPermutation")
        .def(py::init<>())
        .def(py::init<const vector<int16_t> &>())
        .def(py::init<const vector<int16_t> &, bool>())
        .def_readwrite("data", &WickPermutation::data)
        .def_readwrite("negative", &WickPermutation::negative)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self * py::self)
        .def("__hash__", &WickPermutation::hash)
        .def("__repr__",
             [](WickPermutation *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def_static("complete_set", &WickPermutation::complete_set)
        .def_static("non_symmetric", &WickPermutation::non_symmetric)
        .def_static("two_symmetric", &WickPermutation::two_symmetric)
        .def_static("two_anti_symmetric", &WickPermutation::two_anti_symmetric)
        .def_static("four_anti_symmetric",
                    &WickPermutation::four_anti_symmetric)
        .def_static("qc_chem", &WickPermutation::qc_chem)
        .def_static("qc_phys", &WickPermutation::qc_phys)
        .def_static("four_anti", &WickPermutation::four_anti)
        .def_static("pair_anti_symmetric",
                    &WickPermutation::pair_anti_symmetric)
        .def_static("all", &WickPermutation::all)
        .def_static("pair_symmetric", &WickPermutation::pair_symmetric,
                    py::arg("n"), py::arg("hermitian") = false);

    py::bind_vector<vector<WickPermutation>>(m, "VectorWickPermutation");
    py::bind_map<map<pair<string, int>, vector<WickPermutation>>>(
        m, "MapPStrIntVectorWickPermutation");

    py::class_<WickTensor, shared_ptr<WickTensor>>(m, "WickTensor")
        .def(py::init<>())
        .def(py::init<const string &, const vector<WickIndex> &>())
        .def(py::init<const string &, const vector<WickIndex> &,
                      const vector<WickPermutation> &>())
        .def(py::init<const string &, const vector<WickIndex> &,
                      const vector<WickPermutation> &, WickTensorTypes>())
        .def_readwrite("name", &WickTensor::name)
        .def_readwrite("indices", &WickTensor::indices)
        .def_readwrite("perms", &WickTensor::perms)
        .def_readwrite("type", &WickTensor::type)
        .def_static("reset_permutations", &WickTensor::reset_permutations)
        .def_static("parse", &WickTensor::parse)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self * WickPermutation())
        .def("fermi_type", &WickTensor::fermi_type)
        .def("to_str", &WickTensor::to_str)
        .def("__repr__",
             [](WickTensor *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def_static("kronecker_delta", &WickTensor::kronecker_delta)
        .def_static("spin_free", &WickTensor::spin_free)
        .def_static("spin_free_density_matrix",
                    &WickTensor::spin_free_density_matrix)
        .def_static("cre",
                    (WickTensor(*)(const WickIndex &, const string &)) &
                        WickTensor::cre,
                    py::arg("index"), py::arg("name") = string("C"))
        .def_static("cre",
                    (WickTensor(*)(const WickIndex &,
                                   const map<WickIndexTypes, set<WickIndex>> &,
                                   const string &)) &
                        WickTensor::cre,
                    py::arg("index"), py::arg("idx_map"),
                    py::arg("name") = string("C"))
        .def_static("des",
                    (WickTensor(*)(const WickIndex &, const string &)) &
                        WickTensor::des,
                    py::arg("index"), py::arg("name") = string("D"))
        .def_static("des",
                    (WickTensor(*)(const WickIndex &,
                                   const map<WickIndexTypes, set<WickIndex>> &,
                                   const string &)) &
                        WickTensor::des,
                    py::arg("index"), py::arg("idx_map"),
                    py::arg("name") = string("D"))
        .def("get_spin_tag", &WickTensor::get_spin_tag)
        .def("set_spin_tag", &WickTensor::set_spin_tag)
        .def("fermi_type_compare", &WickTensor::fermi_type_compare)
        .def("fermi_type", &WickTensor::fermi_type)
        .def("sort",
             [](WickTensor *self) {
                 double factor = 1.0;
                 self->sort(factor);
                 return factor;
             })
        .def("get_permutation_rules", &WickTensor::get_permutation_rules)
        .def_static("get_index_map", &WickTensor::get_index_map)
        .def_static("get_all_index_permutations",
                    &WickTensor::get_all_index_permutations);

    py::bind_vector<vector<WickTensor>>(m, "VectorWickTensor");

    py::class_<WickString, shared_ptr<WickString>>(m, "WickString")
        .def(py::init<>())
        .def(py::init<const WickTensor &>())
        .def(py::init<const WickTensor &, double>())
        .def(py::init<const vector<WickTensor> &>())
        .def(py::init<const vector<WickTensor> &, const set<WickIndex> &>())
        .def(py::init<const vector<WickTensor> &, const set<WickIndex> &,
                      double>())
        .def_readwrite("tensors", &WickString::tensors)
        .def_readwrite("ctr_indices", &WickString::ctr_indices)
        .def_readwrite("factor", &WickString::factor)
        .def("abs_equal_to", &WickString::abs_equal_to)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self * py::self)
        .def(py::self + py::self)
        .def(py::self * double())
        .def("__abs__", &WickString::abs)
        .def_static("parse", &WickString::parse)
        .def("substitute", &WickString::substitute)
        .def("index_map", &WickString::index_map)
        .def("used_indices", &WickString::used_indices)
        .def("used_spin_tags", &WickString::used_spin_tags)
        .def("group_less", &WickString::group_less)
        .def("has_inactive_ops", &WickString::has_inactive_ops)
        .def("has_external_ops", &WickString::has_external_ops)
        .def("simple_sort", &WickString::simple_sort)
        .def("quick_sort", &WickString::quick_sort)
        .def("simplify_delta", &WickString::simplify_delta)
        .def("__repr__", [](WickString *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<WickString>>(m, "VectorWickString");
    py::bind_map<map<string, pair<WickTensor, vector<WickString>>>>(
        m, "MapStrPVectorWickString");

    py::class_<WickExpr, shared_ptr<WickExpr>>(m, "WickExpr")
        .def(py::init<>())
        .def(py::init<const WickString &>())
        .def(py::init<const vector<WickString> &>())
        .def_readwrite("terms", &WickExpr::terms)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self ^ py::self)
        .def(py::self & py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def_static("parse", &WickExpr::parse, py::arg("tex_expr"),
                    py::arg("idx_map"),
                    py::arg("perm_map") =
                        map<pair<string, int>, vector<WickPermutation>>())
        .def_static("parse_def", &WickExpr::parse_def, py::arg("tex_expr"),
                    py::arg("idx_map"),
                    py::arg("perm_map") =
                        map<pair<string, int>, vector<WickPermutation>>())
        .def_static("split_index_types_static",
                    (WickExpr(*)(const WickString &)) &
                        WickExpr::split_index_types)
        .def("split_index_types",
             (WickExpr(WickExpr::*)() const) & WickExpr::split_index_types)
        .def("to_einsum", &WickExpr::to_einsum)
        .def_static("to_einsum_add_indent", &WickExpr::to_einsum_add_indent,
                    py::arg("x"), py::arg("indent") = 4)
        .def("substitute", &WickExpr::substitute)
        .def("index_map", &WickExpr::index_map)
        .def("expand", &WickExpr::expand, py::arg("max_unctr") = -1,
             py::arg("no_ctr") = false, py::arg("no_unctr_sf_inact") = true)
        .def("simple_sort", &WickExpr::simple_sort)
        .def("simplify_delta", &WickExpr::simplify_delta)
        .def("simplify_zero", &WickExpr::simplify_zero)
        .def("simplify_merge", &WickExpr::simplify_merge)
        .def("simplify", &WickExpr::simplify)
        .def("remove_external", &WickExpr::remove_external)
        .def("remove_inactive", &WickExpr::remove_inactive)
        .def("add_spin_free_trans_symm", &WickExpr::add_spin_free_trans_symm)
        .def("conjugate", &WickExpr::conjugate)
        .def("__repr__", [](WickExpr *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<WickExpr>>(m, "VectorWickExpr");
    py::bind_map<map<string, pair<WickTensor, WickExpr>>>(
        m, "MapStrPWickTensorExpr");

    py::class_<WickCCSD, shared_ptr<WickCCSD>>(m, "WickCCSD")
        .def(py::init<>())
        .def(py::init<bool>())
        .def_readwrite("idx_map", &WickCCSD::idx_map)
        .def_readwrite("perm_map", &WickCCSD::perm_map)
        .def_readwrite("h1", &WickCCSD::h1)
        .def_readwrite("h2", &WickCCSD::h2)
        .def_readwrite("h", &WickCCSD::h)
        .def_readwrite("t1", &WickCCSD::t1)
        .def_readwrite("t2", &WickCCSD::t2)
        .def_readwrite("t", &WickCCSD::t)
        .def_readwrite("ex1", &WickCCSD::ex1)
        .def_readwrite("ex2", &WickCCSD::ex2)
        .def("energy_equations", &WickCCSD::energy_equations)
        .def("t1_equations", &WickCCSD::t1_equations)
        .def("t2_equations", &WickCCSD::t2_equations);

    py::class_<WickUGACCSD, shared_ptr<WickUGACCSD>>(m, "WickUGACCSD")
        .def(py::init<>())
        .def_readwrite("idx_map", &WickUGACCSD::idx_map)
        .def_readwrite("perm_map", &WickUGACCSD::perm_map)
        .def_readwrite("defs", &WickUGACCSD::defs)
        .def_readwrite("h1", &WickUGACCSD::h1)
        .def_readwrite("h2", &WickUGACCSD::h2)
        .def_readwrite("h", &WickUGACCSD::h)
        .def_readwrite("e0", &WickUGACCSD::e0)
        .def_readwrite("t1", &WickUGACCSD::t1)
        .def_readwrite("t2", &WickUGACCSD::t2)
        .def_readwrite("t", &WickUGACCSD::t)
        .def_readwrite("ex1", &WickUGACCSD::ex1)
        .def_readwrite("ex2", &WickUGACCSD::ex2)
        .def("energy_equations", &WickUGACCSD::energy_equations)
        .def("t1_equations", &WickUGACCSD::t1_equations)
        .def("t2_equations", &WickUGACCSD::t2_equations);

    py::class_<WickSCNEVPT2, shared_ptr<WickSCNEVPT2>>(m, "WickSCNEVPT2")
        .def(py::init<>())
        .def_readwrite("idx_map", &WickSCNEVPT2::idx_map)
        .def_readwrite("perm_map", &WickSCNEVPT2::perm_map)
        .def_readwrite("defs", &WickSCNEVPT2::defs)
        .def_readwrite("sub_spaces", &WickSCNEVPT2::sub_spaces)
        .def_readwrite("heff", &WickSCNEVPT2::heff)
        .def_readwrite("hw", &WickSCNEVPT2::hw)
        .def_readwrite("hd", &WickSCNEVPT2::hd)
        .def("build_communicator",
             (WickExpr(WickSCNEVPT2::*)(const string &, const string &,
                                        bool do_sum) const) &
                 WickSCNEVPT2::build_communicator,
             py::arg("bra"), py::arg("ket"), py::arg("do_sum") = true)
        .def("build_communicator",
             (WickExpr(WickSCNEVPT2::*)(const string &, bool do_sum) const) &
                 WickSCNEVPT2::build_communicator,
             py::arg("ket"), py::arg("do_sum") = true)
        .def("build_norm", &WickSCNEVPT2::build_norm, py::arg("ket"),
             py::arg("do_sum") = true)
        .def("to_einsum_orb_energies", &WickSCNEVPT2::to_einsum_orb_energies)
        .def("to_einsum_sum_restriction",
             &WickSCNEVPT2::to_einsum_sum_restriction)
        .def("to_einsum", &WickSCNEVPT2::to_einsum);

    py::class_<WickICNEVPT2, shared_ptr<WickICNEVPT2>, WickSCNEVPT2>(
        m, "WickICNEVPT2")
        .def(py::init<>())
        .def("build_norm", &WickICNEVPT2::build_norm, py::arg("bra"),
             py::arg("ket"), py::arg("do_sum") = true)
        .def("build_rhs", &WickICNEVPT2::build_rhs, py::arg("bra"),
             py::arg("ket"), py::arg("do_sum") = true)
        .def("to_einsum_orb_energies", &WickICNEVPT2::to_einsum_orb_energies)
        .def("to_einsum_sum_restriction",
             &WickICNEVPT2::to_einsum_sum_restriction, py::arg("tensor"),
             py::arg("restrict_cas") = true, py::arg("no_eq") = false)
        .def("to_einsum", &WickICNEVPT2::to_einsum);

    py::class_<WickICMRCI, shared_ptr<WickICMRCI>>(m, "WickICMRCI")
        .def(py::init<>())
        .def_readwrite("idx_map", &WickICMRCI::idx_map)
        .def_readwrite("perm_map", &WickICMRCI::perm_map)
        .def_readwrite("sub_spaces", &WickICMRCI::sub_spaces)
        .def_readwrite("h1", &WickICMRCI::h1)
        .def_readwrite("h2", &WickICMRCI::h2)
        .def_readwrite("h", &WickICMRCI::h)
        .def("build_hamiltonian", &WickICMRCI::build_hamiltonian,
             py::arg("bra"), py::arg("ket"), py::arg("do_sum") = true,
             py::arg("do_comm") = true)
        .def("build_overlap", &WickICMRCI::build_overlap, py::arg("bra"),
             py::arg("ket"), py::arg("do_sum") = true)
        .def("to_einsum_zeros", &WickICMRCI::to_einsum_zeros)
        .def("to_einsum_sum_restriction",
             &WickICMRCI::to_einsum_sum_restriction, py::arg("tensor"),
             py::arg("eq_pattern") = string("+"))
        .def("to_einsum", &WickICMRCI::to_einsum);
}

template <typename S = void>
auto bind_guga(py::module &m)
    -> decltype(typename enable_if<is_void<S>::value>::type()) {

    py::class_<PaldusTable, shared_ptr<PaldusTable>>(m, "PaldusTable")
        .def(py::init<>())
        .def(py::init<int>())
        .def_readwrite("abc", &PaldusTable::abc)
        .def_property_readonly("n_rows", &PaldusTable::n_rows)
        .def_property_readonly("n_elec", &PaldusTable::n_elec)
        .def_property_readonly("twos", &PaldusTable::twos)
        .def("sanity_check", &PaldusTable::sanity_check)
        .def("diff", &PaldusTable::diff)
        .def("accu", &PaldusTable::accu)
        .def("to_step_vector", &PaldusTable::to_step_vector)
        .def("from_step_vector", &PaldusTable::from_step_vector)
        .def("__repr__", &PaldusTable::to_str);

    py::class_<DistinctRowTable<S>, shared_ptr<DistinctRowTable<S>>,
               PaldusTable>(m, "DistinctRowTable")
        .def(py::init<>())
        .def(py::init<int16_t, int16_t, int16_t>())
        .def_readwrite("jd", &DistinctRowTable<S>::jd)
        .def_property_readonly("n_drt", &DistinctRowTable<S>::n_drt)
        .def("initialize", &DistinctRowTable<S>::initialize)
        .def("find_row", &DistinctRowTable<S>::find_row, py::arg("a"),
             py::arg("b"), py::arg("start") = 0)
        .def("initialize", &DistinctRowTable<S>::initialize)
        .def("step_vector_to_arc", &DistinctRowTable<S>::step_vector_to_arc)
        .def("index_of_step_vector",
             &DistinctRowTable<S>::index_of_step_vector);

    py::class_<MRCIDistinctRowTable<S>, shared_ptr<MRCIDistinctRowTable<S>>,
               DistinctRowTable<S>>(m, "MRCIDistinctRowTable")
        .def(py::init<>())
        .def(py::init<int16_t, int16_t, int16_t>())
        .def_readwrite("nref", &MRCIDistinctRowTable<S>::nref)
        .def_readwrite("nex", &MRCIDistinctRowTable<S>::nex)
        .def_readwrite("ncore", &MRCIDistinctRowTable<S>::ncore)
        .def_readwrite("nvirt", &MRCIDistinctRowTable<S>::nvirt)
        .def_readwrite("ts", &MRCIDistinctRowTable<S>::ts)
        .def_readwrite("refs", &MRCIDistinctRowTable<S>::refs)
        .def("initialize_mrci", &MRCIDistinctRowTable<S>::initialize_mrci);
}

template <typename S = void>
auto bind_guga(py::module &m)
    -> decltype(typename enable_if<!is_void<S>::value>::type()) {

    py::class_<DistinctRowTable<S>, shared_ptr<DistinctRowTable<S>>,
               DistinctRowTable<void>>(m, "DistinctRowTable")
        .def(py::init<const vector<typename S::pg_t> &>())
        .def(py::init<int16_t, int16_t, int16_t, typename S::pg_t,
                      const vector<typename S::pg_t> &>())
        .def_readwrite("pgs", &DistinctRowTable<S>::pgs)
        .def_readwrite("orb_sym", &DistinctRowTable<S>::orb_sym);

    py::class_<MRCIDistinctRowTable<S>, shared_ptr<MRCIDistinctRowTable<S>>,
               MRCIDistinctRowTable<void>>(m, "MRCIDistinctRowTable")
        .def(py::init<const vector<typename S::pg_t> &>())
        .def(py::init<int16_t, int16_t, int16_t, typename S::pg_t,
                      const vector<typename S::pg_t> &>())
        .def_readwrite("pgs", &MRCIDistinctRowTable<S>::pgs)
        .def_readwrite("orb_sym", &MRCIDistinctRowTable<S>::orb_sym);
}
