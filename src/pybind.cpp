
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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

#include "block2.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <tuple>

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(vector<double>);
PYBIND11_MAKE_OPAQUE(vector<ActiveTypes>);
// SZ
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpString<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SZ, shared_ptr<SparseMatrixInfo<SZ>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic<SZ>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SZ>>>);
PYBIND11_MAKE_OPAQUE(map<shared_ptr<OpExpr<SZ>>, shared_ptr<SparseMatrix<SZ>>,
                         op_expr_less<SZ>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpString<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SU2, shared_ptr<SparseMatrixInfo<SU2>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic<SU2>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SU2>>>);
PYBIND11_MAKE_OPAQUE(map<shared_ptr<OpExpr<SU2>>, shared_ptr<SparseMatrix<SU2>>,
                         op_expr_less<SU2>>);

template <typename T> struct Array {
    T *data;
    size_t n;
    Array(T *data, size_t n) : data(data), n(n) {}
    T &operator[](size_t idx) { return data[idx]; }
};

template <typename T, typename... Args>
size_t printable(const T &p, const Args &...) {
    return (size_t)&p;
}

template <typename T>
auto printable(const T &p)
    -> decltype(declval<ostream &>() << declval<T>(), T()) {
    return p;
}

template <typename T>
py::class_<Array<T>, shared_ptr<Array<T>>> bind_array(py::module &m,
                                                      const char *name) {
    return py::class_<Array<T>, shared_ptr<Array<T>>>(m, name)
        .def(
            "__setitem__",
            [](Array<T> *self, size_t i, const T &t) {
                self->operator[](i) = t;
            },
            py::keep_alive<1, 3>())
        .def("__getitem__",
             [](Array<T> *self, size_t i) { return self->operator[](i); })
        .def("__len__", [](Array<T> *self) { return self->n; })
        .def("__repr__",
             [name](Array<T> *self) {
                 stringstream ss;
                 ss << name << "(LEN=" << self->n << ")[";
                 for (size_t i = 0; i < self->n; i++)
                     ss << printable<T>(self->data[i]) << ",";
                 ss << "]";
                 return ss.str();
             })
        .def(
            "__iter__",
            [](Array<T> *self) {
                return py::make_iterator<
                    py::return_value_policy::reference_internal, T *, T *, T &>(
                    self->data, self->data + self->n);
            },
            py::keep_alive<0, 1>());
}

template <typename S>
auto bind_cg(py::module &m) -> decltype(typename S::is_sz_t()) {
    py::class_<CG<S>, shared_ptr<CG<S>>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def("initialize", &CG<S>::initialize)
        .def("deallocate", &CG<S>::deallocate)
        .def("wigner_6j", &CG<S>::wigner_6j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"))
        .def("wigner_9j", &CG<S>::wigner_9j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"),
             py::arg("tjg"), py::arg("tjh"), py::arg("tji"))
        .def("racah", &CG<S>::racah, py::arg("ta"), py::arg("tb"),
             py::arg("tc"), py::arg("td"), py::arg("te"), py::arg("tf"))
        .def("transpose_cg", &CG<S>::transpose_cg, py::arg("td"), py::arg("tl"),
             py::arg("tr"));
}

template <typename S>
auto bind_cg(py::module &m) -> decltype(typename S::is_su2_t()) {
    py::class_<CG<S>, shared_ptr<CG<S>>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def("initialize", &CG<S>::initialize)
        .def("deallocate", &CG<S>::deallocate)
        .def_static("triangle", &CG<S>::triangle, py::arg("tja"),
                    py::arg("tjb"), py::arg("tjc"))
        .def("sqrt_delta", &CG<S>::sqrt_delta, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"))
        .def("cg", &CG<S>::cg, py::arg("tja"), py::arg("tjb"), py::arg("tjc"),
             py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_3j", &CG<S>::wigner_3j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &CG<S>::wigner_6j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"))
        .def("wigner_9j", &CG<S>::wigner_9j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"),
             py::arg("tjg"), py::arg("tjh"), py::arg("tji"))
        .def("racah", &CG<S>::racah, py::arg("ta"), py::arg("tb"),
             py::arg("tc"), py::arg("td"), py::arg("te"), py::arg("tf"))
        .def("transpose_cg", &CG<S>::transpose_cg, py::arg("td"), py::arg("tl"),
             py::arg("tr"));
}

template <typename S> void bind_class(py::module &m, const string &name) {

    bind_array<StateInfo<S>>(m, "ArrayStateInfo");
    bind_array<S>(m, ("Array" + name).c_str());
    bind_array<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
        m, "ArrayVectorPLMatInfo");

    py::class_<OpExpr<S>, shared_ptr<OpExpr<S>>>(m, "OpExpr")
        .def(py::init<>())
        .def("get_type", &OpExpr<S>::get_type)
        .def(py::self == py::self)
        .def("__repr__", &to_str<S>);

    py::class_<OpElement<S>, shared_ptr<OpElement<S>>, OpExpr<S>>(m,
                                                                  "OpElement")
        .def(py::init<OpNames, SiteIndex, S>())
        .def(py::init<OpNames, SiteIndex, S, double>())
        .def_readwrite("name", &OpElement<S>::name)
        .def_readwrite("site_index", &OpElement<S>::site_index)
        .def_readwrite("factor", &OpElement<S>::factor)
        .def_readwrite("q_label", &OpElement<S>::q_label)
        .def("abs", &OpElement<S>::abs)
        .def("__mul__", &OpElement<S>::operator*)
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def("__hash__", &OpElement<S>::hash);

    py::class_<OpElementRef<S>, shared_ptr<OpElementRef<S>>, OpExpr<S>>(
        m, "OpElementRef")
        .def(py::init<const shared_ptr<OpElement<S>> &, int8_t, int8_t>())
        .def_readwrite("op", &OpElementRef<S>::op)
        .def_readwrite("factor", &OpElementRef<S>::factor)
        .def_readwrite("trans", &OpElementRef<S>::trans);

    py::class_<OpString<S>, shared_ptr<OpString<S>>, OpExpr<S>>(m, "OpString")
        .def(py::init<const shared_ptr<OpElement<S>> &, double>())
        .def(py::init<const shared_ptr<OpElement<S>> &, double, uint8_t>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const shared_ptr<OpElement<S>> &, double>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const shared_ptr<OpElement<S>> &, double, uint8_t>())
        .def_readwrite("factor", &OpString<S>::factor)
        .def_readwrite("conj", &OpString<S>::conj)
        .def_readwrite("a", &OpString<S>::a)
        .def_readwrite("b", &OpString<S>::b);

    py::class_<OpSumProd<S>, shared_ptr<OpSumProd<S>>, OpString<S>>(m,
                                                                    "OpSumProd")
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const vector<shared_ptr<OpElement<S>>> &,
                      const vector<bool> &, double, uint8_t>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const vector<shared_ptr<OpElement<S>>> &,
                      const vector<bool> &, double>())
        .def(py::init<const vector<shared_ptr<OpElement<S>>> &,
                      const shared_ptr<OpElement<S>> &, const vector<bool> &,
                      double, uint8_t>())
        .def(py::init<const vector<shared_ptr<OpElement<S>>> &,
                      const shared_ptr<OpElement<S>> &, const vector<bool> &,
                      double>())
        .def_readwrite("ops", &OpSumProd<S>::ops)
        .def_readwrite("conjs", &OpSumProd<S>::conjs);

    py::class_<OpSum<S>, shared_ptr<OpSum<S>>, OpExpr<S>>(m, "OpSum")
        .def(py::init<const vector<shared_ptr<OpString<S>>> &>())
        .def_readwrite("strings", &OpSum<S>::strings);

    py::bind_vector<vector<shared_ptr<OpExpr<S>>>>(m, "VectorOpExpr");
    py::bind_vector<vector<shared_ptr<OpElement<S>>>>(m, "VectorOpElement");
    py::bind_vector<vector<shared_ptr<OpString<S>>>>(m, "VectorOpString");

    py::class_<Symbolic<S>, shared_ptr<Symbolic<S>>>(m, "Symbolic")
        .def_readwrite("m", &Symbolic<S>::m)
        .def_readwrite("n", &Symbolic<S>::n)
        .def_readwrite("data", &Symbolic<S>::data)
        .def("get_type", [](Symbolic<S> *self) { return self->get_type(); })
        .def("__matmul__",
             [](const shared_ptr<Symbolic<S>> &self,
                const shared_ptr<Symbolic<S>> &other) { return self * other; });

    py::class_<StateInfo<S>, shared_ptr<StateInfo<S>>>(m, "StateInfo")
        .def(py::init<>())
        .def(py::init<S>())
        .def_readwrite("n", &StateInfo<S>::n)
        .def_readwrite("n_states_total", &StateInfo<S>::n_states_total)
        .def_property_readonly(
            "quanta",
            [](StateInfo<S> *self) { return Array<S>(self->quanta, self->n); })
        .def_property_readonly("n_states",
                               [](StateInfo<S> *self) {
                                   return Array<uint16_t>(self->n_states,
                                                          self->n);
                               })
        .def("allocate",
             [](StateInfo<S> *self, int length) { self->allocate(length); })
        .def("deallocate", &StateInfo<S>::deallocate)
        .def("sort_states", &StateInfo<S>::sort_states)
        .def("copy_data_to", &StateInfo<S>::copy_data_to)
        .def("deep_copy", &StateInfo<S>::deep_copy)
        .def("collect", &StateInfo<S>::collect,
             py::arg("target") = S(0x7FFFFFFF))
        .def("find_state", &StateInfo<S>::find_state)
        .def_static("tensor_product_ref",
                    (StateInfo<S>(*)(const StateInfo<S> &, const StateInfo<S> &,
                                     const StateInfo<S> &)) &
                        StateInfo<S>::tensor_product)
        .def_static(
            "tensor_product",
            (StateInfo<S>(*)(const StateInfo<S> &, const StateInfo<S> &, S)) &
                StateInfo<S>::tensor_product)
        .def_static("get_connection_info", &StateInfo<S>::get_connection_info)
        .def_static("filter", &StateInfo<S>::filter)
        .def("__repr__", [](StateInfo<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<typename SparseMatrixInfo<S>::ConnectionInfo,
               shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>(
        m, "ConnectionInfo")
        .def(py::init<>())
        .def_property_readonly(
            "n",
            [](typename SparseMatrixInfo<S>::ConnectionInfo *self) {
                return self->n[4];
            })
        .def_readwrite("nc", &SparseMatrixInfo<S>::ConnectionInfo::nc)
        .def("deallocate", &SparseMatrixInfo<S>::ConnectionInfo::deallocate)
        .def("__repr__",
             [](typename SparseMatrixInfo<S>::ConnectionInfo *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             });

    py::class_<SparseMatrixInfo<S>, shared_ptr<SparseMatrixInfo<S>>>(
        m, "SparseMatrixInfo")
        .def(py::init<>())
        .def_readwrite("delta_quantum", &SparseMatrixInfo<S>::delta_quantum)
        .def_readwrite("is_fermion", &SparseMatrixInfo<S>::is_fermion)
        .def_readwrite("is_wavefunction", &SparseMatrixInfo<S>::is_wavefunction)
        .def_readwrite("n", &SparseMatrixInfo<S>::n)
        .def_readwrite("cinfo", &SparseMatrixInfo<S>::cinfo)
        .def_property_readonly("quanta",
                               [](SparseMatrixInfo<S> *self) {
                                   return Array<S>(self->quanta, self->n);
                               })
        .def_property_readonly("n_states_total",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<uint32_t>(
                                       self->n, self->n_states_total);
                               })
        .def_property_readonly("n_states_bra",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<uint16_t>(
                                       self->n, self->n_states_bra);
                               })
        .def_property_readonly("n_states_ket",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<uint16_t>(
                                       self->n, self->n_states_ket);
                               })
        .def("initialize", &SparseMatrixInfo<S>::initialize, py::arg("bra"),
             py::arg("ket"), py::arg("dq"), py::arg("is_fermion"),
             py::arg("wfn") = false)
        .def("find_state", &SparseMatrixInfo<S>::find_state, py::arg("q"),
             py::arg("start") = 0)
        .def_property_readonly("total_memory",
                               &SparseMatrixInfo<S>::get_total_memory)
        .def("allocate", &SparseMatrixInfo<S>::allocate, py::arg("length"),
             py::arg("ptr") = nullptr)
        .def("deallocate", &SparseMatrixInfo<S>::deallocate)
        .def("reallocate", &SparseMatrixInfo<S>::reallocate, py::arg("length"))
        .def("__repr__", [](SparseMatrixInfo<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<SparseMatrix<S>, shared_ptr<SparseMatrix<S>>>(m, "SparseMatrix")
        .def(py::init<>())
        .def_readwrite("info", &SparseMatrix<S>::info)
        .def_readwrite("factor", &SparseMatrix<S>::factor)
        .def_readwrite("total_memory", &SparseMatrix<S>::total_memory)
        .def_property(
            "data",
            [](SparseMatrix<S> *self) {
                return py::array_t<double>(self->total_memory, self->data);
            },
            [](SparseMatrix<S> *self, const py::array_t<double> &v) {
                assert(v.size() == self->total_memory);
                memcpy(self->data, v.data(),
                       sizeof(double) * self->total_memory);
            })
        .def("clear", &SparseMatrix<S>::clear)
        .def("load_data", &SparseMatrix<S>::load_data, py::arg("filename"),
             py::arg("load_info") = false)
        .def("save_data", &SparseMatrix<S>::save_data, py::arg("filename"),
             py::arg("save_info") = false)
        .def("copy_data_from", &SparseMatrix<S>::copy_data_from)
        .def("allocate",
             [](SparseMatrix<S> *self,
                const shared_ptr<SparseMatrixInfo<S>> &info) {
                 self->allocate(info);
             })
        .def("deallocate", &SparseMatrix<S>::deallocate)
        .def("reallocate", &SparseMatrix<S>::reallocate, py::arg("length"))
        .def("trace", &SparseMatrix<S>::trace)
        .def("norm", &SparseMatrix<S>::norm)
        .def("__getitem__",
             [](SparseMatrix<S> *self, int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SparseMatrix<S> *self, int idx, const py::array_t<double> &v) {
                 assert(v.size() == (*self)[idx].size());
                 memcpy((*self)[idx].data, v.data(), sizeof(double) * v.size());
             })
        .def("left_canonicalize", &SparseMatrix<S>::left_canonicalize,
             py::arg("rmat"))
        .def("right_canonicalize", &SparseMatrix<S>::right_canonicalize,
             py::arg("lmat"))
        .def("left_multiply", &SparseMatrix<S>::left_multiply, py::arg("lmat"),
             py::arg("l"), py::arg("m"), py::arg("r"), py::arg("lm"),
             py::arg("lm_cinfo"))
        .def("right_multiply", &SparseMatrix<S>::right_multiply,
             py::arg("rmat"), py::arg("l"), py::arg("m"), py::arg("r"),
             py::arg("mr"), py::arg("mr_cinfo"))
        .def("randomize", &SparseMatrix<S>::randomize, py::arg("a") = 0.0,
             py::arg("b") = 1.0)
        .def("contract", &SparseMatrix<S>::contract)
        .def("swap_to_fused_left", &SparseMatrix<S>::swap_to_fused_left)
        .def("swap_to_fused_right", &SparseMatrix<S>::swap_to_fused_right)
        .def("__repr__", [](SparseMatrix<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
        m, "VectorPLMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrixInfo<S>>>>(m,
                                                             "VectorSpMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrix<S>>>>(m, "VectorSpMat");
    py::bind_map<map<OpNames, shared_ptr<SparseMatrix<S>>>>(m,
                                                            "MapOpNamesSpMat");
    py::bind_map<map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                     op_expr_less<S>>>(m, "MapOpExprSpMat");

    py::class_<MPSInfo<S>, shared_ptr<MPSInfo<S>>>(m, "MPSInfo")
        .def_readwrite("n_sites", &MPSInfo<S>::n_sites)
        .def_readwrite("vaccum", &MPSInfo<S>::vaccum)
        .def_readwrite("target", &MPSInfo<S>::target)
        .def_readwrite("orbsym", &MPSInfo<S>::orbsym)
        .def_readwrite("n_syms", &MPSInfo<S>::n_syms)
        .def_readwrite("bond_dim", &MPSInfo<S>::bond_dim)
        .def_readwrite("tag", &MPSInfo<S>::tag)
        .def(py::init([](int n_sites, S vaccum, S target,
                         Array<StateInfo<S>> &basis,
                         const vector<uint8_t> &orbsym, uint8_t n_syms) {
            return make_shared<MPSInfo<S>>(n_sites, vaccum, target, basis.data,
                                           orbsym, n_syms);
        }))
        .def_property_readonly("basis",
                               [](MPSInfo<S> *self) {
                                   Array<StateInfo<S>>(self->basis,
                                                       self->orbsym.empty()
                                                           ? self->n_sites
                                                           : self->n_syms);
                               })
        .def_property_readonly("left_dims_fci",
                               [](MPSInfo<S> *self) {
                                   return Array<StateInfo<S>>(
                                       self->left_dims_fci, self->n_sites + 1);
                               })
        .def_property_readonly("right_dims_fci",
                               [](MPSInfo<S> *self) {
                                   return Array<StateInfo<S>>(
                                       self->right_dims_fci, self->n_sites + 1);
                               })
        .def_property_readonly("left_dims",
                               [](MPSInfo<S> *self) {
                                   return Array<StateInfo<S>>(
                                       self->left_dims, self->n_sites + 1);
                               })
        .def_property_readonly("right_dims",
                               [](MPSInfo<S> *self) {
                                   return Array<StateInfo<S>>(
                                       self->right_dims, self->n_sites + 1);
                               })
        .def("get_basis", &MPSInfo<S>::get_basis)
        .def("get_ancilla_type", &MPSInfo<S>::get_ancilla_type)
        .def("set_bond_dimension_using_occ",
             &MPSInfo<S>::set_bond_dimension_using_occ, py::arg("m"),
             py::arg("occ"), py::arg("bias") = 1.0)
        .def("set_bond_dimension_using_hf",
             &MPSInfo<S>::set_bond_dimension_using_hf, py::arg("m"),
             py::arg("occ"), py::arg("n_local") = 0)
        .def("set_bond_dimension", &MPSInfo<S>::set_bond_dimension)
        .def("get_filename", &MPSInfo<S>::get_filename)
        .def("save_mutable", &MPSInfo<S>::save_mutable)
        .def("deallocate_mutable", &MPSInfo<S>::deallocate_mutable)
        .def("load_mutable", &MPSInfo<S>::load_mutable)
        .def("save_left_dims", &MPSInfo<S>::save_left_dims)
        .def("save_right_dims", &MPSInfo<S>::save_right_dims)
        .def("load_left_dims", &MPSInfo<S>::load_left_dims)
        .def("load_right_dims", &MPSInfo<S>::load_right_dims)
        .def("deallocate", &MPSInfo<S>::deallocate);

    py::class_<CASCIMPSInfo<S>, shared_ptr<CASCIMPSInfo<S>>, MPSInfo<S>>(
        m, "CASCIMPSInfo")
        .def_readwrite("casci_mask", &CASCIMPSInfo<S>::casci_mask)
        .def(py::init([](int n_sites, S vaccum, S target,
                         Array<StateInfo<S>> &basis,
                         const vector<uint8_t> &orbsym, uint8_t n_syms,
                         const vector<ActiveTypes> &casci_mask) {
            return make_shared<CASCIMPSInfo<S>>(n_sites, vaccum, target,
                                                basis.data, orbsym, n_syms,
                                                casci_mask);
        }))
        .def(py::init([](int n_sites, S vaccum, S target,
                         Array<StateInfo<S>> &basis,
                         const vector<uint8_t> &orbsym, uint8_t n_syms,
                         int n_active_sites, int n_active_electrons) {
            return make_shared<CASCIMPSInfo<S>>(
                n_sites, vaccum, target, basis.data, orbsym, n_syms,
                n_active_sites, n_active_electrons);
        }));

    py::class_<AncillaMPSInfo<S>, shared_ptr<AncillaMPSInfo<S>>, MPSInfo<S>>(
        m, "AncillaMPSInfo")
        .def_readwrite("n_physical_sites", &AncillaMPSInfo<S>::n_physical_sites)
        .def(py::init([](int n_sites, S vaccum, S target,
                         Array<StateInfo<S>> &basis,
                         const vector<uint8_t> &orbsym, uint8_t n_syms) {
            return make_shared<AncillaMPSInfo<S>>(n_sites, vaccum, target,
                                                  basis.data, orbsym, n_syms);
        }))
        .def_static("trans_orbsym", &AncillaMPSInfo<S>::trans_orbsym)
        .def("set_thermal_limit", &AncillaMPSInfo<S>::set_thermal_limit);

    py::class_<MPS<S>, shared_ptr<MPS<S>>>(m, "MPS")
        .def(py::init<const shared_ptr<MPSInfo<S>> &>())
        .def(py::init<int, int, int>())
        .def_readwrite("n_sites", &MPS<S>::n_sites)
        .def_readwrite("center", &MPS<S>::center)
        .def_readwrite("dot", &MPS<S>::dot)
        .def_readwrite("info", &MPS<S>::info)
        .def_readwrite("tensors", &MPS<S>::tensors)
        .def_readwrite("canonical_form", &MPS<S>::canonical_form)
        .def("initialize", &MPS<S>::initialize)
        .def("fill_thermal_limit", &MPS<S>::fill_thermal_limit)
        .def("canonicalize", &MPS<S>::canonicalize)
        .def("random_canonicalize", &MPS<S>::random_canonicalize)
        .def("get_filename", &MPS<S>::get_filename)
        .def("load_data", &MPS<S>::load_data)
        .def("save_data", &MPS<S>::save_data)
        .def("load_mutable", &MPS<S>::load_mutable)
        .def("save_mutable", &MPS<S>::save_mutable)
        .def("save_tensor", &MPS<S>::save_tensor)
        .def("load_tensor", &MPS<S>::load_tensor)
        .def("unload_tensor", &MPS<S>::unload_tensor)
        .def("deallocate", &MPS<S>::deallocate);

    bind_cg<S>(m);

    py::class_<OperatorFunctions<S>, shared_ptr<OperatorFunctions<S>>>(
        m, "OperatorFunctions")
        .def_readwrite("cg", &OperatorFunctions<S>::cg)
        .def_readwrite("seq", &OperatorFunctions<S>::seq)
        .def(py::init<const shared_ptr<CG<S>> &>())
        .def("iadd", &OperatorFunctions<S>::iadd, py::arg("a"), py::arg("b"),
             py::arg("scale") = 1.0, py::arg("conj") = false)
        .def("tensor_rotate", &OperatorFunctions<S>::tensor_rotate,
             py::arg("a"), py::arg("c"), py::arg("rot_bra"), py::arg("rot_ket"),
             py::arg("trans"), py::arg("scale") = 1.0)
        .def("tensor_product_diagonal",
             &OperatorFunctions<S>::tensor_product_diagonal, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("opdq"),
             py::arg("scale") = 1.0)
        .def("tensor_product_multiply",
             &OperatorFunctions<S>::tensor_product_multiply, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("v"),
             py::arg("opdq"), py::arg("scale") = 1.0)
        .def("tensor_product", &OperatorFunctions<S>::tensor_product,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("scale") = 1.0)
        .def("product", &OperatorFunctions<S>::product, py::arg("a"),
             py::arg("b"), py::arg("c"), py::arg("scale") = 1.0)
        .def_static("trans_product", &OperatorFunctions<S>::trans_product,
                    py::arg("a"), py::arg("b"), py::arg("trace_right"),
                    py::arg("noise") = 0.0,
                    py::arg("noise_type") = NoiseTypes::DensityMatrix);

    py::class_<OperatorTensor<S>, shared_ptr<OperatorTensor<S>>>(
        m, "OperatorTensor")
        .def(py::init<>())
        .def_readwrite("lmat", &OperatorTensor<S>::lmat)
        .def_readwrite("rmat", &OperatorTensor<S>::rmat)
        .def_readwrite("ops", &OperatorTensor<S>::ops)
        .def("reallocate", &OperatorTensor<S>::reallocate, py::arg("clean"))
        .def("deallocate", &OperatorTensor<S>::deallocate);

    py::class_<DelayedOperatorTensor<S>, shared_ptr<DelayedOperatorTensor<S>>>(
        m, "DelayedOperatorTensor")
        .def(py::init<>())
        .def_readwrite("ops", &DelayedOperatorTensor<S>::ops)
        .def_readwrite("mat", &DelayedOperatorTensor<S>::mat)
        .def_readwrite("lops", &DelayedOperatorTensor<S>::lops)
        .def_readwrite("rops", &DelayedOperatorTensor<S>::rops)
        .def("reallocate", &DelayedOperatorTensor<S>::reallocate,
             py::arg("clean"))
        .def("deallocate", &DelayedOperatorTensor<S>::deallocate);

    py::bind_vector<vector<shared_ptr<OperatorTensor<S>>>>(m, "VectorOpTensor");

    py::class_<TensorFunctions<S>, shared_ptr<TensorFunctions<S>>>(
        m, "TensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S>> &>())
        .def_readwrite("opf", &TensorFunctions<S>::opf)
        .def_static("left_assign", &TensorFunctions<S>::left_assign,
                    py::arg("a"), py::arg("c"))
        .def_static("right_assign", &TensorFunctions<S>::right_assign,
                    py::arg("a"), py::arg("c"))
        .def("left_contract", &TensorFunctions<S>::left_contract, py::arg("a"),
             py::arg("b"), py::arg("c"), py::arg("cexprs") = nullptr)
        .def("right_contract", &TensorFunctions<S>::right_contract,
             py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("cexprs") = nullptr)
        .def("tensor_product_multiply",
             &TensorFunctions<S>::tensor_product_multiply)
        .def("tensor_product_diagonal",
             &TensorFunctions<S>::tensor_product_diagonal)
        .def("tensor_product", &TensorFunctions<S>::tensor_product)
        .def("left_rotate", &TensorFunctions<S>::left_rotate)
        .def("right_rotate", &TensorFunctions<S>::right_rotate)
        .def("numerical_transform", &TensorFunctions<S>::numerical_transform)
        .def("delayed_contract", (shared_ptr<DelayedOperatorTensor<S>>(*)(
                                     const shared_ptr<OperatorTensor<S>> &,
                                     const shared_ptr<OperatorTensor<S>> &,
                                     const shared_ptr<OpExpr<S>> &)) &
                                     TensorFunctions<S>::delayed_contract)
        .def("delayed_contract_simplified",
             (shared_ptr<DelayedOperatorTensor<S>>(*)(
                 const shared_ptr<OperatorTensor<S>> &,
                 const shared_ptr<OperatorTensor<S>> &,
                 const shared_ptr<Symbolic<S>> &,
                 const shared_ptr<Symbolic<S>> &)) &
                 TensorFunctions<S>::delayed_contract);

    py::class_<Partition<S>, shared_ptr<Partition<S>>>(m, "Partition")
        .def(py::init<const shared_ptr<OperatorTensor<S>> &,
                      const shared_ptr<OperatorTensor<S>> &,
                      const shared_ptr<OperatorTensor<S>> &>())
        .def(py::init<const shared_ptr<OperatorTensor<S>> &,
                      const shared_ptr<OperatorTensor<S>> &,
                      const shared_ptr<OperatorTensor<S>> &,
                      const shared_ptr<OperatorTensor<S>> &>())
        .def_readwrite("left", &Partition<S>::left)
        .def_readwrite("right", &Partition<S>::right)
        .def_readwrite("middle", &Partition<S>::middle)
        .def_readwrite("left_op_infos", &Partition<S>::left_op_infos)
        .def_readwrite("right_op_infos", &Partition<S>::right_op_infos)
        .def_static("find_op_info", &Partition<S>::find_op_info)
        .def_static("build_left", &Partition<S>::build_left)
        .def_static("build_right", &Partition<S>::build_right)
        .def_static("get_uniq_labels", &Partition<S>::get_uniq_labels)
        .def_static("get_uniq_sub_labels", &Partition<S>::get_uniq_sub_labels)
        .def_static("deallocate_op_infos_notrunc",
                    &Partition<S>::deallocate_op_infos_notrunc)
        .def_static("copy_op_infos", &Partition<S>::copy_op_infos)
        .def_static("init_left_op_infos", &Partition<S>::init_left_op_infos)
        .def_static("init_left_op_infos_notrunc",
                    &Partition<S>::init_left_op_infos_notrunc)
        .def_static("init_right_op_infos", &Partition<S>::init_right_op_infos)
        .def_static("init_right_op_infos_notrunc",
                    &Partition<S>::init_right_op_infos_notrunc);

    py::bind_vector<vector<shared_ptr<Partition<S>>>>(m, "VectorPartition");

    py::class_<EffectiveHamiltonian<S>, shared_ptr<EffectiveHamiltonian<S>>>(
        m, "EffectiveHamiltonian")
        .def(py::init<const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const shared_ptr<DelayedOperatorTensor<S>> &,
                      const shared_ptr<SparseMatrix<S>> &,
                      const shared_ptr<SparseMatrix<S>> &,
                      const shared_ptr<OpElement<S>> &,
                      const shared_ptr<SymbolicColumnVector<S>> &,
                      const shared_ptr<TensorFunctions<S>> &, bool>())
        .def_readwrite("left_op_infos", &EffectiveHamiltonian<S>::left_op_infos)
        .def_readwrite("right_op_infos",
                       &EffectiveHamiltonian<S>::right_op_infos)
        .def_readwrite("op", &EffectiveHamiltonian<S>::op)
        .def_readwrite("bra", &EffectiveHamiltonian<S>::bra)
        .def_readwrite("ket", &EffectiveHamiltonian<S>::ket)
        .def_readwrite("diag", &EffectiveHamiltonian<S>::diag)
        .def_readwrite("cmat", &EffectiveHamiltonian<S>::cmat)
        .def_readwrite("vmat", &EffectiveHamiltonian<S>::vmat)
        .def_readwrite("tf", &EffectiveHamiltonian<S>::tf)
        .def_readwrite("opdq", &EffectiveHamiltonian<S>::opdq)
        .def_readwrite("compute_diag", &EffectiveHamiltonian<S>::compute_diag)
        .def("__call__", &EffectiveHamiltonian<S>::operator(), py::arg("b"),
             py::arg("c"), py::arg("idx") = 0, py::arg("factor") = 1.0)
        .def("eigs", &EffectiveHamiltonian<S>::eigs)
        .def("multiply", &EffectiveHamiltonian<S>::multiply)
        .def("expect", &EffectiveHamiltonian<S>::expect)
        .def("rk4_apply", &EffectiveHamiltonian<S>::rk4_apply, py::arg("beta"),
             py::arg("const_e"), py::arg("eval_energy") = false)
        .def("expo_apply", &EffectiveHamiltonian<S>::expo_apply,
             py::arg("beta"), py::arg("const_e"), py::arg("iprint") = false)
        .def("deallocate", &EffectiveHamiltonian<S>::deallocate);

    py::class_<MovingEnvironment<S>, shared_ptr<MovingEnvironment<S>>>(
        m, "MovingEnvironment")
        .def(py::init<const shared_ptr<MPO<S>> &, const shared_ptr<MPS<S>> &,
                      const shared_ptr<MPS<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &, const shared_ptr<MPS<S>> &,
                      const shared_ptr<MPS<S>> &, const string &>())
        .def_readwrite("n_sites", &MovingEnvironment<S>::n_sites)
        .def_readwrite("center", &MovingEnvironment<S>::center)
        .def_readwrite("dot", &MovingEnvironment<S>::dot)
        .def_readwrite("mpo", &MovingEnvironment<S>::mpo)
        .def_readwrite("bra", &MovingEnvironment<S>::bra)
        .def_readwrite("ket", &MovingEnvironment<S>::ket)
        .def_readwrite("envs", &MovingEnvironment<S>::envs)
        .def_readwrite("tag", &MovingEnvironment<S>::tag)
        .def("left_contract_rotate",
             &MovingEnvironment<S>::left_contract_rotate)
        .def("right_contract_rotate",
             &MovingEnvironment<S>::right_contract_rotate)
        .def("init_environments", &MovingEnvironment<S>::init_environments,
             py::arg("iprint") = false, py::arg("init_left") = true,
             py::arg("init_right") = true)
        .def("prepare", &MovingEnvironment<S>::prepare)
        .def("move_to", &MovingEnvironment<S>::move_to)
        .def("eff_ham", &MovingEnvironment<S>::eff_ham, py::arg("fuse_type"),
             py::arg("compute_diag"))
        .def("get_left_partition_filename",
             &MovingEnvironment<S>::get_left_partition_filename)
        .def("get_right_partition_filename",
             &MovingEnvironment<S>::get_right_partition_filename)
        .def_static("contract_two_dot", &MovingEnvironment<S>::contract_two_dot,
                    py::arg("i"), py::arg("mps"), py::arg("reduced") = false)
        .def_static("density_matrix", &MovingEnvironment<S>::density_matrix,
                    py::arg("opdq"), py::arg("psi"), py::arg("trace_right"),
                    py::arg("noise"), py::arg("noise_type"))
        .def_static("density_matrix_with_weights",
                    &MovingEnvironment<S>::density_matrix_with_weights,
                    py::arg("opdq"), py::arg("psi"), py::arg("trace_right"),
                    py::arg("noise"), py::arg("mats"), py::arg("weights"),
                    py::arg("noise_type"))
        .def_static(
            "split_density_matrix",
            [](const shared_ptr<SparseMatrix<S>> &dm,
               const shared_ptr<SparseMatrix<S>> &wfn, int k, bool trace_right,
               bool normalize, double cutoff) {
                shared_ptr<SparseMatrix<S>> left = nullptr, right = nullptr;
                double error = MovingEnvironment<S>::split_density_matrix(
                    dm, wfn, k, trace_right, normalize, left, right, cutoff);
                return make_tuple(error, left, right);
            })
        .def_static("propagate_wfn", &MovingEnvironment<S>::propagate_wfn,
                    py::arg("i"), py::arg("n_sites"), py::arg("mps"),
                    py::arg("forward"));

    py::class_<Hamiltonian<S>, shared_ptr<Hamiltonian<S>>>(m, "Hamiltonian")
        .def(py::init<S, S, int, const vector<uint8_t> &>())
        .def_readwrite("n_syms", &Hamiltonian<S>::n_syms)
        .def_readwrite("opf", &Hamiltonian<S>::opf)
        .def_readwrite("n_sites", &Hamiltonian<S>::n_sites)
        .def_readwrite("orb_sym", &Hamiltonian<S>::orb_sym)
        .def_readwrite("vaccum", &Hamiltonian<S>::vaccum)
        .def_readwrite("target", &Hamiltonian<S>::target)
        .def_property_readonly("basis",
                               [](Hamiltonian<S> *self) {
                                   return Array<StateInfo<S>>(self->basis,
                                                              self->n_syms);
                               })
        .def_property_readonly(
            "site_op_infos",
            [](Hamiltonian<S> *self) {
                return Array<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
                    self->site_op_infos, self->n_syms);
            })
        .def("get_site_ops", &Hamiltonian<S>::get_site_ops)
        .def("filter_site_ops", &Hamiltonian<S>::filter_site_ops)
        .def("find_site_op_info", &Hamiltonian<S>::find_site_op_info)
        .def("find_site_norm_op", &Hamiltonian<S>::find_site_norm_op)
        .def("deallocate", &Hamiltonian<S>::deallocate);

    py::class_<HamiltonianQC<S>, shared_ptr<HamiltonianQC<S>>, Hamiltonian<S>>(
        m, "HamiltonianQC")
        .def(py::init<S, S, int, const vector<uint8_t> &,
                      const shared_ptr<FCIDUMP> &>())
        .def_readwrite("fcidump", &HamiltonianQC<S>::fcidump)
        .def_readwrite("mu", &HamiltonianQC<S>::mu)
        .def("op_prims",
             [](HamiltonianQC<S> *self, int idx) { self->op_prims[idx]; })
        .def("v", &HamiltonianQC<S>::v)
        .def("t", &HamiltonianQC<S>::t)
        .def("e", &HamiltonianQC<S>::e)
        .def("init_site_ops", &HamiltonianQC<S>::init_site_ops)
        .def("get_site_ops", &HamiltonianQC<S>::get_site_ops);

    py::class_<typename DMRG<S>::Iteration,
               shared_ptr<typename DMRG<S>::Iteration>>(m, "DMRGIteration")
        .def(py::init<double, double, int, size_t, double>())
        .def(py::init<double, double, int>())
        .def_readwrite("energy", &DMRG<S>::Iteration::energy)
        .def_readwrite("error", &DMRG<S>::Iteration::error)
        .def_readwrite("ndav", &DMRG<S>::Iteration::ndav)
        .def_readwrite("tdav", &DMRG<S>::Iteration::tdav)
        .def_readwrite("nflop", &DMRG<S>::Iteration::nflop)
        .def("__repr__", [](typename DMRG<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<DMRG<S>, shared_ptr<DMRG<S>>>(m, "DMRG")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<uint16_t> &, const vector<double> &>())
        .def_readwrite("iprint", &DMRG<S>::iprint)
        .def_readwrite("cutoff", &DMRG<S>::cutoff)
        .def_readwrite("me", &DMRG<S>::me)
        .def_readwrite("bond_dims", &DMRG<S>::bond_dims)
        .def_readwrite("noises", &DMRG<S>::noises)
        .def_readwrite("energies", &DMRG<S>::energies)
        .def_readwrite("forward", &DMRG<S>::forward)
        .def_readwrite("noise_type", &DMRG<S>::noise_type)
        .def_readwrite("trunc_type", &DMRG<S>::trunc_type)
        .def("update_two_dot", &DMRG<S>::update_two_dot)
        .def("blocking", &DMRG<S>::blocking)
        .def("sweep", &DMRG<S>::sweep)
        .def("solve", &DMRG<S>::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<typename ImaginaryTE<S>::Iteration,
               shared_ptr<typename ImaginaryTE<S>::Iteration>>(
        m, "ImaginaryTEIteration")
        .def(py::init<double, double, double, int, int, size_t, double>())
        .def(py::init<double, double, double, int, int>())
        .def_readwrite("energy", &ImaginaryTE<S>::Iteration::energy)
        .def_readwrite("normsq", &ImaginaryTE<S>::Iteration::normsq)
        .def_readwrite("error", &ImaginaryTE<S>::Iteration::error)
        .def_readwrite("nexpo", &ImaginaryTE<S>::Iteration::nexpo)
        .def_readwrite("nexpok", &ImaginaryTE<S>::Iteration::nexpok)
        .def_readwrite("texpo", &ImaginaryTE<S>::Iteration::texpo)
        .def_readwrite("nflop", &ImaginaryTE<S>::Iteration::nflop)
        .def("__repr__", [](typename ImaginaryTE<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<ImaginaryTE<S>, shared_ptr<ImaginaryTE<S>>>(m, "ImaginaryTE")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<uint16_t> &, TETypes>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<uint16_t> &, TETypes, int>())
        .def_readwrite("iprint", &ImaginaryTE<S>::iprint)
        .def_readwrite("cutoff", &ImaginaryTE<S>::cutoff)
        .def_readwrite("me", &ImaginaryTE<S>::me)
        .def_readwrite("bond_dims", &ImaginaryTE<S>::bond_dims)
        .def_readwrite("noises", &ImaginaryTE<S>::noises)
        .def_readwrite("energies", &ImaginaryTE<S>::energies)
        .def_readwrite("normsqs", &ImaginaryTE<S>::normsqs)
        .def_readwrite("forward", &ImaginaryTE<S>::forward)
        .def_readwrite("n_sub_sweeps", &ImaginaryTE<S>::n_sub_sweeps)
        .def_readwrite("weights", &ImaginaryTE<S>::weights)
        .def_readwrite("mode", &ImaginaryTE<S>::mode)
        .def_readwrite("noise_type", &ImaginaryTE<S>::noise_type)
        .def_readwrite("trunc_type", &ImaginaryTE<S>::trunc_type)
        .def("update_two_dot", &ImaginaryTE<S>::update_two_dot)
        .def("blocking", &ImaginaryTE<S>::blocking)
        .def("sweep", &ImaginaryTE<S>::sweep)
        .def("normalize", &ImaginaryTE<S>::normalize)
        .def("solve", &ImaginaryTE<S>::solve, py::arg("n_sweeps"),
             py::arg("beta"), py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<typename Compress<S>::Iteration,
               shared_ptr<typename Compress<S>::Iteration>>(m,
                                                            "CompressIteration")
        .def(py::init<double, double, size_t, double>())
        .def(py::init<double, double>())
        .def_readwrite("norm", &Compress<S>::Iteration::norm)
        .def_readwrite("error", &Compress<S>::Iteration::error)
        .def_readwrite("tmult", &Compress<S>::Iteration::tmult)
        .def_readwrite("nflop", &Compress<S>::Iteration::nflop)
        .def("__repr__", [](typename Compress<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Compress<S>, shared_ptr<Compress<S>>>(m, "Compress")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<uint16_t> &, const vector<uint16_t> &,
                      const vector<double> &>())
        .def_readwrite("iprint", &Compress<S>::iprint)
        .def_readwrite("cutoff", &Compress<S>::cutoff)
        .def_readwrite("me", &Compress<S>::me)
        .def_readwrite("bra_bond_dims", &Compress<S>::bra_bond_dims)
        .def_readwrite("ket_bond_dims", &Compress<S>::ket_bond_dims)
        .def_readwrite("noises", &Compress<S>::noises)
        .def_readwrite("norms", &Compress<S>::norms)
        .def_readwrite("forward", &Compress<S>::forward)
        .def_readwrite("noise_type", &Compress<S>::noise_type)
        .def_readwrite("trunc_type", &Compress<S>::trunc_type)
        .def("update_two_dot", &Compress<S>::update_two_dot)
        .def("blocking", &Compress<S>::blocking)
        .def("sweep", &Compress<S>::sweep)
        .def("solve", &Compress<S>::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<typename Expect<S>::Iteration,
               shared_ptr<typename Expect<S>::Iteration>>(m, "ExpectIteration")
        .def(py::init<const vector<pair<shared_ptr<OpExpr<S>>, double>> &,
                      double, double, size_t, double>())
        .def(py::init<const vector<pair<shared_ptr<OpExpr<S>>, double>> &,
                      double, double>())
        .def_readwrite("bra_error", &Expect<S>::Iteration::bra_error)
        .def_readwrite("ket_error", &Expect<S>::Iteration::ket_error)
        .def_readwrite("tmult", &Expect<S>::Iteration::tmult)
        .def_readwrite("nflop", &Expect<S>::Iteration::nflop)
        .def("__repr__", [](typename Expect<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Expect<S>, shared_ptr<Expect<S>>>(m, "Expect")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &, uint16_t,
                      uint16_t>())
        .def_readwrite("iprint", &Expect<S>::iprint)
        .def_readwrite("cutoff", &Expect<S>::cutoff)
        .def_readwrite("me", &Expect<S>::me)
        .def_readwrite("bra_bond_dim", &Expect<S>::bra_bond_dim)
        .def_readwrite("ket_bond_dim", &Expect<S>::ket_bond_dim)
        .def_readwrite("expectations", &Expect<S>::expectations)
        .def_readwrite("forward", &Expect<S>::forward)
        .def_readwrite("trunc_type", &Expect<S>::trunc_type)
        .def("update_two_dot", &Expect<S>::update_two_dot)
        .def("blocking", &Expect<S>::blocking)
        .def("sweep", &Expect<S>::sweep)
        .def("solve", &Expect<S>::solve, py::arg("propagate"),
             py::arg("forward") = true)
        .def("get_1pdm_spatial", &Expect<S>::get_1pdm_spatial,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1pdm", &Expect<S>::get_1pdm,
             py::arg("n_physical_sites") = (uint16_t)0U);

    py::class_<MPOSchemer<S>, shared_ptr<MPOSchemer<S>>>(m, "MPOSchemer")
        .def_readwrite("left_trans_site", &MPOSchemer<S>::left_trans_site)
        .def_readwrite("right_trans_site", &MPOSchemer<S>::right_trans_site)
        .def_readwrite("left_new_operator_names",
                       &MPOSchemer<S>::left_new_operator_names)
        .def_readwrite("right_new_operator_names",
                       &MPOSchemer<S>::right_new_operator_names)
        .def_readwrite("left_new_operator_exprs",
                       &MPOSchemer<S>::left_new_operator_exprs)
        .def_readwrite("right_new_operator_exprs",
                       &MPOSchemer<S>::right_new_operator_exprs)
        .def(py::init<uint8_t, uint8_t>())
        .def("copy", &MPOSchemer<S>::copy)
        .def("get_transform_formulas", &MPOSchemer<S>::get_transform_formulas);

    py::class_<MPO<S>, shared_ptr<MPO<S>>>(m, "MPO")
        .def(py::init<int>())
        .def_readwrite("n_sites", &MPO<S>::n_sites)
        .def_readwrite("const_e", &MPO<S>::const_e)
        .def_readwrite("tensors", &MPO<S>::tensors)
        .def_readwrite("left_operator_names", &MPO<S>::left_operator_names)
        .def_readwrite("right_operator_names", &MPO<S>::right_operator_names)
        .def_readwrite("middle_operator_names", &MPO<S>::middle_operator_names)
        .def_readwrite("left_operator_exprs", &MPO<S>::left_operator_exprs)
        .def_readwrite("right_operator_exprs", &MPO<S>::right_operator_exprs)
        .def_readwrite("middle_operator_exprs", &MPO<S>::middle_operator_exprs)
        .def_readwrite("op", &MPO<S>::op)
        .def_readwrite("schemer", &MPO<S>::schemer)
        .def_readwrite("tf", &MPO<S>::tf)
        .def_readwrite("site_op_infos", &MPO<S>::site_op_infos)
        .def_readwrite("schemer", &MPO<S>::schemer)
        .def("get_blocking_formulas", &MPO<S>::get_blocking_formulas)
        .def("get_ancilla_type", &MPO<S>::get_ancilla_type)
        .def("deallocate", &MPO<S>::deallocate);

    py::class_<Rule<S>, shared_ptr<Rule<S>>>(m, "Rule")
        .def(py::init<>())
        .def("__call__", &Rule<S>::operator());

    py::class_<NoTransposeRule<S>, shared_ptr<NoTransposeRule<S>>, Rule<S>>(
        m, "NoTransposeRule")
        .def_readwrite("prim_rule", &NoTransposeRule<S>::prim_rule)
        .def(py::init<const shared_ptr<Rule<S>> &>());

    py::class_<RuleQC<S>, shared_ptr<RuleQC<S>>, Rule<S>>(m, "RuleQC")
        .def(py::init<>())
        .def(py::init<bool, bool, bool, bool, bool, bool>());

    py::class_<SimplifiedMPO<S>, shared_ptr<SimplifiedMPO<S>>, MPO<S>>(
        m, "SimplifiedMPO")
        .def_readwrite("prim_mpo", &SimplifiedMPO<S>::prim_mpo)
        .def_readwrite("rule", &SimplifiedMPO<S>::rule)
        .def_readwrite("collect_terms", &SimplifiedMPO<S>::collect_terms)
        .def(
            py::init<const shared_ptr<MPO<S>> &, const shared_ptr<Rule<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &, const shared_ptr<Rule<S>> &,
                      bool>())
        .def("simplify_expr", &SimplifiedMPO<S>::simplify_expr)
        .def("simplify_symbolic", &SimplifiedMPO<S>::simplify_symbolic)
        .def("simplify", &SimplifiedMPO<S>::simplify);

    py::class_<IdentityMPO<S>, shared_ptr<IdentityMPO<S>>, MPO<S>>(
        m, "IdentityMPO")
        .def(py::init<const Hamiltonian<S> &>());

    py::class_<MPOQC<S>, shared_ptr<MPOQC<S>>, MPO<S>>(m, "MPOQC")
        .def_readwrite("mode", &MPOQC<S>::mode)
        .def(py::init<const HamiltonianQC<S> &>())
        .def(py::init<const HamiltonianQC<S> &, QCTypes>());

    py::class_<PDM1MPOQC<S>, shared_ptr<PDM1MPOQC<S>>, MPO<S>>(m, "PDM1MPOQC")
        .def(py::init<const Hamiltonian<S> &>());

    py::class_<AncillaMPO<S>, shared_ptr<AncillaMPO<S>>, MPO<S>>(m,
                                                                 "AncillaMPO")
        .def_readwrite("n_physical_sites", &AncillaMPO<S>::n_physical_sites)
        .def_readwrite("prim_mpo", &AncillaMPO<S>::prim_mpo)
        .def(py::init<const shared_ptr<MPO<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &, bool>());
}

PYBIND11_MODULE(block2, m) {

    m.doc() = "python interface for block2.";

    // Handle Ctrl-C from python side
    check_signal_() = []() {
        if (PyErr_CheckSignals() != 0)
            throw py::error_already_set();
    };

    py::bind_vector<vector<int>>(m, "VectorInt");
    py::bind_vector<vector<uint16_t>>(m, "VectorUInt16");
    py::bind_vector<vector<double>>(m, "VectorDouble");
    py::bind_vector<vector<uint8_t>>(m, "VectorUInt8")
        .def_property_readonly(
            "ptr",
            [](vector<uint8_t> *self) -> uint8_t * { return &(*self)[0]; })
        .def("__str__", [](vector<uint8_t> *self) {
            stringstream ss;
            ss << "VectorUInt8[ ";
            for (auto p : *self)
                ss << (int)p << " ";
            ss << "]";
            return ss.str();
        });

    m.def(
        "init_memory",
        [](size_t isize, size_t dsize, const string &save_dir) {
            frame_() = new DataFrame(isize, dsize, save_dir);
        },
        py::arg("isize") = size_t(1L << 28),
        py::arg("dsize") = size_t(1L << 30), py::arg("save_dir") = "nodex");

    m.def("release_memory", []() {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        delete frame_();
    });

    m.def("set_mkl_num_threads", [](int n) {
        mkl_set_num_threads(n);
        mkl_set_dynamic(0);
    });

    m.def("read_occ", &read_occ);
    m.def("write_occ", &write_occ);

    py::class_<StackAllocator<uint32_t>>(m, "IntAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<uint32_t>::size)
        .def_readwrite("used", &StackAllocator<uint32_t>::used)
        .def_readwrite("shift", &StackAllocator<uint32_t>::shift);

    py::class_<StackAllocator<double>>(m, "DoubleAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<double>::size)
        .def_readwrite("used", &StackAllocator<double>::used)
        .def_readwrite("shift", &StackAllocator<double>::shift);

    struct Global {};

    py::class_<DataFrame, shared_ptr<DataFrame>>(m, "DataFrame")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, const string &>())
        .def_readwrite("save_dir", &DataFrame::save_dir)
        .def_readwrite("prefix", &DataFrame::prefix)
        .def_readwrite("isize", &DataFrame::isize)
        .def_readwrite("dsize", &DataFrame::dsize)
        .def_readwrite("n_frames", &DataFrame::n_frames)
        .def_readwrite("i_frame", &DataFrame::i_frame)
        .def_readwrite("iallocs", &DataFrame::iallocs)
        .def_readwrite("dallocs", &DataFrame::dallocs)
        .def("activate", &DataFrame::activate)
        .def("load_data", &DataFrame::load_data)
        .def("save_data", &DataFrame::save_data)
        .def("reset", &DataFrame::reset);

    py::class_<Global>(m, "Global")
        .def_property_static(
            "ialloc", [](py::object) { return ialloc_(); },
            [](py::object, StackAllocator<uint32_t> *ia) { ialloc_() = ia; })
        .def_property_static(
            "dalloc", [](py::object) { return dalloc_(); },
            [](py::object, StackAllocator<double> *da) { dalloc_() = da; })
        .def_property_static(
            "frame", [](py::object) { return frame_(); },
            [](py::object, DataFrame *fr) { frame_() = fr; });

    bind_array<uint8_t>(m, "ArrayUInt8")
        .def("__str__", [](Array<uint8_t> *self) {
            stringstream ss;
            ss << "ArrayUInt8(LEN=" << self->n << ")[ ";
            for (size_t i = 0; i < self->n; i++)
                ss << (int)self->data[i] << " ";
            ss << "]";
            return ss.str();
        });
    bind_array<uint16_t>(m, "ArrayUInt16");

    py::class_<Random, shared_ptr<Random>>(m, "Random")
        .def_static("rand_seed", &Random::rand_seed, py::arg("i") = 0U)
        .def_static("rand_int", &Random::rand_int, py::arg("a"), py::arg("b"))
        .def_static("rand_double", &Random::rand_double, py::arg("a") = 0,
                    py::arg("b") = 1)
        .def_static("fill_rand_double",
                    [](py::object, py::array_t<double> &data, double a = 0,
                       double b = 1) {
                        return Random::fill_rand_double(data.mutable_data(),
                                                        data.size(), a, b);
                    });

    py::class_<MatrixRef, shared_ptr<MatrixRef>>(m, "Matrix",
                                                 py::buffer_protocol())
        .def_buffer([](MatrixRef *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(double),
                                   py::format_descriptor<double>::format(), 2,
                                   {self->m, self->n},
                                   {sizeof(double) * self->n, sizeof(double)});
        })
        .def_readwrite("m", &MatrixRef::m)
        .def_readwrite("n", &MatrixRef::n)
        .def("deallocate", &MatrixRef::deallocate);

    py::class_<MatrixFunctions>(m, "MatrixFunctions")
        .def_static(
            "eigs", [](int n, py::array_t<double> &a, py::array_t<double> &w) {
                MatrixFunctions::eigs(MatrixRef(a.mutable_data(), n, n),
                                      DiagonalMatrix(w.mutable_data(), n));
            });

    py::class_<DiagonalMatrix, shared_ptr<DiagonalMatrix>>(
        m, "DiagonalMatrix", py::buffer_protocol())
        .def_buffer([](DiagonalMatrix *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(double),
                                   py::format_descriptor<double>::format(), 1,
                                   {self->n}, {sizeof(double)});
        });

    py::class_<FCIDUMP, shared_ptr<FCIDUMP>>(m, "FCIDUMP")
        .def(py::init<>())
        .def("read", &FCIDUMP::read)
        .def("write", &FCIDUMP::write)
        .def("initialize_su2",
             [](FCIDUMP *self, uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                uint16_t isym, double e, const py::array_t<double> &t,
                const py::array_t<double> &v) {
                 self->initialize_su2(n_sites, n_elec, twos, isym, e, t.data(),
                                      t.size(), v.data(), v.size());
             })
        .def("initialize_sz",
             [](FCIDUMP *self, uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                uint16_t isym, double e, const py::tuple &t,
                const py::tuple &v) {
                 assert(t.size() == 2 && v.size() == 3);
                 py::array_t<double> ta = t[0].cast<py::array_t<double>>();
                 py::array_t<double> tb = t[1].cast<py::array_t<double>>();
                 py::array_t<double> va = v[0].cast<py::array_t<double>>();
                 py::array_t<double> vb = v[1].cast<py::array_t<double>>();
                 py::array_t<double> vab = v[2].cast<py::array_t<double>>();
                 self->initialize_sz(n_sites, n_elec, twos, isym, e, ta.data(),
                                     ta.size(), tb.data(), tb.size(), va.data(),
                                     va.size(), vb.data(), vb.size(),
                                     vab.data(), vab.size());
             })
        .def("deallocate", &FCIDUMP::deallocate)
        .def_property("orb_sym", &FCIDUMP::orb_sym, &FCIDUMP::set_orb_sym)
        .def_property_readonly("n_elec", &FCIDUMP::n_elec)
        .def_property_readonly("twos", &FCIDUMP::twos)
        .def_property_readonly("isym", &FCIDUMP::isym)
        .def_property_readonly("n_sites", &FCIDUMP::n_sites)
        .def_readwrite("params", &FCIDUMP::params)
        .def_readwrite("ts", &FCIDUMP::ts)
        .def_readwrite("vs", &FCIDUMP::vs)
        .def_readwrite("vabs", &FCIDUMP::vabs)
        .def_readwrite("e", &FCIDUMP::e)
        .def_readwrite("total_memory", &FCIDUMP::total_memory)
        .def_readwrite("uhf", &FCIDUMP::uhf);

    py::enum_<OpNames>(m, "OpNames", py::arithmetic())
        .value("H", OpNames::H)
        .value("I", OpNames::I)
        .value("N", OpNames::N)
        .value("NN", OpNames::NN)
        .value("C", OpNames::C)
        .value("D", OpNames::D)
        .value("R", OpNames::R)
        .value("RD", OpNames::RD)
        .value("A", OpNames::A)
        .value("AD", OpNames::AD)
        .value("P", OpNames::P)
        .value("PD", OpNames::PD)
        .value("B", OpNames::B)
        .value("Q", OpNames::Q)
        .value("Zero", OpNames::Zero)
        .value("PDM1", OpNames::PDM1);

    py::enum_<OpTypes>(m, "OpTypes", py::arithmetic())
        .value("Zero", OpTypes::Zero)
        .value("Elem", OpTypes::Elem)
        .value("Prod", OpTypes::Prod)
        .value("Sum", OpTypes::Sum)
        .value("ElemRef", OpTypes::ElemRef)
        .value("SumProd", OpTypes::SumProd);

    py::enum_<NoiseTypes>(m, "NoiseTypes", py::arithmetic())
        .value("Nothing", NoiseTypes::None)
        .value("Wavefunction", NoiseTypes::Wavefunction)
        .value("DensityMatrix", NoiseTypes::DensityMatrix)
        .value("Perturbative", NoiseTypes::Perturbative);

    py::enum_<TruncationTypes>(m, "TruncationTypes", py::arithmetic())
        .value("Physical", TruncationTypes::Physical)
        .value("Reduced", TruncationTypes::Reduced)
        .value("ReducedInversed", TruncationTypes::ReducedInversed);

    py::class_<SiteIndex>(m, "SiteIndex")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<uint8_t>())
        .def(py::init<uint8_t, uint8_t, uint8_t>())
        .def("size", &SiteIndex::size)
        .def("spin_size", &SiteIndex::spin_size)
        .def("s", &SiteIndex::s, py::arg("i") = 0)
        .def("__getitem__",
             [](SiteIndex *self, uint8_t i) { return self->operator[](i); })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__hash__", &SiteIndex::hash)
        .def("__repr__", &SiteIndex::to_str);

    py::class_<SZ>(m, "SZ")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def_readwrite("data", &SZ::data)
        .def_property("n", &SZ::n, &SZ::set_n)
        .def_property("twos", &SZ::twos, &SZ::set_twos)
        .def_property("pg", &SZ::pg, &SZ::set_pg)
        .def_property_readonly("multiplicity", &SZ::multiplicity)
        .def_property_readonly("is_fermion", &SZ::is_fermion)
        .def_property_readonly("count", &SZ::count)
        .def("combine", &SZ::combine)
        .def("__getitem__", &SZ::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SZ::get_ket)
        .def("get_bra", &SZ::get_bra, py::arg("dq"))
        .def("__repr__", &SZ::to_str);

    py::class_<SU2>(m, "SU2")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int>())
        .def_readwrite("data", &SU2::data)
        .def_property("n", &SU2::n, &SU2::set_n)
        .def_property("twos", &SU2::twos, &SU2::set_twos)
        .def_property("twos_low", &SU2::twos_low, &SU2::set_twos_low)
        .def_property("pg", &SU2::pg, &SU2::set_pg)
        .def_property_readonly("multiplicity", &SU2::multiplicity)
        .def_property_readonly("is_fermion", &SU2::is_fermion)
        .def_property_readonly("count", &SU2::count)
        .def("combine", &SU2::combine)
        .def("__getitem__", &SU2::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SU2::get_ket)
        .def("get_bra", &SU2::get_bra, py::arg("dq"))
        .def("__repr__", &SU2::to_str);

    py::enum_<SymTypes>(m, "SymTypes", py::arithmetic())
        .value("RVec", SymTypes::RVec)
        .value("CVec", SymTypes::CVec)
        .value("Mat", SymTypes::Mat);

    py::enum_<AncillaTypes>(m, "AncillaTypes", py::arithmetic())
        .value("Nothing", AncillaTypes::None)
        .value("Ancilla", AncillaTypes::Ancilla);

    py::enum_<ActiveTypes>(m, "ActiveTypes", py::arithmetic())
        .value("Empty", ActiveTypes::Empty)
        .value("Active", ActiveTypes::Active)
        .value("Frozen", ActiveTypes::Frozen);

    py::bind_vector<vector<ActiveTypes>>(m, "VectorActTypes");

    py::enum_<SeqTypes>(m, "SeqTypes", py::arithmetic())
        .value("Nothing", SeqTypes::None)
        .value("Simple", SeqTypes::Simple)
        .value("Auto", SeqTypes::Auto);

    py::class_<BatchGEMMSeq, shared_ptr<BatchGEMMSeq>>(m, "BatchGEMMSeq")
        .def_readwrite("batch", &BatchGEMMSeq::batch)
        .def_readwrite("post_batch", &BatchGEMMSeq::post_batch)
        .def_readwrite("refs", &BatchGEMMSeq::refs)
        .def_readwrite("cumulative_nflop", &BatchGEMMSeq::cumulative_nflop)
        .def_readwrite("mode", &BatchGEMMSeq::mode)
        .def(py::init<>())
        .def(py::init<size_t>())
        .def(py::init<size_t, SeqTypes>())
        .def("iadd", &BatchGEMMSeq::iadd, py::arg("a"), py::arg("b"),
             py::arg("scale") = 1.0, py::arg("cfactor") = 1.0,
             py::arg("conj") = false)
        .def("rotate", &BatchGEMMSeq::rotate, py::arg("a"), py::arg("c"),
             py::arg("bra"), py::arg("conj_bra"), py::arg("ket"),
             py::arg("conj_ket"), py::arg("scale"))
        .def("tensor_product_diagonal", &BatchGEMMSeq::tensor_product_diagonal,
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("scale"))
        .def("tensor_product", &BatchGEMMSeq::tensor_product, py::arg("a"),
             py::arg("conja"), py::arg("b"), py::arg("conjb"), py::arg("c"),
             py::arg("scale"), py::arg("stride"))
        .def("divide_batch", &BatchGEMMSeq::divide_batch)
        .def("check", &BatchGEMMSeq::check)
        .def("prepare", &BatchGEMMSeq::prepare)
        .def("allocate", &BatchGEMMSeq::allocate)
        .def("deallocate", &BatchGEMMSeq::deallocate)
        .def("simple_perform", &BatchGEMMSeq::simple_perform)
        .def("auto_perform", &BatchGEMMSeq::auto_perform)
        .def("perform", &BatchGEMMSeq::perform)
        .def("clear", &BatchGEMMSeq::clear)
        .def("__call__", &BatchGEMMSeq::operator())
        .def("__repr__", [](BatchGEMMSeq *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<PointGroup, shared_ptr<PointGroup>>(m, "PointGroup")
        .def_static("swap_c1", &PointGroup::swap_c1)
        .def_static("swap_ci", &PointGroup::swap_ci)
        .def_static("swap_cs", &PointGroup::swap_cs)
        .def_static("swap_c2", &PointGroup::swap_c2)
        .def_static("swap_c2h", &PointGroup::swap_c2h)
        .def_static("swap_c2v", &PointGroup::swap_c2v)
        .def_static("swap_d2", &PointGroup::swap_d2)
        .def_static("swap_d2h", &PointGroup::swap_d2h);

    py::enum_<FuseTypes>(m, "FuseTypes", py::arithmetic())
        .value("NoFuse", FuseTypes::NoFuse)
        .value("FuseL", FuseTypes::FuseL)
        .value("FuseR", FuseTypes::FuseR)
        .value("FuseLR", FuseTypes::FuseLR);

    py::enum_<TETypes>(m, "TETypes", py::arithmetic())
        .value("TangentSpace", TETypes::TangentSpace)
        .value("RK4", TETypes::RK4);

    py::enum_<QCTypes>(m, "QCTypes", py::arithmetic())
        .value("NC", QCTypes::NC)
        .value("CN", QCTypes::CN)
        .value("NCCN", QCTypes(QCTypes::NC | QCTypes::CN))
        .value("Conventional", QCTypes::Conventional);

    py::module m_su2 = m.def_submodule("su2", "Spin-adapted.");
    bind_class<SU2>(m_su2, "SU2");

    py::module m_sz = m.def_submodule("sz", "Non-spin-adapted.");
    // bind_class<SZ>(m_sz, "SZ");
}
