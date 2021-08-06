
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

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "../block2_core.hpp"

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<char>);
PYBIND11_MAKE_OPAQUE(vector<long long int>);
PYBIND11_MAKE_OPAQUE(vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(vector<double>);
PYBIND11_MAKE_OPAQUE(vector<complex<double>>);
PYBIND11_MAKE_OPAQUE(vector<size_t>);
PYBIND11_MAKE_OPAQUE(vector<vector<uint32_t>>);
PYBIND11_MAKE_OPAQUE(vector<vector<double>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<double>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<int>>);
PYBIND11_MAKE_OPAQUE(vector<pair<int, int>>);
PYBIND11_MAKE_OPAQUE(vector<pair<long long int, int>>);
PYBIND11_MAKE_OPAQUE(vector<pair<long long int, long long int>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Tensor>>);
PYBIND11_MAKE_OPAQUE(vector<vector<shared_ptr<Tensor>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<CSRMatrixRef>>);
// SZ
PYBIND11_MAKE_OPAQUE(vector<SZ>);
PYBIND11_MAKE_OPAQUE(vector<pair<uint8_t, SZ>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<uint8_t, SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SZ, double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<StateInfo<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SZ>>, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SZ, shared_ptr<SparseMatrixInfo<SZ>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<SZ, shared_ptr<SparseMatrixInfo<SZ>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicRowVector<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicColumnVector<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicMatrix<SZ>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<map<OpNames, shared_ptr<SparseMatrix<SZ>>>>);
PYBIND11_MAKE_OPAQUE(map<shared_ptr<OpExpr<SZ>>, shared_ptr<SparseMatrix<SZ>>,
                         op_expr_less<SZ>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<SZ, SZ>, shared_ptr<Tensor>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<pair<SZ, SZ>, shared_ptr<Tensor>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<SU2>);
PYBIND11_MAKE_OPAQUE(vector<pair<uint8_t, SU2>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<uint8_t, SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SU2, double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<StateInfo<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SU2>>, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SU2, shared_ptr<SparseMatrixInfo<SU2>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<SU2, shared_ptr<SparseMatrixInfo<SU2>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicRowVector<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicColumnVector<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicMatrix<SU2>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<map<OpNames, shared_ptr<SparseMatrix<SU2>>>>);
PYBIND11_MAKE_OPAQUE(map<shared_ptr<OpExpr<SU2>>, shared_ptr<SparseMatrix<SU2>>,
                         op_expr_less<SU2>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<SU2, SU2>, shared_ptr<Tensor>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<pair<SU2, SU2>, shared_ptr<Tensor>>>>);
PYBIND11_MAKE_OPAQUE(map<string, string>);

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
        .def("initialize", [](CG<S> *self) { self->initialize(); })
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
        .def("initialize", [](CG<S> *self) { self->initialize(); })
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

template <typename S> void bind_expr(py::module &m) {
    py::class_<OpExpr<S>, shared_ptr<OpExpr<S>>>(m, "OpExpr")
        .def(py::init<>())
        .def("get_type", &OpExpr<S>::get_type)
        .def(py::self == py::self)
        .def("__repr__", &to_str<S>);

    py::bind_vector<vector<pair<shared_ptr<OpExpr<S>>, double>>>(
        m, "VectorPExprDouble");

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

    py::class_<OpProduct<S>, shared_ptr<OpProduct<S>>, OpExpr<S>>(m,
                                                                  "OpProduct")
        .def(py::init<const shared_ptr<OpElement<S>> &, double>())
        .def(py::init<const shared_ptr<OpElement<S>> &, double, uint8_t>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const shared_ptr<OpElement<S>> &, double>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const shared_ptr<OpElement<S>> &, double, uint8_t>())
        .def_readwrite("factor", &OpProduct<S>::factor)
        .def_readwrite("conj", &OpProduct<S>::conj)
        .def_readwrite("a", &OpProduct<S>::a)
        .def_readwrite("b", &OpProduct<S>::b)
        .def("__hash__", &OpProduct<S>::hash);

    py::class_<OpSumProd<S>, shared_ptr<OpSumProd<S>>, OpProduct<S>>(
        m, "OpSumProd")
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const vector<shared_ptr<OpElement<S>>> &,
                      const vector<bool> &, double, uint8_t,
                      const shared_ptr<OpElement<S>> &>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const vector<shared_ptr<OpElement<S>>> &,
                      const vector<bool> &, double, uint8_t>())
        .def(py::init<const shared_ptr<OpElement<S>> &,
                      const vector<shared_ptr<OpElement<S>>> &,
                      const vector<bool> &, double>())
        .def(py::init<const vector<shared_ptr<OpElement<S>>> &,
                      const shared_ptr<OpElement<S>> &, const vector<bool> &,
                      double, uint8_t, const shared_ptr<OpElement<S>> &>())
        .def(py::init<const vector<shared_ptr<OpElement<S>>> &,
                      const shared_ptr<OpElement<S>> &, const vector<bool> &,
                      double, uint8_t>())
        .def(py::init<const vector<shared_ptr<OpElement<S>>> &,
                      const shared_ptr<OpElement<S>> &, const vector<bool> &,
                      double>())
        .def_readwrite("ops", &OpSumProd<S>::ops)
        .def_readwrite("conjs", &OpSumProd<S>::conjs)
        .def_readwrite("c", &OpSumProd<S>::c);

    py::class_<OpSum<S>, shared_ptr<OpSum<S>>, OpExpr<S>>(m, "OpSum")
        .def(py::init<const vector<shared_ptr<OpProduct<S>>> &>())
        .def_readwrite("strings", &OpSum<S>::strings);

    py::bind_vector<vector<shared_ptr<OpExpr<S>>>>(m, "VectorOpExpr");
    py::bind_vector<vector<shared_ptr<OpElement<S>>>>(m, "VectorOpElement");
    py::bind_vector<vector<shared_ptr<OpProduct<S>>>>(m, "VectorOpProduct");

    struct PySymbolic : Symbolic<S> {
        using Symbolic<S>::Symbolic;
        SymTypes get_type() const override {
            PYBIND11_OVERLOAD_PURE(const SymTypes, Symbolic<S>, get_type, );
        }
        shared_ptr<Symbolic<S>> copy() const override {
            PYBIND11_OVERLOAD_PURE(shared_ptr<Symbolic<S>>, Symbolic<S>,
                                   copy, );
        }
    };

    py::class_<Symbolic<S>, PySymbolic, shared_ptr<Symbolic<S>>>(m, "Symbolic")
        .def(py::init<int, int>())
        .def_readwrite("m", &Symbolic<S>::m)
        .def_readwrite("n", &Symbolic<S>::n)
        .def_readwrite("data", &Symbolic<S>::data)
        .def("get_type", [](Symbolic<S> *self) { return self->get_type(); })
        .def("__matmul__",
             [](const shared_ptr<Symbolic<S>> &self,
                const shared_ptr<Symbolic<S>> &other) { return self * other; });

    py::class_<SymbolicRowVector<S>, shared_ptr<SymbolicRowVector<S>>,
               Symbolic<S>>(m, "SymbolicRowVector")
        .def(py::init<int>())
        .def("__getitem__",
             [](SymbolicRowVector<S> *self, int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SymbolicRowVector<S> *self, int idx,
                const shared_ptr<OpExpr<S>> &v) { (*self)[idx] = v; })
        .def("__len__", [](SymbolicRowVector<S> *self) { return self->n; })
        .def("copy", &SymbolicRowVector<S>::copy);

    py::class_<SymbolicColumnVector<S>, shared_ptr<SymbolicColumnVector<S>>,
               Symbolic<S>>(m, "SymbolicColumnVector")
        .def(py::init<int>())
        .def("__getitem__", [](SymbolicColumnVector<S> *self,
                               int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SymbolicColumnVector<S> *self, int idx,
                const shared_ptr<OpExpr<S>> &v) { (*self)[idx] = v; })
        .def("__len__", [](SymbolicColumnVector<S> *self) { return self->m; })
        .def("copy", &SymbolicColumnVector<S>::copy);

    py::class_<SymbolicMatrix<S>, shared_ptr<SymbolicMatrix<S>>, Symbolic<S>>(
        m, "SymbolicMatrix")
        .def(py::init<int, int>())
        .def_readwrite("indices", &SymbolicMatrix<S>::indices)
        .def("__setitem__",
             [](SymbolicMatrix<S> *self, int i, int j,
                const shared_ptr<OpExpr<S>> &v) {
                 (*self)[{i, j}] = v;
             })
        .def("copy", &SymbolicMatrix<S>::copy);

    py::bind_vector<vector<shared_ptr<Symbolic<S>>>>(m, "VectorSymbolic");
    py::bind_vector<vector<shared_ptr<SymbolicRowVector<S>>>>(
        m, "VectorSymbolicRowVector");
    py::bind_vector<vector<shared_ptr<SymbolicColumnVector<S>>>>(
        m, "VectorSymbolicColumnVector");
    py::bind_vector<vector<shared_ptr<SymbolicMatrix<S>>>>(
        m, "VectorSymbolicMatrix");
}

template <typename S> void bind_state_info(py::module &m, const string &name) {

    bind_array<StateInfo<S>>(m, "ArrayStateInfo");
    bind_array<S>(m, ("Array" + name).c_str());
    bind_array<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
        m, "ArrayVectorPLMatInfo");
    py::bind_vector<vector<shared_ptr<StateInfo<S>>>>(m, "VectorStateInfo");
    py::bind_vector<vector<pair<uint8_t, S>>>(m, "VectorPUInt8S");
    py::bind_vector<vector<vector<pair<uint8_t, S>>>>(m, "VectorVectorPUInt8S");
    py::bind_vector<vector<pair<S, double>>>(m, "VectorPSDouble");
    py::bind_vector<vector<vector<pair<S, double>>>>(m, "VectorVectorPSDouble");
    py::bind_vector<vector<vector<vector<pair<S, double>>>>>(
        m, "VectorVectorVectorPSDouble");

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
                                   return Array<ubond_t>(self->n_states,
                                                         self->n);
                               })
        .def("load_data",
             (void (StateInfo<S>::*)(const string &)) & StateInfo<S>::load_data)
        .def("save_data", (void (StateInfo<S>::*)(const string &) const) &
                              StateInfo<S>::save_data)
        .def("allocate",
             [](StateInfo<S> *self, int length) { self->allocate(length); })
        .def("deallocate", &StateInfo<S>::deallocate)
        .def("reallocate", &StateInfo<S>::reallocate, py::arg("length"))
        .def("sort_states", &StateInfo<S>::sort_states)
        .def("copy_data_to", &StateInfo<S>::copy_data_to)
        .def("deep_copy", &StateInfo<S>::deep_copy)
        .def("collect", &StateInfo<S>::collect,
             py::arg("target") = S(S::invalid))
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
}

template <typename S> void bind_sparse(py::module &m) {

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
        .def("initialize_tp",
             &SparseMatrixInfo<S>::ConnectionInfo::initialize_tp)
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
        .def(py::init<const shared_ptr<Allocator<uint32_t>> &>())
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
                                   return py::array_t<ubond_t>(
                                       self->n, self->n_states_bra);
                               })
        .def_property_readonly("n_states_ket",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<ubond_t>(
                                       self->n, self->n_states_ket);
                               })
        .def("initialize", &SparseMatrixInfo<S>::initialize, py::arg("bra"),
             py::arg("ket"), py::arg("dq"), py::arg("is_fermion"),
             py::arg("wfn") = false)
        .def("initialize_trans_contract",
             &SparseMatrixInfo<S>::initialize_trans_contract)
        .def("initialize_contract", &SparseMatrixInfo<S>::initialize_contract)
        .def("initialize_dm", &SparseMatrixInfo<S>::initialize_dm)
        .def("find_state", &SparseMatrixInfo<S>::find_state, py::arg("q"),
             py::arg("start") = 0)
        .def_property_readonly("total_memory",
                               &SparseMatrixInfo<S>::get_total_memory)
        .def("allocate", &SparseMatrixInfo<S>::allocate, py::arg("length"),
             py::arg("ptr") = nullptr)
        .def("extract_state_info", &SparseMatrixInfo<S>::extract_state_info,
             py::arg("right"))
        .def("deallocate", &SparseMatrixInfo<S>::deallocate)
        .def("reallocate", &SparseMatrixInfo<S>::reallocate, py::arg("length"))
        .def("__repr__", [](SparseMatrixInfo<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<SparseMatrix<S>, shared_ptr<SparseMatrix<S>>>(m, "SparseMatrix")
        .def(py::init<>())
        .def(py::init<const shared_ptr<Allocator<double>> &>())
        .def_readwrite("info", &SparseMatrix<S>::info)
        .def_readwrite("factor", &SparseMatrix<S>::factor)
        .def_readwrite("total_memory", &SparseMatrix<S>::total_memory)
        .def("get_type", &SparseMatrix<S>::get_type)
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
        .def("load_data",
             (void (SparseMatrix<S>::*)(
                 const string &, bool,
                 const shared_ptr<Allocator<uint32_t>> &)) &
                 SparseMatrix<S>::load_data,
             py::arg("filename"), py::arg("load_info") = false,
             py::arg("i_alloc") = nullptr)
        .def("save_data",
             (void (SparseMatrix<S>::*)(const string &, bool) const) &
                 SparseMatrix<S>::save_data,
             py::arg("filename"), py::arg("save_info") = false)
        .def("copy_data_from", &SparseMatrix<S>::copy_data_from)
        .def("selective_copy_from", &SparseMatrix<S>::selective_copy_from)
        .def("sparsity", &SparseMatrix<S>::sparsity)
        .def("allocate",
             [](SparseMatrix<S> *self,
                const shared_ptr<SparseMatrixInfo<S>> &info) {
                 self->allocate(info);
             })
        .def("deallocate", &SparseMatrix<S>::deallocate)
        .def("reallocate",
             (void (SparseMatrix<S>::*)(size_t)) & SparseMatrix<S>::reallocate,
             py::arg("length"))
        .def("trace", &SparseMatrix<S>::trace)
        .def("norm", &SparseMatrix<S>::norm)
        .def("__getitem__",
             [](SparseMatrix<S> *self, int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SparseMatrix<S> *self, int idx, const py::array_t<double> &v) {
                 assert(v.size() == (*self)[idx].size());
                 memcpy((*self)[idx].data, v.data(), sizeof(double) * v.size());
             })
        .def("left_split",
             [](SparseMatrix<S> *self, ubond_t bond_dim) {
                 shared_ptr<SparseMatrix<S>> left, right;
                 self->left_split(left, right, bond_dim);
                 return make_tuple(left, right);
             })
        .def("right_split",
             [](SparseMatrix<S> *self, ubond_t bond_dim) {
                 shared_ptr<SparseMatrix<S>> left, right;
                 self->right_split(left, right, bond_dim);
                 return make_tuple(left, right);
             })
        .def("pseudo_inverse", &SparseMatrix<S>::pseudo_inverse,
             py::arg("bond_dim"), py::arg("svd_eps") = 1E-4,
             py::arg("svd_cutoff") = 1E-12)
        .def("left_svd",
             [](SparseMatrix<S> *self) {
                 vector<S> qs;
                 vector<shared_ptr<Tensor>> l, s, r;
                 self->left_svd(qs, l, s, r);
                 return make_tuple(qs, l, s, r);
             })
        .def("right_svd",
             [](SparseMatrix<S> *self) {
                 vector<S> qs;
                 vector<shared_ptr<Tensor>> l, s, r;
                 self->right_svd(qs, l, s, r);
                 return make_tuple(qs, l, s, r);
             })
        .def("left_canonicalize", &SparseMatrix<S>::left_canonicalize,
             py::arg("rmat"))
        .def("right_canonicalize", &SparseMatrix<S>::right_canonicalize,
             py::arg("lmat"))
        .def("left_multiply", &SparseMatrix<S>::left_multiply, py::arg("lmat"),
             py::arg("l"), py::arg("m"), py::arg("r"), py::arg("lm"),
             py::arg("lm_cinfo"), py::arg("nlm"))
        .def("right_multiply", &SparseMatrix<S>::right_multiply,
             py::arg("rmat"), py::arg("l"), py::arg("m"), py::arg("r"),
             py::arg("mr"), py::arg("mr_cinfo"), py::arg("nmr"))
        .def("left_multiply_inplace", &SparseMatrix<S>::left_multiply_inplace,
             py::arg("lmat"), py::arg("l"), py::arg("m"), py::arg("r"),
             py::arg("lm"), py::arg("lm_cinfo"))
        .def("right_multiply_inplace", &SparseMatrix<S>::right_multiply_inplace,
             py::arg("rmat"), py::arg("l"), py::arg("m"), py::arg("r"),
             py::arg("mr"), py::arg("mr_cinfo"))
        .def("randomize", &SparseMatrix<S>::randomize, py::arg("a") = 0.0,
             py::arg("b") = 1.0)
        .def("normalize", &SparseMatrix<S>::normalize)
        .def("contract", &SparseMatrix<S>::contract, py::arg("lmat"),
             py::arg("rmat"), py::arg("trace_right") = false)
        .def("swap_to_fused_left", &SparseMatrix<S>::swap_to_fused_left)
        .def("swap_to_fused_right", &SparseMatrix<S>::swap_to_fused_right)
        .def("__repr__", [](SparseMatrix<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<CSRSparseMatrix<S>, shared_ptr<CSRSparseMatrix<S>>,
               SparseMatrix<S>>(m, "CSRSparseMatrix")
        .def(py::init<>())
        .def_readwrite("csr_data", &CSRSparseMatrix<S>::csr_data)
        .def("__getitem__",
             [](CSRSparseMatrix<S> *self, int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](CSRSparseMatrix<S> *self, int idx, const CSRMatrixRef &v) {
                 (*self)[idx].deallocate();
                 (*self)[idx] = v;
             })
        .def("from_dense", &CSRSparseMatrix<S>::from_dense)
        .def("wrap_dense", &CSRSparseMatrix<S>::wrap_dense)
        .def("to_dense", &CSRSparseMatrix<S>::to_dense);

    py::class_<ArchivedSparseMatrix<S>, shared_ptr<ArchivedSparseMatrix<S>>,
               SparseMatrix<S>>(m, "ArchivedSparseMatrix")
        .def(py::init<const string &, int64_t>())
        .def_readwrite("filename", &ArchivedSparseMatrix<S>::filename)
        .def_readwrite("offset", &ArchivedSparseMatrix<S>::offset)
        .def("load_archive", &ArchivedSparseMatrix<S>::load_archive)
        .def("save_archive", &ArchivedSparseMatrix<S>::save_archive);

    py::class_<DelayedSparseMatrix<S>, shared_ptr<DelayedSparseMatrix<S>>,
               SparseMatrix<S>>(m, "DelayedSparseMatrix")
        .def(py::init<>())
        .def("build", &DelayedSparseMatrix<S>::build)
        .def("copy", &DelayedSparseMatrix<S>::copy)
        .def("selective_copy", &DelayedSparseMatrix<S>::selective_copy);

    py::class_<DelayedSparseMatrix<S, SparseMatrix<S>>,
               shared_ptr<DelayedSparseMatrix<S, SparseMatrix<S>>>,
               DelayedSparseMatrix<S>>(m, "DelayedNormalSparseMatrix")
        .def_readwrite("mat", &DelayedSparseMatrix<S, SparseMatrix<S>>::mat)
        .def(py::init<const shared_ptr<SparseMatrix<S>> &>());

    py::class_<DelayedSparseMatrix<S, CSRSparseMatrix<S>>,
               shared_ptr<DelayedSparseMatrix<S, CSRSparseMatrix<S>>>,
               DelayedSparseMatrix<S>>(m, "DelayedCSRSparseMatrix")
        .def_readwrite("mat", &DelayedSparseMatrix<S, CSRSparseMatrix<S>>::mat)
        .def(py::init<const shared_ptr<CSRSparseMatrix<S>> &>());

    py::class_<DelayedSparseMatrix<S, OpExpr<S>>,
               shared_ptr<DelayedSparseMatrix<S, OpExpr<S>>>,
               DelayedSparseMatrix<S>>(m, "DelayedOpExprSparseMatrix")
        .def_readwrite("m", &DelayedSparseMatrix<S, OpExpr<S>>::m)
        .def_readwrite("op", &DelayedSparseMatrix<S, OpExpr<S>>::op)
        .def(py::init<uint16_t, const shared_ptr<OpExpr<S>> &>())
        .def(py::init<uint16_t, const shared_ptr<OpExpr<S>> &,
                      const shared_ptr<SparseMatrixInfo<S>> &>());

    py::class_<DelayedSparseMatrix<S, Hamiltonian<S>>,
               shared_ptr<DelayedSparseMatrix<S, Hamiltonian<S>>>,
               DelayedSparseMatrix<S, OpExpr<S>>>(m, "DelayedHamilSparseMatrix")
        .def_readwrite("hamil", &DelayedSparseMatrix<S, Hamiltonian<S>>::hamil)
        .def(py::init<const shared_ptr<Hamiltonian<S>> &, uint16_t,
                      const shared_ptr<OpExpr<S>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S>> &, uint16_t,
                      const shared_ptr<OpExpr<S>> &,
                      const shared_ptr<SparseMatrixInfo<S>> &>());

    py::bind_vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
        m, "VectorPLMatInfo");
    py::bind_vector<vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>>(
        m, "VectorVectorPLMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrixInfo<S>>>>(m,
                                                             "VectorSpMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrix<S>>>>(m, "VectorSpMat");
    py::bind_vector<vector<map<OpNames, shared_ptr<SparseMatrix<S>>>>>(
        m, "VectorMapOpNamesSpMat");
    py::bind_map<unordered_map<OpNames, shared_ptr<SparseMatrix<S>>>>(
        m, "MapOpNamesSpMat");
    py::bind_map<
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>(
        m, "MapOpExprSpMat");

    py::bind_vector<vector<pair<pair<S, S>, shared_ptr<Tensor>>>>(
        m, "VectorPSSTensor");
    py::bind_vector<vector<vector<pair<pair<S, S>, shared_ptr<Tensor>>>>>(
        m, "VectorVectorPSSTensor");

    py::class_<SparseMatrixGroup<S>, shared_ptr<SparseMatrixGroup<S>>>(
        m, "SparseMatrixGroup")
        .def(py::init<>())
        .def_readwrite("infos", &SparseMatrixGroup<S>::infos)
        .def_readwrite("offsets", &SparseMatrixGroup<S>::offsets)
        .def_readwrite("total_memory", &SparseMatrixGroup<S>::total_memory)
        .def_readwrite("n", &SparseMatrixGroup<S>::n)
        .def_property(
            "data",
            [](SparseMatrixGroup<S> *self) {
                return py::array_t<double>(self->total_memory, self->data);
            },
            [](SparseMatrixGroup<S> *self, const py::array_t<double> &v) {
                assert(v.size() == self->total_memory);
                memcpy(self->data, v.data(),
                       sizeof(double) * self->total_memory);
            })
        .def("load_data", &SparseMatrixGroup<S>::load_data, py::arg("filename"),
             py::arg("load_info") = false, py::arg("i_alloc") = nullptr)
        .def("save_data", &SparseMatrixGroup<S>::save_data, py::arg("filename"),
             py::arg("save_info") = false)
        .def("allocate",
             [](SparseMatrixGroup<S> *self,
                const vector<shared_ptr<SparseMatrixInfo<S>>> &infos) {
                 self->allocate(infos);
             })
        .def("deallocate", &SparseMatrixGroup<S>::deallocate)
        .def("deallocate_infos", &SparseMatrixGroup<S>::deallocate_infos)
        .def("delta_quanta", &SparseMatrixGroup<S>::delta_quanta)
        .def("randomize", &SparseMatrixGroup<S>::randomize, py::arg("a") = 0.0,
             py::arg("b") = 1.0)
        .def("norm", &SparseMatrixGroup<S>::norm)
        .def("iscale", &SparseMatrixGroup<S>::iscale, py::arg("d"))
        .def("normalize", &SparseMatrixGroup<S>::normalize)
        .def_static("normalize_all", &SparseMatrixGroup<S>::normalize_all)
        .def("left_svd",
             [](SparseMatrixGroup<S> *self) {
                 vector<S> qs;
                 vector<shared_ptr<Tensor>> s, r;
                 vector<vector<shared_ptr<Tensor>>> l;
                 self->left_svd(qs, l, s, r);
                 return make_tuple(qs, l, s, r);
             })
        .def("right_svd",
             [](SparseMatrixGroup<S> *self) {
                 vector<S> qs;
                 vector<shared_ptr<Tensor>> l, s;
                 vector<vector<shared_ptr<Tensor>>> r;
                 self->right_svd(qs, l, s, r);
                 return make_tuple(qs, l, s, r);
             })
        .def("__getitem__",
             [](SparseMatrixGroup<S> *self, int idx) { return (*self)[idx]; });

    py::bind_vector<vector<shared_ptr<SparseMatrixGroup<S>>>>(
        m, "VectorSpMatGroup");
}

template <typename S> void bind_operator(py::module &m) {
    py::class_<OperatorFunctions<S>, shared_ptr<OperatorFunctions<S>>>(
        m, "OperatorFunctions")
        .def_readwrite("cg", &OperatorFunctions<S>::cg)
        .def_readwrite("seq", &OperatorFunctions<S>::seq)
        .def(py::init<const shared_ptr<CG<S>> &>())
        .def("get_type", &OperatorFunctions<S>::get_type)
        .def("iadd", &OperatorFunctions<S>::iadd, py::arg("a"), py::arg("b"),
             py::arg("scale") = 1.0, py::arg("conj") = false)
        .def("tensor_rotate", &OperatorFunctions<S>::tensor_rotate,
             py::arg("a"), py::arg("c"), py::arg("rot_bra"), py::arg("rot_ket"),
             py::arg("trans"), py::arg("scale") = 1.0)
        .def("tensor_product_diagonal",
             &OperatorFunctions<S>::tensor_product_diagonal, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("opdq"),
             py::arg("scale") = 1.0)
        .def("tensor_partial_expectation",
             &OperatorFunctions<S>::tensor_partial_expectation, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("v"),
             py::arg("opdq"), py::arg("scale") = 1.0)
        .def("tensor_product_multiply",
             &OperatorFunctions<S>::tensor_product_multiply, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("v"),
             py::arg("opdq"), py::arg("scale") = 1.0,
             py::arg("tt") = TraceTypes::None)
        .def("tensor_product", &OperatorFunctions<S>::tensor_product,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("scale") = 1.0)
        .def("product", &OperatorFunctions<S>::product, py::arg("a"),
             py::arg("conj"), py::arg("b"), py::arg("c"),
             py::arg("scale") = 1.0)
        .def_static("trans_product", &OperatorFunctions<S>::trans_product,
                    py::arg("a"), py::arg("b"), py::arg("trace_right"),
                    py::arg("noise") = 0.0,
                    py::arg("noise_type") = NoiseTypes::DensityMatrix);

    py::class_<CSROperatorFunctions<S>, shared_ptr<CSROperatorFunctions<S>>,
               OperatorFunctions<S>>(m, "CSROperatorFunctions")
        .def(py::init<const shared_ptr<CG<S>> &>());

    py::class_<OperatorTensor<S>, shared_ptr<OperatorTensor<S>>>(
        m, "OperatorTensor")
        .def(py::init<>())
        .def_readwrite("lmat", &OperatorTensor<S>::lmat)
        .def_readwrite("rmat", &OperatorTensor<S>::rmat)
        .def_readwrite("ops", &OperatorTensor<S>::ops)
        .def("get_type", &OperatorTensor<S>::get_type)
        .def("reallocate", &OperatorTensor<S>::reallocate, py::arg("clean"))
        .def("deallocate", &OperatorTensor<S>::deallocate)
        .def("copy", &OperatorTensor<S>::copy)
        .def("deep_copy", &OperatorTensor<S>::deep_copy,
             py::arg("alloc") = nullptr);

    py::class_<DelayedOperatorTensor<S>, shared_ptr<DelayedOperatorTensor<S>>,
               OperatorTensor<S>>(m, "DelayedOperatorTensor")
        .def(py::init<>())
        .def_readwrite("dops", &DelayedOperatorTensor<S>::dops)
        .def_readwrite("mat", &DelayedOperatorTensor<S>::mat)
        .def_readwrite("lopt", &DelayedOperatorTensor<S>::lopt)
        .def_readwrite("ropt", &DelayedOperatorTensor<S>::ropt);

    py::bind_vector<vector<shared_ptr<OperatorTensor<S>>>>(m, "VectorOpTensor");

    py::class_<TensorFunctions<S>, shared_ptr<TensorFunctions<S>>>(
        m, "TensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S>> &>())
        .def_readwrite("opf", &TensorFunctions<S>::opf)
        .def("get_type", &TensorFunctions<S>::get_type)
        .def("left_assign", &TensorFunctions<S>::left_assign, py::arg("a"),
             py::arg("c"))
        .def("right_assign", &TensorFunctions<S>::right_assign, py::arg("a"),
             py::arg("c"))
        .def("left_contract", &TensorFunctions<S>::left_contract, py::arg("a"),
             py::arg("b"), py::arg("c"), py::arg("cexprs") = nullptr,
             py::arg("delayed") = OpNamesSet())
        .def("right_contract", &TensorFunctions<S>::right_contract,
             py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("cexprs") = nullptr, py::arg("delayed") = OpNamesSet())
        .def("tensor_product_multi_multiply",
             &TensorFunctions<S>::tensor_product_multi_multiply)
        .def("tensor_product_multiply",
             &TensorFunctions<S>::tensor_product_multiply)
        .def("tensor_product_diagonal",
             &TensorFunctions<S>::tensor_product_diagonal)
        .def("tensor_product", &TensorFunctions<S>::tensor_product)
        .def("delayed_left_contract",
             &TensorFunctions<S>::delayed_left_contract)
        .def("delayed_right_contract",
             &TensorFunctions<S>::delayed_right_contract)
        .def("left_rotate", &TensorFunctions<S>::left_rotate)
        .def("right_rotate", &TensorFunctions<S>::right_rotate)
        .def("intermediates", &TensorFunctions<S>::intermediates)
        .def("numerical_transform", &TensorFunctions<S>::numerical_transform)
        .def("post_numerical_transform",
             &TensorFunctions<S>::post_numerical_transform)
        .def("substitute_delayed_exprs",
             &TensorFunctions<S>::substitute_delayed_exprs)
        .def("delayed_contract",
             (shared_ptr<DelayedOperatorTensor<S>>(TensorFunctions<S>::*)(
                 const shared_ptr<OperatorTensor<S>> &,
                 const shared_ptr<OperatorTensor<S>> &,
                 const shared_ptr<OpExpr<S>> &, OpNamesSet delayed) const) &
                 TensorFunctions<S>::delayed_contract)
        .def("delayed_contract_simplified",
             (shared_ptr<DelayedOperatorTensor<S>>(TensorFunctions<S>::*)(
                 const shared_ptr<OperatorTensor<S>> &,
                 const shared_ptr<OperatorTensor<S>> &,
                 const shared_ptr<Symbolic<S>> &,
                 const shared_ptr<Symbolic<S>> &, OpNamesSet delayed) const) &
                 TensorFunctions<S>::delayed_contract);

    py::class_<ArchivedTensorFunctions<S>,
               shared_ptr<ArchivedTensorFunctions<S>>, TensorFunctions<S>>(
        m, "ArchivedTensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S>> &>())
        .def_readwrite("filename", &ArchivedTensorFunctions<S>::filename)
        .def_readwrite("offset", &ArchivedTensorFunctions<S>::offset)
        .def("archive_tensor", &ArchivedTensorFunctions<S>::archive_tensor,
             py::arg("a"));

    py::class_<DelayedTensorFunctions<S>, shared_ptr<DelayedTensorFunctions<S>>,
               TensorFunctions<S>>(m, "DelayedTensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S>> &>());
}

template <typename S> void bind_hamiltonian(py::module &m) {
    py::class_<Hamiltonian<S>, shared_ptr<Hamiltonian<S>>>(m, "Hamiltonian")
        .def(py::init<S, int, const vector<uint8_t> &>())
        .def_readwrite("n_syms", &Hamiltonian<S>::n_syms)
        .def_readwrite("opf", &Hamiltonian<S>::opf)
        .def_readwrite("n_sites", &Hamiltonian<S>::n_sites)
        .def_readwrite("orb_sym", &Hamiltonian<S>::orb_sym)
        .def_readwrite("vacuum", &Hamiltonian<S>::vacuum)
        .def_readwrite("basis", &Hamiltonian<S>::basis)
        .def_readwrite("site_op_infos", &Hamiltonian<S>::site_op_infos)
        .def_readwrite("delayed", &Hamiltonian<S>::delayed)
        .def("get_n_orbs_left", &Hamiltonian<S>::get_n_orbs_left)
        .def("get_n_orbs_right", &Hamiltonian<S>::get_n_orbs_right)
        .def("get_site_ops", &Hamiltonian<S>::get_site_ops)
        .def("filter_site_ops", &Hamiltonian<S>::filter_site_ops)
        .def("find_site_op_info", &Hamiltonian<S>::find_site_op_info)
        .def("deallocate", &Hamiltonian<S>::deallocate);
}

template <typename S> void bind_parallel(py::module &m) {

    py::class_<ParallelCommunicator<S>, shared_ptr<ParallelCommunicator<S>>>(
        m, "ParallelCommunicator")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def_readwrite("size", &ParallelCommunicator<S>::size)
        .def_readwrite("rank", &ParallelCommunicator<S>::rank)
        .def_readwrite("root", &ParallelCommunicator<S>::root)
        .def_readwrite("group", &ParallelCommunicator<S>::group)
        .def_readwrite("grank", &ParallelCommunicator<S>::grank)
        .def_readwrite("gsize", &ParallelCommunicator<S>::gsize)
        .def_readwrite("ngroup", &ParallelCommunicator<S>::ngroup)
        .def_readwrite("tcomm", &ParallelCommunicator<S>::tcomm)
        .def_readwrite("para_type", &ParallelCommunicator<S>::para_type)
        .def("get_parallel_type", &ParallelCommunicator<S>::get_parallel_type)
        .def("barrier", &ParallelCommunicator<S>::barrier)
        .def("split", &ParallelCommunicator<S>::split);

#ifdef _HAS_MPI
    py::class_<MPICommunicator<S>, shared_ptr<MPICommunicator<S>>,
               ParallelCommunicator<S>>(m, "MPICommunicator")
        .def(py::init<>())
        .def(py::init<int>());
#endif

    py::class_<ParallelRule<S>, shared_ptr<ParallelRule<S>>>(m, "ParallelRule")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>())
        .def_readwrite("comm", &ParallelRule<S>::comm)
        .def_readwrite("comm_type", &ParallelRule<S>::comm_type)
        .def("get_parallel_type", &ParallelRule<S>::get_parallel_type)
        .def("set_partition", &ParallelRule<S>::set_partition)
        .def("split", &ParallelRule<S>::split)
        .def("__call__", &ParallelRule<S>::operator())
        .def("is_root", &ParallelRule<S>::is_root)
        .def("available", &ParallelRule<S>::available)
        .def("own", &ParallelRule<S>::own)
        .def("owner", &ParallelRule<S>::owner)
        .def("repeat", &ParallelRule<S>::repeat)
        .def("partial", &ParallelRule<S>::partial);

    py::class_<ParallelTensorFunctions<S>,
               shared_ptr<ParallelTensorFunctions<S>>, TensorFunctions<S>>(
        m, "ParallelTensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S>> &,
                      const shared_ptr<ParallelRule<S>> &>());
}

template <typename S> void bind_rule(py::module &m) {

    py::class_<Rule<S>, shared_ptr<Rule<S>>>(m, "Rule")
        .def(py::init<>())
        .def("__call__", &Rule<S>::operator());

    py::class_<NoTransposeRule<S>, shared_ptr<NoTransposeRule<S>>, Rule<S>>(
        m, "NoTransposeRule")
        .def_readwrite("prim_rule", &NoTransposeRule<S>::prim_rule)
        .def(py::init<const shared_ptr<Rule<S>> &>());
}

template <typename S> void bind_core(py::module &m, const string &name) {

    bind_expr<S>(m);
    bind_state_info<S>(m, name);
    bind_sparse<S>(m);
    bind_cg<S>(m);
    bind_operator<S>(m);
    bind_hamiltonian<S>(m);
    bind_parallel<S>(m);
    bind_rule<S>(m);
}

template <typename S, typename T>
void bind_trans_state_info(py::module &m, const string &aux_name) {

    m.def(("trans_state_info_to_" + aux_name).c_str(),
          &TransStateInfo<S, T>::forward);
}

template <typename S, typename T>
auto bind_trans_state_info_spin_specific(py::module &m, const string &aux_name)
    -> decltype(typename S::is_su2_t(typename T::is_sz_t())) {

    m.def(("trans_connection_state_info_to_" + aux_name).c_str(),
          &TransStateInfo<T, S>::backward_connection);
}

template <typename S = void> void bind_data(py::module &m) {

    py::bind_vector<vector<int>>(m, "VectorInt");
    py::bind_vector<vector<char>>(m, "VectorChar");
    py::bind_vector<vector<long long int>>(m, "VectorLLInt");
    py::bind_vector<vector<pair<int, int>>>(m, "VectorPIntInt");
    py::bind_vector<vector<pair<long long int, int>>>(m, "VectorPLLIntInt");
    py::bind_vector<vector<pair<long long int, long long int>>>(
        m, "VectorPLLIntLLInt");
    py::bind_vector<vector<uint16_t>>(m, "VectorUInt16");
    py::bind_vector<vector<uint32_t>>(m, "VectorUInt32");
    py::bind_vector<vector<double>>(m, "VectorDouble");
    py::bind_vector<vector<long double>>(m, "VectorLDouble");
    py::bind_vector<vector<complex<double>>>(m, "VectorComplexDouble");
    py::bind_vector<vector<size_t>>(m, "VectorULInt");
    py::bind_vector<vector<vector<uint32_t>>>(m, "VectorVectorUInt32");
    py::bind_vector<vector<vector<double>>>(m, "VectorVectorDouble");
    py::bind_vector<vector<vector<int>>>(m, "VectorVectorInt");
    py::bind_vector<vector<vector<vector<double>>>>(m,
                                                    "VectorVectorVectorDouble");
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

    py::class_<array<int, 4>>(m, "Array4Int")
        .def("__setitem__",
             [](array<int, 4> *self, size_t i, int t) { (*self)[i] = t; })
        .def("__getitem__",
             [](array<int, 4> *self, size_t i) { return (*self)[i]; })
        .def("__len__", [](array<int, 4> *self) { return self->size(); })
        .def("__repr__",
             [](array<int, 4> *self) {
                 stringstream ss;
                 ss << "(LEN=" << self->size() << ")[";
                 for (auto x : *self)
                     ss << x << ",";
                 ss << "]";
                 return ss.str();
             })
        .def("__iter__", [](array<int, 4> *self) {
            return py::make_iterator<
                py::return_value_policy::reference_internal, int *, int *,
                int &>(&(*self)[0], &(*self)[0] + self->size());
        });

    py::bind_vector<vector<array<int, 4>>>(m, "VectorArray4Int");

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
    bind_array<uint32_t>(m, "ArrayUInt32");

    if (sizeof(ubond_t) == sizeof(uint8_t)) {
        m.attr("VectorUBond") = m.attr("VectorUInt8");
        m.attr("ArrayUBond") = m.attr("ArrayUInt8");
    } else if (sizeof(ubond_t) == sizeof(uint16_t)) {
        m.attr("VectorUBond") = m.attr("VectorUInt16");
        m.attr("ArrayUBond") = m.attr("ArrayUInt16");
    } else if (sizeof(ubond_t) == sizeof(uint32_t)) {
        m.attr("VectorUBond") = m.attr("VectorUInt32");
        m.attr("ArrayUBond") = m.attr("ArrayUInt32");
    }

    if (sizeof(MKL_INT) == sizeof(int))
        m.attr("VectorMKLInt") = m.attr("VectorInt");
    else if (sizeof(MKL_INT) == sizeof(long long int))
        m.attr("VectorMKLInt") = m.attr("VectorLLInt");

    py::bind_map<map<string, string>>(m, "MapStrStr");
}

template <typename S = void> void bind_types(py::module &m) {

    py::class_<PointGroup, shared_ptr<PointGroup>>(m, "PointGroup")
        .def_static("swap_c1", &PointGroup::swap_c1)
        .def_static("swap_ci", &PointGroup::swap_ci)
        .def_static("swap_cs", &PointGroup::swap_cs)
        .def_static("swap_c2", &PointGroup::swap_c2)
        .def_static("swap_c2h", &PointGroup::swap_c2h)
        .def_static("swap_c2v", &PointGroup::swap_c2v)
        .def_static("swap_d2", &PointGroup::swap_d2)
        .def_static("swap_d2h", &PointGroup::swap_d2h);

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
        .value("BD", OpNames::BD)
        .value("Q", OpNames::Q)
        .value("TR", OpNames::TR)
        .value("TS", OpNames::TS)
        .value("Zero", OpNames::Zero)
        .value("PDM1", OpNames::PDM1)
        .value("PDM2", OpNames::PDM2)
        .value("CCDD", OpNames::CCDD)
        .value("CCD", OpNames::CCD)
        .value("CDC", OpNames::CDC)
        .value("CDD", OpNames::CDD)
        .value("DCC", OpNames::DCC)
        .value("DCD", OpNames::DCD)
        .value("DDC", OpNames::DDC)
        .value("TEMP", OpNames::TEMP);

    py::enum_<DelayedOpNames>(m, "DelayedOpNames", py::arithmetic())
        .value("Nothing", DelayedOpNames::None)
        .value("H", DelayedOpNames::H)
        .value("Normal", DelayedOpNames::Normal)
        .value("R", DelayedOpNames::R)
        .value("RD", DelayedOpNames::RD)
        .value("P", DelayedOpNames::P)
        .value("PD", DelayedOpNames::PD)
        .value("Q", DelayedOpNames::Q)
        .value("CCDD", DelayedOpNames::CCDD)
        .value("CCD", DelayedOpNames::CCD)
        .value("CDD", DelayedOpNames::CDD)
        .value("TR", DelayedOpNames::TR)
        .value("TS", DelayedOpNames::TS)
        .def(py::self & py::self)
        .def(py::self | py::self);

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
        .value("Perturbative", NoiseTypes::Perturbative)
        .value("Collected", NoiseTypes::Collected)
        .value("Reduced", NoiseTypes::Reduced)
        .value("Unscaled", NoiseTypes::Unscaled)
        .value("LowMem", NoiseTypes::LowMem)
        .value("ReducedPerturbative", NoiseTypes::ReducedPerturbative)
        .value("ReducedPerturbativeUnscaled",
               NoiseTypes::ReducedPerturbativeUnscaled)
        .value("PerturbativeUnscaled", NoiseTypes::PerturbativeUnscaled)
        .value("PerturbativeCollected", NoiseTypes::PerturbativeCollected)
        .value("PerturbativeUnscaledCollected",
               NoiseTypes::PerturbativeUnscaledCollected)
        .value("ReducedPerturbativeCollected",
               NoiseTypes::ReducedPerturbativeCollected)
        .value("ReducedPerturbativeUnscaledCollected",
               NoiseTypes::ReducedPerturbativeUnscaledCollected)
        .value("ReducedPerturbativeLowMem",
               NoiseTypes::ReducedPerturbativeLowMem)
        .value("ReducedPerturbativeUnscaledLowMem",
               NoiseTypes::ReducedPerturbativeUnscaledLowMem)
        .value("ReducedPerturbativeCollectedLowMem",
               NoiseTypes::ReducedPerturbativeCollectedLowMem)
        .value("ReducedPerturbativeUnscaledCollectedLowMem",
               NoiseTypes::ReducedPerturbativeUnscaledCollectedLowMem)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<SeqTypes>(m, "SeqTypes", py::arithmetic())
        .value("Nothing", SeqTypes::None)
        .value("Simple", SeqTypes::Simple)
        .value("Auto", SeqTypes::Auto)
        .value("Tasked", SeqTypes::Tasked)
        .value("SimpleTasked", SeqTypes::SimpleTasked)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<DavidsonTypes>(m, "DavidsonTypes", py::arithmetic())
        .value("GreaterThan", DavidsonTypes::GreaterThan)
        .value("LessThan", DavidsonTypes::LessThan)
        .value("CloseTo", DavidsonTypes::CloseTo)
        .value("Harmonic", DavidsonTypes::Harmonic)
        .value("HarmonicGreaterThan", DavidsonTypes::HarmonicGreaterThan)
        .value("HarmonicLessThan", DavidsonTypes::HarmonicLessThan)
        .value("HarmonicCloseTo", DavidsonTypes::HarmonicCloseTo)
        .value("DavidsonPrecond", DavidsonTypes::DavidsonPrecond)
        .value("NoPrecond", DavidsonTypes::NoPrecond)
        .value("Normal", DavidsonTypes::Normal)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<SparseMatrixTypes>(m, "SparseMatrixTypes", py::arithmetic())
        .value("Normal", SparseMatrixTypes::Normal)
        .value("CSR", SparseMatrixTypes::CSR)
        .value("Archived", SparseMatrixTypes::Archived)
        .value("Delayed", SparseMatrixTypes::Delayed);

    py::enum_<ParallelOpTypes>(m, "ParallelOpTypes", py::arithmetic())
        .value("None", ParallelOpTypes::None)
        .value("Repeated", ParallelOpTypes::Repeated)
        .value("Number", ParallelOpTypes::Number)
        .value("Partial", ParallelOpTypes::Partial);

    py::enum_<TensorFunctionsTypes>(m, "TensorFunctionsTypes", py::arithmetic())
        .value("Normal", TensorFunctionsTypes::Normal)
        .value("Archived", TensorFunctionsTypes::Archived)
        .value("Delayed", TensorFunctionsTypes::Delayed);

    py::enum_<OperatorTensorTypes>(m, "OperatorTensorTypes", py::arithmetic())
        .value("Normal", OperatorTensorTypes::Normal)
        .value("Delayed", OperatorTensorTypes::Delayed);

    py::enum_<TraceTypes>(m, "TraceTypes", py::arithmetic())
        .value("Nothing", TraceTypes::None)
        .value("Left", TraceTypes::Left)
        .value("Right", TraceTypes::Right);

    py::enum_<ParallelCommTypes>(m, "ParallelCommTypes", py::arithmetic())
        .value("Nothing", ParallelCommTypes::None)
        .value("NonBlocking", ParallelCommTypes::NonBlocking)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<ParallelRulePartitionTypes>(m, "ParallelRulePartitionTypes",
                                          py::arithmetic())
        .value("Left", ParallelRulePartitionTypes::Left)
        .value("Right", ParallelRulePartitionTypes::Right)
        .value("Middle", ParallelRulePartitionTypes::Middle);

    py::enum_<ParallelTypes>(m, "ParallelTypes", py::arithmetic())
        .value("Serial", ParallelTypes::Serial)
        .value("Distributed", ParallelTypes::Distributed)
        .value("NewScheme", ParallelTypes::NewScheme)
        .def(py::self & py::self)
        .def(py::self | py::self)
        .def(py::self ^ py::self);
}

template <typename S = void> void bind_io(py::module &m) {

    m.def(
        "init_memory",
        [](size_t isize, size_t dsize, const string &save_dir,
           double dmain_ratio, double imain_ratio, int n_frames) {
            frame_() = make_shared<DataFrame>(
                isize, dsize, save_dir, dmain_ratio, imain_ratio, n_frames);
        },
        py::arg("isize") = size_t(1L << 28),
        py::arg("dsize") = size_t(1L << 30), py::arg("save_dir") = "nodex",
        py::arg("dmain_ratio") = 0.7, py::arg("imain_ratio") = 0.7,
        py::arg("n_frames") = 2);

    m.def("release_memory", []() {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    });

    m.def("set_mkl_num_threads", [](int n) {
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(n);
        mkl_set_dynamic(0);
        threading_()->n_threads_mkl = n;
        threading_()->type = threading_()->type | ThreadingTypes::BatchedGEMM;
#else
        throw runtime_error("cannot set number of mkl threads.");
#endif
    });
    m.def("set_omp_num_threads", [](int n) {
#ifdef _OPENMP
        omp_set_num_threads(n);
        threading_()->n_threads_global = n;
        threading_()->type = threading_()->type | ThreadingTypes::Global;
#else
        if(n != 1)
            throw runtime_error("cannot set number of omp threads.");
#endif
    });

    py::class_<Allocator<uint32_t>, shared_ptr<Allocator<uint32_t>>>(
        m, "IntAllocator")
        .def(py::init<>());

    py::class_<Allocator<double>, shared_ptr<Allocator<double>>>(
        m, "DoubleAllocator")
        .def(py::init<>());

    py::class_<VectorAllocator<uint32_t>, shared_ptr<VectorAllocator<uint32_t>>,
               Allocator<uint32_t>>(m, "IntVectorAllocator")
        .def_readwrite("data", &VectorAllocator<uint32_t>::data)
        .def(py::init<>());

    py::class_<VectorAllocator<double>, shared_ptr<VectorAllocator<double>>,
               Allocator<double>>(m, "DoubleVectorAllocator")
        .def_readwrite("data", &VectorAllocator<double>::data)
        .def(py::init<>());

    py::class_<StackAllocator<uint32_t>, shared_ptr<StackAllocator<uint32_t>>,
               Allocator<uint32_t>>(m, "IntStackAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<uint32_t>::size)
        .def_readwrite("used", &StackAllocator<uint32_t>::used)
        .def_readwrite("shift", &StackAllocator<uint32_t>::shift);

    py::class_<StackAllocator<double>, shared_ptr<StackAllocator<double>>,
               Allocator<double>>(m, "DoubleStackAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<double>::size)
        .def_readwrite("used", &StackAllocator<double>::used)
        .def_readwrite("shift", &StackAllocator<double>::shift);

    struct Global {};

    py::class_<FPCodec<double>, shared_ptr<FPCodec<double>>>(m, "DoubleFPCodec")
        .def(py::init<double>())
        .def(py::init<double, size_t>())
        .def_readwrite("ndata", &FPCodec<double>::ndata)
        .def_readwrite("ncpsd", &FPCodec<double>::ncpsd)
        .def("encode",
             [](FPCodec<double> *self, py::array_t<double> arr) {
                 double *tmp = new double[arr.size() + 2];
                 size_t len = self->encode(arr.mutable_data(), arr.size(), tmp);
                 assert(len <= arr.size() + 2);
                 py::array_t<double> arx = py::array_t<double>(len + 1);
                 arx.mutable_data()[0] = arr.size();
                 memcpy(arx.mutable_data() + 1, tmp, len * sizeof(double));
                 delete[] tmp;
                 return arx;
             })
        .def("decode",
             [](FPCodec<double> *self, py::array_t<double> arr) {
                 size_t arr_len = arr.mutable_data()[0];
                 py::array_t<double> arx = py::array_t<double>(arr_len);
                 size_t len = self->decode(arr.mutable_data() + 1, arr_len,
                                           arx.mutable_data());
                 assert(len == arr.size() - 1);
                 return arx;
             })
        .def("write_array",
             [](FPCodec<double> *self, py::array_t<double> arr) {
                 stringstream ss;
                 self->write_array(ss, arr.mutable_data(), arr.size());
                 assert(ss.tellp() % sizeof(double) == 0);
                 size_t len = ss.tellp() / sizeof(double);
                 py::array_t<double> arx = py::array_t<double>(len + 1);
                 arx.mutable_data()[0] = arr.size();
                 ss.clear();
                 ss.seekg(0);
                 ss.read((char *)(arx.mutable_data() + 1),
                         sizeof(double) * len);
                 return arx;
             })
        .def("read_array",
             [](FPCodec<double> *self, py::array_t<double> arr) {
                 size_t arr_len = arr.mutable_data()[0];
                 stringstream ss;
                 ss.write((char *)(arr.mutable_data() + 1),
                          (arr.size() - 1) * sizeof(double));
                 py::array_t<double> arx = py::array_t<double>(arr_len);
                 ss.clear();
                 ss.seekg(0);
                 self->read_array(ss, arx.mutable_data(), arr_len);
                 return arx;
             })
        .def("save",
             [](FPCodec<double> *self, const string &filename,
                py::array_t<double> arr) {
                 ofstream ofs(filename.c_str(), ios::binary);
                 if (!ofs.good())
                     throw runtime_error("DoubleFPCodec::save on '" + filename +
                                         "' failed.");
                 ofs << arr.size();
                 self->write_array(ofs, arr.mutable_data(), arr.size());
                 if (!ofs.good())
                     throw runtime_error("DoubleFPCodec::save on '" + filename +
                                         "' failed.");
                 ofs.close();
             })
        .def("load", [](FPCodec<double> *self, const string &filename) {
            ifstream ifs(filename.c_str(), ios::binary);
            if (!ifs.good())
                throw runtime_error("DoubleFPCodec::load_data on '" + filename +
                                    "' failed.");
            size_t arr_len;
            ifs >> arr_len;
            py::array_t<double> arx = py::array_t<double>(arr_len);
            self->read_array(ifs, arx.mutable_data(), arr_len);
            if (ifs.fail() || ifs.bad())
                throw runtime_error("DoubleFPCodec::load on '" + filename +
                                    "' failed.");
            ifs.close();
            return arx;
        });

    py::class_<KuhnMunkres, shared_ptr<KuhnMunkres>>(m, "KuhnMunkres")
        .def(py::init([](const py::array_t<double> &cost) {
            assert(cost.ndim() == 2);
            const int m = (int)cost.shape()[0], n = (int)cost.shape()[1];
            const int asi = (int)cost.strides()[0] / sizeof(double),
                      asj = (int)cost.strides()[1] / sizeof(double);
            vector<double> arr(m * n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    arr[i * n + j] = cost.data()[asi * i + asj * j];
            return make_shared<KuhnMunkres>(arr, m, n);
        }))
        .def("solve", [](KuhnMunkres *self) {
            auto r = self->solve();
            py::array_t<int> idx(r.second.size());
            memcpy(idx.mutable_data(), r.second.data(),
                   r.second.size() * sizeof(int));
            return make_pair(r.first, idx);
        });

    py::class_<Prime, shared_ptr<Prime>>(m, "Prime")
        .def(py::init<>())
        .def_readwrite("primes", &Prime::primes)
        .def("factors",
             [](Prime *self, Prime::LL n) {
                 vector<pair<Prime::LL, int>> factors;
                 self->factors(n, factors);
                 return factors;
             })
        .def_static("miller_rabin", &Prime::miller_rabin)
        .def("is_prime", &Prime::is_prime);

    py::class_<Combinatorics, shared_ptr<Combinatorics>>(m, "Combinatorics")
        .def(py::init<int>())
        .def("combination", &Combinatorics::combination);

    py::class_<FFT, shared_ptr<FFT>>(m, "FFT")
        .def(py::init<>())
        .def("init", &FFT::init)
        .def("fft",
             [](FFT *self, const py::array_t<complex<double>> &arr) {
                 py::array_t<complex<double>> arx =
                     py::array_t<complex<double>>(arr.size());
                 memcpy(arx.mutable_data(), arr.data(),
                        arr.size() * sizeof(complex<double>));
                 self->fft(arx.mutable_data(), arx.size(), true);
                 return arx;
             })
        .def("ifft",
             [](FFT *self, const py::array_t<complex<double>> &arr) {
                 py::array_t<complex<double>> arx =
                     py::array_t<complex<double>>(arr.size());
                 memcpy(arx.mutable_data(), arr.data(),
                        arr.size() * sizeof(complex<double>));
                 self->fft(arx.mutable_data(), arx.size(), false);
                 return arx;
             })
        .def_static("fftshift",
                    [](const py::array_t<complex<double>> &arr) {
                        py::array_t<complex<double>> arx =
                            py::array_t<complex<double>>(arr.size());
                        memcpy(arx.mutable_data(), arr.data(),
                               arr.size() * sizeof(complex<double>));
                        FFT::fftshift(arx.mutable_data(), arx.size(), true);
                        return arx;
                    })
        .def_static("fftshift",
                    [](const py::array_t<double> &arr) {
                        py::array_t<double> arx =
                            py::array_t<double>(arr.size());
                        memcpy(arx.mutable_data(), arr.data(),
                               arr.size() * sizeof(double));
                        FFT::fftshift(arx.mutable_data(), arx.size(), true);
                        return arx;
                    })
        .def_static("ifftshift",
                    [](const py::array_t<complex<double>> &arr) {
                        py::array_t<complex<double>> arx =
                            py::array_t<complex<double>>(arr.size());
                        memcpy(arx.mutable_data(), arr.data(),
                               arr.size() * sizeof(complex<double>));
                        FFT::fftshift(arx.mutable_data(), arx.size(), false);
                        return arx;
                    })
        .def_static("fftfreq", [](long long int n, double d) {
            py::array_t<double> arx = py::array_t<double>(n);
            FFT::fftfreq(arx.mutable_data(), n, d);
            return arx;
        });

    py::class_<DFT, shared_ptr<DFT>>(m, "DFT")
        .def(py::init<>())
        .def("init", &DFT::init)
        .def("fft",
             [](DFT *self, const py::array_t<complex<double>> &arr) {
                 py::array_t<complex<double>> arx =
                     py::array_t<complex<double>>(arr.size());
                 memcpy(arx.mutable_data(), arr.data(),
                        arr.size() * sizeof(complex<double>));
                 self->fft(arx.mutable_data(), arx.size(), true);
                 return arx;
             })
        .def("ifft", [](DFT *self, const py::array_t<complex<double>> &arr) {
            py::array_t<complex<double>> arx =
                py::array_t<complex<double>>(arr.size());
            memcpy(arx.mutable_data(), arr.data(),
                   arr.size() * sizeof(complex<double>));
            self->fft(arx.mutable_data(), arx.size(), false);
            return arx;
        });

    py::class_<DataFrame, shared_ptr<DataFrame>>(m, "DataFrame")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, const string &>())
        .def(py::init<size_t, size_t, const string &, double>())
        .def(py::init<size_t, size_t, const string &, double, double>())
        .def(py::init<size_t, size_t, const string &, double, double, int>())
        .def_readwrite("save_dir", &DataFrame::save_dir)
        .def_readwrite("mps_dir", &DataFrame::mps_dir)
        .def_readwrite("restart_dir", &DataFrame::restart_dir)
        .def_readwrite("restart_dir_per_sweep",
                       &DataFrame::restart_dir_per_sweep)
        .def_readwrite("restart_dir_optimal_mps",
                       &DataFrame::restart_dir_optimal_mps)
        .def_readwrite("restart_dir_optimal_mps_per_sweep",
                       &DataFrame::restart_dir_optimal_mps_per_sweep)
        .def_readwrite("prefix", &DataFrame::prefix)
        .def_readwrite("prefix_distri", &DataFrame::prefix_distri)
        .def_readwrite("prefix_can_write", &DataFrame::prefix_can_write)
        .def_readwrite("partition_can_write", &DataFrame::partition_can_write)
        .def_readwrite("isize", &DataFrame::isize)
        .def_readwrite("dsize", &DataFrame::dsize)
        .def_readwrite("tread", &DataFrame::tread)
        .def_readwrite("twrite", &DataFrame::twrite)
        .def_readwrite("tasync", &DataFrame::tasync)
        .def_readwrite("fpread", &DataFrame::fpread)
        .def_readwrite("fpwrite", &DataFrame::fpwrite)
        .def_readwrite("n_frames", &DataFrame::n_frames)
        .def_readwrite("i_frame", &DataFrame::i_frame)
        .def_readwrite("iallocs", &DataFrame::iallocs)
        .def_readwrite("dallocs", &DataFrame::dallocs)
        .def_readwrite("peak_used_memory", &DataFrame::peak_used_memory)
        .def_readwrite("load_buffering", &DataFrame::load_buffering)
        .def_readwrite("save_buffering", &DataFrame::save_buffering)
        .def_readwrite("use_main_stack", &DataFrame::use_main_stack)
        .def_readwrite("minimal_disk_usage", &DataFrame::minimal_disk_usage)
        .def_readwrite("fp_codec", &DataFrame::fp_codec)
        .def("update_peak_used_memory", &DataFrame::update_peak_used_memory)
        .def("reset_peak_used_memory", &DataFrame::reset_peak_used_memory)
        .def("activate", &DataFrame::activate)
        .def("load_data", &DataFrame::load_data)
        .def("save_data", &DataFrame::save_data)
        .def("reset", &DataFrame::reset)
        .def("__repr__", [](DataFrame *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::enum_<ThreadingTypes>(m, "ThreadingTypes", py::arithmetic())
        .value("SequentialGEMM", ThreadingTypes::SequentialGEMM)
        .value("BatchedGEMM", ThreadingTypes::BatchedGEMM)
        .value("Quanta", ThreadingTypes::Quanta)
        .value("QuantaBatchedGEMM", ThreadingTypes::QuantaBatchedGEMM)
        .value("Operator", ThreadingTypes::Operator)
        .value("OperatorBatchedGEMM", ThreadingTypes::OperatorBatchedGEMM)
        .value("OperatorQuanta", ThreadingTypes::OperatorQuanta)
        .value("OperatorQuantaBatchedGEMM",
               ThreadingTypes::OperatorQuantaBatchedGEMM)
        .value("Global", ThreadingTypes::Global)
        .def(py::self & py::self)
        .def(py::self ^ py::self)
        .def(py::self | py::self);

    py::class_<Threading, shared_ptr<Threading>>(m, "Threading")
        .def(py::init<>())
        .def(py::init<ThreadingTypes>())
        .def(py::init<ThreadingTypes, int>())
        .def(py::init<ThreadingTypes, int, int>())
        .def(py::init<ThreadingTypes, int, int, int>())
        .def(py::init<ThreadingTypes, int, int, int, int>())
        .def_readwrite("type", &Threading::type)
        .def_readwrite("seq_type", &Threading::seq_type)
        .def_readwrite("n_threads_op", &Threading::n_threads_op)
        .def_readwrite("n_threads_quanta", &Threading::n_threads_quanta)
        .def_readwrite("n_threads_mkl", &Threading::n_threads_mkl)
        .def_readwrite("n_threads_global", &Threading::n_threads_global)
        .def_readwrite("n_levels", &Threading::n_levels)
        .def("openmp_available", &Threading::openmp_available)
        .def("mkl_available", &Threading::mkl_available)
        .def("tbb_available", &Threading::tbb_available)
        .def("activate_global", &Threading::activate_global)
        .def("activate_normal", &Threading::activate_normal)
        .def("activate_operator", &Threading::activate_operator)
        .def("activate_quanta", &Threading::activate_quanta)
        .def("__repr__", [](Threading *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<StackAllocator<uint32_t>>>>(
        m, "VectorIntStackAllocator");
    py::bind_vector<vector<shared_ptr<StackAllocator<double>>>>(
        m, "VectorDoubleStackAllocator");

    py::class_<Global>(m, "Global")
        .def_property_static(
            "ialloc", [](py::object) { return ialloc_(); },
            [](py::object, shared_ptr<StackAllocator<uint32_t>> ia) {
                ialloc_() = ia;
            })
        .def_property_static(
            "dalloc", [](py::object) { return dalloc_(); },
            [](py::object, shared_ptr<StackAllocator<double>> da) {
                dalloc_() = da;
            })
        .def_property_static(
            "frame", [](py::object) { return frame_(); },
            [](py::object, shared_ptr<DataFrame> fr) { frame_() = fr; })
        .def_property_static(
            "threading", [](py::object) { return threading_(); },
            [](py::object, shared_ptr<Threading> th) { threading_() = th; });

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

    py::class_<Parsing, shared_ptr<Parsing>>(m, "Parsing")
        .def_static("to_size_string", &Parsing::to_size_string,
                    py::arg("i") = (size_t)0U, py::arg("suffix") = "B");

    py::class_<ParallelProperty, shared_ptr<ParallelProperty>>(
        m, "ParallelProperty")
        .def_readwrite("owner", &ParallelProperty::owner)
        .def_readwrite("ptype", &ParallelProperty::ptype)
        .def(py::init<>())
        .def(py::init<int, ParallelOpTypes>());

#ifdef _HAS_MPI
    py::class_<block2::MPI, shared_ptr<block2::MPI>>(m, "MPI")
        .def_readwrite("_rank", &block2::MPI::_rank)
        .def_readwrite("_size", &block2::MPI::_size)
        .def(py::init<>())
        .def_static("mpi", &block2::MPI::mpi)
        .def_static("rank", &block2::MPI::rank)
        .def_static("size", &block2::MPI::size);
#endif
}

template <typename S = void> void bind_matrix(py::module &m) {
    py::class_<MatrixRef, shared_ptr<MatrixRef>>(m, "Matrix",
                                                 py::buffer_protocol())
        .def(py::init([](py::array_t<double> mat) {
                 assert(mat.ndim() == 2);
                 assert(mat.strides()[1] == sizeof(double));
                 return MatrixRef(mat.mutable_data(), mat.shape()[0],
                                  mat.shape()[1]);
             }),
             py::keep_alive<0, 1>())
        .def_buffer([](MatrixRef *self) -> py::buffer_info {
            return py::buffer_info(
                self->data, sizeof(double),
                py::format_descriptor<double>::format(), 2,
                {(ssize_t)self->m, (ssize_t)self->n},
                {sizeof(double) * (ssize_t)self->n, sizeof(double)});
        })
        .def_readwrite("m", &MatrixRef::m)
        .def_readwrite("n", &MatrixRef::n)
        .def("__repr__",
             [](MatrixRef *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("allocate", &MatrixRef::allocate, py::arg("alloc") = nullptr)
        .def("deallocate", &MatrixRef::deallocate, py::arg("alloc") = nullptr);

    py::class_<ComplexMatrixRef, shared_ptr<ComplexMatrixRef>>(
        m, "ComplexMatrix", py::buffer_protocol())
        .def(py::init([](py::array_t<complex<double>> mat) {
                 assert(mat.ndim() == 2);
                 assert(mat.strides()[1] == sizeof(complex<double>));
                 return ComplexMatrixRef(mat.mutable_data(), mat.shape()[0],
                                         mat.shape()[1]);
             }),
             py::keep_alive<0, 1>())
        .def_buffer([](ComplexMatrixRef *self) -> py::buffer_info {
            return py::buffer_info(
                self->data, sizeof(complex<double>),
                py::format_descriptor<complex<double>>::format(), 2,
                {(ssize_t)self->m, (ssize_t)self->n},
                {sizeof(complex<double>) * (ssize_t)self->n,
                 sizeof(complex<double>)});
        })
        .def_readwrite("m", &ComplexMatrixRef::m)
        .def_readwrite("n", &ComplexMatrixRef::n)
        .def("__repr__",
             [](ComplexMatrixRef *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("allocate", &ComplexMatrixRef::allocate,
             py::arg("alloc") = nullptr)
        .def("deallocate", &ComplexMatrixRef::deallocate,
             py::arg("alloc") = nullptr);

    py::class_<CSRMatrixRef, shared_ptr<CSRMatrixRef>>(m, "CSRMatrix")
        .def(py::init<>())
        .def(py::init<MKL_INT, MKL_INT>())
        .def_readwrite("m", &CSRMatrixRef::m)
        .def_readwrite("n", &CSRMatrixRef::n)
        .def_readwrite("nnz", &CSRMatrixRef::nnz)
        .def_property(
            "data",
            [](CSRMatrixRef *self) {
                return py::array_t<double>(self->nnz, self->data);
            },
            [](CSRMatrixRef *self, const py::array_t<double> &v) {
                assert(v.size() == self->nnz);
                memcpy(self->data, v.data(), sizeof(double) * self->nnz);
            })
        .def_property(
            "rows",
            [](CSRMatrixRef *self) {
                return py::array_t<MKL_INT>(self->m + 1, self->rows);
            },
            [](CSRMatrixRef *self, const py::array_t<MKL_INT> &v) {
                assert(v.size() == self->m + 1);
                memcpy(self->rows, v.data(), sizeof(MKL_INT) * (self->m + 1));
            })
        .def_property(
            "cols",
            [](CSRMatrixRef *self) {
                return py::array_t<MKL_INT>(self->nnz, self->cols);
            },
            [](CSRMatrixRef *self, const py::array_t<MKL_INT> &v) {
                assert(v.size() == self->nnz);
                memcpy(self->cols, v.data(), sizeof(MKL_INT) * self->nnz);
            })
        .def("__repr__",
             [](CSRMatrixRef *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("size", &CSRMatrixRef::size)
        .def("memory_size", &CSRMatrixRef::memory_size)
        .def("transpose", &CSRMatrixRef::transpose)
        .def("sparsity", &CSRMatrixRef::sparsity)
        .def("deep_copy", &CSRMatrixRef::deep_copy)
        .def("from_dense", &CSRMatrixRef::from_dense)
        .def("to_dense",
             [](CSRMatrixRef *self, py::array_t<double> v) {
                 assert(v.size() == self->size());
                 MatrixRef mat(v.mutable_data(), self->m, self->n);
                 self->to_dense(mat);
             })
        .def("diag", &CSRMatrixRef::diag)
        .def("trace", &CSRMatrixRef::trace)
        .def("allocate", &CSRMatrixRef::allocate)
        .def("deallocate", &CSRMatrixRef::deallocate);

    py::bind_vector<vector<shared_ptr<CSRMatrixRef>>>(m, "VectorCSRMatrix");

    py::class_<Tensor, shared_ptr<Tensor>>(m, "Tensor", py::buffer_protocol())
        .def(py::init<MKL_INT, MKL_INT, MKL_INT>())
        .def(py::init<const vector<MKL_INT> &>())
        .def_buffer([](Tensor *self) -> py::buffer_info {
            vector<ssize_t> shape, strides;
            for (auto x : self->shape)
                shape.push_back(x);
            strides.push_back(sizeof(double));
            for (int i = (int)shape.size() - 1; i > 0; i--)
                strides.push_back(strides.back() * shape[i]);
            reverse(strides.begin(), strides.end());
            return py::buffer_info(&self->data[0], sizeof(double),
                                   py::format_descriptor<double>::format(),
                                   shape.size(), shape, strides);
        })
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("data", &Tensor::data)
        .def_property_readonly("ref", &Tensor::ref)
        .def("__repr__", [](Tensor *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<Tensor>>>(m, "VectorTensor");
    py::bind_vector<vector<vector<shared_ptr<Tensor>>>>(m,
                                                        "VectorVectorTensor");

    py::class_<MatrixFunctions>(m, "MatrixFunctions")
        .def_static("det",
                    [](py::array_t<double> &a) {
                        MKL_INT n = (MKL_INT)Prime::sqrt((Prime::LL)a.size());
                        assert(n * n == (MKL_INT)a.size());
                        return MatrixFunctions::det(
                            MatrixRef(a.mutable_data(), n, n));
                    })
        .def_static("eigs",
                    [](py::array_t<double> &a, py::array_t<double> &w) {
                        MKL_INT n = (MKL_INT)w.size();
                        MatrixFunctions::eigs(
                            MatrixRef(a.mutable_data(), n, n),
                            DiagonalMatrix(w.mutable_data(), n));
                    })
        .def_static("block_eigs", [](py::array_t<double> &a,
                                     py::array_t<double> &w,
                                     const vector<uint8_t> &x) {
            MKL_INT n = (MKL_INT)w.size();
            MatrixFunctions::block_eigs(MatrixRef(a.mutable_data(), n, n),
                                        DiagonalMatrix(w.mutable_data(), n), x);
        });

    py::class_<ComplexMatrixFunctions>(m, "ComplexMatrixFunctions");

    py::class_<CSRMatrixFunctions>(m, "CSRMatrixFunctions");

    py::class_<DiagonalMatrix, shared_ptr<DiagonalMatrix>>(
        m, "DiagonalMatrix", py::buffer_protocol())
        .def_buffer([](DiagonalMatrix *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(double),
                                   py::format_descriptor<double>::format(), 1,
                                   {(ssize_t)self->n}, {sizeof(double)});
        });

    py::class_<FCIDUMP, shared_ptr<FCIDUMP>>(m, "FCIDUMP")
        .def(py::init<>())
        .def("read", &FCIDUMP::read)
        .def("write", &FCIDUMP::write)
        .def("initialize_h1e",
             [](FCIDUMP *self, uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                uint16_t isym, double e, const py::array_t<double> &t) {
                 self->initialize_h1e(n_sites, n_elec, twos, isym, e, t.data(),
                                      t.size());
             })
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
        .def("symmetrize", &FCIDUMP::symmetrize, py::arg("orbsym"),
             "Remove integral elements that violate point group symmetry. "
             "Returns summed error in symmetrization\n\n"
             "    Args:\n"
             "        orbsym : in XOR convention")
        .def("e", &FCIDUMP::e)
        .def(
            "t",
            [](FCIDUMP *self, py::args &args) -> double {
                assert(args.size() == 2 || args.size() == 3);
                if (args.size() == 2)
                    return self->t((uint16_t)args[0].cast<int>(),
                                   (uint16_t)args[1].cast<int>());
                else
                    return self->t((uint8_t)args[0].cast<int>(),
                                   (uint16_t)args[1].cast<int>(),
                                   (uint16_t)args[2].cast<int>());
            },
            "1. (i: int, j: int) -> float\n"
            "    One-electron integral element (SU(2));\n"
            "2. (s: int, i: int, j: int) -> float\n"
            "    One-electron integral element (SZ).\n\n"
            "    Args:\n"
            "        i, j : spatial indices\n"
            "        s : spin index (0=alpha, 1=beta)")
        .def(
            "v",
            [](FCIDUMP *self, py::args &args) -> double {
                assert(args.size() == 4 || args.size() == 6);
                if (args.size() == 4)
                    return self->v((uint16_t)args[0].cast<int>(),
                                   (uint16_t)args[1].cast<int>(),
                                   (uint16_t)args[2].cast<int>(),
                                   (uint16_t)args[3].cast<int>());
                else
                    return self->v((uint8_t)args[0].cast<int>(),
                                   (uint8_t)args[1].cast<int>(),
                                   (uint16_t)args[2].cast<int>(),
                                   (uint16_t)args[3].cast<int>(),
                                   (uint16_t)args[4].cast<int>(),
                                   (uint16_t)args[5].cast<int>());
            },
            "1. (i: int, j: int, k: int, l: int) -> float\n"
            "    Two-electron integral element (SU(2));\n"
            "2. (sij: int, skl: int, i: int, j: int, k: int, l: int) -> float\n"
            "    Two-electron integral element (SZ).\n\n"
            "    Args:\n"
            "        i, j, k, l : spatial indices\n"
            "        sij, skl : spin indices (0=alpha, 1=beta)")
        .def("det_energy", &FCIDUMP::det_energy)
        .def("exchange_matrix", &FCIDUMP::exchange_matrix)
        .def("abs_exchange_matrix", &FCIDUMP::abs_exchange_matrix)
        .def("h1e_matrix", &FCIDUMP::h1e_matrix)
        .def("abs_h1e_matrix", &FCIDUMP::abs_h1e_matrix)
        .def("reorder",
             (void (FCIDUMP::*)(const vector<uint16_t> &)) & FCIDUMP::reorder)
        .def("rotate", &FCIDUMP::rotate)
        .def("deep_copy", &FCIDUMP::deep_copy)
        .def_static("array_reorder", &FCIDUMP::reorder<double>)
        .def_static("array_reorder", &FCIDUMP::reorder<uint8_t>)
        .def_property("orb_sym", &FCIDUMP::orb_sym, &FCIDUMP::set_orb_sym,
                      "Orbital symmetry in molpro convention")
        .def_property_readonly("n_elec", &FCIDUMP::n_elec)
        .def_property_readonly("twos", &FCIDUMP::twos)
        .def_property_readonly("isym", &FCIDUMP::isym)
        .def_property_readonly("n_sites", &FCIDUMP::n_sites)
        .def_readwrite("params", &FCIDUMP::params)
        .def_readwrite("ts", &FCIDUMP::ts)
        .def_readwrite("vs", &FCIDUMP::vs)
        .def_readwrite("vabs", &FCIDUMP::vabs)
        .def_readwrite("vgs", &FCIDUMP::vgs)
        .def_readwrite("const_e", &FCIDUMP::const_e)
        .def_readwrite("total_memory", &FCIDUMP::total_memory)
        .def_readwrite("uhf", &FCIDUMP::uhf);

    py::class_<CompressedFCIDUMP, shared_ptr<CompressedFCIDUMP>, FCIDUMP>(
        m, "CompressedFCIDUMP")
        .def(py::init<double>())
        .def(py::init<double, int>())
        .def(py::init<double, int, size_t>())
        .def("freeze", &CompressedFCIDUMP::freeze)
        .def("unfreeze", &CompressedFCIDUMP::unfreeze)
        .def_readwrite("prec", &CompressedFCIDUMP::prec)
        .def_readwrite("ncache", &CompressedFCIDUMP::ncache)
        .def_readwrite("chunk_size", &CompressedFCIDUMP::chunk_size)
        .def_readwrite("cps_ts", &CompressedFCIDUMP::cps_ts)
        .def_readwrite("cps_vs", &CompressedFCIDUMP::cps_vs)
        .def_readwrite("cps_vabs", &CompressedFCIDUMP::cps_vabs)
        .def_readwrite("cps_vgs", &CompressedFCIDUMP::cps_vgs)
        .def("initialize_su2",
             [](CompressedFCIDUMP *self, uint16_t n_sites, uint16_t n_elec,
                uint16_t twos, uint16_t isym, double e,
                const py::array_t<double> &t, const py::array_t<double> &v) {
                 stringstream st, sv;
                 st.write((char *)t.data(), sizeof(double) * t.size());
                 st.clear(), st.seekg(0);
                 sv.write((char *)v.data(), sizeof(double) * v.size());
                 sv.clear(), sv.seekg(0);
                 self->initialize_su2(n_sites, n_elec, twos, isym, e, st,
                                      t.size(), sv, v.size());
             })
        .def("initialize_su2",
             [](CompressedFCIDUMP *self, uint16_t n_sites, uint16_t n_elec,
                uint16_t twos, uint16_t isym, double e, const string &ft,
                const string &fv) {
                 ifstream ift(ft.c_str(), ios::binary);
                 ifstream ifv(fv.c_str(), ios::binary);
                 if (!ift.good())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + ft +
                         "' failed.");
                 if (!ifv.good())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + fv +
                         "' failed.");
                 size_t t_len, v_len;
                 ift >> t_len;
                 ifv >> v_len;
                 self->initialize_su2(n_sites, n_elec, twos, isym, e, ift,
                                      t_len, ifv, v_len);
                 if (ift.fail() || ift.bad())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + ft +
                         "' failed.");
                 if (ifv.fail() || ifv.bad())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + fv +
                         "' failed.");
                 ifv.close();
                 ift.close();
             })
        .def("initialize_sz", [](CompressedFCIDUMP *self, uint16_t n_sites,
                                 uint16_t n_elec, uint16_t twos, uint16_t isym,
                                 double e, const py::tuple &t,
                                 const py::tuple &v) {
            assert(t.size() == 2 && v.size() == 3);
            if (py::isinstance<py::array_t<double>>(t[0])) {
                py::array_t<double> ta = t[0].cast<py::array_t<double>>();
                py::array_t<double> tb = t[1].cast<py::array_t<double>>();
                py::array_t<double> va = v[0].cast<py::array_t<double>>();
                py::array_t<double> vb = v[1].cast<py::array_t<double>>();
                py::array_t<double> vab = v[2].cast<py::array_t<double>>();
                stringstream sta, stb, sva, svb, svab;
                sta.write((char *)ta.data(), sizeof(double) * ta.size());
                sta.clear(), sta.seekg(0);
                stb.write((char *)tb.data(), sizeof(double) * tb.size());
                stb.clear(), stb.seekg(0);
                sva.write((char *)va.data(), sizeof(double) * va.size());
                sva.clear(), sva.seekg(0);
                svb.write((char *)vb.data(), sizeof(double) * vb.size());
                svb.clear(), svb.seekg(0);
                svab.write((char *)vab.data(), sizeof(double) * vab.size());
                svab.clear(), svab.seekg(0);
                self->initialize_sz(n_sites, n_elec, twos, isym, e, sta,
                                    ta.size(), stb, tb.size(), sva, va.size(),
                                    svb, vb.size(), svab, vab.size());
            } else {
                string fta = t[0].cast<string>();
                string ftb = t[1].cast<string>();
                string fva = v[0].cast<string>();
                string fvb = v[1].cast<string>();
                string fvab = v[2].cast<string>();
                ifstream ifta(fta.c_str(), ios::binary);
                ifstream iftb(ftb.c_str(), ios::binary);
                ifstream ifva(fva.c_str(), ios::binary);
                ifstream ifvb(fvb.c_str(), ios::binary);
                ifstream ifvab(fvab.c_str(), ios::binary);
                if (!ifta.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fta +
                        "' failed.");
                if (!iftb.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + ftb +
                        "' failed.");
                if (!ifva.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fva +
                        "' failed.");
                if (!ifvb.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvb +
                        "' failed.");
                if (!ifvab.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvab +
                        "' failed.");
                size_t ta_len, tb_len, va_len, vb_len, vab_len;
                ifta >> ta_len;
                iftb >> tb_len;
                ifva >> va_len;
                ifvb >> vb_len;
                ifvab >> vab_len;
                self->initialize_sz(n_sites, n_elec, twos, isym, e, ifta,
                                    ta_len, iftb, tb_len, ifva, va_len, ifvb,
                                    vb_len, ifvab, vab_len);
                if (ifta.fail() || ifta.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fta +
                        "' failed.");
                if (iftb.fail() || iftb.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + ftb +
                        "' failed.");
                if (ifva.fail() || ifva.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fva +
                        "' failed.");
                if (ifvb.fail() || ifvb.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvb +
                        "' failed.");
                if (ifvab.fail() || ifvab.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvab +
                        "' failed.");
                ifvab.close();
                ifvb.close();
                ifva.close();
                iftb.close();
                ifta.close();
            }
        });

    py::class_<DyallFCIDUMP, shared_ptr<DyallFCIDUMP>, FCIDUMP>(m,
                                                                "DyallFCIDUMP")
        .def(py::init<const shared_ptr<FCIDUMP> &, uint16_t, uint16_t>())
        .def_readwrite("fcidump", &DyallFCIDUMP::fcidump)
        .def_readwrite("n_inactive", &DyallFCIDUMP::n_inactive)
        .def_readwrite("n_virtual", &DyallFCIDUMP::n_virtual)
        .def_readwrite("n_active", &DyallFCIDUMP::n_active)
        .def("initialize_su2",
             [](DyallFCIDUMP *self, const py::array_t<double> &f) {
                 self->initialize_su2(f.data(), f.size());
             })
        .def("initialize_sz",
             [](DyallFCIDUMP *self, const py::array_t<double> &fa,
                const py::array_t<double> &fb) {
                 self->initialize_sz(fa.data(), fa.size(), fb.data(),
                                     fb.size());
             })
        .def("initialize_from_1pdm_su2",
             [](DyallFCIDUMP *self, py::array_t<double> &dm) {
                 assert(dm.ndim() == 2);
                 assert(dm.strides()[1] == sizeof(double));
                 MatrixRef mr(dm.mutable_data(), dm.shape()[0], dm.shape()[1]);
                 self->initialize_from_1pdm_su2(mr);
             })
        .def("initialize_from_1pdm_sz",
             [](DyallFCIDUMP *self, py::array_t<double> &dm) {
                 assert(dm.ndim() == 2);
                 assert(dm.strides()[1] == sizeof(double));
                 MatrixRef mr(dm.mutable_data(), dm.shape()[0], dm.shape()[1]);
                 self->initialize_from_1pdm_sz(mr);
             });

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
        .def("multiply", &BatchGEMMSeq::multiply, py::arg("a"),
             py::arg("conja"), py::arg("b"), py::arg("conjb"), py::arg("c"),
             py::arg("scale"), py::arg("cfactor"))
        .def("rotate",
             (void (BatchGEMMSeq::*)(const MatrixRef &, const MatrixRef &,
                                     const MatrixRef &, bool, const MatrixRef &,
                                     bool, double)) &
                 BatchGEMMSeq::rotate,
             py::arg("a"), py::arg("c"), py::arg("bra"), py::arg("conj_bra"),
             py::arg("ket"), py::arg("conj_ket"), py::arg("scale"))
        .def("rotate",
             (void (BatchGEMMSeq::*)(const MatrixRef &, bool, const MatrixRef &,
                                     bool, const MatrixRef &, const MatrixRef &,
                                     double)) &
                 BatchGEMMSeq::rotate,
             py::arg("a"), py::arg("conj_a"), py::arg("c"), py::arg("conj_c"),
             py::arg("bra"), py::arg("ket"), py::arg("scale"))
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
        .def("auto_perform",
             (void (BatchGEMMSeq::*)(const MatrixRef &)) &
                 BatchGEMMSeq::auto_perform,
             py::arg("v") = MatrixRef(nullptr, 0, 0))
        .def("auto_perform",
             (void (BatchGEMMSeq::*)(const vector<MatrixRef> &)) &
                 BatchGEMMSeq::auto_perform,
             py::arg("vs"))
        .def("perform", &BatchGEMMSeq::perform)
        .def("clear", &BatchGEMMSeq::clear)
        .def("__call__", &BatchGEMMSeq::operator())
        .def("__repr__", [](BatchGEMMSeq *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });
}

template <typename S = void> void bind_symmetry(py::module &m) {

    py::class_<OpNamesSet>(m, "OpNamesSet")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init([](py::tuple names) {
            vector<OpNames> x(names.size());
            for (size_t i = 0; i < names.size(); i++)
                x[i] = names[i].cast<OpNames>();
            return OpNamesSet(x);
        }))
        .def_readwrite("data", &OpNamesSet::data)
        .def("__call__", &OpNamesSet::operator())
        .def("empty", &OpNamesSet::empty)
        .def_static("normal_ops", &OpNamesSet::normal_ops)
        .def_static("all_ops", &OpNamesSet::all_ops);

    py::class_<SiteIndex>(m, "SiteIndex")
        .def(py::init<>())
        .def(py::init<uint16_t>())
        .def(py::init<uint16_t, uint16_t, uint8_t>())
        .def(py::init([](py::tuple idxs, py::tuple sidxs) {
            vector<uint16_t> x(idxs.size());
            vector<uint8_t> sx(sidxs.size());
            for (size_t i = 0; i < idxs.size(); i++)
                x[i] = idxs[i].cast<uint16_t>();
            for (size_t i = 0; i < sidxs.size(); i++)
                sx[i] = sidxs[i].cast<uint8_t>();
            return SiteIndex(x, sx);
        }))
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
        .def_property_readonly_static("invalid",
                                      [](SZ *self) { return SZ::invalid; })
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
        .def("__hash__", &SZ::hash)
        .def("__repr__", &SZ::to_str);

    py::bind_vector<vector<SZ>>(m, "VectorSZ");

    py::class_<SU2>(m, "SU2")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int>())
        .def_property_readonly_static("invalid",
                                      [](SU2 *self) { return SU2::invalid; })
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
        .def("__hash__", &SU2::hash)
        .def("__repr__", &SU2::to_str);

    py::bind_vector<vector<SU2>>(m, "VectorSU2");
}

#ifdef _EXPLICIT_TEMPLATE

extern template void bind_data<>(py::module &m);
extern template void bind_types<>(py::module &m);
extern template void bind_io<>(py::module &m);
extern template void bind_matrix<>(py::module &m);
extern template void bind_symmetry<>(py::module &m);

extern template void bind_cg<SZ>(py::module &m);
extern template void bind_expr<SZ>(py::module &m);
extern template void bind_state_info<SZ>(py::module &m, const string &name);
extern template void bind_sparse<SZ>(py::module &m);
extern template void bind_operator<SZ>(py::module &m);
extern template void bind_hamiltonian<SZ>(py::module &m);
extern template void bind_parallel<SZ>(py::module &m);
extern template void bind_rule<SZ>(py::module &m);

extern template void bind_cg<SU2>(py::module &m);
extern template void bind_expr<SU2>(py::module &m);
extern template void bind_state_info<SU2>(py::module &m, const string &name);
extern template void bind_sparse<SU2>(py::module &m);
extern template void bind_operator<SU2>(py::module &m);
extern template void bind_hamiltonian<SU2>(py::module &m);
extern template void bind_parallel<SU2>(py::module &m);
extern template void bind_rule<SU2>(py::module &m);

extern template void bind_trans_state_info<SU2, SZ>(py::module &m,
                                                    const string &aux_name);
extern template void bind_trans_state_info<SZ, SU2>(py::module &m,
                                                    const string &aux_name);
extern template auto
bind_trans_state_info_spin_specific<SU2, SZ>(py::module &m,
                                             const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));

#endif
