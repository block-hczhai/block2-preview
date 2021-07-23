
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "../block2_core.hpp"
#include "../block2_dmrg.hpp"

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<ActiveTypes>);
// SZ
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MovingEnvironment<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<EffectiveHamiltonian<SZ>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MovingEnvironment<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<EffectiveHamiltonian<SU2>>>);

template <typename S>
auto bind_spin_specific(py::module &m) -> decltype(typename S::is_su2_t()) {

    py::class_<PDM2MPOQC<S>, shared_ptr<PDM2MPOQC<S>>, MPO<S>>(m, "PDM2MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S>> &>(), py::arg("hamil"))
        .def("get_matrix", &PDM2MPOQC<S>::template get_matrix<double>)
        .def("get_matrix", &PDM2MPOQC<S>::template get_matrix<complex<double>>)
        .def("get_matrix_spatial",
             &PDM2MPOQC<S>::template get_matrix_spatial<double>)
        .def("get_matrix_spatial",
             &PDM2MPOQC<S>::template get_matrix_spatial<complex<double>>);
}

template <typename S>
auto bind_spin_specific(py::module &m) -> decltype(typename S::is_sz_t()) {

    py::class_<PDM2MPOQC<S>, shared_ptr<PDM2MPOQC<S>>, MPO<S>>(m, "PDM2MPOQC")
        .def_property_readonly_static(
            "s_all", [](py::object) { return PDM2MPOQC<S>::s_all; })
        .def_property_readonly_static(
            "s_minimal", [](py::object) { return PDM2MPOQC<S>::s_minimal; })
        .def(py::init<const shared_ptr<Hamiltonian<S>> &>(), py::arg("hamil"))
        .def(py::init<const shared_ptr<Hamiltonian<S>> &, uint16_t>(),
             py::arg("hamil"), py::arg("mask"))
        .def("get_matrix", &PDM2MPOQC<S>::template get_matrix<double>)
        .def("get_matrix", &PDM2MPOQC<S>::template get_matrix<complex<double>>)
        .def("get_matrix_spatial",
             &PDM2MPOQC<S>::template get_matrix_spatial<double>)
        .def("get_matrix_spatial",
             &PDM2MPOQC<S>::template get_matrix_spatial<complex<double>>);

    py::class_<SumMPOQC<S>, shared_ptr<SumMPOQC<S>>, MPO<S>>(m, "SumMPOQC")
        .def_readwrite("ts", &SumMPOQC<S>::ts)
        .def(py::init<const shared_ptr<HamiltonianQC<S>> &,
                      const vector<uint16_t> &>(),
             py::arg("hamil"), py::arg("pts"));
}

template <typename S> void bind_mps(py::module &m) {

    py::class_<SparseTensor<S>, shared_ptr<SparseTensor<S>>>(m, "SparseTensor")
        .def(py::init<>())
        .def(py::init<
             const vector<vector<pair<pair<S, S>, shared_ptr<Tensor>>>> &>())
        .def_readwrite("data", &SparseTensor<S>::data)
        .def("__repr__", [](SparseTensor<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<SparseTensor<S>>>>(m, "VectorSpTensor");

    py::class_<MPSInfo<S>, shared_ptr<MPSInfo<S>>>(m, "MPSInfo")
        .def_readwrite("n_sites", &MPSInfo<S>::n_sites)
        .def_readwrite("vacuum", &MPSInfo<S>::vacuum)
        .def_readwrite("target", &MPSInfo<S>::target)
        .def_readwrite("bond_dim", &MPSInfo<S>::bond_dim)
        .def_readwrite("basis", &MPSInfo<S>::basis)
        .def_readwrite("left_dims_fci", &MPSInfo<S>::left_dims_fci)
        .def_readwrite("right_dims_fci", &MPSInfo<S>::right_dims_fci)
        .def_readwrite("left_dims", &MPSInfo<S>::left_dims)
        .def_readwrite("right_dims", &MPSInfo<S>::right_dims)
        .def_readwrite("tag", &MPSInfo<S>::tag)
        .def(py::init<int>())
        .def(py::init<int, S, S, const vector<shared_ptr<StateInfo<S>>> &>())
        .def(py::init<int, S, S, const vector<shared_ptr<StateInfo<S>>> &,
                      bool>())
        .def("get_ancilla_type", &MPSInfo<S>::get_ancilla_type)
        .def("get_type", &MPSInfo<S>::get_type)
        .def("load_data",
             (void (MPSInfo<S>::*)(const string &)) & MPSInfo<S>::load_data)
        .def("save_data", (void (MPSInfo<S>::*)(const string &) const) &
                              MPSInfo<S>::save_data)
        .def("get_max_bond_dimension", &MPSInfo<S>::get_max_bond_dimension)
        .def("check_bond_dimensions", &MPSInfo<S>::check_bond_dimensions)
        .def("set_bond_dimension_using_occ",
             &MPSInfo<S>::set_bond_dimension_using_occ, py::arg("m"),
             py::arg("occ"), py::arg("bias") = 1.0)
        .def("set_bond_dimension_using_hf",
             &MPSInfo<S>::set_bond_dimension_using_hf, py::arg("m"),
             py::arg("occ"), py::arg("n_local") = 0)
        .def("set_bond_dimension_full_fci",
             &MPSInfo<S>::set_bond_dimension_full_fci,
             py::arg("left_vacuum") = S(S::invalid),
             py::arg("right_vacuum") = S(S::invalid))
        .def("set_bond_dimension_fci", &MPSInfo<S>::set_bond_dimension_fci,
             py::arg("left_vacuum") = S(S::invalid),
             py::arg("right_vacuum") = S(S::invalid))
        .def("set_bond_dimension", &MPSInfo<S>::set_bond_dimension)
        .def("swap_wfn_to_fused_left", &MPSInfo<S>::swap_wfn_to_fused_left)
        .def("swap_wfn_to_fused_right", &MPSInfo<S>::swap_wfn_to_fused_right)
        .def("get_filename", &MPSInfo<S>::get_filename, py::arg("left"),
             py::arg("i"), py::arg("dir") = "")
        .def("save_mutable", &MPSInfo<S>::save_mutable)
        .def("copy_mutable", &MPSInfo<S>::copy_mutable)
        .def("deallocate_mutable", &MPSInfo<S>::deallocate_mutable)
        .def("load_mutable", &MPSInfo<S>::load_mutable)
        .def("save_left_dims", &MPSInfo<S>::save_left_dims)
        .def("save_right_dims", &MPSInfo<S>::save_right_dims)
        .def("load_left_dims", &MPSInfo<S>::load_left_dims)
        .def("load_right_dims", &MPSInfo<S>::load_right_dims)
        .def("deep_copy", &MPSInfo<S>::deep_copy)
        .def("deallocate", &MPSInfo<S>::deallocate);

    py::class_<DynamicMPSInfo<S>, shared_ptr<DynamicMPSInfo<S>>, MPSInfo<S>>(
        m, "DynamicMPSInfo")
        .def_readwrite("iocc", &DynamicMPSInfo<S>::iocc)
        .def_readwrite("n_local", &DynamicMPSInfo<S>::n_local)
        .def(py::init([](int n_sites, S vacuum, S target,
                         const vector<shared_ptr<StateInfo<S>>> &basis,
                         const vector<uint8_t> &iocc) {
            return make_shared<DynamicMPSInfo<S>>(n_sites, vacuum, target,
                                                  basis, iocc);
        }))
        .def("set_left_bond_dimension_local",
             &DynamicMPSInfo<S>::set_left_bond_dimension_local, py::arg("i"),
             py::arg("match_prev") = false)
        .def("set_right_bond_dimension_local",
             &DynamicMPSInfo<S>::set_right_bond_dimension_local, py::arg("i"),
             py::arg("match_prev") = false);

    py::class_<CASCIMPSInfo<S>, shared_ptr<CASCIMPSInfo<S>>, MPSInfo<S>>(
        m, "CASCIMPSInfo")
        .def_readwrite("casci_mask", &CASCIMPSInfo<S>::casci_mask)
        .def(py::init([](int n_sites, S vacuum, S target,
                         const vector<shared_ptr<StateInfo<S>>> &basis,
                         const vector<ActiveTypes> &casci_mask) {
            return make_shared<CASCIMPSInfo<S>>(n_sites, vacuum, target, basis,
                                                casci_mask);
        }))
        .def(py::init([](int n_sites, S vacuum, S target,
                         const vector<shared_ptr<StateInfo<S>>> &basis,
                         int n_active_sites, int n_active_electrons) {
            return make_shared<CASCIMPSInfo<S>>(n_sites, vacuum, target, basis,
                                                n_active_sites,
                                                n_active_electrons);
        }))
        .def(py::init([](int n_sites, S vacuum, S target,
                         const vector<shared_ptr<StateInfo<S>>> &basis,
                         int n_inactive_sites, int n_active_sites,
                         int n_virtual_sites) {
            return make_shared<CASCIMPSInfo<S>>(
                n_sites, vacuum, target, basis, n_inactive_sites,
                n_active_sites, n_virtual_sites);
        }));

    py::class_<MRCIMPSInfo<S>, shared_ptr<MRCIMPSInfo<S>>, MPSInfo<S>>(
        m, "MRCIMPSInfo")
        .def(py::init<int, int, int, S, S,
                      const vector<shared_ptr<StateInfo<S>>> &>())
        .def(py::init<int, int, int, int, S, S,
                      const vector<shared_ptr<StateInfo<S>>> &>())
        .def_readwrite("n_inactive", &MRCIMPSInfo<S>::n_inactive,
                       "Number of inactive orbitals")
        .def_readwrite("n_external", &MRCIMPSInfo<S>::n_external,
                       "Number of external orbitals")
        .def_readwrite("ci_order", &MRCIMPSInfo<S>::ci_order,
                       "Up to how many electrons are allowed in ext. orbitals: "
                       "2 gives MR-CISD");

    py::class_<AncillaMPSInfo<S>, shared_ptr<AncillaMPSInfo<S>>, MPSInfo<S>>(
        m, "AncillaMPSInfo")
        .def_readwrite("n_physical_sites", &AncillaMPSInfo<S>::n_physical_sites)
        .def(py::init([](int n_sites, S vacuum, S target,
                         const vector<shared_ptr<StateInfo<S>>> &basis) {
            return make_shared<AncillaMPSInfo<S>>(n_sites, vacuum, target,
                                                  basis);
        }))
        .def_static("trans_basis", &AncillaMPSInfo<S>::trans_basis)
        .def("set_thermal_limit", &AncillaMPSInfo<S>::set_thermal_limit);

    py::class_<MultiMPSInfo<S>, shared_ptr<MultiMPSInfo<S>>, MPSInfo<S>>(
        m, "MultiMPSInfo")
        .def_readwrite("targets", &MultiMPSInfo<S>::targets)
        .def(py::init<int>())
        .def(py::init([](int n_sites, S vacuum, const vector<S> &target,
                         const vector<shared_ptr<StateInfo<S>>> &basis) {
            return make_shared<MultiMPSInfo<S>>(n_sites, vacuum, target, basis);
        }))
        .def("make_single", &MultiMPSInfo<S>::make_single)
        .def_static("from_mps_info", &MultiMPSInfo<S>::from_mps_info);

    py::class_<MPS<S>, shared_ptr<MPS<S>>>(m, "MPS")
        .def(py::init<const shared_ptr<MPSInfo<S>> &>())
        .def(py::init<int, int, int>())
        .def_readwrite("n_sites", &MPS<S>::n_sites)
        .def_readwrite("center", &MPS<S>::center)
        .def_readwrite("dot", &MPS<S>::dot)
        .def_readwrite("info", &MPS<S>::info)
        .def_readwrite("tensors", &MPS<S>::tensors)
        .def_readwrite("canonical_form", &MPS<S>::canonical_form)
        .def("get_type", &MPS<S>::get_type)
        .def("initialize", &MPS<S>::initialize, py::arg("info"),
             py::arg("init_left") = true, py::arg("init_right") = true)
        .def("fill_thermal_limit", &MPS<S>::fill_thermal_limit)
        .def("canonicalize", &MPS<S>::canonicalize)
        .def("dynamic_canonicalize", &MPS<S>::dynamic_canonicalize)
        .def("random_canonicalize", &MPS<S>::random_canonicalize)
        .def("from_singlet_embedding_wfn", &MPS<S>::from_singlet_embedding_wfn,
             py::arg("cg"), py::arg("para_rule") = nullptr)
        .def("to_singlet_embedding_wfn", &MPS<S>::to_singlet_embedding_wfn,
             py::arg("cg"), py::arg("para_rule") = nullptr)
        .def("move_left", &MPS<S>::move_left, py::arg("cg"),
             py::arg("para_rule") = nullptr)
        .def("move_right", &MPS<S>::move_right, py::arg("cg"),
             py::arg("para_rule") = nullptr)
        .def("flip_fused_form", &MPS<S>::flip_fused_form)
        .def("get_filename", &MPS<S>::get_filename, py::arg("i"),
             py::arg("dir") = "")
        .def("load_data", &MPS<S>::load_data)
        .def("save_data", &MPS<S>::save_data)
        .def("copy_data", &MPS<S>::copy_data)
        .def("load_mutable", &MPS<S>::load_mutable)
        .def("save_mutable", &MPS<S>::save_mutable)
        .def("save_tensor", &MPS<S>::save_tensor)
        .def("load_tensor", &MPS<S>::load_tensor)
        .def("unload_tensor", &MPS<S>::unload_tensor)
        .def("deep_copy", &MPS<S>::deep_copy)
        .def("estimate_storage", &MPS<S>::estimate_storage,
             py::arg("info") = nullptr)
        .def("deallocate", &MPS<S>::deallocate);

    py::bind_vector<vector<shared_ptr<MPS<S>>>>(m, "VectorMPS");

    py::class_<MultiMPS<S>, shared_ptr<MultiMPS<S>>, MPS<S>>(m, "MultiMPS")
        .def(py::init<const shared_ptr<MultiMPSInfo<S>> &>())
        .def(py::init<int, int, int, int>())
        .def_readwrite("nroots", &MultiMPS<S>::nroots)
        .def_readwrite("wfns", &MultiMPS<S>::wfns)
        .def_readwrite("weights", &MultiMPS<S>::weights)
        .def("get_wfn_filename", &MultiMPS<S>::get_wfn_filename)
        .def("save_wavefunction", &MultiMPS<S>::save_wavefunction)
        .def("load_wavefunction", &MultiMPS<S>::load_wavefunction)
        .def("unload_wavefunction", &MultiMPS<S>::unload_wavefunction)
        .def("extract", &MultiMPS<S>::extract)
        .def("iscale", &MultiMPS<S>::iscale)
        .def("make_single", &MultiMPS<S>::make_single)
        .def_static("make_complex", &MultiMPS<S>::make_complex);

    py::class_<ParallelMPS<S>, shared_ptr<ParallelMPS<S>>, MPS<S>>(
        m, "ParallelMPS")
        .def(py::init<const shared_ptr<MPSInfo<S>> &>())
        .def(py::init<int, int, int>())
        .def(py::init<const shared_ptr<MPS<S>> &>())
        .def(py::init<const shared_ptr<MPSInfo<S>> &,
                      const shared_ptr<ParallelRule<S>> &>())
        .def(py::init<int, int, int, const shared_ptr<ParallelRule<S>> &>())
        .def(py::init<const shared_ptr<MPS<S>> &,
                      const shared_ptr<ParallelRule<S>> &>())
        .def_readwrite("conn_centers", &ParallelMPS<S>::conn_centers)
        .def_readwrite("conn_matrices", &ParallelMPS<S>::conn_matrices)
        .def_readwrite("ncenter", &ParallelMPS<S>::ncenter)
        .def_readwrite("ncenter", &ParallelMPS<S>::ncenter)
        .def_readwrite("ncenter", &ParallelMPS<S>::ncenter)
        .def_readwrite("svd_eps", &ParallelMPS<S>::svd_eps)
        .def_readwrite("svd_cutoff", &ParallelMPS<S>::svd_cutoff);

    py::class_<UnfusedMPS<S>, shared_ptr<UnfusedMPS<S>>>(m, "UnfusedMPS")
        .def(py::init<>())
        .def(py::init<const shared_ptr<MPS<S>> &>())
        .def_readwrite("info", &UnfusedMPS<S>::info)
        .def_readwrite("tensors", &UnfusedMPS<S>::tensors)
        .def_readwrite("n_sites", &UnfusedMPS<S>::n_sites)
        .def_readwrite("center", &UnfusedMPS<S>::center)
        .def_readwrite("dot", &UnfusedMPS<S>::dot)
        .def_readwrite("canonical_form", &UnfusedMPS<S>::canonical_form)
        .def_static("forward_left_fused", &UnfusedMPS<S>::forward_left_fused,
                    py::arg("i"), py::arg("mps"), py::arg("wfn"))
        .def_static("forward_right_fused", &UnfusedMPS<S>::forward_right_fused,
                    py::arg("i"), py::arg("mps"), py::arg("wfn"))
        .def_static("forward_mps_tensor", &UnfusedMPS<S>::forward_mps_tensor,
                    py::arg("i"), py::arg("mps"))
        .def_static("backward_left_fused", &UnfusedMPS<S>::backward_left_fused,
                    py::arg("i"), py::arg("mps"), py::arg("spt"),
                    py::arg("wfn"))
        .def_static("backward_right_fused",
                    &UnfusedMPS<S>::backward_right_fused, py::arg("i"),
                    py::arg("mps"), py::arg("spt"), py::arg("wfn"))
        .def_static("backward_mps_tensor", &UnfusedMPS<S>::backward_mps_tensor,
                    py::arg("i"), py::arg("mps"), py::arg("spt"))
        .def("initialize", &UnfusedMPS<S>::initialize)
        .def("finalize", &UnfusedMPS<S>::finalize)
        .def("resolve_singlet_embedding",
             &UnfusedMPS<S>::resolve_singlet_embedding);

    py::class_<DeterminantTRIE<S>, shared_ptr<DeterminantTRIE<S>>>(
        m, "DeterminantTRIE")
        .def(py::init<int>(), py::arg("n_sites"))
        .def(py::init<int, bool>(), py::arg("n_sites"),
             py::arg("enable_look_up"))
        .def_readwrite("data", &DeterminantTRIE<S>::data)
        .def_readwrite("dets", &DeterminantTRIE<S>::dets)
        .def_readwrite("invs", &DeterminantTRIE<S>::invs)
        .def_readwrite("vals", &DeterminantTRIE<S>::vals)
        .def_readwrite("n_sites", &DeterminantTRIE<S>::n_sites)
        .def_readwrite("enable_look_up", &DeterminantTRIE<S>::enable_look_up)
        .def("clear", &DeterminantTRIE<S>::clear)
        .def("copy", &DeterminantTRIE<S>::copy)
        .def("__len__", &DeterminantTRIE<S>::size)
        .def("append", &DeterminantTRIE<S>::push_back, py::arg("det"))
        .def("find", &DeterminantTRIE<S>::find, py::arg("det"))
        .def("__getitem__", &DeterminantTRIE<S>::operator[], py::arg("idx"))
        .def("get_state_occupation", &DeterminantTRIE<S>::get_state_occupation)
        .def("evaluate", &DeterminantTRIE<S>::evaluate, py::arg("mps"),
             py::arg("cutoff") = 0.0);
}

template <typename S> void bind_partition(py::module &m) {

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
        .def("load_data", (void (Partition<S>::*)(bool, const string &)) &
                              Partition<S>::load_data)
        .def("save_data", (void (Partition<S>::*)(bool, const string &) const) &
                              Partition<S>::save_data)
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
             py::arg("c"), py::arg("idx") = 0, py::arg("factor") = 1.0,
             py::arg("all_reduce") = true)
        .def("eigs", &EffectiveHamiltonian<S>::eigs)
        .def("multiply", &EffectiveHamiltonian<S>::multiply)
        .def("inverse_multiply", &EffectiveHamiltonian<S>::inverse_multiply)
        .def("greens_function", &EffectiveHamiltonian<S>::greens_function)
        .def("expect", &EffectiveHamiltonian<S>::expect)
        .def("rk4_apply", &EffectiveHamiltonian<S>::rk4_apply, py::arg("beta"),
             py::arg("const_e"), py::arg("eval_energy") = false,
             py::arg("para_rule") = nullptr)
        .def("expo_apply", &EffectiveHamiltonian<S>::expo_apply,
             py::arg("beta"), py::arg("const_e"), py::arg("symmetric"),
             py::arg("iprint") = false, py::arg("para_rule") = nullptr)
        .def("deallocate", &EffectiveHamiltonian<S>::deallocate);

    py::bind_vector<vector<shared_ptr<EffectiveHamiltonian<S>>>>(
        m, "VectorEffectiveHamiltonian");

    py::class_<LinearEffectiveHamiltonian<S>,
               shared_ptr<LinearEffectiveHamiltonian<S>>>(
        m, "LinearEffectiveHamiltonian")
        .def(py::init<const shared_ptr<EffectiveHamiltonian<S>> &>())
        .def(py::init<const vector<shared_ptr<EffectiveHamiltonian<S>>> &,
                      const vector<double> &>())
        .def("__call__", &LinearEffectiveHamiltonian<S>::operator(),
             py::arg("b"), py::arg("c"))
        .def("eigs", &LinearEffectiveHamiltonian<S>::eigs)
        .def("deallocate", &LinearEffectiveHamiltonian<S>::deallocate)
        .def_readwrite("h_effs", &LinearEffectiveHamiltonian<S>::h_effs)
        .def_readwrite("coeffs", &LinearEffectiveHamiltonian<S>::coeffs)
        .def("__mul__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S>> &self,
                double d) { return self * d; })
        .def("__rmul__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S>> &self,
                double d) { return self * d; })
        .def("__neg__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S>> &self) {
                 return -self;
             })
        .def("__add__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S>> &self,
                const shared_ptr<LinearEffectiveHamiltonian<S>> &other) {
                 return self + other;
             })
        .def("__sub__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S>> &self,
                const shared_ptr<LinearEffectiveHamiltonian<S>> &other) {
                 return self - other;
             });

    py::class_<EffectiveHamiltonian<S, MultiMPS<S>>,
               shared_ptr<EffectiveHamiltonian<S, MultiMPS<S>>>>(
        m, "EffectiveHamiltonianMultiMPS")
        .def(py::init<const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const shared_ptr<DelayedOperatorTensor<S>> &,
                      const vector<shared_ptr<SparseMatrixGroup<S>>> &,
                      const vector<shared_ptr<SparseMatrixGroup<S>>> &,
                      const shared_ptr<OpElement<S>> &,
                      const shared_ptr<SymbolicColumnVector<S>> &,
                      const shared_ptr<TensorFunctions<S>> &, bool>())
        .def_readwrite("left_op_infos",
                       &EffectiveHamiltonian<S, MultiMPS<S>>::left_op_infos)
        .def_readwrite("right_op_infos",
                       &EffectiveHamiltonian<S, MultiMPS<S>>::right_op_infos)
        .def_readwrite("op", &EffectiveHamiltonian<S, MultiMPS<S>>::op)
        .def_readwrite("bra", &EffectiveHamiltonian<S, MultiMPS<S>>::bra)
        .def_readwrite("ket", &EffectiveHamiltonian<S, MultiMPS<S>>::ket)
        .def_readwrite("diag", &EffectiveHamiltonian<S, MultiMPS<S>>::diag)
        .def_readwrite("cmat", &EffectiveHamiltonian<S, MultiMPS<S>>::cmat)
        .def_readwrite("vmat", &EffectiveHamiltonian<S, MultiMPS<S>>::vmat)
        .def_readwrite("tf", &EffectiveHamiltonian<S, MultiMPS<S>>::tf)
        .def_readwrite("opdq", &EffectiveHamiltonian<S, MultiMPS<S>>::opdq)
        .def_readwrite("compute_diag",
                       &EffectiveHamiltonian<S, MultiMPS<S>>::compute_diag)
        .def("__call__", &EffectiveHamiltonian<S, MultiMPS<S>>::operator(),
             py::arg("b"), py::arg("c"), py::arg("idx") = 0,
             py::arg("factor") = 1.0, py::arg("all_reduce") = true)
        .def("eigs", &EffectiveHamiltonian<S, MultiMPS<S>>::eigs)
        .def("expect", &EffectiveHamiltonian<S, MultiMPS<S>>::expect)
        .def("rk4_apply", &EffectiveHamiltonian<S, MultiMPS<S>>::rk4_apply,
             py::arg("beta"), py::arg("const_e"),
             py::arg("eval_energy") = false, py::arg("para_rule") = nullptr)
        .def("expo_apply", &EffectiveHamiltonian<S, MultiMPS<S>>::expo_apply,
             py::arg("beta"), py::arg("const_e"), py::arg("iprint") = false,
             py::arg("para_rule") = nullptr)
        .def("deallocate", &EffectiveHamiltonian<S, MultiMPS<S>>::deallocate);

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
        .def_readwrite("para_rule", &MovingEnvironment<S>::para_rule)
        .def_readwrite("tctr", &MovingEnvironment<S>::tctr)
        .def_readwrite("trot", &MovingEnvironment<S>::trot)
        .def_readwrite("iprint", &MovingEnvironment<S>::iprint)
        .def_readwrite("delayed_contraction",
                       &MovingEnvironment<S>::delayed_contraction)
        .def_readwrite("fuse_center", &MovingEnvironment<S>::fuse_center)
        .def_readwrite("save_partition_info",
                       &MovingEnvironment<S>::save_partition_info)
        .def_readwrite("cached_opt", &MovingEnvironment<S>::cached_opt)
        .def_readwrite("cached_info", &MovingEnvironment<S>::cached_info)
        .def_readwrite("cached_contraction",
                       &MovingEnvironment<S>::cached_contraction)
        .def("left_contract_rotate",
             &MovingEnvironment<S>::left_contract_rotate)
        .def("right_contract_rotate",
             &MovingEnvironment<S>::right_contract_rotate)
        .def("left_contract_rotate_unordered",
             &MovingEnvironment<S>::left_contract_rotate_unordered)
        .def("right_contract_rotate_unordered",
             &MovingEnvironment<S>::right_contract_rotate_unordered)
        .def("parallelize_mps", &MovingEnvironment<S>::parallelize_mps)
        .def("serialize_mps", &MovingEnvironment<S>::serialize_mps)
        .def(
            "left_contract",
            [](MovingEnvironment<S> *self, int iL,
               vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_info) {
                shared_ptr<OperatorTensor<S>> new_left = nullptr;
                self->left_contract(iL, left_op_info, new_left, false);
                return new_left;
            })
        .def("right_contract",
             [](MovingEnvironment<S> *self, int iR,
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
                    &right_op_infos) {
                 shared_ptr<OperatorTensor<S>> new_right = nullptr;
                 self->right_contract(iR, right_op_infos, new_right, false);
                 return new_right;
             })
        .def(
            "left_copy",
            [](MovingEnvironment<S> *self, int iL,
               vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_info) {
                shared_ptr<OperatorTensor<S>> new_left = nullptr;
                self->left_copy(iL, left_op_info, new_left);
                return new_left;
            })
        .def("right_copy",
             [](MovingEnvironment<S> *self, int iR,
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
                    &right_op_infos) {
                 shared_ptr<OperatorTensor<S>> new_right = nullptr;
                 self->right_copy(iR, right_op_infos, new_right);
                 return new_right;
             })
        .def("init_environments", &MovingEnvironment<S>::init_environments,
             py::arg("iprint") = false)
        .def("finalize_environments",
             &MovingEnvironment<S>::finalize_environments,
             py::arg("renormalize_ops") = true)
        .def("prepare", &MovingEnvironment<S>::prepare)
        .def("move_to", &MovingEnvironment<S>::move_to, py::arg("i"),
             py::arg("preserve_data") = false)
        .def("partial_prepare", &MovingEnvironment<S>::partial_prepare)
        .def("get_left_archive_filename",
             &MovingEnvironment<S>::get_left_archive_filename)
        .def("get_middle_archive_filename",
             &MovingEnvironment<S>::get_middle_archive_filename)
        .def("get_right_archive_filename",
             &MovingEnvironment<S>::get_right_archive_filename)
        .def("get_left_partition_filename",
             &MovingEnvironment<S>::get_left_partition_filename)
        .def("get_right_partition_filename",
             &MovingEnvironment<S>::get_right_partition_filename)
        .def("eff_ham", &MovingEnvironment<S>::eff_ham, py::arg("fuse_type"),
             py::arg("forward"), py::arg("compute_diag"), py::arg("bra_wfn"),
             py::arg("ket_wfn"))
        .def("multi_eff_ham", &MovingEnvironment<S>::multi_eff_ham,
             py::arg("fuse_type"), py::arg("forward"), py::arg("compute_diag"))
        .def_static("contract_two_dot", &MovingEnvironment<S>::contract_two_dot,
                    py::arg("i"), py::arg("mps"), py::arg("reduced") = false)
        .def_static("wavefunction_add_noise",
                    &MovingEnvironment<S>::wavefunction_add_noise,
                    py::arg("psi"), py::arg("noise"))
        .def_static("scale_perturbative_noise",
                    &MovingEnvironment<S>::scale_perturbative_noise,
                    py::arg("noise"), py::arg("noise_type"), py::arg("mats"))
        .def_static("density_matrix", &MovingEnvironment<S>::density_matrix,
                    py::arg("vacuum"), py::arg("psi"), py::arg("trace_right"),
                    py::arg("noise"), py::arg("noise_type"),
                    py::arg("scale") = 1.0, py::arg("pkets") = nullptr)
        .def_static("density_matrix_with_multi_target",
                    &MovingEnvironment<S>::density_matrix_with_multi_target,
                    py::arg("vacuum"), py::arg("psi"), py::arg("weights"),
                    py::arg("trace_right"), py::arg("noise"),
                    py::arg("noise_type"), py::arg("scale") = 1.0,
                    py::arg("pkets") = nullptr)
        .def_static("density_matrix_add_wfn",
                    &MovingEnvironment<S>::density_matrix_add_wfn)
        .def_static(
            "density_matrix_add_perturbative_noise",
            &MovingEnvironment<S>::density_matrix_add_perturbative_noise)
        .def_static("density_matrix_add_matrices",
                    &MovingEnvironment<S>::density_matrix_add_matrices)
        .def_static("density_matrix_add_matrix_groups",
                    &MovingEnvironment<S>::density_matrix_add_matrix_groups)
        .def_static("truncate_density_matrix",
                    [](const shared_ptr<SparseMatrix<S>> &dm, int k,
                       double cutoff, TruncationTypes trunc_type) {
                        vector<pair<int, int>> ss;
                        double error =
                            MovingEnvironment<S>::truncate_density_matrix(
                                dm, ss, k, cutoff, trunc_type);
                        return make_pair(error, ss);
                    })
        .def_static("truncate_singular_values",
                    [](const vector<S> &qs, const vector<shared_ptr<Tensor>> &s,
                       int k, double cutoff, TruncationTypes trunc_type) {
                        vector<pair<int, int>> ss;
                        double error =
                            MovingEnvironment<S>::truncate_singular_values(
                                qs, s, ss, k, cutoff, trunc_type);
                        return make_pair(error, ss);
                    })
        .def_static("rotation_matrix_info_from_svd",
                    &MovingEnvironment<S>::rotation_matrix_info_from_svd,
                    py::arg("opdq"), py::arg("qs"), py::arg("ts"),
                    py::arg("trace_right"), py::arg("ilr"), py::arg("im"))
        .def_static(
            "wavefunction_info_from_svd",
            [](const vector<S> &qs,
               const shared_ptr<SparseMatrixInfo<S>> &wfninfo, bool trace_right,
               const vector<int> &ilr, const vector<ubond_t> &im) {
                vector<vector<int>> idx_dm_to_wfn;
                shared_ptr<SparseMatrixInfo<S>> r =
                    MovingEnvironment<S>::wavefunction_info_from_svd(
                        qs, wfninfo, trace_right, ilr, im, idx_dm_to_wfn);
                return make_pair(r, idx_dm_to_wfn);
            })
        .def_static(
            "rotation_matrix_info_from_density_matrix",
            &MovingEnvironment<S>::rotation_matrix_info_from_density_matrix,
            py::arg("dminfo"), py::arg("trace_right"), py::arg("ilr"),
            py::arg("im"))
        .def_static(
            "wavefunction_info_from_density_matrix",
            [](const shared_ptr<SparseMatrixInfo<S>> &dminfo,
               const shared_ptr<SparseMatrixInfo<S>> &wfninfo, bool trace_right,
               const vector<int> &ilr, const vector<ubond_t> &im) {
                vector<vector<int>> idx_dm_to_wfn;
                shared_ptr<SparseMatrixInfo<S>> r =
                    MovingEnvironment<S>::wavefunction_info_from_density_matrix(
                        dminfo, wfninfo, trace_right, ilr, im, idx_dm_to_wfn);
                return make_pair(r, idx_dm_to_wfn);
            })
        .def_static(
            "split_density_matrix",
            [](const shared_ptr<SparseMatrix<S>> &dm,
               const shared_ptr<SparseMatrix<S>> &wfn, int k, bool trace_right,
               bool normalize, double cutoff, TruncationTypes trunc_type) {
                shared_ptr<SparseMatrix<S>> left = nullptr, right = nullptr;
                double error = MovingEnvironment<S>::split_density_matrix(
                    dm, wfn, k, trace_right, normalize, left, right, cutoff,
                    trunc_type);
                return make_tuple(error, left, right);
            },
            py::arg("dm"), py::arg("wfn"), py::arg("k"), py::arg("trace_right"),
            py::arg("normalize"), py::arg("cutoff"),
            py::arg("trunc_type") = TruncationTypes::Physical)
        .def_static(
            "split_wavefunction_svd",
            [](S opdq, const shared_ptr<SparseMatrix<S>> &wfn, int k,
               bool trace_right, bool normalize, double cutoff,
               TruncationTypes trunc_type, DecompositionTypes decomp_type) {
                shared_ptr<SparseMatrix<S>> left = nullptr, right = nullptr;
                double error = MovingEnvironment<S>::split_wavefunction_svd(
                    opdq, wfn, k, trace_right, normalize, left, right, cutoff,
                    trunc_type, decomp_type);
                return make_tuple(error, left, right);
            },
            py::arg("opdq"), py::arg("wfn"), py::arg("k"),
            py::arg("trace_right"), py::arg("normalize"), py::arg("cutoff"),
            py::arg("decomp_type") = DecompositionTypes::PureSVD,
            py::arg("trunc_type") = TruncationTypes::Physical)
        .def_static("propagate_wfn", &MovingEnvironment<S>::propagate_wfn,
                    py::arg("i"), py::arg("n_sites"), py::arg("mps"),
                    py::arg("forward"), py::arg("cg"))
        .def_static("contract_multi_two_dot",
                    &MovingEnvironment<S>::contract_multi_two_dot, py::arg("i"),
                    py::arg("mps"), py::arg("reduced") = false)
        .def_static(
            "multi_split_density_matrix",
            [](const shared_ptr<SparseMatrix<S>> &dm,
               const vector<shared_ptr<SparseMatrixGroup<S>>> &wfns, int k,
               bool trace_right, bool normalize, double cutoff,
               TruncationTypes trunc_type) {
                vector<shared_ptr<SparseMatrixGroup<S>>> new_wfns;
                shared_ptr<SparseMatrix<S>> rot_mat = nullptr;
                double error = MovingEnvironment<S>::multi_split_density_matrix(
                    dm, wfns, k, trace_right, normalize, new_wfns, rot_mat,
                    cutoff, trunc_type);
                return make_tuple(error, new_wfns, rot_mat);
            },
            py::arg("dm"), py::arg("wfns"), py::arg("k"),
            py::arg("trace_right"), py::arg("normalize"), py::arg("cutoff"),
            py::arg("trunc_type") = TruncationTypes::Physical)
        .def_static("propagate_multi_wfn", &MovingEnvironment<S>::propagate_wfn,
                    py::arg("i"), py::arg("n_sites"), py::arg("mps"),
                    py::arg("forward"), py::arg("cg"));

    py::bind_vector<vector<shared_ptr<MovingEnvironment<S>>>>(
        m, "VectorMovingEnvironment");
}

template <typename S> void bind_qc_hamiltonian(py::module &m) {
    py::class_<HamiltonianQC<S>, shared_ptr<HamiltonianQC<S>>, Hamiltonian<S>>(
        m, "HamiltonianQC")
        .def(py::init<>())
        .def(py::init<S, int, const vector<uint8_t> &,
                      const shared_ptr<FCIDUMP> &>())
        .def_readwrite("fcidump", &HamiltonianQC<S>::fcidump)
        .def_property(
            "mu", [](HamiltonianQC<S> *self) { return self->mu; },
            [](HamiltonianQC<S> *self, double mu) { self->set_mu(mu); })
        .def_readwrite("op_prims", &HamiltonianQC<S>::op_prims)
        .def("v", &HamiltonianQC<S>::v)
        .def("t", &HamiltonianQC<S>::t)
        .def("e", &HamiltonianQC<S>::e)
        .def("init_site_ops", &HamiltonianQC<S>::init_site_ops)
        .def("get_site_ops", &HamiltonianQC<S>::get_site_ops);
}

template <typename S, typename FL>
void bind_expect(py::module &m, const string &name) {

    py::class_<typename Expect<S, FL>::Iteration,
               shared_ptr<typename Expect<S, FL>::Iteration>>(
        m, (name + "Iteration").c_str())
        .def(py::init<const vector<pair<shared_ptr<OpExpr<S>>, FL>> &, double,
                      double, size_t, double>())
        .def(py::init<const vector<pair<shared_ptr<OpExpr<S>>, FL>> &, double,
                      double>())
        .def_readwrite("bra_error", &Expect<S, FL>::Iteration::bra_error)
        .def_readwrite("ket_error", &Expect<S, FL>::Iteration::ket_error)
        .def_readwrite("tmult", &Expect<S, FL>::Iteration::tmult)
        .def_readwrite("nflop", &Expect<S, FL>::Iteration::nflop)
        .def("__repr__", [](typename Expect<S, FL>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Expect<S, FL>, shared_ptr<Expect<S, FL>>>(m, name.c_str())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &, ubond_t,
                      ubond_t>())
        .def(
            py::init<const shared_ptr<MovingEnvironment<S>> &, ubond_t, ubond_t,
                     double, const vector<double> &, const vector<int> &>())
        .def_readwrite("iprint", &Expect<S, FL>::iprint)
        .def_readwrite("cutoff", &Expect<S, FL>::cutoff)
        .def_readwrite("beta", &Expect<S, FL>::beta)
        .def_readwrite("partition_weights", &Expect<S, FL>::partition_weights)
        .def_readwrite("me", &Expect<S, FL>::me)
        .def_readwrite("bra_bond_dim", &Expect<S, FL>::bra_bond_dim)
        .def_readwrite("ket_bond_dim", &Expect<S, FL>::ket_bond_dim)
        .def_readwrite("expectations", &Expect<S, FL>::expectations)
        .def_readwrite("forward", &Expect<S, FL>::forward)
        .def_readwrite("trunc_type", &Expect<S, FL>::trunc_type)
        .def_readwrite("ex_type", &Expect<S, FL>::ex_type)
        .def_readwrite("algo_type", &Expect<S, FL>::algo_type)
        .def("update_one_dot", &Expect<S, FL>::update_one_dot)
        .def("update_multi_one_dot", &Expect<S, FL>::update_multi_one_dot)
        .def("update_two_dot", &Expect<S, FL>::update_two_dot)
        .def("update_multi_two_dot", &Expect<S, FL>::update_multi_two_dot)
        .def("blocking", &Expect<S, FL>::blocking)
        .def("sweep", &Expect<S, FL>::sweep)
        .def("solve", &Expect<S, FL>::solve, py::arg("propagate"),
             py::arg("forward") = true)
        .def("get_1pdm_spatial", &Expect<S, FL>::get_1pdm_spatial,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1pdm", &Expect<S, FL>::get_1pdm,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_2pdm_spatial", &Expect<S, FL>::get_2pdm_spatial,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_2pdm", &Expect<S, FL>::get_2pdm,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1npc_spatial", &Expect<S, FL>::get_1npc_spatial, py::arg("s"),
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1npc", &Expect<S, FL>::get_1npc, py::arg("s"),
             py::arg("n_physical_sites") = (uint16_t)0U);
}

template <typename S> void bind_algorithms(py::module &m) {

    py::class_<typename DMRG<S>::Iteration,
               shared_ptr<typename DMRG<S>::Iteration>>(m, "DMRGIteration")
        .def(py::init<const vector<double> &, double, int, int, size_t,
                      double>())
        .def(py::init<const vector<double> &, double, int, int>())
        .def_readwrite("mmps", &DMRG<S>::Iteration::mmps)
        .def_readwrite("energies", &DMRG<S>::Iteration::energies)
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
                      const vector<ubond_t> &, const vector<double> &>())
        .def_readwrite("iprint", &DMRG<S>::iprint)
        .def_readwrite("cutoff", &DMRG<S>::cutoff)
        .def_readwrite("quanta_cutoff", &DMRG<S>::quanta_cutoff)
        .def_readwrite("me", &DMRG<S>::me)
        .def_readwrite("ext_mes", &DMRG<S>::ext_mes)
        .def_readwrite("ext_mpss", &DMRG<S>::ext_mpss)
        .def_readwrite("state_specific", &DMRG<S>::state_specific)
        .def_readwrite("bond_dims", &DMRG<S>::bond_dims)
        .def_readwrite("noises", &DMRG<S>::noises)
        .def_readwrite("davidson_conv_thrds", &DMRG<S>::davidson_conv_thrds)
        .def_readwrite("davidson_max_iter", &DMRG<S>::davidson_max_iter)
        .def_readwrite("davidson_soft_max_iter",
                       &DMRG<S>::davidson_soft_max_iter)
        .def_readwrite("davidson_shift", &DMRG<S>::davidson_shift)
        .def_readwrite("davidson_type", &DMRG<S>::davidson_type)
        .def_readwrite("conn_adjust_step", &DMRG<S>::conn_adjust_step)
        .def_readwrite("energies", &DMRG<S>::energies)
        .def_readwrite("discarded_weights", &DMRG<S>::discarded_weights)
        .def_readwrite("mps_quanta", &DMRG<S>::mps_quanta)
        .def_readwrite("sweep_energies", &DMRG<S>::sweep_energies)
        .def_readwrite("sweep_discarded_weights",
                       &DMRG<S>::sweep_discarded_weights)
        .def_readwrite("sweep_quanta", &DMRG<S>::sweep_quanta)
        .def_readwrite("forward", &DMRG<S>::forward)
        .def_readwrite("noise_type", &DMRG<S>::noise_type)
        .def_readwrite("trunc_type", &DMRG<S>::trunc_type)
        .def_readwrite("decomp_type", &DMRG<S>::decomp_type)
        .def_readwrite("decomp_last_site", &DMRG<S>::decomp_last_site)
        .def_readwrite("sweep_cumulative_nflop",
                       &DMRG<S>::sweep_cumulative_nflop)
        .def_readwrite("sweep_max_pket_size", &DMRG<S>::sweep_max_pket_size)
        .def_readwrite("sweep_max_eff_ham_size",
                       &DMRG<S>::sweep_max_eff_ham_size)
        .def("update_two_dot", &DMRG<S>::update_two_dot)
        .def("update_one_dot", &DMRG<S>::update_one_dot)
        .def("update_multi_two_dot", &DMRG<S>::update_multi_two_dot)
        .def("update_multi_one_dot", &DMRG<S>::update_multi_one_dot)
        .def("blocking", &DMRG<S>::blocking)
        .def("partial_sweep", &DMRG<S>::partial_sweep)
        .def("connection_sweep", &DMRG<S>::connection_sweep)
        .def("unordered_sweep", &DMRG<S>::unordered_sweep)
        .def("sweep", &DMRG<S>::sweep)
        .def("solve", &DMRG<S>::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<typename TDDMRG<S>::Iteration,
               shared_ptr<typename TDDMRG<S>::Iteration>>(m, "TDDMRGIteration")
        .def(py::init<double, double, double, int, int, size_t, double>())
        .def(py::init<double, double, double, int, int>())
        .def_readwrite("mmps", &TDDMRG<S>::Iteration::mmps)
        .def_readwrite("energy", &TDDMRG<S>::Iteration::energy)
        .def_readwrite("normsq", &TDDMRG<S>::Iteration::normsq)
        .def_readwrite("error", &TDDMRG<S>::Iteration::error)
        .def_readwrite("nmult", &TDDMRG<S>::Iteration::nmult)
        .def_readwrite("tmult", &TDDMRG<S>::Iteration::tmult)
        .def_readwrite("nflop", &TDDMRG<S>::Iteration::nflop)
        .def("__repr__", [](typename TDDMRG<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<TDDMRG<S>, shared_ptr<TDDMRG<S>>>(m, "TDDMRG")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<double> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &>())
        .def_readwrite("me", &TDDMRG<S>::me)
        .def_readwrite("lme", &TDDMRG<S>::lme)
        .def_readwrite("rme", &TDDMRG<S>::rme)
        .def_readwrite("iprint", &TDDMRG<S>::iprint)
        .def_readwrite("cutoff", &TDDMRG<S>::cutoff)
        .def_readwrite("bond_dims", &TDDMRG<S>::bond_dims)
        .def_readwrite("noises", &TDDMRG<S>::noises)
        .def_readwrite("energies", &TDDMRG<S>::energies)
        .def_readwrite("normsqs", &TDDMRG<S>::normsqs)
        .def_readwrite("discarded_weights", &TDDMRG<S>::discarded_weights)
        .def_readwrite("forward", &TDDMRG<S>::forward)
        .def_readwrite("n_sub_sweeps", &TDDMRG<S>::n_sub_sweeps)
        .def_readwrite("weights", &TDDMRG<S>::weights)
        .def_readwrite("mode", &TDDMRG<S>::mode)
        .def_readwrite("noise_type", &TDDMRG<S>::noise_type)
        .def_readwrite("trunc_type", &TDDMRG<S>::trunc_type)
        .def_readwrite("decomp_type", &TDDMRG<S>::decomp_type)
        .def_readwrite("decomp_last_site", &TDDMRG<S>::decomp_last_site)
        .def_readwrite("hermitian", &TDDMRG<S>::hermitian)
        .def_readwrite("sweep_cumulative_nflop",
                       &TDDMRG<S>::sweep_cumulative_nflop)
        .def("update_one_dot", &TDDMRG<S>::update_one_dot)
        .def("update_two_dot", &TDDMRG<S>::update_two_dot)
        .def("blocking", &TDDMRG<S>::blocking)
        .def("sweep", &TDDMRG<S>::sweep)
        .def("normalize", &TDDMRG<S>::normalize)
        .def("solve", &TDDMRG<S>::solve, py::arg("n_sweeps"), py::arg("beta"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<typename TimeEvolution<S>::Iteration,
               shared_ptr<typename TimeEvolution<S>::Iteration>>(
        m, "TimeEvolutionIteration")
        .def(py::init<double, double, double, int, int, int, size_t, double>())
        .def(py::init<double, double, double, int, int, int>())
        .def_readwrite("mmps", &TimeEvolution<S>::Iteration::mmps)
        .def_readwrite("energy", &TimeEvolution<S>::Iteration::energy)
        .def_readwrite("normsq", &TimeEvolution<S>::Iteration::normsq)
        .def_readwrite("error", &TimeEvolution<S>::Iteration::error)
        .def_readwrite("nexpo", &TimeEvolution<S>::Iteration::nexpo)
        .def_readwrite("nexpok", &TimeEvolution<S>::Iteration::nexpok)
        .def_readwrite("texpo", &TimeEvolution<S>::Iteration::texpo)
        .def_readwrite("nflop", &TimeEvolution<S>::Iteration::nflop)
        .def("__repr__", [](typename TimeEvolution<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<TimeEvolution<S>, shared_ptr<TimeEvolution<S>>>(m,
                                                               "TimeEvolution")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, TETypes>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, TETypes, int>())
        .def_readwrite("iprint", &TimeEvolution<S>::iprint)
        .def_readwrite("cutoff", &TimeEvolution<S>::cutoff)
        .def_readwrite("me", &TimeEvolution<S>::me)
        .def_readwrite("bond_dims", &TimeEvolution<S>::bond_dims)
        .def_readwrite("noises", &TimeEvolution<S>::noises)
        .def_readwrite("energies", &TimeEvolution<S>::energies)
        .def_readwrite("normsqs", &TimeEvolution<S>::normsqs)
        .def_readwrite("discarded_weights",
                       &TimeEvolution<S>::discarded_weights)
        .def_readwrite("forward", &TimeEvolution<S>::forward)
        .def_readwrite("n_sub_sweeps", &TimeEvolution<S>::n_sub_sweeps)
        .def_readwrite("weights", &TimeEvolution<S>::weights)
        .def_readwrite("mode", &TimeEvolution<S>::mode)
        .def_readwrite("noise_type", &TimeEvolution<S>::noise_type)
        .def_readwrite("trunc_type", &TimeEvolution<S>::trunc_type)
        .def_readwrite("trunc_pattern", &TimeEvolution<S>::trunc_pattern)
        .def_readwrite("decomp_type", &TimeEvolution<S>::decomp_type)
        .def_readwrite("normalize_mps", &TimeEvolution<S>::normalize_mps)
        .def_readwrite("hermitian", &TimeEvolution<S>::hermitian)
        .def_readwrite("sweep_cumulative_nflop",
                       &TimeEvolution<S>::sweep_cumulative_nflop)
        .def("update_one_dot", &TimeEvolution<S>::update_one_dot)
        .def("update_two_dot", &TimeEvolution<S>::update_two_dot)
        .def("update_multi_one_dot", &TimeEvolution<S>::update_multi_one_dot)
        .def("update_multi_two_dot", &TimeEvolution<S>::update_multi_two_dot)
        .def("blocking", &TimeEvolution<S>::blocking)
        .def("sweep", &TimeEvolution<S>::sweep)
        .def("normalize", &TimeEvolution<S>::normalize)
        .def("solve", &TimeEvolution<S>::solve, py::arg("n_sweeps"),
             py::arg("beta"), py::arg("forward") = true, py::arg("tol") = 1E-6);

    py::class_<typename Linear<S>::Iteration,
               shared_ptr<typename Linear<S>::Iteration>>(m, "LinearIteration")
        .def(py::init<const vector<double> &, double, int, int, int, size_t,
                      double>())
        .def(py::init<const vector<double> &, double, int, int, int>())
        .def_readwrite("mmps", &Linear<S>::Iteration::mmps)
        .def_readwrite("targets", &Linear<S>::Iteration::targets)
        .def_readwrite("error", &Linear<S>::Iteration::error)
        .def_readwrite("nmult", &Linear<S>::Iteration::nmult)
        .def_readwrite("nmultp", &Linear<S>::Iteration::nmultp)
        .def_readwrite("tmult", &Linear<S>::Iteration::tmult)
        .def_readwrite("nflop", &Linear<S>::Iteration::nflop)
        .def("__repr__", [](typename Linear<S>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Linear<S>, shared_ptr<Linear<S>>>(m, "Linear")
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<double> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<double> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const shared_ptr<MovingEnvironment<S>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<double> &>())
        .def_readwrite("iprint", &Linear<S>::iprint)
        .def_readwrite("cutoff", &Linear<S>::cutoff)
        .def_readwrite("lme", &Linear<S>::lme)
        .def_readwrite("rme", &Linear<S>::rme)
        .def_readwrite("tme", &Linear<S>::tme)
        .def_readwrite("ext_tmes", &Linear<S>::ext_tmes)
        .def_readwrite("ext_mpss", &Linear<S>::ext_mpss)
        .def_readwrite("ext_targets", &Linear<S>::ext_targets)
        .def_readwrite("ext_target_at_site", &Linear<S>::ext_target_at_site)
        .def_readwrite("bra_bond_dims", &Linear<S>::bra_bond_dims)
        .def_readwrite("ket_bond_dims", &Linear<S>::ket_bond_dims)
        .def_readwrite("target_bra_bond_dim", &Linear<S>::target_bra_bond_dim)
        .def_readwrite("target_ket_bond_dim", &Linear<S>::target_ket_bond_dim)
        .def_readwrite("noises", &Linear<S>::noises)
        .def_readwrite("targets", &Linear<S>::targets)
        .def_readwrite("discarded_weights", &Linear<S>::discarded_weights)
        .def_readwrite("sweep_targets", &Linear<S>::sweep_targets)
        .def_readwrite("sweep_discarded_weights",
                       &Linear<S>::sweep_discarded_weights)
        .def_readwrite("forward", &Linear<S>::forward)
        .def_readwrite("conv_type", &Linear<S>::conv_type)
        .def_readwrite("noise_type", &Linear<S>::noise_type)
        .def_readwrite("trunc_type", &Linear<S>::trunc_type)
        .def_readwrite("decomp_type", &Linear<S>::decomp_type)
        .def_readwrite("eq_type", &Linear<S>::eq_type)
        .def_readwrite("ex_type", &Linear<S>::ex_type)
        .def_readwrite("algo_type", &Linear<S>::algo_type)
        .def_readwrite("precondition_cg", &Linear<S>::precondition_cg)
        .def_readwrite("cg_n_harmonic_projection",
                       &Linear<S>::cg_n_harmonic_projection)
        .def_readwrite("gcrotmk_size", &Linear<S>::gcrotmk_size)
        .def_readwrite("decomp_last_site", &Linear<S>::decomp_last_site)
        .def_readwrite("sweep_cumulative_nflop",
                       &Linear<S>::sweep_cumulative_nflop)
        .def_readwrite("sweep_max_pket_size", &Linear<S>::sweep_max_pket_size)
        .def_readwrite("sweep_max_eff_ham_size",
                       &Linear<S>::sweep_max_eff_ham_size)
        .def_readwrite("minres_conv_thrds", &Linear<S>::minres_conv_thrds)
        .def_readwrite("minres_max_iter", &Linear<S>::minres_max_iter)
        .def_readwrite("minres_soft_max_iter", &Linear<S>::minres_soft_max_iter)
        .def_readwrite("conv_required_sweeps", &Linear<S>::conv_required_sweeps)
        .def_readwrite("gf_omega", &Linear<S>::gf_omega)
        .def_readwrite("gf_eta", &Linear<S>::gf_eta)
        .def_readwrite("gf_extra_omegas", &Linear<S>::gf_extra_omegas)
        .def_readwrite("gf_extra_targets", &Linear<S>::gf_extra_targets)
        .def_readwrite("gf_extra_omegas_at_site",
                       &Linear<S>::gf_extra_omegas_at_site)
        .def_readwrite("gf_extra_eta", &Linear<S>::gf_extra_eta)
        .def_readwrite("gf_extra_ext_targets", &Linear<S>::gf_extra_ext_targets)
        .def_readwrite("right_weight", &Linear<S>::right_weight)
        .def_readwrite("complex_weights", &Linear<S>::complex_weights)
        .def("update_one_dot", &Linear<S>::update_one_dot)
        .def("update_two_dot", &Linear<S>::update_two_dot)
        .def("blocking", &Linear<S>::blocking)
        .def("sweep", &Linear<S>::sweep)
        .def("solve", &Linear<S>::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6);

    bind_expect<S, double>(m, "Expect");
    bind_expect<S, complex<double>>(m, "ComplexExpect");
}

template <typename S> void bind_parallel_dmrg(py::module &m) {

    py::class_<ParallelRuleSumMPO<S>, shared_ptr<ParallelRuleSumMPO<S>>,
               ParallelRule<S>>(m, "ParallelRuleSumMPO")
        .def_readwrite("n_sites", &ParallelRuleSumMPO<S>::n_sites)
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>())
        .def(
            "index_available",
            [](ParallelRuleSumMPO<S> *self, py::args &args) -> bool {
                if (args.size() == 0)
                    return self->index_available();
                else if (args.size() == 1)
                    return self->index_available((uint16_t)args[0].cast<int>());
                else if (args.size() == 2)
                    return self->index_available((uint16_t)args[0].cast<int>(),
                                                 (uint16_t)args[1].cast<int>());
                else if (args.size() == 4)
                    return self->index_available((uint16_t)args[0].cast<int>(),
                                                 (uint16_t)args[1].cast<int>(),
                                                 (uint16_t)args[2].cast<int>(),
                                                 (uint16_t)args[3].cast<int>());
                else {
                    assert(false);
                    return false;
                }
            });

    py::class_<SumMPORule<S>, shared_ptr<SumMPORule<S>>, Rule<S>>(m,
                                                                  "SumMPORule")
        .def_readwrite("prim_rule", &SumMPORule<S>::prim_rule)
        .def_readwrite("para_rule", &SumMPORule<S>::para_rule)
        .def(py::init<const shared_ptr<Rule<S>> &,
                      const shared_ptr<ParallelRuleSumMPO<S>> &>());

    py::class_<ParallelFCIDUMP<S>, shared_ptr<ParallelFCIDUMP<S>>, FCIDUMP>(
        m, "ParallelFCIDUMP")
        .def_readwrite("rule", &ParallelFCIDUMP<S>::rule)
        .def(py::init<const shared_ptr<ParallelRuleSumMPO<S>> &>());

    py::class_<ParallelRuleQC<S>, shared_ptr<ParallelRuleQC<S>>,
               ParallelRule<S>>(m, "ParallelRuleQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleOneBodyQC<S>, shared_ptr<ParallelRuleOneBodyQC<S>>,
               ParallelRule<S>>(m, "ParallelRuleOneBodyQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleNPDMQC<S>, shared_ptr<ParallelRuleNPDMQC<S>>,
               ParallelRule<S>>(m, "ParallelRuleNPDMQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRulePDM1QC<S>, shared_ptr<ParallelRulePDM1QC<S>>,
               ParallelRule<S>>(m, "ParallelRulePDM1QC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRulePDM2QC<S>, shared_ptr<ParallelRulePDM2QC<S>>,
               ParallelRule<S>>(m, "ParallelRulePDM2QC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleSiteQC<S>, shared_ptr<ParallelRuleSiteQC<S>>,
               ParallelRule<S>>(m, "ParallelRuleSiteQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleIdentity<S>, shared_ptr<ParallelRuleIdentity<S>>,
               ParallelRule<S>>(m, "ParallelRuleIdentity")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ClassicParallelMPO<S>, shared_ptr<ClassicParallelMPO<S>>,
               MPO<S>>(m, "ClassicParallelMPO")
        .def_readwrite("prim_mpo", &ClassicParallelMPO<S>::prim_mpo)
        .def_readwrite("rule", &ClassicParallelMPO<S>::rule)
        .def(py::init<const shared_ptr<MPO<S>> &,
                      const shared_ptr<ParallelRule<S>> &>());

    py::class_<ParallelMPO<S>, shared_ptr<ParallelMPO<S>>, MPO<S>>(
        m, "ParallelMPO")
        .def_readwrite("prim_mpo", &ParallelMPO<S>::prim_mpo)
        .def_readwrite("rule", &ParallelMPO<S>::rule)
        .def(py::init<int, const shared_ptr<ParallelRule<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &,
                      const shared_ptr<ParallelRule<S>> &>());
}

template <typename S> void bind_mpo(py::module &m) {

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
        .def(py::init<uint16_t, uint16_t>())
        .def("copy", &MPOSchemer<S>::copy)
        .def("get_transform_formulas", &MPOSchemer<S>::get_transform_formulas);

    py::class_<MPO<S>, shared_ptr<MPO<S>>>(m, "MPO")
        .def(py::init<int>())
        .def_readwrite("n_sites", &MPO<S>::n_sites)
        .def_readwrite("const_e", &MPO<S>::const_e)
        .def_readwrite("tensors", &MPO<S>::tensors)
        .def_readwrite("basis", &MPO<S>::basis)
        .def_readwrite("sparse_form", &MPO<S>::sparse_form)
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
        .def_readwrite("archive_marks", &MPO<S>::archive_marks)
        .def_readwrite("archive_schemer_mark", &MPO<S>::archive_schemer_mark)
        .def_readwrite("archive_filename", &MPO<S>::archive_filename)
        .def("reduce_data", &MPO<S>::reduce_data)
        .def("load_data",
             (void (MPO<S>::*)(const string &, bool)) & MPO<S>::load_data,
             py::arg("filename"), py::arg("minimal") = false)
        .def("save_data",
             (void (MPO<S>::*)(const string &) const) & MPO<S>::save_data)
        .def("get_blocking_formulas", &MPO<S>::get_blocking_formulas)
        .def("get_ancilla_type", &MPO<S>::get_ancilla_type)
        .def("get_parallel_type", &MPO<S>::get_parallel_type)
        .def("estimate_storage", &MPO<S>::estimate_storage, py::arg("info"),
             py::arg("dot"))
        .def("deallocate", &MPO<S>::deallocate)
        .def("deep_copy", &MPO<S>::deep_copy)
        .def("__neg__",
             [](MPO<S> *self) { return -make_shared<MPO<S>>(*self); })
        .def("__mul__", [](MPO<S> *self,
                           double d) { return d * make_shared<MPO<S>>(*self); })
        .def("__rmul__", [](MPO<S> *self, double d) {
            return d * make_shared<MPO<S>>(*self);
        });

    py::class_<AntiHermitianRuleQC<S>, shared_ptr<AntiHermitianRuleQC<S>>,
               Rule<S>>(m, "AntiHermitianRuleQC")
        .def_readwrite("prim_rule", &AntiHermitianRuleQC<S>::prim_rule)
        .def(py::init<const shared_ptr<Rule<S>> &>());

    py::class_<RuleQC<S>, shared_ptr<RuleQC<S>>, Rule<S>>(m, "RuleQC")
        .def(py::init<>())
        .def(py::init<bool, bool, bool, bool, bool, bool>());

    py::class_<SimplifiedMPO<S>, shared_ptr<SimplifiedMPO<S>>, MPO<S>>(
        m, "SimplifiedMPO")
        .def_readwrite("prim_mpo", &SimplifiedMPO<S>::prim_mpo)
        .def_readwrite("rule", &SimplifiedMPO<S>::rule)
        .def_readwrite("collect_terms", &SimplifiedMPO<S>::collect_terms)
        .def_readwrite("use_intermediate", &SimplifiedMPO<S>::use_intermediate)
        .def_readwrite("intermediate_ops", &SimplifiedMPO<S>::intermediate_ops)
        .def(
            py::init<const shared_ptr<MPO<S>> &, const shared_ptr<Rule<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &, const shared_ptr<Rule<S>> &,
                      bool>())
        .def(py::init<const shared_ptr<MPO<S>> &, const shared_ptr<Rule<S>> &,
                      bool, bool>())
        .def(py::init<const shared_ptr<MPO<S>> &, const shared_ptr<Rule<S>> &,
                      bool, bool, OpNamesSet>())
        .def("simplify_expr", &SimplifiedMPO<S>::simplify_expr)
        .def("simplify_symbolic", &SimplifiedMPO<S>::simplify_symbolic)
        .def("simplify", &SimplifiedMPO<S>::simplify);

    py::class_<FusedMPO<S>, shared_ptr<FusedMPO<S>>, MPO<S>>(m, "FusedMPO")
        .def(py::init<const shared_ptr<MPO<S>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, uint16_t,
                      uint16_t>())
        .def(py::init<const shared_ptr<MPO<S>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, uint16_t,
                      uint16_t, const shared_ptr<StateInfo<S>> &>());

    py::class_<IdentityMPO<S>, shared_ptr<IdentityMPO<S>>, MPO<S>>(
        m, "IdentityMPO")
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S,
                      const shared_ptr<OperatorFunctions<S>> &>())
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S, S,
                      const shared_ptr<OperatorFunctions<S>> &>())
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S, S,
                      const shared_ptr<OperatorFunctions<S>> &,
                      const vector<uint8_t> &, const vector<uint8_t> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S>> &>());

    py::class_<SiteMPO<S>, shared_ptr<SiteMPO<S>>, MPO<S>>(m, "SiteMPO")
        .def(py::init<const shared_ptr<Hamiltonian<S>> &,
                      const shared_ptr<OpElement<S>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S>> &,
                      const shared_ptr<OpElement<S>> &, int>());

    py::class_<LocalMPO<S>, shared_ptr<LocalMPO<S>>, MPO<S>>(m, "LocalMPO")
        .def(py::init<const shared_ptr<Hamiltonian<S>> &,
                      const vector<shared_ptr<OpElement<S>>> &>());

    py::class_<MPOQC<S>, shared_ptr<MPOQC<S>>, MPO<S>>(m, "MPOQC")
        .def_readwrite("mode", &MPOQC<S>::mode)
        .def(py::init<const shared_ptr<HamiltonianQC<S>> &>())
        .def(py::init<const shared_ptr<HamiltonianQC<S>> &, QCTypes>())
        .def(py::init<const shared_ptr<HamiltonianQC<S>> &, QCTypes, int>());

    py::class_<PDM1MPOQC<S>, shared_ptr<PDM1MPOQC<S>>, MPO<S>>(m, "PDM1MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S>> &, uint8_t>())
        .def("get_matrix", &PDM1MPOQC<S>::template get_matrix<double>)
        .def("get_matrix", &PDM1MPOQC<S>::template get_matrix<complex<double>>)
        .def("get_matrix_spatial",
             &PDM1MPOQC<S>::template get_matrix_spatial<double>)
        .def("get_matrix_spatial",
             &PDM1MPOQC<S>::template get_matrix_spatial<complex<double>>);

    py::class_<NPC1MPOQC<S>, shared_ptr<NPC1MPOQC<S>>, MPO<S>>(m, "NPC1MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S>> &>());

    py::class_<AncillaMPO<S>, shared_ptr<AncillaMPO<S>>, MPO<S>>(m,
                                                                 "AncillaMPO")
        .def_readwrite("n_physical_sites", &AncillaMPO<S>::n_physical_sites)
        .def_readwrite("prim_mpo", &AncillaMPO<S>::prim_mpo)
        .def(py::init<const shared_ptr<MPO<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &, bool>());

    py::class_<ArchivedMPO<S>, shared_ptr<ArchivedMPO<S>>, MPO<S>>(
        m, "ArchivedMPO")
        .def(py::init<const shared_ptr<MPO<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &, const string &>());

    py::class_<DiagonalMPO<S>, shared_ptr<DiagonalMPO<S>>, MPO<S>>(
        m, "DiagonalMPO")
        .def(py::init<const shared_ptr<MPO<S>> &>())
        .def(py::init<const shared_ptr<MPO<S>> &,
                      const shared_ptr<Rule<S>> &>());

    py::class_<IdentityAddedMPO<S>, shared_ptr<IdentityAddedMPO<S>>, MPO<S>>(
        m, "IdentityAddedMPO")
        .def(py::init<const shared_ptr<MPO<S>> &>());
}

template <typename S> void bind_dmrg(py::module &m, const string &name) {

    bind_mps<S>(m);
    bind_partition<S>(m);
    bind_qc_hamiltonian<S>(m);
    bind_algorithms<S>(m);
    bind_mpo<S>(m);
    bind_parallel_dmrg<S>(m);
    bind_spin_specific<S>(m);
}

template <typename S, typename T>
void bind_trans_mps(py::module &m, const string &aux_name) {

    m.def(("trans_mps_info_to_" + aux_name).c_str(),
          &TransMPSInfo<S, T>::forward);
}

template <typename S, typename T>
auto bind_trans_mps_spin_specific(py::module &m, const string &aux_name)
    -> decltype(typename S::is_su2_t(typename T::is_sz_t())) {

    m.def(("trans_sparse_tensor_to_" + aux_name).c_str(),
          &TransSparseTensor<S, T>::forward);
    m.def(("trans_unfused_mps_to_" + aux_name).c_str(),
          &TransUnfusedMPS<S, T>::forward);
}

template <typename S = void> void bind_dmrg_types(py::module &m) {

    py::enum_<TruncationTypes>(m, "TruncationTypes", py::arithmetic())
        .value("Physical", TruncationTypes::Physical)
        .value("Reduced", TruncationTypes::Reduced)
        .value("ReducedInversed", TruncationTypes::ReducedInversed)
        .value("KeepOne", TruncationTypes::KeepOne)
        .def(py::self * int(), "For KeepOne: Keep X states per quantum number");

    py::enum_<DecompositionTypes>(m, "DecompositionTypes", py::arithmetic())
        .value("DensityMatrix", DecompositionTypes::DensityMatrix)
        .value("SVD", DecompositionTypes::SVD)
        .value("PureSVD", DecompositionTypes::PureSVD);

    py::enum_<SymTypes>(m, "SymTypes", py::arithmetic())
        .value("RVec", SymTypes::RVec)
        .value("CVec", SymTypes::CVec)
        .value("Mat", SymTypes::Mat);

    py::enum_<AncillaTypes>(m, "AncillaTypes", py::arithmetic())
        .value("Nothing", AncillaTypes::None)
        .value("Ancilla", AncillaTypes::Ancilla);

    py::enum_<MPSTypes>(m, "MPSTypes", py::arithmetic())
        .value("Nothing", MPSTypes::None)
        .value("MultiWfn", MPSTypes::MultiWfn)
        .value("MultiCenter", MPSTypes::MultiCenter)
        .def(py::self & py::self)
        .def(py::self | py::self)
        .def(py::self ^ py::self);

    py::enum_<ActiveTypes>(m, "ActiveTypes", py::arithmetic())
        .value("Empty", ActiveTypes::Empty)
        .value("Active", ActiveTypes::Active)
        .value("Frozen", ActiveTypes::Frozen);

    py::bind_vector<vector<ActiveTypes>>(m, "VectorActTypes");

    py::enum_<FuseTypes>(m, "FuseTypes", py::arithmetic())
        .value("NoFuseL", FuseTypes::NoFuseL)
        .value("NoFuseR", FuseTypes::NoFuseR)
        .value("FuseL", FuseTypes::FuseL)
        .value("FuseR", FuseTypes::FuseR)
        .value("FuseLR", FuseTypes::FuseLR);

    py::enum_<ExpectationAlgorithmTypes>(m, "ExpectationAlgorithmTypes",
                                         py::arithmetic())
        .value("Automatic", ExpectationAlgorithmTypes::Automatic)
        .value("Normal", ExpectationAlgorithmTypes::Normal)
        .value("Fast", ExpectationAlgorithmTypes::Fast);

    py::enum_<ExpectationTypes>(m, "ExpectationTypes", py::arithmetic())
        .value("Real", ExpectationTypes::Real)
        .value("Complex", ExpectationTypes::Complex);

    py::enum_<TETypes>(m, "TETypes", py::arithmetic())
        .value("ImagTE", TETypes::ImagTE)
        .value("RealTE", TETypes::RealTE)
        .value("TangentSpace", TETypes::TangentSpace)
        .value("RK4", TETypes::RK4)
        .value("RK4PP", TETypes::RK4PP);

    py::enum_<TruncPatternTypes>(m, "TruncPatternTypes", py::arithmetic())
        .value("Nothing", TruncPatternTypes::None)
        .value("TruncAfterOdd", TruncPatternTypes::TruncAfterOdd)
        .value("TruncAfterEven", TruncPatternTypes::TruncAfterEven);

    py::enum_<QCTypes>(m, "QCTypes", py::arithmetic())
        .value("NC", QCTypes::NC)
        .value("CN", QCTypes::CN)
        .value("NCCN", QCTypes(QCTypes::NC | QCTypes::CN))
        .value("Conventional", QCTypes::Conventional);

    py::enum_<EquationTypes>(m, "EquationTypes", py::arithmetic())
        .value("NormalMinRes", EquationTypes::NormalMinRes)
        .value("NormalCG", EquationTypes::NormalCG)
        .value("NormalGCROT", EquationTypes::NormalGCROT)
        .value("PerturbativeCompression",
               EquationTypes::PerturbativeCompression)
        .value("GreensFunction", EquationTypes::GreensFunction)
        .value("GreensFunctionSquared", EquationTypes::GreensFunctionSquared)
        .value("FitAddition", EquationTypes::FitAddition);

    py::enum_<ConvergenceTypes>(m, "ConvergenceTypes", py::arithmetic())
        .value("LastMinimal", ConvergenceTypes::LastMinimal)
        .value("LastMaximal", ConvergenceTypes::LastMaximal)
        .value("FirstMinimal", ConvergenceTypes::FirstMinimal)
        .value("FirstMaximal", ConvergenceTypes::FirstMaximal)
        .value("MiddleSite", ConvergenceTypes::MiddleSite);

    py::enum_<OpCachingTypes>(m, "OpCachingTypes", py::arithmetic())
        .value("Nothing", OpCachingTypes::None)
        .value("Left", OpCachingTypes::Left)
        .value("Right", OpCachingTypes::Right)
        .value("LeftCopy", OpCachingTypes::LeftCopy)
        .value("RightCopy", OpCachingTypes::RightCopy);
}

template <typename S = void> void bind_dmrg_io(py::module &m) {

    m.def("read_occ", &read_occ);
    m.def("write_occ", &write_occ);

    py::class_<OrbitalOrdering, shared_ptr<OrbitalOrdering>>(m,
                                                             "OrbitalOrdering")
        .def_static("exp_trans", &OrbitalOrdering::exp_trans, py::arg("mat"))
        .def_static("evaluate", &OrbitalOrdering::evaluate, py::arg("n_sites"),
                    py::arg("kmat"), py::arg("ord") = vector<uint16_t>())
        .def_static("fiedler", &OrbitalOrdering::fiedler, py::arg("n_sites"),
                    py::arg("kmat"))
        .def_static("ga_opt", &OrbitalOrdering::ga_opt, py::arg("n_sites"),
                    py::arg("kmat"), py::arg("n_generations") = 10000,
                    py::arg("n_configs") = 54, py::arg("n_elite") = 5,
                    py::arg("clone_rate") = 0.1, py::arg("mutate_rate") = 0.1);
}

#ifdef _EXPLICIT_TEMPLATE

extern template void bind_dmrg_types<>(py::module &m);
extern template void bind_dmrg_io<>(py::module &m);

extern template void bind_qc_hamiltonian<SZ>(py::module &m);
extern template void bind_parallel_dmrg<SZ>(py::module &m);
extern template void bind_mps<SZ>(py::module &m);
extern template void bind_partition<SZ>(py::module &m);
extern template void bind_algorithms<SZ>(py::module &m);
extern template void bind_mpo<SZ>(py::module &m);

extern template void bind_qc_hamiltonian<SU2>(py::module &m);
extern template void bind_parallel_dmrg<SU2>(py::module &m);
extern template void bind_mps<SU2>(py::module &m);
extern template void bind_partition<SU2>(py::module &m);
extern template void bind_algorithms<SU2>(py::module &m);
extern template void bind_mpo<SU2>(py::module &m);

extern template auto bind_spin_specific<SZ>(py::module &m)
    -> decltype(typename SZ::is_sz_t());

extern template void bind_trans_mps<SU2, SZ>(py::module &m,
                                             const string &aux_name);
extern template void bind_trans_mps<SZ, SU2>(py::module &m,
                                             const string &aux_name);
extern template auto
bind_trans_mps_spin_specific<SU2, SZ>(py::module &m, const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));

#endif
