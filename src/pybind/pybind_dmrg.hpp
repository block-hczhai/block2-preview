
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

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "../block2_core.hpp"
#include "../block2_dmrg.hpp"

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<ActiveTypes>);
// SZ
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MovingEnvironment<SZ, double, double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<EffectiveHamiltonian<SZ, double, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SZ, double>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<MovingEnvironment<SU2, double, double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<EffectiveHamiltonian<SU2, double, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SU2, double>>>);

#ifdef _USE_COMPLEX
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        shared_ptr<MovingEnvironment<SZ, complex<double>, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<EffectiveHamiltonian<SZ, complex<double>,
                                                            complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SZ, complex<double>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        shared_ptr<MovingEnvironment<SU2, complex<double>, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<
        EffectiveHamiltonian<SU2, complex<double>, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SU2, complex<double>>>>);
#endif

#ifdef _USE_SINGLE_PREC

// SZ
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MovingEnvironment<SZ, float, float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<EffectiveHamiltonian<SZ, float, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SZ, float>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MovingEnvironment<SU2, float, float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<EffectiveHamiltonian<SU2, float, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SU2, float>>>);

#ifdef _USE_COMPLEX
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<MovingEnvironment<SZ, complex<float>, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        shared_ptr<EffectiveHamiltonian<SZ, complex<float>, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SZ, complex<float>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Partition<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<MPS<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<shared_ptr<MovingEnvironment<SU2, complex<float>, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        shared_ptr<EffectiveHamiltonian<SU2, complex<float>, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseTensor<SU2, complex<float>>>>);
#endif

#endif

class checked_ostream_redirect {
  protected:
    std::streambuf *old = nullptr;
    std::ostream &costream;
    py::detail::pythonbuf buffer;

  public:
    explicit checked_ostream_redirect(
        std::ostream &costream = std::cout,
        const py::object &pyostream = py::module_::import("sys").attr("stdout"))
        : costream(costream), buffer(pyostream), old(nullptr) {
        if (!costream.fail())
            old = costream.rdbuf(&buffer);
    }
    ~checked_ostream_redirect() {
        if (old != nullptr)
            costream.rdbuf(old);
    }
    checked_ostream_redirect(const checked_ostream_redirect &) = delete;
    checked_ostream_redirect(checked_ostream_redirect &&other) = default;
    checked_ostream_redirect &
    operator=(const checked_ostream_redirect &) = delete;
    checked_ostream_redirect &operator=(checked_ostream_redirect &&) = delete;
};

class checked_estream_redirect : public checked_ostream_redirect {
  public:
    explicit checked_estream_redirect(
        std::ostream &costream = std::cerr,
        const py::object &pyostream = py::module_::import("sys").attr("stderr"))
        : checked_ostream_redirect(costream, pyostream) {}
};

template <typename S, typename FL>
auto bind_fl_spin_specific(py::module &m) -> decltype(typename S::is_su2_t()) {

    py::class_<PDM2MPOQC<S, FL>, shared_ptr<PDM2MPOQC<S, FL>>, MPO<S, FL>>(
        m, "PDM2MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &>(),
             py::arg("hamil"))
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, const string &>(),
             py::arg("hamil"), py::arg("tag"))
        .def_static("get_matrix", &PDM2MPOQC<S, FL>::get_matrix)
        .def_static("get_matrix_spatial",
                    &PDM2MPOQC<S, FL>::get_matrix_spatial);
}

template <typename S, typename FL>
auto bind_fl_spin_specific(py::module &m) -> decltype(typename S::is_sz_t()) {

    py::class_<PDM2MPOQC<S, FL>, shared_ptr<PDM2MPOQC<S, FL>>, MPO<S, FL>>(
        m, "PDM2MPOQC")
        .def_property_readonly_static(
            "s_all", [](py::object) { return PDM2MPOQC<S, FL>::s_all; })
        .def_property_readonly_static(
            "s_minimal", [](py::object) { return PDM2MPOQC<S, FL>::s_minimal; })
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &>(),
             py::arg("hamil"))
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, const string &>(),
             py::arg("hamil"), py::arg("tag"))
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, const string &,
                      uint16_t>(),
             py::arg("hamil"), py::arg("tag"), py::arg("mask"))
        .def("get_matrix", &PDM2MPOQC<S, FL>::get_matrix)
        .def("get_matrix_spatial", &PDM2MPOQC<S, FL>::get_matrix_spatial);

    py::class_<SumMPOQC<S, FL>, shared_ptr<SumMPOQC<S, FL>>, MPO<S, FL>>(
        m, "SumMPOQC")
        .def_readwrite("ts", &SumMPOQC<S, FL>::ts)
        .def(py::init<const shared_ptr<HamiltonianQC<S, FL>> &,
                      const vector<uint16_t> &>(),
             py::arg("hamil"), py::arg("pts"));
}

template <typename S, typename FL>
auto bind_fl_spin_specific(py::module &m) -> decltype(typename S::is_sg_t()) {

    py::class_<PDM2MPOQC<S, FL>, shared_ptr<PDM2MPOQC<S, FL>>, MPO<S, FL>>(
        m, "PDM2MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &>(),
             py::arg("hamil"))
        .def_static("get_matrix", &PDM2MPOQC<S, FL>::get_matrix)
        .def_static("get_matrix_spatial",
                    &PDM2MPOQC<S, FL>::get_matrix_spatial);
}

template <typename S> void bind_mps(py::module &m) {

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
             (void(MPSInfo<S>::*)(const string &)) & MPSInfo<S>::load_data)
        .def("save_data", (void(MPSInfo<S>::*)(const string &) const) &
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
        .def("set_bond_dimension_inact_ext_fci",
             &MPSInfo<S>::set_bond_dimension_inact_ext_fci)
        .def("swap_wfn_to_fused_left",
             &MPSInfo<S>::template swap_wfn_to_fused_left<double>)
        .def("swap_wfn_to_fused_right",
             &MPSInfo<S>::template swap_wfn_to_fused_right<double>)
        .def("swap_multi_wfn_to_fused_left",
             &MPSInfo<S>::template swap_multi_wfn_to_fused_left<double>)
        .def("swap_multi_wfn_to_fused_right",
             &MPSInfo<S>::template swap_multi_wfn_to_fused_right<double>)
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
        .def("deallocate", &MPSInfo<S>::deallocate)
        .def_static("condense_basis", &MPSInfo<S>::condense_basis)
        .def_static("split_basis", &MPSInfo<S>::split_basis)
        .def("split", &MPSInfo<S>::split);

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

    py::class_<NEVPTMPSInfo<S>, shared_ptr<NEVPTMPSInfo<S>>, MPSInfo<S>>(
        m, "NEVPTMPSInfo")
        .def(py::init<int, int, int, int, S, S,
                      const vector<shared_ptr<StateInfo<S>>> &>())
        .def(py::init<int, int, int, int, int, S, S,
                      const vector<shared_ptr<StateInfo<S>>> &>())
        .def_readwrite("n_inactive", &NEVPTMPSInfo<S>::n_inactive,
                       "Number of inactive orbitals")
        .def_readwrite("n_external", &NEVPTMPSInfo<S>::n_external,
                       "Number of external orbitals")
        .def_readwrite("n_ex_inactive", &NEVPTMPSInfo<S>::n_ex_inactive)
        .def_readwrite("n_ex_external", &NEVPTMPSInfo<S>::n_ex_external);

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
}

template <typename S, typename FL> void bind_fl_mps(py::module &m) {

    py::class_<SparseTensor<S, FL>, shared_ptr<SparseTensor<S, FL>>>(
        m, "SparseTensor")
        .def(py::init<>())
        .def(py::init<const vector<
                 vector<pair<pair<S, S>, shared_ptr<GTensor<FL>>>>> &>())
        .def_readwrite("data", &SparseTensor<S, FL>::data)
        .def("__repr__", [](SparseTensor<S, FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<SparseTensor<S, FL>>>>(m,
                                                             "VectorSpTensor");

    py::class_<MPS<S, FL>, shared_ptr<MPS<S, FL>>>(m, "MPS")
        .def(py::init<const shared_ptr<MPSInfo<S>> &>())
        .def(py::init<int, int, int>())
        .def(py::init([](const shared_ptr<MPS<S, FL>> &mps) {
            return make_shared<MPS<S, FL>>(*mps);
        }))
        .def_readwrite("n_sites", &MPS<S, FL>::n_sites)
        .def_readwrite("center", &MPS<S, FL>::center)
        .def_readwrite("dot", &MPS<S, FL>::dot)
        .def_readwrite("info", &MPS<S, FL>::info)
        .def_readwrite("tensors", &MPS<S, FL>::tensors)
        .def_readwrite("canonical_form", &MPS<S, FL>::canonical_form)
        .def("get_type", &MPS<S, FL>::get_type)
        .def("initialize", &MPS<S, FL>::initialize, py::arg("info"),
             py::arg("init_left") = true, py::arg("init_right") = true)
        .def("fill_thermal_limit", &MPS<S, FL>::fill_thermal_limit)
        .def("canonicalize", &MPS<S, FL>::canonicalize)
        .def("dynamic_canonicalize", &MPS<S, FL>::dynamic_canonicalize)
        .def("random_canonicalize", &MPS<S, FL>::random_canonicalize)
        .def("set_inact_ext_identity", &MPS<S, FL>::set_inact_ext_identity)
        .def("from_singlet_embedding_wfn",
             &MPS<S, FL>::from_singlet_embedding_wfn, py::arg("cg"),
             py::arg("para_rule") = nullptr)
        .def("to_singlet_embedding_wfn", &MPS<S, FL>::to_singlet_embedding_wfn,
             py::arg("cg"), py::arg("left_vacuum") = S(S::invalid),
             py::arg("para_rule") = nullptr)
        .def("move_left", &MPS<S, FL>::move_left, py::arg("cg"),
             py::arg("para_rule") = nullptr)
        .def("move_right", &MPS<S, FL>::move_right, py::arg("cg"),
             py::arg("para_rule") = nullptr)
        .def("flip_fused_form", &MPS<S, FL>::flip_fused_form)
        .def("get_filename", &MPS<S, FL>::get_filename, py::arg("i"),
             py::arg("dir") = "")
        .def("load_data", &MPS<S, FL>::load_data)
        .def("save_data", &MPS<S, FL>::save_data)
        .def("copy_data", &MPS<S, FL>::copy_data)
        .def("load_mutable", &MPS<S, FL>::load_mutable)
        .def("save_mutable", &MPS<S, FL>::save_mutable)
        .def("save_tensor", &MPS<S, FL>::save_tensor)
        .def("load_tensor", &MPS<S, FL>::load_tensor)
        .def("unload_tensor", &MPS<S, FL>::unload_tensor)
        .def("deep_copy", &MPS<S, FL>::deep_copy, py::arg("tag"))
        .def("estimate_storage", &MPS<S, FL>::estimate_storage,
             py::arg("info") = nullptr)
        .def("deallocate", &MPS<S, FL>::deallocate);

    py::bind_vector<vector<shared_ptr<MPS<S, FL>>>>(m, "VectorMPS");

    py::class_<MultiMPS<S, FL>, shared_ptr<MultiMPS<S, FL>>, MPS<S, FL>>(
        m, "MultiMPS")
        .def(py::init<const shared_ptr<MultiMPSInfo<S>> &>())
        .def(py::init<int, int, int, int>())
        .def_readwrite("nroots", &MultiMPS<S, FL>::nroots)
        .def_readwrite("wfns", &MultiMPS<S, FL>::wfns)
        .def_readwrite("weights", &MultiMPS<S, FL>::weights)
        .def("get_wfn_filename", &MultiMPS<S, FL>::get_wfn_filename)
        .def("save_wavefunction", &MultiMPS<S, FL>::save_wavefunction)
        .def("load_wavefunction", &MultiMPS<S, FL>::load_wavefunction)
        .def("unload_wavefunction", &MultiMPS<S, FL>::unload_wavefunction)
        .def("extract", &MultiMPS<S, FL>::extract)
        .def("iscale", &MultiMPS<S, FL>::iscale)
        .def("make_single", &MultiMPS<S, FL>::make_single)
        .def_static("make_complex", &MultiMPS<S, FL>::make_complex);

    py::class_<ParallelMPS<S, FL>, shared_ptr<ParallelMPS<S, FL>>, MPS<S, FL>>(
        m, "ParallelMPS")
        .def(py::init<const shared_ptr<MPSInfo<S>> &>())
        .def(py::init<int, int, int>())
        .def(py::init<const shared_ptr<MPS<S, FL>> &>())
        .def(py::init<const shared_ptr<MPSInfo<S>> &,
                      const shared_ptr<ParallelRule<S>> &>())
        .def(py::init<int, int, int, const shared_ptr<ParallelRule<S>> &>())
        .def(py::init<const shared_ptr<MPS<S, FL>> &,
                      const shared_ptr<ParallelRule<S>> &>())
        .def_readwrite("conn_centers", &ParallelMPS<S, FL>::conn_centers)
        .def_readwrite("conn_matrices", &ParallelMPS<S, FL>::conn_matrices)
        .def_readwrite("ncenter", &ParallelMPS<S, FL>::ncenter)
        .def_readwrite("ncenter", &ParallelMPS<S, FL>::ncenter)
        .def_readwrite("ncenter", &ParallelMPS<S, FL>::ncenter)
        .def_readwrite("svd_eps", &ParallelMPS<S, FL>::svd_eps)
        .def_readwrite("svd_cutoff", &ParallelMPS<S, FL>::svd_cutoff);

    py::class_<UnfusedMPS<S, FL>, shared_ptr<UnfusedMPS<S, FL>>>(m,
                                                                 "UnfusedMPS")
        .def(py::init<>())
        .def(py::init<const shared_ptr<MPS<S, FL>> &>())
        .def_readwrite("info", &UnfusedMPS<S, FL>::info)
        .def_readwrite("tensors", &UnfusedMPS<S, FL>::tensors)
        .def_readwrite("n_sites", &UnfusedMPS<S, FL>::n_sites)
        .def_readwrite("center", &UnfusedMPS<S, FL>::center)
        .def_readwrite("dot", &UnfusedMPS<S, FL>::dot)
        .def_readwrite("canonical_form", &UnfusedMPS<S, FL>::canonical_form)
        .def_static("forward_left_fused",
                    &UnfusedMPS<S, FL>::forward_left_fused, py::arg("i"),
                    py::arg("mps"), py::arg("wfn"))
        .def_static("forward_right_fused",
                    &UnfusedMPS<S, FL>::forward_right_fused, py::arg("i"),
                    py::arg("mps"), py::arg("wfn"))
        .def_static("forward_mps_tensor",
                    &UnfusedMPS<S, FL>::forward_mps_tensor, py::arg("i"),
                    py::arg("mps"))
        .def_static("backward_left_fused",
                    &UnfusedMPS<S, FL>::backward_left_fused, py::arg("i"),
                    py::arg("mps"), py::arg("spt"), py::arg("wfn"))
        .def_static("backward_right_fused",
                    &UnfusedMPS<S, FL>::backward_right_fused, py::arg("i"),
                    py::arg("mps"), py::arg("spt"), py::arg("wfn"))
        .def_static("backward_mps_tensor",
                    &UnfusedMPS<S, FL>::backward_mps_tensor, py::arg("i"),
                    py::arg("mps"), py::arg("spt"))
        .def("initialize", &UnfusedMPS<S, FL>::initialize)
        .def("finalize", &UnfusedMPS<S, FL>::finalize)
        .def("resolve_singlet_embedding",
             &UnfusedMPS<S, FL>::resolve_singlet_embedding);

    py::class_<DeterminantTRIE<S, FL>, shared_ptr<DeterminantTRIE<S, FL>>>(
        m, "DeterminantTRIE")
        .def(py::init<int>(), py::arg("n_sites"))
        .def(py::init<int, bool>(), py::arg("n_sites"),
             py::arg("enable_look_up"))
        .def_readwrite("data", &DeterminantTRIE<S, FL>::data)
        .def_readwrite("dets", &DeterminantTRIE<S, FL>::dets)
        .def_readwrite("invs", &DeterminantTRIE<S, FL>::invs)
        .def_readwrite("vals", &DeterminantTRIE<S, FL>::vals)
        .def_readwrite("n_sites", &DeterminantTRIE<S, FL>::n_sites)
        .def_readwrite("enable_look_up",
                       &DeterminantTRIE<S, FL>::enable_look_up)
        .def("clear", &DeterminantTRIE<S, FL>::clear)
        .def("copy", &DeterminantTRIE<S, FL>::copy)
        .def("__len__", &DeterminantTRIE<S, FL>::size)
        .def("append", &DeterminantTRIE<S, FL>::push_back, py::arg("det"))
        .def("find", &DeterminantTRIE<S, FL>::find, py::arg("det"))
        .def("__getitem__", &DeterminantTRIE<S, FL>::operator[], py::arg("idx"))
        .def("get_state_occupation",
             &DeterminantTRIE<S, FL>::get_state_occupation)
        .def("evaluate", &DeterminantTRIE<S, FL>::evaluate, py::arg("mps"),
             py::arg("cutoff") = 0.0, py::arg("max_rank") = -1,
             py::arg("ref") = vector<uint8_t>())
        .def("convert_phase", &DeterminantTRIE<S, FL>::convert_phase,
             py::arg("reorder"));
}

template <typename S, typename FL> void bind_fl_partition(py::module &m) {

    py::class_<Partition<S, FL>, shared_ptr<Partition<S, FL>>>(m, "Partition")
        .def(py::init<const shared_ptr<OperatorTensor<S, FL>> &,
                      const shared_ptr<OperatorTensor<S, FL>> &,
                      const shared_ptr<OperatorTensor<S, FL>> &>())
        .def(py::init<const shared_ptr<OperatorTensor<S, FL>> &,
                      const shared_ptr<OperatorTensor<S, FL>> &,
                      const shared_ptr<OperatorTensor<S, FL>> &,
                      const shared_ptr<OperatorTensor<S, FL>> &>())
        .def_readwrite("left", &Partition<S, FL>::left)
        .def_readwrite("right", &Partition<S, FL>::right)
        .def_readwrite("middle", &Partition<S, FL>::middle)
        .def_readwrite("left_op_infos", &Partition<S, FL>::left_op_infos)
        .def_readwrite("right_op_infos", &Partition<S, FL>::right_op_infos)
        .def("load_data", (void(Partition<S, FL>::*)(bool, const string &)) &
                              Partition<S, FL>::load_data)
        .def("save_data",
             (void(Partition<S, FL>::*)(bool, const string &) const) &
                 Partition<S, FL>::save_data)
        .def_static("find_op_info", &Partition<S, FL>::find_op_info)
        .def_static("build_left", &Partition<S, FL>::build_left)
        .def_static("build_right", &Partition<S, FL>::build_right)
        .def_static("get_uniq_labels", &Partition<S, FL>::get_uniq_labels)
        .def_static("get_uniq_sub_labels",
                    &Partition<S, FL>::get_uniq_sub_labels)
        .def_static("deallocate_op_infos_notrunc",
                    &Partition<S, FL>::deallocate_op_infos_notrunc)
        .def_static("copy_op_infos", &Partition<S, FL>::copy_op_infos)
        .def_static("init_left_op_infos", &Partition<S, FL>::init_left_op_infos)
        .def_static("init_left_op_infos_notrunc",
                    &Partition<S, FL>::init_left_op_infos_notrunc)
        .def_static("init_right_op_infos",
                    &Partition<S, FL>::init_right_op_infos)
        .def_static("init_right_op_infos_notrunc",
                    &Partition<S, FL>::init_right_op_infos_notrunc);

    py::bind_vector<vector<shared_ptr<Partition<S, FL>>>>(m, "VectorPartition");

    py::class_<EffectiveHamiltonian<S, FL>,
               shared_ptr<EffectiveHamiltonian<S, FL>>>(m,
                                                        "EffectiveHamiltonian")
        .def(py::init<const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const shared_ptr<DelayedOperatorTensor<S, FL>> &,
                      const shared_ptr<SparseMatrix<S, FL>> &,
                      const shared_ptr<SparseMatrix<S, FL>> &,
                      const shared_ptr<OpElement<S, FL>> &,
                      const shared_ptr<SymbolicColumnVector<S>> &, S,
                      const shared_ptr<TensorFunctions<S, FL>> &, bool>())
        .def_readwrite("left_op_infos",
                       &EffectiveHamiltonian<S, FL>::left_op_infos)
        .def_readwrite("right_op_infos",
                       &EffectiveHamiltonian<S, FL>::right_op_infos)
        .def_readwrite("op", &EffectiveHamiltonian<S, FL>::op)
        .def_readwrite("bra", &EffectiveHamiltonian<S, FL>::bra)
        .def_readwrite("ket", &EffectiveHamiltonian<S, FL>::ket)
        .def_readwrite("diag", &EffectiveHamiltonian<S, FL>::diag)
        .def_readwrite("cmat", &EffectiveHamiltonian<S, FL>::cmat)
        .def_readwrite("vmat", &EffectiveHamiltonian<S, FL>::vmat)
        .def_readwrite("tf", &EffectiveHamiltonian<S, FL>::tf)
        .def_readwrite("hop_mat", &EffectiveHamiltonian<S, FL>::hop_mat)
        .def_readwrite("opdq", &EffectiveHamiltonian<S, FL>::opdq)
        .def_readwrite("hop_left_vacuum",
                       &EffectiveHamiltonian<S, FL>::hop_left_vacuum)
        .def_readwrite("compute_diag",
                       &EffectiveHamiltonian<S, FL>::compute_diag)
        .def_readwrite("wfn_infos", &EffectiveHamiltonian<S, FL>::wfn_infos)
        .def_readwrite("operator_quanta",
                       &EffectiveHamiltonian<S, FL>::operator_quanta)
        .def_readwrite("npdm_fragment_filename",
                       &EffectiveHamiltonian<S, FL>::npdm_fragment_filename)
        .def_readwrite("npdm_scheme", &EffectiveHamiltonian<S, FL>::npdm_scheme)
        .def_readwrite("npdm_parallel_center",
                       &EffectiveHamiltonian<S, FL>::npdm_parallel_center)
        .def_readwrite("npdm_n_sites",
                       &EffectiveHamiltonian<S, FL>::npdm_n_sites)
        .def_readwrite("npdm_center", &EffectiveHamiltonian<S, FL>::npdm_center)
        .def("__call__", &EffectiveHamiltonian<S, FL>::operator(), py::arg("b"),
             py::arg("c"), py::arg("idx") = 0, py::arg("factor") = 1.0,
             py::arg("all_reduce") = true)
        .def("eigs", &EffectiveHamiltonian<S, FL>::eigs)
        .def("multiply", &EffectiveHamiltonian<S, FL>::multiply)
        .def("inverse_multiply", &EffectiveHamiltonian<S, FL>::inverse_multiply)
        .def("expect", &EffectiveHamiltonian<S, FL>::expect)
        .def("rk4_apply", &EffectiveHamiltonian<S, FL>::rk4_apply,
             py::arg("beta"), py::arg("const_e"),
             py::arg("eval_energy") = false, py::arg("para_rule") = nullptr)
        .def("expo_apply", &EffectiveHamiltonian<S, FL>::expo_apply,
             py::arg("beta"), py::arg("const_e"), py::arg("symmetric"),
             py::arg("iprint") = false, py::arg("para_rule") = nullptr)
        .def("deallocate", &EffectiveHamiltonian<S, FL>::deallocate);

    py::class_<EffectiveFunctions<S, FL>,
               shared_ptr<EffectiveFunctions<S, FL>>>(m, "EffectiveFunctions")
        .def_static("greens_function",
                    &EffectiveFunctions<S, FL>::greens_function)
        .def_static("greens_function_squared",
                    &EffectiveFunctions<S, FL>::greens_function_squared)
        .def_static("expo_apply", &EffectiveFunctions<S, FL>::expo_apply);

    py::bind_vector<vector<shared_ptr<EffectiveHamiltonian<S, FL>>>>(
        m, "VectorEffectiveHamiltonian");

    py::class_<LinearEffectiveHamiltonian<S, FL>,
               shared_ptr<LinearEffectiveHamiltonian<S, FL>>>(
        m, "LinearEffectiveHamiltonian")
        .def(py::init<const shared_ptr<EffectiveHamiltonian<S, FL>> &>())
        .def(py::init<const vector<shared_ptr<EffectiveHamiltonian<S, FL>>> &,
                      const vector<FL> &>())
        .def("__call__", &LinearEffectiveHamiltonian<S, FL>::operator(),
             py::arg("b"), py::arg("c"))
        .def("eigs", &LinearEffectiveHamiltonian<S, FL>::eigs)
        .def("deallocate", &LinearEffectiveHamiltonian<S, FL>::deallocate)
        .def_readwrite("h_effs", &LinearEffectiveHamiltonian<S, FL>::h_effs)
        .def_readwrite("coeffs", &LinearEffectiveHamiltonian<S, FL>::coeffs)
        .def("__mul__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &self,
                FL d) { return self * d; })
        .def("__rmul__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &self,
                FL d) { return self * d; })
        .def("__neg__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &self) {
                 return -self;
             })
        .def("__add__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &self,
                const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &other) {
                 return self + other;
             })
        .def("__sub__",
             [](const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &self,
                const shared_ptr<LinearEffectiveHamiltonian<S, FL>> &other) {
                 return self - other;
             });

    py::class_<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>,
               shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>>>(
        m, "EffectiveHamiltonianMultiMPS")
        .def(py::init<const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &,
                      const shared_ptr<DelayedOperatorTensor<S, FL>> &,
                      const vector<shared_ptr<SparseMatrixGroup<S, FL>>> &,
                      const vector<shared_ptr<SparseMatrixGroup<S, FL>>> &,
                      const shared_ptr<OpElement<S, FL>> &,
                      const shared_ptr<SymbolicColumnVector<S>> &, S,
                      const shared_ptr<TensorFunctions<S, FL>> &, bool>())
        .def_readwrite(
            "left_op_infos",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::left_op_infos)
        .def_readwrite(
            "right_op_infos",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::right_op_infos)
        .def_readwrite("op", &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::op)
        .def_readwrite("bra",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::bra)
        .def_readwrite("ket",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::ket)
        .def_readwrite("diag",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::diag)
        .def_readwrite("cmat",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::cmat)
        .def_readwrite("vmat",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::vmat)
        .def_readwrite("tf", &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::tf)
        .def_readwrite("hop_mat",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::hop_mat)
        .def_readwrite("opdq",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::opdq)
        .def_readwrite(
            "hop_left_vacuum",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::hop_left_vacuum)
        .def_readwrite(
            "compute_diag",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::compute_diag)
        .def_readwrite("wfn_infos",
                       &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::wfn_infos)
        .def_readwrite(
            "operator_quanta",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::operator_quanta)
        .def_readwrite(
            "npdm_fragment_filename",
            &EffectiveHamiltonian<S, FL,
                                  MultiMPS<S, FL>>::npdm_fragment_filename)
        .def_readwrite(
            "npdm_scheme",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::npdm_scheme)
        .def_readwrite(
            "npdm_parallel_center",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::npdm_parallel_center)
        .def_readwrite(
            "npdm_n_sites",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::npdm_n_sites)
        .def_readwrite(
            "npdm_center",
            &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::npdm_center)
        .def("__call__",
             &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::operator(),
             py::arg("b"), py::arg("c"), py::arg("idx") = 0,
             py::arg("factor") = 1.0, py::arg("all_reduce") = true)
        .def("eigs", &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::eigs)
        .def("expect", &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::expect)
        .def("rk4_apply",
             &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::rk4_apply,
             py::arg("beta"), py::arg("const_e"),
             py::arg("eval_energy") = false, py::arg("para_rule") = nullptr)
        .def("deallocate",
             &EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>::deallocate);
}

template <typename S, typename FL, typename FLS>
void bind_fl_moving_environment(py::module &m, const string &name) {

    py::class_<MovingEnvironment<S, FL, FLS>,
               shared_ptr<MovingEnvironment<S, FL, FLS>>>(m, name.c_str())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<MPS<S, FLS>> &,
                      const shared_ptr<MPS<S, FLS>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<MPS<S, FLS>> &,
                      const shared_ptr<MPS<S, FLS>> &, const string &>())
        .def_readwrite("n_sites", &MovingEnvironment<S, FL, FLS>::n_sites)
        .def_readwrite("center", &MovingEnvironment<S, FL, FLS>::center)
        .def_readwrite("dot", &MovingEnvironment<S, FL, FLS>::dot)
        .def_readwrite("mpo", &MovingEnvironment<S, FL, FLS>::mpo)
        .def_readwrite("bra", &MovingEnvironment<S, FL, FLS>::bra)
        .def_readwrite("ket", &MovingEnvironment<S, FL, FLS>::ket)
        .def_readwrite("envs", &MovingEnvironment<S, FL, FLS>::envs)
        .def_readwrite("tag", &MovingEnvironment<S, FL, FLS>::tag)
        .def_readwrite("para_rule", &MovingEnvironment<S, FL, FLS>::para_rule)
        .def_readwrite("tctr", &MovingEnvironment<S, FL, FLS>::tctr)
        .def_readwrite("trot", &MovingEnvironment<S, FL, FLS>::trot)
        .def_readwrite("iprint", &MovingEnvironment<S, FL, FLS>::iprint)
        .def_readwrite("delayed_contraction",
                       &MovingEnvironment<S, FL, FLS>::delayed_contraction)
        .def_readwrite("fuse_center",
                       &MovingEnvironment<S, FL, FLS>::fuse_center)
        .def_readwrite("save_partition_info",
                       &MovingEnvironment<S, FL, FLS>::save_partition_info)
        .def_readwrite("cached_opt", &MovingEnvironment<S, FL, FLS>::cached_opt)
        .def_readwrite("cached_info",
                       &MovingEnvironment<S, FL, FLS>::cached_info)
        .def_readwrite("cached_contraction",
                       &MovingEnvironment<S, FL, FLS>::cached_contraction)
        .def_readwrite(
            "fused_contraction_rotation",
            &MovingEnvironment<S, FL, FLS>::fused_contraction_rotation)
        .def("left_contract_rotate",
             &MovingEnvironment<S, FL, FLS>::left_contract_rotate)
        .def("right_contract_rotate",
             &MovingEnvironment<S, FL, FLS>::right_contract_rotate)
        .def("left_contract_rotate_unordered",
             &MovingEnvironment<S, FL, FLS>::left_contract_rotate_unordered)
        .def("right_contract_rotate_unordered",
             &MovingEnvironment<S, FL, FLS>::right_contract_rotate_unordered)
        .def("parallelize_mps", &MovingEnvironment<S, FL, FLS>::parallelize_mps)
        .def("serialize_mps", &MovingEnvironment<S, FL, FLS>::serialize_mps)
        .def(
            "left_contract",
            [](MovingEnvironment<S, FL, FLS> *self, int iL,
               vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_info) {
                shared_ptr<OperatorTensor<S, FL>> new_left = nullptr;
                self->left_contract(iL, left_op_info, new_left, false);
                return new_left;
            })
        .def("right_contract",
             [](MovingEnvironment<S, FL, FLS> *self, int iR,
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
                    &right_op_infos) {
                 shared_ptr<OperatorTensor<S, FL>> new_right = nullptr;
                 self->right_contract(iR, right_op_infos, new_right, false);
                 return new_right;
             })
        .def(
            "left_copy",
            [](MovingEnvironment<S, FL, FLS> *self, int iL,
               vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_info) {
                shared_ptr<OperatorTensor<S, FL>> new_left = nullptr;
                self->left_copy(iL, left_op_info, new_left);
                return new_left;
            })
        .def("right_copy",
             [](MovingEnvironment<S, FL, FLS> *self, int iR,
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
                    &right_op_infos) {
                 shared_ptr<OperatorTensor<S, FL>> new_right = nullptr;
                 self->right_copy(iR, right_op_infos, new_right);
                 return new_right;
             })
        .def("check_singlet_embedding",
             &MovingEnvironment<S, FL, FLS>::check_singlet_embedding)
        .def("init_environments",
             &MovingEnvironment<S, FL, FLS>::init_environments,
             py::arg("iprint") = false,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def("finalize_environments",
             &MovingEnvironment<S, FL, FLS>::finalize_environments,
             py::arg("renormalize_ops") = true,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def("prepare", &MovingEnvironment<S, FL, FLS>::prepare,
             py::arg("start_site") = 0, py::arg("end_site") = -1)
        .def("remove_partition_files",
             &MovingEnvironment<S, FL, FLS>::remove_partition_files)
        .def("move_to", &MovingEnvironment<S, FL, FLS>::move_to, py::arg("i"),
             py::arg("preserve_data") = false)
        .def("partial_prepare", &MovingEnvironment<S, FL, FLS>::partial_prepare)
        .def("get_left_archive_filename",
             &MovingEnvironment<S, FL, FLS>::get_left_archive_filename)
        .def("get_middle_archive_filename",
             &MovingEnvironment<S, FL, FLS>::get_middle_archive_filename)
        .def("get_right_archive_filename",
             &MovingEnvironment<S, FL, FLS>::get_right_archive_filename)
        .def("get_left_partition_filename",
             &MovingEnvironment<S, FL, FLS>::get_left_partition_filename)
        .def("get_right_partition_filename",
             &MovingEnvironment<S, FL, FLS>::get_right_partition_filename)
        .def("get_npdm_fragment_filename",
             &MovingEnvironment<S, FL, FLS>::get_npdm_fragment_filename)
        .def("eff_ham", &MovingEnvironment<S, FL, FLS>::eff_ham,
             py::arg("fuse_type"), py::arg("forward"), py::arg("compute_diag"),
             py::arg("bra_wfn"), py::arg("ket_wfn"))
        .def("multi_eff_ham", &MovingEnvironment<S, FL, FLS>::multi_eff_ham,
             py::arg("fuse_type"), py::arg("forward"), py::arg("compute_diag"))
        .def_static("contract_two_dot",
                    &MovingEnvironment<S, FL, FLS>::contract_two_dot,
                    py::arg("i"), py::arg("mps"), py::arg("reduced") = false)
        .def_static("wavefunction_add_noise",
                    &MovingEnvironment<S, FL, FLS>::wavefunction_add_noise,
                    py::arg("psi"), py::arg("noise"))
        .def_static("scale_perturbative_noise",
                    &MovingEnvironment<S, FL, FLS>::scale_perturbative_noise,
                    py::arg("noise"), py::arg("noise_type"), py::arg("mats"))
        .def_static("density_matrix",
                    &MovingEnvironment<S, FL, FLS>::density_matrix,
                    py::arg("vacuum"), py::arg("psi"), py::arg("trace_right"),
                    py::arg("noise"), py::arg("noise_type"),
                    py::arg("scale") = 1.0, py::arg("pkets") = nullptr)
        .def_static(
            "density_matrix_with_multi_target",
            &MovingEnvironment<S, FL, FLS>::density_matrix_with_multi_target,
            py::arg("vacuum"), py::arg("psi"), py::arg("weights"),
            py::arg("trace_right"), py::arg("noise"), py::arg("noise_type"),
            py::arg("scale") = 1.0, py::arg("pkets") = nullptr)
        .def_static("density_matrix_add_wfn",
                    &MovingEnvironment<S, FL, FLS>::density_matrix_add_wfn)
        .def_static(
            "density_matrix_add_perturbative_noise",
            &MovingEnvironment<S, FL,
                               FLS>::density_matrix_add_perturbative_noise)
        .def_static("density_matrix_add_matrices",
                    &MovingEnvironment<S, FL, FLS>::density_matrix_add_matrices)
        .def_static(
            "density_matrix_add_matrix_groups",
            &MovingEnvironment<S, FL, FLS>::density_matrix_add_matrix_groups)
        .def_static(
            "truncate_density_matrix",
            [](const shared_ptr<SparseMatrix<S, FLS>> &dm, int k,
               typename MovingEnvironment<S, FL, FLS>::FPS cutoff,
               TruncationTypes trunc_type) {
                vector<pair<int, int>> ss;
                vector<typename MovingEnvironment<S, FL, FLS>::FPS> wfn_spectra;
                auto error =
                    MovingEnvironment<S, FL, FLS>::truncate_density_matrix(
                        dm, ss, k, cutoff, false, wfn_spectra, trunc_type);
                return make_pair(error, ss);
            })
        .def_static(
            "truncate_singular_values",
            [](const vector<S> &qs,
               const vector<shared_ptr<
                   GTensor<typename MovingEnvironment<S, FL, FLS>::FPS>>> &s,
               int k, typename MovingEnvironment<S, FL, FLS>::FPS cutoff,
               TruncationTypes trunc_type) {
                vector<pair<int, int>> ss;
                vector<typename MovingEnvironment<S, FL, FLS>::FPS> wfn_spectra;
                auto error =
                    MovingEnvironment<S, FL, FLS>::truncate_singular_values(
                        qs, s, ss, k, cutoff, false, wfn_spectra, trunc_type);
                return make_pair(error, ss);
            })
        .def_static(
            "rotation_matrix_info_from_svd",
            &MovingEnvironment<S, FL, FLS>::rotation_matrix_info_from_svd,
            py::arg("opdq"), py::arg("qs"), py::arg("ts"),
            py::arg("trace_right"), py::arg("ilr"), py::arg("im"))
        .def_static(
            "wavefunction_info_from_svd",
            [](const vector<S> &qs,
               const shared_ptr<SparseMatrixInfo<S>> &wfninfo, bool trace_right,
               const vector<int> &ilr, const vector<ubond_t> &im) {
                vector<vector<int>> idx_dm_to_wfn;
                shared_ptr<SparseMatrixInfo<S>> r =
                    MovingEnvironment<S, FL, FLS>::wavefunction_info_from_svd(
                        qs, wfninfo, trace_right, ilr, im, idx_dm_to_wfn);
                return make_pair(r, idx_dm_to_wfn);
            })
        .def_static(
            "rotation_matrix_info_from_density_matrix",
            &MovingEnvironment<S, FL,
                               FLS>::rotation_matrix_info_from_density_matrix,
            py::arg("dminfo"), py::arg("trace_right"), py::arg("ilr"),
            py::arg("im"))
        .def_static("wavefunction_info_from_density_matrix",
                    [](const shared_ptr<SparseMatrixInfo<S>> &dminfo,
                       const shared_ptr<SparseMatrixInfo<S>> &wfninfo,
                       bool trace_right, const vector<int> &ilr,
                       const vector<ubond_t> &im) {
                        vector<vector<int>> idx_dm_to_wfn;
                        shared_ptr<SparseMatrixInfo<S>> r =
                            MovingEnvironment<S, FL, FLS>::
                                wavefunction_info_from_density_matrix(
                                    dminfo, wfninfo, trace_right, ilr, im,
                                    idx_dm_to_wfn);
                        return make_pair(r, idx_dm_to_wfn);
                    })
        .def_static(
            "split_density_matrix",
            [](const shared_ptr<SparseMatrix<S, FLS>> &dm,
               const shared_ptr<SparseMatrix<S, FLS>> &wfn, int k,
               bool trace_right, bool normalize,
               typename MovingEnvironment<S, FL, FLS>::FPS cutoff,
               TruncationTypes trunc_type) {
                shared_ptr<SparseMatrix<S, FLS>> left = nullptr,
                                                 right = nullptr;
                vector<typename MovingEnvironment<S, FL, FLS>::FPS> wfn_spectra;
                auto error =
                    MovingEnvironment<S, FL, FLS>::split_density_matrix(
                        dm, wfn, k, trace_right, normalize, left, right, cutoff,
                        false, wfn_spectra, trunc_type);
                return make_tuple(error, left, right);
            },
            py::arg("dm"), py::arg("wfn"), py::arg("k"), py::arg("trace_right"),
            py::arg("normalize"), py::arg("cutoff"),
            py::arg("trunc_type") = TruncationTypes::Physical)
        .def_static(
            "split_wavefunction_svd",
            [](S opdq, const shared_ptr<SparseMatrix<S, FLS>> &wfn, int k,
               bool trace_right, bool normalize,
               typename MovingEnvironment<S, FL, FLS>::FPS cutoff,
               TruncationTypes trunc_type, DecompositionTypes decomp_type) {
                shared_ptr<SparseMatrix<S, FLS>> left = nullptr,
                                                 right = nullptr;
                vector<typename MovingEnvironment<S, FL, FLS>::FPS> wfn_spectra;
                auto error =
                    MovingEnvironment<S, FL, FLS>::split_wavefunction_svd(
                        opdq, wfn, k, trace_right, normalize, left, right,
                        cutoff, false, wfn_spectra, trunc_type, decomp_type);
                return make_tuple(error, left, right);
            },
            py::arg("opdq"), py::arg("wfn"), py::arg("k"),
            py::arg("trace_right"), py::arg("normalize"), py::arg("cutoff"),
            py::arg("decomp_type") = DecompositionTypes::PureSVD,
            py::arg("trunc_type") = TruncationTypes::Physical)
        .def_static("propagate_wfn",
                    &MovingEnvironment<S, FL, FLS>::propagate_wfn, py::arg("i"),
                    py::arg("start_site"), py::arg("end_site"), py::arg("mps"),
                    py::arg("forward"), py::arg("cg"))
        .def_static("contract_multi_two_dot",
                    &MovingEnvironment<S, FL, FLS>::contract_multi_two_dot,
                    py::arg("i"), py::arg("mps"), py::arg("reduced") = false)
        .def_static(
            "multi_split_density_matrix",
            [](const shared_ptr<SparseMatrix<S, FLS>> &dm,
               const vector<shared_ptr<SparseMatrixGroup<S, FLS>>> &wfns, int k,
               bool trace_right, bool normalize,
               typename MovingEnvironment<S, FL, FLS>::FPS cutoff,
               TruncationTypes trunc_type) {
                vector<shared_ptr<SparseMatrixGroup<S, FLS>>> new_wfns;
                shared_ptr<SparseMatrix<S, FLS>> rot_mat = nullptr;
                vector<typename MovingEnvironment<S, FL, FLS>::FPS> wfn_spectra;
                auto error =
                    MovingEnvironment<S, FL, FLS>::multi_split_density_matrix(
                        dm, wfns, k, trace_right, normalize, new_wfns, rot_mat,
                        cutoff, false, wfn_spectra, trunc_type);
                return make_tuple(error, new_wfns, rot_mat);
            },
            py::arg("dm"), py::arg("wfns"), py::arg("k"),
            py::arg("trace_right"), py::arg("normalize"), py::arg("cutoff"),
            py::arg("trunc_type") = TruncationTypes::Physical)
        .def_static("propagate_multi_wfn",
                    &MovingEnvironment<S, FL, FLS>::propagate_multi_wfn,
                    py::arg("i"), py::arg("start_site"), py::arg("end_site"),
                    py::arg("mps"), py::arg("forward"), py::arg("cg"));

    py::bind_vector<vector<shared_ptr<MovingEnvironment<S, FL, FLS>>>>(
        m, ("Vector" + name).c_str());
}

template <typename S, typename FL> void bind_fl_qc_hamiltonian(py::module &m) {
    py::class_<HamiltonianQC<S, FL>, shared_ptr<HamiltonianQC<S, FL>>,
               Hamiltonian<S, FL>>(m, "HamiltonianQC")
        .def(py::init<>())
        .def(py::init<S, int, const vector<typename S::pg_t> &,
                      const shared_ptr<FCIDUMP<FL>> &>())
        .def_readwrite("fcidump", &HamiltonianQC<S, FL>::fcidump)
        .def_property(
            "mu", [](HamiltonianQC<S, FL> *self) { return self->mu; },
            [](HamiltonianQC<S, FL> *self, FL mu) { self->set_mu(mu); })
        .def_readwrite("op_prims", &HamiltonianQC<S, FL>::op_prims)
        .def("v", &HamiltonianQC<S, FL>::v)
        .def("t", &HamiltonianQC<S, FL>::t)
        .def("e", &HamiltonianQC<S, FL>::e)
        .def("init_site_ops", &HamiltonianQC<S, FL>::init_site_ops)
        .def("get_site_ops", &HamiltonianQC<S, FL>::get_site_ops);
}

template <typename S, typename FL, typename FLS, typename FLX>
void bind_fl_expect(py::module &m, const string &name) {

    py::class_<typename Expect<S, FL, FLS, FLX>::Iteration,
               shared_ptr<typename Expect<S, FL, FLS, FLX>::Iteration>>(
        m, (name + "Iteration").c_str())
        .def(py::init<const vector<pair<shared_ptr<OpExpr<S>>, FLX>> &,
                      typename Expect<S, FL, FLS, FLX>::FPS,
                      typename Expect<S, FL, FLS, FLX>::FPS, size_t, double>())
        .def(py::init<const vector<pair<shared_ptr<OpExpr<S>>, FLX>> &,
                      typename Expect<S, FL, FLS, FLX>::FPS,
                      typename Expect<S, FL, FLS, FLX>::FPS>())
        .def_readwrite("bra_error",
                       &Expect<S, FL, FLS, FLX>::Iteration::bra_error)
        .def_readwrite("ket_error",
                       &Expect<S, FL, FLS, FLX>::Iteration::ket_error)
        .def_readwrite("tmult", &Expect<S, FL, FLS, FLX>::Iteration::tmult)
        .def_readwrite("nflop", &Expect<S, FL, FLS, FLX>::Iteration::nflop)
        .def("__repr__", [](typename Expect<S, FL, FLS, FLX>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Expect<S, FL, FLS, FLX>, shared_ptr<Expect<S, FL, FLS, FLX>>>(
        m, name.c_str())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      ubond_t, ubond_t>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      ubond_t, ubond_t, typename Expect<S, FL, FLS, FLX>::FPS,
                      const vector<typename Expect<S, FL, FLS, FLX>::FPS> &,
                      const vector<int> &>())
        .def_readwrite("iprint", &Expect<S, FL, FLS, FLX>::iprint)
        .def_readwrite("cutoff", &Expect<S, FL, FLS, FLX>::cutoff)
        .def_readwrite("beta", &Expect<S, FL, FLS, FLX>::beta)
        .def_readwrite("partition_weights",
                       &Expect<S, FL, FLS, FLX>::partition_weights)
        .def_readwrite("me", &Expect<S, FL, FLS, FLX>::me)
        .def_readwrite("bra_bond_dim", &Expect<S, FL, FLS, FLX>::bra_bond_dim)
        .def_readwrite("ket_bond_dim", &Expect<S, FL, FLS, FLX>::ket_bond_dim)
        .def_readwrite("expectations", &Expect<S, FL, FLS, FLX>::expectations)
        .def_readwrite("forward", &Expect<S, FL, FLS, FLX>::forward)
        .def_readwrite("trunc_type", &Expect<S, FL, FLS, FLX>::trunc_type)
        .def_readwrite("ex_type", &Expect<S, FL, FLS, FLX>::ex_type)
        .def_readwrite("algo_type", &Expect<S, FL, FLS, FLX>::algo_type)
        .def_readwrite("zero_dot_algo", &Expect<S, FL, FLS, FLX>::zero_dot_algo)
        .def_readwrite("store_bra_spectra",
                       &Expect<S, FL, FLS, FLX>::store_bra_spectra)
        .def_readwrite("store_ket_spectra",
                       &Expect<S, FL, FLS, FLX>::store_ket_spectra)
        .def_readwrite("wfn_spectra", &Expect<S, FL, FLS, FLX>::wfn_spectra)
        .def_readwrite("sweep_wfn_spectra",
                       &Expect<S, FL, FLS, FLX>::sweep_wfn_spectra)
        .def("update_zero_dot", &Expect<S, FL, FLS, FLX>::update_zero_dot)
        .def("update_one_dot", &Expect<S, FL, FLS, FLX>::update_one_dot)
        .def("update_multi_one_dot",
             &Expect<S, FL, FLS, FLX>::update_multi_one_dot)
        .def("update_two_dot", &Expect<S, FL, FLS, FLX>::update_two_dot)
        .def("update_multi_two_dot",
             &Expect<S, FL, FLS, FLX>::update_multi_two_dot)
        .def("blocking", &Expect<S, FL, FLS, FLX>::blocking)
        .def("sweep", &Expect<S, FL, FLS, FLX>::sweep)
        .def("solve", &Expect<S, FL, FLS, FLX>::solve, py::arg("propagate"),
             py::arg("forward") = true,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def("get_1pdm_spatial", &Expect<S, FL, FLS, FLX>::get_1pdm_spatial,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1pdm", &Expect<S, FL, FLS, FLX>::get_1pdm,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_2pdm_spatial", &Expect<S, FL, FLS, FLX>::get_2pdm_spatial,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_2pdm", &Expect<S, FL, FLS, FLX>::get_2pdm,
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1npc_spatial", &Expect<S, FL, FLS, FLX>::get_1npc_spatial,
             py::arg("s"), py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_1npc", &Expect<S, FL, FLS, FLX>::get_1npc, py::arg("s"),
             py::arg("n_physical_sites") = (uint16_t)0U)
        .def("get_npdm", &Expect<S, FL, FLS, FLX>::get_npdm,
             py::arg("n_physical_sites") = (uint16_t)0U);
}

template <typename S, typename FL, typename FLS>
void bind_fl_dmrg(py::module &m) {

    py::class_<typename DMRG<S, FL, FLS>::Iteration,
               shared_ptr<typename DMRG<S, FL, FLS>::Iteration>>(
        m, "DMRGIteration")
        .def(py::init<const vector<typename DMRG<S, FL, FLS>::FPLS> &,
                      typename DMRG<S, FL, FLS>::FPS, int, int, size_t,
                      double>())
        .def(py::init<const vector<typename DMRG<S, FL, FLS>::FPLS> &,
                      typename DMRG<S, FL, FLS>::FPS, int, int>())
        .def_readwrite("mmps", &DMRG<S, FL, FLS>::Iteration::mmps)
        .def_readwrite("energies", &DMRG<S, FL, FLS>::Iteration::energies)
        .def_readwrite("error", &DMRG<S, FL, FLS>::Iteration::error)
        .def_readwrite("ndav", &DMRG<S, FL, FLS>::Iteration::ndav)
        .def_readwrite("tdav", &DMRG<S, FL, FLS>::Iteration::tdav)
        .def_readwrite("nflop", &DMRG<S, FL, FLS>::Iteration::nflop)
        .def("__repr__", [](typename DMRG<S, FL, FLS>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<DMRG<S, FL, FLS>, shared_ptr<DMRG<S, FL, FLS>>>(m, "DMRG")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRG<S, FL, FLS>::FPS> &>())
        .def_readwrite("iprint", &DMRG<S, FL, FLS>::iprint)
        .def_readwrite("cutoff", &DMRG<S, FL, FLS>::cutoff)
        .def_readwrite("quanta_cutoff", &DMRG<S, FL, FLS>::quanta_cutoff)
        .def_readwrite("me", &DMRG<S, FL, FLS>::me)
        .def_readwrite("cpx_me", &DMRG<S, FL, FLS>::cpx_me)
        .def_readwrite("ext_mes", &DMRG<S, FL, FLS>::ext_mes)
        .def_readwrite("ext_mpss", &DMRG<S, FL, FLS>::ext_mpss)
        .def_readwrite("state_specific", &DMRG<S, FL, FLS>::state_specific)
        .def_readwrite("projection_weights",
                       &DMRG<S, FL, FLS>::projection_weights)
        .def_readwrite("bond_dims", &DMRG<S, FL, FLS>::bond_dims)
        .def_readwrite("noises", &DMRG<S, FL, FLS>::noises)
        .def_readwrite("davidson_conv_thrds",
                       &DMRG<S, FL, FLS>::davidson_conv_thrds)
        .def_readwrite("davidson_max_iter",
                       &DMRG<S, FL, FLS>::davidson_max_iter)
        .def_readwrite("davidson_soft_max_iter",
                       &DMRG<S, FL, FLS>::davidson_soft_max_iter)
        .def_readwrite("davidson_shift", &DMRG<S, FL, FLS>::davidson_shift)
        .def_readwrite("davidson_type", &DMRG<S, FL, FLS>::davidson_type)
        .def_readwrite("conn_adjust_step", &DMRG<S, FL, FLS>::conn_adjust_step)
        .def_readwrite("energies", &DMRG<S, FL, FLS>::energies)
        .def_readwrite("discarded_weights",
                       &DMRG<S, FL, FLS>::discarded_weights)
        .def_readwrite("mps_quanta", &DMRG<S, FL, FLS>::mps_quanta)
        .def_readwrite("sweep_energies", &DMRG<S, FL, FLS>::sweep_energies)
        .def_readwrite("sweep_discarded_weights",
                       &DMRG<S, FL, FLS>::sweep_discarded_weights)
        .def_readwrite("sweep_quanta", &DMRG<S, FL, FLS>::sweep_quanta)
        .def_readwrite("forward", &DMRG<S, FL, FLS>::forward)
        .def_readwrite("noise_type", &DMRG<S, FL, FLS>::noise_type)
        .def_readwrite("trunc_type", &DMRG<S, FL, FLS>::trunc_type)
        .def_readwrite("decomp_type", &DMRG<S, FL, FLS>::decomp_type)
        .def_readwrite("decomp_last_site", &DMRG<S, FL, FLS>::decomp_last_site)
        .def_readwrite("sweep_cumulative_nflop",
                       &DMRG<S, FL, FLS>::sweep_cumulative_nflop)
        .def_readwrite("sweep_max_pket_size",
                       &DMRG<S, FL, FLS>::sweep_max_pket_size)
        .def_readwrite("sweep_max_eff_ham_size",
                       &DMRG<S, FL, FLS>::sweep_max_eff_ham_size)
        .def_readwrite("store_wfn_spectra",
                       &DMRG<S, FL, FLS>::store_wfn_spectra)
        .def_readwrite("wfn_spectra", &DMRG<S, FL, FLS>::wfn_spectra)
        .def_readwrite("sweep_wfn_spectra",
                       &DMRG<S, FL, FLS>::sweep_wfn_spectra)
        .def_readwrite("isweep", &DMRG<S, FL, FLS>::isweep)
        .def_readwrite("site_dependent_bond_dims",
                       &DMRG<S, FL, FLS>::site_dependent_bond_dims)
        .def_readwrite("sweep_start_site", &DMRG<S, FL, FLS>::sweep_start_site)
        .def_readwrite("sweep_end_site", &DMRG<S, FL, FLS>::sweep_end_site)
        .def("update_two_dot", &DMRG<S, FL, FLS>::update_two_dot)
        .def("update_one_dot", &DMRG<S, FL, FLS>::update_one_dot)
        .def("update_multi_two_dot", &DMRG<S, FL, FLS>::update_multi_two_dot)
        .def("update_multi_one_dot", &DMRG<S, FL, FLS>::update_multi_one_dot)
        .def("blocking", &DMRG<S, FL, FLS>::blocking)
        .def("partial_sweep", &DMRG<S, FL, FLS>::partial_sweep)
        .def("connection_sweep", &DMRG<S, FL, FLS>::connection_sweep)
        .def("unordered_sweep", &DMRG<S, FL, FLS>::unordered_sweep)
        .def("sweep", &DMRG<S, FL, FLS>::sweep)
        .def("solve", &DMRG<S, FL, FLS>::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>());
}

template <typename S, typename FL, typename FLS>
void bind_fl_td_dmrg(py::module &m) {

    py::class_<typename TDDMRG<S, FL, FLS>::Iteration,
               shared_ptr<typename TDDMRG<S, FL, FLS>::Iteration>>(
        m, "TDDMRGIteration")
        .def(py::init<typename TDDMRG<S, FL, FLS>::FLLS,
                      typename TDDMRG<S, FL, FLS>::FPS,
                      typename TDDMRG<S, FL, FLS>::FPS, int, int, size_t,
                      double>())
        .def(py::init<typename TDDMRG<S, FL, FLS>::FLLS,
                      typename TDDMRG<S, FL, FLS>::FPS,
                      typename TDDMRG<S, FL, FLS>::FPS, int, int>())
        .def_readwrite("mmps", &TDDMRG<S, FL, FLS>::Iteration::mmps)
        .def_readwrite("energy", &TDDMRG<S, FL, FLS>::Iteration::energy)
        .def_readwrite("normsq", &TDDMRG<S, FL, FLS>::Iteration::normsq)
        .def_readwrite("error", &TDDMRG<S, FL, FLS>::Iteration::error)
        .def_readwrite("nmult", &TDDMRG<S, FL, FLS>::Iteration::nmult)
        .def_readwrite("tmult", &TDDMRG<S, FL, FLS>::Iteration::tmult)
        .def_readwrite("nflop", &TDDMRG<S, FL, FLS>::Iteration::nflop)
        .def("__repr__", [](typename TDDMRG<S, FL, FLS>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<TDDMRG<S, FL, FLS>, shared_ptr<TDDMRG<S, FL, FLS>>>(m, "TDDMRG")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename TDDMRG<S, FL, FLS>::FPS> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &>())
        .def_readwrite("me", &TDDMRG<S, FL, FLS>::me)
        .def_readwrite("lme", &TDDMRG<S, FL, FLS>::lme)
        .def_readwrite("rme", &TDDMRG<S, FL, FLS>::rme)
        .def_readwrite("iprint", &TDDMRG<S, FL, FLS>::iprint)
        .def_readwrite("cutoff", &TDDMRG<S, FL, FLS>::cutoff)
        .def_readwrite("bond_dims", &TDDMRG<S, FL, FLS>::bond_dims)
        .def_readwrite("noises", &TDDMRG<S, FL, FLS>::noises)
        .def_readwrite("energies", &TDDMRG<S, FL, FLS>::energies)
        .def_readwrite("normsqs", &TDDMRG<S, FL, FLS>::normsqs)
        .def_readwrite("discarded_weights",
                       &TDDMRG<S, FL, FLS>::discarded_weights)
        .def_readwrite("forward", &TDDMRG<S, FL, FLS>::forward)
        .def_readwrite("n_sub_sweeps", &TDDMRG<S, FL, FLS>::n_sub_sweeps)
        .def_readwrite("weights", &TDDMRG<S, FL, FLS>::weights)
        .def_readwrite("mode", &TDDMRG<S, FL, FLS>::mode)
        .def_readwrite("noise_type", &TDDMRG<S, FL, FLS>::noise_type)
        .def_readwrite("trunc_type", &TDDMRG<S, FL, FLS>::trunc_type)
        .def_readwrite("decomp_type", &TDDMRG<S, FL, FLS>::decomp_type)
        .def_readwrite("decomp_last_site",
                       &TDDMRG<S, FL, FLS>::decomp_last_site)
        .def_readwrite("hermitian", &TDDMRG<S, FL, FLS>::hermitian)
        .def_readwrite("sweep_cumulative_nflop",
                       &TDDMRG<S, FL, FLS>::sweep_cumulative_nflop)
        .def_readwrite("store_wfn_spectra",
                       &TDDMRG<S, FL, FLS>::store_wfn_spectra)
        .def_readwrite("wfn_spectra", &TDDMRG<S, FL, FLS>::wfn_spectra)
        .def_readwrite("sweep_wfn_spectra",
                       &TDDMRG<S, FL, FLS>::sweep_wfn_spectra)
        .def("update_one_dot", &TDDMRG<S, FL, FLS>::update_one_dot)
        .def("update_two_dot", &TDDMRG<S, FL, FLS>::update_two_dot)
        .def("blocking", &TDDMRG<S, FL, FLS>::blocking)
        .def("sweep", &TDDMRG<S, FL, FLS>::sweep)
        .def("normalize", &TDDMRG<S, FL, FLS>::normalize)
        .def("solve", &TDDMRG<S, FL, FLS>::solve, py::arg("n_sweeps"),
             py::arg("beta"), py::arg("forward") = true, py::arg("tol") = 1E-6,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>());

    py::class_<typename TimeEvolution<S, FL, FLS>::Iteration,
               shared_ptr<typename TimeEvolution<S, FL, FLS>::Iteration>>(
        m, "TimeEvolutionIteration")
        .def(py::init<typename TimeEvolution<S, FL, FLS>::FLLS,
                      typename TimeEvolution<S, FL, FLS>::FPS,
                      typename TimeEvolution<S, FL, FLS>::FPS, int, int, int,
                      size_t, double>())
        .def(py::init<typename TimeEvolution<S, FL, FLS>::FLLS,
                      typename TimeEvolution<S, FL, FLS>::FPS,
                      typename TimeEvolution<S, FL, FLS>::FPS, int, int, int>())
        .def_readwrite("mmps", &TimeEvolution<S, FL, FLS>::Iteration::mmps)
        .def_readwrite("energy", &TimeEvolution<S, FL, FLS>::Iteration::energy)
        .def_readwrite("normsq", &TimeEvolution<S, FL, FLS>::Iteration::normsq)
        .def_readwrite("error", &TimeEvolution<S, FL, FLS>::Iteration::error)
        .def_readwrite("nexpo", &TimeEvolution<S, FL, FLS>::Iteration::nexpo)
        .def_readwrite("nexpok", &TimeEvolution<S, FL, FLS>::Iteration::nexpok)
        .def_readwrite("texpo", &TimeEvolution<S, FL, FLS>::Iteration::texpo)
        .def_readwrite("nflop", &TimeEvolution<S, FL, FLS>::Iteration::nflop)
        .def("__repr__",
             [](typename TimeEvolution<S, FL, FLS>::Iteration *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             });

    py::class_<TimeEvolution<S, FL, FLS>,
               shared_ptr<TimeEvolution<S, FL, FLS>>>(m, "TimeEvolution")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, TETypes>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, TETypes, int>())
        .def_readwrite("iprint", &TimeEvolution<S, FL, FLS>::iprint)
        .def_readwrite("cutoff", &TimeEvolution<S, FL, FLS>::cutoff)
        .def_readwrite("me", &TimeEvolution<S, FL, FLS>::me)
        .def_readwrite("bond_dims", &TimeEvolution<S, FL, FLS>::bond_dims)
        .def_readwrite("noises", &TimeEvolution<S, FL, FLS>::noises)
        .def_readwrite("energies", &TimeEvolution<S, FL, FLS>::energies)
        .def_readwrite("normsqs", &TimeEvolution<S, FL, FLS>::normsqs)
        .def_readwrite("discarded_weights",
                       &TimeEvolution<S, FL, FLS>::discarded_weights)
        .def_readwrite("forward", &TimeEvolution<S, FL, FLS>::forward)
        .def_readwrite("n_sub_sweeps", &TimeEvolution<S, FL, FLS>::n_sub_sweeps)
        .def_readwrite("weights", &TimeEvolution<S, FL, FLS>::weights)
        .def_readwrite("mode", &TimeEvolution<S, FL, FLS>::mode)
        .def_readwrite("noise_type", &TimeEvolution<S, FL, FLS>::noise_type)
        .def_readwrite("trunc_type", &TimeEvolution<S, FL, FLS>::trunc_type)
        .def_readwrite("trunc_pattern",
                       &TimeEvolution<S, FL, FLS>::trunc_pattern)
        .def_readwrite("decomp_type", &TimeEvolution<S, FL, FLS>::decomp_type)
        .def_readwrite("normalize_mps",
                       &TimeEvolution<S, FL, FLS>::normalize_mps)
        .def_readwrite("hermitian", &TimeEvolution<S, FL, FLS>::hermitian)
        .def_readwrite("sweep_cumulative_nflop",
                       &TimeEvolution<S, FL, FLS>::sweep_cumulative_nflop)
        .def_readwrite("store_wfn_spectra",
                       &TimeEvolution<S, FL, FLS>::store_wfn_spectra)
        .def_readwrite("wfn_spectra", &TimeEvolution<S, FL, FLS>::wfn_spectra)
        .def_readwrite("sweep_wfn_spectra",
                       &TimeEvolution<S, FL, FLS>::sweep_wfn_spectra)
        .def("update_one_dot", &TimeEvolution<S, FL, FLS>::update_one_dot)
        .def("update_two_dot", &TimeEvolution<S, FL, FLS>::update_two_dot)
        .def("update_multi_one_dot",
             &TimeEvolution<S, FL, FLS>::update_multi_one_dot)
        .def("update_multi_two_dot",
             &TimeEvolution<S, FL, FLS>::update_multi_two_dot)
        .def("blocking", &TimeEvolution<S, FL, FLS>::blocking)
        .def("sweep", &TimeEvolution<S, FL, FLS>::sweep)
        .def("normalize", &TimeEvolution<S, FL, FLS>::normalize)
        .def("solve", &TimeEvolution<S, FL, FLS>::solve, py::arg("n_sweeps"),
             py::arg("beta"), py::arg("forward") = true, py::arg("tol") = 1E-6,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>());
}

template <typename S, typename FL, typename FLS>
void bind_fl_linear(py::module &m) {

    py::class_<typename Linear<S, FL, FLS>::Iteration,
               shared_ptr<typename Linear<S, FL, FLS>::Iteration>>(
        m, "LinearIteration")
        .def(py::init<const vector<FLS> &, typename Linear<S, FL, FLS>::FPS,
                      int, int, int, size_t, double>())
        .def(py::init<const vector<FLS> &, typename Linear<S, FL, FLS>::FPS,
                      int, int, int>())
        .def_readwrite("mmps", &Linear<S, FL, FLS>::Iteration::mmps)
        .def_readwrite("targets", &Linear<S, FL, FLS>::Iteration::targets)
        .def_readwrite("error", &Linear<S, FL, FLS>::Iteration::error)
        .def_readwrite("nmult", &Linear<S, FL, FLS>::Iteration::nmult)
        .def_readwrite("nmultp", &Linear<S, FL, FLS>::Iteration::nmultp)
        .def_readwrite("tmult", &Linear<S, FL, FLS>::Iteration::tmult)
        .def_readwrite("nflop", &Linear<S, FL, FLS>::Iteration::nflop)
        .def("__repr__", [](typename Linear<S, FL, FLS>::Iteration *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<Linear<S, FL, FLS>, shared_ptr<Linear<S, FL, FLS>>>(m, "Linear")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<typename Linear<S, FL, FLS>::FPS> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<typename Linear<S, FL, FLS>::FPS> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<typename Linear<S, FL, FLS>::FPS> &>())
        .def_readwrite("iprint", &Linear<S, FL, FLS>::iprint)
        .def_readwrite("cutoff", &Linear<S, FL, FLS>::cutoff)
        .def_readwrite("lme", &Linear<S, FL, FLS>::lme)
        .def_readwrite("rme", &Linear<S, FL, FLS>::rme)
        .def_readwrite("tme", &Linear<S, FL, FLS>::tme)
        .def_readwrite("ext_tmes", &Linear<S, FL, FLS>::ext_tmes)
        .def_readwrite("ext_mpss", &Linear<S, FL, FLS>::ext_mpss)
        .def_readwrite("ext_targets", &Linear<S, FL, FLS>::ext_targets)
        .def_readwrite("ext_target_at_site",
                       &Linear<S, FL, FLS>::ext_target_at_site)
        .def_readwrite("bra_bond_dims", &Linear<S, FL, FLS>::bra_bond_dims)
        .def_readwrite("ket_bond_dims", &Linear<S, FL, FLS>::ket_bond_dims)
        .def_readwrite("target_bra_bond_dim",
                       &Linear<S, FL, FLS>::target_bra_bond_dim)
        .def_readwrite("target_ket_bond_dim",
                       &Linear<S, FL, FLS>::target_ket_bond_dim)
        .def_readwrite("noises", &Linear<S, FL, FLS>::noises)
        .def_readwrite("targets", &Linear<S, FL, FLS>::targets)
        .def_readwrite("discarded_weights",
                       &Linear<S, FL, FLS>::discarded_weights)
        .def_readwrite("sweep_targets", &Linear<S, FL, FLS>::sweep_targets)
        .def_readwrite("sweep_discarded_weights",
                       &Linear<S, FL, FLS>::sweep_discarded_weights)
        .def_readwrite("forward", &Linear<S, FL, FLS>::forward)
        .def_readwrite("conv_type", &Linear<S, FL, FLS>::conv_type)
        .def_readwrite("noise_type", &Linear<S, FL, FLS>::noise_type)
        .def_readwrite("trunc_type", &Linear<S, FL, FLS>::trunc_type)
        .def_readwrite("decomp_type", &Linear<S, FL, FLS>::decomp_type)
        .def_readwrite("eq_type", &Linear<S, FL, FLS>::eq_type)
        .def_readwrite("ex_type", &Linear<S, FL, FLS>::ex_type)
        .def_readwrite("algo_type", &Linear<S, FL, FLS>::algo_type)
        .def_readwrite("solver_type", &Linear<S, FL, FLS>::solver_type)
        .def_readwrite("linear_use_precondition",
                       &Linear<S, FL, FLS>::linear_use_precondition)
        .def_readwrite("cg_n_harmonic_projection",
                       &Linear<S, FL, FLS>::cg_n_harmonic_projection)
        .def_readwrite("linear_solver_params",
                       &Linear<S, FL, FLS>::linear_solver_params)
        .def_readwrite("decomp_last_site",
                       &Linear<S, FL, FLS>::decomp_last_site)
        .def_readwrite("sweep_cumulative_nflop",
                       &Linear<S, FL, FLS>::sweep_cumulative_nflop)
        .def_readwrite("sweep_max_pket_size",
                       &Linear<S, FL, FLS>::sweep_max_pket_size)
        .def_readwrite("sweep_max_eff_ham_size",
                       &Linear<S, FL, FLS>::sweep_max_eff_ham_size)
        .def_readwrite("linear_conv_thrds",
                       &Linear<S, FL, FLS>::linear_conv_thrds)
        .def_readwrite("linear_max_iter", &Linear<S, FL, FLS>::linear_max_iter)
        .def_readwrite("linear_soft_max_iter",
                       &Linear<S, FL, FLS>::linear_soft_max_iter)
        .def_readwrite("conv_required_sweeps",
                       &Linear<S, FL, FLS>::conv_required_sweeps)
        .def_readwrite("gf_omega", &Linear<S, FL, FLS>::gf_omega)
        .def_readwrite("gf_eta", &Linear<S, FL, FLS>::gf_eta)
        .def_readwrite("gf_extra_omegas", &Linear<S, FL, FLS>::gf_extra_omegas)
        .def_readwrite("gf_extra_targets",
                       &Linear<S, FL, FLS>::gf_extra_targets)
        .def_readwrite("gf_extra_omegas_at_site",
                       &Linear<S, FL, FLS>::gf_extra_omegas_at_site)
        .def_readwrite("gf_extra_eta", &Linear<S, FL, FLS>::gf_extra_eta)
        .def_readwrite("gf_extra_ext_targets",
                       &Linear<S, FL, FLS>::gf_extra_ext_targets)
        .def_readwrite("right_weight", &Linear<S, FL, FLS>::right_weight)
        .def_readwrite("complex_weights", &Linear<S, FL, FLS>::complex_weights)
        .def_readwrite("store_bra_spectra",
                       &Linear<S, FL, FLS>::store_bra_spectra)
        .def_readwrite("store_ket_spectra",
                       &Linear<S, FL, FLS>::store_ket_spectra)
        .def_readwrite("wfn_spectra", &Linear<S, FL, FLS>::wfn_spectra)
        .def_readwrite("sweep_wfn_spectra",
                       &Linear<S, FL, FLS>::sweep_wfn_spectra)
        .def_readwrite("sweep_start_site",
                       &Linear<S, FL, FLS>::sweep_start_site)
        .def_readwrite("sweep_end_site", &Linear<S, FL, FLS>::sweep_end_site)
        .def("update_one_dot", &Linear<S, FL, FLS>::update_one_dot)
        .def("update_two_dot", &Linear<S, FL, FLS>::update_two_dot)
        .def("blocking", &Linear<S, FL, FLS>::blocking)
        .def("sweep", &Linear<S, FL, FLS>::sweep)
        .def("solve", &Linear<S, FL, FLS>::solve, py::arg("n_sweeps"),
             py::arg("forward") = true, py::arg("tol") = 1E-6,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>());
}

template <typename S, typename FL> void bind_fl_parallel_dmrg(py::module &m) {

    py::class_<ParallelRuleSimple<S, FL>, shared_ptr<ParallelRuleSimple<S, FL>>,
               ParallelRule<S, FL>>(m, "ParallelRuleSimple")
        .def_readwrite("mode", &ParallelRuleSimple<S, FL>::mode)
        .def(py::init<ParallelSimpleTypes,
                      const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<ParallelSimpleTypes,
                      const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>())
        .def(
            "index_prefactor",
            [](ParallelRuleSimple<S, FL> *self, py::args &args) -> FL {
                if (args.size() == 2)
                    return self->index_prefactor((uint16_t)args[0].cast<int>(),
                                                 (uint16_t)args[1].cast<int>());
                else if (args.size() == 4)
                    return self->index_prefactor((uint16_t)args[0].cast<int>(),
                                                 (uint16_t)args[1].cast<int>(),
                                                 (uint16_t)args[2].cast<int>(),
                                                 (uint16_t)args[3].cast<int>());
                else {
                    assert(false);
                    return false;
                }
            });

    py::class_<SumMPORule<S, FL>, shared_ptr<SumMPORule<S, FL>>, Rule<S, FL>>(
        m, "SumMPORule")
        .def_readwrite("prim_rule", &SumMPORule<S, FL>::prim_rule)
        .def_readwrite("para_rule", &SumMPORule<S, FL>::para_rule)
        .def(py::init<const shared_ptr<Rule<S, FL>> &,
                      const shared_ptr<ParallelRuleSimple<S, FL>> &>());

    py::class_<ParallelFCIDUMP<S, FL>, shared_ptr<ParallelFCIDUMP<S, FL>>,
               FCIDUMP<FL>>(m, "ParallelFCIDUMP")
        .def_readwrite("rule", &ParallelFCIDUMP<S, FL>::rule)
        .def_readwrite("fcidump", &ParallelFCIDUMP<S, FL>::fcidump)
        .def(py::init<const shared_ptr<FCIDUMP<FL>> &,
                      const shared_ptr<ParallelRuleSimple<S, FL>> &>());

    py::class_<ParallelRuleQC<S, FL>, shared_ptr<ParallelRuleQC<S, FL>>,
               ParallelRule<S, FL>>(m, "ParallelRuleQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleOneBodyQC<S, FL>,
               shared_ptr<ParallelRuleOneBodyQC<S, FL>>, ParallelRule<S, FL>>(
        m, "ParallelRuleOneBodyQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleNPDMQC<S, FL>, shared_ptr<ParallelRuleNPDMQC<S, FL>>,
               ParallelRule<S, FL>>(m, "ParallelRuleNPDMQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRulePDM1QC<S, FL>, shared_ptr<ParallelRulePDM1QC<S, FL>>,
               ParallelRule<S, FL>>(m, "ParallelRulePDM1QC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRulePDM2QC<S, FL>, shared_ptr<ParallelRulePDM2QC<S, FL>>,
               ParallelRule<S, FL>>(m, "ParallelRulePDM2QC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleSiteQC<S, FL>, shared_ptr<ParallelRuleSiteQC<S, FL>>,
               ParallelRule<S, FL>>(m, "ParallelRuleSiteQC")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ParallelRuleIdentity<S, FL>,
               shared_ptr<ParallelRuleIdentity<S, FL>>, ParallelRule<S, FL>>(
        m, "ParallelRuleIdentity")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>());

    py::class_<ClassicParallelMPO<S, FL>, shared_ptr<ClassicParallelMPO<S, FL>>,
               MPO<S, FL>>(m, "ClassicParallelMPO")
        .def_readwrite("prim_mpo", &ClassicParallelMPO<S, FL>::prim_mpo)
        .def_readwrite("rule", &ClassicParallelMPO<S, FL>::rule)
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<ParallelRule<S, FL>> &>());

    py::class_<ParallelMPO<S, FL>, shared_ptr<ParallelMPO<S, FL>>, MPO<S, FL>>(
        m, "ParallelMPO")
        .def_readwrite("prim_mpo", &ParallelMPO<S, FL>::prim_mpo)
        .def_readwrite("rule", &ParallelMPO<S, FL>::rule)
        .def(py::init<int, const shared_ptr<ParallelRule<S, FL>> &>())
        .def(py::init<int, const shared_ptr<ParallelRule<S, FL>> &,
                      const string &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<ParallelRule<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<ParallelRule<S, FL>> &,
                      const string &>());
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
}

template <typename S, typename FL> void bind_fl_mpo(py::module &m) {

    py::class_<MPO<S, FL>, shared_ptr<MPO<S, FL>>>(m, "MPO")
        .def(py::init<int>())
        .def(py::init<int, const string &>())
        .def_readwrite("n_sites", &MPO<S, FL>::n_sites)
        .def_readwrite("const_e", &MPO<S, FL>::const_e)
        .def_readwrite("tensors", &MPO<S, FL>::tensors)
        .def_readwrite("basis", &MPO<S, FL>::basis)
        .def_readwrite("hamil", &MPO<S, FL>::hamil)
        .def_readwrite("sparse_form", &MPO<S, FL>::sparse_form)
        .def_readwrite("left_operator_names", &MPO<S, FL>::left_operator_names)
        .def_readwrite("right_operator_names",
                       &MPO<S, FL>::right_operator_names)
        .def_readwrite("middle_operator_names",
                       &MPO<S, FL>::middle_operator_names)
        .def_readwrite("left_operator_exprs", &MPO<S, FL>::left_operator_exprs)
        .def_readwrite("right_operator_exprs",
                       &MPO<S, FL>::right_operator_exprs)
        .def_readwrite("middle_operator_exprs",
                       &MPO<S, FL>::middle_operator_exprs)
        .def_readwrite("op", &MPO<S, FL>::op)
        .def_readwrite("left_vacuum", &MPO<S, FL>::left_vacuum)
        .def_readwrite("npdm_scheme", &MPO<S, FL>::npdm_scheme)
        .def_readwrite("npdm_parallel_center",
                       &MPO<S, FL>::npdm_parallel_center)
        .def_readwrite("schemer", &MPO<S, FL>::schemer)
        .def_readwrite("tf", &MPO<S, FL>::tf)
        .def_readwrite("site_op_infos", &MPO<S, FL>::site_op_infos)
        .def_readwrite("schemer", &MPO<S, FL>::schemer)
        .def_readwrite("archive_marks", &MPO<S, FL>::archive_marks)
        .def_readwrite("archive_schemer_mark",
                       &MPO<S, FL>::archive_schemer_mark)
        .def_readwrite("archive_filename", &MPO<S, FL>::archive_filename)
        .def_readwrite("tag", &MPO<S, FL>::tag)
        .def_readwrite("tread", &MPO<S, FL>::tread)
        .def_readwrite("twrite", &MPO<S, FL>::twrite)
        .def("get_summary", &MPO<S, FL>::get_summary)
        .def("get_filename", &MPO<S, FL>::get_filename)
        .def("load_left_operators", &MPO<S, FL>::load_left_operators)
        .def("save_left_operators", &MPO<S, FL>::save_left_operators)
        .def("unload_left_operators", &MPO<S, FL>::unload_left_operators)
        .def("load_right_operators", &MPO<S, FL>::load_right_operators)
        .def("save_right_operators", &MPO<S, FL>::save_right_operators)
        .def("unload_right_operators", &MPO<S, FL>::unload_right_operators)
        .def("load_middle_operators", &MPO<S, FL>::load_middle_operators)
        .def("save_middle_operators", &MPO<S, FL>::save_middle_operators)
        .def("unload_middle_operators", &MPO<S, FL>::unload_middle_operators)
        .def("load_tensor", &MPO<S, FL>::load_tensor, py::arg("i"),
             py::arg("no_ops") = false)
        .def("save_tensor", &MPO<S, FL>::save_tensor)
        .def("unload_tensor", &MPO<S, FL>::unload_tensor)
        .def("load_schemer", &MPO<S, FL>::load_schemer)
        .def("save_schemer", &MPO<S, FL>::save_schemer)
        .def("unload_schemer", &MPO<S, FL>::unload_schemer)
        .def("reduce_data", &MPO<S, FL>::reduce_data)
        .def("load_data",
             (void(MPO<S, FL>::*)(const string &, bool)) &
                 MPO<S, FL>::load_data,
             py::arg("filename"), py::arg("minimal") = false)
        .def("save_data",
             (void(MPO<S, FL>::*)(const string &)) & MPO<S, FL>::save_data)
        .def("get_blocking_formulas", &MPO<S, FL>::get_blocking_formulas)
        .def("get_ancilla_type", &MPO<S, FL>::get_ancilla_type)
        .def("get_parallel_type", &MPO<S, FL>::get_parallel_type)
        .def("estimate_storage", &MPO<S, FL>::estimate_storage, py::arg("info"),
             py::arg("dot"))
        .def("deallocate", &MPO<S, FL>::deallocate)
        .def("deep_copy", &MPO<S, FL>::deep_copy)
        .def("build", &MPO<S, FL>::build,
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def("__neg__",
             [](MPO<S, FL> *self) { return -make_shared<MPO<S, FL>>(*self); })
        .def("__mul__", [](MPO<S, FL> *self,
                           FL d) { return d * make_shared<MPO<S, FL>>(*self); })
        .def("__rmul__", [](MPO<S, FL> *self, FL d) {
            return d * make_shared<MPO<S, FL>>(*self);
        });

    py::class_<AntiHermitianRuleQC<S, FL>,
               shared_ptr<AntiHermitianRuleQC<S, FL>>, Rule<S, FL>>(
        m, "AntiHermitianRuleQC")
        .def_readwrite("prim_rule", &AntiHermitianRuleQC<S, FL>::prim_rule)
        .def(py::init<const shared_ptr<Rule<S, FL>> &>());

    py::class_<RuleQC<S, FL>, shared_ptr<RuleQC<S, FL>>, Rule<S, FL>>(m,
                                                                      "RuleQC")
        .def(py::init<>())
        .def(py::init<bool, bool, bool, bool, bool, bool>());

    py::class_<SimplifiedMPO<S, FL>, shared_ptr<SimplifiedMPO<S, FL>>,
               MPO<S, FL>>(m, "SimplifiedMPO")
        .def_readwrite("prim_mpo", &SimplifiedMPO<S, FL>::prim_mpo)
        .def_readwrite("rule", &SimplifiedMPO<S, FL>::rule)
        .def_readwrite("collect_terms", &SimplifiedMPO<S, FL>::collect_terms)
        .def_readwrite("use_intermediate",
                       &SimplifiedMPO<S, FL>::use_intermediate)
        .def_readwrite("intermediate_ops",
                       &SimplifiedMPO<S, FL>::intermediate_ops)
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &, bool>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &, bool, bool>())
        .def(
            py::init<const shared_ptr<MPO<S, FL>> &,
                     const shared_ptr<Rule<S, FL>> &, bool, bool, OpNamesSet>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &, bool, bool, OpNamesSet,
                      const string &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &, bool, bool, OpNamesSet,
                      const string &, bool>())
        .def("simplify_expr", &SimplifiedMPO<S, FL>::simplify_expr)
        .def("simplify_symbolic", &SimplifiedMPO<S, FL>::simplify_symbolic)
        .def("simplify", &SimplifiedMPO<S, FL>::simplify);

    py::class_<CondensedMPO<S, FL>, shared_ptr<CondensedMPO<S, FL>>,
               MPO<S, FL>>(m, "CondensedMPO")
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const vector<shared_ptr<StateInfo<S>>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, bool>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, bool,
                      const string &>());

    py::class_<FusedMPO<S, FL>, shared_ptr<FusedMPO<S, FL>>, MPO<S, FL>>(
        m, "FusedMPO")
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, uint16_t,
                      uint16_t>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, uint16_t,
                      uint16_t, const shared_ptr<StateInfo<S>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, uint16_t,
                      uint16_t, const shared_ptr<StateInfo<S>> &,
                      const string &>());

    py::class_<IdentityMPO<S, FL>, shared_ptr<IdentityMPO<S, FL>>, MPO<S, FL>>(
        m, "IdentityMPO")
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S,
                      const shared_ptr<OperatorFunctions<S, FL>> &>())
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S,
                      const shared_ptr<OperatorFunctions<S, FL>> &,
                      const string &>())
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S, S,
                      const shared_ptr<OperatorFunctions<S, FL>> &>())
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S, S,
                      const shared_ptr<OperatorFunctions<S, FL>> &,
                      const vector<typename S::pg_t> &,
                      const vector<typename S::pg_t> &>())
        .def(py::init<const vector<shared_ptr<StateInfo<S>>> &,
                      const vector<shared_ptr<StateInfo<S>>> &, S, S,
                      const shared_ptr<OperatorFunctions<S, FL>> &,
                      const vector<typename S::pg_t> &,
                      const vector<typename S::pg_t> &, const string &>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &>())
        .def(
            py::init<const shared_ptr<Hamiltonian<S, FL>> &, const string &>());

    py::class_<SiteMPO<S, FL>, shared_ptr<SiteMPO<S, FL>>, MPO<S, FL>>(
        m, "SiteMPO")
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &,
                      const shared_ptr<OpElement<S, FL>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &,
                      const shared_ptr<OpElement<S, FL>> &, int>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &,
                      const shared_ptr<OpElement<S, FL>> &, int,
                      const string &>());

    py::class_<LocalMPO<S, FL>, shared_ptr<LocalMPO<S, FL>>, MPO<S, FL>>(
        m, "LocalMPO")
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &,
                      const vector<shared_ptr<OpElement<S, FL>>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &,
                      const vector<shared_ptr<OpElement<S, FL>>> &,
                      const string &>());

    py::class_<MPOQC<S, FL>, shared_ptr<MPOQC<S, FL>>, MPO<S, FL>>(m, "MPOQC")
        .def_readwrite("mode", &MPOQC<S, FL>::mode)
        .def(py::init<const shared_ptr<HamiltonianQC<S, FL>> &>())
        .def(py::init<const shared_ptr<HamiltonianQC<S, FL>> &, QCTypes>())
        .def(py::init<const shared_ptr<HamiltonianQC<S, FL>> &, QCTypes,
                      const string &>())
        .def(py::init<const shared_ptr<HamiltonianQC<S, FL>> &, QCTypes,
                      const string &, int>())
        .def(py::init<const shared_ptr<HamiltonianQC<S, FL>> &, QCTypes,
                      const string &, int, int>());

    py::class_<PDM1MPOQC<S, FL>, shared_ptr<PDM1MPOQC<S, FL>>, MPO<S, FL>>(
        m, "PDM1MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, uint8_t>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, uint8_t,
                      const string &>())
        .def("get_matrix", &PDM1MPOQC<S, FL>::get_matrix)
        .def("get_matrix_spatial", &PDM1MPOQC<S, FL>::get_matrix_spatial);

    py::class_<NPC1MPOQC<S, FL>, shared_ptr<NPC1MPOQC<S, FL>>, MPO<S, FL>>(
        m, "NPC1MPOQC")
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, const string &>())
        .def("get_matrix", &NPC1MPOQC<S, FL>::get_matrix)
        .def("get_matrix_spatial", &NPC1MPOQC<S, FL>::get_matrix_spatial);

    py::class_<AncillaMPO<S, FL>, shared_ptr<AncillaMPO<S, FL>>, MPO<S, FL>>(
        m, "AncillaMPO")
        .def_readwrite("n_physical_sites", &AncillaMPO<S, FL>::n_physical_sites)
        .def_readwrite("prim_mpo", &AncillaMPO<S, FL>::prim_mpo)
        .def(py::init<const shared_ptr<MPO<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &, bool>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &, bool, bool>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &, bool, bool,
                      const string &>());

    py::class_<ArchivedMPO<S, FL>, shared_ptr<ArchivedMPO<S, FL>>, MPO<S, FL>>(
        m, "ArchivedMPO")
        .def(py::init<const shared_ptr<MPO<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &, const string &>());

    py::class_<DiagonalMPO<S, FL>, shared_ptr<DiagonalMPO<S, FL>>, MPO<S, FL>>(
        m, "DiagonalMPO")
        .def(py::init<const shared_ptr<MPO<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &, const string &>());

    py::class_<IdentityAddedMPO<S, FL>, shared_ptr<IdentityAddedMPO<S, FL>>,
               MPO<S, FL>>(m, "IdentityAddedMPO")
        .def(py::init<const shared_ptr<MPO<S, FL>> &>())
        .def(py::init<const shared_ptr<MPO<S, FL>> &, const string &>());
}

template <typename FL> void bind_general_fcidump(py::module &m) {

    py::class_<GeneralFCIDUMP<FL>, shared_ptr<GeneralFCIDUMP<FL>>>(
        m, "GeneralFCIDUMP")
        .def(py::init<>())
        .def_readwrite("params", &GeneralFCIDUMP<FL>::params)
        .def_readwrite("const_e", &GeneralFCIDUMP<FL>::const_e)
        .def_readwrite("exprs", &GeneralFCIDUMP<FL>::exprs)
        .def_readwrite("indices", &GeneralFCIDUMP<FL>::indices)
        .def_readwrite("data", &GeneralFCIDUMP<FL>::data)
        .def_readwrite("elem_type", &GeneralFCIDUMP<FL>::elem_type)
        .def_readwrite("order_adjusted", &GeneralFCIDUMP<FL>::order_adjusted)
        .def("add_sum_term",
             [](GeneralFCIDUMP<FL> *self, const py::array_t<FL> &v,
                typename GeneralFCIDUMP<FL>::FP cutoff) {
                 vector<int> shape(v.ndim());
                 vector<size_t> strides(v.ndim());
                 for (int i = 0; i < v.ndim(); i++)
                     shape[i] = v.shape()[i],
                     strides[i] = v.strides()[i] / sizeof(FL);
                 self->add_sum_term(v.data(), (size_t)v.size(), shape, strides,
                                    cutoff);
             })
        .def_static("initialize_from_qc",
                    &GeneralFCIDUMP<FL>::initialize_from_qc, py::arg("fcidump"),
                    py::arg("elem_type"),
                    py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("adjust_order", &GeneralFCIDUMP<FL>::adjust_order,
             py::arg("schemes") = vector<shared_ptr<SpinPermScheme>>(),
             py::arg("merge") = true,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("merge_terms", &GeneralFCIDUMP<FL>::merge_terms,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("twos", &GeneralFCIDUMP<FL>::twos)
        .def("n_sites", &GeneralFCIDUMP<FL>::n_sites)
        .def("n_elec", &GeneralFCIDUMP<FL>::n_elec)
        .def("e", &GeneralFCIDUMP<FL>::e)
        .def_property_readonly("orb_sym",
                               &GeneralFCIDUMP<FL>::template orb_sym<uint8_t>)
        .def_property_readonly("orb_sym_lz",
                               &GeneralFCIDUMP<FL>::template orb_sym<int16_t>)
        .def("__repr__", [](GeneralFCIDUMP<FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });
}

template <typename S, typename FL> void bind_fl_general(py::module &m) {

    py::class_<GeneralHamiltonian<S, FL>, shared_ptr<GeneralHamiltonian<S, FL>>,
               Hamiltonian<S, FL>>(m, "GeneralHamiltonian")
        .def(py::init<>())
        .def(py::init<S, int>())
        .def(py::init<S, int, const vector<typename S::pg_t> &>())
        .def(py::init<S, int, const vector<typename S::pg_t> &, int>())
        .def("get_site_basis", &GeneralHamiltonian<S, FL>::get_site_basis)
        .def("init_site_ops", &GeneralHamiltonian<S, FL>::init_site_ops)
        .def("get_site_string_ops",
             &GeneralHamiltonian<S, FL>::get_site_string_ops)
        .def("deallocate", &GeneralHamiltonian<S, FL>::deallocate)
        .def_static("init_string_quanta",
                    &GeneralHamiltonian<S, FL>::init_string_quanta)
        .def_static("get_sub_expr", &GeneralHamiltonian<S, FL>::get_sub_expr);

    py::class_<GeneralMPO<S, FL>, shared_ptr<GeneralMPO<S, FL>>, MPO<S, FL>>(
        m, "GeneralMPO")
        .def_readwrite("algo_type", &GeneralMPO<S, FL>::algo_type)
        .def_readwrite("discarded_weights",
                       &GeneralMPO<S, FL>::discarded_weights)
        .def_readwrite("afd", &GeneralMPO<S, FL>::afd)
        .def_readwrite("cutoff", &GeneralMPO<S, FL>::cutoff)
        .def_readwrite("max_bond_dim", &GeneralMPO<S, FL>::max_bond_dim)
        .def_readwrite("iprint", &GeneralMPO<S, FL>::iprint)
        .def_readwrite("left_vacuum", &GeneralMPO<S, FL>::left_vacuum)
        .def_readwrite("sum_mpo_mod", &GeneralMPO<S, FL>::sum_mpo_mod)
        .def_readwrite("compute_accurate_svd_error",
                       &GeneralMPO<S, FL>::compute_accurate_svd_error)
        .def_readwrite("csvd_sparsity", &GeneralMPO<S, FL>::csvd_sparsity)
        .def_readwrite("csvd_eps", &GeneralMPO<S, FL>::csvd_eps)
        .def_readwrite("csvd_max_iter", &GeneralMPO<S, FL>::csvd_max_iter)
        .def_readwrite("disjoint_levels", &GeneralMPO<S, FL>::disjoint_levels)
        .def_readwrite("disjoint_all_blocks",
                       &GeneralMPO<S, FL>::disjoint_all_blocks)
        .def_readwrite("disjoint_multiplier",
                       &GeneralMPO<S, FL>::disjoint_multiplier)
        .def_readwrite("block_max_length", &GeneralMPO<S, FL>::block_max_length)
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<GeneralFCIDUMP<FL>> &,
                      MPOAlgorithmTypes>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<GeneralFCIDUMP<FL>> &, MPOAlgorithmTypes,
                      typename GeneralMPO<S, FL>::FP>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<GeneralFCIDUMP<FL>> &, MPOAlgorithmTypes,
                      typename GeneralMPO<S, FL>::FP, int>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<GeneralFCIDUMP<FL>> &, MPOAlgorithmTypes,
                      typename GeneralMPO<S, FL>::FP, int, int>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<GeneralFCIDUMP<FL>> &, MPOAlgorithmTypes,
                      typename GeneralMPO<S, FL>::FP, int, int,
                      const string &>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>());

    py::class_<GeneralNPDMMPO<S, FL>, shared_ptr<GeneralNPDMMPO<S, FL>>,
               MPO<S, FL>>(m, "GeneralNPDMMPO")
        .def_readwrite("cutoff", &GeneralNPDMMPO<S, FL>::cutoff)
        .def_readwrite("iprint", &GeneralNPDMMPO<S, FL>::iprint)
        .def_readwrite("left_vacuum", &GeneralNPDMMPO<S, FL>::left_vacuum)
        .def_readwrite("symbol_free", &GeneralNPDMMPO<S, FL>::symbol_free)
        .def_readwrite("parallel_rule", &GeneralNPDMMPO<S, FL>::parallel_rule)
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<NPDMScheme> &>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<NPDMScheme> &, bool>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<NPDMScheme> &, bool,
                      typename GeneralMPO<S, FL>::FP>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<NPDMScheme> &, bool,
                      typename GeneralMPO<S, FL>::FP, int>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>())
        .def(py::init<const shared_ptr<GeneralHamiltonian<S, FL>> &,
                      const shared_ptr<NPDMScheme> &, bool,
                      typename GeneralMPO<S, FL>::FP, int, const string &>(),
             py::call_guard<checked_ostream_redirect,
                            checked_estream_redirect>());
}

template <typename S, typename FL>
void bind_dmrg(py::module &m, const string &name) {

    if (is_same<typename GMatrix<FL>::FP, FL>::value &&
        is_same<double, FL>::value) {
        bind_mps<S>(m);
        bind_mpo<S>(m);
    }

    bind_fl_mps<S, FL>(m);
    bind_fl_mpo<S, FL>(m);
    bind_fl_partition<S, FL>(m);
    bind_fl_qc_hamiltonian<S, FL>(m);
    bind_fl_parallel_dmrg<S, FL>(m);
    bind_fl_spin_specific<S, FL>(m);

    bind_fl_general<S, FL>(m);

    bind_fl_moving_environment<S, FL, FL>(m, "MovingEnvironment");
    if (!is_same<typename GMatrix<FL>::FP, FL>::value)
        bind_fl_moving_environment<S, FL, typename GMatrix<FL>::FP>(
            m, "MovingEnvironmentX");
    bind_fl_dmrg<S, FL, FL>(m);
    bind_fl_td_dmrg<S, FL, FL>(m);
    bind_fl_linear<S, FL, FL>(m);
    bind_fl_expect<S, FL, FL, FL>(m, "Expect");
    if (!is_same<typename GMatrix<FL>::FC, FL>::value)
        bind_fl_expect<S, FL, FL, typename GMatrix<FL>::FC>(m, "ComplexExpect");
}

template <typename S, typename T>
void bind_trans_mps(py::module &m, const string &aux_name) {

    m.def(("trans_mps_info_to_" + aux_name).c_str(),
          &TransMPSInfo<S, T>::forward);
}

template <typename S, typename FL1, typename FL2>
void bind_fl_trans_mps(py::module &m, const string &aux_name) {

    m.def(("trans_mps_to_" + aux_name).c_str(),
          &TransMPS<S, FL1, FL2>::forward);
}

template <typename S, typename T, typename FL>
void bind_fl_trans_mps_spin_specific(py::module &m, const string &aux_name) {

    m.def(("trans_sparse_tensor_to_" + aux_name).c_str(),
          &TransSparseTensor<S, T, FL>::forward);
    m.def(("trans_unfused_mps_to_" + aux_name).c_str(),
          &TransUnfusedMPS<S, T, FL>::forward);
}

template <typename S = void> void bind_dmrg_types(py::module &m) {

    py::enum_<TruncationTypes>(m, "TruncationTypes", py::arithmetic())
        .value("Physical", TruncationTypes::Physical)
        .value("Reduced", TruncationTypes::Reduced)
        .value("ReducedInversed", TruncationTypes::ReducedInversed)
        .value("SpectraWithMultiplicity",
               TruncationTypes::SpectraWithMultiplicity)
        .value("KeepOne", TruncationTypes::KeepOne)
        .value("RealDensityMatrix", TruncationTypes::RealDensityMatrix)
        .def(py::self * int(), "For KeepOne: Keep X states per quantum number")
        .def(py::self & py::self)
        .def(py::self | py::self);

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
        .value("Fast", ExpectationAlgorithmTypes::Fast)
        .value("SymbolFree", ExpectationAlgorithmTypes::SymbolFree)
        .value("Compressed", ExpectationAlgorithmTypes::Compressed)
        .value("LowMem", ExpectationAlgorithmTypes::LowMem)
        .value("CompressedSymbolFree",
               ExpectationAlgorithmTypes::Compressed |
                   ExpectationAlgorithmTypes::SymbolFree)
        .def(py::self & py::self)
        .def(py::self | py::self);

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
        .value("Normal", EquationTypes::Normal)
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

    py::enum_<LinearSolverTypes>(m, "LinearSolverTypes", py::arithmetic())
        .value("Automatic", LinearSolverTypes::Automatic)
        .value("CG", LinearSolverTypes::CG)
        .value("MinRes", LinearSolverTypes::MinRes)
        .value("GCROT", LinearSolverTypes::GCROT)
        .value("IDRS", LinearSolverTypes::IDRS)
        .value("LSQR", LinearSolverTypes::LSQR)
        .value("Cheby", LinearSolverTypes::Cheby);

    py::enum_<OpCachingTypes>(m, "OpCachingTypes", py::arithmetic())
        .value("Nothing", OpCachingTypes::None)
        .value("Left", OpCachingTypes::Left)
        .value("Right", OpCachingTypes::Right)
        .value("LeftCopy", OpCachingTypes::LeftCopy)
        .value("RightCopy", OpCachingTypes::RightCopy);

    py::enum_<ParallelSimpleTypes>(m, "ParallelSimpleTypes", py::arithmetic())
        .value("Nothing", ParallelSimpleTypes::None)
        .value("I", ParallelSimpleTypes::I)
        .value("J", ParallelSimpleTypes::J)
        .value("IJ", ParallelSimpleTypes::IJ)
        .value("KL", ParallelSimpleTypes::KL);

    py::enum_<ElemOpTypes>(m, "ElemOpTypes", py::arithmetic())
        .value("SU2", ElemOpTypes::SU2)
        .value("SZ", ElemOpTypes::SZ)
        .value("SGF", ElemOpTypes::SGF)
        .value("SGB", ElemOpTypes::SGB);

    py::enum_<MPOAlgorithmTypes>(m, "MPOAlgorithmTypes", py::arithmetic())
        .value("Nothing", MPOAlgorithmTypes::None)
        .value("Bipartite", MPOAlgorithmTypes::Bipartite)
        .value("SVD", MPOAlgorithmTypes::SVD)
        .value("Rescaled", MPOAlgorithmTypes::Rescaled)
        .value("Fast", MPOAlgorithmTypes::Fast)
        .value("Blocked", MPOAlgorithmTypes::Blocked)
        .value("Sum", MPOAlgorithmTypes::Sum)
        .value("Constrained", MPOAlgorithmTypes::Constrained)
        .value("Disjoint", MPOAlgorithmTypes::Disjoint)
        .value("NC", MPOAlgorithmTypes::NC)
        .value("CN", MPOAlgorithmTypes::CN)
        .value("DisjointSVD", MPOAlgorithmTypes::DisjointSVD)
        .value("BlockedSumDisjointSVD",
               MPOAlgorithmTypes::BlockedSumDisjointSVD)
        .value("FastBlockedSumDisjointSVD",
               MPOAlgorithmTypes::FastBlockedSumDisjointSVD)
        .value("BlockedRescaledSumDisjointSVD",
               MPOAlgorithmTypes::BlockedRescaledSumDisjointSVD)
        .value("FastBlockedRescaledSumDisjointSVD",
               MPOAlgorithmTypes::FastBlockedRescaledSumDisjointSVD)
        .value("BlockedDisjointSVD", MPOAlgorithmTypes::BlockedDisjointSVD)
        .value("FastBlockedDisjointSVD",
               MPOAlgorithmTypes::FastBlockedDisjointSVD)
        .value("BlockedRescaledDisjointSVD",
               MPOAlgorithmTypes::BlockedRescaledDisjointSVD)
        .value("FastBlockedRescaledDisjointSVD",
               MPOAlgorithmTypes::FastBlockedRescaledDisjointSVD)
        .value("RescaledDisjointSVD", MPOAlgorithmTypes::RescaledDisjointSVD)
        .value("FastDisjointSVD", MPOAlgorithmTypes::FastDisjointSVD)
        .value("FastRescaledDisjointSVD",
               MPOAlgorithmTypes::FastRescaledDisjointSVD)
        .value("ConstrainedSVD", MPOAlgorithmTypes::ConstrainedSVD)
        .value("BlockedSumConstrainedSVD",
               MPOAlgorithmTypes::BlockedSumConstrainedSVD)
        .value("FastBlockedSumConstrainedSVD",
               MPOAlgorithmTypes::FastBlockedSumConstrainedSVD)
        .value("BlockedRescaledSumConstrainedSVD",
               MPOAlgorithmTypes::BlockedRescaledSumConstrainedSVD)
        .value("FastBlockedRescaledSumConstrainedSVD",
               MPOAlgorithmTypes::FastBlockedRescaledSumConstrainedSVD)
        .value("BlockedConstrainedSVD",
               MPOAlgorithmTypes::BlockedConstrainedSVD)
        .value("FastBlockedConstrainedSVD",
               MPOAlgorithmTypes::FastBlockedConstrainedSVD)
        .value("BlockedRescaledConstrainedSVD",
               MPOAlgorithmTypes::BlockedRescaledConstrainedSVD)
        .value("FastBlockedRescaledConstrainedSVD",
               MPOAlgorithmTypes::FastBlockedRescaledConstrainedSVD)
        .value("RescaledConstrainedSVD",
               MPOAlgorithmTypes::RescaledConstrainedSVD)
        .value("FastConstrainedSVD", MPOAlgorithmTypes::FastConstrainedSVD)
        .value("FastRescaledConstrainedSVD",
               MPOAlgorithmTypes::FastRescaledConstrainedSVD)
        .value("BlockedSumSVD", MPOAlgorithmTypes::BlockedSumSVD)
        .value("FastBlockedSumSVD", MPOAlgorithmTypes::FastBlockedSumSVD)
        .value("BlockedRescaledSumSVD",
               MPOAlgorithmTypes::BlockedRescaledSumSVD)
        .value("FastBlockedRescaledSumSVD",
               MPOAlgorithmTypes::FastBlockedRescaledSumSVD)
        .value("BlockedSumBipartite", MPOAlgorithmTypes::BlockedSumBipartite)
        .value("FastBlockedSumBipartite",
               MPOAlgorithmTypes::FastBlockedSumBipartite)
        .value("BlockedSVD", MPOAlgorithmTypes::BlockedSVD)
        .value("FastBlockedSVD", MPOAlgorithmTypes::FastBlockedSVD)
        .value("BlockedRescaledSVD", MPOAlgorithmTypes::BlockedRescaledSVD)
        .value("FastBlockedRescaledSVD",
               MPOAlgorithmTypes::FastBlockedRescaledSVD)
        .value("BlockedBipartite", MPOAlgorithmTypes::BlockedBipartite)
        .value("FastBlockedBipartite", MPOAlgorithmTypes::FastBlockedBipartite)
        .value("RescaledSVD", MPOAlgorithmTypes::RescaledSVD)
        .value("FastSVD", MPOAlgorithmTypes::FastSVD)
        .value("FastRescaledSVD", MPOAlgorithmTypes::FastRescaledSVD)
        .value("FastBipartite", MPOAlgorithmTypes::FastBipartite)
        .def(py::self & py::self)
        .def(py::self | py::self);
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

extern template void bind_mps<SZ>(py::module &m);
extern template void bind_mpo<SZ>(py::module &m);

extern template void bind_general_fcidump<double>(py::module &m);
extern template void bind_fl_general<SZ, double>(py::module &m);

extern template void bind_fl_mps<SZ, double>(py::module &m);
extern template void bind_fl_mpo<SZ, double>(py::module &m);
extern template void bind_fl_partition<SZ, double>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SZ, double>(py::module &m);
extern template void bind_fl_parallel_dmrg<SZ, double>(py::module &m);

extern template void
bind_fl_moving_environment<SZ, double, double>(py::module &m,
                                               const string &name);
extern template void bind_fl_dmrg<SZ, double, double>(py::module &m);
extern template void bind_fl_td_dmrg<SZ, double, double>(py::module &m);
extern template void bind_fl_linear<SZ, double, double>(py::module &m);
extern template void
bind_fl_expect<SZ, double, double, double>(py::module &m, const string &name);
extern template void
bind_fl_expect<SZ, double, double, complex<double>>(py::module &m,
                                                    const string &name);
extern template auto bind_fl_spin_specific<SZ, double>(py::module &m)
    -> decltype(typename SZ::is_sz_t());

extern template void bind_mps<SU2>(py::module &m);
extern template void bind_mpo<SU2>(py::module &m);

extern template void bind_fl_general<SU2, double>(py::module &m);

extern template void bind_fl_mps<SU2, double>(py::module &m);
extern template void bind_fl_mpo<SU2, double>(py::module &m);
extern template void bind_fl_partition<SU2, double>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SU2, double>(py::module &m);
extern template void bind_fl_parallel_dmrg<SU2, double>(py::module &m);

extern template void
bind_fl_moving_environment<SU2, double, double>(py::module &m,
                                                const string &name);
extern template void bind_fl_dmrg<SU2, double, double>(py::module &m);
extern template void bind_fl_td_dmrg<SU2, double, double>(py::module &m);
extern template void bind_fl_linear<SU2, double, double>(py::module &m);
extern template void
bind_fl_expect<SU2, double, double, double>(py::module &m, const string &name);
extern template void
bind_fl_expect<SU2, double, double, complex<double>>(py::module &m,
                                                     const string &name);
extern template auto bind_fl_spin_specific<SU2, double>(py::module &m)
    -> decltype(typename SU2::is_su2_t());

extern template void bind_trans_mps<SU2, SZ>(py::module &m,
                                             const string &aux_name);
extern template void bind_trans_mps<SZ, SU2>(py::module &m,
                                             const string &aux_name);
extern template auto
bind_fl_trans_mps_spin_specific<SU2, SZ, double>(py::module &m,
                                                 const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));

#ifdef _USE_COMPLEX

extern template void bind_general_fcidump<complex<double>>(py::module &m);
extern template void bind_fl_general<SZ, complex<double>>(py::module &m);

extern template void bind_fl_mps<SZ, complex<double>>(py::module &m);
extern template void bind_fl_mpo<SZ, complex<double>>(py::module &m);
extern template void bind_fl_partition<SZ, complex<double>>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SZ, complex<double>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SZ, complex<double>>(py::module &m);

extern template void
bind_fl_moving_environment<SZ, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SZ, complex<double>, double>(py::module &m,
                                                        const string &name);
extern template void
bind_fl_dmrg<SZ, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_td_dmrg<SZ, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_linear<SZ, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_expect<SZ, complex<double>, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template auto bind_fl_spin_specific<SZ, complex<double>>(py::module &m)
    -> decltype(typename SZ::is_sz_t());

extern template void bind_fl_general<SU2, complex<double>>(py::module &m);

extern template void bind_fl_mps<SU2, complex<double>>(py::module &m);
extern template void bind_fl_mpo<SU2, complex<double>>(py::module &m);
extern template void bind_fl_partition<SU2, complex<double>>(py::module &m);
extern template void
bind_fl_qc_hamiltonian<SU2, complex<double>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SU2, complex<double>>(py::module &m);

extern template void
bind_fl_moving_environment<SU2, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SU2, complex<double>, double>(py::module &m,
                                                         const string &name);
extern template void
bind_fl_dmrg<SU2, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_td_dmrg<SU2, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_linear<SU2, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_expect<SU2, complex<double>, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template auto bind_fl_spin_specific<SU2, complex<double>>(py::module &m)
    -> decltype(typename SU2::is_su2_t());

extern template auto bind_fl_trans_mps_spin_specific<SU2, SZ, complex<double>>(
    py::module &m, const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));

#endif

#ifdef _USE_KSYMM

extern template void bind_mps<SZK>(py::module &m);
extern template void bind_mpo<SZK>(py::module &m);

extern template void bind_fl_general<SZK, double>(py::module &m);

extern template void bind_fl_mps<SZK, double>(py::module &m);
extern template void bind_fl_mpo<SZK, double>(py::module &m);
extern template void bind_fl_partition<SZK, double>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SZK, double>(py::module &m);
extern template void bind_fl_parallel_dmrg<SZK, double>(py::module &m);

extern template void
bind_fl_moving_environment<SZK, double, double>(py::module &m,
                                                const string &name);
extern template void bind_fl_dmrg<SZK, double, double>(py::module &m);
extern template void bind_fl_td_dmrg<SZK, double, double>(py::module &m);
extern template void bind_fl_linear<SZK, double, double>(py::module &m);
extern template void
bind_fl_expect<SZK, double, double, double>(py::module &m, const string &name);
extern template void
bind_fl_expect<SZK, double, double, complex<double>>(py::module &m,
                                                     const string &name);
extern template auto bind_fl_spin_specific<SZK, double>(py::module &m)
    -> decltype(typename SZK::is_sz_t());

extern template void bind_mps<SU2K>(py::module &m);
extern template void bind_mpo<SU2K>(py::module &m);

extern template void bind_fl_general<SU2K, double>(py::module &m);

extern template void bind_fl_mps<SU2K, double>(py::module &m);
extern template void bind_fl_mpo<SU2K, double>(py::module &m);
extern template void bind_fl_partition<SU2K, double>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SU2K, double>(py::module &m);
extern template void bind_fl_parallel_dmrg<SU2K, double>(py::module &m);

extern template void
bind_fl_moving_environment<SU2K, double, double>(py::module &m,
                                                 const string &name);
extern template void bind_fl_dmrg<SU2K, double, double>(py::module &m);
extern template void bind_fl_td_dmrg<SU2K, double, double>(py::module &m);
extern template void bind_fl_linear<SU2K, double, double>(py::module &m);
extern template void
bind_fl_expect<SU2K, double, double, double>(py::module &m, const string &name);
extern template void
bind_fl_expect<SU2K, double, double, complex<double>>(py::module &m,
                                                      const string &name);
extern template auto bind_fl_spin_specific<SU2K, double>(py::module &m)
    -> decltype(typename SU2K::is_su2_t());

extern template void bind_trans_mps<SU2K, SZK>(py::module &m,
                                               const string &aux_name);
extern template void bind_trans_mps<SZK, SU2K>(py::module &m,
                                               const string &aux_name);
extern template auto
bind_fl_trans_mps_spin_specific<SU2K, SZK, double>(py::module &m,
                                                   const string &aux_name)
    -> decltype(typename SU2K::is_su2_t(typename SZK::is_sz_t()));

#ifdef _USE_COMPLEX

extern template void bind_fl_general<SZK, complex<double>>(py::module &m);

extern template void bind_fl_mps<SZK, complex<double>>(py::module &m);
extern template void bind_fl_mpo<SZK, complex<double>>(py::module &m);
extern template void bind_fl_partition<SZK, complex<double>>(py::module &m);
extern template void
bind_fl_qc_hamiltonian<SZK, complex<double>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SZK, complex<double>>(py::module &m);

extern template void
bind_fl_moving_environment<SZK, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SZK, complex<double>, double>(py::module &m,
                                                         const string &name);
extern template void
bind_fl_dmrg<SZK, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_td_dmrg<SZK, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_linear<SZK, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_expect<SZK, complex<double>, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template auto bind_fl_spin_specific<SZK, complex<double>>(py::module &m)
    -> decltype(typename SZK::is_sz_t());

extern template void bind_fl_general<SU2K, complex<double>>(py::module &m);

extern template void bind_fl_mps<SU2K, complex<double>>(py::module &m);
extern template void bind_fl_mpo<SU2K, complex<double>>(py::module &m);
extern template void bind_fl_partition<SU2K, complex<double>>(py::module &m);
extern template void
bind_fl_qc_hamiltonian<SU2K, complex<double>>(py::module &m);
extern template void
bind_fl_parallel_dmrg<SU2K, complex<double>>(py::module &m);

extern template void
bind_fl_moving_environment<SU2K, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SU2K, complex<double>, double>(py::module &m,
                                                          const string &name);
extern template void
bind_fl_dmrg<SU2K, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_td_dmrg<SU2K, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_linear<SU2K, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_expect<SU2K, complex<double>, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template auto bind_fl_spin_specific<SU2K, complex<double>>(py::module &m)
    -> decltype(typename SU2K::is_su2_t());

extern template auto
bind_fl_trans_mps_spin_specific<SU2K, SZK, complex<double>>(
    py::module &m, const string &aux_name)
    -> decltype(typename SU2K::is_su2_t(typename SZK::is_sz_t()));

#endif

#endif

#ifdef _USE_SG

extern template void bind_mps<SGF>(py::module &m);
extern template void bind_mpo<SGF>(py::module &m);

extern template void bind_fl_general<SGF, double>(py::module &m);

extern template void bind_fl_mps<SGF, double>(py::module &m);
extern template void bind_fl_mpo<SGF, double>(py::module &m);
extern template void bind_fl_partition<SGF, double>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SGF, double>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGF, double>(py::module &m);

extern template void
bind_fl_moving_environment<SGF, double, double>(py::module &m,
                                                const string &name);
extern template void bind_fl_dmrg<SGF, double, double>(py::module &m);
extern template void bind_fl_td_dmrg<SGF, double, double>(py::module &m);
extern template void bind_fl_linear<SGF, double, double>(py::module &m);
extern template void
bind_fl_expect<SGF, double, double, double>(py::module &m, const string &name);
extern template void
bind_fl_expect<SGF, double, double, complex<double>>(py::module &m,
                                                     const string &name);
extern template auto bind_fl_spin_specific<SGF, double>(py::module &m)
    -> decltype(typename SGF::is_sg_t());

extern template void bind_mps<SGB>(py::module &m);
extern template void bind_mpo<SGB>(py::module &m);

extern template void bind_fl_general<SGB, double>(py::module &m);

extern template void bind_fl_mps<SGB, double>(py::module &m);
extern template void bind_fl_mpo<SGB, double>(py::module &m);
extern template void bind_fl_partition<SGB, double>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SGB, double>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGB, double>(py::module &m);

extern template void
bind_fl_moving_environment<SGB, double, double>(py::module &m,
                                                const string &name);
extern template void bind_fl_dmrg<SGB, double, double>(py::module &m);
extern template void bind_fl_td_dmrg<SGB, double, double>(py::module &m);
extern template void bind_fl_linear<SGB, double, double>(py::module &m);
extern template void
bind_fl_expect<SGB, double, double, double>(py::module &m, const string &name);
extern template void
bind_fl_expect<SGB, double, double, complex<double>>(py::module &m,
                                                     const string &name);
extern template void bind_trans_mps<SZ, SGF>(py::module &m,
                                             const string &aux_name);
extern template void bind_trans_mps<SGF, SZ>(py::module &m,
                                             const string &aux_name);
extern template auto
bind_fl_trans_mps_spin_specific<SZ, SGF, double>(py::module &m,
                                                 const string &aux_name)
    -> decltype(typename SZ::is_sz_t(typename SGF::is_sg_t()));
extern template auto bind_fl_spin_specific<SGB, double>(py::module &m)
    -> decltype(typename SGB::is_sg_t());

#ifdef _USE_COMPLEX

extern template void bind_fl_general<SGF, complex<double>>(py::module &m);

extern template void bind_fl_mps<SGF, complex<double>>(py::module &m);
extern template void bind_fl_mpo<SGF, complex<double>>(py::module &m);
extern template void bind_fl_partition<SGF, complex<double>>(py::module &m);
extern template void
bind_fl_qc_hamiltonian<SGF, complex<double>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGF, complex<double>>(py::module &m);

extern template void
bind_fl_moving_environment<SGF, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SGF, complex<double>, double>(py::module &m,
                                                         const string &name);
extern template void
bind_fl_dmrg<SGF, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_td_dmrg<SGF, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_linear<SGF, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_expect<SGF, complex<double>, complex<double>, complex<double>>(
    py::module &m, const string &name);

extern template void bind_fl_general<SGB, complex<double>>(py::module &m);

extern template void bind_fl_mps<SGB, complex<double>>(py::module &m);
extern template void bind_fl_mpo<SGB, complex<double>>(py::module &m);
extern template void bind_fl_partition<SGB, complex<double>>(py::module &m);
extern template void
bind_fl_qc_hamiltonian<SGB, complex<double>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGB, complex<double>>(py::module &m);

extern template void
bind_fl_moving_environment<SGB, complex<double>, complex<double>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SGB, complex<double>, double>(py::module &m,
                                                         const string &name);
extern template void
bind_fl_dmrg<SGB, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_td_dmrg<SGB, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_linear<SGB, complex<double>, complex<double>>(py::module &m);
extern template void
bind_fl_expect<SGB, complex<double>, complex<double>, complex<double>>(
    py::module &m, const string &name);

extern template auto bind_fl_trans_mps_spin_specific<SZ, SGF, complex<double>>(
    py::module &m, const string &aux_name)
    -> decltype(typename SZ::is_sz_t(typename SGF::is_sg_t()));

#endif

#endif

#ifdef _USE_SINGLE_PREC

extern template void bind_general_fcidump<float>(py::module &m);
extern template void bind_fl_general<SZ, float>(py::module &m);

extern template void bind_fl_mps<SZ, float>(py::module &m);
extern template void bind_fl_mpo<SZ, float>(py::module &m);
extern template void bind_fl_partition<SZ, float>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SZ, float>(py::module &m);
extern template void bind_fl_parallel_dmrg<SZ, float>(py::module &m);

extern template void
bind_fl_moving_environment<SZ, float, float>(py::module &m, const string &name);
extern template void bind_fl_dmrg<SZ, float, float>(py::module &m);
extern template void bind_fl_td_dmrg<SZ, float, float>(py::module &m);
extern template void bind_fl_linear<SZ, float, float>(py::module &m);
extern template void
bind_fl_expect<SZ, float, float, float>(py::module &m, const string &name);
extern template void
bind_fl_expect<SZ, float, float, complex<float>>(py::module &m,
                                                 const string &name);
extern template auto bind_fl_spin_specific<SZ, float>(py::module &m)
    -> decltype(typename SZ::is_sz_t());

extern template void bind_fl_general<SU2, float>(py::module &m);

extern template void bind_fl_mps<SU2, float>(py::module &m);
extern template void bind_fl_mpo<SU2, float>(py::module &m);
extern template void bind_fl_partition<SU2, float>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SU2, float>(py::module &m);
extern template void bind_fl_parallel_dmrg<SU2, float>(py::module &m);

extern template void
bind_fl_moving_environment<SU2, float, float>(py::module &m,
                                              const string &name);
extern template void bind_fl_dmrg<SU2, float, float>(py::module &m);
extern template void bind_fl_td_dmrg<SU2, float, float>(py::module &m);
extern template void bind_fl_linear<SU2, float, float>(py::module &m);
extern template void
bind_fl_expect<SU2, float, float, float>(py::module &m, const string &name);
extern template void
bind_fl_expect<SU2, float, float, complex<float>>(py::module &m,
                                                  const string &name);
extern template auto bind_fl_spin_specific<SU2, float>(py::module &m)
    -> decltype(typename SU2::is_su2_t());

extern template auto
bind_fl_trans_mps_spin_specific<SU2, SZ, float>(py::module &m,
                                                const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));

extern template void
bind_fl_trans_mps<SU2, float, double>(py::module &m, const string &aux_name);
extern template void
bind_fl_trans_mps<SU2, double, float>(py::module &m, const string &aux_name);
extern template void
bind_fl_trans_mps<SZ, float, double>(py::module &m, const string &aux_name);
extern template void
bind_fl_trans_mps<SZ, double, float>(py::module &m, const string &aux_name);

#ifdef _USE_COMPLEX

extern template void bind_general_fcidump<complex<float>>(py::module &m);
extern template void bind_fl_general<SZ, complex<float>>(py::module &m);

extern template void bind_fl_mps<SZ, complex<float>>(py::module &m);
extern template void bind_fl_mpo<SZ, complex<float>>(py::module &m);
extern template void bind_fl_partition<SZ, complex<float>>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SZ, complex<float>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SZ, complex<float>>(py::module &m);

extern template void
bind_fl_moving_environment<SZ, complex<float>, complex<float>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SZ, complex<float>, float>(py::module &m,
                                                      const string &name);
extern template void
bind_fl_dmrg<SZ, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_td_dmrg<SZ, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_linear<SZ, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_expect<SZ, complex<float>, complex<float>, complex<float>>(
    py::module &m, const string &name);
extern template auto bind_fl_spin_specific<SZ, complex<float>>(py::module &m)
    -> decltype(typename SZ::is_sz_t());

extern template void bind_fl_general<SU2, complex<float>>(py::module &m);

extern template void bind_fl_mps<SU2, complex<float>>(py::module &m);
extern template void bind_fl_mpo<SU2, complex<float>>(py::module &m);
extern template void bind_fl_partition<SU2, complex<float>>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SU2, complex<float>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SU2, complex<float>>(py::module &m);

extern template void
bind_fl_moving_environment<SU2, complex<float>, complex<float>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SU2, complex<float>, float>(py::module &m,
                                                       const string &name);
extern template void
bind_fl_dmrg<SU2, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_td_dmrg<SU2, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_linear<SU2, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_expect<SU2, complex<float>, complex<float>, complex<float>>(
    py::module &m, const string &name);
extern template auto bind_fl_spin_specific<SU2, complex<float>>(py::module &m)
    -> decltype(typename SU2::is_su2_t());

extern template auto
bind_fl_trans_mps_spin_specific<SU2, SZ, complex<float>>(py::module &m,
                                                         const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));

extern template void
bind_fl_trans_mps<SU2, complex<float>, complex<double>>(py::module &m,
                                                        const string &aux_name);
extern template void
bind_fl_trans_mps<SU2, complex<double>, complex<float>>(py::module &m,
                                                        const string &aux_name);
extern template void
bind_fl_trans_mps<SZ, complex<float>, complex<double>>(py::module &m,
                                                       const string &aux_name);
extern template void
bind_fl_trans_mps<SZ, complex<double>, complex<float>>(py::module &m,
                                                       const string &aux_name);

#endif

#ifdef _USE_SG

extern template void bind_fl_general<SGF, float>(py::module &m);

extern template void bind_fl_mps<SGF, float>(py::module &m);
extern template void bind_fl_mpo<SGF, float>(py::module &m);
extern template void bind_fl_partition<SGF, float>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SGF, float>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGF, float>(py::module &m);

extern template void
bind_fl_moving_environment<SGF, float, float>(py::module &m,
                                              const string &name);
extern template void bind_fl_dmrg<SGF, float, float>(py::module &m);
extern template void bind_fl_td_dmrg<SGF, float, float>(py::module &m);
extern template void bind_fl_linear<SGF, float, float>(py::module &m);
extern template void
bind_fl_expect<SGF, float, float, float>(py::module &m, const string &name);
extern template void
bind_fl_expect<SGF, float, float, complex<float>>(py::module &m,
                                                  const string &name);
extern template auto bind_fl_spin_specific<SGF, float>(py::module &m)
    -> decltype(typename SGF::is_sg_t());

extern template void bind_fl_general<SGB, float>(py::module &m);

extern template void bind_fl_mps<SGB, float>(py::module &m);
extern template void bind_fl_mpo<SGB, float>(py::module &m);
extern template void bind_fl_partition<SGB, float>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SGB, float>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGB, float>(py::module &m);

extern template void
bind_fl_moving_environment<SGB, float, float>(py::module &m,
                                              const string &name);
extern template void bind_fl_dmrg<SGB, float, float>(py::module &m);
extern template void bind_fl_td_dmrg<SGB, float, float>(py::module &m);
extern template void bind_fl_linear<SGB, float, float>(py::module &m);
extern template void
bind_fl_expect<SGB, float, float, float>(py::module &m, const string &name);
extern template void
bind_fl_expect<SGB, float, float, complex<float>>(py::module &m,
                                                  const string &name);
extern template auto bind_fl_spin_specific<SGB, float>(py::module &m)
    -> decltype(typename SGB::is_sg_t());

extern template auto
bind_fl_trans_mps_spin_specific<SZ, SGF, float>(py::module &m,
                                                const string &aux_name)
    -> decltype(typename SZ::is_sz_t(typename SGF::is_sg_t()));

extern template void
bind_fl_trans_mps<SGF, float, double>(py::module &m, const string &aux_name);
extern template void
bind_fl_trans_mps<SGF, double, float>(py::module &m, const string &aux_name);
extern template void
bind_fl_trans_mps<SGB, float, double>(py::module &m, const string &aux_name);
extern template void
bind_fl_trans_mps<SGB, double, float>(py::module &m, const string &aux_name);

#ifdef _USE_COMPLEX

extern template void bind_fl_general<SGF, complex<float>>(py::module &m);

extern template void bind_fl_mps<SGF, complex<float>>(py::module &m);
extern template void bind_fl_mpo<SGF, complex<float>>(py::module &m);
extern template void bind_fl_partition<SGF, complex<float>>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SGF, complex<float>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGF, complex<float>>(py::module &m);

extern template void
bind_fl_moving_environment<SGF, complex<float>, complex<float>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SGF, complex<float>, float>(py::module &m,
                                                       const string &name);
extern template void
bind_fl_dmrg<SGF, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_td_dmrg<SGF, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_linear<SGF, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_expect<SGF, complex<float>, complex<float>, complex<float>>(
    py::module &m, const string &name);

extern template void bind_fl_general<SGB, complex<float>>(py::module &m);

extern template void bind_fl_mps<SGB, complex<float>>(py::module &m);
extern template void bind_fl_mpo<SGB, complex<float>>(py::module &m);
extern template void bind_fl_partition<SGB, complex<float>>(py::module &m);
extern template void bind_fl_qc_hamiltonian<SGB, complex<float>>(py::module &m);
extern template void bind_fl_parallel_dmrg<SGB, complex<float>>(py::module &m);

extern template void
bind_fl_moving_environment<SGB, complex<float>, complex<float>>(
    py::module &m, const string &name);
extern template void
bind_fl_moving_environment<SGB, complex<float>, float>(py::module &m,
                                                       const string &name);
extern template void
bind_fl_dmrg<SGB, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_td_dmrg<SGB, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_linear<SGB, complex<float>, complex<float>>(py::module &m);
extern template void
bind_fl_expect<SGB, complex<float>, complex<float>, complex<float>>(
    py::module &m, const string &name);

extern template auto
bind_fl_trans_mps_spin_specific<SZ, SGF, complex<float>>(py::module &m,
                                                         const string &aux_name)
    -> decltype(typename SZ::is_sz_t(typename SGF::is_sg_t()));

extern template void
bind_fl_trans_mps<SGF, complex<float>, complex<double>>(py::module &m,
                                                        const string &aux_name);
extern template void
bind_fl_trans_mps<SGF, complex<double>, complex<float>>(py::module &m,
                                                        const string &aux_name);
extern template void
bind_fl_trans_mps<SGB, complex<float>, complex<double>>(py::module &m,
                                                        const string &aux_name);
extern template void
bind_fl_trans_mps<SGB, complex<double>, complex<float>>(py::module &m,
                                                        const string &aux_name);

#endif

#endif

#endif

#endif
