
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "../block2_big_site.hpp"
#include "../block2_core.hpp"
#include "../block2_dmrg.hpp"

namespace py = pybind11;
using namespace block2;

template <typename S, typename FL> void bind_fl_big_site(py::module &m) {

    py::class_<BigSite<S, FL>, shared_ptr<BigSite<S, FL>>>(m, "BigSite")
        .def(py::init<int>())
        .def_readwrite("n_orbs", &BigSite<S, FL>::n_orbs)
        .def_readwrite("basis", &BigSite<S, FL>::basis)
        .def_readwrite("op_infos", &BigSite<S, FL>::op_infos)
        .def("get_site_ops", &BigSite<S, FL>::get_site_ops)
        .def("get_site_op", &BigSite<S, FL>::get_site_op);

    py::class_<SimplifiedBigSite<S, FL>, shared_ptr<SimplifiedBigSite<S, FL>>,
               BigSite<S, FL>>(m, "SimplifiedBigSite")
        .def(py::init<const shared_ptr<BigSite<S, FL>> &,
                      const shared_ptr<Rule<S, FL>> &>())
        .def_readwrite("big_site", &SimplifiedBigSite<S, FL>::big_site)
        .def_readwrite("rule", &SimplifiedBigSite<S, FL>::rule);

    py::class_<ParallelBigSite<S, FL>, shared_ptr<ParallelBigSite<S, FL>>,
               BigSite<S, FL>>(m, "ParallelBigSite")
        .def(py::init<const shared_ptr<BigSite<S, FL>> &,
                      const shared_ptr<ParallelRule<S, FL>> &>())
        .def_readwrite("big_site", &ParallelBigSite<S, FL>::big_site)
        .def_readwrite("rule", &ParallelBigSite<S, FL>::rule);
}

template <typename S, typename FL>
void bind_fl_sci_big_site_fock(py::module &m) {

    py::class_<SCIFockDeterminant, shared_ptr<SCIFockDeterminant>>(
        m, "SCIFockDeterminant")
        .def(py::init<>())
        .def_readwrite("norbs", &SCIFockDeterminant::norbs)
        .def_readwrite("nAlphaEl", &SCIFockDeterminant::nAlphaEl)
        .def_readwrite("nBetaEl", &SCIFockDeterminant::nBetaEl)
        .def_readwrite("EffDetLen", &SCIFockDeterminant::EffDetLen)
        .def("getocc", &SCIFockDeterminant::getocc);

    py::bind_vector<vector<SCIFockDeterminant>>(m, "VectorSCIFockDeterminant");

    py::class_<SCIFockBigSite<S, FL>, shared_ptr<SCIFockBigSite<S, FL>>,
               BigSite<S, FL>>(m, "SCIFockBigSite")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP<FL>> &,
                      const std::vector<uint8_t> &, int, int, int, bool>(),
             py::arg("nOrb"), py::arg("nOrbThis"), py::arg("isRight"),
             py::arg("fcidump"), py::arg("orbsym"), py::arg("nMaxAlphaEl"),
             py::arg("nMaxBetaEl"), py::arg("nMaxEl"),
             py::arg("verbose") = true,
             "Initialization via generated CI space based on nMax*")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP<FL>> &,
                      const std::vector<uint8_t> &, const vector<vector<int>> &,
                      bool>(),
             py::arg("nOrb"), py::arg("nOrbThis"), py::arg("isRight"),
             py::arg("fcidump"), py::arg("orbsym"), py::arg("occs"),
             py::arg("verbose") = true,
             "Initialization via externally given determinants in `occs`")
        .def_readwrite("excludeQNs", &SCIFockBigSite<S, FL>::excludeQNs)
        .def_static("ras_space", &SCIFockBigSite<S, FL>::ras_space,
                    py::arg("is_right"), py::arg("norb"), py::arg("nalpha"),
                    py::arg("nbeta"), py::arg("nelec"),
                    py::arg("ref") = vector<int>())
        .def_readonly("quantumNumbers", &SCIFockBigSite<S, FL>::quantumNumbers)
        .def_readonly("nOrbOther", &SCIFockBigSite<S, FL>::nOrbOther)
        .def_readonly("nOrbThis", &SCIFockBigSite<S, FL>::nOrbThis)
        .def_readonly("nOrb", &SCIFockBigSite<S, FL>::nOrb)
        .def_readonly("isRight", &SCIFockBigSite<S, FL>::isRight)
        .def_readonly("nMaxAlphaEl", &SCIFockBigSite<S, FL>::nMaxAlphaEl)
        .def_readonly("nMaxBetaEl", &SCIFockBigSite<S, FL>::nMaxBetaEl)
        .def_readonly("nMaxEl", &SCIFockBigSite<S, FL>::nMaxEl)
        .def_readonly("nDet", &SCIFockBigSite<S, FL>::nDet)
        .def_readwrite("sparsityThresh", &SCIFockBigSite<S, FL>::sparsityThresh,
                       "After > #zeros/#tot the sparse matrix is activated")
        .def_readwrite("sparsityStart", &SCIFockBigSite<S, FL>::sparsityStart,
                       "After which matrix size (nCol * nRow) should sparse "
                       "matrices be activated")
        .def_readwrite("eps", &SCIFockBigSite<S, FL>::eps,
                       "Sparsity value threshold. Everything below eps will be "
                       "set to 0.0")
        .def_readwrite("fragSpace", &SCIFockBigSite<S, FL>::fragSpace)
        .def_readwrite("offsets", &SCIFockBigSite<S, FL>::offsets)
        .def_readwrite("nDet", &SCIFockBigSite<S, FL>::nDet)
        .def("setOmpThreads", &SCIFockBigSite<S, FL>::setOmpThreads)
        // vv setter
        .def_property(
            "qnIdxBra", nullptr,
            (void(SCIFockBigSite<S, FL>::*)(const std::vector<int> &)) &
                SCIFockBigSite<S, FL>::setQnIdxBra)
        .def_property(
            "qnIdxKet", nullptr,
            (void(SCIFockBigSite<S, FL>::*)(const std::vector<int> &)) &
                SCIFockBigSite<S, FL>::setQnIdxKet)
        .def("setQnIdxBra",
             (void(SCIFockBigSite<S, FL>::*)(const std::vector<int> &,
                                             const std::vector<char> &)) &
                 SCIFockBigSite<S, FL>::setQnIdxBra)
        .def("setQnIdxKet",
             (void(SCIFockBigSite<S, FL>::*)(const std::vector<int> &,
                                             const std::vector<char> &)) &
                 SCIFockBigSite<S, FL>::setQnIdxKet)
        .def_readwrite("qnIdxBraH", &SCIFockBigSite<S, FL>::qnIdxBraH)
        .def_readwrite("qnIdxKetH", &SCIFockBigSite<S, FL>::qnIdxKetH)
        .def_readwrite("qnIdxBraQ", &SCIFockBigSite<S, FL>::qnIdxBraQ)
        .def_readwrite("qnIdxKetQ", &SCIFockBigSite<S, FL>::qnIdxKetQ)
        .def_readwrite("qnIdxBraI", &SCIFockBigSite<S, FL>::qnIdxBraI)
        .def_readwrite("qnIdxKetI", &SCIFockBigSite<S, FL>::qnIdxKetI)
        .def_readwrite("qnIdxBraA", &SCIFockBigSite<S, FL>::qnIdxBraA)
        .def_readwrite("qnIdxKetA", &SCIFockBigSite<S, FL>::qnIdxKetA)
        .def_readwrite("qnIdxBraB", &SCIFockBigSite<S, FL>::qnIdxBraB)
        .def_readwrite("qnIdxKetB", &SCIFockBigSite<S, FL>::qnIdxKetB)
        .def_readwrite("qnIdxBraP", &SCIFockBigSite<S, FL>::qnIdxBraP)
        .def_readwrite("qnIdxKetP", &SCIFockBigSite<S, FL>::qnIdxKetP)
        .def_readwrite("qnIdxBraR", &SCIFockBigSite<S, FL>::qnIdxBraR)
        .def_readwrite("qnIdxKetR", &SCIFockBigSite<S, FL>::qnIdxKetR)
        .def_readwrite("qnIdxBraC", &SCIFockBigSite<S, FL>::qnIdxBraC)
        .def_readwrite("qnIdxKetC", &SCIFockBigSite<S, FL>::qnIdxKetC);
}

template <typename S, typename FL> void bind_fl_csf_big_site(py::module &m) {

    py::class_<CSFSpace<S, FL>, shared_ptr<CSFSpace<S, FL>>>(m, "CSFSpace")
        .def(py::init<int, int, bool, const std::vector<uint8_t> &>())
        .def("get_config", &CSFSpace<S, FL>::get_config)
        .def("index_config", &CSFSpace<S, FL>::index_config)
        .def("to_string", &CSFSpace<S, FL>::to_string)
        .def("cfg_apply_ops", &CSFSpace<S, FL>::cfg_apply_ops)
        .def("csf_apply_ops", &CSFSpace<S, FL>::csf_apply_ops)
        .def("__getitem__", &CSFSpace<S, FL>::operator[], py::arg("idx"))
        .def_property_readonly("n_configs", &CSFSpace<S, FL>::n_configs)
        .def_property_readonly("n_csfs", &CSFSpace<S, FL>::n_csfs)
        .def_readwrite("qs", &CSFSpace<S, FL>::qs)
        .def_readwrite("qs_idxs", &CSFSpace<S, FL>::qs_idxs)
        .def_readwrite("n_unpaired", &CSFSpace<S, FL>::n_unpaired)
        .def_readwrite("n_unpaired_idxs", &CSFSpace<S, FL>::n_unpaired_idxs)
        .def_readwrite("n_unpaired_shapes", &CSFSpace<S, FL>::n_unpaired_shapes)
        .def_readwrite("csfs", &CSFSpace<S, FL>::csfs)
        .def_readwrite("csf_idxs", &CSFSpace<S, FL>::csf_idxs)
        .def_readwrite("csf_sub_idxs", &CSFSpace<S, FL>::csf_sub_idxs)
        .def_readwrite("csf_offsets", &CSFSpace<S, FL>::csf_offsets)
        .def_readwrite("combinatorics", &CSFSpace<S, FL>::combinatorics)
        .def_readwrite("basis", &CSFSpace<S, FL>::basis)
        .def_readwrite("cg", &CSFSpace<S, FL>::cg)
        .def_readwrite("n_orbs", &CSFSpace<S, FL>::n_orbs)
        .def_readwrite("n_max_elec", &CSFSpace<S, FL>::n_max_elec)
        .def_readwrite("n_max_unpaired", &CSFSpace<S, FL>::n_max_unpaired)
        .def_readwrite("is_right", &CSFSpace<S, FL>::is_right);

    py::class_<CSFBigSite<S, FL>, shared_ptr<CSFBigSite<S, FL>>,
               BigSite<S, FL>>(m, "CSFBigSite")
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP<FL>> &,
                      const std::vector<uint8_t> &>())
        .def(py::init<int, int, bool, const shared_ptr<FCIDUMP<FL>> &,
                      const std::vector<uint8_t> &, int>())
        .def(py::init<shared_ptr<CSFSpace<S, FL>>,
                      const shared_ptr<FCIDUMP<FL>> &,
                      const vector<uint8_t> &>())
        .def("fill_csr_matrix", &CSFBigSite<S, FL>::fill_csr_matrix)
        .def("build_site_op", &CSFBigSite<S, FL>::build_site_op)
        .def_readwrite("fcidump", &CSFBigSite<S, FL>::fcidump)
        .def_readwrite("csf_space", &CSFBigSite<S, FL>::csf_space)
        .def_readwrite("is_right", &CSFBigSite<S, FL>::is_right);
}

template <typename S> void bind_drt_big_site(py::module &m) {

    py::class_<DRT<S>, shared_ptr<DRT<S>>>(m, "DRT")
        .def_readwrite("abc", &DRT<S>::abc)
        .def_readwrite("pgs", &DRT<S>::pgs)
        .def_readwrite("orb_sym", &DRT<S>::orb_sym)
        .def_readwrite("jds", &DRT<S>::jds)
        .def_readwrite("xs", &DRT<S>::xs)
        .def_readwrite("n_sites", &DRT<S>::n_sites)
        .def_readwrite("n_init_qs", &DRT<S>::n_init_qs)
        .def_readwrite("n_core", &DRT<S>::n_core)
        .def_readwrite("n_virt", &DRT<S>::n_virt)
        .def_readwrite("n_ex", &DRT<S>::n_ex)
        .def_readwrite("single_ref", &DRT<S>::single_ref)
        .def(py::init<>())
        .def(py::init<int16_t, int16_t, int16_t>(), py::arg("a"), py::arg("b"),
             py::arg("c"))
        .def(py::init<int16_t, int16_t, int16_t, typename S::pg_t>(),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("ipg"))
        .def(py::init<int16_t, int16_t, int16_t, typename S::pg_t,
                      const vector<typename S::pg_t> &>(),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("ipg"),
             py::arg("orb_sym"))
        .def(py::init<int16_t, int16_t, int16_t, typename S::pg_t,
                      const vector<typename S::pg_t> &, int, int, int>(),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("ipg"),
             py::arg("orb_sym"), py::arg("n_core"), py::arg("n_virt"),
             py::arg("n_ex"))
        .def(py::init<int, S>(), py::arg("n_sites"), py::arg("q"))
        .def(py::init<int, S, const vector<typename S::pg_t> &>(),
             py::arg("n_sites"), py::arg("q"), py::arg("orb_sym"))
        .def(
            py::init<int, S, const vector<typename S::pg_t> &, int, int, int>(),
            py::arg("n_sites"), py::arg("q"), py::arg("orb_sym"),
            py::arg("n_core"), py::arg("n_virt"), py::arg("n_ex"))
        .def(py::init<int, const vector<S> &>(), py::arg("n_sites"),
             py::arg("init_qs"))
        .def(py::init<int, const vector<S> &,
                      const vector<typename S::pg_t> &>(),
             py::arg("n_sites"), py::arg("init_qs"), py::arg("orb_sym"))
        .def(py::init<int, const vector<S> &, const vector<typename S::pg_t> &,
                      int, int, int>(),
             py::arg("n_sites"), py::arg("init_qs"), py::arg("orb_sym"),
             py::arg("n_core"), py::arg("n_virt"), py::arg("n_ex"))
        .def(py::init<int, const vector<S> &, const vector<typename S::pg_t> &,
                      int, int, int, int>(),
             py::arg("n_sites"), py::arg("init_qs"), py::arg("orb_sym"),
             py::arg("n_core"), py::arg("n_virt"), py::arg("n_ex"),
             py::arg("nc_ref"))
        .def(py::init<int, const vector<S> &, const vector<typename S::pg_t> &,
                      int, int, int, int, bool>(),
             py::arg("n_sites"), py::arg("init_qs"), py::arg("orb_sym"),
             py::arg("n_core"), py::arg("n_virt"), py::arg("n_ex"),
             py::arg("nc_ref"), py::arg("single_ref"))
        .def_property_readonly("n_rows", &DRT<S>::n_rows)
        .def("initialize", &DRT<S>::initialize)
        .def("get_init_qs", &DRT<S>::get_init_qs)
        .def("__getitem__", &DRT<S>::operator[], py::arg("i"))
        .def("index", &DRT<S>::index)
        .def("__len__", &DRT<S>::size)
        .def(
            "__iter__",
            [](DRT<S> *self) {
                struct Iter {
                    DRT<S> *drt;
                    typename DRT<S>::LL x;
                    Iter(DRT<S> *drt, typename DRT<S>::LL x) : drt(drt), x(x) {}
                    string operator*() const { return (*drt)[x]; }
                    Iter operator++() { return Iter(drt, ++x); }
                    bool operator==(Iter other) { return x == other.x; }
                };
                return py::make_iterator(Iter(self, 0),
                                         Iter(self, self->size()));
            },
            py::keep_alive<0, 1>())
        .def("q_index", &DRT<S>::q_index)
        .def("q_range", &DRT<S>::q_range)
        .def("__xor__", &DRT<S>::operator^)
        .def("__rshift__", &DRT<S>::operator>>)
        .def("get_basis", &DRT<S>::get_basis)
        .def("__repr__", &DRT<S>::to_str);

    py::class_<HDRT<S>, shared_ptr<HDRT<S>>>(m, "HDRT")
        .def_readwrite("qs", &HDRT<S>::qs)
        .def_readwrite("pgs", &HDRT<S>::pgs)
        .def_readwrite("orb_sym", &HDRT<S>::orb_sym)
        .def_readwrite("jds", &HDRT<S>::jds)
        .def_readwrite("xs", &HDRT<S>::xs)
        .def_readwrite("n_sites", &HDRT<S>::n_sites)
        .def_readwrite("n_init_qs", &HDRT<S>::n_init_qs)
        .def_readwrite("nd", &HDRT<S>::nd)
        .def_readwrite("d_map", &HDRT<S>::d_map)
        .def_readwrite("d_step", &HDRT<S>::d_step)
        .def_readwrite("d_expr", &HDRT<S>::d_expr)
        .def(py::init<>())
        .def(py::init<int, const vector<pair<S, pair<int16_t, int16_t>>> &>())
        .def(py::init<int, const vector<pair<S, pair<int16_t, int16_t>>> &,
                      const vector<typename S::pg_t> &>())
        .def_property_readonly("n_rows", &HDRT<S>::n_rows)
        .def("initialize_steps", &HDRT<S>::initialize_steps)
        .def("initialize", &HDRT<S>::initialize)
        .def("__getitem__", &HDRT<S>::operator[], py::arg("i"))
        .def("index", &HDRT<S>::index)
        .def("__len__", &HDRT<S>::size)
        .def("fill_data", &HDRT<S>::template fill_data<double>)
        .def("__repr__", &HDRT<S>::to_str);
}

template <typename S, typename FL> void bind_fl_drt_big_site(py::module &m) {

    py::class_<ElemMat<S, FL>, shared_ptr<ElemMat<S, FL>>>(m, "ElemMat")
        .def(py::init<int16_t, const vector<FL> &,
                      const vector<pair<int16_t, int16_t>> &>())
        .def_readwrite("data", &ElemMat<S, FL>::data)
        .def_readwrite("indices", &ElemMat<S, FL>::indices)
        .def_readwrite("dq", &ElemMat<S, FL>::dq)
        .def_static("op_matrices", &ElemMat<S, FL>::op_matrices)
        .def_static("multiply", &ElemMat<S, FL>::multiply)
        .def_static("build_matrix", &ElemMat<S, FL>::build_matrix)
        .def("expand", &ElemMat<S, FL>::expand);

    py::bind_vector<vector<ElemMat<S, FL>>>(m, "VectorElemMat");
    py::bind_vector<vector<vector<ElemMat<S, FL>>>>(m, "VectorVectorElemMat");

    py::class_<DRTMPS<S, FL>, shared_ptr<DRTMPS<S, FL>>>(m, "DRTMPS")
        .def(py::init<const shared_ptr<DRT<S>> &,
                      const vector<typename DRTMPS<S, FL>::LL> &,
                      const vector<vector<FL>> &>())
        .def_static("get_k", &DRTMPS<S, FL>::get_k)
        .def_static("get_offsets", &DRTMPS<S, FL>::get_offsets)
        .def_static("from_ci_vector",
                    [](const shared_ptr<DRT<S>> &drt, py::array_t<FL> ci) {
                        return DRTMPS<S, FL>::from_ci_vector(drt, ci.data());
                    })
        .def("to_ci_vector", &DRTMPS<S, FL>::to_ci_vector)
        .def("get_bond_dimensions", &DRTMPS<S, FL>::get_bond_dimensions)
        .def("qr", &DRTMPS<S, FL>::qr)
        .def("svd", &DRTMPS<S, FL>::svd, py::arg("max_bond_dim") = -1,
             py::arg("cutoff") = (FL)0.0)
        .def("dot", &DRTMPS<S, FL>::dot)
        .def("expect", &DRTMPS<S, FL>::expect)
        .def_readwrite("drt", &DRTMPS<S, FL>::drt)
        .def_readwrite("shapes", &DRTMPS<S, FL>::shapes)
        .def_readwrite("offsets", &DRTMPS<S, FL>::offsets)
        .def_readwrite("data", &DRTMPS<S, FL>::data);

    py::class_<HDRTMPO<S, FL>, shared_ptr<HDRTMPO<S, FL>>>(m, "HDRTMPO")
        .def(py::init<const shared_ptr<HDRT<S>> &,
                      const vector<typename HDRTMPO<S, FL>::LL> &,
                      const vector<vector<FL>> &>())
        .def_static("get_offsets", &HDRTMPO<S, FL>::get_offsets)
        .def_static("from_ci_vector",
                    [](const shared_ptr<HDRT<S>> &hdrt, py::array_t<FL> ci) {
                        return HDRTMPO<S, FL>::from_ci_vector(hdrt, ci.data());
                    })
        .def("to_ci_vector", &HDRTMPO<S, FL>::to_ci_vector)
        .def("get_bond_dimensions", &HDRTMPO<S, FL>::get_bond_dimensions)
        .def("qr", &HDRTMPO<S, FL>::qr)
        .def("svd", &HDRTMPO<S, FL>::svd, py::arg("max_bond_dim") = -1,
             py::arg("cutoff") = (FL)0.0)
        .def_readwrite("hdrt", &HDRTMPO<S, FL>::hdrt)
        .def_readwrite("shapes", &HDRTMPO<S, FL>::shapes)
        .def_readwrite("offsets", &HDRTMPO<S, FL>::offsets)
        .def_readwrite("data", &HDRTMPO<S, FL>::data);

    py::class_<HDRTScheme<S, FL>, shared_ptr<HDRTScheme<S, FL>>>(m,
                                                                 "HDRTScheme")
        .def(py::init<const shared_ptr<HDRT<S>> &,
                      const vector<shared_ptr<SpinPermScheme>> &>())
        .def("sort_integral", &HDRTScheme<S, FL>::sort_integral)
        .def("sort_npdm", &HDRTScheme<S, FL>::sort_npdm)
        .def_readwrite("hdrt", &HDRTScheme<S, FL>::hdrt)
        .def_readwrite("schemes", &HDRTScheme<S, FL>::schemes)
        .def_readwrite("expr_mp", &HDRTScheme<S, FL>::expr_mp)
        .def_readwrite("hjumps", &HDRTScheme<S, FL>::hjumps)
        .def_readwrite("ds", &HDRTScheme<S, FL>::ds)
        .def_readwrite("jis", &HDRTScheme<S, FL>::jis)
        .def_readwrite("n_patterns", &HDRTScheme<S, FL>::n_patterns);

    py::class_<DRTBigSite<S, FL>, shared_ptr<DRTBigSite<S, FL>>,
               BigSite<S, FL>>(m, "DRTBigSite")
        .def(py::init<const vector<S> &, bool, int,
                      const vector<typename S::pg_t> &>())
        .def(py::init<const vector<S> &, bool, int,
                      const vector<typename S::pg_t> &,
                      const shared_ptr<FCIDUMP<FL>> &>())
        .def(py::init<const vector<S> &, bool, int,
                      const vector<typename S::pg_t> &,
                      const shared_ptr<FCIDUMP<FL>> &, int>())
        .def_static("get_target_quanta", &DRTBigSite<S, FL>::get_target_quanta,
                    py::arg("is_right"), py::arg("n_orbs"),
                    py::arg("n_max_elec"), py::arg("orb_sym"),
                    py::arg("nc_ref") = 0)
        .def("get_site_op_infos", &DRTBigSite<S, FL>::get_site_op_infos)
        .def("prepare_factors", &DRTBigSite<S, FL>::prepare_factors)
        .def("fill_csr_matrix_from_coo",
             &DRTBigSite<S, FL>::fill_csr_matrix_from_coo)
        .def("fill_csr_matrix", &DRTBigSite<S, FL>::fill_csr_matrix)
        .def("get_site_matrices", &DRTBigSite<S, FL>::get_site_matrices)
        .def("build_npdm",
             [](DRTBigSite<S, FL> *self, const string &expr,
                py::array_t<FL> bra_ci, py::array_t<FL> ket_ci) {
                 return self->build_npdm(expr, bra_ci.data(), ket_ci.data());
             })
        .def_readwrite("n_total_orbs", &DRTBigSite<S, FL>::n_total_orbs)
        .def_readwrite("cutoff", &DRTBigSite<S, FL>::cutoff)
        .def_readwrite("fcidump", &DRTBigSite<S, FL>::fcidump)
        .def_readwrite("gfd", &DRTBigSite<S, FL>::gfd)
        .def_readwrite("drt", &DRTBigSite<S, FL>::drt)
        .def_property(
            "drt", [](DRTBigSite<S, FL> *self) { return self->drt; },
            [](DRTBigSite<S, FL> *self, shared_ptr<DRT<S>> drt) {
                self->drt = drt;
                self->basis = drt->get_basis();
                self->op_infos = self->get_site_op_infos(drt->orb_sym);
            })
        .def_readwrite("factors", &DRTBigSite<S, FL>::factors)
        .def_readwrite("factor_strides", &DRTBigSite<S, FL>::factor_strides)
        .def_readwrite("is_right", &DRTBigSite<S, FL>::is_right);
}

template <typename S, typename FL>
void bind_fl_hamiltonian_big_site(py::module &m) {

    py::class_<HamiltonianQCBigSite<S, FL>,
               shared_ptr<HamiltonianQCBigSite<S, FL>>, HamiltonianQC<S, FL>>(
        m, "HamiltonianQCBigSite")
        .def(py::init<S, int, const vector<typename S::pg_t> &,
                      const shared_ptr<FCIDUMP<FL>> &,
                      const shared_ptr<BigSite<S, FL>> &,
                      const shared_ptr<BigSite<S, FL>> &>(),
             py::arg("vacuum"), py::arg("n_orbs_total"), py::arg("orb_sym"),
             py::arg("fcidump"), py::arg("big_left") = nullptr,
             py::arg("big_right") = nullptr)
        .def_readwrite("big_left", &HamiltonianQCBigSite<S, FL>::big_left)
        .def_readwrite("big_right", &HamiltonianQCBigSite<S, FL>::big_right)
        .def_readwrite("n_orbs", &HamiltonianQCBigSite<S, FL>::n_orbs)
        .def_readwrite("n_orbs_left", &HamiltonianQCBigSite<S, FL>::n_orbs_left)
        .def_readwrite("n_orbs_right",
                       &HamiltonianQCBigSite<S, FL>::n_orbs_right)
        .def_readwrite("n_orbs_cas", &HamiltonianQCBigSite<S, FL>::n_orbs_cas)
        .def_readwrite("full_hamil", &HamiltonianQCBigSite<S, FL>::full_hamil);
}

template <typename S, typename FL, typename FLS>
void bind_fl_dmrg_big_site(py::module &m) {

    py::class_<DMRGBigSite<S, FL, FLS>, shared_ptr<DMRGBigSite<S, FL, FLS>>,
               DMRG<S, FL, FLS>>(m, "DMRGBigSite")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRG<S, FL, FLS>::FPS> &>())
        .def_readwrite("last_site_svd", &DMRGBigSite<S, FL, FLS>::last_site_svd)
        .def_readwrite("last_site_1site",
                       &DMRGBigSite<S, FL, FLS>::last_site_1site)
        .def("blocking", &DMRGBigSite<S, FL, FLS>::blocking);

    py::class_<LinearBigSite<S, FL, FLS>, shared_ptr<LinearBigSite<S, FL, FLS>>,
               Linear<S, FL, FLS>>(m, "LinearBigSite")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<typename Linear<S, FL, FLS>::FPS> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<typename Linear<S, FL, FLS>::FPS> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &,
                      const vector<typename Linear<S, FL, FLS>::FPS> &>())
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &, const vector<ubond_t> &>())
        .def_readwrite("last_site_svd",
                       &LinearBigSite<S, FL, FLS>::last_site_svd)
        .def_readwrite("last_site_1site",
                       &LinearBigSite<S, FL, FLS>::last_site_1site)
        .def("blocking", &LinearBigSite<S, FL, FLS>::blocking);

    py::class_<DMRGBigSiteAQCCOLD<S, FL, FLS>,
               shared_ptr<DMRGBigSiteAQCCOLD<S, FL, FLS>>,
               DMRGBigSite<S, FL, FLS>>(m, "DMRGBigSiteAQCCOLD")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRGBigSite<S, FL, FLS>::FPS> &, FL,
                      typename DMRGBigSite<S, FL, FLS>::FPS,
                      const std::vector<S> &>())
        .def_readwrite("max_aqcc_iter",
                       &DMRGBigSiteAQCCOLD<S, FL, FLS>::max_aqcc_iter)
        .def_readwrite("g_factor", &DMRGBigSiteAQCCOLD<S, FL, FLS>::g_factor)
        .def_readwrite("delta_e", &DMRGBigSiteAQCCOLD<S, FL, FLS>::delta_e)
        .def_readwrite("ref_energy",
                       &DMRGBigSiteAQCCOLD<S, FL, FLS>::ref_energy);

    py::class_<DMRGBigSiteAQCC<S, FL, FLS>,
               shared_ptr<DMRGBigSiteAQCC<S, FL, FLS>>,
               DMRGBigSite<S, FL, FLS>>(m, "DMRGBigSiteAQCC")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &, FL,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRGBigSite<S, FL, FLS>::FPS> &,
                      typename DMRGBigSite<S, FL, FLS>::FPS>(),
             "Frozen/CAS mode: Only one big site at the end")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &, FL,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &, FL,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRGBigSite<S, FL, FLS>::FPS> &,
                      typename DMRGBigSite<S, FL, FLS>::FPS>(),
             "Frozen/CAS mode ACPF2: Only one big site at the end")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &, FL,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRGBigSite<S, FL, FLS>::FPS> &,
                      typename DMRGBigSite<S, FL, FLS>::FPS>(),
             "RAS mode: Big sites on both ends")
        .def(py::init<const shared_ptr<MovingEnvironment<S, FL, FLS>> &, FL,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &, FL,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const shared_ptr<MovingEnvironment<S, FL, FLS>> &,
                      const vector<ubond_t> &,
                      const vector<typename DMRGBigSite<S, FL, FLS>::FPS> &,
                      typename DMRGBigSite<S, FL, FLS>::FPS>(),
             "RAS ACPF2 mode: Big sites on both ends")
        .def_readwrite("smallest_energy",
                       &DMRGBigSiteAQCC<S, FL, FLS>::smallest_energy)
        .def_readwrite("max_aqcc_iter",
                       &DMRGBigSiteAQCC<S, FL, FLS>::max_aqcc_iter)
        .def_readwrite("g_factor", &DMRGBigSiteAQCC<S, FL, FLS>::g_factor)
        .def_readwrite("g_factor2", &DMRGBigSiteAQCC<S, FL, FLS>::g_factor2)
        .def_readwrite("ACPF2_mode", &DMRGBigSiteAQCC<S, FL, FLS>::ACPF2_mode)
        .def_readwrite("RAS_mode", &DMRGBigSiteAQCC<S, FL, FLS>::RAS_mode)
        .def_readwrite("delta_e", &DMRGBigSiteAQCC<S, FL, FLS>::delta_e)
        .def_readwrite("ref_energy", &DMRGBigSiteAQCC<S, FL, FLS>::ref_energy);
}

#ifdef _EXPLICIT_TEMPLATE

#ifdef _USE_SU2SZ
extern template void bind_fl_big_site<SZ, double>(py::module &m);
extern template void bind_fl_hamiltonian_big_site<SZ, double>(py::module &m);
extern template void bind_fl_dmrg_big_site<SZ, double, double>(py::module &m);

extern template void bind_fl_big_site<SU2, double>(py::module &m);
extern template void bind_fl_hamiltonian_big_site<SU2, double>(py::module &m);
extern template void bind_fl_dmrg_big_site<SU2, double, double>(py::module &m);

extern template void bind_fl_sci_big_site_fock<SZ, double>(py::module &m);
extern template void bind_fl_csf_big_site<SU2, double>(py::module &m);

extern template void bind_drt_big_site<SZ>(py::module &m);
extern template void bind_drt_big_site<SU2>(py::module &m);

extern template void bind_fl_drt_big_site<SZ, double>(py::module &m);
extern template void bind_fl_drt_big_site<SU2, double>(py::module &m);
#endif

#endif
