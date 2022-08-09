
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

#include "../pybind_core.hpp"

template void bind_fl_sparse<SGF, complex<float>>(py::module &m);
template void bind_fl_sparse<SGB, complex<float>>(py::module &m);

template void bind_trans_sparse_matrix<SGF, complex<float>, complex<double>>(
    py::module &m, const string &aux_name);
template void bind_trans_sparse_matrix<SGF, complex<double>, complex<float>>(
    py::module &m, const string &aux_name);
template void bind_trans_sparse_matrix<SGB, complex<float>, complex<double>>(
    py::module &m, const string &aux_name);
template void bind_trans_sparse_matrix<SGB, complex<double>, complex<float>>(
    py::module &m, const string &aux_name);
