
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2024 Huanchen Zhai <hczhai@caltech.edu>
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

#include "../pybind_dmrg.hpp"

template void bind_trans_mps<SAny, SAny>(py::module &m, const string &aux_name);
template void bind_trans_multi_mps<SAny, SAny>(py::module &m,
                                               const string &aux_name);
template auto
bind_fl_trans_mps_spin_specific<SAny, SAny, double>(py::module &m,
                                                    const string &aux_name)
    -> decltype(typename SAny::is_sany_t(typename SAny::is_sany_t()));
template auto bind_fl_trans_mpo<SAny, SAny, double>(py::module &m,
                                                    const string &aux_name)
    -> decltype(typename SAny::is_sany_t(typename SAny::is_sany_t()));
