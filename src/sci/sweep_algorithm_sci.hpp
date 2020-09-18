
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

#pragma once

#include "../block2/sweep_algorithm.hpp"

using namespace std;

namespace block2 {

// Density Matrix Renormalization Group for SCI
template <typename S> struct DMRGSCI : DMRG<S> {
    using DMRG<S>::me;
    using DMRG<S>::davidson_soft_max_iter;
    using DMRG<S>::noise_type;
    using DMRG<S>::decomp_type;
    using typename DMRG<S>::Iteration;
    DMRGSCI(const shared_ptr<MovingEnvironment<S>> &me,
            const vector<ubond_t> &bond_dims, const vector<double> &noises)
        : DMRG<S>(me, bond_dims, noises) {}
    Iteration blocking(int i, bool forward, ubond_t bond_dim, double noise,
                       double davidson_conv_thrd) override {
        int dsmi = davidson_soft_max_iter;
        NoiseTypes nt = noise_type;
        DecompositionTypes dt = decomp_type;
        if (me->dot == 1 && !forward && i == me->n_sites - 1) {
            davidson_soft_max_iter = 0;
            noise_type = NoiseTypes::Wavefunction;
            decomp_type = DecompositionTypes::SVD;
        }
        Iteration r =
            DMRG<S>::blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
        if (me->dot == 1 && !forward && i == me->n_sites - 1) {
            r.energies[0] = 0;
            davidson_soft_max_iter = dsmi;
            noise_type = nt;
            decomp_type = dt;
        }
        return r;
    }
};

} // namespace block2
