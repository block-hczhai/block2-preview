
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
    using DMRG<S>::iprint;
    using DMRG<S>::me;
    using DMRG<S>::davidson_soft_max_iter;
    using DMRG<S>::noise_type;
    using DMRG<S>::decomp_type;
    using typename DMRG<S>::Iteration;
    bool last_site_svd = false;
    bool last_site_1site = false;
    DMRGSCI(const shared_ptr<MovingEnvironment<S>> &me,
            const vector<ubond_t> &bond_dims, const vector<double> &noises)
        : DMRG<S>(me, bond_dims, noises) {}
    Iteration blocking(int i, bool forward, ubond_t bond_dim, double noise,
                       double davidson_conv_thrd) override {
        int dsmi = davidson_soft_max_iter;
        NoiseTypes nt = noise_type;
        DecompositionTypes dt = decomp_type;
        if (last_site_1site && forward && i == me->n_sites - 2) {
            assert(me->dot = 2);
            me->dot = 1;
            me->ket->canonical_form[i] = 'K';
            davidson_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            DMRG<S>::blocking(i, forward, bond_dim, 0, davidson_conv_thrd);
            davidson_soft_max_iter = dsmi;
            i++;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        } else if (last_site_1site && !forward && i == me->n_sites - 2) {
            me->dot = 1;
            i = me->n_sites - 1;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        }
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            davidson_soft_max_iter = 0;
            if (noise_type == NoiseTypes::DensityMatrix)
                noise_type = NoiseTypes::Wavefunction;
            decomp_type = DecompositionTypes::SVD;
        }
        Iteration r =
            DMRG<S>::blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            r.energies[0] = 0;
            davidson_soft_max_iter = dsmi;
            noise_type = nt;
            decomp_type = dt;
        }
        if (last_site_1site && forward && i == me->n_sites - 1) {
            me->dot = 2;
            me->center = me->n_sites - 2;
        } else if (last_site_1site && !forward && i == me->n_sites - 1) {
            assert(me->dot = 1);
            davidson_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            DMRG<S>::blocking(i - 1, forward, bond_dim, 0, davidson_conv_thrd);
            davidson_soft_max_iter = dsmi;
            me->envs[i - 1]->right_op_infos.clear();
            me->envs[i - 1]->right = nullptr;
            me->dot = 2;
            me->ket->canonical_form[i - 2] = 'C';
        }
        return r;
    }
};

} // namespace block2
