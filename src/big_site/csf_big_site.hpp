
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

#include "../core/allocator.hpp"
#include "../core/cg.hpp"
#include "../core/integral.hpp"
#include "../core/prime.hpp"
#include "../core/state_info.hpp"
#include "../core/threading.hpp"
#include "big_site.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename = void> struct CSFSpace;

template <typename S> struct CSFSpace<S, typename S::is_su2_t> {
    typedef long long LL;
    vector<S> qs; //!< Quantum numbers.
    vector<int>
        qs_idxs; //!< Starting index in n_unpaired for each quantum number.
    vector<int> n_unpaired;     //!< Number of unpaired electrons.
    vector<LL> n_unpaired_idxs; //!< Starting index in list of [012] configs for
                                //!< each element in n_unpaired.
    vector<pair<LL, LL>>
        n_unpaired_shapes; //!< Number of possible 2 positions and 1 positions
                           //!< for each element in n_unpaired.
    vector<uint8_t> csfs;  //!< [+-] patterns.
    vector<LL> csf_idxs; //!< Starting index in csf_sub_idxs for each number of
                         //!< unpaired electrons. Index of this vector is
                         //!< directly the number of unpaired electrons.
    vector<LL>
        csf_sub_idxs;       //!< Starting index in csfs for each number of plus.
    vector<LL> csf_offsets; //!< Index of CSF in the space for each element in
                            //!< n_unpaired.

    shared_ptr<Combinatorics> combinatorics;
    shared_ptr<StateInfo<S>> basis;
    shared_ptr<CG<S>> cg;
    int n_orbs;
    int n_max_elec;
    int n_max_unpaired;
    bool is_right;
    CSFSpace(int n_orbs, int n_max_elec, bool is_right,
             const vector<uint8_t> &orb_sym = vector<uint8_t>())
        : n_orbs(n_orbs), is_right(is_right), n_max_elec(n_max_elec) {
        assert((int)orb_sym.size() == n_orbs || orb_sym.size() == 0);
        if (n_orbs == 0)
            return;
        combinatorics = make_shared<Combinatorics>(n_orbs);
        vector<shared_ptr<StateInfo<S>>> site_basis(n_orbs);
        S vacuum, target(S::invalid);
        for (int m = 0; m < n_orbs; m++) {
            shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
            b->allocate(3);
            b->quanta[0] = vacuum;
            b->quanta[1] = S(1, 1, orb_sym.size() == 0 ? 0 : orb_sym[m]);
            b->quanta[2] = S(2, 0, 0);
            b->n_states[0] = b->n_states[1] = b->n_states[2] = 1;
            b->sort_states();
            site_basis[m] = b;
        }
        shared_ptr<StateInfo<S>> x = make_shared<StateInfo<S>>(vacuum);
        if (!is_right) {
            for (int i = 0; i < n_orbs; i++)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*x, *site_basis[i], target));
            int max_n = 0;
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > max_n)
                    max_n = x->quanta[q].n();
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() < max_n - n_max_elec ||
                    x->quanta[q].twos() > n_max_elec)
                    x->n_states[q] = 0;
            x->collect();
        } else {
            for (int i = n_orbs - 1; i >= 0; i--)
                x = make_shared<StateInfo<S>>(
                    StateInfo<S>::tensor_product(*site_basis[i], *x, target));
            for (int q = 0; q < x->n; q++)
                if (x->quanta[q].n() > n_max_elec)
                    x->n_states[q] = 0;
            x->collect();
        }
        basis = x;
        qs.resize(basis->n);
        qs_idxs.resize(basis->n + 1, 0);
        for (int i = 0; i < basis->n; i++) {
            qs[i] = basis->quanta[i];
            qs_idxs[i + 1] =
                qs_idxs[i] +
                ((min(qs[i].n(), n_max_elec) - qs[i].twos()) >> 1) + 1;
        }
        n_unpaired.resize(qs_idxs.back());
        n_unpaired_idxs.resize(qs_idxs.back() + 1, 0);
        n_unpaired_shapes.resize(qs_idxs.back());
        n_max_unpaired = 0;
        for (int i = 0; i < basis->n; i++) {
            int ij = qs_idxs[i];
            int n_elec = qs[i].n(), two_s = qs[i].twos();
            for (int j = two_s; j <= min(qs[i].n(), n_max_elec); j += 2, ij++) {
                n_max_unpaired = max(n_max_unpaired, j);
                n_unpaired[ij] = j;
                n_unpaired_shapes[ij] = make_pair(
                    combinatorics->combination(n_orbs, (n_elec - j) >> 1),
                    combinatorics->combination(n_orbs - ((n_elec - j) >> 1),
                                               j));
                n_unpaired_idxs[ij + 1] =
                    n_unpaired_idxs[ij] +
                    n_unpaired_shapes[ij].first * n_unpaired_shapes[ij].second;
            }
        }
        csf_idxs.resize(n_max_unpaired + 2);
        csf_idxs[0] = 0;
        for (int i = 0; i <= n_max_unpaired; i++)
            csf_idxs[i + 1] = csf_idxs[i] + i + 1;
        csf_sub_idxs.resize(csf_idxs.back() + 1);
        csf_sub_idxs[0] = 0;
        csf_sub_idxs[1] = 1;
        for (int i = 1, cl = 0, pcl; i <= n_max_unpaired; i++) {
            pcl = cl;
            cl += ((i & 7) == 1);
            const int cidx = csf_idxs[i], pidx = csf_idxs[i - 1];
            // j is twos
            for (int j = 0; j <= i; j++) {
                csf_sub_idxs[cidx + j + 1] = csf_sub_idxs[cidx + j];
                // j += 1
                if (j > 0)
                    csf_sub_idxs[cidx + j + 1] +=
                        csf_sub_idxs[pidx + j] - csf_sub_idxs[pidx + j - 1];
                // j -= 1
                if (j < i - 1)
                    csf_sub_idxs[cidx + j + 1] +=
                        csf_sub_idxs[pidx + j + 2] - csf_sub_idxs[pidx + j + 1];
                if (cl != pcl && i != 1)
                    csf_sub_idxs[cidx + j + 1] =
                        csf_sub_idxs[cidx + j] +
                        (csf_sub_idxs[cidx + j + 1] - csf_sub_idxs[cidx + j]) /
                            pcl * cl;
            }
        }
        csfs.resize(csf_sub_idxs.back());
        csfs[0] = 0;
        for (int i = 1, cl = 0, pcl; i <= n_max_unpaired; i++) {
            pcl = cl;
            cl += ((i & 7) == 1);
            int cidx = csf_idxs[i], pidx = csf_idxs[i - 1];
            for (int j = 0; j <= i; j++) {
                LL shift = 0, xshift;
                if (j > 0) {
                    shift = csf_sub_idxs[pidx + j] - csf_sub_idxs[pidx + j - 1];
                    if (pcl == cl || pcl == 0)
                        memcpy(csfs.data() + csf_sub_idxs[cidx + j],
                               csfs.data() + csf_sub_idxs[pidx + j - 1],
                               shift * sizeof(uint8_t));
                    else {
                        for (int k = 0, kp = 0; k < shift / pcl; k++, kp += pcl)
                            memcpy(
                                csfs.data() + csf_sub_idxs[cidx + j] + kp + k,
                                csfs.data() + csf_sub_idxs[pidx + j - 1] + kp,
                                pcl * sizeof(uint8_t));
                        shift = shift / pcl * cl;
                    }
                    const LL xh = csf_sub_idxs[cidx + j] + cl - 1;
                    const uint8_t high = (uint8_t)(1) << ((i - 1) & 7);
                    for (int k = 0; k < shift; k += cl)
                        csfs[xh + k] |= high;
                }
                if (j < i - 1) {
                    xshift =
                        csf_sub_idxs[pidx + j + 2] - csf_sub_idxs[pidx + j + 1];
                    if (pcl == cl || pcl == 0)
                        memcpy(csfs.data() + csf_sub_idxs[cidx + j] + shift,
                               csfs.data() + csf_sub_idxs[pidx + j + 1],
                               xshift * sizeof(uint8_t));
                    else {
                        for (int k = 0, kp = 0; k < xshift / pcl;
                             k++, kp += pcl)
                            memcpy(csfs.data() + csf_sub_idxs[cidx + j] +
                                       shift + kp + k,
                                   csfs.data() + csf_sub_idxs[pidx + j + 1] +
                                       kp,
                                   pcl * sizeof(uint8_t));
                        xshift = xshift / pcl * cl;
                    }
                }
            }
        }
        csf_offsets.resize(qs_idxs.back() + 1, 0);
        for (int i = 0; i < basis->n; i++) {
            int ij = qs_idxs[i];
            int n_elec = qs[i].n(), two_s = qs[i].twos();
            for (int j = two_s; j <= min(qs[i].n(), n_max_elec); j += 2, ij++) {
                int cl = max((j >> 3) + !!(j & 7), 1);
                csf_offsets[ij + 1] =
                    csf_offsets[ij] +
                    (n_unpaired_idxs[ij + 1] - n_unpaired_idxs[ij]) *
                        (csf_sub_idxs[csf_idxs[j] + two_s + 1] -
                         csf_sub_idxs[csf_idxs[j] + two_s]) /
                        cl;
            }
            // cout << i << " " << qs[i] << " " << basis->n_states[i] << " "
            //      << csf_offsets[qs_idxs[i + 1]] << " "
            //      << csf_offsets[qs_idxs[i]] << endl;
            assert(basis->n_states[i] ==
                   csf_offsets[qs_idxs[i + 1]] - csf_offsets[qs_idxs[i]]);
        }
        cg = make_shared<CG<S>>((n_max_unpaired + 1) * 2);
        cg->initialize();
    }
    // idx in n_unpaired_idxs to config
    vector<uint8_t> get_config(LL idx) const {
        const int cl = (n_orbs >> 2) + !!(n_orbs & 3);
        vector<uint8_t> r(cl, 0);
        int i_unpaired = (int)(upper_bound(n_unpaired_idxs.begin(),
                                           n_unpaired_idxs.end(), idx) -
                               n_unpaired_idxs.begin() - 1);
        assert(i_unpaired >= 0);
        int i_qs =
            (int)(upper_bound(qs_idxs.begin(), qs_idxs.end(), i_unpaired) -
                  qs_idxs.begin() - 1);
        assert(i_qs >= 0);
        const int n_unpaired = this->n_unpaired[i_unpaired];
        const int n_double = (qs[i_qs].n() - n_unpaired) >> 1;
        const int n_empty = n_orbs - n_double - n_unpaired;
        const int twos = qs[i_qs].twos();
        const LL i_csf = csf_sub_idxs[csf_idxs[n_unpaired] + twos];
        const LL icfg = idx - n_unpaired_idxs[i_unpaired];
        const LL icfgd = icfg / n_unpaired_shapes[i_unpaired].second;
        const LL icfgs = icfg % n_unpaired_shapes[i_unpaired].second;
        LL cur = 0;
        for (int i = 0, l = 0; i < n_double; i++) {
            for (; l < n_orbs; l++) {
                int nn = combinatorics->combination(n_orbs - l - 1,
                                                    n_double - i - 1);
                if (cur + nn > icfgd) {
                    r[l >> 2] |= (3 << ((l & 3) << 1));
                    l++;
                    break;
                } else
                    cur += nn;
            }
        }
        cur = 0;
        for (int i = 0, l = 0, k = n_double; i < n_unpaired; i++) {
            for (; l < n_orbs; l++) {
                if ((r[l >> 2] >> ((l & 3) << 1)) & 2) {
                    k--;
                    continue;
                }
                int nn = combinatorics->combination(n_orbs - k - l - 1,
                                                    n_unpaired - i - 1);
                if (cur + nn > icfgs) {
                    r[l >> 2] |=
                        (1 << (((l & 3) << 1) +
                               ((csfs[i_csf + (i >> 3)] >> (i & 7)) & 1)));
                    l++;
                    break;
                } else
                    cur += nn;
            }
        }
        return r;
    }
    LL index_config(const vector<uint8_t> &cfg) const {
        int n_double = 0, n_unpaired = 0, twos = 0, pg = 0;
        for (int i = 0; i < n_orbs; i++) {
            switch ((cfg[i >> 2] >> ((i & 3) << 1)) & 3) {
            case 3:
                n_double++;
                break;
            case 2:
                twos++;
                n_unpaired++;
                // pg ^= orb_sym[i];
                break;
            case 1:
                twos--;
                n_unpaired++;
                // pg ^= orb_sym[i];
                break;
            default:
                break;
            }
        }
        int i_qs = basis->find_state(S(n_double * 2 + n_unpaired, twos, pg));
        if (i_qs == -1)
            return -1;
        int i_unpaired = (int)(lower_bound(&this->n_unpaired[qs_idxs[i_qs]],
                                           &this->n_unpaired[qs_idxs[i_qs + 1]],
                                           n_unpaired) -
                               &this->n_unpaired[0]);
        if (!(i_unpaired >= qs_idxs[i_qs] && i_unpaired < qs_idxs[i_qs + 1]) ||
            this->n_unpaired[i_unpaired] != n_unpaired)
            return -1;
        LL cur = 0;
        for (int i = 0, l = -1; i < n_double; i++) {
            for (l++; l < n_orbs && ((cfg[l >> 2] >> ((l & 3) << 1)) & 3) != 3;
                 l++)
                cur += combinatorics->combination(n_orbs - l - 1,
                                                  n_double - i - 1);
        }
        cur *= n_unpaired_shapes[i_unpaired].second;
        for (int i = 0, l = -1, p = n_double; i < n_unpaired; i++)
            for (l++; l < n_orbs; l++) {
                uint8_t x = ((cfg[l >> 2] >> ((l & 3) << 1)) & 3);
                if (x == 3) {
                    p--;
                    continue;
                } else if (x == 0)
                    cur += combinatorics->combination(n_orbs - p - l - 1,
                                                      n_unpaired - i - 1);
                else
                    break;
            }
        return n_unpaired_idxs[i_unpaired] + cur;
    }
    string to_string(const vector<uint8_t> &cfg) const {
        const char d[4] = {'0', '-', '+', '2'};
        string r(n_orbs, '0');
        for (int i = 0; i < n_orbs; i++)
            r[i] = d[(cfg[i >> 2] >> ((i & 3) << 1)) & 3];
        return r;
    }
    void set_config_twos(vector<uint8_t> &cfg, int n_unpaired, int twos) const {
        const int n_plus = (n_unpaired + twos) >> 1;
        for (int i = 0, j = 0; i < n_orbs; i++) {
            const uint8_t x = (cfg[i >> 2] >> ((i & 3) << 1)) & 3;
            if (x == 2 || x == 1) {
                if ((x == 2 && j >= n_plus) || (x == 1 && j < n_plus))
                    cfg[i >> 2] ^= (3 << ((i & 3) << 1));
                j++;
            }
        }
    }
    // 0 = D, 1 = C, 0 = -, 2 = +
    template <int8_t L>
    void
    cfg_op_matrix_element(LL ibra, LL iket, uint8_t ops,
                          vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                          vector<array<uint16_t, L>> &orb_idxs,
                          vector<size_t> &data_idxs) const {
        if (data_idxs.size() == 0)
            data_idxs.push_back(0);
        vector<uint8_t> bra = get_config(ibra), ket = get_config(iket);
        array<uint16_t, L> chg_idx;
        int8_t ll = 0;
        double factor = 0;
        for (uint16_t l = 0; l < n_orbs; l++) {
            const uint8_t xbra = (bra[l >> 2] >> ((l & 3) << 1)) & 3;
            const uint8_t xket = (ket[l >> 2] >> ((l & 3) << 1)) & 3;
            if (xbra != xket &&
                !((xbra == 2 || xbra == 1) && (xket == 2 || xket == 1))) {
                if (ll == L)
                    return;
                chg_idx[ll++] = l;
            }
        }
        cfg_csf_apply_ops_l(ibra, iket, ops, ll, chg_idx, mat, orb_idxs,
                            data_idxs);
    }
    void cfg_csf_apply_ops_l(LL ibra, LL iket, uint8_t ops, int8_t ll,
                             array<uint16_t, 1> &chg_idx,
                             vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                             vector<array<uint16_t, 1>> &orb_idxs,
                             vector<size_t> &data_idxs) const {
        if (ll != 0)
            cfg_csf_apply_ops<1>(ibra, iket, ops, chg_idx, mat, orb_idxs,
                                 data_idxs);
    }
    void cfg_csf_apply_ops_l(LL ibra, LL iket, uint8_t ops, int8_t ll,
                             array<uint16_t, 2> &chg_idx,
                             vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                             vector<array<uint16_t, 2>> &orb_idxs,
                             vector<size_t> &data_idxs) const {
        if (ll == 0) {
            if ((ops & 1) != ((ops >> 2) & 1))
                for (uint16_t l = 0; l < n_orbs; l++)
                    cfg_csf_apply_ops<2>(ibra, iket, ops, {l, l}, mat, orb_idxs,
                                         data_idxs);
            return;
        }
        if (ll == 1)
            chg_idx[1] = chg_idx[0];
        cfg_csf_apply_ops<2>(ibra, iket, ops, chg_idx, mat, orb_idxs,
                             data_idxs);
    }
    void cfg_csf_apply_ops_l(LL ibra, LL iket, uint8_t ops, int8_t ll,
                             array<uint16_t, 3> &chg_idx,
                             vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                             vector<array<uint16_t, 3>> &orb_idxs,
                             vector<size_t> &data_idxs) const {
        if ((ops & 15) == 0) {
            if (ll == 1)
                cfg_csf_apply_ops<3, 1>(ibra, iket, ops >> 4, chg_idx, mat,
                                        orb_idxs, data_idxs);
            return;
        }
        if (ll == 0)
            return;
        if (ll == 1) {
            for (uint16_t l = 0; l < chg_idx[0]; l++)
                cfg_csf_apply_ops<3>(ibra, iket, ops, {l, l, chg_idx[0]}, mat,
                                     orb_idxs, data_idxs);
            for (uint16_t l = chg_idx[0]; l < n_orbs; l++)
                cfg_csf_apply_ops<3>(ibra, iket, ops, {chg_idx[0], l, l}, mat,
                                     orb_idxs, data_idxs);
        } else if (ll == 2) {
            cfg_csf_apply_ops<3>(ibra, iket, ops,
                                 {chg_idx[0], chg_idx[0], chg_idx[1]}, mat,
                                 orb_idxs, data_idxs);
            cfg_csf_apply_ops<3>(ibra, iket, ops,
                                 {chg_idx[0], chg_idx[1], chg_idx[1]}, mat,
                                 orb_idxs, data_idxs);
        } else if (ll == 3)
            cfg_csf_apply_ops<3>(ibra, iket, ops, chg_idx, mat, orb_idxs,
                                 data_idxs);
    }
    void cfg_csf_apply_ops_l(LL ibra, LL iket, uint8_t ops, int8_t ll,
                             array<uint16_t, 4> &chg_idx,
                             vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                             vector<array<uint16_t, 4>> &orb_idxs,
                             vector<size_t> &data_idxs) const {
        if ((ops & 15) == 0) {
            if (ll == 0) {
                for (uint16_t l = 0; l < n_orbs; l++) {
                    chg_idx[1] = chg_idx[0] = l;
                    cfg_csf_apply_ops<4, 2>(ibra, iket, ops >> 4, chg_idx, mat,
                                            orb_idxs, data_idxs);
                }
            } else {
                if (ll == 1)
                    chg_idx[1] = chg_idx[0];
                cfg_csf_apply_ops<4, 2>(ibra, iket, ops >> 4, chg_idx, mat,
                                        orb_idxs, data_idxs);
            }
            return;
        }
        if (ll == 0) {
            for (uint16_t l = 0; l < n_orbs; l++) {
                cfg_csf_apply_ops<4>(ibra, iket, ops, {l, l, l, l}, mat,
                                     orb_idxs, data_idxs);
                for (uint16_t k = l + 1; k < n_orbs; k++)
                    cfg_csf_apply_ops<4>(ibra, iket, ops, {l, l, k, k}, mat,
                                         orb_idxs, data_idxs);
            }
        } else if (ll == 1) {
            for (uint16_t l = 0; l < chg_idx[0]; l++)
                cfg_csf_apply_ops<4>(ibra, iket, ops,
                                     {l, l, chg_idx[0], chg_idx[0]}, mat,
                                     orb_idxs, data_idxs);
            for (uint16_t l = chg_idx[0]; l < n_orbs; l++)
                cfg_csf_apply_ops<4>(ibra, iket, ops,
                                     {chg_idx[0], chg_idx[0], l, l}, mat,
                                     orb_idxs, data_idxs);
        } else if (ll == 2) {
            for (uint16_t l = 0; l < chg_idx[0]; l++)
                cfg_csf_apply_ops<4>(ibra, iket, ops,
                                     {l, l, chg_idx[0], chg_idx[1]}, mat,
                                     orb_idxs, data_idxs);
            for (uint16_t l = chg_idx[0]; l < chg_idx[1]; l++)
                cfg_csf_apply_ops<4>(ibra, iket, ops,
                                     {chg_idx[0], l, l, chg_idx[1]}, mat,
                                     orb_idxs, data_idxs);
            for (uint16_t l = chg_idx[1]; l < n_orbs; l++)
                cfg_csf_apply_ops<4>(ibra, iket, ops,
                                     {chg_idx[0], chg_idx[1], l, l}, mat,
                                     orb_idxs, data_idxs);
            cfg_csf_apply_ops<4>(
                ibra, iket, ops,
                {chg_idx[0], chg_idx[0], chg_idx[1], chg_idx[1]}, mat, orb_idxs,
                data_idxs);
        } else if (ll == 3) {
            cfg_csf_apply_ops<4>(
                ibra, iket, ops,
                {chg_idx[0], chg_idx[0], chg_idx[1], chg_idx[2]}, mat, orb_idxs,
                data_idxs);
            cfg_csf_apply_ops<4>(
                ibra, iket, ops,
                {chg_idx[0], chg_idx[1], chg_idx[1], chg_idx[2]}, mat, orb_idxs,
                data_idxs);
            cfg_csf_apply_ops<4>(
                ibra, iket, ops,
                {chg_idx[0], chg_idx[1], chg_idx[2], chg_idx[2]}, mat, orb_idxs,
                data_idxs);
        } else if (ll == 4)
            cfg_csf_apply_ops<4>(ibra, iket, ops, chg_idx, mat, orb_idxs,
                                 data_idxs);
    }
    template <int8_t L, int8_t M = L>
    void cfg_csf_apply_ops(LL ibra, LL iket, uint8_t ops,
                           const array<uint16_t, L> &orbs,
                           vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                           vector<array<uint16_t, L>> &orb_idxs,
                           vector<size_t> &data_idxs) const {
        cfg_apply_ops(iket, ops,
                      vector<uint16_t>(orbs.begin(), orbs.begin() + M), mat,
                      1.0, S(S::invalid), ibra);
        if (mat.size() != data_idxs.back()) {
            orb_idxs.push_back(orbs);
            data_idxs.push_back(mat.size());
        }
    }
    // same site multiplication factor
    double site_factor(int8_t nop, uint8_t q_pattern, uint8_t ops,
                       int8_t ctr_start = 0, uint8_t ctr_order = 0) const {
        int8_t aqj = (q_pattern >> ctr_start) & 1;
        int8_t aqpj = (q_pattern >> (ctr_start + 1)) & 1;
        int8_t adq = 1, bdq = 1, cdq = 0;
        int8_t imin = ctr_start, imax = ctr_start;
        // 3 indices the same but it is special pattern 7
        if (!((ops >> (ctr_start << 1)) & 2))
            return 0;
        double factor = 1;
        for (int8_t i = 1; i < nop; i++) {
            if (!((ctr_order >> i) & 1)) {
                cdq = adq + ((ops >> ((imax + 1) << 1)) & 2) - 1;
                assert(cdq >= 0);
                int8_t cqpj = (q_pattern >> (imax + 2)) & 1;
                factor *= cg->racah(cqpj, bdq, aqj, adq, aqpj, cdq);
                factor *= sqrt((cdq + 1) * (aqpj + 1)) *
                          (((adq + bdq - cdq) & 2) ? -1 : 1);
                adq = cdq, aqpj = cqpj, imax++;
            } else {
                cdq = adq + ((ops >> ((imin - 1) << 1)) & 2) - 1;
                assert(cdq >= 0);
                int8_t cqj = (q_pattern >> (imin - 1)) & 1;
                factor *= cg->racah(aqpj, adq, cqj, bdq, aqj, cdq);
                factor *= sqrt((cdq + 1) * (aqj + 1)) *
                          (((adq + bdq - cdq) & 2) ? -1 : 1);
                adq = cdq, aqj = cqj, imin--;
            }
        }
        assert(imin == 0 && imax == nop - 1);
        return factor;
    }
    // 0 = D, 1 = C, 0 = -, 2 = +
    void cfg_apply_ops(LL iket, uint8_t ops, vector<uint16_t> orb_idxs,
                       vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat,
                       double scale = 1.0, S target_bra_q = S(S::invalid),
                       LL ibra_ref = -1) const {
        int op_len = (int)orb_idxs.size();
        assert(is_sorted(orb_idxs.begin(), orb_idxs.end()));
        vector<uint8_t> ket = get_config(iket), bra = ket;
        double factor = scale, factor_site = 1;
        int iubra = 0, iuket = 0;
        int adq = 0, aket = 0, abra = 0;
        vector<int8_t> site_spin_chg(n_orbs, 0);
        vector<int8_t> site_op_count(n_orbs, 0);
        vector<uint8_t> site_spin_pattern(n_orbs, 0);
        if (is_right)
            for (int i = 0; i < op_len && factor != 0; i++) {
                int l = (int)orb_idxs[i];
                assert(l < n_orbs);
                site_spin_chg[l] += ((ops >> (i << 1)) & 2) ? 1 : -1;
                const uint8_t x = (bra[l >> 2] >> ((l & 3) << 1)) & 3;
                if (site_op_count[l] == 0)
                    site_spin_pattern[l] = (x == 1 || x == 2);
                if ((ops >> (i << 1)) & 1) {
                    // C
                    if (x == 3)
                        factor = 0;
                    else if (x == 2 || x == 1)
                        bra[l >> 2] |= (3 << ((l & 3) << 1)),
                            factor *= -sqrt(2);
                    else
                        bra[l >> 2] |= (2 << ((l & 3) << 1)), factor *= 1.0;
                } else {
                    // D
                    if (x == 0)
                        factor = 0;
                    else if (x == 2 || x == 1)
                        bra[l >> 2] ^= (x << ((l & 3) << 1)), factor *= sqrt(2);
                    else
                        bra[l >> 2] ^= (1 << ((l & 3) << 1)), factor *= 1.0;
                }
                const uint8_t xb = (bra[l >> 2] >> ((l & 3) << 1)) & 3;
                site_spin_pattern[l] |= (xb == 1 || xb == 2)
                                        << (++site_op_count[l]);
            }
        else
            for (int i = op_len - 1; i >= 0 && factor != 0; i--) {
                int l = (int)orb_idxs[i];
                assert(l < n_orbs);
                site_spin_chg[l] += ((ops >> (i << 1)) & 2) ? 1 : -1;
                const uint8_t x = (bra[l >> 2] >> ((l & 3) << 1)) & 3;
                if (site_op_count[l] == 0)
                    site_spin_pattern[l] = (x == 1 || x == 2);
                if ((ops >> (i << 1)) & 1) {
                    // C
                    if (x == 3)
                        factor = 0;
                    else if (x == 2 || x == 1)
                        bra[l >> 2] |= (3 << ((l & 3) << 1)),
                            factor *= -sqrt(2);
                    else
                        bra[l >> 2] |= (2 << ((l & 3) << 1)), factor *= 1.0;
                } else {
                    // D
                    if (x == 0)
                        factor = 0;
                    else if (x == 2 || x == 1)
                        bra[l >> 2] ^= (x << ((l & 3) << 1)), factor *= sqrt(2);
                    else
                        bra[l >> 2] ^= (1 << ((l & 3) << 1)), factor *= 1.0;
                }
                const uint8_t xb = (bra[l >> 2] >> ((l & 3) << 1)) & 3;
                site_spin_pattern[l] =
                    (site_spin_pattern[l] << 1) | (xb == 1 || xb == 2);
                site_op_count[l]++;
            }
        if (factor == 0)
            return;
        if (op_len == 3) {
            const int la = (int)orb_idxs[0], lb = (int)orb_idxs[1],
                      lc = (int)orb_idxs[2];
            int8_t x = (la == lb) | ((lb == lc) << 1);
            const uint8_t ppm = 2 | (2 << 2);
            const uint8_t mpp = (2 << 2) | (2 << 4);
            switch (x) {
            case 1 | 2:
                // (xx)0x +-+(++-) | (xx)1x ++-(-++) (not normal)
                factor_site *=
                    site_factor(site_op_count[la], site_spin_pattern[la],
                                (ops & 8) ? mpp : ppm, 1, 1 << 2);
                break;
            case 2:
                // (xx)0x +-+(++-) | (xx)1x ++-(-++)
                factor_site *=
                    site_factor(site_op_count[lb], site_spin_pattern[lb], ops);
                site_spin_chg[lb] = (ops & 8) ? -2 : 0;
                break;
            case 1:
            default:
                // x(xx)0 +-+ | x(xx)1 ++- (normal)
                factor_site *=
                    site_factor(site_op_count[la], site_spin_pattern[la], ops);
                break;
            }
        } else if (op_len == 4) {
            const int la = (int)orb_idxs[0], lb = (int)orb_idxs[1],
                      lc = (int)orb_idxs[2], ld = (int)orb_idxs[3];
            int8_t x = (la == lb) | ((lb == lc) << 1) | ((lc == ld) << 2);
            const uint8_t ppmm = 2 | (2 << 2);
            const uint8_t mppm = (2 << 2) | (2 << 4);
            const uint8_t mpmp = (2 << 2) | (2 << 6);
            const uint8_t mmpp = (2 << 4) | (2 << 6);
            switch (x) {
            case 1 | 2 | 4:
                if (((ops >> 4) & 2) != ((ops >> 2) & 2))
                    factor_site = 0;
                else
                    // [x((xx)0x)] +--+(++--) | [x((xx)1x)] -++-(-++-)
                    factor_site *=
                        site_factor(site_op_count[la], site_spin_pattern[la],
                                    (ops & 2) ? ppmm : mppm, 1, 1 << 2);
                break;
            case 1 | 2:
                if (((ops >> 4) & 2) != ((ops >> 2) & 2))
                    // [x(x(xx)0)] +-+- | [x(x(xx)1)] ++-- (normal)
                    factor_site *= site_factor(site_op_count[la],
                                               site_spin_pattern[la], ops);
                else {
                    // [x((xx)0x)] +--+(++--) | [x((xx)1x)] -++-(-++-)
                    factor_site *=
                        site_factor(site_op_count[la], site_spin_pattern[la],
                                    (ops & 2) ? ppmm : mppm, 1, 1 << 2);
                    site_spin_chg[la] = 1;
                    site_spin_chg[ld] = -1;
                }
                break;
            case 2 | 4:
                if (((ops >> 4) & 2) != ((ops >> 2) & 2)) {
                    // [(x(xx)0)x] +-+-(-+-+) | [(x(xx)1)x] ++--(-++-)
                    factor_site *= site_factor(
                        site_op_count[lb], site_spin_pattern[lb],
                        (ops & 8) ? (mppm >> 2) : (mpmp >> 2), 0, 1 << 3);
                } else {
                    // [((xx)0x)x] +--+(-++-) | [((xx)1x)x] -++-(--++)
                    factor_site *=
                        site_factor(site_op_count[lb], site_spin_pattern[lb],
                                    (ops & 8) ? (mmpp >> 2) : (mppm >> 2), 1,
                                    (1 << 2) | (1 << 3));
                    site_spin_chg[la] = 1;
                    site_spin_chg[lb] = -1;
                }
                break;
            case 2:
                if (((ops >> 4) & 2) != ((ops >> 2) & 2))
                    factor_site = 0;
                else {
                    // [x((xx)0x)] +--+(++--) | [x((xx)1x)] -++-(-++-)
                    factor_site *=
                        site_factor(site_op_count[lb], site_spin_pattern[lb],
                                    (ops & 2) ? (ppmm >> 2) : (mppm >> 2));
                    site_spin_chg[la] = 1;
                    site_spin_chg[lb] = (ops & 2) ? 0 : -2;
                    site_spin_chg[ld] = -1;
                }
                break;
            default:
                if (((ops >> 4) & 2) == ((ops >> 2) & 2))
                    factor_site = 0;
                else {
                    // [(xx)0(xx)0] +-+-(+-+-) | [(xx)1(xx)1] ++--(++++)
                    if (la == lb)
                        factor_site *= site_factor(site_op_count[la],
                                                   site_spin_pattern[la], ops);
                    if (lc == ld) {
                        factor_site *= site_factor(site_op_count[lc],
                                                   site_spin_pattern[lc], ops);
                        site_spin_chg[lc] = (ops & 8) ? -4 : 0;
                    }
                }
                break;
            }
        } else {
            for (int i = 0; i < op_len && factor_site != 0; i++) {
                int l = (int)orb_idxs[i];
                if (site_op_count[l] > 1) {
                    factor_site *=
                        site_factor(site_op_count[l], site_spin_pattern[l],
                                    ops >> (i << 1));
                    i += site_op_count[l] - 1;
                }
            }
        }
        if (factor_site == 0)
            return;
        factor *= factor_site;
        vector<int> op_idxs;
        uint8_t eff_ops = 0;
        uint8_t op_types = 0; // 2 bra, 1 ket
        op_idxs.reserve(op_len);
        int i_eop = 0, i_opt = 0, xops = ops;
        for (int i = 0; i < n_orbs; i++) {
            uint8_t xbra = (bra[i >> 2] >> ((i & 3) << 1)) & 3;
            uint8_t xket = (ket[i >> 2] >> ((i & 3) << 1)) & 3;
            if (site_op_count[i] % 2 == 1) {
                assert(site_spin_chg[i] == 1 || site_spin_chg[i] == -1);
                eff_ops |= ((site_spin_chg[i] == 1 ? 2 : 0) << i_eop),
                    i_eop += 2;
                if ((xbra == 2 || xbra == 1) && !(xket == 2 || xket == 1))
                    op_types |= (2 << i_opt), i_opt += 2,
                        op_idxs.push_back(iubra);
                else if (!(xbra == 2 || xbra == 1) && (xket == 2 || xket == 1))
                    op_types |= (1 << i_opt), i_opt += 2,
                        op_idxs.push_back(iuket);
                else
                    assert(false);
            } else if (site_op_count[i] == 2) {
                if ((xbra == 2 || xbra == 1) && (xket == 2 || xket == 1) &&
                    site_spin_chg[i] != 0) {
                    if (site_spin_chg[i] == -2)
                        eff_ops |= (2 << i_eop), i_eop += 4;
                    else if (site_spin_chg[i] == -4)
                        i_eop += 4;
                    else
                        eff_ops |= ((xops & 15) << i_eop), i_eop += 4;
                    op_types |= (3 << i_opt), i_opt += 2;
                    op_idxs.push_back(iubra);
                    op_idxs.push_back(iuket);
                }
            } else if (site_op_count[i] == 4)
                assert(site_spin_chg[i] == 0);
            xops >>= (2 * site_op_count[i]);
            iubra += (xbra == 2 || xbra == 1);
            iuket += (xket == 2 || xket == 1);
        }
        int op_n = 0, op_twos = 0, op_pg = 0;
        for (int i = 0; i < op_len; i++) {
            op_twos += ((ops >> (i << 1)) & 2) - 1;
            op_n += (((ops >> (i << 1)) & 1) << 1) - 1;
        }
        S op_dq(op_n, op_twos, op_pg);
        const int ket_i_unpaired =
            (int)(upper_bound(n_unpaired_idxs.begin(), n_unpaired_idxs.end(),
                              iket) -
                  n_unpaired_idxs.begin() - 1);
        assert(ket_i_unpaired >= 0);
        assert(iuket == n_unpaired[ket_i_unpaired]);
        const int ket_i_qs =
            (int)(upper_bound(qs_idxs.begin(), qs_idxs.end(), ket_i_unpaired) -
                  qs_idxs.begin() - 1);
        assert(ket_i_qs >= 0);
        S ket_q = qs[ket_i_qs];
        S bra_qs = ket_q + op_dq;
        for (int ib = 0; ib < bra_qs.count(); ib++) {
            S bra_q = bra_qs[ib];
            if (iubra < bra_q.twos())
                continue;
            set_config_twos(bra, iubra, bra_q.twos());
            LL ibra = index_config(bra);
            if (ibra_ref != -1 && ibra != ibra_ref)
                continue;
            if (ibra == -1)
                continue;
            if (target_bra_q != S(S::invalid) && bra_q != target_bra_q)
                continue;
            const int bra_i_unpaired =
                (int)(upper_bound(n_unpaired_idxs.begin(),
                                  n_unpaired_idxs.end(), ibra) -
                      n_unpaired_idxs.begin() - 1);
            assert(bra_i_unpaired >= 0);
            const int bra_i_qs =
                (int)(upper_bound(qs_idxs.begin(), qs_idxs.end(),
                                  bra_i_unpaired) -
                      qs_idxs.begin() - 1);
            assert(bra_i_qs >= 0);
            assert(iubra == n_unpaired[bra_i_unpaired]);
            LL n_bra, n_ket;
            vector<double> rr =
                csf_apply_ops(iubra, bra_q.twos(), iuket, ket_q.twos(), eff_ops,
                              op_types, op_idxs, factor, n_bra, n_ket);
            assert(rr.size() != 0);
            const LL i_abs_row =
                csf_offsets[bra_i_unpaired] +
                n_bra * (ibra - n_unpaired_idxs[bra_i_unpaired]);
            const LL i_abs_col =
                csf_offsets[ket_i_unpaired] +
                n_ket * (iket - n_unpaired_idxs[ket_i_unpaired]);
            const LL irow = i_abs_row - csf_offsets[qs_idxs[bra_i_qs]];
            const LL icol = i_abs_col - csf_offsets[qs_idxs[ket_i_qs]];
            for (int kb = 0; kb < n_bra; kb++)
                for (int kk = 0; kk < n_ket; kk++) {
                    // cout << (*this)[i_abs_row + kb] << " "
                    //      << (*this)[i_abs_col + kk] << "[" << mat.size()
                    //      << "] = " << rr[kb * n_ket + kk] << endl;
                    mat.emplace_back(make_pair(irow + kb, icol + kk),
                                     rr[kb * n_ket + kk]);
                }
        }
    }
    // only spin coupling inc/dec in ops is used
    // op_types is for ket or bra is zero
    vector<double> csf_apply_ops(int n_unpaired_bra, int twos_bra,
                                 int n_unpaired_ket, int twos_ket, uint8_t ops,
                                 uint8_t op_types, const vector<int> &op_idxs,
                                 double scale, LL &n_bra, LL &n_ket) const {
        return is_right ? csf_apply_ops_impl<true>(n_unpaired_bra, twos_bra,
                                                   n_unpaired_ket, twos_ket,
                                                   ops, op_types, op_idxs,
                                                   scale, n_bra, n_ket)
                        : csf_apply_ops_impl<false>(n_unpaired_bra, twos_bra,
                                                    n_unpaired_ket, twos_ket,
                                                    ops, op_types, op_idxs,
                                                    scale, n_bra, n_ket);
    }
    template <bool vt>
    vector<double> csf_apply_ops_impl(int n_unpaired_bra, int twos_bra,
                                      int n_unpaired_ket, int twos_ket,
                                      uint8_t ops, uint8_t op_types,
                                      const vector<int> &op_idxs, double scale,
                                      LL &n_bra, LL &n_ket) const {
        vector<double> r;
        if (n_unpaired_ket < 0 || n_unpaired_ket > n_max_unpaired ||
            n_unpaired_bra < 0 || n_unpaired_bra > n_max_unpaired)
            return r;
        if (twos_ket < 0 || twos_ket > n_unpaired_ket || twos_bra < 0 ||
            twos_bra > n_unpaired_bra)
            return r;
        LL i_csf_bra = csf_sub_idxs[csf_idxs[n_unpaired_bra] + twos_bra];
        LL j_csf_bra = csf_sub_idxs[csf_idxs[n_unpaired_bra] + twos_bra + 1];
        LL i_csf_ket = csf_sub_idxs[csf_idxs[n_unpaired_ket] + twos_ket];
        LL j_csf_ket = csf_sub_idxs[csf_idxs[n_unpaired_ket] + twos_ket + 1];
        if (j_csf_bra == i_csf_bra || j_csf_ket == i_csf_ket)
            return r;
        int cl_bra = max((n_unpaired_bra >> 3) + !!(n_unpaired_bra & 7), 1);
        int cl_ket = max((n_unpaired_ket >> 3) + !!(n_unpaired_ket & 7), 1);
        n_bra = (j_csf_bra - i_csf_bra) / cl_bra;
        n_ket = (j_csf_ket - i_csf_ket) / cl_ket;
        r.resize(n_bra * n_ket, scale);
        LL ir = 0;
        for (LL ibra = 0, ixbra = i_csf_bra; ibra < n_bra;
             ibra++, ixbra += cl_bra)
            for (LL iket = 0, ixket = i_csf_ket; iket < n_ket;
                 iket++, ixket += cl_ket) {
                double &rr = r[ir++];
                const uint8_t *csf_bra = &csfs[ixbra];
                const uint8_t *csf_ket = &csfs[ixket];
                int bb = 0, bk = 0, db = 0, cb, ck, dc;
                const int ab = 1, ak = 1, da = 0;
                uint8_t xbra, xket;
                for (int jbra = 0, jket = 0, kop = 0, lops = ops,
                         lopt = op_types;
                     jbra < n_unpaired_bra || jket < n_unpaired_ket;) {
                    if ((lopt & 3) == 2 && kop < (int)op_idxs.size() &&
                        jbra == op_idxs[kop]) {
                        xbra = (csf_bra[jbra >> 3] >> (jbra & 7)) & 1;
                        cb = bb + (xbra << 1) - 1, ck = bk + 0,
                        dc = db + (lops & 2) - 1;
                        assert(dc >= 0);
                        rr *= sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                        if (vt) {
                            rr *= cg->wigner_9j(0, bk, ck, ab, db, dc, ab, bb,
                                                cb);
                            rr *= ((0 & 1) & (db & 1)) ? -1 : 1;
                        } else {
                            rr *= cg->wigner_9j(bk, 0, ck, db, ab, dc, bb, ab,
                                                cb);
                            rr *= ((bk & 1) & (ab & 1)) ? -1 : 1;
                        }
                        lops >>= 2, lopt >>= 2;
                        jbra++, kop++;
                    } else if ((lopt & 3) == 1 && kop < (int)op_idxs.size() &&
                               jket == op_idxs[kop]) {
                        xket = (csf_ket[jket >> 3] >> (jket & 7)) & 1;
                        cb = bb + 0, ck = bk + (xket << 1) - 1,
                        dc = db + (lops & 2) - 1;
                        assert(dc >= 0);
                        rr *= sqrt((ck + 1) * (dc + 1) * (0 + 1) * (bb + 1));
                        if (vt) {
                            rr *= cg->wigner_9j(ak, bk, ck, ak, db, dc, 0, bb,
                                                cb);
                            rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                        } else {
                            rr *= cg->wigner_9j(bk, ak, ck, db, ak, dc, bb, 0,
                                                cb);
                            rr *= ((bk & 1) & (ak & 1)) ? -1 : 1;
                        }
                        lops >>= 2, lopt >>= 2;
                        jket++, kop++;
                    } else if ((lopt & 3) == 3 &&
                               kop + 1 < (int)op_idxs.size() &&
                               jbra == op_idxs[kop] &&
                               jket == op_idxs[kop + 1]) {
                        xbra = (csf_bra[jbra >> 3] >> (jbra & 7)) & 1;
                        xket = (csf_ket[jket >> 3] >> (jket & 7)) & 1;
                        cb = bb + (xbra << 1) - 1, ck = bk + (xket << 1) - 1,
                        dc = db + (lops & 2) - 1 + ((lops >> 2) & 2) - 1;
                        assert(dc >= 0);
                        rr *= sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                        if (vt) {
                            rr *= cg->wigner_9j(ak, bk, ck, 2, db, dc, ab, bb,
                                                cb);
                            rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                        } else {
                            rr *= cg->wigner_9j(bk, ak, ck, db, 2, dc, bb, ab,
                                                cb);
                            rr *= ((bk & 1) & (2 & 1)) ? -1 : 1;
                        }
                        lops >>= 4, lopt >>= 2;
                        jbra++, jket++, kop += 2;
                    } else {
                        xbra = (csf_bra[jbra >> 3] >> (jbra & 7)) & 1;
                        xket = (csf_ket[jket >> 3] >> (jket & 7)) & 1;
                        cb = bb + (xbra << 1) - 1, ck = bk + (xket << 1) - 1,
                        dc = db;
                        rr *= sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                        if (vt) {
                            rr *= cg->wigner_9j(ak, bk, ck, da, db, dc, ab, bb,
                                                cb);
                            rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                        } else {
                            rr *= cg->wigner_9j(bk, ak, ck, db, da, dc, bb, ab,
                                                cb);
                            rr *= ((bk & 1) & (da & 1)) ? -1 : 1;
                        }
                        jbra++, jket++;
                    }
                    bb = cb, bk = ck, db = dc;
                }
            }
        return r;
    }
    // csf index to csf
    string operator[](LL idx) const {
        string r(n_orbs, '0');
        int i_unpaired =
            (int)(upper_bound(csf_offsets.begin(), csf_offsets.end(), idx) -
                  csf_offsets.begin() - 1);
        assert(i_unpaired >= 0);
        int i_qs =
            (int)(upper_bound(qs_idxs.begin(), qs_idxs.end(), i_unpaired) -
                  qs_idxs.begin() - 1);
        assert(i_qs >= 0);
        const int n_unpaired = this->n_unpaired[i_unpaired];
        const int n_double = (qs[i_qs].n() - n_unpaired) >> 1;
        const int n_empty = n_orbs - n_double - n_unpaired;
        const int twos = qs[i_qs].twos();
        const LL i_csf = csf_sub_idxs[csf_idxs[n_unpaired] + twos];
        const LL j_csf = csf_sub_idxs[csf_idxs[n_unpaired] + twos + 1];
        const int cl = max((n_unpaired >> 3) + !!(n_unpaired & 7), 1);
        const int ncsf = (int)((j_csf - i_csf) / cl);
        const LL icfg = (idx - csf_offsets[i_unpaired]) / ncsf;
        const LL icfgc = (idx - csf_offsets[i_unpaired]) % ncsf * cl;
        const LL icfgd = icfg / n_unpaired_shapes[i_unpaired].second;
        const LL icfgs = icfg % n_unpaired_shapes[i_unpaired].second;
        LL cur = 0;
        for (int i = 0, l = 0; i < n_double; i++) {
            for (; l < n_orbs; l++) {
                int nn = combinatorics->combination(n_orbs - l - 1,
                                                    n_double - i - 1);
                if (cur + nn > icfgd) {
                    r[l++] = '2';
                    break;
                } else
                    cur += nn;
            }
        }
        cur = 0;
        for (int i = 0, l = 0, k = n_double; i < n_unpaired; i++) {
            for (; l < n_orbs; l++) {
                if (r[l] == '2') {
                    k--;
                    continue;
                }
                int nn = combinatorics->combination(n_orbs - k - l - 1,
                                                    n_unpaired - i - 1);
                if (cur + nn > icfgs) {
                    r[l++] = ((csfs[i_csf + icfgc + (i >> 3)] >> (i & 7)) & 1)
                                 ? '+'
                                 : '-';
                    break;
                } else
                    cur += nn;
            }
        }
        return r;
    }
};

template <typename, typename = void> struct CSFBigSite;

template <typename S> struct CSFBigSite<S, typename S::is_su2_t> : BigSite<S> {
    typedef long long LL;
    using BigSite<S>::n_orbs;
    using BigSite<S>::basis;
    using BigSite<S>::op_infos;
    shared_ptr<FCIDUMP> fcidump;
    shared_ptr<CSFSpace<S>> csf_space;
    bool is_right;
    int iprint;
    CSFBigSite(int n_orbs, int n_max_elec, bool is_right,
               const shared_ptr<FCIDUMP> &fcidump,
               const vector<uint8_t> &orb_sym, int iprint = 0)
        : BigSite<S>(n_orbs), csf_space(make_shared<CSFSpace<S>>(
                                  n_orbs, n_max_elec, is_right, orb_sym)),
          is_right(is_right), fcidump(fcidump), iprint(iprint) {
        basis = csf_space->basis;
        op_infos = get_site_op_infos(orb_sym);
    }
    CSFBigSite(shared_ptr<CSFSpace<S>> csf_space,
               const shared_ptr<FCIDUMP> &fcidump,
               const vector<uint8_t> &orb_sym)
        : BigSite<S>(csf_space->n_orbs), csf_space(csf_space),
          is_right(csf_space->is_right), fcidump(fcidump) {
        basis = csf_space->basis;
        op_infos = get_site_op_infos(orb_sym);
    }
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
    get_site_op_infos(const vector<uint8_t> &orb_sym) {
        shared_ptr<VectorAllocator<uint32_t>> i_alloc =
            make_shared<VectorAllocator<uint32_t>>();
        map<S, shared_ptr<SparseMatrixInfo<S>>> info;
        info[S(0)] = nullptr;
        for (auto ipg : orb_sym) {
            for (int n = -1; n <= 1; n += 2)
                for (int s = 1; s <= 3; s += 2)
                    info[S(n, s, ipg)] = nullptr;
            for (auto jpg : orb_sym)
                for (int n = -2; n <= 2; n += 2)
                    for (int s = 0; s <= 4; s += 2)
                        info[S(n, s, ipg ^ jpg)] = nullptr;
        }
        for (auto &p : info) {
            p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
            p.second->initialize(*basis, *basis, p.first, p.first.is_fermion());
        }
        return vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                info.end());
    }
    static void
    fill_csr_matrix(vector<pair<pair<MKL_INT, MKL_INT>, double>> &data,
                    CSRMatrixRef &mat) {
        const size_t n = data.size();
        assert(mat.data == nullptr);
        assert(mat.alloc != nullptr);
        vector<size_t> idx(n), idx2;
        for (size_t i = 0; i < n; i++)
            idx[i] = i;

        sort(idx.begin(), idx.end(), [&data](size_t i, size_t j) {
            return data[i].first < data[j].first;
        });
        for (auto ii : idx)
            if (idx2.empty() || data[ii].first != data[idx2.back()].first)
                idx2.push_back(ii);
            else
                data[idx2.back()].second += data[ii].second;
        mat.nnz = (MKL_INT)idx2.size();
        mat.allocate();
        if (mat.nnz < mat.size()) {
            MKL_INT cur_row = -1;
            for (size_t k = 0; k < idx2.size(); k++) {
                while (data[idx2[k]].first.first != cur_row)
                    mat.rows[++cur_row] = k;
                mat.data[k] = data[idx2[k]].second,
                mat.cols[k] = data[idx2[k]].first.second;
            }
            while (mat.m != cur_row)
                mat.rows[++cur_row] = mat.nnz;
        } else
            for (size_t k = 0; k < idx2.size(); k++)
                mat.data[k] = data[idx2[k]].second;
    }
    static void fill_csr_matrix_rev(
        const vector<pair<pair<MKL_INT, MKL_INT>, double>> &data,
        CSRMatrixRef &mat, vector<double> &data_rev) {
        const size_t n = data.size();
        assert(mat.data == nullptr);
        assert(mat.alloc != nullptr);
        vector<size_t> idx(n), idx2;
        for (size_t i = 0; i < n; i++)
            idx[i] = i;

        sort(idx.begin(), idx.end(), [&data](size_t i, size_t j) {
            return data[i].first < data[j].first;
        });
        for (auto ii : idx)
            if (idx2.empty() || data[ii].first != data[idx2.back()].first)
                idx2.push_back(ii);
            else
                data_rev[idx2.back()] += data_rev[ii];
        mat.nnz = (MKL_INT)idx2.size();
        mat.allocate();
        if (mat.nnz < mat.size()) {
            MKL_INT cur_row = -1;
            for (size_t k = 0; k < idx2.size(); k++) {
                while (data[idx2[k]].first.first != cur_row)
                    mat.rows[++cur_row] = k;
                mat.data[k] = data_rev[idx2[k]],
                mat.cols[k] = data[idx2[k]].first.second;
            }
            while (mat.m != cur_row)
                mat.rows[++cur_row] = mat.nnz;
        } else
            for (size_t k = 0; k < idx2.size(); k++)
                mat.data[k] = data_rev[idx2[k]];
    }
    void build_site_op(uint8_t ops, const vector<uint16_t> &orb_idxs,
                       const shared_ptr<CSRSparseMatrix<S>> &mat,
                       double scale = 1.0) const {
        int ntg = threading->activate_global();
        vector<vector<pair<pair<MKL_INT, MKL_INT>, double>>> data(ntg);
        vector<shared_ptr<VectorAllocator<double>>> d_allocs(ntg, nullptr);
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int i = 0; i < mat->info->n; i++) {
            const int tid = threading->get_thread_id();
            if (d_allocs[tid] == nullptr)
                d_allocs[tid] = make_shared<VectorAllocator<double>>();
            S ket = mat->info->quanta[i].get_ket();
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            int iket = basis->find_state(ket);
            const int ka = csf_space->csf_offsets[csf_space->qs_idxs[iket]];
            const int kb = csf_space->csf_offsets[csf_space->qs_idxs[iket + 1]];
            const LL uka = csf_space->n_unpaired_idxs[csf_space->qs_idxs[iket]];
            const LL ukb =
                csf_space->n_unpaired_idxs[csf_space->qs_idxs[iket + 1]];
            data[tid].clear();
            for (LL k = uka; k < ukb; k++)
                csf_space->cfg_apply_ops(k, ops, orb_idxs, data[tid], scale,
                                         bra);
            mat->csr_data[i]->alloc = d_allocs[tid];
            fill_csr_matrix(data[tid], *mat->csr_data[i]);
        }
        threading->activate_normal();
    }
    template <typename IntOp, int8_t L, int8_t M>
    void build_complementary_site_ops(
        const vector<uint8_t> &ops, IntOp &xop,
        const vector<pair<array<uint16_t, L>, shared_ptr<CSRSparseMatrix<S>>>>
            &mats,
        double scale = 1.0) const {
        if (mats.size() == 0)
            return;
        int ntg = threading->activate_global();
        // when using openmp, use different vector for different threads
        vector<vector<pair<pair<MKL_INT, MKL_INT>, double>>> pdata(ntg);
        vector<vector<array<uint16_t, M>>> porb_idxs(ntg);
        vector<vector<size_t>> pdata_idxs(ntg);
        vector<vector<double>> pdata_rev(ntg);
        vector<shared_ptr<VectorAllocator<double>>> d_allocs(ntg, nullptr);
        shared_ptr<SparseMatrixInfo<S>> info = mats[0].second->info;
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int i = 0; i < info->n; i++) {
            const int tid = threading->get_thread_id();
            if (d_allocs[tid] == nullptr)
                d_allocs[tid] = make_shared<VectorAllocator<double>>();
            vector<pair<pair<MKL_INT, MKL_INT>, double>> &data = pdata[tid];
            vector<array<uint16_t, M>> &orb_idxs = porb_idxs[tid];
            vector<size_t> &data_idxs = pdata_idxs[tid];
            vector<double> &data_rev = pdata_rev[tid];
            S ket = info->quanta[i].get_ket();
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            int iket = basis->find_state(ket);
            int ibra = basis->find_state(bra);
            const int ba = csf_space->csf_offsets[csf_space->qs_idxs[ibra]];
            const int bb = csf_space->csf_offsets[csf_space->qs_idxs[ibra + 1]];
            const int ka = csf_space->csf_offsets[csf_space->qs_idxs[iket]];
            const int kb = csf_space->csf_offsets[csf_space->qs_idxs[iket + 1]];
            const LL uba = csf_space->n_unpaired_idxs[csf_space->qs_idxs[ibra]];
            const LL ubb =
                csf_space->n_unpaired_idxs[csf_space->qs_idxs[ibra + 1]];
            const LL uka = csf_space->n_unpaired_idxs[csf_space->qs_idxs[iket]];
            const LL ukb =
                csf_space->n_unpaired_idxs[csf_space->qs_idxs[iket + 1]];
            data.clear();
            orb_idxs.clear();
            data_idxs.clear();
            vector<size_t> op_idxs(ops.size() + 1, 0);
            for (uint8_t iop = 0; iop < (uint8_t)ops.size(); iop++) {
                for (LL k = uka; k < ukb; k++)
                    for (LL b = uba; b < ubb; b++)
                        csf_space->template cfg_op_matrix_element<M>(
                            b, k, ops[iop], data, orb_idxs, data_idxs);
                op_idxs[iop + 1] = orb_idxs.size();
            }
            data_rev.resize(data.size());
            for (auto &mat : mats) {
                for (uint8_t iop = 0; iop < (uint8_t)ops.size(); iop++)
                    for (size_t k = op_idxs[iop]; k < op_idxs[iop + 1]; k++) {
                        const double factor = xop(iop, mat.first, orb_idxs[k]);
                        for (size_t l = data_idxs[k]; l < data_idxs[k + 1]; l++)
                            data_rev[l] = data[l].second * factor * scale;
                    }
                mat.second->csr_data[i]->alloc = d_allocs[tid];
                fill_csr_matrix_rev(data, *mat.second->csr_data[i], data_rev);
            }
        }
        threading->activate_normal();
    }
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override {
        uint16_t i, j, k;
        uint8_t s;
        shared_ptr<SparseMatrix<S>> zero =
            make_shared<SparseMatrix<S>>(nullptr);
        zero->factor = 0.0;
        const uint8_t i_ops = 0, c_ops = 3, d_ops = 2, c2_ops = 1, d2_ops = 0;
        const uint8_t a0_ops = c_ops + (c2_ops << 2),
                      a1_ops = c_ops + (c_ops << 2);
        const uint8_t ad0_ops = d_ops + (d2_ops << 2),
                      ad1_ops = d_ops + (d_ops << 2);
        const uint8_t b0_ops = d_ops + (c2_ops << 2),
                      b1_ops = d_ops + (c_ops << 2);
        const uint8_t bd0_ops = c_ops + (d2_ops << 2),
                      bd1_ops = c_ops + (d_ops << 2);
        const uint8_t dxx_ops = d_ops << 4, cxx_ops = c_ops << 4;      // 0
        const uint8_t dcd0_ops = d_ops + (c2_ops << 2) + (d_ops << 4); // 1
        const uint8_t dcd1_ops = d_ops + (c_ops << 2) + (d2_ops << 4); // 2
        const uint8_t cdd0_ops = c_ops + (d2_ops << 2) + (d_ops << 4); // 3
        const uint8_t cdd1_ops = c_ops + (d_ops << 2) + (d2_ops << 4); // 4
        const uint8_t ddc0_ops = d_ops + (d2_ops << 2) + (c_ops << 4); // 5
        const uint8_t ddc1_ops = d_ops + (d_ops << 2) + (c2_ops << 4); // 6
        const uint8_t dcc0_ops = d_ops + (c2_ops << 2) + (c_ops << 4); // 1
        const uint8_t dcc1_ops = d_ops + (c_ops << 2) + (c2_ops << 4); // 2
        const uint8_t ccd0_ops = c_ops + (c2_ops << 2) + (d_ops << 4); // 3
        const uint8_t ccd1_ops = c_ops + (c_ops << 2) + (d2_ops << 4); // 4
        const uint8_t cdc0_ops = c_ops + (d2_ops << 2) + (c_ops << 4); // 5
        const uint8_t cdc1_ops = c_ops + (d_ops << 2) + (c2_ops << 4); // 6
        const uint8_t dcxx_ops = b0_ops << 4, cdxx_ops = bd0_ops << 4; // 0 1
        const uint8_t ddcc0_ops =
            d_ops + (d2_ops << 2) + (c_ops << 4) + (c2_ops << 6); // 2
        const uint8_t ddcc1_ops =
            d_ops + (d_ops << 2) + (c2_ops << 4) + (c2_ops << 6); // 3
        const uint8_t ddcc2_ops =
            d_ops + (d2_ops << 2) + (c2_ops << 4) + (c_ops << 6); // 4
        const uint8_t ddcc3_ops =
            d2_ops + (d_ops << 2) + (c_ops << 4) + (c2_ops << 6); // 5
        const uint8_t ccdd0_ops =
            c_ops + (c2_ops << 2) + (d_ops << 4) + (d2_ops << 6); // 6
        const uint8_t ccdd1_ops =
            c_ops + (c_ops << 2) + (d2_ops << 4) + (d2_ops << 6); // 7
        const uint8_t cddc0_ops =
            c_ops + (d2_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 8
        const uint8_t cddc1_ops =
            c_ops + (d_ops << 2) + (d2_ops << 4) + (c2_ops << 6); // 9
        const uint8_t cddc2_ops =
            c_ops + (d2_ops << 2) + (d2_ops << 4) + (c_ops << 6); // 10
        const uint8_t cddc3_ops =
            c2_ops + (d_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 11
        const uint8_t cdcd0_ops =
            c_ops + (d2_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 12
        const uint8_t cdcd1_ops =
            c_ops + (d_ops << 2) + (c2_ops << 4) + (d2_ops << 6); // 13
        const uint8_t cdcd2_ops =
            c_ops + (d2_ops << 2) + (c2_ops << 4) + (d_ops << 6); // 14
        const uint8_t cdcd3_ops =
            c2_ops + (d_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 15
        const uint8_t dcdc0_ops =
            d_ops + (c2_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 16
        const uint8_t dcdc1_ops =
            d_ops + (c_ops << 2) + (d2_ops << 4) + (c2_ops << 6); // 17
        const uint8_t dccd0_ops =
            d_ops + (c2_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 18
        const uint8_t dccd1_ops =
            d_ops + (c_ops << 2) + (c2_ops << 4) + (d2_ops << 6); // 19
        const uint8_t dccd2_ops =
            d_ops + (c2_ops << 2) + (c2_ops << 4) + (d_ops << 6); // 20
        const uint8_t dccd3_ops =
            d2_ops + (c_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 21
        const uint8_t ccdd2_ops =
            c_ops + (c2_ops << 2) + (d2_ops << 4) + (d_ops << 6); // 6
        const uint8_t ccdd3_ops =
            c2_ops + (c_ops << 2) + (d_ops << 4) + (d2_ops << 6); // 7
        const uint8_t dcdc2_ops =
            d_ops + (c2_ops << 2) + (d2_ops << 4) + (c_ops << 6); // 16
        const uint8_t dcdc3_ops =
            d2_ops + (c_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 17
        unordered_map<
            S, vector<pair<array<uint16_t, 2>, shared_ptr<CSRSparseMatrix<S>>>>>
            p_mats[2], pd_mats[2], q_mats[2];
        unordered_map<
            S, vector<pair<array<uint16_t, 1>, shared_ptr<CSRSparseMatrix<S>>>>>
            r_mats, rd_mats;
        vector<pair<array<uint16_t, 0>, shared_ptr<CSRSparseMatrix<S>>>> h_mats;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            shared_ptr<VectorAllocator<double>> d_alloc =
                make_shared<VectorAllocator<double>>();
            shared_ptr<CSRSparseMatrix<S>> mat =
                make_shared<CSRSparseMatrix<S>>();
            mat->initialize(BigSite<S>::find_site_op_info(op.q_label));
            for (int i = 0; i < mat->info->n; i++)
                mat->csr_data[i]->alloc = d_alloc;
            p.second = mat;
            switch (op.name) {
            case OpNames::I:
                build_site_op(i_ops, {}, mat);
                break;
            case OpNames::C:
                i = is_right ? fcidump->n_sites() - 1 - op.site_index[0]
                             : op.site_index[0];
                build_site_op(c_ops, {i}, mat);
                break;
            case OpNames::D:
                i = is_right ? fcidump->n_sites() - 1 - op.site_index[0]
                             : op.site_index[0];
                build_site_op(d_ops, {i}, mat);
                break;
            case OpNames::A:
                i = is_right ? fcidump->n_sites() - 1 - op.site_index[0]
                             : op.site_index[0];
                j = is_right ? fcidump->n_sites() - 1 - op.site_index[1]
                             : op.site_index[1];
                s = op.site_index.ss();
                if (is_right) {
                    if (j <= i)
                        build_site_op(s ? a1_ops : a0_ops, {j, i}, mat);
                    else
                        build_site_op(s ? a1_ops : a0_ops, {i, j}, mat,
                                      s ? -1 : 1);
                } else {
                    if (i <= j)
                        build_site_op(s ? a1_ops : a0_ops, {i, j}, mat);
                    else
                        build_site_op(s ? a1_ops : a0_ops, {j, i}, mat,
                                      s ? -1 : 1);
                }
                break;
            case OpNames::AD:
                i = is_right ? fcidump->n_sites() - 1 - op.site_index[0]
                             : op.site_index[0];
                j = is_right ? fcidump->n_sites() - 1 - op.site_index[1]
                             : op.site_index[1];
                s = op.site_index.ss();
                // note that ad is defined as ad[i, j] = C[j] * C[i]
                if (is_right) {
                    if (i <= j)
                        build_site_op(s ? ad1_ops : ad0_ops, {i, j}, mat);
                    else
                        build_site_op(s ? ad1_ops : ad0_ops, {j, i}, mat,
                                      s ? -1 : 1);
                } else {
                    if (j <= i)
                        build_site_op(s ? ad1_ops : ad0_ops, {j, i}, mat);
                    else
                        build_site_op(s ? ad1_ops : ad0_ops, {i, j}, mat,
                                      s ? -1 : 1);
                }
                break;
            case OpNames::B:
                i = is_right ? fcidump->n_sites() - 1 - op.site_index[0]
                             : op.site_index[0];
                j = is_right ? fcidump->n_sites() - 1 - op.site_index[1]
                             : op.site_index[1];
                s = op.site_index.ss();
                if (is_right) {
                    if (j <= i)
                        build_site_op(s ? b1_ops : b0_ops, {j, i}, mat);
                    else
                        build_site_op(s ? bd1_ops : bd0_ops, {i, j}, mat,
                                      s ? -1 : 1);
                } else {
                    if (i <= j)
                        build_site_op(s ? bd1_ops : bd0_ops, {i, j}, mat);
                    else
                        build_site_op(s ? b1_ops : b0_ops, {j, i}, mat,
                                      s ? -1 : 1);
                }
                break;
            case OpNames::BD:
                i = is_right ? fcidump->n_sites() - 1 - op.site_index[0]
                             : op.site_index[0];
                j = is_right ? fcidump->n_sites() - 1 - op.site_index[1]
                             : op.site_index[1];
                s = op.site_index.ss();
                if (is_right) {
                    if (j <= i)
                        build_site_op(s ? bd1_ops : bd0_ops, {j, i}, mat);
                    else
                        build_site_op(s ? b1_ops : b0_ops, {i, j}, mat,
                                      s ? -1 : 1);
                } else {
                    if (i <= j)
                        build_site_op(s ? b1_ops : b0_ops, {i, j}, mat);
                    else
                        build_site_op(s ? bd1_ops : bd0_ops, {j, i}, mat,
                                      s ? -1 : 1);
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.ss();
                p_mats[s][op.q_label].push_back(
                    make_pair(array<uint16_t, 2>{i, j}, mat));
                break;
            case OpNames::PD:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.ss();
                pd_mats[s][op.q_label].push_back(
                    make_pair(array<uint16_t, 2>{i, j}, mat));
                break;
            case OpNames::Q:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.ss();
                q_mats[s][op.q_label].push_back(
                    make_pair(array<uint16_t, 2>{i, j}, mat));
                break;
            case OpNames::R:
                i = op.site_index[0];
                r_mats[op.q_label].push_back(
                    make_pair(array<uint16_t, 1>{i}, mat));
                break;
            case OpNames::RD:
                i = op.site_index[0];
                rd_mats[op.q_label].push_back(
                    make_pair(array<uint16_t, 1>{i}, mat));
                break;
            case OpNames::H:
                h_mats.push_back(make_pair(array<uint16_t, 0>{}, mat));
                break;
            default:
                assert(false);
                break;
            }
        }
        const function<double(uint8_t, const array<uint16_t, 2> &,
                              const array<uint16_t, 2> &)> &p0_ops =
            [this](uint8_t iop, const array<uint16_t, 2> &op_idxs,
                   const array<uint16_t, 2> &ctr_idxs) -> double {
            const uint16_t i = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[0]
                                   : ctr_idxs[1];
            const uint16_t j = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[1]
                                   : ctr_idxs[0];
            return i == j ? this->fcidump->v(op_idxs[0], i, op_idxs[1], j)
                          : this->fcidump->v(op_idxs[0], i, op_idxs[1], j) +
                                this->fcidump->v(op_idxs[0], j, op_idxs[1], i);
        };
        const function<double(uint8_t, const array<uint16_t, 2> &,
                              const array<uint16_t, 2> &)> &p1_ops =
            [this](uint8_t iop, const array<uint16_t, 2> &op_idxs,
                   const array<uint16_t, 2> &ctr_idxs) -> double {
            const uint16_t i = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[0]
                                   : ctr_idxs[1];
            const uint16_t j = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[1]
                                   : ctr_idxs[0];
            return i == j ? this->fcidump->v(op_idxs[0], i, op_idxs[1], j)
                          : this->fcidump->v(op_idxs[0], j, op_idxs[1], i) -
                                this->fcidump->v(op_idxs[0], i, op_idxs[1], j);
        };
        const function<double(uint8_t, const array<uint16_t, 2> &,
                              const array<uint16_t, 2> &)> &q0_ops =
            [this](uint8_t iop, const array<uint16_t, 2> &op_idxs,
                   const array<uint16_t, 2> &ctr_idxs) -> double {
            const uint16_t i = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[0]
                                   : ctr_idxs[1];
            const uint16_t j = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[1]
                                   : ctr_idxs[0];
            return i == j ? (iop ? 0
                                 : 2 * this->fcidump->v(op_idxs[0], op_idxs[1],
                                                        j, i) -
                                       this->fcidump->v(op_idxs[0], i, j,
                                                        op_idxs[1]))
                          : (iop ? 2 * this->fcidump->v(op_idxs[0], op_idxs[1],
                                                        i, j) -
                                       this->fcidump->v(op_idxs[0], j, i,
                                                        op_idxs[1])
                                 : 2 * this->fcidump->v(op_idxs[0], op_idxs[1],
                                                        j, i) -
                                       this->fcidump->v(op_idxs[0], i, j,
                                                        op_idxs[1]));
        };
        const function<double(uint8_t, const array<uint16_t, 2> &,
                              const array<uint16_t, 2> &)> &q1_ops =
            [this](uint8_t iop, const array<uint16_t, 2> &op_idxs,
                   const array<uint16_t, 2> &ctr_idxs) -> double {
            const uint16_t i = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[0]
                                   : ctr_idxs[1];
            const uint16_t j = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[1]
                                   : ctr_idxs[0];
            return i == j
                       ? (iop ? 0
                              : this->fcidump->v(op_idxs[0], i, j, op_idxs[1]))
                       : (iop ? -this->fcidump->v(op_idxs[0], j, i, op_idxs[1])
                              : this->fcidump->v(op_idxs[0], i, j, op_idxs[1]));
        };
        const function<double(uint8_t, const array<uint16_t, 1> &,
                              const array<uint16_t, 3> &)> &rr_ops =
            [this](uint8_t iop, const array<uint16_t, 1> &op_idxs,
                   const array<uint16_t, 3> &ctr_idxs) -> double {
            const uint16_t l = this->fcidump->n_sites() - 1 - ctr_idxs[0];
            if (iop == 0)
                return sqrt(2) / 4 * this->fcidump->t(op_idxs[0], l);
            const uint16_t k = this->fcidump->n_sites() - 1 - ctr_idxs[1];
            const uint16_t j = this->fcidump->n_sites() - 1 - ctr_idxs[2];
            const uint8_t x =
                (uint8_t)(iop | ((j == k) << 3) | ((k == l) << 4));
            switch (x) {
            case 1 | (1 << 3) | (1 << 4):
            case 1 | (1 << 3):
                return 0;
            case 1 | (1 << 4): // [04] j > k = l | [06] l > j = k
                return this->fcidump->v(op_idxs[0], j, k, l) -
                       0.5 * this->fcidump->v(op_idxs[0], l, k, j);
            case 1: // [07] j > k > l | [12] l > k > j
                return this->fcidump->v(op_idxs[0], j, k, l) -
                       0.5 * this->fcidump->v(op_idxs[0], l, k, j);
            case 2 | (1 << 3) | (1 << 4):
            case 2 | (1 << 3):
                return 0;
            case 2 | (1 << 4): // [06] l > j = k
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, k, j);
            case 2: // [12] l > k > j
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, k, j);
            case 3 | (1 << 3) | (1 << 4):
                return 0;
            case 3 | (1 << 3): // [02] j = l > k
                return -0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 3 | (1 << 4):
                return 0;
            case 3: // [08] j > l > k | [11] l > j > k
                return this->fcidump->v(op_idxs[0], j, l, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 4 | (1 << 3) | (1 << 4):
                return 0;
            case 4 | (1 << 3): // [02] j = l > k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 4 | (1 << 4):
                return 0;
            case 4: // [11] l > j > k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 5 | (1 << 3) | (1 << 4): // [00] k = l = j
                return this->fcidump->v(op_idxs[0], l, j, k);
            case 5 | (1 << 3): // [03] k = l > j | [01] j = k > l
                return this->fcidump->v(op_idxs[0], l, j, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 5 | (1 << 4): // [05] k > j = l
                return -0.5 * this->fcidump->v(op_idxs[0], l, j, k);
            case 5: // [10] k > l > j | [09] k > j > l
                return -0.5 * this->fcidump->v(op_idxs[0], l, j, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 6 | (1 << 3) | (1 << 4):
                return 0;
            case 6 | (1 << 3): // [01] j = k > l
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            case 6 | (1 << 4): // [05] k > j = l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, j, k);
            case 6: // [10] k > l > j | [09] k > j > l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            default:
                assert(false);
                return 0;
            }
        };
        const function<double(uint8_t, const array<uint16_t, 1> &,
                              const array<uint16_t, 3> &)> &lr_ops =
            [this](uint8_t iop, const array<uint16_t, 1> &op_idxs,
                   const array<uint16_t, 3> &ctr_idxs) -> double {
            const uint16_t j = ctr_idxs[0];
            if (iop == 0)
                return sqrt(2) / 4 * this->fcidump->t(op_idxs[0], j);
            const uint16_t k = ctr_idxs[1];
            const uint16_t l = ctr_idxs[2];
            const uint8_t x =
                (uint8_t)(iop | ((j == k) << 3) | ((k == l) << 4));
            switch (x) {
            case 1 | (1 << 3) | (1 << 4):
            case 1 | (1 << 3):
                return 0;
            case 1 | (1 << 4): // [04] j > k = l | [06] l > j = k
                return this->fcidump->v(op_idxs[0], j, k, l) -
                       0.5 * this->fcidump->v(op_idxs[0], l, k, j);
            case 1: // [12] l > k > j | [07] j > k > l
                return this->fcidump->v(op_idxs[0], l, k, j) -
                       0.5 * this->fcidump->v(op_idxs[0], j, k, l);
            case 2 | (1 << 3) | (1 << 4):
            case 2 | (1 << 3):
                return 0;
            case 2 | (1 << 4): // [06] l > j = k
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, k, j);
            case 2: // [07] j > k > l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, k, l);
            case 3 | (1 << 3) | (1 << 4): // [00] k = l = j
                return -0.5 * this->fcidump->v(op_idxs[0], l, j, k);
            case 3 | (1 << 3): // [03] k = l < j | [01] j = k > l
                return this->fcidump->v(op_idxs[0], l, j, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 3 | (1 << 4): // [05] k > j = l
                return -0.5 * this->fcidump->v(op_idxs[0], l, j, k);
            case 3: // [10] k > l > j | [09] k > j > l
                return this->fcidump->v(op_idxs[0], l, j, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 4 | (1 << 3) | (1 << 4): // [00] k = l = j
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, j, k);
            case 4 | (1 << 3): // [01] j = k > l
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            case 4 | (1 << 4): // [05] k > j = l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, j, k);
            case 4: // [09] k > j > l
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            case 5 | (1 << 3) | (1 << 4):
                return 0;
            case 5 | (1 << 3): // [02] j = l > k
                return -0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 5 | (1 << 4):
                return 0;
            case 5: // [11] l > j > k | [08] j > l > k
                return -0.5 * this->fcidump->v(op_idxs[0], k, l, j) -
                       0.5 * this->fcidump->v(op_idxs[0], j, l, k);
            case 6 | (1 << 3) | (1 << 4):
                return 0;
            case 6 | (1 << 3): // [02] j = l > k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 6 | (1 << 4):
                return 0;
            case 6: // [11] l > j > k | [08] j > l > k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j) +
                       0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, l, k);
            default:
                assert(false);
                return 0;
            }
        };
        const function<double(uint8_t, const array<uint16_t, 1> &,
                              const array<uint16_t, 3> &)> &rrd_ops =
            [this](uint8_t iop, const array<uint16_t, 1> &op_idxs,
                   const array<uint16_t, 3> &ctr_idxs) -> double {
            const uint16_t l = this->fcidump->n_sites() - 1 - ctr_idxs[0];
            if (iop == 0)
                return sqrt(2) / 4 * this->fcidump->t(op_idxs[0], l);
            const uint16_t k = this->fcidump->n_sites() - 1 - ctr_idxs[1];
            const uint16_t j = this->fcidump->n_sites() - 1 - ctr_idxs[2];
            const uint8_t x =
                (uint8_t)(iop | ((j == k) << 3) | ((k == l) << 4));
            switch (x) {
            case 1 | (1 << 3) | (1 << 4): // [00] j = k = l
                return -0.5 * this->fcidump->v(op_idxs[0], j, l, k);
            case 1 | (1 << 3): // [02] j = l > k
                return -0.5 * this->fcidump->v(op_idxs[0], j, l, k);
            case 1 | (1 << 4): // [04] j > k = l | [06] l > j = k
                return this->fcidump->v(op_idxs[0], j, l, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 1: // [08] j > l > k | [11] l > j > k
                return this->fcidump->v(op_idxs[0], j, l, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 2 | (1 << 3) | (1 << 4): // [00] j = k = l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, l, k);
            case 2 | (1 << 3): // [02] j = l > k
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, l, k);
            case 2 | (1 << 4): // [06] l > j = k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 2: // [11] l > j > k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 3 | (1 << 3) | (1 << 4):
            case 3 | (1 << 3):
                return 0;
            case 3 | (1 << 4): // [05] k > j = l
                return -0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 3: // [09] k > j > l | [10] k > l > j
                return -0.5 * this->fcidump->v(op_idxs[0], k, j, l) -
                       0.5 * this->fcidump->v(op_idxs[0], l, j, k);
            case 4 | (1 << 3) | (1 << 4):
            case 4 | (1 << 3):
                return 0;
            case 4 | (1 << 4): // [05] k > j = l
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            case 4: // [09] k > j > l | [10] k > l > j
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l) +
                       0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, j, k);
            case 5 | (1 << 3) | (1 << 4):
                return 0;
            case 5 | (1 << 3): // [03] k = l > j | [01] j = k > l
                return this->fcidump->v(op_idxs[0], l, k, j) -
                       0.5 * this->fcidump->v(op_idxs[0], j, k, l);
            case 5 | (1 << 4):
                return 0;
            case 5: // [07] j > k > l | [12] l > k > j
                return this->fcidump->v(op_idxs[0], j, k, l) -
                       0.5 * this->fcidump->v(op_idxs[0], l, k, j);
            case 6 | (1 << 3) | (1 << 4):
                return 0;
            case 6 | (1 << 3): // [01] j = k > l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, k, l);
            case 6 | (1 << 4):
                return 0;
            case 6: // [12] l > k > j
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], l, k, j);
            default:
                assert(false);
                return 0;
            }
        };
        const function<double(uint8_t, const array<uint16_t, 1> &,
                              const array<uint16_t, 3> &)> &lrd_ops =
            [this](uint8_t iop, const array<uint16_t, 1> &op_idxs,
                   const array<uint16_t, 3> &ctr_idxs) -> double {
            const uint16_t j = ctr_idxs[0];
            if (iop == 0)
                return sqrt(2) / 4 * this->fcidump->t(op_idxs[0], j);
            const uint16_t k = ctr_idxs[1];
            const uint16_t l = ctr_idxs[2];
            const uint8_t x =
                (uint8_t)(iop | ((j == k) << 3) | ((k == l) << 4));
            switch (x) {
            case 1 | (1 << 3) | (1 << 4):
            case 1 | (1 << 3):
                return 0;
            case 1 | (1 << 4): // [05] k > j = l
                return -0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 1: // [10] k > l > j | [09] k > j > l
                return this->fcidump->v(op_idxs[0], l, j, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, j, l);
            case 2 | (1 << 3) | (1 << 4):
            case 2 | (1 << 3):
                return 0;
            case 2 | (1 << 4): // [05] k > j = l
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            case 2: // [09] k > j > l
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, j, l);
            case 3 | (1 << 3) | (1 << 4): // [00] j = k = l
                return this->fcidump->v(op_idxs[0], j, l, k);
            case 3 | (1 << 3): // [02] j = l > k
                return -0.5 * this->fcidump->v(op_idxs[0], j, l, k);
            case 3 | (1 << 4): // [04] j > k = l | [06] l > j = k
                return this->fcidump->v(op_idxs[0], j, l, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 3: // [08] j > l > k | [11] l > j > k
                return -0.5 * this->fcidump->v(op_idxs[0], j, l, k) -
                       0.5 * this->fcidump->v(op_idxs[0], k, l, j);
            case 4 | (1 << 3) | (1 << 4):
                return 0;
            case 4 | (1 << 3): // [02] j = l > k
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, l, k);
            case 4 | (1 << 4): // [06] l > j = k
                return -0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 4: // [08] j > l > k | [11] l > j > k
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, l, k) -
                       0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], k, l, j);
            case 5 | (1 << 3) | (1 << 4):
                return 0;
            case 5 | (1 << 3): // [03] k = l > j | [01] j = k > l
                return this->fcidump->v(op_idxs[0], l, k, j) -
                       0.5 * this->fcidump->v(op_idxs[0], j, k, l);
            case 5 | (1 << 4):
                return 0;
            case 5: // [12] l > k > j | [07] j > k > l
                return this->fcidump->v(op_idxs[0], l, k, j) -
                       0.5 * this->fcidump->v(op_idxs[0], j, k, l);
            case 6 | (1 << 3) | (1 << 4):
                return 0;
            case 6 | (1 << 3): // [01] j = k > l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, k, l);
            case 6 | (1 << 4):
                return 0;
            case 6: // [07] j > k > l
                return 0.5 * sqrt(3) * this->fcidump->v(op_idxs[0], j, k, l);
            default:
                assert(false);
                return 0;
            }
        };
        const function<double(uint8_t, const array<uint16_t, 0> &,
                              const array<uint16_t, 4> &)> &rh_ops =
            [this](uint8_t iop, const array<uint16_t, 0> &op_idxs,
                   const array<uint16_t, 4> &ctr_idxs) -> double {
            const uint16_t l = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[0]
                                   : ctr_idxs[0];
            const uint16_t k = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[1]
                                   : ctr_idxs[1];
            if (iop < 2)
                return iop ? (l == k ? 0 : sqrt(2) * this->fcidump->t(k, l))
                           : sqrt(2) * this->fcidump->t(l, k);
            const uint16_t j = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[2]
                                   : ctr_idxs[2];
            const uint16_t i = this->is_right
                                   ? this->fcidump->n_sites() - 1 - ctr_idxs[3]
                                   : ctr_idxs[3];
            const uint8_t x = (uint8_t)(iop | ((i == j) << 5) |
                                        ((j == k) << 6) | ((k == l) << 7));
            switch (x) {
            case 2 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 2 | (1 << 5) |
                (1 << 6): // [03] i = k = l > j | [01] i = j = k > l
                return this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(i, k, j, l);
            case 2 | (1 << 6) | (1 << 7):
                return 0;
            case 2 | (1 << 5) | (1 << 7): // [10] i = k > j = l
                return -0.5 * this->fcidump->v(i, l, j, k);
            case 2 | (1 << 5): // [34] i = k > l > j | [22] i = k > j > l
                return -0.5 * this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(i, k, j, l);
            case 2 | (1 << 6):
                return 0;
            case 2 | (1 << 7): // [16] i > k > j = l | [28] k > i > j = l
                return -0.5 * this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(j, l, i, k);
            case 2: // [54] i > k > l > j | [64] k > i > l > j | [53] i > k > j
                    // > l | [63] k > i > j > l
                return -0.5 * this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(j, l, i, k) -
                       0.5 * this->fcidump->v(i, k, j, l) -
                       0.5 * this->fcidump->v(j, k, i, l);
            case 3 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 3 | (1 << 5) | (1 << 6): // [01] i = j = k > l
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, j, l);
            case 3 | (1 << 6) | (1 << 7):
                return 0;
            case 3 | (1 << 5) | (1 << 7): // [10] i = k > j = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k);
            case 3 | (1 << 5): // [34] i = k > l > j | [22] i = k > j > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(i, k, j, l);
            case 3 | (1 << 6):
                return 0;
            case 3 | (1 << 7): // [16] i > k > j = l | [28] k > i > j = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, i, k);
            case 3: // [54] i > k > l > j | [64] k > i > l > j | [53] i > k > j
                    // > l | [63] k > i > j > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, i, k) -
                       0.5 * sqrt(3) * this->fcidump->v(i, k, j, l) +
                       0.5 * sqrt(3) * this->fcidump->v(j, k, i, l);
            case 4 | (1 << 5) | (1 << 6) | (1 << 7): // [00] i = k = l = j
                return this->fcidump->v(i, l, j, k);
            case 4 | (1 << 5) | (1 << 6):
                return 0;
            case 4 | (1 << 6) |
                (1 << 7): // [08] i > k = l = j | [06] k > i = j = l
                return this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(j, l, i, k);
            case 4 | (1 << 5) | (1 << 7):
            case 4 | (1 << 5):
                return 0;
            case 4 | (1 << 6): // [41] i > k = l > j | [39] i > j = k > l | [46]
                               // k > i = l > j | [45] k > i = j > l
                return this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(i, k, j, l) -
                       0.5 * this->fcidump->v(j, l, i, k) +
                       this->fcidump->v(j, k, i, l);
            case 4 | (1 << 7):
            case 4:
                return 0;
            case 5 | (1 << 5) | (1 << 6) | (1 << 7):
            case 5 | (1 << 5) | (1 << 6):
                return 0;
            case 5 | (1 << 6) | (1 << 7): // [06] k > i = j = l
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, i, k);
            case 5 | (1 << 5) | (1 << 7):
            case 5 | (1 << 5):
                return 0;
            case 5 | (1 << 6): // [39] i > j = k > l | [46] k > i = l > j
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, j, l) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, i, k);
            case 5 | (1 << 7):
            case 5:
                return 0;
            case 6 | (1 << 5) | (1 << 6) | (1 << 7):
            case 6 | (1 << 5) | (1 << 6):
            case 6 | (1 << 6) | (1 << 7):
                return 0;
            case 6 | (1 << 5) | (1 << 7): // [13] j = l > i = k
                return -0.5 * this->fcidump->v(k, j, l, i);
            case 6 | (1 << 5): // [25] j = l > i > k | [37] j = l > k > i
                return -0.5 * this->fcidump->v(k, j, l, i) -
                       0.5 * this->fcidump->v(l, j, k, i);
            case 6 | (1 << 6):
                return 0;
            case 6 | (1 << 7): // [31] l > j > i = k | [19] j > l > i = k
                return -0.5 * this->fcidump->v(k, j, l, i) -
                       0.5 * this->fcidump->v(k, i, l, j);
            case 6: // [71] l > j > i > k | [72] l > j > k > i | [61] j > l > i
                    // > k | [62] j > l > k > i
                return -0.5 * this->fcidump->v(k, j, l, i) -
                       0.5 * this->fcidump->v(l, j, k, i) -
                       0.5 * this->fcidump->v(k, i, l, j) -
                       0.5 * this->fcidump->v(l, i, k, j);
            case 7 | (1 << 5) | (1 << 6) | (1 << 7):
            case 7 | (1 << 5) | (1 << 6):
            case 7 | (1 << 6) | (1 << 7):
                return 0;
            case 7 | (1 << 5) | (1 << 7): // [13] j = l > i = k
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i);
            case 7 | (1 << 5): // [25] j = l > i > k | [37] j = l > k > i
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, k, i);
            case 7 | (1 << 6):
                return 0;
            case 7 | (1 << 7): // [31] l > j > i = k | [19] j > l > i = k
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, l, j);
            case 7: // // [71] l > j > i > k | [72] l > j > k > i | [61] j > l >
                    // i > k | [62] j > l > k > i
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, k, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, l, j) +
                       0.5 * sqrt(3) * this->fcidump->v(l, i, k, j);
            case 8 | (1 << 5) | (1 << 6) | (1 << 7):
            case 8 | (1 << 5) | (1 << 6):
            case 8 | (1 << 6) | (1 << 7):
            case 8 | (1 << 5) | (1 << 7):
                return 0;
            case 8 | (1 << 5): // [23] i = l > j > k | [36] j = k > l > i | [33]
                               // i = j > l > k | [38] k = l > j > i
                return -0.5 * this->fcidump->v(i, k, l, j) -
                       0.5 * this->fcidump->v(l, j, i, k) +
                       this->fcidump->v(i, j, l, k) +
                       this->fcidump->v(l, k, i, j);
            case 8 | (1 << 6):
            case 8 | (1 << 7):
                return 0;
            case 8: // [55] i > l > j > k | [66] k > j > l > i | [52] i > j > l
                    // > k | [68] k > l > j > i
                return -0.5 * this->fcidump->v(i, k, l, j) -
                       0.5 * this->fcidump->v(l, j, i, k) +
                       this->fcidump->v(i, j, l, k) +
                       this->fcidump->v(l, k, i, j);
            case 9 | (1 << 5) | (1 << 6) | (1 << 7):
            case 9 | (1 << 5) | (1 << 6):
            case 9 | (1 << 6) | (1 << 7):
            case 9 | (1 << 5) | (1 << 7):
                return 0;
            case 9 | (1 << 5): // [23] i = l > j > k | [36] j = k > l > i
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, i, k);
            case 9 | (1 << 6):
            case 9 | (1 << 7):
                return 0;
            case 9: // [55] i > l > j > k | [66] k > j > l > i
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, i, k);
            case 10 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 10 | (1 << 5) |
                (1 << 6): // [04] j = k = l > i | [02] i = j = l > k
                return this->fcidump->v(l, k, i, j) -
                       0.5 * this->fcidump->v(i, k, l, j);
            case 10 | (1 << 6) | (1 << 7):
            case 10 | (1 << 5) | (1 << 7):
            case 10 | (1 << 5):
                return 0;
            case 10 | (1 << 6): // [40] i > j = l > k | [47] k > j = l > i
                return -0.5 * this->fcidump->v(i, k, l, j) -
                       0.5 * this->fcidump->v(l, k, i, j);
            case 10 | (1 << 7):
            case 10:
                return 0;
            case 11 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 11 | (1 << 5) | (1 << 6): // [02] i = j = l > k
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j);
            case 11 | (1 << 6) | (1 << 7):
            case 11 | (1 << 5) | (1 << 7):
            case 11 | (1 << 5):
                return 0;
            case 11 | (1 << 6): // [40] i > j = l > k | [47] k > j = l > i
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j) +
                       0.5 * sqrt(3) * this->fcidump->v(l, k, i, j);
            case 11 | (1 << 7):
            case 11:
                return 0;
            case 12 | (1 << 5) | (1 << 6) | (1 << 7):
            case 12 | (1 << 5) | (1 << 6):
            case 12 | (1 << 6) | (1 << 7):
            case 12 | (1 << 5) | (1 << 7):
            case 12 | (1 << 5):
            case 12 | (1 << 6):
            case 12 | (1 << 7):
                return 0;
            case 12: // [69] l > i > j > k | [60] j > k > l > i | [58] j > i > l
                     // > k | [74] l > k > j > i
                return -0.5 * this->fcidump->v(j, k, l, i) -
                       0.5 * this->fcidump->v(l, i, j, k) +
                       this->fcidump->v(j, i, l, k) +
                       this->fcidump->v(l, k, j, i);
            case 13 | (1 << 5) | (1 << 6) | (1 << 7):
            case 13 | (1 << 5) | (1 << 6):
            case 13 | (1 << 6) | (1 << 7):
            case 13 | (1 << 5) | (1 << 7):
            case 13 | (1 << 5):
            case 13 | (1 << 6):
            case 13 | (1 << 7):
                return 0;
            case 13: // [69] l > i > j > k | [60] j > k > l > i
                return 0.5 * sqrt(3) * this->fcidump->v(j, k, l, i) +
                       0.5 * sqrt(3) * this->fcidump->v(l, i, j, k);
            case 14 | (1 << 5) | (1 << 6) | (1 << 7):
            case 14 | (1 << 5) | (1 << 6):
            case 14 | (1 << 6) | (1 << 7):
            case 14 | (1 << 5) | (1 << 7):
            case 14 | (1 << 5):
                return 0;
            case 14 | (1 << 6): // [50] l > j = k > i | [44] j > k = l > i |
                                // [43] j > i = l > k | [48] l > i = j > k
                return -0.5 * this->fcidump->v(l, k, j, i) +
                       this->fcidump->v(l, i, j, k) -
                       0.5 * this->fcidump->v(j, i, l, k) +
                       this->fcidump->v(j, k, l, i);
            case 14 | (1 << 7):
            case 14:
                return 0;
            case 15 | (1 << 5) | (1 << 6) | (1 << 7):
            case 15 | (1 << 5) | (1 << 6):
            case 15 | (1 << 6) | (1 << 7):
            case 15 | (1 << 5) | (1 << 7):
            case 15 | (1 << 5):
                return 0;
            case 15 | (1 << 6): // [50] l > j = k > i | [43] j > i = l > k
                return 0.5 * sqrt(3) * this->fcidump->v(l, k, j, i) +
                       0.5 * sqrt(3) * this->fcidump->v(j, i, l, k);
            case 15 | (1 << 7):
            case 15:
                return 0;
            case 16 | (1 << 5) | (1 << 6) | (1 << 7):
            case 16 | (1 << 5) | (1 << 6):
            case 16 | (1 << 6) | (1 << 7):
                return 0;
            case 16 | (1 << 5) |
                (1 << 7): // [11] i = l > j = k | [12] j = k > i = l | [09] i =
                          // j > k = l | [14] k = l > i = j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 16 | (1 << 5): // [35] i = l > k > j | [24] j = k > i > l |
                                // [21] i = j > k > l | [26] k = l > i > j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 16 | (1 << 6):
                return 0;
            case 16 | (1 << 7): // [17] i > l > j = k | [30] k > j > i = l |
                                // [15] i > j > k = l | [20] k > l > i = j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 16: // [56] i > l > k > j | [65] k > j > i > l | [51] i > j > k
                     // > l | [67] k > l > i > j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 17 | (1 << 5) | (1 << 6) | (1 << 7):
            case 17 | (1 << 5) | (1 << 6):
            case 17 | (1 << 6) | (1 << 7):
                return 0;
            case 17 | (1 << 5) |
                (1 << 7): // [11] i = l > j = k | [12] j = k > i = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 17 | (1 << 5): // [35] i = l > k > j | [24] j = k > i > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 17 | (1 << 6):
                return 0;
            case 17 | (1 << 7): // [17] i > l > j = k | [30] k > j > i = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 17: // [56] i > l > k > j | [65] k > j > i > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 18 | (1 << 5) | (1 << 6) | (1 << 7):
            case 18 | (1 << 5) | (1 << 6):
                return 0;
            case 18 | (1 << 6) |
                (1 << 7): // [05] l > i = j = k | [07] j > i = k = l
                return -0.5 * this->fcidump->v(j, l, k, i) +
                       this->fcidump->v(j, i, k, l);
            case 18 | (1 << 5) | (1 << 7):
            case 18 | (1 << 5):
            case 18 | (1 << 6):
                return 0;
            case 18 | (1 << 7): // [29] l > i > j = k | [18] j > k > i = l |
                                // [27] j > i > k = l | [32] l > k > i = j
                return -0.5 * this->fcidump->v(j, l, k, i) -
                       0.5 * this->fcidump->v(k, i, j, l) +
                       this->fcidump->v(j, i, k, l) +
                       this->fcidump->v(k, l, j, i);
            case 18: // [70] l > i > k > j | [59] j > k > i > l | [57] j > i > k
                     // > l | [73] l > k > i > j
                return -0.5 * this->fcidump->v(j, l, k, i) -
                       0.5 * this->fcidump->v(k, i, j, l) +
                       this->fcidump->v(j, i, k, l) +
                       this->fcidump->v(k, l, j, i);
            case 19 | (1 << 5) | (1 << 6) | (1 << 7):
            case 19 | (1 << 5) | (1 << 6):
                return 0;
            case 19 | (1 << 6) | (1 << 7): // [05] l > i = j = k
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, k, i);
            case 19 | (1 << 5) | (1 << 7):
            case 19 | (1 << 5):
            case 19 | (1 << 6):
                return 0;
            case 19 | (1 << 7): // [29] l > i > j = k | [18] j > k > i = l
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, k, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, j, l);
            case 19: // [70] l > i > k > j | [59] j > k > i > l
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, k, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, j, l);
            case 20 | (1 << 5) | (1 << 6) | (1 << 7):
            case 20 | (1 << 5) | (1 << 6):
            case 20 | (1 << 6) | (1 << 7):
            case 20 | (1 << 5) | (1 << 7):
            case 20 | (1 << 5):
                return 0;
            case 20 | (1 << 6): // [42] j > i = k > l | [49] l > i = k > j
                return -0.5 * this->fcidump->v(j, i, k, l) -
                       0.5 * this->fcidump->v(j, l, k, i);
            case 20 | (1 << 7):
            case 20:
                return 0;
            case 21 | (1 << 5) | (1 << 6) | (1 << 7):
            case 21 | (1 << 5) | (1 << 6):
            case 21 | (1 << 6) | (1 << 7):
            case 21 | (1 << 5) | (1 << 7):
            case 21 | (1 << 5):
                return 0;
            case 21 | (1 << 6): // [42] j > i = k > l | [49] l > i = k > j
                return 0.5 * sqrt(3) * this->fcidump->v(j, i, k, l) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, k, i);
            case 21 | (1 << 7):
            case 21:
                return 0;
            default:
                assert(false);
                return 0;
            }
        };
        const function<double(uint8_t, const array<uint16_t, 0> &,
                              const array<uint16_t, 4> &)> &lh_ops =
            [this](uint8_t iop, const array<uint16_t, 0> &op_idxs,
                   const array<uint16_t, 4> &ctr_idxs) -> double {
            const uint16_t i = ctr_idxs[0];
            const uint16_t j = ctr_idxs[1];
            if (iop < 2)
                return iop ? (i == j ? 0 : sqrt(2) * this->fcidump->t(i, j))
                           : sqrt(2) * this->fcidump->t(j, i);
            const uint16_t k = ctr_idxs[2];
            const uint16_t l = ctr_idxs[3];
            const uint8_t x = (uint8_t)(iop | ((i == j) << 5) |
                                        ((j == k) << 6) | ((k == l) << 7));
            switch (x) {
            case 2 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 2 | (1 << 5) | (1 << 6):
                return 0;
            case 2 | (1 << 6) |
                (1 << 7): // [08] i > k = l = j | [06] k > i = j = l
                return this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(j, l, i, k);
            case 2 | (1 << 5) | (1 << 7): // [10] i = k > j = l
                return -0.5 * this->fcidump->v(i, l, j, k);
            case 2 | (1 << 5): // [34] i = k > l > j | [22] i = k > j > l
                return -0.5 * this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(i, k, j, l);
            case 2 | (1 << 6):
                return 0;
            case 2 | (1 << 7): // [16] i > k > j = l | [28] k > i > j = l
                return -0.5 * this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(j, l, i, k);
            case 2: // [54] i > k > l > j | [64] k > i > l > j | [53] i > k > j
                    // > l | [63] k > i > j > l
                return -0.5 * this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(j, l, i, k) -
                       0.5 * this->fcidump->v(i, k, j, l) -
                       0.5 * this->fcidump->v(j, k, i, l);
            case 3 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 3 | (1 << 5) | (1 << 6):
                return 0;
            case 3 | (1 << 6) | (1 << 7): // [06] k > i = j = l
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, i, k);
            case 3 | (1 << 5) | (1 << 7): // [10] i = k > j = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k);
            case 3 | (1 << 5): // [34] i = k > l > j | [22] i = k > j > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(i, k, j, l);
            case 3 | (1 << 6):
                return 0;
            case 3 | (1 << 7): // [16] i > k > j = l | [28] k > i > j = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, i, k);
            case 3: // [54] i > k > l > j | [64] k > i > l > j | [53] i > k > j
                    // > l | [63] k > i > j > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, j, k) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, i, k) -
                       0.5 * sqrt(3) * this->fcidump->v(i, k, j, l) +
                       0.5 * sqrt(3) * this->fcidump->v(j, k, i, l);
            case 4 | (1 << 5) | (1 << 6) | (1 << 7): // [00] i = k = l = j
                return this->fcidump->v(i, l, j, k);
            case 4 | (1 << 5) |
                (1 << 6): // [03] i = k = l > j | [01] i = j = k > l
                return this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(i, k, j, l);
            case 4 | (1 << 6) | (1 << 7):
                return 0;
            case 4 | (1 << 5) | (1 << 7):
            case 4 | (1 << 5):
                return 0;
            case 4 | (1 << 6): // [41] i > k = l > j | [39] i > j = k > l | [46]
                               // k > i = l > j | [45] k > i = j > l
                return this->fcidump->v(i, l, j, k) -
                       0.5 * this->fcidump->v(i, k, j, l) -
                       0.5 * this->fcidump->v(j, l, i, k) +
                       this->fcidump->v(j, k, i, l);
            case 4 | (1 << 7):
            case 4:
                return 0;
            case 5 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 5 | (1 << 5) | (1 << 6): // [01] i = j = k > l
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, j, l);
            case 5 | (1 << 6) | (1 << 7):
                return 0;
            case 5 | (1 << 5) | (1 << 7):
            case 5 | (1 << 5):
                return 0;
            case 5 | (1 << 6): // [39] i > j = k > l | [46] k > i = l > j
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, j, l) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, i, k);
            case 5 | (1 << 7):
            case 5:
                return 0;
            case 6 | (1 << 5) | (1 << 6) | (1 << 7):
            case 6 | (1 << 5) | (1 << 6):
            case 6 | (1 << 6) | (1 << 7):
                return 0;
            case 6 | (1 << 5) | (1 << 7): // [13] j = l > i = k
                return -0.5 * this->fcidump->v(k, j, l, i);
            case 6 | (1 << 5): // [25] j = l > i > k | [37] j = l > k > i
                return -0.5 * this->fcidump->v(k, j, l, i) -
                       0.5 * this->fcidump->v(l, j, k, i);
            case 6 | (1 << 6):
                return 0;
            case 6 | (1 << 7): // [31] l > j > i = k | [19] j > l > i = k
                return -0.5 * this->fcidump->v(k, j, l, i) -
                       0.5 * this->fcidump->v(k, i, l, j);
            case 6: // [71] l > j > i > k | [72] l > j > k > i | [61] j > l > i
                    // > k | [62] j > l > k > i
                return -0.5 * this->fcidump->v(k, j, l, i) -
                       0.5 * this->fcidump->v(l, j, k, i) -
                       0.5 * this->fcidump->v(k, i, l, j) -
                       0.5 * this->fcidump->v(l, i, k, j);
            case 7 | (1 << 5) | (1 << 6) | (1 << 7):
            case 7 | (1 << 5) | (1 << 6):
            case 7 | (1 << 6) | (1 << 7):
                return 0;
            case 7 | (1 << 5) | (1 << 7): // [13] j = l > i = k
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i);
            case 7 | (1 << 5): // [25] j = l > i > k | [37] j = l > k > i
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, k, i);
            case 7 | (1 << 6):
                return 0;
            case 7 | (1 << 7): // [31] l > j > i = k | [19] j > l > i = k
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, l, j);
            case 7: // // [71] l > j > i > k | [72] l > j > k > i | [61] j > l >
                    // i > k | [62] j > l > k > i
                return 0.5 * sqrt(3) * this->fcidump->v(k, j, l, i) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, k, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, l, j) +
                       0.5 * sqrt(3) * this->fcidump->v(l, i, k, j);
            case 8 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 8 | (1 << 5) |
                (1 << 6): // [04] j = k = l > i | [02] i = j = l > k
                return this->fcidump->v(l, k, i, j) -
                       0.5 * this->fcidump->v(i, k, l, j);
            case 8 | (1 << 6) | (1 << 7):
            case 8 | (1 << 5) | (1 << 7):
                return 0;
            case 8 | (1 << 5): // [23] i = l > j > k | [36] j = k > l > i | [33]
                               // i = j > l > k | [38] k = l > j > i
                return -0.5 * this->fcidump->v(i, k, l, j) -
                       0.5 * this->fcidump->v(l, j, i, k) +
                       this->fcidump->v(i, j, l, k) +
                       this->fcidump->v(l, k, i, j);
            case 8 | (1 << 6):
            case 8 | (1 << 7):
                return 0;
            case 8: // [55] i > l > j > k | [66] k > j > l > i | [52] i > j > l
                    // > k | [68] k > l > j > i
                return -0.5 * this->fcidump->v(i, k, l, j) -
                       0.5 * this->fcidump->v(l, j, i, k) +
                       this->fcidump->v(i, j, l, k) +
                       this->fcidump->v(l, k, i, j);
            case 9 | (1 << 5) | (1 << 6) | (1 << 7):
                return 0;
            case 9 | (1 << 5) | (1 << 6): // [02] i = j = l > k
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j);
            case 9 | (1 << 6) | (1 << 7):
            case 9 | (1 << 5) | (1 << 7):
                return 0;
            case 9 | (1 << 5): // [23] i = l > j > k | [36] j = k > l > i
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, i, k);
            case 9 | (1 << 6):
            case 9 | (1 << 7):
                return 0;
            case 9: // [55] i > l > j > k | [66] k > j > l > i
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j) -
                       0.5 * sqrt(3) * this->fcidump->v(l, j, i, k);
            case 10 | (1 << 5) | (1 << 6) | (1 << 7):
            case 10 | (1 << 5) | (1 << 6):
            case 10 | (1 << 6) | (1 << 7):
            case 10 | (1 << 5) | (1 << 7):
            case 10 | (1 << 5):
                return 0;
            case 10 | (1 << 6): // [40] i > j = l > k | [47] k > j = l > i
                return -0.5 * this->fcidump->v(i, k, l, j) -
                       0.5 * this->fcidump->v(l, k, i, j);
            case 10 | (1 << 7):
            case 10:
                return 0;
            case 11 | (1 << 5) | (1 << 6) | (1 << 7):
            case 11 | (1 << 5) | (1 << 6):
            case 11 | (1 << 6) | (1 << 7):
            case 11 | (1 << 5) | (1 << 7):
            case 11 | (1 << 5):
                return 0;
            case 11 | (1 << 6): // [40] i > j = l > k | [47] k > j = l > i
                return -0.5 * sqrt(3) * this->fcidump->v(i, k, l, j) +
                       0.5 * sqrt(3) * this->fcidump->v(l, k, i, j);
            case 11 | (1 << 7):
            case 11:
                return 0;
            case 12 | (1 << 5) | (1 << 6) | (1 << 7):
            case 12 | (1 << 5) | (1 << 6):
            case 12 | (1 << 6) | (1 << 7):
            case 12 | (1 << 5) | (1 << 7):
            case 12 | (1 << 5):
            case 12 | (1 << 6):
            case 12 | (1 << 7):
                return 0;
            case 12: // [69] l > i > j > k | [60] j > k > l > i | [58] j > i > l
                     // > k | [74] l > k > j > i
                return -0.5 * this->fcidump->v(j, k, l, i) -
                       0.5 * this->fcidump->v(l, i, j, k) +
                       this->fcidump->v(j, i, l, k) +
                       this->fcidump->v(l, k, j, i);
            case 13 | (1 << 5) | (1 << 6) | (1 << 7):
            case 13 | (1 << 5) | (1 << 6):
            case 13 | (1 << 6) | (1 << 7):
            case 13 | (1 << 5) | (1 << 7):
            case 13 | (1 << 5):
            case 13 | (1 << 6):
            case 13 | (1 << 7):
                return 0;
            case 13: // [69] l > i > j > k | [60] j > k > l > i
                return 0.5 * sqrt(3) * this->fcidump->v(j, k, l, i) +
                       0.5 * sqrt(3) * this->fcidump->v(l, i, j, k);
            case 14 | (1 << 5) | (1 << 6) | (1 << 7):
            case 14 | (1 << 5) | (1 << 6):
            case 14 | (1 << 6) | (1 << 7):
            case 14 | (1 << 5) | (1 << 7):
            case 14 | (1 << 5):
                return 0;
            case 14 | (1 << 6): // [50] l > j = k > i | [44] j > k = l > i |
                                // [43] j > i = l > k | [48] l > i = j > k
                return -0.5 * this->fcidump->v(l, k, j, i) +
                       this->fcidump->v(l, i, j, k) -
                       0.5 * this->fcidump->v(j, i, l, k) +
                       this->fcidump->v(j, k, l, i);
            case 14 | (1 << 7):
            case 14:
                return 0;
            case 15 | (1 << 5) | (1 << 6) | (1 << 7):
            case 15 | (1 << 5) | (1 << 6):
            case 15 | (1 << 6) | (1 << 7):
            case 15 | (1 << 5) | (1 << 7):
            case 15 | (1 << 5):
                return 0;
            case 15 | (1 << 6): // [50] l > j = k > i | [43] j > i = l > k
                return 0.5 * sqrt(3) * this->fcidump->v(l, k, j, i) +
                       0.5 * sqrt(3) * this->fcidump->v(j, i, l, k);
            case 15 | (1 << 7):
            case 15:
                return 0;
            case 16 | (1 << 5) | (1 << 6) | (1 << 7):
            case 16 | (1 << 5) | (1 << 6):
            case 16 | (1 << 6) | (1 << 7):
                return 0;
            case 16 | (1 << 5) |
                (1 << 7): // [11] i = l > j = k | [12] j = k > i = l | [09] i =
                          // j > k = l | [14] k = l > i = j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 16 | (1 << 5): // [35] i = l > k > j | [24] j = k > i > l |
                                // [21] i = j > k > l | [26] k = l > i > j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 16 | (1 << 6):
                return 0;
            case 16 | (1 << 7): // [17] i > l > j = k | [30] k > j > i = l |
                                // [15] i > j > k = l | [20] k > l > i = j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 16: // [56] i > l > k > j | [65] k > j > i > l | [51] i > j > k
                     // > l | [67] k > l > i > j
                return -0.5 * this->fcidump->v(i, l, k, j) -
                       0.5 * this->fcidump->v(k, j, i, l) +
                       this->fcidump->v(i, j, k, l) +
                       this->fcidump->v(k, l, i, j);
            case 17 | (1 << 5) | (1 << 6) | (1 << 7):
            case 17 | (1 << 5) | (1 << 6):
            case 17 | (1 << 6) | (1 << 7):
                return 0;
            case 17 | (1 << 5) |
                (1 << 7): // [11] i = l > j = k | [12] j = k > i = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 17 | (1 << 5): // [35] i = l > k > j | [24] j = k > i > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 17 | (1 << 6):
                return 0;
            case 17 | (1 << 7): // [17] i > l > j = k | [30] k > j > i = l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 17: // [56] i > l > k > j | [65] k > j > i > l
                return 0.5 * sqrt(3) * this->fcidump->v(i, l, k, j) +
                       0.5 * sqrt(3) * this->fcidump->v(k, j, i, l);
            case 18 | (1 << 5) | (1 << 6) | (1 << 7):
            case 18 | (1 << 5) | (1 << 6):
            case 18 | (1 << 6) | (1 << 7):
            case 18 | (1 << 5) | (1 << 7):
            case 18 | (1 << 5):
            case 18 | (1 << 6):
                return 0;
            case 18 | (1 << 7): // [29] l > i > j = k | [18] j > k > i = l |
                                // [27] j > i > k = l | [32] l > k > i = j
                return -0.5 * this->fcidump->v(j, l, k, i) -
                       0.5 * this->fcidump->v(k, i, j, l) +
                       this->fcidump->v(j, i, k, l) +
                       this->fcidump->v(k, l, j, i);
            case 18: // [70] l > i > k > j | [59] j > k > i > l | [57] j > i > k
                     // > l | [73] l > k > i > j
                return -0.5 * this->fcidump->v(j, l, k, i) -
                       0.5 * this->fcidump->v(k, i, j, l) +
                       this->fcidump->v(j, i, k, l) +
                       this->fcidump->v(k, l, j, i);
            case 19 | (1 << 5) | (1 << 6) | (1 << 7):
            case 19 | (1 << 5) | (1 << 6):
            case 19 | (1 << 6) | (1 << 7):
            case 19 | (1 << 5) | (1 << 7):
            case 19 | (1 << 5):
            case 19 | (1 << 6):
                return 0;
            case 19 | (1 << 7): // [29] l > i > j = k | [18] j > k > i = l
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, k, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, j, l);
            case 19: // [70] l > i > k > j | [59] j > k > i > l
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, k, i) -
                       0.5 * sqrt(3) * this->fcidump->v(k, i, j, l);
            case 20 | (1 << 5) | (1 << 6) | (1 << 7):
            case 20 | (1 << 5) | (1 << 6):
                return 0;
            case 20 | (1 << 6) |
                (1 << 7): // [05] l > i = j = k | [07] j > i = k = l
                return -0.5 * this->fcidump->v(j, l, k, i) +
                       this->fcidump->v(j, i, k, l);
            case 20 | (1 << 5) | (1 << 7):
            case 20 | (1 << 5):
                return 0;
            case 20 | (1 << 6): // [42] j > i = k > l | [49] l > i = k > j
                return -0.5 * this->fcidump->v(j, i, k, l) -
                       0.5 * this->fcidump->v(j, l, k, i);
            case 20 | (1 << 7):
            case 20:
                return 0;
            case 21 | (1 << 5) | (1 << 6) | (1 << 7):
            case 21 | (1 << 5) | (1 << 6):
                return 0;
            case 21 | (1 << 6) | (1 << 7): // [05] l > i = j = k
                return -0.5 * sqrt(3) * this->fcidump->v(j, l, k, i);
            case 21 | (1 << 5) | (1 << 7):
            case 21 | (1 << 5):
                return 0;
            case 21 | (1 << 6): // [42] j > i = k > l | [49] l > i = k > j
                return 0.5 * sqrt(3) * this->fcidump->v(j, i, k, l) -
                       0.5 * sqrt(3) * this->fcidump->v(j, l, k, i);
            case 21 | (1 << 7):
            case 21:
                return 0;
            default:
                assert(false);
                return 0;
            }
        };
        if (iprint >= 2)
            cout << "build p .. " << endl;
        for (uint8_t s = 0; s < 2; s++)
            for (auto &mats : p_mats[s])
                build_complementary_site_ops<decltype(p0_ops), 2, 2>(
                    {s ? ad1_ops : ad0_ops}, s ? p1_ops : p0_ops, mats.second);
        if (iprint >= 2)
            cout << "build pd .. " << endl;
        for (uint8_t s = 0; s < 2; s++)
            for (auto &mats : pd_mats[s])
                build_complementary_site_ops<decltype(p0_ops), 2, 2>(
                    {s ? a1_ops : a0_ops}, s ? p1_ops : p0_ops, mats.second,
                    s ? -1.0 : 1.0);
        if (iprint >= 2)
            cout << "build q .. " << endl;
        for (uint8_t s = 0; s < 2; s++)
            for (auto &mats : q_mats[s])
                build_complementary_site_ops<decltype(q0_ops), 2, 2>(
                    is_right ? vector<uint8_t>{s ? b1_ops : b0_ops,
                                               s ? bd1_ops : bd0_ops}
                             : vector<uint8_t>{s ? bd1_ops : bd0_ops,
                                               s ? b1_ops : b0_ops},
                    s ? q1_ops : q0_ops, mats.second);
        if (iprint >= 2)
            cout << "build r .. " << endl;
        for (auto &mats : r_mats)
            build_complementary_site_ops<decltype(lr_ops), 1, 3>(
                {dxx_ops, dcd0_ops, dcd1_ops, cdd0_ops, cdd1_ops, ddc0_ops,
                 ddc1_ops},
                is_right ? rr_ops : lr_ops, mats.second);
        if (iprint >= 2)
            cout << "build rd .. " << endl;
        for (auto &mats : rd_mats)
            build_complementary_site_ops<decltype(lrd_ops), 1, 3>(
                {cxx_ops, dcc0_ops, dcc1_ops, ccd0_ops, ccd1_ops, cdc0_ops,
                 cdc1_ops},
                is_right ? rrd_ops : lrd_ops, mats.second);
        if (iprint >= 2)
            cout << "build h .. " << endl;
        if (is_right)
            build_complementary_site_ops<decltype(rh_ops), 0, 4>(
                {dcxx_ops,  cdxx_ops,  ddcc0_ops, ddcc1_ops, ddcc2_ops,
                 ddcc3_ops, ccdd0_ops, ccdd1_ops, cddc0_ops, cddc1_ops,
                 cddc2_ops, cddc3_ops, cdcd0_ops, cdcd1_ops, cdcd2_ops,
                 cdcd3_ops, dcdc0_ops, dcdc1_ops, dccd0_ops, dccd1_ops,
                 dccd2_ops, dccd3_ops},
                rh_ops, h_mats);
        else
            build_complementary_site_ops<decltype(lh_ops), 0, 4>(
                {cdxx_ops,  dcxx_ops,  ccdd0_ops, ccdd1_ops, ccdd2_ops,
                 ccdd3_ops, ddcc0_ops, ddcc1_ops, cddc0_ops, cddc1_ops,
                 cddc2_ops, cddc3_ops, dcdc0_ops, dcdc1_ops, dcdc2_ops,
                 dcdc3_ops, cdcd0_ops, cdcd1_ops, dccd0_ops, dccd1_ops,
                 dccd2_ops, dccd3_ops},
                lh_ops, h_mats);
    }
};

} // namespace block2
