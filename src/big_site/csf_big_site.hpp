
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

#include "../core/cg.hpp"
#include "../core/integral.hpp"
#include "../core/prime.hpp"
#include "../core/state_info.hpp"
#include "big_site.hpp"
#include <array>
#include <cstdint>
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
                assert(n_unpaired_shapes[ij].first != 0 &&
                       n_unpaired_shapes[ij].second != 0);
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
            cout << i << " " << qs[i] << " " << basis->n_states[i] << " "
                 << csf_offsets[qs_idxs[i + 1]] << " "
                 << csf_offsets[qs_idxs[i]] << endl;
            assert(basis->n_states[i] ==
                   csf_offsets[qs_idxs[i + 1]] - csf_offsets[qs_idxs[i]]);
        }
        cg = make_shared<CG<S>>((n_max_unpaired + 1) * 2);
        cg->initialize();
    }
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
    // n_elec does not include orbital k
    void find_configs(bool occupied, int n_orbs, int n_elec, int k,
                      vector<pair<LL, int>> &results,
                      vector<LL> &insert_idxs) const {

        insert_idxs.resize(max(n_elec + 2, 2));
        insert_idxs[0] = 0; // all < k
        insert_idxs[1] = 1;
        vector<pair<LL, int>> partials[2];
        partials[1].clear();
        partials[1].push_back(make_pair(n_elec == 0 && occupied ? k : 0, 0));
        if (n_elec < 0 || n_elec > n_orbs - 1) {
            insert_idxs[1] = 0;
            partials[1].clear();
            results = partials[1];
            return;
        }
        for (int i = 0, ii = 0; i < n_elec; i++) {
            partials[i & 1].clear();
            for (LL ip = 0; ip < (LL)partials[!(i & 1)].size(); ip++) {
                auto &p = partials[!(i & 1)][ip];
                while (ii < i + 1 && insert_idxs[ii] <= ip)
                    insert_idxs[ii++] = (LL)partials[i & 1].size();
                partials[i & 1].reserve(partials[i & 1].size() + n_orbs -
                                        p.second);
                LL cur = p.first;
                for (int l = p.second; l < n_orbs; l++) {
                    if (l != k)
                        partials[i & 1].push_back(make_pair(cur, l + 1));
                    else if (i == 0)
                        insert_idxs[0] = (LL)partials[i & 1].size();
                    if (!occupied)
                        cur += combinatorics->combination(n_orbs - l - 1,
                                                          n_elec - i - 1);
                    else if (l != k)
                        cur += combinatorics->combination(
                            n_orbs - l - 1, n_elec + (l < k) - i - 1);
                }
            }
            assert(ii == i + 1);
            insert_idxs[ii++] = (LL)partials[i & 1].size();
            memmove(&insert_idxs[1], &insert_idxs[0], ii * sizeof(LL));
            insert_idxs[0] = 0;
        }
        results = partials[!(n_elec & 1)];
    }
    void apply_creation_op(int i_qs, int i_unpaired, int k,
                           vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat) {
        // if position k is zero
        const int n_unpaired = this->n_unpaired[i_unpaired];
        const int n_double = (qs[i_qs].n() - n_unpaired) >> 1;
        const int n_empty = n_orbs - n_double - n_unpaired;
        vector<LL> d_idxs;
        vector<pair<LL, int>> d_configs;
        // ket / bra, k not doubly occupied
        find_configs(false, n_orbs, n_double, k, d_configs, d_idxs);
        // ket, k not singly occupied
        vector<vector<LL>> u_idxs(n_double + 1);
        vector<vector<pair<LL, int>>> ku_configs(n_double + 1);
        for (int j = 0; j <= n_double; j++)
            find_configs(false, n_orbs - n_double, n_unpaired,
                         k - (n_double - j), ku_configs[j], u_idxs[j]);
        // bra, k singly occupied
        vector<vector<LL>> bu_idxs(n_double + 1);
        vector<vector<pair<LL, int>>> bu_configs(n_double + 1);
        for (int j = 0; j <= n_double; j++)
            find_configs(true, n_orbs - n_double, n_unpaired,
                         k - (n_double - j), bu_configs[j], bu_idxs[j]);
        for (int j = 0; j <= n_double; j++)
            for (int i = 0; i <= n_unpaired + 1; i++)
                assert(u_idxs[j][i] == bu_idxs[j][i]);
        // ket, k singly occupied
        vector<vector<LL>> u2_idxs(n_double + 1);
        vector<vector<pair<LL, int>>> ku2_configs(n_double + 1);
        for (int j = 0; j <= n_double; j++)
            find_configs(true, n_orbs - n_double, n_unpaired - 1,
                         k - (n_double - j), ku2_configs[j], u2_idxs[j]);
        // bra, any singly occupied
        vector<vector<pair<LL, int>>> bu2_configs(n_double + 1);
        for (int j = 0; j <= n_double; j++) {
            bu2_configs[j].resize(u2_idxs[j].back());
            for (LL i = 0; i < u2_idxs[j].back(); i++)
                bu2_configs[j][i].first = i;
        }
        // bra, k doubly occupied
        vector<LL> d2_idxs;
        vector<pair<LL, int>> d2_configs;
        find_configs(true, n_orbs, n_double, k, d2_configs, d2_idxs);
        for (int j = 0; j <= n_double + 1; j++)
            assert(d2_idxs[j] == d_idxs[j]);
        S dq = qs[i_qs] + S(1, 1, 0);
        for (int idq = 0; idq < dq.count(); idq++) {
            int i_dqs = basis->find_state(dq[idq]);
            int d_twos = dq[idq].twos(), k_twos = qs[i_qs].twos();
            if (i_dqs == -1)
                continue;
            // |0> -> |+>
            int i_dunpaired = qs_idxs[i_dqs] + ((n_unpaired + 1 - d_twos) >> 1);
            if (i_dunpaired < qs_idxs[i_dqs] ||
                i_dunpaired >= qs_idxs[i_dqs + 1])
                continue;
            assert(this->n_unpaired[i_dunpaired] == n_unpaired + 1);
            vector<vector<double>> rr(n_unpaired + 1);
            for (int i = 0; i <= n_unpaired; i++)
                rr[i] = csf_apply_creation_op(n_unpaired + 1, d_twos,
                                              n_unpaired, k_twos, i, 1.0);
            const LL i_csf_ket = csf_sub_idxs[csf_idxs[n_unpaired] + k_twos];
            const LL j_csf_ket =
                csf_sub_idxs[csf_idxs[n_unpaired] + k_twos + 1];
            const int cl_ket = max((n_unpaired >> 3) + !!(n_unpaired & 7), 1);
            const int n_ket = (int)((j_csf_ket - i_csf_ket) / cl_ket);
            LL i_csf_bra = csf_sub_idxs[csf_idxs[n_unpaired + 1] + d_twos];
            LL j_csf_bra = csf_sub_idxs[csf_idxs[n_unpaired + 1] + d_twos + 1];
            int cl_bra =
                max(((n_unpaired + 1) >> 3) + !!((n_unpaired + 1) & 7), 1);
            int n_bra = (int)((j_csf_bra - i_csf_bra) / cl_bra);
            for (int j = 0; j <= n_double; j++) {
                for (LL kd = d_idxs[j]; kd < d_idxs[j + 1]; kd++) {
                    const LL idrow = csf_offsets[i_dunpaired] +
                                     n_bra * d_configs[kd].first *
                                         n_unpaired_shapes[i_dunpaired].second;
                    const LL idcol = csf_offsets[i_unpaired] +
                                     n_ket * d_configs[kd].first *
                                         n_unpaired_shapes[i_unpaired].second;
                    for (int i = 0; i <= n_unpaired; i++) {
                        for (LL ku = u_idxs[j][i]; ku < u_idxs[j][i + 1];
                             ku++) {
                            LL irow = bu_configs[j][ku].first * n_bra + idrow;
                            LL icol = ku_configs[j][ku].first * n_ket + idcol;
                            for (int kb = 0; kb < n_bra; kb++)
                                for (int kk = 0; kk < n_ket; kk++) {
                                    cout << (*this)[irow + kb] << " "
                                         << (*this)[icol + kk] << " = "
                                         << rr[n_unpaired - i][kb * n_ket + kk]
                                         << endl;
                                    mat.emplace_back(
                                        make_pair(irow + kb, icol + kk),
                                        rr[n_unpaired - i][kb * n_ket + kk]);
                                }
                        }
                    }
                }
            }
            if (n_unpaired == 0)
                continue;
            // |+> -> |2>
            i_dunpaired = qs_idxs[i_dqs] + ((n_unpaired - 1 - d_twos) >> 1);
            if (i_dunpaired < qs_idxs[i_dqs] ||
                i_dunpaired >= qs_idxs[i_dqs + 1])
                continue;
            assert(this->n_unpaired[i_dunpaired] == n_unpaired - 1);
            for (int i = 0; i < n_unpaired; i++)
                rr[i] = csf_apply_creation_op(n_unpaired - 1, d_twos,
                                              n_unpaired, k_twos, i, -sqrt(2));
            i_csf_bra = csf_sub_idxs[csf_idxs[n_unpaired - 1] + d_twos];
            j_csf_bra = csf_sub_idxs[csf_idxs[n_unpaired - 1] + d_twos + 1];
            cl_bra = max(((n_unpaired - 1) >> 3) + !!((n_unpaired - 1) & 7), 1);
            n_bra = (int)((j_csf_bra - i_csf_bra) / cl_bra);
            for (int j = 0; j <= n_double; j++) {
                for (LL kd = d_idxs[j]; kd < d_idxs[j + 1]; kd++) {
                    const LL idrow = csf_offsets[i_dunpaired] +
                                     n_bra * d2_configs[kd].first *
                                         n_unpaired_shapes[i_dunpaired].second;
                    const LL idcol = csf_offsets[i_unpaired] +
                                     n_ket * d_configs[kd].first *
                                         n_unpaired_shapes[i_unpaired].second;
                    for (int i = 0; i <= n_unpaired - 1; i++) {
                        for (LL ku = u2_idxs[j][i]; ku < u2_idxs[j][i + 1];
                             ku++) {
                            LL irow = bu2_configs[j][ku].first * n_bra + idrow;
                            LL icol = ku2_configs[j][ku].first * n_ket + idcol;
                            for (int kb = 0; kb < n_bra; kb++)
                                for (int kk = 0; kk < n_ket; kk++) {
                                    cout << (*this)[irow + kb] << " "
                                         << (*this)[icol + kk] << " = "
                                         << rr[n_unpaired - 1 - i][kb * n_ket + kk]
                                         << endl;
                                    mat.emplace_back(
                                        make_pair(irow + kb, icol + kk),
                                        rr[n_unpaired - 1 - i][kb * n_ket + kk]);
                                }
                        }
                    }
                }
            }
        }
    }
    vector<double> csf_apply_creation_op(int n_unpaired_bra, int twos_bra,
                                         int n_unpaired_ket, int twos_ket,
                                         int k, double scale) {
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
        LL n_bra = (j_csf_bra - i_csf_bra) / cl_bra,
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
                if (n_unpaired_bra != n_unpaired_ket) {
                    for (int j = 0; j < k; j++) {
                        xbra = (csf_bra[j >> 3] >> (j & 7)) & 1;
                        xket = (csf_ket[j >> 3] >> (j & 7)) & 1;
                        cb = bb + (xbra << 1) - 1, ck = bk + (xket << 1) - 1,
                        dc = db;
                        rr *= sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                        rr *= cg->wigner_9j(ak, bk, ck, da, db, dc, ab, bb, cb);
                        rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                        bb = cb, bk = ck, db = dc;
                    }
                    if (n_unpaired_bra == n_unpaired_ket + 1) {
                        xbra = (csf_bra[k >> 3] >> (k & 7)) & 1;
                        cb = bb + (xbra << 1) - 1, ck = bk + 0, dc = 1;
                        rr *= sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                        rr *= cg->wigner_9j(0, bk, ck, ab, db, dc, ab, bb, cb);
                        rr *= ((0 & 1) & (db & 1)) ? -1 : 1;
                        bb = cb, bk = ck, db = dc;
                        for (int j = k; j < n_unpaired_ket; j++) {
                            xbra = (csf_bra[(j + 1) >> 3] >> ((j + 1) & 7)) & 1;
                            xket = (csf_ket[j >> 3] >> (j & 7)) & 1;
                            cb = bb + (xbra << 1) - 1,
                            ck = bk + (xket << 1) - 1, dc = db;
                            rr *=
                                sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                            rr *= cg->wigner_9j(ak, bk, ck, da, db, dc, ab, bb,
                                                cb);
                            rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                            bb = cb, bk = ck, db = dc;
                        }
                    } else {
                        xket = (csf_ket[k >> 3] >> (k & 7)) & 1;
                        cb = bb + 0, ck = bk + (xket << 1) - 1, dc = 1;
                        rr *= sqrt((ck + 1) * (dc + 1) * (0 + 1) * (bb + 1));
                        rr *= cg->wigner_9j(ak, bk, ck, ak, db, dc, 0, bb, cb);
                        rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                        bb = cb, bk = ck, db = dc;
                        for (int j = k; j < n_unpaired_bra; j++) {
                            xbra = (csf_bra[j >> 3] >> (j & 7)) & 1;
                            xket = (csf_ket[(j + 1) >> 3] >> ((j + 1) & 7)) & 1;
                            cb = bb + (xbra << 1) - 1,
                            ck = bk + (xket << 1) - 1, dc = db;
                            rr *=
                                sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                            rr *= cg->wigner_9j(ak, bk, ck, da, db, dc, ab, bb,
                                                cb);
                            rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                            bb = cb, bk = ck, db = dc;
                        }
                    }
                } else {
                    for (int j = 0; j < n_unpaired_bra; j++) {
                        xbra = (csf_bra[j >> 3] >> (j & 7)) & 1;
                        xket = (csf_ket[j >> 3] >> (j & 7)) & 1;
                        cb = bb + (xbra << 1) - 1, ck = bk + (xket << 1) - 1,
                        dc = db;
                        rr *= sqrt((ck + 1) * (dc + 1) * (ab + 1) * (bb + 1));
                        rr *= cg->wigner_9j(ak, bk, ck, da, db, dc, ab, bb, cb);
                        rr *= ((ak & 1) & (db & 1)) ? -1 : 1;
                        bb = cb, bk = ck, db = dc;
                    }
                }
            }
        return r;
    }
};

template <typename, typename = void> struct CSFBigSite;

template <typename S> struct CSFBigSite<S, typename S::is_su2_t> : BigSite<S> {
    using BigSite<S>::n_orbs;
    using BigSite<S>::basis;
    shared_ptr<FCIDUMP> fcidump;
    shared_ptr<CSFSpace<S>> csf_space;
    bool is_right;
    CSFBigSite(int n_orbs, int n_max_elec, bool is_right,
               const shared_ptr<FCIDUMP> &fcidump,
               const vector<uint8_t> &orb_sym)
        : BigSite<S>(n_orbs), csf_space(make_shared<CSFSpace<S>>(
                                  n_orbs, n_max_elec, is_right, orb_sym)),
          is_right(is_right), fcidump(fcidump) {
        basis = csf_space->basis;
    }
};

} // namespace block2
