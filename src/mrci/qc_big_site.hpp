/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

#include "../core/csr_sparse_matrix.hpp"
#include "../core/integral.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/symmetry.hpp"
#include <limits>
#include <unordered_map>

/** A big site. */

namespace block2 {

template <typename, typename = void> struct BigSiteQC;

template <typename S> struct BigSiteQC<S, typename S::is_sz_t> {
    int n_orbs_other, n_orbs_this, n_orbs; //!< *spatial* orbitals
    int n_alpha, n_beta, n_elec; //!< Maximal number of alpha/beta electrons
    bool is_right; //!< Whether orbitals of SCI are right to other orbitals or
                   //!< not
    BigSiteQC() {}
    // : BigSiteQC(1, 1, 1, 1, -1, nullptr, {}) {}

    // /** Initialization via generated CI space based on nMax*
    //  *
    //  * @param nOrb Total (spatial) orbitals
    //  * @param nOrbThis Orbitals handled via SCI
    //  * @param isRight Whether orbitals of SCI are right to other orbitals or
    //  not
    //  * @param nMaxAlphaEl Maximal number of alpha electrons in external space
    //  * @param nMaxBetaEl Maximal number of beta electrons in external space
    //  * @param nMaxEl Maximal number of alpha+beta electrons in external space
    //  * @param fcidump block2 FCIDUMP file
    //  */
    // BigSiteQC(int nOrb_, int nOrbThis_, bool isRight,
    //                    const std::shared_ptr<block2::FCIDUMP>& fcidump,
    //                    const std::vector<uint8_t>& orbsym,
    //                    int nMaxAlphaEl, int nMaxBetaEl, int nMaxEl, bool
    //                    verbose=true):
    //         nOrbOther{nOrb_-nOrbThis_}, nOrbThis{nOrbThis_}, nOrb{nOrb_},
    //         isRight{isRight},
    //         nMaxAlphaEl{nMaxAlphaEl}, nMaxBetaEl{nMaxBetaEl}, nMaxEl{nMaxEl},
    //         verbose{verbose}{
    //     if(nOrbOther < 0)
    //         throw std::invalid_argument("nOrb < nOrbThis?");
    // };
    // /** Initialization via externally given determinants in `occs`.
    //  *
    //  * @param n_orbs Total (spatial) orbitals
    //  * @param n_orbs_this Orbitals handled via SCI
    //  * @param is_right: Whether orbitals of SCI are right to other orbitals
    //  or not
    //  * @param occs  Vector of occupations for filling determinants. If used,
    //  nMax* are ignored!
    //  * @param fcidump block2 FCIDUMP file
    //  */
    // BigSiteQC(int n_orbs, int n_orbs_this, bool is_right,
    //                    const shared_ptr<FCIDUMP>& fcidump,
    //                    const vector<uint8_t>& orbsym,
    //                    const vector<vector<int>>& occs, bool verbose=true):
    //         n_orbs_other(n_orbs-n_orbs_this), n_orbs_this(n_orbs_this),
    //         n_orbs(n_orbs), is_right(is_right), nMaxAlphaEl{-1},
    //         nMaxBetaEl{-1}, nMaxEl{-1}, verbose(verbose) {
    // };
    virtual ~BigSiteQC() = default;

    vector<S> qs; //!< vector of (N,2*Sz) quantum numbers used
    vector<pair<size_t, size_t>>
        offsets; //!< index ranges [start,end) for each quantum number (in order
                 //!< of quantumNumbers)
    size_t n_det; //!< Total number of determinants

    double eps = 1E-12; //!< sparsity value threshold. Everything below eps will
                        //!< be set to 0.0");
    double sparsity_thresh =
        0.75; // After > #zeros/#tot the sparse matrix is activated
    int sparsity_start = 100 * 100; // After which matrix size (nCol * nRow)
                                    // should sparse matrices be activated
    bool verbose = true;

    // Routines for filling the physical operator matrices
    /** Fill Identity */
    virtual void fill_op_i(CSRSparseMatrix<S> &mat) const { throw_error(); };
    /** Fill N */
    virtual void fill_op_n(CSRSparseMatrix<S> &mat) const { throw_error(); };
    /** Fill N^2 */
    virtual void fill_op_nn(CSRSparseMatrix<S> &mat) const { throw_error(); };
    /** Fill H */
    virtual void fill_op_h(CSRSparseMatrix<S> &mat) const { throw_error(); };
    /** Fill a' */
    virtual void fill_op_c(CSRSparseMatrix<S> &mat, int i) const {
        throw_error();
    };
    /** Fill a */
    virtual void fill_op_d(CSRSparseMatrix<S> &mat, int i) const {
        throw_error();
    };
    /** Fill R */
    virtual void
    fill_op_r(vector<pair<int, CSRSparseMatrix<S>>> &entries) const {
        throw_error();
    };
    /** Fill R' */
    virtual void
    fill_op_rd(vector<pair<int, CSRSparseMatrix<S>>> &entries) const {
        throw_error();
    };
    /** Fill A = i j */
    virtual void fill_op_a(CSRSparseMatrix<S> &mat, int i, int j) const {
        throw_error();
    };
    /** Fill A' = j'i' (note order!) */
    virtual void fill_op_ad(CSRSparseMatrix<S> &mat, int i, int j) const {
        throw_error();
    };
    /** Fill B = i'j */
    virtual void fill_op_b(CSRSparseMatrix<S> &mat, int i, int j) const {
        throw_error();
    };
    /** Fill P op */
    virtual void
    fill_op_p(vector<pair<pair<int, int>, CSRSparseMatrix<S>>> &entries) const {
        throw_error();
    };
    /** Fill P' op */
    virtual void fill_op_pd(
        vector<pair<pair<int, int>, CSRSparseMatrix<S>>> &entries) const {
        throw_error();
    };
    /** Fill Q op */
    virtual void
    fill_op_q(vector<pair<pair<int, int>, CSRSparseMatrix<S>>> &entries) const {
        throw_error();
    };
    /** Call this after the fillOps are done*/
    virtual void finalize(){};

  private:
    void throw_error() const {
        throw std::runtime_error(
            "You used the abstract big site and not the actual big site.");
    }
};
} // namespace block2
