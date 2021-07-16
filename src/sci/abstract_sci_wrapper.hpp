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

#include <limits>
#include <unordered_map>
#include "../core/integral.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/csr_sparse_matrix.hpp"
#include "../core/symmetry.hpp"

/** Interface to the SCI code for a big site.
 *
 * @ATTENTION This is still work in progress and some things definitely will be changed.
 *
 * @TODO: -[x] Template symmetry type
 *        -[x] Use symmetry type instead of intPair
 *              - Also for actual SCI code
 *        -[x] Enable point group symmetry
 *        -[ ] More flexible SCI interface; block2-alternative to occs?
 *             -[ ] Divide Ctors or separate Ctor and actual initialization (violates C++ principles but would ease life)
 */

namespace sci {
    template <typename S>
    struct SHasher{
        std::size_t operator()(const S &s) const noexcept {
            return s.hash();
        }
    };

    template <typename, typename = void> struct AbstractSciWrapper;

    template <typename S>
    class AbstractSciWrapper<S, typename S::is_sz_t> {
        // Actually I made this class not abstract (pure virtual fct) in order to ease life.
    public:
        using sizPair = std::pair<std::size_t, std::size_t>;
        using BLSparseMatrix = block2::CSRSparseMatrix<S>;
        struct entryTuple1{ // ease life and define this instead of using std::tuple
            BLSparseMatrix& mat;
            int iOrb;
            entryTuple1(BLSparseMatrix& mat, int iOrb):mat{mat}, iOrb{iOrb} {}
        };
        struct entryTuple2{ // ease life and define this instead of using std::tuple
            BLSparseMatrix& mat;
            int iOrb, jOrb;
            entryTuple2(BLSparseMatrix& mat, int iOrb, int jOrb):mat{mat}, iOrb{iOrb}, jOrb{jOrb}{}
        };
        int nOrbOther, nOrbThis, nOrb; //!< *spatial* orbitals
        int nMaxAlphaEl, nMaxBetaEl, nMaxEl; //!< Maximal number of alpha/beta electrons
        bool isRight; //!< Whether orbitals of SCI are right to other orbitals or not
        AbstractSciWrapper() : AbstractSciWrapper(1, 1, 1, 1, -1, nullptr, {}) {}

        /** Initialization via generated CI space based on nMax*
         *
         * @param nOrb Total (spatial) orbitals
         * @param nOrbThis Orbitals handled via SCI
         * @param isRight Whether orbitals of SCI are right to other orbitals or not
         * @param nMaxAlphaEl Maximal number of alpha electrons in external space
         * @param nMaxBetaEl Maximal number of beta electrons in external space
         * @param nMaxEl Maximal number of alpha+beta electrons in external space
         * @param fcidump block2 FCIDUMP file
         */
        AbstractSciWrapper(int nOrb_, int nOrbThis_, bool isRight,
                           const std::shared_ptr<block2::FCIDUMP>& fcidump,
                           const std::vector<uint8_t>& orbsym,
                           int nMaxAlphaEl, int nMaxBetaEl, int nMaxEl, bool verbose=true):
                nOrbOther{nOrb_-nOrbThis_}, nOrbThis{nOrbThis_}, nOrb{nOrb_},
                isRight{isRight},
                nMaxAlphaEl{nMaxAlphaEl}, nMaxBetaEl{nMaxBetaEl}, nMaxEl{nMaxEl}, verbose{verbose}{
            if(nOrbOther < 0)
                throw std::invalid_argument("nOrb < nOrbThis?");
        };
        /** Initialization via externally given determinants in `occs`.
         *
         * @param nOrb Total (spatial) orbitals
         * @param nOrbThis Orbitals handled via SCI
         * @param isRight: Whether orbitals of SCI are right to other orbitals or not
         * @param occs  Vector of occupations for filling determinants. If used, nMax* are ignored!
         * @param fcidump block2 FCIDUMP file
         */
        AbstractSciWrapper(int nOrb_, int nOrbThis_, bool isRight,
                           const std::shared_ptr<block2::FCIDUMP>& fcidump,
                           const std::vector<uint8_t>& orbsym,
                           const vector<vector<int>>& occs, bool verbose=true):
                nOrbOther{nOrb_-nOrbThis_}, nOrbThis{nOrbThis_}, nOrb{nOrb_},
                isRight{isRight},
                nMaxAlphaEl{-1}, nMaxBetaEl{-1}, nMaxEl{-1}, verbose{verbose}{
            if(nOrbOther < 0)
                throw std::invalid_argument("nOrb < nOrbThis?");
        };
        virtual ~AbstractSciWrapper() = default;

        std::vector<S> quantumNumbers; //!< vector of (N,2*Sz) quantum numbers used
        std::unordered_map<S,int,SHasher<S>> quantumNumberToIdx; //!< quantum number to idx in quantumNumbers vector
        std::vector<sizPair> offsets; //!< index ranges [start,end) for each quantum number (in order of quantumNumbers)
        std::size_t nDet; //!< Total number of determinants

        double eps = 1e-12; //!< Sparsity value threshold. Everything below eps will be set to 0.0");
        double sparsityThresh = 0.75; // After > #zeros/#tot the sparse matrix is activated
        int sparsityStart = 100*100; // After which matrix size (nCol * nRow) should sparse matrices be activated
        bool verbose = true;

        // Routines for filling the physical operator matrices
        /** Fill Identity */
        virtual void fillOp_I(BLSparseMatrix& mat) const {throwError();};
        /** Fill N */
        virtual void fillOp_N(BLSparseMatrix& mat) const {throwError();};
        /** Fill N^2 */
        virtual void fillOp_NN(BLSparseMatrix& mat) const {throwError();};
        /** Fill H */
        virtual void fillOp_H(BLSparseMatrix& mat) const {throwError();};
        /** Fill a' */
        virtual void fillOp_C(const S& deltaQN, BLSparseMatrix& mat, int iOrb) const {throwError();};
        /** Fill a */
        virtual void fillOp_D(const S& deltaQN, BLSparseMatrix& mat, int iOrb) const {throwError();};
        /** Fill R */
        virtual void fillOp_R(const S& deltaQN, std::vector<entryTuple1>& entries) const {throwError();};
        /** Fill R' */
        virtual void fillOp_RD(const S& deltaQN, std::vector<entryTuple1>& entries) const {throwError();};
        /** Fill A = i j */
        virtual void fillOp_A(const S& deltaQN, BLSparseMatrix& mat, int iOrb, int jOrb) const {throwError();};
        /** Fill A' = j'i' (note order!) */
        virtual void fillOp_AD(const S& deltaQN, BLSparseMatrix& mat, int iOrb, int jOrb) const {throwError();};
        /** Fill B = i'j */
        virtual void fillOp_B(const S& deltaQN, BLSparseMatrix& mat, int iOrb, int jOrb) const {throwError();};
        /** Fill P op */
        virtual void fillOp_P(const S& deltaQN, std::vector<entryTuple2>& entries) const {throwError();};
        /** Fill P' op */
        virtual void fillOp_PD(const S& deltaQN, std::vector<entryTuple2>& entries) const {throwError();};
        /** Fill Q op */
        virtual void fillOp_Q(const S& deltaQN, std::vector<entryTuple2>& entries) const {throwError();};
        /** Call this after the fillOps are done*/
        virtual void finalize() {};
    private:
        void throwError() const{
            throw std::runtime_error("You used the abstract sci wrapper and not the actual sci wrapper");
        }
    };
}

