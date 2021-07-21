
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

/** Adapted from sciblock2/sci/sciWrapper.hpp
 * by Huanchen Zhai Jul 18, 2021.
 *
 * Author: Henrik R. Larsson <larsson@caltech.edu>
 */

#pragma once

#include "../core/symmetry.hpp"
#include "big_site.hpp"
#include "sci_fcidump.hpp"
#include "sci_fock_determinant.hpp"
#include <memory>
#include <unordered_map>

using namespace std;

namespace block2 {

template <typename, typename = void> struct SCIFockBigSite;

template <typename S>
struct SCIFockBigSite<S, typename S::is_sz_t> : public BigSite<S> {
    using sizPair = std::pair<std::size_t, std::size_t>;
    using BLSparseMatrix = block2::CSRSparseMatrix<S>;
    struct entryTuple1 { // ease life and define this instead of using
                         // std::tuple
        BLSparseMatrix &mat;
        int iOrb;
        entryTuple1(BLSparseMatrix &mat, int iOrb) : mat{mat}, iOrb{iOrb} {}
    };
    struct entryTuple2 { // ease life and define this instead of using
                         // std::tuple
        BLSparseMatrix &mat;
        int iOrb, jOrb;
        entryTuple2(BLSparseMatrix &mat, int iOrb, int jOrb)
            : mat{mat}, iOrb{iOrb}, jOrb{jOrb} {}
    };
    int nOrbOther, nOrbThis, nOrb; //!< *spatial* orbitals
    int nMaxAlphaEl, nMaxBetaEl,
        nMaxEl; //!< Maximal number of alpha/beta electrons
    bool
        isRight; //!< Whether orbitals of SCI are right to other orbitals or not
    std::vector<S> quantumNumbers; //!< vector of (N,2*Sz) quantum numbers used
    std::unordered_map<S, int>
        quantumNumberToIdx; //!< quantum number to idx in quantumNumbers vector
    std::vector<sizPair> offsets; //!< index ranges [start,end) for each quantum
                                  //!< number (in order of quantumNumbers)
    std::size_t nDet;             //!< Total number of determinants

    double eps = 1e-12; //!< Sparsity value threshold. Everything below eps will
                        //!< be set to 0.0");
    double sparsityThresh =
        0.75; // After > #zeros/#tot the sparse matrix is activated
    int sparsityStart = 100 * 100; // After which matrix size (nCol * nRow)
                                   // should sparse matrices be activated
    bool verbose = true;
    bool excludeQNs = false;
    std::vector<int> qnIdxBraH; //!< vector of quantum number indices USED IN
                                //!< MATRICES for bra site; H op
    std::vector<int> qnIdxKetH; //!< vector of quantum number indices USED IN
                                //!< MATRICES for ket site
                                //!! If empty, use qnIdxBra; H op
    std::vector<int> qnIdxBraI, qnIdxKetI;
    std::vector<int> qnIdxBraQ, qnIdxKetQ;
    std::vector<int> qnIdxBraA, qnIdxKetA;
    std::vector<int> qnIdxBraB, qnIdxKetB;
    std::vector<int> qnIdxBraP, qnIdxKetP;
    std::vector<int> qnIdxBraR, qnIdxKetR;
    std::vector<int> qnIdxBraC, qnIdxKetC;
    // TODO: Use map to speed up lookup (but #quantumNumbers should not be super
    // big)

    SCIFockBigSite() : SCIFockBigSite(1, 1, true, nullptr, {}, {}) {}

    /** Initialization via generated CI space based on nMax*
     *
     * @param nOrb Total (spatial) orbitals
     * @param nOrbThis Orbitals handled via SCI
     * @param isRight: Whether orbitals of SCI are right to other orbitals or
     * not
     * @param fcidump block2 FCIDUMP file
     * @param nMaxAlphaEl Maximal number of alpha electrons in external space
     * @param nMaxBetaEl Maximal number of beta electrons in external space
     * @param nMaxEl Maximal number of alpha+beta electrons in external space
     */
    SCIFockBigSite(int nOrb, int nOrbExt, bool isRight,
                   const std::shared_ptr<block2::FCIDUMP> &fcidump,
                   const std::vector<uint8_t> &orbsym, int nMaxAlphaEl,
                   int nMaxBetaEl, int nMaxEl, bool verbose = true)
        : SCIFockBigSite{nOrb,        nOrbExt,    isRight, fcidump, orbsym,
                         nMaxAlphaEl, nMaxBetaEl, nMaxEl,  {},      verbose} {}

    /** Initialization via externally given determinants in `occs`.
     *
     * @param nOrb Total (spatial) orbitals
     * @param nOrbThis Orbitals handled via SCI
     * @param isRight: Whether orbitals of SCI are right to other orbitals or
     * not
     * @param occs  Vector of occupations for filling determinants. If used,
     * nMax* are ignored!
     * @param fcidump block2 FCIDUMP file
     */
    SCIFockBigSite(int nOrb, int nOrbExt, bool isRight,
                   const std::shared_ptr<block2::FCIDUMP> &fcidump,
                   const std::vector<uint8_t> &orbsym,
                   const vector<vector<int>> &occs, bool verbose = true)
        : SCIFockBigSite{nOrb, nOrbExt, isRight, fcidump, orbsym,
                         -999, -999,    -999,    occs,    verbose} {}
    /** Initialization either via occs or nMax*.
     * See specialized C'tors.
     */
    SCIFockBigSite(int nOrb, int nOrbExt, bool isRight,
                   const std::shared_ptr<block2::FCIDUMP> &fcidump,
                   const std::vector<uint8_t> &orbsym, int nMaxAlphaEl,
                   int nMaxBetaEl, int nMaxEl, const vector<vector<int>> &occs,
                   bool verbose = true);
    virtual ~SCIFockBigSite() = default;
    void get_site_ops(
        uint16_t m,
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
        const override;
    std::vector<SCIFockDeterminant>
        fragSpace; //!< Dummy fragment space. CAS stuff is first
                   //!< fragment; CI is second fragment
    std::unordered_map<SCIFockDeterminant, size_t> fragIndexMap;

    // Routines for filling the physical operator matrices
    /** Fill Identity */
    void fillOp_I(BLSparseMatrix &mat) const;
    /** Fill N */
    void fillOp_N(BLSparseMatrix &mat) const;
    /** Fill N^2 */
    void fillOp_NN(BLSparseMatrix &mat) const;
    /** Fill H */
    void fillOp_H(BLSparseMatrix &mat) const;
    /** Fill a' */
    void fillOp_C(const S &deltaQN, BLSparseMatrix &mat, int iOrb) const;
    /** Fill a */
    void fillOp_D(const S &deltaQN, BLSparseMatrix &mat, int iOrb) const;
    /** Fill R */
    void fillOp_R(const S &deltaQN, std::vector<entryTuple1> &entries) const;
    /** Fill R' */
    void fillOp_RD(const S &deltaQN, std::vector<entryTuple1> &entries) const;
    /** Fill A = i j */
    void fillOp_A(const S &deltaQN, BLSparseMatrix &mat, int iOrb,
                  int jOrb) const;
    /** Fill A' = j'i' (note order!) */
    void fillOp_AD(const S &deltaQN, BLSparseMatrix &mat, int iOrb,
                   int jOrb) const;
    /** Fill B = i'j */
    void fillOp_B(const S &deltaQN, BLSparseMatrix &mat, int iOrb,
                  int jOrb) const;
    /** Fill P op */
    void fillOp_P(const S &deltaQN, std::vector<entryTuple2> &entries) const;
    /** Fill P' op */
    void fillOp_PD(const S &deltaQN, std::vector<entryTuple2> &entries) const;
    /** Fill Q op */
    void fillOp_Q(const S &deltaQN, std::vector<entryTuple2> &entries) const;

    int ompThreads = 1;
    int setOmpThreads(int nThreads = 0) {
        if (nThreads < 1) {
#ifdef _OPENMP
            ompThreads = omp_get_max_threads();
#else
            ompThreads = 1;
#endif
        } else {
            ompThreads = 1;
        }
        return ompThreads;
    }
    /** Call this after the fillOps are done*/
    void finalize() const;
    /** Check whether OMP works. I had weird errors related to clang and wrong
     * OMP libs linking */
    void checkOMP() const;
    // vv for ExcludeQNs classes
    /** Convenience function to set several qnIdx vectors by one input using
     * ops. If ops == {"X"}, then all vectors will be set.*/
    void setQnIdxBra(const std::vector<int> &inp, const std::vector<char> &ops);
    /** Convenience function to set several qnIdx vectors by one input using
     * ops. If ops == {"X"}, then all vectors will be set.*/
    void setQnIdxKet(const std::vector<int> &inp, const std::vector<char> &ops);

    void setQnIdxBra(const std::vector<int> &inp) { setQnIdxBra(inp, {'X'}); }
    void setQnIdxKet(const std::vector<int> &inp) { setQnIdxKet(inp, {'X'}); }
    // construct occs vector for left or right ci space from excitations
    // nalpha, nbeta, nelec are max number of electrons
    static vector<vector<int>> ras_space(bool is_right, int norb, int nalpha,
                                         int nbeta, int nelec) {
        map<pair<int, pair<int, int>>, vector<vector<int>>> mp;
        vector<int> ref(is_right ? 0 : norb * 2);
        if (!is_right)
            for (int i = 0; i < norb * 2; i++)
                ref[i] = i;
        mp[make_pair(0, make_pair(0, 0))].push_back(ref);
        for (int i = 1; i <= nelec; i++) {
            for (int ia = 0; ia <= nalpha; ia++) {
                int ib = i - ia;
                if (ib < 0 || ib > nbeta)
                    continue;
                vector<vector<int>> r;
                if (mp.find(make_pair(i - 1, make_pair(ia - 1, ib))) !=
                    mp.end()) {
                    vector<vector<int>> &mref =
                        mp.at(make_pair(i - 1, make_pair(ia - 1, ib)));
                    for (auto &mm : mref) {
                        for (int j = 0; j < norb; j++) {
                            int pm =
                                (int)(lower_bound(mm.begin(), mm.end(), j * 2) -
                                      mm.begin());
                            if (is_right &&
                                (pm == (int)mm.size() || mm[pm] != j * 2)) {
                                r.push_back(mm);
                                r.back().insert(r.back().begin() + pm, j * 2);
                            } else if (!is_right && !(pm == (int)mm.size() ||
                                                      mm[pm] != j * 2)) {
                                r.push_back(mm);
                                r.back().erase(r.back().begin() + pm);
                            }
                        }
                    }
                }
                if (mp.find(make_pair(i - 1, make_pair(ia, ib - 1))) !=
                    mp.end()) {
                    vector<vector<int>> &mref =
                        mp.at(make_pair(i - 1, make_pair(ia, ib - 1)));
                    for (auto &mm : mref) {
                        for (int j = 0; j < norb; j++) {
                            int pm = (int)(lower_bound(mm.begin(), mm.end(),
                                                       j * 2 + 1) -
                                           mm.begin());
                            if (is_right &&
                                (pm == (int)mm.size() || mm[pm] != j * 2 + 1)) {
                                r.push_back(mm);
                                r.back().insert(r.back().begin() + pm,
                                                j * 2 + 1);
                            } else if (!is_right && !(pm == (int)mm.size() ||
                                                      mm[pm] != j * 2 + 1)) {
                                r.push_back(mm);
                                r.back().erase(r.back().begin() + pm);
                            }
                        }
                    }
                }
                sort(r.begin(), r.end());
                r.resize(distance(r.begin(), unique(r.begin(), r.end())));
                mp[make_pair(i, make_pair(ia, ib))] = r;
            }
        }
        vector<vector<int>> rr;
        for (auto &mm : mp)
            rr.insert(rr.end(), mm.second.begin(), mm.second.end());
        return rr;
    }

  protected:
    SCIFCIDUMPTwoInt ints2;
    SCIFCIDUMPOneInt ints1;
    using TripletVec = vector<pair<pair<MKL_INT, MKL_INT>, double>>;

    /* Get possible ket, bra quantum number combinations for given deltaQN*/
    std::vector<std::pair<int, int>> getQNpairs(const BLSparseMatrix &mat,
                                                const S &deltaQN) const;
    // vv for ExcludeQNs classes
    bool doAllocateEmptyMats() const { return excludeQNs; }
    std::vector<std::pair<int, int>>
    getQNPairsImpl(const BLSparseMatrix &mat, const S &deltaQN,
                   const std::vector<int> &qnIdxBra,
                   const std::vector<int> &qnIdxKet) const;
    bool idxInKet(const int braIdx, const std::vector<int> &qnIdxKet) const;
    std::vector<std::pair<int, int>> getQNpairsH(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraH, qnIdxKetH)
                          : getQNpairs(mat, deltaQN);
    };
    std::vector<std::pair<int, int>> getQNpairsQ(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraQ, qnIdxKetQ)
                          : getQNpairs(mat, deltaQN);
    };
    std::vector<std::pair<int, int>> getQNpairsA(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraA, qnIdxKetA)
                          : getQNpairs(mat, deltaQN);
    };
    std::vector<std::pair<int, int>> getQNpairsB(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraB, qnIdxKetB)
                          : getQNpairs(mat, deltaQN);
    };
    std::vector<std::pair<int, int>> getQNpairsP(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraP, qnIdxKetP)
                          : getQNpairs(mat, deltaQN);
    };
    std::vector<std::pair<int, int>> getQNpairsR(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraR, qnIdxKetR)
                          : getQNpairs(mat, deltaQN);
    };
    std::vector<std::pair<int, int>> getQNpairsC(const BLSparseMatrix &mat,
                                                 const S &deltaQN) const {
        return excludeQNs ? getQNPairsImpl(mat, deltaQN, qnIdxBraC, qnIdxKetC)
                          : getQNpairs(mat, deltaQN);
    };

    int getThreadID() const {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
    // runtime data
    mutable double timeH = 0, timeP = 0, timePD = 0, timeQ = 0, timeR = 0,
                   timeRD = 0, timeC = 0, timeD = 0, timeA = 0, timeAD = 0,
                   timeB = 0;
    mutable double summSparsityH = 0, summSparsityP = 0, summSparsityPD = 0,
                   summSparsityQ = 0, summSparsityR = 0, summSparsityRD = 0,
                   summSparsityC = 0, summSparsityD = 0, summSparsityA = 0,
                   summSparsityAD = 0, summSparsityB = 0;
    // vv counts for each QN
    mutable size_t qnCountsH = 0, qnCountsP = 0, qnCountsPD = 0, qnCountsQ = 0,
                   qnCountsR = 0, qnCountsRD = 0, qnCountsC = 0, qnCountsD = 0,
                   qnCountsA = 0, qnCountsAD = 0, qnCountsB = 0;
    mutable size_t totCountsH = 0, totCountsP = 0, totCountsPD = 0,
                   totCountsQ = 0, totCountsR = 0, totCountsRD = 0,
                   totCountsC = 0, totCountsD = 0, totCountsA = 0,
                   totCountsAD = 0, totCountsB = 0;
    // vv number of tense/sparse matrices
    mutable size_t numDenseH = 0, numDenseP = 0, numDensePD = 0, numDenseQ = 0,
                   numDenseR = 0, numDenseRD = 0, numDenseC = 0, numDenseD = 0,
                   numDenseA = 0, numDenseAD = 0, numDenseB = 0;
    mutable size_t numSparseH = 0, numSparseP = 0, numSparsePD = 0,
                   numSparseQ = 0, numSparseR = 0, numSparseRD = 0,
                   numSparseC = 0, numSparseD = 0, numSparseA = 0,
                   numSparseAD = 0, numSparseB = 0;
    mutable size_t numZeroH = 0, numZeroP = 0, numZeroPD = 0, numZeroQ = 0,
                   numZeroR = 0, numZeroRD = 0, numZeroC = 0, numZeroD = 0,
                   numZeroA = 0, numZeroAD = 0, numZeroB = 0;
    // vv in bytes
    mutable size_t usedMemH = 0, usedMemP = 0, usedMemPD = 0, usedMemQ = 0,
                   usedMemR = 0, usedMemRD = 0, usedMemC = 0, usedMemD = 0,
                   usedMemA = 0, usedMemAD = 0, usedMemB = 0;
    // vv eigenMat is used es entry just for recycling purposes; iRow is coeffs
    // row
    template <bool Symmetrize>
    size_t fillCoeffs(sci_detail::COOSparseMat<double> &cooMat,
                      sci_detail::DenseMat<TripletVec> &coeffs, int iRow,
                      block2::CSRMatrixRef &mat, double &summSparsity,
                      size_t &numSparse, size_t &numDense,
                      size_t &usedMem) const;
    template <bool Cop>
    void fillOp_C_impl(const S &deltaQN, BLSparseMatrix &mat, int iOrb) const;
    template <int Type>
    void fillOp_AB_impl(const S &deltaQN, BLSparseMatrix &mat, int iOrb,
                        int jOrb) const;
    template <bool Dagger>
    void fillOp_R_impl(const S &deltaQN,
                       std::vector<entryTuple1> &entries) const;
    template <bool Dagger>
    void fillOp_P_impl(const S &deltaQN,
                       std::vector<entryTuple2> &entries) const;
};

extern template struct SCIFockBigSite<block2::SZ>;

} // namespace block2
