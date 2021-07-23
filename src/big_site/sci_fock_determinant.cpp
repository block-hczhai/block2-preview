
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

/** Adapted from sciblock2/sci/fragmentDeterminant.cpp
 * by Huanchen Zhai Jul 18, 2021.
 * 
 * Author: Henrik R. Larsson <larsson@caltech.edu>
 */

#include "sci_fock_determinant.hpp"
#include <algorithm>

namespace block2 {

namespace {
using boolVec = std::vector<bool>;
using intVec = std::vector<int>;
} // namespace

std::pair<std::vector<boolVec>, std::vector<intVec>>
SCIFockDeterminant::generateFCIdeterminants(const int nOrb, const int nEl,
                                            const bool fockRep) {
    assert(nEl <= nOrb);
    boolVec occ(nOrb, false);
    for (int i = 0; i < nEl; ++i) {
        occ[i] = true;
    }
    std::vector<boolVec> occVec;
    std::vector<intVec> closedVec;
    intVec occOrbs(nEl);
    // occAlpha needs to be sorted for proper behavior of next_permutation! =>
    // reverse comparison to start with everything in the beginning occupied.
    const auto vecComparison = [](const bool &a, const bool &b) {
        return a > b;
    };
    do {
        if (fockRep) {
            occVec.push_back(occ);
        } else {
            int iEl = 0;
            for (int iOrb = 0; iOrb < nOrb; ++iOrb) {
                if (occ[iOrb]) {
                    occOrbs[iEl++] = iOrb;
                }
            }
            closedVec.push_back(occOrbs);
        }
    } while (std::next_permutation(occ.begin(), occ.end(), vecComparison));
    occVec.shrink_to_fit();
    closedVec.shrink_to_fit();
    return std::make_pair(occVec, closedVec);
}

std::vector<SCIFockDeterminant> SCIFockDeterminant::generateFCIfragmentSpace(
    const int nElAlpha, const int nElBeta, const int nSpinOrb) {
    std::vector<SCIFockDeterminant> vec;
    generateFCIfragmentSpace(nElAlpha, nElBeta, nSpinOrb, vec);
    return vec;
}

void SCIFockDeterminant::generateFCIfragmentSpace(
    const int nElAlpha, const int nElBeta, const int nSpinOrb,
    std::vector<SCIFockDeterminant> &detVec) {
    if ((nSpinOrb % 2) != 0) {
        throw std::invalid_argument(
            "generateFCIfragmentSpace: nSpinOrb not even");
    }
    if (nElAlpha + nElBeta > nSpinOrb) {
        throw std::invalid_argument(
            "generateFCIfragmentSpace: nSpinOrb smaller then nEl");
    }
    // vv not needed
    // std::vector<int> orbitalsAlpha(nSpinOrb/2);
    // std::vector<int> orbitalsBeta(nSpinOrb/2);
    // std::generate(orbitalsAlpha.begin(), orbitalsAlpha.end(), [n = -2] ()
    // mutable { n+=2; return n; }); std::generate(orbitalsBeta.begin(),
    // orbitalsBeta.end(), [n = -1] () mutable { n+=2; return n; });
    // generate all alpha beta strings and then fill determinants.
    const auto occOrbsAlpha =
        generateFCIdeterminants(nSpinOrb / 2, nElAlpha, false).second;
    const auto occOrbsBeta =
        generateFCIdeterminants(nSpinOrb / 2, nElBeta, false).second;
    // now, fill determinants
    const auto origSize = detVec.size();
    detVec.resize(origSize + occOrbsAlpha.size() * occOrbsBeta.size());
    std::size_t ii = origSize;
    if (true) { // F ordering: b is fast
        for (const auto &a : occOrbsAlpha) {
            SCIFockDeterminant detAlpha(nSpinOrb, nElAlpha, nElBeta);
            for (const auto ia : a) {
                detAlpha.setocc(2 * ia, true);
            }
            for (const auto &b : occOrbsBeta) {
                auto det = detAlpha;
                for (const auto ib : b) {
                    det.setocc(2 * ib + 1, true);
                }
                assert(det.consistencyCheck());
                detVec[ii++] = std::move(det);
            }
        }
    } else { // C ordering: a is fast; used in block2
        for (const auto &b : occOrbsBeta) {
            SCIFockDeterminant detBeta(nSpinOrb, nElAlpha, nElBeta);
            for (const auto ib : b) {
                detBeta.setocc(2 * ib + 1, true);
            }
            for (const auto &a : occOrbsAlpha) {
                auto det = detBeta;
                for (const auto ia : a) {
                    det.setocc(2 * ia, true);
                }
                assert(det.consistencyCheck());
                detVec[ii++] = std::move(det);
            }
        }
    }
}

/** Apply creator operator on orbital iOrb to determinant specified by iClosed.
 *
 * @return Pair of phase and new determinant (closed orbitals); properly sorted.
 * If phase is 0, the result is 0
 */
std::pair<int, std::vector<int>>
SCIFockDeterminant::applyCreator(const std::vector<int> &iClosed,
                                 const int iOrb) {
    auto nOrb = (int)(iClosed.size());
    assert(std::is_sorted(iClosed.cbegin(), iClosed.cend()));
    if (nOrb == 0) {
        return std::make_pair<int, std::vector<int>>(1, std::vector<int>{iOrb});
    }
    int idx = (int)(std::lower_bound(iClosed.begin(), iClosed.end(), iOrb) -
                    iClosed.begin());
    if (idx < nOrb and iClosed[idx] == iOrb) { // Already in there
        return std::make_pair<int, std::vector<int>>(
            0, std::vector<int>{}); // Go to Fermi Hell
    } else {
        int phase = idx % 2 == 0 ? 1 : -1;
        std::vector<int> out(nOrb + 1);
        for (int i = 0; i < idx; ++i) {
            out[i] = iClosed[i];
        }
        out[idx] = iOrb;
        for (int i = idx; i < nOrb; ++i) {
            out[i + 1] = iClosed[i];
        }
        assert(std::is_sorted(out.cbegin(), out.cend()));
        return std::make_pair<int, std::vector<int>>(std::move(phase),
                                                     std::move(out));
    }
}

/** Apply annihilation operator on orbital iOrb to determinant specified by
 * iClosed.
 *
 * @return Pair of phase and new determinant (closed orbitals); properly sorted.
 * If phase is 0, the result is 0
 */
std::pair<int, std::vector<int>>
SCIFockDeterminant::applyAnnihilator(const std::vector<int> &iClosed,
                                     const int iOrb) {
    auto nOrb = (int)(iClosed.size());
    assert(std::is_sorted(iClosed.cbegin(), iClosed.cend()));
    if (nOrb == 0) {
        return std::make_pair<int, std::vector<int>>(
            0, std::vector<int>{}); // Go to Fermi Hell
    }
    int idx = (int)(std::lower_bound(iClosed.begin(), iClosed.end(), iOrb) -
                    iClosed.begin());
    if (idx < nOrb and iClosed[idx] == iOrb) { // Already in there
        int phase = idx % 2 == 0 ? 1 : -1;
        std::vector<int> out(nOrb - 1);
        for (int i = 0; i < idx; ++i) {
            out[i] = iClosed[i];
        }
        for (int i = idx + 1; i < nOrb; ++i) {
            out[i - 1] = iClosed[i];
        }
        assert(std::is_sorted(out.cbegin(), out.cend()));
        return std::make_pair<int, std::vector<int>>(std::move(phase),
                                                     std::move(out));
    } else {
        return std::make_pair<int, std::vector<int>>(
            0, std::vector<int>{}); // Go to Fermi Hell
    }
}

double SCIFockDeterminant::Energy(const SCIFCIDUMPOneInt &I1,
                                  const SCIFCIDUMPTwoInt &I2,
                                  const std::vector<int> &closed) const {
    double energy = 0.0;
    for (int i = 0; i < closed.size(); i++) {
        int I = closed[i];
        energy += I1(I, I);
        for (int j = i + 1; j < closed.size(); j++) {
            int J = closed[j];
            energy += I2.Direct(I / 2, J / 2);
            if ((I % 2) == (J % 2)) {
                energy -= I2.Exchange(I / 2, J / 2);
            }
        }
    }
    return energy;
}

double SCIFockDeterminant::Hij_2Excite(const int i, const int j, const int a,
                                       const int b, const SCIFCIDUMPOneInt &I1,
                                       const SCIFCIDUMPTwoInt &I2) const {
    double sgn =
        1.0; // TODO HRL: make a bool out of it? But in parity routine, it is
             // multiplied with float. So dunno. Maybe, make float out of it?
    using std::max;
    using std::min;
    int I = min(i, j), J = max(i, j), A = min(a, b), B = max(a, b);
    parity(min(I, A), max(I, A), sgn);
    parity(min(J, B), max(J, B), sgn);
    if (A > J or B < I) {
        sgn *= -1.;
    }
    return sgn * (I2(A, I, B, J) - I2(A, J, B, I));
}

double SCIFockDeterminant::Hij_1Excite(const int i, const int a,
                                       const SCIFCIDUMPOneInt &I1,
                                       const SCIFCIDUMPTwoInt &I2) const {
    using std::max;
    using std::min;
    double sgn = 1.0;
    parity(min(i, a), max(i, a), sgn);

    double energy = I1(i, a);
    constexpr long one = 1;
    for (int I = 0; I < EffDetLen; I++) {
        long reprBit = repr[I];
        while (reprBit != 0) {
            int pos = sci_detail::ffsl(reprBit);
            int j = I * 64 + pos - 1;
            energy += (I2(i, a, j, j) - I2(i, j, j, a));
            reprBit &= ~(one << (pos - 1));
        }
    }
    energy *= sgn;
    return energy;
}

void SCIFockDeterminant::parity(const int start, const int end,
                                double &parity) const {
    constexpr long one = 1;
    // TODO HRL: see getOcc: %64 and /64 can be optimized
    // TODO:; Avoid duplication with Determinants.hpp
    long mask = (one << (start % 64)) - one;
    long result = repr[start / 64] & mask;
    int nonZeroBits = -sci_detail::BitCount(result);

    assert(end / 64 < repr.size());
    for (int i = start / 64; i < end / 64; i++) {
        nonZeroBits += sci_detail::BitCount(repr[i]);
    }
    mask = (one << (end % 64)) - one;

    result = repr[end / 64] & mask;
    nonZeroBits += sci_detail::BitCount(result);

    parity *= (-2. * (nonZeroBits % 2) + 1);
    if (getocc(start))
        parity *= -1.;
}

} // namespace block2
