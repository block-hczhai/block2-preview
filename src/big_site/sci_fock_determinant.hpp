
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
 *
 * Created by larsson on 1/4/19.
 *
 * Developed by Sandeep Sharma with contributions from James E. T. Smith and
 * Adam A. Holmes, 2017
 *
 * Copyright (C) 2017 Sandeep Sharma
 *
 * Modified and adapted by Henrik R. Larsson, 2019.
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

/** Adapted from sciblock2/sci/fragmentDeterminant.hpp
 * by Huanchen Zhai Jul 18, 2021.
 *
 * Author: Henrik R. Larsson <larsson@caltech.edu>
 */

#pragma once

#include "sci_fcidump.hpp"
#include <cassert>

namespace block2 {

namespace sci_detail {

constexpr int fragmentDetLen = 14;
static_assert(fragmentDetLen % 2 == 0 and fragmentDetLen > 0,
              "fragmentDetLen must be even!");

inline int BitCount(long x) {
    x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
    return (x * 0x0101010101010101ULL) >> 56;

    // unsigned int u2=u>>32, u1=u;

    // return __builtin_popcount(u2)+__builtin_popcount(u);
    /*
    u1 = u1
      - ((u1 >> 1) & 033333333333)
      - ((u1 >> 2) & 011111111111);


    u2 = u2
      - ((u2 >> 1) & 033333333333)
      - ((u2 >> 2) & 011111111111);

    return (((u1 + (u1 >> 3))
         & 030707070707) % 63) +
      (((u2 + (u2 >> 3))
        & 030707070707) % 63);
    */
}

inline int ffsl(long x) {
#ifdef __GNUC__
    return __builtin_ffsl(x);
#else
    throw runtime_error("ffsl not supported!");
    return 0;
#endif
}

} // namespace sci_detail

/** Essentially a slightly modified and slimmed version of Determinant to
 * represent determinants of fragments. */
struct SCIFockDeterminant {
  public:
    // Here, the variables are not static as norbs etc differ for each fragment!
    int norbs = -99;     //!< Number of spin orbitals
    int nAlphaEl = -99;  //!< Number of alpha electrons
    int nBetaEl = -99;   //!< Number of beta electrons
    int EffDetLen = -99; //!< Effective length of determinant: How many entries
                         //!< of repr are really used.

    // 0th position of 0th long is the first position
    // 63rd position of the last long is the last position
    std::array<long, sci_detail::fragmentDetLen>
        repr; //!< 0th position of 0th long is the first position
              //!! 63rd position of the last long is the last position
    static int getEffDetLen(const int norbs_) { return norbs_ / 64 + 1; }

    SCIFockDeterminant() { repr.fill(0); }
    template <typename ReprVec>
    SCIFockDeterminant(const int norbs_, const ReprVec &reprVec)
        : norbs{norbs_}, EffDetLen{getEffDetLen(norbs_)} {
        for (int i = 0; i < sci_detail::fragmentDetLen; ++i) {
            repr[i] = reprVec[i];
        }
        nAlphaEl = Nalpha();
        nBetaEl = Nbeta();
        assert(consistencyCheck());
    }

    SCIFockDeterminant(const int norbs_, const int nAlphaEl_,
                       const int nBetaEl_)
        : norbs{norbs_}, nAlphaEl{nAlphaEl_}, nBetaEl{nBetaEl_},
          EffDetLen{getEffDetLen(norbs_)} {
        //       vvv Actually, I (mis-)use fragmentDet also so I put this in
        //       assertion (for avocado code)
        assert((norbs == 1 or (norbs % 2) == 0) and
               "norbs must be even! We have spin orbitals");
        assert(EffDetLen <= sci_detail::fragmentDetLen and
               "recompile with larger fragmentDetLen!");
        repr.fill(0);
    }
    SCIFockDeterminant(const int norbs_, const std::vector<int> &nOccupied)
        : //      ATTENTION:                                        vv vvv dirty
          //      hack => only works for assert below!
          SCIFockDeterminant{norbs_, (int)(nOccupied.size()), -99} {
        assert((norbs % 2 == 0 or norbs == 1) and
               "this one does ONLY works for 'spatial' orbitals: even (odd) "
               "orbital number is alpha (beta) el");
        for (const auto &i : nOccupied) {
            setocc(i, true);
        }
        nAlphaEl = Nalpha();
        nBetaEl = Nbeta();
    }

    int nEl() const { return nAlphaEl + nBetaEl; }

    bool consistencyCheck() const {
        if (EffDetLen > sci_detail::fragmentDetLen)
            throw std::invalid_argument("SCIFockDeterminant::consistencyCheck:"
                                        " Change fragmentDetLen to " +
                                        Parsing::to_string(EffDetLen) +
                                        " and recompile.");
        if (nEl() != Noccupied())
            throw std::runtime_error(
                "SCIFockDeterminant::consistencyCheck: nel=" +
                Parsing::to_string(nEl()) +
                "but Noccupied()=" + Parsing::to_string(Noccupied()) + "for" +
                Parsing::to_string(*this));
        if (nAlphaEl != Nalpha())
            throw std::runtime_error(
                "SCIFockDeterminant::consistencyCheck: nAlphaEl=" +
                Parsing::to_string(nAlphaEl) +
                "but Nalpha()=" + Parsing::to_string(Nalpha()) + "for" +
                Parsing::to_string(*this));
        return true;
    }

    std::size_t getHash() const noexcept {
        // This is what is used in boost/container_hash/hash.hpp hash_combine;
        // see also https://stackoverflow.com/a/27216842
        // TODO: All info is in repr. So why do I hash norbs etc?
        auto seed = (std::size_t)(norbs);
        seed ^=
            (std::size_t)(nAlphaEl) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= (std::size_t)(nBetaEl) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        // I want to avoid computing and storing the lexical order for all the
        // different determinant types (nel, norb)
        for (int i = 0; i < EffDetLen; i++) {
            seed ^= repr[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

    /** Number of electrons this determinant represents. */
    int Noccupied() const {
        int nelec = 0;
        for (int i = 0; i < EffDetLen; ++i) {
            nelec += sci_detail::BitCount(repr[i]);
        }
        return nelec;
    }

    /** Number of alpha electrons this determinant represents. */
    int Nalpha() const {
        int nalpha = 0;
        constexpr long alleven = 0x5555555555555555;
        for (int i = 0; i < EffDetLen; ++i) {
            long even = repr[i] & alleven;
            nalpha += sci_detail::BitCount(even);
        }
        return nalpha;
    }

    /** Number of beta electrons this determinant represents. */
    int Nbeta() const {
        int nbeta = 0;
        constexpr long allodd = 0xAAAAAAAAAAAAAAAA;
        for (int i = 0; i < EffDetLen; ++i) {
            long odd = repr[i] & allodd;
            nbeta += sci_detail::BitCount(odd);
        }
        return nbeta;
    }

    /** The comparison between determinants is performed.*/
    bool operator<(const SCIFockDeterminant &d) const {
        assert(d.EffDetLen == EffDetLen);
        assert(consistencyCheck());
        assert(d.consistencyCheck());
        // First, sort according to nAlpha/beta el.
        // If both alpha and beta el are identical,
        // sort according to the representation
        if (nAlphaEl < d.nAlphaEl) {
            return true;
        } else if (nAlphaEl > d.nAlphaEl) {
            return false;
        } else if (nBetaEl < d.nBetaEl) {
            return true;
        } else if (nBetaEl > d.nBetaEl) {
            return false;
        }
        assert(nAlphaEl == d.nAlphaEl and nBetaEl == d.nBetaEl);
        // ATTENTION: Just this vv does *not* according to alpha/beta el!
        for (int i = EffDetLen - 1; i >= 0; i--) {
            if (i > 0) {
                assert(repr[i] == 0 and d.repr[i] == 0);
            }
            if (repr[i] < d.repr[i])
                return true;
            else if (repr[i] > d.repr[i])
                return false;
        }
        return false;
    }
    /** Checks if the determinants are equal. */
    bool operator==(const SCIFockDeterminant &d) const {
        assert(consistencyCheck());
        assert(d.consistencyCheck());
        if (nAlphaEl != d.nAlphaEl) {
            return false;
        }
        if (nBetaEl != d.nBetaEl) {
            return false;
        }
        return repr == d.repr;
    }

    /** Sets the occupation of the ith orbital. */
    void setocc(const int i, const bool occ) {
        // assert(i< norbs and i >= 0);
        long Integer = i / 64, bit = i % 64;
        constexpr long one = 1;
        assert(Integer < sci_detail::fragmentDetLen and Integer >= 0);
        if (occ) {
            repr[Integer] |= one << bit;
        } else {
            repr[Integer] &= ~(one << bit);
        }
    }

    /** Gets the occupation of the ith orbital. */
    bool getocc(const int i) const {
        // assert(i<norbs and i >= 0);
        assert(i / 64 < sci_detail::fragmentDetLen and i >= 0);
        // TODO HRL: See JCP, 149, 214110: n mod 64 == n & 63 and n / 64 == n >>
        // 6
        //      accordingly, it should be
        //            long Integer = i >> 6;
        //            long bit = i & 63;
        //             => apparently, g++ does NOT optimize it in such a away
        //      so test whether explicit shifiting and &ing does help.
        //     => less ASM calls!
        long Integer = i / 64, bit = i % 64, reprBit = repr[Integer];
        return ((reprBit >> bit & 1) == 0) ? false : true;
    }
    void parity(const int start, const int end, double &parity) const;

    /** The represenation where each index represents an orbital. */
    std::vector<short> getRepArrayVec() const {
        std::vector<short> repArray(norbs);
        fillRepArrayVec(repArray);
        return repArray;
    }
    /** The represenation where each index represents an orbital. */
    template <typename arrType> void fillRepArrayVec(arrType &repArray) const {
        for (int i = 0; i < norbs; i++) {
            repArray[i] = getocc(i) ? 1 : 0;
        }
    }

    /** Prints the determinant. */
    friend std::ostream &operator<<(std::ostream &os,
                                    const SCIFockDeterminant &d) {
        const auto det = d.getRepArrayVec();
        assert((d.norbs % 2) == 0 and "Norbs needs to be even! Spin orbitals");
        for (int i = 0; i < d.norbs / 2; i++) {
            if (det[2 * i] == false and det[2 * i + 1] == false) {
                os << 0; //<<" ";
            } else if (det[2 * i] == true and det[2 * i + 1] == false) {
                os << "a"; //<<" ";
            } else if (det[2 * i] == false and det[2 * i + 1] == true) {
                os << "b"; //<<" ";
            } else if (det[2 * i] == true and det[2 * i + 1] == true) {
                os << 2; //<<" ";
            }
            if ((i + 1) % 5 == 0) {
                os << "  ";
            }
        }
        return os;
    }

    std::vector<int> getClosed() const {
        // TODO Short instead of int should be enough
        const auto nelec = nEl();
        std::vector<int> closed(nelec);
        int cindex = 0;
        for (int i = 0; i < norbs; ++i) {
            if (getocc(i)) {
                assert(cindex < closed.size());
                closed[cindex++] = i;
                if (cindex >= nelec) {
                    break;
                }
            }
        }
        assert(cindex == nelec);
        return closed;
    }
    double Energy(const SCIFCIDUMPOneInt &I1, const SCIFCIDUMPTwoInt &I2,
                  const std::vector<int> &closed) const;
    /** Calculate the Hamiltonian matrix element connecting determinants
     * connected by :math:`\Gamma = a^\dagger_a a_i`, i.e. single excitation.
     *
     * @param i Creation operator index.
     * @param a Destruction operator index.
     * @param I1 One body integrals.
     * @param I2 Two body integrals.
     * @return
     */
    double Hij_1Excite(const int i, const int a, const SCIFCIDUMPOneInt &I1,
                       const SCIFCIDUMPTwoInt &I2) const;
    /** Calculates the Hamiltonian matrix element connecting determinants
     * connected by :math:`\Gamma = a^\dagger_i a^\dagger_j a_b a_a`, i.e.
     * double excitation.
     *
     * @param i Creation operator index.
     * @param j Creation operator index.
     * @param a Destruction operator index.
     * @param b Destruction operator index.
     * @param I1 One body integrals.
     * @param I2 Two body integrals.
     */
    double Hij_2Excite(const int i, const int j, const int a, const int b,
                       const SCIFCIDUMPOneInt &I1,
                       const SCIFCIDUMPTwoInt &I2) const;

    /** Push_back in existing detVec. */
    static void
    generateFCIfragmentSpace(const int nElAlpha, const int nElBeta,
                             const int nSpinOrb,
                             std::vector<SCIFockDeterminant> &fragDetVec);
    static std::vector<SCIFockDeterminant>
    generateFCIfragmentSpace(const int nElAlpha, const int nElBeta,
                             const int nSpinOrb);

    /** Generates all FCI determinants *either* in Fock (bool) representation
     * (output.first) or in Hilbert (output.second) representation.*/
    static std::pair<std::vector<std::vector<bool>>,
                     std::vector<std::vector<int>>>
    generateFCIdeterminants(const int nOrb, const int nEl, const bool fockRep);

    /** Apply creator operator on orbital iOrb to determinant specified by
     * iClosed.
     *
     * @return Pair of phase and new determinant (closed orbitals); properly
     * sorted. If phase is 0, the result is 0
     */
    static std::pair<int, std::vector<int>>
    applyCreator(const std::vector<int> &iClosed, const int iOrb);
    /** Apply annihilation operator on orbital iOrb to determinant specified by
     * iClosed.
     *
     * @return Pair of phase and new determinant (closed orbitals); properly
     * sorted. If phase is 0, the result is 0
     */
    static std::pair<int, std::vector<int>>
    applyAnnihilator(const std::vector<int> &iClosed, const int iOrb);
};

} // namespace block2

namespace std {

template <> struct hash<block2::SCIFockDeterminant> {
    size_t operator()(const block2::SCIFockDeterminant &det) const noexcept {
        return det.getHash();
    }
};

} // namespace std
