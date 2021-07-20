
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

/** Adapted from sciblock2/sci/fcidumpWrapper.hpp
 * by Huanchen Zhai Jul 18, 2021.
 *
 * Author: Henrik R. Larsson <larsson@caltech.edu>
 */

/** Wrapper of block2::FCIDUMP to the SCI integral interface */

#pragma once

#include "../core/csr_sparse_matrix.hpp"
#include "../core/integral.hpp"
#include <algorithm>
#include <vector>

using namespace std;

namespace block2 {

namespace sci_detail {

/** Replacement of nutil::EigenRowMajorMatX / Eigen::MatrixXd. */
template <typename T> struct DenseMat {
    size_t m, n;
    vector<T> data;
    DenseMat(size_t m, size_t n) : m(m), n(n) { data.resize(m * n); }
    T &operator()(int i, int j) { return *(data.data() + (size_t)i * n + j); }
    const T &operator()(int i, int j) const {
        return *(data.data() + (size_t)i * n + j);
    }
};

/** Replacement of Eigen::SparseMatrix<double, Eigen::RowMajor, MKL_INT>. */
template <typename T> struct COOSparseMat {
    MKL_INT m, n;
    vector<pair<pair<MKL_INT, MKL_INT>, T>> data;
    COOSparseMat() {}
    void resize(MKL_INT m, MKL_INT n) { this->m = m, this->n = n; }
    void reserve(size_t n) { data.resize(n); }
    T &insert(MKL_INT i, MKL_INT j) {
        data.push_back(make_pair(make_pair(i, j), 0));
        return data.back().second;
    }
    void
    setFromTriplets(const vector<pair<pair<MKL_INT, MKL_INT>, T>> &tri_data) {
        data = tri_data;
    }
    void fillCSR(CSRMatrixRef &mat) {
        assert(m == mat.m);
        assert(n == mat.n);
        assert(mat.data == nullptr);
        assert(mat.alloc != nullptr);
        vector<size_t> idx(data.size()), idx2;
        for (size_t i = 0; i < data.size(); i++)
            idx[i] = i;

        sort(idx.begin(), idx.end(), [this](size_t i, size_t j) {
            return this->data[i].first < this->data[j].first;
        });
        for (auto ii : idx)
            if (idx2.empty() || data[ii].first != data[idx2.back()].first)
                idx2.push_back(ii);
            else
                data[idx2.back()].second += data[ii].second;
        mat.nnz = (MKL_INT)idx2.size();
        assert(mat.nnz != mat.size());
        mat.alloc = make_shared<VectorAllocator<double>>();
        mat.allocate();
        MKL_INT cur_row = -1;
        for (size_t k = 0; k < idx2.size(); k++) {
            while (data[idx2[k]].first.first != cur_row)
                mat.rows[++cur_row] = k;
            mat.data[k] = data[idx2[k]].second,
            mat.cols[k] = data[idx2[k]].first.second;
        }
        while (mat.m != cur_row)
            mat.rows[++cur_row] = mat.nnz;
    }
};

} // namespace sci_detail

struct SCIFCIDUMPTwoInt {
    std::shared_ptr<block2::FCIDUMP> fcidump;
    int nOrbOther, nOrbThis, nOrb; //!< *spatial* orbitals
    int nSOrbOther, nSOrbThis;     //!< *spin* orbitals
    sci_detail::DenseMat<double> Direct,
        Exchange; // Only for the external space! Size is #spatial orbitals
    SCIFCIDUMPTwoInt(const std::shared_ptr<block2::FCIDUMP> &fcidump,
                     int nOrbOther, int nOrbThis, bool isRight)
        : fcidump{fcidump}, nOrbOther{nOrbOther}, nOrbThis{nOrbThis},
          nOrb{nOrbOther + nOrbThis},
          nSOrbOther{2 * nOrbOther}, nSOrbThis{2 * nOrbThis},
          Direct(nOrbThis, nOrbThis), Exchange(nOrbThis, nOrbThis) {
        if (not isRight) {
            // Attention: Quick and dirty hack: nOrbOther only serve here as an
            // offset so just set it to 0
            this->nOrbOther = 0;
            this->nSOrbOther = 0;
        }
        // Calc direct and exchange
        for (int i = 0; i < nOrbThis; ++i) {
            for (int j = 0; j < nOrbThis; ++j) {
                Direct(i, j) = (*this)(2 * i, 2 * i, 2 * j, 2 * j);
                Exchange(i, j) = (*this)(2 * i, 2 * j, 2 * j, 2 * i);
            }
        }
    }
    /** Get integral value. i,j,k,l are *spin* orbitals *in the external space*.
     */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    operator()(int i, int j, int k, int l) const { // TODO always use offset
        if (not((i % 2 == j % 2) and (k % 2 == l % 2))) {
            return 0.0; // TODO avoid instances of this
        }
        i += nSOrbOther;
        j += nSOrbOther;
        k += nSOrbOther;
        l += nSOrbOther;
        return intsImpl(i, j, k, l);
    }
    /** Get integral value for R op. i,j,k,l are *spin* orbitals; jkl are *in
     * the external space*. */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    intsR(int i, int j, int k, int l) const {
        if (not((i % 2 == j % 2) and (k % 2 == l % 2))) {
            return 0.0; // TODO avoid instances of this
        }
        j += nSOrbOther;
        k += nSOrbOther;
        l += nSOrbOther;
        return intsImpl(i, j, k, l);
    }
    /** Get integral value for P op. i,j,k,l are *spin* orbitals; j,l are *in
     * the external space*. */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    intsP(int i, int j, int k, int l) const {
        if (not((i % 2 == j % 2) and (k % 2 == l % 2))) {
            return 0.0; // TODO avoid instances of this
        }
        j += nSOrbOther;
        l += nSOrbOther;
        return intsImpl(i, j, k, l);
    }
    /** Get integral value for Q op: This is [xy|kl] - [xl|ky]. x,y,k,l are
     * *spin* orbitals; k,l are *in the external space*. */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    intsQ(int x, int y, int k, int l) const {
        k += nSOrbOther;
        l += nSOrbOther;
        double sumi;
        if (not((x % 2 == y % 2) and (k % 2 == l % 2))) {
            sumi = 0.0;
        } else {
            sumi = intsImpl(x, y, k, l);
        }
        if ((x % 2 == l % 2) and (k % 2 == y % 2)) {
            sumi -= intsImpl(x, l, k, y);
        }
        return sumi;
    }

  private:
    /** Get integral value. i,j,k,l are *spin* orbitals *in the external space*.
     */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    intsImpl(const int i, const int j, const int k, const int l) const {
        assert((i % 2 == j % 2) and (k % 2 == l % 2));
        assert(i >= 0 and j >= 0 and k >= 0 and l >= 0);
        uint16_t I = i / 2;
        uint16_t J = j / 2;
        uint16_t K = k / 2;
        uint16_t L = l / 2;
        assert(I < nOrb and J < nOrb and K < nOrb and L < nOrb);
        uint8_t sl = i % 2;
        uint8_t sr = k % 2;
        assert(fcidump->data != nullptr);
        return fcidump->v(sl, sr, I, J, K, L);
    }
};

struct SCIFCIDUMPOneInt {
    std::shared_ptr<block2::FCIDUMP> fcidump;
    int nOrbOther, nOrbThis, nOrb; //!< *spatial* orbitals
    int nSOrbOther, nSOrbThis;     //!< *spin* orbitals
    SCIFCIDUMPOneInt(const std::shared_ptr<block2::FCIDUMP> &fcidump,
                     int nOrbOther, int nOrbThis, bool isRight)
        : fcidump{fcidump}, nOrbOther{nOrbOther}, nOrbThis{nOrbThis},
          nOrb{nOrbOther + nOrbThis},
          nSOrbOther{2 * nOrbOther}, nSOrbThis{2 * nOrbThis} {
        if (not isRight) {
            // Attention: Quick and dirty hack: nOrbOther only serve here as an
            // offset so just set it to 0
            this->nOrbOther = 0;
            this->nSOrbOther = 0;
        }
    }
    /** Get integral value. i,j are *spin* orbitals in *external* space. */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    operator()(const int i, const int j) const {
        if (i % 2 != j % 2) {
            return 0.0;
        }
        return intsImpl(i + nSOrbOther, j + nSOrbOther);
    }
    /** Get integral value. i,j are *spin* orbitals and j in *external* space.
     */
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    intsS(const int i, const int j) const {
        if (i % 2 != j % 2) {
            return 0.0;
        }
        return intsImpl(i, j + nSOrbOther);
    }

  private:
#ifdef __GNUC__
    [[gnu::always_inline, gnu::hot]]
#endif
    inline double
    intsImpl(const int i, const int j) const {
        assert(i % 2 == j % 2);
        uint16_t I = i / 2;
        uint16_t J = j / 2;
        assert(I < nOrb and J < nOrb);
        uint8_t s = i % 2;
        assert(fcidump->data != nullptr);
        return fcidump->t(s, I, J);
    }
};

} // namespace block2
