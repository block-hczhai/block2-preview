
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2021 Huanchen Zhai <hczhai@caltech.edu>
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

/** Dense algebra of n-dimensional array. */

#pragma once

#include "../core/threading.hpp"
#include "../core/utils.hpp"
#ifdef _HAS_INTEL_MKL
#include "mkl.h"
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

using namespace std;

extern "C" {

#ifndef _HAS_INTEL_MKL

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void FNAME(dgemm)(const char *transa, const char *transb,
                         const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                         const double *alpha, const double *a,
                         const MKL_INT *lda, const double *b,
                         const MKL_INT *ldb, const double *beta, double *c,
                         const MKL_INT *ldc) noexcept;

#endif
}

namespace block2 {

// step = 0, start = 0, stop = 0 : all
// step = start = stop = 0 : None
// step = 0, start = int, stop = -1 : int
struct NDArraySlice {
    static const MKL_INT invalid = numeric_limits<MKL_INT>::max();
    MKL_INT start, stop, step;
    NDArraySlice() : start(invalid), stop(invalid), step(invalid) {}
    NDArraySlice(MKL_INT start, MKL_INT stop, MKL_INT step = 1)
        : start(start), stop(stop), step(step) {}
    bool is_all() const {
        return start == invalid && stop == invalid && step == invalid;
    }
    bool is_int() const { return step == 0 && stop == -1; }
    bool is_none() const { return step == 0 && stop == 0 && start == 0; }
    friend ostream &operator<<(ostream &os, const NDArraySlice &sl) {
        if (sl.is_none())
            os << "None";
        else if (sl.is_int())
            os << sl.start;
        else if (sl.is_all())
            os << ":";
        else {
            (sl.start != invalid) && (os << sl.start);
            os << ":";
            (sl.stop != invalid) && (os << sl.stop);
            (sl.step != 1 && sl.step != invalid) && (os << ":" << sl.step);
        }
        return os;
    }
    static vector<NDArraySlice> parse(const string &x) {
        vector<string> px = Parsing::split(x, ",", false);
        vector<NDArraySlice> r;
        for (auto &ppx : px) {
            ppx = Parsing::trim(ppx);
            if (ppx == "None")
                r.push_back(NDArraySlice(0, 0, 0));
            else {
                auto x = Parsing::split(ppx, ":", false);
                if (x.size() == 1)
                    r.push_back(NDArraySlice(Parsing::to_int(x[0]), -1, 0));
                else if (x.size() == 2)
                    r.push_back(NDArraySlice(
                        x[0].length() == 0 ? invalid : Parsing::to_int(x[0]),
                        x[1].length() == 0 ? invalid : Parsing::to_int(x[1]),
                        1));
                else
                    r.push_back(NDArraySlice(
                        x[0].length() == 0 ? invalid : Parsing::to_int(x[0]),
                        x[1].length() == 0 ? invalid : Parsing::to_int(x[1]),
                        x[2].length() == 0 ? invalid : Parsing::to_int(x[2])));
            }
        }
        return r;
    }
};

struct NDArray {
    shared_ptr<vector<double>> vdata;
    vector<MKL_INT> shape;
    vector<ssize_t> strides;
    double *data;
    NDArray() : NDArray(vector<MKL_INT>{}) {}
    NDArray(const vector<MKL_INT> &shape, const vector<ssize_t> &strides)
        : shape(shape), strides(strides) {
        vdata = make_shared<vector<double>>(size());
        data = vdata->data();
    }
    NDArray(const vector<MKL_INT> &shape) : shape(shape), strides() {
        strides.resize(shape.size());
        ssize_t cur = 1;
        for (int i = ndim() - 1; i >= 0; cur *= shape[i--])
            strides[i] = cur;
        vdata = make_shared<vector<double>>(size());
        data = vdata->data();
    }
    NDArray(const vector<MKL_INT> &shape, const vector<ssize_t> &strides,
            double *data)
        : shape(shape), strides(strides), data(data), vdata(nullptr) {}
    NDArray(const vector<MKL_INT> &shape, const vector<double> &xdata)
        : NDArray(shape) {
        memcpy(data, xdata.data(), sizeof(double) * xdata.size());
    }
    static NDArray random(const vector<MKL_INT> &shape) {
        NDArray r(shape);
        Random::fill<double>(r.data, r.size());
        return r;
    }
    int ndim() const { return (int)shape.size(); }
    size_t size() const {
        return accumulate(shape.cbegin(), shape.cend(), 1,
                          multiplies<size_t>());
    }
    size_t max_size() const {
        size_t cur = 0;
        for (int i = 0; i < ndim(); i++)
            if (strides[i] >= 0)
                cur += (shape[i] - 1) * strides[i];
        return cur + 1;
    }
    vector<MKL_INT> decompose_linear_index(size_t i) const {
        vector<MKL_INT> x(ndim());
        for (int k = ndim() - 1; k >= 0; i /= shape[k], k--)
            x[k] = i % shape[k];
        return x;
    }
    double operator[](const vector<MKL_INT> &idx) const {
        ssize_t j = 0;
        for (int k = ndim() - 1; k >= 0; k--)
            j += strides[k] * idx[k];
        return data[j];
    }
    ssize_t linear_index(size_t i) const {
        ssize_t j = 0;
        for (int k = ndim() - 1; k >= 0; i /= shape[k], k--)
            j += strides[k] * (i % shape[k]);
        return j;
    }
    friend ostream &operator<<(ostream &os, const NDArray &arr) {
        size_t sz = arr.size();
        os << "NDARR :: SIZE = " << sz << " [ ";
        for (auto &sh : arr.shape)
            os << sh << " ";
        os << "] STRIDES = [ ";
        for (auto &sh : arr.strides)
            os << sh << " ";
        os << "] " << endl;
        const size_t sz2 = arr.shape.size() >= 2 ? arr.shape.back() : 0;
        const size_t sz3 =
            arr.shape.size() >= 3 ? arr.shape[arr.shape.size() - 2] * sz2 : 0;
        for (size_t i = 0; i < sz; i++) {
            os << fixed << setprecision(10) << setw(16)
               << arr.data[arr.linear_index(i)];
            if (sz2 && (i + 1) % sz2 == 0) {
                os << endl;
                if (sz3 && (i + 1) % sz3 == 0)
                    os << endl;
            }
        }
        return os;
    }
    bool is_c_order() const {
        size_t cur = 1;
        for (int i = ndim() - 1; i >= 0; cur *= shape[i--])
            if (strides[i] != cur && strides[i] != 0)
                return false;
        return true;
    }
    NDArray reorder_c(vector<int> &idx) const {
        int dim = ndim();
        idx.resize(dim);
        for (int i = 0; i < dim; i++)
            idx[i] = i;
        sort(idx.begin(), idx.end(),
             [this](int i, int j) { return strides[i] > strides[j]; });
        vector<MKL_INT> new_shape(dim);
        vector<ssize_t> new_strides(dim);
        for (int i = 0; i < dim; i++) {
            new_shape[i] = shape[idx[i]];
            new_strides[i] = strides[idx[i]];
        }
        NDArray r(new_shape, new_strides, data);
        r.vdata = vdata;
        return r;
    }
    NDArray slice(const vector<NDArraySlice> &idxs) const {
        ssize_t offset = 0;
        int k = ndim();
        vector<MKL_INT> new_shape;
        vector<ssize_t> new_strides;
        vector<NDArraySlice> pidxs = idxs;
        int n_slice = 0;
        for (auto &x : pidxs)
            if (!x.is_none())
                n_slice++;
        while (n_slice < k)
            pidxs.push_back(NDArraySlice()), n_slice++;
        for (int i = 0, j = 0; i < (int)pidxs.size(); i++) {
            NDArraySlice x = pidxs[i];
            if (x.is_int()) {
                if (x.start == NDArraySlice::invalid)
                    x.start = 0;
                offset += strides[j++] * x.start;
            } else if (x.is_none())
                new_shape.push_back(1), new_strides.push_back(0);
            else {
                if (x.start == NDArraySlice::invalid)
                    x.start = 0;
                else if (x.start < 0)
                    x.start += shape[j];
                if (x.stop == NDArraySlice::invalid)
                    x.stop = shape[j];
                else if (x.stop < 0)
                    x.stop += shape[j];
                if (x.step == NDArraySlice::invalid)
                    x.step = 1;
                if ((x.step > 0 && x.stop <= x.start) ||
                    (x.step < 0 && x.stop >= x.start))
                    new_shape.push_back(0), new_strides.push_back(0);
                else {
                    offset += strides[j] * x.start;
                    new_shape.push_back(
                        abs(x.stop - x.start) / abs(x.step) +
                        !!(abs(x.stop - x.start) % abs(x.step)));
                    new_strides.push_back(strides[j] * x.step);
                }
                j++;
            }
        }
        NDArray r(new_shape, new_strides, data + offset);
        r.vdata = vdata;
        return r;
    }
    // diag (no copy)
    NDArray diag(const vector<int> &perm) const {
        vector<int> idx_map(perm.size(), -1);
        int k = 0;
        for (int i = 0; i < (int)perm.size(); i++)
            if (idx_map[perm[i]] == -1)
                idx_map[perm[i]] = k++;
        vector<MKL_INT> new_shape(k, -1);
        vector<ssize_t> new_strides(k, 0);
        for (int i = 0; i < (int)perm.size(); i++) {
            assert(new_shape[idx_map[perm[i]]] == -1 ||
                   new_shape[idx_map[perm[i]]] == shape[i]);
            new_shape[idx_map[perm[i]]] = shape[i];
            new_strides[idx_map[perm[i]]] += strides[i];
        }
        NDArray r(new_shape, new_strides, data);
        r.vdata = vdata;
        return r;
    }
    // sum the right indices
    NDArray sum(int idx_at) const {
        assert(is_c_order());
        vector<MKL_INT> new_shape(shape.begin(), shape.begin() + idx_at);
        size_t size_right = 1;
        const int dim = ndim();
        for (int i = idx_at; i < dim; i++)
            size_right *= shape[i];
        size_t size_left = size() / size_right;
        NDArray r(new_shape);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (size_t i = 0; i < size_left; i++) {
            const double *__restrict__ a_data = data + i * size_right;
            double rr = 0;
            for (size_t j = 0; j < size_right; j++)
                rr += a_data[j];
            r.data[i] = rr;
        }
        threading->activate_normal();
        return r;
    }
    // broadcast (no copy)
    NDArray broadcast_to(int new_ndim,
                         const vector<int> &orig_idxs = {}) const {
        int dim = ndim();
        vector<MKL_INT> new_shape(new_ndim, 1);
        vector<ssize_t> new_strides(new_ndim, 0);
        if (orig_idxs.size() == 0 && dim != 0)
            for (int i = 0; i < dim; i++) {
                new_shape[i + new_ndim - dim] = shape[i];
                new_strides[i + new_ndim - dim] = strides[i];
            }
        else
            for (int i = 0; i < dim; i++) {
                new_shape[orig_idxs[i]] = shape[i];
                new_strides[orig_idxs[i]] = strides[i];
            }
        NDArray r(new_shape, new_strides, data);
        r.vdata = vdata;
        return r;
    }
    double item() const {
        assert(size() == 1);
        return data[0];
    }
    double norm() const {
        NDArray r;
        vector<int> idx(ndim());
        for (int i = 0; i < ndim(); i++)
            idx[i] = i;
        tensordot(*this, *this, r, idx, idx);
        return sqrt(r.item());
    }
    NDArray to_c_order() const {
        if (is_c_order())
            return *this;
        NDArray r(shape);
        NDArray::transpose(*this, r);
        return r;
    }
    NDArray operator-() const {
        NDArray r(shape, strides);
        size_t sz = size();
        for (size_t i = 0; i < sz; i++)
            r.data[i] = -data[linear_index(i)];
        return r;
    }
    NDArray operator-(const NDArray &other) const { return *this + (-other); }
    NDArray operator+(const NDArray &other) const {
        assert(ndim() == other.ndim());
        int dim = ndim();
        bool same_stride = true;
        NDArray r(shape, strides);
        for (int i = 0; i < dim; i++) {
            if (strides[i] != 0 && other.strides[i] != 0) {
                assert(shape[i] == other.shape[i]);
                same_stride = same_stride && strides[i] == other.strides[i];
            } else if (strides[i] != 0)
                same_stride = false;
            else if (other.strides[i] != 0) {
                r.shape[i] = other.shape[i];
                r.strides[i] = other.strides[i];
                same_stride = false;
            }
        }
        if (same_stride) {
            // const double x = 1.0;
            // MKL_INT n = (MKL_INT)r.size(), inc = 1;
            // memcpy(r.data, data, sizeof(double) * n);
            // FNAME(dgemm)("N", "N", &inc, &n, &inc, &x, &x, &inc, other.data,
            // &inc, &x, r.data, &inc);
            const size_t r_size = r.size();
            for (size_t i = 0; i < r_size; i++) {
                size_t ii = linear_index(i);
                r.data[ii] = data[ii] + other.data[ii];
            }
        } else {
            size_t cur = 1;
            for (int i = r.ndim() - 1; i >= 0; cur *= r.shape[i--])
                if (r.strides[i] != 0)
                    r.strides[i] = cur;
            transpose(*this, r, {}, 1.0, 0.0);
            transpose(other, r, {}, 1.0, 1.0);
        }
        return r;
    }
    // transpose (no copy)
    NDArray transpose(const vector<int> &perm) const {
        int dim = ndim();
        vector<MKL_INT> new_shape(dim);
        vector<ssize_t> new_strides(dim);
        for (int i = 0; i < (int)perm.size(); i++) {
            new_shape[i] = shape[perm[i]];
            new_strides[i] = strides[perm[i]];
        }
        NDArray r(new_shape, new_strides, data);
        r.vdata = vdata;
        return r;
    }
    // b must be C order (modulo permutations) (always copy)
    static void transpose(const NDArray &a, const NDArray &b,
                          const vector<int> &perm = {}, double alpha = 1.0,
                          double beta = 0.0) {
        const int dim = a.ndim();
        vector<int> idx, xperm(perm.size());
        NDArray bx = b.reorder_c(idx);
        for (int i = 0; i < perm.size(); i++)
            xperm[i] = perm[idx[i]];
        assert(bx.is_c_order());
        if (xperm.size() == 0) {
            xperm.reserve(dim);
            for (int i = 0; i < dim; i++)
                xperm.push_back(idx[i]);
        }
        size_t size_left = 1;
        for (int i = 0; i < dim; i++)
            if (i != xperm.back())
                size_left *= a.shape[i];
        const size_t size_right = a.shape[xperm.back()];
        const ssize_t stride_right = a.strides[xperm.back()];
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (size_t j = 0; j < size_left; j++) {
            ssize_t offset_a = 0, jx = j;
            for (int i = dim - 2; i >= 0; jx /= bx.shape[i--])
                offset_a += ((ssize_t)jx % bx.shape[i]) * a.strides[xperm[i]];
            // const double x = 1.0;
            // MKL_INT n = (MKL_INT)size_right, incb = 1,
            //         inca = (MKL_INT)stride_right;
            // FNAME(dgemm)("N", "N", &incb, &n, &incb, &alpha, &x, &incb,
            //       a.data + offset_a, &inca, &beta, bx.data + j * size_right,
            //       &incb);
            const double *__restrict__ a_data = a.data + offset_a;
            double *__restrict__ b_data = bx.data + j * size_right;
            if (beta == 0.0) {
                for (size_t i = 0; i < size_right; i++)
                    b_data[i] = alpha * a_data[i * stride_right];
            } else
                for (size_t i = 0; i < size_right; i++)
                    b_data[i] =
                        alpha * a_data[i * stride_right] + beta * b_data[i];
        }
        threading->activate_normal();
    }
    // outpur br is sorted original br idx in a
    static void tensordot(const NDArray &a, const NDArray &b, NDArray &c,
                          const vector<int> &idxa, const vector<int> &idxb,
                          const vector<int> &br_idxa = {},
                          const vector<int> &br_idxb = {}, double alpha = 1.0,
                          double beta = 0.0) {
        assert(idxa.size() == idxb.size());
        assert(br_idxa.size() == br_idxb.size());
        int nctr = (int)idxa.size(), nbr = (int)br_idxa.size();
        int ndima = a.ndim(), ndimb = b.ndim();
        vector<int> idx, ridx, idxax(idxa.size() + br_idxa.size()),
            idxbx(idxb.size() + br_idxb.size());
        NDArray ax = a.reorder_c(idx);
        ridx.resize(idx.size());
        for (int i = 0; i < ndima; i++)
            ridx[idx[i]] = i;
        for (int i = 0; i < nbr; i++)
            idxax[i] = ridx[br_idxa[i]];
        for (int i = 0; i < nctr; i++)
            idxax[i + nbr] = ridx[idxa[i]];
        NDArray bx = b.reorder_c(idx);
        ridx.resize(idx.size());
        for (int i = 0; i < ndimb; i++)
            ridx[idx[i]] = i;
        for (int i = 0; i < nbr; i++)
            idxbx[i] = ridx[br_idxb[i]];
        for (int i = 0; i < nctr; i++)
            idxbx[i + nbr] = ridx[idxb[i]];
        assert(c.is_c_order());
        int outa[ndima - nctr - nbr], outb[ndimb - nctr - nbr];
        MKL_INT a_free_dim = 1, b_free_dim = 1, ctr_dim = 1, br_dim = 1;
        set<int> idxa_set(idxax.begin(), idxax.end());
        set<int> idxb_set(idxbx.begin(), idxbx.end());
        for (int i = 0, ioa = 0; i < ndima; i++)
            if (!idxa_set.count(i))
                outa[ioa] = i, a_free_dim *= ax.shape[i], ioa++;
        for (int i = 0, iob = 0; i < ndimb; i++)
            if (!idxb_set.count(i))
                outb[iob] = i, b_free_dim *= bx.shape[i], iob++;
        int trans_a = 0, trans_b = 0;

        int ctr_idx[nctr + nbr];
        for (int i = 0; i < nbr; i++)
            ctr_idx[i] = i, br_dim *= ax.shape[idxax[i]];
        for (int i = nbr; i < nbr + nctr; i++)
            ctr_idx[i] = i, ctr_dim *= ax.shape[idxax[i]];
        sort(ctr_idx, ctr_idx + nbr,
             [&idxax](int a, int b) { return idxax[a] < idxax[b]; });
        sort(ctr_idx + nbr, ctr_idx + nbr + nctr,
             [&idxax](int a, int b) { return idxax[a] < idxax[b]; });

        // checking whether permutation is necessary
        if (!ax.is_c_order())
            trans_a = 0;
        else if (nbr != 0 &&
                 (idxax[ctr_idx[0]] != 0 || idxax[ctr_idx[nbr - 1]] != nbr - 1))
            trans_a = 0;
        else if (nctr == 0)
            trans_a = 1;
        else if (idxax[ctr_idx[nbr]] == nbr &&
                 idxax[ctr_idx[nbr + nctr - 1]] == nbr + nctr - 1)
            trans_a = 1;
        else if (idxax[ctr_idx[nbr]] == ndima - nctr &&
                 idxax[ctr_idx[nbr + nctr - 1]] == ndima - 1)
            trans_a = -1;

        if (!bx.is_c_order())
            trans_b = 0;
        else if (nbr != 0 && !(idxbx[ctr_idx[0]] == 0 &&
                               idxbx[ctr_idx[nbr - 1]] == nbr - 1))
            trans_b = 0;
        else if (nctr == 0)
            trans_b = 1;
        else if (idxbx[ctr_idx[nbr]] == nbr &&
                 idxbx[ctr_idx[nbr + nctr - 1]] == nbr + nctr - 1)
            trans_b = 1;
        else if (idxbx[ctr_idx[nbr]] == ndimb - nctr &&
                 idxbx[ctr_idx[nbr + nctr - 1]] == ndimb - 1)
            trans_b = -1;

        // permute or reshape
        if (trans_a == 0) {
            vector<int> perm_a(ndima);
            vector<MKL_INT> new_shape_a(ndima);
            for (int i = 0; i < nbr + nctr; i++)
                perm_a[i] = idxax[ctr_idx[i]];
            for (int i = nbr + nctr; i < ndima; i++)
                perm_a[i] = outa[i - nctr - nbr];
            for (int i = 0; i < ndima; i++)
                new_shape_a[i] = ax.shape[perm_a[i]];
            NDArray axx(new_shape_a);
            transpose(ax, axx, perm_a);
            ax = axx;
            trans_a = 1;
        }

        if (trans_b == 0) {
            vector<int> perm_b(ndimb);
            vector<MKL_INT> new_shape_b(ndimb);
            for (int i = 0; i < nbr + nctr; i++)
                perm_b[i] = idxbx[ctr_idx[i]];
            for (int i = nbr + nctr; i < ndimb; i++)
                perm_b[i] = outb[i - nctr - nbr];
            for (int i = 0; i < ndimb; i++)
                new_shape_b[i] = bx.shape[perm_b[i]];
            NDArray bxx(new_shape_b);
            transpose(bx, bxx, perm_b);
            bx = bxx;
            trans_b = 1;
        }

        const MKL_INT ldb = trans_b == 1 ? b_free_dim : ctr_dim;
        const MKL_INT lda = trans_a == -1 ? ctr_dim : a_free_dim;
        const MKL_INT ldc = b_free_dim;

        if (nbr == 0) {
            threading->activate_global_mkl();
            FNAME(dgemm)
            (trans_b == 1 ? "n" : "t", trans_a == -1 ? "n" : "t", &b_free_dim,
             &a_free_dim, &ctr_dim, &alpha, bx.data, &ldb, ax.data, &lda, &beta,
             c.data, &ldc);
            threading->activate_normal();
        } else {
            const ssize_t stride_a = ax.strides[nbr - 1];
            const ssize_t stride_b = bx.strides[nbr - 1];
            const ssize_t stride_c = c.strides[nbr - 1];
            int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
            for (MKL_INT ibr = 0; ibr < br_dim; ibr++)
                FNAME(dgemm)
                (trans_b == 1 ? "n" : "t", trans_a == -1 ? "n" : "t",
                 &b_free_dim, &a_free_dim, &ctr_dim, &alpha,
                 bx.data + stride_b * ibr, &ldb, ax.data + stride_a * ibr, &lda,
                 &beta, c.data + stride_c * ibr, &ldc);
            threading->activate_normal();

            vector<int> perm_c(c.strides.size());
            for (int i = 0; i < (int)perm_c.size(); i++)
                perm_c[i] = i;
            sort(perm_c.data(), perm_c.data() + nbr,
                 [&idx, &idxax, &ctr_idx](int a, int b) {
                     return idx[idxax[ctr_idx[a]]] < idx[idxax[ctr_idx[b]]];
                 });
            c = c.transpose(perm_c);
        }
    }
    static NDArray einsum(const string &script, const vector<NDArray> &arrs) {
        // explicit mode has '->'
        bool explicit_mode = false;
        string result;
        vector<string> operands;
        int idx = 0;
        for (int i = 0; i < script.length(); i++)
            if (script[i] == ',') {
                operands.push_back(script.substr(idx, i - idx));
                operands.back() = Parsing::trim(operands.back());
                idx = i + 1;
            } else if (i < script.length() - 1 && script[i] == '-' &&
                       script[i + 1] == '>') {
                operands.push_back(script.substr(idx, i - idx));
                operands.back() = Parsing::trim(operands.back());
                idx = i + 2;
                explicit_mode = true;
            }
        if (explicit_mode) {
            result = script.substr(idx);
            result = Parsing::trim(result);
        } else {
            operands.push_back(script.substr(idx));
            operands.back() = Parsing::trim(operands.back());
        }
        if (operands.size() != arrs.size())
            throw runtime_error(
                "number of scripts " + to_string(operands.size()) +
                " does match number of arrays " + to_string(arrs.size()));
        // first pass
        const int _ELLIP = 1, _EMPTY = 2, _MAX_CHAR = 256;
        int char_map[_MAX_CHAR], char_count[_MAX_CHAR];
        // feature of operator
        // 1 = has_ellipsis; 2 = has_empty
        vector<int> op_features(operands.size() + 1, 0);
        idx = 0;
        memset(char_map, -1, sizeof(int) * _MAX_CHAR);
        memset(char_count, 0, sizeof(int) * _MAX_CHAR);
        // empty characters
        char_map['\t'] = char_map[' '] = char_map['\n'] = char_map['\r'] = -3;
        // illegal characters
        char_map['-'] = char_map['.'] = char_map['>'] = char_map['\0'] = -2;
        // first reserved character for ellipsis
        char nxt = '!';
        string ellip = "";
        bool ellip_determined = false;
        for (int iop = 0; iop < operands.size(); iop++) {
            int iellip = -1;
            for (int j = 0; j < operands[iop].length(); j++)
                if (j < operands[iop].length() - 2 && operands[iop][j] == '.' &&
                    operands[iop][j + 1] == '.' &&
                    operands[iop][j + 2] == '.') {
                    if (op_features[iop] & _ELLIP)
                        throw runtime_error(
                            "Multiple ellipses found in script " +
                            operands[iop]);
                    iellip = j;
                    j += 2;
                    op_features[iop] |= _ELLIP;
                } else if (char_map[operands[iop][j]] == -3)
                    op_features[iop] |= _EMPTY;
                else if (char_map[operands[iop][j]] == -2)
                    throw runtime_error("Illegal character " +
                                        string(1, operands[iop][j]) +
                                        " found in script " + operands[iop]);
                else {
                    if (char_map[operands[iop][j]] == -1)
                        char_map[operands[iop][j]] = idx++;
                    char_count[operands[iop][j]]++;
                }
            // remove empty characters inside script
            if (op_features[iop] & _EMPTY) {
                stringstream ss;
                for (int j = 0; j < operands[iop].length(); j++)
                    if (char_map[operands[iop][j]] != -3)
                        ss << operands[iop][j];
                operands[iop] = ss.str();
                op_features[iop] ^= _EMPTY;
            }
            // handle ellipses of operands
            if (op_features[iop] & _ELLIP) {
                int nchar = arrs[iop].ndim() - (operands[iop].length() - 3);
                if (!ellip_determined) {
                    stringstream ss;
                    for (int j = 0; j < nchar; j++) {
                        while (char_map[nxt] != -1)
                            nxt++;
                        char_map[nxt] = -2;
                        ss << nxt;
                    }
                    ellip = ss.str();
                    ellip_determined = true;
                }
                if (nchar != ellip.length())
                    throw runtime_error(
                        "Length of ellipses does not match in " +
                        operands[iop]);
                operands[iop] = operands[iop].replace(iellip, 3, ellip);
                op_features[iop] ^= _ELLIP;
            }
        }
        if (!explicit_mode) {
            // handle implicit mode
            stringstream ss;
            // if there is ellipsis, put it before all other indices
            if (ellip_determined)
                ss << ellip;
            // for all other indices that appearing only once, put them
            // according to alphabet
            // do not use char type here, as it may exceed _MAX_CHAR
            for (int k = 0; k < _MAX_CHAR; k++)
                if (char_count[k] == 1)
                    ss << (char)k;
            result = ss.str();
        } else {
            // handle ellipsis / empty in result script in explicit mode
            stringstream ss;
            for (int j = 0; j < result.length(); j++)
                if (j < result.length() - 2 && result[j] == '.' &&
                    result[j + 1] == '.' && result[j + 2] == '.') {
                    if (op_features.back() & _ELLIP)
                        throw runtime_error(
                            "Multiple ellipses found in output script " +
                            result);
                    // it is okay ellipsis does not appear in any operands
                    // in that case it works as if there is no ellipsis in the
                    // result
                    ss << ellip;
                    j += 2;
                    op_features.back() |= _ELLIP;
                } else if (char_map[result[j]] == -4)
                    throw runtime_error("Repeated character " +
                                        string(1, result[j]) +
                                        " found in output script " + result);
                else if (char_map[result[j]] == -3)
                    continue;
                else if (char_map[result[j]] == -2)
                    throw runtime_error("Illegal character " +
                                        string(1, result[j]) +
                                        " found in script " + result);
                else if (char_count[result[j]] == 0)
                    throw runtime_error(
                        "character " + string(1, result[j]) +
                        " found in output script did not appear in an input");
                else {
                    ss << result[j];
                    char_map[result[j]] = -4;
                }
            result = ss.str();
        }
        // for (int iop = 0; iop < operands.size() - 1; iop++)
        //     cout << operands[iop] << ",";
        // cout << operands.back() << "->" << result << endl;
        // allow possible reorder in future
        vector<string> gscripts = operands;
        vector<NDArray> garrs = arrs;
        // now char_count representes the count of each index
        memset(char_count, 0, sizeof(int) * _MAX_CHAR);
        for (int i = 0; i < (int)gscripts.size(); i++)
            for (int j = 0; j < gscripts[i].length(); j++)
                char_count[gscripts[i][j]]++;
        for (int j = 0; j < (int)result.length(); j++)
            char_count[result[j]]++;
        vector<int> perm, sum_idx;
        // handle internal sum and repeated indices
        for (int i = 0; i < (int)gscripts.size(); i++) {
            memset(char_map, -1, sizeof(int) * _MAX_CHAR);
            perm.resize(gscripts[i].length());
            int k = 0;
            stringstream newss;
            for (int j = 0; j < gscripts[i].length(); j++) {
                if (char_map[gscripts[i][j]] == -1)
                    char_map[gscripts[i][j]] = k++, newss << gscripts[i][j];
                perm[j] = char_map[gscripts[i][j]];
            }
            memset(char_map, 0, sizeof(int) * _MAX_CHAR);
            for (int j = 0; j < gscripts[i].length(); j++)
                char_map[gscripts[i][j]]++;
            // handle repeated indices
            if (k < (int)gscripts[i].length()) {
                garrs[i] = garrs[i].diag(perm);
                gscripts[i] = newss.str();
                assert((int)gscripts[i].length() == k);
            }
            int sum_idx_num = 0;
            sum_idx.resize(k);
            memset(sum_idx.data(), 0, sizeof(int) * sum_idx.size());
            newss = stringstream();
            for (int j = 0; j < k; j++)
                if (char_map[gscripts[i][j]] == char_count[gscripts[i][j]])
                    sum_idx[j] = -1, sum_idx_num++;
                else
                    newss << gscripts[i][j];
            for (int j = 0; j < k; j++)
                if (char_map[gscripts[i][j]] > 1)
                    char_count[gscripts[i][j]] -= char_map[gscripts[i][j]] - 1;
            // handle internal sum
            if (sum_idx_num != 0) {
                perm.resize(k);
                for (int ii = 0, jj = 0, kk = k - sum_idx_num; ii < k; ii++)
                    if (sum_idx[ii] != -1)
                        perm[jj++] = ii;
                    else
                        perm[kk++] = ii;
                NDArray tmp = garrs[i].transpose(perm);
                if (!tmp.is_c_order()) {
                    NDArray tmp2(tmp.shape);
                    NDArray::transpose(tmp, tmp2);
                    tmp = tmp2;
                }
                garrs[i] = tmp.sum(k - sum_idx_num);
                gscripts[i] = newss.str();
            }
        }
        // perform tensordot
        vector<int> idxa, idxb, br_idxa, br_idxb;
        vector<MKL_INT> new_sh, new_br;
        for (int i = 1; i < (int)gscripts.size(); i++) {
            idxa.clear(), idxb.clear();
            br_idxa.clear(), br_idxb.clear();
            new_sh.clear(), new_br.clear();
            memset(char_map, 0, sizeof(int) * _MAX_CHAR);
            for (int j = 0; j < gscripts[0].length(); j++)
                char_map[gscripts[0][j]]++;
            for (int j = 0; j < gscripts[i].length(); j++)
                char_map[gscripts[i][j]]++;
            stringstream newss, newsr;
            for (int j = 0; j < gscripts[0].length(); j++)
                if (char_map[gscripts[0][j]] > 1) {
                    if (char_map[gscripts[0][j]] == char_count[gscripts[0][j]])
                        idxa.push_back(j);
                    else
                        br_idxa.push_back(j), newsr << gscripts[0][j],
                            new_br.push_back(garrs[0].shape[j]);
                } else
                    newss << gscripts[0][j],
                        new_sh.push_back(garrs[0].shape[j]);
            for (int j = 0; j < gscripts[i].length(); j++)
                if (char_map[gscripts[i][j]] > 1) {
                    if (char_map[gscripts[i][j]] == char_count[gscripts[i][j]])
                        idxb.push_back(j);
                    else
                        br_idxb.push_back(j);
                } else
                    newss << gscripts[i][j],
                        new_sh.push_back(garrs[i].shape[j]);
            memset(char_map, -1, sizeof(int) * _MAX_CHAR);
            for (int j = 0; j < idxa.size(); j++)
                char_map[gscripts[0][idxa[j]]] = j;
            for (int j = 0; j < br_idxa.size(); j++)
                char_map[gscripts[0][br_idxa[j]]] = j;
            sort(idxb.begin(), idxb.end(),
                 [&char_map, &gscripts, i](int a, int b) {
                     return char_map[gscripts[i][a]] < char_map[gscripts[i][b]];
                 });
            sort(br_idxb.begin(), br_idxb.end(),
                 [&char_map, &gscripts, i](int a, int b) {
                     return char_map[gscripts[i][a]] < char_map[gscripts[i][b]];
                 });
            new_br.insert(new_br.end(), new_sh.begin(), new_sh.end());
            NDArray tmp(new_br);
            NDArray::tensordot(garrs[0], garrs[i], tmp, idxa, idxb, br_idxa,
                               br_idxb);
            // remove contracted and broadcast index count
            for (auto &x : idxa)
                char_count[gscripts[0][x]] -= 2;
            for (auto &x : br_idxa)
                char_count[gscripts[0][x]]--;
            garrs[0] = tmp;
            gscripts[0] = newsr.str() + newss.str();
        }
        // final transpose (no copy)
        assert(gscripts[0].size() == result.size());
        if (gscripts[0] != result) {
            memset(char_map, -1, sizeof(int) * _MAX_CHAR);
            for (int j = 0; j < gscripts[0].length(); j++)
                char_map[gscripts[0][j]] = j;
            perm.resize(gscripts[0].length());
            for (int j = 0; j < result.length(); j++)
                perm[j] = char_map[result[j]];
            garrs[0] = garrs[0].transpose(perm);
        }
        return garrs[0];
    }
};

} // namespace block2
