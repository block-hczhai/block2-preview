
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

#include "allocator.hpp"
#include "threading.hpp"
#ifdef _HAS_INTEL_MKL
#ifndef MKL_Complex16
#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#endif
#include "mkl.h"
#endif
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace std;

#define _MINTSZ (sizeof(MKL_INT) / sizeof(int32_t))

namespace block2 {

// General Matrix
template <typename FL> struct GMatrix;

// 2D dense matrix stored in stack memory
template <> struct GMatrix<float> {
    typedef double FL;
    typedef float FP;
    typedef complex<float> FC;
    MKL_INT m, n; // m is rows, n is cols
    float *data;
    GMatrix(float *data, MKL_INT m, MKL_INT n) : data(data), m(m), n(n) {}
    float &operator()(MKL_INT i, MKL_INT j) const {
#ifdef _AGGRESSIVE_DEBUG
        assert((size_t)i * n + j < (size_t)m * n);
#endif
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc_<float>() : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<float>() : alloc)->deallocate(data, size());
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(float)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    float trace() const {
        assert(m == n);
        float r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

// 2D dense matrix stored in stack memory
template <> struct GMatrix<double> {
    typedef long double FL;
    typedef double FP;
    typedef complex<double> FC;
    MKL_INT m, n; // m is rows, n is cols
    double *data;
    GMatrix(double *data, MKL_INT m, MKL_INT n) : data(data), m(m), n(n) {}
    double &operator()(MKL_INT i, MKL_INT j) const {
#ifdef _AGGRESSIVE_DEBUG
        assert((size_t)i * n + j < (size_t)m * n);
#endif
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc_<double>() : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<double>() : alloc)
            ->deallocate(data, size());
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(double)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    double trace() const {
        assert(m == n);
        double r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

typedef GMatrix<double> MatrixRef;

template <typename FL, typename = void> struct GDiagonalMatrix;

// Diagonal matrix
template <typename FL>
struct GDiagonalMatrix<FL,
                       typename enable_if<is_floating_point<FL>::value>::type>
    : GMatrix<FL> {
    using GMatrix<FL>::data;
    using GMatrix<FL>::m;
    using GMatrix<FL>::n;
    FL zero = 0.0;
    GDiagonalMatrix(FL *data, MKL_INT n) : GMatrix<FL>(data, n, n) {}
    FL &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? *(data + i) : const_cast<FL &>(zero);
    }
    size_t size() const { return (size_t)m; }
    // need override since size() is changed (which is not virtual)
    void allocate(const shared_ptr<Allocator<FL>> &alloc = nullptr) {
        data = (alloc == nullptr ? dalloc_<FL>() : alloc)->allocate(size());
    }
    void deallocate(const shared_ptr<Allocator<FL>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<FL>() : alloc)->deallocate(data, size());
        data = nullptr;
    }
    void clear() { memset(data, 0, size() * sizeof(FL)); }
    friend ostream &operator<<(ostream &os, const GDiagonalMatrix &mat) {
        os << "DIAG MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        os << "[ ";
        for (MKL_INT j = 0; j < mat.n; j++)
            os << setw(20) << setprecision(14) << mat(j, j) << " ";
        os << "]" << endl;
        return os;
    }
};

typedef GDiagonalMatrix<double> DiagonalMatrix;

template <typename FL, typename = void> struct GIdentityMatrix;

// Identity matrix
template <typename FL>
struct GIdentityMatrix<FL,
                       typename enable_if<is_floating_point<FL>::value>::type>
    : GDiagonalMatrix<FL> {
    using GDiagonalMatrix<FL>::zero;
    FL one = 1.0;
    GIdentityMatrix(MKL_INT n, FL one = 1.0)
        : GDiagonalMatrix<FL>(nullptr, n), one(one) {}
    FL &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? const_cast<FL &>(one) : const_cast<FL &>(zero);
    }
    void allocate() {}
    void deallocate() {}
    void clear() {}
    friend ostream &operator<<(ostream &os, const GIdentityMatrix &mat) {
        os << "IDENT MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        return os;
    }
};

typedef GIdentityMatrix<double> IdentityMatrix;

// complex dense matrix
template <> struct GMatrix<complex<float>> {
    typedef complex<double> FL;
    typedef float FP;
    typedef complex<float> FC;
    MKL_INT m, n; // m is rows, n is cols
    complex<float> *data;
    GMatrix(complex<float> *data, MKL_INT m, MKL_INT n)
        : data(data), m(m), n(n) {}
    complex<float> &operator()(MKL_INT i, MKL_INT j) const {
#ifdef _AGGRESSIVE_DEBUG
        assert((size_t)i * n + j < (size_t)m * n);
#endif
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        data = (complex<float> *)(alloc == nullptr ? dalloc_<float>() : alloc)
                   ->allocate(size() * 2);
    }
    void deallocate(const shared_ptr<Allocator<float>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<float>() : alloc)
            ->deallocate((float *)data, size() * 2);
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(complex<float>)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "CPX-MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    complex<float> trace() const {
        assert(m == n);
        complex<float> r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

// complex dense matrix
template <> struct GMatrix<complex<double>> {
    typedef complex<long double> FL;
    typedef double FP;
    typedef complex<double> FC;
    MKL_INT m, n; // m is rows, n is cols
    complex<double> *data;
    GMatrix(complex<double> *data, MKL_INT m, MKL_INT n)
        : data(data), m(m), n(n) {}
    complex<double> &operator()(MKL_INT i, MKL_INT j) const {
#ifdef _AGGRESSIVE_DEBUG
        assert((size_t)i * n + j < (size_t)m * n);
#endif
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        data = (complex<double> *)(alloc == nullptr ? dalloc_<FP>() : alloc)
                   ->allocate(size() * 2);
    }
    void deallocate(const shared_ptr<Allocator<double>> &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<FP>() : alloc)
            ->deallocate((double *)data, size() * 2);
        data = nullptr;
    }
    void clear() const { memset(data, 0, size() * sizeof(complex<double>)); }
    GMatrix flip_dims() const { return GMatrix(data, n, m); }
    GMatrix shift_ptr(size_t l) const { return GMatrix(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const GMatrix &mat) {
        os << "CPX-MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (MKL_INT i = 0; i < mat.m; i++) {
            os << "[ ";
            for (MKL_INT j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    complex<double> trace() const {
        assert(m == n);
        complex<double> r = 0;
        for (MKL_INT i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

typedef GMatrix<complex<double>> ComplexMatrixRef;

// Diagonal complex matrix
template <typename FL>
struct GDiagonalMatrix<FL, typename enable_if<is_complex<FL>::value>::type>
    : GMatrix<FL> {
    using GMatrix<FL>::data;
    using GMatrix<FL>::m;
    using GMatrix<FL>::n;
    FL zero = 0.0;
    GDiagonalMatrix(FL *data, MKL_INT n) : GMatrix<FL>(data, n, n) {}
    FL &operator()(MKL_INT i, MKL_INT j) const {
        return i == j ? *(data + i) : const_cast<FL &>(zero);
    }
    size_t size() const { return (size_t)m; }
    // need override since size() is changed (which is not virtual)
    void allocate(const shared_ptr<Allocator<typename GMatrix<FL>::FP>> &alloc =
                      nullptr) {
        data = (FL *)(alloc == nullptr ? dalloc_<typename GMatrix<FL>::FP>()
                                       : alloc)
                   ->allocate(size() * 2);
    }
    void deallocate(const shared_ptr<Allocator<typename GMatrix<FL>::FP>>
                        &alloc = nullptr) {
        (alloc == nullptr ? dalloc_<typename GMatrix<FL>::FP>() : alloc)
            ->deallocate((typename GMatrix<FL>::FP *)data, size() * 2);
        data = nullptr;
    }
    void clear() { memset(data, 0, size() * sizeof(FL)); }
    friend ostream &operator<<(ostream &os, const GDiagonalMatrix &mat) {
        os << "DIAG CPX-MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        os << "[ ";
        for (MKL_INT j = 0; j < mat.n; j++)
            os << setw(20) << setprecision(14) << mat(j, j) << " ";
        os << "]" << endl;
        return os;
    }
};

typedef GDiagonalMatrix<complex<double>> ComplexDiagonalMatrix;

// General rank-n dense tensor
template <typename FL, typename IX = MKL_INT> struct GTensor {
    vector<IX> shape;
    shared_ptr<vector<FL>> data;
    GTensor() {}
    GTensor(IX m, IX k, IX n) : shape{m, k, n} {
        data = make_shared<vector<FL>>((size_t)m * k * n);
    }
    GTensor(const vector<IX> &shape) : shape(shape) {
        size_t x = 1;
        for (IX sh : shape)
            x = x * (size_t)sh;
        data = make_shared<vector<FL>>(x);
    }
    size_t size() const { return data->size(); }
    void clear() { memset(data->data(), 0, size() * sizeof(FL)); }
    void truncate(IX n) {
        assert(shape.size() == 1);
        data->resize(n);
        shape[0] = n;
    }
    void truncate_left(IX nl) {
        assert(shape.size() == 2);
        data->resize(nl * shape[1]);
        shape[0] = nl;
    }
    void truncate_right(IX nr) {
        assert(shape.size() == 2);
        for (IX i = 1; i < shape[0]; i++)
            memmove(data->data() + i * nr, data->data() + i * shape[1],
                    nr * sizeof(FL));
        data->resize(shape[0] * nr);
        shape[1] = nr;
    }
    GMatrix<FL> ref() {
        if (shape.size() == 3 && shape[1] == 1)
            return GMatrix<FL>(data->data(), shape[0], shape[2]);
        else if (shape.size() == 2)
            return GMatrix<FL>(data->data(), shape[0], shape[1]);
        else if (shape.size() == 1)
            return GMatrix<FL>(data->data(), shape[0], 1);
        else {
            assert(false);
            return GMatrix<FL>(data->data(), 0, 1);
        }
    }
    FL &operator()(initializer_list<IX> idx) {
        size_t i = 0;
        int k = 0;
        for (auto &ix : idx)
            i = i * shape[k++] + ix;
        return data->at(i);
    }
    // write array in numpy format
    void write_array(ostream &ofs) const {
        const string magic = "\x93NUMPY";
        const char ver_major = 1, ver_minor = 0;
        const size_t pre_len = sizeof(char) * magic.length() +
                               sizeof(ver_major) + sizeof(ver_minor) +
                               (ver_major == 1 ? 2 : 4);
        ofs.write((char *)magic.c_str(), sizeof(char) * magic.length());
        ofs.write((char *)&ver_major, sizeof(ver_major));
        ofs.write((char *)&ver_minor, sizeof(ver_minor));
        stringstream ss;
        ss << "{'descr': ";
        if (is_same<FL, float>::value)
            ss << "'<f4'";
        else if (is_same<FL, double>::value)
            ss << "'<f8'";
        else if (is_same<FL, long double>::value)
            ss << "'<f16'";
        else if (is_same<FL, complex<float>>::value)
            ss << "'<c8'";
        else if (is_same<FL, complex<double>>::value)
            ss << "'<c16'";
        else if (is_same<FL, complex<long double>>::value)
            ss << "'<c32'";
        else
            throw runtime_error("GTensor::write_array: unsupported data type");
        ss << ", 'fortran_order': False, 'shape': (";
        size_t arr_len = 1;
        for (int i = 0; i < (int)shape.size(); i++) {
            ss << shape[i] << (i == (int)shape.size() - 1 ? ")" : ", ");
            arr_len *= shape[i];
        }
        ss << ", }\n";
        string header = ss.str();
        if (((pre_len + header.length()) & 0x3F) != 0)
            header = header +
                     string(0x40 - ((pre_len + header.length()) & 0x3F), ' ');
        assert(((pre_len + header.length()) & 0x3F) == 0);
        if (ver_major == 1) {
            uint16_t header_len = (uint16_t)header.length();
            ofs.write((char *)&header_len, sizeof(header_len));
        } else {
            uint32_t header_len = (uint32_t)header.length();
            ofs.write((char *)&header_len, sizeof(header_len));
        }
        ofs.write((char *)header.c_str(), sizeof(char) * header.length());
        ofs.write((char *)&(*data)[0], sizeof(FL) * arr_len);
    }
    // read array in numpy format
    void read_array(istream &ifs) {
        string magic = "??????";
        char ver_major, ver_minor;
        ifs.read((char *)magic.c_str(), sizeof(char) * magic.length());
        assert(magic == "\x93NUMPY");
        ifs.read((char *)&ver_major, sizeof(ver_major));
        ifs.read((char *)&ver_minor, sizeof(ver_minor));
        assert(ver_major >= 1 && ver_major <= 3 && ver_minor == 0);
        uint32_t header_len = 0;
        if (ver_major == 1) {
            uint16_t header_len_short;
            ifs.read((char *)&header_len_short, sizeof(header_len_short));
            header_len = header_len_short;
        } else
            ifs.read((char *)&header_len, sizeof(header_len));
        string header(header_len, ' ');
        ifs.read((char *)&header[0], sizeof(char) * header.length());
        vector<string> tokens;
        for (int i = 0, j; i < (int)header_len; i++) {
            if (header[i] == '{' || header[i] == '}' || header[i] == ' ' ||
                header[i] == '\n')
                continue;
            else if (header[i] == '\'' || header[i] == '\"') {
                for (j = i + 1; j < header_len; j++)
                    if (header[j] == header[i] && header[j - 1] != '\\')
                        break;
                assert(header[j] == header[i]);
                tokens.push_back(header.substr(i + 1, j - i - 1));
                i = j;
            } else if (header[i] == ':' || header[i] == ',')
                tokens.push_back(string(1, header[i]));
            else if (header[i] == '(') {
                for (j = i + 1; j < header_len; j++)
                    if (header[j] == ')')
                        break;
                assert(header[j] == ')');
                tokens.push_back(header.substr(i + 1, j - i - 1));
                i = j;
            } else {
                for (j = i + 1; j < header_len; j++)
                    if (header[j] == '}' || header[j] == ',' ||
                        header[j] == ':' || header[j] == ' ')
                        break;
                tokens.push_back(header.substr(i, j - i));
                i = j - 1;
            }
        }
        shape.clear();
        for (int i = 0; i < (int)tokens.size(); i++) {
            if (tokens[i] == "descr") {
                assert(i + 2 < (int)tokens.size());
                assert(tokens[i + 1] == ":");
                string dtype = "";
                if (is_same<FL, float>::value)
                    dtype = "<f4";
                else if (is_same<FL, double>::value)
                    dtype = "<f8";
                else if (is_same<FL, long double>::value)
                    dtype = "<f16";
                else if (is_same<FL, complex<float>>::value)
                    dtype = "<c8";
                else if (is_same<FL, complex<double>>::value)
                    dtype = "<c16";
                else if (is_same<FL, complex<long double>>::value)
                    dtype = "<c32";
                if (dtype != tokens[i + 2])
                    throw runtime_error(
                        "GTensor::read_array: unsupported descr : " + dtype);
                i += 2;
            } else if (tokens[i] == "fortran_order") {
                assert(i + 2 < (int)tokens.size());
                assert(tokens[i + 1] == ":");
                if ("False" != tokens[i + 2])
                    throw runtime_error(
                        "GTensor::read_array: unsupported fortran_order : " +
                        tokens[i + 2]);
                i += 2;
            } else if (tokens[i] == "shape") {
                assert(i + 2 < (int)tokens.size());
                assert(tokens[i + 1] == ":");
                vector<string> shape_str =
                    Parsing::split(tokens[i + 2], ",", true);
                for (auto &r : shape_str)
                    shape.push_back(
                        (IX)Parsing::to_long_long(Parsing::trim(r)));
                i += 2;
            } else if (tokens[i] == ",")
                continue;
            else
                throw runtime_error(
                    "GTensor::read_array: unsupported token : '" + tokens[i] +
                    "'");
        }
        size_t arr_len = 1;
        for (IX sh : shape)
            arr_len = arr_len * (size_t)sh;
        data = make_shared<vector<FL>>(arr_len);
        ifs.read((char *)&(*data)[0], sizeof(FL) * arr_len);
    }
    friend ostream &operator<<(ostream &os, const GTensor &ts) {
        os << "TENSOR ( ";
        for (auto sh : ts.shape)
            os << sh << " ";
        os << ")" << endl;
        os << "   DATA [";
        for (auto x : *ts.data)
            os << fixed << setw(20) << setprecision(14) << x << " ";
        os << "]" << endl;
        return os;
    }
};

typedef GTensor<double> Tensor;
typedef GTensor<complex<double>> ComplexTensor;

} // namespace block2
