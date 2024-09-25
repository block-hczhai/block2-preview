
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

#ifdef _OPENMP
#include <omp.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "../block2_core.hpp"

namespace py = pybind11;
using namespace block2;

PYBIND11_MAKE_OPAQUE(vector<int>);
PYBIND11_MAKE_OPAQUE(vector<char>);
PYBIND11_MAKE_OPAQUE(vector<long long int>);
PYBIND11_MAKE_OPAQUE(vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(vector<int16_t>);
PYBIND11_MAKE_OPAQUE(vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(vector<pair<uint32_t, uint32_t>>);
PYBIND11_MAKE_OPAQUE(vector<double>);
PYBIND11_MAKE_OPAQUE(vector<long double>);
PYBIND11_MAKE_OPAQUE(vector<complex<double>>);
PYBIND11_MAKE_OPAQUE(vector<complex<long double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<int, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<int, complex<double>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<vector<int>, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<vector<int>, complex<double>>>);
PYBIND11_MAKE_OPAQUE(vector<size_t>);
PYBIND11_MAKE_OPAQUE(vector<string>);
PYBIND11_MAKE_OPAQUE(vector<vector<uint8_t>>);
PYBIND11_MAKE_OPAQUE(vector<vector<uint16_t>>);
PYBIND11_MAKE_OPAQUE(vector<vector<uint32_t>>);
PYBIND11_MAKE_OPAQUE(vector<vector<long long int>>);
PYBIND11_MAKE_OPAQUE(vector<std::array<int16_t, 4>>);
PYBIND11_MAKE_OPAQUE(vector<std::array<int16_t, 5>>);
PYBIND11_MAKE_OPAQUE(vector<std::array<int16_t, 6>>);
PYBIND11_MAKE_OPAQUE(vector<vector<double>>);
PYBIND11_MAKE_OPAQUE(vector<vector<long double>>);
PYBIND11_MAKE_OPAQUE(vector<vector<complex<double>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<complex<long double>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<double>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<complex<long double>>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<int>>);
PYBIND11_MAKE_OPAQUE(vector<pair<int, int>>);
PYBIND11_MAKE_OPAQUE(vector<pair<long long int, int>>);
PYBIND11_MAKE_OPAQUE(vector<pair<long long int, long long int>>);
PYBIND11_MAKE_OPAQUE(unordered_map<int, int>);
PYBIND11_MAKE_OPAQUE(vector<unordered_map<int, int>>);
PYBIND11_MAKE_OPAQUE(unordered_map<int, pair<int, int>>);
PYBIND11_MAKE_OPAQUE(vector<unordered_map<int, pair<int, int>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SpinOperator, uint16_t>>);
PYBIND11_MAKE_OPAQUE(vector<SpinPermTerm>);
PYBIND11_MAKE_OPAQUE(vector<vector<SpinPermTerm>>);
PYBIND11_MAKE_OPAQUE(vector<SpinPermTensor>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SpinPermScheme>>);
PYBIND11_MAKE_OPAQUE(vector<pair<double, string>>);
PYBIND11_MAKE_OPAQUE(map<vector<uint16_t>, vector<pair<double, string>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<vector<uint16_t>, vector<pair<double, string>>>>);
// double
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GTensor<double>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<shared_ptr<GTensor<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GCSRMatrix<double>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<int, int>, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<long long int, long long int>, double>>);
PYBIND11_MAKE_OPAQUE(map<string, string>);
PYBIND11_MAKE_OPAQUE(map<string, double>);
PYBIND11_MAKE_OPAQUE(map<string, complex<double>>);
PYBIND11_MAKE_OPAQUE(vector<map<string, string>>);
PYBIND11_MAKE_OPAQUE(vector<pair<string, string>>);

// SZ
PYBIND11_MAKE_OPAQUE(vector<SZ>);
PYBIND11_MAKE_OPAQUE(vector<vector<SZ>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<SU2>);
PYBIND11_MAKE_OPAQUE(vector<vector<SU2>>);
// SZK
PYBIND11_MAKE_OPAQUE(vector<SZK>);
PYBIND11_MAKE_OPAQUE(vector<vector<SZK>>);
// SU2K
PYBIND11_MAKE_OPAQUE(vector<SU2K>);
PYBIND11_MAKE_OPAQUE(vector<vector<SU2K>>);
// SGF
PYBIND11_MAKE_OPAQUE(vector<SGF>);
PYBIND11_MAKE_OPAQUE(vector<vector<SGF>>);
// SGB
PYBIND11_MAKE_OPAQUE(vector<SGB>);
PYBIND11_MAKE_OPAQUE(vector<vector<SGB>>);
// SAny
PYBIND11_MAKE_OPAQUE(vector<SAny>);
PYBIND11_MAKE_OPAQUE(vector<vector<SAny>>);

#ifdef _USE_SU2SZ
// SZ
PYBIND11_MAKE_OPAQUE(vector<pair<uint8_t, SZ>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<uint8_t, SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SZ, double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<StateInfo<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SZ>>, double>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<shared_ptr<OpExpr<SZ>>, double>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SZ>>, pair<size_t, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SZ, shared_ptr<SparseMatrixInfo<SZ>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<SZ, shared_ptr<SparseMatrixInfo<SZ>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicRowVector<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicColumnVector<SZ>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicMatrix<SZ>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SZ, double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<unordered_map<string, shared_ptr<SparseMatrix<SZ, double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SZ, double>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SZ>>, shared_ptr<SparseMatrix<SZ, double>>,
        op_expr_less<SZ>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<double>>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<pair<uint8_t, SU2>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<uint8_t, SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SU2, double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpExpr<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<StateInfo<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SU2>>, double>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<shared_ptr<OpExpr<SU2>>, double>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SU2>>, pair<size_t, double>>);
PYBIND11_MAKE_OPAQUE(vector<pair<SU2, shared_ptr<SparseMatrixInfo<SU2>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<SU2, shared_ptr<SparseMatrixInfo<SU2>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrixInfo<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<Symbolic<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicRowVector<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicColumnVector<SU2>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SymbolicMatrix<SU2>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SU2, double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<unordered_map<string, shared_ptr<SparseMatrix<SU2, double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SU2, double>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SU2>>, shared_ptr<SparseMatrix<SU2, double>>,
        op_expr_less<SU2>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<double>>>>>);
#endif

#ifdef _USE_COMPLEX
// complex
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GTensor<complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<shared_ptr<GTensor<complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GCSRMatrix<complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<int, int>, complex<double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<pair<pair<long long int, long long int>, complex<double>>>);
#ifdef _USE_SU2SZ
// SZ
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SZ, complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SZ>>, complex<double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<shared_ptr<OpExpr<SZ>>, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SZ>>, pair<size_t, complex<double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    map<OpNames, shared_ptr<SparseMatrix<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SZ, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        unordered_map<string, shared_ptr<SparseMatrix<SZ, complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SZ, complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SZ>>, shared_ptr<SparseMatrix<SZ, complex<double>>>,
        op_expr_less<SZ>>);
PYBIND11_MAKE_OPAQUE(
    vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<complex<double>>>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SU2, complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SU2>>, complex<double>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<shared_ptr<OpExpr<SU2>>, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SU2>>, pair<size_t, complex<double>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    map<OpNames, shared_ptr<SparseMatrix<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SU2, complex<double>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        unordered_map<string, shared_ptr<SparseMatrix<SU2, complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SU2, complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SU2>>, shared_ptr<SparseMatrix<SU2, complex<double>>>,
        op_expr_less<SU2>>);
PYBIND11_MAKE_OPAQUE(
    vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<complex<double>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<complex<double>>>>>>);
#endif
#endif

#ifdef _USE_SINGLE_PREC

PYBIND11_MAKE_OPAQUE(vector<float>);
PYBIND11_MAKE_OPAQUE(vector<complex<float>>);
PYBIND11_MAKE_OPAQUE(vector<vector<float>>);
PYBIND11_MAKE_OPAQUE(vector<vector<complex<float>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<float>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<vector<complex<float>>>>);
// double
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GTensor<float>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<shared_ptr<GTensor<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GCSRMatrix<float>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<int, int>, float>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<long long int, long long int>, float>>);
#ifdef _USE_SU2SZ
// SZ
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SZ, float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SZ>>, float>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<shared_ptr<OpExpr<SZ>>, float>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SZ>>, pair<size_t, float>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SZ, float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<unordered_map<string, shared_ptr<SparseMatrix<SZ, float>>>>);
PYBIND11_MAKE_OPAQUE(vector<map<OpNames, shared_ptr<SparseMatrix<SZ, float>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SZ>>, shared_ptr<SparseMatrix<SZ, float>>,
        op_expr_less<SZ>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<float>>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SU2, float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SU2>>, float>>);
PYBIND11_MAKE_OPAQUE(vector<vector<pair<shared_ptr<OpExpr<SU2>>, float>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SU2>>, pair<size_t, float>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(map<OpNames, shared_ptr<SparseMatrix<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SU2, float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<unordered_map<string, shared_ptr<SparseMatrix<SU2, float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SU2, float>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SU2>>, shared_ptr<SparseMatrix<SU2, float>>,
        op_expr_less<SU2>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<float>>>>>);
#endif

#ifdef _USE_COMPLEX
// complex
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GTensor<complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<vector<shared_ptr<GTensor<complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<GCSRMatrix<complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<pair<int, int>, complex<float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<pair<pair<long long int, long long int>, complex<float>>>);
#ifdef _USE_SU2SZ
// SZ
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SZ, complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SZ>>, complex<float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<shared_ptr<OpExpr<SZ>>, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SZ>>, pair<size_t, complex<float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    map<OpNames, shared_ptr<SparseMatrix<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SZ, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        unordered_map<string, shared_ptr<SparseMatrix<SZ, complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SZ, complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SZ>>, shared_ptr<SparseMatrix<SZ, complex<float>>>,
        op_expr_less<SZ>>);
PYBIND11_MAKE_OPAQUE(
    vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SZ, SZ>, shared_ptr<GTensor<complex<float>>>>>>);
// SU2
PYBIND11_MAKE_OPAQUE(vector<vector<vector<pair<SU2, complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpProduct<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OpElement<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<pair<shared_ptr<OpExpr<SU2>>, complex<float>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<shared_ptr<OpExpr<SU2>>, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<shared_ptr<OpExpr<SU2>>, pair<size_t, complex<float>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<SparseMatrix<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(vector<shared_ptr<OperatorTensor<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    map<OpNames, shared_ptr<SparseMatrix<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    unordered_map<string, shared_ptr<SparseMatrix<SU2, complex<float>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<
        unordered_map<string, shared_ptr<SparseMatrix<SU2, complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<map<OpNames, shared_ptr<SparseMatrix<SU2, complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(
    map<shared_ptr<OpExpr<SU2>>, shared_ptr<SparseMatrix<SU2, complex<float>>>,
        op_expr_less<SU2>>);
PYBIND11_MAKE_OPAQUE(
    vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<complex<float>>>>>);
PYBIND11_MAKE_OPAQUE(
    vector<vector<pair<pair<SU2, SU2>, shared_ptr<GTensor<complex<float>>>>>>);
#endif
#endif

#endif

template <typename T> struct Array {
    T *data;
    size_t n;
    Array(T *data, size_t n) : data(data), n(n) {}
    T &operator[](size_t idx) { return data[idx]; }
};

template <typename T, typename... Args>
size_t printable(const T &p, const Args &...) {
    return (size_t)&p;
}

template <typename T>
auto printable(const T &p)
    -> decltype(declval<ostream &>() << declval<T>(), T()) {
    return p;
}

template <typename T>
py::class_<Array<T>, shared_ptr<Array<T>>> bind_array(py::module &m,
                                                      const char *name) {
    return py::class_<Array<T>, shared_ptr<Array<T>>>(m, name)
        .def(
            "__setitem__",
            [](Array<T> *self, size_t i, const T &t) {
                self->operator[](i) = t;
            },
            py::keep_alive<1, 3>())
        .def("__getitem__",
             [](Array<T> *self, size_t i) { return self->operator[](i); })
        .def("__len__", [](Array<T> *self) { return self->n; })
        .def("__repr__",
             [name](Array<T> *self) {
                 stringstream ss;
                 ss << name << "(LEN=" << self->n << ")[";
                 for (size_t i = 0; i < self->n; i++)
                     ss << printable<T>(self->data[i]) << ",";
                 ss << "]";
                 return ss.str();
             })
        .def(
            "__iter__",
            [](Array<T> *self) {
                return py::make_iterator<
                    py::return_value_policy::reference_internal, T *, T *, T &>(
                    std::move(self->data), std::move(self->data + self->n));
            },
            py::keep_alive<0, 1>());
}

template <typename S>
auto bind_cg(py::module &m) -> decltype(typename S::is_sany_t()) {
    py::class_<CG<S>, shared_ptr<CG<S>>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("triangle", &CG<S>::triangle, py::arg("tja"),
                    py::arg("tjb"), py::arg("tjc"))
        .def("sqrt_delta", &CG<S>::sqrt_delta, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"))
        .def("cg", &CG<S>::cg, py::arg("tja"), py::arg("tjb"), py::arg("tjc"),
             py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_3j", &CG<S>::wigner_3j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &CG<S>::wigner_6j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"))
        .def("wigner_9j", &CG<S>::wigner_9j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"),
             py::arg("g"), py::arg("h"), py::arg("i"))
        .def("racah", &CG<S>::racah, py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("d"), py::arg("e"), py::arg("f"))
        .def("transpose_cg", &CG<S>::transpose_cg, py::arg("d"), py::arg("l"),
             py::arg("r"));
}

template <typename S>
auto bind_cg(py::module &m) -> decltype(typename S::is_sz_t()) {
    py::class_<CG<S>, shared_ptr<CG<S>>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def("wigner_6j", &CG<S>::wigner_6j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"))
        .def("wigner_9j", &CG<S>::wigner_9j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"),
             py::arg("g"), py::arg("h"), py::arg("i"))
        .def("racah", &CG<S>::racah, py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("d"), py::arg("e"), py::arg("f"))
        .def("transpose_cg", &CG<S>::transpose_cg, py::arg("d"), py::arg("l"),
             py::arg("r"));
}

template <typename S>
auto bind_cg(py::module &m) -> decltype(typename S::is_sg_t()) {
    py::class_<CG<S>, shared_ptr<CG<S>>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def("wigner_6j", &CG<S>::wigner_6j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"))
        .def("wigner_9j", &CG<S>::wigner_9j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"),
             py::arg("g"), py::arg("h"), py::arg("i"))
        .def("racah", &CG<S>::racah, py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("d"), py::arg("e"), py::arg("f"))
        .def("transpose_cg", &CG<S>::transpose_cg, py::arg("d"), py::arg("l"),
             py::arg("r"));
}

template <typename S>
auto bind_cg(py::module &m) -> decltype(typename S::is_su2_t()) {
    py::class_<CG<S>, shared_ptr<CG<S>>>(m, "CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("triangle", &CG<S>::triangle, py::arg("tja"),
                    py::arg("tjb"), py::arg("tjc"))
        .def("sqrt_delta", &CG<S>::sqrt_delta, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"))
        .def("cg", &CG<S>::cg, py::arg("tja"), py::arg("tjb"), py::arg("tjc"),
             py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_3j", &CG<S>::wigner_3j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &CG<S>::wigner_6j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"))
        .def("wigner_9j", &CG<S>::wigner_9j, py::arg("a"), py::arg("b"),
             py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"),
             py::arg("g"), py::arg("h"), py::arg("i"))
        .def("racah", &CG<S>::racah, py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("d"), py::arg("e"), py::arg("f"))
        .def("transpose_cg", &CG<S>::transpose_cg, py::arg("d"), py::arg("l"),
             py::arg("r"));
}

template <typename S> void bind_expr(py::module &m) {
    py::class_<OpExpr<S>, shared_ptr<OpExpr<S>>>(m, "OpExpr")
        .def(py::init<>())
        .def("get_type", &OpExpr<S>::get_type)
        .def(py::self == py::self)
        .def("__repr__", &OpExpr<S>::to_str);

    py::bind_vector<vector<shared_ptr<OpExpr<S>>>>(m, "VectorOpExpr");

    py::class_<OpExprRef<S>, shared_ptr<OpExprRef<S>>, OpExpr<S>>(m,
                                                                  "OpExprRef");

    py::class_<OpCounter<S>, shared_ptr<OpCounter<S>>, OpExpr<S>>(m,
                                                                  "OpCounter")
        .def(py::init<uint64_t>())
        .def_readwrite("data", &OpCounter<S>::data);

    struct PySymbolic : Symbolic<S> {
        using Symbolic<S>::Symbolic;
        SymTypes get_type() const override {
            PYBIND11_OVERLOAD_PURE(const SymTypes, Symbolic<S>, get_type, );
        }
        shared_ptr<Symbolic<S>> copy() const override {
            PYBIND11_OVERLOAD_PURE(shared_ptr<Symbolic<S>>, Symbolic<S>,
                                   copy, );
        }
    };

    py::class_<Symbolic<S>, PySymbolic, shared_ptr<Symbolic<S>>>(m, "Symbolic")
        .def(py::init<int, int>())
        .def_readwrite("m", &Symbolic<S>::m)
        .def_readwrite("n", &Symbolic<S>::n)
        .def_readwrite("data", &Symbolic<S>::data)
        .def("get_type", [](Symbolic<S> *self) { return self->get_type(); })
        .def("__matmul__",
             [](const shared_ptr<Symbolic<S>> &self,
                const shared_ptr<Symbolic<S>> &other) { return self * other; });

    py::class_<SymbolicRowVector<S>, shared_ptr<SymbolicRowVector<S>>,
               Symbolic<S>>(m, "SymbolicRowVector")
        .def(py::init<int>())
        .def("__getitem__",
             [](SymbolicRowVector<S> *self, int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SymbolicRowVector<S> *self, int idx,
                const shared_ptr<OpExpr<S>> &v) { (*self)[idx] = v; })
        .def("__len__", [](SymbolicRowVector<S> *self) { return self->n; })
        .def("copy", &SymbolicRowVector<S>::copy);

    py::class_<SymbolicColumnVector<S>, shared_ptr<SymbolicColumnVector<S>>,
               Symbolic<S>>(m, "SymbolicColumnVector")
        .def(py::init<int>())
        .def("__getitem__", [](SymbolicColumnVector<S> *self,
                               int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SymbolicColumnVector<S> *self, int idx,
                const shared_ptr<OpExpr<S>> &v) { (*self)[idx] = v; })
        .def("__len__", [](SymbolicColumnVector<S> *self) { return self->m; })
        .def("copy", &SymbolicColumnVector<S>::copy);

    py::class_<SymbolicMatrix<S>, shared_ptr<SymbolicMatrix<S>>, Symbolic<S>>(
        m, "SymbolicMatrix")
        .def(py::init<int, int>())
        .def_readwrite("indices", &SymbolicMatrix<S>::indices)
        .def("__setitem__",
             [](SymbolicMatrix<S> *self, int i, int j,
                const shared_ptr<OpExpr<S>> &v) {
                 (*self)[{i, j}] = v;
             })
        .def("copy", &SymbolicMatrix<S>::copy);

    py::bind_vector<vector<shared_ptr<Symbolic<S>>>>(m, "VectorSymbolic");
    py::bind_vector<vector<shared_ptr<SymbolicRowVector<S>>>>(
        m, "VectorSymbolicRowVector");
    py::bind_vector<vector<shared_ptr<SymbolicColumnVector<S>>>>(
        m, "VectorSymbolicColumnVector");
    py::bind_vector<vector<shared_ptr<SymbolicMatrix<S>>>>(
        m, "VectorSymbolicMatrix");
}

template <typename S, typename FL> void bind_fl_expr(py::module &m) {

    py::bind_vector<vector<pair<shared_ptr<OpExpr<S>>, FL>>>(
        m, "VectorPExprDouble");
    py::bind_vector<vector<vector<pair<shared_ptr<OpExpr<S>>, FL>>>>(
        m, "VectorVectorPExprDouble");
    py::bind_map<unordered_map<shared_ptr<OpExpr<S>>, pair<size_t, FL>>>(
        m, "MapExprPSizeDouble");

    py::class_<OpElement<S, FL>, shared_ptr<OpElement<S, FL>>, OpExpr<S>>(
        m, "OpElement")
        .def(py::init<OpNames, SiteIndex, S>())
        .def(py::init<OpNames, SiteIndex, S, FL>())
        .def_readwrite("name", &OpElement<S, FL>::name)
        .def_readwrite("site_index", &OpElement<S, FL>::site_index)
        .def_readwrite("factor", &OpElement<S, FL>::factor)
        .def_readwrite("q_label", &OpElement<S, FL>::q_label)
        .def("abs", &OpElement<S, FL>::abs)
        .def("__mul__", &OpElement<S, FL>::operator*)
        .def(py::self == py::self)
        .def(py::self < py::self)
        .def("__hash__", &OpElement<S, FL>::hash);

    py::class_<OpElementRef<S, FL>, shared_ptr<OpElementRef<S, FL>>, OpExpr<S>>(
        m, "OpElementRef")
        .def(py::init<const shared_ptr<OpElement<S, FL>> &, int8_t, int8_t>())
        .def_readwrite("op", &OpElementRef<S, FL>::op)
        .def_readwrite("factor", &OpElementRef<S, FL>::factor)
        .def_readwrite("trans", &OpElementRef<S, FL>::trans);

    py::class_<OpProduct<S, FL>, shared_ptr<OpProduct<S, FL>>, OpExpr<S>>(
        m, "OpProduct")
        .def(py::init<const shared_ptr<OpElement<S, FL>> &, FL>())
        .def(py::init<const shared_ptr<OpElement<S, FL>> &, FL, uint8_t>())
        .def(py::init<const shared_ptr<OpElement<S, FL>> &,
                      const shared_ptr<OpElement<S, FL>> &, FL>())
        .def(py::init<const shared_ptr<OpElement<S, FL>> &,
                      const shared_ptr<OpElement<S, FL>> &, FL, uint8_t>())
        .def_readwrite("factor", &OpProduct<S, FL>::factor)
        .def_readwrite("conj", &OpProduct<S, FL>::conj)
        .def_readwrite("a", &OpProduct<S, FL>::a)
        .def_readwrite("b", &OpProduct<S, FL>::b)
        .def("get_op", &OpProduct<S, FL>::get_op)
        .def("__hash__", &OpProduct<S, FL>::hash);

    py::class_<OpSumProd<S, FL>, shared_ptr<OpSumProd<S, FL>>,
               OpProduct<S, FL>>(m, "OpSumProd")
        .def(py::init<const shared_ptr<OpElement<S, FL>> &,
                      const vector<shared_ptr<OpElement<S, FL>>> &,
                      const vector<bool> &, FL, uint8_t,
                      const shared_ptr<OpElement<S, FL>> &>())
        .def(py::init<const shared_ptr<OpElement<S, FL>> &,
                      const vector<shared_ptr<OpElement<S, FL>>> &,
                      const vector<bool> &, FL, uint8_t>())
        .def(py::init<const shared_ptr<OpElement<S, FL>> &,
                      const vector<shared_ptr<OpElement<S, FL>>> &,
                      const vector<bool> &, FL>())
        .def(
            py::init<const vector<shared_ptr<OpElement<S, FL>>> &,
                     const shared_ptr<OpElement<S, FL>> &, const vector<bool> &,
                     FL, uint8_t, const shared_ptr<OpElement<S, FL>> &>())
        .def(py::init<const vector<shared_ptr<OpElement<S, FL>>> &,
                      const shared_ptr<OpElement<S, FL>> &,
                      const vector<bool> &, FL, uint8_t>())
        .def(py::init<const vector<shared_ptr<OpElement<S, FL>>> &,
                      const shared_ptr<OpElement<S, FL>> &,
                      const vector<bool> &, FL>())
        .def_readwrite("ops", &OpSumProd<S, FL>::ops)
        .def_readwrite("conjs", &OpSumProd<S, FL>::conjs)
        .def_readwrite("c", &OpSumProd<S, FL>::c);

    py::class_<OpSum<S, FL>, shared_ptr<OpSum<S, FL>>, OpExpr<S>>(m, "OpSum")
        .def(py::init<const vector<shared_ptr<OpProduct<S, FL>>> &>())
        .def_readwrite("strings", &OpSum<S, FL>::strings);

    py::bind_vector<vector<shared_ptr<OpElement<S, FL>>>>(m, "VectorOpElement");
    py::bind_vector<vector<shared_ptr<OpProduct<S, FL>>>>(m, "VectorOpProduct");
}

template <typename S> void bind_state_info(py::module &m, const string &name) {

    bind_array<StateInfo<S>>(m, "ArrayStateInfo");
    bind_array<S>(m, ("Array" + name).c_str());
    bind_array<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
        m, "ArrayVectorPLMatInfo");
    py::bind_vector<vector<shared_ptr<StateInfo<S>>>>(m, "VectorStateInfo");
    py::bind_vector<vector<pair<uint8_t, S>>>(m, "VectorPUInt8S");
    py::bind_vector<vector<vector<pair<uint8_t, S>>>>(m, "VectorVectorPUInt8S");

    py::class_<StateInfo<S>, shared_ptr<StateInfo<S>>>(m, "StateInfo")
        .def(py::init<>())
        .def(py::init<S>())
        .def_readwrite("n", &StateInfo<S>::n)
        .def_readwrite("n_states_total", &StateInfo<S>::n_states_total)
        .def_property_readonly(
            "quanta",
            [](StateInfo<S> *self) { return Array<S>(self->quanta, self->n); })
        .def_property_readonly("n_states",
                               [](StateInfo<S> *self) {
                                   return Array<ubond_t>(self->n_states,
                                                         self->n);
                               })
        .def("load_data",
             (void(StateInfo<S>::*)(const string &)) & StateInfo<S>::load_data)
        .def("save_data", (void(StateInfo<S>::*)(const string &) const) &
                              StateInfo<S>::save_data)
        .def("allocate",
             [](StateInfo<S> *self, int length) { self->allocate(length); })
        .def("deallocate", &StateInfo<S>::deallocate)
        .def("reallocate", &StateInfo<S>::reallocate, py::arg("length"))
        .def("sort_states", &StateInfo<S>::sort_states)
        .def("copy_data_to", &StateInfo<S>::copy_data_to)
        .def("deep_copy", &StateInfo<S>::deep_copy)
        .def("collect", &StateInfo<S>::collect,
             py::arg("target") = S(S::invalid))
        .def("find_state", &StateInfo<S>::find_state)
        .def_static("tensor_product_ref",
                    (StateInfo<S>(*)(const StateInfo<S> &, const StateInfo<S> &,
                                     const StateInfo<S> &)) &
                        StateInfo<S>::tensor_product)
        .def_static(
            "tensor_product",
            (StateInfo<S>(*)(const StateInfo<S> &, const StateInfo<S> &, S)) &
                StateInfo<S>::tensor_product)
        .def_static("get_connection_info", &StateInfo<S>::get_connection_info)
        .def_static("filter", &StateInfo<S>::filter)
        .def("__repr__", [](StateInfo<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<typename StateInfo<S>::ConnectionInfo,
               shared_ptr<typename StateInfo<S>::ConnectionInfo>>(
        m, "StateConnectionInfo")
        .def(py::init<>())
        .def_readwrite("acc_n_states",
                       &StateInfo<S>::ConnectionInfo::acc_n_states)
        .def_readwrite("ij_indices", &StateInfo<S>::ConnectionInfo::ij_indices);
}

template <typename S, typename FL>
void bind_fl_state_info(py::module &m, const string &fname) {
    py::bind_vector<vector<pair<S, FL>>>(m, ("VectorPS" + fname).c_str());
    py::bind_vector<vector<vector<pair<S, FL>>>>(
        m, ("VectorVectorPS" + fname).c_str());
    py::bind_vector<vector<vector<vector<pair<S, FL>>>>>(
        m, ("VectorVectorVectorPS" + fname).c_str());
}

template <typename S> void bind_sparse(py::module &m) {

    py::class_<typename SparseMatrixInfo<S>::ConnectionInfo,
               shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo>>(
        m, "SpMatConnectionInfo")
        .def(py::init<>())
        .def_property_readonly(
            "n",
            [](typename SparseMatrixInfo<S>::ConnectionInfo *self) {
                return self->n[4];
            })
        .def_readwrite("nc", &SparseMatrixInfo<S>::ConnectionInfo::nc)
        .def("initialize_tp",
             &SparseMatrixInfo<S>::ConnectionInfo::initialize_tp)
        .def("deallocate", &SparseMatrixInfo<S>::ConnectionInfo::deallocate)
        .def("__repr__",
             [](typename SparseMatrixInfo<S>::ConnectionInfo *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             });

    py::class_<SparseMatrixInfo<S>, shared_ptr<SparseMatrixInfo<S>>>(
        m, "SparseMatrixInfo")
        .def(py::init<>())
        .def(py::init<const shared_ptr<Allocator<uint32_t>> &>())
        .def_readwrite("delta_quantum", &SparseMatrixInfo<S>::delta_quantum)
        .def_readwrite("is_fermion", &SparseMatrixInfo<S>::is_fermion)
        .def_readwrite("is_wavefunction", &SparseMatrixInfo<S>::is_wavefunction)
        .def_readwrite("n", &SparseMatrixInfo<S>::n)
        .def_readwrite("cinfo", &SparseMatrixInfo<S>::cinfo)
        .def_property_readonly("quanta",
                               [](SparseMatrixInfo<S> *self) {
                                   return Array<S>(self->quanta, self->n);
                               })
        .def_property_readonly("n_states_total",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<uint32_t>(
                                       self->n, self->n_states_total);
                               })
        .def_property_readonly("n_states_bra",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<ubond_t>(
                                       self->n, self->n_states_bra);
                               })
        .def_property_readonly("n_states_ket",
                               [](SparseMatrixInfo<S> *self) {
                                   return py::array_t<ubond_t>(
                                       self->n, self->n_states_ket);
                               })
        .def("initialize", &SparseMatrixInfo<S>::initialize, py::arg("bra"),
             py::arg("ket"), py::arg("dq"), py::arg("is_fermion"),
             py::arg("wfn") = false)
        .def("initialize_trans_contract",
             &SparseMatrixInfo<S>::initialize_trans_contract)
        .def("initialize_contract", &SparseMatrixInfo<S>::initialize_contract)
        .def("initialize_dm", &SparseMatrixInfo<S>::initialize_dm)
        .def("find_state", &SparseMatrixInfo<S>::find_state, py::arg("q"),
             py::arg("start") = 0)
        .def_property_readonly("total_memory",
                               &SparseMatrixInfo<S>::get_total_memory)
        .def("allocate", &SparseMatrixInfo<S>::allocate, py::arg("length"),
             py::arg("ptr") = nullptr)
        .def("extract_state_info", &SparseMatrixInfo<S>::extract_state_info,
             py::arg("right"))
        .def("deallocate", &SparseMatrixInfo<S>::deallocate)
        .def("reallocate", &SparseMatrixInfo<S>::reallocate, py::arg("length"))
        .def("__repr__", [](SparseMatrixInfo<S> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>(
        m, "VectorPLMatInfo");
    py::bind_vector<vector<vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>>>(
        m, "VectorVectorPLMatInfo");
    py::bind_vector<vector<shared_ptr<SparseMatrixInfo<S>>>>(m,
                                                             "VectorSpMatInfo");
    py::bind_vector<vector<pair<S, pair<int16_t, int16_t>>>>(
        m, "VectorPSPInt16Int16");
}

template <typename S, typename FL> void bind_fl_sparse(py::module &m) {

    py::class_<SparseMatrix<S, FL>, shared_ptr<SparseMatrix<S, FL>>>(
        m, "SparseMatrix")
        .def(py::init<>())
        .def(py::init<
             const shared_ptr<Allocator<typename SparseMatrix<S, FL>::FP>> &>())
        .def_readwrite("info", &SparseMatrix<S, FL>::info)
        .def_readwrite("factor", &SparseMatrix<S, FL>::factor)
        .def_readwrite("total_memory", &SparseMatrix<S, FL>::total_memory)
        .def("get_type", &SparseMatrix<S, FL>::get_type)
        .def_property(
            "data",
            [](SparseMatrix<S, FL> *self) {
                return py::array_t<FL>(self->total_memory, self->data);
            },
            [](SparseMatrix<S, FL> *self, const py::array_t<FL> &v) {
                assert(v.size() == self->total_memory);
                memcpy(self->data, v.data(), sizeof(FL) * self->total_memory);
            })
        .def("clear", &SparseMatrix<S, FL>::clear)
        .def("load_data",
             (void(SparseMatrix<S, FL>::*)(
                 const string &, bool,
                 const shared_ptr<Allocator<uint32_t>> &)) &
                 SparseMatrix<S, FL>::load_data,
             py::arg("filename"), py::arg("load_info") = false,
             py::arg("i_alloc") = nullptr)
        .def("save_data",
             (void(SparseMatrix<S, FL>::*)(const string &, bool) const) &
                 SparseMatrix<S, FL>::save_data,
             py::arg("filename"), py::arg("save_info") = false)
        .def("copy_data_from", &SparseMatrix<S, FL>::copy_data_from)
        .def("selective_copy_from", &SparseMatrix<S, FL>::selective_copy_from)
        .def("sparsity", &SparseMatrix<S, FL>::sparsity)
        .def("allocate",
             [](SparseMatrix<S, FL> *self,
                const shared_ptr<SparseMatrixInfo<S>> &info) {
                 self->allocate(info);
             })
        .def("deallocate", &SparseMatrix<S, FL>::deallocate)
        .def("reallocate",
             (void(SparseMatrix<S, FL>::*)(size_t)) &
                 SparseMatrix<S, FL>::reallocate,
             py::arg("length"))
        .def("trace", &SparseMatrix<S, FL>::trace)
        .def("norm", &SparseMatrix<S, FL>::norm)
        .def("iscale", &SparseMatrix<S, FL>::iscale)
        .def("conjugate", &SparseMatrix<S, FL>::conjugate)
        .def("__getitem__",
             [](SparseMatrix<S, FL> *self, int idx) { return (*self)[idx]; })
        .def("__setitem__",
             [](SparseMatrix<S, FL> *self, int idx, const py::array_t<FL> &v) {
                 assert(v.size() == (*self)[idx].size());
                 memcpy((*self)[idx].data, v.data(), sizeof(FL) * v.size());
             })
        .def("left_split",
             [](SparseMatrix<S, FL> *self, ubond_t bond_dim) {
                 shared_ptr<SparseMatrix<S, FL>> left, right;
                 self->left_split(left, right, bond_dim);
                 return std::make_tuple(left, right);
             })
        .def("right_split",
             [](SparseMatrix<S, FL> *self, ubond_t bond_dim) {
                 shared_ptr<SparseMatrix<S, FL>> left, right;
                 self->right_split(left, right, bond_dim);
                 return std::make_tuple(left, right);
             })
        .def("pseudo_inverse", &SparseMatrix<S, FL>::pseudo_inverse,
             py::arg("bond_dim"), py::arg("svd_eps") = 1E-4,
             py::arg("svd_cutoff") = 1E-12)
        .def(
            "left_svd",
            [](SparseMatrix<S, FL> *self) {
                vector<S> qs;
                vector<shared_ptr<GTensor<FL>>> l, r;
                vector<shared_ptr<GTensor<typename SparseMatrix<S, FL>::FP>>> s;
                self->left_svd(qs, l, s, r);
                return std::make_tuple(qs, l, s, r);
            })
        .def(
            "right_svd",
            [](SparseMatrix<S, FL> *self) {
                vector<S> qs;
                vector<shared_ptr<GTensor<FL>>> l, r;
                vector<shared_ptr<GTensor<typename SparseMatrix<S, FL>::FP>>> s;
                self->right_svd(qs, l, s, r);
                return std::make_tuple(qs, l, s, r);
            })
        .def("left_canonicalize", &SparseMatrix<S, FL>::left_canonicalize,
             py::arg("rmat"))
        .def("right_canonicalize", &SparseMatrix<S, FL>::right_canonicalize,
             py::arg("lmat"))
        .def("left_multiply", &SparseMatrix<S, FL>::left_multiply,
             py::arg("lmat"), py::arg("l"), py::arg("m"), py::arg("r"),
             py::arg("lm"), py::arg("lm_cinfo"), py::arg("nlm"))
        .def("right_multiply", &SparseMatrix<S, FL>::right_multiply,
             py::arg("rmat"), py::arg("l"), py::arg("m"), py::arg("r"),
             py::arg("mr"), py::arg("mr_cinfo"), py::arg("nmr"))
        .def("left_multiply_inplace",
             &SparseMatrix<S, FL>::left_multiply_inplace, py::arg("lmat"),
             py::arg("l"), py::arg("m"), py::arg("r"), py::arg("lm"),
             py::arg("lm_cinfo"))
        .def("right_multiply_inplace",
             &SparseMatrix<S, FL>::right_multiply_inplace, py::arg("rmat"),
             py::arg("l"), py::arg("m"), py::arg("r"), py::arg("mr"),
             py::arg("mr_cinfo"))
        .def("randomize", &SparseMatrix<S, FL>::randomize, py::arg("a") = 0.0,
             py::arg("b") = 1.0)
        .def("normalize", &SparseMatrix<S, FL>::normalize)
        .def("deep_copy", &SparseMatrix<S, FL>::deep_copy)
        .def("contract", &SparseMatrix<S, FL>::contract, py::arg("lmat"),
             py::arg("rmat"), py::arg("trace_right") = false)
        .def("swap_to_fused_left", &SparseMatrix<S, FL>::swap_to_fused_left)
        .def("swap_to_fused_right", &SparseMatrix<S, FL>::swap_to_fused_right)
        .def("__repr__", [](SparseMatrix<S, FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::class_<CSRSparseMatrix<S, FL>, shared_ptr<CSRSparseMatrix<S, FL>>,
               SparseMatrix<S, FL>>(m, "CSRSparseMatrix")
        .def(py::init<>())
        .def_readwrite("csr_data", &CSRSparseMatrix<S, FL>::csr_data)
        .def("__getitem__",
             [](CSRSparseMatrix<S, FL> *self, int idx) { return (*self)[idx]; })
        .def(
            "__setitem__",
            [](CSRSparseMatrix<S, FL> *self, int idx, const GCSRMatrix<FL> &v) {
                (*self)[idx].deallocate();
                (*self)[idx] = v;
            })
        .def("from_dense", &CSRSparseMatrix<S, FL>::from_dense)
        .def("wrap_dense", &CSRSparseMatrix<S, FL>::wrap_dense)
        .def("to_dense", &CSRSparseMatrix<S, FL>::to_dense);

    py::class_<ArchivedSparseMatrix<S, FL>,
               shared_ptr<ArchivedSparseMatrix<S, FL>>, SparseMatrix<S, FL>>(
        m, "ArchivedSparseMatrix")
        .def(py::init<const string &, int64_t>())
        .def_readwrite("filename", &ArchivedSparseMatrix<S, FL>::filename)
        .def_readwrite("offset", &ArchivedSparseMatrix<S, FL>::offset)
        .def("load_archive", &ArchivedSparseMatrix<S, FL>::load_archive)
        .def("save_archive", &ArchivedSparseMatrix<S, FL>::save_archive);

    py::class_<DelayedSparseMatrix<S, FL>,
               shared_ptr<DelayedSparseMatrix<S, FL>>, SparseMatrix<S, FL>>(
        m, "DelayedSparseMatrix")
        .def(py::init<>())
        .def("build", &DelayedSparseMatrix<S, FL>::build)
        .def("copy", &DelayedSparseMatrix<S, FL>::copy)
        .def("selective_copy", &DelayedSparseMatrix<S, FL>::selective_copy);

    py::class_<DelayedSparseMatrix<S, FL, SparseMatrix<S, FL>>,
               shared_ptr<DelayedSparseMatrix<S, FL, SparseMatrix<S, FL>>>,
               DelayedSparseMatrix<S, FL>>(m, "DelayedNormalSparseMatrix")
        .def_readwrite("mat",
                       &DelayedSparseMatrix<S, FL, SparseMatrix<S, FL>>::mat)
        .def(py::init<const shared_ptr<SparseMatrix<S, FL>> &>());

    py::class_<DelayedSparseMatrix<S, FL, CSRSparseMatrix<S, FL>>,
               shared_ptr<DelayedSparseMatrix<S, FL, CSRSparseMatrix<S, FL>>>,
               DelayedSparseMatrix<S, FL>>(m, "DelayedCSRSparseMatrix")
        .def_readwrite("mat",
                       &DelayedSparseMatrix<S, FL, CSRSparseMatrix<S, FL>>::mat)
        .def(py::init<const shared_ptr<CSRSparseMatrix<S, FL>> &>());

    py::class_<DelayedSparseMatrix<S, FL, OpExpr<S>>,
               shared_ptr<DelayedSparseMatrix<S, FL, OpExpr<S>>>,
               DelayedSparseMatrix<S, FL>>(m, "DelayedOpExprSparseMatrix")
        .def_readwrite("m", &DelayedSparseMatrix<S, FL, OpExpr<S>>::m)
        .def_readwrite("op", &DelayedSparseMatrix<S, FL, OpExpr<S>>::op)
        .def(py::init<uint16_t, const shared_ptr<OpExpr<S>> &>())
        .def(py::init<uint16_t, const shared_ptr<OpExpr<S>> &,
                      const shared_ptr<SparseMatrixInfo<S>> &>());

    py::class_<DelayedSparseMatrix<S, FL, Hamiltonian<S, FL>>,
               shared_ptr<DelayedSparseMatrix<S, FL, Hamiltonian<S, FL>>>,
               DelayedSparseMatrix<S, FL, OpExpr<S>>>(
        m, "DelayedHamilSparseMatrix")
        .def_readwrite("hamil",
                       &DelayedSparseMatrix<S, FL, Hamiltonian<S, FL>>::hamil)
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, uint16_t,
                      const shared_ptr<OpExpr<S>> &>())
        .def(py::init<const shared_ptr<Hamiltonian<S, FL>> &, uint16_t,
                      const shared_ptr<OpExpr<S>> &,
                      const shared_ptr<SparseMatrixInfo<S>> &>());

    py::bind_vector<vector<shared_ptr<SparseMatrix<S, FL>>>>(m, "VectorSpMat");
    py::bind_vector<vector<map<OpNames, shared_ptr<SparseMatrix<S, FL>>>>>(
        m, "VectorMapOpNamesSpMat");
    py::bind_map<unordered_map<OpNames, shared_ptr<SparseMatrix<S, FL>>>>(
        m, "MapOpNamesSpMat");
    py::bind_map<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>(
        m, "MapStrSpMat");
    py::bind_vector<
        vector<unordered_map<string, shared_ptr<SparseMatrix<S, FL>>>>>(
        m, "VectorMapStrSpMat");
    py::bind_map<
        unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S, FL>>>>(
        m, "MapOpExprSpMat");

    py::bind_vector<vector<pair<pair<S, S>, shared_ptr<GTensor<FL>>>>>(
        m, "VectorPSSTensor");
    py::bind_vector<vector<vector<pair<pair<S, S>, shared_ptr<GTensor<FL>>>>>>(
        m, "VectorVectorPSSTensor");

    py::class_<SparseMatrixGroup<S, FL>, shared_ptr<SparseMatrixGroup<S, FL>>>(
        m, "SparseMatrixGroup")
        .def(py::init<>())
        .def_readwrite("infos", &SparseMatrixGroup<S, FL>::infos)
        .def_readwrite("offsets", &SparseMatrixGroup<S, FL>::offsets)
        .def_readwrite("total_memory", &SparseMatrixGroup<S, FL>::total_memory)
        .def_readwrite("n", &SparseMatrixGroup<S, FL>::n)
        .def_property(
            "data",
            [](SparseMatrixGroup<S, FL> *self) {
                return py::array_t<FL>(self->total_memory, self->data);
            },
            [](SparseMatrixGroup<S, FL> *self, const py::array_t<FL> &v) {
                assert(v.size() == self->total_memory);
                memcpy(self->data, v.data(), sizeof(FL) * self->total_memory);
            })
        .def("load_data", &SparseMatrixGroup<S, FL>::load_data,
             py::arg("filename"), py::arg("load_info") = false,
             py::arg("i_alloc") = nullptr)
        .def("save_data", &SparseMatrixGroup<S, FL>::save_data,
             py::arg("filename"), py::arg("save_info") = false)
        .def("allocate",
             [](SparseMatrixGroup<S, FL> *self,
                const vector<shared_ptr<SparseMatrixInfo<S>>> &infos) {
                 self->allocate(infos);
             })
        .def("deallocate", &SparseMatrixGroup<S, FL>::deallocate)
        .def("deallocate_infos", &SparseMatrixGroup<S, FL>::deallocate_infos)
        .def("delta_quanta", &SparseMatrixGroup<S, FL>::delta_quanta)
        .def("randomize", &SparseMatrixGroup<S, FL>::randomize,
             py::arg("a") = 0.0, py::arg("b") = 1.0)
        .def("norm", &SparseMatrixGroup<S, FL>::norm)
        .def("clear", &SparseMatrixGroup<S, FL>::clear)
        .def("iscale", &SparseMatrixGroup<S, FL>::iscale, py::arg("d"))
        .def("normalize", &SparseMatrixGroup<S, FL>::normalize)
        .def_static("normalize_all", &SparseMatrixGroup<S, FL>::normalize_all)
        .def("left_svd",
             [](SparseMatrixGroup<S, FL> *self) {
                 vector<S> qs;
                 vector<shared_ptr<GTensor<FL>>> r;
                 vector<shared_ptr<GTensor<typename GMatrix<FL>::FP>>> s;
                 vector<vector<shared_ptr<GTensor<FL>>>> l;
                 self->left_svd(qs, l, s, r);
                 return std::make_tuple(qs, l, s, r);
             })
        .def("right_svd",
             [](SparseMatrixGroup<S, FL> *self) {
                 vector<S> qs;
                 vector<shared_ptr<GTensor<FL>>> l;
                 vector<shared_ptr<GTensor<typename GMatrix<FL>::FP>>> s;
                 vector<vector<shared_ptr<GTensor<FL>>>> r;
                 self->right_svd(qs, l, s, r);
                 return std::make_tuple(qs, l, s, r);
             })
        .def("__getitem__", [](SparseMatrixGroup<S, FL> *self, int idx) {
            return (*self)[idx];
        });

    py::bind_vector<vector<shared_ptr<SparseMatrixGroup<S, FL>>>>(
        m, "VectorSpMatGroup");
}

template <typename S, typename FL> void bind_fl_operator(py::module &m) {
    py::class_<OperatorFunctions<S, FL>, shared_ptr<OperatorFunctions<S, FL>>>(
        m, "OperatorFunctions")
        .def_readwrite("cg", &OperatorFunctions<S, FL>::cg)
        .def_readwrite("seq", &OperatorFunctions<S, FL>::seq)
        .def(py::init<const shared_ptr<CG<S>> &>())
        .def("get_type", &OperatorFunctions<S, FL>::get_type)
        .def("iadd", &OperatorFunctions<S, FL>::iadd, py::arg("a"),
             py::arg("b"), py::arg("scale") = 1.0, py::arg("conj") = false)
        .def("tensor_rotate", &OperatorFunctions<S, FL>::tensor_rotate,
             py::arg("a"), py::arg("c"), py::arg("rot_bra"), py::arg("rot_ket"),
             py::arg("trans"), py::arg("scale") = 1.0)
        .def("tensor_product_diagonal",
             &OperatorFunctions<S, FL>::tensor_product_diagonal,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("opdq"), py::arg("scale") = 1.0)
        .def("tensor_left_partial_expectation",
             &OperatorFunctions<S, FL>::tensor_left_partial_expectation,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("v"), py::arg("opdq"), py::arg("scale") = 1.0)
        .def("tensor_right_partial_expectation",
             &OperatorFunctions<S, FL>::tensor_right_partial_expectation,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("v"), py::arg("opdq"), py::arg("scale") = 1.0)
        .def("tensor_product_multiply",
             &OperatorFunctions<S, FL>::tensor_product_multiply,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("v"), py::arg("opdq"), py::arg("scale") = 1.0,
             py::arg("tt") = TraceTypes::None)
        .def("tensor_product", &OperatorFunctions<S, FL>::tensor_product,
             py::arg("conj"), py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("scale") = 1.0)
        .def("product", &OperatorFunctions<S, FL>::product, py::arg("a"),
             py::arg("conj"), py::arg("b"), py::arg("c"),
             py::arg("scale") = 1.0)
        .def_static("trans_product", &OperatorFunctions<S, FL>::trans_product,
                    py::arg("a"), py::arg("b"), py::arg("trace_right"),
                    py::arg("noise") = 0.0,
                    py::arg("noise_type") = NoiseTypes::DensityMatrix);

    py::class_<CSROperatorFunctions<S, FL>,
               shared_ptr<CSROperatorFunctions<S, FL>>,
               OperatorFunctions<S, FL>>(m, "CSROperatorFunctions")
        .def(py::init<const shared_ptr<CG<S>> &>());

    py::class_<OperatorTensor<S, FL>, shared_ptr<OperatorTensor<S, FL>>>(
        m, "OperatorTensor")
        .def(py::init<>())
        .def_readwrite("lmat", &OperatorTensor<S, FL>::lmat)
        .def_readwrite("rmat", &OperatorTensor<S, FL>::rmat)
        .def_readwrite("ops", &OperatorTensor<S, FL>::ops)
        .def("get_type", &OperatorTensor<S, FL>::get_type)
        .def("get_total_memory", &OperatorTensor<S, FL>::get_total_memory)
        .def("reallocate", &OperatorTensor<S, FL>::reallocate, py::arg("clean"))
        .def("deallocate", &OperatorTensor<S, FL>::deallocate)
        .def("copy", &OperatorTensor<S, FL>::copy)
        .def("deep_copy", &OperatorTensor<S, FL>::deep_copy,
             py::arg("alloc") = nullptr, py::arg("ref_alloc") = nullptr);

    py::class_<DelayedOperatorTensor<S, FL>,
               shared_ptr<DelayedOperatorTensor<S, FL>>, OperatorTensor<S, FL>>(
        m, "DelayedOperatorTensor")
        .def(py::init<>())
        .def_readwrite("dops", &DelayedOperatorTensor<S, FL>::dops)
        .def_readwrite("mat", &DelayedOperatorTensor<S, FL>::mat)
        .def_readwrite("stacked_mat",
                       &DelayedOperatorTensor<S, FL>::stacked_mat)
        .def_readwrite("lopt", &DelayedOperatorTensor<S, FL>::lopt)
        .def_readwrite("ropt", &DelayedOperatorTensor<S, FL>::ropt)
        .def_readwrite("exprs", &DelayedOperatorTensor<S, FL>::exprs);

    py::bind_vector<vector<shared_ptr<OperatorTensor<S, FL>>>>(
        m, "VectorOpTensor");

    py::class_<TensorFunctions<S, FL>, shared_ptr<TensorFunctions<S, FL>>>(
        m, "TensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S, FL>> &>())
        .def_readwrite("opf", &TensorFunctions<S, FL>::opf)
        .def("get_type", &TensorFunctions<S, FL>::get_type)
        .def("left_assign", &TensorFunctions<S, FL>::left_assign, py::arg("a"),
             py::arg("c"))
        .def("right_assign", &TensorFunctions<S, FL>::right_assign,
             py::arg("a"), py::arg("c"))
        .def("left_contract", &TensorFunctions<S, FL>::left_contract,
             py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("cexprs") = nullptr, py::arg("delayed") = OpNamesSet())
        .def("right_contract", &TensorFunctions<S, FL>::right_contract,
             py::arg("a"), py::arg("b"), py::arg("c"),
             py::arg("cexprs") = nullptr, py::arg("delayed") = OpNamesSet())
        .def("tensor_product_multi_multiply",
             &TensorFunctions<S, FL>::tensor_product_multi_multiply)
        .def("tensor_product_multiply",
             &TensorFunctions<S, FL>::tensor_product_multiply)
        .def("tensor_product_diagonal",
             &TensorFunctions<S, FL>::tensor_product_diagonal)
        .def("tensor_product", &TensorFunctions<S, FL>::tensor_product)
        .def("delayed_left_contract",
             &TensorFunctions<S, FL>::delayed_left_contract)
        .def("delayed_right_contract",
             &TensorFunctions<S, FL>::delayed_right_contract)
        .def("left_rotate", &TensorFunctions<S, FL>::left_rotate)
        .def("right_rotate", &TensorFunctions<S, FL>::right_rotate)
        .def("tensor_product_npdm_fragment",
             &TensorFunctions<S, FL>::tensor_product_npdm_fragment)
        .def("tensor_product_expectation",
             &TensorFunctions<S, FL>::tensor_product_expectation)
        .def("intermediates", &TensorFunctions<S, FL>::intermediates)
        .def("numerical_transform",
             &TensorFunctions<S, FL>::numerical_transform)
        .def("post_numerical_transform",
             &TensorFunctions<S, FL>::post_numerical_transform)
        .def("substitute_delayed_exprs",
             &TensorFunctions<S, FL>::substitute_delayed_exprs)
        .def("delayed_contract", (shared_ptr<DelayedOperatorTensor<S, FL>>(
                                     TensorFunctions<S, FL>::*)(
                                     const shared_ptr<OperatorTensor<S, FL>> &,
                                     const shared_ptr<OperatorTensor<S, FL>> &,
                                     const shared_ptr<OpExpr<S>> &,
                                     OpNamesSet delayed, bool) const) &
                                     TensorFunctions<S, FL>::delayed_contract)
        .def("delayed_contract_simplified",
             (shared_ptr<DelayedOperatorTensor<S, FL>>(
                 TensorFunctions<S, FL>::*)(
                 const shared_ptr<OperatorTensor<S, FL>> &,
                 const shared_ptr<OperatorTensor<S, FL>> &,
                 const shared_ptr<Symbolic<S>> &,
                 const shared_ptr<Symbolic<S>> &, OpNamesSet delayed, bool,
                 const shared_ptr<Symbolic<S>> &) const) &
                 TensorFunctions<S, FL>::delayed_contract);

    py::class_<ArchivedTensorFunctions<S, FL>,
               shared_ptr<ArchivedTensorFunctions<S, FL>>,
               TensorFunctions<S, FL>>(m, "ArchivedTensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S, FL>> &>())
        .def_readwrite("filename", &ArchivedTensorFunctions<S, FL>::filename)
        .def_readwrite("offset", &ArchivedTensorFunctions<S, FL>::offset)
        .def("archive_tensor", &ArchivedTensorFunctions<S, FL>::archive_tensor,
             py::arg("a"));

    py::class_<DelayedTensorFunctions<S, FL>,
               shared_ptr<DelayedTensorFunctions<S, FL>>,
               TensorFunctions<S, FL>>(m, "DelayedTensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S, FL>> &>());
}

template <typename S, typename FL> void bind_fl_hamiltonian(py::module &m) {
    py::class_<Hamiltonian<S, FL>, shared_ptr<Hamiltonian<S, FL>>>(
        m, "Hamiltonian")
        .def(py::init<S, int, const vector<typename S::pg_t> &>())
        .def_readwrite("opf", &Hamiltonian<S, FL>::opf)
        .def_readwrite("n_sites", &Hamiltonian<S, FL>::n_sites)
        .def_readwrite("orb_sym", &Hamiltonian<S, FL>::orb_sym)
        .def_readwrite("vacuum", &Hamiltonian<S, FL>::vacuum)
        .def_readwrite("basis", &Hamiltonian<S, FL>::basis)
        .def_readwrite("site_op_infos", &Hamiltonian<S, FL>::site_op_infos)
        .def_readwrite("delayed", &Hamiltonian<S, FL>::delayed)
        .def_static("combine_orb_sym", &Hamiltonian<S, FL>::combine_orb_sym)
        .def("get_n_orbs_left", &Hamiltonian<S, FL>::get_n_orbs_left)
        .def("get_n_orbs_right", &Hamiltonian<S, FL>::get_n_orbs_right)
        .def("get_site_ops", &Hamiltonian<S, FL>::get_site_ops)
        .def("filter_site_ops", &Hamiltonian<S, FL>::filter_site_ops)
        .def("find_site_op_info", &Hamiltonian<S, FL>::find_site_op_info)
        .def("get_string_quantum", &Hamiltonian<S, FL>::get_string_quantum)
        .def("deallocate", &Hamiltonian<S, FL>::deallocate);
}

template <typename S> void bind_parallel(py::module &m) {

    py::class_<ParallelCommunicator<S>, shared_ptr<ParallelCommunicator<S>>>(
        m, "ParallelCommunicator")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def_readwrite("size", &ParallelCommunicator<S>::size)
        .def_readwrite("rank", &ParallelCommunicator<S>::rank)
        .def_readwrite("root", &ParallelCommunicator<S>::root)
        .def_readwrite("group", &ParallelCommunicator<S>::group)
        .def_readwrite("grank", &ParallelCommunicator<S>::grank)
        .def_readwrite("gsize", &ParallelCommunicator<S>::gsize)
        .def_readwrite("ngroup", &ParallelCommunicator<S>::ngroup)
        .def_readwrite("tcomm", &ParallelCommunicator<S>::tcomm)
        .def_readwrite("para_type", &ParallelCommunicator<S>::para_type)
        .def("get_parallel_type", &ParallelCommunicator<S>::get_parallel_type)
        .def("barrier", &ParallelCommunicator<S>::barrier)
        .def("split", &ParallelCommunicator<S>::split)
        .def("reduce_sum",
             [](ParallelCommunicator<S> *self, py::array_t<double> arr,
                int owner) {
                 self->reduce_sum(arr.mutable_data(), arr.size(), owner);
             })
        .def("reduce_sum",
             [](ParallelCommunicator<S> *self, py::array_t<float> arr,
                int owner) {
                 self->reduce_sum(arr.mutable_data(), arr.size(), owner);
             })
        .def("reduce_sum",
             [](ParallelCommunicator<S> *self, py::array_t<complex<double>> arr,
                int owner) {
                 self->reduce_sum(arr.mutable_data(), arr.size(), owner);
             })
        .def("reduce_sum",
             [](ParallelCommunicator<S> *self, py::array_t<complex<float>> arr,
                int owner) {
                 self->reduce_sum(arr.mutable_data(), arr.size(), owner);
             })
        .def("broadcast",
             [](ParallelCommunicator<S> *self, py::array_t<double> arr,
                int owner) {
                 self->broadcast(arr.mutable_data(), arr.size(), owner);
             })
        .def("broadcast",
             [](ParallelCommunicator<S> *self, py::array_t<float> arr,
                int owner) {
                 self->broadcast(arr.mutable_data(), arr.size(), owner);
             })
        .def("broadcast",
             [](ParallelCommunicator<S> *self, py::array_t<complex<double>> arr,
                int owner) {
                 self->broadcast(arr.mutable_data(), arr.size(), owner);
             })
        .def("broadcast",
             [](ParallelCommunicator<S> *self, py::array_t<complex<float>> arr,
                int owner) {
                 self->broadcast(arr.mutable_data(), arr.size(), owner);
             })
        .def(
            "broadcast",
            [](ParallelCommunicator<S> *self, py::array_t<int> arr, int owner) {
                self->broadcast(arr.mutable_data(), arr.size(), owner);
            })
        .def("broadcast", [](ParallelCommunicator<S> *self,
                             py::array_t<long long int> arr, int owner) {
            self->broadcast(arr.mutable_data(), arr.size(), owner);
        });

#ifdef _HAS_MPI
    py::class_<MPICommunicator<S>, shared_ptr<MPICommunicator<S>>,
               ParallelCommunicator<S>>(m, "MPICommunicator")
        .def(py::init<>())
        .def(py::init<int>());
#endif

    py::class_<ParallelRule<S>, shared_ptr<ParallelRule<S>>>(m,
                                                             "ParallelRuleBase")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>())
        .def_readwrite("comm", &ParallelRule<S>::comm)
        .def_readwrite("comm_type", &ParallelRule<S>::comm_type)
        .def("get_parallel_type", &ParallelRule<S>::get_parallel_type)
        .def("set_partition", &ParallelRule<S>::set_partition)
        .def("split", &ParallelRule<S>::split)
        .def("is_root", &ParallelRule<S>::is_root);
}

template <typename S, typename FL> void bind_fl_parallel(py::module &m) {

    py::class_<ParallelRule<S, FL>, shared_ptr<ParallelRule<S, FL>>,
               ParallelRule<S>>(m, "ParallelRule")
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &>())
        .def(py::init<const shared_ptr<ParallelCommunicator<S>> &,
                      ParallelCommTypes>())
        .def("__call__", &ParallelRule<S, FL>::operator())
        .def("available", &ParallelRule<S, FL>::available)
        .def("own", &ParallelRule<S, FL>::own)
        .def("owner", &ParallelRule<S, FL>::owner)
        .def("repeat", &ParallelRule<S, FL>::repeat)
        .def("partial", &ParallelRule<S, FL>::partial);

    py::class_<ParallelTensorFunctions<S, FL>,
               shared_ptr<ParallelTensorFunctions<S, FL>>,
               TensorFunctions<S, FL>>(m, "ParallelTensorFunctions")
        .def(py::init<const shared_ptr<OperatorFunctions<S, FL>> &,
                      const shared_ptr<ParallelRule<S, FL>> &>());
}

template <typename S, typename FL> void bind_fl_rule(py::module &m) {

    py::class_<Rule<S, FL>, shared_ptr<Rule<S, FL>>>(m, "Rule")
        .def(py::init<>())
        .def("__call__", &Rule<S, FL>::operator());

    py::class_<NoTransposeRule<S, FL>, shared_ptr<NoTransposeRule<S, FL>>,
               Rule<S, FL>>(m, "NoTransposeRule")
        .def_readwrite("prim_rule", &NoTransposeRule<S, FL>::prim_rule)
        .def(py::init<const shared_ptr<Rule<S, FL>> &>());
}

template <typename S, typename FL>
void bind_core(py::module &m, const string &name, const string &fname) {

    if (is_same<typename GMatrix<FL>::FP, FL>::value && fname == "Double") {
        bind_cg<S>(m);
        bind_expr<S>(m);
        bind_state_info<S>(m, name);
        bind_sparse<S>(m);
        bind_parallel<S>(m);
    }

    bind_fl_expr<S, FL>(m);
    bind_fl_state_info<S, FL>(m, fname);
    bind_fl_sparse<S, FL>(m);
    bind_fl_operator<S, FL>(m);
    bind_fl_hamiltonian<S, FL>(m);
    bind_fl_parallel<S, FL>(m);
    bind_fl_rule<S, FL>(m);
}

template <typename S, typename T>
void bind_trans_state_info(py::module &m, const string &aux_name) {

    m.def(("trans_state_info_to_" + aux_name).c_str(),
          &TransStateInfo<S, T>::forward);
}

template <typename S, typename FL1, typename FL2>
void bind_trans_sparse_matrix(py::module &m, const string &aux_name) {

    m.def(("trans_sparse_matrix_to_" + aux_name).c_str(),
          &TransSparseMatrix<S, FL1, FL2>::forward);
}

template <typename S, typename T>
void bind_trans_state_info_spin_specific(py::module &m,
                                         const string &aux_name) {

    m.def(("trans_connection_state_info_to_" + aux_name).c_str(),
          &TransStateInfo<T, S>::backward_connection);
}

template <typename T, int L>
void bind_array_fixed(py::module &m, const string &name) {

    py::class_<array<T, L>>(m, name.c_str())
        .def("__setitem__",
             [](array<T, L> *self, size_t i, int t) { (*self)[i] = (T)t; })
        .def("__getitem__",
             [](array<T, L> *self, size_t i) { return (*self)[i]; })
        .def("__len__", [](array<T, L> *self) { return self->size(); })
        .def("__repr__",
             [](array<T, L> *self) {
                 stringstream ss;
                 ss << "(LEN=" << self->size() << ")[";
                 for (auto x : *self)
                     ss << (int64_t)x << ",";
                 ss << "]";
                 return ss.str();
             })
        .def("__iter__", [](array<T, L> *self) {
            return py::make_iterator<
                py::return_value_policy::reference_internal, T *, T *, T &>(
                &(*self)[0], &(*self)[0] + self->size());
        });
}

template <typename S = void> void bind_data(py::module &m) {

    py::bind_vector<vector<int>>(m, "VectorInt");
    py::bind_vector<vector<char>>(m, "VectorChar");
    py::bind_vector<vector<long long int>>(m, "VectorLLInt");
    py::bind_vector<vector<pair<int, int>>>(m, "VectorPIntInt");
    py::bind_vector<vector<pair<long long int, int>>>(m, "VectorPLLIntInt");
    py::bind_vector<vector<pair<long long int, long long int>>>(
        m, "VectorPLLIntLLInt");
    py::bind_vector<vector<pair<unsigned long, unsigned long>>>(
        m, "VectorPULIntULInt");
    py::bind_vector<vector<int16_t>>(m, "VectorInt16");
    py::bind_vector<vector<uint16_t>>(m, "VectorUInt16");
    py::bind_vector<vector<uint32_t>>(m, "VectorUInt32");
    py::bind_vector<vector<pair<uint32_t, uint32_t>>>(m, "VectorPUInt32UInt32");
    py::bind_vector<vector<double>, shared_ptr<vector<double>>>(m,
                                                                "VectorDouble");
    py::bind_vector<vector<long double>, shared_ptr<vector<long double>>>(
        m, "VectorLDouble");
    py::bind_vector<vector<complex<double>>,
                    shared_ptr<vector<complex<double>>>>(m,
                                                         "VectorComplexDouble");
    py::bind_vector<vector<complex<long double>>,
                    shared_ptr<vector<complex<long double>>>>(
        m, "VectorComplexLDouble");
    py::bind_vector<vector<pair<int, double>>>(m, "VectorPIntDouble");
    py::bind_vector<vector<pair<int, complex<double>>>>(
        m, "VectorPIntComplexDouble");
    py::bind_vector<vector<pair<vector<int>, double>>>(
        m, "VectorPVectorIntDouble");
    py::bind_vector<vector<pair<vector<int>, complex<double>>>>(
        m, "VectorPVectorIntComplexDouble");
    py::bind_vector<vector<size_t>>(m, "VectorULInt");
    py::bind_vector<vector<string>>(m, "VectorString");
    py::bind_vector<vector<vector<uint8_t>>>(m, "VectorVectorUInt8");
    py::bind_vector<vector<vector<uint16_t>>>(m, "VectorVectorUInt16");
    py::bind_vector<vector<vector<uint32_t>>>(m, "VectorVectorUInt32");
    py::bind_vector<vector<vector<long long int>>>(m, "VectorVectorLLInt");
    py::bind_vector<vector<vector<double>>>(m, "VectorVectorDouble");
    py::bind_vector<vector<vector<long double>>>(m, "VectorVectorLDouble");
    py::bind_vector<vector<vector<complex<double>>>>(
        m, "VectorVectorComplexDouble");
    py::bind_vector<vector<vector<complex<long double>>>>(
        m, "VectorVectorComplexLDouble");
    py::bind_vector<vector<vector<int>>>(m, "VectorVectorInt");
    py::bind_vector<vector<vector<vector<double>>>>(m,
                                                    "VectorVectorVectorDouble");
    py::bind_vector<vector<vector<vector<complex<double>>>>>(
        m, "VectorVectorVectorComplexDouble");
    py::bind_vector<vector<vector<vector<complex<long double>>>>>(
        m, "VectorVectorVectorComplexLDouble");
    py::bind_map<unordered_map<int, int>>(m, "MapIntInt");
    py::bind_vector<vector<unordered_map<int, int>>>(m, "VectorMapIntInt");
    py::bind_map<unordered_map<int, pair<int, int>>>(m, "MapIntPIntInt");
    py::bind_vector<vector<unordered_map<int, pair<int, int>>>>(
        m, "VectorMapIntPIntInt");
    py::bind_vector<vector<pair<double, string>>>(m, "VectorPDoubleStr");
    py::bind_map<map<vector<uint16_t>, vector<pair<double, string>>>>(
        m, "MapVectorUInt16VectorPDoubleStr");
    py::bind_vector<
        vector<map<vector<uint16_t>, vector<pair<double, string>>>>>(
        m, "VectorMapVectorUInt16VectorPDoubleStr");

#ifdef _USE_SINGLE_PREC

    py::bind_vector<vector<float>>(m, "VectorFloat");
    py::bind_vector<vector<complex<float>>>(m, "VectorComplexFloat");
    py::bind_vector<vector<vector<float>>>(m, "VectorVectorFloat");
    py::bind_vector<vector<vector<complex<float>>>>(m,
                                                    "VectorVectorComplexFloat");
    py::bind_vector<vector<vector<vector<float>>>>(m,
                                                   "VectorVectorVectorFloat");
    py::bind_vector<vector<vector<vector<complex<float>>>>>(
        m, "VectorVectorVectorComplexFloat");

#endif

    py::bind_vector<vector<uint8_t>>(m, "VectorUInt8")
        .def_property_readonly(
            "ptr",
            [](vector<uint8_t> *self) -> uint8_t * { return &(*self)[0]; })
        .def("__str__", [](vector<uint8_t> *self) {
            stringstream ss;
            ss << "VectorUInt8[ ";
            for (auto p : *self)
                ss << (int)p << " ";
            ss << "]";
            return ss.str();
        });

    bind_array_fixed<int, 4>(m, "Array4Int");
    bind_array_fixed<int16_t, 4>(m, "Array4Int16");
    bind_array_fixed<int16_t, 5>(m, "Array5Int16");
    bind_array_fixed<int16_t, 6>(m, "Array6Int16");

    py::bind_vector<vector<array<int, 4>>>(m, "VectorArray4Int");
    py::bind_vector<vector<array<int16_t, 4>>>(m, "VectorArray4Int16");
    py::bind_vector<vector<array<int16_t, 5>>>(m, "VectorArray5Int16");
    py::bind_vector<vector<array<int16_t, 6>>>(m, "VectorArray6Int16");

    bind_array<uint8_t>(m, "ArrayUInt8")
        .def("__str__", [](Array<uint8_t> *self) {
            stringstream ss;
            ss << "ArrayUInt8(LEN=" << self->n << ")[ ";
            for (size_t i = 0; i < self->n; i++)
                ss << (int)self->data[i] << " ";
            ss << "]";
            return ss.str();
        });

    bind_array<uint16_t>(m, "ArrayUInt16");
    bind_array<uint32_t>(m, "ArrayUInt32");

    if (sizeof(ubond_t) == sizeof(uint8_t)) {
        m.attr("VectorUBond") = m.attr("VectorUInt8");
        m.attr("VectorVectorUBond") = m.attr("VectorVectorUInt8");
        m.attr("ArrayUBond") = m.attr("ArrayUInt8");
    } else if (sizeof(ubond_t) == sizeof(uint16_t)) {
        m.attr("VectorUBond") = m.attr("VectorUInt16");
        m.attr("VectorVectorUBond") = m.attr("VectorVectorUInt16");
        m.attr("ArrayUBond") = m.attr("ArrayUInt16");
    } else if (sizeof(ubond_t) == sizeof(uint32_t)) {
        m.attr("VectorUBond") = m.attr("VectorUInt32");
        m.attr("VectorVectorUBond") = m.attr("VectorVectorUInt32");
        m.attr("ArrayUBond") = m.attr("ArrayUInt32");
    }

    if (sizeof(MKL_INT) == sizeof(int))
        m.attr("VectorMKLInt") = m.attr("VectorInt");
    else if (sizeof(MKL_INT) == sizeof(long long int))
        m.attr("VectorMKLInt") = m.attr("VectorLLInt");

    py::bind_map<map<string, string>>(m, "MapStrStr")
        .def(py::init([](const py::dict &dict) {
            auto mp =
                std::unique_ptr<map<string, string>>(new map<string, string>());
            for (auto item : dict)
                (*mp)[item.first.cast<string>()] = item.second.cast<string>();
            return mp.release();
        }));
    py::bind_map<map<string, double>>(m, "MapStrDouble");
    py::bind_map<map<string, complex<double>>>(m, "MapStrComplexDouble");
    py::bind_vector<vector<map<string, string>>>(m, "VectorMapStrStr");
    py::bind_vector<vector<pair<string, string>>>(m, "VectorPStrStr");
    py::bind_vector<vector<pair<string, int8_t>>>(m, "VectorPStrInt8");
    py::bind_map<map<pair<string, int8_t>, int>>(m, "MapPStrInt8Int");
}

template <typename S = void> void bind_types(py::module &m) {

    py::class_<PointGroup, shared_ptr<PointGroup>>(m, "PointGroup")
        .def_static("swap_c1", &PointGroup::swap_c1)
        .def_static("swap_ci", &PointGroup::swap_ci)
        .def_static("swap_cs", &PointGroup::swap_cs)
        .def_static("swap_c2", &PointGroup::swap_c2)
        .def_static("swap_c2h", &PointGroup::swap_c2h)
        .def_static("swap_c2v", &PointGroup::swap_c2v)
        .def_static("swap_d2", &PointGroup::swap_d2)
        .def_static("swap_d2h", &PointGroup::swap_d2h)
        .def_static("swap_nopg", &PointGroup::swap_nopg);

    py::enum_<SAnySymmTypes>(m, "SAnySymmTypes", py::arithmetic())
        .value("Nothing", SAnySymmTypes::None)
        .value("U1Fermi", SAnySymmTypes::U1Fermi)
        .value("U1", SAnySymmTypes::U1)
        .value("SU2Fermi", SAnySymmTypes::SU2Fermi)
        .value("SU2", SAnySymmTypes::SU2)
        .value("LZ", SAnySymmTypes::LZ)
        .value("AbelianPG", SAnySymmTypes::AbelianPG)
        .value("ZNFermi", SAnySymmTypes::ZNFermi)
        .value("ZNFermiMax", SAnySymmTypes::ZNFermiMax)
        .value("ZN", SAnySymmTypes::ZN)
        .value("ZNMax", SAnySymmTypes::ZNMax)
        .value("Invalid", SAnySymmTypes::Invalid)
        .def(py::self + int32_t())
        .def(py::self - py::self);

    py::enum_<OpNames>(m, "OpNames", py::arithmetic())
        .value("H", OpNames::H)
        .value("I", OpNames::I)
        .value("N", OpNames::N)
        .value("NN", OpNames::NN)
        .value("C", OpNames::C)
        .value("D", OpNames::D)
        .value("R", OpNames::R)
        .value("RD", OpNames::RD)
        .value("A", OpNames::A)
        .value("AD", OpNames::AD)
        .value("P", OpNames::P)
        .value("PD", OpNames::PD)
        .value("B", OpNames::B)
        .value("BD", OpNames::BD)
        .value("Q", OpNames::Q)
        .value("TR", OpNames::TR)
        .value("TS", OpNames::TS)
        .value("Zero", OpNames::Zero)
        .value("PDM1", OpNames::PDM1)
        .value("PDM2", OpNames::PDM2)
        .value("CCDD", OpNames::CCDD)
        .value("CCD", OpNames::CCD)
        .value("CDC", OpNames::CDC)
        .value("CDD", OpNames::CDD)
        .value("DCC", OpNames::DCC)
        .value("DCD", OpNames::DCD)
        .value("DDC", OpNames::DDC)
        .value("TEMP", OpNames::TEMP)
        .value("XL", OpNames::XL)
        .value("XR", OpNames::XR)
        .value("X", OpNames::X)
        .value("SP", OpNames::SP)
        .value("SM", OpNames::SM)
        .value("SZ", OpNames::SZ);

    py::enum_<DelayedOpNames>(m, "DelayedOpNames", py::arithmetic())
        .value("Nothing", DelayedOpNames::None)
        .value("H", DelayedOpNames::H)
        .value("Normal", DelayedOpNames::Normal)
        .value("R", DelayedOpNames::R)
        .value("RD", DelayedOpNames::RD)
        .value("P", DelayedOpNames::P)
        .value("PD", DelayedOpNames::PD)
        .value("Q", DelayedOpNames::Q)
        .value("CCDD", DelayedOpNames::CCDD)
        .value("CCD", DelayedOpNames::CCD)
        .value("CDD", DelayedOpNames::CDD)
        .value("TR", DelayedOpNames::TR)
        .value("TS", DelayedOpNames::TS)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<OpTypes>(m, "OpTypes", py::arithmetic())
        .value("Zero", OpTypes::Zero)
        .value("Elem", OpTypes::Elem)
        .value("Prod", OpTypes::Prod)
        .value("Sum", OpTypes::Sum)
        .value("ElemRef", OpTypes::ElemRef)
        .value("SumProd", OpTypes::SumProd)
        .value("ExprRef", OpTypes::ExprRef)
        .value("Counter", OpTypes::Counter);

    py::enum_<NoiseTypes>(m, "NoiseTypes", py::arithmetic())
        .value("Nothing", NoiseTypes::None)
        .value("Wavefunction", NoiseTypes::Wavefunction)
        .value("DensityMatrix", NoiseTypes::DensityMatrix)
        .value("Perturbative", NoiseTypes::Perturbative)
        .value("Collected", NoiseTypes::Collected)
        .value("Reduced", NoiseTypes::Reduced)
        .value("Unscaled", NoiseTypes::Unscaled)
        .value("LowMem", NoiseTypes::LowMem)
        .value("MidMem", NoiseTypes::MidMem)
        .value("ReducedPerturbative", NoiseTypes::ReducedPerturbative)
        .value("ReducedPerturbativeUnscaled",
               NoiseTypes::ReducedPerturbativeUnscaled)
        .value("PerturbativeUnscaled", NoiseTypes::PerturbativeUnscaled)
        .value("PerturbativeCollected", NoiseTypes::PerturbativeCollected)
        .value("PerturbativeUnscaledCollected",
               NoiseTypes::PerturbativeUnscaledCollected)
        .value("ReducedPerturbativeCollected",
               NoiseTypes::ReducedPerturbativeCollected)
        .value("ReducedPerturbativeUnscaledCollected",
               NoiseTypes::ReducedPerturbativeUnscaledCollected)
        .value("ReducedPerturbativeLowMem",
               NoiseTypes::ReducedPerturbativeLowMem)
        .value("ReducedPerturbativeUnscaledLowMem",
               NoiseTypes::ReducedPerturbativeUnscaledLowMem)
        .value("ReducedPerturbativeCollectedLowMem",
               NoiseTypes::ReducedPerturbativeCollectedLowMem)
        .value("ReducedPerturbativeUnscaledCollectedLowMem",
               NoiseTypes::ReducedPerturbativeUnscaledCollectedLowMem)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<SeqTypes>(m, "SeqTypes", py::arithmetic())
        .value("Nothing", SeqTypes::None)
        .value("Simple", SeqTypes::Simple)
        .value("Auto", SeqTypes::Auto)
        .value("Tasked", SeqTypes::Tasked)
        .value("SimpleTasked", SeqTypes::SimpleTasked)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<DavidsonTypes>(m, "DavidsonTypes", py::arithmetic())
        .value("GreaterThan", DavidsonTypes::GreaterThan)
        .value("LessThan", DavidsonTypes::LessThan)
        .value("CloseTo", DavidsonTypes::CloseTo)
        .value("Harmonic", DavidsonTypes::Harmonic)
        .value("HarmonicGreaterThan", DavidsonTypes::HarmonicGreaterThan)
        .value("HarmonicLessThan", DavidsonTypes::HarmonicLessThan)
        .value("HarmonicCloseTo", DavidsonTypes::HarmonicCloseTo)
        .value("DavidsonPrecond", DavidsonTypes::DavidsonPrecond)
        .value("NoPrecond", DavidsonTypes::NoPrecond)
        .value("Normal", DavidsonTypes::Normal)
        .value("NonHermitian", DavidsonTypes::NonHermitian)
        .value("Exact", DavidsonTypes::Exact)
        .value("ExactNonHermitian", DavidsonTypes::ExactNonHermitian)
        .value("NonHermitianDavidsonPrecond",
               DavidsonTypes::NonHermitianDavidsonPrecond)
        .value("LeftEigen", DavidsonTypes::LeftEigen)
        .value("ElementProj", DavidsonTypes::ElementProj)
        .value("ExactNonHermitianLeftEigen",
               DavidsonTypes::ExactNonHermitianLeftEigen)
        .value("NonHermitianDavidsonPrecondLeftEigen",
               DavidsonTypes::NonHermitianDavidsonPrecondLeftEigen)
        .value("NonHermitianLeftEigen", DavidsonTypes::NonHermitianLeftEigen)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<SparseMatrixTypes>(m, "SparseMatrixTypes", py::arithmetic())
        .value("Normal", SparseMatrixTypes::Normal)
        .value("CSR", SparseMatrixTypes::CSR)
        .value("Archived", SparseMatrixTypes::Archived)
        .value("Delayed", SparseMatrixTypes::Delayed);

    py::enum_<ParallelOpTypes>(m, "ParallelOpTypes", py::arithmetic())
        .value("None", ParallelOpTypes::None)
        .value("Repeated", ParallelOpTypes::Repeated)
        .value("Number", ParallelOpTypes::Number)
        .value("Partial", ParallelOpTypes::Partial);

    py::enum_<TensorFunctionsTypes>(m, "TensorFunctionsTypes", py::arithmetic())
        .value("Normal", TensorFunctionsTypes::Normal)
        .value("Archived", TensorFunctionsTypes::Archived)
        .value("Delayed", TensorFunctionsTypes::Delayed);

    py::enum_<OperatorTensorTypes>(m, "OperatorTensorTypes", py::arithmetic())
        .value("Normal", OperatorTensorTypes::Normal)
        .value("Delayed", OperatorTensorTypes::Delayed);

    py::enum_<TraceTypes>(m, "TraceTypes", py::arithmetic())
        .value("Nothing", TraceTypes::None)
        .value("Left", TraceTypes::Left)
        .value("Right", TraceTypes::Right);

    py::enum_<ParallelCommTypes>(m, "ParallelCommTypes", py::arithmetic())
        .value("Nothing", ParallelCommTypes::None)
        .value("NonBlocking", ParallelCommTypes::NonBlocking)
        .def(py::self & py::self)
        .def(py::self | py::self);

    py::enum_<ParallelRulePartitionTypes>(m, "ParallelRulePartitionTypes",
                                          py::arithmetic())
        .value("Left", ParallelRulePartitionTypes::Left)
        .value("Right", ParallelRulePartitionTypes::Right)
        .value("Middle", ParallelRulePartitionTypes::Middle);

    py::enum_<ParallelTypes>(m, "ParallelTypes", py::arithmetic())
        .value("Serial", ParallelTypes::Serial)
        .value("Distributed", ParallelTypes::Distributed)
        .value("NewScheme", ParallelTypes::NewScheme)
        .value("Simple", ParallelTypes::Simple)
        .def(py::self & py::self)
        .def(py::self | py::self)
        .def(py::self ^ py::self);

    py::enum_<SpinOperator>(m, "SpinOperator", py::arithmetic())
        .value("C", SpinOperator::C)
        .value("D", SpinOperator::D)
        .value("CA", SpinOperator::CA)
        .value("CB", SpinOperator::CB)
        .value("DA", SpinOperator::DA)
        .value("DB", SpinOperator::DB)
        .value("S", SpinOperator::S)
        .value("SP", SpinOperator::SP)
        .value("SZ", SpinOperator::SZ)
        .value("SM", SpinOperator::SM)
        .def(py::self & py::self)
        .def(py::self ^ py::self);

    py::enum_<GeneralSymmOperator>(m, "GeneralSymmOperator", py::arithmetic())
        .value("C", GeneralSymmOperator::C)
        .value("D", GeneralSymmOperator::D)
        .value("E", GeneralSymmOperator::E)
        .value("F", GeneralSymmOperator::F)
        .value("G", GeneralSymmOperator::G)
        .value("S", GeneralSymmOperator::S)
        .def(py::self & py::self)
        .def(py::self ^ py::self);

    py::enum_<ElemOpTypes>(m, "ElemOpTypes", py::arithmetic())
        .value("SU2", ElemOpTypes::SU2)
        .value("SZ", ElemOpTypes::SZ)
        .value("SGF", ElemOpTypes::SGF)
        .value("SGB", ElemOpTypes::SGB)
        .value("SAny", ElemOpTypes::SAny);

    py::bind_vector<vector<pair<SpinOperator, uint16_t>>>(
        m, "VectorPSpinOpUInt16");
}

template <typename S = void> void bind_io(py::module &m) {

    m.def("set_mkl_num_threads", [](int n) {
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(n);
        mkl_set_dynamic(0);
        threading_()->n_threads_mkl = n;
        threading_()->type = threading_()->type | ThreadingTypes::BatchedGEMM;
#else
        throw runtime_error("cannot set number of mkl threads.");
#endif
    });
    m.def("set_omp_num_threads", [](int n) {
#ifdef _OPENMP
        omp_set_num_threads(n);
        threading_()->n_threads_global = n;
        threading_()->type = threading_()->type | ThreadingTypes::Global;
#else
        if(n != 1)
            throw runtime_error("cannot set number of omp threads.");
#endif
    });

    struct PyCallbackKernel : CallbackKernel {
        using CallbackKernel::CallbackKernel;
        typedef CallbackKernel super_t;
        void compute(const string &name, int iprint) const override {
            py::gil_scoped_acquire gil;
            py::function py_method = py::get_override(this, "compute");
            if (py_method)
                py_method(name, iprint);
            else
                super_t::compute(name, iprint);
        }
    };

    py::class_<CallbackKernel, shared_ptr<CallbackKernel>, PyCallbackKernel>(
        m, "CallbackKernel")
        .def(py::init<>())
        .def("compute", [](CallbackKernel *self, const string &p, int i) {
            self->compute(p, i);
        });

    m.def("set_callback",
          [](shared_ptr<CallbackKernel> kernel) { callback_() = kernel; });

    py::class_<Allocator<uint32_t>, shared_ptr<Allocator<uint32_t>>>(
        m, "IntAllocator")
        .def(py::init<>());

    py::class_<VectorAllocator<uint32_t>, shared_ptr<VectorAllocator<uint32_t>>,
               Allocator<uint32_t>>(m, "IntVectorAllocator")
        .def_readwrite("data", &VectorAllocator<uint32_t>::data)
        .def(py::init<>());

    py::class_<StackAllocator<uint32_t>, shared_ptr<StackAllocator<uint32_t>>,
               Allocator<uint32_t>>(m, "IntStackAllocator")
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<uint32_t>::size)
        .def_readwrite("used", &StackAllocator<uint32_t>::used)
        .def_readwrite("shift", &StackAllocator<uint32_t>::shift);

    struct Global {};

    py::class_<KuhnMunkres, shared_ptr<KuhnMunkres>>(m, "KuhnMunkres")
        .def(py::init([](const py::array_t<double> &cost) {
            assert(cost.ndim() == 2);
            const int m = (int)cost.shape()[0], n = (int)cost.shape()[1];
            const int asi = (int)cost.strides()[0] / sizeof(double),
                      asj = (int)cost.strides()[1] / sizeof(double);
            vector<double> arr(m * n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    arr[i * n + j] = cost.data()[asi * i + asj * j];
            return make_shared<KuhnMunkres>(arr, m, n);
        }))
        .def("solve", [](KuhnMunkres *self) {
            auto r = self->solve();
            py::array_t<int> idx(r.second.size());
            memcpy(idx.mutable_data(), r.second.data(),
                   r.second.size() * sizeof(int));
            return make_pair(r.first, idx);
        });

    py::class_<Flow, shared_ptr<Flow>>(m, "Flow")
        .def(py::init<int>())
        .def_readwrite("n", &Flow::n)
        .def_readwrite("nfs", &Flow::nfs)
        .def_readwrite("inf", &Flow::inf)
        .def_readwrite("resi", &Flow::resi)
        .def_readwrite("dist", &Flow::dist)
        .def("mvc", &Flow::mvc)
        .def("dinic", &Flow::dinic)
        .def("ddfs", &Flow::ddfs)
        .def("dbfs", &Flow::dbfs);

    py::class_<CostFlow, shared_ptr<CostFlow>>(m, "CostFlow")
        .def(py::init<int>())
        .def_readwrite("n", &CostFlow::n)
        .def_readwrite("nfs", &CostFlow::nfs)
        .def_readwrite("inf", &CostFlow::inf)
        .def_readwrite("resi", &CostFlow::resi)
        .def_readwrite("dist", &CostFlow::dist)
        .def_readwrite("pre", &CostFlow::pre)
        .def("mvc", &CostFlow::mvc)
        .def("sap", &CostFlow::sap)
        .def("spfa", &CostFlow::spfa);

    py::class_<Prime, shared_ptr<Prime>>(m, "Prime")
        .def(py::init<>())
        .def_readwrite("primes", &Prime::primes)
        .def("factors",
             [](Prime *self, Prime::LL n) {
                 vector<pair<Prime::LL, int>> factors;
                 self->factors(n, factors);
                 return factors;
             })
        .def_static("miller_rabin", &Prime::miller_rabin)
        .def_static("power", &Prime::power)
        .def_static("sqrt", &Prime::sqrt)
        .def_static("gcd", &Prime::gcd)
        .def_static("inv", &Prime::inv)
        .def_static("quick_multiply", &Prime::quick_multiply)
        .def_static("quick_power", &Prime::quick_power)
        .def("is_prime", &Prime::is_prime);

    py::class_<Combinatorics, shared_ptr<Combinatorics>>(m, "Combinatorics")
        .def(py::init<int>())
        .def("combination", &Combinatorics::combination);

    py::class_<FFT, shared_ptr<FFT>>(m, "FFT")
        .def(py::init<>())
        .def("init", &FFT::init)
        .def("fft",
             [](FFT *self, const py::array_t<complex<double>> &arr) {
                 py::array_t<complex<double>> arx =
                     py::array_t<complex<double>>(arr.size());
                 memcpy(arx.mutable_data(), arr.data(),
                        arr.size() * sizeof(complex<double>));
                 self->fft(arx.mutable_data(), arx.size(), true);
                 return arx;
             })
        .def("ifft",
             [](FFT *self, const py::array_t<complex<double>> &arr) {
                 py::array_t<complex<double>> arx =
                     py::array_t<complex<double>>(arr.size());
                 memcpy(arx.mutable_data(), arr.data(),
                        arr.size() * sizeof(complex<double>));
                 self->fft(arx.mutable_data(), arx.size(), false);
                 return arx;
             })
        .def_static("fftshift",
                    [](const py::array_t<complex<double>> &arr) {
                        py::array_t<complex<double>> arx =
                            py::array_t<complex<double>>(arr.size());
                        memcpy(arx.mutable_data(), arr.data(),
                               arr.size() * sizeof(complex<double>));
                        FFT::fftshift(arx.mutable_data(), arx.size(), true);
                        return arx;
                    })
        .def_static("fftshift",
                    [](const py::array_t<double> &arr) {
                        py::array_t<double> arx =
                            py::array_t<double>(arr.size());
                        memcpy(arx.mutable_data(), arr.data(),
                               arr.size() * sizeof(double));
                        FFT::fftshift(arx.mutable_data(), arx.size(), true);
                        return arx;
                    })
        .def_static("ifftshift",
                    [](const py::array_t<complex<double>> &arr) {
                        py::array_t<complex<double>> arx =
                            py::array_t<complex<double>>(arr.size());
                        memcpy(arx.mutable_data(), arr.data(),
                               arr.size() * sizeof(complex<double>));
                        FFT::fftshift(arx.mutable_data(), arx.size(), false);
                        return arx;
                    })
        .def_static("fftfreq", [](long long int n, double d) {
            py::array_t<double> arx = py::array_t<double>(n);
            FFT::fftfreq(arx.mutable_data(), n, d);
            return arx;
        });

    py::class_<DFT, shared_ptr<DFT>>(m, "DFT")
        .def(py::init<>())
        .def("init", &DFT::init)
        .def("fft",
             [](DFT *self, const py::array_t<complex<double>> &arr) {
                 py::array_t<complex<double>> arx =
                     py::array_t<complex<double>>(arr.size());
                 memcpy(arx.mutable_data(), arr.data(),
                        arr.size() * sizeof(complex<double>));
                 self->fft(arx.mutable_data(), arx.size(), true);
                 return arx;
             })
        .def("ifft", [](DFT *self, const py::array_t<complex<double>> &arr) {
            py::array_t<complex<double>> arx =
                py::array_t<complex<double>>(arr.size());
            memcpy(arx.mutable_data(), arr.data(),
                   arr.size() * sizeof(complex<double>));
            self->fft(arx.mutable_data(), arx.size(), false);
            return arx;
        });

    py::enum_<ThreadingTypes>(m, "ThreadingTypes", py::arithmetic())
        .value("SequentialGEMM", ThreadingTypes::SequentialGEMM)
        .value("BatchedGEMM", ThreadingTypes::BatchedGEMM)
        .value("Quanta", ThreadingTypes::Quanta)
        .value("QuantaBatchedGEMM", ThreadingTypes::QuantaBatchedGEMM)
        .value("Operator", ThreadingTypes::Operator)
        .value("OperatorBatchedGEMM", ThreadingTypes::OperatorBatchedGEMM)
        .value("OperatorQuanta", ThreadingTypes::OperatorQuanta)
        .value("OperatorQuantaBatchedGEMM",
               ThreadingTypes::OperatorQuantaBatchedGEMM)
        .value("Global", ThreadingTypes::Global)
        .def(py::self & py::self)
        .def(py::self ^ py::self)
        .def(py::self | py::self);

    py::class_<Threading, shared_ptr<Threading>>(m, "Threading")
        .def(py::init<>())
        .def(py::init<ThreadingTypes>())
        .def(py::init<ThreadingTypes, int>())
        .def(py::init<ThreadingTypes, int, int>())
        .def(py::init<ThreadingTypes, int, int, int>())
        .def(py::init<ThreadingTypes, int, int, int, int>())
        .def_readwrite("type", &Threading::type)
        .def_readwrite("seq_type", &Threading::seq_type)
        .def_readwrite("n_threads_op", &Threading::n_threads_op)
        .def_readwrite("n_threads_quanta", &Threading::n_threads_quanta)
        .def_readwrite("n_threads_mkl", &Threading::n_threads_mkl)
        .def_readwrite("n_threads_global", &Threading::n_threads_global)
        .def_readwrite("n_levels", &Threading::n_levels)
        .def("openmp_available", &Threading::openmp_available)
        .def("mkl_available", &Threading::mkl_available)
        .def("tbb_available", &Threading::tbb_available)
        .def("activate_global", &Threading::activate_global)
        .def("activate_normal", &Threading::activate_normal)
        .def("activate_operator", &Threading::activate_operator)
        .def("activate_quanta", &Threading::activate_quanta)
        .def("__repr__", [](Threading *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<StackAllocator<uint32_t>>>>(
        m, "VectorIntStackAllocator");

    py::class_<Global>(m, "Global")
        .def_property_static(
            "ialloc", [](py::object) { return ialloc_(); },
            [](py::object, shared_ptr<StackAllocator<uint32_t>> ia) {
                ialloc_() = ia;
            })
        .def_property_static(
            "dalloc", [](py::object) { return dalloc_<double>(); },
            [](py::object, shared_ptr<StackAllocator<double>> da) {
                dalloc_<double>() = da;
            })
        .def_property_static(
            "frame", [](py::object) { return frame_<double>(); },
            [](py::object, shared_ptr<DataFrame<double>> fr) {
                frame_<double>() = fr;
            })
#ifdef _USE_SINGLE_PREC
        .def_property_static(
            "dalloc_float", [](py::object) { return dalloc_<float>(); },
            [](py::object, shared_ptr<StackAllocator<float>> da) {
                dalloc_<float>() = da;
            })
        .def_property_static(
            "frame_float", [](py::object) { return frame_<float>(); },
            [](py::object, shared_ptr<DataFrame<float>> fr) {
                frame_<float>() = fr;
            })
#endif
        .def_property_static(
            "threading", [](py::object) { return threading_(); },
            [](py::object, shared_ptr<Threading> th) { threading_() = th; });

    py::class_<Random, shared_ptr<Random>>(m, "Random")
        .def_static("rand_seed", &Random::rand_seed, py::arg("i") = 0U)
        .def_static("rand_int", &Random::rand_int, py::arg("a"), py::arg("b"))
        .def_static("rand_double", &Random::rand_double, py::arg("a") = 0,
                    py::arg("b") = 1)
        .def_static("fill_rand_double", [](py::array_t<double> &data,
                                           double a = 0, double b = 1) {
            return Random::fill<double>(data.mutable_data(), data.size(), a, b);
        });

    py::class_<Parsing, shared_ptr<Parsing>>(m, "Parsing")
        .def_static("to_size_string", &Parsing::to_size_string,
                    py::arg("i") = (size_t)0U, py::arg("suffix") = "B");

    py::class_<ParallelProperty, shared_ptr<ParallelProperty>>(
        m, "ParallelProperty")
        .def_readwrite("owner", &ParallelProperty::owner)
        .def_readwrite("ptype", &ParallelProperty::ptype)
        .def(py::init<>())
        .def(py::init<int, ParallelOpTypes>());

#ifdef _HAS_MPI
    py::class_<block2::MPI, shared_ptr<block2::MPI>>(m, "MPI")
        .def_readwrite("_rank", &block2::MPI::_rank)
        .def_readwrite("_size", &block2::MPI::_size)
        .def(py::init<>())
        .def_static("mpi", &block2::MPI::mpi)
        .def_static("rank", &block2::MPI::rank)
        .def_static("size", &block2::MPI::size);
#endif

    py::class_<SU2CG, shared_ptr<SU2CG>>(m, "SU2CG")
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("triangle", &SU2CG::triangle, py::arg("tja"),
                    py::arg("tjb"), py::arg("tjc"))
        .def("sqrt_delta", &SU2CG::sqrt_delta, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"))
        .def("cg", &SU2CG::cg, py::arg("tja"), py::arg("tjb"), py::arg("tjc"),
             py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_3j", &SU2CG::wigner_3j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &SU2CG::wigner_6j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"))
        .def("wigner_9j", &SU2CG::wigner_9j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"),
             py::arg("tjg"), py::arg("tjh"), py::arg("tji"))
        .def("racah", &SU2CG::racah, py::arg("ta"), py::arg("tb"),
             py::arg("tc"), py::arg("td"), py::arg("te"), py::arg("tf"))
        .def("transpose_cg", &SU2CG::transpose_cg, py::arg("td"), py::arg("tl"),
             py::arg("tr"))
        .def("phase", &SU2CG::phase, py::arg("ta"), py::arg("tb"),
             py::arg("tc"))
        .def("wigner_d", &SU2CG::wigner_d, py::arg("tj"), py::arg("tmp"),
             py::arg("tms"), py::arg("beta"));

    py::class_<SO3RSHCG, shared_ptr<SO3RSHCG>, SU2CG>(m, "SO3RSHCG")
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("u_star", &SO3RSHCG::u_star, py::arg("tm1"), py::arg("tm1"))
        .def("cg", &SO3RSHCG::cg, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_3j", &SO3RSHCG::wigner_3j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &SO3RSHCG::wigner_6j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"))
        .def("wigner_9j", &SO3RSHCG::wigner_9j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"),
             py::arg("tjg"), py::arg("tjh"), py::arg("tji"))
        .def("racah", &SO3RSHCG::racah, py::arg("ta"), py::arg("tb"),
             py::arg("tc"), py::arg("td"), py::arg("te"), py::arg("tf"))
        .def("transpose_cg", &SO3RSHCG::transpose_cg, py::arg("td"),
             py::arg("tl"), py::arg("tr"))
        .def("phase", &SO3RSHCG::phase, py::arg("ta"), py::arg("tb"),
             py::arg("tc"));

    py::class_<GeneralSymmElement, shared_ptr<GeneralSymmElement>>(
        m, "GeneralSymmElement")
        .def(py::init<GeneralSymmOperator, uint16_t, int16_t>())
        .def(py::init<GeneralSymmOperator, uint16_t, const vector<int16_t> &>())
        .def_readwrite("op", &GeneralSymmElement::op)
        .def_readwrite("index", &GeneralSymmElement::index)
        .def_readwrite("tms", &GeneralSymmElement::tms)
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("to_str", &GeneralSymmElement::to_str)
        .def("__repr__", &GeneralSymmElement::to_str)
        .def("__hash__", &GeneralSymmElement::hash);

    py::bind_vector<vector<GeneralSymmElement>>(m, "VectorGeneralSymmElement");

    py::class_<SpinPermTerm, shared_ptr<SpinPermTerm>>(m, "SpinPermTerm")
        .def(py::init<>())
        .def(py::init<SpinOperator, uint16_t>())
        .def(py::init<SpinOperator, uint16_t, double>())
        .def(py::init<const vector<pair<SpinOperator, uint16_t>> &>())
        .def(py::init<const vector<pair<SpinOperator, uint16_t>> &, double>())
        .def_readwrite("factor", &SpinPermTerm::factor)
        .def_readwrite("ops", &SpinPermTerm::ops)
        .def(-py::self)
        .def(py::self * double())
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("ops_equal_to", &SpinPermTerm::ops_equal_to)
        .def("to_str", &SpinPermTerm::to_str);

    py::bind_vector<vector<SpinPermTerm>>(m, "VectorSpinPermTerm");
    py::bind_vector<vector<vector<SpinPermTerm>>>(m,
                                                  "VectorVectorSpinPermTerm");

    py::class_<SpinPermTensor, shared_ptr<SpinPermTensor>>(m, "SpinPermTensor")
        .def(py::init<>())
        .def(py::init<const vector<vector<SpinPermTerm>> &>())
        .def(py::init<const vector<vector<SpinPermTerm>> &,
                      const vector<int16_t> &>())
        .def_readwrite("data", &SpinPermTensor::data)
        .def_readwrite("tjs", &SpinPermTensor::tjs)
        .def_static("I", &SpinPermTensor::I)
        .def_static("C", &SpinPermTensor::C)
        .def_static("D", &SpinPermTensor::D)
        .def_static("T", &SpinPermTensor::T)
        .def_static("permutation_parity", &SpinPermTensor::permutation_parity)
        .def_static("find_pattern_perm", &SpinPermTensor::find_pattern_perm)
        .def_static("auto_sort_string",
                    (pair<string, int>(*)(const vector<uint16_t> &,
                                          const string &, const string &)) &
                        SpinPermTensor::auto_sort_string)
        .def_static(
            "auto_sort_string",
            (pair<string, int>(*)(const vector<uint16_t> &, const string &)) &
                SpinPermTensor::auto_sort_string)
        .def_static("mul", (SpinPermTensor(*)(const SpinPermTensor &,
                                              const SpinPermTensor &, int16_t,
                                              const SU2CG &)) &
                               SpinPermTensor::mul)
        .def_static("mul", (SpinPermTensor(*)(
                               const SpinPermTensor &, const SpinPermTensor &,
                               const vector<int16_t> &, const SU2CG &)) &
                               SpinPermTensor::mul)
        .def_static("dot_product", &SpinPermTensor::dot_product)
        .def("simplify", &SpinPermTensor::simplify)
        .def("auto_sort", &SpinPermTensor::auto_sort)
        .def("normal_sort", &SpinPermTensor::normal_sort)
        .def("get_cds", &SpinPermTensor::get_cds)
        .def("equal_to_scaled", &SpinPermTensor::equal_to_scaled)
        .def("to_str", &SpinPermTensor::to_str)
        .def("__repr__", [](SpinPermTensor *self) { return self->to_str(); })
        .def(py::self * double())
        .def(py::self + py::self)
        .def(py::self == py::self);

    py::bind_vector<vector<SpinPermTensor>>(m, "VectorSpinPermTensor");

    py::class_<SpinPermRecoupling, shared_ptr<SpinPermRecoupling>>(
        m, "SpinPermRecoupling")
        .def_static("to_str", &SpinPermRecoupling::to_str)
        .def_static("make_cds", &SpinPermRecoupling::make_cds)
        .def_static("make_with_cds", &SpinPermRecoupling::make_with_cds)
        .def_static("get_target_twos", &SpinPermRecoupling::get_target_twos)
        .def_static("split_cds", &SpinPermRecoupling::split_cds)
        .def_static("count_cds", &SpinPermRecoupling::count_cds)
        .def_static("make_tensor", &SpinPermRecoupling::make_tensor)
        .def_static("get_sub_expr", &SpinPermRecoupling::get_sub_expr)
        .def_static("find_split_index", &SpinPermRecoupling::find_split_index)
        .def_static("find_split_indices_from_left",
                    &SpinPermRecoupling::find_split_indices_from_left,
                    py::arg("x"), py::arg("start_depth") = 1)
        .def_static("find_split_indices_from_right",
                    &SpinPermRecoupling::find_split_indices_from_right,
                    py::arg("x"), py::arg("start_depth") = 1)
        .def_static("initialize", &SpinPermRecoupling::initialize, py::arg("n"),
                    py::arg("twos"), py::arg("site_dq") = 1);

    py::class_<SpinRecoupling, shared_ptr<SpinRecoupling>>(m, "SpinRecoupling")
        .def_static("get_level", &SpinRecoupling::get_level)
        .def_static("get_quanta", &SpinRecoupling::get_quanta)
        .def_static("get_twos", &SpinRecoupling::get_twos)
        .def_static("recouple", &SpinRecoupling::recouple)
        .def_static("exchange", &SpinRecoupling::exchange)
        .def_static("sort_indices", &SpinRecoupling::sort_indices)
        .def_static("recouple_split", &SpinRecoupling::recouple_split);

    py::class_<typename SpinRecoupling::Level,
               shared_ptr<typename SpinRecoupling::Level>>(
        m, "SpinRecouplingLevel")
        .def(py::init<>())
        .def_readwrite("left_idx", &SpinRecoupling::Level::left_idx)
        .def_readwrite("mid_idx", &SpinRecoupling::Level::mid_idx)
        .def_readwrite("right_idx", &SpinRecoupling::Level::right_idx)
        .def_readwrite("left_cnt", &SpinRecoupling::Level::left_cnt)
        .def_readwrite("right_cnt", &SpinRecoupling::Level::right_cnt);

    py::class_<SpinPermPattern, shared_ptr<SpinPermPattern>>(m,
                                                             "SpinPermPattern")
        .def_readwrite("n", &SpinPermPattern::n)
        .def_readwrite("mask", &SpinPermPattern::mask)
        .def_readwrite("data", &SpinPermPattern::data)
        .def(py::init<uint16_t>())
        .def(py::init<uint16_t, const vector<uint16_t> &>())
        .def_static("all_reordering", &SpinPermPattern::all_reordering,
                    py::arg("x"), py::arg("mask") = vector<uint16_t>())
        .def_static("initialize", &SpinPermPattern::initialize, py::arg("n"),
                    py::arg("mask") = vector<uint16_t>())
        .def_static("get_unique", &SpinPermPattern::get_unique)
        .def_static("make_matrix", &SpinPermPattern::make_matrix)
        .def("count", &SpinPermPattern::count)
        .def("__getitem__", &SpinPermPattern::operator[])
        .def("get_split_index", &SpinPermPattern::get_split_index)
        .def("to_str", &SpinPermPattern::to_str);

    py::class_<GeneralSymmPermPattern, shared_ptr<GeneralSymmPermPattern>>(
        m, "GeneralSymmPermPattern")
        .def_readwrite("n", &GeneralSymmPermPattern::n)
        .def_readwrite("mask", &GeneralSymmPermPattern::mask)
        .def_readwrite("data", &GeneralSymmPermPattern::data)
        .def(py::init<uint16_t>())
        .def(py::init<uint16_t, const vector<uint16_t> &>())
        .def_static("all_reordering", &GeneralSymmPermPattern::all_reordering,
                    py::arg("x"), py::arg("mask") = vector<uint16_t>())
        .def_static("initialize", &GeneralSymmPermPattern::initialize,
                    py::arg("n"), py::arg("mask") = vector<uint16_t>())
        .def("count", &GeneralSymmPermPattern::count)
        .def("__getitem__", &GeneralSymmPermPattern::operator[])
        .def("get_split_index", &GeneralSymmPermPattern::get_split_index)
        .def("to_str", &GeneralSymmPermPattern::to_str);

    py::class_<SpinPermScheme, shared_ptr<SpinPermScheme>>(m, "SpinPermScheme")
        .def(py::init<>())
        .def(py::init<string>())
        .def(py::init<string, bool>())
        .def(py::init<string, bool, bool>())
        .def(py::init<string, bool, bool, bool>())
        .def(py::init<string, bool, bool, bool, bool>())
        .def(py::init<string, bool, bool, bool, bool,
                      const vector<uint16_t> &>())
        .def_readwrite("index_patterns", &SpinPermScheme::index_patterns)
        .def_readwrite("data", &SpinPermScheme::data)
        .def_readwrite("is_su2", &SpinPermScheme::is_su2)
        .def_readwrite("left_vacuum", &SpinPermScheme::left_vacuum)
        .def_readwrite("mask", &SpinPermScheme::mask)
        .def_static("initialize_sz", &SpinPermScheme::initialize_sz,
                    py::arg("nn"), py::arg("spin_str"),
                    py::arg("is_fermion") = true,
                    py::arg("mask") = vector<uint16_t>())
        .def_static("initialize_sany", &SpinPermScheme::initialize_sany,
                    py::arg("nn"), py::arg("spin_str"),
                    py::arg("fermionic_ops") = string("cdCD"),
                    py::arg("mask") = vector<uint16_t>())
        .def_static("initialize_su2_old", &SpinPermScheme::initialize_su2_old,
                    py::arg("nn"), py::arg("spin_str"),
                    py::arg("is_npdm") = false)
        .def_static("initialize_su2", &SpinPermScheme::initialize_su2,
                    py::arg("nn"), py::arg("spin_str"),
                    py::arg("is_npdm") = false, py::arg("is_drt") = false,
                    py::arg("mask") = vector<uint16_t>())
        .def_static("initialize_su2_old2", &SpinPermScheme::initialize_su2_old2,
                    py::arg("nn"), py::arg("spin_str"),
                    py::arg("is_npdm") = false, py::arg("is_drt") = false,
                    py::arg("mask") = vector<uint16_t>())
        .def("to_str", &SpinPermScheme::to_str);

    py::bind_vector<vector<shared_ptr<SpinPermScheme>>>(m,
                                                        "VectorSpinPermScheme");

    py::class_<NPDMCounter, shared_ptr<NPDMCounter>>(m, "NPDMCounter")
        .def(py::init<int, int>())
        .def_readwrite("n_ops", &NPDMCounter::n_ops)
        .def_readwrite("n_sites", &NPDMCounter::n_sites)
        .def_readwrite("dp", &NPDMCounter::dp)
        .def("count_left", &NPDMCounter::count_left)
        .def("init_left", &NPDMCounter::init_left)
        .def("next_left", &NPDMCounter::next_left)
        .def("find_left", &NPDMCounter::find_left)
        .def("count_right", &NPDMCounter::count_right)
        .def("init_right", &NPDMCounter::init_right)
        .def("index_right", &NPDMCounter::index_right)
        .def("next_right", &NPDMCounter::next_right);

    py::class_<NPDMScheme, shared_ptr<NPDMScheme>>(m, "NPDMScheme")
        .def(py::init<shared_ptr<SpinPermScheme>>())
        .def(py::init<const vector<shared_ptr<SpinPermScheme>> &>())
        .def_readwrite("left_terms", &NPDMScheme::left_terms)
        .def_readwrite("right_terms", &NPDMScheme::right_terms)
        .def_readwrite("middle_terms", &NPDMScheme::middle_terms)
        .def_readwrite("middle_perm_patterns",
                       &NPDMScheme::middle_perm_patterns)
        .def_readwrite("left_blocking", &NPDMScheme::left_blocking)
        .def_readwrite("right_blocking", &NPDMScheme::right_blocking)
        .def_readwrite("middle_blocking", &NPDMScheme::middle_blocking)
        .def_readwrite("last_right_terms", &NPDMScheme::last_right_terms)
        .def_readwrite("last_right_blocking", &NPDMScheme::last_right_blocking)
        .def_readwrite("last_middle_blocking",
                       &NPDMScheme::last_middle_blocking)
        .def_readwrite("local_terms", &NPDMScheme::local_terms)
        .def_readwrite("perms", &NPDMScheme::perms)
        .def_readwrite("n_max_ops", &NPDMScheme::n_max_ops)
        .def("initialize", &NPDMScheme::initialize)
        .def("to_str", &NPDMScheme::to_str);
}

template <typename FL> void bind_fl_data(py::module &m, const string &name) {

    py::class_<AnyCG<FL>, shared_ptr<AnyCG<FL>>>(m, ("AnyCG" + name).c_str())
        .def(py::init<>())
        .def("cg", &AnyCG<FL>::cg, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tma"), py::arg("tmb"), py::arg("tmc"))
        .def("wigner_6j", &AnyCG<FL>::wigner_6j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"))
        .def("wigner_9j", &AnyCG<FL>::wigner_9j, py::arg("tja"), py::arg("tjb"),
             py::arg("tjc"), py::arg("tjd"), py::arg("tje"), py::arg("tjf"),
             py::arg("tjg"), py::arg("tjh"), py::arg("tji"))
        .def("racah", &AnyCG<FL>::racah, py::arg("ta"), py::arg("tb"),
             py::arg("tc"), py::arg("td"), py::arg("te"), py::arg("tf"));

    py::bind_vector<vector<shared_ptr<AnyCG<FL>>>>(
        m, ("VectorAnyCG" + name).c_str());

    py::class_<AnySU2CG<FL>, shared_ptr<AnySU2CG<FL>>, AnyCG<FL>>(
        m, ("AnySU2CG" + name).c_str())
        .def(py::init<>());

    py::class_<AnySO3RSHCG<FL>, shared_ptr<AnySO3RSHCG<FL>>, AnyCG<FL>>(
        m, ("AnySO3RSHCG" + name).c_str())
        .def(py::init<>());

    py::class_<GeneralSymmTerm<FL>, shared_ptr<GeneralSymmTerm<FL>>>(
        m, ("GeneralSymmTerm" + name).c_str())
        .def(py::init<>())
        .def(py::init<GeneralSymmElement>())
        .def(py::init<GeneralSymmElement, FL>())
        .def(py::init<const vector<GeneralSymmElement> &>())
        .def(py::init<const vector<GeneralSymmElement> &, FL>())
        .def_readwrite("factor", &GeneralSymmTerm<FL>::factor)
        .def_readwrite("ops", &GeneralSymmTerm<FL>::ops)
        .def(-py::self)
        .def(py::self * FL())
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("ops_equal_to", &GeneralSymmTerm<FL>::ops_equal_to)
        .def("to_str", &GeneralSymmTerm<FL>::to_str)
        .def("__repr__", &GeneralSymmTerm<FL>::to_str);

    py::bind_vector<vector<GeneralSymmTerm<FL>>>(
        m, ("VectorGeneralSymmTerm" + name).c_str());
    py::bind_vector<vector<vector<GeneralSymmTerm<FL>>>>(
        m, ("VectorVectorGeneralSymmTerm" + name).c_str());

    py::class_<typename GeneralSymmTensor<FL>::Level,
               shared_ptr<typename GeneralSymmTensor<FL>::Level>>(
        m, ("GeneralSymmTensorLevel" + name).c_str())
        .def(py::init<>())
        .def_readwrite("left_idx", &GeneralSymmTensor<FL>::Level::left_idx)
        .def_readwrite("mid_idx", &GeneralSymmTensor<FL>::Level::mid_idx)
        .def_readwrite("right_idx", &GeneralSymmTensor<FL>::Level::right_idx)
        .def_readwrite("left_cnt", &GeneralSymmTensor<FL>::Level::left_cnt)
        .def_readwrite("right_cnt", &GeneralSymmTensor<FL>::Level::right_cnt);

    py::class_<GeneralSymmTensor<FL>, shared_ptr<GeneralSymmTensor<FL>>>(
        m, ("GeneralSymmTensor" + name).c_str())
        .def(py::init<>())
        .def(py::init<const vector<vector<GeneralSymmTerm<FL>>> &>())
        .def(py::init<const vector<vector<GeneralSymmTerm<FL>>> &,
                      const vector<int16_t> &>())
        .def_readwrite("data", &GeneralSymmTensor<FL>::data)
        .def_readwrite("tjs", &GeneralSymmTensor<FL>::tjs)
        .def_static("i", &GeneralSymmTensor<FL>::i)
        .def_static("c", &GeneralSymmTensor<FL>::c)
        .def_static("d", &GeneralSymmTensor<FL>::d)
        .def_static("t", &GeneralSymmTensor<FL>::t)
        .def_static("c_angular", &GeneralSymmTensor<FL>::c_angular)
        .def_static("d_angular", &GeneralSymmTensor<FL>::d_angular)
        .def_static("permutation_parity",
                    &GeneralSymmTensor<FL>::permutation_parity)
        .def_static("find_pattern_perm",
                    &GeneralSymmTensor<FL>::find_pattern_perm)
        .def_static("auto_sort_string",
                    &GeneralSymmTensor<FL>::auto_sort_string)
        .def_static("mul", &GeneralSymmTensor<FL>::mul)
        .def("simplify", &GeneralSymmTensor<FL>::simplify)
        .def("auto_sort", &GeneralSymmTensor<FL>::auto_sort)
        .def("normal_sort", &GeneralSymmTensor<FL>::normal_sort)
        .def("get_cds", &GeneralSymmTensor<FL>::get_cds)
        .def("equal_to_scaled", &GeneralSymmTensor<FL>::equal_to_scaled)
        .def("to_str", &GeneralSymmTensor<FL>::to_str)
        .def("__repr__", &GeneralSymmTensor<FL>::to_str)
        .def_static("get_level", &GeneralSymmTensor<FL>::get_level)
        .def_static("get_quanta", &GeneralSymmTensor<FL>::get_quanta)
        .def_static("count_cds", &GeneralSymmTensor<FL>::count_cds)
        .def_static("parse_expr_angular",
                    &GeneralSymmTensor<FL>::parse_expr_angular, py::arg("expr"),
                    py::arg("idxs"), py::arg("cgs"), py::arg("ii") = 0)
        .def_static("parse_expr_angular_expr",
                    &GeneralSymmTensor<FL>::parse_expr_angular_expr,
                    py::arg("expr"), py::arg("idxs"), py::arg("ii") = 0)
        .def(py::self * FL())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self == py::self);

    py::bind_vector<vector<GeneralSymmTensor<FL>>>(
        m, ("VectorGeneralSymmTensor" + name).c_str());
    py::bind_vector<vector<
        pair<vector<GeneralSymmElement>, vector<pair<vector<int>, FL>>>>>(
        m, ("VectorPVectorGeneralSymmElementVectorPVectorInt" + name + "Double")
               .c_str());
    py::bind_map<map<vector<int>, GeneralSymmTensor<FL>>>(
        m, ("MapVectorIntGeneralSymmTensor" + name).c_str());

    py::class_<GeneralSymmExpr<FL>, shared_ptr<GeneralSymmExpr<FL>>>(
        m, ("GeneralSymmExpr" + name).c_str())
        .def(py::init<const vector<string> &, const string &>())
        .def_readwrite("n_sites", &GeneralSymmExpr<FL>::n_sites)
        .def_readwrite("n_ops", &GeneralSymmExpr<FL>::n_ops)
        .def_readwrite("expr", &GeneralSymmExpr<FL>::expr)
        .def_readwrite("n_reduced_sites", &GeneralSymmExpr<FL>::n_reduced_sites)
        .def_readwrite("max_l", &GeneralSymmExpr<FL>::max_l)
        .def_readwrite("orb_sym", &GeneralSymmExpr<FL>::orb_sym)
        .def_readwrite("site_sym", &GeneralSymmExpr<FL>::site_sym)
        .def_readwrite("data", &GeneralSymmExpr<FL>::data)
        .def_static("orb_names", &GeneralSymmExpr<FL>::orb_names)
        .def(
            "reduce",
            [](GeneralSymmExpr<FL> *self, py::array_t<FL> data,
               typename GeneralSymmExpr<FL>::FP cutoff) -> py::array_t<FL> {
                vector<ssize_t> p(self->data.size() == 0
                                      ? 0
                                      : self->data[0].second[0].first.size(),
                                  (ssize_t)(self->max_l + self->max_l + 1));
                for (const auto &c : self->expr)
                    if (c >= 'A' && c <= 'Z')
                        p.push_back((ssize_t)self->n_reduced_sites);
                py::array_t<FL> reduced_data(p);
                self->reduce(data.data(), reduced_data.mutable_data(), cutoff);
                return reduced_data;
            },
            py::arg("data"),
            py::arg("cutoff") = (typename GeneralSymmExpr<FL>::FP)1E-12)
        .def(
            "reduce_expr",
            [](GeneralSymmExpr<FL> *self, py::array_t<FL> reduced_data,
               typename GeneralSymmExpr<FL>::FP cutoff)
                -> shared_ptr<GeneralFCIDUMP<FL>> {
                shared_ptr<GeneralFCIDUMP<FL>> r =
                    make_shared<GeneralFCIDUMP<FL>>();
                r->elem_type = ElemOpTypes::SAny;
                self->reduce_expr(reduced_data.data(), r->exprs, r->indices,
                                  r->data, cutoff);
                return r;
            },
            py::arg("reduced_data"),
            py::arg("cutoff") = (typename GeneralSymmExpr<FL>::FP)1E-12)
        .def("to_str", &GeneralSymmExpr<FL>::to_str)
        .def("__repr__", &GeneralSymmExpr<FL>::to_str);

    py::class_<GeneralSymmRecoupling<FL>,
               shared_ptr<GeneralSymmRecoupling<FL>>>(
        m, ("GeneralSymmRecoupling" + name).c_str())
        .def_static("recouple", &GeneralSymmRecoupling<FL>::recouple)
        .def_static("exchange", &GeneralSymmRecoupling<FL>::exchange)
        .def_static("sort_indices", &GeneralSymmRecoupling<FL>::sort_indices)
        .def_static("recouple_split",
                    &GeneralSymmRecoupling<FL>::recouple_split);

    py::class_<GeneralSymmPermScheme<FL>,
               shared_ptr<GeneralSymmPermScheme<FL>>>(m,
                                                      "GeneralSymmPermScheme")
        .def(py::init<>())
        .def(py::init<string, const vector<shared_ptr<AnyCG<FL>>> &>())
        .def(py::init<string, const vector<shared_ptr<AnyCG<FL>>> &, bool>())
        .def(py::init<string, const vector<shared_ptr<AnyCG<FL>>> &, bool,
                      bool>())
        .def(py::init<string, const vector<shared_ptr<AnyCG<FL>>> &, bool, bool,
                      const vector<uint16_t> &>())
        .def_readwrite("index_patterns",
                       &GeneralSymmPermScheme<FL>::index_patterns)
        .def_readwrite("data", &GeneralSymmPermScheme<FL>::data)
        .def_readwrite("mask", &GeneralSymmPermScheme<FL>::mask)
        .def_readwrite("targets", &GeneralSymmPermScheme<FL>::targets)
        .def_readwrite("cgs", &GeneralSymmPermScheme<FL>::cgs)
        .def_static("initialize", &GeneralSymmPermScheme<FL>::initialize,
                    py::arg("nn"), py::arg("spin_str"), py::arg("cgs"),
                    py::arg("is_npdm") = true, py::arg("is_drt") = false,
                    py::arg("mask") = vector<uint16_t>())
        .def("to_str", &GeneralSymmPermScheme<FL>::to_str);

    py::bind_vector<vector<shared_ptr<GeneralSymmPermScheme<FL>>>>(
        m, "VectorGeneralSymmPermScheme");
}

template <typename FL> void bind_fl_io(py::module &m, const string &name) {

    m.def(
        name == "Float" ? "init_memory_float" : "init_memory",
        [](size_t isize, size_t dsize, const string &save_dir,
           double dmain_ratio, double imain_ratio, int n_frames) {
            frame_<FL>() = make_shared<DataFrame<FL>>(
                isize, dsize, save_dir, dmain_ratio, imain_ratio, n_frames);
        },
        py::arg("isize") = size_t(1LL << 28),
        py::arg("dsize") = size_t(1LL << 30), py::arg("save_dir") = "nodex",
        py::arg("dmain_ratio") = 0.7, py::arg("imain_ratio") = 0.7,
        py::arg("n_frames") = 2);

    m.def(name == "Float" ? "release_memory_float" : "release_memory", []() {
        frame_<FL>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FL>()->used == 0);
        frame_<FL>() = nullptr;
    });

    py::class_<Allocator<FL>, shared_ptr<Allocator<FL>>>(
        m, (name + "Allocator").c_str())
        .def(py::init<>());

    py::class_<VectorAllocator<FL>, shared_ptr<VectorAllocator<FL>>,
               Allocator<FL>>(m, (name + "VectorAllocator").c_str())
        .def_readwrite("data", &VectorAllocator<FL>::data)
        .def(py::init<>());

    py::class_<StackAllocator<FL>, shared_ptr<StackAllocator<FL>>,
               Allocator<FL>>(m, (name + "StackAllocator").c_str())
        .def(py::init<>())
        .def_readwrite("size", &StackAllocator<FL>::size)
        .def_readwrite("used", &StackAllocator<FL>::used)
        .def_readwrite("shift", &StackAllocator<FL>::shift);

    py::class_<TemporaryAllocator<FL>, shared_ptr<TemporaryAllocator<FL>>,
               StackAllocator<FL>>(m, (name + "TemporaryAllocator").c_str())
        .def(py::init<>());

    py::class_<FPCodec<FL>, shared_ptr<FPCodec<FL>>>(m,
                                                     (name + "FPCodec").c_str())
        .def(py::init<>())
        .def(py::init<FL>())
        .def(py::init<FL, size_t>())
        .def_readwrite("ndata", &FPCodec<FL>::ndata)
        .def_readwrite("ncpsd", &FPCodec<FL>::ncpsd)
        .def("encode",
             [](FPCodec<FL> *self, py::array_t<FL> arr) {
                 FL *tmp = new FL[arr.size() + 2];
                 size_t len = self->encode(arr.mutable_data(), arr.size(), tmp);
                 assert(len <= arr.size() + 2);
                 py::array_t<FL> arx = py::array_t<FL>(len + 1);
                 arx.mutable_data()[0] = (FL)arr.size();
                 memcpy(arx.mutable_data() + 1, tmp, len * sizeof(FL));
                 delete[] tmp;
                 return arx;
             })
        .def("decode",
             [](FPCodec<FL> *self, py::array_t<FL> arr) {
                 size_t arr_len = (size_t)arr.mutable_data()[0];
                 py::array_t<FL> arx = py::array_t<FL>(arr_len);
                 size_t len = self->decode(arr.mutable_data() + 1, arr_len,
                                           arx.mutable_data());
                 assert(len == arr.size() - 1);
                 return arx;
             })
        .def("write_array",
             [](FPCodec<FL> *self, py::array_t<FL> arr) {
                 stringstream ss;
                 self->write_array(ss, arr.mutable_data(), arr.size());
                 assert(ss.tellp() % sizeof(FL) == 0);
                 size_t len = ss.tellp() / sizeof(FL);
                 py::array_t<FL> arx = py::array_t<FL>(len + 1);
                 arx.mutable_data()[0] = (FL)arr.size();
                 ss.clear();
                 ss.seekg(0);
                 ss.read((char *)(arx.mutable_data() + 1), sizeof(FL) * len);
                 return arx;
             })
        .def("read_array",
             [](FPCodec<FL> *self, py::array_t<FL> arr) {
                 size_t arr_len = (size_t)arr.mutable_data()[0];
                 stringstream ss;
                 ss.write((char *)(arr.mutable_data() + 1),
                          (arr.size() - 1) * sizeof(FL));
                 py::array_t<FL> arx = py::array_t<FL>(arr_len);
                 ss.clear();
                 ss.seekg(0);
                 self->read_array(ss, arx.mutable_data(), arr_len);
                 return arx;
             })
        .def(
            "save",
            [](FPCodec<FL> *self, const string &filename, py::array_t<FL> arr) {
                ofstream ofs(filename.c_str(), ios::binary);
                if (!ofs.good())
                    throw runtime_error("FPCodec::save on '" + filename +
                                        "' failed.");
                ofs << arr.size();
                self->write_array(ofs, arr.mutable_data(), arr.size());
                if (!ofs.good())
                    throw runtime_error("FPCodec::save on '" + filename +
                                        "' failed.");
                ofs.close();
            })
        .def("load", [](FPCodec<FL> *self, const string &filename) {
            ifstream ifs(filename.c_str(), ios::binary);
            if (!ifs.good())
                throw runtime_error("FPCodec::load on '" + filename +
                                    "' failed.");
            size_t arr_len;
            ifs >> arr_len;
            py::array_t<FL> arx = py::array_t<FL>(arr_len);
            self->read_array(ifs, arx.mutable_data(), arr_len);
            if (ifs.fail() || ifs.bad())
                throw runtime_error("FPCodec::load on '" + filename +
                                    "' failed.");
            ifs.close();
            return arx;
        });

    py::class_<DataFrame<FL>, shared_ptr<DataFrame<FL>>>(
        m, (name + "DataFrame").c_str())
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, const string &>())
        .def(py::init<size_t, size_t, const string &, FL>())
        .def(py::init<size_t, size_t, const string &, FL, FL>())
        .def(py::init<size_t, size_t, const string &, FL, FL, int>())
        .def_readwrite("save_dir", &DataFrame<FL>::save_dir)
        .def_readwrite("mps_dir", &DataFrame<FL>::mps_dir)
        .def_readwrite("mpo_dir", &DataFrame<FL>::mpo_dir)
        .def_readwrite("restart_dir", &DataFrame<FL>::restart_dir)
        .def_readwrite("restart_dir_per_sweep",
                       &DataFrame<FL>::restart_dir_per_sweep)
        .def_readwrite("restart_dir_optimal_mps",
                       &DataFrame<FL>::restart_dir_optimal_mps)
        .def_readwrite("restart_dir_optimal_mps_per_sweep",
                       &DataFrame<FL>::restart_dir_optimal_mps_per_sweep)
        .def_readwrite("prefix", &DataFrame<FL>::prefix)
        .def_readwrite("prefix_distri", &DataFrame<FL>::prefix_distri)
        .def_readwrite("prefix_can_write", &DataFrame<FL>::prefix_can_write)
        .def_readwrite("partition_can_write",
                       &DataFrame<FL>::partition_can_write)
        .def_readwrite("isize", &DataFrame<FL>::isize)
        .def_readwrite("dsize", &DataFrame<FL>::dsize)
        .def_readwrite("tread", &DataFrame<FL>::tread)
        .def_readwrite("twrite", &DataFrame<FL>::twrite)
        .def_readwrite("tasync", &DataFrame<FL>::tasync)
        .def_readwrite("fpread", &DataFrame<FL>::fpread)
        .def_readwrite("fpwrite", &DataFrame<FL>::fpwrite)
        .def_readwrite("n_frames", &DataFrame<FL>::n_frames)
        .def_readwrite("i_frame", &DataFrame<FL>::i_frame)
        .def_readwrite("iallocs", &DataFrame<FL>::iallocs)
        .def_readwrite("dallocs", &DataFrame<FL>::dallocs)
        .def_readwrite("peak_used_memory", &DataFrame<FL>::peak_used_memory)
        .def_readwrite("load_buffering", &DataFrame<FL>::load_buffering)
        .def_readwrite("save_buffering", &DataFrame<FL>::save_buffering)
        .def_readwrite("use_main_stack", &DataFrame<FL>::use_main_stack)
        .def_readwrite("minimal_disk_usage", &DataFrame<FL>::minimal_disk_usage)
        .def_readwrite("minimal_memory_usage",
                       &DataFrame<FL>::minimal_memory_usage)
        .def_readwrite("compressed_sparse_tensor_storage",
                       &DataFrame<FL>::compressed_sparse_tensor_storage)
        .def_readwrite("fp_codec", &DataFrame<FL>::fp_codec)
        .def("update_peak_used_memory", &DataFrame<FL>::update_peak_used_memory)
        .def("reset_peak_used_memory", &DataFrame<FL>::reset_peak_used_memory)
        .def("activate", &DataFrame<FL>::activate)
        .def("load_data", &DataFrame<FL>::load_data)
        .def("save_data", &DataFrame<FL>::save_data)
        .def("reset", &DataFrame<FL>::reset)
        .def("__repr__", [](DataFrame<FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<StackAllocator<FL>>>>(
        m, ("Vector" + name + "StackAllocator").c_str());
}

template <typename FL> void bind_matrix(py::module &m) {
    py::class_<GMatrix<FL>, shared_ptr<GMatrix<FL>>>(m, "Matrix",
                                                     py::buffer_protocol())
        .def(py::init([](py::array_t<FL> mat) {
                 assert(mat.ndim() == 2);
                 assert(mat.strides()[1] == sizeof(FL));
                 return GMatrix<FL>(mat.mutable_data(), (MKL_INT)mat.shape()[0],
                                    (MKL_INT)mat.shape()[1]);
             }),
             py::keep_alive<0, 1>())
        .def_buffer([](GMatrix<FL> *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(FL),
                                   py::format_descriptor<FL>::format(), 2,
                                   {(ssize_t)self->m, (ssize_t)self->n},
                                   {sizeof(FL) * (ssize_t)self->n, sizeof(FL)});
        })
        .def_readwrite("m", &GMatrix<FL>::m)
        .def_readwrite("n", &GMatrix<FL>::n)
        .def("__repr__",
             [](GMatrix<FL> *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("allocate", &GMatrix<FL>::allocate, py::arg("alloc") = nullptr)
        .def("deallocate", &GMatrix<FL>::deallocate,
             py::arg("alloc") = nullptr);

    py::class_<GMatrix<complex<FL>>, shared_ptr<GMatrix<complex<FL>>>>(
        m, "ComplexMatrix", py::buffer_protocol())
        .def(py::init([](py::array_t<complex<FL>> mat) {
                 assert(mat.ndim() == 2);
                 assert(mat.strides()[1] == sizeof(complex<FL>));
                 return GMatrix<complex<FL>>(mat.mutable_data(),
                                             (MKL_INT)mat.shape()[0],
                                             (MKL_INT)mat.shape()[1]);
             }),
             py::keep_alive<0, 1>())
        .def_buffer([](GMatrix<complex<FL>> *self) -> py::buffer_info {
            return py::buffer_info(
                self->data, sizeof(complex<FL>),
                py::format_descriptor<complex<FL>>::format(), 2,
                {(ssize_t)self->m, (ssize_t)self->n},
                {sizeof(complex<FL>) * (ssize_t)self->n, sizeof(complex<FL>)});
        })
        .def_readwrite("m", &GMatrix<complex<FL>>::m)
        .def_readwrite("n", &GMatrix<complex<FL>>::n)
        .def("__repr__",
             [](GMatrix<complex<FL>> *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("allocate", &GMatrix<complex<FL>>::allocate,
             py::arg("alloc") = nullptr)
        .def("deallocate", &GMatrix<complex<FL>>::deallocate,
             py::arg("alloc") = nullptr);

    py::class_<GMatrixFunctions<FL>>(m, "MatrixFunctions")
        .def_static(
            "elementwise",
            [](const string &f, FL alpha, py::array_t<FL> &a, FL beta,
               py::array_t<FL> &b, py::array_t<FL> &c, FL cfactor) {
                MKL_INT n = (MKL_INT)a.size();
                GMatrixFunctions<FL>::elementwise(
                    f, alpha,
                    GMatrix<FL>(a.mutable_data(), (MKL_INT)a.size(), 1), beta,
                    GMatrix<FL>(b.mutable_data(), (MKL_INT)b.size(), 1),
                    GMatrix<FL>(c.mutable_data(), (MKL_INT)c.size(), 1),
                    cfactor);
            },
            py::arg("f"), py::arg("alpha"), py::arg("a"), py::arg("beta"),
            py::arg("b"), py::arg("c"), py::arg("cfactor") = (FL)1.0)
        .def_static("det",
                    [](py::array_t<FL> &a) {
                        MKL_INT n = (MKL_INT)Prime::sqrt((Prime::LL)a.size());
                        assert(n * n == (MKL_INT)a.size());
                        return GMatrixFunctions<FL>::det(
                            GMatrix<FL>(a.mutable_data(), n, n));
                    })
        .def_static("eigs",
                    [](py::array_t<FL> &a, py::array_t<FL> &w) {
                        MKL_INT n = (MKL_INT)w.size();
                        GMatrixFunctions<FL>::eigs(
                            GMatrix<FL>(a.mutable_data(), n, n),
                            GDiagonalMatrix<FL>(w.mutable_data(), n));
                    })
        .def_static(
            "geigs",
            [](py::array_t<FL> &a, py::array_t<FL> &b, py::array_t<FL> &w) {
                MKL_INT n = (MKL_INT)w.size();
                GMatrixFunctions<FL>::geigs(
                    GMatrix<FL>(a.mutable_data(), n, n),
                    GMatrix<FL>(b.mutable_data(), n, n),
                    GDiagonalMatrix<FL>(w.mutable_data(), n));
            })
        .def_static(
            "accurate_geigs",
            [](py::array_t<FL> &a, py::array_t<FL> &b, py::array_t<FL> &w,
               FL eps) {
                MKL_INT n = (MKL_INT)w.size();
                return GMatrixFunctions<FL>::accurate_geigs(
                    GMatrix<FL>(a.mutable_data(), n, n),
                    GMatrix<FL>(b.mutable_data(), n, n),
                    GDiagonalMatrix<FL>(w.mutable_data(), n), eps);
            },
            py::arg("a"), py::arg("b"), py::arg("w"),
            py::arg("eps") = (FL)1E-12)
        .def_static("block_eigs", [](py::array_t<FL> &a, py::array_t<FL> &w,
                                     const vector<uint8_t> &x) {
            MKL_INT n = (MKL_INT)w.size();
            GMatrixFunctions<FL>::block_eigs(
                GMatrix<FL>(a.mutable_data(), n, n),
                GDiagonalMatrix<FL>(w.mutable_data(), n), x);
        });

    py::class_<GMatrixFunctions<complex<FL>>>(m, "ComplexMatrixFunctions")
        .def_static(
            "elementwise",
            [](const string &f, complex<FL> alpha, py::array_t<complex<FL>> &a,
               complex<FL> beta, py::array_t<complex<FL>> &b,
               py::array_t<complex<FL>> &c, complex<FL> cfactor) {
                MKL_INT n = (MKL_INT)a.size();
                GMatrixFunctions<complex<FL>>::elementwise(
                    f, alpha,
                    GMatrix<complex<FL>>(a.mutable_data(), (MKL_INT)a.size(),
                                         1),
                    beta,
                    GMatrix<complex<FL>>(b.mutable_data(), (MKL_INT)b.size(),
                                         1),
                    GMatrix<complex<FL>>(c.mutable_data(), (MKL_INT)c.size(),
                                         1),
                    cfactor);
            },
            py::arg("f"), py::arg("alpha"), py::arg("a"), py::arg("beta"),
            py::arg("b"), py::arg("c"), py::arg("cfactor") = (complex<FL>)1.0)
        .def_static("eigs",
                    [](py::array_t<complex<FL>> &a, py::array_t<FL> &w) {
                        MKL_INT n = (MKL_INT)w.size();
                        GMatrixFunctions<complex<FL>>::eigs(
                            GMatrix<complex<FL>>(a.mutable_data(), n, n),
                            GDiagonalMatrix<FL>(w.mutable_data(), n));
                    })
        .def_static("geigs",
                    [](py::array_t<complex<FL>> &a, py::array_t<complex<FL>> &b,
                       py::array_t<FL> &w) {
                        MKL_INT n = (MKL_INT)w.size();
                        GMatrixFunctions<complex<FL>>::geigs(
                            GMatrix<complex<FL>>(a.mutable_data(), n, n),
                            GMatrix<complex<FL>>(b.mutable_data(), n, n),
                            GDiagonalMatrix<FL>(w.mutable_data(), n));
                    })
        .def_static(
            "accurate_geigs",
            [](py::array_t<complex<FL>> &a, py::array_t<complex<FL>> &b,
               py::array_t<FL> &w, FL eps) {
                MKL_INT n = (MKL_INT)w.size();
                return GMatrixFunctions<complex<FL>>::accurate_geigs(
                    GMatrix<complex<FL>>(a.mutable_data(), n, n),
                    GMatrix<complex<FL>>(b.mutable_data(), n, n),
                    GDiagonalMatrix<FL>(w.mutable_data(), n), eps);
            },
            py::arg("a"), py::arg("b"), py::arg("w"),
            py::arg("eps") = (FL)1E-12);

    py::class_<IterativeMatrixFunctions<FL>>(m, "IterativeMatrixFunctions")
        .def_static(
            "davidson",
            [](py::object op, py::array_t<FL> &diag, py::list kets, bool iprint,
               typename GMatrix<FL>::FP conv_thrd,
               typename GMatrix<FL>::FP rel_conv_thrd, int max_iter,
               int soft_max_iter, int deflation_min_size,
               int deflation_max_size) {
                auto f = [&op](const GMatrix<FL> &b, const GMatrix<FL> &c) {
                    py::array_t<FL> x =
                        (py::array_t<FL>)op(py::array_t<FL>(b.size(), b.data));
                    GMatrixFunctions<FL>::iadd(
                        c, GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        (FL)1.0);
                };
                vector<GMatrix<FL>> vs;
                for (size_t i = 0; i < kets.size(); i++) {
                    py::array_t<FL> ket = kets[i].cast<py::array_t<FL>>();
                    vs.push_back(GMatrix<FL>(ket.mutable_data(),
                                             (MKL_INT)ket.size(), 1));
                }
                int ndav = 0;
                return std::make_pair(
                    IterativeMatrixFunctions<FL>::davidson(
                        f,
                        GDiagonalMatrix<FL>(diag.mutable_data(),
                                            (MKL_INT)diag.size()),
                        vs, (typename GMatrix<FL>::FP)0.0,
                        DavidsonTypes::Normal, ndav, iprint,
                        (shared_ptr<ParallelCommunicator<SZ>>)nullptr,
                        conv_thrd, rel_conv_thrd, max_iter, soft_max_iter,
                        deflation_min_size, deflation_max_size),
                    ndav);
            },
            py::arg("op"), py::arg("diag"), py::arg("kets"),
            py::arg("iprint") = false, py::arg("conv_thrd") = 5E-6,
            py::arg("rel_conv_thrd") = 0.0, py::arg("max_iter") = 5000,
            py::arg("soft_max_iter") = -1, py::arg("deflation_min_size") = 2,
            py::arg("deflation_max_size") = 50)
        .def_static(
            "davidson_generalized",
            [](py::object op, py::object sop, py::array_t<FL> &diag,
               py::list kets, bool iprint, typename GMatrix<FL>::FP conv_thrd,
               typename GMatrix<FL>::FP rel_conv_thrd, int max_iter,
               int soft_max_iter, int deflation_min_size,
               int deflation_max_size) {
                const function<void(const GMatrix<FL> &, const GMatrix<FL> &)>
                    &f = [&op](const GMatrix<FL> &b, const GMatrix<FL> &c) {
                        py::array_t<FL> x = (py::array_t<FL>)op(
                            py::array_t<FL>(b.size(), b.data));
                        GMatrixFunctions<FL>::iadd(
                            c,
                            GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                            (FL)1.0);
                    };
                const function<void(const GMatrix<FL> &, const GMatrix<FL> &)>
                    &g = [&sop](const GMatrix<FL> &b, const GMatrix<FL> &c) {
                        py::array_t<FL> x = (py::array_t<FL>)sop(
                            py::array_t<FL>(b.size(), b.data));
                        GMatrixFunctions<FL>::iadd(
                            c,
                            GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                            (FL)1.0);
                    };
                vector<GMatrix<FL>> vs;
                for (size_t i = 0; i < kets.size(); i++) {
                    py::array_t<FL> ket = kets[i].cast<py::array_t<FL>>();
                    vs.push_back(GMatrix<FL>(ket.mutable_data(),
                                             (MKL_INT)ket.size(), 1));
                }
                int ndav = 0;
                return std::make_pair(
                    IterativeMatrixFunctions<FL>::davidson_generalized(
                        f, g,
                        GDiagonalMatrix<FL>(diag.mutable_data(),
                                            (MKL_INT)diag.size()),
                        vs, (typename GMatrix<FL>::FP)0.0,
                        DavidsonTypes::Normal, ndav, iprint,
                        (shared_ptr<ParallelCommunicator<SZ>>)nullptr,
                        conv_thrd, rel_conv_thrd, max_iter, soft_max_iter,
                        deflation_min_size, deflation_max_size),
                    ndav);
            },
            py::arg("op"), py::arg("sop"), py::arg("diag"), py::arg("kets"),
            py::arg("iprint") = false, py::arg("conv_thrd") = 5E-6,
            py::arg("rel_conv_thrd") = 0.0, py::arg("max_iter") = 5000,
            py::arg("soft_max_iter") = -1, py::arg("deflation_min_size") = 2,
            py::arg("deflation_max_size") = 50)
        .def_static(
            "conjugate_gradient",
            [](py::object op, py::array_t<FL> &diag, py::array_t<FL> &x,
               py::array_t<FL> &b, FL consta, bool iprint,
               typename GMatrix<FL>::FP conv_thrd, int max_iter,
               int soft_max_iter) {
                auto f = [&op](const GMatrix<FL> &b, const GMatrix<FL> &c) {
                    py::array_t<FL> x =
                        (py::array_t<FL>)op(py::array_t<FL>(b.size(), b.data));
                    GMatrixFunctions<FL>::iadd(
                        c, GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        (FL)1.0);
                };
                int nmult = 0;
                return std::make_pair(
                    IterativeMatrixFunctions<FL>::conjugate_gradient(
                        f,
                        GDiagonalMatrix<FL>(diag.mutable_data(),
                                            (MKL_INT)diag.size()),
                        GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        GMatrix<FL>(b.mutable_data(), (MKL_INT)b.size(), 1),
                        nmult, consta, iprint,
                        (shared_ptr<ParallelCommunicator<SZ>>)nullptr,
                        conv_thrd, max_iter, soft_max_iter),
                    nmult);
            },
            py::arg("op"), py::arg("diag"), py::arg("x"), py::arg("b"),
            py::arg("consta") = (FL)0.0, py::arg("iprint") = false,
            py::arg("conv_thrd") = 5E-6, py::arg("max_iter") = 5000,
            py::arg("soft_max_iter") = -1)
        .def_static(
            "minres",
            [](py::object op, py::array_t<FL> &x, py::array_t<FL> &b, FL consta,
               bool iprint, typename GMatrix<FL>::FP conv_thrd, int max_iter,
               int soft_max_iter) {
                auto f = [&op](const GMatrix<FL> &b, const GMatrix<FL> &c) {
                    py::array_t<FL> x =
                        (py::array_t<FL>)op(py::array_t<FL>(b.size(), b.data));
                    GMatrixFunctions<FL>::iadd(
                        c, GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        (FL)1.0);
                };
                int nmult = 0;
                return std::make_pair(
                    IterativeMatrixFunctions<FL>::minres(
                        f, GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        GMatrix<FL>(b.mutable_data(), (MKL_INT)b.size(), 1),
                        nmult, consta, iprint,
                        (shared_ptr<ParallelCommunicator<SZ>>)nullptr,
                        conv_thrd, max_iter, soft_max_iter),
                    nmult);
            },
            py::arg("op"), py::arg("x"), py::arg("b"),
            py::arg("consta") = (FL)0.0, py::arg("iprint") = false,
            py::arg("conv_thrd") = 5E-6, py::arg("max_iter") = 5000,
            py::arg("soft_max_iter") = -1)
        .def_static(
            "gcrotmk",
            [](py::object op, py::array_t<FL> &diag, py::array_t<FL> &x,
               py::array_t<FL> &b, int m, int k, FL consta, bool iprint,
               typename GMatrix<FL>::FP conv_thrd, int max_iter,
               int soft_max_iter) {
                auto f = [&op](const GMatrix<FL> &b, const GMatrix<FL> &c) {
                    py::array_t<FL> x =
                        (py::array_t<FL>)op(py::array_t<FL>(b.size(), b.data));
                    GMatrixFunctions<FL>::iadd(
                        c, GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        (FL)1.0);
                };
                int nmult = 0, niter = 0;
                return std::make_pair(
                    IterativeMatrixFunctions<FL>::gcrotmk(
                        f,
                        GDiagonalMatrix<FL>(diag.mutable_data(),
                                            (MKL_INT)diag.size()),
                        GMatrix<FL>(x.mutable_data(), (MKL_INT)x.size(), 1),
                        GMatrix<FL>(b.mutable_data(), (MKL_INT)b.size(), 1),
                        nmult, niter, m, k, consta, iprint,
                        (shared_ptr<ParallelCommunicator<SZ>>)nullptr,
                        conv_thrd, max_iter, soft_max_iter),
                    niter);
            },
            py::arg("op"), py::arg("diag"), py::arg("x"), py::arg("b"),
            py::arg("m") = 20, py::arg("k") = -1, py::arg("consta") = (FL)0.0,
            py::arg("iprint") = false, py::arg("conv_thrd") = 5E-6,
            py::arg("max_iter") = 5000, py::arg("soft_max_iter") = -1)
        .def_static(
            "constrained_svd",
            [](py::array_t<FL> &x, MKL_INT rank, FL au, FL av, int max_iter_pi,
               int max_iter_pocs, FL eps_pi, FL eps_pocs, bool iprint)
                -> tuple<py::array_t<FL>, py::array_t<FL>, py::array_t<FL>> {
                GMatrix<FL> xx(x.mutable_data(), (MKL_INT)x.shape()[0],
                               (MKL_INT)x.shape()[1]);
                py::array_t<FL> l(vector<ssize_t>{x.shape()[0], (ssize_t)rank});
                py::array_t<FL> s(vector<ssize_t>{(ssize_t)rank});
                py::array_t<FL> r(vector<ssize_t>{(ssize_t)rank, x.shape()[1]});
                GMatrix<FL> xl(l.mutable_data(), (MKL_INT)l.shape()[0],
                               (MKL_INT)l.shape()[1]);
                GMatrix<FL> xs(s.mutable_data(), 1, (MKL_INT)s.shape()[0]);
                GMatrix<FL> xr(r.mutable_data(), (MKL_INT)r.shape()[0],
                               (MKL_INT)r.shape()[1]);
                IterativeMatrixFunctions<FL>::constrained_svd(
                    xx, rank, xl, xs, xr, au, av, max_iter_pi, max_iter_pocs,
                    eps_pi, eps_pocs, iprint);
                return std::make_tuple(l, s, r);
            },
            py::arg("x"), py::arg("rank"), py::arg("au") = (FL)0.0,
            py::arg("av") = (FL)0.0, py::arg("max_iter_pi") = 1000,
            py::arg("max_iter_pocs") = 1000, py::arg("eps_pi") = (FL)1E-10,
            py::arg("eps_pocs") = (FL)1E-10, py::arg("iprint") = false)
        .def_static(
            "disjoint_svd",
            [](py::array_t<FL> &x, py::array_t<FL> &levels)
                -> tuple<py::array_t<FL>, py::array_t<FL>, py::array_t<FL>> {
                GMatrix<FL> xx(x.mutable_data(), (MKL_INT)x.shape()[0],
                               (MKL_INT)x.shape()[1]);
                ssize_t k = min(x.shape()[0], x.shape()[1]);
                py::array_t<FL> l(vector<ssize_t>{x.shape()[0], k});
                py::array_t<FL> s(vector<ssize_t>{k});
                py::array_t<FL> r(vector<ssize_t>{k, x.shape()[1]});
                GMatrix<FL> xl(l.mutable_data(), (MKL_INT)l.shape()[0],
                               (MKL_INT)l.shape()[1]);
                GMatrix<FL> xs(s.mutable_data(), 1, (MKL_INT)s.shape()[0]);
                GMatrix<FL> xr(r.mutable_data(), (MKL_INT)r.shape()[0],
                               (MKL_INT)r.shape()[1]);
                vector<FL> xlevels(levels.data(),
                                   levels.data() + levels.size());
                IterativeMatrixFunctions<FL>::disjoint_svd(xx, xl, xs, xr,
                                                           xlevels);
                return std::make_tuple(l, s, r);
            },
            py::arg("x"), py::arg("levels") = py::array_t<FL>(0));

    py::class_<IterativeMatrixFunctions<complex<FL>>>(
        m, "ComplexIterativeMatrixFunctions")
        .def_static(
            "constrained_svd",
            [](py::array_t<complex<FL>> &x, MKL_INT rank, FL au, FL av,
               int max_iter_pi, int max_iter_pocs, FL eps_pi, FL eps_pocs,
               bool iprint) -> tuple<py::array_t<complex<FL>>, py::array_t<FL>,
                                     py::array_t<complex<FL>>> {
                GMatrix<complex<FL>> xx(x.mutable_data(), (MKL_INT)x.shape()[0],
                                        (MKL_INT)x.shape()[1]);
                py::array_t<complex<FL>> l(
                    vector<ssize_t>{x.shape()[0], (ssize_t)rank});
                py::array_t<FL> s(vector<ssize_t>{(ssize_t)rank});
                py::array_t<complex<FL>> r(
                    vector<ssize_t>{(ssize_t)rank, x.shape()[1]});
                GMatrix<complex<FL>> xl(l.mutable_data(), (MKL_INT)l.shape()[0],
                                        (MKL_INT)l.shape()[1]);
                GMatrix<FL> xs(s.mutable_data(), 1, (MKL_INT)s.shape()[0]);
                GMatrix<complex<FL>> xr(r.mutable_data(), (MKL_INT)r.shape()[0],
                                        (MKL_INT)r.shape()[1]);
                IterativeMatrixFunctions<complex<FL>>::constrained_svd(
                    xx, rank, xl, xs, xr, au, av, max_iter_pi, max_iter_pocs,
                    eps_pi, eps_pocs, iprint);
                return std::make_tuple(l, s, r);
            },
            py::arg("x"), py::arg("rank"), py::arg("au") = (FL)0.0,
            py::arg("av") = (FL)0.0, py::arg("max_iter_pi") = 1000,
            py::arg("max_iter_pocs") = 1000, py::arg("eps_pi") = (FL)1E-10,
            py::arg("eps_pocs") = (FL)1E-10, py::arg("iprint") = false)
        .def_static(
            "disjoint_svd",
            [](py::array_t<complex<FL>> &x, py::array_t<FL> &levels)
                -> tuple<py::array_t<complex<FL>>, py::array_t<FL>,
                         py::array_t<complex<FL>>> {
                GMatrix<complex<FL>> xx(x.mutable_data(), (MKL_INT)x.shape()[0],
                                        (MKL_INT)x.shape()[1]);
                ssize_t k = min(x.shape()[0], x.shape()[1]);
                py::array_t<complex<FL>> l(vector<ssize_t>{x.shape()[0], k});
                py::array_t<FL> s(vector<ssize_t>{k});
                py::array_t<complex<FL>> r(vector<ssize_t>{k, x.shape()[1]});
                GMatrix<complex<FL>> xl(l.mutable_data(), (MKL_INT)l.shape()[0],
                                        (MKL_INT)l.shape()[1]);
                GMatrix<FL> xs(s.mutable_data(), 1, (MKL_INT)s.shape()[0]);
                GMatrix<complex<FL>> xr(r.mutable_data(), (MKL_INT)r.shape()[0],
                                        (MKL_INT)r.shape()[1]);
                vector<FL> xlevels(levels.data(),
                                   levels.data() + levels.size());
                IterativeMatrixFunctions<complex<FL>>::disjoint_svd(
                    xx, xl, xs, xr, xlevels);
                return std::make_tuple(l, s, r);
            },
            py::arg("x"), py::arg("levels") = py::array_t<FL>(0));
}

template <typename S = void> void bind_post_matrix(py::module &m) {

    py::class_<HubbardFCIDUMP, shared_ptr<HubbardFCIDUMP>, FCIDUMP<double>>(
        m, "HubbardFCIDUMP")
        .def(py::init<uint16_t, double, double>())
        .def(py::init<uint16_t, double, double, bool>())
        .def_readwrite("periodic", &HubbardFCIDUMP::periodic)
        .def_readwrite("const_u", &HubbardFCIDUMP::const_u)
        .def_readwrite("const_t", &HubbardFCIDUMP::const_t);

    py::class_<HeisenbergFCIDUMP, shared_ptr<HeisenbergFCIDUMP>,
               FCIDUMP<double>>(m, "HeisenbergFCIDUMP")
        .def(py::init<const shared_ptr<FCIDUMP<double>> &>())
        .def_readwrite("couplings", &HeisenbergFCIDUMP::couplings);

    py::class_<HubbardKSpaceFCIDUMP, shared_ptr<HubbardKSpaceFCIDUMP>,
               FCIDUMP<double>>(m, "HubbardKSpaceFCIDUMP")
        .def(py::init<uint16_t, double, double>())
        .def_readwrite("const_u", &HubbardKSpaceFCIDUMP::const_u)
        .def_readwrite("const_t", &HubbardKSpaceFCIDUMP::const_t);

    py::class_<DyallFCIDUMP, shared_ptr<DyallFCIDUMP>, FCIDUMP<double>>(
        m, "DyallFCIDUMP")
        .def(
            py::init<const shared_ptr<FCIDUMP<double>> &, uint16_t, uint16_t>())
        .def(py::init<const shared_ptr<FCIDUMP<double>> &, uint16_t, uint16_t,
                      bool>())
        .def_readwrite("fcidump", &DyallFCIDUMP::fcidump)
        .def_readwrite("n_inactive", &DyallFCIDUMP::n_inactive)
        .def_readwrite("n_virtual", &DyallFCIDUMP::n_virtual)
        .def_readwrite("n_active", &DyallFCIDUMP::n_active)
        .def_readwrite("merged_external", &DyallFCIDUMP::merged_external)
        .def("initialize_su2",
             [](DyallFCIDUMP *self, const py::array_t<double> &f) {
                 self->initialize_su2(f.data(), f.size());
             })
        .def("initialize_sz",
             [](DyallFCIDUMP *self, const py::array_t<double> &fa,
                const py::array_t<double> &fb) {
                 self->initialize_sz(fa.data(), fa.size(), fb.data(),
                                     fb.size());
             })
        .def("initialize_from_1pdm_su2",
             [](DyallFCIDUMP *self, py::array_t<double> &dm) {
                 assert(dm.ndim() == 2);
                 assert(dm.strides()[1] == sizeof(double));
                 GMatrix<double> mr(dm.mutable_data(), (MKL_INT)dm.shape()[0],
                                    (MKL_INT)dm.shape()[1]);
                 self->initialize_from_1pdm_su2(mr);
             })
        .def("initialize_from_1pdm_sz",
             [](DyallFCIDUMP *self, py::array_t<double> &dm) {
                 assert(dm.ndim() == 2);
                 assert(dm.strides()[1] == sizeof(double));
                 GMatrix<double> mr(dm.mutable_data(), (MKL_INT)dm.shape()[0],
                                    (MKL_INT)dm.shape()[1]);
                 self->initialize_from_1pdm_sz(mr);
             });

    py::class_<FinkFCIDUMP, shared_ptr<FinkFCIDUMP>, FCIDUMP<double>>(
        m, "FinkFCIDUMP")
        .def(
            py::init<const shared_ptr<FCIDUMP<double>> &, uint16_t, uint16_t>())
        .def(py::init<const shared_ptr<FCIDUMP<double>> &, uint16_t, uint16_t,
                      bool>())
        .def_readwrite("fcidump", &FinkFCIDUMP::fcidump)
        .def_readwrite("n_inactive", &FinkFCIDUMP::n_inactive)
        .def_readwrite("n_virtual", &FinkFCIDUMP::n_virtual)
        .def_readwrite("n_active", &FinkFCIDUMP::n_active)
        .def_readwrite("merged_external", &FinkFCIDUMP::merged_external);
}

template <typename FL> void bind_fl_matrix(py::module &m) {

    py::class_<GCSRMatrix<FL>, shared_ptr<GCSRMatrix<FL>>>(m, "CSRMatrix")
        .def(py::init<>())
        .def(py::init<MKL_INT, MKL_INT>())
        .def_readwrite("m", &GCSRMatrix<FL>::m)
        .def_readwrite("n", &GCSRMatrix<FL>::n)
        .def_readwrite("nnz", &GCSRMatrix<FL>::nnz)
        .def_property(
            "data",
            [](GCSRMatrix<FL> *self) {
                return py::array_t<FL>(self->nnz, self->data);
            },
            [](GCSRMatrix<FL> *self, const py::array_t<double> &v) {
                assert(v.size() == self->nnz);
                memcpy(self->data, v.data(), sizeof(double) * self->nnz);
            })
        .def_property(
            "rows",
            [](GCSRMatrix<FL> *self) {
                return py::array_t<MKL_INT>(self->m + 1, self->rows);
            },
            [](GCSRMatrix<FL> *self, const py::array_t<MKL_INT> &v) {
                assert(v.size() == self->m + 1);
                memcpy(self->rows, v.data(), sizeof(MKL_INT) * (self->m + 1));
            })
        .def_property(
            "cols",
            [](GCSRMatrix<FL> *self) {
                return py::array_t<MKL_INT>(self->nnz, self->cols);
            },
            [](GCSRMatrix<FL> *self, const py::array_t<MKL_INT> &v) {
                assert(v.size() == self->nnz);
                memcpy(self->cols, v.data(), sizeof(MKL_INT) * self->nnz);
            })
        .def("__repr__",
             [](GCSRMatrix<FL> *self) {
                 stringstream ss;
                 ss << *self;
                 return ss.str();
             })
        .def("size", &GCSRMatrix<FL>::size)
        .def("memory_size", &GCSRMatrix<FL>::memory_size)
        .def("transpose", &GCSRMatrix<FL>::transpose)
        .def("sparsity", &GCSRMatrix<FL>::sparsity)
        .def("deep_copy", &GCSRMatrix<FL>::deep_copy)
        .def("from_dense", &GCSRMatrix<FL>::from_dense)
        .def("to_dense",
             [](GCSRMatrix<FL> *self, py::array_t<FL> v) {
                 assert(v.size() == self->size());
                 GMatrix<FL> mat(v.mutable_data(), self->m, self->n);
                 self->to_dense(mat);
             })
        .def("diag", &GCSRMatrix<FL>::diag)
        .def("trace", &GCSRMatrix<FL>::trace)
        .def("allocate", &GCSRMatrix<FL>::allocate)
        .def("deallocate", &GCSRMatrix<FL>::deallocate);

    py::bind_vector<vector<shared_ptr<GCSRMatrix<FL>>>>(m, "VectorCSRMatrix");

    py::class_<GTensor<FL>, shared_ptr<GTensor<FL>>>(m, "Tensor",
                                                     py::buffer_protocol())
        .def(py::init<MKL_INT, MKL_INT, MKL_INT>())
        .def(py::init<const vector<MKL_INT> &>())
        .def(py::init<>())
        .def_buffer([](GTensor<FL> *self) -> py::buffer_info {
            vector<ssize_t> shape, strides;
            for (auto x : self->shape)
                shape.push_back(x);
            if (shape.size() != 0) {
                strides.push_back(sizeof(FL));
                for (int i = (int)shape.size() - 1; i > 0; i--)
                    strides.push_back(strides.back() * shape[i]);
                reverse(strides.begin(), strides.end());
            }
            return py::buffer_info(&(*self->data)[0], sizeof(FL),
                                   py::format_descriptor<FL>::format(),
                                   shape.size(), shape, strides);
        })
        .def_readwrite("shape", &GTensor<FL>::shape)
        .def_readwrite("data", &GTensor<FL>::data)
        .def_property_readonly("ref", &GTensor<FL>::ref)
        .def("save",
             [](GTensor<FL> *self, const string &filename) {
                 ofstream ofs(filename.c_str(), ios::binary);
                 if (!ofs.good())
                     throw runtime_error("GTensor::save on '" + filename +
                                         "' failed.");
                 self->write_array(ofs);
                 if (!ofs.good())
                     throw runtime_error("GTensor::save on '" + filename +
                                         "' failed.");
                 ofs.close();
             })
        .def("load",
             [](GTensor<FL> *self, const string &filename) {
                 ifstream ifs(filename.c_str(), ios::binary);
                 if (!ifs.good())
                     throw runtime_error("GTensor::load on '" + filename +
                                         "' failed.");
                 self->read_array(ifs);
                 if (ifs.fail() || ifs.bad())
                     throw runtime_error("GTensor::load on '" + filename +
                                         "' failed.");
                 ifs.close();
                 return self;
             })
        .def("__repr__", [](GTensor<FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });

    py::bind_vector<vector<shared_ptr<GTensor<FL>>>>(m, "VectorTensor");
    py::bind_vector<vector<vector<shared_ptr<GTensor<FL>>>>>(
        m, "VectorVectorTensor");

    py::bind_vector<vector<pair<pair<int, int>, FL>>>(m, "VectorPPIntFL");
    py::bind_vector<vector<pair<pair<long long int, long long int>, FL>>>(
        m, "VectorPPLLIntFL");

    if (sizeof(MKL_INT) == sizeof(int))
        m.attr("VectorPPMKLIntFL") = m.attr("VectorPPIntFL");
    else if (sizeof(MKL_INT) == sizeof(long long int))
        m.attr("VectorPPMKLIntFL") = m.attr("VectorPPLLIntFL");

    py::class_<GCSRMatrixFunctions<FL>>(m, "CSRMatrixFunctions")
        .def_static("copy", &GCSRMatrixFunctions<FL>::copy)
        .def_static("iscale", &GCSRMatrixFunctions<FL>::iscale)
        .def_static("norm", &GCSRMatrixFunctions<FL>::norm)
        .def_static("dot", &GCSRMatrixFunctions<FL>::dot)
        .def_static("sparse_dot", &GCSRMatrixFunctions<FL>::sparse_dot)
        .def_static("iadd", &GCSRMatrixFunctions<FL>::iadd)
        .def_static("multiply", (void (*)(const GCSRMatrix<FL> &, uint8_t,
                                          const GCSRMatrix<FL> &, uint8_t,
                                          GCSRMatrix<FL> &, FL, FL)) &
                                    GCSRMatrixFunctions<FL>::multiply)
        .def_static("multiply", (void (*)(const GMatrix<FL> &, uint8_t,
                                          const GCSRMatrix<FL> &, uint8_t,
                                          const GMatrix<FL> &, FL, FL)) &
                                    GCSRMatrixFunctions<FL>::multiply)
        .def_static("multiply", (void (*)(const GCSRMatrix<FL> &, uint8_t,
                                          const GMatrix<FL> &, uint8_t,
                                          const GMatrix<FL> &, FL, FL)) &
                                    GCSRMatrixFunctions<FL>::multiply);

    py::class_<GDiagonalMatrix<FL>, shared_ptr<GDiagonalMatrix<FL>>>(
        m, "DiagonalMatrix", py::buffer_protocol())
        .def_buffer([](GDiagonalMatrix<FL> *self) -> py::buffer_info {
            return py::buffer_info(self->data, sizeof(FL),
                                   py::format_descriptor<FL>::format(), 1,
                                   {(ssize_t)self->n}, {sizeof(FL)});
        });

    py::class_<FCIDUMP<FL>, shared_ptr<FCIDUMP<FL>>>(m, "FCIDUMP")
        .def(py::init<>())
        .def("read", &FCIDUMP<FL>::read)
        .def("write", &FCIDUMP<FL>::write)
        .def("initialize_h1e",
             [](FCIDUMP<FL> *self, uint16_t n_sites, uint16_t n_elec,
                uint16_t twos, uint16_t isym, FL e, const py::array_t<FL> &t) {
                 self->initialize_h1e(n_sites, n_elec, twos, isym, e, t.data(),
                                      t.size());
             })
        .def("initialize_su2",
             [](FCIDUMP<FL> *self, uint16_t n_sites, uint16_t n_elec,
                uint16_t twos, uint16_t isym, FL e, const py::array_t<FL> &t,
                const py::array_t<FL> &v) {
                 self->initialize_su2(n_sites, n_elec, twos, isym, e, t.data(),
                                      t.size(), v.data(), v.size());
             })
        .def("initialize_sz",
             [](FCIDUMP<FL> *self, uint16_t n_sites, uint16_t n_elec,
                int16_t twos, uint16_t isym, FL e, const py::tuple &t,
                const py::tuple &v) {
                 assert(t.size() == 2 && v.size() == 3);
                 py::array_t<FL> ta = t[0].cast<py::array_t<FL>>();
                 py::array_t<FL> tb = t[1].cast<py::array_t<FL>>();
                 py::array_t<FL> va = v[0].cast<py::array_t<FL>>();
                 py::array_t<FL> vb = v[1].cast<py::array_t<FL>>();
                 py::array_t<FL> vab = v[2].cast<py::array_t<FL>>();
                 self->initialize_sz(n_sites, n_elec, twos, isym, e, ta.data(),
                                     ta.size(), tb.data(), tb.size(), va.data(),
                                     va.size(), vb.data(), vb.size(),
                                     vab.data(), vab.size());
             })
        .def("deallocate", &FCIDUMP<FL>::deallocate)
        .def("truncate_small", &FCIDUMP<FL>::truncate_small)
        .def("symmetrize",
             (typename GMatrix<FL>::FP(FCIDUMP<FL>::*)(const vector<int> &)) &
                 FCIDUMP<FL>::symmetrize,
             py::arg("orbsym"),
             "Remove integral elements that violate point group symmetry. "
             "Returns summed error in symmetrization\n\n"
             "    Args:\n"
             "        orbsym : in XOR convention")
        .def("symmetrize",
             (typename GMatrix<FL>::FP(FCIDUMP<FL>::*)(
                 const vector<uint8_t> &)) &
                 FCIDUMP<FL>::symmetrize,
             py::arg("orbsym"),
             "Remove integral elements that violate point group symmetry. "
             "Returns summed error in symmetrization\n\n"
             "    Args:\n"
             "        orbsym : in XOR convention")
        .def("symmetrize",
             (typename GMatrix<FL>::FP(FCIDUMP<FL>::*)(
                 const vector<int16_t> &)) &
                 FCIDUMP<FL>::symmetrize,
             py::arg("orbsym"),
             "Remove integral elements that violate point group symmetry. "
             "Returns summed error in symmetrization\n\n"
             "    Args:\n"
             "        orbsym : in Lz convention")
        .def("symmetrize",
             (typename GMatrix<FL>::FP(FCIDUMP<FL>::*)(const vector<int> &,
                                                       int)) &
                 FCIDUMP<FL>::symmetrize,
             py::arg("k_sym"), py::arg("k_mod"),
             "Remove integral elements that violate k symmetry. "
             "Returns summed error in symmetrization")
        .def("e", &FCIDUMP<FL>::e)
        .def(
            "t",
            [](FCIDUMP<FL> *self, py::args &args) -> FL {
                assert(args.size() == 2 || args.size() == 3);
                if (args.size() == 2)
                    return self->t((uint16_t)args[0].cast<int>(),
                                   (uint16_t)args[1].cast<int>());
                else
                    return self->t((uint8_t)args[0].cast<int>(),
                                   (uint16_t)args[1].cast<int>(),
                                   (uint16_t)args[2].cast<int>());
            },
            "1. (i: int, j: int) -> float\n"
            "    One-electron integral element (SU(2));\n"
            "2. (s: int, i: int, j: int) -> float\n"
            "    One-electron integral element (SZ).\n\n"
            "    Args:\n"
            "        i, j : spatial indices\n"
            "        s : spin index (0=alpha, 1=beta)")
        .def(
            "v",
            [](FCIDUMP<FL> *self, py::args &args) -> FL {
                assert(args.size() == 4 || args.size() == 6);
                if (args.size() == 4)
                    return self->v((uint16_t)args[0].cast<int>(),
                                   (uint16_t)args[1].cast<int>(),
                                   (uint16_t)args[2].cast<int>(),
                                   (uint16_t)args[3].cast<int>());
                else
                    return self->v((uint8_t)args[0].cast<int>(),
                                   (uint8_t)args[1].cast<int>(),
                                   (uint16_t)args[2].cast<int>(),
                                   (uint16_t)args[3].cast<int>(),
                                   (uint16_t)args[4].cast<int>(),
                                   (uint16_t)args[5].cast<int>());
            },
            "1. (i: int, j: int, k: int, l: int) -> float\n"
            "    Two-electron integral element (SU(2));\n"
            "2. (sij: int, skl: int, i: int, j: int, k: int, l: int) -> float\n"
            "    Two-electron integral element (SZ).\n\n"
            "    Args:\n"
            "        i, j, k, l : spatial indices\n"
            "        sij, skl : spin indices (0=alpha, 1=beta)")
        .def("count_non_zero", &FCIDUMP<FL>::count_non_zero)
        .def("det_energy", &FCIDUMP<FL>::det_energy)
        .def("exchange_matrix", &FCIDUMP<FL>::exchange_matrix)
        .def("abs_exchange_matrix", &FCIDUMP<FL>::abs_exchange_matrix)
        .def("h1e_matrix", &FCIDUMP<FL>::h1e_matrix, py::arg("s") = -1)
        .def("g2e_1fold", &FCIDUMP<FL>::g2e_1fold, py::arg("sl") = -1,
             py::arg("sr") = -1)
        .def("g2e_4fold", &FCIDUMP<FL>::g2e_4fold, py::arg("sl") = -1,
             py::arg("sr") = -1)
        .def("g2e_8fold", &FCIDUMP<FL>::g2e_8fold, py::arg("sl") = -1,
             py::arg("sr") = -1)
        .def("abs_h1e_matrix", &FCIDUMP<FL>::abs_h1e_matrix)
        .def("reorder", (void(FCIDUMP<FL>::*)(const vector<uint16_t> &)) &
                            FCIDUMP<FL>::reorder)
        .def("rescale", &FCIDUMP<FL>::rescale, py::arg("shift") = (FL)0.0)
        .def("rotate", &FCIDUMP<FL>::rotate)
        .def("deep_copy", &FCIDUMP<FL>::deep_copy)
        .def_static("array_reorder", &FCIDUMP<FL>::template reorder<double>)
        .def_static("array_reorder", &FCIDUMP<FL>::template reorder<uint8_t>)
        .def_property("orb_sym", &FCIDUMP<FL>::template orb_sym<uint8_t>,
                      &FCIDUMP<FL>::template set_orb_sym<uint8_t>,
                      "Orbital symmetry in molpro convention")
        .def_property("orb_sym_lz", &FCIDUMP<FL>::template orb_sym<int16_t>,
                      &FCIDUMP<FL>::template set_orb_sym<int16_t>,
                      "Orbital symmetry in Lz convention")
        .def_property("k_sym", &FCIDUMP<FL>::template k_sym<int>,
                      &FCIDUMP<FL>::template set_k_sym<int>)
        .def_property("k_mod", &FCIDUMP<FL>::k_mod, &FCIDUMP<FL>::set_k_mod)
        .def_property("k_isym", &FCIDUMP<FL>::k_isym, &FCIDUMP<FL>::set_k_isym)
        .def_property_readonly("n_elec", &FCIDUMP<FL>::n_elec)
        .def_property_readonly("twos", &FCIDUMP<FL>::twos)
        .def_property_readonly("isym", &FCIDUMP<FL>::isym)
        .def_property_readonly("n_sites", &FCIDUMP<FL>::n_sites)
        .def_readwrite("params", &FCIDUMP<FL>::params)
        .def_readwrite("ts", &FCIDUMP<FL>::ts)
        .def_readwrite("vs", &FCIDUMP<FL>::vs)
        .def_readwrite("vabs", &FCIDUMP<FL>::vabs)
        .def_readwrite("vgs", &FCIDUMP<FL>::vgs)
        .def_readwrite("const_e", &FCIDUMP<FL>::const_e)
        .def_readwrite("total_memory", &FCIDUMP<FL>::total_memory)
        .def_readwrite("uhf", &FCIDUMP<FL>::uhf)
        .def_readwrite("general", &FCIDUMP<FL>::general);

    py::class_<CompressedFCIDUMP<FL>, shared_ptr<CompressedFCIDUMP<FL>>,
               FCIDUMP<FL>>(m, "CompressedFCIDUMP")
        .def(py::init<typename CompressedFCIDUMP<FL>::FP>())
        .def(py::init<typename CompressedFCIDUMP<FL>::FP, int>())
        .def(py::init<typename CompressedFCIDUMP<FL>::FP, int, size_t>())
        .def("freeze", &CompressedFCIDUMP<FL>::freeze)
        .def("unfreeze", &CompressedFCIDUMP<FL>::unfreeze)
        .def_readwrite("prec", &CompressedFCIDUMP<FL>::prec)
        .def_readwrite("ncache", &CompressedFCIDUMP<FL>::ncache)
        .def_readwrite("chunk_size", &CompressedFCIDUMP<FL>::chunk_size)
        .def_readwrite("cps_ts", &CompressedFCIDUMP<FL>::cps_ts)
        .def_readwrite("cps_vs", &CompressedFCIDUMP<FL>::cps_vs)
        .def_readwrite("cps_vabs", &CompressedFCIDUMP<FL>::cps_vabs)
        .def_readwrite("cps_vgs", &CompressedFCIDUMP<FL>::cps_vgs)
        .def("initialize_su2",
             [](CompressedFCIDUMP<FL> *self, uint16_t n_sites, uint16_t n_elec,
                uint16_t twos, uint16_t isym, FL e, const py::array_t<FL> &t,
                const py::array_t<FL> &v) {
                 stringstream st, sv;
                 st.write((char *)t.data(), sizeof(FL) * t.size());
                 st.clear(), st.seekg(0);
                 sv.write((char *)v.data(), sizeof(FL) * v.size());
                 sv.clear(), sv.seekg(0);
                 self->initialize_su2(n_sites, n_elec, twos, isym, e, st,
                                      t.size(), sv, v.size());
             })
        .def("initialize_su2",
             [](CompressedFCIDUMP<FL> *self, uint16_t n_sites, uint16_t n_elec,
                uint16_t twos, uint16_t isym, FL e, const string &ft,
                const string &fv) {
                 ifstream ift(ft.c_str(), ios::binary);
                 ifstream ifv(fv.c_str(), ios::binary);
                 if (!ift.good())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + ft +
                         "' failed.");
                 if (!ifv.good())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + fv +
                         "' failed.");
                 size_t t_len, v_len;
                 ift >> t_len;
                 ifv >> v_len;
                 self->initialize_su2(n_sites, n_elec, twos, isym, e, ift,
                                      t_len, ifv, v_len);
                 if (ift.fail() || ift.bad())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + ft +
                         "' failed.");
                 if (ifv.fail() || ifv.bad())
                     throw runtime_error(
                         "CompressedFCIDUMP::initialize_su2 on '" + fv +
                         "' failed.");
                 ifv.close();
                 ift.close();
             })
        .def("initialize_sz", [](CompressedFCIDUMP<FL> *self, uint16_t n_sites,
                                 uint16_t n_elec, int16_t twos, uint16_t isym,
                                 FL e, const py::tuple &t, const py::tuple &v) {
            assert(t.size() == 2 && v.size() == 3);
            if (py::isinstance<py::array_t<FL>>(t[0])) {
                py::array_t<FL> ta = t[0].cast<py::array_t<FL>>();
                py::array_t<FL> tb = t[1].cast<py::array_t<FL>>();
                py::array_t<FL> va = v[0].cast<py::array_t<FL>>();
                py::array_t<FL> vb = v[1].cast<py::array_t<FL>>();
                py::array_t<FL> vab = v[2].cast<py::array_t<FL>>();
                stringstream sta, stb, sva, svb, svab;
                sta.write((char *)ta.data(), sizeof(FL) * ta.size());
                sta.clear(), sta.seekg(0);
                stb.write((char *)tb.data(), sizeof(FL) * tb.size());
                stb.clear(), stb.seekg(0);
                sva.write((char *)va.data(), sizeof(FL) * va.size());
                sva.clear(), sva.seekg(0);
                svb.write((char *)vb.data(), sizeof(FL) * vb.size());
                svb.clear(), svb.seekg(0);
                svab.write((char *)vab.data(), sizeof(FL) * vab.size());
                svab.clear(), svab.seekg(0);
                self->initialize_sz(n_sites, n_elec, twos, isym, e, sta,
                                    ta.size(), stb, tb.size(), sva, va.size(),
                                    svb, vb.size(), svab, vab.size());
            } else {
                string fta = t[0].cast<string>();
                string ftb = t[1].cast<string>();
                string fva = v[0].cast<string>();
                string fvb = v[1].cast<string>();
                string fvab = v[2].cast<string>();
                ifstream ifta(fta.c_str(), ios::binary);
                ifstream iftb(ftb.c_str(), ios::binary);
                ifstream ifva(fva.c_str(), ios::binary);
                ifstream ifvb(fvb.c_str(), ios::binary);
                ifstream ifvab(fvab.c_str(), ios::binary);
                if (!ifta.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fta +
                        "' failed.");
                if (!iftb.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + ftb +
                        "' failed.");
                if (!ifva.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fva +
                        "' failed.");
                if (!ifvb.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvb +
                        "' failed.");
                if (!ifvab.good())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvab +
                        "' failed.");
                size_t ta_len, tb_len, va_len, vb_len, vab_len;
                ifta >> ta_len;
                iftb >> tb_len;
                ifva >> va_len;
                ifvb >> vb_len;
                ifvab >> vab_len;
                self->initialize_sz(n_sites, n_elec, twos, isym, e, ifta,
                                    ta_len, iftb, tb_len, ifva, va_len, ifvb,
                                    vb_len, ifvab, vab_len);
                if (ifta.fail() || ifta.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fta +
                        "' failed.");
                if (iftb.fail() || iftb.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + ftb +
                        "' failed.");
                if (ifva.fail() || ifva.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fva +
                        "' failed.");
                if (ifvb.fail() || ifvb.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvb +
                        "' failed.");
                if (ifvab.fail() || ifvab.bad())
                    throw runtime_error(
                        "CompressedFCIDUMP::initialize_sz on '" + fvab +
                        "' failed.");
                ifvab.close();
                ifvb.close();
                ifva.close();
                iftb.close();
                ifta.close();
            }
        });

    py::class_<SpinOrbitalFCIDUMP<FL>, shared_ptr<SpinOrbitalFCIDUMP<FL>>,
               FCIDUMP<FL>>(m, "SpinOrbitalFCIDUMP")
        .def(py::init<const shared_ptr<FCIDUMP<FL>> &>())
        .def_readwrite("prim_fcidump", &SpinOrbitalFCIDUMP<FL>::prim_fcidump);

    py::class_<MRCISFCIDUMP<FL>, shared_ptr<MRCISFCIDUMP<FL>>, FCIDUMP<FL>>(
        m, "MRCISFCIDUMP")
        .def(py::init<const shared_ptr<FCIDUMP<FL>> &, uint16_t, uint16_t>())
        .def_readwrite("prim_fcidump", &MRCISFCIDUMP<FL>::prim_fcidump)
        .def_readwrite("n_inactive", &MRCISFCIDUMP<FL>::n_inactive)
        .def_readwrite("n_virtual", &MRCISFCIDUMP<FL>::n_virtual)
        .def_readwrite("n_active", &MRCISFCIDUMP<FL>::n_active);

    py::class_<BatchGEMMSeq<FL>, shared_ptr<BatchGEMMSeq<FL>>>(m,
                                                               "BatchGEMMSeq")
        .def_readwrite("batch", &BatchGEMMSeq<FL>::batch)
        .def_readwrite("post_batch", &BatchGEMMSeq<FL>::post_batch)
        .def_readwrite("refs", &BatchGEMMSeq<FL>::refs)
        .def_readwrite("cumulative_nflop", &BatchGEMMSeq<FL>::cumulative_nflop)
        .def_readwrite("mode", &BatchGEMMSeq<FL>::mode)
        .def(py::init<>())
        .def(py::init<size_t>())
        .def(py::init<size_t, SeqTypes>())
        .def("iadd", &BatchGEMMSeq<FL>::iadd, py::arg("a"), py::arg("b"),
             py::arg("scale") = 1.0, py::arg("cfactor") = 1.0,
             py::arg("conj") = false)
        .def("multiply", &BatchGEMMSeq<FL>::multiply, py::arg("a"),
             py::arg("conja"), py::arg("b"), py::arg("conjb"), py::arg("c"),
             py::arg("scale"), py::arg("cfactor"))
        .def("rotate",
             (void(BatchGEMMSeq<FL>::*)(
                 const GMatrix<FL> &, const GMatrix<FL> &, const GMatrix<FL> &,
                 uint8_t, const GMatrix<FL> &, uint8_t, FL)) &
                 BatchGEMMSeq<FL>::rotate,
             py::arg("a"), py::arg("c"), py::arg("bra"), py::arg("conj_bra"),
             py::arg("ket"), py::arg("conj_ket"), py::arg("scale"))
        .def("rotate",
             (void(BatchGEMMSeq<FL>::*)(
                 const GMatrix<FL> &, bool, const GMatrix<FL> &, bool,
                 const GMatrix<FL> &, const GMatrix<FL> &, FL)) &
                 BatchGEMMSeq<FL>::rotate,
             py::arg("a"), py::arg("conj_a"), py::arg("c"), py::arg("conj_c"),
             py::arg("bra"), py::arg("ket"), py::arg("scale"))
        .def("tensor_product_diagonal",
             &BatchGEMMSeq<FL>::tensor_product_diagonal, py::arg("conj"),
             py::arg("a"), py::arg("b"), py::arg("c"), py::arg("scale"))
        .def("tensor_product", &BatchGEMMSeq<FL>::tensor_product, py::arg("a"),
             py::arg("conja"), py::arg("b"), py::arg("conjb"), py::arg("c"),
             py::arg("scale"), py::arg("stride"))
        .def("divide_batch", &BatchGEMMSeq<FL>::divide_batch)
        .def("check", &BatchGEMMSeq<FL>::check)
        .def("prepare", &BatchGEMMSeq<FL>::prepare)
        .def("allocate", &BatchGEMMSeq<FL>::allocate)
        .def("deallocate", &BatchGEMMSeq<FL>::deallocate)
        .def("simple_perform", &BatchGEMMSeq<FL>::simple_perform)
        .def("auto_perform",
             (void(BatchGEMMSeq<FL>::*)(const GMatrix<FL> &)) &
                 BatchGEMMSeq<FL>::auto_perform,
             py::arg("v") = GMatrix<FL>(nullptr, 0, 0))
        .def("auto_perform",
             (void(BatchGEMMSeq<FL>::*)(const vector<GMatrix<FL>> &)) &
                 BatchGEMMSeq<FL>::auto_perform,
             py::arg("vs"))
        .def("perform", &BatchGEMMSeq<FL>::perform)
        .def("clear", &BatchGEMMSeq<FL>::clear)
        .def("__call__", &BatchGEMMSeq<FL>::operator())
        .def("__repr__", [](BatchGEMMSeq<FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });
}

template <typename FL> void bind_general_fcidump(py::module &m) {

    py::class_<GeneralFCIDUMP<FL>, shared_ptr<GeneralFCIDUMP<FL>>>(
        m, "GeneralFCIDUMP")
        .def(py::init<>())
        .def_readwrite("params", &GeneralFCIDUMP<FL>::params)
        .def_readwrite("const_e", &GeneralFCIDUMP<FL>::const_e)
        .def_readwrite("exprs", &GeneralFCIDUMP<FL>::exprs)
        .def_readwrite("indices", &GeneralFCIDUMP<FL>::indices)
        .def_readwrite("data", &GeneralFCIDUMP<FL>::data)
        .def_readwrite("elem_type", &GeneralFCIDUMP<FL>::elem_type)
        .def_readwrite("order_adjusted", &GeneralFCIDUMP<FL>::order_adjusted)
        .def(
            "add_eight_fold_term",
            [](GeneralFCIDUMP<FL> *self, const py::array_t<FL> &v,
               typename GeneralFCIDUMP<FL>::FP cutoff, FL factor) {
                self->add_eight_fold_term(v.data(), (size_t)v.size(), cutoff,
                                          factor);
            },
            py::arg("v"), py::arg("cutoff"), py::arg("factor"))
        .def(
            "add_sum_term",
            [](GeneralFCIDUMP<FL> *self, const py::array_t<FL> &v,
               typename GeneralFCIDUMP<FL>::FP cutoff, FL factor,
               const vector<uint16_t> &perm) {
                vector<int> shape(v.ndim());
                vector<size_t> strides(v.ndim());
                for (int i = 0; i < v.ndim(); i++)
                    shape[i] = (int)v.shape()[i],
                    strides[i] = v.strides()[i] / sizeof(FL);
                vector<uint16_t> rperm(v.ndim());
                if (perm.size() == 0)
                    for (int i = 0; i < v.ndim(); i++)
                        rperm[i] = i;
                else
                    for (int i = 0; i < v.ndim(); i++)
                        rperm[perm[i]] = i;
                self->add_sum_term(v.data(), (size_t)v.size(), shape, strides,
                                   cutoff, factor, vector<int>(), rperm);
            },
            py::arg("v"), py::arg("cutoff"), py::arg("factor") = (FL)1.0,
            py::arg("perm") = vector<uint16_t>())
        .def_static("initialize_from_qc",
                    &GeneralFCIDUMP<FL>::initialize_from_qc, py::arg("fcidump"),
                    py::arg("elem_type"),
                    py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("adjust_order",
             (shared_ptr<GeneralFCIDUMP<FL>>(GeneralFCIDUMP<FL>::*)(
                 const string &, bool, typename GeneralFCIDUMP<FL>::FP) const) &
                 GeneralFCIDUMP<FL>::adjust_order,
             py::arg("fermionic_ops"), py::arg("merge") = true,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("adjust_order",
             (shared_ptr<GeneralFCIDUMP<FL>>(GeneralFCIDUMP<FL>::*)(
                 bool, bool, typename GeneralFCIDUMP<FL>::FP) const) &
                 GeneralFCIDUMP<FL>::adjust_order,
             py::arg("merge") = true, py::arg("is_drt") = false,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("adjust_order",
             (shared_ptr<GeneralFCIDUMP<FL>>(GeneralFCIDUMP<FL>::*)(
                 const vector<shared_ptr<SpinPermScheme>> &, bool, bool,
                 typename GeneralFCIDUMP<FL>::FP) const) &
                 GeneralFCIDUMP<FL>::adjust_order,
             py::arg("schemes"), py::arg("merge") = true,
             py::arg("is_drt") = false,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("adjust_order",
             (shared_ptr<GeneralFCIDUMP<FL>>(GeneralFCIDUMP<FL>::*)(
                 const vector<shared_ptr<GeneralSymmPermScheme<FL>>> &, bool,
                 bool, typename GeneralFCIDUMP<FL>::FP) const) &
                 GeneralFCIDUMP<FL>::adjust_order,
             py::arg("schemes"), py::arg("merge") = true,
             py::arg("is_drt") = false,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("merge_terms", &GeneralFCIDUMP<FL>::merge_terms,
             py::arg("cutoff") = (typename GeneralFCIDUMP<FL>::FP)0.0)
        .def("find_mask", &GeneralFCIDUMP<FL>::find_mask)
        .def("twos", &GeneralFCIDUMP<FL>::twos)
        .def("n_sites", &GeneralFCIDUMP<FL>::n_sites)
        .def("n_elec", &GeneralFCIDUMP<FL>::n_elec)
        .def("e", &GeneralFCIDUMP<FL>::e)
        .def_property_readonly("orb_sym",
                               &GeneralFCIDUMP<FL>::template orb_sym<uint8_t>)
        .def_property_readonly("orb_sym_lz",
                               &GeneralFCIDUMP<FL>::template orb_sym<int16_t>)
        .def("__add__", &GeneralFCIDUMP<FL>::add)
        .def("__repr__", [](GeneralFCIDUMP<FL> *self) {
            stringstream ss;
            ss << *self;
            return ss.str();
        });
}

template <typename S = void> void bind_symmetry(py::module &m) {

    py::class_<OpNamesSet>(m, "OpNamesSet")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init([](py::tuple names) {
            vector<OpNames> x(names.size());
            for (size_t i = 0; i < names.size(); i++)
                x[i] = names[i].cast<OpNames>();
            return OpNamesSet(x);
        }))
        .def_readwrite("data", &OpNamesSet::data)
        .def("__call__", &OpNamesSet::operator())
        .def("empty", &OpNamesSet::empty)
        .def_static("normal_ops", &OpNamesSet::normal_ops)
        .def_static("all_ops", &OpNamesSet::all_ops);

    py::class_<SiteIndex>(m, "SiteIndex")
        .def(py::init<>())
        .def(py::init<uint16_t>())
        .def(py::init<uint16_t, uint16_t, uint8_t>())
        .def(py::init([](py::tuple idxs, py::tuple sidxs) {
            vector<uint16_t> x(idxs.size());
            vector<uint8_t> sx(sidxs.size());
            for (size_t i = 0; i < idxs.size(); i++)
                x[i] = idxs[i].cast<uint16_t>();
            for (size_t i = 0; i < sidxs.size(); i++)
                sx[i] = sidxs[i].cast<uint8_t>();
            return SiteIndex(x, sx);
        }))
        .def("size", &SiteIndex::size)
        .def("spin_size", &SiteIndex::spin_size)
        .def("s", &SiteIndex::s, py::arg("i") = 0)
        .def("__getitem__",
             [](SiteIndex *self, uint8_t i) { return self->operator[](i); })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__hash__", &SiteIndex::hash)
        .def("__repr__", &SiteIndex::to_str);

    bind_array_fixed<SAnySymmTypes, SAny::ll>(m, "ArrayLSAnySymmTypes");
    bind_array_fixed<int32_t, SAny::ll>(m, "ArrayLInt32");

    py::class_<SAny>(m, "SAny")
        .def(py::init<>())
        .def(py::init<const array<SAnySymmTypes, SAny::ll> &>())
        .def(py::init<const array<SAnySymmTypes, SAny::ll> &,
                      const array<int32_t, SAny::ll> &>())
        .def(py::init<SAnySymmTypes>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SAny(SAny::invalid); })
        .def_readwrite("types", &SAny::types)
        .def_readwrite("values", &SAny::values)
        .def_static("invalid_init", &SAny::invalid_init)
        .def_static("empty_types_init", &SAny::empty_types_init)
        .def_static("empty_values_init", &SAny::empty_values_init)
        .def_static("init_su2", (SAny(*)(int, int, int)) & SAny::init_su2,
                    py::arg("n") = 0, py::arg("twos") = 0, py::arg("pg") = 0)
        .def_static("init_su2", (SAny(*)(int, int, int, int)) & SAny::init_su2,
                    py::arg("n"), py::arg("twos_low"), py::arg("twos"),
                    py::arg("pg"))
        .def_static("init_sz", &SAny::init_sz, py::arg("n") = 0,
                    py::arg("twos") = 0, py::arg("pg") = 0)
        .def_static("init_sgf", (SAny(*)(int, int)) & SAny::init_sgf,
                    py::arg("n") = 0, py::arg("pg") = 0)
        .def_static("init_sgf", (SAny(*)(int, int, int)) & SAny::init_sgf,
                    py::arg("n"), py::arg("twos"), py::arg("pg"))
        .def_static("init_sgb", &SAny::init_sgb, py::arg("n") = 0,
                    py::arg("pg") = 0)
        .def_property("n", &SAny::n, &SAny::set_n)
        .def_property("u1", &SAny::u1, &SAny::set_u1)
        .def_property("twos", &SAny::twos, &SAny::set_twos)
        .def_property("twos_low", &SAny::twos_low, &SAny::set_twos_low)
        .def_property("pg", &SAny::pg, &SAny::set_pg)
        .def_property_readonly("multiplicity", &SAny::multiplicity)
        .def_property_readonly("is_fermion", &SAny::is_fermion)
        .def_property_readonly("count", &SAny::count)
        .def_static("pg_mul", &SAny::pg_mul)
        .def_static("pg_inv", &SAny::pg_inv)
        .def_static("pg_combine", &SAny::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SAny::pg_equal)
        .def("symm_len", &SAny::symm_len)
        .def("su2_indices", &SAny::su2_indices)
        .def("non_abelian_indices", &SAny::non_abelian_indices)
        .def("u1_indices", &SAny::u1_indices)
        .def("combine", &SAny::combine)
        .def("__getitem__", &SAny::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SAny::get_ket)
        .def("get_bra", &SAny::get_bra, py::arg("dq"))
        .def("__hash__", &SAny::hash)
        .def("__repr__", &SAny::to_str);

    py::bind_vector<vector<SAny>>(m, "VectorSAny");
    py::bind_vector<vector<vector<SAny>>>(m, "VectorVectorSAny");

    py::class_<SZ>(m, "SZ")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SZ(SZ::invalid); })
        .def_readwrite("data", &SZ::data)
        .def_property("n", &SZ::n, &SZ::set_n)
        .def_property("twos", &SZ::twos, &SZ::set_twos)
        .def_property("pg", &SZ::pg, &SZ::set_pg)
        .def_property_readonly("multiplicity", &SZ::multiplicity)
        .def_property_readonly("is_fermion", &SZ::is_fermion)
        .def_property_readonly("count", &SZ::count)
        .def_static("pg_mul", &SZ::pg_mul)
        .def_static("pg_inv", &SZ::pg_inv)
        .def_static("pg_combine", &SZ::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SZ::pg_equal)
        .def("combine", &SZ::combine)
        .def("__getitem__", &SZ::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SZ::get_ket)
        .def("get_bra", &SZ::get_bra, py::arg("dq"))
        .def("__hash__", &SZ::hash)
        .def("__repr__", &SZ::to_str);

    py::bind_vector<vector<SZ>>(m, "VectorSZ");
    py::bind_vector<vector<vector<SZ>>>(m, "VectorVectorSZ");

    py::class_<SU2>(m, "SU2")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SU2(SU2::invalid); })
        .def_readwrite("data", &SU2::data)
        .def_property("n", &SU2::n, &SU2::set_n)
        .def_property("twos", &SU2::twos, &SU2::set_twos)
        .def_property("twos_low", &SU2::twos_low, &SU2::set_twos_low)
        .def_property("pg", &SU2::pg, &SU2::set_pg)
        .def_property_readonly("multiplicity", &SU2::multiplicity)
        .def_property_readonly("is_fermion", &SU2::is_fermion)
        .def_property_readonly("count", &SU2::count)
        .def_static("pg_mul", &SU2::pg_mul)
        .def_static("pg_inv", &SU2::pg_inv)
        .def_static("pg_combine", &SU2::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SU2::pg_equal)
        .def("combine", &SU2::combine)
        .def("__getitem__", &SU2::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SU2::get_ket)
        .def("get_bra", &SU2::get_bra, py::arg("dq"))
        .def("__hash__", &SU2::hash)
        .def("__repr__", &SU2::to_str);

    py::bind_vector<vector<SU2>>(m, "VectorSU2");
    py::bind_vector<vector<vector<SU2>>>(m, "VectorVectorSU2");

    py::class_<SGF>(m, "SGF")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int>())
        .def(py::init<int, int, int>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SGF(SGF::invalid); })
        .def_readwrite("data", &SGF::data)
        .def_property("n", &SGF::n, &SGF::set_n)
        .def_property_readonly("twos", &SGF::twos)
        .def_property("pg", &SGF::pg, &SGF::set_pg)
        .def_property_readonly("multiplicity", &SGF::multiplicity)
        .def_property_readonly("is_fermion", &SGF::is_fermion)
        .def_property_readonly("count", &SGF::count)
        .def_static("pg_mul", &SGF::pg_mul)
        .def_static("pg_inv", &SGF::pg_inv)
        .def_static("pg_combine", &SGF::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SGF::pg_equal)
        .def("combine", &SGF::combine)
        .def("__getitem__", &SGF::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SGF::get_ket)
        .def("get_bra", &SGF::get_bra, py::arg("dq"))
        .def("__hash__", &SGF::hash)
        .def("__repr__", &SGF::to_str);

    py::bind_vector<vector<SGF>>(m, "VectorSGF");
    py::bind_vector<vector<vector<SGF>>>(m, "VectorVectorSGF");

    py::class_<SGB>(m, "SGB")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int>())
        .def(py::init<int, int, int>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SGB(SGB::invalid); })
        .def_readwrite("data", &SGB::data)
        .def_property("n", &SGB::n, &SGB::set_n)
        .def_property_readonly("twos", &SGB::twos)
        .def_property("pg", &SGB::pg, &SGB::set_pg)
        .def_property_readonly("multiplicity", &SGB::multiplicity)
        .def_property_readonly("is_fermion", &SGB::is_fermion)
        .def_property_readonly("count", &SGB::count)
        .def_static("pg_mul", &SGB::pg_mul)
        .def_static("pg_inv", &SGB::pg_inv)
        .def_static("pg_combine", &SGB::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SGB::pg_equal)
        .def("combine", &SGB::combine)
        .def("__getitem__", &SGB::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SGB::get_ket)
        .def("get_bra", &SGB::get_bra, py::arg("dq"))
        .def("__hash__", &SGB::hash)
        .def("__repr__", &SGB::to_str);

    py::bind_vector<vector<SGB>>(m, "VectorSGB");
    py::bind_vector<vector<vector<SGB>>>(m, "VectorVectorSGB");

    py::class_<SZK>(m, "SZK")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int, int>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SZK(SZK::invalid); })
        .def_readwrite("data", &SZK::data)
        .def_property("n", &SZK::n, &SZK::set_n)
        .def_property("twos", &SZK::twos, &SZK::set_twos)
        .def_property("pg", &SZK::pg, &SZK::set_pg)
        .def_property_readonly("pg_pg", &SZK::pg_pg)
        .def_property_readonly("pg_k", &SZK::pg_k)
        .def_property_readonly("pg_k_mod", &SZK::pg_k_mod)
        .def_property_readonly("multiplicity", &SZK::multiplicity)
        .def_property_readonly("is_fermion", &SZK::is_fermion)
        .def_property_readonly("count", &SZK::count)
        .def_static("pg_mul", &SZK::pg_mul)
        .def_static("pg_inv", &SZK::pg_inv)
        .def_static("pg_combine", &SZK::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SZK::pg_equal)
        .def("combine", &SZK::combine)
        .def("__getitem__", &SZK::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SZK::get_ket)
        .def("get_bra", &SZK::get_bra, py::arg("dq"))
        .def("__hash__", &SZK::hash)
        .def("__repr__", &SZK::to_str);

    py::bind_vector<vector<SZK>>(m, "VectorSZK");
    py::bind_vector<vector<vector<SZK>>>(m, "VectorVectorSZK");

    py::class_<SU2K>(m, "SU2K")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<int, int, int>())
        .def(py::init<int, int, int, int>())
        .def(py::init<int, int, int, int, int, int>())
        .def_property_readonly_static(
            "invalid", [](py::object) { return SU2K(SU2K::invalid); })
        .def_readwrite("data", &SU2K::data)
        .def_property("n", &SU2K::n, &SU2K::set_n)
        .def_property("twos", &SU2K::twos, &SU2K::set_twos)
        .def_property("twos_low", &SU2K::twos_low, &SU2K::set_twos_low)
        .def_property("pg", &SU2K::pg, &SU2K::set_pg)
        .def_property_readonly("pg_pg", &SU2K::pg_pg)
        .def_property_readonly("pg_k", &SU2K::pg_k)
        .def_property_readonly("pg_k_mod", &SU2K::pg_k_mod)
        .def_property_readonly("multiplicity", &SU2K::multiplicity)
        .def_property_readonly("is_fermion", &SU2K::is_fermion)
        .def_property_readonly("count", &SU2K::count)
        .def_static("pg_mul", &SU2K::pg_mul)
        .def_static("pg_inv", &SU2K::pg_inv)
        .def_static("pg_combine", &SU2K::pg_combine, py::arg("pg"),
                    py::arg("k") = 0, py::arg("kmod") = 0)
        .def_static("pg_equal", &SU2K::pg_equal)
        .def("combine", &SU2K::combine)
        .def("__getitem__", &SU2K::operator[])
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def("get_ket", &SU2K::get_ket)
        .def("get_bra", &SU2K::get_bra, py::arg("dq"))
        .def("__hash__", &SU2K::hash)
        .def("__repr__", &SU2K::to_str);

    py::bind_vector<vector<SU2K>>(m, "VectorSU2K");
    py::bind_vector<vector<vector<SU2K>>>(m, "VectorVectorSU2K");
}

#ifdef _EXPLICIT_TEMPLATE

extern template void bind_data<>(py::module &m);
extern template void bind_types<>(py::module &m);
extern template void bind_io<>(py::module &m);
extern template void bind_symmetry<>(py::module &m);

extern template void bind_fl_io<double>(py::module &m, const string &name);
extern template void bind_matrix<double>(py::module &m);
extern template void bind_fl_matrix<double>(py::module &m);
extern template void bind_general_fcidump<double>(py::module &m);

extern template void bind_fl_data<double>(py::module &m, const string &name);

extern template void bind_post_matrix<>(py::module &m);

#ifdef _USE_SU2SZ
extern template void bind_cg<SZ>(py::module &m);
extern template void bind_expr<SZ>(py::module &m);
extern template void bind_state_info<SZ>(py::module &m, const string &name);
extern template void bind_sparse<SZ>(py::module &m);
extern template void bind_parallel<SZ>(py::module &m);

extern template void bind_fl_expr<SZ, double>(py::module &m);
extern template void bind_fl_state_info<SZ, double>(py::module &m,
                                                    const string &name);
extern template void bind_fl_sparse<SZ, double>(py::module &m);
extern template void bind_fl_parallel<SZ, double>(py::module &m);
extern template void bind_fl_operator<SZ, double>(py::module &m);
extern template void bind_fl_hamiltonian<SZ, double>(py::module &m);
extern template void bind_fl_rule<SZ, double>(py::module &m);

extern template void bind_cg<SU2>(py::module &m);
extern template void bind_expr<SU2>(py::module &m);
extern template void bind_state_info<SU2>(py::module &m, const string &name);
extern template void bind_sparse<SU2>(py::module &m);
extern template void bind_parallel<SU2>(py::module &m);

extern template void bind_fl_expr<SU2, double>(py::module &m);
extern template void bind_fl_state_info<SU2, double>(py::module &m,
                                                     const string &name);
extern template void bind_fl_sparse<SU2, double>(py::module &m);
extern template void bind_fl_parallel<SU2, double>(py::module &m);
extern template void bind_fl_operator<SU2, double>(py::module &m);
extern template void bind_fl_hamiltonian<SU2, double>(py::module &m);
extern template void bind_fl_rule<SU2, double>(py::module &m);

extern template void bind_trans_state_info<SU2, SZ>(py::module &m,
                                                    const string &aux_name);
extern template void bind_trans_state_info<SZ, SU2>(py::module &m,
                                                    const string &aux_name);
extern template auto
bind_trans_state_info_spin_specific<SU2, SZ>(py::module &m,
                                             const string &aux_name)
    -> decltype(typename SU2::is_su2_t(typename SZ::is_sz_t()));
#endif

#ifdef _USE_COMPLEX

extern template void bind_fl_matrix<complex<double>>(py::module &m);
extern template void bind_general_fcidump<complex<double>>(py::module &m);

#ifdef _USE_SU2SZ
extern template void bind_fl_expr<SZ, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SZ, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SZ, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SZ, complex<double>>(py::module &m);
extern template void bind_fl_operator<SZ, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SZ, complex<double>>(py::module &m);
extern template void bind_fl_rule<SZ, complex<double>>(py::module &m);

extern template void bind_fl_expr<SU2, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SU2, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SU2, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SU2, complex<double>>(py::module &m);
extern template void bind_fl_operator<SU2, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SU2, complex<double>>(py::module &m);
extern template void bind_fl_rule<SU2, complex<double>>(py::module &m);
#endif

#endif

#ifdef _USE_KSYMM
extern template void bind_cg<SZK>(py::module &m);
extern template void bind_expr<SZK>(py::module &m);
extern template void bind_state_info<SZK>(py::module &m, const string &name);
extern template void bind_sparse<SZK>(py::module &m);
extern template void bind_parallel<SZK>(py::module &m);

extern template void bind_fl_expr<SZK, double>(py::module &m);
extern template void bind_fl_state_info<SZK, double>(py::module &m,
                                                     const string &name);
extern template void bind_fl_sparse<SZK, double>(py::module &m);
extern template void bind_fl_parallel<SZK, double>(py::module &m);
extern template void bind_fl_operator<SZK, double>(py::module &m);
extern template void bind_fl_hamiltonian<SZK, double>(py::module &m);
extern template void bind_fl_rule<SZK, double>(py::module &m);

extern template void bind_cg<SU2K>(py::module &m);
extern template void bind_expr<SU2K>(py::module &m);
extern template void bind_state_info<SU2K>(py::module &m, const string &name);
extern template void bind_sparse<SU2K>(py::module &m);
extern template void bind_parallel<SU2K>(py::module &m);

extern template void bind_fl_expr<SU2K, double>(py::module &m);
extern template void bind_fl_state_info<SU2K, double>(py::module &m,
                                                      const string &name);
extern template void bind_fl_sparse<SU2K, double>(py::module &m);
extern template void bind_fl_parallel<SU2K, double>(py::module &m);
extern template void bind_fl_operator<SU2K, double>(py::module &m);
extern template void bind_fl_hamiltonian<SU2K, double>(py::module &m);
extern template void bind_fl_rule<SU2K, double>(py::module &m);

extern template void bind_trans_state_info<SU2K, SZK>(py::module &m,
                                                      const string &aux_name);
extern template void bind_trans_state_info<SZK, SU2K>(py::module &m,
                                                      const string &aux_name);
extern template auto
bind_trans_state_info_spin_specific<SU2K, SZK>(py::module &m,
                                               const string &aux_name)
    -> decltype(typename SU2K::is_su2_t(typename SZK::is_sz_t()));

#ifdef _USE_COMPLEX
extern template void bind_fl_expr<SZK, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SZK, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SZK, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SZK, complex<double>>(py::module &m);
extern template void bind_fl_operator<SZK, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SZK, complex<double>>(py::module &m);
extern template void bind_fl_rule<SZK, complex<double>>(py::module &m);

extern template void bind_fl_expr<SU2K, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SU2K, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SU2K, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SU2K, complex<double>>(py::module &m);
extern template void bind_fl_operator<SU2K, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SU2K, complex<double>>(py::module &m);
extern template void bind_fl_rule<SU2K, complex<double>>(py::module &m);

#endif

#endif

#ifdef _USE_SG
extern template void bind_cg<SGF>(py::module &m);
extern template void bind_expr<SGF>(py::module &m);
extern template void bind_state_info<SGF>(py::module &m, const string &name);
extern template void bind_sparse<SGF>(py::module &m);
extern template void bind_parallel<SGF>(py::module &m);

extern template void bind_fl_expr<SGF, double>(py::module &m);
extern template void bind_fl_state_info<SGF, double>(py::module &m,
                                                     const string &name);
extern template void bind_fl_sparse<SGF, double>(py::module &m);
extern template void bind_fl_parallel<SGF, double>(py::module &m);
extern template void bind_fl_operator<SGF, double>(py::module &m);
extern template void bind_fl_hamiltonian<SGF, double>(py::module &m);
extern template void bind_fl_rule<SGF, double>(py::module &m);

extern template void bind_cg<SGB>(py::module &m);
extern template void bind_expr<SGB>(py::module &m);
extern template void bind_state_info<SGB>(py::module &m, const string &name);
extern template void bind_sparse<SGB>(py::module &m);
extern template void bind_parallel<SGB>(py::module &m);

extern template void bind_fl_expr<SGB, double>(py::module &m);
extern template void bind_fl_state_info<SGB, double>(py::module &m,
                                                     const string &name);
extern template void bind_fl_sparse<SGB, double>(py::module &m);
extern template void bind_fl_parallel<SGB, double>(py::module &m);
extern template void bind_fl_operator<SGB, double>(py::module &m);
extern template void bind_fl_hamiltonian<SGB, double>(py::module &m);
extern template void bind_fl_rule<SGB, double>(py::module &m);

#ifdef _USE_SU2SZ
extern template void bind_trans_state_info<SZ, SGF>(py::module &m,
                                                    const string &aux_name);
extern template void bind_trans_state_info<SGF, SZ>(py::module &m,
                                                    const string &aux_name);
extern template auto
bind_trans_state_info_spin_specific<SZ, SGF>(py::module &m,
                                             const string &aux_name)
    -> decltype(typename SZ::is_sz_t(typename SGF::is_sg_t()));
#endif

#ifdef _USE_COMPLEX
extern template void bind_fl_expr<SGF, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SGF, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SGF, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SGF, complex<double>>(py::module &m);
extern template void bind_fl_operator<SGF, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SGF, complex<double>>(py::module &m);
extern template void bind_fl_rule<SGF, complex<double>>(py::module &m);

extern template void bind_fl_expr<SGB, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SGB, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SGB, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SGB, complex<double>>(py::module &m);
extern template void bind_fl_operator<SGB, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SGB, complex<double>>(py::module &m);
extern template void bind_fl_rule<SGB, complex<double>>(py::module &m);

#endif

#endif

#ifdef _USE_SANY
extern template void bind_cg<SAny>(py::module &m);
extern template void bind_expr<SAny>(py::module &m);
extern template void bind_state_info<SAny>(py::module &m, const string &name);
extern template void bind_sparse<SAny>(py::module &m);
extern template void bind_parallel<SAny>(py::module &m);

extern template void bind_fl_expr<SAny, double>(py::module &m);
extern template void bind_fl_state_info<SAny, double>(py::module &m,
                                                      const string &name);
extern template void bind_fl_sparse<SAny, double>(py::module &m);
extern template void bind_fl_parallel<SAny, double>(py::module &m);
extern template void bind_fl_operator<SAny, double>(py::module &m);
extern template void bind_fl_hamiltonian<SAny, double>(py::module &m);
extern template void bind_fl_rule<SAny, double>(py::module &m);

#ifdef _USE_COMPLEX
extern template void bind_fl_expr<SAny, complex<double>>(py::module &m);
extern template void
bind_fl_state_info<SAny, complex<double>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SAny, complex<double>>(py::module &m);
extern template void bind_fl_parallel<SAny, complex<double>>(py::module &m);
extern template void bind_fl_operator<SAny, complex<double>>(py::module &m);
extern template void bind_fl_hamiltonian<SAny, complex<double>>(py::module &m);
extern template void bind_fl_rule<SAny, complex<double>>(py::module &m);
#endif

#endif

#ifdef _USE_SINGLE_PREC

extern template void bind_fl_io<float>(py::module &m, const string &name);
extern template void bind_matrix<float>(py::module &m);
extern template void bind_fl_matrix<float>(py::module &m);
extern template void bind_general_fcidump<float>(py::module &m);

#ifdef _USE_SU2SZ
extern template void bind_fl_expr<SZ, float>(py::module &m);
extern template void bind_fl_state_info<SZ, float>(py::module &m,
                                                   const string &name);
extern template void bind_fl_sparse<SZ, float>(py::module &m);
extern template void bind_fl_parallel<SZ, float>(py::module &m);
extern template void bind_fl_operator<SZ, float>(py::module &m);
extern template void bind_fl_hamiltonian<SZ, float>(py::module &m);
extern template void bind_fl_rule<SZ, float>(py::module &m);

extern template void bind_fl_expr<SU2, float>(py::module &m);
extern template void bind_fl_state_info<SU2, float>(py::module &m,
                                                    const string &name);
extern template void bind_fl_sparse<SU2, float>(py::module &m);
extern template void bind_fl_parallel<SU2, float>(py::module &m);
extern template void bind_fl_operator<SU2, float>(py::module &m);
extern template void bind_fl_hamiltonian<SU2, float>(py::module &m);
extern template void bind_fl_rule<SU2, float>(py::module &m);

extern template void
bind_trans_sparse_matrix<SU2, float, double>(py::module &m,
                                             const string &aux_name);
extern template void
bind_trans_sparse_matrix<SU2, double, float>(py::module &m,
                                             const string &aux_name);
extern template void
bind_trans_sparse_matrix<SZ, float, double>(py::module &m,
                                            const string &aux_name);
extern template void
bind_trans_sparse_matrix<SZ, double, float>(py::module &m,
                                            const string &aux_name);
#endif

#ifdef _USE_COMPLEX

extern template void bind_fl_matrix<complex<float>>(py::module &m);
extern template void bind_general_fcidump<complex<float>>(py::module &m);

#ifdef _USE_SU2SZ
extern template void bind_fl_expr<SZ, complex<float>>(py::module &m);
extern template void bind_fl_state_info<SZ, complex<float>>(py::module &m,
                                                            const string &name);
extern template void bind_fl_sparse<SZ, complex<float>>(py::module &m);
extern template void bind_fl_parallel<SZ, complex<float>>(py::module &m);
extern template void bind_fl_operator<SZ, complex<float>>(py::module &m);
extern template void bind_fl_hamiltonian<SZ, complex<float>>(py::module &m);
extern template void bind_fl_rule<SZ, complex<float>>(py::module &m);

extern template void bind_fl_expr<SU2, complex<float>>(py::module &m);
extern template void
bind_fl_state_info<SU2, complex<float>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SU2, complex<float>>(py::module &m);
extern template void bind_fl_parallel<SU2, complex<float>>(py::module &m);
extern template void bind_fl_operator<SU2, complex<float>>(py::module &m);
extern template void bind_fl_hamiltonian<SU2, complex<float>>(py::module &m);
extern template void bind_fl_rule<SU2, complex<float>>(py::module &m);

extern template void
bind_trans_sparse_matrix<SU2, complex<float>, complex<double>>(
    py::module &m, const string &aux_name);
extern template void
bind_trans_sparse_matrix<SU2, complex<double>, complex<float>>(
    py::module &m, const string &aux_name);
extern template void
bind_trans_sparse_matrix<SZ, complex<float>, complex<double>>(
    py::module &m, const string &aux_name);
extern template void
bind_trans_sparse_matrix<SZ, complex<double>, complex<float>>(
    py::module &m, const string &aux_name);
#endif

#endif

#ifdef _USE_SG

extern template void bind_fl_expr<SGF, float>(py::module &m);
extern template void bind_fl_state_info<SGF, float>(py::module &m,
                                                    const string &name);
extern template void bind_fl_sparse<SGF, float>(py::module &m);
extern template void bind_fl_parallel<SGF, float>(py::module &m);
extern template void bind_fl_operator<SGF, float>(py::module &m);
extern template void bind_fl_hamiltonian<SGF, float>(py::module &m);
extern template void bind_fl_rule<SGF, float>(py::module &m);

extern template void bind_fl_expr<SGB, float>(py::module &m);
extern template void bind_fl_state_info<SGB, float>(py::module &m,
                                                    const string &name);
extern template void bind_fl_sparse<SGB, float>(py::module &m);
extern template void bind_fl_parallel<SGB, float>(py::module &m);
extern template void bind_fl_operator<SGB, float>(py::module &m);
extern template void bind_fl_hamiltonian<SGB, float>(py::module &m);
extern template void bind_fl_rule<SGB, float>(py::module &m);

extern template void
bind_trans_sparse_matrix<SGF, float, double>(py::module &m,
                                             const string &aux_name);
extern template void
bind_trans_sparse_matrix<SGF, double, float>(py::module &m,
                                             const string &aux_name);
extern template void
bind_trans_sparse_matrix<SGB, float, double>(py::module &m,
                                             const string &aux_name);
extern template void
bind_trans_sparse_matrix<SGB, double, float>(py::module &m,
                                             const string &aux_name);

#ifdef _USE_COMPLEX
extern template void bind_fl_expr<SGF, complex<float>>(py::module &m);
extern template void
bind_fl_state_info<SGF, complex<float>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SGF, complex<float>>(py::module &m);
extern template void bind_fl_parallel<SGF, complex<float>>(py::module &m);
extern template void bind_fl_operator<SGF, complex<float>>(py::module &m);
extern template void bind_fl_hamiltonian<SGF, complex<float>>(py::module &m);
extern template void bind_fl_rule<SGF, complex<float>>(py::module &m);

extern template void bind_fl_expr<SGB, complex<float>>(py::module &m);
extern template void
bind_fl_state_info<SGB, complex<float>>(py::module &m, const string &name);
extern template void bind_fl_sparse<SGB, complex<float>>(py::module &m);
extern template void bind_fl_parallel<SGB, complex<float>>(py::module &m);
extern template void bind_fl_operator<SGB, complex<float>>(py::module &m);
extern template void bind_fl_hamiltonian<SGB, complex<float>>(py::module &m);
extern template void bind_fl_rule<SGB, complex<float>>(py::module &m);

extern template void
bind_trans_sparse_matrix<SGF, complex<float>, complex<double>>(
    py::module &m, const string &aux_name);
extern template void
bind_trans_sparse_matrix<SGF, complex<double>, complex<float>>(
    py::module &m, const string &aux_name);
extern template void
bind_trans_sparse_matrix<SGB, complex<float>, complex<double>>(
    py::module &m, const string &aux_name);
extern template void
bind_trans_sparse_matrix<SGB, complex<double>, complex<float>>(
    py::module &m, const string &aux_name);

#endif

#endif

#endif

#endif
