
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

#include "core/complex_matrix_functions.hpp"

using namespace std;
using namespace block2;

int main(int argc, char *argv[]) {
    int memory = 1e8;
    string scratch("/tmp/block2");
    frame_() = make_shared<DataFrame>((size_t)(0.1 * memory),
                              (size_t)(0.9 * memory), scratch);

    cout << "hello" << endl;
    using cmplx = complex<double>;
    int N = 4;
    int S = 3;
    ComplexMatrixRef A(nullptr, N, N);
    ComplexDiagonalMatrix Adiag(nullptr, N);
    ComplexMatrixRef P(nullptr, N, N);
    ComplexMatrixRef x(nullptr, N, 1);
    ComplexMatrixRef b(nullptr, N, 1);
    A.allocate();
    Adiag.allocate();
    P.allocate();
    x.allocate();
    b.allocate();
    for(int i = 0; i < N; ++i){
        x(i,0) = cmplx(i+18.,i-12.);
        b(i,0) = cmplx(i*i*1.,i+2.);
        for(int j = 0; j < N; ++j){
            A(i,j) = cmplx(1./(i+j+1.), .3/(abs(i-j)+3));
            if(i==j){
                Adiag(i,i) = A(i,i);
            }
        }
    }
    auto op = [&A, N](const ComplexMatrixRef& in, const ComplexMatrixRef& out){
        for(int i = 0; i < N; ++i){
            out(i,0) = 0.;
            for(int j = 0; j < N; ++j){
                out(i,0) += A(i,j) * in(j,0);
            }
        }
    };
    auto rop = [&A, N](const ComplexMatrixRef& in, const ComplexMatrixRef& out){
        for(int i = 0; i < N; ++i){
            out(i,0) = 0.;
            for(int j = 0; j < N; ++j){
                out(i,0) += conj(A(j,i)) * in(j,0);
            }
        }
    };
    int nmult, niter;
    struct pc{
        int root;
        void broadcast(double*, int, int){};
        void broadcast(complex<double>*, int, int){};
    };
    pc* comm = nullptr;
    auto f = ComplexMatrixFunctions::idrs(op,Adiag, x, b, nmult, niter,
                                 S, true, comm,
                                 1e-8, 1e-9, 0., 8,-1,
                                 vector<ComplexMatrixRef>{}, vector<complex<double>>{1.,2,3.});
    cout << x << endl;
    cout << f << endl;
    cout << "LSQR" << endl;
    for(int i = 0; i < N; ++i) {
        x(i, 0) = cmplx(i + 18., i - 12.);
    }
    auto g = ComplexMatrixFunctions::lsqr(op,rop, Adiag,x, b, nmult, niter,
                                 true, comm, 1e-8,
                                 1e-8, 0., 108,18);
    cout << x << endl;
    cout << g << endl;


    return 0;
}
