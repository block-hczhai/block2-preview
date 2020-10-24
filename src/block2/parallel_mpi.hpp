
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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

#ifdef _HAS_MPI

#include "allocator.hpp"
#include "csr_operator_functions.hpp"
#include "csr_sparse_matrix.hpp"
#include "mpi.h"
#include "parallel_rule.hpp"
#include "sparse_matrix.hpp"
#include <chrono>
#include <ios>
#include <memory>
#include <thread>

using namespace std;

namespace block2 {

struct MPI {
    int _ierr, _rank, _size;
    MPI() {
        int flag = 1;
        _ierr = MPI_Initialized(&flag);
        if (!flag) {
            _ierr = MPI_Init(nullptr, nullptr);
            assert(_ierr == 0);
        }
        _ierr = MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        assert(_ierr == 0);
        _ierr = MPI_Comm_size(MPI_COMM_WORLD, &_size);
        assert(_ierr == 0);
        // Try to guard parallel print statement with barrier and sleep
        _ierr = MPI_Barrier(MPI_COMM_WORLD);
        assert(_ierr == 0);
        cout << "MPI INIT: rank " << _rank << " of " << _size << endl;
        std::this_thread::sleep_for(chrono::milliseconds(4));
        _ierr = MPI_Barrier(MPI_COMM_WORLD);
        assert(_ierr == 0);
        if (_rank != 0)
            cout.setstate(ios::failbit);
    }
    ~MPI() {
        cout.clear();
        cout << "MPI FINALIZE: rank " << _rank << " of " << _size << endl;
        MPI_Finalize();
    }
    static MPI &mpi() {
        static MPI _mpi;
        return _mpi;
    }
    static int rank() { return mpi()._rank; }
    static int size() { return mpi()._size; }
};

template <typename S> struct MPICommunicator : ParallelCommunicator<S> {
    using ParallelCommunicator<S>::size;
    using ParallelCommunicator<S>::rank;
    using ParallelCommunicator<S>::root;
    using ParallelCommunicator<S>::tcomm;
    Timer _t;
    MPICommunicator(int root = 0)
        : ParallelCommunicator<S>(MPI::size(), MPI::rank(), root) {}
    ParallelTypes get_parallel_type() const override {
        return ParallelTypes::Distributed;
    }
    void barrier() override {
        _t.get_time();
        int ierr = MPI_Barrier(MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void broadcast(double *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Bcast(data, len, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void broadcast(int *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Bcast(data, len, MPI_INT, owner, MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void broadcast(const shared_ptr<SparseMatrix<S>> &mat, int owner) override {
        if (mat->get_type() == SparseMatrixTypes::Normal)
            broadcast(mat->data, mat->total_memory, owner);
        else if (mat->get_type() == SparseMatrixTypes::CSR) {
            _t.get_time();
            // remove mkl pointer
            // csr sparse matrix cannot be allocated by mkl sparse matrix
            // for mkl sparse matrix the memory for CSRMatrixRef is not
            // contiguous based on current impl this can never be the case
            // unless this matrix is formed by iadd
            shared_ptr<CSRSparseMatrix<S>> cmat =
                make_shared<CSRSparseMatrix<S>>();
            cmat->copy_data_from(mat);
            mat->deallocate();
            *dynamic_pointer_cast<CSRSparseMatrix<S>>(mat) = *cmat;
            vector<int> nnzs(mat->info->n);
            for (int i = 0; i < mat->info->n; i++)
                nnzs[i] = cmat->csr_data[i]->nnz;
            int ierr = MPI_Bcast(nnzs.data(), mat->total_memory, MPI_INT, owner,
                                 MPI_COMM_WORLD);
            assert(ierr == 0);
            size_t dsize = 0, dp = 0;
            for (int i = 0; i < mat->info->n; i++) {
                if (cmat->csr_data[i]->nnz != nnzs[i]) {
                    assert(rank != owner);
                    assert(cmat->csr_data[i]->alloc != nullptr);
                    cmat->csr_data[i]->deallocate();
                    cmat->csr_data[i]->alloc =
                        make_shared<VectorAllocator<double>>();
                    cmat->csr_data[i]->nnz = nnzs[i];
                    cmat->csr_data[i]->allocate();
                    dsize += cmat->csr_data[i]->memory_size();
                }
            }
            vector<double> dt(dsize);
            if (rank == owner)
                for (int i = 0; i < mat->info->n; i++) {
                    memcpy(dt.data() + dp, cmat->csr_data[i]->data,
                           sizeof(double) * cmat->csr_data[i]->memory_size());
                    dp += cmat->csr_data[i]->memory_size();
                }
            ierr = MPI_Bcast(dt.data(), dt.size(), MPI_DOUBLE, owner,
                             MPI_COMM_WORLD);
            assert(ierr == 0);
            dp = 0;
            if (rank != owner) {
                for (int i = 0; i < mat->info->n; i++) {
                    memcpy(cmat->csr_data[i]->data, dt.data() + dp,
                           sizeof(double) * cmat->csr_data[i]->memory_size());
                    dp += cmat->csr_data[i]->memory_size();
                }
            }
            tcomm += _t.get_time();
        } else
            assert(false);
    }
    void allreduce_sum(double *data, size_t len) override {
        _t.get_time();
        int ierr = MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_DOUBLE, MPI_SUM,
                                 MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void allreduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat) override {
        allreduce_sum(mat->data, mat->total_memory);
    }
    void allreduce_sum(const shared_ptr<SparseMatrix<S>> &mat) override {
        assert(mat->get_type() == SparseMatrixTypes::Normal);
        allreduce_sum(mat->data, mat->total_memory);
    }
    void allreduce_sum(vector<S> &vs) override {
        _t.get_time();
        uint32_t sz = (uint32_t)vs.size(), maxsz;
        int ierr = MPI_Allreduce(&sz, &maxsz, 1, MPI_UINT32_T, MPI_MAX,
                                 MPI_COMM_WORLD);
        assert(ierr == 0);
        vector<S> vsrecv(maxsz * size);
        vs.resize(maxsz, S(S::invalid));
        ierr = MPI_Allgather(vs.data(), maxsz, MPI_UINT32_T, vsrecv.data(),
                             maxsz, MPI_UINT32_T, MPI_COMM_WORLD);
        assert(ierr == 0);
        vsrecv.resize(
            distance(vsrecv.begin(),
                     remove(vsrecv.begin(), vsrecv.end(), S(S::invalid))));
        vs = vsrecv;
        tcomm += _t.get_time();
    }
    void allreduce_logical_or(bool &v) override {
        _t.get_time();
        int ierr = MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_C_BOOL, MPI_LOR,
                                 MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void reduce_sum(double *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Reduce(rank == owner ? MPI_IN_PLACE : data, data, len,
                              MPI_DOUBLE, MPI_SUM, owner, MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void reduce_sum(uint64_t *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Reduce(rank == owner ? MPI_IN_PLACE : data, data, len,
                              MPI_UINT64_T, MPI_SUM, owner, MPI_COMM_WORLD);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void reduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat,
                    int owner) override {
        return reduce_sum(mat->data, mat->total_memory, owner);
    }
    void reduce_sum(const shared_ptr<SparseMatrix<S>> &mat,
                    int owner) override {
        if (mat->get_type() == SparseMatrixTypes::Normal)
            return reduce_sum(mat->data, mat->total_memory, owner);
        else if (mat->get_type() == SparseMatrixTypes::CSR) {
            _t.get_time();
            // remove mkl pointer
            shared_ptr<CSRSparseMatrix<S>> cmat =
                make_shared<CSRSparseMatrix<S>>();
            cmat->copy_data_from(mat);
            mat->deallocate();
            *dynamic_pointer_cast<CSRSparseMatrix<S>>(mat) = *cmat;
            shared_ptr<CSROperatorFunctions<S>> copf =
                make_shared<CSROperatorFunctions<S>>(nullptr);
            vector<int> nnzs(mat->info->n), dz, gnnzs;
            vector<double> dt;
            if (rank == owner)
                gnnzs.resize(mat->info->n * size);
            for (int i = 0; i < mat->info->n; i++)
                nnzs[i] = cmat->csr_data[i]->nnz;
            int ierr =
                MPI_Gather(nnzs.data(), mat->info->n, MPI_INT, gnnzs.data(),
                           mat->info->n, MPI_INT, owner, MPI_COMM_WORLD);
            assert(ierr == 0);
            if (rank == owner) {
                dz.resize(size, 0);
                shared_ptr<CSRSparseMatrix<S>> tmp =
                    make_shared<CSRSparseMatrix<S>>();
                tmp->allocate(mat->info);
                for (int i = 0; i < mat->info->n; i++)
                    tmp->deallocate();
                for (int k = 0, dp; k < size; k++) {
                    if (rank == owner)
                        continue;
                    for (int i = 0; i < mat->info->n; i++) {
                        tmp->csr_data[i]->nnz = gnnzs[k * mat->info->n + i];
                        dz[k] += tmp->csr_data[i]->memory_size();
                    }
                    dt.resize(dz[k]);
                    ierr = MPI_Recv(dt.data(), dz[k], MPI_DOUBLE, k, 11,
                                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    assert(ierr == 0);
                    dp = 0;
                    for (int i = 0, dp = 0; i < mat->info->n; i++) {
                        tmp->csr_data[i]->allocate(dt.data() + dp);
                        dp += tmp->csr_data[i]->memory_size();
                    }
                    assert(dp == dz[k]);
                    copf->iadd(cmat, tmp, 1.0, false);
                }
                // remove mkl pointer
                CSRSparseMatrix<S> r;
                r.copy_data_from(cmat);
                cmat->deallocate();
                *dynamic_pointer_cast<CSRSparseMatrix<S>>(mat) = r;
            } else {
                int dsz = 0;
                for (int i = 0; i < mat->info->n; i++)
                    dsz += cmat->csr_data[i]->memory_size();
                dt.resize(dsz);
                int dp = 0;
                for (int i = 0; i < mat->info->n; i++) {
                    memcpy(dt.data() + dp, cmat->csr_data[i]->data,
                           sizeof(double) * cmat->csr_data[i]->memory_size());
                    dp += cmat->csr_data[i]->memory_size();
                }
                assert(dp == dsz);
                ierr = MPI_Send(dt.data(), dsz, MPI_DOUBLE, owner, 11,
                                MPI_COMM_WORLD);
                assert(ierr == 0);
            }
            tcomm += _t.get_time();
        }
    }
};

} // namespace block2

#endif
