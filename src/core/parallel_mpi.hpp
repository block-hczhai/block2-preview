
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
    using ParallelCommunicator<S>::group;
    using ParallelCommunicator<S>::para_type;
    using ParallelCommunicator<S>::tcomm;
    using ParallelCommunicator<S>::tidle;
    using ParallelCommunicator<S>::twait;
    Timer _t;
    const size_t chunk_size = 1 << 30;
    vector<MPI_Request> reqs;
    MPI_Comm comm;
    MPICommunicator(int root = 0)
        : ParallelCommunicator<S>(MPI::size(), MPI::rank(), root) {
        para_type = ParallelTypes::Distributed;
        comm = MPI_COMM_WORLD;
    }
    MPICommunicator(MPI_Comm comm, int size, int rank, int root = 0)
        : ParallelCommunicator<S>(size, rank, root), comm(comm) {
        para_type = ParallelTypes::Distributed;
    }
    ~MPICommunicator() override {
        if (comm != MPI_COMM_WORLD && comm != MPI_COMM_NULL) {
            int ierr = MPI_Comm_free(&comm);
            assert(ierr == 0);
        }
    }
    shared_ptr<ParallelCommunicator<S>> split(int igroup, int irank) override {
        MPI_Comm icomm;
        int jrank, isize, ierr;
        ierr = MPI_Comm_split(comm, igroup == -1 ? MPI_UNDEFINED : igroup,
                              irank, &icomm);
        assert(ierr == 0);
        if (igroup != -1) {
            ierr = MPI_Comm_rank(icomm, &jrank);
            assert(ierr == 0);
            ierr = MPI_Comm_size(icomm, &isize);
            assert(ierr == 0);
        } else
            isize = irank = -1;
        return make_shared<MPICommunicator<S>>(icomm, isize, jrank);
    }
    void barrier() override {
        if (comm == MPI_COMM_NULL)
            return;
        _t.get_time();
        int ierr = MPI_Barrier(comm);
        assert(ierr == 0);
        tidle += _t.get_time();
    }
    void broadcast(double *data, size_t len, int owner) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Bcast(data + offset, min(chunk_size, len - offset),
                                 MPI_DOUBLE, owner, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void broadcast(complex<double> *data, size_t len, int owner) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Bcast((double *)(data + offset),
                                 min(chunk_size, len - offset) * 2, MPI_DOUBLE,
                                 owner, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void ibroadcast(double *data, size_t len, int owner) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            MPI_Request req;
            int ierr = MPI_Ibcast(data + offset, min(chunk_size, len - offset),
                                  MPI_DOUBLE, owner, comm, &req);
            assert(ierr == 0);
            reqs.push_back(req);
        }
        tcomm += _t.get_time();
    }
    void broadcast(int *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Bcast(data, len, MPI_INT, owner, comm);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void broadcast(long long int *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Bcast(data, len, MPI_LONG_LONG, owner, comm);
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
            int ierr =
                MPI_Bcast(nnzs.data(), mat->total_memory, MPI_INT, owner, comm);
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
            ierr = MPI_Bcast(dt.data(), dt.size(), MPI_DOUBLE, owner, comm);
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
    void ibroadcast(const shared_ptr<SparseMatrix<S>> &mat,
                    int owner) override {
        if (mat->get_type() == SparseMatrixTypes::Normal) {
            ibroadcast(mat->data, mat->total_memory, owner);
        } else
            assert(false);
    }
    void allreduce_sum(double *data, size_t len) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Allreduce(MPI_IN_PLACE, data + offset,
                                     min(chunk_size, len - offset), MPI_DOUBLE,
                                     MPI_SUM, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void allreduce_sum(complex<double> *data, size_t len) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Allreduce(MPI_IN_PLACE, (double *)(data + offset),
                                     min(chunk_size, len - offset) * 2,
                                     MPI_DOUBLE, MPI_SUM, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void allreduce_max(double *data, size_t len) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Allreduce(MPI_IN_PLACE, data + offset,
                                     min(chunk_size, len - offset), MPI_DOUBLE,
                                     MPI_MAX, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void allreduce_max(vector<double> &vs) override {
        allreduce_max(vs.data(), vs.size());
    }
    void allreduce_min(double *data, size_t len) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Allreduce(MPI_IN_PLACE, data + offset,
                                     min(chunk_size, len - offset), MPI_DOUBLE,
                                     MPI_MIN, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void allreduce_min(vector<double> &vs) override {
        allreduce_min(vs.data(), vs.size());
    }
    void allreduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat) override {
        allreduce_sum(mat->data, mat->total_memory);
    }
    void allreduce_sum(const shared_ptr<SparseMatrix<S>> &mat) override {
        assert(mat->get_type() == SparseMatrixTypes::Normal);
        allreduce_sum(mat->data, mat->total_memory);
    }
    void allreduce_min(vector<vector<double>> &vs) override {
        vector<double> vx;
        for (size_t i = 0; i < vs.size(); i++)
            vx.insert(vx.end(), vs[i].begin(), vs[i].end());
        allreduce_min(vx.data(), vx.size());
        for (size_t i = 0, j = 0; i < vs.size(); i++) {
            memcpy(vs[i].data(), vx.data() + j, vs[i].size() * sizeof(double));
            j += vs[i].size();
        }
    }
    void allreduce_sum(vector<S> &vs) override {
        _t.get_time();
        uint32_t sz = (uint32_t)vs.size(), maxsz;
        int ierr = MPI_Allreduce(&sz, &maxsz, 1, MPI_UINT32_T, MPI_MAX, comm);
        assert(ierr == 0);
        vector<S> vsrecv(maxsz * size);
        vs.resize(maxsz, S(S::invalid));
        ierr = MPI_Allgather(vs.data(), maxsz, MPI_UINT32_T, vsrecv.data(),
                             maxsz, MPI_UINT32_T, comm);
        assert(ierr == 0);
        vsrecv.resize(
            distance(vsrecv.begin(),
                     remove(vsrecv.begin(), vsrecv.end(), S(S::invalid))));
        vs = vsrecv;
        tcomm += _t.get_time();
    }
    void allreduce_logical_or(bool &v) override {
        _t.get_time();
        int ierr =
            MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_C_BOOL, MPI_LOR, comm);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void allreduce_logical_or(char *data, size_t len) override {
        _t.get_time();
        int ierr =
            MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_CHAR, MPI_LOR, comm);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void allreduce_xor(char *data, size_t len) override {
        _t.get_time();
        int ierr =
            MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_CHAR, MPI_BXOR, comm);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void reduce_sum(double *data, size_t len, int owner) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            int ierr = MPI_Reduce(rank == owner ? MPI_IN_PLACE : data + offset,
                                  data + offset, min(chunk_size, len - offset),
                                  MPI_DOUBLE, MPI_SUM, owner, comm);
            assert(ierr == 0);
        }
        tcomm += _t.get_time();
    }
    void ireduce_sum(double *data, size_t len, int owner) override {
        _t.get_time();
        for (size_t offset = 0; offset < len; offset += chunk_size) {
            MPI_Request req;
            int ierr = MPI_Ireduce(rank == owner ? MPI_IN_PLACE : data + offset,
                                   data + offset, min(chunk_size, len - offset),
                                   MPI_DOUBLE, MPI_SUM, owner, comm, &req);
            assert(ierr == 0);
            reqs.push_back(req);
        }
        tcomm += _t.get_time();
    }
    void reduce_sum(uint64_t *data, size_t len, int owner) override {
        _t.get_time();
        int ierr = MPI_Reduce(rank == owner ? MPI_IN_PLACE : data, data, len,
                              MPI_UINT64_T, MPI_SUM, owner, comm);
        assert(ierr == 0);
        tcomm += _t.get_time();
    }
    void reduce_sum(const shared_ptr<SparseMatrixGroup<S>> &mat,
                    int owner) override {
        return reduce_sum(mat->data, mat->total_memory, owner);
    }
    void ireduce_sum(const shared_ptr<SparseMatrix<S>> &mat,
                     int owner) override {
        return ireduce_sum(mat->data, mat->total_memory, owner);
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
            vector<MKL_INT> nnzs(mat->info->n), dz, gnnzs;
            vector<double> dt;
            if (rank == owner)
                gnnzs.resize(mat->info->n * size);
            for (int i = 0; i < mat->info->n; i++)
                nnzs[i] = cmat->csr_data[i]->nnz;
            int ierr =
                MPI_Gather(nnzs.data(), mat->info->n, MPI_INT, gnnzs.data(),
                           mat->info->n, MPI_INT, owner, comm);
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
                    ierr = MPI_Recv(dt.data(), dz[k], MPI_DOUBLE, k, 11, comm,
                                    MPI_STATUS_IGNORE);
                    assert(ierr == 0);
                    dp = 0;
                    for (int i = 0; i < mat->info->n; i++) {
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
                MKL_INT dsz = 0;
                for (int i = 0; i < mat->info->n; i++)
                    dsz += cmat->csr_data[i]->memory_size();
                dt.resize(dsz);
                MKL_INT dp = 0;
                for (int i = 0; i < mat->info->n; i++) {
                    memcpy(dt.data() + dp, cmat->csr_data[i]->data,
                           sizeof(double) * cmat->csr_data[i]->memory_size());
                    dp += cmat->csr_data[i]->memory_size();
                }
                assert(dp == dsz);
                ierr = MPI_Send(dt.data(), dsz, MPI_DOUBLE, owner, 11, comm);
                assert(ierr == 0);
            }
            tcomm += _t.get_time();
        }
    }
    void waitall() override {
        _t.get_time();
        int ierr =
            MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        assert(ierr == 0);
        twait += _t.get_time();
    }
};

} // namespace block2

#endif
