
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

/** Global information for shared-memory parallelism and threading schemes. */

#pragma once

#ifdef _OPENMP
#include "omp.h"
#endif
#ifdef _HAS_INTEL_MKL
#ifndef MKL_Complex16
#include <complex>
#define MKL_Complex16 std::complex<double>
#endif
#include "mkl.h"
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

#ifndef _HAS_INTEL_MKL
#ifdef MKL_ILP64
#define MKL_INT long long int
#else
#define MKL_INT int
#endif
#endif

namespace block2 {

/**
 * An indicator for where the openMP shared-memory
 * threading should be activated. In the case of nested openMP,
 * the total number of nested threading layers is determined from
 * this enumeration.
 *
 * For each enumerator, the number in brackets is the total number
 * of threading layers.
 */
enum struct ThreadingTypes : uint8_t {
    SequentialGEMM = 0,          //!< [0] seq mkl
    BatchedGEMM = 1,             //!< [1] parallel mkl
    Quanta = 2,                  //!< [1] openmp quanta + seq mkl
    QuantaBatchedGEMM = 2 | 1,   //!< [2] openmp quanta + parallel mkl
    Operator = 4,                //!< [1] openmp operator
    OperatorBatchedGEMM = 4 | 1, //!< [2] openmp operator + parallel mkl
    OperatorQuanta = 4 | 2,      //!< [2] openmp operator + openmp quanta
    OperatorQuantaBatchedGEMM =
        4 | 2 | 1, //!< [3] openmp operator + openmp quanta + parallel mkl
    Global = 8     //!< [1] openmp for general non-core-algorithm tasks
};

inline bool operator&(ThreadingTypes a, ThreadingTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline ThreadingTypes operator|(ThreadingTypes a, ThreadingTypes b) {
    return ThreadingTypes((uint8_t)a | (uint8_t)b);
}

inline ThreadingTypes operator^(ThreadingTypes a, ThreadingTypes b) {
    return ThreadingTypes((uint8_t)a ^ (uint8_t)b);
}

/**
 * Method of GEMM (dense matrix multiplication) parallelism.
 * For CSR matrix multiplication, the only possbile case is ``SeqTypes::None``,
 * but one can still use ``SeqTypes::Simple`` and it will only parallelize
 * dense matrix multiplication.
 */
enum struct SeqTypes : uint8_t {
    None = 0,   //!< GEMM are not parallelized. Parallelism may happen inside
                //!<   each GEMM, if a threaded version of MKL is linked.
    Simple = 1, //!< GEMM written to the different outputs are parallelized,
                //!< otherwise they are executed in sequential. With this mode,
                //!< the code will sort and divide GEMM to several groups
                //!< (batches). Inside each batch, the output addresses are
                //!< guarenteed to be different. The ``cblas_dgemm_batch`` is
                //!< invoked to compute each batch.
    Auto = 2,   //!< DGEMM automatically divided into several batches only when
                //!< there are data dependency. Conflicts of output are
                //!< automatically resolved by introducing temporary arrays. The
                //!< ``cblas_dgemm_batch`` is invoked to compute each batch.
                //!< This option normally requires a large amount of time for
                //!< preprocessing and it will introduce a large number of
                //!< temporary arrays, which is not memory friendly.
    Tasked = 4, //!< GEMM will be evenly divided into ``n_threads`` groups,
                //!< Different groups are executed in different threads.
                //!< Since different threads may write into the same output
                //!< array, there is an additional reduction step after all
                //!< GEMM finishes. This mode is mainly implemented for
                //!< Davidson matrix-vector step (``tensor_product_multiply``),
                //!< where the size of the output array (wavefunction) is small
                //!< compared to that of all input arrays. For blocking/rotation
                //!< step, ``SeqTypes::Tasked`` has no effect and it
                //!< is equivalent to ``SeqTypes::None``.
                //!< The ``cblas_dgemm_batch`` is not used in this mode.
    SimpleTasked = 5 //!< This is the same as ``SeqTypes::Tasked`` for
                     //!< the Davidson matrix-vector step, and the same as
                     //!< ``SeqTypes::Simple`` for other steps.
};

inline bool operator&(SeqTypes a, SeqTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline SeqTypes operator|(SeqTypes a, SeqTypes b) {
    return SeqTypes((uint8_t)a | (uint8_t)b);
}

/**
 * Global information for threading schemes.
 */
struct Threading {
    ThreadingTypes type;                //!< Type of the threading scheme.
    SeqTypes seq_type = SeqTypes::None; //!< Method of dense matrix
                                        //!< multiplication parallelism.
    int n_threads_op = 0,     //!< Number of threads for parallelism over
                              //!< renormalized operators.
        n_threads_quanta = 0, //!< Number of threads for parallelism over
                              //!< symmetry sectors.
        n_threads_mkl = 0,    //!< Number of threads for parallelism within
                              //!< dense matrix multiplications.
        n_threads_global = 0, //!< Number of threads for general tasks
        n_levels = 0;         //!< Number of nested threading layers
    /** Whether openmp compiler option is set. */
    bool openmp_available() const {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }
    /** Whether tbb memory allocator is used. */
    bool tbb_available() const {
#ifdef _HAS_TBB
        return true;
#else
        return false;
#endif
    }
    /** Whether MKL math library is used. */
    bool mkl_available() const {
#ifdef _HAS_INTEL_MKL
        return true;
#else
        return false;
#endif
    }
    /** Check version of the linked MKL library.
     * @return A version string of the linked MKL library if MKL
     *   is linked, or an empty string otherwise.
     */
    string get_mkl_version() const {
#ifdef _HAS_INTEL_MKL
        MKLVersion ver;
        mkl_get_version(&ver);
        stringstream ss;
        ss << ver.MajorVersion << "." << ver.MinorVersion << "."
           << ver.UpdateVersion;
        return ss.str();
#else
        return "";
#endif
    }
    /** Return a string indicating which threaded MKL library
     * is linked.
     */
    string get_mkl_threading_type() const {
#ifdef _HAS_INTEL_MKL
#if _HAS_INTEL_MKL == 0
        return "SEQ";
#elif _HAS_INTEL_MKL == 1
        return "GNU";
#elif _HAS_INTEL_MKL == 2
        return "INTEL";
#elif _HAS_INTEL_MKL == 3
        return "TBB";
#else
        return "???";
#endif
#else
        return "NONE";
#endif
    }
    /** Return a string indicating which ``SeqTypes`` is used. */
    string get_seq_type() const {
        if (seq_type == SeqTypes::Auto)
            return "Auto";
        else if (seq_type == SeqTypes::Tasked)
            return "Tasked";
        else if (seq_type == SeqTypes::SimpleTasked)
            return "SimpleTasked";
        else if (seq_type == SeqTypes::Simple)
            return "Simple";
        else if (seq_type == SeqTypes::None)
            return "None";
        else
            return "???";
    }
    /** If inside a openMP parallel region, return the id
     * of the current thread. */
    int get_thread_id() const {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
    /** Set number of threads for a general task.
     * Parallelism inside MKL will be deactivated for a general task.
     * @return Number of threads for general tasks.
     *   Returns 1 if openMP should not be used for a general task. */
    int activate_global() const {
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(1);
#endif
#ifdef _OPENMP
        omp_set_num_threads(n_threads_global != 0 ? n_threads_global : 1);
        return n_threads_global != 0 ? n_threads_global : 1;
#else
        return 1;
#endif
    }
    /** Set number of threads for a general task with parallelism inside MKL.
     * Parallelism outside MKL will be deactivated.
     * @return Number of threads for general tasks.
     *   Returns 1 if MKL is not supported. */
    int activate_global_mkl() const {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(n_threads_global != 0 ? n_threads_global : 1);
        return n_threads_global != 0 ? n_threads_global : 1;
#else
        return 1;
#endif
    }
    /** Set number of threads for a normal (parallelism over renormalized
     * operators) task.
     * @return Number of threads for parallelism over renormalized operators.
     */
    int activate_normal() const { return activate_operator(); }
    /** Set number of threads for parallelism over renormalized operators.
     * @return Number of threads for parallelism over renormalized operators.
     */
    int activate_operator() const {
#ifdef _HAS_INTEL_MKL
        mkl_set_num_threads(n_threads_mkl != 0 ? n_threads_mkl : 1);
#endif
#ifdef _OPENMP
        omp_set_num_threads(n_threads_op != 0 ? n_threads_op : 1);
        return n_threads_op != 0 ? n_threads_op : 1;
#else
        return 1;
#endif
    }
    /** Set number of threads for parallelism over symmetry sectors.
     * @return Number of threads for parallelism over symmetry sectors.
     */
    int activate_quanta() const {
#ifdef _OPENMP
        omp_set_num_threads(n_threads_quanta != 0 ? n_threads_quanta : 1);
        return n_threads_quanta != 0 ? n_threads_quanta : 1;
#else
        return 1;
#endif
    }
    /** Default constructor.
     * Uses ``ThreadingTypes::Global | ThreadingTypes::BatchedGEMM``
     * with maximal available number of threads, and ``SeqTypes::None``
     * for dense matrix multiplication. */
    Threading() {
        type = ThreadingTypes::SequentialGEMM;
#ifdef _OPENMP
        n_threads_global = omp_get_max_threads();
        omp_set_num_threads(n_threads_global);
        n_levels = 1;
        type = type | ThreadingTypes::Global;
#else
        n_threads_global = 0;
#endif
#ifdef _HAS_INTEL_MKL
        n_threads_mkl = mkl_get_max_threads();
        mkl_set_num_threads(n_threads_mkl);
        mkl_set_dynamic(0);
        n_levels++;
        type = type | ThreadingTypes::BatchedGEMM;
#else
        n_threads_mkl = 0;
#endif
#ifdef _OPENMP
#ifndef _MSC_VER
        if (n_levels != 0)
            omp_set_max_active_levels(n_levels);
#endif
#endif
        n_threads_op = 0;
        n_threads_quanta = 0;
    }
    /** Constructor.
     * @param type Type of the threading scheme.
     * @param nta Number of threads for a general task (if
     * ``ThreadingTypes::Global`` is set) or number of threads in the first
     * threading layer.
     * @param ntb Number of threads in the first threading layer for a
     * non-general threaded task (if ``ThreadingTypes::Global`` is set) or
     * number of threads in the second threading layer.
     * @param ntc Number of threads in the second threading layer for a
     * non-general threaded task (if ``ThreadingTypes::Global`` is set) or
     * number of threads in the third threading layer.
     * @param ntd Number of threads in the third threading layer for a
     * non-general threaded task (if ``ThreadingTypes::Global`` is set).
     */
    Threading(ThreadingTypes type, int nta = -1, int ntb = -1, int ntc = -1,
              int ntd = -1)
        : type(type) {
        if (type & ThreadingTypes::Global) {
            n_threads_global = nta;
            nta = ntb;
            ntb = ntc;
            ntc = ntd;
            type = type ^ ThreadingTypes::Global;
            n_levels = 1;
        } else
            n_levels = 0;
        switch (type) {
        case ThreadingTypes::SequentialGEMM:
            assert(nta == -1 && ntb == -1 && ntc == -1);
            break;
        case ThreadingTypes::BatchedGEMM:
            assert(ntb == -1 && ntc == -1);
            n_threads_mkl = nta;
            n_levels = max(1, n_levels);
            break;
        case ThreadingTypes::Quanta:
            assert(ntb == -1 && ntc == -1);
            n_threads_quanta = nta;
            n_levels = max(1, n_levels);
            break;
        case ThreadingTypes::QuantaBatchedGEMM:
            assert(ntc == -1);
            n_threads_quanta = nta;
            n_threads_mkl = ntb;
            n_levels = max(2, n_levels);
            break;
        case ThreadingTypes::Operator:
            assert(ntb == -1 && ntc == -1);
            n_threads_op = nta;
            n_levels = max(1, n_levels);
            break;
        case ThreadingTypes::OperatorBatchedGEMM:
            assert(ntc == -1);
            n_threads_op = nta;
            n_threads_mkl = ntb;
            n_levels = max(2, n_levels);
            break;
        case ThreadingTypes::OperatorQuanta:
            assert(ntc == -1);
            n_threads_op = nta;
            n_threads_quanta = ntb;
            n_levels = max(2, n_levels);
            break;
        case ThreadingTypes::OperatorQuantaBatchedGEMM:
            n_threads_op = nta;
            n_threads_quanta = ntb;
            n_threads_mkl = ntc;
            n_levels = max(3, n_levels);
            break;
        default:
            assert(false);
        }
        assert(n_threads_global != -1 && n_threads_op != -1 &&
               n_threads_quanta != -1 && n_threads_mkl != -1);
        if (type & ThreadingTypes::BatchedGEMM) {
#ifdef _HAS_INTEL_MKL
            mkl_set_num_threads(n_threads_mkl);
            mkl_set_dynamic(0);
#else
            throw runtime_error("cannot set number of mkl threads.");
#endif
        }
        if (type & ThreadingTypes::Operator) {
#ifdef _OPENMP
            omp_set_num_threads(n_threads_op);
#else
            if (n_threads_op != 1)
                throw runtime_error("cannot set number of omp threads.");
#endif
        } else if (type & ThreadingTypes::Quanta) {
#ifdef _OPENMP
            omp_set_num_threads(n_threads_quanta);
#else
            if (n_threads_quanta != 1)
                throw runtime_error("cannot set number of omp threads.");
#endif
        }
        if (n_threads_global != 0) {
#ifdef _OPENMP
            omp_set_num_threads(n_threads_global);
#else
            if (n_threads_global != 1)
                throw runtime_error("cannot set number of omp threads.");
#endif
        }
#ifdef _OPENMP
#ifndef _MSC_VER
        if (n_levels != 0)
            omp_set_max_active_levels(n_levels);
#endif
#endif
    }
    /** Print threading information. */
    friend ostream &operator<<(ostream &os, const Threading &th) {
        os << " OpenMP = " << th.openmp_available()
           << " TBB = " << th.tbb_available()
           << " MKL = " << th.get_mkl_threading_type() << " "
           << th.get_mkl_version() << " SeqType = " << th.get_seq_type()
           << " MKLIntLen = " << sizeof(MKL_INT) << endl;
        os << " THREADING = " << th.n_levels << " layers : "
           << ((th.type & ThreadingTypes::Global) ? "Global | " : "")
           << ((th.type & ThreadingTypes::Operator) ? "Operator " : "")
           << ((th.type & ThreadingTypes::Quanta) ? "Quanta " : "")
           << ((th.type & ThreadingTypes::BatchedGEMM) ? "BatchedGEMM " : "")
           << endl;
        os << " NUMBER : Global = " << th.n_threads_global
           << " Operator = " << th.n_threads_op
           << " Quanta = " << th.n_threads_quanta
           << " MKL = " << th.n_threads_mkl;
        return os;
    }
};

/** Implementation of the ``threading`` global variable. */
inline shared_ptr<Threading> &threading_() {
    static shared_ptr<Threading> threading = make_shared<Threading>();
    return threading;
}

/** Global variable containing information for shared-memory parallelism schemes
 * and number of threads used for each threading layer. */
#define threading (threading_())

} // namespace block2
