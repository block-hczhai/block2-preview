
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

#ifdef _OPENMP
#include "omp.h"
#endif
#ifdef _HAS_INTEL_MKL
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

enum struct ThreadingTypes : uint8_t {
    SequentialGEMM = 0,          // 0: seq mkl
    BatchedGEMM = 1,             // 1: para mkl
    Quanta = 2,                  // 1: openmp quanta + seq mkl
    QuantaBatchedGEMM = 2 | 1,   // 2: openmp quanta + para mkl
    Operator = 4,                // 1: openmp operator
    OperatorBatchedGEMM = 4 | 1, // 2: openmp operator + para mkl
    OperatorQuanta = 4 | 2,      // 2: openmp operator + openmp quanta
    OperatorQuantaBatchedGEMM =
        4 | 2 | 1, // 3: openmp operator + openmp quanta + para mkl
    Global = 8
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

// Method of DGEMM parallelism
// None:   DGEMM are not parallelized
//         (but parallelism may happen inside each DGEMM)
// Simple: DGEMM are completely parallelized
//         (each DGEMM should write output to different memory)
// Auto:   DGEMM automatically divided into several batches
//         (conflicts of output are automatically resolved by
//         introducing temporary arrays)
// Tasked: DGEMM are collected and then performed in separate
//         threads, the output will then be merged
//         only useful for tensor_product_multiply
enum struct SeqTypes : uint8_t {
    None = 0,
    Simple = 1,
    Auto = 2,
    Tasked = 4,
    SimpleTasked = 5
};

inline bool operator&(SeqTypes a, SeqTypes b) {
    return ((uint8_t)a & (uint8_t)b) != 0;
}

inline SeqTypes operator|(SeqTypes a, SeqTypes b) {
    return SeqTypes((uint8_t)a | (uint8_t)b);
}

struct Threading {
    ThreadingTypes type;
    SeqTypes seq_type = SeqTypes::None;
    int n_threads_op = 0, n_threads_quanta = 0, n_threads_mkl = 0,
        n_threads_global = 0, n_levels = 0;
    bool openmp_available() const {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }
    bool tbb_available() const {
#ifdef _HAS_TBB
        return true;
#else
        return false;
#endif
    }
    bool mkl_available() const {
#ifdef _HAS_INTEL_MKL
        return true;
#else
        return false;
#endif
    }
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
    int get_thread_id() const {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
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
    int activate_normal() const { return activate_operator(); }
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
    int activate_quanta() const {
#ifdef _OPENMP
        omp_set_num_threads(n_threads_quanta != 0 ? n_threads_quanta : 1);
        return n_threads_quanta != 0 ? n_threads_quanta : 1;
#else
        return 1;
#endif
    }
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

inline shared_ptr<Threading> &threading_() {
    static shared_ptr<Threading> threading = make_shared<Threading>();
    return threading;
}

#define threading (threading_())

} // namespace block2
