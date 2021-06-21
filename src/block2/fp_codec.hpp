
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

#include "threading.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

template <typename T> struct FPtraits {
    typedef T U;
    static const int mbits = sizeof(T) * 8;
    static const int ebits = 0;
};

template <> struct FPtraits<float> {
    typedef uint32_t U;
    static const int mbits = 23;
    static const int ebits = 8;
};

template <> struct FPtraits<double> {
    typedef uint64_t U;
    static const int mbits = 52;
    static const int ebits = 11;
};

template <typename T, typename U = typename FPtraits<T>::U,
          int mbits = FPtraits<T>::mbits, int ebits = FPtraits<T>::ebits>
inline string binary_repr(T d) {
    const U e = U(1) << mbits, s = U(1) << (mbits + ebits);
    stringstream ss;
    const U du = (U &)d;
    U duabs = du & (~s);
    if (ebits != 0) {
        const int min_ex = -(int)(U(1) << (ebits - 1)) + 1;
        ss << ((du & s) ? "- " : "+ ");
        for (int i = 0; i < ebits; i++)
            ss << (int)!!(du & (U(1) << (mbits + ebits - 1 - i)));
        ss << " ";
        int ex = (int)((duabs & (~(e - 1))) >> mbits) + min_ex;
        ss << "2^" << ex << " ";
    }
    for (int i = 0; i < mbits; i++)
        ss << (int)!!(du & (U(1) << (mbits - 1 - i)));
    return ss.str();
}

template <typename T, typename U> struct BitsCodec {
    static const int i_length = sizeof(U) * 8;
    U buf;
    T *op_data;
    size_t d_offset;
    int i_offset;
    BitsCodec(T *op_data)
        : op_data(op_data), d_offset(0), i_offset(0), buf(0) {}
    void begin_decode() {
        d_offset = 0;
        i_offset = 0;
        buf = (U &)op_data[d_offset++];
    }
    template <typename X> void encode(X x, int l) {
        buf |= (U)x << i_offset;
        if (i_offset + l >= i_length) {
            op_data[d_offset++] = (T &)buf;
            buf = (U)x >> (i_length - i_offset);
            i_offset += l - i_length;
        } else
            i_offset += l;
    }
    template <typename X> void decode(X &x, int l) {
        x = (X)((buf >> i_offset) & (((U)1 << l) - 1));
        if (i_offset + l >= i_length) {
            buf = (U &)op_data[d_offset++];
            x |= (X)((buf << (i_length - i_offset)) & (((U)1 << l) - 1));
            i_offset += l - i_length;
        } else
            i_offset += l;
    }
    size_t finish_encode() {
        op_data[d_offset++] = (T &)buf;
        return d_offset;
    }
};

template <typename T,                           // floating type to implement
          typename U = typename FPtraits<T>::U, // corresponding integer type
          int mbits = FPtraits<T>::mbits,       // number of bits in significand
          int ebits = FPtraits<T>::ebits        // number of bits in exponent
          >
struct FPCodec {
    static const int m = mbits;       // number of bits in significand
    static const U e = U(1) << mbits; // exponent least significant bit mask
    static const U s = e << ebits;    // sign bit mask
    static const U x = ~(e + s - 1);  // exponent mask
    T prec;
    U prec_u;
    mutable size_t ndata = 0, ncpsd = 0;
    size_t chunk_size = 4096;
    // number of chunks to be processed in the same batch
    size_t n_parallel_chunks = 4096;
    FPCodec() : prec(0), prec_u(0) {}
    FPCodec(T prec) : prec(prec), prec_u((U &)prec & x) {}
    FPCodec(T prec, size_t chunk_size)
        : prec(prec), prec_u((U &)prec & x), chunk_size(chunk_size) {}
    size_t decode(T *ip_data, size_t len, T *op_data) const {
        BitsCodec<T, U> enc(ip_data);
        enc.begin_decode();
        U min_u, prec_ud;
        int ldu;
        enc.decode(prec_ud, ebits);
        enc.decode(min_u, ebits);
        enc.decode(ldu, ebits);
        for (size_t i = 0; i < len; i++) {
            U uex;
            U &udata = (U &)op_data[i];
            enc.decode(udata, 1);
            udata <<= ebits + mbits;
            enc.decode(uex, ldu);
            if (uex == 0 && min_u == prec_ud)
                udata = 0;
            else {
                uex += min_u;
                udata |= uex << mbits;
                int smbits = min((int)(uex - prec_ud), mbits);
                enc.decode(uex, smbits);
                udata |= uex << (mbits - smbits);
            }
        }
        return enc.d_offset;
    }
    size_t encode(T *ip_data, size_t len, T *op_data) const {
        U max_u = 0, min_u = x, prec_ud = prec_u >> mbits;
        for (size_t i = 0; i < len; i++) {
            max_u = max(max_u, (U &)ip_data[i] & x);
            min_u = min(min_u, (U &)ip_data[i] & x);
        }
        if (min_u < prec_u)
            min_u = prec_u;
        int diff_u = (max_u - min_u) >> mbits;
        int ldu = 0;
        for (int ix = 1; diff_u >= ix; ix <<= 1, ldu++)
            ;
        BitsCodec<T, U> enc(op_data);
        enc.encode(prec_ud, ebits);
        enc.encode(min_u >> mbits, ebits);
        enc.encode(ldu, ebits);
        for (size_t i = 0; i < len; i++) {
            U udata = (U &)ip_data[i];
            U uex = udata & x;
            enc.encode(!!(udata & s), 1);
            enc.encode(uex >= min_u ? (uex - min_u) >> mbits : 0, ldu);
            if (uex <= prec_u)
                continue;
            else {
                int smbits = min((int)((uex - prec_u) >> mbits), mbits);
                enc.encode((udata & (e - 1)) >> (mbits - smbits), smbits);
            }
        }
        return enc.finish_encode();
    }
    void write_array(ostream &ofs, T *data, size_t len) const {
        const string magic = "fpc", tail = "end";
        ofs.write((char *)magic.c_str(), 4);
        ofs.write((char *)&chunk_size, sizeof(chunk_size));
        ndata += len;
        size_t nchunk = (size_t)(len / chunk_size + !!(len % chunk_size));
        size_t nbatch = (size_t)(nchunk / n_parallel_chunks +
                                 !!(nchunk % n_parallel_chunks));
        T *pdata = new T[(chunk_size + 1) * min(nchunk, n_parallel_chunks)];
        vector<size_t> cplens(n_parallel_chunks);
        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
        for (size_t ib = 0; ib < nbatch; ib++) {
            size_t n_this_chunk =
                min(nchunk - ib * n_parallel_chunks, n_parallel_chunks);
#pragma omp for schedule(static)
            for (size_t ic = 0; ic < n_this_chunk; ic++) {
                size_t offset = ic * chunk_size;
                size_t batch_offset =
                    (ic + ib * n_parallel_chunks) * chunk_size;
                size_t cklen = min(chunk_size, len - batch_offset);
                cplens[ic] =
                    encode(data + batch_offset, cklen, pdata + offset + ic);
            }
#pragma omp single
            for (size_t ic = 0; ic < n_this_chunk; ic++) {
                size_t offset = ic * chunk_size;
                size_t cplen = cplens[ic];
                ofs.write((char *)&cplen, sizeof(cplen));
                ofs.write((char *)(pdata + offset + ic), sizeof(T) * cplen);
                ncpsd += cplen;
            }
        }
        delete[] pdata;
        threading->activate_normal();
        ofs.write((char *)tail.c_str(), 4);
    }
    void read_array(istream &ifs, T *data, size_t len) const {
        string magic = "???";
        size_t chunk_size;
        ifs.read((char *)magic.c_str(), 4);
        assert(magic == "fpc");
        ifs.read((char *)&chunk_size, sizeof(chunk_size));
        size_t nchunk = (size_t)(len / chunk_size + !!(len % chunk_size));
        size_t nbatch = (size_t)(nchunk / n_parallel_chunks +
                                 !!(nchunk % n_parallel_chunks));
        T *pdata = new T[(chunk_size + 1) * min(nchunk, n_parallel_chunks)];
        vector<size_t> cplens(n_parallel_chunks);
        int ntg = threading->activate_global();
#pragma omp parallel num_threads(ntg)
        for (size_t ib = 0; ib < nbatch; ib++) {
            size_t n_this_chunk =
                min(nchunk - ib * n_parallel_chunks, n_parallel_chunks);
#pragma omp single
            for (size_t ic = 0; ic < n_this_chunk; ic++) {
                size_t &cplen = cplens[ic];
                size_t offset = ic * chunk_size;
                ifs.read((char *)&cplen, sizeof(cplen));
                assert(cplen <= chunk_size + 1);
                ifs.read((char *)(pdata + offset + ic), sizeof(T) * cplen);
            }
#pragma omp for schedule(static)
            for (size_t ic = 0; ic < n_this_chunk; ic++) {
                size_t offset = ic * chunk_size;
                size_t batch_offset =
                    (ic + ib * n_parallel_chunks) * chunk_size;
                size_t cklen = min(chunk_size, len - batch_offset);
                size_t dclen =
                    decode(pdata + offset + ic, cklen, data + batch_offset);
                assert(dclen == cplens[ic]);
            }
        }
        delete[] pdata;
        threading->activate_normal();
        ifs.read((char *)magic.c_str(), 4);
        assert(magic == "end");
    }
    void read_chunks(istream &ifs, size_t len, vector<vector<T>> &chunks,
                     size_t &chunk_size) const {
        string magic = "???";
        ifs.read((char *)magic.c_str(), 4);
        assert(magic == "fpc");
        ifs.read((char *)&chunk_size, sizeof(chunk_size));
        size_t nchunk = (size_t)(len / chunk_size + !!(len % chunk_size));
        chunks.resize(nchunk);
        for (size_t ic = 0; ic < nchunk; ic++) {
            size_t cplen;
            ifs.read((char *)&cplen, sizeof(cplen));
            assert(cplen <= chunk_size + 1);
            chunks[ic].resize(cplen);
            ifs.read((char *)chunks[ic].data(), sizeof(T) * cplen);
        }
        ifs.read((char *)magic.c_str(), 4);
        assert(magic == "end");
    }
};

template <typename T> struct CompressedVector {
    size_t arr_len, chunk_size;
    int ncache;
    mutable int icache;
    mutable vector<vector<T>> cp_data;
    mutable vector<pair<size_t, vector<T>>> cache_data;
    mutable vector<bool> cache_dirty;
    FPCodec<T> fpc;
    CompressedVector(size_t arr_len, T prec, size_t chunk_size, int ncache = 4)
        : fpc(prec), arr_len(arr_len), chunk_size(chunk_size), ncache(ncache),
          icache(0) {}
    CompressedVector(istream &ifs, size_t arr_len, T prec, int ncache = 4)
        : fpc(prec), arr_len(arr_len), ncache(ncache), icache(0) {
        fpc.read_chunks(ifs, arr_len, cp_data, chunk_size);
    }
    CompressedVector(T *arr, size_t arr_len, T prec, int ncache = 4)
        : fpc(prec), arr_len(arr_len), ncache(ncache), icache(0) {
        stringstream ss;
        ss.write((char *)arr, arr_len * sizeof(T));
        ss.clear();
        ss.seekg(0);
        fpc.read_chunks(ss, arr_len, cp_data, chunk_size);
    }
    virtual ~CompressedVector() = default;
    void clear() {
        icache = 0;
        cache_data.resize(1);
        cache_data[icache].second.resize(chunk_size);
        memset(cache_data[icache].second.data(), 0, sizeof(T) * chunk_size);
        size_t nchunk =
            (size_t)(arr_len / chunk_size + !!(arr_len % chunk_size));
        cp_data.resize(nchunk);
        for (size_t ic = 0; ic < nchunk; ic++) {
            size_t alen = min(chunk_size, arr_len - ic * chunk_size);
            cp_data[ic].resize(alen + 1);
            size_t clen = fpc.encode(cache_data[icache].second.data(), alen,
                                     cp_data[ic].data());
            cp_data[ic].resize(clen);
            cp_data[ic].shrink_to_fit();
        }
        cache_dirty.resize(1);
        cache_dirty[0] = false;
        icache = (icache + 1) % ncache;
    }
    void shrink_to_fit() {
        for (auto &cpd : cp_data)
            cpd.shrink_to_fit();
    }
    void finalize() {
        if (!cache_dirty.empty()) {
            for (int ic = 0; ic < ncache; ic++)
                if (cache_dirty[ic]) {
                    size_t dchunk = cache_data[ic].first;
                    size_t alen =
                        min(chunk_size, arr_len - dchunk * chunk_size);
                    cp_data[dchunk].resize(alen + 1);
                    size_t clen = fpc.encode(
                        cache_data[ic].second.data(),
                        min(chunk_size, arr_len - dchunk * chunk_size),
                        cp_data[dchunk].data());
                    cp_data[dchunk].resize(clen);
                }
            cache_dirty.clear();
        }
    }
    virtual T &operator[](size_t i) {
        size_t ichunk = i / chunk_size, k = i % chunk_size;
        for (size_t j = 0; j < cache_data.size(); j++)
            if (cache_data[j].first == ichunk) {
                cache_dirty[j] = true;
                return cache_data[j].second[k];
            }
        if (icache >= (int)cache_dirty.size()) {
            cache_dirty.push_back(true);
            cache_data.push_back(make_pair(0, vector<T>(chunk_size)));
        } else if (cache_dirty[icache]) {
            size_t dchunk = cache_data[icache].first;
            size_t alen = min(chunk_size, arr_len - dchunk * chunk_size);
            cp_data[dchunk].resize(alen + 1);
            size_t clen = fpc.encode(cache_data[icache].second.data(), alen,
                                     cp_data[dchunk].data());
            cp_data[dchunk].resize(clen);
        } else
            cache_dirty[icache] = true;
        cache_data[icache].first = ichunk;
        fpc.decode(cp_data[ichunk].data(),
                   min(chunk_size, arr_len - ichunk * chunk_size),
                   cache_data[icache].second.data());
        int old_icache = icache;
        icache = (icache + 1) % ncache;
        return cache_data[old_icache].second[k];
    }
    virtual T operator[](size_t i) const {
        size_t ichunk = i / chunk_size, k = i % chunk_size;
        for (size_t j = 0; j < cache_data.size(); j++)
            if (cache_data[j].first == ichunk)
                return cache_data[j].second[k];
        if (icache >= (int)cache_dirty.size()) {
            cache_dirty.push_back(false);
            cache_data.push_back(make_pair(0, vector<T>(chunk_size)));
        } else if (cache_dirty[icache]) {
            cache_dirty[icache] = false;
            size_t dchunk = cache_data[icache].first;
            size_t alen = min(chunk_size, arr_len - dchunk * chunk_size);
            cp_data[dchunk].resize(alen + 1);
            size_t clen =
                fpc.encode(cache_data[icache].second.data(),
                           min(chunk_size, arr_len - dchunk * chunk_size),
                           cp_data[dchunk].data());
            cp_data[dchunk].resize(clen);
        }
        cache_data[icache].first = ichunk;
        fpc.decode(cp_data[ichunk].data(),
                   min(chunk_size, arr_len - ichunk * chunk_size),
                   cache_data[icache].second.data());
        int old_icache = icache;
        icache = (icache + 1) % ncache;
        return cache_data[old_icache].second[k];
    }
    size_t size() const { return arr_len; }
};

template <typename T> struct CompressedVectorMT : CompressedVector<T> {
    using CompressedVector<T>::chunk_size;
    using CompressedVector<T>::arr_len;
    using CompressedVector<T>::ncache;
    using CompressedVector<T>::fpc;
    mutable vector<int> icaches;
    mutable vector<vector<pair<size_t, vector<T>>>> cache_datas;
    shared_ptr<CompressedVector<T>> ref_cv;
    CompressedVectorMT(const shared_ptr<CompressedVector<T>> &ref_cv, int ntg)
        : ref_cv(ref_cv), CompressedVector<T>(ref_cv->arr_len, ref_cv->fpc.prec,
                                              ref_cv->chunk_size,
                                              ref_cv->ncache) {
        ref_cv->finalize();
        cache_datas.resize(ntg);
        icaches.resize(ntg);
    }
    T &operator[](size_t i) {
        static T t;
        assert(false);
        return t;
    }
    T operator[](size_t i) const {
        size_t ichunk = i / chunk_size, k = i % chunk_size;
        int tid = threading->get_thread_id();
        int &icache = icaches[tid];
        vector<pair<size_t, vector<T>>> &cache_data = cache_datas[tid];
        for (size_t j = 0; j < cache_data.size(); j++)
            if (cache_data[j].first == ichunk)
                return cache_data[j].second[k];
        if (icache >= (int)cache_data.size())
            cache_data.push_back(make_pair(0, vector<T>(chunk_size)));
        cache_data[icache].first = ichunk;
        fpc.decode(ref_cv->cp_data[ichunk].data(),
                   min(chunk_size, arr_len - ichunk * chunk_size),
                   cache_data[icache].second.data());
        int old_icache = icache;
        icache = (icache + 1) % ncache;
        return cache_data[old_icache].second[k];
    }
    size_t size() const { return arr_len; }
};

} // namespace block2
