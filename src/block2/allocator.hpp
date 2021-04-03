
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

#include "fp_codec.hpp"
#include "utils.hpp"
#ifdef _HAS_TBB
#include "tbb/scalable_allocator.h"
#endif
#include <algorithm>
#ifdef __unix__
#include <execinfo.h>
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

inline void print_trace() {
#ifdef __unix__
    void *array[32];
    size_t size = backtrace(array, 32);
    char **strings = backtrace_symbols(array, size);

    for (size_t i = 0; i < size; i++)
        fprintf(stderr, "%s\n", strings[i]);
#endif
    abort();
}

// Abstract memory allocator
template <typename T> struct Allocator {
    Allocator() {}
    virtual ~Allocator() = default;
    virtual T *allocate(size_t n) { return nullptr; }
    virtual void deallocate(void *ptr, size_t n) {}
    virtual T *reallocate(T *ptr, size_t n, size_t new_n) { return nullptr; }
    virtual shared_ptr<Allocator<T>> copy() const { return nullptr; }
};

// Stack memory allocator
template <typename T> struct StackAllocator : Allocator<T> {
    size_t size, used, shift;
    T *data;
    StackAllocator(T *ptr, size_t max_size)
        : size(max_size), used(0), shift(0), data(ptr) {}
    StackAllocator() : size(0), used(0), shift(0), data(0) {}
    T *allocate(size_t n) override {
        assert(shift == 0);
        if (used + n >= size) {
            cout << "exceeding allowed memory"
                 << " (size=" << size << ", trying to allocate " << n << ") "
                 << (sizeof(T) == 4 ? " (uint32)" : " (double)") << endl;
            print_trace();
            return 0;
        } else
            return data + (used += n) - n;
    }
    void deallocate(void *ptr, size_t n) override {
        if (n == 0)
            return;
        if (used < n || ptr != data + used - n) {
            cout << "deallocation not happening in reverse order" << endl;
            print_trace();
        } else
            used -= n;
    }
    // Change the allocated size in middle of stack memory
    // and introduce a shift for moving memory after it
    T *reallocate(T *ptr, size_t n, size_t new_n) override {
        ptr += shift;
        shift += new_n - n;
        used = used + new_n - n;
        if (ptr == data + used - new_n)
            shift = 0;
        return (T *)ptr;
    }
    friend ostream &operator<<(ostream &os, const StackAllocator &c) {
        os << "SIZE=" << c.size << " PTR=" << c.data << " USED=" << c.used
           << " SHIFT=" << (long)c.shift << endl;
        return os;
    }
};

// Vector memory allocator (with automatic deallocation)
template <typename T> struct VectorAllocator : Allocator<T> {
#ifdef _HAS_TBB
    vector<vector<T, tbb::scalable_allocator<T>>,
           tbb::scalable_allocator<vector<T, tbb::scalable_allocator<T>>>>
        data;
#else
    vector<vector<T>> data;
#endif
    VectorAllocator() {}
    T *allocate(size_t n) override {
        data.emplace_back(n);
        return data.back().data();
    }
    // explicit deallocation is not required for vector allocator
    // can be in arbitrary order
    void deallocate(void *ptr, size_t n) override {
        for (int i = (int)data.size() - 1; i >= 0; i--)
            if (data[i].data() == ptr) {
                assert(data[i].size() == n);
                data.erase(data.begin() + i);
                return;
            }
        cout << "deallocation of unallocated address" << endl;
        abort();
    }
    // Change the allocated size for one allocated block
    T *reallocate(T *ptr, size_t n, size_t new_n) override {
        for (int i = (int)data.size() - 1; i >= 0; i--)
            if (data[i].data() == ptr) {
                cout << "warning: reallocation in vector allocator may cause "
                        "undefined behavior!"
                     << endl;
                assert(data[i].size() == n);
                data[i].resize(new_n);
                return data[i].data();
            }
        cout << "reallocation of unallocated address" << endl;
        abort();
    }
    // When deep-copying objects using VectorAllocator, the other object
    // should have an independent allocator, since VectorAllocator is not global
    shared_ptr<Allocator<T>> copy() const override {
        return make_shared<VectorAllocator<T>>();
    }
    friend ostream &operator<<(ostream &os, const VectorAllocator &c) {
        os << "N-ALLOCATED=" << c.data.size << " USED="
           << accumulate(
                  c.data.begin(), c.data.end(), 0,
                  [](size_t i, const vector<T> &j) { return i + j.size(); })
           << endl;
        return os;
    }
};

// Integer stack memory pointer
inline shared_ptr<StackAllocator<uint32_t>> &ialloc_() {
    static shared_ptr<StackAllocator<uint32_t>> ialloc;
    return ialloc;
}

// Double stack memory pointer
inline shared_ptr<StackAllocator<double>> &dalloc_() {
    static shared_ptr<StackAllocator<double>> dalloc;
    return dalloc;
}

#define ialloc (ialloc_())
#define dalloc (dalloc_())

// DataFrame includes several (n_frames = 2) frames
// Each frame includes one integer stack memory
// and one double stack memory
// Using frames alternatively to avoid data copying
struct DataFrame {
    // save_dir: scartch folder for renormalized operators
    // mps_dir: scartch folder for MPS (default is the same as save_dir)
    // restart_dir: if not empty, save MPS to this dir after each sweep
    // restart_dir_per_sweep: if not empty, save MPS to this dir with sweep
    //   index as suffix, so that MPS from all sweeps will be kept in individual
    //   dir
    string save_dir, mps_dir, restart_dir = "", restart_dir_per_sweep = "";
    string prefix = "F", prefix_distri = "F0";
    bool prefix_can_write = true;
    bool partition_can_write = true;
    size_t isize, dsize;
    int n_frames, i_frame;
    mutable double tread = 0, twrite = 0, tasync = 0; // io time cost
    mutable double fpread = 0, fpwrite = 0;           // fp data io time cost
    mutable Timer _t, _t2;
    vector<shared_ptr<StackAllocator<uint32_t>>> iallocs;
    vector<shared_ptr<StackAllocator<double>>> dallocs;
    mutable vector<size_t> peak_used_memory;
    mutable vector<string> present_filenames;
    mutable vector<pair<string, shared_ptr<stringstream>>> load_buffers;
    mutable vector<pair<string, shared_ptr<stringstream>>> save_buffers;
    mutable vector<shared_future<void>> save_futures;
    bool load_buffering = false, save_buffering = false;
    bool use_main_stack = true;
    bool minimal_disk_usage = false;
    shared_ptr<FPCodec<double>> fp_codec = nullptr;
    // isize and dsize are in Bytes
    DataFrame(size_t isize = 1 << 28, size_t dsize = 1 << 30,
              const string &save_dir = "node0", double dmain_ratio = 0.7,
              double imain_ratio = 0.7, int n_frames = 2)
        : n_frames(n_frames), save_dir(save_dir), mps_dir(save_dir) {
        peak_used_memory.resize(n_frames * 2);
        present_filenames.resize(n_frames);
        load_buffers.resize(n_frames);
        save_buffers.resize(n_frames);
        save_futures.resize(n_frames);
        this->isize = isize >> 2;
        this->dsize = dsize >> 3;
        size_t imain = (size_t)(imain_ratio * this->isize);
        size_t dmain = (size_t)(dmain_ratio * this->dsize);
        size_t ir = (this->isize - imain) / (n_frames - 1);
        size_t dr = (this->dsize - dmain) / (n_frames - 1);
        double *dptr = new double[this->dsize];
        uint32_t *iptr = new uint32_t[this->isize];
        iallocs.push_back(make_shared<StackAllocator<uint32_t>>(iptr, imain));
        dallocs.push_back(make_shared<StackAllocator<double>>(dptr, dmain));
        iptr += imain;
        dptr += dmain;
        for (int i = 0; i < n_frames - 1; i++) {
            iallocs.push_back(
                make_shared<StackAllocator<uint32_t>>(iptr + i * ir, ir));
            dallocs.push_back(
                make_shared<StackAllocator<double>>(dptr + i * dr, dr));
        }
        activate(0);
        if (!Parsing::path_exists(save_dir))
            Parsing::mkdir(save_dir);
        if (!Parsing::path_exists(mps_dir))
            Parsing::mkdir(mps_dir);
    }
    ~DataFrame() { deallocate(); }
    void activate(int i) {
        ialloc_() = iallocs[i_frame = i];
        dalloc_() = dallocs[i_frame];
    }
    void reset(int i) {
        iallocs[i]->used = 0;
        dallocs[i]->used = 0;
        present_filenames[i] = "";
    }
    void reset_buffer(int i) {
        load_buffers[i] = make_pair("", nullptr);
        if (save_buffering && save_futures[i].valid())
            save_futures[i].wait();
        save_buffers[i] = make_pair("", nullptr);
    }
    void rename_data(const string &old_filename,
                     const string &new_filename) const {
        if (!Parsing::rename_file(old_filename, new_filename))
            throw runtime_error("Renaming '" + old_filename + "' to '" +
                                new_filename + "' failed.");
        for (auto &fn : present_filenames)
            fn = "";
    }
    void load_data_from(int i, istream &ifs) const {
        ifs.read((char *)&iallocs[i]->used, sizeof(iallocs[i]->used));
        ifs.read((char *)&dallocs[i]->used, sizeof(dallocs[i]->used));
        ifs.read((char *)iallocs[i]->data, sizeof(uint32_t) * iallocs[i]->used);
        _t2.get_time();
        if (fp_codec != nullptr)
            fp_codec->read_array(ifs, dallocs[i]->data, dallocs[i]->used);
        else
            ifs.read((char *)dallocs[i]->data,
                     sizeof(double) * dallocs[i]->used);
        fpread += _t2.get_time();
    }
    // Load one data frame from disk
    void load_data(int i, const string &filename) const {
        _t.get_time();
        if (present_filenames[i] == filename) {
            return;
        } else if (load_buffers[i].first == filename) {
            shared_ptr<stringstream> ss = make_shared<stringstream>();
            if (load_buffering && present_filenames[i] != "")
                save_data_to(i, *ss);
            load_buffers[i].second->clear();
            load_buffers[i].second->seekg(0);
            load_data_from(i, *load_buffers[i].second);
            load_buffers[i] = make_pair(present_filenames[i], ss);
            present_filenames[i] = filename;
            tread += _t.get_time();
            return;
        } else if (load_buffering && present_filenames[i] != "") {
            shared_ptr<stringstream> ss = make_shared<stringstream>();
            save_data_to(i, *ss);
            load_buffers[i] = make_pair(present_filenames[i], ss);
        }
        if (save_buffers[i].first == filename) {
            if (save_futures[i].valid())
                save_futures[i].wait();
            save_buffers[i].second->clear();
            save_buffers[i].second->seekg(0);
            load_data_from(i, *save_buffers[i].second);
            present_filenames[i] = filename;
            tread += _t.get_time();
            return;
        }
        ifstream ifs(filename.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("DataFrame::load_data on '" + filename +
                                "' failed.");
        load_data_from(i, ifs);
        if (ifs.fail() || ifs.bad())
            throw runtime_error("DataFrame::load_data on '" + filename +
                                "' failed.");
        ifs.close();
        tread += _t.get_time();
        update_peak_used_memory();
        present_filenames[i] = filename;
    }
    void save_data_to(int i, ostream &ofs) const {
        ofs.write((char *)&iallocs[i]->used, sizeof(iallocs[i]->used));
        ofs.write((char *)&dallocs[i]->used, sizeof(dallocs[i]->used));
        ofs.write((char *)iallocs[i]->data,
                  sizeof(uint32_t) * iallocs[i]->used);
        _t2.get_time();
        if (fp_codec != nullptr)
            fp_codec->write_array(ofs, dallocs[i]->data, dallocs[i]->used);
        else
            ofs.write((char *)dallocs[i]->data,
                      sizeof(double) * dallocs[i]->used);
        fpwrite += _t2.get_time();
    }
    static void buffer_save_data(const string &filename,
                                 const shared_ptr<stringstream> &ss,
                                 double *tasync) {
        Timer tx;
        tx.get_time();
        if (Parsing::link_exists(filename))
            Parsing::remove_file(filename);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("DataFrame::buffer_save_data on '" + filename +
                                "' failed.");
        ss->clear();
        ss->seekg(0);
        ofs << ss->rdbuf();
        if (!ofs.good())
            throw runtime_error("DataFrame::buffer_save_data on '" + filename +
                                "' failed.");
        ofs.close();
        *tasync += tx.get_time();
    }
    // Save one data frame to disk
    void save_data(int i, const string &filename) const {
        if (!partition_can_write) {
            update_peak_used_memory();
            present_filenames[i] = filename;
            return;
        }
        _t.get_time();
        if (save_buffering) {
            if (save_futures[i].valid())
                save_futures[i].wait();
            shared_ptr<stringstream> ss = make_shared<stringstream>();
            save_data_to(i, *ss);
            save_buffers[i] = make_pair(filename, ss);
            save_futures[i] = async(launch::async, &DataFrame::buffer_save_data,
                                    filename, ss, &tasync);
            twrite += _t.get_time();
            update_peak_used_memory();
            present_filenames[i] = filename;
            return;
        }
        if (Parsing::link_exists(filename))
            Parsing::remove_file(filename);
        ofstream ofs(filename.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("DataFrame::save_data on '" + filename +
                                "' failed.");
        save_data_to(i, ofs);
        if (!ofs.good())
            throw runtime_error("DataFrame::save_data on '" + filename +
                                "' failed.");
        ofs.close();
        twrite += _t.get_time();
        update_peak_used_memory();
        present_filenames[i] = filename;
    }
    void deallocate() {
        delete[] iallocs[0]->data;
        delete[] dallocs[0]->data;
        iallocs.clear();
        dallocs.clear();
        if (save_buffering)
            for (const auto &ft : save_futures)
                if (ft.valid())
                    ft.wait();
    }
    size_t memory_used() const {
        size_t r = 0;
        for (int i = 0; i < n_frames; i++)
            r += dallocs[i]->used * 8 + iallocs[i]->used * 4;
        return r;
    }
    void update_peak_used_memory() const {
        for (int i = 0; i < n_frames; i++) {
            peak_used_memory[i + 0 * n_frames] =
                max(peak_used_memory[i + 0 * n_frames], dallocs[i]->used * 8);
            peak_used_memory[i + 1 * n_frames] =
                max(peak_used_memory[i + 1 * n_frames], iallocs[i]->used * 4);
        }
    }
    void reset_peak_used_memory() const {
        memset(peak_used_memory.data(), 0,
               sizeof(size_t) * peak_used_memory.size());
    }
    friend ostream &operator<<(ostream &os, const DataFrame &df) {
        os << " UseMainStack = " << df.use_main_stack
           << " MinDiskUsage = " << df.minimal_disk_usage
           << " IBuf = " << df.load_buffering << " OBuf = " << df.save_buffering
           << endl;
        if (df.fp_codec != nullptr)
            os << " FPCompression: prec = " << scientific << setprecision(2)
               << df.fp_codec->prec << " chunk = " << fixed
               << df.fp_codec->chunk_size << endl;
        os << " IMain = " << Parsing::to_size_string(df.iallocs[0]->used * 4)
           << " / " << Parsing::to_size_string(df.iallocs[0]->size * 4);
        os << " DMain = " << Parsing::to_size_string(df.dallocs[0]->used * 8)
           << " / " << Parsing::to_size_string(df.dallocs[0]->size * 8);
        os << " ISeco = " << Parsing::to_size_string(df.iallocs[1]->used * 4)
           << " / " << Parsing::to_size_string(df.iallocs[1]->size * 4);
        os << " DSeco = " << Parsing::to_size_string(df.dallocs[1]->used * 8)
           << " / " << Parsing::to_size_string(df.dallocs[1]->size * 8);
        return os;
    }
};

inline shared_ptr<DataFrame> &frame_() {
    static shared_ptr<DataFrame> frame;
    return frame;
}

#define frame (frame_())

// Function pointer for signal checking
inline void (*&check_signal_())() {
    static void (*check_signal)() = []() {};
    return check_signal;
}

} // namespace block2
