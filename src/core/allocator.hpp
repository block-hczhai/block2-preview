
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

/** Global definition for global stack memory, scratch space, restarting
 * folder, and IO strategies. */

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

/**
 * Print calling stack when an error happens.
 * Not working for non-unix systems.
 */
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

/** Abstract memory allocator.
 * @tparam T The type of the element in the array. */
template <typename T> struct Allocator {
    /** Default constructor. */
    Allocator() {}
    /** Default destructor. */
    virtual ~Allocator() = default;
    /** Allocate a length n array.
     * @param n Number of elements in the array.
     * @return The allocated pointer.
     */
    virtual T *allocate(size_t n) { return nullptr; }
    /** Deallocate a length n array.
     * @param ptr The pointer to be deallocated.
     * @param n Number of elements in the array.
     */
    virtual void deallocate(void *ptr, size_t n) {}
    /** Adjust the size an allocated pointer. No data copying will happen.
     * @param ptr The allocated pointer.
     * @param n Number of elements in original allocation.
     * @param new_n Number of elements in the new allocation.
     * @return The new pointer.
     */
    virtual T *reallocate(T *ptr, size_t n, size_t new_n) { return nullptr; }
    /** Return a copy of the allocator.
     * @return ptr The copy of this allocator.
     */
    virtual shared_ptr<Allocator<T>> copy() const { return nullptr; }
};

/** Stack memory allocator.
 * @tparam T The type of the element in the array. */
template <typename T> struct StackAllocator : Allocator<T> {
    size_t size, //!< Total size of the stack (in number of elements).
        used,    //!< Occupied size of the stack (in number of elements).
        shift; //!< Temporary shift introduced due to deallocation in the middle
               //!< of the stack.
    T *data;   //!< Pointer to the first elemenet in the stack.
    /** Constructor.
     * @param ptr Pointer to the first elemenet in the stack. The stack should
     * be pre-allocated.
     * @param max_size Total size of the stack (in number of elements).
     */
    StackAllocator(T *ptr, size_t max_size)
        : size(max_size), used(0), shift(0), data(ptr) {}
    /** Default constructor. */
    StackAllocator() : size(0), used(0), shift(0), data(0) {}
    /** Allocate a length n array.
     * @param n Number of elements in the array.
     * @return The allocated pointer.
     */
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
    /** Deallocate a length n array.
     * Must be invoked in the reverse order of allocation.
     * @param ptr The pointer to be deallocated.
     * @param n Number of elements in the array.
     */
    void deallocate(void *ptr, size_t n) override {
        if (n == 0)
            return;
        if (used < n || ptr != data + used - n) {
            cout << "deallocation not happening in reverse order" << endl;
            print_trace();
        } else
            used -= n;
    }
    /** Change the allocated size in middle of stack memory
     * and introduce a shift for moving memory after it.
     * @param ptr The allocated pointer.
     * @param n Number of elements in original allocation.
     * @param new_n Number of elements in the new allocation.
     * @return The new pointer.
     */
    T *reallocate(T *ptr, size_t n, size_t new_n) override {
        ptr += shift;
        shift += new_n - n;
        used = used + new_n - n;
        if (ptr == data + used - new_n)
            shift = 0;
        return (T *)ptr;
    }
    /** Print the status of the allocator.
     * @param os The output stream.
     * @param c The object to be printed.
     * @return The output stream.
     */
    friend ostream &operator<<(ostream &os, const StackAllocator &c) {
        os << "SIZE=" << c.size << " PTR=" << c.data << " USED=" << c.used
           << " SHIFT=" << (long)c.shift << endl;
        return os;
    }
};

/** Vector memory allocator.
 * @tparam T The type of the element in the array. */
template <typename T> struct VectorAllocator : Allocator<T> {
#ifdef _HAS_TBB
    vector<vector<T, tbb::scalable_allocator<T>>,
           tbb::scalable_allocator<vector<T, tbb::scalable_allocator<T>>>>
        data; //!< The data blocks allocated using TBB for better threading
              //!< performance.
#else
    vector<vector<T>> data; //!< The allocated data blocks.
#endif
    /** Default constructor. */
    VectorAllocator() {}
    /** Allocate a length n array.
     * @param n Number of elements in the array.
     * @return The allocated pointer.
     */
    T *allocate(size_t n) override {
        data.emplace_back(n);
        return data.back().data();
    }
    /** Deallocate a length n array. Note that explicit deallocation is not
     * required for vector allocator. Can be invoked in arbitrary order.
     * @param ptr The pointer to be deallocated.
     * @param n Number of elements in the array.
     */
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
    /** Change the allocated size for one allocated block.
     * @param ptr The allocated pointer.
     * @param n Number of elements in original allocation.
     * @param new_n Number of elements in the new allocation.
     * @return The new pointer.
     */
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
    /** Return a copy of the allocator. When deep-copying objects using
     * VectorAllocator, the other object should have an independent allocator,
     * since VectorAllocator is not global.
     * @return The copy of this allocator.
     */
    shared_ptr<Allocator<T>> copy() const override {
        return make_shared<VectorAllocator<T>>();
    }
    /** Print the status of the allocator.
     * @param os The output stream.
     * @param c The object to be printed.
     * @return The output stream.
     */
    friend ostream &operator<<(ostream &os, const VectorAllocator &c) {
        os << "N-ALLOCATED=" << c.data.size << " USED="
           << accumulate(
                  c.data.begin(), c.data.end(), 0,
                  [](size_t i, const vector<T> &j) { return i + j.size(); })
           << endl;
        return os;
    }
};

/** Implementation of the ``ialloc`` global variable. */
inline shared_ptr<StackAllocator<uint32_t>> &ialloc_() {
    static shared_ptr<StackAllocator<uint32_t>> ialloc;
    return ialloc;
}

/** Implementation of the ``dalloc`` global variable. */
inline shared_ptr<StackAllocator<double>> &dalloc_() {
    static shared_ptr<StackAllocator<double>> dalloc;
    return dalloc;
}

/** Global variable for the integer stack memory allocator. */
#define ialloc (ialloc_())

/** Global variable for the double stack memory allocator. */
#define dalloc (dalloc_())

/** DataFrame includes several (n_frames = 2) frames.
 * Each frame includes one integer stack memory and one double stack memory.
 * The two frames are used alternatively to avoid data copying. */
struct DataFrame {
    string save_dir, //!< Scartch folder for renormalized operators.
        mps_dir; //!< Scartch folder for MPS (default is the same as save_dir).
    string restart_dir =
               "", //!< If not empty, save MPS to this dir after each sweep.
        restart_dir_per_sweep =
            ""; //!< if not empty, save MPS to this dir with sweep index as
                //!< suffix, so that MPS from all sweeps will be kept in
                //!< individual dirs.
    string restart_dir_optimal_mps =
        ""; //!< If not empty, save MPS to this dir
            //!< whenever an optimal solution is reached in one sweep. For DMRG,
            //!< this is the MPS with the lowest energy. Note that if the best
            //!< solution from the current sweep is worse than the best solution
            //!< from the previous sweep (for example in a reverse schedule),
            //!< the best solution from the current sweep is saved.
    string restart_dir_optimal_mps_per_sweep =
        ""; //!< If not empty, save the optimal MPS from each sweep to this dir
            //!< with sweep index as suffix.
    string prefix = "F", //!< Filename prefix for common scratch files (such as
                         //!< MPS tensors).
        prefix_distri =
            "F0"; //!< Filename prefix for distributed scratch files (such as
                  //!< renormalized operators). When distributed parallelization
                  //!< is used, different procs will have different values for
                  //!< this data.
    bool prefix_can_write =
        true; //!< Whether this proc should be able to write common scratch
              //!< files (such as MPS tensors).
    bool partition_can_write = true; //!< Whether this proc should be able to
                                     //!< write renormalized operators.
    size_t isize,             //!< Max number of elements in all integer stacks.
        dsize;                //!< Max number of elements in all double stacks.
    int n_frames,             //!< Total number of data frames.
        i_frame;              //!< The index of Current activated data frame.
    mutable double tread = 0, //!< IO Time cost for reading scratch files.
        twrite = 0,           //!< IO Time cost for writing scratch files.
        tasync = 0;           //!< IO Time cost for async writing scratch files.
    mutable double fpread = 0, //!< IO Time cost for reading scratch files with
                               //!< floating-point decompression.
        fpwrite = 0;           //!< IO Time cost for writing scratch files with
                               //!< floating-point compression.
    mutable Timer _t,          //!< Temporary timer.
        _t2;                   //!< Auxiliary temporary timer.
    vector<shared_ptr<StackAllocator<uint32_t>>>
        iallocs; //!< Integer stacks allocators.
    vector<shared_ptr<StackAllocator<double>>>
        dallocs; //!< Double stacks allocators.
    mutable vector<size_t>
        peak_used_memory; //!< Peak used memory by stacks (in Bytes). Even
                          //!< indices are for double stacks. Odd indices are
                          //!< for interger stacks.
    mutable vector<string>
        present_filenames; //!< The filename for the current stack memory
                           //!< content for each data frame. Used for tracking
                           //!< loading and saving buffering to avoid loading
                           //!< the same data into memory.
    mutable vector<pair<string, shared_ptr<stringstream>>> load_buffers;
    //!< Buffers for loading. Skpping reading a file with certain filename, if
    //!< the contents of the file with that filename is in the loading buffer.
    mutable vector<pair<string, shared_ptr<stringstream>>> save_buffers;
    //!< Buffers for Async saving.
    mutable vector<shared_future<void>> save_futures;
    //!< Async saving files.
    bool load_buffering = false, //!< Whether load buffering should be used. If
                                 //!< true, memory usage will increase.
        save_buffering =
            false; //!< Whether async saving and saving buffering should be
                   //!< used. If true, memory usage will increase.
    bool use_main_stack =
        true; //!< Whether main stack should be used for storing blocked
              //!< operators in enlarged blocks. If false, these blocked
              //!< operators will be stored in dynamically allocated memory.
    bool minimal_disk_usage =
        false; //!< Whether temporary renormalized operator files should be
               //!< deleted as soon as possible. If true, will save roughly half
               //!< of required storage for renormalized operators.
    shared_ptr<FPCodec<double>> fp_codec =
        nullptr; //!< Floating-point compression codec. If nullptr,
                 //!< floating-point compression will not be used.
    // isize and dsize are in Bytes
    /** Constructor.
     * @param isize Max size (in bytes) of all integer stacks.
     * @param dsize Max size (in bytes) of all double stacks.
     * @param save_dir Scartch folder for renormalized operators.
     * @param dmain_ratio The fraction of stack space occupied by the main
     * double stacks.
     * @param imain_ratio The fraction of stack space occupied by the main
     * integer stacks.
     * @param n_frames Number of data frames.
     */
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
    /** Destructor. */
    ~DataFrame() { deallocate(); }
    /** Activate one data frame.
     * @param i The index of the data frame to be activated.
     */
    void activate(int i) {
        ialloc_() = iallocs[i_frame = i];
        dalloc_() = dallocs[i_frame];
    }
    /** Reset one data frame, marking all stack memory as unused.
     * @param i The index of the data frame to be reset.
     */
    void reset(int i) {
        iallocs[i]->used = 0;
        dallocs[i]->used = 0;
        present_filenames[i] = "";
    }
    /** Reset saving and loading buffers for one data frame.
     * Contents in the loading buffer will be deleted.
     * Unsaved contents in the saving buffer will be saved in disk.
     * @param i The index of the data frame.
     */
    void reset_buffer(int i) {
        load_buffers[i] = make_pair("", nullptr);
        if (save_buffering && save_futures[i].valid())
            save_futures[i].wait();
        save_buffers[i] = make_pair("", nullptr);
    }
    /** Rename one scratch file.
     * @param old_filename original filename.
     * @param new_filename new filename.
     */
    void rename_data(const string &old_filename,
                     const string &new_filename) const {
        if (!Parsing::rename_file(old_filename, new_filename))
            throw runtime_error("Renaming '" + old_filename + "' to '" +
                                new_filename + "' failed.");
        for (auto &fn : present_filenames)
            fn = "";
    }
    /** Load one data frame from input stream.
     * @param i The index of the data frame.
     * @param ifs The input stream.
     */
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
    /** Load one data frame from disk.
     * @param i The index of the data frame.
     * @param filename The filename for the data frame.
     */
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
    /** Save one data frame into output stream.
     * @param i The index of the data frame.
     * @param ofs The output stream.
     */
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
    /** Save the data in buffer stream into disk.
     * @param filename The filename for saving data.
     * @param ss The buffer stream.
     * @param tasync Pointer to the time recorder for async saving.
     */
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
    /** Save one data frame to disk.
     * @param i The index of the data frame.
     * @param filename The filename for the data frame.
     */
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
    /** Deallocate the memory allocated for all stacks.
     * Note that this method is automatically invoked at deconstruction.
     */
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
    /** Return the current used memory in all stacks.
     * @return The current used memory in Bytes.
     */
    size_t memory_used() const {
        size_t r = 0;
        for (int i = 0; i < n_frames; i++)
            r += dallocs[i]->used * 8 + iallocs[i]->used * 4;
        return r;
    }
    /** Update prak used memory statistics. */
    void update_peak_used_memory() const {
        for (int i = 0; i < n_frames; i++) {
            peak_used_memory[i + 0 * n_frames] =
                max(peak_used_memory[i + 0 * n_frames], dallocs[i]->used * 8);
            peak_used_memory[i + 1 * n_frames] =
                max(peak_used_memory[i + 1 * n_frames], iallocs[i]->used * 4);
        }
    }
    /** Reset prak used memory statistics to zero. */
    void reset_peak_used_memory() const {
        memset(peak_used_memory.data(), 0,
               sizeof(size_t) * peak_used_memory.size());
    }
    /** Print the status of the data frame.
     * @param os The output stream.
     * @param df The object to be printed.
     * @return The output stream.
     */
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

/** Implementation of the ``frame`` global variable. */
inline shared_ptr<DataFrame> &frame_() {
    static shared_ptr<DataFrame> frame;
    return frame;
}

/** Global variable for accessing global stack memory and file I/O in scratch
 * space. */
#define frame (frame_())

/** Function pointer for signal checking. */
inline auto check_signal_() -> void (*&)() {
    static void (*check_signal)() = []() {};
    return check_signal;
}

} // namespace block2
