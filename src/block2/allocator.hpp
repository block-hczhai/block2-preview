
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

#include "utils.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace block2 {

// Stack memory allocator
template <typename T> struct StackAllocator {
    size_t size, used, shift;
    T *data;
    StackAllocator(T *ptr, size_t max_size)
        : size(max_size), used(0), shift(0), data(ptr) {}
    StackAllocator() : size(0), used(0), shift(0), data(0) {}
    T *allocate(size_t n) {
        assert(shift == 0);
        if (used + n >= size) {
            cout << "exceeding allowed memory"
                 << (sizeof(T) == 4 ? " (uint32)" : " (double)") << endl;
            abort();
            return 0;
        } else
            return data + (used += n) - n;
    }
    void deallocate(void *ptr, size_t n) {
        if (n == 0)
            return;
        if (used < n || ptr != data + used - n) {
            cout << "deallocation not happening in reverse order" << endl;
            abort();
        } else
            used -= n;
    }
    // Change the allocated size in middle of stack memory
    // and introduce a shift for moving memory after it
    T *reallocate(T *ptr, size_t n, size_t new_n) {
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

// Integer stack memory pointer
inline StackAllocator<uint32_t> *&ialloc_() {
    static StackAllocator<uint32_t> *ialloc;
    return ialloc;
}

// Double stack memory pointer
inline StackAllocator<double> *&dalloc_() {
    static StackAllocator<double> *dalloc;
    return dalloc;
}

#define ialloc (ialloc_())
#define dalloc (dalloc_())

// DataFrame includes several (n_frames = 2) frames
// Each frame includes one integer stack memory
// and one double stack memory
// Using frames alternatively to avoid data copying
struct DataFrame {
    string save_dir, prefix = "F0";
    size_t isize, dsize;
    uint16_t n_frames, i_frame;
    vector<StackAllocator<uint32_t>> iallocs;
    vector<StackAllocator<double>> dallocs;
    // isize and dsize are in Bytes
    DataFrame(size_t isize = 1 << 28, size_t dsize = 1 << 30,
              const string &save_dir = "node0", double main_ratio = 0.7,
              uint16_t n_frames = 2)
        : n_frames(n_frames), save_dir(save_dir) {
        this->isize = isize >> 2;
        this->dsize = dsize >> 3;
        size_t imain = (size_t)(main_ratio * this->isize);
        size_t dmain = (size_t)(main_ratio * this->dsize);
        size_t ir = (this->isize - imain) / (n_frames - 1);
        size_t dr = (this->dsize - dmain) / (n_frames - 1);
        double *dptr = new double[this->dsize];
        uint32_t *iptr = new uint32_t[this->isize];
        iallocs.push_back(StackAllocator<uint32_t>(iptr, imain));
        dallocs.push_back(StackAllocator<double>(dptr, dmain));
        iptr += imain;
        dptr += dmain;
        for (uint16_t i = 0; i < n_frames - 1; i++) {
            iallocs.push_back(StackAllocator<uint32_t>(iptr + i * ir, ir));
            dallocs.push_back(StackAllocator<double>(dptr + i * dr, dr));
        }
        activate(0);
        if (!Parsing::path_exists(save_dir))
            Parsing::mkdir(save_dir);
    }
    void activate(uint16_t i) {
        ialloc = &iallocs[i_frame = i];
        dalloc = &dallocs[i_frame];
    }
    void reset(uint16_t i) {
        iallocs[i].used = 0;
        dallocs[i].used = 0;
    }
    // Load one data frame from disk
    void load_data(uint16_t i, const string &filename) const {
        ifstream ifs(filename.c_str(), ios::binary);
        ifs.read((char *)&dallocs[i].used, sizeof(dallocs[i].used));
        ifs.read((char *)dallocs[i].data, sizeof(double) * dallocs[i].used);
        ifs.read((char *)&iallocs[i].used, sizeof(iallocs[i].used));
        ifs.read((char *)iallocs[i].data, sizeof(uint32_t) * iallocs[i].used);
        ifs.close();
    }
    // Save one data frame to disk
    void save_data(uint16_t i, const string &filename) const {
        ofstream ofs(filename.c_str(), ios::binary);
        ofs.write((char *)&dallocs[i].used, sizeof(dallocs[i].used));
        ofs.write((char *)dallocs[i].data, sizeof(double) * dallocs[i].used);
        ofs.write((char *)&iallocs[i].used, sizeof(iallocs[i].used));
        ofs.write((char *)iallocs[i].data, sizeof(uint32_t) * iallocs[i].used);
        ofs.close();
    }
    void deallocate() {
        delete[] iallocs[0].data;
        delete[] dallocs[0].data;
        iallocs.clear();
        dallocs.clear();
    }
    friend ostream &operator<<(ostream &os, const DataFrame &df) {
        os << "persistent memory used :: I = " << df.iallocs[0].used << "("
           << (df.iallocs[0].used * 100 / df.iallocs[0].size) << "%)"
           << " D = " << df.dallocs[0].used << "("
           << (df.dallocs[0].used * 100 / df.dallocs[0].size) << "%)" << endl;
        os << "exclusive  memory used :: I = " << df.iallocs[1].used << "("
           << (df.iallocs[1].used * 100 / df.iallocs[1].size) << "%)"
           << " D = " << df.dallocs[1].used << "("
           << (df.dallocs[1].used * 100 / df.dallocs[1].size) << "%)" << endl;
        return os;
    }
};

inline DataFrame *&frame_() {
    static DataFrame *frame;
    return frame;
}

#define frame (frame_())

// Function pointer for signal checking
inline void (*&check_signal_())() {
    static void (*check_signal)() = []() {};
    return check_signal;
}

} // namespace block2
