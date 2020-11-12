
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

using namespace std;

namespace block2 {

// Wall time recorder
struct Timer {
    double current;
    Timer() : current(0) {}
    double get_time() {
        struct timeval t;
        gettimeofday(&t, NULL);
        double previous = current;
        current = t.tv_sec + 1E-6 * t.tv_usec;
        return current - previous;
    }
};

// Random number generator
struct Random {
    static mt19937 &rng() {
        static mt19937 _rng;
        return _rng;
    }
    static void rand_seed(unsigned i = 0) {
        rng() = mt19937(
            i ? i : chrono::steady_clock::now().time_since_epoch().count());
    }
    // return a integer in [a, b)
    static int rand_int(int a, int b) {
        assert(b > a);
        return uniform_int_distribution<int>(a, b - 1)(rng());
    }
    // return a double in [a, b)
    static double rand_double(double a = 0, double b = 1) {
        assert(b > a);
        return uniform_real_distribution<double>(a, b)(rng());
    }
    static void fill_rand_double(double *data, size_t n, double a = 0,
                                 double b = 1) {
        uniform_real_distribution<double> distr(a, b);
        for (size_t i = 0; i < n; i++)
            data[i] = distr(rng());
    }
};

// Text file parsing
struct Parsing {
    static vector<string> readlines(istream *input) {
        string h;
        vector<string> r;
        while (!input->eof()) {
            getline(*input, h);
            size_t idx = h.find("!");
            if (idx != string::npos)
                h = string(h, 0, idx);
            while ((idx = h.find("\r")) != string::npos)
                h.replace(idx, 1, "");
            r.push_back(h);
        }
        return r;
    }
    static vector<string> split(const string &s, const string &delim,
                                bool remove_empty = false) {
        vector<string> r;
        size_t last = 0;
        size_t index = s.find_first_of(delim, last);
        while (index != string::npos) {
            if (!remove_empty || index > last)
                r.push_back(s.substr(last, index - last));
            last = index + 1;
            index = s.find_first_of(delim, last);
        }
        if (index > last)
            r.push_back(s.substr(last, index - last));
        return r;
    }
    static string &lower(string &x) {
        transform(x.begin(), x.end(), x.begin(), ::tolower);
        return x;
    }
    static string &trim(string &x) {
        if (x.empty())
            return x;
        x.erase(0, x.find_first_not_of(" \t"));
        x.erase(x.find_last_not_of(" \t") + 1);
        return x;
    }
    template <typename T>
    static string join(T it_start, T it_end, const string &x) {
        stringstream r;
        for (T i = it_start; i != it_end; i++)
            r << *i << x;
        string rr = r.str();
        if (rr.size() != 0)
            rr.resize(rr.size() - x.length());
        return rr;
    }
    static long long to_long_long(const string &x) { return atoll(x.c_str()); }
    static int to_int(const string &x) { return atoi(x.c_str()); }
    static double to_double(const string &x) { return atof(x.c_str()); }
    static string to_string(int i) {
        stringstream ss;
        ss << i;
        return ss.str();
    }
    static string to_size_string(size_t i, const string &suffix = "B") {
        stringstream ss;
        size_t a = 1024;
        if (i < 1000) {
            ss << i << " " << suffix;
            return ss.str();
        } else {
            string prefix = "KMGTPEZY";
            for (size_t j = 0; j < prefix.size(); j++, a *= 1024) {
                for (int k = 10, p = 2; k <= 1000; k *= 10, p--)
                    if (i < k * a) {
                        ss << fixed << setprecision(p) << (i / (long double)a)
                           << " " << prefix[j] << suffix;
                        return ss.str();
                    }
            }
            return "??? B";
        }
    }
    static bool rename_file(const string &old_name, const string &new_name) {
        return rename(old_name.c_str(), new_name.c_str()) == 0;
    }
    static bool remove_file(const string &name) {
        return remove(name.c_str()) == 0;
    }
    static bool link_file(const string &source, const string &name) {
        if (file_exists(name))
            remove_file(name);
        return symlink(source.c_str(), name.c_str()) == 0;
    }
    static bool link_exists(const string &name) {
        struct stat buffer;
        return lstat(name.c_str(), &buffer) == 0 &&
               (buffer.st_mode & S_IFLNK) == S_IFLNK;
    }
    static bool file_exists(const string &name) {
        struct stat buffer;
        return stat(name.c_str(), &buffer) == 0;
    }
    static bool path_exists(const string &name) {
        struct stat buffer;
        return stat(name.c_str(), &buffer) == 0 && (buffer.st_mode & S_IFDIR);
    }
    static void mkdir(const string &name) { ::mkdir(name.c_str(), 0755); }
};

} // namespace block2
