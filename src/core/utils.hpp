
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#ifndef _WIN32
#include <sys/time.h>
#include <unistd.h>
#define _mkdir(x) ::mkdir(x, 0755)
#else
#include <direct.h>
#include <io.h>
#endif
#include <vector>

#ifdef _WIN32
typedef int64_t ssize_t;
#endif

using namespace std;

namespace block2 {

#define CONCAT(a, b) CONCATX(a, b)
#define CONCATX(a, b) a##b

#ifdef _F77UNDERSCORE
#define LFNAME(X) CONCAT(X, _)
#else
#define LFNAME(X) X
#endif

#define FNAME(X) LFNAME(X)

#ifdef _APPLE_ACC_SINGLE_PREC
#define FRETT double
#else
#define FRETT float
#endif

#ifdef _WIN32

struct timeval {
    long long tv_sec, tv_usec;
};

inline int gettimeofday(struct timeval *tp, struct timezone *tzp) {
    chrono::system_clock::duration d =
        chrono::system_clock::now().time_since_epoch();
    chrono::seconds s = chrono::duration_cast<chrono::seconds>(d);
    tp->tv_sec = s.count();
    tp->tv_usec = chrono::duration_cast<chrono::microseconds>(d - s).count();
    return 0;
}

#endif

template <typename FL> inline FL xconj(FL x) { return x; }

template <>
inline complex<long double>
xconj<complex<long double>>(complex<long double> x) {
    return conj(x);
}

template <> inline complex<double> xconj<complex<double>>(complex<double> x) {
    return conj(x);
}

template <> inline complex<float> xconj<complex<float>>(complex<float> x) {
    return conj(x);
}

template <typename FL> inline decltype(abs((FL)0.0)) ximag(FL x) {
    return (decltype(abs((FL)0.0)))0.0;
}

template <>
inline long double ximag<complex<long double>>(complex<long double> x) {
    return imag(x);
}

template <> inline double ximag<complex<double>>(complex<double> x) {
    return imag(x);
}

template <> inline float ximag<complex<float>>(complex<float> x) {
    return imag(x);
}

template <typename FL> inline decltype(abs((FL)0.0)) xreal(FL x) {
    return (decltype(abs((FL)0.0)))x;
}

template <>
inline long double xreal<complex<long double>>(complex<long double> x) {
    return real(x);
}

template <> inline double xreal<complex<double>>(complex<double> x) {
    return real(x);
}

template <> inline float xreal<complex<float>>(complex<float> x) {
    return real(x);
}

template <typename FL> struct alt_fl_type;

template <> struct alt_fl_type<float> {
    typedef double FL;
};

template <> struct alt_fl_type<double> {
    typedef float FL;
};

template <> struct alt_fl_type<complex<float>> {
    typedef complex<double> FL;
};

template <> struct alt_fl_type<complex<double>> {
    typedef complex<float> FL;
};

template <typename FL> struct const_fl_type;

template <> struct const_fl_type<float> {
    typedef double FL;
};

template <> struct const_fl_type<double> {
    typedef long double FL;
};

template <> struct const_fl_type<complex<float>> {
    typedef complex<double> FL;
};

template <> struct const_fl_type<complex<double>> {
    typedef complex<long double> FL;
};

template <typename T> struct is_complex : integral_constant<bool, false> {};

template <typename T> struct is_complex<complex<T>> : is_floating_point<T> {};

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

// Random number generator for multi-thread
struct RandomMT {
    mt19937 rng;
    RandomMT(unsigned i = 0)
        : rng(i ? i
                : (unsigned)chrono::steady_clock::now()
                      .time_since_epoch()
                      .count()) {}
    // return a integer in [a, b)
    int rand_int(int a, int b) {
        assert(b > a);
        return uniform_int_distribution<int>(a, b - 1)(rng);
    }
    // return a double in [a, b)
    double rand_double(double a = 0, double b = 1) {
        assert(b > a);
        return uniform_real_distribution<double>(a, b)(rng);
    }
    template <typename FL> void fill(FL *data, size_t n, FL a = 0, FL b = 1) {
        uniform_real_distribution<FL> distr(a, b);
        for (size_t i = 0; i < n; i++)
            data[i] = distr(rng);
    }
};

// Random number generator
struct Random {
    static mt19937 &rng() {
        static mt19937 _rng;
        return _rng;
    }
    static void rand_seed(unsigned i = 0) {
        rng() = mt19937(i ? i
                          : (unsigned)chrono::steady_clock::now()
                                .time_since_epoch()
                                .count());
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
    template <typename FL>
    static void fill(FL *data, size_t n, FL a = 0, FL b = 1) {
        uniform_real_distribution<FL> distr(a, b);
        for (size_t i = 0; i < n; i++)
            data[i] = distr(rng());
    }
    template <typename FL>
    static void complex_fill(complex<FL> *data, size_t n, FL a = 0, FL b = 1) {
        uniform_real_distribution<FL> distr(a, b);
        for (size_t i = 0; i < n * 2; i++)
            ((FL *)data)[i] = distr(rng());
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
        if (!remove_empty || s.length() > last)
            r.push_back(s.substr(last, s.length() - last));
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
    template <typename T> static string to_string(T i) {
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
            string prefix = "KMGTPEZYB";
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
#ifdef _WIN32
        if (file_exists(old_name) && file_exists(new_name))
            remove_file(new_name);
#endif
        return rename(old_name.c_str(), new_name.c_str()) == 0;
    }
    static bool remove_file(const string &name) {
        return remove(name.c_str()) == 0;
    }
    static string get_filename(const string &path) {
        size_t idx = path.find_last_of("/\\");
        if (idx != string::npos)
            return path.substr(idx + 1);
        else
            return path;
    }
    static string get_pathname(const string &path) {
        size_t idx = path.find_last_of("/\\");
        if (idx != string::npos)
            return path.substr(0, idx);
        else
            return "";
    }
    static string read_link(const string &name) {
#ifdef _WIN32
        return "";
#else
        char buf[255];
        ssize_t cnt = readlink(name.c_str(), buf, 255);
        return string(buf, cnt);
#endif
    }
    static bool link_file(const string &source, const string &name) {
#ifdef _WIN32
        copy_file(source, name);
        return true;
#else
        if (link_exists(source)) {
            string real_src = read_link(source);
            if (get_filename(name) == real_src)
                return true;
        }
        if (file_exists(name))
            remove_file(name);
        assert(get_pathname(source) == get_pathname(name));
        return symlink(get_filename(source).c_str(), name.c_str()) == 0;
#endif
    }
    static void copy_file(const string &source, const string &dest) {
        ifstream ifs(source.c_str(), ios::binary);
        if (!ifs.good())
            throw runtime_error("copy_file source '" + source + "' failed.");
        ofstream ofs(dest.c_str(), ios::binary);
        if (!ofs.good())
            throw runtime_error("copy_file dest '" + dest + "' failed.");
        ofs << ifs.rdbuf();
        if (ifs.fail() || ifs.bad())
            throw runtime_error("copy_file source '" + source + "' failed.");
        if (!ofs.good())
            throw runtime_error("copy_file dest '" + dest + "' failed.");
        ifs.close();
        ofs.close();
    }
    static bool link_exists(const string &name) {
#ifdef _WIN32
        return false;
#else
        struct stat buffer;
        return lstat(name.c_str(), &buffer) == 0 &&
               (buffer.st_mode & S_IFLNK) == S_IFLNK;
#endif
    }
    static bool file_exists(const string &name) {
        struct stat buffer;
        return stat(name.c_str(), &buffer) == 0;
    }
    static bool path_exists(const string &name) {
        struct stat buffer;
        return stat(name.c_str(), &buffer) == 0 && (buffer.st_mode & S_IFDIR);
    }
    static void mkdir(const string &name) { _mkdir(name.c_str()); }
};

} // namespace block2
