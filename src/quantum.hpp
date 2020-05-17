
#ifndef QUANTUM_HPP_
#define QUANTUM_HPP_

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <execinfo.h>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mkl.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#define TINY 1E-20

using namespace std;

namespace block2 {

inline void print_trace(int n_sig) {
    printf("print_trace: got signal %d\n", n_sig);

    void *array[32];
    size_t size = backtrace(array, 32);
    char **strings = backtrace_symbols(array, size);

    for (size_t i = 0; i < size; i++)
        fprintf(stderr, "%s\n", strings[i]);

    abort();
}

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
            print_trace(11);
            return 0;
        } else
            return data + (used += n) - n;
    }
    void deallocate(void *ptr, size_t n) {
        if (n == 0)
            return;
        if (used < n || ptr != data + used - n) {
            cout << "deallocation not happening in reverse order" << endl;
            print_trace(11);
        } else
            used -= n;
    }
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

inline StackAllocator<uint32_t>*& _ialloc() {
    static StackAllocator<uint32_t> *ialloc;
    return ialloc;
}

inline StackAllocator<double>*& _dalloc() {
    static StackAllocator<double> *dalloc;
    return dalloc;
}

#define ialloc (_ialloc())
#define dalloc (_dalloc())

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
    static int to_int(const string &x) { return atoi(x.c_str()); }
    static double to_double(const string &x) { return atof(x.c_str()); }
    static string to_string(int i) {
        stringstream ss;
        ss << i;
        return ss.str();
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

struct DataFrame {
    string save_dir, prefix = "F0";
    size_t isize, dsize;
    uint16_t n_frames, i_frame;
    vector<StackAllocator<uint32_t>> iallocs;
    vector<StackAllocator<double>> dallocs;
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
    void load_data(uint16_t i, const string &filename) const {
        ifstream ifs(filename.c_str(), ios::binary);
        ifs.read((char *)&dallocs[i].used, sizeof(dallocs[i].used));
        ifs.read((char *)dallocs[i].data, sizeof(double) * dallocs[i].used);
        ifs.read((char *)&iallocs[i].used, sizeof(iallocs[i].used));
        ifs.read((char *)iallocs[i].data, sizeof(uint32_t) * iallocs[i].used);
        ifs.close();
    }
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

inline DataFrame *& _frame() {
    static DataFrame *frame;
    return frame;
}

#define frame (_frame())

struct MatrixRef {
    int m, n;
    double *data;
    MatrixRef(double *data, int m, int n) : data(data), m(m), n(n) {}
    double &operator()(int i, int j) const {
        return *(data + (size_t)i * n + j);
    }
    size_t size() const { return (size_t)m * n; }
    void allocate() { data = dalloc->allocate(size()); }
    void deallocate() { dalloc->deallocate(data, size()), data = nullptr; }
    void clear() { memset(data, 0, size() * sizeof(double)); }
    MatrixRef flip_dims() const { return MatrixRef(data, n, m); }
    MatrixRef shift_ptr(size_t l) const { return MatrixRef(data + l, m, n); }
    friend ostream &operator<<(ostream &os, const MatrixRef &mat) {
        os << "MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        for (int i = 0; i < mat.m; i++) {
            os << "[ ";
            for (int j = 0; j < mat.n; j++)
                os << setw(20) << setprecision(14) << mat(i, j) << " ";
            os << "]" << endl;
        }
        return os;
    }
    double trace() const {
        assert(m == n);
        double r = 0;
        for (int i = 0; i < m; i++)
            r += this->operator()(i, i);
        return r;
    }
};

struct DiagonalMatrix : MatrixRef {
    double zero = 0.0;
    DiagonalMatrix(double *data, int n) : MatrixRef(data, n, n) {}
    double &operator()(int i, int j) const {
        return i == j ? *(data + i) : const_cast<double &>(zero);
    }
    size_t size() const { return (size_t)m; }
    void allocate() { data = dalloc->allocate(size()); }
    void deallocate() { dalloc->deallocate(data, size()), data = nullptr; }
    void clear() { memset(data, 0, size() * sizeof(double)); }
    friend ostream &operator<<(ostream &os, const DiagonalMatrix &mat) {
        os << "DIAG MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        os << "[ ";
        for (int j = 0; j < mat.n; j++)
            os << setw(20) << setprecision(14) << mat(j, j) << " ";
        os << "]" << endl;
        return os;
    }
};

struct IdentityMatrix : DiagonalMatrix {
    double one = 1.0;
    IdentityMatrix(int n) : DiagonalMatrix(nullptr, n) {}
    double &operator()(int i, int j) const {
        return i == j ? const_cast<double &>(one) : const_cast<double &>(zero);
    }
    void allocate() {}
    void deallocate() {}
    void clear() {}
    friend ostream &operator<<(ostream &os, const IdentityMatrix &mat) {
        os << "IDENT MAT ( " << mat.m << "x" << mat.n << " )" << endl;
        return os;
    }
};

struct TInt {
    uint16_t n;
    double *data;
    TInt(uint16_t n) : n(n), data(nullptr) {}
    uint32_t find_index(uint16_t i, uint16_t j) const {
        return i < j ? ((uint32_t)j * (j + 1) >> 1) + i
                     : ((uint32_t)i * (i + 1) >> 1) + j;
    }
    size_t size() const { return ((size_t)n * (n + 1) >> 1); }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j) {
        return *(data + find_index(i, j));
    }
    double operator()(uint16_t i, uint16_t j) const {
        return *(data + find_index(i, j));
    }
};

struct V1Int {
    uint32_t n;
    size_t m;
    double *data;
    V1Int(uint32_t n) : n(n), m((size_t)n * n * n * n), data(nullptr) {}
    size_t size() const { return m; }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + (((size_t)i * n + j) * n + k) * n + l);
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return *(data + (((size_t)i * n + j) * n + k) * n + l);
    }
};

struct V8Int {
    uint32_t n, m;
    double *data;
    V8Int(uint32_t n) : n(n), m(n * (n + 1) >> 1), data(nullptr) {}
    size_t find_index(uint32_t i, uint32_t j) const {
        return i < j ? ((size_t)j * (j + 1) >> 1) + i
                     : ((size_t)i * (i + 1) >> 1) + j;
    }
    size_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        uint32_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return find_index(p, q);
    }
    size_t size() const { return ((size_t)m * (m + 1) >> 1); }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + find_index(i, j, k, l));
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return *(data + find_index(i, j, k, l));
    }
};

struct V4Int {
    uint32_t n, m;
    double *data;
    V4Int(uint32_t n) : n(n), m(n * (n + 1) >> 1), data(nullptr) {}
    size_t find_index(uint32_t i, uint32_t j) const {
        return i < j ? ((size_t)j * (j + 1) >> 1) + i
                     : ((size_t)i * (i + 1) >> 1) + j;
    }
    size_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        size_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return p * m + q;
    }
    size_t size() const { return (size_t)m * m; }
    void clear() { memset(data, 0, sizeof(double) * size()); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return *(data + find_index(i, j, k, l));
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return *(data + find_index(i, j, k, l));
    }
};

inline vector<double> read_occ(const string &filename) {
    assert(Parsing::file_exists(filename));
    ifstream ifs(filename.c_str());
    vector<string> lines = Parsing::readlines(&ifs);
    assert(lines.size() >= 1);
    vector<string> vals = Parsing::split(lines[0], " ", true);
    vector<double> r;
    transform(vals.begin(), vals.end(), back_inserter(r), Parsing::to_double);
    return r;
}

struct FCIDUMP {
    map<string, string> params;
    vector<TInt> ts;
    vector<V8Int> vs;
    vector<V4Int> vabs;
    vector<V1Int> vgs;
    double e;
    double *data;
    size_t total_memory;
    bool uhf, general;
    FCIDUMP() : e(0.0), uhf(false), total_memory(0) {}
    void initialize_su2(uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                        uint16_t isym, double e, const double *t, size_t lt,
                        const double *v, size_t lv) {
        params.clear();
        ts.clear();
        vs.clear();
        vabs.clear();
        this->e = e;
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_elec);
        params["ms2"] = Parsing::to_string(twos);
        params["isym"] = Parsing::to_string(isym);
        params["iuhf"] = "0";
        ts.push_back(TInt(n_sites));
        vs.push_back(V8Int(n_sites));
        if (vs[0].size() == lv) {
            general = false;
            total_memory = ts[0].size() + vs[0].size();
            data = dalloc->allocate(total_memory);
            ts[0].data = data;
            vs[0].data = data + ts[0].size();
            memcpy(vs[0].data, v, sizeof(double) * lv);
        } else {
            general = true;
            vs.clear();
            vgs.push_back(V1Int(n_sites));
            assert(lv == vgs[0].size());
            total_memory = ts[0].size() + vgs[0].size();
            data = dalloc->allocate(total_memory);
            ts[0].data = data;
            vgs[0].data = data + ts[0].size();
            memcpy(vgs[0].data, v, sizeof(double) * lv);
        }
        assert(lt == ts[0].size());
        memcpy(ts[0].data, t, sizeof(double) * lt);
        uhf = false;
    }
    void read(const string &filename) {
        params.clear();
        ts.clear();
        vs.clear();
        vabs.clear();
        e = 0.0;
        assert(Parsing::file_exists(filename));
        ifstream ifs(filename.c_str());
        vector<string> lines = Parsing::readlines(&ifs);
        ifs.close();
        bool ipar = true;
        vector<string> pars, ints;
        for (size_t il = 0; il < lines.size(); il++) {
            string l(Parsing::lower(lines[il]));
            if (l.find("&fci") != string::npos)
                l.replace(l.find("&fci"), 4, "");
            if (l.find("/") != string::npos || l.find("&end") != string::npos)
                ipar = false;
            else if (ipar)
                pars.push_back(l);
            else
                ints.push_back(l);
        }
        string par = Parsing::join(pars.begin(), pars.end(), ",");
        for (size_t ip = 0; ip < par.length(); ip++)
            if (par[ip] == ' ')
                par[ip] = ',';
        pars = Parsing::split(par, ",", true);
        string p_key = "";
        for (auto &c : pars) {
            if (c.find("=") != string::npos || p_key.length() == 0) {
                vector<string> cs = Parsing::split(c, "=", true);
                p_key = Parsing::trim(cs[0]);
                params[p_key] = cs.size() == 2 ? Parsing::trim(cs[1]) : "";
            } else {
                string cc = Parsing::trim(c);
                if (cc.length() != 0)
                    params[p_key] = params[p_key].length() == 0
                                        ? cc
                                        : params[p_key] + "," + cc;
            }
        }
        vector<array<uint16_t, 4>> int_idx;
        vector<double> int_val;
        for (auto &l : ints) {
            string ll = Parsing::trim(l);
            if (ll.length() == 0 || ll[0] == '!')
                continue;
            vector<string> ls = Parsing::split(ll, " ", true);
            assert(ls.size() == 5);
            int_idx.push_back({(uint16_t)Parsing::to_int(ls[1]),
                               (uint16_t)Parsing::to_int(ls[2]),
                               (uint16_t)Parsing::to_int(ls[3]),
                               (uint16_t)Parsing::to_int(ls[4])});
            int_val.push_back(Parsing::to_double(ls[0]));
        }
        uint16_t n = (uint16_t)Parsing::to_int(params["norb"]);
        uhf = params.count("iuhf") != 0 && Parsing::to_int(params["iuhf"]) == 1;
        general = params.count("igeneral") != 0 &&
                  Parsing::to_int(params["igeneral"]) == 1;
        if (!uhf) {
            ts.push_back(TInt(n));
            if (!general) {
                vs.push_back(V8Int(n));
                total_memory = ts[0].size() + vs[0].size();
                data = dalloc->allocate(total_memory);
                ts[0].data = data;
                vs[0].data = data + ts[0].size();
                ts[0].clear();
                vs[0].clear();
            } else {
                vgs.push_back(V1Int(n));
                total_memory = ts[0].size() + vgs[0].size();
                data = dalloc->allocate(total_memory);
                ts[0].data = data;
                vgs[0].data = data + ts[0].size();
                ts[0].clear();
                vgs[0].clear();
            }
            for (size_t i = 0; i < int_val.size(); i++) {
                if (int_idx[i][0] + int_idx[i][1] + int_idx[i][2] +
                        int_idx[i][3] ==
                    0)
                    e = int_val[i];
                else if (int_idx[i][2] + int_idx[i][3] == 0)
                    ts[0](int_idx[i][0] - 1, int_idx[i][1] - 1) = int_val[i];
                else if (!general)
                    vs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                          int_idx[i][2] - 1, int_idx[i][3] - 1) = int_val[i];
                else
                    vgs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                           int_idx[i][2] - 1, int_idx[i][3] - 1) = int_val[i];
            }
        } else {
            ts.push_back(TInt(n));
            ts.push_back(TInt(n));
            if (!general) {
                vs.push_back(V8Int(n));
                vs.push_back(V8Int(n));
                vabs.push_back(V4Int(n));
                total_memory =
                    ((ts[0].size() + vs[0].size()) << 1) + vabs[0].size();
                data = dalloc->allocate(total_memory);
                ts[0].data = data;
                ts[1].data = data + ts[0].size();
                vs[0].data = data + (ts[0].size() << 1);
                vs[1].data = data + (ts[0].size() << 1) + vs[0].size();
                vabs[0].data = data + ((ts[0].size() + vs[0].size()) << 1);
                ts[0].clear(), ts[1].clear();
                vs[0].clear(), vs[1].clear(), vabs[0].clear();
            } else {
                for (int i = 0; i < 3; i++)
                    vgs.push_back(V1Int(n));
                total_memory = ts[0].size() * 2 + vgs[0].size() * 3;
                data = dalloc->allocate(total_memory);
                ts[0].data = data;
                ts[1].data = data + ts[0].size();
                vgs[0].data = data + (ts[0].size() << 1);
                vgs[1].data = data + (ts[0].size() << 1) + vgs[0].size();
                vgs[2].data = data + (ts[0].size() << 1) + (vgs[0].size() << 1);
                ts[0].clear(), ts[1].clear();
                vgs[0].clear(), vgs[1].clear(), vgs[2].clear();
            }
            int ip = 0;
            for (size_t i = 0; i < int_val.size(); i++) {
                if (int_idx[i][0] + int_idx[i][1] + int_idx[i][2] +
                        int_idx[i][3] ==
                    0) {
                    ip++;
                    if (ip == 6)
                        e = int_val[i];
                } else if (int_idx[i][2] + int_idx[i][3] == 0) {
                    ts[ip - 3](int_idx[i][0] - 1, int_idx[i][1] - 1) =
                        int_val[i];
                } else {
                    assert(ip <= 2);
                    if (!general) {
                        if (ip < 2)
                            vs[ip](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                   int_idx[i][2] - 1, int_idx[i][3] - 1) =
                                int_val[i];
                        else
                            vabs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                    int_idx[i][2] - 1, int_idx[i][3] - 1) =
                                int_val[i];
                    } else {
                        vgs[ip](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                int_idx[i][2] - 1, int_idx[i][3] - 1) =
                            int_val[i];
                    }
                }
            }
        }
    }
    uint16_t twos() const {
        return (uint16_t)Parsing::to_int(params.at("ms2"));
    }
    uint16_t n_sites() const {
        return (uint16_t)Parsing::to_int(params.at("norb"));
    }
    uint16_t n_elec() const {
        return (uint16_t)Parsing::to_int(params.at("nelec"));
    }
    uint8_t isym() const { return (uint8_t)Parsing::to_int(params.at("isym")); }
    vector<uint8_t> orb_sym() const {
        vector<string> x = Parsing::split(params.at("orbsym"), ",", true);
        vector<uint8_t> r;
        r.reserve(x.size());
        for (auto &xx : x)
            r.push_back((uint8_t)Parsing::to_int(xx));
        return r;
    }
    double t(uint8_t i, uint8_t j) const { return ts[0](i, j); }
    double t(uint8_t s, uint8_t i, uint8_t j) const {
        return uhf ? ts[s](i, j) : ts[0](i, j);
    }
    double v(uint8_t i, uint8_t j, uint8_t k, uint8_t l) const {
        return general ? vgs[0](i, j, k, l) : vs[0](i, j, k, l);
    }
    double v(uint8_t sl, uint8_t sr, uint8_t i, uint8_t j, uint8_t k,
             uint8_t l) const {
        if (uhf) {
            if (sl == sr)
                return general ? vgs[sl](i, j, k, l) : vs[sl](i, j, k, l);
            else if (sl == 0 && sr == 1)
                return general ? vgs[2](i, j, k, l) : vabs[0](i, j, k, l);
            else
                return general ? vgs[2](k, l, i, j) : vabs[0](k, l, i, j);
        } else
            return general ? vgs[0](i, j, k, l) : vs[0](i, j, k, l);
    }
    void deallocate() {
        assert(total_memory != 0);
        dalloc->deallocate(data, total_memory);
        data = nullptr;
        ts.clear();
        vs.clear();
        vabs.clear();
        vgs.clear();
    }
};

template <typename, typename = void> struct CG;

template <typename S> struct CG<S, typename S::is_sz_t> {
    CG() {}
    CG(int n_sqrt_fact) {}
    void initialize(double *ptr = 0) {}
    void deallocate() {}
    long double wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                          int tjf) const noexcept {
        return 1.0L;
    }
    long double wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf,
                          int tjg, int tjh, int tji) const noexcept {
        return 1.0L;
    }
    long double racah(int ta, int tb, int tc, int td, int te, int tf) const
        noexcept {
        return 1.0L;
    }
    long double transpose_cg(int td, int tl, int tr) const noexcept {
        return 1.0L;
    }
};

template <typename S> struct CG<S, typename S::is_su2_t> {
    long double *sqrt_fact;
    int n_sf;
    CG() : n_sf(0), sqrt_fact(nullptr) {}
    CG(int n_sqrt_fact) : n_sf(n_sqrt_fact) {}
    void initialize(double *ptr = 0) {
        assert(n_sf != 0);
        if (ptr == 0)
            ptr = dalloc->allocate(n_sf * 2);
        sqrt_fact = (long double *)ptr;
        sqrt_fact[0] = 1;
        for (int i = 1; i < n_sf; i++)
            sqrt_fact[i] = sqrt_fact[i - 1] * sqrtl(i);
    }
    void deallocate() {
        assert(n_sf != 0);
        dalloc->deallocate(sqrt_fact, n_sf * 2);
        sqrt_fact = nullptr;
    }
    static bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    long double sqrt_delta(int tja, int tjb, int tjc) const {
        return sqrt_fact[(tja + tjb - tjc) >> 1] *
               sqrt_fact[(tja - tjb + tjc) >> 1] *
               sqrt_fact[(-tja + tjb + tjc) >> 1] /
               sqrt_fact[(tja + tjb + tjc + 2) >> 1];
    }
    long double cg(int tja, int tjb, int tjc, int tma, int tmb, int tmc) const {
        return (1 - ((tmc + tja - tjb) & 2)) * sqrt(tjc + 1) *
               wigner_3j(tja, tjb, tjc, tma, tmb, -tmc);
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.21)
    // Adapted from Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_3j(int tja, int tjb, int tjc, int tma, int tmb,
                          int tmc) const {
        if (tma + tmb + tmc != 0 || !triangle(tja, tjb, tjc) ||
            ((tja + tma) & 1) || ((tjb + tmb) & 1) || ((tjc + tmc) & 1))
            return 0;
        const int alpha1 = (tjb - tjc - tma) >> 1,
                  alpha2 = (tja - tjc + tmb) >> 1;
        const int beta1 = (tja + tjb - tjc) >> 1, beta2 = (tja - tma) >> 1,
                  beta3 = (tjb + tmb) >> 1;
        const int max_alpha = max(0, max(alpha1, alpha2));
        const int min_beta = min(beta1, min(beta2, beta3));
        if (max_alpha > min_beta)
            return 0;
        long double factor =
            (1 - ((tja - tjb - tmc) & 2)) * ((max_alpha & 1) ? -1 : 1) *
            sqrt_delta(tja, tjb, tjc) * sqrt_fact[(tja + tma) >> 1] *
            sqrt_fact[(tja - tma) >> 1] * sqrt_fact[(tjb + tmb) >> 1] *
            sqrt_fact[(tjb - tmb) >> 1] * sqrt_fact[(tjc + tmc) >> 1] *
            sqrt_fact[(tjc - tmc) >> 1];
        long double r = 0, rst;
        for (int t = max_alpha; t <= min_beta; ++t, factor = -factor) {
            rst = sqrt_fact[t] * sqrt_fact[t - alpha1] * sqrt_fact[t - alpha2] *
                  sqrt_fact[beta1 - t] * sqrt_fact[beta2 - t] *
                  sqrt_fact[beta3 - t];
            r += factor / (rst * rst);
        }
        return r;
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.36)
    // Adapted from Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                          int tjf) const {
        if (!triangle(tja, tjb, tjc) || !triangle(tja, tje, tjf) ||
            !triangle(tjd, tjb, tjf) || !triangle(tjd, tje, tjc))
            return 0;
        const int alpha1 = (tja + tjb + tjc) >> 1,
                  alpha2 = (tja + tje + tjf) >> 1,
                  alpha3 = (tjd + tjb + tjf) >> 1,
                  alpha4 = (tjd + tje + tjc) >> 1;
        const int beta1 = (tja + tjb + tjd + tje) >> 1,
                  beta2 = (tjb + tjc + tje + tjf) >> 1,
                  beta3 = (tja + tjc + tjd + tjf) >> 1;
        const int max_alpha = max(alpha1, max(alpha2, max(alpha3, alpha4)));
        const int min_beta = min(beta1, min(beta2, beta3));
        if (max_alpha > min_beta)
            return 0;
        long double factor =
            ((max_alpha & 1) ? -1 : 1) * sqrt_delta(tja, tjb, tjc) *
            sqrt_delta(tja, tje, tjf) * sqrt_delta(tjd, tjb, tjf) *
            sqrt_delta(tjd, tje, tjc);
        long double r = 0, rst;
        for (int t = max_alpha; t <= min_beta; ++t, factor = -factor) {
            rst = sqrt_fact[t - alpha1] * sqrt_fact[t - alpha2] *
                  sqrt_fact[t - alpha3] * sqrt_fact[t - alpha4] *
                  sqrt_fact[beta1 - t] * sqrt_fact[beta2 - t] *
                  sqrt_fact[beta3 - t];
            r += factor * sqrt_fact[t + 1] * sqrt_fact[t + 1] / (rst * rst);
        }
        return r;
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.41)
    // Adapted from Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf,
                          int tjg, int tjh, int tji) const {
        if (!triangle(tja, tjb, tjc) || !triangle(tjd, tje, tjf) ||
            !triangle(tjg, tjh, tji) || !triangle(tja, tjd, tjg) ||
            !triangle(tjb, tje, tjh) || !triangle(tjc, tjf, tji))
            return 0;
        const int alpha1 = abs(tja - tji), alpha2 = abs(tjd - tjh),
                  alpha3 = abs(tjb - tjf);
        const int beta1 = tja + tji, beta2 = tjd + tjh, beta3 = tjb + tjf;
        const int max_alpha = max(alpha1, max(alpha2, alpha3));
        const int min_beta = min(beta1, min(beta2, beta3));
        long double r = 0;
        for (int tg = max_alpha; tg <= min_beta; tg += 2) {
            r += (tg + 1) * wigner_6j(tja, tjb, tjc, tjf, tji, tg) *
                 wigner_6j(tjd, tje, tjf, tjb, tg, tjh) *
                 wigner_6j(tjg, tjh, tji, tg, tja, tjd);
        }
        return ((max_alpha & 1) ? -1 : 1) * r;
    }
    // D.M. Brink, G.R. Satchler. Angular Momentum. P142
    long double racah(int ta, int tb, int tc, int td, int te, int tf) {
        return (1 - ((ta + tb + tc + td) & 2)) *
               wigner_6j(ta, tb, te, td, tc, tf);
    }
    long double transpose_cg(int td, int tl, int tr) {
        return (1 - ((td + tl - tr) & 2)) * sqrtl(tr + 1) / sqrtl(tl + 1);
    }
};

struct SZ {
    typedef void is_sz_t;
    uint32_t data;
    SZ() : data(0) {}
    SZ(uint32_t data) : data(data) {}
    SZ(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | ((uint8_t)twos << 8) | pg)) {}
    int n() const { return (int)(((int32_t)data) >> 24); }
    int twos() const { return (int)(int8_t)((data >> 8) & 0xFFU); }
    int pg() const { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data =
            (data & (~0xFFFF00U)) | ((uint32_t)(((uint8_t)twos & 0xFFU) << 8));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return 1; }
    bool is_fermion() const noexcept { return twos() & 1; }
    bool operator==(SZ other) const noexcept { return data == other.data; }
    bool operator!=(SZ other) const noexcept { return data != other.data; }
    bool operator<(SZ other) const noexcept { return data < other.data; }
    SZ operator-() const noexcept {
        return SZ((data & 0xFFU) | (((~data) + (1 << 8)) & 0xFF00U) |
                  (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SZ operator-(SZ other) const noexcept { return *this + (-other); }
    SZ operator+(SZ other) const noexcept {
        return SZ((((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) &
                   0xFF00FF00U) |
                  ((data & 0xFFU) ^ (other.data & 0xFFU)));
    }
    SZ operator[](int i) const noexcept { return *this; }
    SZ get_ket() const noexcept { return *this; }
    SZ get_bra(SZ dq) const noexcept { return *this + dq; }
    SZ combine(SZ bra, SZ ket) const {
        return ket + *this == bra ? ket : SZ(0xFFFFFFFFU);
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept { return 1; }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " SZ=";
        if (twos() & 1)
            ss << twos() << "/2";
        else
            ss << (twos() >> 1);
        ss << " PG=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SZ c) {
        os << c.to_str();
        return os;
    }
};

struct SU2 {
    typedef void is_su2_t;
    uint32_t data;
    SU2() : data(0) {}
    SU2(uint32_t data) : data(data) {}
    SU2(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos << 16) | (twos << 8) | pg)) {}
    SU2(int n, int twos_low, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos_low << 16) | (twos << 8) | pg)) {}
    int n() const noexcept { return (int)(((int32_t)data) >> 24); }
    int twos() const noexcept { return (int)(int16_t)((data >> 8) & 0xFFU); }
    int twos_low() const noexcept {
        return (int)(int16_t)((data >> 16) & 0xFFU);
    }
    int pg() const noexcept { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data = (data & (~0xFFFF00U)) | ((uint32_t)((twos & 0xFFU) << 16)) |
               ((uint32_t)((twos & 0xFFU) << 8));
    }
    void set_twos_low(int twos) {
        data = (data & (~0xFF0000U)) | ((uint32_t)((twos & 0xFFU) << 16));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    int multiplicity() const noexcept { return twos() + 1; }
    bool is_fermion() const noexcept { return twos() & 1; }
    bool operator==(SU2 other) const noexcept { return data == other.data; }
    bool operator!=(SU2 other) const noexcept { return data != other.data; }
    bool operator<(SU2 other) const noexcept { return data < other.data; }
    SU2 operator-() const noexcept {
        return SU2((data & 0xFFFFFFU) | (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SU2 operator-(SU2 other) const noexcept { return *this + (-other); }
    SU2 operator+(SU2 other) const noexcept {
        uint32_t add_data =
            ((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) |
            ((data & 0xFFU) ^ (other.data & 0xFFU));
        uint32_t sub_data_lr =
            ((data & 0xFF00U) << 8) - (other.data & 0xFF0000U);
        sub_data_lr =
            (((sub_data_lr >> 7) + sub_data_lr) ^ (sub_data_lr >> 7)) &
            0xFF0000U;
        uint32_t sub_data_rl =
            ((other.data & 0xFF00U) << 8) - (data & 0xFF0000U);
        sub_data_rl =
            (((sub_data_rl >> 7) + sub_data_rl) ^ (sub_data_rl >> 7)) &
            0xFF0000U;
        return SU2(add_data | min(sub_data_lr, sub_data_rl));
    }
    SU2 operator[](int i) const noexcept {
        return SU2(((data + (i << 17)) & (~0x00FF00U)) |
                   (((data + (i << 17)) & 0xFF0000U) >> 8));
    }
    SU2 get_ket() const noexcept {
        return SU2((data & 0xFF00FFFFU) | ((data & 0xFF00U) << 8));
    }
    SU2 get_bra(SU2 dq) const noexcept {
        return SU2(((data & 0xFF000000U) + (dq.data & 0xFF000000U)) |
                   ((data & 0xFF0000U) >> 8) | (data & 0xFF0000U) |
                   ((data & 0xFFU) ^ (dq.data & 0xFFU)));
    }
    SU2 combine(SU2 bra, SU2 ket) const {
        ket.set_twos_low(bra.twos());
        if (ket.get_bra(*this) != bra ||
            !CG<SU2>::triangle(ket.twos(), this->twos(), bra.twos()))
            return SU2(0xFFFFFFFFU);
        return ket;
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const noexcept {
        return (int)(((data >> 9) - (data >> 17)) & 0x7FU) + 1;
    }
    string to_str() const {
        stringstream ss;
        ss << "< N=" << n() << " S=";
        if (twos_low() != twos()) {
            if (twos_low() & 1)
                ss << twos_low() << "/2~";
            else
                ss << (twos_low() >> 1) << "~";
        }
        if (twos() & 1)
            ss << twos() << "/2";
        else
            ss << (twos() >> 1);
        ss << " PG=" << pg() << " >";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SU2 c) {
        os << c.to_str();
        return os;
    }
};

enum struct OpNames : uint8_t {
    H,
    I,
    N,
    NN,
    C,
    D,
    R,
    RD,
    A,
    AD,
    P,
    PD,
    B,
    Q,
    Zero,
    PDM1
};

inline ostream &operator<<(ostream &os, const OpNames c) {
    const static string repr[] = {"H", "I",  "N",    "NN",  "C", "D",
                                  "R", "RD", "A",    "AD",  "P", "PD",
                                  "B", "Q",  "Zero", "PDM1"};
    os << repr[(uint8_t)c];
    return os;
}

enum struct OpTypes : uint8_t { Zero, Elem, Prod, Sum, ElemRef, SumProd };

template <typename S> struct OpExpr {
    virtual const OpTypes get_type() const { return OpTypes::Zero; }
    bool operator==(const OpExpr &other) const { return true; }
};

struct SiteIndex {
    uint32_t data;
    SiteIndex() : data(0) {}
    SiteIndex(uint32_t data) : data(data) {}
    SiteIndex(uint8_t i) : data(1U | (i << 8)) {}
    SiteIndex(uint8_t i, uint8_t j) : data(2U | 0U | (i << 8) | (j << 16)) {}
    SiteIndex(uint8_t i, uint8_t j, uint8_t s)
        : data(2U | 4U | (i << 8) | (j << 16) | (s << 4)) {}
    SiteIndex(const initializer_list<uint8_t> i,
              const initializer_list<uint8_t> s)
        : data(0) {
        data |= i.size() | (s.size() << 2);
        int x = 8;
        for (auto iit = i.begin(); iit != i.end(); iit++, x += 8)
            data |= (*iit) << x;
        x = 4;
        for (auto sit = s.begin(); sit != s.end(); sit++, x++)
            data |= (*sit) << x;
    }
    uint8_t size() const noexcept { return (uint8_t)(data & 3); }
    uint8_t spin_size() const noexcept { return (uint8_t)((data >> 2) & 3); }
    uint8_t ss() const noexcept { return (data >> 4) & 0xFU; }
    uint8_t s(uint8_t i = 0) const noexcept {
        return !!(data & (1U << (4 + i)));
    }
    uint8_t operator[](uint8_t i) const noexcept {
        return (data >> (1U << (3 + i))) & 0xFFU;
    }
    bool operator==(SiteIndex other) const noexcept {
        return data == other.data;
    }
    bool operator!=(SiteIndex other) const noexcept {
        return data != other.data;
    }
    bool operator<(SiteIndex other) const noexcept { return data < other.data; }
    SiteIndex flip_spatial() const noexcept {
        return SiteIndex((uint32_t)((data & (0xFF0000FFU)) |
                                    ((*this)[0] << 16) | ((*this)[1] << 8)));
    }
    SiteIndex flip() const noexcept {
        return SiteIndex((uint32_t)((data & (0xFF00000FU)) | (s(0) << 5) |
                                    (s(1) << 4) | ((*this)[0] << 16) |
                                    ((*this)[1] << 8)));
    }
    size_t hash() const noexcept { return (size_t)data; }
    vector<uint8_t> to_array() const {
        vector<uint8_t> r;
        r.reserve(size() + spin_size());
        for (uint8_t i = 0; i < size(); i++)
            r.push_back((*this)[i]);
        for (uint8_t i = 0; i < spin_size(); i++)
            r.push_back(s(i));
        return r;
    }
    string to_str() const {
        stringstream ss;
        ss << "[ ";
        for (uint8_t i = 0; i < size(); i++)
            ss << (int)(*this)[i] << " ";
        for (uint8_t i = 0; i < spin_size(); i++)
            ss << (int)s(i) << " ";
        ss << "]";
        return ss.str();
    }
    friend ostream &operator<<(ostream &os, SiteIndex c) {
        os << c.to_str();
        return os;
    }
};

template <typename S> struct OpElement : OpExpr<S> {
    OpNames name;
    SiteIndex site_index;
    double factor;
    S q_label;
    OpElement(OpNames name, SiteIndex site_index, S q_label,
              double factor = 1.0)
        : name(name), site_index(site_index), factor(factor), q_label(q_label) {
    }
    const OpTypes get_type() const override { return OpTypes::Elem; }
    OpElement abs() const { return OpElement(name, site_index, q_label, 1.0); }
    OpElement operator*(double d) const {
        return OpElement(name, site_index, q_label, factor * d);
    }
    bool operator==(const OpElement &other) const {
        return name == other.name && site_index == other.site_index &&
               factor == other.factor;
    }
    bool operator<(const OpElement &other) const {
        if (name != other.name)
            return name < other.name;
        else if (site_index != other.site_index)
            return site_index < other.site_index;
        else if (factor != other.factor)
            return factor < other.factor;
        else
            return false;
    }
    size_t hash() const noexcept {
        size_t h = (size_t)name;
        h ^= site_index.hash() + 0x9E3779B9 + (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(factor) + 0x9E3779B9 + (h << 6) + (h >> 2);
        return h;
    }
    friend ostream &operator<<(ostream &os, const OpElement<S> &c) {
        if (c.factor != 1.0)
            os << "(" << c.factor << " " << c.abs() << ")";
        else if (c.site_index.data == 0)
            os << c.name;
        else if (c.site_index.size() == 1 && c.site_index.spin_size() == 0)
            os << c.name << (int)c.site_index[0];
        else
            os << c.name << c.site_index;
        return os;
    }
};

template <typename S> struct OpElementRef : OpExpr<S> {
    shared_ptr<OpElement<S>> op;
    int8_t factor;
    int8_t trans;
    OpElementRef(const shared_ptr<OpElement<S>> &op, int8_t trans,
                 int8_t factor)
        : op(op), trans(trans), factor(factor) {}
    const OpTypes get_type() const override { return OpTypes::ElemRef; }
};

template <typename S> struct OpString : OpExpr<S> {
    shared_ptr<OpElement<S>> a, b;
    double factor;
    uint8_t conj;
    OpString(const shared_ptr<OpElement<S>> &op, double factor,
             uint8_t conj = 0)
        : factor(factor * op->factor), a(make_shared<OpElement<S>>(op->abs())),
          b(nullptr), conj(conj) {}
    OpString(const shared_ptr<OpElement<S>> &a,
             const shared_ptr<OpElement<S>> &b, double factor, uint8_t conj = 0)
        : factor(factor * (a == nullptr ? 1.0 : a->factor) *
                 (b == nullptr ? 1.0 : b->factor)),
          a(a == nullptr ? nullptr : make_shared<OpElement<S>>(a->abs())),
          b(b == nullptr ? nullptr : make_shared<OpElement<S>>(b->abs())),
          conj(conj) {}
    const OpTypes get_type() const override { return OpTypes::Prod; }
    OpString abs() const { return OpString(a, b, 1.0, conj); }
    shared_ptr<OpElement<S>> get_op() const {
        assert(b == nullptr);
        return a;
    }
    OpString operator*(double d) const {
        return OpString(a, b, factor * d, conj);
    }
    bool operator==(const OpString &other) const {
        return *a == *other.a &&
               (b == nullptr ? other.b == nullptr
                             : (other.b != nullptr && *b == *other.b)) &&
               factor == other.factor && conj == other.conj;
    }
    friend ostream &operator<<(ostream &os, const OpString<S> &c) {
        if (c.factor != 1.0)
            os << "(" << c.factor << " " << c.abs() << ")";
        else {
            os << *c.a << (c.conj & 1 ? "^T " : " ");
            if (c.b != nullptr)
                os << *c.b << (c.conj & 2 ? "^T " : " ");
        }
        return os;
    }
};

template <typename S> struct OpSumProd : OpString<S> {
    vector<shared_ptr<OpElement<S>>> ops;
    vector<bool> conjs;
    OpSumProd(const shared_ptr<OpElement<S>> &lop,
              const vector<shared_ptr<OpElement<S>>> &ops,
              const vector<bool> &conjs, double factor, uint8_t conj = 0)
        : ops(ops), conjs(conjs), OpString<S>(lop, nullptr, factor, conj) {}
    OpSumProd(const vector<shared_ptr<OpElement<S>>> &ops,
              const shared_ptr<OpElement<S>> &rop, const vector<bool> &conjs,
              double factor, uint8_t conj = 0)
        : ops(ops), conjs(conjs), OpString<S>(nullptr, rop, factor, conj) {}
    const OpTypes get_type() const override { return OpTypes::SumProd; }
    OpSumProd operator*(double d) const {
        if (OpString<S>::a == nullptr)
            return OpSumProd(ops, OpString<S>::b, conjs,
                             OpString<S>::factor * d, OpString<S>::conj);
        else if (OpString<S>::b == nullptr)
            return OpSumProd(OpString<S>::a, ops, conjs,
                             OpString<S>::factor * d, OpString<S>::conj);
        else
            assert(false);
    }
    bool operator==(const OpSumProd &other) const {
        if (ops.size() != other.ops.size() ||
            (OpString<S>::a == nullptr) != (other.a == nullptr) ||
            (OpString<S>::b == nullptr) != (other.b == nullptr))
            return false;
        else if (OpString<S>::a == nullptr && !(*OpString<S>::b == *other.b))
            return false;
        else if (OpString<S>::b == nullptr && !(*OpString<S>::a == *other.a))
            return false;
        else if (conjs != other.conjs)
            return false;
        else
            for (size_t i = 0; i < ops.size(); i++)
                if (!(*ops[i] == *other.ops[i]))
                    return false;
        return true;
    }
    friend ostream &operator<<(ostream &os, const OpSumProd<S> &c) {
        if (c.ops.size() != 0) {
            if (c.factor != 1.0)
                os << "(" << c.factor << " ";
            if (c.a != nullptr)
                os << *c.a << (c.conj & 1 ? "^T " : " ");
            os << "{ ";
            for (size_t i = 0; i < c.ops.size() - 1; i++)
                os << *c.ops[i] << (c.conjs[i] ? "^T " : " ") << " + ";
            os << *c.ops.back();
            os << " }" << (c.conj & ((c.a != nullptr) + 1) ? "^T" : "");
            if (c.b != nullptr)
                os << " " << *c.b << (c.conj & 2 ? "^T " : " ");
            if (c.factor != 1.0)
                os << " )";
        }
        return os;
    }
};

template <typename S> struct OpSum : OpExpr<S> {
    vector<shared_ptr<OpString<S>>> strings;
    OpSum(const vector<shared_ptr<OpString<S>>> &strings) : strings(strings) {}
    const OpTypes get_type() const override { return OpTypes::Sum; }
    OpSum operator*(double d) const {
        vector<shared_ptr<OpString<S>>> strs;
        strs.reserve(strings.size());
        for (auto &r : strings)
            if (r->get_type() == OpTypes::Prod)
                strs.push_back(make_shared<OpString<S>>(*r * d));
            else
                strs.push_back(make_shared<OpSumProd<S>>(
                    *dynamic_pointer_cast<OpSumProd<S>>(r) * d));
        return OpSum(strs);
    }
    bool operator==(const OpSum &other) const {
        if (strings.size() != other.strings.size())
            return false;
        for (size_t i = 0; i < strings.size(); i++)
            if (!(*strings[i] == *other.strings[i]))
                return false;
        return true;
    }
    friend ostream &operator<<(ostream &os, const OpSum<S> &c) {
        if (c.strings.size() != 0) {
            for (size_t i = 0; i < c.strings.size() - 1; i++)
                if (c.strings[i]->get_type() == OpTypes::Prod)
                    os << *c.strings[i] << " + ";
                else if (c.strings[i]->get_type() == OpTypes::SumProd)
                    os << *dynamic_pointer_cast<OpSumProd<S>>(c.strings[i])
                       << " + ";
            if (c.strings.back()->get_type() == OpTypes::Prod)
                os << *c.strings.back();
            else if (c.strings.back()->get_type() == OpTypes::SumProd)
                os << *dynamic_pointer_cast<OpSumProd<S>>(c.strings.back());
        }
        return os;
    }
};

template <typename S> inline size_t hash_value(const shared_ptr<OpExpr<S>> &x) {
    assert(x->get_type() == OpTypes::Elem);
    return dynamic_pointer_cast<OpElement<S>>(x)->hash();
}

template <typename S>
inline shared_ptr<OpExpr<S>> abs_value(const shared_ptr<OpExpr<S>> &x) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (x->get_type() == OpTypes::Elem) {
        shared_ptr<OpElement<S>> op = dynamic_pointer_cast<OpElement<S>>(x);
        return op->factor == 1.0 ? x : make_shared<OpElement<S>>(op->abs());
    } else if (x->get_type() == OpTypes::Prod) {
        shared_ptr<OpString<S>> op = dynamic_pointer_cast<OpString<S>>(x);
        return op->factor == 1.0 ? x : make_shared<OpString<S>>(op->abs());
    }
    assert(false);
}

template <typename S> inline string to_str(const shared_ptr<OpExpr<S>> &x) {
    stringstream ss;
    if (x->get_type() == OpTypes::Zero)
        ss << 0;
    else if (x->get_type() == OpTypes::Elem)
        ss << *dynamic_pointer_cast<OpElement<S>>(x);
    else if (x->get_type() == OpTypes::Prod)
        ss << *dynamic_pointer_cast<OpString<S>>(x);
    else if (x->get_type() == OpTypes::Sum)
        ss << *dynamic_pointer_cast<OpSum<S>>(x);
    else if (x->get_type() == OpTypes::SumProd)
        ss << *dynamic_pointer_cast<OpSumProd<S>>(x);
    return ss.str();
}

template <typename S>
inline bool operator==(const shared_ptr<OpExpr<S>> &a,
                       const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() != b->get_type())
        return false;
    switch (a->get_type()) {
    case OpTypes::Zero:
        return *a == *b;
    case OpTypes::Elem:
        return *dynamic_pointer_cast<OpElement<S>>(a) ==
               *dynamic_pointer_cast<OpElement<S>>(b);
    case OpTypes::Prod:
        return *dynamic_pointer_cast<OpString<S>>(a) ==
               *dynamic_pointer_cast<OpString<S>>(b);
    case OpTypes::Sum:
        return *dynamic_pointer_cast<OpSum<S>>(a) ==
               *dynamic_pointer_cast<OpSum<S>>(b);
    case OpTypes::SumProd:
        return *dynamic_pointer_cast<OpSumProd<S>>(a) ==
               *dynamic_pointer_cast<OpSumProd<S>>(b);
    default:
        return false;
    }
}

template <typename S> struct op_expr_less {
    bool operator()(const shared_ptr<OpExpr<S>> &a,
                    const shared_ptr<OpExpr<S>> &b) const {
        assert(a->get_type() == OpTypes::Elem &&
               b->get_type() == OpTypes::Elem);
        return *dynamic_pointer_cast<OpElement<S>>(a) <
               *dynamic_pointer_cast<OpElement<S>>(b);
    }
};

template <typename S>
inline const shared_ptr<OpExpr<S>> operator+(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() == OpTypes::Zero)
        return b;
    else if (b->get_type() == OpTypes::Zero)
        return a;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a), 1.0));
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(b), 1.0));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size() + 1);
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a), 1.0));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size() + 1);
            strs.push_back(dynamic_pointer_cast<OpString<S>>(a));
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(2);
            strs.push_back(dynamic_pointer_cast<OpString<S>>(a));
            strs.push_back(dynamic_pointer_cast<OpString<S>>(b));
            return make_shared<OpSum<S>>(strs);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() + 1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(b), 1.0));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() + 1);
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.push_back(dynamic_pointer_cast<OpString<S>>(b));
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size() +
                         dynamic_pointer_cast<OpSum<S>>(b)->strings.size());
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(a)->strings.end());
            strs.insert(strs.end(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.begin(),
                        dynamic_pointer_cast<OpSum<S>>(b)->strings.end());
            return make_shared<OpSum<S>>(strs);
        }
    }
    assert(false);
}

template <typename S>
inline const shared_ptr<OpExpr<S>> operator+=(shared_ptr<OpExpr<S>> &a,
                                              const shared_ptr<OpExpr<S>> &b) {
    return a = a + b;
}

template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(const shared_ptr<OpExpr<S>> &x,
                                             double d) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (d == 0.0)
        return make_shared<OpExpr<S>>();
    else if (d == 1.0)
        return x;
    else if (x->get_type() == OpTypes::Elem)
        return make_shared<OpElement<S>>(
            *dynamic_pointer_cast<OpElement<S>>(x) * d);
    else if (x->get_type() == OpTypes::Prod)
        return make_shared<OpString<S>>(*dynamic_pointer_cast<OpString<S>>(x) *
                                        d);
    else if (x->get_type() == OpTypes::Sum)
        return make_shared<OpSum<S>>(*dynamic_pointer_cast<OpSum<S>>(x) * d);
    assert(false);
}

template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(double d,
                                             const shared_ptr<OpExpr<S>> &x) {
    return x * d;
}

template <typename S>
inline const shared_ptr<OpExpr<S>> operator*(const shared_ptr<OpExpr<S>> &a,
                                             const shared_ptr<OpExpr<S>> &b) {
    if (a->get_type() == OpTypes::Zero)
        return a;
    else if (b->get_type() == OpTypes::Zero)
        return b;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(b)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S>>(b)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpString<S>>(
                    dynamic_pointer_cast<OpElement<S>>(a), r->a, r->factor));
            }
            return make_shared<OpSum<S>>(strs);
        } else if (b->get_type() == OpTypes::Elem)
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a),
                dynamic_pointer_cast<OpElement<S>>(b), 1.0);
        else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OpString<S>>(b)->b == nullptr);
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(a),
                dynamic_pointer_cast<OpString<S>>(b)->a,
                dynamic_pointer_cast<OpString<S>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Elem) {
            assert(dynamic_pointer_cast<OpString<S>>(a)->b == nullptr);
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpString<S>>(a)->a,
                dynamic_pointer_cast<OpElement<S>>(b),
                dynamic_pointer_cast<OpString<S>>(a)->factor);
        } else if (b->get_type() == OpTypes::Prod) {
            assert(dynamic_pointer_cast<OpString<S>>(a)->b == nullptr);
            assert(dynamic_pointer_cast<OpString<S>>(b)->b == nullptr);
            return make_shared<OpString<S>>(
                dynamic_pointer_cast<OpString<S>>(a)->a,
                dynamic_pointer_cast<OpString<S>>(b)->a,
                dynamic_pointer_cast<OpString<S>>(a)->factor *
                    dynamic_pointer_cast<OpString<S>>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString<S>>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(a)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum<S>>(a)->strings) {
                assert(r->b == nullptr);
                strs.push_back(make_shared<OpString<S>>(
                    r->a, dynamic_pointer_cast<OpElement<S>>(b), r->factor));
            }
            return make_shared<OpSum<S>>(strs);
        }
    }
    assert(false);
}

template <typename S>
inline const shared_ptr<OpExpr<S>>
sum(const vector<shared_ptr<OpExpr<S>>> &xs) {
    const static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
    vector<shared_ptr<OpString<S>>> strs;
    for (auto &r : xs)
        if (r->get_type() == OpTypes::Prod)
            strs.push_back(dynamic_pointer_cast<OpString<S>>(r));
        else if (r->get_type() == OpTypes::SumProd)
            strs.push_back(dynamic_pointer_cast<OpSumProd<S>>(r));
        else if (r->get_type() == OpTypes::Elem)
            strs.push_back(make_shared<OpString<S>>(
                dynamic_pointer_cast<OpElement<S>>(r), 1.0));
        else if (r->get_type() == OpTypes::Sum) {
            strs.reserve(dynamic_pointer_cast<OpSum<S>>(r)->strings.size() +
                         strs.size());
            for (auto &rr : dynamic_pointer_cast<OpSum<S>>(r)->strings)
                strs.push_back(rr);
        }
    return strs.size() != 0 ? make_shared<OpSum<S>>(strs) : zero;
}

template <typename S>
inline const shared_ptr<OpExpr<S>>
dot_product(const vector<shared_ptr<OpExpr<S>>> &a,
            const vector<shared_ptr<OpExpr<S>>> &b) {
    vector<shared_ptr<OpExpr<S>>> xs;
    assert(a.size() == b.size());
    for (size_t k = 0; k < a.size(); k++)
        xs.push_back(a[k] * b[k]);
    return sum(xs);
}

template <typename S>
inline ostream &operator<<(ostream &os, const shared_ptr<OpExpr<S>> &c) {
    os << to_str(c);
    return os;
}

} // namespace block2

namespace std {

template <> struct hash<block2::SZ> {
    size_t operator()(const block2::SZ &s) const noexcept { return s.hash(); }
};

template <> struct less<block2::SZ> {
    bool operator()(const block2::SZ &lhs, const block2::SZ &rhs) const
        noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SZ &a, block2::SZ &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::SU2> {
    size_t operator()(const block2::SU2 &s) const noexcept { return s.hash(); }
};

template <> struct less<block2::SU2> {
    bool operator()(const block2::SU2 &lhs, const block2::SU2 &rhs) const
        noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SU2 &a, block2::SU2 &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::OpElement<block2::SU2>> {
    size_t operator()(const block2::OpElement<block2::SU2> &s) const noexcept {
        return s.hash();
    }
};

template <> struct hash<block2::OpElement<block2::SZ>> {
    size_t operator()(const block2::OpElement<block2::SZ> &s) const noexcept {
        return s.hash();
    }
};

} // namespace std

namespace block2 {

enum struct SymTypes : uint8_t { RVec, CVec, Mat };

template <typename S> struct Symbolic {
    int m, n;
    vector<shared_ptr<OpExpr<S>>> data;
    Symbolic(int m, int n) : m(m), n(n), data(){};
    virtual const SymTypes get_type() const = 0;
    virtual shared_ptr<OpExpr<S>> &
    operator[](const initializer_list<int> ix) = 0;
    virtual shared_ptr<Symbolic<S>> copy() const = 0;
};

template <typename S> struct SymbolicRowVector : Symbolic<S> {
    SymbolicRowVector(int n) : Symbolic<S>(1, n) {
        Symbolic<S>::data =
            vector<shared_ptr<OpExpr<S>>>(n, make_shared<OpExpr<S>>());
    }
    const SymTypes get_type() const override { return SymTypes::RVec; }
    shared_ptr<OpExpr<S>> &operator[](int i) { return Symbolic<S>::data[i]; }
    shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) override {
        auto i = ix.begin();
        return (*this)[*(++i)];
    }
    shared_ptr<Symbolic<S>> copy() const override {
        shared_ptr<Symbolic<S>> r =
            make_shared<SymbolicRowVector<S>>(Symbolic<S>::n);
        r->data = Symbolic<S>::data;
        return r;
    }
};

template <typename S> struct SymbolicColumnVector : Symbolic<S> {
    SymbolicColumnVector(int n) : Symbolic<S>(n, 1) {
        Symbolic<S>::data =
            vector<shared_ptr<OpExpr<S>>>(n, make_shared<OpExpr<S>>());
    }
    const SymTypes get_type() const override { return SymTypes::CVec; }
    shared_ptr<OpExpr<S>> &operator[](int i) { return Symbolic<S>::data[i]; }
    shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) override {
        return (*this)[*ix.begin()];
    }
    shared_ptr<Symbolic<S>> copy() const override {
        shared_ptr<Symbolic<S>> r =
            make_shared<SymbolicColumnVector<S>>(Symbolic<S>::m);
        r->data = Symbolic<S>::data;
        return r;
    }
};

template <typename S> struct SymbolicMatrix : Symbolic<S> {
    vector<pair<int, int>> indices;
    SymbolicMatrix(int m, int n) : Symbolic<S>(m, n) {}
    const SymTypes get_type() const override { return SymTypes::Mat; }
    void add(int i, int j, const shared_ptr<OpExpr<S>> elem) {
        indices.push_back(make_pair(i, j));
        Symbolic<S>::data.push_back(elem);
    }
    shared_ptr<OpExpr<S>> &operator[](const initializer_list<int> ix) override {
        auto j = ix.begin(), i = j++;
        add(*i, *j, make_shared<OpExpr<S>>());
        return Symbolic<S>::data.back();
    }
    shared_ptr<Symbolic<S>> copy() const override {
        shared_ptr<SymbolicMatrix<S>> r =
            make_shared<SymbolicMatrix<S>>(Symbolic<S>::m, Symbolic<S>::n);
        r->data = Symbolic<S>::data;
        r->indices = indices;
        return r;
    }
};

template <typename S>
inline ostream &operator<<(ostream &os, const shared_ptr<Symbolic<S>> sym) {
    switch (sym->get_type()) {
    case SymTypes::RVec:
        os << "SymRVector [SIZE= " << sym->n << " ]" << endl;
        for (size_t i = 0; i < sym->data.size(); i++)
            os << "[ " << i << " ] = " << sym->data[i] << endl;
        break;
    case SymTypes::CVec:
        os << "SymCVector [SIZE= " << sym->m << " ]" << endl;
        for (size_t i = 0; i < sym->data.size(); i++)
            os << "[ " << i << " ] = " << sym->data[i] << endl;
        break;
    case SymTypes::Mat: {
        vector<pair<int, int>> &indices =
            dynamic_pointer_cast<SymbolicMatrix<S>>(sym)->indices;
        os << "SymMatrix [SIZE= " << sym->m << "x" << sym->n << " ]" << endl;
        for (size_t i = 0; i < sym->data.size(); i++)
            os << "[ " << indices[i].first << "," << indices[i].second
               << " ] = " << sym->data[i] << endl;
        break;
    }
    default:
        assert(false);
        break;
    }
    return os;
}

template <typename S>
inline const shared_ptr<Symbolic<S>>
operator*(const shared_ptr<Symbolic<S>> a, const shared_ptr<Symbolic<S>> b) {
    assert(a->n == b->m);
    if (a->get_type() == SymTypes::RVec && b->get_type() == SymTypes::Mat) {
        shared_ptr<SymbolicRowVector<S>> r(
            make_shared<SymbolicRowVector<S>>(b->n));
        vector<pair<int, int>> &idx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(b)->indices;
        vector<shared_ptr<OpExpr<S>>> xs[b->n];
        for (size_t k = 0; k < b->data.size(); k++) {
            int i = idx[k].first, j = idx[k].second;
            xs[j].push_back(a->data[i] * b->data[k]);
        }
        for (size_t j = 0; j < b->n; j++)
            (*r)[j] = sum(xs[j]);
        return r;
    } else if (a->get_type() == SymTypes::Mat &&
               b->get_type() == SymTypes::CVec) {
        shared_ptr<SymbolicColumnVector<S>> r(
            make_shared<SymbolicColumnVector<S>>(a->m));
        vector<pair<int, int>> &idx =
            dynamic_pointer_cast<SymbolicMatrix<S>>(a)->indices;
        vector<shared_ptr<OpExpr<S>>> xs[a->m];
        for (size_t k = 0; k < a->data.size(); k++) {
            int i = idx[k].first, j = idx[k].second;
            xs[i].push_back(a->data[k] * b->data[j]);
        }
        for (size_t i = 0; i < a->m; i++)
            (*r)[i] = sum(xs[i]);
        return r;
    } else if (a->get_type() == SymTypes::RVec &&
               b->get_type() == SymTypes::CVec) {
        shared_ptr<SymbolicColumnVector<S>> r(
            make_shared<SymbolicColumnVector<S>>(1));
        (*r)[0] = dot_product(a->data, b->data);
        return r;
    }
    assert(false);
}

template <typename, typename = void> struct StateInfo;

template <typename S>
struct StateInfo<S, typename enable_if<integral_constant<
                        bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    S *quanta;
    uint16_t *n_states;
    int n_states_total, n;
    StateInfo() : quanta(0), n_states(0), n_states_total(0), n(0) {}
    StateInfo(S q) {
        allocate(1);
        quanta[0] = q, n_states[0] = 1, n_states_total = 1;
    }
    void load_data(const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        ifs.read((char *)&n_states_total, sizeof(n_states_total));
        ifs.read((char *)&n, sizeof(n));
        uint32_t *ptr = ialloc->allocate((n << 1) - (n >> 1));
        ifs.read((char *)ptr, sizeof(uint32_t) * ((n << 1) - (n >> 1)));
        ifs.close();
        quanta = (S *)ptr;
        n_states = (uint16_t *)(ptr + n);
    }
    void save_data(const string &filename) const {
        ofstream ofs(filename.c_str(), ios::binary);
        ofs.write((char *)&n_states_total, sizeof(n_states_total));
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)quanta, sizeof(uint32_t) * ((n << 1) - (n >> 1)));
        ofs.close();
    }
    // need length * 2
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) - (length >> 1));
        n = length;
        quanta = (S *)ptr;
        n_states = (uint16_t *)(ptr + length);
    }
    void reallocate(int length) {
        uint32_t *ptr =
            ialloc->reallocate((uint32_t *)quanta, (n << 1) - (n >> 1),
                               (length << 1) - (length >> 1));
        if (ptr == (uint32_t *)quanta) {
            memmove(ptr + length, n_states, length * sizeof(uint16_t));
            n_states = (uint16_t *)(quanta + length);
        } else {
            memmove(ptr, quanta, length * sizeof(uint32_t));
            memmove(ptr + length, n_states, length * sizeof(uint16_t));
            quanta = (S *)ptr;
            n_states = (uint16_t *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        if (n == 0)
            print_trace(11);
        assert(n != 0);
        ialloc->deallocate((uint32_t *)quanta, (n << 1) - (n >> 1));
        quanta = 0;
        n_states = 0;
    }
    StateInfo deep_copy() const {
        StateInfo other;
        other.allocate(n);
        copy_data_to(other);
        other.n_states_total = n_states_total;
        return other;
    }
    void copy_data_to(StateInfo &other) const {
        assert(other.n == n);
        memcpy(other.quanta, quanta, ((n << 1) - (n >> 1)) * sizeof(uint32_t));
    }
    void sort_states() {
        int idx[n];
        S q[n];
        uint16_t nq[n];
        memcpy(q, quanta, n * sizeof(S));
        memcpy(nq, n_states, n * sizeof(uint16_t));
        for (int i = 0; i < n; i++)
            idx[i] = i;
        sort(idx, idx + n, [&q](int i, int j) { return q[i] < q[j]; });
        for (int i = 0; i < n; i++)
            quanta[i] = q[idx[i]], n_states[i] = nq[idx[i]];
        n_states_total = 0;
        for (int i = 0; i < n; i++)
            n_states_total += n_states[i];
    }
    void collect(S target = 0x7FFFFFFF) {
        int k = -1;
        int nn = upper_bound(quanta, quanta + n, target) - quanta;
        for (int i = 0; i < nn; i++)
            if (n_states[i] == 0)
                continue;
            else if (k != -1 && quanta[i] == quanta[k])
                n_states[k] =
                    (uint16_t)min((uint32_t)n_states[k] + n_states[i], 65535U);
            else {
                k++;
                quanta[k] = quanta[i];
                n_states[k] = n_states[i];
            }
        reallocate(k + 1);
        n_states_total = 0;
        for (int i = 0; i < n; i++)
            n_states_total += n_states[i];
    }
    int find_state(S q) const {
        auto p = lower_bound(quanta, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return p - quanta;
    }
    static StateInfo tensor_product(const StateInfo &a, const StateInfo &b,
                                    const StateInfo &cref) {
        StateInfo c;
        c.allocate(cref.n);
        memcpy(c.quanta, cref.quanta, c.n * sizeof(S));
        memset(c.n_states, 0, c.n * sizeof(uint16_t));
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    int ic = c.find_state(qc[k]);
                    if (ic != -1) {
                        uint32_t nprod =
                            (uint32_t)a.n_states[i] * (uint32_t)b.n_states[j] +
                            (uint32_t)c.n_states[ic];
                        c.n_states[ic] = (uint16_t)min(nprod, 65535U);
                    }
                }
            }
        c.collect();
        return c;
    }
    static StateInfo tensor_product(const StateInfo &a, const StateInfo &b,
                                    S target) {
        int nc = 0;
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++)
                nc += (a.quanta[i] + b.quanta[j]).count();
        StateInfo c;
        c.allocate(nc);
        for (int i = 0, ic = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    c.quanta[ic + k] = qc[k];
                    uint32_t nprod =
                        (uint32_t)a.n_states[i] * (uint32_t)b.n_states[j];
                    c.n_states[ic + k] = (uint16_t)min(nprod, 65535U);
                }
                ic += qc.count();
            }
        c.sort_states();
        c.collect(target);
        return c;
    }
    static StateInfo get_connection_info(const StateInfo &a, const StateInfo &b,
                                         const StateInfo &c) {
        map<S, vector<uint32_t>> mp;
        int nc = 0, iab = 0;
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                nc += qc.count();
                for (int k = 0; k < qc.count(); k++)
                    mp[qc[k]].push_back((i << 16) + j);
            }
        StateInfo ci;
        ci.allocate(nc);
        for (int ic = 0; ic < c.n; ic++) {
            vector<uint32_t> &v = mp.at(c.quanta[ic]);
            ci.n_states[ic] = iab;
            memcpy(ci.quanta + iab, &v[0], v.size() * sizeof(uint32_t));
            iab += v.size();
        }
        ci.reallocate(iab);
        ci.n_states_total = c.n;
        return ci;
    }
    static void filter(StateInfo &a, StateInfo &b, S target) {
        a.n_states_total = 0;
        for (int i = 0; i < a.n; i++) {
            S qb = target - a.quanta[i];
            int x = 0;
            for (int k = 0; k < qb.count(); k++) {
                int idx = b.find_state(qb[k]);
                x += idx == -1 ? 0 : b.n_states[idx];
            }
            a.n_states[i] = (uint16_t)min(x, (int)a.n_states[i]);
            a.n_states_total += a.n_states[i];
        }
        b.n_states_total = 0;
        for (int i = 0; i < b.n; i++) {
            S qa = target - b.quanta[i];
            int x = 0;
            for (int k = 0; k < qa.count(); k++) {
                int idx = a.find_state(qa[k]);
                x += idx == -1 ? 0 : a.n_states[idx];
            }
            b.n_states[i] = (uint16_t)min(x, (int)b.n_states[i]);
            b.n_states_total += b.n_states[i];
        }
    }
    friend ostream &operator<<(ostream &os, const StateInfo<S> &c) {
        for (int i = 0; i < c.n; i++)
            os << c.quanta[i].to_str() << " : " << c.n_states[i] << endl;
        return os;
    }
};

template <typename, typename = void> struct StateProbability;

template <typename S>
struct StateProbability<
    S, typename enable_if<integral_constant<
           bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    S *quanta;
    double *probs;
    int n;
    StateProbability() : quanta(0), probs(0), n(0) {}
    StateProbability(S q) {
        allocate(1);
        quanta[0] = q, probs[0] = 1;
    }
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) + length);
        n = length;
        quanta = (S *)ptr;
        probs = (double *)(ptr + length);
    }
    void reallocate(int length) {
        uint32_t *ptr = ialloc->reallocate((uint32_t *)quanta, (n << 1) + n,
                                           (length << 1) + length);
        if (ptr == (uint32_t *)quanta) {
            memmove(ptr + length, probs, length * sizeof(double));
            probs = (double *)(quanta + length);
        } else {
            memmove(ptr, quanta, length * sizeof(uint32_t));
            memmove(ptr + length, probs, length * sizeof(double));
            quanta = (S *)ptr;
            probs = (double *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        assert(n != 0);
        ialloc->deallocate((uint32_t *)quanta, (n << 1) + n);
        quanta = 0;
        probs = 0;
    }
    void collect(S target = 0x7FFFFFFF) {
        int k = -1;
        int nn = upper_bound(quanta, quanta + n, target) - quanta;
        for (int i = 0; i < nn; i++)
            if (probs[i] == 0.0)
                continue;
            else if (k != -1 && quanta[i] == quanta[k])
                probs[k] = probs[k] + probs[i];
            else {
                k++;
                quanta[k] = quanta[i];
                probs[k] = probs[i];
            }
        reallocate(k + 1);
    }
    int find_state(S q) const {
        auto p = lower_bound(quanta, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return p - quanta;
    }
    static StateProbability<S>
    tensor_product_no_collect(const StateProbability<S> &a,
                              const StateProbability<S> &b,
                              const StateInfo<S> &cref) {
        StateProbability<S> c;
        c.allocate(cref.n);
        memcpy(c.quanta, cref.quanta, c.n * sizeof(uint32_t));
        memset(c.probs, 0, c.n * sizeof(double));
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                S qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    int ic = c.find_state(qc[k]);
                    if (ic != -1)
                        c.probs[ic] += a.probs[i] * b.probs[j];
                }
            }
        return c;
    }
    friend ostream &operator<<(ostream &os, const StateProbability<S> &c) {
        for (int i = 0; i < c.n; i++)
            os << c.quanta[i].to_str() << " : " << c.probs[i] << endl;
        return os;
    }
};

template <typename, typename = void> struct SparseMatrixInfo;

template <typename S>
struct SparseMatrixInfo<
    S, typename enable_if<integral_constant<
           bool, sizeof(S) == sizeof(uint32_t)>::value>::type> {
    S *quanta;
    uint16_t *n_states_bra, *n_states_ket;
    uint32_t *n_states_total;
    S delta_quantum;
    bool is_fermion;
    bool is_wavefunction;
    int n;
    static bool cmp_op_info(const pair<S, shared_ptr<SparseMatrixInfo>> &p,
                            S q) {
        return p.first < q;
    }
    struct ConnectionInfo {
        S *quanta;
        uint32_t *idx;
        uint32_t *stride;
        double *factor;
        uint16_t *ia, *ib, *ic;
        int n[5], nc;
        ConnectionInfo() : nc(-1) { memset(n, -1, sizeof(n)); }
        void initialize_diag(
            S cdq, S opdq, const vector<pair<uint8_t, S>> &subdq,
            const vector<pair<S, shared_ptr<SparseMatrixInfo>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo>>> &binfos,
            const shared_ptr<SparseMatrixInfo> &cinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size());
            vector<uint16_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = vic.size();
                S adq = cja ? -subdq[k].second.get_bra(opdq)
                            : subdq[k].second.get_bra(opdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                if ((adq + bdq)[0].data != 0)
                    continue;
                shared_ptr<SparseMatrixInfo> ainfo =
                    lower_bound(ainfos.begin(), ainfos.end(), adq, cmp_op_info)
                        ->second;
                shared_ptr<SparseMatrixInfo> binfo =
                    lower_bound(binfos.begin(), binfos.end(), bdq, cmp_op_info)
                        ->second;
                assert(ainfo->delta_quantum == adq);
                assert(binfo->delta_quantum == bdq);
                for (int ic = 0; ic < cinfo->n; ic++) {
                    S aq = cinfo->quanta[ic].get_bra(cdq);
                    S bq = -cinfo->quanta[ic].get_ket();
                    int ia = ainfo->find_state(aq), ib = binfo->find_state(bq);
                    if (ia != -1 && ib != -1 && aq == aq.get_bra(adq) &&
                        bq == bq.get_bra(bdq)) {
                        double factor =
                            sqrt(cdq.multiplicity() * opdq.multiplicity() *
                                 aq.multiplicity() * bq.multiplicity()) *
                            cg->wigner_9j(aq.twos(), bq.twos(), cdq.twos(),
                                          adq.twos(), bdq.twos(), opdq.twos(),
                                          aq.twos(), bq.twos(), cdq.twos());
                        if (cja)
                            factor *= cg->transpose_cg(adq.twos(), aq.twos(),
                                                       aq.twos());
                        if (cjb)
                            factor *= cg->transpose_cg(bdq.twos(), bq.twos(),
                                                       bq.twos());
                        factor *= (binfo->is_fermion && (aq.n() & 1)) ? -1 : 1;
                        if (abs(factor) >= TINY) {
                            via.push_back(ia);
                            vib.push_back(ib);
                            vic.push_back(ic);
                            vf.push_back(factor);
                        }
                    }
                }
            }
            n[4] = vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = vic.size();
            uint32_t *ptr = ialloc->allocate((n[4] << 1));
            uint32_t *cptr = ialloc->allocate((nc << 2) + nc - (nc >> 1));
            quanta = (S *)ptr;
            idx = ptr + n[4];
            stride = cptr;
            factor = (double *)(cptr + nc);
            ia = (uint16_t *)(cptr + nc + nc + nc), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, &vidx[0], n[4] * sizeof(uint32_t));
            memset(stride, 0, nc * sizeof(uint32_t));
            memcpy(factor, &vf[0], nc * sizeof(double));
            memcpy(ia, &via[0], nc * sizeof(uint16_t));
            memcpy(ib, &vib[0], nc * sizeof(uint16_t));
            memcpy(ic, &vic[0], nc * sizeof(uint16_t));
        }
        void initialize_wfn(
            S cdq, S vdq, S opdq, const vector<pair<uint8_t, S>> &subdq,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &binfos,
            const shared_ptr<SparseMatrixInfo<S>> &cinfo,
            const shared_ptr<SparseMatrixInfo<S>> &vinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size()), viv;
            vector<uint16_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = viv.size();
                vector<vector<
                    tuple<double, uint32_t, uint16_t, uint16_t, uint16_t>>>
                    pv;
                size_t ip = 0;
                S adq = cja ? -subdq[k].second.get_bra(opdq)
                            : subdq[k].second.get_bra(opdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                shared_ptr<SparseMatrixInfo> ainfo =
                    lower_bound(ainfos.begin(), ainfos.end(), adq, cmp_op_info)
                        ->second;
                shared_ptr<SparseMatrixInfo> binfo =
                    lower_bound(binfos.begin(), binfos.end(), bdq, cmp_op_info)
                        ->second;
                assert(ainfo->delta_quantum == adq);
                assert(binfo->delta_quantum == bdq);
                for (int iv = 0; iv < vinfo->n; iv++) {
                    ip = 0;
                    S lq = vinfo->quanta[iv].get_bra(vdq);
                    S rq = -vinfo->quanta[iv].get_ket();
                    S rqprimes = cjb ? rq + bdq : rq - bdq;
                    for (int r = 0; r < rqprimes.count(); r++) {
                        S rqprime = rqprimes[r];
                        int ib =
                            binfo->find_state(cjb ? bdq.combine(rqprime, rq)
                                                  : bdq.combine(rq, rqprime));
                        if (ib != -1) {
                            S lqprimes = cdq - rqprime;
                            for (int l = 0; l < lqprimes.count(); l++) {
                                S lqprime = lqprimes[l];
                                int ia = ainfo->find_state(
                                    cja ? adq.combine(lqprime, lq)
                                        : adq.combine(lq, lqprime));
                                int ic = cinfo->find_state(
                                    cdq.combine(lqprime, -rqprime));
                                if (ia != -1 && ic != -1) {
                                    double factor =
                                        sqrt(cdq.multiplicity() *
                                             opdq.multiplicity() *
                                             lq.multiplicity() *
                                             rq.multiplicity()) *
                                        cg->wigner_9j(
                                            lqprime.twos(), rqprime.twos(),
                                            cdq.twos(), adq.twos(), bdq.twos(),
                                            opdq.twos(), lq.twos(), rq.twos(),
                                            vdq.twos());
                                    factor *=
                                        (binfo->is_fermion && (lqprime.n() & 1))
                                            ? -1
                                            : 1;
                                    if (cja)
                                        factor *= cg->transpose_cg(
                                            adq.twos(), lq.twos(),
                                            lqprime.twos());
                                    if (cjb)
                                        factor *= cg->transpose_cg(
                                            bdq.twos(), rq.twos(),
                                            rqprime.twos());
                                    if (abs(factor) >= TINY) {
                                        if (pv.size() <= ip)
                                            pv.push_back(
                                                vector<tuple<double, uint32_t,
                                                             uint16_t, uint16_t,
                                                             uint16_t>>());
                                        pv[ip].push_back(
                                            make_tuple(factor, iv, ia, ib, ic));
                                        ip++;
                                    }
                                }
                            }
                        }
                    }
                }
                size_t np = 0;
                for (auto &r : pv)
                    np += r.size();
                vf.reserve(vf.size() + np);
                viv.reserve(viv.size() + np);
                via.reserve(via.size() + np);
                vib.reserve(vib.size() + np);
                vic.reserve(vic.size() + np);
                for (ip = 0; ip < pv.size(); ip++) {
                    for (auto &r : pv[ip]) {
                        vf.push_back(get<0>(r));
                        viv.push_back(get<1>(r));
                        via.push_back(get<2>(r));
                        vib.push_back(get<3>(r));
                        vic.push_back(get<4>(r));
                    }
                }
            }
            n[4] = vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = viv.size();
            uint32_t *ptr = ialloc->allocate((n[4] << 1));
            uint32_t *cptr = ialloc->allocate((nc << 2) + nc - (nc >> 1));
            quanta = (S *)ptr;
            idx = ptr + n[4];
            stride = cptr;
            factor = (double *)(cptr + nc);
            ia = (uint16_t *)(cptr + nc + nc + nc), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, &vidx[0], n[4] * sizeof(uint32_t));
            memcpy(stride, &viv[0], nc * sizeof(uint32_t));
            memcpy(factor, &vf[0], nc * sizeof(double));
            memcpy(ia, &via[0], nc * sizeof(uint16_t));
            memcpy(ib, &vib[0], nc * sizeof(uint16_t));
            memcpy(ic, &vic[0], nc * sizeof(uint16_t));
        }
        void initialize_tp(
            S cdq, const vector<pair<uint8_t, S>> &subdq,
            const StateInfo<S> &bra, const StateInfo<S> &ket,
            const StateInfo<S> &bra_a, const StateInfo<S> &bra_b,
            const StateInfo<S> &ket_a, const StateInfo<S> &ket_b,
            const StateInfo<S> &bra_cinfo, const StateInfo<S> &ket_cinfo,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &ainfos,
            const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &binfos,
            const shared_ptr<SparseMatrixInfo<S>> &cinfo,
            const shared_ptr<CG<S>> &cg) {
            if (ainfos.size() == 0 || binfos.size() == 0) {
                n[4] = nc = 0;
                return;
            }
            vector<uint32_t> vidx(subdq.size()), vstride;
            vector<uint16_t> via, vib, vic;
            vector<double> vf;
            memset(n, -1, sizeof(n));
            for (size_t k = 0; k < subdq.size(); k++) {
                if (n[subdq[k].first] == -1)
                    n[subdq[k].first] = k;
                bool cja = subdq[k].first & 1, cjb = (subdq[k].first & 2) >> 1;
                vidx[k] = vstride.size();
                S adq = cja ? -subdq[k].second.get_bra(cdq)
                            : subdq[k].second.get_bra(cdq),
                  bdq = cjb ? subdq[k].second.get_ket()
                            : -subdq[k].second.get_ket();
                shared_ptr<SparseMatrixInfo<S>> ainfo =
                    lower_bound(ainfos.begin(), ainfos.end(), adq, cmp_op_info)
                        ->second;
                shared_ptr<SparseMatrixInfo<S>> binfo =
                    lower_bound(binfos.begin(), binfos.end(), bdq, cmp_op_info)
                        ->second;
                assert(ainfo->delta_quantum == adq);
                assert(binfo->delta_quantum == bdq);
                for (int ic = 0; ic < cinfo->n; ic++) {
                    int ib = bra.find_state(cinfo->quanta[ic].get_bra(cdq));
                    int ik = ket.find_state(cinfo->quanta[ic].get_ket());
                    int kbed = ib == bra.n - 1 ? bra_cinfo.n
                                               : bra_cinfo.n_states[ib + 1];
                    int kked = ik == ket.n - 1 ? ket_cinfo.n
                                               : ket_cinfo.n_states[ik + 1];
                    uint32_t bra_stride = 0, ket_stride = 0;
                    for (int kb = bra_cinfo.n_states[ib]; kb < kbed; kb++) {
                        uint16_t jba = bra_cinfo.quanta[kb].data >> 16,
                                 jbb = bra_cinfo.quanta[kb].data & (0xFFFFU);
                        ket_stride = 0;
                        for (int kk = ket_cinfo.n_states[ik]; kk < kked; kk++) {
                            uint16_t jka = ket_cinfo.quanta[kk].data >> 16,
                                     jkb =
                                         ket_cinfo.quanta[kk].data & (0xFFFFU);
                            S qa = cja ? adq.combine(ket_a.quanta[jka],
                                                     bra_a.quanta[jba])
                                       : adq.combine(bra_a.quanta[jba],
                                                     ket_a.quanta[jka]),
                              qb = cjb ? bdq.combine(ket_b.quanta[jkb],
                                                     bra_b.quanta[jbb])
                                       : bdq.combine(bra_b.quanta[jbb],
                                                     ket_b.quanta[jkb]);
                            if (qa != S(0xFFFFFFFFU) && qb != S(0xFFFFFFFFU)) {
                                int ia = ainfo->find_state(qa),
                                    ib = binfo->find_state(qb);
                                if (ia != -1 && ib != -1) {
                                    S aq = bra_a.quanta[jba];
                                    S aqprime = ket_a.quanta[jka];
                                    S bq = bra_b.quanta[jbb];
                                    S bqprime = ket_b.quanta[jkb];
                                    S cq = cinfo->quanta[ic].get_bra(cdq);
                                    S cqprime = cinfo->quanta[ic].get_ket();
                                    double factor =
                                        sqrt(cqprime.multiplicity() *
                                             cdq.multiplicity() *
                                             aq.multiplicity() *
                                             bq.multiplicity()) *
                                        cg->wigner_9j(
                                            aqprime.twos(), bqprime.twos(),
                                            cqprime.twos(), adq.twos(),
                                            bdq.twos(), cdq.twos(), aq.twos(),
                                            bq.twos(), cq.twos());
                                    factor *=
                                        (binfo->is_fermion && (aqprime.n() & 1))
                                            ? -1
                                            : 1;
                                    if (cja)
                                        factor *= cg->transpose_cg(
                                            adq.twos(), aq.twos(),
                                            aqprime.twos());
                                    if (cjb)
                                        factor *= cg->transpose_cg(
                                            bdq.twos(), bq.twos(),
                                            bqprime.twos());
                                    if (abs(factor) >= TINY) {
                                        via.push_back(ia);
                                        vib.push_back(ib);
                                        vic.push_back(ic);
                                        vstride.push_back(
                                            bra_stride *
                                                cinfo->n_states_ket[ic] +
                                            ket_stride);
                                        vf.push_back(factor);
                                    }
                                }
                            }
                            ket_stride += (uint32_t)ket_a.n_states[jka] *
                                          ket_b.n_states[jkb];
                        }
                        bra_stride +=
                            (uint32_t)bra_a.n_states[jba] * bra_b.n_states[jbb];
                    }
                }
            }
            n[4] = vidx.size();
            for (int i = 3; i >= 0; i--)
                if (n[i] == -1)
                    n[i] = n[i + 1];
            nc = vstride.size();
            uint32_t *ptr = ialloc->allocate((n[4] << 1));
            uint32_t *cptr = ialloc->allocate((nc << 2) + nc - (nc >> 1));
            quanta = (S *)ptr;
            idx = ptr + n[4];
            stride = cptr;
            factor = (double *)(cptr + nc);
            ia = (uint16_t *)(cptr + nc + nc + nc), ib = ia + nc, ic = ib + nc;
            for (int i = 0; i < n[4]; i++)
                quanta[i] = subdq[i].second;
            memcpy(idx, &vidx[0], n[4] * sizeof(uint32_t));
            memcpy(stride, &vstride[0], nc * sizeof(uint32_t));
            memcpy(factor, &vf[0], nc * sizeof(double));
            memcpy(ia, &via[0], nc * sizeof(uint16_t));
            memcpy(ib, &vib[0], nc * sizeof(uint16_t));
            memcpy(ic, &vic[0], nc * sizeof(uint16_t));
        }
        void reallocate(bool clean) {
            size_t length = (n[4] << 1) + (nc << 2) + nc - (nc >> 1);
            uint32_t *ptr = ialloc->reallocate((uint32_t *)quanta, length,
                                               clean ? 0 : length);
            if (ptr != (uint32_t *)quanta) {
                memmove(ptr, quanta, length * sizeof(uint32_t));
                quanta = (S *)ptr;
                idx = ptr + n[4];
                stride = ptr + (n[4] << 1);
                factor = (double *)(ptr + (n[4] << 1) + nc);
                ia = (uint16_t *)(stride + nc + nc + nc), ib = ia + nc,
                ic = ib + nc;
            }
            if (clean) {
                quanta = nullptr;
                idx = nullptr;
                stride = nullptr;
                factor = nullptr;
                ia = ib = ic = nullptr;
                nc = -1;
                memset(n, -1, sizeof(n));
            }
        }
        void deallocate() {
            assert(n[4] != -1);
            if (n[4] != 0 || nc != 0)
                ialloc->deallocate((uint32_t *)quanta,
                                   (n[4] << 1) + (nc << 2) + nc - (nc >> 1));
            quanta = nullptr;
            idx = nullptr;
            stride = nullptr;
            factor = nullptr;
            ia = ib = ic = nullptr;
            nc = -1;
            memset(n, -1, sizeof(n));
        }
        friend ostream &operator<<(ostream &os, const ConnectionInfo &ci) {
            os << "CI N=" << ci.n[4] << " NC=" << ci.nc << endl;
            for (int i = 0; i < 4; i++)
                os << "CJ=" << i << " : " << ci.n[i] << "~" << ci.n[i + 1]
                   << " ; ";
            os << endl;
            for (int i = 0; i < ci.n[4]; i++)
                os << "(BRA) " << ci.quanta[i].get_bra(S(0)) << " KET "
                   << -ci.quanta[i].get_ket() << " [ " << (int)ci.idx[i] << "~"
                   << (int)(i != ci.n[4] - 1 ? ci.idx[i + 1] : ci.nc) << " ]"
                   << endl;
            for (int i = 0; i < ci.nc; i++)
                os << setw(4) << i << " IA=" << ci.ia[i] << " IB=" << ci.ib[i]
                   << " IC=" << ci.ic[i] << " STR=" << ci.stride[i]
                   << " factor=" << ci.factor[i] << endl;
            return os;
        }
    };
    shared_ptr<ConnectionInfo> cinfo;
    SparseMatrixInfo() : n(-1), cinfo(nullptr) {}
    SparseMatrixInfo deep_copy() const {
        SparseMatrixInfo other;
        other.allocate(n);
        copy_data_to(other);
        other.delta_quantum = delta_quantum;
        other.is_fermion = is_fermion;
        other.is_wavefunction = is_wavefunction;
        return other;
    }
    void copy_data_to(SparseMatrixInfo &other) const {
        assert(other.n == n);
        memcpy(other.quanta, quanta, ((n << 1) + n) * sizeof(S));
    }
    void load_data(const string &filename) {
        ifstream ifs(filename.c_str(), ios::binary);
        load_data(ifs);
        ifs.close();
    }
    void load_data(ifstream &ifs) {
        ifs.read((char *)&delta_quantum, sizeof(delta_quantum));
        ifs.read((char *)&n, sizeof(n));
        uint32_t *ptr = ialloc->allocate((n << 1) + n);
        ifs.read((char *)ptr, sizeof(uint32_t) * ((n << 1) + n));
        ifs.read((char *)&is_fermion, sizeof(is_fermion));
        ifs.read((char *)&is_wavefunction, sizeof(is_wavefunction));
        quanta = (S *)ptr;
        n_states_bra = (uint16_t *)(ptr + n);
        n_states_ket = (uint16_t *)(ptr + n) + n;
        n_states_total = ptr + (n << 1);
        cinfo = nullptr;
    }
    void save_data(const string &filename) const {
        ofstream ofs(filename.c_str(), ios::binary);
        save_data(ofs);
        ofs.close();
    }
    void save_data(ofstream &ofs) const {
        ofs.write((char *)&delta_quantum, sizeof(delta_quantum));
        ofs.write((char *)&n, sizeof(n));
        ofs.write((char *)quanta, sizeof(uint32_t) * ((n << 1) + n));
        ofs.write((char *)&is_fermion, sizeof(is_fermion));
        ofs.write((char *)&is_wavefunction, sizeof(is_wavefunction));
    }
    void initialize_contract(const shared_ptr<SparseMatrixInfo> &linfo,
                             const shared_ptr<SparseMatrixInfo> &rinfo) {
        assert(linfo->is_wavefunction ^ rinfo->is_wavefunction);
        this->is_fermion = false;
        this->is_wavefunction = true;
        shared_ptr<SparseMatrixInfo> winfo =
            linfo->is_wavefunction ? linfo : rinfo;
        delta_quantum = winfo->delta_quantum;
        vector<S> qs;
        qs.reserve(winfo->n);
        if (rinfo->is_wavefunction)
            for (int i = 0; i < rinfo->n; i++) {
                S bra = rinfo->quanta[i].get_bra(delta_quantum);
                if (linfo->find_state(bra) != -1)
                    qs.push_back(rinfo->quanta[i]);
            }
        else
            for (int i = 0; i < linfo->n; i++) {
                S ket = -linfo->quanta[i].get_ket();
                if (rinfo->find_state(ket) != -1)
                    qs.push_back(linfo->quanta[i]);
            }
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(S));
            if (rinfo->is_wavefunction)
                for (int i = 0; i < n; i++) {
                    S bra = quanta[i].get_bra(delta_quantum);
                    n_states_bra[i] =
                        linfo->n_states_bra[linfo->find_state(bra)];
                    n_states_ket[i] =
                        rinfo->n_states_ket[rinfo->find_state(quanta[i])];
                }
            else
                for (int i = 0; i < n; i++) {
                    S ket = -quanta[i].get_ket();
                    n_states_bra[i] =
                        linfo->n_states_bra[linfo->find_state(quanta[i])];
                    n_states_ket[i] =
                        rinfo->n_states_ket[rinfo->find_state(ket)];
                }
            n_states_total[0] = 0;
            for (int i = 0; i < n - 1; i++)
                n_states_total[i + 1] =
                    n_states_total[i] +
                    (uint32_t)n_states_bra[i] * n_states_ket[i];
        }
    }
    void initialize_dm(const shared_ptr<SparseMatrixInfo> &wfn_info, S dq,
                       bool trace_right) {
        this->is_fermion = false;
        this->is_wavefunction = false;
        assert(wfn_info->is_wavefunction);
        delta_quantum = dq;
        vector<S> qs;
        qs.reserve(wfn_info->n);
        if (trace_right)
            for (int i = 0; i < wfn_info->n; i++)
                qs.push_back(
                    wfn_info->quanta[i].get_bra(wfn_info->delta_quantum));
        else
            for (int i = 0; i < wfn_info->n; i++)
                qs.push_back(-wfn_info->quanta[i].get_ket());
        sort(qs.begin(), qs.end());
        qs.resize(distance(qs.begin(), unique(qs.begin(), qs.end())));
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(S));
            if (trace_right)
                for (int i = 0; i < wfn_info->n; i++) {
                    S q = wfn_info->quanta[i].get_bra(wfn_info->delta_quantum);
                    int ii = find_state(q);
                    n_states_bra[ii] = n_states_ket[ii] =
                        wfn_info->n_states_bra[i];
                }
            else
                for (int i = 0; i < wfn_info->n; i++) {
                    S q = -wfn_info->quanta[i].get_ket();
                    int ii = find_state(q);
                    n_states_bra[ii] = n_states_ket[ii] =
                        wfn_info->n_states_ket[i];
                }
            n_states_total[0] = 0;
            for (int i = 0; i < n - 1; i++)
                n_states_total[i + 1] =
                    n_states_total[i] +
                    (uint32_t)n_states_bra[i] * n_states_ket[i];
        }
    }
    void initialize(const StateInfo<S> &bra, const StateInfo<S> &ket, S dq,
                    bool is_fermion, bool wfn = false) {
        this->is_fermion = is_fermion;
        this->is_wavefunction = wfn;
        delta_quantum = dq;
        vector<S> qs;
        qs.reserve(ket.n);
        for (int i = 0; i < ket.n; i++) {
            S q = wfn ? -ket.quanta[i] : ket.quanta[i];
            S bs = dq + q;
            for (int k = 0; k < bs.count(); k++)
                if (bra.find_state(bs[k]) != -1)
                    qs.push_back(dq.combine(bs[k], q));
        }
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(S));
            sort(quanta, quanta + n);
            for (int i = 0; i < n; i++) {
                n_states_ket[i] = ket.n_states[ket.find_state(
                    wfn ? -quanta[i].get_ket() : quanta[i].get_ket())];
                n_states_bra[i] =
                    bra.n_states[bra.find_state(quanta[i].get_bra(dq))];
            }
            n_states_total[0] = 0;
            for (int i = 0; i < n - 1; i++)
                n_states_total[i + 1] =
                    n_states_total[i] +
                    (uint32_t)n_states_bra[i] * n_states_ket[i];
        }
    }
    shared_ptr<StateInfo<S>> extract_state_info(bool right) {
        shared_ptr<StateInfo<S>> info = make_shared<StateInfo<S>>();
        assert(delta_quantum.data == 0);
        info->allocate(n);
        memcpy(info->quanta, quanta, n * sizeof(S));
        memcpy(info->n_states, right ? n_states_ket : n_states_bra,
               n * sizeof(uint16_t));
        info->n_states_total =
            accumulate(info->n_states, info->n_states + n, 0);
        return info;
    }
    int find_state(S q, int start = 0) const {
        auto p = lower_bound(quanta + start, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return p - quanta;
    }
    void sort_states() {
        int idx[n];
        S q[n];
        uint16_t nqb[n], nqk[n];
        memcpy(q, quanta, n * sizeof(S));
        memcpy(nqb, n_states_bra, n * sizeof(uint16_t));
        memcpy(nqk, n_states_ket, n * sizeof(uint16_t));
        for (int i = 0; i < n; i++)
            idx[i] = i;
        sort(idx, idx + n, [&q](int i, int j) { return q[i] < q[j]; });
        for (int i = 0; i < n; i++)
            quanta[i] = q[idx[i]], n_states_bra[i] = nqb[idx[i]],
            n_states_ket[i] = nqk[idx[i]];
        n_states_total[0] = 0;
        for (int i = 0; i < n - 1; i++)
            n_states_total[i + 1] =
                n_states_total[i] + (uint32_t)n_states_bra[i] * n_states_ket[i];
    }
    uint32_t get_total_memory() const {
        return n == 0 ? 0
                      : n_states_total[n - 1] +
                            (uint32_t)n_states_bra[n - 1] * n_states_ket[n - 1];
    }
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) + length);
        quanta = (S *)ptr;
        n_states_bra = (uint16_t *)(ptr + length);
        n_states_ket = (uint16_t *)(ptr + length) + length;
        n_states_total = ptr + (length << 1);
        n = length;
    }
    void deallocate() {
        assert(n != -1);
        ialloc->deallocate((uint32_t *)quanta, (n << 1) + n);
        quanta = nullptr;
        n_states_bra = nullptr;
        n_states_ket = nullptr;
        n_states_total = nullptr;
        n = -1;
    }
    void reallocate(int length) {
        uint32_t *ptr = ialloc->reallocate((uint32_t *)quanta, (n << 1) + n,
                                           (length << 1) + length);
        if (ptr == (uint32_t *)quanta)
            memmove(ptr + length, n_states_bra,
                    (length << 1) * sizeof(uint32_t));
        else {
            memmove(ptr, quanta, ((length << 1) + length) * sizeof(uint32_t));
            quanta = (S *)ptr;
        }
        n_states_bra = (uint16_t *)(ptr + length);
        n_states_ket = (uint16_t *)(ptr + length) + length;
        n_states_total = ptr + (length << 1);
        n = length;
    }
    friend ostream &operator<<(ostream &os, const SparseMatrixInfo<S> &c) {
        os << "DQ=" << c.delta_quantum << " N=" << c.n
           << " SIZE=" << c.get_total_memory() << endl;
        for (int i = 0; i < c.n; i++)
            os << "BRA " << c.quanta[i].get_bra(c.delta_quantum) << " KET "
               << c.quanta[i].get_ket() << " [ " << (int)c.n_states_bra[i]
               << "x" << (int)c.n_states_ket[i] << " ]" << endl;
        return os;
    }
};

extern "C" {

// vector scale
// vector [sx] = double [sa] * vector [sx]
extern void dscal(const int *n, const double *sa, double *sx, const int *incx);

// vector copy
// vector [dy] = [dx]
extern void dcopy(const int *n, const double *dx, const int *incx, double *dy,
                  const int *incy);

// vector addition
// vector [sy] = vector [sy] + double [sa] * vector [sx]
extern void daxpy(const int *n, const double *sa, const double *sx,
                  const int *incx, double *sy, const int *incy);

// vector dot product
extern double ddot(const int *n, const double *dx, const int *incx,
                   const double *dy, const int *incy);

// Euclidean norm of a vector
extern double dnrm2(const int *n, const double *x, const int *incx);

// matrix multiplication
// mat [c] = double [alpha] * mat [a] * mat [b] + double [beta] * mat [c]
extern void dgemm(const char *transa, const char *transb, const int *n,
                  const int *m, const int *k, const double *alpha,
                  const double *a, const int *lda, const double *b,
                  const int *ldb, const double *beta, double *c,
                  const int *ldc);

// matrix-vector multiplication
// vec [y] = double [alpha] * mat [a] * vec [x] + double [beta] * vec [y]
extern void dgemv(const char *trans, const int *m, const int *n,
                  const double *alpha, const double *a, const int *lda,
                  const double *x, const int *incx, const double *beta,
                  double *y, const int *incy);

// linear system a * x = b
extern void dgesv(const int *n, const int *nrhs, double *a, const int *lda,
                  int *ipiv, double *b, const int *ldb, int *info);

// QR factorization
extern void dgeqrf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void dorgqr(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);

// LQ factorization
extern void dgelqf(const int *m, const int *n, double *a, const int *lda,
                   double *tau, double *work, const int *lwork, int *info);
extern void dorglq(const int *m, const int *n, const int *k, double *a,
                   const int *lda, const double *tau, double *work,
                   const int *lwork, int *info);

// eigenvalue problem
extern void dsyev(const char *jobz, const char *uplo, const int *n, double *a,
                  const int *lda, double *w, double *work, const int *lwork,
                  int *info);
}

struct MatrixFunctions {
    // a = b
    static void copy(const MatrixRef &a, const MatrixRef &b, const int inca = 1,
                     const int incb = 1) {
        assert(a.m == b.m && a.n == b.n);
        const int n = a.m * a.n;
        dcopy(&n, b.data, &incb, a.data, &inca);
    }
    static void iscale(const MatrixRef &a, double scale) {
        int n = a.m * a.n, inc = 1;
        dscal(&n, &scale, a.data, &inc);
    }
    static void iadd(const MatrixRef &a, const MatrixRef &b, double scale,
                     bool conj = false) {
        if (!conj) {
            assert(a.m == b.m && a.n == b.n);
            int n = a.m * a.n, inc = 1;
            daxpy(&n, &scale, b.data, &inc, a.data, &inc);
        } else {
            assert(a.m == b.n && a.n == b.m);
            for (int i = 0, inc = 1; i < a.m; i++)
                daxpy(&a.n, &scale, b.data + i, &a.m, a.data, &inc);
        }
    }
    static double norm(const MatrixRef &a) {
        int n = a.m * a.n, inc = 1;
        return dnrm2(&n, a.data, &inc);
    }
    static double dot(const MatrixRef &a, const MatrixRef &b) {
        assert(a.m == b.m && a.n == b.n);
        int n = a.m * a.n, inc = 1;
        return ddot(&n, a.data, &inc, b.data, &inc);
    }
    template <typename T1, typename T2>
    static bool all_close(const T1 &a, const T2 &b, double atol = 1E-8,
                          double rtol = 1E-5, double scale = 1.0) {
        assert(a.m == b.m && a.n == b.n);
        for (int i = 0; i < a.m; i++)
            for (int j = 0; j < a.n; j++)
                if (abs(a(i, j) - scale * b(i, j)) > atol + rtol * abs(b(i, j)))
                    return false;
        return true;
    }
    static void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                         bool conjb, const MatrixRef &c, double scale,
                         double cfactor) {
        if (!conja && !conjb) {
            assert(a.n == b.m && c.m == a.m && c.n == b.n);
            dgemm("n", "n", &c.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (!conja && conjb) {
            assert(a.n == b.n && c.m == a.m && c.n == b.m);
            dgemm("t", "n", &c.n, &c.m, &a.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else if (conja && !conjb) {
            assert(a.m == b.m && c.m == a.n && c.n == b.n);
            dgemm("n", "t", &c.n, &c.m, &b.m, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        } else {
            assert(a.m == b.n && c.m == a.n && c.n == b.m);
            dgemm("t", "t", &c.n, &c.m, &b.n, &scale, b.data, &b.n, a.data,
                  &a.n, &cfactor, c.data, &c.n);
        }
    }
    // c = bra * a * ket.T
    static void rotate(const MatrixRef &a, const MatrixRef &c,
                       const MatrixRef &bra, bool conj_bra,
                       const MatrixRef &ket, bool conj_ket, double scale) {
        MatrixRef work(nullptr, a.m, conj_ket ? ket.m : ket.n);
        work.allocate();
        multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        multiply(bra, conj_bra, work, false, c, scale, 1.0);
        work.deallocate();
    }
    // only diagonal elements so no conj parameters
    static void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                        const MatrixRef &c, double scale) {
        assert(a.m == a.n && b.m == b.n && c.m == a.n && c.n == b.n);
        const double cfactor = 1.0;
        const int k = 1, lda = a.n + 1, ldb = b.n + 1;
        dgemm("t", "n", &b.n, &a.n, &k, &scale, b.data, &ldb, a.data, &lda,
              &cfactor, c.data, &c.n);
    }
    static void tensor_product(const MatrixRef &a, bool conja,
                               const MatrixRef &b, bool conjb,
                               const MatrixRef &c, double scale,
                               uint32_t stride) {
        const double cfactor = 1.0;
        switch (conja | (conjb << 1)) {
        case 0:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const int n = b.m * b.n;
                    dgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (int k = 0; k < b.m; k++)
                        dgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const int n = a.m * a.n;
                    dgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (int k = 0; k < a.m; k++)
                        dgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (int i = 0, inc = 1; i < a.m; i++)
                    for (int j = 0; j < a.n; j++) {
                        const double factor = scale * a(i, j);
                        for (int k = 0; k < b.m; k++)
                            daxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 1:
            if (a.m == 1 && a.n == 1) {
                if (b.n == c.n) {
                    const int n = b.m * b.n;
                    dgemm("n", "n", &n, &a.n, &a.n, &scale, b.data, &n, a.data,
                          &a.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(b.n < c.n);
                    for (int k = 0; k < b.m; k++)
                        dgemm("n", "n", &b.n, &a.n, &a.n, &scale, &b(k, 0),
                              &b.n, a.data, &a.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else if (b.m == 1 && b.n == 1) {
                assert(a.m <= c.n);
                for (int k = 0; k < a.n; k++)
                    dgemm("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (int i = 0, inc = 1; i < a.n; i++)
                    for (int j = 0; j < a.m; j++) {
                        const double factor = scale * a(j, i);
                        for (int k = 0; k < b.m; k++)
                            daxpy(&b.n, &factor, &b(k, 0), &inc,
                                  &c(i * b.m + k, j * b.n + stride), &inc);
                    }
            }
            break;
        case 2:
            if (a.m == 1 && a.n == 1) {
                assert(b.m <= c.n);
                for (int k = 0; k < b.n; k++)
                    dgemm("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                if (a.n == c.n) {
                    const int n = a.m * a.n;
                    dgemm("n", "n", &n, &b.n, &b.n, &scale, a.data, &n, b.data,
                          &b.n, &cfactor, &c(0, stride), &n);
                } else {
                    assert(a.n < c.n);
                    for (int k = 0; k < a.m; k++)
                        dgemm("n", "n", &a.n, &b.n, &b.n, &scale, &a(k, 0),
                              &a.n, b.data, &b.n, &cfactor, &c(k, stride),
                              &c.n);
                }
            } else {
                for (int i = 0, incb = b.n, inc = 1; i < a.m; i++)
                    for (int j = 0; j < a.n; j++) {
                        const double factor = scale * a(i, j);
                        for (int k = 0; k < b.n; k++)
                            daxpy(&b.m, &factor, &b(0, k), &incb,
                                  &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        case 1 | 2:
            if (a.m == 1 && a.n == 1) {
                for (int k = 0; k < b.n; k++)
                    dgemm("t", "n", &b.m, &a.n, &a.n, &scale, &b(0, k), &b.n,
                          a.data, &a.n, &cfactor, &c(k, stride), &c.n);
            } else if (b.m == 1 && b.n == 1) {
                for (int k = 0; k < a.n; k++)
                    dgemm("t", "n", &a.m, &b.n, &b.n, &scale, &a(0, k), &a.n,
                          b.data, &b.n, &cfactor, &c(k, stride), &c.n);
            } else {
                for (int i = 0, incb = b.n, inc = 1; i < a.n; i++)
                    for (int j = 0; j < a.m; j++) {
                        const double factor = scale * a(j, i);
                        for (int k = 0; k < b.n; k++)
                            daxpy(&b.m, &factor, &b(0, k), &incb,
                                  &c(i * b.n + k, j * b.m + stride), &inc);
                    }
            }
            break;
        default:
            assert(false);
        }
    }
    static void lq(const MatrixRef &a, const MatrixRef &l, const MatrixRef &q) {
        int k = min(a.m, a.n), info, lwork = 34 * a.m;
        double work[lwork], tau[k], t[a.m * a.n];
        assert(a.m == l.m && a.n == q.n && l.n == k && q.m == k);
        memcpy(t, a.data, sizeof(t));
        dgeqrf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(l.data, 0, sizeof(double) * k * a.m);
        for (int j = 0; j < a.m; j++)
            memcpy(l.data + j * k, t + j * a.n, sizeof(double) * (j + 1));
        dorgqr(&a.n, &k, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memcpy(q.data, t, sizeof(double) * k * a.n);
    }
    static void qr(const MatrixRef &a, const MatrixRef &q, const MatrixRef &r) {
        int k = min(a.m, a.n), info, lwork = 34 * a.n;
        double work[lwork], tau[k], t[a.m * a.n];
        assert(a.m == q.m && a.n == r.n && q.n == k && r.m == k);
        memcpy(t, a.data, sizeof(t));
        dgelqf(&a.n, &a.m, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        memset(r.data, 0, sizeof(double) * k * a.n);
        for (int j = 0; j < k; j++)
            memcpy(r.data + j * a.n + j, t + j * a.n + j,
                   sizeof(double) * (a.n - j));
        dorglq(&k, &a.m, &k, t, &a.n, tau, work, &lwork, &info);
        assert(info == 0);
        for (int j = 0; j < a.m; j++)
            memcpy(q.data + j * k, t + j * a.n, sizeof(double) * k);
    }
    // eigenvectors are row vectors
    static void eigs(const MatrixRef &a, const DiagonalMatrix &w) {
        assert(a.m == a.n && w.n == a.n);
        int lwork = 34 * a.n, info;
        double work[lwork];
        dsyev("V", "U", &a.n, a.data, &a.n, w.data, work, &lwork, &info);
        assert(info == 0);
    }
    static void olsen_precondition(const MatrixRef &q, const MatrixRef &c,
                                   double ld, const DiagonalMatrix &aa) {
        assert(aa.size() == c.size());
        MatrixRef t(nullptr, c.m, c.n);
        t.allocate();
        copy(t, c);
        for (int i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                t.data[i] /= ld - aa.data[i];
        iadd(q, c, -dot(t, q) / dot(c, t));
        for (int i = 0; i < aa.n; i++)
            if (abs(ld - aa.data[i]) > 1E-12)
                q.data[i] /= ld - aa.data[i];
        t.deallocate();
    }
    template <typename MatMul>
    static vector<double>
    davidson(MatMul op, const DiagonalMatrix &aa, vector<MatrixRef> &bs,
             int &ndav, bool iprint = false, double conv_thrd = 5E-6,
             int max_iter = 5000, int deflation_min_size = 2,
             int deflation_max_size = 50) {
        int k = (int)bs.size();
        if (deflation_min_size < k)
            deflation_min_size = k;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++)
                iadd(bs[i], bs[j], -dot(bs[j], bs[i]));
            iscale(bs[i], 1.0 / sqrt(dot(bs[i], bs[i])));
        }
        vector<double> eigvals;
        vector<MatrixRef> sigmas;
        sigmas.reserve(k);
        for (int i = 0; i < k; i++) {
            sigmas.push_back(MatrixRef(nullptr, bs[i].m, bs[i].n));
            sigmas[i].allocate();
        }
        MatrixRef q(nullptr, bs[0].m, bs[0].n);
        q.allocate();
        q.clear();
        int l = k, ck = 0, msig = 0, m = k, xiter = 0;
        if (iprint)
            cout << endl;
        while (xiter < max_iter) {
            xiter++;
            for (int i = msig; i < m; i++, msig++) {
                sigmas[i].clear();
                op(bs[i], sigmas[i]);
            }
            DiagonalMatrix ld(nullptr, m);
            MatrixRef alpha(nullptr, m, m);
            ld.allocate();
            alpha.allocate();
            for (int i = 0; i < m; i++)
                for (int j = 0; j <= i; j++)
                    alpha(i, j) = dot(bs[i], sigmas[j]);
            eigs(alpha, ld);
            vector<MatrixRef> tmp(m, MatrixRef(nullptr, bs[0].m, bs[0].n));
            for (int i = 0; i < m; i++) {
                tmp[i].allocate();
                copy(tmp[i], bs[i]);
            }
            // note alpha row/column is diff from python
            // b[1:m] = np.dot(b[:], alpha[:, 1:m])
            for (int j = 0; j < m; j++)
                iscale(bs[j], alpha(j, j));
            for (int j = 0; j < m; j++)
                for (int i = 0; i < m; i++)
                    if (i != j)
                        iadd(bs[j], tmp[i], alpha(j, i));
            // sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
            for (int j = 0; j < m; j++) {
                copy(tmp[j], sigmas[j]);
                iscale(sigmas[j], alpha(j, j));
            }
            for (int j = 0; j < m; j++)
                for (int i = 0; i < m; i++)
                    if (i != j)
                        iadd(sigmas[j], tmp[i], alpha(j, i));
            for (int i = m - 1; i >= 0; i--)
                tmp[i].deallocate();
            alpha.deallocate();
            for (int i = 0; i < ck; i++) {
                copy(q, sigmas[i]);
                iadd(q, bs[i], -ld(i, i));
                if (dot(q, q) >= conv_thrd) {
                    ck = i;
                    break;
                }
            }
            copy(q, sigmas[ck]);
            iadd(q, bs[ck], -ld(ck, ck));
            double qq = dot(q, q);
            if (iprint)
                cout << setw(6) << xiter << setw(6) << m << setw(6) << ck
                     << fixed << setw(15) << setprecision(8) << ld.data[ck]
                     << scientific << setw(13) << setprecision(2) << qq << endl;
            olsen_precondition(q, bs[ck], ld.data[ck], aa);
            eigvals.resize(ck + 1);
            if (ck + 1 != 0)
                memcpy(&eigvals[0], ld.data, (ck + 1) * sizeof(double));
            ld.deallocate();
            if (qq < conv_thrd) {
                ck++;
                if (ck == k)
                    break;
            } else {
                if (m >= deflation_max_size)
                    m = msig = deflation_min_size;
                for (int j = 0; j < m; j++)
                    iadd(q, bs[j], -dot(bs[j], q));
                iscale(q, 1.0 / sqrt(dot(q, q)));
                if (m >= (int)bs.size()) {
                    bs.push_back(MatrixRef(nullptr, bs[0].m, bs[0].n));
                    bs[m].allocate();
                    sigmas.push_back(MatrixRef(nullptr, bs[0].m, bs[0].n));
                    sigmas[m].allocate();
                    sigmas[m].clear();
                }
                copy(bs[m], q);
                m++;
            }
            if (xiter == max_iter) {
                cout << "Error : only " << ck << " converged!" << endl;
                assert(false);
            }
        }
        for (int i = (int)bs.size() - 1; i >= k; i--)
            sigmas[i].deallocate(), bs[i].deallocate();
        q.deallocate();
        for (int i = k - 1; i >= 0; i--)
            sigmas[i].deallocate();
        ndav = xiter;
        return eigvals;
    }
    // Computes exp(t*H), the matrix exponential of a general matrix in
    // full, using the irreducible rational Pade approximation
    // Adapted from expokit fortran code dgpadm.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = 4 * m * m + ideg + 1
    // exp(tH) is located at work[ret:ret+m*m]
    static pair<int, int> expo_pade(int ideg, int m, const double *h, int ldh,
                                    double t, double *work) {
        static const double zero = 0.0, one = 1.0, mone = -1.0, two = 2.0;
        static const int inc = 1;
        // check restrictions on input parameters
        int mm = m * m;
        int iflag = 0;
        assert(ldh >= m);
        // initialize pointers
        int icoef = 0, ih2 = icoef + (ideg + 1), ip = ih2 + mm, iq = ip + mm,
            ifree = iq + mm;
        // scaling: seek ns such that ||t*H/2^ns|| < 1/2;
        // and set scale = t/2^ns ...
        memset(work, 0, sizeof(double) * m);
        for (int j = 0; j < m; j++)
            for (int i = 0; i < m; i++)
                work[i] += abs(h[j * m + i]);
        double hnorm = 0.0;
        for (int i = 0; i < m; i++)
            hnorm = max(hnorm, work[i]);
        hnorm = abs(t * hnorm);
        if (hnorm == 0.0) {
            cerr << "Error - null H in expo pade" << endl;
            abort();
        }
        int ns = max(0, (int)(log(hnorm) / log(2.0)) + 2);
        double scale = t / (double)(1LL << ns);
        double scale2 = scale * scale;
        // compute Pade coefficients
        int i = ideg + 1, j = 2 * ideg + 1;
        work[icoef] = 1.0;
        for (int k = 1; k <= ideg; k++)
            work[icoef + k] =
                work[icoef + k - 1] * (double)(i - k) / double(k * (j - k));
        // H2 = scale2*H*H ...
        dgemm("n", "n", &m, &m, &m, &scale2, h, &ldh, h, &ldh, &zero,
              work + ih2, &m);
        // initialize p (numerator) and q (denominator)
        memset(work + ip, 0, sizeof(double) * mm * 2);
        double cp = work[icoef + ideg - 1];
        double cq = work[icoef + ideg];
        for (int j = 0; j < m; j++)
            work[ip + j * (m + 1)] = cp, work[iq + j * (m + 1)] = cq;
        // Apply Horner rule
        int iodd = 1;
        for (int k = ideg - 1; k > 0; k--) {
            int iused = iodd * iq + (1 - iodd) * ip;
            dgemm("n", "n", &m, &m, &m, &one, work + iused, &m, work + ih2, &m,
                  &zero, work + ifree, &m);
            for (int j = 0; j < m; j++)
                work[ifree + j * (m + 1)] += work[icoef + k - 1];
            ip = (1 - iodd) * ifree + iodd * ip;
            iq = iodd * ifree + (1 - iodd) * iq;
            ifree = iused;
            iodd = 1 - iodd;
        }
        // Obtain (+/-)(I + 2*(p\q))
        int *iqp = iodd ? &iq : &ip;
        dgemm("n", "n", &m, &m, &m, &scale, work + *iqp, &m, h, &ldh, &zero,
              work + ifree, &m);
        *iqp = ifree;
        daxpy(&mm, &mone, work + ip, &inc, work + iq, &inc);
        dgesv(&m, &m, work + iq, &m, (int *)work + ih2, work + ip, &m, &iflag);
        if (iflag != 0) {
            cerr << "Problem in DGESV in expo pade" << endl;
            abort();
        }
        dscal(&mm, &two, work + ip, &inc);
        for (int j = 0; j < m; j++)
            work[ip + j * (m + 1)]++;
        int iput = ip;
        if (ns == 0 && iodd) {
            dscal(&mm, &mone, work + ip, &inc);
        } else {
            // squaring : exp(t*H) = (exp(t*H))^(2^ns)
            iodd = 1;
            for (int k = 0; k < ns; k++) {
                int iget = iodd * ip + (1 - iodd) * iq;
                iput = (1 - iodd) * ip + iodd * iq;
                dgemm("n", "n", &m, &m, &m, &one, work + iget, &m, work + iget,
                      &m, &zero, work + iput, &m);
                iodd = 1 - iodd;
            }
        }
        return make_pair(iput, ns);
    }
    // Computes w = exp(t*A)*v - for a (sparse) symmetric matrix A.
    // Adapted from expokit fortran code dsexpv.f:
    //   Roger B. Sidje (rbs@maths.uq.edu.au)
    //   EXPOKIT: Software Package for Computing Matrix Exponentials.
    //   ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
    // lwork = n*(m+1)+n+(m+2)^2+4*(m+2)^2+ideg+1
    template <typename MatMul>
    static int expo_krylov(MatMul op, int n, int m, double t, double *v,
                           double *w, double &tol, double anorm, double *work,
                           int lwork, bool iprint) {
        const int inc = 1;
        const double sqr1 = sqrt(0.1), zero = 0.0;
        const int mxstep = 500, mxreject = 0, ideg = 6;
        const double delta = 1.2, gamma = 0.9;
        int iflag = 0;
        if (lwork < n * (m + 2) + 5 * (m + 2) * (m + 2) + ideg + 1)
            iflag = -1;
        if (m >= n || m <= 0)
            iflag = -3;
        if (iflag != 0) {
            cerr << "bad sizes (in input of expo krylov)" << endl;
            abort();
        }
        // initializations
        int k1 = 2, mh = m + 2, iv = 0, ih = iv + n * (m + 1) + n;
        int ifree = ih + mh * mh, lfree = lwork - ifree, iexph;
        int ibrkflag = 0, mbrkdwn = m, nmult = 0, mx;
        int nreject = 0, nexph = 0, nscale = 0, ns = 0;
        double t_out = abs(t), tbrkdwn = 0.0, t_now = 0.0, t_new = 0.0;
        double step_min = t_out, step_max = 0.0, s_error = 0.0, x_error = 0.0;
        double err_loc;
        int nstep = 0;
        // machine precision
        double eps = 0.0;
        for (double p1 = 4.0 / 3.0, p2, p3; eps == 0.0;)
            p2 = p1 - 1.0, p3 = p2 + p2 + p2, eps = abs(p3 - 1.0);
        if (tol <= eps)
            tol = sqrt(eps);
        double rndoff = eps * anorm, break_tol = 1E-7;
        double sgn = t >= 0 ? 1.0 : -1.0;
        dcopy(&n, v, &inc, w, &inc);
        double beta = dnrm2(&n, w, &inc), vnorm = beta, hump = beta, avnorm;
        // obtain the very first stepsize
        double xm = 1.0 / (double)m, p1;
        p1 = tol * pow((m + 1) / 2.72, m + 1) * sqrt(2.0 * 3.14 * (m + 1));
        t_new = (1.0 / anorm) * pow(p1 / (4.0 * beta * anorm), xm);
        p1 = pow(10.0, round(log10(t_new) - sqr1) - 1);
        t_new = floor(t_new / p1 + 0.55) * p1;
        // step-by-step integration
        for (; t_now < t_out;) {
            nstep++;
            double t_step = min(t_out - t_now, t_new);
            p1 = 1.0 / beta;
            for (int i = 0; i < n; i++)
                work[iv + i] = p1 * w[i];
            memset(work + ih, 0, sizeof(double) * mh * mh);
            // Lanczos loop
            int j1v = iv + n;
            for (int j = 0; j < m; j++) {
                nmult++;
                op(work + j1v - n, work + j1v);
                if (j != 0) {
                    p1 = -work[ih + j * mh + j - 1];
                    daxpy(&n, &p1, work + j1v - n - n, &inc, work + j1v, &inc);
                }
                double hjj = -ddot(&n, work + j1v - n, &inc, work + j1v, &inc);
                daxpy(&n, &hjj, work + j1v - n, &inc, work + j1v, &inc);
                double hj1j = dnrm2(&n, work + j1v, &inc);
                work[ih + j * (mh + 1)] = -hjj;
                // if "happy breakdown" go straightforward at the end
                if (hj1j <= break_tol) {
                    if (iprint)
                        cout << "happy breakdown: mbrkdwn =" << j + 1
                             << " h = " << hj1j << endl;
                    k1 = 0, ibrkflag = 1;
                    mbrkdwn = j + 1, tbrkdwn = t_now;
                    t_step = t_out - t_now;
                    break;
                }
                work[ih + j * mh + j + 1] = hj1j;
                work[ih + (j + 1) * mh + j] = hj1j;
                hj1j = 1.0 / hj1j;
                dscal(&n, &hj1j, work + j1v, &inc);
                j1v += n;
            }
            if (k1 != 0) {
                nmult++;
                op(work + j1v - n, work + j1v);
                avnorm = dnrm2(&n, work + j1v, &inc);
            }
            // set 1 for the 2-corrected scheme
            work[ih + m * mh + m - 1] = 0.0;
            work[ih + m * mh + m + 1] = 1.0;
            // loop while ireject<mxreject until the tolerance is reached
            for (int ireject = 0;;) {
                // compute w = beta*V*exp(t_step*H)*e1
                nexph++;
                mx = mbrkdwn + k1;
                // irreducible rational Pade approximation
                auto xp = expo_pade(ideg, mx, work + ih, mh, sgn * t_step,
                                    work + ifree);
                iexph = xp.first + ifree, ns = xp.second;
                nscale += ns;
                // error estimate
                if (k1 == 0)
                    err_loc = tol;
                else {
                    double p1 = abs(work[iexph + m]) * beta;
                    double p2 = abs(work[iexph + m + 1]) * beta * avnorm;
                    if (p1 > 10.0 * p2)
                        err_loc = p2, xm = 1.0 / (double)m;
                    else if (p1 > p2)
                        err_loc = p1 * p2 / (p1 - p2), xm = 1.0 / (double)m;
                    else
                        err_loc = p1, xm = 1.0 / (double)(m - 1);
                }
                // reject the step-size if the error is not acceptable
                if (k1 != 0 && err_loc > delta * t_step * tol &&
                    (mxreject == 0 || ireject < mxreject)) {
                    double t_old = t_step;
                    t_step = gamma * t_step * pow(t_step * tol / err_loc, xm);
                    p1 = pow(10.0, round(log10(t_step) - sqr1) - 1);
                    t_step = floor(t_step / p1 + 0.55) * p1;
                    if (iprint)
                        cout << "t_step = " << t_old << " err_loc = " << err_loc
                             << " err_required = " << delta * t_old * tol
                             << endl
                             << "  stepsize rejected, stepping down to:"
                             << t_step << endl;
                    ireject++;
                    nreject++;
                    if (mxreject != 0 && ireject > mxreject) {
                        cerr << "failure in expo krylov: ---"
                             << " The requested tolerance is too high. Rerun "
                                "with a smaller value.";
                        abort();
                    }
                } else
                    break;
            }
            // now update w = beta*V*exp(t_step*H)*e1 and the hump
            mx = mbrkdwn + max(0, k1 - 1);
            dgemv("n", &n, &mx, &beta, work + iv, &n, work + iexph, &inc, &zero,
                  w, &inc);
            beta = dnrm2(&n, w, &inc);
            hump = max(hump, beta);
            // suggested value for the next stepsize
            t_new = gamma * t_step * pow(t_step * tol / err_loc, xm);
            p1 = pow(10.0, round(log10(t_new) - sqr1) - 1);
            t_new = floor(t_new / p1 + 0.55) * p1;
            err_loc = max(err_loc, rndoff);
            // update the time covered
            t_now += t_step;
            // display and keep some information
            if (iprint)
                cout << "integration " << nstep << " scale-square =" << ns
                     << " step_size = " << t_step << " err_loc = " << err_loc
                     << " next_step = " << t_new << endl;
            step_min = min(step_min, t_step);
            step_max = max(step_max, t_step);
            s_error += err_loc;
            x_error = max(x_error, err_loc);
            if (mxstep != 0 && nstep >= mxstep) {
                iflag = 1;
                break;
            }
        }
        return nmult;
    }
    template <typename MatMul>
    static int expo_apply(MatMul op, double t, double anorm, MatrixRef &v,
                          double consta = 0.0, bool iprint = false,
                          double conv_thrd = 5E-6,
                          int deflation_max_size = 20) {
        int vm = v.m, vn = v.n, n = vm * vn;
        if (n < 4) {
            const int lwork = 4 * n * n + 7;
            MatrixRef e = MatrixRef(dalloc->allocate(n), vm, vn);
            double *h = dalloc->allocate(n * n);
            double *work = dalloc->allocate(lwork);
            memset(e.data, 0, sizeof(double) * n);
            memset(h, 0, sizeof(double) * n * n);
            for (int i = 0; i < n; i++) {
                e.data[i] = 1.0;
                op(e, MatrixRef(h + i * n, vm, vn));
                h[i * (n + 1)] += consta;
                e.data[i] = 0.0;
            }
            int iptr = expo_pade(6, n, h, n, t, work).first;
            MatrixFunctions::multiply(MatrixRef(work + iptr, n, n), true, v,
                                      false, e, 1.0, 0.0);
            memcpy(v.data, e.data, sizeof(double) * n);
            dalloc->deallocate(work, lwork);
            dalloc->deallocate(h, n * n);
            e.deallocate();
            return n;
        }
        auto lop = [&op, consta, n, vm, vn](double *a, double *b) -> void {
            static int inc = 1;
            memset(b, 0, sizeof(double) * n);
            op(MatrixRef(a, vm, vn), MatrixRef(b, vm, vn));
            daxpy(&n, &consta, a, &inc, b, &inc);
        };
        int m = min(deflation_max_size, n - 1);
        int lwork = n * (m + 2) + 5 * (m + 2) * (m + 2) + 7;
        double *w = dalloc->allocate(n);
        double *work = dalloc->allocate(lwork);
        if (anorm < 1E-10)
            anorm = 1.0;
        int nmult = MatrixFunctions::expo_krylov(
            lop, n, m, t, v.data, w, conv_thrd, anorm, work, lwork, iprint);
        memcpy(v.data, w, sizeof(double) * n);
        dalloc->deallocate(work, lwork);
        dalloc->deallocate(w, n);
        return nmult;
    }
};

template <typename S> struct SparseMatrix {
    shared_ptr<SparseMatrixInfo<S>> info;
    double *data;
    double factor;
    size_t total_memory;
    SparseMatrix()
        : info(nullptr), data(nullptr), factor(1.0), total_memory(0) {}
    void load_data(const string &filename, bool load_info = false) {
        ifstream ifs(filename.c_str(), ios::binary);
        if (load_info) {
            info = make_shared<SparseMatrixInfo<S>>();
            info->load_data(ifs);
        } else
            info = nullptr;
        ifs.read((char *)&factor, sizeof(factor));
        ifs.read((char *)&total_memory, sizeof(total_memory));
        data = dalloc->allocate(total_memory);
        ifs.read((char *)data, sizeof(double) * total_memory);
        ifs.close();
    }
    void save_data(const string &filename, bool save_info = false) const {
        ofstream ofs(filename.c_str(), ios::binary);
        if (save_info)
            info->save_data(ofs);
        ofs.write((char *)&factor, sizeof(factor));
        ofs.write((char *)&total_memory, sizeof(total_memory));
        ofs.write((char *)data, sizeof(double) * total_memory);
        ofs.close();
    }
    void copy_data_from(const SparseMatrix &other) {
        assert(total_memory == other.total_memory);
        memcpy(data, other.data, sizeof(double) * total_memory);
    }
    void clear() { memset(data, 0, sizeof(double) * total_memory); }
    void allocate(const shared_ptr<SparseMatrixInfo<S>> &info,
                  double *ptr = 0) {
        this->info = info;
        total_memory = info->get_total_memory();
        if (total_memory == 0)
            return;
        if (ptr == 0) {
            data = dalloc->allocate(total_memory);
            memset(data, 0, sizeof(double) * total_memory);
        } else
            data = ptr;
    }
    void deallocate() {
        if (total_memory == 0) {
            assert(data == nullptr);
            return;
        }
        dalloc->deallocate(data, total_memory);
        total_memory = 0;
        data = nullptr;
    }
    void reallocate(int length) {
        double *ptr = dalloc->reallocate(data, total_memory, length);
        if (ptr != data && length != 0)
            memmove(ptr, data, length * sizeof(double));
        total_memory = length;
        data = length == 0 ? nullptr : ptr;
    }
    MatrixRef operator[](S q) const { return (*this)[info->find_state(q)]; }
    MatrixRef operator[](int idx) const {
        assert(idx != -1);
        return MatrixRef(data + info->n_states_total[idx],
                         info->n_states_bra[idx], info->n_states_ket[idx]);
    }
    double trace() const {
        double r = 0;
        for (int i = 0; i < info->n; i++)
            r += this->operator[](i).trace();
        return r;
    }
    double norm() const {
        return MatrixFunctions::norm(MatrixRef(data, total_memory, 1));
    }
    void left_canonicalize(const shared_ptr<SparseMatrix<S>> &rmat) {
        int nr = rmat->info->n, n = info->n;
        uint32_t *tmp = ialloc->allocate(nr + 1);
        memset(tmp, 0, sizeof(uint32_t) * (nr + 1));
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            assert(ir != -1);
            tmp[ir + 1] +=
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
        }
        for (int ir = 0; ir < nr; ir++)
            tmp[ir + 1] += tmp[ir];
        double *dt = dalloc->allocate(tmp[nr]);
        uint32_t *it = ialloc->allocate(nr);
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            uint32_t n_states =
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(dt + (tmp[ir] + it[ir]), data + info->n_states_total[i],
                   n_states * sizeof(double));
            it[ir] += n_states;
        }
        for (int ir = 0; ir < nr; ir++)
            assert(it[ir] == tmp[ir + 1] - tmp[ir]);
        for (int ir = 0; ir < nr; ir++) {
            uint32_t nxr = rmat->info->n_states_ket[ir],
                     nxl = (tmp[ir + 1] - tmp[ir]) / nxr;
            assert((tmp[ir + 1] - tmp[ir]) % nxr == 0 && nxl >= nxr);
            MatrixFunctions::qr(MatrixRef(dt + tmp[ir], nxl, nxr),
                                MatrixRef(dt + tmp[ir], nxl, nxr), (*rmat)[ir]);
        }
        memset(it, 0, sizeof(uint32_t) * nr);
        for (int i = 0; i < n; i++) {
            int ir = rmat->info->find_state(info->quanta[i].get_ket());
            uint32_t n_states =
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
            memcpy(data + info->n_states_total[i], dt + (tmp[ir] + it[ir]),
                   n_states * sizeof(double));
            it[ir] += n_states;
        }
        ialloc->deallocate(it, nr);
        dalloc->deallocate(dt, tmp[nr]);
        ialloc->deallocate(tmp, nr + 1);
    }
    void right_canonicalize(const shared_ptr<SparseMatrix<S>> &lmat) {
        int nl = lmat->info->n, n = info->n;
        uint32_t *tmp = ialloc->allocate(nl + 1);
        memset(tmp, 0, sizeof(uint32_t) * (nl + 1));
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            assert(il != -1);
            tmp[il + 1] +=
                (uint32_t)info->n_states_bra[i] * info->n_states_ket[i];
        }
        for (int il = 0; il < nl; il++)
            tmp[il + 1] += tmp[il];
        double *dt = dalloc->allocate(tmp[nl]);
        uint32_t *it = ialloc->allocate(nl);
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            uint32_t nxl = info->n_states_bra[i],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            uint32_t inr = info->n_states_ket[i];
            for (uint32_t k = 0; k < nxl; k++)
                memcpy(dt + (tmp[il] + it[il] + k * nxr),
                       data + info->n_states_total[i] + k * inr,
                       inr * sizeof(double));
            it[il] += inr * nxl;
        }
        for (int il = 0; il < nl; il++)
            assert(it[il] == tmp[il + 1] - tmp[il]);
        for (int il = 0; il < nl; il++) {
            uint32_t nxl = lmat->info->n_states_bra[il],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            assert((tmp[il + 1] - tmp[il]) % nxl == 0 && nxr >= nxl);
            MatrixFunctions::lq(MatrixRef(dt + tmp[il], nxl, nxr), (*lmat)[il],
                                MatrixRef(dt + tmp[il], nxl, nxr));
        }
        memset(it, 0, sizeof(uint32_t) * nl);
        for (int i = 0; i < n; i++) {
            int il = lmat->info->find_state(
                info->quanta[i].get_bra(info->delta_quantum));
            uint32_t nxl = info->n_states_bra[i],
                     nxr = (tmp[il + 1] - tmp[il]) / nxl;
            uint32_t inr = info->n_states_ket[i];
            for (uint32_t k = 0; k < nxl; k++)
                memcpy(data + info->n_states_total[i] + k * inr,
                       dt + (tmp[il] + it[il] + k * nxr), inr * sizeof(double));
            it[il] += inr * nxl;
        }
        ialloc->deallocate(it, nl);
        dalloc->deallocate(dt, tmp[nl]);
        ialloc->deallocate(tmp, nl + 1);
    }
    void left_multiply(const shared_ptr<SparseMatrix<S>> &lmat,
                       const StateInfo<S> &l, const StateInfo<S> &m,
                       const StateInfo<S> &r, const StateInfo<S> &old_fused,
                       const StateInfo<S> &old_fused_cinfo) const {
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = ib == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ib + 1];
            uint32_t p = info->n_states_total[i];
            for (int bb = old_fused_cinfo.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = old_fused_cinfo.quanta[bb].data >> 16,
                         ibbb = old_fused_cinfo.quanta[bb].data & 0xFFFFU;
                int il = lmat->info->find_state(l.quanta[ibba]);
                uint32_t lp = (uint32_t)m.n_states[ibbb] * r.n_states[ik];
                if (il != -1) {
                    assert(lmat->info->n_states_bra[il] ==
                           lmat->info->n_states_ket[il]);
                    assert(lmat->info->n_states_bra[il] == l.n_states[ibba]);
                    MatrixRef tmp(nullptr, l.n_states[ibba], lp);
                    tmp.allocate();
                    MatrixFunctions::multiply(
                        (*lmat)[il], false,
                        MatrixRef(data + p, l.n_states[ibba], lp), false, tmp,
                        lmat->factor, 0.0);
                    memcpy(data + p, tmp.data, sizeof(double) * tmp.size());
                    tmp.deallocate();
                }
                p += l.n_states[ibba] * lp;
            }
            assert(p == (i != info->n - 1 ? info->n_states_total[i + 1]
                                          : total_memory));
        }
    }
    void right_multiply(const shared_ptr<SparseMatrix<S>> &rmat,
                        const StateInfo<S> &l, const StateInfo<S> &m,
                        const StateInfo<S> &r, const StateInfo<S> &old_fused,
                        const StateInfo<S> &old_fused_cinfo) const {
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = info->is_wavefunction ? -info->quanta[i].get_ket()
                                          : info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int kked = ik == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ik + 1];
            uint32_t p = info->n_states_total[i];
            for (int kk = old_fused_cinfo.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = old_fused_cinfo.quanta[kk].data >> 16,
                         ikkb = old_fused_cinfo.quanta[kk].data & 0xFFFFU;
                int ir = rmat->info->find_state(r.quanta[ikkb]);
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                if (ir != -1) {
                    assert(rmat->info->n_states_bra[ir] ==
                           rmat->info->n_states_ket[ir]);
                    assert(rmat->info->n_states_bra[ir] == r.n_states[ikkb]);
                    MatrixRef tmp(nullptr, m.n_states[ikka], r.n_states[ikkb]);
                    tmp.allocate();
                    for (int j = 0; j < l.n_states[ib]; j++) {
                        MatrixFunctions::multiply(
                            MatrixRef(data + p + j * old_fused.n_states[ik],
                                      m.n_states[ikka], r.n_states[ikkb]),
                            false, (*rmat)[ir], false, tmp, rmat->factor, 0.0);
                        memcpy(data + p + j * old_fused.n_states[ik], tmp.data,
                               sizeof(double) * tmp.size());
                    }
                    tmp.deallocate();
                }
                p += lp;
            }
        }
    }
    void randomize(double a = 0.0, double b = 1.0) const {
        Random::fill_rand_double(data, total_memory, a, b);
    }
    void contract(const shared_ptr<SparseMatrix> &lmat,
                  const shared_ptr<SparseMatrix> &rmat) {
        assert(info->is_wavefunction);
        if (lmat->info->is_wavefunction)
            for (int i = 0; i < info->n; i++) {
                int il = lmat->info->find_state(info->quanta[i]);
                int ir = rmat->info->find_state(-info->quanta[i].get_ket());
                if (il != -1 && ir != -1)
                    MatrixFunctions::multiply((*lmat)[il], false, (*rmat)[ir],
                                              false, (*this)[i],
                                              lmat->factor * rmat->factor, 0.0);
            }
        else
            for (int i = 0; i < info->n; i++) {
                int il = lmat->info->find_state(
                    info->quanta[i].get_bra(info->delta_quantum));
                int ir = rmat->info->find_state(info->quanta[i]);
                if (il != -1 && ir != -1)
                    MatrixFunctions::multiply((*lmat)[il], false, (*rmat)[ir],
                                              false, (*this)[i],
                                              lmat->factor * rmat->factor, 0.0);
            }
    }
    void swap_to_fused_left(const shared_ptr<SparseMatrix<S>> &mat,
                            const StateInfo<S> &l, const StateInfo<S> &m,
                            const StateInfo<S> &r,
                            const StateInfo<S> &old_fused,
                            const StateInfo<S> &old_fused_cinfo,
                            const StateInfo<S> &new_fused,
                            const StateInfo<S> &new_fused_cinfo) {
        assert(mat->info->is_wavefunction);
        factor = mat->factor;
        map<uint32_t, map<uint16_t, pair<uint32_t, uint32_t>>> mp;
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = -mat->info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = old_fused.find_state(ket);
            int kked = ik == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ik + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int kk = old_fused_cinfo.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = old_fused_cinfo.quanta[kk].data >> 16,
                         ikkb = old_fused_cinfo.quanta[kk].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ikka] * r.n_states[ikkb];
                mp[(ib << 16) + ikka][ikkb] =
                    make_pair(p, old_fused.n_states[ik]);
                p += lp;
            }
        }
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = -info->quanta[i].get_ket();
            int ib = new_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = ib == new_fused.n - 1 ? new_fused_cinfo.n
                                             : new_fused_cinfo.n_states[ib + 1];
            double *ptr = data + info->n_states_total[i];
            for (int bb = new_fused_cinfo.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = new_fused_cinfo.quanta[bb].data >> 16,
                         ibbb = new_fused_cinfo.quanta[bb].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ibbb] * r.n_states[ik];
                if (mp.count(new_fused_cinfo.quanta[bb].data) &&
                    mp[new_fused_cinfo.quanta[bb].data].count(ik)) {
                    pair<uint32_t, uint32_t> &t =
                        mp.at(new_fused_cinfo.quanta[bb].data).at(ik);
                    for (int j = 0; j < l.n_states[ibba]; j++)
                        memcpy(ptr + j * lp, mat->data + t.first + j * t.second,
                               lp * sizeof(double));
                }
                ptr += (size_t)l.n_states[ibba] * lp;
            }
            assert(ptr - data == (i != info->n - 1 ? info->n_states_total[i + 1]
                                                   : total_memory));
        }
    }
    void swap_to_fused_right(const shared_ptr<SparseMatrix<S>> &mat,
                             const StateInfo<S> &l, const StateInfo<S> &m,
                             const StateInfo<S> &r,
                             const StateInfo<S> &old_fused,
                             const StateInfo<S> &old_fused_cinfo,
                             const StateInfo<S> &new_fused,
                             const StateInfo<S> &new_fused_cinfo) {
        assert(mat->info->is_wavefunction);
        factor = mat->factor;
        map<uint32_t, map<uint16_t, pair<uint32_t, uint32_t>>> mp;
        for (int i = 0; i < mat->info->n; i++) {
            S bra = mat->info->quanta[i].get_bra(mat->info->delta_quantum);
            S ket = -mat->info->quanta[i].get_ket();
            int ib = old_fused.find_state(bra);
            int ik = r.find_state(ket);
            int bbed = ib == old_fused.n - 1 ? old_fused_cinfo.n
                                             : old_fused_cinfo.n_states[ib + 1];
            uint32_t p = mat->info->n_states_total[i];
            for (int bb = old_fused_cinfo.n_states[ib]; bb < bbed; bb++) {
                uint16_t ibba = old_fused_cinfo.quanta[bb].data >> 16,
                         ibbb = old_fused_cinfo.quanta[bb].data & (0xFFFFU);
                uint32_t lp = (uint32_t)m.n_states[ibbb] * r.n_states[ik];
                mp[(ibbb << 16) + ik][ibba] = make_pair(p, lp);
                p += l.n_states[ibba] * lp;
            }
            assert(p == (i != mat->info->n - 1
                             ? mat->info->n_states_total[i + 1]
                             : mat->total_memory));
        }
        for (int i = 0; i < info->n; i++) {
            S bra = info->quanta[i].get_bra(info->delta_quantum);
            S ket = -info->quanta[i].get_ket();
            int ib = l.find_state(bra);
            int ik = new_fused.find_state(ket);
            int kked = ik == new_fused.n - 1 ? new_fused_cinfo.n
                                             : new_fused_cinfo.n_states[ik + 1];
            double *ptr = data + info->n_states_total[i];
            uint32_t lp = new_fused.n_states[ik];
            for (int kk = new_fused_cinfo.n_states[ik]; kk < kked; kk++) {
                uint16_t ikka = new_fused_cinfo.quanta[kk].data >> 16,
                         ikkb = new_fused_cinfo.quanta[kk].data & (0xFFFFU);
                if (mp.count(new_fused_cinfo.quanta[kk].data) &&
                    mp[new_fused_cinfo.quanta[kk].data].count(ib)) {
                    pair<uint32_t, uint32_t> &t =
                        mp.at(new_fused_cinfo.quanta[kk].data).at(ib);
                    for (int j = 0; j < l.n_states[ib]; j++)
                        memcpy(ptr + j * lp, mat->data + t.first + j * t.second,
                               t.second * sizeof(double));
                }
                ptr += (size_t)m.n_states[ikka] * r.n_states[ikkb];
            }
        }
    }
    friend ostream &operator<<(ostream &os, const SparseMatrix<S> &c) {
        os << "DATA = [ ";
        for (int i = 0; i < c.total_memory; i++)
            os << setw(20) << setprecision(14) << c.data[i] << " ";
        os << "]" << endl;
        return os;
    }
};

struct BatchGEMM {
    const CBLAS_LAYOUT layout = CblasRowMajor;
    vector<CBLAS_TRANSPOSE> ta, tb;
    vector<int> n, m, k, gp, lda, ldb, ldc;
    vector<double> alpha, beta;
    vector<const double *> a, b;
    vector<double *> c;
    size_t work;
    BatchGEMM() : work(0) {}
    void dgemm_group(bool conja, bool conjb, int m, int n, int k, double alpha,
                     int lda, int ldb, double beta, int ldc, int gc) {
        ta.push_back(conja ? CblasTrans : CblasNoTrans);
        tb.push_back(conjb ? CblasTrans : CblasNoTrans);
        this->m.push_back(m), this->n.push_back(n), this->k.push_back(k);
        this->alpha.push_back(alpha), this->beta.push_back(beta);
        this->lda.push_back(lda), this->ldb.push_back(ldb),
            this->ldc.push_back(ldc);
        this->gp.push_back(gc);
    }
    void dgemm_array(const double *a, const double *b, double *c) {
        this->a.push_back(a), this->b.push_back(b), this->c.push_back(c);
    }
    void dgemm(bool conja, bool conjb, int m, int n, int k, double alpha,
               const double *a, int lda, const double *b, int ldb, double beta,
               double *c, int ldc) {
        dgemm_group(conja, conjb, m, n, k, alpha, lda, ldb, beta, ldc, 1);
        dgemm_array(a, b, c);
    }
    void iadd(double *a, const double *b, int n, double scale = 1.0,
              double cfactor = 1.0) {
        static double x = 1.0;
        this->dgemm(false, false, n, 1, 1, scale, b, 1, &x, 1, cfactor, a, 1);
    }
    void iscale(double *a, int n, double scale = 1.0) {
        static double x = 1.0;
        this->dgemm(false, false, n, 1, 1, 0.0, a, 1, &x, 1, scale, a, 1);
    }
    void tensor_product(const MatrixRef &a, bool conja, const MatrixRef &b,
                        bool conjb, const MatrixRef &c, double scale,
                        uint32_t stride, double cfactor = 1.0) {
        assert((a.m == 1 && a.n == 1) || (b.m == 1 && b.n == 1));
        if (a.m == 1 && a.n == 1) {
            if (!conjb && b.n == c.n)
                this->dgemm(false, false, b.m * b.n, 1, 1, scale, b.data, 1,
                            a.data, 1, cfactor, &c(0, stride), 1);
            else if (!conjb) {
                this->dgemm_group(false, false, b.n, 1, 1, scale, 1, 1, cfactor,
                                  1, b.m);
                for (int k = 0; k < b.m; k++)
                    this->dgemm_array(&b(k, 0), a.data, &c(k, stride));
            } else {
                this->dgemm_group(false, false, b.m, 1, 1, scale, b.n, 1,
                                  cfactor, 1, b.n);
                for (int k = 0; k < b.n; k++)
                    this->dgemm_array(&b(0, k), a.data, &c(k, stride));
            }
        } else {
            if (!conja && a.n == c.n)
                this->dgemm(false, false, a.m * a.n, 1, 1, scale, a.data, 1,
                            b.data, 1, cfactor, &c(0, stride), 1);
            else if (!conja) {
                this->dgemm_group(false, false, a.n, 1, 1, scale, 1, 1, cfactor,
                                  1, a.m);
                for (int k = 0; k < a.m; k++)
                    this->dgemm_array(&a(k, 0), b.data, &c(k, stride));
            } else {
                this->dgemm_group(false, false, a.m, 1, 1, scale, a.n, 1,
                                  cfactor, 1, a.n);
                for (int k = 0; k < a.n; k++)
                    this->dgemm_array(&a(0, k), b.data, &c(k, stride));
            }
        }
    }
    void multiply(const MatrixRef &a, bool conja, const MatrixRef &b,
                  bool conjb, const MatrixRef &c, double scale,
                  double cfactor) {
        this->dgemm(conja, conjb, c.m, c.n, conjb ? b.n : b.m, scale, a.data,
                    a.n, b.data, b.n, cfactor, c.data, c.n);
    }
    void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                 const MatrixRef &c, double scale) {
        this->dgemm(false, true, a.n, b.n, 1, scale, a.data, a.n + 1, b.data,
                    b.n + 1, 1.0, c.data, c.n);
    }
    void perform(int ii = 0, int kk = 0, int nn = 0) {
        if (nn != 0 || gp.size() != 0)
            cblas_dgemm_batch(layout, &ta[ii], &tb[ii], &m[ii], &n[ii], &k[ii],
                              &alpha[ii], &a[kk], &lda[ii], &b[kk], &ldb[ii],
                              &beta[ii], &c[kk], &ldc[ii],
                              nn == 0 ? (int)gp.size() : nn, &gp[ii]);
    }
    void clear() {
        ta.clear(), tb.clear();
        n.clear(), m.clear(), k.clear(), gp.clear();
        lda.clear(), ldb.clear(), ldc.clear();
        alpha.clear(), beta.clear();
        a.clear(), b.clear(), c.clear();
        work = 0;
    }
    friend ostream &operator<<(ostream &os, const BatchGEMM &c) {
        for (size_t i = 0, k = 0; i < c.gp.size(); k += c.gp[i], i++) {
            os << "[" << setw(3) << i << "] :: GC=" << c.gp[i]
               << " TA=" << (c.ta[i] == CblasTrans ? "T" : "N")
               << " TB=" << (c.tb[i] == CblasTrans ? "T" : "N")
               << " M=" << c.m[i] << " N=" << c.n[i] << " K=" << c.k[i]
               << " ALPHA=" << c.alpha[i] << " BETA=" << c.beta[i]
               << " LDA=" << c.lda[i] << " LDB=" << c.ldb[i]
               << " LDC=" << c.ldc[i] << endl;
            for (size_t j = 0; j < c.gp[i]; j++)
                os << setw(9) << ">" << setw(3) << j << hex
                   << " :: A=" << c.a[k + j] << " B=" << c.b[k + j]
                   << " C=" << c.c[k + j] << dec << endl;
        }
        return os;
    }
};

struct BatchGEMMRef {
    shared_ptr<BatchGEMM> batch;
    int i, k, n, nk;
    size_t nflop, work, rwork = 0;
    int ipost = 0;
    BatchGEMMRef(const shared_ptr<BatchGEMM> &batch, size_t nflop, size_t work,
                 int i, int k, int n, int nk)
        : batch(batch), nflop(nflop), work(work), i(i), k(k), n(n), nk(nk) {}
    void perform() {
        if (n != 0)
            batch->perform(i, k, n);
    }
};

enum struct SeqTypes : uint8_t { None, Simple, Auto };

struct BatchGEMMSeq {
    vector<shared_ptr<BatchGEMM>> batch;
    vector<shared_ptr<BatchGEMM>> post_batch;
    vector<BatchGEMMRef> refs;
    size_t cumulative_nflop = 0;
    size_t max_batch_flops = 1LU << 30;
    size_t max_work, max_rwork;
    double *work, *rwork;
    SeqTypes mode;
    BatchGEMMSeq(size_t max_batch_flops = 1LU << 30,
                 SeqTypes mode = SeqTypes::None)
        : max_batch_flops(max_batch_flops), mode(mode) {
        batch.push_back(make_shared<BatchGEMM>());
        batch.push_back(make_shared<BatchGEMM>());
    }
    void iadd(const MatrixRef &a, const MatrixRef &b, double scale = 1.0,
              double cfactor = 1.0, bool conj = false) {
        static double x = 1;
        if (!conj)
            batch[1]->iadd(a.data, b.data, a.m * a.n, scale, cfactor);
        else
            batch[1]->tensor_product(b, conj, MatrixRef(&x, 1, 1), false, a,
                                     scale, 0, cfactor);
    }
    void rotate(const MatrixRef &a, const MatrixRef &c, const MatrixRef &bra,
                bool conj_bra, const MatrixRef &ket, bool conj_ket,
                double scale) {
        MatrixRef work((double *)0 + batch[0]->work, a.m,
                       conj_ket ? ket.m : ket.n);
        batch[0]->multiply(a, false, ket, conj_ket, work, 1.0, 0.0);
        batch[1]->multiply(bra, conj_bra, work, false, c, scale, 1.0);
        batch[0]->work += work.size();
        batch[1]->work += work.size();
    }
    void tensor_product_diagonal(const MatrixRef &a, const MatrixRef &b,
                                 const MatrixRef &c, double scale) {
        batch[1]->tensor_product_diagonal(a, b, c, scale);
    }
    void tensor_product(const MatrixRef &a, bool conja, const MatrixRef &b,
                        bool conjb, const MatrixRef &c, double scale,
                        uint32_t stride) {
        batch[1]->tensor_product(a, conja, b, conjb, c, scale, stride);
    }
    void divide_batch() {
        size_t cur = 0, cur0 = 0, cwork = 0, pwork = 0;
        int ip = 0, kp = 0;
        for (int i = 0, k = 0; i < batch[1]->gp.size();
             k += batch[1]->gp[i++]) {
            cur += (size_t)batch[1]->m[i] * batch[1]->n[i] * batch[1]->k[i] *
                   batch[1]->gp[i];
            if (batch[0]->gp.size() != 0) {
                cur0 += (size_t)batch[0]->m[i] * batch[0]->n[i] *
                        batch[0]->k[0] * batch[0]->gp[i];
                cwork += (size_t)batch[0]->m[i] * batch[0]->n[i];
            }
            if (max_batch_flops != 0 && cur >= max_batch_flops) {
                if (batch[0]->gp.size() != 0)
                    refs.push_back(BatchGEMMRef(batch[0], cur0, cwork - pwork,
                                                ip, kp, i + 1 - ip,
                                                k + batch[0]->gp[i] - kp));
                refs.push_back(BatchGEMMRef(batch[1], cur, cwork - pwork, ip,
                                            kp, i + 1 - ip,
                                            k + batch[1]->gp[i] - kp));
                if (pwork != 0) {
                    for (size_t kk = kp; kk < k + batch[1]->gp[i]; kk++)
                        batch[0]->c[kk] -= pwork;
                    for (size_t kk = kp; kk < k + batch[1]->gp[i]; kk++)
                        batch[1]->b[kk] -= pwork;
                }
                cur = 0, cur0 = 0, ip = i + 1, kp = k + batch[1]->gp[i];
                pwork = cwork;
            }
        }
        if (cur != 0) {
            if (batch[0]->gp.size() != 0)
                refs.push_back(BatchGEMMRef(batch[0], cur0, cwork - pwork, ip,
                                            kp, batch[1]->gp.size() - ip,
                                            batch[0]->c.size() - kp));
            refs.push_back(BatchGEMMRef(batch[1], cur, cwork - pwork, ip, kp,
                                        batch[1]->gp.size() - ip,
                                        batch[1]->b.size() - kp));
            if (pwork != 0) {
                for (size_t kk = kp; kk < batch[0]->c.size(); kk++)
                    batch[0]->c[kk] -= pwork;
                for (size_t kk = kp; kk < batch[1]->b.size(); kk++)
                    batch[1]->b[kk] -= pwork;
            }
        }
    }
    bool check() {
        for (int ib = !!batch[0]->gp.size(),
                 db = batch[0]->gp.size() == 0 ? 1 : 2;
             ib < refs.size(); ib += db) {
            shared_ptr<BatchGEMM> b = refs[ib].batch;
            if (refs[ib].nk == 0)
                continue;
            double **ptr = (double **)dalloc->allocate(refs[ib].nk);
            uint32_t *len = (uint32_t *)ialloc->allocate(refs[ib].nk);
            uint32_t *idx = (uint32_t *)ialloc->allocate(refs[ib].nk);
            int xi = refs[ib].i, xk = refs[ib].k;
            for (int i = 0, k = 0; i < refs[ib].n; k += b->gp[xi + i++]) {
                for (int kk = k; kk < k + b->gp[xi + i]; kk++)
                    ptr[kk] = b->c[xk + kk],
                    len[kk] = b->m[xi + i] * b->n[xi + i];
            }
            for (int kk = 0; kk < refs[ib].nk; kk++)
                idx[kk] = kk;
            sort(idx, idx + refs[ib].nk,
                 [ptr](uint32_t a, uint32_t b) { return ptr[a] < ptr[b]; });
            for (int kk = 1; kk < refs[ib].nk; kk++)
                if (!(ptr[idx[kk]] >= ptr[idx[kk - 1]] + len[idx[kk - 1]]))
                    return false;
            ialloc->deallocate(idx, refs[ib].nk);
            ialloc->deallocate(len, refs[ib].nk);
            dalloc->deallocate((double *)ptr, refs[ib].nk);
        }
        return true;
    }
    void prepare() {
        divide_batch();
        for (int ib = !!batch[0]->gp.size(),
                 db = batch[0]->gp.size() == 0 ? 1 : 2;
             ib < refs.size(); ib += db) {
            shared_ptr<BatchGEMM> b = refs[ib].batch;
            double **ptr = (double **)dalloc->allocate(refs[ib].nk);
            uint32_t *len = (uint32_t *)ialloc->allocate(refs[ib].nk);
            uint32_t *pos = (uint32_t *)ialloc->allocate(refs[ib].nk);
            uint32_t *idx = (uint32_t *)ialloc->allocate(refs[ib].nk);
            int xi = refs[ib].i, xk = refs[ib].k;
            for (int i = 0, k = 0; i < refs[ib].n; k += b->gp[xi + i++]) {
                for (int kk = k; kk < k + b->gp[xi + i]; kk++)
                    ptr[kk] = b->c[xk + kk],
                    len[kk] = b->m[xi + i] * b->n[xi + i], pos[kk] = xk + kk;
            }
            for (int kk = 0; kk < refs[ib].nk; kk++)
                idx[kk] = kk;
            sort(idx, idx + refs[ib].nk,
                 [ptr](uint32_t a, uint32_t b) { return ptr[a] < ptr[b]; });
            vector<double *> ptrs;
            vector<uint32_t> lens;
            vector<map<pair<uint32_t, uint32_t>, vector<int>>> shifts;
            for (int kk = 0; kk < refs[ib].nk; kk++) {
                if (ptrs.size() == 0) {
                    ptrs.push_back(ptr[idx[kk]]);
                    lens.push_back(len[idx[kk]]);
                    shifts.push_back(
                        map<pair<uint32_t, uint32_t>, vector<int>>());
                    shifts.back()[make_pair(0, len[idx[kk]])].push_back(
                        pos[idx[kk]]);
                } else if (ptr[idx[kk]] >= ptrs.back() &&
                           ptr[idx[kk]] < ptrs.back() + lens.back()) {
                    shifts
                        .back()[make_pair(ptr[idx[kk]] - ptrs.back(),
                                          len[idx[kk]])]
                        .push_back(pos[idx[kk]]);
                    if (ptr[idx[kk]] + len[idx[kk]] > ptrs.back() + lens.back())
                        lens.back() = ptr[idx[kk]] + len[idx[kk]] - ptrs.back();
                } else if (ptr[idx[kk]] == ptrs.back() + lens.back()) {
                    lens.back() += len[idx[kk]];
                    shifts
                        .back()[make_pair(ptr[idx[kk]] - ptrs.back(),
                                          len[idx[kk]])]
                        .push_back(pos[idx[kk]]);
                } else {
                    ptrs.push_back(ptr[idx[kk]]);
                    lens.push_back(len[idx[kk]]);
                    shifts.push_back(
                        map<pair<uint32_t, uint32_t>, vector<int>>());
                    shifts.back()[make_pair(0, len[idx[kk]])].push_back(
                        pos[idx[kk]]);
                }
            }
            ialloc->deallocate(idx, refs[ib].nk);
            ialloc->deallocate(pos, refs[ib].nk);
            ialloc->deallocate(len, refs[ib].nk);
            dalloc->deallocate((double *)ptr, refs[ib].nk);
            vector<size_t> pwork;
            pwork.reserve(ptrs.size());
            vector<vector<pair<uint32_t, vector<int>>>> rshifts;
            for (size_t p = 0; p < ptrs.size(); p++) {
                pwork.push_back(0);
                rshifts.push_back(vector<pair<uint32_t, vector<int>>>());
                uint32_t sh = 0, le = 0;
                for (auto &r : shifts[p]) {
                    if (r.first.first > sh || le == 0)
                        sh = r.first.first, le = r.first.second;
                    if (r.first.first == sh && r.first.second == le)
                        rshifts.back().push_back(make_pair(sh, r.second));
                }
                size_t q = 0;
                for (auto &r : shifts[p]) {
                    if (r.first.first != rshifts.back()[q].first) {
                        assert(r.first.first == rshifts.back()[q - 1].first);
                        rshifts.back()[q - 1].second.insert(
                            rshifts.back()[q - 1].second.end(),
                            r.second.begin(), r.second.end());
                        for (size_t qq = q; qq < rshifts.back().size(); qq++)
                            if (rshifts.back()[qq].first > r.first.first &&
                                rshifts.back()[qq].first <
                                    r.first.first + r.first.second)
                                for (size_t u = 0; u < r.second.size(); u++)
                                    rshifts.back()[qq].second.push_back(-1);
                    } else
                        q++;
                }
                for (auto &r : rshifts[p])
                    if (r.second.size() > pwork.back())
                        pwork.back() = r.second.size();
            }
            refs[ib].rwork = 0;
            for (size_t p = 0; p < ptrs.size(); p++)
                refs[ib].rwork += pwork[p] * lens[p];
            double *rr = 0;
            for (size_t p = 0; p < ptrs.size(); p++) {
                for (auto &r : rshifts[p]) {
                    for (size_t q = 0; q < r.second.size(); q++)
                        if (r.second[q] != -1)
                            b->c[r.second[q]] = rr + q * lens[p] + r.first;
                }
                rr += pwork[p] * lens[p];
            }
            size_t max_pwork = *max_element(pwork.begin(), pwork.end());
            size_t ppost = post_batch.size(), ipost = 0;
            while (max_pwork > (1 << ipost))
                ipost++;
            refs[ib].ipost = ipost + 1;
            for (size_t ip = 0; ip < ipost + 1; ip++)
                post_batch.push_back(make_shared<BatchGEMM>());
            rr = 0;
            for (size_t p = 0; p < ptrs.size(); p++) {
                for (size_t ip = 0, ipx = 1, ipy = 2; ip < ipost;
                     ip++, ipx <<= 1, ipy <<= 1)
                    for (size_t q = 0; q + ipx < pwork[p]; q += ipy)
                        post_batch[ppost + ip]->iadd(rr + q * lens[p],
                                                     rr + (q + ipx) * lens[p],
                                                     lens[p]);
                post_batch[ppost + ipost]->iadd(ptrs[p], rr, lens[p]);
                rr += pwork[p] * lens[p];
            }
        }
    }
    void allocate() {
        max_work = max_rwork = 0;
        for (int ib = 0; ib < refs.size(); ib++) {
            max_work = max(max_work, refs[ib].work);
            max_rwork = max(max_rwork, refs[ib].rwork);
        }
        if (max_work != 0) {
            work = dalloc->allocate(max_work);
            size_t shift = work - (double *)0;
            for (size_t i = 0; i < batch[0]->c.size(); i++)
                batch[0]->c[i] += shift;
            for (size_t i = 0; i < batch[1]->b.size(); i++)
                batch[1]->b[i] += shift;
        }
        if (max_rwork != 0) {
            rwork = dalloc->allocate(max_rwork);
            size_t shift = rwork - (double *)0;
            size_t ipost = 0;
            for (size_t i = 0; i < batch[1]->c.size(); i++)
                batch[1]->c[i] += shift;
            for (int ib = !!batch[0]->gp.size(),
                     db = batch[0]->gp.size() == 0 ? 1 : 2;
                 ib < refs.size(); ib += db) {
                for (size_t k = ipost; k < ipost + refs[ib].ipost - 1; k++)
                    for (size_t i = 0; i < post_batch[k]->a.size(); i++) {
                        post_batch[k]->a[i] += shift;
                        post_batch[k]->c[i] += shift;
                    }
                for (size_t i = 0, p = ipost + refs[ib].ipost - 1;
                     i < post_batch[p]->a.size(); i++)
                    post_batch[p]->a[i] += shift;
                ipost += refs[ib].ipost;
            }
        }
    }
    void deallocate() {
        if (max_rwork != 0)
            dalloc->deallocate(rwork, max_rwork);
        if (max_work != 0)
            dalloc->deallocate(work, max_work);
    }
    void simple_perform() {
        divide_batch();
        assert(check());
        allocate();
        perform();
        deallocate();
        clear();
    }
    void auto_perform() {
        prepare();
        allocate();
        perform();
        deallocate();
        clear();
    }
    void perform() {
        size_t ipost = 0;
        for (auto b : refs) {
            if (b.rwork != 0)
                memset(rwork, 0, sizeof(double) * b.rwork);
            cumulative_nflop += b.nflop;
            b.perform();
            for (size_t ib = ipost; ib < ipost + b.ipost; ib++)
                post_batch[ib]->perform();
            ipost += b.ipost;
        }
        assert(ipost == post_batch.size());
    }
    void operator()(const MatrixRef &c, const MatrixRef &v) {
        size_t cshift = c.data - (double *)0;
        size_t vshift = v.data - (double *)0;
        for (size_t i = 0; i < batch[0]->a.size(); i++)
            batch[0]->a[i] += cshift;
        size_t ipost = 0;
        for (auto b : refs) {
            if (b.ipost != 0)
                for (size_t i = 0;
                     i < post_batch[ipost + b.ipost - 1]->c.size(); i++)
                    post_batch[ipost + b.ipost - 1]->c[i] += vshift;
            ipost += b.ipost;
        }
        perform();
        for (size_t i = 0; i < batch[0]->a.size(); i++)
            batch[0]->a[i] -= cshift;
        ipost = 0;
        for (auto b : refs) {
            if (b.ipost != 0)
                for (size_t i = 0;
                     i < post_batch[ipost + b.ipost - 1]->c.size(); i++)
                    post_batch[ipost + b.ipost - 1]->c[i] -= vshift;
            ipost += b.ipost;
        }
    }
    void clear() {
        for (auto b : batch)
            b->clear();
        post_batch.clear();
        refs.clear();
        max_rwork = max_work = 0;
    }
    friend ostream &operator<<(ostream &os, const BatchGEMMSeq &c) {
        os << endl;
        os << "[0] SIZE = " << c.batch[0]->gp.size()
           << " WORK = " << c.batch[0]->work << endl;
        os << "[1] SIZE = " << c.batch[1]->gp.size()
           << " WORK = " << c.batch[1]->work << endl;
        return os;
    }
};

template <typename S> struct OperatorFunctions {
    shared_ptr<CG<S>> cg;
    shared_ptr<BatchGEMMSeq> seq = nullptr;
    OperatorFunctions(const shared_ptr<CG<S>> &cg) : cg(cg) {
        seq = make_shared<BatchGEMMSeq>(0, SeqTypes::None);
    }
    // a += b * scale
    void iadd(SparseMatrix<S> &a, const SparseMatrix<S> &b, double scale = 1.0,
              bool conj = false) const {
        if (a.info == b.info && !conj) {
            if (seq->mode != SeqTypes::None) {
                seq->iadd(MatrixRef(a.data, 1, a.total_memory),
                          MatrixRef(b.data, 1, b.total_memory),
                          scale * b.factor, a.factor);
                a.factor = 1.0;
            } else {
                if (a.factor != 1.0) {
                    MatrixFunctions::iscale(
                        MatrixRef(a.data, 1, a.total_memory), a.factor);
                    a.factor = 1.0;
                }
                if (scale != 0.0)
                    MatrixFunctions::iadd(MatrixRef(a.data, 1, a.total_memory),
                                          MatrixRef(b.data, 1, b.total_memory),
                                          scale * b.factor);
            }
        } else {
            S bdq = b.info->delta_quantum;
            for (int ia = 0, ib; ia < a.info->n; ia++) {
                S bra = a.info->quanta[ia].get_bra(a.info->delta_quantum);
                S ket = a.info->quanta[ia].get_ket();
                S bq = conj ? bdq.combine(ket, bra) : bdq.combine(bra, ket);
                if (bq != S(0xFFFFFFFF) &&
                    ((ib = b.info->find_state(bq)) != -1)) {
                    double factor = scale * b.factor;
                    if (conj)
                        factor *= cg->transpose_cg(bdq.twos(), bra.twos(),
                                                   ket.twos());
                    if (seq->mode != SeqTypes::None)
                        seq->iadd(a[ia], b[ib], factor, a.factor, conj);
                    else {
                        if (a.factor != 1.0) {
                            MatrixFunctions::iscale(a[ia], a.factor);
                        }
                        if (scale != 0.0)
                            MatrixFunctions::iadd(a[ia], b[ib], factor, conj);
                    }
                }
            }
            a.factor = 1;
        }
    }
    void tensor_rotate(const SparseMatrix<S> &a, const SparseMatrix<S> &c,
                       const SparseMatrix<S> &rot_bra,
                       const SparseMatrix<S> &rot_ket, bool trans,
                       double scale = 1.0) const {
        scale = scale * a.factor * rot_bra.factor * rot_ket.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, cdq = c.info->delta_quantum;
        assert(adq == cdq && a.info->n >= c.info->n);
        for (int ic = 0, ia = 0; ic < c.info->n; ia++, ic++) {
            while (a.info->quanta[ia] != c.info->quanta[ic])
                ia++;
            S cq = c.info->quanta[ic].get_bra(cdq);
            S cqprime = c.info->quanta[ic].get_ket();
            int ibra = rot_bra.info->find_state(cq);
            int iket = rot_ket.info->find_state(cqprime);
            if (seq->mode != SeqTypes::None)
                seq->rotate(a[ia], c[ic], rot_bra[ibra], !trans, rot_ket[iket],
                            trans, scale);
            else
                MatrixFunctions::rotate(a[ia], c[ic], rot_bra[ibra], !trans,
                                        rot_ket[iket], trans, scale);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    void tensor_product_diagonal(uint8_t conj, const SparseMatrix<S> &a,
                                 const SparseMatrix<S> &b,
                                 const SparseMatrix<S> &c, S opdq,
                                 double scale = 1.0) const {
        scale = scale * a.factor * b.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, bdq = b.info->delta_quantum;
        assert(c.info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c.info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = lower_bound(cinfo->quanta + cinfo->n[conj],
                             cinfo->quanta + cinfo->n[conj + 1], abdq) -
                 cinfo->quanta;
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->tensor_product_diagonal(a[ia], b[ib], c[ic],
                                             scale * factor);
            else
                MatrixFunctions::tensor_product_diagonal(a[ia], b[ib], c[ic],
                                                         scale * factor);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    void tensor_product_multiply(uint8_t conj, const SparseMatrix<S> &a,
                                 const SparseMatrix<S> &b,
                                 const SparseMatrix<S> &c,
                                 const SparseMatrix<S> &v, S opdq,
                                 double scale = 1.0) const {
        scale = scale * a.factor * b.factor * c.factor;
        assert(v.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, bdq = b.info->delta_quantum;
        assert(c.info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c.info->cinfo;
        S abdq = opdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = lower_bound(cinfo->quanta + cinfo->n[conj],
                             cinfo->quanta + cinfo->n[conj + 1], abdq) -
                 cinfo->quanta;
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il],
                iv = cinfo->stride[il];
            if (seq->mode == SeqTypes::Simple && il != ixa &&
                iv <= cinfo->stride[il - 1])
                seq->simple_perform();
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->rotate(c[ic], v[iv], a[ia], conj & 1, b[ib], !(conj & 2),
                            scale * factor);
            else {
                seq->cumulative_nflop += (size_t)c[ic].m * c[ic].n *
                                             ((conj & 2) ? b[ib].n : b[ib].n) +
                                         (size_t)a[ia].m * a[ia].n *
                                             ((conj & 2) ? b[ib].n : b[ib].n);
                MatrixFunctions::rotate(c[ic], v[iv], a[ia], conj & 1, b[ib],
                                        !(conj & 2), scale * factor);
            }
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    void tensor_product(uint8_t conj, const SparseMatrix<S> &a,
                        const SparseMatrix<S> &b, SparseMatrix<S> &c,
                        double scale = 1.0) const {
        scale = scale * a.factor * b.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        S adq = a.info->delta_quantum, bdq = b.info->delta_quantum,
          cdq = c.info->delta_quantum;
        assert(c.info->cinfo != nullptr);
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
            c.info->cinfo;
        S abdq = cdq.combine((conj & 1) ? -adq : adq, (conj & 2) ? bdq : -bdq);
        int ik = lower_bound(cinfo->quanta + cinfo->n[conj],
                             cinfo->quanta + cinfo->n[conj + 1], abdq) -
                 cinfo->quanta;
        assert(ik < cinfo->n[conj + 1]);
        int ixa = cinfo->idx[ik];
        int ixb = ik == cinfo->n[4] - 1 ? cinfo->nc : cinfo->idx[ik + 1];
        for (int il = ixa; il < ixb; il++) {
            int ia = cinfo->ia[il], ib = cinfo->ib[il], ic = cinfo->ic[il];
            uint32_t stride = cinfo->stride[il];
            double factor = cinfo->factor[il];
            if (seq->mode != SeqTypes::None)
                seq->tensor_product(a[ia], conj & 1, b[ib], (conj & 2) >> 1,
                                    c[ic], scale * factor, stride);
            else
                MatrixFunctions::tensor_product(a[ia], conj & 1, b[ib],
                                                (conj & 2) >> 1, c[ic],
                                                scale * factor, stride);
        }
        if (seq->mode == SeqTypes::Simple)
            seq->simple_perform();
    }
    // c = a * b * scale
    void product(const SparseMatrix<S> &a, const SparseMatrix<S> &b,
                 const SparseMatrix<S> &c, double scale = 1.0) const {
        scale = scale * a.factor * b.factor;
        assert(c.factor == 1.0);
        if (abs(scale) < TINY)
            return;
        int adq = a.info->delta_quantum.multiplicity() - 1,
            bdq = b.info->delta_quantum.multiplicity() - 1,
            cdq = c.info->delta_quantum.multiplicity() - 1;
        for (int ic = 0; ic < c.info->n; ic++) {
            S cq = c.info->quanta[ic].get_bra(c.info->delta_quantum);
            S cqprime = c.info->quanta[ic].get_ket();
            S aps = cq - a.info->delta_quantum;
            for (int k = 0; k < aps.count(); k++) {
                S aqprime = aps[k];
                int ia = a.info->find_state(
                    a.info->delta_quantum.combine(cq, aps[k]));
                if (ia != -1) {
                    S bl = b.info->delta_quantum.combine(aqprime, cqprime);
                    if (bl != S(0xFFFFFFFFU)) {
                        int ib = b.info->find_state(bl);
                        if (ib != -1) {
                            int aqpj = aqprime.multiplicity() - 1,
                                cqj = cq.multiplicity() - 1,
                                cqpj = cqprime.multiplicity() - 1;
                            double factor =
                                cg->racah(cqpj, bdq, cqj, adq, aqpj, cdq);
                            factor *= sqrt((cdq + 1) * (aqpj + 1)) *
                                      (((adq + bdq - cdq) & 2) ? -1 : 1);
                            MatrixFunctions::multiply(a[ia], false, b[ib],
                                                      false, c[ic],
                                                      scale * factor, 1.0);
                        }
                    }
                }
            }
        }
    }
    static void trans_product(const SparseMatrix<S> &a,
                              const SparseMatrix<S> &b, bool trace_right,
                              double noise = 0.0) {
        double scale = a.factor * a.factor;
        assert(b.factor == 1.0);
        if (abs(scale) < TINY && noise == 0.0)
            return;
        SparseMatrix<S> tmp;
        if (noise != 0.0) {
            tmp.allocate(a.info);
            tmp.randomize(-0.5, 0.5);
        }
        if (trace_right)
            for (int ia = 0; ia < a.info->n; ia++) {
                S qb = a.info->quanta[ia].get_bra(a.info->delta_quantum);
                int ib = b.info->find_state(qb);
                MatrixFunctions::multiply(a[ia], false, a[ia], true, b[ib],
                                          scale, 1.0);
                if (noise != 0.0)
                    MatrixFunctions::multiply(tmp[ia], false, tmp[ia], true,
                                              b[ib], noise * noise, 1.0);
            }
        else
            for (int ia = 0; ia < a.info->n; ia++) {
                S qb = -a.info->quanta[ia].get_ket();
                int ib = b.info->find_state(qb);
                MatrixFunctions::multiply(a[ia], true, a[ia], false, b[ib],
                                          scale, 1.0);
                if (noise != 0.0)
                    MatrixFunctions::multiply(tmp[ia], true, tmp[ia], false,
                                              b[ib], noise * noise, 1.0);
            }
        if (noise != 0.0)
            tmp.deallocate();
    }
};

template <typename S> struct OperatorTensor {
    shared_ptr<Symbolic<S>> lmat, rmat;
    map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
        ops;
    OperatorTensor() : lmat(nullptr), rmat(nullptr) {}
    void reallocate(bool clean) {
        for (auto &p : ops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
    }
    void deallocate() {
        for (auto it = ops.crbegin(); it != ops.crend(); it++)
            it->second->deallocate();
    }
    shared_ptr<OperatorTensor> deep_copy() const {
        shared_ptr<OperatorTensor> r = make_shared<OperatorTensor>();
        r->lmat = lmat, r->rmat = rmat;
        for (auto &p : ops) {
            shared_ptr<SparseMatrix<S>> mat = make_shared<SparseMatrix<S>>();
            mat->allocate(p.second->info);
            mat->copy_data_from(*p.second);
            mat->factor = p.second->factor;
            r->ops[p.first] = mat;
        }
        return r;
    }
};

template <typename S> struct DelayedOperatorTensor {
    vector<shared_ptr<OpExpr<S>>> ops;
    shared_ptr<Symbolic<S>> mat;
    map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
        lops, rops;
    DelayedOperatorTensor() {}
    void reallocate(bool clean) {
        for (auto &p : lops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
        for (auto &p : rops)
            p.second->reallocate(clean ? 0 : p.second->total_memory);
    }
    void deallocate() {
        for (auto it = rops.crbegin(); it != rops.crend(); it++)
            it->second->deallocate();
        for (auto it = lops.crbegin(); it != lops.crend(); it++)
            it->second->deallocate();
    }
};

template <typename S> struct TensorFunctions {
    shared_ptr<OperatorFunctions<S>> opf;
    TensorFunctions(const shared_ptr<OperatorFunctions<S>> &opf) : opf(opf) {}
    static void left_assign(const shared_ptr<OperatorTensor<S>> &a,
                            shared_ptr<OperatorTensor<S>> &c) {
        assert(a->lmat != nullptr);
        assert(a->lmat->get_type() == SymTypes::RVec);
        assert(c->lmat != nullptr);
        assert(c->lmat->get_type() == SymTypes::RVec);
        assert(a->lmat->data.size() == c->lmat->data.size());
        for (size_t i = 0; i < a->lmat->data.size(); i++) {
            if (a->lmat->data[i]->get_type() == OpTypes::Zero)
                c->lmat->data[i] = a->lmat->data[i];
            else {
                assert(a->lmat->data[i] == c->lmat->data[i]);
                auto pa = abs_value(a->lmat->data[i]),
                     pc = abs_value(c->lmat->data[i]);
                c->ops[pc]->copy_data_from(*a->ops[pa]);
                c->ops[pc]->factor = a->ops[pa]->factor;
            }
        }
    }
    static void right_assign(const shared_ptr<OperatorTensor<S>> &a,
                             shared_ptr<OperatorTensor<S>> &c) {
        assert(a->rmat != nullptr);
        assert(a->rmat->get_type() == SymTypes::CVec);
        assert(c->rmat != nullptr);
        assert(c->rmat->get_type() == SymTypes::CVec);
        assert(a->rmat->data.size() == c->rmat->data.size());
        for (size_t i = 0; i < a->rmat->data.size(); i++) {
            if (a->rmat->data[i]->get_type() == OpTypes::Zero)
                c->rmat->data[i] = a->rmat->data[i];
            else {
                assert(a->rmat->data[i] == c->rmat->data[i]);
                auto pa = abs_value(a->rmat->data[i]),
                     pc = abs_value(c->rmat->data[i]);
                c->ops[pc]->copy_data_from(*a->ops[pa]);
                c->ops[pc]->factor = a->ops[pa]->factor;
            }
        }
    }
    void tensor_product_multiply(
        const shared_ptr<OpExpr<S>> &expr,
        const map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                  op_expr_less<S>> &lop,
        const map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                  op_expr_less<S>> &rop,
        const shared_ptr<SparseMatrix<S>> &cmat,
        const shared_ptr<SparseMatrix<S>> &vmat, S opdq) const {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpString<S>> op =
                dynamic_pointer_cast<OpString<S>>(expr);
            assert(op->b != nullptr);
            assert(!(lop.count(op->a) == 0 || rop.count(op->b) == 0));
            shared_ptr<SparseMatrix<S>> lmat = lop.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = rop.at(op->b);
            opf->tensor_product_multiply(op->conj, *lmat, *rmat, *cmat, *vmat,
                                         opdq, op->factor);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product_multiply(x, lop, rop, cmat, vmat, opdq);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    void tensor_product_diagonal(
        const shared_ptr<OpExpr<S>> &expr,
        const map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                  op_expr_less<S>> &lop,
        const map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                  op_expr_less<S>> &rop,
        shared_ptr<SparseMatrix<S>> &mat, S opdq) const {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpString<S>> op =
                dynamic_pointer_cast<OpString<S>>(expr);
            assert(op->b != nullptr);
            assert(!(lop.count(op->a) == 0 || rop.count(op->b) == 0));
            shared_ptr<SparseMatrix<S>> lmat = lop.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = rop.at(op->b);
            opf->tensor_product_diagonal(op->conj, *lmat, *rmat, *mat, opdq,
                                         op->factor);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product_diagonal(x, lop, rop, mat, opdq);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    void
    tensor_product(const shared_ptr<OpExpr<S>> &expr,
                   const map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                             op_expr_less<S>> &lop,
                   const map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                             op_expr_less<S>> &rop,
                   shared_ptr<SparseMatrix<S>> &mat) const {
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpString<S>> op =
                dynamic_pointer_cast<OpString<S>>(expr);
            assert(op->b != nullptr);
            assert(!(lop.count(op->a) == 0 || rop.count(op->b) == 0));
            shared_ptr<SparseMatrix<S>> lmat = lop.at(op->a);
            shared_ptr<SparseMatrix<S>> rmat = rop.at(op->b);
            opf->tensor_product(op->conj, *lmat, *rmat, *mat, op->factor);
        } break;
        case OpTypes::SumProd: {
            shared_ptr<OpSumProd<S>> op =
                dynamic_pointer_cast<OpSumProd<S>>(expr);
            assert((op->a == nullptr) ^ (op->b == nullptr));
            assert(op->ops.size() != 0);
            shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
            if (op->b == nullptr) {
                shared_ptr<OpExpr<S>> opb =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(op->a) != 0 && rop.count(opb) != 0);
                tmp->allocate(rop.at(opb)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    opf->iadd(
                        *tmp,
                        *rop.at(abs_value((shared_ptr<OpExpr<S>>)op->ops[i])),
                        op->factor * op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode == SeqTypes::Simple)
                        opf->seq->simple_perform();
                }
            } else {
                shared_ptr<OpExpr<S>> opa =
                    abs_value((shared_ptr<OpExpr<S>>)op->ops[0]);
                assert(lop.count(opa) != 0 && rop.count(op->b) != 0);
                tmp->allocate(lop.at(opa)->info);
                for (size_t i = 0; i < op->ops.size(); i++) {
                    opf->iadd(
                        *tmp,
                        *lop.at(abs_value((shared_ptr<OpExpr<S>>)op->ops[i])),
                        op->factor * op->ops[i]->factor, op->conjs[i]);
                    if (opf->seq->mode == SeqTypes::Simple)
                        opf->seq->simple_perform();
                }
            }
            if (op->b == nullptr)
                opf->tensor_product(op->conj, *lop.at(op->a), *tmp, *mat, 1.0);
            else
                opf->tensor_product(op->conj, *tmp, *rop.at(op->b), *mat, 1.0);
            tmp->deallocate();
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> op = dynamic_pointer_cast<OpSum<S>>(expr);
            for (auto &x : op->strings)
                tensor_product(x, lop, rop, mat);
        } break;
        case OpTypes::Zero:
            break;
        default:
            assert(false);
            break;
        }
    }
    void left_rotate(const shared_ptr<OperatorTensor<S>> &a,
                     const shared_ptr<SparseMatrix<S>> &mpst_bra,
                     const shared_ptr<SparseMatrix<S>> &mpst_ket,
                     shared_ptr<OperatorTensor<S>> &c) const {
        for (size_t i = 0; i < a->lmat->data.size(); i++)
            if (a->lmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->lmat->data[i]);
                opf->tensor_rotate(*a->ops.at(pa), *c->ops.at(pa), *mpst_bra,
                                   *mpst_ket, false);
            }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    void right_rotate(const shared_ptr<OperatorTensor<S>> &a,
                      const shared_ptr<SparseMatrix<S>> &mpst_bra,
                      const shared_ptr<SparseMatrix<S>> &mpst_ket,
                      shared_ptr<OperatorTensor<S>> &c) const {
        for (size_t i = 0; i < a->rmat->data.size(); i++)
            if (a->rmat->data[i]->get_type() != OpTypes::Zero) {
                auto pa = abs_value(a->rmat->data[i]);
                opf->tensor_rotate(*a->ops.at(pa), *c->ops.at(pa), *mpst_bra,
                                   *mpst_ket, true);
            }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    void numerical_transform(const shared_ptr<OperatorTensor<S>> &a,
                             const shared_ptr<Symbolic<S>> &names,
                             const shared_ptr<Symbolic<S>> &exprs) const {
        assert(names->data.size() == exprs->data.size());
        assert((a->lmat == nullptr) ^ (a->rmat == nullptr));
        if (a->lmat == nullptr)
            a->rmat = names;
        else
            a->lmat = names;
        for (size_t i = 0; i < a->ops.size(); i++) {
            bool found = false;
            for (size_t k = 0; k < names->data.size(); k++) {
                if (exprs->data[k]->get_type() == OpTypes::Zero)
                    continue;
                shared_ptr<OpExpr<S>> nop = abs_value(names->data[k]);
                shared_ptr<OpExpr<S>> expr =
                    exprs->data[k] *
                    (1 / dynamic_pointer_cast<OpElement<S>>(names->data[k])
                             ->factor);
                assert(a->ops.count(nop) != 0);
                switch (expr->get_type()) {
                case OpTypes::Sum: {
                    shared_ptr<OpSum<S>> op =
                        dynamic_pointer_cast<OpSum<S>>(expr);
                    found |= i < op->strings.size();
                    if (i < op->strings.size()) {
                        shared_ptr<OpElement<S>> nexpr =
                            op->strings[i]->get_op();
                        assert(a->ops.count(nexpr) != 0);
                        opf->iadd(*a->ops.at(nop), *a->ops.at(nexpr),
                                  nexpr->factor * op->strings[i]->factor,
                                  op->strings[i]->conj != 0);
                    }
                } break;
                case OpTypes::Zero:
                    break;
                default:
                    assert(false);
                    break;
                }
            }
            if (!found)
                break;
            else if (opf->seq->mode == SeqTypes::Simple)
                opf->seq->simple_perform();
        }
        if (opf->seq->mode == SeqTypes::Auto)
            opf->seq->auto_perform();
    }
    static shared_ptr<DelayedOperatorTensor<S>>
    delayed_contract(const shared_ptr<OperatorTensor<S>> &a,
                     const shared_ptr<OperatorTensor<S>> &b,
                     const shared_ptr<OpExpr<S>> &op) {
        shared_ptr<DelayedOperatorTensor<S>> dopt =
            make_shared<DelayedOperatorTensor<S>>();
        dopt->lops = a->ops;
        dopt->rops = b->ops;
        dopt->ops.push_back(op);
        assert(a->lmat->data.size() == b->rmat->data.size());
        shared_ptr<Symbolic<S>> exprs = a->lmat * b->rmat;
        assert(exprs->data.size() == 1);
        dopt->mat = exprs;
        return dopt;
    }
    static shared_ptr<DelayedOperatorTensor<S>>
    delayed_contract(const shared_ptr<OperatorTensor<S>> &a,
                     const shared_ptr<OperatorTensor<S>> &b,
                     const shared_ptr<Symbolic<S>> &ops,
                     const shared_ptr<Symbolic<S>> &exprs) {
        shared_ptr<DelayedOperatorTensor<S>> dopt =
            make_shared<DelayedOperatorTensor<S>>();
        dopt->lops = a->ops;
        dopt->rops = b->ops;
        dopt->ops = ops->data;
        dopt->mat = exprs;
        return dopt;
    }
    void left_contract(const shared_ptr<OperatorTensor<S>> &a,
                       const shared_ptr<OperatorTensor<S>> &b,
                       shared_ptr<OperatorTensor<S>> &c,
                       const shared_ptr<Symbolic<S>> &cexprs = nullptr) const {
        if (a == nullptr)
            left_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? a->lmat * b->lmat : cexprs;
            assert(exprs->data.size() == c->lmat->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpElement<S>> cop =
                    dynamic_pointer_cast<OpElement<S>>(c->lmat->data[i]);
                shared_ptr<OpExpr<S>> op = abs_value(c->lmat->data[i]);
                shared_ptr<OpExpr<S>> expr = exprs->data[i] * (1 / cop->factor);
                tensor_product(expr, a->ops, b->ops, c->ops.at(op));
            }
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
    void right_contract(const shared_ptr<OperatorTensor<S>> &a,
                        const shared_ptr<OperatorTensor<S>> &b,
                        shared_ptr<OperatorTensor<S>> &c,
                        const shared_ptr<Symbolic<S>> &cexprs = nullptr) const {
        if (a == nullptr)
            right_assign(b, c);
        else {
            shared_ptr<Symbolic<S>> exprs =
                cexprs == nullptr ? b->rmat * a->rmat : cexprs;
            assert(exprs->data.size() == c->rmat->data.size());
            for (size_t i = 0; i < exprs->data.size(); i++) {
                shared_ptr<OpElement<S>> cop =
                    dynamic_pointer_cast<OpElement<S>>(c->rmat->data[i]);
                shared_ptr<OpExpr<S>> op = abs_value(c->rmat->data[i]);
                shared_ptr<OpExpr<S>> expr = exprs->data[i] * (1 / cop->factor);
                tensor_product(expr, b->ops, a->ops, c->ops.at(op));
            }
            if (opf->seq->mode == SeqTypes::Auto)
                opf->seq->auto_perform();
        }
    }
};

template <typename S> struct MPOSchemer {
    uint8_t left_trans_site, right_trans_site;
    shared_ptr<SymbolicRowVector<S>> left_new_operator_names;
    shared_ptr<SymbolicColumnVector<S>> right_new_operator_names;
    shared_ptr<SymbolicRowVector<S>> left_new_operator_exprs;
    shared_ptr<SymbolicColumnVector<S>> right_new_operator_exprs;
    MPOSchemer(uint8_t left_trans_site, uint8_t right_trans_site)
        : left_trans_site(left_trans_site), right_trans_site(right_trans_site) {
    }
    shared_ptr<MPOSchemer> copy() const {
        shared_ptr<MPOSchemer> r =
            make_shared<MPOSchemer>(left_trans_site, right_trans_site);
        r->left_new_operator_names = left_new_operator_names;
        r->right_new_operator_names = right_new_operator_names;
        r->left_new_operator_exprs = left_new_operator_exprs;
        r->right_new_operator_exprs = right_new_operator_exprs;
        return r;
    }
    string get_transform_formulas() const {
        stringstream ss;
        ss << "LEFT  TRANSFORM :: SITE = " << (int)left_trans_site << endl;
        for (int j = 0; j < left_new_operator_names->data.size(); j++) {
            if (left_new_operator_names->data[j]->get_type() != OpTypes::Zero)
                ss << "[" << setw(4) << j << "] " << setw(15)
                   << left_new_operator_names->data[j]
                   << " := " << left_new_operator_exprs->data[j] << endl;
            else
                ss << "[" << setw(4) << j << "] "
                   << left_new_operator_names->data[j] << endl;
        }
        ss << endl;
        ss << "RIGHT TRANSFORM :: SITE = " << (int)right_trans_site << endl;
        for (int j = 0; j < right_new_operator_names->data.size(); j++) {
            if (right_new_operator_names->data[j]->get_type() != OpTypes::Zero)
                ss << "[" << setw(4) << j << "] " << setw(15)
                   << right_new_operator_names->data[j]
                   << " := " << right_new_operator_exprs->data[j] << endl;
            else
                ss << "[" << setw(4) << j << "] "
                   << right_new_operator_names->data[j] << endl;
        }
        ss << endl;
        return ss.str();
    }
};

enum struct AncillaTypes : uint8_t { None, Ancilla };

template <typename S> struct MPO {
    vector<shared_ptr<OperatorTensor<S>>> tensors;
    vector<shared_ptr<Symbolic<S>>> left_operator_names;
    vector<shared_ptr<Symbolic<S>>> right_operator_names;
    vector<shared_ptr<Symbolic<S>>> middle_operator_names;
    vector<shared_ptr<Symbolic<S>>> left_operator_exprs;
    vector<shared_ptr<Symbolic<S>>> right_operator_exprs;
    vector<shared_ptr<Symbolic<S>>> middle_operator_exprs;
    shared_ptr<OpElement<S>> op;
    shared_ptr<MPOSchemer<S>> schemer;
    int n_sites;
    double const_e;
    shared_ptr<TensorFunctions<S>> tf;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> *site_op_infos;
    MPO(int n_sites)
        : n_sites(n_sites), const_e(0.0), op(nullptr), schemer(nullptr),
          tf(nullptr), site_op_infos(nullptr) {}
    virtual AncillaTypes get_ancilla_type() const { return AncillaTypes::None; }
    virtual void deallocate() {}
    string get_blocking_formulas() const {
        stringstream ss;
        for (int i = 0; i < n_sites; i++) {
            ss << "LEFT BLOCKING :: SITE = " << i << endl;
            for (int j = 0; j < left_operator_names[i]->data.size(); j++) {
                if (left_operator_exprs.size() != 0)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << left_operator_names[i]->data[j]
                       << " := " << left_operator_exprs[i]->data[j] << endl;
                else
                    ss << "[" << setw(4) << j << "] "
                       << left_operator_names[i]->data[j] << endl;
            }
            ss << endl;
        }
        for (int i = n_sites - 1; i >= 0; i--) {
            ss << "RIGHT BLOCKING :: SITE = " << i << endl;
            for (int j = 0; j < right_operator_names[i]->data.size(); j++) {
                if (right_operator_exprs.size() != 0)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << right_operator_names[i]->data[j]
                       << " := " << right_operator_exprs[i]->data[j] << endl;
                else
                    ss << "[" << setw(4) << j << "] "
                       << right_operator_names[i]->data[j] << endl;
            }
            ss << endl;
        }
        if (middle_operator_names.size() != 0) {
            for (int i = 0; i < n_sites - 1; i++) {
                ss << "HAMIL PARITITION :: SITE = " << i << endl;
                for (int j = 0; j < middle_operator_names[i]->data.size(); j++)
                    ss << "[" << setw(4) << j << "] " << setw(15)
                       << middle_operator_names[i]->data[j]
                       << " := " << middle_operator_exprs[i]->data[j] << endl;
                ss << endl;
            }
        }
        if (schemer != nullptr)
            ss << schemer->get_transform_formulas() << endl;
        return ss.str();
    }
};

template <typename S> struct Rule {
    Rule() {}
    virtual shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const {
        return nullptr;
    }
};

template <typename S> struct SimplifiedMPO : MPO<S> {
    shared_ptr<MPO<S>> prim_mpo;
    shared_ptr<Rule<S>> rule;
    bool collect_terms;
    SimplifiedMPO(const shared_ptr<MPO<S>> &mpo,
                  const shared_ptr<Rule<S>> &rule, bool collect_terms = true)
        : prim_mpo(mpo), rule(rule), MPO<S>(mpo->n_sites),
          collect_terms(collect_terms) {
        static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        MPO<S>::const_e = mpo->const_e;
        MPO<S>::tensors = mpo->tensors;
        MPO<S>::op = mpo->op;
        MPO<S>::schemer = mpo->schemer;
        MPO<S>::tf = mpo->tf;
        MPO<S>::site_op_infos = mpo->site_op_infos;
        MPO<S>::left_operator_names = mpo->left_operator_names;
        MPO<S>::right_operator_names = mpo->right_operator_names;
        MPO<S>::left_operator_exprs.resize(MPO<S>::n_sites);
        MPO<S>::right_operator_exprs.resize(MPO<S>::n_sites);
        if (MPO<S>::schemer != nullptr) {
            int i = MPO<S>::schemer->left_trans_site;
            for (size_t j = 0;
                 j < MPO<S>::schemer->left_new_operator_names->data.size();
                 j++) {
                if (j < MPO<S>::left_operator_names[i]->data.size() &&
                    MPO<S>::left_operator_names[i]->data[j] ==
                        MPO<S>::schemer->left_new_operator_names->data[j])
                    continue;
                else if (MPO<S>::schemer->left_new_operator_exprs->data[j]
                             ->get_type() == OpTypes::Zero)
                    MPO<S>::schemer->left_new_operator_names->data[j] =
                        MPO<S>::schemer->left_new_operator_exprs->data[j];
            }
            i = MPO<S>::schemer->right_trans_site;
            for (size_t j = 0;
                 j < MPO<S>::schemer->right_new_operator_names->data.size();
                 j++) {
                if (j < MPO<S>::right_operator_names[i]->data.size() &&
                    MPO<S>::right_operator_names[i]->data[j] ==
                        MPO<S>::schemer->right_new_operator_names->data[j])
                    continue;
                else if (MPO<S>::schemer->right_new_operator_exprs->data[j]
                             ->get_type() == OpTypes::Zero)
                    MPO<S>::schemer->right_new_operator_names->data[j] =
                        MPO<S>::schemer->right_new_operator_exprs->data[j];
            }
        }
        for (int i = 0; i < MPO<S>::n_sites; i++) {
            if (i == 0)
                MPO<S>::left_operator_exprs[i] = MPO<S>::tensors[i]->lmat;
            else if (MPO<S>::schemer == nullptr ||
                     i - 1 != MPO<S>::schemer->left_trans_site)
                MPO<S>::left_operator_exprs[i] =
                    MPO<S>::left_operator_names[i - 1] *
                    MPO<S>::tensors[i]->lmat;
            else
                MPO<S>::left_operator_exprs[i] =
                    (shared_ptr<Symbolic<S>>)
                        MPO<S>::schemer->left_new_operator_names *
                    MPO<S>::tensors[i]->lmat;
            if (MPO<S>::schemer != nullptr &&
                i == MPO<S>::schemer->left_trans_site) {
                for (size_t j = 0;
                     j < MPO<S>::left_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::left_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero) {
                        if (j < MPO<S>::schemer->left_new_operator_names->data
                                    .size() &&
                            MPO<S>::left_operator_names[i]->data[j] ==
                                MPO<S>::schemer->left_new_operator_names
                                    ->data[j])
                            MPO<S>::schemer->left_new_operator_names->data[j] =
                                MPO<S>::left_operator_exprs[i]->data[j];
                        MPO<S>::left_operator_names[i]->data[j] =
                            MPO<S>::left_operator_exprs[i]->data[j];
                    }
            } else {
                for (size_t j = 0;
                     j < MPO<S>::left_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::left_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero)
                        MPO<S>::left_operator_names[i]->data[j] =
                            MPO<S>::left_operator_exprs[i]->data[j];
            }
        }
        MPO<S>::right_operator_exprs[MPO<S>::n_sites - 1] =
            MPO<S>::tensors[MPO<S>::n_sites - 1]->rmat;
        for (int i = MPO<S>::n_sites - 1; i >= 0; i--) {
            if (i == MPO<S>::n_sites - 1)
                MPO<S>::right_operator_exprs[i] = MPO<S>::tensors[i]->rmat;
            else if (MPO<S>::schemer == nullptr ||
                     i + 1 != MPO<S>::schemer->right_trans_site)
                MPO<S>::right_operator_exprs[i] =
                    MPO<S>::tensors[i]->rmat *
                    MPO<S>::right_operator_names[i + 1];
            else
                MPO<S>::right_operator_exprs[i] =
                    MPO<S>::tensors[i]->rmat *
                    (shared_ptr<Symbolic<S>>)
                        MPO<S>::schemer->right_new_operator_names;
            if (MPO<S>::schemer != nullptr &&
                i == MPO<S>::schemer->right_trans_site) {
                for (size_t j = 0;
                     j < MPO<S>::right_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::right_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero) {
                        if (j < MPO<S>::schemer->right_new_operator_names->data
                                    .size() &&
                            MPO<S>::right_operator_names[i]->data[j] ==
                                MPO<S>::schemer->right_new_operator_names
                                    ->data[j])
                            MPO<S>::schemer->right_new_operator_names->data[j] =
                                MPO<S>::right_operator_exprs[i]->data[j];
                        MPO<S>::right_operator_names[i]->data[j] =
                            MPO<S>::right_operator_exprs[i]->data[j];
                    }
            } else {
                for (size_t j = 0;
                     j < MPO<S>::right_operator_exprs[i]->data.size(); j++)
                    if (MPO<S>::right_operator_exprs[i]->data[j]->get_type() ==
                        OpTypes::Zero)
                        MPO<S>::right_operator_names[i]->data[j] =
                            MPO<S>::right_operator_exprs[i]->data[j];
            }
        }
        if (mpo->middle_operator_exprs.size() != 0) {
            MPO<S>::middle_operator_names = mpo->middle_operator_names;
            MPO<S>::middle_operator_exprs = mpo->middle_operator_exprs;
        } else {
            vector<uint8_t> px[2];
            for (int i = MPO<S>::n_sites - 1; i >= 0; i--) {
                if (i != MPO<S>::n_sites - 1) {
                    if (MPO<S>::schemer == nullptr ||
                        i != MPO<S>::schemer->left_trans_site) {
                        for (size_t j = 0;
                             j < MPO<S>::left_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::right_operator_names[i + 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::left_operator_names[i]->data[j] =
                                    MPO<S>::right_operator_names[i + 1]
                                        ->data[j];
                            else if (MPO<S>::left_operator_names[i]
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else if (MPO<S>::schemer->right_trans_site -
                                   MPO<S>::schemer->left_trans_site >
                               1) {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->left_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::schemer->left_new_operator_names
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->left_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::right_operator_names[i + 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::schemer->left_new_operator_names
                                    ->data[j] =
                                    MPO<S>::right_operator_names[i + 1]
                                        ->data[j];
                            else if (MPO<S>::schemer->left_new_operator_names
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    }
                    if (MPO<S>::schemer != nullptr &&
                        i == MPO<S>::schemer->left_trans_site) {
                        px[!(i & 1)].resize(px[i & 1].size());
                        memcpy(&px[!(i & 1)][0], &px[i & 1][0],
                               sizeof(uint8_t) * px[!(i & 1)].size());
                        px[i & 1].resize(
                            MPO<S>::left_operator_names[i]->data.size());
                        memset(&px[i & 1][0], 0,
                               sizeof(uint8_t) * px[i & 1].size());
                        map<shared_ptr<OpExpr<S>>, int, op_expr_less<S>> mp;
                        for (size_t j = 0;
                             j < MPO<S>::left_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::left_operator_names[i]
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                mp[abs_value(
                                    MPO<S>::left_operator_names[i]->data[j])] =
                                    j;
                        shared_ptr<SymbolicRowVector<S>> &exprs =
                            MPO<S>::schemer->left_new_operator_exprs;
                        for (size_t j = 0; j < exprs->data.size(); j++) {
                            if (px[!(i & 1)][j] &&
                                j < MPO<S>::left_operator_names[i]
                                        ->data.size() &&
                                MPO<S>::left_operator_names[i]->data[j] ==
                                    MPO<S>::schemer->left_new_operator_names
                                        ->data[j])
                                px[i & 1][j] = 1;
                            else if (px[!(i & 1)][j] &&
                                     exprs->data[j]->get_type() !=
                                         OpTypes::Zero) {
                                shared_ptr<OpSum<S>> op =
                                    dynamic_pointer_cast<OpSum<S>>(
                                        exprs->data[j]);
                                for (size_t k = 0; k < op->strings.size();
                                     k++) {
                                    shared_ptr<OpExpr<S>> expr = abs_value(
                                        (shared_ptr<OpExpr<S>>)op->strings[k]
                                            ->get_op());
                                    if (mp.count(expr) == 0)
                                        op->strings[k]->factor = 0;
                                    else
                                        px[i & 1][mp[expr]] = 1;
                                }
                            }
                        }
                        for (size_t j = 0;
                             j < MPO<S>::left_operator_names[i]->data.size();
                             j++)
                            if (!px[i & 1][j])
                                MPO<S>::left_operator_names[i]->data[j] = zero;
                    }
                }
                if (i != 0) {
                    if (MPO<S>::schemer == nullptr ||
                        i - 1 != MPO<S>::schemer->left_trans_site)
                        px[!(i & 1)].resize(
                            MPO<S>::left_operator_names[i - 1]->data.size());
                    else
                        px[!(i & 1)].resize(
                            MPO<S>::schemer->left_new_operator_names->data
                                .size());
                    memset(&px[!(i & 1)][0], 0,
                           sizeof(uint8_t) * px[!(i & 1)].size());
                    if (MPO<S>::tensors[i]->lmat->get_type() == SymTypes::Mat) {
                        assert(px[i & 1].size() != 0);
                        shared_ptr<SymbolicMatrix<S>> mat =
                            dynamic_pointer_cast<SymbolicMatrix<S>>(
                                MPO<S>::tensors[i]->lmat);
                        for (size_t j = 0; j < mat->data.size(); j++)
                            if (px[i & 1][mat->indices[j].second] &&
                                mat->data[j]->get_type() != OpTypes::Zero)
                                px[!(i & 1)][mat->indices[j].first] = 1;
                    }
                }
            }
            for (int i = 0; i < MPO<S>::n_sites; i++) {
                if (i != 0) {
                    if (MPO<S>::schemer == nullptr ||
                        i != MPO<S>::schemer->right_trans_site) {
                        for (size_t j = 0;
                             j < MPO<S>::right_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::left_operator_names[i - 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::right_operator_names[i]->data[j] =
                                    MPO<S>::left_operator_names[i - 1]->data[j];
                            else if (MPO<S>::right_operator_names[i]
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else if (MPO<S>::schemer->right_trans_site -
                                   MPO<S>::schemer->left_trans_site >
                               1) {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->right_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::schemer->right_new_operator_names
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    } else {
                        for (size_t j = 0;
                             j < MPO<S>::schemer->right_new_operator_names->data
                                     .size();
                             j++)
                            if (MPO<S>::left_operator_names[i - 1]
                                        ->data[j]
                                        ->get_type() == OpTypes::Zero &&
                                !px[i & 1][j])
                                MPO<S>::schemer->right_new_operator_names
                                    ->data[j] =
                                    MPO<S>::left_operator_names[i - 1]->data[j];
                            else if (MPO<S>::schemer->right_new_operator_names
                                         ->data[j]
                                         ->get_type() != OpTypes::Zero)
                                px[i & 1][j] = 1;
                    }
                    if (MPO<S>::schemer != nullptr &&
                        i == MPO<S>::schemer->right_trans_site) {
                        px[!(i & 1)].resize(px[i & 1].size());
                        memcpy(&px[!(i & 1)][0], &px[i & 1][0],
                               sizeof(uint8_t) * px[!(i & 1)].size());
                        px[i & 1].resize(
                            MPO<S>::right_operator_names[i]->data.size());
                        memset(&px[i & 1][0], 0,
                               sizeof(uint8_t) * px[i & 1].size());
                        map<shared_ptr<OpExpr<S>>, int, op_expr_less<S>> mp;
                        for (size_t j = 0;
                             j < MPO<S>::right_operator_names[i]->data.size();
                             j++)
                            if (MPO<S>::right_operator_names[i]
                                    ->data[j]
                                    ->get_type() != OpTypes::Zero)
                                mp[abs_value(
                                    MPO<S>::right_operator_names[i]->data[j])] =
                                    j;
                        shared_ptr<SymbolicColumnVector<S>> &exprs =
                            MPO<S>::schemer->right_new_operator_exprs;
                        for (size_t j = 0; j < exprs->data.size(); j++) {
                            if (px[!(i & 1)][j] &&
                                j < MPO<S>::right_operator_names[i]
                                        ->data.size() &&
                                MPO<S>::right_operator_names[i]->data[j] ==
                                    MPO<S>::schemer->right_new_operator_names
                                        ->data[j])
                                px[i & 1][j] = 1;
                            else if (px[!(i & 1)][j] &&
                                     exprs->data[j]->get_type() !=
                                         OpTypes::Zero) {
                                shared_ptr<OpSum<S>> op =
                                    dynamic_pointer_cast<OpSum<S>>(
                                        exprs->data[j]);
                                for (size_t k = 0; k < op->strings.size();
                                     k++) {
                                    shared_ptr<OpExpr<S>> expr = abs_value(
                                        (shared_ptr<OpExpr<S>>)op->strings[k]
                                            ->get_op());
                                    if (mp.count(expr) == 0)
                                        op->strings[k]->factor = 0;
                                    else
                                        px[i & 1][mp[expr]] = 1;
                                }
                            }
                        }
                        for (size_t j = 0;
                             j < MPO<S>::right_operator_names[i]->data.size();
                             j++)
                            if (!px[i & 1][j])
                                MPO<S>::right_operator_names[i]->data[j] = zero;
                    }
                }
                if (i != MPO<S>::n_sites - 1) {
                    if (MPO<S>::schemer == nullptr ||
                        i + 1 != MPO<S>::schemer->right_trans_site)
                        px[!(i & 1)].resize(
                            MPO<S>::right_operator_names[i + 1]->data.size());
                    else
                        px[!(i & 1)].resize(
                            MPO<S>::schemer->right_new_operator_names->data
                                .size());
                    memset(&px[!(i & 1)][0], 0,
                           sizeof(uint8_t) * px[!(i & 1)].size());
                    if (MPO<S>::tensors[i]->rmat->get_type() == SymTypes::Mat) {
                        shared_ptr<SymbolicMatrix<S>> mat =
                            dynamic_pointer_cast<SymbolicMatrix<S>>(
                                MPO<S>::tensors[i]->rmat);
                        for (size_t j = 0; j < mat->data.size(); j++)
                            if (px[i & 1][mat->indices[j].first] &&
                                mat->data[j]->get_type() != OpTypes::Zero)
                                px[!(i & 1)][mat->indices[j].second] = 1;
                    }
                }
            }
            MPO<S>::middle_operator_names.resize(MPO<S>::n_sites - 1);
            MPO<S>::middle_operator_exprs.resize(MPO<S>::n_sites - 1);
            shared_ptr<SymbolicColumnVector<S>> mpo_op =
                make_shared<SymbolicColumnVector<S>>(1);
            (*mpo_op)[0] = mpo->op;
            for (int i = 0; i < MPO<S>::n_sites - 1; i++) {
                MPO<S>::middle_operator_names[i] = mpo_op;
                if (MPO<S>::schemer == nullptr ||
                    i != MPO<S>::schemer->left_trans_site ||
                    MPO<S>::schemer->right_trans_site -
                            MPO<S>::schemer->left_trans_site >
                        1)
                    MPO<S>::middle_operator_exprs[i] =
                        MPO<S>::left_operator_names[i] *
                        MPO<S>::right_operator_names[i + 1];
                else
                    MPO<S>::middle_operator_exprs[i] =
                        (shared_ptr<Symbolic<S>>)
                            MPO<S>::schemer->left_new_operator_names *
                        MPO<S>::right_operator_names[i + 1];
            }
        }
        simplify();
    }
    shared_ptr<OpExpr<S>> simplify_expr(const shared_ptr<OpExpr<S>> &expr,
                                        S op = S(0xFFFFFFFFU)) {
        static shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        switch (expr->get_type()) {
        case OpTypes::Prod: {
            shared_ptr<OpString<S>> op =
                dynamic_pointer_cast<OpString<S>>(expr);
            assert(op->b != nullptr);
            shared_ptr<OpElementRef<S>> opl = rule->operator()(op->a);
            shared_ptr<OpElementRef<S>> opr = rule->operator()(op->b);
            shared_ptr<OpElement<S>> a = opl == nullptr ? op->a : opl->op;
            shared_ptr<OpElement<S>> b = opr == nullptr ? op->b : opr->op;
            uint8_t conj = (opl != nullptr && opl->trans) |
                           ((opr != nullptr && opr->trans) << 1);
            double factor = (opl != nullptr ? opl->factor : 1.0) *
                            (opr != nullptr ? opr->factor : 1.0) * op->factor;
            return make_shared<OpString<S>>(a, b, factor, conj);
        } break;
        case OpTypes::Sum: {
            shared_ptr<OpSum<S>> ops = dynamic_pointer_cast<OpSum<S>>(expr);
            map<shared_ptr<OpExpr<S>>, vector<shared_ptr<OpString<S>>>,
                op_expr_less<S>>
                mp;
            for (auto &x : ops->strings) {
                if (x->factor == 0)
                    continue;
                shared_ptr<OpElementRef<S>> opl = rule->operator()(x->a);
                shared_ptr<OpElementRef<S>> opr =
                    x->b == nullptr ? nullptr : rule->operator()(x->b);
                shared_ptr<OpElement<S>> a = opl == nullptr ? x->a : opl->op;
                shared_ptr<OpElement<S>> b = opr == nullptr ? x->b : opr->op;
                uint8_t conj = (opl != nullptr && opl->trans) |
                               ((opr != nullptr && opr->trans) << 1);
                double factor = (opl != nullptr ? opl->factor : 1.0) *
                                (opr != nullptr ? opr->factor : 1.0) *
                                x->factor;
                if (!mp.count(a))
                    mp[a] = vector<shared_ptr<OpString<S>>>();
                vector<shared_ptr<OpString<S>>> &px = mp.at(a);
                int g = -1;
                for (size_t k = 0; k < px.size(); k++)
                    if (px[k]->b == b && px[k]->conj == conj) {
                        g = k;
                        break;
                    }
                if (g == -1)
                    px.push_back(make_shared<OpString<S>>(a, b, factor, conj));
                else {
                    px[g]->factor += factor;
                    if (abs(px[g]->factor) < TINY)
                        px.erase(px.begin() + g);
                }
            }
            vector<shared_ptr<OpString<S>>> terms;
            terms.reserve(mp.size());
            for (auto &r : mp)
                terms.insert(terms.end(), r.second.begin(), r.second.end());
            if (terms.size() == 0)
                return zero;
            else if (terms[0]->b == nullptr || terms.size() <= 2)
                return make_shared<OpSum<S>>(terms);
            else if (collect_terms && op != S(0xFFFFFFFFU)) {
                map<shared_ptr<OpExpr<S>>,
                    map<int, vector<shared_ptr<OpString<S>>>>, op_expr_less<S>>
                    mpa[2], mpb[2];
                for (auto &x : terms) {
                    assert(x->a != nullptr && x->b != nullptr);
                    if (x->conj & 1)
                        mpa[1][x->a][x->b->q_label.multiplicity()].push_back(x);
                    else
                        mpa[0][x->a][x->b->q_label.multiplicity()].push_back(x);
                    if (x->conj & 2)
                        mpb[1][x->b][x->a->q_label.multiplicity()].push_back(x);
                    else
                        mpb[0][x->b][x->a->q_label.multiplicity()].push_back(x);
                }
                terms.clear();
                if (mpa[0].size() + mpa[1].size() <=
                    mpb[0].size() + mpb[1].size()) {
                    for (int i = 0; i < 2; i++)
                        for (auto &r : mpa[i]) {
                            int pg = dynamic_pointer_cast<OpElement<S>>(r.first)
                                         ->q_label.pg() ^
                                     op.pg();
                            for (auto &rr : r.second) {
                                if (rr.second.size() == 1)
                                    terms.push_back(rr.second[0]);
                                else {
                                    vector<bool> conjs;
                                    vector<shared_ptr<OpElement<S>>> ops;
                                    conjs.reserve(rr.second.size());
                                    ops.reserve(rr.second.size());
                                    for (auto &s : rr.second) {
                                        if (s->b->q_label.pg() != pg)
                                            continue;
                                        bool cj = (s->conj & 2) != 0,
                                             found = false;
                                        OpElement<S> op = s->b->abs();
                                        for (size_t j = 0; j < ops.size(); j++)
                                            if (conjs[j] == cj &&
                                                op == ops[j]->abs()) {
                                                found = true;
                                                ops[j]->factor +=
                                                    s->b->factor * s->factor;
                                                break;
                                            }
                                        if (!found) {
                                            conjs.push_back((s->conj & 2) != 0);
                                            ops.push_back(dynamic_pointer_cast<
                                                          OpElement<S>>(
                                                (shared_ptr<OpExpr<S>>)s->b *
                                                s->factor));
                                        }
                                    }
                                    uint8_t cjx = i;
                                    if (conjs[0])
                                        conjs.flip(), cjx |= 1 << 1;
                                    if (ops.size() == 1)
                                        terms.push_back(
                                            make_shared<OpString<S>>(
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                ops[0], 1.0, cjx));
                                    else if (ops.size() != 0)
                                        terms.push_back(
                                            make_shared<OpSumProd<S>>(
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                ops, conjs, 1.0, cjx));
                                }
                            }
                        }
                } else {
                    for (int i = 0; i < 2; i++)
                        for (auto &r : mpb[i]) {
                            int pg = dynamic_pointer_cast<OpElement<S>>(r.first)
                                         ->q_label.pg() ^
                                     op.pg();
                            for (auto &rr : r.second) {
                                if (rr.second.size() == 1)
                                    terms.push_back(rr.second[0]);
                                else {
                                    vector<bool> conjs;
                                    vector<shared_ptr<OpElement<S>>> ops;
                                    conjs.reserve(rr.second.size());
                                    ops.reserve(rr.second.size());
                                    for (auto &s : rr.second) {
                                        if (s->a->q_label.pg() != pg)
                                            continue;
                                        bool cj = (s->conj & 1) != 0,
                                             found = false;
                                        OpElement<S> op = s->a->abs();
                                        for (size_t j = 0; j < ops.size(); j++)
                                            if (conjs[j] == cj &&
                                                op == ops[j]->abs()) {
                                                found = true;
                                                ops[j]->factor +=
                                                    s->a->factor * s->factor;
                                                break;
                                            }
                                        if (!found) {
                                            conjs.push_back((s->conj & 1) != 0);
                                            ops.push_back(dynamic_pointer_cast<
                                                          OpElement<S>>(
                                                (shared_ptr<OpExpr<S>>)s->a *
                                                s->factor));
                                        }
                                    }
                                    uint8_t cjx = i << 1;
                                    if (conjs[0])
                                        conjs.flip(), cjx |= 1;
                                    if (ops.size() == 1)
                                        terms.push_back(
                                            make_shared<OpString<S>>(
                                                ops[0],
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                1.0, cjx));
                                    else if (ops.size() != 0)
                                        terms.push_back(
                                            make_shared<OpSumProd<S>>(
                                                ops,
                                                dynamic_pointer_cast<
                                                    OpElement<S>>(r.first),
                                                conjs, 1.0, cjx));
                                }
                            }
                        }
                }
                return make_shared<OpSum<S>>(terms);
            } else
                return make_shared<OpSum<S>>(terms);
        } break;
        case OpTypes::Zero:
        case OpTypes::Elem:
            return expr;
        default:
            assert(false);
            break;
        }
        return expr;
    }
    void simplify_symbolic(const shared_ptr<Symbolic<S>> &name,
                           const shared_ptr<Symbolic<S>> &expr,
                           const shared_ptr<Symbolic<S>> &ref = nullptr) {
        assert(name->data.size() == expr->data.size());
        size_t k = 0;
        for (size_t j = 0; j < name->data.size(); j++) {
            if (name->data[j]->get_type() == OpTypes::Zero)
                continue;
            else if (expr->data[j]->get_type() == OpTypes::Zero &&
                     (ref == nullptr || j >= ref->data.size() ||
                      ref->data[j] != name->data[j]))
                continue;
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(name->data[j]);
            if (rule->operator()(op) != nullptr)
                continue;
            name->data[k] = abs_value(name->data[j]);
            expr->data[k] =
                simplify_expr(expr->data[j], op->q_label) * (1 / op->factor);
            k++;
        }
        name->data.resize(k);
        expr->data.resize(k);
        name->n = expr->n = (int)name->data.size();
    }
    void simplify() {
        if (MPO<S>::schemer != nullptr) {
            simplify_symbolic(
                MPO<S>::schemer->left_new_operator_names,
                MPO<S>::schemer->left_new_operator_exprs,
                MPO<S>::left_operator_names[MPO<S>::schemer->left_trans_site]);
            simplify_symbolic(
                MPO<S>::schemer->right_new_operator_names,
                MPO<S>::schemer->right_new_operator_exprs,
                MPO<S>::right_operator_names[MPO<S>::schemer
                                                 ->right_trans_site]);
        }
        for (int i = 0; i < MPO<S>::n_sites; i++)
            simplify_symbolic(MPO<S>::left_operator_names[i],
                              MPO<S>::left_operator_exprs[i]);
        for (int i = 0; i < MPO<S>::n_sites; i++)
            simplify_symbolic(MPO<S>::right_operator_names[i],
                              MPO<S>::right_operator_exprs[i]);
        for (int i = 0; i < MPO<S>::n_sites - 1; i++) {
            shared_ptr<Symbolic<S>> mexpr = MPO<S>::middle_operator_exprs[i];
            for (size_t j = 0; j < mexpr->data.size(); j++)
                mexpr->data[j] = simplify_expr(mexpr->data[j]);
        }
    }
    AncillaTypes get_ancilla_type() const override {
        return prim_mpo->get_ancilla_type();
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

template <typename S> struct MPSInfo {
    int n_sites;
    S vaccum;
    S target;
    vector<uint8_t> orbsym;
    uint8_t n_syms;
    uint16_t bond_dim;
    StateInfo<S> *basis, *left_dims_fci, *right_dims_fci;
    StateInfo<S> *left_dims, *right_dims;
    string tag = "KET";
    MPSInfo(int n_sites, S vaccum, S target, StateInfo<S> *basis,
            const vector<uint8_t> orbsym, uint8_t n_syms)
        : n_sites(n_sites), vaccum(vaccum), target(target), orbsym(orbsym),
          n_syms(n_syms), basis(basis), bond_dim(0) {
        left_dims_fci = new StateInfo<S>[n_sites + 1];
        left_dims_fci[0] = StateInfo<S>(vaccum);
        for (int i = 0; i < n_sites; i++)
            left_dims_fci[i + 1] = StateInfo<S>::tensor_product(
                left_dims_fci[i], basis[orbsym[i]], target);
        right_dims_fci = new StateInfo<S>[n_sites + 1];
        right_dims_fci[n_sites] = StateInfo<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims_fci[i] = StateInfo<S>::tensor_product(
                basis[orbsym[i]], right_dims_fci[i + 1], target);
        for (int i = 0; i <= n_sites; i++)
            StateInfo<S>::filter(left_dims_fci[i], right_dims_fci[i], target);
        for (int i = 0; i <= n_sites; i++)
            left_dims_fci[i].collect();
        for (int i = n_sites; i >= 0; i--)
            right_dims_fci[i].collect();
        left_dims = new StateInfo<S>[n_sites + 1];
        right_dims = new StateInfo<S>[n_sites + 1];
    }
    virtual AncillaTypes get_ancilla_type() const { return AncillaTypes::None; }
    void set_bond_dimension_using_occ(uint16_t m, const vector<double> &occ,
                                      double bias = 1.0) {
        bond_dim = m;
        // site state probabilities
        StateProbability<S> *site_probs = new StateProbability<S>[n_sites];
        for (int i = 0; i < n_sites; i++) {
            double alpha_occ = occ[i];
            if (bias != 1.0)
                if (alpha_occ > 1)
                    alpha_occ = 1 + pow(alpha_occ - 1, bias);
                else if (alpha_occ < 1)
                    alpha_occ = 1 - pow(1 - alpha_occ, bias);
            alpha_occ /= 2;
            assert(0 <= alpha_occ && alpha_occ <= 1);
            vector<double> probs = {(1 - alpha_occ) * (1 - alpha_occ),
                                    (1 - alpha_occ) * alpha_occ,
                                    alpha_occ * alpha_occ};
            site_probs[i].allocate(basis[orbsym[i]].n);
            for (int j = 0; j < basis[orbsym[i]].n; j++) {
                site_probs[i].quanta[j] = basis[orbsym[i]].quanta[j];
                site_probs[i].probs[j] = probs[basis[orbsym[i]].quanta[j].n()];
            }
        }
        // left and right block probabilities
        StateProbability<S> *left_probs = new StateProbability<S>[n_sites + 1];
        StateProbability<S> *right_probs = new StateProbability<S>[n_sites + 1];
        left_probs[0] = StateProbability<S>(vaccum);
        for (int i = 0; i < n_sites; i++)
            left_probs[i + 1] = StateProbability<S>::tensor_product_no_collect(
                left_probs[i], site_probs[i], left_dims_fci[i + 1]);
        right_probs[n_sites] = StateProbability<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_probs[i] = StateProbability<S>::tensor_product_no_collect(
                site_probs[i], right_probs[i + 1], right_dims_fci[i]);
        // conditional probabilities
        for (int i = 0; i <= n_sites; i++) {
            double *lprobs = dalloc->allocate(left_probs[i].n);
            double *rprobs = dalloc->allocate(right_probs[i].n);
            for (int j = 0; j < left_probs[i].n; j++)
                lprobs[j] = left_probs[i].probs[j] *
                            left_probs[i].quanta[j].multiplicity();
            for (int j = 0; j < right_probs[i].n; j++)
                rprobs[j] = right_probs[i].probs[j] *
                            right_probs[i].quanta[j].multiplicity();
            for (int j = 0; i > 0 && j < left_probs[i].n; j++) {
                if (left_probs[i].probs[j] == 0)
                    continue;
                double x = 0;
                S rks = target - left_probs[i].quanta[j];
                for (int k = 0, ik; k < rks.count(); k++)
                    if ((ik = right_probs[i].find_state(rks[k])) != -1)
                        x += rprobs[ik];
                left_probs[i].probs[j] *= x;
            }
            for (int j = 0; i < n_sites && j < right_probs[i].n; j++) {
                if (right_probs[i].probs[j] == 0)
                    continue;
                double x = 0;
                S lks = target - right_probs[i].quanta[j];
                for (int k = 0, ik; k < lks.count(); k++)
                    if ((ik = left_probs[i].find_state(lks[k])) != -1)
                        x += lprobs[ik];
                right_probs[i].probs[j] *= x;
            }
            dalloc->deallocate(rprobs, right_probs[i].n);
            dalloc->deallocate(lprobs, left_probs[i].n);
        }
        // adjusted temparary fci dims
        StateInfo<S> *left_dims_fci_t = new StateInfo<S>[n_sites + 1];
        StateInfo<S> *right_dims_fci_t = new StateInfo<S>[n_sites + 1];
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims_fci_t[i] = left_dims_fci[i].deep_copy();
            right_dims_fci_t[i] = right_dims_fci[i].deep_copy();
        }
        // left and right block dims
        left_dims[0] = StateInfo<S>(vaccum);
        for (int i = 1; i <= n_sites; i++) {
            left_dims[i].allocate(left_probs[i].n);
            memcpy(left_dims[i].quanta, left_probs[i].quanta,
                   sizeof(S) * left_probs[i].n);
            double prob_sum =
                accumulate(left_probs[i].probs,
                           left_probs[i].probs + left_probs[i].n, 0.0);
            for (int j = 0; j < left_probs[i].n; j++)
                left_dims[i].n_states[j] =
                    min((uint16_t)round(left_probs[i].probs[j] / prob_sum * m),
                        left_dims_fci_t[i].n_states[j]);
            left_dims[i].collect();
            if (i != n_sites) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    left_dims[i], basis[orbsym[i]], left_dims_fci_t[i + 1]);
                for (int j = 0, k; j < left_dims_fci_t[i + 1].n; j++)
                    if ((k = tmp.find_state(
                             left_dims_fci_t[i + 1].quanta[j])) != -1)
                        left_dims_fci_t[i + 1].n_states[j] =
                            min(tmp.n_states[k],
                                left_dims_fci_t[i + 1].n_states[j]);
                for (int j = 0; j < left_probs[i + 1].n; j++)
                    if (tmp.find_state(left_probs[i + 1].quanta[j]) == -1)
                        left_probs[i + 1].probs[j] = 0;
                tmp.deallocate();
            }
        }
        right_dims[n_sites] = StateInfo<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--) {
            right_dims[i].allocate(right_probs[i].n);
            memcpy(right_dims[i].quanta, right_probs[i].quanta,
                   sizeof(S) * right_probs[i].n);
            double prob_sum =
                accumulate(right_probs[i].probs,
                           right_probs[i].probs + right_probs[i].n, 0.0);
            for (int j = 0; j < right_probs[i].n; j++)
                right_dims[i].n_states[j] =
                    min((uint16_t)round(right_probs[i].probs[j] / prob_sum * m),
                        right_dims_fci_t[i].n_states[j]);
            right_dims[i].collect();
            if (i != 0) {
                StateInfo<S> tmp = StateInfo<S>::tensor_product(
                    basis[orbsym[i - 1]], right_dims[i],
                    right_dims_fci_t[i - 1]);
                for (int j = 0, k; j < right_dims_fci_t[i - 1].n; j++)
                    if ((k = tmp.find_state(
                             right_dims_fci_t[i - 1].quanta[j])) != -1)
                        right_dims_fci_t[i - 1].n_states[j] =
                            min(tmp.n_states[k],
                                right_dims_fci_t[i - 1].n_states[j]);
                for (int j = 0; j < right_probs[i - 1].n; j++)
                    if (tmp.find_state(right_probs[i - 1].quanta[j]) == -1)
                        right_probs[i - 1].probs[j] = 0;
                tmp.deallocate();
            }
        }
        for (int i = 0; i < n_sites; i++)
            site_probs[i].reallocate(0);
        for (int i = 0; i <= n_sites; i++)
            left_probs[i].reallocate(0);
        for (int i = n_sites; i >= 0; i--)
            right_probs[i].reallocate(0);
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims_fci_t[i].reallocate(0);
            right_dims_fci_t[i].reallocate(0);
        }
        for (int i = 0; i <= n_sites; i++)
            left_dims[i].reallocate(left_dims[i].n);
        for (int i = n_sites; i >= 0; i--)
            right_dims[i].reallocate(right_dims[i].n);
        assert(ialloc->shift == 0);
        delete[] right_dims_fci_t;
        delete[] left_dims_fci_t;
        delete[] right_probs;
        delete[] left_probs;
        delete[] site_probs;
    }
    void set_bond_dimension(uint16_t m) {
        bond_dim = m;
        left_dims[0] = StateInfo<S>(vaccum);
        for (int i = 0; i < n_sites; i++)
            left_dims[i + 1] = left_dims_fci[i + 1].deep_copy();
        for (int i = 0; i < n_sites; i++)
            if (left_dims[i + 1].n_states_total > m) {
                int new_total = 0;
                for (int k = 0; k < left_dims[i + 1].n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)left_dims[i + 1].n_states[k] *
                                        m / left_dims[i + 1].n_states_total) +
                                   0.1);
                    left_dims[i + 1].n_states[k] =
                        (uint16_t)min(new_n_states, 65535U);
                    new_total += left_dims[i + 1].n_states[k];
                }
                left_dims[i + 1].n_states_total = new_total;
            }
        right_dims[n_sites] = StateInfo<S>(vaccum);
        for (int i = n_sites - 1; i >= 0; i--)
            right_dims[i] = right_dims_fci[i].deep_copy();
        for (int i = n_sites - 1; i >= 0; i--)
            if (right_dims[i].n_states_total > m) {
                int new_total = 0;
                for (int k = 0; k < right_dims[i].n; k++) {
                    uint32_t new_n_states =
                        (uint32_t)(ceil((double)right_dims[i].n_states[k] * m /
                                        right_dims[i].n_states_total) +
                                   0.1);
                    right_dims[i].n_states[k] =
                        (uint16_t)min(new_n_states, 65535U);
                    new_total += right_dims[i].n_states[k];
                }
                right_dims[i].n_states_total = new_total;
            }
        for (int i = -1; i < n_sites - 1; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                left_dims[i + 1], basis[orbsym[i + 1]], target);
            int new_total = 0;
            for (int k = 0; k < left_dims[i + 2].n; k++) {
                int tk = t.find_state(left_dims[i + 2].quanta[k]);
                if (tk == -1)
                    left_dims[i + 2].n_states[k] = 0;
                else if (left_dims[i + 2].n_states[k] > t.n_states[tk])
                    left_dims[i + 2].n_states[k] = t.n_states[tk];
                new_total += left_dims[i + 2].n_states[k];
            }
            left_dims[i + 2].n_states_total = new_total;
            t.deallocate();
        }
        for (int i = n_sites; i > 0; i--) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                basis[orbsym[i - 1]], right_dims[i], target);
            int new_total = 0;
            for (int k = 0; k < right_dims[i - 1].n; k++) {
                int tk = t.find_state(right_dims[i - 1].quanta[k]);
                if (tk == -1)
                    right_dims[i - 1].n_states[k] = 0;
                else if (right_dims[i - 1].n_states[k] > t.n_states[tk])
                    right_dims[i - 1].n_states[k] = t.n_states[tk];
                new_total += right_dims[i - 1].n_states[k];
            }
            right_dims[i - 1].n_states_total = new_total;
            t.deallocate();
        }
    }
    string get_filename(bool left, int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MPS.INFO." << tag
           << (left ? ".LEFT." : ".RIGHT.") << Parsing::to_string(i);
        return ss.str();
    }
    void save_mutable() const {
        for (int i = 0; i < n_sites + 1; i++) {
            left_dims[i].save_data(get_filename(true, i));
            right_dims[i].save_data(get_filename(false, i));
        }
    }
    void load_mutable() const {
        for (int i = 0; i <= n_sites; i++)
            left_dims[i].load_data(get_filename(true, i));
        for (int i = n_sites; i >= 0; i--)
            right_dims[i].load_data(get_filename(false, i));
    }
    void deallocate_mutable() {
        for (int i = 0; i <= n_sites; i++)
            right_dims[i].deallocate();
        for (int i = n_sites; i >= 0; i--)
            left_dims[i].deallocate();
    }
    void save_left_dims(int i) const {
        left_dims[i].save_data(get_filename(true, i));
    }
    void save_right_dims(int i) const {
        right_dims[i].save_data(get_filename(false, i));
    }
    void load_left_dims(int i) {
        left_dims[i].load_data(get_filename(true, i));
    }
    void load_right_dims(int i) {
        right_dims[i].load_data(get_filename(false, i));
    }
    void deallocate() {
        for (int i = 0; i <= n_sites; i++)
            right_dims_fci[i].deallocate();
        for (int i = n_sites; i >= 0; i--)
            left_dims_fci[i].deallocate();
    }
    ~MPSInfo() {
        delete[] left_dims;
        delete[] right_dims;
        delete[] left_dims_fci;
        delete[] right_dims_fci;
    }
};

template <typename S> struct AncillaMPSInfo : MPSInfo<S> {
    int n_physical_sites;
    static vector<uint8_t> trans_orbsym(const vector<uint8_t> &a, int n_sites) {
        vector<uint8_t> b(n_sites << 1, 0);
        for (int i = 0, j = 0; i < n_sites; i++, j += 2)
            b[j] = b[j + 1] = a[i];
        return b;
    }
    AncillaMPSInfo(int n_sites, S vaccum, S target, StateInfo<S> *basis,
                   const vector<uint8_t> &orbsym, uint8_t n_syms)
        : n_physical_sites(n_sites), MPSInfo<S>(n_sites << 1, vaccum, target,
                                                basis,
                                                trans_orbsym(orbsym, n_sites),
                                                n_syms) {}
    AncillaTypes get_ancilla_type() const override {
        return AncillaTypes::Ancilla;
    }
    void set_thermal_limit() {
        MPSInfo<S>::left_dims[0] = StateInfo<S>(MPSInfo<S>::vaccum);
        for (int i = 0; i < MPSInfo<S>::n_sites; i++)
            if (i & 1) {
                S q = MPSInfo<S>::left_dims[i]
                          .quanta[MPSInfo<S>::left_dims[i].n - 1] +
                      MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]].quanta[0];
                assert(q.count() == 1);
                MPSInfo<S>::left_dims[i + 1] = StateInfo<S>(q);
            } else
                MPSInfo<S>::left_dims[i + 1] = StateInfo<S>::tensor_product(
                    MPSInfo<S>::left_dims[i],
                    MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]],
                    MPSInfo<S>::target);
        MPSInfo<S>::right_dims[MPSInfo<S>::n_sites] =
            StateInfo<S>(MPSInfo<S>::vaccum);
        for (int i = MPSInfo<S>::n_sites - 1; i >= 0; i--)
            if (i & 1)
                MPSInfo<S>::right_dims[i] = StateInfo<S>::tensor_product(
                    MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]],
                    MPSInfo<S>::right_dims[i + 1], MPSInfo<S>::target);
            else {
                S q = MPSInfo<S>::basis[MPSInfo<S>::orbsym[i]].quanta[0] +
                      MPSInfo<S>::right_dims[i + 1]
                          .quanta[MPSInfo<S>::right_dims[i + 1].n - 1];
                assert(q.count() == 1);
                MPSInfo<S>::right_dims[i] = StateInfo<S>(q);
            }
    }
};

template <typename S> struct MPS {
    int n_sites, center, dot;
    shared_ptr<MPSInfo<S>> info;
    vector<shared_ptr<SparseMatrix<S>>> tensors;
    string canonical_form;
    MPS(const shared_ptr<MPSInfo<S>> &info)
        : n_sites(0), center(0), dot(0), info(info) {}
    MPS(int n_sites, int center, int dot)
        : n_sites(n_sites), center(center), dot(dot) {
        canonical_form.resize(n_sites);
        for (int i = 0; i < center; i++)
            canonical_form[i] = 'L';
        for (int i = center; i < center + dot; i++)
            canonical_form[i] = 'C';
        for (int i = center + dot; i < n_sites; i++)
            canonical_form[i] = 'R';
    }
    void initialize(const shared_ptr<MPSInfo<S>> &info) {
        this->info = info;
        vector<shared_ptr<SparseMatrixInfo<S>>> mat_infos;
        mat_infos.resize(n_sites);
        tensors.resize(n_sites);
        for (int i = 0; i < center; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                info->left_dims[i], info->basis[info->orbsym[i]],
                info->left_dims_fci[i + 1]);
            mat_infos[i] = make_shared<SparseMatrixInfo<S>>();
            mat_infos[i]->initialize(t, info->left_dims[i + 1], info->vaccum,
                                     false);
            t.reallocate(0);
            mat_infos[i]->reallocate(mat_infos[i]->n);
        }
        mat_infos[center] = make_shared<SparseMatrixInfo<S>>();
        if (dot == 1) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                info->left_dims[center], info->basis[info->orbsym[center]],
                info->left_dims_fci[center + dot]);
            mat_infos[center]->initialize(t, info->right_dims[center + dot],
                                          info->target, false, true);
            t.reallocate(0);
            mat_infos[center]->reallocate(mat_infos[center]->n);
        } else {
            StateInfo<S> tl = StateInfo<S>::tensor_product(
                info->left_dims[center], info->basis[info->orbsym[center]],
                info->left_dims_fci[center + 1]);
            StateInfo<S> tr = StateInfo<S>::tensor_product(
                info->basis[info->orbsym[center + 1]],
                info->right_dims[center + dot],
                info->right_dims_fci[center + 1]);
            mat_infos[center]->initialize(tl, tr, info->target, false, true);
            tl.reallocate(0);
            tr.reallocate(0);
            mat_infos[center]->reallocate(mat_infos[center]->n);
        }
        for (int i = center + dot; i < n_sites; i++) {
            StateInfo<S> t = StateInfo<S>::tensor_product(
                info->basis[info->orbsym[i]], info->right_dims[i + 1],
                info->right_dims_fci[i]);
            mat_infos[i] = make_shared<SparseMatrixInfo<S>>();
            mat_infos[i]->initialize(info->right_dims[i], t, info->vaccum,
                                     false);
            t.reallocate(0);
            mat_infos[i]->reallocate(mat_infos[i]->n);
        }
        for (int i = 0; i < n_sites; i++)
            if (mat_infos[i] != nullptr) {
                tensors[i] = make_shared<SparseMatrix<S>>();
                tensors[i]->allocate(mat_infos[i]);
            }
    }
    void fill_thermal_limit() {
        assert(info->get_ancilla_type() == AncillaTypes::Ancilla);
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                if (i < center || i > center || (i == center && dot == 1)) {
                    int n = info->basis[info->orbsym[i]].n;
                    assert(tensors[i]->total_memory == n);
                    if (i & 1)
                        for (int j = 0; j < n; j++)
                            tensors[i]->data[j] = 1.0;
                    else {
                        double norm = 0;
                        for (int j = 0; j < n; j++)
                            norm += info->basis[info->orbsym[i]]
                                        .quanta[j]
                                        .multiplicity();
                        norm = sqrt(norm);
                        for (int j = 0; j < n; j++)
                            tensors[i]->data[j] =
                                sqrt(info->basis[info->orbsym[i]]
                                         .quanta[j]
                                         .multiplicity()) /
                                norm;
                    }
                } else {
                    assert(!(i & 1));
                    assert(info->basis[info->orbsym[i]].n ==
                           tensors[i]->info->n);
                    double norm = 0;
                    for (int j = 0; j < tensors[i]->info->n; j++)
                        norm += tensors[i]->info->quanta[j].multiplicity();
                    norm = sqrt(norm);
                    for (int j = 0; j < tensors[i]->info->n; j++) {
                        assert((*tensors[i])[j].size() == 1);
                        (*tensors[i])[j](0, 0) =
                            sqrt(tensors[i]->info->quanta[j].multiplicity()) /
                            norm;
                    }
                }
            }
    }
    void canonicalize() {
        for (int i = 0; i < center; i++) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> tmat = make_shared<SparseMatrix<S>>();
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>();
            tmat_info->initialize(info->left_dims[i + 1],
                                  info->left_dims[i + 1], info->vaccum, false);
            tmat->allocate(tmat_info);
            tensors[i]->left_canonicalize(tmat);
            StateInfo<S> l = info->left_dims[i + 1],
                         m = info->basis[info->orbsym[i + 1]];
            StateInfo<S> lm = StateInfo<S>::tensor_product(
                             l, m, info->left_dims_fci[i + 2]),
                         r;
            StateInfo<S> lmc = StateInfo<S>::get_connection_info(l, m, lm);
            if (i + 1 == center && dot == 1)
                r = info->right_dims[center + dot];
            else if (i + 1 == center && dot == 2)
                r = StateInfo<S>::tensor_product(
                    info->basis[info->orbsym[center + 1]],
                    info->right_dims[center + dot],
                    info->right_dims_fci[center + 1]);
            else
                r = info->left_dims[i + 2];
            tensors[i + 1]->left_multiply(tmat, l, m, r, lm, lmc);
            if (i + 1 == center && dot == 2)
                r.deallocate();
            lmc.deallocate();
            lm.deallocate();
            tmat_info->deallocate();
            tmat->deallocate();
        }
        for (int i = n_sites - 1; i >= center + dot; i--) {
            assert(tensors[i] != nullptr);
            shared_ptr<SparseMatrix<S>> tmat = make_shared<SparseMatrix<S>>();
            shared_ptr<SparseMatrixInfo<S>> tmat_info =
                make_shared<SparseMatrixInfo<S>>();
            tmat_info->initialize(info->right_dims[i], info->right_dims[i],
                                  info->vaccum, false);
            tmat->allocate(tmat_info);
            tensors[i]->right_canonicalize(tmat);
            if (dot == 1 && i - 1 == center) {
                shared_ptr<SparseMatrix<S>> tmp =
                    make_shared<SparseMatrix<S>>();
                tmp->allocate(tensors[i - 1]->info);
                tmp->copy_data_from(*tensors[i - 1]);
                tensors[i - 1]->contract(tmp, tmat);
                tmp->deallocate();
            } else {
                StateInfo<S> m = info->basis[info->orbsym[i - 1]],
                             r = info->right_dims[i];
                StateInfo<S> mr = StateInfo<S>::tensor_product(
                    m, r, info->right_dims_fci[i - 1]);
                StateInfo<S> mrc = StateInfo<S>::get_connection_info(m, r, mr);
                StateInfo<S> l;
                if (i - 1 == center + 1 && dot == 2) {
                    l = StateInfo<S>::tensor_product(
                        info->left_dims[center],
                        info->basis[info->orbsym[center]],
                        info->left_dims_fci[center + 1]);
                    tensors[i - 2]->right_multiply(tmat, l, m, r, mr, mrc);
                } else {
                    l = info->right_dims[i - 1];
                    tensors[i - 1]->right_multiply(tmat, l, m, r, mr, mrc);
                }
                if (i - 1 == center + 1 && dot == 2)
                    l.deallocate();
                mrc.deallocate();
                mr.deallocate();
            }
            tmat_info->deallocate();
            tmat->deallocate();
        }
    }
    void random_canonicalize() {
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr) {
                shared_ptr<SparseMatrix<S>> tmat =
                    make_shared<SparseMatrix<S>>();
                shared_ptr<SparseMatrixInfo<S>> tmat_info =
                    make_shared<SparseMatrixInfo<S>>();
                tensors[i]->randomize();
                if (i < center) {
                    tmat_info->initialize(info->left_dims[i + 1],
                                          info->left_dims[i + 1], info->vaccum,
                                          false);
                    tmat->allocate(tmat_info);
                    tensors[i]->left_canonicalize(tmat);
                } else if (i > center) {
                    tmat_info->initialize(info->right_dims[i],
                                          info->right_dims[i], info->vaccum,
                                          false);
                    tmat->allocate(tmat_info);
                    tensors[i]->right_canonicalize(tmat);
                }
                if (i != center) {
                    tmat_info->deallocate();
                    tmat->deallocate();
                }
            }
    }
    string get_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".MPS." << info->tag
           << "." << Parsing::to_string(i);
        return ss.str();
    }
    void load_data() {
        ifstream ifs(get_filename(-1).c_str(), ios::binary);
        ifs.read((char *)&n_sites, sizeof(n_sites));
        ifs.read((char *)&center, sizeof(center));
        ifs.read((char *)&dot, sizeof(dot));
        canonical_form = string(n_sites, ' ');
        ifs.read((char *)&canonical_form[0], sizeof(char) * n_sites);
        vector<uint8_t> bs(n_sites);
        ifs.read((char *)&bs[0], sizeof(uint8_t) * n_sites);
        ifs.close();
        tensors.resize(n_sites, nullptr);
        for (int i = 0; i < n_sites; i++)
            if (bs[i])
                tensors[i] = make_shared<SparseMatrix<S>>();
    }
    void save_data() const {
        ofstream ofs(get_filename(-1).c_str(), ios::binary);
        ofs.write((char *)&n_sites, sizeof(n_sites));
        ofs.write((char *)&center, sizeof(center));
        ofs.write((char *)&dot, sizeof(dot));
        ofs.write((char *)&canonical_form[0], sizeof(char) * n_sites);
        vector<uint8_t> bs(n_sites);
        for (int i = 0; i < n_sites; i++)
            bs[i] = uint8_t(tensors[i] != nullptr);
        ofs.write((char *)&bs[0], sizeof(uint8_t) * n_sites);
        ofs.close();
    }
    void load_mutable() const {
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr)
                tensors[i]->load_data(get_filename(i), true);
    }
    void save_mutable() const {
        for (int i = 0; i < n_sites; i++)
            if (tensors[i] != nullptr)
                tensors[i]->save_data(get_filename(i), true);
    }
    void save_tensor(int i) const {
        assert(tensors[i] != nullptr);
        tensors[i]->save_data(get_filename(i), true);
    }
    void load_tensor(int i) {
        assert(tensors[i] != nullptr);
        tensors[i]->load_data(get_filename(i), true);
    }
    void unload_tensor(int i) {
        assert(tensors[i] != nullptr);
        tensors[i]->info->deallocate();
        tensors[i]->deallocate();
    }
    void deallocate() {
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->deallocate();
        for (int i = n_sites - 1; i >= 0; i--)
            if (tensors[i] != nullptr)
                tensors[i]->info->deallocate();
    }
};

template <typename S> struct Partition {
    shared_ptr<OperatorTensor<S>> left;
    shared_ptr<OperatorTensor<S>> right;
    vector<shared_ptr<OperatorTensor<S>>> middle;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> right_op_infos;
    Partition(const shared_ptr<OperatorTensor<S>> &left,
              const shared_ptr<OperatorTensor<S>> &right,
              const shared_ptr<OperatorTensor<S>> &dot)
        : left(left), right(right), middle{dot} {}
    Partition(const shared_ptr<OperatorTensor<S>> &left,
              const shared_ptr<OperatorTensor<S>> &right,
              const shared_ptr<OperatorTensor<S>> &ldot,
              const shared_ptr<OperatorTensor<S>> &rdot)
        : left(left), right(right), middle{ldot, rdot} {}
    Partition(const Partition &other)
        : left(other.left), right(other.right), middle(other.middle) {}
    static shared_ptr<SparseMatrixInfo<S>> find_op_info(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &op_infos, S q) {
        auto p = lower_bound(op_infos.begin(), op_infos.end(), q,
                             SparseMatrixInfo<S>::cmp_op_info);
        if (p == op_infos.end() || p->first != q)
            return nullptr;
        else
            return p->second;
    }
    static shared_ptr<OperatorTensor<S>> build_left(
        const vector<shared_ptr<Symbolic<S>>> &mats,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos) {
        shared_ptr<OperatorTensor<S>> opt = make_shared<OperatorTensor<S>>();
        assert(mats[0] != nullptr);
        assert(mats[0]->get_type() == SymTypes::RVec);
        opt->lmat = make_shared<SymbolicRowVector<S>>(
            *dynamic_pointer_cast<SymbolicRowVector<S>>(mats[0]));
        for (auto &mat : mats) {
            for (size_t i = 0; i < mat->data.size(); i++)
                if (mat->data[i]->get_type() != OpTypes::Zero) {
                    shared_ptr<OpExpr<S>> op = abs_value(mat->data[i]);
                    opt->ops[op] = make_shared<SparseMatrix<S>>();
                }
        }
        for (auto &p : opt->ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            p.second->allocate(find_op_info(left_op_infos, op->q_label));
        }
        return opt;
    }
    static shared_ptr<OperatorTensor<S>>
    build_right(const vector<shared_ptr<Symbolic<S>>> &mats,
                const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
                    &right_op_infos) {
        shared_ptr<OperatorTensor<S>> opt = make_shared<OperatorTensor<S>>();
        assert(mats[0] != nullptr);
        assert(mats[0]->get_type() == SymTypes::CVec);
        opt->rmat = make_shared<SymbolicColumnVector<S>>(
            *dynamic_pointer_cast<SymbolicColumnVector<S>>(mats[0]));
        for (auto &mat : mats) {
            for (size_t i = 0; i < mat->data.size(); i++)
                if (mat->data[i]->get_type() != OpTypes::Zero) {
                    shared_ptr<OpExpr<S>> op = abs_value(mat->data[i]);
                    opt->ops[op] = make_shared<SparseMatrix<S>>();
                }
        }
        for (auto &p : opt->ops) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(p.first);
            p.second->allocate(find_op_info(right_op_infos, op->q_label));
        }
        return opt;
    }
    static vector<S>
    get_uniq_labels(const vector<shared_ptr<Symbolic<S>>> &mats) {
        vector<S> sl;
        for (auto &mat : mats) {
            assert(mat != nullptr);
            assert(mat->get_type() == SymTypes::RVec ||
                   mat->get_type() == SymTypes::CVec);
            sl.reserve(sl.size() + mat->data.size());
            for (size_t i = 0; i < mat->data.size(); i++) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(mat->data[i]);
                sl.push_back(op->q_label);
            }
        }
        sort(sl.begin(), sl.end());
        sl.resize(distance(sl.begin(), unique(sl.begin(), sl.end())));
        return sl;
    }
    static vector<vector<pair<uint8_t, S>>>
    get_uniq_sub_labels(const shared_ptr<Symbolic<S>> &exprs,
                        const shared_ptr<Symbolic<S>> &mat,
                        const vector<S> &sl) {
        vector<vector<pair<uint8_t, S>>> subsl(sl.size());
        if (exprs == nullptr)
            return subsl;
        assert(mat->data.size() == exprs->data.size());
        for (size_t i = 0; i < mat->data.size(); i++) {
            shared_ptr<OpElement<S>> op =
                dynamic_pointer_cast<OpElement<S>>(mat->data[i]);
            S l = op->q_label;
            size_t idx = lower_bound(sl.begin(), sl.end(), l) - sl.begin();
            assert(idx != sl.size());
            switch (exprs->data[i]->get_type()) {
            case OpTypes::Zero:
                break;
            case OpTypes::Prod: {
                shared_ptr<OpString<S>> op =
                    dynamic_pointer_cast<OpString<S>>(exprs->data[i]);
                assert(op->b != nullptr);
                S bra = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                S ket = (op->conj & 2) ? op->b->q_label : -op->b->q_label;
                S p = l.combine(bra, ket);
                assert(p != S(0xFFFFFFFFU));
                subsl[idx].push_back(make_pair(op->conj, p));
            } break;
            case OpTypes::Sum: {
                shared_ptr<OpSum<S>> sop =
                    dynamic_pointer_cast<OpSum<S>>(exprs->data[i]);
                for (auto &op : sop->strings) {
                    S bra, ket;
                    if (op->get_type() == OpTypes::Prod) {
                        assert(op->b != nullptr);
                        bra = (op->conj & 1) ? -op->a->q_label : op->a->q_label;
                        ket = (op->conj & 2) ? op->b->q_label : -op->b->q_label;
                    } else {
                        assert(op->get_type() == OpTypes::SumProd);
                        shared_ptr<OpSumProd<S>> spop =
                            dynamic_pointer_cast<OpSumProd<S>>(op);
                        assert(spop->ops.size() != 0);
                        if (spop->a != nullptr) {
                            bra = (op->conj & 1) ? -op->a->q_label
                                                 : op->a->q_label;
                            ket = (op->conj & 2) ? spop->ops[0]->q_label
                                                 : -spop->ops[0]->q_label;
                        } else if (spop->b != nullptr) {
                            bra = (op->conj & 1) ? -spop->ops[0]->q_label
                                                 : spop->ops[0]->q_label;
                            ket = (op->conj & 2) ? op->b->q_label
                                                 : -op->b->q_label;
                        } else
                            assert(false);
                    }
                    S p = l.combine(bra, ket);
                    assert(p != S(0xFFFFFFFFU));
                    subsl[idx].push_back(make_pair(op->conj, p));
                }
            } break;
            default:
                assert(false);
            }
        }
        for (size_t i = 0; i < subsl.size(); i++) {
            sort(subsl[i].begin(), subsl[i].end());
            subsl[i].resize(distance(subsl[i].begin(),
                                     unique(subsl[i].begin(), subsl[i].end())));
        }
        return subsl;
    }
    static void deallocate_op_infos_notrunc(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &op_infos_notrunc) {
        for (int i = op_infos_notrunc.size() - 1; i >= 0; i--) {
            op_infos_notrunc[i].second->cinfo->deallocate();
            op_infos_notrunc[i].second->deallocate();
        }
    }
    static void copy_op_infos(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &from_op_infos,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &to_op_infos) {
        assert(to_op_infos.size() == 0);
        to_op_infos.reserve(from_op_infos.size());
        for (size_t i = 0; i < from_op_infos.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> info =
                make_shared<SparseMatrixInfo<S>>(
                    from_op_infos[i].second->deep_copy());
            to_op_infos.push_back(make_pair(from_op_infos[i].first, info));
        }
    }
    static void init_left_op_infos(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos) {
        frame->activate(0);
        bra_info->load_left_dims(m + 1);
        StateInfo<S> ibra = bra_info->left_dims[m + 1], iket = ibra;
        if (bra_info != ket_info) {
            ket_info->load_left_dims(m + 1);
            iket = ket_info->left_dims[m + 1];
        }
        frame->activate(1);
        assert(left_op_infos.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> lop =
                make_shared<SparseMatrixInfo<S>>();
            left_op_infos.push_back(make_pair(sl[i], lop));
            lop->initialize(ibra, iket, sl[i], sl[i].is_fermion());
        }
        frame->activate(0);
        if (bra_info != ket_info)
            iket.deallocate();
        ibra.deallocate();
    }
    static void init_left_op_infos_notrunc(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        const vector<vector<pair<uint8_t, S>>> &subsl,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &prev_left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &site_op_infos,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos_notrunc,
        const shared_ptr<CG<S>> &cg) {
        frame->activate(1);
        bra_info->load_left_dims(m);
        StateInfo<S> ibra_prev = bra_info->left_dims[m], iket_prev = ibra_prev;
        StateInfo<S> ibra_notrunc = StateInfo<S>::tensor_product(
                         ibra_prev, bra_info->basis[bra_info->orbsym[m]],
                         bra_info->left_dims_fci[m + 1]),
                     iket_notrunc = ibra_notrunc;
        StateInfo<S> ibra_cinfo = StateInfo<S>::get_connection_info(
                         ibra_prev, bra_info->basis[bra_info->orbsym[m]],
                         ibra_notrunc),
                     iket_cinfo = ibra_cinfo;
        if (bra_info != ket_info) {
            ket_info->load_left_dims(m);
            iket_prev = ket_info->left_dims[m];
            iket_notrunc = StateInfo<S>::tensor_product(
                iket_prev, ket_info->basis[ket_info->orbsym[m]],
                ket_info->left_dims_fci[m + 1]);
            iket_cinfo = StateInfo<S>::get_connection_info(
                iket_prev, ket_info->basis[ket_info->orbsym[m]], iket_notrunc);
        }
        frame->activate(0);
        assert(left_op_infos_notrunc.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> lop_notrunc =
                make_shared<SparseMatrixInfo<S>>();
            left_op_infos_notrunc.push_back(make_pair(sl[i], lop_notrunc));
            lop_notrunc->initialize(ibra_notrunc, iket_notrunc, sl[i],
                                    sl[i].is_fermion());
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            cinfo->initialize_tp(
                sl[i], subsl[i], ibra_notrunc, iket_notrunc, ibra_prev,
                bra_info->basis[bra_info->orbsym[m]], iket_prev,
                ket_info->basis[ket_info->orbsym[m]], ibra_cinfo, iket_cinfo,
                prev_left_op_infos, site_op_infos, lop_notrunc, cg);
            lop_notrunc->cinfo = cinfo;
        }
        frame->activate(1);
        if (bra_info != ket_info) {
            iket_cinfo.deallocate();
            iket_notrunc.deallocate();
            iket_prev.deallocate();
        }
        ibra_cinfo.deallocate();
        ibra_notrunc.deallocate();
        ibra_prev.deallocate();
    }
    static void init_right_op_infos(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos) {
        frame->activate(0);
        bra_info->load_right_dims(m);
        StateInfo<S> ibra = bra_info->right_dims[m], iket = ibra;
        if (bra_info != ket_info) {
            ket_info->load_right_dims(m);
            iket = ket_info->right_dims[m];
        }
        frame->activate(1);
        assert(right_op_infos.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> rop =
                make_shared<SparseMatrixInfo<S>>();
            right_op_infos.push_back(make_pair(sl[i], rop));
            rop->initialize(ibra, iket, sl[i], sl[i].is_fermion());
        }
        frame->activate(0);
        if (bra_info != ket_info)
            iket.deallocate();
        ibra.deallocate();
    }
    static void init_right_op_infos_notrunc(
        int m, const shared_ptr<MPSInfo<S>> &bra_info,
        const shared_ptr<MPSInfo<S>> &ket_info, const vector<S> &sl,
        const vector<vector<pair<uint8_t, S>>> &subsl,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &prev_right_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &site_op_infos,
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>
            &right_op_infos_notrunc,
        const shared_ptr<CG<S>> &cg) {
        frame->activate(1);
        bra_info->load_right_dims(m + 1);
        StateInfo<S> ibra_prev = bra_info->right_dims[m + 1],
                     iket_prev = ibra_prev;
        StateInfo<S> ibra_notrunc = StateInfo<S>::tensor_product(
                         bra_info->basis[bra_info->orbsym[m]], ibra_prev,
                         bra_info->right_dims_fci[m]),
                     iket_notrunc = ibra_notrunc;
        StateInfo<S> ibra_cinfo = StateInfo<S>::get_connection_info(
                         bra_info->basis[bra_info->orbsym[m]], ibra_prev,
                         ibra_notrunc),
                     iket_cinfo = ibra_cinfo;
        if (bra_info != ket_info) {
            ket_info->load_right_dims(m + 1);
            iket_prev = ket_info->right_dims[m + 1];
            iket_notrunc = StateInfo<S>::tensor_product(
                ket_info->basis[ket_info->orbsym[m]], iket_prev,
                ket_info->right_dims_fci[m]);
            iket_cinfo = StateInfo<S>::get_connection_info(
                ket_info->basis[ket_info->orbsym[m]], iket_prev, iket_notrunc);
        }
        frame->activate(0);
        assert(right_op_infos_notrunc.size() == 0);
        for (size_t i = 0; i < sl.size(); i++) {
            shared_ptr<SparseMatrixInfo<S>> rop_notrunc =
                make_shared<SparseMatrixInfo<S>>();
            right_op_infos_notrunc.push_back(make_pair(sl[i], rop_notrunc));
            rop_notrunc->initialize(ibra_notrunc, iket_notrunc, sl[i],
                                    sl[i].is_fermion());
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> cinfo =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            cinfo->initialize_tp(
                sl[i], subsl[i], ibra_notrunc, iket_notrunc,
                bra_info->basis[bra_info->orbsym[m]], ibra_prev,
                ket_info->basis[ket_info->orbsym[m]], iket_prev, ibra_cinfo,
                iket_cinfo, site_op_infos, prev_right_op_infos, rop_notrunc,
                cg);
            rop_notrunc->cinfo = cinfo;
        }
        frame->activate(1);
        if (bra_info != ket_info) {
            iket_cinfo.deallocate();
            iket_notrunc.deallocate();
            iket_prev.deallocate();
        }
        ibra_cinfo.deallocate();
        ibra_notrunc.deallocate();
        ibra_prev.deallocate();
    }
};

template <typename S> struct EffectiveHamiltonian {
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
        right_op_infos;
    shared_ptr<DelayedOperatorTensor<S>> op;
    shared_ptr<SparseMatrix<S>> bra, ket, diag, cmat, vmat;
    shared_ptr<TensorFunctions<S>> tf;
    S opdq;
    bool compute_diag;
    EffectiveHamiltonian(
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &left_op_infos,
        const vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> &right_op_infos,
        const shared_ptr<DelayedOperatorTensor<S>> &op,
        const shared_ptr<SparseMatrix<S>> &bra,
        const shared_ptr<SparseMatrix<S>> &ket,
        const shared_ptr<OpElement<S>> &hop,
        const shared_ptr<SymbolicColumnVector<S>> &hop_mat,
        const shared_ptr<TensorFunctions<S>> &tf, bool compute_diag = true)
        : left_op_infos(left_op_infos), right_op_infos(right_op_infos), op(op),
          bra(bra), ket(ket), tf(tf), compute_diag(compute_diag) {
        // wavefunction
        if (compute_diag) {
            assert(bra == ket);
            diag = make_shared<SparseMatrix<S>>();
            diag->allocate(ket->info);
        }
        // unique sub labels
        S cdq = ket->info->delta_quantum;
        S vdq = bra->info->delta_quantum;
        opdq = hop->q_label;
        vector<S> msl = Partition<S>::get_uniq_labels({hop_mat});
        assert(msl[0] == opdq);
        vector<vector<pair<uint8_t, S>>> msubsl =
            Partition<S>::get_uniq_sub_labels(op->mat, hop_mat, msl);
        // tensor prodcut diagonal
        if (compute_diag) {
            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> diag_info =
                make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
            diag_info->initialize_diag(cdq, opdq, msubsl[0], left_op_infos,
                                       right_op_infos, diag->info, tf->opf->cg);
            diag->info->cinfo = diag_info;
            tf->tensor_product_diagonal(op->mat->data[0], op->lops, op->rops,
                                        diag, opdq);
            if (tf->opf->seq->mode == SeqTypes::Auto)
                tf->opf->seq->auto_perform();
            diag_info->deallocate();
        }
        // temp wavefunction
        cmat = make_shared<SparseMatrix<S>>();
        vmat = make_shared<SparseMatrix<S>>();
        *cmat = *ket;
        *vmat = *bra;
        // temp wavefunction info
        shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> wfn_info =
            make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
        wfn_info->initialize_wfn(cdq, vdq, opdq, msubsl[0], left_op_infos,
                                 right_op_infos, ket->info, bra->info,
                                 tf->opf->cg);
        cmat->info->cinfo = wfn_info;
        // prepare batch gemm
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            cmat->data = vmat->data = (double *)0;
            tf->tensor_product_multiply(op->mat->data[0], op->lops, op->rops,
                                        cmat, vmat, opdq);
            tf->opf->seq->prepare();
            tf->opf->seq->allocate();
        }
    }
    void operator()(const MatrixRef &b, const MatrixRef &c, int idx = 0,
                    double factor = 1.0) {
        assert(b.m * b.n == cmat->total_memory);
        assert(c.m * c.n == vmat->total_memory);
        cmat->data = b.data;
        vmat->data = c.data;
        cmat->factor = factor;
        tf->tensor_product_multiply(op->mat->data[idx], op->lops, op->rops,
                                    cmat, vmat, opdq);
    }
    // energy, ndav, nflop, tdav
    tuple<double, int, size_t, double> eigs(bool iprint = false) {
        int ndav = 0;
        assert(compute_diag);
        DiagonalMatrix aa(diag->data, diag->total_memory);
        vector<MatrixRef> bs =
            vector<MatrixRef>{MatrixRef(ket->data, ket->total_memory, 1)};
        frame->activate(0);
        Timer t;
        t.get_time();
        vector<double> eners =
            tf->opf->seq->mode == SeqTypes::Auto
                ? MatrixFunctions::davidson(*tf->opf->seq, aa, bs, ndav, iprint)
                : MatrixFunctions::davidson(*this, aa, bs, ndav, iprint);
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(eners[0], ndav, nflop, t.get_time());
    }
    // norm, nflop, tdav
    tuple<double, size_t, double> multiply() {
        bra->clear();
        Timer t;
        t.get_time();
        if (tf->opf->seq->mode == SeqTypes::Auto)
            (*tf->opf->seq)(MatrixRef(ket->data, ket->total_memory, 1),
                            MatrixRef(bra->data, bra->total_memory, 1));
        else
            (*this)(MatrixRef(ket->data, ket->total_memory, 1),
                    MatrixRef(bra->data, bra->total_memory, 1));
        double norm =
            MatrixFunctions::norm(MatrixRef(bra->data, bra->total_memory, 1));
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(norm, nflop, t.get_time());
    }
    // expectations, nflop, tdav
    tuple<vector<pair<shared_ptr<OpExpr<S>>, double>>, size_t, double>
    expect() {
        Timer t;
        t.get_time();
        MatrixRef ktmp(ket->data, ket->total_memory, 1);
        MatrixRef rtmp(bra->data, bra->total_memory, 1);
        MatrixRef btmp(nullptr, bra->total_memory, 1);
        btmp.allocate();
        assert(tf->opf->seq->mode != SeqTypes::Auto);
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations;
        expectations.reserve(op->mat->data.size());
        for (size_t i = 0; i < op->mat->data.size(); i++) {
            if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->name ==
                OpNames::Zero)
                continue;
            else if (dynamic_pointer_cast<OpElement<S>>(op->ops[i])->q_label !=
                     opdq)
                expectations.push_back(make_pair(op->ops[i], 0.0));
            else {
                btmp.clear();
                (*this)(ktmp, btmp, i);
                double r = MatrixFunctions::dot(btmp, rtmp);
                expectations.push_back(make_pair(op->ops[i], r));
            }
        }
        btmp.deallocate();
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(expectations, nflop, t.get_time());
    }
    // k1~k4, energy, norm, nexpo, nflop, texpo
    pair<vector<MatrixRef>, tuple<double, double, int, size_t, double>>
    rk4_apply(double beta, double const_e, bool eval_energy = false) {
        MatrixRef v(ket->data, ket->total_memory, 1);
        vector<MatrixRef> k, r;
        Timer t;
        t.get_time();
        frame->activate(1);
        for (int i = 0; i < 3; i++) {
            r.push_back(MatrixRef(nullptr, ket->total_memory, 1));
            r[i].allocate();
        }
        frame->activate(0);
        for (int i = 0; i < 4; i++) {
            k.push_back(MatrixRef(nullptr, ket->total_memory, 1));
            k[i].allocate(), k[i].clear();
        }
        assert(tf->opf->seq->mode != SeqTypes::Auto);
        const vector<double> ks = vector<double>{0.0, 0.5, 0.5, 1.0};
        const vector<vector<double>> cs = vector<vector<double>>{
            vector<double>{31.0 / 162.0, 14.0 / 162.0, 14.0 / 162.0,
                           -5.0 / 162.0},
            vector<double>{16.0 / 81.0, 20.0 / 81.0, 20.0 / 81.0, -2.0 / 81.0},
            vector<double>{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0}};
        // k0 ~ k3
        for (int i = 0; i < 4; i++) {
            if (i == 0)
                (*this)(v, k[i], 0, beta);
            else {
                MatrixFunctions::copy(r[0], v);
                tf->opf->seq->iadd(r[0], k[i - 1], ks[i], 1.0);
                if (tf->opf->seq->mode != SeqTypes::None)
                    tf->opf->seq->simple_perform();
                (*this)(r[0], k[i], 0, beta);
            }
        }
        // r0 ~ r2
        for (int i = 0; i < 3; i++) {
            MatrixFunctions::copy(r[i], v);
            double factor = exp(beta * (i + 1) / 3 * const_e);
            for (size_t j = 0; j < 4; j++) {
                tf->opf->seq->iadd(r[i], k[j], cs[i][j] * factor, factor);
                if (tf->opf->seq->mode != SeqTypes::None)
                    tf->opf->seq->simple_perform();
            }
        }
        double norm = MatrixFunctions::norm(r[2]);
        double energy = -const_e;
        if (eval_energy) {
            k[0].clear();
            (*this)(r[2], k[0]);
            energy = MatrixFunctions::dot(r[2], k[0]) / (norm * norm);
        }
        for (int i = 3; i >= 0; i--)
            k[i].deallocate();
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_pair(
            r, make_tuple(energy, norm, 4 + eval_energy, nflop, t.get_time()));
    }
    // energy, norm, nexpo, nflop, texpo
    tuple<double, double, int, size_t, double>
    expo_apply(double beta, double const_e, bool iprint = false) {
        assert(compute_diag);
        double anorm =
            MatrixFunctions::norm(MatrixRef(diag->data, diag->total_memory, 1));
        MatrixRef v(ket->data, ket->total_memory, 1);
        Timer t;
        t.get_time();
        int nexpo = tf->opf->seq->mode == SeqTypes::Auto
                        ? MatrixFunctions::expo_apply(*tf->opf->seq, beta,
                                                      anorm, v, const_e, iprint)
                        : MatrixFunctions::expo_apply(*this, beta, anorm, v,
                                                      const_e, iprint);
        double norm = MatrixFunctions::norm(v);
        MatrixRef tmp(nullptr, ket->total_memory, 1);
        tmp.allocate();
        tmp.clear();
        if (tf->opf->seq->mode == SeqTypes::Auto)
            (*tf->opf->seq)(v, tmp);
        else
            (*this)(v, tmp);
        double energy = MatrixFunctions::dot(v, tmp) / (norm * norm);
        tmp.deallocate();
        size_t nflop = tf->opf->seq->cumulative_nflop;
        tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, nflop, t.get_time());
    }
    void deallocate() {
        frame->activate(0);
        if (tf->opf->seq->mode == SeqTypes::Auto) {
            tf->opf->seq->deallocate();
            tf->opf->seq->clear();
        }
        cmat->info->cinfo->deallocate();
        if (compute_diag)
            diag->deallocate();
        op->deallocate();
        for (int i = right_op_infos.size() - 1; i >= 0; i--) {
            if (right_op_infos[i].second->cinfo != nullptr)
                right_op_infos[i].second->cinfo->deallocate();
            right_op_infos[i].second->deallocate();
        }
        for (int i = left_op_infos.size() - 1; i >= 0; i--) {
            if (left_op_infos[i].second->cinfo != nullptr)
                left_op_infos[i].second->cinfo->deallocate();
            left_op_infos[i].second->deallocate();
        }
    }
};

enum FuseTypes : uint8_t { NoFuse = 0, FuseL = 1, FuseR = 2, FuseLR = 3 };

template <typename S> struct MovingEnvironment {
    int n_sites, center, dot;
    shared_ptr<MPO<S>> mpo;
    shared_ptr<MPS<S>> bra, ket;
    vector<shared_ptr<Partition<S>>> envs;
    shared_ptr<SymbolicColumnVector<S>> hop_mat;
    string tag;
    MovingEnvironment(const shared_ptr<MPO<S>> &mpo,
                      const shared_ptr<MPS<S>> &bra,
                      const shared_ptr<MPS<S>> &ket, const string &tag = "DMRG")
        : n_sites(ket->n_sites), center(ket->center), dot(ket->dot), mpo(mpo),
          bra(bra), ket(ket), tag(tag) {
        assert(bra->n_sites == ket->n_sites && mpo->n_sites == ket->n_sites);
        assert(bra->center == ket->center && bra->dot == ket->dot);
        hop_mat = make_shared<SymbolicColumnVector<S>>(1);
        (*hop_mat)[0] = mpo->op;
    }
    void left_contract_rotate(int i) {
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos_notrunc;
        vector<shared_ptr<Symbolic<S>>> mats = {
            mpo->left_operator_names[i - 1]};
        if (mpo->schemer != nullptr && i - 1 == mpo->schemer->left_trans_site)
            mats.push_back(mpo->schemer->left_new_operator_names);
        vector<S> sl = Partition<S>::get_uniq_labels(mats);
        shared_ptr<Symbolic<S>> exprs =
            envs[i - 1]->left == nullptr
                ? nullptr
                : (mpo->left_operator_exprs.size() != 0
                       ? mpo->left_operator_exprs[i - 1]
                       : envs[i - 1]->left->lmat *
                             envs[i - 1]->middle.front()->lmat);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S>::get_uniq_sub_labels(
                exprs, mpo->left_operator_names[i - 1], sl);
        Partition<S>::init_left_op_infos_notrunc(
            i - 1, bra->info, ket->info, sl, subsl, envs[i - 1]->left_op_infos,
            mpo->site_op_infos[bra->info->orbsym[i - 1]], left_op_infos_notrunc,
            mpo->tf->opf->cg);
        frame->activate(0);
        shared_ptr<OperatorTensor<S>> new_left = Partition<S>::build_left(
            {mpo->left_operator_names[i - 1]}, left_op_infos_notrunc);
        mpo->tf->left_contract(envs[i - 1]->left, envs[i - 1]->middle.front(),
                               new_left,
                               mpo->left_operator_exprs.size() != 0
                                   ? mpo->left_operator_exprs[i - 1]
                                   : nullptr);
        bra->load_tensor(i - 1);
        if (bra != ket)
            ket->load_tensor(i - 1);
        frame->reset(1);
        Partition<S>::init_left_op_infos(i - 1, bra->info, ket->info, sl,
                                         envs[i]->left_op_infos);
        frame->activate(1);
        envs[i]->left = Partition<S>::build_left(mats, envs[i]->left_op_infos);
        mpo->tf->left_rotate(new_left, bra->tensors[i - 1], ket->tensors[i - 1],
                             envs[i]->left);
        if (mpo->schemer != nullptr && i - 1 == mpo->schemer->left_trans_site)
            mpo->tf->numerical_transform(envs[i]->left, mats[1],
                                         mpo->schemer->left_new_operator_exprs);
        frame->activate(0);
        if (bra != ket)
            ket->unload_tensor(i - 1);
        bra->unload_tensor(i - 1);
        new_left->deallocate();
        Partition<S>::deallocate_op_infos_notrunc(left_op_infos_notrunc);
        frame->save_data(1, get_left_partition_filename(i));
    }
    void right_contract_rotate(int i) {
        vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> right_op_infos_notrunc;
        vector<shared_ptr<Symbolic<S>>> mats = {
            mpo->right_operator_names[i + dot]};
        if (mpo->schemer != nullptr &&
            i + dot == mpo->schemer->right_trans_site)
            mats.push_back(mpo->schemer->right_new_operator_names);
        vector<S> sl = Partition<S>::get_uniq_labels(mats);
        shared_ptr<Symbolic<S>> exprs =
            envs[i + 1]->right == nullptr
                ? nullptr
                : (mpo->right_operator_exprs.size() != 0
                       ? mpo->right_operator_exprs[i + dot]
                       : envs[i + 1]->middle.back()->rmat *
                             envs[i + 1]->right->rmat);
        vector<vector<pair<uint8_t, S>>> subsl =
            Partition<S>::get_uniq_sub_labels(
                exprs, mpo->right_operator_names[i + dot], sl);
        Partition<S>::init_right_op_infos_notrunc(
            i + dot, bra->info, ket->info, sl, subsl,
            envs[i + 1]->right_op_infos,
            mpo->site_op_infos[bra->info->orbsym[i + dot]],
            right_op_infos_notrunc, mpo->tf->opf->cg);
        frame->activate(0);
        shared_ptr<OperatorTensor<S>> new_right = Partition<S>::build_right(
            {mpo->right_operator_names[i + dot]}, right_op_infos_notrunc);
        mpo->tf->right_contract(envs[i + 1]->right, envs[i + 1]->middle.back(),
                                new_right,
                                mpo->right_operator_exprs.size() != 0
                                    ? mpo->right_operator_exprs[i + dot]
                                    : nullptr);
        bra->load_tensor(i + dot);
        if (bra != ket)
            ket->load_tensor(i + dot);
        frame->reset(1);
        Partition<S>::init_right_op_infos(i + dot, bra->info, ket->info, sl,
                                          envs[i]->right_op_infos);
        frame->activate(1);
        envs[i]->right =
            Partition<S>::build_right(mats, envs[i]->right_op_infos);
        mpo->tf->right_rotate(new_right, bra->tensors[i + dot],
                              ket->tensors[i + dot], envs[i]->right);
        if (mpo->schemer != nullptr &&
            i + dot == mpo->schemer->right_trans_site)
            mpo->tf->numerical_transform(
                envs[i]->right, mats[1],
                mpo->schemer->right_new_operator_exprs);
        frame->activate(0);
        if (bra != ket)
            ket->unload_tensor(i + dot);
        bra->unload_tensor(i + dot);
        new_right->deallocate();
        Partition<S>::deallocate_op_infos_notrunc(right_op_infos_notrunc);
        frame->save_data(1, get_right_partition_filename(i));
    }
    string get_left_partition_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".PART." << tag
           << ".LEFT." << Parsing::to_string(i);
        return ss.str();
    }
    string get_right_partition_filename(int i) const {
        stringstream ss;
        ss << frame->save_dir << "/" << frame->prefix << ".PART." << tag
           << ".RIGHT." << Parsing::to_string(i);
        return ss.str();
    }
    void init_environments(bool iprint = false) {
        envs.clear();
        envs.resize(n_sites);
        for (int i = 0; i < n_sites; i++) {
            envs[i] =
                make_shared<Partition<S>>(nullptr, nullptr, mpo->tensors[i]);
            if (i != n_sites - 1 && dot == 2)
                envs[i]->middle.push_back(mpo->tensors[i + 1]);
        }
        for (int i = 1; i <= center; i++) {
            if (iprint)
                cout << "init .. L = " << i << endl;
            left_contract_rotate(i);
        }
        for (int i = n_sites - dot - 1; i >= center; i--) {
            if (iprint)
                cout << "init .. R = " << i << endl;
            right_contract_rotate(i);
        }
        frame->reset(1);
    }
    void prepare() {
        for (int i = n_sites - 1; i > center; i--) {
            envs[i]->left_op_infos.clear();
            envs[i]->left = nullptr;
        }
        for (int i = 0; i < center; i++) {
            envs[i]->right_op_infos.clear();
            envs[i]->right = nullptr;
        }
    }
    void move_to(int i) {
        if (i > center) {
            frame->load_data(1, get_left_partition_filename(center));
            left_contract_rotate(++center);
        } else if (i < center) {
            frame->load_data(1, get_right_partition_filename(center));
            right_contract_rotate(--center);
        }
        bra->center = ket->center = center;
    }
    shared_ptr<EffectiveHamiltonian<S>> eff_ham(FuseTypes fuse_type,
                                                bool compute_diag) {
        if (dot == 2) {
            vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> left_op_infos,
                right_op_infos;
            shared_ptr<OperatorTensor<S>> new_left, new_right;
            int iL = -1, iR = -1, iM = -1;
            if (fuse_type == FuseTypes::FuseLR)
                iL = center, iR = center + 1, iM = center;
            else if (fuse_type == FuseTypes::FuseR)
                iL = center, iR = center, iM = center - 1;
            else if (fuse_type == FuseTypes::FuseL)
                iL = center + 1, iR = center + 1, iM = center + 1;
            else
                assert(false);
            if (fuse_type & FuseTypes::FuseL) {
                // left contract infos
                vector<shared_ptr<Symbolic<S>>> lmats = {
                    mpo->left_operator_names[iL]};
                if (mpo->schemer != nullptr &&
                    iL == mpo->schemer->left_trans_site &&
                    mpo->schemer->right_trans_site -
                            mpo->schemer->left_trans_site <=
                        1)
                    lmats.push_back(mpo->schemer->left_new_operator_names);
                vector<S> lsl = Partition<S>::get_uniq_labels(lmats);
                shared_ptr<Symbolic<S>> lexprs =
                    envs[iL]->left == nullptr
                        ? nullptr
                        : (mpo->left_operator_exprs.size() != 0
                               ? mpo->left_operator_exprs[iL]
                               : envs[iL]->left->lmat *
                                     envs[iL]->middle.front()->lmat);
                vector<vector<pair<uint8_t, S>>> lsubsl =
                    Partition<S>::get_uniq_sub_labels(
                        lexprs, mpo->left_operator_names[iL], lsl);
                if (envs[iL]->left != nullptr)
                    frame->load_data(1, get_left_partition_filename(iL));
                Partition<S>::init_left_op_infos_notrunc(
                    iL, bra->info, ket->info, lsl, lsubsl,
                    envs[iL]->left_op_infos,
                    mpo->site_op_infos[bra->info->orbsym[iL]], left_op_infos,
                    mpo->tf->opf->cg);
                // left contract
                frame->activate(0);
                new_left = Partition<S>::build_left(lmats, left_op_infos);
                mpo->tf->left_contract(envs[iL]->left, envs[iL]->middle.front(),
                                       new_left,
                                       mpo->left_operator_exprs.size() != 0
                                           ? mpo->left_operator_exprs[iL]
                                           : nullptr);
                if (mpo->schemer != nullptr &&
                    iL == mpo->schemer->left_trans_site &&
                    mpo->schemer->right_trans_site -
                            mpo->schemer->left_trans_site <=
                        1)
                    mpo->tf->numerical_transform(
                        new_left, lmats[1],
                        mpo->schemer->left_new_operator_exprs);
            } else {
                assert(envs[iL]->left != nullptr);
                frame->load_data(1, get_left_partition_filename(iL));
                frame->activate(0);
                Partition<S>::copy_op_infos(envs[iL]->left_op_infos,
                                            left_op_infos);
                new_left = envs[iL]->left->deep_copy();
                for (auto &p : new_left->ops)
                    p.second->info = Partition<S>::find_op_info(
                        left_op_infos, p.second->info->delta_quantum);
            }
            if (fuse_type & FuseTypes::FuseR) {
                // right contract infos
                vector<shared_ptr<Symbolic<S>>> rmats = {
                    mpo->right_operator_names[iR]};
                vector<S> rsl = Partition<S>::get_uniq_labels(rmats);
                shared_ptr<Symbolic<S>> rexprs =
                    envs[iR - 1]->right == nullptr
                        ? nullptr
                        : (mpo->right_operator_exprs.size() != 0
                               ? mpo->right_operator_exprs[iR]
                               : envs[iR - 1]->middle.back()->rmat *
                                     envs[iR - 1]->right->rmat);
                vector<vector<pair<uint8_t, S>>> rsubsl =
                    Partition<S>::get_uniq_sub_labels(
                        rexprs, mpo->right_operator_names[iR], rsl);
                if (envs[iR - 1]->right != nullptr)
                    frame->load_data(1, get_right_partition_filename(iR - 1));
                Partition<S>::init_right_op_infos_notrunc(
                    iR, bra->info, ket->info, rsl, rsubsl,
                    envs[iR - 1]->right_op_infos,
                    mpo->site_op_infos[bra->info->orbsym[iR]], right_op_infos,
                    mpo->tf->opf->cg);
                // right contract
                frame->activate(0);
                new_right = Partition<S>::build_right(rmats, right_op_infos);
                mpo->tf->right_contract(envs[iR - 1]->right,
                                        envs[iR - 1]->middle.back(), new_right,
                                        mpo->right_operator_exprs.size() != 0
                                            ? mpo->right_operator_exprs[iR]
                                            : nullptr);
            } else {
                assert(envs[iR - 1]->right != nullptr);
                frame->load_data(1, get_right_partition_filename(iR - 1));
                frame->activate(0);
                Partition<S>::copy_op_infos(envs[iR - 1]->right_op_infos,
                                            right_op_infos);
                new_right = envs[iR - 1]->right->deep_copy();
                for (auto &p : new_right->ops)
                    p.second->info = Partition<S>::find_op_info(
                        right_op_infos, p.second->info->delta_quantum);
            }
            // delayed left-right contract
            shared_ptr<DelayedOperatorTensor<S>> op =
                mpo->middle_operator_exprs.size() != 0
                    ? TensorFunctions<S>::delayed_contract(
                          new_left, new_right, mpo->middle_operator_names[iM],
                          mpo->middle_operator_exprs[iM])
                    : TensorFunctions<S>::delayed_contract(new_left, new_right,
                                                           mpo->op);
            frame->activate(0);
            frame->reset(1);
            shared_ptr<SymbolicColumnVector<S>> hops =
                mpo->middle_operator_exprs.size() != 0
                    ? dynamic_pointer_cast<SymbolicColumnVector<S>>(
                          mpo->middle_operator_names[iM])
                    : hop_mat;
            shared_ptr<EffectiveHamiltonian<S>> efh =
                make_shared<EffectiveHamiltonian<S>>(
                    left_op_infos, right_op_infos, op, bra->tensors[iL],
                    ket->tensors[iL], mpo->op, hops, mpo->tf, compute_diag);
            return efh;
        } else
            return nullptr;
    }
    static void contract_two_dot(int i, const shared_ptr<MPS<S>> &mps,
                                 bool reduced = false) {
        shared_ptr<SparseMatrix<S>> old_wfn = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrixInfo<S>> old_wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        frame->activate(1);
        mps->load_tensor(i);
        mps->load_tensor(i + 1);
        frame->activate(0);
        if (reduced)
            old_wfn_info->initialize_contract(mps->tensors[i]->info,
                                              mps->tensors[i + 1]->info);
        else {
            frame->activate(1);
            mps->info->load_left_dims(i);
            mps->info->load_right_dims(i + 2);
            StateInfo<S> l = mps->info->left_dims[i],
                         ml = mps->info->basis[mps->info->orbsym[i]],
                         mr = mps->info->basis[mps->info->orbsym[i + 1]],
                         r = mps->info->right_dims[i + 2];
            StateInfo<S> ll = StateInfo<S>::tensor_product(
                l, ml, mps->info->left_dims_fci[i + 1]);
            StateInfo<S> rr = StateInfo<S>::tensor_product(
                mr, r, mps->info->right_dims_fci[i + 1]);
            frame->activate(0);
            old_wfn_info->initialize(ll, rr, mps->info->target, false, true);
            frame->activate(1);
            rr.deallocate();
            ll.deallocate();
            r.deallocate();
            l.deallocate();
            frame->activate(0);
        }
        frame->activate(0);
        old_wfn->allocate(old_wfn_info);
        old_wfn->contract(mps->tensors[i], mps->tensors[i + 1]);
        frame->activate(1);
        mps->unload_tensor(i + 1);
        mps->unload_tensor(i);
        frame->activate(0);
        mps->tensors[i] = old_wfn;
        mps->tensors[i + 1] = nullptr;
    }
    static shared_ptr<SparseMatrix<S>>
    density_matrix(S opdq, const shared_ptr<SparseMatrix<S>> &psi,
                   bool trace_right, double noise) {
        shared_ptr<SparseMatrixInfo<S>> dm_info =
            make_shared<SparseMatrixInfo<S>>();
        dm_info->initialize_dm(psi->info, opdq, trace_right);
        shared_ptr<SparseMatrix<S>> dm = make_shared<SparseMatrix<S>>();
        dm->allocate(dm_info);
        OperatorFunctions<S>::trans_product(*psi, *dm, trace_right, noise);
        return dm;
    }
    static shared_ptr<SparseMatrix<S>>
    density_matrix_with_weights(S opdq, const shared_ptr<SparseMatrix<S>> &psi,
                                bool trace_right, double noise,
                                const vector<MatrixRef> &mats,
                                const vector<double> &weights) {
        double factor = psi->factor, *ptr = psi->data;
        assert(mats.size() == weights.size() - 1);
        psi->factor = factor * sqrt(weights[0]);
        shared_ptr<SparseMatrix<S>> dm =
            density_matrix(opdq, psi, trace_right, noise);
        for (size_t i = 1; i < weights.size(); i++) {
            psi->factor = factor * sqrt(weights[i]);
            psi->data = mats[i - 1].data;
            OperatorFunctions<S>::trans_product(*psi, *dm, trace_right, 0.0);
        }
        psi->data = ptr, psi->factor = factor;
        return dm;
    }
    static double split_density_matrix(const shared_ptr<SparseMatrix<S>> &dm,
                                       const shared_ptr<SparseMatrix<S>> &wfn,
                                       int k, bool trace_right,
                                       shared_ptr<SparseMatrix<S>> &left,
                                       shared_ptr<SparseMatrix<S>> &right) {
        vector<DiagonalMatrix> eigen_values;
        vector<MatrixRef> eigen_values_reduced;
        int k_total = 0;
        for (int i = 0; i < dm->info->n; i++) {
            DiagonalMatrix w(nullptr, dm->info->n_states_bra[i]);
            w.allocate();
            MatrixFunctions::eigs((*dm)[i], w);
            MatrixRef wr(nullptr, w.n, 1);
            wr.allocate();
            MatrixFunctions::copy(wr, MatrixRef(w.data, w.n, 1));
            MatrixFunctions::iscale(wr,
                                    1.0 / dm->info->quanta[i].multiplicity());
            eigen_values.push_back(w);
            eigen_values_reduced.push_back(wr);
            k_total += w.n;
        }
        shared_ptr<SparseMatrixInfo<S>> linfo =
            make_shared<SparseMatrixInfo<S>>();
        shared_ptr<SparseMatrixInfo<S>> rinfo =
            make_shared<SparseMatrixInfo<S>>();
        double error = 0.0;
        vector<pair<int, int>> ss;
        ss.reserve(k_total);
        for (int i = 0; i < (int)eigen_values.size(); i++)
            for (int j = 0; j < eigen_values[i].n; j++)
                ss.push_back(make_pair(i, j));
        if (k != -1 && k_total > k) {
            sort(ss.begin(), ss.end(),
                 [&eigen_values_reduced](const pair<int, int> &a,
                                         const pair<int, int> &b) {
                     return eigen_values_reduced[a.first].data[a.second] >
                            eigen_values_reduced[b.first].data[b.second];
                 });
            for (int i = k; i < k_total; i++)
                error += eigen_values[ss[i].first].data[ss[i].second];
            ss.resize(k);
            sort(ss.begin(), ss.end(),
                 [](const pair<int, int> &a, const pair<int, int> &b) {
                     return a.first != b.first ? a.first < b.first
                                               : a.second < b.second;
                 });
        }
        for (int i = dm->info->n - 1; i >= 0; i--) {
            eigen_values_reduced[i].deallocate();
            eigen_values[i].deallocate();
        }
        vector<uint16_t> ilr, im;
        ilr.reserve(ss.size());
        im.reserve(ss.size());
        if (k != 0)
            ilr.push_back(ss[0].first), im.push_back(1);
        for (int i = 1; i < (int)ss.size(); i++)
            if (ss[i].first != ilr.back())
                ilr.push_back(ss[i].first), im.push_back(1);
            else
                ++im.back();
        int kk = ilr.size();
        linfo->is_fermion = rinfo->is_fermion = false;
        linfo->is_wavefunction = !trace_right;
        rinfo->is_wavefunction = trace_right;
        linfo->delta_quantum =
            trace_right ? dm->info->delta_quantum : wfn->info->delta_quantum;
        rinfo->delta_quantum =
            trace_right ? wfn->info->delta_quantum : dm->info->delta_quantum;
        linfo->allocate(kk);
        rinfo->allocate(kk);
        uint16_t idx_dm_to_wfn[dm->info->n];
        if (trace_right) {
            for (int i = 0; i < wfn->info->n; i++) {
                S pb = wfn->info->quanta[i].get_bra(wfn->info->delta_quantum);
                idx_dm_to_wfn[dm->info->find_state(pb)] = i;
            }
            for (int i = 0; i < kk; i++) {
                linfo->quanta[i] = dm->info->quanta[ilr[i]];
                rinfo->quanta[i] = wfn->info->quanta[idx_dm_to_wfn[ilr[i]]];
                linfo->n_states_bra[i] = dm->info->n_states_bra[ilr[i]];
                linfo->n_states_ket[i] = im[i];
                rinfo->n_states_bra[i] = im[i];
                rinfo->n_states_ket[i] =
                    wfn->info->n_states_ket[idx_dm_to_wfn[ilr[i]]];
            }
            linfo->n_states_total[0] = 0;
            for (int i = 0; i < kk - 1; i++)
                linfo->n_states_total[i + 1] =
                    linfo->n_states_total[i] +
                    (uint32_t)linfo->n_states_bra[i] * linfo->n_states_ket[i];
            rinfo->sort_states();
        } else {
            for (int i = 0; i < wfn->info->n; i++) {
                S pk = -wfn->info->quanta[i].get_ket();
                idx_dm_to_wfn[dm->info->find_state(pk)] = i;
            }
            for (int i = 0; i < kk; i++) {
                linfo->quanta[i] = wfn->info->quanta[idx_dm_to_wfn[ilr[i]]];
                rinfo->quanta[i] = dm->info->quanta[ilr[i]];
                linfo->n_states_bra[i] =
                    wfn->info->n_states_bra[idx_dm_to_wfn[ilr[i]]];
                linfo->n_states_ket[i] = im[i];
                rinfo->n_states_bra[i] = im[i];
                rinfo->n_states_ket[i] = dm->info->n_states_ket[ilr[i]];
            }
            linfo->sort_states();
            rinfo->n_states_total[0] = 0;
            for (int i = 0; i < kk - 1; i++)
                rinfo->n_states_total[i + 1] =
                    rinfo->n_states_total[i] +
                    (uint32_t)rinfo->n_states_bra[i] * rinfo->n_states_ket[i];
        }
        left = make_shared<SparseMatrix<S>>();
        right = make_shared<SparseMatrix<S>>();
        left->allocate(linfo);
        right->allocate(rinfo);
        int iss = 0;
        if (trace_right)
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(left->data + linfo->n_states_total[i] + j,
                                  linfo->n_states_bra[i], 1),
                        MatrixRef(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0),
                            linfo->n_states_bra[i], 1),
                        linfo->n_states_ket[i], 1);
                int iw = idx_dm_to_wfn[ss[iss].first];
                int ir = right->info->find_state(wfn->info->quanta[iw]);
                assert(ir != -1);
                MatrixFunctions::multiply((*left)[i], true, (*wfn)[iw], false,
                                          (*right)[ir], 1.0, 0.0);
                iss += im[i];
            }
        else
            for (int i = 0; i < kk; i++) {
                for (int j = 0; j < im[i]; j++)
                    MatrixFunctions::copy(
                        MatrixRef(right->data + rinfo->n_states_total[i] +
                                      j * (*right)[i].n,
                                  1, (*right)[i].n),
                        MatrixRef(
                            &(*dm)[ss[iss + j].first](ss[iss + j].second, 0), 1,
                            (*right)[i].n));
                int iw = idx_dm_to_wfn[ss[iss].first];
                int il = left->info->find_state(wfn->info->quanta[iw]);
                assert(il != -1);
                MatrixFunctions::multiply((*wfn)[iw], false, (*right)[i], true,
                                          (*left)[il], 1.0, 0.0);
                iss += im[i];
            }
        assert(iss == ss.size());
        return error;
    }
    static void propagate_wfn(int i, int n_sites, const shared_ptr<MPS<S>> &mps,
                              bool forward) {
        shared_ptr<MPSInfo<S>> mps_info = mps->info;
        StateInfo<S> l, m, r, lm, lmc, mr, mrc, p;
        shared_ptr<SparseMatrixInfo<S>> wfn_info =
            make_shared<SparseMatrixInfo<S>>();
        shared_ptr<SparseMatrix<S>> wfn = make_shared<SparseMatrix<S>>();
        bool swapped = false;
        if (forward) {
            if ((swapped = i + 1 != n_sites - 1)) {
                mps_info->load_left_dims(i + 1);
                mps_info->load_right_dims(i + 2);
                l = mps_info->left_dims[i + 1],
                m = mps_info->basis[mps_info->orbsym[i + 1]],
                r = mps_info->right_dims[i + 2];
                lm = StateInfo<S>::tensor_product(
                    l, m, mps_info->left_dims_fci[i + 2]);
                lmc = StateInfo<S>::get_connection_info(l, m, lm);
                mr = StateInfo<S>::tensor_product(
                    m, r, mps_info->right_dims_fci[i + 1]);
                mrc = StateInfo<S>::get_connection_info(m, r, mr);
                shared_ptr<SparseMatrixInfo<S>> owinfo =
                    mps->tensors[i + 1]->info;
                wfn_info->initialize(lm, r, owinfo->delta_quantum,
                                     owinfo->is_fermion,
                                     owinfo->is_wavefunction);
                wfn->allocate(wfn_info);
                mps->load_tensor(i + 1);
                wfn->swap_to_fused_left(mps->tensors[i + 1], l, m, r, mr, mrc,
                                        lm, lmc);
                mps->unload_tensor(i + 1);
                mps->tensors[i + 1] = wfn;
                mps->save_tensor(i + 1);
            }
        } else {
            if ((swapped = i != 0)) {
                mps_info->load_left_dims(i);
                mps_info->load_right_dims(i + 1);
                l = mps_info->left_dims[i],
                m = mps_info->basis[mps_info->orbsym[i]],
                r = mps_info->right_dims[i + 1];
                lm = StateInfo<S>::tensor_product(
                    l, m, mps_info->left_dims_fci[i + 1]);
                lmc = StateInfo<S>::get_connection_info(l, m, lm);
                mr = StateInfo<S>::tensor_product(m, r,
                                                  mps_info->right_dims_fci[i]);
                mrc = StateInfo<S>::get_connection_info(m, r, mr);
                shared_ptr<SparseMatrixInfo<S>> owinfo = mps->tensors[i]->info;
                wfn_info->initialize(l, mr, owinfo->delta_quantum,
                                     owinfo->is_fermion,
                                     owinfo->is_wavefunction);
                wfn->allocate(wfn_info);
                mps->load_tensor(i);
                wfn->swap_to_fused_right(mps->tensors[i], l, m, r, lm, lmc, mr,
                                         mrc);
                mps->unload_tensor(i);
                mps->tensors[i] = wfn;
                mps->save_tensor(i);
            }
        }
        if (swapped) {
            wfn->deallocate();
            wfn_info->deallocate();
            mrc.deallocate();
            mr.deallocate();
            lmc.deallocate();
            lm.deallocate();
            r.deallocate();
            l.deallocate();
        }
    }
};

template <typename S> struct DMRG {
    shared_ptr<MovingEnvironment<S>> me;
    vector<uint16_t> bond_dims;
    vector<double> noises;
    vector<double> energies;
    bool forward;
    DMRG(const shared_ptr<MovingEnvironment<S>> &me,
         const vector<uint16_t> &bond_dims, const vector<double> &noises)
        : me(me), bond_dims(bond_dims), noises(noises), forward(false) {}
    struct Iteration {
        double energy, error;
        int ndav;
        double tdav;
        size_t nflop;
        Iteration(double energy, double error, int ndav, size_t nflop = 0,
                  double tdav = 1.0)
            : energy(energy), error(error), ndav(ndav), nflop(nflop),
              tdav(tdav) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Ndav = " << setw(4) << r.ndav << " E = " << setw(15)
               << r.energy << " Error = " << setw(15) << setprecision(12)
               << r.error << " FLOPS = " << scientific << setw(8)
               << setprecision(2) << (double)r.nflop / r.tdav
               << " Tdav = " << fixed << setprecision(2) << r.tdav;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, uint16_t bond_dim,
                             double noise) {
        frame->activate(0);
        if (me->ket->tensors[i] != nullptr &&
            me->ket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_two_dot(i, me->ket);
        else {
            me->ket->load_tensor(i);
            me->ket->tensors[i + 1] = nullptr;
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, true);
        auto pdi = h_eff->eigs(false);
        h_eff->deallocate();
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        shared_ptr<SparseMatrix<S>> dm = MovingEnvironment<S>::density_matrix(
            h_eff->opdq, h_eff->ket, forward, noise);
        double error = MovingEnvironment<S>::split_density_matrix(
            dm, h_eff->ket, (int)bond_dim, forward, me->ket->tensors[i],
            me->ket->tensors[i + 1]);
        shared_ptr<StateInfo<S>> info = nullptr;
        if (forward) {
            info = me->ket->tensors[i]->info->extract_state_info(forward);
            me->ket->info->left_dims[i + 1] = *info;
            me->ket->info->save_left_dims(i + 1);
            me->ket->canonical_form[i] = 'L';
            me->ket->canonical_form[i + 1] = 'C';
        } else {
            info = me->ket->tensors[i + 1]->info->extract_state_info(forward);
            me->ket->info->right_dims[i + 1] = *info;
            me->ket->info->save_right_dims(i + 1);
            me->ket->canonical_form[i] = 'C';
            me->ket->canonical_form[i + 1] = 'R';
        }
        info->deallocate();
        me->ket->save_tensor(i + 1);
        me->ket->save_tensor(i);
        me->ket->unload_tensor(i + 1);
        me->ket->unload_tensor(i);
        dm->info->deallocate();
        dm->deallocate();
        old_wfn->info->deallocate();
        old_wfn->deallocate();
        MovingEnvironment<S>::propagate_wfn(i, me->n_sites, me->ket, forward);
        return Iteration(get<0>(pdi) + me->mpo->const_e, error, get<1>(pdi),
                         get<2>(pdi), get<3>(pdi));
    }
    Iteration blocking(int i, bool forward, uint16_t bond_dim, double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, bond_dim, noise);
        else
            assert(false);
    }
    double sweep(bool forward, uint16_t bond_dim, double noise) {
        me->prepare();
        vector<double> energies;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            if (me->dot == 2)
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                     << " .. ";
            else
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " .. ";
            cout.flush();
            t.get_time();
            Iteration r = blocking(i, forward, bond_dim, noise);
            cout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << t.get_time() << endl;
            energies.push_back(r.energy);
        }
        return *min_element(energies.begin(), energies.end());
    }
    double solve(int n_sweeps, bool forward = true, double tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        energies.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            cout << "Sweep = " << setw(4) << iw << " | Direction = " << setw(8)
                 << (forward ? "forward" : "backward")
                 << " | Bond dimension = " << setw(4) << bond_dims[iw]
                 << " | Noise = " << scientific << setw(9) << setprecision(2)
                 << noises[iw] << endl;
            double energy = sweep(forward, bond_dims[iw], noises[iw]);
            energies.push_back(energy);
            bool converged = energies.size() >= 2 && tol > 0 &&
                             abs(energies[energies.size() - 1] -
                                 energies[energies.size() - 2]) < tol &&
                             noises[iw] == noises.back() &&
                             bond_dims[iw] == bond_dims.back();
            forward = !forward;
            current.get_time();
            cout << "Time elapsed = " << setw(10) << setprecision(2)
                 << current.current - start.current << endl;
            if (converged)
                break;
        }
        this->forward = forward;
        return energies.back();
    }
};

enum struct TETypes : uint8_t { TangentSpace, RK4 };

template <typename S> struct ImaginaryTE {
    shared_ptr<MovingEnvironment<S>> me;
    vector<uint16_t> bond_dims;
    vector<double> noises;
    vector<double> energies;
    vector<double> normsqs;
    bool forward;
    TETypes mode;
    ImaginaryTE(const shared_ptr<MovingEnvironment<S>> &me,
                const vector<uint16_t> &bond_dims,
                TETypes mode = TETypes::TangentSpace)
        : me(me), bond_dims(bond_dims), noises(vector<double>{0.0}),
          forward(false), mode(mode) {}
    struct Iteration {
        double energy, normsq, error;
        int nexpo, nexpok;
        double texpo;
        size_t nflop;
        Iteration(double energy, double normsq, double error, int nexpo,
                  int nexpok, size_t nflop = 0, double texpo = 1.0)
            : energy(energy), normsq(normsq), error(error), nexpo(nexpo),
              nexpok(nexpok), nflop(nflop), texpo(texpo) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << "Nexpo = " << setw(4) << r.nexpo << "/" << setw(4) << r.nexpok
               << " E = " << setw(15) << r.energy << " Error = " << setw(15)
               << setprecision(12) << r.error << " FLOPS = " << scientific
               << setw(8) << setprecision(2) << (double)r.nflop / r.texpo
               << " Texpo = " << fixed << setprecision(2) << r.texpo;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, double beta,
                             uint16_t bond_dim, double noise) {
        frame->activate(0);
        if (me->ket->tensors[i] != nullptr &&
            me->ket->tensors[i + 1] != nullptr)
            MovingEnvironment<S>::contract_two_dot(i, me->ket);
        else {
            me->ket->load_tensor(i);
            me->ket->tensors[i + 1] = nullptr;
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, true);
        tuple<double, double, int, size_t, double> pdi;
        shared_ptr<SparseMatrix<S>> old_wfn = me->ket->tensors[i];
        TETypes effective_mode = mode;
        if (mode == TETypes::RK4 &&
            ((forward && i + 1 == me->n_sites - 1) || (!forward && i == 0)))
            effective_mode = TETypes::TangentSpace;
        shared_ptr<SparseMatrix<S>> dm;
        if (effective_mode == TETypes::TangentSpace) {
            pdi = h_eff->expo_apply(-beta, me->mpo->const_e, false);
            h_eff->deallocate();
            dm = MovingEnvironment<S>::density_matrix(h_eff->opdq, h_eff->ket,
                                                      forward, noise);
        } else if (effective_mode == TETypes::RK4) {
            const vector<double> weights = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0,
                                            1.0 / 6.0};
            auto pdp = h_eff->rk4_apply(-beta, me->mpo->const_e, false);
            pdi = pdp.second;
            h_eff->deallocate();
            dm = MovingEnvironment<S>::density_matrix_with_weights(
                h_eff->opdq, h_eff->ket, forward, noise, pdp.first, weights);
            frame->activate(1);
            for (int i = pdp.first.size() - 1; i >= 0; i--)
                pdp.first[i].deallocate();
            frame->activate(0);
        }
        double error = MovingEnvironment<S>::split_density_matrix(
            dm, h_eff->ket, (int)bond_dim, forward, me->ket->tensors[i],
            me->ket->tensors[i + 1]);
        shared_ptr<StateInfo<S>> info = nullptr;
        if (forward) {
            info = me->ket->tensors[i]->info->extract_state_info(forward);
            me->ket->info->left_dims[i + 1] = *info;
            me->ket->info->save_left_dims(i + 1);
            me->ket->canonical_form[i] = 'L';
            me->ket->canonical_form[i + 1] = 'C';
        } else {
            info = me->ket->tensors[i + 1]->info->extract_state_info(forward);
            me->ket->info->right_dims[i + 1] = *info;
            me->ket->info->save_right_dims(i + 1);
            me->ket->canonical_form[i] = 'C';
            me->ket->canonical_form[i + 1] = 'R';
        }
        info->deallocate();
        me->ket->save_tensor(i + 1);
        me->ket->save_tensor(i);
        me->ket->unload_tensor(i + 1);
        me->ket->unload_tensor(i);
        dm->info->deallocate();
        dm->deallocate();
        old_wfn->info->deallocate();
        old_wfn->deallocate();
        int expok = 0;
        if (mode == TETypes::TangentSpace && forward &&
            i + 1 != me->n_sites - 1) {
            me->move_to(i + 1);
            me->ket->load_tensor(i + 1);
            shared_ptr<EffectiveHamiltonian<S>> k_eff =
                me->eff_ham(FuseTypes::FuseR, true);
            auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, false);
            k_eff->deallocate();
            me->ket->save_tensor(i + 1);
            me->ket->unload_tensor(i + 1);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        } else if (mode == TETypes::TangentSpace && !forward && i != 0) {
            me->move_to(i - 1);
            me->ket->load_tensor(i);
            shared_ptr<EffectiveHamiltonian<S>> k_eff =
                me->eff_ham(FuseTypes::FuseL, true);
            auto pdk = k_eff->expo_apply(beta, me->mpo->const_e, false);
            k_eff->deallocate();
            me->ket->save_tensor(i);
            me->ket->unload_tensor(i);
            get<3>(pdi) += get<3>(pdk), get<4>(pdi) += get<4>(pdk);
            expok = get<2>(pdk);
        }
        MovingEnvironment<S>::propagate_wfn(i, me->n_sites, me->ket, forward);
        return Iteration(get<0>(pdi) + me->mpo->const_e,
                         get<1>(pdi) * get<1>(pdi), error, get<2>(pdi), expok,
                         get<3>(pdi), get<4>(pdi));
    }
    Iteration blocking(int i, bool forward, double beta, uint16_t bond_dim,
                       double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, beta, bond_dim, noise);
        else
            assert(false);
    }
    pair<double, double> sweep(bool forward, double beta, uint16_t bond_dim,
                               double noise) {
        me->prepare();
        vector<double> energies, normsqs;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            if (me->dot == 2)
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                     << " .. ";
            else
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " .. ";
            cout.flush();
            t.get_time();
            Iteration r = blocking(i, forward, beta, bond_dim, noise);
            cout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << t.get_time() << endl;
            energies.push_back(r.energy);
            normsqs.push_back(r.normsq);
        }
        return make_pair(energies.back(), normsqs.back());
    }
    void normalize() {
        size_t center = me->ket->canonical_form.find('C');
        assert(center != string::npos);
        me->ket->load_tensor(center);
        me->ket->tensors[center]->factor /= sqrt(normsqs.back());
        me->ket->save_tensor(center);
        me->ket->unload_tensor(center);
    }
    double solve(int n_sweeps, double beta, bool forward = true,
                 double tol = 1E-6) {
        if (bond_dims.size() < n_sweeps)
            bond_dims.resize(n_sweeps, bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        energies.clear();
        normsqs.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            cout << "Sweep = " << setw(4) << iw << " | Direction = " << setw(8)
                 << (forward ? "forward" : "backward")
                 << " | Beta = " << setw(10) << setprecision(5)
                 << beta * (iw + 1) << " | Bond dimension = " << setw(4)
                 << bond_dims[iw] << " | Noise = " << scientific << setw(9)
                 << setprecision(2) << noises[iw] << endl;
            auto r = sweep(forward, beta, bond_dims[iw], noises[iw]);
            energies.push_back(r.first);
            normsqs.push_back(r.second);
            normalize();
            forward = !forward;
            current.get_time();
            cout << "Time elapsed = " << setw(10) << setprecision(2)
                 << current.current - start.current << endl;
        }
        this->forward = forward;
        return energies.back();
    }
};

template <typename S> struct Compress {
    shared_ptr<MovingEnvironment<S>> me;
    vector<uint16_t> bra_bond_dims, ket_bond_dims;
    vector<double> noises;
    vector<double> norms;
    bool forward;
    Compress(const shared_ptr<MovingEnvironment<S>> &me,
             const vector<uint16_t> &bra_bond_dims,
             const vector<uint16_t> &ket_bond_dims,
             const vector<double> &noises)
        : me(me), bra_bond_dims(bra_bond_dims), ket_bond_dims(ket_bond_dims),
          noises(noises), forward(false) {}
    struct Iteration {
        double norm, error;
        double tmult;
        size_t nflop;
        Iteration(double norm, double error, size_t nflop = 0,
                  double tmult = 1.0)
            : norm(norm), error(error), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            os << " Norm = " << setw(15) << r.norm << " Error = " << setw(15)
               << setprecision(12) << r.error << " FLOPS = " << scientific
               << setw(8) << setprecision(2) << (double)r.nflop / r.tmult
               << " Tmult = " << fixed << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, uint16_t bra_bond_dim,
                             uint16_t ket_bond_dim, double noise) {
        assert(me->bra != me->ket);
        frame->activate(0);
        for (auto &mps : {me->bra, me->ket}) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S>::contract_two_dot(i, mps, mps == me->ket);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, false);
        auto pdi = h_eff->multiply();
        h_eff->deallocate();
        shared_ptr<SparseMatrix<S>> old_bra = me->bra->tensors[i];
        shared_ptr<SparseMatrix<S>> old_ket = me->ket->tensors[i];
        double bra_error = 0.0;
        for (auto &mps : {me->bra, me->ket}) {
            shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
            shared_ptr<SparseMatrix<S>> dm =
                MovingEnvironment<S>::density_matrix(
                    h_eff->opdq, old_wfn, forward,
                    mps == me->bra ? noise : 0.0);
            int bond_dim =
                mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
            double error = MovingEnvironment<S>::split_density_matrix(
                dm, old_wfn, bond_dim, forward, mps->tensors[i],
                mps->tensors[i + 1]);
            if (mps == me->bra)
                bra_error = error;
            shared_ptr<StateInfo<S>> info = nullptr;
            if (forward) {
                info = mps->tensors[i]->info->extract_state_info(forward);
                mps->info->left_dims[i + 1] = *info;
                mps->info->save_left_dims(i + 1);
                mps->canonical_form[i] = 'L';
                mps->canonical_form[i + 1] = 'C';
            } else {
                info = mps->tensors[i + 1]->info->extract_state_info(forward);
                mps->info->right_dims[i + 1] = *info;
                mps->info->save_right_dims(i + 1);
                mps->canonical_form[i] = 'C';
                mps->canonical_form[i + 1] = 'R';
            }
            info->deallocate();
            mps->save_tensor(i + 1);
            mps->save_tensor(i);
            mps->unload_tensor(i + 1);
            mps->unload_tensor(i);
            dm->info->deallocate();
            dm->deallocate();
            MovingEnvironment<S>::propagate_wfn(i, me->n_sites, mps, forward);
        }
        for (auto &old_wfn : {old_ket, old_bra}) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        return Iteration(get<0>(pdi), bra_error, get<1>(pdi), get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, uint16_t bra_bond_dim,
                       uint16_t ket_bond_dim, double noise) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, bra_bond_dim, ket_bond_dim,
                                  noise);
        else
            assert(false);
    }
    double sweep(bool forward, uint16_t bra_bond_dim, uint16_t ket_bond_dim,
                 double noise) {
        me->prepare();
        vector<double> norms;
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            if (me->dot == 2)
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                     << " .. ";
            else
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " .. ";
            cout.flush();
            t.get_time();
            Iteration r =
                blocking(i, forward, bra_bond_dim, ket_bond_dim, noise);
            cout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << t.get_time() << endl;
            norms.push_back(r.norm);
        }
        return norms.back();
    }
    double solve(int n_sweeps, bool forward = true, double tol = 1E-6) {
        if (bra_bond_dims.size() < n_sweeps)
            bra_bond_dims.resize(n_sweeps, bra_bond_dims.back());
        if (ket_bond_dims.size() < n_sweeps)
            ket_bond_dims.resize(n_sweeps, ket_bond_dims.back());
        if (noises.size() < n_sweeps)
            noises.resize(n_sweeps, noises.back());
        Timer start, current;
        start.get_time();
        norms.clear();
        for (int iw = 0; iw < n_sweeps; iw++) {
            cout << "Sweep = " << setw(4) << iw << " | Direction = " << setw(8)
                 << (forward ? "forward" : "backward")
                 << " | BRA bond dimension = " << setw(4) << bra_bond_dims[iw]
                 << " | Noise = " << scientific << setw(9) << setprecision(2)
                 << noises[iw] << endl;
            double norm = sweep(forward, bra_bond_dims[iw], ket_bond_dims[iw],
                                noises[iw]);
            norms.push_back(norm);
            bool converged =
                norms.size() >= 2 && tol > 0 &&
                abs(norms[norms.size() - 1] - norms[norms.size() - 2]) < tol &&
                noises[iw] == noises.back() &&
                bra_bond_dims[iw] == bra_bond_dims.back();
            forward = !forward;
            current.get_time();
            cout << "Time elapsed = " << setw(10) << setprecision(2)
                 << current.current - start.current << endl;
            if (converged)
                break;
        }
        this->forward = forward;
        return norms.back();
    }
};

template <typename S> struct Expect {
    shared_ptr<MovingEnvironment<S>> me;
    uint16_t bra_bond_dim, ket_bond_dim;
    vector<vector<pair<shared_ptr<OpExpr<S>>, double>>> expectations;
    bool forward;
    Expect(const shared_ptr<MovingEnvironment<S>> &me, uint16_t bra_bond_dim,
           uint16_t ket_bond_dim)
        : me(me), bra_bond_dim(bra_bond_dim), ket_bond_dim(ket_bond_dim),
          forward(false) {
        expectations.resize(me->n_sites - me->dot + 1);
    }
    struct Iteration {
        vector<pair<shared_ptr<OpExpr<S>>, double>> expectations;
        double bra_error, ket_error;
        double tmult;
        size_t nflop;
        Iteration(
            const vector<pair<shared_ptr<OpExpr<S>>, double>> &expectations,
            double bra_error, double ket_error, size_t nflop = 0,
            double tmult = 1.0)
            : expectations(expectations), bra_error(bra_error),
              ket_error(ket_error), nflop(nflop), tmult(tmult) {}
        friend ostream &operator<<(ostream &os, const Iteration &r) {
            os << fixed << setprecision(8);
            if (r.expectations.size() == 1)
                os << " " << setw(14) << r.expectations[0].second;
            else
                os << " Nterms = " << setw(5) << r.expectations.size();
            os << " Error = " << setw(15) << setprecision(12) << r.bra_error
               << "/" << setw(15) << setprecision(12) << r.ket_error
               << " FLOPS = " << scientific << setw(8) << setprecision(2)
               << (double)r.nflop / r.tmult << " Tmult = " << fixed
               << setprecision(2) << r.tmult;
            return os;
        }
    };
    Iteration update_two_dot(int i, bool forward, bool propagate,
                             uint16_t bra_bond_dim, uint16_t ket_bond_dim) {
        frame->activate(0);
        vector<shared_ptr<MPS<S>>> mpss =
            me->bra == me->ket ? vector<shared_ptr<MPS<S>>>{me->bra}
                               : vector<shared_ptr<MPS<S>>>{me->bra, me->ket};
        for (auto &mps : mpss) {
            if (mps->tensors[i] != nullptr && mps->tensors[i + 1] != nullptr)
                MovingEnvironment<S>::contract_two_dot(i, mps, mps == me->ket);
            else {
                mps->load_tensor(i);
                mps->tensors[i + 1] = nullptr;
            }
        }
        shared_ptr<EffectiveHamiltonian<S>> h_eff =
            me->eff_ham(FuseTypes::FuseLR, false);
        auto pdi = h_eff->expect();
        h_eff->deallocate();
        vector<shared_ptr<SparseMatrix<S>>> old_wfns =
            me->bra == me->ket
                ? vector<shared_ptr<SparseMatrix<S>>>{me->bra->tensors[i]}
                : vector<shared_ptr<SparseMatrix<S>>>{me->ket->tensors[i],
                                                      me->bra->tensors[i]};
        double bra_error = 0.0, ket_error = 0.0;
        if (propagate) {
            for (auto &mps : mpss) {
                shared_ptr<SparseMatrix<S>> old_wfn = mps->tensors[i];
                shared_ptr<SparseMatrix<S>> dm =
                    MovingEnvironment<S>::density_matrix(h_eff->opdq, old_wfn,
                                                         forward, 0.0);
                int bond_dim =
                    mps == me->bra ? (int)bra_bond_dim : (int)ket_bond_dim;
                double error = MovingEnvironment<S>::split_density_matrix(
                    dm, old_wfn, bond_dim, forward, mps->tensors[i],
                    mps->tensors[i + 1]);
                if (mps == me->bra)
                    bra_error = error;
                else
                    ket_error = error;
                shared_ptr<StateInfo<S>> info = nullptr;
                if (forward) {
                    info = mps->tensors[i]->info->extract_state_info(forward);
                    mps->info->left_dims[i + 1] = *info;
                    mps->info->save_left_dims(i + 1);
                    mps->canonical_form[i] = 'L';
                    mps->canonical_form[i + 1] = 'C';
                } else {
                    info =
                        mps->tensors[i + 1]->info->extract_state_info(forward);
                    mps->info->right_dims[i + 1] = *info;
                    mps->info->save_right_dims(i + 1);
                    mps->canonical_form[i] = 'C';
                    mps->canonical_form[i + 1] = 'R';
                }
                info->deallocate();
                mps->save_tensor(i + 1);
                mps->save_tensor(i);
                mps->unload_tensor(i + 1);
                mps->unload_tensor(i);
                dm->info->deallocate();
                dm->deallocate();
                MovingEnvironment<S>::propagate_wfn(i, me->n_sites, mps,
                                                    forward);
            }
        }
        for (auto &old_wfn : old_wfns) {
            old_wfn->info->deallocate();
            old_wfn->deallocate();
        }
        return Iteration(get<0>(pdi), bra_error, ket_error, get<1>(pdi),
                         get<2>(pdi));
    }
    Iteration blocking(int i, bool forward, bool propagate,
                       uint16_t bra_bond_dim, uint16_t ket_bond_dim) {
        me->move_to(i);
        if (me->dot == 2)
            return update_two_dot(i, forward, propagate, bra_bond_dim,
                                  ket_bond_dim);
        else
            assert(false);
    }
    void sweep(bool forward, uint16_t bra_bond_dim, uint16_t ket_bond_dim) {
        me->prepare();
        vector<int> sweep_range;
        if (forward)
            for (int it = me->center; it < me->n_sites - me->dot + 1; it++)
                sweep_range.push_back(it);
        else
            for (int it = me->center; it >= 0; it--)
                sweep_range.push_back(it);

        Timer t;
        for (auto i : sweep_range) {
            if (me->dot == 2)
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << "-" << setw(4) << i + 1
                     << " .. ";
            else
                cout << " " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " .. ";
            cout.flush();
            t.get_time();
            Iteration r =
                blocking(i, forward, true, bra_bond_dim, ket_bond_dim);
            cout << r << " T = " << setw(4) << fixed << setprecision(2)
                 << t.get_time() << endl;
            expectations[i] = r.expectations;
        }
    }
    double solve(bool propagate, bool forward = true) {
        Timer start, current;
        start.get_time();
        for (auto &x : expectations)
            x.clear();
        if (propagate) {
            cout << "Expectation | Direction = " << setw(8)
                 << (forward ? "forward" : "backward")
                 << " | BRA bond dimension = " << setw(4) << bra_bond_dim
                 << " | KET bond dimension = " << setw(4) << ket_bond_dim
                 << endl;
            sweep(forward, bra_bond_dim, ket_bond_dim);
            forward = !forward;
            current.get_time();
            cout << "Time elapsed = " << setw(10) << setprecision(2)
                 << current.current - start.current << endl;
            this->forward = forward;
            return 0.0;
        } else {
            Iteration r = blocking(me->center, forward, false, bra_bond_dim,
                                   ket_bond_dim);
            assert(r.expectations.size() != 0);
            return r.expectations[0].second;
        }
    }
    MatrixRef get_1pdm_spatial(uint16_t n_physical_sites = 0U) {
        if (n_physical_sites == 0U)
            n_physical_sites = me->n_sites;
        MatrixRef r(nullptr, n_physical_sites, n_physical_sites);
        r.allocate();
        r.clear();
        for (auto &v : expectations)
            for (auto &x : v) {
                shared_ptr<OpElement<S>> op =
                    dynamic_pointer_cast<OpElement<S>>(x.first);
                assert(op->name == OpNames::PDM1);
                r(op->site_index[0], op->site_index[1]) = x.second;
            }
        return r;
    }
};

template <typename S> struct Hamiltonian {
    S vaccum, target;
    StateInfo<S> *basis;
    vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>> *site_op_infos;
    vector<pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>
        *site_norm_ops;
    uint8_t n_sites, n_syms;
    shared_ptr<OperatorFunctions<S>> opf;
    vector<uint8_t> orb_sym;
    Hamiltonian(S vaccum, S target, int n_sites, const vector<uint8_t> &orb_sym)
        : vaccum(vaccum), target(target), n_sites((uint8_t)n_sites),
          orb_sym(orb_sym) {
        assert((int)this->n_sites == n_sites);
        n_syms = *max_element(orb_sym.begin(), orb_sym.end()) + 1;
        basis = new StateInfo<S>[n_syms];
        site_op_infos =
            new vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>[n_syms];
        site_norm_ops = new vector<
            pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>[n_syms];
        opf = make_shared<OperatorFunctions<S>>(make_shared<CG<S>>(100));
        opf->cg->initialize();
    }
    virtual void get_site_ops(
        uint8_t m,
        map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
            &ops) const = 0;
    void filter_site_ops(uint8_t m, const vector<shared_ptr<Symbolic<S>>> &mats,
                         map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                             op_expr_less<S>> &ops) const {
        vector<shared_ptr<Symbolic<S>>> pmats = mats;
        if (pmats.size() == 2 && pmats[0] == pmats[1])
            pmats.resize(1);
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    ops[abs_value(x)] = nullptr;
                    break;
                case OpTypes::Sum:
                    for (auto &r : dynamic_pointer_cast<OpSum<S>>(x)->strings)
                        ops[abs_value((shared_ptr<OpExpr<S>>)r->get_op())] =
                            nullptr;
                    break;
                default:
                    assert(false);
                }
            }
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), vaccum);
        ops[i_op] = nullptr;
        get_site_ops(m, ops);
        shared_ptr<OpExpr<S>> zero = make_shared<OpExpr<S>>();
        size_t kk;
        for (auto pmat : pmats)
            for (auto &x : pmat->data) {
                shared_ptr<OpExpr<S>> xx;
                switch (x->get_type()) {
                case OpTypes::Zero:
                    break;
                case OpTypes::Elem:
                    xx = abs_value(x);
                    if (ops[xx]->factor == 0.0 || ops[xx]->info->n == 0)
                        x = zero;
                    break;
                case OpTypes::Sum:
                    kk = 0;
                    for (size_t i = 0;
                         i < dynamic_pointer_cast<OpSum<S>>(x)->strings.size();
                         i++) {
                        xx = abs_value((shared_ptr<OpExpr<S>>)
                                           dynamic_pointer_cast<OpSum<S>>(x)
                                               ->strings[i]
                                               ->get_op());
                        shared_ptr<SparseMatrix<S>> &mat = ops[xx];
                        if (!(mat->factor == 0.0 || mat->info->n == 0)) {
                            if (i != kk)
                                dynamic_pointer_cast<OpSum<S>>(x)->strings[kk] =
                                    dynamic_pointer_cast<OpSum<S>>(x)
                                        ->strings[i];
                            kk++;
                        }
                    }
                    if (kk == 0)
                        x = zero;
                    else if (kk !=
                             dynamic_pointer_cast<OpSum<S>>(x)->strings.size())
                        dynamic_pointer_cast<OpSum<S>>(x)->strings.resize(kk);
                    break;
                default:
                    assert(false);
                }
            }
        for (auto pmat : pmats)
            if (pmat->get_type() == SymTypes::Mat) {
                shared_ptr<SymbolicMatrix<S>> smat =
                    dynamic_pointer_cast<SymbolicMatrix<S>>(pmat);
                size_t j = 0;
                for (size_t i = 0; i < smat->indices.size(); i++)
                    if (smat->data[i]->get_type() != OpTypes::Zero) {
                        if (i != j)
                            smat->data[j] = smat->data[i],
                            smat->indices[j] = smat->indices[i];
                        j++;
                    }
                smat->data.resize(j);
                smat->indices.resize(j);
            }
        for (auto it = ops.cbegin(); it != ops.cend();) {
            if (it->second->factor == 0.0 || it->second->info->n == 0)
                ops.erase(it++);
            else
                it++;
        }
    }
    static bool cmp_site_norm_op(
        const pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &p,
        const shared_ptr<OpExpr<S>> &q) {
        return op_expr_less<S>()(p.first, q);
    }
    shared_ptr<SparseMatrixInfo<S>> find_site_op_info(S q,
                                                      uint8_t i_sym) const {
        auto p = lower_bound(site_op_infos[i_sym].begin(),
                             site_op_infos[i_sym].end(), q,
                             SparseMatrixInfo<S>::cmp_op_info);
        if (p == site_op_infos[i_sym].end() || p->first != q)
            return nullptr;
        else
            return p->second;
    }
    shared_ptr<SparseMatrix<S>>
    find_site_norm_op(const shared_ptr<OpExpr<S>> &q, uint8_t i_sym) const {
        auto p = lower_bound(site_norm_ops[i_sym].begin(),
                             site_norm_ops[i_sym].end(), q, cmp_site_norm_op);
        if (p == site_norm_ops[i_sym].end() || !(p->first == q))
            return nullptr;
        else
            return p->second;
    }
    virtual void deallocate() {
        opf->cg->deallocate();
        delete[] site_norm_ops;
        delete[] site_op_infos;
        delete[] basis;
    }
};

struct PointGroup {
    static uint8_t swap_d2h(uint8_t isym) {
        static uint8_t arr_swap[] = {8, 0, 7, 6, 1, 5, 2, 3, 4};
        return arr_swap[isym];
    }
};

template <typename, typename = void> struct HamiltonianQC;

template <typename S>
struct HamiltonianQC<S, typename S::is_sz_t> : Hamiltonian<S> {
    map<OpNames, shared_ptr<SparseMatrix<S>>> op_prims[4];
    shared_ptr<FCIDUMP> fcidump;
    double mu = 0;
    HamiltonianQC(S vaccum, S target, int n_sites,
                  const vector<uint8_t> &orb_sym,
                  const shared_ptr<FCIDUMP> &fcidump)
        : Hamiltonian<S>(vaccum, target, n_sites, orb_sym), fcidump(fcidump) {
        for (int i = 0; i < this->n_syms; i++) {
            this->basis[i].allocate(4);
            this->basis[i].quanta[0] = this->vaccum;
            this->basis[i].quanta[1] = S(1, -1, i);
            this->basis[i].quanta[2] = S(1, 1, i);
            this->basis[i].quanta[3] = S(2, 0, 0);
            this->basis[i].n_states[0] = this->basis[i].n_states[1] =
                this->basis[i].n_states[2] = this->basis[i].n_states[3] = 1;
            this->basis[i].sort_states();
        }
        init_site_ops();
    }
    void init_site_ops() {
        // site operator infos
        for (int i = 0; i < this->n_syms; i++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vaccum] = nullptr;
            for (int n = -1; n <= 1; n += 2)
                for (int s = -1; s <= 1; s += 2)
                    info[S(n, s, i)] = nullptr;
            for (int n = -2; n <= 2; n += 2)
                for (int s = -2; s <= 2; s += 2)
                    info[S(n, s, 0)] = nullptr;
            this->site_op_infos[i] =
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                 info.end());
            for (auto &p : this->site_op_infos[i]) {
                p.second = make_shared<SparseMatrixInfo<S>>();
                p.second->initialize(this->basis[i], this->basis[i], p.first,
                                     p.first.is_fermion());
            }
        }
        op_prims[0][OpNames::I] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::I]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::I])[S(0, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, -1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(2, 0, 0)](0, 0) = 1.0;
        const int sz[2] = {1, -1};
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s][OpNames::N] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::N]->allocate(
                this->find_site_op_info(S(0, 0, 0), 0));
            (*op_prims[s][OpNames::N])[S(0, 0, 0)](0, 0) = 0.0;
            (*op_prims[s][OpNames::N])[S(1, -1, 0)](0, 0) = s;
            (*op_prims[s][OpNames::N])[S(1, 1, 0)](0, 0) = 1 - s;
            (*op_prims[s][OpNames::N])[S(2, 0, 0)](0, 0) = 1.0;
            op_prims[s][OpNames::C] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::C]->allocate(
                this->find_site_op_info(S(1, sz[s], 0), 0));
            (*op_prims[s][OpNames::C])[S(0, 0, 0)](0, 0) = 1.0;
            (*op_prims[s][OpNames::C])[S(1, -sz[s], 0)](0, 0) = s ? -1.0 : 1.0;
            op_prims[s][OpNames::D] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::D]->allocate(
                this->find_site_op_info(S(-1, -sz[s], 0), 0));
            (*op_prims[s][OpNames::D])[S(1, sz[s], 0)](0, 0) = 1.0;
            (*op_prims[s][OpNames::D])[S(2, 0, 0)](0, 0) = s ? -1.0 : 1.0;
        }
        // low (&1): left index, high (>>1): right index
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t s = 0; s < 4; s++) {
            op_prims[s][OpNames::NN] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::NN]->allocate(
                this->find_site_op_info(S(0, 0, 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::N],
                               *op_prims[s >> 1][OpNames::N],
                               *op_prims[s][OpNames::NN]);
            op_prims[s][OpNames::A] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::A]->allocate(
                this->find_site_op_info(S(2, sz_plus[s], 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::C],
                               *op_prims[s >> 1][OpNames::C],
                               *op_prims[s][OpNames::A]);
            op_prims[s][OpNames::AD] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::AD]->allocate(
                this->find_site_op_info(S(-2, -sz_plus[s], 0), 0));
            this->opf->product(*op_prims[s >> 1][OpNames::D],
                               *op_prims[s & 1][OpNames::D],
                               *op_prims[s][OpNames::AD]);
            op_prims[s][OpNames::B] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::B]->allocate(
                this->find_site_op_info(S(0, sz_minus[s], 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::C],
                               *op_prims[s >> 1][OpNames::D],
                               *op_prims[s][OpNames::B]);
        }
        // low (&1): R index, high (>>1): B index
        for (uint8_t s = 0; s < 4; s++) {
            op_prims[s][OpNames::R] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::R]->allocate(
                this->find_site_op_info(S(-1, -sz[s & 1], 0), 0));
            this->opf->product(*op_prims[(s >> 1) | (s & 2)][OpNames::B],
                               *op_prims[s & 1][OpNames::D],
                               *op_prims[s][OpNames::R]);
            op_prims[s][OpNames::RD] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::RD]->allocate(
                this->find_site_op_info(S(1, sz[s & 1], 0), 0));
            this->opf->product(*op_prims[s & 1][OpNames::C],
                               *op_prims[(s >> 1) | (s & 2)][OpNames::B],
                               *op_prims[s][OpNames::RD]);
        }
        // site norm operators
        map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
            ops[this->n_syms];
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), this->vaccum);
        const shared_ptr<OpElement<S>> n_op[2] = {
            make_shared<OpElement<S>>(OpNames::N, SiteIndex({}, {0}),
                                      this->vaccum),
            make_shared<OpElement<S>>(OpNames::N, SiteIndex({}, {1}),
                                      this->vaccum)};
        const shared_ptr<OpElement<S>> nn_op[4] = {
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {0, 0}),
                                      this->vaccum),
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {1, 0}),
                                      this->vaccum),
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {0, 1}),
                                      this->vaccum),
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex({}, {1, 1}),
                                      this->vaccum)};
        for (uint8_t i = 0; i < this->n_syms; i++) {
            ops[i][i_op] = nullptr;
            for (uint8_t s = 0; s < 2; s++)
                ops[i][n_op[s]] = nullptr;
            for (uint8_t s = 0; s < 4; s++)
                ops[i][nn_op[s]] = nullptr;
        }
        for (uint8_t m = 0; m < this->n_sites; m++) {
            for (uint8_t s = 0; s < 2; s++) {
                ops[this->orb_sym[m]]
                   [make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], this->orb_sym[m]))] =
                       nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], this->orb_sym[m]))] = nullptr;
            }
            for (uint8_t s = 0; s < 4; s++) {
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::A,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(2, sz_plus[s], 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::AD,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(-2, -sz_plus[s], 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::B,
                    SiteIndex({m, m}, {(uint8_t)(s & 1), (uint8_t)(s >> 1)}),
                    S(0, sz_minus[s], 0))] = nullptr;
            }
        }
        for (uint8_t i = 0; i < this->n_syms; i++) {
            this->site_norm_ops[i] = vector<
                pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>(
                ops[i].begin(), ops[i].end());
            for (auto &p : this->site_norm_ops[i]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(this->find_site_op_info(op.q_label, i),
                                   op_prims[op.site_index.ss()][op.name]->data);
            }
        }
    }
    void get_site_ops(uint8_t m,
                      map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                          op_expr_less<S>> &ops) const override {
        uint8_t i, j, k, s;
        shared_ptr<SparseMatrix<S>> zero = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            switch (op.name) {
            case OpNames::I:
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
                p.second = this->find_site_norm_op(p.first, this->orb_sym[m]);
                break;
            case OpNames::H:
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(
                    this->find_site_op_info(op.q_label, this->orb_sym[m]));
                (*p.second)[S(0, 0, 0)](0, 0) = 0.0;
                (*p.second)[S(1, -1, this->orb_sym[m])](0, 0) = t(1, m, m);
                (*p.second)[S(1, 1, this->orb_sym[m])](0, 0) = t(0, m, m);
                (*p.second)[S(2, 0, 0)](0, 0) =
                    t(0, m, m) + t(1, m, m) +
                    0.5 * (v(0, 1, m, m, m, m) + v(1, 0, m, m, m, m));
                break;
            case OpNames::R:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[s].at(OpNames::D));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            *op_prims[s + (sp << 1)].at(OpNames::R));
                        tmp->factor = v(s, sp, i, m, m, m);
                        this->opf->iadd(*p.second, *tmp);
                        if (this->opf->seq->mode != SeqTypes::None)
                            this->opf->seq->simple_perform();
                    }
                    tmp->deallocate();
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                s = op.site_index.ss();
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(s, i, m)) < TINY &&
                     abs(v(s, 0, i, m, m, m)) < TINY &&
                     abs(v(s, 1, i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[s].at(OpNames::C));
                    p.second->factor *= t(s, i, m) * 0.5;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    for (uint8_t sp = 0; sp < 2; sp++) {
                        tmp->copy_data_from(
                            *op_prims[s + (sp << 1)].at(OpNames::RD));
                        tmp->factor = v(s, sp, i, m, m, m);
                        this->opf->iadd(*p.second, *tmp);
                        if (this->opf->seq->mode != SeqTypes::None)
                            this->opf->seq->simple_perform();
                    }
                    tmp->deallocate();
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::AD)->data);
                    p.second->factor *= v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.ss();
                if (abs(v(s & 1, s >> 1, i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::A)->data);
                    p.second->factor *= v(s & 1, s >> 1, i, m, k, m);
                }
                break;
            case OpNames::Q:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.ss();
                switch (s) {
                case 0U:
                case 3U:
                    if (abs(v(s & 1, s >> 1, i, m, m, j)) < TINY &&
                        abs(v(s & 1, 0, i, j, m, m)) < TINY &&
                        abs(v(s & 1, 1, i, j, m, m)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                            op.q_label, this->orb_sym[m]));
                        p.second->copy_data_from(
                            *op_prims[(s >> 1) | ((s & 1) << 1)].at(
                                OpNames::B));
                        p.second->factor *= -v(s & 1, s >> 1, i, m, m, j);
                        tmp->allocate(this->find_site_op_info(
                            op.q_label, this->orb_sym[m]));
                        for (uint8_t sp = 0; sp < 2; sp++) {
                            tmp->copy_data_from(
                                *op_prims[sp | (sp << 1)].at(OpNames::B));
                            tmp->factor = v(s & 1, sp, i, j, m, m);
                            this->opf->iadd(*p.second, *tmp);
                            if (this->opf->seq->mode != SeqTypes::None)
                                this->opf->seq->simple_perform();
                        }
                        tmp->deallocate();
                    }
                    break;
                case 1U:
                case 2U:
                    if (abs(v(s & 1, s >> 1, i, m, m, j)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                                               op.q_label, this->orb_sym[m]),
                                           op_prims[(s >> 1) | ((s & 1) << 1)]
                                               .at(OpNames::B)
                                               ->data);
                        p.second->factor *= -v(s & 1, s >> 1, i, m, m, j);
                    }
                    break;
                }
                break;
            default:
                assert(false);
            }
        }
    }
    void deallocate() override {
        for (int8_t s = 3; s >= 0; s--)
            for (auto name : vector<OpNames>{OpNames::RD, OpNames::R})
                op_prims[s][name]->deallocate();
        for (int8_t s = 3; s >= 0; s--)
            for (auto name : vector<OpNames>{OpNames::B, OpNames::AD,
                                             OpNames::A, OpNames::NN})
                op_prims[s][name]->deallocate();
        for (int8_t s = 1; s >= 0; s--)
            for (auto name :
                 vector<OpNames>{OpNames::D, OpNames::C, OpNames::N})
                op_prims[s][name]->deallocate();
        op_prims[0][OpNames::I]->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            for (int j = this->site_op_infos[i].size() - 1; j >= 0; j--)
                this->site_op_infos[i][j].second->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            this->basis[i].deallocate();
        Hamiltonian<S>::deallocate();
    }
    double v(uint8_t sl, uint8_t sr, uint8_t i, uint8_t j, uint8_t k,
             uint8_t l) const {
        return fcidump->v(sl, sr, i, j, k, l);
    }
    double t(uint8_t s, uint8_t i, uint8_t j) const {
        return i == j ? fcidump->t(s, i, i) - mu : fcidump->t(s, i, j);
    }
    double e() const { return fcidump->e; }
};

template <typename S>
struct HamiltonianQC<S, typename S::is_su2_t> : Hamiltonian<S> {
    map<OpNames, shared_ptr<SparseMatrix<S>>> op_prims[2];
    shared_ptr<FCIDUMP> fcidump;
    double mu = 0;
    HamiltonianQC(S vaccum, S target, int n_sites,
                  const vector<uint8_t> &orb_sym,
                  const shared_ptr<FCIDUMP> &fcidump)
        : Hamiltonian<S>(vaccum, target, n_sites, orb_sym), fcidump(fcidump) {
        assert(!fcidump->uhf);
        for (int i = 0; i < this->n_syms; i++) {
            this->basis[i].allocate(3);
            this->basis[i].quanta[0] = vaccum;
            this->basis[i].quanta[1] = S(1, 1, i);
            this->basis[i].quanta[2] = S(2, 0, 0);
            this->basis[i].n_states[0] = this->basis[i].n_states[1] =
                this->basis[i].n_states[2] = 1;
            this->basis[i].sort_states();
        }
        init_site_ops();
    }
    void init_site_ops() {
        // site operator infos
        for (int i = 0; i < this->n_syms; i++) {
            map<S, shared_ptr<SparseMatrixInfo<S>>> info;
            info[this->vaccum] = nullptr;
            info[S(1, 1, i)] = nullptr;
            info[S(-1, 1, i)] = nullptr;
            for (int n = -2; n <= 2; n += 2)
                for (int s = 0; s <= 2; s += 2)
                    info[S(n, s, 0)] = nullptr;
            this->site_op_infos[i] =
                vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(info.begin(),
                                                                 info.end());
            for (auto &p : this->site_op_infos[i]) {
                p.second = make_shared<SparseMatrixInfo<S>>();
                p.second->initialize(this->basis[i], this->basis[i], p.first,
                                     p.first.is_fermion());
            }
        }
        op_prims[0][OpNames::I] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::I]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::I])[S(0, 0, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(1, 1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::I])[S(2, 0, 0, 0)](0, 0) = 1.0;
        op_prims[0][OpNames::N] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::N]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::N])[S(0, 0, 0, 0)](0, 0) = 0.0;
        (*op_prims[0][OpNames::N])[S(1, 1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::N])[S(2, 0, 0, 0)](0, 0) = 2.0;
        op_prims[0][OpNames::NN] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::NN]->allocate(
            this->find_site_op_info(S(0, 0, 0), 0));
        (*op_prims[0][OpNames::NN])[S(0, 0, 0, 0)](0, 0) = 0.0;
        (*op_prims[0][OpNames::NN])[S(1, 1, 1, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::NN])[S(2, 0, 0, 0)](0, 0) = 4.0;
        op_prims[0][OpNames::C] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::C]->allocate(
            this->find_site_op_info(S(1, 1, 0), 0));
        (*op_prims[0][OpNames::C])[S(0, 1, 0, 0)](0, 0) = 1.0;
        (*op_prims[0][OpNames::C])[S(1, 0, 1, 0)](0, 0) = -sqrt(2);
        op_prims[0][OpNames::D] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::D]->allocate(
            this->find_site_op_info(S(-1, 1, 0), 0));
        (*op_prims[0][OpNames::D])[S(1, 0, 1, 0)](0, 0) = sqrt(2);
        (*op_prims[0][OpNames::D])[S(2, 1, 0, 0)](0, 0) = 1.0;
        for (uint8_t s = 0; s < 2; s++) {
            op_prims[s][OpNames::A] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::A]->allocate(
                this->find_site_op_info(S(2, s * 2, 0), 0));
            this->opf->product(*op_prims[0][OpNames::C],
                               *op_prims[0][OpNames::C],
                               *op_prims[s][OpNames::A]);
            op_prims[s][OpNames::AD] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::AD]->allocate(
                this->find_site_op_info(S(-2, s * 2, 0), 0));
            this->opf->product(*op_prims[0][OpNames::D],
                               *op_prims[0][OpNames::D],
                               *op_prims[s][OpNames::AD]);
            op_prims[s][OpNames::B] = make_shared<SparseMatrix<S>>();
            op_prims[s][OpNames::B]->allocate(
                this->find_site_op_info(S(0, s * 2, 0), 0));
            this->opf->product(*op_prims[0][OpNames::C],
                               *op_prims[0][OpNames::D],
                               *op_prims[s][OpNames::B]);
        }
        op_prims[0][OpNames::R] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::R]->allocate(
            this->find_site_op_info(S(-1, 1, 0), 0));
        this->opf->product(*op_prims[0][OpNames::B], *op_prims[0][OpNames::D],
                           *op_prims[0][OpNames::R]);
        op_prims[0][OpNames::RD] = make_shared<SparseMatrix<S>>();
        op_prims[0][OpNames::RD]->allocate(
            this->find_site_op_info(S(1, 1, 0), 0));
        this->opf->product(*op_prims[0][OpNames::C], *op_prims[0][OpNames::B],
                           *op_prims[0][OpNames::RD]);
        // site norm operators
        map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>, op_expr_less<S>>
            ops[this->n_syms];
        const shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), this->vaccum);
        const shared_ptr<OpElement<S>> n_op =
            make_shared<OpElement<S>>(OpNames::N, SiteIndex(), this->vaccum);
        const shared_ptr<OpElement<S>> nn_op =
            make_shared<OpElement<S>>(OpNames::NN, SiteIndex(), this->vaccum);
        for (uint8_t i = 0; i < this->n_syms; i++) {
            ops[i][i_op] = nullptr;
            ops[i][n_op] = nullptr;
            ops[i][nn_op] = nullptr;
        }
        for (uint8_t m = 0; m < this->n_sites; m++) {
            ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                OpNames::C, SiteIndex(m), S(1, 1, this->orb_sym[m]))] = nullptr;
            ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                OpNames::D, SiteIndex(m), S(-1, 1, this->orb_sym[m]))] =
                nullptr;
            for (uint8_t s = 0; s < 2; s++) {
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::A, SiteIndex(m, m, s), S(2, s * 2, 0))] = nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::AD, SiteIndex(m, m, s), S(-2, s * 2, 0))] =
                    nullptr;
                ops[this->orb_sym[m]][make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(m, m, s), S(0, s * 2, 0))] = nullptr;
            }
        }
        for (uint8_t i = 0; i < this->n_syms; i++) {
            this->site_norm_ops[i] = vector<
                pair<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>>>(
                ops[i].begin(), ops[i].end());
            for (auto &p : this->site_norm_ops[i]) {
                OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(this->find_site_op_info(op.q_label, i),
                                   op_prims[op.site_index.ss()][op.name]->data);
            }
        }
    }
    void get_site_ops(uint8_t m,
                      map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>,
                          op_expr_less<S>> &ops) const override {
        uint8_t i, j, k, s;
        shared_ptr<SparseMatrix<S>> zero = make_shared<SparseMatrix<S>>();
        shared_ptr<SparseMatrix<S>> tmp = make_shared<SparseMatrix<S>>();
        zero->factor = 0.0;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            switch (op.name) {
            case OpNames::I:
            case OpNames::N:
            case OpNames::NN:
            case OpNames::C:
            case OpNames::D:
            case OpNames::A:
            case OpNames::AD:
            case OpNames::B:
                p.second = this->find_site_norm_op(p.first, this->orb_sym[m]);
                break;
            case OpNames::H:
                p.second = make_shared<SparseMatrix<S>>();
                p.second->allocate(
                    this->find_site_op_info(op.q_label, this->orb_sym[m]));
                (*p.second)[S(0, 0, 0, 0)](0, 0) = 0.0;
                (*p.second)[S(1, 1, 1, this->orb_sym[m])](0, 0) = t(m, m);
                (*p.second)[S(2, 0, 0, 0)](0, 0) = t(m, m) * 2 + v(m, m, m, m);
                break;
            case OpNames::R:
                i = op.site_index[0];
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(i, m)) < TINY && abs(v(i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[0].at(OpNames::D));
                    p.second->factor *= t(i, m) * sqrt(2) / 4;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    tmp->copy_data_from(*op_prims[0].at(OpNames::R));
                    tmp->factor = v(i, m, m, m);
                    this->opf->iadd(*p.second, *tmp);
                    if (this->opf->seq->mode != SeqTypes::None)
                        this->opf->seq->simple_perform();
                    tmp->deallocate();
                }
                break;
            case OpNames::RD:
                i = op.site_index[0];
                if (this->orb_sym[i] != this->orb_sym[m] ||
                    (abs(t(i, m)) < TINY && abs(v(i, m, m, m)) < TINY))
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    p.second->copy_data_from(*op_prims[0].at(OpNames::C));
                    p.second->factor *= t(i, m) * sqrt(2) / 4;
                    tmp->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]));
                    tmp->copy_data_from(*op_prims[0].at(OpNames::RD));
                    tmp->factor = v(i, m, m, m);
                    this->opf->iadd(*p.second, *tmp);
                    if (this->opf->seq->mode != SeqTypes::None)
                        this->opf->seq->simple_perform();
                    tmp->deallocate();
                }
                break;
            case OpNames::P:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.s();
                if (abs(v(i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::AD)->data);
                    p.second->factor *= v(i, m, k, m);
                }
                break;
            case OpNames::PD:
                i = op.site_index[0];
                k = op.site_index[1];
                s = op.site_index.s();
                if (abs(v(i, m, k, m)) < TINY)
                    p.second = zero;
                else {
                    p.second = make_shared<SparseMatrix<S>>();
                    p.second->allocate(
                        this->find_site_op_info(op.q_label, this->orb_sym[m]),
                        op_prims[s].at(OpNames::A)->data);
                    p.second->factor *= v(i, m, k, m);
                }
                break;
            case OpNames::Q:
                i = op.site_index[0];
                j = op.site_index[1];
                s = op.site_index.s();
                switch (s) {
                case 0U:
                    if (abs(2 * v(i, j, m, m) - v(i, m, m, j)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                                               op.q_label, this->orb_sym[m]),
                                           op_prims[0].at(OpNames::B)->data);
                        p.second->factor *= 2 * v(i, j, m, m) - v(i, m, m, j);
                    }
                    break;
                case 1U:
                    if (abs(v(i, m, m, j)) < TINY)
                        p.second = zero;
                    else {
                        p.second = make_shared<SparseMatrix<S>>();
                        p.second->allocate(this->find_site_op_info(
                                               op.q_label, this->orb_sym[m]),
                                           op_prims[1].at(OpNames::B)->data);
                        p.second->factor *= v(i, m, m, j);
                    }
                    break;
                }
                break;
            default:
                assert(false);
            }
        }
    }
    void deallocate() override {
        for (auto name : vector<OpNames>{OpNames::RD, OpNames::R})
            op_prims[0][name]->deallocate();
        for (auto name : vector<OpNames>{OpNames::B, OpNames::AD, OpNames::A})
            op_prims[1][name]->deallocate();
        for (auto name :
             vector<OpNames>{OpNames::B, OpNames::AD, OpNames::A, OpNames::D,
                             OpNames::C, OpNames::NN, OpNames::N, OpNames::I})
            op_prims[0][name]->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            for (int j = this->site_op_infos[i].size() - 1; j >= 0; j--)
                this->site_op_infos[i][j].second->deallocate();
        for (int i = this->n_syms - 1; i >= 0; i--)
            this->basis[i].deallocate();
        Hamiltonian<S>::deallocate();
    }
    double v(uint8_t i, uint8_t j, uint8_t k, uint8_t l) const {
        return fcidump->v(i, j, k, l);
    }
    double t(uint8_t i, uint8_t j) const {
        return i == j ? fcidump->t(i, i) - mu : fcidump->t(i, j);
    }
    double e() const { return fcidump->e; }
};

template <typename S> struct IdentityMPO : MPO<S> {
    IdentityMPO(const Hamiltonian<S> &hamil) : MPO<S>(hamil.n_sites) {
        shared_ptr<OpElement<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vaccum);
        MPO<S>::op = i_op;
        MPO<S>::const_e = 0.0;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        for (uint8_t m = 0; m < hamil.n_sites; m++) {
            // site tensor
            shared_ptr<Symbolic<S>> pmat;
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(1);
            else if (m == hamil.n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(1);
            else
                pmat = make_shared<SymbolicMatrix<S>>(1, 1);
            (*pmat)[{0, 0}] = i_op;
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(1);
            (*plop)[0] = i_op;
            this->left_operator_names.push_back(plop);
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(1);
            (*prop)[0] = i_op;
            this->right_operator_names.push_back(prop);
            // site operators
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
};

enum QCTypes : uint8_t { NC = 1, CN = 2, Conventional = 4 };

template <typename, typename = void> struct MPOQC;

template <typename S> struct MPOQC<S, typename S::is_sz_t> : MPO<S> {
    QCTypes mode;
    bool symmetrized_p;
    MPOQC(const HamiltonianQC<S> &hamil, QCTypes mode = QCTypes::NC,
          bool symmetrized_p = true)
        : MPO<S>(hamil.n_sites), mode(mode), symmetrized_p(symmetrized_p) {
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S>>(OpNames::H, SiteIndex(), hamil.vaccum);
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vaccum);
        shared_ptr<OpExpr<S>> c_op[hamil.n_sites][2], d_op[hamil.n_sites][2];
        shared_ptr<OpExpr<S>> mc_op[hamil.n_sites][2], md_op[hamil.n_sites][2];
        shared_ptr<OpExpr<S>> rd_op[hamil.n_sites][2], r_op[hamil.n_sites][2];
        shared_ptr<OpExpr<S>> mrd_op[hamil.n_sites][2], mr_op[hamil.n_sites][2];
        shared_ptr<OpExpr<S>> a_op[hamil.n_sites][hamil.n_sites][4];
        shared_ptr<OpExpr<S>> ad_op[hamil.n_sites][hamil.n_sites][4];
        shared_ptr<OpExpr<S>> b_op[hamil.n_sites][hamil.n_sites][4];
        shared_ptr<OpExpr<S>> p_op[hamil.n_sites][hamil.n_sites][4];
        shared_ptr<OpExpr<S>> pd_op[hamil.n_sites][hamil.n_sites][4];
        shared_ptr<OpExpr<S>> q_op[hamil.n_sites][hamil.n_sites][4];
        MPO<S>::op = dynamic_pointer_cast<OpElement<S>>(h_op);
        MPO<S>::const_e = hamil.e();
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        uint8_t trans_l = -1, trans_r = hamil.n_sites;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN))
            trans_l = (hamil.n_sites >> 1) - 1, trans_r = (hamil.n_sites >> 1);
        else if (mode == QCTypes::Conventional)
            trans_l = (hamil.n_sites >> 1) - 1,
            trans_r = (hamil.n_sites >> 1) + 1;
        const int sz[2] = {1, -1};
        const int sz_plus[4] = {2, 0, 0, -2}, sz_minus[4] = {0, -2, 2, 0};
        for (uint8_t m = 0; m < hamil.n_sites; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::C, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil.orb_sym[m]));
                d_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::D, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil.orb_sym[m]));
                mc_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::C, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil.orb_sym[m]), -1.0);
                md_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], hamil.orb_sym[m]), -1.0);
                rd_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::RD, SiteIndex({m}, {s}),
                                              S(1, sz[s], hamil.orb_sym[m]));
                r_op[m][s] =
                    make_shared<OpElement<S>>(OpNames::R, SiteIndex({m}, {s}),
                                              S(-1, -sz[s], hamil.orb_sym[m]));
                mrd_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::RD, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil.orb_sym[m]), -1.0);
                mr_op[m][s] = make_shared<OpElement<S>>(
                    OpNames::R, SiteIndex({m}, {s}),
                    S(-1, -sz[s], hamil.orb_sym[m]), -1.0);
            }
        for (uint8_t i = 0; i < hamil.n_sites; i++)
            for (uint8_t j = 0; j < hamil.n_sites; j++)
                for (uint8_t s = 0; s < 4; s++) {
                    SiteIndex sidx({i, j},
                                   {(uint8_t)(s & 1), (uint8_t)(s >> 1)});
                    a_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::A, sidx,
                        S(2, sz_plus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    ad_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::AD, sidx,
                        S(-2, -sz_plus[s],
                          hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, sidx,
                        S(0, sz_minus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    p_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::P, sidx,
                        S(-2, -sz_plus[s],
                          hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    pd_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PD, sidx,
                        S(2, sz_plus[s], hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    q_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::Q, sidx,
                        S(0, -sz_minus[s],
                          hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                }
        int p;
        bool repeat_m = false;
        for (uint8_t m = 0; m < hamil.n_sites; m++) {
            shared_ptr<Symbolic<S>> pmat;
            int lshape, rshape;
            QCTypes effective_mode;
            if (mode == QCTypes::NC || ((mode & QCTypes::NC) && m <= trans_l) ||
                (mode == QCTypes::Conventional && m <= trans_l + 1 &&
                 !repeat_m))
                effective_mode = QCTypes::NC;
            else if (mode == QCTypes::CN ||
                     ((mode & QCTypes::CN) && m >= trans_r) ||
                     (mode == QCTypes::Conventional && m >= trans_r - 1))
                effective_mode = QCTypes::CN;
            else
                assert(false);
            switch (effective_mode) {
            case QCTypes::NC:
                lshape = 2 + 4 * hamil.n_sites + 12 * m * m;
                rshape = 2 + 4 * hamil.n_sites + 12 * (m + 1) * (m + 1);
                break;
            case QCTypes::CN:
                lshape = 2 + 4 * hamil.n_sites +
                         12 * (hamil.n_sites - m) * (hamil.n_sites - m);
                rshape = 2 + 4 * hamil.n_sites +
                         12 * (hamil.n_sites - m - 1) * (hamil.n_sites - m - 1);
                break;
            }
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (m == hamil.n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (m == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                mat[{0, 2}] = c_op[m][0];
                mat[{0, 3}] = c_op[m][1];
                mat[{0, 4}] = d_op[m][0];
                mat[{0, 5}] = d_op[m][1];
                p = 6;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                        mat[{0, p + j - m - 1}] = rd_op[j][s];
                    p += hamil.n_sites - (m + 1);
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                        mat[{0, p + j - m - 1}] = mr_op[j][s];
                    p += hamil.n_sites - (m + 1);
                }
            } else if (m == hamil.n_sites - 1) {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint8_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = r_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    for (uint8_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = mrd_op[j][s];
                    p += m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    mat[{p, 0}] = d_op[m][s];
                    p += hamil.n_sites - m;
                }
                for (uint8_t s = 0; s < 2; s++) {
                    mat[{p, 0}] = c_op[m][s];
                    p += hamil.n_sites - m;
                }
            }
            switch (effective_mode) {
            case QCTypes::NC:
                if (m == 0) {
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{0, p + s}] = a_op[m][m][s];
                    p += 4;
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{0, p + s}] = ad_op[m][m][s];
                    p += 4;
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{0, p + s}] = b_op[m][m][s];
                    p += 4;
                    assert(p == mat.n);
                } else {
                    if (m != hamil.n_sites - 1) {
                        mat[{0, 0}] = i_op;
                        mat[{1, 0}] = h_op;
                        p = 2;
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint8_t j = 0; j < m; j++)
                                mat[{p + j, 0}] = r_op[j][s];
                            p += m;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint8_t j = 0; j < m; j++)
                                mat[{p + j, 0}] = mrd_op[j][s];
                            p += m;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{p, 0}] = d_op[m][s];
                            p += hamil.n_sites - m;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{p, 0}] = c_op[m][s];
                            p += hamil.n_sites - m;
                        }
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint8_t j = 0; j < m; j++) {
                            for (uint8_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = 0.5 * p_op[j][k][s];
                            p += m;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint8_t j = 0; j < m; j++) {
                            for (uint8_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = 0.5 * pd_op[j][k][s];
                            p += m;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint8_t j = 0; j < m; j++) {
                            for (uint8_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = q_op[j][k][s];
                            p += m;
                        }
                    assert(p == mat.m);
                }
                if (m != 0 && m != hamil.n_sites - 1) {
                    mat[{1, 1}] = i_op;
                    p = 2;
                    // pointers
                    int pi = 1;
                    int pc[2] = {2, 2 + m};
                    int pd[2] = {2 + m * 2, 2 + m * 3};
                    int prd[2] = {2 + m * 4 - m, 2 + m * 3 + hamil.n_sites - m};
                    int pr[2] = {2 + m * 2 + hamil.n_sites * 2 - m,
                                 2 + m + hamil.n_sites * 3 - m};
                    int pa[4] = {2 + hamil.n_sites * 4 + m * m * 0,
                                 2 + hamil.n_sites * 4 + m * m * 1,
                                 2 + hamil.n_sites * 4 + m * m * 2,
                                 2 + hamil.n_sites * 4 + m * m * 3};
                    int pad[4] = {2 + hamil.n_sites * 4 + m * m * 4,
                                  2 + hamil.n_sites * 4 + m * m * 5,
                                  2 + hamil.n_sites * 4 + m * m * 6,
                                  2 + hamil.n_sites * 4 + m * m * 7};
                    int pb[4] = {2 + hamil.n_sites * 4 + m * m * 8,
                                 2 + hamil.n_sites * 4 + m * m * 9,
                                 2 + hamil.n_sites * 4 + m * m * 10,
                                 2 + hamil.n_sites * 4 + m * m * 11};
                    // C
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = 0; j < m; j++)
                            mat[{pc[s] + j, p + j}] = i_op;
                        mat[{pi, p + m}] = c_op[m][s];
                        p += m + 1;
                    }
                    // D
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = 0; j < m; j++)
                            mat[{pd[s] + j, p + j}] = i_op;
                        mat[{pi, p + m}] = d_op[m][s];
                        p += m + 1;
                    }
                    // RD
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{prd[s] + i, p + i - (m + 1)}] = i_op;
                            mat[{pi, p + i - (m + 1)}] = rd_op[i][s];
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = 0; k < m; k++) {
                                    mat[{pd[sp] + k, p + i - (m + 1)}] =
                                        pd_op[i][k][s | (sp << 1)];
                                    mat[{pc[sp] + k, p + i - (m + 1)}] =
                                        q_op[k][i][sp | (s << 1)];
                                }
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = 0; j < m; j++)
                                        for (uint8_t l = 0; l < m; l++) {
                                            double f =
                                                hamil.v(s, sp, i, j, m, l);
                                            mat[{pa[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] =
                                                f * d_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = 0; j < m; j++)
                                        for (uint8_t l = 0; l < m; l++) {
                                            double f0 = 0.5 * hamil.v(s, sp, i,
                                                                      j, m, l),
                                                   f1 = -0.5 * hamil.v(s, sp, i,
                                                                       l, m, j);
                                            mat[{pa[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f0 * d_op[m][sp];
                                            mat[{pa[sp | (s << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f1 * d_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = 0; k < m; k++)
                                    for (uint8_t l = 0; l < m; l++) {
                                        double f = hamil.v(s, sp, i, m, k, l);
                                        mat[{pb[sp | (sp << 1)] + l * m + k,
                                             p + i - (m + 1)}] = f * c_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t j = 0; j < m; j++)
                                    for (uint8_t k = 0; k < m; k++) {
                                        double f =
                                            -1.0 * hamil.v(s, sp, i, j, k, m);
                                        mat[{pb[s | (sp << 1)] + j * m + k,
                                             p + i - (m + 1)}] +=
                                            f * c_op[m][sp];
                                    }
                        }
                        p += hamil.n_sites - (m + 1);
                    }
                    // R
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{pr[s] + i, p + i - (m + 1)}] = i_op;
                            mat[{pi, p + i - (m + 1)}] = mr_op[i][s];
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = 0; k < m; k++) {
                                    mat[{pc[sp] + k, p + i - (m + 1)}] =
                                        -1.0 * p_op[i][k][s | (sp << 1)];
                                    mat[{pd[sp] + k, p + i - (m + 1)}] =
                                        -1.0 * q_op[i][k][s | (sp << 1)];
                                }
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = 0; j < m; j++)
                                        for (uint8_t l = 0; l < m; l++) {
                                            double f = -1.0 * hamil.v(s, sp, i,
                                                                      j, m, l);
                                            mat[{pad[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] =
                                                f * c_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = 0; j < m; j++)
                                        for (uint8_t l = 0; l < m; l++) {
                                            double f0 = -0.5 * hamil.v(s, sp, i,
                                                                       j, m, l),
                                                   f1 = 0.5 * hamil.v(s, sp, i,
                                                                      l, m, j);
                                            mat[{pad[s | (sp << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f0 * c_op[m][sp];
                                            mat[{pad[sp | (s << 1)] + j * m + l,
                                                 p + i - (m + 1)}] +=
                                                f1 * c_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = 0; k < m; k++)
                                    for (uint8_t l = 0; l < m; l++) {
                                        double f =
                                            -1.0 * hamil.v(s, sp, i, m, k, l);
                                        mat[{pb[sp | (sp << 1)] + k * m + l,
                                             p + i - (m + 1)}] = f * d_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t j = 0; j < m; j++)
                                    for (uint8_t k = 0; k < m; k++) {
                                        double f = (-1.0) * (-1.0) *
                                                   hamil.v(s, sp, i, j, k, m);
                                        mat[{pb[sp | (s << 1)] + k * m + j,
                                             p + i - (m + 1)}] =
                                            f * d_op[m][sp];
                                    }
                        }
                        p += hamil.n_sites - (m + 1);
                    }
                    // A
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint8_t i = 0; i < m; i++)
                            for (uint8_t j = 0; j < m; j++)
                                mat[{pa[s] + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{pc[s & 1] + i, p + i * (m + 1) + m}] =
                                c_op[m][s >> 1];
                            mat[{pc[s >> 1] + i, p + m * (m + 1) + i}] =
                                mc_op[m][s & 1];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = a_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // AD
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint8_t i = 0; i < m; i++)
                            for (uint8_t j = 0; j < m; j++)
                                mat[{pad[s] + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{pd[s & 1] + i, p + i * (m + 1) + m}] =
                                md_op[m][s >> 1];
                            mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] =
                                d_op[m][s & 1];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // B
                    for (uint8_t s = 0; s < 4; s++) {
                        for (uint8_t i = 0; i < m; i++)
                            for (uint8_t j = 0; j < m; j++)
                                mat[{pb[s] + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{pc[s & 1] + i, p + i * (m + 1) + m}] =
                                d_op[m][s >> 1];
                            mat[{pd[s >> 1] + i, p + m * (m + 1) + i}] =
                                mc_op[m][s & 1];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = b_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    assert(p == mat.n);
                }
                break;
            case QCTypes::CN:
                if (m == hamil.n_sites - 1) {
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{p + s, 0}] = a_op[m][m][s];
                    p += 4;
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{p + s, 0}] = ad_op[m][m][s];
                    p += 4;
                    for (uint8_t s = 0; s < 4; s++)
                        mat[{p + s, 0}] = b_op[m][m][s];
                    p += 4;
                    assert(p == mat.m);
                } else {
                    if (m != 0) {
                        mat[{1, 0}] = h_op;
                        mat[{1, 1}] = i_op;
                        p = 2;
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{1, p + m}] = c_op[m][s];
                            p += m + 1;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            mat[{1, p + m}] = d_op[m][s];
                            p += m + 1;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{1, p + j - m - 1}] = rd_op[j][s];
                            p += hamil.n_sites - m - 1;
                        }
                        for (uint8_t s = 0; s < 2; s++) {
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{1, p + j - m - 1}] = mr_op[j][s];
                            p += hamil.n_sites - m - 1;
                        }
                    }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                mat[{!!m, p + k - m - 1}] = 0.5 * p_op[j][k][s];
                            p += hamil.n_sites - m - 1;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                mat[{!!m, p + k - m - 1}] =
                                    0.5 * pd_op[j][k][s];
                            p += hamil.n_sites - m - 1;
                        }
                    for (uint8_t s = 0; s < 4; s++)
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                mat[{!!m, p + k - m - 1}] = q_op[j][k][s];
                            p += hamil.n_sites - m - 1;
                        }
                    assert(p == mat.n);
                }
                if (m != 0 && m != hamil.n_sites - 1) {
                    mat[{0, 0}] = i_op;
                    p = 2;
                    // pointers
                    int mm = hamil.n_sites - m - 1;
                    int pm = hamil.n_sites - m;
                    int pi = 0;
                    int pr[2] = {2, 2 + m + 1};
                    int prd[2] = {2 + (m + 1) * 2, 2 + (m + 1) * 3};
                    int pd[2] = {2 + (m + 1) * 4 - m - 1,
                                 2 + (m + 1) * 3 + hamil.n_sites - m - 1};
                    int pc[2] = {2 + (m + 1) * 2 + hamil.n_sites * 2 - m - 1,
                                 2 + (m + 1) + hamil.n_sites * 3 - m - 1};
                    int pa[4] = {2 + hamil.n_sites * 4 + mm * mm * 0,
                                 2 + hamil.n_sites * 4 + mm * mm * 1,
                                 2 + hamil.n_sites * 4 + mm * mm * 2,
                                 2 + hamil.n_sites * 4 + mm * mm * 3};
                    int pad[4] = {2 + hamil.n_sites * 4 + mm * mm * 4,
                                  2 + hamil.n_sites * 4 + mm * mm * 5,
                                  2 + hamil.n_sites * 4 + mm * mm * 6,
                                  2 + hamil.n_sites * 4 + mm * mm * 7};
                    int pb[4] = {2 + hamil.n_sites * 4 + mm * mm * 8,
                                 2 + hamil.n_sites * 4 + mm * mm * 9,
                                 2 + hamil.n_sites * 4 + mm * mm * 10,
                                 2 + hamil.n_sites * 4 + mm * mm * 11};
                    // R
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{p + i, pi}] = r_op[i][s];
                            mat[{p + i, pr[s] + i}] = i_op;
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = m + 1; j < hamil.n_sites;
                                         j++)
                                        for (uint8_t l = m + 1;
                                             l < hamil.n_sites; l++) {
                                            double f =
                                                hamil.v(s, sp, i, j, m, l);
                                            mat[{p + i, pad[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] =
                                                f * c_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = m + 1; j < hamil.n_sites;
                                         j++)
                                        for (uint8_t l = m + 1;
                                             l < hamil.n_sites; l++) {
                                            double f0 = 0.5 * hamil.v(s, sp, i,
                                                                      j, m, l);
                                            double f1 = -0.5 * hamil.v(s, sp, i,
                                                                       l, m, j);
                                            mat[{p + i, pad[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f0 * c_op[m][sp];
                                            mat[{p + i, pad[sp | (s << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f1 * c_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    for (uint8_t l = m + 1; l < hamil.n_sites;
                                         l++) {
                                        double f = hamil.v(s, sp, i, m, k, l);
                                        mat[{p + i, pb[sp | (sp << 1)] +
                                                        (k - m - 1) * mm + l -
                                                        m - 1}] =
                                            f * d_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                    for (uint8_t k = m + 1; k < hamil.n_sites;
                                         k++) {
                                        double f =
                                            (-1.0) * hamil.v(s, sp, i, j, k, m);
                                        mat[{p + i, pb[sp | (s << 1)] +
                                                        (k - m - 1) * mm + j -
                                                        m - 1}] =
                                            f * d_op[m][sp];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = m + 1; k < hamil.n_sites;
                                     k++) {
                                    mat[{p + i, pc[sp] + k}] =
                                        -1.0 * p_op[k][i][sp | (s << 1)];
                                    mat[{p + i, pd[sp] + k}] =
                                        q_op[i][k][s | (sp << 1)];
                                }
                        }
                        p += m;
                    }
                    // RD
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{p + i, pi}] = mrd_op[i][s];
                            mat[{p + i, prd[s] + i}] = i_op;
                            if (!symmetrized_p)
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = m + 1; j < hamil.n_sites;
                                         j++)
                                        for (uint8_t l = m + 1;
                                             l < hamil.n_sites; l++) {
                                            double f = -1.0 * hamil.v(s, sp, i,
                                                                      j, m, l);
                                            mat[{p + i, pa[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] =
                                                f * d_op[m][sp];
                                        }
                            else
                                for (uint8_t sp = 0; sp < 2; sp++)
                                    for (uint8_t j = m + 1; j < hamil.n_sites;
                                         j++)
                                        for (uint8_t l = m + 1;
                                             l < hamil.n_sites; l++) {
                                            double f0 = -0.5 * hamil.v(s, sp, i,
                                                                       j, m, l);
                                            double f1 = 0.5 * hamil.v(s, sp, i,
                                                                      l, m, j);
                                            mat[{p + i, pa[s | (sp << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f0 * d_op[m][sp];
                                            mat[{p + i, pa[sp | (s << 1)] +
                                                            (j - m - 1) * mm +
                                                            l - m - 1}] +=
                                                f1 * d_op[m][sp];
                                        }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    for (uint8_t l = m + 1; l < hamil.n_sites;
                                         l++) {
                                        double f =
                                            -1.0 * hamil.v(s, sp, i, m, k, l);
                                        mat[{p + i, pb[sp | (sp << 1)] +
                                                        (l - m - 1) * mm + k -
                                                        m - 1}] =
                                            f * c_op[m][s];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                    for (uint8_t k = m + 1; k < hamil.n_sites;
                                         k++) {
                                        double f = hamil.v(s, sp, i, j, k, m);
                                        mat[{p + i, pb[s | (sp << 1)] +
                                                        (j - m - 1) * mm + k -
                                                        m - 1}] =
                                            f * c_op[m][sp];
                                    }
                            for (uint8_t sp = 0; sp < 2; sp++)
                                for (uint8_t k = m + 1; k < hamil.n_sites;
                                     k++) {
                                    mat[{p + i, pd[sp] + k}] =
                                        pd_op[k][i][sp | (s << 1)];
                                    mat[{p + i, pc[sp] + k}] =
                                        -1.0 * q_op[k][i][sp | (s << 1)];
                                }
                        }
                        p += m;
                    }
                    // D
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p + m - m, pi}] = d_op[m][s];
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            mat[{p + j - m, pd[s] + j}] = i_op;
                        p += hamil.n_sites - m;
                    }
                    // C
                    for (uint8_t s = 0; s < 2; s++) {
                        mat[{p + m - m, pi}] = c_op[m][s];
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            mat[{p + j - m, pc[s] + j}] = i_op;
                        p += hamil.n_sites - m;
                    }
                    // A
                    for (uint8_t s = 0; s < 4; s++) {
                        mat[{p + (m - m) * pm + m - m, pi}] = a_op[m][m][s];
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{p + (m - m) * pm + i - m, pc[s >> 1] + i}] =
                                c_op[m][s & 1];
                            mat[{p + (i - m) * pm + m - m, pc[s & 1] + i}] =
                                mc_op[m][s >> 1];
                        }
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{p + (i - m) * pm + j - m,
                                     pa[s] + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += pm * pm;
                    }
                    // AD
                    for (uint8_t s = 0; s < 4; s++) {
                        mat[{p + (m - m) * pm + m - m, pi}] = ad_op[m][m][s];
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{p + (m - m) * pm + i - m, pd[s >> 1] + i}] =
                                md_op[m][s & 1];
                            mat[{p + (i - m) * pm + m - m, pd[s & 1] + i}] =
                                d_op[m][s >> 1];
                        }
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{p + (i - m) * pm + j - m,
                                     pad[s] + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += pm * pm;
                    }
                    // B
                    for (uint8_t s = 0; s < 4; s++) {
                        mat[{p + (m - m) * pm + m - m, pi}] = b_op[m][m][s];
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{p + (m - m) * pm + i - m, pd[s >> 1] + i}] =
                                c_op[m][s & 1];
                            mat[{p + (i - m) * pm + m - m, pc[s & 1] + i}] =
                                md_op[m][s >> 1];
                        }
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{p + (i - m) * pm + j - m,
                                     pb[s] + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += pm * pm;
                    }
                    assert(p == mat.m);
                }
                break;
            case QCTypes::NC | QCTypes::CN:
                assert(false);
                break;
            }
            shared_ptr<OperatorTensor<S>> opt = nullptr;
            if (mode != QCTypes::Conventional ||
                !(m == trans_l + 1 && m == trans_r - 1)) {
                opt = make_shared<OperatorTensor<S>>();
                opt->lmat = opt->rmat = pmat;
            } else if (!repeat_m) {
                opt = make_shared<OperatorTensor<S>>();
                opt->rmat = pmat;
            } else {
                opt = this->tensors.back();
                this->tensors.pop_back();
                opt->lmat = pmat;
            }
            // operator names
            if (opt->lmat == pmat) {
                shared_ptr<SymbolicRowVector<S>> plop;
                if (m == hamil.n_sites - 1)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (m != hamil.n_sites - 1) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = 0; j < m + 1; j++)
                            lop[p + j] = c_op[j][s];
                        p += m + 1;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = 0; j < m + 1; j++)
                            lop[p + j] = d_op[j][s];
                        p += m + 1;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            lop[p + j - (m + 1)] = rd_op[j][s];
                        p += hamil.n_sites - (m + 1);
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            lop[p + j - (m + 1)] = mr_op[j][s];
                        p += hamil.n_sites - (m + 1);
                    }
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = 0; j < m + 1; j++) {
                                for (uint8_t k = 0; k < m + 1; k++)
                                    lop[p + k] = a_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = 0; j < m + 1; j++) {
                                for (uint8_t k = 0; k < m + 1; k++)
                                    lop[p + k] = ad_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = 0; j < m + 1; j++) {
                                for (uint8_t k = 0; k < m + 1; k++)
                                    lop[p + k] = b_op[j][k][s];
                                p += m + 1;
                            }
                        break;
                    case QCTypes::CN:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    lop[p + k - m - 1] = 0.5 * p_op[j][k][s];
                                p += hamil.n_sites - m - 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    lop[p + k - m - 1] = 0.5 * pd_op[j][k][s];
                                p += hamil.n_sites - m - 1;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    lop[p + k - m - 1] = q_op[j][k][s];
                                p += hamil.n_sites - m - 1;
                            }
                        break;
                    case QCTypes::NC | QCTypes::CN:
                        assert(false);
                        break;
                    }
                    assert(p == rshape);
                }
                this->left_operator_names.push_back(plop);
            }
            if (opt->rmat == pmat) {
                shared_ptr<SymbolicColumnVector<S>> prop;
                if (m == 0)
                    prop = make_shared<SymbolicColumnVector<S>>(1);
                else
                    prop = make_shared<SymbolicColumnVector<S>>(lshape);
                SymbolicColumnVector<S> &rop = *prop;
                if (m == 0)
                    rop[0] = h_op;
                else {
                    rop[0] = i_op;
                    rop[1] = h_op;
                    p = 2;
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = 0; j < m; j++)
                            rop[p + j] = r_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = 0; j < m; j++)
                            rop[p + j] = mrd_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = m; j < hamil.n_sites; j++)
                            rop[p + j - m] = d_op[j][s];
                        p += hamil.n_sites - m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint8_t j = m; j < hamil.n_sites; j++)
                            rop[p + j - m] = c_op[j][s];
                        p += hamil.n_sites - m;
                    }
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = 0; j < m; j++) {
                                for (uint8_t k = 0; k < m; k++)
                                    rop[p + k] = 0.5 * p_op[j][k][s];
                                p += m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = 0; j < m; j++) {
                                for (uint8_t k = 0; k < m; k++)
                                    rop[p + k] = 0.5 * pd_op[j][k][s];
                                p += m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = 0; j < m; j++) {
                                for (uint8_t k = 0; k < m; k++)
                                    rop[p + k] = q_op[j][k][s];
                                p += m;
                            }
                        break;
                    case QCTypes::CN:
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = m; j < hamil.n_sites; j++) {
                                for (uint8_t k = m; k < hamil.n_sites; k++)
                                    rop[p + k - m] = a_op[j][k][s];
                                p += hamil.n_sites - m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = m; j < hamil.n_sites; j++) {
                                for (uint8_t k = m; k < hamil.n_sites; k++)
                                    rop[p + k - m] = ad_op[j][k][s];
                                p += hamil.n_sites - m;
                            }
                        for (uint8_t s = 0; s < 4; s++)
                            for (uint8_t j = m; j < hamil.n_sites; j++) {
                                for (uint8_t k = m; k < hamil.n_sites; k++)
                                    rop[p + k - m] = b_op[j][k][s];
                                p += hamil.n_sites - m;
                            }
                        break;
                    case QCTypes::NC | QCTypes::CN:
                        assert(false);
                        break;
                    }
                    assert(p == lshape);
                }
                this->right_operator_names.push_back(prop);
            }
            if (mode == QCTypes::Conventional && m == trans_l + 1 &&
                m == trans_r - 1 && !repeat_m) {
                repeat_m = true;
                m--;
                this->tensors.push_back(opt);
                continue;
            }
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN) ||
            mode == QCTypes::Conventional) {
            uint8_t m;
            MPO<S>::schemer = make_shared<MPOSchemer<S>>(trans_l, trans_r);
            // left transform
            m = trans_l;
            int new_rshape =
                2 + 4 * hamil.n_sites +
                12 * (hamil.n_sites - m - 1) * (hamil.n_sites - m - 1);
            MPO<S>::schemer->left_new_operator_names =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            MPO<S>::schemer->left_new_operator_exprs =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            SymbolicRowVector<S> &lop =
                *MPO<S>::schemer->left_new_operator_names;
            SymbolicRowVector<S> &lexpr =
                *MPO<S>::schemer->left_new_operator_exprs;
            for (int i = 0; i < 2 + 4 * hamil.n_sites; i++)
                lop[i] = this->left_operator_names[m]->data[i];
            p = 2 + 4 * hamil.n_sites;
            vector<shared_ptr<OpExpr<S>>> exprs;
            for (uint8_t s = 0; s < 4; s++)
                for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                    for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                        lop[p + k - m - 1] = 0.5 * p_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = 0; g < m + 1; g++)
                            for (uint8_t h = 0; h < m + 1; h++)
                                if (abs(hamil.v(s & 1, s >> 1, j, g, k, h)) >
                                    TINY)
                                    exprs.push_back(
                                        (0.5 *
                                         hamil.v(s & 1, s >> 1, j, g, k, h)) *
                                        ad_op[g][h][s]);
                        lexpr[p + k - m - 1] = sum(exprs);
                    }
                    p += hamil.n_sites - m - 1;
                }
            for (uint8_t s = 0; s < 4; s++)
                for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                    for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                        lop[p + k - m - 1] = 0.5 * pd_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = 0; g < m + 1; g++)
                            for (uint8_t h = 0; h < m + 1; h++)
                                if (abs(hamil.v(s & 1, s >> 1, j, g, k, h)) >
                                    TINY)
                                    exprs.push_back(
                                        (0.5 *
                                         hamil.v(s & 1, s >> 1, j, g, k, h)) *
                                        a_op[g][h][s]);
                        lexpr[p + k - m - 1] = sum(exprs);
                    }
                    p += hamil.n_sites - m - 1;
                }
            for (uint8_t s = 0; s < 4; s++)
                for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                    for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                        lop[p + k - m - 1] = q_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = 0; g < m + 1; g++)
                            for (uint8_t h = 0; h < m + 1; h++) {
                                if (abs(hamil.v(s & 1, s >> 1, j, h, g, k)) >
                                    TINY)
                                    exprs.push_back(
                                        -hamil.v(s & 1, s >> 1, j, h, g, k) *
                                        b_op[g][h][((s & 1) << 1) | (s >> 1)]);
                                if ((s & 1) == (s >> 1))
                                    for (uint8_t sp = 0; sp < 2; sp++)
                                        if (abs(hamil.v(s & 1, sp, j, k, g,
                                                        h)) > TINY)
                                            exprs.push_back(
                                                hamil.v(s & 1, sp, j, k, g, h) *
                                                b_op[g][h][(sp << 1) | sp]);
                            }
                        lexpr[p + k - m - 1] = sum(exprs);
                    }
                    p += hamil.n_sites - m - 1;
                }
            assert(p == new_rshape);
            // right transform
            m = trans_r - 1;
            int new_lshape = 2 + 4 * hamil.n_sites + 12 * (m + 1) * (m + 1);
            MPO<S>::schemer->right_new_operator_names =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            MPO<S>::schemer->right_new_operator_exprs =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            SymbolicColumnVector<S> &rop =
                *MPO<S>::schemer->right_new_operator_names;
            SymbolicColumnVector<S> &rexpr =
                *MPO<S>::schemer->right_new_operator_exprs;
            for (int i = 0; i < 2 + 4 * hamil.n_sites; i++)
                rop[i] = this->right_operator_names[m + 1]->data[i];
            p = 2 + 4 * hamil.n_sites;
            for (uint8_t s = 0; s < 4; s++)
                for (uint8_t j = 0; j < m + 1; j++) {
                    for (uint8_t k = 0; k < m + 1; k++) {
                        rop[p + k] = 0.5 * p_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                            for (uint8_t h = m + 1; h < hamil.n_sites; h++)
                                if (abs(hamil.v(s & 1, s >> 1, j, g, k, h)) >
                                    TINY)
                                    exprs.push_back(
                                        (0.5 *
                                         hamil.v(s & 1, s >> 1, j, g, k, h)) *
                                        ad_op[g][h][s]);
                        rexpr[p + k] = sum(exprs);
                    }
                    p += m + 1;
                }
            for (uint8_t s = 0; s < 4; s++)
                for (uint8_t j = 0; j < m + 1; j++) {
                    for (uint8_t k = 0; k < m + 1; k++) {
                        rop[p + k] = 0.5 * pd_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                            for (uint8_t h = m + 1; h < hamil.n_sites; h++)
                                if (abs(hamil.v(s & 1, s >> 1, j, g, k, h)) >
                                    TINY)
                                    exprs.push_back(
                                        (0.5 *
                                         hamil.v(s & 1, s >> 1, j, g, k, h)) *
                                        a_op[g][h][s]);
                        rexpr[p + k] = sum(exprs);
                    }
                    p += m + 1;
                }
            for (uint8_t s = 0; s < 4; s++)
                for (uint8_t j = 0; j < m + 1; j++) {
                    for (uint8_t k = 0; k < m + 1; k++) {
                        rop[p + k] = q_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                            for (uint8_t h = m + 1; h < hamil.n_sites; h++) {
                                if (abs(hamil.v(s & 1, s >> 1, j, h, g, k)) >
                                    TINY)
                                    exprs.push_back(
                                        -hamil.v(s & 1, s >> 1, j, h, g, k) *
                                        b_op[g][h][((s & 1) << 1) | (s >> 1)]);
                                if ((s & 1) == (s >> 1))
                                    for (uint8_t sp = 0; sp < 2; sp++)
                                        if (abs(hamil.v(s & 1, sp, j, k, g,
                                                        h)) > TINY)
                                            exprs.push_back(
                                                hamil.v(s & 1, sp, j, k, g, h) *
                                                b_op[g][h][(sp << 1) | sp]);
                            }
                        rexpr[p + k] = sum(exprs);
                    }
                    p += m + 1;
                }
            assert(p == new_lshape);
        }
    }
    void deallocate() override {
        for (uint8_t m = MPO<S>::n_sites - 1; m < MPO<S>::n_sites; m--)
            for (auto it = this->tensors[m]->ops.crbegin();
                 it != this->tensors[m]->ops.crend(); ++it) {
                OpElement<S> &op =
                    *dynamic_pointer_cast<OpElement<S>>(it->first);
                if (op.name == OpNames::R || op.name == OpNames::RD ||
                    op.name == OpNames::H ||
                    (op.name == OpNames::Q &&
                     op.site_index.s(0) == op.site_index.s(1)))
                    it->second->deallocate();
            }
    }
};

template <typename S> struct MPOQC<S, typename S::is_su2_t> : MPO<S> {
    QCTypes mode;
    MPOQC(const HamiltonianQC<S> &hamil, QCTypes mode = QCTypes::NC)
        : MPO<S>(hamil.n_sites), mode(mode) {
        shared_ptr<OpExpr<S>> h_op =
            make_shared<OpElement<S>>(OpNames::H, SiteIndex(), hamil.vaccum);
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vaccum);
        shared_ptr<OpExpr<S>> c_op[hamil.n_sites], d_op[hamil.n_sites];
        shared_ptr<OpExpr<S>> mc_op[hamil.n_sites], md_op[hamil.n_sites];
        shared_ptr<OpExpr<S>> trd_op[hamil.n_sites], tr_op[hamil.n_sites];
        shared_ptr<OpExpr<S>> a_op[hamil.n_sites][hamil.n_sites][2];
        shared_ptr<OpExpr<S>> ad_op[hamil.n_sites][hamil.n_sites][2];
        shared_ptr<OpExpr<S>> b_op[hamil.n_sites][hamil.n_sites][2];
        shared_ptr<OpExpr<S>> p_op[hamil.n_sites][hamil.n_sites][2];
        shared_ptr<OpExpr<S>> pd_op[hamil.n_sites][hamil.n_sites][2];
        shared_ptr<OpExpr<S>> q_op[hamil.n_sites][hamil.n_sites][2];
        MPO<S>::op = dynamic_pointer_cast<OpElement<S>>(h_op);
        MPO<S>::const_e = hamil.e();
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        uint8_t trans_l = -1, trans_r = hamil.n_sites;
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN))
            trans_l = (hamil.n_sites >> 1) - 1, trans_r = (hamil.n_sites >> 1);
        else if (mode == QCTypes::Conventional)
            trans_l = (hamil.n_sites >> 1) - 1,
            trans_r = (hamil.n_sites >> 1) + 1;
        for (uint8_t m = 0; m < hamil.n_sites; m++) {
            c_op[m] = make_shared<OpElement<S>>(OpNames::C, SiteIndex(m),
                                                S(1, 1, hamil.orb_sym[m]));
            d_op[m] = make_shared<OpElement<S>>(OpNames::D, SiteIndex(m),
                                                S(-1, 1, hamil.orb_sym[m]));
            mc_op[m] = make_shared<OpElement<S>>(
                OpNames::C, SiteIndex(m), S(1, 1, hamil.orb_sym[m]), -1.0);
            md_op[m] = make_shared<OpElement<S>>(
                OpNames::D, SiteIndex(m), S(-1, 1, hamil.orb_sym[m]), -1.0);
            trd_op[m] = make_shared<OpElement<S>>(
                OpNames::RD, SiteIndex(m), S(1, 1, hamil.orb_sym[m]), 2.0);
            tr_op[m] = make_shared<OpElement<S>>(
                OpNames::R, SiteIndex(m), S(-1, 1, hamil.orb_sym[m]), 2.0);
        }
        for (uint8_t i = 0; i < hamil.n_sites; i++)
            for (uint8_t j = 0; j < hamil.n_sites; j++)
                for (uint8_t s = 0; s < 2; s++) {
                    a_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::A, SiteIndex(i, j, s),
                        S(2, s * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    ad_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::AD, SiteIndex(i, j, s),
                        S(-2, s * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    b_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::B, SiteIndex(i, j, s),
                        S(0, s * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    p_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::P, SiteIndex(i, j, s),
                        S(-2, s * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    pd_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::PD, SiteIndex(i, j, s),
                        S(2, s * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                    q_op[i][j][s] = make_shared<OpElement<S>>(
                        OpNames::Q, SiteIndex(i, j, s),
                        S(0, s * 2, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                }
        int p;
        bool repeat_m = false;
        for (uint8_t m = 0; m < hamil.n_sites; m++) {
            shared_ptr<Symbolic<S>> pmat;
            int lshape, rshape;
            QCTypes effective_mode;
            if (mode == QCTypes::NC || ((mode & QCTypes::NC) && m <= trans_l) ||
                (mode == QCTypes::Conventional && m <= trans_l + 1 &&
                 !repeat_m))
                effective_mode = QCTypes::NC;
            else if (mode == QCTypes::CN ||
                     ((mode & QCTypes::CN) && m >= trans_r) ||
                     (mode == QCTypes::Conventional && m >= trans_r - 1))
                effective_mode = QCTypes::CN;
            else
                assert(false);
            switch (effective_mode) {
            case QCTypes::NC:
                lshape = 2 + 2 * hamil.n_sites + 6 * m * m;
                rshape = 2 + 2 * hamil.n_sites + 6 * (m + 1) * (m + 1);
                break;
            case QCTypes::CN:
                lshape = 2 + 2 * hamil.n_sites +
                         6 * (hamil.n_sites - m) * (hamil.n_sites - m);
                rshape = 2 + 2 * hamil.n_sites +
                         6 * (hamil.n_sites - m - 1) * (hamil.n_sites - m - 1);
                break;
            }
            if (m == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (m == hamil.n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (m == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                mat[{0, 2}] = c_op[m];
                mat[{0, 3}] = d_op[m];
                p = 4;
                for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                    mat[{0, p + j - m - 1}] = trd_op[j];
                p += hamil.n_sites - (m + 1);
                for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                    mat[{0, p + j - m - 1}] = tr_op[j];
                p += hamil.n_sites - (m + 1);
            } else if (m == hamil.n_sites - 1) {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                for (uint8_t j = 0; j < m; j++)
                    mat[{p + j, 0}] = tr_op[j];
                p += m;
                for (uint8_t j = 0; j < m; j++)
                    mat[{p + j, 0}] = trd_op[j];
                p += m;
                mat[{p, 0}] = d_op[m];
                p += hamil.n_sites - m;
                mat[{p, 0}] = c_op[m];
                p += hamil.n_sites - m;
            }
            switch (effective_mode) {
            case QCTypes::NC:
                if (m == 0) {
                    for (uint8_t s = 0; s < 2; s++)
                        mat[{0, p + s}] = a_op[m][m][s];
                    p += 2;
                    for (uint8_t s = 0; s < 2; s++)
                        mat[{0, p + s}] = ad_op[m][m][s];
                    p += 2;
                    for (uint8_t s = 0; s < 2; s++)
                        mat[{0, p + s}] = b_op[m][m][s];
                    p += 2;
                    assert(p == mat.n);
                } else {
                    if (m != hamil.n_sites - 1) {
                        mat[{0, 0}] = i_op;
                        mat[{1, 0}] = h_op;
                        p = 2;
                        for (uint8_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = tr_op[j];
                        p += m;
                        for (uint8_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = trd_op[j];
                        p += m;
                        mat[{p, 0}] = d_op[m];
                        p += hamil.n_sites - m;
                        mat[{p, 0}] = c_op[m];
                        p += hamil.n_sites - m;
                    }
                    vector<double> su2_factor{-0.5, -0.5 * sqrt(3)};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint8_t j = 0; j < m; j++) {
                            for (uint8_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = su2_factor[s] * p_op[j][k][s];
                            p += m;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint8_t j = 0; j < m; j++) {
                            for (uint8_t k = 0; k < m; k++)
                                mat[{p + k, 0}] =
                                    su2_factor[s] * pd_op[j][k][s];
                            p += m;
                        }
                    su2_factor = {1.0, sqrt(3)};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint8_t j = 0; j < m; j++) {
                            for (uint8_t k = 0; k < m; k++)
                                mat[{p + k, 0}] = su2_factor[s] * q_op[j][k][s];
                            p += m;
                        }
                    assert(p == mat.m);
                }
                if (m != 0 && m != hamil.n_sites - 1) {
                    mat[{1, 1}] = i_op;
                    p = 2;
                    // pointers
                    int pi = 1, pc = 2, pd = 2 + m;
                    int prd = 2 + m + m - m, pr = 2 + m + hamil.n_sites - m;
                    int pa0 = 2 + (hamil.n_sites << 1) + m * m * 0;
                    int pa1 = 2 + (hamil.n_sites << 1) + m * m * 1;
                    int pad0 = 2 + (hamil.n_sites << 1) + m * m * 2;
                    int pad1 = 2 + (hamil.n_sites << 1) + m * m * 3;
                    int pb0 = 2 + (hamil.n_sites << 1) + m * m * 4;
                    int pb1 = 2 + (hamil.n_sites << 1) + m * m * 5;
                    // C
                    for (uint8_t j = 0; j < m; j++)
                        mat[{pc + j, p + j}] = i_op;
                    mat[{pi, p + m}] = c_op[m];
                    p += m + 1;
                    // D
                    for (uint8_t j = 0; j < m; j++)
                        mat[{pd + j, p + j}] = i_op;
                    mat[{pi, p + m}] = d_op[m];
                    p += m + 1;
                    // RD
                    for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                        mat[{prd + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = trd_op[i];
                        for (uint8_t k = 0; k < m; k++) {
                            mat[{pd + k, p + i - (m + 1)}] =
                                2.0 * ((-0.5) * pd_op[i][k][0] +
                                       (-0.5 * sqrt(3)) * pd_op[i][k][1]);
                            mat[{pc + k, p + i - (m + 1)}] =
                                2.0 * ((0.5) * q_op[k][i][0] +
                                       (-0.5 * sqrt(3)) * q_op[k][i][1]);
                        }
                        for (uint8_t j = 0; j < m; j++)
                            for (uint8_t l = 0; l < m; l++) {
                                double f0 =
                                    hamil.v(i, j, m, l) + hamil.v(i, l, m, j);
                                double f1 =
                                    hamil.v(i, j, m, l) - hamil.v(i, l, m, j);
                                mat[{pa0 + j * m + l, p + i - (m + 1)}] =
                                    f0 * (-0.5) * d_op[m];
                                mat[{pa1 + j * m + l, p + i - (m + 1)}] =
                                    f1 * (0.5 * sqrt(3)) * d_op[m];
                            }
                        for (uint8_t k = 0; k < m; k++)
                            for (uint8_t l = 0; l < m; l++) {
                                double f = 2.0 * hamil.v(i, m, k, l) -
                                           hamil.v(i, l, k, m);
                                mat[{pb0 + l * m + k, p + i - (m + 1)}] =
                                    f * c_op[m];
                            }
                        for (uint8_t j = 0; j < m; j++)
                            for (uint8_t k = 0; k < m; k++) {
                                double f = hamil.v(i, j, k, m) * sqrt(3);
                                mat[{pb1 + j * m + k, p + i - (m + 1)}] =
                                    f * c_op[m];
                            }
                    }
                    p += hamil.n_sites - (m + 1);
                    // R
                    for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                        mat[{pr + i, p + i - (m + 1)}] = i_op;
                        mat[{pi, p + i - (m + 1)}] = tr_op[i];
                        for (uint8_t k = 0; k < m; k++) {
                            mat[{pc + k, p + i - (m + 1)}] =
                                2.0 * ((-0.5) * p_op[i][k][0] +
                                       (0.5 * sqrt(3)) * p_op[i][k][1]);
                            mat[{pd + k, p + i - (m + 1)}] =
                                2.0 * ((0.5) * q_op[i][k][0] +
                                       (0.5 * sqrt(3)) * q_op[i][k][1]);
                        }
                        for (uint8_t j = 0; j < m; j++)
                            for (uint8_t l = 0; l < m; l++) {
                                double f0 =
                                    hamil.v(i, j, m, l) + hamil.v(i, l, m, j);
                                double f1 =
                                    hamil.v(i, j, m, l) - hamil.v(i, l, m, j);
                                mat[{pad0 + j * m + l, p + i - (m + 1)}] =
                                    f0 * (-0.5) * c_op[m];
                                mat[{pad1 + j * m + l, p + i - (m + 1)}] =
                                    f1 * (-0.5 * sqrt(3)) * c_op[m];
                            }
                        for (uint8_t k = 0; k < m; k++)
                            for (uint8_t l = 0; l < m; l++) {
                                double f = 2.0 * hamil.v(i, m, k, l) -
                                           hamil.v(i, l, k, m);
                                mat[{pb0 + k * m + l, p + i - (m + 1)}] =
                                    f * d_op[m];
                            }
                        for (uint8_t j = 0; j < m; j++)
                            for (uint8_t k = 0; k < m; k++) {
                                double f =
                                    (-1.0) * hamil.v(i, j, k, m) * sqrt(3);
                                mat[{pb1 + k * m + j, p + i - (m + 1)}] =
                                    f * d_op[m];
                            }
                    }
                    p += hamil.n_sites - (m + 1);
                    // A
                    for (uint8_t s = 0; s < 2; s++) {
                        int pa = s ? pa1 : pa0;
                        for (uint8_t i = 0; i < m; i++)
                            for (uint8_t j = 0; j < m; j++)
                                mat[{pa + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{pc + i, p + i * (m + 1) + m}] = c_op[m];
                            mat[{pc + i, p + m * (m + 1) + i}] =
                                s ? mc_op[m] : c_op[m];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = a_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // AD
                    for (uint8_t s = 0; s < 2; s++) {
                        int pad = s ? pad1 : pad0;
                        for (uint8_t i = 0; i < m; i++)
                            for (uint8_t j = 0; j < m; j++)
                                mat[{pad + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{pd + i, p + i * (m + 1) + m}] =
                                s ? md_op[m] : d_op[m];
                            mat[{pd + i, p + m * (m + 1) + i}] = d_op[m];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = ad_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    // B
                    for (uint8_t s = 0; s < 2; s++) {
                        int pb = s ? pb1 : pb0;
                        for (uint8_t i = 0; i < m; i++)
                            for (uint8_t j = 0; j < m; j++)
                                mat[{pb + i * m + j, p + i * (m + 1) + j}] =
                                    i_op;
                        for (uint8_t i = 0; i < m; i++) {
                            mat[{pc + i, p + i * (m + 1) + m}] = d_op[m];
                            mat[{pd + i, p + m * (m + 1) + i}] =
                                s ? mc_op[m] : c_op[m];
                        }
                        mat[{pi, p + m * (m + 1) + m}] = b_op[m][m][s];
                        p += (m + 1) * (m + 1);
                    }
                    assert(p == mat.n);
                }
                break;
            case QCTypes::CN:
                if (m == hamil.n_sites - 1) {
                    for (uint8_t s = 0; s < 2; s++)
                        mat[{p + s, 0}] = a_op[m][m][s];
                    p += 2;
                    for (uint8_t s = 0; s < 2; s++)
                        mat[{p + s, 0}] = ad_op[m][m][s];
                    p += 2;
                    for (uint8_t s = 0; s < 2; s++)
                        mat[{p + s, 0}] = b_op[m][m][s];
                    p += 2;
                    assert(p == mat.m);
                } else {
                    if (m != 0) {
                        mat[{1, 0}] = h_op;
                        mat[{1, 1}] = i_op;
                        p = 2;
                        mat[{1, p + m}] = c_op[m];
                        p += m + 1;
                        mat[{1, p + m}] = d_op[m];
                        p += m + 1;
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            mat[{1, p + j - m - 1}] = trd_op[j];
                        p += hamil.n_sites - m - 1;
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            mat[{1, p + j - m - 1}] = tr_op[j];
                        p += hamil.n_sites - m - 1;
                    }
                    vector<double> su2_factor{-0.5, -0.5 * sqrt(3)};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                mat[{!!m, p + k - m - 1}] =
                                    su2_factor[s] * p_op[j][k][s];
                            p += hamil.n_sites - m - 1;
                        }
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                mat[{!!m, p + k - m - 1}] =
                                    su2_factor[s] * pd_op[j][k][s];
                            p += hamil.n_sites - m - 1;
                        }
                    su2_factor = {1.0, sqrt(3)};
                    for (uint8_t s = 0; s < 2; s++)
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                mat[{!!m, p + k - m - 1}] =
                                    su2_factor[s] * q_op[j][k][s];
                            p += hamil.n_sites - m - 1;
                        }
                    assert(p == mat.n);
                }
                if (m != 0 && m != hamil.n_sites - 1) {
                    mat[{0, 0}] = i_op;
                    p = 2;
                    // pointers
                    int mm = hamil.n_sites - m - 1;
                    int pm = hamil.n_sites - m;
                    int pi = 0, pr = 2, prd = 2 + m + 1;
                    int pd = 2 + m + m + 2 - m - 1,
                        pc = 2 + m + 1 + hamil.n_sites - m - 1;
                    int pa0 = 2 + (hamil.n_sites << 1) + mm * mm * 0;
                    int pa1 = 2 + (hamil.n_sites << 1) + mm * mm * 1;
                    int pad0 = 2 + (hamil.n_sites << 1) + mm * mm * 2;
                    int pad1 = 2 + (hamil.n_sites << 1) + mm * mm * 3;
                    int pb0 = 2 + (hamil.n_sites << 1) + mm * mm * 4;
                    int pb1 = 2 + (hamil.n_sites << 1) + mm * mm * 5;
                    // R
                    for (uint8_t i = 0; i < m; i++) {
                        mat[{p + i, pi}] = tr_op[i];
                        mat[{p + i, pr + i}] = i_op;
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            for (uint8_t l = m + 1; l < hamil.n_sites; l++) {
                                double f0 =
                                    hamil.v(i, j, m, l) + hamil.v(i, l, m, j);
                                double f1 =
                                    hamil.v(i, j, m, l) - hamil.v(i, l, m, j);
                                mat[{p + i, pad0 + (j - m - 1) * mm + l - m -
                                                1}] = f0 * (-0.5) * c_op[m];
                                mat[{p + i,
                                     pad1 + (j - m - 1) * mm + l - m - 1}] =
                                    f1 * (0.5 * sqrt(3)) * c_op[m];
                            }
                        for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                            for (uint8_t l = m + 1; l < hamil.n_sites; l++) {
                                double f = 2.0 * hamil.v(i, m, k, l) -
                                           hamil.v(i, l, k, m);
                                mat[{p + i, pb0 + (k - m - 1) * mm + l - m -
                                                1}] = f * d_op[m];
                            }
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                                double f = hamil.v(i, j, k, m) * sqrt(3);
                                mat[{p + i, pb1 + (k - m - 1) * mm + j - m -
                                                1}] = f * d_op[m];
                            }
                        for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                            mat[{p + i, pc + k}] =
                                2.0 * ((-0.5) * p_op[k][i][0] +
                                       (0.5 * sqrt(3)) * p_op[k][i][1]);
                            mat[{p + i, pd + k}] =
                                2.0 * ((0.5) * q_op[i][k][0] +
                                       (-0.5 * sqrt(3)) * q_op[i][k][1]);
                        }
                    }
                    p += m;
                    // RD
                    for (uint8_t i = 0; i < m; i++) {
                        mat[{p + i, pi}] = trd_op[i];
                        mat[{p + i, prd + i}] = i_op;
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            for (uint8_t l = m + 1; l < hamil.n_sites; l++) {
                                double f0 =
                                    hamil.v(i, j, m, l) + hamil.v(i, l, m, j);
                                double f1 =
                                    hamil.v(i, j, m, l) - hamil.v(i, l, m, j);
                                mat[{p + i, pa0 + (j - m - 1) * mm + l - m -
                                                1}] = f0 * (-0.5) * d_op[m];
                                mat[{p + i,
                                     pa1 + (j - m - 1) * mm + l - m - 1}] =
                                    f1 * (-0.5 * sqrt(3)) * d_op[m];
                            }
                        for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                            for (uint8_t l = m + 1; l < hamil.n_sites; l++) {
                                double f = 2.0 * hamil.v(i, m, k, l) -
                                           hamil.v(i, l, k, m);
                                mat[{p + i, pb0 + (l - m - 1) * mm + k - m -
                                                1}] = f * c_op[m];
                            }
                        for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                            for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                                double f =
                                    (-1.0) * hamil.v(i, j, k, m) * sqrt(3);
                                mat[{p + i, pb1 + (j - m - 1) * mm + k - m -
                                                1}] = f * c_op[m];
                            }
                        for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                            mat[{p + i, pd + k}] =
                                2.0 * ((-0.5) * pd_op[k][i][0] +
                                       (-0.5 * sqrt(3)) * pd_op[k][i][1]);
                            mat[{p + i, pc + k}] =
                                2.0 * ((0.5) * q_op[k][i][0] +
                                       (0.5 * sqrt(3)) * q_op[k][i][1]);
                        }
                    }
                    p += m;
                    // D
                    mat[{p + m - m, pi}] = d_op[m];
                    for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                        mat[{p + j - m, pd + j}] = i_op;
                    p += hamil.n_sites - m;
                    // C
                    mat[{p + m - m, pi}] = c_op[m];
                    for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                        mat[{p + j - m, pc + j}] = i_op;
                    p += hamil.n_sites - m;
                    // A
                    for (uint8_t s = 0; s < 2; s++) {
                        int pa = s ? pa1 : pa0;
                        mat[{p + (m - m) * pm + m - m, pi}] = a_op[m][m][s];
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{p + (m - m) * pm + i - m, pc + i}] = c_op[m];
                            mat[{p + (i - m) * pm + m - m, pc + i}] =
                                s ? mc_op[m] : c_op[m];
                        }
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{p + (i - m) * pm + j - m,
                                     pa + (i - m - 1) * mm + j - m - 1}] = i_op;
                        p += pm * pm;
                    }
                    // AD
                    for (uint8_t s = 0; s < 2; s++) {
                        int pad = s ? pad1 : pad0;
                        mat[{p + (m - m) * pm + m - m, pi}] = ad_op[m][m][s];
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{p + (m - m) * pm + i - m, pd + i}] =
                                s ? md_op[m] : d_op[m];
                            mat[{p + (i - m) * pm + m - m, pd + i}] = d_op[m];
                        }
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{p + (i - m) * pm + j - m,
                                     pad + (i - m - 1) * mm + j - m - 1}] =
                                    i_op;
                        p += pm * pm;
                    }
                    // B
                    for (uint8_t s = 0; s < 2; s++) {
                        int pb = s ? pb1 : pb0;
                        mat[{p + (m - m) * pm + m - m, pi}] = b_op[m][m][s];
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++) {
                            mat[{p + (m - m) * pm + i - m, pd + i}] = c_op[m];
                            mat[{p + (i - m) * pm + m - m, pc + i}] =
                                s ? md_op[m] : d_op[m];
                        }
                        for (uint8_t i = m + 1; i < hamil.n_sites; i++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                                mat[{p + (i - m) * pm + j - m,
                                     pb + (i - m - 1) * mm + j - m - 1}] = i_op;
                        p += pm * pm;
                    }
                    assert(p == mat.m);
                }
                break;
            case QCTypes::NC | QCTypes::CN:
                assert(false);
                break;
            }
            shared_ptr<OperatorTensor<S>> opt = nullptr;
            if (mode != QCTypes::Conventional ||
                !(m == trans_l + 1 && m == trans_r - 1)) {
                opt = make_shared<OperatorTensor<S>>();
                opt->lmat = opt->rmat = pmat;
            } else if (!repeat_m) {
                opt = make_shared<OperatorTensor<S>>();
                opt->rmat = pmat;
            } else {
                opt = this->tensors.back();
                this->tensors.pop_back();
                opt->lmat = pmat;
            }
            // operator names
            if (opt->lmat == pmat) {
                shared_ptr<SymbolicRowVector<S>> plop;
                if (m == hamil.n_sites - 1)
                    plop = make_shared<SymbolicRowVector<S>>(1);
                else
                    plop = make_shared<SymbolicRowVector<S>>(rshape);
                SymbolicRowVector<S> &lop = *plop;
                lop[0] = h_op;
                if (m != hamil.n_sites - 1) {
                    lop[1] = i_op;
                    p = 2;
                    for (uint8_t j = 0; j < m + 1; j++)
                        lop[p + j] = c_op[j];
                    p += m + 1;
                    for (uint8_t j = 0; j < m + 1; j++)
                        lop[p + j] = d_op[j];
                    p += m + 1;
                    for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                        lop[p + j - (m + 1)] = trd_op[j];
                    p += hamil.n_sites - (m + 1);
                    for (uint8_t j = m + 1; j < hamil.n_sites; j++)
                        lop[p + j - (m + 1)] = tr_op[j];
                    p += hamil.n_sites - (m + 1);
                    vector<double> su2_factor;
                    switch (effective_mode) {
                    case QCTypes::NC:
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = 0; j < m + 1; j++) {
                                for (uint8_t k = 0; k < m + 1; k++)
                                    lop[p + k] = a_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = 0; j < m + 1; j++) {
                                for (uint8_t k = 0; k < m + 1; k++)
                                    lop[p + k] = ad_op[j][k][s];
                                p += m + 1;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = 0; j < m + 1; j++) {
                                for (uint8_t k = 0; k < m + 1; k++)
                                    lop[p + k] = b_op[j][k][s];
                                p += m + 1;
                            }
                        break;
                    case QCTypes::CN:
                        su2_factor = {-0.5, -0.5 * sqrt(3)};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    lop[p + k - m - 1] =
                                        su2_factor[s] * p_op[j][k][s];
                                p += hamil.n_sites - m - 1;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    lop[p + k - m - 1] =
                                        su2_factor[s] * pd_op[j][k][s];
                                p += hamil.n_sites - m - 1;
                            }
                        su2_factor = {1.0, sqrt(3)};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                                for (uint8_t k = m + 1; k < hamil.n_sites; k++)
                                    lop[p + k - m - 1] =
                                        su2_factor[s] * q_op[j][k][s];
                                p += hamil.n_sites - m - 1;
                            }
                        break;
                    case QCTypes::NC | QCTypes::CN:
                        assert(false);
                        break;
                    }
                    assert(p == rshape);
                }
                this->left_operator_names.push_back(plop);
            }
            if (opt->rmat == pmat) {
                shared_ptr<SymbolicColumnVector<S>> prop;
                if (m == 0)
                    prop = make_shared<SymbolicColumnVector<S>>(1);
                else
                    prop = make_shared<SymbolicColumnVector<S>>(lshape);
                SymbolicColumnVector<S> &rop = *prop;
                if (m == 0)
                    rop[0] = h_op;
                else {
                    rop[0] = i_op;
                    rop[1] = h_op;
                    p = 2;
                    for (uint8_t j = 0; j < m; j++)
                        rop[p + j] = tr_op[j];
                    p += m;
                    for (uint8_t j = 0; j < m; j++)
                        rop[p + j] = trd_op[j];
                    p += m;
                    for (uint8_t j = m; j < hamil.n_sites; j++)
                        rop[p + j - m] = d_op[j];
                    p += hamil.n_sites - m;
                    for (uint8_t j = m; j < hamil.n_sites; j++)
                        rop[p + j - m] = c_op[j];
                    p += hamil.n_sites - m;
                    vector<double> su2_factor;
                    switch (effective_mode) {
                    case QCTypes::NC:
                        su2_factor = {-0.5, -0.5 * sqrt(3)};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = 0; j < m; j++) {
                                for (uint8_t k = 0; k < m; k++)
                                    rop[p + k] = su2_factor[s] * p_op[j][k][s];
                                p += m;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = 0; j < m; j++) {
                                for (uint8_t k = 0; k < m; k++)
                                    rop[p + k] = su2_factor[s] * pd_op[j][k][s];
                                p += m;
                            }
                        su2_factor = {1.0, sqrt(3)};
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = 0; j < m; j++) {
                                for (uint8_t k = 0; k < m; k++)
                                    rop[p + k] = su2_factor[s] * q_op[j][k][s];
                                p += m;
                            }
                        break;
                    case QCTypes::CN:
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = m; j < hamil.n_sites; j++) {
                                for (uint8_t k = m; k < hamil.n_sites; k++)
                                    rop[p + k - m] = a_op[j][k][s];
                                p += hamil.n_sites - m;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = m; j < hamil.n_sites; j++) {
                                for (uint8_t k = m; k < hamil.n_sites; k++)
                                    rop[p + k - m] = ad_op[j][k][s];
                                p += hamil.n_sites - m;
                            }
                        for (uint8_t s = 0; s < 2; s++)
                            for (uint8_t j = m; j < hamil.n_sites; j++) {
                                for (uint8_t k = m; k < hamil.n_sites; k++)
                                    rop[p + k - m] = b_op[j][k][s];
                                p += hamil.n_sites - m;
                            }
                        break;
                    case QCTypes::NC | QCTypes::CN:
                        assert(false);
                        break;
                    }
                    assert(p == lshape);
                }
                this->right_operator_names.push_back(prop);
            }
            if (mode == QCTypes::Conventional && m == trans_l + 1 &&
                m == trans_r - 1 && !repeat_m) {
                repeat_m = true;
                m--;
                this->tensors.push_back(opt);
                continue;
            }
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
        if (mode == QCTypes(QCTypes::NC | QCTypes::CN) ||
            mode == QCTypes::Conventional) {
            uint8_t m;
            MPO<S>::schemer = make_shared<MPOSchemer<S>>(trans_l, trans_r);
            // left transform
            m = trans_l;
            int new_rshape =
                2 + 2 * hamil.n_sites +
                6 * (hamil.n_sites - m - 1) * (hamil.n_sites - m - 1);
            MPO<S>::schemer->left_new_operator_names =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            MPO<S>::schemer->left_new_operator_exprs =
                make_shared<SymbolicRowVector<S>>(new_rshape);
            SymbolicRowVector<S> &lop =
                *MPO<S>::schemer->left_new_operator_names;
            SymbolicRowVector<S> &lexpr =
                *MPO<S>::schemer->left_new_operator_exprs;
            for (int i = 0; i < 2 + 2 * hamil.n_sites; i++)
                lop[i] = this->left_operator_names[m]->data[i];
            p = 2 + 2 * hamil.n_sites;
            vector<shared_ptr<OpExpr<S>>> exprs;
            vector<double> su2_factor_p = {-0.5, -0.5 * sqrt(3)};
            vector<double> su2_factor_q = {1.0, sqrt(3)};
            for (uint8_t s = 0; s < 2; s++)
                for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                    for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                        lop[p + k - m - 1] = su2_factor_p[s] * p_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = 0; g < m + 1; g++)
                            for (uint8_t h = 0; h < m + 1; h++)
                                if (abs(hamil.v(j, g, k, h)) > TINY)
                                    exprs.push_back(
                                        (su2_factor_p[s] * hamil.v(j, g, k, h) *
                                         (s ? -1 : 1)) *
                                        ad_op[g][h][s]);
                        lexpr[p + k - m - 1] = sum(exprs);
                    }
                    p += hamil.n_sites - m - 1;
                }
            for (uint8_t s = 0; s < 2; s++)
                for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                    for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                        lop[p + k - m - 1] = su2_factor_p[s] * pd_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = 0; g < m + 1; g++)
                            for (uint8_t h = 0; h < m + 1; h++)
                                if (abs(hamil.v(j, g, k, h)) > TINY)
                                    exprs.push_back(
                                        (su2_factor_p[s] * hamil.v(j, g, k, h) *
                                         (s ? -1 : 1)) *
                                        a_op[g][h][s]);
                        lexpr[p + k - m - 1] = sum(exprs);
                    }
                    p += hamil.n_sites - m - 1;
                }
            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                    lop[p + k - m - 1] = su2_factor_q[0] * q_op[j][k][0];
                    exprs.clear();
                    for (uint8_t g = 0; g < m + 1; g++)
                        for (uint8_t h = 0; h < m + 1; h++)
                            if (abs(2 * hamil.v(j, k, g, h) -
                                    hamil.v(j, h, g, k)) > TINY)
                                exprs.push_back((su2_factor_q[0] *
                                                 (2 * hamil.v(j, k, g, h) -
                                                  hamil.v(j, h, g, k))) *
                                                b_op[g][h][0]);
                    lexpr[p + k - m - 1] = sum(exprs);
                }
                p += hamil.n_sites - m - 1;
            }
            for (uint8_t j = m + 1; j < hamil.n_sites; j++) {
                for (uint8_t k = m + 1; k < hamil.n_sites; k++) {
                    lop[p + k - m - 1] = su2_factor_q[1] * q_op[j][k][1];
                    exprs.clear();
                    for (uint8_t g = 0; g < m + 1; g++)
                        for (uint8_t h = 0; h < m + 1; h++)
                            if (abs(hamil.v(j, h, g, k)) > TINY)
                                exprs.push_back(
                                    (su2_factor_q[1] * hamil.v(j, h, g, k)) *
                                    b_op[g][h][1]);
                    lexpr[p + k - m - 1] = sum(exprs);
                }
                p += hamil.n_sites - m - 1;
            }
            assert(p == new_rshape);
            // right transform
            m = trans_r - 1;
            int new_lshape = 2 + 2 * hamil.n_sites + 6 * (m + 1) * (m + 1);
            MPO<S>::schemer->right_new_operator_names =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            MPO<S>::schemer->right_new_operator_exprs =
                make_shared<SymbolicColumnVector<S>>(new_lshape);
            SymbolicColumnVector<S> &rop =
                *MPO<S>::schemer->right_new_operator_names;
            SymbolicColumnVector<S> &rexpr =
                *MPO<S>::schemer->right_new_operator_exprs;
            for (int i = 0; i < 2 + 2 * hamil.n_sites; i++)
                rop[i] = this->right_operator_names[m + 1]->data[i];
            p = 2 + 2 * hamil.n_sites;
            for (uint8_t s = 0; s < 2; s++)
                for (uint8_t j = 0; j < m + 1; j++) {
                    for (uint8_t k = 0; k < m + 1; k++) {
                        rop[p + k] = su2_factor_p[s] * p_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                            for (uint8_t h = m + 1; h < hamil.n_sites; h++)
                                if (abs(hamil.v(j, g, k, h)) > TINY)
                                    exprs.push_back(
                                        (su2_factor_p[s] * hamil.v(j, g, k, h) *
                                         (s ? -1 : 1)) *
                                        ad_op[g][h][s]);
                        rexpr[p + k] = sum(exprs);
                    }
                    p += m + 1;
                }
            for (uint8_t s = 0; s < 2; s++)
                for (uint8_t j = 0; j < m + 1; j++) {
                    for (uint8_t k = 0; k < m + 1; k++) {
                        rop[p + k] = su2_factor_p[s] * pd_op[j][k][s];
                        exprs.clear();
                        for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                            for (uint8_t h = m + 1; h < hamil.n_sites; h++)
                                if (abs(hamil.v(j, g, k, h)) > TINY)
                                    exprs.push_back(
                                        (su2_factor_p[s] * hamil.v(j, g, k, h) *
                                         (s ? -1 : 1)) *
                                        a_op[g][h][s]);
                        rexpr[p + k] = sum(exprs);
                    }
                    p += m + 1;
                }
            for (uint8_t j = 0; j < m + 1; j++) {
                for (uint8_t k = 0; k < m + 1; k++) {
                    rop[p + k] = su2_factor_q[0] * q_op[j][k][0];
                    exprs.clear();
                    for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                        for (uint8_t h = m + 1; h < hamil.n_sites; h++)
                            if (abs(2 * hamil.v(j, k, g, h) -
                                    hamil.v(j, h, g, k)) > TINY)
                                exprs.push_back((su2_factor_q[0] *
                                                 (2 * hamil.v(j, k, g, h) -
                                                  hamil.v(j, h, g, k))) *
                                                b_op[g][h][0]);
                    rexpr[p + k] = sum(exprs);
                }
                p += m + 1;
            }
            for (uint8_t j = 0; j < m + 1; j++) {
                for (uint8_t k = 0; k < m + 1; k++) {
                    rop[p + k] = su2_factor_q[1] * q_op[j][k][1];
                    exprs.clear();
                    for (uint8_t g = m + 1; g < hamil.n_sites; g++)
                        for (uint8_t h = m + 1; h < hamil.n_sites; h++)
                            if (abs(hamil.v(j, h, g, k)) > TINY)
                                exprs.push_back(
                                    (su2_factor_q[1] * hamil.v(j, h, g, k)) *
                                    b_op[g][h][1]);
                    rexpr[p + k] = sum(exprs);
                }
                p += m + 1;
            }
            assert(p == new_lshape);
        }
    }
    void deallocate() override {
        for (uint8_t m = MPO<S>::n_sites - 1; m < MPO<S>::n_sites; m--)
            for (auto it = this->tensors[m]->ops.crbegin();
                 it != this->tensors[m]->ops.crend(); ++it) {
                OpElement<S> &op =
                    *dynamic_pointer_cast<OpElement<S>>(it->first);
                if (op.name == OpNames::R || op.name == OpNames::RD ||
                    op.name == OpNames::H)
                    it->second->deallocate();
            }
    }
};

template <typename, typename = void> struct PDM1MPOQC;

template <typename S> struct PDM1MPOQC<S, typename S::is_su2_t> : MPO<S> {
    PDM1MPOQC(const Hamiltonian<S> &hamil) : MPO<S>(hamil.n_sites) {
        const auto n_sites = MPO<S>::n_sites;
        shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), hamil.vaccum);
        shared_ptr<OpElement<S>> zero_op =
            make_shared<OpElement<S>>(OpNames::Zero, SiteIndex(), hamil.vaccum);
        shared_ptr<OpExpr<S>> c_op[n_sites], d_op[n_sites];
        shared_ptr<OpExpr<S>> b_op[n_sites][n_sites];
        shared_ptr<OpExpr<S>> pdm1_op[n_sites][n_sites];
        for (uint8_t m = 0; m < n_sites; m++) {
            c_op[m] = make_shared<OpElement<S>>(OpNames::C, SiteIndex(m),
                                                S(1, 1, hamil.orb_sym[m]));
            d_op[m] = make_shared<OpElement<S>>(OpNames::D, SiteIndex(m),
                                                S(-1, 1, hamil.orb_sym[m]));
        }
        for (uint8_t i = 0; i < n_sites; i++)
            for (uint8_t j = 0; j < n_sites; j++) {
                b_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::B, SiteIndex(i, j, 0),
                    S(0, 0, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
                pdm1_op[i][j] = make_shared<OpElement<S>>(
                    OpNames::PDM1, SiteIndex(i, j),
                    S(0, 0, hamil.orb_sym[i] ^ hamil.orb_sym[j]));
            }
        MPO<S>::const_e = 0.0;
        MPO<S>::op = zero_op;
        MPO<S>::schemer = nullptr;
        MPO<S>::tf = make_shared<TensorFunctions<S>>(hamil.opf);
        MPO<S>::site_op_infos = hamil.site_op_infos;
        for (uint8_t m = 0; m < n_sites; m++) {
            int lshape = m != n_sites - 1 ? 1 + 2 * (m + 1) : 1;
            int rshape = m != n_sites - 1 ? 1 : 3;
            // left operator names
            shared_ptr<SymbolicRowVector<S>> plop =
                make_shared<SymbolicRowVector<S>>(lshape);
            (*plop)[0] = i_op;
            if (m != n_sites - 1)
                for (uint8_t j = 0; j <= m; j++)
                    (*plop)[1 + j] = c_op[j],
                                (*plop)[1 + (m + 1) + j] = b_op[j][m];
            this->left_operator_names.push_back(plop);
            // right operator names
            shared_ptr<SymbolicColumnVector<S>> prop =
                make_shared<SymbolicColumnVector<S>>(rshape);
            (*prop)[0] = i_op;
            if (m == n_sites - 1)
                (*prop)[1] = b_op[m][m], (*prop)[2] = d_op[m];
            this->right_operator_names.push_back(prop);
            // middle operators
            if (m != n_sites - 1) {
                int mshape = m != n_sites - 2 ? 2 * m + 1 : 4 * (m + 1);
                shared_ptr<SymbolicColumnVector<S>> pmop =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                shared_ptr<SymbolicColumnVector<S>> pmexpr =
                    make_shared<SymbolicColumnVector<S>>(mshape);
                for (uint8_t j = 0; j <= m; j++) {
                    shared_ptr<OpExpr<S>> expr =
                        sqrt(2.0) * (b_op[j][m] * i_op);
                    (*pmop)[2 * j] = pdm1_op[m][j], (*pmexpr)[2 * j] = expr;
                    if (j != m)
                        (*pmop)[2 * j + 1] = pdm1_op[j][m],
                                        (*pmexpr)[2 * j + 1] = expr;
                }
                if (m == n_sites - 2) {
                    uint8_t p = 2 * m + 1;
                    for (uint8_t j = 0; j <= m; j++) {
                        shared_ptr<OpExpr<S>> expr =
                            sqrt(2.0) * (c_op[j] * d_op[m + 1]);
                        (*pmop)[p + 2 * j] = pdm1_op[j][m + 1];
                        (*pmop)[p + 2 * j + 1] = pdm1_op[m + 1][j];
                        (*pmexpr)[p + 2 * j] = (*pmexpr)[p + 2 * j + 1] = expr;
                    }
                    p += 2 * (m + 1);
                    (*pmop)[p] = pdm1_op[m + 1][m + 1];
                    (*pmexpr)[p] = sqrt(2.0) * (i_op * b_op[m + 1][m + 1]);
                }
                this->middle_operator_names.push_back(pmop);
                this->middle_operator_exprs.push_back(pmexpr);
            }
            // site tensors
            shared_ptr<OperatorTensor<S>> opt =
                make_shared<OperatorTensor<S>>();
            int llshape = 1 + 2 * m;
            int lrshape = m != n_sites - 1 ? 1 + 2 * (m + 1) : 1;
            shared_ptr<Symbolic<S>> plmat = nullptr, prmat = nullptr;
            if (m == 0)
                plmat = make_shared<SymbolicRowVector<S>>(lrshape);
            else if (m == n_sites - 1)
                plmat = make_shared<SymbolicColumnVector<S>>(llshape);
            else
                plmat = make_shared<SymbolicMatrix<S>>(llshape, lrshape);
            (*plmat)[{0, 0}] = i_op;
            if (m != n_sites - 1) {
                int pi = 0, pc = 1, p = 1;
                for (uint8_t i = 0; i < m; i++)
                    (*plmat)[{pc + i, p + i}] = i_op;
                (*plmat)[{pi, p + m}] = c_op[m];
                p += m + 1;
                for (uint8_t i = 0; i < m; i++)
                    (*plmat)[{pc + i, p + i}] = d_op[m];
                (*plmat)[{pi, p + m}] = b_op[m][m];
                p += m + 1;
                assert(p == lrshape);
            }
            if (m == n_sites - 1) {
                prmat = make_shared<SymbolicColumnVector<S>>(3);
                prmat->data[0] = i_op;
                prmat->data[1] = b_op[m][m];
                prmat->data[2] = d_op[m];
            } else {
                if (m == n_sites - 2)
                    prmat = make_shared<SymbolicMatrix<S>>(1, 3);
                else if (m == 0)
                    prmat = make_shared<SymbolicRowVector<S>>(1);
                else
                    prmat = make_shared<SymbolicMatrix<S>>(1, 1);
                (*prmat)[{0, 0}] = i_op;
            }
            opt->lmat = plmat, opt->rmat = prmat;
            hamil.filter_site_ops(m, {opt->lmat, opt->rmat}, opt->ops);
            this->tensors.push_back(opt);
        }
    }
    void deallocate() override {}
};

template <typename S> struct AncillaMPO : MPO<S> {
    int n_physical_sites;
    shared_ptr<MPO<S>> prim_mpo;
    AncillaMPO(const shared_ptr<MPO<S>> &mpo, bool npdm = false)
        : n_physical_sites(mpo->n_sites),
          prim_mpo(mpo), MPO<S>(mpo->n_sites << 1) {
        const auto n_sites = MPO<S>::n_sites;
        const shared_ptr<OpExpr<S>> i_op =
            make_shared<OpElement<S>>(OpNames::I, SiteIndex(), S());
        MPO<S>::const_e = mpo->const_e;
        MPO<S>::op = mpo->op;
        MPO<S>::tf = mpo->tf;
        MPO<S>::site_op_infos = mpo->site_op_infos;
        // operator names
        MPO<S>::left_operator_names.resize(n_sites, nullptr);
        MPO<S>::right_operator_names.resize(n_sites, nullptr);
        for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
            MPO<S>::left_operator_names[j] = mpo->left_operator_names[i];
            MPO<S>::left_operator_names[j + 1] =
                MPO<S>::left_operator_names[j]->copy();
            MPO<S>::right_operator_names[j] = mpo->right_operator_names[i];
            if (j - 1 >= 0)
                MPO<S>::right_operator_names[j - 1] =
                    MPO<S>::right_operator_names[j]->copy();
        }
        MPO<S>::right_operator_names[n_sites - 1] =
            make_shared<SymbolicColumnVector<S>>(1);
        MPO<S>::right_operator_names[n_sites - 1]->data[0] = i_op;
        // middle operators
        if (mpo->middle_operator_names.size() != 0) {
            assert(mpo->schemer == nullptr);
            MPO<S>::middle_operator_names.resize(n_sites - 1);
            MPO<S>::middle_operator_exprs.resize(n_sites - 1);
            shared_ptr<SymbolicColumnVector<S>> zero_mat =
                make_shared<SymbolicColumnVector<S>>(1);
            (*zero_mat)[0] =
                make_shared<OpElement<S>>(OpNames::Zero, SiteIndex(), S());
            shared_ptr<SymbolicColumnVector<S>> zero_expr =
                make_shared<SymbolicColumnVector<S>>(1);
            (*zero_expr)[0] = make_shared<OpExpr<S>>();
            for (int i = 0, j = 0; i < n_physical_sites - 1; i++, j += 2) {
                MPO<S>::middle_operator_names[j] =
                    mpo->middle_operator_names[i];
                MPO<S>::middle_operator_exprs[j] =
                    mpo->middle_operator_exprs[i];
                if (!npdm) {
                    MPO<S>::middle_operator_names[j + 1] =
                        mpo->middle_operator_names[i];
                    MPO<S>::middle_operator_exprs[j + 1] =
                        mpo->middle_operator_exprs[i];
                } else {
                    MPO<S>::middle_operator_names[j + 1] = zero_mat;
                    MPO<S>::middle_operator_exprs[j + 1] = zero_expr;
                }
            }
            if (mpo->op != nullptr && mpo->op->name != OpNames::Zero) {
                shared_ptr<SymbolicColumnVector<S>> hop_mat =
                    make_shared<SymbolicColumnVector<S>>(1);
                (*hop_mat)[0] = mpo->op;
                shared_ptr<SymbolicColumnVector<S>> hop_expr =
                    make_shared<SymbolicColumnVector<S>>(1);
                (*hop_expr)[0] = (shared_ptr<OpExpr<S>>)mpo->op * i_op;
                MPO<S>::middle_operator_names[n_sites - 2] = hop_mat;
                MPO<S>::middle_operator_exprs[n_sites - 2] = hop_expr;
            } else {
                MPO<S>::middle_operator_names[n_sites - 2] = zero_mat;
                MPO<S>::middle_operator_exprs[n_sites - 2] = zero_expr;
            }
        }
        // operator tensors
        MPO<S>::tensors.resize(n_sites, nullptr);
        for (int i = 0, j = 0; i < n_physical_sites; i++, j += 2) {
            MPO<S>::tensors[j + 1] = make_shared<OperatorTensor<S>>();
            if (j + 1 != n_sites - 1) {
                MPO<S>::tensors[j] = mpo->tensors[i];
                int rshape = MPO<S>::tensors[j]->lmat->n;
                MPO<S>::tensors[j + 1]->lmat = MPO<S>::tensors[j + 1]->rmat =
                    make_shared<SymbolicMatrix<S>>(rshape, rshape);
                for (int k = 0; k < rshape; k++)
                    (*MPO<S>::tensors[j + 1]->lmat)[{k, k}] = i_op;
                if (mpo->tensors[i]->lmat != mpo->tensors[i]->rmat &&
                    !(mpo->schemer != nullptr &&
                      mpo->schemer->right_trans_site -
                              mpo->schemer->left_trans_site ==
                          2)) {
                    int lshape = mpo->tensors[i + 1]->rmat->m;
                    MPO<S>::tensors[j + 1]->rmat =
                        make_shared<SymbolicMatrix<S>>(lshape, lshape);
                    for (int k = 0; k < lshape; k++)
                        (*MPO<S>::tensors[j + 1]->rmat)[{k, k}] = i_op;
                }
            } else {
                int lshape = mpo->tensors[i]->lmat->m;
                MPO<S>::tensors[j] = make_shared<OperatorTensor<S>>();
                MPO<S>::tensors[j]->lmat = MPO<S>::tensors[j]->rmat =
                    make_shared<SymbolicMatrix<S>>(lshape, 1);
                for (int k = 0; k < lshape; k++)
                    (*MPO<S>::tensors[j]->lmat)[{k, 0}] =
                        mpo->tensors[i]->lmat->data[k];
                if (mpo->tensors[i]->lmat != mpo->tensors[i]->rmat) {
                    lshape = mpo->tensors[i]->rmat->m;
                    MPO<S>::tensors[j]->rmat =
                        make_shared<SymbolicMatrix<S>>(lshape, 1);
                    for (int k = 0; k < lshape; k++)
                        (*MPO<S>::tensors[j]->rmat)[{k, 0}] =
                            mpo->tensors[i]->rmat->data[k];
                }
                MPO<S>::tensors[j]->ops = mpo->tensors[i]->ops;
                MPO<S>::tensors[j + 1]->lmat = MPO<S>::tensors[j + 1]->rmat =
                    make_shared<SymbolicColumnVector<S>>(1);
                MPO<S>::tensors[j + 1]->lmat->data[0] = i_op;
            }
            MPO<S>::tensors[j + 1]->ops[i_op] =
                MPO<S>::tensors[j]->ops.at(i_op);
        }
        // numerical transform
        if (mpo->schemer != nullptr &&
            mpo->schemer->right_trans_site - mpo->schemer->left_trans_site ==
                2) {
            MPO<S>::schemer = mpo->schemer->copy();
            if (n_physical_sites & 1) {
                MPO<S>::schemer->left_trans_site = n_physical_sites - 2;
                MPO<S>::schemer->right_trans_site = n_physical_sites;
            } else {
                MPO<S>::schemer->left_trans_site = n_physical_sites - 1;
                MPO<S>::schemer->right_trans_site = n_physical_sites + 1;
            }
        } else if (mpo->schemer != nullptr)
            assert(false);
        else
            MPO<S>::schemer = nullptr;
    }
    void deallocate() override { prim_mpo->deallocate(); }
};

template <typename S> struct NoTransposeRule : Rule<S> {
    shared_ptr<Rule<S>> prim_rule;
    NoTransposeRule(const shared_ptr<Rule<S>> &rule) : prim_rule(rule) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        shared_ptr<OpElementRef<S>> r = prim_rule->operator()(op);
        return r == nullptr || r->trans ? nullptr : r;
    }
};

template <typename, typename = void> struct RuleQC;

template <typename S> struct RuleQC<S, typename S::is_sz_t> : Rule<S> {
    uint8_t mask;
    const static uint8_t D = 0U, R = 1U, A = 2U, P = 3U, B = 4U, Q = 5U;
    RuleQC(bool d = true, bool r = true, bool a = true, bool p = true,
           bool b = true, bool q = true)
        : mask((d << D) | (r << R) | (a << A) | (p << P) | (b << B) |
               (q << Q)) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        switch (op->name) {
        case OpNames::D:
            return (mask & (1 << D)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::C, op->site_index,
                                               -op->q_label, op->factor),
                                           true, 1)
                                     : nullptr;
        case OpNames::RD:
            return (mask & (1 << R)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::R, op->site_index,
                                               -op->q_label, op->factor),
                                           true, 1)
                                     : nullptr;
        case OpNames::A:
            return (mask & (1 << A)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(OpNames::A,
                                                       op->site_index.flip(),
                                                       op->q_label, op->factor),
                             false, -1)
                       : nullptr;
        case OpNames::AD:
            return (mask & (1 << A))
                       ? (op->site_index[0] >= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A, op->site_index,
                                        -op->q_label, op->factor),
                                    true, 1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A, op->site_index.flip(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::P:
            return (mask & (1 << P)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(OpNames::P,
                                                       op->site_index.flip(),
                                                       op->q_label, op->factor),
                             false, -1)
                       : nullptr;
        case OpNames::PD:
            return (mask & (1 << P))
                       ? (op->site_index[0] >= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P, op->site_index,
                                        -op->q_label, op->factor),
                                    true, 1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P, op->site_index.flip(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::B:
            return (mask & (1 << B)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::B, op->site_index.flip(),
                                 -op->q_label, op->factor),
                             true, 1)
                       : nullptr;
        case OpNames::Q:
            return (mask & (1 << Q)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::Q, op->site_index.flip(),
                                 -op->q_label, op->factor),
                             true, 1)
                       : nullptr;
        default:
            return nullptr;
        }
    }
};

template <typename S> struct RuleQC<S, typename S::is_su2_t> : Rule<S> {
    uint8_t mask;
    const static uint8_t D = 0U, R = 1U, A = 2U, P = 3U, B = 4U, Q = 5U;
    RuleQC(bool d = true, bool r = true, bool a = true, bool p = true,
           bool b = true, bool q = true)
        : mask((d << D) | (r << R) | (a << A) | (p << P) | (b << B) |
               (q << Q)) {}
    shared_ptr<OpElementRef<S>>
    operator()(const shared_ptr<OpElement<S>> &op) const override {
        switch (op->name) {
        case OpNames::D:
            return (mask & (1 << D)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::C, op->site_index,
                                               -op->q_label, op->factor),
                                           true, 1)
                                     : nullptr;
        case OpNames::RD:
            return (mask & (1 << R)) ? make_shared<OpElementRef<S>>(
                                           make_shared<OpElement<S>>(
                                               OpNames::R, op->site_index,
                                               -op->q_label, op->factor),
                                           true, -1)
                                     : nullptr;
        case OpNames::A:
            return (mask & (1 << A)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::A, op->site_index.flip_spatial(),
                                 op->q_label, op->factor),
                             false, op->site_index.s() ? -1 : 1)
                       : nullptr;
        case OpNames::AD:
            return (mask & (1 << A))
                       ? (op->site_index[0] >= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A, op->site_index,
                                        -op->q_label, op->factor),
                                    true, op->site_index.s() ? 1 : -1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::A,
                                        op->site_index.flip_spatial(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::P:
            return (mask & (1 << P)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::P, op->site_index.flip_spatial(),
                                 op->q_label, op->factor),
                             false, op->site_index.s() ? -1 : 1)
                       : nullptr;
        case OpNames::PD:
            return (mask & (1 << P))
                       ? (op->site_index[0] >= op->site_index[1]
                              ? make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P, op->site_index,
                                        -op->q_label, op->factor),
                                    true, op->site_index.s() ? 1 : -1)
                              : make_shared<OpElementRef<S>>(
                                    make_shared<OpElement<S>>(
                                        OpNames::P,
                                        op->site_index.flip_spatial(),
                                        -op->q_label, op->factor),
                                    true, -1))
                       : nullptr;
        case OpNames::B:
            return (mask & (1 << B)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::B, op->site_index.flip_spatial(),
                                 -op->q_label, op->factor),
                             true, op->site_index.s() ? -1 : 1)
                       : nullptr;
        case OpNames::Q:
            return (mask & (1 << Q)) && op->site_index[0] < op->site_index[1]
                       ? make_shared<OpElementRef<S>>(
                             make_shared<OpElement<S>>(
                                 OpNames::Q, op->site_index.flip_spatial(),
                                 -op->q_label, op->factor),
                             true, op->site_index.s() ? -1 : 1)
                       : nullptr;
        default:
            return nullptr;
        }
    }
};

} // namespace block2

#endif /* QUANTUM_HPP_ */
