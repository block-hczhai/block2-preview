
#ifndef QUANTUM_HPP_
#define QUANTUM_HPP_

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <execinfo.h>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

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

enum struct OpNames : uint8_t {
    H,
    I,
    N,
    NN,
    NUD,
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
    PDM1
};

inline ostream &operator<<(ostream &os, const OpNames c) {
    const static string repr[] = {"H",  "I", "N",  "NN",  "NUD", "C",
                                  "D",  "R", "RD", "A",   "AD",  "P",
                                  "PD", "B", "Q",  "PDM1"};
    os << repr[(uint8_t)c];
    return os;
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
            cout << "exceeding allowed memory" << endl;
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

extern StackAllocator<uint32_t> *ialloc;
extern StackAllocator<double> *dalloc;

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
    static mt19937 rng;
    static void rand_seed(unsigned i = 0) {
        rng = mt19937(
            i ? i : chrono::steady_clock::now().time_since_epoch().count());
    }
    // return a integer in [a, b)
    static int rand_int(int a, int b) {
        assert(b > a);
        return uniform_int_distribution<int>(a, b - 1)(rng);
    }
    // return a double in [a, b)
    static double rand_double(double a = 0, double b = 1) {
        assert(b > a);
        return uniform_real_distribution<double>(a, b)(rng);
    }
    template <typename T>
    static void fill_rand_double(const T &t, double a = 0, double b = 1) {
        uniform_real_distribution<double> distr(a, b);
        auto *ptr = t.data;
        for (long i = 0, n = t.get_len(); i < n; i++)
            ptr[i] =
                (typename remove_reference<decltype(ptr[i])>::type)distr(rng);
    }
};

struct CG {
    long double *sqrt_fact;
    int n_sf, n_twoj;
    CG() : n_sf(0), n_twoj(0), sqrt_fact(nullptr) {}
    CG(int n_sqrt_fact, int max_j) : n_sf(n_sqrt_fact), n_twoj(max_j) {}
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
    bool triangle(int tja, int tjb, int tjc) {
        return !((tja + tjb + tjc) & 1) && tjc <= tja + tjb &&
               tjc >= abs(tja - tjb);
    }
    long double sqrt_delta(int tja, int tjb, int tjc) {
        return sqrt_fact[(tja + tjb - tjc) >> 1] *
               sqrt_fact[(tja - tjb + tjc) >> 1] *
               sqrt_fact[(-tja + tjb + tjc) >> 1] /
               sqrt_fact[(tja + tjb + tjc + 2) >> 1];
    }
    long double cg(int tja, int tjb, int tjc, int tma, int tmb, int tmc) {
        return (((tmc + tja - tjb) & 2) ? -1 : 1) * sqrt(tjc + 1) *
               wigner_3j(tja, tjb, tjc, tma, tmb, -tmc);
    }
    // Albert Messiah, Quantum Mechanics. Vol 2. Eq. (C.21)
    // Also see Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_3j(int tja, int tjb, int tjc, int tma, int tmb,
                          int tmc) {
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
            (((tja - tjb - tmc) & 2) ? -1 : 1) * ((max_alpha & 1) ? -1 : 1) *
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
    // Also see Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_6j(int tja, int tjb, int tjc, int tjd, int tje,
                          int tjf) {
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
    // Also see Sebastian's CheMPS2 code Wigner.cpp
    long double wigner_9j(int tja, int tjb, int tjc, int tjd, int tje, int tjf,
                          int tjg, int tjh, int tji) {
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
};

struct SZLabel {
    uint32_t data;
    SZLabel() : data(0) {}
    SZLabel(uint32_t data) : data(data) {}
    SZLabel(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos << 8) | pg)) {}
    SZLabel(const SZLabel &other) : data(other.data) {}
    int n() const { return (int)(((int32_t)data) >> 24); }
    int twos() const { return (int)(int16_t)((data >> 8) & 0xFFU); }
    int pg() const { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data = (data & (~0xFFFF00U)) | ((uint32_t)((twos & 0xFFU) << 8));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    bool operator==(SZLabel other) const noexcept { return data == other.data; }
    bool operator!=(SZLabel other) const noexcept { return data != other.data; }
    bool operator<(SZLabel other) const noexcept { return data < other.data; }
    SZLabel operator-() const noexcept {
        return SZLabel((data & 0xFFU) | (((~data) + (1 << 8)) & (0xFF00U)) |
                       (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SZLabel operator-(SZLabel other) const noexcept { return *this + (-other); }
    SZLabel operator+(SZLabel other) const noexcept {
        return SZLabel((((data & 0xFF00FF00U) + (other.data & 0xFF00FF00U)) &
                        0xFF00FF00U) |
                       ((data & 0xFFU) ^ (other.data & 0xFFU)));
    }
    SZLabel operator[](int i) const noexcept { return *this; }
    SZLabel get_ket() const noexcept { return *this; }
    SZLabel get_bra(SZLabel dq) const noexcept { return *this + dq; }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const { return 1; }
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
    friend ostream &operator<<(ostream &os, SZLabel c) {
        os << c.to_str();
        return os;
    }
};

struct SpinLabel {
    uint32_t data;
    SpinLabel() : data(0) {}
    SpinLabel(uint32_t data) : data(data) {}
    SpinLabel(int n, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos << 16) | (twos << 8) | pg)) {}
    SpinLabel(int n, int twos_low, int twos, int pg)
        : data((uint32_t)((n << 24) | (twos_low << 16) | (twos << 8) | pg)) {}
    SpinLabel(const SpinLabel &other) : data(other.data) {}
    int n() const { return (int)(((int32_t)data) >> 24); }
    int twos() const { return (int)(int16_t)((data >> 8) & 0xFFU); }
    int twos_low() const { return (int)(int16_t)((data >> 16) & 0xFFU); }
    int pg() const { return (int)(data & 0xFFU); }
    void set_n(int n) { data = (data & 0xFFFFFFU) | ((uint32_t)(n << 24)); }
    void set_twos(int twos) {
        data = (data & (~0xFFFF00U)) | ((uint32_t)((twos & 0xFFU) << 16)) |
               ((uint32_t)((twos & 0xFFU) << 8));
    }
    void set_twos_low(int twos) {
        data = (data & (~0xFF0000U)) | ((uint32_t)((twos & 0xFFU) << 16));
    }
    void set_pg(int pg) { data = (data & (~0xFFU)) | ((uint32_t)pg); }
    bool operator==(SpinLabel other) const noexcept {
        return data == other.data;
    }
    bool operator!=(SpinLabel other) const noexcept {
        return data != other.data;
    }
    bool operator<(SpinLabel other) const noexcept { return data < other.data; }
    SpinLabel operator-() const noexcept {
        return SpinLabel((data & 0xFFFFFFU) |
                         (((~data) + (1 << 24)) & (~0xFFFFFFU)));
    }
    SpinLabel operator-(SpinLabel other) const noexcept {
        return *this + (-other);
    }
    SpinLabel operator+(SpinLabel other) const noexcept {
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
        return SpinLabel(add_data | min(sub_data_lr, sub_data_rl));
    }
    SpinLabel operator[](int i) const noexcept {
        return SpinLabel(((data + (i << 9)) & (~0xFF0000U)) |
                         (((data + (i << 9)) & 0xFF00U) << 8));
    }
    SpinLabel get_ket() const noexcept {
        return SpinLabel((data & 0xFF00FFFFU) | ((data & 0xFF00U) << 8));
    }
    SpinLabel get_bra(SpinLabel dq) const noexcept {
        return SpinLabel(((data & 0xFF000000U) + (dq.data & 0xFF000000U)) |
                         ((data & 0xFF0000U) >> 8) | (data & 0xFF0000U) |
                         ((data & 0xFFU) ^ (dq.data & 0xFFU)));
    }
    size_t hash() const noexcept { return (size_t)data; }
    int count() const {
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
    friend ostream &operator<<(ostream &os, SpinLabel c) {
        os << c.to_str();
        return os;
    }
};

enum struct OpTypes { Zero, Elem, Prod, Sum };

struct OpExpr {
    virtual const OpTypes get_type() const { return OpTypes::Zero; }
    bool operator==(const OpExpr &other) const { return true; }
};

struct OpElement : OpExpr {
    OpNames name;
    vector<uint8_t> site_index;
    double factor;
    SpinLabel q_label;
    OpElement(OpNames name, const vector<uint8_t> &site_index, double factor,
              SpinLabel q_label)
        : name(name), site_index(site_index), factor(factor), q_label(q_label) {
    }
    const OpTypes get_type() const override { return OpTypes::Elem; }
    OpElement abs() const { return OpElement(name, site_index, 1.0, q_label); }
    OpElement operator*(double d) const {
        return OpElement(name, site_index, factor * d, q_label);
    }
    bool operator==(const OpElement &other) const {
        return name == other.name && site_index == other.site_index &&
               factor == other.factor;
    }
    size_t hash() const noexcept {
        size_t h = (size_t)name;
        for (auto r : site_index)
            h ^= (size_t)r + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(factor) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
    friend ostream &operator<<(ostream &os, const OpElement &c) {
        if (c.factor != 1.0)
            os << "(" << c.factor << " " << c.abs() << ")";
        else if (c.site_index.size() == 0)
            os << c.name;
        else if (c.site_index.size() == 1)
            os << c.name << (int)c.site_index[0];
        else {
            os << c.name << "[ ";
            for (auto r : c.site_index)
                os << (int)r << " ";
            os << "]";
        }
        return os;
    }
};

struct OpString : OpExpr {
    double factor;
    vector<shared_ptr<OpElement>> ops;
    OpString(const vector<shared_ptr<OpElement>> &ops, double factor)
        : factor(factor), ops() {
        for (auto &elem : ops) {
            this->factor *= elem->factor;
            this->ops.push_back(make_shared<OpElement>(elem->abs()));
        }
    }
    const OpTypes get_type() const override { return OpTypes::Prod; }
    OpString abs() const { return OpString(ops, 1.0); }
    OpString operator*(double d) const { return OpString(ops, factor * d); }
    bool operator==(const OpString &other) const {
        if (ops.size() != other.ops.size() || factor != other.factor)
            return false;
        for (size_t i = 0; i < ops.size(); i++)
            if (!(*ops[i] == *other.ops[i]))
                return false;
        return true;
    }
    friend ostream &operator<<(ostream &os, const OpString &c) {
        if (c.factor != 1.0)
            os << "(" << c.factor << " " << c.abs() << ")";
        else {
            for (auto r : c.ops)
                os << *r << " ";
        }
        return os;
    }
};

struct OpSum : OpExpr {
    vector<shared_ptr<OpString>> strings;
    OpSum(const vector<shared_ptr<OpString>> &strings) : strings(strings) {}
    const OpTypes get_type() const override { return OpTypes::Sum; }
    OpSum operator*(double d) const {
        vector<shared_ptr<OpString>> strs;
        strs.reserve(strings.size());
        for (auto &r : strings)
            strs.push_back(make_shared<OpString>(*r * d));
        return OpSum(strs);
    }
    OpSum abs() const {
        vector<shared_ptr<OpString>> strs;
        strs.reserve(strings.size());
        for (auto &r : strings)
            strs.push_back(make_shared<OpString>(r->abs()));
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
    friend ostream &operator<<(ostream &os, const OpSum &c) {
        for (size_t i = 0; i < c.strings.size() - 1; i++)
            os << *c.strings[i] << " + ";
        os << *c.strings.back();
        return os;
    }
};

inline size_t hash_value(const shared_ptr<OpExpr> x) {
    assert(x->get_type() == OpTypes::Elem);
    return dynamic_pointer_cast<OpElement>(x)->hash();
}

inline shared_ptr<OpExpr> abs_value(const shared_ptr<OpExpr> x) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (x->get_type() == OpTypes::Elem)
        return make_shared<OpElement>(
            dynamic_pointer_cast<OpElement>(x)->abs());
    else if (x->get_type() == OpTypes::Prod)
        return make_shared<OpString>(dynamic_pointer_cast<OpString>(x)->abs());
    else if (x->get_type() == OpTypes::Sum)
        return make_shared<OpSum>(dynamic_pointer_cast<OpSum>(x)->abs());
}

inline string to_str(const shared_ptr<OpExpr> x) {
    stringstream ss;
    if (x->get_type() == OpTypes::Zero)
        ss << 0;
    else if (x->get_type() == OpTypes::Elem)
        ss << *dynamic_pointer_cast<OpElement>(x);
    else if (x->get_type() == OpTypes::Prod)
        ss << *dynamic_pointer_cast<OpString>(x);
    else if (x->get_type() == OpTypes::Sum)
        ss << *dynamic_pointer_cast<OpSum>(x);
    return ss.str();
}

inline bool operator==(const shared_ptr<OpExpr> a, const shared_ptr<OpExpr> b) {
    if (a->get_type() != b->get_type())
        return false;
    else if (a->get_type() == OpTypes::Zero)
        return *a == *b;
    else if (a->get_type() == OpTypes::Elem)
        return *dynamic_pointer_cast<OpElement>(a) ==
               *dynamic_pointer_cast<OpElement>(b);
    else if (a->get_type() == OpTypes::Prod)
        return *dynamic_pointer_cast<OpString>(a) ==
               *dynamic_pointer_cast<OpString>(b);
    else if (a->get_type() == OpTypes::Sum)
        return *dynamic_pointer_cast<OpSum>(a) ==
               *dynamic_pointer_cast<OpSum>(b);
    else
        return false;
}

inline const shared_ptr<OpExpr> operator+(const shared_ptr<OpExpr> a,
                                          const shared_ptr<OpExpr> b) {
    if (a->get_type() == OpTypes::Zero)
        return b;
    else if (b->get_type() == OpTypes::Zero)
        return a;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString>> strs;
            strs.push_back(make_shared<OpString>(
                vector<shared_ptr<OpElement>>(
                    1, dynamic_pointer_cast<OpElement>(a)),
                1.0));
            strs.push_back(make_shared<OpString>(
                vector<shared_ptr<OpElement>>(
                    1, dynamic_pointer_cast<OpElement>(b)),
                1.0));
            return make_shared<OpSum>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum>(b)->strings.size() + 1);
            strs.push_back(make_shared<OpString>(
                vector<shared_ptr<OpElement>>(
                    1, dynamic_pointer_cast<OpElement>(a)),
                1.0));
            for (auto &r : dynamic_pointer_cast<OpSum>(b)->strings)
                strs.push_back(r);
            return make_shared<OpSum>(strs);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum>(b)->strings.size() + 1);
            strs.push_back(dynamic_pointer_cast<OpString>(a));
            for (auto &r : dynamic_pointer_cast<OpSum>(b)->strings)
                strs.push_back(r);
            return make_shared<OpSum>(strs);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpString>> strs;
            strs.reserve(2);
            strs.push_back(dynamic_pointer_cast<OpString>(a));
            strs.push_back(dynamic_pointer_cast<OpString>(b));
            return make_shared<OpSum>(strs);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum>(a)->strings.size() + 1);
            for (auto &r : dynamic_pointer_cast<OpSum>(a)->strings)
                strs.push_back(r);
            strs.push_back(make_shared<OpString>(
                vector<shared_ptr<OpElement>>(
                    1, dynamic_pointer_cast<OpElement>(b)),
                1.0));
            return make_shared<OpSum>(strs);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpString>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum>(a)->strings.size() + 1);
            for (auto &r : dynamic_pointer_cast<OpSum>(a)->strings)
                strs.push_back(r);
            strs.push_back(dynamic_pointer_cast<OpString>(b));
            return make_shared<OpSum>(strs);
        } else if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString>> strs;
            strs.reserve(dynamic_pointer_cast<OpSum>(a)->strings.size() +
                         dynamic_pointer_cast<OpSum>(b)->strings.size());
            for (auto &r : dynamic_pointer_cast<OpSum>(a)->strings)
                strs.push_back(r);
            for (auto &r : dynamic_pointer_cast<OpSum>(b)->strings)
                strs.push_back(r);
            return make_shared<OpSum>(strs);
        }
    }
}

inline const shared_ptr<OpExpr> operator*(const shared_ptr<OpExpr> x,
                                          double d) {
    if (x->get_type() == OpTypes::Zero)
        return x;
    else if (d == 0.0)
        return make_shared<OpExpr>();
    else if (x->get_type() == OpTypes::Elem)
        return make_shared<OpElement>(*dynamic_pointer_cast<OpElement>(x) * d);
    else if (x->get_type() == OpTypes::Prod)
        return make_shared<OpString>(*dynamic_pointer_cast<OpString>(x) * d);
    else if (x->get_type() == OpTypes::Sum)
        return make_shared<OpSum>(*dynamic_pointer_cast<OpSum>(x) * d);
}

inline const shared_ptr<OpExpr> operator*(double d,
                                          const shared_ptr<OpExpr> x) {
    return x * d;
}

inline const shared_ptr<OpExpr> operator*(const shared_ptr<OpExpr> a,
                                          const shared_ptr<OpExpr> b) {
    if (a->get_type() == OpTypes::Zero)
        return a;
    else if (b->get_type() == OpTypes::Zero)
        return b;
    else if (a->get_type() == OpTypes::Elem) {
        if (b->get_type() == OpTypes::Sum) {
            vector<shared_ptr<OpString>> strs;
            for (auto &r : dynamic_pointer_cast<OpSum>(b)->strings) {
                vector<shared_ptr<OpElement>> ops;
                ops.reserve(r->ops.size() + 1);
                ops.push_back(dynamic_pointer_cast<OpElement>(a));
                for (auto &rr : r->ops)
                    ops.push_back(rr);
                strs.push_back(make_shared<OpString>(ops, r->factor));
            }
            return make_shared<OpSum>(strs);
        } else if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpElement>> ops;
            ops.push_back(dynamic_pointer_cast<OpElement>(a));
            ops.push_back(dynamic_pointer_cast<OpElement>(b));
            return make_shared<OpString>(ops, 1.0);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpElement>> ops;
            ops.reserve(dynamic_pointer_cast<OpString>(b)->ops.size() + 1);
            ops.push_back(dynamic_pointer_cast<OpElement>(a));
            for (auto &r : dynamic_pointer_cast<OpString>(b)->ops)
                ops.push_back(r);
            return make_shared<OpString>(
                ops, dynamic_pointer_cast<OpString>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Prod) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpElement>> ops;
            ops.reserve(dynamic_pointer_cast<OpString>(a)->ops.size() + 1);
            for (auto &r : dynamic_pointer_cast<OpString>(a)->ops)
                ops.push_back(r);
            ops.push_back(dynamic_pointer_cast<OpElement>(b));
            return make_shared<OpString>(
                ops, dynamic_pointer_cast<OpString>(a)->factor);
        } else if (b->get_type() == OpTypes::Prod) {
            vector<shared_ptr<OpElement>> ops;
            ops.reserve(dynamic_pointer_cast<OpString>(a)->ops.size() +
                        dynamic_pointer_cast<OpString>(b)->ops.size());
            for (auto &r : dynamic_pointer_cast<OpString>(a)->ops)
                ops.push_back(r);
            for (auto &r : dynamic_pointer_cast<OpString>(b)->ops)
                ops.push_back(r);
            return make_shared<OpString>(
                ops, dynamic_pointer_cast<OpString>(a)->factor *
                         dynamic_pointer_cast<OpString>(b)->factor);
        }
    } else if (a->get_type() == OpTypes::Sum) {
        if (b->get_type() == OpTypes::Elem) {
            vector<shared_ptr<OpString>> strs;
            for (auto &r : dynamic_pointer_cast<OpSum>(a)->strings) {
                vector<shared_ptr<OpElement>> ops;
                ops.reserve(r->ops.size() + 1);
                for (auto &rr : r->ops)
                    ops.push_back(rr);
                ops.push_back(dynamic_pointer_cast<OpElement>(b));
                strs.push_back(make_shared<OpString>(ops, r->factor));
            }
            return make_shared<OpSum>(strs);
        }
    }
}

} // namespace block2

namespace std {

template <> struct hash<block2::SZLabel> {
    size_t operator()(const block2::SZLabel &s) const noexcept {
        return s.hash();
    }
};

template <> struct less<block2::SZLabel> {
    bool operator()(const block2::SZLabel &lhs,
                    const block2::SZLabel &rhs) const noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SZLabel &a, block2::SZLabel &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::SpinLabel> {
    size_t operator()(const block2::SpinLabel &s) const noexcept {
        return s.hash();
    }
};

template <> struct less<block2::SpinLabel> {
    bool operator()(const block2::SpinLabel &lhs,
                    const block2::SpinLabel &rhs) const noexcept {
        return lhs < rhs;
    }
};

inline void swap(block2::SpinLabel &a, block2::SpinLabel &b) {
    a.data ^= b.data, b.data ^= a.data, a.data ^= b.data;
}

template <> struct hash<block2::OpElement> {
    size_t operator()(const block2::OpElement &s) const noexcept {
        return s.hash();
    }
};

} // namespace std

namespace block2 {

struct StateInfo {
    SpinLabel *quanta;
    uint16_t *n_states;
    int n_states_total, n;
    StateInfo() : quanta(0), n_states(0), n_states_total(0), n(0) {}
    StateInfo(SpinLabel q) {
        allocate(1);
        quanta[0] = q, n_states[0] = 1, n_states_total = 1;
    }
    // need length * 2
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) - (length >> 1));
        n = length;
        quanta = (SpinLabel *)ptr;
        n_states = (uint16_t *)(ptr + length);
    }
    void reallocate(int length) {
        uint32_t *ptr =
            ialloc->reallocate((uint32_t *)quanta, (n << 1) - (n >> 1),
                               (length << 1) - (length >> 1));
        if (ptr == (uint32_t *)quanta) {
            memcpy(quanta + length, n_states, length * sizeof(uint32_t));
            n_states = (uint16_t *)(quanta + length);
        } else {
            memcpy(ptr, quanta, length * sizeof(uint32_t));
            memcpy(ptr + length, n_states, length * sizeof(uint16_t));
            quanta = (SpinLabel *)ptr;
            n_states = (uint16_t *)(quanta + length);
        }
        n = length;
    }
    void deallocate() {
        assert(n != 0);
        ialloc->deallocate((uint32_t *)quanta, (n << 1) - (n >> 1));
        quanta = 0;
        n_states = 0;
    }
    StateInfo deep_copy() const {
        StateInfo other;
        other.allocate(n);
        copy_data_to(other);
        return other;
    }
    void copy_data_to(StateInfo &other) const {
        assert(other.n == n);
        memcpy(other.quanta, quanta, ((n << 1) - (n >> 1)) * sizeof(uint32_t));
    }
    void sort_states() {
        int idx[n];
        SpinLabel q[n];
        uint16_t nq[n];
        memcpy(q, quanta, n * sizeof(SpinLabel));
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
    void collect(SpinLabel target = 0x7FFFFFFF) {
        int k = -1;
        int nn = upper_bound(quanta, quanta + n, target) - quanta;
        for (int i = 0; i < nn; i++)
            if (n_states[i] == 0)
                continue;
            else if (k != -1 && quanta[i] == quanta[k])
                n_states[k] += n_states[i];
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
    int find_state(SpinLabel q) const {
        auto p = lower_bound(quanta, quanta + n, q);
        if (p == quanta + n || *p != q)
            return -1;
        else
            return p - quanta;
    }
    static StateInfo tensor_product(const StateInfo &a, const StateInfo &b,
                                    SpinLabel target) {
        int nc = 0;
        for (int i = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++)
                nc += (a.quanta[i] + b.quanta[j]).count();
        StateInfo c;
        c.allocate(nc);
        for (int i = 0, ic = 0; i < a.n; i++)
            for (int j = 0; j < b.n; j++) {
                SpinLabel qc = a.quanta[i] + b.quanta[j];
                for (int k = 0; k < qc.count(); k++) {
                    c.quanta[ic + k] = qc[k];
                    int nprod = (int)a.n_states[i] * (int)b.n_states[j];
                    c.n_states[ic + k] = (uint16_t)min(nprod, 65535);
                }
                ic += qc.count();
            }
        c.sort_states();
        c.collect(target);
        return c;
    }
    static StateInfo filter(StateInfo &a, StateInfo &b, SpinLabel target) {
        a.n_states_total = 0;
        for (int i = 0; i < a.n; i++) {
            SpinLabel qb = target - a.quanta[i];
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
            SpinLabel qa = target - b.quanta[i];
            int x = 0;
            for (int k = 0; k < qa.count(); k++) {
                int idx = a.find_state(qa[k]);
                x += idx == -1 ? 0 : a.n_states[idx];
            }
            b.n_states[i] = (uint16_t)min(x, (int)b.n_states[i]);
            b.n_states_total += b.n_states[i];
        }
    }
    friend ostream &operator<<(ostream &os, const StateInfo &c) {
        for (int i = 0; i < c.n; i++)
            os << c.quanta[i].to_str() << " : " << c.n_states[i] << endl;
        return os;
    }
};

struct SparseMatrixInfo {
    SpinLabel *quanta;
    uint16_t *n_states_bra, *n_states_ket;
    uint32_t *n_states_total;
    SpinLabel delta_quantum;
    bool is_fermion;
    int n;
    SparseMatrixInfo() : n(-1) {}
    void initialize(const StateInfo &bra, const StateInfo &ket, SpinLabel dq,
                    bool is_fermion) {
        this->is_fermion = is_fermion;
        delta_quantum = dq;
        vector<SpinLabel> qs;
        qs.reserve(ket.n);
        for (int i = 0; i < ket.n; i++) {
            SpinLabel q = ket.quanta[i];
            SpinLabel bs = dq + q;
            for (int k = 0; k < bs.count(); k++)
                if (bra.find_state(bs[k]) != -1) {
                    q.set_twos_low(bs[k].twos());
                    qs.push_back(q);
                }
        }
        n = qs.size();
        allocate(n);
        if (n != 0) {
            memcpy(quanta, &qs[0], n * sizeof(SpinLabel));
            sort(quanta, quanta + n);
            for (int i = 0; i < n; i++) {
                n_states_ket[i] =
                    ket.n_states[ket.find_state(quanta[i].get_ket())];
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
    void allocate(int length, uint32_t *ptr = 0) {
        if (ptr == 0)
            ptr = ialloc->allocate((length << 1) + length);
        quanta = (SpinLabel *)ptr;
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
};

struct SparseMatrix {
    SparseMatrixInfo info;
    double *data;
    double factor;
    size_t total_memory;
};

struct Hamiltonian {
    SpinLabel vaccum, target;
    StateInfo *basis;
    SparseMatrixInfo *op_basis[2];
    uint8_t n_sites, n_syms, *orb_sym;
    bool su2;
    Hamiltonian(SpinLabel vaccum, SpinLabel target, int norb, bool su2,
                uint8_t *orb_sym)
        : vaccum(vaccum), target(target), n_sites((uint8_t)norb), su2(su2) {
        assert((int)n_sites == norb);
        basis = new StateInfo[n_sites];
        this->orb_sym = new uint8_t[n_sites];
        memcpy(this->orb_sym, orb_sym, n_sites * sizeof(uint8_t));
        n_syms = *max_element(orb_sym, orb_sym + n_sites) + 1;
        int16_t basis_sym[n_syms];
        memset(basis_sym, (int16_t)(-1), sizeof(basis_sym));
        for (int i = 0; i < n_sites; i++)
            if (basis_sym[orb_sym[i]] == (int16_t)(-1))
                basis_sym[orb_sym[i]] = i;
        if (su2)
            for (int i = 0; i < n_sites; i++)
                if (basis_sym[orb_sym[i]] == i) {
                    basis[i].allocate(3);
                    basis[i].quanta[0] = vaccum;
                    basis[i].quanta[1] = SpinLabel(1, 1, orb_sym[i]);
                    basis[i].quanta[2] = SpinLabel(2, 0, 0);
                    basis[i].n_states[0] = basis[i].n_states[1] =
                        basis[i].n_states[2] = 1;
                    basis[i].sort_states();
                } else
                    basis[i] = basis[basis_sym[orb_sym[i]]];
        else
            for (int i = 0; i < n_sites; i++)
                if (basis_sym[orb_sym[i]] == i) {
                    basis[i].allocate(4);
                    basis[i].quanta[0] = vaccum;
                    basis[i].quanta[1] = SpinLabel(1, -1, orb_sym[i]);
                    basis[i].quanta[2] = SpinLabel(1, 1, orb_sym[i]);
                    basis[i].quanta[3] = SpinLabel(2, 0, 0);
                    basis[i].n_states[0] = basis[i].n_states[1] =
                        basis[i].n_states[2] = basis[i].n_states[3] = 1;
                    basis[i].sort_states();
                } else
                    basis[i] = basis[basis_sym[orb_sym[i]]];
        op_basis[0] = new SparseMatrixInfo[n_syms * 3];
        for (int i = 0; i < n_sites; i++)
            if (basis_sym[orb_sym[i]] == i) {
                op_basis[0][orb_sym[i]].initialize(basis[i], basis[i], vaccum,
                                                   false);
                op_basis[0][orb_sym[i] + n_syms].initialize(
                    basis[i], basis[i], SpinLabel(1, 1, orb_sym[i]), true);
                op_basis[0][orb_sym[i] + n_syms * 2].initialize(
                    basis[i], basis[i], SpinLabel(-1, 1, orb_sym[i]), true);
            }
    }
    ~Hamiltonian() {
        int16_t basis_sym[n_syms];
        memset(basis_sym, (int16_t)(-1), sizeof(basis_sym));
        for (int i = 0; i < n_sites; i++)
            if (basis_sym[orb_sym[i]] == (int16_t)(-1))
                basis_sym[orb_sym[i]] = i;
        for (int i = n_sites - 1; i >= 0; i--)
            if (basis_sym[orb_sym[i]] == i) {
                op_basis[0][i + (n_syms << 1)].deallocate();
                op_basis[0][i + n_syms].deallocate();
                op_basis[0][i].deallocate();
            }
        for (int i = n_sites - 1; i >= 0; i--)
            if (basis_sym[orb_sym[i]] == i)
                basis[i].deallocate();
        delete[] orb_sym;
        delete[] basis;
        delete[] op_basis[0];
    }
};

} // namespace block2

#endif /* QUANTUM_HPP_ */
