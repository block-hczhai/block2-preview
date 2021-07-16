
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

#include "fp_codec.hpp"
#include "integral.hpp"

using namespace std;

namespace block2 {

// Symmetric 2D array for storage of one-electron integrals
struct CompressedTInt {
    // Number of orbitals
    uint16_t n;
    shared_ptr<CompressedVector<double>> cps_data;
    bool general;
    CompressedTInt(uint16_t n, bool general = false)
        : n(n), cps_data(nullptr), general(general) {}
    uint32_t find_index(uint16_t i, uint16_t j) const {
        return general ? (uint32_t)i * n + j
                       : (i < j ? ((uint32_t)j * (j + 1) >> 1) + i
                                : ((uint32_t)i * (i + 1) >> 1) + j);
    }
    size_t size() const {
        return general ? (size_t)n * n : ((size_t)n * (n + 1) >> 1);
    }
    void clear() { cps_data->clear(); }
    double &operator()(uint16_t i, uint16_t j) {
        return (*cps_data)[find_index(i, j)];
    }
    double operator()(uint16_t i, uint16_t j) const {
        return (
            (const CompressedVector<double> &)(*cps_data))[find_index(i, j)];
    }
    void reorder(const CompressedTInt &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint16_t i = 0; i < n; i++)
            for (uint16_t j = 0; j < (general ? n : i + 1); j++)
                (*this)(i, j) = other(ord[i], ord[j]);
    }
    friend ostream &operator<<(ostream &os, CompressedTInt x) {
        os << fixed << setprecision(16);
        for (uint16_t i = 0; i < x.n; i++)
            for (uint16_t j = 0; j < (x.general ? x.n : i + 1); j++)
                if (x(i, j) != 0.0)
                    os << setw(20) << x(i, j) << setw(4) << i + 1 << setw(4)
                       << j + 1 << setw(4) << 0 << setw(4) << 0 << endl;
        return os;
    }
};

// General 4D array for storage of two-electron integrals
struct CompressedV1Int {
    // Number of orbitals
    uint32_t n;
    size_t m;
    shared_ptr<CompressedVector<double>> cps_data;
    CompressedV1Int(uint32_t n)
        : n(n), m((size_t)n * n * n * n), cps_data(nullptr) {}
    size_t size() const { return m; }
    void clear() { cps_data->clear(); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return (*cps_data)[(((size_t)i * n + j) * n + k) * n + l];
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return ((const CompressedVector<double>
                     &)(*cps_data))[(((size_t)i * n + j) * n + k) * n + l];
    }
    void reorder(const CompressedV1Int &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < n; j++)
                for (uint32_t k = 0; k < n; k++)
                    for (uint32_t l = 0; l < n; l++)
                        (*this)(i, j, k, l) =
                            other(ord[i], ord[j], ord[k], ord[l]);
    }
    friend ostream &operator<<(ostream &os, CompressedV1Int x) {
        os << fixed << setprecision(16);
        for (uint32_t i = 0; i < x.n; i++)
            for (uint32_t j = 0; j < x.n; j++)
                for (uint32_t k = 0; k < x.n; k++)
                    for (uint32_t l = 0; l < x.n; l++)
                        if (x(i, j, k, l) != 0.0)
                            os << setw(20) << x(i, j, k, l) << setw(4) << i + 1
                               << setw(4) << j + 1 << setw(4) << k + 1
                               << setw(4) << l + 1 << endl;
        return os;
    }
};

// 4D array with 4-fold symmetry for storage of two-electron integrals
// [ijkl] = [jikl] = [jilk] = [ijlk]
struct CompressedV4Int {
    // n: number of orbitals
    uint32_t n, m;
    shared_ptr<CompressedVector<double>> cps_data;
    CompressedV4Int(uint32_t n)
        : n(n), m(n * (n + 1) >> 1), cps_data(nullptr) {}
    size_t find_index(uint32_t i, uint32_t j) const {
        return i < j ? ((size_t)j * (j + 1) >> 1) + i
                     : ((size_t)i * (i + 1) >> 1) + j;
    }
    size_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        size_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return p * m + q;
    }
    size_t size() const { return (size_t)m * m; }
    void clear() { cps_data->clear(); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return (*cps_data)[find_index(i, j, k, l)];
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return ((const CompressedVector<double>
                     &)(*cps_data))[find_index(i, j, k, l)];
    }
    void reorder(const CompressedV4Int &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j <= i; j++)
                for (uint32_t k = 0; k < n; k++)
                    for (uint32_t l = 0; l <= k; l++)
                        (*this)(i, j, k, l) =
                            other(ord[i], ord[j], ord[k], ord[l]);
    }
    friend ostream &operator<<(ostream &os, CompressedV4Int x) {
        os << fixed << setprecision(16);
        for (uint32_t i = 0; i < x.n; i++)
            for (uint32_t j = 0; j <= i; j++)
                for (uint32_t k = 0; k < x.n; k++)
                    for (uint32_t l = 0; l <= k; l++)
                        if (x(i, j, k, l) != 0.0)
                            os << setw(20) << x(i, j, k, l) << setw(4) << i + 1
                               << setw(4) << j + 1 << setw(4) << k + 1
                               << setw(4) << l + 1 << endl;
        return os;
    }
};

// 4D array with 8-fold symmetry for storage of two-electron integrals
// [ijkl] = [jikl] = [jilk] = [ijlk] = [klij] = [klji] = [lkji] = [lkij]
struct CompressedV8Int {
    // n: number of orbitals
    uint32_t n, m;
    shared_ptr<CompressedVector<double>> cps_data;
    CompressedV8Int(uint32_t n)
        : n(n), m(n * (n + 1) >> 1), cps_data(nullptr) {}
    size_t find_index(uint32_t i, uint32_t j) const {
        return i < j ? ((size_t)j * (j + 1) >> 1) + i
                     : ((size_t)i * (i + 1) >> 1) + j;
    }
    size_t find_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        uint32_t p = (uint32_t)find_index(i, j), q = (uint32_t)find_index(k, l);
        return find_index(p, q);
    }
    size_t size() const { return ((size_t)m * (m + 1) >> 1); }
    void clear() { cps_data->clear(); }
    double &operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
        return (*cps_data)[find_index(i, j, k, l)];
    }
    double operator()(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
        return ((const CompressedVector<double>
                     &)(*cps_data))[find_index(i, j, k, l)];
    }
    void reorder(const CompressedV8Int &other, const vector<uint16_t> &ord) {
        assert(n == other.n);
        for (uint32_t i = 0, ij = 0; i < n; i++)
            for (uint32_t j = 0; j <= i; j++, ij++)
                for (uint32_t k = 0, kl = 0; k <= i; k++)
                    for (uint32_t l = 0; l <= k; l++, kl++)
                        if (ij >= kl)
                            (*this)(i, j, k, l) =
                                other(ord[i], ord[j], ord[k], ord[l]);
    }
    friend ostream &operator<<(ostream &os, CompressedV8Int x) {
        os << fixed << setprecision(16);
        for (uint32_t i = 0, ij = 0; i < x.n; i++)
            for (uint32_t j = 0; j <= i; j++, ij++)
                for (uint32_t k = 0, kl = 0; k <= i; k++)
                    for (uint32_t l = 0; l <= k; l++, kl++)
                        if (ij >= kl && x(i, j, k, l) != 0.0)
                            os << setw(20) << x(i, j, k, l) << setw(4) << i + 1
                               << setw(4) << j + 1 << setw(4) << k + 1
                               << setw(4) << l + 1 << endl;
        return os;
    }
};

// One- and two-electron integrals
struct CompressedFCIDUMP : FCIDUMP {
    using FCIDUMP::const_e;
    using FCIDUMP::general;
    using FCIDUMP::params;
    using FCIDUMP::uhf;
    vector<CompressedTInt> cps_ts;
    vector<CompressedV8Int> cps_vs;
    vector<CompressedV4Int> cps_vabs;
    vector<CompressedV1Int> cps_vgs;
    double prec; // only useful when write into compressed array
    int ncache;
    size_t chunk_size; // only useful when read from FCIDUMP
    CompressedFCIDUMP(double prec, int ncache = 5, size_t chunk_size = 128)
        : FCIDUMP(), prec(prec), ncache(ncache), chunk_size(chunk_size) {}
    // Initialize integrals: U(1) case
    // Two-electron integrals can be three general rank-4 arrays
    // or 8-fold, 8-fold, 4-fold rank-1 arrays
    void initialize_sz(uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                       uint16_t isym, double e, istream &ta, size_t lta,
                       istream &tb, size_t ltb, istream &va, size_t lva,
                       istream &vb, size_t lvb, istream &vab, size_t lvab) {
        params.clear();
        cps_ts.clear();
        cps_vs.clear();
        cps_vabs.clear();
        cps_vgs.clear();
        this->const_e = e;
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_elec);
        params["ms2"] = Parsing::to_string(twos);
        params["isym"] = Parsing::to_string(isym);
        params["iuhf"] = "1";
        cps_ts.push_back(CompressedTInt(n_sites));
        cps_ts.push_back(CompressedTInt(n_sites));
        if (lta != cps_ts[0].size())
            cps_ts[0].general = cps_ts[1].general = true;
        assert(lta == cps_ts[0].size() && ltb == cps_ts[1].size());
        cps_vs.push_back(CompressedV8Int(n_sites));
        cps_vs.push_back(CompressedV8Int(n_sites));
        cps_vabs.push_back(CompressedV4Int(n_sites));
        if (vs[0].size() == lva) {
            assert(cps_vs[1].size() == lvb);
            assert(cps_vabs[0].size() == lvab);
            general = false;
            cps_vs[0].cps_data =
                make_shared<CompressedVector<double>>(va, lva, prec, ncache);
            cps_vs[1].cps_data =
                make_shared<CompressedVector<double>>(vb, lvb, prec, ncache);
            cps_vabs[0].cps_data =
                make_shared<CompressedVector<double>>(vab, lvab, prec, ncache);
        } else {
            general = true;
            cps_vs.clear();
            cps_vabs.clear();
            cps_vgs.push_back(CompressedV1Int(n_sites));
            cps_vgs.push_back(CompressedV1Int(n_sites));
            cps_vgs.push_back(CompressedV1Int(n_sites));
            assert(cps_vgs[0].size() == lva);
            assert(cps_vgs[1].size() == lvb);
            assert(cps_vgs[2].size() == lvab);
            cps_vgs[0].cps_data =
                make_shared<CompressedVector<double>>(va, lva, prec, ncache);
            cps_vgs[1].cps_data =
                make_shared<CompressedVector<double>>(vb, lvb, prec, ncache);
            cps_vgs[2].cps_data =
                make_shared<CompressedVector<double>>(vab, lvab, prec, ncache);
        }
        cps_ts[0].cps_data =
            make_shared<CompressedVector<double>>(ta, lta, prec, ncache);
        cps_ts[1].cps_data =
            make_shared<CompressedVector<double>>(tb, ltb, prec, ncache);
        uhf = true;
        freeze();
    }
    // Initialize integrals: SU(2) case
    // Two-electron integrals can be general rank-4 array or 8-fold rank-1 array
    void initialize_su2(uint16_t n_sites, uint16_t n_elec, uint16_t twos,
                        uint16_t isym, double e, istream &t, size_t lt,
                        istream &v, size_t lv) {
        params.clear();
        cps_ts.clear();
        cps_vs.clear();
        cps_vabs.clear();
        cps_vgs.clear();
        this->const_e = e;
        params["norb"] = Parsing::to_string(n_sites);
        params["nelec"] = Parsing::to_string(n_elec);
        params["ms2"] = Parsing::to_string(twos);
        params["isym"] = Parsing::to_string(isym);
        params["iuhf"] = "0";
        cps_ts.push_back(CompressedTInt(n_sites));
        if (lt != cps_ts[0].size())
            cps_ts[0].general = true;
        assert(lt == cps_ts[0].size());
        cps_vs.push_back(CompressedV8Int(n_sites));
        if (cps_vs[0].size() == lv) {
            general = false;
            cps_vs[0].cps_data =
                make_shared<CompressedVector<double>>(v, lv, prec, ncache);
        } else {
            general = true;
            cps_vs.clear();
            cps_vgs.push_back(CompressedV1Int(n_sites));
            assert(lv == cps_vgs[0].size());
            cps_vgs[0].cps_data =
                make_shared<CompressedVector<double>>(v, lv, prec, ncache);
        }
        cps_ts[0].cps_data =
            make_shared<CompressedVector<double>>(t, lt, prec, ncache);
        uhf = false;
        freeze();
    }
    // Writing FCIDUMP file to disk
    void write(const string &filename) const override {
        ofstream ofs(filename.c_str());
        if (!ofs.good())
            throw runtime_error("FCIDUMP::write on '" + filename + "' failed.");
        ofs << " &FCI NORB=" << setw(4) << (int)n_sites()
            << ",NELEC=" << setw(4) << (int)n_elec() << ",MS2=" << setw(4)
            << (int)twos() << "," << endl;
        assert(params.count("orbsym") != 0);
        ofs << "  ORBSYM=" << params.at("orbsym") << "," << endl;
        ofs << "  ISYM=" << setw(4) << (int)isym() << "," << endl;
        if (uhf)
            ofs << "  IUHF=1," << endl;
        if (general)
            ofs << "  IGENERAL=1," << endl;
        if (cps_ts[0].general)
            ofs << "  ITGENERAL=1," << endl;
        ofs << " &END" << endl;
        auto write_const = [](ofstream &os, double x) {
            os << fixed << setprecision(16);
            os << setw(20) << x << setw(4) << 0 << setw(4) << 0 << setw(4) << 0
               << setw(4) << 0 << endl;
        };
        if (!uhf) {
            if (general)
                ofs << cps_vgs[0];
            else
                ofs << cps_vs[0];
            ofs << cps_ts[0];
            write_const(ofs, this->const_e);
        } else {
            if (general) {
                for (size_t i = 0; i < cps_vgs.size(); i++)
                    ofs << cps_vgs[i], write_const(ofs, 0.0);
            } else {
                for (size_t i = 0; i < cps_vs.size(); i++)
                    ofs << cps_vs[i], write_const(ofs, 0.0);
                ofs << cps_vabs[0], write_const(ofs, 0.0);
            }
            for (size_t i = 0; i < cps_ts.size(); i++)
                ofs << cps_ts[i], write_const(ofs, 0.0);
            write_const(ofs, this->const_e);
        }
        if (!ofs.good())
            throw runtime_error("FCIDUMP::write on '" + filename + "' failed.");
        ofs.close();
    }
    // Parsing a FCIDUMP file
    void read(const string &filename) override {
        params.clear();
        cps_ts.clear();
        cps_vs.clear();
        cps_vabs.clear();
        const_e = 0.0;
        ifstream ifs(filename.c_str());
        if (!ifs.good())
            throw runtime_error("FCIDUMP::read on '" + filename + "' failed.");
        vector<string> lines = Parsing::readlines(&ifs);
        if (ifs.bad())
            throw runtime_error("FCIDUMP::read on '" + filename + "' failed.");
        ifs.close();
        vector<string> pars;
        size_t il = 0;
        for (; il < lines.size(); il++) {
            string l(Parsing::lower(lines[il]));
            if (l.find("&fci") != string::npos)
                l.replace(l.find("&fci"), 4, "");
            if (l.find("/") != string::npos || l.find("&end") != string::npos)
                break;
            else
                pars.push_back(l);
        }
        il++;
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
        size_t int_sz = lines.size() > il ? lines.size() - il : 0;
        vector<array<uint16_t, 4>> int_idx(int_sz);
        vector<double> int_val(int_sz);
        int ntg = threading->activate_global();
#pragma omp parallel for schedule(static) num_threads(ntg)
        for (int64_t ill = 0; ill < (int64_t)int_sz; ill++) {
            string ll = Parsing::trim(lines[il + ill]);
            if (ll.length() == 0 || ll[0] == '!') {
                int_idx[ill][0] = numeric_limits<uint16_t>::max();
                continue;
            }
            vector<string> ls = Parsing::split(ll, " ", true);
            assert(ls.size() == 5);
            int_idx[ill] = array<uint16_t, 4>{(uint16_t)Parsing::to_int(ls[1]),
                                              (uint16_t)Parsing::to_int(ls[2]),
                                              (uint16_t)Parsing::to_int(ls[3]),
                                              (uint16_t)Parsing::to_int(ls[4])};
            int_val[ill] = Parsing::to_double(ls[0]);
        }
        threading->activate_normal();
        uint16_t n = (uint16_t)Parsing::to_int(params["norb"]);
        uhf = params.count("iuhf") != 0 && Parsing::to_int(params["iuhf"]) == 1;
        general = params.count("igeneral") != 0 &&
                  Parsing::to_int(params["igeneral"]) == 1;
        if (!uhf) {
            cps_ts.push_back(CompressedTInt(n));
            cps_ts[0].general = params.count("itgeneral") != 0 &&
                                Parsing::to_int(params["itgeneral"]) == 1;
            cps_ts[0].cps_data = make_shared<CompressedVector<double>>(
                cps_ts[0].size(), prec, chunk_size, ncache);
            cps_ts[0].clear();
            if (!general) {
                cps_vs.push_back(CompressedV8Int(n));
                cps_vs[0].cps_data = make_shared<CompressedVector<double>>(
                    cps_vs[0].size(), prec, chunk_size, ncache);
                cps_vs[0].clear();
            } else {
                cps_vgs.push_back(CompressedV1Int(n));
                cps_vgs[0].cps_data = make_shared<CompressedVector<double>>(
                    cps_vgs[0].size(), prec, chunk_size, ncache);
                cps_vgs[0].clear();
            }
            for (size_t i = 0; i < int_val.size(); i++) {
                if (int_idx[i][0] == numeric_limits<uint16_t>::max())
                    continue;
                if (int_idx[i][0] + int_idx[i][1] + int_idx[i][2] +
                        int_idx[i][3] ==
                    0)
                    const_e = int_val[i];
                else if (int_idx[i][2] + int_idx[i][3] == 0)
                    cps_ts[0](int_idx[i][0] - 1, int_idx[i][1] - 1) =
                        int_val[i];
                else if (!general)
                    cps_vs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                              int_idx[i][2] - 1, int_idx[i][3] - 1) =
                        int_val[i];
                else
                    cps_vgs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                               int_idx[i][2] - 1, int_idx[i][3] - 1) =
                        int_val[i];
            }
        } else {
            cps_ts.push_back(CompressedTInt(n));
            cps_ts.push_back(CompressedTInt(n));
            cps_ts[0].general = cps_ts[1].general =
                params.count("itgeneral") != 0 &&
                Parsing::to_int(params["itgeneral"]) == 1;
            cps_ts[0].cps_data = make_shared<CompressedVector<double>>(
                cps_ts[0].size(), prec, chunk_size, ncache);
            cps_ts[1].cps_data = make_shared<CompressedVector<double>>(
                cps_ts[1].size(), prec, chunk_size, ncache);
            cps_ts[0].clear(), cps_ts[1].clear();
            if (!general) {
                cps_vs.push_back(CompressedV8Int(n));
                cps_vs.push_back(CompressedV8Int(n));
                cps_vabs.push_back(CompressedV4Int(n));
                cps_vs[0].cps_data = make_shared<CompressedVector<double>>(
                    cps_vs[0].size(), prec, chunk_size, ncache);
                cps_vs[1].cps_data = make_shared<CompressedVector<double>>(
                    cps_vs[1].size(), prec, chunk_size, ncache);
                cps_vabs[0].cps_data = make_shared<CompressedVector<double>>(
                    cps_vabs[0].size(), prec, chunk_size, ncache);
                cps_vs[0].clear(), cps_vs[1].clear(), cps_vabs[0].clear();
            } else {
                for (int i = 0; i < 3; i++)
                    cps_vgs.push_back(CompressedV1Int(n));
                cps_vgs[0].cps_data = make_shared<CompressedVector<double>>(
                    cps_vgs[0].size(), prec, chunk_size, ncache);
                cps_vgs[1].cps_data = make_shared<CompressedVector<double>>(
                    cps_vgs[1].size(), prec, chunk_size, ncache);
                cps_vgs[2].cps_data = make_shared<CompressedVector<double>>(
                    cps_vgs[2].size(), prec, chunk_size, ncache);
                cps_vgs[0].clear(), cps_vgs[1].clear(), cps_vgs[2].clear();
            }
            int ip = 0;
            for (size_t i = 0; i < int_val.size(); i++) {
                if (int_idx[i][0] == numeric_limits<uint16_t>::max())
                    continue;
                if (int_idx[i][0] + int_idx[i][1] + int_idx[i][2] +
                        int_idx[i][3] ==
                    0) {
                    ip++;
                    if (ip == 6)
                        const_e = int_val[i];
                } else if (int_idx[i][2] + int_idx[i][3] == 0) {
                    cps_ts[ip - 3](int_idx[i][0] - 1, int_idx[i][1] - 1) =
                        int_val[i];
                } else {
                    assert(ip <= 2);
                    if (!general) {
                        if (ip < 2)
                            cps_vs[ip](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                       int_idx[i][2] - 1, int_idx[i][3] - 1) =
                                int_val[i];
                        else
                            cps_vabs[0](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                        int_idx[i][2] - 1, int_idx[i][3] - 1) =
                                int_val[i];
                    } else {
                        cps_vgs[ip](int_idx[i][0] - 1, int_idx[i][1] - 1,
                                    int_idx[i][2] - 1, int_idx[i][3] - 1) =
                            int_val[i];
                    }
                }
            }
        }
        freeze();
    }
    // Remove integral elements that violate point group symmetry
    // orbsym: in XOR convention
    double symmetrize(const vector<uint8_t> &orbsym) override {
        uint16_t n = n_sites();
        assert((int)orbsym.size() == n);
        double error = 0.0;
        unfreeze();
        for (auto &t : cps_ts)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < (t.general ? n : i + 1); j++)
                    if (orbsym[i] ^ orbsym[j])
                        error += abs(t(i, j)), t(i, j) = 0;
        for (auto &v : cps_vgs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l < n; l++)
                            if (orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l])
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : cps_vabs)
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++)
                    for (int k = 0; k < n; k++)
                        for (int l = 0; l <= k; l++)
                            if (orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l])
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        for (auto &v : cps_vs)
            for (int i = 0, ij = 0; i < n; i++)
                for (int j = 0; j <= i; j++, ij++)
                    for (int k = 0, kl = 0; k <= i; k++)
                        for (int l = 0; l <= k; l++, kl++)
                            if (ij >= kl &&
                                (orbsym[i] ^ orbsym[j] ^ orbsym[k] ^ orbsym[l]))
                                error += abs(v(i, j, k, l)), v(i, j, k, l) = 0;
        freeze();
        return error;
    }
    // orbital rotation
    // rot_mat: (old, new)
    void rotate(const vector<double> &rot_mat) override {
        throw runtime_error("Not implemented!");
    }
    void reorder(const vector<uint16_t> &ord) override {
        uint16_t n = n_sites();
        assert(ord.size() == n);
        vector<CompressedTInt> rts(cps_ts);
        vector<CompressedV1Int> rvgs(cps_vgs);
        vector<CompressedV4Int> rvabs(cps_vabs);
        vector<CompressedV8Int> rvs(cps_vs);
        for (size_t i = 0; i < cps_ts.size(); i++) {
            rts[i].cps_data = make_shared<CompressedVector<double>>(
                rts[i].size(), prec, chunk_size, ncache);
            rts[i].clear();
            rts[i].reorder(cps_ts[i], ord);
        }
        for (size_t i = 0; i < cps_vgs.size(); i++) {
            rvgs[i].cps_data = make_shared<CompressedVector<double>>(
                rvgs[i].size(), prec, chunk_size, ncache);
            rvgs[i].clear();
            rvgs[i].reorder(cps_vgs[i], ord);
        }
        for (size_t i = 0; i < cps_vabs.size(); i++) {
            rvabs[i].cps_data = make_shared<CompressedVector<double>>(
                rvabs[i].size(), prec, chunk_size, ncache);
            rvabs[i].clear();
            rvabs[i].reorder(cps_vabs[i], ord);
        }
        for (size_t i = 0; i < cps_vs.size(); i++) {
            rvs[i].cps_data = make_shared<CompressedVector<double>>(
                rvs[i].size(), prec, chunk_size, ncache);
            rvs[i].clear();
            rvs[i].reorder(cps_vs[i], ord);
        }
        cps_ts = rts, cps_vgs = rvgs, cps_vabs = rvabs, cps_vs = rvs;
        if (params.count("orbsym"))
            set_orb_sym(FCIDUMP::reorder(orb_sym(), ord));
        freeze();
    }
    // One-electron integral element (SU(2))
    double t(uint16_t i, uint16_t j) const override { return cps_ts[0](i, j); }
    // One-electron integral element (SZ)
    double t(uint8_t s, uint16_t i, uint16_t j) const override {
        return uhf ? cps_ts[s](i, j) : cps_ts[0](i, j);
    }
    // Two-electron integral element (SU(2))
    double v(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const override {
        return general ? cps_vgs[0](i, j, k, l) : cps_vs[0](i, j, k, l);
    }
    // Two-electron integral element (SZ)
    double v(uint8_t sl, uint8_t sr, uint16_t i, uint16_t j, uint16_t k,
             uint16_t l) const override {
        if (uhf) {
            if (sl == sr)
                return general ? cps_vgs[sl](i, j, k, l)
                               : cps_vs[sl](i, j, k, l);
            else if (sl == 0 && sr == 1)
                return general ? cps_vgs[2](i, j, k, l)
                               : cps_vabs[0](i, j, k, l);
            else
                return general ? cps_vgs[2](k, l, i, j)
                               : cps_vabs[0](k, l, i, j);
        } else
            return general ? cps_vgs[0](i, j, k, l) : cps_vs[0](i, j, k, l);
    }
    void freeze() {
        int ntg = threading->activate_global();
        for (auto &cs : cps_ts)
            cs.cps_data =
                make_shared<CompressedVectorMT<double>>(cs.cps_data, ntg);
        for (auto &cs : cps_vs)
            cs.cps_data =
                make_shared<CompressedVectorMT<double>>(cs.cps_data, ntg);
        for (auto &cs : cps_vabs)
            cs.cps_data =
                make_shared<CompressedVectorMT<double>>(cs.cps_data, ntg);
        for (auto &cs : cps_vgs)
            cs.cps_data =
                make_shared<CompressedVectorMT<double>>(cs.cps_data, ntg);
    }
    void unfreeze() {
        int ntg = threading->activate_global();
        for (auto &cs : cps_ts)
            if (dynamic_pointer_cast<CompressedVectorMT<double>>(cs.cps_data) !=
                nullptr)
                cs.cps_data = dynamic_pointer_cast<CompressedVectorMT<double>>(
                                  cs.cps_data)
                                  ->ref_cv;
        for (auto &cs : cps_vs)
            if (dynamic_pointer_cast<CompressedVectorMT<double>>(cs.cps_data) !=
                nullptr)
                cs.cps_data = dynamic_pointer_cast<CompressedVectorMT<double>>(
                                  cs.cps_data)
                                  ->ref_cv;
        for (auto &cs : cps_vabs)
            if (dynamic_pointer_cast<CompressedVectorMT<double>>(cs.cps_data) !=
                nullptr)
                cs.cps_data = dynamic_pointer_cast<CompressedVectorMT<double>>(
                                  cs.cps_data)
                                  ->ref_cv;
        for (auto &cs : cps_vgs)
            if (dynamic_pointer_cast<CompressedVectorMT<double>>(cs.cps_data) !=
                nullptr)
                cs.cps_data = dynamic_pointer_cast<CompressedVectorMT<double>>(
                                  cs.cps_data)
                                  ->ref_cv;
    }
    void deallocate() override {
        cps_ts.clear();
        cps_vs.clear();
        cps_vabs.clear();
        cps_vgs.clear();
    }
};

} // namespace block2
