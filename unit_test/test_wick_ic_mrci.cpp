
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

struct WickICMRCI {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    vector<pair<string, string>> sub_spaces;
    WickExpr h1, h2, h;
    WickICMRCI() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("mnxyijkl");
        idx_map[WickIndexTypes::Active] =
            WickIndex::parse_set("mnxyabcdefghpq");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("mnxyrstu");
        perm_map[make_pair("w", 4)] = WickPermutation::qc_phys();
        h1 = WickExpr::parse("SUM <mn> h[mn] E1[m,n]", idx_map, perm_map);
        h2 = WickExpr::parse("0.5 SUM <mnxy> w[mnxy] E2[mn,xy]", idx_map,
                             perm_map);
        h = h1 + h2;
        sub_spaces = {{"reference", ""},
                      {"ijrskltu+", "E1[r,i] E1[s,j] \n + E1[s,i] E1[r,j]"},
                      {"ijrskltu-", "E1[r,i] E1[s,j] \n - E1[s,i] E1[r,j]"},
                      {"rsiatukp+", "E1[r,i] E1[s,a] \n + E1[s,i] E1[r,a]"},
                      {"rsiatukp-", "E1[r,i] E1[s,a] \n - E1[s,i] E1[r,a]"},
                      {"ijrakltp+", "E1[r,j] E1[a,i] \n + E1[r,i] E1[a,j]"},
                      {"ijrakltp-", "E1[r,j] E1[a,i] \n - E1[r,i] E1[a,j]"},
                      {"rsabtupq+", "E1[r,b] E1[s,a] \n + E1[s,b] E1[r,a]"},
                      {"rsabtupq-", "E1[r,b] E1[s,a] \n - E1[s,b] E1[r,a]"},
                      {"ijabklpq+", "E1[b,i] E1[a,j] \n + E1[b,j] E1[a,i]"},
                      {"ijabklpq-", "E1[b,i] E1[a,j] \n - E1[b,j] E1[a,i]"},
                      {"irabktpq1", "E1[r,i] E1[a,b]"},
                      {"irabktpq2", "E1[a,i] E1[r,b]"},
                      {"rabctpqg", "E1[r,b] E1[a,c]"},
                      {"iabckpqg", "E1[b,i] E1[a,c]"}};
    }
    // only block diagonal term will use communicator
    WickExpr build_hamiltonian(const string &bra, const string &ket,
                               bool do_sum = true, bool do_comm = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map);
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map);
        WickExpr expr;
        if (bra == "" && ket == "")
            ;
        else if (bra == "")
            expr = (h * xket);
        else if (ket == "")
            expr = (xbra.conjugate() * h);
        else if (do_comm)
            expr = do_sum ? (xbra.conjugate() & (h ^ xket))
                          : (xbra.conjugate() * (h ^ xket));
        else
            expr = do_sum ? (xbra.conjugate() & (h * xket))
                          : (xbra.conjugate() * (h * xket));
        return expr.expand()
            .remove_external()
            .remove_inactive()
            .add_spin_free_trans_symm()
            .simplify();
    }
    WickExpr build_overlap(const string &bra, const string &ket,
                           bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map);
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map);
        WickExpr expr =
            do_sum ? (xbra.conjugate() & xket) : (xbra.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .remove_inactive()
            .simplify();
    }
    string to_einsum_zeros(const WickTensor &tensor) const {
        stringstream ss;
        ss << tensor.name << " = np.zeros((";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (wi.types == WickIndexTypes::Inactive)
                ss << "ncore, ";
            else if (wi.types == WickIndexTypes::Active)
                ss << "ncas, ";
            else if (wi.types == WickIndexTypes::External)
                ss << "nvirt, ";
        }
        ss << "))";
        return ss.str();
    }
    pair<string, bool>
    to_einsum_sum_restriction(const WickTensor &tensor,
                              string eq_pattern = "+") const {
        stringstream ss, sr;
        ss << "grid = np.indices((";
        bool first_and = false;
        bool has_idx = false;
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            ss << (wi.types == WickIndexTypes::Inactive
                       ? "ncore"
                       : (wi.types == WickIndexTypes::External ? "nvirt"
                                                               : "ncas"));
            if (i != (int)tensor.indices.size() - 1 || i == 0)
                ss << ", ";
            if (i != 0 && wi.types == tensor.indices[i - 1].types) {
                if (wi.name[0] != tensor.indices[i - 1].name[0] + 1)
                    continue;
                if (wi.types == WickIndexTypes::Active) {
                    if (i + 1 < (int)tensor.indices.size() &&
                        tensor.indices[i + 1].types == wi.types &&
                        tensor.indices[i + 1].name[0] == wi.name[0] + 1)
                        continue;
                    if (i - 2 >= 0 && tensor.indices[i - 2].types == wi.types &&
                        tensor.indices[i - 1].name[0] ==
                            tensor.indices[i - 2].name[0] + 1)
                        continue;
                    if (eq_pattern.length() == 2) {
                        if (eq_pattern[0] != '+' && eq_pattern[0] != '-' &&
                            i < tensor.indices.size() / 2)
                            continue;
                        else if (eq_pattern[1] != '+' && eq_pattern[1] != '-' &&
                                 i >= tensor.indices.size() / 2)
                            continue;
                    } else {
                        if (eq_pattern[0] != '+' && eq_pattern[0] != '-')
                            continue;
                    }
                }
                has_idx = true;
                sr << "idx " << (first_and ? "&" : "") << "= grid[" << i - 1
                   << "] <";
                if (eq_pattern.length() == 1)
                    sr << (eq_pattern[0] == '+' ? "=" : "");
                else if (i < tensor.indices.size() / 2)
                    sr << (eq_pattern[0] == '+' ? "=" : "");
                else
                    sr << (eq_pattern[1] == '+' ? "=" : "");
                sr << " grid[" << i << "]" << endl;
                first_and = true;
            }
        }
        ss << "))" << endl;
        return make_pair(ss.str() + sr.str(), has_idx);
    }
    string to_einsum() const {
        stringstream ss, sk;
        WickTensor hmat, deno, norm;
        ss << "xnorms = {}" << endl << endl;
        ss << "xhmats = {}" << endl << endl;
        sk << "keys = [" << endl;
        vector<string> norm_keys(sub_spaces.size());
        for (int i = 0; i < (int)sub_spaces.size(); i++) {
            string key = sub_spaces[i].first, ket_expr = sub_spaces[i].second;
            string mkey = key, skey = key;
            if (key.back() == '+')
                skey = key.substr(0, key.length() - 1), mkey = key;
            else if (key.back() == '-')
                skey = key.substr(0, key.length() - 1), mkey = key;
            else if (key.back() == '1')
                skey = key.substr(0, key.length() - 1), mkey = skey;
            else if (key.back() == '2')
                continue;
            norm_keys[i] = mkey;
            ss << "# compute : overlap " << mkey << endl << endl;
            sk << "    '" << mkey << "'," << endl;
            stringstream sr;
            if (mkey == "reference") {
                sr << "xn = np.ones((1, 1, 1))" << endl;
                sr << "xnorms['" << mkey << "'] = xn" << endl;
                ss << WickExpr::to_einsum_add_indent(sr.str(), 0) << endl;
                continue;
            }
            string nkey = skey;
            map<char, char> ket_bra_map;
            int pidx = (int)skey.length();
            for (int j = 4; j < (int)skey.length(); j++) {
                if (skey[j] == 'p')
                    pidx = j;
                // norm is only non-zero between diff act indices
                if (j >= pidx)
                    ket_bra_map[skey[j + 4 - skey.length()]] = skey[j];
            }
            nkey = skey.substr(0, 4) + skey.substr(pidx);
            int nact = (int)(skey.length() - pidx);
            string bra_expr = ket_expr;
            for (int j = 0; j < (int)bra_expr.length(); j++)
                if (ket_bra_map.count(bra_expr[j]))
                    bra_expr[j] = ket_bra_map[bra_expr[j]];
            norm = WickTensor::parse("norm[" + nkey + "]", idx_map, perm_map);
            sr << to_einsum_zeros(norm) << endl;
            sr << build_overlap(bra_expr, ket_expr, false).to_einsum(norm)
               << endl;
            bool restrict_cas = key.back() == '+' || key.back() == '-';
            bool non_ortho = key.back() == '1' || key.back() == '2';
            if (non_ortho) {
                string ket_expr_2 = sub_spaces[i + 1].second;
                string bra_expr_2 = ket_expr_2;
                for (int j = 0; j < (int)bra_expr_2.length(); j++)
                    if (ket_bra_map.count(bra_expr_2[j]))
                        bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                sr << "norm12 = np.zeros(norm.shape + (2, 2))" << endl;
                sr << "norm12[..., 0, 0] = norm" << endl << endl;
                sr << to_einsum_zeros(norm) << endl;
                sr << build_overlap(bra_expr, ket_expr_2, false).to_einsum(norm)
                   << endl;
                sr << "norm12[..., 0, 1] = norm" << endl << endl;
                sr << to_einsum_zeros(norm) << endl;
                sr << build_overlap(bra_expr_2, ket_expr, false).to_einsum(norm)
                   << endl;
                sr << "norm12[..., 1, 0] = norm" << endl << endl;
                sr << to_einsum_zeros(norm) << endl;
                sr << build_overlap(bra_expr_2, ket_expr_2, false)
                          .to_einsum(norm)
                   << endl;
                sr << "norm12[..., 1, 1] = norm" << endl << endl;

                sr << "dcas = ncas ** " << nact << endl;
                sr << "xn = norm12.reshape((-1, dcas, dcas, 2, 2))" << endl;
                sr << "xn = xn.transpose(0, 1, 3, 2, 4)" << endl;
                sr << "xn = xn.reshape((-1, dcas * 2, dcas * 2))" << endl;
            } else {
                if (nact == 2 && restrict_cas)
                    sr << "dcas = ncas * (ncas " << key.back() << " 1) // 2 "
                       << endl;
                else
                    sr << "dcas = ncas ** " << nact << endl;
                auto si =
                    to_einsum_sum_restriction(norm, string(1, key.back()));
                if (si.second) {
                    sr << si.first;
                    sr << "xn = norm[idx].reshape((-1, dcas, dcas))" << endl;
                } else
                    sr << "xn = norm.reshape((-1, dcas, dcas))" << endl << endl;
            }
            sr << "xnorms['" << mkey << "'] = xn" << endl;
            sr << "print(np.linalg.norm(xn))" << endl;
            sr << "assert np.linalg.norm(xn - xn.transpose(0, 2, 1)) < 1E-10"
               << endl;
            ss << WickExpr::to_einsum_add_indent(sr.str(), 0) << endl;
        }
        sk << "]" << endl << endl;
        for (int k = 0; k < (int)sub_spaces.size(); k++) {
            string xkkey = sub_spaces[k].first;
            string ket_expr = sub_spaces[k].second;
            string kmkey = "";
            if (xkkey.back() == '+')
                kmkey = "+";
            else if (xkkey.back() == '-')
                kmkey = "-";
            else if (xkkey.back() == '2')
                continue;
            string kkey = xkkey == "reference" ? xkkey : xkkey.substr(0, 4);
            string ikkey = xkkey == "reference" ? "" : kkey;
            bool kref = xkkey == "reference";
            bool krestrict_cas = xkkey.back() == '+' || xkkey.back() == '-';
            bool knon_ortho = xkkey.back() == '1' || xkkey.back() == '2';
            for (int b = 0; b < (int)sub_spaces.size(); b++) {
                string xbkey = sub_spaces[b].first;
                string bra_expr = sub_spaces[b].second;
                string bmkey = "";
                if (xbkey.back() == '+')
                    bmkey = "+";
                else if (xbkey.back() == '-')
                    bmkey = "-";
                else if (xbkey.back() == '2')
                    continue;
                string bkey = xbkey == "reference" ? xbkey : xbkey.substr(4, 4);
                string ibkey = xbkey == "reference" ? "" : bkey;
                string mkey = bkey + bmkey + " | H - E0 | " + kkey + kmkey;
                map<char, char> ket_bra_map;
                for (int j = 4; j < 8; j++)
                    ket_bra_map[xbkey[j - 4]] = xbkey[j];
                for (int j = 0; j < (int)bra_expr.length(); j++)
                    if (ket_bra_map.count(bra_expr[j]))
                        bra_expr[j] = ket_bra_map[bra_expr[j]];
                stringstream sr;
                cerr << mkey << endl;
                ss << "# compute : hmat = " << mkey << endl << endl;
                ss << "print('compute : hmat = < " << mkey << " >')" << endl
                   << endl;
                hmat = WickTensor::parse("hmat[" + ibkey + ikkey + "]", idx_map,
                                         perm_map);
                sr << to_einsum_zeros(hmat) << endl;
                bool bref = xbkey == "reference";
                bool brestrict_cas = xbkey.back() == '+' || xbkey.back() == '-';
                bool bnon_ortho = xbkey.back() == '1' || xbkey.back() == '2';
                sr << build_hamiltonian(bra_expr, ket_expr, false, b == k)
                          .to_einsum(hmat)
                   << endl;
                sr << "bdsub, bdcas = xnorms['" << norm_keys[b]
                   << "'].shape[:2]" << endl;
                sr << "kdsub, kdcas = xnorms['" << norm_keys[k]
                   << "'].shape[:2]" << endl;
                auto si = to_einsum_sum_restriction(
                    hmat, (bref ? "" : string(1, xbkey.back())) +
                              (kref ? "" : string(1, xkkey.back())));
                if (bnon_ortho && knon_ortho) {
                    assert(k == b);
                    string ket_expr_2 = sub_spaces[k + 1].second;
                    string bra_expr_2 = sub_spaces[b + 1].second;
                    for (int j = 0; j < (int)bra_expr_2.length(); j++)
                        if (ket_bra_map.count(bra_expr_2[j]))
                            bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                    sr << "hmat12 = np.zeros(hmat.shape + (2, 2))" << endl;
                    sr << "hmat12[..., 0, 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr, ket_expr_2, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 0, 1] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr_2, ket_expr, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1, 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr_2, ket_expr_2, false,
                                            b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1, 1] = hmat" << endl << endl;
                    sr << "hmat = hmat12.reshape((bdsub, bdcas // 2, kdsub, "
                          "kdcas // 2, 2, 2))"
                       << endl;
                    sr << "hmat = hmat.transpose((0, 1, 4, 2, 3, 5))" << endl;
                    sr << "xh = hmat.reshape((bdsub, bdcas, kdsub, kdcas))"
                       << endl;
                } else if (bnon_ortho) {
                    string bra_expr_2 = sub_spaces[b + 1].second;
                    for (int j = 0; j < (int)bra_expr_2.length(); j++)
                        if (ket_bra_map.count(bra_expr_2[j]))
                            bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                    sr << "hmat12 = np.zeros(hmat.shape + (2, ))" << endl;
                    sr << "hmat12[..., 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr_2, ket_expr, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1] = hmat" << endl << endl;
                    if (si.second) {
                        sr << si.first;
                        sr << "hmat = hmat12[idx].reshape((bdsub, bdcas // 2, "
                              "kdsub, kdcas, 2))"
                           << endl;
                    } else
                        sr << "hmat = hmat12.reshape((bdsub, bdcas // 2, "
                              "kdsub, kdcas, 2))"
                           << endl;
                    sr << "hmat = hmat.transpose((0, 1, 4, 2, 3))" << endl;
                    sr << "xh = hmat.reshape((bdsub, bdcas, kdsub, kdcas))"
                       << endl;
                } else if (knon_ortho) {
                    string ket_expr_2 = sub_spaces[k + 1].second;

                    sr << "hmat12 = np.zeros(hmat.shape + (2, ))" << endl;
                    sr << "hmat12[..., 0] = hmat" << endl << endl;
                    sr << to_einsum_zeros(hmat) << endl;
                    sr << build_hamiltonian(bra_expr, ket_expr_2, false, b == k)
                              .to_einsum(hmat)
                       << endl;
                    sr << "hmat12[..., 1] = hmat" << endl << endl;
                    if (si.second) {
                        sr << si.first;
                        sr << "xh = hmat12[idx].reshape((bdsub, bdcas, "
                              "kdsub, kdcas))"
                           << endl;
                    } else
                        sr << "xh = hmat12.reshape((bdsub, bdcas, "
                              "kdsub, kdcas))"
                           << endl;
                } else {
                    if (si.second) {
                        sr << si.first;
                        sr << "xh = hmat[idx].reshape((bdsub, bdcas, kdsub, "
                              "kdcas))"
                           << endl;
                    } else
                        sr << "xh = hmat.reshape((bdsub, bdcas, kdsub, kdcas))"
                           << endl;
                }
                sr << "xhmats[('" << norm_keys[b] << "', '" << norm_keys[k]
                   << "')] = xh" << endl;
                ss << WickExpr::to_einsum_add_indent(sr.str(), 0) << endl;
            }
        }
        return sk.str() + ss.str();
    }
};

class TestWickICMRCI : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickICMRCI, TestICMRCI) {

    WickICMRCI wic;
    cout << wic.to_einsum() << endl;

}
