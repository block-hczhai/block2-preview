
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

struct WickICNEVPT2 {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    map<string, pair<WickTensor, WickExpr>> defs;
    vector<pair<string, string>> sub_spaces;
    WickExpr heff, hw, hd;
    WickICNEVPT2() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("mnxyijkl");
        idx_map[WickIndexTypes::Active] =
            WickIndex::parse_set("mnxyabcdefghpq");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("mnxyrstu");
        perm_map[make_pair("w", 4)] = WickPermutation::qc_phys();
        WickExpr hi = WickExpr::parse("SUM <i> orbe[i] E1[i,i]\n"
                                      "SUM <r> orbe[r] E1[r,r]",
                                      idx_map, perm_map);
        heff = WickExpr::parse("SUM <ab> h[ab] E1[a,b]", idx_map, perm_map);
        hw = WickExpr::parse("0.5 SUM <abcd> w[abcd] E2[ab,cd]", idx_map,
                             perm_map);
        hd = hi + heff + hw;
        sub_spaces = {{"ijrs+", "E1[r,i] E1[s,j] \n + E1[s,i] E1[r,j]"},
                      {"ijrs-", "E1[r,i] E1[s,j] \n - E1[s,i] E1[r,j]"},
                      {"rsiap+", "E1[r,i] E1[s,a] \n + E1[s,i] E1[r,a]"},
                      {"rsiap-", "E1[r,i] E1[s,a] \n - E1[s,i] E1[r,a]"},
                      {"ijrap+", "E1[r,j] E1[a,i] \n + E1[r,i] E1[a,j]"},
                      {"ijrap-", "E1[r,j] E1[a,i] \n - E1[r,i] E1[a,j]"},
                      {"rsabpq+", "E1[r,b] E1[s,a] \n + E1[s,b] E1[r,a]"},
                      {"rsabpq-", "E1[r,b] E1[s,a] \n - E1[s,b] E1[r,a]"},
                      {"ijabpq+", "E1[b,i] E1[a,j] \n + E1[b,j] E1[a,i]"},
                      {"ijabpq-", "E1[b,i] E1[a,j] \n - E1[b,j] E1[a,i]"},
                      {"irabpq1", "E1[r,i] E1[a,b]"},
                      {"irabpq2", "E1[a,i] E1[r,b]"},
                      {"rabcpqg", "E1[r,b] E1[a,c]"},
                      {"iabcpqg", "E1[b,i] E1[a,c]"}};
    }
    WickExpr build_communicator(const string &bra, const string &ket,
                                bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xbra.conjugate() & (hd ^ xket).expand().simplify())
                   : (xbra.conjugate() * (hd ^ xket).expand().simplify());
        return expr.expand()
            .remove_external()
            .add_spin_free_trans_symm()
            .simplify();
    }
    WickExpr build_norm(const string &bra, const string &ket,
                        bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xbra.conjugate() & xket) : (xbra.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .simplify();
    }
    WickExpr build_rhs(const string &bra, const string &ket,
                       bool do_sum = true) const {
        WickExpr xbra = WickExpr::parse(bra, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xbra.conjugate() & xket) : (xbra.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .remove_inactive()
            .simplify();
    }
    string to_einsum_orb_energies(const WickTensor &tensor) const {
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
    string to_einsum_sum_restriction(const WickTensor &tensor,
                                     bool restrict_cas = true,
                                     bool no_eq = false) const {
        stringstream ss, sr;
        ss << "grid = np.indices((";
        bool first_and = false;
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (!restrict_cas && wi.types == WickIndexTypes::Active)
                continue;
            ss << (wi.types == WickIndexTypes::Inactive
                       ? "ncore"
                       : (wi.types == WickIndexTypes::External ? "nvirt"
                                                               : "ncas"));
            if (i != (int)tensor.indices.size() - 1 || i == 0)
                ss << ", ";
            if (i != 0 &&
                tensor.indices[i].types == tensor.indices[i - 1].types) {
                if (wi.types == WickIndexTypes::Active) {
                    if (tensor.indices[i].name[0] !=
                        tensor.indices[i - 1].name[0] + 1)
                        continue;
                    if (i + 1 < (int)tensor.indices.size() &&
                        tensor.indices[i + 1].name[0] ==
                            tensor.indices[i].name[0] + 1)
                        continue;
                    if (i - 2 >= 0 && tensor.indices[i - 1].name[0] ==
                                          tensor.indices[i - 2].name[0] + 1)
                        continue;
                }
                sr << "idx " << (first_and ? "&" : "") << "= grid[" << i - 1
                   << "] <" << (no_eq ? "" : "=") << " grid[" << i << "]"
                   << endl;
                first_and = true;
            }
        }
        ss << "))" << endl;
        return ss.str() + sr.str();
    }
    string to_einsum() const {
        stringstream ss;
        WickTensor hexp, deno, rheq;
        for (int i = 0; i < (int)sub_spaces.size(); i++) {
            string key = sub_spaces[i].first, ket_expr = sub_spaces[i].second;
            string mkey = key, skey = key;
            // this h is actually heff
            string hfull = "SUM <mn> h[mn] E1[m,n] \n"
                           "-2.0 SUM <mnj> w[mjnj] E1[m,n]\n"
                           "+1.0 SUM <mnj> w[mjjn] E1[m,n]\n"
                           "0.5 SUM <mnxy> w[mnxy] E2[mn,xy] \n";
            if (key.back() == '+')
                skey = key.substr(0, key.length() - 1), mkey = skey + "_plus";
            else if (key.back() == '-')
                skey = key.substr(0, key.length() - 1), mkey = skey + "_minus";
            else if (key.back() == '1')
                skey = key.substr(0, key.length() - 1), mkey = skey;
            else if (key.back() == '2')
                continue;
            string rkey = skey;
            map<char, char> ket_bra_map;
            if (skey.length() > 4) {
                for (int j = 4; j < (int)skey.length(); j++)
                    ket_bra_map[skey[j + 4 - skey.length()]] = skey[j];
                rkey = skey.substr(0, skey.length() - ket_bra_map.size() * 2) +
                       skey.substr(skey.length() - ket_bra_map.size());
            }
            string bra_expr = ket_expr;
            for (int j = 0; j < (int)bra_expr.length(); j++)
                if (ket_bra_map.count(bra_expr[j]))
                    bra_expr[j] = ket_bra_map[bra_expr[j]];
            stringstream sr;
            ss << "def compute_" << mkey << "():" << endl;
            hexp = WickTensor::parse("hexp[" + skey + "]", idx_map, perm_map);
            deno = WickTensor::parse("deno[" + skey + "]", idx_map, perm_map);
            rheq = WickTensor::parse("rheq[" + rkey + "]", idx_map, perm_map);
            sr << to_einsum_orb_energies(rheq) << endl;
            sr << build_rhs(bra_expr, hfull, false).to_einsum(rheq) << endl;
            sr << to_einsum_orb_energies(hexp) << endl;
            sr << build_communicator(bra_expr, ket_expr, false).to_einsum(hexp)
               << endl;
            bool restrict_cas = key.back() == '+' || key.back() == '-';
            bool non_ortho = key.back() == '1' || key.back() == '2';
            if (non_ortho) {
                string ket_expr_2 = sub_spaces[i + 1].second;
                string bra_expr_2 = ket_expr_2;
                for (int j = 0; j < (int)bra_expr_2.length(); j++)
                    if (ket_bra_map.count(bra_expr_2[j]))
                        bra_expr_2[j] = ket_bra_map[bra_expr_2[j]];

                sr << "rheq12 = np.zeros(rheq.shape + (2, ))" << endl;
                sr << "rheq12[..., 0] = rheq" << endl << endl;
                sr << to_einsum_orb_energies(rheq) << endl;
                sr << build_rhs(bra_expr_2, hfull, false).to_einsum(rheq)
                   << endl;
                sr << "rheq12[..., 1] = rheq" << endl << endl;

                sr << "hexp12 = np.zeros(hexp.shape + (2, 2, ))" << endl;
                sr << "hexp12[..., 0, 0] = hexp" << endl << endl;
                sr << to_einsum_orb_energies(hexp) << endl;
                sr << build_communicator(bra_expr, ket_expr_2, false)
                          .to_einsum(hexp)
                   << endl;
                sr << "hexp12[..., 0, 1] = hexp" << endl << endl;
                sr << to_einsum_orb_energies(hexp) << endl;
                sr << build_communicator(bra_expr_2, ket_expr, false)
                          .to_einsum(hexp)
                   << endl;
                sr << "hexp12[..., 1, 0] = hexp" << endl << endl;
                sr << to_einsum_orb_energies(hexp) << endl;
                sr << build_communicator(bra_expr_2, ket_expr_2, false)
                          .to_einsum(hexp)
                   << endl;
                sr << "hexp12[..., 1, 1] = hexp" << endl << endl;
                sr << "dcas = ncas ** " << ket_bra_map.size() << endl;
                sr << "xr = rheq12.reshape((-1, dcas * 2))" << endl;
                sr << "xh = hexp12.reshape((-1, dcas, dcas, 2, 2))" << endl;
                sr << "xh = xh.transpose(0, 1, 3, 2, 4)" << endl;
                sr << "xh = xh.reshape((-1, dcas * 2, dcas * 2))" << endl;
            } else {
                if (ket_bra_map.size() == 2 && restrict_cas)
                    sr << "dcas = ncas * (ncas " << key.back() << " 1) // 2 "
                       << endl;
                else
                    sr << "dcas = ncas ** " << ket_bra_map.size() << endl;
                if (skey.length() - ket_bra_map.size() * 2 >= 2) {
                    sr << to_einsum_sum_restriction(rheq, restrict_cas,
                                                    key.back() == '-');
                    sr << "xr = rheq[idx].reshape((-1, dcas))" << endl;
                    sr << to_einsum_sum_restriction(hexp, restrict_cas,
                                                    key.back() == '-');
                    sr << "xh = hexp[idx].reshape((-1, dcas, dcas))" << endl
                       << endl;
                } else {
                    sr << "xr = rheq.reshape((-1, dcas))" << endl << endl;
                    sr << "xh = hexp.reshape((-1, dcas, dcas))" << endl << endl;
                }
            }
            sr << "return -(np.linalg.solve(xh, xr) * xr).sum()" << endl;
            ss << WickExpr::to_einsum_add_indent(sr.str()) << endl;
        }
        return ss.str();
    }
};

class TestWickICNEVPT2 : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickICNEVPT2, TestICNEVPT2) {

    WickICNEVPT2 wic;
    cout << wic.to_einsum() << endl;

}
