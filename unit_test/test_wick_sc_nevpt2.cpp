
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

struct WickSCNEVPT2 {
    map<WickIndexTypes, set<WickIndex>> idx_map;
    map<pair<string, int>, vector<WickPermutation>> perm_map;
    map<string, pair<WickTensor, WickExpr>> defs;
    vector<pair<string, string>> sub_spaces;
    WickExpr heff, hw, hd;
    WickSCNEVPT2() {
        idx_map[WickIndexTypes::Inactive] = WickIndex::parse_set("mnxyijkl");
        idx_map[WickIndexTypes::Active] =
            WickIndex::parse_set("mnxyabcdefghpq");
        idx_map[WickIndexTypes::External] = WickIndex::parse_set("mnxyrstu");
        perm_map[make_pair("w", 4)] = WickPermutation::qc_phys();
        heff = WickExpr::parse("SUM <ab> h[ab] E1[a,b]", idx_map, perm_map);
        hw = WickExpr::parse("0.5 SUM <abcd> w[abcd] E2[ab,cd]", idx_map,
                             perm_map);
        hd = heff + hw;
        sub_spaces = {{"ijrs", "gamma[ij] gamma[rs] w[rsij] E1[r,i] E1[s,j] \n"
                               "gamma[ij] gamma[rs] w[rsji] E1[s,i] E1[r,j]"},
                      {"rsi", "SUM <a> gamma[rs] w[rsia] E1[r,i] E1[s,a] \n"
                              "SUM <a> gamma[rs] w[sria] E1[s,i] E1[r,a]"},
                      {"ijr", "SUM <a> gamma[ij] w[raji] E1[r,j] E1[a,i] \n"
                              "SUM <a> gamma[ij] w[raij] E1[r,i] E1[a,j]"},
                      {"rs", "SUM <ab> gamma[rs] w[rsba] E1[r,b] E1[s,a]"},
                      {"ij", "SUM <ab> gamma[ij] w[baij] E1[b,i] E1[a,j]"},
                      {"ir", "SUM <ab> w[raib] E1[r,i] E1[a,b] \n"
                             "SUM <ab> w[rabi] E1[a,i] E1[r,b] \n"
                             "h[ri] E1[r,i]"},
                      {"r", "SUM <abc> w[rabc] E1[r,b] E1[a,c] \n"
                            "SUM <a> h[ra] E1[r,a] \n"
                            "- SUM <ab> w[rbba] E1[r,a]"},
                      {"i", "SUM <abc> w[baic] E1[b,i] E1[a,c] \n"
                            "SUM <a> h[ai] E1[a,i]"}};
        defs["gamma"] = WickExpr::parse_def(
            "gamma[mn] = 1.0 \n - 0.5 delta[mn]", idx_map, perm_map);
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
    WickExpr build_communicator(const string &ket, bool do_sum = true) const {
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xket.conjugate() & (hd ^ xket).expand().simplify())
                   : (xket.conjugate() * (hd ^ xket).expand().simplify());
        return expr.expand()
            .remove_external()
            .add_spin_free_trans_symm()
            .simplify();
    }
    WickExpr build_norm(const string &ket, bool do_sum = true) const {
        WickExpr xket = WickExpr::parse(ket, idx_map, perm_map)
                            .substitute(defs)
                            .expand()
                            .simplify();
        WickExpr expr =
            do_sum ? (xket.conjugate() & xket) : (xket.conjugate() * xket);
        return expr.expand()
            .add_spin_free_trans_symm()
            .remove_external()
            .simplify();
    }
    string to_einsum_orb_energies(const WickTensor &tensor) const {
        stringstream ss;
        ss << tensor.name << " = ";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            if (wi.types == WickIndexTypes::Inactive)
                ss << "(-1) * ";
            ss << "orbe" << to_str(wi.types);
            ss << "[";
            for (int j = 0; j < (int)tensor.indices.size(); j++) {
                ss << (i == j ? ":" : "None");
                if (j != (int)tensor.indices.size() - 1)
                    ss << ", ";
            }
            ss << "]";
            if (i != (int)tensor.indices.size() - 1)
                ss << " + ";
        }
        return ss.str();
    }
    string to_einsum_sum_restriction(const WickTensor &tensor) const {
        stringstream ss, sr;
        ss << "grid = np.indices((";
        for (int i = 0; i < (int)tensor.indices.size(); i++) {
            auto &wi = tensor.indices[i];
            ss << (wi.types == WickIndexTypes::Inactive ? "ncore" : "nvirt");
            if (i != (int)tensor.indices.size() - 1 || i == 0)
                ss << ", ";
            if (i != 0 &&
                tensor.indices[i].types == tensor.indices[i - 1].types)
                sr << "idx &= grid[" << i - 1 << "] <= grid[" << i << "]"
                   << endl;
        }
        ss << "))" << endl;
        return ss.str() + sr.str();
    }
    string to_einsum() const {
        stringstream ss;
        WickTensor norm, ener, deno;
        for (int i = 0; i < (int)sub_spaces.size(); i++) {
            string key = sub_spaces[i].first, expr = sub_spaces[i].second;
            stringstream sr;
            ss << "def compute_" << key << "():" << endl;
            norm = WickTensor::parse("norm[" + key + "]", idx_map, perm_map);
            ener = WickTensor::parse("hexp[" + key + "]", idx_map, perm_map);
            deno = WickTensor::parse("deno[" + key + "]", idx_map, perm_map);
            sr << to_einsum_orb_energies(deno) << endl;
            sr << "norm = np.zeros_like(deno)" << endl;
            sr << build_norm(expr, false).to_einsum(norm) << endl;
            sr << "hexp = np.zeros_like(deno)" << endl;
            sr << build_communicator(expr, false).to_einsum(ener) << endl;
            sr << "idx = abs(norm) > 1E-14" << endl;
            if (key.length() >= 2)
                sr << to_einsum_sum_restriction(deno) << endl;
            sr << "hexp[idx] = deno[idx] + hexp[idx] / norm[idx]" << endl;
            sr << "xener = -(norm[idx] / hexp[idx]).sum()" << endl;
            sr << "xnorm = norm[idx].sum()" << endl;
            sr << "return xnorm, xener" << endl;
            ss << WickExpr::to_einsum_add_indent(sr.str()) << endl;
        }
        return ss.str();
    }
};

class TestWickSCNEVPT2 : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickSCNEVPT2, TestSCNEVPT2) {

    WickSCNEVPT2 wsc;

    //  cout << wsc.to_einsum() << endl;

    wsc.defs["hbar"] = WickExpr::parse_def(
        "hbar[ab] = h[ab] \n - 0.5 SUM <c> w[accb]", wsc.idx_map, wsc.perm_map);
    wsc.defs["hp"] = WickExpr::parse_def(
        "hp[mn] = h[mn] \n - 1.0 SUM <b> w[mbbn]", wsc.idx_map, wsc.perm_map);
    wsc.defs["E1T"] = WickExpr::parse_def(
        "E1T[a,b] = 2.0 delta[ab] \n - E1[b,a]", wsc.idx_map, wsc.perm_map);
    wsc.defs["E2TX"] = WickExpr::parse_def(
        "E2TX[pq,ab] = E2[ab,pq] \n + delta[pb] E1[a,q] \n"
        "delta[qa] E1[b,p] \n - 2.0 delta[pa] E1[b,q] \n"
        "- 2.0 delta[qb] E1[a,p] \n - 2.0 delta[pb] delta[qa] \n"
        "+ 4.0 delta[ap] delta[bq]",
        wsc.idx_map, wsc.perm_map);
    wsc.defs["E2T"] = WickExpr::parse_def(
        "E2T[pq,ab] = E1T[p,a] E1T[q,b] \n - delta[qa] E1T[p,b]", wsc.idx_map,
        wsc.perm_map);
    wsc.defs["E2T"].second = wsc.defs["E2T"].second.substitute(wsc.defs);
    assert((wsc.defs["E2T"].second - wsc.defs["E2TX"].second)
               .expand()
               .simplify()
               .terms.size() == 0);
    wsc.defs["E3T"] = WickExpr::parse_def(
        "E3T[pqg,abc] = E1T[p,a] E1T[q,b] E1T[g,c] \n"
        " - delta[ag] E2T[pq,cb] \n - delta[aq] E2T[pg,bc] \n"
        " - delta[bg] E2T[pq,ac] \n - delta[aq] delta[bg] E1T[p,c]",
        wsc.idx_map, wsc.perm_map);
    wsc.defs["E3T"].second = wsc.defs["E3T"].second.substitute(wsc.defs);

    vector<WickExpr> xx_ref(19), xx_eq(19);

    xx_ref[11] = WickExpr::parse(R"TEX(
 + 4.0 SUM <rsij> gamma[ij] gamma[rs] w[rsij] w[rsij]
 + 4.0 SUM <rsij> gamma[ij] gamma[rs] w[rsji] w[rsji]
 - 4.0 SUM <rsij> gamma[ij] gamma[rs] w[rsij] w[rsji]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    xx_ref[12] = WickExpr::parse(R"TEX(
 + 2.0 SUM <rsiap> gamma[rs] w[rsip] w[rsia] E1[p,a]
 + 2.0 SUM <rsiap> gamma[rs] w[srip] w[sria] E1[p,a]
 - 1.0 SUM <rsiap> gamma[rs] w[rsip] w[sria] E1[p,a]
 - 1.0 SUM <rsiap> gamma[rs] w[srip] w[rsia] E1[p,a]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    xx_ref[13] = WickExpr::parse(R"TEX(
 + 2.0 SUM <rijap> gamma[ij] w[rpji] w[raji] E1T[p,a]
 + 2.0 SUM <rijap> gamma[ij] w[rpij] w[raij] E1T[p,a]
 - 1.0 SUM <rijap> gamma[ij] w[rpji] w[raij] E1T[p,a]
 - 1.0 SUM <rijap> gamma[ij] w[rpij] w[raji] E1T[p,a]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    xx_ref[14] = WickExpr::parse(R"TEX(
 SUM <rspqab> gamma[rs] w[rsqp] w[rsba] E2[pq,ab]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    xx_ref[15] = WickExpr::parse(R"TEX(
 SUM <ijpqab> gamma[ij] w[qpij] w[baij] E2T[pq,ab]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    xx_ref[16] = WickExpr::parse(R"TEX(
 + 2.0 SUM <irpqab> w[rpiq] w[raib] E1[q,p] E1[a,b]
 - 1.0 SUM <irpqab> w[rpiq] w[rabi] E1[q,p] E1[a,b]
 - 1.0 SUM <irpqab> w[rpqi] w[raib] E1[q,p] E1[a,b]
 + 1.0 SUM <irpqab> w[rpqi] w[rabi] E1[q,b] E1T[p,a]
 + 1.0 SUM <irpqab> w[rpqi] w[rabi] delta[ab] E1[q,p]
 + 4.0 SUM <irpq> w[rpiq] h[ri] E1[q,p]
 - 2.0 SUM <irpq> w[rpqi] h[ri] E1[q,p]
 + 2.0 SUM <ir> h[ri] h[ri]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    // hp substitution has some problems
    xx_ref[17] = WickExpr::parse(R"TEX(
 + 1.0 SUM <rpqgabc> w[rpqg] w[rabc] E1[g,p] E1[q,b] E1[a,c]
 + 2.0 SUM <rpqga> w[rpqg] h[ra] E1[g,p] E1[q,a]
 - 2.0 SUM <rpqgab> w[rpqg] w[rbba] E1[g,p] E1[q,a]
 + 1.0 SUM <rpa> h[rp] h[ra] E1[p,a]
 - 1.0 SUM <rpab> w[rbbp] h[ra] E1[p,a]
 - 1.0 SUM <rpac> h[rp] w[rcca] E1[p,a]
 + 1.0 SUM <rpabc> w[rbbp] w[rcca] E1[p,a]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    xx_ref[18] = WickExpr::parse(R"TEX(
 + 1.0 SUM <ipqgabc> w[qpig] w[baic] E1[g,p] E1T[q,b] E1[a,c]
 + 2.0 SUM <ipqga> w[qpig] h[ai] E1[g,p] E1T[q,a]
 + 1.0 SUM <ipa> h[pi] h[ai] E1T[p,a]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    // Eq (3) ijrs
    xx_eq[11] = wsc.build_norm("gamma[ij] gamma[rs] w[rsij] E1[r,i] E1[s,j] \n"
                               "gamma[ij] gamma[rs] w[rsji] E1[s,i] E1[r,j]",
                               true);
    // Eq (4) rsi
    xx_eq[12] = wsc.build_norm("SUM <a> gamma[rs] w[rsia] E1[r,i] E1[s,a] \n"
                               "SUM <a> gamma[rs] w[sria] E1[s,i] E1[r,a]",
                               true);
    // Eq (5) ijr
    xx_eq[13] = wsc.build_norm("SUM <a> gamma[ij] w[raji] E1[r,j] E1[a,i] \n"
                               "SUM <a> gamma[ij] w[raij] E1[r,i] E1[a,j]",
                               true);
    // Eq (6) rs
    xx_eq[14] =
        wsc.build_norm("SUM <ab> gamma[rs] w[rsba] E1[r,b] E1[s,a]", true);
    // Eq (7) ij
    xx_eq[15] =
        wsc.build_norm("SUM <ab> gamma[ij] w[baij] E1[b,i] E1[a,j]", true);
    // Eq (8) ir
    xx_eq[16] = wsc.build_norm("SUM <ab> w[raib] E1[r,i] E1[a,b] \n"
                               "SUM <ab> w[rabi] E1[a,i] E1[r,b] \n"
                               "h[ri] E1[r,i]",
                               true);
    // Eq (9) r
    xx_eq[17] = wsc.build_norm("SUM <abc> w[rabc] E1[r,b] E1[a,c] \n"
                               "SUM <a> h[ra] E1[r,a] \n"
                               "- SUM <ab> w[rbba] E1[r,a]",
                               true);
    // Eq (10) i
    xx_eq[18] = wsc.build_norm("SUM <abc> w[baic] E1[b,i] E1[a,c] \n"
                               "SUM <a> h[ai] E1[a,i]",
                               true);

    for (auto &ix : vector<int>{11, 12, 13, 14, 15, 16, 17, 18}) {
        WickExpr diff = (xx_eq[ix] - xx_ref[ix]).simplify();
        //   cout << xx_ref[ix] << endl;
        //   cout << xx_eq[ix] << endl;
        cout << "DIFF Eq" << ix << " = " << diff << endl;
        EXPECT_TRUE(diff.terms.size() == 0);
    }

    vector<WickExpr> axx_ref(30), axx_eq(30);

    // revised heff-prime -> hbar
    axx_ref[7] = WickExpr::parse(R"TEX(
 - 1.0 SUM <rspqabd> gamma[rs] w[rsqp] w[rsba] hbar[ad] E2[pq,db]
 - 1.0 SUM <rspqabd> gamma[rs] w[rsqp] w[rsba] hbar[bd] E2[pq,ad]
 - 1.0 SUM <rspqabdef> gamma[rs] w[rsqp] w[rsba] w[daef] E3[pqd,fbe]
 - 0.5 SUM <rspqabdef> gamma[rs] w[rsqp] w[rsba] w[daef] delta[df] E2[pq,eb]
 - 0.5 SUM <rspqabdef> gamma[rs] w[rsqp] w[rsba] w[daef] delta[bd] E2[pq,fe]
 - 1.0 SUM <rspqabdef> gamma[rs] w[rsqp] w[rsba] w[dbef] E3[pqd,afe]
 - 0.5 SUM <rspqabdef> gamma[rs] w[rsqp] w[rsba] w[dbef] delta[ad] E2[pq,ef]
 - 0.5 SUM <rspqabdef> gamma[rs] w[rsqp] w[rsba] w[dbef] delta[df] E2[pq,ae]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    // revised heff-prime -> hbar
    axx_ref[9] = WickExpr::parse(R"TEX(
 + 1.0 SUM <ijpqabc> gamma[ij] w[qpij] w[baij] hbar[ca] E2T[pq,cb]
 + 1.0 SUM <ijpqabc> gamma[ij] w[qpij] w[baij] hbar[cb] E2T[pq,ac]
 - 1.0 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] E3T[pqe,dbc]
 + 2.0 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] delta[ce] E2T[pq,db]
 - 0.5 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] delta[be] E2T[pq,dc]
 - 0.5 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] delta[de] E2T[pq,cb]
 - 1.0 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdeb] E3T[pqe,adc]
 + 2.0 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdeb] delta[ce] E2T[pq,ad]
 - 0.5 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdeb] delta[ae] E2T[pq,cd]
 - 0.5 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdeb] delta[de] E2T[pq,ac]
)TEX",
                                 wsc.idx_map, wsc.perm_map)
                     .substitute(wsc.defs)
                     .expand()
                     .add_spin_free_trans_symm()
                     .simplify();

    // rs
    axx_eq[7] = wsc.build_communicator(
        "SUM <ab> gamma[rs] w[rsba] E1[r,b] E1[s,a]", true);
    // ij
    axx_eq[9] = wsc.build_communicator(
        "SUM <ab> gamma[ij] w[baij] E1[b,i] E1[a,j]", true);

    for (auto &ix : vector<int>{7, 9}) {
        WickExpr diff = (axx_eq[ix] - axx_ref[ix]).simplify();
        //   cout << axx_ref[ix] << endl;
        //   cout << axx_eq[ix] << endl;
        cout << "DIFF A" << ix << " = " << diff << endl;
        EXPECT_TRUE(diff.terms.size() == 0);
    }
}
