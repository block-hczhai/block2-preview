
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestWickSCNEVPT2 : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickSCNEVPT2, TestSCNEVPT2) {

    WickSCNEVPT2 wsc;

    // cout << wsc.to_einsum() << endl;

    vector<WickExpr> axx_ref(26), axx_eq(26), xx_ref(19), xx_eq(19);

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

    xx_eq[11] = wsc.make_x11();
    xx_eq[12] = wsc.make_x12();
    xx_eq[13] = wsc.make_x13();
    xx_eq[14] = wsc.make_x14();
    xx_eq[15] = wsc.make_x15();
    xx_eq[16] = wsc.make_x16();
    xx_eq[17] = wsc.make_x17();
    xx_eq[18] = wsc.make_x18();

    for (auto &ix : vector<int>{11, 12, 13, 14, 15, 16, 17, 18}) {
        WickExpr diff = (xx_eq[ix] - xx_ref[ix]).simplify();
        //   cout << xx_ref[ix] << endl;
        //   cout << xx_eq[ix] << endl;
        cout << "DIFF Eq" << ix << " = " << diff << endl;
        EXPECT_TRUE(diff.terms.size() == 0);
    }

    return;

    //    axx_ref[7] = WickExpr::parse(R"TEX(
    //  - 1.0 SUM <d> hp[ad] E2[pq,db]
    //  - 1.0 SUM <d> hp[bd] E2[pq,ad]
    //  - 1.0 SUM <def> w[daef] E3[pqd,fbe]
    //  - 0.5 SUM <def> w[daef] delta[df] E2[pq,eb]
    //  - 0.5 SUM <def> w[daef] delta[bd] E2[pq,fe]
    //  - 1.0 SUM <def> w[dbef] E3[pqd,afe]
    //  - 0.5 SUM <def> w[dbef] delta[ad] E2[pq,ef]
    //  - 0.5 SUM <def> w[dbef] delta[df] E2[pq,ae]
    // )TEX",
    //                                  wsc.idx_map, wsc.perm_map)
    //                      .substitute(wsc.defs)
    //                      .expand()
    //                      .add_spin_free_trans_symm()
    //                      .simplify();

    axx_ref[7] = WickExpr::parse(R"TEX(
 - 1.0 SUM <rspqabd> gamma[rs] w[rsqp] w[rsba] hp[ad] E2[pq,db]
 - 1.0 SUM <rspqabd> gamma[rs] w[rsqp] w[rsba] hp[bd] E2[pq,ad]
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

    axx_ref[9] = WickExpr::parse(R"TEX(
 + 1.0 SUM <ijpqabc> gamma[ij] w[qpij] w[baij] hp[ca] E2T[pq,cb]
 + 1.0 SUM <ijpqabc> gamma[ij] w[qpij] w[baij] hp[cb] E2T[pq,ac]
 - 1.0 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] E3T[pqe,dbc]
 + 2.0 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] delta[ce] E2T[pq,db]
 - 0.5 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] delta[de] E2T[pq,cb]
 - 0.5 SUM <ijpqabcde> gamma[ij] w[qpij] w[baij] w[cdea] delta[be] E2T[pq,dc]
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

    axx_ref[12] = WickExpr::parse(R"TEX(
 + 1.0 SUM <c> hbar[ca] E1[q,p] E1[c,b]
 - 1.0 SUM <c> hbar[bc] E1[q,p] E1[a,c]
 + 0.5 SUM <cde> w[cdea] E1[q,p] E1[c,e] E1[d,b]
 + 0.5 SUM <cde> w[cdea] E1[q,p] E1[d,b] E1[c,e]
 - 0.5 SUM <cde> w[bced] E1[q,p] E1[a,e] E1[c,d]
 - 0.5 SUM <cde> w[bced] E1[q,p] E1[c,d] E1[a,e]
)TEX",
                                  wsc.idx_map, wsc.perm_map)
                      .substitute(wsc.defs)
                      .expand()
                      .add_spin_free_trans_symm()
                      .simplify();

    axx_ref[13] = WickExpr::parse(R"TEX(
 - 1.0 SUM <c> hbar[bc] E1T[p,a] E1[q,c]
 - 1.0 SUM <c> hbar[bc] delta[qp] E1[c,a]
 + 1.0 SUM <c> hbar[ca] E1T[p,c] E1[q,b]
 + 1.0 SUM <c> hbar[ca] delta[qp] E1[b,c]
 - 0.5 SUM <cde> w[cbed] E1[c,e] E1T[p,a] E1[q,d]
 - 0.5 SUM <cde> w[cbed] E1T[p,a] E1[q,d] E1[c,e]
 - 0.5 SUM <cde> w[cbed] delta[qp] E1[c,e] E1[a,d]
 - 0.5 SUM <cde> w[cbed] delta[qp] E1[a,d] E1[c,e]
 - 0.5 SUM <cde> w[cbed] delta[pc] E1T[e,a] E1[q,d]
 + 0.5 SUM <cde> w[cbed] delta[qe] E1T[p,a] E1[c,d]
 + 0.5 SUM <cde> w[cdea] E1[c,e] E1T[p,d] E1[q,b]
 + 0.5 SUM <cde> w[cdea] E1T[p,d] E1[q,b] E1[c,e]
 + 0.5 SUM <cde> w[cdea] delta[qp] E1[c,e] E1[d,b]
 + 0.5 SUM <cde> w[cdea] delta[qp] E1[d,b] E1[c,e]
 + 0.5 SUM <cde> w[cdea] delta[pc] E1T[e,d] E1[q,b]
 - 0.5 SUM <cde> w[cdea] delta[qe] E1T[p,d] E1[c,b]
)TEX",
                                  wsc.idx_map, wsc.perm_map)
                      .substitute(wsc.defs)
                      .expand()
                      .add_spin_free_trans_symm()
                      .simplify();

    axx_ref[16] = WickExpr::parse(R"TEX(
 + 1.0 SUM <d> hbar[da] E1[g,p] E1[q,b] E1[d,c]
 - 1.0 SUM <d> hbar[cd] E1[g,p] E1[q,b] E1[a,d]
 - 1.0 SUM <d> hbar[bd] E1[g,p] E1[q,d] E1[a,c]
 + 1.0 SUM <def> w[defa] E1[g,p] E1[q,b] E1[d,f] E1[e,c]
 - 1.0 SUM <def> w[dcfe] E1[g,p] E1[q,b] E1[d,f] E1[a,e]
 - 1.0 SUM <def> w[dbfe] E1[g,p] E1[q,e] E1[d,f] E1[a,c]
 + 1.0 SUM <de> w[cdea] E1[g,p] E1[q,b] E1[d,e]
 - 0.5 SUM <de> w[deea] E1[g,p] E1[q,b] E1[d,c]
 - 0.5 SUM <de> w[cdde] E1[g,p] E1[q,b] E1[a,e]
 + 0.5 SUM <de> w[bdde] E1[g,p] E1[q,e] E1[a,c]
)TEX",
                                  wsc.idx_map, wsc.perm_map)
                      .substitute(wsc.defs)
                      .expand()
                      .add_spin_free_trans_symm()
                      .simplify();

    axx_ref[17] = WickExpr::parse(R"TEX(
 - 1.0 SUM <c> hp[ac] E1[g,p] E1[q,c]
 - 1.0 SUM <cef> w[acef] E1[g,p] E1[q,e] E1[c,f]
)TEX",
                                  wsc.idx_map, wsc.perm_map)
                      .substitute(wsc.defs)
                      .expand()
                      .add_spin_free_trans_symm()
                      .simplify();

    axx_ref[18] = WickExpr::parse(R"TEX(
 + 1.0 SUM <d> hbar[da] E1[p,b] E1[d,c]
 - 1.0 SUM <d> hbar[cd] E1[p,b] E1[a,d]
 - 1.0 SUM <d> hbar[bd] E1[p,d] E1[a,c]
 + 1.0 SUM <def> w[defa] E1[p,b] E1[d,f] E1[e,c]
 - 1.0 SUM <def> w[dcfe] E1[p,b] E1[d,f] E1[a,e]
 - 1.0 SUM <def> w[dbfe] E1[p,e] E1[d,f] E1[a,c]
 + 1.0 SUM <de> w[cdea] E1[p,b] E1[d,e]
 - 0.5 SUM <de> w[deea] E1[p,b] E1[d,c]
 - 0.5 SUM <de> w[cdde] E1[p,b] E1[a,e]
 + 0.5 SUM <de> w[bdde] E1[p,e] E1[a,c]
)TEX",
                                  wsc.idx_map, wsc.perm_map)
                      .substitute(wsc.defs)
                      .expand()
                      .add_spin_free_trans_symm()
                      .simplify();

    axx_ref[19] = WickExpr::parse(R"TEX(
 - 1.0 SUM <c> hp[ac] E1[p,c]
 - 1.0 SUM <cef> w[acef] E1[p,e] E1[c,f]
)TEX",
                                  wsc.idx_map, wsc.perm_map)
                      .substitute(wsc.defs)
                      .expand()
                      .add_spin_free_trans_symm()
                      .simplify();

    axx_eq[7] = wsc.make_a7();
    //  axx_eq[9] = wsc.make_a9();
    // axx_eq[12] = wsc.make_a12();
    // axx_eq[13] = wsc.make_a13();
    // axx_eq[16] = wsc.make_a16();
    //  axx_eq[17] = wsc.make_a17();
    //  axx_eq[18] = wsc.make_a18();
    //  axx_eq[19] = wsc.make_a19();

    for (auto &ix : vector<int>{7}) {
        WickExpr diff = (axx_eq[ix] - axx_ref[ix]).simplify();
        cout << axx_ref[ix] << endl;
        cout << axx_eq[ix] << endl;
        cout << "DIFF A" << ix << " = " << diff << endl;
        EXPECT_TRUE(diff.terms.size() == 0);
    }
}
