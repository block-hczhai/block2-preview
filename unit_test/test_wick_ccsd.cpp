
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestWickCCSD : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickCCSD, TestCCSD) {
    WickCCSD wccsd;
    WickExpr t1_ref = WickExpr::parse(R"TEX(
 + 1.0h_{ai}
 - 1.0\sum_{j}h_{ji}t_{aj}
 + 1.0\sum_{b}h_{ab}t_{bi}
 - 1.0\sum_{jb}h_{jb}t_{abji}
 - 1.0\sum_{jb}v_{jaib}t_{bj}
 + 0.5\sum_{jkb}v_{jkib}t_{abkj}
 - 0.5\sum_{jbc}v_{jabc}t_{cbji}
 - 1.0\sum_{jb}h_{jb}t_{bi}t_{aj}
 - 1.0\sum_{jkb}v_{jkib}t_{aj}t_{bk}
 + 1.0\sum_{jbc}v_{jabc}t_{ci}t_{bj}
 - 0.5\sum_{jkbc}v_{jkbc}t_{aj}t_{cbki}
 - 0.5\sum_{jkbc}v_{jkbc}t_{ci}t_{abkj}
 + 1.0\sum_{jkbc}v_{jkbc}t_{cj}t_{abki}
 + 1.0\sum_{jkbc}v_{jkbc}t_{ci}t_{aj}t_{bk}
)TEX",
                                      wccsd.idx_map, wccsd.perm_map);
    vector<WickExpr> t2_ref(5);
    t2_ref[0] = WickExpr::parse(R"TEX(
 + 1.0v_{baji}
)TEX",
                                wccsd.idx_map, wccsd.perm_map);
    t2_ref[1] = WickExpr::parse(R"TEX(
 + 1.0\sum_{k}h_{ki}t_{bakj}
 - 1.0\sum_{k}h_{kj}t_{baki}
 + 1.0\sum_{c}h_{ac}t_{bcji}
 - 1.0\sum_{c}h_{bc}t_{acji}
 - 1.0\sum_{k}v_{kaji}t_{bk}
 + 1.0\sum_{k}v_{kbji}t_{ak}
 - 1.0\sum_{c}v_{baic}t_{cj}
 + 1.0\sum_{c}v_{bajc}t_{ci}
 - 0.5\sum_{kl}v_{klji}t_{balk}
 + 1.0\sum_{kc}v_{kaic}t_{bckj}
 - 1.0\sum_{kc}v_{kajc}t_{bcki}
 - 1.0\sum_{kc}v_{kbic}t_{ackj}
 + 1.0\sum_{kc}v_{kbjc}t_{acki}
 - 0.5\sum_{cd}v_{bacd}t_{dcji}
)TEX",
                                wccsd.idx_map, wccsd.perm_map);
    t2_ref[2] = WickExpr::parse(R"TEX(
 - 1.0\sum_{kc}h_{kc}t_{ak}t_{bcji}
 + 1.0\sum_{kc}h_{kc}t_{bk}t_{acji}
 + 1.0\sum_{kc}h_{kc}t_{ci}t_{bakj}
 - 1.0\sum_{kc}h_{kc}t_{cj}t_{baki}
 + 1.0\sum_{kl}v_{klji}t_{bk}t_{al}
 + 1.0\sum_{kc}v_{kaic}t_{cj}t_{bk}
 - 1.0\sum_{kc}v_{kajc}t_{ci}t_{bk}
 - 1.0\sum_{kc}v_{kbic}t_{cj}t_{ak}
 + 1.0\sum_{kc}v_{kbjc}t_{ci}t_{ak}
 + 1.0\sum_{cd}v_{bacd}t_{di}t_{cj}
 + 1.0\sum_{klc}v_{klic}t_{ak}t_{bclj}
 - 1.0\sum_{klc}v_{klic}t_{bk}t_{aclj}
 + 0.5\sum_{klc}v_{klic}t_{cj}t_{balk}
 - 1.0\sum_{klc}v_{klic}t_{ck}t_{balj}
 - 1.0\sum_{klc}v_{kljc}t_{ak}t_{bcli}
 + 1.0\sum_{klc}v_{kljc}t_{bk}t_{acli}
 - 0.5\sum_{klc}v_{kljc}t_{ci}t_{balk}
 + 1.0\sum_{klc}v_{kljc}t_{ck}t_{bali}
 + 0.5\sum_{kcd}v_{kacd}t_{bk}t_{dcji}
 - 1.0\sum_{kcd}v_{kacd}t_{di}t_{bckj}
 + 1.0\sum_{kcd}v_{kacd}t_{dj}t_{bcki}
 - 1.0\sum_{kcd}v_{kacd}t_{dk}t_{bcji}
 - 0.5\sum_{kcd}v_{kbcd}t_{ak}t_{dcji}
 + 1.0\sum_{kcd}v_{kbcd}t_{di}t_{ackj}
 - 1.0\sum_{kcd}v_{kbcd}t_{dj}t_{acki}
 + 1.0\sum_{kcd}v_{kbcd}t_{dk}t_{acji}
 + 0.5\sum_{klcd}v_{klcd}t_{adji}t_{bclk}
 - 1.0\sum_{klcd}v_{klcd}t_{adki}t_{bclj}
 - 0.5\sum_{klcd}v_{klcd}t_{baki}t_{dclj}
 - 0.5\sum_{klcd}v_{klcd}t_{bdji}t_{aclk}
 + 1.0\sum_{klcd}v_{klcd}t_{bdki}t_{aclj}
 + 0.25\sum_{klcd}v_{klcd}t_{dcji}t_{balk}
 - 0.5\sum_{klcd}v_{klcd}t_{dcki}t_{balj}
)TEX",
                                wccsd.idx_map, wccsd.perm_map);
    t2_ref[3] = WickExpr::parse(R"TEX(
 - 1.0\sum_{klc}v_{klic}t_{cj}t_{bk}t_{al}
 + 1.0\sum_{klc}v_{kljc}t_{ci}t_{bk}t_{al}
 - 1.0\sum_{kcd}v_{kacd}t_{di}t_{cj}t_{bk}
 + 1.0\sum_{kcd}v_{kbcd}t_{di}t_{cj}t_{ak}
 - 1.0\sum_{klcd}v_{klcd}t_{ak}t_{dl}t_{bcji}
 - 0.5\sum_{klcd}v_{klcd}t_{bk}t_{al}t_{dcji}
 + 1.0\sum_{klcd}v_{klcd}t_{bk}t_{dl}t_{acji}
 - 1.0\sum_{klcd}v_{klcd}t_{di}t_{ak}t_{bclj}
 + 1.0\sum_{klcd}v_{klcd}t_{di}t_{bk}t_{aclj}
 - 0.5\sum_{klcd}v_{klcd}t_{di}t_{cj}t_{balk}
 + 1.0\sum_{klcd}v_{klcd}t_{di}t_{ck}t_{balj}
 + 1.0\sum_{klcd}v_{klcd}t_{dj}t_{ak}t_{bcli}
 - 1.0\sum_{klcd}v_{klcd}t_{dj}t_{bk}t_{acli}
 - 1.0\sum_{klcd}v_{klcd}t_{dj}t_{ck}t_{bali}
)TEX",
                                wccsd.idx_map, wccsd.perm_map);
    t2_ref[4] = WickExpr::parse(R"TEX(
 + 1.0\sum_{klcd}v_{klcd}t_{di}t_{cj}t_{bk}t_{al}
)TEX",
                                wccsd.idx_map, wccsd.perm_map);
    WickExpr t1_eq = WickCCSD().t1_equations();
    // cout << t1_eq << endl;
    // cout << t1_ref << endl;
    WickExpr diff = (t1_eq - t1_ref).simplify();
    cout << "DIFF T1 = " << diff << endl;
    EXPECT_TRUE(diff.terms.size() == 0);
    for (int i = 0; i <= 4; i++) {
        WickExpr t2_eq = WickCCSD().t2_equations(i);
        WickExpr x_t2_ref = t2_ref[0];
        for (int j = 1; j <= i; j++)
            x_t2_ref = x_t2_ref + t2_ref[j];
        // cout << t2_eq << endl;
        // cout << x_t2_ref << endl;
        WickExpr diff = (t2_eq - x_t2_ref).simplify();
        cout << "DIFF T2 (order = " << i << ") = " << diff << endl;
        EXPECT_TRUE(diff.terms.size() == 0);
    }
}
