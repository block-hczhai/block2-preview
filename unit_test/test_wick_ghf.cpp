
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestWickGHF : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickGHF, TestGHF) {

    WickGHF wg;
    vector<WickExpr> h2_ref(4), h2_eq(4); // aa, bb, ab, ba
    h2_ref[1] = WickExpr::parse(R"TEX(
 + 0.5 SUM <ijkl> v[jilk] C[j] C[l] D[k] D[i]
 - 1.0 SUM <ijk> v[ijkk] C[i] D[j]
 + 1.0 SUM <ijk> v[ijki] C[j] D[k]
 + 0.5 SUM <ik> v[iikk]
 - 0.5 SUM <ij> v[ijji]
)TEX",
                                wg.idx_map[1], wg.perm_map)
                    .expand()
                    .simplify();

    h2_ref[2] = WickExpr::parse(R"TEX(
 - 0.5 SUM <ijkl> v[ijkl] C[i] C[l] D[k] D[j]
 + 0.5 SUM <ijk> v[ijkk] C[i] D[j]
)TEX",
                                wg.idx_map[2], wg.perm_map)
                    .expand()
                    .simplify();

    h2_ref[3] = WickExpr::parse(R"TEX(
 - 0.5 SUM <ijkl> v[ijkl] C[j] C[k] D[l] D[i]
 + 0.5 SUM <ikl> v[iikl] C[k] D[l]
)TEX",
                                wg.idx_map[3], wg.perm_map)
                    .expand()
                    .simplify();

    h2_eq[1] = wg.make_h2bb();
    h2_eq[2] = wg.make_h2ab();
    h2_eq[3] = wg.make_h2ba();

    vector<string> eq_names = {"aa", "bb", "ab", "ba"};

    for (auto &ix : vector<int>{1, 2, 3}) {
        WickExpr diff = (h2_eq[ix] - h2_ref[ix]).simplify();
        cout << "DIFF H2" << eq_names[ix] << " = " << diff << endl;
        EXPECT_TRUE(diff.terms.size() == 0);
    }
}
