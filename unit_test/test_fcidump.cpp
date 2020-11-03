
#include "block2.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestFCIDUMP : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

TEST_F(TestFCIDUMP, TestRead) {
    FCIDUMP fcidump;
    string filename = "data/CR2.SVP.FCIDUMP";
    fcidump.read(filename);
    uint8_t cr2_orbsym[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5,
                        5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 6, 6,
                        6, 6, 3, 3, 3, 3, 7, 7, 7, 7, 4, 4, 8, 8};
    vector<uint8_t> orbsym = fcidump.orb_sym();
    EXPECT_TRUE(equal(orbsym.begin(), orbsym.end(), cr2_orbsym));
    EXPECT_EQ(fcidump.n_sites(), 42);
    EXPECT_EQ(fcidump.n_elec(), 48);
    EXPECT_EQ(fcidump.isym(), 1);
    EXPECT_EQ(fcidump.twos(), 0);
    EXPECT_EQ(fcidump.uhf, false);
    EXPECT_LT(abs(fcidump.const_e - 181.4321866011428597), 1E-15);
    EXPECT_LT(abs(fcidump.ts[0](0, 0) - (-295.3905965169317369)), 1E-15);
    EXPECT_LT(abs(fcidump.ts[0](0, 3) - 0.1592759388906264), 1E-15);
    EXPECT_EQ(fcidump.ts[0](0, 3), fcidump.ts[0](3, 0));
    EXPECT_EQ(fcidump.ts[0](1, 3), 0.0);
    EXPECT_LT(abs(fcidump.vs[0](0, 2, 1, 1) - (-0.0000819523306820)), 1E-15);
    EXPECT_EQ(fcidump.vs[0](0, 2, 1, 1), fcidump.vs[0](2, 0, 1, 1));
    EXPECT_EQ(fcidump.vs[0](0, 2, 1, 1), fcidump.vs[0](1, 1, 0, 2));
    EXPECT_EQ(fcidump.vs[0](0, 2, 1, 1), fcidump.vs[0](1, 1, 2, 0));
    fcidump.deallocate();
}
