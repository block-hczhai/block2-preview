
#include "quantum.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestFCIDUMP : public ::testing::Test {
  protected:
    size_t isize = 1E7;
    size_t dsize = 1E7;
    void SetUp() override {
        Random::rand_seed(0);
        ialloc = new StackAllocator<uint32_t>(new uint32_t[isize], isize);
        dalloc = new StackAllocator<double>(new double[dsize], dsize);
    }
    void TearDown() override {
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete[] ialloc->data;
        delete[] dalloc->data;
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
    EXPECT_LT(abs(fcidump.e - 181.4321866011428597), 1E-15);
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
