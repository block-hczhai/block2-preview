
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestWickICNEVPT2 : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickICNEVPT2, TestICNEVPT2) {

    WickICNEVPT2 wic;
    cout << wic.to_einsum() << endl;

}
