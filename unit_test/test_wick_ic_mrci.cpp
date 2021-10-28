
#include "ic/wick.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestWickICMRCI : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestWickICMRCI, TestICMRCI) {

    WickICMRCI wic;
    cout << wic.to_einsum() << endl;

}
