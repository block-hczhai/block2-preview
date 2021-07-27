
#include "block2_big_site.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestCSFSpace : public ::testing::Test {
  protected:
    static const int n_tests = 200;
    size_t isize = 1L << 20;
    size_t dsize = 1L << 30;
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

TEST_F(TestCSFSpace, TestCSFSpace) {
    shared_ptr<CSFSpace<SU2>> csf_space =
        make_shared<CSFSpace<SU2>>(4, 4, true);
    vector<pair<pair<MKL_INT, MKL_INT>, double>> mat;
    for (int i = 0; i < csf_space->csf_offsets.back(); i++)
        cout << i << " " << (*csf_space)[i] << endl;
    int k = 0;
    for (int i = 0; i < csf_space->basis->n; i++)
        for (int j = csf_space->qs_idxs[i]; j < csf_space->qs_idxs[i + 1];
             j++) {
            // int i = 1, j = 1;
            cout << "::" << i << " " << j << " " << k << endl;
            csf_space->apply_creation_op(i, j, k, mat);
        }
}
