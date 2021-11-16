
#include "block2_big_site.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestCSFSpace : public ::testing::Test {
  protected:
    static const int n_tests = 200;
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;
    double norm(const vector<pair<pair<MKL_INT, MKL_INT>, double>> &mat) const {
        double normsq = 0;
        for (auto &mmat : mat)
            normsq += mmat.second * mmat.second;
        return sqrt(normsq);
    }
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
    shared_ptr<CSFSpace<SU2, double>> csf_space =
        make_shared<CSFSpace<SU2, double>>(3, 10, false);
    vector<pair<pair<MKL_INT, MKL_INT>, double>> mat;
    for (int i = 0; i < csf_space->csf_offsets.back(); i++)
        cout << i << " " << (*csf_space)[i] << endl;
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++) {
        auto p = csf_space->get_config(i);
        long long pp = csf_space->index_config(p);
        assert(pp == i);
        cout << i << " " << csf_space->to_string(p) << endl;
    }
    vector<uint8_t> x(3, 0);
    shared_ptr<CSFBigSite<SU2, double>> csf_bs =
        make_shared<CSFBigSite<SU2, double>>(csf_space, nullptr, x);
    // test 3-op
    const uint8_t i_ops = 0, c_ops = 3, d_ops = 2, c2_ops = 1, d2_ops = 0;
    const uint8_t a0_ops = c_ops + (c2_ops << 2), a1_ops = c_ops + (c_ops << 2);
    const uint8_t ad0_ops = d_ops + (d2_ops << 2),
                  ad1_ops = d_ops + (d_ops << 2);
    const uint8_t b0_ops = d_ops + (c2_ops << 2), b1_ops = d_ops + (c_ops << 2);
    const uint8_t bd0_ops = c_ops + (d2_ops << 2),
                  bd1_ops = c_ops + (d_ops << 2);
    const uint8_t dxx_ops = d_ops << 4, cxx_ops = c_ops << 4;
    const uint8_t dcd0_ops = d_ops + (c2_ops << 2) + (d_ops << 4);
    const uint8_t dcd1_ops = d_ops + (c_ops << 2) + (d2_ops << 4);
    const uint8_t cdd0_ops = c_ops + (d2_ops << 2) + (d_ops << 4);
    const uint8_t cdd1_ops = c_ops + (d_ops << 2) + (d2_ops << 4);
    const uint8_t ddc0_ops = d_ops + (d2_ops << 2) + (c_ops << 4);
    const uint8_t ddc1_ops = d_ops + (d_ops << 2) + (c2_ops << 4);
    const uint8_t cdc0_ops = c_ops + (d2_ops << 2) + (c_ops << 4);
    const uint8_t cdc1_ops = c_ops + (d_ops << 2) + (c2_ops << 4);
    const uint8_t dcc0_ops = d_ops + (c2_ops << 2) + (c_ops << 4);
    const uint8_t dcc1_ops = d_ops + (c_ops << 2) + (c2_ops << 4);
    const uint8_t ccd0_ops = c_ops + (c2_ops << 2) + (d_ops << 4);
    const uint8_t ccd1_ops = c_ops + (c_ops << 2) + (d2_ops << 4);
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddc0_ops, {1, 2, 2}, mat, -0.5);
    EXPECT_LT(abs(norm(mat) - 2.6220221204253793), 1E-12);
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, cdd0_ops, {1, 2, 2}, mat, -0.5);
    EXPECT_LT(abs(norm(mat) - 2.2360679774997902), 1E-12);
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddc1_ops, {1, 2, 2}, mat, -0.5 * sqrt(3));
    EXPECT_LT(abs(norm(mat) - 2.6220221204253793), 1E-12);

    // test 4-op
    const uint8_t ddcc0_ops =
        d_ops + (d2_ops << 2) + (c_ops << 4) + (c2_ops << 6); // 2
    const uint8_t ddcc1_ops =
        d_ops + (d_ops << 2) + (c2_ops << 4) + (c2_ops << 6); // 3
    const uint8_t ddcc2_ops =
        d_ops + (d2_ops << 2) + (c2_ops << 4) + (c_ops << 6); // 4
    const uint8_t ddcc3_ops =
        d2_ops + (d_ops << 2) + (c_ops << 4) + (c2_ops << 6); // 5
    const uint8_t ccdd0_ops =
        c_ops + (c2_ops << 2) + (d_ops << 4) + (d2_ops << 6); // 6
    const uint8_t ccdd1_ops =
        c_ops + (c_ops << 2) + (d2_ops << 4) + (d2_ops << 6); // 7
    const uint8_t cddc0_ops =
        c_ops + (d2_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 8
    const uint8_t cddc1_ops =
        c_ops + (d_ops << 2) + (d2_ops << 4) + (c2_ops << 6); // 9
    const uint8_t cddc2_ops =
        c_ops + (d2_ops << 2) + (d2_ops << 4) + (c_ops << 6); // 10
    const uint8_t cddc3_ops =
        c2_ops + (d_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 11
    const uint8_t cdcd0_ops =
        c_ops + (d2_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 12
    const uint8_t cdcd1_ops =
        c_ops + (d_ops << 2) + (c2_ops << 4) + (d2_ops << 6); // 13
    const uint8_t cdcd2_ops =
        c_ops + (d2_ops << 2) + (c2_ops << 4) + (d_ops << 6); // 14
    const uint8_t cdcd3_ops =
        c2_ops + (d_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 15
    const uint8_t dcdc0_ops =
        d_ops + (c2_ops << 2) + (d_ops << 4) + (c2_ops << 6); // 16
    const uint8_t dcdc1_ops =
        d_ops + (c_ops << 2) + (d2_ops << 4) + (c2_ops << 6); // 17
    const uint8_t dccd0_ops =
        d_ops + (c2_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 18
    const uint8_t dccd1_ops =
        d_ops + (c_ops << 2) + (c2_ops << 4) + (d2_ops << 6); // 19
    const uint8_t dccd2_ops =
        d_ops + (c2_ops << 2) + (c2_ops << 4) + (d_ops << 6); // 20
    const uint8_t dccd3_ops =
        d2_ops + (c_ops << 2) + (c_ops << 4) + (d2_ops << 6); // 21
    // E0 ddcc i = k = l = j
    for (uint16_t k = 0; k < 3; k++) {
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, ddcc2_ops, {k, k, k, k}, mat);
        EXPECT_LT(abs(norm(mat) - 3.162277660168381), 1E-12);
    }
    // E0 ddcc i > k = l = j
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, ddcc2_ops, {ka, ka, ka, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 1.58113883008419), 1E-12);
    }
    // E0 ddcc i > k = l > j
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddcc2_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.8284271247461907), 1E-12);
    // E1 ddcc i > k = l > j
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddcc3_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 1.6329931618554523), 1E-12);
    // D0 ddcc i = j = k > l
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, ddcc0_ops, {ka, kb, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 1.58113883008419), 1E-12);
    }
    // D1 ddcc i = j = k > l
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, ddcc1_ops, {ka, kb, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 2.738612787525831), 1E-12);
    }
    // A0 dccd l > i = j = k
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, dccd0_ops, {ka, ka, ka, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 1.58113883008419), 1E-12);
    }
    // A1 dccd l > i = j = k
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, dccd1_ops, {ka, ka, ka, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 2.738612787525831), 1E-12);
    }
    // E0 cdcd l > j = k > i
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, cdcd2_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.8284271247461907), 1E-12);
    // E1 cdcd l > j = k > i
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, cdcd3_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 1.6329931618554523), 1E-12);
    // C0 cddc i = j = l > k
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, cddc2_ops, {ka, kb, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 1.58113883008419), 1E-12);
    }
    // C1 cddc i = j = l > k
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, cddc3_ops, {ka, kb, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 2.738612787525831), 1E-12);
    }
    // E0 cddc i > j = l > k
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, cddc2_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.4494897427831788), 1E-12);
    // E1 cddc i > j = l > k
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, cddc3_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 0), 1E-12);
    // B0 ddcc i = k > j = l
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, ddcc0_ops, {ka, ka, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 3.464101615137756), 1E-12);
    }
    // B1 ddcc i = k > j = l
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, ddcc1_ops, {ka, ka, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 0), 1E-12);
    }
    // B0 ddcc i = k > l > j
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddcc0_ops, {0, 1, 2, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.4494897427831788), 1E-12);
    // B1 ddcc i = k > l > j
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddcc1_ops, {0, 1, 2, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 0), 1E-12);
    // B0 ddcc i > k > j = l
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddcc0_ops, {0, 0, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.4494897427831788), 1E-12);
    // B1 ddcc i > k > j = l
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, ddcc1_ops, {0, 0, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 0), 1E-12);
    // B0 dcdc i = l > j = k
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, dcdc0_ops, {ka, ka, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 4.663689526544409), 1E-12);
    }
    // B1 dcdc i = l > j = k
    for (uint16_t k = 0; k < 3; k++) {
        uint16_t ka = k == 2 ? 1 : 0, kb = k == 0 ? 1 : 2;
        mat.clear();
        for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
            csf_space->cfg_apply_ops(i, dcdc1_ops, {ka, ka, kb, kb}, mat);
        EXPECT_LT(abs(norm(mat) - 1.6072751268321592), 1E-12);
    }
    // B0 dcdc i = l > k > j
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, dcdc0_ops, {0, 1, 2, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.8284271247461907), 1E-12);
    // B1 dcdc i = l > k > j
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, dcdc1_ops, {0, 1, 2, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 1.6329931618554523), 1E-12);
    // B0 dcdc i > l > j = k
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, dcdc0_ops, {0, 0, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.8284271247461907), 1E-12);
    // B1 dcdc i > l > j = k
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, dcdc1_ops, {0, 0, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 1.6329931618554523), 1E-12);
    // E0 dccd i > i = k > l
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, dccd2_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 2.4494897427831788), 1E-12);
    // E1 dccd i > j = l > k
    mat.clear();
    for (int i = 0; i < csf_space->n_unpaired_idxs.back(); i++)
        csf_space->cfg_apply_ops(i, dccd3_ops, {0, 1, 1, 2}, mat);
    EXPECT_LT(abs(norm(mat) - 0), 1E-12);

    SU2 q(1, 1, 0);
    auto info = csf_bs->find_site_op_info(q);
    cout << *info << endl;
    shared_ptr<VectorAllocator<double>> d_alloc =
        make_shared<VectorAllocator<double>>();
    shared_ptr<CSRSparseMatrix<SU2, double>> matg =
        make_shared<CSRSparseMatrix<SU2, double>>();
    matg->initialize(info);
    csf_bs->build_site_op(c_ops, {2}, matg, 1);
}
