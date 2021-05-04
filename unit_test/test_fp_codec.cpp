
#include "block2.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestFPCodec : public ::testing::Test {
  protected:
    static const int n_tests = 1000;
    void SetUp() override { Random::rand_seed(0); }
    void TearDown() override {}
};

TEST_F(TestFPCodec, TestDoubleFPCodec) {
    for (int i = 0; i < n_tests; i++) {
        int n;
        if (i < n_tests * 10 / 100)
            n = Random::rand_int(1, 12);
        else if (i < n_tests * 40 / 100)
            n = Random::rand_int(1, 10000);
        else
            n = Random::rand_int(1, 50000);
        int chunk_size = Random::rand_int(1, 1 + n * 4 / 3);
        vector<double> arr(n), arx(n);
        if (Random::rand_int(0, 10) != 0)
            Random::fill_rand_double(arr.data(), n, -5, 5);
        FPCodec<double> fpc(1E-8, chunk_size);
        stringstream ss;
        fpc.write_array(ss, arr.data(), n);
        ss.clear();
        ss.seekg(0);
        fpc.read_array(ss, arx.data(), n);
        EXPECT_TRUE(MatrixFunctions::all_close(
            MatrixRef(arr.data(), n, 1), MatrixRef(arx.data(), n, 1), 2E-8, 0));
    }
}

TEST_F(TestFPCodec, TestFloatFPCodec) {
    for (int i = 0; i < n_tests; i++) {
        int n;
        if (i < n_tests * 10 / 100)
            n = Random::rand_int(1, 12);
        else if (i < n_tests * 40 / 100)
            n = Random::rand_int(1, 10000);
        else
            n = Random::rand_int(1, 50000);
        int chunk_size = Random::rand_int(1, 1 + n * 4 / 3);
        vector<float> arr(n), arx(n);
        if (Random::rand_int(0, 10) != 0)
            Random::fill_rand_float(arr.data(), n, -5, 5);
        FPCodec<float> fpc(1E-4, chunk_size);
        stringstream ss;
        fpc.write_array(ss, arr.data(), n);
        ss.clear();
        ss.seekg(0);
        fpc.read_array(ss, arx.data(), n);
        for (int j = 0; j < n; j++)
            EXPECT_LT(abs(arr[j] - arx[j]), 2E-4);
    }
}

TEST_F(TestFPCodec, TestDoubleCompressedVector) {
    for (int i = 0; i < n_tests; i++) {
        int n;
        if (i < n_tests * 10 / 100)
            n = Random::rand_int(1, 12);
        else if (i < n_tests * 40 / 100)
            n = Random::rand_int(1, 10000);
        else
            n = Random::rand_int(1, 50000);
        int chunk_size = Random::rand_int(5, 100);
        vector<double> arr(n), arx(n);
        if (Random::rand_int(0, 10) != 0)
            Random::fill_rand_double(arr.data(), n, -5, 5);
        double prec = 1E-8;
        FPCodec<double> fpc(prec, chunk_size);
        stringstream ss;
        fpc.write_array(ss, arr.data(), n);
        ss.clear();
        ss.seekg(0);
        CompressedVector<double> carr(ss, n, prec);
        const CompressedVector<double> &ccarr = carr;
        for (int j = 0; j < n / 4 + 1; j++) {
            int h = Random::rand_int(0, n);
            if (Random::rand_int(0, 2))
                EXPECT_LE(abs(ccarr[h] - arr[h]), 2 * prec);
            else {
                arr[h] = Random::rand_double(-5, 5);
                carr[h] = arr[h];
            }
        }
        carr.clear();
        memset(arr.data(), 0, sizeof(double) * arr.size());
        for (int j = 0; j < n / 4 + 1; j++) {
            int h = Random::rand_int(0, n);
            if (Random::rand_int(0, 2))
                EXPECT_LE(abs(ccarr[h] - arr[h]), 2 * prec);
            else {
                arr[h] = Random::rand_double(-5, 5);
                carr[h] = arr[h];
            }
        }
    }
}

TEST_F(TestFPCodec, TestFloatCompressedVector) {
    for (int i = 0; i < n_tests; i++) {
        int n;
        if (i < n_tests * 10 / 100)
            n = Random::rand_int(1, 12);
        else if (i < n_tests * 40 / 100)
            n = Random::rand_int(1, 10000);
        else
            n = Random::rand_int(1, 50000);
        int chunk_size = Random::rand_int(5, 100);
        vector<float> arr(n), arx(n);
        if (Random::rand_int(0, 10) != 0)
            Random::fill_rand_float(arr.data(), n, -5, 5);
        float prec = 1E-4;
        FPCodec<float> fpc(prec, chunk_size);
        stringstream ss;
        fpc.write_array(ss, arr.data(), n);
        ss.clear();
        ss.seekg(0);
        CompressedVector<float> carr(ss, n, prec);
        const CompressedVector<float> &ccarr = carr;
        for (int j = 0; j < n / 4 + 1; j++) {
            int h = Random::rand_int(0, n);
            if (Random::rand_int(0, 2))
                EXPECT_LE(abs(ccarr[h] - arr[h]), 2 * prec);
            else {
                arr[h] = (float)Random::rand_double(-5, 5);
                carr[h] = arr[h];
            }
        }
        carr.clear();
        memset(arr.data(), 0, sizeof(float) * arr.size());
        for (int j = 0; j < n / 4 + 1; j++) {
            int h = Random::rand_int(0, n);
            if (Random::rand_int(0, 2))
                EXPECT_LE(abs(ccarr[h] - arr[h]), 2 * prec);
            else {
                arr[h] = (float)Random::rand_double(-5, 5);
                carr[h] = arr[h];
            }
        }
    }
}
