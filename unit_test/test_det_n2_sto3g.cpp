
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestDETN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;
    typedef typename GMatrix<FL>::FP FP;

    template <typename S>
    void test_dmrg(const S target,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            8);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

template <typename FL>
template <typename S>
void TestDETN2STO3G<FL>::test_dmrg(
    const S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
    const string &name) {

    vector<FL> coeffs = {
        -0.000000915576, -0.000000022619, -0.000002897952, 0.000006060239,
        -0.000002862448, 0.000018360654,  -0.000002862448, 0.000018360654,
        -0.000000038806, -0.000006230805, 0.000015602435,  -0.000002855035,
        0.000029426221,  -0.000002855035, 0.000029426221,  -0.000000088575,
        0.000000238778,  -0.000000117027, 0.000000712452,  -0.000000117027,
        0.000000712452,  0.000037600676,  -0.000008086563, 0.000059495484,
        -0.000008086562, 0.000059495484,  0.000041595105,  -0.000296818876,
        0.000041595102,  -0.000296818876, 0.000016747993,  -0.000039809347,
        0.000135579963,  0.000135579961,  -0.000952459919, 0.000016747993,
        -0.000001085865, -0.000308349332, 0.000440829937,  0.000030996291,
        0.000890887776,  0.000030996273,  0.000890887779,  -0.000003044489,
        0.000007766923,  -0.000003163272, 0.000019982712,  -0.000003163272,
        0.000019982712,  0.001962320994,  -0.000257239828, 0.001458694467,
        -0.000257239805, 0.001458694460,  0.001176010948,  -0.011074379840,
        0.001176010905,  -0.011074379818, 0.000473847561,  -0.002164578901,
        0.004951081277,  0.004951081533,  -0.022990862937, 0.000473847489,
        -0.000006399550, 0.000017030053,  -0.000003370007, 0.000031707052,
        -0.000003370007, 0.000031707052,  0.005025658393,  -0.000437488316,
        0.002059361932,  -0.000437488271, 0.002059361920,  0.002079042998,
        -0.025170595253, 0.002079042999,  -0.025170595224, 0.000724564491,
        -0.003662294041, 0.006822428224,  0.006822428653,  -0.026993007119,
        0.000724564430,  0.000038438609,  -0.000008173491, 0.000060674685,
        -0.000008173491, 0.000060674684,  0.000045920434,  -0.000321840930,
        0.000045920431,  -0.000321840929, 0.000017470649,  -0.000040509801,
        0.000139532821,  0.000139532818,  -0.000988913460, 0.000017470650,
        0.006834841262,  -0.054710489174, 0.006834840873,  -0.054710489028,
        0.000857531645,  -0.000558296661, 0.002406764888,  0.002406764828,
        -0.013142810649, 0.000857531619,  -0.012693540752, 0.038464933006,
        -0.131287878961, -0.131287877970, 0.957506526257,  -0.012693540764,
        0.001861231561,  -0.011546222107, 0.001861231727,  -0.011546222162};

    int norb = hamil->n_sites;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;

    // MPSInfo
    shared_ptr<MPSInfo<S>> mps_info =
        make_shared<MPSInfo<S>>(norb, hamil->vacuum, target, hamil->basis);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);
    shared_ptr<MPS<S, FL>> mps = make_shared<MPS<S, FL>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<S, FL, FL>> me =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_<FP>() << endl;
    frame_<FP>()->activate(0);

    // DMRG
    vector<ubond_t> bdims = {bond_dim};
    vector<FP> noises = {1E-8, 0.0};
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->solve(10, true, 1E-13);

    shared_ptr<DeterminantTRIE<S, FL>> dtrie =
        make_shared<DeterminantTRIE<S, FL>>(mps->n_sites, true);

    shared_ptr<DeterminantTRIE<S, FL>> dtrie_ref =
        make_shared<DeterminantTRIE<S, FL>>(mps->n_sites, true);

    vector<uint8_t> ref = {0, 0, 0, 3, 3, 3, 3, 3, 3, 3};
    do {
        dtrie_ref->push_back(ref);
    } while (next_permutation(ref.begin(), ref.end()));

    dtrie->evaluate(make_shared<UnfusedMPS<S, FL>>(mps), 1E-7);
    cout << dtrie->size() << endl;

    for (int i = 0; i < (int)dtrie_ref->size(); i++) {
        vector<uint8_t> det = (*dtrie_ref)[i];
        cout << name << " === ";
        for (auto x : det)
            cout << (int)x;
        int ii = dtrie->find(det);
        FL val = ii != -1 ? dtrie->vals[ii] : 0.0;
        cout << " = " << setw(22) << fixed << setprecision(12) << val
             << " error = " << scientific << setprecision(3) << setw(10)
             << (abs(val) - abs(coeffs[i])) << endl;

        EXPECT_LT(abs(abs(val) - abs(coeffs[i])), 1E-7);
    }

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
}

#ifdef _USE_COMPLEX
typedef ::testing::Types<complex<double>, double> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestDETN2STO3G, TestFL);

TYPED_TEST(TestDETN2STO3G, TestSU2) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    shared_ptr<HamiltonianQC<SU2, FL>> hamil =
        make_shared<HamiltonianQC<SU2, FL>>(vacuum, fcidump->n_sites(), orbsym,
                                            fcidump);

    this->template test_dmrg<SU2>(target, hamil, "SU2");

    hamil->deallocate();
    fcidump->deallocate();
}

TYPED_TEST(TestDETN2STO3G, TestSZ) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));
    shared_ptr<HamiltonianQC<SZ, FL>> hamil =
        make_shared<HamiltonianQC<SZ, FL>>(vacuum, fcidump->n_sites(), orbsym,
                                           fcidump);

    this->template test_dmrg<SZ>(target, hamil, " SZ");

    hamil->deallocate();
    fcidump->deallocate();
}
