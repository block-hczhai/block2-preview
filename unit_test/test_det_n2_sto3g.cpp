
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDETN2STO3G : public ::testing::Test {
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

TEST_F(TestDETN2STO3G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(8);
    mkl_set_dynamic(0);
#endif

    vector<double> coeffs = {
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

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    hamil.opf->seq->mode = SeqTypes::Simple;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SZ>> mpo =
        make_shared<MPOQC<SZ>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 200;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
        norb, vacuum, target, hamil.basis);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_() << endl;
    frame_()->activate(0);

    // DMRG
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 0.0};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->solve(10, true, 1E-10);

    shared_ptr<DeterminantTRIE<SZ>> dtrie =
        make_shared<DeterminantTRIE<SZ>>(mps->n_sites, true);

    vector<uint8_t> ref = {0, 0, 0, 3, 3, 3, 3, 3, 3, 3};
    do {
        dtrie->push_back(ref);
    } while (next_permutation(ref.begin(), ref.end()));

    dtrie->evaluate(make_shared<UnfusedMPS<SZ>>(mps));

    for (int i = 0; i < (int)dtrie->size(); i++) {
        for (auto x : (*dtrie)[i])
            cout << (int)x;
        cout << " = " << setw(22) << fixed << setprecision(12) << dtrie->vals[i]
             << " error = " << scientific << setprecision(3) << setw(10)
             << (abs(dtrie->vals[i]) - abs(coeffs[i])) << endl;

        EXPECT_LT(abs(abs(dtrie->vals[i]) - abs(coeffs[i])), 1E-7);
    }

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
