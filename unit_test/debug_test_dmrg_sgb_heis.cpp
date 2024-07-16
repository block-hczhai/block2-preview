
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1LL << 30;
    size_t dsize = 1LL << 34;
    typedef typename GMatrix<FL>::FP FP;

    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        cout << "MKL INTEGER SIZE = " << sizeof(MKL_INT) << endl;
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodexx");
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        frame_<FP>()->fp_codec = make_shared<FPCodec<double>>(1E-14, 8 * 1024);
        // threading_() = make_shared<Threading>(ThreadingTypes::BatchedGEMM |
        // ThreadingTypes::Global, 8, 8);
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 28,
            28, 1);
        // threading_() =
        // make_shared<Threading>(ThreadingTypes::OperatorQuantaBatchedGEMM |
        // ThreadingTypes::Global, 16, 16, 16, 16);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *frame_<FP>() << endl;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

typedef ::testing::Types<double> TestFL;

TYPED_TEST_CASE(TestDMRG, TestFL);

TYPED_TEST(TestDMRG, Test) {
    using FL = TypeParam;
    using FP = typename TestFixture::FP;
    using S = SGB;

    PGTypes pg = PGTypes::C1;

    Timer t;
    t.get_time();

    Random::rand_seed(1234);

    S vacuum(0, 0), target(0, 0);
    shared_ptr<GeneralFCIDUMP<FL>> gfd =
        make_shared<GeneralFCIDUMP<FL>>(ElemOpTypes::SGB);
    // int n_sites = 100, twos = 2;
    // gfd->exprs = vector<string>{"PM", "MP", "ZZ"};
    // gfd->indices.resize(3);
    // gfd->data.resize(3);
    // for (int ix = 0; ix < 3; ix++)
    //     for (int i = 0; i < n_sites - 1; i++) {
    //         gfd->indices[ix].push_back(i);
    //         gfd->indices[ix].push_back(i + 1);
    //         gfd->data[ix].push_back(ix == 2 ? 1.0 : 0.5);
    //     }
    // gfd = gfd->adjust_order();
    int n_sites = 4, twos = 1;
    gfd->exprs = vector<string>{"PM", "MP", "ZZ"};
    gfd->indices.resize(3);
    gfd->data.resize(3);
    for (int ix = 0; ix < 3; ix++)
        for (int i = 0; i < n_sites - 1; i += 2) {
            gfd->indices[ix].push_back(i);
            gfd->indices[ix].push_back(i + 1);
            gfd->data[ix].push_back(ix == 2 ? 1.0 : 0.5);
        }
    gfd = gfd->adjust_order();

    shared_ptr<GeneralHamiltonian<S, FL>> gham =
        make_shared<GeneralHamiltonian<S, FL>>(
            vacuum, n_sites, vector<typename S::pg_t>(), twos);

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<GeneralMPO<S, FL>>(
        gham, gfd, MPOAlgorithmTypes::FastBipartite, 1E-14, -1);
    mpo->build();
    mpo = make_shared<SimplifiedMPO<S, FL>>(
        mpo, make_shared<RuleQC<S, FL>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<S> targets = {target};

    shared_ptr<MultiMPSInfo<S>> mps_info =
        make_shared<MultiMPSInfo<S>>(n_sites, vacuum, targets, mpo->basis);
    mps_info->set_bond_dimension(bond_dim);

    Random::rand_seed(585076219);

    shared_ptr<MultiMPS<S, FL>> mps =
        make_shared<MultiMPS<S, FL>>(n_sites, 0, 2, 4);
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
    me->init_environments(true);
    cout << "INIT end .. T = " << t.get_time() << endl;

    // DMRG
    // vector<ubond_t> bdims = {50};
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500, 500,
                             500, 500, 500, 500, 500, 500, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<FP> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5,
                         1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6};
    vector<FP> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6,
                          1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                          5E-7, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    // noises = vector<double>{1E-5};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->cached_contraction = true;
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    // dmrg->cutoff = 1E-20;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    // dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollectedLowMem;
    // dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected;
    dmrg->trunc_type = dmrg->trunc_type;
    // dmrg->davidson_type = DavidsonTypes::GreaterThan;
    // dmrg->davidson_shift = -2086.4;
    dmrg->solve(20, true);
}
