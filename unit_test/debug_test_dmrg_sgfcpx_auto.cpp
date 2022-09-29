
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
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
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 4,
            4, 1);
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

typedef ::testing::Types<complex<double>> TestFL;

TYPED_TEST_CASE(TestDMRG, TestFL);

TYPED_TEST(TestDMRG, Test) {
    using FL = TypeParam;
    using FP = typename TestFixture::FP;
    using S = SGF;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    Timer t;
    t.get_time();

    Random::rand_seed(1234);

    int n_sites = 4, n_elec = 2;
    FL ecore = -74.33832207870078;
    map<pair<int, int>, FL> h1e;
    map<tuple<int, int, int, int>, FL> g2e;
    h1e[make_pair(0, 0)] = FL(-1.2531481, 0.0000000);
    h1e[make_pair(0, 2)] = FL(0.0004231, -0.0000000);
    h1e[make_pair(1, 1)] = FL(-1.2531481, 0.0000000);
    h1e[make_pair(1, 3)] = FL(0.0004036, 0.0001271);
    h1e[make_pair(2, 0)] = FL(0.0004231, 0.0000000);
    h1e[make_pair(2, 2)] = FL(-0.4947662, 0.0000000);
    h1e[make_pair(3, 1)] = FL(0.0004036, -0.0001271);
    h1e[make_pair(3, 3)] = FL(-0.4947662, -0.0000000);
    g2e[make_tuple(0, 0, 0, 0)] = FL(0.7592724, -0.0000000);
    g2e[make_tuple(0, 0, 0, 2)] = FL(-0.0004230, 0.0000000);
    g2e[make_tuple(0, 0, 1, 1)] = FL(0.7592724, -0.0000000);
    g2e[make_tuple(0, 0, 1, 3)] = FL(-0.0004034, -0.0001271);
    g2e[make_tuple(0, 0, 2, 0)] = FL(-0.0004230, -0.0000000);
    g2e[make_tuple(0, 0, 2, 2)] = FL(0.3442732, -0.0000000);
    g2e[make_tuple(0, 0, 3, 1)] = FL(-0.0004034, 0.0001271);
    g2e[make_tuple(0, 0, 3, 3)] = FL(0.3442732, -0.0000000);
    g2e[make_tuple(0, 2, 0, 0)] = FL(-0.0004230, 0.0000000);
    g2e[make_tuple(0, 2, 0, 2)] = FL(0.0000009, 0.0000000);
    g2e[make_tuple(0, 2, 0, 3)] = FL(0.0000003, 0.0000001);
    g2e[make_tuple(0, 2, 1, 1)] = FL(-0.0004230, 0.0000000);
    g2e[make_tuple(0, 2, 1, 2)] = FL(-0.0000003, -0.0000001);
    g2e[make_tuple(0, 2, 1, 3)] = FL(0.0000008, 0.0000003);
    g2e[make_tuple(0, 2, 2, 0)] = FL(0.0000009, -0.0000000);
    g2e[make_tuple(0, 2, 2, 1)] = FL(-0.0000003, 0.0000000);
    g2e[make_tuple(0, 2, 2, 2)] = FL(-0.0000078, 0.0000000);
    g2e[make_tuple(0, 2, 3, 0)] = FL(0.0000003, -0.0000000);
    g2e[make_tuple(0, 2, 3, 1)] = FL(0.0000008, -0.0000003);
    g2e[make_tuple(0, 2, 3, 3)] = FL(-0.0000078, 0.0000000);
    g2e[make_tuple(0, 3, 0, 2)] = FL(0.0000003, 0.0000001);
    g2e[make_tuple(0, 3, 0, 3)] = FL(0.0107282, 0.0025177);
    g2e[make_tuple(0, 3, 1, 2)] = FL(-0.0105106, -0.0033114);
    g2e[make_tuple(0, 3, 1, 3)] = FL(0.0000003, 0.0000001);
    g2e[make_tuple(0, 3, 2, 0)] = FL(0.0000003, 0.0000000);
    g2e[make_tuple(0, 3, 2, 1)] = FL(-0.0109889, 0.0008224);
    g2e[make_tuple(0, 3, 3, 0)] = FL(0.0110199, -0.0000000);
    g2e[make_tuple(0, 3, 3, 1)] = FL(0.0000003, -0.0000000);
    g2e[make_tuple(1, 1, 0, 0)] = FL(0.7592724, -0.0000000);
    g2e[make_tuple(1, 1, 0, 2)] = FL(-0.0004230, 0.0000000);
    g2e[make_tuple(1, 1, 1, 1)] = FL(0.7592724, 0.0000000);
    g2e[make_tuple(1, 1, 1, 3)] = FL(-0.0004034, -0.0001271);
    g2e[make_tuple(1, 1, 2, 0)] = FL(-0.0004230, -0.0000000);
    g2e[make_tuple(1, 1, 2, 2)] = FL(0.3442732, 0.0000000);
    g2e[make_tuple(1, 1, 3, 1)] = FL(-0.0004034, 0.0001271);
    g2e[make_tuple(1, 1, 3, 3)] = FL(0.3442732, 0.0000000);
    g2e[make_tuple(1, 2, 0, 2)] = FL(-0.0000003, -0.0000001);
    g2e[make_tuple(1, 2, 0, 3)] = FL(-0.0105106, -0.0033114);
    g2e[make_tuple(1, 2, 1, 2)] = FL(0.0102339, 0.0040865);
    g2e[make_tuple(1, 2, 1, 3)] = FL(-0.0000002, -0.0000001);
    g2e[make_tuple(1, 2, 2, 0)] = FL(-0.0000003, -0.0000000);
    g2e[make_tuple(1, 2, 2, 1)] = FL(0.0110199, 0.0000000);
    g2e[make_tuple(1, 2, 3, 0)] = FL(-0.0109889, -0.0008224);
    g2e[make_tuple(1, 2, 3, 1)] = FL(-0.0000003, 0.0000000);
    g2e[make_tuple(1, 3, 0, 0)] = FL(-0.0004034, -0.0001271);
    g2e[make_tuple(1, 3, 0, 2)] = FL(0.0000008, 0.0000003);
    g2e[make_tuple(1, 3, 0, 3)] = FL(0.0000003, 0.0000001);
    g2e[make_tuple(1, 3, 1, 1)] = FL(-0.0004034, -0.0001271);
    g2e[make_tuple(1, 3, 1, 2)] = FL(-0.0000002, -0.0000001);
    g2e[make_tuple(1, 3, 1, 3)] = FL(0.0000007, 0.0000005);
    g2e[make_tuple(1, 3, 2, 0)] = FL(0.0000008, 0.0000003);
    g2e[make_tuple(1, 3, 2, 1)] = FL(-0.0000003, -0.0000000);
    g2e[make_tuple(1, 3, 2, 2)] = FL(-0.0000075, -0.0000024);
    g2e[make_tuple(1, 3, 3, 0)] = FL(0.0000003, 0.0000000);
    g2e[make_tuple(1, 3, 3, 1)] = FL(0.0000009, -0.0000000);
    g2e[make_tuple(1, 3, 3, 3)] = FL(-0.0000075, -0.0000024);
    g2e[make_tuple(2, 0, 0, 0)] = FL(-0.0004230, -0.0000000);
    g2e[make_tuple(2, 0, 0, 2)] = FL(0.0000009, 0.0000000);
    g2e[make_tuple(2, 0, 0, 3)] = FL(0.0000003, 0.0000000);
    g2e[make_tuple(2, 0, 1, 1)] = FL(-0.0004230, -0.0000000);
    g2e[make_tuple(2, 0, 1, 2)] = FL(-0.0000003, -0.0000000);
    g2e[make_tuple(2, 0, 1, 3)] = FL(0.0000008, 0.0000003);
    g2e[make_tuple(2, 0, 2, 0)] = FL(0.0000009, -0.0000000);
    g2e[make_tuple(2, 0, 2, 1)] = FL(-0.0000003, 0.0000001);
    g2e[make_tuple(2, 0, 2, 2)] = FL(-0.0000078, -0.0000000);
    g2e[make_tuple(2, 0, 3, 0)] = FL(0.0000003, -0.0000001);
    g2e[make_tuple(2, 0, 3, 1)] = FL(0.0000008, -0.0000003);
    g2e[make_tuple(2, 0, 3, 3)] = FL(-0.0000078, -0.0000000);
    g2e[make_tuple(2, 1, 0, 2)] = FL(-0.0000003, 0.0000000);
    g2e[make_tuple(2, 1, 0, 3)] = FL(-0.0109889, 0.0008224);
    g2e[make_tuple(2, 1, 1, 2)] = FL(0.0110199, -0.0000000);
    g2e[make_tuple(2, 1, 1, 3)] = FL(-0.0000003, -0.0000000);
    g2e[make_tuple(2, 1, 2, 0)] = FL(-0.0000003, 0.0000001);
    g2e[make_tuple(2, 1, 2, 1)] = FL(0.0102339, -0.0040865);
    g2e[make_tuple(2, 1, 3, 0)] = FL(-0.0105106, 0.0033114);
    g2e[make_tuple(2, 1, 3, 1)] = FL(-0.0000002, 0.0000001);
    g2e[make_tuple(2, 2, 0, 0)] = FL(0.3442732, -0.0000000);
    g2e[make_tuple(2, 2, 0, 2)] = FL(-0.0000078, 0.0000000);
    g2e[make_tuple(2, 2, 1, 1)] = FL(0.3442732, -0.0000000);
    g2e[make_tuple(2, 2, 1, 3)] = FL(-0.0000075, -0.0000024);
    g2e[make_tuple(2, 2, 2, 0)] = FL(-0.0000078, -0.0000000);
    g2e[make_tuple(2, 2, 2, 2)] = FL(0.2997146, 0.0000000);
    g2e[make_tuple(2, 2, 3, 1)] = FL(-0.0000075, 0.0000024);
    g2e[make_tuple(2, 2, 3, 3)] = FL(0.2997146, 0.0000000);
    g2e[make_tuple(3, 0, 0, 2)] = FL(0.0000003, -0.0000000);
    g2e[make_tuple(3, 0, 0, 3)] = FL(0.0110199, -0.0000000);
    g2e[make_tuple(3, 0, 1, 2)] = FL(-0.0109889, -0.0008224);
    g2e[make_tuple(3, 0, 1, 3)] = FL(0.0000003, 0.0000000);
    g2e[make_tuple(3, 0, 2, 0)] = FL(0.0000003, -0.0000001);
    g2e[make_tuple(3, 0, 2, 1)] = FL(-0.0105106, 0.0033114);
    g2e[make_tuple(3, 0, 3, 0)] = FL(0.0107282, -0.0025177);
    g2e[make_tuple(3, 0, 3, 1)] = FL(0.0000003, -0.0000001);
    g2e[make_tuple(3, 1, 0, 0)] = FL(-0.0004034, 0.0001271);
    g2e[make_tuple(3, 1, 0, 2)] = FL(0.0000008, -0.0000003);
    g2e[make_tuple(3, 1, 0, 3)] = FL(0.0000003, -0.0000000);
    g2e[make_tuple(3, 1, 1, 1)] = FL(-0.0004034, 0.0001271);
    g2e[make_tuple(3, 1, 1, 2)] = FL(-0.0000003, 0.0000000);
    g2e[make_tuple(3, 1, 1, 3)] = FL(0.0000009, 0.0000000);
    g2e[make_tuple(3, 1, 2, 0)] = FL(0.0000008, -0.0000003);
    g2e[make_tuple(3, 1, 2, 1)] = FL(-0.0000002, 0.0000001);
    g2e[make_tuple(3, 1, 2, 2)] = FL(-0.0000075, 0.0000024);
    g2e[make_tuple(3, 1, 3, 0)] = FL(0.0000003, -0.0000001);
    g2e[make_tuple(3, 1, 3, 1)] = FL(0.0000007, -0.0000005);
    g2e[make_tuple(3, 1, 3, 3)] = FL(-0.0000075, 0.0000024);
    g2e[make_tuple(3, 3, 0, 0)] = FL(0.3442732, 0.0000000);
    g2e[make_tuple(3, 3, 0, 2)] = FL(-0.0000078, 0.0000000);
    g2e[make_tuple(3, 3, 1, 1)] = FL(0.3442732, 0.0000000);
    g2e[make_tuple(3, 3, 1, 3)] = FL(-0.0000075, -0.0000024);
    g2e[make_tuple(3, 3, 2, 0)] = FL(-0.0000078, -0.0000000);
    g2e[make_tuple(3, 3, 2, 2)] = FL(0.2997146, -0.0000000);
    g2e[make_tuple(3, 3, 3, 1)] = FL(-0.0000075, 0.0000024);
    g2e[make_tuple(3, 3, 3, 3)] = FL(0.2997146, -0.0000000);

    S vacuum(0);
    S target(n_elec, 0);

    shared_ptr<GeneralHamiltonian<S, FL>> gham =
        make_shared<GeneralHamiltonian<S, FL>>(
            vacuum, n_sites, vector<typename S::pg_t>(n_sites, 0));

    shared_ptr<GeneralFCIDUMP<FL>> gfd = make_shared<GeneralFCIDUMP<FL>>();
    gfd->elem_type = ElemOpTypes::SGF;
    gfd->exprs.push_back("CD");
    gfd->indices.push_back(vector<uint16_t>());
    gfd->data.push_back(vector<FL>());
    for (auto &r : h1e) {
        gfd->indices.back().push_back(r.first.first);
        gfd->indices.back().push_back(r.first.second);
        gfd->data.back().push_back(r.second);
    }
    gfd->exprs.push_back("CCDD");
    gfd->indices.push_back(vector<uint16_t>());
    gfd->data.push_back(vector<FL>());
    for (auto &r : g2e) {
        gfd->indices.back().push_back(get<0>(r.first));
        gfd->indices.back().push_back(get<2>(r.first));
        gfd->indices.back().push_back(get<3>(r.first));
        gfd->indices.back().push_back(get<1>(r.first));
        gfd->data.back().push_back(0.5 * r.second);
    }
    gfd->const_e = ecore;

    cout << *gfd << endl;
    gfd = gfd->adjust_order();
    cout << *gfd << endl;

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<GeneralMPO<S, FL>>(
        gham, gfd, MPOAlgorithmTypes::FastSVD, 0, -1, true);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<Rule<S, FL>>(),
                                            // make_shared<Rule<S, FL>>(),
                                            false, false, OpNamesSet(), "HQC",
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    shared_ptr<GeneralFCIDUMP<FL>> ifd = make_shared<GeneralFCIDUMP<FL>>();
    ifd->elem_type = ElemOpTypes::SGF;
    ifd->exprs.push_back("CD");
    ifd->indices.push_back(vector<uint16_t>{0, 0});
    ifd->data.push_back(vector<FL>{1.0});
    cout << *ifd << endl;
    gfd = ifd->adjust_order();
    cout << *ifd << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<GeneralMPO<S, FL>>(
        gham, ifd, MPOAlgorithmTypes::SVD, 0.0, -1);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>(),
                                             false, false);
    impo = make_shared<IdentityAddedMPO<S, FL>>(impo);

    ubond_t bond_dim = 200;

    // MPSInfo
    // shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
    //     norb, vacuum, target, hamil->basis);

    // CCSD init
    shared_ptr<MPSInfo<S>> mps_info =
        make_shared<MPSInfo<S>>(n_sites, vacuum, target, mpo->basis);
    // mps_info->set_bond_dimension_full_fci();
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == n_sites || occs.size() * 2 == n_sites);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs, 1);
        // mps_info->set_bond_dimension_using_hf(bond_dim, occs, 0);
    }

    cout << "left mpo dims = ";
    for (int i = 0; i < n_sites; i++)
        cout << mpo->left_operator_names[i]->data.size() << " ";
    cout << endl;
    cout << "right mpo dims = ";
    for (int i = 0; i < n_sites; i++)
        cout << mpo->right_operator_names[i]->data.size() << " ";
    cout << endl;

    cout << "left dims = ";
    for (int i = 0; i <= n_sites; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= n_sites; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;
    // abort();

    // MPS
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    shared_ptr<MPS<S, FL>> mps = make_shared<MPS<S, FL>>(n_sites, 0, 2);
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
    me->save_partition_info = true;
    me->init_environments(true);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_<FP>() << endl;
    frame_<FP>()->activate(0);

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

    mps->dot = 1;
    shared_ptr<MPS<S, FL>> mket = mps->deep_copy("OPI");

    shared_ptr<MovingEnvironment<S, FL, FL>> ime =
        make_shared<MovingEnvironment<S, FL, FL>>(impo, mket, mket, "EX");
    ime->init_environments();
    shared_ptr<Expect<S, FL, FL, FL>> expect =
        make_shared<Expect<S, FL, FL, FL>>(ime, 500, 500);
    expect->iprint = 2;
    expect->cutoff = 1E-14;
    FL xx = expect->solve(false, mket->center != 0);

    cout << xx << endl;

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
}
