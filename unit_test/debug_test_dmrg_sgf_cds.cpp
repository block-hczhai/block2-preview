
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
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            1);
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
    using FC = complex<double>;
    using FP = typename TestFixture::FP;
    using S = SGF;

    PGTypes pg = PGTypes::D2H;

    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    fcidump->read("data/N2.STO3G.FCIDUMP");
    fcidump = make_shared<SpinOrbitalFCIDUMP<FL>>(fcidump);
    cout << "INT end .. T = " << t.get_time() << endl;

    Random::rand_seed(1234);

    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    S vacuum(0);
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    vector<typename S::pg_t> pg_sym = HamiltonianQC<S, FL>::combine_orb_sym(
        orbsym, fcidump->template k_sym<int>(), fcidump->k_mod());
    uint8_t isym = PointGroup::swap_pg(pg)(fcidump->isym());
    S target(fcidump->n_elec(), 0, S::pg_combine(isym, 0, fcidump->k_mod()));
    shared_ptr<HamiltonianQC<S, FL>> hamil =
        make_shared<HamiltonianQC<S, FL>>(vacuum, norb, pg_sym, fcidump);

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<MPOQC<S, FL>>(
        hamil, QCTypes::Conventional, "HQC", hamil->n_sites / 2 / 4 * 4, 2);
    mpo->basis = hamil->basis;
    mpo = make_shared<CondensedMPO<S, FL>>(mpo, mpo->basis);
    // mpo = make_shared<CondensedMPO<S, FL>>(mpo, mpo->basis);
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            false, false);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    shared_ptr<MPO<S, FL>> pmpo = make_shared<PDM1MPOQC<S, FL>>(hamil);
    pmpo->basis = hamil->basis;
    pmpo = make_shared<CondensedMPO<S, FL>>(pmpo, pmpo->basis, true);
    // pmpo = make_shared<CondensedMPO<S, FL>>(pmpo, pmpo->basis, true);
    pmpo = make_shared<SimplifiedMPO<S, FL>>(pmpo, make_shared<RuleQC<S, FL>>(),
                                             false, false);
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        mpo->n_sites, hamil->vacuum, target, mpo->basis);
    mps_info->set_bond_dimension(bond_dim);

    Random::rand_seed(585076219);

    shared_ptr<MPS<S, FL>> mps = make_shared<MPS<S, FL>>(mpo->n_sites, 0, 2);
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
    // vector<ubond_t> bdims = {50};
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<FP> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5, 1E-5,
                         1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6, 0};
    vector<FP> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 5E-7,
                          5E-7, 5E-7, 5E-7, 5E-7, 1E-10};
    // noises = vector<double>{1E-5};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    // dmrg->cpx_me = me_cpx;
    // dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
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
    dmrg->solve(20, true, 0);

    shared_ptr<MovingEnvironment<S, FL, FL>> pme =
        make_shared<MovingEnvironment<S, FL, FL>>(pmpo, mps, mps, "1PDM");
    pme->init_environments(true);

    shared_ptr<Expect<S, FL, FL>> expect =
        make_shared<Expect<S, FL, FL>>(pme, 500, 500);
    expect->solve(true, mps->center == 0);

    GMatrix<FL> dm = expect->get_1pdm(hamil->n_sites);

    vector<tuple<int, int, double>> one_pdm = {
        {0, 0, 1.999989282592},  {0, 1, -0.000025398134},
        {0, 2, 0.000238560621},  {1, 0, -0.000025398134},
        {1, 1, 1.991431489457},  {1, 2, -0.005641787787},
        {2, 0, 0.000238560621},  {2, 1, -0.005641787787},
        {2, 2, 1.985471515555},  {3, 3, 1.999992764813},
        {3, 4, -0.000236022833}, {3, 5, 0.000163863520},
        {4, 3, -0.000236022833}, {4, 4, 1.986371259953},
        {4, 5, 0.018363506969},  {5, 3, 0.000163863520},
        {5, 4, 0.018363506969},  {5, 5, 0.019649294772},
        {6, 6, 1.931412559660},  {7, 7, 0.077134636900},
        {8, 8, 1.931412559108},  {9, 9, 0.077134637190}};

    int kk[2] = {0, 0};
    for (int i = 0; i < dm.m; i++)
        for (int j = 0; j < dm.n; j++)
            if (i % 2 != j % 2)
                EXPECT_LT(abs(dm(i, j)), 1E-5);
            else if (abs(dm(i, j)) > TINY) {
                int ii = i / 2, jj = j / 2, p = i % 2;

                cout << "== SGF 1PDM / " << 0 << "-site ==" << setw(6)
                     << (p == 0 ? "alpha" : "beta") << setw(5) << ii << setw(5)
                     << jj << fixed << setw(22) << fixed << setprecision(12)
                     << dm(i, j) << " error = " << scientific << setprecision(3)
                     << setw(10) << abs(dm(i, j) - get<2>(one_pdm[kk[p]]) / 2)
                     << endl;

                EXPECT_EQ(ii, get<0>(one_pdm[kk[p]]));
                EXPECT_EQ(jj, get<1>(one_pdm[kk[p]]));
                EXPECT_LT(abs(dm(i, j) - get<2>(one_pdm[kk[p]]) / 2), 1E-5);

                kk[p]++;
            }

    EXPECT_EQ(kk[0], (int)one_pdm.size());
    EXPECT_EQ(kk[1], (int)one_pdm.size());

    dm.deallocate();
}
