
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
        frame_() = make_shared<DataFrame>(isize, dsize, "nodexx");
        frame_()->use_main_stack = false;
        frame_()->minimal_disk_usage = true;
        frame_()->fp_codec = make_shared<FPCodec<double>>(1E-14, 8 * 1024);
        // threading_() = make_shared<Threading>(ThreadingTypes::BatchedGEMM |
        // ThreadingTypes::Global, 8, 8);
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 28,
            28, 1);
        // threading_() =
        // make_shared<Threading>(ThreadingTypes::OperatorQuantaBatchedGEMM |
        // ThreadingTypes::Global, 16, 16, 16, 16);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *frame_() << endl;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

typedef ::testing::Types<double> TestFL;

TYPED_TEST_CASE(TestDMRG, TestFL);

TYPED_TEST(TestDMRG, Test) {
    using FL = TypeParam;
    using FC = complex<double>;
    using FP = typename TestFixture::FP;
    using S = SGF;

    PGTypes pg = PGTypes::C1;

    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    fcidump->read("data/H2O.SD.REAL");
    shared_ptr<FCIDUMP<FC>> fd_cpx_h1e = make_shared<FCIDUMP<FC>>();
    fd_cpx_h1e->read("data/H2O.SD.CPX");
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
    shared_ptr<HamiltonianQC<S, FC>> hamil_cpx =
        make_shared<HamiltonianQC<S, FC>>(vacuum, norb, pg_sym, fd_cpx_h1e);

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    mpo = make_shared<SimplifiedMPO<S, FL>>(
        mpo, make_shared<RuleQC<S, FL>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    shared_ptr<MPO<S, FC>> mpo_cpx =
        make_shared<MPOQC<S, FC>>(hamil_cpx, QCTypes::Conventional);
    mpo_cpx = make_shared<SimplifiedMPO<S, FC>>(
        mpo_cpx, make_shared<RuleQC<S, FC>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));

    ubond_t bond_dim = 200;
    vector<S> targets = {target};

    shared_ptr<MultiMPSInfo<S>> mps_info = make_shared<MultiMPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, targets, hamil->basis);
    mps_info->set_bond_dimension(bond_dim);

    Random::rand_seed(585076219);

    shared_ptr<MultiMPS<S, FL>> mps =
        make_shared<MultiMPS<S, FL>>(hamil->n_sites, 0, 2, 4);
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

    shared_ptr<MovingEnvironment<S, FC, FL>> me_cpx =
        make_shared<MovingEnvironment<S, FC, FL>>(mpo_cpx, mps, mps,
                                                  "DMRG-CPX");
    t.get_time();
    cout << "INIT start" << endl;
    me_cpx->init_environments(false);

    cout << *frame_() << endl;
    frame_()->activate(0);

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
    dmrg->cpx_me = me_cpx;
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    // dmrg->me->cached_contraction = true;
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
