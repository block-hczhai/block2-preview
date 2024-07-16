
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
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 28,
            28, 1);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

#ifdef _USE_SINGLE_PREC
typedef ::testing::Types<float> TestFL;
#elif _USE_COMPLEX
typedef ::testing::Types<complex<double>> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestDMRG, TestFL);

TYPED_TEST(TestDMRG, Test) {
    using FL = TypeParam;
    using FP = typename TestFixture::FP;
    using S = SZ;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    vector<double> occs;
    string occ_filename = "data/CR2.SVP.OCC";
    // string occ_filename = "../my_test/cuprate/new2/CPR.CCSD.OCC";
    occs = read_occ(occ_filename);
    string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string filename = "../my_test/cuprate/new2/FCIDUMP";
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "original const = " << fcidump->e() << endl;
    fcidump->rescale();
    cout << "rescaled const = " << fcidump->e() << endl;
    cout << "INT end .. T = " << t.get_time() << endl;
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);
    S vacuum(0);
    S target(fcidump->n_elec(), fcidump->twos(),
             PointGroup::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<S, FL>> hamil =
        make_shared<HamiltonianQC<S, FL>>(vacuum, norb, orbsym, fcidump);

    // abort();

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(
        mpo, make_shared<RuleQC<S, FL>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    ubond_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo<S>> mps_info =
        make_shared<MPSInfo<S>>(norb, vacuum, target, hamil->basis);
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == norb);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs);
    }
    // cout << "left min dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->left_dims_fci[i].n << " ";
    // cout << endl;
    // cout << "right min dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->right_dims_fci[i].n << " ";
    // cout << endl;
    // cout << "left q dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->left_dims[i].n << " ";
    // cout << endl;
    // cout << "right q dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->right_dims[i].n << " ";
    // cout << endl;
    cout << "left dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;
    // abort();

    // MPS
    // Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    t.get_time();
    cout << "MPO start" << endl;
    shared_ptr<MPS<S, FL>> mps = make_shared<MPS<S, FL>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();
    cout << "MPS end .. T = " << t.get_time() << endl;

    // for (int i = 0; i < mps->n_sites; i++)
    //     if (mps->tensors[i] != nullptr)
    //         cout << *mps->tensors[i] << endl;
    // abort();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    frame_<FP>()->activate(0);
    cout << "persistent memory used :: I = " << ialloc_()->used
         << " D = " << dalloc_<FP>()->used << endl;
    frame_<FP>()->activate(1);
    cout << "exclusive  memory used :: I = " << ialloc_()->used
         << " D = " << dalloc_<FP>()->used << endl;
    // abort();
    // ME
    shared_ptr<MovingEnvironment<S, FL, FL>> me =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    // cout << *frame_<FP>() << endl;
    // frame_<FP>()->activate(0);
    // abort();

    // DMRG
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<FP> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5, 1E-5,
                         1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 0.0};
    vector<FP> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6, 1E-6, 1E-6,
                          1E-6, 1E-6, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    // vector<ubond_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->me->cached_contraction = true;
    dmrg->davidson_soft_max_iter = 200;
    dmrg->cutoff = 1E-20;
    dmrg->iprint = 2;
    dmrg->decomp_type = DecompositionTypes::SVD;
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->solve(25, true);

    shared_ptr<MPO<S, FL>> pmpo = make_shared<PDM1MPOQC<S, FL>>(hamil);
    pmpo = make_shared<SimplifiedMPO<S, FL>>(pmpo, make_shared<RuleQC<S, FL>>(),
                                             true);
    shared_ptr<MovingEnvironment<S, FL, FL>> pme =
        make_shared<MovingEnvironment<S, FL, FL>>(pmpo, mps, mps, "1PDM");
    pme->init_environments(false);
    shared_ptr<Expect<S, FL, FL, FL>> expect =
        make_shared<Expect<S, FL, FL, FL>>(pme, 750, 750);
    expect->solve(true, mps->center == 0);
    GMatrix<FL> dm = expect->get_1pdm_spatial();
    dm.deallocate();

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
