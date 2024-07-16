
#include "block2_big_site.hpp"
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1LL << 30;
    size_t dsize = 1LL << 34;
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        cout << "MKL INTEGER SIZE = " << sizeof(MKL_INT) << endl;
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 16,
            16, 1);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

TEST_F(TestDMRG, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    // string occ_filename = "data/CR2.SVP.OCC";
    // string occ_filename = "../my_test/cuprate/new2/CPR.CCSD.OCC";
    // occs = read_occ(occ_filename);
    string filename = "data/H2.6-31GSS.FCIDUMP"; // E = -1.1369814718
    // string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string filename = "../my_test/cuprate/new2/FCIDUMP";
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "INT end .. T = " << t.get_time() << endl;
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);
    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    int norb_ext = 7, norb_thaw = 1;
    int n_alpha = 1, n_beta = 1;

    // BIG LEFT
    shared_ptr<BigSite<SZ>> big_left = make_shared<SCIFockBigSite<SZ>>(
        norb, norb_thaw, false, fcidump, orbsym, n_alpha, n_beta,
        n_alpha + n_beta, true);
    big_left =
        make_shared<SimplifiedBigSite<SZ>>(big_left, make_shared<RuleQC<SZ>>());

    // BIG RIGHT
    shared_ptr<BigSite<SZ>> big_right = make_shared<SCIFockBigSite<SZ>>(
        norb, norb_ext, true, fcidump, orbsym, n_alpha, n_beta,
        n_alpha + n_beta, true);
    big_right = make_shared<SimplifiedBigSite<SZ>>(big_right,
                                                   make_shared<RuleQC<SZ>>());
    shared_ptr<HamiltonianQCBigSite<SZ>> hamil =
        make_shared<HamiltonianQCBigSite<SZ>>(vacuum, norb, orbsym, fcidump,
                                              big_left, big_right);

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    cout << "ORB  LEFT = " << hamil->n_orbs_left << endl;
    cout << "ORB RIGHT = " << hamil->n_orbs_right << endl;
    int ntg = threading_()->n_threads_global;
    threading_()->n_threads_global = 1;
    shared_ptr<MPO<SZ>> mpo = make_shared<MPOQC<SZ>>(hamil, QCTypes::NC);
    threading_()->n_threads_global = ntg;
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true,
                                         true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    ubond_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info =
        make_shared<MPSInfo<SZ>>(hamil->n_sites, vacuum, target, hamil->basis);
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
    for (int i = 0; i <= hamil->n_sites; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= hamil->n_sites; i++)
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
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(hamil->n_sites, 0, 2);
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

    frame_<double>()->activate(0);
    cout << "persistent memory used :: I = " << ialloc_()->used
         << " D = " << dalloc_<FP>()->used << endl;
    frame_<double>()->activate(1);
    cout << "exclusive  memory used :: I = " << ialloc_()->used
         << " D = " << dalloc_<FP>()->used << endl;
    // abort();
    // ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(true);
    cout << "INIT end .. T = " << t.get_time() << endl;

    // cout << *frame_<double>() << endl;
    // frame_<double>()->activate(0);
    // abort();

    // DMRG
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-5, 1E-5, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7,
                             1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 0.0};
    vector<double> davthrs = {2.5E-5, 2.5E-5, 2.5E-5, 2.5E-5, 1E-6, 1E-6,
                              1E-6,   1E-8,   1E-8,   1E-8,   1E-8};
    // vector<ubond_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRGBigSite<SZ>>(me, bdims, noises);
    // dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    dmrg->decomp_type = DecompositionTypes::SVD;
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->solve(30, true);

    // shared_ptr<MPO<SZ>> pmpo = make_shared<PDM1MPOQC<SZ>>(hamil);
    // pmpo =
    //     make_shared<SimplifiedMPO<SZ>>(pmpo, make_shared<RuleQC<SZ>>(),
    //     true);
    // shared_ptr<MovingEnvironment<SZ>> pme =
    //     make_shared<MovingEnvironment<SZ>>(pmpo, mps, mps, "1PDM");
    // pme->init_environments(true);
    // shared_ptr<Expect<SZ>> expect = make_shared<Expect<SZ>>(pme, 750, 750);
    // expect->solve(true, mps->center == 0);
    // MatrixRef dm = expect->get_1pdm_spatial();
    // dm.deallocate();

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
