
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
        frame_<FP>()->fp_codec = make_shared<FPCodec<FP>>(1E-14, 8 * 1024);
        // threading_() = make_shared<Threading>(ThreadingTypes::BatchedGEMM |
        // ThreadingTypes::Global, 8, 8);
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 1, 1,
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

#ifdef _USE_SINGLE_PREC
typedef ::testing::Types<double, float> TestFL;
#elif _USE_COMPLEX
typedef ::testing::Types<double, complex<double>> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestDMRG, TestFL);

TYPED_TEST(TestDMRG, Test) {
    using FL = TypeParam;
    using FP = typename TestFixture::FP;
    using S = SU2;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;
    // string occ_filename = "data/CR2.SVP.OCC";
    // string occ_filename = "data/CR2.SVP.HF"; // E(HF) = -2085.53318786766
    // string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    // string filename = "data/HUBBARD-L2.FCIDUMP"; // E = -1.2360679775
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    // string filename = "data/H4.STO6G.R1.8.FCIDUMP"; // E = -2.1903842178

    string filename =
        "../my_test/test_partial_mpo/FCIDUMP"; // E = -107.65412235

    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "original const = " << fcidump->e() << endl;
    fcidump->rescale();
    cout << "rescaled const = " << fcidump->e() << endl;
    cout << "INT end .. T = " << t.get_time() << endl;

    const bool singlet_embedding = true;
    int se_spin = 0;

    Random::rand_seed(1234);

    S vacuum(0);
    int n_sites = fcidump->n_sites(), norb = n_sites;

    vector<uint8_t> orb_sym = fcidump->template orb_sym<uint8_t>();
    transform(orb_sym.begin(), orb_sym.end(), orb_sym.begin(),
              PointGroup::swap_pg(pg));
    uint8_t isym = PointGroup::swap_pg(pg)(fcidump->isym());
    S target(fcidump->n_elec(), fcidump->twos(),
             S::pg_combine(isym, 0, fcidump->k_mod()));
    if (singlet_embedding)
        target = S(fcidump->n_elec() + se_spin, 0,
                   S::pg_combine(isym, 0, fcidump->k_mod()));

    auto error = fcidump->symmetrize(orb_sym);
    cout << "symm err = " << setprecision(5) << scientific << error << endl;

    shared_ptr<GeneralHamiltonian<S, FL>> gham =
        make_shared<GeneralHamiltonian<S, FL>>(vacuum, n_sites, orb_sym);

    shared_ptr<GeneralFCIDUMP<FL>> gfd = make_shared<GeneralFCIDUMP<FL>>();
    gfd->elem_type = ElemOpTypes::SU2;
    gfd->const_e = fcidump->e();
    gfd->exprs.push_back("(C+D)0");
    gfd->indices.push_back(vector<uint16_t>());
    gfd->data.push_back(vector<FL>());
    for (int i = 0; i < n_sites; i++)
        for (int j = 0; j < n_sites; j++) {
            gfd->indices.back().push_back(i);
            gfd->indices.back().push_back(j);
            gfd->data.back().push_back(sqrt((FL)2.0) * fcidump->t(i, j));
        }
    gfd->exprs.push_back("(C+((C+D)0+D)1)0");
    gfd->indices.push_back(vector<uint16_t>());
    gfd->data.push_back(vector<FL>());
    for (int i = 0; i < n_sites; i++)
        for (int j = 0; j < n_sites; j++)
            for (int k = 0; k < n_sites; k++)
                for (int l = 0; l < n_sites; l++) {
                    gfd->indices.back().push_back(i);
                    gfd->indices.back().push_back(k);
                    gfd->indices.back().push_back(l);
                    gfd->indices.back().push_back(j);
                    gfd->data.back().push_back(2.0 * 0.5 *
                                               fcidump->v(i, j, k, l));
                }

    gfd = gfd->adjust_order();

    // ifd->exprs.push_back("((C+D)0+D)1");
    // ifd->indices.push_back(vector<uint16_t>{0, 1, 2});
    // ifd->exprs.push_back("D");
    // ifd->indices.push_back(vector<uint16_t>{0});
    // ifd->exprs.push_back("");
    // ifd->indices.push_back(vector<uint16_t>{});
    // ifd->data.push_back(vector<FL>{1.0});

    // cout << *ifd << endl;

    // ifd = ifd->adjust_order();

    // cout << *ifd << endl;

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<GeneralMPO<S, FL>>(
        gham, gfd, MPOAlgorithmTypes::FastBipartite, 1E-7, -1);
    mpo->build();
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(
        mpo, make_shared<RuleQC<S, FL>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    mpo = make_shared<IdentityAddedMPO<S, FL>>(mpo);

    ubond_t bond_dim = 200;

    // MPSInfo
    // shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
    //     norb, vacuum, target, hamil->basis);

    // CCSD init
    shared_ptr<MPSInfo<S>> mps_info =
        make_shared<MPSInfo<S>>(norb, vacuum, target, mpo->basis);
    if (singlet_embedding) {
        S left_vacuum = S(se_spin, se_spin, 0);
        mps_info->set_bond_dimension_full_fci(left_vacuum, vacuum);
    }
    // mps_info->set_bond_dimension_full_fci();
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == norb || occs.size() == norb * 4);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs, 1);
        // mps_info->set_bond_dimension_using_hf(bond_dim, occs, 0);
    }

    cout << "left mpo dims = ";
    for (int i = 0; i < norb; i++)
        cout << mpo->left_operator_names[i]->data.size() << " ";
    cout << endl;
    cout << "right mpo dims = ";
    for (int i = 0; i < norb; i++)
        cout << mpo->right_operator_names[i]->data.size() << " ";
    cout << endl;

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
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
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
    // vector<ubond_t> bdims = {50};
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<FP> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5,
                         1E-5, 1E-5, 1E-5, 1E-6, 0.0};
    vector<FP> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6,
                          1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                          5E-7, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    FP tol = 1E-6;
    if (is_same<FP, float>::value) {
        for (auto &x : davthrs)
            x = max(x, (FP)5E-6);
        tol = 5E-6;
    }
    // noises = vector<double>{1E-5};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->cached_contraction = true;
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->davidson_soft_max_iter = 200;
    dmrg->iprint = 2;
    dmrg->cutoff = 1E-20;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    // dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollectedLowMem;
    // dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected;
    dmrg->trunc_type = dmrg->trunc_type;
    // dmrg->davidson_type = DavidsonTypes::GreaterThan;
    // dmrg->davidson_shift = -2086.4;
    dmrg->solve(25, true, tol);

    for (int ii = 0; ii < n_sites; ii++) {

        shared_ptr<GeneralFCIDUMP<FL>> bfd = make_shared<GeneralFCIDUMP<FL>>();
        bfd->elem_type = ElemOpTypes::SU2;
        // bfd->const_e = fcidump->e();
        bfd->exprs.push_back("D");
        bfd->indices.push_back(vector<uint16_t>());
        bfd->data.push_back(vector<FL>());
        for (int j = 0; j < n_sites; j++) {
            bfd->indices.back().push_back(j);
            bfd->data.back().push_back(fcidump->t(ii, j));
        }
        bfd->exprs.push_back("((C+D)0+D)1");
        bfd->indices.push_back(vector<uint16_t>());
        bfd->data.push_back(vector<FL>());
        for (int j = 0; j < n_sites; j++)
            for (int k = 0; k < n_sites; k++)
                for (int l = 0; l < n_sites; l++) {
                    bfd->indices.back().push_back(k);
                    bfd->indices.back().push_back(l);
                    bfd->indices.back().push_back(j);
                    bfd->data.back().push_back(sqrt((FL)2.0) * 0.5 *
                                               fcidump->v(ii, j, k, l));
                }

        bfd = bfd->adjust_order();

        shared_ptr<MPO<S, FL>> pmpo = make_shared<GeneralMPO<S, FL>>(
            gham, bfd, MPOAlgorithmTypes::FastBipartite, 1E-7, -1);
        pmpo->build();
        pmpo = make_shared<SimplifiedMPO<S, FL>>(
            pmpo, make_shared<RuleQC<S, FL>>(), false, false);
        pmpo = make_shared<IdentityAddedMPO<S, FL>>(pmpo);

        S bra_q = pmpo->op->q_label + mps->info->target;

        // cout << pmpo->op->q_label << " " << pmpo->left_vacuum << " "
        //      << mps->info->target << endl;

        shared_ptr<MPSInfo<S>> bra_info =
            make_shared<MPSInfo<S>>(norb, vacuum, bra_q, mpo->basis);
        bra_info->set_bond_dimension_full_fci(pmpo->left_vacuum, vacuum);
        bra_info->set_bond_dimension(bond_dim);
        bra_info->tag = "BRA";

        shared_ptr<MPS<S, FL>> bra =
            make_shared<MPS<S, FL>>(norb, mps->center, 2);
        bra->initialize(bra_info);
        bra->random_canonicalize();
        bra->save_mutable();
        bra_info->save_mutable();

        // cout << *bra->info->left_dims_fci[0] << " "
        //      << *mps->info->left_dims_fci[0] << endl;

        shared_ptr<MovingEnvironment<S, FL, FL>> pme =
            make_shared<MovingEnvironment<S, FL, FL>>(pmpo, bra, mps, "RHS");
        pme->delayed_contraction = OpNamesSet::normal_ops();
        pme->cached_contraction = true;
        pme->init_environments(false);
        vector<ubond_t> bra_dims = {500};
        vector<ubond_t> ket_dims = {500};
        shared_ptr<Linear<S, FL, FL>> linear =
            make_shared<Linear<S, FL, FL>>(pme, bra_dims, ket_dims);
        linear->iprint = 0;
        linear->cutoff = 1E-14;
        FL igf = linear->solve(20, bra->center == 0, 1E-12);
    }
}
