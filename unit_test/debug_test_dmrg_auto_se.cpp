
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
        frame_<FP>()->fp_codec = make_shared<FPCodec<FP>>(1E-14, 8 * 1024);
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
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    // string filename = "data/H4.STO6G.R1.8.FCIDUMP"; // E = -2.1903842178
    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "original const = " << fcidump->e() << endl;
    fcidump->rescale();
    cout << "rescaled const = " << fcidump->e() << endl;
    cout << "INT end .. T = " << t.get_time() << endl;

    const bool singlet_embedding = false;
    int se_spin = 2;

    Random::rand_seed(1234);

    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    S vacuum(0);
    int norb = fcidump->n_sites();
    vector<typename S::pg_t> pg_sym = HamiltonianQC<S, FL>::combine_orb_sym(
        orbsym, fcidump->template k_sym<int>(), fcidump->k_mod());
    uint8_t isym = PointGroup::swap_pg(pg)(fcidump->isym());
    S target(fcidump->n_elec(), fcidump->twos(),
             S::pg_combine(isym, 0, fcidump->k_mod()));
    if (singlet_embedding)
        target = S(fcidump->n_elec() + se_spin, 0,
                   S::pg_combine(isym, 0, fcidump->k_mod()));
    shared_ptr<HamiltonianQC<S, FL>> hamil =
        make_shared<HamiltonianQC<S, FL>>(vacuum, norb, pg_sym, fcidump);

    shared_ptr<GeneralFCIDUMP<FL>> ifd = make_shared<GeneralFCIDUMP<FL>>();
    ifd->elem_type = ElemOpTypes::SU2;
    ifd->const_e = 0.0;
    // ifd->exprs.push_back("((C+D)0+D)1");
    // ifd->indices.push_back(vector<uint16_t>{0, 1, 2});
    // ifd->exprs.push_back("D");
    // ifd->indices.push_back(vector<uint16_t>{0});
    ifd->exprs.push_back("");
    ifd->indices.push_back(vector<uint16_t>{});
    ifd->data.push_back(vector<FL>{1.0});

    cout << *ifd << endl;

    ifd = ifd->adjust_order();

    cout << *ifd << endl;

    cout << SpinPermPattern(3).to_str() << endl;
    cout << SpinPermPattern(2).to_str() << endl;
    cout << SpinPermPattern(1).to_str() << endl;
    cout << SpinPermPattern(0).to_str() << endl;

    cout << SpinPermScheme::initialize_su2(2, "(T+T)0").to_str() << endl;

    fcidump->symmetrize(orbsym);
    shared_ptr<GeneralFCIDUMP<FL>> gfd =
        GeneralFCIDUMP<FL>::initialize_from_qc(fcidump, ElemOpTypes::SU2);

    // cout << *gfd << endl;

    vector<shared_ptr<SpinPermScheme>> psch(gfd->exprs.size());
    for (size_t ix = 0; ix < gfd->exprs.size(); ix++) {
        psch[ix] = make_shared<SpinPermScheme>(gfd->exprs[ix], true);
        cout << "=== " << ix << " ===" << endl;
        cout << psch[ix]->to_str() << endl;
    }

    gfd = gfd->adjust_order();

    // cout << "after" << endl;
    // cout << *gfd << endl;

    // abort();

    shared_ptr<GeneralHamiltonian<S, FL>> gham =
        make_shared<GeneralHamiltonian<S, FL>>(vacuum, norb, orbsym);

    shared_ptr<GeneralFCIDUMP<FL>> igfd =
        make_shared<GeneralFCIDUMP<FL>>(ElemOpTypes::SU2);
    igfd->exprs.push_back("");
    igfd->indices.push_back(vector<uint16_t>());
    igfd->data.push_back(vector<FL>{(FL)1.0});
    igfd->const_e = (FL)0.0;
    igfd = igfd->adjust_order();
    cout << igfd->data.size() << endl;
    cout << igfd->data[0].size() << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<GeneralMPO<S, FL>>(
        gham, igfd, MPOAlgorithmTypes::Bipartite, 1E-7, -1);
    impo->build();

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<GeneralMPO<S, FL>>(
        gham, gfd, MPOAlgorithmTypes::Bipartite, 1E-7, -1);
    mpo->build();
    // shared_ptr<MPO<S, FL>> mpo =
    //     make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    // mpo->basis = hamil->basis;
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

    // Local init
    // shared_ptr<DynamicMPSInfo<S>> mps_info =
    //     make_shared<DynamicMPSInfo<S>>(norb, vacuum, target, hamil->basis,
    //                                      hamil->orb_sym, ioccs);
    // mps_info->n_local = 4;
    // mps_info->set_bond_dimension(bond_dim);

    // Determinant init
    // shared_ptr<DeterminantMPSInfo<S>> mps_info =
    //     make_shared<DeterminantMPSInfo<S>>(norb, vacuum, target,
    //     hamil->basis,
    //                                      hamil->orb_sym, ioccs, fcidump);
    // mps_info->set_bond_dimension(bond_dim);
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
    auto st = mps->estimate_storage(mps_info);
    cout << "MPS memory = " << Parsing::to_size_string(st[0])
         << "; storage = " << Parsing::to_size_string(st[1]) << endl;
    auto st2 = mpo->estimate_storage(mps_info, 2);
    cout << "2-site MPO term = " << Parsing::to_size_string(st2[0])
         << "; memory = " << Parsing::to_size_string(st2[1])
         << "; storage = " << Parsing::to_size_string(st2[2]) << endl;
    auto st3 = mpo->estimate_storage(mps_info, 1);
    cout << "1-site MPO term = " << Parsing::to_size_string(st3[0])
         << "; memory = " << Parsing::to_size_string(st3[1])
         << "; storage = " << Parsing::to_size_string(st3[2]) << endl;
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // for (int i = 0; i < mps->n_sites; i++)
    //     if (mps->tensors[i] != nullptr)
    //         cout << *mps->tensors[i] << endl;
    // abort();

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

    shared_ptr<MPSInfo<S>> bra_info =
        make_shared<MPSInfo<S>>(norb, vacuum, target, hamil->basis);
    bra_info->set_bond_dimension(bond_dim);
    bra_info->tag = "BRA";

    shared_ptr<MPS<S, FL>> bra = make_shared<MPS<S, FL>>(norb, mps->center, 2);
    bra->initialize(bra_info);
    bra->random_canonicalize();
    bra->save_mutable();
    bra_info->save_mutable();

    // shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    // impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S,
    // FL>>());

    // shared_ptr<MovingEnvironment<S, FL, FL>> ime =
    //     make_shared<MovingEnvironment<S, FL, FL>>(impo, bra, mps, "RHS");
    // ime->init_environments();
    // vector<ubond_t> bra_dims = {250};
    // vector<ubond_t> ket_dims = {250};
    // shared_ptr<Linear<S, FL, FL>> linear =
    //     make_shared<Linear<S, FL, FL>>(ime, bra_dims, ket_dims);
    // linear->cutoff = 1E-14;
    // FL igf = linear->solve(20, bra->center == 0, 1E-12);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
