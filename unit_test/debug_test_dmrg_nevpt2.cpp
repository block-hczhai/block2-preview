
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include "block2_big_site.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        cout << "MKL INTEGER SIZE = " << sizeof(MKL_INT) << endl;
        Random::rand_seed(0);
        frame_<double>() = make_shared<DataFrame<double>>(isize, dsize, "nodex");
        frame_<double>()->use_main_stack = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 16,
            16, 1);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<double>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<double>() = nullptr;
    }
};

TEST_F(TestDMRG, Test) {
    // string filename = "data/N2.CAS.PVDZ.T3.FCIDUMP";
    // string filename = "data/N2.CAS.PVDZ.T2.FCIDUMP";
    // string filename = "data/N2.CAS.PVDZ.T1.FCIDUMP";
    string filename = "data/N2.CAS.PVDZ.T0.FCIDUMP";

    Timer t;
    t.get_time();
    Random::rand_seed(0);

    // [T3] -108.939479334269393 | -109.132774263389905
    // uint16_t n_thawed = 3, n_external = 15;
    // [T2] -108.996699933457450 | -109.138615305899393
    // uint16_t n_thawed = 2, n_external = 16;
    // [T1] -109.006414147634061 | -109.139948066590108
    // uint16_t n_thawed = 1, n_external = 17;
    // [T0] -108.996662308260198 | -109.138741359593240
    uint16_t n_thawed = 0, n_external = 18;

    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    fcidump->read(filename);

    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);

    double err = fcidump->symmetrize(orbsym);
    cout << "symm err = " << err << endl;

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_d2h(fcidump->isym()));
    int n_orbs = fcidump->n_sites();

    // Full Hamiltonian
    shared_ptr<BigSite<SZ>> big_left = make_shared<SCIFockBigSite<SZ>>(
        n_orbs, n_thawed, false, fcidump, orbsym, -2, -2, -2, true);
    big_left =
        make_shared<SimplifiedBigSite<SZ>>(big_left, make_shared<RuleQC<SZ>>());
    shared_ptr<BigSite<SZ>> big_right = make_shared<SCIFockBigSite<SZ>>(
        n_orbs, n_external, true, fcidump, orbsym, 2, 2, 2, true);
    big_right = make_shared<SimplifiedBigSite<SZ>>(big_right,
                                                   make_shared<RuleQC<SZ>>());
    shared_ptr<HamiltonianQCBigSite<SZ>> hamil =
        make_shared<HamiltonianQCBigSite<SZ>>(
            vacuum, n_orbs, orbsym, fcidump, n_thawed == 0 ? nullptr : big_left,
            big_right);

    // MPO construction
    cout << "MPO start" << endl;
    cout << "ORB  LEFT = " << hamil->n_orbs_left << endl;
    cout << "ORB RIGHT = " << hamil->n_orbs_right << endl;
    t.get_time();
    int ntg = threading_()->n_threads_global;
    threading_()->n_threads_global = 1;
    shared_ptr<MPO<SZ>> mpo = make_shared<MPOQC<SZ>>(hamil, QCTypes::NC);
    threading_()->n_threads_global = ntg;
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true,
                                         true,
                                         OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 500;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<CASCIMPSInfo<SZ>>(
        hamil->n_sites, vacuum, target, hamil->basis, (int)!!n_thawed,
        hamil->n_orbs_cas, (int)!!n_external);
    mps_info->set_bond_dimension(bond_dim);
    cout << "left dims = ";
    for (int i = 0; i <= hamil->n_sites; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= hamil->n_sites; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;

    // MPS
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(hamil->n_sites, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(true);
    me->delayed_contraction = OpNamesSet::normal_ops();
    me->cached_contraction = true;
    cout << "INIT end .. T = " << t.get_time() << endl;

    // DMRG
    vector<ubond_t> bdims = {500, 500, 500,  500,  500,  750,  750, 750,
                             750, 750, 1000, 1000, 1000, 1000, 1000};
    vector<double> noises = {1E-2, 1E-2, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3,
                             1E-3, 1E-3, 1E-4, 1E-4, 1E-5, 1E-5, 0};
    vector<double> davthrs = {1E-6, 1E-6, 1E-7, 1E-7,  1E-8,
                              1E-8, 1E-9, 1E-9, 1E-10, 1E-10};
    // vector<ubond_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRGBigSite<SZ>> dmrg =
        make_shared<DMRGBigSite<SZ>>(me, bdims, noises);
    dmrg->last_site_svd = true;
    dmrg->last_site_1site = true;
    dmrg->decomp_last_site = false;
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    long double e_casci = dmrg->solve(30, true);

    cout << "E(CASCI) = " << setprecision(15) << e_casci << endl;

    if (mps->center == mps->n_sites - 1 && mps->dot == 2)
        mps->center = mps->n_sites - 2;
    cout << mps->canonical_form << endl;

    // 1PDM MPO construction
    cout << "1PDM MPO start" << endl;
    shared_ptr<MPO<SZ>> pmpo = make_shared<PDM1MPOQC<SZ>>(hamil);
    cout << "1PDM MPO end .. T = " << t.get_time() << endl;

    // 1PDM MPO simplification
    cout << "1PDM MPO simplification start" << endl;
    pmpo = make_shared<SimplifiedMPO<SZ>>(
        pmpo, make_shared<RuleQC<SZ>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;

    // 1PDM ME
    shared_ptr<MovingEnvironment<SZ>> pme =
        make_shared<MovingEnvironment<SZ>>(pmpo, mps, mps, "1PDM");
    t.get_time();
    pme->init_environments(true);
 
    // 1PDM
    shared_ptr<Expect<SZ>> expect =
        make_shared<Expect<SZ>>(pme, bond_dim, bond_dim);
    expect->solve(true, mps->center == 0);

    MatrixRef dm = expect->get_1pdm_spatial(hamil->n_orbs);
    shared_ptr<DyallFCIDUMP> fd_dyall =
        make_shared<DyallFCIDUMP>(fcidump, n_thawed, n_external);
    fd_dyall->initialize_from_1pdm_su2(dm);
    dm.deallocate();

    double errd = fd_dyall->symmetrize(orbsym);
    cout << "symm err = " << errd << endl;

    // No Trans Full hamiltonian
    if (n_thawed != 0)
        hamil->big_left = make_shared<SimplifiedBigSite<SZ>>(
            dynamic_pointer_cast<SimplifiedBigSite<SZ>>(hamil->big_left)
                ->big_site,
            make_shared<NoTransposeRule<SZ>>(make_shared<RuleQC<SZ>>()));
    hamil->big_right = make_shared<SimplifiedBigSite<SZ>>(
        dynamic_pointer_cast<SimplifiedBigSite<SZ>>(hamil->big_right)->big_site,
        make_shared<NoTransposeRule<SZ>>(make_shared<RuleQC<SZ>>()));

    // Dyall hamiltonian
    big_left = make_shared<SCIFockBigSite<SZ>>(
        n_orbs, n_thawed, false, fd_dyall, orbsym, -2, -2, -2, true);
    big_left =
        make_shared<SimplifiedBigSite<SZ>>(big_left, make_shared<RuleQC<SZ>>());
    big_right = make_shared<SCIFockBigSite<SZ>>(
        n_orbs, n_external, true, fd_dyall, orbsym, 2, 2, 2, true);
    big_right = make_shared<SimplifiedBigSite<SZ>>(big_right,
                                                   make_shared<RuleQC<SZ>>());
    shared_ptr<HamiltonianQCBigSite<SZ>> hm_dyall =
        make_shared<HamiltonianQCBigSite<SZ>>(
            vacuum, n_orbs, orbsym, fd_dyall,
            n_thawed == 0 ? nullptr : big_left, big_right);

    // Left MPO construction
    cout << "LMPO start" << endl;
    t.get_time();
    threading_()->n_threads_global = 1;
    shared_ptr<MPO<SZ>> mpo_dyall =
        make_shared<MPOQC<SZ>>(hm_dyall, QCTypes::NC);
    threading_()->n_threads_global = ntg;
    cout << "LMPO end .. T = " << t.get_time() << endl;

    // Left MPO simplification
    cout << "LMPO simplification start" << endl;
    mpo_dyall = make_shared<SimplifiedMPO<SZ>>(
        mpo_dyall, make_shared<RuleQC<SZ>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "LMPO simplification end .. T = " << t.get_time() << endl;

    // Right MPO construction
    cout << "RMPO start" << endl;
    t.get_time();
    threading_()->n_threads_global = 1;
    shared_ptr<MPO<SZ>> rmpo = make_shared<MPOQC<SZ>>(hamil, QCTypes::NC);
    threading_()->n_threads_global = ntg;
    cout << "RMPO end .. T = " << t.get_time() << endl;

    // Right MPO simplification
    cout << "RMPO simplification start" << endl;
    rmpo = make_shared<SimplifiedMPO<SZ>>(
        rmpo, make_shared<NoTransposeRule<SZ>>(make_shared<RuleQC<SZ>>()), true,
        true, OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "RMPO simplification end .. T = " << t.get_time() << endl;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> bra_info =
        make_shared<MPSInfo<SZ>>(hamil->n_sites, vacuum, target, hamil->basis);
    bra_info->set_bond_dimension(bond_dim);
    bra_info->tag = "BRA";
    cout << "left dims = ";
    for (int i = 0; i <= hamil->n_sites; i++)
        cout << bra_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= hamil->n_sites; i++)
        cout << bra_info->right_dims[i]->n_states_total << " ";
    cout << endl;

    // MPS
    shared_ptr<MPS<SZ>> bra =
        make_shared<MPS<SZ>>(hamil->n_sites, mps->center, 2);
    bra->initialize(bra_info);
    bra->random_canonicalize();
    bra->save_mutable();
    bra->deallocate();
    bra_info->save_mutable();
    bra_info->deallocate_mutable();

    cout << mps->canonical_form << endl;
    if (bra->center == 0 && bra->dot == 2)
        bra->move_left(hamil->opf->cg);
    if (bra->center == bra->n_sites - 2 && bra->dot == 2)
        bra->move_right(hamil->opf->cg);
    bra->center = mps->center;
    cout << bra->canonical_form << endl;

    // Linear: - (H0 - E0) |psi1> = (H - E0) | psi0>
    // DE(PT2) = <psi1| (H - E0) | psi0>
    mpo_dyall->const_e -= e_casci;
    mpo_dyall = mpo_dyall * -1;
    shared_ptr<MovingEnvironment<SZ>> lme =
        make_shared<MovingEnvironment<SZ>>(mpo_dyall, bra, bra, "LME");
    lme->init_environments(true);
    lme->delayed_contraction = OpNamesSet::normal_ops();

    rmpo->const_e -= e_casci;
    shared_ptr<MovingEnvironment<SZ>> rme =
        make_shared<MovingEnvironment<SZ>>(rmpo, bra, mps, "RME");
    rme->init_environments(true);

    // Linear
    vector<ubond_t> right_bdims = {(ubond_t)(mps->info->bond_dim + 1000)};
    shared_ptr<LinearBigSite<SZ>> linear = make_shared<LinearBigSite<SZ>>(
        lme, rme, nullptr, bdims, right_bdims, noises);
    linear->last_site_svd = true;
    linear->last_site_1site = true;
    linear->decomp_last_site = false;
    linear->linear_conv_thrds = vector<double>(20, 1E-10);
    linear->iprint = 2;
    linear->decomp_type = DecompositionTypes::DensityMatrix;
    linear->noise_type = NoiseTypes::ReducedPerturbative;
    double e_corr = linear->solve(30, mps->center == 0);

    cout << "E(NEVPT2) = " << setprecision(15) << e_casci + e_corr << endl;

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo_dyall->deallocate();
    hm_dyall->deallocate();
    fcidump->deallocate();
}
