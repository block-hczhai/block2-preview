
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include "block2_mrci.hpp"
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
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 16,
            16, 1);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

TEST_F(TestDMRG, Test) {
    string filename =
        "data/N2.CAS.PVDZ.FCIDUMP";
    string fock_filename =
        "data/N2.CAS.PVDZ.FOCK";

    Timer t;
    t.get_time();
    Random::rand_seed(0);

    uint16_t n_thawed = 3, n_external = 15; // --108.939479334269393 | -109.132774263389905

    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    fcidump->read(filename);
    shared_ptr<FCIDUMP> fd_dyall =
        make_shared<DyallFCIDUMP>(fcidump, n_thawed, n_external);
    fd_dyall->read(fock_filename);

    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);

    double err = fcidump->symmetrize(orbsym);
    cout << "symm err = " << err << endl;
    double errd = fd_dyall->symmetrize(orbsym);
    cout << "symm err = " << errd << endl;

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_d2h(fcidump->isym()));
    int n_orbs = fcidump->n_sites();

    // BIG LEFT
    shared_ptr<BigSite<SZ>> big_left = make_shared<SCIFockBigSite<SZ>>(
        n_orbs, n_thawed, false, fd_dyall, orbsym, -2, -2, -2, true);
    big_left =
        make_shared<SimplifiedBigSite<SZ>>(big_left, make_shared<RuleQC<SZ>>());

    // BIG RIGHT
    shared_ptr<BigSite<SZ>> big_right = make_shared<SCIFockBigSite<SZ>>(
        n_orbs, n_external, true, fd_dyall, orbsym, 2, 2, 2, true);
    big_right = make_shared<SimplifiedBigSite<SZ>>(big_right,
                                                   make_shared<RuleQC<SZ>>());
    shared_ptr<HamiltonianQCBigSite<SZ>> hm_dyall =
        make_shared<HamiltonianQCBigSite<SZ>>(vacuum, n_orbs, orbsym, fd_dyall,
                                              big_left, big_right);

    // MPO construction
    cout << "MPO start" << endl;
    cout << "ORB  LEFT = " << hm_dyall->n_orbs_left << endl;
    cout << "ORB RIGHT = " << hm_dyall->n_orbs_right << endl;
    t.get_time();
    int ntg = threading_()->n_threads_global;
    threading_()->n_threads_global = 1;
    shared_ptr<MPO<SZ>> mpo_dyall =
        make_shared<MPOQC<SZ>>(hm_dyall, QCTypes::NC);
    threading_()->n_threads_global = ntg;
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo_dyall = make_shared<SimplifiedMPO<SZ>>(
        mpo_dyall, make_shared<RuleQC<SZ>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 500;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<CASCIMPSInfo<SZ>>(
        hm_dyall->n_sites, vacuum, target, hm_dyall->basis, (int)!!n_thawed,
        hm_dyall->n_orbs_cas, (int)!!n_external);
    mps_info->set_bond_dimension(bond_dim);
    cout << "left dims = ";
    for (int i = 0; i <= hm_dyall->n_sites; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= hm_dyall->n_sites; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;

    // MPS
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(hm_dyall->n_sites, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo_dyall, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(true);
    me->delayed_contraction = OpNamesSet::normal_ops();
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
    double e_casci = dmrg->solve(30, true);

    cout << "E(CASCI) = " << setprecision(15) << e_casci << endl;

    if (mps->center == mps->n_sites - 1 && mps->dot == 2)
        mps->center = mps->n_sites - 2;

    // Full hamiltonian
    big_left = make_shared<SCIFockBigSite<SZ>>(n_orbs, n_thawed, false, fcidump,
                                               orbsym, -2, -2, -2, true);
    big_left = make_shared<SimplifiedBigSite<SZ>>(
        big_left, make_shared<NoTransposeRule<SZ>>(make_shared<RuleQC<SZ>>()));
    big_right = make_shared<SCIFockBigSite<SZ>>(n_orbs, n_external, true,
                                                fcidump, orbsym, 2, 2, 2, true);
    big_right = make_shared<SimplifiedBigSite<SZ>>(
        big_right, make_shared<NoTransposeRule<SZ>>(make_shared<RuleQC<SZ>>()));
    shared_ptr<HamiltonianQCBigSite<SZ>> hamil =
        make_shared<HamiltonianQCBigSite<SZ>>(vacuum, n_orbs, orbsym, fcidump,
                                              big_left, big_right);

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
    linear->minres_conv_thrds = vector<double>(20, 1E-10);
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
