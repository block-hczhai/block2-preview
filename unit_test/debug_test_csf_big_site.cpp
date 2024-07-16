
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
        threading_()->seq_type = SeqTypes::None;
        cout << *frame_<FP>() << endl;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

TEST_F(TestDMRG, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<CompressedFCIDUMP>(1E-13);
    // shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    string filename =
        "data/H10.STO6G.R1.8.FCIDUMP.C1"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "INT end .. T = " << t.get_time() << endl;

    Random::rand_seed(1234);

    cout << "ORB SYM = ";
    for (int i = 0; i < fcidump->n_sites(); i++)
        cout << setw(2) << (int)fcidump->orb_sym<uint8_t>()[i];
    cout << endl;

    // cout << "HF energy = " << fixed << setprecision(12)
    //      << fcidump->det_energy(ioccs, 0, fcidump->n_sites()) + fcidump->e
    //      << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;

    shared_ptr<BigSite<SU2>> big_left = make_shared<CSFBigSite<SU2>>(
        5, 10, false, fcidump,
        vector<uint8_t>(orbsym.begin(), orbsym.begin() + 5));
    shared_ptr<BigSite<SU2>> big_right = make_shared<CSFBigSite<SU2>>(
        5, 10, true, fcidump,
        vector<uint8_t>(orbsym.begin(), orbsym.begin() + 5));

    shared_ptr<HamiltonianQC<SU2>> hamil =
        make_shared<HamiltonianQCBigSite<SU2>>(vacuum, norb, orbsym, fcidump,
                                               big_left, big_right);
    shared_ptr<MPO<SU2>> mpo = make_shared<MPOQC<SU2>>(hamil, QCTypes::NC);
    int nsite = hamil->n_sites;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SU2>>(mpo, make_shared<RuleQC<SU2>>(), true,
                                          false);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    // mpo->reduce_data();
    // mpo->save_data("mpo.bin");

    // mpo = make_shared<MPO<SU2>>(0);
    // mpo->load_data("mpo.bin", true);

    ubond_t bond_dim = 200;

    // MPSInfo
    // shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
    //     norb, vacuum, target, hamil->basis);

    // CCSD init
    shared_ptr<MPSInfo<SU2>> mps_info =
        make_shared<MPSInfo<SU2>>(nsite, vacuum, target, hamil->basis);
    // mps_info->set_bond_dimension_full_fci();
    mps_info->set_bond_dimension(bond_dim);

    // Local init
    // shared_ptr<DynamicMPSInfo<SU2>> mps_info =
    //     make_shared<DynamicMPSInfo<SU2>>(norb, vacuum, target, hamil->basis,
    //                                      hamil->orb_sym, ioccs);
    // mps_info->n_local = 4;
    // mps_info->set_bond_dimension(bond_dim);

    // Determinant init
    // shared_ptr<DeterminantMPSInfo<SU2>> mps_info =
    //     make_shared<DeterminantMPSInfo<SU2>>(norb, vacuum, target,
    //     hamil->basis,
    //                                      hamil->orb_sym, ioccs, fcidump);
    // mps_info->set_bond_dimension(bond_dim);

    cout << "left dims = ";
    for (int i = 0; i <= nsite; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= nsite; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;
    // abort();

    // MPS
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    shared_ptr<MPS<SU2>> mps = make_shared<MPS<SU2>>(nsite, 0, 2);
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
    shared_ptr<MovingEnvironment<SU2>> me =
        make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_<FP>() << endl;
    frame_<FP>()->activate(0);

    // DMRG
    // vector<ubond_t> bdims = {50};
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500, 500,
                             500, 500, 500, 500, 500, 500, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5,
                             1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6};
    vector<double> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6,
                              1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                              5E-7, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    // noises = vector<double>{1E-5};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->cached_contraction = true;
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    // dmrg->cutoff = 1E-20;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollectedLowMem;
    // dmrg->davidson_type = DavidsonTypes::GreaterThan;
    // dmrg->davidson_shift = -2086.4;
    dmrg->solve(20, true);

    shared_ptr<MPSInfo<SU2>> bra_info =
        make_shared<MPSInfo<SU2>>(norb, vacuum, target, hamil->basis);
    bra_info->set_bond_dimension(bond_dim);
    bra_info->tag = "BRA";

    shared_ptr<MPS<SU2>> bra = make_shared<MPS<SU2>>(norb, 0, 2);
    bra->initialize(bra_info);
    bra->random_canonicalize();
    bra->save_mutable();
    bra_info->save_mutable();

    shared_ptr<MPO<SU2>> impo = make_shared<IdentityMPO<SU2>>(hamil);
    impo = make_shared<SimplifiedMPO<SU2>>(impo, make_shared<Rule<SU2>>());

    shared_ptr<MovingEnvironment<SU2>> ime =
        make_shared<MovingEnvironment<SU2>>(impo, bra, mps, "RHS");
    ime->init_environments();
    vector<ubond_t> bra_dims = {250};
    vector<ubond_t> ket_dims = {250};
    shared_ptr<Linear<SU2>> linear =
        make_shared<Linear<SU2>>(ime, bra_dims, ket_dims);
    linear->cutoff = 1E-14;
    double igf = linear->solve(20, bra->center == 0, 1E-12);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
