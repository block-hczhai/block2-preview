
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

// suppress googletest output for non-root mpi procs
struct MPITest {
    shared_ptr<testing::TestEventListener> tel;
    testing::TestEventListener *def_tel;
    MPITest() {
        if (block2::MPI::rank() != 0) {
            testing::TestEventListeners &tels =
                testing::UnitTest::GetInstance()->listeners();
            def_tel = tels.Release(tels.default_result_printer());
            tel = make_shared<testing::EmptyTestEventListener>();
            tels.Append(tel.get());
        }
    }
    ~MPITest() {
        if (block2::MPI::rank() != 0) {
            testing::TestEventListeners &tels =
                testing::UnitTest::GetInstance()->listeners();
            assert(tel.get() == tels.Release(tel.get()));
            tel = nullptr;
            tels.Append(def_tel);
        }
    }
    static bool okay() {
        static MPITest _mpi_test;
        return _mpi_test.tel != nullptr;
    }
};

class TestDMRG : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1LL << 30;
    size_t dsize = 1LL << 34;
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 20, 20,
            1);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

bool TestDMRG::_mpi = MPITest::okay();

TEST_F(TestDMRG, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    // string occ_filename = "data/CR2.SVP.OCC";
    string occ_filename = "../my_test/cuprate/new2/CPR.CCSD.OCC";
    occs = read_occ(occ_filename);
    // string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    string filename = "../my_test/cuprate/new2/FCIDUMP";
    // string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    // string filename = "data/H8.STO6G.R1.8.FCIDUMP"; // E = -4.3450794024
    // (-12.3741579456) string filename = "data/H4.STO6G.R1.8.FCIDUMP"; // E =
    // -2.1903842183
    fcidump->read(filename);

    vector<uint8_t> ioccs;
    for (auto x : occs)
        ioccs.push_back(uint8_t(x));

    // cout << "HF energy = " << fixed << setprecision(12)
    //      << fcidump->det_energy(ioccs, 0, fcidump->n_sites()) + fcidump->e
    //      << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ>> hamil = make_shared<HamiltonianQC<SZ>>(vacuum, norb, orbsym, fcidump);

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<SZ>> para_comm =
        make_shared<MPICommunicator<SZ>>();
#else
    shared_ptr<ParallelCommunicator<SZ>> para_comm =
        make_shared<ParallelCommunicator<SZ>>(1, 0, 0);
#endif
    shared_ptr<ParallelRule<SZ>> para_rule =
        make_shared<ParallelRuleQC<SZ>>(para_comm);

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    // shared_ptr<MPO<SZ>> mpo =
    //     make_shared<MPOQC<SZ>>(hamil, QCTypes::Conventional);
    // cout << "MPO end .. T = " << t.get_time() << endl;

    // // MPO simplification
    // cout << "MPO simplification start" << endl;
    // mpo = make_shared<SimplifiedMPO<SZ>>(
    //     mpo, make_shared<RuleQC<SZ>>(), true, true,
    //     OpNamesSet({OpNames::R, OpNames::RD}));
    // cout << "MPO simplification end .. T = " << t.get_time() << endl;

    shared_ptr<MPO<SZ>> mpo = make_shared<MPO<SZ>>(0);
    mpo->load_data("../my_test/mpor.bin");
    mpo->tf = make_shared<TensorFunctions<SZ>>(hamil->opf);

    // mpo->save_data("../my_test/mpor.bin");

    ubond_t bond_dim = 200;

    // MPSInfo
    // shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
    //     norb, vacuum, target, hamil->basis);

    // CCSD init
    shared_ptr<MPSInfo<SZ>> mps_info =
        make_shared<MPSInfo<SZ>>(norb, vacuum, target, hamil->basis);
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == norb);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs, 1);
        // mps_info->set_bond_dimension_using_hf(bond_dim, occs, 0);
    }

    cout << "left dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;

    // MPS
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    shared_ptr<ParallelMPS<SZ>> mps =
        make_shared<ParallelMPS<SZ>>(norb, 0, 2, para_rule);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // mps->conn_centers = vector<int>{15, 21, 27};

    // ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(true);
    me->cached_contraction = true;
    me->delayed_contraction = OpNamesSet::normal_ops();
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_<FP>() << endl;
    frame_<FP>()->activate(0);

    // DMRG
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-5, 1E-5, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7,
                             1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7};
    vector<double> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6,
                              1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                              5E-7, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    // noises = vector<double>{1E-5};
    // vector<ubond_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    // dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    // dmrg->cutoff = 0;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    // dmrg->me->fuse_center = 1;
    dmrg->solve(10, true, 1E-12);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
