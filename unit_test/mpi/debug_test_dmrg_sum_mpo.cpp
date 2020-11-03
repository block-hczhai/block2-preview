
#include "block2.hpp"
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
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

bool TestDMRG::_mpi = MPITest::okay();

TEST_F(TestDMRG, Test) {

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(4);
    mkl_set_dynamic(0);
#endif

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<SZ>> para_comm =
        make_shared<MPICommunicator<SZ>>();
#else
    shared_ptr<ParallelCommunicator<SZ>> para_comm =
        make_shared<ParallelCommunicator<SZ>>(1, 0, 0);
#endif
    shared_ptr<ParallelRuleSumMPO<SZ>> para_rule =
        make_shared<ParallelRuleSumMPO<SZ>>(para_comm);

    shared_ptr<FCIDUMP> fcidump = make_shared<ParallelFCIDUMP<SZ>>(para_rule);
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    string occ_filename = "data/CR2.SVP.OCC";
    // string occ_filename = "data/CR2.SVP.HF"; // E(HF) = -2085.53318786766
    occs = read_occ(occ_filename);
    string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    // string filename = "data/H8.STO6G.R1.8.FCIDUMP"; // E = -4.3450794024
    // (-12.3741579456)
    // string filename = "data/H4.STO6G.R1.8.FCIDUMP"; // E = -2.1903842183
    fcidump->read(filename);

    vector<uint8_t> ioccs;
    for (auto x : occs)
        ioccs.push_back(uint8_t(x));

    // cout << "HF energy = " << fixed << setprecision(12)
    //      << fcidump->det_energy(ioccs, 0, fcidump->n_sites()) + fcidump->e
    //      << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    Timer t;
    t.get_time();

    vector<uint16_t> ts;
    para_rule->n_sites = hamil.n_sites;
    for (int i = 0; i < hamil.n_sites; i++)
        if (para_rule->index_available(i))
            ts.push_back(i);

    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SZ>> mpo = make_shared<SumMPOQC<SZ>>(hamil, ts);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SZ>>(
        mpo, make_shared<SumMPORule<SZ>>(make_shared<RuleQC<SZ>>(), para_rule),
        true, true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // MPO parallelization
    cout << "MPO parallelization start" << endl;
    mpo = make_shared<ParallelMPO<SZ>>(mpo, para_rule);
    cout << "MPO parallelization end .. T = " << t.get_time() << endl;

    for (int i = 0; i < para_comm->size; i++) {
        para_comm->barrier();
        if (i == para_comm->rank) {
            cerr << "RANK = " << i << " MPO = ";
            for (int i = 0; i < norb; i++)
                cerr << mpo->left_operator_names[i]->data.size() << " ";
            cerr << endl;
        }
        para_comm->barrier();
    }

    // if (para_comm->rank == 0) {
    //     cerr << mpo->get_blocking_formulas() << endl;
    //     cerr.flush();
    // }
    // para_comm->barrier();
    // abort();

    ubond_t bond_dim = 250;

    // MPSInfo
    // shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
    //     norb, vacuum, target, hamil.basis);

    // CCSD init
    shared_ptr<MPSInfo<SZ>> mps_info =
        make_shared<MPSInfo<SZ>>(norb, vacuum, target, hamil.basis);
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == norb);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs, 1);
        // mps_info->set_bond_dimension_using_hf(bond_dim, occs, 0);
    }

    // Local init
    // shared_ptr<DynamicMPSInfo<SZ>> mps_info =
    //     make_shared<DynamicMPSInfo<SZ>>(norb, vacuum, target, hamil.basis,
    //                                      hamil.orb_sym, ioccs);
    // mps_info->n_local = 4;
    // mps_info->set_bond_dimension(bond_dim);

    // Determinant init
    // shared_ptr<DeterminantMPSInfo<SZ>> mps_info =
    //     make_shared<DeterminantMPSInfo<SZ>>(norb, vacuum, target,
    //     hamil.basis,
    //                                      hamil.orb_sym, ioccs, fcidump);
    // mps_info->set_bond_dimension(bond_dim);

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
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(norb, 0, 2);
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
    hamil.opf->seq->mode = SeqTypes::Simple;
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_() << endl;
    frame_()->activate(0);

    // DMRG
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7,
                             1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 0.0};
    // noises = vector<double>{1E-5};
    // vector<ubond_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    dmrg->iprint = 2;
    // dmrg->cutoff = 0;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg->decomp_type = DecompositionTypes::SVD;
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->fuse_center = hamil.n_sites / 2;
    dmrg->solve(10, true, 1E-12);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
