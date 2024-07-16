
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

class TestTSpaceAncillaH8STO6G : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1LL << 22;
    size_t dsize = 1LL << 30;
    typedef double FP;

    template <typename S, typename FL>
    void test_imag_te(int n_sites, int n_physical_sites, S target,
                      const vector<double> &energies_fted,
                      const vector<double> &energies_m500,
                      const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                      const string &name);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 4, 4,
            4);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

bool TestTSpaceAncillaH8STO6G::_mpi = MPITest::okay();

template <typename S, typename FL>
void TestTSpaceAncillaH8STO6G::test_imag_te(
    int n_sites, int n_physical_sites, S target,
    const vector<double> &energies_fted, const vector<double> &energies_m500,
    const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name) {

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<MPICommunicator<S>>();
#else
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<ParallelCommunicator<S>>(1, 0, 0);
#endif
    shared_ptr<ParallelRule<S, FL>> para_rule =
        make_shared<ParallelRuleQC<S, FL>>(para_comm);
    shared_ptr<ParallelRule<S, FL>> ident_rule =
        make_shared<ParallelRuleIdentity<S, FL>>(para_comm);

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // Ancilla MPO construction
    cout << "Ancilla MPO start" << endl;
    mpo = make_shared<AncillaMPO<S, FL>>(mpo);
    cout << "Ancilla MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // MPO parallelization
    cout << "MPO parallelization start" << endl;
    mpo = make_shared<ParallelMPO<S, FL>>(mpo, para_rule);
    cout << "MPO parallelization end .. T = " << t.get_time() << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<AncillaMPO<S, FL>>(impo);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    impo = make_shared<ParallelMPO<S, FL>>(impo, ident_rule);
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim =
        (ubond_t)min(500U, (uint32_t)numeric_limits<ubond_t>::max());
    double beta = 0.05;
    vector<ubond_t> bdims = {bond_dim};
    vector<double> te_energies, noises = {0.0};

    // Ancilla MPSInfo (thermal)
    Random::rand_seed(0);

    shared_ptr<AncillaMPSInfo<S>> mps_info_thermal =
        make_shared<AncillaMPSInfo<S>>(n_physical_sites, hamil->vacuum, target,
                                       hamil->basis);
    mps_info_thermal->set_thermal_limit();
    mps_info_thermal->tag = "KET";

    // Ancilla MPS (thermal)
    shared_ptr<MPS<S, FL>> mps_thermal =
        make_shared<MPS<S, FL>>(n_sites, n_sites - 2, 2);
    mps_thermal->initialize(mps_info_thermal);
    mps_thermal->fill_thermal_limit();

    // MPS/MPSInfo save mutable (thermal)
    mps_thermal->save_mutable();
    mps_thermal->deallocate();
    mps_info_thermal->save_mutable();
    mps_info_thermal->deallocate_mutable();

    // Ancilla MPSInfo (fitting)
    shared_ptr<AncillaMPSInfo<S>> imps_info = make_shared<AncillaMPSInfo<S>>(
        n_physical_sites, hamil->vacuum, target, hamil->basis);
    imps_info->set_bond_dimension(bond_dim);
    imps_info->tag = "BRA";

    // Ancilla MPS (fitting)
    shared_ptr<MPS<S, FL>> imps =
        make_shared<MPS<S, FL>>(n_sites, n_sites - 2, 2);
    imps->initialize(imps_info);
    imps->random_canonicalize();

    // MPS/MPSInfo save mutable (fitting)
    imps->save_mutable();
    imps->deallocate();
    imps_info->save_mutable();
    imps_info->deallocate_mutable();

    // Identity ME
    shared_ptr<MovingEnvironment<S, FL, FL>> ime =
        make_shared<MovingEnvironment<S, FL, FL>>(impo, imps, mps_thermal,
                                                  "COMPRESS");
    ime->init_environments(false);

    // Linear
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(ime, bdims, bdims, noises);
    double norm = cps->solve(10, imps->center == 0);

    EXPECT_LT(abs(norm - 1.0), 1E-7);

    // TE ME
    shared_ptr<MovingEnvironment<S, FL, FL>> me =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, imps, imps, "TE");
    me->init_environments(false);

    shared_ptr<Expect<S, FL, FL>> ex =
        make_shared<Expect<S, FL, FL>>(me, bond_dim, bond_dim);
    te_energies.push_back(ex->solve(false));

    // Imaginary TE
    shared_ptr<TimeEvolution<S, FL, FL>> te =
        make_shared<TimeEvolution<S, FL, FL>>(me, bdims, TETypes::RK4);
    te->iprint = 2;
    te->n_sub_sweeps = 6;
    te->solve(1, beta / 2.0, imps->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    te->n_sub_sweeps = 1;
    te->mode = TETypes::TangentSpace;
    te->solve(2, beta / 2.0, imps->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    // two-site to one-site transition
    me->dot = 1;

    te->solve(7, beta / 2.0, imps->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    para_comm->reduce_sum(&para_comm->tcomm, 1, para_comm->root);
    para_comm->tcomm /= para_comm->size;
    double tt = t.get_time();

    for (size_t i = 0; i < te_energies.size(); i++) {
        cout << "== " << name << " =="
             << " BETA = " << setw(10) << fixed << setprecision(4)
             << ((FL)i * beta) << " E = " << fixed << setw(22)
             << setprecision(12) << te_energies[i]
             << " error-fted = " << scientific << setprecision(3) << setw(10)
             << (te_energies[i] - energies_fted[i])
             << " error-m500 = " << scientific << setprecision(3) << setw(10)
             << (te_energies[i] - energies_m500[i]) << " T = " << fixed
             << setw(10) << setprecision(3) << tt << " Tcomm = " << fixed
             << setw(10) << setprecision(3) << para_comm->tcomm << " ("
             << setw(3) << fixed << setprecision(0)
             << (para_comm->tcomm * 100 / tt) << "%)" << endl;

        EXPECT_LT(abs(te_energies[i] - energies_m500[i]), 1E-3);
    }

    para_comm->tcomm = 0.0;

    imps_info->deallocate();
    mps_info_thermal->deallocate();
    impo->deallocate();
    mpo->deallocate();
}

TEST_F(TestTSpaceAncillaH8STO6G, TestSU2) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H8.STO6G.R1.8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    vector<double> energies_fted = {
        0.3124038410492045,  -0.0273905176813768, -0.3265074932156511,
        -0.5914620908396366, -0.8276498731818384, -1.0395171725041257,
        -1.2307228748517529, -1.4042806712721763, -1.5626789845611742,
        -1.7079796842651509, -1.8418982445788070};

    vector<double> energies_m500 = {
        0.312403841049,  -0.027389713306, -0.326500930805, -0.591439984794,
        -0.827598404678, -1.039419259243, -1.230558968502, -1.404029934736,
        -1.562319009406, -1.707487414764, -1.841250686976};

    SU2 vacuum(0);
    SU2 target(fcidump->n_sites() * 2, fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int n_physical_sites = fcidump->n_sites();
    int n_sites = n_physical_sites * 2;

    shared_ptr<HamiltonianQC<SU2, double>> hamil =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, n_physical_sites,
                                                orbsym, fcidump);
    hamil->mu = -1.0;
    hamil->fcidump->const_e = 0.0;

    test_imag_te<SU2, double>(n_sites, n_physical_sites, target, energies_fted,
                              energies_m500, hamil, "SU2");

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestTSpaceAncillaH8STO6G, TestSZ) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H8.STO6G.R1.8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    vector<double> energies_fted = {
        0.3124038410492045,  -0.0273905176813768, -0.3265074932156511,
        -0.5914620908396366, -0.8276498731818384, -1.0395171725041257,
        -1.2307228748517529, -1.4042806712721763, -1.5626789845611742,
        -1.7079796842651509, -1.8418982445788070};

    vector<double> energies_m500 = {
        0.312403841049,  -0.027388048069, -0.326490457632, -0.591401772825,
        -0.827502872933, -1.039228830737, -1.230231051484, -1.403519072586,
        -1.561579406450, -1.706474487633, -1.839921660072};

    SZ vacuum(0);
    SZ target(fcidump->n_sites() * 2, fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));
    int n_physical_sites = fcidump->n_sites();
    int n_sites = n_physical_sites * 2;

    shared_ptr<HamiltonianQC<SZ, double>> hamil =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, n_physical_sites, orbsym,
                                               fcidump);
    hamil->mu = -1.0;
    hamil->fcidump->const_e = 0.0;

    test_imag_te<SZ, double>(n_sites, n_physical_sites, target, energies_fted,
                             energies_m500, hamil, "SZ");

    hamil->deallocate();
    fcidump->deallocate();
}
