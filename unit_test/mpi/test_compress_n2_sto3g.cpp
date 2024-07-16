
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

class TestLinearN2STO3G : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1LL << 20;
    size_t dsize = 1LL << 24;
    typedef double FP;

    template <typename S, typename FL>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, int dot);
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

bool TestLinearN2STO3G::_mpi = MPITest::okay();

template <typename S, typename FL>
void TestLinearN2STO3G::test_dmrg(S target,
                                  const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                                  const string &name, int dot) {

    double energy_std = -107.654122447525;

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

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // MPO parallelization
    cout << "MPO parallelization start" << endl;
    mpo = make_shared<ParallelMPO<S, FL>>(mpo, para_rule);
    cout << "MPO parallelization end .. T = " << t.get_time() << endl;

    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> ntr_mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "MPO simplification (no transpose) start" << endl;
    ntr_mpo = make_shared<SimplifiedMPO<S, FL>>(
        ntr_mpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // MPO parallelization (no transpose)
    cout << "MPO parallelization (no transpose) start" << endl;
    ntr_mpo = make_shared<ParallelMPO<S, FL>>(ntr_mpo, para_rule);
    cout << "MPO parallelization (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    impo = make_shared<ParallelMPO<S, FL>>(impo, ident_rule);
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200, bra_bond_dim = 100;
    vector<ubond_t> bdims = {bond_dim};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target, hamil->basis);
    mps_info->set_bond_dimension(bond_dim);
    mps_info->tag = "KET";

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S, FL>> mps =
        make_shared<MPS<S, FL>>(hamil->n_sites, 0, dot);
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
    me->init_environments(false);

    // DMRG
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    long double energy = dmrg->solve(10, mps->center == 0, 1E-8);

    para_comm->reduce_sum(&para_comm->tcomm, 1, para_comm->root);
    para_comm->tcomm /= para_comm->size;
    double tt = t.get_time();

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << tt << " Tcomm = " << fixed << setw(10)
         << setprecision(3) << para_comm->tcomm << " (" << setw(3) << fixed
         << setprecision(0) << (para_comm->tcomm * 100 / tt) << "%)" << endl;

    para_comm->tcomm = 0.0;

    // 1-site can be unstable
    EXPECT_LT(abs(energy - energy_std), dot == 1 ? 1E-4 : 1E-7);

    shared_ptr<MPSInfo<S>> imps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target, hamil->basis);
    imps_info->set_bond_dimension(bra_bond_dim);
    imps_info->tag = "BRA";

    shared_ptr<MPS<S, FL>> imps =
        make_shared<MPS<S, FL>>(hamil->n_sites, mps->center, dot);
    imps->initialize(imps_info);
    imps->random_canonicalize();

    // MPS/MPSInfo save mutable
    imps->save_mutable();
    imps->deallocate();
    imps_info->save_mutable();
    imps_info->deallocate_mutable();

    // Negative identity ME
    shared_ptr<MovingEnvironment<S, FL, FL>> ime =
        make_shared<MovingEnvironment<S, FL, FL>>(-impo, imps, mps, "COMPRESS");
    ime->init_environments();

    // Left ME
    shared_ptr<MovingEnvironment<S, FL, FL>> lme =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, imps, imps, "LINEAR");
    double ce = mpo->const_e;
    mpo->const_e = 0;
    lme->init_environments();

    // Linear
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = bdims;
    noises = {1E-4, 1E-6, 1E-8, 0};
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(lme, ime, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::ReducedPerturbative;
    cps->decomp_type = DecompositionTypes::DensityMatrix;
    double norm = cps->solve(10, mps->center == 0, 1E-10);

    EXPECT_LT(abs(norm - 1.0 / (energy_std - ce)), dot == 1 ? 1E-4 : 1E-7);

    // Energy ME
    shared_ptr<MovingEnvironment<S, FL, FL>> eme =
        make_shared<MovingEnvironment<S, FL, FL>>(ntr_mpo, imps, mps, "EXPECT");
    ntr_mpo->const_e = 0;
    eme->init_environments(false);

    shared_ptr<Expect<S, FL, FL>> ex2 =
        make_shared<Expect<S, FL, FL>>(eme, bond_dim, bond_dim);
    energy = ex2->solve(false);

    para_comm->reduce_sum(&para_comm->tcomm, 1, para_comm->root);
    para_comm->tcomm /= para_comm->size;
    tt = t.get_time();

    cout << "== " << name << " (CPS) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " STD = " << fixed << setw(22) << setprecision(12) << energy_std
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << tt << " Tcomm = " << fixed << setw(10)
         << setprecision(3) << para_comm->tcomm << " (" << setw(3) << fixed
         << setprecision(0) << (para_comm->tcomm * 100 / tt) << "%)" << endl;

    para_comm->tcomm = 0.0;

    EXPECT_LT(abs(energy + 1.0), dot == 1 ? 1E-4 : 1E-7);

    imps_info->deallocate();
    mps_info->deallocate();
    impo->deallocate();
    ntr_mpo->deallocate();
    mpo->deallocate();
}

TEST_F(TestLinearN2STO3G, TestSU2) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, double>> hamil =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2, double>(target, hamil, "SU2/2-site", 2);
    test_dmrg<SU2, double>(target, hamil, "SU2/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestLinearN2STO3G, TestSZ) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));

    double energy_std = -107.654122447525;

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, double>> hamil =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ, double>(target, hamil, "SZ/2-site", 2);
    test_dmrg<SZ, double>(target, hamil, "SZ/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}
