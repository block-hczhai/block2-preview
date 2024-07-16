
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

class TestRealTEH10STO6G : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1LL << 22;
    size_t dsize = 1LL << 30;
    typedef double FP;

    template <typename S, typename FL>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, int dot, TETypes te_type);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->minimal_disk_usage = true;
        frame_<FP>()->use_main_stack = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 4, 4,
            1);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

bool TestRealTEH10STO6G::_mpi = MPITest::okay();

template <typename S, typename FL>
void TestRealTEH10STO6G::test_dmrg(
    S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    int dot, TETypes te_type) {

    double igf_std = -0.2286598562666365;
    double energy_std = -5.424385375684663;
    double energy_dyn = 0.590980380450;

    // TDVP results
    vector<complex<double>> te_refs = {
        {0.497396368871, -0.029426410575}, {0.494212938342, -0.058663437101},
        {0.488941571448, -0.087524400468}, {0.481633079931, -0.115827961232},
        {0.472357195502, -0.143400616680}, {0.461201219425, -0.170078998109},
        {0.448268361065, -0.195711913852}, {0.433675816788, -0.220162093193},
        {0.417552647151, -0.243307597267}, {0.400037514407, -0.265042874961},
        {0.381276343871, -0.285279454006}, {0.361419971751, -0.303946269378},
        {0.340621838809, -0.320989642299}, {0.319035783923, -0.336372933090},
        {0.296813984751, -0.350075899481}, {0.274105084524, -0.362093798584},
        {0.251052535170, -0.372436275343}, {0.227793177711, -0.381126082909},
        {0.204456071825, -0.388197681130}, {0.181161577863, -0.393695758351},
    };

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<MPICommunicator<S>>();
#else
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<ParallelCommunicator<S>>(1, 0, 0);
#endif
    shared_ptr<ParallelRule<S, FL>> para_rule =
        make_shared<ParallelRuleQC<S, FL>>(para_comm);
    shared_ptr<ParallelRule<S, FL>> para_rule_site =
        make_shared<ParallelRuleSiteQC<S, FL>>(para_comm);
    shared_ptr<ParallelRule<S, FL>> para_rule_ident =
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

    cout << "C/D MPO start" << endl;
    bool su2 = S(1, 1, 0).multiplicity() == 2;
    shared_ptr<OpElement<S, FL>> c_op, d_op;
    uint16_t isite = 4;
    if (su2) {
        c_op = make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex({isite}, {}),
                                             S(1, 1, hamil->orb_sym[isite]));
        d_op = make_shared<OpElement<S, FL>>(OpNames::D, SiteIndex({isite}, {}),
                                             S(-1, 1, hamil->orb_sym[isite]));
        igf_std *= -sqrt(2);
    } else {
        c_op =
            make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex({isite}, {0}),
                                          S(1, 1, hamil->orb_sym[isite]));
        d_op =
            make_shared<OpElement<S, FL>>(OpNames::D, SiteIndex({isite}, {0}),
                                          S(-1, -1, hamil->orb_sym[isite]));
    }
    shared_ptr<MPO<S, FL>> cmpo = make_shared<SiteMPO<S, FL>>(hamil, c_op);
    shared_ptr<MPO<S, FL>> dmpo = make_shared<SiteMPO<S, FL>>(hamil, d_op);
    cout << "C/D MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "C/D MPO simplification (no transpose) start" << endl;
    cmpo = make_shared<SimplifiedMPO<S, FL>>(
        cmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    dmpo = make_shared<SimplifiedMPO<S, FL>>(
        dmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "C/D MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // MPO parallelization
    cout << "C/D MPO parallelization start" << endl;
    cmpo = make_shared<ParallelMPO<S, FL>>(cmpo, para_rule_site);
    dmpo = make_shared<ParallelMPO<S, FL>>(dmpo, para_rule_site);
    cout << "C/D MPO parallelization end .. T = " << t.get_time() << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    impo = make_shared<ParallelMPO<S, FL>>(impo, para_rule_ident);
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    // LMPO construction (no transpose)
    cout << "LMPO start" << endl;
    shared_ptr<MPO<S, FL>> lmpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "LMPO end .. T = " << t.get_time() << endl;

    // LMPO simplification (no transpose)
    cout << "LMPO simplification start" << endl;
    lmpo = make_shared<SimplifiedMPO<S, FL>>(
        lmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "LMPO simplification end .. T = " << t.get_time() << endl;

    // MPO parallelization
    cout << "LMPO parallelization start" << endl;
    lmpo = make_shared<ParallelMPO<S, FL>>(lmpo, para_rule);
    cout << "LMPO parallelization end .. T = " << t.get_time() << endl;

    ubond_t ket_bond_dim = 500, bra_bond_dim = 750;
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = {ket_bond_dim};
    vector<double> noises = {1E-6, 1E-8, 1E-10, 0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target, hamil->basis);
    mps_info->set_bond_dimension(ket_bond_dim);
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
        make_shared<DMRG<S, FL, FL>>(me, ket_bdims, noises);
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->decomp_type = DecompositionTypes::SVD;
    long double energy = dmrg->solve(20, mps->center == 0, 1E-12);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    // 1-site can be unstable
    EXPECT_LT(abs(energy - energy_std), dot == 1 ? 1E-4 : 1E-7);

    // D APPLY MPS
    shared_ptr<MPSInfo<S>> dmps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target + d_op->q_label, hamil->basis);
    dmps_info->set_bond_dimension(bra_bond_dim);
    dmps_info->tag = "DBRA";

    shared_ptr<MPS<S, FL>> dmps =
        make_shared<MPS<S, FL>>(hamil->n_sites, mps->center, dot);
    dmps->initialize(dmps_info);
    dmps->random_canonicalize();

    // MPS/MPSInfo save mutable
    dmps->save_mutable();
    dmps->deallocate();
    dmps_info->save_mutable();
    dmps_info->deallocate_mutable();

    // D APPLY ME
    shared_ptr<MovingEnvironment<S, FL, FL>> dme =
        make_shared<MovingEnvironment<S, FL, FL>>(dmpo, dmps, mps, "CPS-D");
    dme->init_environments();

    // LEFT ME
    shared_ptr<MovingEnvironment<S, FL, FL>> llme =
        make_shared<MovingEnvironment<S, FL, FL>>(lmpo, dmps, dmps, "LLHS");
    llme->init_environments();

    // Compression
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(llme, dme, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::ReducedPerturbative;
    cps->decomp_type = DecompositionTypes::SVD;
    cps->eq_type = EquationTypes::PerturbativeCompression;
    double norm = cps->solve(20, mps->center == 0, 1E-12);

    // complex MPS
    shared_ptr<MultiMPS<S, FL>> cpx_ref =
        MultiMPS<S, FL>::make_complex(dmps, "CPX-R");
    shared_ptr<MultiMPS<S, FL>> cpx_mps =
        MultiMPS<S, FL>::make_complex(dmps, "CPX-D");

    double dt = 0.1;
    int n_steps = 20;
    shared_ptr<MovingEnvironment<S, FL, FL>> xme =
        make_shared<MovingEnvironment<S, FL, FL>>(lmpo, cpx_mps, cpx_mps,
                                                  "XTD");
    shared_ptr<MovingEnvironment<S, FL, FL>> mme =
        make_shared<MovingEnvironment<S, FL, FL>>(impo, cpx_ref, cpx_mps, "II");
    lmpo->const_e -= energy;
    xme->init_environments();
    shared_ptr<TimeEvolution<S, FL, FL>> te =
        make_shared<TimeEvolution<S, FL, FL>>(xme, bra_bdims, te_type);
    te->iprint = 2;
    te->n_sub_sweeps = te->mode == TETypes::TangentSpace ? 1 : 2;
    te->normalize_mps = false;
    shared_ptr<Expect<S, FL, FL, complex<FL>>> ex =
        make_shared<Expect<S, FL, FL, complex<FL>>>(mme, bra_bond_dim,
                                                    bra_bond_dim);
    vector<complex<FL>> overlaps;
    for (int i = 0; i < n_steps; i++) {
        if (te->mode == TETypes::TangentSpace)
            te->solve(2, complex<FL>(0, dt / 2), cpx_mps->center == 0);
        else
            te->solve(1, complex<FL>(0, dt), cpx_mps->center == 0);
        mme->init_environments();
        EXPECT_LT(abs(te->energies.back() - energy_dyn), 1E-7);
        complex<FL> overlap = ex->solve(false);
        overlaps.push_back(overlap);
        cout << setprecision(12);
        cout << i * dt << " " << overlap << endl;
    }

    for (size_t i = 0; i < overlaps.size(); i++) {
        cout << "== " << name << " =="
             << " DT = " << setw(10) << fixed << setprecision(4) << (i * dt)
             << " EX = " << fixed << setw(22) << setprecision(12) << overlaps[i]
             << " ERROR = " << scientific << setprecision(3) << setw(10)
             << (overlaps[i] - te_refs[i]) << endl;
        EXPECT_LT(abs(overlaps[i] - te_refs[i]), 1E-6);
    }

    dmps_info->deallocate();
    mps_info->deallocate();
    dmpo->deallocate();
    mpo->deallocate();
}

TEST_F(TestRealTEH10STO6G, TestSU2) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP.LOWDIN";
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

    test_dmrg<SU2, double>(target, hamil, "SU2/2-site/TDVP", 2,
                           TETypes::TangentSpace);
    test_dmrg<SU2, double>(target, hamil, " SU2/2-site/RK4", 2, TETypes::RK4);
    test_dmrg<SU2, double>(target, hamil, "SU2/1-site/TDVP", 1,
                           TETypes::TangentSpace);
    test_dmrg<SU2, double>(target, hamil, " SU2/1-site/RK4", 1, TETypes::RK4);

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestRealTEH10STO6G, TestSZ) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP.LOWDIN";
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

    test_dmrg<SZ, double>(target, hamil, " SZ/2-site/TDVP", 2,
                          TETypes::TangentSpace);
    test_dmrg<SZ, double>(target, hamil, "  SZ/2-site/RK4", 2, TETypes::RK4);
    test_dmrg<SZ, double>(target, hamil, " SZ/1-site/TDVP", 1,
                          TETypes::TangentSpace);
    test_dmrg<SZ, double>(target, hamil, "  SZ/1-site/RK4", 1, TETypes::RK4);

    hamil->deallocate();
    fcidump->deallocate();
}
