
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestLinearN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1LL << 24;
    size_t dsize = 1LL << 28;
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FL FLL;

    template <typename S>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, int dot);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 2, 2,
            2);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

template <typename FL>
template <typename S>
void TestLinearN2STO3G<FL>::test_dmrg(
    S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    int dot) {

    FLL energy_std = -107.654122447525;

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

    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> ntr_mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "MPO simplification (no transpose) start" << endl;
    ntr_mpo = make_shared<SimplifiedMPO<S, FL>>(
        ntr_mpo,
        make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200, bra_bond_dim = 100;
    vector<ubond_t> bdims = {bond_dim};
    vector<FP> noises = {1E-8, 1E-9, 0.0};

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
    dmrg->noise_type = NoiseTypes::Perturbative;
    FLL energy = dmrg->solve(10, mps->center == 0, 1E-8);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

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
    FLL ce = mpo->const_e;
    mpo->const_e = 0;
    lme->init_environments();

    // Linear
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = bdims;
    noises = {1E-4, 1E-6, 1E-8, 0};
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(lme, ime, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::ReducedPerturbative;
    cps->decomp_type = DecompositionTypes::SVD;
    FL norm = cps->solve(10, mps->center == 0, 1E-10);

    EXPECT_LT(abs((FLL)norm - (FLL)1.0 / (energy_std - ce)),
              dot == 1 ? 1E-4 : 1E-7);

    // Energy ME
    shared_ptr<MovingEnvironment<S, FL, FL>> eme =
        make_shared<MovingEnvironment<S, FL, FL>>(ntr_mpo, imps, mps, "EXPECT");
    ntr_mpo->const_e = 0;
    eme->init_environments(false);

    shared_ptr<Expect<S, FL, FL, FL>> ex2 =
        make_shared<Expect<S, FL, FL, FL>>(eme, bond_dim, bond_dim);
    energy = ex2->solve(false);

    cout << "== " << name << " (CPS) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " STD = " << fixed << setw(22) << setprecision(12) << energy_std
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    EXPECT_LT(abs(energy + (FLL)1.0), dot == 1 ? 1E-4 : 1E-7);

    imps_info->deallocate();
    mps_info->deallocate();
    impo->deallocate();
    ntr_mpo->deallocate();
    mpo->deallocate();
}

#ifdef _USE_COMPLEX
typedef ::testing::Types<complex<double>, double> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestLinearN2STO3G, TestFL);

TYPED_TEST(TestLinearN2STO3G, TestSU2) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, FL>> hamil =
        make_shared<HamiltonianQC<SU2, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SU2>(target, hamil, "SU2/2-site", 2);
    this->template test_dmrg<SU2>(target, hamil, "SU2/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}

TYPED_TEST(TestLinearN2STO3G, TestSZ) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, FL>> hamil =
        make_shared<HamiltonianQC<SZ, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SZ>(target, hamil, "SZ/2-site", 2);
    this->template test_dmrg<SZ>(target, hamil, "SZ/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}
