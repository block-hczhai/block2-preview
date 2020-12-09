
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestLinearN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1L << 20;
    size_t dsize = 1L << 24;

    template <typename S>
    void test_dmrg(S target, const HamiltonianQC<S> &hamil, const string &name,
                   int dot);
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8, 8);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

template <typename S>
void TestLinearN2STO3G::test_dmrg(S target, const HamiltonianQC<S> &hamil,
                                  const string &name, int dot) {

    double energy_std = -107.654122447525;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S>>(mpo, make_shared<RuleQC<S>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> ntr_mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "MPO simplification (no transpose) start" << endl;
    ntr_mpo = make_shared<SimplifiedMPO<S>>(
        ntr_mpo, make_shared<NoTransposeRule<S>>(make_shared<RuleQC<S>>()),
        true);
    cout << "MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S>> impo = make_shared<IdentityMPO<S>>(hamil);
    impo = make_shared<SimplifiedMPO<S>>(impo, make_shared<Rule<S>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200, bra_bond_dim = 100;
    vector<ubond_t> bdims = {bond_dim};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        hamil.n_sites, hamil.vacuum, target, hamil.basis);
    mps_info->set_bond_dimension(bond_dim);
    mps_info->tag = "KET";

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S>> mps = make_shared<MPS<S>>(hamil.n_sites, 0, dot);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<S>> me =
        make_shared<MovingEnvironment<S>>(mpo, mps, mps, "DMRG");
    me->init_environments(false);

    // DMRG
    shared_ptr<DMRG<S>> dmrg = make_shared<DMRG<S>>(me, bdims, noises);
    dmrg->noise_type = NoiseTypes::Perturbative;
    double energy = dmrg->solve(10, mps->center == 0, 1E-8);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    EXPECT_LT(abs(energy - energy_std), 1E-7);

    shared_ptr<MPSInfo<S>> imps_info = make_shared<MPSInfo<S>>(
        hamil.n_sites, hamil.vacuum, target, hamil.basis);
    imps_info->set_bond_dimension(bra_bond_dim);
    imps_info->tag = "BRA";

    shared_ptr<MPS<S>> imps =
        make_shared<MPS<S>>(hamil.n_sites, mps->center, dot);
    imps->initialize(imps_info);
    imps->random_canonicalize();

    // MPS/MPSInfo save mutable
    imps->save_mutable();
    imps->deallocate();
    imps_info->save_mutable();
    imps_info->deallocate_mutable();

    // Negative identity ME
    shared_ptr<MovingEnvironment<S>> ime =
        make_shared<MovingEnvironment<S>>(-impo, imps, mps, "COMPRESS");
    ime->init_environments();

    // Left ME
    shared_ptr<MovingEnvironment<S>> lme =
        make_shared<MovingEnvironment<S>>(mpo, imps, imps, "LINEAR");
    double ce = mpo->const_e;
    mpo->const_e = 0;
    lme->init_environments();

    // Linear
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = bdims;
    noises = {1E-4, 1E-6, 1E-8, 0};
    shared_ptr<Linear<S>> cps =
        make_shared<Linear<S>>(lme, ime, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::ReducedPerturbative;
    cps->decomp_type = DecompositionTypes::SVD;
    double norm = cps->solve(10, mps->center == 0, 1E-10);

    EXPECT_LT(abs(norm - 1.0 / (energy_std - ce)), 1E-7);

    // Energy ME
    shared_ptr<MovingEnvironment<S>> eme =
        make_shared<MovingEnvironment<S>>(ntr_mpo, imps, mps, "EXPECT");
    ntr_mpo->const_e = 0;
    eme->init_environments(false);

    shared_ptr<Expect<S>> ex2 = make_shared<Expect<S>>(eme, bond_dim, bond_dim);
    energy = ex2->solve(false);

    cout << "== " << name << " (CPS) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " STD = " << fixed << setw(22) << setprecision(12) << energy_std
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    EXPECT_LT(abs(energy + 1.0), 1E-7);

    imps_info->deallocate();
    mps_info->deallocate();
    impo->deallocate();
    ntr_mpo->deallocate();
    mpo->deallocate();
}

TEST_F(TestLinearN2STO3G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    HamiltonianQC<SU2> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2>(target, hamil, "SU2/2-site", 2);
    test_dmrg<SU2>(target, hamil, "SU2/1-site", 1);

    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestLinearN2STO3G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));

    double energy_std = -107.654122447525;

    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ>(target, hamil, "SZ/2-site", 2);
    test_dmrg<SZ>(target, hamil, "SZ/1-site", 1);

    hamil.deallocate();
    fcidump->deallocate();
}
