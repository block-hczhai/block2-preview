
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestResponseN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1LL << 24;
    size_t dsize = 1LL << 28;
    typedef double FP;

    template <typename S, typename FL>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, int dot);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            8);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

template <typename S, typename FL>
void TestResponseN2STO3G::test_dmrg(
    S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    int dot) {

    double pdm_std = -0.000012699067;
    double energy_std = -107.654122447525;

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

    cout << "C/D MPO start" << endl;
    bool su2 = S(1, 1, 0).multiplicity() == 2;
    shared_ptr<OpElement<S, FL>> c_op, d_op;
    if (su2) {
        c_op = make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex({0}, {}),
                                             S(1, 1, hamil->orb_sym[0]));
        d_op = make_shared<OpElement<S, FL>>(OpNames::D, SiteIndex({1}, {}),
                                             S(-1, 1, hamil->orb_sym[1]));
        pdm_std *= -sqrt(2);
    } else {
        c_op = make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex({0}, {0}),
                                             S(1, 1, hamil->orb_sym[0]));
        d_op = make_shared<OpElement<S, FL>>(OpNames::D, SiteIndex({1}, {0}),
                                             S(-1, -1, hamil->orb_sym[1]));
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

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t ket_bond_dim = 500, bra_bond_dim = 500;
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = {ket_bond_dim};
    vector<FL> noises = {1E-4, 1E-6, 1E-8, 0};

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
    dmrg->noise_type = NoiseTypes::Perturbative;
    long double energy = dmrg->solve(10, mps->center == 0, 1E-8);

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

    // Linear
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(dme, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::DensityMatrix;
    cps->decomp_type = DecompositionTypes::DensityMatrix;
    double norm = cps->solve(10, mps->center == 0, 1E-10);

    // C APPLY MPS
    shared_ptr<MPSInfo<S>> cmps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target, hamil->basis);
    cmps_info->set_bond_dimension(bra_bond_dim);
    cmps_info->tag = "CBRA";

    shared_ptr<MPS<S, FL>> cmps =
        make_shared<MPS<S, FL>>(hamil->n_sites, mps->center, dot);
    cmps->initialize(cmps_info);
    cmps->random_canonicalize();

    // MPS/MPSInfo save mutable
    cmps->save_mutable();
    cmps->deallocate();
    cmps_info->save_mutable();
    cmps_info->deallocate_mutable();

    // C APPLY ME
    shared_ptr<MovingEnvironment<S, FL, FL>> cme =
        make_shared<MovingEnvironment<S, FL, FL>>(cmpo, cmps, dmps, "CPS-C");
    cme->init_environments();

    // Linear
    cps = make_shared<Linear<S, FL, FL>>(cme, bra_bdims, bra_bdims, noises);
    norm = cps->solve(10, cmps->center == 0, 1E-10);

    if (mps->center != cmps->center) {
        cps->noises = {0};
        cps->solve(1, cmps->center == 0, 0);
    }

    // Identity ME
    shared_ptr<MovingEnvironment<S, FL, FL>> ime =
        make_shared<MovingEnvironment<S, FL, FL>>(impo, cmps, mps, "EXP");
    ime->init_environments();

    shared_ptr<Expect<S, FL, FL>> ex =
        make_shared<Expect<S, FL, FL>>(ime, bra_bond_dim, bra_bond_dim);
    ex->solve(true, cmps->center == 0);
    double pdm = ex->expectations[0][0].second;

    cout << "== " << name << " (CPS) ==" << setw(20) << target
         << " PDM = " << fixed << setw(22) << setprecision(12) << pdm
         << " STD = " << fixed << setw(22) << setprecision(12) << pdm_std
         << " error = " << scientific << setprecision(3) << setw(10)
         << (pdm - pdm_std) << " T = " << fixed << setw(10) << setprecision(3)
         << t.get_time() << endl;

    EXPECT_LT(abs(pdm - pdm_std), dot == 1 ? 1E-4 : 1E-7);

    dmps_info->deallocate();
    mps_info->deallocate();
    dmpo->deallocate();
    mpo->deallocate();
}

TEST_F(TestResponseN2STO3G, TestSU2) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

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

TEST_F(TestResponseN2STO3G, TestSZ) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

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
