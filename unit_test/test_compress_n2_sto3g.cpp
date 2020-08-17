
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestCompressN2STO3G : public ::testing::Test {
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

TEST_F(TestCompressN2STO3G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(8);
    mkl_set_dynamic(0);
#endif

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    double energy_std = -107.654122447525;

    int norb = fcidump->n_sites();
    HamiltonianQC<SU2> hamil(vacuum, norb, orbsym, fcidump);

    hamil.opf->seq->mode = SeqTypes::Simple;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SU2>> mpo =
        make_shared<MPOQC<SU2>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo =
        make_shared<SimplifiedMPO<SU2>>(mpo, make_shared<RuleQC<SU2>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    cout << "MPO start" << endl;
    shared_ptr<MPO<SU2>> ntr_mpo =
        make_shared<MPOQC<SU2>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "MPO simplification (no transpose) start" << endl;
    ntr_mpo = make_shared<SimplifiedMPO<SU2>>(
        ntr_mpo, make_shared<NoTransposeRule<SU2>>(make_shared<RuleQC<SU2>>()),
        true);
    cout << "MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<SU2>> impo = make_shared<IdentityMPO<SU2>>(hamil);
    impo = make_shared<SimplifiedMPO<SU2>>(impo, make_shared<Rule<SU2>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    uint16_t bond_dim = 200, bra_bond_dim = 100;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
        norb, vacuum, target, hamil.basis);
    mps_info->set_bond_dimension(bond_dim);
    mps_info->tag = "KET";

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<SU2>> mps = make_shared<MPS<SU2>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<SU2>> me =
        make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
    me->init_environments(false);

    // DMRG
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    double energy = dmrg->solve(10, true, 1E-8);

    cout << "== SU2 (DMRG) ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << energy << " error = " << scientific
         << setprecision(3) << setw(10) << (energy - energy_std)
         << " T = " << fixed << setw(10) << setprecision(3) << t.get_time()
         << endl;

    EXPECT_LT(abs(energy - energy_std), 1E-7);

    shared_ptr<MPSInfo<SU2>> imps_info = make_shared<MPSInfo<SU2>>(
        norb, vacuum, target, hamil.basis);
    imps_info->set_bond_dimension(bra_bond_dim);
    imps_info->tag = "BRA";

    shared_ptr<MPS<SU2>> imps = make_shared<MPS<SU2>>(norb, mps->center, 2);
    imps->initialize(imps_info);
    imps->random_canonicalize();

    // MPS/MPSInfo save mutable
    imps->save_mutable();
    imps->deallocate();
    imps_info->save_mutable();
    imps_info->deallocate_mutable();

    // Identity ME
    shared_ptr<MovingEnvironment<SU2>> ime =
        make_shared<MovingEnvironment<SU2>>(impo, imps, mps, "COMPRESS");
    ime->init_environments();

    // Compress
    vector<uint16_t> bra_bdims = {bra_bond_dim}, ket_bdims = bdims;
    noises = {0.0};
    shared_ptr<Compress<SU2>> cps =
        make_shared<Compress<SU2>>(ime, bra_bdims, ket_bdims, noises);
    double norm = cps->solve(10, mps->center == 0);

    EXPECT_LT(abs(norm - 1.0), 1E-7);

    // Energy ME
    shared_ptr<MovingEnvironment<SU2>> eme =
        make_shared<MovingEnvironment<SU2>>(ntr_mpo, imps, mps, "EXPECT");
    eme->init_environments(false);

    shared_ptr<Expect<SU2>> ex2 = make_shared<Expect<SU2>>(eme, bond_dim, bond_dim);
    energy = ex2->solve(false) + mpo->const_e;

    cout << "== SU2  (CPS) ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << energy << " error = " << scientific
         << setprecision(3) << setw(10) << (energy - energy_std)
         << " T = " << fixed << setw(10) << setprecision(3) << t.get_time()
         << endl;

    EXPECT_LT(abs(energy - energy_std), 1E-7);

    imps_info->deallocate();
    mps_info->deallocate();
    impo->deallocate();
    ntr_mpo->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}


TEST_F(TestCompressN2STO3G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(8);
    mkl_set_dynamic(0);
#endif

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    double energy_std = -107.654122447525;

    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    hamil.opf->seq->mode = SeqTypes::Simple;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SZ>> mpo =
        make_shared<MPOQC<SZ>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo =
        make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    cout << "MPO start" << endl;
    shared_ptr<MPO<SZ>> ntr_mpo =
        make_shared<MPOQC<SZ>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "MPO simplification (no transpose) start" << endl;
    ntr_mpo = make_shared<SimplifiedMPO<SZ>>(
        ntr_mpo, make_shared<NoTransposeRule<SZ>>(make_shared<RuleQC<SZ>>()),
        true);
    cout << "MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<SZ>> impo = make_shared<IdentityMPO<SZ>>(hamil);
    impo = make_shared<SimplifiedMPO<SZ>>(impo, make_shared<Rule<SZ>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    uint16_t bond_dim = 200, bra_bond_dim = 100;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
        norb, vacuum, target, hamil.basis);
    mps_info->set_bond_dimension(bond_dim);
    mps_info->tag = "KET";

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    me->init_environments(false);

    // DMRG
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    double energy = dmrg->solve(10, true, 1E-8);

    cout << "== SZ  (DMRG) ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << energy << " error = " << scientific
         << setprecision(3) << setw(10) << (energy - energy_std)
         << " T = " << fixed << setw(10) << setprecision(3) << t.get_time()
         << endl;

    EXPECT_LT(abs(energy - energy_std), 1E-7);

    shared_ptr<MPSInfo<SZ>> imps_info = make_shared<MPSInfo<SZ>>(
        norb, vacuum, target, hamil.basis);
    imps_info->set_bond_dimension(bra_bond_dim);
    imps_info->tag = "BRA";

    shared_ptr<MPS<SZ>> imps = make_shared<MPS<SZ>>(norb, mps->center, 2);
    imps->initialize(imps_info);
    imps->random_canonicalize();

    // MPS/MPSInfo save mutable
    imps->save_mutable();
    imps->deallocate();
    imps_info->save_mutable();
    imps_info->deallocate_mutable();

    // Identity ME
    shared_ptr<MovingEnvironment<SZ>> ime =
        make_shared<MovingEnvironment<SZ>>(impo, imps, mps, "COMPRESS");
    ime->init_environments();

    // Compress
    vector<uint16_t> bra_bdims = {bra_bond_dim}, ket_bdims = bdims;
    noises = {0.0};
    shared_ptr<Compress<SZ>> cps =
        make_shared<Compress<SZ>>(ime, bra_bdims, ket_bdims, noises);
    double norm = cps->solve(10, mps->center == 0);

    EXPECT_LT(abs(norm - 1.0), 1E-7);

    // Energy ME
    shared_ptr<MovingEnvironment<SZ>> eme =
        make_shared<MovingEnvironment<SZ>>(ntr_mpo, imps, mps, "EXPECT");
    eme->init_environments(false);

    shared_ptr<Expect<SZ>> ex2 = make_shared<Expect<SZ>>(eme, bond_dim, bond_dim);
    energy = ex2->solve(false) + mpo->const_e;

    cout << "== SZ   (CPS) ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << energy << " error = " << scientific
         << setprecision(3) << setw(10) << (energy - energy_std)
         << " T = " << fixed << setw(10) << setprecision(3) << t.get_time()
         << endl;

    EXPECT_LT(abs(energy - energy_std), 1E-7);

    imps_info->deallocate();
    mps_info->deallocate();
    impo->deallocate();
    ntr_mpo->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
