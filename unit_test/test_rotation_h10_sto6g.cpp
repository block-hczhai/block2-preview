
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestRotationH10STO6G : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;

    template <typename S, typename FL>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil_rot,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil_c1,
                   const string &name, int dot, TETypes te_type, double tol);
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
        frame_()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            8);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

template <typename S, typename FL>
void TestRotationH10STO6G::test_dmrg(
    S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
    const shared_ptr<HamiltonianQC<S, FL>> &hamil_rot,
    const shared_ptr<HamiltonianQC<S, FL>> &hamil_c1, const string &name,
    int dot, TETypes te_type, double tol) {

    double energy_std = -5.424385375684663;

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

    ubond_t ket_bond_dim = 500, bra_bond_dim = 1000;
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = {ket_bond_dim};
    vector<FL> noises = {1E-6, 1E-8, 1E-10, 0};

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
    double energy = dmrg->solve(20, mps->center == 0, 1E-12);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    // 1-site can be unstable
    EXPECT_LT(abs(energy - energy_std), dot == 1 ? 1E-4 : 1E-7);

    // MPO construction
    cout << "MPO ROT start" << endl;
    shared_ptr<MPO<S, FL>> mpo_rot =
        make_shared<MPOQC<S, FL>>(hamil_rot, QCTypes::NC);
    cout << "MPO ROT end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO ROT simplification start" << endl;
    mpo_rot = make_shared<SimplifiedMPO<S, FL>>(
        mpo_rot,
        make_shared<AntiHermitianRuleQC<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "MPO ROT simplification end .. T = " << t.get_time() << endl;

    double dt = 0.02;
    int n_steps = (int)(1.0 / dt + 0.1);
    shared_ptr<MovingEnvironment<S, FL, FL>> rme =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo_rot, mps, mps, "ROT");
    rme->init_environments();
    shared_ptr<TimeEvolution<S, FL, FL>> te =
        make_shared<TimeEvolution<S, FL, FL>>(rme, bra_bdims, te_type);
    te->hermitian = false;
    te->iprint = 2;
    te->n_sub_sweeps = te->mode == TETypes::TangentSpace ? 1 : 2;
    te->normalize_mps = false;
    for (int i = 0; i < n_steps; i++) {
        if (te->mode == TETypes::TangentSpace)
            te->solve(2, -dt / 2, mps->center == 0);
        else
            te->solve(1, -dt, mps->center == 0);
        cout << setprecision(12);
        cout << i * dt << " " << te->energies.back() << " "
             << te->normsqs.back() << endl;
    }

    // MPO construction
    cout << "MPO MO start" << endl;
    shared_ptr<MPO<S, FL>> mpo_c1 =
        make_shared<MPOQC<S, FL>>(hamil_c1, QCTypes::Conventional);
    cout << "MPO MO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO MO simplification start" << endl;
    mpo_c1 = make_shared<SimplifiedMPO<S, FL>>(
        mpo_c1, make_shared<RuleQC<S, FL>>(), true);
    cout << "MPO MO simplification end .. T = " << t.get_time() << endl;

    // ME
    shared_ptr<MovingEnvironment<S, FL, FL>> me_c1 =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo_c1, mps, mps, "DMRG");
    me_c1->init_environments(false);

    shared_ptr<Expect<S, FL, FL>> ex =
        make_shared<Expect<S, FL, FL>>(me_c1, bra_bond_dim, bra_bond_dim);
    double ener_c1 = ex->solve(false);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << ener_c1
         << " error = " << scientific << setprecision(3) << setw(10)
         << (ener_c1 - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    EXPECT_LT(abs(ener_c1 - energy_std), tol);

    mpo_c1->deallocate();
    mpo_rot->deallocate();
    mps_info->deallocate();
    mpo->deallocate();
}

TEST_F(TestRotationH10STO6G, TestSU2) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::C1;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP.LOWDIN";
    string filename_c1 = "data/H10.STO6G.R1.8.FCIDUMP.C1";
    string filename_rot = "data/H10.STO6G.R1.8.ROTATION.LOWDIN";
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

    shared_ptr<FCIDUMP<double>> fcidump_rot = make_shared<FCIDUMP<double>>();
    fcidump_rot->read(filename_rot);
    shared_ptr<HamiltonianQC<SU2, double>> hamil_rot =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, norb, orbsym,
                                                fcidump_rot);

    shared_ptr<FCIDUMP<double>> fcidump_c1 = make_shared<FCIDUMP<double>>();
    fcidump_c1->read(filename_c1);
    shared_ptr<HamiltonianQC<SU2, double>> hamil_c1 =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, norb, orbsym,
                                                fcidump_c1);

    test_dmrg<SU2, double>(target, hamil, hamil_rot, hamil_c1, "SU2/2-site/TS",
                           2, TETypes::TangentSpace, 1E-7);
    test_dmrg<SU2, double>(target, hamil, hamil_rot, hamil_c1, "SU2/2-site/RK",
                           2, TETypes::RK4, 1E-7);
    // test_dmrg<SU2, double>(target, hamil, hamil_rot, hamil_c1, "SU2/1-site",
    // 1,
    //                TETypes::TangentSpace);

    hamil_rot->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestRotationH10STO6G, TestSZ) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::C1;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP.LOWDIN";
    string filename_c1 = "data/H10.STO6G.R1.8.FCIDUMP.C1";
    string filename_rot = "data/H10.STO6G.R1.8.ROTATION.LOWDIN";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, double>> hamil =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, norb, orbsym, fcidump);

    shared_ptr<FCIDUMP<double>> fcidump_rot = make_shared<FCIDUMP<double>>();
    fcidump_rot->read(filename_rot);
    shared_ptr<HamiltonianQC<SZ, double>> hamil_rot =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, norb, orbsym,
                                               fcidump_rot);

    shared_ptr<FCIDUMP<double>> fcidump_c1 = make_shared<FCIDUMP<double>>();
    fcidump_c1->read(filename_c1);
    shared_ptr<HamiltonianQC<SZ, double>> hamil_c1 =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, norb, orbsym,
                                               fcidump_c1);

    test_dmrg<SZ, double>(target, hamil, hamil_rot, hamil_c1, "SZ/2-site/TS", 2,
                          TETypes::TangentSpace, 1E-5);
    test_dmrg<SZ, double>(target, hamil, hamil_rot, hamil_c1, "SZ/2-site/RK", 2,
                          TETypes::RK4, 1E-5);
    // test_dmrg<SZ, double>(target, hamil, hamil_rot, hamil_c1, "SZ/1-site", 1,
    //               TETypes::TangentSpace);

    hamil_rot->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
