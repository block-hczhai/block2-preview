
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestNPDMN2STO3GSA : public ::testing::Test {
  protected:
    size_t isize = 1LL << 24;
#ifndef __EMSCRIPTEN__
    size_t dsize = 1LL << 32;
#else
    size_t dsize = 1LL << 28;
#endif
    typedef double FP;

    template <typename S, typename FL>
    void test_dmrg(const vector<S> &targets, const vector<long double> &energies,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, ubond_t bond_dim, uint16_t nroots);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
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

template <typename S, typename FL>
void TestNPDMN2STO3GSA::test_dmrg(const vector<S> &targets,
                                  const vector<long double> &energies,
                                  const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                                  const string &name, ubond_t bond_dim,
                                  uint16_t nroots) {

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo =
        make_shared<MPOQC<S, FL>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(
        mpo, make_shared<RuleQC<S, FL>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    vector<ubond_t> bdims = {bond_dim};
    vector<FL> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    shared_ptr<MultiMPSInfo<S>> mps_info = make_shared<MultiMPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, targets, hamil->basis);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);

    shared_ptr<MultiMPS<S, FL>> mps =
        make_shared<MultiMPS<S, FL>>(hamil->n_sites, 0, 2, nroots);
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
    me->delayed_contraction = OpNamesSet::normal_ops();
    me->cached_contraction = true;

    // DMRG
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected;
    long double energy = dmrg->solve(20, mps->center == 0, 1E-8);

    for (size_t i = 0; i < dmrg->energies.back().size(); i++) {
        cout << "== " << name << " (SA) =="
             << " E[" << setw(2) << i << "] = " << fixed << setw(22)
             << setprecision(12) << dmrg->energies.back()[i]
             << " error = " << scientific << setprecision(3) << setw(10)
             << (dmrg->energies.back()[i] - energies[i]) << endl;

        // EXPECT_LT(abs(dmrg->energies.back()[i] - energies[i]), 1E-7);
    }

    mpo->deallocate();

    // 1PDM MPO construction
    cout << "1PDM MPO start" << endl;
    shared_ptr<MPO<S, FL>> pmpo = make_shared<PDM1MPOQC<S, FL>>(hamil);
    cout << "1PDM MPO end .. T = " << t.get_time() << endl;

    // 1PDM MPO simplification
    cout << "1PDM MPO simplification start" << endl;
    pmpo = make_shared<SimplifiedMPO<S, FL>>(
        pmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;

    for (int iroot = 0; iroot < nroots; iroot++)
        for (int jroot = 0; jroot <= iroot; jroot++) {
            cout << jroot << " -> " << iroot << endl;
            shared_ptr<MultiMPS<S, FL>> imps = mps->extract(
                iroot, mps->info->tag + "-" + Parsing::to_string(iroot));
            shared_ptr<MultiMPS<S, FL>> jmps = mps->extract(
                jroot, mps->info->tag + "-" + Parsing::to_string(jroot));

            // 1PDM ME
            shared_ptr<MovingEnvironment<S, FL, FL>> pme =
                make_shared<MovingEnvironment<S, FL, FL>>(pmpo, imps, jmps,
                                                          "1PDM");
            t.get_time();
            cout << "1PDM INIT start" << endl;
            pme->init_environments(false);
            cout << "1PDM INIT end .. T = " << t.get_time() << endl;

            // 1PDM
            shared_ptr<Expect<S, FL, FL>> expect =
                make_shared<Expect<S, FL, FL>>(pme, bond_dim, bond_dim);
            expect->solve(true, dmrg->forward);

            MatrixRef dm = S(1, 1, 0).multiplicity() != 1
                               ? expect->get_1pdm_spatial()
                               : expect->get_1pdm();
            dm.deallocate();
        }

    // deallocate persistent stack memory
    mps_info->deallocate();
}

TEST_F(TestNPDMN2STO3GSA, TestSU2) {

    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

    SU2 vacuum(0);

    vector<SU2> targets;
    for (int i = 0; i < 4; i++)
        targets.push_back(SU2(fcidump->n_elec(), fcidump->twos(), i));

    vector<long double> energies = {
        -107.654122447525, // < N=14 S=0 PG=0 >
        -107.356943001688, // < N=14 S=1 PG=2|3 >
        -107.356943001688, // < N=14 S=1 PG=2|3 >
        -107.343458537273, // < N=14 S=1 PG=5 >
        -107.319813793867, // < N=15 S=1/2 PG=2|3 >
        -107.319813793866, // < N=15 S=1/2 PG=2|3 >
        -107.306744734757, // < N=14 S=0 PG=2|3 >
        -107.306744734756, // < N=14 S=0 PG=2|3 >
        -107.279409754727, // < N=14 S=1 PG=4|5 >
        -107.279409754727  // < N=14 S=1 PG=4|5 >
    };

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, double>> hamil =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2, double>(targets, energies, hamil, "SU2", 200, 4);

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestNPDMN2STO3GSA, TestSZ) {

    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

    SZ vacuum(0);

    vector<SZ> targets;
    for (int i = 0; i < 4; i++)
        targets.push_back(SZ(fcidump->n_elec(), fcidump->twos(), i));

    vector<long double> energies = {
        -107.654122447526, // < N=14 S=0 PG=0 >
        -107.356943001689, // < N=14 S=-1|0|1 PG=2|3 >
        -107.356943001688, // < N=14 S=-1|0|1 PG=2|3 >
        -107.356943001688, // < N=14 S=-1|0|1 PG=2|3 >
        -107.356943001688, // < N=14 S=-1|0|1 PG=2|3 >
        -107.356943001688, // < N=14 S=-1|0|1 PG=2|3 >
        -107.356943001688, // < N=14 S=-1|0|1 PG=2|3 >
        -107.343458537273, // < N=14 S=-1|0|1 PG=5 >
        -107.343458537273, // < N=14 S=-1|0|1 PG=5 >
        -107.343458537272, // < N=14 S=-1|0|1 PG=5 >
        -107.319813793867, // < N=15 S=-1/2|1/2 PG=2|3 >
        -107.319813793866, // < N=15 S=-1/2|1/2 PG=2|3 >
        -107.319813793866, // < N=15 S=-1/2|1/2 PG=2|3 >
        -107.319813793866, // < N=15 S=-1/2|1/2 PG=2|3 >
        -107.306744734756, // < N=14 S=0 PG=2|3 >
        -107.306744734756  // < N=14 S=0 PG=2|3 >
    };

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, double>> hamil =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ, double>(
        targets, energies, hamil, "SZ",
        (ubond_t)min(400U, (uint32_t)numeric_limits<ubond_t>::max()), 4);

    hamil->deallocate();
    fcidump->deallocate();
}
