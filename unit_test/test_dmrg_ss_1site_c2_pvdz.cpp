
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestOneSiteDMRGStateSpecific : public ::testing::Test {
  protected:
    size_t isize = 1L << 26;
    size_t dsize = 1L << 30;

    template <typename S>
    void test_dmrg(const vector<S> &targets, const vector<double> &energies,
                   const HamiltonianQC<S> &hamil, const string &name,
                   DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
        frame_()->use_main_stack = false;
        frame_()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 16,
            16, 1);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

template <typename S>
void TestOneSiteDMRGStateSpecific::test_dmrg(const vector<S> &targets,
                                      const vector<double> &energies,
                                      const HamiltonianQC<S> &hamil,
                                      const string &name, DecompositionTypes dt,
                                      NoiseTypes nt) {
    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo =
        make_shared<SimplifiedMPO<S>>(mpo, make_shared<RuleQC<S>>(), true, true,
                                      OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S>> impo = make_shared<IdentityMPO<S>>(hamil);
    impo = make_shared<SimplifiedMPO<S>>(impo, make_shared<Rule<S>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 250;
    int nroots = (int)energies.size() / 2;
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500};
    vector<ubond_t> ss_bdims = {500};
    vector<double> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5,
                             1E-5, 1E-5, 1E-5, 1E-5, 0.0};

    t.get_time();

    shared_ptr<MultiMPSInfo<S>> mps_info = make_shared<MultiMPSInfo<S>>(
        hamil.n_sites, hamil.vacuum, targets, hamil.basis);
    // mps_info->load_mutable();
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);

    shared_ptr<MultiMPS<S>> mps =
        make_shared<MultiMPS<S>>(hamil.n_sites, 0, 2, nroots);
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
    me->delayed_contraction = OpNamesSet::normal_ops();
    me->cached_contraction = true;

    // DMRG
    shared_ptr<DMRG<S>> dmrg = make_shared<DMRG<S>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->decomp_type = dt;
    dmrg->noise_type = nt;
    dmrg->solve(6, mps->center == 0, 1E-7);

    me->dot = 1;

    dmrg->solve(25, mps->center == 0, 1E-7);

    mps->dot = 1;
    mps->save_data();

    // deallocate persistent stack memory
    double tt = t.get_time();

    for (int ir = 0; ir < nroots; ir++) {
        cout << "== " << name << " =="
             << " E = " << fixed << setw(22) << setprecision(15)
             << dmrg->energies.back()[ir] << " error = " << scientific
             << setprecision(3) << setw(10)
             << (dmrg->energies.back()[ir] - energies[ir]) << " T = " << fixed
             << setw(10) << setprecision(3) << tt << endl;

        EXPECT_LT(abs(dmrg->energies.back()[ir] - energies[ir]), 5E-4);
    }

    vector<shared_ptr<MPS<S>>> ext_mpss;
    for (int ir = 0; ir < nroots; ir++)
        ext_mpss.push_back(mps->extract(ir, "KET" + Parsing::to_string(ir))
                               ->make_single("SKET" + Parsing::to_string(ir)));

    for (int ir = 0; ir < nroots; ir++) {
        t.get_time();
        if (ext_mpss[ir]->center != ext_mpss[0]->center) {
            cout << "change canonical form ..." << ext_mpss[ir]->center << " "
                 << ext_mpss[0]->center << endl;
            shared_ptr<MovingEnvironment<S>> ime =
                make_shared<MovingEnvironment<S>>(impo, ext_mpss[ir],
                                                  ext_mpss[ir], "IEX");
            ime->init_environments(false);
            ime->delayed_contraction = OpNamesSet::normal_ops();
            shared_ptr<Expect<S>> expect =
                make_shared<Expect<S>>(ime, ext_mpss[ir]->info->bond_dim + 100,
                                       ext_mpss[ir]->info->bond_dim + 100);
            expect->iprint = 2;
            expect->solve(true, ext_mpss[ir]->center == 0);
            ext_mpss[ir]->save_data();
            cout << ext_mpss[ir]->canonical_form << endl;
            cout << ext_mpss[0]->canonical_form << endl;
            assert(ext_mpss[ir]->center == ext_mpss[0]->center);
        }
        shared_ptr<MovingEnvironment<S>> ss_me =
            make_shared<MovingEnvironment<S>>(mpo, ext_mpss[ir], ext_mpss[ir],
                                              "DMRG");
        ss_me->init_environments(false);
        ss_me->delayed_contraction = OpNamesSet::normal_ops();
        ss_me->cached_contraction = true;
        shared_ptr<DMRG<S>> ss_dmrg =
            make_shared<DMRG<S>>(ss_me, ss_bdims, noises);
        ss_dmrg->ext_mpss =
            vector<shared_ptr<MPS<S>>>(ext_mpss.begin(), ext_mpss.begin() + ir);
        for (auto &mps : ss_dmrg->ext_mpss) {
            shared_ptr<MovingEnvironment<S>> ex_me =
                make_shared<MovingEnvironment<S>>(impo, ext_mpss[ir], mps,
                                                  "EX" + mps->info->tag);
            ex_me->init_environments(false);
            ex_me->delayed_contraction = OpNamesSet::normal_ops();
            ss_dmrg->ext_mes.push_back(ex_me);
        }
        ss_dmrg->state_specific = true;
        ss_dmrg->iprint = 2;
        ss_dmrg->decomp_type = dt;
        ss_dmrg->noise_type = nt;
        ss_dmrg->solve(29, ext_mpss[ir]->center == 0, 1E-7);

        cout << "== SS " << name << " =="
             << " E = " << fixed << setw(22) << setprecision(15)
             << ss_dmrg->energies.back()[0] << " error = " << scientific
             << setprecision(3) << setw(10)
             << (ss_dmrg->energies.back()[0] - energies[ir + nroots])
             << " T = " << fixed << setw(10) << setprecision(3) << t.get_time()
             << endl;

        EXPECT_LT(abs(ss_dmrg->energies.back()[0] - energies[ir + nroots]),
                  5E-4);
    }

    mps_info->deallocate();
    mpo->deallocate();
}

TEST_F(TestOneSiteDMRGStateSpecific, TestSU2) {

    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/C2.CAS.PVDZ.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);

    vector<SU2> targets = {SU2(fcidump->n_elec(), fcidump->twos(),
                               PointGroup::swap_pg(pg)(fcidump->isym()))};
    // vector<double> energies = {-75.728288492594714, -75.638913608438372,
    //                            -75.728475014334350, -75.639047312018349};
    vector<double> energies = {-75.728133317624227, -75.638833326817164,
                               -75.629473601924815, -75.728475014334350,
                               -75.639047312018349, -75.629689955315186};

    int norb = fcidump->n_sites();
    HamiltonianQC<SU2> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2>(targets, energies, hamil, "SU2",
                   DecompositionTypes::DensityMatrix,
                   NoiseTypes::ReducedPerturbativeCollected);

    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestOneSiteDMRGStateSpecific, TestSZ) {

    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/C2.CAS.PVDZ.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);

    vector<SZ> targets = {SZ(fcidump->n_elec(), fcidump->twos(),
                             PointGroup::swap_pg(pg)(fcidump->isym()))};
    // vector<double> energies = {-75.727181710145302, -75.637956291287594,
    //                            -75.727871140355631, -75.638645816798174};
    vector<double> energies = {-75.726794668605351, -75.637773501104306,
                               -75.628412417981167, -75.727871140355631,
                               -75.638645816798174, -75.629177339134202};

    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ>(targets, energies, hamil, "SZ",
                  DecompositionTypes::DensityMatrix,
                  NoiseTypes::ReducedPerturbativeCollected);

    hamil.deallocate();
    fcidump->deallocate();
}
