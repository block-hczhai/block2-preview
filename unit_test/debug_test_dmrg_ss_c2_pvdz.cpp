
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestDMRGStateSpecific : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;
    typedef typename GMatrix<FL>::FP FP;

    template <typename S>
    void test_dmrg(const vector<S> &targets, const vector<FL> &energies,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        frame_<FP>()->minimal_memory_usage = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 16,
            16, 1);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *frame_<FP>() << endl;
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
void TestDMRGStateSpecific<FL>::test_dmrg(
    const vector<S> &targets, const vector<FL> &energies,
    const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    DecompositionTypes dt, NoiseTypes nt) {
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

    cout << "MPO add identity start" << endl;
    mpo = make_shared<IdentityAddedMPO<S, FL>>(mpo);
    cout << "MPO add identity end .. T = " << t.get_time() << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 250;
    int nroots = (int)energies.size() / 2;
    vector<ubond_t> bdims = {250, 250, 250, 250, 250};
    vector<ubond_t> ss_bdims = {500};
    vector<FP> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7,
                         1E-7, 1E-7, 1E-7, 1E-7, 0.0};

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
    dmrg->iprint = 1;
    dmrg->decomp_type = dt;
    dmrg->noise_type = nt;
    dmrg->solve(30, mps->center == 0, 1E-7);

    // deallocate persistent stack memory
    double tt = t.get_time();

    for (int ir = 0; ir < nroots; ir++) {
        cout << "== " << name << " =="
             << " E = " << fixed << setw(22) << setprecision(15)
             << dmrg->energies.back()[ir] << " error = " << scientific
             << setprecision(3) << setw(10)
             << (dmrg->energies.back()[ir] - energies[ir]) << " T = " << fixed
             << setw(10) << setprecision(3) << tt << endl;

        EXPECT_LT(abs(dmrg->energies.back()[ir] - energies[ir]), 5E-5);
    }

    vector<shared_ptr<MPS<S, FL>>> ext_mpss;
    for (int ir = 0; ir < nroots; ir++)
        ext_mpss.push_back(mps->extract(ir, "KET" + Parsing::to_string(ir))
                               ->make_single("SKET" + Parsing::to_string(ir)));

    for (int ir = 0; ir < nroots; ir++) {
        t.get_time();
        if (ext_mpss[ir]->center != ext_mpss[0]->center) {
            cout << "change canonical form ..." << ext_mpss[ir]->center << " "
                 << ext_mpss[0]->center << endl;
            shared_ptr<MovingEnvironment<S, FL, FL>> ime =
                make_shared<MovingEnvironment<S, FL, FL>>(impo, ext_mpss[ir],
                                                          ext_mpss[ir], "IEX");
            ime->init_environments(false);
            ime->delayed_contraction = OpNamesSet::normal_ops();
            shared_ptr<Expect<S, FL, FL>> expect =
                make_shared<Expect<S, FL, FL>>(
                    ime, ext_mpss[ir]->info->bond_dim + 100,
                    ext_mpss[ir]->info->bond_dim + 100);
            expect->iprint = 1;
            expect->solve(true, ext_mpss[ir]->center == 0);
            ext_mpss[ir]->save_data();
            cout << ext_mpss[ir]->canonical_form << endl;
            cout << ext_mpss[0]->canonical_form << endl;
            assert(ext_mpss[ir]->center == ext_mpss[0]->center);
        }
        shared_ptr<MovingEnvironment<S, FL, FL>> ss_me =
            make_shared<MovingEnvironment<S, FL, FL>>(mpo, ext_mpss[ir],
                                                      ext_mpss[ir], "DMRG");
        ss_me->init_environments(false);
        ss_me->delayed_contraction = OpNamesSet::normal_ops();
        ss_me->cached_contraction = true;
        shared_ptr<DMRG<S, FL, FL>> ss_dmrg =
            make_shared<DMRG<S, FL, FL>>(ss_me, ss_bdims, noises);
        ss_dmrg->ext_mpss = vector<shared_ptr<MPS<S, FL>>>(
            ext_mpss.begin(), ext_mpss.begin() + ir);
        for (auto &mps : ss_dmrg->ext_mpss) {
            shared_ptr<MovingEnvironment<S, FL, FL>> ex_me =
                make_shared<MovingEnvironment<S, FL, FL>>(
                    impo, ext_mpss[ir], mps, "EX" + mps->info->tag);
            ex_me->init_environments(false);
            ex_me->delayed_contraction = OpNamesSet::normal_ops();
            ss_dmrg->ext_mes.push_back(ex_me);
        }
        ss_dmrg->state_specific = false;
        ss_dmrg->projection_weights = vector<FP>(ir, 5);
        ss_dmrg->iprint = 1;
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
                  5E-5);
    }

    mps_info->deallocate();
    mpo->deallocate();
}

#ifdef _USE_COMPLEX
typedef ::testing::Types<complex<double>, double> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestDMRGStateSpecific, TestFL);

TYPED_TEST(TestDMRGStateSpecific, TestSU2) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/C2.CAS.PVDZ.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);

    vector<SU2> targets = {SU2(fcidump->n_elec(), fcidump->twos(),
                               PointGroup::swap_pg(pg)(fcidump->isym()))};
    // vector<double> energies = {-75.728288492594714, -75.638913608438372,
    //                            -75.728475014334350, -75.639047312018349};
    vector<FL> energies = {-75.728133317624227, -75.638833326817164,
                           -75.629473601924815, -75.728475014334350,
                           -75.639047312018349, -75.629689955315186};

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, FL>> hamil =
        make_shared<HamiltonianQC<SU2, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2",
                                  DecompositionTypes::DensityMatrix,
                                  NoiseTypes::ReducedPerturbativeCollected);

    hamil->deallocate();
    fcidump->deallocate();
}

TYPED_TEST(TestDMRGStateSpecific, TestSZ) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/C2.CAS.PVDZ.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);

    vector<SZ> targets = {SZ(fcidump->n_elec(), fcidump->twos(),
                             PointGroup::swap_pg(pg)(fcidump->isym()))};
    // vector<double> energies = {-75.727181710145302, -75.637956291287594,
    //                            -75.727871140355631, -75.638645816798174};
    vector<FL> energies = {-75.726794668605351, -75.637773501104306,
                           -75.628412417981167, -75.727871140355631,
                           -75.638645816798174, -75.629177339134202};

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, FL>> hamil =
        make_shared<HamiltonianQC<SZ, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ",
                                 DecompositionTypes::DensityMatrix,
                                 NoiseTypes::ReducedPerturbativeCollected);

    hamil->deallocate();
    fcidump->deallocate();
}
