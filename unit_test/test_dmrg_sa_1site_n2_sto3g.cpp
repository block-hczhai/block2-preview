
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestOneSiteDMRGN2STO3GSA : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FL FLL;

    template <typename S>
    void test_dmrg(const vector<S> &targets, const vector<FLL> &energies,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, ubond_t bond_dim, uint16_t nroots);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            8);
        threading_()->seq_type = SeqTypes::None;
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
void TestOneSiteDMRGN2STO3GSA<FL>::test_dmrg(
    const vector<S> &targets, const vector<FLL> &energies,
    const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    ubond_t bond_dim, uint16_t nroots) {

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
    vector<FP> noises = {1E-5, 1E-7, 1E-8, 0.0};

    t.get_time();

    shared_ptr<MultiMPSInfo<S>> mps_info = make_shared<MultiMPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, targets, hamil->basis);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    // 1-site is not very stable
    Random::rand_seed(1234);

    shared_ptr<MultiMPS<S, FL>> mps =
        make_shared<MultiMPS<S, FL>>(hamil->n_sites, 0, 1, nroots);
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
    dmrg->davidson_soft_max_iter = 2000;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected;
    dmrg->trunc_type = dmrg->trunc_type | TruncationTypes::RealDensityMatrix;
    FLL energy = dmrg->solve(10, mps->center == 0, 1E-8);

    // deallocate persistent stack memory
    mps_info->deallocate();

    for (size_t i = 0; i < dmrg->energies.back().size(); i++) {
        cout << "== " << name << " (SA) =="
             << " E[" << setw(2) << i << "] = " << fixed << setw(22)
             << setprecision(12) << dmrg->energies.back()[i]
             << " error = " << scientific << setprecision(3) << setw(10)
             << (dmrg->energies.back()[i] - energies[i]) << endl;

        if (i < dmrg->energies.back().size() / 2)
            EXPECT_LT(abs(dmrg->energies.back()[i] - energies[i]), 1E-6);
    }

    mpo->deallocate();
}

#ifdef _USE_COMPLEX
typedef ::testing::Types<complex<double>, double> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestOneSiteDMRGN2STO3GSA, TestFL);

TYPED_TEST(TestOneSiteDMRGN2STO3GSA, TestSU2) {
    using FL = TypeParam;
    using FLL = typename GMatrix<FL>::FL;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);

    vector<SU2> targets;
    int ne = fcidump->n_elec() / 2;
    for (int i = 0; i < 8; i++)
        for (int na = ne - 1; na <= ne + 1; na++)
            for (int nb = ne - 1; nb <= ne + 1; nb++)
                if (na - nb >= 0)
                    targets.push_back(SU2(na + nb, na - nb, i));

    vector<FLL> energies = {
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
    shared_ptr<HamiltonianQC<SU2, FL>> hamil =
        make_shared<HamiltonianQC<SU2, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2", 200, 10);

    hamil->deallocate();
    fcidump->deallocate();
}

TYPED_TEST(TestOneSiteDMRGN2STO3GSA, TestSZ) {
    using FL = TypeParam;
    using FLL = typename GMatrix<FL>::FL;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);

    vector<SZ> targets;
    int ne = fcidump->n_elec() / 2;
    for (int i = 0; i < 8; i++)
        for (int na = ne - 1; na <= ne + 1; na++)
            for (int nb = ne - 1; nb <= ne + 1; nb++)
                targets.push_back(SZ(na + nb, na - nb, i));

    vector<FLL> energies = {
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
    shared_ptr<HamiltonianQC<SZ, FL>> hamil =
        make_shared<HamiltonianQC<SZ, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SZ>(
        targets, energies, hamil, "SZ",
        (ubond_t)min(400U, (uint32_t)numeric_limits<ubond_t>::max()), 16);

    hamil->deallocate();
    fcidump->deallocate();
}
