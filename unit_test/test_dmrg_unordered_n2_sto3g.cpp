
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL> class TestDMRGUnorderedN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1LL << 20;
    size_t dsize = 1LL << 24;
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FL FLL;

    template <typename S>
    void test_dmrg(const vector<vector<S>> &targets,
                   const vector<vector<FLL>> &energies,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        cout << "MKL INTEGER SIZE = " << sizeof(MKL_INT) << endl;
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            8);
        threading_()->seq_type = SeqTypes::None;
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
void TestDMRGUnorderedN2STO3G<FL>::test_dmrg(
    const vector<vector<S>> &targets, const vector<vector<FLL>> &energies,
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
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<ubond_t> bdims = {bond_dim};
    vector<FP> noises = {1E-6, 1E-7, 1E-8, 1E-9, 0.0};
    vector<FP> no_noises = {0.0};

    t.get_time();

    Random::rand_seed(0);

    for (int i = 0; i < (int)targets.size(); i++)
        for (int j = 0, k = 0; j < (int)targets[i].size(); j++) {

            S target = targets[i][j];

            shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
                hamil->n_sites, hamil->vacuum, target, hamil->basis);
            mps_info->set_bond_dimension(bond_dim);

            // MPS

            shared_ptr<MPS<S, FL>> mps =
                make_shared<MPS<S, FL>>(hamil->n_sites, 0, 2);
            mps->initialize(mps_info);
            mps->random_canonicalize();

            // MPS/MPSInfo save mutable
            mps->save_mutable();
            mps->deallocate();
            mps_info->save_mutable();
            mps_info->deallocate_mutable();

            shared_ptr<ParallelMPS<S, FL>> pmps =
                make_shared<ParallelMPS<S, FL>>(mps);
            pmps->conn_centers =
                vector<int>{hamil->n_sites / 3, 2 * hamil->n_sites / 3};

            // ME
            shared_ptr<MovingEnvironment<S, FL, FL>> me =
                make_shared<MovingEnvironment<S, FL, FL>>(mpo, pmps, pmps,
                                                          "DMRG");
            me->init_environments(false);
            me->cached_contraction = true;

            // DMRG
            shared_ptr<DMRG<S, FL, FL>> dmrg =
                make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
            dmrg->iprint = 0;
            dmrg->decomp_type = dt;
            dmrg->noise_type = nt;
            dmrg->davidson_soft_max_iter = 4000;
            FLL energy = dmrg->solve(10, mps->center == 0, 1E-8);

            cout << "== PAR " << name << " ==" << setw(20) << target
                 << " E = " << fixed << setw(22) << setprecision(12) << energy
                 << " error = " << scientific << setprecision(3) << setw(10)
                 << (energy - energies[i][j]) << " T = " << fixed << setw(10)
                 << setprecision(3) << t.get_time() << endl;

            if (abs(energy - energies[i][j]) >= 1E-7 && k < 3) {
                k++, j--;
                cout << "!!! RETRY ... " << endl;
                continue;
            }

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);

            k = 0;

            me->finalize_environments();

            me->bra = me->ket = make_shared<MPS<S, FL>>(*pmps);

            dmrg = make_shared<DMRG<S, FL, FL>>(me, bdims, no_noises);
            dmrg->iprint = 0;
            dmrg->decomp_type = dt;
            dmrg->noise_type = nt;
            energy = dmrg->solve(1, pmps->center == 0, 1E-8);

            cout << "== SER " << name << " ==" << setw(20) << target
                 << " E = " << fixed << setw(22) << setprecision(12) << energy
                 << " error = " << scientific << setprecision(3) << setw(10)
                 << (energy - energies[i][j]) << " T = " << fixed << setw(10)
                 << setprecision(3) << t.get_time() << endl;

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);

            // deallocate persistent stack memory
            mps_info->deallocate();
        }

    mpo->deallocate();
}

#ifdef _USE_COMPLEX
typedef ::testing::Types<complex<double>, double> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestDMRGUnorderedN2STO3G, TestFL);

TYPED_TEST(TestDMRGUnorderedN2STO3G, TestSU2) {
    using FL = TypeParam;
    using FLL = typename GMatrix<FL>::FL;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

    SU2 vacuum(0);

    vector<vector<SU2>> targets(8);
    for (int i = 0; i < 8; i++) {
        targets[i].resize(3);
        for (int j = 0; j < 3; j++)
            targets[i][j] = SU2(fcidump->n_elec(), j * 2, i);
    }

    vector<vector<FLL>> energies(8);
    energies[0] = {-107.654122447525, -106.939132859668, -107.031449471627};
    energies[1] = {-106.959626154680, -106.999600016661, -106.633790589321};
    energies[2] = {-107.306744734756, -107.356943001688, -106.931515926732};
    energies[3] = {-107.306744734756, -107.356943001688, -106.931515926731};
    energies[4] = {-107.223155479270, -107.279409754727, -107.012640794842};
    energies[5] = {-107.208347039017, -107.343458537272, -106.227634428741};
    energies[6] = {-107.116397543375, -107.208021870379, -107.070427868786};
    energies[7] = {-107.116397543375, -107.208021870379, -107.070427868786};

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, FL>> hamil =
        make_shared<HamiltonianQC<SU2, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2",
                                  DecompositionTypes::DensityMatrix,
                                  NoiseTypes::DensityMatrix);

    targets.resize(2);
    energies.resize(2);

    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2 SVD",
                                  DecompositionTypes::SVD,
                                  NoiseTypes::Wavefunction);
    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2 PURE SVD",
                                  DecompositionTypes::PureSVD,
                                  NoiseTypes::Wavefunction);
    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2 PERT",
                                  DecompositionTypes::DensityMatrix,
                                  NoiseTypes::Perturbative);
    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2 SVD PERT",
                                  DecompositionTypes::SVD,
                                  NoiseTypes::Perturbative);
    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2 RED PERT",
                                  DecompositionTypes::DensityMatrix,
                                  NoiseTypes::ReducedPerturbative);
    this->template test_dmrg<SU2>(targets, energies, hamil, "SU2 SVD RED PERT",
                                  DecompositionTypes::SVD,
                                  NoiseTypes::ReducedPerturbative);

    hamil->deallocate();
    fcidump->deallocate();
}

TYPED_TEST(TestDMRGUnorderedN2STO3G, TestSZ) {
    using FL = TypeParam;
    using FLL = typename GMatrix<FL>::FL;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              [pg](uint8_t x) { return (uint8_t)PointGroup::swap_pg(pg)(x); });

    SZ vacuum(0);

    vector<vector<SZ>> targets(8);
    for (int i = 0; i < 8; i++) {
        targets[i].resize(5);
        for (int j = 0; j < 5; j++)
            targets[i][j] = SZ(fcidump->n_elec(), (j - 2) * 2, i);
    }

    vector<vector<FLL>> energies(8);
    energies[0] = {-107.031449471627, -107.031449471627, -107.654122447525,
                   -107.031449471627, -107.031449471627};
    energies[1] = {-106.633790589321, -106.999600016661, -106.999600016661,
                   -106.999600016661, -106.633790589321};
    energies[2] = {-106.931515926732, -107.356943001688, -107.356943001688,
                   -107.356943001688, -106.931515926732};
    energies[3] = {-106.931515926731, -107.356943001688, -107.356943001688,
                   -107.356943001688, -106.931515926731};
    energies[4] = {-107.012640794842, -107.279409754727, -107.279409754727,
                   -107.279409754727, -107.012640794842};
    energies[5] = {-106.227634428741, -107.343458537272, -107.343458537272,
                   -107.343458537272, -106.227634428741};
    energies[6] = {-107.070427868786, -107.208021870379, -107.208021870379,
                   -107.208021870379, -107.070427868786};
    energies[7] = {-107.070427868786, -107.208021870379, -107.208021870379,
                   -107.208021870379, -107.070427868786};

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, FL>> hamil =
        make_shared<HamiltonianQC<SZ, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ",
                                 DecompositionTypes::DensityMatrix,
                                 NoiseTypes::DensityMatrix);

    targets.resize(2);
    energies.resize(2);

    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ SVD",
                                 DecompositionTypes::SVD,
                                 NoiseTypes::Wavefunction);
    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ PURE SVD",
                                 DecompositionTypes::PureSVD,
                                 NoiseTypes::Wavefunction);
    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ PERT",
                                 DecompositionTypes::DensityMatrix,
                                 NoiseTypes::Perturbative);
    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ SVD PERT",
                                 DecompositionTypes::SVD,
                                 NoiseTypes::Perturbative);
    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ RED PERT",
                                 DecompositionTypes::DensityMatrix,
                                 NoiseTypes::ReducedPerturbative);
    this->template test_dmrg<SZ>(targets, energies, hamil, "SZ SVD RED PERT",
                                 DecompositionTypes::SVD,
                                 NoiseTypes::ReducedPerturbative);

    hamil->deallocate();
    fcidump->deallocate();
}
