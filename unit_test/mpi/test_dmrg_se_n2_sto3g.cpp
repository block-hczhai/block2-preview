
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

// suppress googletest output for non-root mpi procs
struct MPITest {
    shared_ptr<testing::TestEventListener> tel;
    testing::TestEventListener *def_tel;
    MPITest() {
        if (block2::MPI::rank() != 0) {
            testing::TestEventListeners &tels =
                testing::UnitTest::GetInstance()->listeners();
            def_tel = tels.Release(tels.default_result_printer());
            tel = make_shared<testing::EmptyTestEventListener>();
            tels.Append(tel.get());
        }
    }
    ~MPITest() {
        if (block2::MPI::rank() != 0) {
            testing::TestEventListeners &tels =
                testing::UnitTest::GetInstance()->listeners();
            assert(tel.get() == tels.Release(tel.get()));
            tel = nullptr;
            tels.Append(def_tel);
        }
    }
    static bool okay() {
        static MPITest _mpi_test;
        return _mpi_test.tel != nullptr;
    }
};

class TestDMRGSingletEmbedding : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1LL << 20;
    size_t dsize = 1LL << 24;
    typedef double FP;

    template <typename S, typename FL>
    void test_dmrg(const vector<vector<S>> &targets,
                   const vector<vector<long double>> &energies,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 4, 4,
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

bool TestDMRGSingletEmbedding::_mpi = MPITest::okay();

template <typename S, typename FL>
void TestDMRGSingletEmbedding::test_dmrg(
    const vector<vector<S>> &targets, const vector<vector<long double>> &energies,
    const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    DecompositionTypes dt, NoiseTypes nt) {

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<MPICommunicator<S>>();
#else
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<ParallelCommunicator<S>>(1, 0, 0);
#endif
    shared_ptr<ParallelRule<S, FL>> para_rule =
        make_shared<ParallelRuleQC<S, FL>>(para_comm);

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

    // MPO parallelization
    cout << "MPO parallelization start" << endl;
    mpo = make_shared<ParallelMPO<S, FL>>(mpo, para_rule);
    cout << "MPO parallelization end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<ubond_t> bdims = {bond_dim};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    Random::rand_seed(0);

    for (int i = 0; i < (int)targets.size(); i++)
        for (int j = 0, k = 0; j < (int)targets[i].size(); j++) {

            S target = targets[i][j];
            S se_target = S(target.n() + target.twos(), 0, target.pg());
            S left_vacuum = S(target.twos(), target.twos(), 0);
            S right_vacuum = S(0, 0, 0);

            shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
                hamil->n_sites, hamil->vacuum, se_target, hamil->basis);
            mps_info->set_bond_dimension_fci(left_vacuum, right_vacuum);
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

            // ME
            shared_ptr<MovingEnvironment<S, FL, FL>> me =
                make_shared<MovingEnvironment<S, FL, FL>>(mpo, mps, mps,
                                                          "DMRG");
            me->init_environments(false);
            me->delayed_contraction = OpNamesSet::normal_ops();
            me->cached_contraction = true;

            // DMRG
            shared_ptr<DMRG<S, FL, FL>> dmrg =
                make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
            dmrg->iprint = 0;
            dmrg->decomp_type = dt;
            dmrg->noise_type = nt;
            dmrg->davidson_soft_max_iter = 4000;
            long double energy = dmrg->solve(10, mps->center == 0, 1E-8);

            // deallocate persistent stack memory
            mps_info->deallocate();

            para_comm->reduce_sum(&para_comm->tcomm, 1, para_comm->root);
            para_comm->tcomm /= para_comm->size;
            double tt = t.get_time();

            cout << "== " << name << " ==" << setw(20) << target
                 << " E = " << fixed << setw(22) << setprecision(12) << energy
                 << " error = " << scientific << setprecision(3) << setw(10)
                 << (energy - energies[i][j]) << " T = " << fixed << setw(10)
                 << setprecision(3) << tt << " Tcomm = " << fixed << setw(10)
                 << setprecision(3) << para_comm->tcomm << " (" << setw(3)
                 << fixed << setprecision(0) << (para_comm->tcomm * 100 / tt)
                 << "%)" << endl;

            para_comm->tcomm = 0.0;

            if (abs(energy - energies[i][j]) >= 1E-7 && k < 3) {
                k++, j--;
                cout << "!!! RETRY ... " << endl;
                continue;
            }

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);

            k = 0;
        }

    mpo->deallocate();
}

TEST_F(TestDMRGSingletEmbedding, TestSU2) {

    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);

    vector<vector<SU2>> targets(8);
    for (int i = 0; i < 8; i++) {
        targets[i].resize(2);
        for (int j = 1; j < 3; j++)
            targets[i][j - 1] = SU2(fcidump->n_elec(), j * 2, i);
    }

    vector<vector<long double>> energies(8);
    energies[0] = {-106.939132859668, -107.031449471627};
    energies[1] = {-106.999600016661, -106.633790589321};
    energies[2] = {-107.356943001688, -106.931515926732};
    energies[3] = {-107.356943001688, -106.931515926731};
    energies[4] = {-107.279409754727, -107.012640794842};
    energies[5] = {-107.343458537272, -106.227634428741};
    energies[6] = {-107.208021870379, -107.070427868786};
    energies[7] = {-107.208021870379, -107.070427868786};

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, double>> hamil =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2, double>(targets, energies, hamil, "SU2",
                           DecompositionTypes::DensityMatrix,
                           NoiseTypes::DensityMatrix);

    targets.resize(2);
    energies.resize(2);

    test_dmrg<SU2, double>(targets, energies, hamil, "SU2 PERT",
                           DecompositionTypes::DensityMatrix,
                           NoiseTypes::ReducedPerturbative);
    test_dmrg<SU2, double>(targets, energies, hamil, "SU2 PERT COL",
                           DecompositionTypes::DensityMatrix,
                           NoiseTypes::ReducedPerturbativeCollected);
    test_dmrg<SU2, double>(targets, energies, hamil, "SU2 SVD",
                           DecompositionTypes::SVD, NoiseTypes::Wavefunction);
    test_dmrg<SU2, double>(targets, energies, hamil, "SU2 PERT SVD",
                           DecompositionTypes::SVD,
                           NoiseTypes::ReducedPerturbative);

    hamil->deallocate();
    fcidump->deallocate();
}
