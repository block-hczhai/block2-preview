
#include "block2.hpp"
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

class TestSumMPON2STO3G : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 28;

    template <typename S>
    void test_dmrg(const vector<vector<S>> &targets,
                   const vector<vector<double>> &energies,
                   const HamiltonianQC<S> &hamil, const string &name,
                   DecompositionTypes dt, NoiseTypes nt);
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

bool TestSumMPON2STO3G::_mpi = MPITest::okay();

template <typename S>
void TestSumMPON2STO3G::test_dmrg(const vector<vector<S>> &targets,
                                  const vector<vector<double>> &energies,
                                  const HamiltonianQC<S> &hamil,
                                  const string &name, DecompositionTypes dt,
                                  NoiseTypes nt) {

    hamil.opf->seq->mode = SeqTypes::Simple;

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(4);
    mkl_set_dynamic(0);
#endif

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<MPICommunicator<S>>();
#else
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<ParallelCommunicator<S>>(1, 0, 0);
#endif
    shared_ptr<ParallelRuleSumMPO<S>> para_rule =
        make_shared<ParallelRuleSumMPO<S>>(para_comm);

    vector<uint16_t> ts;
    para_rule->n_sites = hamil.n_sites;
    for (int i = 0; i < hamil.n_sites; i++)
        if (para_rule->index_available(i))
            ts.push_back(i);

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo = make_shared<SumMPOQC<S>>(hamil, ts);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S>>(
        mpo, make_shared<SumMPORule<S>>(make_shared<RuleQC<S>>(), para_rule),
        true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // MPO parallelization
    cout << "MPO parallelization start" << endl;
    mpo = make_shared<ParallelMPO<S>>(mpo, para_rule);
    cout << "MPO parallelization end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<ubond_t> bdims = {bond_dim};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    for (int i = 0; i < (int)targets.size(); i++)
        for (int j = 0; j < (int)targets[i].size(); j++) {

            S target = targets[i][j];

            shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
                hamil.n_sites, hamil.vacuum, target, hamil.basis);
            mps_info->set_bond_dimension(bond_dim);

            // MPS
            Random::rand_seed(0);

            shared_ptr<MPS<S>> mps = make_shared<MPS<S>>(hamil.n_sites, 0, 2);
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
            dmrg->iprint = 0;
            dmrg->decomp_type = dt;
            dmrg->noise_type = nt;
            double energy = dmrg->solve(10, mps->center == 0, 1E-8);

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

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);
        }

    mpo->deallocate();
}

TEST_F(TestSumMPON2STO3G, TestSZ) {

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<SZ>> para_comm =
        make_shared<MPICommunicator<SZ>>();
#else
    shared_ptr<ParallelCommunicator<SZ>> para_comm =
        make_shared<ParallelCommunicator<SZ>>(1, 0, 0);
#endif
    shared_ptr<ParallelRuleSumMPO<SZ>> para_rule =
        make_shared<ParallelRuleSumMPO<SZ>>(para_comm);

    shared_ptr<FCIDUMP> fcidump = make_shared<ParallelFCIDUMP<SZ>>(para_rule);
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);

    vector<vector<SZ>> targets(8);
    for (int i = 0; i < 8; i++) {
        targets[i].resize(5);
        for (int j = 0; j < 5; j++)
            targets[i][j] = SZ(fcidump->n_elec(), (j - 2) * 2, i);
    }

    vector<vector<double>> energies(8);
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
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ>(targets, energies, hamil, "SZ",
                  DecompositionTypes::DensityMatrix, NoiseTypes::DensityMatrix);

    targets.resize(2);
    energies.resize(2);

    test_dmrg<SZ>(targets, energies, hamil, "SZ PERT",
                  DecompositionTypes::DensityMatrix,
                  NoiseTypes::ReducedPerturbative);
    test_dmrg<SZ>(targets, energies, hamil, "SZ SVD", DecompositionTypes::SVD,
                  NoiseTypes::Wavefunction);
    test_dmrg<SZ>(targets, energies, hamil, "SZ PERT SVD",
                  DecompositionTypes::SVD, NoiseTypes::ReducedPerturbative);

    hamil.deallocate();
    fcidump->deallocate();
}