
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

class TestDMRGN2STO3GSA : public ::testing::Test {
    static bool _mpi;

  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 30;

    template <typename S>
    void test_dmrg(const vector<S> &targets, const vector<double> &energies,
                   const HamiltonianQC<S> &hamil, const string &name,
                   uint16_t bond_dim, uint16_t nroots);
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

bool TestDMRGN2STO3GSA::_mpi = MPITest::okay();

template <typename S>
void TestDMRGN2STO3GSA::test_dmrg(const vector<S> &targets,
                                  const vector<double> &energies,
                                  const HamiltonianQC<S> &hamil,
                                  const string &name, uint16_t bond_dim,
                                  uint16_t nroots) {

    hamil.opf->seq->mode = SeqTypes::Simple;

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(1);
    mkl_set_dynamic(0);
#endif

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<MPICommunicator<S>>();
#else
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<ParallelCommunicator<S>>(1, 0, 0);
#endif
    shared_ptr<ParallelRule<S>> para_rule =
        make_shared<ParallelRuleQC<S>>(para_comm);

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S>>(mpo, make_shared<RuleQC<S>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // MPO parallelization
    cout << "MPO parallelization start" << endl;
    mpo = make_shared<ParallelMPO<S>>(mpo, para_rule);
    cout << "MPO parallelization end .. T = " << t.get_time() << endl;

    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 1E-7, 0.0};

    t.get_time();

    shared_ptr<MultiMPSInfo<S>> mps_info = make_shared<MultiMPSInfo<S>>(
        hamil.n_sites, hamil.vacuum, targets, hamil.basis);
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

    // DMRG
    shared_ptr<DMRG<S>> dmrg = make_shared<DMRG<S>>(me, bdims, noises);
    dmrg->iprint = 2;
    double energy = dmrg->solve(10, mps->center == 0, 1E-8);

    // deallocate persistent stack memory
    mps_info->deallocate();

    para_comm->reduce_sum(&para_comm->tcomm, 1, para_comm->root);
    para_comm->tcomm /= para_comm->size;
    double tt = t.get_time();

    for (size_t i = 0; i < dmrg->energies.back().size(); i++) {
        cout << "== " << name << " (SA) =="
             << " E[" << setw(2) << i << "] = " << fixed << setw(22)
             << setprecision(12) << dmrg->energies.back()[i]
             << " error = " << scientific << setprecision(3) << setw(10)
             << (dmrg->energies.back()[i] - energies[i]) << " T = " << fixed
             << setw(10) << setprecision(3) << tt << " Tcomm = " << fixed
             << setw(10) << setprecision(3) << para_comm->tcomm << " ("
             << setw(3) << fixed << setprecision(0)
             << (para_comm->tcomm * 100 / tt) << "%)" << endl;

        EXPECT_LT(abs(dmrg->energies.back()[i] - energies[i]), 1E-7);
    }

    para_comm->tcomm = 0.0;

    mpo->deallocate();
}

TEST_F(TestDMRGN2STO3GSA, TestSU2) {

    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
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

    vector<double> energies = {
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
    HamiltonianQC<SU2> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2>(targets, energies, hamil, "SU2", 200, 10);

    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestDMRGN2STO3GSA, TestSZ) {

    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);

    vector<SZ> targets;
    int ne = fcidump->n_elec() / 2;
    for (int i = 0; i < 8; i++)
        for (int na = ne - 1; na <= ne + 1; na++)
            for (int nb = ne - 1; nb <= ne + 1; nb++)
                targets.push_back(SZ(na + nb, na - nb, i));

    vector<double> energies = {
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
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ>(targets, energies, hamil, "SZ", 400, 16);

    hamil.deallocate();
    fcidump->deallocate();
}
