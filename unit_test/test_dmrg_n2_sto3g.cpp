
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRGN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = new DataFrame(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        delete frame_();
    }
};

TEST_F(TestDMRGN2STO3G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    mkl_set_num_threads(8);
    mkl_set_dynamic(0);

    SU2 vacuum(0);

    vector<vector<SU2>> targets(8);
    for (int i = 0; i < 8; i++) {
        targets[i].resize(3);
        for (int j = 0; j < 3; j++)
            targets[i][j] = SU2(fcidump->n_elec(), j * 2, i);
    }

    vector<vector<double>> energies(8);
    energies[0] = {-107.654122447525, -106.939132859668, -107.031449471627};
    energies[1] = {-106.959626154680, -106.999600016661, -106.633790589321};
    energies[2] = {-107.306744734756, -107.356943001688, -106.931515926732};
    energies[3] = {-107.306744734756, -107.356943001688, -106.931515926731};
    energies[4] = {-107.223155479270, -107.279409754727, -107.012640794842};
    energies[5] = {-107.208347039017, -107.343458537272, -106.227634428741};
    energies[6] = {-107.116397543375, -107.208021870379, -107.070427868786};
    energies[7] = {-107.116397543375, -107.208021870379, -107.070427868786};

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

    uint16_t bond_dim = 200;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 0.0};

    t.get_time();

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 3; j++) {

            SU2 target = targets[i][j];

            shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
                norb, vacuum, target, hamil.basis, hamil.orb_sym);
            mps_info->set_bond_dimension(bond_dim);

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
            shared_ptr<DMRG<SU2>> dmrg =
                make_shared<DMRG<SU2>>(me, bdims, noises);
            dmrg->iprint = 0;
            double energy = dmrg->solve(10, true, 1E-8);

            // deallocate persistent stack memory
            mps_info->deallocate();

            cout << "== SU2 ==" << setw(20) << target << " E = " << fixed
                 << setw(22) << setprecision(12) << energy
                 << " error = " << scientific << setprecision(3) << setw(10)
                 << (energy - energies[i][j]) << " T = " << fixed << setw(10)
                 << setprecision(3) << t.get_time() << endl;

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);
        }

    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestDMRGN2STO3G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    mkl_set_num_threads(8);
    mkl_set_dynamic(0);

    SZ vacuum(0);

    vector<vector<SZ>> targets(8);
    for (int i = 0; i < 8; i++) {
        targets[i].resize(3);
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
    mpo = make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    uint16_t bond_dim = 200;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 0.0};

    t.get_time();

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 5; j++) {

            SZ target = targets[i][j];

            shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
                norb, vacuum, target, hamil.basis, hamil.orb_sym);
            mps_info->set_bond_dimension(bond_dim);

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
            shared_ptr<DMRG<SZ>> dmrg =
                make_shared<DMRG<SZ>>(me, bdims, noises);
            dmrg->iprint = 0;
            double energy = dmrg->solve(10, true, 1E-8);

            // deallocate persistent stack memory
            mps_info->deallocate();

            cout << "== SZ  ==" << setw(20) << target << " E = " << fixed
                 << setw(22) << setprecision(12) << energy
                 << " error = " << scientific << setprecision(3) << setw(10)
                 << (energy - energies[i][j]) << " T = " << fixed << setw(10)
                 << setprecision(3) << t.get_time() << endl;

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);
        }

    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
