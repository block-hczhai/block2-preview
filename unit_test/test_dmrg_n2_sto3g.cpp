
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRGN2STO3G : public ::testing::Test {
  protected:
    size_t isize = 1L << 20;
    size_t dsize = 1L << 24;
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

template <typename S>
void test_dmrg(const vector<vector<S>> &targets,
               const vector<vector<double>> &energies,
               const HamiltonianQC<S> &hamil, const string &name,
               DecompositionTypes dt, NoiseTypes nt) {

    hamil.opf->seq->mode = SeqTypes::Simple;

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(8);
    mkl_set_dynamic(0);
#endif

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

    uint16_t bond_dim = 200;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 1E-7, 0.0};

    t.get_time();

    for (int i = 0; i < (int)targets.size(); i++)
        for (int j = 0; j < 3; j++) {

            S target = targets[i][j];

            shared_ptr<MPSInfo<S>> mps_info =
                make_shared<MPSInfo<S>>(hamil.n_sites, hamil.vacuum, target,
                                        hamil.basis, hamil.orb_sym);
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

            cout << "== " << name << " ==" << setw(20) << target
                 << " E = " << fixed << setw(22) << setprecision(12) << energy
                 << " error = " << scientific << setprecision(3) << setw(10)
                 << (energy - energies[i][j]) << " T = " << fixed << setw(10)
                 << setprecision(3) << t.get_time() << endl;

            EXPECT_LT(abs(energy - energies[i][j]), 1E-7);
        }

    mpo->deallocate();
}

TEST_F(TestDMRGN2STO3G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.STO3G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

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

    test_dmrg<SU2>(targets, energies, hamil, "SU2",
                   DecompositionTypes::DensityMatrix,
                   NoiseTypes::DensityMatrix);

    targets.resize(2);
    energies.resize(2);

    test_dmrg<SU2>(targets, energies, hamil, "SU2 PERT",
                   DecompositionTypes::DensityMatrix, NoiseTypes::Perturbative);
    test_dmrg<SU2>(targets, energies, hamil, "SU2 SVD", DecompositionTypes::SVD,
                   NoiseTypes::Wavefunction);

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

    test_dmrg<SZ>(targets, energies, hamil, "SZ",
                  DecompositionTypes::DensityMatrix, NoiseTypes::DensityMatrix);

    targets.resize(2);
    energies.resize(2);

    test_dmrg<SZ>(targets, energies, hamil, "SZ PERT",
                  DecompositionTypes::DensityMatrix, NoiseTypes::Perturbative);
    test_dmrg<SZ>(targets, energies, hamil, "SZ SVD", DecompositionTypes::SVD,
                  NoiseTypes::Wavefunction);

    hamil.deallocate();
    fcidump->deallocate();
}
