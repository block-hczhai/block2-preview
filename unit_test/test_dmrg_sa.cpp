
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestSA : public ::testing::Test {
  protected:
    size_t isize = 1L << 26;
    size_t dsize = 1L << 30;
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

// Multi-target state-averaged DMRG
TEST_F(TestSA, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    // string occ_filename = "data/CR2.SVP.OCC";
    // string occ_filename = "data/CR2.SVP.HF"; // E(HF) = -2085.53318786766
    // occs = read_occ(occ_filename);
    // string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    fcidump->read(filename);

    // vector<uint8_t> ioccs;
    // for (auto x : occs)
    //     ioccs.push_back(uint8_t(x));

    // cout << "HF energy = " << fixed << setprecision(12)
    //      << fcidump->det_energy(ioccs, 0, fcidump->n_sites()) + fcidump->e
    //      << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    vector<SU2> targets = {target};
    targets.push_back(
        SU2(fcidump->n_elec(), 2, PointGroup::swap_pg(pg)(fcidump->isym())));
    targets.push_back(
        SU2(fcidump->n_elec(), 4, PointGroup::swap_pg(pg)(fcidump->isym())));
    targets.push_back(SU2(fcidump->n_elec() + 1, 1,
                          PointGroup::swap_pg(pg)(fcidump->isym())));
    targets.push_back(SU2(fcidump->n_elec(), 0, 2));
    targets.push_back(SU2(fcidump->n_elec(), 2, 2));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    HamiltonianQC<SU2> hamil(vacuum, norb, orbsym, fcidump);

    mkl_set_num_threads(2);
    mkl_set_dynamic(0);

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
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 250;

    shared_ptr<MultiMPSInfo<SU2>> mps_info = make_shared<MultiMPSInfo<SU2>>(
        norb, vacuum, targets, hamil.basis, hamil.orb_sym);
    mps_info->set_bond_dimension(bond_dim);

    cout << "left dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->left_dims[i].n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->right_dims[i].n_states_total << " ";
    cout << endl;

    // MPS
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    // Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    shared_ptr<MultiMPS<SU2>> mps = make_shared<MultiMPS<SU2>>(norb, 0, 2, 24);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // for (int i = 0; i < mps->n_sites; i++)
    //     if (mps->tensors[i] != nullptr)
    //         cout << *mps->tensors[i] << endl;
    // abort();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    hamil.opf->seq->mode = SeqTypes::Simple;
    shared_ptr<MovingEnvironment<SU2>> me =
        make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    // abort();

    cout << *frame_() << endl;
    frame_()->activate(0);

    // DMRG
    // vector<uint16_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
    //                           500, 500, 750, 750, 750, 750, 750};
    // vector<double> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7,
    //                          1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 0.0};
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6};
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    dmrg->iprint = 2;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg->solve(50, true);

    vector<int> multiplicities(dmrg->energies.back().size());
    for (size_t i = 0; i < dmrg->energies.back().size(); i++)
        multiplicities[i] = dmrg->mps_quanta.back()[i][0].first.multiplicity();

    // 1PDM MPO construction
    cout << "1PDM MPO start" << endl;
    shared_ptr<MPO<SU2>> pmpo = make_shared<PDM1MPOQC<SU2>>(hamil);
    cout << "1PDM MPO end .. T = " << t.get_time() << endl;

    // 1PDM MPO simplification
    cout << "1PDM MPO simplification start" << endl;
    pmpo =
        make_shared<SimplifiedMPO<SU2>>(pmpo, make_shared<Rule<SU2>>(), true);
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;

    // 1PDM ME
    shared_ptr<MovingEnvironment<SU2>> pme =
        make_shared<MovingEnvironment<SU2>>(pmpo, mps, mps, "1PDM");
    t.get_time();
    cout << "1PDM INIT start" << endl;
    pme->init_environments(false);
    cout << "1PDM INIT end .. T = " << t.get_time() << endl;

    // 1PDM
    double beta = 60.0;
    shared_ptr<Expect<SU2>> expect = make_shared<Expect<SU2>>(
        pme, bond_dim, bond_dim, beta, dmrg->energies.back(), multiplicities);
    expect->solve(true, dmrg->forward);

    cout << fixed << setprecision(15);
    for (size_t i = 0; i < expect->partition_weights.size(); i++)
        cout << "[" << i << "] = " << dmrg->energies.back()[i] << " "
             << expect->partition_weights[i] << endl;

    // deallocate persistent stack memory
    pmpo->deallocate();
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
