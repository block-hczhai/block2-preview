
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestNPDM : public ::testing::Test {
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

TEST_F(TestNPDM, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);
    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    HamiltonianQC<SU2> hamil(vacuum, target, norb, orbsym, fcidump);

    mkl_set_num_threads(8);
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

    // 1PDM MPO construction
    cout << "1PDM MPO start" << endl;
    shared_ptr<MPO<SU2>> pmpo = make_shared<PDM1MPOQC<SU2>>(hamil);
    cout << "1PDM MPO end .. T = " << t.get_time() << endl;

    // 1PDM MPO simplification
    cout << "1PDM MPO simplification start" << endl;
    pmpo =
        make_shared<SimplifiedMPO<SU2>>(pmpo, make_shared<Rule<SU2>>(), true);
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;
    // cout << pmpo->get_blocking_formulas() << endl;
    // abort();

    // 1NPC MPO construction
    cout << "1NPC MPO start" << endl;
    shared_ptr<MPO<SU2>> nmpo = make_shared<NPC1MPOQC<SU2>>(hamil);
    cout << "1NPC MPO end .. T = " << t.get_time() << endl;

    // 1NPC MPO simplification
    cout << "1NPC MPO simplification start" << endl;
    nmpo =
        make_shared<SimplifiedMPO<SU2>>(nmpo, make_shared<Rule<SU2>>(), true);
    cout << "1NPC MPO simplification end .. T = " << t.get_time() << endl;
    // cout << nmpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
        norb, vacuum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(384666);
    shared_ptr<MPS<SU2>> mps = make_shared<MPS<SU2>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

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

    cout << *frame_() << endl;
    frame_()->activate(0);

    // DMRG
    vector<uint16_t> bdims = {250, 250, 250, 250, 250, 500, 500};
    vector<double> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-8, 0};
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->davidson_conv_thrds = vector<double>{1E-8};
    dmrg->solve(30, true);

    // 1PDM ME
    shared_ptr<MovingEnvironment<SU2>> pme =
        make_shared<MovingEnvironment<SU2>>(pmpo, mps, mps, "1PDM");
    t.get_time();
    cout << "1PDM INIT start" << endl;
    pme->init_environments(false);
    cout << "1PDM INIT end .. T = " << t.get_time() << endl;

    // 1PDM
    shared_ptr<Expect<SU2>> expect =
        make_shared<Expect<SU2>>(pme, bond_dim, bond_dim);
    expect->solve(true, dmrg->forward);
    expect->solve(true, !dmrg->forward);

    MatrixRef dm = expect->get_1pdm_spatial();
    for (int i = 0; i < dm.m; i++)
        for (int j = 0; j < dm.n; j++)
            if (abs(dm(i, j)) > TINY)
                cout << setw(5) << i << setw(5) << j << fixed
                     << setprecision(20) << setw(25) << dm(i, j) << endl;

    dm.deallocate();

    // 1NPC ME
    shared_ptr<MovingEnvironment<SU2>> nme =
        make_shared<MovingEnvironment<SU2>>(nmpo, mps, mps, "1NPC");
    t.get_time();
    cout << "1NPC INIT start" << endl;
    nme->init_environments(false);
    cout << "1NPC INIT end .. T = " << t.get_time() << endl;

    // 1NPC
    expect = make_shared<Expect<SU2>>(nme, bond_dim, bond_dim);
    expect->solve(true, dmrg->forward);
    expect->solve(true, !dmrg->forward);

    MatrixRef dmx = expect->get_1npc_spatial(1);
    for (int i = 0; i < dmx.m; i++)
        for (int j = 0; j < dmx.n; j++)
            if (abs(dmx(i, j)) > TINY)
                cout << setw(5) << i << setw(5) << j << fixed
                     << setprecision(20) << setw(25) << dmx(i, j) << endl;

    dmx.deallocate();

    // deallocate persistent stack memory
    mps_info->deallocate();
    nmpo->deallocate();
    pmpo->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestNPDM, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);
    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, target, norb, orbsym, fcidump);

    mkl_set_num_threads(8);
    mkl_set_dynamic(0);

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

    // 1PDM MPO construction
    cout << "1PDM MPO start" << endl;
    shared_ptr<MPO<SZ>> pmpo = make_shared<PDM1MPOQC<SZ>>(hamil);
    cout << "1PDM MPO end .. T = " << t.get_time() << endl;

    // 1PDM MPO simplification
    cout << "1PDM MPO simplification start" << endl;
    pmpo = make_shared<SimplifiedMPO<SZ>>(pmpo, make_shared<Rule<SZ>>(), true);
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;
    // cout << pmpo->get_blocking_formulas() << endl;
    // abort();

    // 1NPC MPO construction
    cout << "1NPC MPO start" << endl;
    shared_ptr<MPO<SZ>> nmpo = make_shared<NPC1MPOQC<SZ>>(hamil);
    cout << "1NPC MPO end .. T = " << t.get_time() << endl;

    // 1NPC MPO simplification
    cout << "1NPC MPO simplification start" << endl;
    nmpo = make_shared<SimplifiedMPO<SZ>>(nmpo, make_shared<Rule<SZ>>(), true);
    cout << "1NPC MPO simplification end .. T = " << t.get_time() << endl;
    // cout << nmpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
        norb, vacuum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(384666);
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    hamil.opf->seq->mode = SeqTypes::Simple;
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_() << endl;
    frame_()->activate(0);

    // DMRG
    vector<uint16_t> bdims = {250, 250, 250, 250, 250, 500, 500};
    vector<double> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-8, 0};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->davidson_conv_thrds = vector<double>{1E-8};
    dmrg->solve(30, true);

    // 1PDM ME
    shared_ptr<MovingEnvironment<SZ>> pme =
        make_shared<MovingEnvironment<SZ>>(pmpo, mps, mps, "1PDM");
    t.get_time();
    cout << "1PDM INIT start" << endl;
    pme->init_environments(false);
    cout << "1PDM INIT end .. T = " << t.get_time() << endl;

    // 1PDM
    shared_ptr<Expect<SZ>> expect =
        make_shared<Expect<SZ>>(pme, bond_dim, bond_dim);
    expect->solve(true, dmrg->forward);
    expect->solve(true, !dmrg->forward);

    MatrixRef dm = expect->get_1pdm();
    for (int i = 0; i < dm.m; i++)
        for (int j = 0; j < dm.n; j++)
            if (abs(dm(i, j)) > TINY)
                cout << setw(5) << i << setw(5) << j << fixed
                     << setprecision(20) << setw(25) << dm(i, j) << endl;

    dm.deallocate();

    // 1NPC ME
    shared_ptr<MovingEnvironment<SZ>> nme =
        make_shared<MovingEnvironment<SZ>>(nmpo, mps, mps, "1NPC");
    t.get_time();
    cout << "1NPC INIT start" << endl;
    nme->init_environments(false);
    cout << "1NPC INIT end .. T = " << t.get_time() << endl;

    // 1NPC
    expect = make_shared<Expect<SZ>>(nme, bond_dim, bond_dim);
    expect->solve(true, dmrg->forward);
    expect->solve(true, !dmrg->forward);

    MatrixRef dmx = expect->get_1npc(1);
    for (int i = 0; i < dmx.m; i++)
        for (int j = 0; j < dmx.n; j++)
            if (abs(dmx(i, j)) > TINY)
                cout << setw(5) << i << setw(5) << j << fixed
                     << setprecision(20) << setw(25) << dmx(i, j) << endl;

    dmx.deallocate();

    // deallocate persistent stack memory
    mps_info->deallocate();
    nmpo->deallocate();
    pmpo->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
