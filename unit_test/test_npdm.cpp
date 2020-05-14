
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestNPDM : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        Random::rand_seed(0);
        frame = new DataFrame(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame->activate(0);
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete frame;
    }
};

TEST_F(TestNPDM, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              Hamiltonian::swap_d2h);
    SpinLabel vaccum(0);
    SpinLabel target(fcidump->n_elec(), fcidump->twos(),
                     Hamiltonian::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    Hamiltonian hamil(vaccum, target, norb, su2, fcidump, orbsym);
    hamil.opf->seq->mode = SeqTypes::Simple;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO> mpo = make_shared<MPOQCSU2>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO>(mpo, make_shared<RuleQCSU2>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    // 1PDM MPO construction
    cout << "1PDM MPO start" << endl;
    shared_ptr<MPO> pmpo = make_shared<PDM1MPOQCSU2>(hamil);
    cout << "1PDM MPO end .. T = " << t.get_time() << endl;

    // 1PDM MPO simplification
    cout << "1PDM MPO simplification start" << endl;
    pmpo = make_shared<SimplifiedMPO>(pmpo, make_shared<Rule>(), true);
    cout << "1PDM MPO simplification end .. T = " << t.get_time() << endl;
    // cout << pmpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo> mps_info = make_shared<MPSInfo>(
        norb, vaccum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(384666);
    shared_ptr<MPS> mps = make_shared<MPS>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment> me = make_shared<MovingEnvironment>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame << endl;
    frame->activate(0);

    // DMRG
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6};
    shared_ptr<DMRG> dmrg = make_shared<DMRG>(me, bdims, noises);
    dmrg->solve(30, true);

    // 1PDM ME
    shared_ptr<MovingEnvironment> pme = make_shared<MovingEnvironment>(pmpo, mps, mps, "1PDM");
    t.get_time();
    cout << "1PDM INIT start" << endl;
    pme->init_environments(false);
    cout << "1PDM INIT end .. T = " << t.get_time() << endl;

    // 1PDM
    shared_ptr<Expect> expect = make_shared<Expect>(pme, bond_dim, bond_dim);
    expect->solve(true, dmrg->forward);
    expect->solve(true, !dmrg->forward);

    // deallocate persistent stack memory
    mps_info->deallocate();
    pmpo->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
