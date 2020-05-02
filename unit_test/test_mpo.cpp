
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestMPO : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 36;
    void SetUp() override {
        Random::rand_seed(0);
        frame = new DataFrame(isize, dsize);
    }
    void TearDown() override {
        frame->activate(0);
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete frame;
    }
};

TEST_F(TestMPO, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
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

    // MPO
    Timer t;
    t.get_time();
    cout << "MPO start" << endl;
    shared_ptr<MPO> mpo = make_shared<QCMPO>(hamil);
    cout << "MPO end" << endl;
    cout << "T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    shared_ptr<Rule> rule = make_shared<RuleQCSU2>();
    shared_ptr<SimplifiedMPO> smpo = make_shared<SimplifiedMPO>(mpo, rule);
    cout << "MPO simplification end" << endl;
    cout << "T = " << t.get_time() << endl;

    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
