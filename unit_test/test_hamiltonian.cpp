
#include "quantum.hpp"
#include "gtest/gtest.h"

using namespace block2;

class TestHamiltonian : public ::testing::Test {
  protected:
    size_t isize = 1E7;
    size_t dsize = 1E7;
    void SetUp() override {
        Random::rand_seed(0);
        ialloc = new StackAllocator<uint32_t>(new uint32_t[isize], isize);
        dalloc = new StackAllocator<double>(new double[dsize], dsize);
    }
    void TearDown() override {
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete[] ialloc->data;
        delete[] dalloc->data;
    }
};

TEST_F(TestHamiltonian, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    string filename = "data/CR2.SVP.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(), Hamiltonian::swap_d2h);
    SpinLabel vaccum(0);
    SpinLabel target(fcidump->n_elec(), fcidump->twos(),
                     Hamiltonian::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    // cout << ialloc->used << " " << dalloc->used << endl;
    Hamiltonian hamil(vaccum, target, norb, su2, fcidump, orbsym);
    // for (auto &g : hamil.site_norm_ops[0]) {
    //     cout << "OP=" << g.first << endl;
    //     cout << *(g.second->info);
    //     cout << *(g.second);
    // }
    // cout << ialloc->used << " " << dalloc->used << endl;
    hamil.deallocate();
    fcidump->deallocate();
}
