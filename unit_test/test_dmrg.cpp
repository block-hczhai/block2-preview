
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
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

TEST_F(TestDMRG, Test) {
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

    Timer t;
    t.get_time();
    // MPO
    cout << "MPO start" << endl;
    shared_ptr<MPO> mpo = make_shared<QCMPO>(hamil);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO>(mpo, make_shared<RuleQCSU2>());
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;

    uint16_t bond_dim = 500;

    // MPSInfo
    shared_ptr<MPSInfo> mps_info = make_shared<MPSInfo>(
        norb, vaccum, target, hamil.basis, &hamil.orb_sym[0], hamil.n_syms);
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
    Random::rand_seed(1969);
    shared_ptr<MPS> mps = make_shared<MPS>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    frame->activate(0);
    cout << "persistent memory used :: I = " << ialloc->used
         << " D = " << dalloc->used << endl;
    frame->activate(1);
    cout << "exclusive  memory used :: I = " << ialloc->used
         << " D = " << dalloc->used << endl;
    // ME
    shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(hamil.opf);
    shared_ptr<MovingEnvironment> me =
        make_shared<MovingEnvironment>(mpo, mps, mps, tf, hamil.site_op_infos);
    me->init_environments();

    cout << *frame << endl;
    frame->activate(0);

    // DMRG
    shared_ptr<DMRG> dmrg =
        make_shared<DMRG>(me, vector<uint16_t>{bond_dim}, vector<double>{0.0});
    // dmrg->update_two_dot(0, true, 500, 0.0);
    // dmrg->me->move_to(1);
    // dmrg->contract_two_dot(1);
    // dmrg->update_two_dot(1, true, 500, 0.0);
    // // cout << *me->ket->tensors[2]->info << endl;
    // // cout << *me->ket->tensors[2] << endl;
    dmrg->solve(10, true);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
