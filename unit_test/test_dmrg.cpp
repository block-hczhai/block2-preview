
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1E9;
    size_t dsize = 1E9;
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

TEST_F(TestDMRG, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    // string filename = "data/CR2.SVP.FCIDUMP";
    // string filename = "data/N2.STO3G.FCIDUMP";
    string filename = "data/HUBBARD-L8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              Hamiltonian::swap_d2h);
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
    // MPSInfo
    shared_ptr<MPSInfo> mps_info = make_shared<MPSInfo>(
        norb, vaccum, target, hamil.basis, &hamil.orb_sym[0], hamil.n_syms);
    // cout << "left min dims fci = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->left_dims_fci[i].n << " ";
    // cout << endl;
    // cout << "right min dims fci = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->right_dims_fci[i].n << " ";
    // cout << endl;
    // cout << "left max dims fci = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->left_dims_fci[i].n_states_total << " ";
    // cout << endl;
    // cout << "right max dims fci = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->right_dims_fci[i].n_states_total << " ";
    // cout << endl;
    mps_info->set_bond_dimension(500);
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
    cout << ialloc->used << " mpsi " << dalloc->used << endl;
    mps->initialize(mps_info);
    cout << ialloc->used << " mpsf " << dalloc->used << endl;
    mps->random_canonicalize();
    cout << ialloc->used << " mpsff " << dalloc->used << endl;

    // MPO
    cout << "MPO start" << endl;
    shared_ptr<MPO> mpo = make_shared<QCMPO>(hamil);
    cout << "MPO end" << endl;

    // ME
    shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(hamil.opf);
    shared_ptr<MovingEnvironment> me =
        make_shared<MovingEnvironment>(mpo, mps, mps, tf, hamil.site_op_infos);
    me->init_environments();

    // DMRG
    shared_ptr<DMRG> dmrg =
        make_shared<DMRG>(me, vector<uint16_t>{500}, vector<double>{0});
    // dmrg->update_two_dot(0, true, 500, 0.0);
    // dmrg->me->move_to(1);
    // dmrg->contract_two_dot(1);
    // dmrg->update_two_dot(1, true, 500, 0.0);
    // cout << *me->ket->tensors[2]->info << endl;
    // cout << *me->ket->tensors[2] << endl;
    dmrg->solve(10, true);

    // Deallocation
    me->deallocate();
    mpo->deallocate();
    mps->deallocate();
    mps_info->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
