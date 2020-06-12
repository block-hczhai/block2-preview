
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestHamiltonian : public ::testing::Test {
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

TEST_F(TestHamiltonian, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    // string filename = "data/CR2.SVP.FCIDUMP";
    // string filename = "data/N2.STO3G.FCIDUMP";
    string filename = "data/HUBBARD-L8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              Hamiltonian::swap_d2h);
    SpinLabel vacuum(0);
    SpinLabel target(fcidump->n_elec(), fcidump->twos(),
                     Hamiltonian::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    // cout << ialloc->used << " " << dalloc->used << endl;
    Hamiltonian hamil(vacuum, target, norb, su2, fcidump, orbsym);
    // for (auto &g : hamil.site_norm_ops[0]) {
    //     cout << "OP=" << g.first << endl;
    //     cout << *(g.second->info);
    //     cout << *(g.second);
    // }
    // MPSInfo
    shared_ptr<MPSInfo> mps_info = make_shared<MPSInfo>(
        norb, vacuum, target, hamil.basis, &hamil.orb_sym[0], hamil.n_syms);
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
    me->deallocate();

    cout << ialloc->used << " " << dalloc->used << endl;
    shared_ptr<StateInfo> si = make_shared<StateInfo>(StateInfo::tensor_product(
        hamil.basis[0], hamil.basis[hamil.n_syms - 1], hamil.target));
    cout << ialloc->used << " " << dalloc->used << endl;
    cout << *si << endl;
    shared_ptr<SparseMatrixInfo> smi = make_shared<SparseMatrixInfo>();
    smi->initialize(*si, *si, hamil.vacuum, false);
    shared_ptr<SparseMatrix> c = make_shared<SparseMatrix>();
    c->allocate(smi);
    // cout << "total = " << c->total_memory << endl;
    shared_ptr<OpExpr> op_h =
        make_shared<OpElement>(OpNames::H, vector<uint8_t>{}, hamil.vacuum);
    shared_ptr<SparseMatrix> a = mpo->tensors[0]->lop[op_h];
    shared_ptr<SparseMatrix> b = mpo->tensors[1]->lop[op_h];
    // hamil.opf->tensor_product(*a, *b, *c);
    // // cout << "c = " << *c->info << *c << endl;
    c->deallocate();
    smi->deallocate();
    si->deallocate();
    mpo->deallocate();
    mps->deallocate();
    mps_info->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
