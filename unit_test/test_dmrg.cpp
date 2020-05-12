
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 36;
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

TEST_F(TestDMRG, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    string occ_filename = "data/CR2.SVP.OCC";
    occs = read_occ(occ_filename);
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
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO> mpo = make_shared<QCMPO>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO>(mpo, make_shared<RuleQCSU2>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo> mps_info = make_shared<MPSInfo>(
        norb, vaccum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == norb);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs);
    }
    // cout << "left min dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->left_dims_fci[i].n << " ";
    // cout << endl;
    // cout << "right min dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->right_dims_fci[i].n << " ";
    // cout << endl;
    // cout << "left q dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->left_dims[i].n << " ";
    // cout << endl;
    // cout << "right q dims = ";
    // for (int i = 0; i <= norb; i++)
    //     cout << mps_info->right_dims[i].n << " ";
    // cout << endl;
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
    hamil.opf->seq->mode = SeqTypes::Simple;
    shared_ptr<MovingEnvironment> me =
        make_shared<MovingEnvironment>(mpo, mps, mps, tf, hamil.site_op_infos);
    me->init_environments();

    cout << *frame << endl;
    frame->activate(0);

    // DMRG
    vector<uint16_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                              500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 0.0};
    // vector<uint16_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG> dmrg = make_shared<DMRG>(me, bdims, noises);
    dmrg->solve(30, true);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
