
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
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
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(), PointGroup::swap_d2h);
    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(), PointGroup::swap_d2h(fcidump->isym()));
    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    // abort();

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(2);
    mkl_set_dynamic(0);
#endif

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SZ>> mpo = make_shared<MPOQC<SZ>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    ubond_t bond_dim = 100;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
        norb, vacuum, target, hamil.basis);
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
        cout << mps_info->left_dims[i]->n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->right_dims[i]->n_states_total << " ";
    cout << endl;
    // abort();

    // MPS
    // Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    shared_ptr<MPS<SZ>> mps = make_shared<MPS<SZ>>(norb, 0, 2);
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

    frame_()->activate(0);
    cout << "persistent memory used :: I = " << ialloc_()->used
         << " D = " << dalloc_()->used << endl;
    frame_()->activate(1);
    cout << "exclusive  memory used :: I = " << ialloc_()->used
         << " D = " << dalloc_()->used << endl;
    // abort();
    // ME
    hamil.opf->seq->mode = SeqTypes::Simple;
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(true);
    cout << "INIT end .. T = " << t.get_time() << endl;

    // cout << *frame << endl;
    // frame->activate(0);
    // abort();

    // DMRG
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                              500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 0.0};
    // vector<ubond_t> bdims = {bond_dim};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->solve(30, true);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
