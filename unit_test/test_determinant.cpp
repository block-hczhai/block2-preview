
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDET : public ::testing::Test {
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

TEST_F(TestDET, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    fcidump->read(filename);

    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    HamiltonianQC<SZ> hamil(vacuum, norb, orbsym, fcidump);

    mkl_set_num_threads(1);
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
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    uint16_t bond_dim = 250;

    // MPSInfo
    shared_ptr<MPSInfo<SZ>> mps_info = make_shared<MPSInfo<SZ>>(
        norb, vacuum, target, hamil.basis, hamil.orb_sym);
    mps_info->set_bond_dimension(bond_dim);

    cout << "left dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->left_dims[i].n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= norb; i++)
        cout << mps_info->right_dims[i].n_states_total << " ";
    cout << endl;
    // abort();

    // MPS
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    // Random::rand_seed(384666);
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
    vector<uint16_t> bdims = {bond_dim};
    vector<double> noises = {1E-6, 0.0};
    shared_ptr<DMRG<SZ>> dmrg = make_shared<DMRG<SZ>>(me, bdims, noises);
    dmrg->iprint = 2;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg->solve(3, true, 1E-10);

    // shared_ptr<UnfusedMPS<SZ>> unfused = make_shared<UnfusedMPS<SZ>>();
    // unfused->initialize(mps);
    // mps->load_tensor(0);
    // cout << *mps->tensors[0]->info << endl;
    // cout << *mps->tensors[0] << endl;
    // mps->unload_tensor(0);
    // cout << *unfused->tensors[0] << endl;

    shared_ptr<DeterminantTRIE<SZ>> dtrie =
        make_shared<DeterminantTRIE<SZ>>(mps->n_sites, true);
    vector<uint8_t> ref = {1, 1, 1, 2, 2, 2, 3, 3, 3, 3};
    do {
        dtrie->push_back(ref);
    } while (next_permutation(ref.begin(), ref.end()));
    dtrie->evaluate(make_shared<UnfusedMPS<SZ>>(mps));
    for (int i = 0; i < (int)dtrie->size(); i++) {
        for (auto x : (*dtrie)[i])
            cout << (int)x;
        cout << " = " << setprecision(30) << dtrie->vals[i] << endl;
    }
    cout << dtrie->size() << " " << dtrie->data.size() << endl;

    // cout << dtrie->size() << " " << dtrie->data.size() << endl;
    // dtrie->push_back(vector<uint8_t>{0, 0, 0, 0});
    // dtrie->push_back(vector<uint8_t>{0, 1, 3, 0});
    // dtrie->push_back(vector<uint8_t>{1, 3, 0, 0});
    // cout << dtrie->size() << " " << dtrie->data.size() << endl;
    // cout << dtrie->find(vector<uint8_t>{0, 1, 3, 0}) << endl;
    // cout << dtrie->find(vector<uint8_t>{1, 3, 0, 0}) << endl;
    // cout << dtrie->find(vector<uint8_t>{1, 3, 1, 0}) << endl;
    // cout << dtrie->find(vector<uint8_t>{0, 0, 0, 0}) << endl;

    // for (int i = 0; i < dtrie->invs.size(); i++)
    //     cout << dtrie->invs[i] << " ";
    // cout << endl;
    // for (int i = 0; i < dtrie->size(); i++) {
    //     cout << i << " = ";
    //     for (auto x : (*dtrie)[i])
    //         cout << (int)x;
    //     cout << endl;
    // }

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
