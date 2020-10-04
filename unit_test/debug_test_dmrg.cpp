
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
    PGTypes pg = PGTypes::D2H;

    string occ_filename = "data/CR2.SVP.OCC";
    // string occ_filename = "data/CR2.SVP.HF"; // E(HF) = -2085.53318786766
    occs = read_occ(occ_filename);
    string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    fcidump->read(filename);

    vector<uint8_t> ioccs;
    for (auto x : occs)
        ioccs.push_back(uint8_t(x));

    // cout << "HF energy = " << fixed << setprecision(12)
    //      << fcidump->det_energy(ioccs, 0, fcidump->n_sites()) + fcidump->e
    //      << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    HamiltonianQC<SU2> hamil(vacuum, norb, orbsym, fcidump);

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(8);
    mkl_set_dynamic(0);
#endif

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
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    ubond_t bond_dim = 200;

    // MPSInfo
    // shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
    //     norb, vacuum, target, hamil.basis);

    // CCSD init
    shared_ptr<MPSInfo<SU2>> mps_info =
        make_shared<MPSInfo<SU2>>(norb, vacuum, target, hamil.basis);
    if (occs.size() == 0)
        mps_info->set_bond_dimension(bond_dim);
    else {
        assert(occs.size() == norb);
        // for (size_t i = 0; i < occs.size(); i++)
        //     cout << occs[i] << " ";
        mps_info->set_bond_dimension_using_occ(bond_dim, occs, 1);
        // mps_info->set_bond_dimension_using_hf(bond_dim, occs, 0);
    }

    // Local init
    // shared_ptr<DynamicMPSInfo<SU2>> mps_info =
    //     make_shared<DynamicMPSInfo<SU2>>(norb, vacuum, target, hamil.basis,
    //                                      hamil.orb_sym, ioccs);
    // mps_info->n_local = 4;
    // mps_info->set_bond_dimension(bond_dim);

    // Determinant init
    // shared_ptr<DeterminantMPSInfo<SU2>> mps_info =
    //     make_shared<DeterminantMPSInfo<SU2>>(norb, vacuum, target,
    //     hamil.basis,
    //                                      hamil.orb_sym, ioccs, fcidump);
    // mps_info->set_bond_dimension(bond_dim);

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
    Random::rand_seed(0);
    // int x = Random::rand_int(0, 1000000);
    Random::rand_seed(384666);
    // cout << "Random = " << x << endl;
    shared_ptr<MPS<SU2>> mps = make_shared<MPS<SU2>>(norb, 0, 2);
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
    shared_ptr<MovingEnvironment<SU2>> me =
        make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_() << endl;
    frame_()->activate(0);

    // DMRG
    // vector<ubond_t> bdims = {50};
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-5,  1E-5,  1E-6,  1E-6, 1E-6, 1E-6, 1E-7,
                             1E-7,  1E-7,  1E-7,  1E-7, 1E-7, 1E-7};
    vector<double> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5,
                              1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                              1E-6, 1E-6, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7,
                              5E-7};
    // noises = vector<double>{1E-5};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    // dmrg->cutoff = 0;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    // dmrg->decomp_type = DecompositionTypes::SVD;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg->solve(50, true, 0.0);

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
