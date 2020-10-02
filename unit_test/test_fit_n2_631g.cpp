
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestFITN2631G : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;

    template <typename S>
    void test_dmrg(int n_ext, int ci_order, const S target, double energy,
                   HamiltonianQC<S> hamil, const string &name,
                   DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodexx");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

template <typename S>
void TestFITN2631G::test_dmrg(int n_ext, int ci_order, const S target,
                              double energy, HamiltonianQC<S> hamil,
                              const string &name, DecompositionTypes dt,
                              NoiseTypes nt) {

    hamil.opf->seq->mode = SeqTypes::Simple;
    bool dcl = false;
    int dot = 2;

#ifdef _HAS_INTEL_MKL
    mkl_set_num_threads(8);
    mkl_set_dynamic(0);
#endif

    Timer t;
    t.get_time();
    // MPO construction (MRCISD-DMRG)
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional, hamil.n_sites / 3);
    cout << "MPO end .. T = " << t.get_time() << endl;

    cout << "MPO fusing start" << endl;
    shared_ptr<MPSInfo<S>> fusing_mps_info = make_shared<MRCIMPSInfo<S>>(
        hamil.n_sites, n_ext, ci_order, hamil.vacuum, target, hamil.basis);
    // shared_ptr<MPSInfo<S>> fusing_mps_info = make_shared<MPSInfo<S>>(
    //     hamil.n_sites, hamil.vacuum, target, hamil.basis);
    mpo->basis = hamil.basis;
    for (int i = 0; i < n_ext; i++)
        mpo = make_shared<FusedMPO<S>>(
            mpo, mpo->basis, mpo->n_sites - 2, mpo->n_sites - 1,
            fusing_mps_info->right_dims_fci[mpo->n_sites - 2]);
    fusing_mps_info->deallocate();
    cout << "MPO fusing end .. T = " << t.get_time() << endl;

    cout << "MPO sparsification start" << endl;
    int idx = mpo->n_sites - 1;
    for (auto &op : mpo->tensors[idx]->ops) {
        shared_ptr<CSRSparseMatrix<S>> smat = make_shared<CSRSparseMatrix<S>>();
        if (op.second->sparsity() > 0.75) {
            smat->from_dense(op.second);
            op.second->deallocate();
        } else
            smat->wrap_dense(op.second);
        op.second = smat;
    }
    mpo->sparse_form[idx] = 'S';
    mpo->tf = make_shared<TensorFunctions<S>>(
        make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
    mpo->tf->opf->seq = hamil.opf->seq;
    cout << "MPO sparsification end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S>>(mpo, make_shared<RuleQC<S>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<ubond_t> bdims = {200, 250, 300};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info =
        make_shared<MPSInfo<S>>(mpo->n_sites, hamil.vacuum, target, mpo->basis);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S>> mps = make_shared<MPS<S>>(mpo->n_sites, 0, dot);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<S>> me =
        make_shared<MovingEnvironment<S>>(mpo, mps, mps, "DMRG");
    me->init_environments(false);

    // DMRG
    shared_ptr<DMRG<S>> dmrg = make_shared<DMRG<S>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->decomp_type = dt;
    dmrg->noise_type = nt;
    dmrg->decomp_last_site = dcl;
    double ener = dmrg->solve(5, mps->center == 0, 1E-8);

    cout << "== " << name << " ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << ener << " error = " << scientific
         << setprecision(3) << setw(10) << (ener - energy) << " T = " << fixed
         << setw(10) << setprecision(3) << t.get_time() << endl;

    EXPECT_LT(abs(energy - energy), 1E-7);
    mpo->deallocate();

    // MPO2 construction (Full space DMRG)
    cout << "MPO2 start" << endl;
    shared_ptr<MPO<S>> mpo2 =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional, hamil.n_sites / 3);
    cout << "MPO2 end .. T = " << t.get_time() << endl;

    cout << "MPO2 fusing start" << endl;
    fusing_mps_info = make_shared<MPSInfo<S>>(hamil.n_sites, hamil.vacuum,
                                              target, hamil.basis);

    mpo2->basis = hamil.basis;
    for (int i = 0; i < n_ext; i++)
        mpo2 = make_shared<FusedMPO<S>>(
            mpo2, mpo2->basis, mpo2->n_sites - 2, mpo2->n_sites - 1,
            fusing_mps_info->right_dims_fci[mpo2->n_sites - 2]);
    fusing_mps_info->deallocate();
    cout << "MPO2 fusing end .. T = " << t.get_time() << endl;

    cout << "MPO2 sparsification start" << endl;
    idx = mpo2->n_sites - 1;
    for (auto &op : mpo2->tensors[idx]->ops) {
        shared_ptr<CSRSparseMatrix<S>> smat = make_shared<CSRSparseMatrix<S>>();
        if (op.second->sparsity() > 0.75) {
            smat->from_dense(op.second);
            op.second->deallocate();
        } else
            smat->wrap_dense(op.second);
        op.second = smat;
    }
    mpo2->sparse_form[idx] = 'S';
    mpo2->tf = make_shared<TensorFunctions<S>>(
        make_shared<CSROperatorFunctions<S>>(hamil.opf->cg));
    mpo2->tf->opf->seq = hamil.opf->seq;
    cout << "MPO2 sparsification end .. T = " << t.get_time() << endl;

    // MPO2 simplification
    cout << "MPO2 simplification start" << endl;
    mpo2 = make_shared<SimplifiedMPO<S>>(mpo2, make_shared<RuleQC<S>>(), true);
    cout << "MPO2 simplification end .. T = " << t.get_time() << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S>> impo = make_shared<IdentityMPO<S>>(
        mpo2->basis, mpo->basis, hamil.vacuum, hamil.opf);
    impo = make_shared<SimplifiedMPO<S>>(impo, make_shared<Rule<S>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim2 = 270;
    vector<ubond_t> bdims2 = {270, 350, 400};
    vector<ubond_t> bdims1 = {300};

    shared_ptr<MPSInfo<S>> mps_info2 = make_shared<MPSInfo<S>>(
        mpo2->n_sites, hamil.vacuum, target, mpo2->basis);
    mps_info2->set_bond_dimension(bond_dim2);
    mps_info2->tag = "BRA";

    if (mps->center == mps->n_sites - 1)
        mps->center--;
    shared_ptr<MPS<S>> mps2 =
        make_shared<MPS<S>>(mpo2->n_sites, mps->center, dot);
    mps2->initialize(mps_info2);
    mps2->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps2->save_mutable();
    mps2->deallocate();
    mps_info2->save_mutable();
    mps_info2->deallocate_mutable();

    // Identity ME
    shared_ptr<MovingEnvironment<S>> ime =
        make_shared<MovingEnvironment<S>>(impo, mps2, mps, "COMPRESS");
    ime->dot = 2;
    ime->init_environments();

    // Compress
    shared_ptr<Compress<S>> cps = make_shared<Compress<S>>(ime, bdims2, bdims1);
    cps->iprint = 2;
    cps->decomp_type = dt;
    cps->decomp_last_site = dcl;
    double norm = cps->solve(5, mps->center == 0);

    // ME2
    shared_ptr<MovingEnvironment<S>> me2 =
        make_shared<MovingEnvironment<S>>(mpo2, mps2, mps2, "DMRG");
    me2->init_environments(false);

    // DMRG2
    shared_ptr<DMRG<S>> dmrg2 = make_shared<DMRG<S>>(me2, bdims2, noises);
    dmrg2->iprint = 2;
    dmrg2->decomp_type = dt;
    dmrg2->noise_type = nt;
    dmrg2->decomp_last_site = dcl;
    double ener2 = dmrg2->solve(5, mps2->center == 0, 1E-8);

    // deallocate persistent stack memory
    mps_info2->deallocate();
    mps_info->deallocate();
    mpo2->deallocate();
    impo->deallocate();
    mpo->deallocate();
}

TEST_F(TestFITN2631G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.CAS.6-31G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), 0, 0);
    double energy = 0.1;

    HamiltonianQC<SU2> hamil(vacuum, fcidump->n_sites(), orbsym, fcidump);

    test_dmrg<SU2>(5, 2, target, energy, hamil, "SU2", DecompositionTypes::SVD,
                   NoiseTypes::Perturbative);

    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestFITN2631G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.CAS.6-31G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), 0, 0);
    double energy = 0.1;

    HamiltonianQC<SZ> hamil(vacuum, fcidump->n_sites(), orbsym, fcidump);

    test_dmrg<SZ>(5, 2, target, energy, hamil, "SZ", DecompositionTypes::SVD,
                  NoiseTypes::Perturbative);

    hamil.deallocate();
    fcidump->deallocate();
}
