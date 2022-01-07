
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestFITN2631G : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;

    template <typename S, typename FL>
    void test_dmrg(int n_ext, int ci_order, const S target, double energy,
                   const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodexx");
        frame_()->minimal_memory_usage = false;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            1);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

template <typename S, typename FL>
void TestFITN2631G::test_dmrg(int n_ext, int ci_order, const S target,
                              double energy,
                              const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                              const string &name, DecompositionTypes dt,
                              NoiseTypes nt) {

    bool dcl = false;
    int dot = 2;

    Timer t;
    t.get_time();
    // MPO construction (MRCISD-DMRG)
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<MPOQC<S, FL>>(
        hamil, QCTypes::Conventional, "HQC", hamil->n_sites / 3);
    cout << "MPO end .. T = " << t.get_time() << endl;

    cout << "MPO fusing start" << endl;
    shared_ptr<MPSInfo<S>> fusing_mps_info = make_shared<MRCIMPSInfo<S>>(
        hamil->n_sites, n_ext, ci_order, hamil->vacuum, target, hamil->basis);
    // shared_ptr<MPSInfo<S>> fusing_mps_info = make_shared<MPSInfo<S>>(
    //     hamil->n_sites, hamil->vacuum, target, hamil->basis);
    mpo->basis = hamil->basis;
    for (int i = 0; i < n_ext; i++)
        mpo = make_shared<FusedMPO<S, FL>>(
            mpo, mpo->basis, mpo->n_sites - 2, mpo->n_sites - 1,
            fusing_mps_info->right_dims_fci[mpo->n_sites - 2]);
    fusing_mps_info->deallocate();
    cout << "MPO fusing end .. T = " << t.get_time() << endl;

    cout << "MPO sparsification start" << endl;
    int idx = mpo->n_sites - 1;
    mpo->load_tensor(idx);
    for (auto &op : mpo->tensors[idx]->ops) {
        shared_ptr<CSRSparseMatrix<S, FL>> smat =
            make_shared<CSRSparseMatrix<S, FL>>();
        if (op.second->sparsity() > 0.75) {
            smat->from_dense(op.second);
            op.second->deallocate();
        } else
            smat->wrap_dense(op.second);
        op.second = smat;
    }
    mpo->save_tensor(idx);
    mpo->unload_tensor(idx);
    mpo->sparse_form[idx] = 'S';
    mpo->tf = make_shared<TensorFunctions<S, FL>>(
        make_shared<CSROperatorFunctions<S, FL>>(hamil->opf->cg));
    mpo->tf->opf->seq = hamil->opf->seq;
    cout << "MPO sparsification end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<ubond_t> bdims = {200, 250, 300};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        mpo->n_sites, hamil->vacuum, target, mpo->basis);
    mps_info->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S, FL>> mps = make_shared<MPS<S, FL>>(mpo->n_sites, 0, dot);
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    // ME
    shared_ptr<MovingEnvironment<S, FL, FL>> me =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, mps, mps, "DMRG");
    me->init_environments(false);

    // DMRG
    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
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

    // MPO2 construction (Full space DMRG)
    cout << "MPO2 start" << endl;
    shared_ptr<MPO<S, FL>> mpo2 = make_shared<MPOQC<S, FL>>(
        hamil, QCTypes::Conventional, "HQC2", hamil->n_sites / 3);
    cout << "MPO2 end .. T = " << t.get_time() << endl;

    cout << "MPO2 fusing start" << endl;
    fusing_mps_info = make_shared<MPSInfo<S>>(hamil->n_sites, hamil->vacuum,
                                              target, hamil->basis);

    mpo2->basis = hamil->basis;
    for (int i = 0; i < n_ext; i++)
        mpo2 = make_shared<FusedMPO<S, FL>>(
            mpo2, mpo2->basis, mpo2->n_sites - 2, mpo2->n_sites - 1,
            fusing_mps_info->right_dims_fci[mpo2->n_sites - 2]);
    fusing_mps_info->deallocate();
    cout << "MPO2 fusing end .. T = " << t.get_time() << endl;

    cout << "MPO2 sparsification start" << endl;
    idx = mpo2->n_sites - 1;
    mpo2->load_tensor(idx);
    for (auto &op : mpo2->tensors[idx]->ops) {
        shared_ptr<CSRSparseMatrix<S, FL>> smat =
            make_shared<CSRSparseMatrix<S, FL>>();
        if (op.second->sparsity() > 0.75) {
            smat->from_dense(op.second);
            op.second->deallocate();
        } else
            smat->wrap_dense(op.second);
        op.second = smat;
    }
    mpo2->save_tensor(idx);
    mpo2->unload_tensor(idx);
    mpo2->sparse_form[idx] = 'S';
    mpo2->tf = make_shared<TensorFunctions<S, FL>>(
        make_shared<CSROperatorFunctions<S, FL>>(hamil->opf->cg));
    mpo2->tf->opf->seq = hamil->opf->seq;
    cout << "MPO2 sparsification end .. T = " << t.get_time() << endl;

    // MPO2 simplification
    cout << "MPO2 simplification start" << endl;
    mpo2 = make_shared<SimplifiedMPO<S, FL>>(mpo2, make_shared<RuleQC<S, FL>>(),
                                             true);
    cout << "MPO2 simplification end .. T = " << t.get_time() << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(
        mpo2->basis, mpo->basis, hamil->vacuum, hamil->opf);
    // Attention: use trivial Rule or NoTransposeRule(RuleQC)
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    ubond_t bond_dim2 = 270, bond_dim3 = 300;
    vector<ubond_t> bdims2 = {270, 350, 400};
    vector<ubond_t> bdims1 = {300};
    vector<ubond_t> bdims3 = {300};

    shared_ptr<MPSInfo<S>> mps_info2 = make_shared<MPSInfo<S>>(
        mpo2->n_sites, hamil->vacuum, target, mpo2->basis);
    mps_info2->set_bond_dimension(bond_dim2);
    mps_info2->tag = "KET2";

    if (mps->center == mps->n_sites - 1)
        mps->center--;
    shared_ptr<MPS<S, FL>> mps2 =
        make_shared<MPS<S, FL>>(mpo2->n_sites, mps->center, dot);
    mps2->initialize(mps_info2);
    mps2->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps2->save_mutable();
    mps2->deallocate();
    mps_info2->save_mutable();
    mps_info2->deallocate_mutable();

    // Identity ME
    shared_ptr<MovingEnvironment<S, FL, FL>> ime =
        make_shared<MovingEnvironment<S, FL, FL>>(impo, mps2, mps, "COMPRESS");
    ime->dot = 2;
    ime->init_environments();

    // Linear
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(ime, bdims2, bdims1);
    cps->iprint = 2;
    cps->decomp_type = dt;
    cps->decomp_last_site = dcl;
    double norm = cps->solve(5, mps->center == 0);

    // ME2
    shared_ptr<MovingEnvironment<S, FL, FL>> me2 =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo2, mps2, mps2, "DMRG");
    me2->init_environments(false);

    // DMRG2
    shared_ptr<DMRG<S, FL, FL>> dmrg2 =
        make_shared<DMRG<S, FL, FL>>(me2, bdims2, noises);
    dmrg2->iprint = 2;
    dmrg2->decomp_type = dt;
    dmrg2->noise_type = nt;
    dmrg2->decomp_last_site = dcl;
    double ener2 = dmrg2->solve(5, mps2->center == 0, 1E-8);

    // Now add KET1 & KET2 to BRA-ADD
    // the center of the three mpss must match
    shared_ptr<MPSInfo<S>> mps_info3 = make_shared<MPSInfo<S>>(
        mpo->n_sites, hamil->vacuum, target, mpo->basis);
    mps_info3->set_bond_dimension(bond_dim3);
    mps_info3->tag = "BRA-ADD";

    // align mps centers
    if (mps->center != mps2->center) {
        cout << "align mps centers ..." << endl;
        cout << "MPS1 = " << mps->canonical_form << endl;
        cout << "MPS2 = " << mps2->canonical_form << endl;
        assert(mps->dot == 2 && mps2->dot == 2);
        if (mps->center == 0) {
            mps2->center += 1;
            mps2->canonical_form[mps2->n_sites - 1] = 'S';
            while (mps2->center != 0)
                mps2->move_left(mpo->tf->opf->cg);
        } else {
            mps2->canonical_form[0] = 'K';
            while (mps2->center != mps2->n_sites - 1)
                mps2->move_right(mpo->tf->opf->cg);
            mps2->center -= 1;
        }
    }

    cout << "checking overlap ..." << endl;

    // Overlap
    ime = make_shared<MovingEnvironment<S, FL, FL>>(impo, mps2, mps, "IDT");
    ime->init_environments();
    shared_ptr<Expect<S, FL, FL>> ex =
        make_shared<Expect<S, FL, FL>>(ime, 400, 300);
    double overlap = ex->solve(false);
    cout << "OVERLAP = " << setprecision(10) << fixed << overlap << endl;

    shared_ptr<MPS<S, FL>> mps3 =
        make_shared<MPS<S, FL>>(mpo->n_sites, mps->center, dot);
    mps3->initialize(mps_info3);
    mps3->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps3->save_mutable();
    mps3->deallocate();
    mps_info3->save_mutable();
    mps_info3->deallocate_mutable();

    // 0.25 * Identity MPO between mps3 / mps
    shared_ptr<MPO<S, FL>> impo25 = make_shared<IdentityMPO<S, FL>>(
        mpo->basis, mpo->basis, hamil->vacuum, hamil->opf, "I25");
    impo25 =
        make_shared<SimplifiedMPO<S, FL>>(impo25, make_shared<Rule<S, FL>>());
    impo25 = 0.25 * impo25;

    // 0.75 * Identity MPO between mps3 / mps2
    shared_ptr<MPO<S, FL>> impo75 = make_shared<IdentityMPO<S, FL>>(
        mpo->basis, mpo2->basis, hamil->vacuum, hamil->opf, "I75");
    impo75 =
        make_shared<SimplifiedMPO<S, FL>>(impo75, make_shared<Rule<S, FL>>());
    impo75 = 0.75 * impo75;

    shared_ptr<MovingEnvironment<S, FL, FL>> laddme =
        make_shared<MovingEnvironment<S, FL, FL>>(impo25, mps3, mps, "ADDL");
    laddme->init_environments();
    shared_ptr<MovingEnvironment<S, FL, FL>> raddme =
        make_shared<MovingEnvironment<S, FL, FL>>(impo75, mps3, mps2, "ADDR");
    raddme->init_environments();
    shared_ptr<MovingEnvironment<S, FL, FL>> pertme =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, mps3, mps3, "PERT");
    pertme->init_environments();

    cout << "fit mps addition ..." << endl;

    // mps3 = 0.25 mps + 0.75 mps2
    // bdims3 = bond dim for mps3
    // bdims1 = bond dim for mps
    // bond_dim2 = bond dim for mps2
    // note that pertme can also be nullptr, then no perturbative noise will be
    // applied
    shared_ptr<Linear<S, FL, FL>> addmps = make_shared<Linear<S, FL, FL>>(
        pertme, laddme, raddme, bdims3, bdims1, noises);
    addmps->eq_type = EquationTypes::FitAddition;
    addmps->target_ket_bond_dim = bond_dim2;
    addmps->iprint = 2;
    addmps->decomp_type = dt;
    addmps->decomp_last_site = dcl;
    double mps3norm = addmps->solve(5, mps3->center == 0);
    // this can be affected by the relative sign of mps and mps2
    cout << "Norm of fitted MPS = " << setprecision(10) << fixed << mps3norm
         << endl;

    // deallocate persistent stack memory
    mps_info3->deallocate();
    mps_info2->deallocate();
    mps_info->deallocate();
    impo75->deallocate();
    impo25->deallocate();
    mpo2->deallocate();
    impo->deallocate();
    mpo->deallocate();
}

TEST_F(TestFITN2631G, TestSU2) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.CAS.6-31G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), 0, 0);
    double energy = 0.1;

    shared_ptr<HamiltonianQC<SU2, double>> hamil =
        make_shared<HamiltonianQC<SU2, double>>(vacuum, fcidump->n_sites(),
                                                orbsym, fcidump);

    test_dmrg<SU2, double>(5, 2, target, energy, hamil, "SU2",
                           DecompositionTypes::SVD, NoiseTypes::Perturbative);

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestFITN2631G, TestSZ) {
    shared_ptr<FCIDUMP<double>> fcidump = make_shared<FCIDUMP<double>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.CAS.6-31G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), 0, 0);
    double energy = 0.1;

    shared_ptr<HamiltonianQC<SZ, double>> hamil =
        make_shared<HamiltonianQC<SZ, double>>(vacuum, fcidump->n_sites(),
                                               orbsym, fcidump);

    test_dmrg<SZ, double>(5, 2, target, energy, hamil, "SZ",
                          DecompositionTypes::SVD, NoiseTypes::Perturbative);

    hamil->deallocate();
    fcidump->deallocate();
}
