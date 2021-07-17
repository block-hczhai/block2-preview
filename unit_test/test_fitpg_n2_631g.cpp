
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestFITPGN2631G : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;

    template <typename S>
    void test_dmrg(int n_ext, int ci_order, const S target, double energy,
                   const shared_ptr<HamiltonianQC<S>> &hamil,
                   const shared_ptr<HamiltonianQC<S>> &hamil_red,
                   const string &name, DecompositionTypes dt, NoiseTypes nt);
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodexx");
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

template <typename S>
void TestFITPGN2631G::test_dmrg(int n_ext, int ci_order, const S target,
                                double energy,
                                const shared_ptr<HamiltonianQC<S>> &hamil,
                                const shared_ptr<HamiltonianQC<S>> &hamil_red,
                                const string &name, DecompositionTypes dt,
                                NoiseTypes nt) {

    bool dcl = false;
    int dot = 2;

    Timer t;
    t.get_time();

    // MPO construction (MRCISD-DMRG)
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional, hamil->n_sites / 3);
    cout << "MPO end .. T = " << t.get_time() << endl;

    cout << "MPO fusing start" << endl;
    shared_ptr<MPSInfo<S>> fusing_mps_info =
        n_ext != 0
            ? make_shared<MRCIMPSInfo<S>>(hamil->n_sites, n_ext, ci_order,
                                          hamil->vacuum, target, hamil->basis)
            : make_shared<MPSInfo<S>>(hamil->n_sites, hamil->vacuum, target,
                                      hamil->basis);
    mpo->basis = hamil->basis;
    for (int i = 0; i < n_ext; i++)
        mpo = make_shared<FusedMPO<S>>(
            mpo, mpo->basis, mpo->n_sites - 2, mpo->n_sites - 1,
            fusing_mps_info->right_dims_fci[mpo->n_sites - 2]);
    fusing_mps_info->deallocate();
    cout << "MPO fusing end .. T = " << t.get_time() << endl;

    cout << "MPO sparsification start" << endl;
    int idx = mpo->n_sites - 1;
    if (n_ext != 0) {
        for (auto &op : mpo->tensors[idx]->ops) {
            shared_ptr<CSRSparseMatrix<S>> smat =
                make_shared<CSRSparseMatrix<S>>();
            if (op.second->sparsity() > 0.75) {
                smat->from_dense(op.second);
                op.second->deallocate();
            } else
                smat->wrap_dense(op.second);
            op.second = smat;
        }
        mpo->sparse_form[idx] = 'S';
        mpo->tf = make_shared<TensorFunctions<S>>(
            make_shared<CSROperatorFunctions<S>>(hamil->opf->cg));
    }
    cout << "MPO sparsification end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S>>(mpo, make_shared<RuleQC<S>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    ubond_t bond_dim = 200;
    vector<ubond_t> bdims = {200, 250, 300};
    vector<ubond_t> bdims_trans = {300};
    vector<double> noises = {1E-8, 1E-9, 0.0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        mpo->n_sites, hamil->vacuum, target, mpo->basis);
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
    me->init_environments(true);

    // DMRG
    shared_ptr<DMRG<S>> dmrg = make_shared<DMRG<S>>(me, bdims, noises);
    dmrg->iprint = 2;
    dmrg->decomp_type = dt;
    dmrg->noise_type = nt;
    dmrg->decomp_last_site = dcl;
    double ener = dmrg->solve(10, mps->center == 0, 1E-8);

    cout << "== " << name << " ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << ener << " error = " << scientific
         << setprecision(3) << setw(10) << (ener - energy) << " T = " << fixed
         << setw(10) << setprecision(3) << t.get_time() << endl;

    // MPO RED construction (MRCISD-DMRG)
    cout << "MPO RED start" << endl;
    shared_ptr<MPO<S>> mpo_red = make_shared<MPOQC<S>>(
        hamil_red, QCTypes::Conventional, hamil_red->n_sites / 3);
    cout << "MPO RED end .. T = " << t.get_time() << endl;

    cout << "MPO RED fusing start" << endl;
    shared_ptr<MPSInfo<S>> fusing_mps_info_red =
        n_ext != 0
            ? make_shared<MRCIMPSInfo<S>>(hamil_red->n_sites, n_ext, ci_order,
                                          hamil_red->vacuum, target,
                                          hamil_red->basis)
            : make_shared<MPSInfo<S>>(hamil_red->n_sites, hamil_red->vacuum,
                                      target, hamil_red->basis);
    mpo_red->basis = hamil_red->basis;
    for (int i = 0; i < n_ext; i++)
        mpo_red = make_shared<FusedMPO<S>>(
            mpo_red, mpo_red->basis, mpo_red->n_sites - 2, mpo_red->n_sites - 1,
            fusing_mps_info_red->right_dims_fci[mpo_red->n_sites - 2]);
    fusing_mps_info_red->deallocate();
    cout << "MPO RED fusing end .. T = " << t.get_time() << endl;

    cout << "MPO RED sparsification start" << endl;
    if (n_ext != 0) {
        for (auto &op : mpo_red->tensors[idx]->ops) {
            shared_ptr<CSRSparseMatrix<S>> smat =
                make_shared<CSRSparseMatrix<S>>();
            if (op.second->sparsity() > 0.75) {
                smat->from_dense(op.second);
                op.second->deallocate();
            } else
                smat->wrap_dense(op.second);
            op.second = smat;
        }
        mpo_red->sparse_form[idx] = 'S';
        mpo_red->tf = make_shared<TensorFunctions<S>>(
            make_shared<CSROperatorFunctions<S>>(hamil_red->opf->cg));
    }
    cout << "MPO RED sparsification end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO RED simplification start" << endl;
    mpo_red =
        make_shared<SimplifiedMPO<S>>(mpo_red, make_shared<RuleQC<S>>(), true);
    cout << "MPO RED simplification end .. T = " << t.get_time() << endl;

    shared_ptr<MPSInfo<S>> mps_info_red = make_shared<MPSInfo<S>>(
        mpo_red->n_sites, hamil_red->vacuum, target, mpo_red->basis);
    mps_info_red->tag = "KETR";
    mps_info_red->set_bond_dimension(bond_dim);

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S>> mps_red =
        make_shared<MPS<S>>(mpo_red->n_sites, mps->center, dot);
    mps_red->initialize(mps_info_red);
    mps_red->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps_red->save_mutable();
    mps_red->deallocate();
    mps_info_red->save_mutable();
    mps_info_red->deallocate_mutable();

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S>> impo_red = make_shared<IdentityMPO<S>>(
        mpo_red->basis, mpo->basis, hamil->vacuum, hamil->vacuum, hamil->opf,
        hamil_red->orb_sym, hamil->orb_sym);
    // Attention: use trivial Rule or NoTransposeRule(RuleQC)
    impo_red = make_shared<SimplifiedMPO<S>>(impo_red, make_shared<Rule<S>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    // ME
    shared_ptr<MovingEnvironment<S>> me_trans =
        make_shared<MovingEnvironment<S>>(impo_red, mps_red, mps, "DMRG");
    me_trans->init_environments(true);

    // Linear
    shared_ptr<Linear<S>> cps_trans =
        make_shared<Linear<S>>(me_trans, bdims_trans, bdims_trans);
    cps_trans->iprint = 2;
    cps_trans->decomp_type = dt;
    cps_trans->decomp_last_site = dcl;
    double norm_trans = cps_trans->solve(20, mps->center == 0);

    cout << "OVERLAP = " << setprecision(10) << fixed << norm_trans << endl;

    // ME
    shared_ptr<MovingEnvironment<S>> me_red =
        make_shared<MovingEnvironment<S>>(mpo_red, mps_red, mps_red, "DMRG");
    me_red->init_environments(true);

    // DMRG
    shared_ptr<DMRG<S>> dmrg_red =
        make_shared<DMRG<S>>(me_red, bdims_trans, noises);
    dmrg_red->iprint = 2;
    dmrg_red->decomp_type = dt;
    dmrg_red->noise_type = nt;
    dmrg_red->decomp_last_site = dcl;
    double ener_red = dmrg_red->solve(20, mps_red->center == 0, 1E-8);

    cout << "== " << name << " ==" << setw(20) << target << " E = " << fixed
         << setw(22) << setprecision(12) << ener_red
         << " error = " << scientific << setprecision(3) << setw(10)
         << (ener_red - energy) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    impo_red->deallocate();
    mpo_red->deallocate();
    mpo->deallocate();
}

TEST_F(TestFITPGN2631G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.CAS.6-31G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    vector<uint8_t> orbsym_red = orbsym;
    for (auto &ipg : orbsym_red)
        ipg = (ipg & 4) >> 1;

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), 0, 0);
    double energy = 0.1;

    shared_ptr<HamiltonianQC<SU2>> hamil = make_shared<HamiltonianQC<SU2>>(
        vacuum, fcidump->n_sites(), orbsym, fcidump);
    shared_ptr<HamiltonianQC<SU2>> hamil_red = make_shared<HamiltonianQC<SU2>>(
        vacuum, fcidump->n_sites(), orbsym_red, fcidump);

    test_dmrg<SU2>(0, 2, target, energy, hamil, hamil_red, "SU2",
                   DecompositionTypes::SVD, NoiseTypes::Perturbative);

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestFITPGN2631G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/N2.CAS.6-31G.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    vector<uint8_t> orbsym_red = orbsym;
    for (auto &ipg : orbsym_red)
        ipg = (ipg & 4) >> 1;

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), 0, 0);
    double energy = 0.1;

    shared_ptr<HamiltonianQC<SZ>> hamil = make_shared<HamiltonianQC<SZ>>(
        vacuum, fcidump->n_sites(), orbsym, fcidump);
    shared_ptr<HamiltonianQC<SZ>> hamil_red = make_shared<HamiltonianQC<SZ>>(
        vacuum, fcidump->n_sites(), orbsym_red, fcidump);

    test_dmrg<SZ>(0, 2, target, energy, hamil, hamil_red, "SZ",
                  DecompositionTypes::SVD, NoiseTypes::Perturbative);

    hamil->deallocate();
    fcidump->deallocate();
}
