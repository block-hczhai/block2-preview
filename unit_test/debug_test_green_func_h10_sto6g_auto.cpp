
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

template <typename FL>
class TestRTEGreenFunctionH10STO6G : public ::testing::Test {
  protected:
    size_t isize = 1L << 24;
    size_t dsize = 1L << 32;
    typedef typename GMatrix<FL>::FP FP;

    template <typename S>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil,
                   const string &name, int dot);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 8, 8,
            8);
        threading_()->seq_type = SeqTypes::Simple;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

template <typename FL>
template <typename S>
void TestRTEGreenFunctionH10STO6G<FL>::test_dmrg(
    S target, const shared_ptr<HamiltonianQC<S, FL>> &hamil, const string &name,
    int dot) {

    FP igf_std = -0.2286598562666365;
    FP energy_std = -5.424385375684663;

    shared_ptr<GeneralHamiltonian<S, FL>> gham =
        make_shared<GeneralHamiltonian<S, FL>>(hamil->vacuum, hamil->n_sites,
                                               hamil->orb_sym);
    hamil->fcidump->symmetrize(hamil->orb_sym);
    shared_ptr<GeneralFCIDUMP<FL>> gfd =
        GeneralFCIDUMP<FL>::initialize_from_qc(
            hamil->fcidump,
            is_same<S, SU2>::value
                ? ElemOpTypes::SU2
                : (is_same<S, SZ>::value ? ElemOpTypes::SZ : ElemOpTypes::SGF))
            ->adjust_order();

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S, FL>> mpo = make_shared<GeneralMPO<S, FL>>(
        gham, gfd, MPOAlgorithmTypes::FastBipartite, 1E-7, -1);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S, FL>>(mpo, make_shared<RuleQC<S, FL>>(),
                                            true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    cout << "C/D MPO start" << endl;
    bool su2 = S(1, 1, 0).multiplicity() == 2;
    shared_ptr<OpElement<S, FL>> c_op, d_op;
    uint16_t isite = 5;
    if (su2) {
        c_op = make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex({isite}, {}),
                                             S(1, 1, hamil->orb_sym[isite]));
        d_op = make_shared<OpElement<S, FL>>(OpNames::D, SiteIndex({isite}, {}),
                                             S(-1, 1, hamil->orb_sym[isite]));
        igf_std *= -sqrt(2);
    } else {
        c_op =
            make_shared<OpElement<S, FL>>(OpNames::C, SiteIndex({isite}, {0}),
                                          S(1, 1, hamil->orb_sym[isite]));
        d_op =
            make_shared<OpElement<S, FL>>(OpNames::D, SiteIndex({isite}, {0}),
                                          S(-1, -1, hamil->orb_sym[isite]));
    }
    shared_ptr<MPO<S, FL>> cmpo = make_shared<SiteMPO<S, FL>>(hamil, c_op);
    shared_ptr<MPO<S, FL>> dmpo = make_shared<SiteMPO<S, FL>>(hamil, d_op);
    cout << "C/D MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "C/D MPO simplification (no transpose) start" << endl;
    cmpo = make_shared<SimplifiedMPO<S, FL>>(
        cmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    dmpo = make_shared<SimplifiedMPO<S, FL>>(
        dmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
        true);
    cout << "C/D MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S, FL>> impo = make_shared<IdentityMPO<S, FL>>(hamil);
    impo = make_shared<SimplifiedMPO<S, FL>>(impo, make_shared<Rule<S, FL>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    // LMPO construction (no transpose)
    cout << "LMPO start" << endl;
    shared_ptr<MPO<S, FL>> lmpo = make_shared<GeneralMPO<S, FL>>(
        gham, gfd, MPOAlgorithmTypes::FastBipartite, 1E-7, -1);
    cout << "LMPO end .. T = " << t.get_time() << endl;

    // LMPO simplification (no transpose)
    cout << "LMPO simplification start" << endl;
    // lmpo = make_shared<SimplifiedMPO<S, FL>>(
    //     lmpo, make_shared<NoTransposeRule<S, FL>>(make_shared<RuleQC<S, FL>>()),
    //     true);
    cout << "LMPO simplification end .. T = " << t.get_time() << endl;

    ubond_t ket_bond_dim = 500, bra_bond_dim = 750;
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = {ket_bond_dim};
    vector<FP> noises = {1E-6, 1E-8, 1E-10, 0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target, hamil->basis);
    mps_info->set_bond_dimension(ket_bond_dim);
    mps_info->tag = "KET";

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S, FL>> mps =
        make_shared<MPS<S, FL>>(hamil->n_sites, 0, dot);
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
        make_shared<DMRG<S, FL, FL>>(me, ket_bdims, noises);
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    FP energy = dmrg->solve(20, mps->center == 0, 1E-12);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    // 1-site can be unstable
    EXPECT_LT(abs(energy - energy_std), dot == 1 ? 1E-4 : 1E-7);

    // D APPLY MPS
    shared_ptr<MPSInfo<S>> dmps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target + d_op->q_label, hamil->basis);
    dmps_info->set_bond_dimension(bra_bond_dim);
    dmps_info->tag = "DBRA";

    shared_ptr<MPS<S, FL>> dmps =
        make_shared<MPS<S, FL>>(hamil->n_sites, mps->center, dot);
    dmps->initialize(dmps_info);
    dmps->random_canonicalize();

    // MPS/MPSInfo save mutable
    dmps->save_mutable();
    dmps->deallocate();
    dmps_info->save_mutable();
    dmps_info->deallocate_mutable();

    // D APPLY ME
    shared_ptr<MovingEnvironment<S, FL, FL>> dme =
        make_shared<MovingEnvironment<S, FL, FL>>(dmpo, dmps, mps, "CPS-D");
    dme->init_environments();

    // LEFT ME
    shared_ptr<MovingEnvironment<S, FL, FL>> llme =
        make_shared<MovingEnvironment<S, FL, FL>>(lmpo, dmps, dmps, "LLHS");
    llme->init_environments();

    // Compression
    shared_ptr<Linear<S, FL, FL>> cps =
        make_shared<Linear<S, FL, FL>>(llme, dme, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::ReducedPerturbative;
    cps->decomp_type = DecompositionTypes::SVD;
    cps->eq_type = EquationTypes::PerturbativeCompression;
    FL norm = cps->solve(20, mps->center == 0, 1E-12);

    // Y MPS
    shared_ptr<MPSInfo<S>> ymps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target + d_op->q_label, hamil->basis);
    ymps_info->set_bond_dimension(bra_bond_dim);
    ymps_info->tag = "YBRA";

    shared_ptr<MPS<S, FL>> ymps =
        make_shared<MPS<S, FL>>(hamil->n_sites, mps->center, dot);
    ymps->initialize(ymps_info);
    ymps->random_canonicalize();

    // MPS/MPSInfo save mutable
    ymps->save_mutable();
    ymps->deallocate();
    ymps_info->save_mutable();
    ymps_info->deallocate_mutable();

    FL eta = 0.05, omega = -0.17;

    // LEFT ME
    shared_ptr<MovingEnvironment<S, FL, FL>> lme =
        make_shared<MovingEnvironment<S, FL, FL>>(lmpo, ymps, ymps, "LHS");
    lmpo->const_e -= energy;
    lme->init_environments();

    // RIGHT (identity) ME
    shared_ptr<MovingEnvironment<S, FL, FL>> rme =
        make_shared<MovingEnvironment<S, FL, FL>>(impo, ymps, dmps, "RHS");
    rme->init_environments();

    // TARGET ME
    shared_ptr<MovingEnvironment<S, FL, FL>> tme =
        make_shared<MovingEnvironment<S, FL, FL>>(cmpo, mps, ymps, "TARGET");
    tme->init_environments();

    // Linear
    shared_ptr<Linear<S, FL, FL>> linear = make_shared<Linear<S, FL, FL>>(
        lme, rme, tme, bra_bdims, bra_bdims, noises);
    linear->eq_type = EquationTypes::GreensFunction;
    linear->gf_eta = eta;
    linear->gf_omega = omega;
    linear->linear_use_precondition = true;
    linear->noise_type = NoiseTypes::ReducedPerturbative;
    linear->decomp_type = DecompositionTypes::SVD;
    linear->right_weight = 0.2;
    linear->iprint = 2;
    FL igf = linear->solve(20, ymps->center == 0, 1E-12);
    igf = linear->targets.back().back();
    if (!is_same<FL, FP>::value)
        igf = ximag<FL>(igf);

    cout << "== " << name << " (IGF) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << igf
         << " error = " << scientific << setprecision(3) << setw(10)
         << (igf - igf_std) << " T = " << fixed << setw(10) << setprecision(3)
         << t.get_time() << endl;

    EXPECT_LT(abs(igf - igf_std), 1E-4);

    dmps_info->deallocate();
    mps_info->deallocate();
    dmpo->deallocate();
    mpo->deallocate();
}

#ifdef _USE_COMPLEX
typedef ::testing::Types<complex<double>, double> TestFL;
#else
typedef ::testing::Types<double> TestFL;
#endif

TYPED_TEST_CASE(TestRTEGreenFunctionH10STO6G, TestFL);

TYPED_TEST(TestRTEGreenFunctionH10STO6G, TestSU2) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2, FL>> hamil =
        make_shared<HamiltonianQC<SU2, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SU2>(target, hamil, "SU2/2-site", 2);
    this->template test_dmrg<SU2>(target, hamil, "SU2/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}

TYPED_TEST(TestRTEGreenFunctionH10STO6G, TestSZ) {
    using FL = TypeParam;

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ, FL>> hamil =
        make_shared<HamiltonianQC<SZ, FL>>(vacuum, norb, orbsym, fcidump);

    this->template test_dmrg<SZ>(target, hamil, "SZ/2-site", 2);
    this->template test_dmrg<SZ>(target, hamil, "SZ/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}
