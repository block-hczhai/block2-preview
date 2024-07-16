
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestGreenFunctionH10STO6G : public ::testing::Test {
  protected:
    size_t isize = 1LL << 28;
    size_t dsize = 1LL << 32;

    template <typename S>
    void test_dmrg(S target, const shared_ptr<HamiltonianQC<S>> &hamil, const string &name,
                   int dot);
    void SetUp() override {
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->minimal_disk_usage = true;
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 28,
            28, 1);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

template <typename S>
void TestGreenFunctionH10STO6G::test_dmrg(S target,
                                          const shared_ptr<HamiltonianQC<S>> &hamil,
                                          const string &name, int dot) {

    double igf_std = -0.2286598562666365;
    double energy_std = -5.424385375684663;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<S>> mpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<S>>(mpo, make_shared<RuleQC<S>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    cout << "C/D MPO start" << endl;
    bool su2 = S(1, 1, 0).multiplicity() == 2;
    shared_ptr<OpElement<S>> c_op, d_op;
    uint16_t isite = 4;
    if (su2) {
        c_op = make_shared<OpElement<S>>(OpNames::C, SiteIndex({isite}, {}),
                                         S(1, 1, hamil->orb_sym[isite]));
        d_op = make_shared<OpElement<S>>(OpNames::D, SiteIndex({isite}, {}),
                                         S(-1, 1, hamil->orb_sym[isite]));
        igf_std *= -sqrt(2);
    } else {
        c_op = make_shared<OpElement<S>>(OpNames::C, SiteIndex({isite}, {0}),
                                         S(1, 1, hamil->orb_sym[isite]));
        d_op = make_shared<OpElement<S>>(OpNames::D, SiteIndex({isite}, {0}),
                                         S(-1, -1, hamil->orb_sym[isite]));
    }
    shared_ptr<MPO<S>> cmpo = make_shared<SiteMPO<S>>(hamil, c_op);
    shared_ptr<MPO<S>> dmpo = make_shared<SiteMPO<S>>(hamil, d_op);
    cout << "C/D MPO end .. T = " << t.get_time() << endl;

    // MPO simplification (no transpose)
    cout << "C/D MPO simplification (no transpose) start" << endl;
    cmpo = make_shared<SimplifiedMPO<S>>(
        cmpo, make_shared<NoTransposeRule<S>>(make_shared<RuleQC<S>>()), true);
    dmpo = make_shared<SimplifiedMPO<S>>(
        dmpo, make_shared<NoTransposeRule<S>>(make_shared<RuleQC<S>>()), true);
    cout << "C/D MPO simplification (no transpose) end .. T = " << t.get_time()
         << endl;

    // Identity MPO
    cout << "Identity MPO start" << endl;
    shared_ptr<MPO<S>> impo = make_shared<IdentityMPO<S>>(hamil);
    impo = make_shared<SimplifiedMPO<S>>(impo, make_shared<Rule<S>>());
    cout << "Identity MPO end .. T = " << t.get_time() << endl;

    // LMPO construction (no transpose)
    cout << "LMPO start" << endl;
    shared_ptr<MPO<S>> lmpo =
        make_shared<MPOQC<S>>(hamil, QCTypes::Conventional);
    cout << "LMPO end .. T = " << t.get_time() << endl;

    // LMPO simplification (no transpose)
    cout << "LMPO simplification start" << endl;
    lmpo = make_shared<SimplifiedMPO<S>>(
        lmpo, make_shared<NoTransposeRule<S>>(make_shared<RuleQC<S>>()), true);
    cout << "LMPO simplification end .. T = " << t.get_time() << endl;

    ubond_t ket_bond_dim = 500, bra_bond_dim = 750;
    vector<ubond_t> bra_bdims = {bra_bond_dim}, ket_bdims = {ket_bond_dim};
    vector<double> noises = {1E-6, 1E-8, 1E-10, 0};

    t.get_time();

    shared_ptr<MPSInfo<S>> mps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target, hamil->basis);
    mps_info->set_bond_dimension(ket_bond_dim);
    mps_info->tag = "KET";

    // MPS
    Random::rand_seed(0);

    shared_ptr<MPS<S>> mps = make_shared<MPS<S>>(hamil->n_sites, 0, dot);
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
    shared_ptr<DMRG<S>> dmrg = make_shared<DMRG<S>>(me, ket_bdims, noises);
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;
    dmrg->decomp_type = DecompositionTypes::SVD;
    long double energy = dmrg->solve(20, mps->center == 0, 1E-12);

    cout << "== " << name << " (DMRG) ==" << setw(20) << target
         << " E = " << fixed << setw(22) << setprecision(12) << energy
         << " error = " << scientific << setprecision(3) << setw(10)
         << (energy - energy_std) << " T = " << fixed << setw(10)
         << setprecision(3) << t.get_time() << endl;

    EXPECT_LT(abs(energy - energy_std), 1E-7);

    // D APPLY MPS
    shared_ptr<MPSInfo<S>> dmps_info = make_shared<MPSInfo<S>>(
        hamil->n_sites, hamil->vacuum, target + d_op->q_label, hamil->basis);
    dmps_info->set_bond_dimension(bra_bond_dim);
    dmps_info->tag = "DBRA";

    shared_ptr<MPS<S>> dmps =
        make_shared<MPS<S>>(hamil->n_sites, mps->center, dot);
    dmps->initialize(dmps_info);
    dmps->random_canonicalize();

    // MPS/MPSInfo save mutable
    dmps->save_mutable();
    dmps->deallocate();
    dmps_info->save_mutable();
    dmps_info->deallocate_mutable();

    // D APPLY ME
    shared_ptr<MovingEnvironment<S>> dme =
        make_shared<MovingEnvironment<S>>(dmpo, dmps, mps, "CPS-D");
    dme->init_environments();

    // LEFT ME
    shared_ptr<MovingEnvironment<S>> llme =
        make_shared<MovingEnvironment<S>>(lmpo, dmps, dmps, "LLHS");
    llme->init_environments();

    // Compression
    shared_ptr<Linear<S>> cps =
        make_shared<Linear<S>>(llme, dme, bra_bdims, ket_bdims, noises);
    cps->noise_type = NoiseTypes::ReducedPerturbative;
    cps->decomp_type = DecompositionTypes::SVD;
    cps->eq_type = EquationTypes::PerturbativeCompression;
    double norm = cps->solve(20, mps->center == 0, 1E-12);

    // complex MPS
    shared_ptr<MultiMPS<S>> cpx_ref = MultiMPS<S>::make_complex(dmps, "CPX-R");
    shared_ptr<MultiMPS<S>> cpx_mps = MultiMPS<S>::make_complex(dmps, "CPX-D");

    double dt = 0.1;
    int n_steps = 10000;
    shared_ptr<MovingEnvironment<S>> xme =
        make_shared<MovingEnvironment<S>>(lmpo, cpx_mps, cpx_mps, "XTD");
    shared_ptr<MovingEnvironment<S>> mme =
        make_shared<MovingEnvironment<S>>(impo, cpx_ref, cpx_mps, "II");
    lmpo->const_e -= energy;
    xme->init_environments();
    shared_ptr<TimeEvolution<S>> te =
        make_shared<TimeEvolution<S>>(xme, bra_bdims, TETypes::RK4);
    te->iprint = 2;
    te->n_sub_sweeps = te->mode == TETypes::TangentSpace ? 1 : 2;
    te->normalize_mps = false;
    shared_ptr<Expect<S, complex<double>>> ex =
        make_shared<Expect<S, complex<double>>>(mme, bra_bond_dim,
                                                bra_bond_dim);
    vector<complex<double>> rtgf;
    for (int i = 0; i < n_steps; i++) {
        if (te->mode == TETypes::TangentSpace)
            te->solve(2, complex<double>(0, dt / 2), cpx_mps->center == 0);
        else
            te->solve(1, complex<double>(0, dt), cpx_mps->center == 0);
        mme->init_environments();
        complex<double> overlap = ex->solve(false);
        rtgf.push_back(overlap);
        cout << setprecision(10);
        cout << i * dt << " " << overlap << endl;
    }

    double eta = 0.005;

    vector<double> freqs(n_steps);
    FFT::fftfreq(freqs.data(), n_steps, dt);
    FFT::fftshift(freqs.data(), n_steps, true);
    MatrixFunctions::iscale(MatrixRef(freqs.data(), n_steps, 1),
                            2.0 * acos(-1));
    for (int i = 0; i < n_steps; i++)
        rtgf[i] *= complex<double>(0, -dt) * exp(-eta * dt * i);
    FFT().fft(rtgf.data(), n_steps, true);
    FFT::fftshift(rtgf.data(), n_steps, true);

    for (int i = 0; i < n_steps; i++)
        if (freqs[i] >= -0.8 && freqs[i] < -0.2)
            cout << setw(10) << setprecision(5) << freqs[i] << " "
                 << rtgf[i].imag() * (-2 / acos(-1)) << endl;

    dmps_info->deallocate();
    mps_info->deallocate();
    dmpo->deallocate();
    mpo->deallocate();
}

TEST_F(TestGreenFunctionH10STO6G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP.LOWDIN";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SU2>> hamil = make_shared<HamiltonianQC<SU2>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SU2>(target, hamil, "SU2/2-site", 2);
    test_dmrg<SU2>(target, hamil, "SU2/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}

TEST_F(TestGreenFunctionH10STO6G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H10.STO6G.R1.8.FCIDUMP.LOWDIN";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    SZ vacuum(0);
    SZ target(fcidump->n_elec(), fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));

    double energy_std = -107.654122447525;

    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<SZ>> hamil = make_shared<HamiltonianQC<SZ>>(vacuum, norb, orbsym, fcidump);

    test_dmrg<SZ>(target, hamil, "SZ/2-site", 2);
    test_dmrg<SZ>(target, hamil, "SZ/1-site", 1);

    hamil->deallocate();
    fcidump->deallocate();
}
