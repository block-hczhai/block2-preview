
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1LL << 30;
    size_t dsize = 1LL << 34;
    void SetUp() override {
        cout << "BOND INTEGER SIZE = " << sizeof(ubond_t) << endl;
        cout << "MKL INTEGER SIZE = " << sizeof(MKL_INT) << endl;
        Random::rand_seed(0);
        frame_<FP>() = make_shared<DataFrame<FP>>(isize, dsize, "nodex");
        frame_<FP>()->use_main_stack = false;
        frame_<FP>()->minimal_disk_usage = true;
        frame_<FP>()->fp_codec = make_shared<FPCodec<double>>(1E-14, 8 * 1024);
        // threading_() = make_shared<Threading>(ThreadingTypes::BatchedGEMM |
        // ThreadingTypes::Global, 8, 8);
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global, 28,
            28, 1);
        // threading_() =
        // make_shared<Threading>(ThreadingTypes::OperatorQuantaBatchedGEMM |
        // ThreadingTypes::Global, 16, 16, 16, 16);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *frame_<FP>() << endl;
        cout << *threading_() << endl;
    }
    void TearDown() override {
        frame_<FP>()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_<FP>()->used == 0);
        frame_<FP>() = nullptr;
    }
};

TEST_F(TestDMRG, Test) {
    // shared_ptr<FCIDUMP> fcidump = make_shared<CompressedFCIDUMP>(1E-13);
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    vector<double> occs;
    PGTypes pg = PGTypes::D2H;

    string occ_filename = "data/CR2.SVP.OCC";
    string pdm_filename = "data/CR2.SVP.1NPC";
    // string occ_filename = "data/CR2.SVP.HF"; // E(HF) = -2085.53318786766
    // occs = read_occ(occ_filename);
    // string filename = "data/CR2.SVP.FCIDUMP"; // E = -2086.504520308260
    // string occ_filename = "data/H2O.TZVP.OCC";
    // occs = read_occ(occ_filename);
    // string filename = "data/H2O.TZVP.FCIDUMP"; // E = -76.31676
    // pg = PGTypes::C2V;
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/H4.STO6G.R1.8.FCIDUMP";
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "INT end .. T = " << t.get_time() << endl;

    bool reorder = false;
    if (reorder) {
        Random::rand_seed(1234);
        // auto hmat = fcidump->abs_h1e_matrix();
        // auto kmat = fcidump->abs_exchange_matrix();
        // for (size_t i = 0; i < kmat.size(); i++)
        //     kmat[i] = hmat[i] * 1E-7 + kmat[i];
        auto kmat = read_occ(pdm_filename);
        auto omat = fcidump->orb_sym<uint8_t>();
        for (size_t i = 0; i < fcidump->n_sites(); i++)
            for (size_t j = 0; j < fcidump->n_sites(); j++)
                if (omat[i] != omat[j])
                    kmat[i * fcidump->n_sites() + j] =
                        abs(kmat[i * fcidump->n_sites() + j] -
                            occs[i] * occs[j]) *
                        1E-7;
                else
                    kmat[i * fcidump->n_sites() + j] = abs(
                        kmat[i * fcidump->n_sites() + j] - occs[i] * occs[j]);
        // double bias = 1;
        // for (auto &kk : kmat)
        //     if (kk > 1)
        //         kk = 1 + pow(kk - 1, bias);
        //     else if (kk < 1)
        //         kk = 1 - pow(1 - kk, bias);
        int ntasks = 1;
        int n_generations = 12000;
        int n_configs = 100;
        int n_elite = 8;
        vector<uint16_t> mx;
        double mf = 0;
        for (int i = 0; i < ntasks; i++) {
            vector<uint16_t> x = OrbitalOrdering::ga_opt(
                fcidump->n_sites(), kmat, n_generations, n_configs, n_elite);
            double f = OrbitalOrdering::evaluate(fcidump->n_sites(), kmat, x);
            cout << setw(4) << i << " = " << setw(20) << setprecision(10) << f
                 << endl;
            if (mf == 0 || f < mf)
                mf = f, mx = x;
        }
        cout << "BEST = " << setw(20) << setprecision(10) << mf << endl;
        for (int i = 0; i < fcidump->n_sites(); i++)
            cout << setw(4) << mx[i];
        cout << endl;
        fcidump->reorder(mx);
        occs = fcidump->reorder(occs, mx);
    }
    Random::rand_seed(1234);

    cout << "ORB SYM = ";
    for (int i = 0; i < fcidump->n_sites(); i++)
        cout << setw(2) << (int)fcidump->orb_sym<uint8_t>()[i];
    cout << endl;

    vector<uint8_t> ioccs;
    for (auto x : occs)
        ioccs.push_back(uint8_t(x));

    // cout << "HF energy = " << fixed << setprecision(12)
    //      << fcidump->det_energy(ioccs, 0, fcidump->n_sites()) + fcidump->e
    //      << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    shared_ptr<HamiltonianQC<SU2>> hamil =
        make_shared<HamiltonianQC<SU2>>(vacuum, norb, orbsym, fcidump);

    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SU2>> mpo =
        make_shared<MPOQC<SU2>>(hamil, QCTypes::Conventional);
    // mpo = make_shared<ArchivedMPO<SU2>>(mpo);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SU2>>(
        mpo, make_shared<RuleQC<SU2>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    ubond_t bond_dim = 200;

    // MPSInfo
    // shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
    //     norb, vacuum, target, hamil->basis);

    // CCSD init
    shared_ptr<MPSInfo<SU2>> mps_info =
        make_shared<MPSInfo<SU2>>(norb, vacuum, target, hamil->basis);
    // mps_info->set_bond_dimension_full_fci();
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
    //     make_shared<DynamicMPSInfo<SU2>>(norb, vacuum, target, hamil->basis,
    //                                      hamil->orb_sym, ioccs);
    // mps_info->n_local = 4;
    // mps_info->set_bond_dimension(bond_dim);

    // Determinant init
    // shared_ptr<DeterminantMPSInfo<SU2>> mps_info =
    //     make_shared<DeterminantMPSInfo<SU2>>(norb, vacuum, target,
    //     hamil->basis,
    //                                      hamil->orb_sym, ioccs, fcidump);
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
    auto st = mps->estimate_storage(mps_info);
    cout << "MPS memory = " << Parsing::to_size_string(st[0])
         << "; storage = " << Parsing::to_size_string(st[1]) << endl;
    auto st2 = mpo->estimate_storage(mps_info, 2);
    cout << "2-site MPO term = " << Parsing::to_size_string(st2[0])
         << "; memory = " << Parsing::to_size_string(st2[1])
         << "; storage = " << Parsing::to_size_string(st2[2]) << endl;
    auto st3 = mpo->estimate_storage(mps_info, 1);
    cout << "1-site MPO term = " << Parsing::to_size_string(st3[0])
         << "; memory = " << Parsing::to_size_string(st3[1])
         << "; storage = " << Parsing::to_size_string(st3[2]) << endl;
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
    shared_ptr<MovingEnvironment<SU2>> me =
        make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(false);
    cout << "INIT end .. T = " << t.get_time() << endl;

    cout << *frame_<FP>() << endl;
    frame_<FP>()->activate(0);

    // DMRG
    // vector<ubond_t> bdims = {50};
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5,
                             1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6};
    vector<double> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6,
                              1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                              5E-7, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    // noises = vector<double>{1E-5};
    // vector<double> noises = {1E-6};
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->cached_contraction = true;
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    // dmrg->cutoff = 1E-20;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollectedLowMem;
    dmrg->solve(30, true);

    shared_ptr<MPO<SU2>> pmpo = make_shared<PDM1MPOQC<SU2>>(hamil);
    pmpo =
        make_shared<SimplifiedMPO<SU2>>(pmpo, make_shared<RuleQC<SU2>>(), true);
    shared_ptr<MovingEnvironment<SU2>> pme =
        make_shared<MovingEnvironment<SU2>>(pmpo, mps, mps, "1PDM");
    pme->init_environments(true);
    shared_ptr<Expect<SU2>> expect = make_shared<Expect<SU2>>(pme, 750, 750);
    expect->solve(true, mps->center == 0);
    MatrixRef dm = expect->get_1pdm_spatial();
    for (int i = 0; i < dm.m; i++)
        cout << setw(12) << setprecision(8) << (int)orbsym[i] << " ";
    cout << endl;
    for (int i = 0; i < dm.m; i++)
        cout << setw(12) << setprecision(8) << dm(i, i) << " ";
    cout << endl;
    DiagonalMatrix w(nullptr, dm.m);
    w.allocate();
    MatrixFunctions::block_eigs(dm, w, orbsym);
    for (int i = 0; i < w.m; i++)
        cout << setw(12) << setprecision(8) << w(i, i) << " ";
    cout << endl;
    MatrixRef rot(nullptr, dm.m, dm.n);
    rot.allocate();
    MatrixFunctions::transpose(dm, rot);
    double rot_det = MatrixFunctions::det(rot);
    cout << "DET = " << setw(12) << setprecision(8) << rot_det << endl;
    assert(rot_det > 0);
    cout << endl;
    for (int i = 0; i < w.m; i++) {
        for (int j = 0; j < w.n; j++)
            cout << setw(12) << setprecision(8) << dm(i, j) << " ";
        cout << endl;
    }
    vector<double> vrot(rot.data, rot.data + rot.size());
    rot.deallocate();
    w.deallocate();
    dm.deallocate();

    fcidump->rotate(vrot);
    double error = fcidump->symmetrize(orbsym);
    cout << "ROT SYM ERROR = " << setw(12) << setprecision(4) << scientific
         << error << endl;

    ComplexMatrixRef ckappa(nullptr, dm.m, dm.n);
    ckappa.allocate();
    ckappa.clear();
    ComplexMatrixFunctions::fill_complex(ckappa,
                                         MatrixRef(vrot.data(), dm.m, dm.n),
                                         MatrixRef(nullptr, dm.m, dm.n));
    ComplexMatrixFunctions::logarithm(ckappa);
    vector<double> kappa(ckappa.size()), kappa_im(ckappa.size());
    ComplexMatrixFunctions::extract_complex(
        ckappa, MatrixRef(kappa.data(), dm.m, dm.n),
        MatrixRef(kappa_im.data(), dm.m, dm.n));
    ckappa.deallocate();
    double im_norm =
        MatrixFunctions::norm(MatrixRef(kappa_im.data(), dm.m, dm.n));
    cout << "KAPPA IMAG NORM = " << setw(12) << setprecision(4) << scientific
         << im_norm << endl;
    assert(im_norm < 1E-12);
    kappa_im = kappa;
    MatrixFunctions::transpose(MatrixRef(kappa_im.data(), dm.m, dm.n),
                               MatrixRef(kappa.data(), dm.m, dm.n));

    shared_ptr<FCIDUMP> fcidump_rot = make_shared<FCIDUMP>();
    fcidump_rot->initialize_h1e(norb, fcidump->n_elec(), fcidump->twos(),
                                fcidump->isym(), 0.0, kappa.data(),
                                kappa.size());
    shared_ptr<HamiltonianQC<SU2>> hamil_rot =
        make_shared<HamiltonianQC<SU2>>(vacuum, norb, orbsym, fcidump_rot);

    // MPO construction
    shared_ptr<MPO<SU2>> mpo_rot =
        make_shared<MPOQC<SU2>>(hamil_rot, QCTypes::NC);
    mpo_rot = make_shared<SimplifiedMPO<SU2>>(
        mpo_rot,
        make_shared<AntiHermitianRuleQC<SU2>>(make_shared<RuleQC<SU2>>()),
        true);

    double dt = 0.02;
    int n_steps = (int)(1.0 / dt + 0.1);
    shared_ptr<MovingEnvironment<SU2>> rme =
        make_shared<MovingEnvironment<SU2>>(mpo_rot, mps, mps, "ROT");
    rme->init_environments();
    shared_ptr<TimeEvolution<SU2>> te =
        make_shared<TimeEvolution<SU2>>(rme, bdims, TETypes::RK4);
    te->hermitian = false;
    te->iprint = 2;
    te->n_sub_sweeps = te->mode == TETypes::TangentSpace ? 1 : 2;
    te->normalize_mps = false;
    for (int i = 0; i < n_steps; i++) {
        if (te->mode == TETypes::TangentSpace)
            te->solve(2, -dt / 2, mps->center == 0);
        else
            te->solve(1, -dt, mps->center == 0);
        cout << setprecision(12);
        cout << i * dt << " " << te->energies.back() << " "
             << te->normsqs.back() << endl;
    }

    shared_ptr<HamiltonianQC<SU2>> hamil2 =
        make_shared<HamiltonianQC<SU2>>(vacuum, norb, orbsym, fcidump);

    shared_ptr<MPO<SU2>> mpo2 =
        make_shared<MPOQC<SU2>>(hamil2, QCTypes::Conventional);
    mpo2 = make_shared<SimplifiedMPO<SU2>>(
        mpo2, make_shared<RuleQC<SU2>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));

    shared_ptr<MovingEnvironment<SU2>> me2 =
        make_shared<MovingEnvironment<SU2>>(mpo2, mps, mps, "DMRG");
    t.get_time();
    me2->init_environments(false);

    shared_ptr<DMRG<SU2>> dmrg2 = make_shared<DMRG<SU2>>(me2, bdims, noises);
    dmrg2->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg2->me->cached_contraction = true;
    dmrg2->davidson_conv_thrds = davthrs;
    dmrg2->iprint = 2;
    // dmrg->cutoff = 1E-20;
    // dmrg->noise_type = NoiseTypes::Wavefunction;
    dmrg2->decomp_type = DecompositionTypes::DensityMatrix;
    // dmrg->noise_type = NoiseTypes::Perturbative;
    dmrg2->noise_type = NoiseTypes::ReducedPerturbativeCollectedLowMem;
    dmrg2->solve(30, mps->center == 0);

    shared_ptr<MPO<SU2>> pmpo2 = make_shared<PDM1MPOQC<SU2>>(hamil2);
    pmpo2 = make_shared<SimplifiedMPO<SU2>>(pmpo2, make_shared<RuleQC<SU2>>(),
                                            true);
    shared_ptr<MovingEnvironment<SU2>> pme2 =
        make_shared<MovingEnvironment<SU2>>(pmpo2, mps, mps, "1PDM");
    pme2->init_environments(true);
    shared_ptr<Expect<SU2>> expect2 = make_shared<Expect<SU2>>(pme2, 750, 750);
    expect2->solve(true, mps->center == 0);
    MatrixRef dm2 = expect2->get_1pdm_spatial();
    for (int i = 0; i < dm.m; i++)
        cout << setw(12) << setprecision(8) << dm2(i, i) << " ";
    cout << endl;
    cout << endl;
    for (int i = 0; i < w.m; i++) {
        for (int j = 0; j < w.n; j++)
            cout << setw(12) << setprecision(8) << dm2(i, j) << " ";
        cout << endl;
    }
    dm2.deallocate();

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
