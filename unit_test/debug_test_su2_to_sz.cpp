
#include "block2_core.hpp"
#include "block2_dmrg.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestDMRG : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
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
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/HUBBARD-L16.FCIDUMP"; // E = -12.96671541
    string filename = "data/C2.CAS.PVDZ.FCIDUMP.ORIG"; // E = -12.96671541
    // string filename = "data/H4.STO6G.R1.8.FCIDUMP";
    Timer t;
    t.get_time();
    cout << "INT start" << endl;
    fcidump->read(filename);
    cout << "INT end .. T = " << t.get_time() << endl;
    Random::rand_seed(1234);

    cout << "ORB SYM = ";
    for (int i = 0; i < fcidump->n_sites(); i++)
        cout << setw(2) << (int)fcidump->orb_sym<uint8_t>()[i];
    cout << endl;

    vector<uint8_t> orbsym = fcidump->orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    SU2 vacuum(0);
    SU2 target(fcidump->n_elec(), fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    bool su2 = !fcidump->uhf;
    shared_ptr<HamiltonianQC<SU2>> hamil = make_shared<HamiltonianQC<SU2>>(vacuum, norb, orbsym, fcidump);

    shared_ptr<MPO<SU2>> mpo =
        make_shared<MPOQC<SU2>>(hamil, QCTypes::Conventional);
    mpo = make_shared<SimplifiedMPO<SU2>>(
        mpo, make_shared<RuleQC<SU2>>(), true, true,
        OpNamesSet({OpNames::R, OpNames::RD}));

    ubond_t bond_dim = 200;

    shared_ptr<MPSInfo<SU2>> mps_info =
        make_shared<MPSInfo<SU2>>(norb, vacuum, target, hamil->basis);
    mps_info->set_bond_dimension(bond_dim);

    Random::rand_seed(384666);
    shared_ptr<MPS<SU2>> mps = make_shared<MPS<SU2>>(norb, 0, 2);
    mps->initialize(mps_info);
    mps->random_canonicalize();
    mps->save_mutable();
    mps_info->save_mutable();

    // ME
    shared_ptr<MovingEnvironment<SU2>> me =
        make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
    t.get_time();
    me->init_environments(false);

    // DMRG
    vector<ubond_t> bdims = {250, 250, 250, 250, 250, 500, 500, 500,
                             500, 500, 750, 750, 750, 750, 750};
    vector<double> noises = {1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5,
                             1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6};
    vector<double> davthrs = {1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-5, 1E-6,
                              1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6,
                              5E-7, 5E-7, 5E-7, 5E-7, 5E-7, 5E-7};
    shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->cached_contraction = true;
    dmrg->davidson_conv_thrds = davthrs;
    dmrg->iprint = 2;
    dmrg->decomp_type = DecompositionTypes::DensityMatrix;
    dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollectedLowMem;
    dmrg->solve(4, true);

    dmrg->me->dot = 1;
    dmrg->solve(2, true);

    mps->flip_fused_form(mps->center, mpo->tf->opf->cg);

    cout << mps->center << " " << mps->canonical_form << endl;
    mps->info->load_mutable();
    mps->load_mutable();

    SZ targetz(target.n(), target.twos(), target.pg());
    shared_ptr<MPSInfo<SZ>> infoz =
        TransMPSInfo<SU2, SZ>::forward(mps->info, targetz);

    shared_ptr<UnfusedMPS<SU2>> umps = make_shared<UnfusedMPS<SU2>>(mps);

    shared_ptr<MPS<SZ>> gmps = TransUnfusedMPS<SU2, SZ>::forward(umps, "ZKET", mpo->tf->opf->cg, targetz)->finalize();

    // deallocate persistent stack memory
    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();
}
