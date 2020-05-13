
#include "quantum.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestAncilla : public ::testing::Test {
  protected:
    size_t isize = 1L << 28;
    size_t dsize = 1L << 32;
    void SetUp() override {
        Random::rand_seed(0);
        frame = new DataFrame(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame->activate(0);
        assert(ialloc->used == 0 && dalloc->used == 0);
        delete frame;
    }
};

TEST_F(TestAncilla, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    // string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              Hamiltonian::swap_d2h);
    SpinLabel vaccum(0);
    SpinLabel target(fcidump->n_sites() * 2, fcidump->twos(),
                     Hamiltonian::swap_d2h(fcidump->isym()));
    int n_physical_sites = fcidump->n_sites();
    int n_sites = n_physical_sites * 2;
    bool su2 = !fcidump->uhf;
    uint16_t bond_dim = 500;
    double beta = 0.02;
    Hamiltonian hamil(vaccum, target, n_physical_sites, su2, fcidump, orbsym);

    // Ancilla MPSInfo (thermal)
    shared_ptr<AncillaMPSInfo> mps_info_thermal = make_shared<AncillaMPSInfo>(
        n_physical_sites, vaccum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    mps_info_thermal->set_thermal_limit();
    mps_info_thermal->tag = "KET";
    mps_info_thermal->save_mutable();
    mps_info_thermal->deallocate_mutable();

    // Ancilla MPSInfo (initial)
    shared_ptr<AncillaMPSInfo> mps_info = make_shared<AncillaMPSInfo>(
        n_physical_sites, vaccum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    mps_info->set_bond_dimension(bond_dim);
    mps_info->tag = "BRA";
    mps_info->save_mutable();
    mps_info->deallocate_mutable();

    cout << "left dims = ";
    for (int i = 0; i <= n_sites; i++)
        cout << mps_info->left_dims[i].n_states_total << " ";
    cout << endl;
    cout << "right dims = ";
    for (int i = 0; i <= n_sites; i++)
        cout << mps_info->right_dims[i].n_states_total << " ";
    cout << endl;

    Random::rand_seed(1969);
    // Ancilla MPS (thermal)
    shared_ptr<MPS> mps_thermal = make_shared<MPS>(n_sites, n_sites - 2, 2);
    mps_info_thermal->load_mutable();
    mps_thermal->initialize(mps_info_thermal);
    mps_thermal->fill_thermal_limit();

    // Ancilla MPS (initial)
    shared_ptr<MPS> mps = make_shared<MPS>(n_sites, n_sites - 2, 2);
    mps_info->load_mutable();
    mps->initialize(mps_info);
    mps->random_canonicalize();

    // MPS/MPSInfo save mutable
    mps->save_mutable();
    mps->deallocate();
    mps_info->deallocate_mutable();
    mps_thermal->save_mutable();
    mps_thermal->deallocate();
    mps_info_thermal->deallocate_mutable();

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    hamil.mu = 0.0;
    shared_ptr<MPO> mpo = make_shared<QCMPO>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // Ancilla MPO construction
    cout << "Ancilla MPO start" << endl;
    mpo = make_shared<AncillaMPO>(mpo);
    cout << "Ancilla MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO>(mpo, make_shared<RuleQCSU2>());
    // mpo = make_shared<SimplifiedMPO>(mpo, make_shared<Rule>());
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    // Identity MPO
    shared_ptr<MPO> impo = make_shared<IdentityMPO>(hamil);
    impo = make_shared<AncillaMPO>(impo);
    impo = make_shared<SimplifiedMPO>(impo, make_shared<Rule>());

    // Identity ME
    shared_ptr<TensorFunctions> tf = make_shared<TensorFunctions>(hamil.opf);
    hamil.opf->seq->mode = SeqTypes::Simple;
    shared_ptr<MovingEnvironment> ime =
        make_shared<MovingEnvironment>(impo, mps, mps_thermal, tf, hamil.site_op_infos);
    ime->tag = "COMPRESS";
    ime->init_environments();

    // Compress
    vector<uint16_t> bra_bdims = {bond_dim}, ket_bdims = {10};
    vector<double> noises = {0};
    shared_ptr<Compress> cps = make_shared<Compress>(ime, bra_bdims, ket_bdims, noises);
    cps->solve(30, false);

    // ME
    shared_ptr<MovingEnvironment> me =
        make_shared<MovingEnvironment>(mpo, mps, mps, tf, hamil.site_op_infos);
    me->tag = "TE";
    me->init_environments();

    // Imaginary TE
    vector<uint16_t> bdims = {bond_dim};
    shared_ptr<ImaginaryTE> te = make_shared<ImaginaryTE>(me, bdims);
    te->solve(2, beta / 2, cps->forward);

    // deallocate persistent stack memory
    impo->deallocate();
    mpo->deallocate();
    mps_info->deallocate();
    mps_info_thermal->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
