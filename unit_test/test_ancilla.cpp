
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestAncilla : public ::testing::Test {
  protected:
    size_t isize = 1L << 28;
    size_t dsize = 1L << 32;
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

typedef SZ Spin;

TEST_F(TestAncilla, Test) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    string filename = "data/HUBBARD-L8.FCIDUMP"; // E = -6.22563376
    // string filename = "data/N2.STO3G.FCIDUMP"; // E = -107.65412235
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_d2h);
    Spin vaccum(0);
    Spin target(fcidump->n_sites() * 2, fcidump->twos(),
                     PointGroup::swap_d2h(fcidump->isym()));
    int n_physical_sites = fcidump->n_sites();
    int n_sites = n_physical_sites * 2;
    uint16_t bond_dim = 500;
    double beta = 0.0025;
    HamiltonianQC<Spin> hamil(vaccum, target, n_physical_sites, orbsym, fcidump);
    hamil.opf->seq->mode = SeqTypes::Simple;

    // Ancilla MPSInfo (thermal)
    shared_ptr<AncillaMPSInfo<Spin>> mps_info_thermal = make_shared<AncillaMPSInfo<Spin>>(
        n_physical_sites, vaccum, target, hamil.basis, hamil.orb_sym, hamil.n_syms);
    mps_info_thermal->set_thermal_limit();
    mps_info_thermal->tag = "KET";
    mps_info_thermal->save_mutable();
    mps_info_thermal->deallocate_mutable();

    // Ancilla MPSInfo (initial)
    shared_ptr<AncillaMPSInfo<Spin>> mps_info = make_shared<AncillaMPSInfo<Spin>>(
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
    shared_ptr<MPS<Spin>> mps_thermal = make_shared<MPS<Spin>>(n_sites, n_sites - 2, 2);
    mps_info_thermal->load_mutable();
    mps_thermal->initialize(mps_info_thermal);
    mps_thermal->fill_thermal_limit();

    // Ancilla MPS (initial)
    shared_ptr<MPS<Spin>> mps = make_shared<MPS<Spin>>(n_sites, n_sites - 2, 2);
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
    shared_ptr<MPO<Spin>> mpo = make_shared<MPOQC<Spin>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // Ancilla MPO construction
    cout << "Ancilla MPO start" << endl;
    mpo = make_shared<AncillaMPO<Spin>>(mpo);
    cout << "Ancilla MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<Spin>>(mpo, make_shared<RuleQC<Spin>>());
    // mpo = make_shared<SimplifiedMPO>(mpo, make_shared<Rule>());
    cout << "MPO simplification end .. T = " << t.get_time() << endl;
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    // Identity MPO
    shared_ptr<MPO<Spin>> impo = make_shared<IdentityMPO<Spin>>(hamil);
    impo = make_shared<AncillaMPO<Spin>>(impo);
    impo = make_shared<SimplifiedMPO<Spin>>(impo, make_shared<Rule<Spin>>());

    // 1PDM MPO
    shared_ptr<MPO<Spin>> pmpo = make_shared<PDM1MPOQC<Spin>>(hamil);
    pmpo = make_shared<AncillaMPO<Spin>>(pmpo, true);
    pmpo = make_shared<SimplifiedMPO<Spin>>(pmpo, make_shared<Rule<Spin>>());
    // cout << mpo->get_blocking_formulas() << endl;
    // abort();

    // Identity ME
    shared_ptr<MovingEnvironment<Spin>> ime =
        make_shared<MovingEnvironment<Spin>>(impo, mps, mps_thermal, "COMPRESS");
    ime->init_environments();

    // Compress
    vector<uint16_t> bra_bdims = {bond_dim}, ket_bdims = {10};
    vector<double> noises = {0};
    shared_ptr<Compress<Spin>> cps = make_shared<Compress<Spin>>(ime, bra_bdims, ket_bdims, noises);
    cps->solve(30, false);

    // TE ME
    shared_ptr<MovingEnvironment<Spin>> me = make_shared<MovingEnvironment<Spin>>(mpo, mps, mps, "TE");
    me->init_environments();

    // Imaginary TE
    vector<uint16_t> bdims = {bond_dim};
    shared_ptr<ImaginaryTE<Spin>> te = make_shared<ImaginaryTE<Spin>>(me, bdims, TETypes::RK4);
    te->solve(2, beta / 2, cps->forward);

    // 1PDM ME
    shared_ptr<MovingEnvironment<Spin>> pme = make_shared<MovingEnvironment<Spin>>(pmpo, mps, mps, "1PDM");
    pme->init_environments();

    // 1PDM
    shared_ptr<Expect<Spin>> expect = make_shared<Expect<Spin>>(pme, bond_dim, bond_dim);
    expect->solve(true, te->forward);

    // deallocate persistent stack memory
    pmpo->deallocate();
    impo->deallocate();
    mpo->deallocate();
    mps_info->deallocate();
    mps_info_thermal->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
