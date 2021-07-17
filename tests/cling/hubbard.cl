
#pragma cling add_include_path("/usr/local/apps/intel/mkl/include")
#pragma cling load ("mkl_core", "mkl_sequential", "mkl_gf_lp64", "mkl_avx2", "mkl_avx512")
#define _HAS_INTEL_MKL 1
#include "src/block2.hpp"

using namespace block2;
using namespace std;

// Global Settings
Random::rand_seed(0);
frame_() = make_shared<DataFrame>(1 << 20, 1 << 30, "tmp");
frame_()->use_main_stack = false;
frame_()->minimal_disk_usage = true;
threading_() = make_shared<Threading>(ThreadingTypes::SequentialGEMM, 1, 1, 1);
threading_()->seq_type = SeqTypes::None;
cout << *frame_() << endl;
cout << *threading_() << endl;

// Hamiltonian initialization
shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
PGTypes pg = PGTypes::D2H;
string filename = "data/HUBBARD-L16.FCIDUMP";
fcidump->read(filename);
vector<uint8_t> orbsym = fcidump->orb_sym();
transform(orbsym.begin(), orbsym.end(), orbsym.begin(), PointGroup::swap_pg(pg));
SU2 vacuum(0);
SU2 target(fcidump->n_elec(), fcidump->twos(), PointGroup::swap_pg(pg)(fcidump->isym()));
int norb = fcidump->n_sites();
shared_ptr<HamiltonianQC<SU2>> hamil = make_shared<HamiltonianQC<SU2>>(vacuum, norb, orbsym, fcidump);

// MPO/MPS
ubond_t bond_dim = 500;
shared_ptr<MPO<SU2>> mpo = make_shared<MPOQC<SU2>>(hamil, QCTypes::Conventional);
mpo = make_shared<SimplifiedMPO<SU2>>(mpo, make_shared<RuleQC<SU2>>(), true, true,
    OpNamesSet({OpNames::R, OpNames::RD}));
shared_ptr<MPSInfo<SU2>> mps_info = make_shared<MPSInfo<SU2>>(
    norb, vacuum, target, hamil->basis);
mps_info->set_bond_dimension(bond_dim);
shared_ptr<MPS<SU2>> mps = make_shared<MPS<SU2>>(norb, 0, 2);
mps->initialize(mps_info);
mps->random_canonicalize();
mps->save_mutable();
mps->deallocate();
mps_info->save_mutable();
mps_info->deallocate_mutable();

// DMRG
auto me = make_shared<MovingEnvironment<SU2>>(mpo, mps, mps, "DMRG");
me->init_environments(false);
vector<ubond_t> bdims = {bond_dim};
vector<double> noises = {1E-5, 1E-6, 0};
vector<double> davthrs = {1E-6, 1E-7, 1E-8};
shared_ptr<DMRG<SU2>> dmrg = make_shared<DMRG<SU2>>(me, bdims, noises);
dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
dmrg->me->cached_contraction = true;
dmrg->davidson_conv_thrds = davthrs;
dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected;
dmrg->solve(10, true);

mps_info->deallocate();
mpo->deallocate();
hamil->deallocate();
fcidump->deallocate();
