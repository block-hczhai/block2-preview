
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "block2.hpp"
#include <chrono>
#include <ctime>

using namespace std;
using namespace block2;

map<string, string> read_input(const string &filename) {
    if (!Parsing::file_exists(filename)) {
        cerr << "cannot find input file : " << filename << endl;
        abort();
    }
    ifstream ifs(filename.c_str());
    if (!ifs.good())
        throw runtime_error("reading on '" + filename + "' failed.");
    vector<string> lines = Parsing::readlines(&ifs);
    if (ifs.bad())
        throw runtime_error("reading on '" + filename + "' failed.");
    ifs.close();
    map<string, string> params;
    for (auto x : lines) {
        if (Parsing::trim(x).length() != 0 &&
            (Parsing::trim(x)[0] == '!' || Parsing::trim(x)[0] == '#'))
            continue;
        vector<string> line = Parsing::split(x, "=", true);
        if (line.size() == 1)
            params[Parsing::trim(line[0])] = "";
        else if (line.size() == 2)
            params[Parsing::trim(line[0])] = Parsing::trim(line[1]);
        else if (line.size() == 0)
            continue;
        else {
            cerr << "cannot parse input : " << x << endl;
            abort();
        }
    }
    return params;
}

template <typename S, typename FL>
void prepare(const map<string, string> &params) {
    typedef typename GMatrix<FL>::FP FP;

    size_t memory = 4ULL << 30;
    if (params.count("memory") != 0)
        memory = (size_t)Parsing::to_double(params.at("memory"));

    string scratch = "./node0";
    if (params.count("scratch") != 0)
        scratch = params.at("scratch");

    // random scratch file prefix to avoid conflicts
    if (params.count("prefix") != 0 && params.at("prefix") != "auto")
        scratch = params.at("prefix");
    else {
        Random::rand_seed(0);
        stringstream ss;
        ss << hex << Random::rand_int(0, 0xFFFFFF);
        scratch = ss.str();
    }

    double dmain_ratio = 0.4;
    frame_<FP>() = make_shared<DataFrame<FP>>(
        (size_t)(0.1 * memory), (size_t)(0.9 * memory), scratch, dmain_ratio);
    frame_<FP>()->use_main_stack = false;

    FP fp_cps_cutoff = 1E-16;
    if (params.count("fp_cps_cutoff") != 0)
        fp_cps_cutoff = (FP)Parsing::to_double(params.at("fp_cps_cutoff"));

    frame_<FP>()->fp_codec = make_shared<FPCodec<FP>>(fp_cps_cutoff, 1024);

    if (params.count("rand_seed") != 0)
        Random::rand_seed(Parsing::to_int(params.at("rand_seed")));
    else
        Random::rand_seed(0);

    cout << "integer stack memory = " << fixed << setprecision(4)
         << ((frame_<FP>()->isize << 2) / 1E9) << " GB" << endl;
    cout << "double  stack memory = " << fixed << setprecision(4)
         << ((frame_<FP>()->dsize << 3) / 1E9) << " GB" << endl;

    cout << "bond integer size = " << sizeof(ubond_t) << endl;
    cout << "mkl integer size = " << sizeof(MKL_INT) << endl;
}

template <typename S, typename FL> void run(const map<string, string> &params) {
    typedef typename GMatrix<FL>::FP FP;

    prepare<S, FL>(params);

#ifdef _HAS_MPI
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<MPICommunicator<S>>();
#else
    shared_ptr<ParallelCommunicator<S>> para_comm =
        make_shared<ParallelCommunicator<S>>(1, 0, 0);
#endif
    shared_ptr<ParallelRule<S, FL>> para_rule =
        make_shared<ParallelRuleQC<S, FL>>(para_comm);

    if (params.count("simple_parallel") != 0) {
        cout << "using simple parallel ..." << endl;
        para_rule = make_shared<ParallelRuleSimple<S, FL>>(
            ParallelSimpleTypes::IJ, para_comm);
    }

    if (params.count("restart_dir") != 0) {
        para_comm->barrier();
        if (!Parsing::path_exists(params.at("restart_dir")) &&
            para_comm->rank == 0)
            Parsing::mkdir(params.at("restart_dir"));
        para_comm->barrier();
        frame_<FP>()->restart_dir = params.at("restart_dir");
    }

    shared_ptr<FCIDUMP<FL>> fcidump = make_shared<FCIDUMP<FL>>();
    vector<double> occs;
    PGTypes pg = PGTypes::C1;

    if (params.count("occ_file") != 0)
        occs = read_occ(params.at("occ_file"));

    if (params.count("pg") != 0) {
        string xpg = params.at("pg");
        if (xpg == "c1")
            pg = PGTypes::C1;
        else if (xpg == "c2")
            pg = PGTypes::C2;
        else if (xpg == "ci")
            pg = PGTypes::CI;
        else if (xpg == "cs")
            pg = PGTypes::CS;
        else if (xpg == "c2h")
            pg = PGTypes::C2H;
        else if (xpg == "c2v")
            pg = PGTypes::C2V;
        else if (xpg == "d2")
            pg = PGTypes::D2;
        else if (xpg == "d2h")
            pg = PGTypes::D2H;
        else {
            cerr << "unknown point group : " << xpg << endl;
            abort();
        }
    }

    if (params.count("fcidump") != 0) {
        fcidump->read(params.at("fcidump"));
    } else {
        cerr << "'fcidump' parameter not found!" << endl;
        abort();
    }

    if (params.count("noreorder") == 0) {
        cout << "filder reorder" << endl;
        auto hmat = fcidump->abs_h1e_matrix();
        auto kmat = fcidump->abs_exchange_matrix();
        vector<double> gmat(kmat.size());
        for (size_t i = 0; i < kmat.size(); i++)
            gmat[i] = (double)(hmat[i] * (FP)1E-7 + kmat[i]);
        vector<uint16_t> x = OrbitalOrdering::fiedler(fcidump->n_sites(), gmat);
        cout << "BEST = ";
        for (int i = 0; i < fcidump->n_sites(); i++)
            cout << setw(4) << x[i];
        cout << endl;
        fcidump->reorder(x);
        occs = fcidump->reorder(occs, x);
    }

    if (params.count("n_elec") != 0)
        fcidump->params["nelec"] = params.at("n_elec");

    if (params.count("twos") != 0)
        fcidump->params["ms2"] = params.at("twos");

    if (params.count("ipg") != 0)
        fcidump->params["isym"] = params.at("ipg");

    if (params.count("n_threads") != 0) {
        int n_threads = Parsing::to_int(params.at("n_threads"));
        threading_() = make_shared<Threading>(
            ThreadingTypes::OperatorBatchedGEMM | ThreadingTypes::Global,
            n_threads, n_threads, 1);
        threading_()->seq_type = SeqTypes::Tasked;
        cout << *frame_<FP>() << endl;
        cout << *threading_() << endl;
    }

    vector<uint8_t> orbsym = fcidump->template orb_sym<uint8_t>();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));
    for (auto x : orbsym)
        if (x == 8) {
            cerr << "wrong point group symmetry irrep: pg is "
                 << (params.count("pg") != 0 ? params.at("pg") : "c1") << endl;
            abort();
        }
    typename FCIDUMP<FL>::FP integral_error = fcidump->symmetrize(orbsym);
    if (integral_error != 0)
        cout << "integral error = " << scientific << setprecision(5)
             << integral_error << endl;
    cout << fixed;

    S vacuum(0);
    S target(fcidump->n_elec(), fcidump->twos(),
             PointGroup::swap_pg(pg)(fcidump->isym()));
    if (params.count("singlet_embedding") != 0 && is_same<SU2, S>::value)
        target = S(fcidump->n_elec() + fcidump->twos(), 0,
                   PointGroup::swap_pg(pg)(fcidump->isym()));
    int norb = fcidump->n_sites();
    shared_ptr<HamiltonianQC<S, FL>> hamil = nullptr;
    if (params.count("simple_parallel") != 0)
        hamil = make_shared<HamiltonianQC<S, FL>>(
            vacuum, norb, orbsym,
            make_shared<ParallelFCIDUMP<S, FL>>(
                fcidump,
                dynamic_pointer_cast<ParallelRuleSimple<S, FL>>(para_rule)));
    else
        hamil =
            make_shared<HamiltonianQC<S, FL>>(vacuum, norb, orbsym, fcidump);

    hamil->opf->seq->mode = SeqTypes::Simple;

    if (params.count("seq_type") != 0) {
        if (params.at("seq_type") == "none")
            hamil->opf->seq->mode = SeqTypes::None;
        else if (params.at("seq_type") == "simple")
            hamil->opf->seq->mode = SeqTypes::Simple;
        else if (params.at("seq_type") == "auto")
            hamil->opf->seq->mode = SeqTypes::Auto;
        else {
            cerr << "unknown seq type : " << params.at("seq_type") << endl;
            abort();
        }
    }

    QCTypes qc_type = QCTypes::Conventional;
    bool general_mpo = false;

    if (params.count("qc_type") != 0) {
        if (params.at("qc_type") == "general")
            general_mpo = true;
        else if (params.at("qc_type") == "conventional")
            qc_type = QCTypes::Conventional;
        else if (params.at("qc_type") == "nc")
            qc_type = QCTypes::NC;
        else if (params.at("qc_type") == "cn")
            qc_type = QCTypes::CN;
        else {
            cerr << "unknown qc type : " << params.at("qc_type") << endl;
            abort();
        }
    }

    // general mpo parameters
    MPOAlgorithmTypes mpo_algo_type = MPOAlgorithmTypes::FastBipartite;

    if (params.count("mpo_algo_type") != 0) {
        if (!general_mpo) {
            cerr << "mpo algo type only useful with general mpo!" << endl;
            abort();
        }
        if (params.at("mpo_algo_type") == "bipartite")
            mpo_algo_type = MPOAlgorithmTypes::Bipartite;
        else if (params.at("mpo_algo_type") == "fast_bipartite")
            mpo_algo_type = MPOAlgorithmTypes::FastBipartite;
        else if (params.at("mpo_algo_type") == "svd")
            mpo_algo_type = MPOAlgorithmTypes::SVD;
        else if (params.at("mpo_algo_type") == "fast_svd")
            mpo_algo_type = MPOAlgorithmTypes::FastSVD;
        else if (params.at("mpo_algo_type") == "rescaled_svd")
            mpo_algo_type = MPOAlgorithmTypes::RescaledSVD;
        else if (params.at("mpo_algo_type") == "fast_rescaled_svd")
            mpo_algo_type = MPOAlgorithmTypes::FastRescaledSVD;
        else {
            cerr << "unknown mpo algo type : " << params.at("mpo_algo_type")
                 << endl;
            abort();
        }
    }

    FP mpo_cutoff = (FP)0.0;
    int mpo_max_bond_dim = -1;

    if (params.count("mpo_cutoff") != 0)
        mpo_cutoff = (FP)Parsing::to_double(params.at("mpo_cutoff"));

    if (params.count("mpo_max_bond_dim") != 0)
        mpo_max_bond_dim = Parsing::to_int(params.at("mpo_max_bond_dim"));

    // middle transformation site for conventional mpo
    int trans_center = -1;
    if (params.count("simple_parallel") != 0) {
        trans_center = (int)(norb / (1 + 1.0 / sqrt(para_comm->size)));
        if (trans_center >= norb - 2)
            qc_type = QCTypes::NC;
        cout << "using simple parallel ... trans center = " << trans_center
             << endl;
    }
    if (params.count("trans_center") != 0)
        trans_center = Parsing::to_int(params.at("trans_center"));

    Timer t;
    t.get_time();

    shared_ptr<MPO<S, FL>> mpo;

    if (params.count("load_mpo") != 0) {
        string fn = params.at("load_mpo");
        mpo = make_shared<MPO<S, FL>>(0);
        cout << "MPO loading start" << endl;
        mpo->load_data(fn);
        if (mpo->sparse_form.find('S') == string::npos)
            mpo->tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            mpo->tf = make_shared<TensorFunctions<S, FL>>(
                make_shared<CSROperatorFunctions<S, FL>>(hamil->opf->cg));
        mpo->tf->opf->seq = hamil->opf->seq;
        cout << "MPO loading end .. T = " << t.get_time() << endl;

        if (mpo->basis.size() != 0)
            hamil->basis = mpo->basis;
        hamil->n_sites = mpo->n_sites;
        norb = mpo->n_sites;

    } else {
        // MPO construction
        cout << "MPO start" << endl;
        if (general_mpo) {
            shared_ptr<GeneralHamiltonian<S, FL>> gham =
                make_shared<GeneralHamiltonian<S, FL>>(
                    hamil->vacuum, hamil->n_sites, hamil->orb_sym);
            ElemOpTypes elem_type =
                is_same<SU2, S>::value
                    ? ElemOpTypes::SU2
                    : (is_same<SZ, S>::value ? ElemOpTypes::SZ
                                             : ElemOpTypes::SGF);
            shared_ptr<GeneralFCIDUMP<FL>> gfd =
                GeneralFCIDUMP<FL>::initialize_from_qc(hamil->fcidump,
                                                       elem_type);
            gfd = gfd->adjust_order();
            mpo = make_shared<GeneralMPO<S, FL>>(gham, gfd, mpo_algo_type,
                                                 mpo_cutoff, mpo_max_bond_dim);
            mpo->build();
        } else
            mpo =
                make_shared<MPOQC<S, FL>>(hamil, qc_type, "HQC", trans_center);
        cout << "MPO end .. T = " << t.get_time() << endl;

        if (params.count("fused") != 0 || params.count("mrci-fused") != 0) {
            int n_extl = 0, n_extr = 0;

            shared_ptr<MPSInfo<S>> fusing_mps_info;

            if (params.count("mrci-fused") != 0) {
                vector<string> xmrci =
                    Parsing::split(params.at("mrci-fused"), " ", true);
                n_extr = Parsing::to_int(xmrci[0]);
                int mrci_order = Parsing::to_int(xmrci[1]);
                fusing_mps_info = make_shared<MRCIMPSInfo<S>>(
                    hamil->n_sites, n_extr, mrci_order, hamil->vacuum, target,
                    hamil->basis);
            } else {
                vector<string> xfused =
                    Parsing::split(params.at("fused"), " ", true);
                if (xfused.size() == 1)
                    n_extr = Parsing::to_int(xfused[0]);
                else {
                    n_extl = Parsing::to_int(xfused[0]);
                    n_extr = Parsing::to_int(xfused[1]);
                }
                fusing_mps_info = make_shared<MPSInfo<S>>(
                    hamil->n_sites, hamil->vacuum, target, hamil->basis);
            }

            double sparsity = 1.1;

            if (params.count("sparse_mpo") != 0) {
                sparsity = Parsing::to_double(params.at("sparse_mpo"));
                mpo->tf = make_shared<TensorFunctions<S, FL>>(
                    make_shared<CSROperatorFunctions<S, FL>>(hamil->opf->cg));
                mpo->tf->opf->seq = hamil->opf->seq;
            }

            cout << "MPO fusing left start" << endl;
            for (int i = 0; i < n_extl - 1; i++) {
                cout << "fusing left .. " << i + 1 << " / " << n_extl << endl;
                mpo = make_shared<FusedMPO<S, FL>>(mpo, hamil->basis, 0, 1);
                if (i == 10) {
                    for (auto &op : mpo->tensors[0]->ops) {
                        shared_ptr<CSRSparseMatrix<S, FL>> smat =
                            make_shared<CSRSparseMatrix<S, FL>>();
                        if (op.second->get_type() ==
                            SparseMatrixTypes::Normal) {
                            if (op.second->sparsity() > (FP)sparsity) {
                                smat->from_dense(op.second);
                                op.second->deallocate();
                            } else
                                smat->wrap_dense(op.second);
                        }
                        op.second = smat;
                    }
                    mpo->sparse_form[0] = 'S';
                }
                hamil->basis = mpo->basis;
                hamil->n_sites = mpo->n_sites;
            }
            cout << "MPO fusing left end .. T = " << t.get_time() << endl;

            cout << "MPO fusing right start" << endl;
            for (int i = 0; i < n_extr - 1; i++) {
                cout << "fusing right .. " << i + 1 << " / " << n_extr << endl;
                mpo = make_shared<FusedMPO<S, FL>>(
                    mpo, hamil->basis, mpo->n_sites - 2, mpo->n_sites - 1,
                    fusing_mps_info->right_dims_fci[mpo->n_sites - 2]);
                if (i == 10) {
                    for (auto &op : mpo->tensors[mpo->n_sites - 1]->ops) {
                        shared_ptr<CSRSparseMatrix<S, FL>> smat =
                            make_shared<CSRSparseMatrix<S, FL>>();
                        if (op.second->get_type() ==
                            SparseMatrixTypes::Normal) {
                            if (op.second->sparsity() > sparsity) {
                                smat->from_dense(op.second);
                                op.second->deallocate();
                            } else
                                smat->wrap_dense(op.second);
                        }
                        op.second = smat;
                    }
                    mpo->sparse_form[mpo->n_sites - 1] = 'S';
                }
                hamil->basis = mpo->basis;
                hamil->n_sites = mpo->n_sites;
            }
            cout << "MPO fusing right end .. T = " << t.get_time() << endl;

            fusing_mps_info->deallocate();

            norb = hamil->n_sites;
        }

        // MPO simplification
        cout << "MPO simplification start" << endl;
        mpo = make_shared<SimplifiedMPO<S, FL>>(
            mpo, make_shared<RuleQC<S, FL>>(), true);
        cout << "MPO simplification end .. T = " << t.get_time() << endl;
    }

    mpo = make_shared<ParallelMPO<S, FL>>(mpo, para_rule);

    if (params.count("save_mpo") != 0) {
        string fn = params.at("save_mpo");
        cout << "MPO saving start" << endl;
        mpo->save_data(fn);
        cout << "MPO saving end .. T = " << t.get_time() << endl;
    }

    if (params.count("print_mpo") != 0)
        cout << mpo->get_blocking_formulas() << endl;

    if (params.count("print_basis_dims") != 0) {
        cout << "basis dims = ";
        for (int i = 0; i < norb; i++)
            cout << hamil->basis[i]->n_states_total << " ";
        cout << endl;
    }

    if (params.count("print_mpo_dims") != 0) {
        cout << "left mpo dims = ";
        for (int i = 0; i < norb; i++)
            cout << mpo->left_operator_names[i]->data.size() << " ";
        cout << endl;
        cout << "right mpo dims = ";
        for (int i = 0; i < norb; i++)
            cout << mpo->right_operator_names[i]->data.size() << " ";
        cout << endl;
    }

    vector<ubond_t> bdims = {
        250, 250, 250,
        250, 250, (ubond_t)min(500U, (uint32_t)numeric_limits<ubond_t>::max())};
    vector<FP> noises = {1E-7, 1E-8, 1E-8, 1E-9, 1E-9, 0.0};
    vector<FP> davidson_conv_thrds = {5E-6};

    if (params.count("bond_dims") != 0) {
        vector<string> xbdims =
            Parsing::split(params.at("bond_dims"), " ", true);
        bdims.clear();
        for (auto x : xbdims)
            bdims.push_back((ubond_t)Parsing::to_int(x));
    }

    if (params.count("noises") != 0) {
        vector<string> xnoises = Parsing::split(params.at("noises"), " ", true);
        noises.clear();
        for (auto x : xnoises)
            noises.push_back((FP)Parsing::to_double(x));
    }

    if (params.count("davidson_conv_thrds") != 0) {
        davidson_conv_thrds.clear();
        if (params.at("davidson_conv_thrds") != "auto") {
            vector<string> xdavidson_conv_thrds =
                Parsing::split(params.at("davidson_conv_thrds"), " ", true);
            for (auto x : xdavidson_conv_thrds)
                davidson_conv_thrds.push_back((FP)Parsing::to_double(x));
        }
    }

    shared_ptr<MPSInfo<S>> mps_info = nullptr;

    if (params.count("casci") != 0) {
        // active sites, active electrons
        vector<string> xcasci = Parsing::split(params.at("casci"), " ", true);
        mps_info = make_shared<CASCIMPSInfo<S>>(
            norb, vacuum, target, hamil->basis, Parsing::to_int(xcasci[0]),
            Parsing::to_int(xcasci[1]));

        if (params.count("print_casci") != 0) {
            for (auto &cm :
                 dynamic_pointer_cast<CASCIMPSInfo<S>>(mps_info)->casci_mask)
                cout << (cm == ActiveTypes::Active
                             ? "A "
                             : (cm == ActiveTypes::Frozen ? "F " : "E "));
            cout << endl;
        }

    } else if (params.count("mrci") != 0) {
        vector<string> xmrci = Parsing::split(params.at("mrci"), " ", true);
        int n_ext = Parsing::to_int(xmrci[0]);
        int mrci_order = Parsing::to_int(xmrci[1]);
        mps_info =
            make_shared<MRCIMPSInfo<S>>(hamil->n_sites, n_ext, mrci_order,
                                        hamil->vacuum, target, hamil->basis);
    } else
        mps_info = make_shared<MPSInfo<S>>(norb, vacuum, target, hamil->basis);
    double bias = 1.0;
    double cbias = 0.0;

    if (params.count("occ_bias") != 0)
        bias = Parsing::to_double(params.at("occ_bias"));

    if (params.count("occ_cbias") != 0)
        cbias = Parsing::to_double(params.at("occ_cbias"));

    for (auto &c : occs)
        c = c >= 1 ? c - cbias : c + cbias;

    if (params.count("load_mps") != 0) {
        mps_info = make_shared<MPSInfo<S>>(0);
        string mps_info_name = frame_<FP>()->mps_dir + "/" +
                               params.at("load_mps") + "-mps_info.bin";
        if (!Parsing::file_exists(mps_info_name))
            mps_info_name = frame_<FP>()->mps_dir + "/mps_info.bin";
        mps_info->load_data(mps_info_name);
        mps_info->tag = params.at("load_mps");
        mps_info->load_mutable();
        mps_info->bond_dim = mps_info->get_max_bond_dimension();
        para_comm->barrier();
    } else {
        if (params.count("singlet_embedding") != 0 && is_same<SU2, S>::value) {
            S left_vacuum(fcidump->twos(), fcidump->twos(), 0);
            S right_vacuum = vacuum;
            mps_info->set_bond_dimension_fci(left_vacuum, right_vacuum);
        }
        if (occs.size() == 0)
            mps_info->set_bond_dimension(bdims[0]);
        else {
            assert(occs.size() == norb);
            mps_info->set_bond_dimension_using_occ(bdims[0], occs, bias);
        }
    }

    if (params.count("print_fci_dims") != 0) {
        cout << "left fci dims = ";
        for (int i = 0; i <= norb; i++)
            cout << mps_info->left_dims_fci[i]->n_states_total << " ";
        cout << endl;
        cout << "right fci dims = ";
        for (int i = 0; i <= norb; i++)
            cout << mps_info->right_dims_fci[i]->n_states_total << " ";
        cout << endl;
    }

    if (params.count("print_mps_dims") != 0) {
        cout << "left mps dims = ";
        for (int i = 0; i <= norb; i++)
            cout << mps_info->left_dims[i]->n_states_total << " ";
        cout << endl;
        cout << "right mps dims = ";
        for (int i = 0; i <= norb; i++)
            cout << mps_info->right_dims[i]->n_states_total << " ";
        cout << endl;
    }

    int center = 0, dot = 2;

    if (params.count("center") != 0)
        center = Parsing::to_int(params.at("center"));

    if (params.count("dot") != 0)
        dot = Parsing::to_int(params.at("dot"));
    shared_ptr<MPS<S, FL>> mps = nullptr;

    if (params.count("load_mps") != 0) {
        mps_info->tag = params.at("load_mps");
        mps = make_shared<MPS<S, FL>>(mps_info);
        mps->load_data();
        mps->load_mutable();
        mps_info->tag = "KET";
        para_comm->barrier();
    } else {
        mps = make_shared<MPS<S, FL>>(norb, center, dot);
        mps->initialize(mps_info);
        mps->random_canonicalize();
    }

    mps->save_mutable();
    mps->deallocate();
    mps_info->save_mutable();
    mps_info->deallocate_mutable();
    mps->save_data();

    int iprint = 2;
    if (params.count("iprint") != 0)
        iprint = Parsing::to_int(params.at("iprint"));

    shared_ptr<MovingEnvironment<S, FL, FL>> me =
        make_shared<MovingEnvironment<S, FL, FL>>(mpo, mps, mps, "DMRG");
    t.get_time();
    cout << "INIT start" << endl;
    me->init_environments(iprint >= 2);
    cout << "INIT end .. T = " << t.get_time() << endl;

    int n_sweeps = 30;
    bool forward = true;
    FP tol = 1E-6;

    if (params.count("n_sweeps") != 0)
        n_sweeps = Parsing::to_int(params.at("n_sweeps"));

    if (params.count("forward") != 0)
        forward = !!Parsing::to_int(params.at("forward"));

    if (params.count("tol") != 0)
        tol = (FP)Parsing::to_double(params.at("tol"));

    shared_ptr<DMRG<S, FL, FL>> dmrg =
        make_shared<DMRG<S, FL, FL>>(me, bdims, noises);
    dmrg->me->delayed_contraction = OpNamesSet::normal_ops();
    dmrg->me->cached_contraction = true;
    dmrg->davidson_conv_thrds = davidson_conv_thrds;
    dmrg->iprint = iprint;
    dmrg->noise_type = NoiseTypes::ReducedPerturbative;

    if (params.count("noise_type") != 0) {
        if (params.at("noise_type") == "density_matrix")
            dmrg->noise_type = NoiseTypes::DensityMatrix;
        else if (params.at("noise_type") == "wavefunction")
            dmrg->noise_type = NoiseTypes::Wavefunction;
        else if (params.at("noise_type") == "perturbative")
            dmrg->noise_type = NoiseTypes::ReducedPerturbativeCollected;
        else {
            cerr << "unknown noise type : " << params.at("noise_type") << endl;
            abort();
        }
    }

    if (params.count("trunc_type") != 0) {
        if (params.at("trunc_type") == "physical")
            dmrg->trunc_type = TruncationTypes::Physical;
        else if (params.at("trunc_type") == "reduced")
            dmrg->trunc_type = TruncationTypes::Reduced;
        else {
            cerr << "unknown trunc type : " << params.at("trunc_type") << endl;
            abort();
        }
    }

    if (params.count("decomp_type") != 0) {
        if (params.at("decomp_type") == "density_matrix")
            dmrg->decomp_type = DecompositionTypes::DensityMatrix;
        else if (params.at("decomp_type") == "svd")
            dmrg->decomp_type = DecompositionTypes::SVD;
        else if (params.at("decomp_type") == "pure_svd")
            dmrg->decomp_type = DecompositionTypes::PureSVD;
        else {
            cerr << "unknown decomp type : " << params.at("decomp_type")
                 << endl;
            abort();
        }
    }

    // keep some number of states for each sparse block
    if (params.count("trunc_type_keep") != 0) {
        ubond_t n_keep = (ubond_t)Parsing::to_int(params.at("trunc_type_keep"));
        dmrg->trunc_type =
            dmrg->trunc_type | (TruncationTypes::KeepOne * n_keep);
    }

    if (params.count("cutoff") != 0)
        dmrg->cutoff = (FP)Parsing::to_double(params.at("cutoff"));

    typename const_fl_type<FP>::FL ener = 0;
    if (params.count("twodot_to_onedot") == 0)
        ener = dmrg->solve(n_sweeps, forward, tol);
    else {
        int tto = Parsing::to_int(params.at("twodot_to_onedot"));
        dmrg->solve(tto, forward, 0);
        dmrg->me->dot = 1;
        dmrg->bond_dims = dmrg->bond_dims.size() > tto
                              ? vector<ubond_t>(dmrg->bond_dims.begin() + tto,
                                                dmrg->bond_dims.end())
                              : vector<ubond_t>{dmrg->bond_dims.back()};
        dmrg->noises =
            dmrg->noises.size() > tto
                ? vector<FP>(dmrg->noises.begin() + tto, dmrg->noises.end())
                : vector<FP>{dmrg->noises.back()};
        dmrg->davidson_conv_thrds =
            dmrg->davidson_conv_thrds.size() > tto
                ? vector<FP>(dmrg->davidson_conv_thrds.begin() + tto,
                             dmrg->davidson_conv_thrds.end())
                : vector<FP>{dmrg->davidson_conv_thrds.back()};
        ener = dmrg->solve(n_sweeps - tto, forward ^ (tto % 2), tol);
    }

    cout << "DMRG energy = " << setw(20) << setprecision(15) << fixed << ener
         << endl;

    mps->save_data();

    mps_info->deallocate();
    mpo->deallocate();
    hamil->deallocate();
    fcidump->deallocate();

    frame_<FP>()->activate(0);
    assert(ialloc_()->used == 0 &&
           dalloc_<typename GMatrix<FL>::FP>()->used == 0);
    frame_<FP>() = nullptr;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        cout << "usage : block2 <input filename>" << endl;
        abort();
    }

    string input = argv[1];

    auto params = read_input(input);

    auto start = chrono::system_clock::now();
    auto startt = chrono::system_clock::to_time_t(start);
    cout << "START AT " << ctime(&startt) << endl;

    bool single_prec = params.count("single_prec") &&
                       (params.at("single_prec") == "" ||
                        Parsing::to_int(params.at("single_prec")));
    bool use_complex = params.count("use_complex") &&
                       (params.at("use_complex") == "" ||
                        Parsing::to_int(params.at("use_complex")));
    bool sgf = params.count("general_spin") &&
               (params.at("general_spin") == "" ||
                Parsing::to_int(params.at("general_spin")));
    bool su2 = params.count("su2") == 0 || !!Parsing::to_int(params.at("su2"));

    if (single_prec && su2 && use_complex && !sgf) {
        cout << "SPIN-ADAPTED SINGLE-PREC COMPLEX" << endl;
#ifdef _USE_COMPLEX
#ifdef _USE_SINGLE_PREC
        run<SU2, complex<float>>(params);
#endif
#endif
    } else if (single_prec && !su2 && use_complex && !sgf) {
        cout << "NON-SPIN-ADAPTED SINGLE-PREC COMPLEX" << endl;
#ifdef _USE_COMPLEX
#ifdef _USE_SINGLE_PREC
        run<SZ, complex<float>>(params);
#endif
#endif
    } else if (single_prec && !su2 && use_complex && sgf) {
        cout << "GENERAL-SPIN SINGLE-PREC COMPLEX" << endl;
#ifdef _USE_SG
#ifdef _USE_COMPLEX
#ifdef _USE_SINGLE_PREC
        run<SGF, complex<float>>(params);
#endif
#endif
#endif
    } else if (su2 && use_complex && !sgf) {
        cout << "SPIN-ADAPTED COMPLEX" << endl;
#ifdef _USE_COMPLEX
        run<SU2, complex<double>>(params);
#endif
    } else if (use_complex && !sgf) {
        cout << "NON-SPIN-ADAPTED COMPLEX" << endl;
#ifdef _USE_COMPLEX
        run<SZ, complex<double>>(params);
#endif
    } else if (use_complex && sgf) {
        cout << "GENERAL-SPIN COMPLEX" << endl;
#ifdef _USE_SG
#ifdef _USE_COMPLEX
        run<SGF, complex<double>>(params);
#endif
#endif
    } else if (single_prec && su2 && !sgf) {
        cout << "SPIN-ADAPTED SINGLE-PREC" << endl;
#ifdef _USE_SINGLE_PREC
        run<SU2, float>(params);
#endif
    } else if (single_prec && !su2 && !sgf) {
        cout << "NON-SPIN-ADAPTED SINGLE-PREC" << endl;
#ifdef _USE_SINGLE_PREC
        run<SZ, float>(params);
#endif
    } else if (single_prec && sgf) {
        cout << "GENERAL-SPIN SINGLE-PREC" << endl;
#ifdef _USE_SG
#ifdef _USE_SINGLE_PREC
        run<SGF, float>(params);
#endif
#endif
    } else if (su2 && !sgf) {
        cout << "SPIN-ADAPTED" << endl;
        run<SU2, double>(params);
    } else if (!su2 && !sgf) {
        cout << "NON-SPIN-ADAPTED" << endl;
        run<SZ, double>(params);
    } else if (sgf) {
        cout << "GENERAL-SPIN" << endl;
#ifdef _USE_SG
        run<SGF, double>(params);
#endif
    }

    auto finish = chrono::system_clock::now();
    auto finisht = chrono::system_clock::to_time_t(finish);

    cout << "FINISH AT " << ctime(&finisht) << endl;
    cout << "ELAPSED " << fixed << setprecision(3)
         << ((chrono::duration<double>)(finish - start)).count() << " SECONDS"
         << endl;

    return 0;
}
