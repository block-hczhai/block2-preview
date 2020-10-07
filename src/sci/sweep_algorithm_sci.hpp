
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Huanchen Zhai <hczhai@caltech.edu>
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

#pragma once

#include "../block2/sweep_algorithm.hpp"

using namespace std;

namespace block2 {

// Density Matrix Renormalization Group for SCI
template <typename S> struct DMRGSCI : DMRG<S> {
    using DMRG<S>::iprint;
    using DMRG<S>::me;
    using DMRG<S>::davidson_soft_max_iter;
    using DMRG<S>::noise_type;
    using DMRG<S>::decomp_type;
    using typename DMRG<S>::Iteration;
    bool last_site_svd = false;
    bool last_site_1site = false; // ATTENTION: only use in two site algorithm
    DMRGSCI(const shared_ptr<MovingEnvironment<S>> &me,
            const vector<ubond_t> &bond_dims, const vector<double> &noises)
        : DMRG<S>(me, bond_dims, noises) {}
    Iteration blocking(int i, bool forward, ubond_t bond_dim, double noise,
                       double davidson_conv_thrd) override {
        const int dsmi = davidson_soft_max_iter; // Save it as it may be changed here
        const NoiseTypes nt = noise_type;
        const DecompositionTypes dt = decomp_type;
        if(last_site_1site && (i == 0 || i == me->n_sites-1) && me->dot ==1){
            throw std::runtime_error("DMRGSCI: last_site_1site should only be used in two site algorithm.");
        }
        const auto last_site_1_and_forward = last_site_1site && forward && i == me->n_sites - 2;
        const auto last_site_1_and_backward = last_site_1site && !forward && i == me->n_sites - 2;
        if (last_site_1_and_forward){
            assert(me->dot = 2);
            me->dot = 1;
            me->ket->canonical_form[i] = 'K';
            davidson_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            DMRG<S>::blocking(i, forward, bond_dim, 0, davidson_conv_thrd);
            davidson_soft_max_iter = dsmi;
            i++;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        } else if (last_site_1_and_backward){
            me->dot = 1;
            i = me->n_sites - 1;
            if (iprint >= 2) {
                cout << "\r " << (forward ? "-->" : "<--")
                     << " Site = " << setw(4) << i << " LAST .. ";
                cout.flush();
            }
        }
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            davidson_soft_max_iter = 0;
            if (noise_type == NoiseTypes::DensityMatrix)
                noise_type = NoiseTypes::Wavefunction;
            decomp_type = DecompositionTypes::SVD;
        }
        Iteration r =
            DMRG<S>::blocking(i, forward, bond_dim, noise, davidson_conv_thrd);
        if (last_site_svd && me->dot == 1 && !forward && i == me->n_sites - 1) {
            r.energies[0] = 0;
            davidson_soft_max_iter = dsmi;
            noise_type = nt;
            decomp_type = dt;
        }
        if (last_site_1site && forward && i == me->n_sites - 1) {
            me->dot = 2;
            me->center = me->n_sites - 2;
        } else if (last_site_1site && !forward && i == me->n_sites - 1) {
            assert(me->dot = 1);
            davidson_soft_max_iter = 0;
            // skip this site (only do canonicalization)
            DMRG<S>::blocking(i - 1, forward, bond_dim, 0, davidson_conv_thrd);
            davidson_soft_max_iter = dsmi;
            me->envs[i - 1]->right_op_infos.clear();
            me->envs[i - 1]->right = nullptr;
            me->dot = 2;
            me->ket->canonical_form[i - 2] = 'C';
        }
        return r;
    }
};

// hrl: DMRG-CI-AQCC and related methods
// ATTENTION: last_site_1_site must be activated!
template <typename S> struct DMRGSCIAQCC : DMRGSCI<S> {
    using DMRGSCI<S>::iprint;
    using DMRGSCI<S>::me;
    using DMRGSCI<S>::davidson_soft_max_iter;
    using DMRGSCI<S>::davidson_max_iter;
    using DMRGSCI<S>::noise_type;
    using DMRGSCI<S>::decomp_type;
    using DMRGSCI<S>::energies;
    using DMRGSCI<S>::last_site_svd;
    using DMRGSCI<S>::last_site_1site;

    double g_factor = 1.0; // G in +Q formula
    double ref_energy = 1.0; // typically CAS-SCF/Reference energy of CAS
    double delta_e = 0.0; // energy - ref_energy => will be modified during the sweep
    std::vector<S> mod_qns; // Quantum numbers to be modified
    int max_aqcc_iter = 5; // Max iter spent on last site. Convergence depends on davidson conv.
                           // Note that this does not need to be fully converged as we do sweeps anyways.
    DMRGSCIAQCC(const shared_ptr<MovingEnvironment<S>> &me,
        const vector<ubond_t> &bond_dims, const vector<double> &noises,
        double g_factor, double ref_energy,
        const std::vector<S>& mod_qns)
        : DMRGSCI<S>(me, bond_dims, noises),
    // vv weird compile error -> cannot find member types -.-
    //     last_site_svd{true}, last_site_1site{true},
        g_factor{g_factor}, ref_energy{ref_energy},
        mod_qns{mod_qns}
        {
            last_site_svd = true;
            last_site_1site = me->dot == 2;
            modify_mpo_mats(true, 0.0); // Save diagonals
        }

        tuple<double, int, size_t, double> one_dot_eigs_and_perturb(const bool forward, const bool fuse_left,
                                                                    const int i_site,
                                                                    const double davidson_conv_thrd,
                                                                    const double noise,
                                                                    shared_ptr<SparseMatrixGroup<S>>& pket)
        override{
            tuple<double, int, size_t, double> pdi{0.,0,0,0.}; // energy, ndav, nflop, tdav
            const auto doAQCC = i_site == me->n_sites-1 and abs(davidson_soft_max_iter) > 0;
            auto calcDiagIterative = false;
            shared_ptr<EffectiveHamiltonian<S>> h_eff = me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                                                                    // vv diag will be computed in aqcc loop
                                                                    not doAQCC or not calcDiagIterative, me->bra->tensors[i_site],
                                                                    me->ket->tensors[i_site]);
            if (doAQCC){
                // AQCC
                if(energies.size() > 0){
                    delta_e = energies.back().at(0) - ref_energy;
                }
                double last_delta_e = delta_e;
                if(iprint >= 2){
                    cout << endl;
                }
                for(int itAQCC = 0; itAQCC < max_aqcc_iter; ++itAQCC){
                    //
                    // Shift non-reference ops
                    //
                    const auto shift = (1. - g_factor) * delta_e;
                    auto Hop = dynamic_pointer_cast<CSRSparseMatrix<S>>(h_eff->op->rops[me->mpo->op]);
                    if(Hop == nullptr){
                        throw std::runtime_error("MRCIAQCC: No CSRSparseMatrix is used?");
                    }
                    modify_H_mats(Hop, false, shift);
                    //
                    // Compute diagonal
                    //
                    if(calcDiagIterative) {
                        //ATTENTION: This causes a bug in eigs I don't understand
                        // csr_operator_functions.hpp:205: void block2::CSROperatorFunctions<S>::tensor_product_multiply(uint8_t, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, const std::shared_ptr<block2::SparseMatrix<S> >&, S, double) const [with S = block2::SZLong; uint8_t = unsigned char]: Assertion `ik < cinfo->n[conj + 1]' failed.
                        if (itAQCC == 0) {
                            h_eff->diag = make_shared<SparseMatrix < S>>();
                            h_eff->diag->allocate(h_eff->ket->info);
                            shared_ptr<typename SparseMatrixInfo<S>::ConnectionInfo> diag_info =
                                    make_shared<typename SparseMatrixInfo<S>::ConnectionInfo>();
                            S cdq = h_eff->ket->info->delta_quantum;
                            vector<S> msl = Partition<S>::get_uniq_labels({h_eff->hop_mat});
                            vector<vector<pair<uint8_t, S>>> msubsl =
                                    Partition<S>::get_uniq_sub_labels(h_eff->op->mat, h_eff->hop_mat, msl);
                            diag_info->initialize_diag(cdq, h_eff->opdq, msubsl[0], h_eff->left_op_infos,
                                                       h_eff->right_op_infos, h_eff->diag->info, h_eff->tf->opf->cg);
                            h_eff->diag->info->cinfo = diag_info;
                            h_eff->tf->tensor_product_diagonal(h_eff->op->mat->data[0], h_eff->op->lops,
                                                               h_eff->op->rops,
                                                               h_eff->diag, h_eff->opdq);
                            if (h_eff->tf->opf->seq->mode == SeqTypes::Auto)
                                h_eff->tf->opf->seq->auto_perform();
                            diag_info->deallocate();
                            h_eff->compute_diag = true;
                        } else {
                            h_eff->diag->clear();
                            h_eff->tf->tensor_product_diagonal(h_eff->op->mat->data[0], h_eff->op->lops,
                                                               h_eff->op->rops,
                                                               h_eff->diag, h_eff->opdq);
                        }
                    }
                    //
                    // EIG and conv check
                    //
                    { // ATTENTION: For now, redo it as h_Eff modification above does not work
                        modify_mpo_mats(false, shift);
                        h_eff = me->eff_ham(fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                                // vv diag will be computed in aqcc loop
                                            true, me->bra->tensors[i_site],
                                            me->ket->tensors[i_site]);
                    }
                    const auto pdi2 = h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                                                  davidson_soft_max_iter, me->para_rule);
                    const auto energy = std::get<0>(pdi2) + me->mpo->const_e;
                    const auto ndav = std::get<1>(pdi2);
                    std::get<0>(pdi) = std::get<0>(pdi2);
                    std::get<1>(pdi) += std::get<1>(pdi2); // ndav
                    std::get<2>(pdi) += std::get<2>(pdi2); // nflop
                    std::get<3>(pdi) += std::get<3>(pdi2); // tdav
                    delta_e = energy - ref_energy;
                    auto converged = abs(delta_e - last_delta_e) / abs(delta_e) < 1.1 * davidson_conv_thrd;
                    if(iprint >= 2){
                        cout << "\tAQCC: " << setw(2) << itAQCC <<
                             " E=" << fixed << setw(17) << setprecision(10) << energy <<
                             " Delta=" << fixed << setw(17) << setprecision(10) <<  delta_e <<
                            " prevDelta=" << fixed << setw(17) << setprecision(10) <<  last_delta_e <<
                             " nDav=" << setw(3) << ndav <<
                             " conv=" << (converged ? "T" : "F") << endl;
                    }
                    last_delta_e = delta_e;
                    if(converged){
                        break;
                    }
                }
                // ATTENTION: Right now, moving environment contains a copy of the mpo site matrices...
                //                     => so MPO needs to be adjusted as well
                const auto shift = (1. - g_factor) * delta_e;
                modify_mpo_mats(false, shift);
                // vv restore printing
                if(iprint >= 2) {
                    if (last_site_1site) {
                        cout << (forward ? " -->" : " <--") << " Site = " << setw(4) << i_site << " LAST .. ";

                    } else {
                        cout << (forward ? " -->" : " <--") << " Site = " << setw(4) << i_site << " .. ";
                    }
                }
            }else {
                pdi = h_eff->eigs(iprint >= 3, davidson_conv_thrd, davidson_max_iter,
                                  davidson_soft_max_iter, me->para_rule);
            }
            if ((noise_type & NoiseTypes::Perturbative) && noise != 0) {
                pket = h_eff->perturbative_noise(
                        forward, i_site, i_site,
                        fuse_left ? FuseTypes::FuseL : FuseTypes::FuseR,
                        me->ket->info, noise_type, me->para_rule);
            }
            h_eff->deallocate();
            return pdi;
        }

private:
    std::vector<double> mpo_diag_elements; // Save diagonal elements of all operators for adjusting shift
    // save == true: fill mpo_diag_elements (ONLY in ctor); diag_shift will not be used then
    void modify_mpo_mats(const bool save, const double diag_shift) {
        if (save) {
            assert(mpo_diag_elements.size() == 0); // should only be called in C'tor
        }
        auto &ops = me->mpo->tensors.at(me->mpo->n_sites - 1)->ops;
        for (auto &p : ops) {
            OpElement<S> &op = *dynamic_pointer_cast<OpElement<S>>(p.first);
            if (op.name == OpNames::H) {
                auto Hop = dynamic_pointer_cast<CSRSparseMatrix<S>>(p.second);
                modify_H_mats(Hop, save, diag_shift);
                break;
            }
        }
    }
    void modify_H_mats(std::shared_ptr<CSRSparseMatrix<S>>& Hop, const bool save, const double diag_shift){
        std::size_t itVec = 0;
        for(const auto& qn: mod_qns){
            const auto idx = Hop->info->find_state(qn);
            if(idx < 0){
                // Not all QNs make sense for H!
                continue;
                //cerr << "DMRGSCIAQCC: Quantumnumber " << qn << "not found!" << endl;
                //throw std::runtime_error("DMRGSCIAQCC: mod_qns not found in Hop");
            }
            CSRMatrixRef mat = (*Hop)[idx];
            assert(mat.m == mat.n);
            if(mat.nnz == mat.size()){
                auto dmat = mat.dense_ref();
                for (int iRow = 0; iRow < mat.m; iRow++) {
                    if(save){
                        mpo_diag_elements.emplace_back(dmat(iRow,iRow));
                    }else{
                        auto origVal = mpo_diag_elements[itVec++];
                        auto prev = dmat(iRow,iRow);
                        dmat(iRow,iRow) = origVal + diag_shift;
                        assert(abs( mat.dense_ref()(iRow,iRow) - origVal - diag_shift) < 1e-13);
                    }
                }
            }else{
                int nCounts = 0;
                for (int iRow = 0; iRow < mat.m; ++iRow) { // see mat.trace()
                    int rows_end = iRow == mat.m - 1 ? mat.nnz : mat.rows[iRow + 1];
                    int ic = lower_bound(mat.cols + mat.rows[iRow], mat.cols + rows_end, iRow) - mat.cols;
                    if (ic != rows_end && mat.cols[ic] == iRow) {
                        if(save) {
                            mpo_diag_elements.emplace_back(mat.data[ic]);
                        }else{
                            auto origVal = mpo_diag_elements[itVec++];
                            mat.data[ic] = origVal + diag_shift;
                        }
                        ++nCounts;
                    }
                }
                if(nCounts != mat.m and save){ // Do this only once in Ctor!
                    // This is the diagonal so I assume for now that this rarely appears.
                    cerr << "DMRGSCIAQCC: ATTENTION! for qn" << qn << " only " << nCounts
                         << " of " << mat.m << "diagonals are shifted. Change code!" << endl;
                }
            }
        }
        if(save){
            mpo_diag_elements.shrink_to_fit();
        }
    }

    };

} // namespace block2
