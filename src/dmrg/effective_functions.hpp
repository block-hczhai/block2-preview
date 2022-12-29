
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

#pragma once

#include "../core/complex_matrix_functions.hpp"
#include "../core/iterative_matrix_functions.hpp"
#include "effective_hamiltonian.hpp"

using namespace std;

namespace block2 {

template <typename S, typename FL, typename = void> struct EffectiveFunctions;

template <typename S, typename FL>
struct EffectiveFunctions<
    S, FL, typename enable_if<is_floating_point<FL>::value>::type> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FC FC;
    // [bra] = ([H_eff] + omega + i eta)^(-1) x [ket]
    // (real gf, imag gf), (nmult, niter), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double> greens_function(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff,
        typename const_fl_type<FL>::FL const_e, LinearSolverTypes solver_type,
        FL omega, FL eta, const shared_ptr<SparseMatrix<S, FL>> &real_bra,
        pair<int, int> linear_solver_params, bool iprint = false,
        FP conv_thrd = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        if (solver_type == LinearSolverTypes::Automatic)
            solver_type = LinearSolverTypes::GCROT;
        int nmult = 0, nmultx = 0, niter = 0;
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        GMatrix<FL> mket(h_eff->ket->data, (MKL_INT)h_eff->ket->total_memory,
                         1);
        GMatrix<FL> ibra(h_eff->bra->data, (MKL_INT)h_eff->bra->total_memory,
                         1);
        GMatrix<FL> rbra(real_bra->data, (MKL_INT)real_bra->total_memory, 1);
        GMatrix<FL> bre(nullptr, (MKL_INT)h_eff->ket->total_memory, 1);
        GMatrix<FL> cre(nullptr, (MKL_INT)h_eff->ket->total_memory, 1);
        GMatrix<FC> cbra(nullptr, (MKL_INT)h_eff->bra->total_memory, 1);
        GMatrix<FC> cket(nullptr, (MKL_INT)h_eff->bra->total_memory, 1);
        bre.allocate();
        cre.allocate();
        cbra.allocate();
        cket.allocate();
        GDiagonalMatrix<FC> aa(nullptr, 0);
        if (h_eff->compute_diag) {
            aa = GDiagonalMatrix<FC>(nullptr,
                                     (MKL_INT)h_eff->diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++)
                aa.data[i] =
                    FC(h_eff->diag->data[i] + (FL)const_e + omega, eta);
        }
        h_eff->precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &)> &f =
            [h_eff](const GMatrix<FL> &a, const GMatrix<FL> &b) {
                if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                    (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                    return h_eff->tf->operator()(a, b);
                else
                    return (*h_eff)(a, b);
            };
        auto op = [omega, eta, const_e, &f, &bre, &cre,
                   &nmult](const GMatrix<FC> &b, const GMatrix<FC> &c) -> void {
            GMatrixFunctions<FC>::extract_complex(
                b, bre, GMatrix<FL>(nullptr, bre.m, bre.n));
            cre.clear();
            f(bre, cre);
            GMatrixFunctions<FC>::fill_complex(
                c, cre, GMatrix<FL>(nullptr, cre.m, cre.n));
            GMatrixFunctions<FC>::extract_complex(
                b, GMatrix<FL>(nullptr, bre.m, bre.n), bre);
            cre.clear();
            f(bre, cre);
            GMatrixFunctions<FC>::fill_complex(
                c, GMatrix<FL>(nullptr, cre.m, cre.n), cre);
            GMatrixFunctions<FC>::iadd(c, b, FC((FL)const_e + omega, eta));
            nmult += 2;
        };
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        rbra.clear();
        f(ibra, rbra); // TODO not needed for chebychev
        GMatrixFunctions<FL>::iadd(rbra, ibra, (FL)const_e + omega);
        GMatrixFunctions<FL>::iscale(rbra, -1.0 / eta);
        GMatrixFunctions<FC>::fill_complex(cbra, rbra, ibra);
        cket.clear();
        GMatrixFunctions<FC>::fill_complex(
            cket, mket, GMatrix<FL>(nullptr, mket.m, mket.n));
        // solve bra
        FC gf;
        if (solver_type == LinearSolverTypes::GCROT)
            gf = IterativeMatrixFunctions<FC>::gcrotmk(
                op, aa, cbra, cket, nmultx, niter, linear_solver_params.first,
                linear_solver_params.second, 0.0, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
                max_iter, soft_max_iter);
        else if (solver_type == LinearSolverTypes::LSQR) {
            // Implementation uses conventional tolerance of ||r|| instead of
            // ||r||^2
            const auto tol = sqrt(conv_thrd);
            // hrl NOTE: I assume that H is Hermitian. So the only difference of
            // rop cmp to op is the "-eta".
            const auto rop = [omega, eta, const_e, &f, &bre, &cre,
                              &nmult](const GMatrix<FC> &b,
                                      const GMatrix<FC> &c) -> void {
                GMatrixFunctions<FC>::extract_complex(
                    b, bre, GMatrix<FL>(nullptr, bre.m, bre.n));
                cre.clear();
                f(bre, cre);
                GMatrixFunctions<FC>::fill_complex(
                    c, cre, GMatrix<FL>(nullptr, cre.m, cre.n));
                GMatrixFunctions<FC>::extract_complex(
                    b, GMatrix<FL>(nullptr, bre.m, bre.n), bre);
                cre.clear();
                f(bre, cre);
                GMatrixFunctions<FC>::fill_complex(
                    c, GMatrix<FL>(nullptr, cre.m, cre.n), cre);
                GMatrixFunctions<FC>::iadd(c, b, FC((FL)const_e + omega, -eta));
                nmult += 2;
            };
            const FP precond_reg = 1E-8;
            gf = IterativeMatrixFunctions<FC>::lsqr(
                op, rop, aa, cbra, cket, nmultx, niter, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, precond_reg,
                tol, tol, max_iter, soft_max_iter);
            niter++;
        } else if (solver_type == LinearSolverTypes::IDRS) {
            // Use linear_solver_params.first as "S" value in IDR(S)
            // Implementation uses conventional tolerance of ||r|| instead of
            // ||r||^2
            const auto idrs_tol = sqrt(conv_thrd);
            const FP idrs_atol = 0.0;
            const FP precond_reg = 1E-8;
            assert(linear_solver_params.first > 0);
            gf = IterativeMatrixFunctions<FC>::idrs(
                op, aa, cbra, cket, nmultx, niter, linear_solver_params.first,
                iprint, para_rule == nullptr ? nullptr : para_rule->comm,
                precond_reg, idrs_tol, idrs_atol, max_iter, soft_max_iter);
            niter++;
        } else if (solver_type == LinearSolverTypes::Cheby) {
            // Here I only use f and not op, so wrap it for nmult
            const auto Hvec = [f, &nmult](const GMatrix<FL> &a,
                                          const GMatrix<FL> &b) {
                f(a, b);
                ++nmult;
            };
            FP eMin, eMax; // eigenvalue
            FP diagMin = 999999999999990.;
            FP diagMax = -999999999999990.;
            {
                GDiagonalMatrix<FL> adiag(h_eff->diag->data,
                                          (MKL_INT)h_eff->diag->total_memory);
                // for initial vector  => use extrema of diag(H)
                // purely random vectors get stuck to easily
                MKL_INT locMin = 0;
                MKL_INT locMax = 0;
                for (MKL_INT i = 0; i < adiag.size(); i++) {
                    if (adiag.data[i] < diagMin) {
                        diagMin = adiag.data[i];
                        locMin = i;
                    }
                    if (adiag.data[i] > diagMax) {
                        diagMax = adiag.data[i];
                        locMax = i;
                    }
                    diagMax = max(diagMax, adiag.data[i]);
                }
                // Compute lowest and largest eigenvalue
                const auto dav_tol = 1e-6; // 1e-5 is not good enough
                Random rgen;
                rgen.rand_seed(-1);
                GMatrix<FL> uv(nullptr, (MKL_INT)h_eff->ket->total_memory, 1);
                uv.allocate();
                FL xnorm = 0;
                for (MKL_INT i = 0; i < aa.size(); i++) {
                    uv.data[i] =
                        1e-4 *
                        rgen.rand_double(-1, 1); // small noise to avoid getting
                                                 // stuck in symmetry sectors
                    xnorm += uv.data[i] * uv.data[i];
                }
                xnorm -= uv.data[locMin] * uv.data[locMin];
                uv.data[locMin] += 1.;
                xnorm += uv.data[locMin] * uv.data[locMin];
                for (MKL_INT i = 0; i < aa.size(); i++) {
                    uv.data[i] /= sqrt(xnorm);
                }
                int ndav = 0;
                std::vector<GMatrix<FL>> uvv{uv};
                auto evsmall = IterativeMatrixFunctions<FL>::davidson(
                    Hvec, adiag, uvv, 0.0, DavidsonTypes::Normal, ndav, false,
                    para_rule == nullptr ? nullptr : para_rule->comm, dav_tol,
                    300);
                // largest: use CloseTop and huge shift
                xnorm = 0;
                for (MKL_INT i = 0; i < aa.size(); i++) {
                    uv.data[i] = 1e-4 * rgen.rand_double(-1, 1);
                    xnorm += uv.data[i] * uv.data[i];
                }
                xnorm -= uv.data[locMax] * uv.data[locMax];
                uv.data[locMax] += 1.;
                xnorm += uv.data[locMax] * uv.data[locMax];
                for (MKL_INT i = 0; i < aa.size(); i++) {
                    uv.data[i] /= sqrt(xnorm);
                }
                auto evlarge = IterativeMatrixFunctions<FL>::davidson(
                    Hvec, adiag, uvv, 1e9, DavidsonTypes::CloseTo, ndav, false,
                    para_rule == nullptr ? nullptr : para_rule->comm, dav_tol,
                    300);
                eMin = evsmall[0];
                eMax = evlarge[0];
                uv.deallocate();
            }
            const auto nmult_davidson = nmult; // statistics
            // Scaling
            // const FP maxInterval = 0.98; // Make it slightly smaller, for
            // numerics
            const FP maxInterval = 0.58; // Make it even smaller, for numerics
            // TODO adaptive way?
            auto origEmin = eMin;
            auto origEmax = eMax;
            eMax += 1. * abs(eMax);
            eMin -= 1. * abs(eMin); // the limit analysis is too tricky
            const auto scale = 2 * maxInterval / (eMax - eMin); // 1/a = deltaH
            const auto maxNCheby =
                ceil(1.1 / (scale * eta)); // just an estimate
            // That would be fine if we use damping as well
            if (iprint) {
                cout << endl
                     << "cheby: eMin= " << scientific << setprecision(4) << eMin
                     << "; eMax = " << scientific << setprecision(4) << eMax
                     << "; original : eMin= " << scientific << setprecision(4)
                     << origEmin << "; eMax = " << scientific << setprecision(4)
                     << origEmax << ", nmultDav = " << nmult_davidson
                     << ", maxNCheby approx " << maxNCheby << endl;
            }
            // assert(linear_solver_params.first > 0);
            int damping =
                0; // TODO add damping option; linear_solver_params.first
            FC evalShift((FL)const_e + omega, eta);
            const auto nmultpre = nmult;
            gf = IterativeMatrixFunctions<FP>::cheby(
                Hvec, cbra, mket, evalShift, conv_thrd,
                min(max_iter, soft_max_iter), // here: never abort as the values
                                              // may still be ok
                eMin, eMax, maxInterval, damping, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm);
            niter += nmult - nmultpre;
        } else
            throw runtime_error("Invalid solver type of Green's function.");
        gf = xconj<FC>(gf);
        GMatrixFunctions<FC>::extract_complex(cbra, rbra, ibra);
        if (h_eff->compute_diag)
            aa.deallocate();
        cket.deallocate();
        cbra.deallocate();
        cre.deallocate();
        bre.deallocate();
        h_eff->post_precompute();
        uint64_t nflop = h_eff->tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(gf, make_pair(nmult, niter), (size_t)nflop,
                          t.get_time());
    }
    // [ibra] = (([H_eff] + omega)^2 + eta^2)^(-1) x (-eta [ket])
    // [rbra] = -([H_eff] + omega) (1/eta) [bra]
    // (real gf, imag gf), (nmult, numltp), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double> greens_function_squared(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff,
        typename const_fl_type<FL>::FL const_e, FL omega, FL eta,
        const shared_ptr<SparseMatrix<S, FL>> &real_bra,
        int n_harmonic_projection = 0, bool iprint = false, FP conv_thrd = 5E-6,
        int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int nmult = 0, nmultx = 0;
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        GMatrix<FL> mket(h_eff->ket->data, (MKL_INT)h_eff->ket->total_memory,
                         1);
        GMatrix<FL> ibra(h_eff->bra->data, (MKL_INT)h_eff->bra->total_memory,
                         1);
        GMatrix<FL> ktmp(nullptr, (MKL_INT)h_eff->ket->total_memory, 1);
        ktmp.allocate();
        GMatrix<FL> btmp(nullptr, (MKL_INT)h_eff->bra->total_memory, 1);
        btmp.allocate();
        ktmp.clear();
        GMatrixFunctions<FL>::iadd(ktmp, mket, -eta);
        GDiagonalMatrix<FL> aa(nullptr, 0);
        if (h_eff->compute_diag) {
            aa = GDiagonalMatrix<FL>(nullptr,
                                     (MKL_INT)h_eff->diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++) {
                aa.data[i] = h_eff->diag->data[i] + (FL)const_e + omega;
                aa.data[i] = aa.data[i] * aa.data[i] + eta * eta;
            }
        }
        h_eff->precompute();
        const function<void(const GMatrix<FL> &, const GMatrix<FL> &)> &f =
            [h_eff](const GMatrix<FL> &a, const GMatrix<FL> &b) {
                if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                    (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                    return h_eff->tf->operator()(a, b);
                else
                    return (*h_eff)(a, b);
            };
        auto op = [omega, eta, const_e, &f, &btmp,
                   &nmult](const GMatrix<FL> &b, const GMatrix<FL> &c) -> void {
            btmp.clear();
            f(b, btmp);
            GMatrixFunctions<FL>::iadd(btmp, b, (FL)const_e + omega);
            f(btmp, c);
            GMatrixFunctions<FL>::iadd(c, btmp, (FL)const_e + omega);
            GMatrixFunctions<FL>::iadd(c, b, eta * eta);
            nmult += 2;
        };
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        // solve imag part -> ibra
        FL igf = 0;
        int nmultp = 0;
        if (n_harmonic_projection == 0)
            igf = IterativeMatrixFunctions<FL>::conjugate_gradient(
                      op, aa, ibra, ktmp, nmultx, 0.0, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter) /
                  (-eta);
        else if (n_harmonic_projection < 0)
            assert(false);
        else {
            vector<GMatrix<FL>> bs = vector<GMatrix<FL>>(
                n_harmonic_projection,
                GMatrix<FL>(nullptr, (MKL_INT)h_eff->ket->total_memory, 1));
            for (int ih = 0; ih < n_harmonic_projection; ih++) {
                bs[ih].allocate();
                if (ih == 0)
                    GMatrixFunctions<FL>::copy(bs[ih], ibra);
                else
                    Random::fill(bs[ih].data, bs[ih].size());
            }
            IterativeMatrixFunctions<FL>::harmonic_davidson(
                op, aa, bs, 0.0,
                DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond,
                nmultx, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, 1E-4,
                max_iter, soft_max_iter, 2, 50);
            nmultp = nmult;
            nmult = 0;
            igf = IterativeMatrixFunctions<FL>::deflated_conjugate_gradient(
                      op, aa, ibra, ktmp, nmultx, 0.0, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter, bs) /
                  (-eta);
            for (int ih = n_harmonic_projection - 1; ih >= 0; ih--)
                bs[ih].deallocate();
        }
        if (h_eff->compute_diag)
            aa.deallocate();
        btmp.deallocate();
        ktmp.deallocate();
        // compute real part -> rbra
        GMatrix<FL> rbra(real_bra->data, (MKL_INT)real_bra->total_memory, 1);
        rbra.clear();
        f(ibra, rbra);
        GMatrixFunctions<FL>::iadd(rbra, ibra, (FL)const_e + omega);
        GMatrixFunctions<FL>::iscale(rbra, -1 / eta);
        // compute real part green's function
        FL rgf = GMatrixFunctions<FL>::dot(rbra, mket);
        h_eff->post_precompute();
        uint64_t nflop = h_eff->tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(FC(rgf, igf), make_pair(nmult + 1, nmultp),
                          (size_t)nflop, t.get_time());
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    // nexpo is number of complex matrix multiplications
    static tuple<FL, FP, int, size_t, double> expo_apply(
        const shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> &h_eff,
        FC beta, typename const_fl_type<FL>::FL const_e, bool iprint = false,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(h_eff->compute_diag);
        assert(h_eff->ket.size() == 2);
        FP anorm = GMatrixFunctions<FL>::norm(GMatrix<FL>(
            h_eff->diag->data, (MKL_INT)h_eff->diag->total_memory, 1));
        GMatrix<FL> vr(h_eff->ket[0]->data,
                       (MKL_INT)h_eff->ket[0]->total_memory, 1);
        GMatrix<FL> vi(h_eff->ket[1]->data,
                       (MKL_INT)h_eff->ket[1]->total_memory, 1);
        Timer t;
        t.get_time();
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        h_eff->precompute();
        int nexpo =
            (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
             (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                ? GMatrixFunctions<FC>::expo_apply(
                      *h_eff->tf, beta, anorm, vr, vi, (FL)const_e, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm)
                : GMatrixFunctions<FC>::expo_apply(
                      *h_eff, beta, anorm, vr, vi, (FL)const_e, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm);
        FP norm_re = GMatrixFunctions<FL>::norm(vr);
        FP norm_im = GMatrixFunctions<FL>::norm(vi);
        FP norm = sqrt(norm_re * norm_re + norm_im * norm_im);
        GMatrix<FL> tmp_re(nullptr, (MKL_INT)h_eff->ket[0]->total_memory, 1);
        GMatrix<FL> tmp_im(nullptr, (MKL_INT)h_eff->ket[1]->total_memory, 1);
        tmp_re.allocate();
        tmp_im.allocate();
        tmp_re.clear();
        tmp_im.clear();
        if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
            (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
            (*h_eff->tf)(vr, tmp_re), (*h_eff->tf)(vi, tmp_im);
        else
            (*h_eff)(vr, tmp_re), (*h_eff)(vi, tmp_im);
        FL energy = (GMatrixFunctions<FL>::complex_dot(vr, tmp_re) +
                     GMatrixFunctions<FL>::complex_dot(vi, tmp_im)) /
                    (norm * norm);
        tmp_im.deallocate();
        tmp_re.deallocate();
        h_eff->post_precompute();
        uint64_t nflop = h_eff->tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, (size_t)nflop, t.get_time());
    }
    // eigenvalue with mixed real and complex Hamiltonian
    // Find eigenvalues and eigenvectors of [H_eff]
    // energies, ndav, nflop, tdav
    static tuple<vector<typename const_fl_type<FP>::FL>, int, size_t, double>
    eigs_mixed(
        const shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> &h_eff,
        const shared_ptr<EffectiveHamiltonian<S, FC, MultiMPS<S, FC>>> &x_eff,
        bool iprint = false, FP conv_thrd = 5E-6, int max_iter = 5000,
        int soft_max_iter = -1,
        DavidsonTypes davidson_type = DavidsonTypes::Normal, FP shift = 0,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int ndav = 0;
        assert(h_eff->compute_diag && x_eff->compute_diag);
        frame_<FP>()->activate(0);
        GMatrix<FL> bre(nullptr, (MKL_INT)h_eff->ket[0]->total_memory, 1);
        GMatrix<FL> cre(nullptr, (MKL_INT)h_eff->ket[0]->total_memory, 1);
        // need this temp array to avoid double accumulate sum in parallel
        GMatrix<FC> cc(nullptr, (MKL_INT)x_eff->ket[0]->total_memory, 1);
        bre.allocate();
        cre.allocate();
        cc.allocate();
        GDiagonalMatrix<FC> aa =
            GDiagonalMatrix<FC>(nullptr, (MKL_INT)h_eff->diag->total_memory);
        aa.allocate();
        aa.clear();
        GMatrixFunctions<FC>::fill_complex(
            aa, GMatrix<FL>(h_eff->diag->data, aa.m, aa.n),
            GMatrix<FL>(nullptr, aa.m, aa.n));
        GMatrixFunctions<FC>::iadd(
            aa, GMatrix<FC>(x_eff->diag->data, aa.m, aa.n), 1.0);
        vector<GMatrix<FC>> bs;
        for (int i = 0; i < (int)min((MKL_INT)x_eff->ket.size(), (MKL_INT)aa.n);
             i++)
            bs.push_back(GMatrix<FC>(x_eff->ket[i]->data,
                                     (MKL_INT)x_eff->ket[i]->total_memory, 1));
        Timer t;
        t.get_time();
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        x_eff->tf->opf->seq->cumulative_nflop = 0;
        h_eff->precompute();
        x_eff->precompute();
        const function<void(const GMatrix<FC> &, const GMatrix<FC> &)> &f =
            [h_eff, x_eff, bre, cre, cc](const GMatrix<FC> &b,
                                         const GMatrix<FC> &c) {
                // real part
                GMatrixFunctions<FC>::extract_complex(
                    b, bre, GMatrix<FL>(nullptr, bre.m, bre.n));
                cre.clear();
                if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                    (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                    h_eff->tf->operator()(bre, cre);
                else
                    (*h_eff)(bre, cre);
                GMatrixFunctions<FC>::fill_complex(
                    c, cre, GMatrix<FL>(nullptr, cre.m, cre.n));
                GMatrixFunctions<FC>::extract_complex(
                    b, GMatrix<FL>(nullptr, bre.m, bre.n), bre);
                cre.clear();
                if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                    (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                    h_eff->tf->operator()(bre, cre);
                else
                    (*h_eff)(bre, cre);
                GMatrixFunctions<FC>::fill_complex(
                    c, GMatrix<FL>(nullptr, cre.m, cre.n), cre);
                // complex part
                cc.clear();
                if (x_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                    (x_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                    x_eff->tf->operator()(b, cc);
                else
                    (*x_eff)(b, cc);
                GMatrixFunctions<FC>::iadd(c, cc, 1.0);
            };
        vector<FP> xeners = IterativeMatrixFunctions<FC>::harmonic_davidson(
            f, aa, bs, shift, davidson_type, ndav, iprint,
            para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
            max_iter, soft_max_iter);
        vector<typename const_fl_type<FP>::FL> eners(xeners.size());
        for (size_t i = 0; i < xeners.size(); i++)
            eners[i] = (typename const_fl_type<FP>::FL)xeners[i];
        h_eff->post_precompute();
        x_eff->post_precompute();
        uint64_t nflop = h_eff->tf->opf->seq->cumulative_nflop +
                         x_eff->tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        x_eff->tf->opf->seq->cumulative_nflop = 0;
        aa.deallocate();
        cc.deallocate();
        cre.deallocate();
        bre.deallocate();
        assert(h_eff->ket.size() == x_eff->ket.size() * 2);
        for (int i = 0; i < (int)bs.size(); i++)
            GMatrixFunctions<FC>::extract_complex(
                bs[i],
                GMatrix<FL>(h_eff->ket[i + i]->data,
                            (MKL_INT)h_eff->ket[i + i]->total_memory, 1),
                GMatrix<FL>(h_eff->ket[i + i + 1]->data,
                            (MKL_INT)h_eff->ket[i + i + 1]->total_memory, 1));
        return make_tuple(eners, ndav, (size_t)nflop, t.get_time());
    }
};

template <typename S, typename FL>
struct EffectiveFunctions<S, FL,
                          typename enable_if<is_complex<FL>::value>::type> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FC FC;
    // [bra] = ([H_eff] + omega + i eta)^(-1) x [ket]
    // (real gf, imag gf), (nmult, niter), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double> greens_function(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff,
        typename const_fl_type<FL>::FL const_e, LinearSolverTypes solver_type,
        FL omega, FL eta, const shared_ptr<SparseMatrix<S, FL>> &real_bra,
        pair<int, int> linear_solver_params, bool iprint = false,
        FP conv_thrd = 5E-6, int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(real_bra == nullptr);
        if (solver_type == LinearSolverTypes::Automatic)
            solver_type = LinearSolverTypes::GCROT;
        int nmult = 0, nmultx = 0, niter = 0;
        frame_<FP>()->activate(0);
        Timer t;
        t.get_time();
        GMatrix<FL> mket(h_eff->ket->data, (MKL_INT)h_eff->ket->total_memory,
                         1);
        GMatrix<FL> mbra(h_eff->bra->data, (MKL_INT)h_eff->bra->total_memory,
                         1);
        GDiagonalMatrix<FC> aa(nullptr, 0);
        FC const_x = (FL)const_e + omega + FC(0.0, 1.0) * eta;
        if (h_eff->compute_diag) {
            aa = GDiagonalMatrix<FC>(nullptr,
                                     (MKL_INT)h_eff->diag->total_memory);
            aa.allocate();
            for (MKL_INT i = 0; i < aa.size(); i++)
                aa.data[i] = h_eff->diag->data[i] + const_x;
        }
        h_eff->precompute();
        auto op = [h_eff, const_x, &nmult](const GMatrix<FC> &b,
                                           const GMatrix<FC> &c) -> void {
            if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                h_eff->tf->operator()(b, c);
            else
                (*h_eff)(b, c);
            GMatrixFunctions<FC>::iadd(c, b, const_x);
            nmult++;
        };
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        // solve bra
        FC gf;
        if (solver_type == LinearSolverTypes::GCROT)
            gf = IterativeMatrixFunctions<FC>::gcrotmk(
                op, aa, mbra, mket, nmultx, niter, linear_solver_params.first,
                linear_solver_params.second, 0.0, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, conv_thrd,
                max_iter, soft_max_iter);
        else if (solver_type == LinearSolverTypes::LSQR) {
            FC const_y = xconj<FL>(const_x);
            // Implementation uses conventional tolerance of ||r|| instead of
            // ||r||^2
            const auto tol = sqrt(conv_thrd);
            // hrl NOTE: I assume that H is Hermitian. So the only difference of
            // rop cmp to op is the "-eta".
            auto rop = [h_eff, const_y, &nmult](const GMatrix<FC> &b,
                                                const GMatrix<FC> &c) -> void {
                if (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                    (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                    return h_eff->tf->operator()(b, c);
                else
                    return (*h_eff)(b, c);
                GMatrixFunctions<FC>::iadd(c, b, const_y);
                nmult += 1;
            };
            const FP precond_reg = 1E-8;
            gf = IterativeMatrixFunctions<FC>::lsqr(
                op, rop, aa, mbra, mket, nmultx, niter, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, precond_reg,
                tol, tol, max_iter, soft_max_iter);
            niter++;
        } else if (solver_type == LinearSolverTypes::IDRS) {
            // Use linear_solver_params.first as "S" value in IDR(S)
            // Implementation uses conventional tolerance of ||r|| instead of
            // ||r||^2
            const auto idrs_tol = sqrt(conv_thrd);
            const FP idrs_atol = 0.0;
            const FP precond_reg = 1E-8;
            assert(linear_solver_params.first > 0);
            gf = IterativeMatrixFunctions<FC>::idrs(
                op, aa, mbra, mket, nmultx, niter, linear_solver_params.first,
                iprint, para_rule == nullptr ? nullptr : para_rule->comm,
                precond_reg, idrs_tol, idrs_atol, max_iter, soft_max_iter);
            niter++;
        } else
            throw runtime_error("Invalid solver type of Green's function.");
        gf = xconj<FC>(gf);
        if (h_eff->compute_diag)
            aa.deallocate();
        h_eff->post_precompute();
        uint64_t nflop = h_eff->tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum_optional(&nflop, 1,
                                                 para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(gf, make_pair(nmult, niter), (size_t)nflop,
                          t.get_time());
    }
    // [ibra] = (([H_eff] + omega)^2 + eta^2)^(-1) x (-eta [ket])
    // [rbra] = -([H_eff] + omega) (1/eta) [bra]
    // (real gf, imag gf), (nmult, numltp), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double> greens_function_squared(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff,
        typename const_fl_type<FL>::FL const_e, FL omega, FL eta,
        const shared_ptr<SparseMatrix<S, FL>> &real_bra,
        int n_harmonic_projection = 0, bool iprint = false, FP conv_thrd = 5E-6,
        int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(false);
        return make_tuple(0.0, make_pair(0, 0), (size_t)0, 0.0);
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    // nexpo is number of complex matrix multiplications
    static tuple<FL, FP, int, size_t, double> expo_apply(
        const shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> &h_eff,
        FC beta, typename const_fl_type<FL>::FL const_e, bool iprint = false,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(false);
        return make_tuple(0.0, 0.0, 0, (size_t)0, 0.0);
    }
    // eigenvalue with mixed real and complex Hamiltonian
    // Find eigenvalues and eigenvectors of [H_eff]
    // energies, ndav, nflop, tdav
    static tuple<vector<typename const_fl_type<FP>::FL>, int, size_t, double>
    eigs_mixed(
        const shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> &h_eff,
        const shared_ptr<EffectiveHamiltonian<S, FC, MultiMPS<S, FC>>> &x_eff,
        bool iprint = false, FP conv_thrd = 5E-6, int max_iter = 5000,
        int soft_max_iter = -1,
        DavidsonTypes davidson_type = DavidsonTypes::Normal, FP shift = 0,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(false);
        return make_tuple(vector<typename const_fl_type<FP>::FL>{}, 0,
                          (size_t)0, 0.0);
    }
};

} // namespace block2
