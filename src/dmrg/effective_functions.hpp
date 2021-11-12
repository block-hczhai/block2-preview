
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
    static tuple<FC, pair<int, int>, size_t, double>
    greens_function(const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff,
                    FL const_e, LinearSolverTypes solver_type, FL omega, FL eta,
                    const shared_ptr<SparseMatrix<S, FL>> &real_bra,
                    pair<int, int> linear_solver_params, bool iprint = false,
                    FP conv_thrd = 5E-6, int max_iter = 5000,
                    int soft_max_iter = -1,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        if (solver_type == LinearSolverTypes::Automatic)
            solver_type = LinearSolverTypes::GCROT;
        int nmult = 0, nmultx = 0, niter = 0;
        frame->activate(0);
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
                aa.data[i] = FC(h_eff->diag->data[i] + const_e + omega, eta);
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
            GMatrixFunctions<FC>::iadd(c, b, FC(const_e + omega, eta));
            nmult += 2;
        };
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        rbra.clear();
        f(ibra, rbra);
        GMatrixFunctions<FL>::iadd(rbra, ibra, const_e + omega);
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
                              &nmult](const ComplexMatrixRef &b,
                                      const ComplexMatrixRef &c) -> void {
                ComplexMatrixFunctions::extract_complex(
                    b, bre, MatrixRef(nullptr, bre.m, bre.n));
                cre.clear();
                f(bre, cre);
                ComplexMatrixFunctions::fill_complex(
                    c, cre, MatrixRef(nullptr, cre.m, cre.n));
                ComplexMatrixFunctions::extract_complex(
                    b, MatrixRef(nullptr, bre.m, bre.n), bre);
                cre.clear();
                f(bre, cre);
                ComplexMatrixFunctions::fill_complex(
                    c, MatrixRef(nullptr, cre.m, cre.n), cre);
                ComplexMatrixFunctions::iadd(
                    c, b, complex<double>(const_e + omega, -eta));
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
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(gf, make_pair(nmult, niter), (size_t)nflop,
                          t.get_time());
    }
    // [ibra] = (([H_eff] + omega)^2 + eta^2)^(-1) x (-eta [ket])
    // [rbra] = -([H_eff] + omega) (1/eta) [bra]
    // (real gf, imag gf), (nmult, numltp), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double> greens_function_squared(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff, FL const_e,
        FL omega, FL eta, const shared_ptr<SparseMatrix<S, FL>> &real_bra,
        int n_harmonic_projection = 0, bool iprint = false, FP conv_thrd = 5E-6,
        int max_iter = 5000, int soft_max_iter = -1,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        int nmult = 0, nmultx = 0;
        frame->activate(0);
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
                aa.data[i] = h_eff->diag->data[i] + const_e + omega;
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
            GMatrixFunctions<FL>::iadd(btmp, b, const_e + omega);
            f(btmp, c);
            GMatrixFunctions<FL>::iadd(c, btmp, const_e + omega);
            GMatrixFunctions<FL>::iadd(c, b, eta * eta);
            nmult += 2;
        };
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        // solve imag part -> ibra
        FL igf = 0;
        int nmultp = 0;
        if (n_harmonic_projection == 0)
            igf = GMatrixFunctions<FL>::conjugate_gradient(
                      op, aa, ibra, ktmp, nmultx, 0.0, iprint,
                      para_rule == nullptr ? nullptr : para_rule->comm,
                      conv_thrd, max_iter, soft_max_iter) /
                  (-eta);
        else if (n_harmonic_projection < 0) {
            int ndav = 0, ncg = 0;
            int kk = -n_harmonic_projection;
            igf = GMatrixFunctions<FL>::
                      davidson_projected_deflated_conjugate_gradient(
                          op, aa, ibra, ktmp, kk, ncg, ndav, 0.0, iprint,
                          para_rule == nullptr ? nullptr : para_rule->comm,
                          conv_thrd, conv_thrd, max_iter * kk,
                          soft_max_iter * kk) /
                  (-eta);
            nmult = ncg * 2;
            nmultp = ndav * 2;
        } else {
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
            GMatrixFunctions<FL>::harmonic_davidson(
                op, aa, bs, 0.0,
                DavidsonTypes::HarmonicGreaterThan | DavidsonTypes::NoPrecond,
                nmultx, iprint,
                para_rule == nullptr ? nullptr : para_rule->comm, 1E-4,
                max_iter, soft_max_iter, 2, 50);
            nmultp = nmult;
            nmult = 0;
            igf = GMatrixFunctions<FL>::deflated_conjugate_gradient(
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
        GMatrixFunctions<FL>::iadd(rbra, ibra, const_e + omega);
        GMatrixFunctions<FL>::iscale(rbra, -1 / eta);
        // compute real part green's function
        FL rgf = GMatrixFunctions<FL>::dot(rbra, mket);
        h_eff->post_precompute();
        uint64_t nflop = h_eff->tf->opf->seq->cumulative_nflop;
        if (para_rule != nullptr)
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(FC(rgf, igf), make_pair(nmult + 1, nmultp),
                          (size_t)nflop, t.get_time());
    }
    // [ket] = exp( [H_eff] ) | [ket] > (exact)
    // energy, norm, nexpo, nflop, texpo
    // nexpo is number of complex matrix multiplications
    static tuple<FL, FP, int, size_t, double> expo_apply(
        const shared_ptr<EffectiveHamiltonian<S, FL, MultiMPS<S, FL>>> &h_eff,
        FC beta, FL const_e, bool iprint = false,
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
        int nexpo = (h_eff->tf->opf->seq->mode == SeqTypes::Auto ||
                     (h_eff->tf->opf->seq->mode & SeqTypes::Tasked))
                        ? GMatrixFunctions<FC>::expo_apply(
                              *h_eff->tf, beta, anorm, vr, vi, const_e, iprint,
                              para_rule == nullptr ? nullptr : para_rule->comm)
                        : GMatrixFunctions<FC>::expo_apply(
                              *h_eff, beta, anorm, vr, vi, const_e, iprint,
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
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(energy, norm, nexpo + 1, (size_t)nflop, t.get_time());
    }
};

template <typename S, typename FL>
struct EffectiveFunctions<S, FL,
                          typename enable_if<is_complex<FL>::value>::type> {
    typedef typename GMatrix<FL>::FP FP;
    typedef typename GMatrix<FL>::FC FC;
    // [bra] = ([H_eff] + omega + i eta)^(-1) x [ket]
    // (real gf, imag gf), (nmult, niter), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double>
    greens_function(const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff,
                    FL const_e, LinearSolverTypes solver_type, FL omega, FL eta,
                    const shared_ptr<SparseMatrix<S, FL>> &real_bra,
                    pair<int, int> linear_solver_params, bool iprint = false,
                    FP conv_thrd = 5E-6, int max_iter = 5000,
                    int soft_max_iter = -1,
                    const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(real_bra == nullptr);
        if (solver_type == LinearSolverTypes::Automatic)
            solver_type = LinearSolverTypes::GCROT;
        int nmult = 0, nmultx = 0, niter = 0;
        frame->activate(0);
        Timer t;
        t.get_time();
        GMatrix<FL> mket(h_eff->ket->data, (MKL_INT)h_eff->ket->total_memory,
                         1);
        GMatrix<FL> mbra(h_eff->bra->data, (MKL_INT)h_eff->bra->total_memory,
                         1);
        GDiagonalMatrix<FC> aa(nullptr, 0);
        FC const_x = const_e + omega + FC(0.0, 1.0) * eta;
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
            para_rule->comm->reduce_sum(&nflop, 1, para_rule->comm->root);
        h_eff->tf->opf->seq->cumulative_nflop = 0;
        return make_tuple(gf, make_pair(nmult, niter), (size_t)nflop,
                          t.get_time());
    }
    // [ibra] = (([H_eff] + omega)^2 + eta^2)^(-1) x (-eta [ket])
    // [rbra] = -([H_eff] + omega) (1/eta) [bra]
    // (real gf, imag gf), (nmult, numltp), nflop, tmult
    static tuple<FC, pair<int, int>, size_t, double> greens_function_squared(
        const shared_ptr<EffectiveHamiltonian<S, FL>> &h_eff, FL const_e,
        FL omega, FL eta, const shared_ptr<SparseMatrix<S, FL>> &real_bra,
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
        FC beta, FL const_e, bool iprint = false,
        const shared_ptr<ParallelRule<S>> &para_rule = nullptr) {
        assert(false);
        return make_tuple(0.0, 0.0, 0, (size_t)0, 0.0);
    }
};

} // namespace block2