
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020 Henrik R. Larsson <larsson@caltech.edu>
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

/** Adapted from sciblock2/sci/sciWrapper.cpp
 * by Huanchen Zhai Jul 18, 2021.
 *
 * Author: Henrik R. Larsson <larsson@caltech.edu>
 */

// TODO: openmp is still buggy (on hpc, probably still due to different omp
// versions in py interface)
//      and I don't always see speedups. So I want to disable it.
//      => how do I guard proper openmp library?  What happens on HPC?
//      => dynamic schedule is maybe not the best. Better load balancing. also
//      disable it if #dets is small

#ifdef _OPENMP
#ifdef SCI_USE_OMP
#define _SCI_USE_OMP_ON
#endif
#endif

#include "sci_fock_big_site.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/state_info.hpp"
#include "../core/utils.hpp"
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

namespace block2 {

namespace sci_detail {

template <typename... Args>
std::string strPrintf(const std::string &format, Args... args) {
    auto size = snprintf(nullptr, 0, format.c_str(), args...) +
                1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), static_cast<size_t>(size), format.c_str(), args...);
    return std::string(buf.get(),
                       buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename S> void setMatrixToZero(block2::CSRSparseMatrix<S> &mat) {
    mat.factor = 0.0;
    if (mat.csr_data.size() != 0) {
        for (int i = 0; i < mat.info->n; ++i) {
            // For SCIFockBigSiteExcludeQNs, some QNs are not accessed
            if (mat.csr_data[i]->data == nullptr) {
                mat.csr_data[i]->nnz = 0;
                mat.csr_data[i]->allocate();
                assert(mat.csr_data[i]->data != nullptr);
            }
        }
    }
    mat.deallocate(); // Attention! this is important as this mat will later be
                      // overwritten by one common zero factor
}

/** This is for SCIFockBigSiteExcludeQNs: Some matrices may not be allocated as
 * they are excluded...*/
template <typename S, typename EntryTuple>
void allocateEmptyMatrices(std::vector<EntryTuple> &vec) {
    for (auto &tupl : vec) {
        block2::CSRSparseMatrix<S> &mat = tupl.mat;
        if (mat.factor != 0.0) {
            for (auto &m : mat.csr_data) {
                if (m->data == nullptr) {
                    m->nnz = 0;
                    m->allocate();
                }
            }
        }
    }
}

} // namespace sci_detail

template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::checkOMP() const {
#ifdef _SCI_USE_OMP_ON
    auto explainIt = []() {
        cerr << " This may happen if the program links against several openMP "
                "libraries. "
             << endl;
        cerr << " Check it with 'export OMP_DISPLAY_ENV=\"TRUE\"'" << endl;
        cerr << " This also happens if numpy uses a different openMP library "
                "and is imported *before* block. "
                "Further, set LD_PRELOAD to the libgomp.so you need OR (best "
                "thing); export MKL_THREADING_LAYER=GNU."
             << endl;
        cerr << " This routine has been compiled with _SCI_USE_OMP_ON="
             << _OPENMP << endl;
    };
    std::vector<int> nts;
#pragma omp parallel
    {
        const auto iThread = getThreadID();
#pragma omp critical
        nts.emplace_back(iThread);
    }
    if (nts.size() != ompThreads) {
        cerr << "WRONG NTS SIZE:" << nts.size() << "should be ompThreads"
             << endl;
        explainIt();
        throw std::runtime_error("SCIFockBigSite:checkOMP: wrong nts size");
    }
    std::sort(nts.begin(), nts.end());
    for (int i = 0; i < ompThreads; ++i) {
        if (nts.at(i) != i) {
            cerr << "WRONG NTS" << endl;
            explainIt();
            throw std::runtime_error("SCIFockBigSite:checkOMP: wrong nts");
        }
    }
    std::vector<std::vector<int>> calledThreads(ompThreads);
    const int nLoop = ompThreads * 10;
#pragma omp parallel default(shared)
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < nLoop; ++i) {
            calledThreads.at(omp_get_thread_num()).emplace_back(i);
        }
#pragma omp critical
        if (omp_get_thread_num() == 0) {
            int iThread = 0;
            for (int i = 0; i < calledThreads.size(); ++i) {
                iThread += calledThreads[i].size();
            }
            if (iThread != nLoop) {
                cerr << "SCIFockBigSite::checkOMP: OMP has some weird bug! i "
                        "loop "
                        "is not properly distributed."
                     << endl;
                explainIt();
                    cerr <<"ithread: called i:");
                    for (int i = 0; i < calledThreads.size(); ++i) {
                        cerr << i << ":" << calledThreads[i] << endl;
                    }
                    throw std::runtime_error(
                        "SCIFockBigSite:checkOMP: guided problems");
            }
        }
    }
#endif
}

template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::get_site_ops(
    uint16_t m,
    unordered_map<shared_ptr<OpExpr<S>>, shared_ptr<SparseMatrix<S>>> &ops)
    const {
    const int iSite = m;
    int ii, jj; // spin orbital indices
    // For optimization purposes (parallelization + orbital loop elsewhere
    // I want to collect operators of same name and quantum numbers
    // A,AD,B would be for a small site => I assume that it is not big and
    // have not yet optimized it Same for C and D, which is fast
    unordered_map<S, std::vector<entryTuple2>> opsQ, opsP, opsPD;
    unordered_map<S, std::vector<entryTuple1>> opsR, opsRD;
    for (auto &p : ops) {
        shared_ptr<OpElement<S>> pop =
            dynamic_pointer_cast<OpElement<S>>(p.first);
        OpElement<S> &op = *pop;
        auto pmat = make_shared<CSRSparseMatrix<S>>();
        auto &mat = *pmat;
        p.second = pmat;
        // ATTENTION vv if you change allocation, you need to change the
        //                      deallocation routine in MPOQCSCI
        // Also, the CSR stuff is more complicated and I will do the actual
        // allocation
        //      of the individual matrices in the fillOp* routines.
        //      So here, the CSRMatrices are only initialized (i.e., their
        //      sizes are set)
        mat.initialize(BigSite<S>::find_site_op_info(op.q_label));
        const auto &delta_qn = op.q_label;
        if (false and op.name == OpNames::R) { // DEBUG
            cout << "m == " << iSite << "allocate" << op.name << "s"
                 << (int)op.site_index[0] << "," << (int)op.site_index[1]
                 << "ss" << (int)op.site_index.s(0) << (int)op.site_index.s(1)
                 << endl;
            cout << "q_label:" << op.q_label << endl;
        }
        // get orbital indices
        ii = -1;
        jj = -1; // debug
        switch (op.name) {
        case OpNames::C:
        case OpNames::D:
        case OpNames::R:
        case OpNames::RD:
            ii = 2 * op.site_index[0] + op.site_index.s(0);
            break;
        case OpNames::A:
        case OpNames::AD:
        case OpNames::B:
        case OpNames::P:
        case OpNames::PD:
        case OpNames::Q:
            ii = 2 * op.site_index[0] + op.site_index.s(0);
            jj = 2 * op.site_index[1] + op.site_index.s(1);
            break;
        default:
            break;
        }
        switch (op.name) {
        case OpNames::I:
            fillOp_I(mat);
            break;
        case OpNames::N:
            fillOp_N(mat);
            break;
        case OpNames::NN:
            fillOp_NN(mat);
            break;
        case OpNames::H:
            fillOp_H(mat);
            break;
        case OpNames::C:
            fillOp_C(delta_qn, mat, ii);
            break;
        case OpNames::D:
            fillOp_D(delta_qn, mat, ii);
            break;
        case OpNames::R:
            opsR[delta_qn].emplace_back(mat, ii);
            break;
        case OpNames::RD:
            opsRD[delta_qn].emplace_back(mat, ii);
            break;
        case OpNames::A:
            fillOp_A(delta_qn, mat, ii, jj);
            break;
        case OpNames::AD:
            fillOp_AD(delta_qn, mat, ii, jj);
            break;
        case OpNames::B:
            fillOp_B(delta_qn, mat, ii, jj);
            break;
        case OpNames::P:
            opsP[delta_qn].emplace_back(mat, ii, jj);
            break;
        case OpNames::PD:
            opsPD[delta_qn].emplace_back(mat, ii, jj);
            break;
        case OpNames::Q:
            opsQ[delta_qn].emplace_back(mat, ii, jj);
            break;
        default:
            assert(false);
        }
    }
    for (auto &pairs : opsR) {
        fillOp_R(pairs.first, pairs.second);
        pairs.second.clear();
        pairs.second.shrink_to_fit();
    }
    for (auto &pairs : opsRD) {
        fillOp_RD(pairs.first, pairs.second);
        pairs.second.clear();
        pairs.second.shrink_to_fit();
    }
    for (auto &pairs : opsP) {
        fillOp_P(pairs.first, pairs.second);
        pairs.second.clear();
        pairs.second.shrink_to_fit();
    }
    for (auto &pairs : opsPD) {
        fillOp_PD(pairs.first, pairs.second);
        pairs.second.clear();
        pairs.second.shrink_to_fit();
    }
    for (auto &pairs : opsQ) {
        fillOp_Q(pairs.first, pairs.second);
        pairs.second.clear();
        pairs.second.shrink_to_fit();
    }
    finalize();
}

template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::finalize() const {
    // TODO: fragSpace could also be deleted, or at least the unordered_map
    //                        #counts  time  <time>   sparsity  dense sparse
    const std::string fStr =
        "%10d      %12.6f  %.2e  %.3f %9d  %9d  %9d       %6.3f";
    if (not verbose) {
        return;
    }
    cout << "#----------------------------------" << endl;
    cout << "# Done with SCIFockBigSite Op Creation" << endl;
    cout << "# sparsity = #zeros/#tot within QN block" << endl;
    cout << "# #dense/sparse: Per QN" << endl;
    cout << "# #zero: Per operator, not per QN" << endl;
    cout << "# Operator  #counts        tot.time/s <time>  <sparsity>   #dense "
            "     #sparse  #zero mats  overall memory/MB"
         << endl;
    cout << "#  H  "
         << sci_detail::strPrintf(fStr, totCountsH, timeH, timeH / totCountsH,
                                  summSparsityH / qnCountsH, numDenseH,
                                  numSparseH, numZeroH, usedMemH / 1e6)
         << endl;
    cout << "#  C  "
         << sci_detail::strPrintf(fStr, totCountsC, timeC, timeC / totCountsC,
                                  summSparsityC / qnCountsC, numDenseC,
                                  numSparseC, numZeroC, usedMemC / 1e6)
         << endl;
    if (totCountsD > 0)
        cout << "#  D  "
             << sci_detail::strPrintf(fStr, totCountsD, timeD,
                                      timeD / totCountsD,
                                      summSparsityD / qnCountsD, numDenseD,
                                      numSparseD, numZeroD, usedMemD / 1e6)
             << endl;
    cout << "#  R  "
         << sci_detail::strPrintf(fStr, totCountsR, timeR, timeR / totCountsR,
                                  summSparsityR / qnCountsR, numDenseR,
                                  numSparseR, numZeroR, usedMemR / 1e6)
         << endl;
    if (totCountsRD > 0)
        cout << "#  RD "
             << sci_detail::strPrintf(fStr, totCountsRD, timeRD,
                                      timeRD / totCountsRD,
                                      summSparsityRD / qnCountsRD, numDenseRD,
                                      numSparseRD, numZeroRD, usedMemRD / 1e6)
             << endl;
    if (isRight) {
        cout << "#  P  "
             << sci_detail::strPrintf(fStr, totCountsP, timeP,
                                      timeP / totCountsP,
                                      summSparsityP / qnCountsP, numDenseP,
                                      numSparseP, numZeroP, usedMemP / 1e6)
             << endl;
        if (totCountsPD > 0)
            cout << "#  PD "
                 << sci_detail::strPrintf(
                        fStr, totCountsPD, timePD, timePD / totCountsPD,
                        summSparsityPD / qnCountsPD, numDensePD, numSparsePD,
                        numZeroPD, usedMemPD / 1e6)
                 << endl;
        cout << "#  Q  "
             << sci_detail::strPrintf(fStr, totCountsQ, timeQ,
                                      timeQ / totCountsQ,
                                      summSparsityQ / qnCountsQ, numDenseQ,
                                      numSparseQ, numZeroQ, usedMemQ / 1e6)
             << endl;
    } else {
        if (totCountsA > 0)
            cout << "#  A  "
                 << sci_detail::strPrintf(fStr, totCountsA, timeA,
                                          timeA / totCountsA,
                                          summSparsityA / qnCountsA, numDenseA,
                                          numSparseA, numZeroA, usedMemA / 1e6)
                 << endl;
        cout << "#  AD "
             << sci_detail::strPrintf(fStr, totCountsAD, timeAD,
                                      timeAD / totCountsAD,
                                      summSparsityAD / qnCountsAD, numDenseAD,
                                      numSparseAD, numZeroAD, usedMemAD / 1e6)
             << endl;
        cout << "#  B  "
             << sci_detail::strPrintf(fStr, totCountsB, timeB,
                                      timeB / totCountsB,
                                      summSparsityB / qnCountsB, numDenseB,
                                      numSparseB, numZeroB, usedMemB / 1e6)
             << endl;
    }
    auto totalMem = usedMemH + usedMemC + usedMemD + usedMemR + usedMemRD +
                    usedMemP + usedMemPD + usedMemQ + usedMemA + usedMemAD +
                    usedMemB;
    cout << "# total memory=" << totalMem / 1e9 << "GB" << endl;
    auto totalTime = timeH + timeC + timeD + timeR + timeRD + timeP + timePD +
                     timeQ + timeA + timeAD + timeB;
    cout << "# total time=" << totalTime << "s" << endl;
    cout << "#----------------------------------" << endl;
}

template <typename S>
SCIFockBigSite<S, typename S::is_sz_t>::SCIFockBigSite(
    int nOrb, int nOrbThis, bool isRight,
    const std::shared_ptr<block2::FCIDUMP> &fcidump,
    const std::vector<uint8_t> &orbsym, int nMaxAlphaEl, int nMaxBetaEl,
    int nMaxEl, const vector<vector<int>> &poccs, bool verbose)
    : BigSite<S>(nOrbThis), nOrbOther{nOrb - nOrbThis}, nOrbThis{nOrbThis},
      nOrb{nOrb}, isRight{isRight}, nMaxAlphaEl{nMaxAlphaEl},
      nMaxBetaEl{nMaxBetaEl}, nMaxEl{nMaxEl}, verbose{verbose},
      ints2(fcidump, nOrbOther, nOrbThis, isRight),
      ints1(fcidump, nOrbOther, nOrbThis, isRight) {
    vector<vector<int>> occs = poccs;
    if (nMaxEl == -999 and occs.size() == 0) {
        return; // dummy
    }
    if (verbose)
        cout << "#--- init SCIFockBigSite --- " << endl;
    // check fragmentDetLen
    const auto effDetLen = SCIFockDeterminant::getEffDetLen(2 * nOrbThis);
    if (effDetLen > sci_detail::fragmentDetLen)
        throw std::runtime_error(
            "SCIFockBigSite: For " + Parsing::to_string(nOrbThis) +
            "external orbitals, you need a determinant length of" +
            Parsing::to_string(effDetLen) +
            ". Change it in SCIFockDeterminant::fragmentDetLen");
#ifdef NDEBUG
    if (verbose)
        cout << "# ASSERTIONS ARE DISABLED!" << endl;
#else
    if (verbose)
        cout << "# ASSERTIONS ARE ENABLED!" << endl;
#endif
#ifdef _SCI_USE_OMP_ON
    setOmpThreads();
    if (verbose)
        cout << "# WITH OMP using " << ompThreads
             << "threads in operator creation" << endl;
#else
    setOmpThreads(1);
    if (verbose)
        cout << "# WITHOUT OMP" << endl;
#endif
    bool hasSymmetry = false;
    for (auto s : orbsym) {
        if (s != 0) {
            hasSymmetry = true;
            break;
        }
    }
    if (hasSymmetry and verbose) {
        cout << "#- with symmetry -" << endl;
    }
    const auto sOrbThis = 2 * nOrbThis; // spin orbs
    if (nMaxAlphaEl > nOrbThis or nMaxBetaEl > nOrbThis)
        throw std::runtime_error(
            "SCIFockBigSite: nMaxElpha/BetaEl=" +
            Parsing::to_string(nMaxAlphaEl) + Parsing::to_string(nMaxBetaEl) +
            "but nOrbThis=" + Parsing::to_string(nOrbThis));
    if (verbose) {
        if (hasSymmetry) {
            cout << "# nAlpha nBeta -> N  2Sz pg | nStates" << endl;
        } else {
            cout << "# nAlpha nBeta -> N  2Sz | nStates" << endl;
        }
    }

    auto &fSpace = fragSpace;
    const auto offset = isRight ? nOrbOther : 0;
    const auto getPgSym = [&orbsym, offset](const SCIFockDeterminant &a) {
        // Could be improved by checking whether its singly or doubly occupied
        auto occA = a.getClosed();
        uint16_t symA = 0;
        for (auto &o : occA) {
            assert(offset + o / 2 < orbsym.size());
            symA =
                symA ^ orbsym[offset + o / 2]; //  / 2 as spin orb -> spat orb
        }
        return symA;
    };
    const auto getSym = [&getPgSym](const SCIFockDeterminant &a) {
        const auto qN = a.nAlphaEl + a.nBetaEl;
        const auto q2S = a.nAlphaEl - a.nBetaEl;
        return S(qN, q2S, getPgSym(a));
    };
    std::string fStr;
    if (hasSymmetry) {
        // say("#", nAlpha, nBeta, "        ->", nAlpha + nBeta, nAlpha - nBeta,
        // lastPgSym, "|", ii - sStart);
        fStr = "# %3d %3d  ->   %3d  %3d  %1d |  %4d ";
    } else {
        // say("#", nAlpha, nBeta, "        ->", nAlpha + nBeta, nAlpha - nBeta,
        // "|", ii - sStart);
        fStr = "# %3d %3d  ->   %3d  %3d |  %4d ";
    }
    // left thawed space
    if (occs.size() == 0 && nMaxEl <= 0)
        occs = ras_space(isRight, nOrbThis, abs(nMaxBetaEl), abs(nMaxBetaEl),
                         abs(nMaxEl));
    // CI space
    if (occs.size() == 0) {
        // Prelim build up of quantum numbers: Make them sorted (*without* pg
        // symmetry (handled within loop)!)
        std::vector<S> prelimQNs;
        for (int nBeta = 0; nBeta <= nMaxBetaEl; ++nBeta) {
            for (int nAlpha = 0; nAlpha <= nMaxAlphaEl; ++nAlpha) {
                if (nAlpha + nBeta > nMaxEl or nAlpha + nBeta > sOrbThis) {
                    continue;
                }
                const auto qN = nAlpha + nBeta;
                const auto q2S = nAlpha - nBeta;
                prelimQNs.emplace_back(qN, q2S, 0);
            }
        }
        std::sort(prelimQNs.begin(), prelimQNs.end());
        // Now, do the actual loop
        for (auto &pQN : prelimQNs) {
            const auto qN = pQN.n();
            const auto q2S = pQN.twos();
            const int nAlpha = (qN + q2S) / 2;
            const int nBeta = qN - nAlpha;
            assert(nAlpha + nBeta == qN);
            assert(nAlpha - nBeta == q2S);

            const auto sInit = fSpace.size();
            SCIFockDeterminant::generateFCIfragmentSpace(nAlpha, nBeta,
                                                         sOrbThis, fSpace);
            if (hasSymmetry) { // sort symmetry numbers and iterate within their
                               // sectors
                std::stable_sort(
                    fSpace.begin() + sInit, fSpace.end(),
                    [&getPgSym, qN, q2S](const SCIFockDeterminant &a,
                                         const SCIFockDeterminant &b) {
                        return S(qN, q2S, getPgSym(a)) <
                               S(qN, q2S, getPgSym(b));
                    });
                // now iterate within their sectors
                auto lastPgSym = getPgSym(fSpace[sInit]);
                auto sStart = sInit;
                for (int ii = sInit; ii < fSpace.size(); ++ii) {
                    auto newPgSym = getPgSym(fSpace[ii]);
                    if (lastPgSym != newPgSym) {
                        quantumNumbers.emplace_back(nAlpha + nBeta,
                                                    nAlpha - nBeta, lastPgSym);
                        quantumNumberToIdx.emplace(quantumNumbers.back(),
                                                   quantumNumbers.size() - 1);
                        offsets.emplace_back(sStart, ii);
                        // say("#", nAlpha, nBeta, "        ->", nAlpha + nBeta,
                        // nAlpha - nBeta, lastPgSym,
                        //   "|", ii - sStart);
                        if (verbose)
                            cout << sci_detail::strPrintf(
                                        fStr, nAlpha, nBeta, nAlpha + nBeta,
                                        nAlpha - nBeta, lastPgSym, ii - sStart)
                                 << endl;
                        lastPgSym = newPgSym;
                        sStart = ii;
                    }
                }
                quantumNumbers.emplace_back(nAlpha + nBeta, nAlpha - nBeta,
                                            lastPgSym);
                quantumNumberToIdx.emplace(quantumNumbers.back(),
                                           quantumNumbers.size() - 1);
                auto sNow = fSpace.size();
                offsets.emplace_back(sStart, sNow);
                // say("#", nAlpha, nBeta, "        ->", nAlpha + nBeta, nAlpha
                // - nBeta, lastPgSym, "|", sNow - sStart);
                if (verbose)
                    cout << sci_detail::strPrintf(
                                fStr, nAlpha, nBeta, nAlpha + nBeta,
                                nAlpha - nBeta, lastPgSym, sNow - sStart)
                         << endl;
            } else {
                quantumNumbers.emplace_back(nAlpha + nBeta, nAlpha - nBeta, 0);
                quantumNumberToIdx.emplace(quantumNumbers.back(),
                                           quantumNumbers.size() - 1);
                auto sNow = fSpace.size();
                offsets.emplace_back(sInit, sNow);
                auto sAdded = sNow - sInit;
                // say("#", nAlpha, nBeta, "        ->", nAlpha + nBeta, nAlpha
                // - nBeta, "|", sAdded);
                if (verbose)
                    cout << sci_detail::strPrintf(fStr, nAlpha, nBeta,
                                                  nAlpha + nBeta,
                                                  nAlpha - nBeta, sAdded)
                         << endl;
            }
        }
    } else {
        if (verbose)
            cout << "# based on nOccs" << endl;
        int lastN, lastM, nAlpha, nBeta, lastA, lastB, N, M, lastpg, pg;
        // sort them first
        bool needSort = false;
        std::vector<size_t> sortIdx;
        {
            std::vector<S> prelimQNs;
            for (size_t ii = 0; ii < occs.size(); ++ii) {
                SCIFockDeterminant curr(sOrbThis, occs[ii]);
                nAlpha = curr.Nalpha();
                nBeta = curr.Nbeta();
                N = nAlpha + nBeta;
                M = nAlpha - nBeta;
                pg = hasSymmetry ? getPgSym(curr) : 0;
                prelimQNs.emplace_back(N, M, pg);
            }
            sortIdx.resize(prelimQNs.size());
            std::iota(sortIdx.begin(), sortIdx.end(), static_cast<size_t>(0));
            std::stable_sort(sortIdx.begin(), sortIdx.end(),
                             [&prelimQNs](size_t i, size_t j) {
                                 return prelimQNs[i] < prelimQNs[j];
                             });
            for (size_t ii = 0; ii < prelimQNs.size(); ++ii) {
                if (sortIdx[ii] != ii) {
                    needSort = true;
                    break;
                }
            }
            if (needSort) {
                if (verbose)
                    cout << "--- ATTENTION! SORT INPUT nOccs! ---" << endl;
            }
        }
        nMaxBetaEl = 0;
        nMaxAlphaEl = 0;
        nMaxEl = 0;
        fSpace.reserve(occs.size());
        size_t sInit = 0;
        for (size_t ii = 0; ii < occs.size(); ++ii) {
            if (needSort) {
                fSpace.emplace_back(sOrbThis, occs[sortIdx[ii]]);
            } else {
                fSpace.emplace_back(sOrbThis, occs[ii]);
            }

            nAlpha = fSpace[ii].Nalpha();
            nBeta = fSpace[ii].Nbeta();
            N = nAlpha + nBeta;
            M = nAlpha - nBeta;
            nMaxBetaEl = std::max(nBeta, nMaxBetaEl);
            nMaxAlphaEl = std::max(nAlpha, nMaxAlphaEl);
            nMaxEl = std::max(nMaxEl, N);
            pg = hasSymmetry ? getPgSym(fSpace[ii]) : 0;
            if (ii == 0) {
                lastN = N;
                lastM = M;
                lastA = nAlpha;
                lastB = nBeta, lastpg = pg;
            }
            if (N != lastN or M != lastM or pg != lastpg) {
                auto it = quantumNumberToIdx.find({N, M, pg});
                if (it != quantumNumberToIdx.end()) {
                    throw std::runtime_error("QNs not sorted");
                }
                auto sNow = fSpace.size() - 1;
                offsets.emplace_back(sInit, sNow);
                quantumNumbers.emplace_back(lastN, lastM, lastpg);
                quantumNumberToIdx.emplace(quantumNumbers.back(),
                                           quantumNumbers.size() - 1);
                auto sAdded = sNow - sInit;
                sInit = ii;
                if (hasSymmetry) {
                    // say("#", lastA, lastB, "->", lastN, lastM, lastpg, "|",
                    // sAdded);
                    if (verbose)
                        cout << sci_detail::strPrintf(fStr, lastA, lastB, lastN,
                                                      lastM, lastpg, sAdded)
                             << endl;
                } else {
                    // say("#", lastA, lastB, "->", lastN, lastM, "|", sAdded);
                    if (verbose)
                        cout << sci_detail::strPrintf(fStr, lastA, lastB, lastN,
                                                      lastM, sAdded)
                             << endl;
                }
                lastN = N;
                lastM = M;
                lastA = nAlpha;
                lastB = nBeta;
                lastpg = pg;
            }
            // say(ii,":", fSpace[ii]);//.getClosed());
        }
        auto sNow = fSpace.size();
        // last one
        offsets.emplace_back(sInit, sNow);
        auto it = quantumNumberToIdx.find({N, M, pg});
        if (it != quantumNumberToIdx.end()) {
            throw std::runtime_error("QNs not sorted");
        }
        quantumNumbers.emplace_back(N, M, pg);
        quantumNumberToIdx.emplace(quantumNumbers.back(),
                                   quantumNumbers.size() - 1);
        auto sAdded = sNow - sInit;
        // say("#", nAlpha, nBeta, "->", nAlpha + nBeta, nAlpha - nBeta, pg,
        // "|",  sAdded);
        if (verbose) {
            if (hasSymmetry) {
                cout << sci_detail::strPrintf(fStr, nAlpha, nBeta,
                                              nAlpha + nBeta, nAlpha - nBeta,
                                              pg, sAdded)
                     << endl;
            } else {
                cout << sci_detail::strPrintf(fStr, nAlpha, nBeta,
                                              nAlpha + nBeta, nAlpha - nBeta,
                                              sAdded)
                     << endl;
            }
            cout << "# max El: Alpha, Beta, Tot=" << nMaxAlphaEl << " "
                 << nMaxBetaEl << " " << nMaxEl << endl;
            cout << "sizes:" << quantumNumbers.size() << " " << offsets.size()
                 << endl;
        }
    }
    {
        // Just make sure that the states really are sorted according to the
        // internal sorting order
        std::vector<size_t> idx(quantumNumbers.size());
        std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));
        std::sort(idx.begin(), idx.end(), [this](size_t i, size_t j) {
            return quantumNumbers[i] < quantumNumbers[j];
        });
        bool needSort = false;
        for (size_t ii = 0; ii < quantumNumbers.size(); ++ii) {
            if (idx[ii] != ii) {
                needSort = true;
                break;
            }
        }
        if (needSort) {
            if (verbose)
                cout << "--- sort quantum numbers" << endl;
            {
                const auto fSpaceOld = fSpace;
                const auto quantumNumbersOld = quantumNumbers;
                const auto offsetsOld = offsets;
                size_t iDet = 0;
                quantumNumberToIdx.clear();
                offsets.resize(0);
                for (size_t ii = 0; ii < quantumNumbers.size(); ++ii) {
                    quantumNumbers[ii] = quantumNumbersOld[idx[ii]];
                    quantumNumberToIdx.emplace(
                        std::move(quantumNumbersOld[idx[ii]]), ii);
                    auto o1Old = offsetsOld[idx[ii]].first;
                    auto o2Old = offsetsOld[idx[ii]].second;
                    auto size = o2Old - o1Old;
                    offsets.emplace_back(iDet, iDet + size);
                    for (size_t jDet = iDet; jDet < iDet + size; ++jDet) {
                        assert(jDet < fSpace.size());
                        fSpace[jDet] = fSpaceOld[o1Old];
                        ++o1Old;
                    }
                    iDet += size;
                    const auto nAlpha = fSpace[iDet - 1].nAlphaEl;
                    const auto nBeta = fSpace[iDet - 1].nBetaEl;
                    const auto &lastSym = quantumNumbers[ii];
                    // say("#", nAlpha, nBeta, "        ->", lastSym.n(),
                    // lastSym.twos(), lastSym.pg(),
                    //   "|", size);
                    if (verbose) {
                        if (hasSymmetry) {
                            cout << sci_detail::strPrintf(
                                        fStr, nAlpha, nBeta, lastSym.n(),
                                        lastSym.twos(), lastSym.pg(), size)
                                 << endl;
                        } else {
                            cout << sci_detail::strPrintf(fStr, nAlpha, nBeta,
                                                          lastSym.n(),
                                                          lastSym.twos(), size)
                                 << endl;
                        }
                    }
                }
            }
            if (verbose)
                cout << "------------------------ DONE" << endl;
        }
    }
    quantumNumbers.shrink_to_fit();
    offsets.shrink_to_fit();

    nDet = fSpace.size();
    if (verbose)
        cout << "# nDet=" << nDet << endl;
    fragIndexMap.reserve(fSpace.size());
    for (size_t i = 0; i < fSpace.size(); ++i) {
        const auto &det = fSpace[i];
        assert(det.nAlphaEl >= 0 and det.EffDetLen > 0);
        fragIndexMap.insert({det, i});
    }
    if (nOrbThis > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("nOrbThis too big for intT");
    }
    // hz: set big site basis
    shared_ptr<StateInfo<S>> b = make_shared<StateInfo<S>>();
    b->allocate((int)quantumNumbers.size());
    memcpy(b->quanta, quantumNumbers.data(), b->n * sizeof(S));
    for (int iq = 0; iq < (int)quantumNumbers.size(); iq++)
        b->n_states[iq] = offsets[iq].second - offsets[iq].first;
    b->sort_states();
    BigSite<S>::basis = b;
    // hz: set big site op_infos
    shared_ptr<VectorAllocator<uint32_t>> i_alloc =
        make_shared<VectorAllocator<uint32_t>>();
    map<S, shared_ptr<SparseMatrixInfo<S>>> info;
    info[S(0)] = nullptr;
    for (auto ipg : orbsym) {
        for (int n = -1; n <= 1; n += 2)
            for (int s = -3; s <= 3; s += 2)
                info[S(n, s, ipg)] = nullptr;
        for (auto jpg : orbsym)
            for (int n = -2; n <= 2; n += 2)
                for (int s = -4; s <= 4; s += 2)
                    info[S(n, s, ipg ^ jpg)] = nullptr;
    }
    for (auto &p : info) {
        p.second = make_shared<SparseMatrixInfo<S>>(i_alloc);
        p.second->initialize(*b, *b, p.first, p.first.is_fermion());
    }
    BigSite<S>::op_infos = vector<pair<S, shared_ptr<SparseMatrixInfo<S>>>>(
        info.begin(), info.end());
}

///////////////////////////////////////////////////////////////

template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_I(
    BLSparseMatrix &mat) const {
    const auto qSize = static_cast<int>(quantumNumbers.size());
    sci_detail::COOSparseMat<double> smat;
    if (!excludeQNs) {
        for (int iQ = 0; iQ < qSize; ++iQ) {
            auto &sym = quantumNumbers[iQ];
            assert(mat[sym].m == mat[sym].n);
            if (mat[sym].size() <
                sparsityStart) { // Dense; this is important as
                                 // there are also 1 x 1 matrices
                mat[sym].nnz = mat[sym].size();
                mat[sym].alloc = make_shared<VectorAllocator<double>>();
                mat[sym].allocate();
                auto o1 = offsets[iQ].first;
                auto o2 = offsets[iQ].second;
                // auto [o1, o2] = offsets[iQ];
                const auto siz = o2 - o1;
                for (int ii = 0; ii < siz; ++ii) {
                    mat[sym].dense_ref()(ii, ii) = 1.0;
                }
            } else {
                smat.resize(mat[sym].m, mat[sym].n);
                smat.reserve(mat[sym].n);
                auto o1 = offsets[iQ].first;
                auto o2 = offsets[iQ].second;
                // auto [o1, o2] = offsets[iQ];
                const auto siz = o2 - o1;
                for (int ii = 0; ii < siz; ++ii) {
                    smat.insert(ii, ii) = 1.0;
                }
                smat.fillCSR(mat[sym]);
            }
        }
    } else {
        // The code only works if *all* matrices mat[sym] are allocated,
        // but here, we omit some. Allocation of matrices omitted is done with
        // nnz=0 via inBra See comment at
        // https://github.com/h-larsson/sciblock2/commit/830d7986525243a775e44b1073d09abc028520ff#comments
        std::vector<bool> inBra(qSize, false);
        for (const auto iQ : qnIdxBraI) {
            assert(iQ < quantumNumbers.size());
            auto &sym = quantumNumbers[iQ];
            assert(mat[sym].m == mat[sym].n);
            if (not idxInKet(iQ, qnIdxKetI)) {
                continue;
            }
            inBra[iQ] = true;
            if (mat[sym].size() <
                sparsityStart) { // Dense; this is important as there are also 1
                                 // x 1 matrices
                mat[sym].nnz = mat[sym].size();
                mat[sym].alloc = make_shared<VectorAllocator<double>>();
                mat[sym].allocate();
                auto o1 = offsets[iQ].first;
                auto o2 = offsets[iQ].second;
                // auto [o1, o2] = offsets[iQ];
                const auto siz = o2 - o1;
                for (int ii = 0; ii < siz; ++ii) {
                    mat[sym].dense_ref()(ii, ii) = 1.0;
                }
            } else {
                smat.resize(mat[sym].m, mat[sym].n);
                smat.reserve(mat[sym].n);
                auto o1 = offsets[iQ].first;
                auto o2 = offsets[iQ].second;
                // auto [o1, o2] = offsets[iQ];
                const auto siz = o2 - o1;
                for (int ii = 0; ii < siz; ++ii) {
                    smat.insert(ii, ii) = 1.0;
                }
                smat.fillCSR(mat[sym]);
            }
        }
        for (int iQ = 0; iQ < qSize; ++iQ) {
            if (not inBra[iQ]) {
                const auto &sym = quantumNumbers[iQ]; // ket qn number
                smat.resize(mat[sym].m, mat[sym].n);
                smat.fillCSR(mat[sym]);
            }
        }
    }
}
template <typename S>
template <bool Symmetrize>
size_t SCIFockBigSite<S, typename S::is_sz_t>::fillCoeffs(
    sci_detail::COOSparseMat<double> &cooMat,
    sci_detail::DenseMat<TripletVec> &coeffs, const int iRow,
    block2::CSRMatrixRef &mat, double &summSparsity, size_t &numSparse,
    size_t &numDense, size_t &usedMem) const {
    assert(coeffs.m == ompThreads);
    assert(sparsityStart > 1);
    size_t nCount = 0;
    TripletVec symmEls;
    if /* constexpr */ (Symmetrize) {
        symmEls.reserve(coeffs(0, iRow).size());
    }
    for (int ii = 0; ii < ompThreads; ++ii) {
        if /* constexpr */ (not Symmetrize) {
            nCount += coeffs(ii, iRow).size();
        } else { // Bit costly, but only needed for H
            for (const auto &tripl : coeffs(ii, iRow)) {
                ++nCount;
                if (tripl.first.first != tripl.first.second) {
                    symmEls.emplace_back(
                        make_pair(tripl.first.second, tripl.first.first),
                        tripl.second);
                    ++nCount;
                }
            }
        }
    }
    auto sparsity = (mat.size() - nCount) / static_cast<double>(mat.size());
    summSparsity += sparsity;
    const auto isSparse =
        sparsity > sparsityThresh and mat.size() >= sparsityStart;
    if (not isSparse) {
        mat.nnz = mat.size();
        mat.alloc = make_shared<VectorAllocator<double>>();
        mat.allocate();
        memset(mat.data, 0.0, mat.size() * sizeof(double));
        auto matRef = mat.dense_ref();
        for (int ii = 0; ii < ompThreads; ++ii) {
            for (const auto &tripl : coeffs(ii, iRow)) {
                assert(tripl.first.first < mat.m);
                assert(tripl.first.second < mat.n);
                matRef(tripl.first.first, tripl.first.second) = tripl.second;
                if /* constexpr */ (Symmetrize) {
                    assert(mat.m == mat.n);
                    if (tripl.first.first != tripl.first.second) {
                        matRef(tripl.first.second, tripl.first.first) =
                            tripl.second;
                    }
                }
            }
            coeffs(ii, iRow).resize(0); // Prepare for next round
        }
    } else {
        // For now, just append all coeffs to first one...
        coeffs(0, iRow).reserve(nCount);
        for (int ii = 1; ii < ompThreads; ++ii) {
            // TODO vv move is better...
            if (coeffs(ii, iRow).size() > 0) {
                coeffs(0, iRow).insert(coeffs(0, iRow).end(),
                                       coeffs(ii, iRow).begin(),
                                       coeffs(ii, iRow).end());
                coeffs(ii, iRow).resize(0);
            }
        }
        if /* constexpr */ (Symmetrize) {
            coeffs(0, iRow).insert(coeffs(0, iRow).end(), symmEls.begin(),
                                   symmEls.end());
            symmEls.resize(0);
            symmEls.shrink_to_fit();
        }
        cooMat.resize(mat.m, mat.n);
        cooMat.setFromTriplets(coeffs(0, iRow));
        cooMat.fillCSR(mat);
        coeffs(0, iRow).resize(0);
    }
    if (isSparse) {
        ++numSparse;
        //                                     vv bit of overestimation for CSR
        //                                     format
        usedMem += nCount * (sizeof(double) +
                             2 * sizeof(int)); // int in native implementation
    } else {
        ++numDense;
        usedMem += mat.size() * sizeof(double);
    }
    return nCount;
}

template <typename S>
std::vector<std::pair<int, int>>
SCIFockBigSite<S, typename S::is_sz_t>::getQNpairs(const BLSparseMatrix &mat,
                                                   const S &deltaQN) const {
    constexpr bool verbose = false;
    std::vector<std::pair<int, int>> pairs;
    const auto qSize = static_cast<int>(quantumNumbers.size());
    for (int iQ = 0; iQ < qSize; ++iQ) {
        const auto &sym = quantumNumbers[iQ]; // ket qn number
        { // find out whether sym exist in ket quantumnumbers of mat
            auto ptr = mat.info->find_state(sym);
            if (ptr < 0) { // nope
                continue;
            }
        }
        S qnBra{sym.n() + deltaQN.n(), sym.twos() + deltaQN.twos(),
                sym.pg() ^ deltaQN.pg()};
        auto jQit = quantumNumberToIdx.find(qnBra);
        if (jQit == quantumNumberToIdx.end()) {
            if /* constexpr */ (verbose) {
                cout << "????????????????cannot find qnBra" << qnBra << endl;
            }
            continue;
        }
        const auto jQ = jQit->second;
        pairs.emplace_back(iQ, jQ);
    }
    return pairs;
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_N(
    BLSparseMatrix &mat) const {
    throw std::runtime_error("not implemented");
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_NN(
    BLSparseMatrix &mat) const {
    throw std::runtime_error("not implemented");
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_H(
    BLSparseMatrix &mat) const {
    checkOMP();
    Timer clock;
    clock.get_time();
    const auto qnPairs = getQNpairsH(mat, {0, 0, 0});
    const auto qnSiz = qnPairs.size();
    sci_detail::DenseMat<TripletVec> allOpCoeffs(ompThreads, 1);
    size_t nonZeros = 0;
    sci_detail::COOSparseMat<double> smat;
    for (int itQN = 0; itQN < qnSiz; ++itQN) {
        const auto ketQN = qnPairs[itQN].first;
        const auto braQN = qnPairs[itQN].second;
        const auto &sym = quantumNumbers[ketQN];
        const auto o1Ket =
            offsets[ketQN].first; // openMP cant handle auto [...] -.-
        const auto o2Ket =
            offsets[ketQN].second; // openMP cant handle auto [...] -.-
        const int sizKet = o2Ket - o1Ket;
        const auto o1Bra =
            offsets[braQN].first; // openMP cant handle auto [...] -.-
        const auto o2Bra =
            offsets[braQN].second; // openMP cant handle auto [...] -.-
        const auto sizBra = o2Bra - o1Bra;
#ifdef _SCI_USE_OMP_ON
#pragma omp parallel for collapse(2) default(shared) schedule(dynamic)
#endif
        for (int ii = 0; ii < sizBra; ++ii) {
            for (int jj = 0; jj < sizKet; ++jj) {
                const auto iThread = getThreadID();
                const auto iD = o1Bra + ii;
                const auto &bra = fragSpace[iD];
                const auto jD = o1Ket + jj;
                if (iD == jD) { // Diagonal
                    allOpCoeffs(iThread, 0)
                        .emplace_back(
                            make_pair(ii, jj),
                            bra.Energy(ints1, ints2, bra.getClosed()));
                    continue;
                }
                if (iD < jD) {
                    continue; // ATTENTION: Exploit symmetry
                }
                const auto &ket = fragSpace[jD];
                assert(bra.nEl() == ket.nEl());
                // FILL the matrix elements
                // Stolen from Determinant::Hij for various reasons
                // (ExcitationDistance only in Determinants etc)
                std::array<int, 3> cre{-1, -1, -1},
                    des{-1, -1, -1}; // third entry may be accessed although it
                                     // is not used
                int ncre = 0, ndes = 0;
                {
                    long u, b, k;
                    constexpr long one = 1;
                    for (int i = 0; i < bra.EffDetLen; i++) {
                        u = bra.repr[i] ^ ket.repr[i];
                        b = u & bra.repr[i]; // the cre bits
                        k = u & ket.repr[i]; // the des bits
                        while (b != 0 and ncre <= 2) {
                            int pos = sci_detail::ffsl(
                                b); // hrl     Returns one plus the index
                                    // of the least significant 1-bit of x,
                                    // or if x is zero, returns zero.
                            cre[ncre] =
                                pos - 1 +
                                i * 64; // TODO HRL: i * 64 => use shifts
                            ncre++;
                            b &= ~(one << (pos - 1));
                        }
                        while (k != 0 and ndes <= 2) {
                            int pos = sci_detail::ffsl(k);
                            des[ndes] = pos - 1 + i * 64;
                            ndes++;
                            k &= ~(one << (pos - 1));
                        }
                        // ioUtil::say("here",ncre, ndes,"and",cre[0],cre[1],
                        // "and",des[0],des[1], "fullfiled",ncre >= 2 and ndes
                        // >=2);
                        if (ncre > 2 or ndes > 2) {
                            // May happen for space larger than CISD: Particle
                            // number does not help here
                            goto doneHereH;
                        }
                    }
                }
                assert(ncre == ndes);
                if (ncre == 1) {
                    const auto Hij =
                        ket.Hij_1Excite(cre[0], des[0], ints1, ints2);
                    if (std::abs(Hij) > eps) {
                        // Only store upper part
                        allOpCoeffs(iThread, 0)
                            .emplace_back(make_pair(ii, jj), Hij);
                    }
                } else if (ncre == 2) {
                    const auto Hij = ket.Hij_2Excite(des[0], des[1], cre[0],
                                                     cre[1], ints1, ints2);
                    if (std::abs(Hij) > eps) {
                        allOpCoeffs(iThread, 0)
                            .emplace_back(make_pair(ii, jj), Hij);
                    }
                } else {
                    // ioUtil::say("oops",iD,jD, "bra",bra.getClosed(),
                    // "ket",ket.getClosed(), "ncre ndes",ncre,ndes);
                    assert(false and "this should not happen");
                }
            doneHereH:
                continue;
            }
        }
        nonZeros +=
            fillCoeffs<true>(smat, allOpCoeffs, 0, mat[sym], summSparsityH,
                             numSparseH, numDenseH, usedMemH);
    }
    if (nonZeros == 0) { // oops!
        sci_detail::setMatrixToZero(mat);
        numZeroH += 1;
    }
    timeH += clock.get_time();
    qnCountsH += qnSiz;
    totCountsH += 1;
}
template <typename S>
template <bool Cop>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_C_impl(const S &deltaQN,
                                                           BLSparseMatrix &mat,
                                                           int iOrb) const {
    checkOMP();
    const auto iOrbL = iOrb - (isRight ? nOrbOther * 2 : 0);
    Timer clock;
    clock.get_time();
    const auto qnPairs = getQNpairsC(mat, deltaQN);
    const auto qnSiz = qnPairs.size();
    sci_detail::DenseMat<TripletVec> allOpCoeffs(ompThreads, 1);
    size_t nonZeros = 0;
    sci_detail::COOSparseMat<double> smat;
    for (int itQN = 0; itQN < qnSiz; ++itQN) {
        const auto ketQN = qnPairs[itQN].first;
        const auto braQN = qnPairs[itQN].second;
        const auto &sym = quantumNumbers[ketQN];
        const auto o1Ket =
            offsets[ketQN].first; // openMP cant handle auto [...] -.-
        const auto o2Ket =
            offsets[ketQN].second; // openMP cant handle auto [...] -.-
        const int sizKet = o2Ket - o1Ket;
        const auto o1Bra =
            offsets[braQN].first; // openMP cant handle auto [...] -.-
        const auto o2Bra =
            offsets[braQN].second; // openMP cant handle auto [...] -.-
#ifdef _SCI_USE_OMP_ON
#pragma omp parallel for default(shared) schedule(dynamic)
#endif
        for (int jj = 0; jj < sizKet; ++jj) {
            const auto iThread = getThreadID();
            const auto jD = o1Ket + jj;
            const auto &ket = fragSpace[jD];
            const auto closedKet = ket.getClosed();
            std::pair<int, std::vector<int>> applResult;
            if /* constexpr */ (Cop) {
                applResult = SCIFockDeterminant::applyCreator(closedKet, iOrbL);
            } else {
                applResult =
                    SCIFockDeterminant::applyAnnihilator(closedKet, iOrbL);
            }
            const auto phase = applResult.first;
            const auto closedBra = applResult.second;
            // const auto [phase, closedBra] = applResult;
            if (phase != 0) {
                // Find the determinant
                SCIFockDeterminant bra(ket.norbs, closedBra);
                const auto map_val = fragIndexMap.find(bra);
                int iD = map_val == fragIndexMap.cend() ? -1 : map_val->second;
                if (iD >= 0) { // may not be in there
                    auto ii = iD - o1Bra;
                    allOpCoeffs(iThread, 0)
                        .emplace_back(make_pair(ii, jj), phase);
                }
            }
        }
        if /* constexpr */ (Cop) {
            nonZeros +=
                fillCoeffs<false>(smat, allOpCoeffs, 0, mat[sym], summSparsityC,
                                  numSparseC, numDenseC, usedMemC);
        } else {
            nonZeros +=
                fillCoeffs<false>(smat, allOpCoeffs, 0, mat[sym], summSparsityD,
                                  numSparseD, numDenseD, usedMemD);
        }
    }
    if (nonZeros == 0) {
        sci_detail::setMatrixToZero(mat);
        Cop ? numZeroC += 1 : numZeroD += 1;
    }
    if /* constexpr */ (Cop) {
        timeC += clock.get_time();
        qnCountsC += qnSiz;
        totCountsC += 1;
    } else {
        timeD += clock.get_time();
        qnCountsD += qnSiz;
        totCountsD += 1;
    }
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_C(const S &deltaQN,
                                                      BLSparseMatrix &mat,
                                                      int iOrb) const {
    fillOp_C_impl<true>(deltaQN, mat, iOrb);
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_D(const S &deltaQN,
                                                      BLSparseMatrix &mat,
                                                      int iOrb) const {
    fillOp_C_impl<false>(deltaQN, mat, iOrb);
}

template <typename S>
template <bool Dagger>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_R_impl(
    const S &deltaQN, std::vector<entryTuple1> &entries) const {
    checkOMP();
    Timer clock;
    clock.get_time();
    auto &refMat = entries.at(0).mat;
    const int entrySize = entries.size();
    const auto qnPairs = getQNpairsR(refMat, deltaQN);
    sci_detail::DenseMat<TripletVec> allOpCoeffs(ompThreads, entrySize);
    vector<vector<double>> Hijs(ompThreads, vector<double>(entrySize, 0));
    const auto qnSiz = qnPairs.size();
    std::vector<size_t> nonZeros(entrySize, 0);
    sci_detail::COOSparseMat<double> smat;
    for (int itQN = 0; itQN < qnSiz; ++itQN) {
        const auto ketQN = qnPairs[itQN].first;
        const auto braQN = qnPairs[itQN].second;
        const auto &sym = quantumNumbers[ketQN];
        const auto o1Ket =
            offsets[ketQN].first; // openMP cant handle auto [...] -.-
        const auto o2Ket =
            offsets[ketQN].second; // openMP cant handle auto [...] -.-
        const int sizKet = o2Ket - o1Ket;
        const auto o1Bra =
            offsets[braQN].first; // openMP cant handle auto [...] -.-
        const auto o2Bra =
            offsets[braQN].second; // openMP cant handle auto [...] -.-
        const auto sizBra = o2Bra - o1Bra;
#ifdef _SCI_USE_OMP_ON
#pragma omp parallel for collapse(2) default(shared) schedule(dynamic)
#endif
        for (int ii = 0; ii < sizBra; ++ii) {
            for (int jj = 0; jj < sizKet; ++jj) {
                std::vector<int> closed;
                const auto iD = o1Bra + ii;
                const auto iThread = getThreadID();
#ifdef _SCI_USE_OMP_ON
                auto bra = fragSpace[iD]; // copy to ease life
#else
                auto &bra = const_cast<SCIFockDeterminant &>(
                    fragSpace[iD]); // will be restored
#endif
                if /* constexpr */ (not Dagger) {
                    closed = bra.getClosed(); // TODO could be saved...
                }
                const auto jD = o1Ket + jj;
#ifdef _SCI_USE_OMP_ON
                auto ket = fragSpace[jD];
#else
                auto &ket = const_cast<SCIFockDeterminant &>(
                    fragSpace[jD]); // will be restored
#endif
                assert(Dagger ? bra.nEl() - 1 == ket.nEl()
                              : bra.nEl() == ket.nEl() - 1);
                if /* constexpr */ (Dagger) {
                    closed = ket.getClosed(); // TODO: Change bra/ket loop...
                }
                // FILL the matrix elements
                /*
                 * Here, we use the following:
                 * <a|R_x|b> = sum_jkl (xl|jk) <a|j'kl|b>
                 *           = sum_l [ sum_jk h^[xl]_{jk} <a|j'k| l b>
                 *    => sum over one-particle integrals.
                 *    (and also add 1/2 \hat S = sum_l h_xj l / 2)
                 *    Maybe not the best way but the simplest in my
                 * implementation.
                 *
                 *    For R', it is l'k'j (xl|jk) = (xl|kj) (j<=>k possible)
                 *    // TODO similar t H and P creation should be *much* better
                 *
                 */
                auto &Hij = Hijs[iThread];
                memset(Hij.data(), 0, Hij.size() * sizeof(double));
                // vv this is the parity coming from applying ll destruction to
                // ket.
                //      Since ll is looped from 0 to nOrb, I flip the sign for
                //      every occupied orbital l
                double signL = 1.;
                // TODO: vvv The bra loop could also be inside this
                // vvv for R, we have l  |bra>
                //     for R', we have <ket| l' = <ket l|
                auto &chDet = Dagger ? bra : ket;
                for (int ll = 0; ll < 2 * nOrbThis; ++ll) {
                    if (not chDet.getocc(ll)) {
                        continue;
                    }
                    chDet.setocc(ll, false);
                    if (ll % 2 == 0) {
                        chDet.nAlphaEl -= 1;
                    } else {
                        chDet.nBetaEl -= 1;
                    }
                    assert(chDet.consistencyCheck());
                    if (ket == bra) {
                        // energy expression
                        for (int xx = 0; xx < entrySize; ++xx) {
                            const auto iOrb = entries[xx].iOrb;
                            double energy = 0.0;
                            for (int i = 0; i < closed.size(); i++) {
                                energy +=
                                    ints2.intsR(iOrb, ll, closed[i], closed[i]);
                            }
                            energy += ints1.intsS(iOrb, ll) / 2.0;
                            Hij[xx] += signL * energy;
                        }
                    } else {
                        std::array<int, 2> cre{-1, -1},
                            des{-1, -1}; // second entry may be accessed
                                         // although it is not used
                        int ncre = 0, ndes = 0;
                        {
                            long u, b, k;
                            constexpr long one = 1;
                            for (int i = 0; i < bra.EffDetLen; i++) {
                                u = bra.repr[i] ^ ket.repr[i];
                                b = u & bra.repr[i]; // the cre bits
                                k = u & ket.repr[i]; // the des bits
                                while (b != 0 and ncre <= 1) {
                                    int pos = sci_detail::ffsl(b);
                                    cre[ncre] = pos - 1 + i * 64;
                                    ncre++;
                                    b &= ~(one << (pos - 1));
                                }
                                while (k != 0 and ndes <= 1) {
                                    int pos = sci_detail::ffsl(k);
                                    des[ndes] = pos - 1 + i * 64;
                                    ndes++;
                                    k &= ~(one << (pos - 1));
                                }
                                if (ncre > 1 or ndes > 1) {
                                    // I already have the proper symmetry labels
                                    // so this should not happen
                                    // May happen for space larger than CISD:
                                    // Particle number does not help here
                                    // say("\t",ii,jj,"done Here!",Hij);
                                    goto doneHereR3;
                                }
                            }
                        }
                        assert(ncre == ndes);
                        if (ncre == 1) {
                            auto i = cre[0], a = des[0];
                            // <a|R_x|b> = sum_jkl (xl|jk) <a|j'kl|b>
                            double sgn = signL;
                            ket.parity(std::min(i, a), std::max(i, a), sgn);
                            for (int xx = 0; xx < entrySize; ++xx) {
                                const auto integral =
                                    ints2.intsR(entries[xx].iOrb, ll, i, a);
                                if (std::abs(integral) > eps * .01) {
                                    Hij[xx] += sgn * integral;
                                }
                            }
                        } else {
                            assert(false and "this should not happen");
                        }
                    }
                doneHereR3:
                    chDet.setocc(ll, true); // restore
                    if (ll % 2 == 0) {
                        chDet.nAlphaEl += 1;
                    } else {
                        chDet.nBetaEl += 1;
                    }
                    signL *= -1.; // flip it!
                }                 // ll loop
                for (int xx = 0; xx < entrySize; ++xx) {
                    if (std::abs(Hij[xx]) > eps) {
                        allOpCoeffs(iThread, xx)
                            .emplace_back(make_pair(ii, jj), Hij[xx]);
                    }
                }
            }
        }
        for (int xx = 0; xx < entrySize; ++xx) {
            if /* constexpr */ (Dagger) {
                nonZeros[xx] += fillCoeffs<false>(
                    smat, allOpCoeffs, xx, entries[xx].mat[sym], summSparsityRD,
                    numSparseRD, numDenseRD, usedMemRD);
            } else {
                nonZeros[xx] += fillCoeffs<false>(
                    smat, allOpCoeffs, xx, entries[xx].mat[sym], summSparsityR,
                    numSparseR, numDenseR, usedMemR);
            }
        }
    }
    for (int xx = 0; xx < entrySize; ++xx) {
        if (nonZeros[xx] == 0) {
            auto &mat = entries[xx].mat;
            sci_detail::setMatrixToZero(mat);
            Dagger ? numZeroRD += 1 : numZeroR += 1;
        }
    }
    if /* constexpr */ (Dagger) {
        timeRD += clock.get_time();
        qnCountsRD += qnSiz * entrySize;
        totCountsRD += 1;
    } else {
        timeR += clock.get_time();
        qnCountsR += qnSiz * entrySize;
        totCountsR += 1;
    }
    if (doAllocateEmptyMats()) {
        sci_detail::allocateEmptyMatrices<S>(entries);
    }
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_R(
    const S &deltaQN, std::vector<entryTuple1> &entries) const {
    // new R = R + 1/2 S; see
    // https://pyblock-dmrg.readthedocs.io/en/latest/DMRG/unrestricted.html
    // say("-------- fill Op R:",iOrb);
    fillOp_R_impl<false>(deltaQN, entries);
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_RD(
    const S &deltaQN, std::vector<entryTuple1> &entries) const {
    // say("FILL RD");
    fillOp_R_impl<true>(deltaQN, entries);
}
template <typename S>
template <int Type>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_AB_impl(const S &deltaQN,
                                                            BLSparseMatrix &mat,
                                                            int iOrb,
                                                            int jOrb) const {
    checkOMP();
    const auto iOrbL = iOrb - (isRight ? nOrbOther * 2 : 0);
    const auto jOrbL = jOrb - (isRight ? nOrbOther * 2 : 0);
    Timer clock;
    clock.get_time();
    const auto qnPairs =
        Type == 2 ? getQNpairsB(mat, deltaQN) : getQNpairsA(mat, deltaQN);
    sci_detail::DenseMat<TripletVec> allOpCoeffs(ompThreads, 1);
    const auto qnSiz = qnPairs.size();
    size_t nonZeros = 0;
    sci_detail::COOSparseMat<double> smat;
#ifndef _SCI_USE_OMP_ON
    std::pair<int, std::vector<int>> applResult;
    std::pair<int, std::vector<int>> applResult2;
#endif
    for (int itQN = 0; itQN < qnSiz; ++itQN) {
        const auto ketQN = qnPairs[itQN].first;
        const auto braQN = qnPairs[itQN].second;
        const auto &sym = quantumNumbers[ketQN];
        const auto o1Ket =
            offsets[ketQN].first; // openMP cant handle auto [...] -.-
        const auto o2Ket =
            offsets[ketQN].second; // openMP cant handle auto [...] -.-
        const int sizKet = o2Ket - o1Ket;
        const auto o1Bra =
            offsets[braQN].first; // openMP cant handle auto [...] -.-
        const auto o2Bra =
            offsets[braQN].second; // openMP cant handle auto [...] -.-
#ifdef _SCI_USE_OMP_ON
#pragma omp parallel for default(shared) schedule(dynamic)
#endif
        for (int jj = 0; jj < sizKet; ++jj) {
#ifdef _SCI_USE_OMP_ON
            std::pair<int, std::vector<int>> applResult;
            std::pair<int, std::vector<int>> applResult2;
#endif
            const auto iThread = getThreadID();
            const auto jD = o1Ket + jj;
            const auto &ket = fragSpace[jD];
            const auto closedKet = ket.getClosed();
            if /* constexpr */ (Type == 0 or Type == 2) { // A: i j; B: i' j
                applResult =
                    SCIFockDeterminant::applyAnnihilator(closedKet, jOrbL);
            } else { // A': j' i'
                // static_assert(Type == 1);
                applResult = SCIFockDeterminant::applyCreator(closedKet, iOrbL);
            }
            const auto &phase1 = applResult.first;
            const auto &closedKet2 = applResult.second;
            // const auto &[phase1, closedKet2] = applResult;
            if (phase1 != 0) {
                if /* constexpr */ (Type == 0) { // A: i j
                    applResult2 =
                        SCIFockDeterminant::applyAnnihilator(closedKet2, iOrbL);
                } else { // A': j' i'; B: i' j
                    // static_assert(Type == 1 or Type == 2);
                    applResult2 = SCIFockDeterminant::applyCreator(
                        closedKet2, Type == 1 ? jOrbL : iOrbL);
                }
                const auto phase2 = applResult2.first;
                const auto closedBra = applResult2.second;
                // const auto [phase2, closedBra] = applResult2;
                if (phase2 != 0) {
                    // Find the determinant
                    SCIFockDeterminant bra(ket.norbs, closedBra);
                    const auto map_val = fragIndexMap.find(bra);
                    int iD =
                        map_val == fragIndexMap.cend() ? -1 : map_val->second;
                    if (iD >= 0) { // may not be in there
                        int ii = iD - o1Bra;
                        allOpCoeffs(iThread, 0)
                            .emplace_back(make_pair(ii, jj), phase1 * phase2);
                    }
                }
            }
        }
        if /* constexpr */ (Type == 0) {
            nonZeros +=
                fillCoeffs<false>(smat, allOpCoeffs, 0, mat[sym], summSparsityA,
                                  numSparseA, numDenseA, usedMemA);
        } else if /* constexpr */ (Type == 1) {
            nonZeros += fillCoeffs<false>(smat, allOpCoeffs, 0, mat[sym],
                                          summSparsityAD, numSparseAD,
                                          numDenseAD, usedMemAD);
        } else {
            // static_assert(Type == 2);
            nonZeros +=
                fillCoeffs<false>(smat, allOpCoeffs, 0, mat[sym], summSparsityB,
                                  numSparseB, numDenseB, usedMemB);
        }
    }
    if (nonZeros == 0) {
        sci_detail::setMatrixToZero(mat);
    }
    if /* constexpr */ (Type == 0) {
        timeA += clock.get_time();
        qnCountsA += qnSiz;
        totCountsA += 1;
        if (nonZeros == 0) {
            numZeroA++;
        }
    } else if /* constexpr */ (Type == 1) {
        timeAD += clock.get_time();
        qnCountsAD += qnSiz;
        totCountsAD += 1;
        if (nonZeros == 0) {
            numZeroAD++;
        }
    } else {
        // static_assert(Type == 2);
        timeB += clock.get_time();
        qnCountsB += qnSiz;
        totCountsB += 1;
        if (nonZeros == 0) {
            numZeroB++;
        }
    }
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_A(const S &deltaQN,
                                                      BLSparseMatrix &mat,
                                                      int iOrb,
                                                      int jOrb) const {
    // ATTENTION:
    //  block2 definition: A_ij = i' j'; AD_ij = j i
    //   my definition (consistent with 10.1063/1.1638734):
    //          A_ij = i j; AD_ij = j' i'
    // => so here, I need to cale A' routine
    fillOp_AB_impl<1>(deltaQN, mat, jOrb, iOrb);
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_AD(const S &deltaQN,
                                                       BLSparseMatrix &mat,
                                                       int iOrb,
                                                       int jOrb) const {
    /** Fill A' = j'i' (note order!) */
    // See above... AD and A are interchanged
    fillOp_AB_impl<0>(deltaQN, mat, jOrb, iOrb);
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_B(const S &deltaQN,
                                                      BLSparseMatrix &mat,
                                                      int iOrb,
                                                      int jOrb) const {
    fillOp_AB_impl<2>(deltaQN, mat, iOrb, jOrb);
}

template <typename S>
template <bool Dagger>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_P_impl(
    const S &deltaQN, std::vector<entryTuple2> &entries) const {
    checkOMP();
    assert(isRight and "Should not happen for left big site and NC MPO!");
    Timer clock;
    clock.get_time();
    auto &refMat = entries.at(0).mat;
    const int entrySize = entries.size();
    sci_detail::DenseMat<TripletVec> allOpCoeffs(ompThreads, entrySize);
    const auto qnPairs = getQNpairsP(
        refMat, deltaQN); // all entries should generate the same pairs!
    const auto qnSiz = qnPairs.size();
    std::vector<size_t> nonZeros(entrySize, 0);
    sci_detail::COOSparseMat<double> smat;
    for (int itQN = 0; itQN < qnSiz; ++itQN) {
        const auto ketQN = qnPairs[itQN].first;
        const auto braQN = qnPairs[itQN].second;
        const auto &sym = quantumNumbers[ketQN];
        const auto o1Ket =
            offsets[ketQN].first; // openMP cant handle auto [...] -.-
        const auto o2Ket =
            offsets[ketQN].second; // openMP cant handle auto [...] -.-
        const int sizKet = o2Ket - o1Ket;
        const auto o1Bra =
            offsets[braQN].first; // openMP cant handle auto [...] -.-
        const auto o2Bra =
            offsets[braQN].second; // openMP cant handle auto [...] -.-
        const auto sizBra = o2Bra - o1Bra;
#ifdef _SCI_USE_OMP_ON
#pragma omp parallel for collapse(2) default(shared) schedule(dynamic)
#endif
        for (int ii = 0; ii < sizBra; ++ii) {
            for (int jj = 0; jj < sizKet; ++jj) {
                const auto iThread = getThreadID();
                const auto iD = o1Bra + ii;
                const auto &bra = fragSpace[iD];
                const auto jD = o1Ket + jj;
                const auto &ket = fragSpace[jD];
                assert(Dagger ? bra.nEl() == ket.nEl() + 2
                              : bra.nEl() == ket.nEl() - 2);
                // FILL the matrix elements
                // sum_kl [xl|yk] k l (or k' l' for Dagger)
                // ATTENTION: For CISD, this can be simplified. bra = vac and
                // ket = |kl>
                //          => occupied orbitals of ket
                {
                    std::array<int, 3> chg{-1, -1,
                                           -1}; // third entry may be accessed
                                                // although it is not used
                    int nchg = 0;
                    {
                        long u, k;
                        constexpr long one = 1;
                        for (int i = 0; i < bra.EffDetLen; i++) {
                            u = bra.repr[i] ^ ket.repr[i];
                            if /* constexpr */ (Dagger) {
                                k = u & bra.repr[i]; // the cre bits
                                while (k != 0 and nchg <= 2) {
                                    int pos = sci_detail::ffsl(
                                        k); // hrl     Returns one plus the
                                            // index of the least
                                            // significant 1-bit of x, or
                                            // if x is zero, returns zero.
                                    chg[nchg] =
                                        pos - 1 + i * 64; // TODO HRL: i * 64 =>
                                                          // use shifts
                                    nchg++;
                                    k &= ~(one << (pos - 1));
                                }
                            } else {
                                k = u & ket.repr[i]; // the des bits
                                while (k != 0 and nchg <= 2) {
                                    int pos = sci_detail::ffsl(k);
                                    chg[nchg] = pos - 1 + i * 64;
                                    nchg++;
                                    k &= ~(one << (pos - 1));
                                }
                            }
                            if (nchg > 2) {
                                // I already have the proper symmetry labels so
                                // this should not happen
                                // May happen for space larger than CISD:
                                // Particle number does not help here
                                goto doneHereP;
                            }
                        }
                    }
                    assert(nchg == 2);
                    // k l _and_ l k are possible
                    int k = std::min(chg[0], chg[1]);
                    int l = std::max(chg[0], chg[1]);
                    assert(k < l);
                    double Hij;
                    double sgn = 1.0;
                    if (Dagger ? bra.nEl() != 2 : ket.nEl() != 2) {
                        ket.parity(k, l, sgn); // Not needed for CISD
                    }
                    for (int xx = 0; xx < entrySize; ++xx) {
                        const auto integral =
                            -ints2.intsP(entries[xx].iOrb, l, entries[xx].jOrb,
                                         k) +
                            ints2.intsP(entries[xx].iOrb, k, entries[xx].jOrb,
                                        l);
                        Hij = sgn * integral;
                        if (std::abs(Hij) > eps) {
                            allOpCoeffs(iThread, xx)
                                .emplace_back(make_pair(ii, jj), Hij);
                        }
                    }
                }
            doneHereP:
                continue;
            }
        }
        if /* constexpr */ (Dagger) {
            for (int xx = 0; xx < entrySize; ++xx) {
                nonZeros[xx] += fillCoeffs<false>(
                    smat, allOpCoeffs, xx, entries[xx].mat[sym], summSparsityPD,
                    numSparsePD, numDensePD, usedMemPD);
            }
        } else {
            for (int xx = 0; xx < entrySize; ++xx) {
                nonZeros[xx] += fillCoeffs<false>(
                    smat, allOpCoeffs, xx, entries[xx].mat[sym], summSparsityP,
                    numSparseP, numDenseP, usedMemP);
            }
        }
    }
    for (int xx = 0; xx < entrySize; ++xx) {
        if (nonZeros[xx] == 0) {
            auto &mat = entries[xx].mat;
            sci_detail::setMatrixToZero(mat);
            Dagger ? numZeroPD += 1 : numZeroP += 1;
        }
    }
    if /* constexpr */ (Dagger) {
        timePD += clock.get_time();
        qnCountsPD += qnSiz * entrySize;
        totCountsPD += 1;
    } else {
        timeP += clock.get_time();
        qnCountsP += qnSiz * entrySize;
        totCountsP += 1;
    }
    if (doAllocateEmptyMats()) {
        sci_detail::allocateEmptyMatrices<S>(entries);
    }
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_P(
    const S &deltaQN, std::vector<entryTuple2> &entries) const {
    fillOp_P_impl<false>(deltaQN, entries);
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_PD(
    const S &deltaQN, std::vector<entryTuple2> &entries) const {
    fillOp_P_impl<true>(deltaQN, entries);
}
template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::fillOp_Q(
    const S &deltaQN, std::vector<entryTuple2> &entries) const {
    assert(isRight and "Should not happen for left big site and NC MPO!");
    // sum_kl [ [xy|kl] - [xl|ky] ] k' l
    checkOMP();
    Timer clock;
    clock.get_time();
    auto &refMat = entries.at(0).mat;
    const int entrySize = entries.size();
    const auto qnPairs = getQNpairsQ(
        refMat, deltaQN); // all entries should generate the same pairs!
    sci_detail::DenseMat<TripletVec> allOpCoeffs(ompThreads, entrySize);
    const auto qnSiz = qnPairs.size();
    std::vector<size_t> nonZeros(entrySize, 0);
    sci_detail::COOSparseMat<double> smat;
    for (int itQN = 0; itQN < qnSiz; ++itQN) {
        const auto ketQN = qnPairs[itQN].first;
        const auto braQN = qnPairs[itQN].second;
        const auto &sym = quantumNumbers[ketQN];
        const auto o1Ket =
            offsets[ketQN].first; // openMP cant handle auto [...] -.-
        const auto o2Ket =
            offsets[ketQN].second; // openMP cant handle auto [...] -.-
        const int sizKet = o2Ket - o1Ket;
        const auto o1Bra =
            offsets[braQN].first; // openMP cant handle auto [...] -.-
        const auto o2Bra =
            offsets[braQN].second; // openMP cant handle auto [...] -.-
        const auto sizBra = o2Bra - o1Bra;
        // TODO: How does this perform for small sizes? Some qnblocks have just
        // 1 state (sizBra=sizKet=1)
#ifdef _SCI_USE_OMP_ON
#pragma omp parallel for collapse(2) default(shared) schedule(dynamic)
#endif
        for (int ii = 0; ii < sizBra; ++ii) {
            for (int jj = 0; jj < sizKet; ++jj) {
                const auto iThread = getThreadID();
                const auto iD = o1Bra + ii;
                const auto &bra = fragSpace[iD];
                const auto jD = o1Ket + jj;
                if (iD == jD) { // Diagonal
                    auto closed = bra.getClosed();
                    for (int xx = 0; xx < entrySize; ++xx) {
                        double energy = 0.0;
                        for (int i = 0; i < closed.size(); i++) {
                            energy +=
                                ints2.intsQ(entries[xx].iOrb, entries[xx].jOrb,
                                            closed[i], closed[i]);
                        }
                        if (std::abs(energy) > eps) {
                            allOpCoeffs(iThread, xx)
                                .emplace_back(make_pair(ii, ii), energy);
                        }
                    }
                    continue;
                }
                const auto &ket = fragSpace[jD];
                assert(bra.nEl() == ket.nEl());
                // FILL the matrix elements
                {
                    std::array<int, 2> cre{-1, -1}, des{-1, -1};
                    int ncre = 0, ndes = 0;
                    {
                        long u, b, k;
                        constexpr long one = 1;
                        for (int i = 0; i < bra.EffDetLen; i++) {
                            u = bra.repr[i] ^ ket.repr[i];
                            b = u & bra.repr[i]; // the cre bits
                            k = u & ket.repr[i]; // the des bits
                            while (b != 0 and ncre <= 1) {
                                int pos = sci_detail::ffsl(b);
                                cre[ncre] = pos - 1 + i * 64;
                                ncre++;
                                b &= ~(one << (pos - 1));
                            }
                            while (k != 0 and ndes <= 1) {
                                int pos = sci_detail::ffsl(k);
                                des[ndes] = pos - 1 + i * 64;
                                ndes++;
                                k &= ~(one << (pos - 1));
                            }
                            // ioUtil::say("here",ncre,
                            // ndes,"and",cre[0],cre[1], "and",des[0],des[1],
                            // "fullfiled",ncre >= 2 and ndes >=2);
                            if (ncre > 1 or ndes > 1) {
                                // May happen for space larger than CISD:
                                // Particle number does not help here
                                goto doneHereQ;
                            }
                        }
                    }
                    // if(ncre != 1 and ndes != 1){
                    //    say(ii,jj,"ops",bra.getClosed(),"ket",ket.getClosed(),"cre",cre[0],cre[1],"des",des[0],des[1]);
                    //}
                    assert(ncre == 1 and ndes == 1);
                    double sgn = 1.0;
                    ket.parity(min(cre[0], des[0]), max(cre[0], des[0]), sgn);
                    for (int xx = 0; xx < entrySize; ++xx) {
                        const auto integral = ints2.intsQ(
                            entries[xx].iOrb, entries[xx].jOrb, cre[0], des[0]);
                        auto Hij = sgn * integral;
                        if (std::abs(Hij) > eps) {
                            allOpCoeffs(iThread, xx)
                                .emplace_back(make_pair(ii, jj), Hij);
                        }
                    }
                }
            doneHereQ:
                continue;
            }
        }
        for (int xx = 0; xx < entrySize; ++xx) {
            nonZeros[xx] += fillCoeffs<false>(
                smat, allOpCoeffs, xx, entries[xx].mat[sym], summSparsityQ,
                numSparseQ, numDenseQ, usedMemQ);
        }
    }
    for (int xx = 0; xx < entrySize; ++xx) {
        if (nonZeros[xx] == 0) {
            auto &mat = entries[xx].mat;
            sci_detail::setMatrixToZero(mat);
            numZeroQ += 1;
        }
    }
    timeQ += clock.get_time();
    qnCountsQ += qnSiz * entrySize;
    totCountsQ += 1;
    if (doAllocateEmptyMats()) {
        sci_detail::allocateEmptyMatrices<S>(entries);
    }
}

// hz: the following are from sciblock2/src/SciWrapperExcludeQNs.cpp

template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::setQnIdxBra(
    const std::vector<int> &inp, const std::vector<char> &ops) {
    if (ops.size() == 1 and ops[0] == 'X') {
        qnIdxBraH = inp;
        qnIdxBraI = inp;
        qnIdxBraQ = inp;
        qnIdxBraA = inp;
        qnIdxBraB = inp;
        qnIdxBraP = inp;
        qnIdxBraR = inp;
        qnIdxBraC = inp;
        return;
    }
    for (const auto &op : ops) {
        switch (op) {
        case 'H':
            qnIdxBraH = inp;
            break;
        case 'I':
            qnIdxBraI = inp;
            break;
        case 'Q':
            qnIdxBraQ = inp;
            break;
        case 'A':
            qnIdxBraA = inp;
            break;
        case 'B':
            qnIdxBraB = inp;
            break;
        case 'P':
            qnIdxBraP = inp;
            break;
        case 'R':
            qnIdxBraR = inp;
            break;
        case 'C':
            qnIdxBraC = inp;
            break;
        default:
            throw std::invalid_argument("setQnIdxBra: Unknown op '" +
                                        Parsing::to_string(op) + "'");
        }
    }
}

template <typename S>
void SCIFockBigSite<S, typename S::is_sz_t>::setQnIdxKet(
    const std::vector<int> &inp, const std::vector<char> &ops) {
    if (ops.size() == 1 and ops[0] == 'X') {
        qnIdxKetH = inp;
        qnIdxKetI = inp;
        qnIdxKetQ = inp;
        qnIdxKetA = inp;
        qnIdxKetB = inp;
        qnIdxKetP = inp;
        qnIdxKetR = inp;
        qnIdxKetC = inp;
        return;
    }
    for (const auto &op : ops) {
        switch (op) {
        case 'H':
            qnIdxKetH = inp;
            break;
        case 'I':
            qnIdxKetI = inp;
            break;
        case 'Q':
            qnIdxKetQ = inp;
            break;
        case 'A':
            qnIdxKetA = inp;
            break;
        case 'B':
            qnIdxKetB = inp;
            break;
        case 'P':
            qnIdxKetP = inp;
            break;
        case 'R':
            qnIdxKetR = inp;
            break;
        case 'C':
            qnIdxKetC = inp;
            break;
        default:
            throw std::invalid_argument("setQnIdxKet: Unknown op '" +
                                        Parsing::to_string(op) + "'");
        }
    }
}

template <typename S>
bool SCIFockBigSite<S, typename S::is_sz_t>::idxInKet(
    const int braIdx, const std::vector<int> &qnIdxKet) const {
    if (qnIdxKet.size() > 0) {
        bool inKet = false;
        for (auto jQ : qnIdxKet) { // TODO: use map for it
            if (braIdx == jQ) {
                inKet = true;
                break;
            }
        }
        return inKet;
    } else {
        return true;
    }
}

template <typename S>
std::vector<std::pair<int, int>>
SCIFockBigSite<S, typename S::is_sz_t>::getQNPairsImpl(
    const BLSparseMatrix &mat, const S &deltaQN,
    const std::vector<int> &qnIdxBra, const std::vector<int> &qnIdxKet) const {
    std::vector<std::pair<int, int>> pairs;
    const auto qSize = static_cast<int>(quantumNumbers.size());
    std::vector<bool> inBra(qSize, false);
    for (const auto iQ : qnIdxBra) {
        assert(iQ < quantumNumbers.size());
        const auto &sym = quantumNumbers[iQ]; // ket qn number
        { // find out whether sym exist in ket quantumnumbers of mat
            auto ptr = mat.info->find_state(sym);
            if (ptr < 0) { // nope
                continue;
            }
        }
        S qnBra{sym.n() + deltaQN.n(), sym.twos() + deltaQN.twos(),
                sym.pg() ^ deltaQN.pg()};
        auto jQit = quantumNumberToIdx.find(qnBra);
        if (jQit == quantumNumberToIdx.end()) {
            continue;
        }
        if (not idxInKet(iQ, qnIdxKet)) {
            continue;
        }
        const auto jQ = jQit->second;
        inBra[iQ] = true;
        pairs.emplace_back(iQ, jQ);
    }
    // ATTENTION this vv does not capture all BLSparseMatrices =>
    // sci_detail::allocateEmptyMatrices routine in SciWrapper
    //                  (due to batched calculation via vector<entrytuple>)
    for (int iQ = 0; iQ < qSize; ++iQ) {
        if (not inBra[iQ]) {
            const auto &sym = quantumNumbers[iQ]; // ket qn number
            auto ptr = mat.info->find_state(sym);
            if (ptr < 0) {
                continue;
            }
            S qnBra{sym.n() + deltaQN.n(), sym.twos() + deltaQN.twos(),
                    sym.pg() ^ deltaQN.pg()};
            auto jQit = quantumNumberToIdx.find(qnBra);
            if (jQit == quantumNumberToIdx.end()) {
                continue;
            }
            mat[sym].nnz = 0;
            mat[sym].alloc = make_shared<VectorAllocator<double>>();
            mat[sym].allocate();
            assert(mat[sym].data != nullptr);
        }
    }
    return pairs;
}

template class SCIFockBigSite<block2::SZ>;

} // namespace block2
