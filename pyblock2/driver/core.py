#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2022 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

from enum import IntFlag


class SymmetryTypes(IntFlag):
    Nothing = 0
    SU2 = 1
    SZ = 2
    SGF = 4
    SGB = 8
    CPX = 16
    SP = 32
    SGFCPX = 16 | 4
    SPCPX = 32 | 16


class ParallelTypes(IntFlag):
    Nothing = 0
    I = 1
    J = 2
    SI = 3
    SJ = 4
    SIJ = 5
    SKL = 6
    UIJ = 7  # unified middle same/diff
    UKL = 8  # unified middle same/diff
    UIJM2 = 9
    UKLM2 = 10
    UIJM4 = 11
    UKLM4 = 12
    MixUIJUKL = 1000  # not a good strategy
    MixUKLUIJ = 1001  # not a good strategy
    MixUIJSI = 1002


class MPOAlgorithmTypes(IntFlag):
    Nothing = 0
    Bipartite = 1
    SVD = 2
    Rescaled = 4
    Fast = 8
    Blocked = 16
    Sum = 32
    Constrained = 64
    Disjoint = 128
    DisjointSVD = 128 | 2
    BlockedSumDisjointSVD = 128 | 32 | 16 | 2
    FastBlockedSumDisjointSVD = 128 | 32 | 16 | 8 | 2
    BlockedRescaledSumDisjointSVD = 128 | 32 | 16 | 4 | 2
    FastBlockedRescaledSumDisjointSVD = 128 | 32 | 16 | 8 | 4 | 2
    BlockedDisjointSVD = 128 | 16 | 2
    FastBlockedDisjointSVD = 128 | 16 | 8 | 2
    BlockedRescaledDisjointSVD = 128 | 16 | 4 | 2
    FastBlockedRescaledDisjointSVD = 128 | 16 | 8 | 4 | 2
    RescaledDisjointSVD = 128 | 4 | 2
    FastDisjointSVD = 128 | 8 | 2
    FastRescaledDisjointSVD = 128 | 8 | 4 | 2
    ConstrainedSVD = 64 | 2
    BlockedSumConstrainedSVD = 64 | 32 | 16 | 2
    FastBlockedSumConstrainedSVD = 64 | 32 | 16 | 8 | 2
    BlockedRescaledSumConstrainedSVD = 64 | 32 | 16 | 4 | 2
    FastBlockedRescaledSumConstrainedSVD = 64 | 32 | 16 | 8 | 4 | 2
    BlockedConstrainedSVD = 64 | 16 | 2
    FastBlockedConstrainedSVD = 64 | 16 | 8 | 2
    BlockedRescaledConstrainedSVD = 64 | 16 | 4 | 2
    FastBlockedRescaledConstrainedSVD = 64 | 16 | 8 | 4 | 2
    RescaledConstrainedSVD = 64 | 4 | 2
    FastConstrainedSVD = 64 | 8 | 2
    FastRescaledConstrainedSVD = 64 | 8 | 4 | 2
    BlockedSumSVD = 32 | 16 | 2
    FastBlockedSumSVD = 32 | 16 | 8 | 2
    BlockedRescaledSumSVD = 32 | 16 | 4 | 2
    FastBlockedRescaledSumSVD = 32 | 16 | 8 | 4 | 2
    BlockedSumBipartite = 32 | 16 | 1
    FastBlockedSumBipartite = 32 | 16 | 8 | 1
    BlockedSVD = 16 | 2
    FastBlockedSVD = 16 | 8 | 2
    BlockedRescaledSVD = 16 | 4 | 2
    FastBlockedRescaledSVD = 16 | 8 | 4 | 2
    BlockedBipartite = 16 | 1
    FastBlockedBipartite = 16 | 8 | 1
    RescaledSVD = 4 | 2
    FastSVD = 8 | 2
    FastRescaledSVD = 8 | 4 | 2
    FastBipartite = 8 | 1
    NC = 256
    CN = 512
    Conventional = 1024
    ConventionalNC = 1024 | 256
    ConventionalCN = 1024 | 512
    NoTranspose = 2048
    NoRIntermed = 4096
    NoTransConventional = 2048 | 1024
    NoTransConventionalNC = 2048 | 1024 | 256
    NoTransConventionalCN = 2048 | 1024 | 512
    NoRIntermedConventional = 4096 | 1024
    NoTransNoRIntermedConventional = 4096 | 2048 | 1024


class NPDMAlgorithmTypes(IntFlag):
    Nothing = 0
    SymbolFree = 1
    Normal = 2
    Fast = 4
    Compressed = 8
    LowMem = 16
    Default = 1 | 8
    Conventional = 32


class STTypes(IntFlag):
    H = 1
    HT = 2
    HT2T2 = 4
    HT1T2 = 8
    HT1T3 = 16
    HT2T3 = 32
    H_HT = 1 | 2
    H_HT_HT2T2 = 1 | 2 | 4
    H_HT_HTT = 1 | 2 | 4 | 8
    H_HT_HTT_HT1T3 = 1 | 2 | 4 | 8 | 16
    H_HT_HTT_HT2T3 = 1 | 2 | 4 | 8 | 32
    H_HT_HTT_HT3 = 1 | 2 | 4 | 8 | 16 | 32
    H_HT_HT2T2_HT1T3 = 1 | 2 | 4 | 16
    H_HT_HT2T2_HT2T3 = 1 | 2 | 4 | 32
    H_HT_HT2T2_HT3 = 1 | 2 | 4 | 16 | 32


class Block2Wrapper:
    def __init__(self, symm_type=SymmetryTypes.SU2):
        import block2 as b

        self.b = b
        self.symm_type = symm_type
        has_cpx = hasattr(b, "cpx")
        has_sp = hasattr(b, "sp")
        has_spcpx = has_sp and hasattr(b.sp, "cpx")
        has_sgf = hasattr(b, "sgf")
        has_sgb = hasattr(b, "sgb")
        if SymmetryTypes.CPX in symm_type and not has_cpx:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_COMPLEX=ON'!")
        elif SymmetryTypes.SP in symm_type and not has_sp:
            raise RuntimeError(
                "block2 needs to be compiled with '-DUSE_SINGLE_PREC=ON'!"
            )
        elif SymmetryTypes.SGF in symm_type and not has_sgf:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SG=ON'!")
        elif SymmetryTypes.SGB in symm_type and not has_sgb:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SG=ON'!")
        if SymmetryTypes.SPCPX in symm_type:
            self.VectorFL = b.VectorComplexFloat
            self.VectorFP = b.VectorFloat
            self.bx = b.sp.cpx
            self.bc = self.bx
        elif SymmetryTypes.CPX in symm_type:
            self.VectorFL = b.VectorComplexDouble
            self.VectorFP = b.VectorDouble
            self.bx = b.cpx
            self.bc = self.bx
        elif SymmetryTypes.SP in symm_type:
            self.VectorFL = b.VectorFloat
            self.VectorFP = b.VectorFloat
            self.bx = b.sp
            self.bc = b.sp.cpx if has_spcpx else None
        else:
            self.VectorFL = b.VectorDouble
            self.VectorFP = b.VectorDouble
            self.bx = b
            self.bc = b.cpx if has_cpx else None
        if SymmetryTypes.SU2 in symm_type:
            self.bs = self.bx.su2
            self.bcs = self.bc.su2 if self.bc is not None else None
            self.brs = b.su2
            self.SX = b.SU2
            self.VectorSX = b.VectorSU2
        elif SymmetryTypes.SZ in symm_type:
            self.bs = self.bx.sz
            self.bcs = self.bc.sz if self.bc is not None else None
            self.brs = b.sz
            self.SX = b.SZ
            self.VectorSX = b.VectorSZ
        elif SymmetryTypes.SGF in symm_type:
            self.bs = self.bx.sgf
            self.bcs = self.bc.sgf if self.bc is not None else None
            self.brs = b.sgf
            self.SX = b.SGF
            self.VectorSX = b.VectorSGF
        elif SymmetryTypes.SGB in symm_type:
            self.bs = self.bx.sgb
            self.bcs = self.bc.sgb if self.bc is not None else None
            self.brs = b.sgb
            self.SX = b.SGB
            self.VectorSX = b.VectorSGB


class DMRGDriver:
    def __init__(
        self,
        stack_mem=1 << 30,
        scratch="./nodex",
        clean_scratch=True,
        restart_dir=None,
        n_threads=None,
        symm_type=SymmetryTypes.SU2,
        mpi=None,
        stack_mem_ratio=0.4,
    ):
        if mpi is not None and mpi:
            self.mpi = True
        else:
            self.mpi = None
            self.prule = None

        self._scratch = scratch
        self._restart_dir = restart_dir
        self.stack_mem = stack_mem
        self.stack_mem_ratio = stack_mem_ratio
        self.symm_type = symm_type
        self.clean_scratch = clean_scratch
        bw = self.bw

        if n_threads is None:
            n_threads = bw.b.Global.threading.n_threads_global
        bw.b.Global.threading = bw.b.Threading(
            bw.b.ThreadingTypes.OperatorBatchedGEMM | bw.b.ThreadingTypes.Global,
            n_threads,
            n_threads,
            1,
        )
        bw.b.Global.threading.seq_type = bw.b.SeqTypes.Tasked
        self.reorder_idx = None
        self.pg = "c1"
        self.orb_sym = None
        self.ghamil = None

    @property
    def symm_type(self):
        return self._symm_type

    @symm_type.setter
    def symm_type(self, symm_type):
        self._symm_type = symm_type
        self.set_symm_type(symm_type)

    @property
    def scratch(self):
        return self._scratch

    @scratch.setter
    def scratch(self, scratch):
        self._scratch = scratch
        self.frame.save_dir = scratch
        self.frame.mps_dir = scratch
        self.frame.mpo_dir = scratch

    @property
    def restart_dir(self):
        return self._restart_dir

    @restart_dir.setter
    def restart_dir(self, restart_dir):
        self._restart_dir = restart_dir
        self.frame.restart_dir = restart_dir

    def divide_nprocs(self, n):
        """almost evenly divide n procs to two levels.
        Faster than a pure sqrt method when n >= 20000000."""
        bw = self.bw
        fcts = bw.b.Prime().factors(n)
        px = []
        for p, x in fcts:
            px += [p] * x
        if len(px) == 1:
            return 1, n
        elif len(px) == 2:
            return px[0], px[1]
        elif px[-1] >= n // px[-1]:
            return n // px[-1], px[-1]
        else:
            nx = bw.b.Prime.sqrt(n)
            for p in range(nx, 0, -1):
                if n % p == 0:
                    return p, n // p

    def parallelize_integrals(self, para_type, h1e, g2e, const, msize=None, mrank=None):
        import numpy as np

        if para_type == ParallelTypes.Nothing or self.mpi is None:
            return h1e, g2e, const

        if msize is None:
            msize = self.mpi.size
        if mrank is None:
            mrank = self.mpi.rank

        # not a good strategy
        if para_type in [
            ParallelTypes.MixUIJUKL,
            ParallelTypes.MixUKLUIJ,
            ParallelTypes.MixUIJSI,
        ]:
            pt_high = {
                ParallelTypes.MixUIJUKL: ParallelTypes.UIJ,
                ParallelTypes.MixUKLUIJ: ParallelTypes.UKL,
                ParallelTypes.MixUIJSI: ParallelTypes.UIJ,
            }[para_type]
            pt_low = {
                ParallelTypes.MixUIJUKL: ParallelTypes.UKL,
                ParallelTypes.MixUKLUIJ: ParallelTypes.UIJ,
                ParallelTypes.MixUIJSI: ParallelTypes.SI,
            }[para_type]
            na, nb = self.divide_nprocs(msize)
            ra, rb = mrank // nb, mrank % nb
            h1e, g2e, const = self.parallelize_integrals(
                pt_high, h1e, g2e, const, msize=na, mrank=ra
            )
            return self.parallelize_integrals(
                pt_low, h1e, g2e, const, msize=nb, mrank=rb
            )

        if h1e is not None:
            ixs = np.mgrid[tuple(slice(x) for x in h1e.shape)].reshape((h1e.ndim, -1)).T
            sixs = np.sort(ixs, axis=1)
        if g2e is not None:
            gixs = (
                np.mgrid[tuple(slice(x) for x in g2e.shape)].reshape((g2e.ndim, -1)).T
            )
            gsixs = np.sort(gixs, axis=1)

        if para_type == ParallelTypes.I:
            if h1e is not None:
                mask1 = ixs[:, 0] % msize == mrank
            if g2e is not None:
                mask2 = gixs[:, 0] % msize == mrank
        elif para_type == ParallelTypes.SI:
            if h1e is not None:
                mask1 = sixs[:, 0] % msize == mrank
            if g2e is not None:
                mask2 = gsixs[:, 0] % msize == mrank
        elif para_type == ParallelTypes.J:
            if h1e is not None:
                mask1 = ixs[:, 1] % msize == mrank
            if g2e is not None:
                mask2 = gixs[:, 1] % msize == mrank
        elif para_type == ParallelTypes.SJ:
            if h1e is not None:
                mask1 = sixs[:, 1] % msize == mrank
            if g2e is not None:
                mask2 = gsixs[:, 1] % msize == mrank
        elif para_type == ParallelTypes.SIJ:
            if h1e is not None:
                mask1 = sixs[:, 0] % msize == mrank
            if g2e is not None:
                mask2a = (gsixs[:, 1] == gsixs[:, 2]) & (gsixs[:, 1] % msize == mrank)
                mask2b = (
                    (gsixs[:, 1] != gsixs[:, 2])
                    & (gsixs[:, 0] <= gsixs[:, 1])
                    & (
                        (gsixs[:, 1] * (gsixs[:, 1] + 1) // 2 + gsixs[:, 0]) % msize
                        == mrank
                    )
                )
                mask2c = (
                    (gsixs[:, 1] != gsixs[:, 2])
                    & (gsixs[:, 0] > gsixs[:, 1])
                    & (
                        (gsixs[:, 0] * (gsixs[:, 0] + 1) // 2 + gsixs[:, 1]) % msize
                        == mrank
                    )
                )
                mask2 = mask2a | mask2b | mask2c
        elif para_type == ParallelTypes.SKL:
            if h1e is not None:
                mask1 = sixs[:, 1] % msize == mrank
            if g2e is not None:
                mask2a = (gsixs[:, 1] == gsixs[:, 2]) & (gsixs[:, 1] % msize == mrank)
                mask2b = (
                    (gsixs[:, 1] != gsixs[:, 2])
                    & (gsixs[:, 2] <= gsixs[:, 3])
                    & (
                        (gsixs[:, 3] * (gsixs[:, 3] + 1) // 2 + gsixs[:, 2]) % msize
                        == mrank
                    )
                )
                mask2c = (
                    (gsixs[:, 1] != gsixs[:, 2])
                    & (gsixs[:, 2] > gsixs[:, 3])
                    & (
                        (gsixs[:, 2] * (gsixs[:, 2] + 1) // 2 + gsixs[:, 3]) % msize
                        == mrank
                    )
                )
                mask2 = mask2a | mask2b | mask2c
        elif para_type in [ParallelTypes.UIJ, ParallelTypes.UIJM2, ParallelTypes.UIJM4]:
            ptm = {
                ParallelTypes.UIJ: 1,
                ParallelTypes.UIJM2: 2,
                ParallelTypes.UIJM4: 4,
            }[para_type]
            if h1e is not None:
                mask1 = sixs[:, 0] // ptm % msize == mrank
            if g2e is not None:
                mask2a = (gsixs[:, 0] <= gsixs[:, 1]) & (
                    (gsixs[:, 1] * (gsixs[:, 1] + 1) // 2 + gsixs[:, 0]) // ptm % msize
                    == mrank
                )
                mask2b = (gsixs[:, 0] > gsixs[:, 1]) & (
                    (gsixs[:, 0] * (gsixs[:, 0] + 1) // 2 + gsixs[:, 1]) // ptm % msize
                    == mrank
                )
                mask2 = mask2a | mask2b
        elif para_type in [ParallelTypes.UKL, ParallelTypes.UKLM2, ParallelTypes.UKLM4]:
            ptm = {
                ParallelTypes.UKL: 1,
                ParallelTypes.UKLM2: 2,
                ParallelTypes.UKLM4: 4,
            }[para_type]
            if h1e is not None:
                mask1 = sixs[:, 1] // ptm % msize == mrank
            if g2e is not None:
                mask2a = (gsixs[:, 2] <= gsixs[:, 3]) & (
                    (gsixs[:, 3] * (gsixs[:, 3] + 1) // 2 + gsixs[:, 2]) // ptm % msize
                    == mrank
                )
                mask2b = (gsixs[:, 2] > gsixs[:, 3]) & (
                    (gsixs[:, 2] * (gsixs[:, 2] + 1) // 2 + gsixs[:, 3]) // ptm % msize
                    == mrank
                )
                mask2 = mask2a | mask2b
        if h1e is not None:
            h1e[~mask1.reshape(h1e.shape)] = 0.0
        if g2e is not None:
            g2e[~mask2.reshape(g2e.shape)] = 0.0
        if self.mpi.rank != self.mpi.root:
            const = 0
        return h1e, g2e, const

    def set_symm_type(self, symm_type, reset_frame=True):
        self.bw = Block2Wrapper(symm_type)
        bw = self.bw

        # reset_frame only required when switching between dp/sp
        if reset_frame:
            if SymmetryTypes.SP not in bw.symm_type:
                bw.b.Global.frame = bw.b.DoubleDataFrame(
                    int(self.stack_mem * 0.1),
                    int(self.stack_mem * 0.9),
                    self.scratch,
                    self.stack_mem_ratio,
                    self.stack_mem_ratio,
                )
                bw.b.Global.frame.fp_codec = bw.b.DoubleFPCodec(1e-16, 1024)
                bw.b.Global.frame_float = None
                self.frame = bw.b.Global.frame
            else:
                bw.b.Global.frame_float = bw.b.FloatDataFrame(
                    int(self.stack_mem * 0.1),
                    int(self.stack_mem * 0.9),
                    self.scratch,
                    self.stack_mem_ratio,
                    self.stack_mem_ratio,
                )
                bw.b.Global.frame_float.fp_codec = bw.b.FloatFPCodec(1e-16, 1024)
                bw.b.Global.frame = None
                self.frame = bw.b.Global.frame_float
        self.frame.minimal_disk_usage = True
        self.frame.use_main_stack = False

        if self.mpi:
            self.mpi = bw.brs.MPICommunicator()
            self.prule = bw.bs.ParallelRuleSimple(
                bw.b.ParallelSimpleTypes.Nothing, self.mpi
            )

        if self.restart_dir is not None:
            import os

            if self.mpi is None or self.mpi.rank == self.mpi.root:
                if not os.path.isdir(self.restart_dir):
                    os.makedirs(self.restart_dir)
            if self.mpi is not None:
                self.mpi.barrier()
            self.frame.restart_dir = self.restart_dir

    def initialize_system(
        self,
        n_sites,
        n_elec=0,
        spin=0,
        pg_irrep=None,
        orb_sym=None,
        heis_twos=-1,
        heis_twosz=0,
        singlet_embedding=True,
    ):
        bw = self.bw
        import numpy as np

        self.vacuum = bw.SX(0, 0, 0)
        if heis_twos != -1 and bw.SX == bw.b.SU2 and n_elec == 0:
            n_elec = n_sites * heis_twos
        elif heis_twos == 1 and SymmetryTypes.SGB in bw.symm_type and n_elec != 0:
            n_elec = 2 * n_elec - n_sites
        if pg_irrep is None:
            if hasattr(self, "pg_irrep"):
                pg_irrep = self.pg_irrep
            else:
                pg_irrep = 0
        if not SymmetryTypes.SU2 in bw.symm_type or heis_twos != -1:
            singlet_embedding = False
        if singlet_embedding:
            assert heis_twosz == 0
            self.target = bw.SX(n_elec + spin % 2, 0, pg_irrep)
            self.left_vacuum = bw.SX(spin % 2, spin, 0)
        else:
            self.target = bw.SX(
                n_elec if heis_twosz == 0 else heis_twosz, spin, pg_irrep
            )
            self.left_vacuum = self.vacuum
        self.n_sites = n_sites
        self.heis_twos = heis_twos
        if orb_sym is None:
            self.orb_sym = bw.b.VectorUInt8([0] * self.n_sites)
        else:
            if np.array(orb_sym).ndim == 2:
                self.orb_sym = bw.b.VectorUInt8(list(orb_sym[0]) + list(orb_sym[1]))
            else:
                self.orb_sym = bw.b.VectorUInt8(orb_sym)
        self.ghamil = bw.bs.GeneralHamiltonian(
            self.vacuum, self.n_sites, self.orb_sym, self.heis_twos
        )

    def write_fcidump(self, h1e, g2e, ecore=0, filename=None, h1e_symm=False, pg="d2h"):
        bw = self.bw
        import numpy as np

        fcidump = bw.bx.FCIDUMP()
        swap_pg = getattr(bw.b.PointGroup, "swap_" + pg)
        fw_map = np.array([swap_pg(x) for x in range(1, 9)])
        bk_map = np.argsort(fw_map) + 1
        if SymmetryTypes.SZ in bw.symm_type:
            if not h1e_symm:
                mh1e = tuple(x.flatten() for x in h1e)
            else:
                mh1e = tuple(x[np.tril_indices(len(x))] for x in h1e)
            mg2e = tuple(x.flatten() for x in g2e)
            mg2e = (mg2e[0], mg2e[2], mg2e[1])
            fcidump.initialize_sz(
                self.n_sites,
                self.target.n,
                self.target.twos,
                bk_map[self.target.pg],
                ecore,
                mh1e,
                mg2e,
            )
        elif g2e is not None:
            if not h1e_symm:
                mh1e = h1e.flatten()
            else:
                mh1e = h1e[np.tril_indices(len(h1e))]
            fcidump.initialize_su2(
                self.n_sites,
                self.target.n,
                self.target.twos,
                bk_map[self.target.pg],
                ecore,
                mh1e,
                g2e.flatten(),
            )
        else:
            fcidump.initialize_h1e(
                self.n_sites,
                self.target.n,
                self.target.twos,
                bk_map[self.target.pg],
                ecore,
                h1e.flatten(),
            )
        if pg == "d2h" or pg == "c2v":
            orb_sym = [x % 10 for x in self.orb_sym]
        else:
            orb_sym = self.orb_sym
        fcidump.orb_sym = bw.b.VectorUInt8([bk_map[x] for x in orb_sym])
        if filename is not None:
            fcidump.write(filename)
        return fcidump

    def read_fcidump(self, filename, pg="d2h", rescale=None, iprint=1):
        bw = self.bw
        fcidump = bw.bx.FCIDUMP()
        fcidump.read(filename)
        self.pg = pg
        swap_pg = getattr(bw.b.PointGroup, "swap_" + pg)
        self.orb_sym = bw.b.VectorUInt8(map(swap_pg, fcidump.orb_sym))
        for x in self.orb_sym:
            if x == 8:
                raise RuntimeError("Wrong point group symmetry : ", pg)
        self.n_sites = fcidump.n_sites
        self.n_elec = fcidump.n_elec
        self.spin = fcidump.twos
        self.pg_irrep = swap_pg(fcidump.isym)
        if rescale is not None:
            if iprint >= 1:
                print("original const = ", fcidump.const_e)
            if isinstance(rescale, float):
                fcidump.rescale(rescale)
            elif rescale:
                fcidump.rescale()
            if iprint >= 1:
                print("rescaled const = ", fcidump.const_e)
        self.const_e = fcidump.const_e
        self.ecore = fcidump.const_e
        import numpy as np

        symm_err = fcidump.symmetrize(self.orb_sym)
        if iprint >= 1:
            print("symmetrize error = ", symm_err)

        nn = self.n_sites
        mm = nn * (nn + 1) // 2
        ll = mm * (mm + 1) // 2
        if not fcidump.uhf:
            self.h1e = np.array(fcidump.h1e_matrix(), copy=False).reshape((nn, nn))
            if fcidump.general:
                self.g2e = np.array(fcidump.g2e_1fold(), copy=False).reshape(
                    (nn, nn, nn, nn)
                )
            else:
                self.g2e = np.array(fcidump.g2e_8fold(), copy=False).reshape((ll,))
        else:
            self.h1e = tuple(
                np.array(fcidump.h1e_matrix(s=s), copy=False).reshape((nn, nn))
                for s in [0, 1]
            )
            if fcidump.general:
                self.g2e = tuple(
                    np.array(fcidump.g2e_1fold(sl=sl, sr=sr), copy=False).reshape(
                        (nn, nn, nn, nn)
                    )
                    for sl, sr in [(0, 0), (0, 1), (1, 1)]
                )
            else:
                self.g2e = (
                    np.array(fcidump.g2e_8fold(sl=0, sr=0), copy=False).reshape((ll,)),
                    np.array(fcidump.g2e_4fold(sl=0, sr=1), copy=False).reshape(
                        (mm, mm)
                    ),
                    np.array(fcidump.g2e_8fold(sl=1, sr=1), copy=False).reshape((ll,)),
                )
        return fcidump

    def su2_to_sgf(self):
        bw = self.bw
        import numpy as np

        gh1e = np.zeros((self.n_sites * 2,) * 2)
        gg2e = np.zeros((self.n_sites * 2,) * 4)

        for i in range(self.n_sites * 2):
            for j in range(i % 2, self.n_sites * 2, 2):
                gh1e[i, j] = self.h1e[i // 2, j // 2]

        for i in range(self.n_sites * 2):
            for j in range(i % 2, self.n_sites * 2, 2):
                for k in range(self.n_sites * 2):
                    for l in range(k % 2, self.n_sites * 2, 2):
                        gg2e[i, j, k, l] = self.g2e[i // 2, j // 2, k // 2, l // 2]

        self.h1e = gh1e
        self.g2e = gg2e
        self.n_sites = self.n_sites * 2
        if hasattr(self, "orb_sym"):
            self.orb_sym = bw.b.VectorUInt8(
                [self.orb_sym[i // 2] for i in range(self.n_sites)]
            )

    def integral_symmetrize(self, orb_sym, h1e=None, g2e=None, hxe=None, iprint=1):
        bw = self.bw
        import numpy as np

        error = 0
        if hxe is not None:
            assert len(orb_sym) == hxe.ndim
            x = [np.array(m, dtype=int) for m in orb_sym]
            mask = 0
            for i in range(hxe.ndim):
                mask = (
                    mask
                    ^ x[i][(None,) * i + (slice(None),) + (None,) * (hxe.ndim - i - 1)]
                )
            mask = mask != 0
            error += np.sum(np.abs(hxe[mask]))
            hxe[mask] = 0
        if SymmetryTypes.SZ in bw.symm_type:
            if h1e is not None:
                x = np.array(orb_sym, dtype=int)
                if x.ndim == 1:
                    mask = (x[:, None] ^ x[None, :]) != 0
                    error += sum(np.sum(np.abs(h[mask])) for h in h1e)
                    h1e[0][mask] = 0
                    h1e[1][mask] = 0
                else:
                    for i in range(len(h1e)):
                        mask = (x[i][:, None] ^ x[i][None, :]) != 0
                        error += np.sum(np.abs(h1e[i][mask]))
                        h1e[i][mask] = 0
            if g2e is not None:
                x = np.array(orb_sym, dtype=int)
                if x.ndim == 1:
                    mask = (
                        x[:, None, None, None]
                        ^ x[None, :, None, None]
                        ^ x[None, None, :, None]
                        ^ x[None, None, None, :]
                    ) != 0
                    error += sum(np.sum(np.abs(g[mask])) for g in g2e) * 0.5
                    error += np.sum(np.abs(g2e[1][mask])) * 0.5
                    g2e[0][mask] = 0
                    g2e[1][mask] = 0
                    g2e[2][mask] = 0
                else:
                    js = [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
                    for i in range(len(g2e)):
                        mask = (
                            x[js[i][0]][:, None, None, None]
                            ^ x[js[i][1]][None, :, None, None]
                            ^ x[js[i][2]][None, None, :, None]
                            ^ x[js[i][3]][None, None, None, :]
                        ) != 0
                        error += np.sum(np.abs(g2e[i][mask])) * 0.5
                        if i == 1:
                            error += np.sum(np.abs(g2e[i][mask])) * 0.5
                        g2e[i][mask] = 0
        else:
            if h1e is not None:
                x = np.array(orb_sym, dtype=int)
                mask = (x[:, None] ^ x[None, :]) != 0
                error += np.sum(np.abs(h1e[mask]))
                h1e[mask] = 0
            if g2e is not None:
                x = np.array(orb_sym, dtype=int)
                mask = (
                    x[:, None, None, None]
                    ^ x[None, :, None, None]
                    ^ x[None, None, :, None]
                    ^ x[None, None, None, :]
                ) != 0
                error += np.sum(np.abs(g2e[mask])) * 0.5
                g2e[mask] = 0
        if iprint:
            print("integral symmetrize error = ", error)

    def get_conventional_qc_mpo(self, fcidump, algo_type=None, iprint=1):
        """This method cannot take care of parallelization!"""
        bw = self.bw
        hamil = bw.bs.HamiltonianQC(self.vacuum, self.n_sites, self.orb_sym, fcidump)
        import time

        tt = time.perf_counter()
        if algo_type is not None and MPOAlgorithmTypes.NC in algo_type:
            mpo = bw.bs.MPOQC(hamil, bw.b.QCTypes.NC, "HQC")
        elif algo_type is not None and MPOAlgorithmTypes.CN in algo_type:
            mpo = bw.bs.MPOQC(hamil, bw.b.QCTypes.CN, "HQC")
        elif algo_type is None or MPOAlgorithmTypes.Conventional in algo_type:
            if hamil.n_sites == 2:
                print(
                    "MPOAlgorithmTypes.Conventional with only 2 sites may cause error!"
                    + "Please use MPOAlgorithmTypes.NC instead!"
                )
            mpo = bw.bs.MPOQC(hamil, bw.b.QCTypes.Conventional, "HQC")
        else:
            raise RuntimeError("Invalid conventional mpo algo type:", algo_type)
        if algo_type is not None and MPOAlgorithmTypes.NoTranspose in algo_type:
            rule = bw.bs.NoTransposeRule(bw.bs.RuleQC())
        else:
            rule = bw.bs.RuleQC()
        use_r_intermediates = (
            algo_type is None or MPOAlgorithmTypes.NoRIntermed not in algo_type
        )

        if iprint:
            nnz, sz, bdim = mpo.get_summary()
            if self.mpi is not None:
                self.mpi.barrier()
            print(
                "Rank = %5d Ttotal = %10.3f MPO method = %s bond dimension = %7d NNZ = %12d SIZE = %12d SPT = %6.4f"
                % (
                    self.mpi.rank if self.mpi is not None else 0,
                    time.perf_counter() - tt,
                    algo_type.name if algo_type is not None else "Conventional",
                    bdim,
                    nnz,
                    sz,
                    (1.0 * sz - nnz) / sz,
                ),
                flush=True,
            )
            if self.mpi is not None:
                self.mpi.barrier()

        mpo = bw.bs.SimplifiedMPO(
            mpo,
            rule,
            True,
            use_r_intermediates,
            bw.b.OpNamesSet((bw.b.OpNames.R, bw.b.OpNames.RD)),
        )
        if self.mpi:
            mpo = bw.bs.ParallelMPO(mpo, self.prule)
        return mpo

    def get_identity_mpo(self):
        return self.get_mpo(self.expr_builder().add_term("", [], 1.0).finalize())

    def unpack_g2e(self, g2e, n_sites=None):
        import numpy as np

        if n_sites is None:
            n_sites = self.n_sites

        if g2e.ndim == 1:
            m = n_sites * (n_sites + 1) // 2
            xtril = np.tril_indices(m)
            r = np.zeros((m ** 2,), dtype=g2e.dtype)
            r[xtril[0] * m + xtril[1]] = g2e
            r[xtril[1] * m + xtril[0]] = g2e
            g2e = r.reshape((m, m))

        if g2e.ndim == 2:
            m = n_sites
            xtril = np.tril_indices(m)
            r = np.zeros((m ** 2, m ** 2), dtype=g2e.dtype)
            r[(xtril[0] * m + xtril[1])[:, None], xtril[0] * m + xtril[1]] = g2e
            r[(xtril[0] * m + xtril[1])[:, None], xtril[1] * m + xtril[0]] = g2e
            r[(xtril[1] * m + xtril[0])[:, None], xtril[0] * m + xtril[1]] = g2e
            r[(xtril[1] * m + xtril[0])[:, None], xtril[1] * m + xtril[0]] = g2e
            g2e = r.reshape((m, m, m, m))

        return g2e

    def get_qc_mpo(
        self,
        h1e,
        g2e,
        ecore=0,
        para_type=None,
        reorder=None,
        cutoff=1e-20,
        integral_cutoff=1e-20,
        post_integral_cutoff=1e-20,
        algo_type=None,
        normal_order_ref=None,
        normal_order_wick=True,
        symmetrize=True,
        sum_mpo_mod=-1,
        compute_accurate_svd_error=True,
        csvd_sparsity=0.0,
        csvd_eps=1e-10,
        csvd_max_iter=1000,
        disjoint_levels=None,
        disjoint_all_blocks=False,
        disjoint_multiplier=1.0,
        block_max_length=False,
        iprint=1,
    ):
        import numpy as np

        bw = self.bw

        if isinstance(g2e, np.ndarray):
            g2e = self.unpack_g2e(g2e)
        elif isinstance(g2e, tuple):
            g2e = tuple(self.unpack_g2e(x) for x in g2e)

        if SymmetryTypes.SZ in bw.symm_type:
            if h1e is not None and isinstance(h1e, np.ndarray) and h1e.ndim == 2:
                h1e = (h1e, h1e)
            if g2e is not None and isinstance(g2e, np.ndarray) and g2e.ndim == 4:
                g2e = (g2e, g2e, g2e)
        elif SymmetryTypes.SGF in bw.symm_type or SymmetryTypes.SGB in bw.symm_type:
            if (
                h1e is not None
                and hasattr(self, "n_sites")
                and len(h1e) * 2 == self.n_sites
            ):
                gh1e = np.zeros((len(h1e) * 2,) * 2)
                gh1e[0::2, 0::2] = gh1e[1::2, 1::2] = h1e
                h1e = gh1e
            if (
                g2e is not None
                and hasattr(self, "n_sites")
                and len(g2e) * 2 == self.n_sites
            ):
                gg2e = np.zeros((len(g2e) * 2,) * 4)
                gg2e[0::2, 0::2, 0::2, 0::2] = gg2e[0::2, 0::2, 1::2, 1::2] = g2e
                gg2e[1::2, 1::2, 0::2, 0::2] = gg2e[1::2, 1::2, 1::2, 1::2] = g2e
                g2e = gg2e

        if symmetrize and self.orb_sym is not None:
            self.integral_symmetrize(self.orb_sym, h1e=h1e, g2e=g2e, iprint=iprint)

        if integral_cutoff != 0:
            error = 0
            if SymmetryTypes.SZ in bw.symm_type:
                if h1e is not None:
                    for i in range(2):
                        mask = np.abs(h1e[i]) < integral_cutoff
                        error += np.sum(np.abs(h1e[i][mask]))
                        h1e[i][mask] = 0
                if g2e is not None:
                    for i in range(3):
                        mask = np.abs(g2e[i]) < integral_cutoff
                        error += np.sum(np.abs(g2e[i][mask])) * (1 if i == 1 else 0.5)
                        g2e[i][mask] = 0
            else:
                if h1e is not None:
                    mask = np.abs(h1e) < integral_cutoff
                    error += np.sum(np.abs(h1e[mask]))
                    h1e[mask] = 0
                if g2e is not None:
                    mask = np.abs(g2e) < integral_cutoff
                    error += np.sum(np.abs(g2e[mask])) * 0.5
                    g2e[mask] = 0
            if iprint:
                print("integral cutoff error = ", error)

        if reorder is not None:
            if isinstance(reorder, np.ndarray):
                idx = reorder
            elif reorder == "irrep":
                assert self.orb_sym is not None
                if self.pg == "d2h":
                    # D2H
                    # 0   1   2   3   4   5   6   7   (XOR)
                    # A1g B1g B2g B3g A1u B1u B2u B3u
                    # optimal
                    # A1g B1u B3u B2g B2u B3g B1g A1u
                    optimal_reorder = [0, 6, 3, 5, 7, 1, 4, 2, 8]
                elif self.pg == "c2v":
                    # C2V
                    # 0  1  2  3  (XOR)
                    # A1 A2 B1 B2
                    # optimal
                    # A1 B1 B2 A2
                    optimal_reorder = [0, 3, 1, 2, 4]
                else:
                    optimal_reorder = [0, 6, 3, 5, 7, 1, 4, 2, 8]
                orb_opt = [optimal_reorder[x] for x in np.array(self.orb_sym)]
                idx = np.argsort(orb_opt)
            elif reorder == "fiedler" or reorder == True:
                idx = self.orbital_reordering(h1e, g2e)
            else:
                raise RuntimeError("Unknown reorder", reorder)
            if iprint:
                print("reordering = ", idx)
            self.reorder_idx = idx
            if SymmetryTypes.SZ in bw.symm_type:
                for i in enumerate(len(h1e)):
                    h1e[i] = h1e[i][idx][:, idx]
                for i in enumerate(len(g2e)):
                    g2e[i] = g2e[i][idx][:, idx][:, :, idx][:, :, :, idx]
            else:
                h1e = h1e[idx][:, idx]
                g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]
            if self.orb_sym is not None:
                self.orb_sym = bw.b.VectorUInt8(np.array(self.orb_sym)[idx])
                if self.ghamil is not None:
                    self.ghamil = bw.bs.GeneralHamiltonian(
                        self.vacuum, self.n_sites, self.orb_sym, self.heis_twos
                    )
        else:
            self.reorder_idx = None

        if para_type is None:
            para_type = ParallelTypes.SIJ

        if SymmetryTypes.SZ in bw.symm_type:
            if h1e is not None:
                h1e = list(h1e)
                h1e[0], _, ecore = self.parallelize_integrals(
                    para_type, h1e[0], None, ecore
                )
                h1e[1], _, _ = self.parallelize_integrals(para_type, h1e[1], None, 0)
            if g2e is not None:
                g2e = list(g2e)
                _, g2e[0], ecore = self.parallelize_integrals(
                    para_type, None, g2e[0], ecore
                )
                _, g2e[1], _ = self.parallelize_integrals(para_type, None, g2e[1], 0)
                _, g2e[2], _ = self.parallelize_integrals(para_type, None, g2e[2], 0)
        else:
            h1e, g2e, ecore = self.parallelize_integrals(para_type, h1e, g2e, ecore)

        if algo_type is not None and MPOAlgorithmTypes.Conventional in algo_type:
            fd = self.write_fcidump(h1e, g2e, ecore=ecore)
            return self.get_conventional_qc_mpo(fd, algo_type=algo_type, iprint=iprint)

        # build Hamiltonian expression
        b = self.expr_builder()

        if normal_order_ref is None:
            if SymmetryTypes.SU2 in bw.symm_type:
                if h1e is not None:
                    b.add_sum_term("(C+D)0", np.sqrt(2) * h1e)
                if g2e is not None:
                    b.add_sum_term("((C+(C+D)0)1+D)0", g2e.transpose(0, 2, 3, 1))
            elif SymmetryTypes.SZ in bw.symm_type:
                if h1e is not None:
                    b.add_sum_term("cd", h1e[0])
                    b.add_sum_term("CD", h1e[1])
                if g2e is not None:
                    b.add_sum_term("ccdd", 0.5 * g2e[0].transpose(0, 2, 3, 1))
                    b.add_sum_term("cCDd", 0.5 * g2e[1].transpose(0, 2, 3, 1))
                    b.add_sum_term("CcdD", 0.5 * g2e[1].transpose(2, 0, 1, 3))
                    b.add_sum_term("CCDD", 0.5 * g2e[2].transpose(0, 2, 3, 1))
            elif SymmetryTypes.SGF in bw.symm_type:
                if h1e is not None:
                    b.add_sum_term("CD", h1e)
                if g2e is not None:
                    b.add_sum_term("CCDD", 0.5 * g2e.transpose(0, 2, 3, 1))
            elif SymmetryTypes.SGB in bw.symm_type:
                h_terms = FermionTransform.jordan_wigner(h1e, g2e)
                for k, (x, v) in h_terms.items():
                    b.add_term(k, x, v)
        else:
            if SymmetryTypes.SU2 in bw.symm_type:
                h1es, g2es, ecore = NormalOrder.make_su2(
                    h1e, g2e, ecore, normal_order_ref, normal_order_wick
                )
            elif SymmetryTypes.SZ in bw.symm_type:
                h1es, g2es, ecore = NormalOrder.make_sz(
                    h1e, g2e, ecore, normal_order_ref, normal_order_wick
                )
            elif SymmetryTypes.SGF in bw.symm_type:
                h1es, g2es, ecore = NormalOrder.make_sgf(
                    h1e, g2e, ecore, normal_order_ref, normal_order_wick
                )

            if self.mpi is not None:
                ec_arr = np.array([ecore], dtype=float)
                self.mpi.reduce_sum(ec_arr, self.mpi.root)
                if self.mpi.rank == self.mpi.root:
                    ecore = ec_arr[0]
                else:
                    ecore = 0.0

            if post_integral_cutoff != 0:
                error = 0
                for k, v in h1es.items():
                    mask = np.abs(v) < post_integral_cutoff
                    error += np.sum(np.abs(v[mask]))
                    h1es[k][mask] = 0
                for k, v in g2es.items():
                    mask = np.abs(v) < post_integral_cutoff
                    error += np.sum(np.abs(v[mask]))
                    g2es[k][mask] = 0
                if iprint:
                    print("post integral cutoff error = ", error)

            for k, v in h1es.items():
                b.add_sum_term(k, v)
            for k, v in g2es.items():
                b.add_sum_term(k, v)

            if iprint:
                print("normal ordered ecore = ", ecore)

        b.add_const(ecore)
        bx = b.finalize(adjust_order=SymmetryTypes.SGB not in bw.symm_type)

        if iprint:
            if self.mpi is not None:
                for i in range(self.mpi.size):
                    self.mpi.barrier()
                    if i == self.mpi.rank:
                        print(
                            "rank = %5d mpo terms = %10d"
                            % (i, sum([len(x) for x in bx.data]))
                        )
                    self.mpi.barrier()
            else:
                print("mpo terms = %10d" % sum([len(x) for x in bx.data]))

        return self.get_mpo(
            bx,
            iprint=iprint,
            cutoff=cutoff,
            algo_type=algo_type,
            sum_mpo_mod=sum_mpo_mod,
            compute_accurate_svd_error=compute_accurate_svd_error,
            csvd_sparsity=csvd_sparsity,
            csvd_eps=csvd_eps,
            csvd_max_iter=csvd_max_iter,
            disjoint_levels=disjoint_levels,
            disjoint_all_blocks=disjoint_all_blocks,
            disjoint_multiplier=disjoint_multiplier,
            block_max_length=block_max_length,
        )

    def get_mpo(
        self,
        expr,
        iprint=0,
        cutoff=1e-14,
        left_vacuum=None,
        algo_type=None,
        sum_mpo_mod=-1,
        compute_accurate_svd_error=True,
        csvd_sparsity=0.0,
        csvd_eps=1e-10,
        csvd_max_iter=1000,
        disjoint_levels=None,
        disjoint_all_blocks=False,
        disjoint_multiplier=1.0,
        block_max_length=False,
    ):
        bw = self.bw
        import time

        tt = time.perf_counter()
        if left_vacuum is None:
            left_vacuum = bw.SX.invalid
        if algo_type is None:
            algo_type = MPOAlgorithmTypes.FastBipartite
        algo_type = getattr(bw.b.MPOAlgorithmTypes, algo_type.name)
        mpo = bw.bs.GeneralMPO(self.ghamil, expr, algo_type, cutoff, -1, iprint)
        mpo.left_vacuum = left_vacuum
        mpo.sum_mpo_mod = sum_mpo_mod
        mpo.compute_accurate_svd_error = compute_accurate_svd_error
        mpo.csvd_sparsity = csvd_sparsity
        mpo.csvd_eps = csvd_eps
        mpo.csvd_max_iter = csvd_max_iter
        if disjoint_levels is not None:
            mpo.disjoint_levels = bw.VectorFP(disjoint_levels)
        mpo.disjoint_all_blocks = disjoint_all_blocks
        mpo.disjoint_multiplier = disjoint_multiplier
        mpo.block_max_length = block_max_length
        mpo.build()

        if iprint:
            nnz, sz, bdim = mpo.get_summary()
            if self.mpi is not None:
                self.mpi.barrier()
            print(
                "Rank = %5d Ttotal = %10.3f MPO method = %s bond dimension = %7d NNZ = %12d SIZE = %12d SPT = %6.4f"
                % (
                    self.mpi.rank if self.mpi is not None else 0,
                    time.perf_counter() - tt,
                    algo_type.name,
                    bdim,
                    nnz,
                    sz,
                    (1.0 * sz - nnz) / sz,
                ),
                flush=True,
            )
            if self.mpi is not None:
                self.mpi.barrier()

        mpo = bw.bs.SimplifiedMPO(mpo, bw.bs.Rule(), False, False)
        mpo = bw.bs.IdentityAddedMPO(mpo)
        if self.mpi:
            mpo = bw.bs.ParallelMPO(mpo, self.prule)
        return mpo

    def get_spin_square_mpo(self, iprint=1):
        import numpy as np

        bw = self.bw
        b = self.expr_builder()

        if SymmetryTypes.SU2 in bw.symm_type:
            ix2 = np.mgrid[: self.n_sites, : self.n_sites].reshape((2, -1))
            if self.reorder_idx is not None:
                ridx = np.argsort(self.reorder_idx)
                ix2 = np.array(ridx)[ix2]
            b.add_terms(
                "((C+D)2+(C+D)2)0",
                -np.sqrt(3) / 2 * np.ones(ix2.shape[1]),
                np.array([ix2[0], ix2[0], ix2[1], ix2[1]]).T,
            )
        elif SymmetryTypes.SZ in bw.symm_type:
            ix1 = np.mgrid[: self.n_sites].flatten()
            ix2 = np.mgrid[: self.n_sites, : self.n_sites].reshape((2, -1))
            if self.reorder_idx is not None:
                ridx = np.argsort(self.reorder_idx)
                ix1 = np.array(ridx)[ix1]
                ix2 = np.array(ridx)[ix2]
            b.add_terms("cd", 0.75 * np.ones(ix1.shape[0]), np.array([ix1, ix1]).T)
            b.add_terms("CD", 0.75 * np.ones(ix1.shape[0]), np.array([ix1, ix1]).T)
            b.add_terms(
                "ccdd",
                0.25 * np.ones(ix2.shape[1]),
                np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
            )
            b.add_terms(
                "cCDd",
                -0.25 * np.ones(ix2.shape[1]),
                np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
            )
            b.add_terms(
                "CcdD",
                -0.25 * np.ones(ix2.shape[1]),
                np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
            )
            b.add_terms(
                "CCDD",
                0.25 * np.ones(ix2.shape[1]),
                np.array([ix2[0], ix2[1], ix2[1], ix2[0]]).T,
            )
            b.add_terms(
                "cCDd",
                -0.5 * np.ones(ix2.shape[1]),
                np.array([ix2[0], ix2[1], ix2[0], ix2[1]]).T,
            )
            b.add_terms(
                "CcdD",
                -0.5 * np.ones(ix2.shape[1]),
                np.array([ix2[1], ix2[0], ix2[1], ix2[0]]).T,
            )
        elif SymmetryTypes.SGF in bw.symm_type:
            ixa1 = np.mgrid[0 : self.n_sites : 2].flatten()
            ixb1 = np.mgrid[1 : self.n_sites : 2].flatten()
            ixaa2 = np.mgrid[0 : self.n_sites : 2, 0 : self.n_sites : 2].reshape(
                (2, -1)
            )
            ixab2 = np.mgrid[0 : self.n_sites : 2, 1 : self.n_sites : 2].reshape(
                (2, -1)
            )
            ixba2 = np.mgrid[1 : self.n_sites : 2, 0 : self.n_sites : 2].reshape(
                (2, -1)
            )
            ixbb2 = np.mgrid[1 : self.n_sites : 2, 1 : self.n_sites : 2].reshape(
                (2, -1)
            )
            if self.reorder_idx is not None:
                ridx = np.argsort(self.reorder_idx)
                ixa1 = np.array(ridx)[ixa1]
                ixb1 = np.array(ridx)[ixb1]
                ixaa2 = np.array(ridx)[ixaa2]
                ixab2 = np.array(ridx)[ixab2]
                ixba2 = np.array(ridx)[ixba2]
                ixbb2 = np.array(ridx)[ixbb2]
            b.add_terms("CD", 0.75 * np.ones(ixa1.shape[0]), np.array([ixa1, ixa1]).T)
            b.add_terms("CD", 0.75 * np.ones(ixb1.shape[0]), np.array([ixb1, ixb1]).T)
            b.add_terms(
                "CCDD",
                0.25 * np.ones(ixaa2.shape[1]),
                np.array([ixaa2[0], ixaa2[1], ixaa2[1], ixaa2[0]]).T,
            )
            b.add_terms(
                "CCDD",
                -0.25 * np.ones(ixab2.shape[1]),
                np.array([ixab2[0], ixab2[1], ixab2[1], ixab2[0]]).T,
            )
            b.add_terms(
                "CCDD",
                -0.25 * np.ones(ixba2.shape[1]),
                np.array([ixba2[0], ixba2[1], ixba2[1], ixba2[0]]).T,
            )
            b.add_terms(
                "CCDD",
                0.25 * np.ones(ixbb2.shape[1]),
                np.array([ixbb2[0], ixbb2[1], ixbb2[1], ixbb2[0]]).T,
            )
            b.add_terms(
                "CCDD",
                -0.5 * np.ones(ixab2.shape[1]),
                np.array([ixab2[0], ixab2[1], ixba2[0], ixba2[1]]).T,
            )
            b.add_terms(
                "CCDD",
                -0.5 * np.ones(ixba2.shape[1]),
                np.array([ixab2[1], ixab2[0], ixba2[1], ixba2[0]]).T,
            )

        if self.mpi is not None:
            b.iscale(1.0 / self.mpi.size)

        bx = b.finalize()
        return self.get_mpo(bx, iprint)

    def orbital_reordering(self, h1e, g2e):
        bw = self.bw
        import numpy as np

        xmat = np.abs(np.einsum("ijji->ij", g2e, optimize=True))
        kmat = np.abs(h1e) * 1e-7 + xmat
        kmat = bw.b.VectorDouble(kmat.flatten())
        idx = bw.b.OrbitalOrdering.fiedler(len(h1e), kmat)
        return np.array(idx)

    def dmrg(
        self,
        mpo,
        ket,
        n_sweeps=10,
        tol=1e-8,
        bond_dims=None,
        noises=None,
        thrds=None,
        iprint=0,
        dav_type=None,
        cutoff=1e-20,
        dav_max_iter=4000,
        proj_mpss=None,
        proj_weights=None,
        store_wfn_spectra=True,
        spectra_with_multiplicity=False,
    ):
        bw = self.bw
        if bond_dims is None:
            bond_dims = [ket.info.bond_dim]
        if noises is None:
            noises = [1e-5] * 5 + [0]
        if thrds is None:
            if SymmetryTypes.SP not in bw.symm_type:
                thrds = [1e-6] * 4 + [1e-7] * 1
            else:
                thrds = [1e-5] * 4 + [5e-6] * 1
        if dav_type is not None and "LeftEigen" in dav_type:
            bra = ket.deep_copy("BRA")
        else:
            bra = ket
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "DMRG")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        dmrg = bw.bs.DMRG(me, bw.b.VectorUBond(bond_dims), bw.VectorFP(noises))

        if proj_mpss is not None:
            assert proj_weights is not None
            assert len(proj_weights) == len(proj_mpss)
            dmrg.projection_weights = bw.VectorFP(proj_weights)
            dmrg.ext_mpss = bw.bs.VectorMPS(proj_mpss)
            impo = self.get_identity_mpo()
            for ext_mps in dmrg.ext_mpss:
                self.align_mps_center(ext_mps, ket)
                ext_me = bw.bs.MovingEnvironment(
                    impo, ket, ext_mps, "PJ" + ext_mps.info.tag
                )
                ext_me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
                ext_me.init_environments(iprint >= 2)
                dmrg.ext_mes.append(ext_me)

        if dav_type is not None:
            dmrg.davidson_type = getattr(bw.b.DavidsonTypes, dav_type)
        dmrg.noise_type = bw.b.NoiseTypes.ReducedPerturbative
        dmrg.davidson_conv_thrds = bw.VectorFP(thrds)
        dmrg.davidson_max_iter = dav_max_iter + 100
        dmrg.davidson_soft_max_iter = dav_max_iter
        dmrg.store_wfn_spectra = store_wfn_spectra
        dmrg.iprint = iprint
        dmrg.cutoff = cutoff
        dmrg.trunc_type = dmrg.trunc_type | bw.b.TruncationTypes.RealDensityMatrix
        if spectra_with_multiplicity:
            dmrg.trunc_type = (
                dmrg.trunc_type | bw.b.TruncationTypes.SpectraWithMultiplicity
            )
        self._dmrg = dmrg
        if n_sweeps == -1:
            return None
        me.init_environments(iprint >= 2)
        ener = dmrg.solve(n_sweeps, ket.center == 0, tol)

        if self.clean_scratch:
            dmrg.me.remove_partition_files()
            for me in dmrg.ext_mes:
                me.remove_partition_files()

        ket.info.bond_dim = max(ket.info.bond_dim, bond_dims[-1])
        if isinstance(ket, bw.bs.MultiMPS):
            ener = list(dmrg.energies[-1])
        if self.mpi is not None:
            self.mpi.barrier()
        return ener

    def get_dmrg_results(self):
        import numpy as np

        energies = np.array(self._dmrg.energies)
        dws = np.array(self._dmrg.discarded_weights)
        bond_dims = np.array(self._dmrg.bond_dims)[: len(energies)]
        return bond_dims, dws, energies

    def get_bipartite_entanglement(self):
        import numpy as np

        bip_ent = np.zeros(len(self._dmrg.sweep_wfn_spectra), dtype=np.float64)
        for ix, x in enumerate(self._dmrg.sweep_wfn_spectra):
            ldsq = np.array(x, dtype=np.float128) ** 2
            ldsq = ldsq[ldsq != 0]
            bip_ent[ix] = float(np.sum(-ldsq * np.log(ldsq)))
        return bip_ent

    def get_n_orb_rdm_mpos(self, orb_type=1, ij_symm=True, iprint=0):
        bw = self.bw
        if SymmetryTypes.SU2 in bw.symm_type:
            return NotImplemented
        elif SymmetryTypes.SZ in bw.symm_type:
            if orb_type == 1:
                h_terms = OrbitalEntropy.get_one_orb_rdm_h_terms(self.n_sites)
            else:
                h_terms = OrbitalEntropy.get_two_orb_rdm_h_terms(
                    self.n_sites, ij_symm=ij_symm
                )
            mpos = {}
            for ih, htss in h_terms.items():
                if iprint and (self.mpi is None or self.mpi.rank == self.mpi.root):
                    print("%d-ORB RDM MPO site = %r" % (orb_type, ih))
                i_mpos = []
                for hts in htss:
                    b = self.expr_builder()
                    for k, (x, v) in hts.items():
                        b.add_term(k, x, v)
                    if self.mpi is not None:
                        b.iscale(1.0 / self.mpi.size)
                    mpo = self.get_mpo(
                        b.finalize(adjust_order=False), 1 if iprint >= 2 else 0
                    )
                    i_mpos.append(mpo)
                mpos[ih] = i_mpos
        else:
            return NotImplemented

        return mpos

    def get_orbital_entropies(self, ket, orb_type=1, ij_symm=True, iprint=0):
        bw = self.bw
        import numpy as np

        mpos = self.get_n_orb_rdm_mpos(
            orb_type=orb_type, ij_symm=ij_symm, iprint=iprint
        )
        ents = np.zeros((self.n_sites,) * orb_type)
        mket = ket.deep_copy(ket.info.tag + "@ORB-ENT-TMP")
        for ih, i_mpos in mpos.items():
            if iprint and (self.mpi is None or self.mpi.rank == self.mpi.root):
                print("%d-ORB RDM EXPECT site = %r" % (orb_type, ih))
            x = []
            for mpo in i_mpos:
                x.append(self.expectation(mket, mpo, mket, iprint=iprint))
            ld = np.array(x)
            if SymmetryTypes.SU2 in bw.symm_type:
                return NotImplemented
            elif SymmetryTypes.SZ in bw.symm_type:
                if orb_type == 1:
                    assert len(ld) == 4
                elif orb_type == 2:
                    ld = OrbitalEntropy.get_two_orb_rdm_eigvals(ld)
            else:
                return NotImplemented
            ld = ld[ld != 0]
            ent = float(np.sum(-ld * np.log(ld)))
            ents[ih] = ent
        if orb_type == 2 and ij_symm:
            for ih in mpos:
                ents[ih[::-1]] = ents[ih]
        return ents

    def get_orbital_interaction_matrix(self, ket, iprint=0):
        import numpy as np

        s1 = self.get_orbital_entropies(ket, orb_type=1, iprint=iprint)
        s2 = self.get_orbital_entropies(ket, orb_type=2, iprint=iprint)
        return 0.5 * (s1[:, None] + s1[None, :] - s2) * (1 - np.identity(len(s1)))

    def get_conventional_npdm(
        self, ket, pdm_type=1, bra=None, soc=False, site_type=1, iprint=0
    ):
        bw = self.bw
        import numpy as np

        if self.mpi is not None:
            self.mpi.barrier()

        mket = ket.deep_copy("PDM-KET@TMP")
        mpss = [mket]
        if bra is not None and bra != ket:
            mbra = bra.deep_copy("PDM-BRA@TMP")
            mpss.append(mbra)
        else:
            mbra = mket
        for mps in mpss:
            if mps.dot == 2 and site_type != 2:
                mps.dot = 1
                if mps.center == mps.n_sites - 2:
                    mps.center = mps.n_sites - 1
                    mps.canonical_form = mps.canonical_form[:-1] + "S"
                elif mps.center == 0:
                    mps.canonical_form = "K" + mps.canonical_form[1:]
                else:
                    assert False
                mps.save_data()

            if self.mpi is not None:
                self.mpi.barrier()

            mps.load_mutable()
            mps.info.bond_dim = max(
                mps.info.bond_dim, mps.info.get_max_bond_dimension()
            )

        self.align_mps_center(mbra, mket)
        hamil = bw.bs.HamiltonianQC(
            self.vacuum, self.n_sites, self.orb_sym, bw.bx.FCIDUMP()
        )
        if pdm_type == 1:
            pmpo = bw.bs.PDM1MPOQC(hamil, 1 if soc else 0)
        elif pdm_type == 2:
            pmpo = bw.bs.PDM2MPOQC(hamil)
        else:
            raise NotImplementedError()
        if mbra == mket:
            pmpo = bw.bs.SimplifiedMPO(pmpo, bw.bs.RuleQC())
        else:
            pmpo = bw.bs.SimplifiedMPO(pmpo, bw.bs.NoTransposeRule(bw.bs.RuleQC()))
        if self.mpi:
            if pdm_type == 1:
                prule = bw.bs.ParallelRulePDM1QC(self.mpi)
            elif pdm_type == 2:
                prule = bw.bs.ParallelRulePDM2QC(self.mpi)
            pmpo = bw.bs.ParallelMPO(pmpo, prule)

        if soc:
            mb_lv = mbra.info.left_dims_fci[0].quanta[0]
            ml_lv = mket.info.left_dims_fci[0].quanta[0]
            if mb_lv != ml_lv:
                raise RuntimeError(
                    "SOC 1PDM cannot be done with singlet_embedding for mpss"
                    + " with different spins! Please consider setting "
                    + "singlet_embedding=False in DMRGDriver.initialize_system(...)."
                )

        pme = bw.bs.MovingEnvironment(pmpo, mbra, mket, "NPDM")
        pme.init_environments(iprint >= 2)
        pme.cached_contraction = True
        expect = bw.bs.Expect(pme, mbra.info.bond_dim, mket.info.bond_dim)
        if site_type == 0:
            expect.zero_dot_algo = True
        # expect.algo_type = bw.b.ExpectationAlgorithmTypes.Normal
        expect.iprint = iprint
        expect.solve(True, mket.center == 0)

        if self.clean_scratch:
            expect.me.remove_partition_files()

        if pdm_type == 1:
            if SymmetryTypes.SZ in bw.symm_type:
                dmr = expect.get_1pdm(self.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                dm = dm.reshape((self.n_sites, 2, self.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))
                dm = np.concatenate(
                    [dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0
                )
            elif SymmetryTypes.SU2 in bw.symm_type:
                dmr = expect.get_1pdm_spatial(self.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                if soc:
                    if SymmetryTypes.SU2 in bw.symm_type:
                        if hasattr(mbra.info, "targets"):
                            qsbra = mbra.info.targets[0].twos
                        else:
                            qsbra = mbra.info.target.twos
                        # fix different Wigner–Eckart theorem convention
                        dm *= np.sqrt(qsbra + 1)
                    dm = dm / np.sqrt(2)
            else:
                dmr = expect.get_1pdm(self.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()

            if self.reorder_idx is not None:
                rev_idx = np.argsort(self.reorder_idx)
                dm = dm[..., rev_idx, :][..., :, rev_idx]
        elif pdm_type == 2:
            dmr = expect.get_2pdm(self.n_sites)
            dm = np.array(dmr, copy=True)
            dm = dm.reshape(
                (self.n_sites, 2, self.n_sites, 2, self.n_sites, 2, self.n_sites, 2)
            )
            dm = np.transpose(dm, (0, 2, 4, 6, 1, 3, 5, 7))
            dm = np.concatenate(
                [
                    dm[None, :, :, :, :, 0, 0, 0, 0],
                    dm[None, :, :, :, :, 0, 1, 1, 0],
                    dm[None, :, :, :, :, 1, 1, 1, 1],
                ],
                axis=0,
            )
            if self.reorder_idx is not None:
                dm[:, :, :, :, :] = dm[:, rev_idx, :, :, :][:, :, rev_idx, :, :][
                    :, :, :, rev_idx, :
                ][:, :, :, :, rev_idx]

        if self.mpi is not None:
            self.mpi.barrier()
        return dm

    def get_conventional_1pdm(self, ket, *args, **kwargs):
        return self.get_conventional_npdm(ket, pdm_type=1, *args, **kwargs)

    def get_conventional_2pdm(self, ket, *args, **kwargs):
        return self.get_conventional_npdm(ket, pdm_type=2, *args, **kwargs)

    def get_conventional_trans_1pdm(self, bra, ket, *args, **kwargs):
        return self.get_conventional_npdm(ket, pdm_type=1, bra=bra, *args, **kwargs)

    def get_conventional_trans_2pdm(self, bra, ket, *args, **kwargs):
        return self.get_conventional_npdm(ket, pdm_type=2, bra=bra, *args, **kwargs)

    def get_npdm(
        self,
        ket,
        pdm_type=1,
        bra=None,
        soc=False,
        site_type=0,
        algo_type=None,
        su2_coupling=None,
        simulated_parallel=0,
        fused_contraction_rotation=True,
        cutoff=1e-24,
        iprint=0,
    ):
        bw = self.bw
        import numpy as np

        if algo_type is None:
            algo_type = NPDMAlgorithmTypes.Default

        if NPDMAlgorithmTypes.Conventional in algo_type or soc:
            return self.get_conventional_npdm(
                ket, pdm_type, bra, soc, site_type, iprint
            )

        if self.mpi is not None:
            self.mpi.barrier()

        if SymmetryTypes.SU2 in bw.symm_type:
            if su2_coupling is None:
                su2_coupling = "((C+%s)1+D)0"
            op_str = "(C+D)0"
            for _ in range(pdm_type - 1):
                op_str = su2_coupling % op_str
            perm = bw.b.SpinPermScheme.initialize_su2(pdm_type * 2, op_str, True)
            perms = bw.b.VectorSpinPermScheme([perm])
        elif SymmetryTypes.SZ in bw.symm_type:
            op_str = ["cd", "CD"]
            for _ in range(pdm_type - 1):
                op_str = ["c%sd" % x for x in op_str] + ["C%sD" % op_str[-1]]
            perms = bw.b.VectorSpinPermScheme(
                [
                    bw.b.SpinPermScheme.initialize_sz(pdm_type * 2, cd, True)
                    for cd in op_str
                ]
            )
        elif SymmetryTypes.SGF in bw.symm_type:
            op_str = "C" * pdm_type + "D" * pdm_type
            perm = bw.b.SpinPermScheme.initialize_sz(pdm_type * 2, op_str, True)
            perms = bw.b.VectorSpinPermScheme([perm])

        if iprint >= 1:
            print("npdm string =", op_str)

        if iprint >= 3:
            for perm in perms:
                print(perm)

        if simulated_parallel != 0 and self.mpi is not None:
            raise RuntimeError("Cannot simulate parallel in parallel mode!")

        sp_size = 1 if simulated_parallel == 0 else simulated_parallel
        sp_file_names = []

        for sp_rank in range(sp_size):

            if iprint >= 1 and simulated_parallel != 0:
                print("simulated parallel rank =", sp_rank)

            mket = ket.deep_copy("PDM-KET@TMP")
            mpss = [mket]
            if bra is not None and bra != ket:
                mbra = bra.deep_copy("PDM-BRA@TMP")
                mpss.append(mbra)
            else:
                mbra = mket

            for mps in mpss:
                if mps.dot == 2 and site_type != 2:
                    mps.dot = 1
                    if mps.center == mps.n_sites - 2:
                        mps.center = mps.n_sites - 1
                        mps.canonical_form = mps.canonical_form[:-1] + "S"
                    elif mps.center == 0:
                        mps.canonical_form = "K" + mps.canonical_form[1:]
                    else:
                        assert False
                    mps.save_data()

                if self.mpi is not None:
                    self.mpi.barrier()

                mps.load_mutable()
                mps.info.bond_dim = max(
                    mps.info.bond_dim, mps.info.get_max_bond_dimension()
                )

            self.align_mps_center(mbra, mket)

            scheme = bw.b.NPDMScheme(perms)
            pmpo = bw.bs.GeneralNPDMMPO(
                self.ghamil, scheme, NPDMAlgorithmTypes.SymbolFree in algo_type
            )
            pmpo.iprint = 2 if iprint >= 4 else min(iprint, 1)
            if self.mpi:
                pmpo.parallel_rule = self.prule
            if simulated_parallel != 0:
                sp_rule = bw.bs.ParallelRuleSimple(
                    bw.b.ParallelSimpleTypes.Nothing,
                    bw.bs.ParallelCommunicator(sp_size, sp_rank, 0),
                )
                assert sp_rule.is_root()
                pmpo.parallel_rule = sp_rule
            pmpo.build()

            pmpo = bw.bs.SimplifiedMPO(pmpo, bw.bs.Rule(), False, False)
            if self.mpi:
                pmpo = bw.bs.ParallelMPO(pmpo, self.prule)
            if simulated_parallel != 0:
                pmpo = bw.bs.ParallelMPO(pmpo, sp_rule)

            pme = bw.bs.MovingEnvironment(pmpo, mbra, mket, "NPDM")
            if fused_contraction_rotation:
                pme.cached_contraction = False
                pme.fused_contraction_rotation = True
            else:
                pme.cached_contraction = True
                pme.fused_contraction_rotation = False
            pme.init_environments(iprint >= 2)
            expect = bw.bs.Expect(pme, mbra.info.bond_dim, mket.info.bond_dim)
            expect.cutoff = cutoff
            if site_type == 0:
                expect.zero_dot_algo = True
            if NPDMAlgorithmTypes.SymbolFree in algo_type:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.SymbolFree
                if NPDMAlgorithmTypes.LowMem in algo_type:
                    expect.algo_type = (
                        expect.algo_type | bw.b.ExpectationAlgorithmTypes.LowMem
                    )
                if NPDMAlgorithmTypes.Compressed in algo_type:
                    expect.algo_type = (
                        expect.algo_type | bw.b.ExpectationAlgorithmTypes.Compressed
                    )
            elif NPDMAlgorithmTypes.Normal in algo_type:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.Normal
            elif NPDMAlgorithmTypes.Fast in algo_type:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.Fast
            else:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.Automatic

            expect.iprint = iprint
            expect.solve(True, mket.center == 0)

            if simulated_parallel != 0:
                if (
                    NPDMAlgorithmTypes.Compressed in algo_type
                    or expect.algo_type == bw.b.ExpectationAlgorithmTypes.Automatic
                ):
                    sp_file_names.append(
                        pme.get_npdm_fragment_filename(-1)[:-2] + "%d.fpc"
                    )
                else:
                    sp_file_names.append(
                        pme.get_npdm_fragment_filename(-1)[:-2] + "%d.npy"
                    )

            if self.clean_scratch:
                expect.me.remove_partition_files()

        if simulated_parallel != 0:

            if iprint >= 1 and simulated_parallel != 0:
                print("simulated parallel accumulate files...")

            scheme = bw.b.NPDMScheme(perms)
            pmpo = bw.bs.GeneralNPDMMPO(
                self.ghamil, scheme, NPDMAlgorithmTypes.SymbolFree in algo_type
            )
            # recover the default serial prefix
            sp_rule = bw.bs.ParallelRuleSimple(
                bw.b.ParallelSimpleTypes.Nothing, bw.bs.ParallelCommunicator(1, 0, 0)
            )
            pmpo.iprint = 2 if iprint >= 4 else min(iprint, 1)
            pmpo.build()
            pme = bw.bs.MovingEnvironment(pmpo, mbra, mket, "NPDM-SUM")

            fp_codec = bw.b.DoubleFPCodec()
            for i in range(self.n_sites - 1):
                data = 0
                if sp_file_names[0][-4:] == ".fpc":
                    for j in range(sp_size):
                        data = data + fp_codec.load(sp_file_names[j] % i)
                    fp_codec.save(pme.get_npdm_fragment_filename(i) + ".fpc", data)
                else:
                    for j in range(sp_size):
                        data = data + np.load(sp_file_names[j] % i)
                    np.save(pme.get_npdm_fragment_filename(i) + ".npy", data)

            expect = bw.bs.Expect(pme, mbra.info.bond_dim, mket.info.bond_dim)
            if site_type == 0:
                expect.zero_dot_algo = True
            if NPDMAlgorithmTypes.SymbolFree in algo_type:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.SymbolFree
                if NPDMAlgorithmTypes.LowMem in algo_type:
                    expect.algo_type = (
                        expect.algo_type | bw.b.ExpectationAlgorithmTypes.LowMem
                    )
                if NPDMAlgorithmTypes.Compressed in algo_type:
                    expect.algo_type = (
                        expect.algo_type | bw.b.ExpectationAlgorithmTypes.Compressed
                    )
            elif NPDMAlgorithmTypes.Normal in algo_type:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.Normal
            elif NPDMAlgorithmTypes.Fast in algo_type:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.Fast
            else:
                expect.algo_type = bw.b.ExpectationAlgorithmTypes.Automatic

            expect.iprint = iprint

        npdms = list(expect.get_npdm())

        if SymmetryTypes.SU2 in bw.symm_type:
            for ip in range(len(npdms)):
                npdms[ip] *= np.array(np.sqrt(2.0)) ** (scheme.n_max_ops // 2)
        else:
            for ip in range(len(npdms)):
                npdms[ip] = np.array(npdms[ip], copy=False)

        if self.reorder_idx is not None:
            rev_idx = np.argsort(self.reorder_idx)
            for ip in range(len(npdms)):
                for i in range(scheme.n_max_ops):
                    npdms[ip] = npdms[ip][(slice(None),) * i + (rev_idx,)]

        if self.mpi is not None:
            self.mpi.barrier()

        if SymmetryTypes.SU2 in bw.symm_type or SymmetryTypes.SGF in bw.symm_type:
            assert len(npdms) == 1
            npdms = npdms[0]

        return npdms

    def get_1pdm(self, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=1, *args, **kwargs)

    def get_2pdm(self, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=2, *args, **kwargs)

    def get_3pdm(self, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=3, *args, **kwargs)

    def get_4pdm(self, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=4, *args, **kwargs)

    def get_trans_1pdm(self, bra, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=1, bra=bra, *args, **kwargs)

    def get_trans_2pdm(self, bra, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=2, bra=bra, *args, **kwargs)

    def get_trans_3pdm(self, bra, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=3, bra=bra, *args, **kwargs)

    def get_trans_4pdm(self, bra, ket, *args, **kwargs):
        return self.get_npdm(ket, pdm_type=4, bra=bra, *args, **kwargs)

    def align_mps_center(self, ket, ref):
        if self.mpi is not None:
            self.mpi.barrier()
        ket.info.bond_dim = max(ket.info.bond_dim, ket.info.get_max_bond_dimension())
        if ket.center != ref.center:
            if ref.center == 0:
                if ket.dot == 2:
                    ket.center += 1
                    ket.canonical_form = ket.canonical_form[:-1] + "S"
                while ket.center != 0:
                    ket.move_left(self.ghamil.opf.cg, self.prule)
            else:
                ket.canonical_form = "K" + ket.canonical_form[1:]
                while ket.center != ket.n_sites - 1:
                    ket.move_right(self.ghamil.opf.cg, self.prule)
                if ket.dot == 2:
                    ket.center -= 1
            if self.mpi is not None:
                self.mpi.barrier()
            ket.save_data()
            ket.info.save_data(self.scratch + "/%s-mps_info.bin" % ket.info.tag)
        if self.mpi is not None:
            self.mpi.barrier()

    def adjust_mps(self, ket, dot=None):
        if dot is None:
            dot = ket.dot
        bw = self.bw
        if ket.center == 0 and dot == 2:
            if self.mpi is not None:
                self.mpi.barrier()
            if ket.canonical_form[ket.center] in "ST":
                ket.flip_fused_form(ket.center, self.ghamil.opf.cg, self.prule)
            ket.save_data()
            if self.mpi is not None:
                self.mpi.barrier()
            ket.load_mutable()
            ket.info.load_mutable()
            if self.mpi is not None:
                self.mpi.barrier()

        ket.dot = dot
        forward = ket.center == 0
        if (
            ket.canonical_form[ket.center] == "L"
            and ket.center != ket.n_sites - ket.dot
        ):
            ket.center += 1
            forward = True
        elif (
            ket.canonical_form[ket.center] == "C"
            or ket.canonical_form[ket.center] == "M"
        ) and ket.center != 0:
            ket.center -= 1
            forward = False
        if ket.canonical_form[ket.center] == "M" and not isinstance(
            ket, bw.bs.MultiMPS
        ):
            ket.canonical_form = (
                ket.canonical_form[: ket.center]
                + "C"
                + ket.canonical_form[ket.center + 1 :]
            )
        if ket.canonical_form[-1] == "M" and not isinstance(ket, bw.bs.MultiMPS):
            ket.canonical_form = ket.canonical_form[:-1] + "C"
        if dot == 1:
            if ket.canonical_form[0] == "C" and ket.canonical_form[1] == "R":
                ket.canonical_form = "K" + ket.canonical_form[1:]
            elif ket.canonical_form[-1] == "C" and ket.canonical_form[-2] == "L":
                ket.canonical_form = ket.canonical_form[:-1] + "S"
                ket.center = ket.n_sites - 1
            elif ket.center == ket.n_sites - 2 and ket.canonical_form[-2] == "L":
                ket.center = ket.n_sites - 1
            if ket.canonical_form[0] == "M" and ket.canonical_form[1] == "R":
                ket.canonical_form = "J" + ket.canonical_form[1:]
            elif ket.canonical_form[-1] == "M" and ket.canonical_form[-2] == "L":
                ket.canonical_form = ket.canonical_form[:-1] + "T"
                ket.center = ket.n_sites - 1
            elif ket.center == ket.n_sites - 2 and ket.canonical_form[-2] == "L":
                ket.center = ket.n_sites - 1

        ket.save_data()
        if self.mpi is not None:
            self.mpi.barrier()
        return ket, forward

    def split_mps(self, ket, iroot, tag):
        bw = self.bw
        if self.mpi is not None:
            self.mpi.barrier()
        assert isinstance(ket, bw.bs.MultiMPS)
        if len(ket.info.targets) == 1:
            iket = ket.extract(iroot, tag + "@TMP")
        else:
            iket = ket.extract(iroot, tag)
        if self.mpi is not None:
            self.mpi.barrier()
        if len(iket.info.targets) == 1:
            iket = iket.make_single(tag)
        if self.mpi is not None:
            self.mpi.barrier()
        iket = self.adjust_mps(iket)[0]
        iket.info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        return iket

    def multiply(
        self,
        bra,
        mpo,
        ket,
        n_sweeps=10,
        tol=1e-8,
        bond_dims=None,
        bra_bond_dims=None,
        cutoff=1e-24,
        iprint=0,
    ):
        bw = self.bw
        if bra.info.tag == ket.info.tag:
            raise RuntimeError("Same tag for bra and ket!!")
        if bond_dims is None:
            bond_dims = [ket.info.bond_dim]
        if bra_bond_dims is None:
            bra_bond_dims = [bra.info.bond_dim]
        self.align_mps_center(bra, ket)
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "MULT")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments(iprint >= 2)
        cps = bw.bs.Linear(
            me, bw.b.VectorUBond(bra_bond_dims), bw.b.VectorUBond(bond_dims)
        )
        cps.iprint = iprint
        cps.cutoff = cutoff
        norm = cps.solve(n_sweeps, ket.center == 0, tol)

        if self.clean_scratch:
            me.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return norm

    def expectation(self, bra, mpo, ket, iprint=0):
        bw = self.bw
        mbra = bra.deep_copy("EXPE-BRA@TMP")
        if bra != ket:
            mket = ket.deep_copy("EXPE-KET@TMP")
        else:
            mket = mbra
        bond_dim = max(mbra.info.bond_dim, mket.info.bond_dim)
        self.align_mps_center(mbra, mket)
        me = bw.bs.MovingEnvironment(mpo, mbra, mket, "EXPT")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments(iprint >= 2)
        expect = bw.bs.Expect(me, bond_dim, bond_dim)
        expect.iprint = iprint
        ex = expect.solve(False, mket.center != 0)

        if self.clean_scratch:
            me.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return ex

    def fix_restarting_mps(self, mps):
        cg = self.ghamil.opf.cg
        if (
            mps.canonical_form[mps.center] == "L"
            and mps.center != mps.n_sites - mps.dot
        ):
            mps.center += 1
            if mps.canonical_form[mps.center] in "ST" and mps.dot == 2:
                if self.mpi is not None:
                    self.mpi.barrier()
                mps.flip_fused_form(mps.center, cg, self.prule)
                mps.save_data()
                if self.mpi is not None:
                    self.mpi.barrier()
                mps.load_mutable()
                mps.info.load_mutable()
                if self.mpi is not None:
                    self.mpi.barrier()
        elif mps.canonical_form[mps.center] in "CMKJST" and mps.center != 0:
            if mps.canonical_form[mps.center] in "KJ" and mps.dot == 2:
                if self.mpi is not None:
                    self.mpi.barrier()
                mps.flip_fused_form(mps.center, cg, self.prule)
                mps.save_data()
                if self.mpi is not None:
                    self.mpi.barrier()
                mps.load_mutable()
                mps.info.load_mutable()
                if self.mpi is not None:
                    self.mpi.barrier()
            if (
                not mps.canonical_form[mps.center : mps.center + 2] == "CC"
                and mps.dot == 2
            ):
                mps.center -= 1
        elif mps.center == mps.n_sites - 1 and mps.dot == 2:
            if self.mpi is not None:
                self.mpi.barrier()
            if mps.canonical_form[mps.center] in "KJ":
                mps.flip_fused_form(mps.center, cg, self.prule)
            mps.center = mps.n_sites - 2
            mps.save_data()
            if self.mpi is not None:
                self.mpi.barrier()
            mps.load_mutable()
            mps.info.load_mutable()
            if self.mpi is not None:
                self.mpi.barrier()
        elif mps.center == 0 and mps.dot == 2:
            if self.mpi is not None:
                self.mpi.barrier()
            if mps.canonical_form[mps.center] in "ST":
                mps.flip_fused_form(mps.center, cg, self.prule)
            mps.save_data()
            if self.mpi is not None:
                self.mpi.barrier()
            mps.load_mutable()
            mps.info.load_mutable()
            if self.mpi is not None:
                self.mpi.barrier()

    def load_mps(self, tag, nroots=1):
        import os

        bw = self.bw
        mps_info = bw.brs.MPSInfo(0) if nroots == 1 else bw.brs.MultiMPSInfo(0)
        if os.path.isfile(self.scratch + "/%s-mps_info.bin" % tag):
            mps_info.load_data(self.scratch + "/%s-mps_info.bin" % tag)
        else:
            mps_info.load_data(self.scratch + "/mps_info.bin")
        assert mps_info.tag == tag
        mps_info.load_mutable()
        mps_info.bond_dim = max(mps_info.bond_dim, mps_info.get_max_bond_dimension())
        mps = bw.bs.MPS(mps_info) if nroots == 1 else bw.bs.MultiMPS(mps_info)
        mps.load_data()
        mps.load_mutable()
        self.fix_restarting_mps(mps)
        return mps

    def mps_change_singlet_embedding(self, mps, tag, forward, left_vacuum=None):
        cp_mps = mps.deep_copy(tag)
        while cp_mps.center > 0:
            cp_mps.move_left(self.ghamil.opf.cg, self.prule)
        if forward:
            if left_vacuum is None:
                left_vacuum = self.bw.SX.invalid
            cp_mps.to_singlet_embedding_wfn(self.ghamil.opf.cg, left_vacuum, self.prule)
        else:
            cp_mps.from_singlet_embedding_wfn(self.ghamil.opf.cg, self.prule)
        cp_mps.save_data()
        cp_mps.info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        if self.mpi is not None:
            self.mpi.barrier()
        return cp_mps

    def mps_change_to_singlet_embedding(self, mps, tag, left_vacuum=None):
        return self.mps_change_singlet_embedding(
            mps, tag, forward=True, left_vacuum=left_vacuum
        )

    def mps_change_from_singlet_embedding(self, mps, tag, left_vacuum=None):
        return self.mps_change_singlet_embedding(
            mps, tag, forward=False, left_vacuum=left_vacuum
        )

    def mps_change_precision(self, mps, tag):
        bw = self.bw
        assert tag != mps.info.tag
        if SymmetryTypes.SP in bw.symm_type:
            r = bw.bs.trans_mps_to_double(mps, tag)
        else:
            r = bw.bs.trans_mps_to_float(mps, tag)
        r.info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        return r

    def get_random_mps(
        self,
        tag,
        bond_dim=500,
        center=0,
        dot=2,
        target=None,
        nroots=1,
        occs=None,
        full_fci=True,
        left_vacuum=None,
        casci_ncore=0,
        casci_nvirt=0,
    ):
        bw = self.bw
        if target is None:
            target = self.target
        if left_vacuum is None:
            left_vacuum = self.left_vacuum
        if nroots == 1:
            if casci_ncore == 0 and casci_nvirt == 0:
                mps_info = bw.brs.MPSInfo(
                    self.n_sites, self.vacuum, target, self.ghamil.basis
                )
            else:
                casci_ncas = self.n_sites - casci_ncore - casci_nvirt
                mps_info = bw.brs.CASCIMPSInfo(
                    self.n_sites,
                    self.vacuum,
                    target,
                    self.ghamil.basis,
                    casci_ncore,
                    casci_ncas,
                    casci_nvirt,
                )
            mps = bw.bs.MPS(self.n_sites, center, dot)
        else:
            targets = bw.VectorSX([target]) if isinstance(target, bw.SX) else target
            mps_info = bw.brs.MultiMPSInfo(
                self.n_sites, self.vacuum, targets, self.ghamil.basis
            )
            mps = bw.bs.MultiMPS(self.n_sites, center, dot, nroots)
        mps_info.tag = tag
        if full_fci:
            mps_info.set_bond_dimension_full_fci(left_vacuum, self.vacuum)
        else:
            mps_info.set_bond_dimension_fci(left_vacuum, self.vacuum)
        if occs is not None:
            mps_info.set_bond_dimension_using_occ(bond_dim, bw.b.VectorDouble(occs))
        else:
            mps_info.set_bond_dimension(bond_dim)
        mps_info.bond_dim = bond_dim
        mps.initialize(mps_info)
        mps.random_canonicalize()
        if nroots == 1:
            mps.tensors[mps.center].normalize()
        else:
            for xwfn in mps.wfns:
                xwfn.normalize()
        mps.save_mutable()
        mps_info.save_mutable()
        mps.save_data()
        mps_info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        return mps

    def expr_builder(self):
        return ExprBuilder(self.bw)

    def finalize(self):
        bw = self.bw
        bw.b.Global.frame = None


class SOCDMRGDriver(DMRGDriver):
    def __init__(
        self,
        stack_mem=1 << 30,
        scratch="./nodex",
        restart_dir=None,
        n_threads=None,
        symm_type=SymmetryTypes.SU2,
        mpi=None,
    ):
        super().__init__(
            stack_mem=stack_mem,
            scratch=scratch,
            restart_dir=restart_dir,
            n_threads=n_threads,
            symm_type=symm_type,
            mpi=mpi,
        )

    def hybrid_mpo_dmrg(
        self, mpo, mpo_cpx, ket, n_sweeps=10, iprint=0, tol=1e-8, **kwargs
    ):
        bw = self.bw
        assert ket.nroots % 2 == 0
        super().dmrg(mpo, ket, n_sweeps=-1, iprint=iprint, tol=tol, **kwargs)
        self._dmrg.me.cached_contraction = False
        self._dmrg.me.delayed_contraction = bw.b.OpNamesSet()
        self._dmrg.me.init_environments(iprint >= 2)
        cpx_me = bw.bcs.MovingEnvironmentX(mpo_cpx, ket, ket, "DMRG-CPX")
        cpx_me.cached_contraction = False
        cpx_me.init_environments(iprint >= 2)
        self._dmrg.cpx_me = cpx_me
        ener = self._dmrg.solve(n_sweeps, ket.center == 0, tol)

        if self.clean_scratch:
            self._dmrg.me.remove_partition_files()
            self._dmrg.cpx_me.remove_partition_files()

        ket.info.bond_dim = max(ket.info.bond_dim, self._dmrg.bond_dims[-1])
        if isinstance(ket, bw.bs.MultiMPS):
            ener = list(self._dmrg.energies[-1])
        if self.mpi is not None:
            self.mpi.barrier()
        return ener

    def soc_two_step(self, energies, twoss, pdms_dict, hsomo, iprint=1):
        au2cm = 219474.631115585274529
        import numpy as np

        assert len(twoss) == len(energies)

        xnroots = [len(x) for x in energies]
        eners = []
        xtwos = []
        for ix in range(len(energies)):
            eners += energies[ix]
            xtwos += [twoss[ix]] * xnroots[ix]
        eners = np.array(eners)

        pdm0 = pdms_dict[(0, 0)]
        assert pdm0.ndim == 2 and pdm0.shape[0] == pdm0.shape[1]
        ncas = pdm0.shape[0]

        pdms = np.zeros((len(eners), len(eners), ncas, ncas))
        for ist in range(len(eners)):
            for jst in range(len(eners)):
                if ist >= jst and abs(xtwos[ist] - xtwos[jst]) <= 2:
                    pdms[ist, jst] = pdms_dict[(ist, jst)]

        if iprint and (self.mpi is None or self.mpi.rank == self.mpi.root):
            print("HSO.SHAPE = ", hsomo.shape)
            print("PDMS.SHAPE = ", pdms.shape)
        thrds = 29.0  # cm-1
        n_mstates = 0
        for ix, iis in enumerate(twoss):
            n_mstates += (iis + 1) * xnroots[ix]
        hsiso = np.zeros((n_mstates, n_mstates), dtype=complex)
        hdiag = np.zeros((n_mstates,), dtype=complex)

        # separate T^1 to T^1_(-1,0,1)
        def spin_proj(cg, pdm, tjo, tjb, tjk):
            nmo = pdm.shape[0]
            ppdm = np.zeros((tjb + 1, tjk + 1, tjo + 1, nmo, nmo))
            for ibra in range(tjb + 1):
                for iket in range(tjk + 1):
                    for iop in range(tjo + 1):
                        tmb = -tjb + 2 * ibra
                        tmk = -tjk + 2 * iket
                        tmo = -tjo + 2 * iop
                        factor = (-1) ** ((tjb - tmb) // 2) * cg.wigner_3j(
                            tjb, tjo, tjk, -tmb, tmo, tmk
                        )
                        if factor != 0:
                            ppdm[ibra, iket, iop] = pdm * factor
            return ppdm

        # from T^1_(-1,0,1) to Tx, Ty, Tz
        def xyz_proj(ppdm):
            xpdm = np.zeros(ppdm.shape, dtype=complex)
            xpdm[:, :, 0] = (0.5 + 0j) * (ppdm[:, :, 0] - ppdm[:, :, 2])
            xpdm[:, :, 1] = (0.5j + 0) * (ppdm[:, :, 0] + ppdm[:, :, 2])
            xpdm[:, :, 2] = (np.sqrt(0.5) + 0j) * ppdm[:, :, 1]
            return xpdm

        cg = self.ghamil.opf.cg

        qls = []
        imb = 0
        for ibra in range(len(pdms)):
            imk = 0
            tjb = xtwos[ibra]
            for iket in range(len(pdms)):
                tjk = xtwos[iket]
                if ibra >= iket:
                    pdm = pdms[ibra, iket]
                    xpdm = xyz_proj(spin_proj(cg, pdm, 2, tjb, tjk))
                    for ibm in range(xpdm.shape[0]):
                        for ikm in range(xpdm.shape[1]):
                            somat = np.einsum("rij,rij->", xpdm[ibm, ikm], hsomo)
                            hsiso[ibm + imb, ikm + imk] = somat
                            somat *= au2cm
                            if iprint and (
                                self.mpi is None or self.mpi.rank == self.mpi.root
                            ):
                                if abs(somat) > thrds:
                                    print(
                                        (
                                            "I1 = %4d (E1 = %15.8f) S1 = %4.1f MS1 = %4.1f "
                                            + "I2 = %4d (E2 = %15.8f) S2 = %4.1f MS2 = %4.1f Re = %9.3f Im = %9.3f"
                                        )
                                        % (
                                            ibra,
                                            eners[ibra],
                                            tjb / 2,
                                            -tjb / 2 + ibm,
                                            iket,
                                            eners[iket],
                                            tjk / 2,
                                            -tjk / 2 + ikm,
                                            somat.real,
                                            somat.imag,
                                        )
                                    )
                imk += tjk + 1
            for ibm in range(tjb + 1):
                qls.append((ibra, eners[ibra], tjb / 2, -tjb / 2 + ibm))
            hdiag[imb : imb + tjb + 1] = eners[ibra]
            imb += tjb + 1

        for i in range(len(hsiso)):
            for j in range(len(hsiso)):
                if i >= j:
                    hsiso[j, i] = hsiso[i, j].conj()

        self._hsiso = hsiso * au2cm

        symm_err = np.linalg.norm(np.abs(hsiso - hsiso.T.conj()))
        if iprint and (self.mpi is None or self.mpi.rank == self.mpi.root):
            print("SYMM Error (should be small) = ", symm_err)
        assert symm_err < 1e-10
        hfull = hsiso + np.diag(hdiag)
        heig, hvec = np.linalg.eigh(hfull)
        if iprint and (self.mpi is None or self.mpi.rank == self.mpi.root):
            print("Total energies including SO-coupling:\n")
        xhdiag = np.zeros_like(heig)

        for i in range(len(heig)):
            shvec = np.zeros(len(eners))
            ssq = 0
            imb = 0
            for ibra in range(len(eners)):
                tjb = xtwos[ibra]
                shvec[ibra] = np.linalg.norm(hvec[imb : imb + tjb + 1, i]) ** 2
                ssq += shvec[ibra] * (tjb + 2) * tjb / 4
                imb += tjb + 1
            assert abs(np.sum(shvec) - 1) < 1e-7
            iv = np.argmax(np.abs(shvec))
            xhdiag[i] = eners[iv]
            if iprint and (self.mpi is None or self.mpi.rank == self.mpi.root):
                print(
                    " State %4d Total energy: %15.8f <S^2>: %12.6f | largest |coeff|**2 %10.6f from I = %4d E = %15.8f S = %4.1f"
                    % (i, heig[i], ssq, shvec[iv], iv, eners[iv], xtwos[iv] / 2)
                )

        return heig


class NormalOrder:
    @staticmethod
    def def_ix(cidx):
        import numpy as np

        def ix(x):
            p = {"I": cidx, "E": ~cidx}
            if len(x) == 2:
                return np.outer(p[x[0]], p[x[1]])
            else:
                return np.outer(np.outer(np.outer(p[x[0]], p[x[1]]), p[x[2]]), p[x[3]])

        return ix

    @staticmethod
    def def_gctr(cidx, h1e, g2e):
        import numpy as np

        def gctr(x):
            v = g2e if len(x.split("->")[0]) == 4 else h1e
            for ig, g in enumerate(x.split("->")[0]):
                if x.split("->")[0].count(g) == 2:
                    v = v[(slice(None),) * ig + (cidx,)]
            return np.einsum(x, v, optimize=True)

        return gctr

    @staticmethod
    def def_gctr_sz(cidx, h1e, g2e):
        import numpy as np

        def gctr(i, x):
            v = g2e[i] if len(x.split("->")[0]) == 4 else h1e[i]
            for ig, g in enumerate(x.split("->")[0]):
                if x.split("->")[0].count(g) == 2:
                    v = v[(slice(None),) * ig + (cidx,)]
            return np.einsum(x, v, optimize=True)

        return gctr

    @staticmethod
    def make_su2(h1e, g2e, const_e, cidx, use_wick):
        import numpy as np

        if use_wick:
            return WickNormalOrder.make_su2(h1e, g2e, const_e, cidx)

        ix = NormalOrder.def_ix(cidx)
        gctr = NormalOrder.def_gctr(cidx, h1e, g2e)

        h1es = {k: np.zeros_like(h1e) for k in ("CD", "DC")}
        g2es = {k: np.zeros_like(g2e) for k in ("CCDD", "CDCD", "DDCC", "DCDC")}
        const_es = const_e

        const_es += 2.0 * gctr("qq->")
        const_es += 2.0 * gctr("rrss->")
        const_es -= 1.0 * gctr("srrs->")

        np.putmask(h1es["CD"], ix("EI"), h1es["CD"] + gctr("pq->pq"))
        np.putmask(h1es["CD"], ix("EE"), h1es["CD"] + gctr("pq->pq"))
        np.putmask(h1es["DC"], ix("II"), h1es["DC"] + gctr("pq->qp"))
        np.putmask(h1es["CD"], ix("IE"), h1es["CD"] + gctr("pq->pq"))

        np.putmask(h1es["CD"], ix("EI"), h1es["CD"] - 1.0 * gctr("prrs->ps"))
        np.putmask(h1es["CD"], ix("EI"), h1es["CD"] + 2.0 * gctr("prss->pr"))
        np.putmask(h1es["CD"], ix("EE"), h1es["CD"] + 2.0 * gctr("prss->pr"))
        np.putmask(h1es["DC"], ix("II"), h1es["DC"] - 1.0 * gctr("prrs->sp"))
        np.putmask(h1es["DC"], ix("II"), h1es["DC"] + 2.0 * gctr("prss->rp"))
        np.putmask(h1es["CD"], ix("IE"), h1es["CD"] + 2.0 * gctr("prss->pr"))
        np.putmask(h1es["CD"], ix("IE"), h1es["CD"] - 1.0 * gctr("srqs->qr"))

        np.putmask(g2es["DCDC"], ix("IEII"), g2es["DCDC"] + 1.0 * gctr("prqs->sprq"))
        np.putmask(g2es["CCDD"], ix("EEII"), g2es["CCDD"] + 0.5 * gctr("prqs->pqsr"))
        np.putmask(g2es["CCDD"], ix("EEEI"), g2es["CCDD"] + 0.5 * gctr("prqs->pqsr"))
        np.putmask(g2es["DCDC"], ix("IEEI"), g2es["DCDC"] + 1.0 * gctr("prqs->sprq"))
        np.putmask(g2es["CCDD"], ix("EIEE"), g2es["CCDD"] + 1.0 * gctr("prqs->pqsr"))
        np.putmask(g2es["CCDD"], ix("EEIE"), g2es["CCDD"] + 0.5 * gctr("prqs->pqsr"))
        np.putmask(g2es["CCDD"], ix("EEEE"), g2es["CCDD"] + 0.5 * gctr("prqs->pqsr"))
        np.putmask(g2es["DDCC"], ix("IIII"), g2es["DDCC"] + 0.5 * gctr("prqs->rsqp"))
        np.putmask(g2es["DCDC"], ix("IIEI"), g2es["DCDC"] + 1.0 * gctr("prqs->sprq"))
        np.putmask(g2es["CCDD"], ix("IIEE"), g2es["CCDD"] + 0.5 * gctr("prqs->pqsr"))
        np.putmask(g2es["CCDD"], ix("EIEI"), g2es["CCDD"] + 1.0 * gctr("prqs->qprs"))

        h1es = {"(%s+%s)0" % tuple(k): np.sqrt(2) * v for k, v in h1es.items()}
        g2es = {"((%s+(%s+%s)0)1+%s)0" % tuple(k): 2 * v for k, v in g2es.items()}

        return h1es, g2es, const_es

    @staticmethod
    def make_sz(h1e, g2e, const_e, cidx, use_wick):
        import numpy as np

        if use_wick:
            return WickNormalOrder.make_sz(h1e, g2e, const_e, cidx)

        g2e = [g2e[0], g2e[1], g2e[1].transpose(2, 3, 0, 1), g2e[2]]
        ix = NormalOrder.def_ix(cidx)
        gctr = NormalOrder.def_gctr_sz(cidx, h1e, g2e)

        h1k = [("cd", "dc"), ("CD", "DC")]
        g2k = [
            "cddc:cdcd:ccdd:cdcd:ccdd:ddcc:dccd:cddc:cdcd:cdcd:dccd:cdcd:ccdd",
            "cdDC:cdCD:cCdD:cDCd:cCDd:dDcC:dcCD:CdDc:CdcD:CdcD:DcCd:CDcd:CcdD",
            "CDdc:CDcd:CcDd:CdcD:CcdD:DdCc:DCcd:cDdC:cDCd:cDCd:dCcD:cdCD:cCDd",
            "CDDC:CDCD:CCDD:CDCD:CCDD:DDCC:DCCD:CDDC:CDCD:CDCD:DCCD:CDCD:CCDD",
        ]
        g2k = [tuple(x.split(":")) for x in g2k]

        h1es = {k: np.zeros_like(h1e[i]) for i in range(2) for k in set(h1k[i])}
        g2es = {k: np.zeros_like(g2e[i]) for i in range(4) for k in set(g2k[i])}
        const_es = const_e

        const_es += 1.0 * (gctr(0, "qq->") + gctr(1, "qq->"))
        const_es += 0.5 * sum(gctr(i, "qqss->") for i in range(4))
        const_es -= 0.5 * (gctr(0, "sqqs->") + gctr(3, "sqqs->"))

        for i in range(2):
            cd, dc = h1k[i]
            np.putmask(h1es[cd], ix("EI"), h1es[cd] + gctr(i, "pq->pq"))
            np.putmask(h1es[cd], ix("EE"), h1es[cd] + gctr(i, "pq->pq"))
            np.putmask(h1es[dc], ix("II"), h1es[dc] - gctr(i, "pq->qp"))
            np.putmask(h1es[cd], ix("IE"), h1es[cd] + gctr(i, "pq->pq"))

        for i in range(2):
            cd, dc = h1k[i]
            np.putmask(h1es[cd], ix("EI"), h1es[cd] - 1.0 * gctr(i * 3, "pqqs->ps"))
            np.putmask(h1es[cd], ix("EI"), h1es[cd] + 1.0 * gctr(i * 3, "pqss->pq"))
            np.putmask(h1es[cd], ix("EE"), h1es[cd] + 1.0 * gctr(i * 3, "pqss->pq"))
            np.putmask(h1es[dc], ix("II"), h1es[dc] + 1.0 * gctr(i * 3, "pqqs->sp"))
            np.putmask(h1es[dc], ix("II"), h1es[dc] - 1.0 * gctr(i * 3, "pqss->qp"))
            np.putmask(h1es[cd], ix("IE"), h1es[cd] + 1.0 * gctr(i * 3, "pqss->pq"))
            np.putmask(h1es[cd], ix("IE"), h1es[cd] - 1.0 * gctr(i * 3, "sqrs->rq"))
            np.putmask(h1es[cd], ix("EE"), h1es[cd] - 1.0 * gctr(i * 3, "sqrs->rq"))
            np.putmask(h1es[cd], ix("EI"), h1es[cd] + 0.5 * gctr(i + 1, "pqss->pq"))
            np.putmask(h1es[cd], ix("EE"), h1es[cd] + 0.5 * gctr(i + 1, "pqss->pq"))
            np.putmask(h1es[dc], ix("II"), h1es[dc] - 0.5 * gctr(i + 1, "pqss->qp"))
            np.putmask(h1es[cd], ix("IE"), h1es[cd] + 0.5 * gctr(i + 1, "pqss->pq"))
            np.putmask(h1es[dc], ix("II"), h1es[dc] - 0.5 * gctr(2 - i, "rrqs->sq"))
            np.putmask(h1es[cd], ix("IE"), h1es[cd] + 0.5 * gctr(2 - i, "rrqs->qs"))
            np.putmask(h1es[cd], ix("EI"), h1es[cd] + 0.5 * gctr(2 - i, "rrqs->qs"))
            np.putmask(h1es[cd], ix("EE"), h1es[cd] + 0.5 * gctr(2 - i, "rrqs->qs"))

        for i in range(4):
            cdDC, cdCD, cCdD, cDCd, cCDd, dDcC, dcCD = g2k[i][:7]
            CdDc, CdcD, CdcD, DcCd, CDcd, CcdD = g2k[i][7:]
            np.putmask(g2es[cdDC], ix("EIII"), g2es[cdDC] - 0.5 * gctr(i, "prqs->prsq"))
            np.putmask(g2es[cCdD], ix("EEII"), g2es[cCdD] - 0.5 * gctr(i, "prqs->pqrs"))
            np.putmask(g2es[cCdD], ix("EEIE"), g2es[cCdD] - 0.5 * gctr(i, "prqs->pqrs"))
            np.putmask(g2es[cCdD], ix("EIEE"), g2es[cCdD] - 0.5 * gctr(i, "prqs->pqrs"))
            np.putmask(g2es[cCDd], ix("EEIE"), g2es[cCDd] + 0.5 * gctr(i, "prqs->pqsr"))
            np.putmask(g2es[cCdD], ix("EEEE"), g2es[cCdD] - 0.5 * gctr(i, "prqs->pqrs"))
            np.putmask(g2es[dDcC], ix("IIII"), g2es[dDcC] - 0.5 * gctr(i, "prqs->rspq"))
            np.putmask(g2es[dcCD], ix("IIIE"), g2es[dcCD] - 0.5 * gctr(i, "prqs->rpqs"))
            np.putmask(g2es[CdDc], ix("EIII"), g2es[CdDc] + 0.5 * gctr(i, "prqs->qrsp"))
            np.putmask(g2es[DcCd], ix("IIIE"), g2es[DcCd] + 0.5 * gctr(i, "prqs->spqr"))
            np.putmask(g2es[cCdD], ix("IIEE"), g2es[cCdD] - 0.5 * gctr(i, "prqs->pqrs"))
            np.putmask(g2es[CcdD], ix("EIEE"), g2es[CcdD] + 0.5 * gctr(i, "prqs->qprs"))

            eiie = ix("EIIE")
            if i == 0 or i == 3:
                np.putmask(g2es[cDCd], eiie, g2es[cDCd] - 1.0 * gctr(i, "prqs->psqr"))
                np.putmask(g2es[CDcd], eiie, g2es[CDcd] + 1.0 * gctr(i, "prqs->qspr"))
            else:
                np.putmask(g2es[cDCd], eiie, g2es[cDCd] - 0.5 * gctr(i, "prqs->psqr"))
                np.putmask(g2es[CdcD], eiie, g2es[CdcD] - 0.5 * gctr(i, "prqs->qrps"))
                np.putmask(g2es[CDcd], eiie, g2es[CDcd] + 0.5 * gctr(i, "prqs->qspr"))
                np.putmask(g2es[cdCD], eiie, g2es[cdCD] + 0.5 * gctr(i, "prqs->prqs"))

        return h1es, g2es, const_es

    @staticmethod
    def make_sgf(h1e, g2e, const_e, cidx, use_wick):
        import numpy as np

        if use_wick:
            return WickNormalOrder.make_sgf(h1e, g2e, const_e, cidx)

        ix = NormalOrder.def_ix(cidx)
        gctr = NormalOrder.def_gctr(cidx, h1e, g2e)

        h1es = {k: np.zeros_like(h1e) for k in ("CD", "DC")}
        g2es = {k: np.zeros_like(g2e) for k in ("CDDC", "CCDD", "CDCD", "DDCC", "DCCD")}
        const_es = const_e

        const_es += 1.0 * gctr("qq->")
        const_es += 0.5 * gctr("qqss->")
        const_es -= 0.5 * gctr("sqqs->")

        np.putmask(h1es["CD"], ix("EI"), h1es["CD"] + gctr("pq->pq"))
        np.putmask(h1es["CD"], ix("EE"), h1es["CD"] + gctr("pq->pq"))
        np.putmask(h1es["DC"], ix("II"), h1es["DC"] - gctr("pq->qp"))
        np.putmask(h1es["CD"], ix("IE"), h1es["CD"] + gctr("pq->pq"))

        np.putmask(h1es["CD"], ix("EI"), h1es["CD"] - 1.0 * gctr("pqqs->ps"))
        np.putmask(h1es["CD"], ix("EI"), h1es["CD"] + 1.0 * gctr("pqss->pq"))
        np.putmask(h1es["CD"], ix("EE"), h1es["CD"] + 1.0 * gctr("pqss->pq"))
        np.putmask(h1es["DC"], ix("II"), h1es["DC"] + 1.0 * gctr("pqqs->sp"))
        np.putmask(h1es["DC"], ix("II"), h1es["DC"] - 1.0 * gctr("pqss->qp"))
        np.putmask(h1es["CD"], ix("IE"), h1es["CD"] + 1.0 * gctr("pqss->pq"))
        np.putmask(h1es["CD"], ix("IE"), h1es["CD"] - 1.0 * gctr("sqrs->rq"))
        np.putmask(h1es["CD"], ix("EE"), h1es["CD"] - 1.0 * gctr("sqrs->rq"))

        np.putmask(g2es["CDDC"], ix("EIII"), g2es["CDDC"] - 0.5 * gctr("pqrs->pqsr"))
        np.putmask(g2es["CCDD"], ix("EEII"), g2es["CCDD"] - 0.5 * gctr("pqrs->prqs"))
        np.putmask(g2es["CCDD"], ix("EEIE"), g2es["CCDD"] - 0.5 * gctr("pqrs->prqs"))
        np.putmask(g2es["CDCD"], ix("EIIE"), g2es["CDCD"] - 1.0 * gctr("pqrs->psrq"))
        np.putmask(g2es["CCDD"], ix("EIEE"), g2es["CCDD"] - 0.5 * gctr("pqrs->prqs"))
        np.putmask(g2es["CCDD"], ix("EEIE"), g2es["CCDD"] + 0.5 * gctr("pqrs->prsq"))
        np.putmask(g2es["CCDD"], ix("EEEE"), g2es["CCDD"] - 0.5 * gctr("pqrs->prqs"))
        np.putmask(g2es["DDCC"], ix("IIII"), g2es["DDCC"] - 0.5 * gctr("pqrs->qspr"))
        np.putmask(g2es["DCCD"], ix("IIIE"), g2es["DCCD"] - 0.5 * gctr("pqrs->qprs"))
        np.putmask(g2es["CDDC"], ix("EIII"), g2es["CDDC"] + 0.5 * gctr("pqrs->rqsp"))
        np.putmask(g2es["DCCD"], ix("IIIE"), g2es["DCCD"] + 0.5 * gctr("pqrs->sprq"))
        np.putmask(g2es["CCDD"], ix("IIEE"), g2es["CCDD"] - 0.5 * gctr("pqrs->prqs"))
        np.putmask(g2es["CDCD"], ix("EIIE"), g2es["CDCD"] + 1.0 * gctr("pqrs->rspq"))
        np.putmask(g2es["CCDD"], ix("EIEE"), g2es["CCDD"] + 0.5 * gctr("pqrs->rpqs"))

        return h1es, g2es, const_es


class WickNormalOrder:
    @staticmethod
    def make_su2(h1e, g2e, const_e, cidx, iprint=1):
        import block2 as b
        import numpy as np

        if iprint:
            print("-- Normal order (su2) using Wick's theorem")

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.Inactive] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.External] = b.WickIndex.parse_set("pqrsabcdefg")
        perm_map = b.MapPStrIntVectorWickPermutation()
        perm_map[("v", 4)] = b.WickPermutation.qc_chem()

        P = lambda x: b.WickExpr.parse(x, idx_map, perm_map)
        h = P("SUM <pq> h[pq] E1[p,q] + 0.5 SUM <pqrs> v[prqs] E2[pq,rs]")
        eq = h.expand(-1, False, False).simplify()
        WickSpinAdaptation.adjust_spin_coupling(eq)
        exprs = WickSpinAdaptation.get_eq_exprs(eq)

        dts = {}
        const_es = const_e
        tensor_d = {"h": h1e, "v": g2e}

        is_inactive = (
            lambda x: (x & b.WickIndexTypes.Inactive) != b.WickIndexTypes.Nothing
        )

        def ix(x):
            p = {"I": cidx, "E": ~cidx}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((len(cidx),) * len(x))

        def tx(x, ix):
            for ig, ii in enumerate(ix):
                idx = (slice(None),) * ig
                idx += (cidx if is_inactive(ii.types) else ~cidx,)
                x = x[idx]
            return x

        xiter = 0
        for term, (wf, wex) in zip(eq.terms, exprs):
            if len(wex) != 0 and wex not in dts:
                op_len = wex.count("C") + wex.count("D")
                if op_len not in dts:
                    dts[op_len] = {}
                if wex not in dts[op_len]:
                    dts[op_len][wex] = np.zeros((len(cidx),) * op_len, dtype=h1e.dtype)
                dtx = dts[op_len][wex]
            tensors = []
            opidx = []
            result = ""
            mask = ""
            f = term.factor * wf
            for t in term.tensors:
                if t.type == b.WickTensorTypes.Tensor:
                    tensors.append(tx(tensor_d[t.name], t.indices))
                    opidx.append("".join([i.name for i in t.indices]))
                else:
                    mask += "I" if is_inactive(t.indices[0].types) else "E"
                    result += t.indices[0].name
            np_str = ",".join(opidx) + "->" + result
            if 0 not in [x.size for x in tensors]:
                ts = f * np.einsum(np_str, *tensors, optimize=True)
                if len(wex) == 0:
                    const_es += ts
                else:
                    dtx[ix(mask)] += ts.flatten()
            if iprint:
                xr = ("%20.15f" % const_es) if wex == "" else wex
                print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
            xiter += 1
        assert sorted(dts.keys()) == [2, 4]
        return dts[2], dts[4], const_es

    @staticmethod
    def make_sz(h1e, g2e, const_e, cidx, iprint=1):
        import block2 as b
        import numpy as np

        if iprint:
            print("-- Normal order (sz) using Wick's theorem")

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.InactiveAlpha] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.InactiveBeta] = b.WickIndex.parse_set("PQRSIJKLMNO")
        idx_map[b.WickIndexTypes.ExternalAlpha] = b.WickIndex.parse_set("pqrsabcdefg")
        idx_map[b.WickIndexTypes.ExternalBeta] = b.WickIndex.parse_set("PQRSABCDEFG")
        perm_map = b.MapPStrIntVectorWickPermutation()
        perm_map[("vaa", 4)] = b.WickPermutation.qc_chem()
        perm_map[("vbb", 4)] = b.WickPermutation.qc_chem()
        perm_map[("vab", 4)] = b.WickPermutation.qc_chem()[1:]

        P = lambda x: b.WickExpr.parse(x, idx_map, perm_map)
        h1 = P("SUM <pq> ha[pq] C[p] D[q]\n + SUM <PQ> hb[PQ] C[P] D[Q]")
        h2 = P(
            """
            0.5 SUM <prqs> vaa[prqs] C[p] C[q] D[s] D[r]
            0.5 SUM <prQS> vab[prQS] C[p] C[Q] D[S] D[r]
            0.5 SUM <PRqs> vab[qsPR] C[P] C[q] D[s] D[R]
            0.5 SUM <PRQS> vbb[PRQS] C[P] C[Q] D[S] D[R]
            """
        )
        eq = (h1 + h2).expand().simplify()

        ha, hb = h1e
        vaa, vab, vbb = g2e

        dts = {}
        const_es = const_e
        tensor_d = {"ha": ha, "hb": hb, "vaa": vaa, "vab": vab, "vbb": vbb}

        is_alpha = lambda x: (x & b.WickIndexTypes.Alpha) != b.WickIndexTypes.Nothing
        is_inactive = (
            lambda x: (x & b.WickIndexTypes.Inactive) != b.WickIndexTypes.Nothing
        )

        if np.array(cidx).ndim == 2:
            cidxa, cidxb = cidx
        else:
            cidxa, cidxb = cidx, cidx

        def ix(x):
            p = {"i": cidxa, "e": ~cidxa, "I": cidxb, "E": ~cidxb}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((len(cidxa),) * len(x))

        def tx(x, ix):
            for ig, ii in enumerate(ix):
                idx = (slice(None),) * ig
                if is_alpha(ii.types):
                    idx += (cidxa if is_inactive(ii.types) else ~cidxa,)
                else:
                    idx += (cidxb if is_inactive(ii.types) else ~cidxb,)
                x = x[idx]
            return x

        xiter = 0
        for term in eq.terms:
            wex = ""
            for t in term.tensors:
                if t.type != b.WickTensorTypes.Tensor:
                    if t.type == b.WickTensorTypes.CreationOperator:
                        wex += "c" if is_alpha(t.indices[0].types) else "C"
                    elif t.type == b.WickTensorTypes.DestroyOperator:
                        wex += "d" if is_alpha(t.indices[0].types) else "D"
            if len(wex) != 0 and wex not in dts:
                op_len = sum([wex.count(cd) for cd in "cdCD"])
                if op_len not in dts:
                    dts[op_len] = {}
                if wex not in dts[op_len]:
                    dts[op_len][wex] = np.zeros((len(cidxa),) * op_len, dtype=ha.dtype)
                dtx = dts[op_len][wex]
            tensors = []
            opidx = []
            result = ""
            mask = ""
            f = term.factor
            for t in term.tensors:
                if t.type == b.WickTensorTypes.Tensor:
                    tensors.append(tx(tensor_d[t.name], t.indices))
                    opidx.append("".join([i.name for i in t.indices]))
                else:
                    if is_alpha(t.indices[0].types):
                        mask += "i" if is_inactive(t.indices[0].types) else "e"
                    else:
                        mask += "I" if is_inactive(t.indices[0].types) else "E"
                    result += t.indices[0].name
            np_str = ",".join(opidx) + "->" + result
            if 0 not in [x.size for x in tensors]:
                ts = f * np.einsum(np_str, *tensors, optimize=True)
                if len(wex) == 0:
                    const_es += ts
                else:
                    dtx[ix(mask)] += ts.flatten()
            if iprint:
                xr = ("%20.15f" % const_es) if wex == "" else wex
                print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
            xiter += 1
        assert sorted(dts.keys()) == [2, 4]
        return dts[2], dts[4], const_es

    @staticmethod
    def make_sgf(h1e, g2e, const_e, cidx, iprint=1):
        import block2 as b
        import numpy as np

        if iprint:
            print("-- Normal order (sgf) using Wick's theorem")

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.Inactive] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.External] = b.WickIndex.parse_set("pqrsabcdefg")
        perm_map = b.MapPStrIntVectorWickPermutation()
        perm_map[("v", 4)] = b.WickPermutation.qc_chem()

        P = lambda x: b.WickExpr.parse(x, idx_map, perm_map)
        h = P("SUM <pq> h[pq] C[p] D[q] + 0.5 SUM <pqrs> v[pqrs] C[p] C[r] D[s] D[q]")
        eq = h.expand().simplify()

        dts = {}
        const_es = const_e
        tensor_d = {"h": h1e, "v": g2e}

        is_inactive = (
            lambda x: (x & b.WickIndexTypes.Inactive) != b.WickIndexTypes.Nothing
        )

        def ix(x):
            p = {"I": cidx, "E": ~cidx}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((len(cidx),) * len(x))

        def tx(x, ix):
            for ig, ii in enumerate(ix):
                idx = (slice(None),) * ig
                idx += (cidx if is_inactive(ii.types) else ~cidx,)
                x = x[idx]
            return x

        xiter = 0
        for term in eq.terms:
            wex = ""
            for t in term.tensors:
                if t.type != b.WickTensorTypes.Tensor:
                    if t.type == b.WickTensorTypes.CreationOperator:
                        wex += "C"
                    elif t.type == b.WickTensorTypes.DestroyOperator:
                        wex += "D"
            if len(wex) != 0 and wex not in dts:
                op_len = wex.count("C") + wex.count("D")
                if op_len not in dts:
                    dts[op_len] = {}
                if wex not in dts[op_len]:
                    dts[op_len][wex] = np.zeros((len(cidx),) * op_len, dtype=h1e.dtype)
                dtx = dts[op_len][wex]
            tensors = []
            opidx = []
            result = ""
            mask = ""
            f = term.factor
            for t in term.tensors:
                if t.type == b.WickTensorTypes.Tensor:
                    tensors.append(tx(tensor_d[t.name], t.indices))
                    opidx.append("".join([i.name for i in t.indices]))
                else:
                    mask += "I" if is_inactive(t.indices[0].types) else "E"
                    result += t.indices[0].name
            np_str = ",".join(opidx) + "->" + result
            if 0 not in [x.size for x in tensors]:
                ts = f * np.einsum(np_str, *tensors, optimize=True)
                if len(wex) == 0:
                    const_es += ts
                else:
                    dtx[ix(mask)] += ts.flatten()
            if iprint:
                xr = ("%20.15f" % const_es) if wex == "" else wex
                print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
            xiter += 1
        assert sorted(dts.keys()) == [2, 4]
        return dts[2], dts[4], const_es


class ExprBuilder:
    def __init__(self, bw=Block2Wrapper()):
        self.data = bw.bx.GeneralFCIDUMP()
        if SymmetryTypes.SU2 in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SU2
        elif SymmetryTypes.SZ in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SZ
        elif SymmetryTypes.SGF in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SGF
        elif SymmetryTypes.SGB in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SGB
        self.data.const_e = 0.0
        self.bw = bw

    def add_const(self, x):
        self.data.const_e = self.data.const_e + x
        return self

    def add_term(self, expr, idx, val):
        import numpy as np

        self.data.exprs.append(expr)
        self.data.indices.append(self.bw.b.VectorUInt16(idx))
        if (
            not isinstance(val, list)
            and not isinstance(val, tuple)
            and not isinstance(val, np.ndarray)
        ):
            val = [val]
        self.data.data.append(self.bw.VectorFL(val))
        return self

    def add_sum_term(self, expr, arr, cutoff=1e-12, fast=True):
        import numpy as np

        self.data.exprs.append(expr)
        if fast:
            self.data.add_sum_term(np.ascontiguousarray(arr), cutoff)
        else:
            idx, dt = [], []
            for ix in np.ndindex(*arr.shape):
                if abs(arr[ix]) > cutoff:
                    idx.extend(ix)
                    dt.append(arr[ix])
            self.data.indices.append(self.bw.b.VectorUInt16(idx))
            self.data.data.append(self.bw.VectorFL(dt))
        return self

    def add_terms(self, expr, arr, idx, cutoff=1e-12):
        self.data.exprs.append(expr)
        didx, dt = [], []
        for ix, v in zip(idx, arr):
            if abs(v) > cutoff:
                didx.extend(ix)
                dt.append(v)
        self.data.indices.append(self.bw.b.VectorUInt16(didx))
        self.data.data.append(self.bw.VectorFL(dt))
        return self

    def iscale(self, d):
        import numpy as np

        for i, ix in enumerate(self.data.data):
            self.data.data[i] = self.bw.VectorFL(d * np.array(ix))
        return self

    def finalize(self, adjust_order=True, merge=True):
        if adjust_order:
            self.data = self.data.adjust_order(merge=merge)
        elif merge:
            self.data.merge_terms()
        return self.data


class FermionTransform:
    @staticmethod
    def jordan_wigner(h1e, g2e):
        import numpy as np

        # sort indices to ascending order, adjusting fermion sign
        if h1e is not None:
            hixs = np.mgrid[tuple(slice(x) for x in h1e.shape)].reshape((h1e.ndim, -1))
            hop = np.array([0, 1] * h1e.size, dtype="<i1").reshape((-1, h1e.ndim))
            h1e = h1e.reshape((-1,)).copy()
            i, j = hixs
            h1e[i > j] *= -1
            hop[i > j] = hop[i > j, ::-1]
            hixs[:, i > j] = hixs[::-1, i > j]

        if g2e is not None:
            gixs = np.mgrid[tuple(slice(x) for x in g2e.shape)].reshape((g2e.ndim, -1))
            gop = np.array([0, 0, 1, 1] * g2e.size, dtype="<i1").reshape((-1, g2e.ndim))
            g2e = (g2e.transpose(0, 2, 3, 1) * 0.5).reshape((-1,)).copy()
            for x in [0, 1, 2, 0, 1, 0]:
                i, j = gixs[x], gixs[x + 1]
                g2e[i > j] *= -1
                gop[i > j, x : x + 2] = gop[i > j, x : x + 2][:, ::-1]
                gixs[x : x + 2, i > j] = gixs[x : x + 2, i > j][::-1]

        h_terms = {}

        if h1e is not None:
            for v, (xi, xj), (i, j) in zip(h1e, hop, hixs.T):
                if v == 0:
                    continue
                ex = "PM"[xi] + "Z" * (j - i) + "PM"[xj]
                if ex not in h_terms:
                    h_terms[ex] = [[], []]
                h_terms[ex][0].extend([i, *range(i, j), j])
                h_terms[ex][1].append(v)

        if g2e is not None:
            for v, (xi, xj, xk, xl), (i, j, k, l) in zip(g2e, gop, gixs.T):
                if v == 0:
                    continue
                ex = (
                    "PM"[xi]
                    + "Z" * (j - i)
                    + "PM"[xj]
                    + "PM"[xk]
                    + "Z" * (l - k)
                    + "PM"[xl]
                )
                if ex not in h_terms:
                    h_terms[ex] = [[], []]
                h_terms[ex][0].extend([i, *range(i, j), j, k, *range(k, l), l])
                h_terms[ex][1].append(v)

        for k, xv in h_terms.items():
            xv[0] = np.array(xv[0])
            xv[1] = np.array(xv[1]) * 2 ** k.count("Z")

        return h_terms


class OrbitalEntropy:
    # Table 4. J. Chem. Theory Comput. 2013, 9, 2959-2973
    ops = "1-n-N+nN:D-nD:d-Nd:Dd:C-nC:N-nN:Cd:-Nd:c-Nc:Dc:n-nN:nD:Cc:-Nc:nC:nN".split(
        ":"
    )

    @staticmethod
    def parse_expr(x):
        x = x.replace("+", "\n+").replace("-", "\n-")
        x = x.replace("n", "cd").replace("N", "CD")
        x = [k for k in x.split("\n") if k != ""]
        if not x[0].startswith("+") and not x[0].startswith("-"):
            x[0] = "+" + x[0]
        for ik, k in enumerate(x):
            if k[1:] == "1":
                x[ik] = k[0]
        return x

    @staticmethod
    def get_one_orb_rdm_h_terms(n_sites):
        h_terms = {}
        for i in range(n_sites):
            ih_terms = []
            for ix in [0, 5, 10, 15]:
                xh_terms = {}
                for x in OrbitalEntropy.parse_expr(OrbitalEntropy.ops[ix]):
                    xh_terms[x[1:]] = ([i] * len(x[1:]), 1.0 if x[0] == "+" else -1.0)
                ih_terms.append(xh_terms)
            h_terms[(i,)] = ih_terms
        return h_terms

    @staticmethod
    def get_two_orb_rdm_h_terms(n_sites, ij_symm=True, block_symm=True):
        h_terms = {}
        # Table 3. J. Chem. Theory Comput. 2013, 9, 2959-2973
        if block_symm:
            ts = (
                "1/1 1/6 2/5 6/1 1/11 3/9 11/1 6/6 1/16 2/15 -3/14 -4/13 6/11 7/10 "
                + "-8/9 11/6 12/5 16/1 11/11 6/16 8/14 16/6 11/16 12/15 16/11 16/16"
            )
        else:
            ts = (
                "1/1 1/6 2/5 -5/2 6/1 1/11 3/9 -9/3 11/1 6/6 1/16 2/15 -3/14 "
                + "-4/13 -5/12 6/11 7/10 -8/9 9/8 10/7 11/6 12/5 -13/4 14/3 -15/2 16/1 "
                + "11/11 6/16 8/14 -14/8 16/6 11/16 12/15 -15/12 16/11 16/16"
            )
        ts = [[int(v) for v in u.split("/")] for u in ts.split()]
        tsm = (
            "1/1 1/6 6/1 1/11 11/1 6/6 1/16 6/11 11/6 16/1 "
            + "11/11 6/16 16/6 11/16 16/11 16/16"
        )
        tsm = [[int(v) for v in u.split("/")] for u in tsm.split()]
        for i in range(n_sites):
            for j in range(n_sites):
                if ij_symm and j > i:
                    continue
                ih_terms = []
                for ix, iy in ts if i != j else tsm:
                    ff = -1 if ix < 0 else 1
                    ix = abs(ix)
                    xh_terms = {}
                    for x in OrbitalEntropy.parse_expr(OrbitalEntropy.ops[ix - 1]):
                        for y in OrbitalEntropy.parse_expr(OrbitalEntropy.ops[iy - 1]):
                            f = ff if x[0] == y[0] else -ff
                            if i <= j:
                                k = x[1:] + y[1:]
                                v = ([i] * len(x[1:]) + [j] * len(y[1:]), f)
                            else:
                                if len(x) % 2 == 0 and len(y) % 2 == 0:
                                    f *= -1.0
                                k = y[1:] + x[1:]
                                v = ([j] * len(y[1:]) + [i] * len(x[1:]), f)
                            if k in xh_terms:
                                xh_terms[k][0].extend(v[0])
                                xh_terms[k][1].append(v[1])
                            else:
                                xh_terms[k] = (v[0], [v[1]])
                    ih_terms.append(xh_terms)
                h_terms[(i, j)] = ih_terms
        return h_terms

    @staticmethod
    def get_two_orb_rdm_eigvals(ld):
        import numpy as np

        if len(ld) == 16:
            return ld
        elif len(ld) == 26:
            lx = np.zeros((16,), dtype=ld.dtype)
            ix, ip = 0, 0
            for d in [1, 2, 2, 1, 4, 1, 2, 2, 1]:
                if d == 1:
                    lx[ix] = ld[ip]
                else:
                    dd = np.zeros((d, d))
                    dd[np.triu_indices(d)] = ld[ip : ip + d * (d + 1) // 2]
                    lx[ix : ix + d] = np.linalg.eigvalsh(dd, UPLO="U")
                ix += d
                ip += d * (d + 1) // 2
            assert ix == len(lx) and ip == len(ld)
        elif len(ld) == 36:
            lx = np.zeros((16,), dtype=ld.dtype)
            ix, ip = 0, 0
            for d in [1, 2, 2, 1, 4, 1, 2, 2, 1]:
                if d == 1:
                    lx[ix] = ld[ip]
                else:
                    lx[ix : ix + d] = np.linalg.eigvalsh(
                        ld[ip : ip + d * d].reshape(d, d)
                    )
                ix += d
                ip += d * d
            assert ix == len(lx) and ip == len(ld)
        return lx


class WickSpinAdaptation:
    @staticmethod
    def spin_tag_to_pattern(x):
        """[1, 2, 2, 1] -> ((.+(.+.)0)1+.)0 ."""
        if len(x) == 0:
            return ""
        elif len(x) == 2:
            return "(.+.)0"
        elif x[0] == x[-1]:
            return "((.+%s)1+.)0" % WickSpinAdaptation.spin_tag_to_pattern(x[1:-1])
        else:
            for i in range(2, len(x) - 1, 2):
                if len(set(x[:i]) & set(x[i:])) == 0:
                    return "(%s+%s)0" % tuple(
                        WickSpinAdaptation.spin_tag_to_pattern(xx)
                        for xx in [x[:i], x[i:]]
                    )
            raise RuntimeError("Pattern not supported!")

    @staticmethod
    def adjust_spin_coupling(eq):
        """correct up to 4-body terms."""
        for term in eq.terms:
            x = [t.name for t in term.tensors if t.name[0] in "CD"]
            n = len(x)
            ii = len(term.tensors) - n
            tensors = term.tensors[ii:]
            found = True
            factor = 1
            while found:
                found = False
                for i in range(n - 1):
                    if x[i][0] != x[i + 1][0]:
                        continue
                    if (
                        (i > 0 and x[i - 1][1:] == x[i + 1][1:])
                        or (i < n - 2 and x[i + 2][1:] == x[i][1:])
                        or (n >= 6 and i == n - 2 and x[0][1:] == x[i][1:])
                        or (n >= 6 and i == 0 and x[-1][1:] == x[i + 1][1:])
                        or (
                            n >= 8
                            and i <= n - 3
                            and i >= 1
                            and x[0][1:] == x[i + 1][1:]
                            and x[-1][1:] == x[i][1:]
                        )
                        or (
                            n >= 8
                            and i == n - 4
                            and x[0][1:] == x[i][1:]
                            and x[i + 2][1:] == x[i + 3][1:]
                        )
                        or (
                            n >= 8
                            and i == 2
                            and x[i - 2][1:] == x[i - 1][1:]
                            and x[i + 1][1:] == x[-1][1:]
                        )
                        or (
                            i <= n - 4
                            and x[i][1:] == x[i + 3][1:]
                            and x[i + 1][1:] != x[i + 2][1:]
                        )
                    ):
                        factor = -factor
                        tensors[i : i + 2] = tensors[i : i + 2][::-1]
                        x[i : i + 2] = x[i : i + 2][::-1]
                        found = True
                        break
                if found:
                    continue
                for i in range(n - 2):
                    if x[i][0] != x[i + 1][0] or x[i + 1][0] != x[i + 2][0]:
                        continue
                    if (
                        (i > 0 and x[i - 1][1:] == x[i + 2][1:])
                        or (i < n - 3 and x[i + 3][1:] == x[i][1:])
                        or (n >= 6 and i == n - 3 and x[0][1:] == x[i][1:])
                    ):
                        factor = -factor
                        tensors[i : i + 3] = tensors[i : i + 3][::-1]
                        x[i : i + 3] = x[i : i + 3][::-1]
                        found = True
                        break
            term.factor *= factor
            term.tensors[ii:] = tensors

    @staticmethod
    def get_eq_exprs(eq):
        import block2 as b

        cg = b.SU2CG()
        r = []
        for term in eq.terms:
            tensors = [t for t in term.tensors if t.name[0] in "CD"]
            cds = b.VectorUInt8([t.name[0] == "C" for t in tensors])
            spin_pat = WickSpinAdaptation.spin_tag_to_pattern(
                [int(t.name[1:]) for t in tensors]
            )
            indices = b.VectorUInt16(list(range(len(tensors))))
            tensor = b.SpinPermRecoupling.make_tensor(spin_pat, indices, cds, cg).data[
                0
            ]
            assert all(
                [abs(x.factor - y.factor) < 1e-10 for x, y in zip(tensor, tensor[1:])]
            )
            r.append(
                (
                    1.0 / tensor[0].factor,
                    spin_pat.replace(".", "%s") % tuple("DC"[cd] for cd in cds),
                )
            )
        return r


class SimilarityTransform:
    @staticmethod
    def make_su2(
        h1e,
        g2e,
        ecore,
        t1,
        t2,
        scratch,
        n_elec,
        t3=None,
        ncore=0,
        ncas=-1,
        st_type=STTypes.H_HT_HT2T2,
        iprint=1,
    ):
        import block2 as b
        import numpy as np
        import os

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.Inactive] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.External] = b.WickIndex.parse_set("pqrsabcdefg")

        perm_map = b.MapPStrIntVectorWickPermutation()
        perm_map[("v", 4)] = b.WickPermutation.qc_chem()
        perm_map[("t", 2)] = b.WickPermutation.non_symmetric()
        perm_map[("tt", 4)] = b.WickPermutation.pair_symmetric(2, False)
        perm_map[("ttt", 6)] = b.WickPermutation.pair_symmetric(3, False)

        P = lambda x: b.WickExpr.parse(x, idx_map, perm_map)

        h1 = P("SUM <pq> h[pq] E1[p,q]")
        h2 = 0.5 * P("SUM <pqrs> v[prqs] E2[pq,rs]")
        tt1 = P("SUM <ai> t[ia] E1[a,i]")
        tt2 = 0.5 * P("SUM <abij> tt[ijab] E1[a,i] E1[b,j]")
        tt3 = (1.0 / 6.0) * P("SUM <abcijk> ttt[ijkabc] E1[a,i] E1[b,j] E1[c,k]")

        h = (h1 + h2).expand(-1, False, False).simplify()
        tt = (tt1 + tt2).expand(-1, False, False).simplify()

        eq = P("")
        if STTypes.H in st_type:
            eq = eq + h
        if STTypes.HT in st_type:
            eq = eq + (h ^ tt)
        if STTypes.HT1T2 in st_type:
            eq = eq + 0.5 * ((h ^ tt1) ^ tt1) + ((h ^ tt2) ^ tt1)
        if STTypes.HT2T2 in st_type:
            eq = eq + 0.5 * ((h ^ tt2) ^ tt2)
        if STTypes.HT1T3 in st_type:
            eq = eq + ((h ^ tt3) ^ tt1)
        if STTypes.HT2T3 in st_type:
            eq = eq + ((h ^ tt3) ^ tt2)

        eq = eq.expand(6, False, False).simplify()
        WickSpinAdaptation.adjust_spin_coupling(eq)
        exprs = WickSpinAdaptation.get_eq_exprs(eq)

        if not os.path.isdir(scratch):
            os.makedirs(scratch)

        nocc, nvir = t1.shape
        cidx = np.array([True] * nocc + [False] * nvir)

        tensor_d = {"h": h1e, "v": g2e, "t": t1, "tt": t2, "ttt": t3}

        cas_mask = np.array([False] * len(cidx))
        n_elec -= ncore * 2
        if ncas == -1:
            ncas = len(cidx) - ncore
        cas_mask[ncore : ncore + ncas] = True

        if iprint:
            print(
                "(%do, %de) -> CAS(%do, %de)"
                % (len(cidx), n_elec + ncore * 2, ncas, n_elec)
            )
            print("ST Hamiltonian = ")
            print("NTERMS = %5d" % len(eq.terms))
            if iprint >= 3:
                print(eq)

        ccidx = cidx & cas_mask
        vcidx = (~cidx) & cas_mask
        icas = cas_mask[cidx]
        vcas = cas_mask[~cidx]
        xicas = cidx[cas_mask]
        xvcas = (~cidx)[cas_mask]
        h_terms = {}

        def ix(x):
            p = {"I": xicas, "E": xvcas}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((ncas,) * len(x))

        is_inactive = (
            lambda x: (x & b.WickIndexTypes.Inactive) != b.WickIndexTypes.Nothing
        )

        def tx(x, ix, rx, in_cas):
            for ig, (ii, rr) in enumerate(zip(ix, rx)):
                idx = (slice(None),) * ig
                if in_cas:
                    if rr:
                        idx += (icas if is_inactive(ii.types) else vcas,)
                elif rr:
                    idx += (ccidx if is_inactive(ii.types) else vcidx,)
                else:
                    idx += (cidx if is_inactive(ii.types) else ~cidx,)
                x = x[idx]
            return x

        e_terms = {}
        for term, (wf, expr) in zip(eq.terms, exprs):
            if expr not in e_terms:
                e_terms[expr] = []
            e_terms[expr].append((term, wf))

        xiter = 0
        for expr, terms in e_terms.items():
            if expr != "":
                h_terms[expr] = scratch + "/ST-DMRG." + expr + ".npy"
                op_len = expr.count("C") + expr.count("D")
                dtx = np.zeros((ncas,) * op_len, dtype=t1.dtype)
            for term, wf in terms:
                tensors = []
                opidx = []
                result = ""
                mask = ""
                f = term.factor * wf
                for t in term.tensors:
                    if t.type != b.WickTensorTypes.Tensor:
                        if is_inactive(t.indices[0].types):
                            mask += "I"
                        else:
                            mask += "E"
                        result += t.indices[0].name
                for t in term.tensors:
                    if t.type == b.WickTensorTypes.Tensor:
                        cmask = [it.name in result for it in t.indices]
                        tensors.append(
                            tx(tensor_d[t.name], t.indices, cmask, t.name[0] == "t")
                        )
                        opidx.append("".join([i.name for i in t.indices]))
                np_str = ",".join(opidx) + "->" + result
                ts = f * np.einsum(np_str, *tensors, optimize=True)
                if len(expr) == 0:
                    ecore += ts
                else:
                    dtx[ix(mask)] += ts.flatten()
                if iprint >= 2:
                    xr = ("%20.15f" % ecore) if expr == "" else expr
                    print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
                xiter += 1
            if expr != "":
                np.save(h_terms[expr], dtx)
                dtx = None

        if iprint:
            print("ECORE = %20.15f" % ecore)

        return h_terms, ecore, ncas, n_elec

    @staticmethod
    def make_sz(
        h1e,
        g2e,
        ecore,
        t1,
        t2,
        scratch,
        n_elec,
        ncore=0,
        ncas=-1,
        st_type=STTypes.H_HT_HT2T2,
        iprint=1,
    ):
        import block2 as b
        import numpy as np
        import os

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.InactiveAlpha] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.InactiveBeta] = b.WickIndex.parse_set("PQRSIJKLMNO")
        idx_map[b.WickIndexTypes.ExternalAlpha] = b.WickIndex.parse_set("pqrsabcdefg")
        idx_map[b.WickIndexTypes.ExternalBeta] = b.WickIndex.parse_set("PQRSABCDEFG")

        perm_map = b.MapPStrIntVectorWickPermutation()
        perm_map[("vaa", 4)] = b.WickPermutation.qc_chem()
        perm_map[("vbb", 4)] = b.WickPermutation.qc_chem()
        perm_map[("vab", 4)] = b.WickPermutation.qc_chem()[1:]
        perm_map[("ta", 2)] = b.WickPermutation.non_symmetric()
        perm_map[("tb", 2)] = b.WickPermutation.non_symmetric()
        perm_map[("taa", 4)] = b.WickPermutation.four_anti()
        perm_map[("tbb", 4)] = b.WickPermutation.four_anti()
        perm_map[("tab", 4)] = b.WickPermutation.pair_symmetric(2, False)

        P = lambda x: b.WickExpr.parse(x, idx_map, perm_map)

        h1 = P("SUM <pq> ha[pq] C[p] D[q]\n + SUM <PQ> hb[PQ] C[P] D[Q]")
        h2 = P(
            """
            0.5 SUM <prqs> vaa[prqs] C[p] C[q] D[s] D[r]
            0.5 SUM <prQS> vab[prQS] C[p] C[Q] D[S] D[r]
            0.5 SUM <PRqs> vab[qsPR] C[P] C[q] D[s] D[R]
            0.5 SUM <PRQS> vbb[PRQS] C[P] C[Q] D[S] D[R]
            """
        )
        tt1 = P("SUM <ai> ta[ia] C[a] D[i]\n + SUM <AI> tb[IA] C[A] D[I]")
        # this def is consistent with pyscf init t amps
        tt2 = P(
            """
            0.25 SUM <aibj> taa[ijab] C[a] C[b] D[j] D[i]
            0.50 SUM <aiBJ> tab[iJaB] C[a] C[B] D[J] D[i]
            0.50 SUM <AIbj> tab[jIbA] C[A] C[b] D[j] D[I]
            0.25 SUM <AIBJ> tbb[IJAB] C[A] C[B] D[J] D[I]
            """
        )

        h = (h1 + h2).expand().simplify()
        tt = (tt1 + tt2).expand().simplify()

        eq = P("")
        if STTypes.H in st_type:
            eq = eq + h
        if STTypes.HT in st_type:
            eq = eq + (h ^ tt)
        if STTypes.HT1T2 in st_type:
            eq = eq + 0.5 * ((h ^ tt1) ^ tt1) + ((h ^ tt2) ^ tt1)
        if STTypes.HT2T2 in st_type:
            eq = eq + 0.5 * ((h ^ tt2) ^ tt2)

        eq = eq.expand(6).simplify()

        if not os.path.isdir(scratch):
            os.makedirs(scratch)

        ha, hb = h1e
        vaa, vab, vbb = g2e
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2

        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape
        cidxa = np.array([True] * nocca + [False] * nvira)
        cidxb = np.array([True] * noccb + [False] * nvirb)

        tensor_d = {
            "ha": ha,
            "hb": hb,
            "vaa": vaa,
            "vab": vab,
            "vbb": vbb,
            "ta": t1a,
            "tb": t1b,
            "taa": t2aa,
            "tab": t2ab,
            "tbb": t2bb,
        }

        cas_mask = np.array([False] * len(cidxa))
        n_elec -= ncore * 2
        if ncas == -1:
            ncas = len(cidxa) - ncore
        cas_mask[ncore : ncore + ncas] = True

        if iprint:
            print(
                "(%do, %de) -> CAS(%do, %de)"
                % (len(cidxa), n_elec + ncore * 2, ncas, n_elec)
            )
            print("ST Hamiltonian = ")
            print("NTERMS = %5d" % len(eq.terms))
            if iprint >= 3:
                print(eq)

        ccidxa = cidxa & cas_mask
        ccidxb = cidxb & cas_mask
        vcidxa = (~cidxa) & cas_mask
        vcidxb = (~cidxb) & cas_mask
        icasa = cas_mask[cidxa]
        icasb = cas_mask[cidxb]
        vcasa = cas_mask[~cidxa]
        vcasb = cas_mask[~cidxb]
        xicasa = cidxa[cas_mask]
        xicasb = cidxb[cas_mask]
        xvcasa = (~cidxa)[cas_mask]
        xvcasb = (~cidxb)[cas_mask]
        h_terms = {}

        def ix(x):
            p = {"i": xicasa, "e": xvcasa, "I": xicasb, "E": xvcasb}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((ncas,) * len(x))

        is_alpha = lambda x: (x & b.WickIndexTypes.Alpha) != b.WickIndexTypes.Nothing
        is_inactive = (
            lambda x: (x & b.WickIndexTypes.Inactive) != b.WickIndexTypes.Nothing
        )

        def tx(x, ix, rx, in_cas):
            for ig, (ii, rr) in enumerate(zip(ix, rx)):
                idx = (slice(None),) * ig
                if is_alpha(ii.types):
                    if in_cas:
                        if rr:
                            idx += (icasa if is_inactive(ii.types) else vcasa,)
                    elif rr:
                        idx += (ccidxa if is_inactive(ii.types) else vcidxa,)
                    else:
                        idx += (cidxa if is_inactive(ii.types) else ~cidxa,)
                else:
                    if in_cas:
                        if rr:
                            idx += (icasb if is_inactive(ii.types) else vcasb,)
                    elif rr:
                        idx += (ccidxb if is_inactive(ii.types) else vcidxb,)
                    else:
                        idx += (cidxb if is_inactive(ii.types) else ~cidxb,)
                x = x[idx]
            return x

        e_terms = {}

        for term in eq.terms:
            expr = ""
            for t in term.tensors:
                if t.type != b.WickTensorTypes.Tensor:
                    if t.type == b.WickTensorTypes.CreationOperator:
                        expr += "c" if is_alpha(t.indices[0].types) else "C"
                    elif t.type == b.WickTensorTypes.DestroyOperator:
                        expr += "d" if is_alpha(t.indices[0].types) else "D"
            if expr not in e_terms:
                e_terms[expr] = []
            e_terms[expr].append(term)

        xiter = 0
        for expr, terms in e_terms.items():
            if expr != "":
                h_terms[expr] = scratch + "/ST-DMRG." + expr + ".npy"
                dtx = np.zeros((ncas,) * len(expr), dtype=t1a.dtype)
            for term in terms:
                tensors = []
                opidx = []
                result = ""
                mask = ""
                f = term.factor
                for t in term.tensors:
                    if t.type != b.WickTensorTypes.Tensor:
                        if is_inactive(t.indices[0].types):
                            mask += "i" if is_alpha(t.indices[0].types) else "I"
                        else:
                            mask += "e" if is_alpha(t.indices[0].types) else "E"
                        result += t.indices[0].name
                for t in term.tensors:
                    if t.type == b.WickTensorTypes.Tensor:
                        cmask = [it.name in result for it in t.indices]
                        tensors.append(
                            tx(tensor_d[t.name], t.indices, cmask, t.name[0] == "t")
                        )
                        opidx.append("".join([i.name for i in t.indices]))
                np_str = ",".join(opidx) + "->" + result
                if 0 not in [x.size for x in tensors]:
                    ts = f * np.einsum(np_str, *tensors, optimize=True)
                    if len(expr) == 0:
                        ecore += ts
                    else:
                        dtx[ix(mask)] += ts.flatten()
                if iprint >= 2:
                    xr = ("%20.15f" % ecore) if expr == "" else expr
                    print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
                xiter += 1
            if expr != "":
                np.save(h_terms[expr], dtx)
                dtx = None

        if iprint:
            print("ECORE = %20.15f" % ecore)

        return h_terms, ecore, ncas, n_elec

    @staticmethod
    def make_sgf(
        h1e,
        g2e,
        ecore,
        t1,
        t2,
        scratch,
        n_elec,
        t3=None,
        ncore=0,
        ncas=-1,
        st_type=STTypes.H_HT_HT2T2,
        iprint=1,
    ):
        import block2 as b
        import numpy as np
        import os

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.Inactive] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.External] = b.WickIndex.parse_set("pqrsabcdefg")

        perm_map = b.MapPStrIntVectorWickPermutation()
        perm_map[("v", 4)] = b.WickPermutation.four_anti()
        perm_map[("t", 2)] = b.WickPermutation.non_symmetric()
        perm_map[("tt", 4)] = b.WickPermutation.pair_anti_symmetric(2)
        perm_map[("ttt", 6)] = b.WickPermutation.pair_anti_symmetric(3)

        P = lambda x: b.WickExpr.parse(x, idx_map, perm_map)

        h1 = P("SUM <pq> h[pq] C[p] D[q]")
        h2 = 0.25 * P("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
        tt1 = P("SUM <ai> t[ia] C[a] D[i]")
        tt2 = 0.25 * P("SUM <abij> tt[ijab] C[a] C[b] D[j] D[i]")
        tt3 = (1.0 / 36.0) * P("SUM <abcijk> ttt[ijkabc] C[a] C[b] C[c] D[k] D[j] D[i]")

        h = (h1 + h2).expand().simplify()
        tt = (tt1 + tt2).expand().simplify()

        eq = P("")
        if STTypes.H in st_type:
            eq = eq + h
        if STTypes.HT in st_type:
            eq = eq + (h ^ tt)
        if STTypes.HT1T2 in st_type:
            eq = eq + 0.5 * ((h ^ tt1) ^ tt1) + ((h ^ tt2) ^ tt1)
        if STTypes.HT2T2 in st_type:
            eq = eq + 0.5 * ((h ^ tt2) ^ tt2)
        if STTypes.HT1T3 in st_type:
            eq = eq + ((h ^ tt3) ^ tt1)
        if STTypes.HT2T3 in st_type:
            eq = eq + ((h ^ tt3) ^ tt2)

        eq = eq.expand(6).simplify()

        if not os.path.isdir(scratch):
            os.makedirs(scratch)

        nocc, nvir = t1.shape
        cidx = np.array([True] * nocc + [False] * nvir)

        g2e = g2e.transpose(0, 2, 1, 3) - g2e.transpose(0, 2, 3, 1)

        tensor_d = {"h": h1e, "v": g2e, "t": t1, "tt": t2, "ttt": t3}

        cas_mask = np.array([False] * len(cidx))
        n_elec -= ncore
        if ncas == -1:
            ncas = len(cidx) - ncore
        cas_mask[ncore : ncore + ncas] = True

        if iprint:
            print(
                "(%do, %de) -> CAS(%do, %de)"
                % (len(cidx), n_elec + ncore, ncas, n_elec)
            )
            print("ST Hamiltonian = ")
            print("NTERMS = %5d" % len(eq.terms))
            if iprint >= 3:
                print(eq)

        ccidx = cidx & cas_mask
        vcidx = (~cidx) & cas_mask
        icas = cas_mask[cidx]
        vcas = cas_mask[~cidx]
        xicas = cidx[cas_mask]
        xvcas = (~cidx)[cas_mask]
        h_terms = {}

        def ix(x):
            p = {"I": xicas, "E": xvcas}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((ncas,) * len(x))

        is_inactive = (
            lambda x: (x & b.WickIndexTypes.Inactive) != b.WickIndexTypes.Nothing
        )

        def tx(x, ix, rx, in_cas):
            for ig, (ii, rr) in enumerate(zip(ix, rx)):
                idx = (slice(None),) * ig
                if in_cas:
                    if rr:
                        idx += (icas if is_inactive(ii.types) else vcas,)
                elif rr:
                    idx += (ccidx if is_inactive(ii.types) else vcidx,)
                else:
                    idx += (cidx if is_inactive(ii.types) else ~cidx,)
                x = x[idx]
            return x

        e_terms = {}

        for term in eq.terms:
            expr = ""
            for t in term.tensors:
                if t.type != b.WickTensorTypes.Tensor:
                    if t.type == b.WickTensorTypes.CreationOperator:
                        expr += "C"
                    elif t.type == b.WickTensorTypes.DestroyOperator:
                        expr += "D"
            if expr not in e_terms:
                e_terms[expr] = []
            e_terms[expr].append(term)

        xiter = 0
        for expr, terms in e_terms.items():
            if expr != "":
                h_terms[expr] = scratch + "/ST-DMRG." + expr + ".npy"
                dtx = np.zeros((ncas,) * len(expr), dtype=t1.dtype)
            for term in terms:
                tensors = []
                opidx = []
                result = ""
                mask = ""
                f = term.factor
                for t in term.tensors:
                    if t.type != b.WickTensorTypes.Tensor:
                        if is_inactive(t.indices[0].types):
                            mask += "I"
                        else:
                            mask += "E"
                        result += t.indices[0].name
                for t in term.tensors:
                    if t.type == b.WickTensorTypes.Tensor:
                        cmask = [it.name in result for it in t.indices]
                        tensors.append(
                            tx(tensor_d[t.name], t.indices, cmask, t.name[0] == "t")
                        )
                        opidx.append("".join([i.name for i in t.indices]))
                np_str = ",".join(opidx) + "->" + result
                ts = f * np.einsum(np_str, *tensors, optimize=True)
                if len(expr) == 0:
                    ecore += ts
                else:
                    dtx[ix(mask)] += ts.flatten()
                if iprint >= 2:
                    xr = ("%20.15f" % ecore) if expr == "" else expr
                    print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
                xiter += 1
            if expr != "":
                np.save(h_terms[expr], dtx)
                dtx = None

        if iprint:
            print("ECORE = %20.15f" % ecore)

        return h_terms, ecore, ncas, n_elec
