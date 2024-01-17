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
    SAny = 64
    SAnySU2 = 64 | 1
    SAnySZ = 64 | 2
    SAnySGF = 64 | 4
    SAnySGFCPX = 64 | 16 | 4
    SO4 = 128
    PHSU2 = 256
    SO3 = 512
    LZ = 1024
    SAnySO4 = 64 | 128
    SAnyPHSU2 = 64 | 256
    SAnySO3 = 64 | 512
    SAnySU2LZ = 64 | 1 | 1024
    SAnySZLZ = 64 | 2 | 1024
    SAnySGFLZ = 64 | 4 | 1024


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
    Length = 8192
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
    BlockedLengthSVD = 16 | 2 | 8192
    FastBlockedLengthSVD = 16 | 8 | 2 | 8192
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
        has_su2 = hasattr(b, "su2")
        has_sz = hasattr(b, "sz")
        has_sany = hasattr(b, "sany")
        if SymmetryTypes.CPX in symm_type and not has_cpx:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_COMPLEX=ON'!")
        elif SymmetryTypes.SP in symm_type and not has_sp:
            raise RuntimeError(
                "block2 needs to be compiled with '-DUSE_SINGLE_PREC=ON'!"
            )
        elif SymmetryTypes.SAny in symm_type and not has_sany:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SANY=ON'!")
        elif (
            SymmetryTypes.SGF in symm_type
            and not has_sgf
            and (SymmetryTypes.SAny not in symm_type)
        ):
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SG=ON'!")
        elif (
            SymmetryTypes.SGB in symm_type
            and not has_sgb
            and (SymmetryTypes.SAny not in symm_type)
        ):
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SG=ON'!")
        elif (
            SymmetryTypes.SU2 in symm_type
            and not has_su2
            and (SymmetryTypes.SAny not in symm_type)
        ):
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SU2SZ=ON'!")
        elif (
            SymmetryTypes.SZ in symm_type
            and not has_sz
            and (SymmetryTypes.SAny not in symm_type)
        ):
            raise RuntimeError("block2 needs to be compiled with '-DUSE_SU2SZ=ON'!")
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
        self.VectorPG = b.VectorUInt8
        if SymmetryTypes.SAny in symm_type:
            self.bs = self.bx.sany
            self.bcs = self.bc.sany if self.bc is not None else None
            self.brs = b.sany
            self.SX = self.SXT = b.SAny
            if SymmetryTypes.LZ in symm_type and SymmetryTypes.SU2 in symm_type:

                def init_su2_lz(n, twos_low, twos, lz=None):
                    if lz is None:
                        twos_low, twos, lz = twos_low, twos_low, twos
                    q = self.SXT()
                    q.types[0] = self.b.SAnySymmTypes.U1Fermi
                    q.types[1] = self.b.SAnySymmTypes.SU2Fermi
                    q.types[2] = self.b.SAnySymmTypes.SU2Fermi
                    q.types[3] = self.b.SAnySymmTypes.LZ
                    q.values[0] = n
                    q.values[1] = twos_low
                    q.values[2] = twos
                    q.values[3] = lz
                    return q

                self.SX = init_su2_lz
            elif SymmetryTypes.LZ in symm_type and SymmetryTypes.SZ in symm_type:

                def init_sz_lz(n, twos, lz):
                    q = self.SXT()
                    q.types[0] = self.b.SAnySymmTypes.U1Fermi
                    q.types[1] = self.b.SAnySymmTypes.U1Fermi
                    q.types[2] = self.b.SAnySymmTypes.LZ
                    q.values[0] = n
                    q.values[1] = twos
                    q.values[2] = lz
                    return q

                self.SX = init_sz_lz
            elif SymmetryTypes.LZ in symm_type and SymmetryTypes.SGF in symm_type:

                def init_sgf_lz(n, lz):
                    q = self.SXT()
                    q.types[0] = self.b.SAnySymmTypes.U1Fermi
                    q.types[1] = self.b.SAnySymmTypes.LZ
                    q.values[0] = n
                    q.values[1] = lz
                    return q

                self.SX = init_sgf_lz
            elif SymmetryTypes.SU2 in symm_type or SymmetryTypes.SO3 in symm_type:
                self.SX = b.SAny.init_su2
            elif SymmetryTypes.SZ in symm_type:
                self.SX = b.SAny.init_sz
            elif SymmetryTypes.SGF in symm_type:
                self.SX = b.SAny.init_sgf
            elif SymmetryTypes.SO4 in symm_type:

                def init_so4(n, twos, nh=None, twosh=None):
                    q = self.SXT()
                    q.types[0] = q.types[1] = self.b.SAnySymmTypes.SU2
                    q.types[2] = q.types[3] = self.b.SAnySymmTypes.SU2Fermi
                    q.values[0] = n
                    q.values[1] = n if nh is None else nh
                    q.values[2] = twos
                    q.values[3] = twos if twosh is None else twosh
                    return q

                self.SX = init_so4
            elif SymmetryTypes.PHSU2 in symm_type:

                def init_phsu2(n, twos, nh=None):
                    q = self.SXT()
                    q.types[0] = self.b.SAnySymmTypes.U1Fermi
                    q.types[1] = q.types[2] = self.b.SAnySymmTypes.SU2
                    q.values[0] = twos
                    q.values[1] = n
                    q.values[2] = n if nh is None else nh
                    return q

                self.SX = init_phsu2
            self.VectorSX = b.VectorSAny
            self.VectorVectorSX = b.VectorVectorSAny
            self.VectorPG = b.VectorInt
        elif SymmetryTypes.SU2 in symm_type:
            self.bs = self.bx.su2
            self.bcs = self.bc.su2 if self.bc is not None else None
            self.brs = b.su2
            self.SX = self.SXT = b.SU2
            self.VectorSX = b.VectorSU2
            self.VectorVectorSX = b.VectorVectorSU2
        elif SymmetryTypes.SZ in symm_type:
            self.bs = self.bx.sz
            self.bcs = self.bc.sz if self.bc is not None else None
            self.brs = b.sz
            self.SX = self.SXT = b.SZ
            self.VectorSX = b.VectorSZ
            self.VectorVectorSX = b.VectorVectorSZ
        elif SymmetryTypes.SGF in symm_type:
            self.bs = self.bx.sgf
            self.bcs = self.bc.sgf if self.bc is not None else None
            self.brs = b.sgf
            self.SX = self.SXT = b.SGF
            self.VectorSX = b.VectorSGF
            self.VectorVectorSX = b.VectorVectorSGF
        elif SymmetryTypes.SGB in symm_type:
            self.bs = self.bx.sgb
            self.bcs = self.bc.sgb if self.bc is not None else None
            self.brs = b.sgb
            self.SX = self.SXT = b.SGB
            self.VectorSX = b.VectorSGB
            self.VectorVectorSX = b.VectorVectorSGB

    def set_symmetry_groups(self, *args):
        assert self.SXT == self.b.SAny and len(args) <= 6

        def init_sany(*qargs):
            q = self.SXT()
            assert len(qargs) == len(args)
            for ix, (ta, qa) in enumerate(zip(args, qargs)):
                if ta.startswith("Z") and ta.endswith("Fermi"):
                    q.types[ix] = self.b.SAnySymmTypes.ZNFermi + int(ta[1:-5])
                elif ta.startswith("Z"):
                    q.types[ix] = self.b.SAnySymmTypes.ZN + int(ta[1:])
                else:
                    q.types[ix] = getattr(self.b.SAnySymmTypes, ta)
                q.values[ix] = qa
            return q

        self.SX = init_sany


class DMRGDriver:
    def __init__(
        self,
        stack_mem=1 << 30,
        scratch="./nodex",
        clean_scratch=True,
        restart_dir=None,
        n_threads=None,
        n_mkl_threads=1,
        symm_type=SymmetryTypes.SU2,
        mpi=None,
        stack_mem_ratio=0.4,
        fp_codec_cutoff=1e-16,
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
        self.fp_codec_cutoff = fp_codec_cutoff
        self.symm_type = symm_type
        self.clean_scratch = clean_scratch
        bw = self.bw

        if n_threads is None:
            n_threads = bw.b.Global.threading.n_threads_global
        bw.b.Global.threading = bw.b.Threading(
            bw.b.ThreadingTypes.OperatorBatchedGEMM | bw.b.ThreadingTypes.Global,
            n_threads,
            n_threads // n_mkl_threads,
            n_mkl_threads,
        )
        bw.b.Global.threading.seq_type = bw.b.SeqTypes.Tasked
        self.reorder_idx = None
        self.pg = "c1"
        self.orb_sym = None
        self.ghamil = None
        self._dmrg = None
        self._sweep_wfn_spectra = None

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
                if self.fp_codec_cutoff != -1:
                    bw.b.Global.frame.fp_codec = bw.b.DoubleFPCodec(
                        self.fp_codec_cutoff, 1024
                    )
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
                if self.fp_codec_cutoff != -1:
                    bw.b.Global.frame_float.fp_codec = bw.b.FloatFPCodec(
                        self.fp_codec_cutoff, 1024
                    )
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

    def set_symmetry_groups(self, *args):
        self.bw.set_symmetry_groups(*args)

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
        pauli_mode=False,
        vacuum=None,
        left_vacuum=None,
        target=None,
        hamil_init=True,
    ):
        bw = self.bw
        import numpy as np

        if target is None:
            if heis_twos != -1 and bw.SX == bw.b.SU2 and n_elec == 0:
                n_elec = n_sites * heis_twos
            elif heis_twos == 1 and SymmetryTypes.SGB in bw.symm_type and n_elec != 0:
                n_elec = 2 * n_elec - n_sites
            if pg_irrep is None:
                if hasattr(self, "pg_irrep"):
                    pg_irrep = self.pg_irrep
                else:
                    pg_irrep = 0
            if (
                SymmetryTypes.SU2 not in bw.symm_type
                and SymmetryTypes.PHSU2 not in bw.symm_type
                and SymmetryTypes.SO4 not in bw.symm_type
                and SymmetryTypes.SO3 not in bw.symm_type
            ) or heis_twos != -1:
                singlet_embedding = False
            if SymmetryTypes.SO4 in bw.symm_type:
                self.vacuum = bw.SX(0, 0)
                if singlet_embedding:
                    self.target = bw.SX(0, 0)
                    self.left_vacuum = bw.SX(abs(n_elec - n_sites), spin)
                else:
                    self.target = bw.SX(abs(n_elec - n_sites), spin)
                    self.left_vacuum = bw.SX(0, 0)
            elif SymmetryTypes.PHSU2 in bw.symm_type:
                self.vacuum = bw.SX(0, 0)
                if singlet_embedding:
                    self.target = bw.SX(0, spin + abs(n_elec - n_sites) % 2)
                    self.left_vacuum = bw.SX(
                        abs(n_elec - n_sites), abs(n_elec - n_sites) % 2
                    )
                else:
                    self.target = bw.SX(abs(n_elec - n_sites), spin)
                    self.left_vacuum = bw.SX(0, 0)
            elif SymmetryTypes.SO3 in bw.symm_type:
                self.vacuum = bw.SX(0, 0, 0) if vacuum is None else vacuum
                if singlet_embedding:
                    assert heis_twosz == 0
                    self.target = bw.SX(n_elec, 0, 0)
                    self.left_vacuum = bw.SX(0, pg_irrep, 0)
                else:
                    self.target = bw.SX(n_elec, pg_irrep, 0)
                    self.left_vacuum = self.vacuum
            elif SymmetryTypes.LZ in bw.symm_type and SymmetryTypes.SGF in bw.symm_type:
                self.vacuum = bw.SX(0, 0) if vacuum is None else vacuum
                assert heis_twosz == 0
                self.target = bw.SX(n_elec, pg_irrep)
                self.left_vacuum = self.vacuum
            else:
                self.vacuum = bw.SX(0, 0, 0) if vacuum is None else vacuum
                if singlet_embedding:
                    assert heis_twosz == 0
                    self.target = bw.SX(n_elec + spin % 2, 0, pg_irrep)
                    self.left_vacuum = bw.SX(spin % 2, spin, 0)
                else:
                    self.target = bw.SX(
                        n_elec if heis_twosz == 0 else heis_twosz, spin, pg_irrep
                    )
                    self.left_vacuum = self.vacuum
        else:
            self.vacuum = bw.SX(0, 0, 0) if vacuum is None else vacuum
            self.target = target
            self.left_vacuum = self.vacuum if left_vacuum is None else left_vacuum
        self.n_sites = n_sites
        self.heis_twos = heis_twos
        if orb_sym is None:
            self.orb_sym = bw.VectorPG([0] * self.n_sites)
        else:
            if np.array(orb_sym).ndim == 2:
                self.orb_sym = bw.VectorPG(list(orb_sym[0]) + list(orb_sym[1]))
            else:
                self.orb_sym = bw.VectorPG(orb_sym)
        if hamil_init:
            if SymmetryTypes.SO4 in bw.symm_type:
                self.ghamil = self.get_so4_hamiltonian()
            elif SymmetryTypes.PHSU2 in bw.symm_type:
                self.ghamil = self.get_phsu2_hamiltonian()
            elif SymmetryTypes.SO3 in bw.symm_type:
                self.ghamil = self.get_so3_hamiltonian()
            elif SymmetryTypes.LZ in bw.symm_type:
                self.ghamil = self.get_lz_hamiltonian()
            elif pauli_mode:
                self.ghamil = self.get_pauli_hamiltonian()
            else:
                self.ghamil = bw.bs.GeneralHamiltonian(
                    self.vacuum, self.n_sites, self.orb_sym, self.heis_twos
                )

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

        if para_type == ParallelTypes.Nothing or (self.mpi is None and msize is None):
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
        if (self.mpi is not None and self.mpi.rank != self.mpi.root) or (
            self.mpi is None and mrank != 0
        ):
            const = 0
        return h1e, g2e, const

    def get_pauli_hamiltonian(self):
        assert SymmetryTypes.SGB in self.symm_type
        GH = self.bw.bs.GeneralHamiltonian
        super_self = self
        import numpy as np

        class PauliHamiltonian(GH):
            def __init__(self, vacuum, n_sites, orb_sym):
                GH.__init__(self)
                self.opf = super_self.bw.bs.OperatorFunctions(super_self.bw.brs.CG())
                self.vacuum = vacuum
                self.n_sites = n_sites
                self.orb_sym = orb_sym
                self.basis = super_self.bw.brs.VectorStateInfo(
                    [self.get_site_basis(m) for m in range(self.n_sites)]
                )
                self.site_op_infos = super_self.bw.brs.VectorVectorPLMatInfo(
                    [super_self.bw.brs.VectorPLMatInfo() for _ in range(self.n_sites)]
                )
                self.site_norm_ops = super_self.bw.bs.VectorMapStrSpMat(
                    [super_self.bw.bs.MapStrSpMat() for _ in range(self.n_sites)]
                )
                self.init_site_ops()

            def get_site_basis(self, m):
                """Single site states."""
                bz = super_self.bw.brs.StateInfo(super_self.bw.SX(0, 0))
                bz.n_states[0] = 2
                return bz

            def init_site_ops(self):
                """Initialize operator quantum numbers at each site (site_op_infos)
                and primitive (single character) site operators (site_norm_ops)."""
                op_defs = {
                    "": np.array([1.0, 0.0, 0.0, 1.0]),
                    "I": np.array([1.0, 0.0, 0.0, 1.0]),
                    "X": np.array([0.0, 1.0, 1.0, 0.0]),
                    "Y": np.array([0.0, -1.0, 1.0, 0.0]),
                    "Z": np.array([-1.0, 0.0, 0.0, 1.0]),
                    "N": np.array([0.0, 0.0, 0.0, 1.0]),
                }
                i_alloc = super_self.bw.b.IntVectorAllocator()
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                q = self.vacuum
                for m in range(self.n_sites):
                    mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                    mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                    self.site_op_infos[m].append((q, mat))
                for m in range(self.n_sites):
                    info = self.find_site_op_info(m, super_self.bw.SX(0, 0, 0))
                    for op, x in op_defs.items():
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0))] = x
                        self.site_norm_ops[m][op] = mat

            def get_site_string_ops(self, m, ops):
                """Construct longer site operators from primitive ones."""
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                for k in ops:
                    if k in self.site_norm_ops[m]:
                        ops[k] = self.site_norm_ops[m][k]
                    else:
                        xx = self.site_norm_ops[m][k[0]]
                        for p in k[1:]:
                            xp = self.site_norm_ops[m][p]
                            q = xx.info.delta_quantum + xp.info.delta_quantum
                            mat = super_self.bw.bs.SparseMatrix(d_alloc)
                            mat.allocate(self.find_site_op_info(m, q))
                            self.opf.product(0, xx, xp, mat)
                            xx = mat
                        ops[k] = self.site_norm_ops[m][k] = xx
                return ops

            def init_string_quanta(self, exprs, term_l, left_vacuum):
                """Quantum number for string operators (orbital independent part)."""
                return super_self.bw.VectorVectorSX(
                    [
                        super_self.bw.VectorSX([self.vacuum] * (len(expr) + 1))
                        for expr in exprs
                    ]
                )

            def get_string_quanta(self, ref, expr, idxs, k):
                """Quantum number for string operators (orbital dependent part)."""
                return self.vacuum, self.vacuum

            def get_string_quantum(self, expr, idxs):
                """Total quantum number for a string operator."""
                return self.vacuum

            def deallocate(self):
                """Release memory."""
                for ops in self.site_norm_ops:
                    for p in ops.values():
                        p.deallocate()
                for infos in self.site_op_infos:
                    for _, p in infos:
                        p.deallocate()
                for bz in self.basis:
                    bz.deallocate()

        return PauliHamiltonian(self.vacuum, self.n_sites, self.orb_sym)

    def get_so4_hamiltonian(self):
        assert SymmetryTypes.SO4 in self.symm_type
        GH = self.bw.bs.GeneralHamiltonian
        super_self = self
        import numpy as np

        class SO4Hamiltonian(GH):
            def __init__(self, vacuum, n_sites, orb_sym):
                GH.__init__(self)
                self.opf = super_self.bw.bs.OperatorFunctions(super_self.bw.brs.CG())
                self.vacuum = vacuum
                self.n_sites = n_sites
                self.orb_sym = orb_sym
                self.basis = super_self.bw.brs.VectorStateInfo(
                    [self.get_site_basis(m) for m in range(self.n_sites)]
                )
                self.site_op_infos = super_self.bw.brs.VectorVectorPLMatInfo(
                    [super_self.bw.brs.VectorPLMatInfo() for _ in range(self.n_sites)]
                )
                self.site_norm_ops = super_self.bw.bs.VectorMapStrSpMat(
                    [super_self.bw.bs.MapStrSpMat() for _ in range(self.n_sites)]
                )
                self.init_site_ops()

            def get_site_basis(self, m):
                """Single site states."""
                bz = super_self.bw.brs.StateInfo()
                bz.allocate(2)
                bz.quanta[0] = super_self.bw.SX(1, 0)
                bz.quanta[1] = super_self.bw.SX(0, 1)
                bz.n_states[0] = bz.n_states[1] = 1
                bz.sort_states()
                return bz

            def init_site_ops(self):
                """Initialize operator quantum numbers at each site (site_op_infos)
                and primitive (single character) site operators (site_norm_ops)."""
                i_alloc = super_self.bw.b.IntVectorAllocator()
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                # site op infos
                max_n, max_s = 10, 10
                max_n_odd, max_s_odd = max_n | 1, max_s | 1
                max_n_even, max_s_even = max_n_odd ^ 1, max_s_odd ^ 1
                for m in range(self.n_sites):
                    qs = {self.vacuum}
                    for n in range(1, max_n_odd + 1, 2):
                        for s in range(1, max_s_odd + 1, 2):
                            qs.add(super_self.bw.SX(n, s))
                    for n in range(0, max_n_even + 1, 2):
                        for s in range(0, max_s_even + 1, 2):
                            qs.add(super_self.bw.SX(n, s))
                    for q in sorted(qs):
                        mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                        mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                        self.site_op_infos[m].append((q, mat))

                # prim ops
                for m in range(self.n_sites):
                    # ident
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(0, 0))
                    mat.allocate(info)
                    mat[info.find_state(super_self.bw.SX(1, 0, 1, 0))] = np.array([1.0])
                    mat[info.find_state(super_self.bw.SX(0, 1, 0, 1))] = np.array([1.0])
                    self.site_norm_ops[m][""] = mat

                    # G
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(1, 1))
                    mat.allocate(info)
                    mat[info.find_state(super_self.bw.SX(0, 1, 1, 0))] = np.array(
                        [-(2**0.5)]
                    )
                    mat[info.find_state(super_self.bw.SX(1, 0, 0, 1))] = np.array(
                        [-(2**0.5) if m % 2 == 0 else 2**0.5]
                    )
                    self.site_norm_ops[m]["G[1,1]"] = mat

            def get_site_string_op(self, m, expr):
                """Construct longer site operators from primitive ones."""
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                if expr in self.site_norm_ops[m]:
                    return self.site_norm_ops[m][expr]
                l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                a = self.get_site_string_op(m, expr[l.left_idx : l.mid_idx - 1])
                b = self.get_site_string_op(m, expr[l.mid_idx : l.right_idx - 1])
                idq = super_self.bw.b.SpinRecoupling.get_quanta(expr, l, False)
                r = super_self.bw.bs.SparseMatrix(d_alloc)
                r.allocate(self.find_site_op_info(m, super_self.bw.SX(*idq)))
                self.opf.product(0, a, b, r)
                self.site_norm_ops[m][expr] = r
                return r

            def init_string_quanta(self, exprs, term_l, left_vacuum):
                rr = super_self.bw.VectorVectorSX()
                for ix, expr in enumerate(exprs):
                    r = super_self.bw.VectorSX(
                        [super_self.bw.SX(0, 0)] * (term_l[ix] + 1)
                    )
                    l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                    gr = super_self.bw.b.SpinRecoupling.get_quanta(expr, l, False)
                    r[-1] = super_self.bw.SX(*gr)
                    lacc = 0
                    while True:  # (.+(.+(.+.)0)0)0
                        l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                        if l.right_idx == -1:
                            break
                        exprl, exprr = (
                            expr[0 : l.mid_idx - 1],
                            expr[l.mid_idx : l.right_idx - 1],
                        )
                        lr = super_self.bw.b.SpinRecoupling.get_level(exprr, 0)
                        gr = super_self.bw.b.SpinRecoupling.get_quanta(exprr, lr, False)
                        lacc += exprl.count("G")
                        r[lacc] = super_self.bw.SX(*gr)
                        expr = exprr
                    rr.append(r)
                return rr

            def get_string_quanta(self, ref, expr, idxs, k):
                """Quantum number for string operators (orbital dependent part)."""
                return ref[k], ref[-1] - ref[k]

            def get_string_quantum(self, expr, idxs):
                """Total quantum number for a string operator."""
                l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                g = super_self.bw.b.SpinRecoupling.get_quanta(expr, l, False)
                return super_self.bw.SX(*g)

            def deallocate(self):
                """Release memory."""
                for ops in self.site_norm_ops:
                    for p in ops.values():
                        p.deallocate()
                for infos in self.site_op_infos:
                    for _, p in infos:
                        p.deallocate()
                for bz in self.basis:
                    bz.deallocate()

        return SO4Hamiltonian(self.vacuum, self.n_sites, self.orb_sym)

    def get_phsu2_hamiltonian(self):
        assert SymmetryTypes.PHSU2 in self.symm_type
        GH = self.bw.bs.GeneralHamiltonian
        super_self = self
        import numpy as np

        class PHSU2Hamiltonian(GH):
            def __init__(self, vacuum, n_sites, orb_sym):
                GH.__init__(self)
                self.opf = super_self.bw.bs.OperatorFunctions(super_self.bw.brs.CG())
                self.vacuum = vacuum
                self.n_sites = n_sites
                self.orb_sym = orb_sym
                self.basis = super_self.bw.brs.VectorStateInfo(
                    [self.get_site_basis(m) for m in range(self.n_sites)]
                )
                self.site_op_infos = super_self.bw.brs.VectorVectorPLMatInfo(
                    [super_self.bw.brs.VectorPLMatInfo() for _ in range(self.n_sites)]
                )
                self.site_norm_ops = super_self.bw.bs.VectorMapStrSpMat(
                    [super_self.bw.bs.MapStrSpMat() for _ in range(self.n_sites)]
                )
                self.init_site_ops()

            def get_site_basis(self, m):
                """Single site states."""
                bz = super_self.bw.brs.StateInfo()
                bz.allocate(3)
                bz.quanta[0] = super_self.bw.SX(1, 0)
                bz.quanta[1] = super_self.bw.SX(0, -1)
                bz.quanta[2] = super_self.bw.SX(0, 1)
                bz.n_states[0] = bz.n_states[1] = bz.n_states[2] = 1
                bz.sort_states()
                return bz

            def init_site_ops(self):
                """Initialize operator quantum numbers at each site (site_op_infos)
                and primitive (single character) site operators (site_norm_ops)."""
                i_alloc = super_self.bw.b.IntVectorAllocator()
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                # site op infos
                max_n, max_s = 10, 10
                max_n_odd, max_s_odd = max_n | 1, max_s | 1
                max_n_even, max_s_even = max_n_odd ^ 1, max_s_odd ^ 1
                for m in range(self.n_sites):
                    qs = {self.vacuum}
                    for n in range(1, max_n_odd + 1, 2):
                        for s in range(-max_s_odd, max_s_odd + 1, 2):
                            qs.add(super_self.bw.SX(n, s))
                    for n in range(0, max_n_even + 1, 2):
                        for s in range(-max_s_even, max_s_even + 1, 2):
                            qs.add(super_self.bw.SX(n, s))
                    for q in sorted(qs):
                        mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                        mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                        self.site_op_infos[m].append((q, mat))

                # prim ops
                for m in range(self.n_sites):
                    # ident
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(0, 0))
                    mat.allocate(info)
                    mat[info.find_state(super_self.bw.SX(1, 0, 1))] = np.array([1.0])
                    mat[info.find_state(super_self.bw.SX(0, -1, 0))] = np.array([1.0])
                    mat[info.find_state(super_self.bw.SX(0, 1, 0))] = np.array([1.0])
                    self.site_norm_ops[m][""] = mat

                    # E
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(1, 1))
                    mat.allocate(info)
                    # n=1, s=0 -> n=0, s=1 // n=0, s=-1 -> n=1, s=0
                    mat[info.find_state(super_self.bw.SX(0, 0, 1))] = np.array(
                        [-(2**0.5)]
                    )
                    mat[info.find_state(super_self.bw.SX(1, -1, 0))] = np.array(
                        [1.0 if m % 2 == 0 else -1.0]
                    )
                    self.site_norm_ops[m]["E"] = mat

                    # F
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(1, -1))
                    mat.allocate(info)
                    # n=1, s=0 -> n=0, s=-1 // n=0, s=1 -> n=1, s=0
                    mat[info.find_state(super_self.bw.SX(0, 0, 1))] = np.array(
                        [2**0.5 if m % 2 == 0 else -(2**0.5)]
                    )
                    mat[info.find_state(super_self.bw.SX(1, 1, 0))] = np.array([1.0])
                    self.site_norm_ops[m]["F"] = mat

            def get_site_string_op(self, m, expr):
                """Construct longer site operators from primitive ones."""
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                if expr in self.site_norm_ops[m]:
                    return self.site_norm_ops[m][expr]
                l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                a = self.get_site_string_op(m, expr[l.left_idx : l.mid_idx - 1])
                b = self.get_site_string_op(m, expr[l.mid_idx : l.right_idx - 1])
                idq = super_self.bw.b.SpinPermRecoupling.get_target_twos(expr)
                dq = a.info.delta_quantum + b.info.delta_quantum
                r = super_self.bw.bs.SparseMatrix(d_alloc)
                r.allocate(
                    self.find_site_op_info(m, super_self.bw.SX(idq, dq.values[0]))
                )
                self.opf.product(0, a, b, r)
                self.site_norm_ops[m][expr] = r
                return r

            def init_string_quanta(self, exprs, term_l, left_vacuum):
                rr = super_self.bw.VectorVectorSX()
                for ix, expr in enumerate(exprs):
                    r = super_self.bw.VectorSX(
                        [super_self.bw.SX(0, 0)] * (term_l[ix] + 1)
                    )
                    gr = super_self.bw.b.SpinPermRecoupling.get_target_twos(expr)
                    r[-1] = super_self.bw.SX(gr, expr.count("E") - expr.count("F"))
                    lacc, nacc = 0, 0
                    while True:  # (.+(.+(.+.)0)0)0
                        l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                        if l.right_idx == -1:
                            break
                        exprl, exprr = (
                            expr[0 : l.mid_idx - 1],
                            expr[l.mid_idx : l.right_idx - 1],
                        )
                        lacc += exprl.count("E") + exprl.count("F")
                        nacc += exprl.count("E") - exprl.count("F")
                        r[lacc] = super_self.bw.SX(
                            super_self.bw.b.SpinPermRecoupling.get_target_twos(exprr),
                            nacc,
                        )
                        expr = exprr
                    rr.append(r)
                return rr

            def get_string_quanta(self, ref, expr, idxs, k):
                """Quantum number for string operators (orbital dependent part)."""
                return ref[k], ref[-1] - ref[k]

            def get_string_quantum(self, expr, idxs):
                """Total quantum number for a string operator."""
                idq = super_self.bw.b.SpinPermRecoupling.get_target_twos(expr)
                return super_self.bw.SX(idq, expr.count("E") - expr.count("F"))

            def deallocate(self):
                """Release memory."""
                for ops in self.site_norm_ops:
                    for p in ops.values():
                        p.deallocate()
                for infos in self.site_op_infos:
                    for _, p in infos:
                        p.deallocate()
                for bz in self.basis:
                    bz.deallocate()

        return PHSU2Hamiltonian(self.vacuum, self.n_sites, self.orb_sym)

    def get_so3_hamiltonian(self):
        assert SymmetryTypes.SO3 in self.symm_type
        GH = self.bw.bs.GeneralHamiltonian
        super_self = self
        import numpy as np

        class SO3Hamiltonian(GH):
            def __init__(self, vacuum, n_sites, orb_sym):
                GH.__init__(self)
                self.opf = super_self.bw.bs.OperatorFunctions(super_self.bw.brs.CG())
                self.vacuum = vacuum
                self.n_sites = n_sites
                self.orb_sym = orb_sym
                self.basis = super_self.bw.brs.VectorStateInfo(
                    [self.get_site_basis(m) for m in range(self.n_sites)]
                )
                self.site_op_infos = super_self.bw.brs.VectorVectorPLMatInfo(
                    [super_self.bw.brs.VectorPLMatInfo() for _ in range(self.n_sites)]
                )
                self.site_norm_ops = super_self.bw.bs.VectorMapStrSpMat(
                    [super_self.bw.bs.MapStrSpMat() for _ in range(self.n_sites)]
                )
                self.init_site_ops()

            def get_site_basis(self, m):
                """Single site states."""
                bz = super_self.bw.brs.StateInfo()
                if self.orb_sym[m] == 0:
                    bz.allocate(2)
                    bz.quanta[0] = super_self.bw.SX(0, 0, 0)
                    bz.quanta[1] = super_self.bw.SX(1, 0, 0)
                    bz.n_states[0] = bz.n_states[1] = 1
                elif self.orb_sym[m] == 1:
                    bz.allocate(4)
                    bz.quanta[0] = super_self.bw.SX(0, 0, 0)
                    bz.quanta[1] = super_self.bw.SX(1, 2, 0)
                    bz.quanta[2] = super_self.bw.SX(2, 2, 0)
                    bz.quanta[3] = super_self.bw.SX(3, 0, 0)
                    bz.n_states[0] = bz.n_states[1] = 1
                    bz.n_states[2] = bz.n_states[3] = 1
                elif self.orb_sym[m] == 2:
                    bz.allocate(8)
                    bz.quanta[0] = super_self.bw.SX(0, 0, 0)
                    bz.quanta[1] = super_self.bw.SX(1, 4, 0)
                    bz.quanta[2] = super_self.bw.SX(2, 2, 0)
                    bz.quanta[3] = super_self.bw.SX(2, 6, 0)
                    bz.quanta[4] = super_self.bw.SX(3, 2, 0)
                    bz.quanta[5] = super_self.bw.SX(3, 6, 0)
                    bz.quanta[6] = super_self.bw.SX(4, 4, 0)
                    bz.quanta[7] = super_self.bw.SX(5, 0, 0)
                    bz.n_states[0] = bz.n_states[1] = 1
                    bz.n_states[2] = bz.n_states[3] = 1
                    bz.n_states[4] = bz.n_states[5] = 1
                    bz.n_states[6] = bz.n_states[7] = 1
                else:
                    return NotImplemented
                bz.sort_states()
                return bz

            def init_site_ops(self):
                """Initialize operator quantum numbers at each site (site_op_infos)
                and primitive (single character) site operators (site_norm_ops)."""
                i_alloc = super_self.bw.b.IntVectorAllocator()
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                # site op infos
                max_n, max_s = 20, 20
                for m in range(self.n_sites):
                    qs = {self.vacuum}
                    for n in range(-max_n, max_n + 1, 1):
                        for s in range(0, max_s + 1, 2):
                            qs.add(super_self.bw.SX(n, s, 0))
                    for q in sorted(qs):
                        mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                        mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                        self.site_op_infos[m].append((q, mat))

                # prim ops
                for m in range(self.n_sites):
                    l = self.orb_sym[m] * 2

                    # ident
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(0, 0, 0))
                    mat.allocate(info)
                    if l == 0:
                        mat[info.find_state(super_self.bw.SX(0, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[info.find_state(super_self.bw.SX(1, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                    elif l == 2:
                        mat[info.find_state(super_self.bw.SX(0, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[info.find_state(super_self.bw.SX(1, 2, 2, 0))] = np.array(
                            [1.0]
                        )
                        mat[info.find_state(super_self.bw.SX(2, 2, 2, 0))] = np.array(
                            [1.0]
                        )
                        mat[info.find_state(super_self.bw.SX(3, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                    else:
                        raise NotImplementedError()
                    self.site_norm_ops[m][""] = mat

                    # C
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(1, l, 0))
                    mat.allocate(info)
                    if l == 0:
                        mat[info.find_state(super_self.bw.SX(0, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                    elif l == 2:
                        mat[info.find_state(super_self.bw.SX(0, l, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[info.find_state(super_self.bw.SX(1, l, l, 0))] = np.array(
                            [2**0.5]
                        )
                        mat[info.find_state(super_self.bw.SX(2, 0, l, 0))] = np.array(
                            [3**0.5]
                        )
                    else:
                        raise NotImplementedError()
                    self.site_norm_ops[m]["C%d" % l] = mat

                    # D
                    mat = super_self.bw.bs.SparseMatrix(d_alloc)
                    info = self.find_site_op_info(m, super_self.bw.SX(-1, l, 0))
                    mat.allocate(info)
                    if l == 0:
                        mat[info.find_state(super_self.bw.SX(1, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                    elif l == 2:
                        mat[info.find_state(super_self.bw.SX(1, 0, l, 0))] = np.array(
                            [3**0.5]
                        )
                        mat[info.find_state(super_self.bw.SX(2, l, l, 0))] = np.array(
                            [2**0.5]
                        )
                        mat[info.find_state(super_self.bw.SX(3, l, 0, 0))] = np.array(
                            [1.0]
                        )
                    else:
                        raise NotImplementedError()
                    self.site_norm_ops[m]["D%d" % l] = mat

            def get_site_string_op(self, m, expr):
                """Construct longer site operators from primitive ones."""
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                if expr in self.site_norm_ops[m]:
                    return self.site_norm_ops[m][expr]
                l = super_self.bw.b.GeneralSymmTensor.get_level(expr, 0)
                a = self.get_site_string_op(m, expr[l.left_idx : l.mid_idx - 1])
                b = self.get_site_string_op(m, expr[l.mid_idx : l.right_idx - 1])
                idq = super_self.bw.b.GeneralSymmTensor.get_quanta(expr, l)
                r = super_self.bw.bs.SparseMatrix(d_alloc)
                dq = super_self.bw.SX(
                    expr.count("C") - expr.count("D"), idq[0] if len(idq) != 0 else 0, 0
                )
                r.allocate(self.find_site_op_info(m, dq))
                self.opf.product(0, a, b, r)
                self.site_norm_ops[m][expr] = r
                return r

            def init_string_quanta(self, exprs, term_l, left_vacuum):
                rr = super_self.bw.VectorVectorSX()
                for ix, expr in enumerate(exprs):
                    r = super_self.bw.VectorSX(
                        [super_self.bw.SX(0, 0, 0)] * (term_l[ix] + 1)
                    )
                    l = super_self.bw.b.GeneralSymmTensor.get_level(expr, 0)
                    gr = super_self.bw.b.GeneralSymmTensor.get_quanta(expr, l)
                    r[-1] = super_self.bw.SX(
                        expr.count("C") - expr.count("D"),
                        gr[0] if len(gr) != 0 else 0,
                        0,
                    )
                    lacc, nacc = 0, 0
                    while True:  # (.+(.+(.+.)0)0)0
                        l = super_self.bw.b.GeneralSymmTensor.get_level(expr, 0)
                        if l.right_idx == -1:
                            break
                        exprl, exprr = (
                            expr[0 : l.mid_idx - 1],
                            expr[l.mid_idx : l.right_idx - 1],
                        )
                        lacc += exprl.count("C") + exprl.count("D")
                        nacc += exprl.count("C") - exprl.count("D")
                        lr = super_self.bw.b.GeneralSymmTensor.get_level(exprr, 0)
                        gr = super_self.bw.b.GeneralSymmTensor.get_quanta(exprr, lr)
                        r[lacc] = super_self.bw.SX(
                            nacc, gr[0] if len(gr) != 0 else 0, 0
                        )
                        expr = exprr
                    rr.append(r)
                return rr

            def get_string_quanta(self, ref, expr, idxs, k):
                """Quantum number for string operators (orbital dependent part)."""
                return ref[k], ref[-1] - ref[k]

            def get_string_quantum(self, expr, idxs):
                """Total quantum number for a string operator."""
                l = super_self.bw.b.GeneralSymmTensor.get_level(expr, 0)
                g = super_self.bw.b.GeneralSymmTensor.get_quanta(expr, l)
                return super_self.bw.SX(
                    expr.count("C") - expr.count("D"), g[0] if len(g) != 0 else 0, 0
                )

            def deallocate(self):
                """Release memory."""
                for ops in self.site_norm_ops:
                    for p in ops.values():
                        p.deallocate()
                for infos in self.site_op_infos:
                    for _, p in infos:
                        p.deallocate()
                for bz in self.basis:
                    bz.deallocate()

        return SO3Hamiltonian(self.vacuum, self.n_sites, self.orb_sym)

    def get_lz_hamiltonian(self):
        assert SymmetryTypes.LZ in self.symm_type
        GH = self.bw.bs.GeneralHamiltonian
        super_self = self
        import numpy as np

        class LZHamiltonian(GH):
            def __init__(self, vacuum, n_sites, orb_sym):
                GH.__init__(self)
                self.opf = super_self.bw.bs.OperatorFunctions(super_self.bw.brs.CG())
                self.vacuum = vacuum
                self.n_sites = n_sites
                self.orb_sym = orb_sym
                self.basis = super_self.bw.brs.VectorStateInfo(
                    [self.get_site_basis(m) for m in range(self.n_sites)]
                )
                self.site_op_infos = super_self.bw.brs.VectorVectorPLMatInfo(
                    [super_self.bw.brs.VectorPLMatInfo() for _ in range(self.n_sites)]
                )
                self.site_norm_ops = super_self.bw.bs.VectorMapStrSpMat(
                    [super_self.bw.bs.MapStrSpMat() for _ in range(self.n_sites)]
                )
                self.init_site_ops()

            def get_site_basis(self, m):
                """Single site states."""
                bz = super_self.bw.brs.StateInfo()
                if SymmetryTypes.SU2 in super_self.symm_type:
                    bz.allocate(3)
                    bz.quanta[0] = super_self.bw.SX(0, 0, 0)
                    bz.quanta[1] = super_self.bw.SX(1, 1, self.orb_sym[m])
                    bz.quanta[2] = super_self.bw.SX(2, 0, 2 * self.orb_sym[m])
                    bz.n_states[0] = bz.n_states[1] = bz.n_states[2] = 1
                elif SymmetryTypes.SZ in super_self.symm_type:
                    bz.allocate(4)
                    bz.quanta[0] = super_self.bw.SX(0, 0, 0)
                    bz.quanta[1] = super_self.bw.SX(1, 1, self.orb_sym[m])
                    bz.quanta[2] = super_self.bw.SX(1, -1, self.orb_sym[m])
                    bz.quanta[3] = super_self.bw.SX(2, 0, 2 * self.orb_sym[m])
                    bz.n_states[0] = bz.n_states[1] = 1
                    bz.n_states[2] = bz.n_states[3] = 1
                else:
                    bz.allocate(2)
                    bz.quanta[0] = super_self.bw.SX(0, 0)
                    bz.quanta[1] = super_self.bw.SX(1, self.orb_sym[m])
                    bz.n_states[0] = bz.n_states[1] = 1
                bz.sort_states()
                return bz

            def init_site_ops(self):
                """Initialize operator quantum numbers at each site (site_op_infos)
                and primitive (single character) site operators (site_norm_ops)."""
                i_alloc = super_self.bw.b.IntVectorAllocator()
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                # site op infos
                max_n, max_s = 10, 10
                max_n_odd, max_s_odd = max_n | 1, max_s | 1
                max_n_even, max_s_even = max_n_odd ^ 1, max_s_odd ^ 1
                for m in range(self.n_sites):
                    qs = {self.vacuum}
                    if SymmetryTypes.SU2 in super_self.symm_type:
                        for n in range(-max_n_odd, max_n_odd + 1, 2):
                            for s in range(1, max_s_odd + 1, 2):
                                for nz in range(-max_n_odd, max_n_odd + 1, 2):
                                    qs.add(super_self.bw.SX(n, s, nz * self.orb_sym[m]))
                        for n in range(-max_n_even, max_n_even + 1, 2):
                            for s in range(0, max_s_even + 1, 2):
                                for nz in range(-max_n_even, max_n_even + 1, 2):
                                    qs.add(super_self.bw.SX(n, s, nz * self.orb_sym[m]))
                    elif SymmetryTypes.SZ in super_self.symm_type:
                        for n in range(-max_n_odd, max_n_odd + 1, 2):
                            for s in range(-max_s_odd, max_s_odd + 1, 2):
                                for nz in range(-max_n_odd, max_n_odd + 1, 2):
                                    qs.add(super_self.bw.SX(n, s, nz * self.orb_sym[m]))
                        for n in range(-max_n_even, max_n_even + 1, 2):
                            for s in range(-max_s_even, max_s_even + 1, 2):
                                for nz in range(-max_n_even, max_n_even + 1, 2):
                                    qs.add(super_self.bw.SX(n, s, nz * self.orb_sym[m]))
                    elif SymmetryTypes.SGF in super_self.symm_type:
                        for n in range(-max_n, max_n + 1, 1):
                            for nz in range(-max_n, max_n + 1, 1):
                                qs.add(super_self.bw.SX(n, nz * self.orb_sym[m]))
                    for q in sorted(qs):
                        mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                        mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                        self.site_op_infos[m].append((q, mat))

                # prim ops
                for m in range(self.n_sites):
                    if SymmetryTypes.SU2 in super_self.symm_type:
                        # ident
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(m, super_self.bw.SX(0, 0, 0))
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[
                            info.find_state(super_self.bw.SX(1, 1, 1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        mat[
                            info.find_state(
                                super_self.bw.SX(2, 0, 0, 2 * self.orb_sym[m])
                            )
                        ] = np.array([1.0])
                        self.site_norm_ops[m][""] = mat

                        # C
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(1, 1, self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 1, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[
                            info.find_state(super_self.bw.SX(1, 0, 1, self.orb_sym[m]))
                        ] = np.array([-(2**0.5)])
                        self.site_norm_ops[m]["C"] = mat

                        # D
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(-1, 1, -self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[
                            info.find_state(super_self.bw.SX(1, 0, 1, self.orb_sym[m]))
                        ] = np.array([2**0.5])
                        mat[
                            info.find_state(
                                super_self.bw.SX(2, 1, 0, 2 * self.orb_sym[m])
                            )
                        ] = np.array([1.0])
                        self.site_norm_ops[m]["D"] = mat
                    elif SymmetryTypes.SZ in super_self.symm_type:
                        # ident
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(m, super_self.bw.SX(0, 0, 0))
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[
                            info.find_state(super_self.bw.SX(1, 1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        mat[
                            info.find_state(super_self.bw.SX(1, -1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        mat[
                            info.find_state(super_self.bw.SX(2, 0, 2 * self.orb_sym[m]))
                        ] = np.array([1.0])
                        self.site_norm_ops[m][""] = mat

                        # c
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(1, 1, self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[
                            info.find_state(super_self.bw.SX(1, -1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        self.site_norm_ops[m]["c"] = mat

                        # d
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(-1, -1, -self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[
                            info.find_state(super_self.bw.SX(1, 1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        mat[
                            info.find_state(super_self.bw.SX(2, 0, 2 * self.orb_sym[m]))
                        ] = np.array([1.0])
                        self.site_norm_ops[m]["d"] = mat

                        # C
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(1, -1, self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0, 0))] = np.array(
                            [1.0]
                        )
                        mat[
                            info.find_state(super_self.bw.SX(1, 1, self.orb_sym[m]))
                        ] = np.array([-1.0])
                        self.site_norm_ops[m]["C"] = mat

                        # D
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(-1, 1, -self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[
                            info.find_state(super_self.bw.SX(1, -1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        mat[
                            info.find_state(super_self.bw.SX(2, 0, 2 * self.orb_sym[m]))
                        ] = np.array([-1.0])
                        self.site_norm_ops[m]["D"] = mat
                    elif SymmetryTypes.SGF in super_self.symm_type:
                        # ident
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(m, super_self.bw.SX(0, 0))
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0))] = np.array([1.0])
                        mat[
                            info.find_state(super_self.bw.SX(1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        self.site_norm_ops[m][""] = mat

                        # C
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(1, self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[info.find_state(super_self.bw.SX(0, 0))] = np.array([1.0])
                        self.site_norm_ops[m]["C"] = mat

                        # D
                        mat = super_self.bw.bs.SparseMatrix(d_alloc)
                        info = self.find_site_op_info(
                            m, super_self.bw.SX(-1, -self.orb_sym[m])
                        )
                        mat.allocate(info)
                        mat[
                            info.find_state(super_self.bw.SX(1, self.orb_sym[m]))
                        ] = np.array([1.0])
                        self.site_norm_ops[m]["D"] = mat

            def get_site_string_op(self, m, expr):
                """Construct longer site operators from primitive ones."""
                d_alloc = super_self.bw.b.DoubleVectorAllocator()
                if expr in self.site_norm_ops[m]:
                    return self.site_norm_ops[m][expr]
                if SymmetryTypes.SU2 in super_self.symm_type:
                    l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                    a = self.get_site_string_op(m, expr[l.left_idx : l.mid_idx - 1])
                    b = self.get_site_string_op(m, expr[l.mid_idx : l.right_idx - 1])
                    idq = super_self.bw.b.SpinPermRecoupling.get_target_twos(expr)
                    jdq = a.info.delta_quantum + b.info.delta_quantum
                    dq = super_self.bw.SX(jdq.values[0], idq, idq, jdq.values[3])
                else:
                    a = self.get_site_string_op(m, expr[:1])
                    b = self.get_site_string_op(m, expr[1:])
                    dq = a.info.delta_quantum + b.info.delta_quantum
                r = super_self.bw.bs.SparseMatrix(d_alloc)
                r.allocate(self.find_site_op_info(m, dq))
                self.opf.product(0, a, b, r)
                self.site_norm_ops[m][expr] = r
                return r

            def init_string_quanta(self, exprs, term_l, left_vacuum):
                if SymmetryTypes.SU2 in super_self.symm_type:
                    rr = super_self.bw.VectorVectorSX()
                    for ix, expr in enumerate(exprs):
                        r = super_self.bw.VectorSX(
                            [super_self.bw.SX(0, 0, 0)] * (term_l[ix] + 1)
                        )
                        gr = super_self.bw.b.SpinPermRecoupling.get_target_twos(expr)
                        r[-1] = super_self.bw.SX(
                            expr.count("C") - expr.count("D"), gr, 0
                        )
                        lacc, nacc = 0, 0
                        while True:  # (.+(.+(.+.)0)0)0
                            l = super_self.bw.b.SpinRecoupling.get_level(expr, 0)
                            if l.right_idx == -1:
                                break
                            exprl, exprr = (
                                expr[0 : l.mid_idx - 1],
                                expr[l.mid_idx : l.right_idx - 1],
                            )
                            lacc += exprl.count("C") + exprl.count("D")
                            nacc += exprl.count("C") - exprl.count("D")
                            r[lacc] = super_self.bw.SX(
                                nacc,
                                super_self.bw.b.SpinPermRecoupling.get_target_twos(
                                    exprr
                                ),
                                0,
                            )
                            expr = exprr
                        rr.append(r)
                    return rr
                elif SymmetryTypes.SZ in super_self.symm_type:
                    qs = {
                        "": super_self.bw.SX(0, 0, 0),
                        "c": super_self.bw.SX(1, 1, 0),
                        "d": super_self.bw.SX(-1, -1, 0),
                        "C": super_self.bw.SX(1, -1, 0),
                        "D": super_self.bw.SX(-1, 1, 0),
                    }
                else:
                    qs = {
                        "": super_self.bw.SX(0, 0),
                        "C": super_self.bw.SX(1, 0),
                        "D": super_self.bw.SX(-1, 0),
                    }
                from itertools import accumulate

                return super_self.bw.VectorVectorSX(
                    [
                        super_self.bw.VectorSX(
                            list(
                                accumulate(
                                    [qs[""]] + [qs[x] for x in expr], lambda x, y: x + y
                                )
                            )
                        )
                        for expr in exprs
                    ]
                )

            def get_string_quanta(self, ref, expr, idxs, k):
                """Quantum number for string operators (orbital dependent part)."""
                l, r = ref[k], ref[-1] - ref[k]
                iv = (
                    1
                    + (SymmetryTypes.SZ in super_self.symm_type)
                    + 2 * (SymmetryTypes.SU2 in super_self.symm_type)
                )
                j = 0
                for ixx, ex in enumerate(expr):
                    if ex not in "cdCD":
                        continue
                    ipg = self.orb_sym[idxs[j]]
                    if j < k:
                        l.values[iv] += ipg if ex in "cC" else -ipg
                    else:
                        r.values[iv] += ipg if ex in "cC" else -ipg
                    j += 1
                return l, r

            def get_string_quantum(self, expr, idxs):
                """Total quantum number for a string operator."""
                if SymmetryTypes.SU2 in super_self.symm_type:
                    idq = super_self.bw.b.SpinPermRecoupling.get_target_twos(expr)
                    qs = lambda ix: {
                        "": super_self.bw.SX(0, 0, 0),
                        "C": super_self.bw.SX(1, 1, self.orb_sym[ix]),
                        "D": super_self.bw.SX(-1, 1, -self.orb_sym[ix]),
                    }
                    r = qs(0)[""]
                    j = 0
                    for ixx, ex in enumerate(expr):
                        if ex in "CD":
                            r = r + qs(idxs[j])[ex]
                            j += 1
                    r.values[1] = r.values[2] = idq
                    return r
                elif SymmetryTypes.SZ in super_self.symm_type:
                    qs = lambda ix: {
                        "": super_self.bw.SX(0, 0, 0),
                        "c": super_self.bw.SX(1, 1, self.orb_sym[ix]),
                        "d": super_self.bw.SX(-1, -1, -self.orb_sym[ix]),
                        "C": super_self.bw.SX(1, -1, self.orb_sym[ix]),
                        "D": super_self.bw.SX(-1, 1, -self.orb_sym[ix]),
                    }
                else:
                    qs = lambda ix: {
                        "": super_self.bw.SX(0, 0),
                        "C": super_self.bw.SX(1, self.orb_sym[ix]),
                        "D": super_self.bw.SX(-1, -self.orb_sym[ix]),
                    }
                return sum([qs(0)[""]] + [qs(ix)[ex] for ex, ix in zip(expr, idxs)])

            def deallocate(self):
                """Release memory."""
                for ops in self.site_norm_ops:
                    for p in ops.values():
                        p.deallocate()
                for infos in self.site_op_infos:
                    for _, p in infos:
                        p.deallocate()
                for bz in self.basis:
                    bz.deallocate()

        return LZHamiltonian(self.vacuum, self.n_sites, self.orb_sym)

    def get_custom_hamiltonian(self, site_basis, site_ops, orb_dependent_ops="cdCD"):
        GH = self.bw.bs.GeneralHamiltonian
        super_self = self
        import numpy as np
        from itertools import accumulate

        if (
            SymmetryTypes.SZ in super_self.symm_type
            or SymmetryTypes.SAny in super_self.symm_type
        ):

            class CustomHamiltonian(GH):
                def __init__(self, vacuum, n_sites, orb_sym):
                    GH.__init__(self)
                    self.opf = super_self.bw.bs.OperatorFunctions(
                        super_self.bw.brs.CG()
                    )
                    self.vacuum = vacuum
                    self.n_sites = n_sites
                    self.orb_sym = orb_sym
                    self.basis = super_self.bw.brs.VectorStateInfo(
                        [self.get_site_basis(m) for m in range(self.n_sites)]
                    )
                    self.site_op_infos = super_self.bw.brs.VectorVectorPLMatInfo(
                        [
                            super_self.bw.brs.VectorPLMatInfo()
                            for _ in range(self.n_sites)
                        ]
                    )
                    self.site_norm_ops = super_self.bw.bs.VectorMapStrSpMat(
                        [super_self.bw.bs.MapStrSpMat() for _ in range(self.n_sites)]
                    )
                    self.init_site_ops()

                def get_site_basis(self, m):
                    """Single site states."""
                    assert len(site_basis) == self.n_sites
                    bz = super_self.bw.brs.StateInfo()
                    bz.allocate(len(site_basis[m]))
                    for ix, (k, v) in enumerate(site_basis[m]):
                        bz.quanta[ix] = k
                        bz.n_states[ix] = v
                    bz.sort_states()
                    return bz

                def init_site_ops(self):
                    """Initialize operator quantum numbers at each site (site_op_infos)
                    and primitive (single character) site operators (site_norm_ops)."""
                    i_alloc = super_self.bw.b.IntVectorAllocator()
                    d_alloc = super_self.bw.b.DoubleVectorAllocator()
                    # site op infos
                    if SymmetryTypes.SZ in super_self.symm_type:
                        max_n, max_s = 10, 10
                        max_n_odd, max_s_odd = max_n | 1, max_s | 1
                        max_n_even, max_s_even = max_n_odd ^ 1, max_s_odd ^ 1
                        for m in range(self.n_sites):
                            qs = {self.vacuum}
                            for n in range(-max_n_odd, max_n_odd + 1, 2):
                                for s in range(-max_s_odd, max_s_odd + 1, 2):
                                    qs.add(super_self.bw.SX(n, s, self.orb_sym[m]))
                            for n in range(-max_n_even, max_n_even + 1, 2):
                                for s in range(-max_s_even, max_s_even + 1, 2):
                                    qs.add(super_self.bw.SX(n, s, 0))
                            for q in sorted(qs):
                                mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                                mat.initialize(
                                    self.basis[m], self.basis[m], q, q.is_fermion
                                )
                                self.site_op_infos[m].append((q, mat))
                    else:
                        for m in range(self.n_sites):
                            qs = {self.vacuum}
                            for iter in range(20):
                                new_qs = set()
                                for q in qs:
                                    for k, _ in site_basis[m]:
                                        new_q = q + k
                                        for iq in range(new_q.count):
                                            new_qs.add(new_q[iq])
                                        new_q = q - k
                                        for iq in range(new_q.count):
                                            new_qs.add(new_q[iq])
                                qs = new_qs
                            for q in sorted(qs):
                                mat = super_self.bw.brs.SparseMatrixInfo(i_alloc)
                                mat.initialize(
                                    self.basis[m], self.basis[m], q, q.is_fermion
                                )
                                self.site_op_infos[m].append((q, mat))

                    assert len(site_ops) == self.n_sites

                    # prim ops
                    for m, ops in enumerate(site_ops):
                        assert "" in ops  # must have identity

                        q_map = {}
                        pv = 0
                        for ix, (k, v) in enumerate(site_basis[m]):
                            for iv in range(v):
                                q_map[pv + iv] = (iv, k, pv + v)
                            pv += v

                        for name, op in ops.items():
                            assert op.shape == (pv, pv)
                            blocks = []
                            dq = site_basis[m][0][0]
                            for i in range(op.shape[0]):
                                if q_map[i][0] != 0:
                                    continue
                                for j in range(op.shape[1]):
                                    if q_map[j][0] != 0:
                                        continue
                                    mat = np.array(
                                        op[i : q_map[i][2], j : q_map[j][2]], copy=True
                                    )
                                    if np.linalg.norm(mat) >= 1e-20:
                                        dq = q_map[i][1] - q_map[j][1]
                                        blocks.append((q_map[j][1], mat))

                            mat = super_self.bw.bs.SparseMatrix(d_alloc)
                            info = self.find_site_op_info(m, dq)
                            assert info is not None
                            mat.allocate(info)
                            for q, mx in blocks:
                                mat[info.find_state(q)] = mx
                            self.site_norm_ops[m][name] = mat

                def get_site_string_ops(self, m, ops):
                    """Construct longer site operators from primitive ones."""
                    d_alloc = super_self.bw.b.DoubleVectorAllocator()
                    for k in ops:
                        if k in self.site_norm_ops[m]:
                            ops[k] = self.site_norm_ops[m][k]
                        else:
                            xx = self.site_norm_ops[m][k[0]]
                            for p in k[1:]:
                                xp = self.site_norm_ops[m][p]
                                q = xx.info.delta_quantum + xp.info.delta_quantum
                                mat = super_self.bw.bs.SparseMatrix(d_alloc)
                                mat.allocate(self.find_site_op_info(m, q))
                                self.opf.product(0, xx, xp, mat)
                                xx = mat
                            ops[k] = self.site_norm_ops[m][k] = xx
                    return ops

                def init_string_quanta(self, exprs, term_l, left_vacuum):
                    """Quantum number for string operators (orbital independent part)."""
                    qs = {}
                    for norm_ops in self.site_norm_ops:
                        for k, v in norm_ops.items():
                            if k not in qs:
                                qs[k] = v.info.delta_quantum
                                qs[k].pg = 0
                    return super_self.bw.VectorVectorSX(
                        [
                            super_self.bw.VectorSX(
                                list(
                                    accumulate(
                                        [qs[""]] + [qs[x] for x in expr],
                                        lambda x, y: x + y,
                                    )
                                )
                            )
                            for expr in exprs
                        ]
                    )

                def get_string_quanta(self, ref, expr, idxs, k):
                    """Quantum number for string operators (orbital dependent part)."""
                    l, r = ref[k], ref[-1] - ref[k]
                    for j, (ex, ix) in enumerate(zip(expr, idxs)):
                        ipg = self.orb_sym[ix]
                        if ex not in orb_dependent_ops:
                            pass
                        elif j < k:
                            l.pg = l.pg ^ ipg
                        else:
                            r.pg = r.pg ^ ipg
                    return l, r

                def get_string_quantum(self, expr, idxs):
                    """Total quantum number for a string operator."""
                    qs = lambda ix: {
                        k: v.info.delta_quantum
                        for k, v in self.site_norm_ops[ix].items()
                    }
                    return sum([qs(0)[""]] + [qs(ix)[ex] for ex, ix in zip(expr, idxs)])

                def deallocate(self):
                    """Release memory."""
                    for ops in self.site_norm_ops:
                        for p in ops.values():
                            p.deallocate()
                    for infos in self.site_op_infos:
                        for _, p in infos:
                            p.deallocate()
                    for bz in self.basis:
                        bz.deallocate()

            return CustomHamiltonian(self.vacuum, self.n_sites, self.orb_sym)
        else:
            return NotImplemented

    def write_fcidump(self, h1e, g2e, ecore=0, filename=None, h1e_symm=False, pg="d2h"):
        bw = self.bw
        import numpy as np

        fcidump = bw.bx.FCIDUMP()
        swap_pg = getattr(bw.b.PointGroup, "swap_" + pg)
        fw_map = np.array([swap_pg(x) for x in range(1, 9)])
        bk_map = np.argsort(fw_map) + 1
        if SymmetryTypes.SZ in bw.symm_type:
            if not h1e_symm:
                mh1e = tuple(x.ravel() for x in h1e)
            else:
                mh1e = tuple(x[np.tril_indices(len(x))] for x in h1e)
            mg2e = tuple(x.ravel() for x in g2e)
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
                mh1e = h1e.ravel()
            else:
                mh1e = h1e[np.tril_indices(len(h1e))]
            fcidump.initialize_su2(
                self.n_sites,
                self.target.n,
                self.target.twos,
                bk_map[self.target.pg],
                ecore,
                mh1e,
                g2e.ravel(),
            )
        else:
            fcidump.initialize_h1e(
                self.n_sites,
                self.target.n,
                self.target.twos,
                bk_map[self.target.pg],
                ecore,
                h1e.ravel(),
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
        self.orb_sym = bw.VectorPG(map(swap_pg, fcidump.orb_sym))
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
            self.orb_sym = bw.VectorPG(
                [self.orb_sym[i // 2] for i in range(self.n_sites)]
            )

    def integral_symmetrize(
        self, orb_sym, h1e=None, g2e=None, hxe=None, k_symm=None, iprint=1
    ):
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
                    if k_symm is None:
                        mask = (x[:, None] ^ x[None, :]) != 0
                    else:
                        mask = (x[:, None] - x[None, :]) != 0
                    error += sum(np.sum(np.abs(h[mask])) for h in h1e)
                    h1e[0][mask] = 0
                    h1e[1][mask] = 0
                else:
                    for i in range(len(h1e)):
                        if k_symm is None:
                            mask = (x[i][:, None] ^ x[i][None, :]) != 0
                        else:
                            mask = (x[i][:, None] - x[i][None, :]) != 0
                        error += np.sum(np.abs(h1e[i][mask]))
                        h1e[i][mask] = 0
            if g2e is not None:
                x = np.array(orb_sym, dtype=int)
                if x.ndim == 1:
                    if k_symm is None:
                        mask = (
                            x[:, None, None, None]
                            ^ x[None, :, None, None]
                            ^ x[None, None, :, None]
                            ^ x[None, None, None, :]
                        ) != 0
                    else:
                        mask = (
                            x[:, None, None, None]
                            - x[None, :, None, None]
                            + x[None, None, :, None]
                            - x[None, None, None, :]
                        ) != 0
                    error += sum(np.sum(np.abs(g[mask])) for g in g2e) * 0.5
                    error += np.sum(np.abs(g2e[1][mask])) * 0.5
                    g2e[0][mask] = 0
                    g2e[1][mask] = 0
                    g2e[2][mask] = 0
                else:
                    js = [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
                    for i in range(len(g2e)):
                        if k_symm is None:
                            mask = (
                                x[js[i][0]][:, None, None, None]
                                ^ x[js[i][1]][None, :, None, None]
                                ^ x[js[i][2]][None, None, :, None]
                                ^ x[js[i][3]][None, None, None, :]
                            ) != 0
                        else:
                            mask = (
                                x[js[i][0]][:, None, None, None]
                                - x[js[i][1]][None, :, None, None]
                                + x[js[i][2]][None, None, :, None]
                                - x[js[i][3]][None, None, None, :]
                            ) != 0
                        error += np.sum(np.abs(g2e[i][mask])) * 0.5
                        if i == 1:
                            error += np.sum(np.abs(g2e[i][mask])) * 0.5
                        g2e[i][mask] = 0
        else:
            if h1e is not None:
                x = np.array(orb_sym, dtype=int)
                if k_symm is None:
                    mask = (x[:, None] ^ x[None, :]) != 0
                else:
                    mask = (x[:, None] - x[None, :]) != 0
                error += np.sum(np.abs(h1e[mask]))
                h1e[mask] = 0
            if g2e is not None:
                x = np.array(orb_sym, dtype=int)
                if k_symm is None:
                    mask = (
                        x[:, None, None, None]
                        ^ x[None, :, None, None]
                        ^ x[None, None, :, None]
                        ^ x[None, None, None, :]
                    ) != 0
                else:
                    mask = (
                        x[:, None, None, None]
                        - x[None, :, None, None]
                        + x[None, None, :, None]
                        - x[None, None, None, :]
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

    def get_identity_mpo(self, ancilla=False):
        return self.get_mpo(
            self.expr_builder().add_term("", [], 1.0).finalize(), ancilla=ancilla
        )

    def unpack_g2e(self, g2e, n_sites=None):
        import numpy as np

        if n_sites is None:
            n_sites = self.n_sites

        if g2e.ndim == 1:
            m = n_sites * (n_sites + 1) // 2
            xtril = np.tril_indices(m)
            r = np.zeros((m**2,), dtype=g2e.dtype)
            r[xtril[0] * m + xtril[1]] = g2e
            r[xtril[1] * m + xtril[0]] = g2e
            g2e = r.reshape((m, m))

        if g2e.ndim == 2:
            m = n_sites
            xtril = np.tril_indices(m)
            r = np.zeros((m**2, m**2), dtype=g2e.dtype)
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
        ecore=0.0,
        para_type=None,
        reorder=None,
        cutoff=1e-20,
        integral_cutoff=1e-20,
        post_integral_cutoff=1e-20,
        fast_cutoff=1e-20,
        unpack_g2e=True,
        algo_type=None,
        normal_order_ref=None,
        normal_order_single_ref=None,
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
        add_ident=True,
        esptein_nesbet_partition=False,
        ancilla=False,
        reorder_imat=None,
        gaopt_opts=None,
        iprint=1,
    ):
        import numpy as np

        bw = self.bw

        if unpack_g2e:
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
            x_orb_sym = bw.VectorPG(self.orb_sym)
            if self.reorder_idx is not None:
                rev_idx = np.argsort(self.reorder_idx)
                x_orb_sym = bw.VectorPG(np.array(self.orb_sym)[rev_idx])
            k_symm = 0 if SymmetryTypes.LZ in bw.symm_type else None
            self.integral_symmetrize(
                x_orb_sym, h1e=h1e, g2e=g2e, k_symm=k_symm, iprint=iprint
            )

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
            prev_reord = self.reorder_idx is not None
            if isinstance(reorder, np.ndarray):
                idx = reorder
            elif reorder == "irrep":
                assert self.orb_sym is not None
                if self.reorder_idx is not None:
                    rev_idx = np.argsort(self.reorder_idx)
                    x_orb_sym = bw.VectorPG(np.array(self.orb_sym)[rev_idx])
                else:
                    x_orb_sym = self.orb_sym
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
                orb_opt = [optimal_reorder[x] for x in np.array(x_orb_sym)]
                idx = np.argsort(orb_opt)
            elif (reorder == "fiedler" or reorder == True) and reorder_imat is None:
                idx = self.orbital_reordering(h1e, g2e, method="fiedler")
            elif reorder == "gaopt" and reorder_imat is None:
                if gaopt_opts is None:
                    idx = self.orbital_reordering(h1e, g2e, method="gaopt")
                else:
                    idx = self.orbital_reordering(
                        h1e, g2e, method="gaopt", **gaopt_opts
                    )
            elif (reorder == "fiedler" or reorder == True) and reorder_imat is not None:
                idx = self.orbital_reordering_interaction_matrix(
                    reorder_imat, method="fiedler"
                )
            elif reorder == "gaopt" and reorder_imat is not None:
                if gaopt_opts is None:
                    idx = self.orbital_reordering_interaction_matrix(
                        reorder_imat, method="gaopt"
                    )
                else:
                    idx = self.orbital_reordering_interaction_matrix(
                        reorder_imat, method="gaopt", **gaopt_opts
                    )
            else:
                raise RuntimeError("Unknown reorder", reorder)
            if iprint:
                print("reordering = ", idx)
            self.reorder_idx = idx
            if SymmetryTypes.SZ in bw.symm_type:
                if h1e is not None:
                    h1e = [x[idx][:, idx] for x in h1e]
                if g2e is not None:
                    g2e = [x[idx][:, idx][:, :, idx][:, :, :, idx] for x in g2e]
            else:
                if h1e is not None:
                    h1e = h1e[idx][:, idx]
                if g2e is not None:
                    g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]
            if self.orb_sym is not None and not prev_reord:
                self.orb_sym = bw.VectorPG(np.array(self.orb_sym)[idx])
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

        if esptein_nesbet_partition:
            assert SymmetryTypes.SU2 in bw.symm_type
            xp = np.mgrid[: self.n_sites]
            xi, xj = xp[:, None], xp[None, :]
            if h1e is not None:
                h1e = h1e.copy()
                h1e[xi != xj] = 0
            xi, xj = xi[:, :, None, None], xj[:, :, None, None]
            xk, xl = xp[None, None, :, None], xp[None, None, None, :]
            if g2e is not None:
                g2e = g2e.copy()
                g2e[(~((xi == xj) & (xk == xl))) & (~((xj == xk) & (xi == xl)))] = 0

        if normal_order_ref is None:
            if SymmetryTypes.SU2 in bw.symm_type:
                if h1e is not None:
                    b.add_sum_term("(C+D)0", h1e, cutoff=fast_cutoff, factor=np.sqrt(2))
                if g2e is not None:
                    if not unpack_g2e and g2e.ndim == 1:
                        b.data.exprs.append("((C+(C+D)0)1+D)0")
                        b.data.add_eight_fold_term(g2e, cutoff=fast_cutoff, factor=1.0)
                    else:
                        b.add_sum_term(
                            "((C+(C+D)0)1+D)0",
                            g2e,
                            cutoff=fast_cutoff,
                            perm=[0, 2, 3, 1],
                        )
            elif SymmetryTypes.SZ in bw.symm_type:
                if h1e is not None:
                    b.add_sum_term("cd", h1e[0], cutoff=fast_cutoff)
                    b.add_sum_term("CD", h1e[1], cutoff=fast_cutoff)
                if g2e is not None:
                    b.add_sum_term(
                        "ccdd",
                        g2e[0],
                        cutoff=fast_cutoff,
                        perm=[0, 2, 3, 1],
                        factor=0.5,
                    )
                    b.add_sum_term(
                        "cCDd",
                        g2e[1],
                        cutoff=fast_cutoff,
                        perm=[0, 2, 3, 1],
                        factor=0.5,
                    )
                    b.add_sum_term(
                        "CcdD",
                        g2e[1],
                        cutoff=fast_cutoff,
                        perm=[2, 0, 1, 3],
                        factor=0.5,
                    )
                    b.add_sum_term(
                        "CCDD",
                        g2e[2],
                        cutoff=fast_cutoff,
                        perm=[0, 2, 3, 1],
                        factor=0.5,
                    )
            elif SymmetryTypes.SGF in bw.symm_type:
                if h1e is not None:
                    b.add_sum_term("CD", h1e, cutoff=fast_cutoff)
                if g2e is not None:
                    b.add_sum_term(
                        "CCDD", g2e, cutoff=fast_cutoff, perm=[0, 2, 3, 1], factor=0.5
                    )
            elif SymmetryTypes.SGB in bw.symm_type:
                h_terms = FermionTransform.jordan_wigner(h1e, g2e)
                for k, (x, v) in h_terms.items():
                    b.add_term(k, x, v)
        else:
            if SymmetryTypes.SU2 in bw.symm_type:
                if normal_order_single_ref is not None:
                    assert normal_order_wick
                    h1es, g2es, ecore = WickNormalOrder.make_su2_open_shell(
                        h1e, g2e, ecore, normal_order_ref, normal_order_single_ref
                    )
                else:
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
            add_ident=add_ident,
            ancilla=ancilla,
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
        add_ident=True,
        ancilla=False,
    ):
        bw = self.bw
        import time

        tt = time.perf_counter()
        if left_vacuum is None:
            left_vacuum = bw.SXT.invalid
        if algo_type is None:
            algo_type = MPOAlgorithmTypes.FastBipartite
        algo_type = getattr(bw.b.MPOAlgorithmTypes, algo_type.name)
        assert self.ghamil is not None
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

        if ancilla:
            mpo = bw.bs.AncillaMPO(mpo)
        mpo = bw.bs.SimplifiedMPO(mpo, bw.bs.Rule(), False, False)
        if add_ident:
            mpo = bw.bs.IdentityAddedMPO(mpo)
        if self.mpi:
            mpo = bw.bs.ParallelMPO(mpo, self.prule)
        return mpo

    def get_site_mpo(self, op, site_index, iprint=1):
        import numpy as np

        bw = self.bw
        b = self.expr_builder()

        if self.reorder_idx is not None:
            ridx = np.argsort(self.reorder_idx)
            site_index = ridx[site_index]

        mpo_lq = None
        if SymmetryTypes.SU2 in bw.symm_type:
            assert op in ["C", "D"]
            mpo_lq = bw.SX(-1, 1, 0) if op == "C" else bw.SX(1, 1, 0)
            b.add_term(op, [site_index], 2**0.5)
        elif SymmetryTypes.SZ in bw.symm_type:
            assert op in ["c", "d", "C", "D"]
            b.add_term(op, [site_index], 1.0)
        elif SymmetryTypes.SGF in bw.symm_type:
            assert op in ["C", "D"]
            b.add_term(op, [site_index], 1.0)

        if self.mpi is not None:
            b.iscale(1.0 / self.mpi.size)

        bx = b.finalize()
        return self.get_mpo(bx, iprint, left_vacuum=mpo_lq)

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
            ix1 = np.mgrid[: self.n_sites].ravel()
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
            ixa1 = np.mgrid[0 : self.n_sites : 2].ravel()
            ixb1 = np.mgrid[1 : self.n_sites : 2].ravel()
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

    def get_mpo_any_fermionic(self, op_list, ecore=None, **kwargs):
        """
        Args:
            op_list : list[tuple[str, float]]
                A list of second quantized fermionic operators and coefficients.
                + is creation, - is destroy.
                Example: [
                    ('+_3 +_4 -_1 -_3', 0.0068705380508780715),
                    ('+_3 +_4 -_4 -_3', -0.009852150878546906)
                ]
            ecore : core energy
        """
        from itertools import groupby

        assert SymmetryTypes.SGF in self.symm_type
        b = self.expr_builder()
        kmap = {"+": "C", "-": "D"}
        pattern = lambda x: "".join(p[0] for p in x[0].split())
        op_dict = {
            k: list(g) for k, g in groupby(sorted(op_list, key=pattern), key=pattern)
        }
        exprs = ["".join(kmap[x] for x in k) for k in op_dict.keys()]
        indices = [
            [int(x.split("_")[1]) for h, _ in g for x in h.split(" ")]
            for g in op_dict.values()
        ]
        values = [[v for _, v in g] for g in op_dict.values()]
        for (
            ex,
            ix,
            vl,
        ) in zip(exprs, indices, values):
            b.add_term(ex, ix, vl)
        if ecore is not None:
            b.add_const(ecore)
        return self.get_mpo(b.finalize(), **kwargs)

    def get_mpo_any_pauli(self, op_list, ecore=None, **kwargs):
        """
        Args:
            op_list : list[tuple[str, float]]
                A list of Pauli strings and coefficients.
                Characters in the string can be IXYZ.
                Example: [
                    ('IIXXXIIX', 0.0559742284070319),
                    ('IIIXIIXZ', 0.0018380565450674)
                ]
            ecore : core energy
        """
        assert SymmetryTypes.SGB in self.symm_type
        import numpy as np

        b = self.expr_builder()
        idxs = np.arange(self.n_sites, dtype=int)
        for ops, val in op_list:
            num_y = ops.count("Y")
            assert num_y % 2 == 0
            b.add_term(ops, idxs, val.real * (1 - num_y % 4))
        if ecore is not None:
            b.add_const(ecore)
        return self.get_mpo(b.finalize(adjust_order=False), **kwargs)

    def orbital_reordering(self, h1e, g2e, method="fiedler", **kwargs):
        bw = self.bw
        import numpy as np

        if np.array(h1e).ndim == 3:
            h1e = h1e[0] + h1e[1]
        if np.array(g2e).ndim == 5:
            g2e = g2e[0] + g2e[1] * 2 + g2e[2]

        xmat = np.abs(np.einsum("ijji->ij", g2e, optimize=True))
        kmat = np.abs(h1e) * 1e-7 + xmat
        kmat = bw.b.VectorDouble(kmat.ravel())
        idx = bw.b.OrbitalOrdering.fiedler(len(h1e), kmat)

        if method == "gaopt":
            opts = dict(
                n_generations=10000,
                n_configs=len(h1e) * 2,
                n_elite=8,
                clone_rate=0.1,
                mutate_rate=0.1,
            )
            n_tasks = kwargs.pop("n_tasks", 64)
            opts.update(kwargs)
            idxs = []
            for i_task in range(0, n_tasks):
                bw.b.Random.rand_seed(1234 + i_task)
                idx = bw.b.OrbitalOrdering.ga_opt(len(h1e), kmat, **opts)
                f = bw.b.OrbitalOrdering.evaluate(len(h1e), kmat, idx)
                idx = tuple(idx)
                idxs.append(idx)
            idx = sorted(list(set(idxs)))[0]

        return np.array(idx, dtype=int)

    def orbital_reordering_interaction_matrix(self, imat, method="fiedler", **kwargs):
        bw = self.bw
        import numpy as np

        kmat = bw.b.VectorDouble(imat.ravel())
        idx = bw.b.OrbitalOrdering.fiedler(len(imat), kmat)

        if method == "gaopt":
            opts = dict(
                n_generations=10000,
                n_configs=len(imat) * 2,
                n_elite=8,
                clone_rate=0.1,
                mutate_rate=0.1,
            )
            n_tasks = kwargs.pop("n_tasks", 64)
            opts.update(kwargs)
            idxs = []
            for i_task in range(0, n_tasks):
                bw.b.Random.rand_seed(1234 + i_task)
                idx = bw.b.OrbitalOrdering.ga_opt(len(imat), kmat, **opts)
                f = bw.b.OrbitalOrdering.evaluate(len(imat), kmat, idx)
                idx = tuple(idx)
                idxs.append(idx)
            idx = sorted(list(set(idxs)))[0]

        return np.array(idx, dtype=int)

    # return min_ket < ket | mpo | ket > / < ket | ket >
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
        twosite_to_onesite=None,
        dav_max_iter=4000,
        dav_def_max_size=50,
        proj_mpss=None,
        proj_weights=None,
        store_wfn_spectra=True,
        spectra_with_multiplicity=False,
        lowmem_noise=False,
        sweep_start=0,
        forward=None,
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
        if lowmem_noise:
            dmrg.noise_type = bw.b.NoiseTypes.ReducedPerturbativeCollectedLowMem
        else:
            dmrg.noise_type = bw.b.NoiseTypes.ReducedPerturbativeCollected
        dmrg.davidson_conv_thrds = bw.VectorFP(thrds)
        dmrg.davidson_max_iter = dav_max_iter + 100
        dmrg.davidson_soft_max_iter = dav_max_iter
        dmrg.davidson_def_max_size = dav_def_max_size
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
        if forward is None:
            forward = ket.center == 0
        if twosite_to_onesite is None:
            ener = dmrg.solve(n_sweeps, forward, tol, sweep_start)
        else:
            assert twosite_to_onesite < n_sweeps
            if sweep_start < twosite_to_onesite:
                ener = dmrg.solve(twosite_to_onesite, forward, 0, sweep_start)
                dmrg.me.dot = 1
                for ext_me in dmrg.ext_mes:
                    ext_me.dot = 1
            ener = dmrg.solve(n_sweeps, forward, tol, twosite_to_onesite)
            ket.dot = 1
            if self.mpi is not None:
                self.mpi.barrier()
            ket.save_data()
            if self.mpi is not None:
                self.mpi.barrier()

        if self.clean_scratch:
            dmrg.me.remove_partition_files()
            for me in dmrg.ext_mes:
                me.remove_partition_files()

        ket.info.bond_dim = max(ket.info.bond_dim, bond_dims[-1])
        if isinstance(ket, bw.bs.MultiMPS):
            ener = list(dmrg.energies[-1])
        if self.mpi is not None:
            self.mpi.barrier()
        if store_wfn_spectra:
            self._sweep_wfn_spectra = dmrg.sweep_wfn_spectra
        return ener

    # return exp(-target_t * mpo) @ ket
    def td_dmrg(
        self,
        mpo,
        ket,
        delta_t=None,
        target_t=None,
        final_mps_tag=None,
        te_type="rk4",
        n_steps=None,
        bond_dims=None,
        n_sub_sweeps=2,
        normalize_mps=False,
        hermitian=False,
        iprint=0,
        cutoff=1e-20,
        krylov_conv_thrd=5e-6,
        krylov_subspace_size=20,
    ):
        bw = self.bw
        import numpy as np

        if n_steps is None:
            n_steps = int(abs(target_t) / abs(delta_t) + 0.1)
        elif target_t is None:
            target_t = n_steps * delta_t
        elif delta_t is None:
            delta_t = target_t / n_steps

        assert np.abs(abs(n_steps * delta_t) - abs(target_t)) < 1e-10
        is_imag_te = abs(np.imag(delta_t)) < 1e-10

        if iprint:
            print(
                "Time Evolution  DELTA T = RE %15.8f + IM %15.8f"
                % (np.real(delta_t), np.imag(delta_t))
            )
            print(
                "Time Evolution TARGET T = RE %15.8f + IM %15.8f"
                % (np.real(target_t), np.imag(target_t))
            )
            print("Time Evolution   NSTEPS = %10d" % n_steps)

        mket = ket.deep_copy("TD-KET@TMP")

        me = bw.bs.MovingEnvironment(mpo, mket, mket, "TDDMRG")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments(iprint >= 2)

        if bond_dims is None:
            bond_dims = [mket.info.bond_dim]

        te = bw.bs.TimeEvolution(
            me,
            bw.b.VectorUBond(bond_dims),
            bw.b.TETypes.RK4 if te_type == "rk4" else bw.b.TETypes.TangentSpace,
        )
        te.hermitian = hermitian
        te.iprint = iprint
        te.n_sub_sweeps = 1
        if te.mode != bw.b.TETypes.TangentSpace:
            te.n_sub_sweeps = n_sub_sweeps
        te.normalize_mps = normalize_mps
        te.cutoff = cutoff
        te.krylov_conv_thrd = krylov_conv_thrd
        te.krylov_subspace_size = krylov_subspace_size

        te_times = []
        te_energies = []
        te_normsqs = []
        te_discarded_weights = []
        for i in range(n_steps):
            if te.mode == bw.b.TETypes.TangentSpace:
                te.solve(2, delta_t / 2, mket.center == 0)
            else:
                te.solve(1, delta_t, mket.center == 0)
            if is_imag_te and iprint:
                print(
                    "T = %10.5f <E> = %20.15f + %10.5fi <Norm^2> = %20.15f + %10.5fi"
                    % (
                        (i + 1) * delta_t,
                        np.real(te.energies[-1]),
                        np.imag(te.energies[-1]),
                        np.real(te.normsqs[-1]),
                        np.imag(te.normsqs[-1]),
                    )
                )
            elif iprint:
                print(
                    "T = %10.5f + %10.5fi <E> = %20.15f + %10.5fi <Norm^2> = %20.15f + %10.5fi"
                    % (
                        (i + 1) * np.real(delta_t),
                        (i + 1) * np.imag(delta_t),
                        np.real(te.energies[-1]),
                        np.imag(te.energies[-1]),
                        np.real(te.normsqs[-1]),
                        np.imag(te.normsqs[-1]),
                    )
                )
            te_times.append((i + 1) * delta_t)
            te_energies.append(te.energies[-1])
            te_normsqs.append(te.normsqs[-1])
            te_discarded_weights.append(te.discarded_weights[-1])

        from collections import namedtuple

        TEResult = namedtuple("TEResult", ["times", "energies", "normsqs", "dws"])
        self._te = TEResult(te_times, te_energies, te_normsqs, te_discarded_weights)

        return mket.deep_copy(
            "TD-" + ket.info.tag if final_mps_tag is None else final_mps_tag
        )

    def get_dmrg_results(self):
        import numpy as np

        energies = np.array(self._dmrg.energies)
        dws = np.array(self._dmrg.discarded_weights)
        bond_dims = np.array(self._dmrg.bond_dims)[: len(energies)]
        return bond_dims, dws, energies

    def get_bipartite_entanglement(self, ket=None):
        import numpy as np

        if ket is not None:
            self.expectation(ket, self.get_identity_mpo(), ket, store_ket_spectra=True)
        bip_ent = np.zeros(len(self._sweep_wfn_spectra), dtype=np.float64)
        for ix, x in enumerate(self._sweep_wfn_spectra):
            if hasattr(np, "float128"):
                ldsq = np.array(x, dtype=np.float128) ** 2
            else:
                ldsq = np.array(x, dtype=np.float64) ** 2
            ldsq = ldsq[ldsq != 0]
            bip_ent[ix] = float(np.sum(-ldsq * np.log(ldsq)))
        return bip_ent

    def get_n_orb_rdm_mpos(self, orb_type=1, ij_symm=True, iprint=0):
        bw = self.bw
        if SymmetryTypes.SU2 in bw.symm_type:
            return NotImplemented
        elif SymmetryTypes.SZ in bw.symm_type or SymmetryTypes.SGF in bw.symm_type:
            is_sgf = SymmetryTypes.SGF in bw.symm_type
            if orb_type == 1:
                h_terms = OrbitalEntropy.get_one_orb_rdm_h_terms(
                    self.n_sites, is_sgf=is_sgf
                )
            else:
                h_terms = OrbitalEntropy.get_two_orb_rdm_h_terms(
                    self.n_sites, ij_symm=ij_symm, is_sgf=is_sgf
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

    def get_orbital_entropies(
        self, ket, orb_type=1, ij_symm=True, use_npdm=True, iprint=0
    ):
        bw = self.bw
        import numpy as np

        if use_npdm:
            return self.get_orbital_entropies_use_npdm(ket, orb_type, iprint=iprint)

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
            elif SymmetryTypes.SGF in bw.symm_type:
                if orb_type == 1:
                    assert len(ld) == 2
                elif orb_type == 2:
                    ld = OrbitalEntropy.get_two_orb_rdm_eigvals(ld)
            else:
                return NotImplemented
            ld = ld[ld != 0]
            ent = float(np.sum(-ld * np.log(ld)).real)
            ents[ih] = ent
        if orb_type == 2 and ij_symm:
            for ih in mpos:
                ents[ih[::-1]] = ents[ih]
        return ents

    def get_orbital_entropies_use_npdm(self, ket, orb_type=1, iprint=0):
        bw = self.bw
        import numpy as np

        is_sgf = SymmetryTypes.SGF in bw.symm_type
        ents = np.zeros((self.n_sites,) * orb_type)
        mket = ket.deep_copy(ket.info.tag + "@ORB-ENT-TMP")
        if orb_type == 1:
            exprs, nx = OrbitalEntropy.get_one_orb_rdm_exprs(is_sgf=is_sgf)
        else:
            exprs, nx = OrbitalEntropy.get_two_orb_rdm_exprs(is_sgf=is_sgf)

        pdms = self.get_npdm(
            mket,
            pdm_type=[len(k) // 2 for (k, _), _ in exprs.items()],
            npdm_expr=[k for (k, _), _ in exprs.items()],
            mask=[list(m) for (_, m), _ in exprs.items()],
            iprint=iprint,
        )

        rrdms = np.zeros(
            ents.shape + (nx,),
            dtype=complex if SymmetryTypes.CPX in bw.symm_type else float,
        )
        for ((_, m), v), pdm in zip(exprs.items(), pdms):
            for ix, f in v:
                if orb_type == 1:
                    rrdms[..., ix] += pdm * f
                elif orb_type == 2:
                    if len(set(m)) == 0:
                        rrdms[..., ix] += pdm[None, None] * f
                    elif len(set(m)) == 1 and m[0] == 0:
                        rrdms[..., ix] += pdm[:, None] * f
                    elif len(set(m)) == 1 and m[0] == 1:
                        rrdms[..., ix] += pdm[None, :] * f
                    else:
                        rrdms[..., ix] += pdm * f

        if orb_type == 1:
            for i in range(self.n_sites):
                ld = np.array(rrdms[i])
                ld[np.abs(ld) < 1e-14] = 0
                ld = ld[ld != 0]
                ent = float(np.sum(-ld * np.log(ld)).real)
                ents[i] = ent
        elif orb_type == 2:
            for i in range(self.n_sites):
                for j in range(self.n_sites):
                    ld = np.array(rrdms[i, j])
                    ld = OrbitalEntropy.get_two_orb_rdm_eigvals(ld, diag_only=i == j)
                    ld[np.abs(ld) < 1e-14] = 0
                    ld = ld[ld != 0]
                    ent = float(np.sum(-ld * np.log(ld)).real)
                    ents[i, j] = ent
        # do not apply orbital reordering for entropies
        if self.reorder_idx is not None:
            idx = self.reorder_idx
            for i in range(orb_type):
                ents = ents[(slice(None),) * i + (idx,)]
        if self.mpi is not None:
            self.mpi.barrier()
        return ents

    def get_orbital_interaction_matrix(self, ket, use_npdm=True, iprint=0):
        import numpy as np

        s1 = self.get_orbital_entropies(
            ket, orb_type=1, use_npdm=use_npdm, iprint=iprint
        )
        s2 = self.get_orbital_entropies(
            ket, orb_type=2, use_npdm=use_npdm, iprint=iprint
        )
        return 0.5 * (s1[:, None] + s1[None, :] - s2) * (1 - np.identity(len(s1)))

    def get_conventional_npdm(
        self,
        ket,
        pdm_type=1,
        bra=None,
        soc=False,
        site_type=1,
        iprint=0,
        max_bond_dim=None,
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
            if max_bond_dim is not None:
                mps.info.bond_dim = max_bond_dim

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
                rev_idx = np.argsort(self.reorder_idx)
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
        npdm_expr=None,
        mask=None,
        simulated_parallel=0,
        fused_contraction_rotation=True,
        cutoff=1e-24,
        iprint=0,
        max_bond_dim=None,
    ):
        bw = self.bw
        import numpy as np

        if algo_type is None:
            algo_type = NPDMAlgorithmTypes.Default

        if NPDMAlgorithmTypes.Conventional in algo_type or soc:
            return self.get_conventional_npdm(
                ket, pdm_type, bra, soc, site_type, iprint, max_bond_dim
            )

        if self.mpi is not None:
            self.mpi.barrier()

        if SymmetryTypes.SU2 in bw.symm_type:
            if npdm_expr is not None and "%s" not in npdm_expr:
                op_str = npdm_expr
            else:
                su2_coupling = "((C+%s)1+D)0" if npdm_expr is None else npdm_expr
                op_str = "(C+D)0"
                for _ in range(pdm_type - 1):
                    op_str = su2_coupling % op_str
            if mask is None:
                perm = bw.b.SpinPermScheme.initialize_su2(pdm_type * 2, op_str, True)
            else:
                perm = bw.b.SpinPermScheme.initialize_su2(
                    pdm_type * 2, op_str, True, mask=bw.b.VectorUInt16(mask)
                )
            perms = bw.b.VectorSpinPermScheme([perm])
        elif SymmetryTypes.SZ in bw.symm_type:
            if npdm_expr is not None and isinstance(npdm_expr, str):
                op_str = [npdm_expr]
            elif npdm_expr is not None:
                op_str = npdm_expr
            else:
                op_str = ["cd", "CD"]
                for _ in range(pdm_type - 1):
                    op_str = ["c%sd" % x for x in op_str] + ["C%sD" % op_str[-1]]
            if mask is None:
                perms = bw.b.VectorSpinPermScheme(
                    [
                        bw.b.SpinPermScheme.initialize_sz(pdm_type * 2, cd, True)
                        for cd in op_str
                    ]
                )
            elif len(mask) != 0 and not isinstance(mask[0], int):
                assert len(mask) == len(op_str)
                pts = (
                    [pdm_type] * len(op_str) if isinstance(pdm_type, int) else pdm_type
                )
                perms = bw.b.VectorSpinPermScheme(
                    [
                        bw.b.SpinPermScheme.initialize_sz(
                            pt * 2, cd, True, mask=bw.b.VectorUInt16(xm)
                        )
                        for cd, xm, pt in zip(op_str, mask, pts)
                    ]
                )
            else:
                pts = (
                    [pdm_type] * len(op_str) if isinstance(pdm_type, int) else pdm_type
                )
                perms = bw.b.VectorSpinPermScheme(
                    [
                        bw.b.SpinPermScheme.initialize_sz(
                            pt * 2, cd, True, mask=bw.b.VectorUInt16(mask)
                        )
                        for cd, pt in zip(op_str, pts)
                    ]
                )
        elif SymmetryTypes.SGF in bw.symm_type:
            if npdm_expr is not None and isinstance(npdm_expr, str):
                op_str = [npdm_expr]
            elif npdm_expr is not None:
                op_str = npdm_expr
            else:
                op_str = ["C" * pdm_type + "D" * pdm_type]
            if mask is None:
                perms = bw.b.VectorSpinPermScheme(
                    [
                        bw.b.SpinPermScheme.initialize_sz(pdm_type * 2, cd, True)
                        for cd in op_str
                    ]
                )
            elif len(mask) != 0 and not isinstance(mask[0], int):
                assert len(mask) == len(op_str)
                pts = (
                    [pdm_type] * len(op_str) if isinstance(pdm_type, int) else pdm_type
                )
                perms = bw.b.VectorSpinPermScheme(
                    [
                        bw.b.SpinPermScheme.initialize_sz(
                            pt * 2, cd, True, mask=bw.b.VectorUInt16(xm)
                        )
                        for cd, xm, pt in zip(op_str, mask, pts)
                    ]
                )
            else:
                pts = (
                    [pdm_type] * len(op_str) if isinstance(pdm_type, int) else pdm_type
                )
                perms = bw.b.VectorSpinPermScheme(
                    [
                        bw.b.SpinPermScheme.initialize_sz(
                            pt * 2, cd, True, mask=bw.b.VectorUInt16(mask)
                        )
                        for cd, pt in zip(op_str, pts)
                    ]
                )

        if iprint >= 1:
            print("npdm string =", op_str)

        if iprint >= 3:
            for perm in perms:
                print(perm.to_str())

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

                mps.info.load_mutable()
                mps.info.bond_dim = max(
                    mps.info.bond_dim, mps.info.get_max_bond_dimension()
                )
                if max_bond_dim is not None:
                    mps.info.bond_dim = max_bond_dim
                mps.info.deallocate_mutable()

            self.align_mps_center(mbra, mket)

            scheme = bw.b.NPDMScheme(perms)
            opdq = (mbra.info.target - mket.info.target)[0]
            if iprint >= 4:
                print("NPDM dq =", opdq)
                print(scheme.to_str())
            pmpo = bw.bs.GeneralNPDMMPO(
                self.ghamil, scheme, NPDMAlgorithmTypes.SymbolFree in algo_type
            )
            pmpo.delta_quantum = opdq
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
                for i in range(npdms[ip].ndim):
                    npdms[ip] = npdms[ip][(slice(None),) * i + (rev_idx,)]

        if self.mpi is not None:
            self.mpi.barrier()

        if SymmetryTypes.SU2 in bw.symm_type:
            assert len(npdms) == 1
            npdms = npdms[0]
        elif SymmetryTypes.SGF in bw.symm_type and (
            npdm_expr is None or isinstance(npdm_expr, str)
        ):
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

    def get_csf_coefficients(
        self, ket, cutoff=0.1, given_dets=None, max_print=200, iprint=1
    ):
        bw = self.bw
        iprint = iprint >= 1 and (self.mpi is None or self.mpi.rank == self.mpi.root)
        import numpy as np, time

        if ket.center != 0:
            ket = self.copy_mps(ket, tag="CSF-TMP")
            self.align_mps_center(ket, ref=0)
            if iprint:
                print("mps center changed (temporarily)")

        tx = time.perf_counter()
        dtrie = bw.bs.DeterminantTRIE(ket.n_sites, True)
        ddstr = "0+-2" if SymmetryTypes.SU2 in bw.symm_type else "0ab2"
        if given_dets is not None:
            uniq = set()
            for det in given_dets:
                ddet = det
                if isinstance(ddet, str):
                    ddet = [ddstr.index(x) for x in ddet]
                if tuple(ddet) in uniq:
                    continue
                else:
                    uniq.add(tuple(ddet))
                dtrie.append(bw.b.VectorUInt8(ddet))
        dtrie.evaluate(bw.bs.UnfusedMPS(ket), cutoff)
        if iprint:
            print("dtrie finished", time.perf_counter() - tx)
        dname = "CSF" if SymmetryTypes.SU2 in bw.symm_type else "DET"
        if iprint:
            print("Number of %s = %10d (cutoff = %9.5g)" % (dname, len(dtrie), cutoff))
        dvals = np.array(dtrie.vals)
        gidx = np.argsort(np.abs(dvals))[::-1][:max_print]
        if iprint:
            print(
                "Sum of weights of included %s = %20.15f\n"
                % (dname, (dvals**2).sum())
            )
            for ii, idx in enumerate(gidx):
                if self.reorder_idx is not None:
                    rev_idx = np.argsort(self.reorder_idx)
                    det = "".join([ddstr[x] for x in np.array(dtrie[idx])[rev_idx]])
                else:
                    det = "".join([ddstr[x] for x in np.array(dtrie[idx])])
                val = dvals[idx]
                print(dname, "%10d" % ii, det, " = %20.15f" % val)
            if len(dvals) > max_print:
                print(" ... and more ... ")
        dets = np.zeros((len(dtrie), ket.n_sites), dtype=np.uint8)
        for i in range(len(dtrie)):
            if self.reorder_idx is not None:
                rev_idx = np.argsort(self.reorder_idx)
                dets[i] = np.array(dtrie[i])[rev_idx]
            else:
                dets[i] = np.array(dtrie[i])
        return dets, dvals

    def compress_mps(self, ket, max_bond_dim=None):
        refc = ket.n_sites - ket.dot if ket.center == 0 else 0
        self.align_mps_center(ket, refc, max_bond_dim=max_bond_dim)
        return ket

    def align_mps_center(self, ket, ref, max_bond_dim=None):
        if self.mpi is not None:
            self.mpi.barrier()
        refc = ref if isinstance(ref, int) else ref.center
        ket.info.bond_dim = max(ket.info.bond_dim, ket.info.get_max_bond_dimension())
        if max_bond_dim is not None:
            ket.info.bond_dim = max_bond_dim
        if ket.center != refc:
            if refc == 0:
                if ket.dot == 2:
                    ket.center += 1
                    if ket.canonical_form[-1] == "C":
                        ket.canonical_form = ket.canonical_form[:-1] + "S"
                    else:
                        ket.canonical_form = ket.canonical_form[:-1] + "T"
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

    # if restarting from the middle, this method should not be used
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

        if dot == 2 and ket.center == ket.n_sites - 1:
            if self.mpi is not None:
                self.mpi.barrier()
            if ket.canonical_form[ket.center] in "KJ":
                ket.flip_fused_form(ket.center, self.ghamil.opf.cg, self.prule)
            ket.center = ket.n_sites - 2
            ket.save_data()
            if self.mpi is not None:
                self.mpi.barrier()
            ket.load_mutable()
            ket.info.load_mutable()
            if self.mpi is not None:
                self.mpi.barrier()

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

    # bra = mpo @ ket
    # bra = (left_mpo)^(-1) @ (mpo @ ket)
    def multiply(
        self,
        bra,
        mpo,
        ket,
        n_sweeps=10,
        tol=1e-8,
        bond_dims=None,
        bra_bond_dims=None,
        noises=None,
        noise_mpo=None,
        thrds=None,
        left_mpo=None,
        cutoff=1e-24,
        linear_max_iter=4000,
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
        if thrds is None:
            if SymmetryTypes.SP not in bw.symm_type:
                thrds = [1e-6] * 4 + [1e-7] * 1
            else:
                thrds = [1e-5] * 4 + [5e-6] * 1
        if noises is not None and noises[0] != 0 and noise_mpo is not None:
            pme = bw.bs.MovingEnvironment(noise_mpo, bra, bra, "PERT-CPS")
            pme.init_environments(iprint >= 2)
        elif left_mpo is not None:
            pme = bw.bs.MovingEnvironment(left_mpo, bra, bra, "L-MULT")
            pme.init_environments(iprint >= 2)
        else:
            pme = None
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "MULT")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = pme is None  # not allowed by perturbative noise
        me.init_environments(iprint >= 2)
        if noises is None or noises[0] == 0:
            cps = bw.bs.Linear(
                pme, me, bw.b.VectorUBond(bra_bond_dims), bw.b.VectorUBond(bond_dims)
            )
        else:
            cps = bw.bs.Linear(
                pme,
                me,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(bond_dims),
                bw.VectorFP(noises),
            )
        if noises is not None and noises[0] != 0:
            cps.noise_type = bw.b.NoiseTypes.ReducedPerturbative
            cps.decomp_type = bw.b.DecompositionTypes.SVD
        if noises is not None and noises[0] != 0 and left_mpo is None:
            cps.eq_type = bw.b.EquationTypes.PerturbativeCompression
        cps.iprint = iprint
        cps.cutoff = cutoff
        cps.linear_conv_thrds = bw.VectorFP(thrds)
        cps.linear_max_iter = linear_max_iter + 100
        cps.linear_soft_max_iter = linear_max_iter
        norm = cps.solve(n_sweeps, ket.center == 0, tol)

        if self.clean_scratch:
            me.remove_partition_files()
            if pme is not None:
                pme.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return norm

    # bra = mpo_a @ ket_a + mpo_b @ ket_b
    def addition(
        self,
        bra,
        ket_a,
        ket_b,
        mpo_a=None,
        mpo_b=None,
        n_sweeps=10,
        tol=1e-8,
        bra_bond_dims=None,
        ket_a_bond_dims=None,
        ket_b_bond_dims=None,
        noises=None,
        noise_mpo=None,
        cutoff=1e-24,
        iprint=0,
    ):
        bw = self.bw
        if bra.info.tag == ket_a.info.tag or bra.info.tag == ket_b.info.tag:
            raise RuntimeError("Same tag for bra and ket!!")
        if ket_a_bond_dims is None:
            ket_a_bond_dims = [ket_a.info.bond_dim]
        if ket_b_bond_dims is None:
            ket_b_bond_dims = [ket_b.info.bond_dim]
        if bra_bond_dims is None:
            bra_bond_dims = [bra.info.bond_dim]
        self.align_mps_center(ket_b, ket_a)
        self.align_mps_center(bra, ket_a)
        if noises is not None and noises[0] != 0 and noise_mpo is not None:
            pme = bw.bs.MovingEnvironment(noise_mpo, bra, bra, "PERT-CPS")
            pme.init_environments(iprint >= 2)
        else:
            pme = None
        if mpo_a is None:
            mpo_a = self.get_identity_mpo()
        elif isinstance(mpo_a, (int, float, complex)):
            mpo_a = mpo_a * self.get_identity_mpo()
        if mpo_b is None:
            mpo_b = self.get_identity_mpo()
        elif isinstance(mpo_b, (int, float, complex)):
            mpo_b = mpo_b * self.get_identity_mpo()
        lme = bw.bs.MovingEnvironment(mpo_a, bra, ket_a, "ADD-L")
        lme.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        lme.init_environments(iprint >= 2)
        rme = bw.bs.MovingEnvironment(mpo_b, bra, ket_b, "ADD-R")
        rme.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        rme.init_environments(iprint >= 2)
        if noises is None or noises[0] == 0:
            cps = bw.bs.Linear(
                pme,
                lme,
                rme,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(ket_a_bond_dims),
            )
        else:
            cps = bw.bs.Linear(
                pme,
                lme,
                rme,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(ket_a_bond_dims),
                bw.VectorFP(noises),
            )
        cps.target_ket_bond_dim = ket_b_bond_dims[-1]
        if noises is not None and noises[0] != 0:
            cps.noise_type = bw.b.NoiseTypes.ReducedPerturbative
            cps.decomp_type = bw.b.DecompositionTypes.SVD
        cps.eq_type = bw.b.EquationTypes.FitAddition
        cps.iprint = iprint
        cps.cutoff = cutoff
        norm = cps.solve(n_sweeps, ket_a.center == 0, tol)

        if self.clean_scratch:
            lme.remove_partition_files()
            rme.remove_partition_files()
            if pme is not None:
                pme.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return norm

    # return < bra | mpo | ket >
    def expectation(
        self, bra, mpo, ket, store_bra_spectra=False, store_ket_spectra=False, iprint=0
    ):
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
        if bra == ket:
            store_bra_spectra = store_bra_spectra or store_ket_spectra
            store_ket_spectra = store_bra_spectra or store_ket_spectra
        expect.store_bra_spectra = store_bra_spectra
        expect.store_ket_spectra = store_ket_spectra
        expect.iprint = iprint
        if store_bra_spectra or store_ket_spectra:
            ex = expect.solve(True, mket.center == 0)
            self._sweep_wfn_spectra = expect.sweep_wfn_spectra
        else:
            ex = expect.solve(False, mket.center != 0)

        if self.clean_scratch:
            me.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return ex

    # bra = (mpo + omega + 1j * eta)^(-1) @ (rmpo @ ket)
    # return < bra | rmpo | ket >
    def greens_function(
        self,
        bra,
        mpo,
        rmpo,
        ket,
        omega,
        eta,
        n_sweeps=10,
        tol=1e-8,
        bra_bond_dims=None,
        ket_bond_dims=None,
        noises=None,
        thrds=None,
        cutoff=1e-24,
        linear_max_iter=4000,
        iprint=0,
    ):
        bw = self.bw
        if bra.info.tag == ket.info.tag:
            raise RuntimeError("Same tag for bra and ket!!")
        if ket_bond_dims is None:
            ket_bond_dims = [ket.info.bond_dim]
        if bra_bond_dims is None:
            bra_bond_dims = [bra.info.bond_dim]
        self.align_mps_center(bra, ket)
        if thrds is None:
            if SymmetryTypes.SP not in bw.symm_type:
                thrds = [1e-6] * 4 + [1e-7] * 1
            else:
                thrds = [1e-5] * 4 + [5e-6] * 1
        lme = bw.bs.MovingEnvironment(mpo, bra, bra, "LHS")
        lme.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        lme.init_environments(iprint >= 2)
        rme = bw.bs.MovingEnvironment(rmpo, bra, ket, "RHS")
        rme.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        rme.init_environments(iprint >= 2)

        if noises is None or noises[0] == 0:
            cps = bw.bs.Linear(
                lme,
                rme,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(ket_bond_dims),
            )
        else:
            cps = bw.bs.Linear(
                lme,
                rme,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(ket_bond_dims),
                bw.VectorFP(noises),
            )
        if noises is not None and noises[0] != 0:
            cps.noise_type = bw.b.NoiseTypes.ReducedPerturbative
            cps.decomp_type = bw.b.DecompositionTypes.SVD
        cps.iprint = iprint
        cps.cutoff = cutoff
        cps.eq_type = bw.b.EquationTypes.GreensFunction
        cps.linear_conv_thrds = bw.VectorFP(thrds)
        cps.linear_max_iter = linear_max_iter + 100
        cps.linear_soft_max_iter = linear_max_iter
        cps.gf_eta = eta
        cps.gf_omega = omega

        cps.solve(n_sweeps, ket.center == 0, tol)
        rgf, igf = cps.targets[-1]

        if self.clean_scratch:
            lme.remove_partition_files()
            rme.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return rgf + 1j * igf

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

    def copy_mps(self, mps, tag):
        ket = mps.deep_copy(tag)
        ket.info.save_data(self.scratch + "/%s-mps_info.bin" % ket.info.tag)
        return ket

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
            if (
                cp_mps.center == cp_mps.n_sites - 2
                and cp_mps.dot == 2
                and cp_mps.canonical_form[cp_mps.center] == "L"
            ):
                if cp_mps.canonical_form[-1] == "C":
                    cp_mps.canonical_form = cp_mps.canonical_form[:-1] + "S"
                else:
                    cp_mps.canonical_form = cp_mps.canonical_form[:-1] + "T"
                cp_mps.center = cp_mps.n_sites - 1
            cp_mps.move_left(self.ghamil.opf.cg, self.prule)
        if forward:
            if left_vacuum is None:
                left_vacuum = self.bw.SXT.invalid
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

    def mps_change_complex(self, mps, tag):
        bw = self.bw
        assert tag != mps.info.tag
        if SymmetryTypes.CPX in bw.symm_type:
            r = bw.bs.trans_mps_to_real(mps, tag)
        else:
            r = bw.bs.trans_mps_to_complex(mps, tag)
        r.info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        return r

    def mps_change_to_sz(self, mps, tag, sz=None):
        bw = self.bw
        assert SymmetryTypes.SU2 in bw.symm_type
        assert tag != mps.info.tag
        mps.info.load_mutable()
        mps.load_mutable()
        umps = bw.bs.trans_unfused_mps_to_sz(
            bw.bs.UnfusedMPS(mps), tag, self.ghamil.opf.cg
        )
        if sz is not None:
            umps.resolve_singlet_embedding(sz)
        zmps = umps.finalize()

        zmps.info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        zmps = self.adjust_mps(zmps, dot=mps.dot)[0]
        return zmps

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
        mrci_order=0,
        orig_dot=False,
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
            elif mrci_order == 0:
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
            else:
                mps_info = bw.brs.MRCIMPSInfo(
                    self.n_sites,
                    casci_ncore,
                    casci_nvirt,
                    mrci_order,
                    self.vacuum,
                    target,
                    self.ghamil.basis,
                )
            mps = bw.bs.MPS(self.n_sites, center, dot if orig_dot else 1)
        else:
            targets = bw.VectorSX([target]) if isinstance(target, bw.SXT) else target
            mps_info = bw.brs.MultiMPSInfo(
                self.n_sites, self.vacuum, targets, self.ghamil.basis
            )
            mps = bw.bs.MultiMPS(self.n_sites, center, dot if orig_dot else 1, nroots)
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
        if dot != 1 and not orig_dot:
            mps = self.adjust_mps(mps, dot=dot)[0]
        return mps

    def get_ancilla_mps(
        self,
        tag,
        center=0,
        dot=2,
        target=None,
        full_fci=True,
    ):
        bw = self.bw
        if target is None:
            target = bw.SX(self.n_sites * 2, 0, 0)
        mps_info = bw.brs.AncillaMPSInfo(
            self.n_sites, self.vacuum, target, self.ghamil.basis
        )
        mps = bw.bs.MPS(self.n_sites * 2, center, dot)
        mps_info.tag = tag
        if full_fci:
            mps_info.set_bond_dimension_full_fci(self.vacuum, self.vacuum)
        else:
            mps_info.set_bond_dimension_fci(self.vacuum, self.vacuum)
        mps_info.set_thermal_limit()
        mps_info.bond_dim = mps_info.get_max_bond_dimension()
        mps.initialize(mps_info)
        mps.fill_thermal_limit()
        mps.canonicalize()
        mps.save_mutable()
        mps_info.save_mutable()
        mps.save_data()
        mps_info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        return mps

    def get_mps_from_csf_coefficients(
        self, dets, dvals, tag, dot=2, target=None, full_fci=True, left_vacuum=None
    ):
        bw = self.bw
        assert self.reorder_idx is None

        dtrie = bw.bs.DeterminantTRIE(self.n_sites, True)
        ddstr = "0+-2" if SymmetryTypes.SU2 in bw.symm_type else "0ab2"
        map_dets = {}
        assert len(dets) == len(dvals)
        for it, det in enumerate(dets):
            ddet = det
            if isinstance(ddet, str):
                ddet = [ddstr.index(x) for x in ddet]
            if tuple(ddet) not in map_dets:
                map_dets[tuple(ddet)] = dvals[it]
            else:
                map_dets[tuple(ddet)] += dvals[it]
        for det, val in map_dets.items():
            dtrie.append(bw.b.VectorUInt8(det))
            dtrie.vals.append(val)
        dtrie.sort_dets()

        if target is None:
            target = self.target
        if left_vacuum is None:
            left_vacuum = self.left_vacuum
        mps_info = bw.brs.MPSInfo(self.n_sites, self.vacuum, target, self.ghamil.basis)
        mps_info.tag = tag
        if full_fci:
            mps_info.set_bond_dimension_full_fci(left_vacuum, self.vacuum)
        else:
            mps_info.set_bond_dimension_fci(left_vacuum, self.vacuum)
        mps_info.bond_dim = len(dets)

        mps = dtrie.construct_mps(mps_info).finalize()
        mps.info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        mps = self.adjust_mps(mps, dot=dot)[0]
        return mps

    def expr_builder(self):
        return ExprBuilder(self.bw)

    def finalize(self):
        bw = self.bw
        bw.b.Global.frame = None


class SOCDMRGDriver(DMRGDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                    dtx[ix(mask)] += ts.ravel()
            if iprint:
                xr = ("%20.15f" % const_es) if wex == "" else wex
                print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
            xiter += 1
        assert sorted(dts.keys()) == [2, 4]
        return dts[2], dts[4], const_es

    @staticmethod
    def make_su2_open_shell(h1e, g2e, const_e, cidx, midx, iprint=1):
        import block2 as b
        import numpy as np

        if iprint:
            print("-- Normal order (su2 open shell) using Wick's theorem")

        try:
            idx_map = b.MapWickIndexTypesSet()
        except ImportError:
            raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

        idx_map[b.WickIndexTypes.Inactive] = b.WickIndex.parse_set("pqrsijklmno")
        idx_map[b.WickIndexTypes.External] = b.WickIndex.parse_set("pqrsabcdefg")
        idx_map[b.WickIndexTypes.Single] = b.WickIndex.parse_set("pqrsrstuvwx")
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
        is_single = lambda x: (x & b.WickIndexTypes.Single) != b.WickIndexTypes.Nothing

        assert cidx.dtype == bool

        def ix(x):
            p = {"I": cidx, "S": midx, "E": ~cidx & ~midx}
            r = np.outer(p[x[0]], p[x[1]])
            for i in range(2, len(x)):
                r = np.outer(r, p[x[i]])
            return r.reshape((len(cidx),) * len(x))

        def tx(x, ix):
            for ig, ii in enumerate(ix):
                idx = (slice(None),) * ig
                idx += (
                    cidx
                    if is_inactive(ii.types)
                    else (midx if is_single(ii.types) else ~cidx & ~midx),
                )
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
                    mask += (
                        "I"
                        if is_inactive(t.indices[0].types)
                        else ("S" if is_single(t.indices[0].types) else "E")
                    )
                    result += t.indices[0].name
            np_str = ",".join(opidx) + "->" + result
            if 0 not in [x.size for x in tensors]:
                ts = f * np.einsum(np_str, *tensors, optimize=True)
                if len(wex) == 0:
                    const_es += ts
                else:
                    dtx[ix(mask)] += ts.ravel()
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
        assert cidxa.dtype == bool
        assert cidxb.dtype == bool

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
                    dtx[ix(mask)] += ts.ravel()
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

        assert cidx.dtype == bool

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
                    dtx[ix(mask)] += ts.ravel()
            if iprint:
                xr = ("%20.15f" % const_es) if wex == "" else wex
                print("%4d / %4d --" % (xiter, len(eq.terms)), xr, np_str, mask, f)
            xiter += 1
        assert sorted(dts.keys()) == [2, 4]
        return dts[2], dts[4], const_es


class ExprBuilder:
    def __init__(self, bw=None):
        if bw is None:
            bw = Block2Wrapper()
        self.data = bw.bx.GeneralFCIDUMP()
        if SymmetryTypes.SU2 in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SU2
        elif SymmetryTypes.SZ in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SZ
        elif SymmetryTypes.SGF in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SGF
        elif SymmetryTypes.SGB in bw.symm_type:
            self.data.elem_type = bw.b.ElemOpTypes.SGB
        else:
            self.data.elem_type = bw.b.ElemOpTypes.SGF
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
            nn = self.bw.b.SpinPermRecoupling.count_cds(expr)
            if nn != 0:
                assert len(idx) % nn == 0
                val = [val] * (len(idx) // nn)
            else:
                val = [val]
        self.data.data.append(self.bw.VectorFL(val))
        return self

    def add_sum_term(self, expr, arr, cutoff=1e-12, fast=True, factor=1.0, perm=None):
        import numpy as np

        self.data.exprs.append(expr)
        if fast:
            self.data.add_sum_term(
                np.ascontiguousarray(arr),
                cutoff,
                factor,
                self.bw.b.VectorUInt16([] if perm is None else perm),
            )
        else:
            idx, dt = [], []
            if perm is not None:
                arr = arr.transpose(*perm)
            for ix in np.ndindex(*arr.shape):
                if abs(arr[ix]) > cutoff:
                    idx.extend(ix)
                    dt.append(arr[ix] * factor)
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

    def finalize(self, adjust_order=True, merge=True, is_drt=False, fermionic_ops=None):
        if adjust_order:
            if fermionic_ops is not None:
                assert (
                    SymmetryTypes.SU2 not in self.bw.symm_type
                    and SymmetryTypes.PHSU2 not in self.bw.symm_type
                    and SymmetryTypes.SO4 not in self.bw.symm_type
                    and SymmetryTypes.SO3 not in self.bw.symm_type
                )
                self.data = self.data.adjust_order(fermionic_ops, merge=merge)
            else:
                self.data = self.data.adjust_order(merge=merge, is_drt=is_drt)
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
    ops_ghf = "1-N:D:C:N".split(":")

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
    def get_one_orb_rdm_h_terms(n_sites, is_sgf=False):
        h_terms = {}
        ops = OrbitalEntropy.ops_ghf if is_sgf else OrbitalEntropy.ops
        for i in range(n_sites):
            ih_terms = []
            for ix in [0, 3] if is_sgf else [0, 5, 10, 15]:
                xh_terms = {}
                for x in OrbitalEntropy.parse_expr(ops[ix]):
                    xh_terms[x[1:]] = ([i] * len(x[1:]), 1.0 if x[0] == "+" else -1.0)
                ih_terms.append(xh_terms)
            h_terms[(i,)] = ih_terms
        return h_terms

    @staticmethod
    def get_two_orb_rdm_h_terms(n_sites, ij_symm=True, block_symm=True, is_sgf=False):
        h_terms = {}
        # Table 3. J. Chem. Theory Comput. 2013, 9, 2959-2973
        if block_symm:
            if is_sgf:
                ts = "1/1 1/4 2/3 4/1 4/4"
            else:
                ts = (
                    "1/1 1/6 2/5 6/1 1/11 3/9 11/1 6/6 1/16 2/15 -3/14 -4/13 6/11 7/10 "
                    + "-8/9 11/6 12/5 16/1 11/11 6/16 8/14 16/6 11/16 12/15 16/11 16/16"
                )
        else:
            if is_sgf:
                ts = "1/1 1/4 2/3 -3/2 4/1 4/4"
            else:
                ts = (
                    "1/1 1/6 2/5 -5/2 6/1 1/11 3/9 -9/3 11/1 6/6 1/16 2/15 -3/14 "
                    + "-4/13 -5/12 6/11 7/10 -8/9 9/8 10/7 11/6 12/5 -13/4 14/3 -15/2 16/1 "
                    + "11/11 6/16 8/14 -14/8 16/6 11/16 12/15 -15/12 16/11 16/16"
                )
        ts = [[int(v) for v in u.split("/")] for u in ts.split()]
        if is_sgf:
            tsm = "1/1 1/4 4/1 4/4"
            ops = OrbitalEntropy.ops_ghf
        else:
            tsm = (
                "1/1 1/6 6/1 1/11 11/1 6/6 1/16 6/11 11/6 16/1 "
                + "11/11 6/16 16/6 11/16 16/11 16/16"
            )
            ops = OrbitalEntropy.ops
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
                    for x in OrbitalEntropy.parse_expr(ops[ix - 1]):
                        for y in OrbitalEntropy.parse_expr(ops[iy - 1]):
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
    def get_one_orb_rdm_exprs(is_sgf=False):
        ops = OrbitalEntropy.ops_ghf if is_sgf else OrbitalEntropy.ops
        exprs = {}
        for iix, ix in enumerate([0, 3] if is_sgf else [0, 5, 10, 15]):
            for x in OrbitalEntropy.parse_expr(ops[ix]):
                kk = (x[1:], (0,) * len(x[1:]))
                if kk not in exprs:
                    exprs[kk] = []
                exprs[kk].append((iix, 1.0 if x[0] == "+" else -1.0))
        return exprs, (2 if is_sgf else 4)

    @staticmethod
    def get_two_orb_rdm_exprs(is_sgf=False):
        exprs = {}
        # Table 3. J. Chem. Theory Comput. 2013, 9, 2959-2973
        if is_sgf:
            ts = "1/1 1/4 2/3 -3/2 4/1 4/4"
            ops = OrbitalEntropy.ops_ghf
        else:
            ts = (
                "1/1 1/6 2/5 -5/2 6/1 1/11 3/9 -9/3 11/1 6/6 1/16 2/15 -3/14 "
                + "-4/13 -5/12 6/11 7/10 -8/9 9/8 10/7 11/6 12/5 -13/4 14/3 -15/2 16/1 "
                + "11/11 6/16 8/14 -14/8 16/6 11/16 12/15 -15/12 16/11 16/16"
            )
            ops = OrbitalEntropy.ops
        ts = [[int(v) for v in u.split("/")] for u in ts.split()]
        for ii, (ix, iy) in enumerate(ts):
            ff = -1 if ix < 0 else 1
            for x in OrbitalEntropy.parse_expr(ops[abs(ix) - 1]):
                for y in OrbitalEntropy.parse_expr(ops[iy - 1]):
                    kk = (x[1:] + y[1:], (0,) * len(x[1:]) + (1,) * len(y[1:]))
                    if kk not in exprs:
                        exprs[kk] = []
                    exprs[kk].append((ii, ff if x[0] == y[0] else -ff))
        return exprs, len(ts)

    @staticmethod
    def get_two_orb_rdm_eigvals(ld, diag_only=False):
        import numpy as np

        if diag_only and len(ld) == 6:
            return ld[np.array([0, 1, 4, 5], dtype=int)]
        elif diag_only and len(ld) == 36:
            return ld[
                np.array(
                    [0, 1, 4, 5, 8, 9, 10, 15, 20, 25, 26, 27, 30, 31, 34, 35],
                    dtype=int,
                )
            ]
        if len(ld) == 16 or len(ld) == 4:
            return ld
        elif len(ld) == 26 or len(ld) == 5:
            if len(ld) == 26:
                lx = np.zeros((16,), dtype=ld.dtype)
                dds = [1, 2, 2, 1, 4, 1, 2, 2, 1]
            else:
                lx = np.zeros((4,), dtype=ld.dtype)
                dds = [1, 2, 1]
            ix, ip = 0, 0
            for d in dds:
                if d == 1:
                    lx[ix] = ld[ip]
                else:
                    dd = np.zeros((d, d), dtype=ld.dtype)
                    dd[np.triu_indices(d)] = ld[ip : ip + d * (d + 1) // 2]
                    lx[ix : ix + d] = np.linalg.eigvalsh(dd, UPLO="U")
                ix += d
                ip += d * (d + 1) // 2
            assert ix == len(lx) and ip == len(ld)
        elif len(ld) == 36 or len(ld) == 6:
            if len(ld) == 36:
                lx = np.zeros((16,), dtype=ld.dtype)
                dds = [1, 2, 2, 1, 4, 1, 2, 2, 1]
            else:
                lx = np.zeros((4,), dtype=ld.dtype)
                dds = [1, 2, 1]
            ix, ip = 0, 0
            for d in dds:
                if d == 1:
                    lx[ix] = ld[ip]
                else:
                    lx[ix : ix + d] = np.linalg.eigvalsh(
                        ld[ip : ip + d * d].reshape(d, d)
                    )
                ix += d
                ip += d * d
            assert ix == len(lx) and ip == len(ld)
        else:
            lx = 0
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
                    dtx[ix(mask)] += ts.ravel()
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
                        dtx[ix(mask)] += ts.ravel()
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
                    dtx[ix(mask)] += ts.ravel()
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
