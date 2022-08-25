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


class Block2Wrapper:
    def __init__(self, symm_type=SymmetryTypes.SU2):
        import block2 as b

        self.b = b
        self.symm_type = symm_type
        if SymmetryTypes.SPCPX in symm_type:
            self.VectorFL = b.VectorComplexFloat
            self.VectorFP = b.VectorFloat
            self.bx = b.sp.cpx
        elif SymmetryTypes.CPX in symm_type:
            self.VectorFL = b.VectorComplexDouble
            self.VectorFP = b.VectorDouble
            self.bx = b.cpx
        elif SymmetryTypes.SP in symm_type:
            self.VectorFL = b.VectorFloat
            self.VectorFP = b.VectorFloat
            self.bx = b.sp
        else:
            self.VectorFL = b.VectorDouble
            self.VectorFP = b.VectorDouble
            self.bx = b
        if SymmetryTypes.SU2 in symm_type:
            self.bs = self.bx.su2
            self.brs = b.su2
            self.SX = b.SU2
            self.VectorSX = b.VectorSU2
        elif SymmetryTypes.SZ in symm_type:
            self.bs = self.bx.sz
            self.brs = b.sz
            self.SX = b.SZ
            self.VectorSX = b.VectorSZ
        elif SymmetryTypes.SGF in symm_type:
            self.bs = self.bx.sgf
            self.brs = b.sgf
            self.SX = b.SGF
            self.VectorSX = b.VectorSGF
        elif SymmetryTypes.SGB in symm_type:
            self.bs = self.bx.sgb
            self.brs = b.sgb
            self.SX = b.SGB
            self.VectorSX = b.VectorSGB


class DMRGDriver:
    def __init__(
        self,
        stack_mem=1 << 30,
        scratch="./nodex",
        restart_dir=None,
        n_threads=None,
        symm_type=SymmetryTypes.SU2,
        mpi=None,
    ):
        if mpi is not None and mpi:
            self.mpi = True
        else:
            self.mpi = None
            self.prule = None

        self.scratch = scratch
        self.stack_mem = stack_mem
        self.restart_dir = restart_dir
        self.set_symm_type(symm_type)
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

    def parallelize_integrals(self, para_type, h1e, g2e, const):
        import numpy as np

        if para_type == ParallelTypes.Nothing or self.mpi is None:
            return h1e, g2e, const
        ixs = np.mgrid[tuple(slice(x) for x in h1e.shape)].reshape((h1e.ndim, -1)).T
        sixs = np.sort(ixs, axis=1)
        gixs = np.mgrid[tuple(slice(x) for x in g2e.shape)].reshape((g2e.ndim, -1)).T
        gsixs = np.sort(gixs, axis=1)

        if para_type == ParallelTypes.I:
            mask1 = ixs[:, 0] % self.mpi.size == self.mpi.rank
            mask2 = gixs[:, 0] % self.mpi.size == self.mpi.rank
        elif para_type == ParallelTypes.SI:
            mask1 = sixs[:, 0] % self.mpi.size == self.mpi.rank
            mask2 = gsixs[:, 0] % self.mpi.size == self.mpi.rank
        elif para_type == ParallelTypes.J:
            mask1 = ixs[:, 1] % self.mpi.size == self.mpi.rank
            mask2 = gixs[:, 1] % self.mpi.size == self.mpi.rank
        elif para_type == ParallelTypes.SJ:
            mask1 = sixs[:, 1] % self.mpi.size == self.mpi.rank
            mask2 = gsixs[:, 1] % self.mpi.size == self.mpi.rank
        elif para_type == ParallelTypes.SIJ:
            mask1 = sixs[:, 0] % self.mpi.size == self.mpi.rank
            mask2a = (gsixs[:, 1] == gsixs[:, 2]) & (
                gsixs[:, 1] % self.mpi.size == self.mpi.rank
            )
            mask2b = (
                (gsixs[:, 1] != gsixs[:, 2])
                & (gsixs[:, 0] <= gsixs[:, 1])
                & (
                    (gsixs[:, 1] * (gsixs[:, 1] + 1) // 2 + gsixs[:, 0]) % self.mpi.size
                    == self.mpi.rank
                )
            )
            mask2c = (
                (gsixs[:, 1] != gsixs[:, 2])
                & (gsixs[:, 0] > gsixs[:, 1])
                & (
                    (gsixs[:, 0] * (gsixs[:, 0] + 1) // 2 + gsixs[:, 1]) % self.mpi.size
                    == self.mpi.rank
                )
            )
            mask2 = mask2a | mask2b | mask2c
        h1e[~mask1.reshape(h1e.shape)] = 0.0
        g2e[~mask2.reshape(g2e.shape)] = 0.0
        if self.mpi.rank != self.mpi.root:
            const = 0
        return h1e, g2e, const

    def set_symm_type(self, symm_type):
        self.bw = Block2Wrapper(symm_type)
        bw = self.bw

        if SymmetryTypes.SP not in bw.symm_type:
            bw.b.Global.frame = bw.b.DoubleDataFrame(
                int(self.stack_mem * 0.1), int(self.stack_mem * 0.9), self.scratch
            )
            bw.b.Global.frame.fp_codec = bw.b.DoubleFPCodec(1e-16, 1024)
            bw.b.Global.frame_float = None
            self.frame = bw.b.Global.frame
        else:
            bw.b.Global.frame_float = bw.b.FloatDataFrame(
                int(self.stack_mem * 0.1), int(self.stack_mem * 0.9), self.scratch
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
    ):
        bw = self.bw
        self.vacuum = bw.SX(0, 0, 0)
        if heis_twos != -1 and bw.SX == bw.b.SU2 and n_elec == 0:
            n_elec = n_sites * heis_twos
        if pg_irrep is None:
            if hasattr(self, "pg_irrep"):
                pg_irrep = self.pg_irrep
            else:
                pg_irrep = 0
        self.target = bw.SX(n_elec if heis_twosz == 0 else heis_twosz, spin, pg_irrep)
        self.n_sites = n_sites
        if orb_sym is None:
            self.orb_sym = bw.b.VectorUInt8([0] * self.n_sites)
        else:
            self.orb_sym = bw.b.VectorUInt8(orb_sym)
        self.ghamil = bw.bs.GeneralHamiltonian(
            self.vacuum, self.n_sites, self.orb_sym, heis_twos
        )

    def read_fcidump(self, filename, pg="d2h", rescale=None, iprint=1):
        bw = self.bw
        fcidump = bw.bx.FCIDUMP()
        fcidump.read(filename)
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
        import numpy as np

        self.h1e = np.array(fcidump.h1e_matrix(), copy=False).reshape(
            (self.n_sites,) * 2
        )
        self.g2e = np.array(fcidump.g2e_1fold(), copy=False).reshape(
            (self.n_sites,) * 4
        )
        if iprint >= 1:
            print("symmetrize error = ", fcidump.symmetrize(self.orb_sym))
        return fcidump

    def get_mpo(self, expr, iprint=0):
        bw = self.bw
        mpo = bw.bs.GeneralMPO(
            self.ghamil, expr, bw.b.MPOAlgorithmTypes.FastBipartite, 0.0, -1, iprint > 0
        )
        mpo = bw.bs.SimplifiedMPO(mpo, bw.bs.Rule(), False, False)
        if self.mpi:
            mpo = bw.bs.ParallelMPO(mpo, self.prule)
        return mpo

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
        me.init_environments(iprint >= 2)
        dmrg = bw.bs.DMRG(me, bw.b.VectorUBond(bond_dims), bw.VectorFP(noises))
        if dav_type is not None:
            dmrg.davidson_type = getattr(bw.b.DavidsonTypes, dav_type)
        dmrg.noise_type = bw.b.NoiseTypes.ReducedPerturbative
        dmrg.davidson_conv_thrds = bw.VectorFP(thrds)
        dmrg.davidson_max_iter = dav_max_iter + 100
        dmrg.davidson_soft_max_iter = dav_max_iter
        dmrg.iprint = iprint
        dmrg.cutoff = cutoff
        dmrg.trunc_type = dmrg.trunc_type | bw.b.TruncationTypes.RealDensityMatrix
        ener = dmrg.solve(n_sweeps, ket.center == 0, tol)
        ket.info.bond_dim = max(ket.info.bond_dim, bond_dims[-1])
        if isinstance(ket, bw.bs.MultiMPS):
            ener = list(dmrg.sweep_energies[-1])
        self._dmrg = dmrg
        if self.mpi is not None:
            self.mpi.barrier()
        return ener

    def align_mps_center(self, ket, ref):
        if self.mpi is not None:
            self.mpi.barrier()
        ket.info.bond_dim = max(ket.info.bond_dim, ket.info.get_max_bond_dimension())
        if ket.center != ref.center:
            if ref.center == 0:
                ket.center += 1
                ket.canonical_form = ket.canonical_form[:-1] + "S"
                while ket.center != 0:
                    ket.move_left(self.ghamil.opf.cg, self.prule)
            else:
                ket.canonical_form = "K" + ket.canonical_form[1:]
                while ket.center != ket.n_sites - 1:
                    ket.move_right(self.ghamil.opf.cg, self.prule)
                ket.center -= 1
        if self.mpi is not None:
            self.mpi.barrier()

    def multiply(self, bra, mpo, ket, n_sweeps=10, tol=1e-8, bond_dims=None, iprint=0):
        bw = self.bw
        if bond_dims is None:
            bond_dims = [ket.info.bond_dim]
        self.align_mps_center(bra, ket)
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "MULT")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments(iprint >= 3)
        cps = bw.bs.Linear(me, bw.b.VectorUBond(bond_dims), bw.b.VectorUBond(bond_dims))
        cps.iprint = iprint
        norm = cps.solve(n_sweeps, ket.center == 0, tol)
        if self.mpi is not None:
            self.mpi.barrier()
        return norm

    def expectation(self, bra, mpo, ket, iprint=0):
        bw = self.bw
        bond_dim = max(bra.info.bond_dim, ket.info.bond_dim)
        self.align_mps_center(bra, ket)
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "EXPT")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments(iprint >= 3)
        expect = bw.bs.Expect(me, bond_dim, bond_dim)
        expect.iprint = iprint
        ex = expect.solve(False, ket.center != 0)
        if self.mpi is not None:
            self.mpi.barrier()
        return ex

    def fix_restarting_mps(self, mps):
        bw = self.bw
        cg = bw.bs.CG(200)
        cg.initialize()
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
        mps_info.tag = tag
        mps_info.load_mutable()
        mps_info.bond_dim = max(mps_info.bond_dim, mps_info.get_max_bond_dimension())
        mps = bw.bs.MPS(mps_info) if nroots == 1 else bw.bs.MultiMPS(mps_info)
        mps.load_data()
        mps.load_mutable()
        self.fix_restarting_mps(mps)
        return mps

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
        self, tag, bond_dim=500, center=0, dot=2, target=None, nroots=1, occs=None
    ):
        bw = self.bw
        if target is None:
            target = self.target
        if nroots == 1:
            mps_info = bw.brs.MPSInfo(
                self.n_sites, self.vacuum, target, self.ghamil.basis
            )
            mps = bw.bs.MPS(self.n_sites, center, dot)
        else:
            targets = bw.VectorSX([target]) if isinstance(target, bw.SX) else target
            mps_info = bw.brs.MultiMPSInfo(
                self.n_sites, self.vacuum, targets, self.ghamil.basis
            )
            mps = bw.bs.MultiMPS(self.n_sites, center, dot, nroots)
        mps_info.tag = tag
        mps_info.set_bond_dimension_full_fci(self.vacuum, self.vacuum)
        if occs is not None:
            mps_info.set_bond_dimension_using_occ(bond_dim, bw.b.VectorDouble(occs))
        else:
            mps_info.set_bond_dimension(bond_dim)
        mps_info.bond_dim = bond_dim
        mps.initialize(mps_info)
        mps.random_canonicalize()
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
        self.data.exprs.append(expr)
        self.data.indices.append(self.bw.b.VectorUInt16(idx))
        self.data.data.append(self.bw.VectorFL([val]))
        return self

    def add_sum_term(self, expr, arr, cutoff=1e-12):
        import numpy as np

        self.data.exprs.append(expr)
        idx, dt = [], []
        for ix in np.ndindex(*arr.shape):
            if abs(arr[ix]) > cutoff:
                idx.extend(ix)
                dt.append(arr[ix])
        self.data.indices.append(self.bw.b.VectorUInt16(idx))
        self.data.data.append(self.bw.VectorFL(dt))
        return self

    def finalize(self):
        self.data = self.data.adjust_order()
        return self.data
