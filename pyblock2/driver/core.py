
from enum import Enum

class SymmetryTypes(Enum):
    SU2 = 0
    SZ = 1
    SGF = 2
    SGB = 3

class Block2Wrapper:
    def __init__(self, symm_type=SymmetryTypes.SU2):
        import block2 as b
        if symm_type == SymmetryTypes.SU2:
            import block2.su2 as bs
            self.SX = b.SU2
            self.VectorSX = b.VectorSU2
        elif symm_type == SymmetryTypes.SZ:
            import block2.sz as bs
            self.SX = b.SZ
            self.VectorSX = b.VectorSZ
        elif symm_type == SymmetryTypes.SGF:
            import block2.sgf as bs
            self.SX = b.SGF
            self.VectorSX = b.VectorSGF
        elif symm_type == SymmetryTypes.SGB:
            import block2.sgb as bs
            self.SX = b.SGB
            self.VectorSX = b.VectorSGB
        self.symm_type = symm_type
        self.b = b
        self.bs = bs

class DMRGDriver:
    def __init__(self, stack_mem=1 << 30, scratch='./nodex', n_threads=None, symm_type=SymmetryTypes.SU2):

        self.bw = Block2Wrapper(symm_type)
        bw = self.bw

        bw.b.Global.frame = bw.b.DoubleDataFrame(int(stack_mem * 0.1), int(stack_mem * 0.9), scratch)
        if n_threads is None:
            n_threads = bw.b.Global.threading.n_threads_global
        bw.b.Global.threading = bw.b.Threading(
            bw.b.ThreadingTypes.OperatorBatchedGEMM | bw.b.ThreadingTypes.Global,
            n_threads, n_threads, 1)
        bw.b.Global.threading.seq_type = bw.b.SeqTypes.Tasked
        bw.b.Global.frame.fp_codec = bw.b.DoubleFPCodec(1E-16, 1024)
        bw.b.Global.frame.minimal_disk_usage = True
        bw.b.Global.frame.use_main_stack = False
    
    def initialize_system(self, n_sites, n_elec=0, spin=0, orb_sym=None, heis_twos=-1, heis_twosz=0):
        bw = self.bw
        self.vacuum = bw.SX(0, 0, 0)
        if heis_twos != -1 and bw.SX == bw.b.SU2 and n_elec == 0:
            n_elec = n_sites * heis_twos
        self.target = bw.SX(n_elec if heis_twosz == 0 else heis_twosz, spin, 0)
        self.n_sites = n_sites
        if orb_sym is None:
            self.orb_sym = bw.b.VectorUInt8([0] * self.n_sites)
        else:
            self.orb_sym = bw.b.VectorUInt8(orb_sym)
        self.ghamil = bw.bs.GeneralHamiltonian(self.vacuum, self.n_sites, self.orb_sym, heis_twos)
    
    def get_mpo(self, expr, iprint=0):
        bw = self.bw
        mpo = bw.bs.GeneralMPO(self.ghamil, expr, bw.b.MPOAlgorithmTypes.FastBipartite, 0.0, -1, iprint > 0)
        mpo = bw.bs.SimplifiedMPO(mpo, bw.bs.Rule(), False, False)
        return mpo
    
    def dmrg(self, mpo, ket, n_sweeps=10, tol=1E-8, bond_dims=None, noises=None, thrds=None, iprint=0):
        bw = self.bw
        if bond_dims is None:
            bond_dims = [ket.info.bond_dim]
        if noises is None:
            noises = [1E-5] * 5 + [0]
        if thrds is None:
            thrds = [1E-6] * 4 + [1E-7] * 1
        me = bw.bs.MovingEnvironment(mpo, ket, ket, "DMRG")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments(iprint >= 3)
        dmrg = bw.bs.DMRG(me, bw.b.VectorUBond(bond_dims), bw.b.VectorDouble(noises))
        dmrg.noise_type = bw.b.NoiseTypes.ReducedPerturbative
        dmrg.davidson_conv_thrds = bw.b.VectorDouble(thrds)
        dmrg.davidson_max_iter = 5000
        dmrg.davidson_soft_max_iter = 4000
        dmrg.iprint = iprint
        ener = dmrg.solve(n_sweeps, ket.center == 0, tol)
        ket.info.bond_dim = max(ket.info.bond_dim, bond_dims[-1])
        if isinstance(ket, bw.bs.MultiMPS):
            ener = list(dmrg.sweep_energies[-1])
        return ener
    
    def align_mps_center(self, ket, ref):
        ket.info.bond_dim = max(ket.info.bond_dim, ket.info.get_max_bond_dimension())
        if ket.center != ref.center:
            if ref.center == 0:
                ket.center += 1
                ket.canonical_form = ket.canonical_form[:-1] + 'S'
                while ket.center != 0:
                    ket.move_left(self.ghamil.opf.cg, None)
            else:
                ket.canonical_form = 'K' + ket.canonical_form[1:]
                while ket.center != ket.n_sites - 1:
                    ket.move_right(self.ghamil.opf.cg, None)
                ket.center -= 1

    def multiply(self, bra, mpo, ket, n_sweeps=10, tol=1E-8, bond_dims=None, iprint=0):
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
        return ex
    
    def get_random_mps(self, tag, bond_dim=500, center=0, target=None, nroots=1):
        bw = self.bw
        if target is None:
            target = self.target
        if nroots == 1:
            mps_info = bw.bs.MPSInfo(self.n_sites, self.vacuum, target, self.ghamil.basis)
            mps = bw.bs.MPS(self.n_sites, center, 2)
        else:
            targets = bw.VectorSX([target]) if isinstance(target, bw.SX) else target
            mps_info = bw.bs.MultiMPSInfo(self.n_sites, self.vacuum, targets, self.ghamil.basis)
            mps = bw.bs.MultiMPS(self.n_sites, center, 2, nroots)
        mps_info.tag = tag
        mps_info.set_bond_dimension(bond_dim)
        mps_info.bond_dim = bond_dim
        mps.initialize(mps_info)
        mps.random_canonicalize()
        mps.save_mutable()
        mps_info.save_mutable()
        return mps

    def expr_builder(self):
        return ExprBuilder(self.bw)
    
    def finalize(self):
        bw = self.bw
        bw.b.Global.frame = None


class ExprBuilder:
    def __init__(self, bw=Block2Wrapper()):
        self.data = bw.b.GeneralFCIDUMP()
        if bw.symm_type == SymmetryTypes.SU2:
            self.data.elem_type = bw.b.ElemOpTypes.SU2
        elif bw.symm_type == SymmetryTypes.SZ:
            self.data.elem_type = bw.b.ElemOpTypes.SZ
        elif bw.symm_type == SymmetryTypes.SGF:
            self.data.elem_type = bw.b.ElemOpTypes.SGF
        elif bw.symm_type == SymmetryTypes.SGB:
            self.data.elem_type = bw.b.ElemOpTypes.SGB
        self.data.const_e = 0.0
        self.bw = bw

    def add_const(self, x):
        self.data.const_e = self.data.const_e + x
        return self

    def add_term(self, expr, idx, val):
        self.data.exprs.append(expr)
        self.data.indices.append(self.bw.b.VectorUInt16(idx))
        self.data.data.append(self.bw.b.VectorDouble([val]))
        return self

    def add_sum_term(self, expr, arr, cutoff=1E-12):
        import numpy as np
        self.data.exprs.append(expr)
        idx, dt = [], []
        for ix in np.ndindex(*arr.shape):
            if abs(arr[ix]) > cutoff:
                idx.extend(ix)
                dt.append(arr[ix])
        self.data.indices.append(self.bw.b.VectorUInt16(idx))
        self.data.data.append(self.bw.b.VectorDouble(dt))
        return self

    def finalize(self):
        self.data = self.data.adjust_order()
        return self.data
