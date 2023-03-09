
.. highlight:: bash

Custom Hamiltonian
==================

In this tutorial, we provide an example python script for performing DMRG using custom Hamiltonians,
where the operators and states at local Hilbert space at every site can be redefined.
It is also possible to use different local Hilbert space for different sites.
New letters can be introduced for representing new operators.

.. highlight:: python3

In the following example, we implement a custom Hamiltonian for the Hubbard model.
In the standard implementation, the on-site term was represented as ``cdCD``.
Here we instead introduce a single letter ``N`` for the ``cdCD`` term.
The matrix representation of ``N`` is given in the ``init_site_ops`` method. ::

    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    import numpy as np
    from itertools import accumulate

    L = 8
    U = 2

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=L, n_elec=L, spin=0, orb_sym=None)

    GH = driver.bw.bs.GeneralHamiltonian

    class HubbardHamiltonian(GH):

        def __init__(self, vacuum, n_sites, orb_sym):
            GH.__init__(self)
            self.opf = driver.bw.bs.OperatorFunctions(driver.bw.brs.CG())
            self.vacuum = vacuum
            self.n_sites = n_sites
            self.orb_sym = orb_sym
            self.basis = driver.bw.brs.VectorStateInfo([
                self.get_site_basis(m) for m in range(self.n_sites)
            ])
            self.site_op_infos = driver.bw.brs.VectorVectorPLMatInfo([
                driver.bw.brs.VectorPLMatInfo() for _ in range(self.n_sites)
            ])
            self.site_norm_ops = driver.bw.bs.VectorMapStrSpMat([
                driver.bw.bs.MapStrSpMat() for _ in range(self.n_sites)
            ])
            self.init_site_ops()

        def get_site_basis(self, m):
            """Single site states."""
            bz = driver.bw.brs.StateInfo()
            bz.allocate(4)
            bz.quanta[0] = driver.bw.SX(0, 0, 0)
            bz.quanta[1] = driver.bw.SX(1, 1, self.orb_sym[m])
            bz.quanta[2] = driver.bw.SX(1, -1, self.orb_sym[m])
            bz.quanta[3] = driver.bw.SX(2, 0, 0)
            bz.n_states[0] = bz.n_states[1] = bz.n_states[2] = bz.n_states[3] = 1
            bz.sort_states()
            return bz

        def init_site_ops(self):
            """Initialize operator quantum numbers at each site (site_op_infos)
            and primitive (single character) site operators (site_norm_ops)."""
            i_alloc = driver.bw.b.IntVectorAllocator()
            d_alloc = driver.bw.b.DoubleVectorAllocator()
            # site op infos
            max_n, max_s = 10, 10
            max_n_odd, max_s_odd = max_n | 1, max_s | 1
            max_n_even, max_s_even = max_n_odd ^ 1, max_s_odd ^ 1
            for m in range(self.n_sites):
                qs = {self.vacuum}
                for n in range(-max_n_odd, max_n_odd + 1, 2):
                    for s in range(-max_s_odd, max_s_odd + 1, 2):
                        qs.add(driver.bw.SX(n, s, self.orb_sym[m]))
                for n in range(-max_n_even, max_n_even + 1, 2):
                    for s in range(-max_s_even, max_s_even + 1, 2):
                        qs.add(driver.bw.SX(n, s, 0))
                for q in sorted(qs):
                    mat = driver.bw.brs.SparseMatrixInfo(i_alloc)
                    mat.initialize(self.basis[m], self.basis[m], q, q.is_fermion)
                    self.site_op_infos[m].append((q, mat))

            # prim ops
            for m in range(self.n_sites):

                # ident
                mat = driver.bw.bs.SparseMatrix(d_alloc)
                info = self.find_site_op_info(m, driver.bw.SX(0, 0, 0))
                mat.allocate(info)
                mat[info.find_state(driver.bw.SX(0, 0, 0))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(1, 1, self.orb_sym[m]))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(1, -1, self.orb_sym[m]))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(2, 0, 0))] = np.array([1.0])
                self.site_norm_ops[m][""] = mat

                # C alpha
                mat = driver.bw.bs.SparseMatrix(d_alloc)
                info = self.find_site_op_info(m, driver.bw.SX(1, 1, self.orb_sym[m]))
                mat.allocate(info)
                mat[info.find_state(driver.bw.SX(0, 0, 0))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(1, -1, self.orb_sym[m]))] = np.array([1.0])
                self.site_norm_ops[m]["c"] = mat

                # D alpha
                mat = driver.bw.bs.SparseMatrix(d_alloc)
                info = self.find_site_op_info(m, driver.bw.SX(-1, -1, self.orb_sym[m]))
                mat.allocate(info)
                mat[info.find_state(driver.bw.SX(1, 1, self.orb_sym[m]))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(2, 0, 0))] = np.array([1.0])
                self.site_norm_ops[m]["d"] = mat

                # C beta
                mat = driver.bw.bs.SparseMatrix(d_alloc)
                info = self.find_site_op_info(m, driver.bw.SX(1, -1, self.orb_sym[m]))
                mat.allocate(info)
                mat[info.find_state(driver.bw.SX(0, 0, 0))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(1, 1, self.orb_sym[m]))] = np.array([-1.0])
                self.site_norm_ops[m]["C"] = mat

                # D beta
                mat = driver.bw.bs.SparseMatrix(d_alloc)
                info = self.find_site_op_info(m, driver.bw.SX(-1, 1, self.orb_sym[m]))
                mat.allocate(info)
                mat[info.find_state(driver.bw.SX(1, -1, self.orb_sym[m]))] = np.array([1.0])
                mat[info.find_state(driver.bw.SX(2, 0, 0))] = np.array([-1.0])
                self.site_norm_ops[m]["D"] = mat

                # Nup * Ndn
                mat = driver.bw.bs.SparseMatrix(d_alloc)
                info = self.find_site_op_info(m, driver.bw.SX(0, 0, 0))
                mat.allocate(info)
                mat[info.find_state(driver.bw.SX(2, 0, 0))] = np.array([1.0])
                self.site_norm_ops[m]["N"] = mat

        def get_site_string_ops(self, m, ops):
            """Construct longer site operators from primitive ones."""
            d_alloc = driver.bw.b.DoubleVectorAllocator()
            for k in ops:
                if k in self.site_norm_ops[m]:
                    ops[k] = self.site_norm_ops[m][k]
                else:
                    xx = self.site_norm_ops[m][k[0]]
                    for p in k[1:]:
                        xp = self.site_norm_ops[m][p]
                        q = xx.info.delta_quantum + xp.info.delta_quantum
                        mat = driver.bw.bs.SparseMatrix(d_alloc)
                        mat.allocate(self.find_site_op_info(m, q))
                        self.opf.product(0, xx, xp, mat)
                        xx = mat
                    ops[k] = self.site_norm_ops[m][k] = xx
            return ops

        def init_string_quanta(self, exprs, term_l, left_vacuum):
            """Quantum number for string operators (orbital independent part)."""
            qs = {
                'N': driver.bw.SX(0, 0, 0),
                'c':  driver.bw.SX(1, 1, 0),
                'C':  driver.bw.SX(1, -1, 0),
                'd':  driver.bw.SX(-1, -1, 0),
                'D':  driver.bw.SX(-1, 1, 0),
            }
            return driver.bw.VectorVectorSX([driver.bw.VectorSX(list(accumulate(
                [qs['N']] + [qs[x] for x in expr], lambda x, y: x + y)))
                for expr in exprs
            ])

        def get_string_quanta(self, ref, expr, idxs, k):
            """Quantum number for string operators (orbital dependent part)."""
            l, r = ref[k], ref[-1] - ref[k]
            for j, (ex, ix) in enumerate(zip(expr, idxs)):
                ipg = self.orb_sym[ix]
                if ex == "N":
                    pass
                elif j < k:
                    l.pg = l.pg ^ ipg
                else:
                    r.pg = r.pg ^ ipg
            return l, r

        def get_string_quantum(self, expr, idxs):
            """Total quantum number for a string operator."""
            qs = lambda ix: {
                'N': driver.bw.SX(0, 0, 0),
                'c':  driver.bw.SX(1, 1, self.orb_sym[ix]),
                'C':  driver.bw.SX(1, -1, self.orb_sym[ix]),
                'd':  driver.bw.SX(-1, -1, self.orb_sym[ix]),
                'D':  driver.bw.SX(-1, 1, self.orb_sym[ix]),
            }
            return sum([qs(0)['N']] + [qs(ix)[ex] for ex, ix in zip(expr, idxs)])

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


    driver.ghamil = HubbardHamiltonian(driver.vacuum, driver.n_sites, driver.orb_sym)

    b = driver.expr_builder()

    # make sure the indices in every term are non-descending
    for t, ops in zip([-1, 1, -1, 1], ["cd", "dc", "CD", "DC"]):
        b.add_term(ops, np.array([[i, i + 1] for i in range(L - 1)]).flatten(),
            [t] * (L - 1))
    b.add_term("N", np.array([[i, ] for i in range(L)]).flatten(), [U] * L)
    mpo = driver.get_mpo(b.finalize(adjust_order=False), iprint=2)

    def run_dmrg(driver, mpo):
        ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
        bond_dims = [250] * 4 + [500] * 4
        noises = [1e-4] * 4 + [1e-5] * 4 + [0]
        thrds = [1e-10] * 8
        return driver.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=1,
        )

    energies = run_dmrg(driver, mpo)
    print('DMRG energy = %20.15f' % energies)
