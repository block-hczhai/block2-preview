
.. highlight:: bash

.. _tutorial_spin_projected_dmrg:

Spin-Projected DMRG
===================

Spin-projected DMRG (SP-DMRG) is a powerful technique for generating reliable initial guess
Matrix Product States (MPS) for spin-adapted DMRG, particularly in systems with numerous
competing broken-symmetry states. Due to its high computational cost, SP-DMRG is typically
performed only at small bond dimensions. The resulting optimized MPS can then serve as a
qualitatively reliable initial guess for subsequent, larger-scale optimization using
spin-adapted DMRG (under SU2 symmetry).

Reference for the spin-projected DMRG algorithm:

* Li, Z., Chan, G. K.-L. Spin-Projected Matrix Product States: Versatile Tool for Strongly Correlated Systems. *Journal of Chemical Theory and Computation* 2017, **13**, 2681-2695. doi: `10.1021/acs.jctc.7b00270 <https://doi.org/10.1021/acs.jctc.7b00270>`_

The following example shows how to use spin-projected DMRG to generate the initial guess MPS.
We study the three broken-symmetry states of the Fe4S4 active space model. The integral file
can be found using ::

    wget -O Fe4S4.FCIDUMP https://raw.githubusercontent.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters/main/Fe2S2_and_Fe4S4/Fe4S4/fe4s4

Exact MPO
---------

In the first example, we use an exact MPO for the Hamiltonian, this can be done directly
in the particle-number U1 symmetry mode. MPS can be initialized using a broken-symmetry
determinant in the particle-number and projected spin symmetry mode, and then transformed
to the particle-number U1 symmetry mode.

.. highlight:: python3

SP-DMRG is performed with particle-number U1 symmetry only, and the final MPS is transformed
to the SU2 symmetry mode (``ket2, tag='KETX-0'``) which can be later loaded in the SU2
symmetry mode to do spin-adapted DMRG with larger bond dimensions (not performed here). ::

    import numpy as np, sys
    import itertools
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
    from pyblock2.algebra.io import MPSTools

    istate = int(sys.argv[1])

    driver = DMRGDriver(scratch="/tmp", symm_type=SymmetryTypes.SAnySZ, stack_mem=120 << 30, fp_codec_cutoff=0.0, n_threads=64)

    bond_dims = [50] * 8 + [100] * 8
    noises = [1E-5] * (len(bond_dims) - 4) + [0] * 4
    thrds = [1E-7] * len(bond_dims)
    n_sweeps = len(bond_dims)

    driver.read_fcidump(filename='Fe4S4.FCIDUMP', pg='d2h')
    driver.spin = 0
    twos = 0

    npts = driver.get_spin_projection_npts(n_sites=driver.n_sites, n_elec=driver.n_elec, twos=twos)
    print("NPTS = %d" % npts)

    driver.set_symmetry_groups("U1Fermi", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=2, simple_const=True, add_ident=False)
    print("MPO = ", mpo.get_bond_dims())
    pmpo = driver.get_spin_projection_mpo(twos=twos, twosz=driver.spin, npts=npts, use_sz_symm=False, cutoff=1E-12, add_ident=True, iprint=1)

    target = driver.target

    driver.set_symmetry_groups("U1Fermi", "U1", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    n_sites = driver.n_sites

    xdstr = [
        '22aaaaa2aaaa222222222222b2bbbbbbbb22',
        '22aaaaabbb2b222222222222a2aaabbbbb22',
        '222aaaab2bbb222222222222bbbbbaaaaa22',
    ][istate]

    print(istate, xdstr)

    ket = driver.get_mps_from_csf_coefficients([xdstr], dvals=[1.0], tag='KET', dot=1)
    driver.align_mps_center(ket, ref=0)
    ket = driver.adjust_mps(ket, dot=2)[0]
    pket = driver.mps_change_symm(ket, 'PKET-0', target)

    energy = driver.dmrg(mpo, pket, stacked_mpo=pmpo, metric_mpo=pmpo, context_ket=ket, n_sweeps=n_sweeps, bond_dims=bond_dims,
        noises=noises, thrds=thrds, lowmem_noise=True, twosite_to_onesite=None, tol=1E-12, cutoff=1E-24, iprint=2,
        dav_max_iter=400, dav_def_max_size=20)
    print('DMRG energy = %20.15f' % energy)

    pmpo, mpo = None, None

    ket = driver.adjust_mps(ket, dot=1)[0]
    driver.align_mps_center(ket, ref=0)

    pyket = MPSTools.from_block2(ket)
    pyuket = MPSTools.trans_sz_to_su2(pyket, driver.basis, ket.info.target, target_twos=0)

    driver.symm_type = SymmetryTypes.SU2
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    impo = driver.get_identity_mpo()
    hmpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=2, simple_const=True, add_ident=True, fast_no_orb_dep_op=True)

    ket2 = MPSTools.to_block2(pyuket, driver.basis, tag='KETX-0')
    ket2.info.save_data(driver.scratch + "/%s-mps_info.bin" % ket2.info.tag)
    ket2.load_tensor(ket2.center)
    ket2.tensors[ket2.center].normalize()
    ket2.save_tensor(ket2.center)
    ket2.unload_tensor(ket2.center)
    norm = driver.expectation(ket2, impo, ket2)
    print('Norm = ', norm)

    ket2.info.load_mutable()
    print('UMPS MAX BOND = ', ket2.info.get_max_bond_dimension())

    energy = driver.expectation(ket2, hmpo, ket2, iprint=2)
    print('STATE %d Expt energy = %20.15f' % (istate, energy))

    fe_idxs = [[2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [24, 25, 26, 27, 28], [29, 30, 31, 32, 33]]

    dm = driver.get_npdm(ket2, pdm_type=2, npdm_expr='((C+D)2+(C+D)2)0', mask=(0, 0, 1, 1), iprint=2, max_bond_dim=3000)
    dm = dm * (0.5 * -np.sqrt(3) / 2)
    fe_idxs = np.array([x for xx in fe_idxs for x in xx], dtype=int)
    dm = np.einsum('ijkl->ik', dm[fe_idxs, :][:, fe_idxs].reshape((4, 5, 4, 5)))

    import matplotlib.pyplot as plt
    plt.matshow(dm, cmap='ocean_r')
    plt.gcf().set_dpi(300)
    plt.savefig("%02d-bip-spin-corr.png" % istate, dpi=300)


Compressed MPO
--------------

In the second example, we use a compressed Hamiltonian MPO, which can potentially save
some computational cost. Note that to ensure that the Hamiltonian exactly preserves the
total spin symmetry, the SVD compression needs to be done in the SU2 symmetry mode.
After compression, the Hamiltonian MPO is transformed to lower symmetries. ::

    import numpy as np, sys
    import itertools
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
    from pyblock2.algebra.io import MPSTools

    istate = int(sys.argv[1])

    driver = DMRGDriver(scratch="/tmp", symm_type=SymmetryTypes.SAnySU2, stack_mem=120 << 30, fp_codec_cutoff=0.0, n_threads=64)

    bond_dims = [50] * 8 + [100] * 8
    noises = [1E-5] * (len(bond_dims) - 4) + [0] * 4
    thrds = [1E-7] * len(bond_dims)
    n_sweeps = len(bond_dims)

    driver.read_fcidump(filename='Fe4S4.FCIDUMP', pg='d2h')
    driver.spin = 0
    twos = 0

    npts = driver.get_spin_projection_npts(n_sites=driver.n_sites, n_elec=driver.n_elec, twos=twos)
    print("NPTS = %d" % npts)

    driver.set_symmetry_groups("U1Fermi", "SU2", "SU2", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    umpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=2, simple_const=True, add_ident=False,
        algo_type=MPOAlgorithmTypes.FastBlockedSVD, cutoff=1E-7, integral_cutoff=1E-12, fast_no_orb_dep_op=True)
    print("UMPO = ", umpo.get_bond_dims())

    driver.set_symmetry_groups("U1Fermi", "U1", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    zmpo = driver.mpo_change_symm(umpo, add_ident=False)
    umpo = None
    print("ZMPO = ", zmpo.get_bond_dims())

    driver.set_symmetry_groups("U1Fermi", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    mpo = driver.mpo_change_symm(zmpo, add_ident=True)
    zmpo = None
    print("MPO = ", mpo.get_bond_dims())
    pmpo = driver.get_spin_projection_mpo(twos=twos, twosz=driver.spin, npts=npts, use_sz_symm=False, cutoff=1E-12, add_ident=True, iprint=1)

    driver.set_symmetry_groups("U1Fermi", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    target = driver.target

    driver.symm_type = SymmetryTypes.SAnySZ
    driver.set_symmetry_groups("U1Fermi", "U1", "AbelianPG")
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    n_sites = driver.n_sites

    xdstr = [
        '22aaaaa2aaaa222222222222b2bbbbbbbb22',
        '22aaaaabbb2b222222222222a2aaabbbbb22',
        '222aaaab2bbb222222222222bbbbbaaaaa22',
    ][istate]

    print(istate, xdstr)

    ket = driver.get_mps_from_csf_coefficients([xdstr], dvals=[1.0], tag='KET', dot=1)
    driver.align_mps_center(ket, ref=0)
    ket = driver.adjust_mps(ket, dot=2)[0]
    pket = driver.mps_change_symm(ket, 'PKET-0', target)

    energy = driver.dmrg(mpo, pket, stacked_mpo=pmpo, metric_mpo=pmpo, context_ket=ket, n_sweeps=n_sweeps, bond_dims=bond_dims,
        noises=noises, thrds=thrds, lowmem_noise=True, twosite_to_onesite=None, tol=1E-12, cutoff=1E-24, iprint=2,
        dav_max_iter=400, dav_def_max_size=20)
    print('DMRG energy = %20.15f' % energy)

    pmpo, mpo = None, None

    ket = driver.adjust_mps(ket, dot=1)[0]
    driver.align_mps_center(ket, ref=0)

    pyket = MPSTools.from_block2(ket)
    pyuket = MPSTools.trans_sz_to_su2(pyket, driver.basis, ket.info.target, target_twos=0)

    driver.symm_type = SymmetryTypes.SU2
    driver.initialize_system(n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, orb_sym=driver.orb_sym)

    impo = driver.get_identity_mpo()
    hmpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=2, simple_const=True, add_ident=True, fast_no_orb_dep_op=True)

    ket2 = MPSTools.to_block2(pyuket, driver.basis, tag='KETX-0')
    ket2.info.save_data(driver.scratch + "/%s-mps_info.bin" % ket2.info.tag)
    ket2.load_tensor(ket2.center)
    ket2.tensors[ket2.center].normalize()
    ket2.save_tensor(ket2.center)
    ket2.unload_tensor(ket2.center)
    norm = driver.expectation(ket2, impo, ket2)
    print('Norm = ', norm)

    ket2.info.load_mutable()
    print('UMPS MAX BOND = ', ket2.info.get_max_bond_dimension())

    energy = driver.expectation(ket2, hmpo, ket2, iprint=2)
    print('STATE %d Expt energy = %20.15f' % (istate, energy))

    fe_idxs = [[2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [24, 25, 26, 27, 28], [29, 30, 31, 32, 33]]

    dm = driver.get_npdm(ket2, pdm_type=2, npdm_expr='((C+D)2+(C+D)2)0', mask=(0, 0, 1, 1), iprint=2, max_bond_dim=3000)
    dm = dm * (0.5 * -np.sqrt(3) / 2)
    fe_idxs = np.array([x for xx in fe_idxs for x in xx], dtype=int)
    dm = np.einsum('ijkl->ik', dm[fe_idxs, :][:, fe_idxs].reshape((4, 5, 4, 5)))

    import matplotlib.pyplot as plt
    plt.matshow(dm, cmap='ocean_r')
    plt.gcf().set_dpi(300)
    plt.savefig("%02d-svd-spin-corr.png" % istate, dpi=300)

