
.. highlight:: bash

Keywords
========

In this section we provide a complete list of allowed keywords for the input file used
in ``block2main`` with a short description for each keyword.

Global Settings
---------------

\# / \!
    If a line starts with '!' or '#', the line will be ignored.

outputlevel
    Optional. Followed by one integer.
    0 = Silent.
    1 = Print information for each sweep.
    2 = Print information for iteration at each site (default).
    3 = Print information for each Davidson/CG iteration.

orbitals
    Required for most normal cases.
    Not required if reloading MPO or when ``orbital_rotation`` is the calculation type, or when ``model`` is given.
    Followed by the file name for the orbital definition and integrals, in FCIDUMP format or ``hdf5`` format (used only in ``libdmet``).
    Only ``nonspinadapted`` is supported for ``orbitals`` with ``hdf5`` format .

integral\_tol
    Optional. The integral values smaller than ``integral_tol`` will be discarded.
    Default is 1E-12 (for integral with ``hdf5`` format) or 0 (for integral with FCIDUMP format).

model
    Optional. Can be used to perform calculations for some simple model Hamiltonian and the ``orbitals`` keyword can be skipped. For example,
    ``model hubbard 16 1 2`` will calculate ground state for 1-dimensional non-periodic Hubbard model with 16 sites and nearest-neighbor interaction, t = 1 and U = 2.
    ``model hubbard_periodic 16 1 2`` will do the calculation for the periodic Hubbard model.
    ``model hubbard_kspace 16 1 2`` will do the calculation for the periodic Hubbard model in the momentum space. One can then use this together with ``k_symmetry`` to utilize the translational symmetry or not use it if the keyword ``k_symmetry`` is not given.
    ``model hubbard 16 1 2 per-site`` will print the energy for each site.

prefix
    Optional. Path to scratch folder. Default is ``./nodex/``.

num\_thrds
    Optional. Followed by an integer for the number of OpenMP threads to use.
    Default is 28 (if there is no ``hf_occ integral`` in the input file) or 1
    (to be compatible with ``StackBlock`` when there is ``hf_occ integral`` in the input file).
    Note that the environment variable ``OMP_NUM_THREADS`` is ignored.

mkl\_thrds
    Optional. Followed by an integer for the number of OpenMP threads to use for the MKL library. Default is 1.

mem
    Optional. Followed by an integer and a letter as the unit (g or G). Stack memory for doubles.
    Default is 2 GB. Note that the code may use a large amount of memory via dynamic allocation, which is not controlled by this number.

intmem
    Optional. Followed by an integer and a letter as the unit (g or G). Stack memory for integers.
    Default is 10% of ``mem``.

mem_ratio
    Optional. Followed by a float number (0.0 ~ 1.0). The ratio of main stack memory. Default is 0.4.

min\_mpo\_mem
    Optional. Followed by auto, True, or False. If True, MPO building and simplification will cost much less memory.
    But the computational cost will be higher due to IO cost. Default is auto, which is True if number of orbitals is >= 120.

qc\_mpo\_type
    Optional. Followed by auto (default), conventional, nc, or cn. The Hamiltonian MPO formalism type.
    The default is to use Conventional for non-big-site, and NC for big-site.
    Conventional DMRG is overall 50% faster than NC, but the cost of the middle site is 2 times higher than NC.
    If the memory is limited and ``min_mpo_mem`` is used, one should set NC MPO type to make memory cost more uniform.

cached\_contraction
    Optional. Followed by an integer 0 or 1 (default). If 1, cached contraction is used for improving performance.

nonspinadapted
    Optional. If given, the code will work in the non-spin-adapted ``SZ`` mode. Otherwise, it will work in the spin-adapted ``SU2`` mode.

k\_symmetry
    Optional. If given, the code will work in the non-spin-adapted or spin-adapted mode with additionally the K symmetry.
    Requiring the code to be built with ``-DUSE_KSYMM``.

use\_complex
    Optional. If given, the code will work in the complex number mode, where the integral, MPO and MPS contain all complex numbers.
    FCIDUMP with real or complex integral can be accepted in this mode.
    Requiring the code to be built with ``-DUSE_COMPLEX``. Conflict with ``use_hybrid_complex`` (checked).

use\_hybrid\_complex
    Optional. If given, the code will work in the hybrid complex number mode, where the MPO is split into real and complex sub-MPOs.
    MPS rotation matrix are real matrices but center site tensor is complex.
    FCIDUMP with real or complex integral can be accepted in this mode.
    Requiring the code to be built with ``-DUSE_COMPLEX``. Conflict with ``use_complex`` (checked).

use\_general\_spin
    Optional. If given, the code will work in (fermionic) spin orbital (rather than spatial orbital).
    FCIDUMP will be intepreted as integrals between spin orbitals.
    If the FCIDUMP is actually the normal FCIDUMP for spatial orbitals, the extra keyword ``trans_integral_to_spin_orbital``
    is required to make it work with general spin.
    Requiring the code to be built with ``-DUSE_SG``. Currently cannot be used together with ``k_symmetry``.

single\_prec
    Optional. If given, the code will work in single precision (float) rather than double precision (double).

integral\_rescale
    Optional. ``auto`` (default) or ``none`` or floating point number. If ``auto`` and the calculation is done
    with single precision, the average diagonal of the one-electron integral will be moved to the energy constant.
    Ideally, with single precision, we want the energy constant to be close to the final dmrg energy.
    If ``auto`` and the calculation is done with double precision, nothing will happen.
    If ``none``, nothing will happen.
    If the value of ``integral_rescale`` is a number, the energy constant will be adjust to the given number by shifting
    the average diagonal of the one-electron integral. This should only be used when the particle number of the calculation
    is a constant (namely, ``nelec`` contains only one number).

check\_dav\_tol
    Optional. ``auto`` (default) or 1 or 0. If ``auto`` or ``1`` and the calculation is done with single precision,
    the davidson tolerance will be set to be no lower than 5E-6.

trans\_integral\_to\_spin\_orbital
    Optional. If given, the FCIDUMP (in spatial orbitals) will be reinterpretted to work with general spin.
    Only makes sense together with ``use_general_spin``.

singlet\_embedding
    Optional. If given, the code will use the singlet embedding formalism.
    Only have effects in the spin-adapted ``SU2`` mode. No effects if it is a restart calculation.

conn\_centers
    Optional. Followed by a list of indices of connection sites or by ``auto`` and the number of processor groups. If ``conn_centers`` is given, the parallelism over sites will be used (MPI required, ``twodot`` only). For example, ``conn_centers auto 5`` will divide the processors into 5 groups.
    Only supports the standard DMRG calculation.

restart\_dir
    Optional. Followed by directory name. If ``restart_dir`` is given, after each sweep, the MPS will be backed up in the given directory.

restart\_dir\_per\_sweep
    Optional. Followed by directory name. If ``restart_dir_per_sweep`` is given, after each sweep, the MPS will be backed up in the given directory name followed by the sweep index as the name suffix. This will save MPSs generated from all sweeps.

fp\_cps\_cutoff
    Optional. Followed by a small fractional number. Sets the float-point number cutoff for saving disk storage. Default is ``1E-16``.

release\_integral
    Optional. If given, memory used by stroring the full integral will be release after building MPO (but before DMRG).

Calculation Types
-----------------

The default calculation type is DMRG (without the need to write any keywords).

fullrestart
    Optional. If given, the initial MPS will be read from disk.
    Normally this keyword will be automatically added if any of the ``restart_*`` keywords are used.

oh / restart\_oh
    Expectation value calculation on the DMRG optimized MPS or reloaded MPS.

onepdm / restart\_onepdm
    One-particle density matrix calculation on the DMRG optimized MPS or reloaded MPS.
    ``onepdm`` can run with either ``twodot_to_onedot``, ``onedot`` or ``twodot``.

twopdm / restart\_twopdm
    Two-particle density matrix calculation on the DMRG optimized MPS or reloaded MPS.

threepdm / restart\_threepdm
    Three-particle density matrix calculation on the DMRG optimized MPS or reloaded MPS.
    Cannot be used together with ``conventional_npdm``.

fourpdm / restart\_fourpdm
    Four-particle density matrix calculation on the DMRG optimized MPS or reloaded MPS.
    Cannot be used together with ``conventional_npdm``.

tran\_onepdm / restart\_tran\_onepdm
    One-particle transition density matrix among a set of MPSs.

tran\_twopdm / restart\_tran\_twopdm
    Two-particle transition density matrix among a set of MPSs.

tran\_threepdm / restart\_tran\_threepdm
    Three-particle transition density matrix among a set of MPSs.
    Cannot be used together with ``conventional_npdm``.

tran\_fourpdm / restart\_tran\_fourpdm
    Four-particle transition density matrix among a set of MPSs.
    Cannot be used together with ``conventional_npdm``.

tran\_oh / restart\_tran\_oh
    Operator overlap between each pair in a set of MPSs.

diag\_twopdm / restart\_diag\_twopdm
    Diagonal two-particle density matrix calculation.

correlation / restart\_correlation
    Spin and charge correlation function.

copy\_mps / restart\_copy\_mps
    Copy MPS with one tag to another tag. Followed by the tag name for the output MPS.
    The input MPS tag is given by ``mps_tags``.
    The MPS transformation is also handled with this calculation type.

sample / restart\_sample
    Printing configuration state function (CSF) or determinant coefficients.

orbital\_rotation
    Orbital rotation of an MPS to generate another MPS.

compression
    MPS compression.

delta\_t
    Followed by a single float value or complex value as the time step for the time evolution.
    The computation will apply :math:`\exp (-\Delta t H) |\psi\rangle` (with multiple steps).
    So when it is a real float value, we will do imaginary time evolution of the MPS
    (namely, optimizing to ground state or finite-temperature state).
    When it is a pure imaginary value, we will do real time evolution of the MPS 
    (namely, solving the time dependent Schrodinger equation).
    General complex value can also be supported, but may not be useful.

stopt\_dmrg
    First step of stochastic perturbative DMRG, which is the normal DMRG with a small bond dimension.

stopt\_compression
    Second step of stochastic perturbative DMRG, which is the compression of :math:`QV |\Psi_0\rangle`.
    In general a bond diemension that is much larger than the first step should be used.

stopt\_sampling
    Third step of stochastic perturbative DMRG. Followed by an integer as the number of CSF / determinants to be sampled.
    If any of the first and second step is done in the non-spin-adapted mode, the determinants will be sampled and this step must also be in the non-spin-adapted mode. Otherwise, CSF will be sampled if the keyword ``nonspinadapted`` is given, and determinants will be sampled if the keyword ``nonspinadapted`` is not given.

restart\_nevpt2\_npdm
    Compute 1-4 PDM for DMRG-SC-NEVPT2. If there are multiple roots, the calculation will be performed for all roots.
    The 1-4PDM will be used to compute the SC-NEVPT2 intermediate Eqs. (A16) and (A22) in the spin-free NEVPT2 paper.
    Only the two SC-NEVPT2 intermediates will be written into the disk.

restart\_mps\_nevpt
    Followed by three integers, representing the number of active, inactive, and external orbitals.
    Compute the ``V_i`` and ``V_a`` correlation energy in DMRG-SC-NEVPT2 using MPS compression.
    Only the spin-adapted version is implemented. If there are multiple roots, the keyword ``nevpt_state_num`` is
    required to set which root should be used to compute the correlation energy.

Calculation Modifiers
---------------------

target\_t
    Optional. Followed by a single float value as the total time for time evolution.
    This keyword should be used only together with ``delta_t``. Default is 1.

te\_type
    Optional. Followed by ``rk4`` or ``tangent_space``. This keyword sets the time evolution algorithm.
    This keyword should be used only together with ``delta_t``. Default is ``rk4``.

statespecific
    If ``statespecific`` keyword is in the input (with no associated value).
    This option implies that a previous state-averaged dmrg calculation has already been performed.
    This calculation will refine each individual state. This keyword should be used only with DMRG calculation type.

soc
    If ``soc`` keyword is in the input (with no associated value), the (normal or transition) one pdm for triplet excitation operators will be calculated (which can be used for spin-orbit coupling calculation). This keyword should be used only together with ``onepdm``, ``tran_onepdm``, ``restart_onepdm``, or ``restart_tran_onepdm``. Not supported for ``nonspinadapted``.

overlap
    If ``overlap`` keyword is in the input (with no associated value), the expectation of identity operator will be calculated (which can be used for the overlap matrix between states).
    Otherwise, when the `overlap` keyword is not given, the full Hamiltonian is used.
    For compression, if this keyword is in the input, it directly compresses the given MPS. Otherwise, the contration of full Hamiltonian MPO and MPS is compressed.
    This keyword should only be used together with ``oh``, ``tran_oh``, ``restart_oh``, ``restart_tran_oh``, ``compression``, and ``stopt_compression``.

nat\_orbs
    If given, the natural orbitals will be computed.
    Optionally followed by the filename for storing the rotated integrals (FCIDUMP).
    If no value is associated with the keyword ``nat_orbs``, the rotated integrals will not be computed.
    This keyword can only be used together with ``restart_onepdm`` or ``onepdm``.

nat\_km\_reorder
    Optional keyword with no associated value. If given, the artificial reordering in the natural orbitals will be removed using Kuhn-Munkres algorithm. This keyword can only be used together with ``restart_onepdm`` or ``onepdm``.
    And the keyword ``nat_orbs`` must also exist.

nat\_positive_def
    Optional keyword with no associated value. If given, artificial rotation in the logarithm of the rotation matrix can be avoid, by make the rotation matrix quasi-positive-definite, with "quasi" in the sense that the rotation matrix is not Hermitian. This keyword can only be used together with ``restart_onepdm`` or ``onepdm``.
    And the keyword ``nat_orbs`` must also exist.

trans\_mps\_to\_sz
    Optional keyword with no associated value. If given, the MPS will be transformed to non-spin-adapted before being saved. This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``.

trans\_mps\_to\_singlet\_embedding
    Optional keyword with no associated value. If given, the MPS will be transformed to singlet-embedding format before being saved. This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``.

trans\_mps\_from\_singlet\_embedding
    Optional keyword with no associated value. If given, the MPS will be transformed to non-singlet-embedding format before being saved. This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``.

trans\_mps\_to\_complex
    Optional keyword with no associated value. If given, the MPS will be transformed to complex wavefunction with real rotation matrix before being saved.
    This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``, and optionally with ``split_states``.
    This keyword is conflict with other ``trans\_mps\_*`` keywords.
    To load this MPS in the subsequent calculations, the keyword ``complex_mps`` must be used.

split\_states
    Optional keyword with no associated value. If given, the state averaged MPS will be split into individual MPSs.
    This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``, and optionally with ``trans_mps_to_complex``.
    This keyword is conflict with other ``trans\_mps\_*`` keywords.
    The individual MPS will be the tag given by the keyword ``restart_copy_mps`` or ``copy_mps`` with ``-<n>`` appended,
    where ``n`` is the root index counting from zero.

resolve\_twosz
    Optional. Followed by an integer, which is two times the projected spin.
    The transformed SZ MPS will have the specified projected spin.
    If the keyword ``resolve_twosz`` is not given, an MPS with ensemble of all possible projected spins will be produced (which is often not very useful).
    This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``.
    And the keyword ``trans_mps_to_sz`` must also exist.

normalize\_mps
    Optional keyword with no associated value. If given, the transformed SZ MPS will be normalized.
    This keyword can only be used together with ``restart_copy_mps`` or ``copy_mps``.
    And the keyword ``trans_mps_to_sz`` must also exist.

big\_site
    Optional. Followed by a string for the implementation of the big site.
    Possible implementations are ``folding``, ``fock`` (only with ``nonspinadapted``),
    ``csf`` (only without ``nonspinadapted``).
    This keyword can only be used in dynamic correlation calculations.
    If this keyword is not given, the dynamic correlation calculation will be performed with normal MPS
    with no big sites.

expt\_algo\_type
    Optional. Followed by a string ``auto``, ``fast``, ``normal``, ``symbolfree``, or ``lowmem``. Default is ``auto``.
    This keyword can only be used with density matrix or transition density matrix calculations.
    ``auto`` is ``fast`` if ``conventional_npdm`` is given, or ``symbolfree`` if ``conventional_npdm`` is not given.
    ``normal`` uses less memory compared to ``fast``, but the complexity can be higher.
    ``lowmem`` uses less memory compared to ``symbolfree``, but the complexity can be higher.
    ``symbolfree`` is in general more efficient than ``fast`` and ``normal``,
    but it is only available if ``conventional_npdm`` is not given.
    For 3- and 4-particle density matrices, when this keyword is not ``auto`` or ``symbolfree``,
    it may consume a significant large amount of memory to store the symbols.

conventional\_npdm
    Optional, mainly for backward compatibility. If given, will use the conventional manual npdm code.
    This is only available for 1- and 2- particle density matrices.
    For most cases, the conventional manual code is slower.
    For soc 1-particle density matrix, only the conventional manual code is available.

simple\_parallel
    Optional. Followed by an empty string (same as ``ij``) or ``ij`` or ``kl``. When this keyword is not given,
    the conventional parallel rule for QC-DMRG will be used. Otherwise, the simple parallel scheme based on
    distributing integral according to ``ij`` or ``kl`` indices is used.
    When ``qc_mpo_type`` is auto, this simple scheme will also change the center for middle transformation
    to reduce the MPO bond dimension. The simple parallel scheme may be good for saving per-processor MPO memory cost
    for large scale parallelized DMRG.

condense\_mpo
    Optional. Followed by an integer (must be a power of 2, default is 1).
    When ``condense_mpo`` is not 1, ``block2`` will merge every two adjacent MPO sites into a larger site (after the MPO is created),
    repeating ``log(condense_mpo)`` times, until the total number of sites is ``n_sites / condense_mpo``.
    Not working with SU2 symmetry. When ``condense_mpo 2`` is used with general spin, the calculation will be done with
    two spin orbitals as a site rather than one spin orbital. Not working with ``twopdm`` related keywords.
    Require the keyword ``simple_parallel`` for the parallelization of the condensed MPO.

one\_body\_parallel\_rule
    Optional keyword with no associated value. If given, the more efficient parallelization rule will be
    used to distribute the MPO. This rule only works when the two-body term is zero or purely local.
    Real space Huabbard model is one of the case. For such Hamiltonian, the default (quantum chemistry)
    parallelization rule can still work, but may have no improvements with multiple processors.
    If this keyword is used with non-trivial two-body term, runtime error may happen.

complex\_mps
    Optional keyword with no associated value. If given, complex expectation values will be computed
    for MPS with complex wavefunction tensor and real rotation matrices (in non-complex mode).
    Should be used together with ``pdm``, ``oh``, or (complex) ``delta_t`` type calculations.
    In complex mode, this should not be used as everything is complex.

tran\_bra\_range
    Optional. Followed by the range parameter of bra state indices for computing transition density matrices.
    Normally two numbers are given, which is the starting index and endding index (not included).

tran\_ket\_range
    Optional. Followed by the range parameter of ket state indices for computing transition density matrices.
    Normally two numbers are given, which is the starting index and endding index (not included).

tran\_triangular
    Optional keyword with no associated value. If given, only the transition density matrices with bra state
    index equal to or greater than the ket state index will be computed.

skip\_inact\_ext\_sites
    Optional keyword with no associated value. If given, for uncontracted dynamic correlation calculations,
    the sweeps will skip inactive and external sites,
    so that the efficiency can be higher and the accuracy is not affected. This should only be used with
    uncontracted dynamic correlation keywords (checked) without any big sites. Normally it is useful only for
    dynamic correlation with singles (such as ``mrcis``).

full\_integral
    Optional keyword with no associated value. If **not** given, and it is a dynamic correlation with singles
    (namely, with keywords ``nevpt2s``, ``mrcis``, ``mrrept2s``, ``nevpt2-i``, ``nevpt2-r``, ``mrrept2-i``, or
    ``mrrept2-r``), the two-electron integral elements with more than two virtual indices will be set to zero.
    This should save some MPO contruction time, without affecting the sweep time cost and accuracy.
    If this keyword is given, the full integral elements will be used for constructing MPO.

nevpt\_state\_num
    Followed by a single integer, the index of the root (counting from zero) used for SC-NEVPT2.
    Only useful for the calculation type ``restart_mps_nevpt``.

Uncontracted Dynamic Correlation
--------------------------------

There can only be at most one dynamic correlation keyword (checked).
Any of the following keyword must be followed by 2 integers
(representing number of orbitals in the active space and number of electrons in the active space),
or 3 integers (representing number of orbitals in the inactive, active, and external space, respectively).

dmrgfci
    Not useful for general purpose. Treating the inactive and external space using full Configuration Interaction (FCI).

casci
    Treating the inactive space as a single CSF (all occupied) and the external space as a single CSF (all empty).

mrci
    *Same as* ``mrcisd``.

mrcis
    Multi-configuration CI with singles. The inactive / virtual space can have at most one hole / electron.

mrcisd
    Multi-configuration CI with singles and doubles. The inactive / virtual space can have at most two holes / electrons.

mrcisdt
    Multi-configuration CI with singles and doubles and triples. The inactive / virtual space can have at most three holes / electrons.

nevpt2
    *Same as* ``nevpt2sd``.

nevpt2s
    Second order N-Electron Valence States for Multireference Perturbation Theory with singles.
    The inactive / virtual space can have at most one hole / electron.

nevpt2sd
    Second order N-Electron Valence States for Multireference Perturbation Theory with singles and doubles.
    The inactive / virtual space can have at most two holes / electrons.
    The zeroth-order Hamiltonian is Dyall's Hamiltonian.

mrrept2
    *Same as* ``mrrept2sd``.

mrrept2s
    Second order Restraining the Excitation degree Multireference Perturbation Theory (MRREPT) with singles.
    The inactive / virtual space can have at most one hole / electron.

mrrept2sd
    Second order Restraining the Excitation degree Multireference Perturbation Theory (MRREPT) with singles and doubles.
    The inactive / virtual space can have at most two holes / electrons.
    The zeroth-order Hamiltonian is Fink's Hamiltonian.

Schedule
--------

onedot
    Using the one-site DMRG algorithm.
    ``onedot`` will be implicitly used if you restart from a ``onedot`` mps (can be obtained from previous run with ``twodot_to_onedot``).

twodot
    Default. Using the two-site DMRG algorithm.

twodot\_to\_onedot
    Followed by a single number to indicate the sweep iteration when to switch from the two-site DMRG algorithm to the one-site DMRG algorithm. The sweep iteration is counted from zero.

schedule
    Optional. Followed by the word ``default`` or a multi-line DMRG schedule with the last line being ``end``.
    If not given, the defualt schedule will be used.
    Between the keyword ``schedule`` and ``end`` each line needs to have four values. They are corresponding
    to starting sweep iteration (counting from zero), MPS bond dimension, tolerance for the Davidson iteration,
    and noise, respectively. Starting sweep iteration is the sweep iteration in which the given parameters
    in the line should take effect.
    For each line, alternatively, one can provide ``n_sites - 1`` values for the MPS bond dimension,
    where the ith number represents the right virtual bond dimension for the MPS tensor at site i.
    If this is the case, the site-dependent MPS bond dimension truncation will be used.

store_wfn_spectra
    Optional with no associated value. If given, the singular values at each left-right partition during the last DMRG sweep
    will be stored as ``sweep_wfn_spectra.npy`` after convergence. Only works with DMRG type calculation.
    The stored array is a numpy array of 1 dimensional numpy array.
    The inner arrays normally do not have all the same length.
    For spin-adapted, each singular values correspond to a multiplet.
    So for non-singlet, the wavefunction spectra have different interpretation between SU2 and SZ.
    Additionally, when this keyword is given, the bipartite entanglement of the MPS will be computed, as
    :math:`S_k = - \sum_i \Lambda_i^2 \log \Lambda_i^2` where :math:`\Lambda_i` are all singular values found at site k.
    The bipartite entanglement will be printed and stored as ``sweep_wfn_entropy.npy`` as a 1 dimensional numpy array.

extrapolation
    Optional. Should only be used for standard DMRG calculation with the reverse schedule.
    Will print the extrapolated energy and generate the energy extrapolation plot (saved as a figure).

maxiter
    Optional. Followed by an integer. Maximum number of sweep iterations. Default is 1.

sweep\_tol
    Optional. Followed by a small float number. Convergence for the sweep. Default is 1E-6.

startM
    Optional. Followed by an integer. Starting bond dimension in the default schedule.
    Default is 250.

maxM
    Required for default schedule. Followed by an integer.
    Maximum bond dimension in the default schedule.

lowmem\_noise
    Optional. If given, the noise step will require less memory but potentially worse openmp load-balancing.

dm\_noise
    Optional. If given, the density matrix noise will be used instead of the default perturbative noise.
    Density matrix noise is much cheaper but not very effective.

cutoff
    Optional. Followed by a small float number. States with eigenvalue below this number will be discarded,
    even when the bond dimension is large enough to keep this state. Default is 1E-14.

svd\_cutoff
    Optional. Followed by a small float number. Cutoff of singular values used in parallel over sites.
    Default is 1E-12.

svd\_eps
    Optional. Followed by a small float number. Accuracy of SVD for connection sites used in parallel over sites.
    Default is 1E-4.

trunc\_type
    Optional. Can be ``physical`` (default) or ``reduced``, where ``reduced`` re-weight eigenvalues by their multiplicities (only useful in the ``SU2`` mode).

decomp\_type
    Optional. Can be ``density_matrix`` (default) or ``svd``, where `svd` may be less numerical stable and not working with ``nroots > 1``.

real\_density\_matrix
    Optional. Only have effects in the complex mode and when ``decomp_type`` is ``density_matrix``.
    If given, the imaginary part of the density matrix will be discarded before diagonalization.
    This means that all rotation matrices will be orthogonal rather than unitary, although they will be stored as complex matrices.
    For complex mode DMRG with more than one roots, this keyword has to be used (not checked).

davidson\_max\_iter
    Optional. Maximal number of iterations in Davidson. Default is 5000.
    If this number is reached but convergence is not achieved, the calculation will abort.

davidson\_soft\_max\_iter
    Optional. Maximal number of iterations in Davidson. Default is -1.
    If this number is reached but convergence is not achieved, the calculation will continue as if the convergence is achieved.
    If this numebr is -1, or larger than or equal to ``davidson_max_iter``,
    this keyword has no effect and ``davidson_max_iter`` is used instead.

n\_sub\_sweeps
    Optional. Number of sweeps for each time step. Defualt is 2.
    This keyword only has effect when used with ``delta_t`` and when ``te_type`` is ``rk4``.

System Definition
-----------------

nelec
    Optional. Followed by one or more integrers. Number of electrons in the target wavefunction.
    If not given, the value from FCIDUMP is used (and the keyword ``orbtials`` must be given).

spin
    Optional. Followed by one or more integrers.
    Two times the total spin of the target wavefunction in spin-adapted calculation.
    Or Two times the projected spin (number of alpha electrons minus number of beta electrons) of the target wavefunction in non-spin-adapted calculation.
    If not given, the value from FCIDUMP is used. If FCIDUMP is not given, 0 is used.

irrep
    Optional. Followed by one or more integrers.
    Point group irreducible representation of the target wavefunction.
    If not given, the value from FCIDUMP is used. If FCIDUMP is not given, 1 is used.
    MOLPRO notation is used, where 1 always means the trivial irreducible representation.

sym
    Optional. Followed by a lowercase string for the (Abelian) point group name. Default is ``d2h``.
    If the real point group is ``c1`` or ``c2``, setting ``sym d2h`` will also work.

k\_irrep
    Optional. Followed by one or more integrers.
    LZ / K irreducible representation number of the target wavefunction.
    If not given, the value from FCIDUMP is used. If FCIDUMP is not given, 0 is used.

k\_mod
    Optional. Followed by one integer.
    Modulus for the K symmetry. Zero means LZ symmetry.
    If not given, the value from FCIDUMP is used. If FCIDUMP is not given, 0 is used.

nroots
    Optional. Followed by one integer. Number of roots. Default is 1.
    For ``nroots > 1``, ``oh`` or ``restart_oh`` will calculate the expectation of Hamiltonian on every state. ``tran_oh`` or ``restart_tran_oh`` will calculate the expectation of Hamiltonian on every possible pair of states as bra and ket states.
    The parameters for the quantum number of the MPS, namely ``spin``, ``isym`` and ``nelec`` can also take multiple numbers. This can also be combined with ``nroots > 1``, which will then enable transition density matrix between MPS with different quantum numbers to be calculated (in a single run). This kind of calulation usually needs a larger ``nroots`` than the ``nroots`` actually needed, otherwise, some excited states with different quantum number from the ground-state may be missing. To save time, one may first do a calculation with larger ``nroots`` and small bond dimensions, and then do ``fullrestart`` and change ``nroots`` to a smaller value. Then only the lowest ``nroots`` MPSs will be restarted.

weights
    Optional. Followed by a list of fractional numbers. The weights of each state for the state average calculation.
    If not given, equal weight will be used for all states.

mps\_tags
    Optional. Followed by a single string or a list of strings.
    The MPS in scratch directory with the specific tag/tags will be loaded for restart (for ``statespecific``, ``restart_onepdm``, etc.).
    The default MPS tag for input/output is ``KET``.

read\_mps\_tags
    Optional. Followed by a string. The tag for the constant (right hand side) MPS for compression.
    The tag of the output MPS in compression is set using ``mps_tags``.

proj\_mps\_tags
    Optional. Followed by a single string or a list of strings. The tag for the MPSs to be projected out during DMRG.
    Must be used together with ``proj_weights``. The projection will be done by changing Hamiltonian from :math:`\hat{H}`
    to :math:`\hat{H} + \sum_i w_i |\phi_i\rangle \langle \phi_i|` (the level shift approach),
    where :math:`|\phi_i\rangle` are the MPSs to be projected out. :math:`w_i` are the weights.

proj\_weights
    Optional. Followed by a single float number or a list of float numbers.
    Can be used together with ``proj_mps_tags``. The number of float numbers in this keyword must be equal to the
    length of ``proj_mps_tags``. Normally, the weights are positive and they should be larger than the energy gap.
    If the weight is too small, you will get unphyiscal eigenvalues as ``Ei + wi``, where ``Ei`` is the energy of the
    MPSs to be projected out.
    If ``statespecific`` keyword is in the input, it will change the projection method from the orthogonalization method
    :math:`\hat{H} - \sum_i |\phi_i\rangle \langle \phi_i|` to the level shift approach 
    :math:`\hat{H} + \sum_i w_i |\phi_i\rangle \langle \phi_i|`.

symmetrize\_ints
    Optional. Followed by a small float number.
    Setting the largest allowed value for the integral element that violates the point group or K symmetry.
    Default is 1E-10. The symmetry-breaking integral elements will be discarded in the calculation anyway.
    Setting this keyword will only control whether the calculation can be performed or an error will be generated.

occ
    Optional. Followed by a list of float numbers between 0 and 2 for spatial orbital occupation numbers,
    or a list of float numbers between 0 and 1 for spin orbital occupation numbers,
    or a list of float numbers between 0 and 1 for the probability for each of four states at each site (experimental).
    This keyword should only be used together with ``warmup occ``.

bias
    Optional. Followed by a non-negative float number. If not 1.0, sets an power based bias to occ.

cbias
    Optional. Followed by a non-negative float number. If not 0.0, sets a constant shift towards the equal-possibility occ.
    ``cbias`` is normally useful for shifting integral occ, while ``bias`` only shifts fractional occ.

init_mps_center
    Optional. Followed by a site index (counting from zero). Default is zero.
    This is the canonical center for the initial guess MPS.

full\_fci\_space
    Optional, not useful for general user. If ``full_fci_space`` keyword is in the input (with no associated value), the full fci space is used (including block quantum numbers outside the space of the wavefunction target quantum number).

trans\_mps\_info
    Optional, experimental. If ``trans_mps_info`` keyword is in the input (with no associated value), the ``MPSInfo`` will be initialized using ``SZ`` quantum numbers if in ``SU2`` mode, or using ``SU2`` quantum numbers if in ``SZ`` mode. A transformation of ``MPSInfo`` is then performed between ``SZ`` and ``SU2`` quantum numbers. ``MultiMPSInfo`` cannot be supported with this keyword.

random\_mps\_init
    Optional. If given, the initial guess for the output MPS in compression will be random initialized in the way set by the ``warmup`` keyword. Otherwise, the constant right hand side MPS will be copied as the the initial guess for the output MPS.

warmup
    Optional. If ``wamup occ`` then the initial guess will be generated using occupation numbers.
    Otherwise, the initial guess will be generated assuming every quantum number has the same probability (default).

Orbital Reordering
------------------

There can only be at most one orbital reordering keyword (checked).

noreorder
    The order of orbitals is not changed.

nofiedler
    *Same as* ``noreorder``.

gaopt
    Genetic algorithm for orbital ordering.
    Followed by (optionally) the configuration file for the ``gaopt`` subroutine.
    Default parameters for the genetic algorithm will be used if no configuration file is given.

fiedler
    Default. Fiedler orbital reordering.

irrep\_reorder
    Group orbitals with the same irrep together.

reorder
    Followed by the name of a file including the space-sparated orbital reordering indices (counting from one).

Unused Keywords
---------------

hf\_occ integral
    Optional. For StackBlock compatibility only.
